//! Legacy Expr-based symbolic preparation for BVP systems.
//!
//! This is deliberately separate from [`super::legacy_lambdify`].  The
//! preparation stage differentiates and simplifies `Expr` trees once, while
//! the Lambdify module owns only compiled runtime callbacks.  Keeping those
//! costs in separate modules makes parity and stage telemetry unambiguous.

use crate::symbolic::symbolic_engine::Expr;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::legacy::{BvpMatrixBackend, Jacobian};

pub(crate) fn variables_for_functions(vector_of_functions: &[Expr]) -> Vec<Vec<String>> {
    vector_of_functions
        .iter()
        .map(Expr::all_arguments_are_variables)
        .collect()
}

pub(crate) fn parse_variables(variable_string: &[String]) -> Vec<Expr> {
    Expr::parse_vector_expression(variable_string.iter().map(String::as_str).collect())
}

/// Differentiate each row independently and preserve the symbolic sparse list.
///
/// This is the current ExprLegacy equivalent of the historical `smat parallel`
/// preparation path: Rayon parallelizes rows, while each nonzero derivative is
/// simplified exactly once and retained in the ordered `(row, col, Expr)` list.
pub(crate) fn jacobian_smart(
    vector_of_functions: &[Expr],
    vector_of_variables_len: usize,
    variable_string: &[String],
    variables_for_all_discrete: &[Vec<String>],
) -> (Vec<Vec<Expr>>, Vec<(usize, usize, Expr)>) {
    let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = vector_of_functions
        .par_iter()
        .enumerate()
        .map(|(row, function)| {
            let mut derivatives = Vec::with_capacity(vector_of_variables_len);
            let mut sparse_entries = Vec::new();
            for col in 0..vector_of_variables_len {
                let variable = &variable_string[col];
                if variables_for_all_discrete[row].contains(variable) {
                    let derivative = Expr::diff(function, variable).simplify();
                    if !derivative.is_zero() {
                        sparse_entries.push((row, col, derivative.clone()));
                    }
                    derivatives.push(derivative);
                } else {
                    derivatives.push(Expr::Const(0.0));
                }
            }
            (derivatives, sparse_entries)
        })
        .collect();

    let dense = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
    let sparse = rows_and_sparse
        .into_iter()
        .flat_map(|(_, entries)| entries)
        .collect();
    (dense, sparse)
}

pub(crate) fn bandwidth(symbolic_jacobian: &[Vec<Expr>]) -> (usize, usize) {
    let n = symbolic_jacobian.len();
    (0..n)
        .into_par_iter()
        .map(|row| {
            let mut lower = 0;
            let mut upper = 0;
            for col in 0..n {
                if symbolic_jacobian[row][col] != Expr::Const(0.0) {
                    if col > row {
                        upper = upper.max(col - row);
                    } else if row > col {
                        lower = lower.max(row - col);
                    }
                }
            }
            (lower, upper)
        })
        .reduce(
            || (0, 0),
            |left, right| (left.0.max(right.0), left.1.max(right.1)),
        )
}

/// Builds the ExprLegacy Jacobian cache for the sparse-first and dense
/// compatibility routes.
///
/// This is intentionally kept in the symbolic module rather than beside the
/// runtime callback builders. The returned cache is still the historical
/// `Expr` cache, but its construction is a cold symbolic stage and must not be
/// confused with the Mutex-based callback execution stage.
pub(crate) fn calc_jacobian_parallel_smart_optimized(
    jacobian: &mut Jacobian,
    use_configured_bandwidth: bool,
) {
    assert!(
        !jacobian.variables_for_all_disrete.is_empty(),
        "symbolic Jacobian preparation requires discretized variable usage"
    );
    assert!(
        !jacobian.vector_of_functions.is_empty(),
        "vector_of_functions is empty"
    );
    assert!(
        !jacobian.vector_of_variables.is_empty(),
        "vector_of_variables is empty"
    );

    let variable_string = &jacobian.variable_string;
    let function_count = jacobian.vector_of_functions.len();
    let variable_count = jacobian.vector_of_variables.len();
    let sparse_first_backend = matches!(
        jacobian.backend_config.matrix_backend,
        BvpMatrixBackend::Banded | BvpMatrixBackend::FaerSparseCol
    );
    let bandwidth = use_configured_bandwidth
        .then_some(jacobian.bandwidth)
        .flatten();
    let mut timings = HashMap::new();

    let variable_sets_begin = Instant::now();
    let variable_sets: Vec<HashSet<&String>> = jacobian
        .variables_for_all_disrete
        .iter()
        .map(|vars| vars.iter().collect())
        .collect();
    timings.insert(
        "symbolic jacobian variable sets time".to_string(),
        variable_sets_begin.elapsed().as_secs_f64(),
    );

    if sparse_first_backend {
        let differentiation_begin = Instant::now();
        let sparse_rows: Vec<Vec<(usize, usize, Expr)>> = (0..function_count)
            .into_par_iter()
            .map(|row_index| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    (
                        (row_index + ku + 1).min(variable_count),
                        row_index.saturating_sub(kl + 1),
                    )
                } else {
                    (variable_count, 0)
                };
                let mut entries = Vec::new();
                for col in left_border..right_border {
                    let variable = &variable_string[col];
                    if variable_sets[row_index].contains(variable) {
                        let partial =
                            Expr::diff(&jacobian.vector_of_functions[row_index], variable);
                        if !partial.is_zero() {
                            entries.push((row_index, col, partial));
                        }
                    }
                }
                entries
            })
            .collect();
        timings.insert(
            "symbolic jacobian row differentiation time".to_string(),
            differentiation_begin.elapsed().as_secs_f64(),
        );

        let flatten_begin = Instant::now();
        jacobian.symbolic_jacobian_sparse = sparse_rows.into_iter().flatten().collect();
        timings.insert(
            "symbolic jacobian sparse cache flatten time".to_string(),
            flatten_begin.elapsed().as_secs_f64(),
        );

        let dense_cache_begin = Instant::now();
        jacobian.symbolic_jacobian.clear();
        timings.insert(
            "symbolic jacobian dense cache materialize time".to_string(),
            dense_cache_begin.elapsed().as_secs_f64(),
        );
    } else {
        let differentiation_begin = Instant::now();
        let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = (0..function_count)
            .into_par_iter()
            .map(|row_index| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    (
                        (row_index + ku + 1).min(variable_count),
                        row_index.saturating_sub(kl + 1),
                    )
                } else {
                    (variable_count, 0)
                };
                let mut row = vec![Expr::Const(0.0); variable_count];
                let mut entries = Vec::new();
                for col in left_border..right_border {
                    let variable = &variable_string[col];
                    if variable_sets[row_index].contains(variable) {
                        let partial =
                            Expr::diff(&jacobian.vector_of_functions[row_index], variable);
                        if !partial.is_zero() {
                            entries.push((row_index, col, partial.clone()));
                        }
                        row[col] = partial;
                    }
                }
                (row, entries)
            })
            .collect();
        timings.insert(
            "symbolic jacobian row differentiation time".to_string(),
            differentiation_begin.elapsed().as_secs_f64(),
        );

        let dense_cache_begin = Instant::now();
        jacobian.symbolic_jacobian = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
        timings.insert(
            "symbolic jacobian dense cache materialize time".to_string(),
            dense_cache_begin.elapsed().as_secs_f64(),
        );

        let flatten_begin = Instant::now();
        jacobian.symbolic_jacobian_sparse = rows_and_sparse
            .into_iter()
            .flat_map(|(_, entries)| entries)
            .collect();
        timings.insert(
            "symbolic jacobian sparse cache flatten time".to_string(),
            flatten_begin.elapsed().as_secs_f64(),
        );
    }
    jacobian.set_symbolic_jacobian_timer_snapshot(timings);
}

/// Detects `(lower_bandwidth, upper_bandwidth)` from the prepared Expr cache.
/// Sparse entries are preferred so a Banded/Faer system does not scan a dense
/// compatibility matrix.
pub(crate) fn find_bandwidths(jacobian: &mut Jacobian) {
    if !jacobian.symbolic_jacobian_sparse.is_empty() {
        let (lower, upper) = jacobian
            .symbolic_jacobian_sparse
            .par_iter()
            .map(|(row, col, _)| {
                if col > row {
                    (0, col - row)
                } else {
                    (row - col, 0)
                }
            })
            .reduce(
                || (0, 0),
                |left, right| (left.0.max(right.0), left.1.max(right.1)),
            );
        jacobian.bandwidth = Some((lower, upper));
        return;
    }

    let matrix = &jacobian.symbolic_jacobian;
    let n = matrix.len();
    let (lower, upper) = (0..n)
        .into_par_iter()
        .map(|row| {
            let mut row_lower = 0;
            let mut row_upper = 0;
            for col in 0..n {
                if matrix[row][col] != Expr::Const(0.0) {
                    if col > row {
                        row_upper = row_upper.max(col - row);
                    } else if row > col {
                        row_lower = row_lower.max(row - col);
                    }
                }
            }
            (row_lower, row_upper)
        })
        .reduce(
            || (0, 0),
            |left, right| (left.0.max(right.0), left.1.max(right.1)),
        );
    jacobian.bandwidth = Some((lower, upper));
}

/// Historical smart parallel differentiation entry point, kept separate from
/// the optimized sparse-first builder for callers that explicitly request the
/// original simplified dense-cache semantics.
pub(crate) fn calc_jacobian_parallel_smart(jacobian: &mut Jacobian) {
    assert!(
        !jacobian.variables_for_all_disrete.is_empty(),
        "symbolic Jacobian preparation requires discretized variable usage"
    );
    assert!(
        !jacobian.vector_of_functions.is_empty(),
        "vector_of_functions is empty"
    );
    assert!(
        !jacobian.vector_of_variables.is_empty(),
        "vector_of_variables is empty"
    );

    let variable_names = jacobian.variable_string.clone();
    let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = jacobian
        .vector_of_functions
        .par_iter()
        .enumerate()
        .map(|(row, function)| {
            let mut derivatives = Vec::with_capacity(jacobian.vector_of_variables.len());
            let mut sparse_entries = Vec::new();
            for col in 0..jacobian.vector_of_variables.len() {
                if jacobian.variables_for_all_disrete[row].contains(&variable_names[col]) {
                    let derivative = Expr::diff(function, &variable_names[col]).simplify();
                    if !derivative.is_zero() {
                        sparse_entries.push((row, col, derivative.clone()));
                    }
                    derivatives.push(derivative);
                } else {
                    derivatives.push(Expr::Const(0.0));
                }
            }
            (derivatives, sparse_entries)
        })
        .collect();
    jacobian.symbolic_jacobian = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
    jacobian.symbolic_jacobian_sparse = rows_and_sparse
        .into_iter()
        .flat_map(|(_, entries)| entries)
        .collect();
}

/// Historical full parallel differentiation entry point.
pub(crate) fn calc_jacobian_parallel(jacobian: &mut Jacobian) {
    assert!(
        !jacobian.vector_of_functions.is_empty(),
        "vector_of_functions is empty"
    );
    assert!(
        !jacobian.vector_of_variables.is_empty(),
        "vector_of_variables is empty"
    );

    let variable_names = jacobian.variable_string.clone();
    let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = jacobian
        .vector_of_functions
        .par_iter()
        .enumerate()
        .map(|(row, function)| {
            let mut derivatives = Vec::with_capacity(jacobian.vector_of_variables.len());
            let mut sparse_entries = Vec::new();
            for (col, variable) in variable_names.iter().enumerate() {
                let derivative = Expr::diff(function, variable).simplify();
                if !derivative.is_zero() {
                    sparse_entries.push((row, col, derivative.clone()));
                }
                derivatives.push(derivative);
            }
            (derivatives, sparse_entries)
        })
        .collect();
    jacobian.symbolic_jacobian = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
    jacobian.symbolic_jacobian_sparse = rows_and_sparse
        .into_iter()
        .flat_map(|(_, entries)| entries)
        .collect();
}

#[cfg(test)]
mod tests {
    use super::jacobian_smart;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn expr_legacy_jacobian_keeps_ordered_sparse_entries() {
        let functions = vec![
            Expr::parse_expression("x + 2*y"),
            Expr::parse_expression("x*y"),
        ];
        let variables = vec!["x".to_string(), "y".to_string()];
        let active = vec![variables.clone(), variables.clone()];
        let (dense, sparse) = jacobian_smart(&functions, 2, &variables, &active);

        assert_eq!(dense.len(), 2);
        assert_eq!(sparse.len(), 4);
        assert_eq!((sparse[0].0, sparse[0].1), (0, 0));
        assert_eq!((sparse[1].0, sparse[1].1), (0, 1));
        assert_eq!((sparse[2].0, sparse[2].1), (1, 0));
        assert_eq!((sparse[3].0, sparse[3].1), (1, 1));
    }
}
