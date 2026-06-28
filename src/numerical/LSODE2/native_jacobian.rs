use crate::numerical::BDF::BDF_solver::BdfJacobian;
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, SharedIvpParameterValues,
    SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    SelectedSymbolicIvpBackendKind, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_sparse_backend,
};
use faer::sparse::Triplet;
use nalgebra::DVector;
use std::sync::{Arc, RwLock};

type CompiledEntry = (usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>);

/// Native symbolic Jacobian storage requested by the LSODE2 linear backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeJacobianStorage {
    Dense,
    SparseTriplets,
    Banded { bandwidth: Option<(usize, usize)> },
}

/// Builds a native Jacobian evaluator from symbolic IVP equations.
///
/// The evaluator uses the same argument order as the IVP lambdify layer:
/// `t, y0, y1, ...`.
pub fn compile_native_symbolic_jacobian(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    storage: NativeJacobianStorage,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    compile_native_symbolic_jacobian_with_parameters(
        equations, variables, time_arg, None, None, storage,
    )
}

/// Builds a native symbolic Jacobian evaluator with IVP parameter support.
///
/// Parameterized evaluators mirror the shared IVP lambdify/AOT input order:
/// `t, params..., y...`.
pub fn compile_native_symbolic_jacobian_with_parameters(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
    storage: NativeJacobianStorage,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    let parameter_values_handle =
        equation_parameter_values.map(|values| Arc::new(RwLock::new(values)));
    compile_native_symbolic_jacobian_with_parameter_handle(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
    )
}

/// Builds a native symbolic Jacobian evaluator backed by shared parameter values.
///
/// Residual and native Jacobian can use the same parameter storage, preventing
/// stale Newton matrices when a prepared solver updates parameters.
pub fn compile_native_symbolic_jacobian_with_parameter_handle(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    let symbolic_jacobian = build_symbolic_jacobian(equations, variables);
    validate_parameter_handle(equation_parameters, parameter_values_handle.as_ref());

    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |parameters| parameters.len()),
    );
    names.push(time_arg.to_string());
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().cloned());
    }
    names.extend(variables.iter().cloned());
    let name_refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();

    let rows = symbolic_jacobian.len();
    let cols = symbolic_jacobian.first().map_or(0, |row| row.len());
    let entries = compile_nonzero_entries(&symbolic_jacobian, &name_refs);

    match storage {
        NativeJacobianStorage::Dense => Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
            let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
            let args = build_args(t, parameter_values.as_ref(), y);
            let mut matrix = nalgebra::DMatrix::<f64>::zeros(rows, cols);
            for (row, col, eval) in &entries {
                matrix[(*row, *col)] = eval(&args);
            }
            BdfJacobian::Dense(matrix)
        }),
        NativeJacobianStorage::SparseTriplets => {
            Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
                let args = build_args(t, parameter_values.as_ref(), y);
                let triplets = entries
                    .iter()
                    .map(|(row, col, eval)| Triplet::new(*row, *col, eval(&args)))
                    .collect::<Vec<_>>();
                BdfJacobian::SparseTriplets { n: rows, triplets }
            })
        }
        NativeJacobianStorage::Banded { bandwidth } => {
            let (kl, ku) = bandwidth.unwrap_or_else(|| infer_bandwidth(rows, cols, &entries));
            Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
                let args = build_args(t, parameter_values.as_ref(), y);
                let mut banded = Banded::<f64>::zeros(rows, kl, ku)
                    .expect("symbolic Jacobian bandwidth should define valid banded storage");
                for (row, col, eval) in &entries {
                    banded
                        .set(*row, *col, eval(&args))
                        .expect("compiled symbolic entry must fit inferred bandwidth");
                }
                BdfJacobian::Banded(banded)
            })
        }
    }
}

/// Builds a native Jacobian evaluator from compiled sparse AOT callbacks.
///
/// This path keeps LSODE2 sparse/banded Jacobian evaluation in the AOT branch
/// instead of falling back to lambdified symbolic Jacobian entries.
pub fn compile_native_sparse_aot_jacobian_with_parameter_handle(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    generated_backend: SymbolicIvpGeneratedBackendConfig,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
) -> Result<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>, IvpBackendError> {
    let generated_backend = with_lsode2_sparse_jacobian_artifact_suffix(generated_backend);
    let options = SymbolicIvpProblemOptions::new()
        .with_equation_parameters(equation_parameters.unwrap_or(&[]).to_vec())
        .with_equation_parameter_values(
            equation_parameter_values.unwrap_or_else(|| DVector::zeros(0)),
        )
        .with_symbolic_assembly_backend(symbolic_assembly_backend);

    let prepared = prepare_generated_symbolic_ivp_sparse_backend(
        equations.to_vec(),
        variables.to_vec(),
        time_arg.to_string(),
        options,
        generated_backend,
    )
    .map_err(|err| IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    })?;

    if prepared.selected_backend != SelectedSymbolicIvpBackendKind::AotCompiled {
        return Err(IvpBackendError::GeneratedBackendFailure {
            message: "LSODE2 sparse/banded AOT Jacobian path expected a compiled sparse backend"
                .to_string(),
        });
    }

    let linked = prepared
        .linked_backend
        .ok_or_else(|| IvpBackendError::GeneratedBackendFailure {
            message:
                "LSODE2 sparse/banded AOT Jacobian path selected compiled backend but no runtime link is available"
                    .to_string(),
        })?;

    validate_parameter_handle(equation_parameters, parameter_values_handle.as_ref());

    let rows = prepared.jacobian_structure.rows;
    let cols = prepared.jacobian_structure.cols;
    let pattern = prepared
        .jacobian_structure
        .row_indices
        .iter()
        .copied()
        .zip(prepared.jacobian_structure.col_indices.iter().copied())
        .collect::<Vec<_>>();

    match storage {
        NativeJacobianStorage::Dense => {
            Ok(Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
                let args = build_args(t, parameter_values.as_ref(), y);
                let mut values = vec![0.0_f64; pattern.len()];
                (linked.jacobian_values_eval)(args.as_slice(), values.as_mut_slice());
                let mut matrix = nalgebra::DMatrix::<f64>::zeros(rows, cols);
                for ((row, col), value) in pattern.iter().zip(values.iter().copied()) {
                    matrix[(*row, *col)] = value;
                }
                BdfJacobian::Dense(matrix)
            }))
        }
        NativeJacobianStorage::SparseTriplets => {
            Ok(Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
                let args = build_args(t, parameter_values.as_ref(), y);
                let mut values = vec![0.0_f64; pattern.len()];
                (linked.jacobian_values_eval)(args.as_slice(), values.as_mut_slice());
                let triplets = pattern
                    .iter()
                    .zip(values.iter().copied())
                    .map(|((row, col), value)| Triplet::new(*row, *col, value))
                    .collect::<Vec<_>>();
                BdfJacobian::SparseTriplets { n: rows, triplets }
            }))
        }
        NativeJacobianStorage::Banded { bandwidth } => {
            let (kl, ku) =
                bandwidth.unwrap_or_else(|| infer_bandwidth_from_pattern(rows, cols, &pattern));
            Ok(Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
                let args = build_args(t, parameter_values.as_ref(), y);
                let mut values = vec![0.0_f64; pattern.len()];
                (linked.jacobian_values_eval)(args.as_slice(), values.as_mut_slice());
                let mut banded = Banded::<f64>::zeros(rows, kl, ku)
                    .expect("AOT sparse Jacobian pattern should define valid banded storage");
                for ((row, col), value) in pattern.iter().zip(values.iter().copied()) {
                    banded
                        .set(*row, *col, value)
                        .expect("AOT sparse Jacobian entry must fit inferred banded storage");
                }
                BdfJacobian::Banded(banded)
            }))
        }
    }
}

fn with_lsode2_sparse_jacobian_artifact_suffix(
    mut config: SymbolicIvpGeneratedBackendConfig,
) -> SymbolicIvpGeneratedBackendConfig {
    const SUFFIX: &str = "_sj";
    if let Some(crate_name) = config.crate_name_override.clone() {
        config.crate_name_override = Some(format!("{crate_name}{SUFFIX}"));
    }
    if let Some(module_name) = config.module_name_override.clone() {
        config.module_name_override = Some(format!("{module_name}{SUFFIX}"));
    }
    config
}

fn validate_parameter_handle(
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
) {
    match (equation_parameters, parameter_values_handle) {
        (Some(parameters), Some(handle)) => {
            let values = handle
                .read()
                .expect("shared IVP parameter state lock poisoned");
            assert_eq!(
                parameters.len(),
                values.len(),
                "native symbolic Jacobian parameter count mismatch"
            );
        }
        (Some(parameters), None) => {
            assert!(
                parameters.is_empty(),
                "native symbolic Jacobian expected parameter values"
            );
        }
        (None, Some(handle)) => {
            let values = handle
                .read()
                .expect("shared IVP parameter state lock poisoned");
            assert!(
                values.is_empty(),
                "native symbolic Jacobian got values without parameter names"
            );
        }
        (None, None) => {}
    }
}

fn read_parameter_values(handle: Option<&SharedIvpParameterValues>) -> Option<DVector<f64>> {
    handle.map(|handle| {
        handle
            .read()
            .expect("shared IVP parameter state lock poisoned")
            .clone()
    })
}

fn build_symbolic_jacobian(equations: &[Expr], variables: &[String]) -> Vec<Vec<Expr>> {
    equations
        .iter()
        .map(|expr| {
            variables
                .iter()
                .map(|variable| expr.diff(variable).simplify())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn compile_nonzero_entries(
    symbolic_jacobian: &[Vec<Expr>],
    name_refs: &[&str],
) -> Vec<CompiledEntry> {
    let mut entries = Vec::new();
    for (row, symbolic_row) in symbolic_jacobian.iter().enumerate() {
        for (col, expr) in symbolic_row.iter().enumerate() {
            if !expr.is_zero() {
                entries.push((
                    row,
                    col,
                    Expr::lambdify_borrowed_thread_safe(expr, name_refs),
                ));
            }
        }
    }
    entries
}

fn infer_bandwidth(rows: usize, cols: usize, entries: &[CompiledEntry]) -> (usize, usize) {
    let mut kl = 0usize;
    let mut ku = 0usize;
    for (row, col, _) in entries {
        kl = kl.max(row.saturating_sub(*col));
        ku = ku.max(col.saturating_sub(*row));
    }
    if rows == cols && rows > 0 {
        (kl, ku)
    } else {
        (rows.saturating_sub(1), cols.saturating_sub(1))
    }
}

fn infer_bandwidth_from_pattern(
    rows: usize,
    cols: usize,
    pattern: &[(usize, usize)],
) -> (usize, usize) {
    let mut kl = 0usize;
    let mut ku = 0usize;
    for (row, col) in pattern {
        kl = kl.max(row.saturating_sub(*col));
        ku = ku.max(col.saturating_sub(*row));
    }
    if rows == cols && rows > 0 {
        (kl, ku)
    } else {
        (rows.saturating_sub(1), cols.saturating_sub(1))
    }
}

fn build_args(t: f64, parameter_values: Option<&DVector<f64>>, y: &DVector<f64>) -> Vec<f64> {
    let mut args =
        Vec::with_capacity(1 + y.len() + parameter_values.map_or(0, |values| values.len()));
    args.push(t);
    if let Some(values) = parameter_values {
        args.extend(values.iter().copied());
    }
    args.extend(y.iter().copied());
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;

    #[test]
    fn sparse_native_jacobian_evaluates_only_symbolic_nonzeros() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::SparseTriplets,
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![1.0, 2.0]));
        let BdfJacobian::SparseTriplets { n, triplets } = out else {
            panic!("expected sparse triplet Jacobian");
        };
        assert_eq!(n, 2);
        assert_eq!(triplets.len(), 3);
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 0 && entry.val == -2.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 1 && entry.val == 1.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 1 && entry.col == 0 && entry.val == 3.0)
        );
    }

    #[test]
    fn dense_native_jacobian_evaluates_symbolic_entries_into_dense_matrix() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1 - 4*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::Dense,
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![1.0, 2.0]));
        let BdfJacobian::Dense(matrix) = out else {
            panic!("expected dense Jacobian");
        };
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], -2.0);
        assert_eq!(matrix[(0, 1)], 1.0);
        assert_eq!(matrix[(1, 0)], 3.0);
        assert_eq!(matrix[(1, 1)], -4.0);
    }

    #[test]
    fn banded_native_jacobian_preserves_symbolic_bandwidth() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1 - 4*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::Banded { bandwidth: None },
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![1.0, 2.0]));
        let BdfJacobian::Banded(banded) = out else {
            panic!("expected banded Jacobian");
        };
        assert_eq!(banded.n(), 2);
        assert_eq!(banded.kl(), 1);
        assert_eq!(banded.ku(), 1);
        assert_eq!(banded[(0, 0)], -2.0);
        assert_eq!(banded[(0, 1)], 1.0);
        assert_eq!(banded[(1, 0)], 3.0);
        assert_eq!(banded[(1, 1)], -4.0);
    }

    #[test]
    fn native_jacobian_uses_parameter_input_order() {
        let equations = vec![
            Expr::parse_expression("a*y1 + b*y2"),
            Expr::parse_expression("y1"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian_with_parameters(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(DVector::from_vec(vec![2.0, -3.0])),
            NativeJacobianStorage::SparseTriplets,
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![10.0, 20.0]));
        let BdfJacobian::SparseTriplets { n, triplets } = out else {
            panic!("expected sparse triplet Jacobian");
        };
        assert_eq!(n, 2);
        assert_eq!(triplets.len(), 3);
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 0 && entry.val == 2.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 1 && entry.val == -3.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 1 && entry.col == 0 && entry.val == 1.0)
        );
    }

    #[test]
    fn native_jacobian_reads_updated_shared_parameter_values() {
        let equations = vec![
            Expr::parse_expression("a*y1 + b*y2"),
            Expr::parse_expression("y1"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string()];
        let handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0, -3.0])));
        let mut jacobian = compile_native_symbolic_jacobian_with_parameter_handle(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(handle.clone()),
            NativeJacobianStorage::SparseTriplets,
        );

        {
            let mut values = handle
                .write()
                .expect("shared IVP parameter state lock should be writable");
            *values = DVector::from_vec(vec![5.0, 7.0]);
        }

        let out = jacobian(0.0, &DVector::from_vec(vec![10.0, 20.0]));
        let BdfJacobian::SparseTriplets { triplets, .. } = out else {
            panic!("expected sparse triplet Jacobian");
        };
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 0 && entry.val == 5.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 1 && entry.val == 7.0)
        );
    }

    #[test]
    fn sparse_aot_jacobian_artifact_suffix_avoids_residual_name_collision() {
        let cfg =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/lsode2-tests")
                .with_crate_name_override(Some("generated_lsode2_same_name".to_string()))
                .with_module_name_override(Some("generated_lsode2_same_name".to_string()));
        let patched = with_lsode2_sparse_jacobian_artifact_suffix(cfg);

        assert_eq!(
            patched.crate_name_override.as_deref(),
            Some("generated_lsode2_same_name_sj")
        );
        assert_eq!(
            patched.module_name_override.as_deref(),
            Some("generated_lsode2_same_name_sj")
        );
    }
}
