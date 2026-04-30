//! Sparse Jacobian builders for packed [`Atom`] expressions.
//!
//! This module is intentionally View-native: the hot path converts boxed
//! `Expr` residuals into packed [`Atom`] once, then performs band-aware sparse
//! Jacobian construction directly on `AtomView` / `Atom` without materializing
//! back into `Expr`.

use ahash::{HashMap, HashSet};
use rayon::prelude::*;

use super::{
    Atom,
    conversions::expr_to_atom,
    state::{Symbol, Workspace},
};
use crate::symbolic::symbolic_engine::Expr;

/// One nonzero sparse Jacobian entry produced by the View-native symbolic path.
#[derive(Clone)]
pub struct SparseAtomJacobianEntry {
    pub row: usize,
    pub col: usize,
    pub value: Atom,
}

/// Prepared packed representation of a sparse symbolic system.
///
/// The expensive `Expr -> Atom` conversion and variable lookup preprocessing are
/// performed once here so that repeated Jacobian builds can stay on the View
/// path.
pub struct PreparedSparseAtomSystem {
    atoms: Vec<Atom>,
    variable_symbols: Vec<Symbol>,
    column_indices_per_row: Vec<Vec<usize>>,
}

impl PreparedSparseAtomSystem {
    /// Prepare packed atoms and sparse variable-to-column lookup data.
    pub fn from_exprs(
        functions: &[Expr],
        variable_names: &[String],
        variables_for_all_discrete: &[Vec<String>],
    ) -> Self {
        let atoms = functions.iter().map(expr_to_atom).collect::<Vec<_>>();
        Self::from_atoms(&atoms, variable_names, variables_for_all_discrete)
    }

    /// Prepare sparse lookup data starting from already packed residual atoms.
    pub fn from_atoms(
        functions: &[Atom],
        variable_names: &[String],
        variables_for_all_discrete: &[Vec<String>],
    ) -> Self {
        let atoms = functions.to_vec();
        let variable_symbols = variable_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let column_indices_per_row =
            build_column_indices_per_row(variable_names, variables_for_all_discrete);

        Self {
            atoms,
            variable_symbols,
            column_indices_per_row,
        }
    }

    /// Number of residual equations in the prepared system.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Returns `true` if the prepared system has no residual equations.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Access the prepared packed residual atoms.
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    /// Access the flattened variable symbols in solver order.
    pub fn variable_symbols(&self) -> &[Symbol] {
        &self.variable_symbols
    }

    /// Access sparse column indices relevant to each residual row.
    pub fn column_indices_per_row(&self) -> &[Vec<usize>] {
        &self.column_indices_per_row
    }

    /// Build a sparse symbolic Jacobian directly over packed atoms.
    pub fn calc_sparse_jacobian_with_bandwidth(
        &self,
        bandwidth: Option<(usize, usize)>,
    ) -> Vec<SparseAtomJacobianEntry> {
        let n_vars = self.variable_symbols.len();
        let rows: Vec<Vec<SparseAtomJacobianEntry>> = self
            .atoms
            .par_iter()
            .enumerate()
            .map(|(row, atom)| {
                let (left, right) = band_window(row, n_vars, bandwidth);
                Workspace::get_local().with(|ws| {
                    let relevant_cols =
                        band_limited_columns(&self.column_indices_per_row[row], left, right);
                    let mut entries = Vec::with_capacity(relevant_cols.len());
                    let mut partial = ws.new_atom();
                    for &col in relevant_cols {
                        let has_nonzero = atom
                            .as_view()
                            .try_derivative_with_ws_into(
                                self.variable_symbols[col],
                                ws,
                                &mut partial,
                            )
                            .expect("sparse atom Jacobian build: malformed der(...) expression");
                        if has_nonzero {
                            entries.push(SparseAtomJacobianEntry {
                                row,
                                col,
                                value: partial.into_inner(),
                            });
                            partial = ws.new_atom();
                        }
                    }
                    entries
                })
            })
            .collect();

        rows.into_iter().flatten().collect()
    }
}

/// Convenience helper for one-shot use from legacy `Expr`-based callers.
pub fn calc_sparse_jacobian_atom_with_bandwidth(
    functions: &[Expr],
    variable_names: &[String],
    variables_for_all_discrete: &[Vec<String>],
    bandwidth: Option<(usize, usize)>,
) -> Vec<SparseAtomJacobianEntry> {
    PreparedSparseAtomSystem::from_exprs(functions, variable_names, variables_for_all_discrete)
        .calc_sparse_jacobian_with_bandwidth(bandwidth)
}

/// One-shot sparse Jacobian builder starting from already packed residual atoms.
pub fn calc_sparse_jacobian_atom_from_atoms_with_bandwidth(
    functions: &[Atom],
    variable_names: &[String],
    variables_for_all_discrete: &[Vec<String>],
    bandwidth: Option<(usize, usize)>,
) -> Vec<SparseAtomJacobianEntry> {
    PreparedSparseAtomSystem::from_atoms(functions, variable_names, variables_for_all_discrete)
        .calc_sparse_jacobian_with_bandwidth(bandwidth)
}

fn build_column_indices_per_row(
    variable_names: &[String],
    variables_for_all_discrete: &[Vec<String>],
) -> Vec<Vec<usize>> {
    let index_by_name: HashMap<&str, usize> = variable_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect();

    variables_for_all_discrete
        .iter()
        .map(|vars| {
            let mut seen = HashSet::default();
            let mut cols = Vec::new();
            for var in vars {
                if let Some(&idx) = index_by_name.get(var.as_str()) {
                    if seen.insert(idx) {
                        cols.push(idx);
                    }
                }
            }
            cols.sort_unstable();
            cols
        })
        .collect()
}

fn band_window(
    row_index: usize,
    n_vars: usize,
    bandwidth: Option<(usize, usize)>,
) -> (usize, usize) {
    if let Some((kl, ku)) = bandwidth {
        let right = std::cmp::min(row_index + ku + 1, n_vars);
        let left = if row_index as i32 - (kl as i32) - 1 < 0 {
            0
        } else {
            row_index - kl - 1
        };
        (left, right)
    } else {
        (0, n_vars)
    }
}

fn band_limited_columns(sorted_cols: &[usize], left: usize, right: usize) -> &[usize] {
    let start = sorted_cols.partition_point(|&col| col < left);
    let end = sorted_cols.partition_point(|&col| col < right);
    &sorted_cols[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::View::conversions::atom_to_expr;

    fn parse_expr(input: &str) -> Expr {
        Expr::parse_expression(input)
    }

    #[test]
    fn prepared_sparse_atom_system_builds_sparse_entries() {
        let functions = vec![parse_expr("x0 + x1^2"), parse_expr("x1*x2 + sin(x0)")];
        let variable_names = vec!["x0".to_string(), "x1".to_string(), "x2".to_string()];
        let variables_for_all_discrete = vec![
            vec!["x0".to_string(), "x1".to_string()],
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
        ];

        let prepared = PreparedSparseAtomSystem::from_exprs(
            &functions,
            &variable_names,
            &variables_for_all_discrete,
        );
        let sparse = prepared.calc_sparse_jacobian_with_bandwidth(None);

        assert!(!sparse.is_empty(), "expected non-empty sparse Jacobian");
        let rendered = sparse
            .iter()
            .map(|entry| (entry.row, entry.col, atom_to_expr(&entry.value).to_string()))
            .collect::<Vec<_>>();

        assert!(
            rendered.iter().any(|(r, c, _)| *r == 0 && *c == 0),
            "missing dF0/dx0"
        );
        assert!(
            rendered.iter().any(|(r, c, _)| *r == 0 && *c == 1),
            "missing dF0/dx1"
        );
        assert!(
            rendered.iter().any(|(r, c, _)| *r == 1 && *c == 2),
            "missing dF1/dx2"
        );
    }

    #[test]
    fn prepared_sparse_atom_system_respects_bandwidth_window() {
        let functions = vec![parse_expr("x0 + x3"), parse_expr("x1 + x2")];
        let variable_names = vec![
            "x0".to_string(),
            "x1".to_string(),
            "x2".to_string(),
            "x3".to_string(),
        ];
        let variables_for_all_discrete = vec![
            vec!["x0".to_string(), "x3".to_string()],
            vec!["x1".to_string(), "x2".to_string()],
        ];

        let sparse = calc_sparse_jacobian_atom_with_bandwidth(
            &functions,
            &variable_names,
            &variables_for_all_discrete,
            Some((0, 1)),
        );

        assert!(
            sparse
                .iter()
                .all(|entry| !(entry.row == 0 && entry.col == 3)),
            "band-limited sweep should exclude far-off-band x3 derivative"
        );
    }
}
