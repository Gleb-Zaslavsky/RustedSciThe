//! Sparse Jacobian builders for packed [`Atom`] expressions.
//!
//! This module is intentionally View-native: the hot path converts boxed
//! `Expr` residuals into packed [`Atom`] once, then performs band-aware sparse
//! Jacobian construction directly on `AtomView` / `Atom` without materializing
//! back into `Expr`.

use ahash::{HashMap, HashSet};
use rayon::prelude::*;
use std::sync::Arc;

use super::{
    Atom, AtomView, DerivativeError,
    conversions::expr_to_atom,
    state::{Symbol, Workspace},
};
use crate::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetry};
use crate::symbolic::symbolic_engine::Expr;

/// One nonzero sparse Jacobian entry produced by the View-native symbolic path.
#[derive(Clone, Debug)]
pub struct SparseAtomJacobianEntry {
    pub row: usize,
    pub col: usize,
    pub value: Atom,
}

/// A symbolic derivative failure tied to its equation and variable column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseAtomJacobianError {
    pub row: usize,
    pub col: usize,
    pub source: DerivativeError,
}

impl std::fmt::Display for SparseAtomJacobianError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "row {}, column {}: {}",
            self.row, self.col, self.source
        )
    }
}

impl std::error::Error for SparseAtomJacobianError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Prepared packed representation of a sparse symbolic system.
///
/// The expensive `Expr -> Atom` conversion and variable lookup preprocessing are
/// performed once here so that repeated Jacobian builds can stay on the View
/// path.
pub struct PreparedSparseAtomSystem {
    atoms: Arc<[Atom]>,
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
        Self::from_owned_atoms(atoms, variable_names, variables_for_all_discrete)
    }

    /// Prepare sparse lookup data starting from already packed residual atoms.
    pub fn from_atoms(
        functions: &[Atom],
        variable_names: &[String],
        variables_for_all_discrete: &[Vec<String>],
    ) -> Self {
        Self::from_owned_atoms(
            functions.to_vec(),
            variable_names,
            variables_for_all_discrete,
        )
    }

    fn from_owned_atoms(
        atoms: Vec<Atom>,
        variable_names: &[String],
        variables_for_all_discrete: &[Vec<String>],
    ) -> Self {
        let atoms: Arc<[Atom]> = atoms.into();
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

    /// Prepare native sparse work from variables actually present in each row.
    ///
    /// This avoids probing every state variable for every equation in sparse
    /// IVPs. The general `from_atoms` constructor keeps its caller-supplied
    /// dependency contract for discretized BVP systems.
    pub fn from_atoms_discovering_dependencies(
        functions: &[Atom],
        variable_names: &[String],
    ) -> Self {
        Self::from_shared_atoms_discovering_dependencies(
            Arc::from(functions.to_vec()),
            variable_names,
        )
    }

    /// Prepare sparse lookup data while sharing an already-owned Atom graph.
    ///
    /// This avoids cloning every packed expression when a caller needs both
    /// residual evaluators and a Jacobian plan from the same prepared system.
    pub(crate) fn from_shared_atoms_discovering_dependencies(
        atoms: Arc<[Atom]>,
        variable_names: &[String],
    ) -> Self {
        let variable_symbols = variable_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let column_by_symbol: HashMap<Symbol, usize> = variable_symbols
            .iter()
            .copied()
            .enumerate()
            .map(|(column, symbol)| (symbol, column))
            .collect();
        let column_indices_per_row = atoms
            .iter()
            .map(|atom| {
                let mut columns = Vec::new();
                collect_variable_columns(atom.as_view(), &column_by_symbol, &mut columns);
                columns.sort_unstable();
                columns.dedup();
                columns
            })
            .collect();

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

    /// Number of derivative candidates retained after dependency filtering.
    pub fn derivative_candidate_count(&self) -> usize {
        self.column_indices_per_row.iter().map(Vec::len).sum()
    }

    /// Build a sparse symbolic Jacobian directly over packed atoms.
    pub fn calc_sparse_jacobian_with_bandwidth(
        &self,
        bandwidth: Option<(usize, usize)>,
    ) -> Vec<SparseAtomJacobianEntry> {
        self.try_calc_sparse_jacobian_with_bandwidth(bandwidth)
            .expect("sparse atom Jacobian build: malformed der(...) expression")
    }

    /// Fallible native variant used by typed IVP/BVP preparation boundaries.
    ///
    /// The infallible method above remains for established compatibility
    /// callers, while new solver backends can surface malformed symbolic
    /// derivatives as a normal preparation error instead of panicking.
    pub fn try_calc_sparse_jacobian_with_bandwidth(
        &self,
        bandwidth: Option<(usize, usize)>,
    ) -> Result<Vec<SparseAtomJacobianEntry>, SparseAtomJacobianError> {
        let n_vars = self.variable_symbols.len();
        let rows: Result<Vec<Vec<SparseAtomJacobianEntry>>, SparseAtomJacobianError> = self
            .atoms
            .par_iter()
            .enumerate()
            .map(
                |(row, atom)| -> Result<Vec<SparseAtomJacobianEntry>, SparseAtomJacobianError> {
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
                                .map_err(|source| SparseAtomJacobianError { row, col, source })?;
                            if has_nonzero {
                                entries.push(SparseAtomJacobianEntry {
                                    row,
                                    col,
                                    value: partial.into_inner(),
                                });
                                partial = ws.new_atom();
                            }
                        }
                        Ok(entries)
                    })
                },
            )
            .collect();

        Ok(rows?.into_iter().flatten().collect())
    }

    /// Builds the sparse Jacobian and attributes the Atom differentiation
    /// portion to the optional IVP telemetry stream.
    ///
    /// The public non-telemetry method remains the zero-overhead compatibility
    /// path. `SparsePattern` at the caller remains an inclusive aggregate for
    /// this operation; `SymbolicDifferentiation` is its typed child stage.
    pub fn calc_sparse_jacobian_with_bandwidth_and_telemetry(
        &self,
        bandwidth: Option<(usize, usize)>,
        telemetry: &IvpTelemetry,
    ) -> Vec<SparseAtomJacobianEntry> {
        let started = telemetry.start_cold_stage(IvpColdStage::SymbolicDifferentiation);
        let result = self.calc_sparse_jacobian_with_bandwidth(bandwidth);
        telemetry.record_cold_stage(IvpColdStage::SymbolicDifferentiation, started);
        result
    }
}

fn collect_variable_columns(
    view: AtomView<'_>,
    column_by_symbol: &HashMap<Symbol, usize>,
    columns: &mut Vec<usize>,
) {
    match view {
        AtomView::Num(_) => {}
        AtomView::Var(variable) => {
            if let Some(column) = column_by_symbol.get(&variable.get_symbol()) {
                columns.push(*column);
            }
        }
        AtomView::Fun(function) => {
            for child in function.iter() {
                collect_variable_columns(child, column_by_symbol, columns);
            }
        }
        AtomView::Pow(power) => {
            let (base, exponent) = power.get_base_exp();
            collect_variable_columns(base, column_by_symbol, columns);
            collect_variable_columns(exponent, column_by_symbol, columns);
        }
        AtomView::Mul(product) => {
            for child in product.iter() {
                collect_variable_columns(child, column_by_symbol, columns);
            }
        }
        AtomView::Add(sum) => {
            for child in sum.iter() {
                collect_variable_columns(child, column_by_symbol, columns);
            }
        }
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

    #[test]
    fn discovered_dependencies_match_all_columns_for_nonlinear_and_parameterized_rows() {
        // Deliberately non-lexical ABI order. Time and parameters are not state columns.
        let variables = ["z", "x", "y", "unused"].map(str::to_string).to_vec();
        let atoms = [
            "exp(x*y)+sin(z)+p*t",
            "x^y+p*z",
            "x*x+x*y+x*z",
            "p*x+t",
            "42",
            "exp(p*t)",
        ]
        .map(|source| crate::symbolic::View::parser::parse(source).unwrap());
        let discovered =
            PreparedSparseAtomSystem::from_atoms_discovering_dependencies(&atoms, &variables);
        assert_eq!(
            discovered.column_indices_per_row(),
            &[
                vec![0, 1, 2],
                vec![0, 1, 2],
                vec![0, 1, 2],
                vec![1],
                vec![],
                vec![]
            ]
        );
        let exhaustive = PreparedSparseAtomSystem::from_atoms(
            &atoms,
            &variables,
            &vec![variables.clone(); atoms.len()],
        );
        for bandwidth in [None, Some((0, 0)), Some((1, 2))] {
            let actual = discovered
                .try_calc_sparse_jacobian_with_bandwidth(bandwidth)
                .unwrap();
            let expected = exhaustive
                .try_calc_sparse_jacobian_with_bandwidth(bandwidth)
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!((actual.row, actual.col), (expected.row, expected.col));
                assert_eq!(actual.value, expected.value);
            }
        }
    }

    #[test]
    fn discovered_dependencies_keep_diffusion_jacobian_candidates_linear() {
        const DIMENSION: usize = 2048;
        let variables = (0..DIMENSION)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        let equations = (0..DIMENSION)
            .map(|row| {
                let diagonal = format!("y{row}");
                let mut terms = vec![format!("-2*{diagonal}")];
                if row > 0 {
                    terms.push(format!("y{}", row - 1));
                }
                if row + 1 < DIMENSION {
                    terms.push(format!("y{}", row + 1));
                }
                Expr::parse_expression(&terms.join("+"))
            })
            .collect::<Vec<_>>();
        let atoms = equations.iter().map(expr_to_atom).collect::<Vec<_>>();
        let prepared =
            PreparedSparseAtomSystem::from_atoms_discovering_dependencies(&atoms, &variables);

        assert_eq!(
            prepared.derivative_candidate_count(),
            3 * DIMENSION - 2,
            "a tridiagonal chain should consider only its three local columns per interior row"
        );
        let entries = prepared
            .try_calc_sparse_jacobian_with_bandwidth(None)
            .expect("diffusion chain derivatives should prepare");
        assert_eq!(entries.len(), 3 * DIMENSION - 2);
        assert!(
            entries
                .iter()
                .all(|entry| { entry.row.abs_diff(entry.col) <= 1 })
        );
        for entry in entries {
            let expected = Atom::new_num(if entry.row == entry.col { -2 } else { 1 });
            assert_eq!(entry.value, expected);
        }
    }

    #[test]
    fn sparse_derivative_error_preserves_row_and_column() {
        let equation = crate::parse!("der(0,x)").expect("malformed derivative tag should parse");
        let prepared = PreparedSparseAtomSystem::from_atoms_discovering_dependencies(
            &[equation],
            &["x".to_string()],
        );

        let error = prepared
            .try_calc_sparse_jacobian_with_bandwidth(None)
            .expect_err("malformed derivative syntax should return a typed failure");
        assert_eq!((error.row, error.col), (0, 0));
        assert_eq!(
            error.source,
            super::DerivativeError::DerivativeTargetMustBeFunction
        );
        assert!(error.to_string().contains("row 0, column 0"));
    }

    #[test]
    fn discovered_dependency_plan_shares_prepared_atom_graph() {
        let variables = vec!["x".to_string()];
        let atoms: Arc<[Atom]> = vec![crate::symbolic::View::parser::parse("x^2").unwrap()].into();
        let prepared = PreparedSparseAtomSystem::from_shared_atoms_discovering_dependencies(
            Arc::clone(&atoms),
            &variables,
        );

        assert!(Arc::ptr_eq(&prepared.atoms, &atoms));
        assert_eq!(prepared.column_indices_per_row(), &[vec![0]]);
        assert_eq!(
            prepared
                .try_calc_sparse_jacobian_with_bandwidth(None)
                .unwrap()
                .len(),
            1
        );
    }
}
