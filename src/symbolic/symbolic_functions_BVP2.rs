//! Banded-oriented symbolic helpers for the BVP lambdify pipeline.
//!
//! The main symbolic BVP module, [`crate::symbolic::symbolic_functions_BVP`],
//! already carries the production sparse mainline. This companion module keeps
//! the new banded-specific design and helper logic isolated so the original
//! file does not become even heavier.
//!
//! Current scope of this module:
//! - infer node-major block layout from the discretized BVP system,
//! - compile residual/Jacobian evaluators for the banded lambdify path,
//! - evaluate Jacobians directly into native `BandedAssembly`,
//! - expose chunking/threshold controls used by the runtime callbacks.

use crate::somelinalg::banded::{
    banded_assembly::BandedAssembly, BandedError, LinearSolverConfig, NodeMajorLayout,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::Jacobian;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Work partitioning strategy for future parallel banded Jacobian generation.
///
/// The first implementation will likely start with diagonal chunks because
/// that matches `BandedAssembly` storage directly. We keep the enum explicit so
/// later experiments can compare diagonal-oriented and entry-oriented
/// scheduling without redesigning the surrounding API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandedJacobianChunking {
    /// Parallelize over scalar diagonals of the banded storage.
    Diagonal,
    /// Parallelize over pre-grouped symbolic nonzero entries.
    EntryChunks,
}

/// Solver family intended for the banded lambdify path.
///
/// `BlockTridiagonalNative` is the primary target because it already outperforms
/// the generic banded LU path in local benchmarks. The other variants are kept
/// explicit so the pipeline can later expose fallback/compare options without
/// overloading the storage layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandedLinearSolverBackend {
    /// Native Rust direct solver operating on `BlockTridiagonal`.
    BlockTridiagonalNative,
    /// Generic banded LU path for future wider-band experiments.
    GeneralBandedNative,
    /// Fallback path when a banded-native branch is unavailable or rejected.
    FaerSparseFallback,
}

/// User-facing configuration for the future banded lambdify branch.
///
/// The sparse mainline already has `BvpBackendConfig`; this struct is narrower
/// and only captures the extra choices unique to the banded path.
#[derive(Clone, Debug)]
pub struct BandedLambdifyConfig {
    /// Parallel work decomposition used by symbolic lambdify generators.
    pub jacobian_chunking: BandedJacobianChunking,
    /// Linear solver backend used after conversion to a solver-oriented format.
    pub linear_solver_backend: BandedLinearSolverBackend,
    /// Native solver policy/fallback choices reused from the banded linalg layer.
    pub linear_solver_config: LinearSolverConfig,
    /// Threshold below which generated Jacobian entries are treated as zero.
    pub structural_threshold: f64,
}

impl Default for BandedLambdifyConfig {
    fn default() -> Self {
        Self {
            jacobian_chunking: BandedJacobianChunking::Diagonal,
            linear_solver_backend: BandedLinearSolverBackend::BlockTridiagonalNative,
            linear_solver_config: LinearSolverConfig::default(),
            structural_threshold: 0.0,
        }
    }
}

/// Structural summary of the discretized symbolic BVP as seen by the banded path.
///
/// This is the crucial bridge between the symbolic layer and the native banded
/// linear algebra layer:
/// - `scalar_bandwidth` describes the raw Jacobian sparsity in scalar indices,
/// - `layout` captures the intended node-major blocking,
/// - `block_tridiagonal_compatible` tells us whether the current symbolic
///   bandwidth is at least compatible with the fastest native solver target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BandedStructurePlan {
    /// Number of scalar unknowns in the discretized Newton system.
    pub n_unknowns: usize,
    /// Number of residual equations in the discretized Newton system.
    pub n_equations: usize,
    /// Scalar half-bandwidth `(kl, ku)` already tracked by the symbolic path.
    pub scalar_bandwidth: (usize, usize),
    /// Inferred node-major layout used by the block-tridiagonal solver path.
    pub layout: NodeMajorLayout,
    /// Heuristic flag indicating whether the system fits the fast
    /// block-tridiagonal target under node-major ordering.
    pub block_tridiagonal_compatible: bool,
}

type BandedScalarEvaluator = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// One compiled scalar diagonal of the banded Jacobian.
///
/// Performance-wise this is the key data layout for the future banded lambdify
/// mainline:
/// - compile phase groups scalar nonzeros by diagonal,
/// - each diagonal stores a dense `Vec<Option<...>>` indexed by the natural
///   position inside that diagonal,
/// - runtime evaluation becomes a straight linear pass without lookups,
///   hashing, or sparse triplet assembly.
struct CompiledBandedDiagonal {
    evaluators: Vec<Option<BandedScalarEvaluator>>,
}

/// One compiled scalar symbolic entry of the Jacobian.
///
/// This representation is used by the `EntryChunks` runtime mode where we
/// evaluate entries independently and scatter them into `BandedAssembly`.
struct CompiledBandedEntry {
    row: usize,
    col: usize,
    evaluator: BandedScalarEvaluator,
}

impl BandedStructurePlan {
    /// Returns the scalar matrix dimension of the discretized system.
    #[inline]
    pub fn n(&self) -> usize {
        self.n_unknowns
    }

    /// Number of discretization nodes under node-major blocking.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.layout.n_nodes()
    }

    /// Number of state variables grouped into one node block.
    #[inline]
    pub fn vars_per_node(&self) -> usize {
        self.layout.vars_per_node()
    }
}

/// Returns the base symbolic variable name before the discretization suffix.
///
/// Examples:
/// - `"y_0"` -> `"y"`
/// - `"temperature_17"` -> `"temperature"`
/// - `"u"` -> `"u"`
fn base_variable_name(name: &str) -> &str {
    name.rsplit_once('_').map_or(name, |(base, _)| base)
}

fn compile_banded_scalar_evaluator(expr: &Expr, argument_names: &[&str]) -> BandedScalarEvaluator {
    Expr::lambdify_borrowed_thread_safe(expr, argument_names)
}

impl Jacobian {
    /// Infers the future banded runtime layout from the already discretized BVP system.
    ///
    /// The current banded-native solver target assumes node-major ordering:
    /// each grid node contributes one dense block of size `n_equations_in_system`.
    /// This helper derives exactly that view from the flattened symbolic variable
    /// list stored in `self.variable_string`.
    ///
    /// The method is intentionally conservative:
    /// - it requires a square Newton system,
    /// - it requires at least one symbolic variable name,
    /// - it treats missing `self.bandwidth` as "band information not prepared yet".
    ///
    /// Later code generation stages can call this once and reuse the returned
    /// plan for storage allocation, solver selection, and validation.
    pub fn infer_banded_structure_plan(&self) -> Result<BandedStructurePlan, BandedError> {
        let n_unknowns = self.vector_of_variables.len();
        let n_equations = self.vector_of_functions.len();

        if n_unknowns == 0 || n_equations == 0 || n_unknowns != n_equations {
            return Err(BandedError::DimensionMismatch);
        }

        let scalar_bandwidth = self.bandwidth.ok_or(BandedError::DimensionMismatch)?;
        if self.variable_string.is_empty() {
            return Err(BandedError::DimensionMismatch);
        }

        // Variable names in the discretized BVP follow a node-major pattern such as:
        // y_0, z_0, y_1, z_1, ...
        // Counting unique base names in that flattened list gives us the block size.
        let mut seen: HashSet<&str> = HashSet::new();
        let mut vars_per_node = 0usize;
        for name in &self.variable_string {
            let inserted = seen.insert(base_variable_name(name));
            if inserted {
                vars_per_node += 1;
            }
        }

        if vars_per_node == 0 || n_unknowns % vars_per_node != 0 {
            return Err(BandedError::DimensionMismatch);
        }

        let n_nodes = n_unknowns / vars_per_node;
        let layout = NodeMajorLayout::new(n_nodes, vars_per_node)?;

        // Under node-major ordering, same-node plus neighboring-node coupling
        // yields scalar offsets within +/- (2 * block_size - 1).
        // This is a useful fast check before we try to lower into a true
        // `BlockTridiagonal` representation.
        let max_scalar_half_bandwidth = vars_per_node
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(BandedError::DimensionMismatch)?;

        let block_tridiagonal_compatible = scalar_bandwidth.0 <= max_scalar_half_bandwidth
            && scalar_bandwidth.1 <= max_scalar_half_bandwidth;

        Ok(BandedStructurePlan {
            n_unknowns,
            n_equations,
            scalar_bandwidth,
            layout,
            block_tridiagonal_compatible,
        })
    }

    /// Allocates empty native banded storage sized for the current symbolic problem.
    ///
    /// This is the first reusable building block for the future lambdified banded
    /// Jacobian generator: symbolic lowering code will allocate the target storage
    /// once, fill diagonals in parallel, and then pass the populated assembly to
    /// the banded linear-solver bridge.
    pub fn allocate_banded_assembly_from_plan(
        &self,
        plan: &BandedStructurePlan,
    ) -> Result<BandedAssembly, BandedError> {
        BandedAssembly::zeros(
            plan.n_unknowns,
            plan.scalar_bandwidth.0,
            plan.scalar_bandwidth.1,
        )
    }

    /// Convenience helper that performs structure inference and immediately
    /// allocates the corresponding empty banded storage.
    pub fn allocate_banded_assembly(&self) -> Result<BandedAssembly, BandedError> {
        let plan = self.infer_banded_structure_plan()?;
        self.allocate_banded_assembly_from_plan(&plan)
    }

    /// Returns `true` when the current symbolic problem is a plausible candidate
    /// for the fast node-major block-tridiagonal solver path.
    ///
    /// This is intentionally a lightweight heuristic rather than a full formal
    /// proof of structure. The full validation will happen once the real banded
    /// Jacobian generator can inspect explicit nonzero positions.
    pub fn is_block_tridiagonal_candidate(&self) -> bool {
        self.infer_banded_structure_plan()
            .map(|plan| plan.block_tridiagonal_compatible)
            .unwrap_or(false)
    }

    /// Builds a compile-time plan for fast banded Jacobian evaluation.
    ///
    /// This method performs the expensive symbolic preparation once:
    /// - optional parameter substitution when numeric parameter values are known,
    /// - grouping sparse symbolic Jacobian entries by diagonal,
    /// - compiling each scalar expression into a thread-safe numeric closure,
    /// - storing them in direct positional order for branch-light runtime loops.
    ///
    /// The returned data structure is intentionally optimized for the runtime
    /// closure rather than for readability.
    fn compile_banded_diagonal_plan(
        &self,
        plan: &BandedStructurePlan,
    ) -> Result<Vec<CompiledBandedDiagonal>, BandedError> {
        let parameter_map = self.parameter_substitution_map();
        let argument_names = self.runtime_argument_names(&parameter_map)?;
        let argument_name_refs: Vec<&str> =
            argument_names.iter().map(|name| name.as_str()).collect();

        let mut diagonals: Vec<CompiledBandedDiagonal> =
            Vec::with_capacity(plan.scalar_bandwidth.0 + plan.scalar_bandwidth.1 + 1);
        for offset in -(plan.scalar_bandwidth.0 as isize)..=(plan.scalar_bandwidth.1 as isize) {
            let len = plan.n_unknowns.saturating_sub(offset.unsigned_abs());
            diagonals.push(CompiledBandedDiagonal {
                evaluators: std::iter::repeat_with(|| None).take(len).collect(),
            });
        }

        // Lambdification dominates setup for large meshes. Compile independent
        // nonzeros in parallel, then scatter closures into diagonal storage in
        // source order so runtime layout remains deterministic.
        let compiled_entries: Vec<(usize, usize, BandedScalarEvaluator)> = self
            .symbolic_jacobian_sparse
            .par_iter()
            .filter_map(|(row, col, expr)| {
                let offset = *col as isize - *row as isize;
                if offset < -(plan.scalar_bandwidth.0 as isize)
                    || offset > plan.scalar_bandwidth.1 as isize
                {
                    return None;
                }

                let diag_index = (offset + plan.scalar_bandwidth.0 as isize) as usize;
                let pos = if offset >= 0 { *row } else { *col };
                let evaluator = if let Some(ref map) = parameter_map {
                    let prepared_expr = expr.set_variable_from_map(map);
                    compile_banded_scalar_evaluator(&prepared_expr, &argument_name_refs)
                } else {
                    compile_banded_scalar_evaluator(expr, &argument_name_refs)
                };

                Some((diag_index, pos, evaluator))
            })
            .collect();

        for (diag_index, pos, evaluator) in compiled_entries {
            diagonals[diag_index].evaluators[pos] = Some(evaluator);
        }

        Ok(diagonals)
    }

    /// Builds an entry-wise compile plan used by `EntryChunks` runtime mode.
    fn compile_banded_entry_plan(
        &self,
        plan: &BandedStructurePlan,
    ) -> Result<Vec<CompiledBandedEntry>, BandedError> {
        let parameter_map = self.parameter_substitution_map();
        let argument_names = self.runtime_argument_names(&parameter_map)?;
        let argument_name_refs: Vec<&str> =
            argument_names.iter().map(|name| name.as_str()).collect();
        Ok(self
            .symbolic_jacobian_sparse
            .par_iter()
            .filter_map(|(row, col, expr)| {
                let offset = *col as isize - *row as isize;
                if offset < -(plan.scalar_bandwidth.0 as isize)
                    || offset > plan.scalar_bandwidth.1 as isize
                {
                    return None;
                }

                let evaluator = if let Some(ref map) = parameter_map {
                    let prepared_expr = expr.set_variable_from_map(map);
                    compile_banded_scalar_evaluator(&prepared_expr, &argument_name_refs)
                } else {
                    compile_banded_scalar_evaluator(expr, &argument_name_refs)
                };

                Some(CompiledBandedEntry {
                    row: *row,
                    col: *col,
                    evaluator,
                })
            })
            .collect())
    }

    /// Returns a high-performance banded Jacobian evaluator producing
    /// `BandedAssembly` directly.
    ///
    /// Performance design:
    /// - all symbolic lambdification happens once at setup time,
    /// - runtime evaluates pre-grouped diagonal arrays,
    /// - no triplet collection and no dynamic sparse indexing are used,
    /// - when parameter values are already known, parameter substitution is
    ///   done at compile time and the runtime closure accepts only unknowns.
    ///
    /// The returned closure is thread-safe and intentionally immutable. That
    /// makes it easier to reuse in future frozen-Jacobian branches.
    pub(crate) fn generate_banded_jacobian_assembly_from_plan_parallel(
        &self,
        plan: &BandedStructurePlan,
        config: &BandedLambdifyConfig,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<BandedAssembly, BandedError> + Send + Sync>, BandedError>
    {
        let n = plan.n_unknowns;
        let kl = plan.scalar_bandwidth.0;
        let ku = plan.scalar_bandwidth.1;
        let threshold = config.structural_threshold.abs();

        enum CompiledPlan {
            Diagonal(Vec<CompiledBandedDiagonal>),
            Entry(Vec<CompiledBandedEntry>),
        }

        let compiled = match config.jacobian_chunking {
            BandedJacobianChunking::Diagonal => {
                CompiledPlan::Diagonal(self.compile_banded_diagonal_plan(&plan)?)
            }
            BandedJacobianChunking::EntryChunks => {
                CompiledPlan::Entry(self.compile_banded_entry_plan(&plan)?)
            }
        };

        Ok(Box::new(
            move |unknowns: &[f64]| -> Result<BandedAssembly, BandedError> {
                if unknowns.len() != n {
                    return Err(BandedError::DimensionMismatch);
                }

                let mut asm = BandedAssembly::zeros(n, kl, ku)?;
                match &compiled {
                    CompiledPlan::Diagonal(compiled_diagonals) => {
                        asm.diagonals_mut()
                            .par_iter_mut()
                            .zip(compiled_diagonals.par_iter())
                            .for_each(|(diag_values, compiled_diag)| {
                                for (slot, maybe_eval) in
                                    diag_values.iter_mut().zip(compiled_diag.evaluators.iter())
                                {
                                    let value = maybe_eval
                                        .as_ref()
                                        .map(|eval| eval(unknowns))
                                        .unwrap_or(0.0);
                                    *slot = if value.abs() < threshold { 0.0 } else { value };
                                }
                            });
                    }
                    CompiledPlan::Entry(compiled_entries) => {
                        let values: Vec<(usize, usize, f64)> = compiled_entries
                            .par_iter()
                            .map(|entry| (entry.row, entry.col, (entry.evaluator)(unknowns)))
                            .collect();
                        for (row, col, mut value) in values {
                            if value.abs() < threshold {
                                value = 0.0;
                            }
                            asm.set(row, col, value)?;
                        }
                    }
                }

                Ok(asm)
            },
        ))
    }

    pub fn generate_banded_jacobian_assembly_parallel(
        &self,
        config: &BandedLambdifyConfig,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<BandedAssembly, BandedError> + Send + Sync>, BandedError>
    {
        let plan = self.infer_banded_structure_plan()?;
        self.generate_banded_jacobian_assembly_from_plan_parallel(&plan, config)
    }

    /// Returns a parallel residual evaluator for the banded lambdify branch.
    ///
    /// This keeps the residual side lightweight:
    /// - compile phase builds one closure per residual equation,
    /// - runtime just evaluates them in parallel into a plain `Vec<f64>`.
    pub fn generate_banded_residual_parallel(
        &self,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, BandedError> + Send + Sync>, BandedError>
    {
        let n_unknowns = self.vector_of_variables.len();
        let parameter_map = self.parameter_substitution_map();
        let argument_names = self.runtime_argument_names(&parameter_map)?;
        let argument_name_refs: Vec<&str> =
            argument_names.iter().map(|name| name.as_str()).collect();

        let compiled_residuals: Vec<BandedScalarEvaluator> = self
            .vector_of_functions
            .par_iter()
            .map(|expr| {
                if let Some(ref map) = parameter_map {
                    let prepared_expr = expr.set_variable_from_map(map);
                    compile_banded_scalar_evaluator(&prepared_expr, &argument_name_refs)
                } else {
                    compile_banded_scalar_evaluator(expr, &argument_name_refs)
                }
            })
            .collect();

        Ok(Box::new(
            move |unknowns: &[f64]| -> Result<Vec<f64>, BandedError> {
                if unknowns.len() != n_unknowns {
                    return Err(BandedError::DimensionMismatch);
                }

                Ok(compiled_residuals
                    .par_iter()
                    .map(|eval| eval(unknowns))
                    .collect())
            },
        ))
    }

    /// Builds a parameter substitution map when the current symbolic problem
    /// already has concrete parameter values.
    ///
    /// This enables a cheap but important performance optimization:
    /// if parameters are fixed for the whole Newton solve, we substitute them
    /// once in symbolic form and keep the runtime closures purely state-based.
    fn parameter_substitution_map(&self) -> Option<HashMap<String, f64>> {
        match (
            self.parameters_string.is_empty(),
            self.parameter_values.as_ref(),
        ) {
            (true, _) => None,
            (false, Some(values)) if values.len() == self.parameters_string.len() => Some(
                self.parameters_string
                    .iter()
                    .cloned()
                    .zip(values.iter().copied())
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Determines the runtime argument order expected by banded compiled closures.
    ///
    /// Fast path:
    /// - if parameters were substituted away, use only the unknown vector.
    ///
    /// Conservative path:
    /// - if parameter names exist but values are missing, reject for now rather
    ///   than silently compiling a slower/ambiguous calling convention.
    fn runtime_argument_names(
        &self,
        parameter_map: &Option<HashMap<String, f64>>,
    ) -> Result<Vec<String>, BandedError> {
        if parameter_map.is_some() || self.parameters_string.is_empty() {
            return Ok(self.variable_string.clone());
        }

        Err(BandedError::DimensionMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::{BandedJacobianChunking, BandedLambdifyConfig};
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::Jacobian;

    #[test]
    fn infers_node_major_banded_structure() {
        let mut jac = Jacobian::new();
        jac.vector_of_functions = vec![Expr::Const(0.0); 4];
        jac.vector_of_variables = vec![
            Expr::Var("y_0".to_string()),
            Expr::Var("z_0".to_string()),
            Expr::Var("y_1".to_string()),
            Expr::Var("z_1".to_string()),
        ];
        jac.variable_string = vec![
            "y_0".to_string(),
            "z_0".to_string(),
            "y_1".to_string(),
            "z_1".to_string(),
        ];
        jac.bandwidth = Some((3, 3));

        let plan = jac.infer_banded_structure_plan().unwrap();
        assert_eq!(plan.n_nodes(), 2);
        assert_eq!(plan.vars_per_node(), 2);
        assert!(plan.block_tridiagonal_compatible);
    }

    #[test]
    fn generates_banded_jacobian_assembly_from_sparse_symbolic_entries() {
        let mut jac = Jacobian::new();
        jac.vector_of_functions = vec![Expr::Const(0.0); 4];
        jac.vector_of_variables = vec![
            Expr::Var("y_0".to_string()),
            Expr::Var("z_0".to_string()),
            Expr::Var("y_1".to_string()),
            Expr::Var("z_1".to_string()),
        ];
        jac.variable_string = vec![
            "y_0".to_string(),
            "z_0".to_string(),
            "y_1".to_string(),
            "z_1".to_string(),
        ];
        jac.bandwidth = Some((3, 3));
        jac.symbolic_jacobian_sparse = vec![
            (0, 0, Expr::Const(2.0) * Expr::Var("y_0".to_string())),
            (0, 1, Expr::Const(1.0)),
            (1, 0, Expr::Const(-1.0)),
            (1, 1, Expr::Const(3.0) * Expr::Var("z_0".to_string())),
            (2, 2, Expr::Const(4.0) * Expr::Var("y_1".to_string())),
            (2, 3, Expr::Const(2.0)),
            (3, 2, Expr::Const(-2.0)),
            (3, 3, Expr::Const(5.0) * Expr::Var("z_1".to_string())),
        ];

        let generator = jac
            .generate_banded_jacobian_assembly_parallel(&BandedLambdifyConfig {
                jacobian_chunking: BandedJacobianChunking::Diagonal,
                ..BandedLambdifyConfig::default()
            })
            .unwrap();

        let asm = generator(&[2.0, 3.0, 4.0, 5.0]).unwrap();

        assert_eq!(asm.get(0, 0).unwrap(), 4.0);
        assert_eq!(asm.get(0, 1).unwrap(), 1.0);
        assert_eq!(asm.get(1, 0).unwrap(), -1.0);
        assert_eq!(asm.get(1, 1).unwrap(), 9.0);
        assert_eq!(asm.get(2, 2).unwrap(), 16.0);
        assert_eq!(asm.get(3, 3).unwrap(), 25.0);
    }

    #[test]
    fn substitutes_fixed_parameters_before_compilation() {
        let mut jac = Jacobian::new();
        jac.vector_of_functions =
            vec![Expr::Var("a".to_string()) * Expr::Var("y_0".to_string()) + Expr::Const(1.0)];
        jac.vector_of_variables = vec![Expr::Var("y_0".to_string())];
        jac.variable_string = vec!["y_0".to_string()];
        jac.bandwidth = Some((0, 0));
        jac.symbolic_jacobian_sparse = vec![(0, 0, Expr::Var("a".to_string()))];
        jac.set_params(Some(&["a"]));
        jac.set_param_values(Some(vec![7.0]));

        let residual = jac.generate_banded_residual_parallel().unwrap();
        let jacobian = jac
            .generate_banded_jacobian_assembly_parallel(&BandedLambdifyConfig::default())
            .unwrap();

        let r = residual(&[3.0]).unwrap();
        let j = jacobian(&[3.0]).unwrap();

        assert_eq!(r[0], 22.0);
        assert_eq!(j.get(0, 0).unwrap(), 7.0);
    }

    #[test]
    fn banded_chunking_modes_produce_identical_assembly_values() {
        let mut jac = Jacobian::new();
        jac.vector_of_functions = vec![Expr::Const(0.0); 4];
        jac.vector_of_variables = vec![
            Expr::Var("y_0".to_string()),
            Expr::Var("z_0".to_string()),
            Expr::Var("y_1".to_string()),
            Expr::Var("z_1".to_string()),
        ];
        jac.variable_string = vec![
            "y_0".to_string(),
            "z_0".to_string(),
            "y_1".to_string(),
            "z_1".to_string(),
        ];
        jac.bandwidth = Some((3, 3));
        jac.symbolic_jacobian_sparse = vec![
            (0, 0, Expr::Const(2.0) * Expr::Var("y_0".to_string())),
            (0, 1, Expr::Const(1.0)),
            (1, 0, Expr::Const(-1.0)),
            (1, 1, Expr::Const(3.0) * Expr::Var("z_0".to_string())),
            (2, 2, Expr::Const(4.0) * Expr::Var("y_1".to_string())),
            (2, 3, Expr::Const(2.0)),
            (3, 2, Expr::Const(-2.0)),
            (3, 3, Expr::Const(5.0) * Expr::Var("z_1".to_string())),
        ];

        let diagonal = jac
            .generate_banded_jacobian_assembly_parallel(&BandedLambdifyConfig {
                jacobian_chunking: BandedJacobianChunking::Diagonal,
                ..BandedLambdifyConfig::default()
            })
            .unwrap();
        let entry_chunks = jac
            .generate_banded_jacobian_assembly_parallel(&BandedLambdifyConfig {
                jacobian_chunking: BandedJacobianChunking::EntryChunks,
                ..BandedLambdifyConfig::default()
            })
            .unwrap();

        let x = [2.0, 3.0, 4.0, 5.0];
        let d = diagonal(&x).unwrap();
        let e = entry_chunks(&x).unwrap();

        for row in 0..4 {
            for col in 0..4 {
                let dv = d.get(row, col).unwrap();
                let ev = e.get(row, col).unwrap();
                assert!(
                    (dv - ev).abs() <= 1e-12,
                    "chunking mismatch at ({row},{col}): diagonal={dv}, entry={ev}"
                );
            }
        }
    }

    #[test]
    fn banded_structural_threshold_zeros_tiny_entries() {
        let mut jac = Jacobian::new();
        jac.vector_of_functions = vec![Expr::Const(0.0); 2];
        jac.vector_of_variables = vec![Expr::Var("y_0".to_string()), Expr::Var("z_0".to_string())];
        jac.variable_string = vec!["y_0".to_string(), "z_0".to_string()];
        jac.bandwidth = Some((1, 1));
        jac.symbolic_jacobian_sparse = vec![(0, 0, Expr::Const(1e-14)), (1, 1, Expr::Const(2.0))];

        let generator = jac
            .generate_banded_jacobian_assembly_parallel(&BandedLambdifyConfig {
                jacobian_chunking: BandedJacobianChunking::EntryChunks,
                structural_threshold: 1e-12,
                ..BandedLambdifyConfig::default()
            })
            .unwrap();

        let asm = generator(&[1.0, 1.0]).unwrap();
        assert_eq!(asm.get(0, 0).unwrap(), 0.0);
        assert_eq!(asm.get(1, 1).unwrap(), 2.0);
    }
}
