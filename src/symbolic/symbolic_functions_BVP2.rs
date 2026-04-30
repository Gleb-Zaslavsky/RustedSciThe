//! Banded-oriented symbolic helpers for the BVP lambdify pipeline.
//!
//! The main symbolic BVP module, [`crate::symbolic::symbolic_functions_BVP`],
//! already carries the production sparse mainline. This companion module keeps
//! the new banded-specific design and helper logic isolated so the original
//! file does not become even heavier.
//!
//! Current scope of this module:
//! - declare the architectural contract for the future banded lambdify path,
//! - infer a node-major block layout from the already discretized BVP system,
//! - provide storage/layout allocation helpers that later generators can reuse.
//!
//! Deliberately not implemented yet:
//! - symbolic lambdify generators returning `BandedAssembly`,
//! - `JacEnum::Banded` runtime hookup,
//! - direct solver integration inside `BVP_traits`.
//!
//! Those next steps will build on the typed helpers introduced here.

use crate::somelinalg::banded::{
    BandedError, LinearSolverConfig, NodeMajorLayout, banded_assembly::BandedAssembly,
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
    offset: isize,
    evaluators: Vec<Option<BandedScalarEvaluator>>,
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

fn compile_banded_scalar_evaluator(
    expr: &Expr,
    argument_names: &[String],
) -> BandedScalarEvaluator {
    Expr::lambdify_borrowed_thread_safe(
        expr,
        argument_names
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>()
            .as_slice(),
    )
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
        let mut seen = HashSet::new();
        let mut vars_per_node = 0usize;
        for name in &self.variable_string {
            let inserted = seen.insert(base_variable_name(name).to_string());
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
        let asm_template = BandedAssembly::zeros(
            plan.n_unknowns,
            plan.scalar_bandwidth.0,
            plan.scalar_bandwidth.1,
        )?;

        let mut diagonals: Vec<CompiledBandedDiagonal> =
            Vec::with_capacity(asm_template.num_diagonals());
        for offset in asm_template.offsets() {
            let len = asm_template
                .diag_len(offset)
                .ok_or(BandedError::DimensionMismatch)?;
            diagonals.push(CompiledBandedDiagonal {
                offset,
                evaluators: std::iter::repeat_with(|| None).take(len).collect(),
            });
        }

        for (row, col, expr) in &self.symbolic_jacobian_sparse {
            let offset = *col as isize - *row as isize;
            if offset < -(plan.scalar_bandwidth.0 as isize)
                || offset > plan.scalar_bandwidth.1 as isize
            {
                continue;
            }

            let diag_index = (offset + plan.scalar_bandwidth.0 as isize) as usize;
            let pos = if offset >= 0 { *row } else { *col };

            let prepared_expr = if let Some(ref map) = parameter_map {
                expr.set_variable_from_map(map)
            } else {
                expr.clone()
            };

            diagonals[diag_index].evaluators[pos] =
                Some(compile_banded_scalar_evaluator(&prepared_expr, &argument_names));
        }

        Ok(diagonals)
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
    pub fn generate_banded_jacobian_assembly_parallel(
        &self,
        _config: &BandedLambdifyConfig,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<BandedAssembly, BandedError> + Send + Sync>, BandedError>
    {
        let plan = self.infer_banded_structure_plan()?;
        let compiled_diagonals = self.compile_banded_diagonal_plan(&plan)?;
        let n = plan.n_unknowns;
        let kl = plan.scalar_bandwidth.0;
        let ku = plan.scalar_bandwidth.1;

        Ok(Box::new(move |unknowns: &[f64]| -> Result<BandedAssembly, BandedError> {
            if unknowns.len() != n {
                return Err(BandedError::DimensionMismatch);
            }

            let mut asm = BandedAssembly::zeros(n, kl, ku)?;

            asm.diagonals_mut()
                .par_iter_mut()
                .zip(compiled_diagonals.par_iter())
                .for_each(|(diag_values, compiled_diag)| {
                    let _offset = compiled_diag.offset;
                    for (slot, maybe_eval) in diag_values.iter_mut().zip(compiled_diag.evaluators.iter()) {
                        *slot = maybe_eval
                            .as_ref()
                            .map(|eval| eval(unknowns))
                            .unwrap_or(0.0);
                    }
                });

            Ok(asm)
        }))
    }

    /// Returns a parallel residual evaluator for the banded lambdify branch.
    ///
    /// This keeps the residual side lightweight:
    /// - compile phase builds one closure per residual equation,
    /// - runtime just evaluates them in parallel into a plain `Vec<f64>`.
    pub fn generate_banded_residual_parallel(
        &self,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, BandedError> + Send + Sync>, BandedError> {
        let n_unknowns = self.vector_of_variables.len();
        let parameter_map = self.parameter_substitution_map();
        let argument_names = self.runtime_argument_names(&parameter_map)?;

        let compiled_residuals: Vec<BandedScalarEvaluator> = self
            .vector_of_functions
            .iter()
            .map(|expr| {
                let prepared_expr = if let Some(ref map) = parameter_map {
                    expr.set_variable_from_map(map)
                } else {
                    expr.clone()
                };
                compile_banded_scalar_evaluator(&prepared_expr, &argument_names)
            })
            .collect();

        Ok(Box::new(move |unknowns: &[f64]| -> Result<Vec<f64>, BandedError> {
            if unknowns.len() != n_unknowns {
                return Err(BandedError::DimensionMismatch);
            }

            Ok(compiled_residuals
                .par_iter()
                .map(|eval| eval(unknowns))
                .collect())
        }))
    }

    /// Builds a parameter substitution map when the current symbolic problem
    /// already has concrete parameter values.
    ///
    /// This enables a cheap but important performance optimization:
    /// if parameters are fixed for the whole Newton solve, we substitute them
    /// once in symbolic form and keep the runtime closures purely state-based.
    fn parameter_substitution_map(&self) -> Option<HashMap<String, f64>> {
        match (self.parameters_string.is_empty(), self.parameter_values.as_ref()) {
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
            (
                0,
                0,
                Expr::Const(2.0) * Expr::Var("y_0".to_string()),
            ),
            (0, 1, Expr::Const(1.0)),
            (1, 0, Expr::Const(-1.0)),
            (
                1,
                1,
                Expr::Const(3.0) * Expr::Var("z_0".to_string()),
            ),
            (
                2,
                2,
                Expr::Const(4.0) * Expr::Var("y_1".to_string()),
            ),
            (2, 3, Expr::Const(2.0)),
            (3, 2, Expr::Const(-2.0)),
            (
                3,
                3,
                Expr::Const(5.0) * Expr::Var("z_1".to_string()),
            ),
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
        jac.vector_of_functions = vec![
            Expr::Var("a".to_string()) * Expr::Var("y_0".to_string()) + Expr::Const(1.0),
        ];
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
}
