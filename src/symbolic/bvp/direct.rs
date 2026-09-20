//! Direct, no-Mutex BVP symbolic runtime.
//!
//! This module owns the direct Banded lambdify helpers. Its runtime callback
//! evaluates disjoint diagonal/entry output regions and therefore does not use
//! the legacy per-entry `Mutex` assembly. When AtomView preparation is
//! available, residuals and sparse derivatives are compiled directly from
//! packed `Atom` values; the old public
//! `symbolic_functions_BVP2` name is only a compatibility facade.
//!
//! Current scope of this module:
//! - infer node-major block layout from the discretized BVP system,
//! - compile residual/Jacobian evaluators for the banded lambdify path,
//! - evaluate Jacobians directly into native `BandedAssembly`,
//! - expose chunking/threshold controls used by the runtime callbacks.

use crate::somelinalg::banded::{
    BandedError, LinearSolverConfig, NodeMajorLayout, banded_assembly::BandedAssembly,
};
use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::evaluate::{FunctionMap, PreparedVariableContext};
use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
use crate::symbolic::View::lambdify::lambdify_with_context;
use crate::symbolic::View::state::Symbol;
use crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle;
use crate::symbolic::bvp::telemetry::{
    BvpDirectJacobianTelemetry, BvpDirectJacobianTelemetrySnapshot, BvpLambdifyExecutionPolicy,
    BvpLambdifyTelemetryMode, flatten_lambdify_args,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::wrap_symbol;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

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
    /// Runtime policy for independent residual/Jacobian evaluator work.
    pub execution_policy: BvpLambdifyExecutionPolicy,
    /// Parallel work decomposition used by symbolic lambdify generators.
    pub jacobian_chunking: BandedJacobianChunking,
    /// Linear solver backend used after conversion to a solver-oriented format.
    pub linear_solver_backend: BandedLinearSolverBackend,
    /// Native solver policy/fallback choices reused from the banded linalg layer.
    pub linear_solver_config: LinearSolverConfig,
    /// Threshold below which generated Jacobian entries are treated as zero.
    pub structural_threshold: f64,
    /// Collection mode for direct Jacobian callback telemetry.
    ///
    /// Low-level direct constructors default to `Detailed` for diagnostics;
    /// solver-owned production callbacks override this with the selected
    /// Lambdify mode, normally `Off`.
    pub telemetry_mode: BvpLambdifyTelemetryMode,
}

impl Default for BandedLambdifyConfig {
    fn default() -> Self {
        Self {
            execution_policy: BvpLambdifyExecutionPolicy::default(),
            jacobian_chunking: BandedJacobianChunking::Diagonal,
            linear_solver_backend: BandedLinearSolverBackend::BlockTridiagonalNative,
            linear_solver_config: LinearSolverConfig::default(),
            structural_threshold: 0.0,
            telemetry_mode: BvpLambdifyTelemetryMode::Detailed,
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

/// Owned symbolic input for the direct Banded runtime.
///
/// The direct path owns this prepared snapshot instead of borrowing or
/// extending the legacy `Jacobian` host. The `Expr` fields are retained for
/// compatibility and standalone callers; AtomView callers attach packed
/// residual/Jacobian payloads with [`DirectBandedProblem::with_atom_view_data`]
/// so callback compilation does not use the compatibility cache.
#[derive(Clone, Default)]
pub struct DirectBandedProblem {
    /// Discretized residual expressions, used by the direct residual callback.
    pub vector_of_functions: Vec<Expr>,
    /// Discretized Newton unknown expressions.
    pub vector_of_variables: Vec<Expr>,
    /// Runtime argument names in unknown-vector order.
    pub variable_string: Vec<String>,
    /// Optional symbolic parameters that may be substituted before compiling.
    pub parameters_string: Vec<String>,
    /// Fixed parameter values, when the prepared problem is fully bound.
    pub parameter_values: Option<Vec<f64>>,
    /// Shared numeric binding used by prepared callbacks.
    ///
    /// When present, parameter values are read at callback time. This keeps
    /// symbolic compilation and numeric rebinding separate without changing
    /// the historical standalone constructor semantics.
    parameter_binding: Option<BvpParameterBindingHandle>,
    /// Sparse symbolic Jacobian entries `(row, column, expression)`.
    pub symbolic_jacobian_sparse: Vec<(usize, usize, Expr)>,
    /// AtomView residuals retained for native Banded Lambdify.
    pub atom_vector_of_functions: Option<Vec<Atom>>,
    /// AtomView sparse derivatives retained for native Banded Lambdify.
    pub atom_symbolic_jacobian_sparse: Option<Vec<SparseAtomJacobianEntry>>,
    /// Prepared scalar lower/upper bandwidth.
    pub bandwidth: Option<(usize, usize)>,
}

impl fmt::Debug for DirectBandedProblem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectBandedProblem")
            .field("vector_of_functions_len", &self.vector_of_functions.len())
            .field("vector_of_variables_len", &self.vector_of_variables.len())
            .field("variable_string", &self.variable_string)
            .field("parameters_string", &self.parameters_string)
            .field("parameter_values", &self.parameter_values)
            .field(
                "symbolic_jacobian_sparse_len",
                &self.symbolic_jacobian_sparse.len(),
            )
            .field(
                "atom_vector_of_functions_len",
                &self.atom_vector_of_functions.as_ref().map(Vec::len),
            )
            .field(
                "atom_symbolic_jacobian_sparse_len",
                &self.atom_symbolic_jacobian_sparse.as_ref().map(Vec::len),
            )
            .field("bandwidth", &self.bandwidth)
            .finish()
    }
}

impl DirectBandedProblem {
    /// Creates an owned direct-path problem from already prepared symbolic data.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vector_of_functions: Vec<Expr>,
        vector_of_variables: Vec<Expr>,
        variable_string: Vec<String>,
        parameters_string: Vec<String>,
        parameter_values: Option<Vec<f64>>,
        symbolic_jacobian_sparse: Vec<(usize, usize, Expr)>,
        bandwidth: Option<(usize, usize)>,
    ) -> Self {
        Self {
            vector_of_functions,
            vector_of_variables,
            variable_string,
            parameters_string,
            parameter_values,
            parameter_binding: None,
            symbolic_jacobian_sparse,
            atom_vector_of_functions: None,
            atom_symbolic_jacobian_sparse: None,
            bandwidth,
        }
    }

    /// Attaches already-prepared AtomView data without changing the legacy
    /// constructor or forcing callers through `Atom -> Expr` conversion.
    pub fn with_atom_view_data(
        mut self,
        residuals: Vec<Atom>,
        jacobian: Vec<SparseAtomJacobianEntry>,
    ) -> Self {
        self.atom_vector_of_functions = Some(residuals);
        self.atom_symbolic_jacobian_sparse = Some(jacobian);
        self
    }

    /// Attaches the prepared numeric binding used by solver-owned callbacks.
    pub(crate) fn with_parameter_binding(mut self, binding: BvpParameterBindingHandle) -> Self {
        self.parameter_binding = Some(binding);
        self
    }

    #[inline]
    fn uses_dynamic_parameter_binding(&self) -> bool {
        self.parameter_binding.is_some()
    }

    #[inline]
    fn parameter_values_for_callback(&self) -> Option<Arc<[f64]>> {
        self.parameter_binding
            .as_ref()
            .and_then(BvpParameterBindingHandle::snapshot)
            .or_else(|| self.parameter_values.clone().map(Arc::from))
    }
}

type BandedScalarEvaluator = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

#[inline]
fn validate_callback_value(
    stage: &'static str,
    row: usize,
    col: usize,
    value: f64,
) -> Result<f64, BandedError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BandedError::NonFiniteCallbackValue {
            stage,
            row,
            col,
            value,
        })
    }
}

/// Direct BVP Jacobian runtime and its lock-free telemetry stream.
///
/// The callback itself remains available through `callback()` for compatibility,
/// while new code can retain this handle and read `telemetry_snapshot()` after
/// a solve without reconstructing metrics from log strings.
pub struct DirectBandedJacobianRuntime {
    callback: Box<dyn Fn(&[f64]) -> Result<BandedAssembly, BandedError> + Send + Sync>,
    telemetry: BvpDirectJacobianTelemetry,
}

impl DirectBandedJacobianRuntime {
    /// Borrows the immutable no-Mutex callback.
    pub fn callback(
        &self,
    ) -> &(dyn Fn(&[f64]) -> Result<BandedAssembly, BandedError> + Send + Sync) {
        &*self.callback
    }

    /// Clones the telemetry handle for solver-level aggregation.
    pub fn telemetry(&self) -> BvpDirectJacobianTelemetry {
        self.telemetry.clone()
    }

    /// Reads callback count, work and elapsed time without taking a lock.
    pub fn telemetry_snapshot(&self) -> BvpDirectJacobianTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Consumes the wrapper and returns the historical callback shape.
    pub fn into_callback(
        self,
    ) -> Box<dyn Fn(&[f64]) -> Result<BandedAssembly, BandedError> + Send + Sync> {
        self.callback
    }
}

/// One compiled scalar diagonal of the banded Jacobian.
///
/// Performance-wise this is the key data layout for the future banded lambdify
/// mainline:
/// - compile phase groups scalar nonzeros by diagonal,
/// - each diagonal stores only `(position, evaluator)` pairs for structurally
///   present entries,
/// - runtime evaluation becomes a straight linear pass without lookups,
///   hashing, structural-zero branches, or sparse triplet assembly.
struct CompiledBandedDiagonal {
    /// Only structurally present entries are compiled. The surrounding
    /// `BandedAssembly` is already zero-initialized, so visiting structural
    /// zero slots in every callback would only add branches and writes.
    evaluators: Vec<(usize, BandedScalarEvaluator)>,
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

fn atom_variable_context(argument_names: &[String]) -> Arc<PreparedVariableContext> {
    let symbols = argument_names
        .iter()
        .map(|name| Symbol::new(wrap_symbol!(name.as_str())))
        .collect::<Vec<_>>();
    Arc::new(PreparedVariableContext::new(symbols.as_slice()))
}

fn compile_atom_scalar_evaluator(
    atom: &Atom,
    context: &PreparedVariableContext,
    function_map: &FunctionMap,
) -> BandedScalarEvaluator {
    lambdify_with_context(atom, context, function_map)
}

impl DirectBandedProblem {
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
        let n_unknowns = self
            .atom_vector_of_functions
            .as_ref()
            .and_then(|_| self.atom_symbolic_jacobian_sparse.as_ref())
            .map(|_| self.variable_string.len())
            .unwrap_or(self.vector_of_variables.len());
        let n_equations = self
            .atom_vector_of_functions
            .as_ref()
            .map(Vec::len)
            .unwrap_or(self.vector_of_functions.len());

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
        let atom_native = self.atom_symbolic_jacobian_sparse.is_some();
        let parameter_map = (!atom_native && !self.uses_dynamic_parameter_binding())
            .then(|| self.parameter_substitution_map())
            .flatten();
        let argument_names = if atom_native {
            self.atom_runtime_argument_names()?
        } else if self.uses_dynamic_parameter_binding() {
            self.flattened_runtime_argument_names()
        } else {
            self.runtime_argument_names(&parameter_map)?
        };
        let atom_context = atom_native.then(|| atom_variable_context(&argument_names));
        let atom_function_map = atom_native.then(FunctionMap::new);
        let argument_name_refs: Vec<&str> =
            argument_names.iter().map(|name| name.as_str()).collect();

        let mut diagonals: Vec<CompiledBandedDiagonal> =
            Vec::with_capacity(plan.scalar_bandwidth.0 + plan.scalar_bandwidth.1 + 1);
        for _offset in -(plan.scalar_bandwidth.0 as isize)..=(plan.scalar_bandwidth.1 as isize) {
            diagonals.push(CompiledBandedDiagonal {
                evaluators: Vec::new(),
            });
        }

        // Lambdification dominates setup for large meshes. Compile independent
        // nonzeros in parallel, then scatter closures into diagonal storage in
        // source order so runtime layout remains deterministic.
        let compiled_entries: Vec<(usize, usize, BandedScalarEvaluator)> =
            if let Some(entries) = self.atom_symbolic_jacobian_sparse.as_ref() {
                entries
                    .par_iter()
                    .filter_map(|entry| {
                        let offset = entry.col as isize - entry.row as isize;
                        if offset < -(plan.scalar_bandwidth.0 as isize)
                            || offset > plan.scalar_bandwidth.1 as isize
                        {
                            return None;
                        }
                        let diag_index = (offset + plan.scalar_bandwidth.0 as isize) as usize;
                        let pos = if offset >= 0 { entry.row } else { entry.col };
                        Some((
                            diag_index,
                            pos,
                            compile_atom_scalar_evaluator(
                                &entry.value,
                                atom_context.as_ref().expect("Atom context must exist"),
                                atom_function_map
                                    .as_ref()
                                    .expect("Atom function map must exist"),
                            ),
                        ))
                    })
                    .collect()
            } else {
                self.symbolic_jacobian_sparse
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
                    .collect()
            };

        for (diag_index, pos, evaluator) in compiled_entries {
            diagonals[diag_index].evaluators.push((pos, evaluator));
        }

        Ok(diagonals)
    }

    /// Builds an entry-wise compile plan used by `EntryChunks` runtime mode.
    fn compile_banded_entry_plan(
        &self,
        plan: &BandedStructurePlan,
    ) -> Result<Vec<CompiledBandedEntry>, BandedError> {
        let atom_native = self.atom_symbolic_jacobian_sparse.is_some();
        let parameter_map = (!atom_native && !self.uses_dynamic_parameter_binding())
            .then(|| self.parameter_substitution_map())
            .flatten();
        let argument_names = if atom_native {
            self.atom_runtime_argument_names()?
        } else if self.uses_dynamic_parameter_binding() {
            self.flattened_runtime_argument_names()
        } else {
            self.runtime_argument_names(&parameter_map)?
        };
        let atom_context = atom_native.then(|| atom_variable_context(&argument_names));
        let atom_function_map = atom_native.then(FunctionMap::new);
        let argument_name_refs: Vec<&str> =
            argument_names.iter().map(|name| name.as_str()).collect();
        if let Some(entries) = self.atom_symbolic_jacobian_sparse.as_ref() {
            return Ok(entries
                .par_iter()
                .filter_map(|entry| {
                    let offset = entry.col as isize - entry.row as isize;
                    if offset < -(plan.scalar_bandwidth.0 as isize)
                        || offset > plan.scalar_bandwidth.1 as isize
                    {
                        return None;
                    }
                    Some(CompiledBandedEntry {
                        row: entry.row,
                        col: entry.col,
                        evaluator: compile_atom_scalar_evaluator(
                            &entry.value,
                            atom_context.as_ref().expect("Atom context must exist"),
                            atom_function_map
                                .as_ref()
                                .expect("Atom function map must exist"),
                        ),
                    })
                })
                .collect());
        }

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
        Ok(self
            .generate_banded_jacobian_runtime_from_plan_parallel(plan, config)?
            .into_callback())
    }

    /// Builds the direct Banded callback together with its typed telemetry.
    ///
    /// This is the preferred entry point for new solver code.  The callback
    /// uses immutable compiled evaluators and disjoint output ownership; no
    /// per-entry `Mutex` is used.  The compatibility method above intentionally
    /// drops the telemetry handle for callers that still expect only a closure.
    pub fn generate_banded_jacobian_runtime_from_plan_parallel(
        &self,
        plan: &BandedStructurePlan,
        config: &BandedLambdifyConfig,
    ) -> Result<DirectBandedJacobianRuntime, BandedError> {
        let n = plan.n_unknowns;
        let kl = plan.scalar_bandwidth.0;
        let ku = plan.scalar_bandwidth.1;
        let threshold = config.structural_threshold.abs();
        let atom_native = self.atom_symbolic_jacobian_sparse.is_some();
        let parameter_binding = self.parameter_binding.clone();
        let parameter_values = self.parameter_values_for_callback();
        let parameter_count = self.parameters_string.len();
        let telemetry = match config.telemetry_mode {
            BvpLambdifyTelemetryMode::Off => BvpDirectJacobianTelemetry::disabled(),
            BvpLambdifyTelemetryMode::Counters => BvpDirectJacobianTelemetry::counters(),
            BvpLambdifyTelemetryMode::Detailed => BvpDirectJacobianTelemetry::new(),
        };
        let telemetry_for_callback = telemetry.clone();
        let telemetry_enabled = telemetry.is_enabled();
        let telemetry_timing_enabled = telemetry.timing_enabled();
        let execution_policy = config.execution_policy;

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
        let work_items = match &compiled {
            CompiledPlan::Diagonal(diagonals) => diagonals
                .iter()
                .map(|diagonal| diagonal.evaluators.len())
                .sum(),
            CompiledPlan::Entry(entries) => entries.len(),
        };
        let storage_writes = match &compiled {
            CompiledPlan::Diagonal(diagonals) => diagonals
                .iter()
                .map(|diagonal| diagonal.evaluators.len())
                .sum(),
            CompiledPlan::Entry(entries) => entries.len(),
        };
        let parallel_tasks = match &compiled {
            CompiledPlan::Diagonal(diagonals) => diagonals
                .iter()
                .map(|diagonal| {
                    BvpLambdifyExecutionPolicy::auto_task_count(diagonal.evaluators.len())
                })
                .sum::<usize>(),
            CompiledPlan::Entry(entries) => entries.len(),
        };

        let callback = Box::new(
            move |unknowns: &[f64]| -> Result<BandedAssembly, BandedError> {
                let begin = telemetry_timing_enabled.then(Instant::now);
                if unknowns.len() != n {
                    if telemetry_enabled {
                        telemetry_for_callback.record_error();
                    }
                    return Err(BandedError::DimensionMismatch);
                }

                let argument_prepare_begin = telemetry_timing_enabled.then(Instant::now);
                let uses_parameter_abi = atom_native || parameter_binding.is_some();
                let needs_parameter_prefix = uses_parameter_abi && parameter_count > 0;
                let runtime_args: Cow<'_, [f64]> = if needs_parameter_prefix {
                    Cow::Owned(flatten_lambdify_args(
                        parameter_binding
                            .as_ref()
                            .and_then(BvpParameterBindingHandle::snapshot)
                            .as_deref()
                            .or(parameter_values.as_deref()),
                        unknowns,
                    ))
                } else {
                    Cow::Borrowed(unknowns)
                };
                let argument_prepare_elapsed = argument_prepare_begin
                    .map(|started| started.elapsed())
                    .unwrap_or_default();
                let parallel =
                    execution_policy.should_parallel_with_tasks(work_items, parallel_tasks);
                let result = (|| -> Result<BandedAssembly, BandedError> {
                    let assembly_alloc_begin = telemetry_timing_enabled.then(Instant::now);
                    let mut asm = BandedAssembly::zeros(n, kl, ku)?;
                    let assembly_alloc_elapsed = assembly_alloc_begin
                        .map(|started| started.elapsed())
                        .unwrap_or_default();
                    let evaluator_begin = telemetry_timing_enabled.then(Instant::now);
                    let mut storage_write_elapsed = std::time::Duration::ZERO;
                    match &compiled {
                        CompiledPlan::Diagonal(compiled_diagonals) => {
                            let evaluate_diagonal =
                                |(diag_index, (diag_values, compiled_diag)): (
                                    usize,
                                    (&mut Vec<f64>, &CompiledBandedDiagonal),
                                )| {
                                    let offset = diag_index as isize - kl as isize;
                                    for (pos, eval) in &compiled_diag.evaluators {
                                        let value = eval(runtime_args.as_ref());
                                        let (row, col) = if offset >= 0 {
                                            (*pos, offset as usize + *pos)
                                        } else {
                                            ((-offset) as usize + *pos, *pos)
                                        };
                                        let value =
                                            validate_callback_value("Jacobian", row, col, value)?;
                                        let slot = diag_values
                                            .get_mut(*pos)
                                            .ok_or(BandedError::DimensionMismatch)?;
                                        *slot = if value.abs() < threshold { 0.0 } else { value };
                                    }
                                    Ok(())
                                };
                            let split_long_diagonals =
                                parallel && compiled_diagonals.len() < rayon::current_num_threads();
                            if split_long_diagonals {
                                for (diag_index, (diag_values, compiled_diag)) in asm
                                    .diagonals_mut()
                                    .iter_mut()
                                    .zip(compiled_diagonals.iter())
                                    .enumerate()
                                {
                                    let offset = diag_index as isize - kl as isize;
                                    if compiled_diag.evaluators.len() > 1 {
                                        let values: Vec<(usize, f64)> = compiled_diag
                                            .evaluators
                                            .par_iter()
                                            .map(|(pos, eval)| {
                                                let value = eval(runtime_args.as_ref());
                                                let (row, col) = if offset >= 0 {
                                                    (*pos, offset as usize + *pos)
                                                } else {
                                                    ((-offset) as usize + *pos, *pos)
                                                };
                                                let value = validate_callback_value(
                                                    "Jacobian", row, col, value,
                                                )?;
                                                Ok::<(usize, f64), BandedError>((*pos, value))
                                            })
                                            .collect::<Result<Vec<_>, BandedError>>()?;
                                        for (pos, value) in values {
                                            let slot = diag_values
                                                .get_mut(pos)
                                                .ok_or(BandedError::DimensionMismatch)?;
                                            *slot =
                                                if value.abs() < threshold { 0.0 } else { value };
                                        }
                                    } else {
                                        evaluate_diagonal((
                                            diag_index,
                                            (diag_values, compiled_diag),
                                        ))?;
                                    }
                                }
                            } else if parallel {
                                asm.diagonals_mut()
                                    .par_iter_mut()
                                    .zip(compiled_diagonals.par_iter())
                                    .enumerate()
                                    .try_for_each(evaluate_diagonal)?;
                            } else {
                                asm.diagonals_mut()
                                    .iter_mut()
                                    .zip(compiled_diagonals.iter())
                                    .enumerate()
                                    .try_for_each(evaluate_diagonal)?;
                            }
                        }
                        CompiledPlan::Entry(compiled_entries) => {
                            let evaluate_entry = |entry: &CompiledBandedEntry| {
                                let value = validate_callback_value(
                                    "Jacobian",
                                    entry.row,
                                    entry.col,
                                    (entry.evaluator)(runtime_args.as_ref()),
                                )?;
                                Ok((entry.row, entry.col, value))
                            };
                            if parallel {
                                // Parallel evaluation first collects values so the final
                                // scatter remains single-owner and lock-free. The sequential
                                // branch below deliberately avoids this temporary allocation.
                                let values: Vec<(usize, usize, f64)> = compiled_entries
                                    .par_iter()
                                    .map(evaluate_entry)
                                    .collect::<Result<Vec<_>, BandedError>>()?;
                                let storage_write_begin =
                                    telemetry_timing_enabled.then(Instant::now);
                                for (row, col, mut value) in values {
                                    if value.abs() < threshold {
                                        value = 0.0;
                                    }
                                    asm.set(row, col, value)?;
                                }
                                storage_write_elapsed = storage_write_begin
                                    .map(|started| started.elapsed())
                                    .unwrap_or_default();
                            } else {
                                let storage_write_begin =
                                    telemetry_timing_enabled.then(Instant::now);
                                for entry in compiled_entries {
                                    let (row, col, mut value) = evaluate_entry(entry)?;
                                    if value.abs() < threshold {
                                        value = 0.0;
                                    }
                                    asm.set(row, col, value)?;
                                }
                                storage_write_elapsed = storage_write_begin
                                    .map(|started| started.elapsed())
                                    .unwrap_or_default();
                            }
                        }
                    }

                    if telemetry_enabled {
                        telemetry_for_callback.record_dispatch_with_tasks(
                            parallel,
                            matches!(&compiled, CompiledPlan::Diagonal(_)),
                            parallel_tasks,
                        );
                        telemetry_for_callback.record_stage_breakdown(
                            argument_prepare_elapsed,
                            evaluator_begin
                                .map(|started| started.elapsed())
                                .unwrap_or_default(),
                            storage_write_elapsed,
                            assembly_alloc_elapsed,
                            work_items,
                            storage_writes,
                        );
                        telemetry_for_callback.record_call(
                            begin.map(|started| started.elapsed()).unwrap_or_default(),
                            work_items,
                        );
                    }
                    Ok(asm)
                })();
                if telemetry_enabled && result.is_err() {
                    telemetry_for_callback.record_error();
                }
                result
            },
        );

        Ok(DirectBandedJacobianRuntime {
            callback,
            telemetry,
        })
    }

    pub fn generate_banded_jacobian_assembly_parallel(
        &self,
        config: &BandedLambdifyConfig,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<BandedAssembly, BandedError> + Send + Sync>, BandedError>
    {
        let plan = self.infer_banded_structure_plan()?;
        self.generate_banded_jacobian_assembly_from_plan_parallel(&plan, config)
    }

    /// Public direct-path constructor that preserves telemetry for callers.
    pub fn generate_banded_jacobian_runtime_parallel(
        &self,
        config: &BandedLambdifyConfig,
    ) -> Result<DirectBandedJacobianRuntime, BandedError> {
        let plan = self.infer_banded_structure_plan()?;
        self.generate_banded_jacobian_runtime_from_plan_parallel(&plan, config)
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
        self.generate_banded_residual_with_config(&BandedLambdifyConfig::default())
    }

    /// Returns a Banded residual evaluator with an explicit runtime policy.
    pub fn generate_banded_residual_with_config(
        &self,
        config: &BandedLambdifyConfig,
    ) -> Result<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, BandedError> + Send + Sync>, BandedError>
    {
        let atom_native = self.atom_vector_of_functions.is_some();
        let n_unknowns = if atom_native {
            self.variable_string.len()
        } else {
            self.vector_of_variables.len()
        };
        let parameter_map = (!atom_native && !self.uses_dynamic_parameter_binding())
            .then(|| self.parameter_substitution_map())
            .flatten();
        let argument_names = if atom_native {
            self.atom_runtime_argument_names()?
        } else if self.uses_dynamic_parameter_binding() {
            self.flattened_runtime_argument_names()
        } else {
            self.runtime_argument_names(&parameter_map)?
        };
        let atom_context = atom_native.then(|| atom_variable_context(&argument_names));
        let atom_function_map = atom_native.then(FunctionMap::new);
        let argument_name_refs: Vec<&str> =
            argument_names.iter().map(|name| name.as_str()).collect();

        let compiled_residuals: Vec<BandedScalarEvaluator> =
            if let Some(functions) = self.atom_vector_of_functions.as_ref() {
                functions
                    .par_iter()
                    .map(|atom| {
                        compile_atom_scalar_evaluator(
                            atom,
                            atom_context.as_ref().expect("Atom context must exist"),
                            atom_function_map
                                .as_ref()
                                .expect("Atom function map must exist"),
                        )
                    })
                    .collect()
            } else {
                self.vector_of_functions
                    .par_iter()
                    .map(|expr| {
                        if let Some(ref map) = parameter_map {
                            let prepared_expr = expr.set_variable_from_map(map);
                            compile_banded_scalar_evaluator(&prepared_expr, &argument_name_refs)
                        } else {
                            compile_banded_scalar_evaluator(expr, &argument_name_refs)
                        }
                    })
                    .collect()
            };
        let parameter_binding = self.parameter_binding.clone();
        let parameter_values = self.parameter_values_for_callback();
        let parameter_count = self.parameters_string.len();
        let execution_policy = config.execution_policy;

        Ok(Box::new(
            move |unknowns: &[f64]| -> Result<Vec<f64>, BandedError> {
                if unknowns.len() != n_unknowns {
                    return Err(BandedError::DimensionMismatch);
                }

                let uses_parameter_abi = atom_native || parameter_binding.is_some();
                let needs_parameter_prefix = uses_parameter_abi && parameter_count > 0;
                let runtime_args: Cow<'_, [f64]> = if needs_parameter_prefix {
                    Cow::Owned(flatten_lambdify_args(
                        parameter_binding
                            .as_ref()
                            .and_then(BvpParameterBindingHandle::snapshot)
                            .as_deref()
                            .or(parameter_values.as_deref()),
                        unknowns,
                    ))
                } else {
                    Cow::Borrowed(unknowns)
                };

                if execution_policy.should_parallel(compiled_residuals.len()) {
                    compiled_residuals
                        .par_iter()
                        .enumerate()
                        .map(|(row, eval)| {
                            validate_callback_value("residual", row, 0, eval(runtime_args.as_ref()))
                        })
                        .collect::<Result<Vec<_>, BandedError>>()
                } else {
                    compiled_residuals
                        .iter()
                        .enumerate()
                        .map(|(row, eval)| {
                            validate_callback_value("residual", row, 0, eval(runtime_args.as_ref()))
                        })
                        .collect::<Result<Vec<_>, BandedError>>()
                }
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

    /// Returns the full callback ABI for a prepared parameterized Expr path.
    ///
    /// Unlike the historical fixed-value path, parameters stay symbolic in
    /// the compiled closure and are supplied as `[parameters..., unknowns...]`
    /// on every callback invocation.
    fn flattened_runtime_argument_names(&self) -> Vec<String> {
        self.parameters_string
            .iter()
            .cloned()
            .chain(self.variable_string.iter().cloned())
            .collect()
    }

    /// Determines the full AtomView argument ABI: `[parameters..., unknowns...]`.
    ///
    /// Unlike the Expr compatibility path, Atom expressions are not rewritten
    /// by symbolic parameter substitution here. Concrete parameter values are
    /// therefore supplied once per callback through one shared argument slice,
    /// not once per compiled entry.
    fn atom_runtime_argument_names(&self) -> Result<Vec<String>, BandedError> {
        if self.parameters_string.is_empty() {
            return Ok(self.variable_string.clone());
        }
        if self.uses_dynamic_parameter_binding()
            || self
                .parameter_values
                .as_ref()
                .is_some_and(|values| values.len() == self.parameters_string.len())
        {
            return Ok(self
                .parameters_string
                .iter()
                .cloned()
                .chain(self.variable_string.iter().cloned())
                .collect());
        }
        Err(BandedError::DimensionMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::{BandedJacobianChunking, BandedLambdifyConfig, DirectBandedProblem};
    use crate::parse;
    use crate::somelinalg::banded::BandedError;
    use crate::symbolic::View::jacobian::PreparedSparseAtomSystem;
    use crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle;
    use crate::symbolic::bvp::telemetry::{
        BvpDirectJacobianTelemetrySnapshot, BvpLambdifyTelemetryMode,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn infers_node_major_banded_structure() {
        let mut jac = DirectBandedProblem::default();
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
        let mut jac = DirectBandedProblem::default();
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
    fn direct_banded_runtime_reports_typed_telemetry_without_changing_values() {
        let mut jac = DirectBandedProblem::default();
        jac.vector_of_functions = vec![Expr::Const(0.0); 2];
        jac.vector_of_variables = vec![Expr::Var("y_0".to_string()), Expr::Var("y_1".to_string())];
        jac.variable_string = vec!["y_0".to_string(), "y_1".to_string()];
        jac.bandwidth = Some((1, 1));
        jac.symbolic_jacobian_sparse = vec![
            (0, 0, Expr::Const(2.0) * Expr::Var("y_0".to_string())),
            (0, 1, Expr::Const(1.0)),
            (1, 0, Expr::Const(-1.0)),
            (1, 1, Expr::Const(3.0) * Expr::Var("y_1".to_string())),
        ];

        let runtime = jac
            .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig::default())
            .unwrap();
        let asm = runtime.callback()(&[2.0, 4.0]).unwrap();
        let snapshot = runtime.telemetry_snapshot();

        assert_eq!(asm.get(0, 0).unwrap(), 4.0);
        assert_eq!(asm.get(1, 1).unwrap(), 12.0);
        assert_eq!(snapshot.calls, 1);
        assert_eq!(snapshot.work_items, 4);
        assert_eq!(snapshot.errors, 0);
        assert!(snapshot.elapsed > std::time::Duration::ZERO);
        assert_eq!(snapshot.evaluator_calls, 4);
        assert_eq!(snapshot.storage_writes, 4);
        assert_eq!(snapshot.parallel_dispatches, 1);
        assert_eq!(snapshot.sequential_dispatches, 0);
        assert_eq!(snapshot.diagonal_dispatches, 1);
        assert_eq!(snapshot.entry_dispatches, 0);
        assert!(snapshot.assembly_alloc_elapsed >= std::time::Duration::ZERO);
        assert!(snapshot.argument_prepare_elapsed >= std::time::Duration::ZERO);
        assert!(snapshot.evaluator_elapsed > std::time::Duration::ZERO);

        assert!(runtime.callback()(&[2.0]).is_err());
        assert_eq!(runtime.telemetry_snapshot().errors, 1);
    }

    #[test]
    fn direct_banded_runtime_can_disable_callback_telemetry_without_changing_values() {
        let mut jac = DirectBandedProblem::default();
        jac.vector_of_functions = vec![Expr::Const(0.0)];
        jac.vector_of_variables = vec![Expr::Var("y_0".to_string())];
        jac.variable_string = vec!["y_0".to_string()];
        jac.bandwidth = Some((0, 0));
        jac.symbolic_jacobian_sparse = vec![(0, 0, Expr::Const(3.0))];

        let runtime = jac
            .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig {
                telemetry_mode: BvpLambdifyTelemetryMode::Off,
                ..BandedLambdifyConfig::default()
            })
            .unwrap();
        let assembly = runtime.callback()(&[2.0]).unwrap();

        assert_eq!(assembly.get(0, 0).unwrap(), 3.0);
        assert_eq!(
            runtime.telemetry_snapshot(),
            BvpDirectJacobianTelemetrySnapshot::default()
        );
    }

    #[test]
    fn diagonal_runtime_skips_structural_zero_slots() {
        let mut jac = DirectBandedProblem::default();
        jac.vector_of_functions = vec![Expr::Const(0.0); 4];
        jac.vector_of_variables = (0..4)
            .map(|index| Expr::Var(format!("y_{index}")))
            .collect();
        jac.variable_string = (0..4).map(|index| format!("y_{index}")).collect();
        jac.bandwidth = Some((3, 3));
        jac.symbolic_jacobian_sparse = vec![
            (0, 0, Expr::Const(2.0)),
            (0, 1, Expr::Const(3.0)),
            (1, 0, Expr::Const(4.0)),
            (1, 1, Expr::Const(5.0)),
            (2, 2, Expr::Const(6.0)),
            (2, 3, Expr::Const(7.0)),
            (3, 2, Expr::Const(8.0)),
            (3, 3, Expr::Const(9.0)),
        ];

        let runtime = jac
            .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig::default())
            .unwrap();
        let assembly = runtime.callback()(&[0.0; 4]).unwrap();
        let snapshot = runtime.telemetry_snapshot();

        assert_eq!(assembly.get(0, 0).unwrap(), 2.0);
        assert_eq!(assembly.get(0, 3).unwrap(), 0.0);
        assert_eq!(assembly.get(3, 0).unwrap(), 0.0);
        assert_eq!(snapshot.evaluator_calls, 8);
        assert_eq!(snapshot.storage_writes, 8);
    }

    #[test]
    fn direct_banded_residual_rejects_non_finite_callback_values() {
        let mut problem = DirectBandedProblem::default();
        problem.vector_of_functions = vec![Expr::Const(f64::NAN)];
        problem.vector_of_variables = vec![Expr::Var("y_0".to_string())];
        problem.variable_string = vec!["y_0".to_string()];

        let residual = problem.generate_banded_residual_parallel().unwrap();
        let error = residual(&[0.0]).expect_err("NaN must not cross the callback boundary");

        assert!(matches!(
            error,
            BandedError::NonFiniteCallbackValue {
                stage: "residual",
                row: 0,
                col: 0,
                value
            } if value.is_nan()
        ));
    }

    #[test]
    fn direct_banded_jacobian_rejects_non_finite_callback_values_before_thresholding() {
        let mut problem = DirectBandedProblem::default();
        problem.vector_of_functions = vec![Expr::Const(0.0)];
        problem.vector_of_variables = vec![Expr::Var("y_0".to_string())];
        problem.variable_string = vec!["y_0".to_string()];
        problem.bandwidth = Some((0, 0));
        problem.symbolic_jacobian_sparse = vec![(0, 0, Expr::Const(f64::INFINITY))];

        for chunking in [
            BandedJacobianChunking::Diagonal,
            BandedJacobianChunking::EntryChunks,
        ] {
            let jacobian = problem
                .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig {
                    jacobian_chunking: chunking,
                    ..BandedLambdifyConfig::default()
                })
                .unwrap();
            let error = jacobian.callback()(&[0.0])
                .expect_err("infinity must not be converted into a stored Jacobian value");

            assert!(matches!(
                error,
                BandedError::NonFiniteCallbackValue {
                    stage: "Jacobian",
                    row: 0,
                    col: 0,
                    value
                } if value.is_infinite()
            ));
            assert_eq!(jacobian.telemetry_snapshot().errors, 1);
        }
    }

    #[test]
    fn substitutes_fixed_parameters_before_compilation() {
        let mut jac = DirectBandedProblem::default();
        jac.vector_of_functions =
            vec![Expr::Var("a".to_string()) * Expr::Var("y_0".to_string()) + Expr::Const(1.0)];
        jac.vector_of_variables = vec![Expr::Var("y_0".to_string())];
        jac.variable_string = vec!["y_0".to_string()];
        jac.bandwidth = Some((0, 0));
        jac.symbolic_jacobian_sparse = vec![(0, 0, Expr::Var("a".to_string()))];
        jac.parameters_string = vec!["a".to_string()];
        jac.parameter_values = Some(vec![7.0]);

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
    fn atom_native_banded_runtime_uses_parameter_abi_without_expr_roundtrip() {
        let residuals = vec![parse!("a*y_0 + y_1").unwrap(), parse!("y_0 - y_1").unwrap()];
        let variable_names = vec!["y_0".to_string(), "y_1".to_string()];
        let atom_system = PreparedSparseAtomSystem::from_atoms(
            &residuals,
            &variable_names,
            &[variable_names.clone(), variable_names.clone()],
        );
        let atom_jacobian = atom_system.calc_sparse_jacobian_with_bandwidth(Some((1, 1)));

        let mut problem =
            DirectBandedProblem::default().with_atom_view_data(residuals, atom_jacobian);
        problem.vector_of_functions = vec![Expr::Const(0.0); 2];
        problem.vector_of_variables = variable_names
            .iter()
            .map(|name| Expr::Var(name.clone()))
            .collect();
        problem.variable_string = variable_names;
        problem.parameters_string = vec!["a".to_string()];
        problem.parameter_values = Some(vec![7.0]);
        problem.bandwidth = Some((1, 1));

        let residual = problem.generate_banded_residual_parallel().unwrap();
        let jacobian = problem
            .generate_banded_jacobian_assembly_parallel(&BandedLambdifyConfig::default())
            .unwrap();

        let x = [2.0, 3.0];
        let r = residual(&x).unwrap();
        let j = jacobian(&x).unwrap();

        assert_eq!(r, vec![17.0, -1.0]);
        assert_eq!(j.get(0, 0).unwrap(), 7.0);
        assert_eq!(j.get(0, 1).unwrap(), 1.0);
        assert_eq!(j.get(1, 0).unwrap(), 1.0);
        assert_eq!(j.get(1, 1).unwrap(), -1.0);
    }

    #[test]
    fn atom_native_unparameterized_runtime_preserves_state_values() {
        let residuals = vec![parse!("y_0 + y_1").unwrap(), parse!("y_0 - y_1").unwrap()];
        let variable_names = vec!["y_0".to_string(), "y_1".to_string()];
        let atom_system = PreparedSparseAtomSystem::from_atoms(
            &residuals,
            &variable_names,
            &[variable_names.clone(), variable_names.clone()],
        );
        let atom_jacobian = atom_system.calc_sparse_jacobian_with_bandwidth(Some((1, 1)));

        let mut problem =
            DirectBandedProblem::default().with_atom_view_data(residuals, atom_jacobian);
        problem.vector_of_functions = vec![Expr::Const(0.0); 2];
        problem.vector_of_variables = variable_names
            .iter()
            .map(|name| Expr::Var(name.clone()))
            .collect();
        problem.variable_string = variable_names;
        problem.bandwidth = Some((1, 1));

        let residual = problem.generate_banded_residual_parallel().unwrap();
        let jacobian = problem
            .generate_banded_jacobian_assembly_parallel(&BandedLambdifyConfig::default())
            .unwrap();

        assert_eq!(residual(&[2.0, 3.0]).unwrap(), vec![5.0, -1.0]);
        let assembly = jacobian(&[2.0, 3.0]).unwrap();
        assert_eq!(assembly.get(0, 0).unwrap(), 1.0);
        assert_eq!(assembly.get(0, 1).unwrap(), 1.0);
        assert_eq!(assembly.get(1, 0).unwrap(), 1.0);
        assert_eq!(assembly.get(1, 1).unwrap(), -1.0);
    }

    #[test]
    fn prepared_banded_runtime_rebinds_numeric_parameters_without_recompilation() {
        let mut problem = DirectBandedProblem::default();
        problem.vector_of_functions =
            vec![Expr::Var("a".to_string()) * Expr::Var("y_0".to_string())];
        problem.vector_of_variables = vec![Expr::Var("y_0".to_string())];
        problem.variable_string = vec!["y_0".to_string()];
        problem.parameters_string = vec!["a".to_string()];
        problem.parameter_values = Some(vec![2.0]);
        problem.bandwidth = Some((0, 0));
        problem.symbolic_jacobian_sparse = vec![(0, 0, Expr::Var("a".to_string()))];

        let binding = BvpParameterBindingHandle::new(Some(vec![2.0]));
        problem = problem.with_parameter_binding(binding.clone());
        let residual = problem.generate_banded_residual_parallel().unwrap();
        let jacobian = problem
            .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig::default())
            .unwrap();

        assert_eq!(residual(&[3.0]).unwrap(), vec![6.0]);
        assert_eq!(jacobian.callback()(&[3.0]).unwrap().get(0, 0).unwrap(), 2.0);

        binding.replace(Some(vec![5.0]));
        assert_eq!(residual(&[3.0]).unwrap(), vec![15.0]);
        assert_eq!(jacobian.callback()(&[3.0]).unwrap().get(0, 0).unwrap(), 5.0);
    }

    #[test]
    fn banded_chunking_modes_produce_identical_assembly_values() {
        let mut jac = DirectBandedProblem::default();
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

        let entry_runtime = jac
            .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig {
                jacobian_chunking: BandedJacobianChunking::EntryChunks,
                ..BandedLambdifyConfig::default()
            })
            .unwrap();
        entry_runtime.callback()(&x).unwrap();
        let snapshot = entry_runtime.telemetry_snapshot();
        assert_eq!(snapshot.entry_dispatches, 1);
        assert_eq!(snapshot.diagonal_dispatches, 0);
        assert_eq!(snapshot.evaluator_calls, 8);
        assert_eq!(snapshot.storage_writes, 8);
    }

    #[test]
    fn banded_sequential_and_parallel_runtime_policies_are_identical() {
        let mut problem = DirectBandedProblem::default();
        problem.vector_of_functions = vec![
            Expr::Var("y_0".to_string()) + Expr::Var("z_0".to_string()),
            Expr::Var("y_0".to_string()) - Expr::Var("z_0".to_string()),
        ];
        problem.vector_of_variables =
            vec![Expr::Var("y_0".to_string()), Expr::Var("z_0".to_string())];
        problem.variable_string = vec!["y_0".to_string(), "z_0".to_string()];
        problem.bandwidth = Some((1, 1));
        problem.symbolic_jacobian_sparse = vec![
            (0, 0, Expr::Const(1.0)),
            (0, 1, Expr::Const(1.0)),
            (1, 0, Expr::Const(1.0)),
            (1, 1, Expr::Const(-1.0)),
        ];

        let sequential_config = BandedLambdifyConfig {
            execution_policy:
                crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Sequential,
            ..BandedLambdifyConfig::default()
        };
        let parallel_config = BandedLambdifyConfig {
            execution_policy:
                crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Parallel {
                    min_work: 0,
                },
            ..BandedLambdifyConfig::default()
        };
        let threshold_config = BandedLambdifyConfig {
            execution_policy:
                crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Parallel {
                    min_work: 1,
                },
            ..BandedLambdifyConfig::default()
        };
        let above_threshold_config = BandedLambdifyConfig {
            execution_policy:
                crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Parallel {
                    // The policy counts scalar evaluator calls, not the
                    // number of diagonal containers.
                    min_work: 5,
                },
            ..BandedLambdifyConfig::default()
        };
        let auto_config = BandedLambdifyConfig {
            execution_policy: crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Auto {
                min_work: 0,
            },
            ..BandedLambdifyConfig::default()
        };
        let sequential_jacobian = problem
            .generate_banded_jacobian_runtime_parallel(&sequential_config)
            .unwrap();
        let parallel_jacobian = problem
            .generate_banded_jacobian_runtime_parallel(&parallel_config)
            .unwrap();
        let threshold_jacobian = problem
            .generate_banded_jacobian_runtime_parallel(&threshold_config)
            .unwrap();
        let above_threshold_jacobian = problem
            .generate_banded_jacobian_runtime_parallel(&above_threshold_config)
            .unwrap();
        let auto_jacobian = problem
            .generate_banded_jacobian_runtime_parallel(&auto_config)
            .unwrap();
        let sequential_residual = problem
            .generate_banded_residual_with_config(&sequential_config)
            .unwrap();
        let parallel_residual = problem
            .generate_banded_residual_with_config(&parallel_config)
            .unwrap();

        let unknowns = [2.0, 3.0];
        let seq_j = sequential_jacobian.callback()(&unknowns).unwrap();
        let par_j = parallel_jacobian.callback()(&unknowns).unwrap();
        threshold_jacobian.callback()(&unknowns).unwrap();
        above_threshold_jacobian.callback()(&unknowns).unwrap();
        let auto_j = auto_jacobian.callback()(&unknowns).unwrap();
        let seq_snapshot = sequential_jacobian.telemetry_snapshot();
        let par_snapshot = parallel_jacobian.telemetry_snapshot();
        let threshold_snapshot = threshold_jacobian.telemetry_snapshot();
        let above_threshold_snapshot = above_threshold_jacobian.telemetry_snapshot();
        let auto_snapshot = auto_jacobian.telemetry_snapshot();
        assert_eq!(seq_snapshot.sequential_dispatches, 1);
        assert_eq!(seq_snapshot.parallel_dispatches, 0);
        assert_eq!(par_snapshot.sequential_dispatches, 0);
        assert_eq!(par_snapshot.parallel_dispatches, 1);
        assert_eq!(threshold_snapshot.parallel_dispatches, 1);
        assert_eq!(above_threshold_snapshot.sequential_dispatches, 1);
        assert_eq!(
            auto_snapshot.parallel_dispatches + auto_snapshot.sequential_dispatches,
            1
        );
        assert_eq!(
            sequential_residual(&unknowns).unwrap(),
            parallel_residual(&unknowns).unwrap()
        );
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(seq_j.get(row, col).unwrap(), par_j.get(row, col).unwrap());
                assert_eq!(seq_j.get(row, col).unwrap(), auto_j.get(row, col).unwrap());
            }
        }
        assert_eq!(sequential_residual(&unknowns).unwrap(), vec![5.0, -1.0]);
    }

    #[test]
    fn auto_splits_a_long_single_diagonal_without_changing_values() {
        let n = 64;
        let mut problem = DirectBandedProblem::default();
        problem.vector_of_functions = vec![Expr::Const(0.0); n];
        problem.vector_of_variables = (0..n)
            .map(|index| Expr::Var(format!("y_{index}")))
            .collect();
        problem.variable_string = (0..n).map(|index| format!("y_{index}")).collect();
        problem.bandwidth = Some((0, 0));
        problem.symbolic_jacobian_sparse = (0..n)
            .map(|index| (index, index, Expr::Const(index as f64 + 1.0)))
            .collect();

        let sequential = problem
            .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig {
                execution_policy:
                    crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Sequential,
                ..BandedLambdifyConfig::default()
            })
            .unwrap();
        let auto = problem
            .generate_banded_jacobian_runtime_parallel(&BandedLambdifyConfig {
                execution_policy:
                    crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Auto {
                        min_work: 0,
                    },
                ..BandedLambdifyConfig::default()
            })
            .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let unknowns = vec![1.0; n];
        let (expected, actual) = pool.install(|| {
            (
                sequential.callback()(&unknowns).unwrap(),
                auto.callback()(&unknowns).unwrap(),
            )
        });

        for index in 0..n {
            assert_eq!(
                expected.get(index, index).unwrap(),
                actual.get(index, index).unwrap()
            );
        }
        let snapshot = auto.telemetry_snapshot();
        assert_eq!(snapshot.parallel_dispatches, 1);
        assert_eq!(snapshot.sequential_dispatches, 0);
        assert_eq!(snapshot.evaluator_calls, n as u64);
        assert_eq!(snapshot.effective_task_count, 8);
    }

    #[test]
    fn banded_structural_threshold_zeros_tiny_entries() {
        let mut jac = DirectBandedProblem::default();
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
