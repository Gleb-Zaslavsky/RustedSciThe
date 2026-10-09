//! Low-overhead telemetry contract for preparation, collocation and backends.
//!
//! `Off` is the default.  It must not read clocks or inspect capacities.  The
//! counters below are logical events, not process-wide allocator statistics.

use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy;
use crate::symbolic::ivp_telemetry::IvpTelemetryContractError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpSciTelemetryMode {
    #[default]
    Off,
    Counters,
    Timings,
}

/// Stable identity for an aggregate timing scope.
///
/// The flat timing fields are retained for compatibility, while this enum
/// gives reports enough structure to avoid adding parent and child scopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvpSciTelemetryStage {
    Preparation,
    FullSolve,
    Newton,
    MeshDefectEstimation,
    OutputConstruction,
    ExprToAtom,
    SymbolicJacobian,
    Pattern,
    Lowering,
    EvaluatorCompilation,
    ResidualEvaluatorCompilation,
    JacobianEvaluatorCompilation,
    Binding,
    ResidualEvaluation,
    JacobianEvaluation,
    JacobianOutputAssembly,
    Callback,
    Collocation,
    MeshRefinement,
    LinearAssembly,
    BandedScalarFallbackAssembly,
    Factorization,
    SparseSymbolicAnalysis,
    SparseNumericFactorization,
    LinearSolve,
    BandedStructuredFactorization,
    BandedScalarFallbackFactorization,
    BandedStructuredSolve,
    BandedScalarFallbackSolve,
    BandedResidualGuard,
    BandedRhsPermutation,
}

/// Whether a timing includes nested work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciTelemetryScopeKind {
    Inclusive,
    Exclusive,
}

/// Actual numeric route selected by the collocation Banded adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciBandedRoute {
    Structured,
    ScalarFallback,
    SparseFallback,
}

/// Timing metadata used by compact reports and downstream aggregation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvpSciTelemetryScope {
    pub stage: BvpSciTelemetryStage,
    pub kind: BvpSciTelemetryScopeKind,
    pub parent: Option<BvpSciTelemetryStage>,
    /// Number of measured invocations represented by this aggregate scope.
    /// Zero means that the stage was not observed, not that it took zero time.
    pub calls: u64,
    pub elapsed_ms: Option<f64>,
}

/// One modified-Newton observation retained by detailed telemetry.
///
/// This is intentionally a compact summary rather than a callback-level event
/// stream. It is enough to distinguish a bad linear correction from a stale
/// Jacobian or exhausted backtracking without putting a lock or allocation in
/// the residual/Jacobian hot path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvpSciNewtonTraceEntry {
    pub iteration: u64,
    pub residual_before: f64,
    pub residual_after: Option<f64>,
    pub step_inf_norm: f64,
    /// SciPy affine-invariant criterion before the trial: `||J^-1 r||^2`.
    pub affine_cost_before: f64,
    /// The same criterion after the accepted/last trial, if finite.
    pub affine_cost_after: Option<f64>,
    /// Accepted trial scale (`1`, `0.5`, ...), or the last attempted scale.
    pub armijo_alpha: f64,
    pub backtracking_trials: u64,
    pub jacobian_refreshed: bool,
    pub accepted: bool,
}

/// Point-in-time report for one prepared model and its numerical lifecycle.
///
/// Counter fields are meaningful in both `Counters` and `Timings` modes.
/// Timing fields are `None` unless clock collection was explicitly enabled;
/// this makes disabled and inapplicable measurements distinguishable from a
/// measured zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BvpSciTelemetrySnapshot {
    pub mode: BvpSciTelemetryMode,
    /// Shared generated-IVP lifecycle/runtime snapshot for an AOT plan.
    /// Kept separate because its scopes use the IVP ABI and are not additive
    /// with BVP numerical scopes.
    pub aot: Option<crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot>,
    /// Callback and numerical event counters.
    pub rhs_calls: u64,
    pub boundary_calls: u64,
    pub jacobian_calls: u64,
    pub finite_difference_probes: u64,
    pub newton_iterations: u64,
    pub newton_jacobian_refreshes: u64,
    pub newton_backtracking_trials: u64,
    pub newton_accepted_steps: u64,
    pub newton_rejected_steps: u64,
    /// Detailed-only modified-Newton trace; empty in Off/Counters modes.
    pub newton_residual_history: Vec<BvpSciNewtonTraceEntry>,
    pub collocation_evaluations: u64,
    pub mesh_defect_probes: u64,
    pub mesh_refinements: u64,
    pub mesh_points_added: u64,
    pub linear_assemblies: u64,
    pub factorizations: u64,
    pub sparse_symbolic_analyses: u64,
    pub sparse_numeric_factorizations: u64,
    pub linear_solves: u64,
    pub banded_structured_factorizations: u64,
    pub banded_scalar_fallback_factorizations: u64,
    pub banded_scalar_fallback_assemblies: u64,
    pub banded_structured_solves: u64,
    pub banded_scalar_fallback_solves: u64,
    pub banded_sparse_fallback_factorizations: u64,
    pub banded_sparse_fallback_solves: u64,
    /// Assembly work retained for a possible sparse safety handoff. This is
    /// intentionally separate from sparse factorization cost.
    pub banded_sparse_fallback_assemblies: u64,
    pub banded_residual_checks: u64,
    pub banded_fallback_switches: u64,
    pub banded_rhs_permutations: u64,
    pub banded_rhs_permutation_entries: u64,
    /// Relative residual of the latest structured core factorization.
    pub banded_core_factor_residual_relative: Option<f64>,
    /// Smallest absolute diagonal pivot observed in the latest core factor.
    pub banded_min_abs_pivot: Option<f64>,
    /// Largest block multiplier norm observed in the latest core factor.
    pub banded_max_multiplier_norm: Option<f64>,
    /// Smallest absolute diagonal pivot in the latest border Schur factor.
    pub banded_border_min_abs_pivot: Option<f64>,
    /// Largest lower multiplier in the latest border Schur factor.
    pub banded_border_max_multiplier_norm: Option<f64>,
    /// Infinity norm of the latest structured linear-solve residual.
    pub banded_last_residual_inf: Option<f64>,
    /// Infinity norm of the latest reordered structured solution.
    pub banded_last_solution_inf: Option<f64>,
    pub parameter_rebinds: u64,
    pub continuation_solves: u64,
    pub restarts: u64,
    pub copies: u64,
    pub workspace_resizes: u64,
    pub output_writes: u64,
    pub argument_bindings: u64,
    pub residual_evaluations: u64,
    pub jacobian_evaluations: u64,
    pub jacobian_output_assemblies: u64,
    /// Logical workspace growth events, not process-wide heap allocations.
    pub allocations: u64,
    pub atom_conversions: u64,
    pub symbolic_jacobian_derivations: u64,
    pub pattern_entries: u64,
    pub lowering_calls: u64,
    pub evaluator_compilations: u64,
    pub residual_evaluator_compilations: u64,
    pub jacobian_evaluator_compilations: u64,
    pub singular_term_applications: u64,
    pub dense_output_constructions: u64,
    pub preparation_calls: u64,
    pub newton_scope_calls: u64,
    pub mesh_defect_estimation_calls: u64,
    pub output_construction_calls: u64,
    /// Number of public `solve` calls observed by telemetry. The count is
    /// recorded before entering the numerical core, so error paths remain
    /// visible even when timing collection is disabled.
    pub full_solve_calls: u64,
    /// Number of callback requests dispatched to worker threads.
    pub parallel_dispatches: u64,
    /// Number of callback requests kept on the calling thread.
    pub sequential_dispatches: u64,
    /// Maximum Rayon worker count observed by a parallel callback.
    pub max_worker_threads: u64,
    /// Inclusive stage timings. Parent and child scopes are diagnostic and
    /// must not be added together as a total.
    pub preparation_ms: Option<f64>,
    pub expr_to_atom_ms: Option<f64>,
    pub symbolic_jacobian_ms: Option<f64>,
    pub pattern_ms: Option<f64>,
    /// Atom IR lowering time, separate from symbolic differentiation and
    /// evaluator materialization.
    pub lowering_ms: Option<f64>,
    pub evaluator_compilation_ms: Option<f64>,
    pub residual_evaluator_compilation_ms: Option<f64>,
    pub jacobian_evaluator_compilation_ms: Option<f64>,
    pub binding_ms: Option<f64>,
    pub residual_evaluation_ms: Option<f64>,
    pub jacobian_evaluation_ms: Option<f64>,
    pub jacobian_output_assembly_ms: Option<f64>,
    pub callback_ms: Option<f64>,
    pub collocation_ms: Option<f64>,
    pub mesh_refinement_ms: Option<f64>,
    pub newton_ms: Option<f64>,
    pub mesh_defect_estimation_ms: Option<f64>,
    pub output_construction_ms: Option<f64>,
    pub linear_assembly_ms: Option<f64>,
    pub factorization_ms: Option<f64>,
    pub sparse_symbolic_analysis_ms: Option<f64>,
    pub sparse_numeric_factorization_ms: Option<f64>,
    /// Compatibility alias for the historical linear-solve timing field.
    pub solve_ms: Option<f64>,
    pub linear_solve_ms: Option<f64>,
    /// Inclusive wall-clock time of the most recent public solve call.
    pub full_solve_ms: Option<f64>,
    /// Inclusive wall-clock time accumulated across all public solve calls.
    pub full_solve_total_ms: Option<f64>,
    pub banded_structured_factorization_ms: Option<f64>,
    pub banded_scalar_fallback_factorization_ms: Option<f64>,
    pub banded_scalar_fallback_assembly_ms: Option<f64>,
    pub banded_structured_solve_ms: Option<f64>,
    pub banded_scalar_fallback_solve_ms: Option<f64>,
    pub banded_sparse_fallback_factorization_ms: Option<f64>,
    pub banded_sparse_fallback_solve_ms: Option<f64>,
    pub banded_sparse_fallback_assembly_ms: Option<f64>,
    pub banded_residual_guard_ms: Option<f64>,
    pub banded_rhs_permutation_ms: Option<f64>,
    pub timing_scopes: Vec<BvpSciTelemetryScope>,
}

/// Typed failure returned when a published telemetry snapshot violates its
/// reporting contract.  This is intentionally a post-run diagnostic API: it
/// is never evaluated from a residual, Jacobian, or linear-algebra callback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BvpSciTelemetryContractError {
    DuplicateScope(BvpSciTelemetryStage),
    MissingParent {
        stage: BvpSciTelemetryStage,
        parent: BvpSciTelemetryStage,
    },
    SelfParent(BvpSciTelemetryStage),
    TimingDataWithoutTimingMode,
    FullSolveTotalBelowLast {
        last_ms: u64,
        total_ms: u64,
    },
    Aot(IvpTelemetryContractError),
}

impl fmt::Display for BvpSciTelemetryContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateScope(stage) => {
                write!(formatter, "duplicate telemetry scope: {stage:?}")
            }
            Self::MissingParent { stage, parent } => {
                write!(
                    formatter,
                    "telemetry scope {stage:?} references missing parent {parent:?}"
                )
            }
            Self::SelfParent(stage) => {
                write!(formatter, "telemetry scope {stage:?} is its own parent")
            }
            Self::TimingDataWithoutTimingMode => {
                write!(
                    formatter,
                    "timing data is present while telemetry timing mode is disabled"
                )
            }
            Self::FullSolveTotalBelowLast { last_ms, total_ms } => write!(
                formatter,
                "full_solve_total_ms ({total_ms}) is below full_solve_ms ({last_ms})"
            ),
            Self::Aot(error) => write!(formatter, "AOT telemetry contract violation: {error}"),
        }
    }
}

impl std::error::Error for BvpSciTelemetryContractError {}

impl BvpSciTelemetrySnapshot {
    /// Return the preparation measurement that belongs in a user-facing
    /// report. AOT preparation lives in the shared IVP snapshot and is not
    /// copied into the BVP scope, because those scopes have different parents
    /// and are not additive. `None` means that the scope was not measured.
    pub fn report_preparation_ms(&self) -> Option<f64> {
        self.preparation_ms.or_else(|| {
            self.aot.as_ref().and_then(|snapshot| {
                let timing = snapshot
                    .cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::SolverPreparation);
                (timing.calls > 0).then(|| timing.elapsed.as_secs_f64() * 1e3)
            })
        })
    }

    /// Identify which lifecycle owns `report_preparation_ms`.
    pub fn report_preparation_scope(&self) -> &'static str {
        if self.preparation_ms.is_some() {
            "bvp"
        } else if self.report_preparation_ms().is_some() {
            "aot_ivp"
        } else {
            "not_measured"
        }
    }

    /// Validate the non-additive timing graph and the attached AOT snapshot.
    ///
    /// Reports may contain both parent and child inclusive scopes.  This
    /// method therefore checks identity and parent references, but never
    /// compares or sums elapsed values from different scopes.
    pub fn validate_contract(&self) -> Result<(), BvpSciTelemetryContractError> {
        let has_timing_fields = [
            self.preparation_ms,
            self.expr_to_atom_ms,
            self.symbolic_jacobian_ms,
            self.pattern_ms,
            self.lowering_ms,
            self.evaluator_compilation_ms,
            self.residual_evaluator_compilation_ms,
            self.jacobian_evaluator_compilation_ms,
            self.binding_ms,
            self.residual_evaluation_ms,
            self.jacobian_evaluation_ms,
            self.jacobian_output_assembly_ms,
            self.callback_ms,
            self.collocation_ms,
            self.mesh_refinement_ms,
            self.newton_ms,
            self.mesh_defect_estimation_ms,
            self.output_construction_ms,
            self.linear_assembly_ms,
            self.factorization_ms,
            self.sparse_symbolic_analysis_ms,
            self.sparse_numeric_factorization_ms,
            self.solve_ms,
            self.linear_solve_ms,
            self.full_solve_ms,
            self.full_solve_total_ms,
            self.banded_structured_factorization_ms,
            self.banded_scalar_fallback_factorization_ms,
            self.banded_scalar_fallback_assembly_ms,
            self.banded_structured_solve_ms,
            self.banded_scalar_fallback_solve_ms,
            self.banded_sparse_fallback_factorization_ms,
            self.banded_sparse_fallback_solve_ms,
            self.banded_sparse_fallback_assembly_ms,
            self.banded_residual_guard_ms,
            self.banded_rhs_permutation_ms,
        ]
        .into_iter()
        .any(|value| value.is_some());
        if self.mode != BvpSciTelemetryMode::Timings {
            if !self.timing_scopes.is_empty() || has_timing_fields {
                return Err(BvpSciTelemetryContractError::TimingDataWithoutTimingMode);
            }
        }

        let mut stages = HashSet::with_capacity(self.timing_scopes.len());
        for scope in &self.timing_scopes {
            if !stages.insert(scope.stage) {
                return Err(BvpSciTelemetryContractError::DuplicateScope(scope.stage));
            }
            if scope.parent == Some(scope.stage) {
                return Err(BvpSciTelemetryContractError::SelfParent(scope.stage));
            }
            if let Some(parent) = scope.parent {
                if !stages.contains(&parent)
                    && !self
                        .timing_scopes
                        .iter()
                        .any(|candidate| candidate.stage == parent)
                {
                    return Err(BvpSciTelemetryContractError::MissingParent {
                        stage: scope.stage,
                        parent,
                    });
                }
            }
        }

        if let (Some(last), Some(total)) = (self.full_solve_ms, self.full_solve_total_ms) {
            if last > total + f64::EPSILON {
                return Err(BvpSciTelemetryContractError::FullSolveTotalBelowLast {
                    last_ms: (last * 1_000_000.0) as u64,
                    total_ms: (total * 1_000_000.0) as u64,
                });
            }
        }
        if let Some(aot) = &self.aot {
            aot.validate_contract()
                .map_err(BvpSciTelemetryContractError::Aot)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct TelemetryInner {
    rhs_calls: AtomicU64,
    boundary_calls: AtomicU64,
    jacobian_calls: AtomicU64,
    finite_difference_probes: AtomicU64,
    newton_iterations: AtomicU64,
    newton_jacobian_refreshes: AtomicU64,
    newton_backtracking_trials: AtomicU64,
    newton_accepted_steps: AtomicU64,
    newton_rejected_steps: AtomicU64,
    newton_residual_history: Option<Mutex<Vec<BvpSciNewtonTraceEntry>>>,
    collocation_evaluations: AtomicU64,
    mesh_defect_probes: AtomicU64,
    mesh_refinements: AtomicU64,
    mesh_points_added: AtomicU64,
    linear_assemblies: AtomicU64,
    factorizations: AtomicU64,
    sparse_symbolic_analyses: AtomicU64,
    sparse_numeric_factorizations: AtomicU64,
    linear_solves: AtomicU64,
    banded_structured_factorizations: AtomicU64,
    banded_scalar_fallback_factorizations: AtomicU64,
    banded_scalar_fallback_assemblies: AtomicU64,
    banded_structured_solves: AtomicU64,
    banded_scalar_fallback_solves: AtomicU64,
    banded_sparse_fallback_factorizations: AtomicU64,
    banded_sparse_fallback_solves: AtomicU64,
    banded_sparse_fallback_assemblies: AtomicU64,
    banded_residual_checks: AtomicU64,
    banded_fallback_switches: AtomicU64,
    banded_rhs_permutations: AtomicU64,
    banded_rhs_permutation_entries: AtomicU64,
    parameter_rebinds: AtomicU64,
    continuation_solves: AtomicU64,
    restarts: AtomicU64,
    copies: AtomicU64,
    workspace_resizes: AtomicU64,
    output_writes: AtomicU64,
    argument_bindings: AtomicU64,
    residual_evaluations: AtomicU64,
    jacobian_evaluations: AtomicU64,
    jacobian_output_assemblies: AtomicU64,
    allocations: AtomicU64,
    atom_conversions: AtomicU64,
    symbolic_jacobian_derivations: AtomicU64,
    pattern_entries: AtomicU64,
    evaluator_compilations: AtomicU64,
    residual_evaluator_compilations: AtomicU64,
    jacobian_evaluator_compilations: AtomicU64,
    singular_term_applications: AtomicU64,
    dense_output_constructions: AtomicU64,
    preparation_calls: AtomicU64,
    full_solve_calls: AtomicU64,
    newton_scope_calls: AtomicU64,
    mesh_defect_estimation_calls: AtomicU64,
    output_construction_calls: AtomicU64,
    parallel_dispatches: AtomicU64,
    sequential_dispatches: AtomicU64,
    max_worker_threads: AtomicU64,
    preparation_ns: AtomicU64,
    expr_to_atom_ns: AtomicU64,
    symbolic_jacobian_ns: AtomicU64,
    pattern_ns: AtomicU64,
    lowering_ns: AtomicU64,
    lowering_calls: AtomicU64,
    evaluator_compilation_ns: AtomicU64,
    residual_evaluator_compilation_ns: AtomicU64,
    jacobian_evaluator_compilation_ns: AtomicU64,
    binding_ns: AtomicU64,
    residual_evaluation_ns: AtomicU64,
    jacobian_evaluation_ns: AtomicU64,
    jacobian_output_assembly_ns: AtomicU64,
    callback_ns: AtomicU64,
    collocation_ns: AtomicU64,
    mesh_refinement_ns: AtomicU64,
    newton_ns: AtomicU64,
    mesh_defect_estimation_ns: AtomicU64,
    output_construction_ns: AtomicU64,
    linear_assembly_ns: AtomicU64,
    factorization_ns: AtomicU64,
    sparse_symbolic_analysis_ns: AtomicU64,
    sparse_numeric_factorization_ns: AtomicU64,
    solve_ns: AtomicU64,
    full_solve_ns: AtomicU64,
    last_full_solve_ns: AtomicU64,
    banded_structured_factorization_ns: AtomicU64,
    banded_scalar_fallback_factorization_ns: AtomicU64,
    banded_scalar_fallback_assembly_ns: AtomicU64,
    banded_structured_solve_ns: AtomicU64,
    banded_scalar_fallback_solve_ns: AtomicU64,
    banded_sparse_fallback_factorization_ns: AtomicU64,
    banded_sparse_fallback_solve_ns: AtomicU64,
    banded_sparse_fallback_assembly_ns: AtomicU64,
    banded_residual_guard_ns: AtomicU64,
    banded_rhs_permutation_ns: AtomicU64,
    banded_core_factor_residual_relative_bits: AtomicU64,
    banded_min_abs_pivot_bits: AtomicU64,
    banded_max_multiplier_norm_bits: AtomicU64,
    banded_border_min_abs_pivot_bits: AtomicU64,
    banded_border_max_multiplier_norm_bits: AtomicU64,
    banded_last_residual_inf_bits: AtomicU64,
    banded_last_solution_inf_bits: AtomicU64,
}

impl TelemetryInner {
    fn new(mode: BvpSciTelemetryMode) -> Self {
        Self {
            newton_residual_history: (mode == BvpSciTelemetryMode::Timings)
                .then(|| Mutex::new(Vec::new())),
            banded_core_factor_residual_relative_bits: AtomicU64::new(u64::MAX),
            banded_min_abs_pivot_bits: AtomicU64::new(u64::MAX),
            banded_max_multiplier_norm_bits: AtomicU64::new(u64::MAX),
            banded_border_min_abs_pivot_bits: AtomicU64::new(u64::MAX),
            banded_border_max_multiplier_norm_bits: AtomicU64::new(u64::MAX),
            banded_last_residual_inf_bits: AtomicU64::new(u64::MAX),
            banded_last_solution_inf_bits: AtomicU64::new(u64::MAX),
            ..Self::default()
        }
    }
}

fn decode_optional_f64(bits: u64) -> Option<f64> {
    (bits != u64::MAX).then(|| f64::from_bits(bits))
}

/// Runtime telemetry handle shared by a prepared model and its callbacks.
///
/// The disabled handle contains no `Arc`, does not read the clock and does not
/// touch atomics. This is the required production default for hot callbacks.
#[derive(Clone, Debug)]
pub struct BvpSciTelemetry {
    mode: BvpSciTelemetryMode,
    inner: Option<Arc<TelemetryInner>>,
}

/// RAII timer for a public solve call.
///
/// Keeping this guard in telemetry rather than in the numerical controller
/// ensures that all early-return error paths contribute to the same inclusive
/// solve scope without duplicating timing code at every return site.
pub struct BvpSciTelemetryTimer {
    telemetry: BvpSciTelemetry,
    started: Instant,
    kind: BvpSciTelemetryTimerKind,
}

#[derive(Clone, Copy, Debug)]
enum BvpSciTelemetryTimerKind {
    FullSolve,
    Stage(BvpSciTelemetryStage),
}

impl Drop for BvpSciTelemetryTimer {
    fn drop(&mut self) {
        let Some(inner) = &self.telemetry.inner else {
            return;
        };
        let elapsed = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        match self.kind {
            BvpSciTelemetryTimerKind::FullSolve => {
                inner.full_solve_ns.fetch_add(elapsed, Ordering::Relaxed);
                inner.last_full_solve_ns.store(elapsed, Ordering::Relaxed);
            }
            BvpSciTelemetryTimerKind::Stage(stage) => {
                let slot = match stage {
                    BvpSciTelemetryStage::Newton => &inner.newton_ns,
                    BvpSciTelemetryStage::MeshDefectEstimation => &inner.mesh_defect_estimation_ns,
                    BvpSciTelemetryStage::OutputConstruction => &inner.output_construction_ns,
                    _ => return,
                };
                slot.fetch_add(elapsed, Ordering::Relaxed);
            }
        }
    }
}

impl Default for BvpSciTelemetry {
    fn default() -> Self {
        Self::disabled()
    }
}

impl BvpSciTelemetry {
    /// Construct the default zero-overhead telemetry handle.
    ///
    /// The returned handle does not allocate, read a clock or update atomics.
    pub fn disabled() -> Self {
        Self {
            mode: BvpSciTelemetryMode::Off,
            inner: None,
        }
    }

    /// Construct a counter-only handle without wall-clock measurements.
    pub fn counters() -> Self {
        Self::with_mode(BvpSciTelemetryMode::Counters)
    }

    /// Construct a handle collecting counters and inclusive stage timings.
    pub fn timings() -> Self {
        Self::with_mode(BvpSciTelemetryMode::Timings)
    }

    /// Construct a handle for an explicit telemetry mode.
    pub fn with_mode(mode: BvpSciTelemetryMode) -> Self {
        Self {
            mode,
            inner: (mode != BvpSciTelemetryMode::Off).then(|| Arc::new(TelemetryInner::new(mode))),
        }
    }

    /// Return the collection mode of this handle.
    pub fn mode(&self) -> BvpSciTelemetryMode {
        self.mode
    }

    /// Check whether two handles update the same counter storage.
    ///
    /// This is used only at lifecycle boundaries to prevent double-counting
    /// instrumented adapters; it is not part of any numerical hot loop.
    pub(crate) fn shares_storage(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, None) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn start_timing(&self) -> Option<Instant> {
        (self.mode == BvpSciTelemetryMode::Timings).then(Instant::now)
    }

    /// Start an inclusive timer covering one public solve call.
    #[inline]
    pub fn start_full_solve(&self) -> Option<BvpSciTelemetryTimer> {
        if let Some(inner) = &self.inner {
            inner.full_solve_calls.fetch_add(1, Ordering::Relaxed);
        }
        let started = self.start_timing()?;
        Some(BvpSciTelemetryTimer {
            telemetry: self.clone(),
            started,
            kind: BvpSciTelemetryTimerKind::FullSolve,
        })
    }

    /// Start an aggregate numerical phase timer nested inside a public solve.
    /// Only the three top-level phases use this generic timer; backend and
    /// callback scopes retain their specialized low-overhead record methods.
    #[inline]
    pub fn start_stage(&self, stage: BvpSciTelemetryStage) -> Option<BvpSciTelemetryTimer> {
        let started = self.start_timing()?;
        if let Some(inner) = &self.inner {
            match stage {
                BvpSciTelemetryStage::Newton => {
                    inner.newton_scope_calls.fetch_add(1, Ordering::Relaxed);
                }
                BvpSciTelemetryStage::MeshDefectEstimation => {
                    inner
                        .mesh_defect_estimation_calls
                        .fetch_add(1, Ordering::Relaxed);
                }
                BvpSciTelemetryStage::OutputConstruction => {
                    inner
                        .output_construction_calls
                        .fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        }
        Some(BvpSciTelemetryTimer {
            telemetry: self.clone(),
            started,
            kind: BvpSciTelemetryTimerKind::Stage(stage),
        })
    }

    #[inline]
    fn add_duration(slot: &AtomicU64, started: Option<Instant>) {
        if let Some(started) = started {
            Self::add_duration_value(slot, Some(started.elapsed()));
        }
    }

    #[inline]
    fn add_duration_value(slot: &AtomicU64, elapsed: Option<Duration>) {
        if let Some(elapsed) = elapsed {
            slot.fetch_add(
                elapsed.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
    }

    #[inline]
    fn record_call(
        counter: &AtomicU64,
        timer: &AtomicU64,
        started: Option<Instant>,
        mode: BvpSciTelemetryMode,
    ) {
        counter.fetch_add(1, Ordering::Relaxed);
        if mode == BvpSciTelemetryMode::Timings {
            Self::add_duration(timer, started);
        }
    }

    #[inline]
    pub fn record_rhs(&self, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        Self::record_call(&inner.rhs_calls, &inner.callback_ns, started, self.mode);
    }

    #[inline]
    pub fn record_argument_binding(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner.argument_bindings.fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.binding_ns, started);
        }
    }

    #[inline]
    pub fn record_residual_evaluation(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner.residual_evaluations.fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.residual_evaluation_ns, started);
        }
    }

    #[inline]
    pub fn record_jacobian_evaluation(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner.jacobian_evaluations.fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.jacobian_evaluation_ns, started);
        }
    }

    #[inline]
    pub fn record_jacobian_output_assembly(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner
                .jacobian_output_assemblies
                .fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.jacobian_output_assembly_ns, started);
        }
    }

    #[inline]
    pub fn record_copy(&self) {
        if let Some(inner) = &self.inner {
            inner.copies.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_output_writes(&self, count: usize) {
        if let Some(inner) = &self.inner {
            inner
                .output_writes
                .fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_boundary(&self, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        Self::record_call(
            &inner.boundary_calls,
            &inner.callback_ns,
            started,
            self.mode,
        );
    }

    #[inline]
    pub fn record_jacobian(&self, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        Self::record_call(
            &inner.jacobian_calls,
            &inner.callback_ns,
            started,
            self.mode,
        );
    }

    pub fn record_finite_difference_probe(&self) {
        if let Some(inner) = &self.inner {
            inner
                .finite_difference_probes
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_newton_iteration(&self) {
        if let Some(inner) = &self.inner {
            inner.newton_iterations.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_newton_jacobian_refresh(&self) {
        if let Some(inner) = &self.inner {
            inner
                .newton_jacobian_refreshes
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_newton_backtracking_trial(&self) {
        if let Some(inner) = &self.inner {
            inner
                .newton_backtracking_trials
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_newton_accepted_step(&self) {
        if let Some(inner) = &self.inner {
            inner.newton_accepted_steps.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_newton_rejected_step(&self) {
        if let Some(inner) = &self.inner {
            inner.newton_rejected_steps.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Store one detailed Newton observation. This is disabled unless timing
    /// telemetry was explicitly requested, so production callbacks remain
    /// allocation-free and lock-free.
    pub fn record_newton_trace(&self, entry: BvpSciNewtonTraceEntry) {
        let Some(inner) = &self.inner else { return };
        let Some(history) = &inner.newton_residual_history else {
            return;
        };
        if let Ok(mut history) = history.lock() {
            history.push(entry);
        }
    }

    #[inline]
    pub fn record_mesh_defect_probe(&self) {
        if let Some(inner) = &self.inner {
            inner.mesh_defect_probes.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_linear_assembly(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner.linear_assemblies.fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.linear_assembly_ns, started);
        }
    }

    #[inline]
    pub fn record_factorization(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner.factorizations.fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.factorization_ns, started);
        }
    }

    /// Record the one-time faer sparse symbolic analysis for a fixed pattern.
    ///
    /// Parameter continuation and Newton refreshes should increment the
    /// numeric counter without incrementing this counter while the sparse
    /// structure remains unchanged.
    #[inline]
    pub fn record_sparse_symbolic_analysis(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner
                .sparse_symbolic_analyses
                .fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.sparse_symbolic_analysis_ns, started);
        }
    }

    /// Record one faer sparse numeric factorization using a prepared pattern.
    #[inline]
    pub fn record_sparse_numeric_factorization(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner
                .sparse_numeric_factorizations
                .fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.sparse_numeric_factorization_ns, started);
        }
    }

    #[inline]
    pub fn record_linear_solve(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner.linear_solves.fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.solve_ns, started);
        }
    }

    /// Record which Banded factorization route actually ran.
    #[inline]
    pub fn record_banded_factorization(&self, route: BvpSciBandedRoute, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        match route {
            BvpSciBandedRoute::Structured => {
                inner
                    .banded_structured_factorizations
                    .fetch_add(1, Ordering::Relaxed);
                Self::add_duration(&inner.banded_structured_factorization_ns, started);
            }
            BvpSciBandedRoute::ScalarFallback => {
                inner
                    .banded_scalar_fallback_factorizations
                    .fetch_add(1, Ordering::Relaxed);
                Self::add_duration(&inner.banded_scalar_fallback_factorization_ns, started);
            }
            BvpSciBandedRoute::SparseFallback => {
                inner
                    .banded_sparse_fallback_factorizations
                    .fetch_add(1, Ordering::Relaxed);
                Self::add_duration(&inner.banded_sparse_fallback_factorization_ns, started);
            }
        }
    }

    /// Record direct assembly of the scalar safety matrix.
    ///
    /// This is separate from `linear_assembly_ms`: the latter covers the
    /// complete backend assembly, while this field exposes the extra work
    /// paid even when the structured route succeeds.
    #[inline]
    pub fn record_banded_scalar_fallback_assembly(&self, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        inner
            .banded_scalar_fallback_assemblies
            .fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&inner.banded_scalar_fallback_assembly_ns, started);
    }

    /// Record copying the original global triplets into the lazy sparse
    /// safety handoff. This cost is paid during every numeric assembly, even
    /// when the structured route remains healthy, so it must not be hidden in
    /// sparse factorization timing.
    #[inline]
    pub fn record_banded_sparse_fallback_assembly(&self, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        inner
            .banded_sparse_fallback_assemblies
            .fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&inner.banded_sparse_fallback_assembly_ns, started);
    }

    /// Record which Banded solve route actually ran.
    #[inline]
    pub fn record_banded_solve(&self, route: BvpSciBandedRoute, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        match route {
            BvpSciBandedRoute::Structured => {
                inner
                    .banded_structured_solves
                    .fetch_add(1, Ordering::Relaxed);
                Self::add_duration(&inner.banded_structured_solve_ns, started);
            }
            BvpSciBandedRoute::ScalarFallback => {
                inner
                    .banded_scalar_fallback_solves
                    .fetch_add(1, Ordering::Relaxed);
                Self::add_duration(&inner.banded_scalar_fallback_solve_ns, started);
            }
            BvpSciBandedRoute::SparseFallback => {
                inner
                    .banded_sparse_fallback_solves
                    .fetch_add(1, Ordering::Relaxed);
                Self::add_duration(&inner.banded_sparse_fallback_solve_ns, started);
            }
        }
    }

    /// Record a numerical residual check used to validate a structured solve.
    #[inline]
    pub fn record_banded_residual_guard(&self, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        inner.banded_residual_checks.fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&inner.banded_residual_guard_ns, started);
    }

    /// Record a switch from the structured Banded route to its safety route.
    #[inline]
    pub fn record_banded_fallback_switch(&self) {
        if let Some(inner) = &self.inner {
            inner
                .banded_fallback_switches
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record factor-quality diagnostics produced once per structured factor.
    /// These are point estimates, not a claim to be a full condition number.
    pub fn record_banded_factor_diagnostics(
        &self,
        factor_residual_relative: f64,
        min_abs_pivot: f64,
        max_multiplier_norm: f64,
    ) {
        let Some(inner) = &self.inner else { return };
        inner
            .banded_core_factor_residual_relative_bits
            .store(factor_residual_relative.to_bits(), Ordering::Relaxed);
        inner
            .banded_min_abs_pivot_bits
            .store(min_abs_pivot.to_bits(), Ordering::Relaxed);
        inner
            .banded_max_multiplier_norm_bits
            .store(max_multiplier_norm.to_bits(), Ordering::Relaxed);
    }

    /// Record diagnostics for the small dense Schur/border factor.
    pub fn record_banded_border_factor_diagnostics(
        &self,
        min_abs_pivot: f64,
        max_multiplier_norm: f64,
    ) {
        let Some(inner) = &self.inner else { return };
        inner
            .banded_border_min_abs_pivot_bits
            .store(min_abs_pivot.to_bits(), Ordering::Relaxed);
        inner
            .banded_border_max_multiplier_norm_bits
            .store(max_multiplier_norm.to_bits(), Ordering::Relaxed);
    }

    /// Record the solve residual and solution scale without materializing a
    /// dense global matrix. A finite but enormous solution is evidence for a
    /// Schur/conditioning problem even when core pivots look healthy.
    pub fn record_banded_solve_diagnostics(&self, residual_inf: f64, solution_inf: f64) {
        let Some(inner) = &self.inner else { return };
        inner
            .banded_last_residual_inf_bits
            .store(residual_inf.to_bits(), Ordering::Relaxed);
        inner
            .banded_last_solution_inf_bits
            .store(solution_inf.to_bits(), Ordering::Relaxed);
    }

    /// Record the reusable global-to-structured RHS/solution permutation.
    #[inline]
    pub fn record_banded_rhs_permutation(&self, entries: usize, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        inner
            .banded_rhs_permutations
            .fetch_add(1, Ordering::Relaxed);
        inner
            .banded_rhs_permutation_entries
            .fetch_add(entries as u64, Ordering::Relaxed);
        Self::add_duration(&inner.banded_rhs_permutation_ns, started);
    }

    #[inline]
    pub fn record_collocation(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner
                .collocation_evaluations
                .fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.collocation_ns, started);
        }
    }

    /// Record one accepted mesh change and the number of inserted nodes.
    #[inline]
    pub fn record_mesh_refinement(&self, started: Option<Instant>, points_added: usize) {
        if let Some(inner) = &self.inner {
            inner.mesh_refinements.fetch_add(1, Ordering::Relaxed);
            inner
                .mesh_points_added
                .fetch_add(points_added as u64, Ordering::Relaxed);
            Self::add_duration(&inner.mesh_refinement_ns, started);
        }
    }

    #[inline]
    pub fn record_workspace_resize(&self) {
        if let Some(inner) = &self.inner {
            inner.workspace_resizes.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_parameter_rebind(&self) {
        if let Some(inner) = &self.inner {
            inner.parameter_rebinds.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_continuation_solve(&self) {
        if let Some(inner) = &self.inner {
            inner.continuation_solves.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record one resize that increased at least one workspace capacity.
    /// This intentionally does not claim to be a global allocator counter.
    pub fn record_allocation(&self) {
        if let Some(inner) = &self.inner {
            inner.allocations.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_restart(&self) {
        if let Some(inner) = &self.inner {
            inner.restarts.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_preparation(&self, started: Option<Instant>) {
        if let Some(inner) = &self.inner {
            inner.preparation_calls.fetch_add(1, Ordering::Relaxed);
            Self::add_duration(&inner.preparation_ns, started);
        }
    }

    #[inline]
    pub fn record_atom_conversion(&self, started: Option<Instant>, entries: u64) {
        let Some(inner) = &self.inner else { return };
        inner.atom_conversions.fetch_add(entries, Ordering::Relaxed);
        Self::add_duration(&inner.expr_to_atom_ns, started);
    }

    #[inline]
    pub fn record_symbolic_jacobian(&self, started: Option<Instant>, derivatives: u64) {
        let Some(inner) = &self.inner else { return };
        inner
            .symbolic_jacobian_derivations
            .fetch_add(derivatives, Ordering::Relaxed);
        Self::add_duration(&inner.symbolic_jacobian_ns, started);
    }

    #[inline]
    pub fn record_pattern(&self, started: Option<Instant>, entries: u64) {
        let Some(inner) = &self.inner else { return };
        inner.pattern_entries.fetch_add(entries, Ordering::Relaxed);
        Self::add_duration(&inner.pattern_ns, started);
    }

    /// Record a pattern scope whose start was captured before a nested stage.
    ///
    /// AtomView discovers dependencies before differentiating. The public
    /// `record_pattern` helper is convenient for one contiguous scope, but it
    /// cannot represent that boundary without making pattern timing overlap
    /// the differentiation timer. This duration-based companion preserves
    /// the true structural scope while keeping the actual entry count.
    #[inline]
    pub fn record_pattern_elapsed(&self, elapsed: Option<Duration>, entries: u64) {
        let Some(inner) = &self.inner else { return };
        inner.pattern_entries.fetch_add(entries, Ordering::Relaxed);
        Self::add_duration_value(&inner.pattern_ns, elapsed);
    }

    /// Record Atom IR lowering independently from symbolic differentiation.
    ///
    /// Lowering is an optional fast path: unsupported expressions may leave
    /// this scope unobserved while the prepared evaluator path remains valid.
    #[inline]
    pub fn record_lowering(&self, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        inner.lowering_calls.fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&inner.lowering_ns, started);
    }

    #[inline]
    pub fn record_evaluator_compilation(&self, started: Option<Instant>, evaluators: u64) {
        let Some(inner) = &self.inner else { return };
        inner
            .evaluator_compilations
            .fetch_add(evaluators, Ordering::Relaxed);
        Self::add_duration(&inner.evaluator_compilation_ns, started);
    }

    /// Record residual evaluator materialization separately from Jacobian
    /// evaluator materialization. The aggregate evaluator fields remain
    /// populated for compatibility with older reports.
    #[inline]
    pub fn record_residual_evaluator_compilation(&self, started: Option<Instant>, evaluators: u64) {
        let Some(inner) = &self.inner else { return };
        inner
            .evaluator_compilations
            .fetch_add(evaluators, Ordering::Relaxed);
        inner
            .residual_evaluator_compilations
            .fetch_add(evaluators, Ordering::Relaxed);
        Self::add_duration(&inner.evaluator_compilation_ns, started);
        Self::add_duration(&inner.residual_evaluator_compilation_ns, started);
    }

    /// Record Jacobian evaluator materialization separately from residual
    /// evaluator materialization. This makes AtomView and ExprLegacy
    /// preparation directly comparable without changing the runtime ABI.
    #[inline]
    pub fn record_jacobian_evaluator_compilation(&self, started: Option<Instant>, evaluators: u64) {
        let Some(inner) = &self.inner else { return };
        inner
            .evaluator_compilations
            .fetch_add(evaluators, Ordering::Relaxed);
        inner
            .jacobian_evaluator_compilations
            .fetch_add(evaluators, Ordering::Relaxed);
        Self::add_duration(&inner.evaluator_compilation_ns, started);
        Self::add_duration(&inner.jacobian_evaluator_compilation_ns, started);
    }

    #[inline]
    pub fn record_singular_term_application(&self) {
        if let Some(inner) = &self.inner {
            inner
                .singular_term_applications
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_dense_output_construction(&self) {
        if let Some(inner) = &self.inner {
            inner
                .dense_output_constructions
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record the actual branch selected for one independent callback group.
    ///
    /// This distinguishes a configured `Parallel`/`Auto` policy from a real
    /// dispatch. It is intentionally one atomic update per callback group,
    /// not one update per scalar evaluator.
    #[inline]
    pub fn record_dispatch(&self, policy: BvpLambdifyExecutionPolicy, work: usize) {
        // This records the branch actually selected by the policy, not a
        // promise that Rayon created a new thread. `max_worker_threads` is
        // therefore interpreted together with dispatch counters in reports.
        let Some(inner) = &self.inner else { return };
        let parallel = work > 1 && policy.should_parallel_with_tasks(work, work);
        if parallel {
            inner.parallel_dispatches.fetch_add(1, Ordering::Relaxed);
            inner
                .max_worker_threads
                .fetch_max(rayon::current_num_threads() as u64, Ordering::Relaxed);
        } else {
            inner.sequential_dispatches.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> BvpSciTelemetrySnapshot {
        let Some(inner) = &self.inner else {
            return BvpSciTelemetrySnapshot {
                mode: self.mode,
                ..BvpSciTelemetrySnapshot::default()
            };
        };
        let timed = |ns: u64, calls: u64| {
            (self.mode == BvpSciTelemetryMode::Timings && calls > 0)
                .then(|| ns as f64 / 1_000_000.0)
        };
        let mut snapshot = BvpSciTelemetrySnapshot {
            mode: self.mode,
            aot: None,
            rhs_calls: inner.rhs_calls.load(Ordering::Relaxed),
            boundary_calls: inner.boundary_calls.load(Ordering::Relaxed),
            jacobian_calls: inner.jacobian_calls.load(Ordering::Relaxed),
            finite_difference_probes: inner.finite_difference_probes.load(Ordering::Relaxed),
            newton_iterations: inner.newton_iterations.load(Ordering::Relaxed),
            newton_jacobian_refreshes: inner.newton_jacobian_refreshes.load(Ordering::Relaxed),
            newton_backtracking_trials: inner.newton_backtracking_trials.load(Ordering::Relaxed),
            newton_accepted_steps: inner.newton_accepted_steps.load(Ordering::Relaxed),
            newton_rejected_steps: inner.newton_rejected_steps.load(Ordering::Relaxed),
            newton_residual_history: inner
                .newton_residual_history
                .as_ref()
                .and_then(|history| history.lock().ok())
                .map(|history| history.clone())
                .unwrap_or_default(),
            collocation_evaluations: inner.collocation_evaluations.load(Ordering::Relaxed),
            mesh_defect_probes: inner.mesh_defect_probes.load(Ordering::Relaxed),
            mesh_refinements: inner.mesh_refinements.load(Ordering::Relaxed),
            mesh_points_added: inner.mesh_points_added.load(Ordering::Relaxed),
            linear_assemblies: inner.linear_assemblies.load(Ordering::Relaxed),
            factorizations: inner.factorizations.load(Ordering::Relaxed),
            sparse_symbolic_analyses: inner.sparse_symbolic_analyses.load(Ordering::Relaxed),
            sparse_numeric_factorizations: inner
                .sparse_numeric_factorizations
                .load(Ordering::Relaxed),
            linear_solves: inner.linear_solves.load(Ordering::Relaxed),
            banded_structured_factorizations: inner
                .banded_structured_factorizations
                .load(Ordering::Relaxed),
            banded_scalar_fallback_factorizations: inner
                .banded_scalar_fallback_factorizations
                .load(Ordering::Relaxed),
            banded_scalar_fallback_assemblies: inner
                .banded_scalar_fallback_assemblies
                .load(Ordering::Relaxed),
            banded_structured_solves: inner.banded_structured_solves.load(Ordering::Relaxed),
            banded_scalar_fallback_solves: inner
                .banded_scalar_fallback_solves
                .load(Ordering::Relaxed),
            banded_sparse_fallback_factorizations: inner
                .banded_sparse_fallback_factorizations
                .load(Ordering::Relaxed),
            banded_sparse_fallback_solves: inner
                .banded_sparse_fallback_solves
                .load(Ordering::Relaxed),
            banded_sparse_fallback_assemblies: inner
                .banded_sparse_fallback_assemblies
                .load(Ordering::Relaxed),
            banded_residual_checks: inner.banded_residual_checks.load(Ordering::Relaxed),
            banded_fallback_switches: inner.banded_fallback_switches.load(Ordering::Relaxed),
            banded_rhs_permutations: inner.banded_rhs_permutations.load(Ordering::Relaxed),
            banded_rhs_permutation_entries: inner
                .banded_rhs_permutation_entries
                .load(Ordering::Relaxed),
            banded_core_factor_residual_relative: decode_optional_f64(
                inner
                    .banded_core_factor_residual_relative_bits
                    .load(Ordering::Relaxed),
            ),
            banded_min_abs_pivot: decode_optional_f64(
                inner.banded_min_abs_pivot_bits.load(Ordering::Relaxed),
            ),
            banded_max_multiplier_norm: decode_optional_f64(
                inner
                    .banded_max_multiplier_norm_bits
                    .load(Ordering::Relaxed),
            ),
            banded_border_min_abs_pivot: decode_optional_f64(
                inner
                    .banded_border_min_abs_pivot_bits
                    .load(Ordering::Relaxed),
            ),
            banded_border_max_multiplier_norm: decode_optional_f64(
                inner
                    .banded_border_max_multiplier_norm_bits
                    .load(Ordering::Relaxed),
            ),
            banded_last_residual_inf: decode_optional_f64(
                inner.banded_last_residual_inf_bits.load(Ordering::Relaxed),
            ),
            banded_last_solution_inf: decode_optional_f64(
                inner.banded_last_solution_inf_bits.load(Ordering::Relaxed),
            ),
            parameter_rebinds: inner.parameter_rebinds.load(Ordering::Relaxed),
            continuation_solves: inner.continuation_solves.load(Ordering::Relaxed),
            restarts: inner.restarts.load(Ordering::Relaxed),
            copies: inner.copies.load(Ordering::Relaxed),
            workspace_resizes: inner.workspace_resizes.load(Ordering::Relaxed),
            output_writes: inner.output_writes.load(Ordering::Relaxed),
            argument_bindings: inner.argument_bindings.load(Ordering::Relaxed),
            residual_evaluations: inner.residual_evaluations.load(Ordering::Relaxed),
            jacobian_evaluations: inner.jacobian_evaluations.load(Ordering::Relaxed),
            jacobian_output_assemblies: inner.jacobian_output_assemblies.load(Ordering::Relaxed),
            allocations: inner.allocations.load(Ordering::Relaxed),
            atom_conversions: inner.atom_conversions.load(Ordering::Relaxed),
            symbolic_jacobian_derivations: inner
                .symbolic_jacobian_derivations
                .load(Ordering::Relaxed),
            pattern_entries: inner.pattern_entries.load(Ordering::Relaxed),
            lowering_calls: inner.lowering_calls.load(Ordering::Relaxed),
            evaluator_compilations: inner.evaluator_compilations.load(Ordering::Relaxed),
            residual_evaluator_compilations: inner
                .residual_evaluator_compilations
                .load(Ordering::Relaxed),
            jacobian_evaluator_compilations: inner
                .jacobian_evaluator_compilations
                .load(Ordering::Relaxed),
            singular_term_applications: inner.singular_term_applications.load(Ordering::Relaxed),
            dense_output_constructions: inner.dense_output_constructions.load(Ordering::Relaxed),
            preparation_calls: inner.preparation_calls.load(Ordering::Relaxed),
            newton_scope_calls: inner.newton_scope_calls.load(Ordering::Relaxed),
            mesh_defect_estimation_calls: inner
                .mesh_defect_estimation_calls
                .load(Ordering::Relaxed),
            output_construction_calls: inner.output_construction_calls.load(Ordering::Relaxed),
            full_solve_calls: inner.full_solve_calls.load(Ordering::Relaxed),
            parallel_dispatches: inner.parallel_dispatches.load(Ordering::Relaxed),
            sequential_dispatches: inner.sequential_dispatches.load(Ordering::Relaxed),
            max_worker_threads: inner.max_worker_threads.load(Ordering::Relaxed),
            preparation_ms: (self.mode == BvpSciTelemetryMode::Timings
                && inner.preparation_calls.load(Ordering::Relaxed) > 0)
                .then(|| inner.preparation_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0),
            expr_to_atom_ms: timed(
                inner.expr_to_atom_ns.load(Ordering::Relaxed),
                inner.atom_conversions.load(Ordering::Relaxed),
            ),
            symbolic_jacobian_ms: timed(
                inner.symbolic_jacobian_ns.load(Ordering::Relaxed),
                inner.symbolic_jacobian_derivations.load(Ordering::Relaxed),
            ),
            pattern_ms: timed(
                inner.pattern_ns.load(Ordering::Relaxed),
                inner.pattern_entries.load(Ordering::Relaxed),
            ),
            lowering_ms: timed(
                inner.lowering_ns.load(Ordering::Relaxed),
                inner.lowering_calls.load(Ordering::Relaxed),
            ),
            evaluator_compilation_ms: timed(
                inner.evaluator_compilation_ns.load(Ordering::Relaxed),
                inner.evaluator_compilations.load(Ordering::Relaxed),
            ),
            residual_evaluator_compilation_ms: timed(
                inner
                    .residual_evaluator_compilation_ns
                    .load(Ordering::Relaxed),
                inner
                    .residual_evaluator_compilations
                    .load(Ordering::Relaxed),
            ),
            jacobian_evaluator_compilation_ms: timed(
                inner
                    .jacobian_evaluator_compilation_ns
                    .load(Ordering::Relaxed),
                inner
                    .jacobian_evaluator_compilations
                    .load(Ordering::Relaxed),
            ),
            binding_ms: timed(
                inner.binding_ns.load(Ordering::Relaxed),
                inner.argument_bindings.load(Ordering::Relaxed),
            ),
            residual_evaluation_ms: timed(
                inner.residual_evaluation_ns.load(Ordering::Relaxed),
                inner.residual_evaluations.load(Ordering::Relaxed),
            ),
            jacobian_evaluation_ms: timed(
                inner.jacobian_evaluation_ns.load(Ordering::Relaxed),
                inner.jacobian_evaluations.load(Ordering::Relaxed),
            ),
            jacobian_output_assembly_ms: timed(
                inner.jacobian_output_assembly_ns.load(Ordering::Relaxed),
                inner.jacobian_output_assemblies.load(Ordering::Relaxed),
            ),
            callback_ms: timed(
                inner.callback_ns.load(Ordering::Relaxed),
                inner.rhs_calls.load(Ordering::Relaxed)
                    + inner.boundary_calls.load(Ordering::Relaxed)
                    + inner.jacobian_calls.load(Ordering::Relaxed),
            ),
            collocation_ms: timed(
                inner.collocation_ns.load(Ordering::Relaxed),
                inner.collocation_evaluations.load(Ordering::Relaxed),
            ),
            mesh_refinement_ms: timed(
                inner.mesh_refinement_ns.load(Ordering::Relaxed),
                inner.mesh_refinements.load(Ordering::Relaxed),
            ),
            newton_ms: timed(
                inner.newton_ns.load(Ordering::Relaxed),
                inner.newton_scope_calls.load(Ordering::Relaxed),
            ),
            mesh_defect_estimation_ms: timed(
                inner.mesh_defect_estimation_ns.load(Ordering::Relaxed),
                inner.mesh_defect_estimation_calls.load(Ordering::Relaxed),
            ),
            output_construction_ms: timed(
                inner.output_construction_ns.load(Ordering::Relaxed),
                inner.output_construction_calls.load(Ordering::Relaxed),
            ),
            linear_assembly_ms: timed(
                inner.linear_assembly_ns.load(Ordering::Relaxed),
                inner.linear_assemblies.load(Ordering::Relaxed),
            ),
            factorization_ms: timed(
                inner.factorization_ns.load(Ordering::Relaxed),
                inner.factorizations.load(Ordering::Relaxed),
            ),
            sparse_symbolic_analysis_ms: timed(
                inner.sparse_symbolic_analysis_ns.load(Ordering::Relaxed),
                inner.sparse_symbolic_analyses.load(Ordering::Relaxed),
            ),
            sparse_numeric_factorization_ms: timed(
                inner
                    .sparse_numeric_factorization_ns
                    .load(Ordering::Relaxed),
                inner.sparse_numeric_factorizations.load(Ordering::Relaxed),
            ),
            solve_ms: timed(
                inner.solve_ns.load(Ordering::Relaxed),
                inner.linear_solves.load(Ordering::Relaxed),
            ),
            linear_solve_ms: timed(
                inner.solve_ns.load(Ordering::Relaxed),
                inner.linear_solves.load(Ordering::Relaxed),
            ),
            full_solve_ms: timed(
                inner.last_full_solve_ns.load(Ordering::Relaxed),
                inner.full_solve_calls.load(Ordering::Relaxed),
            ),
            full_solve_total_ms: timed(
                inner.full_solve_ns.load(Ordering::Relaxed),
                inner.full_solve_calls.load(Ordering::Relaxed),
            ),
            banded_structured_factorization_ms: timed(
                inner
                    .banded_structured_factorization_ns
                    .load(Ordering::Relaxed),
                inner
                    .banded_structured_factorizations
                    .load(Ordering::Relaxed),
            ),
            banded_scalar_fallback_factorization_ms: timed(
                inner
                    .banded_scalar_fallback_factorization_ns
                    .load(Ordering::Relaxed),
                inner
                    .banded_scalar_fallback_factorizations
                    .load(Ordering::Relaxed),
            ),
            banded_scalar_fallback_assembly_ms: timed(
                inner
                    .banded_scalar_fallback_assembly_ns
                    .load(Ordering::Relaxed),
                inner
                    .banded_scalar_fallback_assemblies
                    .load(Ordering::Relaxed),
            ),
            banded_structured_solve_ms: timed(
                inner.banded_structured_solve_ns.load(Ordering::Relaxed),
                inner.banded_structured_solves.load(Ordering::Relaxed),
            ),
            banded_scalar_fallback_solve_ms: timed(
                inner
                    .banded_scalar_fallback_solve_ns
                    .load(Ordering::Relaxed),
                inner.banded_scalar_fallback_solves.load(Ordering::Relaxed),
            ),
            banded_sparse_fallback_factorization_ms: timed(
                inner
                    .banded_sparse_fallback_factorization_ns
                    .load(Ordering::Relaxed),
                inner
                    .banded_sparse_fallback_factorizations
                    .load(Ordering::Relaxed),
            ),
            banded_sparse_fallback_solve_ms: timed(
                inner
                    .banded_sparse_fallback_solve_ns
                    .load(Ordering::Relaxed),
                inner.banded_sparse_fallback_solves.load(Ordering::Relaxed),
            ),
            banded_sparse_fallback_assembly_ms: timed(
                inner
                    .banded_sparse_fallback_assembly_ns
                    .load(Ordering::Relaxed),
                inner
                    .banded_sparse_fallback_assemblies
                    .load(Ordering::Relaxed),
            ),
            banded_residual_guard_ms: timed(
                inner.banded_residual_guard_ns.load(Ordering::Relaxed),
                inner.banded_residual_checks.load(Ordering::Relaxed),
            ),
            banded_rhs_permutation_ms: timed(
                inner.banded_rhs_permutation_ns.load(Ordering::Relaxed),
                inner.banded_rhs_permutations.load(Ordering::Relaxed),
            ),
            timing_scopes: Vec::new(),
        };
        if self.mode == BvpSciTelemetryMode::Timings {
            let scope = |stage, parent, elapsed_ms| {
                let calls = match stage {
                    BvpSciTelemetryStage::Preparation => snapshot.preparation_calls,
                    BvpSciTelemetryStage::FullSolve => snapshot.full_solve_calls,
                    BvpSciTelemetryStage::Newton => snapshot.newton_scope_calls,
                    BvpSciTelemetryStage::MeshDefectEstimation => {
                        snapshot.mesh_defect_estimation_calls
                    }
                    BvpSciTelemetryStage::OutputConstruction => snapshot.output_construction_calls,
                    BvpSciTelemetryStage::ExprToAtom => u64::from(snapshot.atom_conversions > 0),
                    BvpSciTelemetryStage::SymbolicJacobian => {
                        u64::from(snapshot.symbolic_jacobian_derivations > 0)
                    }
                    BvpSciTelemetryStage::Pattern => u64::from(snapshot.pattern_entries > 0),
                    BvpSciTelemetryStage::Lowering => snapshot.lowering_calls,
                    BvpSciTelemetryStage::EvaluatorCompilation => {
                        u64::from(snapshot.evaluator_compilations > 0)
                    }
                    BvpSciTelemetryStage::ResidualEvaluatorCompilation => {
                        snapshot.residual_evaluator_compilations
                    }
                    BvpSciTelemetryStage::JacobianEvaluatorCompilation => {
                        snapshot.jacobian_evaluator_compilations
                    }
                    BvpSciTelemetryStage::Binding => snapshot.argument_bindings,
                    BvpSciTelemetryStage::ResidualEvaluation => snapshot.residual_evaluations,
                    BvpSciTelemetryStage::JacobianEvaluation => snapshot.jacobian_evaluations,
                    BvpSciTelemetryStage::JacobianOutputAssembly => {
                        snapshot.jacobian_output_assemblies
                    }
                    BvpSciTelemetryStage::Callback => snapshot
                        .rhs_calls
                        .saturating_add(snapshot.boundary_calls)
                        .saturating_add(snapshot.jacobian_calls),
                    BvpSciTelemetryStage::Collocation => snapshot.collocation_evaluations,
                    BvpSciTelemetryStage::MeshRefinement => snapshot.mesh_refinements,
                    BvpSciTelemetryStage::LinearAssembly => snapshot.linear_assemblies,
                    BvpSciTelemetryStage::BandedScalarFallbackAssembly => {
                        snapshot.banded_scalar_fallback_assemblies
                    }
                    BvpSciTelemetryStage::Factorization => snapshot.factorizations,
                    BvpSciTelemetryStage::SparseSymbolicAnalysis => {
                        snapshot.sparse_symbolic_analyses
                    }
                    BvpSciTelemetryStage::SparseNumericFactorization => {
                        snapshot.sparse_numeric_factorizations
                    }
                    BvpSciTelemetryStage::LinearSolve => snapshot.linear_solves,
                    BvpSciTelemetryStage::BandedStructuredFactorization => {
                        snapshot.banded_structured_factorizations
                    }
                    BvpSciTelemetryStage::BandedScalarFallbackFactorization => {
                        snapshot.banded_scalar_fallback_factorizations
                    }
                    BvpSciTelemetryStage::BandedStructuredSolve => {
                        snapshot.banded_structured_solves
                    }
                    BvpSciTelemetryStage::BandedScalarFallbackSolve => {
                        snapshot.banded_scalar_fallback_solves
                    }
                    BvpSciTelemetryStage::BandedResidualGuard => snapshot.banded_residual_checks,
                    BvpSciTelemetryStage::BandedRhsPermutation => snapshot.banded_rhs_permutations,
                };
                BvpSciTelemetryScope {
                    stage,
                    kind: BvpSciTelemetryScopeKind::Inclusive,
                    parent,
                    calls,
                    elapsed_ms,
                }
            };
            snapshot.timing_scopes = vec![
                scope(
                    BvpSciTelemetryStage::Preparation,
                    None,
                    snapshot.preparation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::FullSolve,
                    None,
                    snapshot.full_solve_ms,
                ),
                scope(
                    BvpSciTelemetryStage::Newton,
                    Some(BvpSciTelemetryStage::FullSolve),
                    snapshot.newton_ms,
                ),
                scope(
                    BvpSciTelemetryStage::MeshDefectEstimation,
                    Some(BvpSciTelemetryStage::FullSolve),
                    snapshot.mesh_defect_estimation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::OutputConstruction,
                    Some(BvpSciTelemetryStage::FullSolve),
                    snapshot.output_construction_ms,
                ),
                scope(
                    BvpSciTelemetryStage::ExprToAtom,
                    Some(BvpSciTelemetryStage::Preparation),
                    snapshot.expr_to_atom_ms,
                ),
                scope(
                    BvpSciTelemetryStage::SymbolicJacobian,
                    Some(BvpSciTelemetryStage::Preparation),
                    snapshot.symbolic_jacobian_ms,
                ),
                scope(
                    BvpSciTelemetryStage::Pattern,
                    Some(BvpSciTelemetryStage::Preparation),
                    snapshot.pattern_ms,
                ),
                scope(
                    BvpSciTelemetryStage::Lowering,
                    Some(BvpSciTelemetryStage::Preparation),
                    snapshot.lowering_ms,
                ),
                scope(
                    BvpSciTelemetryStage::EvaluatorCompilation,
                    Some(BvpSciTelemetryStage::Preparation),
                    snapshot.evaluator_compilation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::ResidualEvaluatorCompilation,
                    Some(BvpSciTelemetryStage::EvaluatorCompilation),
                    snapshot.residual_evaluator_compilation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::JacobianEvaluatorCompilation,
                    Some(BvpSciTelemetryStage::EvaluatorCompilation),
                    snapshot.jacobian_evaluator_compilation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::Binding,
                    Some(BvpSciTelemetryStage::Callback),
                    snapshot.binding_ms,
                ),
                scope(
                    BvpSciTelemetryStage::ResidualEvaluation,
                    Some(BvpSciTelemetryStage::Callback),
                    snapshot.residual_evaluation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::JacobianEvaluation,
                    Some(BvpSciTelemetryStage::Callback),
                    snapshot.jacobian_evaluation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::JacobianOutputAssembly,
                    Some(BvpSciTelemetryStage::Callback),
                    snapshot.jacobian_output_assembly_ms,
                ),
                scope(BvpSciTelemetryStage::Callback, None, snapshot.callback_ms),
                scope(
                    BvpSciTelemetryStage::Collocation,
                    Some(BvpSciTelemetryStage::Newton),
                    snapshot.collocation_ms,
                ),
                scope(
                    BvpSciTelemetryStage::MeshRefinement,
                    Some(BvpSciTelemetryStage::FullSolve),
                    snapshot.mesh_refinement_ms,
                ),
                scope(
                    BvpSciTelemetryStage::LinearAssembly,
                    Some(BvpSciTelemetryStage::Newton),
                    snapshot.linear_assembly_ms,
                ),
                scope(
                    BvpSciTelemetryStage::BandedScalarFallbackAssembly,
                    Some(BvpSciTelemetryStage::LinearAssembly),
                    snapshot.banded_scalar_fallback_assembly_ms,
                ),
                scope(
                    BvpSciTelemetryStage::Factorization,
                    Some(BvpSciTelemetryStage::Newton),
                    snapshot.factorization_ms,
                ),
                scope(
                    BvpSciTelemetryStage::SparseSymbolicAnalysis,
                    Some(BvpSciTelemetryStage::Factorization),
                    snapshot.sparse_symbolic_analysis_ms,
                ),
                scope(
                    BvpSciTelemetryStage::SparseNumericFactorization,
                    Some(BvpSciTelemetryStage::Factorization),
                    snapshot.sparse_numeric_factorization_ms,
                ),
                scope(
                    BvpSciTelemetryStage::LinearSolve,
                    Some(BvpSciTelemetryStage::Newton),
                    snapshot.linear_solve_ms,
                ),
                scope(
                    BvpSciTelemetryStage::BandedStructuredFactorization,
                    Some(BvpSciTelemetryStage::Factorization),
                    snapshot.banded_structured_factorization_ms,
                ),
                scope(
                    BvpSciTelemetryStage::BandedScalarFallbackFactorization,
                    Some(BvpSciTelemetryStage::Factorization),
                    snapshot.banded_scalar_fallback_factorization_ms,
                ),
                scope(
                    BvpSciTelemetryStage::BandedStructuredSolve,
                    Some(BvpSciTelemetryStage::LinearSolve),
                    snapshot.banded_structured_solve_ms,
                ),
                scope(
                    BvpSciTelemetryStage::BandedScalarFallbackSolve,
                    Some(BvpSciTelemetryStage::LinearSolve),
                    snapshot.banded_scalar_fallback_solve_ms,
                ),
                scope(
                    BvpSciTelemetryStage::BandedResidualGuard,
                    Some(BvpSciTelemetryStage::BandedStructuredSolve),
                    snapshot.banded_residual_guard_ms,
                ),
                scope(
                    BvpSciTelemetryStage::BandedRhsPermutation,
                    Some(BvpSciTelemetryStage::LinearSolve),
                    snapshot.banded_rhs_permutation_ms,
                ),
            ];
        }
        snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_telemetry_has_no_runtime_storage_or_scopes() {
        let telemetry = BvpSciTelemetry::disabled();
        telemetry.record_banded_factorization(BvpSciBandedRoute::Structured, None);
        telemetry.record_banded_solve(BvpSciBandedRoute::ScalarFallback, None);
        telemetry.record_banded_scalar_fallback_assembly(None);
        let snapshot = telemetry.snapshot();

        assert_eq!(snapshot.mode, BvpSciTelemetryMode::Off);
        assert_eq!(snapshot.banded_structured_factorizations, 0);
        assert_eq!(snapshot.banded_scalar_fallback_solves, 0);
        assert_eq!(snapshot.banded_scalar_fallback_assemblies, 0);
        assert!(snapshot.full_solve_ms.is_none());
        assert!(snapshot.timing_scopes.is_empty());
    }

    #[test]
    fn timing_snapshot_describes_nested_and_banded_routes() {
        let telemetry = BvpSciTelemetry::timings();
        assert!(telemetry.snapshot().full_solve_ms.is_none());
        telemetry.record_banded_factorization(BvpSciBandedRoute::Structured, None);
        telemetry.record_banded_factorization(BvpSciBandedRoute::ScalarFallback, None);
        telemetry.record_banded_scalar_fallback_assembly(None);
        telemetry.record_banded_solve(BvpSciBandedRoute::Structured, None);
        telemetry.record_banded_residual_guard(None);
        telemetry.record_banded_rhs_permutation(12, None);
        telemetry.record_banded_fallback_switch();
        telemetry.record_newton_jacobian_refresh();
        telemetry.record_newton_backtracking_trial();
        telemetry.record_newton_accepted_step();
        telemetry.record_newton_rejected_step();
        {
            let _timer = telemetry.start_full_solve();
            let _newton = telemetry.start_stage(BvpSciTelemetryStage::Newton);
            let _defect = telemetry.start_stage(BvpSciTelemetryStage::MeshDefectEstimation);
            let _output = telemetry.start_stage(BvpSciTelemetryStage::OutputConstruction);
        }

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.banded_structured_factorizations, 1);
        assert_eq!(snapshot.banded_scalar_fallback_factorizations, 1);
        assert_eq!(snapshot.banded_scalar_fallback_assemblies, 1);
        assert_eq!(snapshot.banded_structured_solves, 1);
        assert_eq!(snapshot.banded_residual_checks, 1);
        assert_eq!(snapshot.banded_rhs_permutation_entries, 12);
        assert_eq!(snapshot.banded_fallback_switches, 1);
        assert_eq!(snapshot.newton_jacobian_refreshes, 1);
        assert_eq!(snapshot.newton_backtracking_trials, 1);
        assert_eq!(snapshot.newton_accepted_steps, 1);
        assert_eq!(snapshot.newton_rejected_steps, 1);
        assert!(snapshot.full_solve_ms.is_some());
        assert!(snapshot.full_solve_total_ms.is_some());
        assert_eq!(snapshot.full_solve_calls, 1);
        assert_eq!(snapshot.newton_scope_calls, 1);
        assert_eq!(snapshot.mesh_defect_estimation_calls, 1);
        assert_eq!(snapshot.output_construction_calls, 1);
        assert!(snapshot.newton_ms.is_some());
        assert!(snapshot.mesh_defect_estimation_ms.is_some());
        assert!(snapshot.output_construction_ms.is_some());
        snapshot
            .validate_contract()
            .expect("timing scope graph should satisfy the telemetry contract");

        let structured_factorization = snapshot
            .timing_scopes
            .iter()
            .find(|scope| scope.stage == BvpSciTelemetryStage::BandedStructuredFactorization)
            .expect("structured factorization scope is present");
        assert_eq!(
            structured_factorization.parent,
            Some(BvpSciTelemetryStage::Factorization)
        );
        assert_eq!(
            structured_factorization.kind,
            BvpSciTelemetryScopeKind::Inclusive
        );
        assert!(snapshot
            .timing_scopes
            .iter()
            .any(|scope| scope.stage == BvpSciTelemetryStage::FullSolve));
        let newton = snapshot
            .timing_scopes
            .iter()
            .find(|scope| scope.stage == BvpSciTelemetryStage::Newton)
            .expect("Newton phase scope is present");
        assert_eq!(newton.calls, 1);
        assert_eq!(newton.parent, Some(BvpSciTelemetryStage::FullSolve));
    }

    #[test]
    fn evaluator_compilation_scopes_remain_distinct() {
        let telemetry = BvpSciTelemetry::timings();
        telemetry.record_residual_evaluator_compilation(telemetry.start_timing(), 3);
        telemetry.record_jacobian_evaluator_compilation(telemetry.start_timing(), 5);
        let snapshot = telemetry.snapshot();

        assert_eq!(snapshot.evaluator_compilations, 8);
        assert_eq!(snapshot.residual_evaluator_compilations, 3);
        assert_eq!(snapshot.jacobian_evaluator_compilations, 5);
        assert!(snapshot.residual_evaluator_compilation_ms.is_some());
        assert!(snapshot.jacobian_evaluator_compilation_ms.is_some());
        assert!(snapshot.timing_scopes.iter().any(|scope| {
            scope.stage == BvpSciTelemetryStage::ResidualEvaluatorCompilation
                && scope.parent == Some(BvpSciTelemetryStage::EvaluatorCompilation)
        }));
        assert!(snapshot.timing_scopes.iter().any(|scope| {
            scope.stage == BvpSciTelemetryStage::JacobianEvaluatorCompilation
                && scope.parent == Some(BvpSciTelemetryStage::EvaluatorCompilation)
        }));
        snapshot
            .validate_contract()
            .expect("split evaluator scopes should satisfy the telemetry contract");
    }

    #[test]
    fn pattern_elapsed_preserves_entries_without_nested_derivative_time() {
        let telemetry = BvpSciTelemetry::timings();
        telemetry.record_pattern_elapsed(Some(Duration::from_millis(1)), 9);
        let snapshot = telemetry.snapshot();

        assert_eq!(snapshot.pattern_entries, 9);
        assert!(snapshot.pattern_ms.is_some());
        snapshot
            .validate_contract()
            .expect("elapsed pattern scope should satisfy the telemetry contract");
    }

    #[test]
    fn counters_record_solve_calls_without_reading_a_clock() {
        let telemetry = BvpSciTelemetry::counters();
        assert!(telemetry.start_full_solve().is_none());
        let snapshot = telemetry.snapshot();

        assert_eq!(snapshot.full_solve_calls, 1);
        assert!(snapshot.full_solve_ms.is_none());
        assert!(snapshot.full_solve_total_ms.is_none());
        assert!(snapshot.timing_scopes.is_empty());
        snapshot
            .validate_contract()
            .expect("counter-only telemetry must have no timing graph");
    }

    #[test]
    fn telemetry_contract_rejects_duplicate_scopes_and_bad_full_solve_totals() {
        let telemetry = BvpSciTelemetry::timings();
        {
            let _timer = telemetry.start_full_solve();
        }
        let mut snapshot = telemetry.snapshot();
        let duplicate = snapshot
            .timing_scopes
            .first()
            .copied()
            .expect("full-solve timing should create one scope");
        snapshot.timing_scopes.push(duplicate);
        assert_eq!(
            snapshot.validate_contract(),
            Err(BvpSciTelemetryContractError::DuplicateScope(
                duplicate.stage
            ))
        );

        let mut snapshot = telemetry.snapshot();
        snapshot.full_solve_ms = Some(2.0);
        snapshot.full_solve_total_ms = Some(1.0);
        assert!(matches!(
            snapshot.validate_contract(),
            Err(BvpSciTelemetryContractError::FullSolveTotalBelowLast { .. })
        ));
    }
}
