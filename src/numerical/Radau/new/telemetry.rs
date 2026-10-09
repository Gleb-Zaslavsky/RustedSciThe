//! Optional, zero-cost-when-disabled telemetry contracts.
//!
//! Stage timing must distinguish inclusive and exclusive scopes and cover
//! symbolic preparation, backend binding, callbacks, Newton, linear solves,
//! allocations, copies, continuation, and publication.
//!
//! `Off` is the production default: it does not read the clock, inspect Vec
//! capacities, or maintain counters.  `Counters` records operation counts
//! without `Instant` calls.  `Timings` adds wall-clock scopes.  Parent scopes
//! such as `linear_ms` are inclusive diagnostics and must not be added to
//! child scopes such as factorization and solve time by report consumers.

use super::error::RadauStage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Preparation stages imported from the shared symbolic frontend pipeline.
pub(crate) enum RadauFrontendStage {
    ExprLegacyPrepare,
    AtomConversion,
    AtomPattern,
    AtomDifferentiation,
    AtomEvaluatorCompile,
    AotPreparation,
}

/// Repeated callback work.  Binding is intentionally separate from evaluator
/// time so a report can distinguish copied arguments from symbolic execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RadauCallbackStage {
    ArgumentBinding,
    ResidualEvaluation,
    JacobianEvaluation,
    JacobianOutputAssembly,
    ParameterRebind,
}

/// Repeated numerical backend work.  These are inclusive scopes: for example,
/// factorization does not include Jacobian callback or shifted-matrix assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RadauLinearStage {
    JacobianAssembly,
    Factorization,
    RealSolve,
    ComplexSolve,
    Invalidate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Controls the cost of Radau instrumentation.
pub(crate) enum RadauTelemetryMode {
    #[default]
    Off,
    Counters,
    Timings,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Exact repeated-operation counts collected when telemetry is enabled.
pub(crate) struct RadauCounters {
    /// Number of residual callback invocations.
    pub residual_calls: u64,
    /// Number of Jacobian callback invocations.
    pub jacobian_calls: u64,
    /// Number of all real/complex linear solves.
    pub linear_solves: u64,
    /// Number of Newton iterations across step attempts.
    pub newton_iterations: u64,
    /// Number of embedded error estimates.
    pub error_estimates: u64,
    /// Number of accepted steps.
    pub accepted_steps: u64,
    /// Number of rejected step attempts.
    pub rejected_steps: u64,
    /// Explicit workspace/materialization allocation events observed by the
    /// instrumentation hooks. This is not a process-wide allocator count.
    pub allocations: u64,
    /// Explicit buffer copy events.
    pub copies: u64,
    /// Argument-buffer binding operations.
    pub argument_bindings: u64,
    /// Residual evaluator operations.
    pub residual_evaluations: u64,
    /// Jacobian evaluator operations.
    pub jacobian_evaluations: u64,
    /// Residual probes performed internally by the finite-difference fallback.
    pub finite_difference_probes: u64,
    /// Sparse/banded Jacobian projection operations.
    pub jacobian_output_assemblies: u64,
    /// Parameter value rebind operations.
    pub parameter_rebinds: u64,
    /// Shifted Jacobian assembly operations.
    pub jacobian_assemblies: u64,
    /// Numeric factorization operations.
    pub factorizations: u64,
    /// Real shifted-system solves.
    pub real_solves: u64,
    /// Complex shifted-system solves.
    pub complex_solves: u64,
    /// Factorization invalidation operations.
    pub invalidations: u64,
    /// Elements written to caller-owned output buffers.
    pub output_writes: u64,
    /// Workspace resize operations.
    pub workspace_resizes: u64,
    /// Symbolic frontend preparation operations.
    pub frontend_preparations: u64,
    /// AOT resolver/cache hits and misses during cold preparation.
    pub aot_resolution_hits: u64,
    pub aot_resolution_misses: u64,
    pub aot_reconnects: u64,
    pub aot_build_attempts: u64,
    pub aot_build_retries: u64,
    pub aot_build_successes: u64,
    pub aot_build_failures: u64,
    pub aot_link_attempts: u64,
    pub aot_link_successes: u64,
    pub aot_link_failures: u64,
    pub aot_runtime_ready: u64,
    /// High-level callback dispatch decisions made by Radau.
    pub parallel_dispatches: u64,
    pub sequential_dispatches: u64,
    /// Rayon worker count observed for the selected callback policy. This is
    /// metadata, not a count of worker callback invocations. It is the
    /// effective pool size returned by Rayon, not necessarily the value of
    /// `RAYON_NUM_THREADS` requested by the process.
    pub worker_count: u64,
    /// Positive `RAYON_NUM_THREADS` when the process supplied it, otherwise
    /// zero. This is a request/provenance field, not an observed worker count.
    pub configured_worker_count: u64,
    /// Chunk and worker activity imported from the shared AOT runtime.
    pub aot_chunk_dispatches: u64,
    pub aot_parallel_dispatches: u64,
    pub aot_chunks: u64,
    pub aot_worker_callbacks: u64,
    /// Sticky applicability marker: at least one callback had multiple
    /// independent jobs for which parallel dispatch was meaningful.
    pub parallel_dispatch_applicable: u64,
    /// Sticky applicability marker: the selected AOT route exposed chunked
    /// callback work. A zero chunk count is otherwise ambiguous in reports.
    pub aot_chunking_applicable: u64,
}

/// One adaptive controller decision retained by timing telemetry.
///
/// This trace is intentionally absent when telemetry is `Off` or `Counters`.
/// It is a diagnostic tool for separating callback/linear work from a change
/// in the accepted-step trajectory between frontends or execution routes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RadauAdaptiveStepTrace {
    pub h_abs: f64,
    pub error_norm: f64,
    pub accepted: bool,
    pub was_retry: bool,
    pub next_h_abs: f64,
    pub factor_invalidated: bool,
    pub jacobian_refreshed: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
/// Diagnostic wall-clock scopes; child scopes are not additive with parents.
pub(crate) struct RadauTimings {
    /// Inclusive preparation scope.
    pub preparation_ms: f64,
    /// Inclusive callback scope.
    pub callback_ms: f64,
    /// Inclusive Newton scope.
    pub newton_ms: f64,
    /// Inclusive linear scope.
    pub linear_ms: f64,
    /// Output assembly scope.
    pub output_ms: f64,
    /// Error estimate and step-control scope.
    pub step_control_ms: f64,
    /// ExprLegacy preparation scope.
    pub expr_legacy_prepare_ms: f64,
    /// Expr-to-Atom conversion scope.
    pub atom_conversion_ms: f64,
    /// Atom dependency/pattern construction scope.
    pub atom_pattern_ms: f64,
    /// Native Atom differentiation scope.
    pub atom_differentiation_ms: f64,
    /// Native evaluator compilation scope.
    pub atom_evaluator_compile_ms: f64,
    /// Callback argument binding scope.
    pub binding_ms: f64,
    /// Residual evaluator scope.
    pub residual_evaluation_ms: f64,
    /// Jacobian evaluator scope.
    pub jacobian_evaluation_ms: f64,
    /// Jacobian layout projection scope.
    pub jacobian_output_assembly_ms: f64,
    /// Parameter rebind scope.
    pub parameter_rebind_ms: f64,
    /// Shifted Jacobian assembly scope.
    pub jacobian_assembly_ms: f64,
    /// Numeric factorization scope.
    pub factorization_ms: f64,
    /// Real solve scope.
    pub real_solve_ms: f64,
    /// Complex solve scope.
    pub complex_solve_ms: f64,
    /// Factor invalidation scope.
    pub invalidate_ms: f64,
    /// Workspace resize scope.
    pub workspace_ms: f64,
    /// Atom residual preparation scope.
    pub atom_residual_prepare_ms: f64,
    /// Atom Jacobian preparation scope.
    pub atom_jacobian_prepare_ms: f64,
    /// Atom dependency analysis scope.
    pub atom_dependency_ms: f64,
    /// Shared symbolic Jacobian scope.
    pub symbolic_jacobian_ms: f64,
    /// Symbolic simplification scope.
    pub simplification_ms: f64,
    /// Layout construction scope.
    pub layout_ms: f64,
    /// Backend binding scope.
    pub backend_binding_ms: f64,
    /// Shared AOT lifecycle stages. These are cold scopes and are not additive
    /// with the inclusive `preparation_ms` parent.
    pub aot_cache_lookup_ms: f64,
    pub aot_lowering_ms: f64,
    pub aot_source_generation_ms: f64,
    pub aot_materialize_ms: f64,
    pub aot_build_ms: f64,
    pub aot_link_ms: f64,
    pub aot_publication_ms: f64,
    pub aot_input_abi_ms: f64,
    pub aot_problem_key_ms: f64,
    pub aot_atom_plan_ms: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Opt-in counters and timings owned by one preparation/session.
pub(crate) struct RadauTelemetry {
    mode: RadauTelemetryMode,
    pub counters: RadauCounters,
    pub timings: RadauTimings,
    pub aot_artifact_keys: Vec<String>,
    /// Actual first integration step after explicit configuration or probing.
    pub initial_h_abs: Option<f64>,
    pub adaptive_steps: Vec<RadauAdaptiveStepTrace>,
}

impl RadauTelemetry {
    /// Create telemetry with the requested cost policy.
    pub(crate) const fn new(mode: RadauTelemetryMode) -> Self {
        Self {
            mode,
            counters: RadauCounters {
                residual_calls: 0,
                jacobian_calls: 0,
                linear_solves: 0,
                newton_iterations: 0,
                error_estimates: 0,
                accepted_steps: 0,
                rejected_steps: 0,
                allocations: 0,
                copies: 0,
                argument_bindings: 0,
                residual_evaluations: 0,
                jacobian_evaluations: 0,
                finite_difference_probes: 0,
                jacobian_output_assemblies: 0,
                parameter_rebinds: 0,
                jacobian_assemblies: 0,
                factorizations: 0,
                real_solves: 0,
                complex_solves: 0,
                invalidations: 0,
                output_writes: 0,
                workspace_resizes: 0,
                frontend_preparations: 0,
                aot_resolution_hits: 0,
                aot_resolution_misses: 0,
                aot_reconnects: 0,
                aot_build_attempts: 0,
                aot_build_retries: 0,
                aot_build_successes: 0,
                aot_build_failures: 0,
                aot_link_attempts: 0,
                aot_link_successes: 0,
                aot_link_failures: 0,
                aot_runtime_ready: 0,
                parallel_dispatches: 0,
                sequential_dispatches: 0,
                worker_count: 0,
                configured_worker_count: 0,
                aot_chunk_dispatches: 0,
                aot_parallel_dispatches: 0,
                aot_chunks: 0,
                aot_worker_callbacks: 0,
                parallel_dispatch_applicable: 0,
                aot_chunking_applicable: 0,
            },
            timings: RadauTimings {
                preparation_ms: 0.0,
                callback_ms: 0.0,
                newton_ms: 0.0,
                linear_ms: 0.0,
                output_ms: 0.0,
                step_control_ms: 0.0,
                expr_legacy_prepare_ms: 0.0,
                atom_conversion_ms: 0.0,
                atom_pattern_ms: 0.0,
                atom_differentiation_ms: 0.0,
                atom_evaluator_compile_ms: 0.0,
                binding_ms: 0.0,
                residual_evaluation_ms: 0.0,
                jacobian_evaluation_ms: 0.0,
                jacobian_output_assembly_ms: 0.0,
                parameter_rebind_ms: 0.0,
                jacobian_assembly_ms: 0.0,
                factorization_ms: 0.0,
                real_solve_ms: 0.0,
                complex_solve_ms: 0.0,
                invalidate_ms: 0.0,
                workspace_ms: 0.0,
                atom_residual_prepare_ms: 0.0,
                atom_jacobian_prepare_ms: 0.0,
                atom_dependency_ms: 0.0,
                symbolic_jacobian_ms: 0.0,
                simplification_ms: 0.0,
                layout_ms: 0.0,
                backend_binding_ms: 0.0,
                aot_cache_lookup_ms: 0.0,
                aot_lowering_ms: 0.0,
                aot_source_generation_ms: 0.0,
                aot_materialize_ms: 0.0,
                aot_build_ms: 0.0,
                aot_link_ms: 0.0,
                aot_publication_ms: 0.0,
                aot_input_abi_ms: 0.0,
                aot_problem_key_ms: 0.0,
                aot_atom_plan_ms: 0.0,
            },
            aot_artifact_keys: Vec::new(),
            initial_h_abs: None,
            adaptive_steps: Vec::new(),
        }
    }

    pub(crate) const fn mode(&self) -> RadauTelemetryMode {
        self.mode
    }

    /// Change mode and reset the snapshot at a lifecycle boundary.
    ///
    /// Resetting prevents a report from combining counters collected under
    /// incompatible cost policies, especially when a prepared session is
    /// reused for continuation.
    pub(crate) fn set_mode(&mut self, mode: RadauTelemetryMode) {
        if self.mode != mode {
            *self = Self::new(mode);
        }
    }

    /// Count a high-level solver stage when instrumentation is enabled.
    pub(crate) fn count_stage(&mut self, stage: RadauStage) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        match stage {
            RadauStage::Residual => self.counters.residual_calls += 1,
            RadauStage::Jacobian => self.counters.jacobian_calls += 1,
            RadauStage::LinearSolve => self.counters.linear_solves += 1,
            _ => {}
        }
    }

    /// Count an accepted or rejected step attempt.
    pub(crate) fn count_step(&mut self, accepted: bool) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        if accepted {
            self.counters.accepted_steps += 1;
        } else {
            self.counters.rejected_steps += 1;
        }
    }

    /// Retain one adaptive controller decision in the diagnostic trace.
    pub(crate) fn record_adaptive_step(&mut self, trace: RadauAdaptiveStepTrace) {
        if self.mode == RadauTelemetryMode::Timings {
            self.adaptive_steps.push(trace);
        }
    }

    /// Record the actual initial absolute step used by the controller.
    pub(crate) fn record_initial_step(&mut self, h_abs: f64) {
        if self.mode == RadauTelemetryMode::Timings {
            self.initial_h_abs = Some(h_abs);
        }
    }

    /// Count one callback operation by its detailed stage.
    pub(crate) fn count_callback(&mut self, stage: RadauCallbackStage) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        match stage {
            RadauCallbackStage::ArgumentBinding => self.counters.argument_bindings += 1,
            RadauCallbackStage::ResidualEvaluation => self.counters.residual_evaluations += 1,
            RadauCallbackStage::JacobianEvaluation => self.counters.jacobian_evaluations += 1,
            RadauCallbackStage::JacobianOutputAssembly => {
                self.counters.jacobian_output_assemblies += 1
            }
            RadauCallbackStage::ParameterRebind => self.counters.parameter_rebinds += 1,
        }
    }

    /// Count one linear backend operation by its detailed stage.
    pub(crate) fn count_linear(&mut self, stage: RadauLinearStage) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        match stage {
            RadauLinearStage::JacobianAssembly => self.counters.jacobian_assemblies += 1,
            RadauLinearStage::Factorization => self.counters.factorizations += 1,
            RadauLinearStage::RealSolve => self.counters.real_solves += 1,
            RadauLinearStage::ComplexSolve => self.counters.complex_solves += 1,
            RadauLinearStage::Invalidate => self.counters.invalidations += 1,
        }
    }

    /// Count one Newton iteration without reading the clock in counter mode.
    pub(crate) fn count_newton_iteration(&mut self) {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.newton_iterations += 1;
        }
    }

    /// Count one embedded error estimate.
    pub(crate) fn count_error_estimate(&mut self) {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.error_estimates += 1;
        }
    }

    /// Count successful writes into caller-owned output storage.
    pub(crate) fn count_output_writes(&mut self, count: usize) {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.output_writes = self.counters.output_writes.saturating_add(count as u64);
        }
    }

    /// Count a materialized buffer copy.
    pub(crate) fn count_copy(&mut self) {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.copies += 1;
        }
    }

    /// Count one observed capacity growth or materialization allocation.
    pub(crate) fn count_allocation(&mut self) {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.allocations += 1;
        }
    }

    /// Count a workspace resize operation.
    pub(crate) fn count_workspace_resize(&mut self) {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.workspace_resizes += 1;
        }
    }

    /// Count one symbolic frontend preparation.
    pub(crate) fn count_frontend_preparation(&mut self) {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.frontend_preparations += 1;
        }
    }

    /// Record one policy decision without timing or allocation overhead when
    /// telemetry is disabled. `tasks` is the number of independent callback
    /// jobs, not the number of scalar output elements in one job.
    pub(crate) fn record_policy_dispatch(
        &mut self,
        policy: crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy,
        work: usize,
        tasks: usize,
    ) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        if tasks > 1 {
            self.counters.parallel_dispatch_applicable = 1;
        }
        self.counters.worker_count = self
            .counters
            .worker_count
            .max(rayon::current_num_threads() as u64);
        if let Some(configured) = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
        {
            self.counters.configured_worker_count =
                self.counters.configured_worker_count.max(configured);
        }
        if policy.should_parallel_with_tasks(work, tasks) {
            self.counters.parallel_dispatches += 1;
        } else {
            self.counters.sequential_dispatches += 1;
        }
    }

    /// Add optional workspace timing to the current snapshot.
    pub(crate) fn add_workspace_timing_ms(&mut self, elapsed_ms: f64) {
        if self.mode == RadauTelemetryMode::Timings {
            self.timings.workspace_ms += elapsed_ms;
        }
    }

    /// Add optional Newton timing to the current snapshot.
    pub(crate) fn add_newton_timing_ms(&mut self, elapsed_ms: f64) {
        if self.mode == RadauTelemetryMode::Timings {
            self.timings.newton_ms += elapsed_ms;
        }
    }

    /// Add optional output assembly timing to the current snapshot.
    pub(crate) fn add_output_timing_ms(&mut self, elapsed_ms: f64) {
        if self.mode == RadauTelemetryMode::Timings {
            self.timings.output_ms += elapsed_ms;
        }
    }

    /// Add optional embedded error/control timing to the current snapshot.
    pub(crate) fn add_step_control_timing_ms(&mut self, elapsed_ms: f64) {
        if self.mode == RadauTelemetryMode::Timings {
            self.timings.step_control_ms += elapsed_ms;
        }
    }

    /// Import shared symbolic IVP preparation timings without double-running
    /// work.  This is called once per prepared AtomNative plan; warm callback
    /// scopes remain measured by the Radau session itself.
    pub(crate) fn absorb_ivp_preparation(
        &mut self,
        snapshot: &crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot,
    ) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        use crate::symbolic::ivp_telemetry::IvpColdStage;
        let ms = |stage| snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1_000.0;
        self.timings.atom_conversion_ms += ms(IvpColdStage::ExprToAtom);
        self.timings.atom_pattern_ms += ms(IvpColdStage::SparsePattern);
        self.timings.layout_ms += ms(IvpColdStage::LayoutPlanning);
        self.timings.atom_dependency_ms += ms(IvpColdStage::AtomDependencyAnalysis);
        self.timings.symbolic_jacobian_ms += ms(IvpColdStage::SymbolicJacobian);
        self.timings.atom_differentiation_ms += ms(IvpColdStage::SymbolicDifferentiation);
        self.timings.simplification_ms += ms(IvpColdStage::Simplification);
        self.timings.atom_residual_prepare_ms += ms(IvpColdStage::AtomResidualPreparation);
        self.timings.atom_jacobian_prepare_ms += ms(IvpColdStage::AtomJacobianPreparation);
        self.timings.atom_evaluator_compile_ms +=
            ms(IvpColdStage::NativeJacobianEvaluatorPreparation);
        self.timings.backend_binding_ms += ms(IvpColdStage::BackendBinding);
        self.timings.aot_cache_lookup_ms += ms(IvpColdStage::AotCacheLookup);
        self.timings.aot_lowering_ms += ms(IvpColdStage::AotLowering);
        self.timings.aot_source_generation_ms += ms(IvpColdStage::AotSourceGeneration);
        self.timings.aot_materialize_ms += ms(IvpColdStage::AotMaterialization);
        self.timings.aot_build_ms += ms(IvpColdStage::AotBuild);
        self.timings.aot_link_ms += ms(IvpColdStage::AotLink);
        self.timings.aot_publication_ms += ms(IvpColdStage::AotPublication);
        self.timings.aot_input_abi_ms += ms(IvpColdStage::AotInputAbiPreparation);
        self.timings.aot_problem_key_ms += ms(IvpColdStage::AotProblemKeyConstruction);
        self.timings.aot_atom_plan_ms += ms(IvpColdStage::AotAtomPlanPreparation);
        self.counters.aot_resolution_hits += snapshot.aot_resolution_hits;
        self.counters.aot_resolution_misses += snapshot.aot_resolution_misses;
        self.counters.aot_reconnects += snapshot.aot_reconnects;
        self.counters.aot_build_attempts += snapshot.aot_build_attempts;
        self.counters.aot_build_retries += snapshot.aot_build_retries;
        self.counters.aot_build_successes += snapshot.aot_build_successes;
        self.counters.aot_build_failures += snapshot.aot_build_failures;
        self.counters.aot_link_attempts += snapshot.aot_link_attempts;
        self.counters.aot_link_successes += snapshot.aot_link_successes;
        self.counters.aot_link_failures += snapshot.aot_link_failures;
        self.counters.aot_runtime_ready += snapshot.aot_runtime_ready;
        for key in &snapshot.aot_artifact_keys {
            if !self
                .aot_artifact_keys
                .iter()
                .any(|existing| existing == key)
            {
                self.aot_artifact_keys.push(key.clone());
            }
        }
    }

    /// Merge numerical stages collected by the reusable solve workspace.
    ///
    /// Symbolic callback preparation owns callback telemetry, while the
    /// numerical workspace owns Newton, output, and linear-backend stages.
    /// Combining them only at the lifecycle boundary keeps the inner loop
    /// free of cross-owner bookkeeping and gives callers one complete report.
    /// The caller must drop the runner first when Rust borrow rules otherwise
    /// prevent simultaneous access to the callback session and workspace.
    pub(crate) fn absorb_runtime(&mut self, snapshot: &RadauTelemetry) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        self.counters.linear_solves += snapshot.counters.linear_solves;
        self.counters.newton_iterations += snapshot.counters.newton_iterations;
        self.counters.error_estimates += snapshot.counters.error_estimates;
        self.counters.allocations += snapshot.counters.allocations;
        self.counters.copies += snapshot.counters.copies;
        self.counters.jacobian_assemblies += snapshot.counters.jacobian_assemblies;
        self.counters.factorizations += snapshot.counters.factorizations;
        self.counters.real_solves += snapshot.counters.real_solves;
        self.counters.complex_solves += snapshot.counters.complex_solves;
        self.counters.invalidations += snapshot.counters.invalidations;
        self.counters.output_writes += snapshot.counters.output_writes;
        self.counters.workspace_resizes += snapshot.counters.workspace_resizes;
        self.counters.parallel_dispatches += snapshot.counters.parallel_dispatches;
        self.counters.sequential_dispatches += snapshot.counters.sequential_dispatches;
        self.counters.worker_count = self
            .counters
            .worker_count
            .max(snapshot.counters.worker_count);
        self.counters.configured_worker_count = self
            .counters
            .configured_worker_count
            .max(snapshot.counters.configured_worker_count);
        self.counters.aot_chunk_dispatches += snapshot.counters.aot_chunk_dispatches;
        self.counters.aot_parallel_dispatches += snapshot.counters.aot_parallel_dispatches;
        self.counters.aot_chunks += snapshot.counters.aot_chunks;
        self.counters.aot_worker_callbacks += snapshot.counters.aot_worker_callbacks;
        self.timings.newton_ms += snapshot.timings.newton_ms;
        self.timings.linear_ms += snapshot.timings.linear_ms;
        self.timings.output_ms += snapshot.timings.output_ms;
        self.timings.step_control_ms += snapshot.timings.step_control_ms;
        self.timings.jacobian_assembly_ms += snapshot.timings.jacobian_assembly_ms;
        self.timings.factorization_ms += snapshot.timings.factorization_ms;
        self.timings.real_solve_ms += snapshot.timings.real_solve_ms;
        self.timings.complex_solve_ms += snapshot.timings.complex_solve_ms;
        self.timings.invalidate_ms += snapshot.timings.invalidate_ms;
        self.timings.workspace_ms += snapshot.timings.workspace_ms;
        if self.mode == RadauTelemetryMode::Timings {
            if self.initial_h_abs.is_none() {
                self.initial_h_abs = snapshot.initial_h_abs;
            }
            self.adaptive_steps
                .extend_from_slice(&snapshot.adaptive_steps);
        }
    }

    /// Merge callbacks evaluated by a direct native step adapter.
    ///
    /// Direct callbacks use a short-lived snapshot because the numerical
    /// workspace is borrowed by the same step. Keeping this merge explicit
    /// avoids interior mutability and preserves the zero-cost `Off` route.
    pub(crate) fn absorb_callback_runtime(&mut self, snapshot: &RadauTelemetry) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        self.counters.residual_calls += snapshot.counters.residual_calls;
        self.counters.jacobian_calls += snapshot.counters.jacobian_calls;
        self.counters.residual_evaluations += snapshot.counters.residual_evaluations;
        self.counters.jacobian_evaluations += snapshot.counters.jacobian_evaluations;
        self.timings.callback_ms += snapshot.timings.callback_ms;
        self.timings.residual_evaluation_ms += snapshot.timings.residual_evaluation_ms;
        self.timings.jacobian_evaluation_ms += snapshot.timings.jacobian_evaluation_ms;
    }

    /// Import warm execution counters from the shared generated-IVP runtime.
    /// Cold stages are intentionally excluded because preparation telemetry is
    /// already absorbed separately and parent/child scopes are not additive.
    pub(crate) fn absorb_ivp_runtime(
        &mut self,
        snapshot: &crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot,
    ) {
        if self.mode == RadauTelemetryMode::Off {
            return;
        }
        // The session records the public policy decision once per callback.
        // The shared AOT runtime also exposes these fields, but importing them
        // here would count the same decision twice. Import only AOT-specific
        // chunk/worker activity below.
        self.counters.aot_chunk_dispatches += snapshot.aot_chunk_dispatches;
        self.counters.aot_parallel_dispatches += snapshot.aot_parallel_dispatches;
        self.counters.aot_chunks += snapshot.aot_chunks;
        self.counters.aot_worker_callbacks += snapshot.aot_worker_callbacks;
        self.counters.worker_count = self
            .counters
            .worker_count
            .max(snapshot.lambdify_worker_count as u64);
        if snapshot.aot_chunks > 0 {
            self.counters.aot_chunking_applicable = 1;
        }
        if snapshot.aot_parallel_dispatches > 0 {
            self.counters.parallel_dispatch_applicable = 1;
        }
        self.counters.copies = self.counters.copies.saturating_add(snapshot.copies);
        if self.mode == RadauTelemetryMode::Timings {
            use crate::symbolic::ivp_telemetry::IvpWarmStage;
            let ms = |stage| snapshot.warm_stage(stage).elapsed.as_secs_f64() * 1_000.0;
            self.timings.binding_ms += ms(IvpWarmStage::ArgumentBinding);
            self.timings.residual_evaluation_ms += ms(IvpWarmStage::ResidualEvaluation);
            self.timings.jacobian_evaluation_ms += ms(IvpWarmStage::JacobianCallback);
        }
    }

    /// Measure one callback operation, or only count it in counter mode.
    pub(crate) fn measure_callback<T, F>(&mut self, stage: RadauCallbackStage, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.count_callback(stage);
        if self.mode != RadauTelemetryMode::Timings {
            return operation();
        }
        let started = std::time::Instant::now();
        let result = operation();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        self.timings.callback_ms += elapsed_ms;
        match stage {
            RadauCallbackStage::ArgumentBinding => self.timings.binding_ms += elapsed_ms,
            RadauCallbackStage::ResidualEvaluation => {
                self.timings.residual_evaluation_ms += elapsed_ms
            }
            RadauCallbackStage::JacobianEvaluation => {
                self.timings.jacobian_evaluation_ms += elapsed_ms
            }
            RadauCallbackStage::JacobianOutputAssembly => {
                self.timings.jacobian_output_assembly_ms += elapsed_ms
            }
            RadauCallbackStage::ParameterRebind => self.timings.parameter_rebind_ms += elapsed_ms,
        }
        result
    }

    /// Measure one linear operation, or only count it in counter mode.
    pub(crate) fn measure_linear<T, F>(&mut self, stage: RadauLinearStage, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.count_linear(stage);
        if self.mode != RadauTelemetryMode::Timings {
            return operation();
        }
        let started = std::time::Instant::now();
        let result = operation();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        self.timings.linear_ms += elapsed_ms;
        match stage {
            RadauLinearStage::JacobianAssembly => self.timings.jacobian_assembly_ms += elapsed_ms,
            RadauLinearStage::Factorization => self.timings.factorization_ms += elapsed_ms,
            RadauLinearStage::RealSolve => self.timings.real_solve_ms += elapsed_ms,
            RadauLinearStage::ComplexSolve => self.timings.complex_solve_ms += elapsed_ms,
            RadauLinearStage::Invalidate => self.timings.invalidate_ms += elapsed_ms,
        }
        result
    }

    /// Add a high-level timing scope while preserving inclusive-scope semantics.
    pub(crate) fn add_timing_ms(&mut self, stage: RadauStage, elapsed_ms: f64) {
        if self.mode != RadauTelemetryMode::Timings {
            return;
        }
        match stage {
            RadauStage::Preparation => self.timings.preparation_ms += elapsed_ms,
            RadauStage::Residual | RadauStage::Jacobian => self.timings.callback_ms += elapsed_ms,
            RadauStage::Newton => self.timings.newton_ms += elapsed_ms,
            RadauStage::LinearSolve => self.timings.linear_ms += elapsed_ms,
            RadauStage::Output => self.timings.output_ms += elapsed_ms,
            RadauStage::StepControl => {}
        }
    }

    /// Time symbolic preparation only when timing telemetry is enabled.
    /// Measure one symbolic preparation stage.
    pub(crate) fn measure_frontend<T, F>(&mut self, stage: RadauFrontendStage, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        if self.mode != RadauTelemetryMode::Off {
            self.counters.frontend_preparations += 1;
        }
        if self.mode != RadauTelemetryMode::Timings {
            return operation();
        }
        let started = std::time::Instant::now();
        let result = operation();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        self.timings.preparation_ms += elapsed_ms;
        match stage {
            RadauFrontendStage::ExprLegacyPrepare => {
                self.timings.expr_legacy_prepare_ms += elapsed_ms
            }
            RadauFrontendStage::AtomConversion => self.timings.atom_conversion_ms += elapsed_ms,
            RadauFrontendStage::AtomPattern => self.timings.atom_pattern_ms += elapsed_ms,
            RadauFrontendStage::AtomDifferentiation => {
                self.timings.atom_differentiation_ms += elapsed_ms
            }
            RadauFrontendStage::AtomEvaluatorCompile => {
                self.timings.atom_evaluator_compile_ms += elapsed_ms
            }
            RadauFrontendStage::AotPreparation => {}
        }
        result
    }
}
