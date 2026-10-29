//! Stable public API for the second-generation Radau implementation.
//!
//! Preparation is explicit and separate from solving.  This makes frontend,
//! matrix layout, parameter-value rebinding, output retention, and telemetry
//! visible policy choices instead of implicit legacy behavior.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use nalgebra::{DMatrix, DVector};

use crate::numerical::logging::{SolverLogGuard, SolverLogLevel};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::ivp_telemetry::{IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetryMode};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    build_expr_legacy_banded_jacobian, build_expr_legacy_sparse_jacobian,
};
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};

use super::new::analytic_jacobian::AnalyticJacobianCallback;
use super::new::aot::AotPlan;
use super::new::callbacks::PreparedSymbolicCallbacks;
use super::new::config::{
    RadauAssembly as InternalAssembly, RadauConfig as InternalConfig,
    RadauExecution as InternalExecution, RadauJacobianSource as InternalJacobianSource,
    RadauMatrixLayout as InternalLayout,
};
use super::new::error::RadauError as InternalError;
use super::new::finite_difference::FiniteDifferenceJacobianCallback;
use super::new::native_callbacks::{NativeJacobianFn, NativeResidualCallback, NativeResidualFn};
use super::new::output::{RadauOutput, RadauOutputPolicy as InternalOutputPolicy};
use super::new::solver::{
    RadauSolveResult, try_solve_dense_with_callbacks_with_output,
    try_solve_symbolic_dense_with_output,
};
use super::new::telemetry::{RadauTelemetry, RadauTelemetryMode as InternalTelemetryMode};

/// Symbolic frontend used to prepare residual and Jacobian callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauFrontend {
    /// Compile the legacy expression graph into callbacks.
    ExprLegacy,
    /// Convert once to AtomView and keep symbolic work native thereafter.
    AtomViewNative,
}

/// Runtime family used by the public prepared solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauExecution {
    /// Interpret symbolic expressions through the selected Lambdify frontend.
    Lambdify,
    /// Generate and link a native callback artifact through the shared IVP AOT
    /// lifecycle. This route is explicit because it may invoke a toolchain.
    Aot,
    /// Evaluate caller-owned native residual/Jacobian closures.
    ///
    /// This execution mode is selected by [`RadauNativeSolver`] and is not a
    /// symbolic frontend. A residual-only model uses the typed finite-
    /// difference Jacobian adapter.
    NativeCallbacks,
}

/// Matrix storage selected once at preparation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauMatrixLayout {
    Dense,
    Sparse,
    /// Infer the compact band from symbolic state dependencies at preparation.
    BandedAuto,
    /// Use a caller-provided band as a validated structural hint.
    Banded {
        lower: usize,
        upper: usize,
    },
}

/// Jacobian source policy reserved by the numerical contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauJacobianSource {
    Analytic,
    Constant,
    FiniteDifference,
}

/// Collection policy for optional Radau telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauTelemetryMode {
    /// Disable counters and timing scopes.
    Off,
    /// Collect discrete operation counts without reading the clock.
    Counters,
    /// Collect wall-clock durations without discrete operation counts.
    Timings,
    /// Collect both discrete counters and wall-clock durations.
    Detailed,
}

/// Runtime policy for independent residual/Jacobian callback work.
///
/// `Sequential` is the compatibility default. `Parallel` is an explicit
/// force-parallel policy for sufficiently large callback work, while `Auto`
/// uses the shared LSODE2 calibration and falls back to sequential execution
/// when dispatch overhead is unlikely to amortize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauExecutionPolicy {
    Sequential,
    Parallel { min_work: usize },
    Auto { min_work: usize },
}

/// Retained trajectory policy.
#[derive(Debug, Clone, PartialEq)]
pub enum RadauOutputPolicy {
    /// Store no accepted-step history.
    FinalOnly,
    /// Store cubic continuous-output segments for arbitrary interpolation.
    Dense,
    /// Evaluate and retain only these requested times.
    Sampled(Vec<f64>),
}

/// Public AOT lifecycle and code-generation policy.
///
/// The generated-backend configuration is deliberately reused from the shared
/// IVP layer. Radau owns the numerical callback boundary, while that layer
/// owns artifact identity, cache provenance, compiler execution and linked
/// runtime lifetime.
#[derive(Debug, Clone)]
pub struct RadauAotConfig {
    pub generated: SymbolicIvpGeneratedBackendConfig,
}

impl Default for RadauAotConfig {
    fn default() -> Self {
        Self {
            generated: SymbolicIvpGeneratedBackendConfig::default(),
        }
    }
}

impl RadauAotConfig {
    /// Create a default AOT policy using the shared IVP defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build missing artifacts with the release compiler profile.
    pub fn build_if_missing_release(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self {
            generated: SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                output_parent_dir,
            ),
        }
    }

    /// Require a previously published artifact and never invoke a compiler.
    pub fn require_prebuilt() -> Self {
        Self {
            generated: SymbolicIvpGeneratedBackendConfig::require_prebuilt(),
        }
    }

    /// Always rebuild and link into a fresh release artifact location.
    pub fn rebuild_always_release(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated = SymbolicIvpGeneratedBackendConfig::new()
            .with_output_parent_dir(Some(output_parent_dir.into()))
            .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release,
            })
            .with_c_gcc();
        Self { generated }
    }

    /// Override the resolver used for artifact lookup and publication.
    pub fn with_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.generated = self.generated.with_resolver(resolver);
        self
    }

    /// Select the generated source backend used before compilation.
    pub fn with_codegen_backend(mut self, backend: AotCodegenBackend) -> Self {
        self.generated = self.generated.with_aot_codegen_backend(backend);
        self
    }

    /// Select the C compiler command used by the generated backend.
    pub fn with_c_compiler(mut self, compiler: impl Into<String>) -> Self {
        self.generated = self.generated.with_aot_c_compiler(compiler);
        self
    }

    /// Configure residual callback chunking for the generated artifact.
    pub fn with_residual_chunking(mut self, strategy: ResidualChunkingStrategy) -> Self {
        self.generated = self.generated.with_residual_chunking_strategy(strategy);
        self
    }

    /// Configure sparse and compact-banded Jacobian chunking.
    pub fn with_jacobian_chunking(mut self, strategy: SparseChunkingStrategy) -> Self {
        self.generated = self
            .generated
            .with_sparse_jacobian_chunking_strategy(strategy);
        self
    }
}

/// Public solver configuration.
#[derive(Debug, Clone)]
pub struct RadauConfig {
    pub t0: f64,
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: f64,
    pub first_step: Option<f64>,
    pub max_step: f64,
    pub max_steps: usize,
    pub max_newton_iterations: usize,
    pub max_retries: usize,
    pub execution: RadauExecution,
    pub frontend: RadauFrontend,
    pub matrix_layout: RadauMatrixLayout,
    pub jacobian_source: RadauJacobianSource,
    pub telemetry: RadauTelemetryMode,
    /// Logging threshold for solve operations; independent of telemetry.
    pub logging_level: SolverLogLevel,
    /// Callback scheduling policy shared by Lambdify and AOT routes.
    pub execution_policy: RadauExecutionPolicy,
    pub output: RadauOutputPolicy,
    /// Shared AOT lifecycle policy, required when `execution` is `Aot`.
    pub aot: Option<RadauAotConfig>,
}

impl Default for RadauConfig {
    fn default() -> Self {
        Self {
            t0: 0.0,
            t_bound: 1.0,
            rtol: 1e-3,
            atol: 1e-6,
            first_step: None,
            max_step: f64::INFINITY,
            max_steps: 1_000_000,
            max_newton_iterations: 6,
            max_retries: 12,
            execution: RadauExecution::Lambdify,
            frontend: RadauFrontend::ExprLegacy,
            matrix_layout: RadauMatrixLayout::Dense,
            jacobian_source: RadauJacobianSource::Analytic,
            telemetry: RadauTelemetryMode::Off,
            logging_level: SolverLogLevel::Off,
            execution_policy: RadauExecutionPolicy::Sequential,
            output: RadauOutputPolicy::FinalOnly,
            aot: None,
        }
    }
}

/// Symbolic IVP description consumed during preparation.
#[derive(Debug, Clone)]
pub struct RadauProblem {
    residual: Vec<Expr>,
    jacobian: Option<Vec<Expr>>,
    independent_variable: String,
    variables: Vec<String>,
    parameters: Vec<String>,
}

impl RadauProblem {
    /// Create a problem from residual expressions and ordered variable names.
    pub fn new(
        residual: Vec<Expr>,
        variables: Vec<String>,
        independent_variable: impl Into<String>,
    ) -> Self {
        Self {
            residual,
            jacobian: None,
            independent_variable: independent_variable.into(),
            variables,
            parameters: Vec::new(),
        }
    }

    /// Supply an explicit row-major symbolic Jacobian for Lambdify or AOT.
    ///
    /// AOT includes this payload in symbolic preparation and artifact identity;
    /// it never silently replaces an explicit Jacobian with one derived from
    /// the residual graph. ExprLegacy Lambdify derives a compact analytic state
    /// Jacobian from the residual when this method is not called and the
    /// analytic Jacobian source is selected.
    pub fn with_jacobian(mut self, jacobian: Vec<Expr>) -> Self {
        self.jacobian = Some(jacobian);
        self
    }

    /// Declare the ordered names of runtime-bound symbolic parameters.
    pub fn with_parameter_names(mut self, parameters: Vec<String>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Legacy name for [`Self::with_parameter_names`].
    #[deprecated(note = "use with_parameter_names; this declares names, not values")]
    pub fn with_parameters(self, parameters: Vec<String>) -> Self {
        self.with_parameter_names(parameters)
    }

    pub fn dimension(&self) -> usize {
        self.residual.len()
    }

    /// Number of state variables declared by this problem.
    ///
    /// This is distinct from [`Self::dimension`], which reports the number of
    /// residual equations. Valid square systems normally have equal counts.
    pub fn state_dimension(&self) -> usize {
        self.variables.len()
    }
}

/// Machine-readable public error category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauErrorKind {
    Configuration,
    Callback,
    NonFiniteCallback,
    Shape,
    Newton,
    StepBudget,
    StepUnderflow,
    StepControl,
    Unsupported,
    Workspace,
    LinearSolve,
    Output,
    Aot,
}

/// Typed public error without exposing internal migration modules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadauError {
    kind: RadauErrorKind,
    message: String,
}

impl RadauError {
    pub fn kind(&self) -> RadauErrorKind {
        self.kind
    }
}

impl fmt::Display for RadauError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl Error for RadauError {}

impl From<InternalError> for RadauError {
    fn from(error: InternalError) -> Self {
        let kind = match &error {
            InternalError::InvalidConfiguration(_) => RadauErrorKind::Configuration,
            InternalError::Callback { .. } => RadauErrorKind::Callback,
            InternalError::NonFiniteCallback { .. } => RadauErrorKind::NonFiniteCallback,
            InternalError::ShapeMismatch { .. } => RadauErrorKind::Shape,
            InternalError::NewtonFailure { .. } => RadauErrorKind::Newton,
            InternalError::StepBudgetExceeded { .. } => RadauErrorKind::StepBudget,
            InternalError::StepUnderflow { .. } => RadauErrorKind::StepUnderflow,
            InternalError::StepControlFailure { .. } => RadauErrorKind::StepControl,
            InternalError::UnsupportedRoute(_) => RadauErrorKind::Unsupported,
            InternalError::AotLifecycle { .. } => RadauErrorKind::Aot,
            InternalError::WorkspaceSizeOverflow { .. } => RadauErrorKind::Workspace,
            InternalError::LinearSolveFailure { .. } => RadauErrorKind::LinearSolve,
            InternalError::OutputNotAvailable
            | InternalError::OutputTimeOutsideInterval { .. }
            | InternalError::InvalidDenseOutput => RadauErrorKind::Output,
        };
        Self {
            kind,
            message: error.to_string(),
        }
    }
}

/// Public telemetry report. Keys are stable names suitable for text/JSON
/// export; timing scopes are diagnostic and parent/child scopes are not
/// additive. `allocations` counts explicit workspace/materialization events
/// observed by Radau, not every heap allocation in the process. `worker_count`
/// is the effective Rayon pool size; `configured_worker_count` is the positive
/// `RAYON_NUM_THREADS` request when present, or zero when the process did not
/// provide one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadauTelemetryScopeKind {
    /// An inclusive top-level diagnostic scope.
    Inclusive,
    /// A diagnostic scope measured inside the named parent scope.
    Child,
    /// A scope with no declared parent in this report.
    Standalone,
}

/// Machine-readable relationship for one timing key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RadauTelemetryScopeMetadata {
    pub kind: RadauTelemetryScopeKind,
    pub parent: Option<&'static str>,
}

/// One adaptive-controller attempt captured when timing telemetry is enabled.
///
/// The trace is intentionally diagnostic rather than a solver control API. It
/// makes frontend/backend trajectory differences explainable without treating
/// callback or linear timings as if they determined the accepted-step path.
#[derive(Debug, Clone, PartialEq)]
pub struct RadauAdaptiveStepTelemetry {
    pub h_abs: f64,
    pub error_norm: f64,
    pub newton_iterations: usize,
    pub accepted: bool,
    pub was_retry: bool,
    pub next_h_abs: f64,
    pub factor_invalidated: bool,
    pub jacobian_refreshed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RadauTelemetryReport {
    pub mode: RadauTelemetryMode,
    pub counters: BTreeMap<&'static str, u64>,
    pub timings_ms: BTreeMap<&'static str, f64>,
    pub timing_scopes: BTreeMap<&'static str, RadauTelemetryScopeMetadata>,
    pub aot_artifact_keys: Vec<String>,
    /// Actual initial absolute step used by the adaptive controller.
    pub initial_h_abs: Option<f64>,
    /// Per-attempt adaptive trace; populated only in `Timings` mode.
    pub adaptive_steps: Vec<RadauAdaptiveStepTelemetry>,
}

impl RadauTelemetryReport {
    fn timing_scope_metadata() -> BTreeMap<&'static str, RadauTelemetryScopeMetadata> {
        use RadauTelemetryScopeKind::{Child, Inclusive, Standalone};

        let mut scopes = BTreeMap::from([
            ("preparation_ms", Inclusive),
            ("callback_ms", Inclusive),
            ("newton_ms", Inclusive),
            ("linear_ms", Inclusive),
            ("output_ms", Standalone),
            ("step_control_ms", Standalone),
            ("parameter_rebind_ms", Standalone),
            ("invalidate_ms", Standalone),
            ("workspace_ms", Standalone),
        ])
        .into_iter()
        .map(|(key, kind)| (key, RadauTelemetryScopeMetadata { kind, parent: None }))
        .collect::<BTreeMap<_, _>>();

        let mut add_children = |parent: &'static str, keys: &[&'static str]| {
            for key in keys {
                scopes.insert(
                    key,
                    RadauTelemetryScopeMetadata {
                        kind: Child,
                        parent: Some(parent),
                    },
                );
            }
        };
        add_children(
            "preparation_ms",
            &[
                "expr_legacy_prepare_ms",
                "atom_conversion_ms",
                "atom_pattern_ms",
                "atom_differentiation_ms",
                "atom_evaluator_compile_ms",
                "atom_residual_prepare_ms",
                "atom_jacobian_prepare_ms",
                "atom_dependency_ms",
                "symbolic_jacobian_ms",
                "simplification_ms",
                "layout_ms",
                "backend_binding_ms",
                "aot_cache_lookup_ms",
                "aot_lowering_ms",
                "aot_source_generation_ms",
                "aot_materialize_ms",
                "aot_build_ms",
                "aot_link_ms",
                "aot_publication_ms",
                "aot_input_abi_ms",
                "aot_problem_key_ms",
                "aot_atom_plan_ms",
            ],
        );
        add_children(
            "callback_ms",
            &[
                "binding_ms",
                "residual_evaluation_ms",
                "jacobian_evaluation_ms",
                "jacobian_output_assembly_ms",
            ],
        );
        add_children(
            "linear_ms",
            &[
                "jacobian_assembly_ms",
                "factorization_ms",
                "real_solve_ms",
                "complex_solve_ms",
            ],
        );
        scopes
    }

    fn from_internal(telemetry: &RadauTelemetry) -> Self {
        let mode = match telemetry.mode() {
            InternalTelemetryMode::Off => RadauTelemetryMode::Off,
            InternalTelemetryMode::Counters => RadauTelemetryMode::Counters,
            InternalTelemetryMode::Timings => RadauTelemetryMode::Timings,
            InternalTelemetryMode::Detailed => RadauTelemetryMode::Detailed,
        };
        let c = telemetry.counters;
        let t = telemetry.timings;
        let counters = BTreeMap::from([
            ("residual_calls", c.residual_calls),
            ("jacobian_calls", c.jacobian_calls),
            ("linear_solves", c.linear_solves),
            ("newton_iterations", c.newton_iterations),
            ("error_estimates", c.error_estimates),
            ("accepted_steps", c.accepted_steps),
            ("rejected_steps", c.rejected_steps),
            ("allocations", c.allocations),
            ("copies", c.copies),
            ("argument_bindings", c.argument_bindings),
            // Keep evaluator-level counts public alongside solver-stage
            // counters; callback stories must not infer one from the other.
            ("residual_evaluations", c.residual_evaluations),
            ("jacobian_evaluations", c.jacobian_evaluations),
            ("finite_difference_probes", c.finite_difference_probes),
            ("jacobian_output_assemblies", c.jacobian_output_assemblies),
            ("parameter_rebinds", c.parameter_rebinds),
            ("jacobian_assemblies", c.jacobian_assemblies),
            ("factorizations", c.factorizations),
            ("real_solves", c.real_solves),
            ("complex_solves", c.complex_solves),
            ("output_writes", c.output_writes),
            ("workspace_resizes", c.workspace_resizes),
            ("frontend_preparations", c.frontend_preparations),
            ("aot_resolution_hits", c.aot_resolution_hits),
            ("aot_resolution_misses", c.aot_resolution_misses),
            ("aot_reconnects", c.aot_reconnects),
            ("aot_build_attempts", c.aot_build_attempts),
            ("aot_build_retries", c.aot_build_retries),
            ("aot_build_successes", c.aot_build_successes),
            ("aot_build_failures", c.aot_build_failures),
            ("aot_link_attempts", c.aot_link_attempts),
            ("aot_link_successes", c.aot_link_successes),
            ("aot_link_failures", c.aot_link_failures),
            ("aot_runtime_ready", c.aot_runtime_ready),
            ("parallel_dispatches", c.parallel_dispatches),
            ("sequential_dispatches", c.sequential_dispatches),
            ("worker_count", c.worker_count),
            ("configured_worker_count", c.configured_worker_count),
            ("aot_chunk_dispatches", c.aot_chunk_dispatches),
            ("aot_parallel_dispatches", c.aot_parallel_dispatches),
            ("aot_chunks", c.aot_chunks),
            ("aot_worker_callbacks", c.aot_worker_callbacks),
            (
                "parallel_dispatch_applicable",
                c.parallel_dispatch_applicable,
            ),
            ("aot_chunking_applicable", c.aot_chunking_applicable),
        ]);
        let timings_ms = BTreeMap::from([
            ("preparation_ms", t.preparation_ms),
            ("callback_ms", t.callback_ms),
            ("newton_ms", t.newton_ms),
            ("linear_ms", t.linear_ms),
            ("output_ms", t.output_ms),
            ("step_control_ms", t.step_control_ms),
            ("expr_legacy_prepare_ms", t.expr_legacy_prepare_ms),
            ("symbolic_jacobian_ms", t.symbolic_jacobian_ms),
            ("atom_conversion_ms", t.atom_conversion_ms),
            ("atom_pattern_ms", t.atom_pattern_ms),
            ("atom_differentiation_ms", t.atom_differentiation_ms),
            ("atom_evaluator_compile_ms", t.atom_evaluator_compile_ms),
            ("binding_ms", t.binding_ms),
            ("residual_evaluation_ms", t.residual_evaluation_ms),
            ("jacobian_evaluation_ms", t.jacobian_evaluation_ms),
            ("jacobian_output_assembly_ms", t.jacobian_output_assembly_ms),
            ("parameter_rebind_ms", t.parameter_rebind_ms),
            ("jacobian_assembly_ms", t.jacobian_assembly_ms),
            ("factorization_ms", t.factorization_ms),
            ("real_solve_ms", t.real_solve_ms),
            ("complex_solve_ms", t.complex_solve_ms),
            ("workspace_ms", t.workspace_ms),
            ("aot_cache_lookup_ms", t.aot_cache_lookup_ms),
            ("aot_lowering_ms", t.aot_lowering_ms),
            ("aot_source_generation_ms", t.aot_source_generation_ms),
            ("aot_materialize_ms", t.aot_materialize_ms),
            ("aot_build_ms", t.aot_build_ms),
            ("aot_link_ms", t.aot_link_ms),
            ("aot_publication_ms", t.aot_publication_ms),
            ("aot_input_abi_ms", t.aot_input_abi_ms),
            ("aot_problem_key_ms", t.aot_problem_key_ms),
            ("aot_atom_plan_ms", t.aot_atom_plan_ms),
        ]);
        Self {
            mode,
            counters,
            timings_ms,
            timing_scopes: Self::timing_scope_metadata(),
            aot_artifact_keys: telemetry.aot_artifact_keys.clone(),
            initial_h_abs: telemetry.initial_h_abs,
            adaptive_steps: telemetry
                .adaptive_steps
                .iter()
                .map(|step| RadauAdaptiveStepTelemetry {
                    h_abs: step.h_abs,
                    error_norm: step.error_norm,
                    newton_iterations: step.newton_iterations,
                    accepted: step.accepted,
                    was_retry: step.was_retry,
                    next_h_abs: step.next_h_abs,
                    factor_invalidated: step.factor_invalidated,
                    jacobian_refreshed: step.jacobian_refreshed,
                })
                .collect(),
        }
    }

    fn from_internal_pair(preparation: &RadauTelemetry, runtime: &RadauTelemetry) -> Self {
        let mut report = Self::from_internal(preparation);
        let runtime = Self::from_internal(runtime);
        for (key, value) in runtime.counters {
            let entry = report.counters.entry(key).or_default();
            if matches!(
                key,
                "parallel_dispatch_applicable"
                    | "aot_chunking_applicable"
                    | "worker_count"
                    | "configured_worker_count"
            ) {
                *entry = (*entry).max(value);
            } else {
                *entry += value;
            }
        }
        for (key, value) in runtime.timings_ms {
            *report.timings_ms.entry(key).or_default() += value;
        }
        for key in runtime.aot_artifact_keys {
            if !report
                .aot_artifact_keys
                .iter()
                .any(|existing| existing == &key)
            {
                report.aot_artifact_keys.push(key);
            }
        }
        report.adaptive_steps.extend(runtime.adaptive_steps);
        if report.initial_h_abs.is_none() {
            report.initial_h_abs = runtime.initial_h_abs;
        }
        report
    }
}

/// A completed solve and its optional post-processing data.
#[derive(Debug, Clone, PartialEq)]
pub struct RadauSolution {
    pub t: f64,
    pub y: Vec<f64>,
    pub attempts: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    output: RadauOutput,
    telemetry: RadauTelemetryReport,
}

impl RadauSolution {
    pub fn telemetry(&self) -> &RadauTelemetryReport {
        &self.telemetry
    }

    pub fn sample(&self, t: f64) -> Result<Vec<f64>, RadauError> {
        let mut output = vec![0.0; self.y.len()];
        self.sample_into(t, &mut output)?;
        Ok(output)
    }

    pub fn sample_into(&self, t: f64, output: &mut [f64]) -> Result<(), RadauError> {
        self.output.sample_into(t, output).map_err(Into::into)
    }

    /// Evaluate several output times into row-major `[time][component]`
    /// storage. This is convenient for plotting and post-processing.
    pub fn sample_many(&self, times: &[f64]) -> Result<Vec<f64>, RadauError> {
        self.output
            .sample_many(times, self.y.len())
            .map_err(Into::into)
    }

    /// Return the output points retained by this solve as a time-by-state
    /// matrix. Dense output returns accepted-step boundaries; sampled output
    /// returns its requested times; `FinalOnly` returns the terminal sample.
    pub fn output_trajectory(&self) -> Result<(DVector<f64>, DMatrix<f64>), RadauError> {
        let times = match &self.output {
            RadauOutput::FinalOnly => vec![self.t],
            RadauOutput::Sampled { times, .. } => times.clone(),
            RadauOutput::Dense(history) => {
                let Some(first) = history.segments().first() else {
                    return Ok((
                        DVector::from_vec(vec![self.t]),
                        DMatrix::from_row_slice(1, self.y.len(), &self.y),
                    ));
                };
                let mut times = Vec::with_capacity(history.segments().len() + 1);
                times.push(first.interval().0);
                times.extend(
                    history
                        .segments()
                        .iter()
                        .map(|segment| segment.interval().1),
                );
                times
            }
        };
        let flattened = match &self.output {
            RadauOutput::FinalOnly => self.y.clone(),
            RadauOutput::Dense(_) | RadauOutput::Sampled { .. } => self.sample_many(&times)?,
        };
        let values = DMatrix::from_row_slice(times.len(), self.y.len(), &flattened);
        Ok((DVector::from_vec(times), values))
    }
}

/// Prepared, reusable symbolic Radau model.
pub struct RadauSolver {
    callbacks: PreparedSymbolicCallbacks,
    config: InternalConfig,
    aot: Option<RadauAotConfig>,
    parameter_count: usize,
    preparation_telemetry: RadauTelemetry,
    last_solve_telemetry: Option<RadauTelemetryReport>,
    logging_level: SolverLogLevel,
}

/// Prepared native-callback Radau model.
///
/// The residual callback is always required. When `jacobian` is `Some`, the
/// supplied analytic dense Jacobian is used. When it is `None`, the solver
/// owns a component-wise forward-difference adapter and evaluates the same
/// residual callback for the Jacobian probes. Both modes use the new Radau
/// numerical core and its typed errors/telemetry.
pub struct RadauNativeSolver {
    config: InternalConfig,
    residual: NativeResidualFn,
    jacobian: Option<NativeJacobianFn>,
    logging_level: SolverLogLevel,
}

impl RadauNativeSolver {
    /// Prepare a native callback model without symbolic preparation.
    pub fn prepare<F, J>(
        config: RadauConfig,
        residual: F,
        jacobian: Option<J>,
    ) -> Result<Self, RadauError>
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        let residual: NativeResidualFn = std::sync::Arc::new(residual);
        let jacobian = jacobian.map(|callback| std::sync::Arc::new(callback) as NativeJacobianFn);
        Self::from_shared_callbacks(config, residual, jacobian)
    }

    /// Internal constructor used by `UniversalODESolver` after its existing
    /// callback builder has erased closure types into shared callbacks.
    pub(crate) fn from_shared_callbacks(
        config: RadauConfig,
        residual: NativeResidualFn,
        jacobian: Option<NativeJacobianFn>,
    ) -> Result<Self, RadauError> {
        if config.matrix_layout != RadauMatrixLayout::Dense {
            return Err(RadauError {
                kind: RadauErrorKind::Unsupported,
                message: "native Radau callbacks currently require Dense layout".to_owned(),
            });
        }
        let logging_level = config.logging_level;
        let mut internal = config.to_internal();
        internal.execution = InternalExecution::Native;
        internal.assembly = None;
        internal.jacobian_source = if jacobian.is_some() {
            InternalJacobianSource::Analytic
        } else {
            InternalJacobianSource::FiniteDifference
        };
        internal.validate().map_err(RadauError::from)?;
        Ok(Self {
            config: internal,
            residual,
            jacobian,
            logging_level,
        })
    }

    /// Solve from the supplied initial state, using analytic or FD Jacobians.
    pub fn solve(&mut self, y0: &[f64]) -> Result<RadauSolution, RadauError> {
        let _logging_scope = SolverLogGuard::enter(self.logging_level);
        crate::numerical::logging::solver_info!("Radau native solve started");
        let mut residual = NativeResidualCallback::new(self.residual.clone());
        let mut jacobian = match self.jacobian.clone() {
            Some(callback) => {
                NativeJacobianAdapter::Analytic(AnalyticJacobianCallback::new(callback))
            }
            None => NativeJacobianAdapter::FiniteDifference(FiniteDifferenceJacobianCallback::new(
                self.residual.clone(),
                self.config.atol,
                self.config.telemetry != InternalTelemetryMode::Off,
            )),
        };
        let solve_result = try_solve_dense_with_callbacks_with_output(
            &self.config,
            &mut residual,
            &mut jacobian,
            y0,
        )
        .map_err(RadauError::from);
        let (result, output, mut telemetry) = match solve_result {
            Ok(solved) => solved,
            Err(error) => {
                crate::numerical::logging::solver_warn!("Radau native solve failed: {error}");
                return Err(error);
            }
        };
        if let NativeJacobianAdapter::FiniteDifference(callback) = &jacobian {
            telemetry.counters.finite_difference_probes = callback.probes();
        }
        let solution = RadauSolution {
            t: result.t,
            y: result.y,
            attempts: result.attempts,
            accepted_steps: result.accepted_steps,
            rejected_steps: result.rejected_steps,
            output,
            telemetry: RadauTelemetryReport::from_internal(&telemetry),
        };
        crate::numerical::logging::solver_info!(
            "Radau native solve completed: accepted_steps={}, rejected_steps={}",
            solution.accepted_steps,
            solution.rejected_steps
        );
        Ok(solution)
    }

    /// Legacy alias for solving the same callback model from a supplied state.
    #[deprecated(note = "this is a fresh solve from y0, not continuation; use solve")]
    pub fn continue_with_initial_state(&mut self, y0: &[f64]) -> Result<RadauSolution, RadauError> {
        self.solve(y0)
    }

    /// Change the integration interval while retaining callback closures.
    pub fn restart(&mut self, t0: f64, t_bound: f64) -> Result<(), RadauError> {
        self.config.t0 = t0;
        self.config.t_bound = t_bound;
        self.config.validate().map_err(RadauError::from)
    }

    /// Return the public configuration represented by this native model.
    pub fn config(&self) -> RadauConfig {
        let mut config = RadauConfig::from_internal(&self.config);
        config.execution = RadauExecution::NativeCallbacks;
        config.frontend = RadauFrontend::ExprLegacy;
        config.jacobian_source = if self.jacobian.is_some() {
            RadauJacobianSource::Analytic
        } else {
            RadauJacobianSource::FiniteDifference
        };
        config.logging_level = self.logging_level;
        config
    }

    /// Return the active telemetry mode without cloning the full configuration.
    pub fn telemetry_mode(&self) -> RadauTelemetryMode {
        self.config.telemetry.into_public()
    }

    /// Return the active solver logging threshold.
    pub fn logging_level(&self) -> SolverLogLevel {
        self.logging_level
    }

    /// Set the logging threshold independently of telemetry collection.
    pub fn set_logging_level(&mut self, level: SolverLogLevel) {
        self.logging_level = level;
    }
}

enum NativeJacobianAdapter {
    Analytic(AnalyticJacobianCallback),
    FiniteDifference(FiniteDifferenceJacobianCallback),
}

impl super::new::callbacks::DenseJacobianCallback for NativeJacobianAdapter {
    fn eval_into(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), InternalError> {
        match self {
            Self::Analytic(callback) => callback.eval_into(t, y, out),
            Self::FiniteDifference(callback) => callback.eval_into(t, y, out),
        }
    }
}

impl RadauSolver {
    /// Prepare symbolic callbacks once. Subsequent solves with newly bound
    /// parameter values reuse the prepared frontend and do not repeat symbolic
    /// construction. Each solve starts from its explicitly supplied initial
    /// state; it is not solution-seeded continuation.
    pub fn prepare(problem: RadauProblem, mut config: RadauConfig) -> Result<Self, RadauError> {
        let symbolic_bandwidth = if matches!(
            config.matrix_layout,
            RadauMatrixLayout::BandedAuto | RadauMatrixLayout::Banded { .. }
        ) {
            Some(
                if let Some(jacobian) = problem.jacobian.as_deref() {
                    crate::symbolic::structural::infer_jacobian_bandwidth(
                        jacobian,
                        problem.dimension(),
                        &problem.variables,
                    )
                } else {
                    crate::symbolic::structural::infer_rowwise_bandwidth(
                        &problem.residual,
                        &problem.variables,
                    )
                }
                .map_err(|message| RadauError {
                    kind: RadauErrorKind::Configuration,
                    message,
                })?,
            )
        } else {
            None
        };
        if config.matrix_layout == RadauMatrixLayout::BandedAuto {
            let inferred = symbolic_bandwidth.expect("auto layout computed symbolic structure");
            config.matrix_layout = RadauMatrixLayout::Banded {
                lower: inferred.0,
                upper: inferred.1,
            };
        } else if let RadauMatrixLayout::Banded { lower, upper } = config.matrix_layout {
            crate::symbolic::structural::validate_bandwidth_hint(
                symbolic_bandwidth.expect("explicit band computed symbolic structure"),
                (lower, upper),
            )
            .map_err(|message| RadauError {
                kind: RadauErrorKind::Configuration,
                message,
            })?;
        }
        let logging_level = config.logging_level;
        let internal = config.to_internal();
        internal.validate().map_err(RadauError::from)?;
        let assembly = internal.assembly.ok_or_else(|| RadauError {
            kind: RadauErrorKind::Configuration,
            message: "a symbolic frontend is required".to_owned(),
        })?;
        let parameter_count = problem.parameters.len();
        let mut preparation_telemetry = RadauTelemetry::new(config.telemetry.into_internal());
        let mut stored_aot = config.aot.clone();
        let callbacks = if config.execution == RadauExecution::Aot {
            let aot = stored_aot.as_ref().ok_or_else(|| RadauError {
                kind: RadauErrorKind::Unsupported,
                message: "Radau AOT configuration is missing".to_owned(),
            })?;
            let prepared_aot = AotPlan::prepare_with_policy(
                assembly,
                internal.matrix_layout,
                problem.residual,
                problem.jacobian,
                problem.independent_variable,
                problem.variables,
                problem.parameters,
                aot.generated.clone(),
                &mut preparation_telemetry,
                internal.execution_policy,
            )?;
            if let Some(resolver) = prepared_aot.updated_resolver {
                if let Some(aot) = stored_aot.as_mut() {
                    aot.generated.resolver = Some(resolver);
                }
            }
            PreparedSymbolicCallbacks::Aot(prepared_aot.plan)
        } else {
            let variables = problem
                .variables
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let parameters = problem
                .parameters
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let derive_expr_jacobian = config.execution == RadauExecution::Lambdify
                && config.frontend == RadauFrontend::ExprLegacy
                && config.jacobian_source == RadauJacobianSource::Analytic
                && problem.jacobian.is_none();
            if derive_expr_jacobian {
                let symbolic_telemetry = IvpTelemetry::with_mode(match config.telemetry {
                    RadauTelemetryMode::Off => IvpTelemetryMode::Off,
                    RadauTelemetryMode::Counters => IvpTelemetryMode::Counters,
                    RadauTelemetryMode::Timings => IvpTelemetryMode::Timings,
                    RadauTelemetryMode::Detailed => IvpTelemetryMode::Detailed,
                });
                let callbacks = preparation_telemetry.measure_frontend(
                    super::new::telemetry::RadauFrontendStage::ExprLegacyPrepare,
                    || {
                        let jacobian = match internal.matrix_layout {
                            super::new::config::RadauMatrixLayout::Banded { lower, upper } => {
                                build_expr_legacy_banded_jacobian(
                                    &problem.residual,
                                    &problem.variables,
                                    (lower, upper),
                                    &symbolic_telemetry,
                                )
                            }
                            _ => build_expr_legacy_sparse_jacobian(
                                &problem.residual,
                                &problem.variables,
                                &symbolic_telemetry,
                            ),
                        };
                        PreparedSymbolicCallbacks::prepare_expr_sparse(
                            problem.residual,
                            jacobian,
                            &problem.independent_variable,
                            &variables,
                            &parameters,
                            internal.execution_policy,
                        )
                    },
                );
                preparation_telemetry
                    .absorb_expr_jacobian_preparation(&symbolic_telemetry.snapshot());
                callbacks?
            } else {
                PreparedSymbolicCallbacks::prepare_with_telemetry_and_policy(
                    assembly,
                    problem.residual,
                    problem.jacobian,
                    &problem.independent_variable,
                    &variables,
                    &parameters,
                    &mut preparation_telemetry,
                    internal.execution_policy,
                )?
            }
        };
        Ok(Self {
            callbacks,
            config: internal,
            aot: stored_aot,
            parameter_count,
            preparation_telemetry,
            last_solve_telemetry: None,
            logging_level,
        })
    }

    pub fn config(&self) -> RadauConfig {
        let mut config = RadauConfig::from_internal(&self.config);
        config.aot = self.aot.clone();
        config.logging_level = self.logging_level;
        config
    }

    /// Return the active telemetry mode without cloning the full configuration.
    pub fn telemetry_mode(&self) -> RadauTelemetryMode {
        self.config.telemetry.into_public()
    }

    /// Return the active solver logging threshold.
    pub fn logging_level(&self) -> SolverLogLevel {
        self.logging_level
    }

    /// Set the logging threshold independently of telemetry collection.
    pub fn set_logging_level(&mut self, level: SolverLogLevel) {
        self.logging_level = level;
    }

    /// Return the native report for the most recent solve attempt, including
    /// a failed attempt when runtime telemetry was produced before the error.
    pub fn last_solve_telemetry(&self) -> Option<&RadauTelemetryReport> {
        self.last_solve_telemetry.as_ref()
    }

    /// Solve with no parameters. Parameterized problems should use the
    /// explicit value-taking method so accidental zero bindings are visible.
    pub fn solve(&mut self, y0: &[f64]) -> Result<RadauSolution, RadauError> {
        self.try_solve_with_parameter_values(y0, &[])
    }

    /// Solve from `y0` after binding the ordered parameter values.
    ///
    /// The prepared symbolic callbacks are reused and exactly the supplied
    /// `y0` is used. For caller-managed parameter continuation, pass the
    /// previous solution's terminal state as the next `y0`; this solver does
    /// not retain or advance that state implicitly.
    pub fn try_solve_with_parameter_values(
        &mut self,
        y0: &[f64],
        parameters: &[f64],
    ) -> Result<RadauSolution, RadauError> {
        let _logging_scope = SolverLogGuard::enter(self.logging_level);
        crate::numerical::logging::solver_info!("Radau solve started");
        self.last_solve_telemetry = None;
        if parameters.len() != self.parameter_count {
            return Err(RadauError {
                kind: RadauErrorKind::Shape,
                message: format!(
                    "parameter length mismatch: expected {}, got {}",
                    self.parameter_count,
                    parameters.len()
                ),
            });
        }
        let mut session = self.callbacks.session_with_telemetry(self.config.telemetry);
        session.set_jacobian_policy(self.config.jacobian_source, self.config.atol);
        session
            .rebind_parameters(parameters)
            .map_err(RadauError::from)?;
        let solve_result = try_solve_symbolic_dense_with_output(&self.config, &mut session, y0)
            .map_err(RadauError::from);
        session.absorb_runtime_telemetry();
        let telemetry = RadauTelemetryReport::from_internal_pair(
            &self.preparation_telemetry,
            &session.telemetry(),
        );
        self.last_solve_telemetry = Some(telemetry.clone());
        match solve_result {
            Ok((result, output)) => {
                let solution = self.solution(result, output, telemetry);
                crate::numerical::logging::solver_info!(
                    "Radau solve completed: accepted_steps={}, rejected_steps={}",
                    solution.accepted_steps,
                    solution.rejected_steps
                );
                Ok(solution)
            }
            Err(error) => {
                crate::numerical::logging::solver_warn!("Radau solve failed: {error}");
                Err(error)
            }
        }
    }

    /// Starts a fresh solve over the configured time interval after binding
    /// new parameter values. The supplied initial state is always used.
    pub fn try_solve_with_parameter_values_in_interval(
        &mut self,
        t0: f64,
        y0: &[f64],
        t_bound: f64,
        parameters: &[f64],
    ) -> Result<RadauSolution, RadauError> {
        self.restart(t0, t_bound)?;
        self.try_solve_with_parameter_values(y0, parameters)
    }

    /// Compatibility alias for [`Self::try_solve_with_parameter_values`].
    #[deprecated(note = "use try_solve_with_parameter_values")]
    pub fn solve_with_parameter_values(
        &mut self,
        y0: &[f64],
        parameters: &[f64],
    ) -> Result<RadauSolution, RadauError> {
        self.try_solve_with_parameter_values(y0, parameters)
    }

    /// Legacy alias for [`Self::solve_with_parameter_values`].
    #[deprecated(note = "use solve_with_parameter_values; this binds values and starts from y0")]
    pub fn solve_with_parameters(
        &mut self,
        y0: &[f64],
        parameters: &[f64],
    ) -> Result<RadauSolution, RadauError> {
        self.try_solve_with_parameter_values(y0, parameters)
    }

    /// Legacy alias for [`Self::solve_with_parameter_values`].
    #[deprecated(note = "this starts from the supplied y0; use solve_with_parameter_values")]
    pub fn continue_with_parameters(
        &mut self,
        y0: &[f64],
        parameters: &[f64],
    ) -> Result<RadauSolution, RadauError> {
        self.try_solve_with_parameter_values(y0, parameters)
    }

    /// Change the integration interval while retaining prepared callbacks.
    pub fn restart(&mut self, t0: f64, t_bound: f64) -> Result<(), RadauError> {
        self.config.t0 = t0;
        self.config.t_bound = t_bound;
        self.config.validate().map_err(RadauError::from)
    }

    fn solution(
        &self,
        result: RadauSolveResult,
        output: RadauOutput,
        telemetry: RadauTelemetryReport,
    ) -> RadauSolution {
        RadauSolution {
            t: result.t,
            y: result.y,
            attempts: result.attempts,
            accepted_steps: result.accepted_steps,
            rejected_steps: result.rejected_steps,
            output,
            telemetry,
        }
    }
}

impl RadauConfig {
    /// Set solver logging independently of telemetry collection.
    pub fn with_logging_level(mut self, level: SolverLogLevel) -> Self {
        self.logging_level = level;
        self
    }

    fn to_internal(&self) -> InternalConfig {
        InternalConfig {
            t0: self.t0,
            t_bound: self.t_bound,
            rtol: self.rtol,
            atol: self.atol,
            first_step: self.first_step,
            max_step: self.max_step,
            max_steps: self.max_steps,
            max_newton_iterations: self.max_newton_iterations,
            max_retries: self.max_retries,
            execution: match self.execution {
                RadauExecution::Lambdify => InternalExecution::Lambdify,
                RadauExecution::Aot => InternalExecution::Aot,
                RadauExecution::NativeCallbacks => InternalExecution::Native,
            },
            assembly: (self.execution != RadauExecution::NativeCallbacks).then(|| {
                match self.frontend {
                    RadauFrontend::ExprLegacy => InternalAssembly::ExprLegacy,
                    RadauFrontend::AtomViewNative => InternalAssembly::AtomViewNative,
                }
            }),
            matrix_layout: self.matrix_layout.into_internal(),
            jacobian_source: self.jacobian_source.into_internal(),
            telemetry: self.telemetry.into_internal(),
            execution_policy: self.execution_policy.into_internal(),
            output: self.output.clone().into_internal(),
        }
    }

    fn from_internal(config: &InternalConfig) -> Self {
        Self {
            t0: config.t0,
            t_bound: config.t_bound,
            rtol: config.rtol,
            atol: config.atol,
            first_step: config.first_step,
            max_step: config.max_step,
            max_steps: config.max_steps,
            max_newton_iterations: config.max_newton_iterations,
            max_retries: config.max_retries,
            execution: match config.execution {
                InternalExecution::Aot => RadauExecution::Aot,
                InternalExecution::Lambdify => RadauExecution::Lambdify,
                InternalExecution::Native => RadauExecution::NativeCallbacks,
            },
            frontend: match config.assembly {
                Some(InternalAssembly::AtomViewNative) => RadauFrontend::AtomViewNative,
                _ => RadauFrontend::ExprLegacy,
            },
            matrix_layout: config.matrix_layout.into_public(),
            jacobian_source: config.jacobian_source.into_public(),
            telemetry: config.telemetry.into_public(),
            logging_level: SolverLogLevel::Off,
            execution_policy: config.execution_policy.into_public(),
            output: config.output.clone().into_public(),
            aot: None,
        }
    }
}

impl RadauExecutionPolicy {
    fn into_internal(self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::Sequential => IvpLambdifyExecutionPolicy::Sequential,
            Self::Parallel { min_work } => IvpLambdifyExecutionPolicy::Parallel { min_work },
            Self::Auto { min_work } => IvpLambdifyExecutionPolicy::Auto { min_work },
        }
    }
}

impl IvpLambdifyExecutionPolicy {
    fn into_public(self) -> RadauExecutionPolicy {
        match self {
            Self::Sequential => RadauExecutionPolicy::Sequential,
            Self::Parallel { min_work } => RadauExecutionPolicy::Parallel { min_work },
            Self::Auto { min_work } => RadauExecutionPolicy::Auto { min_work },
        }
    }
}

impl RadauMatrixLayout {
    fn into_internal(self) -> InternalLayout {
        match self {
            Self::Dense => InternalLayout::Dense,
            Self::Sparse => InternalLayout::Sparse,
            Self::BandedAuto => unreachable!("auto layout is resolved before internal config"),
            Self::Banded { lower, upper } => InternalLayout::Banded { lower, upper },
        }
    }
}

impl InternalLayout {
    fn into_public(self) -> RadauMatrixLayout {
        match self {
            Self::Dense => RadauMatrixLayout::Dense,
            Self::Sparse => RadauMatrixLayout::Sparse,
            Self::Banded { lower, upper } => RadauMatrixLayout::Banded { lower, upper },
        }
    }
}

impl RadauJacobianSource {
    fn into_internal(self) -> InternalJacobianSource {
        match self {
            Self::Analytic => InternalJacobianSource::Analytic,
            Self::Constant => InternalJacobianSource::Constant,
            Self::FiniteDifference => InternalJacobianSource::FiniteDifference,
        }
    }
}

impl InternalJacobianSource {
    fn into_public(self) -> RadauJacobianSource {
        match self {
            Self::Analytic => RadauJacobianSource::Analytic,
            Self::Constant => RadauJacobianSource::Constant,
            Self::FiniteDifference => RadauJacobianSource::FiniteDifference,
        }
    }
}

impl RadauTelemetryMode {
    fn into_internal(self) -> InternalTelemetryMode {
        match self {
            Self::Off => InternalTelemetryMode::Off,
            Self::Counters => InternalTelemetryMode::Counters,
            Self::Timings => InternalTelemetryMode::Timings,
            Self::Detailed => InternalTelemetryMode::Detailed,
        }
    }
}

impl InternalTelemetryMode {
    fn into_public(self) -> RadauTelemetryMode {
        match self {
            Self::Off => RadauTelemetryMode::Off,
            Self::Counters => RadauTelemetryMode::Counters,
            Self::Timings => RadauTelemetryMode::Timings,
            Self::Detailed => RadauTelemetryMode::Detailed,
        }
    }
}

impl RadauOutputPolicy {
    fn into_internal(self) -> InternalOutputPolicy {
        match self {
            Self::FinalOnly => InternalOutputPolicy::FinalOnly,
            Self::Dense => InternalOutputPolicy::Dense,
            Self::Sampled(times) => InternalOutputPolicy::Sampled(times),
        }
    }
}

impl InternalOutputPolicy {
    fn into_public(self) -> RadauOutputPolicy {
        match self {
            Self::FinalOnly => RadauOutputPolicy::FinalOnly,
            Self::Dense => RadauOutputPolicy::Dense,
            Self::Sampled(times) => RadauOutputPolicy::Sampled(times),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_report_exposes_non_additive_scope_relationships() {
        let telemetry = RadauTelemetry::new(InternalTelemetryMode::Timings);
        let report = RadauTelemetryReport::from_internal(&telemetry);

        assert_eq!(
            report.timing_scopes["preparation_ms"],
            RadauTelemetryScopeMetadata {
                kind: RadauTelemetryScopeKind::Inclusive,
                parent: None,
            }
        );
        assert_eq!(
            report.timing_scopes["atom_pattern_ms"],
            RadauTelemetryScopeMetadata {
                kind: RadauTelemetryScopeKind::Child,
                parent: Some("preparation_ms"),
            }
        );
        assert_eq!(
            report.timing_scopes["factorization_ms"],
            RadauTelemetryScopeMetadata {
                kind: RadauTelemetryScopeKind::Child,
                parent: Some("linear_ms"),
            }
        );
        assert_eq!(
            report.timing_scopes["workspace_ms"],
            RadauTelemetryScopeMetadata {
                kind: RadauTelemetryScopeKind::Standalone,
                parent: None,
            }
        );
    }
}
