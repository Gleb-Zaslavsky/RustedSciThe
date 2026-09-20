use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::Instant;

use crate::Utils::logger::save_matrix_to_file;
use crate::Utils::plots::plots;
use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, LinearSolveTiming, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::*;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::BvpDerivativeScheme;
use crate::numerical::BVP_Damp::factor_runtime::prepare_factor_owner_runtime;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, ApplyFrozenGeneratedSolverState,
    BandedGeneratedBackendMode, BuildFrozenSolverRequest, FrozenGeneratedSolverState,
    FrozenSolverBuildRequest, GeneratedBackendConfig, SparseGeneratedBackendMode,
    try_generate_and_apply_frozen_solver_state,
};
use crate::numerical::BVP_Damp::prepared_runtime::{
    BvpPreparedResourceSnapshot, BvpPreparedRuntime, BvpRuntimeRevision, PreparedPlanFingerprint,
    fingerprint_bytes, fingerprint_debug,
};
use crate::numerical::BVP_Damp::solver_common::{
    DEFAULT_MAX_ITERATIONS, cleanup_registered_aot_artifacts, default_dense_method_name,
    default_forward_scheme_name, default_placeholder_y, default_sparse_method_name,
    frozen_point_mesh,
};
use crate::numerical::BVP_Damp::telemetry::{
    BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode, BvpTelemetryRecorder, BvpTelemetrySnapshot,
};
use crate::somelinalg::banded::LinearSolverConfig;
use crate::symbolic::bvp::telemetry::{
    BvpGenerationTelemetrySnapshot, BvpLambdifyTelemetry, BvpLambdifyTelemetryMode,
};
use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_functions_BVP::{
    BvpBackendIntegrationError, BvpMatrixBackend, BvpSymbolicAssemblyBackend,
};

use chrono::Local;

use log::info;

use simplelog::*;

use std::fs::File;

/// User-facing setup options for the frozen BVP solver.
#[derive(Clone)]
pub struct FrozenSolverOptions {
    /// Residual discretization scheme name.
    pub scheme: String,
    /// Nonlinear solver strategy name.
    pub strategy: String,
    /// Optional strategy-specific parameters.
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
    /// Optional linear-system method override.
    pub linear_sys_method: Option<String>,
    /// Matrix backend/method selector.
    pub method: String,
    /// Absolute convergence tolerance.
    pub tolerance: f64,
    /// Maximum nonlinear iterations.
    pub max_iterations: usize,
    /// Generated-backend configuration used by sparse solver paths.
    pub generated_backend_config: GeneratedBackendConfig,
}

/// Runtime counters, timers, and generated-backend diagnostics for Frozen Newton.
#[derive(Clone, Debug)]
pub struct FrozenBvpStatistics {
    pub counters: HashMap<String, usize>,
    pub timers: HashMap<String, String>,
    pub diagnostics: HashMap<String, String>,
    /// Typed telemetry for new consumers; legacy maps above remain compatible.
    pub telemetry: BvpTelemetrySnapshot,
}

impl FrozenSolverOptions {
    /// Creates frozen solver options from explicit values.
    pub fn new(
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,
        max_iterations: usize,
    ) -> Self {
        Self {
            scheme: default_forward_scheme_name(),
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
            generated_backend_config: GeneratedBackendConfig::default(),
        }
    }

    /// Attaches an explicit generated-backend configuration.
    pub fn with_generated_backend_config(mut self, config: GeneratedBackendConfig) -> Self {
        self.generated_backend_config = config;
        self
    }

    /// Selects the runtime telemetry level for Lambdify callbacks.
    ///
    /// The default is [`BvpLambdifyTelemetryMode::Off`]. Use `Counters` for
    /// cheap call counts or `Detailed` when callback wall-clock timings are
    /// needed for a diagnostic/story run.
    pub fn with_lambdify_telemetry_mode(mut self, mode: BvpLambdifyTelemetryMode) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_lambdify_telemetry_mode(mode);
        self
    }

    /// Selects typed solver decision logging for diagnostic runs.
    pub fn with_bvp_logging_mode(mut self, mode: BvpLoggingMode) -> Self {
        self.generated_backend_config = self.generated_backend_config.with_bvp_logging_mode(mode);
        self
    }

    /// Configures bounded typed decision logging for the solver runtime.
    pub fn with_bvp_logging_config(mut self, config: BvpLoggingConfig) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_bvp_logging_config(config);
        self
    }

    /// Selects whether solver counters and stage timers are collected.
    pub fn with_bvp_telemetry_mode(mut self, mode: BvpTelemetryMode) -> Self {
        self.generated_backend_config = self.generated_backend_config.with_bvp_telemetry_mode(mode);
        self
    }

    /// Returns options with an explicit AtomView AOT optimization profile.
    ///
    /// The default is `AtomOptimizationProfile::Full`, which preserves the
    /// historical CSE-enabled pipeline. Diagnostic profiles such as `NoCse`
    /// are useful for correctness and performance A/B checks.
    pub fn with_atom_optimization_profile(mut self, profile: AtomOptimizationProfile) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_atom_optimization_profile(profile);
        self
    }

    /// Selects the residual discretization scheme with a typed public API.
    ///
    /// The stored value is still the legacy string consumed by the symbolic
    /// discretization layer, but user code should prefer this method over raw
    /// `"forward"` / `"trapezoid"` strings.
    pub fn with_scheme(mut self, scheme: BvpDerivativeScheme) -> Self {
        self.scheme = scheme.as_legacy_str().to_string();
        self
    }

    /// Selects the legacy forward derivative discretization.
    pub fn forward_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Forward)
    }

    /// Selects the trapezoid derivative discretization.
    pub fn trapezoid_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Trapezoid)
    }

    /// Compatibility escape hatch for legacy/custom scheme strings.
    pub fn with_scheme_name(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = scheme.into();
        self
    }

    /// Overrides the native linear solver configuration used by generated banded callbacks.
    pub fn with_banded_linear_solver_config(mut self, config: LinearSolverConfig) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_banded_linear_solver_config(config);
        self
    }

    /// Overrides detailed nonlinear strategy parameters.
    pub fn with_strategy_params(
        mut self,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
    ) -> Self {
        self.strategy_params = strategy_params;
        self
    }

    /// Overrides the absolute convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Overrides the nonlinear iteration limit.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Overrides the linear-system method.
    pub fn with_linear_sys_method(mut self, linear_sys_method: Option<String>) -> Self {
        self.linear_sys_method = linear_sys_method;
        self
    }

    /// Attaches a high-level sparse generated-backend mode.
    pub fn with_sparse_generated_backend_mode(mut self, mode: SparseGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
        self
    }

    /// Attaches a high-level banded generated-backend mode.
    ///
    /// Banded modes route generated callbacks through native `Banded` matrix
    /// assembly and faithful LAPACK-style banded LU with `refine = 0`.
    pub fn with_banded_generated_backend_mode(mut self, mode: BandedGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_banded_mode(mode);
        self
    }

    /// Selects the symbolic assembly backend used before lambdify/AOT lowering.
    pub fn with_symbolic_assembly_backend(mut self, backend: BvpSymbolicAssemblyBackend) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_symbolic_assembly_backend(backend);
        self
    }

    /// Selects the matrix backend through the typed API.
    ///
    /// The historical `method: String` field is retained for compatibility,
    /// while the normalized generated-backend configuration becomes the source
    /// of truth for new callers.
    pub fn with_matrix_backend(mut self, backend: MatrixBackend) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_matrix_backend_override(backend);
        self
    }

    /// Creates production-oriented sparse frozen solver options with standard defaults.
    ///
    /// This is the preferred starting point for most sparse/frozen BVP users.
    pub fn sparse_frozen() -> Self {
        Self::default().with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Creates production-oriented banded frozen solver options with standard defaults.
    ///
    /// This selects generated `Banded` matrix callbacks and faithful
    /// LAPACK-style native banded LU (`refine = 0`) while preserving the frozen
    /// Newton strategy.
    pub fn banded_frozen() -> Self {
        Self::default().with_banded_generated_backend_mode(BandedGeneratedBackendMode::Defaults)
    }

    /// Creates production-oriented dense frozen solver options with standard defaults.
    pub fn dense_frozen() -> Self {
        Self {
            method: default_dense_method_name(),
            ..Self::default()
        }
    }

    /// Creates production-oriented dense solver options for the naive strategy.
    pub fn dense_naive() -> Self {
        Self {
            strategy: "Naive".to_string(),
            strategy_params: None,
            ..Self::dense_frozen()
        }
    }

    /// Uses the standard sparse generated-backend defaults.
    pub fn with_sparse_generated_backend_defaults(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Uses the standard banded generated-backend defaults.
    pub fn with_banded_generated_backend_defaults(self) -> Self {
        self.with_banded_generated_backend_mode(BandedGeneratedBackendMode::Defaults)
    }

    /// Uses lambdify callbacks with generated `Banded` matrix assembly.
    pub fn with_banded_lambdify(self) -> Self {
        self.with_banded_generated_backend_mode(BandedGeneratedBackendMode::Lambdify)
    }

    /// Builds a banded release AOT backend on demand.
    pub fn with_banded_aot_build_if_missing_release(self) -> Self {
        self.with_banded_generated_backend_mode(BandedGeneratedBackendMode::BuildIfMissingRelease)
    }

    /// Requires a prebuilt sparse generated backend.
    pub fn with_sparse_aot_require_prebuilt(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt)
    }

    /// Builds a sparse release AOT backend on demand.
    pub fn with_sparse_aot_build_if_missing_release(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease)
    }

    /// Uses AtomView symbolic assembly plus on-demand `gcc`-compiled sparse C AOT.
    ///
    /// Prefer this when runtime throughput matters more than startup latency.
    pub fn with_sparse_atomview_c_gcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand `tcc`-compiled sparse C AOT.
    ///
    /// Prefer this for practical repeated-solve workflows on the same large BVP.
    pub fn with_sparse_atomview_c_tcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        )
    }

    /// User-facing alias for the currently recommended repeated-solve compiled path.
    pub fn with_sparse_atomview_for_repeated_solves(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_for_repeated_solves(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand `gcc`-compiled banded C AOT.
    pub fn with_banded_atomview_c_gcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand `tcc`-compiled banded C AOT.
    pub fn with_banded_atomview_c_tcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand Zig-compiled banded AOT.
    pub fn with_banded_atomview_zig(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
        )
    }

    /// User-facing alias for the currently recommended repeated-solve banded path.
    pub fn with_banded_atomview_for_repeated_solves(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_for_repeated_solves(),
        )
    }
}

impl Default for FrozenSolverOptions {
    fn default() -> Self {
        Self {
            scheme: default_forward_scheme_name(),
            strategy: "Frozen".to_string(),
            strategy_params: Some(HashMap::from([("Frozen_naive".to_string(), None)])),
            linear_sys_method: None,
            method: default_sparse_method_name(),
            tolerance: 1e-6,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            generated_backend_config: GeneratedBackendConfig::default(),
        }
    }
}

pub struct NRBVP {
    pub eq_system: Vec<Expr>,
    pub initial_guess: DMatrix<f64>,
    pub values: Vec<String>,
    pub arg: String,
    pub param_names: Vec<String>,
    pub param_values: Option<Vec<f64>>,
    pub BorderConditions: HashMap<String, Vec<(usize, f64)>>,
    pub t0: f64,
    pub t_end: f64,
    pub n_steps: usize,
    pub scheme: String,
    pub strategy: String,
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
    pub linear_sys_method: Option<String>,
    pub method: String,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_error: f64,
    pub result: Option<DVector<f64>>,
    pub x_mesh: DVector<f64>,

    pub fun: Box<dyn Fun>,
    pub jac: Option<Box<dyn Jac>>,
    pub p: f64,
    pub y: Box<dyn VectorType>,
    m: usize, // iteration counter without jacobian recalculation
    /// Internal prepared factor for direct Dense/faer Frozen solves.
    /// Banded keeps ownership inside `BandedMatrixType`; legacy callers never
    /// see this field or the factor-owner runtime.
    /// Common prepared-runtime owner for reusable Dense/faer factors.
    ///
    /// Banded ownership remains inside `BandedMatrixType`; callbacks and
    /// mesh/layout are separate migration slices.
    factor_owner: BvpPreparedRuntime,
    prepared_runtime_revision: BvpRuntimeRevision,
    jac_recalc: bool,
    error_old: f64,
    variable_string: Vec<String>, // vector of indexed variable names
    bandwidth: (usize, usize),
    generated_backend_config: GeneratedBackendConfig,
    generated_backend_selected_backend: Option<SelectedBackendKind>,
    generated_backend_runtime_diagnostics: HashMap<String, String>,
    telemetry_counters: BvpTelemetryRecorder,
    generation_telemetry: Option<BvpGenerationTelemetrySnapshot>,
    atom_discretization_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot>,
    legacy_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    atom_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    direct_banded_jacobian_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpDirectJacobianTelemetry>,
    parameter_binding: Option<crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle>,
    custom_timer: CustomTimer,
    no_reports: bool,
}

impl ApplyFrozenGeneratedSolverState for NRBVP {
    fn apply_generated_solver_state(&mut self, state: FrozenGeneratedSolverState) {
        if let Some(updated_resolver) = state.updated_resolver.clone() {
            self.generated_backend_config.resolver = Some(updated_resolver);
        }
        self.fun = state.fun;
        self.jac = state.jac;
        self.variable_string = state.variable_string;
        self.bandwidth = state.bandwidth;
        self.generated_backend_selected_backend = Some(state.selected_backend);
        let backend_fallback = matches!(
            self.generated_backend_config
                .effective_backend_policy(&self.method),
            BackendSelectionPolicy::PreferAotThenLambdify
                | BackendSelectionPolicy::PreferAotThenNumeric
        ) && matches!(
            state.selected_backend,
            SelectedBackendKind::Lambdify | SelectedBackendKind::Numeric
        );
        self.telemetry_counters.record_backend_selection(
            match state.selected_backend {
                SelectedBackendKind::Numeric => 1,
                SelectedBackendKind::Lambdify => 2,
                SelectedBackendKind::AotCompiled => 3,
                SelectedBackendKind::AotRegisteredButNotBuilt => 4,
                SelectedBackendKind::AotMissing => 5,
            },
            backend_fallback,
        );
        self.generated_backend_runtime_diagnostics = state.runtime_diagnostics;
        self.generation_telemetry = state.generation_telemetry;
        self.atom_discretization_telemetry = state.atom_discretization_telemetry;
        self.legacy_lambdify_telemetry = state.legacy_lambdify_telemetry;
        self.atom_lambdify_telemetry = state.atom_lambdify_telemetry;
        self.direct_banded_jacobian_telemetry = state.direct_banded_jacobian_telemetry;
        self.parameter_binding = state.parameter_binding;
        self.invalidate_linear_runtime();
        self.prepared_runtime_revision
            .mark_prepared_with_fingerprint(self.prepared_plan_fingerprint());
    }
}

impl BuildFrozenSolverRequest for NRBVP {
    fn build_solver_request(&self) -> FrozenSolverBuildRequest {
        let h = (self.t_end - self.t0) / self.n_steps as f64;
        let effective_method = self.generated_backend_config.effective_method(&self.method);
        FrozenSolverBuildRequest {
            eq_system: self.eq_system.clone(),
            values: self.values.clone(),
            arg: self.arg.clone(),
            param_names: (!self.param_names.is_empty()).then(|| self.param_names.clone()),
            param_values: self.param_values.clone(),
            t0: self.t0,
            n_steps: Some(self.n_steps),
            h: Some(h),
            mesh: None,
            border_conditions: self.BorderConditions.clone(),
            scheme: self.scheme.clone(),
            method: effective_method.clone(),
            bandwidth: None,
            backend_policy: self
                .generated_backend_config
                .effective_backend_policy(&effective_method),
            resolver: self.generated_backend_config.resolver.clone(),
            aot_execution_policy: self.generated_backend_config.aot_execution_policy.clone(),
            aot_build_policy: self.generated_backend_config.aot_build_policy,
            aot_compile_config: self.generated_backend_config.aot_compile_config.clone(),
            aot_codegen_backend: self.generated_backend_config.aot_codegen_backend,
            aot_c_compiler: self.generated_backend_config.aot_c_compiler.clone(),
            aot_chunking_policy: self.generated_backend_config.aot_chunking_policy,
            atom_optimization_profile: self.generated_backend_config.atom_optimization_profile,
            symbolic_assembly_backend: self.generated_backend_config.symbolic_assembly_backend,
            matrix_backend_override: self.generated_backend_config.matrix_backend_override,
            banded_linear_solver_config: self.generated_backend_config.banded_linear_solver_config,
            lambdify_telemetry_mode: self.generated_backend_config.lambdify_telemetry_mode,
            lambdify_execution_policy: self.generated_backend_config.lambdify_execution_policy,
        }
    }
}

impl NRBVP {
    /// Returns the resource state paired with the prepared Frozen runtime.
    pub(crate) fn prepared_resource_snapshot_for_diagnostics(&self) -> BvpPreparedResourceSnapshot {
        self.factor_owner.resource_snapshot()
    }

    /// Computes the identity captured by a prepared Frozen runtime.
    ///
    /// This check is performed only at the prepared-solve boundary. It catches
    /// direct writes to historical public fields without adding work to the
    /// residual/Jacobian callbacks.
    fn prepared_plan_fingerprint(&self) -> PreparedPlanFingerprint {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        fingerprint_debug(&mut hash, &self.eq_system);
        fingerprint_debug(&mut hash, &self.initial_guess.as_slice());
        fingerprint_debug(&mut hash, &self.values);
        fingerprint_debug(&mut hash, &self.arg);
        fingerprint_debug(&mut hash, &self.BorderConditions);
        fingerprint_debug(&mut hash, &(self.t0, self.t_end, self.n_steps));
        fingerprint_debug(&mut hash, &self.scheme);
        fingerprint_debug(&mut hash, &self.strategy);
        fingerprint_debug(&mut hash, &self.strategy_params);
        fingerprint_debug(&mut hash, &self.linear_sys_method);
        fingerprint_debug(&mut hash, &self.method);
        fingerprint_debug(&mut hash, &self.tolerance);
        fingerprint_debug(&mut hash, &self.max_iterations);
        fingerprint_debug(&mut hash, &self.param_names);
        fingerprint_debug(&mut hash, &self.param_values);
        fingerprint_debug(&mut hash, &self.x_mesh.as_slice());
        fingerprint_debug(&mut hash, &self.generated_backend_config);
        fingerprint_debug(&mut hash, &self.generated_backend_selected_backend);
        fingerprint_bytes(&mut hash, b"bvp-frozen-prepared-plan-v1");
        PreparedPlanFingerprint(hash)
    }

    /// Drops all state derived from the current Jacobian.
    ///
    /// Frozen reuse is valid only while the callback inputs and discretization
    /// remain unchanged. Parameter/continuation changes must therefore clear
    /// both the matrix and its owned factor, rather than merely forcing the
    /// next iteration to recalculate the Jacobian.
    fn invalidate_linear_runtime(&mut self) {
        self.prepared_runtime_revision.factor_invalidated();
        let had_owned_factor = self.factor_owner.has_factor();
        self.factor_owner.invalidate_numeric_jacobian();
        if had_owned_factor {
            self.telemetry_counters.record_factorization_invalidation();
        }
        self.jac_recalc = true;
        self.m = 0;
        self.error_old = 0.0;
    }

    #[inline]
    fn effective_runtime_method(&self) -> String {
        self.generated_backend_config.effective_method(&self.method)
    }

    pub fn new(
        eq_system: Vec<Expr>,        //
        initial_guess: DMatrix<f64>, // initial guess
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations
    ) -> NRBVP {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        let y0 = default_placeholder_y();

        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        let x_mesh = frozen_point_mesh(t0, t_end, n_steps);
        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        NRBVP {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            param_names: Vec::new(),
            param_values: None,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            scheme: BvpDerivativeScheme::Forward.as_legacy_str().to_string(),
            tolerance,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            max_iterations,
            max_error: 0.0,
            result: None,
            x_mesh,
            fun: boxed_fun,
            jac: None,
            p: 0.0,
            y: y0,
            m: 0,
            factor_owner: BvpPreparedRuntime::new(),
            prepared_runtime_revision: BvpRuntimeRevision::default(),
            jac_recalc: true,
            error_old: 0.0,
            variable_string: Vec::new(), // vector of indexed variable names
            bandwidth: (0, 0),
            generated_backend_config: GeneratedBackendConfig::default(),
            generated_backend_selected_backend: None,
            generated_backend_runtime_diagnostics: HashMap::new(),
            telemetry_counters: BvpTelemetryRecorder::default(),
            generation_telemetry: None,
            atom_discretization_telemetry: None,
            legacy_lambdify_telemetry: None,
            atom_lambdify_telemetry: None,
            direct_banded_jacobian_telemetry: None,
            parameter_binding: None,
            custom_timer: CustomTimer::new(),
            no_reports: false,
        }
    }

    /// Creates a new solver instance with an explicit generated-backend configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_generated_backend_config(
        eq_system: Vec<Expr>,
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,
        max_iterations: usize,
        generated_backend_config: GeneratedBackendConfig,
    ) -> NRBVP {
        Self::new(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
        )
        .with_generated_backend_config(generated_backend_config)
    }

    /// Creates a new solver instance with a high-level sparse generated-backend mode.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_sparse_generated_backend_mode(
        eq_system: Vec<Expr>,
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,
        max_iterations: usize,
        mode: SparseGeneratedBackendMode,
    ) -> NRBVP {
        Self::new_with_generated_backend_config(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
            GeneratedBackendConfig::from_sparse_mode(mode),
        )
    }

    /// Creates a solver from a grouped options object instead of many positional arguments.
    ///
    /// This is the preferred public construction path for new code. The other
    /// constructor variants are retained as compatibility entrypoints.
    pub fn new_with_options(
        eq_system: Vec<Expr>,
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        options: FrozenSolverOptions,
    ) -> NRBVP {
        Self::new_with_generated_backend_config(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            options.strategy,
            options.strategy_params,
            options.linear_sys_method,
            options.method,
            options.tolerance,
            options.max_iterations,
            options.generated_backend_config,
        )
        .with_scheme_name(options.scheme)
    }

    /// Returns a solver configured with the provided generated-backend settings.
    pub fn with_generated_backend_config(mut self, config: GeneratedBackendConfig) -> Self {
        self.telemetry_counters.set_logging_config(
            crate::numerical::BVP_Damp::telemetry::BvpLoggingConfig {
                mode: config.bvp_logging_mode,
                max_events: config.bvp_logging_max_events,
            },
        );
        self.telemetry_counters
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.custom_timer
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.generated_backend_config = config;
        self
    }

    /// Returns a solver configured with the selected residual discretization scheme.
    pub fn with_scheme(mut self, scheme: BvpDerivativeScheme) -> Self {
        self.scheme = scheme.as_legacy_str().to_string();
        self
    }

    /// Returns a solver configured with the legacy forward derivative discretization.
    pub fn forward_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Forward)
    }

    /// Returns a solver configured with the trapezoid derivative discretization.
    pub fn trapezoid_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Trapezoid)
    }

    /// Compatibility escape hatch for legacy/custom scheme strings.
    pub fn with_scheme_name(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = scheme.into();
        self
    }

    /// Returns a solver configured with a high-level sparse generated-backend mode.
    pub fn with_sparse_generated_backend_mode(mut self, mode: SparseGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
        self
    }

    /// Returns a solver configured with the selected symbolic assembly backend.
    pub fn with_symbolic_assembly_backend(mut self, backend: BvpSymbolicAssemblyBackend) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_symbolic_assembly_backend(backend);
        self
    }

    /// Returns a solver configured with the standard sparse generated-backend defaults.
    pub fn with_sparse_generated_backend_defaults(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Returns a solver configured to require a prebuilt sparse AOT backend.
    pub fn with_sparse_aot_require_prebuilt(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt)
    }

    /// Returns a solver configured to build a sparse release AOT backend on demand.
    pub fn with_sparse_aot_build_if_missing_release(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease)
    }

    /// Returns a solver configured for AtomView symbolic assembly plus `gcc`-compiled sparse C AOT.
    pub fn with_sparse_atomview_c_gcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
        )
    }

    /// Returns a solver configured for AtomView symbolic assembly plus `tcc`-compiled sparse C AOT.
    pub fn with_sparse_atomview_c_tcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        )
    }

    /// Returns a solver configured for the recommended repeated-solve compiled path.
    pub fn with_sparse_atomview_for_repeated_solves(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_for_repeated_solves(),
        )
    }

    /// Returns a solver with an explicit generated-backend policy override.
    pub fn with_backend_policy_override(
        mut self,
        backend_policy: Option<BackendSelectionPolicy>,
    ) -> Self {
        self.generated_backend_config.backend_policy_override = backend_policy;
        self
    }

    /// Returns a solver with an explicit generated-backend resolver snapshot.
    pub fn with_aot_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.generated_backend_config.resolver = resolver;
        self
    }

    /// Returns a solver with an explicit solver-level AOT execution policy.
    pub fn with_aot_execution_policy(mut self, policy: AotExecutionPolicy) -> Self {
        self.generated_backend_config.aot_execution_policy = policy;
        self
    }

    /// Returns a solver with an explicit solver-level AOT build policy.
    pub fn with_aot_build_policy(mut self, policy: AotBuildPolicy) -> Self {
        self.generated_backend_config.aot_build_policy = policy;
        self
    }

    /// Returns a solver with explicit solver-level AOT chunking overrides.
    pub fn with_aot_chunking_policy(mut self, policy: AotChunkingPolicy) -> Self {
        self.generated_backend_config.aot_chunking_policy = policy;
        self
    }

    /// Returns a solver with an explicit AtomView AOT optimization profile.
    pub fn with_atom_optimization_profile(mut self, profile: AtomOptimizationProfile) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_atom_optimization_profile(profile);
        self
    }

    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn task_check(&self) {
        if self.t_end < self.t0 {
            panic!("Frozen BVP task check failed: t_end must be greater than t0");
        }

        if self.n_steps < 1 {
            panic!("Frozen BVP task check failed: n_steps must be greater than 1");
        }
        if self.max_iterations < 1 {
            panic!("Frozen BVP task check failed: max_iterations must be greater than 1");
        }
        let (m, n) = self.initial_guess.shape();
        if m != self.values.len() {
            panic!(
                "Frozen BVP task check failed: initial guess row count must match number of unknowns, rows = {}, values = {}",
                m,
                self.values.len()
            );
        }
        if n != self.n_steps {
            panic!(
                "Frozen BVP task check failed: initial guess column count must equal number of steps"
            );
        }
        if self.tolerance < 0.0 {
            panic!("Frozen BVP task check failed: tolerance must be greater than 0.0");
        }
        if self.max_error < 0.0 {
            panic!("Frozen BVP task check failed: max_error must be greater than 0.0");
        }
        if self.BorderConditions.is_empty() {
            panic!("Frozen BVP task check failed: boundary conditions must be specified");
        }
        if self.BorderConditions.len() != self.values.len() {
            panic!(
                "Frozen BVP task check failed: boundary conditions must be specified for each unknown"
            );
        }
    }

    /// Fallible counterpart of [`NRBVP::task_check`] for typed callers.
    ///
    /// The historical `task_check()` and `eq_generate()` methods remain
    /// compatibility panic wrappers; generated preparation uses this method so
    /// malformed task documents and strategy parameters stay inside `Result`.
    pub fn try_task_check(&self) -> Result<(), BvpBackendIntegrationError> {
        let invalid_problem =
            |field: &str, message: String| BvpBackendIntegrationError::InvalidProblem {
                field: field.to_string(),
                message,
            };
        let invalid_option = |field: &str, value: String, message: String| {
            BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: field.to_string(),
                value,
                message,
            }
        };

        if self.values.is_empty() {
            return Err(invalid_problem(
                "values",
                "at least one unknown is required".into(),
            ));
        }
        if self.initial_guess.shape() != (self.values.len(), self.n_steps) {
            return Err(invalid_problem(
                "initial_guess",
                format!(
                    "shape {:?} must be ({}, {})",
                    self.initial_guess.shape(),
                    self.values.len(),
                    self.n_steps
                ),
            ));
        }
        if !self.t0.is_finite() || !self.t_end.is_finite() || self.t_end <= self.t0 {
            return Err(invalid_problem(
                "interval",
                format!(
                    "expected finite t_end > t0, got [{}, {}]",
                    self.t0, self.t_end
                ),
            ));
        }
        if self.n_steps < 1 {
            return Err(invalid_problem(
                "n_steps",
                format!("expected n_steps >= 1, got {}", self.n_steps),
            ));
        }
        if self.max_iterations < 1 {
            return Err(invalid_option(
                "max_iterations",
                self.max_iterations.to_string(),
                "expected max_iterations >= 1".into(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(invalid_option(
                "tolerance",
                self.tolerance.to_string(),
                "expected a finite positive tolerance".into(),
            ));
        }
        if !matches!(
            self.scheme.to_ascii_lowercase().as_str(),
            "forward" | "trapezoid" | "trapezoidal"
        ) {
            return Err(invalid_option(
                "scheme",
                self.scheme.clone(),
                "supported values are forward and trapezoid".into(),
            ));
        }
        let effective_method = self.generated_backend_config.effective_method(&self.method);
        if BvpMatrixBackend::from_legacy_method(&effective_method).is_none() {
            return Err(invalid_option(
                "method",
                effective_method,
                "unknown matrix backend".into(),
            ));
        }
        match self.strategy.as_str() {
            "Naive" => {
                if self.strategy_params.is_some() {
                    return Err(invalid_option(
                        "strategy_params",
                        format!("{:?}", self.strategy_params),
                        "Naive strategy does not accept strategy parameters".into(),
                    ));
                }
            }
            "Frozen" => {
                let Some(params) = self.strategy_params.as_ref() else {
                    return Err(invalid_option(
                        "strategy_params",
                        "missing".into(),
                        "Frozen strategy requires exactly one strategy parameter".into(),
                    ));
                };
                if params.len() != 1 {
                    return Err(invalid_option(
                        "strategy_params",
                        format!("{params:?}"),
                        "Frozen strategy requires exactly one strategy parameter".into(),
                    ));
                }
                let (name, value) = params.iter().next().expect("len checked above");
                let valid = match (name.as_str(), value.as_ref()) {
                    ("Frozen_naive", None) => true,
                    ("every_m", Some(values)) => {
                        values.len() == 1 && values[0].is_finite() && values[0] > 0.0
                    }
                    ("at_high_morm", Some(values)) => {
                        values.len() == 1 && values[0].is_finite() && values[0] > 0.0
                    }
                    ("at_low_speed", Some(values)) => {
                        values.len() == 1 && values[0].is_finite() && values[0] <= 1.0
                    }
                    ("complex", Some(values)) => {
                        values.len() == 3 && values.iter().all(|value| value.is_finite())
                    }
                    _ => false,
                };
                if !valid {
                    return Err(invalid_option(
                        "strategy_params",
                        format!("{params:?}"),
                        "unsupported Frozen strategy parameter shape or value".into(),
                    ));
                }
            }
            strategy => {
                return Err(invalid_option(
                    "strategy",
                    strategy.to_string(),
                    "supported values are Frozen and Naive".into(),
                ));
            }
        }
        if self.BorderConditions.is_empty()
            || self.BorderConditions.len() != self.values.len()
            || self
                .BorderConditions
                .keys()
                .any(|name| !self.values.iter().any(|value| value == name))
        {
            return Err(invalid_problem(
                "boundary_conditions",
                format!(
                    "expected one boundary-condition entry per unknown, got {} for {} unknowns",
                    self.BorderConditions.len(),
                    self.values.len()
                ),
            ));
        }
        Ok(())
    }

    pub fn try_eq_generate(&mut self) -> Result<(), BvpBackendIntegrationError> {
        self.try_task_check()?;
        let effective_method = self.generated_backend_config.effective_method(&self.method);
        let effective_policy = self
            .generated_backend_config
            .effective_backend_policy(&effective_method);
        if effective_policy == BackendSelectionPolicy::NumericOnly {
            return Err(BvpBackendIntegrationError::PipelinePanicked(
                "NumericOnly is intentionally not available for the frozen BVP solver; use the damped solver with numeric_rhs for pure numeric finite-difference discretization, or use symbolic Lambdify/AOT with Frozen"
                    .to_string(),
            ));
        }
        try_generate_and_apply_frozen_solver_state(
            self,
            "building frozen BVP generated solver state",
        )
    }

    /// Compatibility-only wrapper over [`NRBVP::try_eq_generate`].
    ///
    /// Prefer the fallible `try_*` entrypoint in new code so backend/build/runtime
    /// errors stay typed all the way to the caller.
    pub fn eq_generate(&mut self) {
        self.try_eq_generate().unwrap_or_else(|err| {
            panic!("Frozen BVP generated solver state build failed: {err:?}")
        });
    } // end of method eq_generate

    /// Installs an optional compiled AOT resolver used by backend selection.
    pub fn set_aot_resolver(&mut self, resolver: Option<AotResolver>) {
        let mut config = self.generated_backend_config.clone();
        config.resolver = resolver;
        self.set_generated_backend_config(config);
    }

    /// Installs the solver-level AOT execution policy.
    pub fn set_aot_execution_policy(&mut self, policy: AotExecutionPolicy) {
        let mut config = self.generated_backend_config.clone();
        config.aot_execution_policy = policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured solver-level AOT execution policy.
    pub fn aot_execution_policy(&self) -> &AotExecutionPolicy {
        &self.generated_backend_config.aot_execution_policy
    }

    /// Installs the solver-level AOT build policy.
    pub fn set_aot_build_policy(&mut self, policy: AotBuildPolicy) {
        let mut config = self.generated_backend_config.clone();
        config.aot_build_policy = policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured solver-level AOT build policy.
    pub fn aot_build_policy(&self) -> AotBuildPolicy {
        self.generated_backend_config.aot_build_policy
    }

    /// Installs explicit solver-level AOT chunking overrides.
    pub fn set_aot_chunking_policy(&mut self, policy: AotChunkingPolicy) {
        let mut config = self.generated_backend_config.clone();
        config.aot_chunking_policy = policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured solver-level AOT chunking overrides.
    pub fn aot_chunking_policy(&self) -> AotChunkingPolicy {
        self.generated_backend_config.aot_chunking_policy
    }

    /// Installs an explicit AtomView AOT optimization profile.
    pub fn set_atom_optimization_profile(&mut self, profile: AtomOptimizationProfile) {
        let mut config = self.generated_backend_config.clone();
        config.atom_optimization_profile = profile;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured AtomView AOT optimization profile.
    pub fn atom_optimization_profile(&self) -> AtomOptimizationProfile {
        self.generated_backend_config.atom_optimization_profile
    }

    /// Returns the configured compiled AOT resolver, if present.
    pub fn aot_resolver(&self) -> Option<&AotResolver> {
        self.generated_backend_config.resolver.as_ref()
    }

    /// Installs an explicit generated-backend selection policy override.
    pub fn set_backend_policy_override(&mut self, backend_policy: Option<BackendSelectionPolicy>) {
        let mut config = self.generated_backend_config.clone();
        config.backend_policy_override = backend_policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured generated-backend selection policy override, if present.
    pub fn backend_policy_override(&self) -> Option<BackendSelectionPolicy> {
        self.generated_backend_config.backend_policy_override
    }

    /// Installs the complete generated-backend configuration in one call.
    pub fn set_generated_backend_config(&mut self, config: GeneratedBackendConfig) {
        self.invalidate_linear_runtime();
        self.telemetry_counters.set_logging_config(
            crate::numerical::BVP_Damp::telemetry::BvpLoggingConfig {
                mode: config.bvp_logging_mode,
                max_events: config.bvp_logging_max_events,
            },
        );
        self.telemetry_counters
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.custom_timer
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.generated_backend_config = config;
        self.prepared_runtime_revision.configuration_changed();
    }

    /// Changes Lambdify runtime telemetry for the next generated callback build.
    pub fn set_lambdify_telemetry_mode(&mut self, mode: BvpLambdifyTelemetryMode) {
        let config = self
            .generated_backend_config
            .clone()
            .with_lambdify_telemetry_mode(mode);
        self.set_generated_backend_config(config);
    }

    /// Changes typed solver decision logging for subsequent runtime events.
    pub fn set_bvp_logging_mode(&mut self, mode: BvpLoggingMode) {
        let config = self
            .generated_backend_config
            .clone()
            .with_bvp_logging_mode(mode);
        self.set_generated_backend_config(config);
    }

    /// Replaces the complete bounded typed decision logging policy.
    pub fn set_bvp_logging_config(&mut self, config: BvpLoggingConfig) {
        self.set_generated_backend_config(
            self.generated_backend_config
                .clone()
                .with_bvp_logging_config(config),
        );
    }

    /// Changes solver counter/timer collection for subsequent solves.
    pub fn set_bvp_telemetry_mode(&mut self, mode: BvpTelemetryMode) {
        self.generated_backend_config.bvp_telemetry_mode = mode;
        self.telemetry_counters.set_telemetry_mode(mode);
        self.custom_timer.set_telemetry_mode(mode);
    }

    /// Returns the Lambdify callback telemetry policy used for new callbacks.
    pub fn lambdify_telemetry_mode(&self) -> BvpLambdifyTelemetryMode {
        self.generated_backend_config.lambdify_telemetry_mode
    }

    /// Replaces boundary conditions through the revision-tracked API.
    ///
    /// The public field remains for source compatibility; prepared callers
    /// should use this setter so factors and generated callbacks are invalidated
    /// when the physical boundary problem changes.
    pub fn set_boundary_conditions(&mut self, conditions: HashMap<String, Vec<(usize, f64)>>) {
        if self.BorderConditions != conditions {
            self.BorderConditions = conditions;
            self.invalidate_linear_runtime();
            self.prepared_runtime_revision.problem_changed();
            self.result = None;
        }
    }

    /// Fallible typed setter for symbolic parameter names.
    pub fn try_set_params(
        &mut self,
        params: Option<&[&str]>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let param_names: Vec<String> = params
            .map(|items| items.iter().map(|name| (*name).to_string()).collect())
            .unwrap_or_default();
        if let Some(values) = self.param_values.as_ref() {
            if values.len() != param_names.len() {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "param_names".to_string(),
                    value: format!("{} names", param_names.len()),
                    message: format!(
                        "param_values length {} must match param_names length {}",
                        values.len(),
                        param_names.len()
                    ),
                });
            }
        }
        if self.param_names != param_names {
            self.param_names = param_names;
            self.parameter_binding = None;
            self.invalidate_linear_runtime();
            self.prepared_runtime_revision.parameters_changed();
        }
        Ok(())
    }

    /// Compatibility wrapper for [`Self::try_set_params`].
    pub fn set_params(&mut self, params: Option<&[&str]>) {
        self.try_set_params(params)
            .unwrap_or_else(|error| panic!("invalid BVP parameter names: {error:?}"));
    }

    /// Fallible typed setter for the numeric values of symbolic parameters.
    /// Parameters are evaluator inputs, not Newton unknowns and are never
    /// included in the symbolic derivative layout.
    pub fn try_set_param_values(
        &mut self,
        values: Option<Vec<f64>>,
    ) -> Result<(), BvpBackendIntegrationError> {
        if let Some(values_ref) = values.as_ref() {
            if values_ref.len() != self.param_names.len() {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "param_values".to_string(),
                    value: format!("{} values", values_ref.len()),
                    message: format!(
                        "expected exactly {} values for declared symbolic parameters",
                        self.param_names.len()
                    ),
                });
            }
        }
        if self.param_values != values {
            self.param_values = values;
            if let Some(binding) = &self.parameter_binding {
                binding.replace(self.param_values.clone());
                self.prepared_runtime_revision
                    .refresh_prepared_fingerprint(self.prepared_plan_fingerprint());
            } else {
                self.prepared_runtime_revision.parameters_changed();
            }
            self.invalidate_linear_runtime();
        }
        Ok(())
    }

    /// Compatibility wrapper for [`Self::try_set_param_values`].
    pub fn set_param_values(&mut self, values: Option<Vec<f64>>) {
        self.try_set_param_values(values)
            .unwrap_or_else(|error| panic!("invalid BVP parameter values: {error:?}"));
    }

    /// Sets the symbolic assembly backend used before lambdify/AOT lowering.
    pub fn set_symbolic_assembly_backend(&mut self, backend: BvpSymbolicAssemblyBackend) {
        let config = self
            .generated_backend_config
            .clone()
            .with_symbolic_assembly_backend(backend);
        self.set_generated_backend_config(config);
    }

    /// Installs a high-level sparse generated-backend mode on an existing solver.
    pub fn set_sparse_generated_backend_mode(&mut self, mode: SparseGeneratedBackendMode) {
        self.set_generated_backend_config(GeneratedBackendConfig::from_sparse_mode(mode));
    }

    /// Installs the standard sparse generated-backend defaults.
    pub fn set_sparse_generated_backend_defaults(&mut self) {
        self.set_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults);
    }

    /// Installs AtomView symbolic assembly plus `gcc`-compiled sparse C AOT.
    pub fn set_sparse_atomview_c_gcc(&mut self) {
        self.set_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
        );
    }

    /// Installs AtomView symbolic assembly plus `tcc`-compiled sparse C AOT.
    pub fn set_sparse_atomview_c_tcc(&mut self) {
        self.set_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        );
    }

    /// Installs the recommended repeated-solve compiled path.
    pub fn set_sparse_atomview_for_repeated_solves(&mut self) {
        self.set_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_for_repeated_solves(),
        );
    }

    /// Returns the full generated-backend configuration.
    pub fn generated_backend_config(&self) -> &GeneratedBackendConfig {
        &self.generated_backend_config
    }

    /// Returns the normalized configuration and the backend selected so far.
    /// This is read-only and never triggers equation generation.
    pub fn resolved_plan(&self) -> crate::numerical::BVP_Damp::resolved_plan::BvpResolvedPlan {
        let mut plan =
            crate::numerical::BVP_Damp::resolved_plan::BvpResolvedPlan::from_common_for_solver(
                &self.generated_backend_config,
                &self.method,
                &self.scheme,
                crate::numerical::BVP_Damp::resolved_plan::strategy_from_name(&self.strategy),
            );
        if let Some(selected) = self.generated_backend_selected_backend {
            plan = plan.with_selected_backend(selected);
        }
        plan
    }

    /// Removes registered generated AOT artifact directories owned by this solver resolver.
    ///
    /// This is an explicit lifecycle operation for cold-build/story/debug workflows. Call it only
    /// after compiled callbacks from this solver are no longer needed; the method does not try to
    /// unregister process-local linked callbacks or unload dynamic libraries.
    pub fn cleanup_registered_aot_artifacts(&mut self) -> std::io::Result<usize> {
        cleanup_registered_aot_artifacts(&mut self.generated_backend_config)
    }

    /// Returns accumulated operation statistics and actual backend diagnostics.
    pub fn get_statistics(&self) -> FrozenBvpStatistics {
        let telemetry_counters = self.telemetry_counters.snapshot();
        let mut counters = telemetry_counters.to_legacy_map();
        if let Some(jac) = &self.factor_owner.old_jac {
            let shape = jac.shape();
            counters.insert("number of jacobian elements".to_string(), shape.0 * shape.1);
        }
        counters.insert("length of y vector".to_string(), self.y.len());
        counters.insert("number of grid points".to_string(), self.x_mesh.len());
        let mut diagnostics = self.generated_backend_runtime_diagnostics.clone();
        self.append_generated_backend_diagnostics(&mut diagnostics);
        let timings = self.custom_timer.snapshot();
        let scopes = self
            .telemetry_counters
            .scopes_snapshot(&timings, &telemetry_counters);
        let storage = self
            .factor_owner
            .old_jac
            .as_ref()
            .map(|jac| {
                crate::numerical::BVP_Damp::telemetry::BvpStorageBytes::for_solver_method(
                    &self.effective_runtime_method(),
                    jac.shape().0,
                    jac.shape().1,
                    self.bandwidth,
                )
            })
            .unwrap_or_default();
        FrozenBvpStatistics {
            counters,
            timers: self.custom_timer.get_all(),
            diagnostics,
            telemetry: BvpTelemetrySnapshot {
                telemetry_mode: self.telemetry_counters.telemetry_mode(),
                counters: telemetry_counters,
                timings,
                scopes,
                storage,
                plan: Some(self.resolved_plan()),
                log_events: self.telemetry_counters.log_events_snapshot(),
                log_events_dropped: self.telemetry_counters.dropped_log_events(),
                solve_id: self.telemetry_counters.solve_id(),
                logging_config: self.telemetry_counters.logging_config(),
                atom_discretization: self.atom_discretization_telemetry,
                generation: self.generation_telemetry,
                legacy_lambdify: self
                    .legacy_lambdify_telemetry
                    .as_ref()
                    .map(BvpLambdifyTelemetry::snapshot),
                atom_lambdify: self
                    .atom_lambdify_telemetry
                    .as_ref()
                    .map(BvpLambdifyTelemetry::snapshot),
                direct_banded_jacobian: self
                    .direct_banded_jacobian_telemetry
                    .as_ref()
                    .map(crate::symbolic::bvp::telemetry::BvpDirectJacobianTelemetry::snapshot),
            },
        }
    }

    fn append_generated_backend_diagnostics(&self, diagnostics: &mut HashMap<String, String>) {
        let config = &self.generated_backend_config;
        let effective_method = self.effective_runtime_method();
        diagnostics.insert(
            "generated.backend_policy".to_string(),
            format!("{:?}", config.effective_backend_policy(&effective_method)),
        );
        diagnostics.insert("generated.effective_method".to_string(), effective_method);
        diagnostics.insert(
            "generated.selected_backend".to_string(),
            self.generated_backend_selected_backend
                .map(|backend| format!("{backend:?}"))
                .unwrap_or_else(|| "not_generated".to_string()),
        );
        diagnostics.insert(
            "generated.symbolic_assembly_backend".to_string(),
            format!("{:?}", config.symbolic_assembly_backend),
        );
        diagnostics.insert(
            "generated.matrix_backend_override".to_string(),
            config
                .matrix_backend_override
                .map(|backend| format!("{backend:?}"))
                .unwrap_or_else(|| "none".to_string()),
        );
        diagnostics.insert(
            "aot.build_policy".to_string(),
            config.aot_build_policy.as_str().to_string(),
        );
        diagnostics.insert(
            "aot.execution_policy".to_string(),
            config.aot_execution_policy.as_str().to_string(),
        );
        diagnostics.insert(
            "aot.codegen_backend".to_string(),
            format!("{:?}", config.aot_codegen_backend),
        );
        diagnostics.insert(
            "aot.c_compiler".to_string(),
            config
                .aot_c_compiler
                .clone()
                .unwrap_or_else(|| "none".to_string()),
        );
        diagnostics.insert(
            "aot.chunking.residual".to_string(),
            config
                .aot_chunking_policy
                .residual
                .map(|strategy| format!("{strategy:?}"))
                .unwrap_or_else(|| "default".to_string()),
        );
        diagnostics.insert(
            "aot.chunking.sparse_jacobian".to_string(),
            config
                .aot_chunking_policy
                .sparse_jacobian
                .map(|strategy| format!("{strategy:?}"))
                .unwrap_or_else(|| "default".to_string()),
        );
    }

    /// Returns the selected symbolic assembly backend.
    pub fn symbolic_assembly_backend(&self) -> BvpSymbolicAssemblyBackend {
        self.generated_backend_config.symbolic_assembly_backend
    }
    pub fn set_new_step(&mut self, p: f64, y: Box<dyn VectorType>, initial_guess: DMatrix<f64>) {
        self.invalidate_linear_runtime();
        self.p = p;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_p(&mut self, p: f64) {
        if self.p != p {
            self.invalidate_linear_runtime();
        }
        self.p = p;
    }

    /// Fallible Newton iteration used by the typed solve path.
    pub fn try_iteration(&mut self) -> Result<Box<dyn VectorType>, BvpBackendIntegrationError> {
        let p = self.p;
        let y = &*self.y;
        let fun = &self.fun;
        let fun_begin = Instant::now();
        self.telemetry_counters.record_residual_call();
        let new_fun = fun.try_call(p, y).map_err(|error| {
            BvpBackendIntegrationError::CallbackExecutionFailed {
                stage: "residual".to_string(),
                message: error.to_string(),
            }
        })?;
        self.custom_timer.append_to_fun_time(fun_begin.elapsed());
        let now = Instant::now();

        let reused_factorization;
        if self.jac_recalc {
            info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let begin = Instant::now();
            self.custom_timer.jac_tic();
            let callback_result = match self.jac.as_mut() {
                Some(jacobian) => jacobian.try_call(p, y),
                None => {
                    self.custom_timer.jac_tac();
                    return Err(BvpBackendIntegrationError::PipelinePanicked(
                        "Frozen BVP iteration requires an installed Jacobian callback".into(),
                    ));
                }
            };
            let new_j = match callback_result {
                Ok(jacobian) => jacobian,
                Err(error) => {
                    self.custom_timer.jac_tac();
                    return Err(BvpBackendIntegrationError::CallbackExecutionFailed {
                        stage: "Jacobian".to_string(),
                        message: error.to_string(),
                    });
                }
            };
            info!("jacobian recalculation time: ");
            let elapsed = begin.elapsed();
            elapsed_time(elapsed);
            self.custom_timer.jac_tac();
            if self
                .factor_owner
                .old_jac
                .as_ref()
                .map(|jacobian| jacobian.factorization_ready())
                .unwrap_or(false)
            {
                self.telemetry_counters.record_factorization_invalidation();
            }
            // The cached Jacobian is read-only during a frozen reuse window.
            // Store the callback result directly instead of cloning the full
            // matrix on every iteration.
            self.factor_owner.old_jac = Some(new_j);
            let fresh_jacobian = self.factor_owner.old_jac.as_ref().ok_or_else(|| {
                BvpBackendIntegrationError::PipelinePanicked(
                    "Frozen BVP Jacobian callback returned no cached matrix".into(),
                )
            })?;
            *self.factor_owner.borrow_mut() = prepare_factor_owner_runtime(
                fresh_jacobian.as_ref(),
                self.bandwidth,
                self.linear_sys_method.as_deref(),
            );
            self.prepared_runtime_revision
                .mark_numeric_jacobian_current();
            if self.factor_owner.borrow().is_some() {
                self.prepared_runtime_revision.mark_factor_current();
            }
            reused_factorization = false;
            self.m = 0;
            self.telemetry_counters.record_jacobian_recalculation();
        } else {
            self.m = self.m + 1;
            reused_factorization = self
                .factor_owner
                .borrow()
                .as_ref()
                .map(|owner| owner.has_solved_rhs())
                .unwrap_or(false)
                || self
                    .factor_owner
                    .old_jac
                    .as_ref()
                    .map(|jacobian| jacobian.factorization_ready())
                    .unwrap_or(false);
        }

        let new_j = self
            .factor_owner
            .old_jac
            .as_ref()
            .ok_or_else(|| {
                BvpBackendIntegrationError::PipelinePanicked(
                    "Frozen BVP iteration requires a cached Jacobian matrix when reuse is enabled"
                        .into(),
                )
            })?
            .as_ref();

        //   println!("new fun = {:?}", &new_fun);
        let linear_begin = Instant::now();
        let (delta, linear_timing) = if let Some(owner) = self.factor_owner.borrow_mut().as_mut() {
            let (delta, factorization, rhs_solve) =
                owner.try_solve(&*new_fun).map_err(|error| {
                    BvpBackendIntegrationError::LinearSolveFailed {
                        backend: "owned-factor".to_string(),
                        matrix_rows: new_j.shape().0,
                        matrix_columns: new_j.shape().1,
                        rhs_len: new_fun.len(),
                        message: format!("{error:?}"),
                    }
                })?;
            (
                delta,
                LinearSolveTiming {
                    factorization,
                    rhs_solve,
                },
            )
        } else {
            new_j.solve_sys_with_timing(
                &*new_fun,
                self.linear_sys_method.clone(),
                self.tolerance,
                self.max_iterations,
                self.bandwidth,
                y,
            )
        };
        self.custom_timer
            .append_to_linear_sys_time(linear_begin.elapsed() + linear_timing.factorization);
        self.custom_timer
            .append_to_factorization_time(linear_timing.factorization);
        self.custom_timer
            .append_to_rhs_solve_time(linear_timing.rhs_solve);
        self.telemetry_counters.record_linear_solve();
        self.telemetry_counters.record_rhs_solve();
        if reused_factorization {
            self.telemetry_counters.record_factorization_cache_hit();
        } else {
            self.telemetry_counters.record_factorization();
        }
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        //  println!(" \n \n dy= {:?}", &delta);
        // element wise subtraction
        let new_y = y - &*delta;

        Ok(new_y)
    }

    /// Legacy panic-wrapper retained for callers using the historical API.
    pub fn iteration(&mut self) -> Box<dyn VectorType> {
        self.try_iteration().unwrap_or_else(|error| {
            panic!("Frozen BVP iteration failed during fallible runtime path: {error:?}")
        })
    }
    pub fn main_loop(&mut self) -> Option<DVector<f64>> {
        self.try_main_loop().unwrap_or_else(|err| {
            panic!("Frozen BVP main loop failed during fallible runtime path: {err:?}")
        })
    }

    /// Fallible internal Newton loop used by [`NRBVP::try_solver`].
    pub fn try_main_loop(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        info!("solving system of equations with Newton-Raphson method! \n \n");
        let y: DMatrix<f64> = self.initial_guess.clone();
        let y: Vec<f64> = y.iter().cloned().collect();
        let y: DVector<f64> = DVector::from_vec(y);
        self.y = Vectors_type_casting(&y.clone(), self.method.clone());
        let mut i = 0;

        while i < self.max_iterations {
            self.telemetry_counters.record_iteration();
            let iteration_started = self.telemetry_counters.start_iteration_scope();
            let iteration_result = self.try_iteration();
            if iteration_result.is_err() {
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
            }
            let new_y = iteration_result?;
            let y1 = new_y.subtract(&*self.y);
            let dy: Box<dyn VectorType> = y1.clone_box();

            let error = dy.norm();
            self.jac_recalc = frozen_jac_recalc(
                &self.strategy,
                &self.strategy_params,
                &self.factor_owner.old_jac,
                self.m,
                error,
                self.error_old,
            );
            self.error_old = error;
            info!(" \n \n error = {:?} \n \n", &error);
            if error < self.tolerance {
                log::info!("converged in {} iterations, error = {}", i, error);
                self.result = Some(new_y.to_DVectorType());
                self.max_error = error;
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
                return Ok(Some(new_y.to_DVectorType()));
            } else {
                let new_y: Box<dyn VectorType> = new_y.clone_box();
                self.y = new_y;
                i += 1;
                self.telemetry_counters
                    .finish_iteration_scope(iteration_started);
            }
        }
        Ok(None)
    }
    /// Fallible solve path without logging setup.
    ///
    /// This is the preferred entrypoint for internal/runtime callers that want
    /// typed backend and execution errors but do not need the higher-level
    /// logging wrapper provided by [`NRBVP::try_solve`].
    pub fn try_solver(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        // TODO! сравнить явный мэш с неявным
        // let test_mesh = Some((0..100).map(|x| 0.01 * x as f64).collect::<Vec<f64>>());
        self.telemetry_counters.begin_solve();
        self.custom_timer.start();
        let begin = Instant::now();
        let res = (|| {
            self.custom_timer.symbolic_operations_tic();
            self.try_eq_generate()?;
            self.custom_timer.symbolic_operations_tac();
            self.try_main_loop()
        })();
        self.custom_timer.finish();
        self.telemetry_counters.record_termination(
            matches!(&res, Ok(Some(_))),
            if matches!(&res, Ok(Some(_))) {
                1.0
            } else {
                0.0
            },
        );
        let res = res?;
        let end = begin.elapsed();
        elapsed_time(end);

        Ok(res)
    }

    /// Solves using callbacks prepared by an earlier `try_eq_generate` call.
    ///
    /// Numeric parameter values may be rebound between calls; structural or
    /// configuration changes are rejected until preparation is repeated.
    pub fn try_solver_prepared(
        &mut self,
    ) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        if !self
            .prepared_runtime_revision
            .is_current_with_fingerprint(self.prepared_plan_fingerprint())
        {
            return Err(BvpBackendIntegrationError::PreparedRuntimeInvalidated {
                reason:
                    "prepared Frozen plan is stale; call try_eq_generate before try_solver_prepared"
                        .to_string(),
            });
        }
        self.try_main_loop()
    }
    /// Compatibility-only wrapper over [`NRBVP::try_solver`].
    ///
    /// New production-facing code should call [`NRBVP::try_solver`] or
    /// [`NRBVP::try_solve`] instead.
    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.try_solver()
            .unwrap_or_else(|err| panic!("Frozen BVP solver failed before Newton loop: {err:?}"))
    }

    /// Main public fallible solve entrypoint with logging support.
    ///
    /// This is the preferred production-facing solve path.
    pub fn try_solve(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let logger_instance = if self.no_reports {
            let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                LevelFilter::Info,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            )]);
            logger_instance
        } else {
            let date_and_time = Local::now().format("%Y-%m-%d_%H-%M");
            let name = format!("log_{}.txt", date_and_time);
            let file = File::create(&name).map_err(|err| {
                BvpBackendIntegrationError::LogFileCreationFailed {
                    path: name.clone(),
                    message: err.to_string(),
                }
            })?;
            let logger_instance = CombinedLogger::init(vec![
                TermLogger::new(
                    LevelFilter::Info,
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto,
                ),
                WriteLogger::new(LevelFilter::Info, Config::default(), file),
            ]);
            logger_instance
        };
        match logger_instance {
            Ok(()) => {
                let res = self.try_solver()?;
                log::info!("Program ended");
                Ok(res)
            }
            Err(_) => self.try_solver(),
        }
    }
    /// Compatibility-only wrapper over [`NRBVP::try_solve`].
    ///
    /// New code should prefer [`NRBVP::try_solve`] so AOT/logging/runtime failures
    /// remain typed instead of turning into a panic.
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        self.try_solve().unwrap_or_else(|err| {
            panic!("Frozen BVP solve failed before convergence loop: {err:?}")
        })
    }
    pub fn dont_save_log(&mut self, dont_save_log: bool) {
        self.no_reports = dont_save_log;
    }
    pub fn save_to_file(&self) {
        //let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let result_DMatrix = self
            .get_result()
            .expect("Frozen BVP save_to_file requires a computed solution matrix");
        let _ = save_matrix_to_file(
            &result_DMatrix,
            &self.values,
            "result.txt",
            &self.x_mesh,
            &self.arg,
        );
    }
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self
            .result
            .clone()
            .expect("Frozen BVP get_result requires a converged solution vector")
            .clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice())
                .transpose();
        let permutted_results = matrix_of_results;
        Some(permutted_results)
    }

    /// Converts the computed solution into the unified postprocessing dataset.
    pub fn postprocess_dataset(&self) -> Result<PostprocessDataset, PostprocessError> {
        if self.result.is_none() {
            return Err(PostprocessError::InvalidDataset(
                "Frozen BVP postprocess_dataset requires a converged solution vector".to_string(),
            ));
        }
        let values = self.get_result().ok_or_else(|| {
            PostprocessError::InvalidDataset(
                "Frozen BVP postprocess_dataset requires a converged solution vector".to_string(),
            )
        })?;
        PostprocessDataset::new(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            values,
        )
    }

    /// Executes a declarative postprocessing plan using the modern facade.
    pub fn execute_postprocessing(
        &self,
        plan: &PostprocessPlan,
    ) -> Result<PostprocessReport, PostprocessError> {
        let dataset = self.postprocess_dataset()?;
        plan.execute(&dataset)
    }

    pub fn plot_result(&self) {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self
            .result
            .clone()
            .expect("Frozen BVP plot_result requires a converged solution vector")
            .clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice())
                .transpose();
        for _col in matrix_of_results.column_iter() {
            //   println!( "{:?}", DVector::from_column_slice(_col.as_slice()) );
        }
        info!(
            "matrix of results has shape {:?}",
            matrix_of_results.shape()
        );
        info!("length of x mesh : {:?}", n_steps);
        info!("number of Ys: {:?}", number_of_Ys);
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            matrix_of_results,
        );
        info!("result plotted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot;
    use std::time::Duration;

    #[test]
    fn frozen_statistics_keep_legacy_projection_and_typed_snapshot_in_sync() {
        let solver = NRBVP::new(
            Vec::new(),
            DMatrix::zeros(0, 0),
            Vec::new(),
            String::new(),
            HashMap::new(),
            0.0,
            0.0,
            1,
            "Naive".to_string(),
            None,
            None,
            "Dense".to_string(),
            1e-8,
            1,
        );
        let stats = solver.get_statistics();

        assert_eq!(
            stats.counters["number of iterations"],
            stats.telemetry.counters.iterations as usize
        );
        assert_eq!(
            stats.counters["number of factorizations"],
            stats.telemetry.counters.factorizations as usize
        );
        assert_eq!(
            stats.counters["number of RHS solves"],
            stats.telemetry.counters.rhs_solves as usize
        );
        assert!(stats.telemetry.timings.total >= stats.telemetry.timings.jacobian);
    }

    #[test]
    fn frozen_statistics_expose_atom_discretization_telemetry() {
        let mut solver = NRBVP::new(
            Vec::new(),
            DMatrix::zeros(0, 0),
            Vec::new(),
            String::new(),
            HashMap::new(),
            0.0,
            0.0,
            1,
            "Naive".to_string(),
            None,
            None,
            "Dense".to_string(),
            1e-8,
            1,
        );
        let expected = BvpAtomDiscretizationTelemetrySnapshot {
            boundary_conditions: Duration::from_micros(2),
            discretization: Duration::from_micros(3),
            boundary_application: Duration::from_micros(5),
            flat_list: Duration::from_micros(7),
            consistency: Duration::from_micros(11),
            bounds_and_tolerances: Duration::from_micros(13),
            total: Duration::from_micros(41),
        };
        solver.atom_discretization_telemetry = Some(expected);

        let actual = solver
            .get_statistics()
            .telemetry
            .atom_discretization
            .expect("Frozen statistics should preserve Atom preparation telemetry");
        assert_eq!(actual, expected);
    }

    #[test]
    fn changing_continuation_parameter_invalidates_owned_factor() {
        let mut solver = NRBVP::new(
            Vec::new(),
            DMatrix::zeros(1, 1),
            vec!["y".to_string()],
            "x".to_string(),
            HashMap::from([("y".to_string(), vec![(0, 0.0)])]),
            0.0,
            1.0,
            1,
            "Naive".to_string(),
            None,
            None,
            "Dense".to_string(),
            1e-8,
            1,
        );
        *solver.factor_owner.borrow_mut() = Some(
            prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
                .expect("dense factor owner runtime"),
        );

        solver.set_p(1.0);

        assert!(solver.factor_owner.borrow().is_none());
        assert!(solver.factor_owner.old_jac.is_none());
        assert!(solver.jac_recalc);
        assert_eq!(
            solver
                .telemetry_counters
                .snapshot()
                .factorization_invalidations,
            1
        );
    }

    #[test]
    fn changing_parameters_invalidates_owned_factor() {
        let mut solver = NRBVP::new(
            Vec::new(),
            DMatrix::zeros(1, 1),
            vec!["y".to_string()],
            "x".to_string(),
            HashMap::from([("y".to_string(), vec![(0, 0.0)])]),
            0.0,
            1.0,
            1,
            "Naive".to_string(),
            None,
            None,
            "Dense".to_string(),
            1e-8,
            1,
        );
        *solver.factor_owner.borrow_mut() = Some(
            prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
                .expect("dense factor owner runtime"),
        );

        solver.set_params(Some(&["alpha"]));
        assert!(solver.factor_owner.borrow().is_none());
        assert!(solver.factor_owner.old_jac.is_none());
        assert!(solver.jac_recalc);

        *solver.factor_owner.borrow_mut() = Some(
            prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
                .expect("dense factor owner runtime"),
        );
        solver.set_param_values(Some(vec![1.0]));

        assert!(solver.factor_owner.borrow().is_none());
        assert!(solver.factor_owner.old_jac.is_none());
        assert!(solver.jac_recalc);
        assert_eq!(
            solver
                .telemetry_counters
                .snapshot()
                .factorization_invalidations,
            2
        );
    }

    #[test]
    fn changing_backend_policy_invalidates_owned_factor() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        *solver.factor_owner.borrow_mut() = Some(
            prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
                .expect("dense factor owner runtime"),
        );

        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        assert!(solver.factor_owner.borrow().is_none());
        assert!(solver.factor_owner.old_jac.is_none());
        assert!(solver.jac_recalc);
        assert_eq!(
            solver
                .telemetry_counters
                .snapshot()
                .factorization_invalidations,
            1
        );
    }

    #[test]
    fn frozen_try_iteration_surfaces_residual_callback_panic_as_typed_error() {
        let mut solver = sparse_surface_test_solver();
        solver.fun = convert_to_fun(Box::new(|_, _| {
            panic!("frozen residual callback failed deliberately")
        }));

        let error = solver
            .try_iteration()
            .expect_err("a callback panic must not escape Frozen try_iteration");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::CallbackExecutionFailed { stage, message }
                if stage == "residual"
                    && message.contains("frozen residual callback failed deliberately")
        ));
    }

    use crate::numerical::BVP_Damp::BVP_traits::convert_to_fun;
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy,
        BandedGeneratedBackendMode, GeneratedBackendConfig, SparseGeneratedBackendMode,
    };
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        LinkedSparseAotBackend, register_linked_sparse_backend, unregister_linked_sparse_backend,
    };
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::codegen::codegen_orchestrator::{
        ParallelExecutorConfig, ParallelFallbackPolicy,
    };
    use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
    use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
    use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
    use faer::Col;
    use nalgebra::{DMatrix, DVector};
    use std::sync::Arc;

    fn sparse_surface_test_solver() -> NRBVP {
        NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.1),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            0.0,
            1.0,
            4,
            FrozenSolverOptions::sparse_frozen(),
        )
    }

    fn sparse_surface_test_solver_with_naive_strategy() -> NRBVP {
        let options = FrozenSolverOptions::sparse_frozen()
            .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
            .with_tolerance(1e-6)
            .with_max_iterations(10);
        NRBVP::new_with_options(
            vec![Expr::parse_expression("y-z"), Expr::parse_expression("-z")],
            DMatrix::from_element(2, 8, 0.5),
            vec!["z".to_string(), "y".to_string()],
            "x".to_string(),
            HashMap::from([
                ("z".to_string(), vec![(0usize, 1.0f64)]),
                ("y".to_string(), vec![(1usize, 1.0f64)]),
            ]),
            0.0,
            1.0,
            8,
            options,
        )
    }

    fn frozen_linear_solver(n_steps: usize, options: FrozenSolverOptions) -> NRBVP {
        // y'' = 0 represented as y' = z, z' = 0 with
        // y(0)=0 and z(0)=1. This gives y=x, z=1 and is exact for
        // the forward first-order BVP stencil, so backend coverage is not
        // polluted by discretization error.
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = x;
            guess[i * values.len() + 1] = 1.0;
        }

        NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")],
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice()),
            values,
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, 0.0f64)]),
                ("z".to_string(), vec![(0usize, 1.0f64)]),
            ]),
            t0,
            t_end,
            n_steps,
            options,
        )
    }

    fn assert_frozen_linear_solution_quality(
        solver: &NRBVP,
        n_steps: usize,
        rms_tol: f64,
        max_abs_tol: f64,
    ) {
        let solution = solver
            .get_result()
            .expect("frozen BVP solver should store a solution matrix");
        assert_eq!(solution.nrows(), n_steps);
        assert_eq!(solution.ncols(), 2);

        let h = 1.0 / n_steps as f64;
        let mut sq_sum = 0.0;
        let mut max_abs = 0.0;
        for i in 0..solution.nrows() {
            // Frozen stores the reduced unknown vector. With both linear BVP
            // boundary conditions placed at the left edge, row 0 corresponds
            // to the first free mesh point, not to the boundary itself.
            let x = (i + 1) as f64 * h;
            let y_err = (solution[(i, 0)] - x).abs();
            let z_err = (solution[(i, 1)] - 1.0).abs();
            assert!(
                solution[(i, 0)].is_finite(),
                "frozen y contains non-finite values"
            );
            assert!(
                solution[(i, 1)].is_finite(),
                "frozen z contains non-finite values"
            );
            let err = y_err.max(z_err);
            sq_sum += err * err;
            max_abs = f64::max(max_abs, err);
        }
        let rms = (sq_sum / solution.nrows() as f64).sqrt();
        assert!(
            rms <= rms_tol,
            "frozen linear BVP RMS error too large: rms={rms:e}, tol={rms_tol:e}"
        );
        assert!(
            max_abs <= max_abs_tol,
            "frozen linear BVP max error too large: max={max_abs:e}, tol={max_abs_tol:e}"
        );
    }

    #[derive(Debug)]
    struct FrozenCombustionStoryRow {
        source: &'static str,
        variant: &'static str,
        total_ms: f64,
        solution_diff: f64,
        symbolic_ms: f64,
        linear_ms: f64,
        jacobian_ms: f64,
        residual_ms: f64,
        initial_generate_ms: f64,
        initial_symbolic_jacobian_ms: f64,
        post_build_rebind_ms: f64,
        compile_link_ms: f64,
        residual_jobs: f64,
        jacobian_jobs: f64,
        iterations: usize,
        linear_solves: usize,
        jacobian_rebuilds: usize,
        selected_backend: String,
        build_policy: String,
    }

    fn frozen_combustion_solver(
        n_steps: usize,
        matrix: &'static str,
        config: GeneratedBackendConfig,
    ) -> NRBVP {
        let names = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
        let unknowns = Expr::parse_vector_expression(names.clone());
        let teta = unknowns[0].clone();
        let q = unknowns[1].clone();
        let c0 = unknowns[2].clone();
        let j0 = unknowns[3].clone();
        let j1 = unknowns[5].clone();

        let dt = Expr::Const(600.0);
        let t_scale = Expr::Const(600.0);
        let lambda = Expr::Const(0.07);
        let q_heat = Expr::Const(3000.0 * 1e3 * 0.034);
        let a = Expr::Const(1.3e5);
        let e = Expr::Const(5000.0 * 4.184);
        let m = Expr::Const(34.2 / 1000.0);
        let gas_r = Expr::Const(8.314);
        let ro_m = Expr::Const((34.2 / 1000.0) * 2e6 / (8.314 * 1500.0));
        let qm = Expr::Const((3e-4_f64).powi(2) / 600.0);
        let qs = Expr::Const((3e-4_f64).powi(2));
        let ro_d = Expr::Const(2.88e-4);
        let pe_d = Expr::Const(1.50e-3);
        let rate = a
            * Expr::exp(-e / (gas_r * (teta.clone() * t_scale.clone() + dt.clone())))
            * c0.clone()
            * (ro_m.clone() / Expr::Const(0.342));
        let eqs = vec![
            q.clone() / lambda,
            q * Expr::Const(0.0090168) - q_heat * rate.clone() * qm,
            j0.clone() / ro_d.clone(),
            j0 * pe_d.clone()
                - (m.clone() * Expr::Const(-1.0) * rate.clone() * ro_m.clone() / m.clone())
                    * qs.clone(),
            j1.clone() / ro_d,
            j1 * pe_d - (m.clone() * rate * ro_m / m) * qs,
        ];
        let boundary_conditions = HashMap::from([
            ("Teta".to_string(), vec![(0, (1000.0 - 600.0) / 600.0)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(), vec![(0, 1.0)]),
            ("J0".to_string(), vec![(1, 1e-7)]),
            ("C1".to_string(), vec![(0, 1e-3)]),
            ("J1".to_string(), vec![(1, 1e-7)]),
        ]);
        let initial_guess = DMatrix::from_element(names.len(), n_steps, 0.99);
        let options = match matrix {
            "Sparse" => FrozenSolverOptions::sparse_frozen(),
            "Banded" => FrozenSolverOptions::banded_frozen(),
            _ => panic!("unsupported Frozen combustion story matrix route: {matrix}"),
        }
        .with_generated_backend_config(config)
        .with_tolerance(1e-6)
        .with_max_iterations(100);
        let mut solver = NRBVP::new_with_options(
            eqs,
            initial_guess,
            names.iter().map(|name| (*name).to_string()).collect(),
            "x".to_string(),
            boundary_conditions,
            0.0,
            1.0,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn frozen_story_timer_ms(stats: &FrozenBvpStatistics, prefix: &str) -> f64 {
        stats
            .timers
            .iter()
            .find(|(name, _)| name.starts_with(prefix))
            .and_then(|(name, value)| {
                let value = value
                    .split(',')
                    .next_back()
                    .and_then(|raw| raw.trim().parse::<f64>().ok())?;
                Some(if name.contains("ms") {
                    value
                } else {
                    value * 1_000.0
                })
            })
            .unwrap_or(f64::NAN)
    }

    fn frozen_story_diagnostic_ms(stats: &FrozenBvpStatistics, key: &str) -> f64 {
        stats
            .diagnostics
            .get(key)
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    }

    fn frozen_story_diagnostic_string(stats: &FrozenBvpStatistics, key: &str) -> String {
        stats
            .diagnostics
            .get(key)
            .cloned()
            .unwrap_or_else(|| "-".to_string())
    }

    fn frozen_story_linf_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    fn run_frozen_combustion_story_row(
        n_steps: usize,
        matrix: &'static str,
        source: &'static str,
        variant: &'static str,
        config: GeneratedBackendConfig,
        baseline: Option<&DMatrix<f64>>,
    ) -> (
        FrozenCombustionStoryRow,
        DMatrix<f64>,
        GeneratedBackendConfig,
    ) {
        let begin = Instant::now();
        let mut solver = frozen_combustion_solver(n_steps, matrix, config);
        solver.try_solve().unwrap_or_else(|err| {
            panic!("{source}/{variant} Frozen combustion solve failed: {err:?}")
        });
        collect_frozen_story_row(begin, solver, source, variant, baseline)
    }

    fn collect_frozen_story_row(
        begin: Instant,
        solver: NRBVP,
        source: &'static str,
        variant: &'static str,
        baseline: Option<&DMatrix<f64>>,
    ) -> (
        FrozenCombustionStoryRow,
        DMatrix<f64>,
        GeneratedBackendConfig,
    ) {
        let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
        let solution = solver
            .get_result()
            .expect("successful Frozen solve should expose its result");
        assert!(
            solution.iter().all(|value| value.is_finite()),
            "{source}/{variant} returned non-finite values"
        );
        let stats = solver.get_statistics();
        let row = FrozenCombustionStoryRow {
            source,
            variant,
            total_ms,
            solution_diff: baseline
                .map(|reference| frozen_story_linf_diff(&solution, reference))
                .unwrap_or(0.0),
            symbolic_ms: frozen_story_timer_ms(&stats, "Symbolic Operations"),
            linear_ms: frozen_story_timer_ms(&stats, "Linear System"),
            jacobian_ms: frozen_story_timer_ms(&stats, "Jacobian"),
            residual_ms: frozen_story_timer_ms(&stats, "Function"),
            initial_generate_ms: frozen_story_diagnostic_ms(
                &stats,
                "generated.handoff.initial_generate_wall_ms",
            ),
            initial_symbolic_jacobian_ms: frozen_story_diagnostic_ms(
                &stats,
                "generated.handoff.initial.symbolic_jacobian_time_ms",
            ),
            post_build_rebind_ms: frozen_story_diagnostic_ms(
                &stats,
                "generated.handoff.post_build_rebind_wall_ms",
            ),
            compile_link_ms: frozen_story_diagnostic_ms(&stats, "generated.aot.compile_link_ms"),
            residual_jobs: frozen_story_diagnostic_ms(&stats, "aot.runtime.residual.actual_jobs"),
            jacobian_jobs: frozen_story_diagnostic_ms(
                &stats,
                "aot.runtime.sparse_jacobian.actual_jobs",
            ),
            iterations: stats.counters["number of iterations"],
            linear_solves: stats.counters["number of solving linear systems"],
            jacobian_rebuilds: stats.counters["number of jacobians recalculations"],
            selected_backend: frozen_story_diagnostic_string(&stats, "generated.selected_backend"),
            build_policy: frozen_story_diagnostic_string(&stats, "aot.build_policy"),
        };
        (row, solution, solver.generated_backend_config().clone())
    }

    fn frozen_polynomial_two_point_solver(n_steps: usize, config: GeneratedBackendConfig) -> NRBVP {
        // Non-combustion nonlinear BVP with exact solution y = 1 + x^2:
        // y' = z,
        // z' = 2 + 0.1 * (y - (1 + x^2))^2.
        //
        // Frozen currently requires a boundary-condition key for every state
        // variable, so we use y(-1) and z(1), both taken from the exact profile.
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = -1.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + i as f64 * h;
            let y = 1.0 + x * x;
            let z = 2.0 * x;
            guess[i * values.len()] = y;
            guess[i * values.len() + 1] = z;
        }

        let y_left = 2.0;
        let z_right = 2.0;
        let options = FrozenSolverOptions::banded_frozen()
            .with_generated_backend_config(config)
            .with_tolerance(1e-6)
            .with_max_iterations(40);
        let mut solver = NRBVP::new_with_options(
            vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("2.0 + 0.1*(y - (1.0 + x*x))^2"),
            ],
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice()),
            values,
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, y_left)]),
                ("z".to_string(), vec![(1usize, z_right)]),
            ]),
            t0,
            t_end,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn run_frozen_polynomial_story_row(
        n_steps: usize,
        source: &'static str,
        variant: &'static str,
        config: GeneratedBackendConfig,
        baseline: Option<&DMatrix<f64>>,
    ) -> (
        FrozenCombustionStoryRow,
        DMatrix<f64>,
        GeneratedBackendConfig,
    ) {
        let begin = Instant::now();
        let mut solver = frozen_polynomial_two_point_solver(n_steps, config);
        let solve_result = solver.try_solve().unwrap_or_else(|err| {
            panic!("{source}/{variant} Frozen nonlinear polynomial solve failed: {err:?}")
        });
        assert!(
            solve_result.is_some(),
            "{source}/{variant} Frozen nonlinear polynomial solve did not converge within the configured iteration/tolerance budget"
        );
        collect_frozen_story_row(begin, solver, source, variant, baseline)
    }

    fn frozen_story_mean_std(values: &[f64]) -> (f64, f64) {
        let finite = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if finite.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let variance = finite
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / finite.len() as f64;
        (mean, variance.sqrt())
    }

    fn print_frozen_combustion_story(title: &str, rows: &[FrozenCombustionStoryRow]) {
        println!("[BVP Frozen story] {title}: correctness/backend selection");
        println!("source   | variant    | selected_backend | build_policy    | solve_diff");
        println!("{}", "-".repeat(82));
        for row in rows {
            println!(
                "{:<8} | {:<10} | {:<16} | {:<15} | {:.6e}",
                row.source, row.variant, row.selected_backend, row.build_policy, row.solution_diff
            );
        }
        println!();
        println!("[BVP Frozen story] {title}: wall-clock and Newton stages; milliseconds");
        println!(
            "source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re"
        );
        println!("{}", "-".repeat(118));
        for row in rows {
            println!(
                "{:<8} | {:<10} | {:>8.3} | {:>11.3} | {:>9.3} | {:>6.3} | {:>6.3} | {:>5} | {:>6} | {:>6}",
                row.source,
                row.variant,
                row.total_ms,
                row.symbolic_ms,
                row.linear_ms,
                row.jacobian_ms,
                row.residual_ms,
                row.iterations,
                row.linear_solves,
                row.jacobian_rebuilds,
            );
        }
        println!();
        println!(
            "[BVP Frozen story] {title}: generated handoff and compiled callback stages; milliseconds"
        );
        println!(
            "source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs"
        );
        println!("{}", "-".repeat(120));
        for row in rows {
            println!(
                "{:<8} | {:<10} | {:>16.3} | {:>15.3} | {:>9.3} | {:>12.3} | {:>8.3} | {:>8.3}",
                row.source,
                row.variant,
                row.initial_generate_ms,
                row.initial_symbolic_jacobian_ms,
                row.post_build_rebind_ms,
                row.compile_link_ms,
                row.residual_jobs,
                row.jacobian_jobs,
            );
        }
        println!();
        println!("[BVP Frozen story] {title}: repeated-run summary; milliseconds");
        println!(
            "source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff"
        );
        println!("{}", "-".repeat(126));
        let mut identities = rows
            .iter()
            .map(|row| (row.source, row.variant))
            .collect::<Vec<_>>();
        identities.sort_unstable();
        identities.dedup();
        for (source, variant) in identities {
            let selected = rows
                .iter()
                .filter(|row| row.source == source && row.variant == variant)
                .collect::<Vec<_>>();
            let (total_mean, total_std) =
                frozen_story_mean_std(&selected.iter().map(|row| row.total_ms).collect::<Vec<_>>());
            let (symbolic_mean, symbolic_std) = frozen_story_mean_std(
                &selected
                    .iter()
                    .map(|row| row.symbolic_ms)
                    .collect::<Vec<_>>(),
            );
            let (linear_mean, linear_std) = frozen_story_mean_std(
                &selected.iter().map(|row| row.linear_ms).collect::<Vec<_>>(),
            );
            let max_diff = selected
                .iter()
                .map(|row| row.solution_diff)
                .fold(0.0_f64, f64::max);
            println!(
                "{:<8} | {:<10} | {:>9.3} +/- {:<9.3} | {:>12.3} +/- {:<9.3} | {:>10.3} +/- {:<9.3} | {:.6e}",
                source,
                variant,
                total_mean,
                total_std,
                symbolic_mean,
                symbolic_std,
                linear_mean,
                linear_std,
                max_diff,
            );
        }
    }

    fn frozen_tcc_config(
        matrix: &'static str,
        policy: AotBuildPolicy,
        chunking: AotChunkingPolicy,
        execution: AotExecutionPolicy,
    ) -> GeneratedBackendConfig {
        let config = match matrix {
            "Sparse" => GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
            "Banded" => GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
            _ => panic!("unsupported Frozen tcc story matrix route: {matrix}"),
        };
        config
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("tcc")
            .with_aot_compile_dev_fastest()
            .with_aot_chunking_policy(chunking)
            .with_aot_execution_policy(execution)
            .with_aot_build_policy(policy)
    }

    fn frozen_whole_chunking() -> AotChunkingPolicy {
        AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::Whole),
            Some(SparseChunkingStrategy::Whole),
        )
    }

    fn frozen_chunk4_execution() -> (AotChunkingPolicy, AotExecutionPolicy) {
        (
            AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            ),
            AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(4),
                max_sparse_jobs: Some(4),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
        )
    }

    #[test]
    #[ignore = "heavy Frozen combustion-1000 Banded AtomView Lambdify vs tcc whole/chunk4 end-to-end story; run in release with --nocapture"]
    fn frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story() {
        let n_steps = 1_000;
        let repetitions = 2;
        let mut rows = Vec::new();
        let (chunk4, parallel) = frozen_chunk4_execution();

        for _ in 0..repetitions {
            let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "Lambdify",
                "AtomView",
                GeneratedBackendConfig::banded_lambdify_defaults(),
                None,
            );
            rows.push(baseline_row);

            let (whole_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "AOT",
                "tcc/whole",
                frozen_tcc_config(
                    "Banded",
                    AotBuildPolicy::RebuildAlways {
                        profile: AotBuildProfile::Release,
                    },
                    frozen_whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                Some(&baseline),
            );
            rows.push(whole_row);

            let (chunked_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "AOT",
                "tcc/chunk4",
                frozen_tcc_config(
                    "Banded",
                    AotBuildPolicy::RebuildAlways {
                        profile: AotBuildProfile::Release,
                    },
                    chunk4,
                    parallel.clone(),
                ),
                Some(&baseline),
            );
            rows.push(chunked_row);
        }

        print_frozen_combustion_story(
            "combustion-1000 Banded AtomView Lambdify vs tcc AOT cold routes",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-5),
            "Frozen AOT variants must remain equivalent to the Lambdify baseline"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "Frozen AOT cold routes must execute freshly compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "tcc/chunk4")
                .all(|row| row.residual_jobs > 1.0 && row.jacobian_jobs > 1.0),
            "Frozen chunk4 route must expose real callback-level parallel execution"
        );
    }

    #[test]
    #[ignore = "heavy Frozen combustion-1000 AOT artifact lifecycle story: BuildIfMissing followed by strict RequirePrebuilt reuse"]
    fn frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story() {
        let n_steps = 1_000;
        let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "Lambdify",
            "AtomView",
            GeneratedBackendConfig::banded_lambdify_defaults(),
            None,
        );
        let (built_row, _, built_config) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "AOT",
            "build",
            frozen_tcc_config(
                "Banded",
                AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
                frozen_whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            Some(&baseline),
        );
        assert!(
            built_config.resolver.is_some(),
            "BuildIfMissing must leave a resolver snapshot for strict reuse"
        );
        let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        let mut rows = vec![baseline_row, built_row];
        for _ in 0..3 {
            let (prebuilt_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "AOT",
                "prebuilt",
                strict_config.clone(),
                Some(&baseline),
            );
            rows.push(prebuilt_row);
        }

        print_frozen_combustion_story(
            "combustion-1000 Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-5),
            "Frozen lifecycle rows must remain equivalent to Lambdify"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "both built and strict prebuilt rows must execute compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "prebuilt")
                .all(|row| row.build_policy == "RequirePrebuilt"),
            "warm rows must be strict RequirePrebuilt executions, not fallback builds"
        );
    }

    #[test]
    #[ignore = "heavy Frozen combustion-1000 Sparse AtomView tcc artifact lifecycle: BuildIfMissing followed by strict RequirePrebuilt reuse"]
    fn frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story() {
        let n_steps = 1_000;
        let sparse_lambdify = GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
        let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
            n_steps,
            "Sparse",
            "Lambdify",
            "AtomView",
            sparse_lambdify,
            None,
        );
        let (built_row, _, built_config) = run_frozen_combustion_story_row(
            n_steps,
            "Sparse",
            "AOT",
            "build",
            frozen_tcc_config(
                "Sparse",
                AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
                frozen_whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            Some(&baseline),
        );
        assert!(
            built_config.resolver.is_some(),
            "Sparse BuildIfMissing must leave a resolver snapshot for strict reuse"
        );
        let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        let mut rows = vec![baseline_row, built_row];
        for _ in 0..3 {
            let (prebuilt_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Sparse",
                "AOT",
                "prebuilt",
                strict_config.clone(),
                Some(&baseline),
            );
            rows.push(prebuilt_row);
        }

        print_frozen_combustion_story(
            "combustion-1000 Sparse AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-5),
            "Frozen Sparse lifecycle rows must remain equivalent to Lambdify"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "Frozen Sparse build and strict prebuilt rows must execute compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "prebuilt")
                .all(|row| {
                    row.build_policy == "RequirePrebuilt" && row.compile_link_ms.is_nan()
                }),
            "Frozen Sparse prebuilt rows must neither fall back nor compile again"
        );
    }

    #[test]
    #[ignore = "Frozen non-combustion nonlinear polynomial BVP: Banded AtomView Lambdify vs tcc BuildIfMissing -> RequirePrebuilt"]
    fn frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story() {
        let n_steps = 80;
        let (baseline_row, baseline, _) = run_frozen_polynomial_story_row(
            n_steps,
            "Lambdify",
            "AtomView",
            GeneratedBackendConfig::banded_lambdify_defaults(),
            None,
        );
        let (built_row, _, built_config) = run_frozen_polynomial_story_row(
            n_steps,
            "AOT",
            "build",
            frozen_tcc_config(
                "Banded",
                AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
                frozen_whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            Some(&baseline),
        );
        assert!(
            built_config.resolver.is_some(),
            "Polynomial BuildIfMissing must leave a resolver snapshot for strict reuse"
        );
        let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        let mut rows = vec![baseline_row, built_row];
        for _ in 0..2 {
            let (prebuilt_row, _, _) = run_frozen_polynomial_story_row(
                n_steps,
                "AOT",
                "prebuilt",
                strict_config.clone(),
                Some(&baseline),
            );
            rows.push(prebuilt_row);
        }

        print_frozen_combustion_story(
            "nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-6),
            "Frozen nonlinear polynomial lifecycle rows must remain equivalent to Lambdify"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "Frozen nonlinear polynomial build and strict prebuilt rows must execute compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "prebuilt")
                .all(|row| {
                    row.build_policy == "RequirePrebuilt" && row.compile_link_ms.is_nan()
                }),
            "Frozen nonlinear polynomial prebuilt rows must neither fall back nor compile again"
        );
    }

    #[test]
    fn generated_backend_surface_builder_methods_update_solver_config() {
        let solver = sparse_surface_test_solver()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
            .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
            ))
            .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            solver.aot_execution_policy(),
            &AotExecutionPolicy::SequentialOnly
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
        assert_eq!(
            solver.aot_chunking_policy(),
            AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
            )
        );
        assert_eq!(
            solver.atom_optimization_profile(),
            AtomOptimizationProfile::NoCse
        );
    }

    #[test]
    fn symbolic_assembly_backend_is_exposed_on_solver_surface() {
        let mut solver = sparse_surface_test_solver()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);

        assert_eq!(
            solver.symbolic_assembly_backend(),
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            solver.generated_backend_config().symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );

        solver.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
        assert_eq!(
            solver.symbolic_assembly_backend(),
            BvpSymbolicAssemblyBackend::ExprLegacy
        );
        assert_eq!(
            solver.generated_backend_config().symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::ExprLegacy
        );
    }

    #[test]
    fn banded_frozen_lambdify_mode_sets_banded_matrix_and_lambdify_policy() {
        let options = FrozenSolverOptions::banded_frozen().with_banded_lambdify();

        assert_eq!(options.method, "Sparse");
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::LambdifyOnly)
        );
        assert_eq!(
            options.generated_backend_config.matrix_backend_override,
            Some(MatrixBackend::Banded)
        );
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::UseIfAvailable
        );
    }

    #[test]
    fn banded_frozen_generated_backend_mode_build_if_missing_sets_release_aot_policy() {
        let options = FrozenSolverOptions::banded_frozen()
            .with_banded_generated_backend_mode(BandedGeneratedBackendMode::BuildIfMissingRelease);

        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            options.generated_backend_config.matrix_backend_override,
            Some(MatrixBackend::Banded)
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn frozen_sparse_lambdify_linear_bvp_solves_against_exact_profile() {
        let options = FrozenSolverOptions::sparse_frozen()
            .with_generated_backend_config(
                GeneratedBackendConfig::sparse_defaults()
                    .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly)),
            )
            .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
            .with_tolerance(1e-8)
            .with_max_iterations(20);
        let mut solver = frozen_linear_solver(24, options);
        solver.dont_save_log(true);

        solver
            .try_solve()
            .expect("sparse frozen lambdify linear BVP should solve");

        assert_frozen_linear_solution_quality(&solver, 24, 1e-10, 1e-9);
        assert!(
            solver.jac.is_some(),
            "sparse frozen route should prepare a Jacobian"
        );
        assert!(
            !solver.variable_string.is_empty(),
            "sparse frozen route should prepare reduced variable metadata"
        );
    }

    #[test]
    fn frozen_banded_default_atomview_lambdify_linear_bvp_solves_against_exact_profile() {
        let options = FrozenSolverOptions::banded_frozen()
            .with_banded_lambdify()
            .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
            .with_tolerance(1e-8)
            .with_max_iterations(20);
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        let mut solver = frozen_linear_solver(24, options);
        solver.dont_save_log(true);

        solver
            .try_solve()
            .expect("default banded frozen AtomView lambdify linear BVP should solve");

        assert_frozen_linear_solution_quality(&solver, 24, 1e-7, 1e-6);
        assert!(
            solver.jac.is_some(),
            "banded frozen route should prepare a Jacobian"
        );
        assert!(
            solver.bandwidth.0 + solver.bandwidth.1 > 0,
            "banded frozen route should expose non-empty bandwidth metadata"
        );
        let stats = solver.get_statistics();
        assert_eq!(
            stats.diagnostics.get("generated.selected_backend"),
            Some(&"Lambdify".to_string()),
            "Frozen statistics must report the backend that supplied callbacks"
        );
        assert_eq!(
            stats.diagnostics.get("generated.symbolic_assembly_backend"),
            Some(&"AtomView".to_string())
        );
        assert!(
            stats
                .diagnostics
                .contains_key("generated.handoff.initial_generate_wall_ms"),
            "Frozen must preserve symbolic handoff timing diagnostics"
        );
        assert!(
            stats.counters["number of iterations"] > 0
                && stats.counters["number of jacobians recalculations"] > 0
                && stats.counters["number of solving linear systems"] > 0,
            "Frozen end-to-end solve must expose its Newton work counters: {:?}",
            stats.counters
        );
        assert!(
            stats
                .timers
                .keys()
                .any(|key| key.starts_with("Symbolic Operations")),
            "Frozen end-to-end solve must expose backend preparation timing"
        );
    }

    #[test]
    fn try_eq_generate_surfaces_missing_prebuilt_aot_as_typed_error() {
        let mut solver =
            sparse_surface_test_solver_with_naive_strategy().with_sparse_aot_require_prebuilt();

        let err = solver
            .try_eq_generate()
            .expect_err("try_eq_generate should return a typed AOT availability error");

        assert!(matches!(
            err,
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
        ));
    }

    #[test]
    fn try_solve_surfaces_missing_prebuilt_aot_as_typed_error() {
        let mut solver =
            sparse_surface_test_solver_with_naive_strategy().with_sparse_aot_require_prebuilt();
        solver.dont_save_log(true);

        let err = solver
            .try_solve()
            .expect_err("try_solve should return a typed AOT availability error");

        assert!(matches!(
            err,
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
        ));
    }

    #[test]
    fn sparse_generated_backend_presets_are_exposed_on_solver_surface() {
        let solver = sparse_surface_test_solver().with_sparse_aot_build_if_missing_release();

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            solver.aot_build_policy(),
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn sparse_generated_backend_mode_is_exposed_on_solver_surface() {
        let mut solver = sparse_surface_test_solver()
            .with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt);

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);

        solver.set_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease);
        assert_eq!(
            solver.aot_build_policy(),
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn sparse_frozen_options_preset_sets_production_defaults() {
        let options = FrozenSolverOptions::sparse_frozen();

        assert_eq!(options.scheme, "forward");
        assert_eq!(options.strategy, "Frozen");
        assert_eq!(options.method, "Sparse");
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::UseIfAvailable
        );
        assert_eq!(
            options.generated_backend_config.aot_execution_policy,
            AotExecutionPolicy::Auto
        );
        assert_eq!(
            options.generated_backend_config.aot_chunking_policy,
            AotChunkingPolicy::default()
        );
    }

    #[test]
    fn banded_frozen_options_preset_uses_auto_aot_chunking_defaults() {
        let options = FrozenSolverOptions::banded_frozen();

        assert_eq!(
            options.generated_backend_config.matrix_backend_override,
            Some(MatrixBackend::Banded)
        );
        assert_eq!(
            options.generated_backend_config.aot_execution_policy,
            AotExecutionPolicy::Auto
        );
        assert_eq!(
            options.generated_backend_config.aot_chunking_policy,
            AotChunkingPolicy::default()
        );
    }

    #[test]
    fn frozen_options_scheme_builder_methods_set_legacy_scheme_flag() {
        let options = FrozenSolverOptions::sparse_frozen().trapezoid_derivative();
        assert_eq!(options.scheme, "trapezoid");

        let options = options.forward_derivative();
        assert_eq!(options.scheme, "forward");

        let options = options.with_scheme(BvpDerivativeScheme::Trapezoid);
        assert_eq!(options.scheme, "trapezoid");

        let options = options.with_scheme_name("custom-experimental");
        assert_eq!(options.scheme, "custom-experimental");
    }

    #[test]
    fn frozen_solver_scheme_builder_methods_feed_generated_request() {
        let solver = sparse_surface_test_solver().trapezoid_derivative();
        assert_eq!(solver.scheme, "trapezoid");
        assert_eq!(solver.build_solver_request().scheme, "trapezoid");

        let solver = solver.forward_derivative();
        assert_eq!(solver.scheme, "forward");
        assert_eq!(solver.build_solver_request().scheme, "forward");

        let solver = solver.with_scheme(BvpDerivativeScheme::Trapezoid);
        assert_eq!(solver.scheme, "trapezoid");
        assert_eq!(solver.build_solver_request().scheme, "trapezoid");

        let solver = solver.with_scheme_name("custom-experimental");
        assert_eq!(solver.scheme, "custom-experimental");
        assert_eq!(solver.build_solver_request().scheme, "custom-experimental");
    }

    #[test]
    fn dense_frozen_options_preset_sets_dense_defaults() {
        let options = FrozenSolverOptions::dense_frozen();

        assert_eq!(options.strategy, "Frozen");
        assert_eq!(options.method, "Dense");
    }

    #[test]
    fn dense_naive_options_preset_sets_dense_defaults() {
        let options = FrozenSolverOptions::dense_naive();

        assert_eq!(options.strategy, "Naive");
        assert!(options.strategy_params.is_none());
        assert_eq!(options.method, "Dense");
    }

    #[test]
    fn constructor_style_sparse_generated_backend_mode_sets_solver_config() {
        let solver = NRBVP::new_with_sparse_generated_backend_mode(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.1),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            0.0,
            1.0,
            4,
            "Frozen".to_string(),
            None,
            None,
            "Sparse".to_string(),
            1e-6,
            10,
            SparseGeneratedBackendMode::RequirePrebuilt,
        );

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
    }

    #[test]
    fn options_style_solver_setup_sets_sparse_generated_backend_mode() {
        let options = FrozenSolverOptions::sparse_frozen().with_sparse_aot_require_prebuilt();

        let solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.1),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            0.0,
            1.0,
            4,
            options,
        );

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
    }

    #[test]
    fn test_newton_raphson_solver() {
        // Define a simple equation: x^2 - 4 = 0
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-2;
        let max_iterations = 100;

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 100;
        let ones = vec![1.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
        assert_eq!(&eq_system.len(), &2);
        let options = FrozenSolverOptions::dense_naive()
            .with_tolerance(tolerance)
            .with_max_iterations(max_iterations);
        let mut nr = NRBVP::new_with_options(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            options,
        );
        nr.try_eq_generate()
            .expect("dense frozen solver should generate through the fallible API");

        assert_eq!(nr.eq_system.len(), 2);
        nr.dont_save_log(true);
        // Solve the equation at t=0 with initial guess y=[2.0]
        //    nr.set_new_step(0.0, DVector::from_element(1, 2.0), DVector::from_element(1, 2.0));
        let _solution = nr
            .try_solve()
            .expect("dense frozen solver should solve through the fallible API")
            .unwrap();
    }

    #[test]
    fn sparse_eq_generate_uses_bundle_handoff_without_breaking_metadata() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();

        solver
            .try_eq_generate()
            .expect("sparse frozen handoff should generate through the fallible API");

        assert!(solver.jac.is_some());
        assert!(!solver.variable_string.is_empty());
        assert!(solver.bandwidth.0 + solver.bandwidth.1 > 0);
        let _ = &solver.fun;
    }

    #[test]
    fn numeric_only_is_rejected_for_frozen_solver_instead_of_lambdify_fallback() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        let err = solver
            .try_eq_generate()
            .expect_err("Frozen NumericOnly must not silently fall back to symbolic lambdify");
        assert!(matches!(
            err,
            BvpBackendIntegrationError::PipelinePanicked(message)
                if message.contains("not available for the frozen BVP solver")
        ));
    }

    #[test]
    fn build_solver_request_carries_optional_aot_resolver() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));

        let request = solver.build_solver_request();
        assert!(request.resolver.is_some());
    }

    #[test]
    fn build_solver_request_carries_parameter_names_and_values() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_params(Some(&["alpha", "beta"]));
        solver.set_param_values(Some(vec![1.5, -0.25]));

        let request = solver.build_solver_request();
        assert_eq!(
            request.param_names,
            Some(vec!["alpha".to_string(), "beta".to_string()])
        );
        assert_eq!(request.param_values, Some(vec![1.5, -0.25]));
    }

    #[test]
    #[should_panic(expected = "param_values length must match param_names length")]
    fn solver_surface_rejects_parameter_value_length_mismatch() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_params(Some(&["alpha", "beta"]));
        solver.set_param_values(Some(vec![1.5]));
    }

    #[test]
    fn build_solver_request_uses_backend_policy_override_when_present() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        let request = solver.build_solver_request();
        assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
    }

    #[test]
    fn generated_backend_config_is_exposed_as_user_facing_solver_setting() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();

        let config = GeneratedBackendConfig::with_parts(
            Some(BackendSelectionPolicy::NumericOnly),
            Some(AotResolver::new(AotRegistry::new())),
        );
        solver.set_generated_backend_config(config);

        let request = solver.build_solver_request();
        assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
        assert!(request.resolver.is_some());
        assert_eq!(
            solver.generated_backend_config().backend_policy_override,
            Some(BackendSelectionPolicy::NumericOnly)
        );
    }

    #[test]
    fn generated_backend_config_can_be_applied_during_solver_construction() {
        let solver = sparse_surface_test_solver_with_naive_strategy()
            .with_generated_backend_config(GeneratedBackendConfig::with_parts(
                Some(BackendSelectionPolicy::NumericOnly),
                Some(AotResolver::new(AotRegistry::new())),
            ));

        assert_eq!(
            solver.generated_backend_config().backend_policy_override,
            Some(BackendSelectionPolicy::NumericOnly)
        );
        assert!(solver.generated_backend_config().resolver.is_some());
    }

    #[test]
    fn cleanup_registered_aot_artifacts_is_safe_without_registered_artifacts() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);

        solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));
        assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);
    }

    #[test]
    fn build_solver_request_carries_surface_aot_policies() {
        let config = GeneratedBackendConfig::new()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_resolver(Some(AotResolver::new(AotRegistry::new())))
            .with_aot_execution_policy(AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 2,
                max_residual_jobs: Some(4),
                max_sparse_jobs: Some(2),
                fallback_policy: ParallelFallbackPolicy::Never,
            }))
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
            .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByOutputCount {
                    max_outputs_per_chunk: 6,
                }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
            ))
            .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

        let solver =
            sparse_surface_test_solver_with_naive_strategy().with_generated_backend_config(config);

        let request = solver.build_solver_request();
        assert_eq!(request.aot_build_policy, AotBuildPolicy::RequirePrebuilt);
        assert_eq!(
            request.aot_chunking_policy.residual,
            Some(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 6
            })
        );
        assert_eq!(
            request.aot_chunking_policy.sparse_jacobian,
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 })
        );
        assert_eq!(
            request.atom_optimization_profile,
            AtomOptimizationProfile::NoCse
        );
        match request.aot_execution_policy {
            AotExecutionPolicy::Parallel(inner) => {
                assert_eq!(inner.jobs_per_worker, 2);
                assert_eq!(inner.max_residual_jobs, Some(4));
                assert_eq!(inner.max_sparse_jobs, Some(2));
            }
            other => std::panic!("expected parallel execution policy, got {other:?}"),
        }
        let _ = AotBuildProfile::Debug;
    }

    #[test]
    fn eq_generate_build_if_missing_saves_compiled_resolver_for_next_request() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy()
            .with_generated_backend_config(
                GeneratedBackendConfig::new()
                    .with_backend_policy_override(Some(
                        BackendSelectionPolicy::PreferAotThenLambdify,
                    ))
                    .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Release,
                    }),
            );

        solver
            .try_eq_generate()
            .expect("build-if-missing path should generate through the fallible API");

        let saved_resolver = solver
            .generated_backend_config()
            .resolver
            .as_ref()
            .expect("first build-if-missing run should save updated resolver");
        assert!(
            !saved_resolver.registry().is_empty(),
            "first build-if-missing run should register at least one compiled artifact"
        );

        let updated_config = solver
            .generated_backend_config()
            .clone()
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        solver.set_generated_backend_config(updated_config);

        let next_request = solver.build_solver_request();
        assert!(next_request.resolver.is_some());

        let next_state = next_request.generate().expect(
            "next request should reuse the saved compiled resolver and generate successfully",
        );
        assert!(
            next_state.jac.is_some(),
            "successful generation through the saved resolver should still provide a Jacobian callback"
        );
    }

    #[test]
    fn second_eq_generate_reuses_saved_resolver_and_runs_linked_compiled_backend() {
        let values = vec!["z".to_string(), "y".to_string()];
        let n_steps = 8;
        let mut solver = sparse_surface_test_solver_with_naive_strategy()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            });

        solver
            .try_eq_generate()
            .expect("first sparse generation should succeed through the fallible API");
        let saved_resolver = solver
            .generated_backend_config()
            .resolver
            .clone()
            .expect("first build-if-missing run should save updated resolver");

        let y = Col::from_fn(values.len() * n_steps, |index| 0.3 + index as f64 * 0.02);
        let baseline = solver.fun.call(0.0, &y).to_DVectorType();

        let problem_keys = saved_resolver.registry().problem_keys();
        assert_eq!(
            problem_keys.len(),
            1,
            "build-if-missing should register exactly one artifact for this isolated test"
        );
        let problem_key = problem_keys[0].clone();
        let resolved = saved_resolver.resolve_by_problem_key(&problem_key);
        assert!(
            resolved.is_compiled(),
            "saved resolver should see compiled artifact"
        );

        let baseline_values: Vec<f64> = baseline.iter().copied().collect();
        register_linked_sparse_backend(LinkedSparseAotBackend::new(
            problem_key.clone(),
            resolved.registered.manifest.io.residual_len,
            (
                resolved.registered.manifest.io.jacobian_rows,
                resolved.registered.manifest.io.jacobian_cols,
            ),
            resolved.registered.manifest.io.jacobian_nnz.unwrap_or(0),
            Arc::new(move |_args, out| {
                for (dst, src) in out.iter_mut().zip(baseline_values.iter()) {
                    *dst = *src + 55.0;
                }
            }),
            Arc::new(move |_args, out| {
                for (index, value) in out.iter_mut().enumerate() {
                    *value = 900.0 + index as f64;
                }
            }),
        ));

        solver.set_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        solver
            .try_eq_generate()
            .expect("second sparse generation should reuse resolver through the fallible API");
        let residual = solver.fun.call(0.0, &y).to_DVectorType();

        for (actual, expected) in residual.iter().zip(baseline.iter()) {
            assert!((actual - (expected + 55.0)).abs() < 1e-10);
        }

        unregister_linked_sparse_backend(&problem_key);
    }
}
