//! Modern universal IVP facade.
//!
//! This module is the user-facing entry point for selecting IVP solvers without
//! manually navigating the solver-specific modules. Unlike the historical
//! `HashMap<String, SolverParam>` facade, this implementation routes through the
//! current typed solver options of `BE`, `BDF`, and `Radau`, while still keeping
//! thin legacy wrappers for older call-sites that pass string-keyed parameter
//! bags.

use crate::Utils::plots::plots_ref;
use crate::numerical::BDF::BDF_api::{
    BdfSolveError, BdfSolverOptions, BdfStopConditionError, BdfTelemetryMode,
    ODEsolver as BdfOdeSolver,
};
use crate::numerical::BE::{BE, BeError, BeSolverOptions};
use crate::numerical::LSODE2::{Lsode2Error, Lsode2ProblemConfig, Lsode2Solver};
use crate::numerical::NonStiff_api::nonstiffODE;
#[cfg(test)]
use crate::numerical::Radau::Radau_main::{Radau, RadauSolverOptions, RadauStatistics};
#[cfg(not(test))]
use crate::numerical::Radau::{
    RadauConfig, RadauExecution, RadauFrontend, RadauJacobianSource, RadauMatrixLayout,
    RadauNativeSolver, RadauOutputPolicy, RadauProblem, RadauSolver, RadauTelemetryMode,
};
use crate::numerical::Radau::{RadauError, RadauSolution};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::IvpBackendError;
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, IvpBackendStatistics, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

type NativeResidualFn = Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>;
type NativeJacobianFn = Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>;

/// Explicit method names supported by the historical non-stiff adapter.
///
/// The `Other` variant is deliberately retained for downstream extensions,
/// but standard methods no longer rely on ad-hoc strings at the universal API
/// boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NonStiffMethod {
    Rk45,
    Dopri,
    Ab4,
    Other(String),
}

impl NonStiffMethod {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Rk45 => "RK45",
            Self::Dopri => "DOPRI",
            Self::Ab4 => "AB4",
            Self::Other(name) => name.as_str(),
        }
    }

    pub fn from_name(name: impl Into<String>) -> Self {
        let name = name.into();
        match name.to_ascii_uppercase().as_str() {
            "RK45" => Self::Rk45,
            "DOPRI" => Self::Dopri,
            "AB4" => Self::Ab4,
            _ => Self::Other(name),
        }
    }
}

/// Typed solver selection for the universal IVP facade.
#[derive(Clone, Debug)]
pub enum SolverType {
    NonStiff(NonStiffMethod),
    /// Fixed-order Radau IIA route backed by the production Radau API.
    ///
    /// The universal facade intentionally does not expose the historical
    /// order switch. Use `numerical::Radau::RadauSolver` directly when a
    /// solver-specific configuration is required.
    Radau,
    Bdf,
    BackwardEuler,
    Lsode2,
}

#[derive(Clone, Debug)]
pub enum SolverParam {
    Float(f64),
    Int(usize),
    Bool(bool),
    OptionalFloat(Option<f64>),
    OptionalInt(Option<usize>),
    OptionalMatrix(Option<DMatrix<f64>>),
}

#[derive(Clone, Debug, Default)]
pub struct UniversalIvpStatistics {
    pub method_label: String,
    pub backend_label: String,
    pub setup_ms_total: f64,
    pub solve_ms_total: f64,
    pub integration_loop_ms_total: f64,
    pub bdf_step_ms_total: f64,
    pub output_collection_ms_total: f64,
    pub result_assembly_ms_total: f64,
    pub linear_factorization_ms_total: f64,
    pub linear_solve_ms_total: f64,
    pub residual_calls: usize,
    pub residual_ms_total: f64,
    pub jacobian_calls: usize,
    pub jacobian_ms_total: f64,
    pub step_calls: usize,
    pub accepted_steps_total: usize,
    pub candidate_step_attempts_total: usize,
    pub rejected_step_attempts_total: usize,
    pub linear_solve_attempts_total: usize,
    pub nonlinear_solve_calls: usize,
    pub nonlinear_iterations_total: usize,
    /// BDF RHS calls, including initialization and finite-difference probes.
    pub bdf_nfev_total: usize,
    /// BDF Jacobian evaluations, including the initial Jacobian.
    pub bdf_njev_total: usize,
    /// BDF shifted-Jacobian factorization attempts, including failures.
    pub bdf_nlu_total: usize,
    /// Native Radau residual probes used to form finite-difference Jacobians.
    pub finite_difference_probes_total: usize,
}

impl UniversalIvpStatistics {
    pub fn avg_residual_ms(&self) -> Option<f64> {
        (self.residual_calls > 0).then(|| self.residual_ms_total / self.residual_calls as f64)
    }

    pub fn avg_jacobian_ms(&self) -> Option<f64> {
        (self.jacobian_calls > 0).then(|| self.jacobian_ms_total / self.jacobian_calls as f64)
    }

    pub fn avg_nonlinear_iterations(&self) -> Option<f64> {
        (self.nonlinear_solve_calls > 0)
            .then(|| self.nonlinear_iterations_total as f64 / self.nonlinear_solve_calls as f64)
    }

    pub fn table_report(&self) -> String {
        format!(
            "method={} backend={} setup_ms_total={:.3} solve_ms_total={:.3} integration_loop_ms={:.3}(nested) bdf_step_ms={:.3}(nested) output_collection_ms={:.3}(nested) result_assembly_ms={:.3}(nested) linear_factorization_ms={:.3}(nested) linear_solve_ms={:.3}(nested) steps={} accepted_steps={} candidate_steps={} rejected_step_attempts={} linear_solve_attempts={} res_calls={} res_ms_total={:.3} res_ms_avg={:.6} jac_calls={} jac_ms_total={:.3} jac_ms_avg={:.6} nonlinear_solves={} nonlinear_iters_total={} nonlinear_iters_avg={:.3} bdf_nfev={} bdf_njev={} bdf_nlu={} radau_fd_probes={}",
            self.method_label,
            self.backend_label,
            self.setup_ms_total,
            self.solve_ms_total,
            self.integration_loop_ms_total,
            self.bdf_step_ms_total,
            self.output_collection_ms_total,
            self.result_assembly_ms_total,
            self.linear_factorization_ms_total,
            self.linear_solve_ms_total,
            self.step_calls,
            self.accepted_steps_total,
            self.candidate_step_attempts_total,
            self.rejected_step_attempts_total,
            self.linear_solve_attempts_total,
            self.residual_calls,
            self.residual_ms_total,
            self.avg_residual_ms().unwrap_or(0.0),
            self.jacobian_calls,
            self.jacobian_ms_total,
            self.avg_jacobian_ms().unwrap_or(0.0),
            self.nonlinear_solve_calls,
            self.nonlinear_iterations_total,
            self.avg_nonlinear_iterations().unwrap_or(0.0),
            self.bdf_nfev_total,
            self.bdf_njev_total,
            self.bdf_nlu_total,
            self.finite_difference_probes_total,
        )
    }

    fn from_backend_stats(
        method_label: impl Into<String>,
        backend_label: impl Into<String>,
        stats: &IvpBackendStatistics,
    ) -> Self {
        Self {
            method_label: method_label.into(),
            backend_label: backend_label.into(),
            setup_ms_total: stats.backend_prepare_ms_total,
            solve_ms_total: stats.solve_ms_total,
            integration_loop_ms_total: stats.integration_loop_ms_total,
            bdf_step_ms_total: stats.bdf_step_ms_total,
            output_collection_ms_total: stats.output_collection_ms_total,
            result_assembly_ms_total: stats.result_assembly_ms_total,
            linear_factorization_ms_total: stats.linear_factorization_ms_total,
            linear_solve_ms_total: stats.linear_solve_ms_total,
            residual_calls: stats.residual_calls,
            residual_ms_total: stats.residual_ms_total,
            jacobian_calls: stats.jacobian_calls,
            jacobian_ms_total: stats.jacobian_ms_total,
            step_calls: stats.step_calls,
            accepted_steps_total: stats.accepted_steps_total,
            candidate_step_attempts_total: stats.candidate_step_attempts_total,
            rejected_step_attempts_total: stats.rejected_step_attempts_total,
            linear_solve_attempts_total: stats.linear_solve_attempts_total,
            nonlinear_solve_calls: stats.nonlinear_solve_calls,
            nonlinear_iterations_total: stats.nonlinear_iterations_total,
            bdf_nfev_total: stats.bdf_nfev_total,
            bdf_njev_total: stats.bdf_njev_total,
            bdf_nlu_total: stats.bdf_nlu_total,
            finite_difference_probes_total: 0,
        }
    }

    fn from_radau_solution(
        method_label: impl Into<String>,
        backend_label: impl Into<String>,
        solution: &RadauSolution,
    ) -> Self {
        let report = solution.telemetry();
        let counter = |key: &str| report.counters.get(key).copied().unwrap_or(0) as usize;
        let timing = |key: &str| report.timings_ms.get(key).copied().unwrap_or(0.0);
        let linear_solve_attempts = counter("real_solves") + counter("complex_solves");
        Self {
            method_label: method_label.into(),
            backend_label: backend_label.into(),
            setup_ms_total: timing("preparation_ms"),
            solve_ms_total: timing("callback_ms")
                + timing("newton_ms")
                + timing("linear_ms")
                + timing("step_control_ms")
                + timing("output_ms"),
            integration_loop_ms_total: 0.0,
            bdf_step_ms_total: 0.0,
            output_collection_ms_total: timing("output_ms"),
            result_assembly_ms_total: 0.0,
            linear_factorization_ms_total: timing("factorization_ms"),
            linear_solve_ms_total: timing("real_solve_ms") + timing("complex_solve_ms"),
            residual_calls: counter("residual_evaluations"),
            residual_ms_total: timing("residual_evaluation_ms"),
            jacobian_calls: counter("jacobian_evaluations"),
            jacobian_ms_total: timing("jacobian_evaluation_ms"),
            step_calls: solution.attempts,
            accepted_steps_total: solution.accepted_steps,
            candidate_step_attempts_total: solution.attempts,
            rejected_step_attempts_total: solution.rejected_steps,
            linear_solve_attempts_total: linear_solve_attempts,
            nonlinear_solve_calls: solution.attempts,
            nonlinear_iterations_total: counter("newton_iterations"),
            bdf_nfev_total: 0,
            bdf_njev_total: 0,
            bdf_nlu_total: counter("factorizations"),
            finite_difference_probes_total: counter("finite_difference_probes"),
        }
    }

    #[cfg(test)]
    fn from_radau_stats(
        method_label: impl Into<String>,
        backend_label: impl Into<String>,
        stats: &RadauStatistics,
    ) -> Self {
        Self {
            method_label: method_label.into(),
            backend_label: backend_label.into(),
            setup_ms_total: stats.backend_prepare_ms_total,
            solve_ms_total: stats.solve_ms_total,
            integration_loop_ms_total: 0.0,
            bdf_step_ms_total: 0.0,
            output_collection_ms_total: 0.0,
            // Radau has not yet instrumented these scopes.
            result_assembly_ms_total: 0.0,
            linear_factorization_ms_total: 0.0,
            linear_solve_ms_total: 0.0,
            residual_calls: stats.residual_calls,
            residual_ms_total: stats.residual_ms_total,
            jacobian_calls: stats.jacobian_calls,
            jacobian_ms_total: stats.jacobian_ms_total,
            step_calls: stats.step_calls,
            accepted_steps_total: 0,
            candidate_step_attempts_total: 0,
            rejected_step_attempts_total: 0,
            linear_solve_attempts_total: stats.linear_solves,
            nonlinear_solve_calls: stats.newton_solve_calls,
            nonlinear_iterations_total: stats.newton_iterations_total,
            bdf_nfev_total: 0,
            bdf_njev_total: 0,
            bdf_nlu_total: stats.lu_factorizations,
            finite_difference_probes_total: 0,
        }
    }
}

#[derive(Debug)]
pub enum UniversalOdeError {
    Backend(IvpBackendError),
    Bdf(BdfSolveError),
    BdfStopCondition(BdfStopConditionError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
    Radau(RadauError),
    LegacyRadauDisabled,
    UnsupportedGeneratedBackendForMethod { method: String },
}

impl std::fmt::Display for UniversalOdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(err) => write!(f, "{err}"),
            Self::Bdf(err) => write!(f, "{err}"),
            Self::BdfStopCondition(err) => write!(f, "{err}"),
            Self::BackwardEuler(err) => write!(f, "{err}"),
            Self::Lsode2(err) => write!(f, "{err}"),
            Self::Radau(err) => write!(f, "{err}"),
            Self::LegacyRadauDisabled => write!(
                f,
                "legacy Radau is disabled in ODE_api2; use numerical::Radau::RadauSolver"
            ),
            Self::UnsupportedGeneratedBackendForMethod { method } => {
                write!(
                    f,
                    "generated/AOT backend selection is not supported for method `{method}`"
                )
            }
        }
    }
}

impl std::error::Error for UniversalOdeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(err) => Some(err),
            Self::Bdf(err) => Some(err),
            Self::BdfStopCondition(err) => Some(err),
            Self::BackwardEuler(err) => Some(err),
            Self::Lsode2(err) => Some(err),
            Self::Radau(err) => Some(err),
            Self::LegacyRadauDisabled => None,
            Self::UnsupportedGeneratedBackendForMethod { .. } => None,
        }
    }
}

impl From<IvpBackendError> for UniversalOdeError {
    fn from(value: IvpBackendError) -> Self {
        Self::Backend(value)
    }
}

impl From<BdfStopConditionError> for UniversalOdeError {
    fn from(value: BdfStopConditionError) -> Self {
        Self::BdfStopCondition(value)
    }
}

impl From<BdfSolveError> for UniversalOdeError {
    fn from(value: BdfSolveError) -> Self {
        Self::Bdf(value)
    }
}

impl From<BeError> for UniversalOdeError {
    fn from(value: BeError) -> Self {
        Self::BackwardEuler(value)
    }
}

impl From<Lsode2Error> for UniversalOdeError {
    fn from(value: Lsode2Error) -> Self {
        Self::Lsode2(value)
    }
}

impl From<RadauError> for UniversalOdeError {
    fn from(value: RadauError) -> Self {
        Self::Radau(value)
    }
}

pub enum SolverInstance {
    NonStiff(nonstiffODE),
    #[cfg(test)]
    Radau(Radau),
    #[cfg(not(test))]
    Radau(RadauSolver),
    #[cfg(not(test))]
    RadauNative(RadauNativeSolver),
    BDF(BdfOdeSolver),
    BE(BE),
    LSODE2(Lsode2Solver),
}

pub struct UniversalODESolver {
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    solver_type: SolverType,
    solver_instance: Option<SolverInstance>,
    t_result: Option<DVector<f64>>,
    y_result: Option<DMatrix<f64>>,
    stop_condition: Option<HashMap<String, f64>>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
    bdf_telemetry_mode: BdfTelemetryMode,
    first_step: Option<f64>,
    vectorized: bool,
    jac_sparsity: Option<DMatrix<f64>>,
    neighborhood_check: Option<f64>,
    parallel: bool,
    generated_backend_config: Option<SymbolicIvpGeneratedBackendConfig>,
    lsode2_problem_config: Option<Lsode2ProblemConfig>,
    native_residual: Option<NativeResidualFn>,
    native_jacobian: Option<NativeJacobianFn>,
    radau_solution: Option<RadauSolution>,
    solver_params_legacy: HashMap<String, SolverParam>,
}

impl UniversalODESolver {
    /// Create a modern universal solver facade.
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Self {
        Self {
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            solver_type,
            solver_instance: None,
            t_result: None,
            y_result: None,
            stop_condition: None,
            step_size: None,
            tolerance: None,
            max_iterations: None,
            rtol: None,
            atol: None,
            max_step: None,
            bdf_telemetry_mode: BdfTelemetryMode::Off,
            first_step: None,
            vectorized: false,
            jac_sparsity: None,
            neighborhood_check: None,
            parallel: false,
            generated_backend_config: None,
            lsode2_problem_config: None,
            native_residual: None,
            native_jacobian: None,
            radau_solution: None,
            solver_params_legacy: HashMap::new(),
        }
    }

    fn method_label(&self) -> String {
        match &self.solver_type {
            SolverType::NonStiff(method) => method.as_str().to_string(),
            SolverType::Radau => "Radau".to_string(),
            SolverType::Bdf => "BDF".to_string(),
            SolverType::BackwardEuler => "BackwardEuler".to_string(),
            SolverType::Lsode2 => "LSODE2".to_string(),
        }
    }

    fn backend_label(&self) -> String {
        if let Some(config) = self.generated_backend_config.as_ref() {
            format!(
                "{:?}:{:?}:{:?}",
                config.build_policy, config.aot_codegen_backend, config.aot_c_compiler
            )
        } else {
            "Lambdify".to_string()
        }
    }

    /// Legacy constructor kept for old call-sites that still think in terms of
    /// the historical facade. Prefer [`Self::new`] for new code.
    pub fn new_legacy(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Self {
        Self::new(eq_system, values, arg, solver_type, t0, y0, t_bound)
    }

    pub fn set_step_size(&mut self, step: f64) {
        self.step_size = Some(step);
        self.solver_params_legacy
            .insert("step_size".to_string(), SolverParam::Float(step));
    }

    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = Some(tolerance);
        self.solver_params_legacy
            .insert("tolerance".to_string(), SolverParam::Float(tolerance));
    }

    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = Some(max_iter);
        self.solver_params_legacy
            .insert("max_iterations".to_string(), SolverParam::Int(max_iter));
    }

    pub fn set_rtol(&mut self, rtol: f64) {
        self.rtol = Some(rtol);
        self.solver_params_legacy
            .insert("rtol".to_string(), SolverParam::Float(rtol));
    }

    pub fn set_atol(&mut self, atol: f64) {
        self.atol = Some(atol);
        self.solver_params_legacy
            .insert("atol".to_string(), SolverParam::Float(atol));
    }

    pub fn set_max_step(&mut self, max_step: f64) {
        self.max_step = Some(max_step);
        self.solver_params_legacy
            .insert("max_step".to_string(), SolverParam::Float(max_step));
    }

    pub fn set_first_step(&mut self, first_step: Option<f64>) {
        self.first_step = first_step;
        self.solver_params_legacy.insert(
            "first_step".to_string(),
            SolverParam::OptionalFloat(first_step),
        );
    }

    pub fn set_vectorized(&mut self, vectorized: bool) {
        self.vectorized = vectorized;
        self.solver_params_legacy
            .insert("vectorized".to_string(), SolverParam::Bool(vectorized));
    }

    pub fn set_jac_sparsity(&mut self, jac_sparsity: Option<DMatrix<f64>>) {
        self.jac_sparsity = jac_sparsity.clone();
        self.solver_params_legacy.insert(
            "jac_sparsity".to_string(),
            SolverParam::OptionalMatrix(jac_sparsity),
        );
    }

    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
        self.solver_params_legacy
            .insert("parallel".to_string(), SolverParam::Bool(parallel));
    }

    pub fn set_stop_condition(&mut self, stop_condition: HashMap<String, f64>) {
        self.stop_condition = Some(stop_condition);
    }

    pub fn set_neighborhood_check(&mut self, tolerance: f64) {
        self.neighborhood_check = Some(tolerance);
        self.solver_params_legacy.insert(
            "neighborhood_check".to_string(),
            SolverParam::Float(tolerance),
        );
    }

    pub fn set_generated_backend_config(&mut self, config: SymbolicIvpGeneratedBackendConfig) {
        self.generated_backend_config = Some(config);
    }

    pub fn set_lsode2_problem_config(&mut self, config: Lsode2ProblemConfig) {
        self.lsode2_problem_config = Some(config);
    }

    /// Installs pure numerical callbacks `f(t, y)` and optional `df/dy`.
    ///
    /// For implicit methods:
    /// - if Jacobian is provided, it is used directly;
    /// - if Jacobian is absent, finite-difference Jacobians are used.
    pub fn set_native_ode_callbacks<F, J>(&mut self, residual: F, jacobian: Option<J>)
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        self.native_residual = Some(Arc::new(residual));
        self.native_jacobian = jacobian.map(|jac| Arc::new(jac) as NativeJacobianFn);
    }

    pub fn set_generated_backend_mode(&mut self, mode: DenseIvpGeneratedBackendMode) {
        let config = self
            .generated_backend_config
            .clone()
            .map(|current| {
                let mut next = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
                next.resolver = current.resolver.clone();
                next.aot_options = current.aot_options;
                next.aot_codegen_backend = current.aot_codegen_backend;
                next.aot_c_compiler = current.aot_c_compiler.clone();
                next.output_parent_dir = current.output_parent_dir.clone();
                next.crate_name_override = current.crate_name_override.clone();
                next.module_name_override = current.module_name_override.clone();
                next
            })
            .unwrap_or_else(|| SymbolicIvpGeneratedBackendConfig::from_mode(mode));
        self.set_generated_backend_config(config);
    }

    pub fn set_generated_backend_c_tcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        );
    }

    pub fn set_generated_backend_c_gcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        );
    }

    pub fn set_generated_backend_zig(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        );
    }

    pub fn set_generated_backend_for_repeated_solves(
        &mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .for_repeated_solves(),
        );
    }

    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.set_generated_backend_config(config);
        self
    }

    pub fn with_lsode2_problem_config(mut self, config: Lsode2ProblemConfig) -> Self {
        self.set_lsode2_problem_config(config);
        self
    }

    /// Builder-style alias for [`Self::set_native_ode_callbacks`].
    pub fn with_native_ode_callbacks<F, J>(mut self, residual: F, jacobian: Option<J>) -> Self
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        self.set_native_ode_callbacks(residual, jacobian);
        self
    }

    /// Selects telemetry for the BDF route. Telemetry remains disabled by default.
    pub fn with_bdf_telemetry_mode(mut self, mode: BdfTelemetryMode) -> Self {
        self.bdf_telemetry_mode = mode;
        self
    }

    pub fn with_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        self.set_generated_backend_mode(mode);
        self
    }

    pub fn with_generated_backend_c_tcc(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.set_generated_backend_c_tcc(output_parent_dir);
        self
    }

    pub fn with_generated_backend_c_gcc(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.set_generated_backend_c_gcc(output_parent_dir);
        self
    }

    pub fn with_generated_backend_zig(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.set_generated_backend_zig(output_parent_dir);
        self
    }

    pub fn with_generated_backend_for_repeated_solves(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_generated_backend_for_repeated_solves(output_parent_dir);
        self
    }

    pub fn set_parameter(&mut self, key: &str, value: SolverParam) {
        self.solver_params_legacy
            .insert(key.to_string(), value.clone());
        match (key, value) {
            ("step_size", SolverParam::Float(v)) => self.set_step_size(v),
            ("tolerance", SolverParam::Float(v)) => self.set_tolerance(v),
            ("max_iterations", SolverParam::Int(v)) => self.set_max_iterations(v),
            ("rtol", SolverParam::Float(v)) => self.set_rtol(v),
            ("atol", SolverParam::Float(v)) => self.set_atol(v),
            ("max_step", SolverParam::Float(v)) => self.set_max_step(v),
            ("first_step", SolverParam::OptionalFloat(v)) => self.set_first_step(v),
            ("first_step", SolverParam::Float(v)) => self.set_first_step(Some(v)),
            ("vectorized", SolverParam::Bool(v)) => self.set_vectorized(v),
            ("jac_sparsity", SolverParam::OptionalMatrix(v)) => self.set_jac_sparsity(v),
            ("parallel", SolverParam::Bool(v)) => self.set_parallel(v),
            ("neighborhood_check", SolverParam::Float(v)) => self.set_neighborhood_check(v),
            _ => {}
        }
    }

    pub fn set_parameters(&mut self, params: HashMap<String, SolverParam>) {
        for (key, value) in params {
            self.set_parameter(&key, value);
        }
    }

    /// Legacy string-key setter preserved for compatibility with older wrappers.
    pub fn set_parameter_legacy(&mut self, key: &str, value: SolverParam) {
        self.set_parameter(key, value);
    }

    /// Legacy bulk string-key setter preserved for compatibility with older wrappers.
    pub fn set_parameters_legacy(&mut self, params: HashMap<String, SolverParam>) {
        self.set_parameters(params);
    }

    pub fn initialize(&mut self) {
        self.try_initialize()
            .expect("universal ODE solver initialization should succeed");
    }

    /// Legacy initialize wrapper. Prefer [`Self::try_initialize`] or [`Self::initialize`].
    pub fn initialize_legacy(&mut self) {
        self.initialize();
    }

    pub fn try_initialize(&mut self) -> Result<(), UniversalOdeError> {
        self.solver_instance = Some(match &self.solver_type {
            SolverType::NonStiff(method) => {
                if self.generated_backend_config.is_some() {
                    return Err(UniversalOdeError::UnsupportedGeneratedBackendForMethod {
                        method: method.as_str().to_string(),
                    });
                }
                let mut solver = nonstiffODE::new(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    method.as_str().to_string(),
                    self.t0,
                    self.y0.clone(),
                    self.t_bound,
                    self.step_size.unwrap_or(1e-3),
                    self.stop_condition.clone(),
                );
                if let Some(tol) = self.neighborhood_check {
                    solver.set_neighborhood_check(tol);
                }
                SolverInstance::NonStiff(solver)
            }
            SolverType::Radau => {
                #[cfg(test)]
                {
                    let mut options = RadauSolverOptions::new(
                        crate::numerical::Radau::Radau_main::RadauOrder::Order5,
                        self.eq_system.clone(),
                        self.values.clone(),
                        self.arg.clone(),
                        self.tolerance.unwrap_or(1e-6),
                        self.max_iterations.unwrap_or(50),
                        self.step_size,
                        self.t0,
                        self.t_bound,
                        self.y0.clone(),
                    )
                    .with_parallel(self.parallel);
                    if let Some(config) = self.generated_backend_config.clone() {
                        options = options.with_generated_backend_config(config);
                    }
                    let mut solver = Radau::new_with_options(options);
                    if let Some(rhs) = self.native_residual.clone() {
                        let jac = self.native_jacobian.clone();
                        solver.set_native_ode_callbacks(
                            move |t: f64, y: &DVector<f64>| rhs(t, y),
                            jac.map(|jac_fun| move |t: f64, y: &DVector<f64>| jac_fun(t, y)),
                        );
                    }
                    if let Some(stop_condition) = self.stop_condition.clone() {
                        solver.set_stop_condition(stop_condition);
                    }
                    SolverInstance::Radau(solver)
                }
                #[cfg(not(test))]
                {
                    let mut config = RadauConfig::default();
                    config.t0 = self.t0;
                    config.t_bound = self.t_bound;
                    config.rtol = self.tolerance.unwrap_or(1e-6);
                    config.atol = self.tolerance.unwrap_or(1e-6);
                    config.max_step = self.step_size.unwrap_or(f64::INFINITY);
                    config.max_newton_iterations = self.max_iterations.unwrap_or(50);
                    config.execution = RadauExecution::Lambdify;
                    config.frontend = RadauFrontend::ExprLegacy;
                    config.matrix_layout = RadauMatrixLayout::Dense;
                    config.jacobian_source = RadauJacobianSource::Analytic;
                    config.telemetry = RadauTelemetryMode::Counters;
                    config.output = RadauOutputPolicy::FinalOnly;

                    if let Some(rhs) = self.native_residual.clone() {
                        let jacobian = self.native_jacobian.clone();
                        SolverInstance::RadauNative(RadauNativeSolver::from_shared_callbacks(
                            config, rhs, jacobian,
                        )?)
                    } else {
                        // The universal facade receives symbolic residuals, so it
                        // can prepare the analytic Jacobian without asking users
                        // to duplicate the model as a callback.  The row-major
                        // ordering matches `RadauProblem::with_jacobian`.
                        let jacobian = self
                            .eq_system
                            .iter()
                            .flat_map(|equation| {
                                self.values.iter().map(|variable| equation.diff(variable))
                            })
                            .collect();
                        let problem = RadauProblem::new(
                            self.eq_system.clone(),
                            self.values.clone(),
                            self.arg.clone(),
                        )
                        .with_jacobian(jacobian);
                        SolverInstance::Radau(RadauSolver::prepare(problem, config)?)
                    }
                }
            }
            SolverType::Bdf => {
                let mut options = BdfSolverOptions::for_bdf(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0,
                    self.y0.clone(),
                    self.t_bound,
                    self.max_step.unwrap_or(1e-3),
                    self.rtol.unwrap_or(1e-5),
                    self.atol.unwrap_or(1e-5),
                    self.jac_sparsity.clone(),
                    self.vectorized,
                    self.first_step,
                )
                .with_telemetry_mode(self.bdf_telemetry_mode);
                if let Some(config) = self.generated_backend_config.clone() {
                    options = options.with_generated_backend_config(config);
                }
                let mut solver = BdfOdeSolver::new_with_options(options);
                if let Some(rhs) = self.native_residual.clone() {
                    let jac = self.native_jacobian.clone();
                    solver.set_native_ode_callbacks(
                        move |t: f64, y: &DVector<f64>| rhs(t, y),
                        jac.map(|jac_fun| move |t: f64, y: &DVector<f64>| jac_fun(t, y)),
                    );
                }
                if let Some(stop_condition) = self.stop_condition.clone() {
                    solver.try_set_stop_condition(stop_condition)?;
                }
                SolverInstance::BDF(solver)
            }
            SolverType::BackwardEuler => {
                let mut options = BeSolverOptions::new(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.tolerance.unwrap_or(1e-6),
                    self.max_iterations.unwrap_or(50),
                    self.step_size,
                    self.t0,
                    self.t_bound,
                    self.y0.clone(),
                );
                if let Some(config) = self.generated_backend_config.clone() {
                    options = options.with_generated_backend_config(config);
                }
                let mut solver = BE::try_new_with_options(options)?;
                if let Some(rhs) = self.native_residual.clone() {
                    let jac = self.native_jacobian.clone();
                    solver.set_native_ode_callbacks(
                        move |t: f64, y: &DVector<f64>| rhs(t, y),
                        jac.map(|jac_fun| move |t: f64, y: &DVector<f64>| jac_fun(t, y)),
                    );
                }
                if let Some(stop_condition) = self.stop_condition.clone() {
                    solver.set_stop_condition(stop_condition);
                }
                if let Some(tol) = self.neighborhood_check {
                    solver.try_set_neighborhood_check(tol)?;
                }
                SolverInstance::BE(solver)
            }
            SolverType::Lsode2 => {
                let mut config = self.lsode2_problem_config.clone().unwrap_or_else(|| {
                    Lsode2ProblemConfig::new(
                        self.eq_system.clone(),
                        self.values.clone(),
                        self.arg.clone(),
                        self.t0,
                        self.y0.clone(),
                        self.t_bound,
                        self.max_step.unwrap_or(1e-3),
                        self.rtol.unwrap_or(1e-5),
                        self.atol.unwrap_or(1e-5),
                    )
                });

                config = config
                    .with_first_step(self.first_step)
                    .with_jac_sparsity(self.jac_sparsity.clone())
                    .with_vectorized(self.vectorized);

                if let Some(generated) = self.generated_backend_config.clone() {
                    let backend = config.backend.clone().with_generated_backend(generated);
                    config = config.with_backend(backend);
                }

                SolverInstance::LSODE2(Lsode2Solver::new(config)?)
            }
        });
        Ok(())
    }

    pub fn try_solve(&mut self) -> Result<(), UniversalOdeError> {
        if self.solver_instance.is_none() {
            self.try_initialize()?;
        }

        match self
            .solver_instance
            .as_mut()
            .expect("solver instance should be initialized")
        {
            SolverInstance::NonStiff(solver) => {
                solver.solve();
                let (t_result, y_result) = solver.get_result();
                self.t_result = Some(t_result);
                self.y_result = Some(y_result);
            }
            #[cfg(test)]
            SolverInstance::Radau(solver) => {
                solver.try_solve()?;
                let (t_result, y_result) = solver.get_result();
                self.t_result = t_result;
                self.y_result = y_result;
            }
            #[cfg(not(test))]
            SolverInstance::Radau(solver) => {
                let solution = solver.solve(self.y0.as_slice())?;
                let state = DVector::from_vec(solution.y.clone());
                self.t_result = Some(DVector::from_vec(vec![solution.t]));
                self.y_result = Some(DMatrix::from_row_slice(1, state.len(), state.as_slice()));
                self.radau_solution = Some(solution);
            }
            #[cfg(not(test))]
            SolverInstance::RadauNative(solver) => {
                let solution = solver.solve(self.y0.as_slice())?;
                let state = DVector::from_vec(solution.y.clone());
                self.t_result = Some(DVector::from_vec(vec![solution.t]));
                self.y_result = Some(DMatrix::from_row_slice(1, state.len(), state.as_slice()));
                self.radau_solution = Some(solution);
            }
            SolverInstance::BDF(solver) => {
                solver.try_solve()?;
            }
            SolverInstance::BE(solver) => {
                solver.try_solve()?;
                let (t_result, y_result) = solver.get_result();
                self.t_result = t_result;
                self.y_result = y_result;
            }
            SolverInstance::LSODE2(solver) => {
                solver.solve()?;
                let (t_result, y_result) = solver.get_result();
                self.t_result = Some(t_result);
                self.y_result = Some(y_result);
            }
        }
        Ok(())
    }

    pub fn solve(&mut self) {
        self.try_solve()
            .expect("universal ODE solver execution should succeed");
    }

    /// Legacy solve wrapper mirroring the historical facade.
    pub fn solve_legacy(&mut self) {
        self.solve();
    }

    pub fn get_result(&self) -> (Option<DVector<f64>>, Option<DMatrix<f64>>) {
        self.get_result_ref()
            .map_or((None, None), |(t, y)| (Some(t.clone()), Some(y.clone())))
    }

    /// Borrows the latest trajectory without cloning its time/state buffers.
    pub fn get_result_ref(&self) -> Option<(&DVector<f64>, &DMatrix<f64>)> {
        match self.solver_instance.as_ref() {
            Some(SolverInstance::BDF(solver)) => Some(solver.get_result_ref()),
            _ => Some((self.t_result.as_ref()?, self.y_result.as_ref()?)),
        }
    }

    pub fn get_status(&self) -> Option<String> {
        match self.solver_instance.as_ref()? {
            SolverInstance::NonStiff(solver) => Some(solver.get_status().clone()),
            #[cfg(test)]
            SolverInstance::Radau(solver) => Some(solver.get_status().clone()),
            #[cfg(not(test))]
            SolverInstance::Radau(_) => Some(if self.radau_solution.is_some() {
                "finished".to_string()
            } else {
                "initialized".to_string()
            }),
            #[cfg(not(test))]
            SolverInstance::RadauNative(_) => Some(if self.radau_solution.is_some() {
                "finished".to_string()
            } else {
                "initialized".to_string()
            }),
            SolverInstance::BDF(solver) => Some(solver.get_status().to_string()),
            SolverInstance::BE(solver) => Some(solver.get_status()),
            SolverInstance::LSODE2(solver) => Some(solver.status().to_string()),
        }
    }

    pub fn get_statistics(&self) -> Option<UniversalIvpStatistics> {
        let method = self.method_label();
        let backend = self.backend_label();
        match self.solver_instance.as_ref()? {
            SolverInstance::NonStiff(_) => None,
            #[cfg(test)]
            SolverInstance::Radau(solver) => Some(UniversalIvpStatistics::from_radau_stats(
                method,
                backend,
                &solver.get_statistics(),
            )),
            #[cfg(not(test))]
            SolverInstance::Radau(_) => self.radau_solution.as_ref().map(|solution| {
                UniversalIvpStatistics::from_radau_solution(method, backend, solution)
            }),
            #[cfg(not(test))]
            SolverInstance::RadauNative(_) => self.radau_solution.as_ref().map(|solution| {
                UniversalIvpStatistics::from_radau_solution(
                    method,
                    "NativeCallbacks".to_string(),
                    solution,
                )
            }),
            SolverInstance::BDF(solver) => Some(UniversalIvpStatistics::from_backend_stats(
                method,
                backend,
                &solver.get_statistics(),
            )),
            SolverInstance::BE(solver) => Some(UniversalIvpStatistics::from_backend_stats(
                method,
                backend,
                &solver.get_statistics(),
            )),
            SolverInstance::LSODE2(solver) => {
                let summary = solver.summary();
                Some(UniversalIvpStatistics::from_backend_stats(
                    method,
                    format!(
                        "{}:{}:{}",
                        summary.resolved_source,
                        summary.resolved_structure,
                        summary.linear_solver_backend
                    ),
                    &summary.statistics,
                ))
            }
        }
    }

    pub fn statistics_report(&self) -> Option<String> {
        self.get_statistics().map(|stats| stats.table_report())
    }

    pub fn plot_result(&self) {
        if let Some((t_result, y_result)) = self.get_result_ref() {
            plots_ref(&self.arg, &self.values, t_result, y_result);
        }
    }

    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self.solver_instance.as_ref() {
            Some(SolverInstance::NonStiff(solver)) => solver.save_result(),
            Some(SolverInstance::BDF(solver)) => solver.save_result(),
            Some(SolverInstance::LSODE2(_)) => Ok(()),
            _ => Ok(()),
        }
    }
}

impl UniversalODESolver {
    pub fn rk45(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff(NonStiffMethod::Rk45),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    pub fn dopri(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff(NonStiffMethod::Dopri),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    pub fn ab4(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff(NonStiffMethod::Ab4),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    /// Construct the universal fixed-order Radau route.
    ///
    /// The historical `RadauOrder` argument was intentionally removed from
    /// this facade. The production universal route uses the maintained Radau
    /// configuration; solver-specific policies remain available through
    /// [`crate::numerical::Radau::RadauSolver`].
    pub fn radau(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: Option<f64>,
    ) -> Self {
        let mut solver = Self::new(eq_system, values, arg, SolverType::Radau, t0, y0, t_bound);
        solver.set_tolerance(tolerance);
        solver.set_max_iterations(max_iterations);
        if let Some(step) = step_size {
            solver.set_step_size(step);
        }
        solver
    }

    pub fn bdf(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
    ) -> Self {
        let mut solver = Self::new(eq_system, values, arg, SolverType::Bdf, t0, y0, t_bound);
        solver.set_max_step(max_step);
        solver.set_rtol(rtol);
        solver.set_atol(atol);
        solver
    }

    pub fn backward_euler(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: Option<f64>,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::BackwardEuler,
            t0,
            y0,
            t_bound,
        );
        solver.set_tolerance(tolerance);
        solver.set_max_iterations(max_iterations);
        if let Some(step) = step_size {
            solver.set_step_size(step);
        }
        solver
    }

    pub fn lsode2(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
    ) -> Self {
        let mut solver = Self::new(eq_system, values, arg, SolverType::Lsode2, t0, y0, t_bound);
        solver.set_max_step(max_step);
        solver.set_rtol(rtol);
        solver.set_atol(atol);
        solver
    }

    /// Build the universal facade directly from a prepared LSODE2 problem config.
    ///
    /// This is the most ergonomic entrypoint when the caller already assembled
    /// LSODE2 backend/controller/native options and still wants to execute
    /// through the shared `UniversalODESolver` API surface.
    pub fn lsode2_with_problem_config(config: Lsode2ProblemConfig) -> Self {
        let mut solver = Self::lsode2(
            config.eq_system.clone(),
            config.values.clone(),
            config.arg.clone(),
            config.t0,
            config.y0.clone(),
            config.t_bound,
            config.max_step,
            config.rtol,
            config.atol,
        );
        solver.set_first_step(config.first_step);
        solver.set_vectorized(config.vectorized);
        solver.set_jac_sparsity(config.jac_sparsity.clone());
        solver.set_lsode2_problem_config(config);
        solver
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_method_names_are_typed_and_round_trip() {
        assert_eq!(NonStiffMethod::from_name("rk45"), NonStiffMethod::Rk45);
        assert_eq!(NonStiffMethod::from_name("DOPRI"), NonStiffMethod::Dopri);
        assert_eq!(NonStiffMethod::from_name("ab4"), NonStiffMethod::Ab4);
        assert_eq!(NonStiffMethod::Rk45.as_str(), "RK45");
        assert_eq!(NonStiffMethod::Dopri.as_str(), "DOPRI");
        assert_eq!(NonStiffMethod::Ab4.as_str(), "AB4");

        let extension = NonStiffMethod::from_name("custom-explicit");
        assert_eq!(extension.as_str(), "custom-explicit");
        match SolverType::NonStiff(extension) {
            SolverType::NonStiff(NonStiffMethod::Other(name)) => {
                assert_eq!(name, "custom-explicit")
            }
            other => panic!("unexpected solver selection: {other:?}"),
        }
    }

    fn simple_decay_problem() -> (Vec<Expr>, Vec<String>, String, DVector<f64>) {
        let x = Expr::Var("x".to_string());
        let eq_system = vec![-x.clone()];
        let values = vec!["x".to_string()];
        let arg = "t".to_string();
        let y0 = DVector::from_vec(vec![1.0]);
        (eq_system, values, arg, y0)
    }

    fn decay_rhs(_t: f64, y: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![-y[0]])
    }

    fn decay_jac(_t: f64, _y: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, 1, &[-1.0])
    }

    fn stiff_diagonal_problem() -> (Vec<Expr>, Vec<String>, String, DVector<f64>, f64) {
        let x0 = Expr::Var("x0".to_string());
        let x1 = Expr::Var("x1".to_string());
        let eq_system = vec![Expr::Const(-1000.0) * x0.clone(), -x1.clone()];
        let values = vec!["x0".to_string(), "x1".to_string()];
        let arg = "t".to_string();
        let y0 = DVector::from_vec(vec![1.0, 1.0]);
        let t_bound = 0.02;
        (eq_system, values, arg, y0, t_bound)
    }

    fn stiff_diagonal_rhs(_t: f64, y: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![-1000.0 * y[0], -y[1]])
    }

    fn stiff_diagonal_jac(_t: f64, _y: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[-1000.0, 0.0, 0.0, -1.0])
    }

    fn stiff_coupled_problem() -> (Vec<Expr>, Vec<String>, String, DVector<f64>, f64) {
        let x0 = Expr::Var("x0".to_string());
        let x1 = Expr::Var("x1".to_string());
        // x0' = -1000*x0 + 999*x1, x1' = -x1
        let eq_system = vec![
            Expr::Const(-1000.0) * x0.clone() + Expr::Const(999.0) * x1.clone(),
            -x1.clone(),
        ];
        let values = vec!["x0".to_string(), "x1".to_string()];
        let arg = "t".to_string();
        let y0 = DVector::from_vec(vec![0.0, 1.0]);
        let t_bound = 0.02;
        (eq_system, values, arg, y0, t_bound)
    }

    fn stiff_coupled_rhs(_t: f64, y: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![-1000.0 * y[0] + 999.0 * y[1], -y[1]])
    }

    fn stiff_coupled_jac(_t: f64, _y: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[-1000.0, 999.0, 0.0, -1.0])
    }

    fn robertson_problem() -> (Vec<Expr>, Vec<String>, String, DVector<f64>, f64) {
        let eq_system = vec![
            Expr::parse_expression("-0.04*x + 10000.0*y*z"),
            Expr::parse_expression("0.04*x - 10000.0*y*z - 30000000.0*y*y"),
            Expr::parse_expression("30000000.0*y*y"),
        ];
        let values = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let arg = "t".to_string();
        let y0 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let t_bound = 0.2;
        (eq_system, values, arg, y0, t_bound)
    }

    fn robertson_rhs(_t: f64, y: &DVector<f64>) -> DVector<f64> {
        let x = y[0];
        let yv = y[1];
        let z = y[2];
        DVector::from_vec(vec![
            -0.04 * x + 10000.0 * yv * z,
            0.04 * x - 10000.0 * yv * z - 30000000.0 * yv * yv,
            30000000.0 * yv * yv,
        ])
    }

    fn robertson_jac(_t: f64, y: &DVector<f64>) -> DMatrix<f64> {
        let yv = y[1];
        let z = y[2];
        DMatrix::from_row_slice(
            3,
            3,
            &[
                -0.04,
                10000.0 * z,
                10000.0 * yv,
                0.04,
                -10000.0 * z - 60000000.0 * yv,
                -10000.0 * yv,
                0.0,
                60000000.0 * yv,
                0.0,
            ],
        )
    }

    fn assert_robertson_invariants(y: &DMatrix<f64>) {
        let x_final = y[(y.nrows() - 1, 0)];
        let y_final = y[(y.nrows() - 1, 1)];
        let z_final = y[(y.nrows() - 1, 2)];
        let sum = x_final + y_final + z_final;
        assert!(
            (sum - 1.0).abs() < 5e-6,
            "Robertson mass drift too high: x+y+z={sum}"
        );
        assert!(x_final >= -1e-10, "x became negative: {x_final}");
        assert!(y_final >= -1e-10, "y became negative: {y_final}");
        assert!(z_final >= -1e-10, "z became negative: {z_final}");
    }

    fn assert_finished_status(solver: &UniversalODESolver) {
        let status = solver
            .get_status()
            .expect("solver status should be available after solve");
        assert!(
            status.starts_with("finished"),
            "unexpected solver status: {status}"
        );
    }

    fn assert_basic_runtime_stats(
        solver: &UniversalODESolver,
        require_lu_activity: bool,
        require_jacobian_activity: bool,
    ) {
        let stats = solver
            .get_statistics()
            .expect("statistics should be available after solve");
        assert!(stats.step_calls > 0, "expected positive step_calls");
        assert!(stats.residual_calls > 0, "expected positive residual_calls");
        if require_jacobian_activity {
            let jacobian_activity = stats.jacobian_calls > 0 || stats.bdf_njev_total > 0;
            assert!(
                jacobian_activity,
                "expected Jacobian activity through callbacks or BDF njev counter"
            );
        }
        if require_lu_activity {
            assert!(
                stats.bdf_nlu_total > 0,
                "expected positive LU factorization count for stiff run"
            );
        }
    }

    #[test]
    fn universal_ode_api_legacy_nonstiff_smoke() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver = UniversalODESolver::new_legacy(
            eq_system,
            values,
            arg,
            SolverType::NonStiff(NonStiffMethod::Rk45),
            0.0,
            y0,
            0.1,
        );
        solver.set_parameter_legacy("step_size", SolverParam::Float(1e-3));
        solver.initialize_legacy();
        solver.solve_legacy();
        let (t, y) = solver.get_result();
        assert!(t.is_some());
        assert!(y.is_some());
        let status = solver.get_status().unwrap_or_default();
        assert!(
            status == "finished" || status == "finished_native_faithful",
            "unexpected LSODE2 status: {status}"
        );
    }

    #[test]
    fn universal_ode_api_be_exposes_statistics() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            0.0,
            y0,
            0.1,
            1e-8,
            20,
            Some(1e-2),
        );
        solver.solve();
        let stats = solver
            .get_statistics()
            .expect("BE should expose normalized statistics");
        assert_eq!(stats.method_label, "BackwardEuler");
        assert!(stats.step_calls > 0);
    }

    #[test]
    fn universal_ode_api_lsode2_exposes_statistics() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let config = crate::numerical::LSODE2::Lsode2ProblemConfig::new(
            eq_system.clone(),
            values.clone(),
            arg.clone(),
            0.0,
            y0.clone(),
            0.1,
            1e-2,
            1e-6,
            1e-8,
        )
        .with_linear_system_structure(crate::numerical::LSODE2::Lsode2LinearSystemStructure::Dense)
        .with_linear_solver_policy(crate::numerical::LSODE2::Lsode2LinearSolverPolicy::Auto);

        let mut solver =
            UniversalODESolver::lsode2(eq_system, values, arg, 0.0, y0, 0.1, 1e-2, 1e-6, 1e-8)
                .with_lsode2_problem_config(config);
        solver.solve();

        let stats = solver
            .get_statistics()
            .expect("LSODE2 should expose normalized statistics");
        assert_eq!(stats.method_label, "LSODE2");
        assert!(stats.backend_label.contains("dense"));
        let status = solver
            .get_status()
            .expect("LSODE2 status should be available after solve");
        assert!(
            status.starts_with("finished"),
            "unexpected LSODE2 status: {status}"
        );
        assert!(
            stats.step_calls > 0 || stats.residual_calls > 0 || stats.solve_ms_total > 0.0,
            "expected non-zero LSODE2 solve activity counters"
        );
    }

    #[test]
    fn universal_ode_api_lsode2_with_problem_config_solves() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let config = crate::numerical::LSODE2::Lsode2ProblemConfig::new(
            eq_system, values, arg, 0.0, y0, 0.1, 0.05, 1e-6, 1e-8,
        )
        .with_linear_system_structure(crate::numerical::LSODE2::Lsode2LinearSystemStructure::Dense)
        .with_linear_solver_policy(crate::numerical::LSODE2::Lsode2LinearSolverPolicy::Auto);

        let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
        solver.solve();
        let status = solver.get_status().unwrap_or_default();
        assert!(
            status == "finished" || status == "finished_native_faithful",
            "unexpected LSODE2 status: {status}"
        );
    }

    #[test]
    fn universal_ode_api_rejects_generated_backend_for_nonstiff() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver = UniversalODESolver::rk45(eq_system, values, arg, 0.0, y0, 0.1, 1e-3)
            .with_generated_backend_c_tcc("target/test-artifacts/ode-api2");
        let err = solver
            .try_initialize()
            .expect_err("nonstiff facade should reject generated backend selection");
        assert!(matches!(
            err,
            UniversalOdeError::UnsupportedGeneratedBackendForMethod { .. }
        ));
    }

    #[test]
    fn universal_ode_api_bdf_native_callbacks_with_jacobian() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, 0.0, y0, 0.1, 1e-2, 1e-6, 1e-8)
                .with_bdf_telemetry_mode(BdfTelemetryMode::Counters)
                .with_native_ode_callbacks(decay_rhs, Some(decay_jac));
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, true, true);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y_final = y[(y.nrows() - 1, 0)];
        let expected = (-0.1f64).exp();
        assert!((y_final - expected).abs() < 5e-3);
    }

    #[test]
    fn universal_bdf_result_access_borrows_solver_storage() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, 0.0, y0, 0.1, 1e-2, 1e-6, 1e-8)
                .with_bdf_telemetry_mode(BdfTelemetryMode::Counters)
                .with_native_ode_callbacks(
                    decay_rhs,
                    Some(decay_jac as fn(f64, &DVector<f64>) -> DMatrix<f64>),
                );
        solver.solve();

        assert!(solver.t_result.is_none());
        assert!(solver.y_result.is_none());
        let report = solver.statistics_report().expect("BDF telemetry report");
        assert!(report.contains("accepted_steps="));
        assert!(report.contains("candidate_steps="));
        assert!(report.contains("linear_solve_attempts="));
        assert!(report.contains("bdf_step_ms="));
        assert!(report.contains("output_collection_ms="));
        let (t, y) = solver.get_result_ref().expect("borrowed BDF trajectory");
        assert!(!t.is_empty());
        assert_eq!(y.nrows(), t.len());
        let (owned_t, owned_y) = solver.get_result();
        assert_eq!(owned_t.as_ref(), Some(t));
        assert_eq!(owned_y.as_ref(), Some(y));
    }

    #[test]
    fn universal_ode_api_bdf_native_callbacks_fd_jacobian() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, 0.0, y0, 0.1, 1e-2, 1e-6, 1e-8)
                .with_bdf_telemetry_mode(BdfTelemetryMode::Counters)
                .with_native_ode_callbacks(
                    decay_rhs,
                    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
                );
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, true, false);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y_final = y[(y.nrows() - 1, 0)];
        let expected = (-0.1f64).exp();
        assert!((y_final - expected).abs() < 1e-2);
    }

    #[test]
    fn universal_ode_api_be_native_callbacks_fd_jacobian() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            0.0,
            y0,
            0.1,
            1e-8,
            20,
            Some(1e-2),
        )
        .with_native_ode_callbacks(
            decay_rhs,
            Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
        );
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, false);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y_final = y[(y.nrows() - 1, 0)];
        let expected = (-0.1f64).exp();
        assert!((y_final - expected).abs() < 1e-2);
    }

    #[test]
    fn universal_ode_api_radau_native_callbacks_with_jacobian() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, SolverType::Radau, 0.0, y0, 0.1)
                .with_native_ode_callbacks(decay_rhs, Some(decay_jac));
        solver.set_step_size(1e-2);
        solver.set_tolerance(1e-8);
        solver.set_max_iterations(20);
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, true);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y_final = y[(y.nrows() - 1, 0)];
        let expected = (-0.1f64).exp();
        assert!((y_final - expected).abs() < 1e-4);
    }

    #[test]
    fn universal_ode_api_bdf_native_stiff_diagonal_with_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = stiff_diagonal_problem();
        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, 0.0, y0, t_bound, 1e-5, 1e-10, 1e-12)
                .with_bdf_telemetry_mode(BdfTelemetryMode::Counters)
                .with_native_ode_callbacks(stiff_diagonal_rhs, Some(stiff_diagonal_jac));
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, true, true);
        let (t, y) = solver.get_result();
        let t = t.expect("time grid");
        let y = y.expect("solution matrix");
        let t_final = t[t.len() - 1];
        let y0_final = y[(y.nrows() - 1, 0)];
        let y1_final = y[(y.nrows() - 1, 1)];
        let expected0 = (-1000.0 * t_final).exp();
        let expected1 = (-t_final).exp();
        assert!((y0_final - expected0).abs() < 5e-5);
        assert!((y1_final - expected1).abs() < 5e-5);
    }

    #[test]
    fn universal_ode_api_bdf_native_stiff_diagonal_fd_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = stiff_diagonal_problem();
        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, 0.0, y0, t_bound, 1e-5, 1e-10, 1e-12)
                .with_bdf_telemetry_mode(BdfTelemetryMode::Counters)
                .with_native_ode_callbacks(
                    stiff_diagonal_rhs,
                    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
                );
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, true, false);
        let (t, y) = solver.get_result();
        let t = t.expect("time grid");
        let y = y.expect("solution matrix");
        let t_final = t[t.len() - 1];
        let y0_final = y[(y.nrows() - 1, 0)];
        let y1_final = y[(y.nrows() - 1, 1)];
        let expected0 = (-1000.0 * t_final).exp();
        let expected1 = (-t_final).exp();
        assert!((y0_final - expected0).abs() < 2e-4);
        assert!((y1_final - expected1).abs() < 2e-4);
    }

    #[test]
    fn universal_ode_api_be_native_stiff_coupled_with_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = stiff_coupled_problem();
        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            0.0,
            y0,
            t_bound,
            1e-10,
            40,
            Some(1e-4),
        )
        .with_native_ode_callbacks(stiff_coupled_rhs, Some(stiff_coupled_jac));
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, true);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y0_final = y[(y.nrows() - 1, 0)];
        let y1_final = y[(y.nrows() - 1, 1)];
        let expected0 = (-t_bound).exp() - (-1000.0 * t_bound).exp();
        let expected1 = (-t_bound).exp();
        assert!((y0_final - expected0).abs() < 6e-3);
        assert!((y1_final - expected1).abs() < 6e-3);
    }

    #[test]
    fn universal_ode_api_be_native_stiff_coupled_fd_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = stiff_coupled_problem();
        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            0.0,
            y0,
            t_bound,
            1e-10,
            40,
            Some(1e-4),
        )
        .with_native_ode_callbacks(
            stiff_coupled_rhs,
            Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
        );
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, false);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y0_final = y[(y.nrows() - 1, 0)];
        let y1_final = y[(y.nrows() - 1, 1)];
        let expected0 = (-t_bound).exp() - (-1000.0 * t_bound).exp();
        let expected1 = (-t_bound).exp();
        assert!((y0_final - expected0).abs() < 8e-3);
        assert!((y1_final - expected1).abs() < 8e-3);
    }

    #[test]
    fn universal_ode_api_radau_native_stiff_diagonal_with_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = stiff_diagonal_problem();
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, SolverType::Radau, 0.0, y0, t_bound)
                .with_native_ode_callbacks(stiff_diagonal_rhs, Some(stiff_diagonal_jac));
        solver.set_step_size(1e-4);
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(40);
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, true);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y0_final = y[(y.nrows() - 1, 0)];
        let y1_final = y[(y.nrows() - 1, 1)];
        let expected0 = (-1000.0 * t_bound).exp();
        let expected1 = (-t_bound).exp();
        assert!((y0_final - expected0).abs() < 5e-6);
        assert!((y1_final - expected1).abs() < 5e-6);
    }

    #[test]
    fn universal_ode_api_radau_native_stiff_diagonal_fd_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = stiff_diagonal_problem();
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, SolverType::Radau, 0.0, y0, t_bound)
                .with_native_ode_callbacks(
                    stiff_diagonal_rhs,
                    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
                );
        solver.set_step_size(1e-4);
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(40);
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, false);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        let y0_final = y[(y.nrows() - 1, 0)];
        let y1_final = y[(y.nrows() - 1, 1)];
        let expected0 = (-1000.0 * t_bound).exp();
        let expected1 = (-t_bound).exp();
        assert!((y0_final - expected0).abs() < 5e-5);
        assert!((y1_final - expected1).abs() < 5e-5);
    }

    #[test]
    fn universal_ode_api_bdf_native_robertson_with_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = robertson_problem();
        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, 0.0, y0, t_bound, 5e-4, 1e-9, 1e-12)
                .with_bdf_telemetry_mode(BdfTelemetryMode::Counters)
                .with_native_ode_callbacks(robertson_rhs, Some(robertson_jac));
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, true, true);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        assert_robertson_invariants(&y);
    }

    #[test]
    fn universal_ode_api_bdf_native_robertson_fd_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = robertson_problem();
        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, 0.0, y0, t_bound, 5e-4, 1e-9, 1e-12)
                .with_bdf_telemetry_mode(BdfTelemetryMode::Counters)
                .with_native_ode_callbacks(
                    robertson_rhs,
                    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
                );
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, true, false);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        assert_robertson_invariants(&y);
    }

    #[test]
    fn universal_ode_api_be_native_robertson_with_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = robertson_problem();
        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            0.0,
            y0,
            t_bound,
            1e-10,
            80,
            Some(2e-4),
        )
        .with_native_ode_callbacks(robertson_rhs, Some(robertson_jac));
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, true);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        assert_robertson_invariants(&y);
    }

    #[test]
    fn universal_ode_api_be_native_robertson_fd_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = robertson_problem();
        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            0.0,
            y0,
            t_bound,
            1e-10,
            80,
            Some(2e-4),
        )
        .with_native_ode_callbacks(
            robertson_rhs,
            Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
        );
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, false);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        assert_robertson_invariants(&y);
    }

    #[test]
    fn universal_ode_api_radau_native_robertson_with_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = robertson_problem();
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, SolverType::Radau, 0.0, y0, t_bound)
                .with_native_ode_callbacks(robertson_rhs, Some(robertson_jac));
        solver.set_step_size(2e-4);
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(80);
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, true);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        assert_robertson_invariants(&y);
    }

    #[test]
    fn universal_ode_api_radau_native_robertson_fd_jacobian_small_step() {
        let (eq_system, values, arg, y0, t_bound) = robertson_problem();
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, SolverType::Radau, 0.0, y0, t_bound)
                .with_native_ode_callbacks(
                    robertson_rhs,
                    Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
                );
        solver.set_step_size(2e-4);
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(80);
        solver.solve();
        assert_finished_status(&solver);
        assert_basic_runtime_stats(&solver, false, false);
        let (_, y) = solver.get_result();
        let y = y.expect("solution matrix");
        assert_robertson_invariants(&y);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Tests
/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests2 {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    #[test]
    fn test_universal_rk45() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let step_size = 1e-4;

        let mut solver =
            UniversalODESolver::rk45(eq_system, values, arg, t0, y0, t_bound, step_size);

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());

        let y_final = y_result.clone().unwrap()[(y_result.as_ref().unwrap().nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        assert_relative_eq!(y_final, expected, epsilon = 1e-2);
    }
    #[test]
    fn test_rk45_exponential() {
        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0 y'(0) = 1
        // solution y(x) = 1/2 (e^x - e^(-x))

        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;

        let y0 = DVector::from_vec(vec![0.0, 1.0]);
        let t_bound = 0.5;
        let step = 1e-3;
        let solver_type = SolverType::NonStiff(NonStiffMethod::Rk45);
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = 0.5 * (x.exp() - (-x).exp());
            assert_relative_eq!(y, expected, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_ab4_cos() {
        // y'' + y = 0,
        // y0' =y1,
        // y1' =- y0
        // y(0) = 1, y'(0) = 0 (solution: y = cos(x))
        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        // cos(0)=1, -sin(0)=0
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI;
        let step = 1e-5;
        let solver_type = SolverType::NonStiff(NonStiffMethod::Ab4);
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        let y1: DVector<f64> = y_final.column(1).into();
        // println!("{:?} \n {:?}", y0, y1);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x.cos();
            let expected1 = -x.sin();
            assert_relative_eq!(y, expected, epsilon = 1e-4);
            assert_relative_eq!(y1[i], expected1, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_rk45_cos2() {
        // y'' + y = 0,
        // y0' =y1,
        // y1' =- y0
        // y(0) = 1, y'(0) = 0 (solution: y = cos(x))
        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        // cos(0)=1, -sin(0)=0
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI;
        let step = 1e-5;
        let solver_type = SolverType::NonStiff(NonStiffMethod::Rk45);
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        let y1: DVector<f64> = y_final.column(1).into();
        // println!("{:?} \n {:?}", y0, y1);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x.cos();
            let expected1 = -x.sin();
            assert_relative_eq!(y, expected, epsilon = 1e-4);
            assert_relative_eq!(y1[i], expected1, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_universal_radau() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::radau(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(1e-3),
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }

    #[test]
    fn test_universal_bdf() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }
    #[test]
    fn test_bdf_exponential() {
        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0 y' = 1
        // solution y(x) = 1/2 (e^x - e^(-x))

        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;

        let y0 = DVector::from_vec(vec![0.0, 1.0]);
        let t_bound = 0.5;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = 0.5 * (x.exp() - (-x).exp());
            assert_relative_eq!(y, expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_bdf_linear() {
        // y'' = 0, y(0) = 0, y(1) = 1 (solution: y = x)
        // y0' = y1
        // y1' = 0
        //
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let y0 = DVector::from_vec(vec![0.0, 0.9997384083916304]);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut solver =
            UniversalODESolver::bdf(eq_vec, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x;
            assert_relative_eq!(y, expected, epsilon = 1e-3);
        }
    }
    #[test]
    fn test_universal_backward_euler() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(1e-3),
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }
    #[test]
    fn test_direct_setting() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, SolverType::Radau, t0, y0, t_bound);
        solver.set_max_iterations(100);
        solver.set_tolerance(1e-6);
        solver.set_step_size(1e-3);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }

    #[test]
    fn test_universal_stop_condition_rk45() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0; // Large bound to ensure stop condition triggers first
        let step_size = 0.01;

        let mut solver =
            UniversalODESolver::rk45(eq_system, values, arg, t0, y0, t_bound, step_size);

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);
        solver.set_neighborhood_check(1e-2);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= 1e-2);
    }

    #[test]
    fn test_universal_stop_condition_radau() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;

        let mut solver = UniversalODESolver::radau(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            1e-2,
            50,
            Some(0.01),
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= 1e-2); // Uses Radau's tolerance
    }

    #[test]
    fn test_universal_stop_condition_bdf() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-3);

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= 1e-2); // Uses BDF's atol
    }

    #[test]
    fn universal_bdf_propagates_invalid_stop_condition() {
        let mut solver = UniversalODESolver::bdf(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
        );
        solver.set_stop_condition(HashMap::from([("unknown".to_string(), 0.0)]));

        assert!(matches!(
            solver.try_initialize(),
            Err(UniversalOdeError::BdfStopCondition(
                BdfStopConditionError::UnknownVariable(name)
            )) if name == "unknown"
        ));
    }

    #[test]
    fn test_universal_stop_condition_be() {
        // Test: y' = y, y(0) = 1, stop when y reaches 1.5
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;

        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(0.01),
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 1.5);
        solver.set_stop_condition(stop_condition);
        solver.set_neighborhood_check(1e-2);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 1.5).abs() <= 1e-2);
    }

    #[test]
    fn test_universal_no_stop_condition() {
        // Test without stop condition - should run to t_bound
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let step_size = 0.1;

        let mut solver =
            UniversalODESolver::rk45(eq_system, values, arg, t0, y0, t_bound, step_size);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "finished");
        let (t_result, _) = solver.get_result();
        let t_res = t_result.unwrap();
        let final_t = t_res[t_res.len() - 1];
        assert!((final_t - t_bound).abs() <= step_size);
    }
}
