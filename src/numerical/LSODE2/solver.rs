use super::algorithm::{
    Lsode2AlgorithmController, Lsode2AlgorithmSnapshot, Lsode2ControllerExecutionCapabilities,
    Lsode2MethodFamily, Lsode2SwitchDecision, Lsode2SwitchTelemetry,
};
use super::config::{
    Lsode2JacobianBackend, Lsode2LinearSolverBackend, Lsode2NativeExecutionConfig,
    Lsode2ProblemConfig,
};
use super::linear_backends::{FaerSparseBdfLinearBackend, FaithfulBandedBdfLinearBackend};
use super::native_integration::{
    Lsode2NativeIntegrationLimits, Lsode2NativeIntegrationSummary, run_native_integration,
    run_native_integration_for_method,
};
use super::native_jacobian::{
    NativeJacobianStorage, compile_native_symbolic_jacobian_with_parameter_handle,
};
use super::native_preflight::{Lsode2NativeStepProbeSummary, run_native_step_preflight};
use super::native_step_engine::Lsode2NativeStepMethod;
use super::statistics::Lsode2NativeStatistics;
use crate::numerical::BDF::BDF_api::{BdfSolverOptions, ODEsolver as BdfOdeSolver};
use crate::symbolic::symbolic_ivp::IvpBackendError;
use crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics;
use nalgebra::{DMatrix, DVector};
use std::fmt;
use std::time::Instant;

/// Error type for the LSODE2 facade.
#[derive(Debug)]
pub enum Lsode2Error {
    UnsupportedBackend(&'static str),
    InvalidConfig(String),
    GeneratedBackend(IvpBackendError),
    NativeStep(String),
}

impl fmt::Display for Lsode2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBackend(message) => write!(f, "{message}"),
            Self::InvalidConfig(message) => write!(f, "{message}"),
            Self::GeneratedBackend(err) => write!(f, "{err}"),
            Self::NativeStep(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for Lsode2Error {}

impl From<IvpBackendError> for Lsode2Error {
    fn from(value: IvpBackendError) -> Self {
        Self::GeneratedBackend(value)
    }
}

/// Compact user-facing snapshot produced after an LSODE2 solve.
#[derive(Debug, Clone)]
pub struct Lsode2SolveSummary {
    pub method: &'static str,
    pub jacobian_backend: &'static str,
    pub linear_solver_backend: &'static str,
    pub status: String,
    pub time_points: usize,
    pub variable_count: usize,
    pub final_t: Option<f64>,
    pub final_y: Option<DVector<f64>>,
    pub max_abs_solution: f64,
    pub algorithm: Lsode2AlgorithmSnapshot,
    pub statistics: IvpBackendStatistics,
    pub native_statistics: Lsode2NativeStatistics,
    pub native_step_probe: Option<Lsode2NativeStepProbeSummary>,
    pub native_integration_preview: Option<Lsode2NativeIntegrationSummary>,
    pub native_integration_solve: Option<Lsode2NativeIntegrationSummary>,
}

/// First LSODE2 solver facade.
///
/// This is intentionally a thin wrapper over the tested `BDF` implementation.
/// The wrapper gives us a stable LSODE2 configuration API while the next
/// milestones replace the dense-only internals with sparse/banded production
/// linear backends.
pub struct Lsode2Solver {
    config: Lsode2ProblemConfig,
    inner: BdfOdeSolver,
    algorithm: Lsode2AlgorithmController,
    backend_prepared: bool,
    native_statistics: Lsode2NativeStatistics,
    native_step_probe: Option<Lsode2NativeStepProbeSummary>,
    native_integration_preview: Option<Lsode2NativeIntegrationSummary>,
    native_integration_solve: Option<Lsode2NativeIntegrationSummary>,
    native_override_status: Option<String>,
    native_override_result: Option<(DVector<f64>, DMatrix<f64>)>,
}

impl Lsode2Solver {
    pub fn new(config: Lsode2ProblemConfig) -> Result<Self, Lsode2Error> {
        validate_supported_milestone1_backend(&config)?;
        validate_parameter_config(&config)?;
        validate_controller_config(&config)?;

        let mut options = BdfSolverOptions::new(
            config.eq_system.clone(),
            config.values.clone(),
            config.arg.clone(),
            config.method.as_bdf_method_name(),
            config.t0,
            config.y0.clone(),
            config.t_bound,
            config.max_step,
            config.rtol,
            config.atol,
            config.jac_sparsity.clone(),
            config.vectorized,
            config.first_step,
        )
        .with_max_bdf_order(config.controller.max_bdf_order)
        .with_generated_backend_config(config.backend.generated_backend.clone());
        if let Some(parameters) = config.equation_parameters.clone() {
            options = options.with_equation_parameters(parameters);
        }
        if let Some(values) = config.equation_parameter_values.clone() {
            options = options.with_equation_parameter_values(values);
        }

        let mut inner = BdfOdeSolver::new_with_options(options);
        install_native_jacobian_factory(&mut inner, &config);
        install_linear_backend_factory(&mut inner, config.backend.linear_solver_backend);

        let controller_capabilities = controller_execution_capabilities(&config);
        Ok(Self {
            algorithm: Lsode2AlgorithmController::new_with_capabilities(
                config.controller,
                controller_capabilities,
            ),
            config,
            inner,
            backend_prepared: false,
            native_statistics: Lsode2NativeStatistics::default(),
            native_step_probe: None,
            native_integration_preview: None,
            native_integration_solve: None,
            native_override_status: None,
            native_override_result: None,
        })
    }

    pub fn config(&self) -> &Lsode2ProblemConfig {
        &self.config
    }

    /// Prepares symbolic residual/Jacobian callbacks and linear backend wiring.
    ///
    /// This is separated from [`Self::solve`] so benchmark and diagnostic code
    /// can measure backend preparation independently from time integration.
    /// Repeated calls are cheap no-ops unless a future mutator invalidates the
    /// prepared backend.
    pub fn prepare(&mut self) -> Result<(), Lsode2Error> {
        if !self.backend_prepared {
            let started = Instant::now();
            self.inner.try_generate()?;
            self.native_statistics
                .record_backend_prepare_duration(started.elapsed());
            self.native_statistics
                .sync_from_bridge(&self.inner.get_statistics());
            self.backend_prepared = true;
        }
        Ok(())
    }

    pub fn is_prepared(&self) -> bool {
        self.backend_prepared
    }

    pub fn solve(&mut self) -> Result<(), Lsode2Error> {
        self.native_integration_preview = None;
        self.native_integration_solve = None;
        self.native_override_status = None;
        self.native_override_result = None;
        let decision = self.algorithm_switch_decision_stateful();
        self.algorithm.record_switch_decision(decision);
        self.native_statistics.record_algorithm_decision(&decision);
        match self.config.native_execution {
            Lsode2NativeExecutionConfig::ExperimentalNativeSolve {
                max_step_attempts,
                max_accepted_steps,
            } => {
                let started = Instant::now();
                let native_method = native_method_for_decision(&decision);
                let solve_summary = self
                    .run_native_integration_preview_internal_with_method(
                        Lsode2NativeIntegrationLimits::new(max_step_attempts, max_accepted_steps),
                        false,
                        Some(native_method),
                    )?
                    .ok_or_else(|| {
                        Lsode2Error::NativeStep(
                            "experimental native solve requires a sparse or banded LSODE2 backend"
                                .to_string(),
                        )
                    })?;
                self.native_step_probe =
                    Some(native_step_probe_from_integration_summary(&solve_summary));
                self.native_override_result =
                    Some(native_result_from_integration_summary(&solve_summary));
                self.native_override_status = Some(
                    if solve_summary.reached_t_bound {
                        "finished_native_experimental"
                    } else {
                        "finished_native_experimental_partial"
                    }
                    .to_string(),
                );
                self.native_integration_solve = Some(solve_summary);
                self.native_statistics
                    .record_solve_duration(started.elapsed());
                return Ok(());
            }
            Lsode2NativeExecutionConfig::Disabled => {
                self.prepare()?;
                self.native_step_probe = self.run_native_step_probe()?;
            }
            Lsode2NativeExecutionConfig::PreviewBeforeBridge {
                max_step_attempts,
                max_accepted_steps,
            } => {
                self.prepare()?;
                let native_method = native_method_for_decision(&decision);
                let preview = self.run_native_integration_preview_internal_with_method(
                    Lsode2NativeIntegrationLimits::new(max_step_attempts, max_accepted_steps),
                    false,
                    Some(native_method),
                )?;
                self.native_integration_preview = preview.clone();
                self.native_step_probe = preview
                    .as_ref()
                    .map(native_step_probe_from_integration_summary);
            }
        }
        let started = Instant::now();
        let bridge_accepted_before = self.native_statistics.bridge_accepted_steps;
        self.inner.main_loop();
        self.native_statistics
            .record_solve_duration(started.elapsed());
        self.native_statistics
            .sync_from_bridge(&self.inner.get_statistics());
        let bridge_accepted_after = self.inner.get_result().0.len();
        self.native_statistics.bridge_accepted_steps = bridge_accepted_after;
        let accepted_steps_delta = bridge_accepted_after.saturating_sub(bridge_accepted_before);
        self.algorithm
            .record_accepted_steps_for_switch_probe(accepted_steps_delta);
        Ok(())
    }

    /// Solves the problem and returns a compact diagnostics/result snapshot.
    pub fn solve_with_summary(&mut self) -> Result<Lsode2SolveSummary, Lsode2Error> {
        self.solve()?;
        Ok(self.summary())
    }

    pub fn solve_panicking(&mut self) {
        self.solve().expect("LSODE2 solve should succeed");
    }

    pub fn get_result(&self) -> (DVector<f64>, DMatrix<f64>) {
        self.native_override_result
            .clone()
            .unwrap_or_else(|| self.inner.get_result())
    }

    pub fn status(&self) -> &str {
        self.native_override_status
            .as_deref()
            .unwrap_or_else(|| self.inner.get_status())
    }

    pub fn statistics(&self) -> IvpBackendStatistics {
        self.inner.get_statistics()
    }

    /// Returns LSODE2-native statistics collected at the facade level.
    ///
    /// During the current bridge milestone this includes:
    /// - explicit LSODE2 prepare/solve timings,
    /// - Adams/BDF controller decisions,
    /// - a synchronized mirror of bridge/BDF backend counters.
    ///
    /// Future native LSODE2 steps will additionally populate native step,
    /// nonlinear, Jacobian-refresh, and explicit linear-solve counters here.
    pub fn native_statistics(&self) -> Lsode2NativeStatistics {
        self.native_statistics.clone()
    }

    pub fn native_step_probe(&self) -> Option<&Lsode2NativeStepProbeSummary> {
        self.native_step_probe.as_ref()
    }

    pub fn native_integration_preview(&self) -> Option<&Lsode2NativeIntegrationSummary> {
        self.native_integration_preview.as_ref()
    }

    pub fn native_integration_solve(&self) -> Option<&Lsode2NativeIntegrationSummary> {
        self.native_integration_solve.as_ref()
    }

    /// Runs the current native sparse/banded step engine for a bounded number of
    /// step attempts and returns its multi-step summary.
    ///
    /// This does not replace the bridge-backed [`Self::solve`] path yet. It is
    /// a controlled solver-level entry point for the emerging native engine so
    /// tests and stories can compare real native stepping behavior without
    /// reaching into internal modules.
    pub fn run_native_integration_preview(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        self.run_native_integration_preview_internal(limits, true)
    }

    pub fn run_native_integration_preview_for_family(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
        family: Lsode2MethodFamily,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        let method = match family {
            Lsode2MethodFamily::Bdf => Lsode2NativeStepMethod::BdfLike,
            Lsode2MethodFamily::Adams => Lsode2NativeStepMethod::AdamsLike,
        };
        self.run_native_integration_preview_internal_with_method(limits, true, Some(method))
    }

    fn run_native_integration_preview_internal(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
        record_decision: bool,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        self.run_native_integration_preview_internal_with_method(limits, record_decision, None)
    }

    fn run_native_integration_preview_internal_with_method(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
        record_decision: bool,
        method_override: Option<Lsode2NativeStepMethod>,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        let decision = if record_decision {
            self.algorithm_switch_decision_stateful()
        } else {
            self.algorithm_switch_decision()
        };
        if record_decision {
            self.algorithm.record_switch_decision(decision);
            self.native_statistics.record_algorithm_decision(&decision);
        }
        let method = method_override.unwrap_or_else(|| native_method_for_decision(&decision));
        let outcome = match method {
            Lsode2NativeStepMethod::BdfLike => {
                run_native_integration(&self.config, limits).map_err(Lsode2Error::from)?
            }
            Lsode2NativeStepMethod::AdamsLike => {
                run_native_integration_for_method(&self.config, limits, method)
                    .map_err(Lsode2Error::from)?
            }
        };
        if let Some(summary) = &outcome.summary {
            let family = method_family_from_native_step_method(method);
            let cost =
                integration_method_cost_estimate(&outcome.statistics, summary.accepted_steps);
            self.native_statistics
                .record_native_method_cost_sample(family, cost);
        }
        merge_native_statistics(&mut self.native_statistics, &outcome.statistics);
        if let Some(summary) = &outcome.summary {
            self.algorithm
                .record_accepted_steps_for_switch_probe(summary.accepted_steps);
        }
        Ok(outcome.summary)
    }

    pub fn statistics_report(&self) -> String {
        format!(
            "{} | {}",
            self.inner.statistics_report(),
            self.native_statistics.table_report()
        )
    }

    pub fn bdf_max_order_cap(&self) -> usize {
        self.inner.bdf_max_order_cap()
    }

    pub fn bdf_current_order(&self) -> usize {
        self.inner.bdf_current_order()
    }

    pub fn bdf_equal_step_count(&self) -> usize {
        self.inner.bdf_equal_step_count()
    }

    pub fn algorithm_snapshot(&self) -> Lsode2AlgorithmSnapshot {
        self.algorithm.snapshot_with_bdf_runtime(
            Some(self.bdf_current_order()),
            Some(self.bdf_max_order_cap()),
            Some(self.bdf_equal_step_count()),
        )
    }

    /// Returns the current controller method-family decision for current
    /// runtime telemetry.
    ///
    /// The current executable engine is BDF. This method makes any temporary
    /// fallback from the future Adams/BDF policy visible without forcing users
    /// to inspect the full solve summary.
    pub fn algorithm_switch_decision(&self) -> Lsode2SwitchDecision {
        self.algorithm
            .switch_decision(self.algorithm_switch_telemetry_from_runtime())
    }

    fn algorithm_switch_decision_stateful(&mut self) -> Lsode2SwitchDecision {
        let telemetry = self.algorithm_switch_telemetry_from_runtime();
        self.algorithm.switch_decision_stateful(telemetry)
    }

    /// Returns the controller decision for explicit telemetry.
    ///
    /// Story tests and future adaptive integration code can use this to probe
    /// the same policy with suspected-stiffness or convergence-failure signals.
    pub fn algorithm_switch_decision_with_telemetry(
        &self,
        telemetry: Lsode2SwitchTelemetry,
    ) -> Lsode2SwitchDecision {
        self.algorithm.switch_decision(telemetry)
    }

    fn algorithm_switch_telemetry_from_runtime(&self) -> Lsode2SwitchTelemetry {
        let accepted_steps = self
            .native_statistics
            .native_step_accepts
            .max(self.native_statistics.bridge_accepted_steps);
        let rejected_steps = self.native_statistics.native_step_rejects_error_test
            + self.native_statistics.native_step_rejects_nonlinear;
        let convergence_failures = self.native_statistics.native_step_rejects_nonlinear;

        let mut telemetry = Lsode2SwitchTelemetry::default()
            .with_accepted_steps(accepted_steps)
            .with_rejected_steps(rejected_steps)
            .with_convergence_failures(convergence_failures);
        if let Some(cost) = self.native_statistics.adams_step_cost_estimate() {
            telemetry = telemetry.with_adams_step_cost_estimate(cost);
        }
        if let Some(cost) = self.native_statistics.bdf_step_cost_estimate() {
            telemetry = telemetry.with_bdf_step_cost_estimate(cost);
        }
        telemetry
    }

    /// Builds a snapshot from the current solver result and statistics.
    ///
    /// This is useful for story/benchmark tables: callers do not need to know
    /// the internal `(t_result, y_result)` layout just to print final values and
    /// counters.
    pub fn summary(&self) -> Lsode2SolveSummary {
        let (t, y) = self.get_result();
        let final_t = (!t.is_empty()).then(|| t[t.len() - 1]);
        let final_y = (y.nrows() > 0).then(|| {
            DVector::from_iterator(y.ncols(), (0..y.ncols()).map(|col| y[(y.nrows() - 1, col)]))
        });
        let max_abs_solution = y.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));

        Lsode2SolveSummary {
            method: self.config.method.label(),
            jacobian_backend: self.config.backend.jacobian_backend.label(),
            linear_solver_backend: self.config.backend.linear_solver_backend.label(),
            status: self.status().to_string(),
            time_points: t.len(),
            variable_count: y.ncols(),
            final_t,
            final_y,
            max_abs_solution,
            algorithm: self.algorithm_snapshot(),
            statistics: self.statistics(),
            native_statistics: self.native_statistics(),
            native_step_probe: self.native_step_probe.clone(),
            native_integration_preview: self.native_integration_preview.clone(),
            native_integration_solve: self.native_integration_solve.clone(),
        }
    }

    /// Updates numeric values for symbolic equation parameters.
    ///
    /// Updating parameters invalidates prepared BDF state.  The next
    /// [`Self::prepare`] or [`Self::solve`] call rebuilds callbacks and cached
    /// initial quantities consistently.
    pub fn set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), Lsode2Error> {
        let expected = self
            .config
            .equation_parameters
            .as_ref()
            .map_or(0, |parameters| parameters.len());
        if expected != values.len() {
            return Err(Lsode2Error::GeneratedBackend(
                IvpBackendError::ParameterCountMismatch {
                    expected,
                    actual: values.len(),
                },
            ));
        }

        self.config.equation_parameter_values = Some(values.clone());
        self.inner.set_parameter_values(values)?;
        self.backend_prepared = false;
        Ok(())
    }

    fn run_native_step_probe(
        &mut self,
    ) -> Result<Option<Lsode2NativeStepProbeSummary>, Lsode2Error> {
        let outcome = run_native_step_preflight(&self.config).map_err(Lsode2Error::from)?;
        merge_native_statistics(&mut self.native_statistics, &outcome.statistics);
        Ok(outcome.summary)
    }
}

fn native_method_for_decision(decision: &Lsode2SwitchDecision) -> Lsode2NativeStepMethod {
    match decision.preferred_family {
        Lsode2MethodFamily::Adams => Lsode2NativeStepMethod::AdamsLike,
        Lsode2MethodFamily::Bdf => Lsode2NativeStepMethod::BdfLike,
    }
}

fn method_family_from_native_step_method(method: Lsode2NativeStepMethod) -> Lsode2MethodFamily {
    match method {
        Lsode2NativeStepMethod::BdfLike => Lsode2MethodFamily::Bdf,
        Lsode2NativeStepMethod::AdamsLike => Lsode2MethodFamily::Adams,
    }
}

fn integration_method_cost_estimate(
    run_stats: &Lsode2NativeStatistics,
    accepted_steps: usize,
) -> f64 {
    let acc = accepted_steps.max(1) as f64;
    let residual_density = run_stats.native_residual_calls as f64 / acc;
    let jacobian_density = run_stats.native_jacobian_calls as f64 / acc;
    let linear_density = run_stats.native_linear_solve_calls as f64 / acc;
    let nonlinear_reject_density = run_stats.native_step_rejects_nonlinear as f64 / acc;
    let correction_continue_density = run_stats.native_nonlinear_continue_count as f64 / acc;
    residual_density
        + 2.0 * jacobian_density
        + 2.5 * linear_density
        + 1.5 * nonlinear_reject_density
        + 0.5 * correction_continue_density
}

fn merge_native_statistics(dst: &mut Lsode2NativeStatistics, src: &Lsode2NativeStatistics) {
    dst.native_step_attempts += src.native_step_attempts;
    dst.native_step_accepts += src.native_step_accepts;
    dst.native_step_rejects_error_test += src.native_step_rejects_error_test;
    dst.native_step_rejects_nonlinear += src.native_step_rejects_nonlinear;
    dst.native_nonlinear_continue_count += src.native_nonlinear_continue_count;
    dst.native_nonlinear_converged_count += src.native_nonlinear_converged_count;
    dst.native_nonlinear_diverged_count += src.native_nonlinear_diverged_count;
    dst.native_nonlinear_iteration_limit_count += src.native_nonlinear_iteration_limit_count;
    dst.native_stale_jacobian_retry_count += src.native_stale_jacobian_retry_count;
    dst.native_jacobian_refresh_requests += src.native_jacobian_refresh_requests;
    dst.native_linear_solve_calls += src.native_linear_solve_calls;
    dst.native_linear_solve_ms_total += src.native_linear_solve_ms_total;
    dst.native_residual_calls += src.native_residual_calls;
    dst.native_residual_ms_total += src.native_residual_ms_total;
    dst.native_jacobian_calls += src.native_jacobian_calls;
    dst.native_jacobian_ms_total += src.native_jacobian_ms_total;
    dst.native_jcur_current_count += src.native_jcur_current_count;
    dst.native_jcur_stale_count += src.native_jcur_stale_count;
    dst.native_ipup_up_to_date_count += src.native_ipup_up_to_date_count;
    dst.native_ipup_needs_update_count += src.native_ipup_needs_update_count;
    dst.native_predictor_ipup_trigger_none_count += src.native_predictor_ipup_trigger_none_count;
    dst.native_predictor_ipup_trigger_predictor_rc_ccmax_count +=
        src.native_predictor_ipup_trigger_predictor_rc_ccmax_count;
    dst.native_predictor_ipup_trigger_predictor_msbp_count +=
        src.native_predictor_ipup_trigger_predictor_msbp_count;
    dst.native_predictor_ipup_trigger_predictor_rc_ccmax_and_msbp_count +=
        src.native_predictor_ipup_trigger_predictor_rc_ccmax_and_msbp_count;
    dst.native_predictor_ipup_trigger_failure_path_count +=
        src.native_predictor_ipup_trigger_failure_path_count;
    dst.native_ipup_trigger_none_count += src.native_ipup_trigger_none_count;
    dst.native_ipup_trigger_predictor_rc_ccmax_count +=
        src.native_ipup_trigger_predictor_rc_ccmax_count;
    dst.native_ipup_trigger_predictor_msbp_count += src.native_ipup_trigger_predictor_msbp_count;
    dst.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count +=
        src.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count;
    dst.native_ipup_trigger_failure_path_count += src.native_ipup_trigger_failure_path_count;
    dst.native_kflag_ok_count += src.native_kflag_ok_count;
    dst.native_kflag_error_test_failure_count += src.native_kflag_error_test_failure_count;
    dst.native_kflag_repeated_error_test_failure_count +=
        src.native_kflag_repeated_error_test_failure_count;
    dst.native_kflag_convergence_failure_count += src.native_kflag_convergence_failure_count;
    dst.native_kflag_repeated_convergence_failure_count +=
        src.native_kflag_repeated_convergence_failure_count;
    dst.native_icf_none_count += src.native_icf_none_count;
    dst.native_icf_refresh_requested_count += src.native_icf_refresh_requested_count;
    dst.native_icf_refresh_did_not_recover_count += src.native_icf_refresh_did_not_recover_count;
    dst.native_iret_normal_flow_count += src.native_iret_normal_flow_count;
    dst.native_iret_rescale_history_count += src.native_iret_rescale_history_count;
    dst.native_iret_retry_after_error_test_failure_count +=
        src.native_iret_retry_after_error_test_failure_count;
    dst.native_iret_restart_with_derivative_refresh_count +=
        src.native_iret_restart_with_derivative_refresh_count;
    dst.native_redo_none_count += src.native_redo_none_count;
    dst.native_redo_corrector_refresh_same_step_count +=
        src.native_redo_corrector_refresh_same_step_count;
    dst.native_redo_corrector_failure_retry_count += src.native_redo_corrector_failure_retry_count;
    dst.native_redo_error_test_retry_count += src.native_redo_error_test_retry_count;
    dst.native_redo_repeated_error_reset_count += src.native_redo_repeated_error_reset_count;
    dst.native_redo_history_or_step_size_changed_count +=
        src.native_redo_history_or_step_size_changed_count;
    dst.native_ialth_zero_count += src.native_ialth_zero_count;
    dst.native_ialth_positive_count += src.native_ialth_positive_count;
    dst.native_ialth_sum += src.native_ialth_sum;
}

fn native_step_probe_from_integration_summary(
    summary: &Lsode2NativeIntegrationSummary,
) -> Lsode2NativeStepProbeSummary {
    Lsode2NativeStepProbeSummary {
        outcome: summary.last_report.outcome_label(),
        accepted: summary.last_report.accepted(),
        iterations: summary.total_iterations,
        attempted_steps: summary.attempted_steps,
        accepted_steps: summary.accepted_steps,
        rejected_steps: summary.rejected_steps,
        t_trial: summary.first_report.predicted.t_trial,
        h_trial: summary.first_report.predicted.h_trial,
        final_t: summary.final_t,
        final_h: summary.final_h,
        telemetry: summary.last_report.telemetry,
    }
}

fn native_result_from_integration_summary(
    summary: &Lsode2NativeIntegrationSummary,
) -> (DVector<f64>, DMatrix<f64>) {
    let t_result = DVector::from_vec(summary.accepted_t_history.clone());
    let rows = summary.accepted_y_history.len();
    let cols = summary.accepted_y_history.first().map_or(0, Vec::len);
    let mut y_result = DMatrix::zeros(rows, cols);
    for (row, values) in summary.accepted_y_history.iter().enumerate() {
        for (col, value) in values.iter().copied().enumerate() {
            y_result[(row, col)] = value;
        }
    }
    (t_result, y_result)
}

fn validate_supported_milestone1_backend(config: &Lsode2ProblemConfig) -> Result<(), Lsode2Error> {
    match config.backend.jacobian_backend {
        Lsode2JacobianBackend::SymbolicGenerated => {}
        Lsode2JacobianBackend::AnalyticClosure => {
            return Err(Lsode2Error::UnsupportedBackend(
                "LSODE2 milestone 1 supports symbolic generated Jacobians only; analytic closures are planned for the next backend layer",
            ));
        }
        Lsode2JacobianBackend::FiniteDifference => {
            return Err(Lsode2Error::UnsupportedBackend(
                "LSODE2 milestone 1 supports symbolic generated Jacobians only; finite-difference Jacobians are planned for the next backend layer",
            ));
        }
    }

    Ok(())
}

fn validate_parameter_config(config: &Lsode2ProblemConfig) -> Result<(), Lsode2Error> {
    match (
        config.equation_parameters.as_ref(),
        config.equation_parameter_values.as_ref(),
    ) {
        (Some(parameters), Some(values)) if parameters.len() != values.len() => Err(
            Lsode2Error::GeneratedBackend(IvpBackendError::ParameterCountMismatch {
                expected: parameters.len(),
                actual: values.len(),
            }),
        ),
        (Some(parameters), None) if !parameters.is_empty() => Err(Lsode2Error::GeneratedBackend(
            IvpBackendError::MissingParameterValues {
                expected: parameters.len(),
            },
        )),
        (None, Some(values)) if !values.is_empty() => Err(Lsode2Error::GeneratedBackend(
            IvpBackendError::ParameterCountMismatch {
                expected: 0,
                actual: values.len(),
            },
        )),
        _ => Ok(()),
    }
}

fn validate_controller_config(config: &Lsode2ProblemConfig) -> Result<(), Lsode2Error> {
    config
        .controller
        .validate()
        .map_err(Lsode2Error::InvalidConfig)?;
    let plan = config
        .controller
        .execution_plan_with_capabilities(controller_execution_capabilities(config));
    if !plan.is_executable_now() {
        return Err(Lsode2Error::UnsupportedBackend(plan.message));
    }
    Ok(())
}

fn controller_execution_capabilities(
    config: &Lsode2ProblemConfig,
) -> Lsode2ControllerExecutionCapabilities {
    let adams_engine_available = matches!(
        config.native_execution,
        Lsode2NativeExecutionConfig::PreviewBeforeBridge { .. }
            | Lsode2NativeExecutionConfig::ExperimentalNativeSolve { .. }
    ) && matches!(
        config.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::SparseFaer | Lsode2LinearSolverBackend::BandedFaithful
    );
    Lsode2ControllerExecutionCapabilities {
        adams_engine_available,
    }
}

fn install_linear_backend_factory(inner: &mut BdfOdeSolver, backend: Lsode2LinearSolverBackend) {
    match backend {
        Lsode2LinearSolverBackend::Dense => {}
        Lsode2LinearSolverBackend::SparseFaer => {
            inner
                .set_bdf_linear_backend_factory(|| Box::new(FaerSparseBdfLinearBackend::default()));
        }
        Lsode2LinearSolverBackend::BandedFaithful => {
            inner.set_bdf_linear_backend_factory(|| {
                Box::new(FaithfulBandedBdfLinearBackend::default())
            });
        }
    }
}

fn install_native_jacobian_factory(inner: &mut BdfOdeSolver, config: &Lsode2ProblemConfig) {
    if config.backend.jacobian_backend != Lsode2JacobianBackend::SymbolicGenerated {
        return;
    }

    let storage = match config.backend.linear_solver_backend {
        Lsode2LinearSolverBackend::Dense => return,
        Lsode2LinearSolverBackend::SparseFaer => NativeJacobianStorage::SparseTriplets,
        Lsode2LinearSolverBackend::BandedFaithful => NativeJacobianStorage::Banded,
    };

    let equations = config.eq_system.clone();
    let variables = config.values.clone();
    let time_arg = config.arg.clone();
    let equation_parameters = config.equation_parameters.clone();
    inner.set_bdf_native_jacobian_factory(move |parameter_values_handle| {
        compile_native_symbolic_jacobian_with_parameter_handle(
            &equations,
            &variables,
            time_arg.as_str(),
            equation_parameters.as_deref(),
            parameter_values_handle,
            storage,
        )
    });
}
