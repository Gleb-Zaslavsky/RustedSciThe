use super::algorithm::{
    Lsode2AlgorithmController, Lsode2AlgorithmSnapshot, Lsode2ControllerExecutionCapabilities,
    Lsode2ControllerMode, Lsode2MethodFamily, Lsode2SwitchDecision, Lsode2SwitchReason,
    Lsode2SwitchTelemetry,
};
use super::config::{
    Lsode2JacobianBackend, Lsode2LinearSolverBackend, Lsode2NativeExecutionConfig,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2ResolvedPlan, Lsode2StopComparator,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use super::linear_backends::{FaerSparseBdfLinearBackend, FaithfulBandedBdfLinearBackend};
use super::native_integration::{
    run_native_integration, run_native_integration_for_method,
    run_native_integration_for_method_with_policy, Lsode2NativeIntegrationLimits,
    Lsode2NativeIntegrationSummary, Lsode2NativeTerminationKind,
};
use super::native_jacobian::{
    compile_native_sparse_aot_jacobian_with_parameter_handle,
    compile_native_symbolic_jacobian_with_parameter_handle, NativeJacobianStorage,
};
use super::native_preflight::{run_native_step_preflight, Lsode2NativeStepProbeSummary};
use super::native_step_engine::Lsode2NativeStepMethod;
use super::statistics::Lsode2NativeStatistics;
use crate::numerical::BDF::BDF_api::{BdfSolverOptions, ODEsolver as BdfOdeSolver};
use crate::symbolic::symbolic_ivp::{IvpBackendError, IvpSymbolicAssemblyBackend};
use crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Default)]
struct Lsode2SwitchTelemetryHints {
    stiffness_ratio: Option<f64>,
    adams_step_size_cap_estimate: Option<f64>,
    bdf_step_size_cap_estimate: Option<f64>,
}

impl Lsode2SwitchTelemetryHints {
    fn absorb(&mut self, telemetry: Lsode2SwitchTelemetry) {
        if let Some(value) = telemetry
            .stiffness_ratio
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            self.stiffness_ratio = Some(
                self.stiffness_ratio
                    .map_or(value, |current| current.max(value)),
            );
        }
        if let Some(value) = telemetry
            .adams_step_size_cap_estimate
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            self.adams_step_size_cap_estimate = Some(
                self.adams_step_size_cap_estimate
                    .map_or(value, |current| current.max(value)),
            );
        }
        if let Some(value) = telemetry
            .bdf_step_size_cap_estimate
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            self.bdf_step_size_cap_estimate = Some(
                self.bdf_step_size_cap_estimate
                    .map_or(value, |current| current.max(value)),
            );
        }
    }

    fn apply_to(self, mut telemetry: Lsode2SwitchTelemetry) -> Lsode2SwitchTelemetry {
        let stiffness_missing_or_placeholder = match telemetry.stiffness_ratio {
            None => true,
            Some(value) => !value.is_finite() || value <= 0.0,
        };
        if stiffness_missing_or_placeholder {
            telemetry.stiffness_ratio = self.stiffness_ratio;
        }
        if telemetry.adams_step_size_cap_estimate.is_none() {
            telemetry.adams_step_size_cap_estimate = self.adams_step_size_cap_estimate;
        }
        if telemetry.bdf_step_size_cap_estimate.is_none() {
            telemetry.bdf_step_size_cap_estimate = self.bdf_step_size_cap_estimate;
        }
        telemetry
    }
}

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
    pub linear_solver_reason: &'static str,
    pub resolved_source: &'static str,
    pub resolved_structure: &'static str,
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
    pub native_termination_kind: Option<&'static str>,
}

/// First LSODE2 solver facade.
///
/// This is intentionally a thin wrapper over the tested `BDF` implementation.
/// The wrapper gives us a stable LSODE2 configuration API while the next
/// milestones replace the dense-only internals with sparse/banded production
/// linear backends.
pub struct Lsode2Solver {
    config: Lsode2ProblemConfig,
    resolved_plan: Lsode2ResolvedPlan,
    inner: BdfOdeSolver,
    algorithm: Lsode2AlgorithmController,
    backend_prepared: bool,
    native_statistics: Lsode2NativeStatistics,
    native_step_probe: Option<Lsode2NativeStepProbeSummary>,
    native_integration_preview: Option<Lsode2NativeIntegrationSummary>,
    native_integration_solve: Option<Lsode2NativeIntegrationSummary>,
    native_override_status: Option<String>,
    native_override_result: Option<(DVector<f64>, DMatrix<f64>)>,
    last_native_switch_telemetry: Option<Lsode2SwitchTelemetry>,
    switch_telemetry_hints: Lsode2SwitchTelemetryHints,
}

impl Lsode2Solver {
    pub fn new(mut config: Lsode2ProblemConfig) -> Result<Self, Lsode2Error> {
        config.sync_legacy_backend_from_policy();
        validate_supported_backend_routes(&config)?;
        validate_parameter_config(&config)?;
        validate_controller_config(&config)?;
        validate_stop_condition_config(&config)?;
        let resolved_plan = config.resolve_plan();

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
        .with_generated_backend_config(config.backend.generated_backend.clone())
        .with_symbolic_assembly_backend(match config.residual_jacobian_source {
            Lsode2ResidualJacobianSource::Symbolic { assembly, .. } => match assembly {
                Lsode2SymbolicAssemblyBackend::ExprLegacy => IvpSymbolicAssemblyBackend::ExprLegacy,
                Lsode2SymbolicAssemblyBackend::AtomView => IvpSymbolicAssemblyBackend::AtomView,
            },
            Lsode2ResidualJacobianSource::Analytical => IvpSymbolicAssemblyBackend::ExprLegacy,
        });
        if let Some(parameters) = config.equation_parameters.clone() {
            options = options.with_equation_parameters(parameters);
        }
        if let Some(values) = config.equation_parameter_values.clone() {
            options = options.with_equation_parameter_values(values);
        }

        let mut inner = BdfOdeSolver::new_with_options(options);
        if let Some(legacy_stop_conditions) = bridge_compatible_stop_conditions(&config) {
            inner.set_stop_condition(legacy_stop_conditions);
        }
        install_native_jacobian_factory(&mut inner, &config);
        install_linear_backend_factory(&mut inner, resolved_plan.linear_solver.to_backend());

        let controller_capabilities = controller_execution_capabilities(&config);
        Ok(Self {
            algorithm: Lsode2AlgorithmController::new_with_capabilities(
                config.controller,
                controller_capabilities,
            ),
            config,
            resolved_plan,
            inner,
            backend_prepared: false,
            native_statistics: Lsode2NativeStatistics::default(),
            native_step_probe: None,
            native_integration_preview: None,
            native_integration_solve: None,
            native_override_status: None,
            native_override_result: None,
            last_native_switch_telemetry: None,
            switch_telemetry_hints: Lsode2SwitchTelemetryHints::default(),
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
        let mut decision = self.algorithm_switch_decision_stateful();
        self.algorithm.record_switch_decision(decision);
        self.native_statistics.record_algorithm_decision(&decision);
        let mut run_bridge = false;
        match self.config.native_execution {
            Lsode2NativeExecutionConfig::NativeSolve {
                max_step_attempts,
                max_accepted_steps,
            } => {
                if self.should_run_switch_probe_before_full_native_solve(decision) {
                    self.run_switch_probe_before_full_native_solve(decision)?;
                    decision = self.algorithm_switch_decision_stateful();
                    self.algorithm.record_switch_decision(decision);
                    self.native_statistics.record_algorithm_decision(&decision);
                }
                let started = Instant::now();
                let limits =
                    Lsode2NativeIntegrationLimits::new(max_step_attempts, max_accepted_steps);
                let maybe_solve_summary =
                    if self.config.controller.mode == Lsode2ControllerMode::AutomaticAdamsBdf {
                        self.run_native_integration_internal_with_method(limits, true, None)?
                    } else {
                        let native_method = native_method_for_decision(&decision);
                        self.run_native_integration_internal_with_method(
                            limits,
                            false,
                            Some(native_method),
                        )?
                    };
                if let Some(solve_summary) = maybe_solve_summary {
                    self.native_step_probe =
                        Some(native_step_probe_from_integration_summary(&solve_summary));
                    self.native_override_result =
                        Some(native_result_from_integration_summary(&solve_summary));
                    self.native_override_status = Some(
                        status_for_native_termination_kind(solve_summary.termination_kind)
                            .to_string(),
                    );
                    self.native_integration_solve = Some(solve_summary);
                    self.native_statistics
                        .record_solve_duration(started.elapsed());
                    return Ok(());
                }
                // Keep bridge path reachable as an explicit fallback when
                // native faithful integration is unavailable for a given route.
                run_bridge = true;
                self.prepare()?;
                self.native_step_probe = self.run_native_step_probe()?;
            }
            Lsode2NativeExecutionConfig::BridgeSolve | Lsode2NativeExecutionConfig::Disabled => {
                run_bridge = true;
                self.prepare()?;
                self.native_step_probe = self.run_native_step_probe()?;
            }
            Lsode2NativeExecutionConfig::ProbeBeforeBridge {
                max_step_attempts,
                max_accepted_steps,
            } => {
                self.prepare()?;
                let native_method = native_method_for_decision(&decision);
                let preview = self.run_native_integration_internal_with_method(
                    Lsode2NativeIntegrationLimits::new(max_step_attempts, max_accepted_steps),
                    false,
                    Some(native_method),
                )?;
                self.native_integration_preview = preview.clone();
                self.native_step_probe = preview
                    .as_ref()
                    .map(native_step_probe_from_integration_summary);
                // ProbeBeforeBridge must always continue to the bridge solve path.
                run_bridge = true;
            }
        }
        if !run_bridge {
            return Ok(());
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
        let bridge = self.inner.get_statistics();
        let bridge_has_activity = bridge.solve_calls > 0
            || bridge.step_calls > 0
            || bridge.residual_calls > 0
            || bridge.jacobian_calls > 0;
        if bridge_has_activity {
            return bridge;
        }

        let native = &self.native_statistics;
        let native_has_activity = native.solve_calls > 0
            || native.backend_prepare_calls > 0
            || native.native_step_attempts > 0
            || native.native_residual_calls > 0
            || native.native_jacobian_calls > 0
            || native.native_linear_solve_calls > 0;
        if !native_has_activity {
            return bridge;
        }

        IvpBackendStatistics {
            backend_prepare_calls: native
                .backend_prepare_calls
                .max(native.bridge_prepare_calls),
            backend_prepare_ms_total: if native.backend_prepare_ms_total > 0.0 {
                native.backend_prepare_ms_total
            } else {
                native.bridge_prepare_ms_total
            },
            solve_calls: native.solve_calls.max(native.bridge_solve_calls),
            solve_ms_total: if native.solve_ms_total > 0.0 {
                native.solve_ms_total
            } else {
                native.bridge_solve_ms_total
            },
            step_calls: native.bridge_step_calls.max(native.native_step_attempts),
            nonlinear_solve_calls: native.bridge_nonlinear_solve_calls.max(
                native.native_nonlinear_converged_count + native.native_nonlinear_continue_count,
            ),
            nonlinear_iterations_total: native.bridge_nonlinear_iterations_total,
            residual_calls: native
                .bridge_residual_calls
                .max(native.native_residual_calls),
            residual_ms_total: if native.native_residual_ms_total > 0.0 {
                native.native_residual_ms_total
            } else {
                native.bridge_residual_ms_total
            },
            jacobian_calls: native
                .bridge_jacobian_calls
                .max(native.native_jacobian_calls),
            jacobian_ms_total: if native.native_jacobian_ms_total > 0.0 {
                native.native_jacobian_ms_total
            } else {
                native.bridge_jacobian_ms_total
            },
            bdf_nfev_total: native.bridge_bdf_nfev_total,
            bdf_njev_total: native
                .bridge_bdf_njev_total
                .max(native.native_jacobian_calls),
            bdf_nlu_total: native
                .bridge_bdf_nlu_total
                .max(native.native_linear_solve_calls),
        }
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

    /// Runs the current native sparse/banded step engine for a bounded number
    /// of step attempts and returns its multi-step summary.
    pub fn run_native_integration_preview(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        self.run_native_integration_probe(limits)
    }

    /// Preferred API name for a bounded native-step integration probe.
    pub fn run_native_integration_probe(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        self.run_native_integration_internal(limits, true)
    }

    pub fn run_native_integration_preview_for_family(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
        family: Lsode2MethodFamily,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        self.run_native_integration_probe_for_family(limits, family)
    }

    /// Preferred API name for a bounded native-step integration probe with an
    /// explicit method family override.
    pub fn run_native_integration_probe_for_family(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
        family: Lsode2MethodFamily,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        let method = match family {
            Lsode2MethodFamily::Bdf => Lsode2NativeStepMethod::BdfLike,
            Lsode2MethodFamily::Adams => Lsode2NativeStepMethod::AdamsLike,
        };
        self.run_native_integration_internal_with_method(limits, true, Some(method))
    }

    fn run_native_integration_internal(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
        record_decision: bool,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        self.run_native_integration_internal_with_method(limits, record_decision, None)
    }

    fn run_native_integration_internal_with_method(
        &mut self,
        limits: Lsode2NativeIntegrationLimits,
        record_decision: bool,
        method_override: Option<Lsode2NativeStepMethod>,
    ) -> Result<Option<Lsode2NativeIntegrationSummary>, Lsode2Error> {
        let lsoda_probe_flow_enabled = self.lsoda_probe_flow_enabled();
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
        let outcome = if lsoda_probe_flow_enabled
            && record_decision
            && method_override.is_none()
            && self.config.controller.mode == Lsode2ControllerMode::AutomaticAdamsBdf
        {
            let config = &self.config;
            let algorithm = &mut self.algorithm;
            let native_statistics = &mut self.native_statistics;
            let last_native_switch_telemetry = &mut self.last_native_switch_telemetry;
            let switch_telemetry_hints = self.switch_telemetry_hints;
            run_native_integration_for_method_with_policy(
                config,
                limits,
                method,
                |report, current| {
                    let family = method_family_from_native_step_method(current);
                    native_statistics
                        .record_native_method_cost_sample(family, report.iterations.max(1) as f64);
                    *last_native_switch_telemetry = Some(report.telemetry);
                    if report.accepted() {
                        algorithm.record_accepted_steps_for_switch_probe(1);
                    }
                    let mut telemetry = Lsode2SwitchTelemetry::default()
                        .with_accepted_steps(native_statistics.native_step_accepts + 1)
                        .with_rejected_steps(
                            native_statistics.native_step_rejects_error_test
                                + native_statistics.native_step_rejects_nonlinear,
                        )
                        .with_convergence_failures(native_statistics.native_step_rejects_nonlinear)
                        .with_adams_cost_samples(native_statistics.native_adams_cost_samples)
                        .with_bdf_cost_samples(native_statistics.native_bdf_cost_samples);
                    if let Some(cost) = native_statistics.adams_step_cost_estimate() {
                        telemetry = telemetry.with_adams_step_cost_estimate(cost);
                    }
                    if let Some(cost) = native_statistics.bdf_step_cost_estimate() {
                        telemetry = telemetry.with_bdf_step_cost_estimate(cost);
                    }
                    if telemetry
                        .stiffness_ratio
                        .is_none_or(|ratio| !ratio.is_finite() || ratio <= 0.0)
                    {
                        telemetry.stiffness_ratio = report.telemetry.stiffness_ratio;
                    }
                    if telemetry.adams_step_size_cap_estimate.is_none() {
                        telemetry.adams_step_size_cap_estimate =
                            report.telemetry.adams_step_size_cap_estimate;
                    }
                    if telemetry.bdf_step_size_cap_estimate.is_none() {
                        telemetry.bdf_step_size_cap_estimate =
                            report.telemetry.bdf_step_size_cap_estimate;
                    }
                    telemetry = switch_telemetry_hints.apply_to(telemetry);
                    let decision = algorithm.switch_decision_stateful(telemetry);
                    algorithm.record_switch_decision_at(decision, report.accepted_t());
                    native_statistics.record_algorithm_decision(&decision);
                    decision.executed_family().map(native_method_for_family)
                },
            )
            .map_err(Lsode2Error::from)?
        } else {
            match method {
                Lsode2NativeStepMethod::BdfLike => {
                    run_native_integration(&self.config, limits).map_err(Lsode2Error::from)?
                }
                Lsode2NativeStepMethod::AdamsLike => {
                    run_native_integration_for_method(&self.config, limits, method)
                        .map_err(Lsode2Error::from)?
                }
            }
        };
        if let Some(summary) = &outcome.summary {
            let family = method_family_from_native_step_method(method);
            let cost =
                integration_method_cost_estimate(&outcome.statistics, summary.accepted_steps);
            if lsoda_probe_flow_enabled {
                self.native_statistics
                    .record_native_method_cost_sample(family, cost);
                for report in &summary.attempt_reports {
                    self.absorb_native_switch_telemetry(report.telemetry);
                }
            }
        }
        merge_native_statistics(&mut self.native_statistics, &outcome.statistics);
        if lsoda_probe_flow_enabled {
            if let Some(summary) = &outcome.summary {
                self.algorithm
                    .record_accepted_steps_for_switch_probe(summary.accepted_steps);
            }
        }
        Ok(outcome.summary)
    }

    fn lsoda_probe_flow_enabled(&self) -> bool {
        self.config.controller.mode == Lsode2ControllerMode::AutomaticAdamsBdf
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
        telemetry = telemetry
            .with_adams_cost_samples(self.native_statistics.native_adams_cost_samples)
            .with_bdf_cost_samples(self.native_statistics.native_bdf_cost_samples);
        if let Some(native) = self.last_native_switch_telemetry {
            if telemetry.stiffness_ratio.is_none() {
                telemetry.stiffness_ratio = native.stiffness_ratio;
            }
            if telemetry.adams_step_size_cap_estimate.is_none() {
                telemetry.adams_step_size_cap_estimate = native.adams_step_size_cap_estimate;
            }
            if telemetry.bdf_step_size_cap_estimate.is_none() {
                telemetry.bdf_step_size_cap_estimate = native.bdf_step_size_cap_estimate;
            }
        }
        self.switch_telemetry_hints.apply_to(telemetry)
    }

    fn should_run_switch_probe_before_full_native_solve(
        &self,
        decision: Lsode2SwitchDecision,
    ) -> bool {
        self.config.controller.mode == Lsode2ControllerMode::AutomaticAdamsBdf
            && (decision.reason == Lsode2SwitchReason::SwitchProbeWarmup
                || decision.reason == Lsode2SwitchReason::InsufficientCostEvidence)
            && !decision.uses_fallback
            && decision.executed_family().is_some()
    }

    fn probe_method_for_switch_choreography(
        &self,
        decision: &Lsode2SwitchDecision,
    ) -> Lsode2NativeStepMethod {
        if decision.reason != Lsode2SwitchReason::InsufficientCostEvidence {
            return native_method_for_decision(decision);
        }

        let min_samples = self.config.controller.min_cost_samples_for_switch.max(1);
        let adams_samples = self.native_statistics.native_adams_cost_samples;
        let bdf_samples = self.native_statistics.native_bdf_cost_samples;

        // Strictly targeted evidence collection:
        // when cost-evidence gate is the only blocker, probe the family that
        // is currently missing samples, instead of always running symmetric
        // cross-family probes.
        if adams_samples < min_samples && bdf_samples >= min_samples {
            return Lsode2NativeStepMethod::AdamsLike;
        }
        if bdf_samples < min_samples && adams_samples >= min_samples {
            return Lsode2NativeStepMethod::BdfLike;
        }

        native_method_for_decision(decision)
    }

    fn probe_limits_for_switch_choreography(
        &self,
        decision: &Lsode2SwitchDecision,
    ) -> Lsode2NativeIntegrationLimits {
        if decision.reason == Lsode2SwitchReason::SwitchProbeWarmup {
            // ICOUNT-like warmup gate: consume enough accepted steps to open
            // probe window deterministically in one bounded integration pass.
            let probe_accepted_steps = self
                .config
                .controller
                .method_switch_probe_steps
                .saturating_add(1)
                .max(1);
            let probe_attempt_steps = probe_accepted_steps.saturating_mul(4).max(1);
            return Lsode2NativeIntegrationLimits::new(probe_attempt_steps, probe_accepted_steps);
        }

        // Cost-evidence gate: collect one additional accepted sample from the
        // targeted family.
        Lsode2NativeIntegrationLimits::new(8, 1)
    }

    fn run_switch_probe_before_full_native_solve(
        &mut self,
        decision: Lsode2SwitchDecision,
    ) -> Result<(), Lsode2Error> {
        // LSODA-first choreography:
        // - warmup gate: run one bounded probe to open the ICOUNT-like window;
        // - cost-evidence gate: run one targeted probe for the missing family.
        let limits = self.probe_limits_for_switch_choreography(&decision);
        let probe_method = self.probe_method_for_switch_choreography(&decision);
        let _probe_summary =
            self.run_native_integration_internal_with_method(limits, false, Some(probe_method))?;
        Ok(())
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
            linear_solver_backend: self.resolved_plan.linear_solver.label(),
            linear_solver_reason: self.resolved_plan.linear_solver_reason,
            resolved_source: self.resolved_plan.source.label(),
            resolved_structure: self.resolved_plan.structure.label(),
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
            native_termination_kind: self
                .native_integration_solve
                .as_ref()
                .map(|summary| summary.termination_kind.label()),
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
        if let Some(summary) = outcome.summary.as_ref() {
            if self.lsoda_probe_flow_enabled() {
                self.absorb_native_switch_telemetry(summary.telemetry);
            }
        }
        Ok(outcome.summary)
    }

    fn absorb_native_switch_telemetry(&mut self, telemetry: Lsode2SwitchTelemetry) {
        self.last_native_switch_telemetry = Some(telemetry);
        self.switch_telemetry_hints.absorb(telemetry);
    }
}

fn native_method_for_decision(decision: &Lsode2SwitchDecision) -> Lsode2NativeStepMethod {
    // Mirror controller choreography: execute the family that is actually
    // runnable on the current path (fallback-aware), not merely preferred.
    // This keeps native stepping aligned with controller fallback semantics.
    let family = decision
        .executed_family()
        .unwrap_or(decision.preferred_family);
    native_method_for_family(family)
}

fn native_method_for_family(family: Lsode2MethodFamily) -> Lsode2NativeStepMethod {
    match family {
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

fn status_for_native_termination_kind(kind: Lsode2NativeTerminationKind) -> &'static str {
    match kind {
        Lsode2NativeTerminationKind::ReachedTBound
        | Lsode2NativeTerminationKind::ReachedStopCondition
        | Lsode2NativeTerminationKind::ReachedTBoundAndStopCondition => "finished_native_faithful",
        Lsode2NativeTerminationKind::LimitsExhausted => "finished_native_faithful_partial",
    }
}

fn validate_supported_backend_routes(config: &Lsode2ProblemConfig) -> Result<(), Lsode2Error> {
    match config.backend.jacobian_backend {
        Lsode2JacobianBackend::SymbolicGenerated => {}
        Lsode2JacobianBackend::AnalyticClosure => {
            if config.analytical_callbacks.is_none() {
                return Err(Lsode2Error::InvalidConfig(
                    "LSODE2 analytical route requires residual and jacobian callbacks".to_string(),
                ));
            }
            if !matches!(
                config.native_execution,
                Lsode2NativeExecutionConfig::NativeSolve { .. }
            ) {
                return Err(Lsode2Error::UnsupportedBackend(
                    "LSODE2 analytical route is currently native-only; set native_solve execution mode",
                ));
            }
        }
        Lsode2JacobianBackend::FiniteDifference => {
            if config.analytical_callbacks.is_none() {
                return Err(Lsode2Error::InvalidConfig(
                    "LSODE2 finite-difference Jacobian backend requires analytical residual callback route".to_string(),
                ));
            }
            if !matches!(
                config.native_execution,
                Lsode2NativeExecutionConfig::NativeSolve { .. }
            ) {
                return Err(Lsode2Error::UnsupportedBackend(
                    "LSODE2 finite-difference Jacobian backend is currently native-only; set native_solve execution mode",
                ));
            }
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

fn validate_stop_condition_config(config: &Lsode2ProblemConfig) -> Result<(), Lsode2Error> {
    for condition in &config.stop_conditions {
        if !condition.target.is_finite() {
            return Err(Lsode2Error::InvalidConfig(format!(
                "LSODE2 stop condition target for `{}` must be finite",
                condition.variable
            )));
        }
        if matches!(condition.comparator, Lsode2StopComparator::AbsDistance)
            && !condition.tolerance.is_finite()
        {
            return Err(Lsode2Error::InvalidConfig(format!(
                "LSODE2 stop condition tolerance for `{}` must be finite",
                condition.variable
            )));
        }
        if !config.values.iter().any(|name| name == &condition.variable) {
            return Err(Lsode2Error::InvalidConfig(format!(
                "LSODE2 stop condition references unknown variable `{}`",
                condition.variable
            )));
        }
    }
    Ok(())
}

fn bridge_compatible_stop_conditions(config: &Lsode2ProblemConfig) -> Option<HashMap<String, f64>> {
    if config.stop_conditions.is_empty() {
        return None;
    }

    // Bridge BDF path supports scalar stop map semantics only.
    // We forward all configured conditions as target values so bridge mode can
    // still stop early when thresholds are approached.
    let mut conditions = HashMap::with_capacity(config.stop_conditions.len());
    for stop in &config.stop_conditions {
        conditions.insert(stop.variable.clone(), stop.target);
    }
    Some(conditions)
}

fn controller_execution_capabilities(
    config: &Lsode2ProblemConfig,
) -> Lsode2ControllerExecutionCapabilities {
    let adams_engine_available = matches!(
        config.native_execution,
        Lsode2NativeExecutionConfig::ProbeBeforeBridge { .. }
            | Lsode2NativeExecutionConfig::NativeSolve { .. }
    ) && matches!(
        config.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::Dense
            | Lsode2LinearSolverBackend::SparseFaer
            | Lsode2LinearSolverBackend::BandedFaithful
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
    let equation_parameter_values = config.equation_parameter_values.clone();
    let generated_backend = config.backend.generated_backend.clone();
    let symbolic_assembly_backend = match config.residual_jacobian_source {
        Lsode2ResidualJacobianSource::Symbolic { assembly, .. } => assembly,
        Lsode2ResidualJacobianSource::Analytical => Lsode2SymbolicAssemblyBackend::ExprLegacy,
    };
    let use_sparse_aot_jacobian = matches!(
        config.residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            execution: Lsode2SymbolicExecutionMode::Aot { .. },
            ..
        }
    );
    inner.set_bdf_native_jacobian_factory(move |parameter_values_handle| {
        if use_sparse_aot_jacobian {
            compile_native_sparse_aot_jacobian_with_parameter_handle(
                &equations,
                &variables,
                time_arg.as_str(),
                equation_parameters.as_deref(),
                equation_parameter_values.clone(),
                parameter_values_handle,
                storage,
                generated_backend.clone(),
                match symbolic_assembly_backend {
                    Lsode2SymbolicAssemblyBackend::ExprLegacy => {
                        IvpSymbolicAssemblyBackend::ExprLegacy
                    }
                    Lsode2SymbolicAssemblyBackend::AtomView => IvpSymbolicAssemblyBackend::AtomView,
                },
            )
            .expect("LSODE2 AOT sparse Jacobian backend should prepare compiled callbacks")
        } else {
            compile_native_symbolic_jacobian_with_parameter_handle(
                &equations,
                &variables,
                time_arg.as_str(),
                equation_parameters.as_deref(),
                parameter_values_handle,
                storage,
            )
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::LSODE2::{
        Lsode2ControllerConfig, Lsode2MethodFamily, Lsode2SwitchReason,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn native_method_selector_uses_executable_family_when_fallback_is_active() {
        let decision = Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Adams,
            executable_family: Some(Lsode2MethodFamily::Bdf),
            uses_fallback: true,
            reason: Lsode2SwitchReason::AdamsEngineUnavailable,
            message: "test fallback",
        };

        let method = native_method_for_decision(&decision);
        assert_eq!(method, Lsode2NativeStepMethod::BdfLike);
    }

    #[test]
    fn native_method_selector_uses_preferred_family_when_executable_is_not_set() {
        let decision = Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Bdf,
            executable_family: None,
            uses_fallback: false,
            reason: Lsode2SwitchReason::FixedController,
            message: "test fixed controller",
        };

        let method = native_method_for_decision(&decision);
        assert_eq!(method, Lsode2NativeStepMethod::BdfLike);
    }

    #[test]
    fn switch_telemetry_hints_accumulate_cross_family_native_signals() {
        let mut hints = Lsode2SwitchTelemetryHints::default();
        hints.absorb(
            Lsode2SwitchTelemetry::default()
                .with_adams_step_size_cap_estimate(2.5)
                .with_stiffness_ratio(0.7),
        );
        hints.absorb(Lsode2SwitchTelemetry::default().with_bdf_step_size_cap_estimate(1.1));

        let merged = hints.apply_to(
            Lsode2SwitchTelemetry::default()
                .with_accepted_steps(10)
                .with_rejected_steps(1),
        );
        assert_eq!(merged.adams_step_size_cap_estimate, Some(2.5));
        assert_eq!(merged.bdf_step_size_cap_estimate, Some(1.1));
        assert_eq!(merged.stiffness_ratio, Some(0.7));
        assert_eq!(merged.accepted_steps, 10);
        assert_eq!(merged.rejected_steps, 1);
    }

    #[test]
    fn switch_telemetry_hints_drive_dstoda_hold_reason_in_automatic_mode() {
        let mut hints = Lsode2SwitchTelemetryHints::default();
        hints.absorb(
            Lsode2SwitchTelemetry::default()
                .with_adams_step_size_cap_estimate(1.2)
                .with_stiffness_ratio(0.5),
        );
        let telemetry = hints.apply_to(
            Lsode2SwitchTelemetry::default()
                .with_accepted_steps(64)
                .with_rejected_steps(0),
        );

        let decision = Lsode2ControllerConfig::automatic_adams_bdf()
            .switch_decision_with_probe_gate_and_capabilities(
                telemetry,
                Some(true),
                Lsode2ControllerExecutionCapabilities {
                    adams_engine_available: true,
                },
            );

        assert_eq!(decision.preferred_family, Lsode2MethodFamily::Adams);
        assert_eq!(decision.reason, Lsode2SwitchReason::SwitchAdvantageNotMet);
        assert!(decision.message.contains("step-advantage"));
    }

    #[test]
    fn switch_telemetry_hints_keep_stronger_stiffness_and_cap_signals() {
        let mut hints = Lsode2SwitchTelemetryHints::default();
        hints.absorb(
            Lsode2SwitchTelemetry::default()
                .with_stiffness_ratio(0.25)
                .with_adams_step_size_cap_estimate(2.0)
                .with_bdf_step_size_cap_estimate(0.6),
        );
        hints.absorb(
            Lsode2SwitchTelemetry::default()
                .with_stiffness_ratio(0.1)
                .with_adams_step_size_cap_estimate(1.2)
                .with_bdf_step_size_cap_estimate(0.4),
        );

        let merged = hints.apply_to(Lsode2SwitchTelemetry::default());
        assert_eq!(merged.stiffness_ratio, Some(0.25));
        assert_eq!(merged.adams_step_size_cap_estimate, Some(2.0));
        assert_eq!(merged.bdf_step_size_cap_estimate, Some(0.6));
    }

    #[test]
    fn warmup_preprobe_collects_targeted_probe_evidence_without_forced_cross_family_sampling() {
        let mut solver = Lsode2Solver::new(
            Lsode2ProblemConfig::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".to_string()],
                "t".to_string(),
                0.0,
                DVector::from_vec(vec![1.0]),
                1.0,
                0.02,
                1.0e-6,
                1.0e-8,
            )
            .with_native_sparse_faer_backend()
            .with_controller(Lsode2ControllerConfig::automatic_adams_bdf())
            .with_faithful_bdf_solve(512, 256),
        )
        .expect("automatic native config should build");

        // Force current-family context to BDF to exercise the BDF->Adams
        // probe direction in warmup choreography.
        solver
            .algorithm
            .record_switch_decision(Lsode2SwitchDecision {
                preferred_family: Lsode2MethodFamily::Bdf,
                executable_family: Some(Lsode2MethodFamily::Bdf),
                uses_fallback: false,
                reason: Lsode2SwitchReason::StiffnessSuspected,
                message: "test setup",
            });

        assert_eq!(solver.native_statistics.native_adams_cost_samples, 0);
        assert_eq!(solver.native_statistics.native_bdf_cost_samples, 0);

        solver
            .run_switch_probe_before_full_native_solve(Lsode2SwitchDecision {
                preferred_family: Lsode2MethodFamily::Bdf,
                executable_family: Some(Lsode2MethodFamily::Bdf),
                uses_fallback: false,
                reason: Lsode2SwitchReason::SwitchProbeWarmup,
                message: "test warmup",
            })
            .expect("warmup probe should run for the active family");

        assert!(
            solver.native_statistics.native_bdf_cost_samples > 0,
            "warmup pre-probe should record BDF cost evidence"
        );
        assert_eq!(
            solver.native_statistics.native_adams_cost_samples, 0,
            "warmup pre-probe should not force cross-family Adams sampling"
        );
        assert!(
            solver.algorithm.switch_state().switch_probe_ready(),
            "warmup pre-probe should open ICOUNT-like switch gate"
        );
    }

    #[test]
    fn insufficient_cost_preprobe_targets_missing_family_without_forced_symmetric_probe() {
        let mut solver = Lsode2Solver::new(
            Lsode2ProblemConfig::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".to_string()],
                "t".to_string(),
                0.0,
                DVector::from_vec(vec![1.0]),
                1.0,
                0.02,
                1.0e-6,
                1.0e-8,
            )
            .with_native_sparse_faer_backend()
            .with_controller(
                Lsode2ControllerConfig::automatic_adams_bdf().with_min_cost_samples_for_switch(3),
            )
            .with_faithful_bdf_solve(512, 256),
        )
        .expect("automatic native config should build");

        solver.native_statistics.native_bdf_cost_samples = 3;
        solver.native_statistics.native_bdf_step_cost_accum = 3.0;
        solver.native_statistics.native_adams_cost_samples = 0;
        solver.native_statistics.native_adams_step_cost_accum = 0.0;

        solver
            .run_switch_probe_before_full_native_solve(Lsode2SwitchDecision {
                preferred_family: Lsode2MethodFamily::Bdf,
                executable_family: Some(Lsode2MethodFamily::Bdf),
                uses_fallback: false,
                reason: Lsode2SwitchReason::InsufficientCostEvidence,
                message: "test insufficient cost",
            })
            .expect("insufficient-cost probe should run on missing family");

        assert!(
            solver.native_statistics.native_adams_cost_samples > 0,
            "insufficient-cost probe should collect missing Adams evidence"
        );
        assert!(
            solver.native_statistics.native_bdf_cost_samples >= 3,
            "insufficient-cost probe should not drop existing BDF evidence"
        );
    }
}
