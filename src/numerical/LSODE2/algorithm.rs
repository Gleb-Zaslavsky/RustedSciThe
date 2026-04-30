use super::adams_engine::{
    Lsode2AdamsDcfodeError, Lsode2AdamsDcfodeTables, Lsode2AdamsOrderCoefficients,
};

pub use super::method_switch::{
    LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS, LSODE2_DEFAULT_STIFFNESS_RATIO_THRESHOLD,
    LSODE2_MAX_ADAMS_ORDER, LSODE2_MAX_BDF_ORDER, Lsode2ControllerMode, Lsode2MethodFamily,
    Lsode2MethodSwitchPolicy, Lsode2MethodSwitchState, Lsode2SwitchDecision, Lsode2SwitchReason,
    Lsode2SwitchTelemetry,
};

/// Algorithm-level configuration independent of symbolic/linear backends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2ControllerConfig {
    pub mode: Lsode2ControllerMode,
    pub max_adams_order: usize,
    pub max_bdf_order: usize,
    pub stiffness_ratio_threshold: f64,
    pub method_switch_probe_steps: usize,
}

/// Current executable status for one controller configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lsode2ControllerExecutionPlan {
    pub executable_family: Option<Lsode2MethodFamily>,
    pub uses_fallback: bool,
    pub requires_adams_engine: bool,
    pub message: &'static str,
}

impl Lsode2ControllerExecutionPlan {
    pub fn is_executable_now(self) -> bool {
        self.executable_family.is_some()
    }
}

/// Runtime capabilities that affect which controller families are executable.
///
/// Adams/BDF policy is computed independently from backend details, but the
/// executable family still depends on whether an Adams step engine is
/// available in the active runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Lsode2ControllerExecutionCapabilities {
    pub adams_engine_available: bool,
}

impl Default for Lsode2ControllerConfig {
    fn default() -> Self {
        Self::bdf_only()
    }
}

impl Lsode2ControllerConfig {
    pub fn adams_only() -> Self {
        Self {
            mode: Lsode2ControllerMode::AdamsOnly,
            max_adams_order: LSODE2_MAX_ADAMS_ORDER,
            max_bdf_order: LSODE2_MAX_BDF_ORDER,
            stiffness_ratio_threshold: LSODE2_DEFAULT_STIFFNESS_RATIO_THRESHOLD,
            method_switch_probe_steps: LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS,
        }
    }

    pub fn bdf_only() -> Self {
        Self {
            mode: Lsode2ControllerMode::BdfOnly,
            max_adams_order: LSODE2_MAX_ADAMS_ORDER,
            max_bdf_order: LSODE2_MAX_BDF_ORDER,
            stiffness_ratio_threshold: LSODE2_DEFAULT_STIFFNESS_RATIO_THRESHOLD,
            method_switch_probe_steps: LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS,
        }
    }

    pub fn automatic_adams_bdf() -> Self {
        Self {
            mode: Lsode2ControllerMode::AutomaticAdamsBdf,
            max_adams_order: LSODE2_MAX_ADAMS_ORDER,
            max_bdf_order: LSODE2_MAX_BDF_ORDER,
            stiffness_ratio_threshold: LSODE2_DEFAULT_STIFFNESS_RATIO_THRESHOLD,
            method_switch_probe_steps: LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS,
        }
    }

    pub fn with_max_adams_order(mut self, order: usize) -> Self {
        self.max_adams_order = order;
        self
    }

    pub fn with_max_bdf_order(mut self, order: usize) -> Self {
        self.max_bdf_order = order;
        self
    }

    pub fn with_stiffness_ratio_threshold(mut self, threshold: f64) -> Self {
        self.stiffness_ratio_threshold = threshold;
        self
    }

    pub fn with_method_switch_probe_steps(mut self, steps: usize) -> Self {
        self.method_switch_probe_steps = steps;
        self
    }

    pub fn validate(self) -> Result<(), String> {
        if self.max_adams_order == 0 || self.max_adams_order > LSODE2_MAX_ADAMS_ORDER {
            return Err(format!(
                "LSODE2 Adams max order must be in 1..={LSODE2_MAX_ADAMS_ORDER}, got {}",
                self.max_adams_order
            ));
        }
        if self.max_bdf_order == 0 || self.max_bdf_order > LSODE2_MAX_BDF_ORDER {
            return Err(format!(
                "LSODE2 BDF max order must be in 1..={LSODE2_MAX_BDF_ORDER}, got {}",
                self.max_bdf_order
            ));
        }
        if !self.stiffness_ratio_threshold.is_finite() || self.stiffness_ratio_threshold <= 0.0 {
            return Err(format!(
                "LSODE2 stiffness ratio threshold must be finite and positive, got {}",
                self.stiffness_ratio_threshold
            ));
        }
        if self.method_switch_probe_steps == 0 {
            return Err(
                "LSODE2 method-switch probe steps must be at least 1 (ODEPACK ICOUNT-style gate)"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn execution_plan(self) -> Lsode2ControllerExecutionPlan {
        self.execution_plan_with_capabilities(Lsode2ControllerExecutionCapabilities::default())
    }

    pub fn execution_plan_with_capabilities(
        self,
        capabilities: Lsode2ControllerExecutionCapabilities,
    ) -> Lsode2ControllerExecutionPlan {
        match self.mode {
            Lsode2ControllerMode::AdamsOnly => {
                if capabilities.adams_engine_available {
                    Lsode2ControllerExecutionPlan {
                        executable_family: Some(Lsode2MethodFamily::Adams),
                        uses_fallback: false,
                        requires_adams_engine: true,
                        message: "Adams-only controller is active",
                    }
                } else {
                    Lsode2ControllerExecutionPlan {
                        executable_family: None,
                        uses_fallback: false,
                        requires_adams_engine: true,
                        message: "Adams-only controller requires native Adams execution support",
                    }
                }
            }
            Lsode2ControllerMode::BdfOnly => Lsode2ControllerExecutionPlan {
                executable_family: Some(Lsode2MethodFamily::Bdf),
                uses_fallback: false,
                requires_adams_engine: false,
                message: "BDF-only controller is active",
            },
            Lsode2ControllerMode::AutomaticAdamsBdf => {
                if capabilities.adams_engine_available {
                    Lsode2ControllerExecutionPlan {
                        executable_family: Some(Lsode2MethodFamily::Bdf),
                        uses_fallback: false,
                        requires_adams_engine: true,
                        message: "automatic Adams/BDF controller is active",
                    }
                } else {
                    Lsode2ControllerExecutionPlan {
                        executable_family: Some(Lsode2MethodFamily::Bdf),
                        uses_fallback: true,
                        requires_adams_engine: true,
                        message: "automatic Adams/BDF controller uses BDF fallback because native Adams execution is unavailable",
                    }
                }
            }
        }
    }

    pub fn preferred_family(self, telemetry: Lsode2SwitchTelemetry) -> Lsode2MethodFamily {
        self.switch_decision(telemetry).preferred_family
    }

    pub fn switch_decision(self, telemetry: Lsode2SwitchTelemetry) -> Lsode2SwitchDecision {
        self.switch_decision_with_probe_gate(telemetry, None)
    }

    pub fn switch_decision_with_probe_gate(
        self,
        telemetry: Lsode2SwitchTelemetry,
        probe_gate: Option<bool>,
    ) -> Lsode2SwitchDecision {
        self.switch_decision_with_probe_gate_and_capabilities(
            telemetry,
            probe_gate,
            Lsode2ControllerExecutionCapabilities::default(),
        )
    }

    pub fn switch_decision_with_probe_gate_and_capabilities(
        self,
        telemetry: Lsode2SwitchTelemetry,
        probe_gate: Option<bool>,
        capabilities: Lsode2ControllerExecutionCapabilities,
    ) -> Lsode2SwitchDecision {
        let plan = self.execution_plan_with_capabilities(capabilities);
        let policy = Lsode2MethodSwitchPolicy::default()
            .with_stiffness_ratio_threshold(self.stiffness_ratio_threshold)
            .with_minimum_accepted_steps_for_nonstiff_probe(self.method_switch_probe_steps);
        let (preferred_family, base_reason) =
            policy.preferred_family_and_reason_with_probe_gate(self.mode, telemetry, probe_gate);

        let (executable_family, reason, message) = match (self.mode, preferred_family) {
            (Lsode2ControllerMode::AdamsOnly, _) => {
                if capabilities.adams_engine_available {
                    (
                        Some(Lsode2MethodFamily::Adams),
                        Lsode2SwitchReason::FixedController,
                        plan.message,
                    )
                } else {
                    (None, Lsode2SwitchReason::FixedController, plan.message)
                }
            }
            (_, Lsode2MethodFamily::Bdf) => {
                let reason = match base_reason {
                    Lsode2SwitchReason::ConvergenceTrouble => {
                        Lsode2SwitchReason::ConvergenceTrouble
                    }
                    Lsode2SwitchReason::StiffnessSuspected => {
                        Lsode2SwitchReason::StiffnessSuspected
                    }
                    Lsode2SwitchReason::CostPreferenceBdf => Lsode2SwitchReason::CostPreferenceBdf,
                    Lsode2SwitchReason::SwitchProbeWarmup => Lsode2SwitchReason::SwitchProbeWarmup,
                    _ => Lsode2SwitchReason::FixedController,
                };
                let message = match reason {
                    Lsode2SwitchReason::ConvergenceTrouble => {
                        "automatic policy prefers BDF because recent step telemetry reports convergence trouble"
                    }
                    Lsode2SwitchReason::StiffnessSuspected => {
                        "automatic policy prefers BDF because stiffness telemetry exceeds the configured threshold"
                    }
                    Lsode2SwitchReason::CostPreferenceBdf => {
                        "automatic policy prefers BDF because cost telemetry estimates BDF step cost lower than Adams"
                    }
                    Lsode2SwitchReason::SwitchProbeWarmup => {
                        "automatic policy keeps BDF during warmup before non-stiff switch probing"
                    }
                    _ => plan.message,
                };
                (Some(Lsode2MethodFamily::Bdf), reason, message)
            }
            (Lsode2ControllerMode::AutomaticAdamsBdf, Lsode2MethodFamily::Adams) => {
                if capabilities.adams_engine_available {
                    let reason = match base_reason {
                        Lsode2SwitchReason::CostPreferenceAdams => {
                            Lsode2SwitchReason::CostPreferenceAdams
                        }
                        _ => Lsode2SwitchReason::NonstiffPreference,
                    };
                    let message = match reason {
                        Lsode2SwitchReason::CostPreferenceAdams => {
                            "automatic policy prefers Adams because cost telemetry estimates Adams step cost lower than BDF"
                        }
                        _ => "automatic policy prefers Adams for non-stiff telemetry window",
                    };
                    (Some(Lsode2MethodFamily::Adams), reason, message)
                } else {
                    (
                        plan.executable_family,
                        Lsode2SwitchReason::AdamsEngineUnavailable,
                        "automatic policy prefers Adams, but native Adams execution is unavailable on this path; using BDF fallback",
                    )
                }
            }
            (Lsode2ControllerMode::BdfOnly, Lsode2MethodFamily::Adams) => (
                Some(Lsode2MethodFamily::Bdf),
                Lsode2SwitchReason::FixedController,
                plan.message,
            ),
        };
        let uses_fallback = executable_family != Some(preferred_family);

        Lsode2SwitchDecision {
            preferred_family,
            executable_family,
            uses_fallback,
            reason,
            message,
        }
    }
}

/// User-facing algorithm snapshot included in solve summaries.
#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2AlgorithmSnapshot {
    pub controller_mode: &'static str,
    pub active_family: &'static str,
    pub preferred_family: &'static str,
    pub executed_family: Option<&'static str>,
    pub switch_reason: &'static str,
    pub switch_uses_fallback: bool,
    pub method_switching_enabled: bool,
    pub max_adams_order: usize,
    pub max_bdf_order: usize,
    pub stiffness_ratio_threshold: f64,
    pub method_switch_probe_steps: usize,
    pub switch_probe_countdown: isize,
    pub switch_probe_ready: bool,
    pub bdf_current_order: Option<usize>,
    pub bdf_max_order_cap: Option<usize>,
    pub bdf_equal_step_count: Option<usize>,
    pub note: &'static str,
}

/// Minimal controller state used by the current BDF-backed facade.
#[derive(Debug, Clone)]
pub struct Lsode2AlgorithmController {
    config: Lsode2ControllerConfig,
    execution_capabilities: Lsode2ControllerExecutionCapabilities,
    active_family: Lsode2MethodFamily,
    switch_state: Lsode2MethodSwitchState,
    last_decision: Option<Lsode2SwitchDecision>,
    adams_tables: Lsode2AdamsDcfodeTables,
}

impl Lsode2AlgorithmController {
    pub fn new(config: Lsode2ControllerConfig) -> Self {
        Self::new_with_capabilities(config, Lsode2ControllerExecutionCapabilities::default())
    }

    pub fn new_with_capabilities(
        config: Lsode2ControllerConfig,
        execution_capabilities: Lsode2ControllerExecutionCapabilities,
    ) -> Self {
        let active_family = config
            .execution_plan_with_capabilities(execution_capabilities)
            .executable_family
            .unwrap_or(Lsode2MethodFamily::Adams);
        Self {
            config,
            execution_capabilities,
            active_family,
            switch_state: {
                let mut state = Lsode2MethodSwitchState {
                    mused: active_family,
                    mcur: active_family,
                    ..Lsode2MethodSwitchState::default()
                };
                state.set_switch_probe_initial_countdown(config.method_switch_probe_steps);
                state
            },
            last_decision: None,
            adams_tables: Lsode2AdamsDcfodeTables::generate(),
        }
    }

    pub fn config(&self) -> Lsode2ControllerConfig {
        self.config
    }

    pub fn active_family(&self) -> Lsode2MethodFamily {
        self.active_family
    }

    pub fn execution_plan(&self) -> Lsode2ControllerExecutionPlan {
        self.config
            .execution_plan_with_capabilities(self.execution_capabilities)
    }

    pub fn switch_decision(&self, telemetry: Lsode2SwitchTelemetry) -> Lsode2SwitchDecision {
        self.config
            .switch_decision_with_probe_gate_and_capabilities(
                telemetry,
                None,
                self.execution_capabilities,
            )
    }

    pub fn switch_decision_stateful(
        &mut self,
        telemetry: Lsode2SwitchTelemetry,
    ) -> Lsode2SwitchDecision {
        let probe_gate = if self.config.mode == Lsode2ControllerMode::AutomaticAdamsBdf {
            Some(self.switch_state.consume_switch_probe_gate())
        } else {
            None
        };
        self.config
            .switch_decision_with_probe_gate_and_capabilities(
                telemetry,
                probe_gate,
                self.execution_capabilities,
            )
    }

    pub fn record_switch_decision(&mut self, decision: Lsode2SwitchDecision) {
        self.switch_state.record_decision(decision);
        self.active_family = self.switch_state.mcur;
        self.last_decision = Some(decision);
    }

    pub fn record_accepted_steps_for_switch_probe(&mut self, accepted_steps: usize) {
        if self.config.mode != Lsode2ControllerMode::AutomaticAdamsBdf {
            return;
        }
        self.switch_state
            .record_accepted_steps_for_probe(accepted_steps);
    }

    pub fn switch_state(&self) -> Lsode2MethodSwitchState {
        self.switch_state
    }

    pub fn adams_tables(&self) -> &Lsode2AdamsDcfodeTables {
        &self.adams_tables
    }

    pub fn adams_order_coefficients(
        &self,
        order: usize,
    ) -> Result<Lsode2AdamsOrderCoefficients, Lsode2AdamsDcfodeError> {
        self.adams_tables.order(order)
    }

    pub fn snapshot(&self) -> Lsode2AlgorithmSnapshot {
        self.snapshot_with_bdf_runtime(None, None, None)
    }

    pub fn snapshot_with_bdf_runtime(
        &self,
        bdf_current_order: Option<usize>,
        bdf_max_order_cap: Option<usize>,
        bdf_equal_step_count: Option<usize>,
    ) -> Lsode2AlgorithmSnapshot {
        let decision = self.last_decision.unwrap_or_else(|| {
            self.config
                .switch_decision_with_probe_gate_and_capabilities(
                    Lsode2SwitchTelemetry::default(),
                    Some(self.switch_state.switch_probe_ready()),
                    self.execution_capabilities,
                )
        });
        let method_switching_enabled = self.config.mode == Lsode2ControllerMode::AutomaticAdamsBdf;

        Lsode2AlgorithmSnapshot {
            controller_mode: self.config.mode.label(),
            active_family: self.active_family.label(),
            preferred_family: decision.preferred_family.label(),
            executed_family: decision.executed_family().map(Lsode2MethodFamily::label),
            switch_reason: decision.reason.label(),
            switch_uses_fallback: decision.uses_fallback,
            method_switching_enabled,
            max_adams_order: self.config.max_adams_order,
            max_bdf_order: self.config.max_bdf_order,
            stiffness_ratio_threshold: self.config.stiffness_ratio_threshold,
            method_switch_probe_steps: self.config.method_switch_probe_steps,
            switch_probe_countdown: self.switch_state.switch_probe_countdown(),
            switch_probe_ready: self.switch_state.switch_probe_ready(),
            bdf_current_order,
            bdf_max_order_cap,
            bdf_equal_step_count,
            note: decision.message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdf_only_controller_snapshot_is_explicit() {
        let controller = Lsode2AlgorithmController::new(Lsode2ControllerConfig::bdf_only());
        let snapshot = controller.snapshot();

        assert_eq!(snapshot.controller_mode, "bdf_only");
        assert_eq!(snapshot.active_family, "bdf");
        assert_eq!(snapshot.preferred_family, "bdf");
        assert_eq!(snapshot.executed_family, Some("bdf"));
        assert_eq!(snapshot.switch_reason, "fixed_controller");
        assert!(!snapshot.switch_uses_fallback);
        assert!(!snapshot.method_switching_enabled);
        assert_eq!(snapshot.max_bdf_order, 5);
        assert_eq!(
            snapshot.stiffness_ratio_threshold,
            LSODE2_DEFAULT_STIFFNESS_RATIO_THRESHOLD
        );
        assert_eq!(snapshot.bdf_current_order, None);
        assert_eq!(snapshot.bdf_max_order_cap, None);
    }

    #[test]
    fn adams_only_controller_snapshot_is_explicit_when_native_adams_is_unavailable() {
        let controller = Lsode2AlgorithmController::new(Lsode2ControllerConfig::adams_only());
        let snapshot = controller.snapshot();

        assert_eq!(snapshot.controller_mode, "adams_only");
        assert_eq!(snapshot.active_family, "adams");
        assert_eq!(snapshot.preferred_family, "adams");
        assert_eq!(snapshot.executed_family, None);
        assert_eq!(snapshot.switch_reason, "fixed_controller");
        assert!(!snapshot.method_switching_enabled);
        assert!(
            snapshot
                .note
                .contains("requires native Adams execution support")
        );
    }

    #[test]
    fn automatic_controller_is_a_declared_future_mode() {
        let controller =
            Lsode2AlgorithmController::new(Lsode2ControllerConfig::automatic_adams_bdf());
        let snapshot = controller.snapshot();
        let plan = controller.execution_plan();

        assert_eq!(snapshot.controller_mode, "automatic_adams_bdf");
        assert_eq!(snapshot.active_family, "bdf");
        assert_eq!(snapshot.preferred_family, "bdf");
        assert_eq!(snapshot.executed_family, Some("bdf"));
        assert_eq!(snapshot.switch_reason, "switch_probe_warmup");
        assert!(!snapshot.switch_uses_fallback);
        assert!(snapshot.method_switching_enabled);
        assert!(snapshot.note.contains("warmup"));
        assert_eq!(plan.executable_family, Some(Lsode2MethodFamily::Bdf));
        assert!(plan.uses_fallback);
        assert!(plan.requires_adams_engine);
    }

    #[test]
    fn controller_snapshot_can_include_bdf_runtime_state() {
        let controller = Lsode2AlgorithmController::new(Lsode2ControllerConfig::bdf_only());
        let snapshot = controller.snapshot_with_bdf_runtime(Some(3), Some(4), Some(2));

        assert_eq!(snapshot.active_family, "bdf");
        assert_eq!(snapshot.bdf_current_order, Some(3));
        assert_eq!(snapshot.bdf_max_order_cap, Some(4));
        assert_eq!(snapshot.bdf_equal_step_count, Some(2));
    }

    #[test]
    fn controller_config_rejects_invalid_order_caps() {
        assert!(
            Lsode2ControllerConfig::bdf_only()
                .with_max_adams_order(0)
                .validate()
                .is_err()
        );
        assert!(
            Lsode2ControllerConfig::bdf_only()
                .with_max_adams_order(LSODE2_MAX_ADAMS_ORDER + 1)
                .validate()
                .is_err()
        );
        assert!(
            Lsode2ControllerConfig::bdf_only()
                .with_max_bdf_order(0)
                .validate()
                .is_err()
        );
        assert!(
            Lsode2ControllerConfig::bdf_only()
                .with_max_bdf_order(LSODE2_MAX_BDF_ORDER + 1)
                .validate()
                .is_err()
        );
        assert!(
            Lsode2ControllerConfig::bdf_only()
                .with_stiffness_ratio_threshold(0.0)
                .validate()
                .is_err()
        );
        assert!(
            Lsode2ControllerConfig::bdf_only()
                .with_method_switch_probe_steps(0)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn execution_plan_marks_adams_only_as_not_executable_yet() {
        let plan = Lsode2ControllerConfig::adams_only().execution_plan();

        assert!(!plan.is_executable_now());
        assert_eq!(plan.executable_family, None);
        assert!(plan.requires_adams_engine);
        assert!(!plan.uses_fallback);
        assert!(plan.message.contains("Adams-only controller"));
    }

    #[test]
    fn automatic_switch_policy_prefers_adams_after_warmup_but_executes_bdf_fallback() {
        let controller =
            Lsode2AlgorithmController::new(Lsode2ControllerConfig::automatic_adams_bdf());
        let decision = controller
            .switch_decision(Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(21));

        assert_eq!(decision.preferred_family, Lsode2MethodFamily::Adams);
        assert_eq!(decision.executed_family(), Some(Lsode2MethodFamily::Bdf));
        assert!(decision.uses_fallback);
        assert_eq!(decision.reason, Lsode2SwitchReason::AdamsEngineUnavailable);
        assert!(
            decision
                .message
                .contains("native Adams execution is unavailable")
        );
    }

    #[test]
    fn automatic_switch_policy_keeps_bdf_in_warmup_window() {
        let controller =
            Lsode2AlgorithmController::new(Lsode2ControllerConfig::automatic_adams_bdf());
        let decision = controller
            .switch_decision(Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(2));

        assert_eq!(decision.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(decision.executed_family(), Some(Lsode2MethodFamily::Bdf));
        assert!(!decision.uses_fallback);
        assert_eq!(decision.reason, Lsode2SwitchReason::SwitchProbeWarmup);
        assert!(decision.message.contains("warmup"));
    }

    #[test]
    fn automatic_switch_policy_respects_custom_probe_window() {
        let controller = Lsode2AlgorithmController::new(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(5),
        );

        let warmup = controller
            .switch_decision(Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(5));
        assert_eq!(warmup.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(warmup.reason, Lsode2SwitchReason::SwitchProbeWarmup);

        let after_warmup = controller
            .switch_decision(Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(6));
        assert_eq!(after_warmup.preferred_family, Lsode2MethodFamily::Adams);
        assert_eq!(
            after_warmup.reason,
            Lsode2SwitchReason::AdamsEngineUnavailable
        );
    }

    #[test]
    fn automatic_switch_policy_prefers_bdf_for_stiffness_or_convergence_trouble() {
        let config =
            Lsode2ControllerConfig::automatic_adams_bdf().with_stiffness_ratio_threshold(10.0);

        let stiff_decision =
            config.switch_decision(Lsode2SwitchTelemetry::default().with_stiffness_ratio(10.0));
        assert_eq!(stiff_decision.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(
            stiff_decision.executed_family(),
            Some(Lsode2MethodFamily::Bdf)
        );
        assert!(!stiff_decision.uses_fallback);
        assert_eq!(
            stiff_decision.reason,
            Lsode2SwitchReason::StiffnessSuspected
        );

        let failed_decision = config.switch_decision(
            Lsode2SwitchTelemetry::default()
                .with_stiffness_ratio(0.1)
                .with_convergence_failures(1),
        );
        assert_eq!(failed_decision.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(
            failed_decision.reason,
            Lsode2SwitchReason::ConvergenceTrouble
        );
    }

    #[test]
    fn fixed_controller_switch_policy_ignores_telemetry() {
        let decision = Lsode2ControllerConfig::bdf_only().switch_decision(
            Lsode2SwitchTelemetry::default()
                .with_stiffness_ratio(0.0)
                .with_rejected_steps(0),
        );

        assert_eq!(decision.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(decision.executed_family(), Some(Lsode2MethodFamily::Bdf));
        assert!(!decision.uses_fallback);
        assert_eq!(decision.reason, Lsode2SwitchReason::FixedController);
    }

    #[test]
    fn controller_records_mused_mcur_like_state() {
        let mut controller =
            Lsode2AlgorithmController::new(Lsode2ControllerConfig::automatic_adams_bdf());
        let first = controller
            .switch_decision(Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(21));
        controller.record_switch_decision(first);
        let s0 = controller.switch_state();
        assert_eq!(s0.mcur, Lsode2MethodFamily::Bdf);
        assert_eq!(s0.mused, Lsode2MethodFamily::Bdf);
        assert_eq!(s0.fallback_count, 1);

        let second = Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Adams,
            executable_family: Some(Lsode2MethodFamily::Adams),
            uses_fallback: false,
            reason: Lsode2SwitchReason::NonstiffPreference,
            message: "adams executable",
        };
        controller.record_switch_decision(second);
        let s1 = controller.switch_state();
        assert_eq!(s1.mused, Lsode2MethodFamily::Bdf);
        assert_eq!(s1.mcur, Lsode2MethodFamily::Adams);
        assert_eq!(s1.switch_count, 1);
    }

    #[test]
    fn controller_stateful_probe_gate_opens_after_icount_like_countdown() {
        let mut controller =
            Lsode2AlgorithmController::new(Lsode2ControllerConfig::automatic_adams_bdf());

        let warmup = controller.switch_decision_stateful(Lsode2SwitchTelemetry::quiet_nonstiff());
        assert_eq!(warmup.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(warmup.reason, Lsode2SwitchReason::SwitchProbeWarmup);

        controller.record_accepted_steps_for_switch_probe(20);
        let still_warmup =
            controller.switch_decision_stateful(Lsode2SwitchTelemetry::quiet_nonstiff());
        assert_eq!(still_warmup.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(still_warmup.reason, Lsode2SwitchReason::SwitchProbeWarmup);

        controller.record_accepted_steps_for_switch_probe(1);
        let probe_ready =
            controller.switch_decision_stateful(Lsode2SwitchTelemetry::quiet_nonstiff());
        assert_eq!(probe_ready.preferred_family, Lsode2MethodFamily::Adams);
        assert_eq!(
            probe_ready.reason,
            Lsode2SwitchReason::AdamsEngineUnavailable
        );
        assert!(probe_ready.uses_fallback);

        let consumed_gate =
            controller.switch_decision_stateful(Lsode2SwitchTelemetry::quiet_nonstiff());
        assert_eq!(consumed_gate.preferred_family, Lsode2MethodFamily::Bdf);
        assert_eq!(consumed_gate.reason, Lsode2SwitchReason::SwitchProbeWarmup);
    }

    #[test]
    fn controller_bootstraps_adams_coefficients_from_faithful_dcfode_branch() {
        let controller = Lsode2AlgorithmController::new(Lsode2ControllerConfig::adams_only());
        let q1 = controller.adams_order_coefficients(1).unwrap();
        let q12 = controller.adams_order_coefficients(12).unwrap();

        assert_eq!(q1.el[1], 1.0);
        assert_eq!(q1.el[2], 1.0);
        assert_eq!(q1.tesco2, 2.0);
        assert!(q12.tesco2.is_finite());
    }
}
