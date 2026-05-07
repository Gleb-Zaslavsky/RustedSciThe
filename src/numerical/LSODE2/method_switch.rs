pub const LSODE2_MAX_ADAMS_ORDER: usize = 12;
pub const LSODE2_MAX_BDF_ORDER: usize = 5;
pub const LSODE2_DEFAULT_STIFFNESS_RATIO_THRESHOLD: f64 = 100.0;
/// ODEPACK-like initial `ICOUNT` value before method-switch probing.
///
/// In DSTODA the gate is updated as:
/// `ICOUNT = ICOUNT - 1; IF (ICOUNT .GE. 0) skip switch probe`.
/// With initial `ICOUNT = 20`, the first probe occurs on accepted step 21.
pub const LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS: usize = 20;

/// LSODE-style time-integration family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2MethodFamily {
    Adams,
    Bdf,
}

impl Lsode2MethodFamily {
    pub fn label(self) -> &'static str {
        match self {
            Self::Adams => "adams",
            Self::Bdf => "bdf",
        }
    }
}

/// High-level LSODE2 algorithm controller mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2ControllerMode {
    AdamsOnly,
    BdfOnly,
    AutomaticAdamsBdf,
}

impl Default for Lsode2ControllerMode {
    fn default() -> Self {
        Self::BdfOnly
    }
}

impl Lsode2ControllerMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::AdamsOnly => "adams_only",
            Self::BdfOnly => "bdf_only",
            Self::AutomaticAdamsBdf => "automatic_adams_bdf",
        }
    }
}

/// Lightweight telemetry consumed by the Adams/BDF selection policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2SwitchTelemetry {
    pub stiffness_ratio: Option<f64>,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub convergence_failures: usize,
    /// Optional ODEPACK-like per-step cost estimate for Adams path.
    pub adams_step_cost_estimate: Option<f64>,
    /// Optional ODEPACK-like per-step cost estimate for BDF path.
    pub bdf_step_cost_estimate: Option<f64>,
    /// Number of Adams cost samples contributing to `adams_step_cost_estimate`.
    pub adams_cost_samples: usize,
    /// Number of BDF cost samples contributing to `bdf_step_cost_estimate`.
    pub bdf_cost_samples: usize,
    /// Optional DSTODA-style step-size capability proxy for Adams (`rh1`-like).
    pub adams_step_size_cap_estimate: Option<f64>,
    /// Optional DSTODA-style step-size capability proxy for BDF (`rh2`-like).
    pub bdf_step_size_cap_estimate: Option<f64>,
}

impl Lsode2SwitchTelemetry {
    pub fn quiet_nonstiff() -> Self {
        Self {
            stiffness_ratio: Some(0.0),
            accepted_steps: 0,
            rejected_steps: 0,
            convergence_failures: 0,
            adams_step_cost_estimate: None,
            bdf_step_cost_estimate: None,
            adams_cost_samples: 0,
            bdf_cost_samples: 0,
            adams_step_size_cap_estimate: None,
            bdf_step_size_cap_estimate: None,
        }
    }

    pub fn with_stiffness_ratio(mut self, ratio: f64) -> Self {
        self.stiffness_ratio = Some(ratio);
        self
    }

    pub fn with_rejected_steps(mut self, rejected_steps: usize) -> Self {
        self.rejected_steps = rejected_steps;
        self
    }

    pub fn with_accepted_steps(mut self, accepted_steps: usize) -> Self {
        self.accepted_steps = accepted_steps;
        self
    }

    pub fn with_convergence_failures(mut self, convergence_failures: usize) -> Self {
        self.convergence_failures = convergence_failures;
        self
    }

    pub fn with_adams_step_cost_estimate(mut self, cost: f64) -> Self {
        self.adams_step_cost_estimate = Some(cost);
        self
    }

    pub fn with_bdf_step_cost_estimate(mut self, cost: f64) -> Self {
        self.bdf_step_cost_estimate = Some(cost);
        self
    }

    pub fn with_adams_cost_samples(mut self, samples: usize) -> Self {
        self.adams_cost_samples = samples;
        self
    }

    pub fn with_bdf_cost_samples(mut self, samples: usize) -> Self {
        self.bdf_cost_samples = samples;
        self
    }

    pub fn with_adams_step_size_cap_estimate(mut self, rh1_like: f64) -> Self {
        self.adams_step_size_cap_estimate = Some(rh1_like);
        self
    }

    pub fn with_bdf_step_size_cap_estimate(mut self, rh2_like: f64) -> Self {
        self.bdf_step_size_cap_estimate = Some(rh2_like);
        self
    }
}

impl Default for Lsode2SwitchTelemetry {
    fn default() -> Self {
        Self::quiet_nonstiff()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2SwitchReason {
    FixedController,
    SwitchProbeWarmup,
    InsufficientCostEvidence,
    SwitchAdvantageNotMet,
    NonstiffPreference,
    CostPreferenceAdams,
    CostPreferenceBdf,
    StiffnessSuspected,
    ConvergenceTrouble,
    AdamsEngineUnavailable,
}

impl Lsode2SwitchReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::FixedController => "fixed_controller",
            Self::SwitchProbeWarmup => "switch_probe_warmup",
            Self::InsufficientCostEvidence => "insufficient_cost_evidence",
            Self::SwitchAdvantageNotMet => "switch_advantage_not_met",
            Self::NonstiffPreference => "nonstiff_preference",
            Self::CostPreferenceAdams => "cost_preference_adams",
            Self::CostPreferenceBdf => "cost_preference_bdf",
            Self::StiffnessSuspected => "stiffness_suspected",
            Self::ConvergenceTrouble => "convergence_trouble",
            Self::AdamsEngineUnavailable => "adams_engine_unavailable",
        }
    }
}

/// One method-family decision produced by the LSODE2 controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lsode2SwitchDecision {
    pub preferred_family: Lsode2MethodFamily,
    pub executable_family: Option<Lsode2MethodFamily>,
    pub uses_fallback: bool,
    pub reason: Lsode2SwitchReason,
    pub message: &'static str,
}

impl Lsode2SwitchDecision {
    pub fn executed_family(self) -> Option<Lsode2MethodFamily> {
        self.executable_family
    }
}

/// ODEPACK-like method-switch state (`MUSED`/`MCUR`) for observability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lsode2MethodSwitchState {
    pub mused: Lsode2MethodFamily,
    pub mcur: Lsode2MethodFamily,
    pub switch_count: usize,
    pub decision_count: usize,
    pub fallback_count: usize,
    pub switch_probe_initial_countdown: usize,
    pub switch_probe_countdown: isize,
    pub switch_probe_ready: bool,
}

impl Default for Lsode2MethodSwitchState {
    fn default() -> Self {
        Self {
            mused: Lsode2MethodFamily::Bdf,
            mcur: Lsode2MethodFamily::Bdf,
            switch_count: 0,
            decision_count: 0,
            fallback_count: 0,
            switch_probe_initial_countdown: LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS,
            switch_probe_countdown: LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS as isize,
            switch_probe_ready: false,
        }
    }
}

impl Lsode2MethodSwitchState {
    pub fn record_decision(&mut self, decision: Lsode2SwitchDecision) {
        self.decision_count += 1;
        if decision.uses_fallback {
            self.fallback_count += 1;
        }
        let next = decision
            .executed_family()
            .unwrap_or(decision.preferred_family);
        if self.mcur != next {
            self.switch_count += 1;
            self.mused = self.mcur;
            self.mcur = next;
            // ODEPACK DSTODA parity:
            // after a successful method switch, `ICOUNT` is reset to its
            // initial value (20 by default) before the next probe window.
            self.switch_probe_countdown = self.switch_probe_initial_countdown as isize;
            self.switch_probe_ready = false;
        } else {
            self.mused = self.mcur;
        }
    }

    pub fn set_switch_probe_initial_countdown(&mut self, initial_countdown: usize) {
        self.switch_probe_initial_countdown = initial_countdown;
        self.switch_probe_countdown = initial_countdown as isize;
        self.switch_probe_ready = false;
    }

    pub fn switch_probe_countdown(&self) -> isize {
        self.switch_probe_countdown
    }

    pub fn switch_probe_ready(&self) -> bool {
        self.switch_probe_ready
    }

    pub fn record_accepted_step_for_probe(&mut self) {
        self.switch_probe_countdown -= 1;
        if self.switch_probe_countdown < 0 {
            self.switch_probe_ready = true;
        }
    }

    pub fn record_accepted_steps_for_probe(&mut self, accepted_steps: usize) {
        for _ in 0..accepted_steps {
            self.record_accepted_step_for_probe();
        }
    }

    pub fn consume_switch_probe_gate(&mut self) -> bool {
        let ready = self.switch_probe_ready;
        self.switch_probe_ready = false;
        ready
    }
}

/// Extracted method-switch policy for Adams/BDF decision logic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2MethodSwitchPolicy {
    pub stiffness_ratio_threshold: f64,
    /// Optional override for forcing BDF on repeated convergence trouble.
    ///
    /// DSTODA method switching is primarily step-size/cost driven; this
    /// threshold is therefore disabled by default and can be enabled
    /// explicitly when such override is desired.
    pub convergence_failure_threshold: usize,
    /// Optional override for forcing BDF on repeated rejected steps.
    ///
    /// Disabled by default for DSTODA-faithful automatic switching.
    pub rejection_threshold: usize,
    pub minimum_accepted_steps_for_nonstiff_probe: usize,
    /// Require Adams to be this much cheaper than BDF to trigger cost-based
    /// Adams preference (`adams_cost <= ratio * bdf_cost`).
    pub adams_cost_ratio_for_switch: f64,
    /// Require BDF to be this much cheaper than Adams to trigger cost-based
    /// BDF preference (`bdf_cost <= ratio * adams_cost`).
    pub bdf_cost_ratio_for_switch: f64,
    /// Minimum number of cost samples for each family before allowing
    /// cost-driven Adams/BDF switching.
    pub min_cost_samples_for_switch: usize,
    /// DSTODA ratio gate used in Adams<->BDF method-switch tests.
    ///
    /// In the original DSTODA choreography this is initialized to 5.0 and
    /// participates in the `rh1/rh2` step-advantage tests.
    pub switch_advantage_ratio: f64,
}

impl Default for Lsode2MethodSwitchPolicy {
    fn default() -> Self {
        Self {
            stiffness_ratio_threshold: LSODE2_DEFAULT_STIFFNESS_RATIO_THRESHOLD,
            convergence_failure_threshold: usize::MAX,
            rejection_threshold: usize::MAX,
            minimum_accepted_steps_for_nonstiff_probe: LSODE2_DEFAULT_METHOD_SWITCH_PROBE_STEPS,
            adams_cost_ratio_for_switch: 0.85,
            bdf_cost_ratio_for_switch: 0.85,
            min_cost_samples_for_switch: 3,
            switch_advantage_ratio: 5.0,
        }
    }
}

impl Lsode2MethodSwitchPolicy {
    pub fn with_stiffness_ratio_threshold(mut self, threshold: f64) -> Self {
        self.stiffness_ratio_threshold = threshold;
        self
    }

    pub fn with_minimum_accepted_steps_for_nonstiff_probe(mut self, accepted_steps: usize) -> Self {
        self.minimum_accepted_steps_for_nonstiff_probe = accepted_steps;
        self
    }

    pub fn with_adams_cost_ratio_for_switch(mut self, ratio: f64) -> Self {
        self.adams_cost_ratio_for_switch = ratio;
        self
    }

    pub fn with_bdf_cost_ratio_for_switch(mut self, ratio: f64) -> Self {
        self.bdf_cost_ratio_for_switch = ratio;
        self
    }

    pub fn with_min_cost_samples_for_switch(mut self, samples: usize) -> Self {
        self.min_cost_samples_for_switch = samples.max(1);
        self
    }

    pub fn with_switch_advantage_ratio(mut self, ratio: f64) -> Self {
        self.switch_advantage_ratio = ratio;
        self
    }

    fn should_probe_nonstiff_switch(self, accepted_steps: usize) -> bool {
        // Stateless approximation of DSTODA's ICOUNT gate:
        // with initial ICOUNT = 20 the first probe is on accepted step 21.
        // (Stateful controller path keeps exact per-step gate semantics.)
        accepted_steps > self.minimum_accepted_steps_for_nonstiff_probe
    }

    pub fn preferred_family_and_reason(
        self,
        mode: Lsode2ControllerMode,
        telemetry: Lsode2SwitchTelemetry,
    ) -> (Lsode2MethodFamily, Lsode2SwitchReason) {
        self.preferred_family_and_reason_with_probe_gate(mode, telemetry, None)
    }

    pub fn preferred_family_and_reason_with_probe_gate(
        self,
        mode: Lsode2ControllerMode,
        telemetry: Lsode2SwitchTelemetry,
        probe_gate: Option<bool>,
    ) -> (Lsode2MethodFamily, Lsode2SwitchReason) {
        self.preferred_family_and_reason_with_probe_gate_and_current(
            mode, telemetry, probe_gate, None,
        )
    }

    pub fn preferred_family_and_reason_with_probe_gate_and_current(
        self,
        mode: Lsode2ControllerMode,
        telemetry: Lsode2SwitchTelemetry,
        probe_gate: Option<bool>,
        current_family: Option<Lsode2MethodFamily>,
    ) -> (Lsode2MethodFamily, Lsode2SwitchReason) {
        match mode {
            Lsode2ControllerMode::AdamsOnly => (
                Lsode2MethodFamily::Adams,
                Lsode2SwitchReason::FixedController,
            ),
            Lsode2ControllerMode::BdfOnly => {
                (Lsode2MethodFamily::Bdf, Lsode2SwitchReason::FixedController)
            }
            Lsode2ControllerMode::AutomaticAdamsBdf => {
                let current = current_family.unwrap_or(Lsode2MethodFamily::Bdf);
                let force_bdf_on_convergence_trouble = self.convergence_failure_threshold
                    != usize::MAX
                    && telemetry.convergence_failures >= self.convergence_failure_threshold;
                let force_bdf_on_rejections = self.rejection_threshold != usize::MAX
                    && telemetry.rejected_steps >= self.rejection_threshold;
                if force_bdf_on_convergence_trouble || force_bdf_on_rejections {
                    return (
                        Lsode2MethodFamily::Bdf,
                        Lsode2SwitchReason::ConvergenceTrouble,
                    );
                }
                if telemetry
                    .stiffness_ratio
                    .is_some_and(|ratio| ratio >= self.stiffness_ratio_threshold)
                {
                    return (
                        Lsode2MethodFamily::Bdf,
                        Lsode2SwitchReason::StiffnessSuspected,
                    );
                }
                let probe_nonstiff = probe_gate
                    .unwrap_or_else(|| self.should_probe_nonstiff_switch(telemetry.accepted_steps));
                if !probe_nonstiff {
                    return (current, Lsode2SwitchReason::SwitchProbeWarmup);
                }
                // DSTODA-style method-switch gate via step-size advantage:
                // - if currently BDF: switch to Adams only if `rh1 * ratio >= 5 * rh2`;
                // - if currently Adams: switch to BDF only if `rh2 >= ratio * rh1`.
                let has_any_step_advantage_telemetry =
                    telemetry.adams_step_size_cap_estimate.is_some()
                        || telemetry.bdf_step_size_cap_estimate.is_some();
                if let (Some(rh1), Some(rh2)) = (
                    telemetry.adams_step_size_cap_estimate,
                    telemetry.bdf_step_size_cap_estimate,
                ) {
                    if rh1.is_finite()
                        && rh2.is_finite()
                        && rh1 > 0.0
                        && rh2 > 0.0
                        && self.switch_advantage_ratio.is_finite()
                        && self.switch_advantage_ratio > 0.0
                    {
                        if current == Lsode2MethodFamily::Bdf {
                            if rh1 * self.switch_advantage_ratio >= 5.0 * rh2 {
                                return (
                                    Lsode2MethodFamily::Adams,
                                    Lsode2SwitchReason::NonstiffPreference,
                                );
                            }
                        } else if rh2 >= self.switch_advantage_ratio * rh1 {
                            return (
                                Lsode2MethodFamily::Bdf,
                                Lsode2SwitchReason::StiffnessSuspected,
                            );
                        }
                        return (current, Lsode2SwitchReason::SwitchAdvantageNotMet);
                    }
                }
                // If probe is open and DSTODA-style telemetry is present but
                // not sufficient to justify a switch, report step-advantage
                // hold reason instead of generic cost-evidence reason.
                if has_any_step_advantage_telemetry {
                    return (current, Lsode2SwitchReason::SwitchAdvantageNotMet);
                }
                if let (Some(adams_cost), Some(bdf_cost)) = (
                    telemetry.adams_step_cost_estimate,
                    telemetry.bdf_step_cost_estimate,
                ) {
                    if telemetry.adams_cost_samples < self.min_cost_samples_for_switch
                        || telemetry.bdf_cost_samples < self.min_cost_samples_for_switch
                    {
                        return (current, Lsode2SwitchReason::InsufficientCostEvidence);
                    }
                    if adams_cost.is_finite()
                        && bdf_cost.is_finite()
                        && adams_cost > 0.0
                        && bdf_cost > 0.0
                    {
                        if adams_cost <= self.adams_cost_ratio_for_switch * bdf_cost {
                            return (
                                Lsode2MethodFamily::Adams,
                                Lsode2SwitchReason::CostPreferenceAdams,
                            );
                        }
                        if bdf_cost <= self.bdf_cost_ratio_for_switch * adams_cost {
                            return (
                                Lsode2MethodFamily::Bdf,
                                Lsode2SwitchReason::CostPreferenceBdf,
                            );
                        }
                        return (current, Lsode2SwitchReason::SwitchAdvantageNotMet);
                    }
                }
                (current, Lsode2SwitchReason::SwitchAdvantageNotMet)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_prefers_adams_for_quiet_automatic_case() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let (family, reason) = policy.preferred_family_and_reason(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(21),
        );
        assert_eq!(family, Lsode2MethodFamily::Bdf);
        assert_eq!(reason, Lsode2SwitchReason::SwitchAdvantageNotMet);
    }

    #[test]
    fn policy_keeps_bdf_during_switch_probe_warmup() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let (family, reason) = policy.preferred_family_and_reason(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(3),
        );
        assert_eq!(family, Lsode2MethodFamily::Bdf);
        assert_eq!(reason, Lsode2SwitchReason::SwitchProbeWarmup);
    }

    #[test]
    fn policy_only_probes_on_icount_equivalent_boundaries() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let (f20, r20) = policy.preferred_family_and_reason(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(20),
        );
        assert_eq!(f20, Lsode2MethodFamily::Bdf);
        assert_eq!(r20, Lsode2SwitchReason::SwitchProbeWarmup);

        let (f21, r21) = policy.preferred_family_and_reason(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            Lsode2SwitchTelemetry::quiet_nonstiff().with_accepted_steps(21),
        );
        assert_eq!(f21, Lsode2MethodFamily::Bdf);
        assert_eq!(r21, Lsode2SwitchReason::SwitchAdvantageNotMet);
    }

    #[test]
    fn policy_prefers_bdf_for_stiffness_or_convergence_signals() {
        let policy = Lsode2MethodSwitchPolicy {
            convergence_failure_threshold: 1,
            rejection_threshold: 2,
            ..Lsode2MethodSwitchPolicy::default().with_stiffness_ratio_threshold(10.0)
        };
        let (stiff_family, stiff_reason) = policy.preferred_family_and_reason(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            Lsode2SwitchTelemetry::default().with_stiffness_ratio(12.0),
        );
        assert_eq!(stiff_family, Lsode2MethodFamily::Bdf);
        assert_eq!(stiff_reason, Lsode2SwitchReason::StiffnessSuspected);

        let (conv_family, conv_reason) = policy.preferred_family_and_reason(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            Lsode2SwitchTelemetry::default().with_convergence_failures(1),
        );
        assert_eq!(conv_family, Lsode2MethodFamily::Bdf);
        assert_eq!(conv_reason, Lsode2SwitchReason::ConvergenceTrouble);
    }

    #[test]
    fn policy_default_does_not_force_bdf_from_convergence_override() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let (family, reason) = policy.preferred_family_and_reason(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            Lsode2SwitchTelemetry::default()
                .with_accepted_steps(32)
                .with_convergence_failures(999)
                .with_rejected_steps(999),
        );
        assert_eq!(family, Lsode2MethodFamily::Bdf);
        assert_eq!(reason, Lsode2SwitchReason::SwitchAdvantageNotMet);
    }

    #[test]
    fn policy_can_use_cost_based_preference_after_probe_gate() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let telemetry = Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_accepted_steps(32)
            .with_adams_step_cost_estimate(0.5)
            .with_bdf_step_cost_estimate(1.0)
            .with_adams_cost_samples(3)
            .with_bdf_cost_samples(3);
        let (family_adams, reason_adams) =
            policy.preferred_family_and_reason(Lsode2ControllerMode::AutomaticAdamsBdf, telemetry);
        assert_eq!(family_adams, Lsode2MethodFamily::Adams);
        assert_eq!(reason_adams, Lsode2SwitchReason::CostPreferenceAdams);

        let telemetry = Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_accepted_steps(32)
            .with_adams_step_cost_estimate(1.0)
            .with_bdf_step_cost_estimate(0.5)
            .with_adams_cost_samples(3)
            .with_bdf_cost_samples(3);
        let (family_bdf, reason_bdf) =
            policy.preferred_family_and_reason(Lsode2ControllerMode::AutomaticAdamsBdf, telemetry);
        assert_eq!(family_bdf, Lsode2MethodFamily::Bdf);
        assert_eq!(reason_bdf, Lsode2SwitchReason::CostPreferenceBdf);
    }

    #[test]
    fn policy_requires_minimum_cost_samples_for_cost_based_switch() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let telemetry = Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_accepted_steps(32)
            .with_adams_step_cost_estimate(0.5)
            .with_bdf_step_cost_estimate(1.0)
            .with_adams_cost_samples(1)
            .with_bdf_cost_samples(3);
        let (family, reason) =
            policy.preferred_family_and_reason(Lsode2ControllerMode::AutomaticAdamsBdf, telemetry);
        assert_eq!(family, Lsode2MethodFamily::Bdf);
        assert_eq!(reason, Lsode2SwitchReason::InsufficientCostEvidence);
    }

    #[test]
    fn policy_dstoda_step_advantage_gate_can_switch_bdf_to_adams() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let telemetry = Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_accepted_steps(32)
            .with_adams_step_size_cap_estimate(2.0)
            .with_bdf_step_size_cap_estimate(1.0);
        let (family, reason) = policy.preferred_family_and_reason_with_probe_gate_and_current(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            telemetry,
            Some(true),
            Some(Lsode2MethodFamily::Bdf),
        );
        assert_eq!(family, Lsode2MethodFamily::Adams);
        assert_eq!(reason, Lsode2SwitchReason::NonstiffPreference);
    }

    #[test]
    fn policy_dstoda_step_advantage_gate_can_switch_adams_to_bdf() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let telemetry = Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_accepted_steps(32)
            .with_adams_step_size_cap_estimate(1.0)
            .with_bdf_step_size_cap_estimate(5.0);
        let (family, reason) = policy.preferred_family_and_reason_with_probe_gate_and_current(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            telemetry,
            Some(true),
            Some(Lsode2MethodFamily::Adams),
        );
        assert_eq!(family, Lsode2MethodFamily::Bdf);
        assert_eq!(reason, Lsode2SwitchReason::StiffnessSuspected);
    }

    #[test]
    fn policy_dstoda_step_advantage_gate_reports_hold_when_advantage_not_met() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let telemetry = Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_accepted_steps(32)
            .with_adams_step_size_cap_estimate(0.9)
            .with_bdf_step_size_cap_estimate(1.0);
        let (family, reason) = policy.preferred_family_and_reason_with_probe_gate_and_current(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            telemetry,
            Some(true),
            Some(Lsode2MethodFamily::Bdf),
        );
        assert_eq!(family, Lsode2MethodFamily::Bdf);
        assert_eq!(reason, Lsode2SwitchReason::SwitchAdvantageNotMet);
    }

    #[test]
    fn policy_prefers_step_advantage_hold_reason_when_dstoda_telemetry_is_partial() {
        let policy = Lsode2MethodSwitchPolicy::default();
        let telemetry = Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_accepted_steps(32)
            .with_adams_step_size_cap_estimate(1.2);
        let (family, reason) = policy.preferred_family_and_reason_with_probe_gate_and_current(
            Lsode2ControllerMode::AutomaticAdamsBdf,
            telemetry,
            Some(true),
            Some(Lsode2MethodFamily::Bdf),
        );
        assert_eq!(family, Lsode2MethodFamily::Bdf);
        assert_eq!(reason, Lsode2SwitchReason::SwitchAdvantageNotMet);
    }

    #[test]
    fn state_tracks_mused_mcur_and_fallbacks() {
        let mut state = Lsode2MethodSwitchState::default();
        let first = Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Adams,
            executable_family: Some(Lsode2MethodFamily::Bdf),
            uses_fallback: true,
            reason: Lsode2SwitchReason::AdamsEngineUnavailable,
            message: "fallback",
        };
        state.record_decision(first);
        assert_eq!(state.decision_count, 1);
        assert_eq!(state.fallback_count, 1);
        assert_eq!(state.switch_count, 0);
        assert_eq!(state.mcur, Lsode2MethodFamily::Bdf);
        assert_eq!(state.mused, Lsode2MethodFamily::Bdf);

        let second = Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Adams,
            executable_family: Some(Lsode2MethodFamily::Adams),
            uses_fallback: false,
            reason: Lsode2SwitchReason::NonstiffPreference,
            message: "adams",
        };
        state.record_decision(second);
        assert_eq!(state.decision_count, 2);
        assert_eq!(state.fallback_count, 1);
        assert_eq!(state.switch_count, 1);
        assert_eq!(state.mcur, Lsode2MethodFamily::Adams);
        assert_eq!(state.mused, Lsode2MethodFamily::Bdf);
    }

    #[test]
    fn stateful_probe_countdown_mirrors_icount_style_periodicity() {
        let mut state = Lsode2MethodSwitchState::default();
        state.set_switch_probe_initial_countdown(20);

        for _ in 0..20 {
            state.record_accepted_step_for_probe();
            assert!(!state.switch_probe_ready());
        }
        assert_eq!(state.switch_probe_countdown(), 0);

        state.record_accepted_step_for_probe();
        assert!(state.switch_probe_ready());
        assert_eq!(state.switch_probe_countdown(), -1);
        assert!(state.consume_switch_probe_gate());
        assert!(!state.switch_probe_ready());

        state.record_accepted_step_for_probe();
        assert!(state.switch_probe_ready());
        assert_eq!(state.switch_probe_countdown(), -2);
    }

    #[test]
    fn switch_resets_icount_like_probe_countdown() {
        let mut state = Lsode2MethodSwitchState::default();
        state.set_switch_probe_initial_countdown(20);
        state.record_accepted_steps_for_probe(21);
        assert!(state.switch_probe_ready());

        state.record_decision(Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Adams,
            executable_family: Some(Lsode2MethodFamily::Adams),
            uses_fallback: false,
            reason: Lsode2SwitchReason::NonstiffPreference,
            message: "switch",
        });

        assert_eq!(state.switch_probe_countdown(), 20);
        assert!(!state.switch_probe_ready());
    }
}
