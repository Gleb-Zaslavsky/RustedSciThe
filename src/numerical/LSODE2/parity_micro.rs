use crate::numerical::LSODE2::algorithm::{
    Lsode2AlgorithmController, Lsode2ControllerExecutionCapabilities,
};
use crate::numerical::LSODE2::step_cycle::Lsode2StepMethod;
use crate::numerical::LSODE2::{
    Lsode2ControllerConfig, Lsode2CorrectionAssessment, Lsode2CorrectionControlConfig,
    Lsode2CorrectionController, Lsode2CorrectionStatus, Lsode2ErrorControlConfig,
    Lsode2ErrorController, Lsode2Icf, Lsode2Ipup, Lsode2IpupTrigger, Lsode2Iredo, Lsode2Iret,
    Lsode2JacobianUpdateRequest, Lsode2Kflag, Lsode2MethodFamily, Lsode2NativeStatistics,
    Lsode2RedoStage, Lsode2RetryAction, Lsode2RuntimeState, Lsode2StepControlConfig,
    Lsode2StepCycle, Lsode2StepCycleOutcome, Lsode2SwitchReason, Lsode2SwitchTelemetry,
    Lsode2Tolerance,
};
// Replicate small DSTODA helper logic needed for expectations.
fn dstoda_adams_pdh(h: f64, pdlast: f64) -> f64 {
    if !h.is_finite() || !pdlast.is_finite() {
        return 0.0;
    }
    (h.abs() * pdlast).max(1.0e-6)
}

fn dstoda_adams_sm1(index_1_based: usize) -> f64 {
    const SM1: [f64; 12] = [
        0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025,
    ];
    if (1..=12).contains(&index_1_based) {
        SM1[index_1_based - 1]
    } else {
        SM1[11]
    }
}

fn dstoda_rh_sm_for_expectation(dsm: f64, l: usize) -> f64 {
    if !dsm.is_finite() || dsm <= 0.0 || l == 0 {
        return 0.0;
    }
    let exsm = 1.0 / l as f64;
    1.0 / (1.2 * dsm.powf(exsm) + 1.2e-6)
}

fn dstoda_rh_dn_for_expectation(ddn: f64, nq: usize) -> f64 {
    if !ddn.is_finite() || ddn <= 0.0 || nq == 0 {
        return 0.0;
    }
    let exdn = 1.0 / nq as f64;
    1.0 / (1.3 * ddn.powf(exdn) + 1.3e-6)
}

fn expected_adams_rh_nq_without_qplus1(
    h: f64,
    pdlast: f64,
    order_current: usize,
    current_error_norm: f64,
    lower_error_norm: f64,
) -> (usize, f64) {
    let l = order_current + 1;
    let pdh = dstoda_adams_pdh(h, pdlast);
    let mut rhsm = dstoda_rh_sm_for_expectation(current_error_norm, l);
    let mut rhdn = if order_current > 1 {
        dstoda_rh_dn_for_expectation(lower_error_norm, order_current)
    } else {
        0.0
    };
    if pdh > 0.0 {
        rhsm = rhsm.min(dstoda_adams_sm1(order_current) / pdh);
        if order_current > 1 {
            rhdn = rhdn.min(dstoda_adams_sm1(order_current - 1) / pdh);
        }
    }

    let (mut order_new, mut rh) = if rhsm < rhdn {
        (order_current.saturating_sub(1).max(1), rhdn)
    } else {
        (order_current, rhsm)
    };
    let stability_limited_bypass = pdh > 0.0
        && order_new >= 1
        && order_new <= 12
        && (rh * pdh * 1.00001) < dstoda_adams_sm1(order_new);
    if rh < 1.1 && !stability_limited_bypass {
        order_new = order_current;
        rh = 1.0;
    }
    if !rh.is_finite() || rh <= 0.0 {
        order_new = order_current;
        rh = 1.0;
    }
    (order_new, rh)
}

#[test]
fn adams_divergence_detection_in_controller() {
    // Build a correction controller and assert DEL>2*DELP is detected as Diverged
    let controller = Lsode2CorrectionController::scalar(
        1.0e-3,
        1.0e-6,
        Lsode2CorrectionControlConfig::default(),
    )
    .expect("build correction controller");

    // previous weighted norm small, current large to force divergence ratio > 2
    let prev_norm = Some(1.0e-6);
    let assessment = controller
        .assess_iteration_with_dstoda_context(
            1,
            &[1.0],    // weight source (y scale)
            &[1.0e-2], // correction (current)
            &[1.0e-2], // accumulated
            prev_norm,
            Some(10.0), // previous_rate_estimate (not used here)
            2,          // iteration >=2 to enable DEL/DELP check
            None,
            None,
        )
        .expect("assessment should run");

    assert_eq!(assessment.status, Lsode2CorrectionStatus::Diverged);
}

#[test]
fn dstoda_adams_del_delp_divergence_gate_is_strict_and_iteration_gated() {
    let controller = Lsode2CorrectionController::scalar(
        1.0e-3,
        1.0e-6,
        Lsode2CorrectionControlConfig::default(),
    )
    .expect("build correction controller");

    let y_scale = [1.0];
    // `previous_weighted_norm` is in weighted-norm units (DEL/DELP parity).
    let prev_norm = Some(1.0);

    // Iteration 1: even with large DEL, Adams first-pass path must not diverge.
    let first_pass = controller
        .assess_iteration_with_dstoda_context(
            1,
            &y_scale,
            &[3.0e-3], // DEL / DELP = 3.0 (would diverge if M>=2)
            &[3.0e-3],
            prev_norm,
            Some(1.0),
            1,
            None,
            Some(
                crate::numerical::LSODE2::correction::Lsode2DstodaCorrectorContext {
                    method_is_adams: true,
                    previous_rate_max: Some(1.0),
                    h_el1_abs: Some(1.0),
                    roundoff_tolerance: Some(0.0),
                },
            ),
        )
        .expect("first-pass assessment should succeed");
    assert_eq!(
        first_pass.status,
        Lsode2CorrectionStatus::Continue,
        "Adams first pass should not take divergence terminal branch"
    );

    // Iteration 2, near-boundary DEL < 2*DELP should NOT diverge (strict > 2*DELP).
    let boundary = controller
        .assess_iteration_with_dstoda_context(
            1,
            &y_scale,
            &[1.999e-3],
            &[1.999e-3],
            prev_norm,
            Some(1.0),
            2,
            None,
            Some(
                crate::numerical::LSODE2::correction::Lsode2DstodaCorrectorContext {
                    method_is_adams: true,
                    previous_rate_max: Some(1.0),
                    h_el1_abs: Some(1.0),
                    roundoff_tolerance: Some(0.0),
                },
            ),
        )
        .expect("boundary assessment should succeed");
    assert_ne!(
        boundary.status,
        Lsode2CorrectionStatus::Diverged,
        "DSTODA branch is strict: DEL < 2*DELP should not diverge"
    );

    // Iteration 2, DEL > 2*DELP must diverge.
    let diverged = controller
        .assess_iteration_with_dstoda_context(
            1,
            &y_scale,
            &[2.1e-3],
            &[2.1e-3],
            prev_norm,
            Some(1.0),
            2,
            None,
            Some(
                crate::numerical::LSODE2::correction::Lsode2DstodaCorrectorContext {
                    method_is_adams: true,
                    previous_rate_max: Some(1.0),
                    h_el1_abs: Some(1.0),
                    roundoff_tolerance: Some(0.0),
                },
            ),
        )
        .expect("divergence assessment should succeed");
    assert_eq!(diverged.status, Lsode2CorrectionStatus::Diverged);
}

#[test]
fn dstoda_adams_acor_over_tesco2_local_error_scaling_matches_fortran_for_multiple_orders() {
    let controller = Lsode2CorrectionController::scalar(
        1.0e-3,
        1.0e-6,
        Lsode2CorrectionControlConfig::default(),
    )
    .expect("build correction controller");
    let tables = crate::numerical::LSODE2::adams_engine::Lsode2AdamsDcfodeTables::default();
    let y_scale = [1.0, 1.0];
    let expected = [1.25_f64, -0.75_f64];

    for q in 1..=12 {
        let tesco2 = tables.order(q).expect("supported Adams order").tesco2;
        let acor = [expected[0] * tesco2, expected[1] * tesco2];
        let assessment = controller
            .assess_iteration_with_dstoda_context(
                q,
                &y_scale,
                &acor,
                &acor,
                Some(1.0),
                Some(1.0),
                2,
                Some(tesco2),
                Some(
                    crate::numerical::LSODE2::correction::Lsode2DstodaCorrectorContext {
                        method_is_adams: true,
                        previous_rate_max: Some(1.0),
                        h_el1_abs: Some(1.0),
                        roundoff_tolerance: Some(0.0),
                    },
                ),
            )
            .expect("ACOR/TESCO assessment should succeed");
        assert!(
            (assessment.local_error[0] - expected[0]).abs() < 1.0e-12
                && (assessment.local_error[1] - expected[1]).abs() < 1.0e-12,
            "ACOR/TESCO(2,q) local error mismatch at q={q}: got={:?}",
            assessment.local_error
        );
    }
}

#[test]
fn adams_pdlast_sm1_limits_rh_selection() {
    // Build an Adams-like step cycle and exercise PDEST/PDLAST behaviour and RH limit
    let state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
        .expect("runtime state");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error controller");

    let mut cycle =
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike);

    // Seed an Adams Lipschitz estimate (PDESC/PDLAST)
    let assessment = Lsode2CorrectionAssessment {
        order: 1,
        iteration: 2,
        weighted_norm: 0.0,
        accumulated_weighted_norm: 0.0,
        previous_weighted_norm: None,
        previous_rate_max: Some(1.0),
        convergence_ratio: None,
        convergence_rate_estimate: None,
        rate_max_estimate: None,
        pdest_candidate: Some(5.0),
        convergence_measure: 0.0,
        status: Lsode2CorrectionStatus::Converged,
        local_error: vec![0.0],
        needs_jacobian_refresh: false,
    };

    cycle.record_adams_lipschitz_estimate_from_assessment(&assessment);

    // Verify PDEST/PDLAST behavior
    assert_eq!(cycle.adams_pdest(), 5.0);
    assert_eq!(cycle.adams_pdlast(), 5.0);

    // Run a post-accept order selection which should clear PDEST but keep PDLAST
    let decision = cycle
        .select_post_accept_order(&[0.905], 1.0, None)
        .expect("select order");

    // compute expected PDH and SM1 limits
    let h = cycle.state().h();
    let pdlast = cycle.adams_pdlast();
    let pdh = dstoda_adams_pdh(h, pdlast);
    let sm1 = dstoda_adams_sm1(cycle.state().order());
    // If stability-limiting applies, RH should not exceed sm1/pdh
    if pdh > 0.0 {
        let expected_limit = sm1 / pdh;
        // allow small floating tolerance
        let tol = 1.0e-12;
        assert!(
            decision.suggested_growth <= expected_limit + tol
                || decision.suggested_growth <= 1.0 + tol,
            "suggested_growth should respect SM1/PDH limit (got {} <= {}); decision={:?}",
            decision.suggested_growth,
            expected_limit,
            decision
        );
    }

    // PDEST should be cleared after RH selection; PDLAST should remain
    assert_eq!(
        cycle.adams_pdest(),
        0.0,
        "PDEST should be cleared after RH selection"
    );
    assert_eq!(
        cycle.adams_pdlast(),
        5.0,
        "PDLAST should retain last nonzero estimate"
    );
}

#[test]
fn dstoda_adams_rh_nq_sequence_replay_matches_fortran_style_branch_ordering() {
    let state = Lsode2RuntimeState::new(0.0, &[0.0], 0.2, 3, Lsode2StepControlConfig::default())
        .expect("runtime state should initialize for Adams RH/NQ replay");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(0.0, 1.0),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error control should initialize for Adams RH/NQ replay");
    let mut cycle =
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike);
    cycle
        .state_mut()
        .set_order(2)
        .expect("order update should succeed");

    #[derive(Clone, Copy)]
    struct Case {
        pdlast: f64,
        dsm: f64,
        ddn: f64,
    }

    let cases = [
        // Case 0: strong stability cap + strict 1.1 gate => stay at current order and RH=1.
        Case {
            pdlast: 10.0,
            dsm: 0.7,
            ddn: 0.05,
        },
        // Case 1: same local errors but weaker stability cap => lower-order candidate with RH>1.1.
        Case {
            pdlast: 1.0,
            dsm: 0.7,
            ddn: 0.05,
        },
        // Case 2: order=1, RH<1.1 but stability-limited bypass keeps sub-10% RH (no forced RH=1).
        Case {
            pdlast: 1.0,
            dsm: 0.9,
            ddn: 0.0,
        },
    ];

    let mut replay_actual = Vec::new();
    let mut replay_expected = Vec::new();
    for case in cases {
        cycle.record_adams_lipschitz_estimate_from_assessment(&Lsode2CorrectionAssessment {
            order: cycle.state().order(),
            iteration: 2,
            weighted_norm: 0.0,
            accumulated_weighted_norm: 0.0,
            previous_weighted_norm: None,
            previous_rate_max: None,
            convergence_ratio: None,
            convergence_rate_estimate: None,
            rate_max_estimate: None,
            pdest_candidate: Some(case.pdlast),
            convergence_measure: 0.0,
            status: Lsode2CorrectionStatus::Converged,
            local_error: vec![0.0],
            needs_jacobian_refresh: false,
        });
        let order_current = cycle.state().order();
        if order_current > 1 {
            let tesco1 = crate::numerical::LSODE2::adams_engine::Lsode2AdamsDcfodeTables::default()
                .order(order_current)
                .expect("supported Adams order")
                .tesco1;
            cycle
                .state_mut()
                .nordsieck_mut()
                .set_col(order_current, &[case.ddn * tesco1])
                .expect("nordsieck column update should succeed");
        }

        let h_before = cycle.state().h();
        let expected = expected_adams_rh_nq_without_qplus1(
            h_before,
            case.pdlast,
            order_current,
            case.dsm,
            case.ddn,
        );
        replay_expected.push(expected);

        let decision = cycle
            .select_post_accept_order(&[0.0], case.dsm, None)
            .expect("order selection should succeed");
        replay_actual.push((decision.order_new, decision.suggested_growth));

        let h_after = h_before * decision.suggested_growth;
        cycle
            .state_mut()
            .set_order(decision.order_new)
            .expect("order update for replay should succeed");
        cycle
            .state_mut()
            .set_step_size(h_after)
            .expect("step-size update for replay should succeed");
    }

    assert_eq!(
        replay_actual.len(),
        replay_expected.len(),
        "replay lengths should match"
    );
    for (idx, ((actual_q, actual_rh), (expected_q, expected_rh))) in
        replay_actual.iter().zip(replay_expected.iter()).enumerate()
    {
        assert_eq!(
            *actual_q, *expected_q,
            "Adams RH/NQ replay order mismatch at case #{idx}"
        );
        assert!(
            (*actual_rh - *expected_rh).abs() <= 1.0e-12,
            "Adams RH replay mismatch at case #{idx}: actual={actual_rh:e} expected={expected_rh:e}"
        );
    }
}

#[test]
fn dstoda_switch_choreography_label_matrix_reason_cost_stiff_gates() {
    #[derive(Debug, Clone, Copy)]
    struct Case {
        label: &'static str,
        telemetry: Lsode2SwitchTelemetry,
        probe_gate: Option<bool>,
        current: Option<Lsode2MethodFamily>,
        expected_family: Lsode2MethodFamily,
        expected_reason: Lsode2SwitchReason,
    }

    let config = Lsode2ControllerConfig::automatic_adams_bdf()
        .with_method_switch_probe_steps(1)
        .with_stiffness_ratio_threshold(10.0)
        .with_convergence_failure_threshold(2)
        .with_min_cost_samples_for_switch(3);
    let caps = Lsode2ControllerExecutionCapabilities {
        adams_engine_available: true,
    };

    // Label names below are parity-audit labels for this Rust replay matrix.
    // They intentionally map to concrete DSTODA/LSODA-like gate families:
    // warmup gate, step-advantage hold, cost evidence gate, and stiffness/convergence overrides.
    let cases = [
        Case {
            label: "L-WARMUP-ICOUNT",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff(),
            probe_gate: Some(false),
            current: Some(Lsode2MethodFamily::Adams),
            expected_family: Lsode2MethodFamily::Adams,
            expected_reason: Lsode2SwitchReason::SwitchProbeWarmup,
        },
        Case {
            label: "L-STEP-ADV-HOLD",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_size_cap_estimate(1.0)
                .with_bdf_step_size_cap_estimate(1.0),
            probe_gate: Some(true),
            current: Some(Lsode2MethodFamily::Adams),
            expected_family: Lsode2MethodFamily::Adams,
            expected_reason: Lsode2SwitchReason::SwitchAdvantageNotMet,
        },
        Case {
            label: "L-COST-EVIDENCE-MISS",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_cost_estimate(2.0)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(1)
                .with_bdf_cost_samples(3),
            probe_gate: Some(true),
            current: Some(Lsode2MethodFamily::Adams),
            expected_family: Lsode2MethodFamily::Adams,
            expected_reason: Lsode2SwitchReason::InsufficientCostEvidence,
        },
        Case {
            label: "L-COST-PREF-BDF",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_cost_estimate(2.0)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(3)
                .with_bdf_cost_samples(3),
            probe_gate: Some(true),
            current: Some(Lsode2MethodFamily::Adams),
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::CostPreferenceBdf,
        },
        Case {
            label: "L-STIFF-OVERRIDE",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff().with_stiffness_ratio(10.0),
            probe_gate: Some(false),
            current: Some(Lsode2MethodFamily::Adams),
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::StiffnessSuspected,
        },
        Case {
            label: "L-CONV-OVERRIDE",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff().with_convergence_failures(2),
            probe_gate: Some(true),
            current: Some(Lsode2MethodFamily::Adams),
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::ConvergenceTrouble,
        },
    ];

    for case in cases {
        let decision = config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
            case.telemetry,
            case.probe_gate,
            caps,
            case.current,
        );

        assert_eq!(
            decision.preferred_family, case.expected_family,
            "{}: unexpected family {:?}, expected {:?}",
            case.label, decision.preferred_family, case.expected_family
        );
        assert_eq!(
            decision.reason, case.expected_reason,
            "{}: unexpected reason {:?}, expected {:?}",
            case.label, decision.reason, case.expected_reason
        );
        assert!(
            !decision.uses_fallback,
            "{}: parity case should not use fallback path",
            case.label
        );
    }
}

#[test]
fn dstoda_switch_choreography_branch_precedence_matrix_matches_fortran_style_ordering() {
    #[derive(Debug, Clone, Copy)]
    struct Case {
        label: &'static str,
        telemetry: Lsode2SwitchTelemetry,
        probe_gate: Option<bool>,
        current: Lsode2MethodFamily,
        expected_family: Lsode2MethodFamily,
        expected_reason: Lsode2SwitchReason,
    }

    let config = Lsode2ControllerConfig::automatic_adams_bdf()
        .with_method_switch_probe_steps(20)
        .with_stiffness_ratio_threshold(10.0)
        .with_convergence_failure_threshold(2)
        .with_min_cost_samples_for_switch(3);
    let caps = Lsode2ControllerExecutionCapabilities {
        adams_engine_available: true,
    };

    let cases = [
        // Fortran-style ordering: convergence trouble gate is checked first.
        Case {
            label: "P1-CONV-OVERRIDES-STIFF-AND-COST",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_convergence_failures(2)
                .with_stiffness_ratio(1.0e3)
                .with_adams_step_cost_estimate(0.1)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(3)
                .with_bdf_cost_samples(3),
            probe_gate: Some(false),
            current: Lsode2MethodFamily::Adams,
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::ConvergenceTrouble,
        },
        // Stiffness gate is checked before probe warmup gate.
        Case {
            label: "P2-STIFF-OVERRIDES-PROBE-WARMUP",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff().with_stiffness_ratio(10.0),
            probe_gate: Some(false),
            current: Lsode2MethodFamily::Adams,
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::StiffnessSuspected,
        },
        // When no hard stiff signal exists and probe is closed, stay in warmup.
        Case {
            label: "P3-PROBE-WARMUP-HOLDS-CURRENT",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_cost_estimate(0.1)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(3)
                .with_bdf_cost_samples(3),
            probe_gate: Some(false),
            current: Lsode2MethodFamily::Bdf,
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::SwitchProbeWarmup,
        },
        // With probe open and full RH caps present, step-advantage branch has
        // precedence over cost branch.
        Case {
            label: "P4-STEP-ADVANTAGE-HAS-PRECEDENCE-OVER-COST",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_size_cap_estimate(0.9)
                .with_bdf_step_size_cap_estimate(1.0)
                .with_adams_step_cost_estimate(0.1)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(3)
                .with_bdf_cost_samples(3),
            probe_gate: Some(true),
            current: Lsode2MethodFamily::Bdf,
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::SwitchAdvantageNotMet,
        },
        // Partial RH telemetry should still return step-advantage hold reason.
        Case {
            label: "P5-PARTIAL-RH-TELEMETRY-PREFERS-STEP-ADV-HOLD",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_size_cap_estimate(1.2)
                .with_adams_step_cost_estimate(0.1)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(3)
                .with_bdf_cost_samples(3),
            probe_gate: Some(true),
            current: Lsode2MethodFamily::Bdf,
            expected_family: Lsode2MethodFamily::Bdf,
            expected_reason: Lsode2SwitchReason::SwitchAdvantageNotMet,
        },
        // Cost branch remains reachable when RH-based telemetry is absent.
        Case {
            label: "P6-COST-BRANCH-ONLY-WHEN-NO-RH-TELEMETRY",
            telemetry: Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_cost_estimate(0.1)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(3)
                .with_bdf_cost_samples(3),
            probe_gate: Some(true),
            current: Lsode2MethodFamily::Bdf,
            expected_family: Lsode2MethodFamily::Adams,
            expected_reason: Lsode2SwitchReason::CostPreferenceAdams,
        },
    ];

    for case in cases {
        let decision = config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
            case.telemetry,
            case.probe_gate,
            caps,
            Some(case.current),
        );
        assert_eq!(
            decision.preferred_family, case.expected_family,
            "{}: family mismatch",
            case.label
        );
        assert_eq!(
            decision.reason, case.expected_reason,
            "{}: reason mismatch",
            case.label
        );
    }
}

#[test]
fn lsoda_method_switch_handoff_preserves_history_and_step_counters_like_jstart_minus_one() {
    let mut state = Lsode2RuntimeState::new(
        0.0,
        &[1.0],
        0.1,
        5,
        Lsode2StepControlConfig {
            raise_order_after_accepts: 1,
            ..Lsode2StepControlConfig::default()
        },
    )
    .expect("runtime state should initialize");
    state
        .reconcile_first_nordsieck_derivative(&[0.1])
        .expect("initial derivative refresh should clear");

    let accepted = [
        (0.1, 1.1, 2usize),
        (0.2, 1.21, 3usize),
        (0.3, 1.331, 3usize),
    ];
    for (t, y, q) in accepted {
        state
            .accept_step(t, &[y], 1.0, 0.1, q)
            .expect("accepted setup step should succeed");
    }
    state.set_order(3).expect("test state order should be set");

    let before = state.snapshot();
    let history_before = (0..=3)
        .map(|age| state.y_history().block(age).unwrap()[0])
        .collect::<Vec<_>>();
    let nordsieck_before = (0..=3)
        .map(|order| state.nordsieck().col(order).unwrap()[0])
        .collect::<Vec<_>>();

    state
        .prepare_for_method_switch_handoff(12)
        .expect("BDF->Adams handoff should expand max order without cold restart");

    let after = state.snapshot();
    assert_eq!(
        after.accepted_steps, before.accepted_steps,
        "LSODA JSTART=-1 handoff must not reset NST-like accepted-step count"
    );
    assert_eq!(
        after.rejected_steps, before.rejected_steps,
        "LSODA JSTART=-1 handoff must not reset rejection counters"
    );
    assert_eq!(after.t, before.t);
    assert_eq!(after.h, before.h);
    assert_eq!(after.order, before.order);
    assert_eq!(after.max_order, 12);
    assert!(
        after.first_derivative_refresh_requested,
        "JSTART=-1 class handoff should force YH(:,2) refresh on next predictor"
    );
    for age in 0..=3 {
        assert_eq!(
            state.y_history().block(age).unwrap()[0],
            history_before[age],
            "raw solution history age {age} should survive method handoff"
        );
    }
    for order in 0..=3 {
        assert_eq!(
            state.nordsieck().col(order).unwrap()[0],
            nordsieck_before[order],
            "available Nordsieck column {order} should survive method handoff"
        );
    }
    assert_eq!(
        state.nordsieck().col(6).unwrap()[0],
        0.0,
        "newly exposed higher-order columns should start clean after expansion"
    );
}

#[test]
fn lsoda_method_switch_handoff_to_bdf_clamps_order_and_marks_matrix_stale() {
    let state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 12, Lsode2StepControlConfig::default())
        .expect("runtime state should initialize");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-6, 1.0e-9),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error control should initialize");
    let mut cycle =
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike);
    cycle
        .state_mut()
        .set_order(8)
        .expect("high Adams order should be legal before BDF switch");

    cycle
        .prepare_for_method_switch_handoff(Lsode2StepMethod::BdfLike, 5)
        .expect("Adams->BDF handoff should clamp to BDF MAXORD");

    assert_eq!(cycle.method(), Lsode2StepMethod::BdfLike);
    assert_eq!(cycle.state().order(), 5);
    assert_eq!(cycle.state().max_order(), 5);
    assert!(
        !cycle.jacobian_currency().is_current(),
        "Adams->BDF handoff should not carry an Adams-family current matrix into BDF"
    );
    assert!(
        cycle.state().first_derivative_refresh_requested(),
        "new BDF family should enter next predictor through JSTART=-1-style refresh"
    );
}

#[test]
fn lsoda_switch_state_records_mused_mcur_tsw_and_jstart_minus_one_on_real_switch() {
    let mut controller = Lsode2AlgorithmController::new_with_capabilities(
        Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        Lsode2ControllerExecutionCapabilities {
            adams_engine_available: true,
        },
    );
    let initial = controller.switch_state();
    assert_eq!(
        initial.mcur,
        Lsode2MethodFamily::Adams,
        "LSODA automatic mode starts in Adams/METH=1 when native Adams is available"
    );
    assert_eq!(initial.mused, Lsode2MethodFamily::Adams);
    assert_eq!(initial.tsw, None);
    assert_eq!(initial.last_handoff_jstart, None);

    let decision = controller.switch_decision_stateful(
        Lsode2SwitchTelemetry::default()
            .with_stiffness_ratio(1.0e6)
            .with_accepted_steps(1),
    );
    assert_eq!(decision.executed_family(), Some(Lsode2MethodFamily::Bdf));
    controller.record_switch_decision_at(decision, Some(0.125));

    let switched = controller.switch_state();
    assert_eq!(switched.mused, Lsode2MethodFamily::Adams);
    assert_eq!(switched.mcur, Lsode2MethodFamily::Bdf);
    assert_eq!(
        switched.tsw,
        Some(0.125),
        "LSODA driver sets TSW=TN when METH changes after an accepted step"
    );
    assert_eq!(
        switched.last_handoff_jstart,
        Some(-1),
        "Rust handoff visibility should expose the LSODA JSTART=-1 class setup"
    );

    let hold = controller.switch_decision_stateful(
        Lsode2SwitchTelemetry::default()
            .with_stiffness_ratio(1.0e6)
            .with_accepted_steps(2),
    );
    controller.record_switch_decision_at(hold, Some(0.25));
    let held = controller.switch_state();
    assert_eq!(held.mused, Lsode2MethodFamily::Bdf);
    assert_eq!(held.mcur, Lsode2MethodFamily::Bdf);
    assert_eq!(
        held.tsw,
        Some(0.125),
        "TSW should remain the previous switch time when no new method change occurs"
    );
    assert_eq!(
        held.last_handoff_jstart, None,
        "No-switch decision must not masquerade as a JSTART=-1 handoff"
    );
}

#[test]
fn lsoda_switch_probe_gate_and_tsw_ordering_survive_warmup_and_reset_windows() {
    let mut controller = Lsode2AlgorithmController::new_with_capabilities(
        Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(2),
        Lsode2ControllerExecutionCapabilities {
            adams_engine_available: true,
        },
    );

    assert_eq!(
        controller.switch_state().switch_probe_countdown(),
        2,
        "DSTODA-style ICOUNT starts at the configured probe window"
    );

    controller.record_accepted_steps_for_switch_probe(1);
    assert!(
        !controller.switch_state().switch_probe_ready(),
        "ICOUNT=2 should not permit a method-switch probe after only one accepted step"
    );
    let warmup = controller.switch_decision_stateful(
        Lsode2SwitchTelemetry::default()
            .with_accepted_steps(1)
            .with_adams_step_size_cap_estimate(0.1)
            .with_bdf_step_size_cap_estimate(1.0),
    );
    assert_eq!(warmup.reason, Lsode2SwitchReason::SwitchProbeWarmup);
    assert_eq!(warmup.executed_family(), Some(Lsode2MethodFamily::Adams));
    controller.record_switch_decision_at(warmup, Some(0.05));
    assert_eq!(controller.switch_state().mused, Lsode2MethodFamily::Adams);
    assert_eq!(controller.switch_state().mcur, Lsode2MethodFamily::Adams);
    assert_eq!(
        controller.switch_state().tsw,
        None,
        "TSW must not be populated by warmup/no-switch decisions"
    );
    assert_eq!(controller.switch_state().last_handoff_jstart, None);

    controller.record_accepted_steps_for_switch_probe(2);
    assert!(
        controller.switch_state().switch_probe_ready(),
        "Fortran ICOUNT semantics permit the first probe only after the window crosses below zero"
    );
    let switch_to_bdf = controller.switch_decision_stateful(
        Lsode2SwitchTelemetry::default()
            .with_accepted_steps(3)
            .with_adams_step_size_cap_estimate(0.1)
            .with_bdf_step_size_cap_estimate(1.0),
    );
    assert_eq!(switch_to_bdf.reason, Lsode2SwitchReason::StiffnessSuspected);
    assert_eq!(
        switch_to_bdf.executed_family(),
        Some(Lsode2MethodFamily::Bdf)
    );
    controller.record_switch_decision_at(switch_to_bdf, Some(0.30));
    let switched = controller.switch_state();
    assert_eq!(switched.mused, Lsode2MethodFamily::Adams);
    assert_eq!(switched.mcur, Lsode2MethodFamily::Bdf);
    assert_eq!(switched.tsw, Some(0.30));
    assert_eq!(switched.last_handoff_jstart, Some(-1));
    assert_eq!(
        switched.switch_probe_countdown(),
        2,
        "A real method switch resets the DSTODA ICOUNT probe window"
    );
    assert!(
        !switched.switch_probe_ready(),
        "A real method switch consumes and resets probe readiness"
    );

    let post_switch_warmup = controller.switch_decision_stateful(
        Lsode2SwitchTelemetry::default()
            .with_accepted_steps(4)
            .with_adams_step_size_cap_estimate(10.0)
            .with_bdf_step_size_cap_estimate(1.0),
    );
    assert_eq!(
        post_switch_warmup.reason,
        Lsode2SwitchReason::SwitchProbeWarmup
    );
    assert_eq!(
        post_switch_warmup.executed_family(),
        Some(Lsode2MethodFamily::Bdf)
    );
    controller.record_switch_decision_at(post_switch_warmup, Some(0.35));
    let held = controller.switch_state();
    assert_eq!(held.mused, Lsode2MethodFamily::Bdf);
    assert_eq!(held.mcur, Lsode2MethodFamily::Bdf);
    assert_eq!(
        held.tsw,
        Some(0.30),
        "TSW must remain the last real switch time through post-switch warmup holds"
    );
    assert_eq!(
        held.last_handoff_jstart, None,
        "post-switch warmup hold must not report a fresh JSTART=-1 handoff"
    );
}

#[test]
fn dstoda_label_by_label_matrix_terminal_near_terminal_replay_matches_fortran_style() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Row {
        label: &'static str,
        action: Lsode2RetryAction,
        kflag: Lsode2Kflag,
        icf: Lsode2Icf,
        iret: Lsode2Iret,
        iredo: Lsode2Iredo,
        ipup: Lsode2Ipup,
        ipup_trigger: Lsode2IpupTrigger,
    }

    fn row_from(label: &'static str, action: Lsode2RetryAction, cycle: &Lsode2StepCycle) -> Row {
        Row {
            label,
            action,
            kflag: cycle.kflag(),
            icf: cycle.icf(),
            iret: cycle.iret(),
            iredo: cycle.iredo(),
            ipup: cycle.ipup(),
            ipup_trigger: cycle.ipup_trigger(),
        }
    }

    let mut actual = Vec::new();

    // Label 410: stale-J same-step refresh request.
    let mut c_410_430 = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_convergence_failures: 2,
        ..Lsode2StepControlConfig::default()
    });
    c_410_430
        .state_mut()
        .set_step_size(0.8)
        .expect("step-size update should succeed");
    c_410_430
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");
    c_410_430.mark_jacobian_stale();
    let r410 = c_410_430
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("label-410 branch should succeed");
    actual.push(row_from("410", r410.action, &c_410_430));

    // Label 430 (nonterminal): retract/shrink after refresh did not recover.
    c_410_430.mark_jacobian_stale();
    let r430_retry = c_410_430
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("label-430 retry branch should succeed");
    actual.push(row_from("430-retry", r430_retry.action, &c_410_430));

    // Label 430 (terminal / MXNCF): repeated convergence failure terminal class.
    c_410_430.mark_jacobian_current();
    let r430_term = c_410_430
        .reject_after_nonlinear_failure()
        .expect("label-430 terminal branch should succeed");
    actual.push(row_from("430-term", r430_term.action, &c_410_430));

    // Label 500: first regular error-test reject.
    let mut c_500 = make_bdf_cycle_with_step_config(Lsode2StepControlConfig::default());
    c_500
        .state_mut()
        .set_step_size(1.0)
        .expect("step-size update should succeed");
    c_500
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");
    let o500 = c_500
        .finish_with_local_error(c_500.state().t(), &[1.0], &[1.1e-3])
        .expect("label-500 branch should succeed");
    let r500 = match o500 {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("label-500 expected rejected outcome, got {other:?}"),
    };
    actual.push(row_from("500", r500.action, &c_500));

    // Label 620: second consecutive error-test reject with RH<=0.2 clamp.
    let mut c_620 = make_bdf_cycle_with_step_config(Lsode2StepControlConfig::default());
    c_620
        .state_mut()
        .set_step_size(1.0)
        .expect("step-size update should succeed");
    c_620
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");
    c_620
        .state_mut()
        .reject_after_error_test_with_hint(0.8, 3)
        .expect("preload error-test reject should succeed");
    let h_before_620 = c_620.state().h();
    let o620 = c_620
        .finish_with_local_error(c_620.state().t(), &[1.0], &[1.1e-3])
        .expect("label-620 branch should succeed");
    let r620 = match o620 {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("label-620 expected rejected outcome, got {other:?}"),
    };
    assert!(
        r620.h_new.abs() <= h_before_620.abs() * 0.2 + 1.0e-300,
        "label-620 RH clamp parity violated: h_new={:e}, h_before={:e}",
        r620.h_new,
        h_before_620
    );
    actual.push(row_from("620", r620.action, &c_620));

    // Label 640 (nonterminal reset): repeated error-test reset with derivative refresh request.
    let mut c_640_reset = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_error_test_failures: 6,
        ..Lsode2StepControlConfig::default()
    });
    c_640_reset
        .state_mut()
        .set_step_size(1.0)
        .expect("step-size update should succeed");
    c_640_reset
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");
    c_640_reset
        .state_mut()
        .reject_after_error_test_with_hint(0.5, 3)
        .expect("preload #1 should succeed");
    c_640_reset
        .state_mut()
        .reject_after_error_test_with_hint(0.25, 3)
        .expect("preload #2 should succeed");
    let o640_reset = c_640_reset
        .finish_with_local_error(c_640_reset.state().t(), &[1.0], &[1.1e-3])
        .expect("label-640 reset branch should succeed");
    let r640_reset = match o640_reset {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("label-640 reset expected rejected outcome, got {other:?}"),
    };
    actual.push(row_from("640-reset", r640_reset.action, &c_640_reset));

    // Label 640 (terminal): repeated error-test terminal class.
    let mut c_640_term = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_error_test_failures: 3,
        ..Lsode2StepControlConfig::default()
    });
    c_640_term
        .state_mut()
        .set_step_size(1.0)
        .expect("step-size update should succeed");
    c_640_term
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");
    c_640_term
        .state_mut()
        .reject_after_error_test_with_hint(0.5, 3)
        .expect("preload #1 should succeed");
    c_640_term
        .state_mut()
        .reject_after_error_test_with_hint(0.25, 3)
        .expect("preload #2 should succeed");
    let o640_term = c_640_term
        .finish_with_local_error(c_640_term.state().t(), &[1.0], &[1.1e-3])
        .expect("label-640 terminal branch should succeed");
    let r640_term = match o640_term {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("label-640 terminal expected rejected outcome, got {other:?}"),
    };
    actual.push(row_from("640-term", r640_term.action, &c_640_term));

    // Label 670: HMIN terminal under convergence-failure class.
    let mut c_670 = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        h_min: 0.09,
        max_convergence_failures: 10,
        ..Lsode2StepControlConfig::default()
    });
    c_670
        .state_mut()
        .set_order(2)
        .expect("order update should succeed");
    c_670.mark_jacobian_current();
    let r670 = c_670
        .reject_after_nonlinear_failure()
        .expect("label-670 branch should succeed");
    actual.push(row_from("670", r670.action, &c_670));

    let expected = vec![
        Row {
            label: "410",
            action: Lsode2RetryAction::RetryWithJacobianRefresh,
            kflag: Lsode2Kflag::ConvergenceFailure,
            icf: Lsode2Icf::RefreshRequested,
            iret: Lsode2Iret::NormalFlow,
            iredo: Lsode2Iredo::CorrectorRefreshSameStep,
            ipup: Lsode2Ipup::NeedsJacobianUpdate,
            ipup_trigger: Lsode2IpupTrigger::FailurePath,
        },
        Row {
            label: "430-retry",
            action: Lsode2RetryAction::RetryWithJacobianRefresh,
            kflag: Lsode2Kflag::ConvergenceFailure,
            icf: Lsode2Icf::RefreshDidNotRecover,
            iret: Lsode2Iret::NormalFlow,
            iredo: Lsode2Iredo::CorrectorFailureRetry,
            ipup: Lsode2Ipup::NeedsJacobianUpdate,
            ipup_trigger: Lsode2IpupTrigger::FailurePath,
        },
        Row {
            label: "430-term",
            action: Lsode2RetryAction::FailRepeatedConvergenceFailures,
            kflag: Lsode2Kflag::RepeatedConvergenceFailure,
            icf: Lsode2Icf::RefreshDidNotRecover,
            iret: Lsode2Iret::NormalFlow,
            iredo: Lsode2Iredo::CorrectorFailureRetry,
            ipup: Lsode2Ipup::UpToDate,
            ipup_trigger: Lsode2IpupTrigger::None,
        },
        Row {
            label: "500",
            action: Lsode2RetryAction::Retry,
            kflag: Lsode2Kflag::ErrorTestFailure,
            icf: Lsode2Icf::None,
            iret: Lsode2Iret::RetryAfterErrorTestFailure,
            iredo: Lsode2Iredo::ErrorTestRetry,
            ipup: Lsode2Ipup::UpToDate,
            ipup_trigger: Lsode2IpupTrigger::None,
        },
        Row {
            label: "620",
            action: Lsode2RetryAction::Retry,
            kflag: Lsode2Kflag::ErrorTestFailure,
            icf: Lsode2Icf::None,
            iret: Lsode2Iret::RetryAfterErrorTestFailure,
            iredo: Lsode2Iredo::ErrorTestRetry,
            ipup: Lsode2Ipup::UpToDate,
            ipup_trigger: Lsode2IpupTrigger::None,
        },
        Row {
            label: "640-reset",
            action: Lsode2RetryAction::Retry,
            kflag: Lsode2Kflag::ErrorTestFailure,
            icf: Lsode2Icf::None,
            iret: Lsode2Iret::RestartWithDerivativeRefresh,
            iredo: Lsode2Iredo::RepeatedErrorReset,
            ipup: Lsode2Ipup::NeedsJacobianUpdate,
            ipup_trigger: Lsode2IpupTrigger::FailurePath,
        },
        Row {
            label: "640-term",
            action: Lsode2RetryAction::FailRepeatedErrorTestFailures,
            kflag: Lsode2Kflag::RepeatedErrorTestFailure,
            icf: Lsode2Icf::None,
            iret: Lsode2Iret::NormalFlow,
            iredo: Lsode2Iredo::None,
            ipup: Lsode2Ipup::UpToDate,
            ipup_trigger: Lsode2IpupTrigger::None,
        },
        Row {
            label: "670",
            action: Lsode2RetryAction::FailStepSizeUnderflow,
            kflag: Lsode2Kflag::RepeatedConvergenceFailure,
            icf: Lsode2Icf::RefreshDidNotRecover,
            iret: Lsode2Iret::NormalFlow,
            iredo: Lsode2Iredo::CorrectorFailureRetry,
            ipup: Lsode2Ipup::UpToDate,
            ipup_trigger: Lsode2IpupTrigger::None,
        },
    ];

    assert_eq!(actual, expected);
}

fn make_bdf_cycle_with_step_config(step_config: Lsode2StepControlConfig) -> Lsode2StepCycle {
    let state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 4, step_config)
        .expect("runtime state should initialize for parity micro cycle");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error control should initialize for parity micro cycle");
    Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::BdfLike)
}

#[test]
fn dstoda_label_670_hmin_terminal_preserves_convergence_terminal_flag_group() {
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        h_min: 0.09,
        max_convergence_failures: 10,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_order(2)
        .expect("order update should succeed");
    cycle.mark_jacobian_current();

    let terminal = cycle
        .reject_after_nonlinear_failure()
        .expect("HMIN branch should produce terminal retry");
    assert_eq!(terminal.action, Lsode2RetryAction::FailStepSizeUnderflow);
    assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
    assert_eq!(cycle.kflag_code(), -2);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
    assert_eq!(cycle.iret(), Lsode2Iret::NormalFlow);
    assert_eq!(cycle.iredo(), Lsode2Iredo::CorrectorFailureRetry);
    assert_eq!(cycle.iredo().code(), 1);

    // DSTODA label-670 parity: terminal HMIN exit happens before issuing
    // a new IPUP/JCUR refresh request.
    assert_eq!(cycle.ipup(), Lsode2Ipup::UpToDate);
    assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::None);
    assert_eq!(
        cycle.jacobian_update_request(),
        Lsode2JacobianUpdateRequest::None
    );
}

#[test]
fn dstoda_label_430_mxncf_terminal_preserves_convergence_terminal_flag_group() {
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_convergence_failures: 2,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_order(2)
        .expect("order update should succeed");

    // First pass: ICF=1 refresh request on stale-J path.
    let first = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("first failure should request same-step refresh");
    assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshRequested);
    assert_eq!(cycle.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
    assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
    assert_eq!(
        cycle.jacobian_update_request(),
        Lsode2JacobianUpdateRequest::Requested
    );

    // Second pass: retract/shrink path (`ICF=2`) and nonterminal retry.
    cycle.mark_jacobian_stale();
    let second = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("second failure should retract/shrink and keep retryable path");
    assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
    assert_eq!(cycle.iredo(), Lsode2Iredo::CorrectorFailureRetry);

    // Third pass: MXNCF terminal.
    cycle.mark_jacobian_current();
    let third = cycle
        .reject_after_nonlinear_failure()
        .expect("third failure should reach MXNCF-like terminal");
    assert_eq!(
        third.action,
        Lsode2RetryAction::FailRepeatedConvergenceFailures
    );
    assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
    assert_eq!(cycle.kflag_code(), -2);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
    assert_eq!(cycle.iret(), Lsode2Iret::NormalFlow);
    assert_eq!(cycle.iredo(), Lsode2Iredo::CorrectorFailureRetry);
    assert_eq!(cycle.iredo().code(), 1);

    // DSTODA label-430 parity: terminal MXNCF exit should not issue a new
    // Jacobian-refresh request (`IPUP=MITER`) on terminal step.
    assert_eq!(cycle.ipup(), Lsode2Ipup::UpToDate);
    assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::None);
    assert_eq!(
        cycle.jacobian_update_request(),
        Lsode2JacobianUpdateRequest::None
    );
}

#[test]
fn dstoda_msbp_and_mxncf_counter_replay_matches_fortran_style_failure_progression() {
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_convergence_failures: 2,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_order(2)
        .expect("order update should succeed");

    // Arm MSBP replay path exactly like DSTODA predictor gate:
    // mark current at NST=0 (NSLP=0), then advance accepted-step counter to NST=20.
    cycle.mark_jacobian_current();
    let mut t = cycle.state().t();
    for _ in 0..20 {
        t += cycle.state().h();
        cycle
            .state_mut()
            .accept_step(t, &[1.0], 1.0, 1.0e-6, 2)
            .expect("accept_step should advance NST for MSBP replay");
    }

    let mut stats = Lsode2NativeStatistics::default();

    let _ = cycle
        .predict()
        .expect("predict should succeed on MSBP replay");
    assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::PredictorMsbp);
    stats.record_predictor_ipup_trigger(cycle.ipup_trigger());
    stats.record_dstoda_flags(
        cycle.jacobian_currency(),
        cycle.ipup(),
        cycle.ipup_trigger(),
        cycle.kflag(),
        cycle.icf(),
        cycle.iret(),
        cycle.redo_stage(),
    );

    // 410: stale-J same-step refresh.
    let first = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("first failure should request same-step refresh");
    assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    stats.record_dstoda_flags(
        cycle.jacobian_currency(),
        cycle.ipup(),
        cycle.ipup_trigger(),
        cycle.kflag(),
        cycle.icf(),
        cycle.iret(),
        cycle.redo_stage(),
    );

    // 430 nonterminal: retract/shrink retry.
    cycle.mark_jacobian_stale();
    let second = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("second failure should retract and remain retryable");
    assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    stats.record_dstoda_flags(
        cycle.jacobian_currency(),
        cycle.ipup(),
        cycle.ipup_trigger(),
        cycle.kflag(),
        cycle.icf(),
        cycle.iret(),
        cycle.redo_stage(),
    );

    // 430 terminal (MXNCF class).
    cycle.mark_jacobian_current();
    let third = cycle
        .reject_after_nonlinear_failure()
        .expect("third failure should terminate by MXNCF class");
    assert_eq!(
        third.action,
        Lsode2RetryAction::FailRepeatedConvergenceFailures
    );
    stats.record_dstoda_flags(
        cycle.jacobian_currency(),
        cycle.ipup(),
        cycle.ipup_trigger(),
        cycle.kflag(),
        cycle.icf(),
        cycle.iret(),
        cycle.redo_stage(),
    );

    // Fortran-style counter expectations for this deterministic replay:
    // - one predictor-driven MSBP trigger,
    // - two KFLAG=-2 nonterminal convergence failures,
    // - one terminal repeated convergence failure class.
    assert_eq!(stats.native_predictor_ipup_trigger_predictor_msbp_count, 1);
    assert_eq!(stats.native_ipup_trigger_predictor_msbp_count, 1);
    assert_eq!(stats.native_kflag_convergence_failure_count, 2);
    assert_eq!(stats.native_kflag_repeated_convergence_failure_count, 1);
    assert_eq!(stats.native_icf_refresh_requested_count, 1);
    assert_eq!(stats.native_icf_refresh_did_not_recover_count, 2);
    assert_eq!(stats.native_ipup_trigger_failure_path_count, 2);
}

#[test]
fn dstoda_label_520_repeated_error_reset_replay_preserves_flag_choreography() {
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        h_min: 1.0e-12,
        max_error_test_failures: 10,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_step_size(1.0)
        .expect("step-size update should succeed");
    cycle
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");

    // Preload two consecutive error-test rejects so next attempt enters
    // DSTODA label-520 repeated-error reset choreography (`KFLAG <= -3`).
    cycle
        .state_mut()
        .reject_after_error_test_with_hint(0.5, 3)
        .expect("first preload reject should succeed");
    cycle
        .state_mut()
        .reject_after_error_test_with_hint(0.25, 3)
        .expect("second preload reject should succeed");

    let h_before = cycle.state().h();
    let outcome = cycle
        .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
        .expect("third reject should route through label-520 reset branch");

    let retry = match outcome {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("expected rejected outcome, got {other:?}"),
    };

    assert_eq!(retry.action, Lsode2RetryAction::Retry);
    assert_eq!(retry.order_new, 1);
    let expected_h =
        h_before * (cycle.state().step_control_config().h_min / h_before.abs()).max(0.1);
    assert_eq!(retry.h_new, expected_h);

    // DSTODA label-520 replay expectations:
    // - still error-test class (KFLAG=-1 group)
    // - restart-with-derivative-refresh intent (IRET)
    // - repeated-error reset redo marker (IREDO)
    // - Jacobian update requested through failure-path IPUP choreography.
    assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
    assert_eq!(cycle.kflag_code(), -1);
    assert_eq!(cycle.iret(), Lsode2Iret::RestartWithDerivativeRefresh);
    assert_eq!(cycle.redo_stage(), Lsode2RedoStage::RepeatedErrorReset);
    assert_eq!(cycle.iredo(), Lsode2Iredo::RepeatedErrorReset);
    assert_eq!(cycle.iredo().code(), 3);
    assert_eq!(cycle.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
    assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
    assert_eq!(
        cycle.jacobian_update_request(),
        Lsode2JacobianUpdateRequest::Requested
    );
    assert!(
        cycle.state().snapshot().first_derivative_refresh_requested,
        "label-520 reset should request first-derivative refresh for next retry"
    );
}

#[test]
fn dstoda_stiff_bdf_error_failure_500_620_640_full_trace_replays_fortran_style_flags_and_h() {
    // One-cycle replay slice for stiff BDF error-test choreography:
    // label-500 -> label-620 -> label-640 reset -> label-640 terminal.
    //
    // With tiny HMIN, label-640 reset path applies RH=0.1 deterministically.
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        h_min: 1.0e-14,
        max_error_test_failures: 4,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_step_size(1.0)
        .expect("step-size should set for stiff full-trace replay");
    cycle
        .state_mut()
        .set_order(3)
        .expect("order should set for stiff full-trace replay");

    let h0 = cycle.state().h();
    let t0 = cycle.state().t();

    // #1: label-500 class (regular retry).
    let o1 = cycle
        .finish_with_local_error(cycle.state().t(), &[1.0], &[1.1e-3])
        .expect("trace step #1 should evaluate");
    let r1 = match o1 {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("trace step #1 expected rejection, got {other:?}"),
    };
    let h1 = cycle.state().h();
    assert_eq!(r1.action, Lsode2RetryAction::Retry);
    assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
    assert_eq!(cycle.iret(), Lsode2Iret::RetryAfterErrorTestFailure);
    assert_eq!(cycle.iredo(), Lsode2Iredo::ErrorTestRetry);
    assert_eq!(cycle.state().order(), 3);
    assert_eq!(cycle.state().t(), t0);
    assert!(h1.abs() < h0.abs(), "label-500 should shrink H");

    // #2: label-620 class (KFLAG<=-2 clamp with RH<=0.2).
    let o2 = cycle
        .finish_with_local_error(cycle.state().t(), &[1.0], &[1.1e-3])
        .expect("trace step #2 should evaluate");
    let r2 = match o2 {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("trace step #2 expected rejection, got {other:?}"),
    };
    let h2 = cycle.state().h();
    assert_eq!(r2.action, Lsode2RetryAction::Retry);
    assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
    assert_eq!(cycle.iret(), Lsode2Iret::RetryAfterErrorTestFailure);
    assert_eq!(cycle.iredo(), Lsode2Iredo::ErrorTestRetry);
    assert!(
        h2.abs() <= h1.abs() * 0.2 + 1.0e-300,
        "label-620 clamp expected: h2={h2:e}, h1={h1:e}"
    );

    // #3: label-640 reset class (retry + derivative-refresh reset).
    let o3 = cycle
        .finish_with_local_error(cycle.state().t(), &[1.0], &[1.1e-3])
        .expect("trace step #3 should evaluate");
    let r3 = match o3 {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("trace step #3 expected rejection, got {other:?}"),
    };
    let h3 = cycle.state().h();
    assert_eq!(r3.action, Lsode2RetryAction::Retry);
    assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
    assert_eq!(cycle.iret(), Lsode2Iret::RestartWithDerivativeRefresh);
    assert_eq!(cycle.iredo(), Lsode2Iredo::RepeatedErrorReset);
    assert_eq!(
        cycle.state().order(),
        1,
        "label-640 reset should force NQ=1"
    );
    assert!(
        (h3.abs() - h2.abs() * 0.1).abs() <= (h2.abs() * 1.0e-12 + 1.0e-300),
        "label-640 reset should apply RH=0.1 with tiny HMIN: h3={h3:e}, h2={h2:e}"
    );

    // #4: label-640 terminal class (repeated error-test failure).
    let o4 = cycle
        .finish_with_local_error(cycle.state().t(), &[1.0], &[1.1e-3])
        .expect("trace step #4 should evaluate");
    let r4 = match o4 {
        Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
        other => panic!("trace step #4 expected rejection, got {other:?}"),
    };
    let h4 = cycle.state().h();
    assert_eq!(r4.action, Lsode2RetryAction::FailRepeatedErrorTestFailures);
    assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedErrorTestFailure);
    assert_eq!(cycle.iret(), Lsode2Iret::NormalFlow);
    assert_eq!(cycle.iredo(), Lsode2Iredo::None);
    assert_eq!(
        cycle.state().order(),
        1,
        "terminal repeated-error step should keep forced NQ=1"
    );
    assert!(
        (h4.abs() - h3.abs() * 0.1).abs() <= (h3.abs() * 1.0e-12 + 1.0e-300),
        "terminal label-640 path should still apply reset RH=0.1 on state: h4={h4:e}, h3={h3:e}"
    );
}

#[test]
fn dstoda_nhnil_mxhnil_counter_replay_tracks_tplusheqt_guard() {
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_hnil_warnings: 2,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_step_size(f64::EPSILON / 4.0)
        .expect("step-size update should succeed");

    // Force TN+H == TN in machine arithmetic.
    cycle
        .state_mut()
        .set_order(1)
        .expect("order update should succeed");
    let t = 1.0_f64;
    let y0 = cycle.state().y().to_vec();
    let max_order = cycle.state().max_order();
    let config = cycle.state().step_control_config();
    let mut state =
        Lsode2RuntimeState::new(t, y0.as_slice(), f64::EPSILON / 4.0, max_order, config)
            .expect("runtime state reinit for NHNIL replay");
    state
        .set_order(1)
        .expect("runtime state order update should succeed");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error control should initialize");
    let mut cycle =
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::BdfLike);

    cycle.predict().expect("first predict should succeed");
    cycle.predict().expect("second predict should succeed");
    cycle.predict().expect("third predict should succeed");

    let snapshot = cycle.state().snapshot();
    assert_eq!(snapshot.null_step_count, 3);
    assert_eq!(snapshot.null_step_warning_count, 2);
    assert_eq!(snapshot.null_step_warning_cap, 2);
    assert!(snapshot.null_step_warning_cap_reached);
}

#[test]
fn dstoda_label_520_full_trace_null_step_heavy_h_sequence_replay_matches_fortran_style_reset() {
    let h0 = f64::EPSILON / 4.0;
    let step_config = Lsode2StepControlConfig {
        h_min: 1.0e-300,
        max_error_test_failures: 6,
        max_hnil_warnings: 2,
        ..Lsode2StepControlConfig::default()
    };
    let state = Lsode2RuntimeState::new(1.0, &[1.0], h0, 4, step_config)
        .expect("runtime state should initialize for label-520 full-trace replay");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error control should initialize for label-520 full-trace replay");
    let mut cycle =
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::BdfLike);
    cycle
        .state_mut()
        .set_step_size(h0)
        .expect("step-size update should succeed");
    cycle
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");

    // Preload two error-test failures so the next reject enters the
    // `KFLAG <= -3` repeated-error reset branch (label-520/640 choreography).
    cycle
        .state_mut()
        .reject_after_error_test_with_hint(h0 * 0.5, 3)
        .expect("first preload reject should succeed");
    cycle
        .state_mut()
        .reject_after_error_test_with_hint(h0 * 0.25, 3)
        .expect("second preload reject should succeed");

    let expected_h_sequence = [
        h0 * 0.025,    // first label-520 reset from h0*0.25
        h0 * 0.0025,   // second repeated reset
        h0 * 0.00025,  // third repeated reset
        h0 * 0.000025, // terminal repeated-error failure step
    ];

    for (idx, expected_h) in expected_h_sequence.iter().enumerate() {
        // Null-step-heavy replay: with tiny H at T=1.0, predictor should hit
        // TN+H == TN and increment NHNIL telemetry every attempt.
        let predicted = cycle.predict().expect("predict should succeed");
        assert_eq!(
            predicted.t_trial,
            cycle.state().t(),
            "TN+H == TN parity expected in null-step-heavy replay"
        );

        let outcome = cycle
            .finish_with_local_error(cycle.state().t(), &[1.0], &[1.0e12])
            .expect("forced large local error should reject");

        let retry = match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
            other => panic!("expected rejected outcome, got {other:?}"),
        };

        let h_actual = cycle.state().h();
        let tol = expected_h.abs() * 1.0e-12 + 1.0e-300;
        assert!(
            (h_actual - expected_h).abs() <= tol,
            "unexpected H at replay step {idx}: got={h_actual:e}, expected={expected_h:e}, tol={tol:e}"
        );

        if idx + 1 < expected_h_sequence.len() {
            assert_eq!(
                retry.action,
                Lsode2RetryAction::Retry,
                "intermediate label-520 replay steps should remain retryable"
            );
            assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
            assert_eq!(cycle.iret(), Lsode2Iret::RestartWithDerivativeRefresh);
            assert_eq!(cycle.iredo(), Lsode2Iredo::RepeatedErrorReset);
            assert_eq!(cycle.redo_stage(), Lsode2RedoStage::RepeatedErrorReset);
            assert!(
                cycle.state().snapshot().first_derivative_refresh_requested,
                "label-520 reset should request first-derivative refresh on retryable steps"
            );
        } else {
            assert_eq!(
                retry.action,
                Lsode2RetryAction::FailRepeatedErrorTestFailures,
                "last replay step should hit terminal repeated-error class"
            );
            assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedErrorTestFailure);
            assert_eq!(cycle.kflag_code(), -1);
            assert!(
                !cycle.state().snapshot().first_derivative_refresh_requested,
                "terminal repeated-error step should not keep derivative-refresh request armed"
            );
        }
    }

    let snapshot = cycle.state().snapshot();
    assert_eq!(
        snapshot.null_step_count,
        expected_h_sequence.len(),
        "null-step replay should increment NHNIL once per predictor attempt"
    );
    assert_eq!(snapshot.null_step_warning_count, 2);
    assert_eq!(snapshot.null_step_warning_cap, 2);
    assert!(snapshot.null_step_warning_cap_reached);
}

#[test]
fn dstoda_label_520_full_trace_null_step_heavy_h_sequence_replay_matches_fortran_style_reset_for_adams()
 {
    let h0 = f64::EPSILON / 4.0;
    let step_config = Lsode2StepControlConfig {
        h_min: 1.0e-300,
        max_error_test_failures: 6,
        max_hnil_warnings: 2,
        ..Lsode2StepControlConfig::default()
    };
    let state = Lsode2RuntimeState::new(1.0, &[1.0], h0, 4, step_config)
        .expect("runtime state should initialize for label-520 full-trace replay (Adams)");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error control should initialize for label-520 full-trace replay (Adams)");
    let mut cycle =
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike);
    cycle
        .state_mut()
        .set_step_size(h0)
        .expect("step-size update should succeed");
    cycle
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");

    // Preload two error-test failures so the next reject enters the
    // `KFLAG <= -3` repeated-error reset branch (label-520/640 choreography).
    cycle
        .state_mut()
        .reject_after_error_test_with_hint(h0 * 0.5, 3)
        .expect("first preload reject should succeed");
    cycle
        .state_mut()
        .reject_after_error_test_with_hint(h0 * 0.25, 3)
        .expect("second preload reject should succeed");

    let expected_h_sequence = [
        h0 * 0.025,    // first label-520 reset from h0*0.25
        h0 * 0.0025,   // second repeated reset
        h0 * 0.00025,  // third repeated reset
        h0 * 0.000025, // terminal repeated-error failure step
    ];

    for (idx, expected_h) in expected_h_sequence.iter().enumerate() {
        // Null-step-heavy replay: with tiny H at T=1.0, predictor should hit
        // TN+H == TN and increment NHNIL telemetry every attempt.
        let predicted = cycle.predict().expect("predict should succeed");
        assert_eq!(
            predicted.t_trial,
            cycle.state().t(),
            "TN+H == TN parity expected in null-step-heavy replay"
        );

        let outcome = cycle
            .finish_with_local_error(cycle.state().t(), &[1.0], &[1.0e12])
            .expect("forced large local error should reject");

        let retry = match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, .. } => retry,
            other => panic!("expected rejected outcome, got {other:?}"),
        };

        let h_actual = cycle.state().h();
        let tol = expected_h.abs() * 1.0e-12 + 1.0e-300;
        assert!(
            (h_actual - expected_h).abs() <= tol,
            "unexpected H at replay step {idx}: got={h_actual:e}, expected={expected_h:e}, tol={tol:e}"
        );

        if idx + 1 < expected_h_sequence.len() {
            assert_eq!(
                retry.action,
                Lsode2RetryAction::Retry,
                "intermediate label-520 replay steps should remain retryable"
            );
            assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
            assert_eq!(cycle.iret(), Lsode2Iret::RestartWithDerivativeRefresh);
            assert_eq!(cycle.iredo(), Lsode2Iredo::RepeatedErrorReset);
            assert_eq!(cycle.redo_stage(), Lsode2RedoStage::RepeatedErrorReset);
            assert!(
                cycle.state().snapshot().first_derivative_refresh_requested,
                "label-520 reset should request first-derivative refresh on retryable steps"
            );
        } else {
            assert_eq!(
                retry.action,
                Lsode2RetryAction::FailRepeatedErrorTestFailures,
                "last replay step should hit terminal repeated-error class"
            );
            assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedErrorTestFailure);
            assert_eq!(cycle.kflag_code(), -1);
            assert!(
                !cycle.state().snapshot().first_derivative_refresh_requested,
                "terminal repeated-error step should not keep derivative-refresh request armed"
            );
        }
    }

    let snapshot = cycle.state().snapshot();
    assert_eq!(
        snapshot.null_step_count,
        expected_h_sequence.len(),
        "null-step replay should increment NHNIL once per predictor attempt"
    );
    assert_eq!(snapshot.null_step_warning_count, 2);
    assert_eq!(snapshot.null_step_warning_cap, 2);
    assert!(snapshot.null_step_warning_cap_reached);
}

#[test]
fn dstoda_icf1_icf2_mxncf_h_sequence_replay_matches_fortran_choreography() {
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_convergence_failures: 2,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_step_size(0.8)
        .expect("step-size update should succeed");
    cycle
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");

    // Fortran-like progression:
    // 1) ICF=1 same-step stale-J refresh, no H retract.
    let first = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("ICF=1 refresh path should succeed");
    assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    assert_eq!(first.h_new, 0.8);
    assert_eq!(first.order_new, 3);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshRequested);
    assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
    assert_eq!(cycle.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
    assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::FailurePath);

    // 2) ICF=2 retract path after stale refresh did not recover:
    //    shrink H and force NQ=1.
    cycle.mark_jacobian_stale();
    let second = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("ICF=2 retract path should succeed");
    assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    assert_eq!(second.h_new, 0.2);
    assert_eq!(second.order_new, 1);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
    assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
    assert_eq!(cycle.iredo(), Lsode2Iredo::CorrectorFailureRetry);

    // 3) MXNCF terminal on next convergence failure.
    cycle.mark_jacobian_current();
    let third = cycle
        .reject_after_nonlinear_failure()
        .expect("MXNCF terminal path should succeed");
    assert_eq!(
        third.action,
        Lsode2RetryAction::FailRepeatedConvergenceFailures
    );
    assert_eq!(third.h_new, 0.05);
    assert_eq!(third.order_new, 1);
    assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
    assert_eq!(cycle.iret(), Lsode2Iret::NormalFlow);
    assert_eq!(cycle.iredo(), Lsode2Iredo::CorrectorFailureRetry);

    // Terminal branch should not queue a new Jacobian refresh.
    assert_eq!(cycle.ipup(), Lsode2Ipup::UpToDate);
    assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::None);
    assert_eq!(
        cycle.jacobian_update_request(),
        Lsode2JacobianUpdateRequest::None
    );

    // Runtime state keeps final H/NQ consistent with terminal branch.
    let snapshot = cycle.state().snapshot();
    assert_eq!(snapshot.h, 0.05);
    assert_eq!(snapshot.order, 1);
}

#[test]
fn dstoda_icf1_icf2_retraction_replays_hscal_history_and_rc_stability() {
    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_convergence_failures: 3,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_step_size(0.8)
        .expect("step-size update should succeed");
    cycle
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");
    cycle
        .state_mut()
        .nordsieck_mut()
        .set_col(1, &[2.0])
        .expect("nordsieck col 1 should be writable");
    cycle
        .state_mut()
        .nordsieck_mut()
        .set_col(2, &[4.0])
        .expect("nordsieck col 2 should be writable");
    cycle
        .state_mut()
        .nordsieck_mut()
        .set_col(3, &[8.0])
        .expect("nordsieck col 3 should be writable");

    // Seed nontrivial RC and lock that stale-J retry/retraction choreography
    // does not mutate RC on its own.
    cycle.force_dstoda_coefficient_ratio_for_test(1.4);
    assert!((cycle.dstoda_state().coefficient_ratio() - 1.4).abs() < 1.0e-12);

    // ICF=1 path: same-step retry; history must stay unchanged.
    cycle.mark_jacobian_stale();
    let first = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("ICF=1 stale-J retry should succeed");
    assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    assert_eq!(first.h_new, 0.8);
    assert_eq!(first.order_new, 3);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshRequested);
    assert_eq!(cycle.state().nordsieck().col(1).unwrap(), &[2.0]);
    assert_eq!(cycle.state().nordsieck().col(2).unwrap(), &[4.0]);
    assert_eq!(cycle.state().nordsieck().col(3).unwrap(), &[8.0]);

    // ICF=2 path: retract/shrink to order=1.
    cycle.mark_jacobian_stale();
    let second = cycle
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("ICF=2 retract path should succeed");
    assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    assert_eq!(second.h_new, 0.2);
    assert_eq!(second.order_new, 1);
    assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);

    // HSCAL = HNEW/HOLD = 0.2/0.8 = 0.25:
    // YH(:,2) scales by 0.25, higher columns are cleared once order drops to 1.
    assert_eq!(cycle.state().nordsieck().col(1).unwrap(), &[0.5]);
    assert_eq!(cycle.state().nordsieck().col(2).unwrap(), &[0.0]);
    assert_eq!(cycle.state().nordsieck().col(3).unwrap(), &[0.0]);

    // RC remains finite and unchanged by retry/retract branch choreography itself.
    // (In Fortran this branch scales history/step, while RC updates are tied to
    // accepted-step coefficient updates and Jacobian-current resets.)
    let rc_after = cycle.dstoda_state().coefficient_ratio();
    assert!((rc_after - 1.4).abs() < 1.0e-12);
}

#[test]
fn dstoda_full_step_icf_ipup_410_430_progression_replays_fortran_flags_and_h_history() {
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct TraceRow {
        icf: Lsode2Icf,
        ipup: Lsode2Ipup,
        ipup_trigger: Lsode2IpupTrigger,
        kflag: Lsode2Kflag,
        h: f64,
        order: usize,
        action: Lsode2RetryAction,
        rejected_steps: usize,
        convergence_failures: usize,
    }

    let mut cycle = make_bdf_cycle_with_step_config(Lsode2StepControlConfig {
        max_convergence_failures: 2,
        ..Lsode2StepControlConfig::default()
    });
    cycle
        .state_mut()
        .set_step_size(0.8)
        .expect("step-size update should succeed");
    cycle
        .state_mut()
        .set_order(3)
        .expect("order update should succeed");
    let correction = Lsode2CorrectionController::scalar(
        1.0e-3,
        1.0e-6,
        Lsode2CorrectionControlConfig::default(),
    )
    .expect("build correction controller");

    let mut trace = Vec::new();

    // Attempt 1: stale Jacobian -> label-410 same-step refresh (ICF=1).
    cycle.mark_jacobian_stale();
    let predicted_1 = cycle.predict().expect("predict attempt #1 should succeed");
    let outcome_1 = cycle
        .finish_with_correction(
            predicted_1.t_trial,
            predicted_1.y_pred.as_slice(),
            &[1.0e-2],
            &[1.0e-2],
            &correction,
            Some(1.0e-4),
            Some(0.5),
            2,
        )
        .expect("attempt #1 should complete");
    let (action_1, state_1) = match outcome_1 {
        Lsode2StepCycleOutcome::NonlinearRejected { retry, state, .. } => (retry.action, state),
        other => panic!("attempt #1 expected nonlinear reject, got {other:?}"),
    };
    trace.push(TraceRow {
        icf: cycle.icf(),
        ipup: cycle.ipup(),
        ipup_trigger: cycle.ipup_trigger(),
        kflag: cycle.kflag(),
        h: cycle.state().h(),
        order: cycle.state().order(),
        action: action_1,
        rejected_steps: state_1.rejected_steps,
        convergence_failures: state_1.convergence_failures,
    });

    // Attempt 2: after refresh path was used, next failure -> label-430 retract (ICF=2).
    cycle.mark_jacobian_current();
    let predicted_2 = cycle.predict().expect("predict attempt #2 should succeed");
    let outcome_2 = cycle
        .finish_with_correction(
            predicted_2.t_trial,
            predicted_2.y_pred.as_slice(),
            &[1.0e-2],
            &[1.0e-2],
            &correction,
            Some(1.0e-4),
            Some(0.5),
            2,
        )
        .expect("attempt #2 should complete");
    let (action_2, state_2) = match outcome_2 {
        Lsode2StepCycleOutcome::NonlinearRejected { retry, state, .. } => (retry.action, state),
        other => panic!("attempt #2 expected nonlinear reject, got {other:?}"),
    };
    trace.push(TraceRow {
        icf: cycle.icf(),
        ipup: cycle.ipup(),
        ipup_trigger: cycle.ipup_trigger(),
        kflag: cycle.kflag(),
        h: cycle.state().h(),
        order: cycle.state().order(),
        action: action_2,
        rejected_steps: state_2.rejected_steps,
        convergence_failures: state_2.convergence_failures,
    });

    // Attempt 3: MXNCF terminal class (still label-430 family, terminal branch).
    cycle.mark_jacobian_current();
    let predicted_3 = cycle.predict().expect("predict attempt #3 should succeed");
    let outcome_3 = cycle
        .finish_with_correction(
            predicted_3.t_trial,
            predicted_3.y_pred.as_slice(),
            &[1.0e-2],
            &[1.0e-2],
            &correction,
            Some(1.0e-4),
            Some(0.5),
            2,
        )
        .expect("attempt #3 should complete");
    let (action_3, state_3) = match outcome_3 {
        Lsode2StepCycleOutcome::NonlinearRejected { retry, state, .. } => (retry.action, state),
        other => panic!("attempt #3 expected nonlinear reject, got {other:?}"),
    };
    trace.push(TraceRow {
        icf: cycle.icf(),
        ipup: cycle.ipup(),
        ipup_trigger: cycle.ipup_trigger(),
        kflag: cycle.kflag(),
        h: cycle.state().h(),
        order: cycle.state().order(),
        action: action_3,
        rejected_steps: state_3.rejected_steps,
        convergence_failures: state_3.convergence_failures,
    });

    let expected = [
        TraceRow {
            icf: Lsode2Icf::RefreshRequested,
            ipup: Lsode2Ipup::NeedsJacobianUpdate,
            ipup_trigger: Lsode2IpupTrigger::FailurePath,
            kflag: Lsode2Kflag::ConvergenceFailure,
            h: 0.8,
            order: 3,
            action: Lsode2RetryAction::RetryWithJacobianRefresh,
            rejected_steps: 0,
            convergence_failures: 0,
        },
        TraceRow {
            icf: Lsode2Icf::RefreshDidNotRecover,
            ipup: Lsode2Ipup::NeedsJacobianUpdate,
            ipup_trigger: Lsode2IpupTrigger::FailurePath,
            kflag: Lsode2Kflag::ConvergenceFailure,
            h: 0.2,
            order: 1,
            action: Lsode2RetryAction::RetryWithJacobianRefresh,
            rejected_steps: 1,
            convergence_failures: 1,
        },
        TraceRow {
            icf: Lsode2Icf::RefreshDidNotRecover,
            ipup: Lsode2Ipup::UpToDate,
            ipup_trigger: Lsode2IpupTrigger::None,
            kflag: Lsode2Kflag::RepeatedConvergenceFailure,
            h: 0.05,
            order: 1,
            action: Lsode2RetryAction::FailRepeatedConvergenceFailures,
            rejected_steps: 2,
            convergence_failures: 2,
        },
    ];

    assert_eq!(trace, expected);
}

#[test]
fn dstoda_lmax_nq_trace_replays_fortran_style_order_cap_history_for_bdf() {
    let mut state = Lsode2RuntimeState::new(
        0.0,
        &[1.0],
        0.1,
        2, // LMAX parity slice
        Lsode2StepControlConfig {
            raise_order_after_accepts: 1,
            ..Lsode2StepControlConfig::default()
        },
    )
    .expect("runtime state with LMAX=2 should initialize");
    let max_order = state.max_order();

    let mut order_trace = Vec::new();
    let mut t = 0.0_f64;
    let mut y = 1.0_f64;
    for _ in 0..6 {
        t += 0.1;
        y *= 0.9;
        let decision = state
            .accept_step(t, &[y], 1.0, 0.1, 2)
            .expect("accept_step should succeed");
        order_trace.push(decision.order_new);
    }

    // Fortran-style expectation for this deterministic setup:
    // q starts at 1 and quickly rises to the LMAX cap, then stays there.
    assert_eq!(order_trace, vec![1, 2, 2, 2, 2, 2]);
    assert!(
        order_trace.iter().all(|&q| q <= max_order),
        "NQ trace should remain capped by LMAX"
    );
}
