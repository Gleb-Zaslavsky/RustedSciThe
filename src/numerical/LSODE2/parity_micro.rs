use crate::numerical::LSODE2::{
    Lsode2CorrectionAssessment, Lsode2CorrectionController, Lsode2CorrectionControlConfig,
    Lsode2CorrectionStatus, Lsode2ErrorControlConfig, Lsode2ErrorController,
     Lsode2RuntimeState, Lsode2StepControlConfig,
    Lsode2StepCycle,  Lsode2Tolerance,
};
use crate::numerical::LSODE2::step_cycle::Lsode2StepMethod;
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
            &[1.0],                 // weight source (y scale)
            &[1.0e-2],              // correction (current)
            &[1.0e-2],              // accumulated
            prev_norm,
            Some(10.0),            // previous_rate_estimate (not used here)
            2,                     // iteration >=2 to enable DEL/DELP check
            None,
            None,
        )
        .expect("assessment should run");

    assert_eq!(assessment.status, Lsode2CorrectionStatus::Diverged);
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

    let mut cycle = Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike);

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
    let decision = cycle.select_post_accept_order(&[0.905], 1.0, None).expect("select order");

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
        assert!(decision.suggested_growth <= expected_limit + tol || decision.suggested_growth <= 1.0 + tol,
            "suggested_growth should respect SM1/PDH limit (got {} <= {}); decision={:?}",
            decision.suggested_growth,
            expected_limit,
            decision
        );
    }

    // PDEST should be cleared after RH selection; PDLAST should remain
    assert_eq!(cycle.adams_pdest(), 0.0, "PDEST should be cleared after RH selection");
    assert_eq!(cycle.adams_pdlast(), 5.0, "PDLAST should retain last nonzero estimate");
}
