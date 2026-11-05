//! Fixed-mesh gates for the Cantera-style Damped numerical core.
//!
//! These tests intentionally exclude RST adaptive refinement so Newton-core
//! changes can be evaluated independently from mesh strategy behavior.

use super::*;
use crate::numerical::BVP_Damp::telemetry::BvpLogEventKind;
use nalgebra::DMatrix;
use std::collections::HashMap;

#[path = "cantera_candidate.rs"]
mod candidate;
#[path = "cantera_transient.rs"]
mod transient;

fn scalar_newton_control(residual: impl Fn(f64) -> f64 + Send + Sync + 'static, jac: f64) -> NRBVP {
    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("y")],
        DMatrix::from_element(1, 1, 0.1),
        vec!["y".into()],
        "x".into(),
        HashMap::from([("y".into(), vec![(0, 0.0)])]),
        0.0,
        1.0,
        1,
        DampedSolverOptions::dense_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-6),
    );
    solver.y = Box::new(DVector::from_element(1, 0.1));
    solver.bounds_vec = vec![(-100.0, 100.0)];
    solver.rel_tolerance_vec = vec![0.0];
    solver.fun = Box::new(FunEnum::Dense(Box::new(move |_, y| {
        DVector::from_element(1, residual(y[0]))
    })));
    solver.set_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    solver.set_bvp_logging_mode(BvpLoggingMode::Detailed);
    solver.factor_owner.old_jac = Some(Box::new(DMatrix::from_element(1, 1, jac)));
    solver
}

#[test]
fn fixed_jacobian_damping_trace_exposes_legacy_and_cantera_schedule_difference() {
    // F(y)=y^2-2, y0=0.1, fixed J=F'(y0)=0.2. The scale is constant
    // because rtol=0, so comparing |J^-1 F| also compares weighted norms.
    let base = 0.1_f64;
    let jac = 0.2_f64;
    let correction = (base * base - 2.0) / jac;
    let mut alpha = 1.0_f64;
    let mut reference_trials = Vec::new();
    for _ in 0..7 {
        reference_trials.push(alpha);
        let trial = base - alpha * correction;
        let next_correction = (trial * trial - 2.0) / jac;
        // Cantera's full-grid norm includes the eliminated zero BC correction.
        let s0 = correction.abs() / (1e-6 * 2.0_f64.sqrt());
        let s1 = next_correction.abs() / (1e-6 * 2.0_f64.sqrt());
        if s1 < 1.0 || s1 < s0 {
            break;
        }
        alpha /= 2.0_f64.sqrt();
    }
    assert_eq!(reference_trials.len(), 6);
    assert!((reference_trials[5] - 1.0 / (4.0 * 2.0_f64.sqrt())).abs() < 1e-14);

    let mut solver = scalar_newton_control(|y| y * y - 2.0, jac);
    let (status, candidate) = solver
        .try_damped_step()
        .expect("valid frozen-Jacobian step");
    assert_eq!(status, 0);
    let candidate = candidate.expect("the legacy schedule accepts a trial");
    let stats = solver.get_statistics().telemetry;
    let trials: Vec<f64> = stats
        .log_events
        .iter()
        .filter(|event| event.kind == BvpLogEventKind::DampingTrial)
        .map(|event| event.value_a)
        .collect();
    assert_eq!(trials.len(), 4);
    let legacy_alpha = 1.0 / (16.0 * 2.0_f64.sqrt());
    assert!(
        (trials[3] - legacy_alpha).abs() < 1e-14,
        "trials={trials:?}"
    );
    assert!((candidate.get_val(0) - (base - legacy_alpha * correction)).abs() < 1e-12);
    assert!((solver.y.get_val(0) - base).abs() < 1e-15);
    assert_eq!(stats.counters.linear_solves, 5);
    assert_eq!(stats.counters.damping_rejections, 3);
    // This is a pre-migration control, not a Cantera parity assertion.
    assert!((trials[3] - reference_trials[3]).abs() > 0.1);
}

#[test]
fn rejected_damping_trials_keep_the_base_iterate_and_report_exhaustion() {
    let mut solver = scalar_newton_control(|_| 1.0, 1.0);
    let (status, candidate) = solver
        .try_damped_step()
        .expect("rejections are not backend errors");
    assert_eq!(status, -2);
    assert!(candidate.is_none());
    assert!((solver.y.get_val(0) - 0.1).abs() < 1e-15);
    let stats = solver.get_statistics().telemetry;
    assert_eq!(stats.counters.damping_trials, 7);
    assert_eq!(stats.counters.damping_rejections, 7);
    assert_eq!(stats.counters.linear_solves, 8);
}

#[test]
fn outward_step_at_bound_stops_without_a_damping_trial_or_state_change() {
    let mut solver = scalar_newton_control(|_| 1.0, 1.0);
    solver.bounds_vec = vec![(0.1, 1.0)];
    let (status, candidate) = solver
        .try_damped_step()
        .expect("an inadmissible direction is a Newton outcome, not a backend error");
    assert_eq!(status, -3);
    assert!(candidate.is_none());
    assert!((solver.y.get_val(0) - 0.1).abs() < 1e-15);
    let stats = solver.get_statistics().telemetry;
    assert_eq!(stats.counters.damping_trials, 0);
    assert_eq!(stats.counters.linear_solves, 1);
}

#[test]
fn weighted_norm_distinguishes_small_correction_on_a_sensitive_component() {
    use crate::numerical::BVP_Damp::BVP_utils_damped::weighted_norm_reduced_unknowns;

    let state = DVector::from_vec(vec![1e6, 1e-6]);
    let large_absolute_step = DVector::from_vec(vec![1.0, 0.0]);
    let small_absolute_step = DVector::from_vec(vec![0.0, 1e-6]);
    let boundaries = [(2, 0, 0.0), (3, 1, 0.0)];
    let mut scratch = [0.0; 2];
    let norm = |step: &DVector<f64>, scratch: &mut [f64; 2]| {
        weighted_norm_reduced_unknowns(
            &state,
            step,
            &[1e-3, 1e-3],
            1e-9,
            2,
            2,
            &[0, 1],
            &boundaries,
            scratch,
        )
        .unwrap()
    };
    let large_scaled = norm(&large_absolute_step, &mut scratch);
    let small_scaled = norm(&small_absolute_step, &mut scratch);
    assert!(large_absolute_step.norm() > small_absolute_step.norm());
    assert!(
        large_scaled < 1.0,
        "scaled large correction: {large_scaled}"
    );
    assert!(
        small_scaled > 1.0,
        "scaled small correction: {small_scaled}"
    );
}

#[test]
fn fixed_mesh_constant_state_problem_converges_to_the_exact_profile() {
    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("0")],
        DMatrix::from_element(1, 4, 0.25),
        vec!["y".to_string()],
        "x".to_string(),
        HashMap::from([("y".to_string(), vec![(0, 1.0)])]),
        0.0,
        1.0,
        4,
        DampedSolverOptions::dense_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-10)
            .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-10)]))
            .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
            .with_max_iterations(10),
    );
    solver.set_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    solver
        .try_eq_generate(None, None)
        .expect("the fixed-mesh symbolic callbacks should prepare");

    let outcome = solver
        .try_solver()
        .expect("the fixed-mesh Newton solve should complete without infrastructure errors");
    assert!(
        outcome.is_some(),
        "the exact constant profile should converge"
    );

    let profile = solver
        .get_result()
        .expect("a converged fixed-mesh profile should be published");
    assert_eq!(
        profile.nrows(),
        5,
        "the original four intervals remain active"
    );
    assert_eq!(profile.ncols(), 1);
    for value in profile.iter() {
        assert!(
            (*value - 1.0).abs() <= 1e-10,
            "expected y(x)=1 on the full mesh, got {value:.16e}"
        );
    }

    let residual = solver
        .fun
        .try_call(solver.p, &*solver.y)
        .expect("the final discrete residual callback should succeed");
    assert!(residual.norm() <= 1e-10, "final residual was {residual:?}");

    let stats = solver.get_statistics();
    assert_eq!(stats.telemetry.counters.grid_refinements, 0);
    assert_eq!(solver.x_mesh.len(), 5);
}

#[test]
fn fixed_mesh_linear_bvp_requires_and_accepts_a_newton_correction() {
    const INTERVALS: usize = 16;
    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("y")],
        DMatrix::from_element(1, INTERVALS, 1.25),
        vec!["y".to_string()],
        "x".to_string(),
        HashMap::from([("y".to_string(), vec![(0, 1.0)])]),
        0.0,
        1.0,
        INTERVALS,
        DampedSolverOptions::dense_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-9)
            .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-9)]))
            .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
            .with_max_iterations(10),
    );
    solver.set_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    solver
        .try_eq_generate(None, None)
        .expect("the fixed-grid symbolic callbacks should prepare");

    let outcome = solver
        .try_solver()
        .expect("the fixed-grid Newton solve should not fail at the backend level");
    assert!(outcome.is_some(), "the linear BVP should converge");

    let profile = solver
        .get_result()
        .expect("the converged fixed-grid profile should be published");
    assert_eq!(profile.nrows(), INTERVALS + 1);
    assert_eq!(profile.ncols(), 1);
    assert!((profile[(0, 0)] - 1.0).abs() <= 1e-12);
    for pair in profile.column(0).as_slice().windows(2) {
        let dx = 1.0 / INTERVALS as f64;
        let discrete_rhs = (pair[1] - pair[0]) / dx;
        assert!(
            (discrete_rhs - pair[0]).abs() <= 1e-8,
            "fixed-grid residual is too large: y_i={:.12e}, y_(i+1)={:.12e}, residual={:.12e}",
            pair[0],
            pair[1],
            discrete_rhs - pair[0]
        );
    }

    let stats = solver.get_statistics();
    assert!(stats.telemetry.counters.iterations > 0);
    assert!(stats.telemetry.counters.linear_solves > 0);
    assert_eq!(stats.telemetry.counters.grid_refinements, 0);
    assert_eq!(solver.x_mesh.len(), INTERVALS + 1);
}

#[test]
#[ignore = "fixed-mesh Newton trace for the parser regression"]
fn fixed_mesh_parser_fixture_traces_newton_outcome() {
    let mut solver = NRBVP::new_with_options(
        vec![
            Expr::parse_expression("y-z"),
            Expr::parse_expression("-z^3"),
        ],
        DMatrix::zeros(2, 10),
        vec!["z".into(), "y".into()],
        "x".into(),
        HashMap::from([("z".into(), vec![(0, 1.0)]), ("y".into(), vec![(1, 1.0)])]),
        0.0,
        1.0,
        10,
        DampedSolverOptions::dense_damped()
            .with_strategy_params(Some(SolverParams {
                max_jac: Some(3),
                max_damp_iter: Some(10),
                ..SolverParams::default()
            }))
            .with_abs_tolerance(1e-5)
            .with_rel_tolerance(HashMap::from([("z".into(), 1e-4), ("y".into(), 1e-4)]))
            .with_bounds(HashMap::from([
                ("z".into(), (-10.0, 10.0)),
                ("y".into(), (-7.0, 7.0)),
            ]))
            .with_max_iterations(100),
    );
    solver.set_bvp_telemetry_mode(BvpTelemetryMode::Detailed);
    solver.set_bvp_logging_mode(BvpLoggingMode::Detailed);
    solver.try_eq_generate(None, None).unwrap();
    let result = solver.try_solver();
    let stats = solver.get_statistics();
    println!(
        "result={:?} counters={:?}",
        result.as_ref().map(|v| v.is_some()),
        stats.telemetry.counters
    );
    if matches!(&result, Ok(Some(_))) {
        println!(
            "final residual norm={:.16e}",
            solver.try_calc_residual(solver.y.clone_box()).unwrap()
        );
    }
    for event in &stats.telemetry.log_events {
        println!("{event:?}");
    }
    assert!(
        result.is_ok(),
        "the diagnostic should not fail at the callback boundary"
    );
}
