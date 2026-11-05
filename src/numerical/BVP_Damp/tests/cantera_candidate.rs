//! Fixed-grid, test-only Cantera-style local Newton candidate.
//!
//! Uses the actual RST residual/Jacobian callbacks and factor owner. This is
//! not the Cantera implementation, a transient controller, or a public solver.

use super::*;
use crate::numerical::BVP_Damp::BVP_utils_damped::{
    convergence_condition, weighted_norm_reduced_unknowns,
};

#[path = "cantera_recovery.rs"]
mod recovery;

#[derive(Debug)]
struct CandidateReport {
    status: i32,
    iterations: usize,
    jacobians: usize,
    trials: usize,
    linear_solves: usize,
    residual_norm: f64,
    state: DVector<f64>,
    accepted_factors: Vec<f64>,
    trial_trace: Vec<(f64, f64, f64, f64, bool)>,
}

#[derive(Clone, Copy)]
enum TrialSchedule {
    Cantera,
    Legacy,
}

#[derive(Clone, Copy)]
enum StepNorm {
    CanteraWeightedRms,
    RstEuclidean,
}

fn weighted_norm(
    solver: &NRBVP,
    state: &dyn VectorType,
    correction: &dyn VectorType,
    positions: &[usize],
    scratch: &mut [f64],
) -> Result<f64, String> {
    weighted_norm_reduced_unknowns(
        state,
        correction,
        &solver.rel_tolerance_vec,
        solver.abs_tolerance,
        solver.values.len(),
        solver.x_mesh.len(),
        positions,
        &solver.BC_position_and_value,
        scratch,
    )
    .ok_or_else(|| "invalid or non-finite weighted norm input".to_string())
}

fn solve_fixed_mesh(solver: &mut NRBVP) -> Result<CandidateReport, String> {
    solve_fixed_mesh_with_controls(solver, TrialSchedule::Cantera, StepNorm::CanteraWeightedRms)
}

fn solve_fixed_mesh_with_schedule(
    solver: &mut NRBVP,
    schedule: TrialSchedule,
) -> Result<CandidateReport, String> {
    solve_fixed_mesh_with_controls(solver, schedule, StepNorm::CanteraWeightedRms)
}

fn solve_fixed_mesh_with_controls(
    solver: &mut NRBVP,
    schedule: TrialSchedule,
    step_norm: StepNorm,
) -> Result<CandidateReport, String> {
    let packed = DVector::from_column_slice(solver.initial_guess.as_slice());
    solver.y = Vectors_type_casting(&packed, solver.effective_runtime_method());
    let full_len = solver.x_mesh.len() * solver.values.len();
    let mut fixed = vec![false; full_len];
    for &(position, _, _) in &solver.BC_position_and_value {
        if position >= full_len || fixed[position] {
            return Err("invalid boundary position in candidate fixture".into());
        }
        fixed[position] = true;
    }
    let positions: Vec<usize> = (0..full_len).filter(|&position| !fixed[position]).collect();
    if positions.len() != solver.y.len() {
        return Err("candidate fixture has inconsistent packed layout".into());
    }
    let mut scratch = vec![0.0; solver.values.len()];
    let max_age = solver
        .strategy_params
        .as_ref()
        .and_then(|p| p.max_jac)
        .unwrap_or(5);
    let max_trials = solver
        .strategy_params
        .as_ref()
        .and_then(|p| p.max_damp_iter)
        .unwrap_or(7);
    if max_trials == 0 {
        return Err("candidate requires at least one damping trial".into());
    }
    let legacy_exponent = solver
        .strategy_params
        .as_ref()
        .and_then(|p| p.damp_factor)
        .unwrap_or(0.5);
    let mut age = 0_usize;
    let mut force_jacobian = true;
    let mut failed_refreshes = 0_usize;
    let mut report = CandidateReport {
        status: -4,
        iterations: 0,
        jacobians: 0,
        trials: 0,
        linear_solves: 0,
        residual_norm: f64::NAN,
        state: packed,
        accepted_factors: Vec::new(),
        trial_trace: Vec::new(),
    };

    for _ in 0..solver.max_iterations {
        if age > max_age {
            force_jacobian = true;
        }
        if force_jacobian {
            solver.jac_recalc = true;
            solver
                .try_recalc_jacobian()
                .map_err(|error| format!("Jacobian: {error:?}"))?;
            report.jacobians += 1;
            age = 0;
            force_jacobian = false;
        }
        let (correction, _, _, _) = solver
            .try_step_with_linear_telemetry(solver.p, &*solver.y)
            .map_err(|error| format!("base correction: {error:?}"))?;
        report.iterations += 1;
        report.linear_solves += 1;
        age += 1;
        let s0 = match step_norm {
            StepNorm::CanteraWeightedRms => {
                weighted_norm(solver, &*solver.y, &*correction, &positions, &mut scratch)?
            }
            StepNorm::RstEuclidean => correction.norm(),
        };
        let (mut alpha, _) = bound_step_Cantera2_with_update_sign_and_limit(
            &*solver.y,
            &*correction,
            &solver.bounds_vec,
            -1.0,
        );
        if !alpha.is_finite() {
            return Err("non-finite bound factor".into());
        }

        let mut status = -3;
        let mut accepted = None;
        if alpha >= 1e-10 {
            status = -2;
            for trial_index in 0..max_trials {
                let scaled = correction.mul_float(alpha);
                let trial = &*solver.y - &*scaled;
                let (next, _, _, _) = solver
                    .try_step_with_linear_telemetry(solver.p, &*trial)
                    .map_err(|error| format!("trial correction: {error:?}"))?;
                report.trials += 1;
                report.linear_solves += 1;
                let s1 = match step_norm {
                    StepNorm::CanteraWeightedRms => {
                        weighted_norm(solver, &*trial, &*next, &positions, &mut scratch)?
                    }
                    StepNorm::RstEuclidean => next.norm(),
                };
                let limit = match step_norm {
                    StepNorm::CanteraWeightedRms => 1.0,
                    StepNorm::RstEuclidean => convergence_condition(
                        &*trial,
                        &solver.abs_tolerance,
                        &solver.rel_tolerance_vec,
                    ),
                };
                let accepted_trial = s1 < limit || s1 < s0;
                report
                    .trial_trace
                    .push((alpha, s0, s1, limit, accepted_trial));
                if accepted_trial {
                    status = if s1 > limit { 0 } else { 1 };
                    accepted = Some(trial);
                    report.accepted_factors.push(alpha);
                    break;
                }
                alpha /= match schedule {
                    TrialSchedule::Cantera => 2.0_f64.sqrt(),
                    TrialSchedule::Legacy => 2.0_f64.powf(trial_index as f64 + legacy_exponent),
                };
            }
        }

        if let Some(trial) = accepted {
            solver.y = trial;
            if status == 1 {
                report.status = 1;
                break;
            }
        } else if age > 1 && failed_refreshes <= 3 {
            force_jacobian = true;
            failed_refreshes += 1;
        } else {
            report.status = status;
            break;
        }
    }
    report.state = solver.y.to_DVectorType();
    report.residual_norm = solver
        .try_calc_residual(solver.y.clone_box())
        .map_err(|error| format!("final residual: {error:?}"))?;
    Ok(report)
}

fn linear_fixture() -> NRBVP {
    NRBVP::new_with_options(
        vec![Expr::parse_expression("y")],
        DMatrix::from_element(1, 16, 1.25),
        vec!["y".into()],
        "x".into(),
        HashMap::from([("y".into(), vec![(0, 1.0)])]),
        0.0,
        1.0,
        16,
        DampedSolverOptions::dense_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-9)
            .with_rel_tolerance(HashMap::from([("y".into(), 1e-9)]))
            .with_bounds(HashMap::from([("y".into(), (-10.0, 10.0))]))
            .with_max_iterations(10),
    )
}

#[test]
fn fixed_mesh_linear_candidate_and_production_match_discrete_solution() {
    let mut legacy = linear_fixture();
    legacy.set_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    legacy.try_eq_generate(None, None).unwrap();
    assert!(legacy.try_solver().unwrap().is_some());
    let legacy_state = legacy.y.to_DVectorType();

    let mut candidate = linear_fixture();
    candidate.try_eq_generate(None, None).unwrap();
    let report = solve_fixed_mesh(&mut candidate).unwrap();
    println!("Cantera-style local candidate: {report:?}");
    assert_eq!(report.status, 1);
    assert!(report.residual_norm < 1e-9);
    assert!((&legacy_state - &report.state).amax() < 1e-9);
    assert_eq!(report.accepted_factors, [1.0]);
}

fn nonlinear_fixture() -> NRBVP {
    NRBVP::new_with_options(
        vec![Expr::parse_expression("0 - (y*y)")],
        DMatrix::from_element(1, 20, 0.8),
        vec!["y".into()],
        "x".into(),
        HashMap::from([("y".into(), vec![(0, 1.0)])]),
        0.0,
        1.0,
        20,
        DampedSolverOptions::dense_damped()
            .with_scheme_name("trapezoid")
            .with_strategy_params(Some(SolverParams {
                max_jac: Some(5),
                max_damp_iter: Some(7),
                ..SolverParams::default()
            }))
            .with_abs_tolerance(1e-9)
            .with_rel_tolerance(HashMap::from([("y".into(), 1e-9)]))
            .with_bounds(HashMap::from([("y".into(), (0.0, 3.0))]))
            .with_max_iterations(30),
    )
}

#[test]
fn fixed_mesh_nonlinear_candidate_and_production_match_manufactured_solution() {
    let mut legacy = nonlinear_fixture();
    let mut candidate = nonlinear_fixture();
    legacy.set_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    legacy.try_eq_generate(None, None).unwrap();
    candidate.try_eq_generate(None, None).unwrap();
    let constant_probe = DVector::from_element(20, 1.0);
    let probe_residual = legacy
        .fun
        .try_call(legacy.p, &constant_probe)
        .unwrap()
        .to_DVectorType();
    assert!(
        probe_residual
            .iter()
            .all(|value| (*value - 0.05).abs() < 1e-12)
    );
    let start = DVector::from_column_slice(legacy.initial_guess.as_slice());
    let legacy_f = legacy
        .fun
        .try_call(legacy.p, &start)
        .unwrap()
        .to_DVectorType();
    let candidate_f = candidate
        .fun
        .try_call(candidate.p, &start)
        .unwrap()
        .to_DVectorType();
    assert!((&legacy_f - &candidate_f).amax() < 1e-14);
    let legacy_j = legacy
        .jac
        .as_mut()
        .unwrap()
        .try_call(legacy.p, &start)
        .unwrap()
        .to_DMatrixType();
    let candidate_j = candidate
        .jac
        .as_mut()
        .unwrap()
        .try_call(candidate.p, &start)
        .unwrap()
        .to_DMatrixType();
    assert!((&legacy_j - &candidate_j).amax() < 1e-14);
    let legacy_result = legacy.try_solver().unwrap();
    let legacy_state = legacy.y.to_DVectorType();
    let report = solve_fixed_mesh(&mut candidate).unwrap();
    println!("nonlinear candidate={report:?}");
    println!(
        "nonlinear legacy counters={:?}",
        legacy.get_statistics().telemetry.counters
    );
    assert!(
        legacy_result.is_some(),
        "legacy nonlinear fixture did not converge"
    );
    assert_eq!(report.status, 1);
    assert!(report.residual_norm < 1e-8);
    assert!((&legacy_state - &report.state).amax() < 1e-7);
    for (index, value) in report.state.iter().enumerate() {
        let x = (index + 1) as f64 / 20.0;
        let exact = 1.0 / (1.0 + x);
        assert!(
            (value - exact).abs() < 2e-4,
            "x={x} value={value} exact={exact}"
        );
    }
}

#[test]
fn fixed_mesh_candidate_retries_stale_jacobian_then_reports_blocked_bound() {
    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("1")],
        DMatrix::zeros(1, 2),
        vec!["y".into()],
        "x".into(),
        HashMap::from([("y".into(), vec![(0, 0.0)])]),
        0.0,
        1.0,
        2,
        DampedSolverOptions::dense_damped()
            .with_strategy_params(Some(SolverParams {
                max_jac: Some(5),
                max_damp_iter: Some(7),
                ..SolverParams::default()
            }))
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(HashMap::from([("y".into(), 1e-8)]))
            .with_bounds(HashMap::from([("y".into(), (0.0, 0.5))]))
            .with_max_iterations(10),
    );
    solver.try_eq_generate(None, None).unwrap();
    let report = solve_fixed_mesh(&mut solver).unwrap();
    assert_eq!(report.status, -3);
    assert_eq!(report.iterations, 3);
    assert_eq!(report.jacobians, 2);
    assert_eq!(report.trials, 1);
    assert_eq!(report.linear_solves, 4);
    assert!((report.state[0] - 0.25).abs() < 1e-12);
    assert!((report.state[1] - 0.5).abs() < 1e-12);
    assert!((report.residual_norm - 0.5 / 2.0_f64.sqrt()).abs() < 1e-12);
    assert!(
        solver.result.is_none(),
        "a blocked candidate must not publish a profile"
    );
}

fn parser_fixture() -> NRBVP {
    NRBVP::new_with_options(
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
                max_jac: Some(5),
                max_damp_iter: Some(7),
                ..SolverParams::default()
            }))
            .with_abs_tolerance(1e-5)
            .with_rel_tolerance(HashMap::from([("z".into(), 1e-4), ("y".into(), 1e-4)]))
            .with_bounds(HashMap::from([
                ("z".into(), (-10.0, 10.0)),
                ("y".into(), (-7.0, 7.0)),
            ]))
            .with_max_iterations(100),
    )
}

fn large_profile_metrics(solver: &NRBVP, packed: &DVector<f64>) -> (f64, f64, f64) {
    let full = construct_full_solution(packed.clone(), solver.BC_position_and_value.clone());
    let mut min_species = f64::INFINITY;
    let mut sum_error = 0.0_f64;
    let mut max_temperature = f64::NEG_INFINITY;
    for node in full.as_slice().chunks_exact(26) {
        max_temperature = max_temperature.max(1200.0 + 300.0 * node[0]);
        let species = (2..26).step_by(2).map(|index| node[index]);
        let mut sum = 0.0;
        for value in species {
            min_species = min_species.min(value);
            sum += value;
        }
        sum_error = sum_error.max((sum - 1.0).abs());
    }
    (min_species, sum_error, max_temperature)
}

#[test]
fn fixed_mesh_parser_ab_uses_identical_callbacks_and_reports_both_outcomes() {
    let mut legacy = parser_fixture();
    let mut candidate = parser_fixture();
    legacy.set_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    legacy.try_eq_generate(None, None).unwrap();
    candidate.try_eq_generate(None, None).unwrap();
    let start = DVector::from_column_slice(legacy.initial_guess.as_slice());
    let legacy_f = legacy
        .fun
        .try_call(legacy.p, &start)
        .unwrap()
        .to_DVectorType();
    let candidate_f = candidate
        .fun
        .try_call(candidate.p, &start)
        .unwrap()
        .to_DVectorType();
    assert!((&legacy_f - &candidate_f).amax() < 1e-14);
    let legacy_j = legacy
        .jac
        .as_mut()
        .unwrap()
        .try_call(legacy.p, &start)
        .unwrap()
        .to_DMatrixType();
    let candidate_j = candidate
        .jac
        .as_mut()
        .unwrap()
        .try_call(candidate.p, &start)
        .unwrap()
        .to_DMatrixType();
    assert!((&legacy_j - &candidate_j).amax() < 1e-14);

    let legacy_result = legacy.try_solver().unwrap();
    let legacy_residual = legacy.try_calc_residual(legacy.y.clone_box()).unwrap();
    let legacy_stats = legacy.get_statistics().telemetry.counters;
    let final_state = legacy.y.to_DVectorType();
    let final_residual = legacy
        .fun
        .try_call(legacy.p, &final_state)
        .unwrap()
        .to_DVectorType();
    let final_jacobian = legacy
        .jac
        .as_mut()
        .unwrap()
        .try_call(legacy.p, &final_state)
        .unwrap()
        .to_DMatrixType();
    let fresh_correction = final_jacobian
        .clone()
        .lu()
        .solve(&final_residual)
        .expect("independent dense LU should solve the final linearization");
    let linear_error = (&final_jacobian * &fresh_correction - &final_residual).norm();
    assert!(
        linear_error < 1e-10,
        "fresh linear solve residual={linear_error:e}"
    );
    let direction = &fresh_correction / fresh_correction.norm();
    let epsilon = 1e-6;
    let plus = &final_state + epsilon * &direction;
    let minus = &final_state - epsilon * &direction;
    let f_plus = legacy
        .fun
        .try_call(legacy.p, &plus)
        .unwrap()
        .to_DVectorType();
    let f_minus = legacy
        .fun
        .try_call(legacy.p, &minus)
        .unwrap()
        .to_DVectorType();
    let directional_fd = (f_plus - f_minus) / (2.0 * epsilon);
    let directional_error = (&final_jacobian * &direction - directional_fd).amax();
    assert!(
        directional_error < 1e-7,
        "Jacobian directional error={directional_error:e}"
    );
    let full_len = legacy.x_mesh.len() * legacy.values.len();
    let positions: Vec<usize> = (0..full_len)
        .filter(|position| {
            !legacy
                .BC_position_and_value
                .iter()
                .any(|bc| bc.0 == *position)
        })
        .collect();
    let mut scratch = vec![0.0; legacy.values.len()];
    let fresh_weighted_norm = weighted_norm(
        &legacy,
        &final_state,
        &fresh_correction,
        &positions,
        &mut scratch,
    )
    .unwrap();
    let reused_correction = legacy
        .try_step_with_linear_telemetry(legacy.p, &*legacy.y)
        .unwrap()
        .0
        .to_DVectorType();
    let reused_weighted_norm = weighted_norm(
        &legacy,
        &final_state,
        &reused_correction,
        &positions,
        &mut scratch,
    )
    .unwrap();
    let production_convergence_threshold =
        convergence_condition(&*legacy.y, &legacy.abs_tolerance, &legacy.rel_tolerance_vec);
    let report = solve_fixed_mesh(&mut candidate).unwrap();
    let mut fresh_jacobian_control = parser_fixture();
    fresh_jacobian_control.try_eq_generate(None, None).unwrap();
    fresh_jacobian_control
        .strategy_params
        .as_mut()
        .expect("parser fixture has explicit solver strategy")
        .max_jac = Some(0);
    let fresh_jacobian_report = solve_fixed_mesh(&mut fresh_jacobian_control).unwrap();
    let mut schedule_control = parser_fixture();
    schedule_control.try_eq_generate(None, None).unwrap();
    let schedule_report =
        solve_fixed_mesh_with_schedule(&mut schedule_control, TrialSchedule::Legacy).unwrap();
    let mut euclidean_control = parser_fixture();
    euclidean_control.try_eq_generate(None, None).unwrap();
    let euclidean_report = solve_fixed_mesh_with_controls(
        &mut euclidean_control,
        TrialSchedule::Cantera,
        StepNorm::RstEuclidean,
    )
    .unwrap();
    let mut euclidean_legacy_control = parser_fixture();
    euclidean_legacy_control
        .try_eq_generate(None, None)
        .unwrap();
    let euclidean_legacy_report = solve_fixed_mesh_with_controls(
        &mut euclidean_legacy_control,
        TrialSchedule::Legacy,
        StepNorm::RstEuclidean,
    )
    .unwrap();
    let mut seeded_candidate = parser_fixture();
    seeded_candidate.initial_guess = DMatrix::from_column_slice(2, 10, final_state.as_slice());
    seeded_candidate.try_eq_generate(None, None).unwrap();
    let seeded_report = solve_fixed_mesh(&mut seeded_candidate).unwrap();
    println!(
        "legacy converged={} residual={legacy_residual:.12e} counters={legacy_stats:?}",
        legacy_result.is_some()
    );
    println!(
        "legacy final correction: reused-J L2={:.12e} weighted={reused_weighted_norm:.12e}; fresh-J L2={:.12e} weighted={fresh_weighted_norm:.12e}; production_threshold={production_convergence_threshold:.12e}; linear_error={linear_error:.3e}, directional_error={directional_error:.3e}",
        reused_correction.norm(),
        fresh_correction.norm()
    );
    assert!(fresh_weighted_norm.is_finite());
    assert!(
        fresh_weighted_norm > 1.0,
        "the production-published profile is outside the Cantera correction threshold"
    );
    assert!(
        reused_weighted_norm > 1.0,
        "the production-published profile remains outside the threshold even with its reused Jacobian"
    );
    assert!(
        reused_correction.norm() < production_convergence_threshold,
        "the production scalar threshold explains why the reused-J correction was accepted"
    );
    for (label, control) in [
        ("weighted + Cantera schedule", &report),
        (
            "weighted + Cantera schedule + fresh J",
            &fresh_jacobian_report,
        ),
        ("weighted + RST schedule", &schedule_report),
        ("Euclidean + Cantera schedule", &euclidean_report),
        ("Euclidean + RST schedule", &euclidean_legacy_report),
    ] {
        println!(
            "{label}: status={} iterations={} jacobians={} trials={} solves={} residual={:.6e} accepted={:?} trace_head={:?}",
            control.status,
            control.iterations,
            control.jacobians,
            control.trials,
            control.linear_solves,
            control.residual_norm,
            control.accepted_factors,
            control.trial_trace.iter().take(8).collect::<Vec<_>>()
        );
    }
    assert_eq!(
        report.status, -2,
        "Cantera norm + schedule is the failing control"
    );
    assert_eq!(
        schedule_report.status, -2,
        "changing the schedule alone must not be reported as the cause"
    );
    assert_eq!(
        euclidean_report.status, -2,
        "changing the norm alone must not be reported as the cause"
    );
    assert_eq!(
        euclidean_legacy_report.status, 1,
        "the matched legacy norm/schedule control should converge"
    );
    assert_eq!(fresh_jacobian_report.status, 1);
    assert_eq!(fresh_jacobian_report.iterations, 7);
    assert_eq!(fresh_jacobian_report.jacobians, 7);
    assert!(fresh_jacobian_report.residual_norm < 1e-6);
    assert_eq!(
        euclidean_legacy_report.iterations as u64,
        legacy_stats.iterations
    );
    assert_eq!(
        euclidean_legacy_report.jacobians as u64,
        legacy_stats.jacobian_recalculations
    );
    assert_eq!(
        euclidean_legacy_report.trials as u64,
        legacy_stats.damping_trials
    );
    assert_eq!(
        euclidean_legacy_report.linear_solves as u64,
        legacy_stats.linear_solves
    );
    assert!(
        (&legacy.y.to_DVectorType() - &euclidean_legacy_report.state).amax() < 1e-12,
        "the matched legacy control should reproduce the production profile"
    );
    println!(
        "seeded candidate: status={} iterations={} trials={} solves={} residual={:.6e}",
        seeded_report.status,
        seeded_report.iterations,
        seeded_report.trials,
        seeded_report.linear_solves,
        seeded_report.residual_norm
    );
    assert_eq!(
        seeded_report.status, 1,
        "the local candidate should converge from the nearby legacy profile"
    );
    assert!(
        seeded_report.residual_norm < legacy_residual * 0.01,
        "the fresh-J correction should substantially reduce the discrete residual"
    );
    assert!(legacy_residual.is_finite());
    assert!(report.residual_norm.is_finite());
    assert!(matches!(report.status, -4 | -3 | -2 | 1));
    assert_eq!(legacy.x_mesh, candidate.x_mesh);
    if legacy_result.is_some() && report.status == 1 {
        println!(
            "packed solution max difference={:.12e}",
            (&legacy.y.to_DVectorType() - &report.state).amax()
        );
    }
}

#[test]
#[ignore = "large fixed-grid legacy versus test-only Cantera-style local-core diagnostic"]
fn large_kithe_fixed_mesh_ab() {
    use crate::numerical::BVP_Damp::test_large_native_regressions::cantera_fixed_grid_fixture;
    use crate::numerical::BVP_api::BvpNativeSolver;

    let mut legacy_facade = cantera_fixed_grid_fixture();
    let mut candidate_facade = cantera_fixed_grid_fixture();
    let BvpNativeSolver::Damped(legacy) = legacy_facade.native_mut() else {
        panic!("expected Damped legacy fixture");
    };
    let BvpNativeSolver::Damped(candidate) = candidate_facade.native_mut() else {
        panic!("expected Damped candidate fixture");
    };
    for solver in [&mut *legacy, &mut *candidate] {
        solver
            .try_set_initial_guess(solver.initial_guess.clone())
            .unwrap();
        solver.try_eq_generate(None, None).unwrap();
    }
    let initial = legacy.y.to_DVectorType();
    assert_eq!(initial, candidate.y.to_DVectorType());
    assert_eq!(legacy.x_mesh, candidate.x_mesh);
    let legacy_f = legacy
        .fun
        .try_call(legacy.p, &*legacy.y)
        .unwrap()
        .to_DVectorType();
    let candidate_f = candidate
        .fun
        .try_call(candidate.p, &*candidate.y)
        .unwrap()
        .to_DVectorType();
    assert!((&legacy_f - &candidate_f).amax() < 1e-12);
    let legacy_j = legacy
        .jac
        .as_mut()
        .unwrap()
        .try_call(legacy.p, &*legacy.y)
        .unwrap()
        .to_DMatrixType();
    let candidate_j = candidate
        .jac
        .as_mut()
        .unwrap()
        .try_call(candidate.p, &*candidate.y)
        .unwrap()
        .to_DMatrixType();
    assert!((&legacy_j - &candidate_j).amax() < 1e-12);

    let legacy_result = legacy
        .try_solver()
        .expect("legacy solver should not have a backend error");
    let legacy_residual = legacy.try_calc_residual(legacy.y.clone_box()).unwrap();
    let legacy_counts = legacy.get_statistics().telemetry.counters;
    let report = solve_fixed_mesh(candidate).expect("candidate should not have a backend error");
    println!(
        "large legacy: converged={} residual={legacy_residual:.12e} counters={legacy_counts:?}",
        legacy_result.is_some()
    );
    println!(
        "large candidate: status={} iterations={} jacobians={} trials={} linear_solves={} residual={:.12e} accepted_factors={:?}",
        report.status,
        report.iterations,
        report.jacobians,
        report.trials,
        report.linear_solves,
        report.residual_norm,
        report.accepted_factors
    );
    assert!(
        legacy_result.is_some(),
        "the known corrected legacy control should converge"
    );
    assert!(report.residual_norm.is_finite());
    assert!(matches!(report.status, -4 | -3 | -2 | 1));
    if report.status == 1 {
        let legacy_metrics = large_profile_metrics(legacy, &legacy.y.to_DVectorType());
        let candidate_metrics = large_profile_metrics(candidate, &report.state);
        let difference = &legacy.y.to_DVectorType() - &report.state;
        println!(
            "large packed solution max difference={:.12e}",
            difference.amax()
        );
        println!(
            "large profiles (minimum species, sum error, max temperature): legacy={legacy_metrics:?} candidate={candidate_metrics:?}"
        );
        assert!(candidate_metrics.0 >= -0.1 - 1e-12);
        assert!(candidate_metrics.1.is_finite());
        assert!(candidate_metrics.2.is_finite());
        for (value, &(lower, upper)) in report.state.iter().zip(&candidate.bounds_vec) {
            assert!(value.is_finite() && *value >= lower - 1e-12 && *value <= upper + 1e-12);
        }
        let direction = &difference / difference.norm();
        let eps = 1e-7;
        let plus = Vectors_type_casting(
            &(&report.state + eps * &direction),
            candidate.effective_runtime_method(),
        );
        let minus = Vectors_type_casting(
            &(&report.state - eps * &direction),
            candidate.effective_runtime_method(),
        );
        let f_plus = candidate
            .fun
            .try_call(candidate.p, &*plus)
            .unwrap()
            .to_DVectorType();
        let f_minus = candidate
            .fun
            .try_call(candidate.p, &*minus)
            .unwrap()
            .to_DVectorType();
        let finite_difference = (f_plus - f_minus) / (2.0 * eps);
        let jacobian = candidate
            .jac
            .as_mut()
            .unwrap()
            .try_call(candidate.p, &*candidate.y)
            .unwrap()
            .to_DMatrixType();
        let analytical = jacobian * direction;
        let error = (&analytical - finite_difference).amax();
        let scale = analytical.amax().max(1.0);
        println!("large candidate directional Jacobian: max error={error:.6e}, scale={scale:.6e}");
        assert!(error <= 5e-5 * scale);
    }
}
