use crate::numerical::BDF::BDF_api::{
    BdfNativeJacobianSource, BdfSolverOptions, BdfTelemetryMode, ODEsolver,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn native_scalar_solver(
    t0: f64,
    y0: f64,
    t_bound: f64,
    rtol: f64,
    atol: f64,
    max_step: f64,
    rhs: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
    jac: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
) -> ODEsolver {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        t0,
        DVector::from_element(1, y0),
        t_bound,
        max_step,
        rtol,
        atol,
        None,
        false,
        Some((max_step / 8.0).min(0.01)),
    )
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut solver = ODEsolver::new_with_options(options);
    solver.set_native_ode_callbacks_with_jacobian_source(
        move |t, y| DVector::from_element(1, rhs(t, y[0])),
        BdfNativeJacobianSource::StateDependent(std::sync::Arc::new(move |t, y| {
            DMatrix::from_element(1, 1, jac(t, y[0]))
        })),
    );
    solver
}

#[test]
fn bdf_nonautonomous_full_trajectory_matches_closed_form() {
    let t0: f64 = -0.2;
    let y0: f64 = 0.7;
    let t_bound: f64 = 1.0;
    let exact = |t: f64| y0 * (0.5 * (t * t - t0 * t0)).exp();
    let mut solver =
        native_scalar_solver(t0, y0, t_bound, 1e-9, 1e-12, 0.15, |t, y| t * y, |t, _| t);
    solver.try_solve().expect("nonautonomous analytic solve");

    let (times, states) = solver.get_result_ref();
    assert_eq!(solver.get_status(), "finished");
    assert_eq!(times[0], t0);
    assert_eq!(*times.as_slice().last().unwrap(), t_bound);
    assert_eq!(times.len(), states.nrows());

    let max_abs_error = times
        .iter()
        .zip(states.column(0).iter())
        .map(|(&time, &state)| (state - exact(time)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_error <= 2e-8,
        "nonautonomous full-trajectory error={max_abs_error:e}"
    );
    assert!(times.as_slice().windows(2).all(|pair| pair[1] > pair[0]));

    let stats = solver.get_statistics();
    assert_eq!(
        stats.bdf_njev_total, 1,
        "analytic Jacobian is evaluated once"
    );
    assert_eq!(
        stats.candidate_step_attempts_total,
        stats.accepted_steps_total + stats.rejected_step_attempts_total
    );
    println!(
        "[BDF nonautonomous reference] samples={} max_abs_error={max_abs_error:.3e} rhs={} jac={} accepted={} rejected={} status=ok",
        times.len(),
        stats.bdf_nfev_total,
        stats.bdf_njev_total,
        stats.accepted_steps_total,
        stats.rejected_step_attempts_total,
    );
}

#[test]
fn bdf_analytic_error_decreases_with_tighter_tolerances() {
    let rate: f64 = 8.0;
    let t_bound: f64 = 1.0;
    let exact = (-rate * t_bound).exp();
    let tolerances = [1e-3, 1e-5, 1e-7];
    let mut errors = Vec::with_capacity(tolerances.len());
    let mut work = Vec::with_capacity(tolerances.len());

    for rtol in tolerances {
        let mut solver = native_scalar_solver(
            0.0,
            1.0,
            t_bound,
            rtol,
            rtol * 1e-3,
            0.2,
            move |_, y| -rate * y,
            move |_, _| -rate,
        );
        solver.try_solve().expect("stiff-decay tolerance solve");
        let (_, states) = solver.get_result_ref();
        let error = (states[(states.nrows() - 1, 0)] - exact).abs();
        errors.push(error);
        work.push(solver.get_statistics().accepted_steps_total);
    }

    assert!(errors[1] < errors[0], "error did not fall: {errors:?}");
    assert!(errors[2] < errors[1], "error did not fall: {errors:?}");
    assert!(work[0] < work[1] && work[1] < work[2], "work={work:?}");
    println!(
        "[BDF tolerance refinement] rtol={tolerances:?} final_abs_error={errors:?} accepted_steps={work:?} status=ok"
    );
}

#[test]
fn bdf_constant_jacobian_work_is_independent_of_newton_refreshes() {
    let rate = 35.0;
    let t_bound = 0.4;
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-35*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_element(1, 1.0),
        t_bound,
        0.05,
        1e-8,
        1e-11,
        None,
        false,
        Some(0.005),
    )
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut solver = ODEsolver::new_with_options(options);
    solver.set_native_ode_callbacks_with_jacobian_source(
        move |_, y| DVector::from_element(1, -rate * y[0]),
        BdfNativeJacobianSource::Constant(DMatrix::from_element(1, 1, -rate)),
    );
    solver.try_solve().expect("constant-J stiff solve");

    let stats = solver.get_statistics();
    let (_, states) = solver.get_result_ref();
    let error = (states[(states.nrows() - 1, 0)] - (-rate * t_bound).exp()).abs();
    assert!(error <= 2e-8, "constant-J final error={error:e}");
    assert_eq!(stats.bdf_njev_total, 1);
    assert!(stats.accepted_steps_total > 0);
    assert!(stats.bdf_nlu_total <= stats.linear_solve_attempts_total);
    println!(
        "[BDF constant-J work] final_abs_error={error:.3e} rhs={} jac={} factorizations={} linear_solves={} accepted={} rejected={} status=ok",
        stats.bdf_nfev_total,
        stats.bdf_njev_total,
        stats.bdf_nlu_total,
        stats.linear_solve_attempts_total,
        stats.accepted_steps_total,
        stats.rejected_step_attempts_total,
    );
}
