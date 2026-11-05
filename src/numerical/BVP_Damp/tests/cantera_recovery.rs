//! Experimental recovery for one explicitly chosen artificial PDE.
//!
//! This is not a generic BVP transient model or a Cantera `Sim1D` port. The
//! test-only Dense solver rebuilds its Jacobian each iteration, uses an explicit
//! component mask, and leaves domain-specific Cantera resets unimplemented.

use super::super::transient::quadrature_mass;
use super::*;

#[derive(Debug)]
struct TransientAttempt {
    state: Option<DVector<f64>>,
    iterations: usize,
    trials: usize,
    bound_factor: f64,
    limiting_index: Option<usize>,
}

fn full_positions(solver: &NRBVP) -> Vec<usize> {
    (0..solver.values.len() * solver.x_mesh.len())
        .filter(|position| {
            !solver
                .BC_position_and_value
                .iter()
                .any(|bc| bc.0 == *position)
        })
        .collect()
}

fn implicit_residual(
    solver: &NRBVP,
    state: &DVector<f64>,
    previous: &DVector<f64>,
    mass: &DMatrix<f64>,
    dt: f64,
) -> Result<DVector<f64>, String> {
    let residual = solver
        .fun
        .try_call(solver.p, state)
        .map_err(|error| format!("steady residual: {error:?}"))?
        .to_DVectorType();
    Ok(residual + mass * (state - previous) / dt)
}

fn implicit_attempt(
    solver: &mut NRBVP,
    previous: &DVector<f64>,
    mass: &DMatrix<f64>,
    dt: f64,
) -> Result<TransientAttempt, String> {
    if !dt.is_finite() || dt <= 0.0 || mass.shape() != (previous.len(), previous.len()) {
        return Err("invalid test-only transient dimensions or step".into());
    }
    let positions = full_positions(solver);
    if positions.len() != previous.len() || solver.bounds_vec.len() != previous.len() {
        return Err("invalid test-only packed layout".into());
    }
    let mut scratch = vec![0.0; solver.values.len()];
    let mut state = previous.clone();
    let mut trials = 0;
    let mut last_limit = (1.0, None);
    for iteration in 1..=30 {
        let residual = implicit_residual(solver, &state, previous, mass, dt)?;
        let jacobian = solver
            .jac
            .as_mut()
            .ok_or("missing steady Jacobian")?
            .try_call(solver.p, &state)
            .map_err(|error| format!("Jacobian: {error:?}"))?
            .to_DMatrixType()
            + mass / dt;
        let factor = jacobian.lu();
        let correction = factor
            .solve(&residual)
            .ok_or("singular test-only transient Jacobian")?;
        let s0 = weighted_norm(solver, &state, &correction, &positions, &mut scratch)?;
        let (mut alpha, limiting_index) = bound_step_Cantera2_with_update_sign_and_limit(
            &state,
            &correction,
            &solver.bounds_vec,
            -1.0,
        );
        if !alpha.is_finite() {
            return Err("non-finite bound factor".into());
        }
        last_limit = (alpha, limiting_index);
        if alpha < 1e-10 {
            return Ok(TransientAttempt {
                state: None,
                iterations: iteration,
                trials,
                bound_factor: last_limit.0,
                limiting_index: last_limit.1,
            });
        }
        let mut accepted = None;
        for _ in 0..7 {
            trials += 1;
            let next = &state - alpha * &correction;
            let next_residual = implicit_residual(solver, &next, previous, mass, dt)?;
            let next_correction = factor
                .solve(&next_residual)
                .ok_or("singular transient factor")?;
            let s1 = weighted_norm(solver, &next, &next_correction, &positions, &mut scratch)?;
            if s1 < 1.0 || s1 < s0 {
                accepted = Some((next, s1 <= 1.0));
                break;
            }
            alpha /= 2.0_f64.sqrt();
        }
        match accepted {
            Some((next, true)) => {
                return Ok(TransientAttempt {
                    state: Some(next),
                    iterations: iteration,
                    trials,
                    bound_factor: last_limit.0,
                    limiting_index: last_limit.1,
                });
            }
            Some((next, false)) => state = next,
            None => {
                return Ok(TransientAttempt {
                    state: None,
                    iterations: iteration,
                    trials,
                    bound_factor: last_limit.0,
                    limiting_index: last_limit.1,
                });
            }
        }
    }
    Ok(TransientAttempt {
        state: None,
        iterations: 30,
        trials,
        bound_factor: last_limit.0,
        limiting_index: last_limit.1,
    })
}

fn linear_solver(upper_bound: f64) -> NRBVP {
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
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(HashMap::from([("y".into(), 1e-8)]))
            .with_bounds(HashMap::from([("y".into(), (-100.0, upper_bound))])),
    );
    solver.try_eq_generate(None, None).unwrap();
    solver
}

#[test]
fn implicit_linear_step_satisfies_the_chosen_quadrature_residual() {
    let mut solver = linear_solver(100.0);
    let previous = DVector::zeros(2);
    let mass = quadrature_mass(&[0.0, 0.5, 1.0], 1, &[1, 2], &[true], false);
    let report = implicit_attempt(&mut solver, &previous, &mass, 0.1).unwrap();
    let next = report.state.expect("linear implicit problem should solve");
    assert!(
        implicit_residual(&solver, &next, &previous, &mass, 0.1)
            .unwrap()
            .norm()
            < 1e-11
    );
    assert_eq!(previous, DVector::zeros(2));
    assert!(solver.result.is_none());
}

#[test]
fn rejected_implicit_step_does_not_publish_or_replace_previous_state() {
    let mut solver = linear_solver(0.25);
    let previous = DVector::zeros(2);
    let before = solver.y.to_DVectorType();
    let mass = quadrature_mass(&[0.0, 0.5, 1.0], 1, &[1, 2], &[true], false);
    let report = implicit_attempt(&mut solver, &previous, &mass, 0.1).unwrap();
    assert!(
        report.state.is_none(),
        "bounds make the first interval impossible"
    );
    assert!(report.iterations > 0);
    assert_eq!(previous, DVector::zeros(2));
    assert_eq!(solver.y.to_DVectorType(), before);
    assert!(solver.result.is_none());
}

#[test]
#[ignore = "experimental artificial-PDE recovery diagnostic, not a production acceptance gate"]
fn parser_artificial_pde_recovery_from_original_guess() {
    let mut solver = parser_fixture();
    solver.try_eq_generate(None, None).unwrap();
    let initial = DVector::from_column_slice(solver.initial_guess.as_slice());
    let positions = full_positions(&solver);
    let mass = quadrature_mass(
        solver.x_mesh.as_slice(),
        2,
        &positions,
        &[true, true],
        false,
    );
    // Same continuous artificial PDE, but with the temporal term evaluated at
    // the right endpoint. Keep the steady Forward callbacks unchanged.
    let right_mass =
        2.0 * quadrature_mass(solver.x_mesh.as_slice(), 2, &positions, &[true, true], true) - &mass;
    let base_jacobian = solver
        .jac
        .as_mut()
        .unwrap()
        .try_call(solver.p, &initial)
        .unwrap()
        .to_DMatrixType();
    let base_residual = solver
        .fun
        .try_call(solver.p, &initial)
        .unwrap()
        .to_DVectorType();
    for (label, operator) in [("left", &mass), ("right", &right_mass)] {
        for diagnostic_dt in [1e-1, 1e-2, 1e-3, 1e-5] {
            let matrix = &base_jacobian + operator / diagnostic_dt;
            let singular_values = matrix.clone().svd(false, false).singular_values;
            let largest = singular_values.max();
            let smallest = singular_values.min();
            match matrix.clone().lu().solve(&base_residual) {
                Some(correction) => {
                    let linear_error = (&matrix * &correction - &base_residual).norm();
                    println!(
                        "{label} operator dt={diagnostic_dt:.1e} sigma_min={smallest:.4e} sigma_max={largest:.4e} correction_max={:.4e} linear_error={linear_error:.4e}",
                        correction.amax()
                    );
                }
                None => println!(
                    "{label} operator dt={diagnostic_dt:.1e} sigma_min={smallest:.4e} sigma_max={largest:.4e} singular LU"
                ),
            }
        }
    }
    for mask in [[true, false], [false, true], [true, true]] {
        let mask_left = quadrature_mass(solver.x_mesh.as_slice(), 2, &positions, &mask, false);
        let mask_average = quadrature_mass(solver.x_mesh.as_slice(), 2, &positions, &mask, true);
        let mask_right = 2.0 * mask_average - &mask_left;
        for (label, operator) in [("left", mask_left), ("right", mask_right)] {
            let matrix = &base_jacobian + operator / 1e-3;
            let singular_values = matrix.svd(false, false).singular_values;
            println!(
                "mask={mask:?} {label} dt=1e-3 sigma_min={:.4e} sigma_max={:.4e}",
                singular_values.min(),
                singular_values.max()
            );
        }
    }
    let right_trial = implicit_attempt(&mut solver, &initial, &right_mass, 1e-3).unwrap();
    println!("right-endpoint transient control: {right_trial:?}");
    if let Some(ref state) = right_trial.state {
        let residual = implicit_residual(&solver, state, &initial, &right_mass, 1e-3).unwrap();
        println!(
            "right-endpoint transient residual norm={:.6e}",
            residual.norm()
        );
    }
    let steady = solve_fixed_mesh(&mut solver).unwrap();
    assert_ne!(
        steady.status, 1,
        "this control starts from the known failed steady attempt"
    );
    let mut committed = initial.clone();
    let mut dt = 1e-3;
    let mut successful = 0;
    let mut failed = 0;
    for cycle in 0..4 {
        for _ in 0..10 {
            let before = committed.clone();
            let attempt = implicit_attempt(&mut solver, &before, &mass, dt).unwrap();
            match attempt.state {
                Some(next) => {
                    committed = next;
                    successful += 1;
                }
                None => {
                    assert_eq!(committed, before);
                    dt *= 0.5;
                    failed += 1;
                }
            }
            println!(
                "transient cycle={cycle} dt={dt:.4e} committed_steps={successful} failures={failed} newton_iters={} trials={} bound_factor={:.6e} limiting={:?}",
                attempt.iterations,
                attempt.trials,
                attempt.bound_factor,
                attempt
                    .limiting_index
                    .and_then(|index| solver.variable_string.get(index))
            );
            if dt < 1e-12 {
                println!("recovery stopped at the experimental minimum dt");
                return;
            }
        }
        solver.initial_guess = DMatrix::from_column_slice(2, 10, committed.as_slice());
        let retry = solve_fixed_mesh(&mut solver).unwrap();
        println!(
            "steady retry cycle={cycle} status={} residual={:.8e}",
            retry.status, retry.residual_norm
        );
        if retry.status == 1 {
            return;
        }
    }
    assert!(successful + failed > 0);
    println!(
        "recovery did not reach steady convergence; original and committed states kept distinct"
    );
}
