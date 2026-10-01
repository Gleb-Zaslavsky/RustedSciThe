use super::*;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn make_stiff_reference_solver(case: &crate::numerical::ivp_test_support::StiffIvpCase) -> BE {
    let values = (0..case.initial.len())
        .map(|index| format!("y{index}"))
        .collect();
    let y0 = DVector::from_column_slice(case.initial);
    let mut solver = BE::new();
    match case.jacobian {
        Some(jacobian) => solver
            .try_set_native_initial(
                values,
                "t".to_string(),
                1e-10,
                24,
                Some(case.step),
                0.0,
                case.t_end,
                y0,
                case.rhs,
                Some(jacobian),
            )
            .unwrap(),
        None => solver
            .try_set_native_initial::<_, fn(f64, &DVector<f64>) -> DMatrix<f64>>(
                values,
                "t".to_string(),
                1e-10,
                24,
                Some(case.step),
                0.0,
                case.t_end,
                y0,
                case.rhs,
                None,
            )
            .unwrap(),
    }
    solver
}

fn assert_reference_endpoint(
    name: &str,
    actual: &DVector<f64>,
    reference: &[f64],
    abs_tolerance: f64,
    rel_tolerance: f64,
) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let mut max_scaled_error = 0.0_f64;
    for (index, (&value, &expected)) in actual.iter().zip(reference).enumerate() {
        let allowed = abs_tolerance.max(rel_tolerance * expected.abs());
        let scaled = (value - expected).abs() / allowed;
        max_scaled_error = max_scaled_error.max(scaled);
        assert!(
            scaled <= 1.0,
            "{name} endpoint component {index}: actual={value:.12e}, reference={expected:.12e}, allowed={allowed:.3e}"
        );
    }
    max_scaled_error
}

fn make_diffusion_solver(dimension: usize, analytic_jacobian: bool, final_time: f64) -> BE {
    let mut solver = BE::new();
    let jacobian =
        analytic_jacobian.then_some(diffusion_jacobian as fn(f64, &DVector<f64>) -> DMatrix<f64>);
    solver
        .try_set_native_initial(
            (0..dimension).map(|i| format!("y{i}")).collect(),
            "t".to_string(),
            1e-10,
            16,
            Some(0.01),
            0.0,
            final_time,
            DVector::from_fn(dimension, |i, _| 1.0 + i as f64 / dimension as f64),
            diffusion_rhs,
            jacobian,
        )
        .unwrap();
    solver
}

fn diffusion_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
    DVector::from_fn(y.len(), |index, _| {
        let left = if index == 0 { 0.0 } else { y[index - 1] };
        let right = if index + 1 == y.len() {
            0.0
        } else {
            y[index + 1]
        };
        0.05 * (left - 2.0 * y[index] + right)
    })
}

fn diffusion_jacobian(_: f64, y: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(y.len(), y.len(), |row, column| match row.abs_diff(column) {
        0 => -0.1,
        1 => 0.05,
        _ => 0.0,
    })
}

#[test]
fn be_diffusion_analytic_and_fd_jacobian_routes_match() {
    let mut analytic = make_diffusion_solver(8, true, 0.2);
    let mut finite_difference = make_diffusion_solver(8, false, 0.2);
    analytic.try_solve().unwrap();
    finite_difference.try_solve().unwrap();

    let (analytic_times, analytic_states) = analytic.get_result();
    let (fd_times, fd_states) = finite_difference.get_result();
    assert_eq!(analytic_times.unwrap(), fd_times.unwrap());
    let analytic_states = analytic_states.unwrap();
    let fd_states = fd_states.unwrap();
    let max_diff = analytic_states
        .iter()
        .zip(fd_states.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-8, "Jacobian route drift {max_diff:e}");
}

#[test]
fn be_robertson_matches_independent_stiff_reference_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_robertson_matches_independent_stiff_reference_story",
    );
    let case = crate::numerical::ivp_test_support::robertson();
    let mut solver = make_stiff_reference_solver(&case);
    solver.try_solve().unwrap();

    let (_, states) = solver.get_result();
    let states = states.unwrap();
    let final_state = states.row(states.nrows() - 1).transpose();
    let reference = case
        .reference
        .expect("Robertson fixture should use its tabulated endpoint");
    let max_scaled_error = assert_reference_endpoint(
        case.name,
        &final_state,
        reference,
        case.abs_tolerance,
        case.rel_tolerance,
    );
    let mass = final_state.iter().sum::<f64>();
    assert!(
        (mass - 1.0).abs() < 2e-10,
        "Robertson mass balance {mass:.12e}"
    );
    assert_eq!(solver.status(), BeStatus::Finished);
    println!(
        "[BE independent reference] case={} t_end={} h={} steps={} max_scaled_endpoint_error={max_scaled_error:.4} mass_balance={:.3e}",
        case.name,
        case.t_end,
        case.step,
        solver.get_statistics().step_calls,
        (mass - 1.0).abs()
    );
}

#[test]
fn be_hires_matches_independent_stiff_reference_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_hires_matches_independent_stiff_reference_story",
    );
    let case = crate::numerical::ivp_test_support::hires();
    let initial = DVector::from_column_slice(case.initial);
    let reference_coarse = crate::numerical::ivp_test_support::integrate_rk4(
        case.rhs, 0.0, case.t_end, &initial, 2e-4,
    );
    let reference = crate::numerical::ivp_test_support::integrate_rk4(
        case.rhs, 0.0, case.t_end, &initial, 1e-4,
    );
    let refinement_error = (&reference - &reference_coarse).amax();
    assert!(
        refinement_error < 2e-8,
        "HIRES RK4 reference did not refine: {refinement_error:e}"
    );
    let mut solver = make_stiff_reference_solver(&case);
    solver.try_solve().unwrap();

    let (_, states) = solver.get_result();
    let states = states.unwrap();
    let final_state = states.row(states.nrows() - 1).transpose();
    let max_scaled_error = assert_reference_endpoint(
        case.name,
        &final_state,
        reference.as_slice(),
        case.abs_tolerance,
        case.rel_tolerance,
    );
    assert_eq!(solver.status(), BeStatus::Finished);
    assert!(final_state.iter().all(|value| value.is_finite()));
    println!(
        "[BE independent reference] case={} t_end={} h={} reference_RK4_h=1e-4 refinement_linf={refinement_error:.3e} steps={} max_scaled_endpoint_error={max_scaled_error:.4}",
        case.name,
        case.t_end,
        case.step,
        solver.get_statistics().step_calls,
    );
}

#[test]
fn be_large_combustion_chain_matches_refined_rk4_reference_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_large_combustion_chain_matches_refined_rk4_reference_story",
    );
    use crate::numerical::ivp_test_support::{
        combustion_chain_initial, combustion_chain_rhs, integrate_rk4,
    };

    let initial = combustion_chain_initial();
    let t_end = 0.2;
    let reference_coarse = integrate_rk4(combustion_chain_rhs, 0.0, t_end, &initial, 2e-5);
    let reference = integrate_rk4(combustion_chain_rhs, 0.0, t_end, &initial, 1e-5);
    let refinement_error = (&reference - &reference_coarse).amax();
    assert!(
        refinement_error < 2e-8,
        "RK4 reference did not refine: {refinement_error:e}"
    );

    let dimension = initial.len();
    let mut solver = BE::new();
    solver
        .try_set_native_initial::<_, fn(f64, &DVector<f64>) -> DMatrix<f64>>(
            (0..dimension).map(|index| format!("y{index}")).collect(),
            "t".to_string(),
            1e-10,
            24,
            Some(0.001),
            0.0,
            t_end,
            initial,
            combustion_chain_rhs,
            None,
        )
        .unwrap();
    solver.try_solve().unwrap();
    let (_, states) = solver.get_result();
    let states = states.unwrap();
    let final_state = states.row(states.nrows() - 1).transpose();
    let max_scaled_error = final_state
        .iter()
        .zip(reference.iter())
        .map(|(&actual, &expected)| (actual - expected).abs() / 0.02_f64.max(0.03 * expected.abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        max_scaled_error <= 1.0,
        "combustion-chain error exceeds first-order BE budget: {max_scaled_error:.4}"
    );
    assert_eq!(solver.status(), BeStatus::Finished);
    assert!(final_state.iter().all(|value| value.is_finite()));
    assert!(final_state.iter().take(9).all(|value| *value >= -1e-8));
    println!(
        "[BE independent reference] case=combustion-chain dimension={dimension} t_end={t_end} BE_h=0.001 reference_RK4_h=1e-5 refinement_linf={refinement_error:.3e} max_scaled_endpoint_error={max_scaled_error:.4} steps={}",
        solver.get_statistics().step_calls
    );
}

#[test]
#[ignore = "release lifecycle story: requires tcc and validates real AOT build/link provenance"]
fn be_aot_build_link_provenance_and_timing_scopes_story() {
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetryExecution, IvpTelemetryMode};
    use crate::symbolic::symbolic_ivp_generated::{
        SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_aot_build_link_provenance_and_timing_scopes_story",
    );
    const DIMENSION: usize = 8;
    const STEP: f64 = 0.002;
    const FINAL_TIME: f64 = 0.02;
    let lambdify_workload = crate::numerical::ivp_workloads::diffusion_chain(DIMENSION);
    let aot_workload = crate::numerical::ivp_workloads::diffusion_chain(DIMENSION);
    let parameter_refs: Vec<&str> = lambdify_workload
        .parameter_names
        .iter()
        .map(String::as_str)
        .collect();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let output_dir = PathBuf::from(format!(
        "target/be-aot-lifecycle-story-{}-{nonce}",
        std::process::id()
    ));
    let config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(output_dir))
        .with_c_tcc()
        .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        });

    let lambdify_started = Instant::now();
    let mut lambdify = BE::try_new_with_options(
        BeSolverOptions::new(
            lambdify_workload.equations,
            lambdify_workload.variables,
            lambdify_workload.time_variable,
            1e-10,
            20,
            Some(STEP),
            0.0,
            FINAL_TIME,
            lambdify_workload.initial_state,
        )
        .with_symbolic_assembly_backend(BeSymbolicAssemblyBackend::AtomViewNative)
        .with_telemetry_mode(BeTelemetryMode::Timings),
    )
    .unwrap();
    lambdify
        .try_set_equation_parameters(Some(&parameter_refs))
        .unwrap();
    lambdify
        .set_parameter_values(lambdify_workload.parameter_values)
        .unwrap();
    lambdify.try_solve().unwrap();
    let lambdify_e2e_ms = lambdify_started.elapsed().as_secs_f64() * 1_000.0;

    let aot_started = Instant::now();
    let mut solver = BE::try_new_with_options(
        BeSolverOptions::new(
            aot_workload.equations,
            aot_workload.variables,
            aot_workload.time_variable,
            1e-10,
            20,
            Some(STEP),
            0.0,
            FINAL_TIME,
            aot_workload.initial_state,
        )
        .with_symbolic_assembly_backend(BeSymbolicAssemblyBackend::AtomViewNative)
        .with_generated_backend_config(config)
        .with_telemetry_mode(BeTelemetryMode::Timings),
    )
    .unwrap();
    solver
        .try_set_equation_parameters(Some(&parameter_refs))
        .unwrap();
    solver
        .set_parameter_values(aot_workload.parameter_values)
        .unwrap();
    solver.try_solve().unwrap();
    let aot_e2e_ms = aot_started.elapsed().as_secs_f64() * 1_000.0;

    let (lambdify_times, lambdify_states) = lambdify.trajectory();
    let (aot_times, aot_states) = solver.trajectory();
    assert_eq!(lambdify_times.len(), aot_times.len());
    assert_eq!(lambdify_states.shape(), aot_states.shape());
    let max_time_diff = lambdify_times
        .iter()
        .zip(aot_times.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let max_state_diff = lambdify_states
        .iter()
        .zip(aot_states.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_time_diff <= 1e-12,
        "trajectory time mismatch: {max_time_diff:e}"
    );
    assert!(
        max_state_diff <= 1e-10,
        "trajectory state mismatch: {max_state_diff:e}"
    );

    let snapshot = solver
        .symbolic_ivp_telemetry_snapshot()
        .expect("AOT lifecycle story must collect telemetry");
    assert_eq!(snapshot.mode, IvpTelemetryMode::Detailed);
    assert_eq!(snapshot.execution, IvpTelemetryExecution::Aot);
    assert!(
        snapshot.aot_build_attempts > 0,
        "expected a generated build"
    );
    assert!(snapshot.aot_link_attempts > 0, "expected a generated link");
    assert!(snapshot.aot_build_successes > 0);
    assert!(snapshot.aot_link_successes > 0);
    assert!(snapshot.aot_runtime_ready > 0);
    assert!(snapshot.aot_resolution_misses > 0);
    assert!(!snapshot.aot_artifact_keys.is_empty());

    let materialize = snapshot.cold_stage(IvpColdStage::AotMaterialization);
    let build = snapshot.cold_stage(IvpColdStage::AotBuild);
    let link = snapshot.cold_stage(IvpColdStage::AotLink);
    let lookup = snapshot.cold_stage(IvpColdStage::AotCacheLookup);
    assert!(materialize.calls > 0 && !materialize.elapsed.is_zero());
    assert_eq!(build.calls, snapshot.aot_build_attempts);
    assert_eq!(link.calls, snapshot.aot_link_attempts);
    assert!(lookup.calls > 0);
    assert!(!build.elapsed.is_zero());
    assert!(!link.elapsed.is_zero());

    println!(
        "[BE cold E2E comparison] fixture=shared-diffusion-chain-{DIMENSION} route_order=lambdify_then_aot telemetry=Timings; includes BE setup+parameter binding+solve; AOT uses RebuildAlways; not a process-cold/benchmark baseline\nroute | wall_ms\nLambdify-AtomViewNative | {lambdify_e2e_ms:.3}\nAOT-AtomViewNative-tcc | {aot_e2e_ms:.3}\nparity max_time_diff={max_time_diff:.3e} max_state_diff={max_state_diff:.3e}"
    );
    println!(
        "[BE AOT lifecycle] execution={} hits={} misses={} build_attempts={} build_successes={} link_attempts={} link_successes={} runtime_ready={} artifact_keys={:?}",
        snapshot.execution.label(),
        snapshot.aot_resolution_hits,
        snapshot.aot_resolution_misses,
        snapshot.aot_build_attempts,
        snapshot.aot_build_successes,
        snapshot.aot_link_attempts,
        snapshot.aot_link_successes,
        snapshot.aot_runtime_ready,
        snapshot.aot_artifact_keys,
    );
    println!(
        "stage | calls | elapsed_ms (scopes are diagnostic; parent/child values are not additive)\nmaterialize | {} | {:.3}\nbuild | {} | {:.3}\nlink | {} | {:.3}\ncache_lookup | {} | {:.3}",
        materialize.calls,
        materialize.elapsed.as_secs_f64() * 1_000.0,
        build.calls,
        build.elapsed.as_secs_f64() * 1_000.0,
        link.calls,
        link.elapsed.as_secs_f64() * 1_000.0,
        lookup.calls,
        lookup.elapsed.as_secs_f64() * 1_000.0,
    );
    assert_eq!(solver.status(), BeStatus::Finished);
}

#[test]
fn be_native_full_solve_scaling_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_native_full_solve_scaling_story",
    );
    println!(
        "[BE native full solve story] native diffusion-chain; telemetry=Timings; timings are diagnostic, not thresholds"
    );
    println!(
        "dimension | steps | repetitions | wall_ms | solver_ms | residual_calls | jacobian_calls | accepted | status"
    );

    for (dimension, repetitions) in [(3, 8), (8, 6), (16, 4), (32, 3)] {
        let steps = 8;
        let mut solver = make_diffusion_solver(dimension, true, steps as f64 * 0.01);
        solver
            .try_set_telemetry_mode(BeTelemetryMode::Timings)
            .unwrap();
        solver.try_solve().unwrap();
        let before = solver.get_statistics();
        let detailed_before = solver.detailed_statistics();

        let started = Instant::now();
        for _ in 0..repetitions {
            solver.try_solve().unwrap();
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let stats = solver.get_statistics();
        let detailed = solver.detailed_statistics();
        println!(
            "{dimension} | {steps} | {repetitions} | {elapsed_ms:.3} | {:.3} | {} | {} | {} | {}",
            stats.solve_ms_total - before.solve_ms_total,
            stats.residual_calls - before.residual_calls,
            stats.jacobian_calls - before.jacobian_calls,
            detailed.accepted_steps - detailed_before.accepted_steps,
            solver.status().as_str()
        );
        assert_eq!(solver.status(), BeStatus::Finished);
        assert!(solver.y.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn be_symbolic_parameter_rebind_matches_fresh_solve_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_symbolic_parameter_rebind_matches_fresh_solve_story",
    );
    let make_parameterized = |rate: f64| {
        let mut solver = BE::new();
        solver
            .try_set_initial(
                vec![Expr::parse_expression("-rate*y")],
                vec!["y".to_string()],
                "t".to_string(),
                1e-12,
                20,
                Some(0.1),
                0.0,
                0.5,
                DVector::from_vec(vec![1.0]),
            )
            .unwrap();
        solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
        solver
            .set_parameter_values(DVector::from_vec(vec![rate]))
            .unwrap();
        solver
    };

    let mut rebound = make_parameterized(1.0);
    rebound
        .try_set_telemetry_mode(BeTelemetryMode::Timings)
        .unwrap();
    rebound.try_solve().unwrap();
    let prepared_calls = rebound.get_statistics().backend_prepare_calls;
    let before = rebound.get_statistics();
    let started = Instant::now();
    rebound
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .unwrap();
    rebound.try_solve().unwrap();
    let rebound_wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let after = rebound.get_statistics();

    let mut fresh = make_parameterized(2.0);
    fresh
        .try_set_telemetry_mode(BeTelemetryMode::Timings)
        .unwrap();
    let fresh_started = Instant::now();
    fresh.try_solve().unwrap();
    let fresh_wall_ms = fresh_started.elapsed().as_secs_f64() * 1_000.0;
    let rebound_result = rebound.get_result();
    let fresh_result = fresh.get_result();
    assert_eq!(rebound_result.0.unwrap(), fresh_result.0.unwrap());
    let rebound_states = rebound_result.1.unwrap();
    let fresh_states = fresh_result.1.unwrap();
    let max_diff = rebound_states
        .iter()
        .zip(fresh_states.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-12, "rebind trajectory drift {max_diff:e}");
    assert_eq!(
        rebound.get_statistics().backend_prepare_calls,
        prepared_calls,
        "parameter updates must reuse symbolic preparation"
    );

    println!("[BE symbolic parameter continuation] correctness parity; diagnostic timings only");
    println!(
        "route | wall_ms | solver_ms | residual_calls | jacobian_calls | max_state_diff | preparations"
    );
    println!(
        "rebind | {rebound_wall_ms:.3} | {:.3} | {} | {} | {max_diff:.3e} | {}",
        after.solve_ms_total - before.solve_ms_total,
        after.residual_calls - before.residual_calls,
        after.jacobian_calls - before.jacobian_calls,
        rebound.get_statistics().backend_prepare_calls - prepared_calls
    );
    println!(
        "fresh | {fresh_wall_ms:.3} | {:.3} | {} | {} | {max_diff:.3e} | {}",
        fresh.get_statistics().solve_ms_total,
        fresh.get_statistics().residual_calls,
        fresh.get_statistics().jacobian_calls,
        fresh.get_statistics().backend_prepare_calls
    );
}

#[test]
fn be_repeated_symbolic_parameter_rebind_matches_fresh_solutions_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_repeated_symbolic_parameter_rebind_matches_fresh_solutions_story",
    );
    let make_parameterized = |rate: f64| {
        let mut solver = BE::new();
        solver
            .try_set_initial(
                vec![Expr::parse_expression("-rate*y")],
                vec!["y".to_string()],
                "t".to_string(),
                1e-12,
                20,
                Some(0.1),
                0.0,
                0.5,
                DVector::from_vec(vec![1.0]),
            )
            .unwrap();
        solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
        solver
            .set_parameter_values(DVector::from_vec(vec![rate]))
            .unwrap();
        solver
    };

    let rates = [0.5, 1.0, 2.0, 4.0, 1.5, 0.5];
    let mut reused = make_parameterized(rates[0]);
    reused
        .try_set_telemetry_mode(BeTelemetryMode::Timings)
        .unwrap();
    reused.try_solve().unwrap();
    let preparation_count = reused.get_statistics().backend_prepare_calls;

    println!(
        "[BE repeated parameter rebind] targets={}; preparation must remain constant",
        rates.len()
    );
    println!("target | rate | reused_final | fresh_final | abs_diff | cumulative_preparations");
    for (target, rate) in rates.iter().copied().enumerate() {
        reused
            .set_parameter_values(DVector::from_vec(vec![rate]))
            .unwrap();
        reused.try_solve().unwrap();
        let reused_final = reused.y[0];

        let mut fresh = make_parameterized(rate);
        fresh.try_solve().unwrap();
        let fresh_final = fresh.y[0];
        let diff = (reused_final - fresh_final).abs();
        assert!(
            diff < 1e-12,
            "target {target} at rate {rate} drifted by {diff:e}"
        );
        assert_eq!(
            reused.get_statistics().backend_prepare_calls,
            preparation_count,
            "value-only parameter rebind must not prepare again"
        );
        println!(
            "{target} | {rate:.3} | {reused_final:.12e} | {fresh_final:.12e} | {diff:.3e} | {}",
            reused.get_statistics().backend_prepare_calls
        );
    }
}

#[test]
fn be_accepted_state_continuation_after_parameter_change_matches_segmented_reference_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_accepted_state_continuation_after_parameter_change_matches_segmented_reference_story",
    );
    let make_parameterized = |rate: f64, initial_value: f64, initial_time: f64, final_time: f64| {
        let mut solver = BE::new();
        solver
            .try_set_initial(
                vec![Expr::parse_expression("-rate*y")],
                vec!["y".to_string()],
                "t".to_string(),
                1e-12,
                20,
                Some(0.125),
                initial_time,
                final_time,
                DVector::from_vec(vec![initial_value]),
            )
            .unwrap();
        solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
        solver
            .set_parameter_values(DVector::from_vec(vec![rate]))
            .unwrap();
        solver
    };

    let mut continued = make_parameterized(1.0, 1.0, 0.0, 0.5);
    continued
        .try_set_telemetry_mode(BeTelemetryMode::Timings)
        .unwrap();
    continued.try_solve().unwrap();
    let preparations_before = continued.get_statistics().backend_prepare_calls;
    let before = continued.get_statistics();
    let started = Instant::now();
    continued
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .unwrap();
    continued.try_continue_to(1.0).unwrap();
    let continue_wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let after = continued.get_statistics();
    let continuation_stats = continued.continuation_statistics().clone();

    let mut first = make_parameterized(1.0, 1.0, 0.0, 0.5);
    first.try_solve().unwrap();
    let first_times = first.get_result().0.unwrap().clone();
    let first_states = first.get_result().1.unwrap().clone();
    let midpoint = first_states[(first_states.nrows() - 1, 0)];
    let mut second = make_parameterized(2.0, midpoint, 0.5, 1.0);
    second.try_solve().unwrap();
    let expected_times = [
        first_times.as_slice(),
        &second.get_result().0.unwrap().as_slice()[1..],
    ]
    .concat();
    let expected_states = [
        first_states.as_slice(),
        &second.get_result().1.unwrap().as_slice()[1..],
    ]
    .concat();
    let (actual_times, actual_states) = continued.get_result();
    assert_eq!(actual_times.unwrap().as_slice(), expected_times);
    let actual_states = actual_states.unwrap();
    let max_diff = actual_states
        .iter()
        .zip(expected_states.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-12, "continued trajectory drift {max_diff:e}");
    assert_eq!(after.backend_prepare_calls, preparations_before);
    assert_eq!(continuation_stats.attempts, 1);
    assert_eq!(continuation_stats.completed, 1);
    assert_eq!(continuation_stats.failures, 0);

    println!(
        "[BE accepted-state parameter continuation] two segments, rates=1 -> 2; diagnostic timings"
    );
    println!(
        "wall_ms | solver_ms | continuation_ms | residual_calls | jacobian_calls | preparations | max_state_diff"
    );
    println!(
        "{continue_wall_ms:.3} | {:.3} | {:.3} | {} | {} | {} | {max_diff:.3e}",
        after.solve_ms_total - before.solve_ms_total,
        continuation_stats.elapsed_ms_total,
        after.residual_calls - before.residual_calls,
        after.jacobian_calls - before.jacobian_calls,
        after.backend_prepare_calls
    );
}
