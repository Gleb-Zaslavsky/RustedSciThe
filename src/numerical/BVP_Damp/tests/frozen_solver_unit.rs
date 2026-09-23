//! Unit tests extracted from the production module.
//!
//! Keeping these tests in a separate file keeps solver implementation and
//! test-only story machinery independently navigable.

//! Heavy Frozen/AOT story tests.
//!
//! Release command for the three combustion-1000 Frozen stories:
//!
//! ```text
//! cargo test --release --lib --no-default-features numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_combustion_1000 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Each verbose story is mirrored to `test_reports/BVP_Damp_AOT_Frozen/`.
use super::*;
use crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot;
use std::time::Duration;

// Keep the verbose Frozen story tables in the same report system as the
// other AOT stories without putting filesystem work in solver timings.
macro_rules! println {
        () => {
            crate::Utils::test_reporting::capture_test_line(format_args!(""));
        };
        ($($arg:tt)*) => {
            crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
        };
    }

#[test]
fn frozen_statistics_keep_legacy_projection_and_typed_snapshot_in_sync() {
    let solver = NRBVP::new(
        Vec::new(),
        DMatrix::zeros(0, 0),
        Vec::new(),
        String::new(),
        HashMap::new(),
        0.0,
        0.0,
        1,
        "Naive".to_string(),
        None,
        None,
        "Dense".to_string(),
        1e-8,
        1,
    );
    let stats = solver.get_statistics();

    assert_eq!(
        stats.counters["number of iterations"],
        stats.telemetry.counters.iterations as usize
    );
    assert_eq!(
        stats.counters["number of factorizations"],
        stats.telemetry.counters.factorizations as usize
    );
    assert_eq!(
        stats.counters["number of RHS solves"],
        stats.telemetry.counters.rhs_solves as usize
    );
    assert!(stats.telemetry.timings.total >= stats.telemetry.timings.jacobian);
}

#[test]
fn frozen_statistics_expose_atom_discretization_telemetry() {
    let mut solver = NRBVP::new(
        Vec::new(),
        DMatrix::zeros(0, 0),
        Vec::new(),
        String::new(),
        HashMap::new(),
        0.0,
        0.0,
        1,
        "Naive".to_string(),
        None,
        None,
        "Dense".to_string(),
        1e-8,
        1,
    );
    let expected = BvpAtomDiscretizationTelemetrySnapshot {
        boundary_conditions: Duration::from_micros(2),
        discretization: Duration::from_micros(3),
        boundary_application: Duration::from_micros(5),
        flat_list: Duration::from_micros(7),
        consistency: Duration::from_micros(11),
        bounds_and_tolerances: Duration::from_micros(13),
        total: Duration::from_micros(41),
    };
    solver.atom_discretization_telemetry = Some(expected);

    let actual = solver
        .get_statistics()
        .telemetry
        .atom_discretization
        .expect("Frozen statistics should preserve Atom preparation telemetry");
    assert_eq!(actual, expected);
}

#[test]
fn changing_continuation_parameter_invalidates_owned_factor() {
    let mut solver = NRBVP::new(
        Vec::new(),
        DMatrix::zeros(1, 1),
        vec!["y".to_string()],
        "x".to_string(),
        HashMap::from([("y".to_string(), vec![(0, 0.0)])]),
        0.0,
        1.0,
        1,
        "Naive".to_string(),
        None,
        None,
        "Dense".to_string(),
        1e-8,
        1,
    );
    *solver.factor_owner.borrow_mut() = Some(
        prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
            .expect("dense factor owner runtime"),
    );

    solver.set_p(1.0);

    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.jac_recalc);
    assert_eq!(
        solver
            .telemetry_counters
            .snapshot()
            .factorization_invalidations,
        1
    );
}

#[test]
fn changing_parameters_invalidates_owned_factor() {
    let mut solver = NRBVP::new(
        Vec::new(),
        DMatrix::zeros(1, 1),
        vec!["y".to_string()],
        "x".to_string(),
        HashMap::from([("y".to_string(), vec![(0, 0.0)])]),
        0.0,
        1.0,
        1,
        "Naive".to_string(),
        None,
        None,
        "Dense".to_string(),
        1e-8,
        1,
    );
    *solver.factor_owner.borrow_mut() = Some(
        prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
            .expect("dense factor owner runtime"),
    );

    solver.set_params(Some(&["alpha"]));
    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.jac_recalc);

    *solver.factor_owner.borrow_mut() = Some(
        prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
            .expect("dense factor owner runtime"),
    );
    solver.set_param_values(Some(vec![1.0]));

    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.jac_recalc);
    assert_eq!(
        solver
            .telemetry_counters
            .snapshot()
            .factorization_invalidations,
        2
    );
}

#[test]
fn changing_backend_policy_invalidates_owned_factor() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    *solver.factor_owner.borrow_mut() = Some(
        prepare_factor_owner_runtime(&DMatrix::from_row_slice(1, 1, &[2.0]), (0, 0), None)
            .expect("dense factor owner runtime"),
    );

    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.jac_recalc);
    assert_eq!(
        solver
            .telemetry_counters
            .snapshot()
            .factorization_invalidations,
        1
    );
}

#[test]
fn frozen_try_iteration_surfaces_residual_callback_panic_as_typed_error() {
    let mut solver = sparse_surface_test_solver();
    solver.fun = convert_to_fun(Box::new(|_, _| {
        panic!("frozen residual callback failed deliberately")
    }));

    let error = solver
        .try_iteration()
        .expect_err("a callback panic must not escape Frozen try_iteration");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::CallbackExecutionFailed { stage, message }
            if stage == "residual"
                && message.contains("frozen residual callback failed deliberately")
    ));
}

use crate::numerical::BVP_Damp::BVP_traits::convert_to_fun;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy,
    BandedGeneratedBackendMode, GeneratedBackendConfig, SparseGeneratedBackendMode,
};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedSparseAotBackend, register_linked_sparse_backend, unregister_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_orchestrator::{
    ParallelExecutorConfig, ParallelFallbackPolicy,
};
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
use faer::Col;
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

fn sparse_surface_test_solver() -> NRBVP {
    NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
        DMatrix::from_element(2, 4, 0.1),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        4,
        FrozenSolverOptions::sparse_frozen(),
    )
}

fn sparse_surface_test_solver_with_naive_strategy() -> NRBVP {
    let options = FrozenSolverOptions::sparse_frozen()
        .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
        .with_tolerance(1e-6)
        .with_max_iterations(10);
    NRBVP::new_with_options(
        vec![Expr::parse_expression("y-z"), Expr::parse_expression("-z")],
        DMatrix::from_element(2, 8, 0.5),
        vec!["z".to_string(), "y".to_string()],
        "x".to_string(),
        HashMap::from([
            ("z".to_string(), vec![(0usize, 1.0f64)]),
            ("y".to_string(), vec![(1usize, 1.0f64)]),
        ]),
        0.0,
        1.0,
        8,
        options,
    )
}

fn frozen_linear_solver(n_steps: usize, options: FrozenSolverOptions) -> NRBVP {
    // y'' = 0 represented as y' = z, z' = 0 with
    // y(0)=0 and z(0)=1. This gives y=x, z=1 and is exact for
    // the forward first-order BVP stencil, so backend coverage is not
    // polluted by discretization error.
    let values = vec!["y".to_string(), "z".to_string()];
    let t0 = 0.0;
    let t_end = 1.0;
    let h = (t_end - t0) / n_steps as f64;
    let mut guess = vec![0.0; values.len() * n_steps];
    for i in 0..n_steps {
        let x = t0 + (i as f64) * h;
        guess[i * values.len()] = x;
        guess[i * values.len() + 1] = 1.0;
    }

    NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")],
        DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice()),
        values,
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(0usize, 1.0f64)]),
        ]),
        t0,
        t_end,
        n_steps,
        options,
    )
}

fn assert_frozen_linear_solution_quality(
    solver: &NRBVP,
    n_steps: usize,
    rms_tol: f64,
    max_abs_tol: f64,
) {
    let solution = solver
        .get_result()
        .expect("frozen BVP solver should store a solution matrix");
    assert_eq!(solution.nrows(), n_steps);
    assert_eq!(solution.ncols(), 2);

    let h = 1.0 / n_steps as f64;
    let mut sq_sum = 0.0;
    let mut max_abs = 0.0;
    for i in 0..solution.nrows() {
        // Frozen stores the reduced unknown vector. With both linear BVP
        // boundary conditions placed at the left edge, row 0 corresponds
        // to the first free mesh point, not to the boundary itself.
        let x = (i + 1) as f64 * h;
        let y_err = (solution[(i, 0)] - x).abs();
        let z_err = (solution[(i, 1)] - 1.0).abs();
        assert!(
            solution[(i, 0)].is_finite(),
            "frozen y contains non-finite values"
        );
        assert!(
            solution[(i, 1)].is_finite(),
            "frozen z contains non-finite values"
        );
        let err = y_err.max(z_err);
        sq_sum += err * err;
        max_abs = f64::max(max_abs, err);
    }
    let rms = (sq_sum / solution.nrows() as f64).sqrt();
    assert!(
        rms <= rms_tol,
        "frozen linear BVP RMS error too large: rms={rms:e}, tol={rms_tol:e}"
    );
    assert!(
        max_abs <= max_abs_tol,
        "frozen linear BVP max error too large: max={max_abs:e}, tol={max_abs_tol:e}"
    );
}

#[derive(Debug)]
struct FrozenCombustionStoryRow {
    source: &'static str,
    variant: &'static str,
    total_ms: f64,
    solution_diff: f64,
    symbolic_ms: f64,
    linear_ms: f64,
    jacobian_ms: f64,
    residual_ms: f64,
    initial_generate_ms: f64,
    initial_symbolic_jacobian_ms: f64,
    post_build_rebind_ms: f64,
    compile_link_ms: f64,
    residual_jobs: f64,
    jacobian_jobs: f64,
    iterations: usize,
    linear_solves: usize,
    jacobian_rebuilds: usize,
    selected_backend: String,
    build_policy: String,
}

fn frozen_combustion_solver(
    n_steps: usize,
    matrix: &'static str,
    config: GeneratedBackendConfig,
) -> NRBVP {
    let names = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
    let unknowns = Expr::parse_vector_expression(names.clone());
    let teta = unknowns[0].clone();
    let q = unknowns[1].clone();
    let c0 = unknowns[2].clone();
    let j0 = unknowns[3].clone();
    let j1 = unknowns[5].clone();

    let dt = Expr::Const(600.0);
    let t_scale = Expr::Const(600.0);
    let lambda = Expr::Const(0.07);
    let q_heat = Expr::Const(3000.0 * 1e3 * 0.034);
    let a = Expr::Const(1.3e5);
    let e = Expr::Const(5000.0 * 4.184);
    let m = Expr::Const(34.2 / 1000.0);
    let gas_r = Expr::Const(8.314);
    let ro_m = Expr::Const((34.2 / 1000.0) * 2e6 / (8.314 * 1500.0));
    let qm = Expr::Const((3e-4_f64).powi(2) / 600.0);
    let qs = Expr::Const((3e-4_f64).powi(2));
    let ro_d = Expr::Const(2.88e-4);
    let pe_d = Expr::Const(1.50e-3);
    let rate = a
        * Expr::exp(-e / (gas_r * (teta.clone() * t_scale.clone() + dt.clone())))
        * c0.clone()
        * (ro_m.clone() / Expr::Const(0.342));
    let eqs = vec![
        q.clone() / lambda,
        q * Expr::Const(0.0090168) - q_heat * rate.clone() * qm,
        j0.clone() / ro_d.clone(),
        j0 * pe_d.clone()
            - (m.clone() * Expr::Const(-1.0) * rate.clone() * ro_m.clone() / m.clone())
                * qs.clone(),
        j1.clone() / ro_d,
        j1 * pe_d - (m.clone() * rate * ro_m / m) * qs,
    ];
    let boundary_conditions = HashMap::from([
        ("Teta".to_string(), vec![(0, (1000.0 - 600.0) / 600.0)]),
        ("q".to_string(), vec![(1, 1e-10)]),
        ("C0".to_string(), vec![(0, 1.0)]),
        ("J0".to_string(), vec![(1, 1e-7)]),
        ("C1".to_string(), vec![(0, 1e-3)]),
        ("J1".to_string(), vec![(1, 1e-7)]),
    ]);
    let initial_guess = DMatrix::from_element(names.len(), n_steps, 0.99);
    let options = match matrix {
        "Sparse" => FrozenSolverOptions::sparse_frozen(),
        "Banded" => FrozenSolverOptions::banded_frozen(),
        _ => panic!("unsupported Frozen combustion story matrix route: {matrix}"),
    }
    .with_generated_backend_config(config)
    .with_tolerance(1e-6)
    .with_max_iterations(100);
    let mut solver = NRBVP::new_with_options(
        eqs,
        initial_guess,
        names.iter().map(|name| (*name).to_string()).collect(),
        "x".to_string(),
        boundary_conditions,
        0.0,
        1.0,
        n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn frozen_story_timer_ms(stats: &FrozenBvpStatistics, prefix: &str) -> f64 {
    stats
        .timers
        .iter()
        .find(|(name, _)| name.starts_with(prefix))
        .and_then(|(name, value)| {
            let value = value
                .split(',')
                .next_back()
                .and_then(|raw| raw.trim().parse::<f64>().ok())?;
            Some(if name.contains("ms") {
                value
            } else {
                value * 1_000.0
            })
        })
        .unwrap_or(f64::NAN)
}

fn frozen_story_diagnostic_ms(stats: &FrozenBvpStatistics, key: &str) -> f64 {
    stats
        .diagnostics
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn frozen_story_diagnostic_string(stats: &FrozenBvpStatistics, key: &str) -> String {
    stats
        .diagnostics
        .get(key)
        .cloned()
        .unwrap_or_else(|| "-".to_string())
}

fn frozen_story_linf_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn run_frozen_combustion_story_row(
    n_steps: usize,
    matrix: &'static str,
    source: &'static str,
    variant: &'static str,
    config: GeneratedBackendConfig,
    baseline: Option<&DMatrix<f64>>,
) -> (
    FrozenCombustionStoryRow,
    DMatrix<f64>,
    GeneratedBackendConfig,
) {
    let begin = Instant::now();
    let mut solver = frozen_combustion_solver(n_steps, matrix, config);
    solver
        .try_solve()
        .unwrap_or_else(|err| panic!("{source}/{variant} Frozen combustion solve failed: {err:?}"));
    collect_frozen_story_row(begin, solver, source, variant, baseline)
}

fn collect_frozen_story_row(
    begin: Instant,
    solver: NRBVP,
    source: &'static str,
    variant: &'static str,
    baseline: Option<&DMatrix<f64>>,
) -> (
    FrozenCombustionStoryRow,
    DMatrix<f64>,
    GeneratedBackendConfig,
) {
    let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
    let solution = solver
        .get_result()
        .expect("successful Frozen solve should expose its result");
    assert!(
        solution.iter().all(|value| value.is_finite()),
        "{source}/{variant} returned non-finite values"
    );
    let stats = solver.get_statistics();
    let row = FrozenCombustionStoryRow {
        source,
        variant,
        total_ms,
        solution_diff: baseline
            .map(|reference| frozen_story_linf_diff(&solution, reference))
            .unwrap_or(0.0),
        symbolic_ms: frozen_story_timer_ms(&stats, "Symbolic Operations"),
        linear_ms: frozen_story_timer_ms(&stats, "Linear System"),
        jacobian_ms: frozen_story_timer_ms(&stats, "Jacobian"),
        residual_ms: frozen_story_timer_ms(&stats, "Function"),
        initial_generate_ms: frozen_story_diagnostic_ms(
            &stats,
            "generated.handoff.initial_generate_wall_ms",
        ),
        initial_symbolic_jacobian_ms: frozen_story_diagnostic_ms(
            &stats,
            "generated.handoff.initial.symbolic_jacobian_time_ms",
        ),
        post_build_rebind_ms: frozen_story_diagnostic_ms(
            &stats,
            "generated.handoff.post_build_rebind_wall_ms",
        ),
        compile_link_ms: frozen_story_diagnostic_ms(&stats, "generated.aot.compile_link_ms"),
        residual_jobs: frozen_story_diagnostic_ms(&stats, "aot.runtime.residual.actual_jobs"),
        jacobian_jobs: frozen_story_diagnostic_ms(
            &stats,
            "aot.runtime.sparse_jacobian.actual_jobs",
        ),
        iterations: stats.counters["number of iterations"],
        linear_solves: stats.counters["number of solving linear systems"],
        jacobian_rebuilds: stats.counters["number of jacobians recalculations"],
        selected_backend: frozen_story_diagnostic_string(&stats, "generated.selected_backend"),
        build_policy: frozen_story_diagnostic_string(&stats, "aot.build_policy"),
    };
    (row, solution, solver.generated_backend_config().clone())
}

fn frozen_polynomial_two_point_solver(n_steps: usize, config: GeneratedBackendConfig) -> NRBVP {
    // Non-combustion nonlinear BVP with exact solution y = 1 + x^2:
    // y' = z,
    // z' = 2 + 0.1 * (y - (1 + x^2))^2.
    //
    // Frozen currently requires a boundary-condition key for every state
    // variable, so we use y(-1) and z(1), both taken from the exact profile.
    let values = vec!["y".to_string(), "z".to_string()];
    let t0 = -1.0;
    let t_end = 1.0;
    let h = (t_end - t0) / n_steps as f64;
    let mut guess = vec![0.0; values.len() * n_steps];
    for i in 0..n_steps {
        let x = t0 + i as f64 * h;
        let y = 1.0 + x * x;
        let z = 2.0 * x;
        guess[i * values.len()] = y;
        guess[i * values.len() + 1] = z;
    }

    let y_left = 2.0;
    let z_right = 2.0;
    let options = FrozenSolverOptions::banded_frozen()
        .with_generated_backend_config(config)
        .with_tolerance(1e-6)
        .with_max_iterations(40);
    let mut solver = NRBVP::new_with_options(
        vec![
            Expr::parse_expression("z"),
            Expr::parse_expression("2.0 + 0.1*(y - (1.0 + x*x))^2"),
        ],
        DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice()),
        values,
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0usize, y_left)]),
            ("z".to_string(), vec![(1usize, z_right)]),
        ]),
        t0,
        t_end,
        n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn run_frozen_polynomial_story_row(
    n_steps: usize,
    source: &'static str,
    variant: &'static str,
    config: GeneratedBackendConfig,
    baseline: Option<&DMatrix<f64>>,
) -> (
    FrozenCombustionStoryRow,
    DMatrix<f64>,
    GeneratedBackendConfig,
) {
    let begin = Instant::now();
    let mut solver = frozen_polynomial_two_point_solver(n_steps, config);
    let solve_result = solver.try_solve().unwrap_or_else(|err| {
        panic!("{source}/{variant} Frozen nonlinear polynomial solve failed: {err:?}")
    });
    assert!(
        solve_result.is_some(),
        "{source}/{variant} Frozen nonlinear polynomial solve did not converge within the configured iteration/tolerance budget"
    );
    collect_frozen_story_row(begin, solver, source, variant, baseline)
}

fn frozen_story_mean_std(values: &[f64]) -> (f64, f64) {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / finite.len() as f64;
    (mean, variance.sqrt())
}

fn print_frozen_combustion_story(title: &str, rows: &[FrozenCombustionStoryRow]) {
    println!("[BVP Frozen story] {title}: correctness/backend selection");
    println!("source   | variant    | selected_backend | build_policy    | solve_diff");
    println!("{}", "-".repeat(82));
    for row in rows {
        println!(
            "{:<8} | {:<10} | {:<16} | {:<15} | {:.6e}",
            row.source, row.variant, row.selected_backend, row.build_policy, row.solution_diff
        );
    }
    println!();
    println!("[BVP Frozen story] {title}: wall-clock and Newton stages; milliseconds");
    println!(
        "source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re"
    );
    println!("{}", "-".repeat(118));
    for row in rows {
        println!(
            "{:<8} | {:<10} | {:>8.3} | {:>11.3} | {:>9.3} | {:>6.3} | {:>6.3} | {:>5} | {:>6} | {:>6}",
            row.source,
            row.variant,
            row.total_ms,
            row.symbolic_ms,
            row.linear_ms,
            row.jacobian_ms,
            row.residual_ms,
            row.iterations,
            row.linear_solves,
            row.jacobian_rebuilds,
        );
    }
    println!();
    println!(
        "[BVP Frozen story] {title}: generated handoff and compiled callback stages; milliseconds"
    );
    println!(
        "source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs"
    );
    println!("{}", "-".repeat(120));
    for row in rows {
        println!(
            "{:<8} | {:<10} | {:>16.3} | {:>15.3} | {:>9.3} | {:>12.3} | {:>8.3} | {:>8.3}",
            row.source,
            row.variant,
            row.initial_generate_ms,
            row.initial_symbolic_jacobian_ms,
            row.post_build_rebind_ms,
            row.compile_link_ms,
            row.residual_jobs,
            row.jacobian_jobs,
        );
    }
    println!();
    println!("[BVP Frozen story] {title}: repeated-run summary; milliseconds");
    println!(
        "source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff"
    );
    println!("{}", "-".repeat(126));
    let mut identities = rows
        .iter()
        .map(|row| (row.source, row.variant))
        .collect::<Vec<_>>();
    identities.sort_unstable();
    identities.dedup();
    for (source, variant) in identities {
        let selected = rows
            .iter()
            .filter(|row| row.source == source && row.variant == variant)
            .collect::<Vec<_>>();
        let (total_mean, total_std) =
            frozen_story_mean_std(&selected.iter().map(|row| row.total_ms).collect::<Vec<_>>());
        let (symbolic_mean, symbolic_std) = frozen_story_mean_std(
            &selected
                .iter()
                .map(|row| row.symbolic_ms)
                .collect::<Vec<_>>(),
        );
        let (linear_mean, linear_std) =
            frozen_story_mean_std(&selected.iter().map(|row| row.linear_ms).collect::<Vec<_>>());
        let max_diff = selected
            .iter()
            .map(|row| row.solution_diff)
            .fold(0.0_f64, f64::max);
        println!(
            "{:<8} | {:<10} | {:>9.3} +/- {:<9.3} | {:>12.3} +/- {:<9.3} | {:>10.3} +/- {:<9.3} | {:.6e}",
            source,
            variant,
            total_mean,
            total_std,
            symbolic_mean,
            symbolic_std,
            linear_mean,
            linear_std,
            max_diff,
        );
    }
}

fn frozen_tcc_config(
    matrix: &'static str,
    policy: AotBuildPolicy,
    chunking: AotChunkingPolicy,
    execution: AotExecutionPolicy,
) -> GeneratedBackendConfig {
    let config = match matrix {
        "Sparse" => GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        "Banded" => GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
        _ => panic!("unsupported Frozen tcc story matrix route: {matrix}"),
    };
    config
        .with_aot_codegen_backend(AotCodegenBackend::C)
        .with_aot_c_compiler("tcc")
        .with_aot_compile_dev_fastest()
        .with_aot_chunking_policy(chunking)
        .with_aot_execution_policy(execution)
        .with_aot_build_policy(policy)
}

fn frozen_whole_chunking() -> AotChunkingPolicy {
    AotChunkingPolicy::with_parts(
        Some(ResidualChunkingStrategy::Whole),
        Some(SparseChunkingStrategy::Whole),
    )
}

fn frozen_chunk4_execution() -> (AotChunkingPolicy, AotExecutionPolicy) {
    (
        AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
        ),
        AotExecutionPolicy::Parallel(ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: Some(4),
            max_sparse_jobs: Some(4),
            fallback_policy: ParallelFallbackPolicy::Never,
        }),
    )
}

#[test]
#[ignore = "heavy Frozen combustion-1000 Banded AtomView Lambdify vs tcc whole/chunk4 end-to-end story; run in release with --nocapture"]
fn frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story() {
    let _test_report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_Damp_AOT_Frozen",
        concat!(
            module_path!(),
            "::frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story"
        ),
    );
    let n_steps = 1_000;
    let repetitions = 2;
    let mut rows = Vec::new();
    let (chunk4, parallel) = frozen_chunk4_execution();

    for _ in 0..repetitions {
        let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "Lambdify",
            "AtomView",
            GeneratedBackendConfig::banded_lambdify_defaults(),
            None,
        );
        rows.push(baseline_row);

        let (whole_row, _, _) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "AOT",
            "tcc/whole",
            frozen_tcc_config(
                "Banded",
                AotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                },
                frozen_whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            Some(&baseline),
        );
        rows.push(whole_row);

        let (chunked_row, _, _) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "AOT",
            "tcc/chunk4",
            frozen_tcc_config(
                "Banded",
                AotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                },
                chunk4,
                parallel.clone(),
            ),
            Some(&baseline),
        );
        rows.push(chunked_row);
    }

    print_frozen_combustion_story(
        "combustion-1000 Banded AtomView Lambdify vs tcc AOT cold routes",
        &rows,
    );
    assert!(
        rows.iter().all(|row| row.solution_diff <= 1e-5),
        "Frozen AOT variants must remain equivalent to the Lambdify baseline"
    );
    assert!(
        rows.iter()
            .filter(|row| row.source == "AOT")
            .all(|row| row.selected_backend == "AotCompiled"),
        "Frozen AOT cold routes must execute freshly compiled callbacks"
    );
    assert!(
        rows.iter()
            .filter(|row| row.variant == "tcc/chunk4")
            .all(|row| row.residual_jobs > 1.0 && row.jacobian_jobs > 1.0),
        "Frozen chunk4 route must expose real callback-level parallel execution"
    );
}

#[test]
#[ignore = "heavy Frozen combustion-1000 AOT artifact lifecycle story: BuildIfMissing followed by strict RequirePrebuilt reuse"]
fn frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story() {
    let _test_report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_Damp_AOT_Frozen",
        concat!(
            module_path!(),
            "::frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story"
        ),
    );
    let n_steps = 1_000;
    let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
        n_steps,
        "Banded",
        "Lambdify",
        "AtomView",
        GeneratedBackendConfig::banded_lambdify_defaults(),
        None,
    );
    let (built_row, _, built_config) = run_frozen_combustion_story_row(
        n_steps,
        "Banded",
        "AOT",
        "build",
        frozen_tcc_config(
            "Banded",
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            },
            frozen_whole_chunking(),
            AotExecutionPolicy::SequentialOnly,
        ),
        Some(&baseline),
    );
    assert!(
        built_config.resolver.is_some(),
        "BuildIfMissing must leave a resolver snapshot for strict reuse"
    );
    let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    let mut rows = vec![baseline_row, built_row];
    for _ in 0..3 {
        let (prebuilt_row, _, _) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "AOT",
            "prebuilt",
            strict_config.clone(),
            Some(&baseline),
        );
        rows.push(prebuilt_row);
    }

    print_frozen_combustion_story(
        "combustion-1000 Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
        &rows,
    );
    assert!(
        rows.iter().all(|row| row.solution_diff <= 1e-5),
        "Frozen lifecycle rows must remain equivalent to Lambdify"
    );
    assert!(
        rows.iter()
            .filter(|row| row.source == "AOT")
            .all(|row| row.selected_backend == "AotCompiled"),
        "both built and strict prebuilt rows must execute compiled callbacks"
    );
    assert!(
        rows.iter()
            .filter(|row| row.variant == "prebuilt")
            .all(|row| row.build_policy == "RequirePrebuilt"),
        "warm rows must be strict RequirePrebuilt executions, not fallback builds"
    );
}

#[test]
#[ignore = "heavy Frozen combustion-1000 Sparse AtomView tcc artifact lifecycle: BuildIfMissing followed by strict RequirePrebuilt reuse"]
fn frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story() {
    let _test_report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_Damp_AOT_Frozen",
        concat!(
            module_path!(),
            "::frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story"
        ),
    );
    let n_steps = 1_000;
    let sparse_lambdify = GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
    let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
        n_steps,
        "Sparse",
        "Lambdify",
        "AtomView",
        sparse_lambdify,
        None,
    );
    let (built_row, _, built_config) = run_frozen_combustion_story_row(
        n_steps,
        "Sparse",
        "AOT",
        "build",
        frozen_tcc_config(
            "Sparse",
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            },
            frozen_whole_chunking(),
            AotExecutionPolicy::SequentialOnly,
        ),
        Some(&baseline),
    );
    assert!(
        built_config.resolver.is_some(),
        "Sparse BuildIfMissing must leave a resolver snapshot for strict reuse"
    );
    let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    let mut rows = vec![baseline_row, built_row];
    for _ in 0..3 {
        let (prebuilt_row, _, _) = run_frozen_combustion_story_row(
            n_steps,
            "Sparse",
            "AOT",
            "prebuilt",
            strict_config.clone(),
            Some(&baseline),
        );
        rows.push(prebuilt_row);
    }

    print_frozen_combustion_story(
        "combustion-1000 Sparse AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
        &rows,
    );
    assert!(
        rows.iter().all(|row| row.solution_diff <= 1e-5),
        "Frozen Sparse lifecycle rows must remain equivalent to Lambdify"
    );
    assert!(
        rows.iter()
            .filter(|row| row.source == "AOT")
            .all(|row| row.selected_backend == "AotCompiled"),
        "Frozen Sparse build and strict prebuilt rows must execute compiled callbacks"
    );
    assert!(
        rows.iter()
            .filter(|row| row.variant == "prebuilt")
            .all(|row| { row.build_policy == "RequirePrebuilt" && row.compile_link_ms.is_nan() }),
        "Frozen Sparse prebuilt rows must neither fall back nor compile again"
    );
}

#[test]
#[ignore = "Frozen non-combustion nonlinear polynomial BVP: Banded AtomView Lambdify vs tcc BuildIfMissing -> RequirePrebuilt"]
fn frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story() {
    let n_steps = 80;
    let (baseline_row, baseline, _) = run_frozen_polynomial_story_row(
        n_steps,
        "Lambdify",
        "AtomView",
        GeneratedBackendConfig::banded_lambdify_defaults(),
        None,
    );
    let (built_row, _, built_config) = run_frozen_polynomial_story_row(
        n_steps,
        "AOT",
        "build",
        frozen_tcc_config(
            "Banded",
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            },
            frozen_whole_chunking(),
            AotExecutionPolicy::SequentialOnly,
        ),
        Some(&baseline),
    );
    assert!(
        built_config.resolver.is_some(),
        "Polynomial BuildIfMissing must leave a resolver snapshot for strict reuse"
    );
    let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    let mut rows = vec![baseline_row, built_row];
    for _ in 0..2 {
        let (prebuilt_row, _, _) = run_frozen_polynomial_story_row(
            n_steps,
            "AOT",
            "prebuilt",
            strict_config.clone(),
            Some(&baseline),
        );
        rows.push(prebuilt_row);
    }

    print_frozen_combustion_story(
        "nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
        &rows,
    );
    assert!(
        rows.iter().all(|row| row.solution_diff <= 1e-6),
        "Frozen nonlinear polynomial lifecycle rows must remain equivalent to Lambdify"
    );
    assert!(
        rows.iter()
            .filter(|row| row.source == "AOT")
            .all(|row| row.selected_backend == "AotCompiled"),
        "Frozen nonlinear polynomial build and strict prebuilt rows must execute compiled callbacks"
    );
    assert!(
        rows.iter()
            .filter(|row| row.variant == "prebuilt")
            .all(|row| { row.build_policy == "RequirePrebuilt" && row.compile_link_ms.is_nan() }),
        "Frozen nonlinear polynomial prebuilt rows must neither fall back nor compile again"
    );
}

#[test]
fn generated_backend_surface_builder_methods_update_solver_config() {
    let solver = sparse_surface_test_solver()
        .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
        .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
        .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
        ))
        .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        solver.aot_execution_policy(),
        &AotExecutionPolicy::SequentialOnly
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
    assert_eq!(
        solver.aot_chunking_policy(),
        AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
        )
    );
    assert_eq!(
        solver.atom_optimization_profile(),
        AtomOptimizationProfile::NoCse
    );
}

#[test]
fn symbolic_assembly_backend_is_exposed_on_solver_surface() {
    let mut solver = sparse_surface_test_solver()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);

    assert_eq!(
        solver.symbolic_assembly_backend(),
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        solver.generated_backend_config().symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );

    solver.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    assert_eq!(
        solver.symbolic_assembly_backend(),
        BvpSymbolicAssemblyBackend::ExprLegacy
    );
    assert_eq!(
        solver.generated_backend_config().symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::ExprLegacy
    );
}

#[test]
fn banded_frozen_lambdify_mode_sets_banded_matrix_and_lambdify_policy() {
    let options = FrozenSolverOptions::banded_frozen().with_banded_lambdify();

    assert_eq!(options.method, "Sparse");
    assert_eq!(
        options.generated_backend_config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        options.generated_backend_config.backend_policy_override,
        Some(BackendSelectionPolicy::LambdifyOnly)
    );
    assert_eq!(
        options.generated_backend_config.matrix_backend_override,
        Some(MatrixBackend::Banded)
    );
    assert_eq!(
        options.generated_backend_config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        options.generated_backend_config.aot_build_policy,
        AotBuildPolicy::UseIfAvailable
    );
}

#[test]
fn banded_frozen_generated_backend_mode_build_if_missing_sets_release_aot_policy() {
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_generated_backend_mode(BandedGeneratedBackendMode::BuildIfMissingRelease);

    assert_eq!(
        options.generated_backend_config.backend_policy_override,
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        options.generated_backend_config.matrix_backend_override,
        Some(MatrixBackend::Banded)
    );
    assert_eq!(
        options.generated_backend_config.aot_build_policy,
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
}

#[test]
fn frozen_sparse_lambdify_linear_bvp_solves_against_exact_profile() {
    let options = FrozenSolverOptions::sparse_frozen()
        .with_generated_backend_config(
            GeneratedBackendConfig::sparse_defaults()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly)),
        )
        .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
        .with_tolerance(1e-8)
        .with_max_iterations(20);
    let mut solver = frozen_linear_solver(24, options);
    solver.dont_save_log(true);

    solver
        .try_solve()
        .expect("sparse frozen lambdify linear BVP should solve");

    assert_frozen_linear_solution_quality(&solver, 24, 1e-10, 1e-9);
    assert!(
        solver.jac.is_some(),
        "sparse frozen route should prepare a Jacobian"
    );
    assert!(
        !solver.variable_string.is_empty(),
        "sparse frozen route should prepare reduced variable metadata"
    );
}

#[test]
fn frozen_banded_default_atomview_lambdify_linear_bvp_solves_against_exact_profile() {
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
        .with_tolerance(1e-8)
        .with_max_iterations(20);
    assert_eq!(
        options.generated_backend_config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    let mut solver = frozen_linear_solver(24, options);
    solver.dont_save_log(true);

    solver
        .try_solve()
        .expect("default banded frozen AtomView lambdify linear BVP should solve");

    assert_frozen_linear_solution_quality(&solver, 24, 1e-7, 1e-6);
    assert!(
        solver.jac.is_some(),
        "banded frozen route should prepare a Jacobian"
    );
    assert!(
        solver.bandwidth.0 + solver.bandwidth.1 > 0,
        "banded frozen route should expose non-empty bandwidth metadata"
    );
    let stats = solver.get_statistics();
    assert_eq!(
        stats.diagnostics.get("generated.selected_backend"),
        Some(&"Lambdify".to_string()),
        "Frozen statistics must report the backend that supplied callbacks"
    );
    assert_eq!(
        stats.diagnostics.get("generated.symbolic_assembly_backend"),
        Some(&"AtomView".to_string())
    );
    assert!(
        stats
            .diagnostics
            .contains_key("generated.handoff.initial_generate_wall_ms"),
        "Frozen must preserve symbolic handoff timing diagnostics"
    );
    assert!(
        stats.counters["number of iterations"] > 0
            && stats.counters["number of jacobians recalculations"] > 0
            && stats.counters["number of solving linear systems"] > 0,
        "Frozen end-to-end solve must expose its Newton work counters: {:?}",
        stats.counters
    );
    assert!(
        stats
            .timers
            .keys()
            .any(|key| key.starts_with("Symbolic Operations")),
        "Frozen end-to-end solve must expose backend preparation timing"
    );
}

#[test]
fn try_eq_generate_surfaces_missing_prebuilt_aot_as_typed_error() {
    let mut solver =
        sparse_surface_test_solver_with_naive_strategy().with_sparse_aot_require_prebuilt();

    let err = solver
        .try_eq_generate()
        .expect_err("try_eq_generate should return a typed AOT availability error");

    assert!(matches!(
        err,
        BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
    ));
}

#[test]
fn try_solve_surfaces_missing_prebuilt_aot_as_typed_error() {
    let mut solver =
        sparse_surface_test_solver_with_naive_strategy().with_sparse_aot_require_prebuilt();
    solver.dont_save_log(true);

    let err = solver
        .try_solve()
        .expect_err("try_solve should return a typed AOT availability error");

    assert!(matches!(
        err,
        BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
    ));
}

#[test]
fn sparse_generated_backend_presets_are_exposed_on_solver_surface() {
    let solver = sparse_surface_test_solver().with_sparse_aot_build_if_missing_release();

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        solver.aot_build_policy(),
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
}

#[test]
fn sparse_generated_backend_mode_is_exposed_on_solver_surface() {
    let mut solver = sparse_surface_test_solver()
        .with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt);

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);

    solver.set_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease);
    assert_eq!(
        solver.aot_build_policy(),
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
}

#[test]
fn sparse_frozen_options_preset_sets_production_defaults() {
    let options = FrozenSolverOptions::sparse_frozen();

    assert_eq!(options.scheme, "forward");
    assert_eq!(options.strategy, "Frozen");
    assert_eq!(options.method, "Sparse");
    assert_eq!(
        options.generated_backend_config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        options.generated_backend_config.backend_policy_override,
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        options.generated_backend_config.aot_build_policy,
        AotBuildPolicy::UseIfAvailable
    );
    assert_eq!(
        options.generated_backend_config.aot_execution_policy,
        AotExecutionPolicy::Auto
    );
    assert_eq!(
        options.generated_backend_config.aot_chunking_policy,
        AotChunkingPolicy::default()
    );
}

#[test]
fn banded_frozen_options_preset_uses_auto_aot_chunking_defaults() {
    let options = FrozenSolverOptions::banded_frozen();

    assert_eq!(
        options.generated_backend_config.matrix_backend_override,
        Some(MatrixBackend::Banded)
    );
    assert_eq!(
        options.generated_backend_config.aot_execution_policy,
        AotExecutionPolicy::Auto
    );
    assert_eq!(
        options.generated_backend_config.aot_chunking_policy,
        AotChunkingPolicy::default()
    );
}

#[test]
fn frozen_options_scheme_builder_methods_set_legacy_scheme_flag() {
    let options = FrozenSolverOptions::sparse_frozen().trapezoid_derivative();
    assert_eq!(options.scheme, "trapezoid");

    let options = options.forward_derivative();
    assert_eq!(options.scheme, "forward");

    let options = options.with_scheme(BvpDerivativeScheme::Trapezoid);
    assert_eq!(options.scheme, "trapezoid");

    let options = options.with_scheme_name("custom-experimental");
    assert_eq!(options.scheme, "custom-experimental");
}

#[test]
fn frozen_solver_scheme_builder_methods_feed_generated_request() {
    let solver = sparse_surface_test_solver().trapezoid_derivative();
    assert_eq!(solver.scheme, "trapezoid");
    assert_eq!(solver.build_solver_request().scheme, "trapezoid");

    let solver = solver.forward_derivative();
    assert_eq!(solver.scheme, "forward");
    assert_eq!(solver.build_solver_request().scheme, "forward");

    let solver = solver.with_scheme(BvpDerivativeScheme::Trapezoid);
    assert_eq!(solver.scheme, "trapezoid");
    assert_eq!(solver.build_solver_request().scheme, "trapezoid");

    let solver = solver.with_scheme_name("custom-experimental");
    assert_eq!(solver.scheme, "custom-experimental");
    assert_eq!(solver.build_solver_request().scheme, "custom-experimental");
}

#[test]
fn dense_frozen_options_preset_sets_dense_defaults() {
    let options = FrozenSolverOptions::dense_frozen();

    assert_eq!(options.strategy, "Frozen");
    assert_eq!(options.method, "Dense");
}

#[test]
fn dense_naive_options_preset_sets_dense_defaults() {
    let options = FrozenSolverOptions::dense_naive();

    assert_eq!(options.strategy, "Naive");
    assert!(options.strategy_params.is_none());
    assert_eq!(options.method, "Dense");
}

#[test]
fn constructor_style_sparse_generated_backend_mode_sets_solver_config() {
    let solver = NRBVP::new_with_sparse_generated_backend_mode(
        vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
        DMatrix::from_element(2, 4, 0.1),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        4,
        "Frozen".to_string(),
        None,
        None,
        "Sparse".to_string(),
        1e-6,
        10,
        SparseGeneratedBackendMode::RequirePrebuilt,
    );

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
}

#[test]
fn options_style_solver_setup_sets_sparse_generated_backend_mode() {
    let options = FrozenSolverOptions::sparse_frozen().with_sparse_aot_require_prebuilt();

    let solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
        DMatrix::from_element(2, 4, 0.1),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        4,
        options,
    );

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
}

#[test]
fn test_newton_raphson_solver() {
    // Define a simple equation: x^2 - 4 = 0
    let eq1 = Expr::parse_expression("y-z");
    let eq2 = Expr::parse_expression("-z");
    let eq_system = vec![eq1, eq2];

    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-2;
    let max_iterations = 100;

    let t0 = 0.0;
    let t_end = 1.0;
    let n_steps = 100;
    let ones = vec![1.0; values.len() * n_steps];
    let initial_guess: DMatrix<f64> =
        DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
    let mut BorderConditions = HashMap::new();
    BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
    BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
    assert_eq!(&eq_system.len(), &2);
    let options = FrozenSolverOptions::dense_naive()
        .with_tolerance(tolerance)
        .with_max_iterations(max_iterations);
    let mut nr = NRBVP::new_with_options(
        eq_system,
        initial_guess,
        values,
        arg,
        BorderConditions,
        t0,
        t_end,
        n_steps,
        options,
    );
    nr.try_eq_generate()
        .expect("dense frozen solver should generate through the fallible API");

    assert_eq!(nr.eq_system.len(), 2);
    nr.dont_save_log(true);
    // Solve the equation at t=0 with initial guess y=[2.0]
    //    nr.set_new_step(0.0, DVector::from_element(1, 2.0), DVector::from_element(1, 2.0));
    let _solution = nr
        .try_solve()
        .expect("dense frozen solver should solve through the fallible API")
        .unwrap();
}

#[test]
fn sparse_eq_generate_uses_bundle_handoff_without_breaking_metadata() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();

    solver
        .try_eq_generate()
        .expect("sparse frozen handoff should generate through the fallible API");

    assert!(solver.jac.is_some());
    assert!(!solver.variable_string.is_empty());
    assert!(solver.bandwidth.0 + solver.bandwidth.1 > 0);
    let _ = &solver.fun;
}

#[test]
fn prepared_frozen_solver_detects_direct_residual_callback_replacement() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    solver
        .try_eq_generate()
        .expect("baseline Frozen preparation should succeed");

    let replacement = convert_to_fun(Box::new(|_, y: &dyn VectorType| y.clone_box()));
    let _old_callback = std::mem::replace(&mut solver.fun, replacement);
    let error = solver
        .try_solver_prepared()
        .expect_err("direct Frozen callback replacement must invalidate the plan");

    assert!(matches!(
        error,
        BvpBackendIntegrationError::PreparedRuntimeInvalidated { reason }
            if reason.contains("stale")
    ));
}

#[test]
fn frozen_mesh_setter_invalidates_prepared_callbacks_and_rejects_one_point_mesh() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    solver
        .try_eq_generate()
        .expect("baseline Frozen preparation should succeed");
    let original_guess = solver.initial_guess.clone();

    solver
        .try_set_mesh(0.0, 2.0, 6)
        .expect("valid Frozen mesh change should be accepted");
    assert_eq!(solver.n_steps, 6);
    assert_eq!(solver.x_mesh.len(), 6);
    assert_eq!(solver.initial_guess.ncols(), 6);
    assert_eq!(solver.initial_guess[(0, 0)], original_guess[(0, 0)]);
    assert!(solver.jac.is_none());
    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.variable_string.is_empty());
    assert!(
        solver
            .try_solver_prepared()
            .expect_err("mesh mutation must invalidate prepared Frozen callbacks")
            .to_string()
            .contains("stale")
    );

    let error = solver
        .try_set_mesh(0.0, 1.0, 1)
        .expect_err("a one-point Frozen mesh must be rejected");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::InvalidProblem { field, .. } if field == "n_steps"
    ));
}

#[test]
fn numeric_only_is_rejected_for_frozen_solver_instead_of_lambdify_fallback() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

    let err = solver
        .try_eq_generate()
        .expect_err("Frozen NumericOnly must not silently fall back to symbolic lambdify");
    assert!(matches!(
        err,
        BvpBackendIntegrationError::PipelinePanicked(message)
            if message.contains("not available for the frozen BVP solver")
    ));
}

#[test]
fn build_solver_request_carries_optional_aot_resolver() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));

    let request = solver.build_solver_request();
    assert!(request.resolver.is_some());
}

#[test]
fn build_solver_request_carries_parameter_names_and_values() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    solver.set_params(Some(&["alpha", "beta"]));
    solver.set_param_values(Some(vec![1.5, -0.25]));

    let request = solver.build_solver_request();
    assert_eq!(
        request.param_names,
        Some(vec!["alpha".to_string(), "beta".to_string()])
    );
    assert_eq!(request.param_values, Some(vec![1.5, -0.25]));
}

#[test]
#[should_panic(expected = "expected exactly 2 values for declared symbolic parameters")]
fn solver_surface_rejects_parameter_value_length_mismatch() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    solver.set_params(Some(&["alpha", "beta"]));
    solver.set_param_values(Some(vec![1.5]));
}

#[test]
fn build_solver_request_uses_backend_policy_override_when_present() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

    let request = solver.build_solver_request();
    assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
}

#[test]
fn generated_backend_config_is_exposed_as_user_facing_solver_setting() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();

    let config = GeneratedBackendConfig::with_parts(
        Some(BackendSelectionPolicy::NumericOnly),
        Some(AotResolver::new(AotRegistry::new())),
    );
    solver.set_generated_backend_config(config);

    let request = solver.build_solver_request();
    assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
    assert!(request.resolver.is_some());
    assert_eq!(
        solver.generated_backend_config().backend_policy_override,
        Some(BackendSelectionPolicy::NumericOnly)
    );
}

#[test]
fn generated_backend_config_can_be_applied_during_solver_construction() {
    let solver = sparse_surface_test_solver_with_naive_strategy().with_generated_backend_config(
        GeneratedBackendConfig::with_parts(
            Some(BackendSelectionPolicy::NumericOnly),
            Some(AotResolver::new(AotRegistry::new())),
        ),
    );

    assert_eq!(
        solver.generated_backend_config().backend_policy_override,
        Some(BackendSelectionPolicy::NumericOnly)
    );
    assert!(solver.generated_backend_config().resolver.is_some());
}

#[test]
fn cleanup_registered_aot_artifacts_is_safe_without_registered_artifacts() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy();
    assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);

    solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));
    assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);
}

#[test]
fn build_solver_request_carries_surface_aot_policies() {
    let config = GeneratedBackendConfig::new()
        .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
        .with_resolver(Some(AotResolver::new(AotRegistry::new())))
        .with_aot_execution_policy(AotExecutionPolicy::Parallel(ParallelExecutorConfig {
            jobs_per_worker: 2,
            max_residual_jobs: Some(4),
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        }))
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
        .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 6,
            }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
        ))
        .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

    let solver =
        sparse_surface_test_solver_with_naive_strategy().with_generated_backend_config(config);

    let request = solver.build_solver_request();
    assert_eq!(request.aot_build_policy, AotBuildPolicy::RequirePrebuilt);
    assert_eq!(
        request.aot_chunking_policy.residual,
        Some(ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 6
        })
    );
    assert_eq!(
        request.aot_chunking_policy.sparse_jacobian,
        Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 })
    );
    assert_eq!(
        request.atom_optimization_profile,
        AtomOptimizationProfile::NoCse
    );
    match request.aot_execution_policy {
        AotExecutionPolicy::Parallel(inner) => {
            assert_eq!(inner.jobs_per_worker, 2);
            assert_eq!(inner.max_residual_jobs, Some(4));
            assert_eq!(inner.max_sparse_jobs, Some(2));
        }
        other => std::panic!("expected parallel execution policy, got {other:?}"),
    }
    let _ = AotBuildProfile::Debug;
}

#[test]
fn eq_generate_build_if_missing_saves_compiled_resolver_for_next_request() {
    let mut solver = sparse_surface_test_solver_with_naive_strategy()
        .with_generated_backend_config(
            GeneratedBackendConfig::new()
                .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
                .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                }),
        );

    solver
        .try_eq_generate()
        .expect("build-if-missing path should generate through the fallible API");

    let saved_resolver = solver
        .generated_backend_config()
        .resolver
        .as_ref()
        .expect("first build-if-missing run should save updated resolver");
    assert!(
        !saved_resolver.registry().is_empty(),
        "first build-if-missing run should register at least one compiled artifact"
    );

    let updated_config = solver
        .generated_backend_config()
        .clone()
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    solver.set_generated_backend_config(updated_config);

    let next_request = solver.build_solver_request();
    assert!(next_request.resolver.is_some());

    let next_state = next_request
        .generate()
        .expect("next request should reuse the saved compiled resolver and generate successfully");
    assert!(
        next_state.jac.is_some(),
        "successful generation through the saved resolver should still provide a Jacobian callback"
    );
}

#[test]
fn second_eq_generate_reuses_saved_resolver_and_runs_linked_compiled_backend() {
    let values = vec!["z".to_string(), "y".to_string()];
    let n_steps = 8;
    let mut solver = sparse_surface_test_solver_with_naive_strategy()
        .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
        .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        });

    solver
        .try_eq_generate()
        .expect("first sparse generation should succeed through the fallible API");
    let saved_resolver = solver
        .generated_backend_config()
        .resolver
        .clone()
        .expect("first build-if-missing run should save updated resolver");

    let y = Col::from_fn(values.len() * n_steps, |index| 0.3 + index as f64 * 0.02);
    let baseline = solver.fun.call(0.0, &y).to_DVectorType();

    let problem_keys = saved_resolver.registry().problem_keys();
    assert_eq!(
        problem_keys.len(),
        1,
        "build-if-missing should register exactly one artifact for this isolated test"
    );
    let problem_key = problem_keys[0].clone();
    let resolved = saved_resolver.resolve_by_problem_key(&problem_key);
    assert!(
        resolved.is_compiled(),
        "saved resolver should see compiled artifact"
    );

    let baseline_values: Vec<f64> = baseline.iter().copied().collect();
    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        problem_key.clone(),
        resolved.registered.manifest.io.residual_len,
        (
            resolved.registered.manifest.io.jacobian_rows,
            resolved.registered.manifest.io.jacobian_cols,
        ),
        resolved.registered.manifest.io.jacobian_nnz.unwrap_or(0),
        Arc::new(move |_args, out| {
            for (dst, src) in out.iter_mut().zip(baseline_values.iter()) {
                *dst = *src + 55.0;
            }
        }),
        Arc::new(move |_args, out| {
            for (index, value) in out.iter_mut().enumerate() {
                *value = 900.0 + index as f64;
            }
        }),
    ));

    solver.set_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    solver
        .try_eq_generate()
        .expect("second sparse generation should reuse resolver through the fallible API");
    let residual = solver.fun.call(0.0, &y).to_DVectorType();

    for (actual, expected) in residual.iter().zip(baseline.iter()) {
        assert!((actual - (expected + 55.0)).abs() < 1e-10);
    }

    unregister_linked_sparse_backend(&problem_key);
}
