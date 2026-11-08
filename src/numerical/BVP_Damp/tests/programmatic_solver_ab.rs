use std::{collections::HashMap, time::Instant};

use crate::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonOptions, CorrectionNewtonTelemetryMode,
};
use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{DampedSolverOptions, NRBVP, SolverParams};
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use crate::numerical::BVP_Damp::telemetry::{BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode};
use crate::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSciSparseFactorization, BvpSolver,
    BvpSolverConfig, BvpSolverTelemetrySnapshot,
};
use crate::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciOptions, BvpSciTelemetryMode,
};
use crate::symbolic::View::{conversions::try_expr_to_atom, state::Symbol};
use crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

const TASK_NODES: usize = 21;
const FROZEN_NODES: usize = 10_001;

#[derive(tabled::Tabled)]
struct RouteRow {
    route: &'static str,
    requested_nodes: usize,
    outcome: String,
    native_status: String,
    last_native_event: String,
    actual_nodes: usize,
    iterations: u64,
    residuals: u64,
    jacobians: u64,
    linear_solves: u64,
    rejections: u64,
    boundary_error: String,
    max_abs_state: String,
    wall_ms: String,
    #[tabled(skip)]
    solution_values: Vec<f64>,
}

fn programmatic_problem(nodes: usize) -> BvpProblem {
    let mesh = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect();
    BvpProblem::new(
        vec![
            Expr::parse_expression("y-z"),
            Expr::parse_expression("-z^3"),
        ],
        vec!["z".into(), "y".into()],
        "x",
        mesh,
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 1.0,
            },
            BvpBoundaryCondition {
                state: 1,
                side: BvpBoundarySide::Right,
                value: 1.0,
            },
        ],
    )
    .expect("programmatic reference problem should validate")
}

fn damped_options(
    method: &str,
    bounds: [(f64, f64); 2],
    strategy_params: SolverParams,
) -> DampedSolverOptions {
    let generated_backend = if method == "Sparse" {
        GeneratedBackendConfig::sparse_defaults()
    } else {
        GeneratedBackendConfig::default()
    }
    .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
    let mut options = if method == "Sparse" {
        DampedSolverOptions::sparse_damped()
    } else {
        DampedSolverOptions::dense_damped()
    };
    options.method = method.to_string();
    options
        .with_generated_backend_config(generated_backend)
        .with_abs_tolerance(1e-5)
        .with_max_iterations(20)
        .with_strategy_params(Some(strategy_params))
        .with_rel_tolerance(HashMap::from([
            ("z".to_string(), 1e-4),
            ("y".to_string(), 1e-4),
        ]))
        .with_bounds(HashMap::from([
            ("z".to_string(), bounds[0]),
            ("y".to_string(), bounds[1]),
        ]))
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(1024),
        )
}

fn damped_route(
    nodes: usize,
    method: &str,
    bounds: [(f64, f64); 2],
    strategy_params: SolverParams,
) -> BvpSolver {
    damped_route_with_scheme(nodes, method, bounds, strategy_params, "forward")
}

fn damped_route_with_scheme(
    nodes: usize,
    method: &str,
    bounds: [(f64, f64); 2],
    strategy_params: SolverParams,
    scheme: &str,
) -> BvpSolver {
    BvpSolver::new(
        programmatic_problem(nodes),
        BvpSolverConfig::Damped {
            options: damped_options(method, bounds, strategy_params).with_scheme_name(scheme),
            initial_guess: DMatrix::zeros(2, nodes - 1),
        },
    )
    .expect("programmatic Damped route should prepare")
}

fn damped_route_with_optional_bounds(
    nodes: usize,
    bounds: Option<[(f64, f64); 2]>,
    guess_value: f64,
) -> BvpSolver {
    let mut options = damped_options(
        "Sparse",
        bounds.unwrap_or([(-1.0e6, 1.0e6); 2]),
        SolverParams::default(),
    );
    if bounds.is_none() {
        options.bounds = None;
    }
    BvpSolver::new(
        programmatic_problem(nodes),
        BvpSolverConfig::Damped {
            options,
            initial_guess: DMatrix::from_element(2, nodes - 1, guess_value),
        },
    )
    .expect("Damped bounds-sweep route should prepare")
}

#[test]
fn task_rhs_parsing_and_expr_atom_derivatives_match_analytic_values() {
    let rhs = [
        Expr::parse_expression("y-z"),
        Expr::parse_expression("-z^3"),
    ];
    let variables = ["z", "y"];
    let values = [
        rhs[0].lambdify1(&variables),
        rhs[1].lambdify1(&variables),
        rhs[0].diff("z").simplify().lambdify1(&variables),
        rhs[0].diff("y").simplify().lambdify1(&variables),
        rhs[1].diff("z").simplify().lambdify1(&variables),
        rhs[1].diff("y").simplify().lambdify1(&variables),
    ];
    let atoms = rhs.map(|expression| try_expr_to_atom(&expression).unwrap());
    let atom_jacobian = [
        atoms[0]
            .try_derivative(Symbol::new(crate::wrap_symbol!("z")))
            .unwrap(),
        atoms[0]
            .try_derivative(Symbol::new(crate::wrap_symbol!("y")))
            .unwrap(),
        atoms[1]
            .try_derivative(Symbol::new(crate::wrap_symbol!("z")))
            .unwrap(),
        atoms[1]
            .try_derivative(Symbol::new(crate::wrap_symbol!("y")))
            .unwrap(),
    ];
    let atom_values = [
        atoms[0].lambdify_borrowed_thread_safe(&variables),
        atoms[1].lambdify_borrowed_thread_safe(&variables),
        atom_jacobian[0].lambdify_borrowed_thread_safe(&variables),
        atom_jacobian[1].lambdify_borrowed_thread_safe(&variables),
        atom_jacobian[2].lambdify_borrowed_thread_safe(&variables),
        atom_jacobian[3].lambdify_borrowed_thread_safe(&variables),
    ];

    for z in [-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
        let args = [z, 0.75];
        let expected = [0.75 - z, -z.powi(3), -1.0, 1.0, -3.0 * z * z, 0.0];
        for index in 0..values.len() {
            let expr_value = values[index](&args);
            let atom_value = atom_values[index](&args);
            assert!(
                (expr_value - expected[index]).abs() <= 1e-12,
                "Expr component {index} at z={z}: got {expr_value}, expected {}",
                expected[index]
            );
            assert!(
                (atom_value - expected[index]).abs() <= 1e-12,
                "AtomView component {index} at z={z}: got {atom_value}, expected {}",
                expected[index]
            );
        }
    }
}

fn damped_full_jacobian_fd_error(scheme: &str, assembly: BvpSymbolicAssemblyBackend) -> f64 {
    let generated = GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(assembly);
    let mut facade = BvpSolver::new(
        programmatic_problem(11),
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::sparse_damped()
                .with_generated_backend_config(generated)
                .with_scheme_name(scheme)
                .with_strategy_params(Some(SolverParams::default()))
                .with_abs_tolerance(1e-5)
                .with_rel_tolerance(HashMap::from([("z".into(), 1e-4), ("y".into(), 1e-4)]))
                .with_bounds(HashMap::from([
                    ("z".into(), (-10.0, 10.0)),
                    ("y".into(), (-7.0, 7.0)),
                ]))
                .with_max_iterations(20),
            initial_guess: DMatrix::zeros(2, 10),
        },
    )
    .unwrap();
    let crate::numerical::BVP_api::BvpNativeSolver::Damped(solver) = facade.native_mut() else {
        panic!("expected Damped native solver");
    };
    solver
        .try_set_initial_guess(solver.initial_guess.clone())
        .unwrap();
    solver.try_eq_generate(None, None).unwrap();

    let unknown_count = solver.y.len();
    let initial_state = solver.y.to_DVectorType();
    assert_eq!(initial_state.len(), unknown_count);
    let probe_states = [
        initial_state,
        DVector::from_fn(unknown_count, |index, _| {
            0.15 * ((index + 1) as f64 * 0.37).sin()
        }),
    ];
    let mut max_scaled_error = 0.0_f64;
    for state in probe_states {
        let typed_state = Vectors_type_casting(&state, "Sparse".into());
        let analytic = solver
            .jac
            .as_mut()
            .unwrap()
            .try_call(solver.p, &*typed_state)
            .unwrap_or_else(|error| {
                panic!("{assembly:?}/{scheme} Jacobian callback failed: {error:?}")
            })
            .to_DMatrixType();
        for column in 0..unknown_count {
            let step = 1e-6 * (1.0 + state[column].abs());
            let mut plus = state.clone();
            let mut minus = state.clone();
            plus[column] += step;
            minus[column] -= step;
            let plus = Vectors_type_casting(&plus, "Sparse".into());
            let minus = Vectors_type_casting(&minus, "Sparse".into());
            let plus_residual = solver
                .fun
                .try_call(solver.p, &*plus)
                .unwrap()
                .to_DVectorType();
            let minus_residual = solver
                .fun
                .try_call(solver.p, &*minus)
                .unwrap()
                .to_DVectorType();
            for row in 0..unknown_count {
                let finite_difference = (plus_residual[row] - minus_residual[row]) / (2.0 * step);
                let exact = analytic[(row, column)];
                max_scaled_error =
                    max_scaled_error.max((exact - finite_difference).abs() / (1.0 + exact.abs()));
            }
        }
    }
    max_scaled_error
}

#[test]
fn damped_forward_and_trapezoid_jacobians_match_their_residuals() {
    for assembly in [
        BvpSymbolicAssemblyBackend::ExprLegacy,
        BvpSymbolicAssemblyBackend::AtomView,
    ] {
        for scheme in ["forward", "trapezoid"] {
            let error = damped_full_jacobian_fd_error(scheme, assembly);
            println!("Damped {assembly:?}/{scheme}: max scaled Jacobian FD error={error:.3e}");
            assert!(
                error < 2e-5,
                "Damped {assembly:?}/{scheme} Jacobian mismatch: scaled error={error:e}"
            );
        }
    }
}

#[test]
fn parsed_python_style_combustion_fixture_matches_native_damped_route() {
    let density_m = (34.2 / 1000.0) * 2.0e6 / (8.314 * 1500.0);
    let rate = format!(
        "1.3e5 * exp(-20920 / (8.314 * (Teta * 600 + 600))) * C0 * ({density_m:.17e} / 0.342)"
    );
    let equations = vec![
        Expr::parse_expression("q / 0.07"),
        Expr::parse_expression(&format!("q * 0.0090168 - 102000 * ({rate}) * 1.5e-10")),
        Expr::parse_expression("J0 / 2.88e-4"),
        Expr::parse_expression(&format!("J0 * 1.5e-3 + ({rate}) * {density_m:.17e} * 9e-8")),
        Expr::parse_expression("J1 / 2.88e-4"),
        Expr::parse_expression(&format!("J1 * 1.5e-3 - ({rate}) * {density_m:.17e} * 9e-8")),
    ];
    let names = ["Teta", "q", "C0", "J0", "C1", "J1"];
    let bounds = HashMap::from([
        ("Teta".to_string(), (0.0, 10.0)),
        ("q".to_string(), (-1.0e20, 1.0e20)),
        ("C0".to_string(), (0.0, 1.5)),
        ("J0".to_string(), (-1.0e2, 1.0e2)),
        ("C1".to_string(), (0.0, 1.5)),
        ("J1".to_string(), (-1.0e2, 1.0e2)),
    ]);
    let relative_tolerances = names
        .iter()
        .map(|name| ((*name).to_string(), 1.0e-5))
        .collect();
    let backend = GeneratedBackendConfig::sparse_defaults()
        .with_matrix_backend_override(MatrixBackend::SparseCol)
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
        .with_lambdify_execution_policy(
            crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Sequential,
        );
    let options = DampedSolverOptions::sparse_damped()
        .with_generated_backend_config(backend)
        .with_abs_tolerance(1.0e-6)
        .with_rel_tolerance(relative_tolerances)
        .with_max_iterations(100)
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(6),
            max_damp_iter: Some(6),
            damp_factor: Some(0.5),
            adaptive: None,
        }))
        .with_bounds(bounds)
        .with_loglevel(Some("error".to_string()));
    let boundary_conditions = HashMap::from([
        ("Teta".to_string(), vec![(0, 2.0 / 3.0)]),
        ("q".to_string(), vec![(1, 1.0e-10)]),
        ("C0".to_string(), vec![(0, 1.0)]),
        ("J0".to_string(), vec![(1, 1.0e-7)]),
        ("C1".to_string(), vec![(0, 1.0e-3)]),
        ("J1".to_string(), vec![(1, 1.0e-7)]),
    ]);
    let mut solver = NRBVP::new_with_options(
        equations,
        DMatrix::from_element(names.len(), 1_000, 0.99),
        names.iter().map(|name| (*name).to_string()).collect(),
        "x".to_string(),
        boundary_conditions,
        0.0,
        1.0,
        1_000,
        options,
    );
    solver.set_bvp_telemetry_mode(BvpTelemetryMode::Detailed);
    solver.set_bvp_logging_config(
        BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(4_096),
    );

    let outcome = solver.try_solve();
    let snapshot = solver.get_statistics().telemetry;
    let outcome_summary = outcome.as_ref().map(|result| {
        result
            .as_ref()
            .map(|matrix| (matrix.nrows(), matrix.ncols()))
    });
    println!(
        "[parsed combustion parity] outcome_shape={outcome_summary:?}; status={:?}; counters={:?}; terminal_event={:?}",
        snapshot.last_solve_status,
        snapshot.counters,
        snapshot.log_events.last(),
    );
    assert!(
        outcome.is_ok(),
        "parsed combustion RHS must solve: {outcome:?}"
    );
    assert!(
        solver.get_result().is_some(),
        "successful solve must publish result"
    );
    assert_eq!(
        snapshot.last_solve_status,
        crate::numerical::BVP_Damp::telemetry::BvpSolveStatus::Converged
    );
}

fn telemetry_counts(solver: &BvpSolver) -> (u64, u64, u64, u64, u64) {
    match solver.telemetry_snapshot() {
        BvpSolverTelemetrySnapshot::CorrectionNewton(Some(snapshot)) => (
            snapshot.accepted_steps as u64,
            snapshot.residual_evaluations as u64,
            snapshot.jacobian_evaluations as u64,
            snapshot.linear_solves as u64,
            snapshot.damping_rejections as u64,
        ),
        BvpSolverTelemetrySnapshot::CorrectionNewton(None) => (0, 0, 0, 0, 0),
        BvpSolverTelemetrySnapshot::Damped(snapshot)
        | BvpSolverTelemetrySnapshot::Frozen(snapshot) => (
            snapshot.counters.iterations,
            snapshot.counters.residual_calls,
            snapshot.counters.jacobian_recalculations,
            snapshot.counters.linear_solves,
            snapshot.counters.damping_rejections,
        ),
        BvpSolverTelemetrySnapshot::Sci(snapshot) => (
            snapshot.newton_iterations,
            snapshot.rhs_calls,
            snapshot.newton_jacobian_refreshes,
            snapshot.linear_solves,
            snapshot.newton_rejected_steps,
        ),
    }
}

fn report_damped_attempt(label: &str, solver: &mut BvpSolver) -> bool {
    let outcome = solver.solve();
    let converged = outcome.is_ok();
    let outcome_summary = match &outcome {
        Ok(_) => "Converged".to_string(),
        Err(error) => format!("{error:?}"),
    };
    println!("\n[{label}] solve outcome: {outcome_summary}");
    let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
        panic!("{label} should expose Damped telemetry");
    };
    let counters = &snapshot.counters;
    println!(
        "[{label}] status={:?}; iterations={} residual_calls={} jacobian_requests={} jacobian_refreshes={} linear_solves={} damping_trials={} damping_rejections={} factorizations={} cache_hits={} invalidations={}",
        snapshot.last_solve_status,
        counters.iterations,
        counters.residual_calls,
        counters.jacobian_requests,
        counters.jacobian_recalculations,
        counters.linear_solves,
        counters.damping_trials,
        counters.damping_rejections,
        counters.factorizations,
        counters.factorization_cache_hits,
        counters.factorization_invalidations,
    );
    println!(
        "[{label}] scopes={:?}; timings_ms={{total:{:.3}, residual:{:.3}, jacobian:{:.3}, linear:{:.3}, factorization:{:.3}, rhs_solve:{:.3}}}",
        snapshot.scopes,
        snapshot.timings.total.as_secs_f64() * 1e3,
        snapshot.timings.residual.as_secs_f64() * 1e3,
        snapshot.timings.jacobian.as_secs_f64() * 1e3,
        snapshot.timings.linear_system.as_secs_f64() * 1e3,
        snapshot.timings.factorization.as_secs_f64() * 1e3,
        snapshot.timings.rhs_solve.as_secs_f64() * 1e3,
    );
    println!(
        "[{label}] decision trace: {} events, {} dropped",
        snapshot.log_events.len(),
        snapshot.log_events_dropped
    );
    for event in &snapshot.log_events {
        if !matches!(
            event.kind,
            crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::JacobianRefresh
                | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingTrial
                | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingRejected
                | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::BoundLimitedStep
                | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::NonFiniteValue
                | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::NearSingularSystem
                | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::Termination
        ) {
            continue;
        }
        println!(
            "[{label}] seq={} it={} trial={} {:?} a={:.6e} b={:.6e}",
            event.sequence,
            event.iteration,
            event.damping_trial,
            event.kind,
            event.value_a,
            event.value_b,
        );
    }
    converged
}

#[test]
fn damped_converged_control_vs_task_failure_compares_detailed_telemetry() {
    let mut classic_control = damped_route(
        11,
        "Sparse",
        [(-10.0, 10.0), (-7.0, 7.0)],
        SolverParams::default(),
    );
    let mut wide_bounds_same_mesh =
        damped_route(11, "Sparse", [(-1.0e6, 1.0e6); 2], SolverParams::default());
    let task_strategy = SolverParams {
        max_jac: None,
        max_damp_iter: None,
        damp_factor: None,
        adaptive: None,
    };
    let mut task_configuration =
        damped_route(TASK_NODES, "Sparse", [(-1.0e6, 1.0e6); 2], task_strategy);

    let classic_converged =
        report_damped_attempt("classic test_BVP_Damp1 control", &mut classic_control);
    let wide_bounds_converged =
        report_damped_attempt("same mesh, task-wide bounds", &mut wide_bounds_same_mesh);
    let task_converged =
        report_damped_attempt("task-parser-equivalent failure", &mut task_configuration);
    assert!(classic_converged, "classic Damped control should converge");
    assert!(
        !wide_bounds_converged,
        "the same-mesh wide-bounds diagnostic should reproduce the sensitivity"
    );
    assert!(
        !task_converged,
        "task-equivalent Damped route is the reproducer"
    );
}

#[derive(Debug, tabled::Tabled)]
struct BoundsSweepRow {
    intervals: usize,
    initial_guess: &'static str,
    bounds: &'static str,
    outcome: String,
    iterations: u64,
    residual_calls: u64,
    jacobian_refreshes: u64,
    damping_trials: u64,
    damping_rejections: u64,
    bound_limited_steps: usize,
    terminal_max_abs_state: String,
    terminal_residual_norm: String,
}

fn run_bounds_sweep_case(
    intervals: usize,
    initial_guess_name: &'static str,
    initial_guess: f64,
    bounds_name: &'static str,
    bounds: Option<[(f64, f64); 2]>,
) -> BoundsSweepRow {
    let mut solver = damped_route_with_optional_bounds(intervals + 1, bounds, initial_guess);
    let outcome = solver.solve();
    let outcome_text = match &outcome {
        Ok(_) => "Converged".to_string(),
        Err(error) => format!("{error:?}"),
    };
    let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
        panic!("bounds sweep must use Damped telemetry");
    };
    let bound_limited_steps = snapshot
        .log_events
        .iter()
        .filter(|event| {
            event.kind == crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::BoundLimitedStep
        })
        .count();
    let (terminal_max_abs_state, terminal_residual_norm) =
        if let crate::numerical::BVP_api::BvpNativeSolver::Damped(native) = solver.native_mut() {
            let state = native.y.to_DVectorType();
            let max_abs = state
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
            let residual_norm = native.try_calc_residual(native.y.clone_box()).ok();
            (
                format!("{max_abs:.3e}"),
                residual_norm.map_or_else(|| "unavailable".into(), |value| format!("{value:.3e}")),
            )
        } else {
            unreachable!("bounds sweep route must be Damped")
        };

    println!(
        "\n[bounds sweep {intervals} intervals / {bounds_name}] status={:?}, events={} dropped={}, terminal max|state|={}, residual probe={}",
        snapshot.last_solve_status,
        snapshot.log_events.len(),
        snapshot.log_events_dropped,
        terminal_max_abs_state,
        terminal_residual_norm,
    );
    for event in snapshot.log_events.iter().filter(|event| {
        event.iteration <= 5
            && matches!(
                event.kind,
                crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::BoundLimitedStep
                    | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingRejected
                    | crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::Termination
            )
    }) {
        println!(
            "  event it={} trial={} {:?} alpha_or_step={:.4e} correction_norm={:.4e}",
            event.iteration, event.damping_trial, event.kind, event.value_a, event.value_b
        );
    }

    BoundsSweepRow {
        intervals,
        initial_guess: initial_guess_name,
        bounds: bounds_name,
        outcome: outcome_text,
        iterations: snapshot.counters.iterations,
        residual_calls: snapshot.counters.residual_calls,
        jacobian_refreshes: snapshot.counters.jacobian_recalculations,
        damping_trials: snapshot.counters.damping_trials,
        damping_rejections: snapshot.counters.damping_rejections,
        bound_limited_steps,
        terminal_max_abs_state,
        terminal_residual_norm,
    }
}

#[test]
fn damped_reference_problem_bounds_and_mesh_sweep_reports_detailed_telemetry() {
    let bounds_cases: [(&str, Option<[(f64, f64); 2]>); 4] = [
        ("classic z+/-10,y+/-7", Some([(-10.0, 10.0), (-7.0, 7.0)])),
        ("5x classic", Some([(-50.0, 50.0), (-35.0, 35.0)])),
        ("100x classic", Some([(-1000.0, 1000.0), (-700.0, 700.0)])),
        ("task-wide +/-1e6", Some([(-1.0e6, 1.0e6); 2])),
    ];
    let mut rows = Vec::new();
    for intervals in [10, 20] {
        for (guess_name, guess_value) in [("zero", 0.0), ("one", 1.0)] {
            for (name, bounds) in bounds_cases {
                rows.push(run_bounds_sweep_case(
                    intervals,
                    guess_name,
                    guess_value,
                    name,
                    bounds,
                ));
            }
        }
    }
    println!("\n{}", tabled::Table::new(&rows));

    let sweep_row = |intervals, initial_guess, bounds| {
        rows.iter()
            .find(|row| {
                row.intervals == intervals
                    && row.initial_guess == initial_guess
                    && row.bounds == bounds
            })
            .unwrap_or_else(|| {
                panic!(
                    "missing bounds-sweep row: intervals={intervals}, guess={initial_guess}, bounds={bounds}"
                )
            })
    };
    for initial_guess in ["zero", "one"] {
        let row = sweep_row(10, initial_guess, "classic z+/-10,y+/-7");
        assert_eq!(row.outcome, "Converged", "10-interval control: {row:?}");
        assert!(row.iterations > 0 && row.residual_calls > 0);
        assert!(row.jacobian_refreshes > 0 && row.damping_trials > 0);
        let residual: f64 = row.terminal_residual_norm.parse().unwrap();
        assert!(residual < 2e-3, "terminal residual={residual:e}: {row:?}");
    }
    for initial_guess in ["zero", "one"] {
        let row = sweep_row(20, initial_guess, "5x classic");
        assert_eq!(
            row.outcome, "Converged",
            "20-interval relaxed-bounds control: {row:?}"
        );
        let residual: f64 = row.terminal_residual_norm.parse().unwrap();
        assert!(residual < 1e-3, "terminal residual={residual:e}: {row:?}");
    }

    let classic_params = SolverParams {
        max_jac: Some(5),
        max_damp_iter: Some(5),
        damp_factor: None,
        adaptive: None,
    };
    let classic_options = damped_options("Sparse", [(-10.0, 10.0), (-7.0, 7.0)], classic_params);
    let mut classic_fixed_mesh = BvpSolver::new(
        programmatic_problem(11),
        BvpSolverConfig::Damped {
            options: classic_options,
            initial_guess: DMatrix::from_element(2, 10, 1.0),
        },
    )
    .expect("classic fixed-mesh control should prepare");
    let classic_solved = classic_fixed_mesh.solve().is_ok();
    let BvpSolverTelemetrySnapshot::Damped(classic_telemetry) =
        classic_fixed_mesh.telemetry_snapshot()
    else {
        panic!("classic fixed-mesh control should expose Damped telemetry");
    };
    println!(
        "\n[classic fixed-mesh control] outcome={:?}, iterations={}, residual_calls={}, jacobian_refreshes={}, damping_trials={}, rejections={}, events={}, dropped={}",
        classic_telemetry.last_solve_status,
        classic_telemetry.counters.iterations,
        classic_telemetry.counters.residual_calls,
        classic_telemetry.counters.jacobian_recalculations,
        classic_telemetry.counters.damping_trials,
        classic_telemetry.counters.damping_rejections,
        classic_telemetry.log_events.len(),
        classic_telemetry.log_events_dropped,
    );
    assert!(
        classic_solved,
        "classic fixed-mesh Damped control should converge"
    );
    assert_eq!(
        classic_telemetry.last_solve_status,
        crate::numerical::BVP_Damp::telemetry::BvpSolveStatus::Converged
    );
}

fn run_route(route: &'static str, requested_nodes: usize, mut solver: BvpSolver) -> RouteRow {
    let started = Instant::now();
    let outcome = match solver.solve() {
        Ok(_) => "Converged".to_string(),
        Err(error) => {
            let message = format!("{error:?}");
            if message.chars().count() > 96 {
                format!("{}...", message.chars().take(96).collect::<String>())
            } else {
                message
            }
        }
    };
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let (native_status, last_native_event) = match solver.telemetry_snapshot() {
        BvpSolverTelemetrySnapshot::Damped(snapshot)
        | BvpSolverTelemetrySnapshot::Frozen(snapshot) => {
            let last_event = snapshot.log_events.last().map_or_else(
                || "-".to_string(),
                |event| {
                    format!(
                        "{:?}@it{}({:.2e},{:.2e})",
                        event.kind, event.iteration, event.value_a, event.value_b
                    )
                },
            );
            (format!("{:?}", snapshot.last_solve_status), last_event)
        }
        _ => ("route-native".to_string(), "-".to_string()),
    };
    let (iterations, residuals, jacobians, linear_solves, rejections) = telemetry_counts(&solver);

    let mut actual_nodes = 0;
    let mut boundary_error = None::<f64>;
    let mut max_abs_state = None::<f64>;
    let mut solution_values = Vec::new();
    if let Some(solution) = solver.solution_view() {
        actual_nodes = solution.mesh().len();
        let left_z_error = (solution.value(0, 0).unwrap() - 1.0).abs();
        let right_y_error = (solution.value(actual_nodes - 1, 1).unwrap() - 1.0).abs();
        boundary_error = Some(left_z_error.max(right_y_error));
        let mut maximum = 0.0_f64;
        for node in 0..actual_nodes {
            for state in 0..solution.state_count() {
                let value = solution.value(node, state).unwrap();
                assert!(value.is_finite(), "{route} published a non-finite value");
                maximum = maximum.max(value.abs());
                solution_values.push(value);
            }
        }
        assert!(
            boundary_error.unwrap() <= 1e-4,
            "{route} published a profile that violates the imposed boundary values"
        );
        max_abs_state = Some(maximum);
    } else {
        assert!(
            outcome != "Converged",
            "{route} reported convergence without publishing a solution"
        );
    }

    RouteRow {
        route,
        requested_nodes,
        outcome,
        native_status,
        last_native_event,
        actual_nodes,
        iterations,
        residuals,
        jacobians,
        linear_solves,
        rejections,
        boundary_error: boundary_error.map_or_else(|| "-".into(), |value| format!("{value:.3e}")),
        max_abs_state: max_abs_state.map_or_else(|| "-".into(), |value| format!("{value:.3e}")),
        wall_ms: format!("{wall_ms:.3}"),
        solution_values,
    }
}

#[test]
fn task_reference_problem_programmatic_cross_solver_diagnostic() {
    let task_problem = programmatic_problem(TASK_NODES);
    let intervals = TASK_NODES - 1;
    let task_seed = DMatrix::zeros(2, intervals);
    let task_strategy = SolverParams {
        max_jac: None,
        max_damp_iter: None,
        damp_factor: None,
        adaptive: None,
    };

    let small_mesh_nodes = 11;
    let damped = damped_route(
        TASK_NODES,
        "Sparse",
        [(-1.0e6, 1.0e6); 2],
        task_strategy.clone(),
    );
    let damped_sparse_narrow = damped_route(
        small_mesh_nodes,
        "Sparse",
        [(-10.0, 10.0); 2],
        task_strategy.clone(),
    );
    let damped_dense_wide = damped_route(
        small_mesh_nodes,
        "Dense",
        [(-1.0e6, 1.0e6); 2],
        task_strategy.clone(),
    );
    let damped_dense_narrow = damped_route(
        small_mesh_nodes,
        "Dense",
        [(-10.0, 10.0); 2],
        task_strategy.clone(),
    );
    let damped_small_mesh = damped_route(
        small_mesh_nodes,
        "Sparse",
        [(-1.0e6, 1.0e6); 2],
        task_strategy.clone(),
    );
    let damped_sparse_defaults = damped_route(
        small_mesh_nodes,
        "Sparse",
        [(-10.0, 10.0), (-7.0, 7.0)],
        SolverParams::default(),
    );
    let damped_dense_defaults = damped_route(
        small_mesh_nodes,
        "Dense",
        [(-10.0, 10.0), (-7.0, 7.0)],
        SolverParams::default(),
    );
    let damped_sparse_wide_defaults = damped_route(
        small_mesh_nodes,
        "Sparse",
        [(-1.0e6, 1.0e6); 2],
        SolverParams::default(),
    );
    let damped_trapezoid_10 = damped_route_with_scheme(
        small_mesh_nodes,
        "Sparse",
        [(-10.0, 10.0), (-7.0, 7.0)],
        SolverParams::default(),
        "trapezoid",
    );
    let damped_trapezoid_20 = damped_route_with_scheme(
        TASK_NODES,
        "Sparse",
        [(-10.0, 10.0), (-7.0, 7.0)],
        SolverParams::default(),
        "trapezoid",
    );
    let damped_sparse_classic_20 = damped_route(
        TASK_NODES,
        "Sparse",
        [(-10.0, 10.0), (-7.0, 7.0)],
        SolverParams::default(),
    );
    let correction_options = CorrectionNewtonOptions {
        absolute_tolerance: 1e-5,
        default_relative_tolerance: 1e-5,
        max_iterations: 20,
        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
        ..CorrectionNewtonOptions::default()
    };
    let correction = BvpSolver::new(
        task_problem.clone(),
        BvpSolverConfig::CorrectionNewton {
            options: correction_options,
            initial_guess: task_seed.clone(),
        },
    )
    .expect("programmatic CorrectionNewton route should prepare");

    let sci_options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_matrix_layout(BvpSciMatrixLayout::BandedAuto)
        .with_tolerance(1e-5)
        .with_limits(10_000, 20, 10)
        .with_telemetry(BvpSciTelemetryMode::Detailed);
    let sci = BvpSolver::new(
        task_problem.clone(),
        BvpSolverConfig::Sci {
            options: sci_options,
            sparse_factorization: BvpSciSparseFactorization::default(),
            aot_config: None,
            initial_guess: DMatrix::zeros(2, TASK_NODES),
        },
    )
    .expect("programmatic BVP_sci route should prepare");

    let sci_reference_nodes = 101;
    let sci_reference_options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_matrix_layout(BvpSciMatrixLayout::BandedAuto)
        .with_tolerance(1e-6)
        .with_limits(10_000, 20, 10)
        .with_telemetry(BvpSciTelemetryMode::Detailed);
    let sci_reference = BvpSolver::new(
        programmatic_problem(sci_reference_nodes),
        BvpSolverConfig::Sci {
            options: sci_reference_options,
            sparse_factorization: BvpSciSparseFactorization::default(),
            aot_config: None,
            initial_guess: DMatrix::from_fn(2, sci_reference_nodes, |state, _| {
                if state == 0 { 1.0 } else { 0.0 }
            }),
        },
    )
    .expect("programmatic SciPy-reference BVP_sci route should prepare");
    let sci_reference_sparse = BvpSolver::new(
        programmatic_problem(sci_reference_nodes),
        BvpSolverConfig::Sci {
            options: BvpSciOptions::default()
                .with_execution(BvpSciExecution::Lambdify)
                .with_assembly(BvpSciAssembly::ExprLegacy)
                .with_matrix_layout(BvpSciMatrixLayout::Sparse)
                .with_tolerance(1e-5)
                .with_limits(10_000, 20, 10)
                .with_telemetry(BvpSciTelemetryMode::Detailed),
            sparse_factorization: BvpSciSparseFactorization::default(),
            aot_config: None,
            initial_guess: DMatrix::from_fn(2, sci_reference_nodes, |state, _| {
                if state == 0 { 1.0 } else { 0.0 }
            }),
        },
    )
    .expect("programmatic Sparse SciPy-reference BVP_sci route should prepare");

    let frozen_problem = programmatic_problem(FROZEN_NODES);
    let frozen_backend = GeneratedBackendConfig::banded_lambdify_defaults();
    let frozen_options = FrozenSolverOptions::banded_frozen()
        .with_generated_backend_config(frozen_backend)
        .with_tolerance(1e-5)
        .with_max_iterations(20)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_mode(BvpLoggingMode::Detailed);
    let frozen = BvpSolver::new(
        frozen_problem,
        BvpSolverConfig::Frozen {
            options: frozen_options,
            initial_guess: DMatrix::zeros(2, FROZEN_NODES - 1),
        },
    )
    .expect("large-mesh programmatic Frozen route should prepare");

    let rows = vec![
        run_route(
            "Damped Sparse wide / 10",
            small_mesh_nodes,
            damped_small_mesh,
        ),
        run_route(
            "Damped Sparse narrow / 10",
            small_mesh_nodes,
            damped_sparse_narrow,
        ),
        run_route(
            "Damped Dense wide / 10",
            small_mesh_nodes,
            damped_dense_wide,
        ),
        run_route(
            "Damped Dense narrow / 10",
            small_mesh_nodes,
            damped_dense_narrow,
        ),
        run_route(
            "Damped Sparse classic settings / 10",
            small_mesh_nodes,
            damped_sparse_defaults,
        ),
        run_route(
            "Damped Dense classic settings / 10",
            small_mesh_nodes,
            damped_dense_defaults,
        ),
        run_route(
            "Damped Sparse wide bounds + default strategy / 10",
            small_mesh_nodes,
            damped_sparse_wide_defaults,
        ),
        run_route(
            "Damped Sparse Trapezoid / 10",
            small_mesh_nodes,
            damped_trapezoid_10,
        ),
        run_route("Damped / 20 intervals", TASK_NODES, damped),
        run_route(
            "Damped Sparse classic settings / 20",
            TASK_NODES,
            damped_sparse_classic_20,
        ),
        run_route(
            "Damped Sparse Trapezoid / 20",
            TASK_NODES,
            damped_trapezoid_20,
        ),
        run_route("CorrectionNewton / 20 intervals", TASK_NODES, correction),
        run_route("BVP_sci / 21 seed nodes", TASK_NODES, sci),
        run_route("Frozen Banded / 1000 intervals", FROZEN_NODES, frozen),
        run_route(
            "BVP_sci / 101 nodes, z=1/y=0",
            sci_reference_nodes,
            sci_reference,
        ),
        run_route(
            "BVP_sci Sparse / 101 nodes",
            sci_reference_nodes,
            sci_reference_sparse,
        ),
    ];
    println!(
        "Programmatic BVP task-equivalent solver A/B (RHS: z'=y-z, y'=-z^3; BC: z(0)=1, y(1)=1; zero seed)\n{}",
        tabled::Table::new(&rows)
    );
    assert_eq!(rows.len(), 16);
    let mut route_names = rows.iter().map(|row| row.route).collect::<Vec<_>>();
    route_names.sort_unstable();
    route_names.dedup();
    assert_eq!(route_names.len(), rows.len(), "route labels must be unique");

    let row = |route: &str| {
        rows.iter()
            .find(|row| row.route == route)
            .unwrap_or_else(|| panic!("missing diagnostic route {route}"))
    };
    let assert_failed_without_publication = |route: &str| {
        let failed = row(route);
        assert_ne!(
            failed.outcome, "Converged",
            "{route} unexpectedly converged"
        );
        assert_eq!(failed.actual_nodes, 0, "{route} published a failed profile");
        assert!(
            failed.solution_values.is_empty(),
            "{route} retained values for an unpublished profile"
        );
        assert!(failed.iterations > 0, "{route} did no Newton work");
        assert!(failed.residuals > 0, "{route} did no residual work");
        assert!(failed.jacobians > 0, "{route} did no Jacobian rebuilds");
        assert!(failed.linear_solves > 0, "{route} did no linear solves");
    };

    let sparse_classic = rows
        .iter()
        .find(|row| row.route == "Damped Sparse classic settings / 10")
        .unwrap();
    let dense_classic = rows
        .iter()
        .find(|row| row.route == "Damped Dense classic settings / 10")
        .unwrap();
    assert_eq!(sparse_classic.outcome, "Converged");
    assert_eq!(dense_classic.outcome, "Converged");
    for classic in [sparse_classic, dense_classic] {
        assert_eq!(classic.native_status, "Converged");
        assert_eq!(classic.actual_nodes, 11);
        assert_eq!(classic.iterations, 8);
        assert_eq!(classic.residuals, 23);
        assert_eq!(
            classic.jacobians, 3,
            "count Jacobian rebuilds, not callbacks"
        );
        assert_eq!(classic.linear_solves, 23);
        assert_eq!(classic.rejections, 9);
        let boundary_error: f64 = classic.boundary_error.parse().unwrap();
        assert!(boundary_error <= 1e-12, "boundary error={boundary_error:e}");
    }
    assert_eq!(sparse_classic.iterations, dense_classic.iterations);
    assert_eq!(sparse_classic.residuals, dense_classic.residuals);
    assert_eq!(sparse_classic.jacobians, dense_classic.jacobians);
    let profile_difference = sparse_classic
        .solution_values
        .iter()
        .zip(&dense_classic.solution_values)
        .map(|(sparse, dense)| (sparse - dense).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        profile_difference <= 1e-12,
        "Dense/Sparse profile difference={profile_difference:e}"
    );

    for route in [
        "Damped Sparse wide / 10",
        "Damped Sparse narrow / 10",
        "Damped Dense wide / 10",
        "Damped Dense narrow / 10",
        "Damped Sparse wide bounds + default strategy / 10",
        "Damped Sparse Trapezoid / 10",
        "Damped / 20 intervals",
        "Damped Sparse classic settings / 20",
        "Damped Sparse Trapezoid / 20",
        "CorrectionNewton / 20 intervals",
        "BVP_sci / 21 seed nodes",
        "Frozen Banded / 1000 intervals",
        "BVP_sci / 101 nodes, z=1/y=0",
        "BVP_sci Sparse / 101 nodes",
    ] {
        assert_failed_without_publication(route);
    }

    assert!(
        row("BVP_sci / 21 seed nodes")
            .outcome
            .starts_with("Sci(MaxNodesExceeded"),
        "the task-equivalent BVP_sci seed route should report its node-limit failure"
    );
    assert!(
        row("CorrectionNewton / 20 intervals")
            .outcome
            .starts_with("CorrectionNewton(NonConvergence"),
        "CorrectionNewton failure should remain typed as non-convergence"
    );
}
