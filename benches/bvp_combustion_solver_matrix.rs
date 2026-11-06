//! Tabled release comparison of the established combustion BVP fixtures.
//!
//! CorrectionNewton uses its fixed seed mesh, BVP_sci uses its native adaptive
//! collocation controller, and BVP_Damp is measured independently with each of
//! its four mesh-refinement strategies. This is a workload matrix, not a
//! perf-gate: use repeated perf-gate runs for baseline decisions.

use std::collections::HashMap;
use std::time::Instant;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonMatrixLayout, CorrectionNewtonOptions, CorrectionNewtonProblem,
    CorrectionNewtonSymbolicFrontend, CorrectionNewtonTelemetryMode,
};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedSolverOptions, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use RustedSciThe::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use RustedSciThe::numerical::BVP_Damp::telemetry::{BvpLoggingMode, BvpTelemetryMode};
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSciSparseFactorization, BvpSolver,
    BvpSolverConfig, BvpSolverTelemetrySnapshot,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciOptions, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use tabled::Tabled;

const START_INTERVALS: usize = 10;
const REPEATS: usize = 3;

struct Fixture {
    name: &'static str,
    problem: CorrectionNewtonProblem,
}

#[derive(Tabled)]
struct Row {
    fixture: &'static str,
    solver: &'static str,
    refinement: &'static str,
    repeat: usize,
    seed_nodes: usize,
    result_nodes: String,
    construct_ms: String,
    solve_ms: String,
    telemetry: String,
    profile_min: String,
    profile_max_abs: String,
    status: String,
}

fn comb01_problem(steps: usize) -> CorrectionNewtonProblem {
    let (pe, da, heat, beta, gamma) = (5.0, 2.0, 2.0, 5.0, 0.5);
    let approximate = |x: f64| {
        let fuel = (-da * x).exp();
        let flux = -da * fuel;
        [heat * (1.0 - fuel), -heat * flux, fuel, flux]
    };
    let guess = DMatrix::from_fn(4, steps, |state, column| {
        let node = if state == 0 || state == 2 {
            column + 1
        } else {
            column
        };
        approximate(node as f64 / steps as f64)[state]
    });
    let rate = format!("{da} * Y * exp({beta} * (1.0 - 1.0 / (1.0 + {gamma} * theta)))");
    CorrectionNewtonProblem {
        equations: vec![
            Expr::parse_expression("q"),
            Expr::parse_expression(&format!("{pe} * q - {heat} * ({rate})")),
            Expr::parse_expression("J"),
            Expr::parse_expression(&format!("{pe} * J + ({rate})")),
        ],
        initial_guess: guess,
        state_names: ["theta", "q", "Y", "J"].map(str::to_owned).to_vec(),
        independent_variable: "x".into(),
        boundary_conditions: HashMap::from([
            ("theta".into(), vec![(0, 0.0)]),
            ("Y".into(), vec![(0, 1.0)]),
            ("q".into(), vec![(1, 0.0)]),
            ("J".into(), vec![(1, 0.0)]),
        ]),
        interval: (0.0, 1.0),
        steps,
    }
}

fn comb02_problem(steps: usize) -> CorrectionNewtonProblem {
    let (pe, da1, da2) = (2.0, 3.0, 12.0);
    let (q1, q2, beta1, beta2, gamma) = (1.0, 2.0, 12.0, 1.0, 0.5);
    let approximate = |x: f64| {
        let a = (-da1 * x).exp();
        let b = da1 / (da2 - da1) * ((-da1 * x).exp() - (-da2 * x).exp());
        let c = 1.0 - a - b;
        let ja = -da1 * a;
        let jb = da1 / (da2 - da1) * (-da1 * (-da1 * x).exp() + da2 * (-da2 * x).exp());
        let jc = -ja - jb;
        let theta = q1 * (1.0 - a) + q2 * c;
        let flux = -(q1 + q2) * ja - q2 * jb;
        [theta, flux, a, ja, b, jb, c, jc]
    };
    let guess = DMatrix::from_fn(8, steps, |state, column| {
        let node = if matches!(state, 0 | 2 | 4 | 6) {
            column + 1
        } else {
            column
        };
        approximate(node as f64 / steps as f64)[state]
    });
    let thermal = format!("(1.0 - 1.0 / (1.0 + {gamma} * theta))");
    let r1 = format!("{da1} * A * exp({beta1} * {thermal})");
    let r2 = format!("{da2} * B * exp({beta2} * {thermal})");
    CorrectionNewtonProblem {
        equations: vec![
            Expr::parse_expression("q"),
            Expr::parse_expression(&format!("{pe} * q - {q1} * ({r1}) - {q2} * ({r2})")),
            Expr::parse_expression("JA"),
            Expr::parse_expression(&format!("{pe} * JA + ({r1})")),
            Expr::parse_expression("JB"),
            Expr::parse_expression(&format!("{pe} * JB - ({r1}) + ({r2})")),
            Expr::parse_expression("JC"),
            Expr::parse_expression(&format!("{pe} * JC - ({r2})")),
        ],
        initial_guess: guess,
        state_names: ["theta", "q", "A", "JA", "B", "JB", "C", "JC"]
            .map(str::to_owned)
            .to_vec(),
        independent_variable: "x".into(),
        boundary_conditions: HashMap::from([
            ("theta".into(), vec![(0, 0.0)]),
            ("A".into(), vec![(0, 1.0)]),
            ("B".into(), vec![(0, 0.0)]),
            ("C".into(), vec![(0, 0.0)]),
            ("q".into(), vec![(1, 0.0)]),
            ("JA".into(), vec![(1, 0.0)]),
            ("JB".into(), vec![(1, 0.0)]),
            ("JC".into(), vec![(1, 0.0)]),
        ]),
        interval: (0.0, 1.0),
        steps,
    }
}

fn comb02_perturbed_problem(steps: usize) -> CorrectionNewtonProblem {
    let reference =
        include_str!("../src/numerical/BVP_CorrectionNewton/reference/comb02_scipy.csv")
            .lines()
            .skip(1)
            .map(|line| {
                let values = line
                    .split(',')
                    .map(|field| field.parse::<f64>().expect("numeric COMB-02 reference"))
                    .collect::<Vec<_>>();
                (values[0], values[1..].to_vec())
            })
            .collect::<Vec<_>>();
    let sample = |x: f64| {
        let scaled = x * (reference.len() - 1) as f64;
        let left = scaled.floor() as usize;
        let right = (left + 1).min(reference.len() - 1);
        let fraction = scaled - left as f64;
        reference[left]
            .1
            .iter()
            .zip(&reference[right].1)
            .map(|(a, b)| a + fraction * (b - a))
            .collect::<Vec<_>>()
    };
    let mut profile = (0..=steps)
        .map(|node| {
            let x = node as f64 / steps as f64;
            let mut state = sample(x);
            let offset = (std::f64::consts::PI * x).sin();
            state[2] += 0.005 * offset;
            state[4] += 0.015 * offset;
            state[6] = 1.0 - state[2] - state[4];
            state[0] = 3.0 - 3.0 * state[2] - 2.0 * state[4];
            state
        })
        .collect::<Vec<_>>();
    let dx = 1.0 / steps as f64;
    for node in 0..=steps {
        let (left, right) = match node {
            0 => (0, 1),
            n if n == steps => (steps - 1, steps),
            n => (n - 1, n + 1),
        };
        let divisor = if node == 0 || node == steps {
            dx
        } else {
            2.0 * dx
        };
        let ja = (profile[right][2] - profile[left][2]) / divisor;
        let jb = (profile[right][4] - profile[left][4]) / divisor;
        let jc = (profile[right][6] - profile[left][6]) / divisor;
        profile[node][3] = ja;
        profile[node][5] = jb;
        profile[node][7] = jc;
        profile[node][1] = -3.0 * ja - 2.0 * jb;
    }
    let mut problem = comb02_problem(steps);
    for state in 0..8 {
        for column in 0..steps {
            let node = if matches!(state, 0 | 2 | 4 | 6) {
                column + 1
            } else {
                column
            };
            problem.initial_guess[(state, column)] = profile[node][state];
        }
    }
    problem
}

fn fixtures() -> Vec<Fixture> {
    let mut kithe =
        RustedSciThe::numerical::BVP_Damp::correction_newton_kithe_combustion_perf_fixture();
    let old_steps = kithe.steps;
    let old_guess = kithe.initial_guess.clone();
    kithe.steps = START_INTERVALS;
    kithe.initial_guess =
        DMatrix::from_fn(kithe.state_names.len(), START_INTERVALS, |state, col| {
            old_guess[(state, col * old_steps / START_INTERVALS)]
        });
    vec![
        Fixture {
            name: "KiThe-26-state",
            problem: kithe,
        },
        Fixture {
            name: "COMB-01",
            problem: comb01_problem(START_INTERVALS),
        },
        Fixture {
            name: "COMB-02",
            problem: comb02_perturbed_problem(START_INTERVALS),
        },
    ]
}

fn as_bvp_problem(fixture: &CorrectionNewtonProblem) -> BvpProblem {
    let conditions = fixture
        .boundary_conditions
        .iter()
        .flat_map(|(name, entries)| {
            let state = fixture
                .state_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap();
            entries
                .iter()
                .map(move |&(endpoint, value)| BvpBoundaryCondition {
                    state,
                    side: if endpoint == 0 {
                        BvpBoundarySide::Left
                    } else {
                        BvpBoundarySide::Right
                    },
                    value,
                })
        })
        .collect();
    let mesh = (0..=fixture.steps)
        .map(|node| {
            fixture.interval.0
                + (fixture.interval.1 - fixture.interval.0) * node as f64 / fixture.steps as f64
        })
        .collect();
    BvpProblem::new(
        fixture.equations.clone(),
        fixture.state_names.clone(),
        fixture.independent_variable.clone(),
        mesh,
        conditions,
    )
    .expect("combustion fixture validates through BVP_api")
}

fn full_seed(fixture: &CorrectionNewtonProblem) -> DMatrix<f64> {
    DMatrix::from_fn(
        fixture.state_names.len(),
        fixture.steps + 1,
        |state, node| {
            let conditions = &fixture.boundary_conditions[&fixture.state_names[state]];
            if let Some((_, value)) = conditions.iter().find(|(end, _)| *end == 0) {
                if node == 0 {
                    return *value;
                }
                return fixture.initial_guess[(state, node - 1)];
            }
            if let Some((_, value)) = conditions.iter().find(|(end, _)| *end == 1) {
                if node == fixture.steps {
                    return *value;
                }
            }
            fixture.initial_guess[(state, node)]
        },
    )
}

fn bounds(fixture: &CorrectionNewtonProblem) -> HashMap<String, (f64, f64)> {
    fixture
        .state_names
        .iter()
        .map(|name| {
            let range = match name.as_str() {
                "u0" => (-1.0, 3.0),
                "theta" => (-0.5, 4.0),
                "Y" | "A" | "B" | "C" => (-0.2, 1.2),
                _ if name.starts_with('u')
                    && name[1..].parse::<usize>().is_ok_and(|i| i % 2 == 0) =>
                {
                    (0.0, 1.1)
                }
                _ if name.starts_with('u') => (-1.0e4, 1.0e4),
                _ => (-30.0, 30.0),
            };
            (name.clone(), range)
        })
        .collect()
}

fn correction_options(fixture: &CorrectionNewtonProblem) -> CorrectionNewtonOptions {
    CorrectionNewtonOptions {
        symbolic_frontend: CorrectionNewtonSymbolicFrontend::ExprLegacy,
        matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
        absolute_tolerance: 1e-5,
        default_relative_tolerance: 1e-5,
        bounds: bounds(fixture),
        max_iterations: 100,
        max_jacobian_age: 1,
        max_damping_trials: 12,
        telemetry_mode: CorrectionNewtonTelemetryMode::Timings,
        ..CorrectionNewtonOptions::default()
    }
}

fn damped_options(
    fixture: &CorrectionNewtonProblem,
    strategy: GridRefinementMethod,
) -> DampedSolverOptions {
    let generated = GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    DampedSolverOptions::sparse_damped()
        .with_generated_backend_config(generated)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_mode(BvpLoggingMode::Off)
        .with_scheme_name("trapezoid")
        .with_abs_tolerance(1e-5)
        .with_rel_tolerance(
            fixture
                .state_names
                .iter()
                .map(|name| (name.clone(), 1e-5))
                .collect(),
        )
        .with_max_iterations(150)
        .with_bounds(bounds(fixture))
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(1),
            max_damp_iter: Some(12),
            adaptive: Some(AdaptiveGridConfig {
                version: 1,
                max_refinements: 8,
                grid_method: strategy,
            }),
            ..SolverParams::default()
        }))
}

fn telemetry(snapshot: &BvpSolverTelemetrySnapshot) -> String {
    match snapshot {
        BvpSolverTelemetrySnapshot::CorrectionNewton(Some(value)) => format!(
            "it={} res={} jac={} fact={} linear={} res_ms={:.2} jac_ms={:.2}",
            value.accepted_steps,
            value.residual_evaluations,
            value.jacobian_evaluations,
            value.factorizations,
            value.linear_solves,
            value.residual_callback_time.as_secs_f64() * 1e3,
            value.jacobian_callback_time.as_secs_f64() * 1e3,
        ),
        BvpSolverTelemetrySnapshot::Damped(value) => format!(
            "it={} refine={} res={} jac={} fact={} linear={} native_ms={:.2} symbolic_ms={:.2} residual_ms={:.2} jac_ms={:.2} linear_ms={:.2}",
            value.counters.iterations,
            value.counters.grid_refinements,
            value.counters.residual_calls,
            value.counters.jacobian_recalculations,
            value.counters.factorizations,
            value.counters.linear_solves,
            value.timings.total.as_secs_f64() * 1e3,
            value.timings.symbolic_operations.as_secs_f64() * 1e3,
            value.timings.residual.as_secs_f64() * 1e3,
            value.timings.jacobian.as_secs_f64() * 1e3,
            value.timings.linear_system.as_secs_f64() * 1e3,
        ),
        BvpSolverTelemetrySnapshot::Sci(value) => format!(
            "it={} refine={} rhs={} jac={} fact={} linear={} native_ms={:?} callback_ms={:?} mesh_ms={:?} factor_ms={:?}",
            value.newton_iterations,
            value.mesh_refinements,
            value.rhs_calls,
            value.jacobian_calls,
            value.factorizations,
            value.linear_solves,
            value.full_solve_ms,
            value.callback_ms,
            value.mesh_refinement_ms,
            value.factorization_ms,
        ),
        other => format!("{other:?}"),
    }
}

fn one_run(
    fixture: &Fixture,
    solver_name: &'static str,
    refinement: &'static str,
    config: BvpSolverConfig,
    repeat: usize,
) -> Row {
    let prep = Instant::now();
    let mut solver = match BvpSolver::new(as_bvp_problem(&fixture.problem), config) {
        Ok(solver) => solver,
        Err(error) => {
            return failed_row(
                fixture,
                solver_name,
                refinement,
                repeat,
                prep.elapsed().as_secs_f64() * 1e3,
                format!("construct: {error}"),
            );
        }
    };
    let construct_ms = prep.elapsed().as_secs_f64() * 1e3;
    let solve_started = Instant::now();
    let result = solver.solve();
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
    let stats = telemetry(&solver.telemetry_snapshot());
    let (nodes, minimum, maximum) = solver.solution_view().map_or_else(
        || ("-".into(), "-".into(), "-".into()),
        |view| {
            let mut minimum = f64::INFINITY;
            let mut maximum = 0.0_f64;
            for node in 0..view.mesh().len() {
                for state in 0..view.state_count() {
                    let value = view.value(node, state).unwrap_or(f64::NAN);
                    minimum = minimum.min(value);
                    maximum = maximum.max(value.abs());
                }
            }
            (
                view.mesh().len().to_string(),
                format!("{minimum:.4e}"),
                format!("{maximum:.4e}"),
            )
        },
    );
    Row {
        fixture: fixture.name,
        solver: solver_name,
        refinement,
        repeat,
        seed_nodes: fixture.problem.steps + 1,
        result_nodes: nodes,
        construct_ms: format!("{construct_ms:.3}"),
        solve_ms: format!("{solve_ms:.3}"),
        telemetry: stats,
        profile_min: minimum,
        profile_max_abs: maximum,
        status: result.map_or_else(|error| format!("solve: {error}"), |_| "ok".into()),
    }
}

fn failed_row(
    fixture: &Fixture,
    solver: &'static str,
    refinement: &'static str,
    repeat: usize,
    construct_ms: f64,
    status: String,
) -> Row {
    Row {
        fixture: fixture.name,
        solver,
        refinement,
        repeat,
        seed_nodes: fixture.problem.steps + 1,
        result_nodes: "-".into(),
        construct_ms: format!("{construct_ms:.3}"),
        solve_ms: "-".into(),
        telemetry: "unavailable".into(),
        profile_min: "-".into(),
        profile_max_abs: "-".into(),
        status,
    }
}

fn main() {
    let strategies = [
        ("TwoPoint", GridRefinementMethod::TwoPoint(0.02, 0.02, 2.5)),
        ("Pearson", GridRefinementMethod::Pearson(1e-3, 1.4)),
        (
            "GrcarSmooke",
            GridRefinementMethod::GrcarSmooke(1e-3, 1e-2, 1.4),
        ),
        ("DoublePoints", GridRefinementMethod::DoublePoints),
    ];
    let mut rows = Vec::with_capacity(18 * REPEATS);
    for fixture in fixtures() {
        for repeat in 1..=REPEATS {
            rows.push(one_run(
                &fixture,
                "CorrectionNewton",
                "fixed mesh",
                BvpSolverConfig::CorrectionNewton {
                    options: correction_options(&fixture.problem),
                    initial_guess: fixture.problem.initial_guess.clone(),
                },
                repeat,
            ));

            let mut sci = BvpSciOptions::default();
            sci.execution = BvpSciExecution::Lambdify;
            sci.assembly = Some(BvpSciAssembly::ExprLegacy);
            sci.matrix_layout = BvpSciMatrixLayout::Sparse;
            sci.tolerance = 1e-5;
            sci.boundary_tolerance = Some(1e-5);
            sci.max_nodes = 4096;
            sci.max_newton_iterations = 100;
            sci.max_mesh_refinements = 8;
            sci.telemetry = BvpSciTelemetryMode::Detailed;
            rows.push(one_run(
                &fixture,
                "BVP_sci",
                "native collocation refinement",
                BvpSolverConfig::Sci {
                    options: sci,
                    sparse_factorization: BvpSciSparseFactorization::StandardFaer,
                    aot_config: None,
                    initial_guess: full_seed(&fixture.problem),
                },
                repeat,
            ));

            for (name, method) in &strategies {
                rows.push(one_run(
                    &fixture,
                    "BVP_Damp",
                    name,
                    BvpSolverConfig::Damped {
                        options: damped_options(&fixture.problem, method.clone()),
                        initial_guess: fixture.problem.initial_guess.clone(),
                    },
                    repeat,
                ));
            }
        }
    }
    let report = format!(
        "# Combustion BVP solver and refinement matrix\n\nThree independent samples per route. CorrectionNewton uses its fixed uniform 10-interval mesh; BVP_sci uses its native adaptive collocation controller; BVP_Damp starts from 10 intervals and is run independently with each named refinement strategy. All routes use 1e-5 absolute/relative tolerances, ExprLegacy/Lambdify and Sparse linear algebra. `construct_ms` measures solver construction only. BVP_Damp may lazily prepare generated callbacks during `solve_ms`; its native symbolic/preparation stages are reported separately in telemetry. Failed attempts remain visible and do not abort the matrix.\n\n{}\n",
        tabled::Table::new(&rows)
    );
    match write_test_report(
        "BVP_combustion_solver_matrix",
        "combustion_solver_family_and_damped_refinement_matrix",
        &report,
    ) {
        Ok(path) => println!("combustion solver matrix report={}", path.display()),
        Err(error) => eprintln!("unable to write combustion solver matrix report: {error}"),
    }
    println!("{}", tabled::Table::new(&rows));
}
