use crate::error::RunnerError;
use crate::registry::ScenarioOutput;
use RustedSciThe::numerical::BE::{
    BE, BeSolverOptions, BeStatus, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use RustedSciThe::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetryExecution, IvpWarmStage};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[path = "../../../benches/support/be_bench_support.rs"]
#[allow(dead_code)]
mod shared_bench_support;

pub fn run_diffusion_chain() -> Result<ScenarioOutput, RunnerError> {
    const DIMENSION: usize = 32;
    const STEP: f64 = 0.001;
    const FINAL_TIME: f64 = 0.016;

    let full_started = Instant::now();
    let setup_started = Instant::now();
    let mut solver = shared_bench_support::make_diffusion_solver(
        DIMENSION,
        true,
        BeTelemetryMode::Off,
        STEP,
        FINAL_TIME,
    );
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let solve_started = Instant::now();
    solver
        .try_solve()
        .map_err(|error| RunnerError::Scenario(format!("Backward Euler solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;

    if solver.status() != BeStatus::Finished {
        return Err(RunnerError::Scenario(format!(
            "Backward Euler did not reach the final time: {:?}",
            solver.status()
        )));
    }
    let (times, states) = solver.trajectory();
    if times.is_empty()
        || states.nrows() != times.len()
        || states.ncols() != DIMENSION
        || times.iter().any(|time| !time.is_finite())
        || states.iter().any(|value| !value.is_finite())
        || (times[times.len() - 1] - FINAL_TIME).abs() > 1.0e-12
    {
        return Err(RunnerError::Scenario(
            "Backward Euler returned an invalid trajectory or final time".into(),
        ));
    }

    let final_state = states.row(states.nrows() - 1);
    let minimum = final_state.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = final_state
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if minimum < -1.0e-10 || maximum > 2.0 + 1.0e-10 {
        return Err(RunnerError::Scenario(format!(
            "Backward Euler diffusion bounds failed: min={minimum:e}, max={maximum:e}"
        )));
    }
    black_box(states);

    Ok(ScenarioOutput {
        metrics: vec![
            ("solver_setup_ms".into(), "ms".into(), setup_ms),
            ("solve_ms".into(), "ms".into(), solve_ms),
            ("scenario_wall_ms".into(), "ms".into(), scenario_wall_ms),
        ],
        native_scopes: Vec::new(),
        counters: vec![
            ("state_dimension".into(), DIMENSION as u64),
            ("trajectory_points".into(), times.len() as u64),
        ],
        diagnostics: vec![
            ("final_time".into(), "time".into(), times[times.len() - 1]),
            ("final_state_min".into(), "state".into(), minimum),
            ("final_state_max".into(), "state".into(), maximum),
        ],
    })
}

const FRONTEND_EXPR: u8 = 0;
const FRONTEND_ATOM: u8 = 1;
const ROUTE_LAMBDIFY: u8 = 0;
const ROUTE_AOT: u8 = 1;
const BE_STEPS: usize = 40;
const BE_H: f64 = 0.001;
const BE_T_END: f64 = BE_STEPS as f64 * BE_H;

#[derive(Clone, Copy)]
enum Frontend {
    ExprLegacy,
    AtomView,
}

impl Frontend {
    fn from_id(id: u8) -> Self {
        match id {
            FRONTEND_EXPR => Self::ExprLegacy,
            FRONTEND_ATOM => Self::AtomView,
            _ => unreachable!("registered BE frontend id"),
        }
    }

    fn public(self) -> BeSymbolicAssemblyBackend {
        match self {
            Self::ExprLegacy => BeSymbolicAssemblyBackend::ExprLegacy,
            Self::AtomView => BeSymbolicAssemblyBackend::AtomViewNative,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomView => "AtomViewNative",
        }
    }
}

#[derive(Clone, Copy)]
enum Route {
    Lambdify,
    Aot,
}

impl Route {
    fn from_id(id: u8) -> Self {
        match id {
            ROUTE_LAMBDIFY => Self::Lambdify,
            ROUTE_AOT => Self::Aot,
            _ => unreachable!("registered BE route id"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Lambdify => "Lambdify",
            Self::Aot => "AOT(tcc,warm-cache)",
        }
    }
}

struct BeRun {
    times: Vec<f64>,
    states: nalgebra::DMatrix<f64>,
    setup_ms: f64,
    solve_ms: f64,
    scenario_wall_ms: f64,
    telemetry: Option<RustedSciThe::symbolic::ivp_telemetry::IvpTelemetrySnapshot>,
    details: RustedSciThe::numerical::BE::BeDetailedStatistics,
}

fn run_symbolic_cell_phase<const FRONTEND_ID: u8, const ROUTE_ID: u8>(
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let frontend = Frontend::from_id(FRONTEND_ID);
    let route = Route::from_id(ROUTE_ID);
    let plain = run_once(frontend, route, BeTelemetryMode::Off)?;
    black_box((&plain.times, &plain.states));
    if !diagnostic_only {
        let final_state = plain.states.row(plain.states.nrows() - 1);
        return Ok(ScenarioOutput {
            metrics: vec![
                ("solver_setup_ms".into(), "ms/off".into(), plain.setup_ms),
                ("solve_ms".into(), "ms/off".into(), plain.solve_ms),
                (
                    "scenario_wall_ms".into(),
                    "ms/off".into(),
                    plain.scenario_wall_ms,
                ),
            ],
            native_scopes: Vec::new(),
            counters: Vec::new(),
            diagnostics: vec![(
                "final_mass".into(),
                "conserved quantity".into(),
                final_state.iter().sum(),
            )],
        });
    }
    let detailed = run_once(frontend, route, BeTelemetryMode::Detailed)?;

    if plain.times.len() != detailed.times.len() || plain.states.shape() != detailed.states.shape()
    {
        return Err(RunnerError::Scenario(
            "BE telemetry mode changed trajectory shape".into(),
        ));
    }
    let max_trajectory_diff = plain
        .times
        .iter()
        .zip(&detailed.times)
        .map(|(left, right)| (left - right).abs())
        .chain(
            plain
                .states
                .iter()
                .zip(detailed.states.iter())
                .map(|(left, right)| (left - right).abs()),
        )
        .fold(0.0_f64, f64::max);
    if max_trajectory_diff > 1.0e-12 {
        return Err(RunnerError::Scenario(format!(
            "BE telemetry mode changed the trajectory by {max_trajectory_diff:e}"
        )));
    }

    let final_state = detailed.states.row(detailed.states.nrows() - 1);
    let final_mass = final_state.iter().sum::<f64>();
    let min_concentration = detailed
        .states
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if (detailed.times.last().copied().unwrap_or(f64::NAN) - BE_T_END).abs() > 1.0e-12
        || (final_mass - 1.0).abs() > 1.0e-10
        || min_concentration < -1.0e-10
        || detailed.states.iter().any(|value| !value.is_finite())
    {
        return Err(RunnerError::Scenario(format!(
            "BE symbolic combustion invariant failed: t_end={}, mass={final_mass:e}, min={min_concentration:e}",
            detailed.times.last().copied().unwrap_or(f64::NAN)
        )));
    }
    let reference = combustion_rk4_reference(&detailed.times);
    let mut max_reference_error = 0.0_f64;
    for (row_index, expected) in reference.iter().enumerate() {
        for (actual, expected) in detailed.states.row(row_index).iter().zip(expected) {
            max_reference_error = max_reference_error.max((actual - expected).abs());
        }
    }
    // Backward Euler is first-order; this bound covers its expected O(h)
    // discretization error while still catching wrong equations/routes.
    if max_reference_error > 5.0e-6 {
        return Err(RunnerError::Scenario(format!(
            "BE symbolic combustion disagrees with independent RK4 reference: max_error={max_reference_error:e}"
        )));
    }

    let snapshot = detailed.telemetry.as_ref().ok_or_else(|| {
        RunnerError::Scenario("BE Detailed route did not publish symbolic telemetry".into())
    })?;
    let expected_execution = match route {
        Route::Lambdify => IvpTelemetryExecution::Lambdify,
        Route::Aot => IvpTelemetryExecution::Aot,
    };
    if snapshot.execution != expected_execution {
        return Err(RunnerError::Scenario(format!(
            "BE requested {} but selected {:?}",
            route.label(),
            snapshot.execution
        )));
    }
    if matches!(route, Route::Aot) && snapshot.aot_runtime_ready == 0 {
        return Err(RunnerError::Scenario(
            "BE AOT route did not report a ready generated runtime".into(),
        ));
    }

    let mut metrics = Vec::new();
    for (stage, id) in [
        (IvpColdStage::SolverPreparation, "preparation_ms"),
        (IvpColdStage::SymbolicJacobian, "symbolic_jacobian_ms"),
        (
            IvpColdStage::ResidualLambdification,
            "residual_lambdification_ms",
        ),
        (
            IvpColdStage::JacobianLambdification,
            "jacobian_lambdification_ms",
        ),
        (IvpColdStage::AotBuild, "aot_build_ms"),
        (IvpColdStage::AotLink, "aot_link_ms"),
    ] {
        let timing = snapshot.cold_stage(stage);
        if timing.recorded {
            metrics.push((
                id.into(),
                if matches!(stage, IvpColdStage::AotBuild | IvpColdStage::AotLink) {
                    "ms/detailed/aot-cold".into()
                } else {
                    "ms/detailed/preparation".into()
                },
                timing.elapsed.as_secs_f64() * 1_000.0,
            ));
        }
    }
    for (stage, id) in [
        (IvpWarmStage::ResidualEvaluation, "residual_ms"),
        (IvpWarmStage::JacobianEvaluation, "jacobian_ms"),
        (IvpWarmStage::Factorization, "factorization_ms"),
        (IvpWarmStage::RhsSolve, "linear_solve_ms"),
        (IvpWarmStage::Solve, "native_solve_stage_ms"),
    ] {
        let timing = snapshot.warm_stage(stage);
        if timing.recorded {
            metrics.push((
                id.into(),
                "ms/detailed/solve".into(),
                timing.elapsed.as_secs_f64() * 1_000.0,
            ));
        }
    }
    metrics.extend([
        (
            "be_factorization_ms_total".into(),
            "ms/detailed/be".into(),
            detailed.details.factorization_ms_total,
        ),
        (
            "be_linear_solve_ms_total".into(),
            "ms/detailed/be".into(),
            detailed.details.linear_solve_ms_total,
        ),
    ]);

    let mut diagnostics = vec![
        ("final_mass".into(), "conserved quantity".into(), final_mass),
        (
            "off_vs_detailed_max_diff".into(),
            "absolute".into(),
            max_trajectory_diff,
        ),
        (
            "max_rk4_reference_abs_error".into(),
            "absolute".into(),
            max_reference_error,
        ),
        (
            "minimum_concentration".into(),
            "state".into(),
            min_concentration,
        ),
    ];
    for stage in [
        IvpColdStage::AotMaterialization,
        IvpColdStage::AotCacheLookup,
        IvpColdStage::AotLowering,
        IvpColdStage::AotSourceGeneration,
        IvpColdStage::AotPublication,
    ] {
        let timing = snapshot.cold_stage(stage);
        if timing.recorded {
            metrics.push((
                format!("aot_cold_{}_ms", stage.label()),
                "ms/detailed/aot-cold".into(),
                timing.elapsed.as_secs_f64() * 1_000.0,
            ));
        }
    }
    if matches!(route, Route::Aot) {
        diagnostics.extend([
            (
                "aot_build_attempts".into(),
                "count".into(),
                snapshot.aot_build_attempts as f64,
            ),
            (
                "aot_link_attempts".into(),
                "count".into(),
                snapshot.aot_link_attempts as f64,
            ),
            (
                "aot_runtime_ready".into(),
                "count".into(),
                snapshot.aot_runtime_ready as f64,
            ),
        ]);
    }

    Ok(ScenarioOutput {
        metrics,
        native_scopes: Vec::new(),
        counters: vec![
            ("trajectory_points".into(), detailed.times.len() as u64),
            ("state_dimension".into(), 3),
            ("residual_evaluations".into(), snapshot.residual_evaluations),
            ("jacobian_evaluations".into(), snapshot.jacobian_evaluations),
            (
                "factorizations".into(),
                detailed.details.factorization_calls as u64,
            ),
            (
                "linear_solves".into(),
                detailed.details.linear_solve_calls as u64,
            ),
            (
                "accepted_steps".into(),
                detailed.details.accepted_steps as u64,
            ),
            ("failed_steps".into(), detailed.details.failed_steps as u64),
        ],
        diagnostics,
    })
}

fn run_symbolic_cell<const FRONTEND_ID: u8, const ROUTE_ID: u8>()
-> Result<ScenarioOutput, RunnerError> {
    run_symbolic_cell_phase::<FRONTEND_ID, ROUTE_ID>(false)
}

fn diagnose<const FRONTEND_ID: u8, const ROUTE_ID: u8>(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_symbolic_cell_phase::<FRONTEND_ID, ROUTE_ID>(true)
}

fn combustion_rk4_reference(times: &[f64]) -> Vec<[f64; 3]> {
    fn rhs(y: [f64; 3]) -> [f64; 3] {
        let [fuel, radical, _product] = y;
        let reaction = 0.1 * fuel * radical;
        let secondary = 0.2 * radical * radical;
        [
            -0.3 * fuel + reaction,
            0.3 * fuel - reaction - secondary,
            secondary,
        ]
    }
    fn add_scaled(y: [f64; 3], k: [f64; 3], scale: f64) -> [f64; 3] {
        [
            y[0] + scale * k[0],
            y[1] + scale * k[1],
            y[2] + scale * k[2],
        ]
    }

    let mut state = [0.9, 0.1, 0.0];
    let mut reference = Vec::with_capacity(times.len());
    reference.push(state);
    for interval in times.windows(2) {
        let dt = (interval[1] - interval[0]) / 100.0;
        for _ in 0..100 {
            let k1 = rhs(state);
            let k2 = rhs(add_scaled(state, k1, 0.5 * dt));
            let k3 = rhs(add_scaled(state, k2, 0.5 * dt));
            let k4 = rhs(add_scaled(state, k3, dt));
            state = [
                state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
                state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
                state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            ];
        }
        reference.push(state);
    }
    reference
}

fn run_once(
    frontend: Frontend,
    route: Route,
    telemetry: BeTelemetryMode,
) -> Result<BeRun, RunnerError> {
    let full_started = Instant::now();
    let equations = vec![
        Expr::parse_expression("-0.3*fuel+rate*fuel*radical"),
        Expr::parse_expression("0.3*fuel-rate*fuel*radical-0.2*radical^2"),
        Expr::parse_expression("0.2*radical^2"),
    ];
    let options = BeSolverOptions::new(
        equations,
        vec!["fuel".into(), "radical".into(), "product".into()],
        "t".into(),
        1.0e-10,
        16,
        Some(BE_H),
        0.0,
        BE_T_END,
        nalgebra::DVector::from_vec(vec![0.9, 0.1, 0.0]),
    )
    .with_telemetry_mode(telemetry)
    .with_symbolic_assembly_backend(frontend.public());
    let options = match route {
        Route::Lambdify => options,
        Route::Aot => options.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                PathBuf::from("target/perf-gates-aot/be")
                    .join(frontend.label().to_ascii_lowercase()),
            )
            .with_c_tcc(),
        ),
    };

    let setup_started = Instant::now();
    let mut solver = BE::try_new_with_options(options)
        .map_err(|error| RunnerError::Scenario(format!("BE setup: {error}")))?;
    solver
        .try_set_equation_parameters(Some(&["rate"]))
        .map_err(|error| RunnerError::Scenario(format!("BE parameter schema: {error}")))?;
    solver
        .set_parameter_values(nalgebra::DVector::from_vec(vec![0.1]))
        .map_err(|error| RunnerError::Scenario(format!("BE parameter binding: {error}")))?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let solve_started = Instant::now();
    solver
        .try_solve()
        .map_err(|error| RunnerError::Scenario(format!("BE solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;
    if solver.status() != BeStatus::Finished {
        return Err(RunnerError::Scenario(format!(
            "BE did not finish: {:?}",
            solver.status()
        )));
    }
    let (times, states) = solver.trajectory();
    if times.is_empty() || states.nrows() != times.len() || states.ncols() != 3 {
        return Err(RunnerError::Scenario(
            "BE produced an invalid trajectory shape".into(),
        ));
    }
    let telemetry_snapshot = solver.symbolic_ivp_telemetry_snapshot();
    if telemetry != BeTelemetryMode::Off && telemetry_snapshot.is_none() {
        return Err(RunnerError::Scenario(
            "BE telemetry was enabled but no snapshot was returned".into(),
        ));
    }
    let details = solver.detailed_statistics();
    black_box((&times, &states));
    Ok(BeRun {
        times: times.iter().copied().collect(),
        states: states.clone(),
        setup_ms,
        solve_ms,
        scenario_wall_ms,
        telemetry: telemetry_snapshot,
        details,
    })
}

macro_rules! be_scenario {
    ($frontend_id:ident, $frontend:literal, $frontend_key:literal, $route_id:ident, $execution:literal, $suffix:literal) => {
        crate::registry::Scenario {
            id: concat!("be_combustion_", $frontend_key, "_", $suffix),
            solver: "be",
            groups: &["ivp", "stiff", "be-matrix"],
            workload: "parameterized combustion chain, dimension=3, 40 fixed steps, t=[0,0.04]",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: "Dense (fixed implementation)",
            parallel_policy: "Sequential (fixed; no parallel dispatch API)",
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "Matched BE parameterized combustion route; layout and dispatch are intentionally fixed",
            run: run_symbolic_cell::<{ $frontend_id }, { $route_id }>,
            diagnose: Some(diagnose::<{ $frontend_id }, { $route_id }>),
        }
    };
}

pub static MATRIX_SCENARIOS: [crate::registry::Scenario; 4] = [
    be_scenario!(
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        ROUTE_LAMBDIFY,
        "Lambdify",
        "lambdify"
    ),
    be_scenario!(
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        ROUTE_LAMBDIFY,
        "Lambdify",
        "lambdify"
    ),
    be_scenario!(
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        ROUTE_AOT,
        "AOT(warm-cache)",
        "aot"
    ),
    be_scenario!(
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        ROUTE_AOT,
        "AOT(warm-cache)",
        "aot"
    ),
];
