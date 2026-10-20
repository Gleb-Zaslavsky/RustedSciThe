use crate::error::RunnerError;
use crate::registry::ScenarioOutput;
use RustedSciThe::numerical::BE::{
    BE, BeSolverOptions, BeStatus, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use RustedSciThe::numerical::ivp_workloads::{WorkloadKind, build_workload};
use RustedSciThe::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetryExecution, IvpWarmStage};
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[path = "../../../benches/support/be_bench_support.rs"]
#[allow(dead_code)]
mod shared_bench_support;

pub fn run_diffusion_chain() -> Result<ScenarioOutput, RunnerError> {
    const DIMENSION: usize = 64;
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
    if times.len() != BE_STEPS + 1
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
const BE_DIMENSION: usize = 64;
const BE_STEPS: usize = 16;
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
            diagnostics: vec![
                (
                    "final_state_min".into(),
                    "state".into(),
                    final_state.iter().copied().fold(f64::INFINITY, f64::min),
                ),
                (
                    "final_state_max".into(),
                    "state".into(),
                    final_state
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                ),
            ],
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
    let minimum = detailed
        .states
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let maximum = detailed
        .states
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if (detailed.times.last().copied().unwrap_or(f64::NAN) - BE_T_END).abs() > 1.0e-12
        || minimum < -1.0e-10
        || maximum > 1.0
        || detailed.states.ncols() != BE_DIMENSION
        || detailed.states.iter().any(|value| !value.is_finite())
    {
        return Err(RunnerError::Scenario(format!(
            "BE symbolic diffusion invariant failed: t_end={}, min={minimum:e}, max={maximum:e}",
            detailed.times.last().copied().unwrap_or(f64::NAN)
        )));
    }
    let reference = diffusion_rk4_reference(&detailed.times);
    let mut max_scaled_reference_error = 0.0_f64;
    for (row_index, expected) in reference.iter().enumerate() {
        for (actual, expected) in detailed.states.row(row_index).iter().zip(expected) {
            let scale = 0.02 + 0.03 * expected.abs();
            max_scaled_reference_error =
                max_scaled_reference_error.max((actual - expected).abs() / scale);
        }
    }
    // This scale-aware bound allows Backward Euler's O(h) error while catching
    // a wrong symbolic route or a materially incorrect trajectory.
    if max_scaled_reference_error > 1.0 {
        return Err(RunnerError::Scenario(format!(
            "BE symbolic diffusion disagrees with independent RK4 reference: max_scaled_error={max_scaled_reference_error:e}"
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
        (
            "final_state_min".into(),
            "state".into(),
            final_state.iter().copied().fold(f64::INFINITY, f64::min),
        ),
        (
            "final_state_max".into(),
            "state".into(),
            final_state
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        ),
        (
            "off_vs_detailed_max_diff".into(),
            "absolute".into(),
            max_trajectory_diff,
        ),
        (
            "rk4_max_scaled_error".into(),
            "scaled".into(),
            max_scaled_reference_error,
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
            ("state_dimension".into(), BE_DIMENSION as u64),
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

fn diffusion_rk4_reference(times: &[f64]) -> Vec<Vec<f64>> {
    fn rhs(time: f64, state: &[f64]) -> Vec<f64> {
        (0..state.len())
            .map(|index| {
                let left = if index == 0 { 0.0 } else { state[index - 1] };
                let right = if index + 1 == state.len() {
                    0.0
                } else {
                    state[index + 1]
                };
                -20.0 * state[index]
                    + 4.0 * (left - 2.0 * state[index] + right)
                    + 0.20 * (-time).exp()
                    - 0.010 * state[index] * state[index]
            })
            .collect()
    }
    fn add_scaled(state: &[f64], derivative: &[f64], scale: f64) -> Vec<f64> {
        state
            .iter()
            .zip(derivative)
            .map(|(value, rate)| value + scale * rate)
            .collect()
    }

    // This independently integrates the canonical workload at a finer step
    // than BE, providing a correctness guard without timing the oracle.
    const REFINEMENT: usize = 8;
    let workload = build_workload(WorkloadKind::DiffusionChain, BE_DIMENSION);
    let mut state = workload.initial_state.as_slice().to_vec();
    let mut reference = Vec::with_capacity(times.len());
    reference.push(state.clone());
    for interval in times.windows(2) {
        let dt = (interval[1] - interval[0]) / REFINEMENT as f64;
        let mut time = interval[0];
        for _ in 0..REFINEMENT {
            let k1 = rhs(time, &state);
            let k2 = rhs(time + 0.5 * dt, &add_scaled(&state, &k1, 0.5 * dt));
            let k3 = rhs(time + 0.5 * dt, &add_scaled(&state, &k2, 0.5 * dt));
            let k4 = rhs(time + dt, &add_scaled(&state, &k3, dt));
            state = state
                .iter()
                .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(k4.iter()))))
                .map(|(value, (a, (b, (c, d))))| value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0)
                .collect();
            time += dt;
        }
        reference.push(state.clone());
    }
    reference
}

fn run_once(
    frontend: Frontend,
    route: Route,
    telemetry: BeTelemetryMode,
) -> Result<BeRun, RunnerError> {
    let full_started = Instant::now();
    let workload = build_workload(WorkloadKind::DiffusionChain, BE_DIMENSION);
    let parameter_names = workload
        .parameter_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let parameter_values = workload.parameter_values;
    let options = BeSolverOptions::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        1.0e-10,
        16,
        Some(BE_H),
        0.0,
        BE_T_END,
        workload.initial_state,
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
        .try_set_equation_parameters(Some(&parameter_names))
        .map_err(|error| RunnerError::Scenario(format!("BE parameter schema: {error}")))?;
    solver
        .set_parameter_values(parameter_values)
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
    if times.len() != BE_STEPS + 1
        || states.nrows() != times.len()
        || states.ncols() != BE_DIMENSION
    {
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
            id: concat!("be_diff64_", $frontend_key, "_", $suffix),
            solver: "be",
            groups: &["ivp", "stiff", "be-matrix"],
            workload: "parameterized diffusion chain, dimension=64, 16 fixed steps, t=[0,0.016]",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: "Dense (fixed implementation)",
            parallel_policy: "Sequential (fixed; no parallel dispatch API)",
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 2,
            gate_metrics: &["scenario_wall_ms"],
            description: "Matched BE canonical diffusion-chain route; dense layout and sequential dispatch are fixed",
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
