use crate::error::RunnerError;
use crate::registry::ScenarioOutput;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::{
    FrozenSolverOptions, NRBVP as FrozenBvp,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
};
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::symbolic::bvp::telemetry::{
    BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode,
};
use RustedSciThe::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use RustedSciThe::symbolic::codegen::codegen_provider_api::MatrixBackend;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

const N_STEPS: usize = 32;
const CONTINUATION_STEPS: usize = 4;
pub(crate) const FAMILY_DAMPED: u8 = 0;
pub(crate) const FAMILY_FROZEN: u8 = 1;
pub(crate) const WORKLOAD_OSCILLATOR: u8 = 0;
pub(crate) const WORKLOAD_NONLINEAR: u8 = 1;
pub(crate) const WORKLOAD_FROZEN_LINEAR: u8 = 2;
pub(crate) const MATRIX_DENSE: u8 = 0;
pub(crate) const MATRIX_SPARSE: u8 = 1;
pub(crate) const MATRIX_BANDED: u8 = 2;
pub(crate) const FRONTEND_EXPR_LEGACY: u8 = 0;
pub(crate) const FRONTEND_ATOM_VIEW: u8 = 1;
pub(crate) const ROUTE_LAMBDIFY: u8 = 0;
pub(crate) const ROUTE_AOT: u8 = 1;
pub(crate) const DISPATCH_SEQUENTIAL: u8 = 0;
pub(crate) const DISPATCH_PARALLEL: u8 = 1;
pub(crate) const DISPATCH_AUTO: u8 = 2;

#[derive(Clone, Copy)]
enum Workload {
    Oscillator,
    Nonlinear,
    FrozenLinear,
}

impl Workload {
    fn from_id(id: u8) -> Self {
        match id {
            WORKLOAD_OSCILLATOR => Self::Oscillator,
            WORKLOAD_NONLINEAR => Self::Nonlinear,
            WORKLOAD_FROZEN_LINEAR => Self::FrozenLinear,
            _ => unreachable!("registered BVP_Damp workload id"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Oscillator => "oscillator",
            Self::Nonlinear => "nonlinear-exact",
            Self::FrozenLinear => "linear-exact",
        }
    }

    fn equations(self) -> Vec<Expr> {
        match self {
            Self::Oscillator => vec![
                Expr::parse_expression("alpha*z"),
                Expr::parse_expression("-alpha*y"),
            ],
            Self::Nonlinear => vec![
                Expr::parse_expression("alpha*z"),
                Expr::parse_expression("2*y^3"),
            ],
            Self::FrozenLinear => vec![
                Expr::parse_expression("alpha*z"),
                Expr::parse_expression("0.0"),
            ],
        }
    }

    fn interval(self) -> f64 {
        match self {
            Self::Oscillator => std::f64::consts::FRAC_PI_2,
            Self::Nonlinear => 1.0,
            Self::FrozenLinear => 1.0,
        }
    }

    fn initial_guess(self) -> DMatrix<f64> {
        DMatrix::from_fn(2, N_STEPS, |row, column| {
            let x = self.interval() * column as f64 / N_STEPS as f64;
            match self {
                Self::Oscillator => {
                    if row == 0 {
                        x.sin()
                    } else {
                        x.cos()
                    }
                }
                Self::Nonlinear => {
                    if row == 0 {
                        1.0 / (1.0 + x)
                    } else {
                        -1.0 / (1.0 + x).powi(2)
                    }
                }
                Self::FrozenLinear => {
                    if row == 0 {
                        x
                    } else {
                        1.0
                    }
                }
            }
        })
    }

    fn boundary_conditions(self, frozen: bool) -> HashMap<String, Vec<(usize, f64)>> {
        match (self, frozen) {
            (Self::Oscillator, false) => {
                HashMap::from([("y".to_string(), vec![(0, 0.0), (1, 1.0)])])
            }
            (Self::Nonlinear, false) => {
                HashMap::from([("y".to_string(), vec![(0, 1.0), (1, 0.5)])])
            }
            (Self::FrozenLinear, false) | (Self::FrozenLinear, true) => HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            // Frozen takes one endpoint condition per state variable. These
            // are equivalent two-point constraints for the oscillator and
            // the established y-left/z-right nonlinear fixture.
            (Self::Oscillator, true) => HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            (Self::Nonlinear, true) => HashMap::from([
                ("y".to_string(), vec![(0, 1.0)]),
                ("z".to_string(), vec![(1, -0.25)]),
            ]),
        }
    }
}

#[derive(Clone, Copy)]
enum MatrixRoute {
    Dense,
    Sparse,
    Banded,
}

impl MatrixRoute {
    fn from_id(id: u8) -> Self {
        match id {
            MATRIX_DENSE => Self::Dense,
            MATRIX_SPARSE => Self::Sparse,
            MATRIX_BANDED => Self::Banded,
            _ => unreachable!("registered BVP_Damp matrix id"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse-faer",
            Self::Banded => "Banded-faithful",
        }
    }

    fn backend(self) -> MatrixBackend {
        match self {
            Self::Dense => MatrixBackend::Dense,
            Self::Sparse => MatrixBackend::SparseCol,
            Self::Banded => MatrixBackend::Banded,
        }
    }
}

#[derive(Clone, Copy)]
enum Frontend {
    ExprLegacy,
    AtomView,
}

impl Frontend {
    fn from_id(id: u8) -> Self {
        match id {
            FRONTEND_EXPR_LEGACY => Self::ExprLegacy,
            FRONTEND_ATOM_VIEW => Self::AtomView,
            _ => unreachable!("registered BVP_Damp frontend id"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomView => "AtomView",
        }
    }

    fn backend(self) -> BvpSymbolicAssemblyBackend {
        match self {
            Self::ExprLegacy => BvpSymbolicAssemblyBackend::ExprLegacy,
            Self::AtomView => BvpSymbolicAssemblyBackend::AtomView,
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
            _ => unreachable!("registered BVP_Damp route id"),
        }
    }
}

#[derive(Clone, Copy)]
enum Dispatch {
    Sequential,
    Parallel,
    Auto,
}

impl Dispatch {
    fn from_id(id: u8) -> Self {
        match id {
            DISPATCH_SEQUENTIAL => Self::Sequential,
            DISPATCH_PARALLEL => Self::Parallel,
            DISPATCH_AUTO => Self::Auto,
            _ => unreachable!("registered BVP_Damp dispatch id"),
        }
    }

    fn lambdify_policy(self) -> BvpLambdifyExecutionPolicy {
        match self {
            Self::Sequential => BvpLambdifyExecutionPolicy::Sequential,
            Self::Parallel => BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
            Self::Auto => BvpLambdifyExecutionPolicy::Auto { min_work: 0 },
        }
    }
}

#[derive(Clone)]
struct RunConfig {
    family: u8,
    workload: Workload,
    matrix: MatrixRoute,
    frontend: Frontend,
    route: Route,
    dispatch: Dispatch,
}

fn generated_backend_config(
    matrix: MatrixRoute,
    frontend: Frontend,
    route: Route,
    dispatch: Dispatch,
    telemetry: BvpTelemetryMode,
    detailed_callbacks: bool,
) -> GeneratedBackendConfig {
    let mut config = match (route, matrix) {
        (Route::Aot, MatrixRoute::Sparse) => {
            GeneratedBackendConfig::sparse_build_if_missing_release()
        }
        (Route::Aot, MatrixRoute::Banded) => {
            GeneratedBackendConfig::banded_build_if_missing_release()
        }
        _ => GeneratedBackendConfig::default(),
    }
    .with_matrix_backend_override(matrix.backend())
    .with_symbolic_assembly_backend(frontend.backend())
    .with_lambdify_telemetry_mode(if detailed_callbacks {
        BvpLambdifyTelemetryMode::Detailed
    } else {
        BvpLambdifyTelemetryMode::Off
    })
    .with_bvp_telemetry_mode(telemetry);

    if matches!(route, Route::Lambdify) {
        config = config
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_lambdify_execution_policy(dispatch.lambdify_policy());
    } else {
        config = config
            .with_backend_policy_override(Some(BackendSelectionPolicy::AotOnly))
            .with_aot_execution_policy(AotExecutionPolicy::Auto)
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            });
    }
    config
}

fn make_damped_options(config: &RunConfig, telemetry: BvpTelemetryMode) -> DampedSolverOptions {
    let generated = generated_backend_config(
        config.matrix,
        config.frontend,
        config.route,
        config.dispatch,
        telemetry,
        telemetry == BvpTelemetryMode::Detailed,
    );
    match config.matrix {
        MatrixRoute::Dense => DampedSolverOptions::dense_damped(),
        MatrixRoute::Sparse => DampedSolverOptions::sparse_damped(),
        MatrixRoute::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
    }
    .with_generated_backend_config(generated)
    .with_strategy_params(Some(SolverParams::default()))
    .with_abs_tolerance(1.0e-9)
    .with_rel_tolerance(HashMap::from([
        ("y".to_string(), 1.0e-7),
        ("z".to_string(), 1.0e-7),
    ]))
    .with_max_iterations(30)
    .with_bounds(HashMap::from([
        ("y".to_string(), (-10.0, 10.0)),
        ("z".to_string(), (-10.0, 10.0)),
    ]))
}

fn make_frozen_options(config: &RunConfig, telemetry: BvpTelemetryMode) -> FrozenSolverOptions {
    let generated = generated_backend_config(
        config.matrix,
        config.frontend,
        config.route,
        config.dispatch,
        telemetry,
        telemetry == BvpTelemetryMode::Detailed,
    );
    match config.matrix {
        MatrixRoute::Dense => FrozenSolverOptions::dense_frozen(),
        MatrixRoute::Sparse => FrozenSolverOptions::sparse_frozen(),
        MatrixRoute::Banded => FrozenSolverOptions::banded_frozen().with_banded_lambdify(),
    }
    .with_generated_backend_config(generated)
    .with_tolerance(1.0e-9)
    .with_max_iterations(30)
}

#[derive(Clone)]
struct RunResult {
    solution: DMatrix<f64>,
    setup_ms: f64,
    prepare_ms: f64,
    first_solve_ms: f64,
    continuation_ms: f64,
    continuation_prepare_ms: f64,
    scenario_wall_ms: f64,
    telemetry: RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot,
}

fn run_damped(config: &RunConfig, telemetry: BvpTelemetryMode) -> Result<RunResult, RunnerError> {
    let full_started = Instant::now();
    let setup_started = Instant::now();
    let mut solver = NRBVP::new_with_options(
        config.workload.equations(),
        config.workload.initial_guess(),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        config.workload.boundary_conditions(false),
        0.0,
        config.workload.interval(),
        N_STEPS,
        make_damped_options(config, telemetry),
    );
    solver.dont_save_log(true);
    solver
        .try_set_parameter_names(Some(&["alpha"]))
        .map_err(|error| {
            RunnerError::Scenario(format!("BVP_Damp parameter declaration: {error:?}"))
        })?;
    solver
        .try_set_parameter_values(Some(vec![1.0]))
        .map_err(|error| RunnerError::Scenario(format!("BVP_Damp parameter binding: {error:?}")))?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let prepare_started = Instant::now();
    solver
        .try_eq_generate(None, None)
        .map_err(|error| RunnerError::Scenario(format!("BVP_Damp Damped prepare: {error:?}")))?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    solver
        .try_solver_prepared()
        .map_err(|error| RunnerError::Scenario(format!("BVP_Damp Damped solve: {error:?}")))?
        .ok_or_else(|| RunnerError::Scenario("BVP_Damp Damped returned no initial solve".into()))?;
    let first_solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;

    let continuation_started = Instant::now();
    let mut continuation_prepare_ms = 0.0;
    for index in 1..CONTINUATION_STEPS {
        solver
            .try_set_parameter_values(Some(vec![1.0 + index as f64 * 0.1]))
            .map_err(|error| {
                RunnerError::Scenario(format!("BVP_Damp Damped continuation bind: {error:?}"))
            })?;
        if matches!(config.route, Route::Aot) {
            let prepare_started = Instant::now();
            solver.try_eq_generate(None, None).map_err(|error| {
                RunnerError::Scenario(format!(
                    "BVP_Damp Damped AOT continuation prepare: {error:?}"
                ))
            })?;
            continuation_prepare_ms += prepare_started.elapsed().as_secs_f64() * 1_000.0;
        }
        solver
            .try_solver_prepared()
            .map_err(|error| {
                RunnerError::Scenario(format!("BVP_Damp Damped continuation solve: {error:?}"))
            })?
            .ok_or_else(|| {
                RunnerError::Scenario("BVP_Damp Damped continuation returned no solution".into())
            })?;
    }
    let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;
    let solution = solver
        .get_result()
        .ok_or_else(|| RunnerError::Scenario("BVP_Damp Damped result is missing".into()))?;
    let telemetry = solver.get_statistics().telemetry;

    validate_solution(config, &solution)?;
    if matches!(config.route, Route::Aot)
        && telemetry
            .plan
            .as_ref()
            .and_then(|plan| plan.selected_backend)
            != Some(SelectedBackendKind::AotCompiled)
    {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp Damped {} {} route did not select compiled AOT",
            config.matrix.label(),
            config.frontend.label()
        )));
    }
    Ok(RunResult {
        solution,
        setup_ms,
        prepare_ms,
        first_solve_ms,
        continuation_ms,
        continuation_prepare_ms,
        scenario_wall_ms,
        telemetry,
    })
}

fn run_frozen(config: &RunConfig, telemetry: BvpTelemetryMode) -> Result<RunResult, RunnerError> {
    let full_started = Instant::now();
    let setup_started = Instant::now();
    let mut solver = FrozenBvp::new_with_options(
        config.workload.equations(),
        config.workload.initial_guess(),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        config.workload.boundary_conditions(true),
        0.0,
        config.workload.interval(),
        N_STEPS,
        make_frozen_options(config, telemetry),
    );
    solver.dont_save_log(true);
    solver
        .try_set_parameter_names(Some(&["alpha"]))
        .map_err(|error| {
            RunnerError::Scenario(format!("BVP_Damp Frozen parameter declaration: {error:?}"))
        })?;
    solver
        .try_set_parameter_values(Some(vec![1.0]))
        .map_err(|error| {
            RunnerError::Scenario(format!("BVP_Damp Frozen parameter binding: {error:?}"))
        })?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let prepare_started = Instant::now();
    solver
        .try_eq_generate()
        .map_err(|error| RunnerError::Scenario(format!("BVP_Damp Frozen prepare: {error:?}")))?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    solver
        .try_solver_prepared()
        .map_err(|error| RunnerError::Scenario(format!("BVP_Damp Frozen solve: {error:?}")))?
        .ok_or_else(|| RunnerError::Scenario("BVP_Damp Frozen returned no initial solve".into()))?;
    let first_solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;

    let continuation_started = Instant::now();
    let mut continuation_prepare_ms = 0.0;
    for index in 1..CONTINUATION_STEPS {
        solver
            .try_set_parameter_values(Some(vec![1.0 + index as f64 * 0.1]))
            .map_err(|error| {
                RunnerError::Scenario(format!("BVP_Damp Frozen continuation bind: {error:?}"))
            })?;
        if matches!(config.route, Route::Aot) {
            let prepare_started = Instant::now();
            solver.try_eq_generate().map_err(|error| {
                RunnerError::Scenario(format!(
                    "BVP_Damp Frozen AOT continuation prepare: {error:?}"
                ))
            })?;
            continuation_prepare_ms += prepare_started.elapsed().as_secs_f64() * 1_000.0;
        }
        solver
            .try_solver_prepared()
            .map_err(|error| {
                RunnerError::Scenario(format!("BVP_Damp Frozen continuation solve: {error:?}"))
            })?
            .ok_or_else(|| {
                RunnerError::Scenario("BVP_Damp Frozen continuation returned no solution".into())
            })?;
    }
    let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;
    let solution = solver
        .get_result()
        .ok_or_else(|| RunnerError::Scenario("BVP_Damp Frozen result is missing".into()))?;
    let telemetry = solver.get_statistics().telemetry;

    validate_solution(config, &solution)?;
    if matches!(config.route, Route::Aot)
        && telemetry
            .plan
            .as_ref()
            .and_then(|plan| plan.selected_backend)
            != Some(SelectedBackendKind::AotCompiled)
    {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp Frozen {} {} route did not select compiled AOT",
            config.matrix.label(),
            config.frontend.label()
        )));
    }
    Ok(RunResult {
        solution,
        setup_ms,
        prepare_ms,
        first_solve_ms,
        continuation_ms,
        continuation_prepare_ms,
        scenario_wall_ms,
        telemetry,
    })
}

fn validate_solution(config: &RunConfig, solution: &DMatrix<f64>) -> Result<(), RunnerError> {
    let expected_nodes = if config.family == FAMILY_FROZEN {
        N_STEPS
    } else {
        N_STEPS + 1
    };
    if solution.nrows() != expected_nodes
        || solution.ncols() != 2
        || solution.iter().any(|value| !value.is_finite())
    {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp {} {} {} produced invalid solution shape {:?}",
            if config.family == FAMILY_DAMPED {
                "Damped"
            } else {
                "Frozen"
            },
            config.workload.label(),
            config.matrix.label(),
            solution.shape()
        )));
    }
    if config.family == FAMILY_FROZEN && matches!(config.workload, Workload::FrozenLinear) {
        let max_error = (0..solution.nrows())
            .map(|row| {
                let x = (row + 1) as f64 / N_STEPS as f64;
                (solution[(row, 0)] - 1.3 * x)
                    .abs()
                    .max((solution[(row, 1)] - 1.0).abs())
            })
            .fold(0.0_f64, f64::max);
        if max_error > 1.0e-8 {
            return Err(RunnerError::Scenario(format!(
                "BVP_Damp Frozen linear exact-profile error too large: {max_error:e}"
            )));
        }
        return Ok(());
    }
    let conditions = config
        .workload
        .boundary_conditions(config.family == FAMILY_FROZEN);
    let last_node = solution.nrows() - 1;
    let boundary_error = conditions
        .iter()
        .flat_map(|(variable, entries)| {
            let column = if variable == "y" { 0 } else { 1 };
            entries.iter().map(move |(endpoint, expected)| {
                let row = if *endpoint == 0 { 0 } else { last_node };
                (solution[(row, column)] - expected).abs()
            })
        })
        .fold(0.0_f64, f64::max);
    if boundary_error > 1.0e-6 {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp boundary residual too large: {boundary_error:e}; first={:?}, last={:?}",
            [solution[(0, 0)], solution[(0, 1)]],
            [solution[(last_node, 0)], solution[(last_node, 1)]]
        )));
    }
    Ok(())
}

fn run_phase<
    const FAMILY: u8,
    const WORKLOAD: u8,
    const MATRIX: u8,
    const FRONTEND: u8,
    const ROUTE: u8,
    const DISPATCH: u8,
>(
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let config = RunConfig {
        family: FAMILY,
        workload: Workload::from_id(WORKLOAD),
        matrix: MatrixRoute::from_id(MATRIX),
        frontend: Frontend::from_id(FRONTEND),
        route: Route::from_id(ROUTE),
        dispatch: Dispatch::from_id(DISPATCH),
    };
    if matches!(config.route, Route::Aot) && matches!(config.matrix, MatrixRoute::Dense) {
        return Err(RunnerError::Scenario(
            "BVP_Damp Dense AOT is intentionally not a registered route".into(),
        ));
    }

    // Keep the gate solve uninstrumented. A second, identically configured
    // solve provides counters and stage telemetry without contaminating it.
    let plain = if FAMILY == FAMILY_DAMPED {
        run_damped(&config, BvpTelemetryMode::Off)?
    } else {
        run_frozen(&config, BvpTelemetryMode::Off)?
    };
    black_box(&plain.solution);
    if !diagnostic_only {
        return Ok(ScenarioOutput {
            metrics: vec![
                ("solver_setup_ms".into(), "ms/off".into(), plain.setup_ms),
                ("prepare_ms".into(), "ms/off".into(), plain.prepare_ms),
                (
                    "first_solve_ms".into(),
                    "ms/off".into(),
                    plain.first_solve_ms,
                ),
                (
                    "continuation_ms".into(),
                    "ms/off".into(),
                    plain.continuation_ms,
                ),
                (
                    "scenario_wall_ms".into(),
                    "ms/off".into(),
                    plain.scenario_wall_ms,
                ),
                (
                    "continuation_prepare_ms".into(),
                    "ms/off".into(),
                    plain.continuation_prepare_ms,
                ),
            ],
            native_scopes: Vec::new(),
            counters: Vec::new(),
            diagnostics: vec![(
                "max_abs_solution".into(),
                "state".into(),
                plain.solution.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
            )],
        });
    }
    let diagnostic = if FAMILY == FAMILY_DAMPED {
        run_damped(&config, BvpTelemetryMode::Detailed)?
    } else {
        run_frozen(&config, BvpTelemetryMode::Detailed)?
    };
    let solution_diff = plain
        .solution
        .iter()
        .zip(diagnostic.solution.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    if solution_diff > 1.0e-8 {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp telemetry changed solution: max difference {solution_diff:e}"
        )));
    }

    let stats = diagnostic.telemetry;
    let mut metrics = vec![
        (
            "native_scope_solve_ms".into(),
            "ms/detailed/typed-scope".into(),
            stats.scopes.solve.elapsed.as_secs_f64() * 1_000.0,
        ),
        (
            "native_scope_mesh_revision_ms".into(),
            "ms/detailed/typed-scope".into(),
            stats.scopes.mesh_revision.elapsed.as_secs_f64() * 1_000.0,
        ),
        (
            "native_scope_iteration_ms".into(),
            "ms/detailed/typed-scope".into(),
            stats.scopes.iteration.elapsed.as_secs_f64() * 1_000.0,
        ),
        (
            "native_scope_damping_trial_ms".into(),
            "ms/detailed/typed-scope".into(),
            stats.scopes.damping_trial.elapsed.as_secs_f64() * 1_000.0,
        ),
        (
            "stage_residual_ms".into(),
            "ms/detailed".into(),
            stats.timings.residual.as_secs_f64() * 1_000.0,
        ),
        (
            "stage_jacobian_ms".into(),
            "ms/detailed".into(),
            stats.timings.jacobian.as_secs_f64() * 1_000.0,
        ),
        (
            "stage_linear_ms".into(),
            "ms/detailed".into(),
            stats.timings.linear_system.as_secs_f64() * 1_000.0,
        ),
        (
            "stage_factorization_ms".into(),
            "ms/detailed".into(),
            stats.timings.factorization.as_secs_f64() * 1_000.0,
        ),
    ];
    if let Some(generation) = stats.generation {
        metrics.extend([
            (
                "stage_generation_ms".into(),
                "ms/detailed".into(),
                generation.total.as_secs_f64() * 1_000.0,
            ),
            (
                "stage_symbolic_jacobian_ms".into(),
                "ms/detailed".into(),
                generation.symbolic_jacobian.as_secs_f64() * 1_000.0,
            ),
            (
                "stage_backend_selection_ms".into(),
                "ms/detailed".into(),
                generation.backend_selection.as_secs_f64() * 1_000.0,
            ),
            (
                "stage_runtime_binding_ms".into(),
                "ms/detailed".into(),
                generation.runtime_binding.as_secs_f64() * 1_000.0,
            ),
        ]);
    }
    if let Some(atom) = stats.atom_discretization {
        metrics.push((
            "stage_atom_discretization_ms".into(),
            "ms/detailed".into(),
            atom.total.as_secs_f64() * 1_000.0,
        ));
    }
    if let Some(callback) = stats.legacy_lambdify.or(stats.atom_lambdify) {
        metrics.push((
            "stage_lambdify_callbacks_ms".into(),
            "ms/detailed".into(),
            (callback.residual_elapsed + callback.jacobian_elapsed).as_secs_f64() * 1_000.0,
        ));
    }
    let solution_max_abs = diagnostic
        .solution
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let native_scopes = [
        ("native_scope_solve_ms", "Solve"),
        ("native_scope_mesh_revision_ms", "MeshRevision"),
        ("native_scope_iteration_ms", "Iteration"),
        ("native_scope_damping_trial_ms", "DampingTrial"),
    ]
    .into_iter()
    .map(|(metric_id, stage)| crate::registry::NativeMetricScope {
        metric_id: metric_id.into(),
        stage: stage.into(),
        kind: "typed_aggregate".into(),
        parent: None,
    })
    .collect();
    Ok(ScenarioOutput {
        metrics,
        native_scopes,
        counters: vec![
            ("mesh_nodes".into(), diagnostic.solution.nrows() as u64),
            ("state_dimension".into(), diagnostic.solution.ncols() as u64),
            ("iterations".into(), stats.counters.iterations),
            ("residual_calls".into(), stats.counters.residual_calls),
            ("jacobian_requests".into(), stats.counters.jacobian_requests),
            ("factorizations".into(), stats.counters.factorizations),
            (
                "factorization_cache_hits".into(),
                stats.counters.factorization_cache_hits,
            ),
            ("grid_refinements".into(), stats.counters.grid_refinements),
        ],
        diagnostics: vec![
            ("max_abs_solution".into(), "state".into(), solution_max_abs),
            (
                "off_vs_detailed_max_diff".into(),
                "absolute".into(),
                solution_diff,
            ),
        ],
    })
}

fn run<
    const FAMILY: u8,
    const WORKLOAD: u8,
    const MATRIX: u8,
    const FRONTEND: u8,
    const ROUTE: u8,
    const DISPATCH: u8,
>() -> Result<ScenarioOutput, RunnerError> {
    run_phase::<FAMILY, WORKLOAD, MATRIX, FRONTEND, ROUTE, DISPATCH>(false)
}

fn diagnose<
    const FAMILY: u8,
    const WORKLOAD: u8,
    const MATRIX: u8,
    const FRONTEND: u8,
    const ROUTE: u8,
    const DISPATCH: u8,
>(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_phase::<FAMILY, WORKLOAD, MATRIX, FRONTEND, ROUTE, DISPATCH>(true)
}

macro_rules! scenario {
    ($family_id:ident, $family:literal, $workload_id:ident, $workload:literal,
     $matrix_id:ident, $matrix:literal, $frontend_id:ident, $frontend:literal,
     $route_id:ident, $execution:literal, $dispatch_id:ident, $dispatch:literal,
     $suffix:literal) => {
        crate::registry::Scenario {
            id: concat!(
                "bvp_damp_", $family, "_", $workload, "_", $frontend, "_", $matrix, "_", $suffix
            ),
            solver: "bvp_damp",
            groups: &["bvp", "bvp-damp", "bvp-damp-matrix", $family],
            workload: concat!(
                "BVP_Damp ", $family, "; ", $workload,
                "; n_steps=32; 4-point alpha continuation"
            ),
            frontend: $frontend,
            execution: $execution,
            linear_algebra: $matrix,
            parallel_policy: $dispatch,
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "Matched BVP_Damp dashboard route; AOT is restricted to supported Sparse/Banded Auto cases and requires TCC",
            run: run::<
                { $family_id },
                { $workload_id },
                { $matrix_id },
                { $frontend_id },
                { $route_id },
                { $dispatch_id },
            >,
            diagnose: Some(diagnose::<
                { $family_id },
                { $workload_id },
                { $matrix_id },
                { $frontend_id },
                { $route_id },
                { $dispatch_id },
            >),
        }
    };
}

macro_rules! frontend_matrix {
    ($family_id:ident, $family:literal, $workload_id:ident, $workload:literal,
     $frontend_id:ident, $frontend:literal) => {
        [
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_DENSE,
                "dense",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_SEQUENTIAL,
                "Sequential",
                "lambdify_seq"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_DENSE,
                "dense",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_PARALLEL,
                "Parallel(min_work=0)",
                "lambdify_parallel"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_DENSE,
                "dense",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_AUTO,
                "Auto(min_work=0)",
                "lambdify_auto"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_SPARSE,
                "sparse_faer",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_SEQUENTIAL,
                "Sequential",
                "lambdify_seq"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_SPARSE,
                "sparse_faer",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_PARALLEL,
                "Parallel(min_work=0)",
                "lambdify_parallel"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_SPARSE,
                "sparse_faer",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_AUTO,
                "Auto(min_work=0)",
                "lambdify_auto"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_BANDED,
                "banded_faithful",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_SEQUENTIAL,
                "Sequential",
                "lambdify_seq"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_BANDED,
                "banded_faithful",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_PARALLEL,
                "Parallel(min_work=0)",
                "lambdify_parallel"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_BANDED,
                "banded_faithful",
                $frontend_id,
                $frontend,
                ROUTE_LAMBDIFY,
                "Lambdify",
                DISPATCH_AUTO,
                "Auto(min_work=0)",
                "lambdify_auto"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_SPARSE,
                "sparse_faer",
                $frontend_id,
                $frontend,
                ROUTE_AOT,
                "AOT(warm-cache)",
                DISPATCH_AUTO,
                "Auto",
                "aot_auto"
            ),
            scenario!(
                $family_id,
                $family,
                $workload_id,
                $workload,
                MATRIX_BANDED,
                "banded_faithful",
                $frontend_id,
                $frontend,
                ROUTE_AOT,
                "AOT(warm-cache)",
                DISPATCH_AUTO,
                "Auto",
                "aot_auto"
            ),
        ]
    };
}

pub static SCENARIOS: [[crate::registry::Scenario; 11]; 6] = [
    frontend_matrix!(
        FAMILY_DAMPED,
        "damped",
        WORKLOAD_OSCILLATOR,
        "oscillator",
        FRONTEND_EXPR_LEGACY,
        "exprlegacy"
    ),
    frontend_matrix!(
        FAMILY_DAMPED,
        "damped",
        WORKLOAD_OSCILLATOR,
        "oscillator",
        FRONTEND_ATOM_VIEW,
        "atomview"
    ),
    frontend_matrix!(
        FAMILY_DAMPED,
        "damped",
        WORKLOAD_NONLINEAR,
        "nonlinear",
        FRONTEND_EXPR_LEGACY,
        "exprlegacy"
    ),
    frontend_matrix!(
        FAMILY_DAMPED,
        "damped",
        WORKLOAD_NONLINEAR,
        "nonlinear",
        FRONTEND_ATOM_VIEW,
        "atomview"
    ),
    frontend_matrix!(
        FAMILY_FROZEN,
        "frozen",
        WORKLOAD_FROZEN_LINEAR,
        "linear_exact",
        FRONTEND_EXPR_LEGACY,
        "exprlegacy"
    ),
    frontend_matrix!(
        FAMILY_FROZEN,
        "frozen",
        WORKLOAD_FROZEN_LINEAR,
        "linear_exact",
        FRONTEND_ATOM_VIEW,
        "atomview"
    ),
];
