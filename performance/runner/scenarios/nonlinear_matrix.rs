//! Matched dense symbolic Newton routes for the first-tier nonlinear solver.
//!
//! The matrix holds equations, size, initial point, tolerance and method fixed.
//! It exercises the two symbolic frontends, Lambdify Sequential/Parallel, and
//! the supported dense AOT route. AOT has no user-visible callback dispatch
//! policy, so it is not given artificial sequential/parallel variants.

use crate::error::RunnerError;
use crate::registry::{Scenario, ScenarioOutput};
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    DiagnosticsOptions, LambdifyExecutionPolicy, NewtonMethod, NonlinearSolverMethod,
    PreparationStage, PreparationTelemetryMode, PreparedSymbolicNonlinearProblem, SolveOptions,
    SymbolicAotBuildPolicy, SymbolicBackendConfig, SymbolicBackendKind,
    SymbolicGeneratedBackendConfig, SymbolicLambdifyFrontend, SymbolicNonlinearProblem,
    SymbolicProblemOptions, TerminationReason,
};
use RustedSciThe::numerical::SolverTelemetryMode;
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use nalgebra::DVector;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const DIMENSION: usize = 64;
const FRONTEND_EXPR: u8 = 0;
const FRONTEND_ATOM: u8 = 1;
const ROUTE_LAMBDIFY: u8 = 0;
const ROUTE_AOT: u8 = 1;
const POLICY_SEQUENTIAL: u8 = 0;
const POLICY_PARALLEL: u8 = 1;

#[derive(Clone, Copy)]
enum Frontend {
    ExprLegacy,
    AtomViewNative,
}

impl Frontend {
    fn from_id(id: u8) -> Self {
        match id {
            FRONTEND_EXPR => Self::ExprLegacy,
            FRONTEND_ATOM => Self::AtomViewNative,
            _ => unreachable!("registered nonlinear frontend id"),
        }
    }

    fn public(self) -> SymbolicLambdifyFrontend {
        match self {
            Self::ExprLegacy => SymbolicLambdifyFrontend::ExprLegacy,
            Self::AtomViewNative => SymbolicLambdifyFrontend::AtomViewNative,
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::ExprLegacy => "expr",
            Self::AtomViewNative => "atom",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    Lambdify,
    Aot,
}

impl Route {
    fn from_id(id: u8) -> Self {
        match id {
            ROUTE_LAMBDIFY => Self::Lambdify,
            ROUTE_AOT => Self::Aot,
            _ => unreachable!("registered nonlinear route id"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Lambdify => "Lambdify",
            Self::Aot => "AOT(warm-cache)",
        }
    }
}

#[derive(Clone, Copy)]
enum Dispatch {
    Sequential,
    Parallel,
}

impl Dispatch {
    fn from_id(id: u8) -> Self {
        match id {
            POLICY_SEQUENTIAL => Self::Sequential,
            POLICY_PARALLEL => Self::Parallel,
            _ => unreachable!("registered nonlinear dispatch id"),
        }
    }

    fn public(self) -> LambdifyExecutionPolicy {
        match self {
            Self::Sequential => LambdifyExecutionPolicy::Sequential,
            Self::Parallel => LambdifyExecutionPolicy::Parallel { min_work: 1 },
        }
    }
}

struct NonlinearRun {
    solution: DVector<f64>,
    residual_norm: f64,
    max_solution_error: f64,
    iterations: u64,
    residual_evaluations: u64,
    jacobian_evaluations: u64,
    linear_factorizations: u64,
    linear_solves: u64,
    preparation_ms: f64,
    binding_ms: f64,
    solve_ms: f64,
    scenario_wall_ms: f64,
    diagnostics: Vec<(String, String, f64)>,
}

fn run_cell_phase<const FRONTEND_ID: u8, const ROUTE_ID: u8, const POLICY_ID: u8>(
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let frontend = Frontend::from_id(FRONTEND_ID);
    let route = Route::from_id(ROUTE_ID);
    let dispatch = Dispatch::from_id(POLICY_ID);
    let off = run_once(frontend, route, dispatch, SolverTelemetryMode::Off)?;
    black_box(&off.solution);
    if !diagnostic_only {
        return Ok(ScenarioOutput {
            metrics: vec![
                (
                    "solver_setup_ms".into(),
                    "ms/off".into(),
                    off.preparation_ms,
                ),
                (
                    "parameter_binding_ms".into(),
                    "ms/off".into(),
                    off.binding_ms,
                ),
                ("solve_ms".into(), "ms/off".into(), off.solve_ms),
                (
                    "scenario_wall_ms".into(),
                    "ms/off".into(),
                    off.scenario_wall_ms,
                ),
            ],
            native_scopes: Vec::new(),
            counters: Vec::new(),
            diagnostics: vec![
                ("residual_norm".into(), "absolute".into(), off.residual_norm),
                (
                    "max_solution_error".into(),
                    "absolute".into(),
                    off.max_solution_error,
                ),
            ],
        });
    }
    let detailed = run_once(frontend, route, dispatch, SolverTelemetryMode::Detailed)?;

    let max_telemetry_solution_diff = off
        .solution
        .iter()
        .zip(detailed.solution.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    if max_telemetry_solution_diff > 1.0e-12 {
        return Err(RunnerError::Scenario(format!(
            "nonlinear telemetry changed the solution by {max_telemetry_solution_diff:e}"
        )));
    }

    let mut metrics = Vec::new();
    metrics.extend(detailed.diagnostics);
    Ok(ScenarioOutput {
        metrics,
        native_scopes: Vec::new(),
        counters: vec![
            ("state_dimension".into(), DIMENSION as u64),
            ("newton_iterations".into(), detailed.iterations),
            ("residual_evaluations".into(), detailed.residual_evaluations),
            ("jacobian_evaluations".into(), detailed.jacobian_evaluations),
            (
                "linear_factorizations".into(),
                detailed.linear_factorizations,
            ),
            ("linear_solves".into(), detailed.linear_solves),
        ],
        diagnostics: vec![
            (
                "off_vs_detailed_max_solution_diff".into(),
                "absolute".into(),
                max_telemetry_solution_diff,
            ),
            (
                "detailed_residual_norm".into(),
                "absolute".into(),
                detailed.residual_norm,
            ),
            (
                "detailed_max_solution_error".into(),
                "absolute".into(),
                detailed.max_solution_error,
            ),
        ],
    })
}

fn run_cell<const FRONTEND_ID: u8, const ROUTE_ID: u8, const POLICY_ID: u8>()
-> Result<ScenarioOutput, RunnerError> {
    run_cell_phase::<FRONTEND_ID, ROUTE_ID, POLICY_ID>(false)
}

fn diagnose<const FRONTEND_ID: u8, const ROUTE_ID: u8, const POLICY_ID: u8>(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_cell_phase::<FRONTEND_ID, ROUTE_ID, POLICY_ID>(true)
}

fn run_once(
    frontend: Frontend,
    route: Route,
    dispatch: Dispatch,
    telemetry: SolverTelemetryMode,
) -> Result<NonlinearRun, RunnerError> {
    let target = DVector::from_iterator(
        DIMENSION,
        (0..DIMENSION).map(|index| 1.0 + index as f64 * 0.01),
    );
    let initial = &target * 0.9;
    let variables = (0..DIMENSION)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let equations = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            let mut equation = format!("p0*({variable}^2-{:.17})", target[index].powi(2));
            if index > 0 {
                equation.push_str(&format!("+0.02*(x{}-{:.17})", index - 1, target[index - 1]));
            }
            if index + 1 < DIMENSION {
                equation.push_str(&format!("+0.02*(x{}-{:.17})", index + 1, target[index + 1]));
            }
            equation
        })
        .collect::<Vec<_>>();
    let full_started = Instant::now();

    let preparation_telemetry = if telemetry == SolverTelemetryMode::Detailed {
        PreparationTelemetryMode::Collect
    } else {
        PreparationTelemetryMode::Disabled
    };
    let problem_options = SymbolicProblemOptions::new()
        .with_variables(variables)
        .with_equation_parameters(vec!["p0".into()])
        .with_equation_parameter_values(DVector::from_vec(vec![1.0]))
        .with_lambdify_frontend(frontend.public())
        .with_lambdify_execution_policy(dispatch.public())
        .with_preparation_telemetry(preparation_telemetry);

    let setup_started = Instant::now();
    let problem = match route {
        Route::Lambdify => {
            PreparedSymbolicNonlinearProblem::from_strings(equations, problem_options)
                .map_err(|error| RunnerError::Scenario(format!("nonlinear preparation: {error}")))?
        }
        Route::Aot => {
            let profile = AotBuildProfile::Release;
            // Keep both telemetry passes on the same runner-owned cache. A
            // forced rebuild in the Detailed pass would try to replace a DLL
            // that the preceding Off pass has loaded on Windows.
            let build_policy = SymbolicAotBuildPolicy::BuildIfMissing { profile };
            let output_dir = PathBuf::from("target/perf-gates-aot/nonlinear").join(frontend.key());
            let generated = SymbolicNonlinearProblem::from_strings_with_generated_backend(
                equations,
                problem_options.with_backend_config(SymbolicBackendConfig::aot()),
                SymbolicGeneratedBackendConfig::defaults()
                    .with_build_policy(build_policy)
                    .with_output_parent_dir(Some(output_dir)),
            )
            .map_err(|error| {
                RunnerError::Scenario(format!("nonlinear AOT preparation: {error}"))
            })?;
            generated.into_prepared()
        }
    };
    let preparation_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    if problem.preparation_report().effective_backend
        != match route {
            Route::Lambdify => SymbolicBackendKind::Lambdify,
            Route::Aot => SymbolicBackendKind::Aot,
        }
    {
        return Err(RunnerError::Scenario(format!(
            "nonlinear requested {} but selected {:?}",
            route.label(),
            problem.preparation_report().effective_backend
        )));
    }

    let binding_started = Instant::now();
    let bound = problem
        .bind_values(DVector::from_vec(vec![1.0]))
        .map_err(|error| RunnerError::Scenario(format!("nonlinear parameter binding: {error}")))?;
    let binding_ms = binding_started.elapsed().as_secs_f64() * 1_000.0;

    let solve_options = SolveOptions {
        tolerance: 1.0e-10,
        max_iterations: 80,
        diagnostics: DiagnosticsOptions {
            telemetry_mode: telemetry,
            collect_history: false,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    };
    let solve_started = Instant::now();
    let result = NonlinearSolverMethod::Newton(NewtonMethod)
        .solve(&bound, initial, solve_options)
        .map_err(|error| RunnerError::Scenario(format!("nonlinear Newton solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;

    let max_solution_error = result
        .x
        .iter()
        .zip(target.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if result.termination != TerminationReason::Converged
        || !result.residual_norm.is_finite()
        || result.residual_norm > 1.0e-8
        || max_solution_error > 1.0e-6
    {
        return Err(RunnerError::Scenario(format!(
            "nonlinear quadratic-chain failed: termination={:?}, residual={:e}, max_error={max_solution_error:e}",
            result.termination, result.residual_norm
        )));
    }

    let mut diagnostic_metrics = Vec::new();
    if telemetry == SolverTelemetryMode::Detailed {
        if !result.statistics.timings_collected {
            return Err(RunnerError::Scenario(
                "nonlinear Detailed solve did not report timing availability".into(),
            ));
        }
        let report = problem.preparation_report();
        diagnostic_metrics.push((
            "preparation_total_ms".into(),
            "ms/detailed/preparation".into(),
            report.preparation_duration.as_secs_f64() * 1_000.0,
        ));
        for stage in PreparationStage::ALL.iter().copied() {
            if let Some(timing) = report
                .detailed
                .as_ref()
                .and_then(|detailed| detailed.stage(stage))
                .and_then(|timing| timing.wall_time)
            {
                diagnostic_metrics.push((
                    format!("prepare_stage_{}_ms", preparation_stage_key(stage)),
                    "ms/detailed/preparation-stage".into(),
                    timing.as_secs_f64() * 1_000.0,
                ));
            }
        }
        if let Some(build_duration) = report.build_duration {
            diagnostic_metrics.push((
                "aot_build_ms".into(),
                "ms/detailed/aot-build".into(),
                build_duration.as_secs_f64() * 1_000.0,
            ));
        }
        for (id, duration) in [
            ("residual_ms", result.statistics.residual_duration),
            ("jacobian_ms", result.statistics.jacobian_duration),
            ("linear_solve_ms", result.statistics.linear_solve_duration),
            (
                "linear_factorization_ms",
                result.statistics.linear_factorization_duration,
            ),
            (
                "linear_system_solve_ms",
                result.statistics.linear_system_solve_duration,
            ),
            ("solver_native_total_ms", result.statistics.total_duration),
        ] {
            diagnostic_metrics.push((
                id.into(),
                "ms/detailed/solve".into(),
                duration.as_secs_f64() * 1_000.0,
            ));
        }
        if route == Route::Aot && report.artifact_key.is_none() {
            return Err(RunnerError::Scenario(
                "nonlinear AOT route selected without publishing an artifact identity".into(),
            ));
        }
    }

    black_box((&result.x, &result.residual));
    Ok(NonlinearRun {
        solution: result.x,
        residual_norm: result.residual_norm,
        max_solution_error,
        iterations: result.statistics.iterations as u64,
        residual_evaluations: result.statistics.residual_evaluations as u64,
        jacobian_evaluations: result.statistics.jacobian_evaluations as u64,
        linear_factorizations: result.statistics.linear_factorizations as u64,
        linear_solves: result.statistics.linear_solves as u64,
        preparation_ms,
        binding_ms,
        solve_ms,
        scenario_wall_ms,
        diagnostics: diagnostic_metrics,
    })
}

fn preparation_stage_key(stage: PreparationStage) -> &'static str {
    match stage {
        PreparationStage::InputValidation => "input_validation",
        PreparationStage::ExpressionParsing => "expression_parsing",
        PreparationStage::ExpressionGraphMaterialization => "expression_graph_materialization",
        PreparationStage::AtomConversion => "atom_conversion",
        PreparationStage::AtomDependencyAnalysis => "atom_dependency_analysis",
        PreparationStage::AtomDifferentiation => "atom_differentiation",
        PreparationStage::JacobianDifferentiation => "jacobian_differentiation",
        PreparationStage::ResidualCallbackPreparation => "residual_callback_preparation",
        PreparationStage::JacobianCallbackPreparation => "jacobian_callback_preparation",
        PreparationStage::ParameterBinding => "parameter_binding",
        PreparationStage::PreparedProblemAssembly => "prepared_problem_assembly",
        PreparationStage::AotMaterialization => "aot_materialization",
        PreparationStage::FinalValidation => "final_validation",
    }
}

macro_rules! nonlinear_scenario {
    ($frontend_id:ident, $frontend:literal, $frontend_key:literal,
     $route_id:ident, $execution:literal, $route_key:literal,
     $policy_id:ident, $policy:literal, $suffix:literal) => {
        Scenario {
            id: concat!("nonlinear_quad64_", $frontend_key, "_", $suffix),
            solver: "nonlinear-systems",
            groups: &["nonlinear", "nonlinear-matrix", "symbolic", "dense"],
            workload: "quadratic-chain, dimension=64, known root, parameter p0=1",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: "Dense (fixed nalgebra implementation)",
            parallel_policy: $policy,
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: concat!(
                "Matched dense Newton ",
                $route_key,
                " route; no selectable sparse/banded matrix API"
            ),
            run: run_cell::<{ $frontend_id }, { $route_id }, { $policy_id }>,
            diagnose: Some(diagnose::<{ $frontend_id }, { $route_id }, { $policy_id }>),
        }
    };
}

pub static MATRIX_SCENARIOS: [Scenario; 6] = [
    nonlinear_scenario!(
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        ROUTE_LAMBDIFY,
        "Lambdify",
        "lambdify",
        POLICY_SEQUENTIAL,
        "Sequential",
        "lambdify_seq"
    ),
    nonlinear_scenario!(
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        ROUTE_LAMBDIFY,
        "Lambdify",
        "lambdify",
        POLICY_PARALLEL,
        "Parallel(min_work=1)",
        "lambdify_parallel"
    ),
    nonlinear_scenario!(
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        ROUTE_LAMBDIFY,
        "Lambdify",
        "lambdify",
        POLICY_SEQUENTIAL,
        "Sequential",
        "lambdify_seq"
    ),
    nonlinear_scenario!(
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        ROUTE_LAMBDIFY,
        "Lambdify",
        "lambdify",
        POLICY_PARALLEL,
        "Parallel(min_work=1)",
        "lambdify_parallel"
    ),
    nonlinear_scenario!(
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        ROUTE_AOT,
        "AOT(warm-cache)",
        "aot",
        POLICY_SEQUENTIAL,
        "not-applicable (AOT has no callback dispatch policy)",
        "aot"
    ),
    nonlinear_scenario!(
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        ROUTE_AOT,
        "AOT(warm-cache)",
        "aot",
        POLICY_SEQUENTIAL,
        "not-applicable (AOT has no callback dispatch policy)",
        "aot"
    ),
];
