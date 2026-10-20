use crate::error::RunnerError;
use crate::registry::{NativeMetricScope, ScenarioOutput};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecutionPolicy, BvpSciMatrixLayout, BvpSciOutputPolicy, BvpSciSolver,
    BvpSciSolverBuilder, BvpSciStatus, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const NODES: usize = 33;

pub fn run_exponential_dense() -> Result<ScenarioOutput, RunnerError> {
    let full_started = Instant::now();
    let setup_started = Instant::now();
    let mesh = (0..NODES)
        .map(|index| index as f64 / (NODES - 1) as f64)
        .collect::<Vec<_>>();
    let initial_state = mesh.iter().map(|x| (-x).exp()).collect::<Vec<_>>();
    let mut solver = BvpSciSolver::builder(vec![Expr::parse_expression("-y")], vec!["y".into()])
        .with_independent_variable("x")
        .with_mesh_and_initial_state(mesh.clone(), initial_state)
        .with_boundary_callback(|left, _right, _parameters, residual| {
            residual[0] = left[0] - 1.0;
            Ok(())
        })
        .with_tolerance(1.0e-6)
        .with_limits(256, 12, 8)
        .with_output_policy(BvpSciOutputPolicy::MeshAndValues)
        .build()
        .map_err(|error| RunnerError::Scenario(format!("BVP_sci preparation: {error}")))?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let solve_started = Instant::now();
    let solution = solver
        .solve()
        .map_err(|error| RunnerError::Scenario(format!("BVP_sci solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;

    if solution.status != BvpSciStatus::Success
        || solution.x.len() != solution.y.len()
        || solution.y.iter().any(|value| !value.is_finite())
        || !solution.residual_norm.is_finite()
    {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci returned invalid solution: status={:?}, nodes={}, residual_norm={}",
            solution.status,
            solution.x.len(),
            solution.residual_norm
        )));
    }

    let max_exact_error = solution
        .x
        .iter()
        .zip(&solution.y)
        .map(|(x, y)| (y - (-x).exp()).abs())
        .fold(0.0, f64::max);
    let final_state = *solution
        .y
        .last()
        .ok_or_else(|| RunnerError::Scenario("BVP_sci returned no state values".into()))?;
    if max_exact_error > 2.0e-5 || (solution.x[0] - 0.0).abs() > 1e-14 {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci exponential reference check failed: max_error={max_exact_error:e}"
        )));
    }
    black_box((&solution.x, &solution.y));

    Ok(ScenarioOutput {
        metrics: vec![
            ("solver_setup_ms".into(), "ms".into(), setup_ms),
            ("solve_ms".into(), "ms".into(), solve_ms),
            ("scenario_wall_ms".into(), "ms".into(), scenario_wall_ms),
        ],
        native_scopes: Vec::new(),
        counters: vec![
            ("final_nodes".into(), solution.x.len() as u64),
            ("state_dimension".into(), 1),
            ("newton_iterations".into(), solution.iterations as u64),
        ],
        diagnostics: vec![
            ("final_state_at_one".into(), "state".into(), final_state),
            (
                "max_exact_solution_abs_error".into(),
                "absolute".into(),
                max_exact_error,
            ),
            (
                "residual_norm".into(),
                "normalized".into(),
                solution.residual_norm,
            ),
        ],
    })
}

const MATRIX_DENSE: u8 = 0;
const MATRIX_SPARSE: u8 = 1;
const MATRIX_BANDED: u8 = 2;
const FRONTEND_EXPR: u8 = 0;
const FRONTEND_ATOM: u8 = 1;
const ROUTE_LAMBDIFY: u8 = 0;
const ROUTE_AOT: u8 = 1;
const POLICY_SEQUENTIAL: u8 = 0;
const POLICY_PARALLEL: u8 = 1;
const POLICY_AUTO: u8 = 2;
const MATRIX_NODES: usize = 33;

#[derive(Clone, Copy)]
enum Layout {
    Dense,
    Sparse,
    Banded,
}

impl Layout {
    fn from_id(id: u8) -> Self {
        match id {
            MATRIX_DENSE => Self::Dense,
            MATRIX_SPARSE => Self::Sparse,
            MATRIX_BANDED => Self::Banded,
            _ => unreachable!("registered BVP_sci layout id"),
        }
    }

    fn public(self) -> BvpSciMatrixLayout {
        match self {
            Self::Dense => BvpSciMatrixLayout::Dense,
            Self::Sparse => BvpSciMatrixLayout::Sparse,
            // A scalar system has zero lower/upper bandwidth. Larger BVP_sci
            // matrices use the wider block bandwidth covered by solver stories.
            Self::Banded => BvpSciMatrixLayout::Banded { lower: 0, upper: 0 },
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse-faer",
            Self::Banded => "Banded",
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
            FRONTEND_EXPR => Self::ExprLegacy,
            FRONTEND_ATOM => Self::AtomView,
            _ => unreachable!("registered BVP_sci frontend id"),
        }
    }

    fn public(self) -> BvpSciAssembly {
        match self {
            Self::ExprLegacy => BvpSciAssembly::ExprLegacy,
            Self::AtomView => BvpSciAssembly::AtomViewNative,
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
            _ => unreachable!("registered BVP_sci execution route id"),
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
    Auto,
}

impl Dispatch {
    fn from_id(id: u8) -> Self {
        match id {
            POLICY_SEQUENTIAL => Self::Sequential,
            POLICY_PARALLEL => Self::Parallel,
            POLICY_AUTO => Self::Auto,
            _ => unreachable!("registered BVP_sci dispatch id"),
        }
    }

    fn public(self) -> BvpSciExecutionPolicy {
        match self {
            Self::Sequential => BvpSciExecutionPolicy::Sequential,
            Self::Parallel => BvpSciExecutionPolicy::Parallel { min_work: 0 },
            Self::Auto => BvpSciExecutionPolicy::Auto { min_work: 0 },
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sequential => "Sequential",
            Self::Parallel => "Parallel(min_work=0)",
            Self::Auto => "Auto(min_work=0)",
        }
    }
}

struct BvpSciRun {
    solution: RustedSciThe::numerical::BVP_sci::new::BvpSciSolution,
    setup_ms: f64,
    solve_ms: f64,
    scenario_wall_ms: f64,
    telemetry: RustedSciThe::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
}

fn run_matrix_cell_phase<
    const FRONTEND_ID: u8,
    const LAYOUT_ID: u8,
    const ROUTE_ID: u8,
    const POLICY_ID: u8,
>(
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let frontend = Frontend::from_id(FRONTEND_ID);
    let layout = Layout::from_id(LAYOUT_ID);
    let route = Route::from_id(ROUTE_ID);
    let dispatch = Dispatch::from_id(POLICY_ID);
    let plain = run_once(frontend, layout, route, dispatch, BvpSciTelemetryMode::Off)?;
    black_box((&plain.solution.x, &plain.solution.y));
    if !diagnostic_only {
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
                "max_exact_solution_abs_error".into(),
                "absolute".into(),
                exact_error(&plain.solution),
            )],
        });
    }
    let detailed = run_once(
        frontend,
        layout,
        route,
        dispatch,
        BvpSciTelemetryMode::Detailed,
    )?;
    let solution_diff = plain
        .solution
        .y
        .iter()
        .zip(&detailed.solution.y)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    if solution_diff > 1.0e-10 {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci Detailed telemetry changed the solution by {solution_diff:e}"
        )));
    }

    let snapshot = detailed.telemetry;
    let mut metrics = Vec::new();
    for (name, value) in [
        ("preparation_ms", snapshot.preparation_ms),
        ("rhs_callback_ms", snapshot.rhs_callback_ms),
        ("boundary_callback_ms", snapshot.boundary_callback_ms),
        ("residual_ms", snapshot.residual_evaluation_ms),
        ("jacobian_ms", snapshot.jacobian_evaluation_ms),
        ("linear_assembly_ms", snapshot.linear_assembly_ms),
        ("factorization_ms", snapshot.factorization_ms),
        ("linear_solve_ms", snapshot.linear_solve_ms),
        ("mesh_refinement_ms", snapshot.mesh_refinement_ms),
    ] {
        if let Some(value) = value {
            metrics.push((name.into(), "ms/detailed".into(), value));
        }
    }
    let mut diagnostics = vec![
        (
            "max_exact_solution_abs_error".into(),
            "absolute".into(),
            exact_error(&detailed.solution),
        ),
        (
            "off_vs_detailed_max_diff".into(),
            "absolute".into(),
            solution_diff,
        ),
        (
            "residual_norm".into(),
            "normalized".into(),
            detailed.solution.residual_norm,
        ),
    ];
    if let Some(aot) = &snapshot.aot {
        diagnostics.extend([
            (
                "aot_build_attempts".into(),
                "count".into(),
                aot.aot_build_attempts as f64,
            ),
            (
                "aot_link_attempts".into(),
                "count".into(),
                aot.aot_link_attempts as f64,
            ),
            (
                "aot_runtime_ready".into(),
                "count".into(),
                aot.aot_runtime_ready as f64,
            ),
            (
                "aot_parallel_dispatches".into(),
                "count".into(),
                aot.aot_parallel_dispatches as f64,
            ),
        ]);
        // Cold preparation/build/link and warm callback stages are separate,
        // nested scopes. Preserve only observed values; absence is not zero.
        for stage in [
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::AotMaterialization,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::AotBuild,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::AotLink,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::AotCacheLookup,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::AotLowering,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::AotSourceGeneration,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::AotPublication,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::SolverPreparation,
            RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::BridgePreparation,
        ] {
            let timing = aot.cold_stage(stage);
            if timing.recorded {
                metrics.push((
                    format!("aot_cold_{}_ms", stage.label()),
                    "ms/detailed/aot-cold".into(),
                    timing.elapsed.as_secs_f64() * 1_000.0,
                ));
            }
        }
        for stage in [
            RustedSciThe::symbolic::ivp_telemetry::IvpWarmStage::AotChunkDispatch,
            RustedSciThe::symbolic::ivp_telemetry::IvpWarmStage::AotWorkerExecution,
            RustedSciThe::symbolic::ivp_telemetry::IvpWarmStage::AotArgumentCopy,
            RustedSciThe::symbolic::ivp_telemetry::IvpWarmStage::AotOutputWrite,
        ] {
            let timing = aot.warm_stage(stage);
            if timing.recorded {
                metrics.push((
                    format!("aot_warm_{}_ms", stage.label()),
                    "ms/detailed/aot-warm".into(),
                    timing.elapsed.as_secs_f64() * 1_000.0,
                ));
            }
        }
    }

    let native_scopes: Vec<NativeMetricScope> = snapshot
        .timing_scopes
        .iter()
        .filter_map(|scope| {
            let metric_id = match scope.stage {
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::Preparation => "preparation_ms",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::Callback => "rhs_callback_ms",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::ResidualEvaluation => "residual_ms",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::JacobianEvaluation => "jacobian_ms",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::LinearAssembly => "linear_assembly_ms",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::Factorization => "factorization_ms",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::LinearSolve => "linear_solve_ms",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::MeshRefinement => "mesh_refinement_ms",
                _ => return None,
            };
            if !metrics.iter().any(|(id, _, _)| id == metric_id) {
                return None;
            }
            let kind = match scope.kind {
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryScopeKind::Inclusive => "inclusive",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryScopeKind::Exclusive => "exclusive",
            };
            Some(NativeMetricScope {
                metric_id: metric_id.into(),
                stage: format!("{:?}", scope.stage),
                kind: kind.into(),
                parent: scope.parent.map(|parent| format!("{:?}", parent)),
            })
        })
        .collect();
    let callback_scope = snapshot.timing_scopes.iter().find(|scope| {
        scope.stage
            == RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryStage::Callback
    });
    let mut native_scopes = native_scopes;
    if let Some(scope) = callback_scope {
        if metrics
            .iter()
            .any(|(id, _, _)| id == "boundary_callback_ms")
        {
            let kind = match scope.kind {
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryScopeKind::Inclusive => "inclusive",
                RustedSciThe::numerical::BVP_sci::new::telemetry::BvpSciTelemetryScopeKind::Exclusive => "exclusive",
            };
            native_scopes.push(NativeMetricScope {
                metric_id: "boundary_callback_ms".into(),
                stage: format!("{:?}", scope.stage),
                kind: kind.into(),
                parent: scope.parent.map(|parent| format!("{:?}", parent)),
            });
        }
    }

    Ok(ScenarioOutput {
        metrics,
        native_scopes,
        counters: vec![
            ("final_nodes".into(), detailed.solution.x.len() as u64),
            ("newton_iterations".into(), snapshot.newton_iterations),
            ("residual_evaluations".into(), snapshot.residual_evaluations),
            ("jacobian_evaluations".into(), snapshot.jacobian_evaluations),
            ("factorizations".into(), snapshot.factorizations),
            ("linear_solves".into(), snapshot.linear_solves),
            ("mesh_refinements".into(), snapshot.mesh_refinements),
            ("parallel_dispatches".into(), snapshot.parallel_dispatches),
            (
                "sequential_dispatches".into(),
                snapshot.sequential_dispatches,
            ),
            (
                "sparse_symbolic_analyses".into(),
                snapshot.sparse_symbolic_analyses,
            ),
            (
                "sparse_symbolic_reuses".into(),
                snapshot.sparse_symbolic_reuses,
            ),
            (
                "banded_structured_factorizations".into(),
                snapshot.banded_structured_factorizations,
            ),
        ],
        diagnostics,
    })
}

fn run_matrix_cell<
    const FRONTEND_ID: u8,
    const LAYOUT_ID: u8,
    const ROUTE_ID: u8,
    const POLICY_ID: u8,
>() -> Result<ScenarioOutput, RunnerError> {
    run_matrix_cell_phase::<FRONTEND_ID, LAYOUT_ID, ROUTE_ID, POLICY_ID>(false)
}

fn diagnose<const FRONTEND_ID: u8, const LAYOUT_ID: u8, const ROUTE_ID: u8, const POLICY_ID: u8>(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_matrix_cell_phase::<FRONTEND_ID, LAYOUT_ID, ROUTE_ID, POLICY_ID>(true)
}

fn run_once(
    frontend: Frontend,
    layout: Layout,
    route: Route,
    dispatch: Dispatch,
    telemetry: BvpSciTelemetryMode,
) -> Result<BvpSciRun, RunnerError> {
    let full_started = Instant::now();
    let mesh = (0..MATRIX_NODES)
        .map(|index| index as f64 / (MATRIX_NODES - 1) as f64)
        .collect::<Vec<_>>();
    // Start away from the exact exponential so the measured solve includes
    // Newton/Jacobian/factorization work rather than a zero-residual shortcut.
    let initial_state = mesh.iter().map(|x| 1.0 - 0.5 * x).collect::<Vec<_>>();
    let mut builder =
        BvpSciSolverBuilder::new(vec![Expr::parse_expression("-y")], vec!["y".into()])
            .with_independent_variable("x")
            .with_mesh_and_initial_state(mesh, initial_state)
            .with_boundary_callback(|left, _right, _parameters, residual| {
                residual[0] = left[0] - 1.0;
                Ok(())
            })
            .with_frontend(frontend.public())
            .with_matrix_layout(layout.public())
            .with_execution_policy(dispatch.public())
            .with_tolerance(1.0e-6)
            .with_limits(256, 12, 8)
            .with_telemetry(telemetry)
            .with_output_policy(BvpSciOutputPolicy::MeshAndValues);
    if matches!(route, Route::Aot) {
        let scenario_id = format!(
            "{}-{}-{}-{}",
            frontend.label().to_ascii_lowercase(),
            layout.label().to_ascii_lowercase(),
            dispatch
                .label()
                .to_ascii_lowercase()
                .replace(['(', ')', '='], ""),
            "bvp-sci"
        );
        let config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
            PathBuf::from("target/perf-gates-aot/bvp-sci").join(scenario_id),
        )
        .with_c_tcc();
        builder = builder.with_aot_generated_backend(config);
    }

    let setup_started = Instant::now();
    let mut solver = builder
        .build()
        .map_err(|error| RunnerError::Scenario(format!("BVP_sci build: {error:?}")))?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    if solver.plan().is_aot() != matches!(route, Route::Aot) {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci requested {} but selected {}",
            route.label(),
            if solver.plan().is_aot() {
                "AOT"
            } else {
                "Lambdify"
            }
        )));
    }

    let solve_started = Instant::now();
    let solution = solver
        .solve()
        .map_err(|error| RunnerError::Scenario(format!("BVP_sci solve: {error:?}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;
    if solution.status != BvpSciStatus::Success
        || solution.x.len() != solution.y.len()
        || solution.y.iter().any(|value| !value.is_finite())
        || !solution.residual_norm.is_finite()
    {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci invalid solution: status={:?}, nodes={}, residual={}",
            solution.status,
            solution.x.len(),
            solution.residual_norm
        )));
    }
    let error = exact_error(&solution);
    if error > 2.0e-5 || (solution.x[0] - 0.0).abs() > 1.0e-14 {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci y'=-y reference mismatch for {} {}: max_error={error:e}",
            frontend.label(),
            layout.label()
        )));
    }
    let telemetry_snapshot = solver.plan().telemetry_snapshot();
    if matches!(route, Route::Aot) && telemetry_snapshot.aot.is_none() {
        return Err(RunnerError::Scenario(
            "BVP_sci AOT route did not publish AOT lifecycle telemetry".into(),
        ));
    }
    black_box((&solution.x, &solution.y));
    Ok(BvpSciRun {
        solution,
        setup_ms,
        solve_ms,
        scenario_wall_ms,
        telemetry: telemetry_snapshot,
    })
}

fn exact_error(solution: &RustedSciThe::numerical::BVP_sci::new::BvpSciSolution) -> f64 {
    solution
        .x
        .iter()
        .zip(&solution.y)
        .map(|(x, y)| (y - (-x).exp()).abs())
        .fold(0.0, f64::max)
}

macro_rules! scenario {
    ($frontend_id:ident, $frontend:literal, $layout_id:ident, $layout:literal,
     $route_id:ident, $execution:literal, $dispatch_id:ident, $dispatch:literal, $suffix:literal) => {
        crate::registry::Scenario {
            id: concat!("bvp_sci_exp_", $frontend, "_", $layout, "_", $suffix),
            solver: "bvp_sci",
            groups: &["bvp", "bvp-sci", "bvp-sci-matrix"],
            workload: "y'=-y, y(0)=1; nodes=33",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: $layout,
            parallel_policy: match $dispatch_id {
                POLICY_SEQUENTIAL => "Sequential",
                POLICY_PARALLEL => "Parallel(min_work=0)",
                POLICY_AUTO => "Auto(min_work=0)",
                _ => "unsupported",
            },
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "Matched BVP_sci exponential full solve; StandardFaer Sparse only, SuperLU remains separately opt-in",
            run: run_matrix_cell::<
                { $frontend_id },
                { $layout_id },
                { $route_id },
                { $dispatch_id },
            >,
            diagnose: Some(diagnose::<
                { $frontend_id },
                { $layout_id },
                { $route_id },
                { $dispatch_id },
            >),
        }
    };
}

macro_rules! layout_matrix {
    ($frontend_id:ident, $frontend:literal) => {
        [
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_DENSE,
                "Dense",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_SEQUENTIAL,
                "Sequential",
                "lambdify_seq"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_DENSE,
                "Dense",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_PARALLEL,
                "Parallel",
                "lambdify_parallel"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_DENSE,
                "Dense",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_AUTO,
                "Auto",
                "lambdify_auto"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_SPARSE,
                "Sparse",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_SEQUENTIAL,
                "Sequential",
                "lambdify_seq"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_SPARSE,
                "Sparse",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_PARALLEL,
                "Parallel",
                "lambdify_parallel"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_SPARSE,
                "Sparse",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_AUTO,
                "Auto",
                "lambdify_auto"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_BANDED,
                "Banded",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_SEQUENTIAL,
                "Sequential",
                "lambdify_seq"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_BANDED,
                "Banded",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_PARALLEL,
                "Parallel",
                "lambdify_parallel"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_BANDED,
                "Banded",
                ROUTE_LAMBDIFY,
                "Lambdify",
                POLICY_AUTO,
                "Auto",
                "lambdify_auto"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_DENSE,
                "Dense",
                ROUTE_AOT,
                "AOT(warm-cache)",
                POLICY_SEQUENTIAL,
                "Sequential",
                "aot_seq"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_SPARSE,
                "Sparse",
                ROUTE_AOT,
                "AOT(warm-cache)",
                POLICY_SEQUENTIAL,
                "Sequential",
                "aot_seq"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_BANDED,
                "Banded",
                ROUTE_AOT,
                "AOT(warm-cache)",
                POLICY_SEQUENTIAL,
                "Sequential",
                "aot_seq"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_DENSE,
                "Dense",
                ROUTE_AOT,
                "AOT(warm-cache)",
                POLICY_PARALLEL,
                "Parallel",
                "aot_parallel"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_SPARSE,
                "Sparse",
                ROUTE_AOT,
                "AOT(warm-cache)",
                POLICY_PARALLEL,
                "Parallel",
                "aot_parallel"
            ),
            scenario!(
                $frontend_id,
                $frontend,
                MATRIX_BANDED,
                "Banded",
                ROUTE_AOT,
                "AOT(warm-cache)",
                POLICY_PARALLEL,
                "Parallel",
                "aot_parallel"
            ),
        ]
    };
}

pub static MATRIX_SCENARIOS: [[crate::registry::Scenario; 15]; 2] = [
    layout_matrix!(FRONTEND_EXPR, "ExprLegacy"),
    layout_matrix!(FRONTEND_ATOM, "AtomViewNative"),
];
