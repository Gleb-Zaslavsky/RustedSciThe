//! Large, established combustion workloads for the two BVP solvers.
//!
//! The BVP_sci rows reuse its six-state combustion story equations on a
//! 1000-node mesh. The BVP_Damp rows use the established chemical
//! combustion-3000 fixture. They
//! have separate equations and are not presented as cross-solver parity rows.

use crate::error::RunnerError;
use crate::registry::{NativeMetricScope, Scenario, ScenarioOutput};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
};
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecutionPolicy, BvpSciMatrixLayout, BvpSciOutputPolicy,
    BvpSciSolverBuilder, BvpSciStatus, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode;
use RustedSciThe::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use RustedSciThe::symbolic::codegen::codegen_provider_api::MatrixBackend;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const NODES: usize = 1_000;
const DAMP_STEPS: usize = 3_000;

#[derive(Clone, Copy)]
enum Frontend {
    ExprLegacy,
    AtomView,
}

impl Frontend {
    fn assembly(self) -> BvpSciAssembly {
        match self {
            Self::ExprLegacy => BvpSciAssembly::ExprLegacy,
            Self::AtomView => BvpSciAssembly::AtomViewNative,
        }
    }

    fn damp_assembly(self) -> BvpSymbolicAssemblyBackend {
        match self {
            Self::ExprLegacy => BvpSymbolicAssemblyBackend::ExprLegacy,
            Self::AtomView => BvpSymbolicAssemblyBackend::AtomView,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomView => "AtomView",
        }
    }
}

#[derive(Clone, Copy)]
enum Layout {
    Sparse,
    Banded,
}

impl Layout {
    fn damp_backend(self) -> MatrixBackend {
        match self {
            Self::Sparse => MatrixBackend::SparseCol,
            Self::Banded => MatrixBackend::Banded,
        }
    }
}

#[derive(Clone, Copy)]
enum Route {
    Lambdify,
    Aot,
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::Lambdify => "Lambdify",
            Self::Aot => "AOT(warm-cache)",
        }
    }
}

#[derive(Clone, Copy)]
struct BvpSciCase {
    frontend: Frontend,
    route: Route,
}

#[derive(Clone, Copy)]
struct BvpDampCase {
    frontend: Frontend,
    layout: Layout,
    route: Route,
}

struct BvpSciRun {
    solution: RustedSciThe::numerical::BVP_sci::new::BvpSciSolution,
    setup_ms: f64,
    solve_ms: f64,
    wall_ms: f64,
    telemetry: RustedSciThe::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
}

fn combustion_equations() -> Vec<Expr> {
    let unknowns = Expr::parse_vector_expression(vec!["Teta", "q", "C0", "J0", "C1", "J1"]);
    let [teta, q, c0, j0, _c1, j1] = unknowns.as_slice() else {
        unreachable!("six combustion state expressions")
    };
    let dt = Expr::Const(600.0);
    let t_scale = Expr::Const(600.0);
    let lambda = Expr::Const(0.07);
    let q_heat = Expr::Const(3000.0 * 1e3 * 0.034);
    let a = Expr::Const(1.3e5);
    let e = Expr::Const(5000.0 * 4.184);
    let m = Expr::Const(34.2 / 1000.0);
    let r_g = Expr::Const(8.314);
    let ro_m = Expr::Const((34.2 / 1000.0) * 2e6 / (8.314 * 1500.0));
    let qm = Expr::Const((3e-4_f64).powi(2) / 600.0);
    let qs = Expr::Const((3e-4_f64).powi(2));
    let pe_q = Expr::Const(0.0090168);
    let ro_d = Expr::Const(2.88e-4);
    let pe_d = Expr::Const(1.50e-3);
    let minus = Expr::Const(-1.0);
    let m_reag = Expr::Const(0.342);
    let rate = a
        * Expr::exp(-e / (r_g * (teta.clone() * t_scale + dt)))
        * c0.clone()
        * (ro_m.clone() / m_reag);

    vec![
        q.clone() / lambda,
        q.clone() * pe_q - q_heat * rate.clone() * qm,
        j0.clone() / ro_d.clone(),
        j0.clone() * pe_d.clone()
            - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone(),
        j1.clone() / ro_d,
        j1.clone() * pe_d - (m.clone() * rate * ro_m / m) * qs,
    ]
}

fn bvp_sci_combustion_equations() -> Vec<Expr> {
    let unknowns = Expr::parse_vector_expression(vec!["Teta", "q", "C0", "J0", "C1", "J1"]);
    let [teta, q, c0, j0, _c1, j1] = unknowns.as_slice() else {
        unreachable!("six-state BVP_sci combustion equations")
    };
    let dt = Expr::Const(600.0);
    let t_scale = Expr::Const(600.0);
    let lambda = Expr::Const(0.07);
    let q_heat = Expr::Const(3000.0 * 1e3 * 0.034);
    let a = Expr::Const(1.3e5);
    let e = Expr::Const(5000.0 * 4.184);
    let m = Expr::Const(34.2 / 1000.0);
    let r_g = Expr::Const(8.314);
    let ro_m = Expr::Const((34.2 / 1000.0) * 2e6 / (8.314 * 1500.0));
    let qm = Expr::Const((3e-4_f64).powi(2) / 600.0);
    let qs = Expr::Const((3e-4_f64).powi(2));
    let pe_q = Expr::Const(0.0090168);
    let ro_d = Expr::Const(2.88e-4);
    let pe_d = Expr::Const(1.50e-3);
    let minus = Expr::Const(-1.0);
    let m_reag = Expr::Const(0.342);
    let rate = a
        * Expr::exp(-e / (r_g * (teta.clone() * t_scale + dt)))
        * c0.clone()
        * (ro_m.clone() / m_reag);

    vec![
        q.clone() / lambda,
        q.clone() * pe_q - q_heat * rate.clone() * qm,
        j0.clone() / ro_d.clone(),
        j0.clone() * pe_d.clone()
            - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone(),
        j1.clone() / ro_d,
        j1.clone() * pe_d - (m.clone() * rate.clone() * ro_m.clone() / m.clone()) * qs.clone(),
    ]
}

fn combustion_boundary_values() -> HashMap<String, Vec<(usize, f64)>> {
    HashMap::from([
        ("Teta".into(), vec![(0, (1000.0_f64 - 600.0) / 600.0)]),
        ("q".into(), vec![(1, 1e-10)]),
        ("C0".into(), vec![(0, 1.0)]),
        ("J0".into(), vec![(1, 1e-7)]),
        ("C1".into(), vec![(0, 1e-3)]),
        ("J1".into(), vec![(1, 1e-7)]),
    ])
}

fn bvp_sci_once(case: BvpSciCase, mode: BvpSciTelemetryMode) -> Result<BvpSciRun, RunnerError> {
    let wall_started = Instant::now();
    let mesh = (0..NODES)
        .map(|index| index as f64 / (NODES - 1) as f64)
        .collect::<Vec<_>>();
    let states = vec!["Teta", "q", "C0", "J0", "C1", "J1"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let initial_state = vec![0.99; NODES * 6];
    let frontend = case.frontend;
    let mut builder = BvpSciSolverBuilder::new(bvp_sci_combustion_equations(), states)
        .with_independent_variable("x")
        .with_mesh_and_initial_state(mesh, initial_state)
        .with_boundary_callback(|left, right, _parameters, residual| {
            residual[0] = left[0];
            residual[1] = right[1];
            residual[2] = left[2] - 1.0;
            residual[3] = right[3];
            residual[4] = left[4] - 1e-3;
            residual[5] = right[5];
            Ok(())
        })
        .with_frontend(frontend.assembly())
        .with_matrix_layout(BvpSciMatrixLayout::Sparse)
        .with_execution_policy(BvpSciExecutionPolicy::Sequential)
        .with_tolerance(1e-6)
        .with_limits(NODES * 2, 40, 4)
        .with_telemetry(mode)
        .with_output_policy(BvpSciOutputPolicy::MeshAndValues);
    if matches!(case.route, Route::Aot) {
        let config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
            PathBuf::from("target/perf-gates-aot/bvp-sci-combustion")
                .join(frontend.label().to_ascii_lowercase()),
        )
        .with_c_tcc();
        builder = builder.with_aot_generated_backend(config);
    }
    let setup_started = Instant::now();
    let mut solver = builder.build().map_err(|error| {
        RunnerError::Scenario(format!("BVP_sci combustion-1000 build: {error:?}"))
    })?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1e3;
    if solver.plan().is_aot() != matches!(case.route, Route::Aot) {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci requested {} but selected a different execution route",
            case.route.label()
        )));
    }
    let solve_started = Instant::now();
    let solution = solver.solve().map_err(|error| {
        RunnerError::Scenario(format!("BVP_sci combustion-1000 solve: {error:?}"))
    })?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
    if solution.status != BvpSciStatus::Success
        || solution.y.len() != solution.x.len() * 6
        || solution.y.iter().any(|value| !value.is_finite())
        || !solution.residual_norm.is_finite()
        || solution.residual_norm > 1e-4
    {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci combustion-1000 invalid result: status={:?}, nodes={}, residual={:e}",
            solution.status,
            solution.x.len(),
            solution.residual_norm
        )));
    }
    let boundary_error = bvp_sci_boundary_error(&solution);
    if boundary_error > 1e-6 {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci combustion-1000 boundary error is too large: {boundary_error:e}"
        )));
    }
    let telemetry = solver.plan().telemetry_snapshot();
    black_box((&solution.x, &solution.y));
    Ok(BvpSciRun {
        solution,
        setup_ms,
        solve_ms,
        wall_ms: wall_started.elapsed().as_secs_f64() * 1e3,
        telemetry,
    })
}

fn bvp_sci_phase(case: BvpSciCase, diagnostic: bool) -> Result<ScenarioOutput, RunnerError> {
    let plain = bvp_sci_once(case, BvpSciTelemetryMode::Off)?;
    if !diagnostic {
        return Ok(ScenarioOutput {
            metrics: vec![
                ("solver_setup_ms".into(), "ms/off".into(), plain.setup_ms),
                ("solve_ms".into(), "ms/off".into(), plain.solve_ms),
                ("scenario_wall_ms".into(), "ms/off".into(), plain.wall_ms),
            ],
            counters: vec![
                ("final_nodes".into(), plain.solution.x.len() as u64),
                ("state_dimension".into(), 6),
                ("newton_iterations".into(), plain.solution.iterations as u64),
            ],
            diagnostics: vec![
                (
                    "residual_norm".into(),
                    "normalized".into(),
                    plain.solution.residual_norm,
                ),
                (
                    "boundary_error".into(),
                    "absolute".into(),
                    bvp_sci_boundary_error(&plain.solution),
                ),
            ],
            ..ScenarioOutput::default()
        });
    }
    let measured = bvp_sci_once(case, BvpSciTelemetryMode::Detailed)?;
    let solution_diff = plain
        .solution
        .y
        .iter()
        .zip(&measured.solution.y)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    if solution_diff > 1e-8 {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci telemetry changed combustion-1000 solution by {solution_diff:e}"
        )));
    }
    let snapshot = measured.telemetry;
    let metrics: Vec<(String, String, f64)> = [
        ("preparation_ms", snapshot.preparation_ms),
        ("rhs_callback_ms", snapshot.rhs_callback_ms),
        ("boundary_callback_ms", snapshot.boundary_callback_ms),
        ("residual_ms", snapshot.residual_evaluation_ms),
        ("jacobian_ms", snapshot.jacobian_evaluation_ms),
        ("linear_assembly_ms", snapshot.linear_assembly_ms),
        ("factorization_ms", snapshot.factorization_ms),
        ("linear_solve_ms", snapshot.linear_solve_ms),
        ("mesh_refinement_ms", snapshot.mesh_refinement_ms),
    ]
    .into_iter()
    .filter_map(|(id, value)| value.map(|ms| (id.into(), "ms/detailed".into(), ms)))
    .collect();
    let native_scopes = snapshot
        .timing_scopes
        .iter()
        .filter_map(|scope| {
            use RustedSciThe::numerical::BVP_sci::new::telemetry::{
                BvpSciTelemetryScopeKind, BvpSciTelemetryStage,
            };
            let metric_id = match scope.stage {
                BvpSciTelemetryStage::Preparation => "preparation_ms",
                BvpSciTelemetryStage::Callback => "rhs_callback_ms",
                BvpSciTelemetryStage::ResidualEvaluation => "residual_ms",
                BvpSciTelemetryStage::JacobianEvaluation => "jacobian_ms",
                BvpSciTelemetryStage::LinearAssembly => "linear_assembly_ms",
                BvpSciTelemetryStage::Factorization => "factorization_ms",
                BvpSciTelemetryStage::LinearSolve => "linear_solve_ms",
                BvpSciTelemetryStage::MeshRefinement => "mesh_refinement_ms",
                _ => return None,
            };
            if !metrics.iter().any(|(id, _, _)| id == metric_id) {
                return None;
            }
            let kind = match scope.kind {
                BvpSciTelemetryScopeKind::Inclusive => "inclusive",
                BvpSciTelemetryScopeKind::Exclusive => "exclusive",
            };
            Some(NativeMetricScope {
                metric_id: metric_id.into(),
                stage: format!("{:?}", scope.stage),
                kind: kind.into(),
                parent: scope.parent.map(|parent| format!("{:?}", parent)),
            })
        })
        .collect();
    Ok(ScenarioOutput {
        metrics,
        native_scopes,
        counters: vec![
            ("final_nodes".into(), measured.solution.x.len() as u64),
            ("state_dimension".into(), 6),
            ("newton_iterations".into(), snapshot.newton_iterations),
            ("residual_evaluations".into(), snapshot.residual_evaluations),
            ("jacobian_evaluations".into(), snapshot.jacobian_evaluations),
            ("factorizations".into(), snapshot.factorizations),
            ("linear_solves".into(), snapshot.linear_solves),
            ("mesh_refinements".into(), snapshot.mesh_refinements),
        ],
        diagnostics: vec![
            (
                "off_vs_detailed_max_diff".into(),
                "absolute".into(),
                solution_diff,
            ),
            (
                "residual_norm".into(),
                "normalized".into(),
                measured.solution.residual_norm,
            ),
            (
                "boundary_error".into(),
                "absolute".into(),
                bvp_sci_boundary_error(&measured.solution),
            ),
        ],
        ..ScenarioOutput::default()
    })
}

fn bvp_sci_boundary_error(solution: &RustedSciThe::numerical::BVP_sci::new::BvpSciSolution) -> f64 {
    let right = (solution.x.len() - 1) * 6;
    [
        solution.y[0].abs(),
        (solution.y[right + 1] - 1e-10).abs(),
        (solution.y[2] - 1.0).abs(),
        solution.y[right + 3].abs(),
        (solution.y[4] - 1e-3).abs(),
        solution.y[right + 5].abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max)
}

fn bvp_sci_run<const FRONTEND: u8, const ROUTE: u8>() -> Result<ScenarioOutput, RunnerError> {
    bvp_sci_phase(
        BvpSciCase {
            frontend: if FRONTEND == 0 {
                Frontend::ExprLegacy
            } else {
                Frontend::AtomView
            },
            route: if ROUTE == 0 {
                Route::Lambdify
            } else {
                Route::Aot
            },
        },
        false,
    )
}

fn bvp_sci_diagnose<const FRONTEND: u8, const ROUTE: u8>(
    _: &Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    bvp_sci_phase(
        BvpSciCase {
            frontend: if FRONTEND == 0 {
                Frontend::ExprLegacy
            } else {
                Frontend::AtomView
            },
            route: if ROUTE == 0 {
                Route::Lambdify
            } else {
                Route::Aot
            },
        },
        true,
    )
}

// This is the established BVP_Damp combustion chemistry fixture, not the
// distinct combustion-like workload used by the BVP_sci cross-solver story.
fn damp_equations() -> Vec<Expr> {
    combustion_equations()
}

fn damp_options(case: BvpDampCase, mode: BvpTelemetryMode) -> DampedSolverOptions {
    let names = ["Teta", "q", "C0", "J0", "C1", "J1"];
    let bounds = HashMap::from([
        ("Teta".into(), (0.0, 10.0)),
        ("q".into(), (-1e20, 1e20)),
        ("C0".into(), (0.0, 1.5)),
        ("J0".into(), (-1e2, 1e2)),
        ("C1".into(), (0.0, 1.5)),
        ("J1".into(), (-1e2, 1e2)),
    ]);
    let rel_tolerance = names
        .iter()
        .map(|name| ((*name).to_string(), 1e-5))
        .collect();
    let mut generated = match (case.layout, case.route) {
        (Layout::Sparse, Route::Aot) => GeneratedBackendConfig::sparse_build_if_missing_release(),
        (Layout::Banded, Route::Aot) => GeneratedBackendConfig::banded_build_if_missing_release(),
        (Layout::Sparse, Route::Lambdify) => GeneratedBackendConfig::sparse_defaults(),
        (Layout::Banded, Route::Lambdify) => GeneratedBackendConfig::banded_lambdify_defaults(),
    }
    .with_matrix_backend_override(case.layout.damp_backend())
    .with_symbolic_assembly_backend(case.frontend.damp_assembly())
    .with_lambdify_telemetry_mode(if mode == BvpTelemetryMode::Detailed {
        BvpLambdifyTelemetryMode::Detailed
    } else {
        BvpLambdifyTelemetryMode::Off
    })
    .with_bvp_telemetry_mode(mode);
    if matches!(case.route, Route::Lambdify) {
        generated =
            generated.with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
    } else {
        generated = generated
            .with_backend_policy_override(Some(BackendSelectionPolicy::AotOnly))
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            });
    }
    let mut options = match case.layout {
        Layout::Sparse => DampedSolverOptions::sparse_damped(),
        Layout::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
    }
    .with_generated_backend_config(generated)
    .with_strategy_params(Some(SolverParams {
        max_jac: Some(6),
        max_damp_iter: Some(6),
        damp_factor: Some(0.5),
        adaptive: None,
    }))
    .with_abs_tolerance(1e-6)
    .with_rel_tolerance(rel_tolerance)
    .with_max_iterations(100)
    .with_bounds(bounds)
    .with_loglevel(Some("error".into()));
    if mode == BvpTelemetryMode::Off {
        options = options.with_bvp_telemetry_mode(BvpTelemetryMode::Off);
    }
    options
}

struct BvpDampRun {
    solution: DMatrix<f64>,
    wall_ms: f64,
    prepare_ms: f64,
    solve_ms: f64,
    telemetry: RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot,
}

fn damp_boundary_error(solution: &DMatrix<f64>) -> f64 {
    let right = solution.nrows() - 1;
    [
        solution[(0, 0)] - (1000.0_f64 - 600.0) / 600.0,
        solution[(right, 1)] - 1e-10,
        solution[(0, 2)] - 1.0,
        solution[(right, 3)] - 1e-7,
        solution[(0, 4)] - 1e-3,
        solution[(right, 5)] - 1e-7,
    ]
    .into_iter()
    .map(f64::abs)
    .fold(0.0_f64, f64::max)
}

fn bvp_damp_once(case: BvpDampCase, mode: BvpTelemetryMode) -> Result<BvpDampRun, RunnerError> {
    let wall_started = Instant::now();
    let names = ["Teta", "q", "C0", "J0", "C1", "J1"];
    let initial_guess = DMatrix::from_element(names.len(), DAMP_STEPS, 0.99);
    let mut solver = NRBVP::new_with_options(
        damp_equations(),
        initial_guess,
        names.iter().map(|name| (*name).to_string()).collect(),
        "x".into(),
        combustion_boundary_values(),
        0.0,
        1.0,
        DAMP_STEPS,
        damp_options(case, mode),
    );
    solver.dont_save_log(true);
    let prepare_started = Instant::now();
    solver.try_eq_generate(None, None).map_err(|error| {
        RunnerError::Scenario(format!("BVP_Damp combustion prepare: {error:?}"))
    })?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
    let solve_started = Instant::now();
    solver
        .try_solver_prepared()
        .map_err(|error| RunnerError::Scenario(format!("BVP_Damp combustion solve: {error:?}")))?
        .ok_or_else(|| RunnerError::Scenario("BVP_Damp combustion returned no solution".into()))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
    let solution = solver
        .get_result()
        .ok_or_else(|| RunnerError::Scenario("BVP_Damp combustion result missing".into()))?;
    if solution.shape() != (DAMP_STEPS + 1, 6) || solution.iter().any(|value| !value.is_finite()) {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp combustion invalid solution shape {:?}",
            solution.shape()
        )));
    }
    let boundary_error = damp_boundary_error(&solution);
    if boundary_error > 1e-5 {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp combustion boundary error is too large: {boundary_error:e}"
        )));
    }
    let telemetry = solver.get_statistics().telemetry;
    if matches!(case.route, Route::Aot)
        && telemetry
            .plan
            .as_ref()
            .and_then(|plan| plan.selected_backend)
            != Some(SelectedBackendKind::AotCompiled)
    {
        return Err(RunnerError::Scenario(
            "BVP_Damp combustion did not select compiled AOT".into(),
        ));
    }
    black_box(&solution);
    Ok(BvpDampRun {
        solution,
        wall_ms: wall_started.elapsed().as_secs_f64() * 1e3,
        prepare_ms,
        solve_ms,
        telemetry,
    })
}

fn bvp_damp_phase(case: BvpDampCase, diagnostic: bool) -> Result<ScenarioOutput, RunnerError> {
    let plain = bvp_damp_once(case, BvpTelemetryMode::Off)?;
    if !diagnostic {
        return Ok(ScenarioOutput {
            metrics: vec![
                ("prepare_ms".into(), "ms/off".into(), plain.prepare_ms),
                ("solve_ms".into(), "ms/off".into(), plain.solve_ms),
                ("scenario_wall_ms".into(), "ms/off".into(), plain.wall_ms),
            ],
            counters: vec![
                ("mesh_nodes".into(), plain.solution.nrows() as u64),
                ("state_dimension".into(), 6),
            ],
            diagnostics: vec![
                (
                    "max_abs_solution".into(),
                    "state".into(),
                    plain
                        .solution
                        .iter()
                        .fold(0.0_f64, |max, value| max.max(value.abs())),
                ),
                (
                    "boundary_error".into(),
                    "absolute".into(),
                    damp_boundary_error(&plain.solution),
                ),
            ],
            ..ScenarioOutput::default()
        });
    }
    let measured = bvp_damp_once(case, BvpTelemetryMode::Detailed)?;
    let solution_diff = plain
        .solution
        .iter()
        .zip(measured.solution.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    if solution_diff > 1e-8 {
        return Err(RunnerError::Scenario(format!(
            "BVP_Damp telemetry changed combustion solution by {solution_diff:e}"
        )));
    }
    let stats = measured.telemetry;
    Ok(ScenarioOutput {
        metrics: vec![
            (
                "stage_residual_ms".into(),
                "ms/detailed".into(),
                stats.timings.residual.as_secs_f64() * 1e3,
            ),
            (
                "stage_jacobian_ms".into(),
                "ms/detailed".into(),
                stats.timings.jacobian.as_secs_f64() * 1e3,
            ),
            (
                "stage_linear_ms".into(),
                "ms/detailed".into(),
                stats.timings.linear_system.as_secs_f64() * 1e3,
            ),
            (
                "stage_factorization_ms".into(),
                "ms/detailed".into(),
                stats.timings.factorization.as_secs_f64() * 1e3,
            ),
            (
                "native_scope_solve_ms".into(),
                "ms/detailed/typed-scope".into(),
                stats.scopes.solve.elapsed.as_secs_f64() * 1e3,
            ),
        ],
        native_scopes: vec![NativeMetricScope {
            metric_id: "native_scope_solve_ms".into(),
            stage: "Solve".into(),
            kind: "typed_aggregate".into(),
            parent: None,
        }],
        counters: vec![
            ("mesh_nodes".into(), measured.solution.nrows() as u64),
            ("state_dimension".into(), 6),
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
            (
                "off_vs_detailed_max_diff".into(),
                "absolute".into(),
                solution_diff,
            ),
            (
                "boundary_error".into(),
                "absolute".into(),
                damp_boundary_error(&measured.solution),
            ),
        ],
    })
}

fn bvp_damp_run<const FRONTEND: u8, const LAYOUT: u8, const ROUTE: u8>()
-> Result<ScenarioOutput, RunnerError> {
    bvp_damp_phase(
        BvpDampCase {
            frontend: if FRONTEND == 0 {
                Frontend::ExprLegacy
            } else {
                Frontend::AtomView
            },
            layout: if LAYOUT == 0 {
                Layout::Sparse
            } else {
                Layout::Banded
            },
            route: if ROUTE == 0 {
                Route::Lambdify
            } else {
                Route::Aot
            },
        },
        false,
    )
}

fn bvp_damp_diagnose<const FRONTEND: u8, const LAYOUT: u8, const ROUTE: u8>(
    _: &Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    bvp_damp_phase(
        BvpDampCase {
            frontend: if FRONTEND == 0 {
                Frontend::ExprLegacy
            } else {
                Frontend::AtomView
            },
            layout: if LAYOUT == 0 {
                Layout::Sparse
            } else {
                Layout::Banded
            },
            route: if ROUTE == 0 {
                Route::Lambdify
            } else {
                Route::Aot
            },
        },
        true,
    )
}

macro_rules! sci_scenario {
    ($id:literal, $frontend_id:literal, $frontend:literal, $route_id:literal, $execution:literal, $run:path) => {
        Scenario {
            id: $id,
            solver: "bvp_sci",
            groups: &["bvp", "bvp-large", "bvp-sci-large"],
            workload: "six-state BVP_sci combustion story system; nodes=1000; sparse collocation",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: "Sparse-faer",
            parallel_policy: "Sequential",
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "Large BVP_sci solve using the established six-state combustion story equations",
            run: $run,
            diagnose: Some(bvp_sci_diagnose::<$frontend_id, $route_id>),
        }
    };
}

macro_rules! damp_scenario {
    ($id:literal, $frontend_id:literal, $frontend:literal, $layout_id:literal, $layout:literal, $route_id:literal, $execution:literal, $run:path) => {
        Scenario {
            id: $id,
            solver: "bvp_damp",
            groups: &["bvp", "bvp-large", "bvp-damp-large"],
            workload: "six-state chemical combustion; n_steps=3000; one full solve",
            frontend: $frontend,
            execution: $execution,
            linear_algebra: $layout,
            parallel_policy: "Sequential",
            telemetry: "off wall gate + Detailed stage diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "Large BVP_Damp combustion workload matched to the established 3000-step story corpus",
            run: $run,
            diagnose: Some(bvp_damp_diagnose::<$frontend_id, $layout_id, $route_id>),
        }
    };
}

pub static SCENARIOS: [Scenario; 10] = [
    sci_scenario!(
        "bvp_sci_combustion1000_expr_lambdify_sparse",
        0,
        "ExprLegacy",
        0,
        "Lambdify",
        bvp_sci_run::<0, 0>
    ),
    sci_scenario!(
        "bvp_sci_combustion1000_atom_lambdify_sparse",
        1,
        "AtomViewNative",
        0,
        "Lambdify",
        bvp_sci_run::<1, 0>
    ),
    sci_scenario!(
        "bvp_sci_combustion1000_expr_aot_sparse",
        0,
        "ExprLegacy",
        1,
        "AOT(warm-cache)",
        bvp_sci_run::<0, 1>
    ),
    sci_scenario!(
        "bvp_sci_combustion1000_atom_aot_sparse",
        1,
        "AtomViewNative",
        1,
        "AOT(warm-cache)",
        bvp_sci_run::<1, 1>
    ),
    damp_scenario!(
        "bvp_damp_combustion3000_expr_lambdify_sparse",
        0,
        "ExprLegacy",
        0,
        "Sparse-faer",
        0,
        "Lambdify",
        bvp_damp_run::<0, 0, 0>
    ),
    damp_scenario!(
        "bvp_damp_combustion3000_atom_lambdify_sparse",
        1,
        "AtomView",
        0,
        "Sparse-faer",
        0,
        "Lambdify",
        bvp_damp_run::<1, 0, 0>
    ),
    damp_scenario!(
        "bvp_damp_combustion3000_atom_lambdify_banded",
        1,
        "AtomView",
        1,
        "Banded-faithful",
        0,
        "Lambdify",
        bvp_damp_run::<1, 1, 0>
    ),
    damp_scenario!(
        "bvp_damp_combustion3000_expr_aot_banded",
        0,
        "ExprLegacy",
        1,
        "Banded-faithful",
        1,
        "AOT(warm-cache)",
        bvp_damp_run::<0, 1, 1>
    ),
    damp_scenario!(
        "bvp_damp_combustion3000_atom_aot_sparse",
        1,
        "AtomView",
        0,
        "Sparse-faer",
        1,
        "AOT(warm-cache)",
        bvp_damp_run::<1, 0, 1>
    ),
    damp_scenario!(
        "bvp_damp_combustion3000_atom_aot_banded",
        1,
        "AtomView",
        1,
        "Banded-faithful",
        1,
        "AOT(warm-cache)",
        bvp_damp_run::<1, 1, 1>
    ),
];

#[cfg(test)]
mod tests {
    use super::{DAMP_STEPS, NODES, SCENARIOS};

    #[test]
    fn large_bvp_gate_set_uses_real_mesh_and_state_dimensions() {
        assert_eq!(SCENARIOS.len(), 10);
        assert!(
            SCENARIOS
                .iter()
                .all(|scenario| scenario.groups.contains(&"bvp-large"))
        );
        assert!(
            SCENARIOS
                .iter()
                .filter(|scenario| scenario.solver == "bvp_sci")
                .all(|scenario| scenario.workload.contains("nodes=1000"))
        );
        assert!(
            SCENARIOS
                .iter()
                .filter(|scenario| scenario.solver == "bvp_damp")
                .all(|scenario| scenario.workload.contains("n_steps=3000"))
        );
    }

    #[test]
    fn large_bvp_debug_routes_report_the_declared_dimensions() {
        for scenario in SCENARIOS
            .iter()
            .filter(|scenario| !scenario.execution.starts_with("AOT"))
        {
            let output =
                (scenario.run)().unwrap_or_else(|error| panic!("{}: {error}", scenario.id));
            let dimension = output
                .counters
                .iter()
                .find(|(id, _)| id == "state_dimension")
                .expect("large BVP scenario reports state dimension")
                .1;
            assert_eq!(dimension, 6, "{}", scenario.id);
            let size_id = if scenario.solver == "bvp_sci" {
                "final_nodes"
            } else {
                "mesh_nodes"
            };
            let size = output
                .counters
                .iter()
                .find(|(id, _)| id == size_id)
                .expect("large BVP scenario reports actual mesh size")
                .1;
            assert!(
                size >= if scenario.solver == "bvp_sci" {
                    NODES as u64
                } else {
                    (DAMP_STEPS + 1) as u64
                },
                "{} reported unexpectedly small mesh size {size}",
                scenario.id
            );
        }
    }
}
