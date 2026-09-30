//! Lambdify-only stress stories for the LSODE2 symbolic runtime.
//!
//! These tests deliberately exclude AOT. They establish the workload needed
//! before changing the evaluator architecture: a parameterized tridiagonal
//! system at several sizes, both symbolic frontends, both production matrix
//! routes, and explicit versus automatic linear-backend selection.
//!
//! The story suite keeps symbolic frontend selection, linear backend selection,
//! and Lambdify evaluator execution policy as separate axes. This prevents an
//! `Auto` linear-solver row from being mistaken for an `Auto` callback row.

use super::{
    Lsode2ControllerConfig, Lsode2LinearSolverChoice, Lsode2LinearSolverPolicy,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use crate::symbolic::ivp_telemetry::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetrySnapshot, IvpWarmStage,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use nalgebra::DVector;
use std::hint::black_box;
use std::time::{Duration, Instant};

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

#[derive(Clone, Copy)]
enum MatrixRoute {
    Sparse,
    Banded,
}

impl MatrixRoute {
    fn label(self) -> &'static str {
        match self {
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
    }
}

#[derive(Clone, Copy)]
enum Frontend {
    ExprLegacy,
    AtomViewNative,
}

impl Frontend {
    fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomViewNative => "AtomViewNative",
        }
    }

    fn assembly(self) -> Lsode2SymbolicAssemblyBackend {
        match self {
            Self::ExprLegacy => Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Self::AtomViewNative => Lsode2SymbolicAssemblyBackend::AtomView,
        }
    }
}

#[derive(Clone, Copy)]
enum EvaluatorPolicy {
    Sequential,
    Parallel { min_work: usize },
    Auto { min_work: usize },
}

impl EvaluatorPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::Sequential => "Sequential",
            Self::Parallel { .. } => "Parallel",
            Self::Auto { .. } => "Auto",
        }
    }

    fn runtime(self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::Sequential => IvpLambdifyExecutionPolicy::Sequential,
            Self::Parallel { min_work } => IvpLambdifyExecutionPolicy::Parallel { min_work },
            Self::Auto { min_work } => IvpLambdifyExecutionPolicy::Auto { min_work },
        }
    }
}

#[derive(Clone, Copy)]
enum LinearPolicy {
    Auto,
    Force,
}

impl LinearPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Force => "Force",
        }
    }

    fn apply(self, matrix: MatrixRoute, config: Lsode2ProblemConfig) -> Lsode2ProblemConfig {
        match self {
            Self::Auto => config.with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto),
            Self::Force => {
                let choice = match matrix {
                    MatrixRoute::Sparse => Lsode2LinearSolverChoice::FaerSparseLu,
                    MatrixRoute::Banded => Lsode2LinearSolverChoice::LapackFaithfulBandedLu,
                };
                config.with_linear_solver_policy(Lsode2LinearSolverPolicy::Force(choice))
            }
        }
    }
}

struct StressRun {
    dimension: usize,
    frontend: &'static str,
    matrix: &'static str,
    policy: &'static str,
    evaluator_policy: &'static str,
    parameter_profile: &'static str,
    total_ms: f64,
    prepare_ms: f64,
    solve_ms: f64,
    final_norm: f64,
    final_state: DVector<f64>,
    telemetry: crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot,
}

struct StageBreakdownRun {
    dimension: usize,
    frontend: &'static str,
    matrix: &'static str,
    prepare_ms: f64,
    solve_ms: f64,
    prepared_telemetry: IvpTelemetrySnapshot,
    solved_telemetry: IvpTelemetrySnapshot,
}

fn chain_equations(dimension: usize) -> Vec<Expr> {
    super::workload_fixtures::diffusion_chain(dimension).equations
}

fn stress_config(
    dimension: usize,
    frontend: Frontend,
    matrix: MatrixRoute,
    policy: LinearPolicy,
    parameter_values: [f64; 4],
    telemetry: IvpTelemetry,
) -> Lsode2ProblemConfig {
    stress_config_with_evaluator_policy(
        dimension,
        frontend,
        matrix,
        policy,
        parameter_values,
        telemetry,
        EvaluatorPolicy::Sequential,
    )
}

fn stress_config_with_evaluator_policy(
    dimension: usize,
    frontend: Frontend,
    matrix: MatrixRoute,
    policy: LinearPolicy,
    parameter_values: [f64; 4],
    telemetry: IvpTelemetry,
    evaluator_policy: EvaluatorPolicy,
) -> Lsode2ProblemConfig {
    let equations = chain_equations(dimension);
    let initial = DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| 0.2 + 0.01 * (index % 11) as f64),
    );
    let mut config = Lsode2ProblemConfig::new(
        equations,
        (0..dimension).map(|index| format!("y{index}")).collect(),
        "t".to_string(),
        0.0,
        initial,
        8.0,
        0.05,
        1.0e-7,
        1.0e-9,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "d".to_string(),
        "q".to_string(),
        "nl".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_column_slice(&parameter_values))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: frontend.assembly(),
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_controller(
        Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
    )
    .with_faithful_bdf_solve(50_000, 50_000)
    .with_lambdify_execution_policy(evaluator_policy.runtime())
    .with_telemetry(telemetry);

    config = match matrix {
        MatrixRoute::Sparse => config.with_native_sparse_faer_backend(),
        MatrixRoute::Banded => config.with_native_banded_faithful_backend(),
    };
    policy.apply(matrix, config)
}

fn run_stress_case(
    dimension: usize,
    frontend: Frontend,
    matrix: MatrixRoute,
    policy: LinearPolicy,
    profile: &'static str,
    parameter_values: [f64; 4],
) -> StressRun {
    run_stress_case_with_evaluator_policy(
        dimension,
        frontend,
        matrix,
        policy,
        profile,
        parameter_values,
        EvaluatorPolicy::Sequential,
    )
}

fn run_stress_case_with_evaluator_policy(
    dimension: usize,
    frontend: Frontend,
    matrix: MatrixRoute,
    policy: LinearPolicy,
    profile: &'static str,
    parameter_values: [f64; 4],
    evaluator_policy: EvaluatorPolicy,
) -> StressRun {
    let telemetry = IvpTelemetry::detailed();
    let config = stress_config_with_evaluator_policy(
        dimension,
        frontend,
        matrix,
        policy,
        parameter_values,
        telemetry,
        evaluator_policy,
    );
    let started_total = Instant::now();
    let mut solver = Lsode2Solver::new(config).expect("Lambdify stress config should build");
    let started_prepare = Instant::now();
    solver
        .prepare()
        .expect("Lambdify stress preparation should succeed");
    let prepare_ms = started_prepare.elapsed().as_secs_f64() * 1_000.0;
    let started_solve = Instant::now();
    let summary = solver
        .solve_with_summary()
        .expect("Lambdify stress solve should succeed");
    let solve_ms = started_solve.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = started_total.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary
        .final_y
        .as_ref()
        .expect("Lambdify stress solve should expose final state");
    let telemetry = solver.telemetry_snapshot();

    StressRun {
        dimension,
        frontend: frontend.label(),
        matrix: matrix.label(),
        policy: policy.label(),
        evaluator_policy: evaluator_policy.label(),
        parameter_profile: profile,
        total_ms,
        prepare_ms,
        solve_ms,
        final_norm: final_y.norm(),
        final_state: final_y.clone(),
        telemetry,
    }
}

fn run_stage_breakdown_case(
    dimension: usize,
    frontend: Frontend,
    matrix: MatrixRoute,
) -> StageBreakdownRun {
    let telemetry = IvpTelemetry::detailed();
    let config = stress_config(
        dimension,
        frontend,
        matrix,
        LinearPolicy::Auto,
        [20.0, 4.0, 0.20, 0.010],
        telemetry,
    );
    let mut solver = Lsode2Solver::new(config).expect("stage breakdown config should build");

    let prepare_started = Instant::now();
    solver
        .prepare()
        .expect("stage breakdown preparation should succeed");
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let prepared_telemetry = solver.telemetry_snapshot();

    let solve_started = Instant::now();
    solver
        .solve_with_summary()
        .expect("stage breakdown solve should succeed");
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let solved_telemetry = solver.telemetry_snapshot();

    StageBreakdownRun {
        dimension,
        frontend: frontend.label(),
        matrix: matrix.label(),
        prepare_ms,
        solve_ms,
        prepared_telemetry,
        solved_telemetry,
    }
}

const COLD_STAGE_BREAKDOWN: &[IvpColdStage] = &[
    IvpColdStage::Validation,
    IvpColdStage::ParameterBinding,
    IvpColdStage::ExprToAtom,
    IvpColdStage::AtomDependencyAnalysis,
    IvpColdStage::NativeJacobianEvaluatorPreparation,
    IvpColdStage::AotInputAbiPreparation,
    IvpColdStage::SymbolicJacobian,
    IvpColdStage::SymbolicDifferentiation,
    IvpColdStage::Simplification,
    IvpColdStage::AtomToExpr,
    IvpColdStage::SparsePattern,
    IvpColdStage::LayoutPlanning,
    IvpColdStage::ResidualCompilation,
    IvpColdStage::ResidualLambdification,
    IvpColdStage::JacobianCompilation,
    IvpColdStage::JacobianLambdification,
];

const WARM_STAGE_BREAKDOWN: &[IvpWarmStage] = &[
    IvpWarmStage::ArgumentBinding,
    IvpWarmStage::ResidualEvaluation,
    IvpWarmStage::ResidualOutputAssembly,
    IvpWarmStage::JacobianEvaluation,
    IvpWarmStage::JacobianOutputAssembly,
    IvpWarmStage::Factorization,
    IvpWarmStage::RhsSolve,
    IvpWarmStage::Controller,
    IvpWarmStage::ControllerStepSetup,
    IvpWarmStage::ControllerPredictor,
    IvpWarmStage::ControllerIteration,
    IvpWarmStage::ControllerOutcome,
    IvpWarmStage::ControllerStopCondition,
    IvpWarmStage::ControllerMethodPolicy,
    IvpWarmStage::ControllerMethodSwitch,
];

fn elapsed_delta(after: Duration, before: Duration) -> Duration {
    if after >= before {
        after - before
    } else {
        Duration::ZERO
    }
}

fn cold_stage_kind(stage: IvpColdStage) -> &'static str {
    match stage {
        IvpColdStage::SymbolicJacobian
        | IvpColdStage::SparsePattern
        | IvpColdStage::ResidualCompilation
        | IvpColdStage::JacobianCompilation => "inclusive",
        _ => "leaf",
    }
}

fn warm_stage_kind(stage: IvpWarmStage) -> &'static str {
    match stage {
        IvpWarmStage::Controller | IvpWarmStage::ControllerIteration => "inclusive",
        _ => "leaf",
    }
}

fn print_stage_breakdown_run(run: &StageBreakdownRun) {
    for stage in COLD_STAGE_BREAKDOWN {
        let prepared = run.prepared_telemetry.cold_stage(*stage);
        let solved = run.solved_telemetry.cold_stage(*stage);
        let solve_delta = elapsed_delta(solved.elapsed, prepared.elapsed);
        let solve_calls = solved.calls.saturating_sub(prepared.calls);
        reportln!(
            "cold | {} | {} | {:>3} | {:<27} | {:<9} | {:>5} | {:>10.6} | {:>5} | {:>10.6}",
            run.frontend,
            run.matrix,
            run.dimension,
            stage.label(),
            cold_stage_kind(*stage),
            prepared.calls,
            prepared.elapsed.as_secs_f64() * 1_000.0,
            solve_calls,
            solve_delta.as_secs_f64() * 1_000.0,
        );
    }

    for stage in WARM_STAGE_BREAKDOWN {
        let prepared = run.prepared_telemetry.warm_stage(*stage);
        let solved = run.solved_telemetry.warm_stage(*stage);
        let solve_delta = elapsed_delta(solved.elapsed, prepared.elapsed);
        let solve_calls = solved.calls.saturating_sub(prepared.calls);
        reportln!(
            "warm | {} | {} | {:>3} | {:<27} | {:<9} | {:>5} | {:>10.6} | {:>5} | {:>10.6}",
            run.frontend,
            run.matrix,
            run.dimension,
            stage.label(),
            warm_stage_kind(*stage),
            prepared.calls,
            prepared.elapsed.as_secs_f64() * 1_000.0,
            solve_calls,
            solve_delta.as_secs_f64() * 1_000.0,
        );
    }
}

#[test]
#[ignore = "release Lambdify stage baseline; compares cold scopes and warm callback scopes"]
fn lsode2_lambdify_frontend_stage_breakdown_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::lambdify_stress_story_tests::lsode2_lambdify_frontend_stage_breakdown_story",
    );
    let dimensions = std::env::var("LSODE2_LAMBDIFY_STAGE_DIMENSIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 3)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| vec![32, 128]);
    let routes = [
        (Frontend::ExprLegacy, MatrixRoute::Sparse),
        (Frontend::AtomViewNative, MatrixRoute::Sparse),
        (Frontend::ExprLegacy, MatrixRoute::Banded),
        (Frontend::AtomViewNative, MatrixRoute::Banded),
    ];
    let mut runs = Vec::with_capacity(dimensions.len() * routes.len());

    reportln!(
        "[LSODE2 Lambdify stage breakdown] build_profile={profile}; dimensions={dimensions:?}; same task per dimension/frontend/matrix; AOT excluded; callback evaluator=Sequential",
        profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        dimensions = dimensions,
    );
    reportln!(
        "[LSODE2 Lambdify stage breakdown] parent scopes are inclusive; prepare and solve deltas are reported separately; stage rows must not be summed"
    );
    reportln!(
        "phase | frontend | matrix | dim | stage | scope | prepare_calls | prepare_ms | solve_delta_calls | solve_delta_ms"
    );
    reportln!("--- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---:");
    for dimension in dimensions {
        for (frontend, matrix) in routes {
            let run = run_stage_breakdown_case(dimension, frontend, matrix);
            print_stage_breakdown_run(&run);
            runs.push(run);
        }
    }

    reportln!("[LSODE2 Lambdify stage breakdown] wall-clock and counters");
    reportln!(
        "frontend | matrix | dim | prepare_ms | solve_ms | residual_calls | jacobian_calls | jacobian_rebuilds | linear_solves | accepted | rejected | solve_cold_delta"
    );
    reportln!("--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---");
    for run in &runs {
        let prepared = &run.prepared_telemetry;
        let solved = &run.solved_telemetry;
        let solve_cold_delta = COLD_STAGE_BREAKDOWN
            .iter()
            .filter_map(|stage| {
                let calls = solved
                    .cold_stage(*stage)
                    .calls
                    .saturating_sub(prepared.cold_stage(*stage).calls);
                (calls > 0).then_some(format!("{}:{}", stage.label(), calls))
            })
            .collect::<Vec<_>>();
        reportln!(
            "{} | {} | {} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {}",
            run.frontend,
            run.matrix,
            run.dimension,
            run.prepare_ms,
            run.solve_ms,
            solved.residual_evaluations,
            solved.jacobian_evaluations,
            solved.jacobian_rebuilds,
            solved.linear_solve_requests,
            solved.accepted_steps,
            solved.rejected_steps,
            if solve_cold_delta.is_empty() {
                "none".to_string()
            } else {
                solve_cold_delta.join(",")
            },
        );
        assert!(solved.residual_evaluations > 0);
        assert!(solved.jacobian_evaluations > 0);
        assert!(solved.linear_solve_requests > 0);
    }
}

fn print_run(run: &StressRun, repetition: usize) {
    let telemetry = &run.telemetry;
    let expr_to_atom = telemetry.cold_stage(IvpColdStage::ExprToAtom);
    let symbolic_jacobian = telemetry.cold_stage(IvpColdStage::SymbolicJacobian);
    let residual_compilation = telemetry.cold_stage(IvpColdStage::ResidualCompilation);
    let jacobian_compilation = telemetry.cold_stage(IvpColdStage::JacobianCompilation);
    let residual = telemetry.warm_stage(IvpWarmStage::ResidualEvaluation);
    let jacobian = telemetry.warm_stage(IvpWarmStage::JacobianEvaluation);
    let factor = telemetry.warm_stage(IvpWarmStage::Factorization);
    let rhs = telemetry.warm_stage(IvpWarmStage::RhsSolve);
    reportln!(
        "{profile} | {dimension:>3} | {frontend:<18} | {matrix:<7} | {policy:<5} | {evaluator_policy:<10} | {repetition:>3} | {total_ms:>8.3} | {prepare_ms:>8.3} | {solve_ms:>8.3} | {residual_ms:>8.3} | {jacobian_ms:>8.3} | {factor_ms:>8.3} | {rhs_ms:>8.3} | {residuals:>8} | {jacobians:>8} | {jac_rebuilds:>8} | {linear_solves:>8} | {accepted:>8} | {rejected:>8} | {final_norm:>8.3e}",
        profile = run.parameter_profile,
        dimension = run.dimension,
        frontend = run.frontend,
        matrix = run.matrix,
        policy = run.policy,
        evaluator_policy = run.evaluator_policy,
        repetition = repetition,
        total_ms = run.total_ms,
        prepare_ms = run.prepare_ms,
        solve_ms = run.solve_ms,
        residual_ms = residual.elapsed.as_secs_f64() * 1_000.0,
        jacobian_ms = jacobian.elapsed.as_secs_f64() * 1_000.0,
        factor_ms = factor.elapsed.as_secs_f64() * 1_000.0,
        rhs_ms = rhs.elapsed.as_secs_f64() * 1_000.0,
        residuals = telemetry.residual_evaluations,
        jacobians = telemetry.jacobian_evaluations,
        jac_rebuilds = telemetry.jacobian_rebuilds,
        linear_solves = telemetry.linear_solve_requests,
        accepted = telemetry.accepted_steps,
        rejected = telemetry.rejected_steps,
        final_norm = run.final_norm,
    );
    reportln!(
        "[telemetry] profile={profile} dim={dimension} frontend={frontend} matrix={matrix} linear_policy={policy} evaluator_policy={evaluator_policy} rep={repetition} cold_expr_to_atom_ms={expr_to_atom_ms:.3} cold_symbolic_jacobian_ms={symbolic_jacobian_ms:.3} cold_residual_compile_ms={residual_compilation_ms:.3} cold_jacobian_compile_ms={jacobian_compilation_ms:.3} residual_calls={residual_calls} jacobian_calls={jacobian_calls} factor_calls={factor_calls} rhs_calls={rhs_calls} argument_binding_calls={argument_binding_calls} residual_output_calls={residual_output_calls} jacobian_output_calls={jacobian_output_calls} scalar_evaluations={scalar_evaluations} conversions={conversions} copies={copies} copied_bytes={copied_bytes} allocated_bytes={allocated_bytes} parallel_dispatches={parallel_dispatches} sequential_dispatches={sequential_dispatches}",
        profile = run.parameter_profile,
        dimension = run.dimension,
        frontend = run.frontend,
        matrix = run.matrix,
        policy = run.policy,
        evaluator_policy = run.evaluator_policy,
        repetition = repetition,
        expr_to_atom_ms = expr_to_atom.elapsed.as_secs_f64() * 1_000.0,
        symbolic_jacobian_ms = symbolic_jacobian.elapsed.as_secs_f64() * 1_000.0,
        residual_compilation_ms = residual_compilation.elapsed.as_secs_f64() * 1_000.0,
        jacobian_compilation_ms = jacobian_compilation.elapsed.as_secs_f64() * 1_000.0,
        residual_calls = residual.calls,
        jacobian_calls = jacobian.calls,
        factor_calls = factor.calls,
        rhs_calls = rhs.calls,
        argument_binding_calls = telemetry.warm_stage(IvpWarmStage::ArgumentBinding).calls,
        residual_output_calls = telemetry
            .warm_stage(IvpWarmStage::ResidualOutputAssembly)
            .calls,
        jacobian_output_calls = telemetry
            .warm_stage(IvpWarmStage::JacobianOutputAssembly)
            .calls,
        scalar_evaluations = telemetry.scalar_evaluations,
        conversions = telemetry.conversions,
        copies = telemetry.copies,
        copied_bytes = telemetry.copied_bytes,
        allocated_bytes = telemetry.allocated_bytes,
        parallel_dispatches = telemetry.parallel_dispatches,
        sequential_dispatches = telemetry.sequential_dispatches,
    );
}

#[test]
#[ignore = "release Lambdify-only stress baseline; use --ignored --nocapture"]
fn lsode2_lambdify_large_sparse_banded_frontend_policy_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::lambdify_stress_story_tests::tests::lsode2_lambdify_large_sparse_banded_frontend_policy_story",
    );

    let profiles = [
        ("baseline", [20.0, 4.0, 0.20, 0.010]),
        ("diffusive", [24.0, 8.0, 0.25, 0.008]),
    ];
    let dimensions = std::env::var("LSODE2_LAMBDIFY_STRESS_DIMENSIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 3)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| vec![12, 32, 64, 128]);
    let repeats = std::env::var("LSODE2_LAMBDIFY_STRESS_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|repeats| *repeats > 0)
        .unwrap_or(3);
    let routes = [
        (
            Frontend::ExprLegacy,
            MatrixRoute::Sparse,
            LinearPolicy::Auto,
        ),
        (
            Frontend::AtomViewNative,
            MatrixRoute::Sparse,
            LinearPolicy::Auto,
        ),
        (
            Frontend::ExprLegacy,
            MatrixRoute::Banded,
            LinearPolicy::Auto,
        ),
        (
            Frontend::AtomViewNative,
            MatrixRoute::Banded,
            LinearPolicy::Auto,
        ),
        (
            Frontend::ExprLegacy,
            MatrixRoute::Sparse,
            LinearPolicy::Force,
        ),
        (
            Frontend::ExprLegacy,
            MatrixRoute::Banded,
            LinearPolicy::Force,
        ),
        (
            Frontend::AtomViewNative,
            MatrixRoute::Sparse,
            LinearPolicy::Force,
        ),
        (
            Frontend::AtomViewNative,
            MatrixRoute::Banded,
            LinearPolicy::Force,
        ),
    ];

    reportln!(
        "[LSODE2 Lambdify stress] AOT excluded; callback evaluator policy is Sequential for this legacy baseline"
    );
    reportln!("[LSODE2 Lambdify stress] linear_policy and evaluator_policy are separate axes");
    reportln!(
        "dimensions={dimensions:?}; repeats={repeats}; t_bound=8.0; max_step=0.05; Dense intentionally excluded"
    );
    reportln!(
        "profile | dim | frontend           | matrix  | linear | evaluator  | rep | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | factor_ms | rhs_ms | residuals | jacobians | jac_rebuilds | linear_solves | accepted | rejected | final_norm"
    );
    reportln!(
        "--- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    );

    let mut successful_runs = 0usize;
    for (profile, parameter_values) in profiles {
        for dimension in dimensions.iter().copied() {
            let mut reference_state: Option<DVector<f64>> = None;
            for (frontend, matrix, policy) in routes {
                for repetition in 1..=repeats {
                    let run = run_stress_case(
                        dimension,
                        frontend,
                        matrix,
                        policy,
                        profile,
                        parameter_values,
                    );
                    print_run(&run, repetition);
                    assert_eq!(run.telemetry.state_dimension, dimension);
                    assert_eq!(run.telemetry.residual_dimension, dimension);
                    assert_eq!(run.telemetry.parameter_count, 4);
                    assert!(run.telemetry.residual_evaluations > 0);
                    assert!(run.telemetry.jacobian_evaluations > 0);
                    assert!(run.telemetry.linear_solve_requests > 0);
                    assert!(run.final_norm.is_finite());
                    if let Some(reference) = reference_state.as_ref() {
                        let max_diff = run
                            .final_state
                            .iter()
                            .zip(reference.iter())
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f64, f64::max);
                        assert!(
                            max_diff <= 5.0e-6,
                            "Lambdify route drift at profile={profile}, dimension={dimension}, frontend={}, matrix={}, policy={}: {max_diff:e}",
                            run.frontend,
                            run.matrix,
                            run.policy
                        );
                    } else {
                        reference_state = Some(run.final_state.clone());
                    }
                    successful_runs += 1;
                }
            }
        }
    }

    reportln!(
        "[LSODE2 Lambdify stress] completed_runs={successful_runs}; expected_runs={}",
        profiles.len() * dimensions.len() * routes.len() * repeats
    );
    assert_eq!(
        successful_runs,
        profiles.len() * dimensions.len() * routes.len() * repeats
    );
}

#[test]
#[ignore = "release Lambdify evaluator-policy matrix; use --ignored --nocapture"]
fn lsode2_lambdify_evaluator_policy_matrix_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::lambdify_stress_story_tests::tests::lsode2_lambdify_evaluator_policy_matrix_story",
    );
    let dimensions = std::env::var("LSODE2_LAMBDIFY_POLICY_DIMENSIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 3)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| vec![32, 128]);
    let repeats = std::env::var("LSODE2_LAMBDIFY_POLICY_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|repeats| *repeats > 0)
        .unwrap_or(3);
    let frontends = [Frontend::ExprLegacy, Frontend::AtomViewNative];
    let matrices = [MatrixRoute::Sparse, MatrixRoute::Banded];
    let policies = [
        EvaluatorPolicy::Sequential,
        EvaluatorPolicy::Parallel { min_work: 64 },
        EvaluatorPolicy::Auto { min_work: 64 },
    ];

    reportln!(
        "[LSODE2 Lambdify evaluator policy] AOT excluded; dimensions={dimensions:?}; repeats={repeats}; same symbolic task and linear policy for every evaluator row"
    );
    reportln!(
        "frontend | matrix | evaluator_policy | rep | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | parallel_dispatches | sequential_dispatches | residual_calls | jacobian_calls | jacobian_rebuilds | linear_solves | accepted | rejected | max_state_diff"
    );
    reportln!(
        "--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    );

    for dimension in dimensions {
        for frontend in frontends {
            for matrix in matrices {
                let mut reference_state: Option<DVector<f64>> = None;
                for evaluator_policy in policies {
                    for repetition in 1..=repeats {
                        let run = run_stress_case_with_evaluator_policy(
                            dimension,
                            frontend,
                            matrix,
                            LinearPolicy::Auto,
                            "policy-matrix",
                            [20.0, 4.0, 0.20, 0.010],
                            evaluator_policy,
                        );
                        let telemetry = &run.telemetry;
                        let max_diff = reference_state
                            .as_ref()
                            .map(|reference| {
                                run.final_state
                                    .iter()
                                    .zip(reference.iter())
                                    .map(|(actual, expected)| (actual - expected).abs())
                                    .fold(0.0_f64, f64::max)
                            })
                            .unwrap_or(0.0);
                        reportln!(
                            "{} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {:.3e}",
                            run.frontend,
                            run.matrix,
                            run.evaluator_policy,
                            repetition,
                            run.total_ms,
                            run.prepare_ms,
                            run.solve_ms,
                            telemetry
                                .warm_stage(IvpWarmStage::ResidualEvaluation)
                                .elapsed
                                .as_secs_f64()
                                * 1_000.0,
                            telemetry
                                .warm_stage(IvpWarmStage::JacobianEvaluation)
                                .elapsed
                                .as_secs_f64()
                                * 1_000.0,
                            telemetry.parallel_dispatches,
                            telemetry.sequential_dispatches,
                            telemetry.residual_evaluations,
                            telemetry.jacobian_evaluations,
                            telemetry.jacobian_rebuilds,
                            telemetry.linear_solve_requests,
                            telemetry.accepted_steps,
                            telemetry.rejected_steps,
                            max_diff,
                        );
                        assert_eq!(
                            telemetry.lambdify_execution_policy,
                            evaluator_policy.runtime(),
                            "telemetry policy drift for {} / {}",
                            run.frontend,
                            run.matrix
                        );
                        assert!(run.final_norm.is_finite());
                        assert!(telemetry.residual_evaluations > 0);
                        assert!(telemetry.jacobian_evaluations > 0);
                        assert!(telemetry.linear_solve_requests > 0);
                        if matches!(evaluator_policy, EvaluatorPolicy::Sequential) {
                            assert_eq!(telemetry.parallel_dispatches, 0);
                        }
                        assert!(max_diff <= 5.0e-6);
                        if reference_state.is_none() {
                            reference_state = Some(run.final_state.clone());
                        }
                    }
                }
            }
        }
    }
}

struct CallbackOnlyRun {
    frontend: &'static str,
    dimension: usize,
    evaluator_policy: &'static str,
    repetitions: usize,
    residual_ms: f64,
    jacobian_ms: f64,
    checksum: f64,
    telemetry: IvpTelemetrySnapshot,
    residual_reference: DVector<f64>,
    jacobian_reference: Vec<f64>,
}

fn run_callback_only_case(
    dimension: usize,
    frontend: Frontend,
    evaluator_policy: EvaluatorPolicy,
    repetitions: usize,
) -> CallbackOnlyRun {
    let telemetry = IvpTelemetry::detailed();
    let problem = prepare_symbolic_ivp_problem(
        chain_equations(dimension),
        (0..dimension).map(|index| format!("y{index}")).collect(),
        "t".to_string(),
        SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(match frontend {
                Frontend::ExprLegacy => IvpSymbolicAssemblyBackend::ExprLegacy,
                Frontend::AtomViewNative => IvpSymbolicAssemblyBackend::AtomView,
            })
            .with_equation_parameters(vec![
                "k".to_string(),
                "d".to_string(),
                "q".to_string(),
                "nl".to_string(),
            ])
            .with_equation_parameter_values(DVector::from_vec(vec![20.0, 4.0, 0.2, 0.01]))
            .with_lambdify_execution_policy(evaluator_policy.runtime())
            .with_telemetry(telemetry),
    )
    .expect("callback-only IVP preparation should succeed");
    let state = DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| 0.25 + 0.001 * (index % 17) as f64),
    );

    let started_residual = Instant::now();
    let mut residual_reference = DVector::zeros(dimension);
    let mut checksum = 0.0;
    for repetition in 0..repetitions {
        let residual = (problem.residual)(0.5 + repetition as f64 * 0.001, &state);
        if repetition == 0 {
            residual_reference = residual.clone();
        }
        checksum += residual[repetition % dimension];
    }
    let residual_ms = started_residual.elapsed().as_secs_f64() * 1_000.0;

    let started_jacobian = Instant::now();
    let mut jacobian_reference = vec![0.0; dimension * dimension];
    for repetition in 0..repetitions {
        let jacobian = (problem.jacobian)(0.5 + repetition as f64 * 0.001, &state);
        if repetition == 0 {
            jacobian_reference = jacobian.iter().copied().collect();
        }
        checksum += jacobian[(repetition % dimension, repetition % dimension)];
    }
    let jacobian_ms = started_jacobian.elapsed().as_secs_f64() * 1_000.0;
    black_box(checksum);

    CallbackOnlyRun {
        frontend: frontend.label(),
        dimension,
        evaluator_policy: evaluator_policy.label(),
        repetitions,
        residual_ms,
        jacobian_ms,
        checksum,
        telemetry: problem.telemetry.snapshot(),
        residual_reference,
        jacobian_reference,
    }
}

#[test]
#[ignore = "release callback-only Lambdify baseline; use --ignored --nocapture"]
fn lsode2_lambdify_callback_only_policy_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::lambdify_stress_story_tests::tests::lsode2_lambdify_callback_only_policy_story",
    );
    let dimensions = std::env::var("LSODE2_LAMBDIFY_CALLBACK_DIMENSIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 3)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| vec![16, 64, 128]);
    let repetitions = std::env::var("LSODE2_LAMBDIFY_CALLBACK_REPETITIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|repetitions| *repetitions > 0)
        .unwrap_or(10);
    let policies = [
        EvaluatorPolicy::Sequential,
        EvaluatorPolicy::Parallel { min_work: 64 },
        EvaluatorPolicy::Auto { min_work: 64 },
    ];

    reportln!(
        "[LSODE2 Lambdify callback-only] preparation excluded; dimensions={dimensions:?}; repetitions={repetitions}; ExprLegacy vs AtomViewNative"
    );
    reportln!(
        "frontend | dim | evaluator_policy | repetitions | residual_ms | jacobian_ms | parallel_dispatches | sequential_dispatches | worker_count | residual_calls | jacobian_calls | max_residual_diff | max_jacobian_diff | checksum"
    );
    reportln!(
        "--- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    );

    for dimension in dimensions {
        for frontend in [Frontend::ExprLegacy, Frontend::AtomViewNative] {
            let mut reference: Option<CallbackOnlyRun> = None;
            for evaluator_policy in policies {
                let run =
                    run_callback_only_case(dimension, frontend, evaluator_policy, repetitions);
                let (max_residual_diff, max_jacobian_diff) = reference
                    .as_ref()
                    .map(|reference| {
                        let residual_diff = run
                            .residual_reference
                            .iter()
                            .zip(reference.residual_reference.iter())
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f64, f64::max);
                        let jacobian_diff = run
                            .jacobian_reference
                            .iter()
                            .zip(reference.jacobian_reference.iter())
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f64, f64::max);
                        (residual_diff, jacobian_diff)
                    })
                    .unwrap_or((0.0, 0.0));
                let telemetry = &run.telemetry;
                reportln!(
                    "{} | {} | {} | {} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {:.3e} | {:.3e} | {:.6e}",
                    run.frontend,
                    run.dimension,
                    run.evaluator_policy,
                    run.repetitions,
                    run.residual_ms,
                    run.jacobian_ms,
                    telemetry.parallel_dispatches,
                    telemetry.sequential_dispatches,
                    telemetry.lambdify_worker_count,
                    telemetry.residual_evaluations,
                    telemetry.jacobian_evaluations,
                    max_residual_diff,
                    max_jacobian_diff,
                    run.checksum,
                );
                assert_eq!(
                    telemetry.lambdify_execution_policy,
                    evaluator_policy.runtime()
                );
                assert_eq!(telemetry.residual_evaluations, repetitions as u64);
                assert_eq!(telemetry.jacobian_evaluations, repetitions as u64);
                assert!(max_residual_diff <= 1.0e-12);
                assert!(max_jacobian_diff <= 1.0e-12);
                if matches!(evaluator_policy, EvaluatorPolicy::Sequential) {
                    assert_eq!(telemetry.parallel_dispatches, 0);
                }
                if reference.is_none() {
                    reference = Some(run);
                }
            }
        }
    }
}

#[test]
fn lsode2_lambdify_prepared_parameter_rebind_detailed_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::lambdify_stress_story_tests::tests::lsode2_lambdify_prepared_parameter_rebind_detailed_story",
    );
    let dimension = 64usize;
    let telemetry = IvpTelemetry::detailed();
    let equations = chain_equations(dimension);
    let problem = prepare_symbolic_ivp_problem(
        equations,
        (0..dimension).map(|index| format!("y{index}")).collect(),
        "t".to_string(),
        SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
            .with_equation_parameters(vec![
                "k".to_string(),
                "d".to_string(),
                "q".to_string(),
                "nl".to_string(),
            ])
            .with_equation_parameter_values(DVector::from_vec(vec![20.0, 4.0, 0.2, 0.01]))
            .with_telemetry(telemetry),
    )
    .expect("prepared parameterized Lambdify problem should build");
    let state = DVector::from_element(dimension, 0.25);
    let profiles = [
        ("baseline", [20.0, 4.0, 0.20, 0.010]),
        ("diffusive", [24.0, 8.0, 0.25, 0.008]),
        ("mild", [16.0, 2.0, 0.15, 0.005]),
    ];

    for (profile, values) in profiles {
        problem
            .set_parameter_values(DVector::from_column_slice(&values))
            .expect("parameter rebind should preserve the prepared callbacks");
        let residual = (problem.residual)(0.5, &state);
        let jacobian = (problem.jacobian)(0.5, &state);
        assert_eq!(residual.len(), dimension);
        assert_eq!(jacobian.nrows(), dimension);
        assert_eq!(jacobian.ncols(), dimension);
        assert!(residual.iter().all(|value| value.is_finite()));
        assert!(jacobian.iter().all(|value| value.is_finite()));
        reportln!(
            "[LSODE2 Lambdify parameter rebind] profile={profile}; residual_l2={:.6e}; jacobian_frobenius={:.6e}",
            residual.norm(),
            jacobian.norm()
        );
    }

    let snapshot = problem.telemetry.snapshot();
    reportln!("[LSODE2 Lambdify parameter rebind] typed telemetry follows");
    reportln!("{}", snapshot.pretty_report());
    assert_eq!(snapshot.parameter_binds, profiles.len() as u64);
    assert_eq!(snapshot.symbolic_jacobian_builds, 1);
    assert_eq!(snapshot.state_dimension, dimension);
    assert_eq!(snapshot.residual_dimension, dimension);
    assert_eq!(snapshot.parameter_count, 4);
    assert_eq!(
        snapshot.warm_stage(IvpWarmStage::ResidualEvaluation).calls,
        profiles.len() as u64
    );
    assert_eq!(
        snapshot.warm_stage(IvpWarmStage::JacobianEvaluation).calls,
        profiles.len() as u64
    );
    assert!(snapshot.cold_stage(IvpColdStage::ResidualCompilation).calls > 0);
}
