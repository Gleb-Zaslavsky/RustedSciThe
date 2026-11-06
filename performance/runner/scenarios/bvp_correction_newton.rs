use crate::error::RunnerError;
use crate::registry::{NativeMetricScope, Scenario, ScenarioOutput};
use RustedSciThe::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonMatrixLayout, CorrectionNewtonOptions, CorrectionNewtonParameterLoopOutcome,
    CorrectionNewtonProblem, CorrectionNewtonSolver, CorrectionNewtonSymbolicFrontend,
    CorrectionNewtonTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const STEPS: usize = 24000;
const BRATU_STEPS: usize = 32768;
const KITHE_STEPS: usize = 256;
const STATE_NAMES: [&str; 6] = ["Teta", "q", "C0", "J0", "C1", "J1"];

#[derive(Clone, Copy, Debug)]
enum Frontend {
    Expr,
    Atom,
}

impl Frontend {
    fn label(self) -> &'static str {
        match self {
            Self::Expr => "exprlegacy",
            Self::Atom => "atomview",
        }
    }

    fn native(self) -> CorrectionNewtonSymbolicFrontend {
        match self {
            Self::Expr => CorrectionNewtonSymbolicFrontend::ExprLegacy,
            Self::Atom => CorrectionNewtonSymbolicFrontend::AtomView,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Sparse,
    Banded,
}

impl Layout {
    fn label(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Banded => "banded",
        }
    }

    fn native(self) -> CorrectionNewtonMatrixLayout {
        match self {
            Self::Sparse => CorrectionNewtonMatrixLayout::Sparse,
            Self::Banded => CorrectionNewtonMatrixLayout::BandedAuto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Execution {
    Lambdify,
    Aot,
}

impl Execution {
    fn label(self) -> &'static str {
        match self {
            Self::Lambdify => "lambdify",
            Self::Aot => "aot",
        }
    }
}

fn problem() -> CorrectionNewtonProblem {
    let states = STATE_NAMES.map(str::to_owned).to_vec();
    let equations = [
        "q",
        "-0.5*q + 0.2*exp(Teta)*C0",
        "J0",
        "-0.3*J0 - 0.5*C0 + 0.1*Teta",
        "J1",
        "-0.2*J1 - 0.3*C1 + 0.05*C0^2",
    ]
    .into_iter()
    .map(Expr::parse_expression)
    .collect::<Vec<_>>();
    let initial_guess =
        DMatrix::from_fn(
            STATE_NAMES.len(),
            STEPS,
            |state, _| {
                if state == 2 { 1.0 } else { 0.0 }
            },
        );
    CorrectionNewtonProblem {
        equations,
        initial_guess,
        state_names: states.clone(),
        independent_variable: "x".into(),
        boundary_conditions: HashMap::from([
            ("Teta".into(), vec![(0, 0.0)]),
            ("q".into(), vec![(1, 0.0)]),
            ("C0".into(), vec![(0, 1.0)]),
            ("J0".into(), vec![(1, 0.0)]),
            ("C1".into(), vec![(0, 0.0)]),
            ("J1".into(), vec![(1, 0.0)]),
        ]),
        interval: (0.0, 1.0),
        steps: STEPS,
    }
}

fn bratu_problem() -> CorrectionNewtonProblem {
    CorrectionNewtonProblem {
        equations: ["z", "-2*exp(y)"]
            .into_iter()
            .map(Expr::parse_expression)
            .collect(),
        initial_guess: DMatrix::from_fn(
            2,
            BRATU_STEPS,
            |state, _| {
                if state == 0 { 0.25 } else { 0.0 }
            },
        ),
        state_names: vec!["y".into(), "z".into()],
        independent_variable: "x".into(),
        boundary_conditions: HashMap::from([
            ("y".into(), vec![(0, 0.0)]),
            ("z".into(), vec![(1, 0.0)]),
        ]),
        // Symmetry reduces the standard Dirichlet Bratu problem to half-domain
        // conditions y(0)=0 and y'(0.5)=z(0.5)=0.
        interval: (0.0, 0.5),
        steps: BRATU_STEPS,
    }
}

fn native_stage_scopes() -> Vec<NativeMetricScope> {
    [
        ("stage_residual_ms", "ResidualCallback", "callback_ms"),
        ("stage_jacobian_ms", "JacobianCallback", "callback_ms"),
        (
            "stage_jacobian_assembly_ms",
            "JacobianAssembly",
            "linear_ms",
        ),
        ("stage_factorization_ms", "Factorization", "linear_ms"),
        ("stage_linear_solve_ms", "LinearSolve", "linear_ms"),
    ]
    .into_iter()
    .map(|(metric_id, stage, parent)| NativeMetricScope {
        metric_id: metric_id.into(),
        stage: stage.into(),
        kind: "native_child".into(),
        parent: Some(parent.into()),
    })
    .collect()
}

fn execute_bratu(
    frontend: Frontend,
    layout: Layout,
    telemetry: CorrectionNewtonTelemetryMode,
) -> Result<ScenarioOutput, RunnerError> {
    let solver_wall_ms;
    let mut iterations = 0_u64;
    let mut residual_calls = 0_u64;
    let mut jacobian_calls = 0_u64;
    let mut factorizations = 0_u64;
    let mut linear_solves = 0_u64;
    let mut residual_ms = 0.0;
    let mut jacobian_ms = 0.0;
    let mut assembly_ms = 0.0;
    let mut factorization_ms = 0.0;
    let mut linear_solve_ms = 0.0;
    let mut max_residual = 0.0_f64;

    {
        let options = CorrectionNewtonOptions {
            symbolic_frontend: frontend.native(),
            matrix_layout: layout.native(),
            absolute_tolerance: 1.0e-7,
            relative_tolerances: HashMap::from([("y".into(), 1.0e-7), ("z".into(), 1.0e-7)]),
            bounds: HashMap::from([("y".into(), (-1.0, 5.0)), ("z".into(), (-20.0, 20.0))]),
            max_iterations: 50,
            telemetry_mode: telemetry,
            ..CorrectionNewtonOptions::default()
        };
        let wall_started = Instant::now();
        let mut solver =
            CorrectionNewtonSolver::from_symbolic(bratu_problem(), options).map_err(|error| {
                RunnerError::Scenario(format!("CorrectionNewton Bratu prepare: {error}"))
            })?;
        let report = solver.solve().map_err(|error| {
            RunnerError::Scenario(format!("CorrectionNewton Bratu solve: {error}"))
        })?;
        solver_wall_ms = wall_started.elapsed().as_secs_f64() * 1_000.0;
        if report.state.iter().any(|value| !value.is_finite())
            || report
                .residual_norm
                .is_none_or(|value| !value.is_finite() || value > 1.0e-5)
        {
            return Err(RunnerError::Scenario(format!(
                "CorrectionNewton Bratu correctness gate failed: residual={:?}",
                report.residual_norm
            )));
        }
        let full = solver.full_grid_values(&report.state).map_err(|error| {
            RunnerError::Scenario(format!("CorrectionNewton Bratu profile: {error}"))
        })?;
        if full.shape() != (BRATU_STEPS + 1, 2)
            || full.iter().any(|value| !value.is_finite())
            || full.column(0).iter().any(|value| *value < -1.0e-10)
            || full.column(0).iter().any(|value| *value > 5.0)
            || full[(0, 0)].abs() > 1.0e-12
            || full[(BRATU_STEPS, 1)].abs() > 1.0e-12
        {
            return Err(RunnerError::Scenario(
                "CorrectionNewton Bratu profile failed shape, boundary, finiteness, or bounds checks".into(),
            ));
        }
        max_residual = max_residual.max(report.residual_norm.unwrap_or(f64::INFINITY));
        iterations += report.iterations as u64;
        residual_calls += report.telemetry.residual_evaluations as u64;
        jacobian_calls += report.telemetry.jacobian_evaluations as u64;
        factorizations += report.telemetry.factorizations as u64;
        linear_solves += report.telemetry.linear_solves as u64;
        residual_ms += report.telemetry.residual_callback_time.as_secs_f64() * 1_000.0;
        jacobian_ms += report.telemetry.jacobian_callback_time.as_secs_f64() * 1_000.0;
        assembly_ms += report.telemetry.jacobian_assembly_time.as_secs_f64() * 1_000.0;
        factorization_ms += report.telemetry.factorization_time.as_secs_f64() * 1_000.0;
        linear_solve_ms += report.telemetry.linear_solve_time.as_secs_f64() * 1_000.0;
        black_box((&full, &report));
    }

    let diagnostics = telemetry != CorrectionNewtonTelemetryMode::Off;
    let metrics = if diagnostics {
        vec![
            ("stage_residual_ms".into(), "ms/solve".into(), residual_ms),
            ("stage_jacobian_ms".into(), "ms/solve".into(), jacobian_ms),
            (
                "stage_jacobian_assembly_ms".into(),
                "ms/solve".into(),
                assembly_ms,
            ),
            (
                "stage_factorization_ms".into(),
                "ms/solve".into(),
                factorization_ms,
            ),
            (
                "stage_linear_solve_ms".into(),
                "ms/solve".into(),
                linear_solve_ms,
            ),
        ]
    } else {
        vec![("scenario_wall_ms".into(), "ms/solve".into(), solver_wall_ms)]
    };
    let counters = diagnostics
        .then_some(vec![
            ("state_dimension".into(), (2 * BRATU_STEPS) as u64),
            ("fresh_solves".into(), 1),
            ("iterations".into(), iterations),
            ("residual_evaluations".into(), residual_calls),
            ("jacobian_evaluations".into(), jacobian_calls),
            ("factorizations".into(), factorizations),
            ("linear_solves".into(), linear_solves),
        ])
        .unwrap_or_default();
    Ok(ScenarioOutput {
        metrics,
        native_scopes: diagnostics.then(native_stage_scopes).unwrap_or_default(),
        counters,
        diagnostics: vec![("max_residual_norm".into(), "absolute".into(), max_residual)],
    })
}

fn execute_kithe_combustion(
    telemetry: CorrectionNewtonTelemetryMode,
) -> Result<ScenarioOutput, RunnerError> {
    let solver_wall_ms;
    let mut iterations = 0_u64;
    let mut residual_calls = 0_u64;
    let mut jacobian_calls = 0_u64;
    let mut factorizations = 0_u64;
    let mut linear_solves = 0_u64;
    let mut residual_ms = 0.0;
    let mut jacobian_ms = 0.0;
    let mut assembly_ms = 0.0;
    let mut factorization_ms = 0.0;
    let mut linear_solve_ms = 0.0;
    let mut max_residual = 0.0_f64;
    let mut max_boundary_error = 0.0_f64;

    {
        let problem =
            RustedSciThe::numerical::BVP_Damp::correction_newton_kithe_combustion_perf_fixture();
        let source_steps = problem.steps;
        let source_guess = problem.initial_guess.clone();
        let component_count = problem.state_names.len();
        let mut problem = problem;
        problem.steps = KITHE_STEPS;
        problem.initial_guess = DMatrix::from_fn(component_count, KITHE_STEPS, |state, node| {
            let source_position =
                node as f64 * (source_steps - 1) as f64 / (KITHE_STEPS - 1) as f64;
            let lower = source_position.floor() as usize;
            let upper = (lower + 1).min(source_steps - 1);
            let fraction = source_position - lower as f64;
            source_guess[(state, lower)] * (1.0 - fraction)
                + source_guess[(state, upper)] * fraction
        });
        let relative_tolerances = problem
            .state_names
            .iter()
            .map(|name| (name.clone(), 1.0e-5))
            .collect();
        let bounds = problem
            .state_names
            .iter()
            .enumerate()
            .map(|(state, name)| {
                let bound = if state == 0 {
                    (-1.0, 3.0)
                } else if state % 2 == 1 {
                    (-1.0e4, 1.0e4)
                } else {
                    (-0.1, 1.1)
                };
                (name.clone(), bound)
            })
            .collect();
        let boundary_conditions = problem.boundary_conditions.clone();
        let steps = problem.steps;
        let options = CorrectionNewtonOptions {
            symbolic_frontend: CorrectionNewtonSymbolicFrontend::AtomView,
            matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
            absolute_tolerance: 1.0e-5,
            relative_tolerances,
            bounds,
            max_iterations: 150,
            telemetry_mode: telemetry,
            ..CorrectionNewtonOptions::default()
        };
        let wall_started = Instant::now();
        let mut solver = CorrectionNewtonSolver::from_symbolic(problem, options)
            .map_err(|error| RunnerError::Scenario(format!("KiThe combustion prepare: {error}")))?;
        let report = solver
            .solve()
            .map_err(|error| RunnerError::Scenario(format!("KiThe combustion solve: {error}")))?;
        solver_wall_ms = wall_started.elapsed().as_secs_f64() * 1_000.0;
        let residual_norm = report.residual_norm.unwrap_or(f64::INFINITY);
        if !residual_norm.is_finite()
            || report
                .correction_norm
                .is_none_or(|value| !value.is_finite() || value >= 1.0)
            || report.state.iter().any(|value| !value.is_finite())
        {
            return Err(RunnerError::Scenario(format!(
                "KiThe combustion correctness gate failed: correction={:?}, residual={residual_norm:e}",
                report.correction_norm
            )));
        }
        let full = solver
            .full_grid_values(&report.state)
            .map_err(|error| RunnerError::Scenario(format!("KiThe combustion profile: {error}")))?;
        if full.shape() != (steps + 1, component_count)
            || full.iter().any(|value| !value.is_finite())
        {
            return Err(RunnerError::Scenario(
                "KiThe combustion profile has invalid shape or non-finite values".into(),
            ));
        }
        for (name, constraints) in boundary_conditions {
            let state = name
                .strip_prefix('u')
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .ok_or_else(|| RunnerError::Scenario(format!("invalid fixture state {name}")))?;
            for (endpoint, expected) in constraints {
                let node = if endpoint == 0 { 0 } else { steps };
                max_boundary_error = max_boundary_error.max((full[(node, state)] - expected).abs());
            }
        }
        if max_boundary_error > 1.0e-10 {
            return Err(RunnerError::Scenario(format!(
                "KiThe combustion boundary gate failed: error={max_boundary_error:e}"
            )));
        }
        max_residual = max_residual.max(residual_norm);
        iterations += report.iterations as u64;
        residual_calls += report.telemetry.residual_evaluations as u64;
        jacobian_calls += report.telemetry.jacobian_evaluations as u64;
        factorizations += report.telemetry.factorizations as u64;
        linear_solves += report.telemetry.linear_solves as u64;
        residual_ms += report.telemetry.residual_callback_time.as_secs_f64() * 1_000.0;
        jacobian_ms += report.telemetry.jacobian_callback_time.as_secs_f64() * 1_000.0;
        assembly_ms += report.telemetry.jacobian_assembly_time.as_secs_f64() * 1_000.0;
        factorization_ms += report.telemetry.factorization_time.as_secs_f64() * 1_000.0;
        linear_solve_ms += report.telemetry.linear_solve_time.as_secs_f64() * 1_000.0;
        black_box((&full, &report));
    }

    let diagnostics = telemetry != CorrectionNewtonTelemetryMode::Off;
    let metrics = if diagnostics {
        vec![
            ("stage_residual_ms".into(), "ms/solve".into(), residual_ms),
            ("stage_jacobian_ms".into(), "ms/solve".into(), jacobian_ms),
            (
                "stage_jacobian_assembly_ms".into(),
                "ms/solve".into(),
                assembly_ms,
            ),
            (
                "stage_factorization_ms".into(),
                "ms/solve".into(),
                factorization_ms,
            ),
            (
                "stage_linear_solve_ms".into(),
                "ms/solve".into(),
                linear_solve_ms,
            ),
        ]
    } else {
        vec![("scenario_wall_ms".into(), "ms/solve".into(), solver_wall_ms)]
    };
    let counters = diagnostics
        .then_some(vec![
            ("state_dimension".into(), (26 * KITHE_STEPS) as u64),
            ("fresh_solves".into(), 1),
            ("iterations".into(), iterations),
            ("residual_evaluations".into(), residual_calls),
            ("jacobian_evaluations".into(), jacobian_calls),
            ("factorizations".into(), factorizations),
            ("linear_solves".into(), linear_solves),
        ])
        .unwrap_or_default();
    Ok(ScenarioOutput {
        metrics,
        native_scopes: diagnostics.then(native_stage_scopes).unwrap_or_default(),
        counters,
        diagnostics: vec![
            (
                "max_boundary_abs_error".into(),
                "absolute".into(),
                max_boundary_error,
            ),
            ("max_residual_norm".into(), "absolute".into(), max_residual),
        ],
    })
}

fn execute(
    frontend: Frontend,
    layout: Layout,
    execution: Execution,
    telemetry: CorrectionNewtonTelemetryMode,
) -> Result<ScenarioOutput, RunnerError> {
    let solver_wall_ms;
    let mut iterations = 0_u64;
    let mut residual_calls = 0_u64;
    let mut jacobian_calls = 0_u64;
    let mut factorizations = 0_u64;
    let mut linear_solves = 0_u64;
    let mut residual_ms = 0.0;
    let mut jacobian_ms = 0.0;
    let mut assembly_ms = 0.0;
    let mut factorization_ms = 0.0;
    let mut linear_solve_ms = 0.0;
    let mut max_residual = 0.0_f64;
    let mut max_boundary_error = 0.0_f64;
    let mut max_bound_violation = 0.0_f64;

    {
        let mut options = CorrectionNewtonOptions {
            symbolic_frontend: frontend.native(),
            matrix_layout: layout.native(),
            absolute_tolerance: 1.0e-6,
            relative_tolerances: STATE_NAMES
                .into_iter()
                .map(|name| (name.to_owned(), 1.0e-5))
                .collect(),
            bounds: HashMap::from([
                ("Teta".into(), (-10.0, 10.0)),
                ("q".into(), (-10.0, 10.0)),
                ("C0".into(), (-2.0, 2.0)),
                ("J0".into(), (-10.0, 10.0)),
                ("C1".into(), (-2.0, 2.0)),
                ("J1".into(), (-10.0, 10.0)),
            ]),
            max_iterations: 100,
            telemetry_mode: telemetry,
            ..CorrectionNewtonOptions::default()
        };
        let wall_started = Instant::now();
        let mut solver = match execution {
            Execution::Lambdify => CorrectionNewtonSolver::from_symbolic(problem(), options),
            Execution::Aot => {
                let config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                    PathBuf::from("target/perf-gates-aot/bvp-correction-newton").join(format!(
                        "{}-{}-{}",
                        frontend.label(),
                        layout.label(),
                        execution.label()
                    )),
                )
                .with_c_tcc();
                options = options.with_aot_config(config);
                CorrectionNewtonSolver::from_symbolic(problem(), options)
            }
        }
        .map_err(|error| RunnerError::Scenario(format!("CorrectionNewton prepare: {error}")))?;
        let report = solver
            .solve()
            .map_err(|error| RunnerError::Scenario(format!("CorrectionNewton solve: {error}")))?;
        solver_wall_ms = wall_started.elapsed().as_secs_f64() * 1_000.0;
        if report.state.iter().any(|value| !value.is_finite())
            || report
                .residual_norm
                .is_none_or(|value| !value.is_finite() || value > 1.0e-5)
        {
            return Err(RunnerError::Scenario(format!(
                "CorrectionNewton correctness gate failed: residual={:?}",
                report.residual_norm
            )));
        }
        let full = solver
            .full_grid_values(&report.state)
            .map_err(|error| RunnerError::Scenario(format!("CorrectionNewton profile: {error}")))?;
        let expected_shape = (STEPS + 1, STATE_NAMES.len());
        if full.shape() != expected_shape {
            return Err(RunnerError::Scenario(format!(
                "CorrectionNewton profile shape mismatch: expected {expected_shape:?}, got {:?}",
                full.shape()
            )));
        }
        if full.iter().any(|value| !value.is_finite()) {
            return Err(RunnerError::Scenario(
                "CorrectionNewton profile contains a non-finite value".into(),
            ));
        }
        for state in 0..STATE_NAMES.len() {
            let boundary_value = match state {
                0 => full[(0, state)],
                1 | 3 | 5 => full[(STEPS, state)],
                2 | 4 => full[(0, state)],
                _ => unreachable!(),
            };
            let expected = match state {
                2 => 1.0,
                _ => 0.0,
            };
            max_boundary_error = max_boundary_error.max((boundary_value - expected).abs());
        }
        for (state, name) in STATE_NAMES.iter().enumerate() {
            let (lower, upper) = match *name {
                "C0" | "C1" => (-2.0, 2.0),
                _ => (-10.0, 10.0),
            };
            for node in 0..=STEPS {
                let value = full[(node, state)];
                max_bound_violation = max_bound_violation.max((lower - value).max(value - upper));
            }
        }
        max_residual = max_residual.max(report.residual_norm.unwrap_or(f64::INFINITY));
        iterations += report.iterations as u64;
        residual_calls += report.telemetry.residual_evaluations as u64;
        jacobian_calls += report.telemetry.jacobian_evaluations as u64;
        factorizations += report.telemetry.factorizations as u64;
        linear_solves += report.telemetry.linear_solves as u64;
        residual_ms += report.telemetry.residual_callback_time.as_secs_f64() * 1_000.0;
        jacobian_ms += report.telemetry.jacobian_callback_time.as_secs_f64() * 1_000.0;
        assembly_ms += report.telemetry.jacobian_assembly_time.as_secs_f64() * 1_000.0;
        factorization_ms += report.telemetry.factorization_time.as_secs_f64() * 1_000.0;
        linear_solve_ms += report.telemetry.linear_solve_time.as_secs_f64() * 1_000.0;
        black_box((&full, &report));
    }
    if max_boundary_error > 1.0e-12 || max_bound_violation > 1.0e-10 {
        return Err(RunnerError::Scenario(format!(
            "CorrectionNewton boundary/bounds check failed: boundary={max_boundary_error:e}, bounds={max_bound_violation:e}"
        )));
    }
    let diagnostics = telemetry != CorrectionNewtonTelemetryMode::Off;
    let metrics = if diagnostics {
        vec![
            ("stage_residual_ms".into(), "ms/solve".into(), residual_ms),
            ("stage_jacobian_ms".into(), "ms/solve".into(), jacobian_ms),
            (
                "stage_jacobian_assembly_ms".into(),
                "ms/solve".into(),
                assembly_ms,
            ),
            (
                "stage_factorization_ms".into(),
                "ms/solve".into(),
                factorization_ms,
            ),
            (
                "stage_linear_solve_ms".into(),
                "ms/solve".into(),
                linear_solve_ms,
            ),
        ]
    } else {
        vec![("scenario_wall_ms".into(), "ms/solve".into(), solver_wall_ms)]
    };
    let native_scopes = if diagnostics {
        [
            (
                "stage_residual_ms",
                "ResidualCallback",
                "native_child",
                "callback_ms",
            ),
            (
                "stage_jacobian_ms",
                "JacobianCallback",
                "native_child",
                "callback_ms",
            ),
            (
                "stage_jacobian_assembly_ms",
                "JacobianAssembly",
                "native_child",
                "linear_ms",
            ),
            (
                "stage_factorization_ms",
                "Factorization",
                "native_child",
                "linear_ms",
            ),
            (
                "stage_linear_solve_ms",
                "LinearSolve",
                "native_child",
                "linear_ms",
            ),
        ]
        .into_iter()
        .map(|(metric_id, stage, kind, parent)| NativeMetricScope {
            metric_id: metric_id.into(),
            stage: stage.into(),
            kind: kind.into(),
            parent: Some(parent.into()),
        })
        .collect()
    } else {
        Vec::new()
    };
    Ok(ScenarioOutput {
        metrics,
        native_scopes,
        counters: diagnostics
            .then_some(vec![
                ("state_dimension".into(), (STATE_NAMES.len() * STEPS) as u64),
                ("fresh_solves".into(), 1),
                ("iterations".into(), iterations),
                ("residual_evaluations".into(), residual_calls),
                ("jacobian_evaluations".into(), jacobian_calls),
                ("factorizations".into(), factorizations),
                ("linear_solves".into(), linear_solves),
            ])
            .unwrap_or_default(),
        diagnostics: vec![
            (
                "max_boundary_abs_error".into(),
                "absolute".into(),
                max_boundary_error,
            ),
            (
                "max_bound_violation".into(),
                "absolute".into(),
                max_bound_violation,
            ),
            ("max_residual_norm".into(), "absolute".into(), max_residual),
        ],
    })
}

fn execute_parameter_continuation(
    frontend: Frontend,
    layout: Layout,
    execution: Execution,
    telemetry: CorrectionNewtonTelemetryMode,
) -> Result<ScenarioOutput, RunnerError> {
    const POINTS: usize = 10;
    let states = (0..STATE_NAMES.len())
        .map(|index| format!("y{index}"))
        .collect::<Vec<_>>();
    let equations = states
        .iter()
        .map(|state| Expr::parse_expression(&format!("-rate*{state}")))
        .collect::<Vec<_>>();
    let problem = CorrectionNewtonProblem {
        equations,
        initial_guess: DMatrix::from_element(STATE_NAMES.len(), STEPS, 1.0),
        state_names: states.clone(),
        independent_variable: "x".into(),
        boundary_conditions: states
            .into_iter()
            .map(|state| (state, vec![(0, 1.0)]))
            .collect(),
        interval: (0.0, 1.0),
        steps: STEPS,
    };
    let mut options = CorrectionNewtonOptions {
        symbolic_frontend: frontend.native(),
        matrix_layout: layout.native(),
        absolute_tolerance: 1.0e-7,
        default_relative_tolerance: 1.0e-7,
        max_iterations: 8,
        telemetry_mode: telemetry,
        ..CorrectionNewtonOptions::default()
    };
    if execution == Execution::Aot {
        let config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
            PathBuf::from("target/perf-gates-aot/bvp-correction-newton-parameter-loop").join(
                format!(
                    "{}-{}-{}",
                    frontend.label(),
                    layout.label(),
                    execution.label()
                ),
            ),
        )
        .with_c_tcc();
        options = options.with_aot_config(config);
    }

    let wall_started = Instant::now();
    let mut solver = CorrectionNewtonSolver::from_symbolic(problem, options).map_err(|error| {
        RunnerError::Scenario(format!("CorrectionNewton continuation prepare: {error}"))
    })?;
    solver
        .bind_parameters(vec!["rate".into()], vec![0.5])
        .map_err(|error| {
            RunnerError::Scenario(format!("CorrectionNewton continuation bind: {error}"))
        })?;

    let parameter_vectors = (0..POINTS)
        .map(|point| vec![0.5 + point as f64 * 0.1])
        .collect::<Vec<_>>();
    let report = solver
        .solve_parameter_loop(&parameter_vectors)
        .map_err(|error| {
            RunnerError::Scenario(format!("CorrectionNewton parameter loop: {error}"))
        })?;
    if let Some(error) = report.stop_error.as_ref() {
        return Err(RunnerError::Scenario(format!(
            "CorrectionNewton parameter loop stopped before completing its schedule: {error}"
        )));
    }
    if report.points.len() != POINTS {
        return Err(RunnerError::Scenario(format!(
            "CorrectionNewton parameter loop completed {} of {POINTS} points",
            report.points.len()
        )));
    }

    let mut iterations = 0_u64;
    let mut residual_calls = 0_u64;
    let mut jacobian_calls = 0_u64;
    let mut factorizations = 0_u64;
    let mut linear_solves = 0_u64;
    let mut residual_ms = 0.0;
    let mut jacobian_ms = 0.0;
    let mut assembly_ms = 0.0;
    let mut factorization_ms = 0.0;
    let mut linear_solve_ms = 0.0;
    let solver_wall_ms = wall_started.elapsed().as_secs_f64() * 1_000.0;
    let mut max_endpoint_error = 0.0_f64;
    let mut max_residual_norm = 0.0_f64;
    for (point_index, (point, parameters)) in
        report.points.iter().zip(&parameter_vectors).enumerate()
    {
        if point.index != point_index || &point.parameters != parameters {
            return Err(RunnerError::Scenario(format!(
                "CorrectionNewton loop point {point_index} lost schedule order or parameter values"
            )));
        }
        let expected_guess_source = if point_index == 0 {
            None
        } else {
            Some(point_index - 1)
        };
        if point.guess_source_index != expected_guess_source {
            return Err(RunnerError::Scenario(format!(
                "CorrectionNewton loop point {point_index} used guess {:?}, expected {:?}",
                point.guess_source_index, expected_guess_source
            )));
        }
        let (mesh, values, solve_report) = match &point.outcome {
            CorrectionNewtonParameterLoopOutcome::Converged {
                mesh,
                values,
                report,
            } => (mesh, values, report),
            CorrectionNewtonParameterLoopOutcome::Failed { error } => {
                return Err(RunnerError::Scenario(format!(
                    "CorrectionNewton parameter loop point {point_index} failed: {error}"
                )));
            }
        };
        if values.shape() != (STEPS + 1, STATE_NAMES.len())
            || mesh.len() != STEPS + 1
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(RunnerError::Scenario(format!(
                "CorrectionNewton loop profile shape/finiteness failed at point {point_index}"
            )));
        }
        let rate = parameters[0];
        let half_step = 0.5 * rate / STEPS as f64;
        let exact_endpoint = ((1.0 - half_step) / (1.0 + half_step)).powi(STEPS as i32);
        let endpoint_error = (0..STATE_NAMES.len())
            .map(|state| (values[(STEPS, state)] - exact_endpoint).abs())
            .fold(0.0_f64, f64::max);
        max_endpoint_error = max_endpoint_error.max(endpoint_error);
        let residual_norm = solve_report.residual_norm.unwrap_or(f64::INFINITY);
        max_residual_norm = max_residual_norm.max(residual_norm);
        if !residual_norm.is_finite() || residual_norm > 1.0e-6 || endpoint_error > 1.0e-6 {
            return Err(RunnerError::Scenario(format!(
                "CorrectionNewton continuation correctness failed at point {point_index}: residual={residual_norm:e}, endpoint_error={endpoint_error:e}"
            )));
        }
        iterations += solve_report.iterations as u64;
        residual_calls += solve_report.telemetry.residual_evaluations as u64;
        jacobian_calls += solve_report.telemetry.jacobian_evaluations as u64;
        factorizations += solve_report.telemetry.factorizations as u64;
        linear_solves += solve_report.telemetry.linear_solves as u64;
        residual_ms += solve_report.telemetry.residual_callback_time.as_secs_f64() * 1_000.0;
        jacobian_ms += solve_report.telemetry.jacobian_callback_time.as_secs_f64() * 1_000.0;
        assembly_ms += solve_report.telemetry.jacobian_assembly_time.as_secs_f64() * 1_000.0;
        factorization_ms += solve_report.telemetry.factorization_time.as_secs_f64() * 1_000.0;
        linear_solve_ms += solve_report.telemetry.linear_solve_time.as_secs_f64() * 1_000.0;
    }
    if max_endpoint_error > 1.0e-6 {
        return Err(RunnerError::Scenario(format!(
            "CorrectionNewton continuation profile gate failed: max endpoint error={max_endpoint_error:e}"
        )));
    }
    black_box(&report);

    let diagnostics = telemetry != CorrectionNewtonTelemetryMode::Off;
    let metrics = if diagnostics {
        vec![
            ("stage_residual_ms".into(), "ms/loop".into(), residual_ms),
            ("stage_jacobian_ms".into(), "ms/loop".into(), jacobian_ms),
            (
                "stage_jacobian_assembly_ms".into(),
                "ms/loop".into(),
                assembly_ms,
            ),
            (
                "stage_factorization_ms".into(),
                "ms/loop".into(),
                factorization_ms,
            ),
            (
                "stage_linear_solve_ms".into(),
                "ms/loop".into(),
                linear_solve_ms,
            ),
        ]
    } else {
        vec![("scenario_wall_ms".into(), "ms/loop".into(), solver_wall_ms)]
    };
    Ok(ScenarioOutput {
        metrics,
        native_scopes: diagnostics.then(native_stage_scopes).unwrap_or_default(),
        counters: diagnostics
            .then_some(vec![
                ("state_dimension".into(), (STATE_NAMES.len() * STEPS) as u64),
                ("parameter_points".into(), POINTS as u64),
                ("iterations".into(), iterations),
                ("residual_evaluations".into(), residual_calls),
                ("jacobian_evaluations".into(), jacobian_calls),
                ("factorizations".into(), factorizations),
                ("linear_solves".into(), linear_solves),
            ])
            .unwrap_or_default(),
        diagnostics: vec![
            (
                "max_final_endpoint_error".into(),
                "absolute".into(),
                max_endpoint_error,
            ),
            (
                "final_residual_norm".into(),
                "absolute".into(),
                max_residual_norm,
            ),
        ],
    })
}

macro_rules! route {
    ($name:ident, $frontend:expr, $layout:expr, $execution:expr, $telemetry:expr) => {
        pub fn $name() -> Result<ScenarioOutput, RunnerError> {
            execute($frontend, $layout, $execution, $telemetry)
        }
    };
}

macro_rules! continuation_route {
    ($name:ident, $frontend:expr, $layout:expr, $execution:expr, $telemetry:expr) => {
        pub fn $name() -> Result<ScenarioOutput, RunnerError> {
            execute_parameter_continuation($frontend, $layout, $execution, $telemetry)
        }
    };
}

route!(
    expr_sparse_lambdify,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);

pub fn bratu32768_expr_sparse_lambdify() -> Result<ScenarioOutput, RunnerError> {
    execute_bratu(
        Frontend::Expr,
        Layout::Sparse,
        CorrectionNewtonTelemetryMode::Off,
    )
}

pub fn diagnose_bratu32768_expr_sparse_lambdify() -> Result<ScenarioOutput, RunnerError> {
    execute_bratu(
        Frontend::Expr,
        Layout::Sparse,
        CorrectionNewtonTelemetryMode::Timings,
    )
}

pub fn kithe_combustion26x256_atom_sparse_lambdify() -> Result<ScenarioOutput, RunnerError> {
    execute_kithe_combustion(CorrectionNewtonTelemetryMode::Off)
}

pub fn diagnose_kithe_combustion26x256_atom_sparse_lambdify() -> Result<ScenarioOutput, RunnerError>
{
    execute_kithe_combustion(CorrectionNewtonTelemetryMode::Timings)
}

#[cfg(test)]
mod bratu_tests {
    use super::{Frontend, Layout, execute_bratu, execute_kithe_combustion};
    use RustedSciThe::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode;

    #[test]
    fn bratu_perf_fixture_is_a_large_convergent_sparse_problem() {
        let output = execute_bratu(
            Frontend::Expr,
            Layout::Sparse,
            CorrectionNewtonTelemetryMode::Off,
        )
        .expect("32768-interval Bratu should converge through CorrectionNewton Sparse");
        for (metric, unit, value) in &output.metrics {
            println!("Bratu smoke: {metric}={value:.3} {unit}");
        }
        assert!(
            output
                .diagnostics
                .iter()
                .any(|(name, _, value)| { name == "max_residual_norm" && value.is_finite() })
        );
    }

    #[test]
    fn kithe_26_state_combustion_perf_fixture_passes_sparse_atomview() {
        let output = execute_kithe_combustion(CorrectionNewtonTelemetryMode::Off)
            .expect("KiThe 26-state fixed-grid combustion should converge via AtomView/Sparse");
        for (metric, unit, value) in &output.metrics {
            println!("KiThe combustion smoke: {metric}={value:.3} {unit}");
        }
        assert!(
            output
                .diagnostics
                .iter()
                .any(|(name, _, value)| { name == "max_residual_norm" && value.is_finite() })
        );
        assert!(
            output
                .diagnostics
                .iter()
                .any(|(name, _, value)| { name == "max_boundary_abs_error" && *value <= 1.0e-10 })
        );
    }
}
route!(
    atom_sparse_lambdify,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);
route!(
    expr_banded_lambdify,
    Frontend::Expr,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);
route!(
    atom_banded_lambdify,
    Frontend::Atom,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);
route!(
    expr_sparse_aot,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
route!(
    atom_sparse_aot,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
route!(
    expr_banded_aot,
    Frontend::Expr,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
route!(
    atom_banded_aot,
    Frontend::Atom,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
route!(
    diagnose_expr_sparse_lambdify,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
route!(
    diagnose_atom_sparse_lambdify,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
route!(
    diagnose_expr_banded_lambdify,
    Frontend::Expr,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
route!(
    diagnose_atom_banded_lambdify,
    Frontend::Atom,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
route!(
    diagnose_expr_sparse_aot,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);
route!(
    diagnose_atom_sparse_aot,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);
route!(
    diagnose_expr_banded_aot,
    Frontend::Expr,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);
route!(
    diagnose_atom_banded_aot,
    Frontend::Atom,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);

continuation_route!(
    continuation_expr_sparse_lambdify,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    continuation_atom_sparse_lambdify,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    continuation_expr_banded_lambdify,
    Frontend::Expr,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    continuation_atom_banded_lambdify,
    Frontend::Atom,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    continuation_expr_sparse_aot,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    continuation_atom_sparse_aot,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    continuation_expr_banded_aot,
    Frontend::Expr,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    continuation_atom_banded_aot,
    Frontend::Atom,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Off
);
continuation_route!(
    diagnose_continuation_expr_sparse_lambdify,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
continuation_route!(
    diagnose_continuation_atom_sparse_lambdify,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
continuation_route!(
    diagnose_continuation_expr_banded_lambdify,
    Frontend::Expr,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
continuation_route!(
    diagnose_continuation_atom_banded_lambdify,
    Frontend::Atom,
    Layout::Banded,
    Execution::Lambdify,
    CorrectionNewtonTelemetryMode::Timings
);
continuation_route!(
    diagnose_continuation_expr_sparse_aot,
    Frontend::Expr,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);
continuation_route!(
    diagnose_continuation_atom_sparse_aot,
    Frontend::Atom,
    Layout::Sparse,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);
continuation_route!(
    diagnose_continuation_expr_banded_aot,
    Frontend::Expr,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);
continuation_route!(
    diagnose_continuation_atom_banded_aot,
    Frontend::Atom,
    Layout::Banded,
    Execution::Aot,
    CorrectionNewtonTelemetryMode::Timings
);

pub fn diagnose(scenario: &Scenario) -> Result<ScenarioOutput, RunnerError> {
    match scenario.id {
        "bvp_correction_newton_parameter_continuation12000x10_expr_sparse_lambdify" => {
            diagnose_continuation_expr_sparse_lambdify()
        }
        "bvp_correction_newton_parameter_continuation12000x10_atom_sparse_lambdify" => {
            diagnose_continuation_atom_sparse_lambdify()
        }
        "bvp_correction_newton_parameter_continuation12000x10_expr_banded_lambdify" => {
            diagnose_continuation_expr_banded_lambdify()
        }
        "bvp_correction_newton_parameter_continuation12000x10_atom_banded_lambdify" => {
            diagnose_continuation_atom_banded_lambdify()
        }
        "bvp_correction_newton_parameter_continuation12000x10_expr_sparse_aot" => {
            diagnose_continuation_expr_sparse_aot()
        }
        "bvp_correction_newton_parameter_continuation12000x10_atom_sparse_aot" => {
            diagnose_continuation_atom_sparse_aot()
        }
        "bvp_correction_newton_parameter_continuation12000x10_expr_banded_aot" => {
            diagnose_continuation_expr_banded_aot()
        }
        "bvp_correction_newton_parameter_continuation12000x10_atom_banded_aot" => {
            diagnose_continuation_atom_banded_aot()
        }
        "bvp_correction_newton_kithe_combustion26x256_atom_sparse_lambdify" => {
            diagnose_kithe_combustion26x256_atom_sparse_lambdify()
        }
        "bvp_correction_newton_bratu32768_expr_sparse_lambdify" => {
            diagnose_bratu32768_expr_sparse_lambdify()
        }
        "bvp_correction_newton_combustion24000_expr_sparse_lambdify" => {
            diagnose_expr_sparse_lambdify()
        }
        "bvp_correction_newton_combustion24000_atom_sparse_lambdify" => {
            diagnose_atom_sparse_lambdify()
        }
        "bvp_correction_newton_combustion24000_expr_banded_lambdify" => {
            diagnose_expr_banded_lambdify()
        }
        "bvp_correction_newton_combustion24000_atom_banded_lambdify" => {
            diagnose_atom_banded_lambdify()
        }
        "bvp_correction_newton_combustion24000_expr_sparse_aot" => diagnose_expr_sparse_aot(),
        "bvp_correction_newton_combustion24000_atom_sparse_aot" => diagnose_atom_sparse_aot(),
        "bvp_correction_newton_combustion24000_expr_banded_aot" => diagnose_expr_banded_aot(),
        "bvp_correction_newton_combustion24000_atom_banded_aot" => diagnose_atom_banded_aot(),
        _ => Err(RunnerError::Scenario(format!(
            "no CorrectionNewton diagnostic route registered for {}",
            scenario.id
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{Execution, Frontend, Layout, execute, execute_parameter_continuation};
    use RustedSciThe::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode;

    #[test]
    fn nonlinear_combustion_gate_checks_expr_and_atom_routes() {
        for frontend in [Frontend::Expr, Frontend::Atom] {
            for layout in [Layout::Sparse, Layout::Banded] {
                for execution in [Execution::Lambdify, Execution::Aot] {
                    let wall = execute(
                        frontend,
                        layout,
                        execution,
                        CorrectionNewtonTelemetryMode::Off,
                    )
                    .expect("CorrectionNewton wall route should pass correctness checks");
                    let wall_metric = wall
                        .metrics
                        .iter()
                        .find(|(name, _, _)| name == "scenario_wall_ms")
                        .expect("telemetry-off route reports one-solve wall time");
                    println!(
                        "CorrectionNewton one-solve {frontend:?}/{layout:?}/{execution:?}: {:.3} {}",
                        wall_metric.2, wall_metric.1
                    );
                    let result = execute(
                        frontend,
                        layout,
                        execution,
                        CorrectionNewtonTelemetryMode::Counters,
                    )
                    .expect(
                        "nonlinear combustion perf-gate fixture should pass correctness checks",
                    );
                    let state_dimension = result
                        .counters
                        .iter()
                        .find(|(name, _)| name == "state_dimension")
                        .map(|(_, value)| *value);
                    assert_eq!(state_dimension, Some(6 * super::STEPS as u64));
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .all(|(_, _, value)| value.is_finite())
                    );
                    assert!(result.diagnostics.iter().any(|(name, _, value)| {
                        name == "max_boundary_abs_error" && *value <= 1.0e-12
                    }));
                    assert!(result.diagnostics.iter().any(|(name, _, value)| {
                        name == "max_bound_violation" && *value <= 1.0e-10
                    }));
                }
            }
        }
    }

    #[test]
    fn large_parameter_continuation_reuses_the_previous_accepted_profile() {
        let output = execute_parameter_continuation(
            Frontend::Atom,
            Layout::Banded,
            Execution::Lambdify,
            CorrectionNewtonTelemetryMode::Counters,
        )
        .expect("large ten-point BVP continuation should converge and match its exact profile");
        assert!(
            output
                .counters
                .iter()
                .any(|(name, count)| name == "parameter_points" && *count == 10)
        );
        assert!(
            output
                .diagnostics
                .iter()
                .any(|(name, _, error)| { name == "max_final_endpoint_error" && *error <= 1.0e-6 })
        );
    }
}
