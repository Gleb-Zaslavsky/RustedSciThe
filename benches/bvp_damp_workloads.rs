//! Compact BVP_Damp release dashboard.
//!
//! This is intentionally a bounded table driver rather than a Criterion group.
//! The existing `bvp_*` Criterion benches remain useful for statistical runs;
//! this dashboard is the compact, script-friendly matrix for release evidence.
//!
//! The default matrix covers Damped/Frozen, Dense/Sparse/Banded,
//! ExprLegacy/AtomView and Lambdify Sequential/Parallel/Auto. AOT rows are
//! opt-in through `BVP_DAMP_BENCH_ROUTES=aot,lambdify` because a cold compiler
//! lifecycle is much more expensive than the ordinary callback matrix.

use nalgebra::DMatrix;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;
use tabled::{Table, Tabled};

use RustedSciThe::Utils::test_reporting::{capture_test_block, write_test_report};
use RustedSciThe::numerical::BVP_Damp::MatrixBackend;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::{
    FrozenSolverOptions, NRBVP as FrozenBvp,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::symbolic::bvp::telemetry::{
    BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode,
};
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;

#[derive(Clone, Copy, Debug)]
enum SolverFamily {
    Damped,
    Frozen,
}

impl SolverFamily {
    fn label(self) -> &'static str {
        match self {
            Self::Damped => "Damped",
            Self::Frozen => "Frozen",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum MatrixRoute {
    Dense,
    Sparse,
    Banded,
}

impl MatrixRoute {
    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse-faer",
            Self::Banded => "Banded",
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

#[derive(Clone, Copy, Debug)]
enum Frontend {
    ExprLegacy,
    AtomView,
}

impl Frontend {
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

#[derive(Clone, Copy, Debug)]
enum Execution {
    Sequential,
    Parallel,
    Auto,
}

impl Execution {
    fn label(self) -> &'static str {
        match self {
            Self::Sequential => "Sequential",
            Self::Parallel => "Parallel",
            Self::Auto => "Auto",
        }
    }

    fn policy(self) -> BvpLambdifyExecutionPolicy {
        match self {
            Self::Sequential => BvpLambdifyExecutionPolicy::Sequential,
            Self::Parallel => BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
            Self::Auto => BvpLambdifyExecutionPolicy::Auto { min_work: 0 },
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Workload {
    Oscillator,
    Nonlinear,
}

impl Workload {
    fn label(self) -> &'static str {
        match self {
            Self::Oscillator => "oscillator",
            Self::Nonlinear => "nonlinear-exact",
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
        }
    }

    fn interval(self) -> f64 {
        match self {
            Self::Oscillator => std::f64::consts::FRAC_PI_2,
            Self::Nonlinear => 1.0,
        }
    }
}

#[derive(Clone, Debug, Tabled)]
struct DashboardRow {
    solver: String,
    workload: String,
    matrix: String,
    frontend: String,
    execution: String,
    route: String,
    n_steps: usize,
    continuation: usize,
    prepare_ms: String,
    solve_ms: String,
    continuation_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    linear_ms: String,
    factor_ms: String,
    iterations: u64,
    factorizations: u64,
    residual_calls: u64,
    jacobian_calls: u64,
    max_abs: String,
    parity_diff: String,
    status: String,
}

#[derive(Clone, Copy)]
struct RunConfig {
    family: SolverFamily,
    workload: Workload,
    matrix: MatrixRoute,
    frontend: Frontend,
    execution: Execution,
    route: &'static str,
    n_steps: usize,
    continuation: usize,
}

struct Measured {
    prepare_ms: f64,
    solve_ms: f64,
    continuation_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    factor_ms: f64,
    iterations: u64,
    factorizations: u64,
    residual_calls: u64,
    jacobian_calls: u64,
    max_abs: f64,
    generation_ms: Option<f64>,
}

fn parse_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_usizes(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .collect()
}

fn selected(values: &[String], label: &str) -> bool {
    values.iter().any(|value| value == label)
}

fn options_config(
    matrix: MatrixRoute,
    frontend: Frontend,
    execution: Execution,
    route: &'static str,
) -> GeneratedBackendConfig {
    let mut config = match (route, matrix) {
        ("aot", MatrixRoute::Sparse) => GeneratedBackendConfig::sparse_build_if_missing_release(),
        ("aot", MatrixRoute::Banded) => GeneratedBackendConfig::banded_build_if_missing_release(),
        _ => GeneratedBackendConfig::default(),
    }
    .with_matrix_backend_override(matrix.backend())
    .with_symbolic_assembly_backend(frontend.backend())
    .with_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Detailed)
    .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed);

    if route == "lambdify" {
        config = config
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_lambdify_execution_policy(execution.policy());
    }
    config
}

fn initial_guess(workload: Workload, n_steps: usize) -> DMatrix<f64> {
    DMatrix::from_fn(2, n_steps, |row, column| {
        let x = workload.interval() * column as f64 / n_steps as f64;
        match workload {
            Workload::Oscillator => {
                if row == 0 {
                    x.sin()
                } else {
                    x.cos()
                }
            }
            Workload::Nonlinear => {
                if row == 0 {
                    1.0 / (1.0 + x)
                } else {
                    -1.0 / (1.0 + x).powi(2)
                }
            }
        }
    })
}

fn boundary_conditions(workload: Workload) -> HashMap<String, Vec<(usize, f64)>> {
    match workload {
        Workload::Oscillator => HashMap::from([("y".to_string(), vec![(0, 0.0), (1, 1.0)])]),
        Workload::Nonlinear => HashMap::from([("y".to_string(), vec![(0, 1.0), (1, 0.5)])]),
    }
}

fn format_ms(value: f64) -> String {
    format!("{value:.3}")
}

fn empty_row(config: RunConfig, error: impl std::fmt::Display) -> DashboardRow {
    DashboardRow {
        solver: config.family.label().to_string(),
        workload: config.workload.label().to_string(),
        matrix: config.matrix.label().to_string(),
        frontend: config.frontend.label().to_string(),
        execution: config.execution.label().to_string(),
        route: config.route.to_string(),
        n_steps: config.n_steps,
        continuation: config.continuation,
        prepare_ms: "-".to_string(),
        solve_ms: "-".to_string(),
        continuation_ms: "-".to_string(),
        residual_ms: "-".to_string(),
        jacobian_ms: "-".to_string(),
        linear_ms: "-".to_string(),
        factor_ms: "-".to_string(),
        iterations: 0,
        factorizations: 0,
        residual_calls: 0,
        jacobian_calls: 0,
        max_abs: "-".to_string(),
        parity_diff: "-".to_string(),
        status: format!("error: {error}"),
    }
}

fn build_damped(config: RunConfig) -> NRBVP {
    let options = match config.matrix {
        MatrixRoute::Dense => DampedSolverOptions::dense_damped(),
        MatrixRoute::Sparse => DampedSolverOptions::sparse_damped(),
        MatrixRoute::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
    }
    .with_generated_backend_config(options_config(
        config.matrix,
        config.frontend,
        config.execution,
        config.route,
    ))
    .with_strategy_params(Some(SolverParams::default()))
    .with_abs_tolerance(1e-9)
    .with_rel_tolerance(HashMap::from([
        ("y".to_string(), 1e-7),
        ("z".to_string(), 1e-7),
    ]))
    .with_max_iterations(30)
    .with_bounds(HashMap::from([
        ("y".to_string(), (-10.0, 10.0)),
        ("z".to_string(), (-10.0, 10.0)),
    ]));
    let mut solver = NRBVP::new_with_options(
        config.workload.equations(),
        initial_guess(config.workload, config.n_steps),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        boundary_conditions(config.workload),
        0.0,
        config.workload.interval(),
        config.n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn build_frozen(config: RunConfig) -> FrozenBvp {
    let options = match config.matrix {
        MatrixRoute::Dense => FrozenSolverOptions::dense_frozen(),
        MatrixRoute::Sparse => FrozenSolverOptions::sparse_frozen(),
        MatrixRoute::Banded => FrozenSolverOptions::banded_frozen().with_banded_lambdify(),
    }
    .with_generated_backend_config(options_config(
        config.matrix,
        config.frontend,
        config.execution,
        config.route,
    ))
    .with_tolerance(1e-9)
    .with_max_iterations(30);
    let mut solver = FrozenBvp::new_with_options(
        config.workload.equations(),
        initial_guess(config.workload, config.n_steps),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        boundary_conditions(config.workload),
        0.0,
        config.workload.interval(),
        config.n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn run_one(config: RunConfig) -> Result<DashboardRow, String> {
    let measured: Result<Measured, String> = match config.family {
        SolverFamily::Damped => {
            let mut solver = build_damped(config);
            solver
                .try_set_params(Some(&["alpha"]))
                .map_err(|error| format!("parameter declaration: {error:?}"))?;
            solver
                .try_set_param_values(Some(vec![1.0]))
                .map_err(|error| format!("parameter binding: {error:?}"))?;
            let preparation = Instant::now();
            solver
                .try_eq_generate(None, None)
                .map_err(|error| format!("prepare: {error:?}"))?;
            let prepare_ms = preparation.elapsed().as_secs_f64() * 1e3;
            let solve_started = Instant::now();
            solver
                .try_solver_prepared()
                .map_err(|error| format!("solve: {error:?}"))?
                .ok_or_else(|| "solve returned no result".to_string())?;
            let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
            let continuation_started = Instant::now();
            for index in 1..config.continuation {
                solver
                    .try_set_param_values(Some(vec![1.0 + index as f64 * 0.1]))
                    .map_err(|error| format!("continuation bind: {error:?}"))?;
                solver
                    .try_solver_prepared()
                    .map_err(|error| format!("continuation solve: {error:?}"))?
                    .ok_or_else(|| "continuation returned no result".to_string())?;
            }
            let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
            let result = solver
                .get_result()
                .ok_or_else(|| "missing result".to_string())?;
            let stats = solver.get_statistics();
            let max_abs = result
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
            let counters = stats.telemetry.counters;
            let timings = stats.telemetry.timings;
            Ok(Measured {
                prepare_ms,
                solve_ms,
                continuation_ms,
                residual_ms: timings.residual.as_secs_f64() * 1e3,
                jacobian_ms: timings.jacobian.as_secs_f64() * 1e3,
                linear_ms: timings.linear_system.as_secs_f64() * 1e3,
                factor_ms: timings.factorization.as_secs_f64() * 1e3,
                iterations: counters.iterations,
                factorizations: counters.factorizations,
                residual_calls: counters.residual_calls,
                jacobian_calls: counters.jacobian_requests,
                max_abs,
                generation_ms: stats
                    .telemetry
                    .generation
                    .map(|value| value.total.as_secs_f64() * 1e3),
            })
        }
        SolverFamily::Frozen => {
            let mut solver = build_frozen(config);
            solver
                .try_set_params(Some(&["alpha"]))
                .map_err(|error| format!("parameter declaration: {error:?}"))?;
            solver
                .try_set_param_values(Some(vec![1.0]))
                .map_err(|error| format!("parameter binding: {error:?}"))?;
            let preparation = Instant::now();
            solver
                .try_eq_generate()
                .map_err(|error| format!("prepare: {error:?}"))?;
            let prepare_ms = preparation.elapsed().as_secs_f64() * 1e3;
            let solve_started = Instant::now();
            solver
                .try_solver_prepared()
                .map_err(|error| format!("solve: {error:?}"))?
                .ok_or_else(|| "solve returned no result".to_string())?;
            let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
            let continuation_started = Instant::now();
            for index in 1..config.continuation {
                solver
                    .try_set_param_values(Some(vec![1.0 + index as f64 * 0.1]))
                    .map_err(|error| format!("continuation bind: {error:?}"))?;
                solver
                    .try_solver_prepared()
                    .map_err(|error| format!("continuation solve: {error:?}"))?
                    .ok_or_else(|| "continuation returned no result".to_string())?;
            }
            let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
            let result = solver
                .get_result()
                .ok_or_else(|| "missing result".to_string())?;
            let stats = solver.get_statistics();
            let max_abs = result
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
            let counters = stats.telemetry.counters;
            let timings = stats.telemetry.timings;
            Ok(Measured {
                prepare_ms,
                solve_ms,
                continuation_ms,
                residual_ms: timings.residual.as_secs_f64() * 1e3,
                jacobian_ms: timings.jacobian.as_secs_f64() * 1e3,
                linear_ms: timings.linear_system.as_secs_f64() * 1e3,
                factor_ms: timings.factorization.as_secs_f64() * 1e3,
                iterations: counters.iterations,
                factorizations: counters.factorizations,
                residual_calls: counters.residual_calls,
                jacobian_calls: counters.jacobian_requests,
                max_abs,
                generation_ms: stats
                    .telemetry
                    .generation
                    .map(|value| value.total.as_secs_f64() * 1e3),
            })
        }
    };
    let measured = match measured {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let prepare_ms = measured.generation_ms.unwrap_or(measured.prepare_ms);
    Ok(DashboardRow {
        solver: config.family.label().to_string(),
        workload: config.workload.label().to_string(),
        matrix: config.matrix.label().to_string(),
        frontend: config.frontend.label().to_string(),
        execution: config.execution.label().to_string(),
        route: config.route.to_string(),
        n_steps: config.n_steps,
        continuation: config.continuation,
        prepare_ms: format_ms(prepare_ms),
        solve_ms: format_ms(measured.solve_ms),
        continuation_ms: format_ms(measured.continuation_ms),
        residual_ms: format_ms(measured.residual_ms),
        jacobian_ms: format_ms(measured.jacobian_ms),
        linear_ms: format_ms(measured.linear_ms),
        factor_ms: format_ms(measured.factor_ms),
        iterations: measured.iterations,
        factorizations: measured.factorizations,
        residual_calls: measured.residual_calls,
        jacobian_calls: measured.jacobian_calls,
        max_abs: format!("{:.3e}", measured.max_abs),
        parity_diff: "n/a".to_string(),
        status: "ok".to_string(),
    })
}

fn main() {
    let workloads = parse_list("BVP_DAMP_BENCH_WORKLOADS", "oscillator,nonlinear-exact");
    let solvers = parse_list("BVP_DAMP_BENCH_SOLVERS", "damped,frozen");
    let matrices = parse_list("BVP_DAMP_BENCH_MATRICES", "dense,sparse,banded");
    let routes = parse_list("BVP_DAMP_BENCH_ROUTES", "lambdify");
    let frontends = parse_list("BVP_DAMP_BENCH_FRONTENDS", "exprlegacy,atomview");
    let executions = parse_list("BVP_DAMP_BENCH_EXECUTIONS", "sequential,parallel,auto");
    let n_steps_values = parse_usizes("BVP_DAMP_BENCH_N_STEPS", "32");
    let continuation = parse_usize("BVP_DAMP_BENCH_CONTINUATION", 4).max(1);

    let mut rows = Vec::new();
    for family in [SolverFamily::Damped, SolverFamily::Frozen] {
        if !selected(&solvers, family.label().to_ascii_lowercase().as_str()) {
            continue;
        }
        for workload in [Workload::Oscillator, Workload::Nonlinear] {
            if !selected(&workloads, workload.label()) {
                continue;
            }
            for &n_steps in &n_steps_values {
                for matrix in [MatrixRoute::Dense, MatrixRoute::Sparse, MatrixRoute::Banded] {
                    let matrix_key = match matrix {
                        MatrixRoute::Dense => "dense",
                        MatrixRoute::Sparse => "sparse",
                        MatrixRoute::Banded => "banded",
                    };
                    if !selected(&matrices, matrix_key) {
                        continue;
                    }
                    for frontend in [Frontend::ExprLegacy, Frontend::AtomView] {
                        if !selected(&frontends, frontend.label().to_ascii_lowercase().as_str()) {
                            continue;
                        }
                        for route in ["lambdify", "aot"] {
                            if !selected(&routes, route)
                                || (route == "aot" && matches!(matrix, MatrixRoute::Dense))
                            {
                                continue;
                            }
                            let executions_for_route = if route == "aot" {
                                vec![Execution::Auto]
                            } else {
                                [Execution::Sequential, Execution::Parallel, Execution::Auto]
                                    .into_iter()
                                    .filter(|execution| {
                                        selected(
                                            &executions,
                                            execution.label().to_ascii_lowercase().as_str(),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            };
                            for execution in executions_for_route {
                                let config = RunConfig {
                                    family,
                                    workload,
                                    matrix,
                                    frontend,
                                    execution,
                                    route,
                                    n_steps,
                                    continuation,
                                };
                                let row = match catch_unwind(AssertUnwindSafe(|| run_one(config))) {
                                    Ok(Ok(row)) => row,
                                    Ok(Err(error)) => empty_row(config, error),
                                    Err(_) => empty_row(config, "panic or route failure"),
                                };
                                rows.push(row);
                            }
                        }
                    }
                }
            }
        }
    }

    let table = Table::new(&rows).to_string();
    let report = format!(
        "status: {}\n\nconfiguration: n_steps={n_steps_values:?}; continuation={continuation}; routes={routes:?}\ntelemetry: detailed; scopes are diagnostic and not additive\n\n{table}\n",
        if rows.iter().all(|row| row.status == "ok") {
            "passed"
        } else {
            "completed-with-route-errors"
        }
    );
    capture_test_block(&report);
    if let Err(error) = write_test_report("BVP_Damp_release", "bvp_damp_workloads", &report) {
        eprintln!("[BVP_Damp dashboard] unable to write report: {error}");
    }
}
