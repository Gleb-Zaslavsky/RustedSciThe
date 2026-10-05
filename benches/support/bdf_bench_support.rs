use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode, ODEsolver};
use RustedSciThe::numerical::ivp_workloads::{WorkloadKind, build_workload, dense_coupled};
use RustedSciThe::symbolic::ivp_telemetry::{IvpTelemetry, IvpTelemetrySnapshot};
use RustedSciThe::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use nalgebra::DVector;
use std::time::Instant;

pub fn workloads_from_env(variable: &str, default: &str) -> Vec<WorkloadKind> {
    std::env::var(variable)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|name| match name.trim().to_ascii_lowercase().as_str() {
            "stiff-scalar" => WorkloadKind::StiffScalar,
            "robertson" => WorkloadKind::Robertson,
            "combustion-like" => WorkloadKind::CombustionLike,
            "three-body" => WorkloadKind::ThreeBody,
            "diffusion-chain" => WorkloadKind::DiffusionChain,
            other => panic!(
                "unknown {variable} item {other:?}; expected stiff-scalar, robertson, combustion-like, three-body, diffusion-chain"
            ),
        })
        .collect()
}

pub fn dimensions(kind: WorkloadKind) -> Vec<usize> {
    if !matches!(kind, WorkloadKind::DiffusionChain) {
        return vec![0];
    }
    std::env::var("BDF_BENCH_DIFFUSION_DIMENSIONS")
        .unwrap_or_else(|_| "16,64".to_string())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("BDF_BENCH_DIFFUSION_DIMENSIONS must be comma-separated positive integers")
        })
        .collect()
}

pub fn dense_dimensions() -> Vec<usize> {
    std::env::var("BDF_BENCH_DENSE_DIMENSIONS")
        .unwrap_or_else(|_| "32,64,100".to_string())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("BDF_BENCH_DENSE_DIMENSIONS must be comma-separated positive integers")
        })
        .collect()
}

pub fn prepare_symbolic_frontend(
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> (f64, IvpTelemetrySnapshot) {
    let (equations, variables, time_variable, options) =
        symbolic_frontend_inputs(dimension, assembly, true);
    prepare_symbolic_inputs(equations, variables, time_variable, options)
}

pub fn symbolic_frontend_inputs(
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    detailed_telemetry: bool,
) -> (
    Vec<RustedSciThe::symbolic::symbolic_engine::Expr>,
    Vec<String>,
    String,
    SymbolicIvpProblemOptions,
) {
    let workload = dense_coupled(dimension);
    let telemetry = if detailed_telemetry {
        IvpTelemetry::detailed()
    } else {
        IvpTelemetry::disabled()
    };
    let mut options = SymbolicIvpProblemOptions::new()
        .with_symbolic_assembly_backend(assembly)
        .with_telemetry(telemetry);
    options = options
        .with_equation_parameters(workload.parameter_names)
        .with_equation_parameter_values(workload.parameter_values);
    (
        workload.equations,
        workload.variables,
        workload.time_variable,
        options,
    )
}

pub fn prepare_symbolic_inputs(
    equations: Vec<RustedSciThe::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    time_variable: String,
    options: SymbolicIvpProblemOptions,
) -> (f64, IvpTelemetrySnapshot) {
    let started = Instant::now();
    let prepared = prepare_symbolic_ivp_problem(equations, variables, time_variable, options)
        .expect("symbolic IVP frontend preparation");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let snapshot = prepared.telemetry.snapshot();
    drop(prepared);
    (elapsed_ms, snapshot)
}

pub fn make_solver(
    kind: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> ODEsolver {
    make_solver_with_telemetry(kind, dimension, assembly, BdfTelemetryMode::Off)
}

pub fn make_solver_with_telemetry(
    kind: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    telemetry_mode: BdfTelemetryMode,
) -> ODEsolver {
    let workload = build_workload(kind, dimension.max(1));
    let (t_bound, max_step) = match kind {
        WorkloadKind::StiffScalar => (0.05, 0.001),
        WorkloadKind::Robertson => (0.1, 0.002),
        WorkloadKind::CombustionLike => (0.01, 0.0005),
        WorkloadKind::ThreeBody => (0.01, 0.001),
        WorkloadKind::DiffusionChain => (0.01, 0.001),
    };
    let mut options = BdfSolverOptions::for_bdf(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        t_bound,
        max_step,
        1e-7,
        1e-10,
        None,
        false,
        Some(max_step),
    )
    .with_symbolic_assembly_backend(assembly)
    .with_telemetry_mode(telemetry_mode);
    if !workload.parameter_names.is_empty() {
        options = options
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }
    ODEsolver::new_with_options(options)
}

pub fn make_dense_coupled_solver(
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    telemetry_mode: BdfTelemetryMode,
) -> ODEsolver {
    let workload = dense_coupled(dimension);
    let options = BdfSolverOptions::for_bdf(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.01,
        0.001,
        1e-7,
        1e-10,
        None,
        false,
        Some(0.001),
    )
    .with_symbolic_assembly_backend(assembly)
    .with_equation_parameters(workload.parameter_names)
    .with_equation_parameter_values(workload.parameter_values)
    .with_telemetry_mode(telemetry_mode);
    ODEsolver::new_with_options(options)
}

pub fn final_state(solver: &ODEsolver) -> DVector<f64> {
    let (_, trajectory) = solver.get_result_ref();
    trajectory
        .row(trajectory.nrows() - 1)
        .transpose()
        .into_owned()
}
