use crate::error::RunnerError;
use crate::registry::{Scenario, ScenarioOutput};
use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode, ODEsolver};
use RustedSciThe::numerical::ivp_workloads::{WorkloadKind, build_workload};
use RustedSciThe::symbolic::codegen::codegen_runtime_api::{
    recommended_dense_jacobian_chunking_for_parallelism,
    recommended_residual_chunking_for_parallelism,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use RustedSciThe::symbolic::symbolic_ivp::{IvpSymbolicAssemblyBackend, SymbolicIvpAotOptions};
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[path = "../../../benches/support/bdf_bench_support.rs"]
mod shared_bench_support;

#[derive(Clone, Copy)]
enum BdfRoute {
    LambdifySequential,
    LambdifyParallel,
    AotWhole,
    AotParallel2,
}

impl BdfRoute {
    fn lambdify_policy(self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::LambdifySequential | Self::AotWhole | Self::AotParallel2 => {
                IvpLambdifyExecutionPolicy::Sequential
            }
            Self::LambdifyParallel => IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        }
    }

    fn aot_config(self) -> Result<Option<SymbolicIvpGeneratedBackendConfig>, RunnerError> {
        let parallel = match self {
            Self::AotWhole => false,
            Self::AotParallel2 => true,
            Self::LambdifySequential | Self::LambdifyParallel => return Ok(None),
        };
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("performance")
            .join("local")
            .join("aot")
            .join("bdf");
        fs::create_dir_all(&directory)?;
        let aot_options = if parallel {
            SymbolicIvpAotOptions {
                residual_strategy: recommended_residual_chunking_for_parallelism(DIMENSION, 2),
                jacobian_strategy: recommended_dense_jacobian_chunking_for_parallelism(
                    DIMENSION, 2,
                ),
            }
        } else {
            SymbolicIvpAotOptions::default()
        };
        Ok(Some(
            SymbolicIvpGeneratedBackendConfig::new()
                .with_output_parent_dir(Some(directory))
                .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
                .with_aot_options(aot_options)
                .with_c_tcc(),
        ))
    }

    fn is_aot(self) -> bool {
        matches!(self, Self::AotWhole | Self::AotParallel2)
    }

    fn label(self) -> &'static str {
        match self {
            Self::LambdifySequential => "Lambdify Sequential",
            Self::LambdifyParallel => "Lambdify Parallel(min_work=1)",
            Self::AotWhole => "AOT Whole",
            Self::AotParallel2 => "AOT Parallel2",
        }
    }
}

const DIMENSION: usize = 256;
const T_BOUND: f64 = 0.01;
const MAX_STEP: f64 = 0.001;

fn make_diffusion_solver(
    assembly: IvpSymbolicAssemblyBackend,
    route: BdfRoute,
    telemetry: BdfTelemetryMode,
) -> Result<ODEsolver, RunnerError> {
    let workload = build_workload(WorkloadKind::DiffusionChain, DIMENSION);
    let mut options = BdfSolverOptions::for_bdf(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        T_BOUND,
        MAX_STEP,
        1.0e-7,
        1.0e-10,
        None,
        false,
        Some(MAX_STEP),
    )
    .with_symbolic_assembly_backend(assembly)
    .with_lambdify_execution_policy(route.lambdify_policy())
    .with_telemetry_mode(telemetry);
    if !workload.parameter_names.is_empty() {
        options = options
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }
    let solver = ODEsolver::new_with_options(options);
    match route.aot_config()? {
        Some(config) => Ok(solver.with_generated_backend_config(config)),
        None => Ok(solver),
    }
}

fn run_diffusion_route(
    assembly: IvpSymbolicAssemblyBackend,
    route: BdfRoute,
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let full_solve_started = Instant::now();
    let setup_started = Instant::now();
    let mut solver = make_diffusion_solver(assembly, route, BdfTelemetryMode::Off)?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;

    let solve_started = Instant::now();
    solver.try_solve().map_err(|error| {
        RunnerError::Scenario(format!("BDF {} solve failed: {error}", route.label()))
    })?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
    let scenario_wall_ms = full_solve_started.elapsed().as_secs_f64() * 1000.0;
    if solver.get_status() != "finished" {
        return Err(RunnerError::Scenario(format!(
            "BDF diffusion chain did not finish: status={}",
            solver.get_status()
        )));
    }

    let final_state = shared_bench_support::final_state(&solver);
    if final_state.len() != DIMENSION || final_state.iter().any(|value| !value.is_finite()) {
        return Err(RunnerError::Scenario(
            "BDF diffusion chain returned an invalid final state".into(),
        ));
    }
    // The fixed diffusion workload obeys a discrete maximum principle for
    // these nonnegative initial conditions and homogeneous boundary values.
    let minimum = final_state.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = final_state
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if minimum < -1e-8 || maximum > 2.0 + 1e-8 {
        return Err(RunnerError::Scenario(format!(
            "BDF diffusion bounds failed: min={minimum:e}, max={maximum:e}"
        )));
    }
    black_box(&final_state);

    if !diagnostic_only {
        return Ok(ScenarioOutput {
            metrics: vec![
                ("solver_setup_ms".into(), "ms/off".into(), setup_ms),
                ("solve_ms".into(), "ms/off".into(), solve_ms),
                ("scenario_wall_ms".into(), "ms/off".into(), scenario_wall_ms),
            ],
            native_scopes: Vec::new(),
            counters: Vec::new(),
            diagnostics: vec![
                ("final_state_min".into(), "state".into(), minimum),
                ("final_state_max".into(), "state".into(), maximum),
            ],
        });
    }

    // Gather stage timings independently; they must not contaminate the
    // telemetry-Off wall-clock gate above.
    let mut diagnostic_solver = make_diffusion_solver(assembly, route, BdfTelemetryMode::Detailed)?;
    diagnostic_solver.try_solve().map_err(|error| {
        RunnerError::Scenario(format!(
            "BDF {} timed diagnostic failed: {error}",
            route.label()
        ))
    })?;
    if diagnostic_solver.get_status() != "finished" {
        return Err(RunnerError::Scenario(format!(
            "BDF {} timed diagnostic did not finish",
            route.label()
        )));
    }
    let diagnostic_state = shared_bench_support::final_state(&diagnostic_solver);
    let telemetry_state_max_diff = final_state
        .iter()
        .zip(diagnostic_state.iter())
        .map(|(plain, timed)| (plain - timed).abs())
        .fold(0.0_f64, f64::max);
    let stats = diagnostic_solver.get_statistics();
    if stats.accepted_steps_total == 0 || telemetry_state_max_diff > 1.0e-12 {
        return Err(RunnerError::Scenario(format!(
            "BDF telemetry route invalid: accepted steps {}, max state diff={telemetry_state_max_diff:e}",
            stats.accepted_steps_total
        )));
    }
    if route.is_aot()
        && diagnostic_solver
            .aot_provenance()
            .is_none_or(|provenance| provenance.telemetry.aot_runtime_ready == 0)
    {
        return Err(RunnerError::Scenario(format!(
            "BDF {} did not report a ready AOT runtime",
            route.label()
        )));
    }
    let preparation = diagnostic_solver.preparation_timings();

    Ok(ScenarioOutput {
        metrics: vec![
            (
                "stage_backend_prepare_ms".into(),
                "ms/instrumented".into(),
                stats.backend_prepare_ms_total,
            ),
            (
                "stage_integration_loop_ms".into(),
                "ms/instrumented/nested".into(),
                stats.integration_loop_ms_total,
            ),
            (
                "stage_linear_factorization_ms".into(),
                "ms/instrumented/nested".into(),
                stats.linear_factorization_ms_total,
            ),
            (
                "stage_linear_solve_ms".into(),
                "ms/instrumented/nested".into(),
                stats.linear_solve_ms_total,
            ),
            (
                "stage_residual_ms".into(),
                "ms/instrumented".into(),
                stats.residual_ms_total,
            ),
            (
                "stage_jacobian_ms".into(),
                "ms/instrumented".into(),
                stats.jacobian_ms_total,
            ),
            (
                "stage_prepare_total_ms".into(),
                "ms/instrumented".into(),
                preparation.total_ms,
            ),
        ],
        native_scopes: Vec::new(),
        counters: vec![
            ("state_dimension".into(), final_state.len() as u64),
            ("accepted_steps".into(), stats.accepted_steps_total as u64),
            (
                "candidate_steps".into(),
                stats.candidate_step_attempts_total as u64,
            ),
            (
                "rejected_steps".into(),
                stats.rejected_step_attempts_total as u64,
            ),
            ("nfev".into(), stats.bdf_nfev_total as u64),
            ("njev".into(), stats.bdf_njev_total as u64),
            ("nlu".into(), stats.bdf_nlu_total as u64),
        ],
        diagnostics: vec![
            ("final_state_min".into(), "state".into(), minimum),
            ("final_state_max".into(), "state".into(), maximum),
            (
                "telemetry_state_max_diff".into(),
                "absolute".into(),
                telemetry_state_max_diff,
            ),
        ],
    })
}

macro_rules! declare_diffusion_route {
    ($name:ident, $assembly:expr, $route:expr) => {
        pub fn $name() -> Result<ScenarioOutput, RunnerError> {
            run_diffusion_route($assembly, $route, false)
        }
    };
}

pub fn diagnose(scenario: &Scenario) -> Result<ScenarioOutput, RunnerError> {
    let assembly = match scenario.frontend {
        "ExprLegacy" => IvpSymbolicAssemblyBackend::ExprLegacy,
        "AtomView" => IvpSymbolicAssemblyBackend::AtomView,
        other => {
            return Err(RunnerError::Scenario(format!(
                "unsupported BDF frontend {other}"
            )));
        }
    };
    let route = match (scenario.execution, scenario.parallel_policy) {
        ("Lambdify", "Sequential") => BdfRoute::LambdifySequential,
        ("Lambdify", "Parallel(min_work=1)") => BdfRoute::LambdifyParallel,
        ("AOT(warm-cache)", "Whole(default chunks)") => BdfRoute::AotWhole,
        ("AOT(warm-cache)", "Parallel2") => BdfRoute::AotParallel2,
        _ => {
            return Err(RunnerError::Scenario(format!(
                "unsupported BDF route {}",
                scenario.id
            )));
        }
    };
    run_diffusion_route(assembly, route, true)
}

declare_diffusion_route!(
    diffusion_expr_lambdify_seq,
    IvpSymbolicAssemblyBackend::ExprLegacy,
    BdfRoute::LambdifySequential
);
declare_diffusion_route!(
    diffusion_atom_lambdify_seq,
    IvpSymbolicAssemblyBackend::AtomView,
    BdfRoute::LambdifySequential
);
declare_diffusion_route!(
    diffusion_expr_lambdify_parallel,
    IvpSymbolicAssemblyBackend::ExprLegacy,
    BdfRoute::LambdifyParallel
);
declare_diffusion_route!(
    diffusion_atom_lambdify_parallel,
    IvpSymbolicAssemblyBackend::AtomView,
    BdfRoute::LambdifyParallel
);
declare_diffusion_route!(
    diffusion_expr_aot_whole,
    IvpSymbolicAssemblyBackend::ExprLegacy,
    BdfRoute::AotWhole
);
declare_diffusion_route!(
    diffusion_atom_aot_whole,
    IvpSymbolicAssemblyBackend::AtomView,
    BdfRoute::AotWhole
);
declare_diffusion_route!(
    diffusion_expr_aot_parallel2,
    IvpSymbolicAssemblyBackend::ExprLegacy,
    BdfRoute::AotParallel2
);
declare_diffusion_route!(
    diffusion_atom_aot_parallel2,
    IvpSymbolicAssemblyBackend::AtomView,
    BdfRoute::AotParallel2
);

#[cfg(test)]
mod tests {
    use super::diffusion_expr_lambdify_seq;

    #[test]
    fn extended_diffusion_gate_solves_a_256_state_problem() {
        let output = diffusion_expr_lambdify_seq().expect("BDF extended diffusion solve");
        let wall_ms = output
            .metrics
            .iter()
            .find(|(id, _, _)| id == "scenario_wall_ms")
            .expect("wall gate metric")
            .2;
        assert!(wall_ms.is_finite() && wall_ms > 0.0);
        assert!(
            output
                .diagnostics
                .iter()
                .any(|(id, _, value)| id == "final_state_min" && value.is_finite())
        );
    }
}
