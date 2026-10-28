//! Large ordered parameter loops through the public solver facades.
//!
//! The BVP cases reuse the crate's six-state combustion model. The IVP cases
//! reuse the shared three-state combustion kinetics fixture and its conserved
//! quantity `A + 2*B`. These are correctness-gated stress workloads, not
//! synthetic dimension-scaled toy systems.

use crate::error::RunnerError;
use crate::registry::ScenarioOutput;
use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpParameterLoopReport, BvpProblem,
    BvpSciSparseFactorization, BvpSolver, BvpSolverConfig,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciOptions, BvpSciStatus,
    BvpSciTelemetryMode,
};
use RustedSciThe::numerical::LSODE2::{
    IvpTelemetry, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::Radau::api::{RadauConfig, RadauProblem, RadauTelemetryMode};
use RustedSciThe::numerical::StiffAPI::StiffSolverRequest;
use RustedSciThe::numerical::ivp_workloads::{WorkloadKind, build_workload};
use RustedSciThe::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode;
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

const BVP_STATES: [&str; 6] = ["Teta", "q", "C0", "J0", "C1", "J1"];
const BVP_PARAMETER_VALUES: [f64; 10] = [
    1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045,
];
const BVP_SCI_NODES: usize = 1_000;
const DAMP_INITIAL_INTERVALS: usize = 3_000;
const IVP_PARAMETER_POINTS: usize = 10;
const IVP_T_BOUND: f64 = 0.5;
const DIFFUSION_STATES: usize = 128;

fn combustion_equations() -> Vec<Expr> {
    let unknowns = Expr::parse_vector_expression(BVP_STATES.to_vec());
    let [teta, q, c0, j0, _c1, j1] = unknowns.as_slice() else {
        unreachable!("combustion system has six state equations")
    };
    let heat_scale = Expr::parse_expression("heat_scale");
    let dt = Expr::Const(600.0);
    let t_scale = Expr::Const(600.0);
    let lambda = Expr::Const(0.07);
    let q_heat = heat_scale * Expr::Const(3000.0 * 1e3 * 0.034);
    let a = Expr::Const(1.3e5);
    let e = Expr::Const(5000.0 * 4.184);
    let molar_mass = Expr::Const(34.2 / 1000.0);
    let gas_constant = Expr::Const(8.314);
    let rho_m = Expr::Const((34.2 / 1000.0) * 2e6 / (8.314 * 1500.0));
    let q_m = Expr::Const((3e-4_f64).powi(2) / 600.0);
    let q_s = Expr::Const((3e-4_f64).powi(2));
    let pe_q = Expr::Const(0.0090168);
    let rho_d = Expr::Const(2.88e-4);
    let pe_d = Expr::Const(1.50e-3);
    let minus = Expr::Const(-1.0);
    let m_reag = Expr::Const(0.342);
    let rate = a
        * Expr::exp(-e / (gas_constant * (teta.clone() * t_scale + dt)))
        * c0.clone()
        * (rho_m.clone() / m_reag);

    vec![
        q.clone() / lambda,
        q.clone() * pe_q - q_heat * rate.clone() * q_m,
        j0.clone() / rho_d.clone(),
        j0.clone() * pe_d.clone()
            - (molar_mass.clone() * minus * rate.clone() * rho_m.clone() / molar_mass.clone())
                * q_s.clone(),
        j1.clone() / rho_d,
        j1.clone() * pe_d - (molar_mass.clone() * rate * rho_m / molar_mass) * q_s,
    ]
}

fn bvp_mesh(nodes: usize) -> Vec<f64> {
    (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect()
}

fn bvp_boundaries() -> Vec<BvpBoundaryCondition> {
    vec![
        BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: (1000.0 - 600.0) / 600.0,
        },
        BvpBoundaryCondition {
            state: 1,
            side: BvpBoundarySide::Right,
            value: 1e-10,
        },
        BvpBoundaryCondition {
            state: 2,
            side: BvpBoundarySide::Left,
            value: 1.0,
        },
        BvpBoundaryCondition {
            state: 3,
            side: BvpBoundarySide::Right,
            value: 1e-7,
        },
        BvpBoundaryCondition {
            state: 4,
            side: BvpBoundarySide::Left,
            value: 1e-3,
        },
        BvpBoundaryCondition {
            state: 5,
            side: BvpBoundarySide::Right,
            value: 1e-7,
        },
    ]
}

fn bvp_problem(nodes: usize, with_external_parameter: bool) -> Result<BvpProblem, RunnerError> {
    let problem = BvpProblem::new(
        combustion_equations(),
        BVP_STATES.iter().map(|name| (*name).to_owned()).collect(),
        "x",
        bvp_mesh(nodes),
        bvp_boundaries(),
    )
    .map_err(|error| RunnerError::Scenario(format!("BVP combustion problem: {error}")))?;
    if with_external_parameter {
        problem
            .with_external_parameters(vec!["heat_scale".into()], vec![1.0])
            .map_err(|error| {
                RunnerError::Scenario(format!("BVP combustion external parameter: {error}"))
            })
    } else {
        Ok(problem)
    }
}

fn bvp_vectors() -> Vec<Vec<f64>> {
    BVP_PARAMETER_VALUES
        .iter()
        .map(|value| vec![*value])
        .collect()
}

fn run_bvp_sci(mode: BvpSciTelemetryMode) -> Result<ScenarioOutput, RunnerError> {
    let problem = bvp_problem(BVP_SCI_NODES, true)?;
    let initial_guess = DMatrix::from_element(BVP_STATES.len(), BVP_SCI_NODES, 0.99);
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::AtomViewNative)
        .with_matrix_layout(BvpSciMatrixLayout::Sparse)
        .with_tolerance(1e-6)
        .with_limits(BVP_SCI_NODES + 64, 20, 4)
        .with_telemetry(mode);
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Sci {
            options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess,
        },
    )
    .map_err(|error| RunnerError::Scenario(format!("BVP_sci facade preparation: {error}")))?;
    let started = Instant::now();
    let report = solver
        .solve_parameter_loop(&bvp_vectors())
        .map_err(|error| RunnerError::Scenario(format!("BVP_sci facade loop: {error}")))?;
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let BvpParameterLoopReport::Sci(report) = report else {
        return Err(RunnerError::Scenario(
            "BVP_api did not preserve the BVP_sci loop report variant".into(),
        ));
    };
    let mut max_boundary_error = 0.0_f64;
    let mut max_residual = 0.0_f64;
    let mut rhs_calls = 0_u64;
    let mut jacobian_calls = 0_u64;
    let mut newton_iterations = 0_u64;
    let mut factorizations = 0_u64;
    let mut linear_solves = 0_u64;
    let mut newton_ms = 0.0_f64;
    let mut factorization_ms = 0.0_f64;
    let mut linear_solve_ms = 0.0_f64;
    let mut rhs_callback_ms = 0.0_f64;
    for point in &report.points {
        let solution = point.result.as_ref().map_err(|error| {
            RunnerError::Scenario(format!(
                "BVP_sci heat_scale={} failed: {error:?}",
                point.external_parameters[0]
            ))
        })?;
        if solution.status != BvpSciStatus::Success {
            return Err(RunnerError::Scenario(format!(
                "BVP_sci heat_scale={} did not converge: {:?}",
                point.external_parameters[0], solution.status
            )));
        }
        if solution.y.len() != solution.x.len() * BVP_STATES.len()
            || solution.y.iter().any(|value| !value.is_finite())
        {
            return Err(RunnerError::Scenario(
                "BVP_sci combustion returned invalid mesh/state storage".into(),
            ));
        }
        for condition in bvp_boundaries() {
            let node = match condition.side {
                BvpBoundarySide::Left => 0,
                BvpBoundarySide::Right => solution.x.len() - 1,
            };
            let actual = solution.y[node * BVP_STATES.len() + condition.state];
            max_boundary_error = max_boundary_error.max((actual - condition.value).abs());
        }
        max_residual = max_residual.max(solution.residual_norm);
        rhs_calls = rhs_calls.saturating_add(point.telemetry.rhs_calls);
        jacobian_calls = jacobian_calls.saturating_add(point.telemetry.jacobian_calls);
        newton_iterations = newton_iterations.saturating_add(point.telemetry.newton_iterations);
        factorizations = factorizations.saturating_add(point.telemetry.factorizations);
        linear_solves = linear_solves.saturating_add(point.telemetry.linear_solves);
        for pass in &point.telemetry.mesh_passes {
            newton_ms += pass.newton_ms.unwrap_or(0.0);
            factorization_ms += pass.factorization_ms.unwrap_or(0.0);
            linear_solve_ms += pass.linear_solve_ms.unwrap_or(0.0);
            rhs_callback_ms += pass.rhs_callback_ms.unwrap_or(0.0);
        }
    }
    if report.points.len() != BVP_PARAMETER_VALUES.len()
        || max_boundary_error > 2e-5
        || max_residual > 1e-5
    {
        return Err(RunnerError::Scenario(format!(
            "BVP_sci loop quality gate failed: points={}, boundary={max_boundary_error:e}, residual={max_residual:e}",
            report.points.len()
        )));
    }
    black_box(&report);
    Ok(ScenarioOutput {
        metrics: vec![(
            if mode == BvpSciTelemetryMode::Off {
                "scenario_wall_ms"
            } else {
                "diagnostic_loop_wall_ms"
            }
            .into(),
            if mode == BvpSciTelemetryMode::Off {
                "ms/off"
            } else {
                "ms/detailed"
            }
            .into(),
            wall_ms,
        )],
        counters: vec![
            ("parameter_points".into(), report.points.len() as u64),
            ("state_dimension".into(), BVP_STATES.len() as u64),
            ("initial_mesh_nodes".into(), BVP_SCI_NODES as u64),
            ("rhs_calls_cumulative".into(), rhs_calls),
            ("jacobian_calls_cumulative".into(), jacobian_calls),
            ("newton_iterations_cumulative".into(), newton_iterations),
            ("factorizations_cumulative".into(), factorizations),
            ("linear_solves_cumulative".into(), linear_solves),
        ],
        diagnostics: vec![
            (
                "maximum_boundary_error".into(),
                "absolute".into(),
                max_boundary_error,
            ),
            (
                "maximum_residual_norm".into(),
                "normalized".into(),
                max_residual,
            ),
            (
                "native_newton_ms_cumulative".into(),
                "ms/detailed".into(),
                newton_ms,
            ),
            (
                "native_factorization_ms_cumulative".into(),
                "ms/detailed".into(),
                factorization_ms,
            ),
            (
                "native_linear_solve_ms_cumulative".into(),
                "ms/detailed".into(),
                linear_solve_ms,
            ),
            (
                "native_rhs_callback_ms_cumulative".into(),
                "ms/detailed".into(),
                rhs_callback_ms,
            ),
        ],
        ..ScenarioOutput::default()
    })
}

pub fn bvp_sci_combustion_loop() -> Result<ScenarioOutput, RunnerError> {
    run_bvp_sci(BvpSciTelemetryMode::Off)
}

pub fn diagnose_bvp_sci_combustion_loop(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_bvp_sci(BvpSciTelemetryMode::Detailed)
}

fn damp_options(mode: BvpTelemetryMode) -> DampedSolverOptions {
    let bounds = HashMap::from([
        ("Teta".into(), (0.0, 10.0)),
        ("q".into(), (-1e20, 1e20)),
        ("C0".into(), (0.0, 1.5)),
        ("J0".into(), (-1e2, 1e2)),
        ("C1".into(), (0.0, 1.5)),
        ("J1".into(), (-1e2, 1e2)),
    ]);
    let rel_tolerance = BVP_STATES
        .iter()
        .map(|name| ((*name).to_owned(), 1e-5))
        .collect();
    let mut config = GeneratedBackendConfig::sparse_defaults()
        .with_symbolic_assembly_backend(
            RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend::AtomView,
        )
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_lambdify_telemetry_mode(if mode == BvpTelemetryMode::Detailed {
            BvpLambdifyTelemetryMode::Detailed
        } else {
            BvpLambdifyTelemetryMode::Off
        });
    config = config.with_bvp_telemetry_mode(mode);
    DampedSolverOptions::sparse_damped()
        .with_generated_backend_config(config)
        .with_abs_tolerance(1e-6)
        .with_rel_tolerance(rel_tolerance)
        .with_max_iterations(100)
        .with_bounds(bounds)
        .with_bvp_telemetry_mode(mode)
}

fn run_bvp_damped(mode: BvpTelemetryMode) -> Result<ScenarioOutput, RunnerError> {
    let problem = bvp_problem(DAMP_INITIAL_INTERVALS + 1, false)?;
    let initial_guess = DMatrix::from_element(BVP_STATES.len(), DAMP_INITIAL_INTERVALS, 0.99);
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options: damp_options(mode),
            initial_guess,
        },
    )
    .map_err(|error| RunnerError::Scenario(format!("Damped facade preparation: {error}")))?;
    solver
        .bind_parameters(vec!["heat_scale".into()], vec![1.0])
        .map_err(|error| RunnerError::Scenario(format!("Damped parameter binding: {error}")))?;

    let started = Instant::now();
    let report = solver
        .solve_parameter_loop(&bvp_vectors())
        .map_err(|error| RunnerError::Scenario(format!("Damped facade loop: {error}")))?;
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let BvpParameterLoopReport::Damped(report) = report else {
        return Err(RunnerError::Scenario(
            "BVP_api did not preserve the Damped loop report variant".into(),
        ));
    };
    if report.stop_error.is_some() || report.points.len() != BVP_PARAMETER_VALUES.len() {
        return Err(RunnerError::Scenario(format!(
            "Damped loop returned {} points or failed checkpoint recovery",
            report.points.len()
        )));
    }
    let mut max_boundary_error = 0.0_f64;
    let mut max_abs_solution = 0.0_f64;
    let mut total_iterations = 0_u64;
    let mut total_residual_calls = 0_u64;
    let mut total_jacobian_calls = 0_u64;
    let mut total_factorizations = 0_u64;
    let mut total_linear_solves = 0_u64;
    let mut total_residual_ms = 0.0_f64;
    let mut total_jacobian_ms = 0.0_f64;
    let mut total_factorization_ms = 0.0_f64;
    let mut total_linear_ms = 0.0_f64;
    let mut final_mesh_nodes = 0_usize;
    for point in &report.points {
        match &point.outcome {
            RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::DampedParameterLoopOutcome::Converged {
                mesh,
                values,
                telemetry,
            } => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(RunnerError::Scenario(
                        "Damped combustion loop returned non-finite profile values".into(),
                    ));
                }
                final_mesh_nodes = mesh.len();
                max_abs_solution = max_abs_solution.max(
                    values.iter().fold(0.0_f64, |max, value| max.max(value.abs())),
                );
                total_iterations = total_iterations
                    .saturating_add(telemetry.counters.iterations as u64);
                total_residual_calls = total_residual_calls
                    .saturating_add(telemetry.counters.residual_calls as u64);
                total_jacobian_calls = total_jacobian_calls
                    .saturating_add(telemetry.counters.jacobian_requests as u64);
                total_factorizations = total_factorizations
                    .saturating_add(telemetry.counters.factorizations as u64);
                total_linear_solves = total_linear_solves
                    .saturating_add(telemetry.counters.linear_solves as u64);
                total_residual_ms += telemetry.timings.residual.as_secs_f64() * 1e3;
                total_jacobian_ms += telemetry.timings.jacobian.as_secs_f64() * 1e3;
                total_factorization_ms += telemetry.timings.factorization.as_secs_f64() * 1e3;
                total_linear_ms += telemetry.timings.linear_system.as_secs_f64() * 1e3;
                for condition in bvp_boundaries() {
                    let node = match condition.side {
                        BvpBoundarySide::Left => 0,
                        BvpBoundarySide::Right => values.nrows() - 1,
                    };
                    max_boundary_error = max_boundary_error
                        .max((values[(node, condition.state)] - condition.value).abs());
                }
            }
            outcome => {
                return Err(RunnerError::Scenario(format!(
                    "Damped heat_scale={} did not converge: {outcome:?}",
                    point.parameters[0]
                )));
            }
        }
    }
    if max_boundary_error > 2e-5 {
        return Err(RunnerError::Scenario(format!(
            "Damped loop boundary error is too large: {max_boundary_error:e}"
        )));
    }
    black_box(&report);
    Ok(ScenarioOutput {
        metrics: vec![(
            if mode == BvpTelemetryMode::Off {
                "scenario_wall_ms"
            } else {
                "diagnostic_loop_wall_ms"
            }
            .into(),
            if mode == BvpTelemetryMode::Off {
                "ms/off"
            } else {
                "ms/detailed"
            }
            .into(),
            wall_ms,
        )],
        counters: vec![
            ("parameter_points".into(), report.points.len() as u64),
            (
                "initial_mesh_intervals".into(),
                DAMP_INITIAL_INTERVALS as u64,
            ),
            ("final_mesh_nodes".into(), final_mesh_nodes as u64),
            ("newton_iterations_cumulative".into(), total_iterations),
            ("residual_calls_cumulative".into(), total_residual_calls),
            ("jacobian_calls_cumulative".into(), total_jacobian_calls),
            ("factorizations_cumulative".into(), total_factorizations),
            ("linear_solves_cumulative".into(), total_linear_solves),
        ],
        diagnostics: vec![
            (
                "maximum_boundary_error".into(),
                "absolute".into(),
                max_boundary_error,
            ),
            (
                "maximum_absolute_state".into(),
                "state".into(),
                max_abs_solution,
            ),
            (
                "native_residual_ms_cumulative".into(),
                "ms/detailed".into(),
                total_residual_ms,
            ),
            (
                "native_jacobian_ms_cumulative".into(),
                "ms/detailed".into(),
                total_jacobian_ms,
            ),
            (
                "native_factorization_ms_cumulative".into(),
                "ms/detailed".into(),
                total_factorization_ms,
            ),
            (
                "native_linear_ms_cumulative".into(),
                "ms/detailed".into(),
                total_linear_ms,
            ),
        ],
        ..ScenarioOutput::default()
    })
}

pub fn bvp_damped_combustion_loop() -> Result<ScenarioOutput, RunnerError> {
    run_bvp_damped(BvpTelemetryMode::Off)
}

pub fn diagnose_bvp_damped_combustion_loop(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_bvp_damped(BvpTelemetryMode::Detailed)
}

fn stiff_snapshot_counts(
    telemetry: Option<&RustedSciThe::numerical::StiffAPI::StiffTelemetrySnapshot>,
) -> (u64, u64, u64, u64) {
    use RustedSciThe::numerical::StiffAPI::StiffTelemetrySnapshot;

    match telemetry {
        Some(StiffTelemetrySnapshot::Bdf { statistics, .. })
        | Some(StiffTelemetrySnapshot::BackwardEuler { statistics, .. }) => (
            statistics.accepted_steps_total as u64,
            statistics.residual_calls as u64,
            statistics.jacobian_calls as u64,
            statistics.linear_solve_attempts_total as u64,
        ),
        Some(StiffTelemetrySnapshot::Lsode2 { native, .. }) => (
            native.native_step_accepts as u64,
            native.native_residual_calls as u64,
            native.native_jacobian_calls as u64,
            native.native_linear_solve_calls as u64,
        ),
        Some(StiffTelemetrySnapshot::Radau(report)) => {
            let count = |name: &str| report.counters.get(name).copied().unwrap_or(0);
            (
                count("accepted_steps"),
                count("residual_calls"),
                count("jacobian_calls"),
                count("linear_solves"),
            )
        }
        None => (0, 0, 0, 0),
    }
}

fn combustion_parameter_vectors(parameters: &[f64]) -> Vec<Vec<f64>> {
    (0..IVP_PARAMETER_POINTS)
        .map(|index| {
            let mut point = parameters.to_vec();
            point[0] *= 0.90 + 0.02 * index as f64;
            point
        })
        .collect()
}

fn run_stiff_combustion(
    solver_name: &'static str,
    detailed: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let workload = build_workload(WorkloadKind::CombustionLike, 0);
    let parameters = workload.parameter_values.as_slice().to_vec();
    let parameter_vectors = combustion_parameter_vectors(&parameters);
    let initial_a_plus_two_b = workload.initial_state[0] + 2.0 * workload.initial_state[1];
    let t_bound = IVP_T_BOUND;
    let equations = workload.equations.clone();
    let variables = workload.variables.clone();
    let time_variable = workload.time_variable.clone();
    let parameter_names = workload.parameter_names.clone();
    let request = match solver_name {
        "bdf" => {
            let options = BdfSolverOptions::for_bdf(
                equations,
                variables,
                time_variable,
                0.0,
                workload.initial_state.clone(),
                t_bound,
                1.0e-3,
                1.0e-7,
                1.0e-10,
                None,
                false,
                None,
            )
            .with_equation_parameters(parameter_names)
            .with_equation_parameter_values(workload.parameter_values)
            .with_telemetry_mode(if detailed {
                BdfTelemetryMode::Detailed
            } else {
                BdfTelemetryMode::Off
            });
            StiffSolverRequest::Bdf { options }
        }
        "radau" => {
            let problem = RadauProblem::new(equations, variables, time_variable)
                .with_equation_parameters(parameter_names);
            let config = RadauConfig {
                t_bound,
                first_step: Some(t_bound * 1e-3),
                max_step: t_bound * 0.05,
                rtol: 1e-7,
                atol: 1e-10,
                telemetry: if detailed {
                    RadauTelemetryMode::Detailed
                } else {
                    RadauTelemetryMode::Off
                },
                ..RadauConfig::default()
            };
            StiffSolverRequest::Radau {
                problem,
                config,
                initial_state: workload.initial_state.as_slice().to_vec(),
                parameters,
            }
        }
        "lsode2" => {
            let config = Lsode2ProblemConfig::new(
                equations,
                variables,
                time_variable,
                0.0,
                workload.initial_state.clone(),
                t_bound,
                1.0e-3,
                1.0e-7,
                1.0e-10,
            )
            .with_equation_parameters(parameter_names)
            .with_equation_parameter_values(workload.parameter_values)
            .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
                execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
            })
            .with_bridge_solve()
            .with_telemetry(if detailed {
                IvpTelemetry::detailed()
            } else {
                IvpTelemetry::disabled()
            });
            StiffSolverRequest::Lsode2 {
                config,
                logging_level: None,
            }
        }
        _ => unreachable!("only the three requested StiffAPI routes are registered"),
    };
    let mut solver = request.prepare(None).map_err(|error| {
        RunnerError::Scenario(format!("StiffAPI {solver_name} prepare: {error}"))
    })?;
    let started = Instant::now();
    let report =
        solver.try_solve_parameter_loop(workload.initial_state.clone(), &parameter_vectors);
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    if report.points.len() != IVP_PARAMETER_POINTS {
        return Err(RunnerError::Scenario(format!(
            "StiffAPI {solver_name} returned {} points, expected {IVP_PARAMETER_POINTS}",
            report.points.len()
        )));
    }
    let (last_steps, last_residual_calls, last_jacobian_calls, last_linear_solves) =
        stiff_snapshot_counts(
            report
                .points
                .last()
                .and_then(|point| point.telemetry.as_ref()),
        );
    let mut checksum = 0.0;
    let mut min_species = f64::INFINITY;
    let mut max_conservation_drift = 0.0_f64;
    let mut samples = 0_u64;
    for point in &report.points {
        let solution = point.result.as_ref().map_err(|error| {
            RunnerError::Scenario(format!(
                "StiffAPI {solver_name} combustion k={} failed: {error}",
                point.parameter_values[0]
            ))
        })?;
        if solution.values.ncols() != 3
            || solution.values.nrows() < 1
            || solution.values.iter().any(|value| !value.is_finite())
        {
            return Err(RunnerError::Scenario(format!(
                "StiffAPI {solver_name} returned invalid combustion trajectory shape {:?}",
                solution.values.shape()
            )));
        }
        let final_row = solution.values.nrows() - 1;
        for row in 0..solution.values.nrows() {
            let a = solution.values[(row, 0)];
            let b = solution.values[(row, 1)];
            let temperature = solution.values[(row, 2)];
            min_species = min_species.min(a).min(b);
            max_conservation_drift =
                max_conservation_drift.max((a + 2.0 * b - initial_a_plus_two_b).abs());
            if temperature <= 0.0 {
                return Err(RunnerError::Scenario(format!(
                    "StiffAPI {solver_name} produced nonpositive temperature at trajectory row {row}"
                )));
            }
        }
        if (solution.x_mesh[solution.x_mesh.len() - 1] - t_bound).abs() > 1e-10 {
            return Err(RunnerError::Scenario(format!(
                "StiffAPI {solver_name} returned an incomplete time interval"
            )));
        }
        checksum += solution.values[(final_row, 0)]
            + solution.values[(final_row, 1)]
            + solution.values[(final_row, 2)];
        samples += solution.values.nrows() as u64;
    }
    if min_species < -2e-6 || max_conservation_drift > 2e-5 {
        return Err(RunnerError::Scenario(format!(
            "StiffAPI {solver_name} combustion invariant failed: min_species={min_species:e}, conservation_drift={max_conservation_drift:e}"
        )));
    }
    black_box(&report);
    Ok(ScenarioOutput {
        metrics: vec![(
            if detailed {
                "diagnostic_loop_wall_ms"
            } else {
                "scenario_wall_ms"
            }
            .into(),
            if detailed { "ms/detailed" } else { "ms/off" }.into(),
            wall_ms,
        )],
        counters: vec![
            ("parameter_points".into(), report.points.len() as u64),
            ("trajectory_rows_cumulative".into(), samples),
            ("last_point_accepted_steps".into(), last_steps),
            ("last_point_residual_calls".into(), last_residual_calls),
            ("last_point_jacobian_calls".into(), last_jacobian_calls),
            ("last_point_linear_solves".into(), last_linear_solves),
        ],
        diagnostics: vec![
            ("final_state_checksum".into(), "state".into(), checksum),
            (
                "minimum_reactive_species".into(),
                "state".into(),
                min_species,
            ),
            (
                "maximum_conservation_drift".into(),
                "absolute".into(),
                max_conservation_drift,
            ),
        ],
        ..ScenarioOutput::default()
    })
}

pub fn stiff_api_bdf_combustion_loop() -> Result<ScenarioOutput, RunnerError> {
    run_stiff_combustion("bdf", false)
}

pub fn diagnose_stiff_api_bdf_combustion_loop(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_stiff_combustion("bdf", true)
}

pub fn stiff_api_radau_combustion_loop() -> Result<ScenarioOutput, RunnerError> {
    run_stiff_combustion("radau", false)
}

pub fn diagnose_stiff_api_radau_combustion_loop(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_stiff_combustion("radau", true)
}

pub fn stiff_api_lsode2_combustion_loop() -> Result<ScenarioOutput, RunnerError> {
    run_stiff_combustion("lsode2", false)
}

pub fn diagnose_stiff_api_lsode2_combustion_loop(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_stiff_combustion("lsode2", true)
}

fn run_stiff_diffusion_chain(detailed: bool) -> Result<ScenarioOutput, RunnerError> {
    let workload = build_workload(WorkloadKind::DiffusionChain, DIFFUSION_STATES);
    let base_parameters = workload.parameter_values.as_slice();
    let parameter_vectors: Vec<Vec<f64>> = (0..IVP_PARAMETER_POINTS)
        .map(|index| {
            let mut point = base_parameters.to_vec();
            point[0] *= 0.95 + 0.01 * index as f64;
            point
        })
        .collect();
    let options = BdfSolverOptions::for_bdf(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state.clone(),
        0.1,
        1.0e-3,
        1.0e-7,
        1.0e-10,
        None,
        false,
        None,
    )
    .with_equation_parameters(workload.parameter_names)
    .with_equation_parameter_values(workload.parameter_values)
    .with_telemetry_mode(if detailed {
        BdfTelemetryMode::Detailed
    } else {
        BdfTelemetryMode::Off
    });
    let mut solver = StiffSolverRequest::Bdf { options }
        .prepare(None)
        .map_err(|error| {
            RunnerError::Scenario(format!("StiffAPI BDF diffusion prepare: {error}"))
        })?;
    let started = Instant::now();
    let report =
        solver.try_solve_parameter_loop(workload.initial_state.clone(), &parameter_vectors);
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    if report.points.len() != IVP_PARAMETER_POINTS {
        return Err(RunnerError::Scenario(format!(
            "StiffAPI BDF diffusion loop returned {} points, expected {IVP_PARAMETER_POINTS}",
            report.points.len()
        )));
    }
    let (last_steps, last_residual_calls, last_jacobian_calls, last_linear_solves) =
        stiff_snapshot_counts(
            report
                .points
                .last()
                .and_then(|point| point.telemetry.as_ref()),
        );
    let mut checksum = 0.0;
    let mut max_abs_state = 0.0_f64;
    let mut trajectory_rows = 0_u64;
    for point in &report.points {
        let solution = point.result.as_ref().map_err(|error| {
            RunnerError::Scenario(format!(
                "StiffAPI BDF diffusion k={} failed: {error}",
                point.parameter_values[0]
            ))
        })?;
        if solution.values.ncols() != DIFFUSION_STATES
            || solution.values.nrows() == 0
            || solution.values.iter().any(|value| !value.is_finite())
            || (solution.x_mesh[solution.x_mesh.len() - 1] - 0.1).abs() > 1e-10
        {
            return Err(RunnerError::Scenario(
                "StiffAPI BDF diffusion returned an invalid trajectory".into(),
            ));
        }
        for value in solution.values.iter() {
            max_abs_state = max_abs_state.max(value.abs());
        }
        checksum += solution.values.row(solution.values.nrows() - 1).sum();
        trajectory_rows += solution.values.nrows() as u64;
    }
    if !checksum.is_finite() || max_abs_state <= 0.0 {
        return Err(RunnerError::Scenario(
            "StiffAPI BDF diffusion failed finite-state quality checks".into(),
        ));
    }
    black_box(&report);
    Ok(ScenarioOutput {
        metrics: vec![(
            if detailed {
                "diagnostic_loop_wall_ms"
            } else {
                "scenario_wall_ms"
            }
            .into(),
            if detailed { "ms/detailed" } else { "ms/off" }.into(),
            wall_ms,
        )],
        counters: vec![
            ("parameter_points".into(), report.points.len() as u64),
            ("state_dimension".into(), DIFFUSION_STATES as u64),
            ("trajectory_rows_cumulative".into(), trajectory_rows),
            ("last_point_accepted_steps".into(), last_steps),
            ("last_point_residual_calls".into(), last_residual_calls),
            ("last_point_jacobian_calls".into(), last_jacobian_calls),
            ("last_point_linear_solves".into(), last_linear_solves),
        ],
        diagnostics: vec![
            ("final_state_checksum".into(), "state".into(), checksum),
            (
                "maximum_absolute_state".into(),
                "state".into(),
                max_abs_state,
            ),
        ],
        ..ScenarioOutput::default()
    })
}

pub fn stiff_api_bdf_diffusion_chain_loop() -> Result<ScenarioOutput, RunnerError> {
    run_stiff_diffusion_chain(false)
}

pub fn diagnose_stiff_api_bdf_diffusion_chain_loop(
    _: &crate::registry::Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_stiff_diffusion_chain(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bvp_facade_combustion_parameter_loops_pass_pointwise_quality_gates() {
        for (name, run) in [
            (
                "BVP_Damp",
                bvp_damped_combustion_loop as fn() -> Result<ScenarioOutput, RunnerError>,
            ),
            ("BVP_sci", bvp_sci_combustion_loop as fn() -> _),
        ] {
            let output = run().unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                output
                    .counters
                    .iter()
                    .find(|(name, _)| name == "parameter_points")
                    .unwrap()
                    .1,
                10
            );
            assert!(output.metrics[0].2 > 0.0);
        }
    }

    #[test]
    fn stiff_api_combustion_parameter_loops_pass_all_three_routes() {
        for (name, run) in [
            (
                "BDF",
                stiff_api_bdf_combustion_loop as fn() -> Result<ScenarioOutput, RunnerError>,
            ),
            ("Radau", stiff_api_radau_combustion_loop as fn() -> _),
            ("LSODE2", stiff_api_lsode2_combustion_loop as fn() -> _),
        ] {
            let output = run().unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                output
                    .counters
                    .iter()
                    .find(|(name, _)| name == "parameter_points")
                    .unwrap()
                    .1,
                IVP_PARAMETER_POINTS as u64
            );
            assert!(output.metrics[0].2 > 0.0);
        }
    }

    #[test]
    fn stiff_api_large_diffusion_parameter_loop_preserves_shape_and_finiteness() {
        let output = stiff_api_bdf_diffusion_chain_loop().unwrap();
        assert_eq!(
            output
                .counters
                .iter()
                .find(|(name, _)| name == "state_dimension")
                .unwrap()
                .1,
            DIFFUSION_STATES as u64
        );
        assert_eq!(
            output
                .counters
                .iter()
                .find(|(name, _)| name == "parameter_points")
                .unwrap()
                .1,
            IVP_PARAMETER_POINTS as u64
        );
    }

    #[test]
    fn facade_parameter_loop_diagnostic_passes_emit_native_counts() {
        for (name, output) in [
            (
                "BVP_sci",
                diagnose_bvp_sci_combustion_loop(&test_scenario()).unwrap(),
            ),
            (
                "BVP_Damp",
                diagnose_bvp_damped_combustion_loop(&test_scenario()).unwrap(),
            ),
        ] {
            assert!(output.metrics[0].2 > 0.0, "{name} diagnostic wall time");
            for counter in ["newton_iterations_cumulative", "factorizations_cumulative"] {
                assert!(
                    output
                        .counters
                        .iter()
                        .find(|(id, _)| id == counter)
                        .is_some_and(|(_, value)| *value > 0),
                    "{name} Detailed telemetry should report nonzero {counter}"
                );
            }
        }

        for (name, run) in [
            (
                "StiffAPI BDF",
                diagnose_stiff_api_bdf_combustion_loop
                    as fn(&crate::registry::Scenario) -> Result<ScenarioOutput, RunnerError>,
            ),
            (
                "StiffAPI Radau",
                diagnose_stiff_api_radau_combustion_loop as fn(&crate::registry::Scenario) -> _,
            ),
            (
                "StiffAPI LSODE2",
                diagnose_stiff_api_lsode2_combustion_loop as fn(&crate::registry::Scenario) -> _,
            ),
            (
                "StiffAPI diffusion",
                diagnose_stiff_api_bdf_diffusion_chain_loop as fn(&crate::registry::Scenario) -> _,
            ),
        ] {
            let output = run(&test_scenario()).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(output.metrics[0].2 > 0.0, "{name} diagnostic wall time");
            for counter in ["last_point_accepted_steps", "last_point_residual_calls"] {
                assert!(
                    output
                        .counters
                        .iter()
                        .find(|(id, _)| id == counter)
                        .is_some_and(|(_, value)| *value > 0),
                    "{name} Detailed telemetry should report nonzero {counter}"
                );
            }
        }
    }

    fn test_scenario() -> crate::registry::Scenario {
        crate::registry::Scenario {
            id: "diagnostic-test",
            solver: "facade",
            groups: &[],
            workload: "test fixture",
            frontend: "test",
            execution: "test",
            linear_algebra: "test",
            parallel_policy: "test",
            telemetry: "Detailed",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "diagnostic extraction test",
            run: bvp_sci_combustion_loop,
            diagnose: None,
        }
    }
}
