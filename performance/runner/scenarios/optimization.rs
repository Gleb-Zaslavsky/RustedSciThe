//! Matched symbolic fitting routes for the optimization solver family.
//!
//! These are numerical correctness/timing adapters over the public APIs, not
//! replacements for the solver-specific Criterion and STORY matrices. AOT is
//! intentionally absent: these fitting routes currently expose Lambdify only.

use crate::error::RunnerError;
use crate::registry::{Scenario, ScenarioOutput};
use RustedSciThe::numerical::Nonlinear_systems::engine::SolveStatistics;
use RustedSciThe::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresTelemetryMode, LevenbergMarquardt,
};
use RustedSciThe::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;
use RustedSciThe::numerical::SolverTelemetryMode;
use RustedSciThe::numerical::optimization::gavin_fitting::GavinSymbolicFitting;
use RustedSciThe::numerical::optimization::lm_gavin::GavinLmUpdatePolicy;
use RustedSciThe::numerical::optimization::sym_fitting::Fitting;
use RustedSciThe::numerical::optimization::universal_fitting::{
    Method, UniversalFitting, UniversalFittingResult,
};
use RustedSciThe::numerical::optimization::varpro::symbolic::SymbolicVarProBuilder;
use nalgebra::DVector;
use std::hint::black_box;
use std::time::Instant;

// The nonlinear fitting routes use 20k observations. VarPro is substantially
// cheaper per observation, so it needs a larger workload to stay above timer
// noise and provide a useful regression gate.
const SAMPLES: usize = 20_001;
const VARPRO_SAMPLES: usize = 200_010;
const EXPECTED: [f64; 4] = [2.0, 0.5, 1.0, 2.0];
const FAMILY_GAVIN: u8 = 0;
const FAMILY_FITTING: u8 = 1;
const FAMILY_VARPRO: u8 = 2;
const FAMILY_UNIVERSAL_LM: u8 = 3;
const FAMILY_UNIVERSAL_GAVIN: u8 = 4;
const FAMILY_UNIVERSAL_VARPRO: u8 = 5;
const FRONTEND_EXPR: u8 = 0;
const FRONTEND_ATOM: u8 = 1;
const POLICY_SEQUENTIAL: u8 = 0;
const POLICY_PARALLEL: u8 = 1;
const GAVIN_MARQUARDT: u8 = 0;
const GAVIN_QUADRATIC: u8 = 1;
const GAVIN_NIELSEN: u8 = 2;

const MODEL: &str = "a*exp(-x/tau)+c*sin(x/omega)";

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
            _ => unreachable!("registered optimization frontend id"),
        }
    }

    fn public(self) -> SymbolicLambdifyFrontend {
        match self {
            Self::ExprLegacy => SymbolicLambdifyFrontend::ExprLegacy,
            Self::AtomView => SymbolicLambdifyFrontend::AtomViewNative,
        }
    }
}

#[derive(Clone, Copy)]
enum CallbackPolicy {
    Sequential,
    Parallel,
}

impl CallbackPolicy {
    fn from_id(id: u8) -> Self {
        match id {
            POLICY_SEQUENTIAL => Self::Sequential,
            POLICY_PARALLEL => Self::Parallel,
            _ => unreachable!("registered optimization callback policy id"),
        }
    }

    fn public(
        self,
    ) -> RustedSciThe::numerical::Nonlinear_systems::symbolic::LambdifyExecutionPolicy {
        use RustedSciThe::numerical::Nonlinear_systems::symbolic::LambdifyExecutionPolicy;
        match self {
            Self::Sequential => LambdifyExecutionPolicy::Sequential,
            Self::Parallel => LambdifyExecutionPolicy::Parallel { min_work: 1 },
        }
    }
}

#[derive(Clone, Copy)]
enum GavinPolicy {
    Marquardt,
    Quadratic,
    Nielsen,
}

impl GavinPolicy {
    fn from_id(id: u8) -> Self {
        match id {
            GAVIN_MARQUARDT => Self::Marquardt,
            GAVIN_QUADRATIC => Self::Quadratic,
            GAVIN_NIELSEN => Self::Nielsen,
            _ => unreachable!("registered Gavin update policy id"),
        }
    }

    fn public(self) -> GavinLmUpdatePolicy {
        match self {
            Self::Marquardt => GavinLmUpdatePolicy::Marquardt,
            Self::Quadratic => GavinLmUpdatePolicy::Quadratic,
            Self::Nielsen => GavinLmUpdatePolicy::Nielsen,
        }
    }
}

struct Data {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Data {
    fn reference_fit() -> Self {
        Self::reference_fit_with_samples(SAMPLES)
    }

    fn reference_varpro_fit() -> Self {
        Self::reference_fit_with_samples(VARPRO_SAMPLES)
    }

    fn reference_fit_with_samples(samples: usize) -> Self {
        let x = (0..samples)
            .map(|index| 20.0 * index as f64 / (samples - 1) as f64)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|&sample| 2.0 * (-sample / 0.5_f64).exp() + (sample / 2.0_f64).sin())
            .collect();
        Self { x, y }
    }
}

struct FitSnapshot {
    parameters: Vec<f64>,
    r_squared: Option<f64>,
    counters: Vec<(String, u64)>,
    diagnostics: Vec<(String, String, f64)>,
}

fn run_cell<
    const FAMILY: u8,
    const FRONTEND_ID: u8,
    const POLICY_ID: u8,
    const GAVIN_POLICY_ID: u8,
>(
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let frontend = Frontend::from_id(FRONTEND_ID);
    let callback_policy = CallbackPolicy::from_id(POLICY_ID);
    let gavin_policy = GavinPolicy::from_id(GAVIN_POLICY_ID);
    let data = if matches!(FAMILY, FAMILY_VARPRO | FAMILY_UNIVERSAL_VARPRO) {
        Data::reference_varpro_fit()
    } else {
        Data::reference_fit()
    };

    let off_started = Instant::now();
    let off = run_once::<FAMILY>(&data, frontend, callback_policy, gavin_policy, false)?;
    let scenario_wall_ms = off_started.elapsed().as_secs_f64() * 1_000.0;
    black_box(&off.parameters);

    if !diagnostic_only {
        let max_parameter_error = expected_error::<FAMILY>(&off.parameters);
        if max_parameter_error > 2.0e-2
            || off.r_squared.is_none_or(|r2| !r2.is_finite() || r2 < 0.99)
        {
            return Err(RunnerError::Scenario(format!(
                "optimization gate route failed correctness: parameter_error={max_parameter_error:e}, r_squared={:?}",
                off.r_squared
            )));
        }
        return Ok(ScenarioOutput {
            metrics: vec![("scenario_wall_ms".into(), "ms/off".into(), scenario_wall_ms)],
            native_scopes: Vec::new(),
            counters: Vec::new(),
            diagnostics: vec![
                (
                    "known_parameter_max_error".into(),
                    "absolute".into(),
                    max_parameter_error,
                ),
                (
                    "r_squared".into(),
                    "unitless".into(),
                    off.r_squared.unwrap_or(f64::NAN),
                ),
            ],
        });
    }

    let detailed = run_once::<FAMILY>(&data, frontend, callback_policy, gavin_policy, true)?;
    let max_parameter_diff = off
        .parameters
        .iter()
        .zip(&detailed.parameters)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let max_parameter_error = expected_error::<FAMILY>(&detailed.parameters);
    if max_parameter_diff > 1.0e-8 || max_parameter_error > 2.0e-2 {
        return Err(RunnerError::Scenario(format!(
            "optimization route failed: off/detailed diff={max_parameter_diff:e}, known-parameter error={max_parameter_error:e}"
        )));
    }
    if detailed
        .r_squared
        .is_none_or(|r2| !r2.is_finite() || r2 < 0.99)
    {
        return Err(RunnerError::Scenario(format!(
            "optimization route produced invalid fit quality: r_squared={:?}",
            detailed.r_squared
        )));
    }

    let metrics = detailed.diagnostics;
    Ok(ScenarioOutput {
        metrics,
        native_scopes: Vec::new(),
        counters: detailed.counters,
        diagnostics: vec![
            (
                "off_vs_detailed_max_parameter_diff".into(),
                "absolute".into(),
                max_parameter_diff,
            ),
            (
                "known_parameter_max_error".into(),
                "absolute".into(),
                max_parameter_error,
            ),
            (
                "r_squared".into(),
                "unitless".into(),
                detailed.r_squared.expect("validated R squared above"),
            ),
        ],
    })
}

fn run_cell_gate<
    const FAMILY: u8,
    const FRONTEND_ID: u8,
    const POLICY_ID: u8,
    const GAVIN_POLICY_ID: u8,
>() -> Result<ScenarioOutput, RunnerError> {
    run_cell::<FAMILY, FRONTEND_ID, POLICY_ID, GAVIN_POLICY_ID>(false)
}

fn diagnose<
    const FAMILY: u8,
    const FRONTEND_ID: u8,
    const POLICY_ID: u8,
    const GAVIN_POLICY_ID: u8,
>(
    _: &Scenario,
) -> Result<ScenarioOutput, RunnerError> {
    run_cell::<FAMILY, FRONTEND_ID, POLICY_ID, GAVIN_POLICY_ID>(true)
}

fn expected_error<const FAMILY: u8>(parameters: &[f64]) -> f64 {
    let expected: &[f64] = if matches!(FAMILY, FAMILY_VARPRO | FAMILY_UNIVERSAL_VARPRO) {
        // VarPro reports nonlinear taus first, followed by linear coefficients.
        &[0.5, 2.0, 2.0, 1.0]
    } else {
        &EXPECTED
    };
    parameters
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max)
}

fn run_once<const FAMILY: u8>(
    data: &Data,
    frontend: Frontend,
    callback_policy: CallbackPolicy,
    gavin_policy: GavinPolicy,
    detailed: bool,
) -> Result<FitSnapshot, RunnerError> {
    if matches!(FAMILY, FAMILY_GAVIN | FAMILY_UNIVERSAL_GAVIN) {
        return run_gavin::<FAMILY>(data, frontend, callback_policy, gavin_policy, detailed);
    }
    if FAMILY == FAMILY_VARPRO {
        return run_varpro(data, detailed);
    }
    if FAMILY == FAMILY_UNIVERSAL_VARPRO {
        return run_universal_varpro(data, detailed);
    }
    if FAMILY == FAMILY_FITTING {
        return run_fitting(data, frontend, callback_policy, detailed);
    }
    run_universal_lm(data, frontend, callback_policy, detailed)
}

fn run_gavin<const FAMILY: u8>(
    data: &Data,
    frontend: Frontend,
    callback_policy: CallbackPolicy,
    gavin_policy: GavinPolicy,
    detailed: bool,
) -> Result<FitSnapshot, RunnerError> {
    let telemetry = if detailed {
        SolverTelemetryMode::Detailed
    } else {
        SolverTelemetryMode::Off
    };
    if FAMILY == FAMILY_UNIVERSAL_GAVIN {
        let universal = UniversalFitting::new()
            .with_method(Method::GAVIN)
            .with_data(data.x.clone(), data.y.clone())
            .with_equation_str(MODEL)
            .with_arg("x")
            .with_unknowns(["a", "tau", "c", "omega"])
            .with_initial_guess(vec![1.9, 0.55, 0.9, 1.9])
            .with_frontend(frontend.public())
            .with_lambdify_execution_policy(callback_policy.public())
            .with_gavin_update_policy(gavin_policy.public())
            .with_gavin_telemetry(telemetry);
        // The generic LM telemetry channel must stay Off for Gavin; its own
        // controller uses SolverTelemetryMode through the dedicated setter.
        let universal = universal.with_telemetry(LeastSquaresTelemetryMode::Off);
        return run_universal_gavin(universal.build().map_err(scenario_error)?, detailed);
    }
    let builder = GavinSymbolicFitting::new()
        .try_with_equation(MODEL)
        .map_err(scenario_error)?
        .with_data(data.x.clone(), data.y.clone())
        .with_sample_parameter("x")
        .with_unknowns(["a", "tau", "c", "omega"])
        .with_initial_guess(vec![1.9, 0.55, 0.9, 1.9])
        .with_frontend(frontend.public())
        .with_lambdify_execution_policy(callback_policy.public())
        .with_update_policy(gavin_policy.public())
        .with_telemetry(telemetry)
        .with_preparation_telemetry(if detailed {
            RustedSciThe::numerical::Nonlinear_systems::symbolic::PreparationTelemetryMode::Collect
        } else {
            RustedSciThe::numerical::Nonlinear_systems::symbolic::PreparationTelemetryMode::Disabled
        });
    let fit = builder.try_solve().map_err(scenario_error)?;
    if !fit.fit.termination.is_converged() {
        return Err(RunnerError::Scenario(format!(
            "Gavin termination was {:?}",
            fit.fit.termination
        )));
    }
    let parameters = fit.fit.parameters.iter().copied().collect::<Vec<_>>();
    let telemetry = fit.telemetry;
    if detailed && telemetry.is_none() {
        return Err(RunnerError::Scenario(
            "Gavin Detailed route omitted requested telemetry".into(),
        ));
    }
    if telemetry
        .as_ref()
        .is_some_and(|stats| stats.telemetry_mode != telemetry_mode(detailed))
    {
        return Err(RunnerError::Scenario(
            "Gavin telemetry mode did not match request".into(),
        ));
    }
    let counters = telemetry.as_ref().map_or_else(Vec::new, |telemetry| {
        vec![
            ("iterations".into(), telemetry.iterations as u64),
            (
                "objective_evaluations".into(),
                telemetry.objective_evaluations as u64,
            ),
            (
                "analytic_jacobian_evaluations".into(),
                telemetry.analytic_jacobian_evaluations as u64,
            ),
            (
                "finite_difference_jacobians".into(),
                telemetry.finite_difference_jacobians as u64,
            ),
            ("accepted_steps".into(), telemetry.accepted_steps as u64),
            ("rejected_steps".into(), telemetry.rejected_steps as u64),
            ("linear_solves".into(), telemetry.linear_solves as u64),
        ]
    });
    let mut diagnostics = telemetry.as_ref().map_or_else(Vec::new, |telemetry| {
        vec![
            (
                "solver_native_total_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.total_time.as_secs_f64() * 1_000.0,
            ),
            (
                "objective_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.objective_time.as_secs_f64() * 1_000.0,
            ),
            (
                "jacobian_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.analytic_jacobian_time.as_secs_f64() * 1_000.0,
            ),
            (
                "linearization_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.linearization_time.as_secs_f64() * 1_000.0,
            ),
            (
                "linear_solve_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.linear_solve_time.as_secs_f64() * 1_000.0,
            ),
        ]
    });
    if detailed {
        diagnostics.push((
            "symbolic_preparation_ms".into(),
            "ms/detailed/preparation".into(),
            fit.preparation.preparation_duration.as_secs_f64() * 1_000.0,
        ));
    }
    Ok(FitSnapshot {
        parameters,
        r_squared: fit.fit.r_squared,
        counters,
        diagnostics,
    })
}

fn run_fitting(
    data: &Data,
    frontend: Frontend,
    callback_policy: CallbackPolicy,
    detailed: bool,
) -> Result<FitSnapshot, RunnerError> {
    let mut fit = Fitting::new()
        .try_with_equation_str(MODEL)
        .map_err(scenario_error)?
        .with_data(data.x.clone(), data.y.clone())
        .with_arg("x".into())
        .with_unknowns(["a", "tau", "c", "omega"].map(str::to_owned).to_vec())
        .with_initial_guess(vec![1.9, 0.55, 0.9, 1.9])
        .with_symbolic_frontend(frontend.public())
        .with_lambdify_execution_policy(callback_policy.public())
        .with_tolerance(1.0e-10)
        .with_f_tolerance(1.0e-10)
        .with_g_tolerance(1.0e-10)
        .with_max_iterations(1_000)
        .with_telemetry(if detailed {
            LeastSquaresTelemetryMode::Detailed
        } else {
            LeastSquaresTelemetryMode::Off
        });
    fit = fit.try_build().map_err(scenario_error)?;
    let parameters = parameters_from_map(
        fit.solution_map()
            .ok_or_else(|| RunnerError::Scenario("Fitting returned no solution map".into()))?,
        &["a", "tau", "c", "omega"],
    )?;
    let stats = fit.last_statistics().cloned().unwrap_or_default();
    if detailed && !stats.timings_collected {
        return Err(RunnerError::Scenario(
            "FittingSolver Detailed route omitted requested timings".into(),
        ));
    }
    let r_squared = fit.r_squared();
    Ok(snapshot_from_statistics(
        parameters, r_squared, stats, detailed,
    ))
}

fn run_universal_lm(
    data: &Data,
    frontend: Frontend,
    callback_policy: CallbackPolicy,
    detailed: bool,
) -> Result<FitSnapshot, RunnerError> {
    let result = universal_common(data, frontend, callback_policy)
        .with_method(Method::LM)
        .with_telemetry(if detailed {
            LeastSquaresTelemetryMode::Detailed
        } else {
            LeastSquaresTelemetryMode::Off
        })
        .build()
        .map_err(scenario_error)?;
    let UniversalFittingResult::LM(fit) = result else {
        return Err(RunnerError::Scenario(
            "UniversalFitting LM selector returned the wrong method".into(),
        ));
    };
    let parameters = parameters_from_map(
        fit.solution_map()
            .ok_or_else(|| RunnerError::Scenario("Universal LM returned no solution".into()))?,
        &["a", "tau", "c", "omega"],
    )?;
    let stats = fit.last_statistics().cloned().unwrap_or_default();
    if detailed && !stats.timings_collected {
        return Err(RunnerError::Scenario(
            "UniversalFitting LM Detailed route omitted requested timings".into(),
        ));
    }
    Ok(snapshot_from_statistics(
        parameters,
        fit.r_squared(),
        stats,
        detailed,
    ))
}

fn run_universal_gavin(
    result: UniversalFittingResult,
    detailed: bool,
) -> Result<FitSnapshot, RunnerError> {
    let UniversalFittingResult::GAVIN(fit) = result else {
        return Err(RunnerError::Scenario(
            "UniversalFitting Gavin selector returned the wrong method".into(),
        ));
    };
    if !fit.fit.termination.is_converged() {
        return Err(RunnerError::Scenario(format!(
            "Universal Gavin termination was {:?}",
            fit.fit.termination
        )));
    }
    let parameters = fit.fit.parameters.iter().copied().collect::<Vec<_>>();
    let telemetry = fit.telemetry;
    if detailed && telemetry.is_none() {
        return Err(RunnerError::Scenario(
            "Universal Gavin Detailed route omitted telemetry".into(),
        ));
    }
    if telemetry
        .as_ref()
        .is_some_and(|stats| stats.telemetry_mode != telemetry_mode(detailed))
    {
        return Err(RunnerError::Scenario(
            "Universal Gavin telemetry mode did not match request".into(),
        ));
    }
    let counters = telemetry.as_ref().map_or_else(Vec::new, |telemetry| {
        vec![
            ("iterations".into(), telemetry.iterations as u64),
            (
                "objective_evaluations".into(),
                telemetry.objective_evaluations as u64,
            ),
            (
                "analytic_jacobian_evaluations".into(),
                telemetry.analytic_jacobian_evaluations as u64,
            ),
            (
                "finite_difference_jacobians".into(),
                telemetry.finite_difference_jacobians as u64,
            ),
            ("accepted_steps".into(), telemetry.accepted_steps as u64),
            ("rejected_steps".into(), telemetry.rejected_steps as u64),
            ("linear_solves".into(), telemetry.linear_solves as u64),
        ]
    });
    let diagnostics = telemetry.as_ref().map_or_else(Vec::new, |telemetry| {
        vec![
            (
                "solver_native_total_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.total_time.as_secs_f64() * 1_000.0,
            ),
            (
                "objective_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.objective_time.as_secs_f64() * 1_000.0,
            ),
            (
                "jacobian_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.analytic_jacobian_time.as_secs_f64() * 1_000.0,
            ),
            (
                "linearization_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.linearization_time.as_secs_f64() * 1_000.0,
            ),
            (
                "linear_solve_ms".into(),
                "ms/detailed/solve".into(),
                telemetry.linear_solve_time.as_secs_f64() * 1_000.0,
            ),
        ]
    });
    Ok(FitSnapshot {
        parameters,
        r_squared: fit.fit.r_squared,
        counters,
        diagnostics,
    })
}

fn run_varpro(data: &Data, detailed: bool) -> Result<FitSnapshot, RunnerError> {
    let telemetry = if detailed {
        LeastSquaresTelemetryMode::Detailed
    } else {
        LeastSquaresTelemetryMode::Off
    };
    let builder = SymbolicVarProBuilder::new("x")
        .with_parameters(["tau", "omega"])
        .with_initial_parameters(vec![0.55, 1.9])
        .with_data(
            DVector::from_vec(data.x.clone()),
            DVector::from_vec(data.y.clone()),
        )
        .with_basis_str("tau", "exp(-x/tau)")
        .with_basis_str("omega", "sin(x/omega)");
    let solver = LevenbergMarquardt::new()
        .with_gtol(0.0)
        .with_telemetry(telemetry)
        .try_with_max_iterations(1_000)
        .map_err(scenario_error)?;
    let fit = builder.solve_with_solver(solver).map_err(scenario_error)?;
    if !fit.minimization_report.termination.was_successful() {
        return Err(RunnerError::Scenario(format!(
            "VarPro termination was {:?}",
            fit.minimization_report.termination
        )));
    }
    let parameter_names: &[&str] = &["tau", "omega", "c0", "c1"];
    let parameters = parameters_from_map(
        fit.solution_map()
            .ok_or_else(|| RunnerError::Scenario("VarPro returned no solution map".into()))?,
        parameter_names,
    )?;
    let stats = fit.minimization_report.statistics.clone();
    if detailed && !stats.timings_collected {
        return Err(RunnerError::Scenario(
            "VarPro Detailed route omitted requested timings".into(),
        ));
    }
    let r_squared = fit.r_squared();
    Ok(snapshot_from_statistics(
        parameters, r_squared, stats, detailed,
    ))
}

fn run_universal_varpro(data: &Data, detailed: bool) -> Result<FitSnapshot, RunnerError> {
    let result = UniversalFitting::new()
        .with_method(Method::VARPRO)
        .with_data(data.x.clone(), data.y.clone())
        .with_arg("x")
        .with_unknowns(["tau", "omega"])
        .with_initial_guess(vec![0.55, 1.9])
        .with_basis_str("tau", "exp(-x/tau)")
        .with_basis_str("omega", "sin(x/omega)")
        .with_telemetry(if detailed {
            LeastSquaresTelemetryMode::Detailed
        } else {
            LeastSquaresTelemetryMode::Off
        })
        .build()
        .map_err(scenario_error)?;
    let UniversalFittingResult::VARPRO(fit) = result else {
        return Err(RunnerError::Scenario(
            "UniversalFitting VarPro selector returned the wrong method".into(),
        ));
    };
    if !fit.minimization_report.termination.was_successful() {
        return Err(RunnerError::Scenario(format!(
            "Universal VarPro termination was {:?}",
            fit.minimization_report.termination
        )));
    }
    let parameters = parameters_from_map(
        fit.solution_map()
            .ok_or_else(|| RunnerError::Scenario("Universal VarPro returned no solution".into()))?,
        &["tau", "omega", "c0", "c1"],
    )?;
    if detailed && !fit.minimization_report.statistics.timings_collected {
        return Err(RunnerError::Scenario(
            "UniversalFitting VarPro Detailed route omitted requested timings".into(),
        ));
    }
    Ok(snapshot_from_statistics(
        parameters,
        fit.r_squared(),
        fit.minimization_report.statistics.clone(),
        detailed,
    ))
}

fn telemetry_mode(detailed: bool) -> SolverTelemetryMode {
    if detailed {
        SolverTelemetryMode::Detailed
    } else {
        SolverTelemetryMode::Off
    }
}

fn universal_common(
    data: &Data,
    frontend: Frontend,
    callback_policy: CallbackPolicy,
) -> UniversalFitting {
    UniversalFitting::new()
        .with_data(data.x.clone(), data.y.clone())
        .with_equation_str(MODEL)
        .with_arg("x")
        .with_unknowns(["a", "tau", "c", "omega"])
        .with_initial_guess(vec![1.9, 0.55, 0.9, 1.9])
        .with_frontend(frontend.public())
        .with_lambdify_execution_policy(callback_policy.public())
}

fn snapshot_from_statistics(
    parameters: Vec<f64>,
    r_squared: Option<f64>,
    stats: SolveStatistics,
    detailed: bool,
) -> FitSnapshot {
    let counters = vec![
        ("iterations".into(), stats.iterations as u64),
        (
            "residual_evaluations".into(),
            stats.residual_evaluations as u64,
        ),
        (
            "jacobian_evaluations".into(),
            stats.jacobian_evaluations as u64,
        ),
        (
            "linear_factorizations".into(),
            stats.linear_factorizations as u64,
        ),
        ("linear_solves".into(), stats.linear_solves as u64),
    ];
    let diagnostics = if detailed {
        vec![
            (
                "residual_ms".into(),
                "ms/detailed/solve".into(),
                stats.residual_duration.as_secs_f64() * 1_000.0,
            ),
            (
                "jacobian_ms".into(),
                "ms/detailed/solve".into(),
                stats.jacobian_duration.as_secs_f64() * 1_000.0,
            ),
            (
                "linear_factorization_ms".into(),
                "ms/detailed/solve".into(),
                stats.linear_factorization_duration.as_secs_f64() * 1_000.0,
            ),
            (
                "linear_system_solve_ms".into(),
                "ms/detailed/solve".into(),
                stats.linear_system_solve_duration.as_secs_f64() * 1_000.0,
            ),
            (
                "solver_native_total_ms".into(),
                "ms/detailed/solve".into(),
                stats.total_duration.as_secs_f64() * 1_000.0,
            ),
        ]
    } else {
        Vec::new()
    };
    FitSnapshot {
        parameters,
        r_squared,
        counters,
        diagnostics,
    }
}

fn parameters_from_map(
    mut solution: std::collections::HashMap<String, f64>,
    names: &[&str],
) -> Result<Vec<f64>, RunnerError> {
    names
        .iter()
        .map(|name| {
            solution.remove(*name).ok_or_else(|| {
                RunnerError::Scenario(format!("fit result omitted parameter {name}"))
            })
        })
        .collect()
}

fn scenario_error(error: impl std::fmt::Display) -> RunnerError {
    RunnerError::Scenario(format!("optimization solve: {error}"))
}

macro_rules! optimization_scenario {
    ($family:ident, $frontend_id:ident, $frontend:literal, $frontend_key:literal,
     $policy_id:ident, $policy:literal, $gavin_policy_id:ident, $method:literal, $suffix:literal) => {
        Scenario {
            id: concat!("optimization_fit20001_", $suffix),
            solver: "optimisation",
            groups: &[
                "optimization",
                "optimization-large",
                $method,
                "fitting",
                "lambdify",
            ],
            workload: if matches!($family, FAMILY_VARPRO | FAMILY_UNIVERSAL_VARPRO) {
                "exponential-sinusoidal VarPro fit, 200010 samples, 2 nonlinear parameters"
            } else {
                "exponential-sinusoidal fit, 20001 samples, 4 fitted parameters"
            },
            frontend: $frontend,
            execution: "Lambdify",
            linear_algebra: "Dense (fixed fitting implementation)",
            parallel_policy: $policy,
            telemetry: "off wall gate + Detailed native stage diagnostics",
            // The VarPro sample count is part of its workload contract and
            // invalidates only those two old short-workload baselines.
            version: if matches!($family, FAMILY_VARPRO | FAMILY_UNIVERSAL_VARPRO) {
                3
            } else {
                2
            },
            gate_metrics: &["scenario_wall_ms"],
            description: concat!(
                "Matched ",
                $method,
                " frontend=",
                $frontend_key,
                " symbolic fitting route; policy=",
                $policy,
                "; no AOT route"
            ),
            run: run_cell_gate::<{ $family }, { $frontend_id }, { $policy_id }, { $gavin_policy_id }>,
            diagnose: Some(diagnose::<{ $family }, { $frontend_id }, { $policy_id }, { $gavin_policy_id }>),
        }
    };
}

pub static MATRIX_SCENARIOS: [Scenario; 14] = [
    optimization_scenario!(
        FAMILY_GAVIN,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_MARQUARDT,
        "Gavin/Marquardt",
        "gavin_expr_marquardt"
    ),
    optimization_scenario!(
        FAMILY_GAVIN,
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_MARQUARDT,
        "Gavin/Marquardt",
        "gavin_atom_marquardt"
    ),
    optimization_scenario!(
        FAMILY_GAVIN,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_QUADRATIC,
        "Gavin/Quadratic",
        "gavin_expr_quadratic"
    ),
    optimization_scenario!(
        FAMILY_GAVIN,
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_QUADRATIC,
        "Gavin/Quadratic",
        "gavin_atom_quadratic"
    ),
    optimization_scenario!(
        FAMILY_GAVIN,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_NIELSEN,
        "Gavin/Nielsen",
        "gavin_expr_nielsen"
    ),
    optimization_scenario!(
        FAMILY_GAVIN,
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_NIELSEN,
        "Gavin/Nielsen",
        "gavin_atom_nielsen"
    ),
    optimization_scenario!(
        FAMILY_FITTING,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_MARQUARDT,
        "FittingSolver",
        "fitting_expr_seq"
    ),
    optimization_scenario!(
        FAMILY_FITTING,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_PARALLEL,
        "Parallel(min_work=1)",
        GAVIN_MARQUARDT,
        "FittingSolver",
        "fitting_expr_parallel"
    ),
    optimization_scenario!(
        FAMILY_FITTING,
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_MARQUARDT,
        "FittingSolver",
        "fitting_atom_seq"
    ),
    optimization_scenario!(
        FAMILY_FITTING,
        FRONTEND_ATOM,
        "AtomView",
        "atom",
        POLICY_PARALLEL,
        "Parallel(min_work=1)",
        GAVIN_MARQUARDT,
        "FittingSolver",
        "fitting_atom_parallel"
    ),
    optimization_scenario!(
        FAMILY_VARPRO,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential (fixed)",
        GAVIN_MARQUARDT,
        "VarPro",
        "varpro_expr"
    ),
    optimization_scenario!(
        FAMILY_UNIVERSAL_LM,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_MARQUARDT,
        "UniversalFitting/LM",
        "universal_lm_expr"
    ),
    optimization_scenario!(
        FAMILY_UNIVERSAL_GAVIN,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential",
        GAVIN_MARQUARDT,
        "UniversalFitting/Gavin",
        "universal_gavin_expr"
    ),
    optimization_scenario!(
        FAMILY_UNIVERSAL_VARPRO,
        FRONTEND_EXPR,
        "ExprLegacy",
        "expr",
        POLICY_SEQUENTIAL,
        "Sequential (fixed)",
        GAVIN_MARQUARDT,
        "UniversalFitting/VarPro",
        "universal_varpro_expr"
    ),
];

#[cfg(test)]
mod tests {
    use super::{MATRIX_SCENARIOS, SAMPLES, VARPRO_SAMPLES};

    #[test]
    fn varpro_perf_gates_use_independently_versioned_large_workloads() {
        assert!(VARPRO_SAMPLES >= SAMPLES * 10);

        for suffix in ["varpro_expr", "universal_varpro_expr"] {
            let scenario = MATRIX_SCENARIOS
                .iter()
                .find(|scenario| scenario.id.ends_with(suffix))
                .unwrap_or_else(|| panic!("missing VarPro scenario: {suffix}"));
            assert_eq!(
                scenario.version, 3,
                "{} must invalidate its old baseline",
                scenario.id
            );
            assert!(scenario.workload.contains("200010 samples"));

            let output =
                (scenario.run)().unwrap_or_else(|error| panic!("{} failed: {error}", scenario.id));
            let known_error = output
                .diagnostics
                .iter()
                .find(|(name, _, _)| name == "known_parameter_max_error")
                .map(|(_, _, value)| *value)
                .expect("VarPro gate must report the known-parameter check");
            assert!(
                known_error <= 2.0e-2,
                "{} error={known_error:e}",
                scenario.id
            );
        }
    }

    #[test]
    fn large_optimization_routes_pass_debug_correctness_and_telemetry_parity() {
        for scenario in MATRIX_SCENARIOS.iter() {
            let output =
                (scenario.run)().unwrap_or_else(|error| panic!("{} failed: {error}", scenario.id));
            assert!(
                output
                    .diagnostics
                    .iter()
                    .any(|(name, _, _)| name == "known_parameter_max_error"),
                "{} omitted its known-parameter check",
                scenario.id
            );
            let detailed = scenario
                .diagnose
                .expect("optimization matrix must provide diagnostics")(
                scenario
            )
            .unwrap_or_else(|error| panic!("{} diagnostics failed: {error}", scenario.id));
            assert!(
                detailed
                    .diagnostics
                    .iter()
                    .any(|(name, _, _)| name == "off_vs_detailed_max_parameter_diff"),
                "{} omitted its Off-versus-Detailed parity check",
                scenario.id
            );
        }
    }
}
