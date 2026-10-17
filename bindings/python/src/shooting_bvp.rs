//! Python bindings for the deliberately small symbolic shooting solver.
//!
//! ShootingBVP in the core models a scalar second-order boundary problem as a
//! two-state first-order IVP. This adapter keeps that limitation explicit and
//! delegates symbolic preparation, IVP dispatch, logging and telemetry to Rust.

use std::collections::HashMap;

use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustedscithe_core::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, PreparationStage, PreparationTelemetryMode, SymbolicLambdifyFrontend,
};
use rustedscithe_core::numerical::ODE_api2::{NonStiffMethod, SolverParam, SolverType};
use rustedscithe_core::numerical::ShootingBVP::Shooting_simple::{
    BoundaryCondition, BoundaryConditionType, ShootingBvpError, ShootingTelemetry,
};
use rustedscithe_core::numerical::ShootingBVP::builder::{ShootingBvpProblem, shooting_bvp};
use rustedscithe_core::numerical::ShootingBVP::symbolic::ShootingSymbolicOptions;
use rustedscithe_core::numerical::SolverTelemetryMode;

use crate::contract::SolverOptions;
use crate::errors::{
    ExhaustionError, InvalidConfigurationError, PreparationError, SolverError,
    UnsupportedOptionError, annotate_native, catch_panic,
};
use crate::expr::PyExpr;

fn normalize(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn boundary_kind(value: &str) -> PyResult<BoundaryConditionType> {
    match normalize(value).as_str() {
        "dirichlet" => Ok(BoundaryConditionType::Dirichlet),
        "neumann" => Ok(BoundaryConditionType::Neumann),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported shooting boundary type '{other}'; expected 'dirichlet' or 'neumann'"
        ))),
    }
}

fn parse_ivp_method(value: &str) -> PyResult<SolverType> {
    match normalize(value).as_str() {
        "rk45" => Ok(SolverType::NonStiff(NonStiffMethod::Rk45)),
        "dopri" => Ok(SolverType::NonStiff(NonStiffMethod::Dopri)),
        "ab4" => Ok(SolverType::NonStiff(NonStiffMethod::Ab4)),
        "radau" | "radau5" => Ok(SolverType::Radau),
        "bdf" => Ok(SolverType::Bdf),
        "backward_euler" | "be" => Ok(SolverType::BackwardEuler),
        "lsode2" => Ok(SolverType::Lsode2),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported shooting IVP method '{other}'; expected rk45, dopri, ab4, radau, bdf, backward_euler or lsode2"
        ))),
    }
}

fn parse_solver_parameters(
    values: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, SolverParam>> {
    let mut result = HashMap::new();
    let Some(values) = values else {
        return Ok(result);
    };

    for (key, value) in values.iter() {
        let name = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("solver_parameters keys must be strings"))?;
        if name.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "solver_parameters keys must not be empty",
            ));
        }
        let parameter = if value.is_none() {
            SolverParam::OptionalFloat(None)
        } else if let Ok(value) = value.extract::<bool>() {
            SolverParam::Bool(value)
        } else if let Ok(value) = value.extract::<usize>() {
            SolverParam::Int(value)
        } else if let Ok(value) = value.extract::<f64>() {
            if !value.is_finite() {
                return Err(InvalidConfigurationError::new_err(format!(
                    "solver parameter '{name}' must be finite"
                )));
            }
            SolverParam::Float(value)
        } else if let Ok(rows) = value.extract::<Vec<Vec<f64>>>() {
            let width = rows.first().map_or(0, Vec::len);
            if width == 0 || rows.iter().any(|row| row.len() != width) {
                return Err(InvalidConfigurationError::new_err(format!(
                    "solver parameter '{name}' must be a non-empty rectangular matrix"
                )));
            }
            if rows.iter().flatten().any(|entry| !entry.is_finite()) {
                return Err(InvalidConfigurationError::new_err(format!(
                    "solver parameter '{name}' matrix must contain only finite values"
                )));
            }
            let flat = rows.into_iter().flatten().collect::<Vec<_>>();
            let height = flat.len() / width;
            SolverParam::OptionalMatrix(Some(DMatrix::from_row_slice(height, width, &flat)))
        } else {
            return Err(PyTypeError::new_err(format!(
                "solver parameter '{name}' must be bool, int, float, None, or a rectangular list-of-lists matrix"
            )));
        };
        result.insert(name, parameter);
    }
    Ok(result)
}

fn symbolic_options(
    options: &SolverOptions,
) -> PyResult<(ShootingSymbolicOptions, SolverTelemetryMode)> {
    options.validate_aot_options()?;
    if options.execution() != "lambdify" {
        return Err(UnsupportedOptionError::new_err(
            "ShootingBVP currently supports only execution='lambdify'; AOT is not implemented",
        ));
    }
    if options.matrix_layout() != "dense" {
        return Err(UnsupportedOptionError::new_err(
            "ShootingBVP is a small-system method and supports only matrix_layout='dense'",
        ));
    }
    let frontend = match options.frontend() {
        "expr_legacy" => SymbolicLambdifyFrontend::ExprLegacy,
        "atom_native" => SymbolicLambdifyFrontend::AtomViewNative,
        _ => unreachable!("common SolverOptions validates symbolic frontend"),
    };
    let execution_policy = match options.execution_policy() {
        "sequential" => LambdifyExecutionPolicy::Sequential,
        "parallel" => LambdifyExecutionPolicy::Parallel { min_work: 1 },
        "auto" => {
            return Err(UnsupportedOptionError::new_err(
                "ShootingBVP supports execution_policy='sequential' or 'parallel', not 'auto'",
            ));
        }
        _ => unreachable!("common SolverOptions validates callback policy"),
    };
    let telemetry_mode = match options.telemetry() {
        "off" => SolverTelemetryMode::Off,
        "counters" => SolverTelemetryMode::Counters,
        "timings" => SolverTelemetryMode::Timings,
        "detailed" => SolverTelemetryMode::Detailed,
        _ => unreachable!("common SolverOptions validates telemetry mode"),
    };
    Ok((
        ShootingSymbolicOptions {
            frontend,
            execution_policy,
            preparation_telemetry: if telemetry_mode.is_enabled() {
                PreparationTelemetryMode::Collect
            } else {
                PreparationTelemetryMode::Disabled
            },
        },
        telemetry_mode,
    ))
}

fn map_shooting_error(error: ShootingBvpError) -> PyErr {
    let native_kind = crate::errors::native_variant(&error);
    let stage = match error {
        ShootingBvpError::SymbolicPreparation(_) => "prepare",
        _ => "solve",
    };
    let message = error.to_string();
    let mapped = match error {
        ShootingBvpError::MissingProblemField { .. }
        | ShootingBvpError::InvalidProblemDimension { .. }
        | ShootingBvpError::InvalidConfiguration { .. }
        | ShootingBvpError::InvalidInterval { .. } => InvalidConfigurationError::new_err(message),
        ShootingBvpError::SymbolicPreparation(_) => PreparationError::new_err(message),
        ShootingBvpError::IterationExhaustion { .. } => ExhaustionError::new_err(message),
        _ => SolverError::new_err(message),
    };
    annotate_native(mapped, stage, "shooting_bvp", "lambdify", &native_kind)
}

/// Detached result from a scalar second-order ShootingBVP solve.
#[pyclass(name = "ShootingBvpSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct PyShootingBvpSolution {
    mesh: Vec<f64>,
    states: Vec<Vec<f64>>,
    shooting_value: f64,
    boundary_values: Vec<f64>,
    frontend: String,
    preparation_ms: f64,
    telemetry_counters: Option<HashMap<String, usize>>,
    telemetry_timings_ms: Option<HashMap<String, f64>>,
    telemetry_mode: String,
    preparation_stages_ms: HashMap<String, f64>,
}

#[pymethods]
impl PyShootingBvpSolution {
    /// Return the independent-variable mesh as an owned Python list.
    #[getter]
    fn mesh(&self) -> Vec<f64> {
        self.mesh.clone()
    }

    /// Return state values in sample-major order as an owned nested list.
    #[getter]
    fn states(&self) -> Vec<Vec<f64>> {
        self.states.clone()
    }

    /// Return the initial slope selected by the shooting iteration.
    #[getter]
    fn shooting_value(&self) -> f64 {
        self.shooting_value
    }

    /// Return the two boundary values reached by the final IVP integration.
    #[getter]
    fn boundary_values(&self) -> Vec<f64> {
        self.boundary_values.clone()
    }

    /// Return the symbolic frontend used to prepare the IVP callbacks.
    #[getter]
    fn frontend(&self) -> &str {
        &self.frontend
    }

    /// Return measured symbolic preparation wall time in milliseconds.
    #[getter]
    fn preparation_ms(&self) -> f64 {
        self.preparation_ms
    }

    /// Combined convenience view; counters are represented as numeric values.
    #[getter]
    fn telemetry(&self) -> Option<HashMap<String, f64>> {
        let mut combined = HashMap::new();
        if let Some(counters) = &self.telemetry_counters {
            for (name, value) in counters {
                combined.insert(name.clone(), *value as f64);
            }
        }
        if let Some(timings) = &self.telemetry_timings_ms {
            combined.extend(timings.iter().map(|(name, value)| (name.clone(), *value)));
        }
        (!combined.is_empty()).then_some(combined)
    }

    /// Discrete counters retain integer types; `None` means collection was off.
    #[getter]
    fn telemetry_counters(&self) -> Option<HashMap<String, usize>> {
        self.telemetry_counters.clone()
    }

    /// Stage durations in milliseconds; scopes can overlap and are not additive.
    #[getter]
    fn telemetry_timings_ms(&self) -> Option<HashMap<String, f64>> {
        self.telemetry_timings_ms.clone()
    }

    /// Return the configured solve telemetry mode (`off`, `counters`, `timings`, or `detailed`).
    #[getter]
    fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    /// Exclusive symbolic preparation stage durations, when measured.
    #[getter]
    fn preparation_stages_ms(&self) -> HashMap<String, f64> {
        self.preparation_stages_ms.clone()
    }

    /// Return the mesh as an owned one-dimensional NumPy array.
    fn mesh_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.mesh.clone())
    }

    /// Return sample-major state values as an owned two-dimensional NumPy array.
    fn states_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        PyArray2::from_vec2(py, &self.states)
            .map_err(|error| InvalidConfigurationError::new_err(error.to_string()))
    }
}

/// Configured symbolic shooting solver for small scalar second-order BVPs.
#[pyclass(name = "ShootingBvpSolver", module = "rustedscithe")]
pub struct PyShootingBvpSolver {
    problem: ShootingBvpProblem,
    telemetry_mode: SolverTelemetryMode,
    route: String,
}

#[pymethods]
impl PyShootingBvpSolver {
    #[new]
    #[pyo3(signature=(equations, state_variables, independent_variable, t_start, t_end, left_boundary_type, left_boundary_value, right_boundary_type, right_boundary_value, *, initial_guess=0.0, tolerance=1e-8, max_iterations=100, step_size=0.01, ivp_method="rk45", solver_parameters=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        equations: Vec<PyExpr>,
        state_variables: Vec<String>,
        independent_variable: String,
        t_start: f64,
        t_end: f64,
        left_boundary_type: &str,
        left_boundary_value: f64,
        right_boundary_type: &str,
        right_boundary_value: f64,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_method: &str,
        solver_parameters: Option<&Bound<'_, PyDict>>,
        options: Option<&SolverOptions>,
    ) -> PyResult<Self> {
        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let (symbolic, telemetry_mode) = symbolic_options(&options)?;
        options.validate_for("shooting_bvp")?;
        let logging_level = crate::contract::native_logging_level(&options, "shooting_bvp")?;
        let equations = equations
            .into_iter()
            .map(|expression| expression.inner)
            .collect();
        let problem = catch_panic("construct", "shooting_bvp", "lambdify", || {
            let mut builder = shooting_bvp(equations, state_variables, independent_variable)
                .interval(t_start, t_end)
                .boundary_conditions(
                    BoundaryCondition {
                        bc_type: boundary_kind(left_boundary_type)?,
                        value: left_boundary_value,
                    },
                    BoundaryCondition {
                        bc_type: boundary_kind(right_boundary_type)?,
                        value: right_boundary_value,
                    },
                )
                .initial_guess(initial_guess)
                .tolerance(tolerance)
                .max_iterations(max_iterations)
                .step_size(step_size)
                .ivp_solver(parse_ivp_method(ivp_method)?)
                .symbolic_options(symbolic)
                .logging_level(logging_level);
            for (name, value) in parse_solver_parameters(solver_parameters)? {
                builder = builder.solver_parameter(name, value);
            }
            builder.build().map_err(map_shooting_error)
        })?;
        let route = format!(
            "{}:lambdify:{}",
            options.frontend(),
            options.execution_policy()
        );
        Ok(Self {
            problem,
            telemetry_mode,
            route,
        })
    }

    /// Run the configured shooting iteration and return an owned trajectory.
    fn solve(&self) -> PyResult<PyShootingBvpSolution> {
        catch_panic("solve", "shooting_bvp", &self.route, || {
            let report = self
                .problem
                .solve(self.telemetry_mode)
                .map_err(map_shooting_error)?;
            let solution = report.solve.solution;
            let states = (0..solution.y.ncols())
                .map(|column| solution.y.column(column).iter().copied().collect())
                .collect();
            let (telemetry_counters, telemetry_timings_ms) = report
                .solve
                .telemetry
                .map(|telemetry| shooting_telemetry_maps(telemetry, self.telemetry_mode))
                .unwrap_or((None, None));
            let preparation_stages_ms = report
                .preparation
                .detailed
                .as_ref()
                .map(|detailed| {
                    detailed
                        .stages
                        .iter()
                        .filter_map(|stage| {
                            stage.wall_time.map(|duration| {
                                (
                                    preparation_stage_name(stage.stage).to_owned(),
                                    duration.as_secs_f64() * 1_000.0,
                                )
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(PyShootingBvpSolution {
                mesh: solution.x_mesh.as_slice().to_vec(),
                states,
                shooting_value: solution.s,
                boundary_values: solution.bound_values.as_slice().to_vec(),
                frontend: format!("{:?}", report.frontend),
                preparation_ms: report.preparation_wall_time.as_secs_f64() * 1_000.0,
                telemetry_counters,
                telemetry_timings_ms,
                telemetry_mode: match self.telemetry_mode {
                    SolverTelemetryMode::Off => "off",
                    SolverTelemetryMode::Counters => "counters",
                    SolverTelemetryMode::Timings => "timings",
                    SolverTelemetryMode::Detailed => "detailed",
                }
                .to_owned(),
                preparation_stages_ms,
            })
        })
    }
}

fn shooting_telemetry_maps(
    value: ShootingTelemetry,
    mode: SolverTelemetryMode,
) -> (Option<HashMap<String, usize>>, Option<HashMap<String, f64>>) {
    let counters = mode.collects_counters().then(|| {
        HashMap::from([
            (
                "secant_iterations".to_owned(),
                value.counters.secant_iterations,
            ),
            (
                "boundary_residual_evaluations".to_owned(),
                value.counters.boundary_residual_evaluations,
            ),
            ("ivp_solves".to_owned(), value.counters.ivp_solves),
        ])
    });
    let timings = mode.collects_timings().then(|| {
        HashMap::from([
            (
                "residual_ivp_ms".to_owned(),
                value.timings.residual_ivp_time.as_secs_f64() * 1_000.0,
            ),
            (
                "final_ivp_ms".to_owned(),
                value.timings.final_ivp_time.as_secs_f64() * 1_000.0,
            ),
            (
                "secant_phase_ms".to_owned(),
                value.timings.secant_phase_time.as_secs_f64() * 1_000.0,
            ),
            (
                "total_ms".to_owned(),
                value.timings.total_time.as_secs_f64() * 1_000.0,
            ),
        ])
    });
    (counters, timings)
}

fn preparation_stage_name(stage: PreparationStage) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::PyExpr;

    #[test]
    fn shooting_binding_solves_harmonic_bvp_and_exposes_telemetry() {
        Python::initialize();
        let options = SolverOptions::new(
            Some("atom_native"),
            Some("lambdify"),
            Some("dense"),
            Some("sequential"),
            Some("timings"),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let solver = PyShootingBvpSolver::new(
            vec![PyExpr::parse("y1").unwrap(), PyExpr::parse("-y0").unwrap()],
            vec!["y0".into(), "y1".into()],
            "x".into(),
            0.0,
            std::f64::consts::FRAC_PI_2,
            "dirichlet",
            0.0,
            "dirichlet",
            1.0,
            0.8,
            1e-9,
            30,
            0.001,
            "bdf",
            None,
            Some(&options),
        )
        .unwrap();
        let solution = solver.solve().unwrap();
        assert_eq!(solution.frontend, "AtomViewNative");
        assert!(solution.mesh.len() > 2);
        assert!((solution.states.last().unwrap()[0] - 1.0).abs() < 1e-6);
        assert!(
            (solution.shooting_value - 1.0).abs() < 1e-5,
            "unexpected shooting value {}, initial {:?}, endpoint {:?}",
            solution.shooting_value,
            solution.states.first(),
            solution.boundary_values
        );
        let telemetry = solution.telemetry().unwrap();
        assert!(telemetry.contains_key("total_ms"));
        assert_eq!(solution.telemetry_mode, "timings");
        assert!(solution.telemetry_counters.is_none());
        assert!(
            solution
                .telemetry_timings_ms
                .as_ref()
                .unwrap()
                .contains_key("total_ms")
        );
        assert!(!solution.preparation_stages_ms.is_empty());
    }

    #[test]
    fn shooting_binding_preserves_counter_only_telemetry_semantics() {
        let telemetry = shooting_telemetry_maps(
            ShootingTelemetry {
                mode: SolverTelemetryMode::Counters,
                counters: Default::default(),
                timings: Default::default(),
            },
            SolverTelemetryMode::Counters,
        );
        assert!(telemetry.0.is_some());
        assert!(telemetry.1.is_none());
    }

    #[test]
    fn shooting_binding_accepts_method_specific_ivp_parameters() {
        Python::initialize();
        let options =
            SolverOptions::new(None, None, None, None, Some("off"), None, None, None, None)
                .unwrap();
        let mut builder = shooting_bvp(
            vec![
                PyExpr::parse("y1").unwrap().inner,
                PyExpr::parse("-y0").unwrap().inner,
            ],
            vec!["y0".into(), "y1".into()],
            "x",
        )
        .interval(0.0, 1.0)
        .boundary_conditions(
            BoundaryCondition {
                bc_type: BoundaryConditionType::Dirichlet,
                value: 0.0,
            },
            BoundaryCondition {
                bc_type: BoundaryConditionType::Dirichlet,
                value: 0.84,
            },
        )
        .initial_guess(0.5)
        .ivp_solver(SolverType::NonStiff(NonStiffMethod::Rk45))
        .symbolic_options(symbolic_options(&options).unwrap().0);
        builder = builder.solver_parameter("max_steps", SolverParam::Int(2000));
        let problem = builder.build().unwrap();
        let report = problem.solve(SolverTelemetryMode::Off).unwrap();
        assert!(
            (report.solve.solution.y[(0, report.solve.solution.y.ncols() - 1)] - 0.84).abs() < 1e-5
        );
    }

    #[test]
    fn shooting_binding_rejects_aot_and_maps_native_exhaustion() {
        Python::initialize();
        let options =
            SolverOptions::new(None, Some("aot"), None, None, None, None, None, None, None)
                .unwrap();
        let error = PyShootingBvpSolver::new(
            vec![PyExpr::parse("y1").unwrap(), PyExpr::parse("-y0").unwrap()],
            vec!["y0".into(), "y1".into()],
            "x".into(),
            0.0,
            1.0,
            "dirichlet",
            0.0,
            "dirichlet",
            1.0,
            0.0,
            1e-8,
            10,
            0.01,
            "rk45",
            None,
            Some(&options),
        );
        let error = match error {
            Ok(_) => panic!("AOT must be rejected by ShootingBVP"),
            Err(error) => error,
        };
        Python::attach(|py| assert!(error.is_instance_of::<UnsupportedOptionError>(py)));
        assert_eq!(
            crate::errors::native_variant(&ShootingBvpError::IterationExhaustion {
                max_iterations: 1,
                residual: 1.0,
            }),
            "IterationExhaustion"
        );
    }
}
