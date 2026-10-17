//! Typed builder API for second-order shooting problems.
//!
//! This facade keeps boundary conditions and IVP method selection typed. It
//! adapts to the existing symbolic Lambdify implementation only at solve time,
//! so the compatibility `BVPShooting` and its legacy boundary map remain intact.

use super::Shooting_simple::{BoundaryCondition, BoundaryConditionType, ShootingBvpError};
use super::Shooting_sym_wrap::BVPShooting;
use super::symbolic::{ShootingSymbolicOptions, ShootingSymbolicSolveReport};
use crate::numerical::ODE_api2::{SolverParam, SolverType};
use crate::numerical::SolverTelemetryMode;
use crate::numerical::logging::SolverLogLevel;
use crate::symbolic::symbolic_engine::Expr;
use std::collections::HashMap;

/// Builder for a scalar second-order BVP represented as a two-state IVP.
///
/// The state convention is `[y, y']`; therefore a Dirichlet boundary fixes
/// state component zero and a Neumann boundary fixes component one.
pub struct ShootingBvpBuilder {
    equations: Vec<Expr>,
    state_variables: Vec<String>,
    independent_variable: String,
    interval: Option<(f64, f64)>,
    left_boundary: Option<BoundaryCondition>,
    right_boundary: Option<BoundaryCondition>,
    initial_guess: f64,
    tolerance: f64,
    max_iterations: usize,
    step_size: f64,
    ivp_solver: SolverType,
    solver_parameters: HashMap<String, SolverParam>,
    symbolic_options: ShootingSymbolicOptions,
    logging_level: SolverLogLevel,
}

impl ShootingBvpBuilder {
    /// Starts a builder with symbolic first-order equations and state names.
    pub fn new(
        equations: Vec<Expr>,
        state_variables: Vec<String>,
        independent_variable: impl Into<String>,
    ) -> Self {
        Self {
            equations,
            state_variables,
            independent_variable: independent_variable.into(),
            interval: None,
            left_boundary: None,
            right_boundary: None,
            initial_guess: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.01,
            ivp_solver: SolverType::NonStiff(crate::numerical::ODE_api2::NonStiffMethod::Rk45),
            solver_parameters: HashMap::new(),
            symbolic_options: ShootingSymbolicOptions::default(),
            logging_level: SolverLogLevel::Off,
        }
    }

    /// Sets the finite, increasing independent-variable interval.
    pub fn interval(mut self, start: f64, end: f64) -> Self {
        self.interval = Some((start, end));
        self
    }

    /// Sets the left and right Dirichlet/Neumann boundary conditions.
    pub fn boundary_conditions(
        mut self,
        left: BoundaryCondition,
        right: BoundaryCondition,
    ) -> Self {
        self.left_boundary = Some(left);
        self.right_boundary = Some(right);
        self
    }

    /// Sets the initial value guessed for the unconstrained left state.
    pub fn initial_guess(mut self, value: f64) -> Self {
        self.initial_guess = value;
        self
    }

    /// Sets the absolute boundary residual tolerance.
    pub fn tolerance(mut self, value: f64) -> Self {
        self.tolerance = value;
        self
    }

    /// Sets the maximum number of secant updates.
    pub fn max_iterations(mut self, value: usize) -> Self {
        self.max_iterations = value;
        self
    }

    /// Sets the IVP integration step-size hint.
    pub fn step_size(mut self, value: f64) -> Self {
        self.step_size = value;
        self
    }

    /// Selects a native IVP method through the typed `ODE_api2` enum.
    pub fn ivp_solver(mut self, solver: SolverType) -> Self {
        self.ivp_solver = solver;
        self
    }

    /// Adds one method-specific option understood by the selected IVP solver.
    pub fn solver_parameter(mut self, name: impl Into<String>, value: SolverParam) -> Self {
        self.solver_parameters.insert(name.into(), value);
        self
    }

    /// Selects the symbolic Lambdify frontend and execution policy.
    pub fn symbolic_options(mut self, options: ShootingSymbolicOptions) -> Self {
        self.symbolic_options = options;
        self
    }

    /// Enables Rust solver logging at the requested level; off by default.
    pub fn logging_level(mut self, level: SolverLogLevel) -> Self {
        self.logging_level = level;
        self
    }

    /// Validates the problem definition and returns an immutable solve config.
    pub fn build(self) -> Result<ShootingBvpProblem, ShootingBvpError> {
        let (start, end) = self
            .interval
            .ok_or(ShootingBvpError::MissingProblemField { field: "interval" })?;
        let left_boundary = self
            .left_boundary
            .ok_or(ShootingBvpError::MissingProblemField {
                field: "left boundary condition",
            })?;
        let right_boundary = self
            .right_boundary
            .ok_or(ShootingBvpError::MissingProblemField {
                field: "right boundary condition",
            })?;
        if self.state_variables.len() != 2 || self.equations.len() != 2 {
            return Err(ShootingBvpError::InvalidProblemDimension {
                states: self.state_variables.len(),
                equations: self.equations.len(),
            });
        }
        if !start.is_finite() || !end.is_finite() || end <= start {
            return Err(ShootingBvpError::InvalidInterval { start, end });
        }
        for (field, value) in [
            ("initial_guess", self.initial_guess),
            ("left boundary value", left_boundary.value),
            ("right boundary value", right_boundary.value),
        ] {
            if !value.is_finite() {
                return Err(ShootingBvpError::InvalidConfiguration {
                    field,
                    requirement: "be finite",
                });
            }
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(ShootingBvpError::InvalidConfiguration {
                field: "tolerance",
                requirement: "be finite and greater than zero",
            });
        }
        if self.max_iterations == 0 {
            return Err(ShootingBvpError::InvalidConfiguration {
                field: "max_iterations",
                requirement: "be greater than zero",
            });
        }
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(ShootingBvpError::InvalidConfiguration {
                field: "step_size",
                requirement: "be finite and greater than zero",
            });
        }
        if self.state_variables.iter().any(|name| name.is_empty())
            || self.independent_variable.is_empty()
        {
            return Err(ShootingBvpError::InvalidConfiguration {
                field: "symbol names",
                requirement: "be non-empty",
            });
        }
        Ok(ShootingBvpProblem {
            equations: self.equations,
            state_variables: self.state_variables,
            independent_variable: self.independent_variable,
            interval: (start, end),
            left_boundary,
            right_boundary,
            initial_guess: self.initial_guess,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            step_size: self.step_size,
            ivp_solver: self.ivp_solver,
            solver_parameters: self.solver_parameters,
            symbolic_options: self.symbolic_options,
            logging_level: self.logging_level,
        })
    }
}

/// Validated, typed shooting problem ready to execute through Lambdify.
pub struct ShootingBvpProblem {
    equations: Vec<Expr>,
    state_variables: Vec<String>,
    independent_variable: String,
    interval: (f64, f64),
    left_boundary: BoundaryCondition,
    right_boundary: BoundaryCondition,
    initial_guess: f64,
    tolerance: f64,
    max_iterations: usize,
    step_size: f64,
    ivp_solver: SolverType,
    solver_parameters: HashMap<String, SolverParam>,
    symbolic_options: ShootingSymbolicOptions,
    logging_level: SolverLogLevel,
}

impl ShootingBvpProblem {
    /// Executes the configured IVP/shooting solve and returns both symbolic
    /// preparation evidence and numerical telemetry when requested.
    pub fn solve(
        &self,
        telemetry_mode: SolverTelemetryMode,
    ) -> Result<ShootingSymbolicSolveReport, ShootingBvpError> {
        let mut boundary_map: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
        for (boundary_index, condition) in [(0, self.left_boundary), (1, self.right_boundary)] {
            let variable_index = match condition.bc_type {
                BoundaryConditionType::Dirichlet => 0,
                BoundaryConditionType::Neumann => 1,
            };
            boundary_map
                .entry(self.state_variables[variable_index].clone())
                .or_default()
                .push((boundary_index, condition.value));
        }

        let mut solver = BVPShooting::new(
            self.equations.clone(),
            self.state_variables.clone(),
            self.independent_variable.clone(),
            boundary_map,
            self.interval,
        );
        solver.set_logging_level(self.logging_level);
        solver.try_solve_symbolic(
            self.initial_guess,
            self.tolerance,
            self.max_iterations,
            self.step_size,
            self.ivp_solver.clone(),
            self.solver_parameters.clone(),
            self.symbolic_options,
            telemetry_mode,
        )
    }
}

/// Starts a typed builder for the symbolic shooting route.
pub fn shooting_bvp(
    equations: Vec<Expr>,
    state_variables: Vec<String>,
    independent_variable: impl Into<String>,
) -> ShootingBvpBuilder {
    ShootingBvpBuilder::new(equations, state_variables, independent_variable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP as DampedBvp, SolverParams,
    };
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;
    use crate::numerical::ODE_api2::NonStiffMethod;
    use crate::numerical::ShootingBVP::Shooting_simple::BoundaryConditionType;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    fn interpolate(mesh: &[f64], samples: &[f64], x: f64) -> f64 {
        assert_eq!(mesh.len(), samples.len());
        assert!(mesh.len() >= 2);
        let upper = mesh
            .partition_point(|point| *point < x)
            .clamp(1, mesh.len() - 1);
        let lower = upper - 1;
        let fraction = (x - mesh[lower]) / (mesh[upper] - mesh[lower]);
        samples[lower] + fraction * (samples[upper] - samples[lower])
    }

    #[test]
    fn builder_requires_interval_and_explicit_boundaries() {
        let result = shooting_bvp(
            vec![Expr::parse_expression("dy"), Expr::parse_expression("0")],
            vec!["y".into(), "dy".into()],
            "x",
        )
        .build();

        assert!(matches!(
            result,
            Err(ShootingBvpError::MissingProblemField { field: "interval" })
        ));
    }

    #[test]
    fn builder_preserves_typed_mixed_boundary_configuration_and_solves() {
        let problem = shooting_bvp(
            vec![Expr::parse_expression("dy"), Expr::parse_expression("0")],
            vec!["y".into(), "dy".into()],
            "x",
        )
        .interval(0.0, 1.0)
        .boundary_conditions(
            BoundaryCondition {
                value: 0.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 2.0,
                bc_type: BoundaryConditionType::Neumann,
            },
        )
        .initial_guess(0.0)
        .tolerance(1e-9)
        .max_iterations(12)
        .step_size(0.01)
        .ivp_solver(SolverType::NonStiff(
            crate::numerical::ODE_api2::NonStiffMethod::Rk45,
        ))
        .build()
        .unwrap();

        let report = problem.solve(SolverTelemetryMode::Detailed).unwrap();
        assert!((report.solve.solution.bound_values[1] - 2.0).abs() < 1e-7);
        assert!(report.solve.telemetry.is_some());
    }

    #[test]
    fn exact_exponential_problem_matches_across_frontends_and_ivp_solvers() {
        let methods = [
            SolverType::NonStiff(NonStiffMethod::Rk45),
            SolverType::Bdf,
            SolverType::Radau,
        ];
        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            for method in methods.iter().cloned() {
                let builder = shooting_bvp(
                    vec![Expr::parse_expression("dy"), Expr::parse_expression("y")],
                    vec!["y".into(), "dy".into()],
                    "x",
                )
                .interval(0.0, 1.0)
                .boundary_conditions(
                    BoundaryCondition {
                        value: 0.0,
                        bc_type: BoundaryConditionType::Dirichlet,
                    },
                    BoundaryCondition {
                        value: 1.0_f64.sinh(),
                        bc_type: BoundaryConditionType::Dirichlet,
                    },
                )
                .initial_guess(0.8)
                .tolerance(1e-8)
                .max_iterations(30)
                .step_size(0.002)
                .ivp_solver(method)
                .solver_parameter("rtol", SolverParam::Float(1e-9))
                .solver_parameter("atol", SolverParam::Float(1e-11))
                .solver_parameter("max_step", SolverParam::Float(0.02))
                .symbolic_options(ShootingSymbolicOptions {
                    frontend,
                    ..ShootingSymbolicOptions::default()
                });

                let report = builder
                    .build()
                    .unwrap()
                    .solve(SolverTelemetryMode::Detailed)
                    .unwrap();
                assert!(
                    (report.solve.solution.bound_values[0] - 1.0_f64.sinh()).abs() < 2e-6,
                    "endpoint value mismatch: frontend={frontend:?}, ivp={:?}",
                    report.frontend
                );
                assert!(
                    (report.solve.solution.bound_values[1] - 1.0_f64.cosh()).abs() < 2e-5,
                    "endpoint derivative mismatch: frontend={frontend:?}"
                );
                assert_eq!(report.frontend, frontend);
                assert!(report.solve.telemetry.is_some());
            }
        }
    }

    #[test]
    fn harmonic_oscillator_builder_route_matches_exact_endpoint_state() {
        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let problem = shooting_bvp(
                vec![Expr::parse_expression("dy"), Expr::parse_expression("-y")],
                vec!["y".into(), "dy".into()],
                "x",
            )
            .interval(0.0, std::f64::consts::FRAC_PI_2)
            .boundary_conditions(
                BoundaryCondition {
                    value: 1.0,
                    bc_type: BoundaryConditionType::Dirichlet,
                },
                BoundaryCondition {
                    value: 0.0,
                    bc_type: BoundaryConditionType::Dirichlet,
                },
            )
            .initial_guess(0.0)
            .tolerance(1e-9)
            .max_iterations(20)
            .step_size(0.001)
            .ivp_solver(SolverType::Bdf)
            .solver_parameter("rtol", SolverParam::Float(1e-9))
            .solver_parameter("atol", SolverParam::Float(1e-11))
            .solver_parameter("max_step", SolverParam::Float(0.02))
            .symbolic_options(ShootingSymbolicOptions {
                frontend,
                ..ShootingSymbolicOptions::default()
            })
            .build()
            .unwrap();

            let report = problem.solve(SolverTelemetryMode::Off).unwrap();
            assert!(report.solve.telemetry.is_none());
            assert!(report.solve.solution.bound_values[0].abs() < 1e-6);
            assert!(
                (report.solve.solution.bound_values[1] + 1.0).abs() < 1e-5,
                "endpoint state mismatch for {frontend:?}: {:?}",
                report.solve.solution.bound_values
            );
        }
    }

    #[test]
    fn shooting_matches_independent_damped_bvp_on_common_mesh() {
        // y'' + y = 0, y(0)=0, y(pi/2)=1 has exact solution
        // y=sin(x), y'=cos(x). The two solvers use different meshes and
        // independent algorithms; compare interpolated trajectories as well
        // as each solver against the analytic solution.
        let end = std::f64::consts::FRAC_PI_2;
        let shooting = shooting_bvp(
            vec![Expr::parse_expression("dy"), Expr::parse_expression("-y")],
            vec!["y".into(), "dy".into()],
            "x",
        )
        .interval(0.0, end)
        .boundary_conditions(
            BoundaryCondition {
                value: 0.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
        )
        .initial_guess(0.8)
        .tolerance(1e-9)
        .max_iterations(20)
        .step_size(0.001)
        .ivp_solver(SolverType::Bdf)
        .solver_parameter("rtol", SolverParam::Float(1e-9))
        .solver_parameter("atol", SolverParam::Float(1e-11))
        .solver_parameter("max_step", SolverParam::Float(0.015))
        .build()
        .unwrap()
        .solve(SolverTelemetryMode::Off)
        .unwrap()
        .solve
        .solution;

        let n_steps = 120;
        let h = end / n_steps as f64;
        let mut initial_values = Vec::with_capacity(2 * n_steps);
        for i in 0..n_steps {
            let x = i as f64 * h;
            // Perturb the exact profile so this checks an actual Newton solve.
            initial_values.push(x.sin() + 0.01 * x * (end - x));
            initial_values.push(x.cos() + 0.01);
        }
        let initial_guess =
            DMatrix::from_column_slice(2, n_steps, DVector::from_vec(initial_values).as_slice());
        let boundary_conditions = HashMap::from([("y".to_string(), vec![(0, 0.0), (1, 1.0)])]);
        let mut damped = DampedBvp::new_numeric_with_jacobian_options(
            initial_guess,
            vec!["y".into(), "dy".into()],
            "x".into(),
            boundary_conditions,
            0.0,
            end,
            n_steps,
            DampedSolverOptions::dense_damped()
                .trapezoid_derivative()
                .with_strategy_params(Some(SolverParams::default()))
                .with_abs_tolerance(1e-9)
                .with_rel_tolerance(HashMap::from([
                    ("y".to_string(), 1e-7),
                    ("dy".to_string(), 1e-7),
                ]))
                .with_bounds(HashMap::from([
                    ("y".to_string(), (-2.0, 2.0)),
                    ("dy".to_string(), (-2.0, 2.0)),
                ]))
                .with_max_iterations(40),
            |_x, state: &DVector<f64>, _params| DVector::from_vec(vec![state[1], -state[0]]),
            |_x, _state: &DVector<f64>, _params| {
                DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0])
            },
        );
        damped.dont_save_log(true);
        damped
            .try_solve()
            .expect("BVP_Damp numerical solve should complete")
            .expect("BVP_Damp should converge");

        let damped_dataset = damped
            .postprocess_dataset()
            .expect("converged BVP_Damp result should expose its mesh and values");
        let damped_mesh = damped_dataset.axis.as_slice();
        let damped_y: Vec<_> = (0..damped_dataset.values.nrows())
            .map(|row| damped_dataset.values[(row, 0)])
            .collect();
        let damped_dy: Vec<_> = (0..damped_dataset.values.nrows())
            .map(|row| damped_dataset.values[(row, 1)])
            .collect();
        let shooting_mesh = shooting.x_mesh.as_slice();
        let shooting_y: Vec<_> = (0..shooting.y.ncols())
            .map(|column| shooting.y[(0, column)])
            .collect();
        let shooting_dy: Vec<_> = (0..shooting.y.ncols())
            .map(|column| shooting.y[(1, column)])
            .collect();

        let mut max_cross_solver_difference: f64 = 0.0;
        let mut max_analytic_error: f64 = 0.0;
        for i in 0..=80 {
            let x = end * i as f64 / 80.0;
            let damped_y_at_x = interpolate(damped_mesh, &damped_y, x);
            let damped_dy_at_x = interpolate(damped_mesh, &damped_dy, x);
            let shooting_y_at_x = interpolate(shooting_mesh, &shooting_y, x);
            let shooting_dy_at_x = interpolate(shooting_mesh, &shooting_dy, x);
            max_cross_solver_difference = max_cross_solver_difference
                .max((damped_y_at_x - shooting_y_at_x).abs())
                .max((damped_dy_at_x - shooting_dy_at_x).abs());
            max_analytic_error = max_analytic_error
                .max((damped_y_at_x - x.sin()).abs())
                .max((damped_dy_at_x - x.cos()).abs())
                .max((shooting_y_at_x - x.sin()).abs())
                .max((shooting_dy_at_x - x.cos()).abs());
        }

        assert!(
            max_cross_solver_difference < 2e-3,
            "interpolated solver trajectories differ by {max_cross_solver_difference:e}"
        );
        assert!(
            max_analytic_error < 2e-3,
            "one of the interpolated trajectories differs from the exact solution by {max_analytic_error:e}"
        );
    }
}
