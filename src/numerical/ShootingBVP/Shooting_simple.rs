//! # Shooting Method for Boundary Value Problems (BVP)
//!
//! This module implements the shooting method to solve second-order boundary value problems
//! of the form: y'' = f(x, y, y') with various boundary condition types.
//!
//! ## Supported Boundary Conditions
//! 1. **Dirichlet-Dirichlet**: y(a) = α, y(b) = β
//! 2. **Dirichlet-Neumann**: y(a) = α, y'(b) = γ  
//! 3. **Neumann-Dirichlet**: y'(a) = α, y(b) = β
//! 4. **Neumann-Neumann**: y'(a) = α, y'(b) = γ
//!
//! ## Method Overview
//! The shooting method converts a BVP into an initial value problem (IVP) by:
//! 1. Guessing the unknown initial condition (y'(a) or y(a))
//! 2. Solving the IVP with known initial condition
//! 3. Adjusting the guess until the boundary condition at x = b is satisfied
//!
//! ## Parameters
//! - **a**: Left boundary point (start of domain)
//! - **b**: Right boundary point (end of domain)  
//! - **left_bc**: Left boundary condition (value, type)
//! - **right_bc**: Right boundary condition (value, type)
//! - **initial_guess**: Initial guess for unknown condition
//! - **tolerance**: Convergence tolerance for secant method
//! - **max_iterations**: Maximum iterations for secant method
//! - **step_size**: Step size for RK4 integration
//!
//! ## Usage Example
//! ```rust, ignore
//! use nalgebra::DVector;
//!use RustedSciThe::numerical::ShootingBVP::Shooting_simple::rk4_ivp_solver;
//! // Define ODE system: y'' = y becomes [y', y]
//! let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
//!     DVector::from_vec(vec![y[1], y[0]])
//! };
//!
//! // Set up BVP: y'' = y, y(0) = 0, y'(1) = γ (mixed boundary conditions)
//! let problem = BoundaryValueProblem {
//!     ode_system,
//!     a: 0.0,      // Left boundary
//!     b: 1.0,      // Right boundary  
//!     left_bc: BoundaryCondition { value: 0.0, bc_type: BoundaryConditionType::Dirichlet },  // y(0) = 0
//!     right_bc: BoundaryCondition { value: 1.0, bc_type: BoundaryConditionType::Neumann }, // y'(1) = 1
//! };
//!
//! let solver = ShootingMethodSolver {
//!     initial_guess: 1.0,  // Initial guess for y'(0)
//!     tolerance: 1e-6,
//!     max_iterations: 100,
//!     step_size: 0.01,
//! };
//!
//! let (slope, solution) = solver.solve(&problem, |x0, y0, x_end, h| {
//!     rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
//! }).unwrap();
//! ```

use log::{debug, error, info};
use nalgebra::{DMatrix, DVector};

/// Boundary condition types
#[derive(Debug, Clone, Copy)]
pub enum BoundaryConditionType {
    /// Dirichlet: y = value
    Dirichlet,
    /// Neumann: y' = value  
    Neumann,
}

/// Boundary condition specification
#[derive(Debug, Clone, Copy)]
pub struct BoundaryCondition {
    pub value: f64,
    pub bc_type: BoundaryConditionType,
}

/// Represents the BVP problem: y'' = f(x, y, y') with various boundary conditions.
pub struct BoundaryValueProblem<F> {
    /// The ODE system in 1st-order form: dy/dx = f(x, y)
    pub ode_system: F,
    pub a: f64,
    pub b: f64,
    pub left_bc: BoundaryCondition,
    pub right_bc: BoundaryCondition,
    // Backward compatibility
    pub alpha: f64,
    pub beta: f64,
}

impl<F> BoundaryValueProblem<F> {
    /// Create BVP with new boundary condition format
    pub fn new(
        ode_system: F,
        a: f64,
        b: f64,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Self {
        Self {
            ode_system,
            a,
            b,
            left_bc,
            right_bc,
            alpha: left_bc.value,
            beta: right_bc.value,
        }
    }

    /// Create BVP with legacy format (Dirichlet-Dirichlet)
    pub fn legacy(ode_system: F, a: f64, b: f64, alpha: f64, beta: f64) -> Self {
        Self {
            ode_system,
            a,
            b,
            left_bc: BoundaryCondition {
                value: alpha,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            right_bc: BoundaryCondition {
                value: beta,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            alpha,
            beta,
        }
    }
}

/// Configuration for the shooting method solver.
pub struct ShootingMethodSolver {
    pub initial_guess: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub step_size: f64,
    pub result: ShootingMethodResult,
}
#[derive(Debug, Clone)]
pub struct ShootingMethodResult {
    pub x_mesh: DVector<f64>,
    pub y: DMatrix<f64>,
    pub s: f64,
    pub bound_values: DVector<f64>,
}
impl Default for ShootingMethodResult {
    fn default() -> Self {
        Self {
            x_mesh: DVector::zeros(0),
            y: DMatrix::zeros(0, 0),
            s: 0.0,
            bound_values: DVector::zeros(0),
        }
    }
}
/// Secant method for finding roots of a function f(x).
fn secant_method<F>(
    f: F,
    initial_guess: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<f64, &'static str>
where
    F: Fn(f64) -> f64,
{
    debug!(
        "Starting secant method with initial_guess={}, tolerance={}, max_iterations={}",
        initial_guess, tolerance, max_iterations
    );

    let mut x_prev = initial_guess;
    let mut x_curr = initial_guess * 1.1; // Small perturbation
    let mut f_prev = f(x_prev);
    let mut f_curr = f(x_curr);

    debug!(
        "Initial values: x_prev={}, x_curr={}, f_prev={}, f_curr={}",
        x_prev, x_curr, f_prev, f_curr
    );

    for iteration in 0..max_iterations {
        if f_curr.abs() < tolerance {
            info!(
                "Secant method converged after {} iterations with x={}, f(x)={}",
                iteration, x_curr, f_curr
            );
            return Ok(x_curr);
        }

        let x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev);
        debug!(
            "Iteration {}: x_next={}, f_curr={}",
            iteration, x_next, f_curr
        );

        x_prev = x_curr;
        x_curr = x_next;
        f_prev = f_curr;
        f_curr = f(x_curr);
    }

    error!(
        "Secant method did not converge after {} iterations",
        max_iterations
    );
    Err("Secant method did not converge.")
}

impl ShootingMethodSolver {
    /// Solves the BVP using the shooting method.
    pub fn solve<F>(
        &mut self,
        problem: &BoundaryValueProblem<F>,
        ivp_solver: impl Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>),
    ) -> Result<ShootingMethodResult, &'static str>
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64>,
    {
        info!("Starting shooting method solver");
        debug!(
            "Problem parameters: a={}, b={}, left_bc={:?}, right_bc={:?}",
            problem.a, problem.b, problem.left_bc, problem.right_bc
        );
        debug!(
            "Solver parameters: initial_guess={}, tolerance={}, max_iterations={}, step_size={}",
            self.initial_guess, self.tolerance, self.max_iterations, self.step_size
        );

        // Define the error function based on boundary condition types
        let error = |guess: f64| -> f64 {
            let y0 = match problem.left_bc.bc_type {
                BoundaryConditionType::Dirichlet => {
                    // y(a) = known, y'(a) = guess
                    DVector::from_vec(vec![problem.left_bc.value, guess])
                }
                BoundaryConditionType::Neumann => {
                    // y'(a) = known, y(a) = guess
                    DVector::from_vec(vec![guess, problem.left_bc.value])
                }
            };
            info!("starting ivp solver");
            let sol_matrix = ivp_solver(problem.a, y0, problem.b, self.step_size).0;
            let sol = sol_matrix.column(sol_matrix.ncols() - 1); // Get final column
            info!("get IVP solution {}", sol);
            let error_val = match problem.right_bc.bc_type {
                BoundaryConditionType::Dirichlet => {
                    // Check y(b) = right_bc.value
                    sol[0] - problem.right_bc.value
                }
                BoundaryConditionType::Neumann => {
                    // Check y'(b) = right_bc.value
                    sol[1] - problem.right_bc.value
                }
            };

            debug!(
                "Error function evaluation: guess={}, y(b)={}, y'(b)={}, error={}",
                guess, sol[0], sol[1], error_val
            );
            error_val
        };

        // Find the unknown initial condition using secant method
        let unknown_initial = secant_method(
            error,
            self.initial_guess,
            self.tolerance,
            self.max_iterations,
        )?;

        info!("Found unknown initial condition: {}", unknown_initial);

        // Compute the final solution
        let y0 = match problem.left_bc.bc_type {
            BoundaryConditionType::Dirichlet => {
                DVector::from_vec(vec![problem.left_bc.value, unknown_initial])
            }
            BoundaryConditionType::Neumann => {
                DVector::from_vec(vec![unknown_initial, problem.left_bc.value])
            }
        };
        info!("starting final solution...");
        let (sol_matrix, x_mesh) = ivp_solver(problem.a, y0, problem.b, self.step_size);
        //  info!("final solution{:?}", sol_matrix);
        let binding = sol_matrix.clone();
        let last_points = binding.column(sol_matrix.ncols() - 1);
        //let x_mesh = create_mesh(x0, x_end, step_size);
        info!(
            "Final solution: y({})={}, y'({})={}",
            problem.b, last_points[0], problem.b, last_points[1]
        );
        self.result = ShootingMethodResult {
            x_mesh: x_mesh,
            y: sol_matrix.clone(),
            s: unknown_initial,
            bound_values: last_points.into_owned(),
        };
        Ok(self.result.clone())
    }

    pub fn new() -> Self {
        Self {
            initial_guess: 0.0,
            tolerance: 0.0,
            max_iterations: 0,
            step_size: 0.0,
            result: ShootingMethodResult::default(),
        }
    }
    pub fn get_solution(&self) -> ShootingMethodResult {
        return self.result.clone();
    }
    pub fn get_y(&self) -> DMatrix<f64> {
        return self.result.y.clone();
    }
    pub fn get_x(&self) -> DVector<f64> {
        return self.result.x_mesh.clone();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
pub fn RK45_ivp_solver_and_mesh<F>(
    x0: f64,
    y0: DVector<f64>,
    x_end: f64,
    step_size: f64,

    ode_system: F,
) -> (DMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    let ivp_sol = rk4_ivp_solver(x0, y0, x_end, step_size, ode_system);

    let x_mesh = create_mesh(x0, x_end, step_size);
    (ivp_sol, x_mesh)
}
/// Solves an initial value problem (IVP) using the 4th-order Runge-Kutta method.
///
/// # Arguments
/// * `x0` - Initial x value.
/// * `y0` - Initial state vector (y and y' packed into a DVector).
/// * `x_end` - End of the integration interval.
/// * `step_size` - Step size for numerical integration.
/// * `ode_system` - Function defining the ODE system: dy/dx = f(x, y).
///
/// # Returns
/// * `DMatrix<f64>` - Matrix where each column is the solution at a time step
pub fn rk4_ivp_solver<F>(
    x0: f64,
    y0: DVector<f64>,
    x_end: f64,
    step_size: f64,
    ode_system: F,
) -> DMatrix<f64>
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    // Handle case where no integration is needed
    if (x_end - x0).abs() < f64::EPSILON {
        return DMatrix::from_column_slice(y0.len(), 1, y0.as_slice());
    }

    let n_steps = ((x_end - x0) / step_size).ceil() as usize + 1;
    let mut solution = DMatrix::zeros(y0.len(), n_steps);
    let mut x = x0;
    let mut y = y0.clone();
    let mut step_idx = 0;

    // Store initial condition
    solution.set_column(step_idx, &y);
    step_idx += 1;

    while x < x_end && step_idx < n_steps {
        let h = step_size.min(x_end - x);

        let k1 = ode_system(x, &y);
        let k2 = ode_system(x + h / 2.0, &(y.clone() + (h / 2.0) * &k1));
        let k3 = ode_system(x + h / 2.0, &(y.clone() + (h / 2.0) * &k2));
        let k4 = ode_system(x + h, &(y.clone() + h * &k3));

        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        x += h;

        solution.set_column(step_idx, &y);
        step_idx += 1;
    }

    // Return only the used columns

    solution.columns(0, step_idx).into_owned()
}

pub fn create_mesh(x0: f64, x_end: f64, step_size: f64) -> DVector<f64> {
    let n_steps = ((x_end - x0) / step_size).ceil() as usize + 1;
    let mut x_mesh: Vec<f64> = Vec::new();
    let mut x = x0;
    x_mesh.push(x);
    let mut step_idx = 0;
    while x < x_end && step_idx < n_steps {
        let h = step_size.min(x_end - x);
        x += h;
        x_mesh.push(x);
        step_idx += 1;
    }
    let x_mesh = DVector::from_vec(x_mesh);
    x_mesh
}
/////////////////////////////////////////////////////////////////////////
//          tests
//////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;

    use simplelog::*; // For floating-point comparisons

    fn init_logger() {
        let _ = SimpleLogger::init(LevelFilter::Debug, Config::default());
    }
    #[test]
    fn second_order_simple_bvp() {
        init_logger();

        // Define the ODE system: [y1' = y2, y2' = y1] (for y'' = y)
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // dy/dx = [y2, y1]
        };

        // Set up the BVP: y'' = y, y(0) = 0, y(1) = 1
        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 1.0);

        // Configure the solver
        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        // Solve the BVP using RK4
        match solver.solve(&problem, |x0, y0, x_end, h| {
            RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
        }) {
            Ok(result) => {
                println!("Found initial slope s = {}", result.s);
                println!("solution {:?}", result.bound_values);
                println!(
                    "Solution at x = b: y = {}, y' = {}",
                    result.bound_values[0], result.bound_values[1]
                );
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    #[test]
    fn test_linear_ode() {
        init_logger();

        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 1.0_f64.sinh());

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        // Check boundary condition at x = b
        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-6);

        // Check initial slope s ≈ y'(0) = 1 (exact for this problem)
        assert_abs_diff_eq!(result.s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_non_convergence() {
        init_logger();

        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 1.0);

        // Force failure with unrealistic tolerance and low iterations
        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-20,
            max_iterations: 5,
            step_size: 0.1,
            result: ShootingMethodResult::default(),
        };

        let result = solver.solve(&problem, |x0, y0, x_end, h| {
            RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
        });

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Secant method did not converge.");
    }
    #[test]
    fn test_zero_step() {
        init_logger();

        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 0.0]) // y'' = 0
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 0.0, 0.0, 0.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-6,
            max_iterations: 10,
            step_size: 0.1,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        // Solution should remain at initial condition
        assert_eq!(result.bound_values[0], problem.alpha);
        assert_eq!(result.bound_values[1], solver.initial_guess);
    }
    #[test]
    fn test_adaptive_step_size() {
        init_logger();

        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -y[0]]) // y'' = -y (harmonic oscillator)
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, std::f64::consts::PI, 0.0, 0.0);

        // Test with coarse step size (should still converge)
        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-4,
            max_iterations: 50,
            step_size: 0.1,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-3);
    }

    #[test]
    fn test_nonlinear_pendulum() {
        init_logger();

        // Nonlinear pendulum: y'' + sin(y) = 0
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -y[0].sin()]) // [y', -sin(y)]
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 0.5);

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-5);
        assert!(result.s.is_finite());
    }
    // stiff problems is not for shooting method
    #[test]
    #[should_panic]
    fn test_stiff_problem() {
        init_logger();

        // Stiff problem: y'' - 100*y = 0
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 100.0 * y[0]])
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 0.1, 1.0, (10.0_f64).exp());

        let mut solver = ShootingMethodSolver {
            initial_guess: 10.0,
            tolerance: 1e-6,
            max_iterations: 1000,
            step_size: 1e-5, // Small step for stiff problem
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("solution {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-3);
        assert_abs_diff_eq!(result.s, 10.0, epsilon = 1e-3);
    }

    #[test]
    fn test_large_domain() {
        init_logger();

        // Simple linear problem over large domain
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 0.0]) // y'' = 0
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 100.0, 0.0, 50.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.5,
            tolerance: 1e-6,
            max_iterations: 50,
            step_size: 1.0,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-6);
        assert_abs_diff_eq!(result.s, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_domain() {
        init_logger();

        // Test with negative domain
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };
        //y(-1) = sinh(-1), y(0) =0
        let problem = BoundaryValueProblem::legacy(ode_system, -1.0, 0.0, (-1.0_f64).sinh(), 0.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: -1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let _ = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        // solution y(x) = 1/2 (e^x - e^(-x))
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        let z: DVector<f64> = y_mesh.row(1).transpose().into_owned();
        for i in 0..x_mesh.len() {
            let exect = (x_mesh[i].exp() - (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y[i], exect, epsilon = 1e-4);
            let exect_z = (x_mesh[i].exp() + (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(z[i], exect_z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_oscillatory_solution() {
        init_logger();

        // Harmonic oscillator with specific frequency
        let omega = 2.0;
        let ode_system = move |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -omega * omega * y[0]])
        };

        let problem =
            BoundaryValueProblem::legacy(ode_system, 0.0, std::f64::consts::PI / omega, 1.0, -1.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-5);
        assert_abs_diff_eq!(result.s, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dirichlet_neumann() {
        init_logger();

        // y'' = y, y(0) = 0, y'(1) = cosh(1)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 0.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 1.0_f64.cosh(),
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        for i in 0..x_mesh.len() {
            let exect = (x_mesh[i].exp() - (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y[i], exect, epsilon = 1e-6);
        }
        assert_abs_diff_eq!(
            result.bound_values[1],
            problem.right_bc.value,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(result.s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_neumann_dirichlet() {
        init_logger();

        // y'' = y, y'(0) = 1, y(1) = sinh(1)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
            BoundaryCondition {
                value: 1.0_f64.sinh(),
                bc_type: BoundaryConditionType::Dirichlet,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(
            result.bound_values[0],
            problem.right_bc.value,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(result.s, 0.0, epsilon = 1e-6);
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y[i], x_mesh[i].sinh(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_neumann_neumann() {
        init_logger();

        // y'' = 0, y'(0) = 1, y'(1) = 1 (linear function)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], 0.0]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(
            result.bound_values[1],
            problem.right_bc.value,
            epsilon = 1e-6
        );
        assert!(result.s.is_finite());
    }

    #[test]
    fn test_dirichlet_neumann_linear() {
        init_logger();

        // y'' = 0, y(0) = 2, y'(1) = 3 (linear function y = 3x + 2)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], 0.0]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 2.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 3.0,
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 3.0,
            tolerance: 1e-8,
            max_iterations: 50,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[1], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.s, 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.bound_values[0], 5.0, epsilon = 1e-6); // y(1) = 3*1 + 2 = 5
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y[i], 3.0 * x_mesh[i] + 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dirichlet_neumann_exponential() {
        init_logger();

        // y'' - y = 0, y(0) = 1, y'(1) = e^1 (solution: y = e^x)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 1.0_f64.exp(),
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[1], 1.0_f64.exp(), epsilon = 1e-6);
        assert_abs_diff_eq!(result.s, 1.0, epsilon = 1e-6);

        let y: DVector<f64> = result.y.row(0).transpose().into_owned();
        let x_mesh = result.x_mesh.clone();
        assert!(y.len() == x_mesh.len());
        let z: DVector<f64> = result.y.row(1).transpose().into_owned();
        for i in 0..y.len() {
            let error = y[i] - (x_mesh[i].exp());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
            let error = z[i] - (x_mesh[i].exp());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_dirichlet_neumann_oscillator() {
        init_logger();

        // y'' + y = 0, y(0) = 0, y'(π) = 1 (solution: y = -sin(x))
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], -y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            std::f64::consts::PI,
            BoundaryCondition {
                value: 0.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 2.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        let y: DVector<f64> = result.y.row(0).transpose().into_owned();
        let x_mesh = result.x_mesh.clone();
        assert!(y.len() == x_mesh.len());
        let z: DVector<f64> = result.y.row(1).transpose().into_owned();
        for i in 0..y.len() {
            let error = y[i] - (-x_mesh[i].sin());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
            let error = z[i] - (-x_mesh[i].cos());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
        }

        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[0], 0.0, epsilon = 1e-5); //y(pi) = -sin(pi)=0
        assert_abs_diff_eq!(result.bound_values[1], 1.0, epsilon = 1e-5); // y'(pi) = cos(pi) =1
        assert_abs_diff_eq!(result.s, -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_rk4_exponential_growth() {
        // Test y' = y with y(0) = 1, exact solution: y(x) = e^x
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[0]]) // dy/dx = y
        };

        let y0 = DVector::from_vec(vec![1.0]);
        let result_matrix = RK45_ivp_solver_and_mesh(0.0, y0, 1.0, 0.01, ode_system).0;
        println!("final result {:?}", &result_matrix);
        let final_result = result_matrix.column(result_matrix.ncols() - 1);
        println!(
            "result matrix shape: ({}, {})",
            result_matrix.nrows(),
            result_matrix.ncols()
        );

        assert_abs_diff_eq!(final_result[0], 1.0_f64.exp(), epsilon = 1e-4);
    }

    #[test]
    fn test_rk4_linear_system() {
        // Test system: y1' = y2, y2' = -y1 (harmonic oscillator)
        // With y1(0) = 1, y2(0) = 0, exact: y1(x) = cos(x), y2(x) = -sin(x)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], -y[0]]) };

        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        println!("Input y0 dimensions: {}", y0.len());
        let result_matrix =
            RK45_ivp_solver_and_mesh(0.0, y0, std::f64::consts::PI / 2.0, 0.001, ode_system).0;
        let final_result = result_matrix.column(result_matrix.ncols() - 1);
        println!(
            "Result matrix shape: ({}, {})",
            result_matrix.nrows(),
            result_matrix.ncols()
        );
        println!("final result {:?}", final_result);
        assert_eq!(result_matrix.nrows(), 2); // Should have 2 state variables
        assert!(result_matrix.ncols() > 1); // Should have multiple time steps
        assert_abs_diff_eq!(final_result[0], 0.0, epsilon = 1e-6); // y'(pi/2)= cos(π/2) = 0
        assert_abs_diff_eq!(final_result[1], -1.0, epsilon = 1e-6); //y(pi/2) = -sin(π/2) = -1
    }
}
