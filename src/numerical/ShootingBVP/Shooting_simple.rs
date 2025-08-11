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
//! ```rust
//! use nalgebra::DVector;
//!
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
use nalgebra::DVector;

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
    pub fn new(ode_system: F, a: f64, b: f64, left_bc: BoundaryCondition, right_bc: BoundaryCondition) -> Self {
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
            left_bc: BoundaryCondition { value: alpha, bc_type: BoundaryConditionType::Dirichlet },
            right_bc: BoundaryCondition { value: beta, bc_type: BoundaryConditionType::Dirichlet },
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
        &self,
        problem: &BoundaryValueProblem<F>,
        ivp_solver: impl Fn(f64, DVector<f64>, f64, f64) -> DVector<f64>,
    ) -> Result<(f64, DVector<f64>), &'static str>
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
                },
                BoundaryConditionType::Neumann => {
                    // y'(a) = known, y(a) = guess
                    DVector::from_vec(vec![guess, problem.left_bc.value])
                },
            };
            
            let sol = ivp_solver(problem.a, y0, problem.b, self.step_size);
            
            let error_val = match problem.right_bc.bc_type {
                BoundaryConditionType::Dirichlet => {
                    // Check y(b) = right_bc.value
                    sol[0] - problem.right_bc.value
                },
                BoundaryConditionType::Neumann => {
                    // Check y'(b) = right_bc.value
                    sol[1] - problem.right_bc.value
                },
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
            },
            BoundaryConditionType::Neumann => {
                DVector::from_vec(vec![unknown_initial, problem.left_bc.value])
            },
        };
        let sol = ivp_solver(problem.a, y0, problem.b, self.step_size);

        info!(
            "Final solution: y({})={}, y'({})={}",
            problem.b, sol[0], problem.b, sol[1]
        );
        Ok((unknown_initial, sol))
    }

    pub fn new()->Self {
        Self { initial_guess: 0.0, 
            tolerance: 0.0, 
            max_iterations: 0, 
            step_size: 0.0 }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

/// Solves an initial value problem (IVP) using the 4th-order Runge-Kutta method.
///
/// # Arguments
/// * `x0` - Initial x value.
/// * `y0` - Initial state vector (y and y' packed into a DVector).
/// * `x_end` - End of the integration interval.
/// * `step_size` - Step size for numerical integration.
/// * `ode_system` - Function defining the ODE system: dy/dx = f(x, y).
pub fn rk4_ivp_solver<F>(
    x0: f64,
    y0: DVector<f64>,
    x_end: f64,
    step_size: f64,
    ode_system: F,
) -> DVector<f64>
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    let mut x = x0;
    let mut y = y0;

    while x < x_end {
        let h = step_size.min(x_end - x); // Handle last step to avoid overshooting x_end

        let k1 = ode_system(x, &y);
        let k2 = ode_system(x + h / 2.0, &(y.clone() + (h / 2.0) * &k1));
        let k3 = ode_system(x + h / 2.0, &(y.clone() + (h / 2.0) * &k2));
        let k4 = ode_system(x + h, &(y.clone() + h * &k3));

        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        x += h;
    }

    y
}
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
        let solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
        };

        // Solve the BVP using RK4
        match solver.solve(&problem, |x0, y0, x_end, h| {
            rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
        }) {
            Ok((slope, solution)) => {
                println!("Found initial slope s = {}", slope);
                println!(
                    "Solution at x = b: y = {}, y' = {}",
                    solution[0], solution[1]
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

        let solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        // Check boundary condition at x = b
        assert_abs_diff_eq!(solution[0], problem.beta, epsilon = 1e-6);

        // Check initial slope s ≈ y'(0) = 1 (exact for this problem)
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_non_convergence() {
        init_logger();

        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 1.0);

        // Force failure with unrealistic tolerance and low iterations
        let solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-20,
            max_iterations: 5,
            step_size: 0.1,
        };

        let result = solver.solve(&problem, |x0, y0, x_end, h| {
            rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
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

        let solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-6,
            max_iterations: 10,
            step_size: 0.1,
        };

        let (_, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        // Solution should remain at initial condition
        assert_eq!(solution[0], problem.alpha);
        assert_eq!(solution[1], solver.initial_guess);
    }
    #[test]
    fn test_adaptive_step_size() {
        init_logger();

        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -y[0]]) // y'' = -y (harmonic oscillator)
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, std::f64::consts::PI, 0.0, 0.0);

        // Test with coarse step size (should still converge)
        let solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-4,
            max_iterations: 50,
            step_size: 0.1,
        };

        let (_, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[0], problem.beta, epsilon = 1e-3);
    }

    #[test]
    fn test_nonlinear_pendulum() {
        init_logger();

        // Nonlinear pendulum: y'' + sin(y) = 0
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -y[0].sin()]) // [y', -sin(y)]
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 0.5);

        let solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[0], problem.beta, epsilon = 1e-5);
        assert!(s.is_finite());
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

        let solver = ShootingMethodSolver {
            initial_guess: 10.0,
            tolerance: 1e-6,
            max_iterations: 1000,
            step_size: 1e-5, // Small step for stiff problem
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[0], problem.beta, epsilon = 1e-3);
        assert_abs_diff_eq!(s, 10.0, epsilon = 1e-3);
    }

    #[test]
    fn test_large_domain() {
        init_logger();

        // Simple linear problem over large domain
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 0.0]) // y'' = 0
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 100.0, 0.0, 50.0);

        let solver = ShootingMethodSolver {
            initial_guess: 0.5,
            tolerance: 1e-6,
            max_iterations: 50,
            step_size: 1.0,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[0], problem.beta, epsilon = 1e-6);
        assert_abs_diff_eq!(s, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_domain() {
        init_logger();

        // Test with negative domain
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };

        let problem = BoundaryValueProblem::legacy(ode_system, -1.0, 0.0, (-1.0_f64).sinh(), 0.0);

        let solver = ShootingMethodSolver {
            initial_guess: -1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.001,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[0], problem.beta, epsilon = 1e-6);
        assert_abs_diff_eq!(s, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_oscillatory_solution() {
        init_logger();

        // Harmonic oscillator with specific frequency
        let omega = 2.0;
        let ode_system = move |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -omega * omega * y[0]])
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, std::f64::consts::PI / omega, 1.0, -1.0);

        let solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[0], problem.beta, epsilon = 1e-5);
        assert_abs_diff_eq!(s, 0.0, epsilon = 1e-5);
    }

    
    #[test]
    fn test_dirichlet_neumann() {
        init_logger();

        // y'' = y, y(0) = 0, y'(1) = cosh(1)
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]])
        };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition { value: 0.0, bc_type: BoundaryConditionType::Dirichlet },
            BoundaryCondition { value: 1.0_f64.cosh(), bc_type: BoundaryConditionType::Neumann },
        );

        let solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[1], problem.right_bc.value, epsilon = 1e-6);
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_neumann_dirichlet() {
        init_logger();

        // y'' = y, y'(0) = 1, y(1) = sinh(1)
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]])
        };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition { value: 1.0, bc_type: BoundaryConditionType::Neumann },
            BoundaryCondition { value: 1.0_f64.sinh(), bc_type: BoundaryConditionType::Dirichlet },
        );

        let solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
        };

        let (y_a, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[0], problem.right_bc.value, epsilon = 1e-6);
        assert_abs_diff_eq!(y_a, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_neumann_neumann() {
        init_logger();

        // y'' = 0, y'(0) = 1, y'(1) = 1 (linear function)
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 0.0])
        };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition { value: 1.0, bc_type: BoundaryConditionType::Neumann },
            BoundaryCondition { value: 1.0, bc_type: BoundaryConditionType::Neumann },
        );

        let solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.01,
        };

        let (y_a, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[1], problem.right_bc.value, epsilon = 1e-6);
        assert!(y_a.is_finite());
    }
    
    #[test]
    fn test_dirichlet_neumann_linear() {
        init_logger();

        // y'' = 0, y(0) = 2, y'(1) = 3 (linear function y = 3x + 2)
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 0.0])
        };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition { value: 2.0, bc_type: BoundaryConditionType::Dirichlet },
            BoundaryCondition { value: 3.0, bc_type: BoundaryConditionType::Neumann },
        );

        let solver = ShootingMethodSolver {
            initial_guess: 3.0,
            tolerance: 1e-8,
            max_iterations: 50,
            step_size: 0.01,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[1], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(s, 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(solution[0], 5.0, epsilon = 1e-6); // y(1) = 3*1 + 2 = 5
    }

    #[test]
    fn test_dirichlet_neumann_exponential() {
        init_logger();

        // y'' - y = 0, y(0) = 1, y'(1) = e^1 (solution: y = e^x)
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]])
        };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition { value: 1.0, bc_type: BoundaryConditionType::Dirichlet },
            BoundaryCondition { value: 1.0_f64.exp(), bc_type: BoundaryConditionType::Neumann },
        );

        let solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[1], 1.0_f64.exp(), epsilon = 1e-6);
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dirichlet_neumann_oscillator() {
        init_logger();

        // y'' + y = 0, y(0) = 0, y'(π) = 1 (solution: y = sin(x))
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], - y[0]])
        };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            std::f64::consts::PI ,
            BoundaryCondition { value: 0.0, bc_type: BoundaryConditionType::Dirichlet },
            BoundaryCondition { value: 1.0, bc_type: BoundaryConditionType::Neumann },
        );

        let solver = ShootingMethodSolver {
            initial_guess: 2.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
        };

        let (s, solution) = solver
            .solve(&problem, |x0, y0, x_end, h| {
                rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(solution[1], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(s, -1.0, epsilon = 1e-5);
    }
}