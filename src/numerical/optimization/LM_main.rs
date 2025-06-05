use crate::numerical::trust_region_subproblem::solve_trust_region_subproblem;
use nalgebra::{DMatrix, DVector};

/// Configuration parameters for Levenberg-Marquardt algorithm
#[derive(Debug, Clone)]
pub struct LMConfig {
    pub ftol: f64,              // Function tolerance
    pub xtol: f64,              // Parameter tolerance
    pub gtol: f64,              // Gradient tolerance
    pub stepbound: f64,         // Initial step bound factor
    pub max_evaluations: usize, // Maximum function evaluations
    pub scale_diag: bool,       // Whether to scale diagonal matrix
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            stepbound: 100.0,
            max_evaluations: 1000,
            scale_diag: true,
        }
    }
}

/// Termination reasons for the algorithm
#[derive(Debug, Clone, PartialEq)]
pub enum TerminationReason {
    /// Converged successfully
    Converged { ftol: bool, xtol: bool },
    /// Gradient orthogonality condition satisfied
    Orthogonal,
    /// Residuals are essentially zero
    ResidualsZero,
    /// Maximum evaluations reached
    MaxEvaluationsReached,
    /// No improvement possible due to machine precision
    NoImprovementPossible(String),
    /// Numerical error occurred
    NumericalError(String),
    /// User function evaluation failed
    FunctionEvaluationFailed,
}

/// Result of Levenberg-Marquardt optimization
#[derive(Debug, Clone)]
pub struct LMResult {
    pub parameters: DVector<f64>,
    pub residuals: DVector<f64>,
    pub objective_value: f64,
    pub termination: TerminationReason,
    pub num_evaluations: usize,
    pub num_jacobian_evaluations: usize,
}

/// Trait for problems that can be solved with Levenberg-Marquardt
pub trait LeastSquaresProblem {
    /// Evaluate residuals at given parameters
    fn residuals(&self, params: &DVector<f64>) -> Option<DVector<f64>>;

    /// Evaluate Jacobian at given parameters
    fn jacobian(&self, params: &DVector<f64>) -> Option<DMatrix<f64>>;

    /// Get number of parameters
    fn num_parameters(&self) -> usize;

    /// Get number of residuals
    fn num_residuals(&self) -> usize;
}

/// Main Levenberg-Marquardt optimization function
pub fn levenberg_marquardt<P: LeastSquaresProblem>(
    problem: &P,
    initial_params: DVector<f64>,
    config: LMConfig,
) -> LMResult {
    const MACHINE_EPS: f64 = f64::EPSILON;
    const P1: f64 = 0.1;
    const P0001: f64 = 1e-4;

    // Initialize state
    let mut params = initial_params;
    let n = problem.num_parameters();
    let m = problem.num_residuals();

    // Validate dimensions
    if params.len() != n {
        return LMResult {
            parameters: params,
            residuals: DVector::zeros(m),
            objective_value: f64::INFINITY,
            termination: TerminationReason::NumericalError(
                "Parameter dimension mismatch".to_string(),
            ),
            num_evaluations: 0,
            num_jacobian_evaluations: 0,
        };
    }

    // Initialize counters and state variables
    let mut num_evaluations = 0;
    let mut num_jacobian_evaluations = 0;
    let mut lambda = 0.0;
    let mut delta = 0.0;
    let mut first_iteration = true;

    // Initialize diagonal scaling matrix
    let mut diag: DVector<f64> = DVector::from_element(n, 1.0);

    // Evaluate initial point
    let mut residuals = match problem.residuals(&params) {
        Some(r) => r,
        None => {
            return create_error_result(params, TerminationReason::FunctionEvaluationFailed, 0, 0);
        }
    };
    num_evaluations += 1;

    let mut residuals_norm = residuals.norm();
    let mut objective_value = 0.5 * residuals_norm * residuals_norm;

    // Check for zero residuals
    if residuals_norm <= MACHINE_EPS.sqrt() {
        return LMResult {
            parameters: params,
            residuals,
            objective_value,
            termination: TerminationReason::ResidualsZero,
            num_evaluations,
            num_jacobian_evaluations,
        };
    }

    // MAIN ITERATION LOOP
    loop {
        // STEP 1: COMPUTE JACOBIAN
        let jacobian = match problem.jacobian(&params) {
            Some(j) => j,
            None => {
                return create_error_result(
                    params,
                    TerminationReason::FunctionEvaluationFailed,
                    num_evaluations,
                    num_jacobian_evaluations,
                );
            }
        };
        num_jacobian_evaluations += 1;

        // Validate Jacobian dimensions
        if jacobian.nrows() != m || jacobian.ncols() != n {
            return create_error_result(
                params,
                TerminationReason::NumericalError("Jacobian dimension mismatch".to_string()),
                num_evaluations,
                num_jacobian_evaluations,
            );
        }

        let jacobian_t = jacobian.transpose();

        // STEP 2: UPDATE DIAGONAL SCALING
        if config.scale_diag {
            for j in 0..n {
                let col_norm: f64 = jacobian.column(j).norm();
                if col_norm > 0.0 {
                    diag[j] = diag[j].max(col_norm);
                }
            }
        }

        // STEP 3: COMPUTE GRADIENT AND CHECK OPTIMALITY
        let gradient = &jacobian_t * &residuals;
        let gnorm = compute_scaled_gradient_norm(&jacobian, &residuals, residuals_norm);

        // Check gradient-based termination
        if gnorm <= config.gtol {
            return LMResult {
                parameters: params,
                residuals,
                objective_value,
                termination: TerminationReason::Orthogonal,
                num_evaluations,
                num_jacobian_evaluations,
            };
        }

        // STEP 4: INITIALIZE TRUST REGION ON FIRST ITERATION
        if first_iteration {
            let xnorm = compute_scaled_norm(&params, &diag);
            delta = if xnorm == 0.0 {
                config.stepbound
            } else {
                config.stepbound * xnorm
            };
            first_iteration = false;
        }

        // STEP 5: TRUST REGION SUBPROBLEM LOOP
        loop {
            // Solve trust region subproblem
            let tr_result = match solve_trust_region_subproblem(
                &jacobian,
                &jacobian_t,
                &residuals,
                &diag,
                delta,
                lambda,
            ) {
                Ok(result) => result,
                Err(msg) => {
                    return create_error_result(
                        params,
                        TerminationReason::NumericalError(format!(
                            "Trust region solver failed: {}",
                            msg
                        )),
                        num_evaluations,
                        num_jacobian_evaluations,
                    );
                }
            };

            lambda = tr_result.lambda;
            let step = tr_result.step;
            let pnorm = tr_result.scaled_step_norm;

            // STEP 6: EVALUATE NEW POINT
            let new_params = &params - &step;
            let new_residuals = match problem.residuals(&new_params) {
                Some(r) => r,
                None => {
                    return create_error_result(
                        params,
                        TerminationReason::FunctionEvaluationFailed,
                        num_evaluations,
                        num_jacobian_evaluations,
                    );
                }
            };
            num_evaluations += 1;

            let new_residuals_norm = new_residuals.norm();
            let new_objective_value = 0.5 * new_residuals_norm * new_residuals_norm;

            // STEP 7: COMPUTE REDUCTION RATIO
            let (predicted_reduction, actual_reduction, ratio) = compute_reduction_ratio(
                &jacobian,
                &step,
                &residuals,
                residuals_norm,
                new_residuals_norm,
                lambda,
                pnorm,
            );

            // STEP 8: UPDATE TRUST REGION RADIUS
            let (new_delta, new_lambda) = update_trust_region_radius(
                delta,
                lambda,
                ratio,
                pnorm,
                predicted_reduction,
                actual_reduction,
                residuals_norm,
                new_residuals_norm,
            );
            delta = new_delta;
            // !!!!!
            //   let new_lambda = 1e-3;
            lambda = new_lambda;

            // STEP 9: ACCEPT OR REJECT STEP
            let step_accepted = ratio >= P0001;

            if step_accepted {
                // Accept the step
                params = new_params;
                residuals = new_residuals;
                residuals_norm = new_residuals_norm;
                objective_value = new_objective_value;
                break; // Exit trust region loop, recompute Jacobian
            }

            // STEP 10: CONVERGENCE AND TERMINATION TESTS

            // Check for zero residuals
            if residuals_norm <= MACHINE_EPS.sqrt() {
                return LMResult {
                    parameters: params,
                    residuals,
                    objective_value,
                    termination: TerminationReason::ResidualsZero,
                    num_evaluations,
                    num_jacobian_evaluations,
                };
            }

            // Function tolerance test
            let ftol_satisfied = predicted_reduction.abs() <= config.ftol
                && actual_reduction.abs() <= config.ftol
                && ratio * 0.5 <= 1.0;

            // Parameter tolerance test
            let xnorm = compute_scaled_norm(&params, &diag);
            let xtol_satisfied = delta <= config.xtol * xnorm;

            if ftol_satisfied || xtol_satisfied {
                return LMResult {
                    parameters: params,
                    residuals,
                    objective_value,
                    termination: TerminationReason::Converged {
                        ftol: ftol_satisfied,
                        xtol: xtol_satisfied,
                    },
                    num_evaluations,
                    num_jacobian_evaluations,
                };
            }

            // Maximum evaluations check
            if num_evaluations >= config.max_evaluations {
                return LMResult {
                    parameters: params,
                    residuals,
                    objective_value,
                    termination: TerminationReason::MaxEvaluationsReached,
                    num_evaluations,
                    num_jacobian_evaluations,
                };
            }

            // Check for no improvement possible due to machine precision
            if predicted_reduction.abs() <= MACHINE_EPS
                && actual_reduction.abs() <= MACHINE_EPS
                && ratio * 0.5 <= 1.0
            {
                return LMResult {
                    parameters: params,
                    residuals,
                    objective_value,
                    termination: TerminationReason::NoImprovementPossible("ftol".to_string()),
                    num_evaluations,
                    num_jacobian_evaluations,
                };
            }

            if delta <= MACHINE_EPS * xnorm {
                return LMResult {
                    parameters: params,
                    residuals,
                    objective_value,
                    termination: TerminationReason::NoImprovementPossible("xtol".to_string()),
                    num_evaluations,
                    num_jacobian_evaluations,
                };
            }

            if gnorm <= MACHINE_EPS {
                return LMResult {
                    parameters: params,
                    residuals,
                    objective_value,
                    termination: TerminationReason::NoImprovementPossible("gtol".to_string()),
                    num_evaluations,
                    num_jacobian_evaluations,
                };
            }

            // Continue trust region loop with updated lambda and delta
        }
    }
}

/// Compute scaled gradient norm for optimality check
fn compute_scaled_gradient_norm(
    jacobian: &DMatrix<f64>,
    residuals: &DVector<f64>,
    residuals_norm: f64,
) -> f64 {
    let mut max_scaled_grad: f64 = 0.0;

    for j in 0..jacobian.ncols() {
        let col = jacobian.column(j);
        let col_norm = col.norm();
        if col_norm > 0.0 && residuals_norm > 0.0 {
            let grad_component = col.dot(residuals).abs();
            let scaled_grad: f64 = grad_component / (col_norm * residuals_norm);
            max_scaled_grad = max_scaled_grad.max(scaled_grad);
        }
    }

    max_scaled_grad
}

/// Compute ||D*x|| where D is diagonal
fn compute_scaled_norm(x: &DVector<f64>, diag: &DVector<f64>) -> f64 {
    x.iter()
        .zip(diag.iter())
        .map(|(&xi, &di)| (di * xi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute predicted and actual reduction and their ratio
fn compute_reduction_ratio(
    jacobian: &DMatrix<f64>,
    step: &DVector<f64>,
    residuals: &DVector<f64>,
    residuals_norm: f64,
    new_residuals_norm: f64,
    lambda: f64,
    pnorm: f64,
) -> (f64, f64, f64) {
    // Predicted reduction from quadratic model
    let temp1 = ((jacobian * step).norm() / residuals_norm).powi(2);
    let temp2 = ((lambda.sqrt() * pnorm) / residuals_norm).powi(2);
    let predicted_reduction = temp1 + temp2 / 2.0;

    // Actual reduction
    let actual_reduction = if new_residuals_norm * 0.1 < residuals_norm {
        1.0 - (new_residuals_norm / residuals_norm).powi(2)
    } else {
        -1.0
    };

    // Ratio
    let ratio = if predicted_reduction == 0.0 {
        0.0
    } else {
        actual_reduction / predicted_reduction
    };

    (predicted_reduction, actual_reduction, ratio)
}
/*Key Features:

Poor Agreement (ratio ≤ 0.25):

Shrinks the trust region radius
Increases λ to make the algorithm more like steepest descent
Uses directional derivative information when the function actually increased
Good Agreement (ratio ≥ 0.75 or λ = 0):

Expands the trust region radius
Decreases λ to make the algorithm more like Gauss-Newton
Special handling for pure Gauss-Newton steps (λ = 0)
Moderate Agreement (0.25 < ratio < 0.75 and λ > 0):

Keeps current trust region radius and λ values
Safety Bounds:

Ensures λ never becomes negative
Prevents trust region radius from becoming too small
Limits the shrinkage factor to prevent overly conservative steps
Adaptive Shrinkage:

Uses different shrinkage strategies based on whether the function improved or got worse
Considers the directional derivative when the function increased */
/// Update trust region radius and lambda based on reduction ratio

fn update_trust_region_radius(
    delta: f64,
    lambda: f64,
    ratio: f64,
    pnorm: f64,
    predicted_reduction: f64,
    actual_reduction: f64,
    residuals_norm: f64,
    new_residuals_norm: f64,
) -> (f64, f64) {
    const P1: f64 = 0.1;
    const P25: f64 = 0.25;
    const P75: f64 = 0.75;
    const P5: f64 = 0.5;

    let mut new_delta = delta;
    let mut new_lambda = lambda;

    if ratio <= P25 {
        // Poor agreement between model and actual reduction
        // Shrink trust region and increase lambda

        let mut temp = if actual_reduction >= 0.0 {
            // Some reduction achieved
            P5
        } else {
            // Function increased, use directional derivative
            let dir_derivative = -(predicted_reduction * 2.0 - actual_reduction);
            if dir_derivative != 0.0 {
                P5 * dir_derivative / (dir_derivative + P5 * actual_reduction)
            } else {
                P5
            }
        }; // temp

        // Ensure minimum shrinkage factor
        if new_residuals_norm * P1 >= residuals_norm || temp < P1 {
            temp = P1;
        }

        // Update delta: shrink but don't make it too small relative to step norm
        new_delta = temp * delta.min(pnorm * 10.0);

        // Increase lambda (make more like steepest descent)
        new_lambda = lambda / temp;
    } else if lambda == 0.0 || ratio >= P75 {
        // Good agreement between model and actual reduction
        // Expand trust region and decrease lambda

        // Expand trust region
        new_delta = pnorm / P5;

        // Decrease lambda (make more like Gauss-Newton)
        new_lambda = lambda * P5;
    } else {
        // Moderate agreement - keep current values
        // This happens when 0.25 < ratio < 0.75 and lambda > 0
        new_delta = delta;
        new_lambda = lambda;
    }

    // Ensure lambda doesn't become negative
    new_lambda = new_lambda.max(0.0);

    // Ensure delta doesn't become too small
    new_delta = new_delta.max(f64::EPSILON * 100.0);

    (new_delta, new_lambda)
}

/// Helper function to create error result
fn create_error_result(
    params: DVector<f64>,
    termination: TerminationReason,
    num_evaluations: usize,
    num_jacobian_evaluations: usize,
) -> LMResult {
    LMResult {
        parameters: params.clone(),
        residuals: DVector::zeros(0), // Empty residuals for error case
        objective_value: f64::INFINITY,
        termination,
        num_evaluations,
        num_jacobian_evaluations,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                         TESTING
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_update_trust_region_poor_ratio() {
        let delta = 1.0;
        let lambda = 0.1;
        let ratio = 0.1; // Poor ratio
        let pnorm = 0.8;
        let predicted_reduction = 0.5;
        let actual_reduction = 0.05; // Much less than predicted
        let residuals_norm = 1.0;
        let new_residuals_norm = 0.95;

        let (new_delta, new_lambda) = update_trust_region_radius(
            delta,
            lambda,
            ratio,
            pnorm,
            predicted_reduction,
            actual_reduction,
            residuals_norm,
            new_residuals_norm,
        );

        // Should shrink delta and increase lambda
        assert!(new_delta < delta);
        assert!(new_lambda > lambda);
    }

    #[test]
    fn test_update_trust_region_good_ratio() {
        let delta = 1.0;
        let lambda = 0.1;
        let ratio = 0.8; // Good ratio
        let pnorm = 0.8;
        let predicted_reduction = 0.5;
        let actual_reduction = 0.4; // Close to predicted
        let residuals_norm = 1.0;
        let new_residuals_norm = 0.6;

        let (new_delta, new_lambda) = update_trust_region_radius(
            delta,
            lambda,
            ratio,
            pnorm,
            predicted_reduction,
            actual_reduction,
            residuals_norm,
            new_residuals_norm,
        );

        // Should expand delta and decrease lambda
        assert!(new_delta > delta);
        assert!(new_lambda < lambda);
    }

    #[test]
    fn test_update_trust_region_gauss_newton() {
        let delta = 1.0;
        let lambda = 0.0; // Gauss-Newton step
        let ratio = 0.5; // Moderate ratio
        let pnorm = 0.8;
        let predicted_reduction = 0.5;
        let actual_reduction = 0.25;
        let residuals_norm = 1.0;
        let new_residuals_norm = 0.75;

        let (new_delta, new_lambda) = update_trust_region_radius(
            delta,
            lambda,
            ratio,
            pnorm,
            predicted_reduction,
            actual_reduction,
            residuals_norm,
            new_residuals_norm,
        );

        // With lambda=0, should expand regardless of moderate ratio
        assert!(new_delta > delta);
        assert_relative_eq!(new_lambda, 0.0);
    }

    #[test]
    fn test_update_trust_region_function_increase() {
        let delta = 1.0;
        let lambda = 0.1;
        let ratio = -0.5; // Function increased
        let pnorm = 0.8;
        let predicted_reduction = 0.5;
        let actual_reduction = -0.25; // Function got worse
        let residuals_norm = 1.0;
        let new_residuals_norm = 1.25;

        let (new_delta, new_lambda) = update_trust_region_radius(
            delta,
            lambda,
            ratio,
            pnorm,
            predicted_reduction,
            actual_reduction,
            residuals_norm,
            new_residuals_norm,
        );

        // Should significantly shrink delta and increase lambda
        assert!(new_delta < delta);
        assert!(new_lambda > lambda);
        assert!(new_delta <= 0.1 * delta); // Should shrink by at least factor of 10
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use nalgebra::{DMatrix, DVector, dmatrix, dvector};

    // Test problem implementations

    /// Simple quadratic problem: f(x) = 0.5 * ||Ax - b||²
    struct QuadraticProblem {
        a: DMatrix<f64>,
        b: DVector<f64>,
    }

    impl LeastSquaresProblem for QuadraticProblem {
        fn residuals(&self, params: &DVector<f64>) -> Option<DVector<f64>> {
            Some(&self.a * params - &self.b)
        }

        fn jacobian(&self, _params: &DVector<f64>) -> Option<DMatrix<f64>> {
            Some(self.a.clone())
        }

        fn num_parameters(&self) -> usize {
            self.a.ncols()
        }
        fn num_residuals(&self) -> usize {
            self.a.nrows()
        }
    }

    /// Rosenbrock function: f(x,y) = [10*(y - x²), 1 - x]
    struct RosenbrockProblem;

    impl LeastSquaresProblem for RosenbrockProblem {
        fn residuals(&self, params: &DVector<f64>) -> Option<DVector<f64>> {
            if params.len() != 2 {
                return None;
            }
            let x = params[0];
            let y = params[1];
            Some(dvector![10.0 * (y - x * x), 1.0 - x])
        }

        fn jacobian(&self, params: &DVector<f64>) -> Option<DMatrix<f64>> {
            if params.len() != 2 {
                return None;
            }
            let x = params[0];
            Some(dmatrix![
                -20.0 * x, 10.0;
                -1.0,       0.0;
            ])
        }

        fn num_parameters(&self) -> usize {
            2
        }
        fn num_residuals(&self) -> usize {
            2
        }
    }

    /// Powell's function: more challenging test case
    struct PowellProblem;

    impl LeastSquaresProblem for PowellProblem {
        fn residuals(&self, params: &DVector<f64>) -> Option<DVector<f64>> {
            if params.len() != 4 {
                return None;
            }
            let x1 = params[0];
            let x2 = params[1];
            let x3 = params[2];
            let x4 = params[3];

            Some(dvector![
                x1 + 10.0 * x2,
                (5.0_f64).sqrt() * (x3 - x4),
                (x2 - 2.0 * x3).powi(2),
                (10.0_f64).sqrt() * (x1 - x4).powi(2)
            ])
        }

        fn jacobian(&self, params: &DVector<f64>) -> Option<DMatrix<f64>> {
            if params.len() != 4 {
                return None;
            }
            let x1 = params[0];
            let x2 = params[1];
            let x3 = params[2];
            let x4 = params[3];

            Some(dmatrix![
                1.0,                    10.0,                   0.0,                        0.0;
                0.0,                    0.0,                    (5.0_f64).sqrt(),          -(5.0_f64).sqrt();
                0.0,                    2.0 * (x2 - 2.0 * x3), -4.0 * (x2 - 2.0 * x3),    0.0;
                2.0 * (10.0_f64).sqrt() * (x1 - x4), 0.0,      0.0,                        -2.0 * (10.0_f64).sqrt() * (x1 - x4);
            ])
        }

        fn num_parameters(&self) -> usize {
            4
        }
        fn num_residuals(&self) -> usize {
            4
        }
    }

    /// Exponential fitting problem: fit y = a * exp(b * x) + c
    struct ExponentialFitProblem {
        x_data: Vec<f64>,
        y_data: Vec<f64>,
    }

    impl LeastSquaresProblem for ExponentialFitProblem {
        fn residuals(&self, params: &DVector<f64>) -> Option<DVector<f64>> {
            if params.len() != 3 {
                return None;
            }
            let a = params[0];
            let b = params[1];
            let c = params[2];

            let residuals: Vec<f64> = self
                .x_data
                .iter()
                .zip(self.y_data.iter())
                .map(|(&x, &y)| a * (b * x).exp() + c - y)
                .collect();

            Some(DVector::from_vec(residuals))
        }

        fn jacobian(&self, params: &DVector<f64>) -> Option<DMatrix<f64>> {
            if params.len() != 3 {
                return None;
            }
            let a = params[0];
            let b = params[1];

            let mut jac = DMatrix::zeros(self.x_data.len(), 3);
            for (i, &x) in self.x_data.iter().enumerate() {
                let exp_bx = (b * x).exp();
                jac[(i, 0)] = exp_bx; // ∂/∂a
                jac[(i, 1)] = a * x * exp_bx; // ∂/∂b  
                jac[(i, 2)] = 1.0; // ∂/∂c
            }

            Some(jac)
        }

        fn num_parameters(&self) -> usize {
            3
        }
        fn num_residuals(&self) -> usize {
            self.x_data.len()
        }
    }

    /// Problem that always fails function evaluation
    struct FailingProblem;

    impl LeastSquaresProblem for FailingProblem {
        fn residuals(&self, _params: &DVector<f64>) -> Option<DVector<f64>> {
            None
        }
        fn jacobian(&self, _params: &DVector<f64>) -> Option<DMatrix<f64>> {
            None
        }
        fn num_parameters(&self) -> usize {
            2
        }
        fn num_residuals(&self) -> usize {
            2
        }
    }

    // Basic functionality tests

    #[test]
    fn test_simple_quadratic_problem() {
        // Solve Ax = b where A = I, b = [1, 2]
        let problem = QuadraticProblem {
            a: DMatrix::identity(2, 2),
            b: dvector![1.0, 2.0],
        };

        let initial = dvector![0.0, 0.0];
        let config = LMConfig::default();

        let result = levenberg_marquardt(&problem, initial, config);

        assert!(matches!(
            result.termination,
            TerminationReason::Converged { .. }
        ));
        assert_relative_eq!(result.parameters[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.parameters[1], 2.0, epsilon = 1e-10);
        assert!(result.objective_value < 1e-20);
    }

    #[test]
    fn test_overdetermined_system() {
        // More equations than unknowns
        let problem = QuadraticProblem {
            a: dmatrix![
                1.0, 0.0;
                0.0, 1.0;
                1.0, 1.0;
            ],
            b: dvector![1.0, 2.0, 3.1], // Slightly inconsistent
        };

        let initial = dvector![0.0, 0.0];
        let result = levenberg_marquardt(&problem, initial, LMConfig::default());

        assert!(matches!(
            result.termination,
            TerminationReason::Converged { .. }
        ));
        // Should find least squares solution
        assert_relative_eq!(result.parameters[0], 1.05, epsilon = 1e-10);
        assert_relative_eq!(result.parameters[1], 2.05, epsilon = 1e-10);
    }

    #[test]
    fn test_rosenbrock_function() {
        let problem = RosenbrockProblem;
        let initial = dvector![-1.2, 1.0]; // Standard starting point

        let mut config = LMConfig::default();
        config.max_evaluations = 1000;
        config.ftol = 1e-12;
        config.xtol = 1e-12;

        let result = levenberg_marquardt(&problem, initial, config);

        assert!(matches!(
            result.termination,
            TerminationReason::Converged { .. }
        ));
        assert_relative_eq!(result.parameters[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.parameters[1], 1.0, epsilon = 1e-6);
        assert!(result.objective_value < 1e-10);
    }

    #[test]
    fn test_powell_function() {
        let problem = PowellProblem;
        let initial = dvector![3.0, -1.0, 0.0, 1.0];

        let mut config = LMConfig::default();
        config.max_evaluations = 2000;

        let result = levenberg_marquardt(&problem, initial, config);

        assert!(
            matches!(result.termination, TerminationReason::Converged { .. })
                || matches!(result.termination, TerminationReason::Orthogonal)
        );
        // Powell's function has minimum at origin
        for &param in result.parameters.iter() {
            assert_abs_diff_eq!(param, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_exponential_fitting() {
        // Generate synthetic data: y = 2 * exp(0.5 * x) + 1
        let x_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.2).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| 2.0 * (0.5 * x).exp() + 1.0 + 0.01 * (x * 10.0).sin()) // Add small noise
            .collect();

        let problem = ExponentialFitProblem { x_data, y_data };
        let initial = dvector![1.0, 1.0, 0.0]; // Initial guess

        let mut config = LMConfig::default();
        config.max_evaluations = 500;

        let result = levenberg_marquardt(&problem, initial, config);

        assert!(matches!(
            result.termination,
            TerminationReason::Converged { .. }
        ));
        assert_relative_eq!(result.parameters[0], 2.0, epsilon = 1e-2); // a ≈ 2
        assert_relative_eq!(result.parameters[1], 0.5, epsilon = 1e-2); // b ≈ 0.5  
        assert_relative_eq!(result.parameters[2], 1.0, epsilon = 1e-2); // c ≈ 1
    }

    // Edge cases and error conditions

    #[test]
    fn test_zero_residuals() {
        let problem = QuadraticProblem {
            a: DMatrix::identity(2, 2),
            b: dvector![1.0, 2.0],
        };

        // Start at the exact solution
        let initial = dvector![1.0, 2.0];
        let result = levenberg_marquardt(&problem, initial, LMConfig::default());

        assert!(matches!(
            result.termination,
            TerminationReason::ResidualsZero
        ));
        assert!(result.objective_value < 1e-20);
    }

    #[test]
    fn test_function_evaluation_failure() {
        let problem = FailingProblem;
        let initial = dvector![1.0, 2.0];
        let result = levenberg_marquardt(&problem, initial, LMConfig::default());

        assert!(matches!(
            result.termination,
            TerminationReason::FunctionEvaluationFailed
        ));
    }

    #[test]
    fn test_max_evaluations_reached() {
        let problem = RosenbrockProblem;
        let initial = dvector![-1.2, 1.0];

        let mut config = LMConfig::default();
        config.max_evaluations = 5; // Very low limit

        let result = levenberg_marquardt(&problem, initial, config);

        assert!(matches!(
            result.termination,
            TerminationReason::MaxEvaluationsReached
        ));
        assert_eq!(result.num_evaluations, 5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let problem = QuadraticProblem {
            a: DMatrix::identity(2, 2),
            b: dvector![1.0, 2.0],
        };

        // Wrong dimension initial parameters
        let initial = dvector![1.0, 2.0, 3.0];
        let result = levenberg_marquardt(&problem, initial, LMConfig::default());

        assert!(matches!(
            result.termination,
            TerminationReason::NumericalError(_)
        ));
    }

    // Configuration parameter tests

    #[test]
    fn test_different_tolerances() {
        let problem = RosenbrockProblem;
        let initial = dvector![-1.2, 1.0];

        // Test with loose tolerance
        let mut config_loose = LMConfig::default();
        config_loose.ftol = 1e-4;
        config_loose.xtol = 1e-4;

        let result_loose = levenberg_marquardt(&problem, initial.clone(), config_loose);

        // Test with tight tolerance
        let mut config_tight = LMConfig::default();
        config_tight.ftol = 1e-12;
        config_tight.xtol = 1e-12;
        config_tight.max_evaluations = 2000;

        let result_tight = levenberg_marquardt(&problem, initial, config_tight);

        // Both should converge, but tight tolerance should be more accurate
        assert!(matches!(
            result_loose.termination,
            TerminationReason::Converged { .. }
        ));
        assert!(matches!(
            result_tight.termination,
            TerminationReason::Converged { .. }
        ));
    }
}
