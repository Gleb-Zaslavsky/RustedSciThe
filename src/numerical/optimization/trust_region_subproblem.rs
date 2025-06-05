use log::{info, warn};
use nalgebra::{Cholesky, DMatrix, DVector};
use std::f64;
/*

*/
/// Result of trust region subproblem solution
#[derive(Debug, Clone)]
pub struct TrustRegionResult {
    /// The computed step vector p
    pub step: DVector<f64>,
    /// The Levenberg-Marquardt parameter lambda
    pub lambda: f64,
    /// Norm of the scaled step ||D*p||
    pub scaled_step_norm: f64,
    /// Whether the solution is on the trust region boundary
    pub on_boundary: bool,
}

/// Solve the trust region subproblem:
/// min ||J*p - r||² subject to ||D*p|| ≤ delta
///
/// This is equivalent to solving: (J^T*J + λ*D²)*p = J^T*r
/// where λ ≥ 0 is chosen such that ||D*p|| ≤ delta
///
/// # Arguments
/// * `jacobian` - The Jacobian matrix J (m×n)
/// * `jacobian_t` - The transposed Jacobian J^T (n×m)
/// * `residuals` - The residual vector r (m×1)
/// * `diag` - Diagonal scaling vector D (n×1)
/// * `delta` - Trust region radius
/// * `lambda_prev` - Previous lambda value (for warm start)
///
/// # Returns
/// TrustRegionResult containing the step and related information
/*This implementation provides:

Complete Trust Region Solver: Handles both the case where the Gauss-Newton step is within the trust region and the case where we need to find λ > 0.

Robust Linear System Solving: Uses Cholesky decomposition when possible (for positive definite systems) and falls back to LU decomposition.

Root Finding for λ: Uses Newton's method to find the λ value that makes ||D*p|| = δ when the Gauss-Newton step is outside the trust region.

Good Initial Guess: Provides heuristics for estimating an initial λ value to speed up convergence.

Error Handling: Comprehensive error checking for dimension mismatches and numerical issues.

Efficient Implementation: Reuses computed matrices and avoids unnecessary allocations. */
pub fn solve_trust_region_subproblem(
    jacobian: &DMatrix<f64>,
    jacobian_t: &DMatrix<f64>,
    residuals: &DVector<f64>,
    diag: &DVector<f64>,
    delta: f64,
    lambda_prev: f64,
) -> Result<TrustRegionResult, &'static str> {
    let n = jacobian.ncols();
    let m = jacobian.nrows();

    // Validate inputs
    if jacobian_t.nrows() != n || jacobian_t.ncols() != m {
        return Err("Jacobian transpose dimensions mismatch");
    }
    if residuals.len() != m {
        return Err("Residuals dimension mismatch");
    }
    if diag.len() != n {
        return Err("Diagonal vector dimension mismatch");
    }
    if delta <= 0.0 {
        return Err("Trust region radius must be positive");
    }

    // Compute J^T * r (gradient of ||J*p - r||²)
    let jt_r = jacobian_t * residuals;

    // Compute J^T * J
    let jtj = jacobian_t * jacobian;

    // Create diagonal matrix D² for regularization
    let diag_squared: DVector<f64> = diag.map(|x| x * x);

    // Try Gauss-Newton step first (lambda = 0)  (J^T*J + λ*D²)*p = J^T*r => J^T*J*p = J^T*r or J*p = r (Gauss-Newton step)
    if let Some(step) = try_solve_linear_system(&jtj, &jt_r, &diag_squared, 0.0) {
        let scaled_norm = compute_scaled_norm(&step, diag);

        if scaled_norm <= delta {
            info!(
                "Gauss-Newton step within trust region: scaled norm = {:.4e}, delta = {:.4e}, step = {:?}",
                scaled_norm, delta, step
            );
            println!(
                "Gauss-Newton step within trust region: scaled norm = {:.4e}, delta = {:.4e}, step = {:?}",
                scaled_norm, delta, step
            );
            return Ok(TrustRegionResult {
                step,
                lambda: 0.0,
                scaled_step_norm: scaled_norm,
                on_boundary: false,
            });
        }
    }

    // Need to find lambda > 0 such that ||D*p|| = delta
    let lambda = find_lambda_root_finding(&jtj, &jt_r, &diag_squared, delta, lambda_prev)?;
    println!("Found lambda: {:.4e}", lambda);
    // Solve with the found lambda
    let step = try_solve_linear_system(&jtj, &jt_r, &diag_squared, lambda)
        .ok_or("Failed to solve linear system with computed lambda")?;

    let scaled_norm = compute_scaled_norm(&step, diag);

    Ok(TrustRegionResult {
        step,
        lambda,
        scaled_step_norm: scaled_norm,
        on_boundary: true,
    })
}

/// Try to solve the linear system (J^T*J + λ*D²)*p = -J^T*r
fn try_solve_linear_system(
    jtj: &DMatrix<f64>,
    jt_r: &DVector<f64>,
    diag_squared: &DVector<f64>,
    lambda: f64,
) -> Option<DVector<f64>> {
    let n = jtj.nrows();

    // Form the regularized matrix: J^T*J + λ*D²
    let mut regularized_matrix = -jtj.clone();
    for i in 0..n {
        regularized_matrix[(i, i)] += lambda * diag_squared[i];
    }

    // Try Cholesky decomposition (works if matrix is positive definite)
    if let Some(chol) = Cholesky::new(regularized_matrix.clone()) {
        return Some(chol.solve(jt_r));
    }

    // Fallback to LU decomposition if Cholesky fails
    if let Some(lu) = regularized_matrix.lu().solve(jt_r) {
        return Some(lu);
    }

    None
}

/// Find lambda using root finding such that ||D*p(λ)|| = delta
fn find_lambda_root_finding(
    jtj: &DMatrix<f64>,
    jt_r: &DVector<f64>,
    diag_squared: &DVector<f64>,
    delta: f64,
    lambda_init: f64,
) -> Result<f64, &'static str> {
    const MAX_ITER: usize = 50;
    const TOL: f64 = 1e-12;

    let mut lambda = lambda_init.max(0.0);

    // If lambda_init is 0, find a good starting point
    if lambda == 0.0 {
        lambda = estimate_initial_lambda(jtj, jt_r, diag_squared, delta);
    }
    println!("___________FINDING LAMBDA_____________________________\n");
    println!("Initial lambda: {:.4e}", lambda);
    // Newton's method to find lambda
    for _iter in 0..MAX_ITER {
        // Solve for current lambda
        let step = try_solve_linear_system(jtj, jt_r, diag_squared, lambda)
            .ok_or("Failed to solve linear system in root finding")?;

        // Compute scaled norm
        let scaled_norm = (0..step.len())
            .map(|i| (diag_squared[i].sqrt() * step[i]).powi(2))
            .sum::<f64>()
            .sqrt();

        // Check convergence
        let residual = scaled_norm - delta;
        if residual.abs() <= TOL * delta {
            return Ok(lambda);
        }

        // Newton's method update for lambda
        // We're solving: ||D*p(λ)|| - delta = 0
        let derivative = compute_lambda_derivative(jtj, &step, diag_squared, lambda)?;

        if derivative.abs() < f64::EPSILON {
            break;
        }
        println!(
            "{}: lambda = {:.4e}, residual = {:.4e}, derivative = {:.4e}",
            _iter, lambda, residual, derivative
        );
        let lambda_new = lambda - residual / derivative;
        lambda = lambda_new.max(0.0);
        println!("________________END OF ITERATION________________________\n");
    }

    Ok(lambda)
}

/// Estimate initial lambda using heuristics
fn estimate_initial_lambda(
    jtj: &DMatrix<f64>,
    jt_r: &DVector<f64>,
    diag_squared: &DVector<f64>,
    delta: f64,
) -> f64 {
    // Use the diagonal of J^T*J to estimate lambda
    let diag_jtj: Vec<f64> = (0..jtj.nrows()).map(|i| jtj[(i, i)]).collect();
    let max_diag = diag_jtj.iter().fold(0.0, |a: f64, &b| a.max(b));

    // Estimate based on gradient norm and trust region radius
    let grad_norm = jt_r.norm();
    let diag_norm = diag_squared
        .iter()
        .map(|&x| x.sqrt())
        .fold(0.0, |a: f64, b| a.max(b));

    if diag_norm > 0.0 && delta > 0.0 {
        (grad_norm / (delta * diag_norm)).max(max_diag * 1e-6)
    } else {
        max_diag * 1e-3
    }
}

/// Compute derivative of ||D*p(λ)|| with respect to lambda
fn compute_lambda_derivative(
    jtj: &DMatrix<f64>,
    step: &DVector<f64>,
    diag_squared: &DVector<f64>,
    lambda: f64,
) -> Result<f64, &'static str> {
    let n = jtj.nrows();

    // Form (J^T*J + λ*D²)
    let mut regularized_matrix = jtj.clone();
    for i in 0..n {
        regularized_matrix[(i, i)] += lambda * diag_squared[i];
    }

    // Solve (J^T*J + λ*D²) * v = D² * p
    let rhs: DVector<f64> = DVector::from_iterator(n, (0..n).map(|i| diag_squared[i] * step[i]));

    let v = try_solve_linear_system(
        &(jtj - &regularized_matrix + &regularized_matrix),
        &rhs,
        &DVector::zeros(n),
        0.0,
    )
    .ok_or("Failed to solve for derivative computation")?;

    // Compute derivative: -p^T * D² * v / ||D*p||
    let scaled_norm = compute_scaled_norm(step, &diag_squared.map(|x| x.sqrt()));
    if scaled_norm < f64::EPSILON {
        return Err("Scaled norm too small for derivative computation");
    }

    let numerator: f64 = (0..n).map(|i| step[i] * diag_squared[i] * v[i]).sum();
    Ok(-numerator / scaled_norm)
}

/// Compute ||D*p|| where D is given as a vector
fn compute_scaled_norm(step: &DVector<f64>, diag: &DVector<f64>) -> f64 {
    step.iter()
        .zip(diag.iter())
        .map(|(&p, &d)| (d * p).powi(2))
        .sum::<f64>()
        .sqrt()
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                           TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_simple_trust_region() {
        // Simple 2x2 problem
        let jacobian = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];
        let delta = 0.5;

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0)
                .unwrap();

        // Should be on boundary since unconstrained solution has norm sqrt(2) > 0.5
        assert!(result.on_boundary);
        assert_relative_eq!(result.scaled_step_norm, delta, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_newton_step() {
        // Problem where Gauss-Newton step is within trust region
        let jacobian = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![0.1, 0.1];
        let diag = dvector![1.0, 1.0];
        let delta = 1.0;

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0)
                .unwrap();

        // Should not be on boundary since unconstrained solution has small norm
        assert!(!result.on_boundary);
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
    }
    #[test]
    fn test_case1() {
        let a = &[
            33., -40., 44., -43., -37., -1., -40., 48., 43., -11., -40., 43.,
        ];
        let j = DMatrix::from_row_slice(4, 3, a);
        let residual = DVector::from_column_slice(&[7., -1., 0., -1.]);

        let diag = DVector::from_column_slice(&[18.2, 18.2, 3.2]);
        let param = solve_trust_region_subproblem(&j, &j.transpose(), &residual, &diag, 0.5, 0.2);

        assert_relative_eq!(param.clone().unwrap().lambda, 34.628643558156341f64);
        let p_r =
            DVector::from_column_slice(&[0.017591648698939, -0.020395135814051, 0.059285196018896]);
        assert_relative_eq!(param.clone().unwrap().step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case2() {
        let a = &[
            -7., 28., -40., 29., 7., -49., -39., 43., -25., -47., -11., 34.,
        ];
        let j = DMatrix::from_row_slice(4, 3, a);
        let residual = DVector::from_column_slice(&[-7., -8., -8., -10.]);

        let diag = DVector::from_column_slice(&[0.2, 13.2, 1.2]);
        let param = solve_trust_region_subproblem(&j, &j.transpose(), &residual, &diag, 0.5, 0.2);

        let p_r = DVector::from_column_slice(&[
            -0.048474221517806,
            -0.007207732068190,
            0.083138659283539,
        ]);
        assert_relative_eq!(param.unwrap().step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case3() {
        let a = &[
            8., -42., -34., -31., -30., -15., -36., -1., 27., 22., 44., 6.,
        ];
        let j = DMatrix::from_row_slice(4, 3, a);
        let residual = DVector::from_column_slice(&[1., -5., 2., 7.]);

        let diag = DVector::from_column_slice(&[4.2, 8.2, 11.2]);
        let param = solve_trust_region_subproblem(&j, &j.transpose(), &residual, &diag, 0.5, 0.2);

        assert_relative_eq!(
            param.clone().unwrap().lambda,
            0.017646940861467262f64,
            epsilon = 1e-14
        );
        let p_r =
            DVector::from_column_slice(&[-0.008462374169585, 0.033658082419054, 0.037230479167632]);
        assert_relative_eq!(param.unwrap().step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case4() {
        let a = &[
            14., -12., 20., -11., 19., 38., -4., -11., -14., 12., -20., 11.,
        ];
        let j = DMatrix::from_row_slice(4, 3, a);
        let residual = DVector::from_column_slice(&[-5., 3., -2., 7.]);

        let diag = DVector::from_column_slice(&[6.2, 1.2, 0.2]);
        let param = solve_trust_region_subproblem(&j, &j.transpose(), &residual, &diag, 0.5, 0.2);

        assert_relative_eq!(param.clone().unwrap().lambda, 0.);
        let p_r = DVector::from_column_slice(&[
            -0.000277548738904,
            -0.046232379576219,
            0.266724338086713,
        ]);
        assert_relative_eq!(param.unwrap().step, p_r, epsilon = 1e-14);
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, dmatrix, dvector};
    use std::f64;

    // Helper function to create test matrices
    fn create_simple_jacobian() -> DMatrix<f64> {
        dmatrix![
            1.0, 0.0;
            0.0, 1.0;
            1.0, 1.0;
        ]
    }

    fn create_ill_conditioned_jacobian() -> DMatrix<f64> {
        dmatrix![
            1.0, 1.0;
            1.0, 1.0 + 1e-12;
        ]
    }

    // Basic functionality tests

    #[test]
    fn test_gauss_newton_step_within_trust_region() {
        // Simple problem where Gauss-Newton step is within trust region
        let jacobian = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![0.1, 0.2];
        let diag = dvector![1.0, 1.0];
        let delta = 1.0; // Large trust region
        let lambda_prev = 0.0;

        let result = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &residuals,
            &diag,
            delta,
            lambda_prev,
        )
        .unwrap();

        // Should use Gauss-Newton step (lambda = 0)
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        assert!(!result.on_boundary);

        // Step should be -J^T * r for this simple case
        assert_relative_eq!(result.step[0], -0.1, epsilon = 1e-10); //????????????
        assert_relative_eq!(result.step[1], -0.2, epsilon = 1e-10);

        // Scaled step norm should be less than delta
        assert!(result.scaled_step_norm < delta);
    }

    #[test]
    fn test_constrained_step_on_boundary() {
        // Problem where Gauss-Newton step exceeds trust region
        let jacobian = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0]; // Large residuals
        let diag = dvector![1.0, 1.0];
        let delta = 0.5; // Small trust region
        let lambda_prev = 0.0;

        let result = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &residuals,
            &diag,
            delta,
            lambda_prev,
        )
        .unwrap();
        println!("Result: {:?}", result);
        // Should be on boundary with lambda > 0
        assert!(result.lambda > 0.0);

        assert!(result.on_boundary);

        // Scaled step norm should equal delta (within tolerance)
        assert_relative_eq!(result.scaled_step_norm, delta, epsilon = 1e-8);
    }

    #[test]
    fn test_overdetermined_system() {
        let jacobian = create_simple_jacobian(); // 3x2 matrix
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0, 0.5];
        let diag = dvector![1.0, 1.0];
        let delta = 0.8;
        let lambda_prev = 0.0;

        let result = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &residuals,
            &diag,
            delta,
            lambda_prev,
        )
        .unwrap();

        assert!(result.step.len() == 2);
        assert!(result.scaled_step_norm <= delta + 1e-10);

        // Verify that step reduces the objective ||J*p - r||²
        let jp_minus_r = &jacobian * &result.step - &residuals;
        let new_objective = jp_minus_r.norm_squared();
        let old_objective = residuals.norm_squared();
        assert!(new_objective < old_objective);
    }

    #[test]
    fn test_different_diagonal_scaling() {
        let jacobian = dmatrix![
            2.0, 0.0;
            0.0, 0.1;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let delta = 1.0;
        let lambda_prev = 0.0;

        // Test with uniform scaling
        let diag_uniform = dvector![1.0, 1.0];
        let result_uniform = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &residuals,
            &diag_uniform,
            delta,
            lambda_prev,
        )
        .unwrap();

        // Test with non-uniform scaling
        let diag_scaled = dvector![0.1, 10.0];
        let result_scaled = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &residuals,
            &diag_scaled,
            delta,
            lambda_prev,
        )
        .unwrap();

        // Results should be different due to different scaling
        assert!((result_uniform.step - result_scaled.step).norm() > 1e-6);

        // Both should respect their respective trust regions
        assert!(result_uniform.scaled_step_norm <= delta + 1e-10);
        assert!(result_scaled.scaled_step_norm <= delta + 1e-10);
    }

    #[test]
    fn test_warm_start_with_previous_lambda() {
        let jacobian = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
            0.5, 0.5;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0, 0.5];
        let diag = dvector![1.0, 1.0];
        let delta = 0.5;

        // Solve without warm start
        let result_cold =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0)
                .unwrap();

        // Solve with warm start using previous lambda
        let result_warm = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &residuals,
            &diag,
            delta,
            result_cold.lambda,
        )
        .unwrap();

        // Results should be very similar
        assert_relative_eq!(result_cold.lambda, result_warm.lambda, epsilon = 1e-8);
        assert_relative_eq!(
            (result_cold.step - result_warm.step).norm(),
            0.0,
            epsilon = 1e-8
        );
    }

    // Edge cases and error conditions

    #[test]
    fn test_dimension_mismatch_jacobian_transpose() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let wrong_jacobian_t = dmatrix![1.0; 0.0]; // Wrong dimensions
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];

        let result = solve_trust_region_subproblem(
            &jacobian,
            &wrong_jacobian_t,
            &residuals,
            &diag,
            1.0,
            0.0,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Jacobian transpose dimensions mismatch"
        );
    }

    #[test]
    fn test_dimension_mismatch_residuals() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let jacobian_t = jacobian.transpose();
        let wrong_residuals = dvector![1.0]; // Wrong size
        let diag = dvector![1.0, 1.0];

        let result = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &wrong_residuals,
            &diag,
            1.0,
            0.0,
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Residuals dimension mismatch");
    }

    #[test]
    fn test_dimension_mismatch_diagonal() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let wrong_diag = dvector![1.0]; // Wrong size

        let result = solve_trust_region_subproblem(
            &jacobian,
            &jacobian_t,
            &residuals,
            &wrong_diag,
            1.0,
            0.0,
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Diagonal vector dimension mismatch");
    }

    #[test]
    fn test_negative_trust_region_radius() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, -1.0, 0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Trust region radius must be positive");
    }

    #[test]
    fn test_zero_trust_region_radius() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, 0.0, 0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Trust region radius must be positive");
    }

    #[test]
    fn test_singular_jacobian() {
        // Rank-deficient Jacobian
        let jacobian = dmatrix![
            1.0, 1.0;
            1.0, 1.0;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];
        let delta = 1.0;

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0);

        // Should still work due to regularization
        assert!(result.is_ok());
        let solution = result.unwrap();
        println!("Solution: {:?}", solution);
        assert!(solution.lambda < 0.0); // Should add regularization
        assert!(solution.scaled_step_norm <= delta + 1e-10);
    }

    #[test]
    fn test_ill_conditioned_jacobian() {
        let jacobian = create_ill_conditioned_jacobian();
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];
        let delta = 0.5;

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0);

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.scaled_step_norm <= delta + 1e-8);
    }

    #[test]
    fn test_zero_residuals() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![0.0, 0.0];
        let diag = dvector![1.0, 1.0];
        let delta = 1.0;

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0)
                .unwrap();

        // With zero residuals, step should be zero
        assert_relative_eq!(result.step.norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        assert!(!result.on_boundary);
    }

    #[test]
    fn test_very_small_trust_region() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];
        let delta = 1e-10; // Very small trust region

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0)
                .unwrap();

        assert!(result.on_boundary);
        assert!(result.lambda > 1e6); // Should need large lambda
        assert_relative_eq!(result.scaled_step_norm, delta, epsilon = 1e-12);
    }

    #[test]
    fn test_very_large_trust_region() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![0.1, 0.1];
        let diag = dvector![1.0, 1.0];
        let delta = 1e6; // Very large trust region

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0)
                .unwrap();

        // Should use Gauss-Newton step
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        assert!(!result.on_boundary);
        assert!(result.scaled_step_norm < delta);
    }

    // Numerical accuracy tests

    #[test]
    fn test_lambda_convergence_accuracy() {
        let jacobian = dmatrix![
            2.0, 1.0;
            1.0, 2.0;
            1.0, 1.0;
        ];
        let jacobian_t = jacobian.transpose();
        let residuals = dvector![1.0, 1.0, 0.5];
        let diag = dvector![1.0, 1.0,];
        let delta = 1.0;

        let result =
            solve_trust_region_subproblem(&jacobian, &jacobian_t, &residuals, &diag, delta, 0.0)
                .unwrap();
        println!("Result: {:?}", result);
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
    }
}
