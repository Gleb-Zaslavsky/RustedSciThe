//! Trust region subproblem solver for Levenberg-Marquardt algorithm.
//!
//! This module implements a solver for the trust region subproblem:
//! min ||J*p - r||² subject to ||D*p|| ≤ delta
//!
//! The implementation follows the approach used in MINPACK's LMPAR routine.
/*
ALGORITHM: Trust Region Subproblem Solver

INPUT:
  - J: Jacobian matrix (m×n)
  - J^T: Transposed Jacobian (n×m)
  - r: residuals vector (m×1)
  - D: diagonal scaling vector (n×1)
  - δ: trust region radius
  - λ_prev: previous lambda (warm start)

OUTPUT:
  - p: step vector
  - λ: Levenberg-Marquardt parameter
  - ||D*p||: scaled step norm
  - on_boundary: boolean flag

BEGIN
  1. VALIDATE_INPUTS()
     - Check dimension compatibility
     - Ensure δ > 0

  2. COMPUTE_PRELIMINARIES()
     - g = J^T * r  (gradient)
     - H = J^T * J  (Hessian approximation)
     - D² = diag(D[i]²)  (squared diagonal elements)

  3. TRY_GAUSS_NEWTON_STEP()
     - Solve: H * p = g  (λ = 0)
     - IF solution exists AND ||D*p|| ≤ δ THEN
         RETURN (p, λ=0, ||D*p||, on_boundary=false)

  4. FIND_LAMBDA_BY_ROOT_FINDING()
     - λ = ESTIMATE_INITIAL_LAMBDA(H, g, D², δ, λ_prev)
     - FOR iter = 1 to MAX_ITER DO
         a. Solve: (H + λ*D²) * p = g
         b. norm = ||D*p||
         c. residual = norm - δ
         d. IF |residual| ≤ tolerance THEN BREAK
         e. derivative = COMPUTE_LAMBDA_DERIVATIVE(H, p, D², λ)
         f. λ = λ - residual/derivative  (Newton update)
         g. λ = max(0, λ)  (ensure non-negative)

  5. FINAL_SOLVE()
     - Solve: (H + λ*D²) * p = g
     - RETURN (p, λ, ||D*p||, on_boundary=true)
END

SUBROUTINE: SOLVE_LINEAR_SYSTEM(A, b, D², λ)
BEGIN
  1. Form: A_reg = A + λ*diag(D²)
  2. TRY Cholesky decomposition of A_reg
     - IF successful THEN solve and return
  3. FALLBACK to LU decomposition
     - IF successful THEN solve and return
  4. RETURN failure
END

SUBROUTINE: ESTIMATE_INITIAL_LAMBDA(H, g, D², δ)
BEGIN
  1. max_diag = max(diagonal elements of H)
  2. grad_norm = ||g||
  3. diag_norm = max(sqrt(D²[i]))
  4. IF diag_norm > 0 AND δ > 0 THEN
       λ = max(grad_norm/(δ*diag_norm), max_diag*1e-6)
     ELSE
       λ = max_diag*1e-3
  5. RETURN λ
END

SUBROUTINE: COMPUTE_LAMBDA_DERIVATIVE(H, p, D², λ)
BEGIN
  1. Form: A = H + λ*D²
  2. rhs = D² ⊙ p  (element-wise product)
  3. Solve: A * v = rhs
  4. numerator = p^T * (D² ⊙ v)
  5. denominator = ||D*p||
  6. RETURN -numerator/denominator
END

*/
use nalgebra::{Cholesky, DMatrix, DVector};
use num_traits::Float;

#[cfg(test)]
use approx::assert_relative_eq;

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

/// Solve the trust region subproblem using the approach from MINPACK.
///
/// Given a Jacobian matrix J and residual vector r, this routine solves:
/// min ||J*p - r||² subject to ||D*p|| ≤ delta
///
/// This is equivalent to solving: (J^T*J + λ*D²)*p = J^T*r
/// where λ ≥ 0 is chosen such that ||D*p|| ≤ delta
///
/// # Arguments
/// * `jacobian` - The Jacobian matrix J (m×n)
/// * `residuals` - The residual vector r (m×1)
/// * `diag` - Diagonal scaling vector D (n×1)
/// * `delta` - Trust region radius
/// * `lambda_prev` - Previous lambda value (for warm start)
///
/// # Returns
/// TrustRegionResult containing the step and related information
pub fn solve_trust_region_subproblem(
    jacobian: &DMatrix<f64>,
    residuals: &DVector<f64>,
    diag: &DVector<f64>,
    delta: f64,
    lambda_prev: f64,
) -> Result<TrustRegionResult, &'static str> {
    let n = jacobian.ncols();
    let m = jacobian.nrows();

    // Validate inputs
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
    let jt_r = jacobian.transpose() * residuals;

    // Compute J^T * J
    let jtj = jacobian.transpose() * jacobian;

    // Try Gauss-Newton step first (lambda = 0)
    if let Some(step) = try_solve_linear_system(&jtj, &jt_r, diag, 0.0) {
        let scaled_norm = compute_scaled_norm(&step, diag);
        if scaled_norm <= delta * 0.1 {
            return Ok(TrustRegionResult {
                step,
                lambda: 0.0,
                scaled_step_norm: scaled_norm,
                on_boundary: false,
            });
        }
    }

    // Need to find lambda > 0 such that ||D*p|| = delta
    let lambda = find_lambda_root_finding(&jtj, &jt_r, diag, delta, lambda_prev)?;

    // Solve with the found lambda
    let step = try_solve_linear_system(&jtj, &jt_r, diag, lambda)
        .ok_or("Failed to solve linear system with computed lambda")?;

    let scaled_norm = compute_scaled_norm(&step, diag);

    Ok(TrustRegionResult {
        step,
        lambda,
        scaled_step_norm: scaled_norm,
        on_boundary: true,
    })
}

/// Try to solve the linear system (J^T*J + λ*D²)*p = J^T*r
fn try_solve_linear_system(
    jtj: &DMatrix<f64>,
    jt_r: &DVector<f64>,
    diag: &DVector<f64>,
    lambda: f64,
) -> Option<DVector<f64>> {
    let n = jtj.nrows();

    // Form the regularized matrix: J^T*J + λ*D²
    let mut regularized_matrix = jtj.clone();
    for i in 0..n {
        regularized_matrix[(i, i)] += lambda * diag[i] * diag[i];
    }

    // Try Cholesky decomposition (works if matrix is positive definite)
    if let Some(chol) = Cholesky::new(regularized_matrix.clone()) {
        let solution = chol.solve(jt_r);
        {
            return Some(solution);
        }
    }

    // Fallback to LU decomposition if Cholesky fails
    let lu = regularized_matrix.lu();
    lu.solve(jt_r)
}

/// Find lambda using Newton's method such that ||D*p(λ)|| = delta
fn find_lambda_root_finding(
    jtj: &DMatrix<f64>,
    jt_r: &DVector<f64>,
    diag: &DVector<f64>,
    delta: f64,
    lambda_init: f64,
) -> Result<f64, &'static str> {
    const MAX_ITER: usize = 50;
    const TOL: f64 = 1e-12;

    let mut lambda = lambda_init.max(0.0);

    // If lambda_init is 0, find a good starting point
    if lambda == 0.0 {
        lambda = estimate_initial_lambda(jtj, jt_r, diag, delta);
    }

    for _iter in 0..MAX_ITER {
        // Solve for current lambda
        let step = try_solve_linear_system(jtj, jt_r, diag, lambda)
            .ok_or("Failed to solve linear system in root finding")?;

        // Compute scaled norm
        let scaled_norm = compute_scaled_norm(&step, diag);

        // Check convergence
        let residual = scaled_norm - delta;
        if residual.abs() <= TOL * delta {
            return Ok(lambda);
        }

        // Newton's method update for lambda
        let derivative = compute_lambda_derivative(jtj, &step, diag, lambda)?;

        if derivative.abs() < f64::EPSILON {
            break;
        }

        let lambda_new = lambda - residual / derivative;
        lambda = lambda_new.max(0.0);
    }

    Ok(lambda)
}

/// Estimate initial lambda using heuristics similar to MINPACK
fn estimate_initial_lambda(
    jtj: &DMatrix<f64>,
    jt_r: &DVector<f64>,
    diag: &DVector<f64>,
    delta: f64,
) -> f64 {
    // Use the diagonal of J^T*J to estimate lambda
    let max_diag = (0..jtj.nrows())
        .map(|i| jtj[(i, i)])
        .fold(0.0, |a, b| a.max(b));

    // Estimate based on gradient norm and trust region radius
    let grad_norm = jt_r.norm();
    let diag_max = diag.iter().fold(0.0, |a, &b| a.max(b));

    if diag_max > 0.0 && delta > 0.0 {
        (grad_norm / (delta * diag_max)).max(max_diag * 1e-6)
    } else {
        max_diag * 1e-3
    }
}

/// Compute derivative of ||D*p(λ)|| with respect to lambda
fn compute_lambda_derivative(
    jtj: &DMatrix<f64>,
    step: &DVector<f64>,
    diag: &DVector<f64>,
    lambda: f64,
) -> Result<f64, &'static str> {
    let n = jtj.nrows();

    // Form (J^T*J + λ*D²)
    let mut regularized_matrix = jtj.clone();
    for i in 0..n {
        regularized_matrix[(i, i)] += lambda * diag[i] * diag[i];
    }

    // Solve (J^T*J + λ*D²) * v = D² * p
    let mut rhs = DVector::zeros(n);
    for i in 0..n {
        rhs[i] = diag[i] * diag[i] * step[i];
    }

    let lu = regularized_matrix.lu();
    let v = lu
        .solve(&rhs)
        .ok_or("Failed to solve for derivative computation")?;

    // Compute derivative: -p^T * D² * v / ||D*p||
    let scaled_norm = compute_scaled_norm(step, diag);
    if scaled_norm < f64::EPSILON {
        return Err("Scaled norm too small for derivative computation");
    }

    let numerator: f64 = (0..n).map(|i| step[i] * diag[i] * diag[i] * v[i]).sum();

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

/// Compute the 2-norm of a vector (similar to enorm in the original codebase)
pub fn enorm(v: &DVector<f64>) -> f64 {
    v.norm()
}

/// Machine epsilon for f64
pub fn epsmch() -> f64 {
    f64::EPSILON
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_gauss_newton_step_within_trust_region() {
        // Simple problem where Gauss-Newton step is within trust region
        let jacobian = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
        ];
        let residuals = dvector![0.1, 0.2];
        let diag = dvector![1.0, 1.0];
        let delta = 1.0; // Large trust region
        let lambda_prev = 0.0;

        let result =
            solve_trust_region_subproblem(&jacobian, &residuals, &diag, delta, lambda_prev)
                .unwrap();

        // Should use Gauss-Newton step (lambda = 0)
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        assert!(!result.on_boundary);

        // Step should be -J^T * r for this simple case
        assert_relative_eq!(result.step[0], -0.1, epsilon = 1e-10);
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
        let residuals = dvector![1.0, 1.0]; // Large residuals
        let diag = dvector![1.0, 1.0];
        let delta = 0.5; // Small trust region
        let lambda_prev = 0.0;

        let result =
            solve_trust_region_subproblem(&jacobian, &residuals, &diag, delta, lambda_prev)
                .unwrap();

        // Should be on boundary with lambda > 0
        assert!(result.lambda > 0.0);
        assert!(result.on_boundary);

        // Scaled step norm should equal delta (within tolerance)
        assert_relative_eq!(result.scaled_step_norm, delta, epsilon = 1e-8);
    }

    #[test]
    fn test_overdetermined_system() {
        let jacobian = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
            1.0, 1.0;
        ];
        let residuals = dvector![1.0, 1.0, 0.5];
        let diag = dvector![1.0, 1.0];
        let delta = 0.8;
        let lambda_prev = 0.0;

        let result =
            solve_trust_region_subproblem(&jacobian, &residuals, &diag, delta, lambda_prev)
                .unwrap();

        assert_eq!(result.step.len(), 2);
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
        let residuals = dvector![1.0, 1.0];
        let delta = 1.0;
        let lambda_prev = 0.0;

        // Test with uniform scaling
        let diag_uniform = dvector![1.0, 1.0];
        let result_uniform =
            solve_trust_region_subproblem(&jacobian, &residuals, &diag_uniform, delta, lambda_prev)
                .unwrap();

        // Test with non-uniform scaling
        let diag_scaled = dvector![0.1, 10.0];
        let result_scaled =
            solve_trust_region_subproblem(&jacobian, &residuals, &diag_scaled, delta, lambda_prev)
                .unwrap();

        // Results should be different due to different scaling
        assert!((result_uniform.step - result_scaled.step).norm() > 1e-6);

        // Both should respect their respective trust regions
        assert!(result_uniform.scaled_step_norm <= delta + 1e-10);
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_case1() {
        // Test case from original trust_region.rs
        let j = dmatrix![
            33., -37., -40.;
            -40., -1., 48.;
            44., -40., 43.;
            -43., -11., 43.;
        ];
        let residual = dvector![7., -1., 0., -1.];
        let diag = dvector![18.2, 18.2, 3.2];
        let delta = 0.5;
        let lambda_prev = 0.2;

        let result =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, lambda_prev).unwrap();

        // Expected values from original test
        assert_relative_eq!(result.lambda, 34.628643558156341, epsilon = 1e-12);
        let p_expected = dvector![0.017591648698939, -0.020395135814051, 0.059285196018896];
        println!("step: {:?}, expected: {:?}", result.step, p_expected);
        assert_relative_eq!(result.step, p_expected, epsilon = 1e-14);
        assert!(result.on_boundary);
        assert_relative_eq!(result.scaled_step_norm, delta, epsilon = 1e-10);
    }

    #[test]
    fn test_case2() {
        // Test case from original trust_region.rs
        let j = dmatrix![
            -7., 7., -25.;
            28., -49., -39.;
            -40., -39., 43.;
            29., 43., -47.;
        ];
        let residual = dvector![-7., -8., -8., -10.];
        let diag = dvector![10.2, 13.2, 1.2];
        let delta = 0.5;
        let lambda_prev = 0.2;

        let result =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, lambda_prev).unwrap();

        // This case should use Gauss-Newton step (lambda = 0)
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        let p_expected = dvector![-0.048474221517806, -0.007207732068190, 0.083138659283539];
        println!("step: {:?}, expected: {:?}", result.step, p_expected);
        assert_relative_eq!(result.step, p_expected, epsilon = 1e-14);
        assert!(!result.on_boundary);
    }

    #[test]
    fn test_case3() {
        let j = dmatrix![
            8., -30., -36.;
            -42., -15., -1.;
            -34., -36., 27.;
            -31., 1., 22.;
        ];
        let residual = dvector![1., -5., 2., 7.];
        let diag = dvector![4.2, 8.2, 11.2];
        let delta = 0.5;
        let lambda_prev = 0.2;

        let result =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, lambda_prev).unwrap();

        assert_relative_eq!(result.lambda, 0.017646940861467262, epsilon = 1e-14);
        let p_expected = dvector![-0.008462374169585, 0.033658082419054, 0.037230479167632];
        println!("step: {:?}, expected: {:?}", result.step, p_expected);
        assert_relative_eq!(result.step, p_expected, epsilon = 1e-14);
        assert!(result.on_boundary);
    }

    #[test]
    fn test_case4() {
        let j = dmatrix![
            14., 19., -4.;
            -12., 38., -11.;
            20., -4., -14.;
            -11., -11., 12.;
        ];
        let residual = dvector![-5., 3., -2., 7.];
        let diag = dvector![6.2, 1.2, 0.2];
        let delta = 0.5;
        let lambda_prev = 0.2;

        let result =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, lambda_prev).unwrap();

        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        let p_expected = dvector![-0.000277548738904, -0.046232379576219, 0.266724338086713];
        println!("step: {:?}, expected: {:?}", result.step, p_expected);
        assert_relative_eq!(result.step, p_expected, epsilon = 1e-14);
        assert!(!result.on_boundary);
    }

    #[test]
    fn test_simple_identity_jacobian() {
        // Simple test with identity Jacobian
        let j = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
        ];
        let residual = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];
        let delta = 0.5;
        let lambda_prev = 0.0;

        let result =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, lambda_prev).unwrap();

        // Should be on boundary since unconstrained solution has norm sqrt(2) > 0.5
        assert!(result.on_boundary);
        assert!(result.lambda > 0.0);
        assert_relative_eq!(result.scaled_step_norm, delta, epsilon = 1e-10);
    }

    #[test]
    fn test_small_residuals_gauss_newton() {
        // Test where Gauss-Newton step is within trust region
        let j = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
        ];
        let residual = dvector![0.1, 0.1];
        let diag = dvector![1.0, 1.0];
        let delta = 1.0; // Large trust region
        let lambda_prev = 0.0;

        let result =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, lambda_prev).unwrap();

        // Should use Gauss-Newton step
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        assert!(!result.on_boundary);
        assert_relative_eq!(result.step[0], -0.1, epsilon = 1e-10);
        assert_relative_eq!(result.step[1], -0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_overdetermined_system() {
        // 3x2 overdetermined system
        let j = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
            1.0, 1.0;
        ];
        let residual = dvector![1.0, 1.0, 0.5];
        let diag = dvector![1.0, 1.0];
        let delta = 0.8;
        let lambda_prev = 0.0;

        let result =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, lambda_prev).unwrap();

        assert_eq!(result.step.len(), 2);
        assert!(result.scaled_step_norm <= delta + 1e-10);

        // Verify that step reduces the objective ||J*p - r||²
        let jp_minus_r = &j * &result.step - &residual;
        let new_objective = jp_minus_r.norm_squared();
        let old_objective = residual.norm_squared();
        assert!(new_objective < old_objective);
    }

    #[test]
    fn test_different_diagonal_scaling() {
        let j = dmatrix![
            2.0, 0.0;
            0.0, 0.1;
        ];
        let residual = dvector![1.0, 1.0];
        let delta = 1.0;
        let lambda_prev = 0.0;

        // Test with uniform scaling
        let diag_uniform = dvector![1.0, 1.0];
        let result_uniform =
            solve_trust_region_subproblem(&j, &residual, &diag_uniform, delta, lambda_prev)
                .unwrap();

        // Test with non-uniform scaling
        let diag_scaled = dvector![0.1, 10.0];
        let result_scaled =
            solve_trust_region_subproblem(&j, &residual, &diag_scaled, delta, lambda_prev).unwrap();

        // Results should be different due to different scaling
        assert!((result_uniform.step - result_scaled.step).norm() > 1e-6);

        // Both should respect their respective trust regions
        assert!(result_uniform.scaled_step_norm <= delta + 1e-10);
        assert!(result_scaled.scaled_step_norm <= delta + 1e-10);
    }

    #[test]
    fn test_warm_start_with_previous_lambda() {
        let j = dmatrix![
            1.0, 0.0;
            0.0, 1.0;
            0.5, 0.5;
        ];
        let residual = dvector![1.0, 1.0, 0.5];
        let diag = dvector![1.0, 1.0];
        let delta = 0.5;

        // Solve without warm start
        let result_cold = solve_trust_region_subproblem(&j, &residual, &diag, delta, 0.0).unwrap();

        // Solve with warm start using previous lambda
        let result_warm =
            solve_trust_region_subproblem(&j, &residual, &diag, delta, result_cold.lambda).unwrap();

        // Results should be very similar
        assert_relative_eq!(result_cold.lambda, result_warm.lambda, epsilon = 1e-8);
        assert_relative_eq!(
            (result_cold.step - result_warm.step).norm(),
            0.0,
            epsilon = 1e-8
        );
    }

    #[test]
    fn test_zero_residuals() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residual = dvector![0.0, 0.0];
        let diag = dvector![1.0, 1.0];
        let delta = 1.0;

        let result = solve_trust_region_subproblem(&j, &residual, &diag, delta, 0.0).unwrap();

        // With zero residuals, step should be zero
        assert_relative_eq!(result.step.norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        assert!(!result.on_boundary);
    }

    #[test]
    fn test_very_small_trust_region() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residual = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];
        let delta = 1e-10; // Very small trust region

        let result = solve_trust_region_subproblem(&j, &residual, &diag, delta, 0.0).unwrap();

        assert!(result.on_boundary);
        assert!(result.lambda > 1e6); // Should need large lambda
        assert_relative_eq!(result.scaled_step_norm, delta, epsilon = 1e-12);
    }

    #[test]
    fn test_very_large_trust_region() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residual = dvector![0.1, 0.1];
        let diag = dvector![1.0, 1.0];
        let delta = 1e6; // Very large trust region

        let result = solve_trust_region_subproblem(&j, &residual, &diag, delta, 0.0).unwrap();

        // Should use Gauss-Newton step
        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-10);
        assert!(!result.on_boundary);
        assert!(result.scaled_step_norm < delta);
    }

    // Error condition tests
    #[test]
    fn test_dimension_mismatch_residuals() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let wrong_residual = dvector![1.0]; // Wrong size
        let diag = dvector![1.0, 1.0];

        let result = solve_trust_region_subproblem(&j, &wrong_residual, &diag, 1.0, 0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Residuals dimension mismatch");
    }

    #[test]
    fn test_dimension_mismatch_diagonal() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residual = dvector![1.0, 1.0];
        let wrong_diag = dvector![1.0]; // Wrong size

        let result = solve_trust_region_subproblem(&j, &residual, &wrong_diag, 1.0, 0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Diagonal vector dimension mismatch");
    }

    #[test]
    fn test_negative_trust_region_radius() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residual = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];

        let result = solve_trust_region_subproblem(&j, &residual, &diag, -1.0, 0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Trust region radius must be positive");
    }

    #[test]
    fn test_zero_trust_region_radius() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residual = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];

        let result = solve_trust_region_subproblem(&j, &residual, &diag, 0.0, 0.0);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Trust region radius must be positive");
    }
}
 */
