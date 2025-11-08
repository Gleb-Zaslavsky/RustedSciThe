use super::NR::solve_linear_system;
use nalgebra::{DMatrix, DVector};
/// Nielsen p. 125
pub fn Powell_dogleg_method(
    Jy: DMatrix<f64>,
    Fy: DVector<f64>,
    scaling: DVector<f64>,
    delta: f64,
    solver: String,
) -> DVector<f64> {
    // Minimize ||J*p + F||^2 subject to ||scaling.*p|| <= delta

    // STEP 1: Compute Gauss-Newton step
    // Compute D^(-1) g where g = J^T f (gradient)
    let p_gauss: DVector<f64> =
        solve_linear_system(solver, &Jy, &(-Fy.clone())).expect("Failed to solve linear system"); // J \ (-F)
    assert_eq!(p_gauss.len(), scaling.len(), "length of vectors differ");
    //  println!("p_gauss, {:?}, scaling {:?}", p_gauss, scaling.clone()* p_gauss.clone());
    let scaled_norm = (scaling.clone().component_mul(&p_gauss.clone())).norm();

    if scaled_norm <= delta {
        println!("scaled norm of gauss step <= delta => return gauss step");
        return p_gauss; // Gauss-Newton step is within trust region
    }

    // STEP 2: Compute steepest descent (sd) step
    // scaled gradient
    let gradient: DVector<f64> = Jy.transpose() * Fy;
    let p_steepest = -gradient.clone();
    let gradient_scaled: DVector<f64> = gradient.component_div(&scaling);

    // Optimal step length for steepest descent

    let alpha_optimal = (gradient_scaled.norm() / (Jy * gradient_scaled).norm()).powf(2.0);
    let p_dl: DVector<f64>;
    if p_gauss.norm() < delta {
        println!("scaled norm of gauss step <= delta => return gauss step");
        p_dl = p_gauss;
    } else if p_steepest.norm() * alpha_optimal > delta {
        println!(
            " norm of stepest descent step*alpha > delta => return  p_steepest*(delta/||p_steepest||)"
        );
        p_dl = p_steepest.clone() * (delta / p_steepest.norm());
    } else {
        println!(
            "||p_steepest|| {}, alpha {}, delta {}",
            p_steepest.norm(),
            alpha_optimal,
            delta
        );
        let a = alpha_optimal * p_steepest.clone();
        let b = p_gauss.clone();
        let c = a.clone().transpose().dot(&(b.clone() - a.clone()));
        let b_min_a = (b.clone() - a.clone()).norm();
        let L = (c.powf(2.0) + b_min_a.powf(2.0) * (delta.powf(2.0) - a.norm().powf(2.0))).sqrt();
        let beta: f64;
        if c <= 0.0 {
            beta = (-c + L) / b_min_a.powf(2.0);
        } else {
            beta = (delta.powf(2.0) - a.norm().powf(2.0)) / (c + L);
        }
        println!("beta = {}", beta);
        p_dl = alpha_optimal * p_steepest + beta * p_gauss;
    }
    return p_dl;
}
/*
pub fn gsl_dogleg_method(Jy: DMatrix<f64>, Fy: DVector<f64>, scaling: DVector<f64>, delta: f64, solver: String) -> DVector<f64> {
    // Minimize ||J*p + F||^2 subject to ||scaling.*p|| <= delta


    //  Compute steepest descent (sd) step
    // Compute D^(-1) g where g = J^T f (gradient)
    let  gradient: DVector<f64> =  Jy.transpose() * Fy;
    let gradient_scaled: DVector<f64> =  gradient.component_div(&scaling);
    let scaled_norm = gradient_scaled.norm_squared();
    // Optimal step length for steepest descent

    // Compute D^(-2) g
    let grad_div_D2 = gradient_scaled.component_div(&scaling);
    // J D^(-2) g
    let J_grad_div_D2 = Jy * grad_div_D2;
    let norm_J_grad_div_D2 = J_grad_div_D2.norm_squared();
    // Compute steepest descent step length
    let alpha_optimal = (scaled_norm / norm_J_grad_div_D2).powf(2.0);
    // Compute steepest descent step length: alpha*J^T*f = D^(-2)*J^T*f*||D^(-1)*g||^2/ ||J D^(-2) g ||^2
    let dx_steepest_descent = - alpha_optimal * gradient.component_div(&scaling).component_div(&scaling);
    let norm_steepest_descent = (scaling* dx_steepest_descent).norm_squared();

    if norm_steepest_descent > delta {
    //  Steepest descent step exceeds trust region
    // IF norm_Dsd ≥ δ THEN:
    // Truncate steepest descent to trust region boundary
      let dx = (delta / norm_steepest_descent) *  dx_steepest_descent;
      return dx;
    }
    else {
      //  Compute Gauss-Newton step
    let dx_gauss = solve_linear_system(solver, &Jy, &(-Fy)).expect("Failed to solve linear system");  // J \ (-F)
    let scaled_norm_gauss = (scaling * dx_gauss).norm_squared();
    }
            // Compute dogleg step length

}
pub fn dogleg_method(Jy: DMatrix<f64>, Fy: DVector<f64>, scaling: DVector<f64>, delta: f64, solver: String) -> DVector<f64> {
    // Minimize ||J*p + F||^2 subject to ||scaling.*p|| <= delta
    let alpha: f64;
    let snm: f64;
    // STEP 1: Compute Gauss-Newton step
    let p_gn = -solve_linear_system(solver, &Jy, &Fy).expect("Failed to solve linear system");  // J \ (-F)
    let scaled_norm = (scaling * p_gn).norm_squared();

    if scaled_norm <= delta {
        return p_gn;  // Gauss-Newton step is within trust region
    }

    // STEP 2: Compute steepest descent step
    // scaled gradient
    let  gradient: DVector<f64> =  Jy.transpose() * Fy;
    let gradient_scaled: DVector<f64> =  gradient.component_div(&scaling);
    let gradient_scaled__norm = gradient_scaled.norm_squared();
    if gradient_scaled__norm > 0.0 {
        // Normalize and rescale.
        let gradient_rescaled = (gradient_scaled / gradient_scaled__norm).component_div(&scaling);
        let tn = (Jy*gradient_rescaled).norm_squared();
        snm = gradient_scaled__norm/tn.powf(2.0);
        if snm<delta {
            let residual_norm  = Fy.norm_squared();
            let d_scaled = delta/scaled_norm;
        } else {
            alpha = 0.0;
        }
    } else {
        alpha = delta/scaled_norm;
        snm = 0.0;
    };
    // Optimal step length for steepest descent

    let alpha_optimal = gradient_scaled.norm_squared() / (Jy * gradient_scaled).norm_squared();
    p_sd = alpha_optimal * p_sd_direction;
    scaled_norm_sd = norm(scaling .* p_sd);

    if (scaled_norm_sd >= delta) {
        // Pure steepest descent, scaled to trust region boundary
        return (delta / scaled_norm_sd) * p_sd;
    }

    // STEP 3: Dogleg path - convex combination of SD and GN
    // Find tau such that ||scaling .* (p_sd + tau*(p_gn - p_sd))|| = delta

    diff = p_gn - p_sd;
    a = norm(scaling * diff)^2;
    b = 2 * (scaling * p_sd).transpose() * (scaling * diff);
    c = norm(scaling * p_sd)^2 - delta^2;

    // Solve quadratic: a*tau^2 + b*tau + c = 0
    discriminant = b^2 - 4*a*c;
    tau = (-b + sqrt(discriminant)) / (2*a);

    return p_sd + tau * diff;
}

*/
/// Error types for the dogleg algorithm
#[derive(Debug)]
pub enum DoglegError {
    DimensionMismatch,
    SingularMatrix,
    NumericalError,
    AllocationError,
}

/// State structure for the dogleg algorithm
pub struct DoglegState {
    /// Number of observations
    n: usize,
    /// Number of parameters  
    p: usize,
    /// Gauss-Newton step, size p
    dx_gn: DVector<f64>,
    /// Steepest descent step, size p
    dx_sd: DVector<f64>,
    /// ||D dx_gn|| - scaled norm of GN step
    norm_dgn: f64,
    /// ||D dx_sd|| - scaled norm of SD step  
    norm_dsd: f64,
    /// ||D^{-1} g|| - scaled norm of gradient
    norm_dinvg: f64,
    /// ||J D^{-2} g|| - norm of scaled Jacobian-gradient product
    norm_jdinv2g: f64,
    /// Workspace vector, length p
    workp: DVector<f64>,
    /// Workspace vector, length n
    workn: DVector<f64>,
    /// Flag indicating if GN step has been computed
    gn_computed: bool,
}

impl DoglegState {
    /// Create new dogleg state
    pub fn new(n: usize, p: usize) -> Result<Self, DoglegError> {
        Ok(DoglegState {
            n,
            p,
            dx_gn: DVector::zeros(p),
            dx_sd: DVector::zeros(p),
            norm_dgn: -1.0, // negative indicates not computed
            norm_dsd: 0.0,
            norm_dinvg: 0.0,
            norm_jdinv2g: 0.0,
            workp: DVector::zeros(p),
            workn: DVector::zeros(n),
            gn_computed: false,
        })
    }

    /// Initialize dogleg method prior to iteration loop
    /// Computes the steepest descent step
    pub fn preloop(
        &mut self,
        jacobian: &DMatrix<f64>,
        _residual: &DVector<f64>,
        gradient: &DVector<f64>,
        diag: &DVector<f64>,
    ) -> Result<(), DoglegError> {
        // Validate dimensions
        if jacobian.nrows() != self.n || jacobian.ncols() != self.p {
            return Err(DoglegError::DimensionMismatch);
        }
        if gradient.len() != self.p || diag.len() != self.p {
            return Err(DoglegError::DimensionMismatch);
        }

        // STEP 1: Compute D^{-1} g and its norm
        // workp = D^{-1} g
        for i in 0..self.p {
            if diag[i] == 0.0 {
                return Err(DoglegError::SingularMatrix);
            }
            self.workp[i] = gradient[i] / diag[i];
        }
        self.norm_dinvg = self.workp.norm();

        // STEP 2: Compute D^{-2} g
        // workp = D^{-2} g (divide again by diagonal)
        for i in 0..self.p {
            self.workp[i] /= diag[i];
        }

        // STEP 3: Compute J D^{-2} g and its norm
        // workn = J * D^{-2} g
        self.workn = jacobian * &self.workp;
        self.norm_jdinv2g = self.workn.norm();

        if self.norm_jdinv2g == 0.0 {
            return Err(DoglegError::NumericalError);
        }

        // STEP 4: Compute steepest descent step length
        let u = self.norm_dinvg / self.norm_jdinv2g;
        let alpha = u * u; // step length for steepest descent

        // STEP 5: Set steepest descent step
        // dx_sd = -alpha * D^{-2} g
        self.dx_sd = -alpha * &self.workp;

        // STEP 6: Compute scaled norm of steepest descent step
        self.norm_dsd = self.scaled_norm(&self.dx_sd, diag);

        // Reset GN computation flag
        self.gn_computed = false;
        self.norm_dgn = -1.0;

        Ok(())
    }

    /// Compute the dogleg step
    pub fn step(
        &mut self,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
        diag: &DVector<f64>,
        delta: f64,
    ) -> Result<DVector<f64>, DoglegError> {
        // CASE 1: Steepest descent step exceeds trust region
        if self.norm_dsd >= delta {
            // Truncate steepest descent to trust region boundary
            let dx = (delta / self.norm_dsd) * &self.dx_sd;
            return Ok(dx);
        }

        // CASE 2: Steepest descent step is inside trust region
        // Compute Gauss-Newton step if needed
        if !self.gn_computed {
            self.compute_gauss_newton_step(jacobian, residual)?;
            self.norm_dgn = self.scaled_norm(&self.dx_gn, diag);
            self.gn_computed = true;
        }

        // CASE 2a: Gauss-Newton step is inside trust region
        if self.norm_dgn <= delta {
            // GN step is optimal since it minimizes quadratic model
            println!("gauss newton step norm <= delta: return gauss newton step");
            return Ok(self.dx_gn.clone());
        }

        // CASE 2b: Use dogleg interpolation
        // Find point on dogleg path: dx = dx_sd + β*(dx_gn - dx_sd)
        // where β ∈ [0,1] such that ||D*dx|| = delta
        let beta = self.compute_dogleg_beta(1.0, delta, diag)?;
        println!("beta = {}", beta);
        // Compute final dogleg step
        // dx = dx_sd + β * (dx_gn - dx_sd)
        let dx_diff = &self.dx_gn - &self.dx_sd;
        let dx = &self.dx_sd + beta * dx_diff;

        Ok(dx)
    }

    /// Compute double dogleg step (enhanced version)
    pub fn double_step(
        &mut self,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
        gradient: &DVector<f64>,
        diag: &DVector<f64>,
        delta: f64,
    ) -> Result<DVector<f64>, DoglegError> {
        const ALPHA_FAC: f64 = 0.8; // recommended value from Dennis and Mei

        // CASE 1: Steepest descent step exceeds trust region
        if self.norm_dsd >= delta {
            let dx = (delta / self.norm_dsd) * &self.dx_sd;
            return Ok(dx);
        }

        // Compute Gauss-Newton step if needed
        if !self.gn_computed {
            self.compute_gauss_newton_step(jacobian, residual)?;
            self.norm_dgn = self.scaled_norm(&self.dx_gn, diag);
            self.gn_computed = true;
        }

        // CASE 2a: Gauss-Newton step is inside trust region
        if self.norm_dgn <= delta {
            return Ok(self.dx_gn.clone());
        }

        // CASE 2b: Double dogleg computation
        // Compute parameters for double dogleg
        let v_ratio = self.norm_dinvg / self.norm_jdinv2g;
        let u = v_ratio * v_ratio;

        // Compute v = g^T dx_gn
        let v = gradient.dot(&self.dx_gn);

        // Compute c = ||D^{-1} g||^4 / (||J D^{-2} g||^2 * |g^T dx_gn|)
        let c = u * (self.norm_dinvg / v.abs()) * self.norm_dinvg;

        // Compute t = 1 - alpha_fac*(1-c)
        let t = 1.0 - ALPHA_FAC * (1.0 - c);

        if t * self.norm_dgn <= delta {
            // Set dx = (delta / ||D dx_gn||) dx_gn
            let dx = (delta / self.norm_dgn) * &self.dx_gn;
            return Ok(dx);
        } else {
            // Use double dogleg step with parameter t
            let beta = self.compute_dogleg_beta(t, delta, diag)?;

            // Compute: dx = dx_sd + β*(t*dx_gn - dx_sd)
            let t_dx_gn = t * &self.dx_gn;
            let dx_diff = t_dx_gn - &self.dx_sd;
            let dx = &self.dx_sd + beta * dx_diff;

            return Ok(dx);
        }
    }

    /// Compute predicted reduction in objective function
    pub fn predicted_reduction(
        &mut self,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
        gradient: &DVector<f64>,
        dx: &DVector<f64>,
    ) -> Result<f64, DoglegError> {
        // pred = -g^T dx - (1/2) dx^T J^T J dx
        let linear_term = -gradient.dot(dx);

        // Compute J*dx
        self.workn = jacobian * dx;
        let quadratic_term = -0.5 * dx.dot(&(jacobian.transpose() * &self.workn));

        Ok(linear_term + quadratic_term)
    }

    /// Compute Gauss-Newton step by solving J * dx_gn = -f
    fn compute_gauss_newton_step(
        &mut self,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
    ) -> Result<(), DoglegError> {
        // Form normal equations: J^T J dx = -J^T f
        let jtj = jacobian.transpose() * jacobian;
        let jtf = jacobian.transpose() * residual;
        let rhs = -jtf;

        // Solve using LU decomposition
        match jtj.lu().solve(&rhs) {
            Some(solution) => {
                self.dx_gn = solution;
                Ok(())
            }
            None => Err(DoglegError::SingularMatrix),
        }
    }

    /// Compute beta parameter for dogleg interpolation
    /// Solves: a*β² + b*β + c = 0 for β ∈ [0,1]
    fn compute_dogleg_beta(
        &mut self,
        t: f64,
        delta: f64,
        diag: &DVector<f64>,
    ) -> Result<f64, DoglegError> {
        // Compute: workp = t*dx_gn - dx_sd
        self.workp = t * &self.dx_gn - &self.dx_sd;

        // a = ||D(t*dx_gn - dx_sd)||²
        let a = self.scaled_norm(&self.workp, diag).powi(2);

        // Compute D^T D (t*dx_gn - dx_sd) for b calculation
        for i in 0..self.p {
            self.workp[i] *= diag[i] * diag[i];
        }

        // b = 2 * dx_sd^T * D^T * D * (t*dx_gn - dx_sd)
        let b = 2.0 * self.dx_sd.dot(&self.workp);

        // c = ||D dx_sd||² - delta² = (||D dx_sd|| + delta)(||D dx_sd|| - delta)
        let c = (self.norm_dsd + delta) * (self.norm_dsd - delta);

        // Solve quadratic equation using numerically stable formula
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return Err(DoglegError::NumericalError);
        }

        let beta = if b > 0.0 {
            (-2.0 * c) / (b + discriminant.sqrt())
        } else {
            (-b + discriminant.sqrt()) / (2.0 * a)
        };

        // Ensure beta is in valid range [0, 1]
        Ok(beta.max(0.0).min(1.0))
    }

    /// Compute scaled norm ||D * x||
    fn scaled_norm(&self, x: &DVector<f64>, diag: &DVector<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() {
            let scaled = diag[i] * x[i];
            sum += scaled * scaled;
        }
        sum.sqrt()
    }
}

/// High-level interface for dogleg algorithm
pub struct DoglegSolver {
    state: DoglegState,
    use_double_dogleg: bool,
}

impl DoglegSolver {
    /// Create new dogleg solver
    pub fn new(n: usize, p: usize, use_double_dogleg: bool) -> Result<Self, DoglegError> {
        Ok(DoglegSolver {
            state: DoglegState::new(n, p)?,
            use_double_dogleg,
        })
    }

    /// Initialize solver for new iteration
    pub fn initialize(
        &mut self,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
        gradient: &DVector<f64>,
        diag: &DVector<f64>,
    ) -> Result<(), DoglegError> {
        self.state.preloop(jacobian, residual, gradient, diag)
    }

    /// Compute dogleg step
    pub fn solve_step(
        &mut self,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
        gradient: &DVector<f64>,
        diag: &DVector<f64>,
        delta: f64,
    ) -> Result<DVector<f64>, DoglegError> {
        if self.use_double_dogleg {
            self.state
                .double_step(jacobian, residual, gradient, diag, delta)
        } else {
            self.state.step(jacobian, residual, diag, delta)
        }
    }

    /// Compute predicted reduction
    pub fn predicted_reduction(
        &mut self,
        jacobian: &DMatrix<f64>,
        residual: &DVector<f64>,
        gradient: &DVector<f64>,
        dx: &DVector<f64>,
    ) -> Result<f64, DoglegError> {
        self.state
            .predicted_reduction(jacobian, residual, gradient, dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    // Helper function to create a simple test problem
    fn create_test_problem() -> (DMatrix<f64>, DVector<f64>, DVector<f64>, DVector<f64>) {
        // Simple 2x2 test case
        let jacobian = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]);

        let residual = DVector::from_vec(vec![1.0, 2.0]);
        let gradient = jacobian.transpose() * &residual; // J^T * r
        let diag = DVector::from_vec(vec![1.0, 1.0]); // Identity scaling

        (jacobian, residual, gradient, diag)
    }

    // Helper function to create a more complex test problem
    fn create_complex_test_problem() -> (DMatrix<f64>, DVector<f64>, DVector<f64>, DVector<f64>) {
        let jacobian = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 1.0, 2.0, 2.0]);

        let residual = DVector::from_vec(vec![1.0, -1.0, 0.5]);
        let gradient = jacobian.transpose() * &residual;
        let diag = DVector::from_vec(vec![2.0, 1.5]); // Non-identity scaling

        (jacobian, residual, gradient, diag)
    }

    #[test]
    fn test_dogleg_state_creation() {
        let state = DoglegState::new(3, 2);
        assert!(state.is_ok());

        let state = state.unwrap();
        assert_eq!(state.n, 3);
        assert_eq!(state.p, 2);
        assert_eq!(state.dx_gn.len(), 2);
        assert_eq!(state.dx_sd.len(), 2);
        assert_eq!(state.workp.len(), 2);
        assert_eq!(state.workn.len(), 3);
        assert!(!state.gn_computed);
        assert_eq!(state.norm_dgn, -1.0);
    }

    #[test]
    fn test_dogleg_state_preloop() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut state = DoglegState::new(2, 2).unwrap();

        let result = state.preloop(&jacobian, &residual, &gradient, &diag);
        assert!(result.is_ok());

        // Check that steepest descent step was computed
        assert!(state.norm_dsd > 0.0);
        assert!(state.norm_dinvg > 0.0);
        assert!(state.norm_jdinv2g > 0.0);

        // Verify steepest descent step direction (should be opposite to scaled gradient)
        let expected_direction = -gradient.component_div(&diag).component_div(&diag);
        let actual_direction = state.dx_sd.normalize();
        let expected_direction_norm = expected_direction.normalize();

        // Check if directions are parallel (dot product close to ±1)
        let dot_product = actual_direction.dot(&expected_direction_norm).abs();
        assert!(dot_product > 0.9, "Steepest descent direction incorrect");
    }

    #[test]
    fn test_dogleg_state_preloop_dimension_mismatch() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut state = DoglegState::new(3, 2).unwrap(); // Wrong dimensions

        let result = state.preloop(&jacobian, &residual, &gradient, &diag);
        assert!(matches!(result, Err(DoglegError::DimensionMismatch)));
    }

    #[test]
    fn test_dogleg_state_preloop_singular_matrix() {
        let jacobian = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 1.0]);
        let residual = DVector::from_vec(vec![1.0, 2.0]);
        let gradient = jacobian.transpose() * &residual;
        let diag = DVector::from_vec(vec![0.0, 1.0]); // Singular scaling

        let mut state = DoglegState::new(2, 2).unwrap();
        let result = state.preloop(&jacobian, &residual, &gradient, &diag);
        assert!(matches!(result, Err(DoglegError::SingularMatrix)));
    }

    #[test]
    fn test_dogleg_step_steepest_descent_case() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut state = DoglegState::new(2, 2).unwrap();

        state
            .preloop(&jacobian, &residual, &gradient, &diag)
            .unwrap();

        // Set trust region smaller than steepest descent step
        let delta = state.norm_dsd * 0.5;
        let step = state.step(&jacobian, &residual, &diag, delta).unwrap();

        // Check that step is truncated steepest descent
        let scaled_norm = state.scaled_norm(&step, &diag);
        assert_relative_eq!(scaled_norm, delta, epsilon = 1e-10);

        // Check direction is same as steepest descent
        let step_direction = step.normalize();
        let sd_direction = state.dx_sd.normalize();
        let dot_product = step_direction.dot(&sd_direction);
        assert!(
            dot_product > 0.99,
            "Step should be in steepest descent direction"
        );
    }

    #[test]
    fn test_dogleg_step_gauss_newton_case() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut state = DoglegState::new(2, 2).unwrap();

        state
            .preloop(&jacobian, &residual, &gradient, &diag)
            .unwrap();

        // Set large trust region
        let delta = 100.0;
        let step = state.step(&jacobian, &residual, &diag, delta).unwrap();

        // Should compute Gauss-Newton step
        assert!(state.gn_computed);

        // Step should be close to Gauss-Newton step
        let diff = (&step - &state.dx_gn).norm();
        assert!(
            diff < 1e-10,
            "Step should be Gauss-Newton step for large trust region"
        );
    }

    #[test]
    fn test_dogleg_step_interpolation_case() {
        let (jacobian, residual, gradient, diag) = create_complex_test_problem();
        let mut state = DoglegState::new(3, 2).unwrap();

        state
            .preloop(&jacobian, &residual, &gradient, &diag)
            .unwrap();

        // Force computation of Gauss-Newton step
        state
            .compute_gauss_newton_step(&jacobian, &residual)
            .unwrap();
        state.norm_dgn = state.scaled_norm(&state.dx_gn, &diag);
        state.gn_computed = true;

        // Set trust region between SD and GN norms
        let delta = (state.norm_dsd + state.norm_dgn) * 0.5;
        let step = state.step(&jacobian, &residual, &diag, delta).unwrap();

        // Check that step norm equals trust region radius
        let scaled_norm = state.scaled_norm(&step, &diag);
        assert_relative_eq!(scaled_norm, delta, epsilon = 1e-8);

        // Step should be between SD and GN steps
        let is_between_sd_gn = {
            let step_minus_sd = &step - &state.dx_sd;
            let gn_minus_sd = &state.dx_gn - &state.dx_sd;
            let projection = step_minus_sd.dot(&gn_minus_sd) / gn_minus_sd.norm_squared();
            projection >= 0.0 && projection <= 1.0
        };
        assert!(
            is_between_sd_gn,
            "Step should be on dogleg path between SD and GN"
        );
    }

    #[test]
    fn test_dogleg_solver_creation() {
        let solver = DoglegSolver::new(3, 2, false);
        assert!(solver.is_ok());

        let solver = solver.unwrap();
        assert_eq!(solver.state.n, 3);
        assert_eq!(solver.state.p, 2);
        assert!(!solver.use_double_dogleg);

        let double_solver = DoglegSolver::new(3, 2, true);
        assert!(double_solver.is_ok());
        assert!(double_solver.unwrap().use_double_dogleg);
    }

    #[test]
    fn test_dogleg_solver_initialize() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut solver = DoglegSolver::new(2, 2, false).unwrap();

        let result = solver.initialize(&jacobian, &residual, &gradient, &diag);
        assert!(result.is_ok());

        // Check that initialization was successful
        assert!(solver.state.norm_dsd > 0.0);
        assert!(solver.state.norm_dinvg > 0.0);
    }

    #[test]
    fn test_dogleg_solver_solve_step() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut solver = DoglegSolver::new(2, 2, false).unwrap();

        solver
            .initialize(&jacobian, &residual, &gradient, &diag)
            .unwrap();

        let delta = 1.0;
        let step = solver.solve_step(&jacobian, &residual, &gradient, &diag, delta);
        assert!(step.is_ok());

        let step = step.unwrap();
        assert_eq!(step.len(), 2);

        // Check that step satisfies trust region constraint
        let scaled_norm = solver.state.scaled_norm(&step, &diag);
        assert!(
            scaled_norm <= delta + 1e-10,
            "Step should satisfy trust region constraint"
        );
    }

    #[test]
    fn test_dogleg_solver_double_dogleg() {
        let (jacobian, residual, gradient, diag) = create_complex_test_problem();
        let mut solver = DoglegSolver::new(3, 2, true).unwrap(); // Enable double dogleg

        solver
            .initialize(&jacobian, &residual, &gradient, &diag)
            .unwrap();

        let delta = 0.5;
        let step = solver.solve_step(&jacobian, &residual, &gradient, &diag, delta);
        assert!(step.is_ok());

        let step = step.unwrap();
        let scaled_norm = solver.state.scaled_norm(&step, &diag);
        assert!(
            scaled_norm <= delta + 1e-10,
            "Double dogleg step should satisfy trust region"
        );
    }

    #[test]
    fn test_dogleg_solver_predicted_reduction() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut solver = DoglegSolver::new(2, 2, false).unwrap();

        solver
            .initialize(&jacobian, &residual, &gradient, &diag)
            .unwrap();

        let delta = 1.0;
        let step = solver
            .solve_step(&jacobian, &residual, &gradient, &diag, delta)
            .unwrap();

        let pred_reduction = solver.predicted_reduction(&jacobian, &residual, &gradient, &step);
        assert!(pred_reduction.is_ok());

        let pred_reduction = pred_reduction.unwrap();
        // For a descent direction, predicted reduction should be positive
        assert!(
            pred_reduction > 0.0,
            "Predicted reduction should be positive for descent step"
        );
    }

    #[test]
    fn test_dogleg_beta_computation() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut state = DoglegState::new(2, 2).unwrap();

        state
            .preloop(&jacobian, &residual, &gradient, &diag)
            .unwrap();
        state
            .compute_gauss_newton_step(&jacobian, &residual)
            .unwrap();
        state.gn_computed = true;

        let delta = (state.norm_dsd + 1.0) * 0.7; // Between SD and some larger value
        let beta = state.compute_dogleg_beta(1.0, delta, &diag);

        assert!(beta.is_ok());
        let beta = beta.unwrap();
        assert!(beta >= 0.0 && beta <= 1.0, "Beta should be in [0,1]");
    }
    /*
        #[test]
    fn test_scaled_norm_computation() {
        let state = DoglegState::new(3, 2).unwrap();
        let vector = DVector::from_vec(vec![1.0, 2.0]);
        let diag = DVector::from_vec(vec![2.0, 0.5]);

        let scaled_norm = state.scaled_norm(&vector, &diag);
        let expected = ((2.0 * 1.0).powf(&2.0) + (0.5 * 2.0).powf(2.0)).sqrt();

        assert_relative_eq!(scaled_norm, expected, epsilon = 1e-10);
    }

     */
    #[test]
    fn test_problem_solve() {
        let (jacobian, residual, gradient, diag) = create_test_problem();
        let mut solver = DoglegSolver::new(2, 2, false).unwrap();
        println!(
            "residual {:?} ,\n gradient {:?} \n, diag {:?}",
            residual, gradient, diag
        );
        solver
            .initialize(&jacobian, &residual, &gradient, &diag)
            .unwrap();

        let delta = 100.0;
        let step = solver.solve_step(&jacobian, &residual, &gradient, &diag, delta);

        assert!(step.is_ok());
        println!("Step: {}", step.unwrap());
        let step1 = Powell_dogleg_method(jacobian, residual, diag, delta, "lu".to_owned());
        println!("step1: {}", step1);
    }
}
