use nalgebra::{DMatrix, DVector};

/// A least squares minimization problem.
/// code from modules \numerical\optimization\problem_LM.rs, LM_optimization.rs, trust_region_LM.rs, utils.rs, qr_LM.rs
///  is rewrite of Levenberg-Marquardt crate https://crates.io/crates/levenberg-marquardt
///  We get rid of most of generics because in this crate we need only f64
/// Thus code is more simple and clear
/// This is what LevenbergMarquardt needs
/// to compute the residuals and the Jacobian. See the [module documentation](index.html)
/// for a usage example.
pub trait LeastSquaresProblem {
    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &DVector<f64>);

    /// Get the current parameter vector `$\vec{x}$`.
    fn params(&self) -> DVector<f64>;

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>>;

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<DMatrix<f64>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::optimization::LM_optimization::LevenbergMarquardt;

    /// Simple quadratic problem: minimize ||Ax - b||^2
    /// where A = [[1, 2], [3, 4], [5, 6]] and b = [1, 2, 3]
    #[derive(Clone)]
    struct QuadraticProblem {
        params: DVector<f64>,
        a: DMatrix<f64>,
        b: DVector<f64>,
    }

    impl QuadraticProblem {
        fn new() -> Self {
            Self {
                params: DVector::from_vec(vec![0.0, 0.0]),
                a: DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                b: DVector::from_vec(vec![1.0, 2.0, 3.0]),
            }
        }
    }

    impl LeastSquaresProblem for QuadraticProblem {
        fn set_params(&mut self, x: &DVector<f64>) {
            self.params.copy_from(x);
        }

        fn params(&self) -> DVector<f64> {
            self.params.clone()
        }

        fn residuals(&self) -> Option<DVector<f64>> {
            Some(&self.a * &self.params - &self.b)
        }

        fn jacobian(&self) -> Option<DMatrix<f64>> {
            Some(self.a.clone())
        }
    }

    /// Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    /// Residuals: r1 = a-x, r2 = sqrt(b)*(y-x^2)
    #[derive(Clone)]
    struct RosenbrockProblem {
        params: DVector<f64>,
        a: f64,
        b: f64,
    }

    impl RosenbrockProblem {
        fn new() -> Self {
            Self {
                params: DVector::from_vec(vec![-1.2, 1.0]), // Standard starting point
                a: 1.0,
                b: 100.0,
            }
        }
    }

    impl LeastSquaresProblem for RosenbrockProblem {
        fn set_params(&mut self, x: &DVector<f64>) {
            self.params.copy_from(x);
        }

        fn params(&self) -> DVector<f64> {
            self.params.clone()
        }

        fn residuals(&self) -> Option<DVector<f64>> {
            let x = self.params[0];
            let y = self.params[1];
            Some(DVector::from_vec(vec![
                self.a - x,
                (self.b).sqrt() * (y - x * x),
            ]))
        }

        fn jacobian(&self) -> Option<DMatrix<f64>> {
            let x = self.params[0];
            Some(DMatrix::from_row_slice(
                2,
                2,
                &[-1.0, 0.0, -2.0 * x * (self.b).sqrt(), (self.b).sqrt()],
            ))
        }
    }

    /// Exponential fitting problem: fit y = a * exp(b * x) to data points
    #[derive(Clone)]
    struct ExponentialFitProblem {
        params: DVector<f64>,
        x_data: DVector<f64>,
        y_data: DVector<f64>,
    }

    impl ExponentialFitProblem {
        fn new() -> Self {
            // Generate some synthetic data: y = 2 * exp(0.5 * x) + noise
            let x_data = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
            let y_data = DVector::from_vec(vec![2.1, 3.2, 5.8, 9.1, 14.8]);

            Self {
                params: DVector::from_vec(vec![1.0, 1.0]), // Initial guess for [a, b]
                x_data,
                y_data,
            }
        }
    }

    impl LeastSquaresProblem for ExponentialFitProblem {
        fn set_params(&mut self, x: &DVector<f64>) {
            self.params.copy_from(x);
        }

        fn params(&self) -> DVector<f64> {
            self.params.clone()
        }

        fn residuals(&self) -> Option<DVector<f64>> {
            let a = self.params[0];
            let b = self.params[1];
            let mut residuals = DVector::zeros(self.x_data.len());

            for i in 0..self.x_data.len() {
                let x = self.x_data[i];
                let y_pred = a * (b * x).exp();
                residuals[i] = y_pred - self.y_data[i];
            }

            Some(residuals)
        }

        fn jacobian(&self) -> Option<DMatrix<f64>> {
            let a = self.params[0];
            let b = self.params[1];
            let mut jacobian = DMatrix::zeros(self.x_data.len(), 2);

            for i in 0..self.x_data.len() {
                let x = self.x_data[i];
                let exp_bx = (b * x).exp();
                jacobian[(i, 0)] = exp_bx; // ∂/∂a
                jacobian[(i, 1)] = a * x * exp_bx; // ∂/∂b
            }

            Some(jacobian)
        }
    }

    /// Powell's function: a classic test problem
    /// f(x) = (x1 + 10*x2)^2 + 5*(x3 - x4)^2 + (x2 - 2*x3)^4 + 10*(x1 - x4)^4
    #[derive(Clone)]
    struct PowellProblem {
        params: DVector<f64>,
    }

    impl PowellProblem {
        fn new() -> Self {
            Self {
                params: DVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]),
            }
        }
    }

    impl LeastSquaresProblem for PowellProblem {
        fn set_params(&mut self, x: &DVector<f64>) {
            self.params.copy_from(x);
        }

        fn params(&self) -> DVector<f64> {
            self.params.clone()
        }

        fn residuals(&self) -> Option<DVector<f64>> {
            let x1 = self.params[0];
            let x2 = self.params[1];
            let x3 = self.params[2];
            let x4 = self.params[3];

            Some(DVector::from_vec(vec![
                x1 + 10.0 * x2,
                (5.0_f64).sqrt() * (x3 - x4),
                (x2 - 2.0 * x3).powi(2),
                (10.0_f64).sqrt() * (x1 - x4).powi(2),
            ]))
        }

        fn jacobian(&self) -> Option<DMatrix<f64>> {
            let x1 = self.params[0];
            let x2 = self.params[1];
            let x3 = self.params[2];
            let x4 = self.params[3];

            Some(DMatrix::from_row_slice(
                4,
                4,
                &[
                    1.0,
                    10.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    (5.0_f64).sqrt(),
                    -(5.0_f64).sqrt(),
                    0.0,
                    2.0 * (x2 - 2.0 * x3),
                    -4.0 * (x2 - 2.0 * x3),
                    0.0,
                    2.0 * (10.0_f64).sqrt() * (x1 - x4),
                    0.0,
                    0.0,
                    -2.0 * (10.0_f64).sqrt() * (x1 - x4),
                ],
            ))
        }
    }

    #[test]
    fn test_quadratic_problem() {
        let problem = QuadraticProblem::new();
        let (result, report) = LevenbergMarquardt::new().minimize(problem);

        println!("Quadratic Problem:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());

        assert!(report.termination.was_successful());
    }

    #[test]
    fn test_rosenbrock_problem() {
        let problem = RosenbrockProblem::new();
        let (result, report) = LevenbergMarquardt::new().with_tol(1e-12).minimize(problem);

        println!("\nRosenbrock Problem:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());

        // The minimum should be at (1, 1)
        let final_params = result.params();
        assert!((final_params[0] - 1.0).abs() < 1e-6);
        assert!((final_params[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_fit_problem() {
        let problem = ExponentialFitProblem::new();
        let (result, report) = LevenbergMarquardt::new().minimize(problem);

        println!("\nExponential Fit Problem:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());

        // Should recover approximately a=2, b=0.5
        let final_params = result.params();
        assert!((final_params[0] - 2.0).abs() < 0.5);
        assert!((final_params[1] - 0.5).abs() < 0.5);
    }

    #[test]
    fn test_powell_problem() {
        let problem = PowellProblem::new();
        let (result, report) = LevenbergMarquardt::new().with_tol(1e-10).minimize(problem);

        println!("\nPowell Problem:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());

        // The minimum should be at (0, 0, 0, 0)
        let final_params = result.params();
        for &param in final_params.iter() {
            assert!(param.abs() < 1e-6);
        }
    }

    #[test]
    fn test_simple_linear_system() {
        // Solve Ax = b where A = [[2, 1], [1, 3]] and b = [3, 4]
        // Solution should be x = [1, 1]
        #[derive(Clone)]
        struct LinearSystem {
            params: DVector<f64>,
        }

        impl LinearSystem {
            fn new() -> Self {
                Self {
                    params: DVector::from_vec(vec![0.0, 0.0]),
                }
            }
        }

        impl LeastSquaresProblem for LinearSystem {
            fn set_params(&mut self, x: &DVector<f64>) {
                self.params.copy_from(x);
            }

            fn params(&self) -> DVector<f64> {
                self.params.clone()
            }

            fn residuals(&self) -> Option<DVector<f64>> {
                let x1 = self.params[0];
                let x2 = self.params[1];
                Some(DVector::from_vec(vec![
                    2.0 * x1 + x2 - 3.0,
                    x1 + 3.0 * x2 - 4.0,
                ]))
            }

            fn jacobian(&self) -> Option<DMatrix<f64>> {
                Some(DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]))
            }
        }

        let problem = LinearSystem::new();
        let (result, report) = LevenbergMarquardt::new().minimize(problem);

        println!("\nLinear System Problem:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());

        // Solution should be [1, 1]
        let final_params = result.params();
        assert!((final_params[0] - 1.0).abs() < 1e-10);
        assert!((final_params[1] - 1.0).abs() < 1e-10);
        assert!(report.objective_function < 1e-20);
    }
}
