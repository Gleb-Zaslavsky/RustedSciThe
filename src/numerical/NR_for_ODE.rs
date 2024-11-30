use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector, Matrix};
use log::info;
// solve algebraic nonlinear system with free parameter t
pub struct NRODE {
    pub eq_system: Vec<Expr>,    //
    initial_guess: DVector<f64>, // initial guess
    values: Vec<String>,
    arg: String,
    tolerance: f64,               // tolerance
    max_iterations: usize,        // max number of iterations
    max_error: f64,               // max error
    result: Option<DVector<f64>>, // result of the iteration

    fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>>,
    t: f64,
    y: DVector<f64>,
}

impl NRODE {
    pub fn new(
        eq_system: Vec<Expr>,        //
        initial_guess: DVector<f64>, // initial guess
        values: Vec<String>,
        arg: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations
        max_error: f64,        // max error
    ) -> NRODE {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        NRODE {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            tolerance,
            max_iterations,
            max_error,
            result: None,

            fun: Box::new(|_t, y| y.clone()),
            jac: None,
            t: 0.0,
            y: initial_guess.clone(),
        }
    }
    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn eq_generate(&mut self) {
        let mut jacobian_instance = Jacobian::new();
        jacobian_instance.generate_IVP_ODEsolver(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
        );
        //     ("Jacobian = {:?}", jacobian_instance.readable_jacobian);
        let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            jacobian_instance.lambdified_functions_IVP_DVector;

        let jac = jacobian_instance.function_jacobian_IVP_DMatrix;

        let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
            Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> { jac(t, &y) });

        self.fun = fun;
        self.jac = Some(jac_wrapped);
    }

    pub fn set_new_step(&mut self, t: f64, y: DVector<f64>, initial_guess: DVector<f64>) {
        self.t = t;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_t(&mut self, t: f64) {
        self.t = t;
    }

    ///Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self) -> DVector<f64> {
        let t = self.t;
        let y = &self.y;
        info!("Newton-Raphson iteration {}", &y);
        let new_f = (self.fun)(t, &y);
        // info!("new_f = {:?}", &new_f);
        let new_j = &self.jac.as_mut().unwrap()(t, &y);

        let _x = self.initial_guess.clone();

        let j_inverse = new_j.clone().try_inverse().unwrap();

        let delta: DVector<f64> = j_inverse * new_f;
        //  info!("dx = {:?},new_j = {:?},", &delta, &new_j);
        // element wise subtraction
        let new_y: DVector<f64> = y - delta;

        new_y
    }
    // main function to solve the system of equations

    pub fn solve(&mut self) -> Option<DVector<f64>> {
        info!("solving system of equations with Newton-Raphson method");
        let mut y: DVector<f64> = self.initial_guess.clone();
        let mut i = 0;
        while i < self.max_iterations {
            let new_y = self.iteration();

            let dy = new_y.clone() - y.clone();

            let error = Matrix::norm(&dy);
            // info!("new_x = {:?}, x = {:?}", &new_y, &y);
            if error < self.tolerance {
                self.result = Some(new_y.clone());
                self.max_error = error;
                return Some(new_y);
            } else {
                y = new_y.clone();
                self.y = new_y;
                i += 1;
                // info!("iteration = {}, error = {}", i, error)
            }
        }
        None
    }

    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.result.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_newton_raphson_solver() {
        // Define a simple equation: x^2 - 4 = 0
        let eq1 = Expr::parse_expression("z^2+y^2-25.0*x");
        let eq2 = Expr::parse_expression("z+y-7.0*x");
        let eq_system = vec![eq1, eq2];

        let initial_guess = DVector::from_vec(vec![2.0, 3.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-6;
        let max_iterations = 100;
        let max_error = 0.0;

        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRODE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            max_error,
        );
        nr.eq_generate();
        assert_eq!(nr.eq_system.len(), 2);

        // Solve the equation at t=0 with initial guess y=[2.0]
        //    nr.set_new_step(0.0, DVector::from_element(1, 2.0), DVector::from_element(1, 2.0));
        let _solution = nr.solve().unwrap();
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }
}
