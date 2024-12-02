use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector, Matrix};
use std::fmt::Display;
use log::info;
// solve algebraic nonlinear system with free parameter t
//#[derive(Debug)]
pub struct NRE {
    pub eq_system: Vec<Expr>,        //
    pub initial_guess: DVector<f64>, // initial guess
    pub values: Vec<String>,
    pub arg: String,
    pub tolerance: f64,               // tolerance
    pub max_iterations: usize,        // max number of iterations
    pub max_error: f64,               // max error
    pub result: Option<DVector<f64>>, // result of the iteration

    pub fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    pub jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>>,
    pub t: f64,
    pub y: DVector<f64>,
    pub dt: f64,
    n: usize,
    pub global_timestepping: bool,
    pub t_bound: Option<f64>,
}

impl Display for NRE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //   write!(f, "{}", self.eq_system);
        write!(
            f,
            "Initial guess: {:?}, tolerance: {}, max_iterations: {}, max_error: {}, result: {:?}",
            self.initial_guess, self.tolerance, self.max_iterations, self.max_error, self.result
        )
    }
}

impl NRE {
    pub fn new(
        eq_system: Vec<Expr>,        //
        initial_guess: DVector<f64>, // initial guess
        values: Vec<String>,
        arg: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations

        dt: f64,
        global_timestepping: bool,
        t_bound: Option<f64>,
    ) -> NRE {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        NRE {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            tolerance,
            max_iterations,

            dt,
            global_timestepping,
            t_bound,
            result: None,

            fun: Box::new(|_t, y| y.clone()),
            jac: None,
            t: 0.0,
            y: initial_guess.clone(),
            max_error: 1e-3,
            n: 0,
        }
    }
    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn eq_generate(&mut self) {
        info!("generating equations and jacobian");
        let mut jacobian_instance = Jacobian::new();
        jacobian_instance.generate_IVP_ODEsolver(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
        );
        // println!("Jacobian = {:?}", jacobian_instance.symbolic_jacobian);
        let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            jacobian_instance.lambdified_functions_IVP_DVector;

        let jac = jacobian_instance.function_jacobian_IVP_DMatrix;

        let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
            Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> { jac(t, &y) });

        self.fun = fun;
        self.jac = Some(jac_wrapped);
        self.n = self.eq_system.len();
        assert_eq!(&self.eq_system.len(), &self.n);
        // assert_eq!(&(self.jac.unwrap()).is_empty(), &false, "jac is empty");
        //assert_eq!(&(self.fun).is_empty(), &false, "fun is empty");
    }

    pub fn set_new_step(&mut self, t: f64, y: DVector<f64>, initial_guess: DVector<f64>) {
        self.t = t;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_t(&mut self, t: f64) {
        self.t = t;
    }
    pub fn set_initial_guess(&mut self, initial_guess: DVector<f64>) {
        self.initial_guess = initial_guess;
    }

    ///Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self) -> DVector<f64> {
        assert_eq!(&self.y.is_empty(), &false, "y is empty");
        let t = self.t;
        let y = &self.y;
        let f = (self.fun)(t, &y);
        let new_j = &self.jac.as_mut().unwrap()(t, &y);
        let dt = if self.global_timestepping == true {
            self.dt
        } else {
            let t_bound = self
                .t_bound
                .expect("if global_timestepping = false, t_bound must be set");
            let JF = new_j * &f;
            let eps = self.tolerance; // ?????????????????????
            let norm_JF = JF.amax();
            let dt_ = if norm_JF > 0.0 {
                (2.0 * eps / norm_JF).sqrt()
            } else {
                t_bound - self.t
            };
            let dt = dt_.min(t_bound - self.t);
            self.dt = dt; // update global dt for next iteration
            dt
        };

        let y_k_minus_1 = &self.initial_guess;

        //   println!("Newton-Raphson iteration {}", &y);
        let new_G = y - y_k_minus_1 - dt * f;
        //   println!("new_f = {:?}", &new_G);

        let I = DMatrix::identity(self.n, self.n);
        // if new_j is jacobian of jacobian of f(t_k+1, y_k+1),  then jacobian of function G = y_k+1 - y_k - h*f(t_k+1, y_k+1) is
        let J = I - dt * new_j;
        //    println!("J = {:?} /n", &J);
        //equation J*deltay  = -G
        let lu = J.lu();
        let neg_f = -1.0 * new_G;
        let delta_y = lu.solve(&neg_f).expect("The matrix should be invertible");
        //    println!("delta_y = {:?},\n", &delta_y );
        let new_y: DVector<f64> = y + delta_y;

        new_y
    }
    // main function to solve the system of equations

    pub fn solve(&mut self) -> Option<DVector<f64>> {
        //  println!("solving system of equations with Newton-Raphson method");
        let mut y: DVector<f64> = self.initial_guess.clone();
        self.y = y.clone();
        let mut i = 0;
        while i < self.max_iterations {
            let new_y = self.iteration();

            let dy = new_y.clone() - y.clone();

            let error = Matrix::norm(&dy);
            //  println!("new_y = {:?}, dy = {:?}, error = {}", &new_y, &dy, error);
            if error < self.tolerance {
                //  println!("converged in {} iterations", i);
                self.result = Some(new_y.clone());
                self.max_error = error;
                return Some(new_y);
            } else {
                y = new_y.clone();
                self.y = new_y;
                i += 1;
                //  if i==5 {panic!("Too many iterations")}
                //  println!("\n \n iteration = {}, error = {}", i, error)
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
    fn test_newton_raphson_solver_for_Euler() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-3;
        let max_iterations = 50;

        let h = 1e-5;

        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            true,
            None,
        );
        nr.eq_generate();

        assert_eq!(nr.eq_system.len(), 2);
        nr.set_t(1.0);
        let solution = nr.solve().unwrap();
        assert_eq!(solution.len(), 2);
        // assert_eq!()
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }

    #[test]
    fn test_newton_raphson_solver_for_Euler_2() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-3;
        let max_iterations = 50;

        let h = 1e-5;

        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            false,
            Some(1.0),
        );
        nr.eq_generate();

        assert_eq!(nr.eq_system.len(), 2);
        nr.set_t(1.0);
        let solution = nr.solve().unwrap();
        assert_eq!(solution.len(), 2);
        // assert_eq!()
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }
}
