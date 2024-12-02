use core::fmt::Display;

/// Backward Euler method for solving systems of ordinary differential equation
/// Newton-Raphson calculation on each step of the method is made by using the analytic jacobian
use crate::numerical::NR_for_Euler::NRE;
use crate::symbolic::symbolic_engine::Expr;
use crate::Utils::plots::plots;
use nalgebra::{DMatrix, DVector};
//use ndarray_linalg::Norm;
use log::info;
use std::time::Instant;
pub enum Equation {
    LHS(Vec<Expr>),
    RHS(Vec<Expr>),
}
//#[derive(Debug)]
pub struct BE {
    pub newton: NRE,
    y0: DVector<f64>,
    t0: f64,
    t_bound: f64,
    t: f64,
    y: DVector<f64>,
    t_old: Option<f64>,
    t_result: DVector<f64>,
    y_result: DMatrix<f64>,
    status: String,
    message: Option<String>,
    h: Option<f64>,
    global_timestepping: bool,
}
impl Display for BE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BE {{ t0: {}, t_bound: {}, t: {}, y: {:?} }}",
            self.t0, self.t_bound, self.t, self.y
        )
    }
}

impl BE {
    pub fn new() -> BE {
        let nr_new = NRE::new(
            Vec::new(),
            DVector::zeros(0),
            Vec::new(),
            String::new(),
            0.0,
            0,
            0.0,
            true,
            None,
        );
        BE {
            newton: nr_new,
            y0: DVector::zeros(0),
            t0: 0.0,
            t_bound: 0.0,
            t: 0.0,
            y: DVector::zeros(0),
            t_old: None,
            t_result: DVector::zeros(0),
            y_result: DMatrix::zeros(0, 0),
            status: "running".to_string(),
            message: None,
            h: None,
            global_timestepping: true,
        }
    }
    pub fn set_initial(
        &mut self,
        eq_system: Vec<Expr>, //
        values: Vec<String>,
        arg: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations
        h: Option<f64>,
        t0: f64,
        t_bound: f64,
        y0: DVector<f64>,
    ) -> () {
        let initial_guess = y0.clone();
        let nr = if let Some(dt) = h {
            self.global_timestepping = true;
            NRE::new(
                eq_system,
                initial_guess,
                values,
                arg,
                tolerance,
                max_iterations,
                dt,
                true,
                None,
            )
        } else {
            info!("global_timestepping = false");
            self.global_timestepping = false;
            NRE::new(
                eq_system,
                initial_guess,
                values,
                arg,
                tolerance,
                max_iterations,
                1e-4,
                false,
                Some(t_bound),
            )
        };

        self.newton = nr;
        self.t0 = t0;
        self.t_bound = t_bound;
        self.y0 = y0.clone();
        self.t = t0;
        self.y = y0.clone();
        self.check();
    }
    pub fn check(&self) -> () {
        assert_eq!(!self.y.is_empty(), true, "initial y is empty");
        assert_eq!(!self.newton.eq_system.is_empty(), true, "system is empty");
        assert_eq!(
            !self.newton.initial_guess.is_empty(),
            true,
            "guess is empty"
        );
        assert_eq!(!self.newton.values.is_empty(), true, "values are empty");
        assert_eq!(!self.newton.arg.is_empty(), true, "arg is empty");
        assert_eq!(self.newton.tolerance >= 0.0, true, "tolerance is empty");
        assert_eq!(
            self.newton.max_iterations >= 1,
            true,
            "max_iterations is empty"
        );
        assert_eq!(self.newton.dt >= 0.0, true, "h is empty");
        assert_eq!(
            self.global_timestepping == true || self.h.is_none(),
            true,
            "for global timestepping h must be set"
        );
    }

    pub fn _step_impl(&mut self) -> (bool, Option<String>) {
        let nr = &mut self.newton;
        let guess_i = self.y.clone();
        let t_i = nr.dt + self.t;
        nr.set_t(t_i);
        //  info("guess = {:?}, t_i = {:?} ", &guess_i, t_i);
        nr.set_initial_guess(guess_i);
        nr.solve();
        let result = nr.get_result();
        if result.is_none() {
            info!("result is None");
            return (
                false,
                Some("maximum number of iterations reached".to_string()),
            );
        } else {
            self.y = nr.get_result().expect("REASON");

            self.t = t_i;
            //   info("y = {:?} ", &self.y);
            return (true, None);
        }
    }

    ///________________________________________________________________________________________________________________________________
    pub fn step(&mut self) {
        //  let (success, message_) =self.Solver_instance._step_impl();

        // Analogue of step function in https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/base.py
        let t = self.t;
        if t == self.t_bound {
            self.t_old = Some(t);

            self.status = "finished".to_string();
        } else {
            let (success, message_) = self._step_impl();

            if let Some(message_str) = message_ {
                self.message = Some(message_str.to_string());
            } else {
                self.message = None;
            }

            if success == false {
                self.status = "failed".to_string();
            } else {
                self.t_old = Some(t);
                let _status: String = "running".to_string();
                if (self.t - self.t_bound) >= 0.0 {
                    //self.Solver_instance.direction *
                    self.status = "finished".to_string();
                }
            }
        }
    } //step

    pub fn main_loop(&mut self) -> () {
        let start = Instant::now();
        // Analogue of https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/ivp.py

        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<f64> = Vec::new();
        let mut _i: i64 = 0;
        while integr_status.is_none() {
            self.step();
            let _status: i8 = 0;
            //   info("\n iteration: {}", i);
            //if i == 100 {panic!()}
            _i += 1;
            if self.status == "finished".to_string() {
                integr_status = Some(0)
            } else if self.status == "failed".to_string() {
                integr_status = Some(-1);
                break;
            }
            //  info("i: {}, t: {}, y: {:?}, _status: {}", i, self.Solver_instance.t, self.Solver_instance.y, _status);
            t.push(self.t);
            y.push(self.y.clone());
            // info("time  {:?}, len {}", t, t.len())
        }

        let rows = &y.len();
        let cols = &y[0].len();

        let mut flat_vec: Vec<f64> = Vec::new();
        for vector in y.iter() {
            flat_vec.extend(vector)
        }
        let y_res: DMatrix<f64> = DMatrix::from_vec(*cols, *rows, flat_vec).transpose();
        let t_res = DVector::from_vec(t);

        // info("time  {:?}, len {}", &t_res, t_res.len());
        //info("y  {:?}, len {:?}", &y_res, y_res.shape());
        let duration = start.elapsed();
        info!("Program took {} milliseconds to run", duration.as_millis());
        self.t_result = t_res.clone();
        self.y_result = y_res.clone();
    } //

    pub fn solve(&mut self) -> () {
        self.newton.eq_generate();
        self.main_loop();
    }
    pub fn plot_result(&self) -> () {
        plots(
            self.newton.arg.clone(),
            self.newton.values.clone(),
            self.t_result.clone(),
            self.y_result.clone(),
        );
        info!("result plotted");
    }

    pub fn get_result(&self) -> (Option<DVector<f64>>, Option<DMatrix<f64>>) {
        (Some(self.t_result.clone()), Some(self.y_result.clone()))
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_newton_raphson_solver_for_Euler_1() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let y0 = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-2;
        let max_iterations = 50;

        let h = Some(1e-2);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut solver = BE::new();

        solver.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        info!(
            "y = {:?}, initial_guess = {:?}",
            solver.newton.y, solver.newton.initial_guess
        );
        solver.newton.eq_generate();
        let (success, message) = solver._step_impl();
        assert_eq!(solver.y.len(), 2);
        assert_eq!(success, true, "success = {} must be true", success);
        assert_eq!(message, None, "message = {:?} must be None", message);
    }

    #[test]

    fn test_newton_raphson_solver_for_Euler_2() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let y0 = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-2;
        let max_iterations = 50;

        let h = Some(1e-2);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut solver = BE::new();

        solver.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        info!(
            "y = {:?}, initial_guess = {:?}",
            solver.newton.y, solver.newton.initial_guess
        );
        solver.newton.eq_generate();
        solver.step();
        assert_eq!(solver.status, "running".to_string());
    }

    #[test]

    fn test_newton_raphson_solver_for_Euler_3() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let y0 = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-2;
        let max_iterations = 50;

        let h = Some(1e-2);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut solver = BE::new();

        solver.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        info!(
            "y = {:?}, initial_guess = {:?}",
            solver.newton.y, solver.newton.initial_guess
        );
        solver.newton.eq_generate();
        solver.solve();
        assert_eq!(solver.status, "finished".to_string());
    }

    #[test]
    fn test_newton_raphson_solver_for_Euler_4() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let y0 = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-2;
        let max_iterations = 50;

        let h = None;
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut solver = BE::new();

        solver.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        info!(
            "y = {:?}, initial_guess = {:?}",
            solver.newton.y, solver.newton.initial_guess
        );
        solver.newton.eq_generate();
        solver.solve();
        let res = solver.get_result();
        let _result = res.1.unwrap();
        // assert_eq!(result.shape(), (2, 1)) ;
        assert_eq!(solver.status, "finished".to_string());
    }
}

/*

    pub fn set_step(&mut self, step: f64) {
        self.step = step;
    }
    pub fn set_initial_сonditions(&mut self, initial_сonditions: Vec<f64>) {
        self.initial_сonditions = initial_сonditions;
    }
    /// because we need two types of unknown variables - y_k and y_k-1 -> ctrate y_k_min_1
    pub fn y_k_min_1_generate(&mut self) -> Vec<Expr> {
        let mut y_k_min_1:Vec<Expr> = Vec::new();
        for y_i_k in self.y.iter_mut() {
            let y_i_k = y_i_k.to_string();
            let y_i_k_minus_1:String = format!("{}_min_1", y_i_k);
            let y_i_k_min_1: Expr = Expr::Var(y_i_k_minus_1);
            y_k_min_1.push(y_i_k_min_1);
        }
        y_k_min_1
    }
    pub fn set_eq_from_str(&mut self, eq_system_string: Vec<String>, unkn_vars: Vec<String>) {
        let eq_system = eq_system_string.iter().map(|x| Expr::parse_expression(x)).collect::<Vec<Expr>>();
        assert!(!eq_system.is_empty(), "Equation system should not be empty.");
        self.RHS = eq_system;
        let y = unkn_vars.iter().map(|x| Expr::Var(x.to_string())).collect::<Vec<Expr>>();
        self.y = y;

    }

    pub fn set_equation_system(&mut self){
        assert_eq!(self.y.len()==self.RHS.len(), true, "LHS and RHS should have the same length.");
        let y_k_min_1 = self.y_k_min_1_generate();
        /// constructing equations
        /// y_k = y_k-1 - h * f(t_k+1, y_k+1)
        /// equation = y_k - y_k-1 - h * f(t_k+1, y_k+1)
        let y_k = self.y.clone();
        let h = Expr::Const(self.step);
        let mut equations: Vec<Expr> = Vec::new();
        for  i in 0..self.y.clone().iter().len() {
            let equation = y_k[i].clone() - y_k_min_1[i].clone()    - h.clone() *self.RHS[i].clone() ;
            info!("equation {}: {}",i, equation);
            equations.push(equation);

    }

        self.equations = equations;
    }// end of set_equation_system
fn main() {
    // Example ODE: y' = -2y + 1
    let f = |t: f64, y: f64| -2.0 * y + 1.0;
    let y0 = 0.5;
    let t0 = 0.0;
    let tn = 2.0;
    let h = 0.1;

    let result = backward_euler(f, y0, t0, tn, h);

    for (t, y) in result {
        info!("t: {:.2}, y: {:.5}", t, y);
    }
}


extern crate nalgebra as na;

use na::{DVector, DMatrix};
use std::f64::EPSILON;

fn backward_euler<F>(f: F, y0: DVector<f64>, t0: f64, tn: f64, h: f64) -> Vec<(f64, DVector<f64>)>
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    let mut t = t0;
    let mut y = y0.clone();
    let mut result = vec![(t, y.clone())];

    while t < tn {
        let t_next = t + h;
        let y_next = newton_raphson(|y_next| y_next.clone() - y.clone() - h * f(t_next, &y_next), y.clone(), 1e-6, 100);
        t = t_next;
        y = y_next.clone();
        result.push((t, y));
    }

    result
}

fn newton_raphson<F>(f: F, initial_guess: DVector<f64>, tol: f64, max_iter: usize) -> DVector<f64>
where
    F: Fn(DVector<f64>) -> DVector<f64>,
{
    let mut x = initial_guess;
    for _ in 0..max_iter {
        let fx = f(x.clone());
        if fx.norm() < tol {
            return x;
        }
        let jacobian = approximate_jacobian(&f, &x);
        let delta_x = jacobian.lu().solve(&fx).unwrap();
        x = x - delta_x;
    }
    x
}

fn approximate_jacobian<F>(f: &F, x: &DVector<f64>) -> DMatrix<f64>
where
    F: Fn(DVector<f64>) -> DVector<f64>,
{
    let n = x.len();
    let mut jacobian = DMatrix::zeros(n, n);
    let fx = f(x.clone());

    for i in 0..n {
        let mut x_eps = x.clone();
        x_eps[i] += EPSILON;
        let fx_eps = f(x_eps);
        let df = (fx_eps - fx.clone()) / EPSILON;
        for j in 0..n {
            jacobian[(j, i)] = df[j];
        }
    }

    jacobian
}

fn main() {
    // Example system of ODEs: y1' = -2y1 + y2, y2' = y1 - 2y2
    let f = |t: f64, y: &DVector<f64>| {
        let mut dy = DVector::zeros(2);
        dy[0] = -2.0 * y[0] + y[1];
        dy[1] = y[0] - 2.0 * y[1];
        dy
    };

    let y0 = DVector::from_vec(vec![1.0, 0.0]);
    let t0 = 0.0;
    let tn = 2.0;
    let h = 0.1;

    let result = backward_euler(f, y0, t0, tn, h);

    for (t, y) in result {
        info!("t: {:.2}, y: {:?}", t, y);
    }
}

*/
