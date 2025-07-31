use crate::numerical::BDF::BDF_solver::BDF;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
extern crate nalgebra as na;
use crate::Utils::plots::plots;
use crate::numerical::BDF::common::NumberOrVec;
use crate::numerical::BE::BE;
use crate::numerical::NonStiff_api::{DormandPrince, RK45};
use crate::numerical::Radau::Radau_main::{Radau, RadauOrder};
use csv::Writer;
use na::{DMatrix, DVector};
use std::env;
use std::path::Path;
use std::time::Instant;

static COMPLEX: [&str; 2] = ["BDF", "RADAU"];

const EASY: [&str; 2] = ["RK45", "DOPRI"];
pub enum Solvers {
    BE(BE),
    BDF(BDF),
    RADAU(Radau),
    RK45(RK45),
    DOPRI(DormandPrince),
}

impl Solvers {
    pub fn new(name: &str) -> Solvers {
        match name {
            //    "BE" => Solvers::BE(BE::new()),
            "BDF" => Solvers::BDF(BDF::new()),
            "RADAU" => Solvers::RADAU(Radau::new(RadauOrder::Order5)),
            "RK45" => Solvers::RK45(RK45::new()),
            "DOPRI" => Solvers::DOPRI(DormandPrince::new()),
            _ => panic!("Unknown solver name"),
        }
    }
}

trait Solver {
    fn step(&mut self, t_bound: f64, status: &mut String, message: &mut Option<String>);
}

impl Solver for BDF {
    fn step(&mut self, t_bound: f64, status: &mut String, message: &mut Option<String>) {
        let t = self.t;
        if t == t_bound {
            self.t_old = Some(t);
            *status = "finished".to_string();
        } else {
            let (success, message_) = self._step_impl();
            if let Some(message_str) = message_ {
                *message = Some(message_str.to_string());
            } else {
                *message = None;
            }

            if !success {
                *status = "failed".to_string();
            } else {
                self.t_old = Some(t);
                *status = "running".to_string();
                if self.direction * (self.t - t_bound) >= 0.0 {
                    *status = "finished".to_string();
                }
            }
        }
    }
}
impl Solver for RK45 {
    fn step(&mut self, t_bound: f64, status: &mut String, _message: &mut Option<String>) {
        let t = self.t;
        if t == t_bound {
            *status = "finished".to_string();
        } else {
            let success = self._step_impl();

            if !success {
                *status = "failed".to_string();
            } else {
                *status = "running".to_string();
                if (self.t - t_bound) >= 0.0 {
                    *status = "finished".to_string();
                }
            }
        }
    }
}

impl Solver for DormandPrince {
    fn step(&mut self, t_bound: f64, status: &mut String, _message: &mut Option<String>) {
        let t = self.t;
        if t == t_bound {
            *status = "finished".to_string();
        } else {
            let success = self._step_impl();

            if !success {
                *status = "failed".to_string();
            } else {
                *status = "running".to_string();
                if (self.t - t_bound) >= 0.0 {
                    *status = "finished".to_string();
                }
            }
        }
    }
}

pub struct ODEsolver {
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    method: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    max_step: f64,
    rtol: f64,
    atol: f64,
    jac_sparsity: Option<DMatrix<f64>>,
    vectorized: bool,
    first_step: Option<f64>,
    status: String,
    solver_instance: Solvers,
    message: Option<String>,
    t_result: DVector<f64>,
    y_result: DMatrix<f64>,
}

impl ODEsolver {
    pub fn new_complex(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        method: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        jac_sparsity: Option<DMatrix<f64>>,
        vectorized: bool,
        first_step: Option<f64>,
    ) -> Self {
        if !COMPLEX.contains(&method.as_str()) {
            panic!("new_complex not implemented, please, use new_easy")
        }
        let solver_instance = Solvers::new(&method);
        ODEsolver {
            eq_system,
            values,
            arg,
            method,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,
            jac_sparsity,
            vectorized,
            first_step,
            status: "running".to_string(),
            solver_instance,
            message: None,
            t_result: DVector::zeros(1),
            y_result: DMatrix::zeros(1, 1),
        }
    }
    pub fn new_easy(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        method: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
    ) -> Self {
        if !EASY.contains(&method.as_str()) {
            panic!("new_easy not implemented, please, use new_complex")
        }
        let solver_instance = Solvers::new(&method);
        ODEsolver {
            eq_system,
            values,
            arg,
            method,
            t0,
            y0,
            t_bound,
            max_step,
            rtol: 0.0,
            atol: 0.0,
            jac_sparsity: None,
            vectorized: false,
            first_step: None,
            status: "running".to_string(),
            solver_instance,
            message: None,
            t_result: DVector::zeros(1),
            y_result: DMatrix::zeros(1, 1),
        }
    }
    pub fn generate(&mut self) {
        let mut jacobian_instance = Jacobian::new();
        jacobian_instance.generate_IVP_ODEsolver(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
        );
        let fun = jacobian_instance.lambdified_functions_IVP_DVector;
        let jac = jacobian_instance.function_jacobian_IVP_DMatrix;

        if self.method == "BDF" {
            let mut solver_instance = BDF::new();
            solver_instance.set_initial(
                fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                Some(jac),
                None,
                self.vectorized,
                self.first_step,
            );
            self.solver_instance = Solvers::BDF(solver_instance);
        }
        // RADAU
        else if self.method == "RADAU" {
            let mut solver_instance = Radau::new(RadauOrder::Order5);
            solver_instance.set_initial(
                self.eq_system.clone(),
                self.values.clone(),
                self.arg.clone(),
                self.rtol, // tolerance
                50,        // max_iterations
                self.first_step,
                self.t0,
                self.t_bound,
                self.y0.clone(),
            );
            solver_instance.newton.eq_generate();
            self.solver_instance = Solvers::RADAU(solver_instance);
        }
        // BDF
        else if self.method == "RK45" {
            let mut solver_instance = RK45::new();
            solver_instance.set_initial(fun, self.y0.clone(), self.t0, self.max_step)
        } else if self.method == "DOPRI" {
            let mut solver_instance = DormandPrince::new();
            solver_instance.set_initial(fun, self.y0.clone(), self.t0, self.max_step)
        }
        /*
        else if self.method == "BE" {
            let mut solver_instance = BE::new();
            solver_instance.set_initial(

            );
            self.solver_instance = Solvers::BE(solver_instance);
        }

        */
    }

    pub fn main_loop(&mut self) {
        let start = Instant::now();
        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<f64> = Vec::new();
        let mut _i: i64 = 0;

        while integr_status.is_none() {
            match &mut self.solver_instance {
                Solvers::BDF(bdf) => {
                    bdf.step(self.t_bound, &mut self.status, &mut self.message);
                }
                Solvers::RADAU(radau) => {
                    radau.step();
                    self.status = radau.status.clone();
                    self.message = radau.message.clone();
                }
                Solvers::RK45(rk45) => {
                    rk45.step(self.t_bound, &mut self.status, &mut self.message);
                }
                Solvers::DOPRI(dopri) => {
                    dopri.step(self.t_bound, &mut self.status, &mut self.message);
                }
                _ => panic!("Unknown Solver"),
            };

            _i += 1;
            if self.status == "finished" {
                integr_status = Some(0);
            } else if self.status == "failed" {
                integr_status = Some(-1);
                break;
            }

            match &self.solver_instance {
                Solvers::BDF(bdf) => {
                    let y_i = bdf.y.clone();
                    let t_i = bdf.t;
                    t.push(t_i);
                    y.push(y_i);
                }
                Solvers::RADAU(radau) => {
                    let y_i = radau.y.clone();
                    let t_i = radau.t;
                    t.push(t_i);
                    y.push(y_i);
                }
                Solvers::RK45(rk45) => {
                    let y_i = rk45.y.clone();
                    let t_i = rk45.t;
                    t.push(t_i);
                    y.push(y_i);
                }
                Solvers::DOPRI(dopri) => {
                    let y_i = dopri.y.clone();
                    let t_i = dopri.t;
                    t.push(t_i);
                    y.push(y_i);
                }
                _ => panic!("Unknown Solver"),
            }
        }

        let rows = y.len();
        let cols = y[0].len();
        let mut flat_vec: Vec<f64> = Vec::new();
        for vector in y.iter() {
            flat_vec.extend(vector.iter());
        }
        let y_res: DMatrix<f64> = DMatrix::from_vec(cols, rows, flat_vec).transpose();
        let t_res = DVector::from_vec(t);
        let duration = start.elapsed();
        println!("Program took {} milliseconds to run", duration.as_millis());

        self.t_result = t_res;
        self.y_result = y_res;
    }

    pub fn solve(&mut self) {
        self.generate();
        self.main_loop();
    }

    pub fn plot_result(&self) {
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.t_result.clone(),
            self.y_result.clone(),
        );
        println!("result plotted");
    }

    pub fn get_result(&self) -> (DVector<f64>, DMatrix<f64>) {
        (self.t_result.clone(), self.y_result.clone())
    }

    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        let current_dir = env::current_dir().expect("Failed to get current directory");
        let path = Path::new(&current_dir); //.join("f:\\RUST\\RustProjects_\\RustedSciThe3\\src\\numerical\\results\\");
        let file_name = format!("{}+{}.csv", self.arg, self.values.join("+"));
        let full_path = path.join(file_name);

        let mut wtr = Writer::from_path(full_path)?;

        // Write column titles
        wtr.write_record(&[&self.arg, "values"])?;

        // Write time column
        wtr.write_record(self.t_result.iter().map(|&x| x.to_string()))?;

        // Write y columns
        for (i, col) in self.y_result.column_iter().enumerate() {
            let col_name = format!("{}", &self.values[i]);
            wtr.write_record(&[
                &col_name,
                &col.iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ])?;
        }

        println!("result saved");
        wtr.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests_radau_api {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_radau_api_simple_linear_ode() {
        // Test: y' = -y, y(0) = 1
        // Exact solution: y(t) = exp(-t)
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;
        let first_step = Some(0.01);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        // Check that we got results
        assert!(t_result.len() > 0);
        assert!(y_result.nrows() > 0);

        // Check final value: y(1) ≈ exp(-1) ≈ 0.3679
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        assert_relative_eq!(final_y, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_radau_api_exponential_growth() {
        // Test: y' = y, y(0) = 1
        // Exact solution: y(t) = exp(t)
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;
        let first_step = Some(0.01);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        // Check final value: y(0.5) ≈ exp(0.5) ≈ 1.6487
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        let expected = (0.5_f64).exp();
        assert_relative_eq!(final_y, expected, epsilon = 1e-2);

        // Verify solution against exact solution at multiple points
        let f_exact = |t: f64| t.exp();
        for (t, y_row) in t_result.iter().zip(y_result.row_iter()) {
            assert_relative_eq!(y_row[0], f_exact(*t), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_api_linear_system_2x2() {
        // Test system: y1' = -2*y1 + y2, y2' = y1 - 2*y2
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = 1/2 * e^(-3t) * (e^(2t) + 1)
        //                y2(t) = 1/2 * e^(-3t) * (-1 + e^(2t))
        let eq1 = Expr::parse_expression("-2*y1+y2");
        let eq2 = Expr::parse_expression("y1-2*y2");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = 1.0;
        let max_step = 0.001;
        let rtol = 1e-6;
        let atol = 1e-8;
        let first_step = Some(0.001);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        // Check that we have 2 variables
        assert_eq!(y_result.ncols(), 2);

        // Check final values against exact solution
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        let final_y2 = y_result[(y_result.nrows() - 1, 1)];

        let y1_exact = 0.5 * f64::exp(-3.0) * (f64::exp(2.0) + 1.0);
        let y2_exact = 0.5 * f64::exp(-3.0) * (-1.0 + f64::exp(2.0));

        assert_relative_eq!(final_y1, y1_exact, epsilon = 1e-2);
        assert_relative_eq!(final_y2, y2_exact, epsilon = 1e-2);

        // Verify exact solution throughout integration
        let f_y1 = |t: f64| 0.5 * f64::exp(-3.0 * t) * (f64::exp(2.0 * t) + 1.0);
        let f_y2 = |t: f64| 0.5 * f64::exp(-3.0 * t) * (-1.0 + f64::exp(2.0 * t));

        for (t, y_row) in t_result.iter().zip(y_result.row_iter()) {
            let y1_exact = f_y1(*t);
            let y2_exact = f_y2(*t);
            assert_relative_eq!(y_row[0], y1_exact, epsilon = 1e-2);
            assert_relative_eq!(y_row[1], y2_exact, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_api_harmonic_oscillator() {
        // Test: y1' = y2, y2' = -y1 (harmonic oscillator)
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = cos(t), y2(t) = -sin(t)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI / 2.0; // π/2
        let max_step = 0.01;
        let rtol = 1e-8;
        let atol = 1e-10;
        let first_step = Some(0.01);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        // At t = π/2: y1 should be ≈ 0, y2 should be ≈ -1
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        let final_y2 = y_result[(y_result.nrows() - 1, 1)];

        assert_relative_eq!(final_y1, 0.0, epsilon = 1e-2);
        assert_relative_eq!(final_y2, -1.0, epsilon = 1e-2);

        // Compare with exact solution throughout integration
        let f_y1 = |t: f64| f64::cos(t);
        let f_y2 = |t: f64| -f64::sin(t);

        for (t, y_row) in t_result.iter().zip(y_result.row_iter()) {
            let y1_exact = f_y1(*t);
            let y2_exact = f_y2(*t);
            assert_relative_eq!(y_row[0], y1_exact, epsilon = 1e-2);
            assert_relative_eq!(y_row[1], y2_exact, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_api_nonlinear_ode() {
        // Test: y' = y^2, y(0) = 1
        // Exact solution: y(t) = 1/(1-t) for t < 1
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;
        let first_step = Some(0.01);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        // At t = 0.5: y(0.5) = 1/(1-0.5) = 2
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        let expected = 1.0 / (1.0 - 0.5);
        assert_relative_eq!(final_y, expected, epsilon = 1e-2);

        // Verify exact solution throughout integration
        let f_exact = |t: f64| 1.0 / (1.0 - t);
        for (t, y_row) in t_result.iter().zip(y_result.row_iter()) {
            assert_relative_eq!(y_row[0], f_exact(*t), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_api_van_der_pol_oscillator() {
        // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1^2)y2 - y1
        // With μ = 0.1 (weakly nonlinear)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("0.1*(1.0 - y1*y1)*y2 - y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![2.0, 0.0]); // Start away from equilibrium
        let t_bound = 5.0;
        let max_step = 0.1;
        let rtol = 1e-6;
        let atol = 1e-8;
        let first_step = Some(0.1);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (_, y_result) = solver.get_result();

        // Check that solution remains bounded (Van der Pol has limit cycle)
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        let final_y2 = y_result[(y_result.nrows() - 1, 1)];

        assert!(final_y1.abs() < 10.0); // Should remain bounded
        assert!(final_y2.abs() < 10.0);
    }

    #[test]
    fn test_radau_api_stiff_system() {
        // Stiff system: y1' = -1000*y1 + y2, y2' = y1 - y2
        // This tests Radau's ability to handle stiff problems
        let eq1 = Expr::parse_expression("-1000.0*y1 + y2");
        let eq2 = Expr::parse_expression("y1 - y2");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 1.0]);
        let t_bound = 0.1;
        let max_step = 0.01; // Small step size for stiff problem
        let rtol = 1e-6;
        let atol = 1e-8;
        let first_step = Some(0.01);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (_, y_result) = solver.get_result();

        // For stiff systems, just check that solution doesn't blow up
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        let final_y2 = y_result[(y_result.nrows() - 1, 1)];

        assert!(final_y1.is_finite());
        assert!(final_y2.is_finite());
        assert!(final_y1.abs() < 100.0); // Should remain reasonable
        assert!(final_y2.abs() < 100.0);
    }

    #[test]
    fn test_radau_api_three_body_problem_simplified() {
        // Simplified 3-body problem (2D, one body)
        // y1' = y3, y2' = y4, y3' = -y1/r^3, y4' = -y2/r^3
        // where r = sqrt(y1^2 + y2^2)
        let eq1 = Expr::parse_expression("y3");
        let eq2 = Expr::parse_expression("y4");
        let eq3 = Expr::parse_expression("-y1/((y1*y1 + y2*y2)^1.5)");
        let eq4 = Expr::parse_expression("-y2/((y1*y1 + y2*y2)^1.5)");
        let eq_system = vec![eq1, eq2, eq3, eq4];
        let values = vec![
            "y1".to_string(),
            "y2".to_string(),
            "y3".to_string(),
            "y4".to_string(),
        ];
        let arg = "t".to_string();
        let method = "RADAU".to_string();
        let t0 = 0.0;
        // Initial conditions for circular orbit
        let y0 = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
        let t_bound = 1.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;
        let first_step = Some(0.01);

        let mut solver = ODEsolver::new_complex(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            first_step,
        );

        solver.solve();
        let (_, y_result) = solver.get_result();

        // Check that solution remains bounded and physical
        let final_position_x = y_result[(y_result.nrows() - 1, 0)];
        let final_position_y = y_result[(y_result.nrows() - 1, 1)];
        let final_velocity_x = y_result[(y_result.nrows() - 1, 2)];
        let final_velocity_y = y_result[(y_result.nrows() - 1, 3)];

        // All values should be finite and bounded
        assert!(final_position_x.is_finite());
        assert!(final_position_y.is_finite());
        assert!(final_velocity_x.is_finite());
        assert!(final_velocity_y.is_finite());

        // Check that we're still in a reasonable orbital region
        let distance =
            (final_position_x * final_position_x + final_position_y * final_position_y).sqrt();
        assert!(distance > 0.1 && distance < 10.0);
    }
}
