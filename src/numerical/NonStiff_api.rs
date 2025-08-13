use crate::Utils::plots::plots;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use csv::Writer;
use nalgebra::{DMatrix, DVector};
use std::env;
use std::path::Path;
use std::time::Instant;
pub struct nonstiffODE {
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    method: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    h_step: f64,
    solver_instance: Solvers,
    status: String,
    message: Option<String>,
    t_result: DVector<f64>,
    y_result: DMatrix<f64>,
}
pub enum Solvers {
    RK45(RK45),
    DOPRI(DormandPrince),
    AB4(AdamsBashforth4),
}
impl Solvers {
    pub fn new(name: &str) -> Solvers {
        match name {
            "RK45" => Solvers::RK45(RK45::new()),
            "DOPRI" => Solvers::DOPRI(DormandPrince::new()),
            "AB4" => Solvers::AB4(AdamsBashforth4::new()),
            _ => panic!("Unknown solver name"),
        }
    }
}

trait Solver {
    fn step(&mut self, t_bound: f64, status: &mut String, message: &mut Option<String>);
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

impl Solver for AdamsBashforth4 {
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

impl nonstiffODE {
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        method: String,
        // start point
        t0: f64,
        // initial condition
        y0: DVector<f64>,
        t_bound: f64,
        h_step: f64,
    ) -> Self {
        let solver_instance = Solvers::new(&method);
        nonstiffODE {
            eq_system,
            values,
            arg,
            method,
            t0,
            y0,
            t_bound,
            h_step,
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

        if self.method == "RK45" {
            let mut solver_instance = RK45::new();
            solver_instance.set_initial(fun, self.y0.clone(), self.t0, self.h_step);
            self.solver_instance = Solvers::RK45(solver_instance);
        } else if self.method == "DOPRI" {
            let mut solver_instance = DormandPrince::new();
            solver_instance.set_initial(fun, self.y0.clone(), self.t0, self.h_step);
            self.solver_instance = Solvers::DOPRI(solver_instance);
        } else if self.method == "AB4" {
            let mut solver_instance = AdamsBashforth4::new();
            solver_instance.set_initial(fun, self.y0.clone(), self.t0, self.h_step);
            self.solver_instance = Solvers::AB4(solver_instance);
        }
    }
    pub fn main_loop(&mut self) {
        let start = Instant::now();
        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<f64> = Vec::new();
        let mut _i: i64 = 0;

        while integr_status.is_none() {
            match &mut self.solver_instance {
                Solvers::RK45(rk45) => {
                    rk45.step(self.t_bound, &mut self.status, &mut self.message);
                }
                Solvers::DOPRI(dopri) => {
                    dopri.step(self.t_bound, &mut self.status, &mut self.message);
                }
                Solvers::AB4(ab4) => {
                    ab4.step(self.t_bound, &mut self.status, &mut self.message);
                }
            };

            _i += 1;
            if self.status == "finished" {
                integr_status = Some(0);
            } else if self.status == "failed" {
                integr_status = Some(-1);
                break;
            }

            match &self.solver_instance {
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
                Solvers::AB4(ab4) => {
                    let y_i = ab4.y.clone();
                    let t_i = ab4.t;
                    t.push(t_i);
                    y.push(y_i);
                }
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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
pub struct RK45 {
    f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    y0: DVector<f64>,
    t0: f64,
    #[allow(dead_code)]
    t_end: f64,
    pub t: f64,
    pub y: DVector<f64>,
    h: f64,
}
impl RK45 {
    pub fn new() -> RK45 {
        RK45 {
            f: Box::new(|_t, y| {
                let mut dydt = DVector::zeros(y.len());
                dydt[0] = y[1];
                dydt[1] = -y[0];
                dydt
            }),
            y0: DVector::from_vec(vec![1.0, 0.0]),
            t0: 0.0,
            t: 0.0,
            y: DVector::from_vec(vec![1.0, 0.0]),
            t_end: 10.0,
            h: 0.1,
        }
    } //new
    pub fn set_initial(
        &mut self,
        f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        y0: DVector<f64>,
        t0: f64,
        h: f64,
    ) {
        println!("step {}", h);
        self.f = f;
        self.y0 = y0.clone();
        self.t0 = t0;
        self.h = h;
        self.y = y0;
        self.t = t0;
    }

    pub fn _step_impl(&mut self) -> bool {
        // Butcher tableau coefficients for RK45
        let a: [[f64; 6]; 6] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
            [
                1932.0 / 2197.0,
                -7200.0 / 2197.0,
                7296.0 / 2197.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                439.0 / 216.0,
                -8.0,
                3680.0 / 513.0,
                -845.0 / 4104.0,
                0.0,
                0.0,
            ],
            [
                -8.0 / 27.0,
                2.0,
                -3544.0 / 2565.0,
                1859.0 / 4104.0,
                -11.0 / 40.0,
                0.0,
            ],
        ];
        let c = [0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0];
        let b = [
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        ];

        let mut t = self.t;
        let y = &self.y;
        let f = &self.f;
        let h = self.h;

        let mut k = vec![DVector::zeros(y.len()); 6];

        k[0] = h * f(t, &y);
        for i in 1..6 {
            let mut y_temp = y.clone();
            for j in 0..i {
                y_temp += a[i - 1][j] * &k[j];
            }
            k[i] = h * f(t + c[i], &y_temp);
        }

        let mut y_next = y.clone();
        for i in 0..6 {
            y_next += b[i] * &k[i];
        }

        t += h;
        self.t = t;
        self.y = y_next.clone();
        return true;
    }
}

pub struct DormandPrince {
    pub f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    pub y0: DVector<f64>,
    pub t0: f64,
    pub t: f64,
    pub y: DVector<f64>,
    h: f64,
}

impl DormandPrince {
    pub fn new() -> DormandPrince {
        DormandPrince {
            f: Box::new(|_t, y| {
                let mut dydt = DVector::zeros(y.len());
                dydt[0] = y[1];
                dydt[1] = -y[0];
                dydt
            }),
            y0: DVector::from_vec(vec![1.0, 0.0]),
            t0: 0.0,
            t: 0.0,
            y: DVector::from_vec(vec![1.0, 0.0]),

            h: 0.1,
        }
    }

    pub fn set_initial(
        &mut self,
        f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        y0: DVector<f64>,
        t0: f64,
        h: f64,
    ) {
        self.f = f;
        self.y0 = y0.clone();
        self.t0 = t0;
        self.h = h;
        self.y = y0;
        self.t = t0;
    }

    pub fn _step_impl(&mut self) -> bool {
        // Butcher tableau coefficients for Dormand-Prince
        let a: [[f64; 6]; 6] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 45.0, 32.0 / 45.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 6561.0,
                64448.0 / 6561.0,
                -212.0 / 6561.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -3556.0 / 3168.0,
                46732.0 / 3168.0,
                -4275.0 / 3168.0,
                2187.0 / 6561.0,
                0.0,
            ],
        ];
        let c = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
        let b = [
            19.0 / 216.0,
            0.0,
            16.0 / 216.0,
            512.0 / 1083.0,
            256.0 / 1083.0,
            -212.0 / 6561.0,
            1.0 / 8.0,
        ];

        let mut t = self.t;
        let y = &self.y;
        let f = &self.f;
        let h = self.h;

        let mut k = vec![DVector::zeros(y.len()); 6];

        k[0] = h * f(t, &y);
        for i in 1..6 {
            let mut y_temp = y.clone();
            for j in 0..i {
                y_temp += a[i - 1][j] * &k[j];
            }
            k[i] = h * f(t + c[i] * h, &y_temp);
        }

        let mut y_next = y.clone();
        for i in 0..6 {
            y_next += b[i] * &k[i];
        }

        t += h;
        self.t = t;
        self.y = y_next.clone();
        return true;
    }
}

pub struct AdamsBashforth4 {
    f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    y0: DVector<f64>,
    t0: f64,
    pub t: f64,
    pub y: DVector<f64>,
    h: f64,
    f_history: Vec<DVector<f64>>,
    step_count: usize,
}

impl AdamsBashforth4 {
    pub fn new() -> AdamsBashforth4 {
        AdamsBashforth4 {
            f: Box::new(|_t, y| {
                let mut dydt = DVector::zeros(y.len());
                dydt[0] = y[1];
                dydt[1] = -y[0];
                dydt
            }),
            y0: DVector::from_vec(vec![1.0, 0.0]),
            t0: 0.0,
            t: 0.0,
            y: DVector::from_vec(vec![1.0, 0.0]),
            h: 0.1,
            f_history: Vec::new(),
            step_count: 0,
        }
    }

    pub fn set_initial(
        &mut self,
        f: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        y0: DVector<f64>,
        t0: f64,
        h: f64,
    ) {
        self.f = f;
        self.y0 = y0.clone();
        self.t0 = t0;
        self.h = h;
        self.y = y0;
        self.t = t0;
        self.f_history.clear();
        self.step_count = 0;
    }

    pub fn _step_impl(&mut self) -> bool {
        if self.step_count < 4 {
            self._rk4_step();
        } else {
            let f_n = &self.f_history[3];
            let f_n1 = &self.f_history[2];
            let f_n2 = &self.f_history[1];
            let f_n3 = &self.f_history[0];

            let y_next = &self.y
                + self.h
                    * (55.0 / 24.0 * f_n - 59.0 / 24.0 * f_n1 + 37.0 / 24.0 * f_n2
                        - 9.0 / 24.0 * f_n3);

            self.t += self.h;
            self.y = y_next;
        }

        let f_current = (self.f)(self.t, &self.y);
        if self.f_history.len() >= 4 {
            self.f_history.remove(0);
        }
        self.f_history.push(f_current);
        self.step_count += 1;

        true
    }

    fn _rk4_step(&mut self) {
        let h = self.h;
        let t = self.t;
        let y = &self.y;
        let f = &self.f;

        let k1 = h * f(t, y);
        let k2 = h * f(t + h / 2.0, &(y + &k1 / 2.0));
        let k3 = h * f(t + h / 2.0, &(y + &k2 / 2.0));
        let k4 = h * f(t + h, &(y + &k3));

        self.y = y + (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4) / 6.0;
        self.t += h;
    }
}

////////////////////////////////////////////////////////////////////////////////////////
//          TESTS
///////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests_nonstiff_api {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_RK45_api_simple_linear_ode() {
        // Test: y' = -y, y(0) = 1
        // Exact solution: y(t) = exp(-t)
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "RK45".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;

        let step = 1e-4;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        println!("t_result: {:?}", t_result.shape());
        println!("y_result: {:?}", y_result.shape());
        // Check that we got results
        assert!(t_result.len() > 0);
        assert!(y_result.nrows() > 0);

        // Check final value: y(1) ≈ exp(-1) ≈ 0.3679
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        assert_relative_eq!(final_y, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_RK45_api_exponential_growth() {
        // Test: y' = y, y(0) = 1
        // Exact solution: y(t) = exp(t)
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "RK45".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let step = 0.01;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_RK45_api_linear_system_2x2() {
        // Test system: y1' = -2*y1 + y2, y2' = y1 - 2*y2
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = 1/2 * e^(-3t) * (e^(2t) + 1)
        //                y2(t) = 1/2 * e^(-3t) * (-1 + e^(2t))
        let eq1 = Expr::parse_expression("-2*y1+y2");
        let eq2 = Expr::parse_expression("y1-2*y2");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "RK45".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = 1.0;

        let step = 0.001;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        println!("t_result: {:?}", t_result.shape());
        println!("y_result: {:?}", y_result.shape());
        println!("{}", y_result[(y_result.nrows() - 1, 0)]);
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
    fn test_RK45_api_harmonic_oscillator() {
        // Test: y1' = y2, y2' = -y1 (harmonic oscillator)
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = cos(t), y2(t) = -sin(t)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "RK45".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI / 2.0; // π/2

        let step = 1e-3;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_RK45_api_nonlinear_ode() {
        // Test: y' = y^2, y(0) = 1
        // Exact solution: y(t) = 1/(1-t) for t < 1
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "RK45".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let step = 1e-3;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_RK45_api_van_der_pol_oscillator() {
        // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1^2)y2 - y1
        // With μ = 0.1 (weakly nonlinear)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("0.1*(1.0 - y1*y1)*y2 - y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "RK45".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![2.0, 0.0]); // Start away from equilibrium
        let t_bound = 5.0;

        let step = 0.1;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

        solver.solve();
        let (_, y_result) = solver.get_result();

        // Check that solution remains bounded (Van der Pol has limit cycle)
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        let final_y2 = y_result[(y_result.nrows() - 1, 1)];

        assert!(final_y1.abs() < 10.0); // Should remain bounded
        assert!(final_y2.abs() < 10.0);
    }
    ///////////////////////////////ab4 TESTS/////////////////////////////////////////////////////
    #[test]
    fn test_AB4_api_simple_linear_ode() {
        // Test: y' = -y, y(0) = 1
        // Exact solution: y(t) = exp(-t)
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "AB4".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;

        let step = 1e-4;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_AB4_api_exponential_growth() {
        // Test: y' = y, y(0) = 1
        // Exact solution: y(t) = exp(t)
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "AB4".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let step = 0.01;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_AB4_api_linear_system_2x2() {
        // Test system: y1' = -2*y1 + y2, y2' = y1 - 2*y2
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = 1/2 * e^(-3t) * (e^(2t) + 1)
        //                y2(t) = 1/2 * e^(-3t) * (-1 + e^(2t))
        let eq1 = Expr::parse_expression("-2*y1+y2");
        let eq2 = Expr::parse_expression("y1-2*y2");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "AB4".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = 1.0;

        let step = 0.001;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_AB4_api_harmonic_oscillator() {
        // Test: y1' = y2, y2' = -y1 (harmonic oscillator)
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = cos(t), y2(t) = -sin(t)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "AB4".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI / 2.0; // π/2

        let step = 1e-3;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_AB4_api_nonlinear_ode() {
        // Test: y' = y^2, y(0) = 1
        // Exact solution: y(t) = 1/(1-t) for t < 1
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "AB4".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let step = 1e-3;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

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
    fn test_ab4_api_van_der_pol_oscillator() {
        // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1^2)y2 - y1
        // With μ = 0.1 (weakly nonlinear)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("0.1*(1.0 - y1*y1)*y2 - y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "AB4".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![2.0, 0.0]); // Start away from equilibrium
        let t_bound = 5.0;

        let step = 0.1;

        let mut solver = nonstiffODE::new(eq_system, values, arg, method, t0, y0, t_bound, step);

        solver.solve();
        let (_, y_result) = solver.get_result();

        // Check that solution remains bounded (Van der Pol has limit cycle)
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        let final_y2 = y_result[(y_result.nrows() - 1, 1)];

        assert!(final_y1.abs() < 10.0); // Should remain bounded
        assert!(final_y2.abs() < 10.0);
    }
}
