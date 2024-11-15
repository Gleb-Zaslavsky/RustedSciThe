use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use crate::numerical::BDF::BDF_solver::BDF;
extern crate nalgebra as na;
use na::{DMatrix, DVector};
use crate::numerical::BDF::common::NumberOrVec;
use crate::numerical::NonStiff_api::{RK45, DormandPrince}; 
use crate::Utils::plots::plots;
use std::env;
use std::path::Path;
use crate::numerical::BE::BE;
use std::time::Instant;
use csv::Writer;

static COMPLEX: [&str; 1] = ["BDF"];

const EASY: [&str; 2] = ["RK45", "DOPRI"];
pub enum Solvers {
    BE(BE),
    BDF(BDF),
    RK45(RK45),
    DOPRI(DormandPrince),
}

impl Solvers {
    pub fn new(name: &str) -> Solvers {
        match name {
        //    "BE" => Solvers::BE(BE::new()),
            "BDF" => Solvers::BDF(BDF::new()),
            "RK45" => Solvers::RK45(RK45::new()),
            "DOPRI"=> Solvers::DOPRI(DormandPrince::new()),
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
        if !COMPLEX.contains(&method.as_str()){
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
    pub fn new_easy ( 
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        method: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64) -> Self {
            if !EASY.contains(&method.as_str()){
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
                vectorized:false,
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
        jacobian_instance.generate_IVP_ODEsolver(self.eq_system.clone(), self.values.clone(), self.arg.clone());
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
        }// BDF
        else if self.method == "RK45" {
            let mut solver_instance = RK45::new();
            solver_instance.set_initial(
                fun, self.y0.clone(), self.t0, self.max_step)
            }
        else if self.method=="DOPRI"{

            let mut solver_instance = DormandPrince::new();
            solver_instance.set_initial(
                fun, self.y0.clone(), self.t0, self.max_step)
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
        plots(self.arg.clone(), self.values.clone(), self.t_result.clone(), self.y_result.clone());
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
            wtr.write_record(&[&col_name, &col.iter().map(|&x| x.to_string()).collect::<Vec<_>>().join(",")])?;
        }

        println!("result saved");
        wtr.flush()?;
        Ok(())
    }
}
