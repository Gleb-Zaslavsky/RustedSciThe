use crate::numerical::BDF::BDF_solver::BDF;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
extern crate nalgebra as na;
use crate::Utils::plots::plots;
use crate::numerical::BDF::common::NumberOrVec;
use na::{DMatrix, DVector};

use csv::Writer;
use std::collections::HashMap;
use std::time::Instant;

pub enum Solvers {
    //  BE(BE),
    BDF(BDF),
}
impl Solvers {
    pub fn new(&self) -> Solvers {
        match self {
            //   BE => Solvers::BE(BE::new()),
            _BDF => Solvers::BDF(BDF::new()),
        }
    }
    pub fn by_name(&self, name: &str) -> Solvers {
        match name {
            //  "BE" => Solvers::BE(BE::new()),
            "BDF" => Solvers::BDF(BDF::new()),
            _ => panic!("Unknown solver name"),
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
    #[allow(dead_code)]
    jac_sparsity: Option<DMatrix<f64>>,
    vectorized: bool,
    first_step: Option<f64>,

    status: String,
    Solver_instance: BDF,
    message: Option<String>,

    t_result: DVector<f64>,
    y_result: DMatrix<f64>,
    stop_condition: Option<HashMap<String, f64>>,
}
impl ODEsolver {
    pub fn new(
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
        let New = BDF::new();

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
            Solver_instance: New,
            message: None,

            t_result: DVector::zeros(1),
            y_result: DMatrix::zeros(1, 1),
            stop_condition: None,
        }
    } //new
    
    pub fn set_stop_condition(&mut self, stop_condition: HashMap<String, f64>) {
        self.stop_condition = Some(stop_condition);
    }
    
    fn check_stop_condition(&self, y: &DVector<f64>) -> bool {
        if let Some(ref conditions) = self.stop_condition {
            for (var_name, target_value) in conditions {
                if let Some(var_index) = self.values.iter().position(|v| v == var_name) {
                    let current_value = y[var_index];
                    if (current_value - target_value).abs() <= self.atol {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn generate(&mut self) {
        let mut Jacobian_instance = Jacobian::new();
        Jacobian_instance.generate_IVP_ODEsolver(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
        );
        let fun = Jacobian_instance.lambdified_functions_IVP_DVector;
        let jac = Jacobian_instance.function_jacobian_IVP_DMatrix;
        if self.method == "BDF" {
            let mut Solver_instance = BDF::new();
            Solver_instance.set_initial(
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
            self.Solver_instance = Solver_instance;
        }
    }
    pub fn step(&mut self) {
        //  let (success, message_) =self.Solver_instance._step_impl();

        // Analogue of step function in https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/base.py
        let t = self.Solver_instance.t;
        if t == self.t_bound {
            self.Solver_instance.t_old = Some(t);

            self.status = "finished".to_string();
        } else {
            let (success, message_) = self.Solver_instance._step_impl();
            if let Some(message_str) = message_ {
                self.message = Some(message_str.to_string());
            } else {
                self.message = None;
            }

            if success == false {
                self.status = "failed".to_string();
            } else {
                self.Solver_instance.t_old = Some(t);
                let _status: String = "running".to_string();
                if self.Solver_instance.direction * (self.Solver_instance.t - self.t_bound) >= 0.0 {
                    self.status = "finished".to_string();
                }
            }
        }
    } //step
    #[warn(unused_assignments)]
    pub fn main_loop(&mut self) -> () {
        // Analogue of https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/ivp.py
        let start = Instant::now();
        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<f64> = Vec::new();
        let mut _i: i64 = 0;
        while integr_status.is_none() {
            self.step();
            let _status: i8 = 0;
            //   println!("\n iteration: {}", i);
            //if i == 100 {panic!()}
            _i += 1;
            if self.status == "finished".to_string() {
                integr_status = Some(0)
            } else if self.status == "failed".to_string() {
                integr_status = Some(-1);
                break;
            }
            // Check stop condition before storing solution
            if self.check_stop_condition(&self.Solver_instance.y) {
                self.status = "stopped_by_condition".to_string();
                integr_status = Some(0);
            }
            
            //  println!("i: {}, t: {}, y: {:?}, status: {}", i, self.Solver_instance.t, self.Solver_instance.y, status);
            t.push(self.Solver_instance.t);
            y.push(self.Solver_instance.y.clone());
            // println!("time  {:?}, len {}", t, t.len())
        }

        let rows = &y.len();
        let cols = &y[0].len();

        let mut flat_vec: Vec<f64> = Vec::new();
        for vector in y.iter() {
            flat_vec.extend(vector)
        }
        let y_res: DMatrix<f64> = DMatrix::from_vec(*cols, *rows, flat_vec).transpose();
        let t_res = DVector::from_vec(t);
        let duration = start.elapsed();
        println!("Program took {} milliseconds to run", duration.as_millis());
        // println!("time  {:?}, len {}", &t_res, t_res.len());
        //println!("y  {:?}, len {:?}", &y_res, y_res.shape());

        self.t_result = t_res.clone();
        self.y_result = y_res.clone();
    } //

    pub fn solve(&mut self) -> () {
        self.generate();
        self.main_loop();
    }

    pub fn plot_result(&self) -> () {
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
    
    pub fn get_status(&self) -> &String {
        &self.status
    }

    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!(
            "f:\\RUST\\RustProjects_\\RustedSciThe3\\src\\numerical\\results\\{}+{}.csv",
            self.arg,
            self.values.join("+")
        );
        let mut wtr = Writer::from_path(path)?;

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
        print!("result saved");
        wtr.flush()?;
        Ok(())
    }
}

/*
pub fn ODEsolver(
    eq_system:  Vec<Expr>,
    values:Vec<String>,
    arg: String,
    method: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    max_step: f64,
    rtol: f64,
    atol: f64,
   // jac: Option<Box<dyn Fn(f64, DVector<f64>) -> DMatrix<f64>>>,
    jac_sparsity: Option<DMatrix<f64>>,
    vectorized: bool,
    first_step: Option<f64>,


)->(){



    let (success, message) = Solver_instance._step_impl();

   fn step(mut BDF_instance:BDF, status:String) -> (Option<String>, String){
    // Analogue of step function in https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/base.py
    let t =  BDF_instance.t;
    if t == t_bound{
        BDF_instance.t_old =Some(t);
        let message = None;
        let status:String = "finished".to_string();

    }else{
    let (success, message) = BDF_instance._step_impl();

        if success == false{
            let status:String = "failed".to_string();
        }else{
            BDF_instance.t_old =Some(t);
            let status:String = "running".to_string();
            if BDF_instance.direction * (BDF_instance.t - t_bound as f64) >= 0.0{
                let status:String = "finished".to_string();
            }
        }
       ( message, status)

    }






    }

}




}

//let mut BDF_instance = BDF::_step_impl(&mut self)
//BDF.set_initial(self, fun, t0, y0, t_bound, max_step, rtol, atol, jac, jac_sparsity, vectorized, first_step);


//let mut BDF_instance = BDF.set_initial(self, fun, t0, y0, t_bound, max_step, rtol, atol, jac, jac_sparsity, vectorized, first_step);
*/
#[cfg(test)]
mod tests_stop_conditions {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    #[test]
    fn test_bdf_stop_condition_single_variable() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0; // Large bound to ensure stop condition triggers first
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3; // This will be used as tolerance for stop condition
        let jac_sparsity = None;
        let vectorized = false;
        let first_step = None;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol,
            jac_sparsity, vectorized, first_step,
        );
        
        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);
        
        solver.solve();
        
        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= atol);
    }

    #[test]
    fn test_bdf_stop_condition_multiple_variables() {
        // Test system: y1' = y2, y2' = -y1, stop when y1 reaches 0.0
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;
        let jac_sparsity = None;
        let vectorized = false;
        let first_step = None;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol,
            jac_sparsity, vectorized, first_step,
        );
        
        let mut stop_condition = HashMap::new();
        stop_condition.insert("y1".to_string(), 0.0);
        solver.set_stop_condition(stop_condition);
        
        solver.solve();
        
        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        assert!(final_y1.abs() <= atol);
    }

    #[test]
    fn test_bdf_no_stop_condition() {
        // Test without stop condition - should run to t_bound
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let max_step = 0.1;
        let rtol = 1e-6;
        let atol = 1e-6;
        let jac_sparsity = None;
        let vectorized = false;
        let first_step = None;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol,
            jac_sparsity, vectorized, first_step,
        );
        
        solver.solve();
        
        assert_eq!(solver.get_status(), "finished");
        let (t_result, _) = solver.get_result();
        let final_t = t_result[t_result.len() - 1];
        assert!((final_t - t_bound).abs() <= max_step);
    }

    #[test]
    fn test_bdf_stop_condition_nonlinear() {
        // Test: y' = y^2, y(0) = 1, stop when y reaches 1.5
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;
        let jac_sparsity = None;
        let vectorized = false;
        let first_step = None;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol,
            jac_sparsity, vectorized, first_step,
        );
        
        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 1.5);
        solver.set_stop_condition(stop_condition);
        
        solver.solve();
        
        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        assert!((final_y - 1.5).abs() <= atol);
    }
}