use crate::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP as NRBDVPd;
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::any::Any;
use std::collections::HashMap;
//BVP is general api for all variants of BVP solvers

pub struct BVP {
    pub eq_system: Vec<Expr>, // the system of ODEs defined in the symbolic format
    pub initial_guess: DMatrix<f64>, // initial guess s - matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
    pub values: Vec<String>,         //unknown variables
    pub arg: String,                 // time or coordinate
    pub BorderConditions: HashMap<String, (usize, f64)>, // hashmap where keys are variable names and values are tuples with the index of the boundary condition (0 for inititial condition 1 for ending condition) and the value.
    pub t0: f64,                                         // initial value of argument
    pub t_end: f64,                                      // end of argument
    pub n_steps: usize,                                  //  number of  steps
    pub scheme: String,   // define crate using for matrices and vectors
    pub strategy: String, // name of the strategy
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>, // solver parameters
    pub linear_sys_method: Option<String>, // method for solving linear system
    pub method: String,   // define crate using for matrices and vectors
    pub tolerance: f64,   // abs_tolerance for NR_Damp_solver_damped
    pub max_iterations: usize, // maximum number of iterations
    // fields only for damped version of method
    pub rel_tolerance: Option<HashMap<String, f64>>, // absolute tolerance - hashmap of the var names and values of tolerance for them

    pub Bounds: Option<HashMap<String, (f64, f64)>>,
    pub loglevel: Option<String>, //
    pub structure: Option<NRBVP>,
    pub structure_damp: Option<NRBDVPd>,
}

impl BVP {
    pub fn from_hashmap(hashmap: &HashMap<String, &dyn Any>) -> BVP {
        let eq_system: Vec<Expr> = hashmap
            .get("eq_system")
            .and_then(|v| v.downcast_ref::<Vec<Expr>>())
            .cloned()
            .unwrap_or_default();
        let initial_guess: DMatrix<f64> = hashmap
            .get("initial_guess")
            .and_then(|v| v.downcast_ref::<DMatrix<f64>>())
            .cloned()
            .unwrap_or_default();
        let values: Vec<String> = hashmap
            .get("values")
            .and_then(|v| v.downcast_ref::<Vec<String>>())
            .cloned()
            .unwrap_or_default();
        let arg: String = hashmap
            .get("arg")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let BorderConditions: HashMap<String, (usize, f64)> = hashmap
            .get("BorderConditions")
            .and_then(|v| v.downcast_ref::<HashMap<String, (usize, f64)>>())
            .cloned()
            .unwrap_or_default();
        let t0: f64 = hashmap
            .get("t0")
            .and_then(|v| v.downcast_ref::<f64>())
            .cloned()
            .unwrap_or_default();
        let t_end: f64 = hashmap
            .get("t_end")
            .and_then(|v| v.downcast_ref::<f64>())
            .cloned()
            .unwrap_or_default();
        let n_steps: usize = hashmap
            .get("n_steps")
            .and_then(|v| v.downcast_ref::<usize>())
            .cloned()
            .unwrap_or_default();
        let scheme: String = hashmap
            .get("scheme")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let strategy: String = hashmap
            .get("strategy")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let strategy_params: Option<HashMap<String, Option<Vec<f64>>>> = hashmap
            .get("strategy_params")
            .and_then(|v| v.downcast_ref::<Option<HashMap<String, Option<Vec<f64>>>>>())
            .cloned()
            .unwrap_or_default();
        let linear_sys_method: Option<String> = hashmap
            .get("linear_sys_method")
            .and_then(|v| v.downcast_ref::<Option<String>>())
            .cloned()
            .unwrap_or_default();
        let method: String = hashmap
            .get("method")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let tolerance: f64 = hashmap
            .get("tolerance")
            .and_then(|v| v.downcast_ref::<f64>())
            .cloned()
            .unwrap_or_default();
        let max_iterations: usize = hashmap
            .get("max_iterations")
            .and_then(|v| v.downcast_ref::<usize>())
            .cloned()
            .unwrap_or_default();
        let rel_tolerance: Option<HashMap<String, f64>> = hashmap
            .get("rel_tolerance")
            .and_then(|v| v.downcast_ref::<Option<HashMap<String, f64>>>())
            .cloned()
            .unwrap_or_default();
        let Bounds: Option<HashMap<String, (f64, f64)>> = hashmap
            .get("Bounds")
            .and_then(|v| v.downcast_ref::<Option<HashMap<String, (f64, f64)>>>())
            .cloned()
            .unwrap_or_default();
        let loglevel: Option<String> = hashmap
            .get("loglevel")
            .and_then(|v| v.downcast_ref::<Option<String>>())
            .cloned()
            .unwrap_or_default();

        BVP {
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
            rel_tolerance,
            Bounds,
            loglevel,
            structure: None,
            structure_damp: None,
        }
    }
    pub fn new(
        eq_system: Vec<Expr>,        // the system of ODEs defined in the symbolic format
        initial_guess: DMatrix<f64>, // initial guess s - matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
        values: Vec<String>,         //unknown variables
        arg: String,                 // time or coordinate
        BorderConditions: HashMap<String, (usize, f64)>, // hashmap where keys are variable names and values are tuples with the index of the boundary condition (0 for inititial condition 1 for ending condition) and the value.
        t0: f64,                                         // initial value of argument
        t_end: f64,                                      // end of argument
        n_steps: usize,                                  //  number of  steps
        scheme: String,
        strategy: String, // name of the strategy
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>, // solver parameters
        linear_sys_method: Option<String>, // method for solving linear system
        method: String,   // define crate using for matrices and vectors
        tolerance: f64,   // abs_tolerance for NR_Damp_solver_damped
        max_iterations: usize, // maximum number of iterations
        // fields only for damped version of method
        rel_tolerance: Option<HashMap<String, f64>>, // absolute tolerance - hashmap of the var names and values of tolerance for them
        Bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>, //
    ) -> BVP {
        let (structure, structure_damp, rel_tolerance, Bounds) =
            if strategy == "Frozen" || strategy == "Naive" {
                let rel_tolerance: Option<HashMap<String, f64>> = None;
                let Bounds: Option<HashMap<String, (f64, f64)>> = None;
                let nrbvp = NRBVP::new(
                    eq_system.clone(),
                    initial_guess.clone(),
                    values.clone(),
                    arg.clone(),
                    BorderConditions.clone(),
                    t0,
                    t_end,
                    n_steps,
                    strategy.clone(),
                    strategy_params.clone(),
                    linear_sys_method.clone(),
                    method.clone(),
                    tolerance,
                    max_iterations,
                );
                let structure = Some(nrbvp);
                let structure_damp = None;
                (structure, structure_damp, rel_tolerance, Bounds)
            } else if strategy == "Damped" {
                let nrbvpd = NRBDVPd::new(
                    eq_system.clone(),
                    initial_guess.clone(),
                    values.clone(),
                    arg.clone(),
                    BorderConditions.clone(),
                    t0,
                    t_end,
                    n_steps,
                    scheme.clone(),
                    strategy.clone(),
                    strategy_params.clone(),
                    linear_sys_method.clone(),
                    method.clone(),
                    tolerance,
                    rel_tolerance.clone(),
                    max_iterations,
                    Bounds.clone(),
                    loglevel.clone(),
                );
                let structure = None;
                let structure_damp = Some(nrbvpd);

                (structure, structure_damp, rel_tolerance, Bounds)
            } else {
                panic!("Unknown strategy");
            };
        BVP {
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
            rel_tolerance,
            Bounds,
            loglevel,
            structure,
            structure_damp,
        }
    }
    pub fn plot_result(&self) {
        if let Some(ref structure) = self.structure {
            structure.plot_result();
        }
        if let Some(ref structure_damp) = self.structure_damp {
            structure_damp.plot_result();
        }
    }
    pub fn gnuplot_result(&self) {
        if let Some(ref structure) = self.structure {
            structure.plot_result();
        }
        if let Some(ref structure_damp) = self.structure_damp {
            structure_damp.gnuplot_result();
        }
    }
    pub fn solve(&mut self) {
        if let Some(ref mut structure) = self.structure {
            structure.solve();
        }
        if let Some(ref mut structure_damp) = self.structure_damp {
            structure_damp.solve();
        }
    }
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        if let Some(structure) = &self.structure {
            let res = structure.get_result();
            return res;
        }
        match &self.structure_damp {
            Some(structure_damp) => {
                let res = structure_damp.get_result();
                return res;
            }
            _ => {
                panic!("Invalid structure!");
            }
        }
    }
    pub fn save_to_file(&mut self, filename: Option<String>) {
        if let Some(structure) = &mut self.structure {
            structure.save_to_file();
        }
        if let Some(structure_damp) = &mut self.structure_damp {
            structure_damp.save_to_file(filename);
        }
    }
    pub fn save_to_csv(&mut self, filename: Option<String>) {
        if let Some(structure) = &mut self.structure {
            structure.save_to_file();
        }
        if let Some(structure_damp) = &mut self.structure_damp {
            structure_damp.save_to_csv(filename);
        }
    }
}
