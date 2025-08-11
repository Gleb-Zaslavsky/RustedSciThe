use crate::numerical::ShootingBVP::Shooting_simple::{BoundaryValueProblem, ShootingMethodSolver, rk4_ivp_solver,
    BoundaryCondition, BoundaryConditionType};

use crate::symbolic::symbolic_engine::Expr;
use std::collections::HashMap;

use nalgebra::DVector;


pub struct BVPShooting {
    pub eq_vec: Vec<Expr>,
    pub values: Vec<String>,
    pub arg: String,
    /// Map{name_of_variable vec![{index, value_of_bc}]} index =0/1 for left/right border 
    pub BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
    pub solver: ShootingMethodSolver,
    pub borders: (f64, f64),
    pub solution: DVector<f64>
}
impl BVPShooting {
    pub fn new(
        eq_vec: Vec<Expr>,
        values: Vec<String>,
        arg: String,
       BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
       borders: (f64, f64)
  
    ) -> Self {
        BVPShooting {
            eq_vec,
            values,
            arg,
            solver: ShootingMethodSolver::new(),
            BoundaryConditions:BoundaryConditions,
            borders: borders,
            solution: DVector::zeros(0)
        }
    }

    pub fn generate_eq_closure(&self) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> {
        let eq_vec = self.eq_vec.clone();
        let values = self.values.clone();
        let arg = self.arg.clone();
        let n = eq_vec.len();
        let residual = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let mut args = Vec::with_capacity(1 + n);
            args.push(t);
            args.extend(y.iter().cloned());
            let mut res = DVector::zeros(n);
            let mut all_var_names = vec![arg.as_str()];
            all_var_names.extend(values.iter().map(|s| s.as_str()));
            for i in 0..n {
                res[i] = eq_vec[i].eval_expression(all_var_names.clone(), &args);
            }
            res
        });
        residual
    }
    //
    
   pub fn BC_create(&self) -> (BoundaryCondition, BoundaryCondition) {
        let values = &self.values;
        let map_of_bc = &self.BoundaryConditions;

        let mut left_bc = BoundaryCondition { value: 0.0, bc_type: BoundaryConditionType::Dirichlet };
        let mut right_bc = BoundaryCondition { value: 0.0, bc_type: BoundaryConditionType::Dirichlet };
        
        for var_name in values {
            if let Some(conditions) = map_of_bc.get(var_name) {
                let var_idx = values.iter().position(|v| v == var_name).unwrap();
                for (boundary_idx, value) in conditions {
                    let bc_type: BoundaryConditionType = if var_idx ==0 {
                        BoundaryConditionType::Dirichlet
                    } else { BoundaryConditionType::Neumann};
                    
                    if *boundary_idx == 0 { 
                        // 0 means left boundary
                        left_bc = BoundaryCondition { value: *value, bc_type: bc_type };
                    } else {
                        // 1 means right boundary
                        right_bc = BoundaryCondition { value: *value,bc_type: bc_type };
                    }
                }
            }
        }
        (left_bc, right_bc)
    }
    /// simplest version of shooting method 
    /// - standard RK45 for IVP
    /// - symbolic 
    pub fn simple_solve(&mut self, 
        initial_guess: f64, 
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
    ) {
        let ode_system = self.generate_eq_closure();
        let (a, b) = self.borders;
        let (left_bc, right_bc) = self.BC_create();
        
        let problem = BoundaryValueProblem::new(ode_system, a, b, left_bc, right_bc);
        
        let solver = ShootingMethodSolver {
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
        };
        let (_, solution) = solver.solve(&problem, |x0, y0, x_end, h| {
            rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
        })
        .unwrap();
        self.solution = solution; 
    }
    
        pub fn simple_solve_generic<F>(&mut self, 
        initial_guess: f64, 
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        IVP_solver:F
    ) 
    where F: Fn(Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,DVector<f64>,f64, f64) ->DVector<f64> 
    {
        let ode_system = self.generate_eq_closure();
        let (a, b) = self.borders;
        let (left_bc, right_bc) = self.BC_create();
        
        let problem = BoundaryValueProblem::new(ode_system, a, b, left_bc, right_bc);
        
        let solver = ShootingMethodSolver {
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
        };
        
        let (_, solution) = solver.solve(&problem, |x0, y0, x_end, h| {
            rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
        })
        .unwrap();
        self.solution = solution; 
    }
    
}
