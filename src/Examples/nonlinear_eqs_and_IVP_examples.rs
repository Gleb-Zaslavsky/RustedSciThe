// Copyright (c)  by Gleb E. Zaslavkiy
//MIT License
#![allow(non_snake_case)]

use crate::symbolic::symbolic_engine::Expr;

use crate::numerical::BE::BE;
use crate::numerical::ODE_api2::{SolverType, UniversalODESolver};
use nalgebra::DVector;
#[allow(dead_code)]
pub fn ivp_examples(example: usize) {
    match example {
        0 => {
            /*
            //use the shortest way to solve system of equations
            // first define system of equations and initial guess
            let mut NR_instanse = NR::new();
            let vec_of_expressions = vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()];
            let initial_guess = vec![1.0, 1.0];
            // solve
            NR_instanse.eq_generate_from_str(
                vec_of_expressions,
                None,
                initial_guess,
                1e-6,
                100,
                None,
            );
            NR_instanse.main_loop();
            println!("result = {:?} \n", NR_instanse.get_result().unwrap());
            // or more verbose way...
            // first define system of equations

            let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];

            let initial_guess = vec![1.0, 1.0];
            let mut NR_instanse = NR::new();
            let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
            let values = vec!["x".to_string(), "y".to_string()];
            NR_instanse.set_equation_system(
                vec_of_expr,
                Some(values.clone()),
                initial_guess,
                1e-6,
                100,
            );
            NR_instanse.eq_generate();
            NR_instanse.solver();

            println!("result = {:?} \n", NR_instanse.get_result().unwrap());
             */
        }

        1 => {
            //create instance of structure for symbolic equation system and Jacobian

            // define argument and unknown variables

            let y = Expr::Var("y".to_string());
            let z: Expr = Expr::Var("z".to_string());
            //define equation system
            let eq1: Expr =
                Expr::Const(-1.0f64) * z.clone() + (Expr::Const(-1.0) * y.clone()).exp();
            let eq2: Expr = y;
            let eq_system = vec![eq1, eq2];
            // set unkown variables
            let values = vec!["z".to_string(), "y".to_string()];
            // set argument
            let arg = "x".to_string();
            // set method
            // set initial conditions
            let t0 = 0.0;
            let y0 = vec![1.0, 1.0];
            let t_bound = 1.0;
            // set solver parameters (optional)
            let atol = 1e-5;
            let rtol = 1e-5;
            let max_step = 1e-3;
            let mut solver = UniversalODESolver::new(
                eq_system,
                values,
                arg,
                SolverType::Bdf,
                t0,
                DVector::from_vec(y0),
                t_bound,
            );
            solver.set_max_step(max_step);
            solver.set_rtol(rtol);
            solver.set_atol(atol);
            solver
                .try_solve()
                .expect("ODE_api2 BDF solve should succeed");
            solver.plot_result();
            let _ = solver.save_result();
        }
        2 => {
            //Example 2 the laziest way to solve ODE
            // set RHS of system as vector of strings
            let RHS = vec!["-z-exp(-y)", "y"];
            // parse RHS as symbolic expressions
            let Equations = Expr::parse_vector_expression(RHS.clone());
            let values = vec!["z".to_string(), "y".to_string()];
            println!("Equations = {:?}", Equations);
            // set argument
            let arg = "x".to_string();
            // set method
            // set initial conditions
            let t0 = 0.0;
            let y0 = vec![1.0, 1.0];
            let t_bound = 1.0;
            // set solver parameters (optional)
            let atol = 1e-5;
            let rtol = 1e-5;
            let max_step = 1e-3;
            let mut solver = UniversalODESolver::new(
                Equations,
                values,
                arg,
                SolverType::Bdf,
                t0,
                DVector::from_vec(y0),
                t_bound,
            );
            solver.set_max_step(max_step);
            solver.set_rtol(rtol);
            solver.set_atol(atol);
            solver
                .try_solve()
                .expect("ODE_api2 BDF solve should succeed");
            solver.plot_result();
        }

        3 => {
            println!(
                "This numbered demo was retired with the algebraic NR_for_ODE module. Use Nonlinear_systems for nonlinear equations."
            );
        }

        7 => {
            let rhs = Expr::parse_vector_expression(vec!["-z-y", "y"]);
            let mut solver = UniversalODESolver::new(
                rhs,
                vec!["z".to_string(), "y".to_string()],
                "x".to_string(),
                SolverType::NonStiff(crate::numerical::ODE_api2::NonStiffMethod::Dopri),
                0.0,
                DVector::from_vec(vec![1.0, 1.0]),
                1.0,
            );
            solver.set_max_step(1e-3);
            solver
                .try_solve()
                .expect("ODE_api2 DOPRI solve should succeed");
            solver.plot_result();
        }

        4 => {
            //  Backward Euler method: linear ODE
            let eq1 = Expr::parse_expression("z+y");
            let eq2 = Expr::parse_expression("z");
            let eq_system = vec![eq1, eq2];
            println!("eq_system = {:?}", eq_system);
            let y0 = DVector::from_vec(vec![1.0, 1.0]);
            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let tolerance = 1e-2;
            let max_iterations = 100;

            let h = Some(1e-3);
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
            println!(
                "y = {:?}, initial_guess = {:?}",
                solver.newton.y, solver.newton.initial_guess
            );
            solver.newton.eq_generate();
            solver.solve();
            let result = solver.get_result();
            println!("\n result 1 = {:?}", result.1);
            println!("result = {:?}", result.1.unwrap().shape());
            solver.plot_result();
            //   println!("result = {:?}", result);
        }
        5 => {
            //  Backward Euler method: slightly non-linear ODE
            let RHS = vec!["-z-exp(-y)", "y"];
            // parse RHS as symbolic expressions
            let Equations = Expr::parse_vector_expression(RHS.clone());
            let values = vec!["z".to_string(), "y".to_string()];
            println!("eq_system = {:?}", Equations);
            let y0 = DVector::from_vec(vec![1.0, 1.0]);

            let arg = "x".to_string();
            let tolerance = 1e-2;
            let max_iterations = 500;

            let h = Some(1e-3);
            let t0 = 0.0;
            let t_bound = 1.0;

            let mut solver = BE::new();

            solver.set_initial(
                Equations,
                values,
                arg,
                tolerance,
                max_iterations,
                h,
                t0,
                t_bound,
                y0,
            );
            println!(
                "y = {:?}, initial_guess = {:?}",
                solver.newton.y, solver.newton.initial_guess
            );
            solver.newton.eq_generate();
            solver.solve();
            #[allow(unused_variables)]
            let result = solver.get_result();
            //  println!("\n result 0 = {:?}", result.0);
            // println!("\n result 1 = {:?}", result.1);
            // println!("result = {:?}", result.1.unwrap().shape());
            solver.plot_result();
        }

        6 => {
            //  Backward Euler method: slightly non-linear ODE: adaptative time step
            let RHS = vec!["-z-exp(-y)", "y"];
            // parse RHS as symbolic expressions
            let Equations = Expr::parse_vector_expression(RHS.clone());
            let values = vec!["z".to_string(), "y".to_string()];
            println!("eq_system = {:?}", Equations);
            let y0 = DVector::from_vec(vec![1.0, 1.0]);

            let arg = "x".to_string();
            let tolerance = 1e-2;
            let max_iterations = 500;

            let h = None;
            let t0 = 0.0;
            let t_bound = 1.0;

            let mut solver = BE::new();

            solver.set_initial(
                Equations,
                values,
                arg,
                tolerance,
                max_iterations,
                h,
                t0,
                t_bound,
                y0,
            );
            println!(
                "y = {:?}, initial_guess = {:?}",
                solver.newton.y, solver.newton.initial_guess
            );
            solver.newton.eq_generate();
            solver.solve();
            #[allow(unused_variables)]
            let result = solver.get_result();
            //  println!("\n result 0 = {:?}", result.0);
            // println!("\n result 1 = {:?}", result.1);
            // println!("result = {:?}", result.1.unwrap().shape());
            solver.plot_result();
        }

        _ => {
            println!("example not found");
        }
    }
    //_________________________________________________
}
