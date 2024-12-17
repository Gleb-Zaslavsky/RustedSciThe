// Copyright (c)  by Gleb E. Zaslavkiy
//MIT License
#![allow(non_snake_case)]



use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;


use crate::numerical::NR_for_ODE::NRODE;
use crate::numerical::ODE_api::ODEsolver;
use crate::numerical::BE::BE;
use crate::numerical::NR::NR;
use nalgebra:: DVector;



fn ivp_examples(example: usize) {
    
    match example {
   
        0 => {
            //use the shortest way to solve system of equations
            // first define system of equations and initial guess
            let mut NR_instanse = NR::new();
            let vec_of_expressions = vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()];
            let initial_guess = vec![1.0, 1.0];
            // solve
            NR_instanse.eq_generate_from_str(vec_of_expressions, initial_guess, 1e-6, 100);
            NR_instanse.solve();
            println!("result = {:?} \n", NR_instanse.get_result().unwrap());
            // or more verbose way...
            // first define system of equations

            let vec_of_expressions = vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()];
            let mut Jacobian_instance = Jacobian::new();
            Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
            Jacobian_instance.set_variables(vec!["x", "y"]);
            Jacobian_instance.calc_jacobian();
            Jacobian_instance.jacobian_generate(vec!["x", "y"]);
            Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
            Jacobian_instance.readable_jacobian();
            println!(
                "Jacobian_instance: functions  {:?}. Variables {:?}",
                Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables
            );
            println!(
                "Jacobian_instance: Jacobian  {:?} readable {:?}. \n",
                Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian
            );
            let initial_guess = vec![1.0, 1.0];
            // in case you are interested in Jacobian value at initial guess
            Jacobian_instance.evaluate_func_jacobian_DMatrix(initial_guess.clone());
            Jacobian_instance.evaluate_funvector_lambdified_DVector(initial_guess.clone());
            let guess_jacobian = Jacobian_instance.evaluated_jacobian_DMatrix.clone();
            println!("guess Jacobian = {:?} \n", guess_jacobian.try_inverse());
            // defining NR method instance and solving
            let mut NR_instanse = NR::new();
            NR_instanse.set_equation_sysytem(Jacobian_instance, initial_guess, 1e-6, 100);
            NR_instanse.solve();
            println!("result = {:?} \n", NR_instanse.get_result().unwrap());
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
            let method = "BDF".to_string();
            // set initial conditions
            let t0 = 0.0;
            let y0 = vec![1.0, 1.0];
            let t_bound = 1.0;
            // set solver parameters (optional)
            let first_step = None;
            let atol = 1e-5;
            let rtol = 1e-5;
            let max_step = 1e-3;
            let jac_sparsity = None;
            let vectorized = false;
            // create instance of ODE solver and solve the system
            let mut ODE_instance = ODEsolver::new_complex(
                eq_system,
                values,
                arg,
                method,
                t0,
                y0.into(),
                t_bound,
                max_step,
                rtol,
                atol,
                jac_sparsity,
                vectorized,
                first_step,
            );

            ODE_instance.solve();
            // plot the solution
            ODE_instance.plot_result();
            let _ = ODE_instance.save_result();
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
            let method = "BDF".to_string();
            // set initial conditions
            let t0 = 0.0;
            let y0 = vec![1.0, 1.0];
            let t_bound = 1.0;
            // set solver parameters (optional)
            let first_step = None;
            let atol = 1e-5;
            let rtol = 1e-5;
            let max_step = 1e-3;
            let jac_sparsity = None;
            let vectorized = false;
            // create instance of ODE solver and solve the system
            let mut ODE_instance = ODEsolver::new_complex(
                Equations,
                values,
                arg,
                method,
                t0,
                y0.into(),
                t_bound,
                max_step,
                rtol,
                atol,
                jac_sparsity,
                vectorized,
                first_step,
            );

            ODE_instance.solve();
            ODE_instance.plot_result();
        }

        3 => {
            let eq1 = Expr::parse_expression("z^2+y^2-10.0*x");
            let eq2 = Expr::parse_expression("z-y-4.0*x");
            let eq_system = vec![eq1, eq2];
            println!("eq_system = {:?}", eq_system);
            let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
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
            nr.set_t(1.0);

            let _ = nr.solve().unwrap();
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

        7 => {
            //Non-stiff equations: use ODE general api ODEsolver
            // RK45 and Dormand-Prince methods are available

            let RHS = vec!["-z-y", "y"];
            // parse RHS as symbolic expressions
            let Equations = Expr::parse_vector_expression(RHS.clone());
            let values = vec!["z".to_string(), "y".to_string()];
            println!("Equations = {:?}", Equations);
            // set argument
            let arg = "x".to_string();
            // set method
            let method = "DOPRI".to_string();
            // set initial conditions
            let t0 = 0.0;
            let y0 = vec![1.0, 1.0];
            let t_bound = 1.0;
            // set solver parameters (optional)

            let max_step = 1e-3;

            // create instance of ODE solver and solve the system
            let mut ODE_instance = ODEsolver::new_easy(
                Equations,
                values,
                arg,
                method,
                t0,
                y0.into(),
                t_bound,
                max_step,
            );

            ODE_instance.solve();
            ODE_instance.plot_result();
        }



       
        _ => {
            println!("example not found");
        }
    }
    //_________________________________________________
}
