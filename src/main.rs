// Copyright (c)  by Gleb E. Zaslavkiy
//MIT License
#![allow(non_snake_case)]
use std::collections::HashMap;
pub mod Examples;
pub mod symbolic;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use crate::symbolic::symbolic_functions_BVP::Jacobian as JacobianBVP;
pub mod numerical;
use crate::numerical::BE::BE;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{NRBVP as NRBDVPd, SolverParams};
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP;
use crate::numerical::BVP_api::BVP;
use crate::numerical::Examples_and_utils::NonlinEquation;
use crate::numerical::NR_for_ODE::NRODE;
use crate::numerical::Nonlinear_systems::NR::NR;
use crate::numerical::ODE_api::ODEsolver;
pub mod global;

use nalgebra::{DMatrix, DVector};

pub mod Utils;
pub mod somelinalg;

fn main() {
    let example = 19;
    match example {
        0 => {
            // FUNCTION OF MULTIPLE VARIABLES
            //parse expression from string to symbolic expression
            let input = "exp(x)+log(y)"; //log(x)/y-x^2.3 *log(x+y+y^2.6)-exp(x-y)/(x+y) +  (log((x-y)/(x+y)))^2
            // here you've got symbolic expression
            let parsed_expression = Expr::parse_expression(input);
            println!(" parsed_expression {}", parsed_expression);
            // turn symbolic expression to a pretty human-readable string
            let parsed_function = parsed_expression.sym_to_str("x");
            println!("{}, sym to string: {}  \n", input, parsed_function);
            // return vec of all arguments
            let all = parsed_expression.all_arguments_are_variables();
            println!("all arguments are variables {:?}", all);
            let variables = parsed_expression.extract_variables();
            println!("variables {:?}", variables);

            // differentiate with respect to x and y
            let df_dx = parsed_expression.diff("x");
            let df_dy = parsed_expression.diff("y");
            println!("df_dx = {}, df_dy = {}", df_dx, df_dy);
            //convert symbolic expression to a Rust function and evaluate the function
            let args = vec!["x", "y"];
            let function_of_x_and_y =
                parsed_expression.lambdify_borrowed_thread_safe(args.as_slice());
            let f_res = function_of_x_and_y(&[1.0, 2.0]);
            println!("f_res = {}", f_res);
            // or you dont want to pass arguments you can use lambdify_wrapped, arguments will be found inside function
            let function_of_x_and_y = parsed_expression.lambdify_wrapped();
            let f_res = function_of_x_and_y(vec![1.0, 2.0]);
            println!("f_res2 = {}", f_res);

            let start = vec![1.0, 1.0];
            let end = vec![2.0, 2.0];
            // evaluate function of 2 or more arguments using linspace for defining vectors of arguments
            let result = parsed_expression.lamdified_from_linspace(start.clone(), end.clone(), 10);
            println!("evaluated function of 2 arguments = {:?}", result);
            //  find vector of derivatives with respect to all arguments
            let vector_of_derivatives = parsed_expression.diff_multi();
            println!(
                "vector_of_derivatives = {:?}, {}",
                vector_of_derivatives,
                vector_of_derivatives.len()
            );
            // compare numerical and analtical derivatives for a given linspace defined by start, end values and number of values.
            // max_norm - maximum norm of the difference between numerical and analtical derivatives
            let comparsion = parsed_expression.compare_num(start, end, 100, 1e-6);
            println!(" result_of compare = {:?}", comparsion);
        }
        1 => {
            //  FUNTION OF 1 VARIABLE (processing of them has a slightly easier syntax then for multiple variables)
            // function of 1 argument (1D examples)
            let input = "log(x)";
            let f = Expr::parse_expression(input);
            //convert symbolic expression to a Rust function and evaluate the function
            let f_res = f.lambdify1D()(1.0);
            let df_dx = f.diff("x");
            println!("df_dx = {}, log(1) = {}", df_dx, f_res);

            let input = "x+exp(x)";
            let f = Expr::parse_expression(input);
            let f_res = f.lambdify1D()(1.0);
            println!("f_res = {}", f_res);
            let start = 0.0;
            let end = 10f64;
            let num_values = 100;
            let max_norm = 1e-6;
            // compare numerical and analtical derivatives for a given linspace defined by start, end values and number of values.
            // a norm of the difference between the two of them is returned, and the answer is true if the norm is below max_norm
            let (norm, res) = f.compare_num1D("x", start, end, num_values, max_norm);
            println!("norm = {}, res = {}", norm, res);
        }
        3 => {
            // SOME USEFUL FEATURES
            // a symbolic function can be defined in a more straightforward way without parsing expression
            // first define symbolic variables
            let vector_of_symbolic_vars = Expr::Symbols("a, b, c");
            println!("vector_of_symbolic_vars = {:?}", vector_of_symbolic_vars);
            let (a, b, c) = (
                vector_of_symbolic_vars[0].clone(),
                // consruct symbolic expression
                vector_of_symbolic_vars[1].clone(),
                vector_of_symbolic_vars[2].clone(),
            );
            let symbolic_expression = a + Expr::exp(b * c);
            println!("symbolic_expression = {:?}", symbolic_expression);
            // if you want to change a variable inti constant:
            let expression_with_const = symbolic_expression.set_variable("a", 1.0);
            println!("expression_with_const = {:?}", expression_with_const);
            let parsed_function = expression_with_const.sym_to_str("a");
            println!("{}, sym to string:", parsed_function);
        }
        4 => {
            // JACOBIAN
            // instance of Jacobian structure
            let mut Jacobian_instance = Jacobian::new();
            // function of 2 or more arguments
            let vec_of_expressions = vec!["2*x^3+y".to_string(), "1.0".to_string()];
            // set vector of functions
            Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
            // set vector of variables
            //  Jacobian_instance.set_varvecor_from_str("x, y");
            Jacobian_instance.set_variables(vec!["x", "y"]);
            // calculate symbolic jacobian
            Jacobian_instance.calc_jacobian();
            // transform into human...kind of readable form
            Jacobian_instance.readable_jacobian();
            // generate jacobian made of regular rust functions
            Jacobian_instance.jacobian_generate(vec!["x", "y"]);

            println!(
                "Jacobian_instance: functions  {:?}. Variables {:?}",
                Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables
            );
            println!(
                "Jacobian_instance: Jacobian  {:?} readable {:?}.",
                Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian
            );
            for i in 0..Jacobian_instance.symbolic_jacobian.len() {
                for j in 0..Jacobian_instance.symbolic_jacobian[i].len() {
                    println!(
                        "Jacobian_instance: Jacobian  {} row  {} colomn {:?}",
                        i, j, Jacobian_instance.symbolic_jacobian[i][j]
                    );
                }
            }
            // calculate element of jacobian (just for control)
            let ij_element =
                Jacobian_instance.calc_ij_element(0, 0, vec!["x", "y"], vec![10.0, 2.0]);
            println!("ij_element = {:?} \n", ij_element);

            // or first lambdify
            Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);

            // evaluate jacobian to nalgebra matrix format
            Jacobian_instance.evaluate_func_jacobian_DMatrix(vec![10.0, 2.0]);
            println!(
                "Jacobian_DMatrix = {:?} \n",
                Jacobian_instance.evaluated_jacobian_DMatrix
            );
            // evaluate function vector to nalgebra matrix format
            Jacobian_instance.evaluate_funvector_lambdified_DVector(vec![10.0, 2.0]);
            println!(
                "function vector after evaluate_funvector_lambdified_DMatrix = {:?} \n",
                Jacobian_instance.evaluated_functions_DVector
            );
        }
        5 => {
            use crate::numerical::Nonlinear_systems::NR::Method;
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
            NR_instanse.solve();
            println!("result = {:?} \n", NR_instanse.get_result().unwrap());
            // or more verbose way...
            // first define system of equations
            let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];
            let initial_guess = vec![1.0, 1.0];
            // defining NR method instance and solving
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
            NR_instanse.solve();
            println!("result = {:?} \n", NR_instanse.get_result().unwrap());

            // equations with bounds
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
            let Bounds = HashMap::from([
                ("x".to_string(), (0.0, 10.0)),
                ("y".to_string(), (0.0, 10.0)),
            ]);
            NR_instanse.set_solver_params(
                Some("info".to_string()),
                None,
                None,
                Some(Bounds),
                None,
                None,
            );
            NR_instanse.eq_generate();
            NR_instanse.main_loop();
            let solution = NR_instanse.get_result().unwrap();
            println!("solution = {:?} \n", solution);

            // equations
            let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");
            let Boubds = HashMap::from([
                ("N0".to_string(), (0.0, 1.0)),
                ("N1".to_string(), (0.0, 1.0)),
                ("N2".to_string(), (0.0, 1.0)),
                ("Np".to_string(), (0.0, 10.0)),
                ("Lambda0".to_string(), (-100.0, 100.0)),
                ("Lambda1".to_string(), (-100.0, 100.0)),
            ]);
            let dG0 = Expr::Const(-450.0e3);
            let dG1 = Expr::Const(-150.0e3);
            let dG2 = Expr::Const(-50e3);
            let dGm0 = Expr::Const(8.314 * 450e5);
            let dGm1 = Expr::Const(8.314 * 150e5);
            let dGm2 = Expr::Const(8.314 * 50e5);
            let N0 = symbolic[0].clone();
            let N1 = symbolic[1].clone();
            let N2 = symbolic[2].clone();
            let Np = symbolic[3].clone();
            let Lambda0 = symbolic[4].clone();
            let Lambda1 = symbolic[5].clone();

            let RT = Expr::Const(8.314) * Expr::Const(273.15);
            let eq_mu = vec![
                Lambda0.clone()
                    + Expr::Const(2.0) * Lambda1.clone()
                    + (dG0.clone() + RT.clone() * Expr::ln(N0.clone() / Np.clone())) / dGm0.clone(),
                Lambda0
                    + Lambda1.clone()
                    + (dG1 + RT.clone() * Expr::ln(N1.clone() / Np.clone())) / dGm1.clone(),
                Expr::Const(2.0) * Lambda1
                    + (dG2 + RT * Expr::ln(N2.clone() / Np.clone())) / dGm2.clone(),
            ];
            let eq_sum_mole_numbers = vec![N0.clone() + N1.clone() + N2.clone() - Np.clone()];
            let composition_eq = vec![
                N0.clone() + N1.clone() - Expr::Const(0.999),
                Expr::Const(2.0) * N0.clone() + N1.clone() + Expr::Const(2.0) * N2
                    - Expr::Const(1.501),
            ];

            let mut full_system_sym = Vec::new();
            full_system_sym.extend(eq_mu.clone());
            full_system_sym.extend(eq_sum_mole_numbers.clone());
            full_system_sym.extend(composition_eq.clone());

            for eq in &full_system_sym {
                println!("eq: {}", eq);
            }
            // solver
            let initial_guess = vec![0.5, 0.5, 0.5, 1.0, 2.0, 2.0];
            let unknowns: Vec<String> = symbolic.iter().map(|x| x.to_string()).collect();
            let mut solver = NR::new();
            solver.set_equation_system(
                full_system_sym.clone(),
                Some(unknowns.clone()),
                initial_guess,
                1e-4,
                100,
            );
            solver.set_solver_params(
                Some("info".to_string()),
                None,
                Some(0.5),
                Some(Boubds),
                Some(Method::damped),
                None,
            );
            solver.eq_generate();

            solver.solve();
            let solution = solver.get_result().expect("Failed to get result");
            let solution: Vec<f64> = solution.data.into();
            let map_of_solutions: HashMap<String, f64> = unknowns
                .iter()
                .zip(solution.iter())
                .map(|(k, v)| (k.to_string(), *v))
                .collect();

            let map_of_solutions = map_of_solutions;
            let N0 = map_of_solutions.get("N0").unwrap();
            let N1 = map_of_solutions.get("N1").unwrap();
            let N2 = map_of_solutions.get("N2").unwrap();
            let Np = map_of_solutions.get("Np").unwrap();
            let _Lambda0 = map_of_solutions.get("Lambda0").unwrap();
            let _Lambda1 = map_of_solutions.get("Lambda1").unwrap();
            let d1 = *N0 + *N1 - 0.999;
            let d2 = N0 + N1 + N2 - Np;
            let d3 = 2.0 * N0 + N1 + 2.0 * N2 - 1.501;
            println!("d1: {}", d1);
            println!("d2: {}", d2);
            println!("d3: {}", d3);
            println!("map_of_solutions: {:?}", map_of_solutions);
            assert!(d1.abs() < 1e-3);
            assert!(d2.abs() < 1e-2);
            assert!(d3.abs() < 1e-2);
        }
        6 => {
            // INDEXED VARIABLES
            let (matrix_of_indexed, vec_of_names) = Expr::IndexedVars2D(1, 10, "x");
            println!("matrix_of_indexed = {:?} \n", matrix_of_indexed);
            println!("vec_of_names = {:?} \n", vec_of_names);
        }
        7 => {
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
        8 => {
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

        9 => {
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
        10 => {
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
        11 => {
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

        12 => {
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

        13 => {
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

        14 => {
            //BVP jacobian matrix demonstration
            let RHS = vec!["-z-y", "y"];
            // parse RHS as symbolic expressions
            let Equations = Expr::parse_vector_expression(RHS.clone());
            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let n_steps = 3;
            let scheme = "forward".to_string();
            let h = 1e-4;
            let BorderConditions = HashMap::from([
                ("z".to_string(), vec![(0, 1000.0)]),
                ("y".to_string(), vec![(1, 333.0)]),
            ]);

            let Y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            let mut Jacobian_instance = JacobianBVP::new();

            // Create BVP discretization system
            Jacobian_instance.discretization_system_BVP_par(
                Equations.clone(),
                values.clone(),
                arg.clone(),
                0.0,
                Some(n_steps),
                Some(h),
                None,
                BorderConditions.clone(),
                None,
                None,
                scheme.clone(),
            );

            // Calculate jacobian
            Jacobian_instance.calc_jacobian_parallel_smart();

            // Set up variables for lambdification
            let variable_strs: Vec<String> = Jacobian_instance.variable_string.clone();
            let variable_strs: Vec<&str> = variable_strs.iter().map(|s| s.as_str()).collect();

            // Test Dense matrix jacobian
            Jacobian_instance.lambdify_jacobian_DMatrix_par(&arg, variable_strs.clone());
            Jacobian_instance.lambdify_residual_DVector(&arg, variable_strs.clone());

            println!(
                "vector of variables {:?}",
                Jacobian_instance.variable_string
            );
            let Ys = DVector::from_vec(Y.clone());

            // Evaluate jacobian and residual using trait objects
            use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
            let variables_dense = &*Vectors_type_casting(&Ys, "Dense".to_string());

            if let Some(ref mut jac) = Jacobian_instance.jac_function {
                let J_eval = jac.call(4.0, variables_dense);
                println!("Jacobian Dense: J_eval = {:?} \n", J_eval.to_DMatrixType());
            }

            let F_eval = Jacobian_instance
                .residiual_function
                .call(4.0, variables_dense);
            println!("Residual Dense: F_eval = {:?} \n", F_eval.to_DVectorType());

            // Test Sparse matrix jacobian (faer crate)
            use faer::col::{Col, ColRef};
            let Ys3: Col<f64> = ColRef::from_slice(Y.as_slice()).to_owned();

            Jacobian_instance.lambdify_jacobian_SparseColMat_parallel2(&arg, variable_strs.clone());
            Jacobian_instance.lambdify_residual_Col_parallel2(&arg, variable_strs.clone());

            let variables_sparse = &*Vectors_type_casting(&Ys, "Sparse".to_string());

            if let Some(ref mut jac) = Jacobian_instance.jac_function {
                let J_eval_sparse = jac.call(4.0, variables_sparse);
                println!(
                    "Jacobian Sparse (faer): J_eval = {:?} \n",
                    J_eval_sparse.to_DMatrixType()
                );
            }

            let F_eval_sparse = Jacobian_instance
                .residiual_function
                .call(4.0, variables_sparse);
            println!(
                "Residual Sparse (faer): F_eval = {:?} \n",
                F_eval_sparse.to_DVectorType()
            );
        }
        15 => {
            let eq1 = Expr::parse_expression("y-z");
            let eq2 = Expr::parse_expression("-z^2");
            let eq_system = vec![eq1, eq2];

            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 50;

            let t0 = 0.0;
            let t_end = 1.0;
            let n_steps = 200;
            let strategy = "Naive".to_string(); //
            let strategy_params = None;
            let method = "Sparse".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.0; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let mut BorderConditions = HashMap::new();
            BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
            BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
            assert_eq!(&eq_system.len(), &2);
            let mut nr = NRBVP::new(
                eq_system,
                initial_guess,
                values,
                arg,
                BorderConditions,
                t0,
                t_end,
                n_steps,
                strategy,
                strategy_params,
                linear_sys_method,
                method,
                tolerance,
                max_iterations,
            );

            println!("solving system");
            #[allow(unused_variables)]
            let solution = nr.solve().unwrap();
            // println!("result = {:?}", solution);
            nr.plot_result();
        }
        16 => {
            /*
             frozen jacobian strategy - recalculating jacobian on condition:
             Description of strategy                                                                 key of strategy        value user must provude for strategy
            1. only first time:                                                                      "Frozen_naive"                   None
            2. every m-th time, where m is a parameter of the strategy:                                "every_m"                    m
            3. every time when the solution norm greater than a certain threshold A:                   "at_high_norm".               A
            4. when norm of (i-1) iter multiplied by certain value B(<1) is lower than norm of i-th iter : "at_low_speed".            B
            5. complex - combined strategies 2,3,4                                                         "complex"              vec of  parameters [m, A, B]

             */
            let eq1 = Expr::parse_expression("y-z");
            let eq2 = Expr::parse_expression("-z^2");
            let eq_system = vec![eq1, eq2];

            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 1500;

            let t0 = 0.0;
            let t_end = 1.0;
            let n_steps = 200; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
            let strategy = "Frozen".to_string(); //
            let strategy_params = Some(HashMap::from([(
                "complex".to_string(),
                Some(Vec::from([2f64, 5.0, 1e-1])),
            )]));
            /*
              or
              Some(HashMap::from([("Frozen_naive".to_string(), None)]));
              or
              Some(HashMap::from([("every_m".to_string(),
                    Some(Vec::from( [ 5 as f64]  ))
                    )]));
              or
              Some(HashMap::from([("at_high_morm".to_string(),
             Some(Vec::from( [ 5 as f64]  ))
            )]));
            or
            Some(HashMap::from([("at_low_speed".to_string(),
             Some(Vec::from( [ 1e-2]  ))
            )]));
            or
             Some(HashMap::from([("complex".to_string(),
             Some(Vec::from( [ 2.0, 5.0, 1e-, ]  ))
            )]));
              */
            let method = "Sparse".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.0; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let mut BorderConditions = HashMap::new();
            BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
            BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
            assert_eq!(&eq_system.len(), &2);
            let mut nr = NRBVP::new(
                eq_system,
                initial_guess,
                values,
                arg,
                BorderConditions,
                t0,
                t_end,
                n_steps,
                strategy,
                strategy_params,
                linear_sys_method,
                method,
                tolerance,
                max_iterations,
            );

            println!("solving system");
            #[allow(unused_variables)]
            let solution = nr.solve().unwrap();
            // println!("result = {:?}", solution);
            nr.plot_result();
        }
        17 => {
            let eq1 = Expr::parse_expression("y-z");
            let eq2 = Expr::parse_expression("-z^3");
            let eq_system = vec![eq1, eq2];

            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 20;

            let t0 = 0.0;
            let t_end = 1.0;
            let n_steps = 1600; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
            let scheme = "forward".to_string();
            let strategy = "Damped".to_string(); //
            let strategy_params = Some(SolverParams::default());

            let method = "Sparse".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.0; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let mut BorderConditions = HashMap::new();
            BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
            BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
            let Bounds = HashMap::from([
                ("z".to_string(), (-10.0, 10.0)),
                ("y".to_string(), (-7.0, 7.0)),
            ]);
            let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
            assert_eq!(&eq_system.len(), &2);
            let mut nr = NRBDVPd::new(
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
                Some(rel_tolerance),
                max_iterations,
                Some(Bounds),
                None,
            );

            println!("solving system");
            #[allow(unused_variables)]
            let solution = nr.solve().unwrap();
            // println!("result = {:?}", solution);
            nr.plot_result();
        }

        18 => {
            // enum collection of BVP problems for nonlinear equations with exact solution
            //EXAMPLES OF EXACT SOLUTION OF BVP OF NONLINEAR DIFFERENTIAL EQUATION
            /*
             the Lane-Emden equation of index 5:
            y′′+2xy′+y**5=0,y(0)=1,y′(0)=0
            y'=z
            z'=- 2*x*z - y**5
            With initial conditions y(0)=1,y′(0)=0
            exact solution:
            y = (1+(x^2)/3)^(-0.5)


            A very simple non-linear system to analyze is what I like to call the "Parachute Equation" which is essentially

            y''+ ky'^2−g=0(1)
            With initial conditions y(0)=0
             and y˙(0)=0
            exact solution:
            y= k*(log( (e2√g√kt+1)/2)−√g√kt)


            Clairaut's equation.
            y′′′=(x−1^2)2+y2+y′−2
            y(1)=1, y′(1)=0, y′′(1)=2
            exact solution:
            yp(x)=1+(x−1)**2−1/6(x−1)**3+1/12(x−1)**4+...

            two-point boundary value problem:
            y''= -(2.0/a)*(1+2.0*ln(y))*y
            y(-1) = exp(-1/a)
            y(1) = exp(-1/a)
            exact solution:
            y(x)=exp(-x^2/a)
            .
            */
            let ne = NonlinEquation::ParachuteEquation; //  Clairaut   LaneEmden5  ParachuteEquation  TwoPointBVP

            let eq_system = ne.setup();

            let values = ne.values();
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 200;
            let t0 = ne.span(None, None).0;
            let t_end = ne.span(None, None).1;
            let n_steps = 30; //
            let strategy = "Damped".to_string(); //
            let strategy_params = Some(SolverParams {
                max_jac: Some(100),
                max_damp_iter: Some(100),
                damp_factor: None,
                adaptive: None, // TODO: Add adaptive grid support if needed
            });
            let scheme = "forward".to_string();
            let method = "Sparse".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.5; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let BorderConditions = ne.boundary_conditions();
            let Bounds = ne.Bounds();
            let rel_tolerance = ne.rel_tolerance();

            let mut nr = NRBDVPd::new(
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
                Some(rel_tolerance),
                max_iterations,
                Some(Bounds),
                None,
            );

            println!("solving system");
            nr.solve();
            let solution = nr.get_result().unwrap();
            //  print!("result = {:?}\n", solution.column(1).iter().map(|x| *x).collect::<Vec<_>>());
            let y_numer = solution.column(0);
            let y_numer: Vec<f64> = y_numer.iter().map(|x| *x).collect();

            nr.plot_result();
            //compare with exact solution
            let y_exact = ne.exact_solution(None, None, Some(y_numer.len()));
            let n = &y_exact.len();
            // println!("numerical result = {:?}",  y_numer);
            println!("\n \n y exact{:?}, {}", &y_exact, &y_exact.len());
            println!("\n \n y numer{:?}, {}", &y_numer, &y_numer.len());
            let comparsion: Vec<f64> = y_numer
                .into_iter()
                .zip(y_exact.clone())
                .map(|(y_n_i, y_e_i)| (y_n_i - y_e_i).powf(2.0))
                .collect();
            let norm = comparsion.iter().sum::<f64>().sqrt() / (*n as f64);
            let max_residual = comparsion
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let position = comparsion.iter().position(|&x| x == *max_residual).unwrap();
            let relativ_residual = max_residual.abs() / y_exact[position];
            println!(
                "maximum relative residual of numerical solution wioth respect to exact solution = {}",
                relativ_residual
            );
            println!("norm = {}", norm);
            nr.save_to_file(None)

            //BVP is general api for all variants of BVP solvers
        }
        19 => {
            let eq1 = Expr::parse_expression("y-z");
            let eq2 = Expr::parse_expression("-z^3");
            let eq_system = vec![eq1, eq2];

            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 20;

            let t0 = 0.0;
            let t_end = 1.0;
            let n_steps = 1_000; //
            let strategy = "Damped".to_string(); // 

            let strategy_params = match strategy.as_str() {
                "Naive" => None,
                "Damped" => Some(HashMap::from([
                    ("max_jac".to_string(), None), // maximum iterations with old Jacobian, None means the default number is taken-3
                    ("maxDampIter".to_string(), None), // maximum number of damped steps, None means the default value of 5 is used
                    ("DampFacor".to_string(), None), // factor to decrease the damping coefficient, None means the default value of 0.5 is used
                    (
                        "adaptive".to_string(), // adaptive strategy parameters, None means no grid refinement is used, if Some - grid refinement is enabled
                        // first parameter means what criteria to choose is refinement needed or not in the current iteration, second parameter means
                        // maximum number of refinments allowed
                        Some(vec![1.0, 5.0]), //  None
                    ),
                    // the name of grid refinement strategy, this key-value pair will be used only if "adaptive" is Some, in opposite case this pair
                    // will be ignored: vector of parametrs is used inside the grid refinement algorithm
                    //  ("pearson".to_string(), Some(vec![0.2] ) ) (""two_point".to_string(), Some(vec![0.2, 0.5, 1.4])),
                    ("two_point".to_string(), Some(vec![0.2, 0.5, 1.4])),
                ])),
                "Frozen" => Some(HashMap::from([(
                    "every_m".to_string(),
                    Some(Vec::from([5 as f64])),
                )])),
                &_ => panic!("Invalid strategy!"),
            };

            let scheme = "trapezoid".to_string();
            let method = "Sparse".to_string(); //   "Sparse" or "Dense"
            let linear_sys_method = None;
            let ones = vec![0.0; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let mut BorderConditions = HashMap::new();
            BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
            BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
            let Bounds = HashMap::from([
                ("z".to_string(), (-10.0, 10.0)),
                ("y".to_string(), (-7.0, 7.0)),
            ]);
            let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
            assert_eq!(&eq_system.len(), &2);
            let mut nr = BVP::new(
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
                Some(rel_tolerance),
                Some(Bounds),
                Some("info".to_string()), // Some("error".to_string()),Some("warn".to_string()),
            );

            println!("solving system");
            #[allow(unused_variables)]
            nr.solve();
            // println!("result = {:?}", solution);
            // get solution plot using plotters crate or gnuplot crate (gnuplot library MUST BE INSTALLED AND IN THE PATH)
            nr.plot_result();
            nr.gnuplot_result();
            // save to txt
            nr.save_to_file(None);
            // save to csv
            nr.save_to_csv(None);
        }

        20 => {
            //Utils::profiling::pprof_profiling();
            Utils::sys_info::this_system_info();
        }
        /**/
        _ => {
            println!("example not found");
        }
    }
    //_________________________________________________
}
