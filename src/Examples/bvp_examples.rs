#![allow(non_snake_case)]
use std::collections::HashMap;

use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;

use crate::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP as NRBDVPd;
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP;
use crate::numerical::BVP_api::BVP;
use crate::numerical::Examples_and_utils::NonlinEquation;

use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, CsVec};

pub fn bvp_examples(example: usize) {
    match example {
        0 => {
            //BVP jacobian matrix
            let RHS = vec!["-z-y", "y"];
            // parse RHS as symbolic expressions
            let Equations = Expr::parse_vector_expression(RHS.clone());
            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let n_steps = 3;
            let scheme = "forward".to_string();
            let h = 1e-4;
            let BorderConditions = HashMap::from([
                ("z".to_string(), (0, 1000.0)),
                ("y".to_string(), (1, 333.0)),
            ]);

            let Y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            let mut Jacobian_instance = Jacobian::new();
            // creating analytic discretized algebraic system, its functional representation, analytic Jacobian matrix and its functional representation
            Jacobian_instance.generate_BVP(
                Equations.clone(),
                values.clone(),
                arg.clone(),
                0.0,
                None,
                Some(n_steps),
                Some(h),
                None,
                BorderConditions.clone(),
                None,
                None,
                scheme.clone(),
            );

            // analytic Jacobian matrix
            #[allow(unused_variables)]
            let J = &Jacobian_instance.symbolic_jacobian;
            // its functional representation
            let J_func = &Jacobian_instance.function_jacobian_IVP_DMatrix;
            // analytic discretized algebraic system,
            #[allow(unused_variables)]
            let F = &Jacobian_instance.vector_of_functions;
            // its functional representation
            #[allow(unused_variables)]
            let F_func = &Jacobian_instance.lambdified_functions_IVP_DVector;
            let varvect = &Jacobian_instance.vector_of_variables;
            println!("vector of variables {:?}", varvect);
            let Ys = DVector::from_vec(Y.clone());
            let J_eval1 = J_func(4.0, &Ys);
            println!("Jacobian Dense: J_eval = {:?} \n", J_eval1);
            // SPARSE JACOBIAN MATRIX with nalgebra (+sparse feature) crate

            Jacobian_instance.generate_BVP_CsMatrix(
                Equations.clone(),
                values.clone(),
                arg.clone(),
                0.0,
                None,
                Some(n_steps),
                Some(h),
                None,
                BorderConditions.clone(),
                None,
                None,
                scheme.clone(),
            );
            let J_func3 = &Jacobian_instance.function_jacobian_IVP_CsMatrix;
            let J_eval3 = J_func3(4.0, &Ys);
            println!("Jacobian Sparse with CsMatrix: J_eval = {:?} \n", J_eval3);

            // SPARSE JACOBIAN MATRIX with  crate
            Jacobian_instance.generate_BVP_CsMat(
                Equations.clone(),
                values.clone(),
                arg.clone(),
                0.0,
                None,
                Some(n_steps),
                Some(h),
                None,
                BorderConditions.clone(),
                None,
                None,
                scheme.clone(),
            );
            let J_func2: &Box<dyn Fn(f64, &CsVec<f64>) -> CsMat<f64>> =
                &Jacobian_instance.function_jacobian_IVP_CsMat;
            let F_func2 = &Jacobian_instance.lambdified_functions_IVP_CsVec;
            let Ys2 = CsVec::new(Y.len(), vec![0, 1, 2, 3, 4, 5], Y.clone());
            println!("Ys = {:?} \n", &Ys2);
            let F_eval2 = F_func2(4.0, &Ys2);
            println!("F_eval = {:?} \n", F_eval2);
            let J_eval2: CsMat<f64> = J_func2(4.0, &Ys2);

            println!("Jacobian Sparse with CsMat: J_eval = {:?} \n", J_eval2);

            // SPARSE JACOBIAN MATRIX with sprs crate
            let mut Jacobian_instance = Jacobian::new();
            Jacobian_instance.generate_BVP_SparseColMat(
                Equations.clone(),
                values.clone(),
                arg.clone(),
                0.0,
                None,
                Some(n_steps),
                Some(h),
                None,
                BorderConditions.clone(),
                None,
                None,
                scheme.clone(),
            );
            let J_func3 = &Jacobian_instance.function_jacobian_IVP_SparseColMat;
            let F_func3 = &Jacobian_instance.lambdified_functions_IVP_Col;
            use faer::col::{from_slice, Col};
            let Ys3: Col<f64> = from_slice(Y.as_slice()).to_owned();
            println!("Ys = {:?} \n", &Ys3);
            let F_eval3 = F_func3(4.0, &Ys3);
            println!("F_eval = {:?} \n", F_eval3);
            #[allow(unused_variables)]
            let J_eval2 = J_func3(4.0, &Ys3);

            println!(
                "Jacobian Sparse with SparseColMat: J_eval = {:?} \n",
                J_eval3
            );
        }
        1 => {
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
            BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
            BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
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
        2 => {
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
            let n_steps = 100; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
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
            BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
            BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
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
        3 => {
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
            let strategy_params = Some(HashMap::from([
                ("max_jac".to_string(), None),
                ("maxDampIter".to_string(), None),
                ("DampFacor".to_string(), None),
                ("adaptive".to_string(), None),
            ]));

            let method = "Sparse".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.0; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let mut BorderConditions = HashMap::new();
            BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
            BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
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

        4 => {
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
            let ne = NonlinEquation::TwoPointBVP; //  Clairaut   LaneEmden5  ParachuteEquation  TwoPointBVP

            let eq_system = ne.setup();

            let values = ne.values();
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 200;
            let t0 = ne.span(None, None).0;
            let t_end = ne.span(None, None).1;
            let n_steps = 10; //
            let strategy = "Damped".to_string(); //
            let strategy_params = Some(HashMap::from([
                ("max_jac".to_string(), None),
                ("maxDampIter".to_string(), None),
                ("DampFacor".to_string(), None),
                ("adaptive".to_string(), None),
            ]));
            let scheme = "forward".to_string();
            let method = "Sparse".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.99; values.len() * n_steps];
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
            let y_exact = ne.exact_solution(None, None, Some(n_steps));
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
            println!("maximum relative residual of numerical solution wioth respect to exact solution = {}", relativ_residual);
            println!("norm = {}", norm);
            nr.save_to_file(None)

            //BVP is general api for all variants of BVP solvers
        }
        5 => {
            let eq1 = Expr::parse_expression("y-z");
            let eq2 = Expr::parse_expression("-z^3");
            let eq_system = vec![eq1, eq2];

            let values = vec!["z".to_string(), "y".to_string()];
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 20;

            let t0 = 0.0;
            let t_end = 1.0;
            let n_steps = 10; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
            let strategy = "Damped".to_string(); //

            let strategy_params = match strategy.as_str() {
                "Naive" => None,
                "Damped" => Some(HashMap::from([
                    ("max_jac".to_string(), None),
                    ("maxDampIter".to_string(), None),
                    ("DampFacor".to_string(), None),
                    (
                        "adaptive".to_string(),
                        Some(vec![1.0, 5.0]), //  None
                    ),
                    //  ("pearson".to_string(), Some(vec![0.2] ) )
                    ("grcar_smooke".to_string(), Some(vec![0.2, 0.5])),
                ])),
                "Frozen" => Some(HashMap::from([(
                    "every_m".to_string(),
                    Some(Vec::from([5 as f64])),
                )])),
                &_ => panic!("Invalid strategy!"),
            };

            let scheme = "trapezoid".to_string();
            let method = "Dense".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.0; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let mut BorderConditions = HashMap::new();
            BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
            BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
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
                None,
            );

            println!("solving system");
            #[allow(unused_variables)]
            nr.solve();
            // println!("result = {:?}", solution);

            nr.plot_result();
            nr.save_to_file(None);
        }

        _ => {
            println!("example not found");
        }
    }
}
