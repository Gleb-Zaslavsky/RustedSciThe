#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
///  
///  Example#1
 /// ```
 ///
 /// //use the shortest way to solve system of equations
 ///    // first define system of equations and initial guess
 ///  use RustedSciThe::numerical::NR::NR;
 ///    let mut NR_instanse = NR::new();
 ///    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()]; 
///   let initial_guess = vec![1.0, 1.0];
///    // solve
///    NR_instanse.eq_generate_from_str(vec_of_expressions,initial_guess, 1e-6, 100, );
///    NR_instanse.solve();
///    println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
/// Example#2
/// ```
///    // or more verbose way...
///    // first define system of equations
///     use RustedSciThe::symbolic::symbolic_engine::Expr;
///     use RustedSciThe::symbolic::symbolic_functions::Jacobian;
///   use RustedSciThe::numerical::NR::NR;
///    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()]; 
///    let mut Jacobian_instance = Jacobian::new();
///     Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
///     Jacobian_instance.set_variables(vec!["x", "y"]);
///     Jacobian_instance.calc_jacobian();
///     Jacobian_instance.jacobian_generate(vec!["x", "y"]);
///     Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
///     Jacobian_instance.readable_jacobian();
///     println!("Jacobian_instance: functions  {:?}. Variables {:?}", Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables);
///      println!("Jacobian_instance: Jacobian  {:?} readable {:?}. \n", Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian);
///     let initial_guess = vec![1.0, 1.0];
///     // in case you are interested in Jacobian value at initial guess
///     Jacobian_instance.evaluate_func_jacobian_DMatrix(initial_guess.clone());
///     Jacobian_instance.evaluate_funvector_lambdified_DVector(initial_guess.clone());
///     let guess_jacobian = (Jacobian_instance.evaluated_jacobian_DMatrix).clone();
///     println!("guess Jacobian = {:?} \n", guess_jacobian.try_inverse());
///     // defining NR method instance and solving
///     let mut NR_instanse = NR::new();
///     NR_instanse.set_equation_sysytem(Jacobian_instance, initial_guess, 1e-6, 100, );
///     NR_instanse.solve();
///     println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
pub mod NR;
pub mod NR_for_ODE;
pub mod NR_for_Euler;
/// Example#3
///     ```
/// // Beckward-Euler method 
///         use RustedSciThe:::symbolic::symbolic_engine::Expr;
///         use RustedSciThe::numerical::BE::BE;
///          //  Backward Euler method: slightly non-linear ODE
///          let RHS = vec!["-z-exp(-y)", "y"];
///          // parse RHS as symbolic expressions
///          let Equations = Expr::parse_vector_expression(RHS.clone());
///          let values = vec![  "z".to_string(), "y".to_string()];
///          println!("eq_system = {:?}",  Equations);
///          let y0 = DVector::from_vec(vec![1.0, 1.0]);
       
///          let arg = "x".to_string();
///          let tolerance = 1e-2;
///          let max_iterations = 500;
    
///          let h = Some(1e-3);
///          let t0 = 0.0;
///          let t_bound = 1.0;
  
///          let mut solver = BE::new();
     
///          solver.set_initial( Equations, values, arg, tolerance, max_iterations, h, t0, t_bound, y0);
///          println!("y = {:?}, initial_guess = {:?}", solver.newton.y,solver.newton.initial_guess);
///          solver.newton.eq_generate();
///          solver.solve();
///          let result = solver.get_result();
///        //  println!("\n result 0 = {:?}", result.0);
///         // println!("\n result 1 = {:?}", result.1);
///         // println!("result = {:?}", result.1.unwrap().shape());
///          solver.plot_result();
///     ```
pub mod BE;
///pub mod BDF;
/// SOLVER OF STIFF IVP
/// direct rewrite to Rust python code from SciPy
/// API for this is RustedSciThe::numerical::ODE_api::ODEsolver;
pub mod BDF;
// API for this is RustedSciThe::numerical::ODE_api::ODEsolver;
pub mod NonStiff_api;
/// usage of BDF solver
///  general api for ODE solvers  (now written only BDF)
/// Example#1
/// ```
///use RustedSciThe::symbolic::symbolic_engine::Expr;
/// use RustedSciThe::symbolic::symbolic_functions::Jacobian;
/// use RustedSciThe::numerical::ODE_api::ODEsolver;
/// 
///       //create instance of structure for symbolic equation system and Jacobian
/// let mut Jacobian_instance = Jacobian::new();
// define argument andunknown variables
/// let x = Expr::Var("x".to_string()); // argument
/// let y = Expr::Var("y".to_string());
/// let z:Expr = Expr::Var("z".to_string());
//define equation system
///  let eq1:Expr = Expr::Const(-1.0 as f64)*z.clone() - (Expr::Const(-1.0 as f64)*y.clone() ).exp();
/// let eq2:Expr = y;
/// let eq_system = vec![eq1, eq2];
// set unkown variables
/// let values = vec![  "z".to_string(), "y".to_string()];
// set argument
/// let arg = "x".to_string();
// set method
/// let method = "BDF".to_string();
// set initial conditions
/// let t0 = 0.0;
/// let y0 = vec![1.0, 1.0];
/// let t_bound = 1.0;
/// // set solver parameters (optional)
/// let first_step = None;
/// let atol = 1e-5;
/// let rtol = 1e-5;
/// let max_step = 1e-3;
/// let jac_sparsity = None;
/// let vectorized = false;
// create instance of ODE solver and solve the system
///let mut ODE_instance = ODEsolver::new_complex(
///    eq_system,
///    values,
///    arg,
///    method,
///    t0,
///    y0.into(),
///    t_bound,
///    max_step,
///    rtol,
///    atol,
///    jac_sparsity,
///    vectorized,
///   first_step);
/// ODE_instance.solve();
/// // plot the solution
/// ODE_instance.plot_result();
/// ODE_instance.save_result();
/// ```
/// Non-stiff equations: use ODE general api ODEsolver 
/// RK45 and Dormand-Prince methods are available
/// Example#3
/// ```
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
///  use RustedSciThe::numerical::ODE_api::ODEsolver;
///          //Example 2 the laziest way to solve ODE
///    // set RHS of system as vector of strings
///    let RHS = vec!["-z-exp(-y)", "y"];
///    // parse RHS as symbolic expressions
///    let Equations = Expr::parse_vector_expression(RHS.clone());
///    let values = vec![  "z".to_string(), "y".to_string()];
///    println!("Equations = {:?}", Equations);   
///    // set argument
///    let arg = "x".to_string();
///      // set method
///      let method = "BDF".to_string();
///      // set initial conditions
///      let t0 = 0.0;
///      let y0 = vec![1.0, 1.0];
///      let t_bound = 1.0;
///      // set solver parameters (optional)
///      let first_step = None;
///      let atol = 1e-5;
///      let rtol = 1e-5;
///      let max_step = 1e-3;
///      let jac_sparsity = None;
///      let vectorized = false;
///      // create instance of ODE solver and solve the system
///      let mut ODE_instance = ODEsolver::new_complex(
///        Equations,
///        values,
///        arg,
///        method,
///        t0,
///        y0.into(),
///        t_bound,
///        max_step,
///        rtol,
///        atol,   
///        jac_sparsity,
///        vectorized,
///        first_step
///    );
///    ODE_instance.solve();
///    ODE_instance.plot_result();
///    ODE_instance.save_result();
///  ```
/// Example#2
/// ```
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
///  use RustedSciThe::numerical::ODE_api::ODEsolver;
///   let RHS = vec!["-z-y", "y"];
  // parse RHS as symbolic expressions
///   let Equations = Expr::parse_vector_expression(RHS.clone());
///   let values = vec![  "z".to_string(), "y".to_string()];
///   println!("Equations = {:?}", Equations);   
///   // set argument
///   let arg = "x".to_string();
///     // set method
///     let method = "DOPRI".to_string(); 
///     // set initial conditions
///     let t0 = 0.0;
///     let y0 = vec![1.0, 1.0];
 ///    let t_bound = 1.0;
 ///    // set solver parameters (optional)
/// 
///     let max_step = 1e-3;
/// 
///     // create instance of ODE solver and solve the system
///     let mut ODE_instance = ODEsolver::new_easy(
///       Equations,
///       values,
///       arg,
///       method,
///       t0,
///       y0.into(),
///       t_bound,
///       max_step,
 ///  );
///   ODE_instance.solve();
///   ODE_instance.plot_result();
///  ```
pub mod ODE_api;
/// tiny module to plot result of IVP computation
pub mod plots;


pub mod BVP_sci;





pub mod Examples_and_utils;
//BVP section
 /*
     frozen jacobian strategy - recalculating jacobian on condition:
     Description of strategy                                                                 key of strategy        value user must provude for strategy
    1. only first time:                                                                      "Frozen_naive"                   None
    2. every m-th time, where m is a parameter of the strategy:                                "every_m"                    m
    3. every time when the solution norm greater than a certain threshold A:                   "at_high_norm".               A
    4. when norm of (i-1) iter multiplied by certain value B(<1) is lower than norm of i-th iter : "at_low_speed".            B
    5. complex - combined strategies 2,3,4                                                         "complex"              vec of  parameters [m, A, B]

     */
/// Example#5
/// ```
///      use nalgebra::DMatrix;
///      use nalgebra::DVector;
///      use std::collections::HashMap;
///     use  RustedSciThe::symbolic::symbolic_engine::Expr;
///      use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP;
///    let eq1 = Expr::parse_expression("y-z");
///        let eq2 = Expr::parse_expression("-z^2");
///        let eq_system = vec![eq1, eq2];
    
///
///        let values = vec!["z".to_string(), "y".to_string()];
///        let arg = "x".to_string();
///        let tolerance = 1e-5;
///        let max_iterations = 1500;
///        let max_error = 0.0;
///        let t0 = 0.0;
///        let t_end = 1.0;
///        let n_steps = 100; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min, 
///        let strategy =   "Frozen".to_string();//
///        let  strategy_params = Some(HashMap::from([("complex".to_string(), 
///       Some(Vec::from( [2f64, 5.0, 1e-1, ]  ))
///      )]));
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
///        let method =   "Sparse".to_string();// or  "Dense"
///        let linear_sys_method = None;
///        let ones = vec![0.0; values.len()*n_steps];
///        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
///        let mut BorderConditions = HashMap::new();
///        BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
///        BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
///        assert_eq!(&eq_system.len(), &2);
///        let mut nr =  NRBVP::new(eq_system,
///             initial_guess, 
///             values, 
///             arg,
///             BorderConditions, t0, t_end, n_steps,strategy, strategy_params, linear_sys_method, method, tolerance, max_iterations, max_error);

///        println!("solving system");
///        #[allow(unused_variables)]
///        let solution = nr.solve().unwrap();
///       // println!("result = {:?}", solution);
///        nr.plot_result();
///    ```

/*
Modified Newton method or Damped Newton method for solving a system of nonlinear ordinary differential equations.

This code implements a modified Newton method for solving a system of non-linear boundary value problems..
The code mostly inspired by sources listed below:
-  Cantera MultiNewton solver (MultiNewton.cpp )
- TWOPNT fortran solver (see "The Twopnt Program for Boundary Value Problems" by J. F. Grcar and Chemkin Theory Manual p.261)
 */


/// Example#4
/// ```
/// use nalgebra::DMatrix;
///  use nalgebra::DVector;
///      use std::collections::HashMap;
///     use  RustedSciThe::symbolic::symbolic_engine::Expr;
///     use  RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP as NRBDVPd;
///     let eq1 = Expr::parse_expression("y-z");
///        let eq2 = Expr::parse_expression("-z^3");
///        let eq_system = vec![eq1, eq2];
///        let values = vec!["z".to_string(), "y".to_string()];
///        let arg = "x".to_string();
///        let tolerance = 1e-5;
///        let max_iterations = 20;
///        let max_error = 1e-6;
///        let t0 = 0.0;
///        let t_end = 1.0;
///        let n_steps = 50; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min, 
///        let strategy =   "Damped".to_string();//
///        let  strategy_params = Some(HashMap::from([("max_jac".to_string(), 
///       None,    ), ("maxDampIter".to_string(), 
///       None,    ), ("DampFacor".to_string(), 
///       None,    )
///      ])); 
///        let method =   "Sparse".to_string();// or  "Dense"
///        let linear_sys_method = None;
///        let ones = vec![0.0; values.len()*n_steps];
///        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
///        let mut BorderConditions = HashMap::new();
///        BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
///        BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
///        let Bounds = HashMap::from([  ("z".to_string(), (-10.0, 10.0),    ), ("y".to_string(), (-7.0, 7.0),    ) ]);
///        let rel_tolerance =  HashMap::from([  ("z".to_string(), 1e-4    ), ("y".to_string(), 1e-4,    ) ]);
///        assert_eq!(&eq_system.len(), &2);
///        let mut nr =  NRBDVPd::new(eq_system,
///             initial_guess, 
///             values, 
///             arg,
///             BorderConditions, t0, t_end, n_steps,strategy, strategy_params, linear_sys_method, method, tolerance, Some(rel_tolerance), max_iterations, max_error, Some(Bounds));

///        println!("solving system");
///        #[allow(unused_variables)]
///        let solution = nr.solve().unwrap();
///       // println!("result = {:?}", solution);
///        nr.plot_result();
///    ```
/// 
pub mod BVP_Damp;