#![allow(non_snake_case)]
use std::collections::HashMap;
pub mod symbolic;

use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
pub mod numerical;
use crate::numerical::NR::NR;
use crate::numerical::BE::BE;
use crate::numerical::ODE_api::ODEsolver;
use crate::numerical::NR_for_ODE:: NRODE;
use sprs::{CsMat, CsVec};
use nalgebra::{DMatrix, DVector, };
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP as NRBDVPd;
use crate::numerical::Examples_and_utils::NonlinEquation;
pub mod somelinalg;


fn main() {
  let example = 18;
   match example {
    
  0 => {


    // FUNCTION OF MULTIPLE VARIABLES
    //parse expression from string to symbolic expression
   let input = "exp(x)+log(y)";   //log(x)/y-x^2.3 *log(x+y+y^2.6)-exp(x-y)/(x+y) +  (log((x-y)/(x+y)))^2
      // here you've got symbolic expression
   let parsed_expression = Expr::parse_expression(input);
   println!(" parsed_expression {}", parsed_expression);
   // turn symbolic expression to a pretty human-readable string
   let parsed_function = parsed_expression.sym_to_str("x");
   println!("{}, sym to string: {}  \n",input,  parsed_function);
   // return vec of all arguments
   let  all = parsed_expression.all_arguments_are_variables();
   println!("all arguments are variables {:?}",all);
   let variables = parsed_expression.extract_variables();
   println!("variables {:?}",variables);

   // differentiate with respect to x and y
   let df_dx = parsed_expression.diff("x");
   let df_dy = parsed_expression.diff("y");
   println!("df_dx = {}, df_dy = {}", df_dx, df_dy);
  //convert symbolic expression to a Rust function and evaluate the function
  let args = vec!["x","y"];
  let function_of_x_and_y = parsed_expression.lambdify( args );
  let f_res = function_of_x_and_y( vec![1.0, 2.0] );
  println!("f_res = {}", f_res);
  // or you dont want to pass arguments you can use lambdify_wrapped, arguments will be found inside function
  let function_of_x_and_y = parsed_expression.lambdify_wrapped( );
  let f_res = function_of_x_and_y( vec![1.0, 2.0] );
  println!("f_res2 = {}", f_res);

  let start = vec![ 1.0, 1.0];
  let end = vec![ 2.0, 2.0];
  // evaluate function of 2 or more arguments using linspace for defining vectors of arguments
    let result = parsed_expression.lamdified_from_linspace(start.clone(), end.clone(), 10); 
    println!("evaluated function of 2 arguments = {:?}", result);
   //  find vector of derivatives with respect to all arguments
    let vector_of_derivatives = parsed_expression.diff_multi();
    println!("vector_of_derivatives = {:?}, {}", vector_of_derivatives, vector_of_derivatives.len());
   // compare numerical and analtical derivatives for a given linspace defined by start, end values and number of values.
   // max_norm - maximum norm of the difference between numerical and analtical derivatives
  let comparsion = parsed_expression.compare_num(start, end, 100, 1e-6);
  println!(" result_of compare = {:?}", comparsion);


}
  1=> {
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
    let vector_of_symbolic_vars = Expr::Symbols( "a, b, c");
    println!("vector_of_symbolic_vars = {:?}", vector_of_symbolic_vars);
    let ( a,  b,  c) = (vector_of_symbolic_vars[0].clone(), 
    // consruct symbolic expression
    vector_of_symbolic_vars[1].clone(), vector_of_symbolic_vars[2]. clone());
    let symbolic_expression =  a + Expr::exp(b * c);
    println!("symbolic_expression = {:?}", symbolic_expression);
    // if you want to change a variable inti constant:
    let expression_with_const =  symbolic_expression.set_variable("a", 1.0);
    println!("expression_with_const = {:?}", expression_with_const);
    let parsed_function = expression_with_const.sym_to_str("a");
    println!("{}, sym to string:",  parsed_function);

   }
   4 => {
      // JACOBIAN
      // instance of Jacobian structure
      let mut Jacobian_instance = Jacobian::new();
      // function of 2 or more arguments 
     let vec_of_expressions = vec![ "2*x^3+y".to_string(), "1.0".to_string()]; 
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

      println!("Jacobian_instance: functions  {:?}. Variables {:?}", Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables);
       println!("Jacobian_instance: Jacobian  {:?} readable {:?}.", Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian);
      for i in 0.. Jacobian_instance.symbolic_jacobian.len() {
        for j in 0.. Jacobian_instance.symbolic_jacobian[i].len() {
          println!("Jacobian_instance: Jacobian  {} row  {} colomn {:?}", i, j, Jacobian_instance.symbolic_jacobian[i][j]);
        }
       
      }
      // calculate element of jacobian (just for control)
      let ij_element = Jacobian_instance.calc_ij_element(0, 0,  vec!["x", "y"],vec![10.0, 2.0]) ;
      println!("ij_element = {:?} \n", ij_element);
      // evaluate jacobian to numerical values
       Jacobian_instance.evaluate_func_jacobian(&vec![10.0, 2.0]);
       println!("Jacobian = {:?} \n", Jacobian_instance.evaluated_jacobian);
       // lambdify and evaluate function vector to numerical values
      Jacobian_instance. lambdify_and_ealuate_funcvector(vec!["x", "y"], vec![10.0, 2.0]);
       println!("function vector = {:?} \n", Jacobian_instance.evaluated_functions);
       // or first lambdify
       Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
       // then evaluate
       Jacobian_instance.evaluate_funvector_lambdified(vec![10.0, 2.0]);
       println!("function vector after evaluate_funvector_lambdified = {:?} \n", Jacobian_instance.evaluated_functions);
       // evaluate jacobian to nalgebra matrix format
       Jacobian_instance.evaluate_func_jacobian_DMatrix(vec![10.0, 2.0]);
       println!("Jacobian_DMatrix = {:?} \n", Jacobian_instance.evaluated_jacobian_DMatrix);
       // evaluate function vector to nalgebra matrix format
       Jacobian_instance.evaluate_funvector_lambdified_DVector(vec![10.0, 2.0]);
       println!("function vector after evaluate_funvector_lambdified_DMatrix = {:?} \n", Jacobian_instance.evaluated_functions_DVector);
      

   }
   5 => {  
    //use the shortest way to solve system of equations
    // first define system of equations and initial guess
    let mut NR_instanse = NR::new();
    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()]; 
    let initial_guess = vec![1.0, 1.0];
    // solve
    NR_instanse.eq_generate_from_str(vec_of_expressions,initial_guess, 1e-6, 100);
    NR_instanse.solve();
    println!("result = {:?} \n", NR_instanse.get_result().unwrap());
    // or more verbose way...
    // first define system of equations
    
    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()]; 
    let mut Jacobian_instance = Jacobian::new();
     Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
     Jacobian_instance.set_variables(vec!["x", "y"]);
     Jacobian_instance.calc_jacobian();
     Jacobian_instance.jacobian_generate(vec!["x", "y"]);
     Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
     Jacobian_instance.readable_jacobian();
     println!("Jacobian_instance: functions  {:?}. Variables {:?}", Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables);
      println!("Jacobian_instance: Jacobian  {:?} readable {:?}. \n", Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian);
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
    6=> {
      // INDEXED VARIABLES
      let (matrix_of_indexed, vec_of_names) = Expr::IndexedVars2D(1, 10, "x");
      println!("matrix_of_indexed = {:?} \n", matrix_of_indexed);
      println!("vec_of_names = {:?} \n", vec_of_names);
    }
    7=>{
      //create instance of structure for symbolic equation system and Jacobian

      // define argument and unknown variables
 
      let y = Expr::Var("y".to_string());
      let z:Expr = Expr::Var("z".to_string());
      //define equation system
      let eq1:Expr = Expr::Const(-1.0f64)*z.clone() + (Expr::Const(-1.0)*y.clone() ).exp();
      let eq2:Expr = y;
      let eq_system = vec![eq1, eq2];
      // set unkown variables
      let values = vec![  "z".to_string(), "y".to_string()];
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
        first_step
    );
 
    ODE_instance.solve();
    // plot the solution
    ODE_instance.plot_result();
    let _ =  ODE_instance.save_result();

    }
   8 => {
         //Example 2 the laziest way to solve ODE
    // set RHS of system as vector of strings
    let RHS = vec!["-z-exp(-y)", "y"];
    // parse RHS as symbolic expressions
    let Equations = Expr::parse_vector_expression(RHS.clone());
    let values = vec![  "z".to_string(), "y".to_string()];
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
        first_step
    );
 
    ODE_instance.solve();
    ODE_instance.plot_result();
   } 

   9=> {

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
    let mut nr = NRODE::new(eq_system, initial_guess, values, arg, tolerance, max_iterations, max_error);
    nr.eq_generate();
  
    assert_eq!(nr.eq_system.len(), 2);
    nr.set_t(1.0); 

  
    let _ = nr.solve().unwrap();

   }
   10=> {
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
   
        solver.set_initial(eq_system, values, arg, tolerance, max_iterations, h, t0, t_bound, y0);
        println!("y = {:?}, initial_guess = {:?}", solver.newton.y,solver.newton.initial_guess);
        solver.newton.eq_generate();
        solver.solve();
        let result = solver.get_result();
        println!("\n result 1 = {:?}", result.1);
        println!("result = {:?}", result.1.unwrap().shape());
        solver.plot_result();
     //   println!("result = {:?}", result);
        
   }
   11=> {
          //  Backward Euler method: slightly non-linear ODE
          let RHS = vec!["-z-exp(-y)", "y"];
          // parse RHS as symbolic expressions
          let Equations = Expr::parse_vector_expression(RHS.clone());
          let values = vec![  "z".to_string(), "y".to_string()];
          println!("eq_system = {:?}",  Equations);
          let y0 = DVector::from_vec(vec![1.0, 1.0]);
       
          let arg = "x".to_string();
          let tolerance = 1e-2;
          let max_iterations = 500;
    
          let h = Some(1e-3);
          let t0 = 0.0;
          let t_bound = 1.0;
  
          let mut solver = BE::new();
     
          solver.set_initial( Equations, values, arg, tolerance, max_iterations, h, t0, t_bound, y0);
          println!("y = {:?}, initial_guess = {:?}", solver.newton.y,solver.newton.initial_guess);
          solver.newton.eq_generate();
          solver.solve();
          #[allow(unused_variables)]
          let result = solver.get_result();
        //  println!("\n result 0 = {:?}", result.0);
         // println!("\n result 1 = {:?}", result.1);
         // println!("result = {:?}", result.1.unwrap().shape());
          solver.plot_result();
   }

   12=> {
    //  Backward Euler method: slightly non-linear ODE: adaptative time step 
    let RHS = vec!["-z-exp(-y)", "y"];
    // parse RHS as symbolic expressions
    let Equations = Expr::parse_vector_expression(RHS.clone());
    let values = vec![  "z".to_string(), "y".to_string()];
    println!("eq_system = {:?}",  Equations);
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
 
    let arg = "x".to_string();
    let tolerance = 1e-2;
    let max_iterations = 500;

    let h = None;
    let t0 = 0.0;
    let t_bound = 1.0;

    let mut solver = BE::new();

    solver.set_initial( Equations, values, arg, tolerance, max_iterations, h, t0, t_bound, y0);
    println!("y = {:?}, initial_guess = {:?}", solver.newton.y,solver.newton.initial_guess);
    solver.newton.eq_generate();
    solver.solve();
    #[allow(unused_variables)]
    let result = solver.get_result();
  //  println!("\n result 0 = {:?}", result.0);
   // println!("\n result 1 = {:?}", result.1);
   // println!("result = {:?}", result.1.unwrap().shape());
    solver.plot_result();
}

13 =>{ //Non-stiff equations: use ODE general api ODEsolver 
  // RK45 and Dormand-Prince methods are available

  let RHS = vec!["-z-y", "y"];
  // parse RHS as symbolic expressions
  let Equations = Expr::parse_vector_expression(RHS.clone());
  let values = vec![  "z".to_string(), "y".to_string()];
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

14 =>{ //BVP jacobian matrix
  let RHS = vec!["-z-y", "y"];
  // parse RHS as symbolic expressions
  let Equations = Expr::parse_vector_expression(RHS.clone());
  let values = vec![  "z".to_string(), "y".to_string()];
  let arg = "x".to_string();
  let n_steps = 3;
  let h = 1e-4;
  let BorderConditions = HashMap::from([
    ("z".to_string(), (0, 1000.0)),
    ("y".to_string(), (1, 333.0)),
      ]);  

  let Y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
  let mut Jacobian_instance = Jacobian::new();
  // creating analytic discretized algebraic system, its functional representation, analytic Jacobian matrix and its functional representation
  Jacobian_instance.generate_BVP(Equations.clone(), values.clone(), arg.clone(),0.0,None, Some( n_steps),
  Some (h), None, BorderConditions.clone(), None, None );

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
  println!("vector of variables {:?}",varvect);
  let Ys = DVector::from_vec(Y.clone());
  let J_eval1 = J_func(4.0, &Ys);
  println!("Jacobian Dense: J_eval = {:?} \n", J_eval1);
// SPARSE JACOBIAN MATRIX with nalgebra (+sparse feature) crate
 
  Jacobian_instance.generate_BVP_CsMatrix(Equations.clone(), values.clone(), arg.clone(),0.0,None, Some( n_steps),
  Some (h), None, BorderConditions.clone(), None, None);
  let J_func3 = &Jacobian_instance.function_jacobian_IVP_CsMatrix;
  let J_eval3 = J_func3(4.0, &Ys);
  println!("Jacobian Sparse with CsMatrix: J_eval = {:?} \n", J_eval3);

// SPARSE JACOBIAN MATRIX with  crate
  Jacobian_instance.generate_BVP_CsMat(Equations.clone(), values.clone(), arg.clone(), 0.0, None,Some( n_steps),
  Some (h), None, BorderConditions.clone(), None, None);
  let J_func2:   &Box<dyn Fn(f64, &CsVec<f64>) -> CsMat<f64> >= &Jacobian_instance.function_jacobian_IVP_CsMat;
  let F_func2 = &Jacobian_instance.lambdified_functions_IVP_CsVec;
  let Ys2 = CsVec::new(Y.len(), vec![0, 1, 2, 3, 4, 5], Y.clone());
  println!("Ys = {:?} \n", &Ys2);
  let F_eval2 = F_func2(4.0, &Ys2);
  println!("F_eval = {:?} \n", F_eval2);
  let J_eval2: CsMat<f64> = J_func2(4.0, &Ys2);

  println!("Jacobian Sparse with CsMat: J_eval = {:?} \n", J_eval2);

  // SPARSE JACOBIAN MATRIX with sprs crate
  let mut Jacobian_instance = Jacobian::new();
  Jacobian_instance.generate_BVP_SparseColMat(Equations.clone(), values.clone(), arg.clone(), 0.0, None,Some( n_steps),
  Some (h), None, BorderConditions.clone(), None, None);
  let J_func3 = &Jacobian_instance.function_jacobian_IVP_SparseColMat;
  let F_func3 = &Jacobian_instance.lambdified_functions_IVP_Col;
  use faer::col::{Col, from_slice};
  let Ys3:Col<f64> = from_slice( Y.as_slice()).to_owned();
  println!("Ys = {:?} \n", &Ys3);
  let F_eval3 = F_func3(4.0, &Ys3);
  println!("F_eval = {:?} \n", F_eval3);
  #[allow(unused_variables)]
  let J_eval2 = J_func3(4.0, &Ys3);

  println!("Jacobian Sparse with SparseColMat: J_eval = {:?} \n", J_eval3);

  }
  15=>{
    
    let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^2");
        let eq_system = vec![eq1, eq2];
    

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-5;
        let max_iterations = 50;
        let max_error = 0.0;
        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 200;
        let strategy =   "Naive".to_string();//
        let  strategy_params = None;
        let method =   "Sparse".to_string();// or  "Dense"
        let linear_sys_method= None;
        let ones = vec![0.0; values.len()*n_steps];
        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
        BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
        assert_eq!(&eq_system.len(), &2);
        let mut nr =  NRBVP::new(eq_system,
             initial_guess, 
             values, 
             arg,
             BorderConditions, t0, t_end, n_steps,strategy, strategy_params, linear_sys_method, method, tolerance, max_iterations, max_error);

        println!("solving system");
        #[allow(unused_variables)]
        let solution = nr.solve().unwrap();
       // println!("result = {:?}", solution);
        nr.plot_result();

  }
  16=>{
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
        let max_error = 0.0;
        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 100; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min, 
        let strategy =   "Frozen".to_string();//
        let  strategy_params = Some(HashMap::from([("complex".to_string(), 
       Some(Vec::from( [2f64, 5.0, 1e-1, ]  ))
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
        let method =   "Sparse".to_string();// or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.0; values.len()*n_steps];
        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
        BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
        assert_eq!(&eq_system.len(), &2);
        let mut nr =  NRBVP::new(eq_system,
             initial_guess, 
             values, 
             arg,
             BorderConditions, t0, t_end, n_steps,strategy, strategy_params, linear_sys_method, method, tolerance, max_iterations, max_error);

        println!("solving system");
        #[allow(unused_variables)]
        let solution = nr.solve().unwrap();
       // println!("result = {:?}", solution);
        nr.plot_result();

  }
  17=>{
   
    let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];
    

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-5;
        let max_iterations = 20;
        let max_error = 1e-6;
        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 50; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min, 
        let strategy =   "Damped".to_string();//
        let  strategy_params = Some(HashMap::from([("max_jac".to_string(), 
       None,    ), ("maxDampIter".to_string(), 
       None,    ), ("DampFacor".to_string(), 
       None,    )
   
      ]));
    
        let method =   "Sparse".to_string();// or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.0; values.len()*n_steps];
        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
        BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
        let Bounds = HashMap::from([  ("z".to_string(), (-10.0, 10.0),    ), ("y".to_string(), (-7.0, 7.0),    ) ]);
        let rel_tolerance =  HashMap::from([  ("z".to_string(), 1e-4    ), ("y".to_string(), 1e-4,    ) ]);
        assert_eq!(&eq_system.len(), &2);
        let mut nr =  NRBDVPd::new(eq_system,
             initial_guess, 
             values, 
             arg,
             BorderConditions, t0, t_end, n_steps,strategy, strategy_params, linear_sys_method, method, tolerance, Some(rel_tolerance), max_iterations, max_error, Some(Bounds));

        println!("solving system");
        #[allow(unused_variables)]
        let solution = nr.solve().unwrap();
       // println!("result = {:?}", solution);
        nr.plot_result();

  }

  18=>{
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
        let ne=  NonlinEquation:: TwoPointBVP; //  Clairaut   LaneEmden5  ParachuteEquation  TwoPointBVP
        
        let eq_system =  ne.setup();
    

        let values = ne.values();
        let arg = "x".to_string();
        let tolerance = 1e-5;
        let max_iterations = 200;
        let max_error = 1e-6;
        let t0 = ne.span(None, None).0;
        let t_end = ne.span(None, None).1;
        let n_steps = 100; // 
        let strategy =   "Damped".to_string();//
        let  strategy_params = Some(HashMap::from([("max_jac".to_string(), 
       None,    ), ("maxDampIter".to_string(), 
       None,    ), ("DampFacor".to_string(), 
       None,    )
   
      ]));
    
        let method =   "Sparse".to_string();// or  "Dense"
        let linear_sys_method = None;
        let ones = vec![0.99; values.len()*n_steps];
        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let  BorderConditions = ne.boundary_conditions();
        let Bounds = ne.Bounds();
        let rel_tolerance =  ne.rel_tolerance();
       
        let mut nr =  NRBDVPd::new(eq_system,
             initial_guess, 
             values, 
             arg,
             BorderConditions, t0, t_end, n_steps,strategy, strategy_params, linear_sys_method, method, tolerance, Some(rel_tolerance), max_iterations, max_error, Some(Bounds));

        println!("solving system");
        nr.solve();
        let solution = nr.get_result().unwrap();
      //  print!("result = {:?}\n", solution.column(1).iter().map(|x| *x).collect::<Vec<_>>());
        let y_numer = solution.column(0);
        let y_numer:Vec<f64> = y_numer.iter().map(|x| *x).collect();
       
        nr.plot_result();
        //compare with exact solution
       let y_exact = ne.exact_solution(None, None, Some(n_steps));
       let n = &y_exact.len();
      // println!("numerical result = {:?}",  y_numer);
       println!("\n \n y exact{:?}, {}",  &y_exact, &y_exact.len());
       println!("\n \n y numer{:?}, {}",  &y_numer, &y_numer.len()  );
       let comparsion:Vec<f64> = y_numer.into_iter().zip(y_exact.clone()).map( | (y_n_i, y_e_i)| (y_n_i- y_e_i).powf(2.0)).collect();
       let norm = comparsion.iter().sum::<f64>().sqrt()/(*n as f64);
      let max_residual = comparsion.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
      let position = comparsion.iter().position(|&x| x == *max_residual).unwrap();
      let relativ_residual = max_residual.abs()/y_exact[position];
       println!("maximum relative residual of numerical solution wioth respect to exact solution = {}", relativ_residual);
       println!("norm = {}", norm);


       

  }
  /*
  
  
   */
  _ => {
    println!("example not found");  }
  }
   //_________________________________________________
   
  


}

