pub mod dogleg;

pub mod LM_utils;
///  Example#1
/// ```
///
/// //use the shortest way to solve system of equations
///    // first define system of equations and initial guess
///  use RustedSciThe::numerical::Nonlinear_systems::NR::NR;
///     
/// //use the shortest way to solve system of equations
///    // first define system of equations and initial guess
///    let mut NR_instanse = NR::new();
///    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()];
///   let initial_guess = vec![1.0, 1.0];
///    // solve
///    NR_instanse.eq_generate_from_str(vec_of_expressions, None, initial_guess, 1e-6, 100);
///    NR_instanse.main_loop();
///    println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
/// Example#2
/// ```
///    // or more verbose way...
///
///    // first define system of equations
///     use RustedSciThe::numerical::Nonlinear_systems::NR::NR;
///     use RustedSciThe::symbolic::symbolic_engine::Expr;
///     use RustedSciThe::symbolic::symbolic_functions::Jacobian;
///      use nalgebra::DVector;
///     let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];
///
///     let initial_guess = vec![1.0, 1.0];
///     let mut NR_instanse = NR::new();
///     let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
///     let values = vec!["x".to_string(), "y".to_string()];
///     NR_instanse.set_equation_system(vec_of_expr, Some(values.clone()), initial_guess, 1e-6, 100);
///     NR_instanse.eq_generate();
///     NR_instanse.main_loop();
///     let solution = NR_instanse.get_result().unwrap();
///      assert_eq!(solution,  DVector::from_vec(vec![3.0, -1.0])   );
///     println!("result = {:?} \n", NR_instanse.get_result().unwrap());
//
///  ```
pub mod NR;
pub mod NR_LM;
pub mod NR_LM_Nielsen;
pub mod NR_LM_minpack;
pub mod NR_damped;
pub mod NR_trust_region;
