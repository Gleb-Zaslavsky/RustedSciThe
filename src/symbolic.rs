#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
/// a module turns a String expression into a symbolic expression
///
///# Example
/// ```
///  use RustedSciThe::symbolic::symbolic_functions::Jacobian;
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
///let input = "x^2.3* log(x+y+y^2.6)"; //log(x)/y-x^2.3 log(x+y+y^2.6)-exp(x-y)/(x+y)
/// let parsed_expression = Expr::parse_expression(input);
///println!(" parsed_expression {}", parsed_expression);
/// let parsed_function = parsed_expression.lambdify( vec!["x","y"]);
/// println!("{}, Rust function: {}  \n",input,  parsed_function(vec![1.0, 2.0]));
///  ```
/// ________________________________________________________________________________________________________________________________
pub mod parse_expr;
///____________________________________________________________________________________________________________________________
/// # Symbolic engine
/// a module
/// 1) turns a String expression into a symbolic expression
/// 2) turns a symbolic expression into a Rust function
/// 3) turns a symbolic expression into a string expression for printing and control results
///# Example#
/// ```
///
/// use RustedSciThe::symbolic::parse_expr;
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
///let input = "x^2.3* log(x+y+y^2.6)"; //log(x)/y-x^2.3 log(x+y+y^2.6)-exp(x-y)/(x+y)
/// let parsed_expression = Expr::parse_expression(input);
///println!(" parsed_expression {}", parsed_expression);
/// let parsed_function = parsed_expression.sym_to_str("x");
/// println!("{}, Rust function: {}  \n",input,  parsed_function);
///  
/// ```
/// Example2#
/// ```
///
/// use RustedSciThe::symbolic::parse_expr;
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
/// let input = "exp(x)+log(y)";   //log(x)/y-x^2.3 *log(x+y+y^2.6)-exp(x-y)/(x+y) +  (log((x-y)/(x+y)))^2
///      // here you've got symbolic expression
///   let parsed_expression = Expr::parse_expression(input);
///   println!(" parsed_expression {}", parsed_expression);
///   // turn symbolic expression to a pretty human-readable string
///  let parsed_function = parsed_expression.sym_to_str("x");
///   println!("{}, sym to string: {}  \n",input,  parsed_function);
///   // return vec of all arguments
///   let  all = parsed_expression.all_arguments_are_variables();
///   println!("all arguments are variables {:?}",all);
///   let variables = parsed_expression.extract_variables();
///  println!("variables {:?}",variables);
///    // differentiate with respect to x and y
///   let df_dx = parsed_expression.diff("x");
///   let df_dy = parsed_expression.diff("y");
///  println!("df_dx = {}, df_dy = {}", df_dx, df_dy);
///  //convert symbolic expression to a Rust function and evaluate the function
///  let args = vec!["x","y"];
///  let function_of_x_and_y = parsed_expression.lambdify( args );
///  let f_res = function_of_x_and_y( (&[1.0, 2.0]).to_vec() );
///  println!("f_res = {}", f_res);
///  // or you dont want to pass arguments you can use lambdify_wrapped, arguments will be found inside function
///  let function_of_x_and_y = parsed_expression.lambdify_wrapped( );
///  let f_res = function_of_x_and_y( (&[1.0, 2.0]).to_vec() );
///  println!("f_res2 = {}", f_res);
///  let start = vec![ 1.0, 1.0];
///  let end = vec![ 2.0, 2.0];
///  // evaluate function of 2 or more arguments using linspace for defining vectors of arguments
///    let result = parsed_expression.lamdified_from_linspace(start.clone(), end.clone(), 10);
///    println!("evaluated function of 2 arguments = {:?}", result);
///   //  find vector of derivatives with respect to all arguments
///    let vector_of_derivatives = parsed_expression.diff_multi();
///    println!("vector_of_derivatives = {:?}, {}", vector_of_derivatives, vector_of_derivatives.len());
///   // compare numerical and analtical derivatives for a given linspace defined by start, end values and number of values.
///   // max_norm - maximum norm of the difference between numerical and analtical derivatives
///  let comparsion = parsed_expression.compare_num(start, end, 100, 1e-6);
///  println!(" result_of compare = {:?}", comparsion);
/// ```
/// Example3#
/// ```
/// use  RustedSciThe::symbolic::symbolic_engine::Expr;
///  use  RustedSciThe::symbolic::symbolic_functions::Jacobian;
///    let input = "log(x)";
///   let f = Expr::parse_expression(input);
///   //convert symbolic expression to a Rust function and evaluate the function
///   let f_res = f.lambdify1D()(1.0);
///   let df_dx = f.diff("x");
///   println!("df_dx = {}, log(1) = {}", df_dx, f_res);

///   let input = "x+exp(x)";
///   let f = Expr::parse_expression(input);
///   let f_res = f.lambdify1D()(1.0);
///   println!("f_res = {}", f_res);
///   let start = 0.0;
///   let end = 10 as f64;
///   let num_values = 100;
///   let max_norm = 1e-6;
///   // compare numerical and analtical derivatives for a given linspace defined by start, end values and number of values.
///  // a norm of the difference between the two of them is returned, and the answer is true if the norm is below max_norm
///   let (norm, res) = f.compare_num1D("x", start, end, num_values, max_norm);
///  println!("norm = {}, res = {}", norm, res);
/// ```
/// ________________________________________________________________________________________________________________________________________________
pub mod symbolic_engine;
pub mod symbolic_engine_derivatives;
///________________________________________________________________________________________________________________________________________________
///
/// calculate symbolic jacobian and evaluate it
/// Example#
/// ```
///  use  RustedSciThe::symbolic::symbolic_functions::Jacobian;
///  let mut Jacobian_instance = Jacobian::new();
///       // function of 2 or more arguments
///       let vec_of_expressions = vec![ "2*x^3+y".to_string(), "1".to_string()];
///       // set vector of functions
///      Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
///      // set vector of variables
///      Jacobian_instance.set_varvecor_from_str("x, y");
///      // calculate symbolic jacobian
///      Jacobian_instance.calc_jacobian();
///      // transform into human...kind of readable form
///      Jacobian_instance.readable_jacobian();
///      // generate jacobian made of regular rust functions
///      Jacobian_instance.jacobian_generate(vec!["x", "y"]);

///     println!("Jacobian_instance: functions  {:?}. Variables {:?}", Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables);
///      println!("Jacobian_instance: Jacobian  {:?} readable {:?}.", Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian);
///     for i in 0.. Jacobian_instance.symbolic_jacobian.len() {
///       for j in 0.. Jacobian_instance.symbolic_jacobian[i].len() {
///        println!("Jacobian_instance: Jacobian  {} row  {} colomn {:?}", i, j, Jacobian_instance.symbolic_jacobian[i][j]);
///      }  
///    }
///    // calculate element of jacobian (just for control)
///    let ij_element = Jacobian_instance.calc_ij_element(0, 0,  vec!["x", "y"],vec![10.0, 2.0]) ;
///    println!("ij_element = {:?} \n", ij_element);
///    // evaluate jacobian to numerical values
///     Jacobian_instance. lambdify_and_ealuate_funcvector(vec!["x", "y"], vec![10.0, 2.0]);
///     println!("function vector = {:?} \n", Jacobian_instance.evaluated_functions);
///    // lambdify and evaluate function vector to numerical values
///    Jacobian_instance. lambdify_and_ealuate_funcvector(vec!["x", "y"], vec![10.0, 2.0]);
///    println!("function vector = {:?} \n", Jacobian_instance.evaluated_functions);
///     // or first lambdify
///     Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
///     // then evaluate
///     Jacobian_instance.evaluate_funvector_lambdified(vec![10.0, 2.0]);
///     println!("function vector after evaluate_funvector_lambdified = {:?} \n", Jacobian_instance.evaluated_functions);
///        // evaluate jacobian to nalgebra matrix format
///     Jacobian_instance.evaluate_func_jacobian_DMatrix(vec![10.0, 2.0]);
///     println!("Jacobian_DMatrix = {:?} \n", Jacobian_instance.evaluated_jacobian_DMatrix);
///        Jacobian_instance.evaluate_funvector_lambdified_DVector(vec![10.0, 2.0]);
///     println!("function vector after evaluate_funvector_lambdified_DMatrix = {:?} \n", Jacobian_instance.evaluated_functions_DVector);
/// ```
pub mod symbolic_functions;
pub mod symbolic_functions_BVP;
//pub mod symbolic_functions_BVP_;
pub mod symbolic_traits;
///______________________________________________________________________________________________________________________________________________
/// the collection of utility functions mainly for bracket parsing and proceeding
/// _____________________________________________________________________________________________________________________________________________
pub mod utils;
