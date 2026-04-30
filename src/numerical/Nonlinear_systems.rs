pub mod LM_Nielsen;
pub mod LM_utils;
pub mod LM_vanilla;
pub mod NR_damped;
pub mod engine;
pub mod error;
///  Example#1
/// ```
///
/// use RustedSciThe::numerical::Nonlinear_systems::prelude::*;
/// use nalgebra::DVector;
///
/// let problem = SymbolicNonlinearProblem::from_strings(
///     vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
///     Some(vec!["x".to_string(), "y".to_string()]),
///     None,
///     None,
/// )
/// .expect("problem");
///
/// let result = SolverEngine::new(NewtonMethod, SolveOptions::default())
///     .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
///     .expect("solve");
///
/// assert!((result.x[0] - 3.0).abs() < 1e-8);
///  ```
/// Example#2
/// ```
/// use RustedSciThe::numerical::Nonlinear_systems::prelude::*;
/// use nalgebra::DVector;
///
/// let options = SymbolicProblemOptions::new()
///     .with_variables(vec!["x".to_string(), "y".to_string()])
///     .with_lambdify_backend();
///
/// let problem = SymbolicNonlinearProblem::from_strings_with_options(
///     vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
///     options,
/// )
/// .expect("problem");
///
/// let solve_options = SolveOptions {
///     tolerance: 1e-8,
///     max_iterations: 50,
///     ..SolveOptions::default()
/// };
///
/// let solution = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
///     .solve(&problem, DVector::from_vec(vec![1.0, 1.0]), solve_options)
///     .expect("solve");
///
/// assert!((solution.x[0] - 3.0).abs() < 1e-6);
///  ```
pub mod nonlinear_solver_tests;
pub mod prelude;
pub mod problem;
pub mod symbolic;
pub mod symbolic_aot;
pub mod symbolic_aot_lifecycle_tests;
pub mod symbolic_aot_solver_tests;
pub mod symbolic_aot_test_support;
pub mod symbolic_backend;
pub mod symbolic_generated;
pub mod trust_region;
pub mod trust_region_LM;
pub mod trust_region_lmpar;
