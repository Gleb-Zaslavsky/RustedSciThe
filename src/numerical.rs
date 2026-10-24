#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
//! Numerical solvers for BVPs, IVPs, nonlinear systems, optimization and
//! related scientific-computing tasks.
//!
//! For new universal IVP code, use [`ODE_api2`] with its typed
//! [`ODE_api2::SolverType`] selection, or use a dedicated solver module when solver-specific controls
//! are needed. The retired `ODE_api` and `NR_for_ODE` modules are intentionally
//! not kept as compatibility shims.
pub mod BDF;
/// Shared opt-in logging policy and timestamped file sink for numerical solvers.
pub mod logging;
pub use logging::{SolverLogGuard, SolverLogLevel, SolverLoggingError, solver_log_path};

/// Collection level shared by numerical solver telemetry APIs.
///
/// The modes separate the cost of integer event accounting from clock reads:
/// `Timings` measures durations only, while `Detailed` collects both counters
/// and timings. Solvers may expose only the levels they can implement without
/// misrepresenting their native instrumentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverTelemetryMode {
    /// Do not collect counters or timings.
    #[default]
    Off,
    /// Collect discrete work counters without reading the clock.
    Counters,
    /// Collect elapsed-time measurements without updating discrete counters.
    Timings,
    /// Collect both discrete counters and elapsed-time measurements.
    Detailed,
}

impl SolverTelemetryMode {
    /// Whether discrete event counters should be updated.
    #[inline]
    pub const fn collects_counters(self) -> bool {
        matches!(self, Self::Counters | Self::Detailed)
    }

    /// Whether elapsed-time scopes should read the monotonic clock.
    #[inline]
    pub const fn collects_timings(self) -> bool {
        matches!(self, Self::Timings | Self::Detailed)
    }

    /// Whether either telemetry channel is active.
    #[inline]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Backward Euler method (good for stiff ODEs)
///
/// Example#1
/// ```
///         use RustedSciThe::symbolic::symbolic_engine::Expr;
///         use RustedSciThe::numerical::BE::BE;
///         use nalgebra::DVector;
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
/// ```
pub mod BE;
#[cfg(test)]
pub(crate) mod ivp_test_support;
pub mod ivp_workloads;
/// Backward-Euler Newton implementation, retained at its historical path.
pub use BE::NR_for_Euler;
/// Boundary value problems solvers for stiff nonlinear ODEs
///
/// solver NR_Damp_solver_frozen implements NR method with damping
/// but without adaptive grid.
/// Frozen jacobian strategy - recalculating jacobian on condition:
/// Description of strategy                                                                 key of strategy        value user must provude for strategy
/// 1. only first time:                                                                      "Frozen_naive"                   None
/// 2. every m-th time, where m is a parameter of the strategy:                                "every_m"                    m
/// 3. every time when the solution norm greater than a certain threshold A:                   "at_high_norm".               A
/// 4. when norm of (i-1) iter multiplied by certain value B(<1) is lower than norm of i-th iter : "at_low_speed".            B
/// 5. complex - combined strategies 2,3,4                                                         "complex"              vec of  parameters [m, A, B]
///
///
/// Example#1
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
///       
///        let t0 = 0.0;
///        let t_end = 1.0;
///        let n_steps = 100; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
///        let strategy =   "Frozen".to_string();//
///        let  strategy_params = Some(HashMap::from([("complex".to_string(),
///       Some(Vec::from( [2f64, 5.0, 1e-1, ]  ))
///      )]));
/// //
///  // or
/// // Some(HashMap::from([("Frozen_naive".to_string(), None)]));
/// //  or
/// //  Some(HashMap::from([("every_m".to_string(),
/// //        Some(Vec::from( [ 5 as f64]  ))
/// //        )]));
/// //  or
/// //  Some(HashMap::from([("at_high_morm".to_string(),
/// // Some(Vec::from( [ 5 as f64]  ))
/// //)]));
/// //or
/// //Some(HashMap::from([("at_low_speed".to_string(),
/// // Some(Vec::from( [ 1e-2]  ))
/// //)]));
/// //or
/// // Some(HashMap::from([("complex".to_string(),
/// // Some(Vec::from( [ 2.0, 5.0, 1e-, ]  ))
/// //)]));
/// //  */
///        let method =   "Sparse".to_string();// or  "Dense"
///        let linear_sys_method = None;
///        let ones = vec![0.0; values.len()*n_steps];
///        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
///        let mut BorderConditions = HashMap::new();
///        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
///        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
///        assert_eq!(&eq_system.len(), &2);
///        let mut nr =  NRBVP::new(eq_system,
///             initial_guess,
///             values,
///             arg,
///             BorderConditions, t0, t_end, n_steps,strategy, strategy_params, linear_sys_method, method, tolerance, max_iterations);

///        println!("solving system");
///        #[allow(unused_variables)]
///        let solution = nr.solve().unwrap();
///       // println!("result = {:?}", solution);
///        nr.plot_result();
///    ```
///
/// NR_Damp_solver_damped solver implements
/// Modified Newton method or Damped Newton method with adaptive grid for solving Boundary value problems for systems of nonlinear ordinary differential equations.
///
/// The code mostly inspired by sources listed below:
/// Cantera MultiNewton solver (MultiNewton.cpp )
/// TWOPNT fortran solver (see "The Twopnt Program for Boundary Value Problems" by J. F. Grcar and Chemkin Theory Manual p.261)
///

/// Example#2
/// ```
/// use nalgebra::DMatrix;
///  use nalgebra::DVector;
///      use std::collections::HashMap;
///     use  RustedSciThe::symbolic::symbolic_engine::Expr;
/// use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::SolverParams;
///     use  RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP as NRBDVPd;
///     let eq1 = Expr::parse_expression("y-z");
///        let eq2 = Expr::parse_expression("-z^3");
///        let eq_system = vec![eq1, eq2];
///        let values = vec!["z".to_string(), "y".to_string()];
///        let arg = "x".to_string();
///        let tolerance = 1e-5;
///        let max_iterations = 20;
///        let t0 = 0.0;
///        let t_end = 1.0;
///        let n_steps = 50; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
///        let strategy =   "Damped".to_string();//
///        let  strategy_params = SolverParams::default();
/// let scheme = "forward".to_string();
///        let method =   "Sparse".to_string();// or  "Dense"
///        let linear_sys_method = None;
///        let ones = vec![0.0; values.len()*n_steps];
///        let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
///        let mut BorderConditions = HashMap::new();
///        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
///        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
///        let Bounds = HashMap::from([  ("z".to_string(), (-10.0, 10.0),    ), ("y".to_string(), (-7.0, 7.0),    ) ]);
///        let rel_tolerance =  HashMap::from([  ("z".to_string(), 1e-4    ), ("y".to_string(), 1e-4,    ) ]);
///        assert_eq!(&eq_system.len(), &2);
///        let mut nr =  NRBDVPd::new(eq_system,
///             initial_guess,
///             values,
///             arg,
///             BorderConditions, t0, t_end, n_steps,scheme, strategy, Some(strategy_params), linear_sys_method, method, tolerance, Some(rel_tolerance), max_iterations,  Some(Bounds),None);

///        println!("solving system");
///        #[allow(unused_variables)]
///        let solution = nr.solve().unwrap();
///       // println!("result = {:?}", solution);
///        nr.plot_result();
///    ```
/// API for both solvers
/// Example#3
/// ```
///  use nalgebra::DMatrix;
///  use nalgebra::DVector;
///      use std::collections::HashMap;
/// use RustedSciThe::numerical::BVP_api::BVP;
///     use  RustedSciThe::symbolic::symbolic_engine::Expr;
///     let eq1 = Expr::parse_expression("y-z");
///  let eq2 = Expr::parse_expression("-z^3");
/// let eq_system = vec![eq1, eq2];
///
///
/// let values = vec!["z".to_string(), "y".to_string()];
/// let arg = "x".to_string();
/// let tolerance = 1e-5;
/// let max_iterations = 20;
///
/// let t0 = 0.0;
/// let t_end = 1.0;
/// let n_steps = 50; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
///  let strategy =   "Damped".to_string();//
///
/// let  strategy_params =
/// match strategy.as_str() {
///     "Naive" => None,
///     "Damped"=> Some(HashMap::from([("max_jac".to_string(),
///     None,    ), ("maxDampIter".to_string(),
///    None,    ), ("DampFacor".to_string(),
///     None,    )   , ("adaptive".to_string(),
///           None,    )
///
///    ])),
///    "Frozen" => Some(HashMap::from([("every_m".to_string(),
///   Some(Vec::from( [ 5 as f64]  ))
///    )])),
///     &_=>panic!("Invalid strategy!")
///
///
/// };
/// let scheme = "forward".to_string();
/// let method =   "Sparse".to_string();// or  "Dense"
/// let linear_sys_method = None;
/// let ones = vec![0.0; values.len()*n_steps];
/// let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
/// let mut BorderConditions = HashMap::new();
/// BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
/// BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
/// let Bounds = HashMap::from([  ("z".to_string(), (-10.0, 10.0),    ), ("y".to_string(), (-7.0, 7.0),    ) ]);
/// let rel_tolerance =  HashMap::from([  ("z".to_string(), 1e-4    ), ("y".to_string(), 1e-4,    ) ]);
/// assert_eq!(&eq_system.len(), &2);
/// let mut nr =  BVP::new(eq_system,
///      initial_guess,
///      values,
///      arg,
///      BorderConditions, t0, t_end, n_steps, scheme, strategy, strategy_params, linear_sys_method, method, tolerance,
///        max_iterations,  Some(rel_tolerance),Some(Bounds), None);
///
/// println!("solving system");
/// nr.solve();
/// nr.plot_result();
/// nr.save_to_file(None);
///    ```
pub mod BVP_Damp;
//
pub mod BVP_api;
/// Typed solver-discovery facade. Prefer this for quick cross-solver
/// experiments; use the solver-specific modules for detailed configuration.
pub use BVP_api::{
    BvpApiError, BvpBoundaryCondition, BvpBoundarySide, BvpMatrixBackend, BvpProblem, BvpSolution,
    BvpSolver, BvpSolverBuilder, BvpSolverKind, BvpSolverOptions,
};
/// BVP solver with collocation method
/// rewritten in Rust from Python code from SciPy
pub mod BVP_sci;
///  a collection of test examples of exect solutions of BVPs for testing purposes
pub mod Examples_and_utils;

pub mod LSODE2;
/// RK45 and Dormand-Prince methods
pub mod NonStiff_api;
/// solvers of nonlinear algebraic equations
pub mod Nonlinear_systems;
/// Universal typed API for the supported IVP solver routes.
///
/// Use this for a common interface across methods; use dedicated modules such
/// as `BDF`, `BE`, `LSODE2`, or `Radau` for solver-specific configuration.
pub mod ODE_api2;
/// Radau solver for nonlinear and stiff ODEs.
///
/// The stable entry point is `RadauSolver`; the archived `Radau_main` module
/// is not part of the public API. See `examples/radau_public_api_guide.rs`.
///
/// ```
/// use RustedSciThe::numerical::Radau::{
///     RadauConfig, RadauFrontend, RadauProblem, RadauSolver,
/// };
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
///
/// let problem = RadauProblem::new(
///     vec![Expr::parse_expression("-y")],
///     vec!["y".to_owned()],
///     "t",
/// );
/// let config = RadauConfig {
///     frontend: RadauFrontend::ExprLegacy,
///     t_bound: 1.0,
///     ..RadauConfig::default()
/// };
/// let mut solver = RadauSolver::prepare(problem, config)?;
/// let solution = solver.solve(&[1.0])?;
/// assert!((solution.y[0] - (-1.0f64).exp()).abs() < 1e-5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub mod Radau;
pub mod Rosenbrock;
/// shooting method for solving BVP
pub mod ShootingBVP;
pub mod data_processing;
/// Shared interpolation and extrapolation algorithms.
pub mod interpolation;
/// collection of optimization algorithms
pub mod optimization;
