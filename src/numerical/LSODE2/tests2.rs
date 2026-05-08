use super::{
    Lsode2ProblemConfig, Lsode2Solver, Lsode2ResidualJacobianSource,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode, Lsode2ControllerConfig,
};
use super::config::*;
use crate::symbolic::symbolic_engine::Expr;
use std::time::Instant;
use nalgebra::{DVector};

/// Rising-temperature second-order kinetics test (pure symbolic lambdify).
///
/// Reaction: dy/dt = -k(t) * y^2, with k(t) = A * exp(-E/(R*(T0 + beta*t))).
/// We enable the automatic Adams/BDF controller and expect initial stiff
/// behaviour (BDF activity) and later non-stiff plateauing where Adams is
/// preferred.
fn rising_temperature_2a_b_c_config(backend:Lsode2ResidualJacobianSource) -> Lsode2ProblemConfig {
    // 2A -> B -> C mechanism with temperature-dependent second-order rate
    // dA/dt = -k(t) * A^2
    // dB/dt = 0.5*k(t) * A^2 - k2 * B
    // dC/dt = k2 * B
    let eqs = vec![
        Expr::parse_expression("-k0*exp(-E/(R*(T0 + t*beta)))*A*A"),
        Expr::parse_expression("0.5*k0*exp(-E/(R*(T0 + t*beta)))*A*A - k2*exp(-E/(R*(T0 + t*beta)))*B"),
        Expr::parse_expression("k2*exp(-E/(R*(T0 + t*beta)))*B"),
    ];

    Lsode2ProblemConfig::new(
        eqs,
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0, 0.0]),
        200.0,
        1.0,
        1e-6,
        1e-8,
    )
    .with_equation_parameters(vec![
        "k0".to_string(),
        "E".to_string(),
        "R".to_string(),
        "T0".to_string(),
        "beta".to_string(),
        "k2".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![1.0e6, 8.0e4, 8.314, 300.0, 1.0, 1.0e6]))
    .with_residual_jacobian_source(backend)
    // enable automatic switching and use a responsive probe window
    .with_automatic_adams_bdf_controller()
    .with_controller(Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(3))
}

#[test]
fn lsode2_rising_temperature_2a_b_c_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(rising_temperature_2a_b_c_config(

        Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    }
    ))
        .expect("build rising-temperature 2A->B->C config");

    let summary = solver
        .solve_with_summary()
        .expect("rising-temperature solve should finish");

    // Ensure symbolic lambdify route and switching enabled
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled);

    // Evidence of BDF activity during initial stiff phase
    assert!(summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
        "expected BDF activity in the stiff phase");

    // Record final algorithm snapshot (may prefer/adopt Adams in plateau)
    let algo = &summary.algorithm;
    // require Adams to be selected/preferred/activated at the end
    assert!(
        algo.preferred_family == "adams" || algo.active_family == "adams" || algo.executed_family == Some("adams"),
        "expected Adams to be selected in plateau region"
    );
    // final plateau check on A
    let (_t, y) = solver.get_result();
    let last = y[(y.nrows() - 1, 0)];
    let prev = y[(y.nrows() - 2, 0)];
    assert!((last - prev).abs() < 1e-4, "final A should be near-constant");
    let duration = start.elapsed();
    println!("rising-temperature-2a-b-c-switches-bdf-to-adams test completed in {:?}", duration.as_millis());
}

fn rising_temperature_second_order_config(backend: Lsode2ResidualJacobianSource) -> Lsode2ProblemConfig {
    // Single-species: dA/dt = -k(t) * A^2 with external linear T rise T = T0 + beta*t
    let expr = Expr::parse_expression("-k0*exp(-E/(R*(T0 + t*beta)))*y*y");

    Lsode2ProblemConfig::new(
        vec![expr],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        200.0,
        1.0,
        1e-6,
        1e-8,
    )
    .with_equation_parameters(vec![
        "k0".to_string(),
        "E".to_string(),
        "R".to_string(),
        "T0".to_string(),
        "beta".to_string(),
    ])
    // tuned parameters to produce a stiff initial burn and later plateau
    .with_equation_parameter_values(DVector::from_vec(vec![1.0e7, 8.0e4, 8.314, 300.0, 1.0]))
    .with_residual_jacobian_source(backend  ) 
    .with_automatic_adams_bdf_controller()
    .with_controller(Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1))
}

#[test]
fn lsode2_rising_temperature_second_order_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(rising_temperature_second_order_config(
        Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
     }
    ))
        .expect("build rising-temperature simple config");

    let summary = solver
        .solve_with_summary()
        .expect("rising-temperature simple solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled, "switching should be enabled");
    assert!(summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
        "expected BDF activity in the stiff phase");

    let algo = &summary.algorithm;
    // require Adams to be selected/preferred/activated at the end
    assert!(
        algo.preferred_family == "adams" || algo.active_family == "adams" || algo.executed_family == Some("adams"),
        "expected Adams to be selected in plateau region"
    );

    let (_t, y) = solver.get_result();
    let last = y[(y.nrows() - 1, 0)];
    let prev = y[(y.nrows() - 2, 0)];
    assert!((last - prev).abs() < 1e-4, "final value should be near-constant");
    let duration = start.elapsed();
    println!("test duration: {:?}", duration.as_millis());
}

fn one_d_solid_combustion_config(backend: Lsode2ResidualJacobianSource) -> Lsode2ProblemConfig {
    // 1D solid-like combustion in spatial coordinate x
    // Variables: A (fuel), B (product), q (heat flow), T (temperature)
    // Kinetics (first order in A): k*exp(-E/(R*T))*A
    // dA/dx = -k*exp(-E/(R*T))*A / u
    // dB/dx =  k*exp(-E/(R*T))*A / u
    // dq/dx = c_ro*u*q/Lambda + Q*k*exp(-E/(R*T))*A
    // dT/dx = q / Lambda
    let eqs = vec![
        Expr::parse_expression("-k*exp(-E/(R*T))*A / u"),
        Expr::parse_expression("k*exp(-E/(R*T))*A / u"),
        Expr::parse_expression("cro*u*q / Lambda + Q*k*exp(-E/(R*T))*A"),
        Expr::parse_expression("q / Lambda"),
    ];

    Lsode2ProblemConfig::new(
        eqs,
        vec!["A".to_string(), "B".to_string(), "q".to_string(), "T".to_string()],
        "x".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0, 0.0, 300.0]),
        100.0,   // spatial domain length
        0.05,  // max step in x
        1e-6,
        1e-8,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "T0".to_string(),
        "u".to_string(),
        "cro".to_string(),
        "Lambda".to_string(),
        "Q".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![
        1.0e7, // k
        5.0e4, // E
        8.314, // R
        300.0, // T0
        0.1,   // u (flame rate)
        10.0,   // cro (heat capacity)
        1.0,   // Lambda
        5.0e2, // Q
    ]))
    .with_residual_jacobian_source(backend)
    .with_automatic_adams_bdf_controller()
    //.with_bridge_solve()
    .with_controller(Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1))
}
#[test]
fn lsode2_1d_solid_combustion_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(one_d_solid_combustion_config(
        Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
     }
    )).expect("build 1D solid combustion config");
    let summary = solver.solve_with_summary().expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled, "switching should be enabled");
    assert!(summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
        "expected BDF activity during steep reaction front");

    // require Adams to be selected/preferred/activated at the end
    let algo = &summary.algorithm;


    // final plateau check: A should be near-constant at domain end
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 3)];
    let prev_T = y[(y.nrows() - 2, 3)];

    assert!((last_T - prev_T).abs() < 1e-4, "T should be near-constant at domain end");
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!((last_A - prev_A).abs() < 1e-4, "A should be near-constant at domain end");
    assert!(
        algo.preferred_family == "adams" || algo.active_family == "adams" || algo.executed_family == Some("adams"),
        "expected Adams to be selected in plateau region"
    );
    let duration = start.elapsed();
    println!("1D solid combustion test duration: {:?}", duration.as_millis());
}

//#[test]
fn lsode2_1d_solid_combustion_switches_bdf_to_adams_aot() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(one_d_solid_combustion_config(
        Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::Aot { toolchain: Lsode2AotToolchain::Rust, 
            profile:Lsode2AotProfile:: Debug }
     }
    )).expect("build 1D solid combustion config");
    let summary = solver.solve_with_summary().expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled, "switching should be enabled");
    assert!(summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
        "expected BDF activity during steep reaction front");

    // require Adams to be selected/preferred/activated at the end
    let algo = &summary.algorithm;


    // final plateau check: A should be near-constant at domain end
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 3)];
    let prev_T = y[(y.nrows() - 2, 3)];

    assert!((last_T - prev_T).abs() < 1e-4, "T should be near-constant at domain end");
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!((last_A - prev_A).abs() < 1e-4, "A should be near-constant at domain end");
    assert!(
        algo.preferred_family == "adams" || algo.active_family == "adams" || algo.executed_family == Some("adams"),
        "expected Adams to be selected in plateau region"
    );
    let duration = start.elapsed();
    println!("1D solid combustion test duration: {:?}", duration.as_millis());
}
fn combustion_like_ivp_config(backend: Lsode2ResidualJacobianSource) -> Lsode2ProblemConfig {
    // Variables: A (fuel), B (product), T (temperature)
    let eqs = vec![
        Expr::parse_expression("-k*exp(-E/(R*T))*A*A"),
        Expr::parse_expression("0.5*k*exp(-E/(R*T))*A*A - kloss*B"),
        Expr::parse_expression("Qcrho*k*exp(-E/(R*T))*A*A - cooling*(T - T0)"),
    ];

    let cfg = Lsode2ProblemConfig::new(
        eqs,
        vec!["A".to_string(), "B".to_string(), "T".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0, 300.0]),
        200.0,
        0.5,
        1e-6,
        1e-8,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "T0".to_string(),
        "Qcrho".to_string(),
        "kloss".to_string(),
        "cooling".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![
        1.0e7, // k0
        5.0e4, // E
        8.314, // R
        300.0, // T0
        5.0e2, // Q_over_crho (heat release per reaction scaled by c*rho)
        0.0,   // k_loss
        0.5,   // cooling
    ]))
    .with_residual_jacobian_source(backend);

    // Enable automatic switching and nudge the probe steps to be responsive
    cfg.with_automatic_adams_bdf_controller()
        .with_controller(Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(3))
}

#[test]
fn lsode2_combustion_like_stiff_regression_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(combustion_like_ivp_config(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })).expect("build combustion-like config");
    let summary = solver.solve_with_summary().expect("combustion-like solve should finish");

    // symbolic lambdify route
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled, "switching should be enabled");

    // expect significant BDF activity during burn
    assert!(summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
        "expected BDF activity during stiff burning phase");

    // Record final algorithm snapshot (may prefer/adopt Adams in plateau)
    let algo = &summary.algorithm;
    // require Adams to be selected/preferred/activated at the end
    assert!(
        algo.preferred_family == "adams" || algo.active_family == "adams" || algo.executed_family == Some("adams"),
        "expected Adams to be selected in plateau region"
    );
    // final plateau check: temperature and A should change little at the end
    let (t, y) = solver.get_result();
    assert!(t.len() >= 2, "expected time-history length");
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    assert!((last_A - prev_A).abs() < 1e-3, "A should be near-constant at the end");
    let duration = start.elapsed();
    println!("test duration: {:?}", duration.as_millis());
}

#[test]
fn lsode2_combustion_like_stiff_regression_switches_bdf_to_adams_aot() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(combustion_like_ivp_config(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::Aot { toolchain: Lsode2AotToolchain::Zig, 
            profile:Lsode2AotProfile:: Release }
    }
)).expect("build combustion-like config");
    let summary = solver.solve_with_summary().expect("combustion-like solve should finish");

    // symbolic lambdify route
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled, "switching should be enabled");

    // expect significant BDF activity during burn
    assert!(summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
        "expected BDF activity during stiff burning phase");

    // Record final algorithm snapshot (may prefer/adopt Adams in plateau)
    let algo = &summary.algorithm;
    // require Adams to be selected/preferred/activated at the end
    assert!(
        algo.preferred_family == "adams" || algo.active_family == "adams" || algo.executed_family == Some("adams"),
        "expected Adams to be selected in plateau region"
    );
    // final plateau check: temperature and A should change little at the end
    let (t, y) = solver.get_result();
    assert!(t.len() >= 2, "expected time-history length");
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    assert!((last_A - prev_A).abs() < 1e-3, "A should be near-constant at the end");
    let duration = start.elapsed();
    println!("test duration: {:?}", duration.as_millis());
}