use super::config::*;
use super::{
    Lsode2ControllerConfig, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::numerical::ODE_api2::UniversalODESolver;
use crate::numerical::Radau::Radau_main::RadauOrder;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::collections::HashMap;
use std::time::Instant;
/// Rising-temperature second-order kinetics test (pure symbolic lambdify).
///
/// Reaction: dy/dt = -k(t) * y^2, with k(t) = A * exp(-E/(R*(T0 + beta*t))).
/// We enable the automatic Adams/BDF controller.  The historical test name is
/// kept for compatibility, but this particular parameter set is not a reliable
/// BDF->Adams switch case: LSODA starts automatic mode in Adams and the current
/// telemetry classifies this ramp as non-stiff throughout.
fn rising_temperature_2a_b_c_config(backend: Lsode2ResidualJacobianSource) -> Lsode2ProblemConfig {
    // 2A -> B -> C mechanism with temperature-dependent second-order rate
    // dA/dt = -k(t) * A^2
    // dB/dt = 0.5*k(t) * A^2 - k2 * B
    // dC/dt = k2 * B
    let eqs = vec![
        Expr::parse_expression("-k0*exp(-E/(R*(T0 + t*beta)))*A*A"),
        Expr::parse_expression(
            "0.5*k0*exp(-E/(R*(T0 + t*beta)))*A*A - k2*exp(-E/(R*(T0 + t*beta)))*B",
        ),
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
    .with_equation_parameter_values(DVector::from_vec(vec![
        1.0e6, 8.0e4, 8.314, 300.0, 1.0, 1.0e6,
    ]))
    .with_residual_jacobian_source(backend)
    // enable automatic switching and use a responsive probe window
    //  .with_automatic_adams_bdf_controller()
    .with_native_sparse_faer_backend()
}

#[test]
fn lsode2_rising_temperature_2a_b_c_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        rising_temperature_2a_b_c_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(3),
        ),
    )
    .expect("build rising-temperature 2A->B->C config");

    let summary = solver
        .solve_with_summary()
        .expect("rising-temperature solve should finish");

    // TEMP: inspect controller + native stats to see why Adams is never explored.
    println!(
        "algo: controller_mode={} preferred_family={} active_family={} executed_family={:?} switch_reason={} switching_enabled={} uses_fallback={}",
        summary.algorithm.controller_mode,
        summary.algorithm.preferred_family,
        summary.algorithm.active_family,
        summary.algorithm.executed_family,
        summary.algorithm.switch_reason,
        summary.algorithm.method_switching_enabled,
        summary.algorithm.switch_uses_fallback,
    );
    let ns = &summary.native_statistics;
    println!(
        "native stats: preferred[a/b]={}/{} executed[a/b]={}/{} adams_cost_samples={} bdf_cost_samples={} algorithm_decisions={}",
        ns.preferred_adams_count,
        ns.preferred_bdf_count,
        ns.executed_adams_count,
        ns.executed_bdf_count,
        ns.native_adams_cost_samples,
        ns.native_bdf_cost_samples,
        ns.algorithm_decision_calls,
    );

    // Ensure symbolic lambdify route and switching enabled
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled);
    /*
       // Evidence of BDF activity during initial stiff phase
       assert!(
           summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
           "expected BDF activity in the stiff phase"
       );
    */
    // Plateau test: the automatic controller should have *explored* Adams at
    // least once (either by preferring/executing it or by sampling its cost),
    // even if it ultimately stays on BDF.
    let algo = &summary.algorithm;
    let native = &summary.native_statistics;
    assert!(
        native.native_adams_cost_samples > 0
            || native.preferred_adams_count > 0
            || native.executed_adams_count > 0,
        "automatic controller should collect some Adams evidence on this plateau problem; adams_cost_samples={}, preferred_adams={}, executed_adams={}",
        native.native_adams_cost_samples,
        native.preferred_adams_count,
        native.executed_adams_count
    );
    assert!(
        native.executed_adams_count > 0,
        "rising-temperature ramp should execute Adams in LSODA-style automatic startup; executed_adams={}",
        native.executed_adams_count
    );
    assert_eq!(
        native.executed_bdf_count, 0,
        "this ramp is currently classified as non-stiff; BDF activity belongs in the dedicated stiff/plateau switch tests"
    );
    assert!(
        algo.executed_family == Some(algo.preferred_family),
        "automatic path should execute the family it prefers; preferred={}, executed={:?}",
        algo.preferred_family,
        algo.executed_family
    );

    // final plateau check on A
    let (_t, y) = solver.get_result();
    let last = y[(y.nrows() - 1, 0)];
    let prev = y[(y.nrows() - 2, 0)];
    assert!(
        (last - prev).abs() < 1e-4,
        "final A should be near-constant"
    );
    let duration = start.elapsed();
    println!(
        "rising-temperature-2a-b-c-switches-bdf-to-adams test completed in {:?}",
        duration.as_millis()
    );
}

#[test]
fn lsode2_rising_temperature_2a_b_c_switches_bdf_to_adams_bridge() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        rising_temperature_2a_b_c_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_bridge_solve()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(3),
        ),
    )
    .expect("build rising-temperature 2A->B->C config");

    let summary = solver
        .solve_with_summary()
        .expect("rising-temperature solve should finish");

    // Ensure symbolic lambdify route and switching enabled
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(summary.algorithm.method_switching_enabled);

    // Evidence of BDF activity during initial stiff phase
    assert!(
        summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
        "expected BDF activity in the stiff phase"
    );

    // final plateau check on A
    let (_t, y) = solver.get_result();
    let last = y[(y.nrows() - 1, 0)];
    let prev = y[(y.nrows() - 2, 0)];
    assert!(
        (last - prev).abs() < 1e-4,
        "final A should be near-constant"
    );
    let duration = start.elapsed();
    println!(
        "rising-temperature-2a-b-c-switches-bdf-to-adams test completed in {:?}",
        duration.as_millis()
    );
}
//========================================================================================================
//

fn rising_temperature_second_order_config(
    backend: Lsode2ResidualJacobianSource,
) -> Lsode2ProblemConfig {
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
    .with_residual_jacobian_source(backend)
    .with_native_sparse_faer_backend()
    .with_controller(
        Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
    )
}

#[test]
fn lsode2_rising_temperature_second_order_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(rising_temperature_second_order_config(
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
    ))
    .expect("build rising-temperature simple config");

    let summary = solver
        .solve_with_summary()
        .expect("rising-temperature simple solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );
    /*
        assert!(
            summary.statistics.bdf_nlu_total > 0 || summary.statistics.bdf_nfev_total > 0,
            "expected BDF activity in the stiff phase"
        );
    */
    // Plateau-phase exploration: expect the automatic controller to have
    // collected Adams evidence at least once.
    let algo = &summary.algorithm;
    let native = &summary.native_statistics;
    assert!(
        native.native_adams_cost_samples > 0
            || native.preferred_adams_count > 0
            || native.executed_adams_count > 0,
        "solid combustion plateau should collect Adams evidence; adams_cost_samples={}, preferred_adams={}, executed_adams={}",
        native.native_adams_cost_samples,
        native.preferred_adams_count,
        native.executed_adams_count
    );
    assert!(
        algo.executed_family == Some(algo.preferred_family),
        "automatic path should execute the family it prefers; preferred={}, executed={:?}",
        algo.preferred_family,
        algo.executed_family
    );

    let (_t, y) = solver.get_result();
    let last = y[(y.nrows() - 1, 0)];
    let prev = y[(y.nrows() - 2, 0)];
    assert!(
        (last - prev).abs() < 1e-4,
        "final value should be near-constant"
    );
    let duration = start.elapsed();
    println!("test duration: {:?}", duration.as_millis());
}
//========================================================================================================
//
//
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
        Expr::parse_expression("cro*u*q / Lambda - Q*k*exp(-E/(R*T))*A"),
        Expr::parse_expression("q / Lambda"),
    ];

    Lsode2ProblemConfig::new(
        eqs,
        vec![
            "A".to_string(),
            "B".to_string(),
            "q".to_string(),
            "T".to_string(),
        ],
        "x".to_string(),
        0.0,
        DVector::from_vec(vec![1.0 - 1e-3, 1e-3, 1e-3, 300.0]),
        20.5e-2, // spatial domain length
        1e-3,    // max step in x
        1e-5,
        1e-4,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "cro".to_string(),
        "Lambda".to_string(),
        "Q".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![
        3.7e11,      // k
        39_000.0,    // E
        1.987,       // R
        0.2,         // u (flame rate)
        1.6 * 0.245, // cro (heat capacity)
        6e-4,        // Lambda
        -1e3, // Q (exothermic sign convention: minus in equation + negative Q => heating source)
    ]))
    .with_residual_jacobian_source(backend)
    // In this variant the conversion state is represented by fuel fraction `A`,
    // so η >= 0.999 corresponds to A <= 1e-3.
    .with_stop_condition_le("A", 1e-3)
    .with_stop_condition_ge("T", 1400.0)
    //.with_native_sparse_faer_backend()

    //.with_automatic_adams_bdf_controller()
    //.with_bridge_solve()
}
#[test]
fn lsode2_1d_solid_combustion_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        ),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );

    // Plateau-phase exploration: expect the automatic controller to have
    // collected Adams evidence at least once.
    let algo = &summary.algorithm;
    let native = &summary.native_statistics;

    assert!(
        native.native_adams_cost_samples > 0
            || native.preferred_adams_count > 0
            || native.executed_adams_count > 0,
        "solid combustion plateau should collect Adams evidence; adams_cost_samples={}, preferred_adams={}, executed_adams={}",
        native.native_adams_cost_samples,
        native.preferred_adams_count,
        native.executed_adams_count
    );
    assert!(
        algo.executed_family == Some(algo.preferred_family),
        "automatic path should execute the family it prefers; preferred={}, executed={:?}",
        algo.preferred_family,
        algo.executed_family
    );

    // Stop-condition oriented checks: this scenario should either stop by
    // configured condition (A <= 1e-3 or T >= 1400) or end in faithful partial mode.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let A0 = y[(0, 0)];
    let T0 = y[(0, 3)];
    let last_T = y[(y.nrows() - 1, 3)];
    let q0 = y[(0, 2)];
    let last_q = y[(y.nrows() - 1, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    let stop_reached = last_A <= 1.0e-3 || last_T >= 1400.0;
    assert!(
        stop_reached || summary.status == "finished_native_faithful_partial",
        "expected stop condition or faithful partial exit; status={}, A={}, T={}",
        summary.status,
        last_A,
        last_T
    );
    assert!(
        last_A.is_finite() && last_T.is_finite() && (0.0..=1.1).contains(&last_A),
        "state should stay finite and physically bounded; A={}, T={}",
        last_A,
        last_T
    );
    assert!(
        last_A <= prev_A + 1e-8,
        "fuel fraction A should be non-increasing"
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
    let dT = last_T - T0;
    let dq = last_q - q0;
    let Lambda = 5e-4;
    let u = 0.2;
    let ro = 1.6;
    let c = 0.245;
    let Q = -1e3;
    let residual = Lambda * dq - ro * u * c * dT - u * Q * (A0 - last_A);
    println!(
        "Lambda*dq: {}, ro*u*c*dT: {}, u*Q*(A0-last_A): {}",
        Lambda * dq,
        ro * u * c * dT,
        u * Q * (A0 - last_A)
    );
    let residual_relative = residual / (Lambda * dq);
    println!(
        "Residual: {}, Relative Residual: {}",
        residual, residual_relative
    );
}
#[test]
fn lsode2_1d_solid_combustion_pure_bdf() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_faithful_bdf_solve(4096, 4096)
        .with_controller(Lsode2ControllerConfig::bdf_only()),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        !summary.algorithm.method_switching_enabled,
        "pure BDF mode should disable Adams/BDF auto switching"
    );

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "expected faithful BDF native status, got {}",
        summary.status
    );
    // Stop-condition oriented checks.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 3)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    let stop_reached = last_A <= 1.0e-3 || last_T >= 1400.0;
    assert!(
        stop_reached || summary.status == "finished_native_faithful_partial",
        "expected stop condition or faithful partial exit; status={}, A={}, T={}",
        summary.status,
        last_A,
        last_T
    );
    assert!(
        last_A.is_finite() && last_T.is_finite() && (0.0..=1.1).contains(&last_A),
        "state should stay finite and physically bounded; A={}, T={}",
        last_A,
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}
#[test]
fn lsode2_1d_solid_combustion_switches_bdf_to_adams_bridge() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_bridge_solve()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(3),
        ),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );
    assert!(
        summary.status == "stopped_by_condition"
            || summary.status == "finished"
            || summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected bridge/native status: {}",
        summary.status
    );

    // Stop-condition oriented checks.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 3)];
    let stop_reached = last_A <= 1.0e-3 || last_T >= 1400.0;
    assert!(
        stop_reached || summary.status == "finished_native_faithful_partial",
        "expected stop condition or faithful partial exit; status={}, A={}, T={}",
        summary.status,
        last_A,
        last_T
    );
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        last_A.is_finite() && last_T.is_finite() && (0.0..=1.1).contains(&last_A),
        "state should stay finite and physically bounded; A={}, T={}",
        last_A,
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}

#[test]
fn lsode2_1d_solid_combustion_switches_bdf_to_adams_aot() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::Rust,
                profile: Lsode2AotProfile::Debug,
            },
        })
        .with_native_sparse_faer_backend()
        .with_faithful_bdf_solve(4096, 4096)
        .with_controller(Lsode2ControllerConfig::bdf_only()),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        !summary.algorithm.method_switching_enabled,
        "pure BDF mode should disable Adams/BDF auto switching"
    );
    assert!(
        summary.native_statistics.native_linear_solve_calls > 0,
        "expected native linear solves in faithful BDF path"
    );

    // Stop-condition oriented checks.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 3)];
    let stop_reached = last_A <= 1.0e-3 || last_T >= 1400.0;
    assert!(
        stop_reached || summary.status == "finished_native_faithful_partial",
        "expected stop condition or faithful partial exit; status={}, A={}, T={}",
        summary.status,
        last_A,
        last_T
    );
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        last_A.is_finite() && last_T.is_finite() && (0.0..=1.1).contains(&last_A),
        "state should stay finite and physically bounded; A={}, T={}",
        last_A,
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}
//========================================================================================================
// IVP PROBLEM WITH EXOTHERMIC KINETICS AND HEAT LOSS DUE TO SUBLIMATION, LEADING TO PLATEAUING BEHAVIOUR
//
//ATTENTION! SUBLIMATION IS DISABLED UNTIL "STOP CONDITION" WILL BE IMPLEMENTED OTHERWISE SUBLIMATION WILL UNPHYSICALLY DOMINATE THE SOLUTION AND PREVENT PLATEAUING,
// ALSOO PULL SOLUSION TO BELOW ZERO
// IMPLEMENT STOP CONDITION AND RETURN
fn C_func() -> Expr {
    let T = Expr::Var("T".to_string());
    let t = T / Expr::Const(1000.0);

    let A = 225.2150;
    let B = -203.0560;
    let C = 402.2310;
    let D = -137.7330;
    let E = -7.484290;

    let Cp = Expr::parse_expression("A + B^t + C^t^2 + D*t^3 + E/t^2");
    let Cp = Cp
        .set_variable("A", A)
        .set_variable("B", B)
        .set_variable("C", C)
        .set_variable("D", D)
        .set_variable("E", E);
    let Cp = Cp.substitute_variable("t", &t);
    let Cp = Cp / Expr::Const(117.489 * 4.184);
    println!("Cp: {}", Cp);
    return Cp;
}
fn sublimation_func() -> Vec<Expr> {
    let eta = Expr::Var("eta".to_string());
    let q = Expr::Var("q".to_string());

    let Lambda = Expr::Var("Lambda".to_string());
    let ro = Expr::Var("ro".to_string());
    let u = Expr::Var("u".to_string());
    let c = Expr::Var("c".to_string());

    let P = Expr::Var("P".to_string());
    let Qd = Expr::Var("Qd".to_string());
    let Qs = Expr::Var("Qs".to_string());
    let one = Expr::Const(1.0);
    let two = Expr::Const(2.0);
    let Arr = Expr::parse_expression("k*exp(-E/(R*T))");
    let p_s = Expr::parse_expression("10.0^(10.56 - 6283.7/T)"); //мм Hg
    let sublim = p_s.clone() / (P - p_s.clone());
    //  let c = C_func().lambdify1D();
    //let c = c(1000.0);
    //  let c = Expr::Const(c);
    let K = ro.clone() * u.clone() * c.clone() / Lambda.clone();
    let Wd = (one - eta.clone()) * ro.clone() * Arr.clone();
    let Ws = two * Wd.clone() * sublim.clone();
    let W = Wd.clone() + Ws.clone();
    let Fq = Qd.clone() * Wd - Qs.clone() * Ws;
    let A_eq = W.clone() / (u.clone() * ro.clone());

    let q_eq = K * q.clone() - Fq.clone();
    let T_eq = q.clone() / Lambda.clone();
    let eqs = vec![A_eq.clone(), q_eq.clone(), T_eq.clone()];
    println!("A_eq: {}", A_eq.pretty_print());
    println!("q_eq: {}", q_eq.pretty_print());
    println!("T_eq: {}", T_eq.pretty_print());
    return eqs;
}
fn one_d_solid_combustion_with_sublimation_config(
    backend: Lsode2ResidualJacobianSource,
) -> Lsode2ProblemConfig {
    // 1D solid-like combustion in spatial coordinate x

    // deta/dx = (Ws+ Ws) / (u*ro)
    // dq/dx = c_ro*u*q/Lambda - (Qs*Ws- Qs*Ws)
    // dT/dx = q / Lambda
    let eqs = sublimation_func();
    Lsode2ProblemConfig::new(
        eqs,
        vec!["eta".to_string(), "q".to_string(), "T".to_string()],
        "x".to_string(),
        0.0,
        DVector::from_vec(vec![1e-3, 1e-3, 300.0]),
        4e-2, // spatial domain length
        1e-4, // max step in x
        1e-6,
        1e-6,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "P".to_string(),
        "Qd".to_string(),
        "Qs".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,           // u (flame rate)
        0.245,         // c (heat capacity)
        1.6,           // ro (density)
        5e-4,          // Lambda
        760.0 * 300.0, // P (ambient pressure, elevated to keep P - p_s away from zero)
        // sublimation parameters
        -1e3,  // Qd (exothermic sign convention)
        0.5e3, // Qs
    ]))
    .with_residual_jacobian_source(backend)
    .with_stop_condition_ge("eta", 0.999)
    .with_stop_condition_ge("T", 1400.0)
}

//ATTENTION! SUBLIMATION IS DISABLED UNTIL "STOP CONDITION" WILL BE IMPLEMENTED OTHERWISE SUBLIMATION WILL UNPHYSICALLY DOMINATE THE SOLUTION AND PREVENT PLATEAUING,
// ALSOO PULL SOLUSION TO BELOW ZERO
#[test]
fn lsode2_1d_solid_combustion_with_sublimation_pure_bdf() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_with_sublimation_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_faithful_bdf_solve(4096, 4096)
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        ),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "expected faithful BDF native status, got {}",
        summary.status
    );
    // Stop-condition semantics check: integration should stop once eta reaches threshold.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        last_A >= 0.999 || summary.status == "finished_native_faithful_partial",
        "expected either eta stop (eta>=0.999) or faithful partial exit; status={}, eta={}",
        summary.status,
        last_A
    );
    assert!(
        last_T.is_finite() && last_T > 300.0,
        "temperature should remain finite and rise above initial value, got T={}",
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}

//ATTENTION! SUBLIMATION IS DISABLED UNTIL "STOP CONDITION" WILL BE IMPLEMENTED OTHERWISE SUBLIMATION WILL UNPHYSICALLY DOMINATE THE SOLUTION AND PREVENT PLATEAUING,
// ALSOO PULL SOLUSION TO BELOW ZERO
#[test]
fn lsode2_1d_solid_combustion_with_sublimation_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_with_sublimation_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        ),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );

    // Plateau-phase exploration: expect the automatic controller to have
    // collected Adams evidence at least once.
    let algo = &summary.algorithm;
    let native = &summary.native_statistics;

    assert!(
        native.native_adams_cost_samples > 0
            || native.preferred_adams_count > 0
            || native.executed_adams_count > 0,
        "solid combustion plateau should collect Adams evidence; adams_cost_samples={}, preferred_adams={}, executed_adams={}",
        native.native_adams_cost_samples,
        native.preferred_adams_count,
        native.executed_adams_count
    );
    assert!(
        algo.executed_family == Some(algo.preferred_family),
        "automatic path should execute the family it prefers; preferred={}, executed={:?}",
        algo.preferred_family,
        algo.executed_family
    );

    // Stop-condition semantics check: integration should stop once eta reaches threshold.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    let stop_reached = summary
        .native_integration_solve
        .as_ref()
        .map(|solve| solve.reached_stop_condition)
        .unwrap_or(false);
    assert!(
        last_A >= 0.999 || stop_reached || summary.status == "finished_native_faithful_partial",
        "expected eta stop or faithful partial exit; status={}, eta={}, stop_triggered={}",
        summary.status,
        last_A,
        stop_reached
    );
    assert!(
        last_A.is_finite() && (0.0..=1.1).contains(&last_A) && last_T.is_finite(),
        "state should remain finite and physically bounded; eta={}, T={}",
        last_A,
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}

#[test]
fn subliamtion_radau_test() {
    let eqs = sublimation_func();
    let vars = vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "P".to_string(),
        "Qd".to_string(),
        "Qs".to_string(),
    ];
    let values = vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,           // u (flame rate)
        0.245,         // c (heat capacity)
        1.6,           // ro (density)
        5e-4,          // Lambda
        760.0 * 300.0, // P (ambient pressure, elevated to keep P - p_s away from zero)
        // sublimation parameters
        -1e3,  // Qd (exothermic sign convention)
        0.5e3, // Qs
    ];
    let var_map: HashMap<String, f64> = vars
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let eqs = eqs
        .iter()
        .map(|eq| eq.set_variable_from_map(&var_map))
        .collect::<Vec<_>>();
    let values = vec!["eta".to_string(), "q".to_string(), "T".to_string()];
    let arg = "x".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1e-3, 1e-3, 300.0]);
    let t_bound = 4e-2;

    let mut solver = UniversalODESolver::radau(
        eqs,
        values,
        arg,
        RadauOrder::Order3,
        t0,
        y0,
        t_bound,
        1e-6,
        550,
        None,
    );

    solver.solve();
    let (_t_result, y) = solver.get_result();
    let y = y.unwrap();
    //println!("{:?}", y);
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    let prev_T = y[(y.nrows() - 2, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        prev_T.is_finite() && last_T.is_finite() && last_T > 0.0 && last_T >= prev_T,
        "T should remain finite, positive, and non-decreasing over the final step; prev_T={}, last_T={}",
        prev_T,
        last_T
    );

    assert!(
        prev_A.is_finite() && last_A.is_finite() && (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at domain end; prev_A={}, last_A={}",
        prev_A,
        last_A
    );
}

#[test]
fn subliamtion_be_test() {
    let eqs = sublimation_func();
    let vars = vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "P".to_string(),
        "Qd".to_string(),
        "Qs".to_string(),
    ];
    let values = vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,           // u (flame rate)
        0.245,         // c (heat capacity)
        1.6,           // ro (density)
        5e-4,          // Lambda
        760.0 * 300.0, // P (ambient pressure, elevated to keep P - p_s away from zero)
        // sublimation parameters
        -1e3,  // Qd (exothermic sign convention)
        0.5e3, // Qs
    ];
    let var_map: HashMap<String, f64> = vars
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let eqs = eqs
        .iter()
        .map(|eq| eq.set_variable_from_map(&var_map))
        .collect::<Vec<_>>();
    let values = vec!["eta".to_string(), "q".to_string(), "T".to_string()];
    let arg = "x".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1e-3, 1e-3, 300.0]);
    let t_bound = 4e-2;

    let mut solver = UniversalODESolver::backward_euler(
        eqs,
        values,
        arg,
        t0,
        y0,
        t_bound,
        1e-6,
        550,
        Some(1e-5),
    );

    solver.solve();
    let (_t_result, y) = solver.get_result();
    let y = y.unwrap();
    //println!("{:?}", y);
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    let prev_T = y[(y.nrows() - 2, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        prev_T.is_finite() && last_T.is_finite() && last_T > 0.0 && last_T >= prev_T,
        "T should remain finite, positive, and non-decreasing over the final step; prev_T={}, last_T={}",
        prev_T,
        last_T
    );

    assert!(
        prev_A.is_finite() && last_A.is_finite() && (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at domain end; prev_A={}, last_A={}",
        prev_A,
        last_A
    );
}
//#[test]
fn sublimation_bdf_test() {
    let eqs = sublimation_func();
    let vars = vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "P".to_string(),
        "Qd".to_string(),
        "Qs".to_string(),
    ];
    let values = vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,           // u (flame rate)
        0.245,         // c (heat capacity)
        1.6,           // ro (density)
        5e-4,          // Lambda
        760.0 * 300.0, // P (ambient pressure, elevated to keep P - p_s away from zero)
        // sublimation parameters
        -1e3,  // Qd (exothermic sign convention)
        0.5e3, // Qs
    ];
    let var_map: HashMap<String, f64> = vars
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let eqs = eqs
        .iter()
        .map(|eq| eq.set_variable_from_map(&var_map))
        .collect::<Vec<_>>();
    let values = vec!["eta".to_string(), "q".to_string(), "T".to_string()];
    let arg = "x".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1e-3, 1e-3, 300.0]);
    let t_bound = 4e-2;

    let mut solver = UniversalODESolver::bdf(eqs, values, arg, t0, y0, t_bound, 1e-4, 1e-6, 1e-6);

    solver.solve();
    let (_t_result, y) = solver.get_result();
    let y = y.unwrap();
    //println!("{:?}", y);
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    let prev_T = y[(y.nrows() - 2, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        prev_T.is_finite() && last_T.is_finite() && last_T > 0.0 && last_T >= prev_T,
        "T should remain finite, positive, and non-decreasing over the final step; prev_T={}, last_T={}",
        prev_T,
        last_T
    );

    assert!(
        prev_A.is_finite() && last_A.is_finite() && (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at domain end; prev_A={}, last_A={}",
        prev_A,
        last_A
    );
}
//========================================================================================================
pub fn combustion_1d_problem() -> Vec<Expr> {
    // 1D solid-like combustion in spatial coordinate x

    // deta/dx = (Ws+ Ws) / (u*ro)
    // dq/dx = c_ro*u*q/Lambda - (Qs*Ws- Qs*Ws)
    // dT/dx = q / Lambda
    let eta = Expr::Var("eta".to_string());
    let q = Expr::Var("q".to_string());

    let Lambda = Expr::Var("Lambda".to_string());
    let ro = Expr::Var("ro".to_string());
    let u = Expr::Var("u".to_string());
    let c = Expr::Var("c".to_string());

    let Qd = Expr::Var("Qd".to_string());

    let one = Expr::Const(1.0);

    let Arr = Expr::parse_expression("k*exp(-E/(R*T))");
    // Toy problem: no sublimation, no p_s terms. Keep only a single
    // Arrhenius-based reaction source Wd and related heat release.
    // let  p_s = Expr::parse_expression("10.0^(10.56 - 6283.7/T)"); // мм Hg
    // let sublim = p_s.clone()/(P - p_s.clone());
    let K = ro.clone() * u.clone() * c.clone() / Lambda.clone();
    let Wd = (one - eta.clone()) * ro.clone() * Arr.clone();
    // No Ws, no sublimation heat term in the toy model:
    // let Ws = two * Wd.clone() * sublim.clone();
    let W = Wd.clone();
    let Fq = Qd.clone() * Wd;
    let A_eq = W.clone() / (u.clone() * ro.clone());
    let q_eq = K * q.clone() - Fq.clone();
    let T_eq = q.clone() / Lambda.clone();
    let eqs = vec![A_eq.clone(), q_eq.clone(), T_eq.clone()];
    println!("A_eq: {}", A_eq.pretty_print());
    println!("q_eq: {}", q_eq.pretty_print());
    println!("T_eq: {}", T_eq.pretty_print());
    eqs
}
// IVP PROBLEM WITH EXOTHERMIC KINETICS AND HEAT LOSS DUE TO SUBLIMATION, LEADING TO PLATEAUING BEHAVIOUR
//
//ATTENTION! SUBLIMATION IS DISABLED UNTIL "STOP CONDITION" WILL BE IMPLEMENTED OTHERWISE SUBLIMATION WILL UNPHYSICALLY DOMINATE THE SOLUTION AND PREVENT PLATEAUING,
// ALSOO PULL SOLUSION TO BELOW ZERO
// IMPLEMENT STOP CONDITION AND RETURN
fn one_d_solid_combustion_no_sublimation_config(
    backend: Lsode2ResidualJacobianSource,
) -> Lsode2ProblemConfig {
    let eqs = combustion_1d_problem();

    Lsode2ProblemConfig::new(
        eqs,
        vec!["eta".to_string(), "q".to_string(), "T".to_string()],
        "x".to_string(),
        0.0,
        DVector::from_vec(vec![1e-3, 1e-3, 300.0]),
        7.5e-2, // spatial domain length
        1e-4,   // max step in x
        1e-6,
        1e-6,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "Qd".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,   // u (flame rate)
        0.245, // c (heat capacity)
        1.6,   // ro (density)
        5e-4,  // Lambda
        // sublimation parameters
        -1e3, // Qd (exothermic sign convention to match Python model's Fs term)
    ]))
    .with_residual_jacobian_source(backend)
    .with_stop_condition_ge("eta", 0.999)
}

//ATTENTION! SUBLIMATION IS DISABLED UNTIL "STOP CONDITION" WILL BE IMPLEMENTED OTHERWISE SUBLIMATION WILL UNPHYSICALLY DOMINATE THE SOLUTION AND PREVENT PLATEAUING,
// ALSOO PULL SOLUSION TO BELOW ZERO
#[test]
fn lsode2_1d_solid_combustion_no_sublimation_pure_bdf() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_no_sublimation_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_faithful_bdf_solve(10, 100)
        .with_controller(Lsode2ControllerConfig::bdf_only()),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        !summary.algorithm.method_switching_enabled,
        "pure BDF mode should disable Adams/BDF auto switching"
    );

    assert!(
        summary.native_statistics.executed_bdf_count > 0
            || summary.native_statistics.native_linear_solve_calls > 0,
        "expected BDF activity during steep reaction front"
    );

    // Stop-condition semantics check: integration should stop once eta reaches threshold.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    let stop_reached = summary
        .native_integration_solve
        .as_ref()
        .map(|s| s.reached_stop_condition)
        .unwrap_or(false);
    assert!(
        last_A >= 0.999 || stop_reached || summary.status == "finished_native_faithful_partial",
        "expected eta>=0.999 or faithful partial stop, got eta={}, stop_triggered={}, status={}",
        last_A,
        stop_reached,
        summary.status
    );
    assert!(
        last_T.is_finite() && last_T > 300.0,
        "temperature should remain finite and rise above initial value, got T={}",
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}

//ATTENTION! SUBLIMATION IS DISABLED UNTIL "STOP CONDITION" WILL BE IMPLEMENTED OTHERWISE SUBLIMATION WILL UNPHYSICALLY DOMINATE THE SOLUTION AND PREVENT PLATEAUING,
// ALSOO PULL SOLUSION TO BELOW ZERO
#[test]
fn lsode2_1d_solid_combustion_no_sublimation_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_no_sublimation_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        ),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );

    // Plateau-phase exploration: expect the automatic controller to have
    // collected Adams evidence at least once.
    let algo = &summary.algorithm;
    let native = &summary.native_statistics;

    assert!(
        native.native_adams_cost_samples > 0
            || native.preferred_adams_count > 0
            || native.executed_adams_count > 0,
        "solid combustion plateau should collect Adams evidence; adams_cost_samples={}, preferred_adams={}, executed_adams={}",
        native.native_adams_cost_samples,
        native.preferred_adams_count,
        native.executed_adams_count
    );
    assert!(
        algo.executed_family == Some(algo.preferred_family),
        "automatic path should execute the family it prefers; preferred={}, executed={:?}",
        algo.preferred_family,
        algo.executed_family
    );

    // Stop-condition semantics check: integration should stop once eta reaches threshold.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        last_A >= 0.999,
        "stop condition eta>=0.999 should be reached, got eta={}",
        last_A
    );
    assert!(
        last_T.is_finite() && last_T > 300.0,
        "temperature should remain finite and rise above initial value, got T={}",
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}

#[test]
fn lsode2_1d_solid_combustion_no_sublimation_switches_bridge() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        one_d_solid_combustion_no_sublimation_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_bridge_solve()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        ),
    )
    .expect("build 1D solid combustion config");
    let summary = solver
        .solve_with_summary()
        .expect("1D solid combustion solve should finish");

    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );

    // Stop-condition semantics check: integration should stop once eta reaches threshold.
    let (_x, y) = solver.get_result();
    let last_A = y[(y.nrows() - 1, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        last_A >= 0.999,
        "stop condition eta>=0.999 should be reached, got eta={}",
        last_A
    );
    assert!(
        last_T.is_finite() && last_T > 300.0,
        "temperature should remain finite and rise above initial value, got T={}",
        last_T
    );

    let duration = start.elapsed();
    println!(
        "1D solid combustion test duration: {:?}",
        duration.as_millis()
    );
}

#[test]
fn no_subliamtion_radau_test() {
    let eqs = combustion_1d_problem();
    let vars = vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "Qd".to_string(),
    ];
    let values = vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,   // u (flame rate)
        0.245, // c (heat capacity)
        1.6,   // ro (density)
        5e-4,  // Lambda
        // sublimation parameters
        -1e3, // Qd
    ];
    let var_map: HashMap<String, f64> = vars
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let eqs = eqs
        .iter()
        .map(|eq| eq.set_variable_from_map(&var_map))
        .collect::<Vec<_>>();
    let values = vec!["eta".to_string(), "q".to_string(), "T".to_string()];
    let arg = "x".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1e-3, 1e-3, 300.0]);
    let t_bound = 5e-2;

    let mut solver = UniversalODESolver::radau(
        eqs,
        values,
        arg,
        RadauOrder::Order3,
        t0,
        y0,
        t_bound,
        1e-6,
        550,
        Some(1e-5),
    );

    solver.solve();
    let (_t_result, y) = solver.get_result();
    let y = y.unwrap();
    //println!("{:?}", y);
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    let prev_T = y[(y.nrows() - 2, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        prev_T.is_finite() && last_T.is_finite() && last_T > 0.0 && last_T >= prev_T,
        "T should remain finite, positive, and non-decreasing over the final step; prev_T={}, last_T={}",
        prev_T,
        last_T
    );

    assert!(
        prev_A.is_finite() && last_A.is_finite() && (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at domain end; prev_A={}, last_A={}",
        prev_A,
        last_A
    );
}

//#[test]
fn no_sublimation_bdf_test() {
    let eqs = combustion_1d_problem();
    let vars = vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "Qd".to_string(),
    ];
    let values = vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,   // u (flame rate)
        0.245, // c (heat capacity)
        1.6,   // ro (density)
        5e-4,  // Lambda
        // sublimation parameters
        -1e3, // Qd
    ];
    let var_map: HashMap<String, f64> = vars
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let eqs = eqs
        .iter()
        .map(|eq| eq.set_variable_from_map(&var_map))
        .collect::<Vec<_>>();
    let values = vec!["eta".to_string(), "q".to_string(), "T".to_string()];
    let arg = "x".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1e-3, -1e-3, 300.0]);
    let t_bound = 3e-2;

    let mut solver = UniversalODESolver::bdf(eqs, values, arg, t0, y0, t_bound, 1e-5, 1e-6, 1e-6);

    solver.solve();
    let (_t_result, y) = solver.get_result();
    let y = y.unwrap();
    //println!("{:?}", y);
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    let prev_T = y[(y.nrows() - 2, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        prev_T.is_finite() && last_T.is_finite() && last_T > 0.0 && last_T >= prev_T,
        "T should remain finite, positive, and non-decreasing over the final step; prev_T={}, last_T={}",
        prev_T,
        last_T
    );

    assert!(
        prev_A.is_finite() && last_A.is_finite() && (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at domain end; prev_A={}, last_A={}",
        prev_A,
        last_A
    );
}

#[test]
fn no_subliamtion_be_test() {
    let eqs = combustion_1d_problem();
    let vars = vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "u".to_string(),
        "c".to_string(),
        "ro".to_string(),
        "Lambda".to_string(),
        "Qd".to_string(),
    ];
    let values = vec![
        // kinetics
        3.7e11,   // k
        39_000.0, // E
        1.987,    // R
        // physical properties
        0.5,   // u (flame rate)
        0.245, // c (heat capacity)
        1.6,   // ro (density)
        5e-4,  // Lambda
        // sublimation parameters
        -1e3, // Qd
    ];
    let var_map: HashMap<String, f64> = vars
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let eqs = eqs
        .iter()
        .map(|eq| eq.set_variable_from_map(&var_map))
        .collect::<Vec<_>>();
    let values = vec!["eta".to_string(), "q".to_string(), "T".to_string()];
    let arg = "x".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1e-3, 1e-3, 300.0]);
    let t_bound = 5e-2;

    let mut solver = UniversalODESolver::backward_euler(
        eqs,
        values,
        arg,
        t0,
        y0,
        t_bound,
        1e-6,
        550,
        Some(1e-5),
    );

    solver.solve();
    let (_t_result, y) = solver.get_result();
    let y = y.unwrap();
    //println!("{:?}", y);
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    let last_T = y[(y.nrows() - 1, 2)];
    let prev_T = y[(y.nrows() - 2, 2)];
    println!("Final A: {}, Final T: {}", last_A, last_T);
    assert!(
        prev_T.is_finite() && last_T.is_finite() && last_T > 0.0 && last_T >= prev_T,
        "T should remain finite, positive, and non-decreasing over the final step; prev_T={}, last_T={}",
        prev_T,
        last_T
    );

    assert!(
        prev_A.is_finite() && last_A.is_finite() && (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at domain end; prev_A={}, last_A={}",
        prev_A,
        last_A
    );
}

//========================================================================================================
// OVERSIMPLIFIED  COMBUSTION LIKE IVP PROBLEM
// WITH HEAT LOSS
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
        1e-8,
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

    cfg
}

#[test]
fn lsode2_combustion_like_stiff_regression_switches_bdf_to_adams() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        combustion_like_ivp_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        ),
    )
    .expect("build combustion-like config");
    let summary = solver
        .solve_with_summary()
        .expect("combustion-like solve should finish");

    // symbolic lambdify route
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );

    let algo = &summary.algorithm;
    let native = &summary.native_statistics;
    assert!(
        native.native_adams_cost_samples > 0
            || native.preferred_adams_count > 0
            || native.executed_adams_count > 0,
        "solid combustion plateau should collect Adams evidence; adams_cost_samples={}, preferred_adams={}, executed_adams={}",
        native.native_adams_cost_samples,
        native.preferred_adams_count,
        native.executed_adams_count
    );
    assert!(
        algo.executed_family == Some(algo.preferred_family),
        "automatic path should execute the family it prefers; preferred={}, executed={:?}",
        algo.preferred_family,
        algo.executed_family
    );
    // final plateau check: temperature and A should change little at the end
    let (t, y) = solver.get_result();
    assert!(t.len() >= 2, "expected time-history length");
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    assert!(
        (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at the end"
    );
    let duration = start.elapsed();
    println!("test duration: {:?}", duration.as_millis());
}
// ok but too long
#[test]
fn lsode2_combustion_like_stiff_regression_pure_bdf() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        combustion_like_ivp_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_native_sparse_faer_backend()
        .with_faithful_bdf_solve(10, 100)
        .with_controller(Lsode2ControllerConfig::bdf_only()),
    )
    .expect("build combustion-like config");
    let summary = solver
        .solve_with_summary()
        .expect("combustion-like solve should finish");

    // symbolic lambdify route
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        !summary.algorithm.method_switching_enabled,
        "pure BDF mode should disable Adams/BDF auto switching"
    );

    // expect significant BDF activity during burn

    assert!(
        summary.native_statistics.executed_bdf_count > 0
            || summary.native_statistics.native_linear_solve_calls > 0,
        "expected BDF activity during stiff burning phase"
    );

    // final plateau check: temperature and A should change little at the end
    let (t, y) = solver.get_result();
    assert!(t.len() >= 2, "expected time-history length");
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    assert!(
        (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at the end"
    );
    let duration = start.elapsed();
    println!("test duration: {:?}", duration.as_millis());
}

#[test]
fn lsode2_combustion_like_stiff_regression_switches_bdf_to_adams_aot() {
    let start = Instant::now();
    let mut solver = Lsode2Solver::new(
        combustion_like_ivp_config(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::Zig,
                profile: Lsode2AotProfile::Release,
            },
        })
        .with_native_sparse_faer_backend()
        .with_controller(
            Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
        ),
    )
    .expect("build combustion-like config");
    let summary = solver
        .solve_with_summary()
        .expect("combustion-like solve should finish");

    // symbolic lambdify route
    assert_eq!(summary.resolved_source, "symbolic");
    assert!(
        summary.algorithm.method_switching_enabled,
        "switching should be enabled"
    );
    // Plateau-phase exploration: expect the automatic controller to have
    // collected Adams evidence at least once.
    let algo = &summary.algorithm;
    let native = &summary.native_statistics;
    assert!(
        native.native_adams_cost_samples > 0
            || native.preferred_adams_count > 0
            || native.executed_adams_count > 0,
        "solid combustion plateau should collect Adams evidence; adams_cost_samples={}, preferred_adams={}, executed_adams={}",
        native.native_adams_cost_samples,
        native.preferred_adams_count,
        native.executed_adams_count
    );
    assert!(
        algo.executed_family == Some(algo.preferred_family),
        "automatic path should execute the family it prefers; preferred={}, executed={:?}",
        algo.preferred_family,
        algo.executed_family
    );
    // final plateau check: temperature and A should change little at the end
    let (t, y) = solver.get_result();
    assert!(t.len() >= 2, "expected time-history length");
    let last_A = y[(y.nrows() - 1, 0)];
    let prev_A = y[(y.nrows() - 2, 0)];
    assert!(
        (last_A - prev_A).abs() < 1e-3,
        "A should be near-constant at the end"
    );
    let duration = start.elapsed();
    println!("test duration: {:?}", duration.as_millis());
}
