//! LSODE2 Three-Body Problem
//!
//! Goal:
//! solve a classical three-body gravitational system with LSODE2 using
//! strict tolerances and visualize the trajectory of the first body in 2D.
//!
//! This example keeps the original physics from `three_body_problem.rs` but
//! switches to the modern LSODE2 builder-style API.
//!
//! Run:
//! cargo run --example lsode2_three_body_problem --release

use RustedSciThe::numerical::LSODE2::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2BackendConfig, Lsode2LinearSolverPolicy,
    Lsode2LinearSystemStructure, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::Utils::animation_2d::create_2d_animation;
use nalgebra::{DMatrix, DVector, RowDVector};
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendFlavor {
    Lambdify,
    AotTcc,
}


fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn body_row(solution: &DMatrix<f64>, index: usize) -> RowDVector<f64> {
    solution.row(index).into_owned()
}

fn two_row_projection(solution: &DMatrix<f64>, x_index: usize, y_index: usize) -> DMatrix<f64> {
    let x = body_row(solution, x_index);
    let y = body_row(solution, y_index);
    DMatrix::from_rows(&[x, y])
}

fn energy_and_center_of_mass_checks(solution: &DMatrix<f64>, times: &DVector<f64>) {
    assert_eq!(solution.nrows(), 12, "three-body state should have 12 rows");
    assert_eq!(solution.ncols(), times.len(), "solution columns must match time samples");

    let k = 39.47841760435743;
    let m0 = 1.0;
    let m1 = 0.5;
    let m2 = 0.75;
    let m_sum = m0 + m1 + m2;

    let x0 = solution.row(0);
    let vx0 = solution.row(1);
    let y0 = solution.row(2);
    let vy0 = solution.row(3);
    let x1 = solution.row(4);
    let vx1 = solution.row(5);
    let y1 = solution.row(6);
    let vy1 = solution.row(7);
    let x2 = solution.row(8);
    let vx2 = solution.row(9);
    let y2 = solution.row(10);
    let vy2 = solution.row(11);

    let mut initial_energy = 0.0_f64;
    let mut initial_cm_x = 0.0_f64;
    let mut initial_cm_y = 0.0_f64;
    let mut initial_cm_vx = 0.0_f64;
    let mut initial_cm_vy = 0.0_f64;
    let mut max_energy_drift = 0.0_f64;
    let mut max_cm_velocity_drift = 0.0_f64;
    let mut max_cm_position_drift = 0.0_f64;

    for i in 0..solution.ncols() {
        let r01 = ((x0[i] - x1[i]).powi(2) + (y0[i] - y1[i]).powi(2)).sqrt();
        let r02 = ((x0[i] - x2[i]).powi(2) + (y0[i] - y2[i]).powi(2)).sqrt();
        let r12 = ((x1[i] - x2[i]).powi(2) + (y1[i] - y2[i]).powi(2)).sqrt();

        let kinetic = 0.5 * m0 * (vx0[i].powi(2) + vy0[i].powi(2))
            + 0.5 * m1 * (vx1[i].powi(2) + vy1[i].powi(2))
            + 0.5 * m2 * (vx2[i].powi(2) + vy2[i].powi(2));
        let potential = -k * (m0 * m1 / r01 + m0 * m2 / r02 + m1 * m2 / r12);
        let energy = kinetic + potential;

        let cm_x = (m0 * x0[i] + m1 * x1[i] + m2 * x2[i]) / m_sum;
        let cm_y = (m0 * y0[i] + m1 * y1[i] + m2 * y2[i]) / m_sum;
        let cm_vx = (m0 * vx0[i] + m1 * vx1[i] + m2 * vx2[i]) / m_sum;
        let cm_vy = (m0 * vy0[i] + m1 * vy1[i] + m2 * vy2[i]) / m_sum;

        if i == 0 {
            initial_energy = energy;
            initial_cm_x = cm_x;
            initial_cm_y = cm_y;
            initial_cm_vx = cm_vx;
            initial_cm_vy = cm_vy;
        }

        max_energy_drift = max_energy_drift.max((energy - initial_energy).abs());
        max_cm_velocity_drift = max_cm_velocity_drift.max(
            ((cm_vx - initial_cm_vx).powi(2) + (cm_vy - initial_cm_vy).powi(2)).sqrt(),
        );

        let time = times[i];
        let expected_cm_x = initial_cm_x + initial_cm_vx * time;
        let expected_cm_y = initial_cm_y + initial_cm_vy * time;
        max_cm_position_drift = max_cm_position_drift.max(
            ((cm_x - expected_cm_x).powi(2) + (cm_y - expected_cm_y).powi(2)).sqrt(),
        );
    }

    println!("Energy / COM checks:");
    println!("  max |E-E0|              = {max_energy_drift:.3e}");
    println!("  max |Vcm-Vcm0|          = {max_cm_velocity_drift:.3e}");
    println!("  max |Rcm-Rcm0-Vcm0*t|    = {max_cm_position_drift:.3e}");

    assert!(max_energy_drift.is_finite());
    assert!(max_cm_velocity_drift.is_finite());
    assert!(max_cm_position_drift.is_finite());
}

fn three_body_solution(t_bound: f64, backend: BackendFlavor) -> (DMatrix<f64>, DVector<f64>) {
    let k = 39.47841760435743;
    let m0 = 1.0;
    let m1 = 0.5;
    let m2 = 0.75;

    let params: HashMap<String, f64> = HashMap::from([
        ("k".to_string(), k),
        ("m0".to_string(), m0),
        ("m1".to_string(), m1),
        ("m2".to_string(), m2),
    ]);

    let r01 = Expr::parse_expression("((x0 - x1)^2 + (y0 - y1)^2)^0.5");
    let r02 = Expr::parse_expression("((x0 - x2)^2 + (y0 - y2)^2)^0.5");
    let r12 = Expr::parse_expression("((x1 - x2)^2 + (y1 - y2)^2)^0.5");

    let eq_vx0 = Expr::parse_expression("-k * (m1*(x0 - x1)/R01^3 + m2*(x0 - x2)/R02^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R02", &r02)
        .set_variable_from_map(&params);
    let eq_vy0 = Expr::parse_expression("-k * (m1*(y0 - y1)/R01^3 + m2*(y0 - y2)/R02^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R02", &r02)
        .set_variable_from_map(&params);

    let eq_vx1 = Expr::parse_expression("-k * (m0*(x1 - x0)/R01^3 + m2*(x1 - x2)/R12^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);
    let eq_vy1 = Expr::parse_expression("-k * (m0*(y1 - y0)/R01^3 + m2*(y1 - y2)/R12^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);

    let eq_vx2 = Expr::parse_expression("-k * (m0*(x2 - x0)/R02^3 + m1*(x2 - x1)/R12^3)")
        .substitute_variable("R02", &r02)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);
    let eq_vy2 = Expr::parse_expression("-k * (m0*(y2 - y0)/R02^3 + m1*(y2 - y1)/R12^3)")
        .substitute_variable("R02", &r02)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);

    let eq_sys = vec![
        Expr::parse_expression("vx0"),
        eq_vx0,
        Expr::parse_expression("vy0"),
        eq_vy0,
        Expr::parse_expression("vx1"),
        eq_vx1,
        Expr::parse_expression("vy1"),
        eq_vy1,
        Expr::parse_expression("vx2"),
        eq_vx2,
        Expr::parse_expression("vy2"),
        eq_vy2,
    ];

    let unknowns = vec![
        "x0".to_string(),
        "vx0".to_string(),
        "y0".to_string(),
        "vy0".to_string(),
        "x1".to_string(),
        "vx1".to_string(),
        "y1".to_string(),
        "vy1".to_string(),
        "x2".to_string(),
        "vx2".to_string(),
        "y2".to_string(),
        "vy2".to_string(),
    ];

    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![
        0.0, 0.0, // x0, vx0
        0.0, 0.0, // y0, vy0
        1.0, 0.0, // x1, vx1
        0.0, 5.0, // y1, vy1
        0.0, -0.4, // x2, vx2
        33.0, 0.0, // y2, vy2
    ]);

    let mut config = Lsode2ProblemConfig::new(
        eq_sys,
        unknowns,
        "t".to_string(),
        t0,
        y0,
        t_bound,
        0.001,
        1e-10,
        1e-12,
    );

    config = match backend {
        BackendFlavor::Lambdify => config
            .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::AtomView,
                execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
            })
            .with_backend(Lsode2BackendConfig::native_sparse_faer()),
        BackendFlavor::AotTcc => config
            .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::AtomView,
                execution: Lsode2SymbolicExecutionMode::Aot {
                    toolchain: Lsode2AotToolchain::CTcc,
                    profile: Lsode2AotProfile::Release,
                },
            })
            .with_backend(Lsode2BackendConfig::native_sparse_faer_aot_c_tcc(
                "target/lsode2-three-body-aot",
            )),
    };

    config = config
        .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
        .with_faithful_bdf_solve(250_000, 250_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();
    assert!(
        matches!(
            solver.get_status().as_deref(),
            Some("finished_native_faithful") | Some("finished_native_faithful_partial")
        ),
        "LSODE2 should finish natively, got {:?}",
        solver.get_status()
    );

    let (t, y) = solver.get_result();
    let t = t.expect("LSODE2 should produce a time mesh");
    let y = y.expect("LSODE2 should produce a solution matrix");
    (y.transpose(), t)
}

fn main() {
    println!("Generating three-body trajectories with LSODE2...");
    let now = Instant::now();

    const ACTIVE_BACKEND: BackendFlavor = BackendFlavor::AotTcc;

    println!("Active backend flavor: {:?}", ACTIVE_BACKEND);
    if matches!(ACTIVE_BACKEND, BackendFlavor::AotTcc) && !command_available("tcc") {
        panic!("AOT backend selected for the three-body example, but tcc was not found on PATH");
    }

    let (solution, times) = three_body_solution(500.0, ACTIVE_BACKEND);
    energy_and_center_of_mass_checks(&solution, &times);

    let positions = two_row_projection(&solution, 0, 2);
    println!("{} points generated", times.len());
    println!("{:?} solution shape", positions.shape());
    println!("Use mouse to pan the view, scroll to zoom");
    println!("Solver time: {:.2?}", now.elapsed().as_millis() as f64);

    create_2d_animation(
        positions,
        times,
        Some(("x0".to_string(), "y0".to_string())),
        Some(500.0),
    );
}
