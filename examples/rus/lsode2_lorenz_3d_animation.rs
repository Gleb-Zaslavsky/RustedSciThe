//! LSODE2: Lorenz 3D animation
//!
//! Цель:
//! решить систему Лоренца через LSODE2 и визуализировать полную траекторию
//! с помощью нативного 3D animation helper-а.
//!
//! Этот пример повторяет существующий пример анимации, но использует LSODE2
//! builder API вместо legacy BDF-конструктора.
//!
//! запуск: cargo run --example lsode2_lorenz_3d_animation

use RustedSciThe::numerical::LSODE2::{
    Lsode2BackendConfig, Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::Utils::animation_3d::create_3d_animation;
use nalgebra::{DMatrix, DVector};

fn solve_lorenz(t_bound: f64) -> (DMatrix<f64>, DVector<f64>) {
    let config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("10.0 * (y - x)"),
            Expr::parse_expression("x * (28.0 - z) - y"),
            Expr::parse_expression("x * y - (8.0 / 3.0) * z"),
        ],
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 1.0, 1.0]),
        t_bound,
        0.001,
        1e-8,
        1e-10,
    )
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_backend(Lsode2BackendConfig::native_sparse_faer())
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_faithful_bdf_solve(100_000, 100_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();
    assert_eq!(solver.get_status().as_deref(), Some("finished_native_faithful"));

    let (t, y) = solver.get_result();
    let t = t.expect("LSODE2 should produce a time mesh");
    let y = y.expect("LSODE2 should produce a solution matrix");
    (y.transpose(), t)
}

fn main() {
    println!("Generating Lorenz 3D trajectory with LSODE2...");
    let (positions, times) = solve_lorenz(50.0);
    println!("{} points generated", times.len());
    println!("Use mouse to rotate the view, scroll to zoom");

    create_3d_animation(
        positions,
        times,
        Some(("X".to_string(), "Y".to_string(), "Z".to_string())),
        Some(100.0),
    );
}
