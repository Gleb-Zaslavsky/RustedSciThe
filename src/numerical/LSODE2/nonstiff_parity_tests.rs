use super::{
    Lsode2ControllerConfig, Lsode2NativeExecutionConfig, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2Solver,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn nonstiff_scalar_decay_exact_sparse_config() -> Lsode2ProblemConfig {
    // Exact solution: y(t) = exp(-t), y(0)=1
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        5.0,
        1.0,
        1.0e-8,
        1.0e-10,
    )
    .with_first_step(Some(0.2))
    .with_native_sparse_faer_backend()
    .with_controller(Lsode2ControllerConfig::adams_only())
    .with_native_execution(Lsode2NativeExecutionConfig::native_solve(200_000, 200_000))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
    .with_analytical_callbacks(
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
    )
}

fn nonstiff_two_scale_exact_banded_config() -> Lsode2ProblemConfig {
    // y2' = -2*y2, y2(0)=1 => y2 = exp(-2t)
    // y1' = -y1 + y2, y1(0)=0 => y1 = exp(-t) - exp(-2t)
    Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-y1 + y2"),
            Expr::parse_expression("-2.0*y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![0.0, 1.0]),
        5.0,
        1.0,
        1.0e-8,
        1.0e-10,
    )
    .with_first_step(Some(0.2))
    .with_native_banded_faithful_backend()
    .with_controller(Lsode2ControllerConfig::adams_only())
    .with_native_execution(Lsode2NativeExecutionConfig::native_solve(200_000, 200_000))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
    .with_analytical_callbacks(
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0] + y[1], -2.0 * y[1]]),
        |_t, _y: &DVector<f64>| DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 0.0, -2.0]),
    )
}

#[test]
fn lsode2_adams_nonstiff_scalar_exact_sparse_matches_closed_form() {
    let mut solver = Lsode2Solver::new(nonstiff_scalar_decay_exact_sparse_config())
        .expect("nonstiff scalar sparse Adams config should build");
    let summary = solver
        .solve_with_summary()
        .expect("nonstiff scalar sparse Adams solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected status for nonstiff scalar sparse Adams run: {}",
        summary.status
    );
    assert_eq!(summary.algorithm.controller_mode, "adams_only");
    assert_eq!(summary.algorithm.preferred_family, "adams");
    assert_eq!(
        summary.algorithm.executed_family,
        Some(summary.algorithm.preferred_family)
    );

    let final_t = summary.final_t.expect("final t should be present");
    assert!(
        final_t >= 4.999,
        "nonstiff scalar sparse Adams run should reach near t_bound, got t={final_t:e}"
    );
    let y_final = summary.final_y.expect("final y should be present")[0];
    let exact = (-final_t).exp();
    assert!(
        (y_final - exact).abs() < 5.0e-7,
        "nonstiff scalar sparse Adams mismatch: got={y_final:e}, exact={exact:e}"
    );

    assert!(
        summary.native_statistics.executed_adams_count > 0,
        "expected native Adams execution evidence in adams_only mode"
    );
}

#[test]
fn lsode2_adams_nonstiff_two_scale_exact_banded_matches_closed_form() {
    let mut solver = Lsode2Solver::new(nonstiff_two_scale_exact_banded_config())
        .expect("nonstiff two-scale banded Adams config should build");
    let summary = solver
        .solve_with_summary()
        .expect("nonstiff two-scale banded Adams solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected status for nonstiff two-scale banded Adams run: {}",
        summary.status
    );
    assert_eq!(summary.algorithm.controller_mode, "adams_only");
    assert_eq!(summary.algorithm.preferred_family, "adams");
    assert_eq!(
        summary.algorithm.executed_family,
        Some(summary.algorithm.preferred_family)
    );

    let final_t = summary.final_t.expect("final t should be present");
    assert!(
        final_t >= 4.999,
        "nonstiff two-scale banded Adams run should reach near t_bound, got t={final_t:e}"
    );
    let y = summary.final_y.expect("final y should be present");
    let y1_exact = (-final_t).exp() - (-2.0 * final_t).exp();
    let y2_exact = (-2.0 * final_t).exp();
    assert!(
        (y[0] - y1_exact).abs() < 2.0e-6,
        "nonstiff two-scale banded y1 mismatch: got={:e}, exact={:e}",
        y[0],
        y1_exact
    );
    assert!(
        (y[1] - y2_exact).abs() < 2.0e-6,
        "nonstiff two-scale banded y2 mismatch: got={:e}, exact={:e}",
        y[1],
        y2_exact
    );

    assert!(
        summary.native_statistics.executed_adams_count > 0,
        "expected native Adams execution evidence in adams_only mode"
    );
}
