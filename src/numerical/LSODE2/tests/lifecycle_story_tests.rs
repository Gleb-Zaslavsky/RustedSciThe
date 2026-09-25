//! Lifecycle and caller-owned-buffer correctness stories for LSODE2 Lambdify.

use super::native_jacobian::{NativeJacobianStorage, try_prepare_native_atomview_jacobian_runtime};
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::ivp_telemetry::{IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

macro_rules! println {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

#[test]
fn lsode2_atomview_native_parameter_rebind_parity_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_atomview_native_parameter_rebind_parity_story",
    );

    let equations = vec![
        Expr::parse_expression("a*y1 + y2 - t"),
        Expr::parse_expression("y1 - b*y2"),
    ];
    let variables = vec!["y1".to_string(), "y2".to_string()];
    let parameters = vec!["a".to_string(), "b".to_string()];
    let initial_parameters = DVector::from_vec(vec![2.0, -0.5]);
    let rebound_parameters = DVector::from_vec(vec![3.25, 0.75]);
    let states = [
        (0.25, DVector::from_vec(vec![1.2, -0.7])),
        (1.75, DVector::from_vec(vec![-0.4, 2.1])),
    ];

    let make_options = |backend, telemetry| {
        SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(backend)
            .with_equation_parameters(parameters.clone())
            .with_equation_parameter_values(initial_parameters.clone())
            .with_telemetry(telemetry)
    };
    let expr_telemetry = IvpTelemetry::detailed();
    let atom_telemetry = IvpTelemetry::detailed();
    let expr = prepare_symbolic_ivp_problem(
        equations.clone(),
        variables.clone(),
        "t".to_string(),
        make_options(
            IvpSymbolicAssemblyBackend::ExprLegacy,
            expr_telemetry.clone(),
        ),
    )
    .expect("ExprLegacy parity fixture should prepare");
    let atom = prepare_symbolic_ivp_problem(
        equations,
        variables,
        "t".to_string(),
        make_options(IvpSymbolicAssemblyBackend::AtomView, atom_telemetry.clone()),
    )
    .expect("AtomViewNative parity fixture should prepare");

    let compare = |label: &str,
                   expr: &crate::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem,
                   atom: &crate::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem,
                   states: &[(f64, DVector<f64>)]| {
        let mut max_residual_diff = 0.0_f64;
        let mut max_jacobian_diff = 0.0_f64;
        for (time, state) in states {
            let expr_residual = expr
                .try_evaluate_residual(*time, state)
                .expect("ExprLegacy residual should evaluate");
            let atom_residual = atom
                .try_evaluate_residual(*time, state)
                .expect("AtomViewNative residual should evaluate");
            let mut expr_residual_into = DVector::zeros(expr_residual.len());
            expr.try_evaluate_residual_into(*time, state, &mut expr_residual_into)
                .expect("ExprLegacy residual_into should evaluate");
            let mut atom_residual_into = DVector::zeros(atom_residual.len());
            atom.try_evaluate_residual_into(*time, state, &mut atom_residual_into)
                .expect("AtomViewNative residual_into should evaluate");
            max_residual_diff = max_residual_diff.max(
                expr_residual
                    .iter()
                    .zip(atom_residual.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
            assert!(
                expr_residual
                    .iter()
                    .zip(expr_residual_into.iter())
                    .all(|(left, right)| (left - right).abs() <= 1.0e-12),
                "ExprLegacy residual_into drifted for {label}"
            );
            assert!(
                atom_residual
                    .iter()
                    .zip(atom_residual_into.iter())
                    .all(|(left, right)| (left - right).abs() <= 1.0e-12),
                "AtomViewNative residual_into drifted for {label}"
            );

            let expr_jacobian = expr
                .try_evaluate_jacobian(*time, state)
                .expect("ExprLegacy Jacobian should evaluate");
            let atom_jacobian = atom
                .try_evaluate_jacobian(*time, state)
                .expect("AtomViewNative Jacobian should evaluate");
            max_jacobian_diff = max_jacobian_diff.max(
                expr_jacobian
                    .iter()
                    .zip(atom_jacobian.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
        }
        println!(
            "binding={label} | residual_max_diff={max_residual_diff:.3e} | jacobian_max_diff={max_jacobian_diff:.3e}"
        );
        assert!(
            max_residual_diff <= 1.0e-12,
            "residual parity drift for {label}: {max_residual_diff:e}"
        );
        assert!(
            max_jacobian_diff <= 1.0e-12,
            "Jacobian parity drift for {label}: {max_jacobian_diff:e}"
        );
    };

    compare("initial", &expr, &atom, &states);
    let invalid_parameters = DVector::from_vec(vec![999.0]);
    assert!(matches!(
        expr.set_parameter_values(invalid_parameters.clone()),
        Err(
            crate::symbolic::symbolic_ivp::IvpBackendError::ParameterCountMismatch {
                expected: 2,
                actual: 1
            }
        )
    ));
    assert!(matches!(
        atom.set_parameter_values(invalid_parameters),
        Err(
            crate::symbolic::symbolic_ivp::IvpBackendError::ParameterCountMismatch {
                expected: 2,
                actual: 1
            }
        )
    ));
    compare("after rejected rebind", &expr, &atom, &states);
    let mut wrong_output = DVector::zeros(1);
    assert!(matches!(
        atom.try_evaluate_residual_into(0.25, &states[0].1, &mut wrong_output),
        Err(crate::symbolic::symbolic_ivp::IvpBackendError::InvalidOutputShape {
            stage,
            expected: 2,
            actual: 1
        }) if stage == "residual"
    ));
    expr.set_parameter_values(rebound_parameters.clone())
        .expect("ExprLegacy rebind should succeed");
    atom.set_parameter_values(rebound_parameters)
        .expect("AtomViewNative rebind should succeed");
    compare("rebound", &expr, &atom, &states);

    let expr_snapshot = expr_telemetry.snapshot();
    let atom_snapshot = atom_telemetry.snapshot();
    println!(
        "[LSODE2 AtomViewNative parity] parameters=2; states=2; preparation is reused after numeric rebind"
    );
    println!(
        "route | expr_to_atom_calls | atom_to_expr_calls | residual_calls | jacobian_calls | parameter_binds"
    );
    println!(
        "ExprLegacy | {:>17} | {:>17} | {:>14} | {:>14} | {:>15}",
        expr_snapshot.cold_stage(IvpColdStage::ExprToAtom).calls,
        expr_snapshot.cold_stage(IvpColdStage::AtomToExpr).calls,
        expr_snapshot.residual_evaluations,
        expr_snapshot.jacobian_evaluations,
        expr_snapshot.parameter_binds,
    );
    println!(
        "AtomViewNative | {:>13} | {:>17} | {:>14} | {:>14} | {:>15}",
        atom_snapshot.cold_stage(IvpColdStage::ExprToAtom).calls,
        atom_snapshot.cold_stage(IvpColdStage::AtomToExpr).calls,
        atom_snapshot.residual_evaluations,
        atom_snapshot.jacobian_evaluations,
        atom_snapshot.parameter_binds,
    );
    println!(
        "[LSODE2 AtomViewNative telemetry]\n{}",
        atom_snapshot.pretty_report()
    );

    assert_eq!(expr_snapshot.parameter_binds, 1);
    assert_eq!(atom_snapshot.parameter_binds, 1);
    assert_eq!(atom_snapshot.cold_stage(IvpColdStage::ExprToAtom).calls, 1);
    assert_eq!(atom_snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
}

#[test]
fn lsode2_atomview_native_caller_owned_jacobian_layout_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_atomview_native_caller_owned_jacobian_layout_story",
    );

    let equations = vec![
        Expr::parse_expression("-2*y1 + y2"),
        Expr::parse_expression("3*y1 - 4*y2"),
    ];
    let variables = vec!["y1".to_string(), "y2".to_string()];
    let state = DVector::from_vec(vec![1.0, 2.0]);
    let telemetry = IvpTelemetry::detailed();
    let mut dense = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::Dense,
        telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("native dense runtime should prepare");
    let mut sparse = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::SparseTriplets,
        telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("native sparse runtime should prepare");
    let mut banded = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::Banded { bandwidth: None },
        telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("native banded runtime should prepare");

    let mut dense_out = DMatrix::zeros(2, 2);
    let mut sparse_out = vec![0.0; sparse.sparse_pattern().len()];
    let banded_len = banded
        .banded_layout()
        .expect("banded runtime should expose compact layout")
        .2;
    let mut banded_out = vec![0.0; banded_len];
    let started = Instant::now();
    for _ in 0..3 {
        dense
            .try_evaluate_dense_into(0.0, &state, &mut dense_out)
            .expect("dense caller-owned evaluation should succeed");
    }
    let dense_elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let started = Instant::now();
    for _ in 0..3 {
        sparse
            .try_evaluate_sparse_values_into(0.0, &state, &mut sparse_out)
            .expect("sparse caller-owned evaluation should succeed");
    }
    let sparse_elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let started = Instant::now();
    for _ in 0..3 {
        banded
            .try_evaluate_banded_values_into(0.0, &state, &mut banded_out)
            .expect("banded caller-owned evaluation should succeed");
    }
    let banded_elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let snapshot = telemetry.snapshot();

    assert_eq!(
        dense_out,
        DMatrix::from_row_slice(2, 2, &[-2.0, 1.0, 3.0, -4.0])
    );
    assert_eq!(sparse_out, vec![-2.0, 1.0, 3.0, -4.0]);
    let banded_matrix = Banded::from_vec(2, 1, 1, banded_out)
        .expect("caller-owned compact banded output should remain valid");
    assert_eq!(banded_matrix[(0, 0)], -2.0);
    assert_eq!(banded_matrix[(0, 1)], 1.0);
    assert_eq!(banded_matrix[(1, 0)], 3.0);
    assert_eq!(banded_matrix[(1, 1)], -4.0);

    println!(
        "[LSODE2 AtomViewNative caller-owned Jacobian] repeats=3; measured work excludes report file I/O"
    );
    println!("route | rows | cols | sparse_nnz | band_kl | band_ku | band_data | elapsed_ms");
    println!(
        "Dense | {} | {} | - | - | - | - | {:.3}",
        dense.rows(),
        dense.cols(),
        dense_elapsed_ms
    );
    println!(
        "Sparse | {} | {} | {} | - | - | - | {:.3}",
        sparse.rows(),
        sparse.cols(),
        sparse.sparse_pattern().len(),
        sparse_elapsed_ms
    );
    let (band_kl, band_ku, band_data) = banded.banded_layout().unwrap_or((0, 0, 0));
    println!(
        "Banded | {} | {} | - | {} | {} | {} | {:.3}",
        banded.rows(),
        banded.cols(),
        band_kl,
        band_ku,
        band_data,
        banded_elapsed_ms
    );
    println!(
        "telemetry | jacobian_evaluations={} | copies={} bytes | errors={} | expr_to_atom={} | atom_to_expr={}",
        snapshot.jacobian_evaluations,
        snapshot.copied_bytes,
        snapshot.errors,
        snapshot.cold_stage(IvpColdStage::ExprToAtom).calls,
        snapshot.cold_stage(IvpColdStage::AtomToExpr).calls,
    );
    println!("{}", snapshot.pretty_report());
}
