//! Correctness gates for the typed Dense/faer/Banded linear-solve boundary.
//!
//! These tests are separate from the historical `BVP_traits` adapter tests.
//! They cover the three first-class BVP_Damp backends only; sprs and other
//! matrix implementations remain compatibility routes.

use crate::numerical::BVP_Damp::BVP_traits::{BandedMatrixType, BvpLinearSolveError, MatrixType};
use crate::somelinalg::banded::{
    NodeMajorLayout, banded_assembly::BandedAssembly, solver_policy::LinearSolverConfig,
};
use faer::col::Col;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::{DMatrix, DVector};
use std::time::Duration;

#[test]
fn typed_dense_linear_solve_reports_dimension_mismatch() {
    let matrix = DMatrix::<f64>::identity(2, 2);
    let rhs = DVector::from_element(1, 1.0);
    let old = DVector::from_element(2, 0.0);

    let error = matrix
        .try_solve_sys(&rhs, None, 1e-10, 10, (1, 1), &old)
        .expect_err("a short RHS must be rejected before factorization");

    assert_eq!(
        error,
        BvpLinearSolveError::DimensionMismatch {
            matrix_rows: 2,
            matrix_columns: 2,
            rhs_len: 1,
        }
    );
}

#[test]
fn typed_dense_linear_solve_reports_factorization_failure() {
    let matrix = DMatrix::<f64>::zeros(2, 2);
    let rhs = DVector::from_element(2, 1.0);
    let old = DVector::from_element(2, 0.0);

    let error = matrix
        .try_solve_sys(&rhs, None, 1e-10, 10, (1, 1), &old)
        .expect_err("a singular dense matrix must not look like a valid step");

    assert!(matches!(
        error,
        BvpLinearSolveError::FactorizationFailed {
            backend: "dense",
            ..
        }
    ));
}

#[test]
fn typed_dense_linear_solve_preserves_solution_and_timing_shape() {
    let matrix = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 2.0]);
    let rhs = DVector::from_vec(vec![9.0, 8.0]);
    let old = DVector::zeros(2);

    let (solution, timing) = matrix
        .try_solve_sys_with_timing(&rhs, None, 1e-10, 10, (1, 1), &old)
        .expect("well-conditioned dense system should solve");

    assert!((solution.to_DVectorType()[0] - 2.0).abs() < 1e-12);
    assert!((solution.to_DVectorType()[1] - 3.0).abs() < 1e-12);
    assert!(timing.rhs_solve > Duration::ZERO);
}

#[test]
fn typed_faer_linear_solve_reports_direct_lu_timing() {
    let triplets = [
        Triplet::new(0, 0, 3.0),
        Triplet::new(0, 1, 1.0),
        Triplet::new(1, 0, 1.0),
        Triplet::new(1, 1, 2.0),
    ];
    let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(2, 2, &triplets)
        .expect("well-conditioned faer matrix should be constructible");
    let rhs = Col::from_fn(2, |index| if index == 0 { 9.0 } else { 8.0 });
    let old = Col::zeros(2);

    let (solution, timing) = matrix
        .try_solve_sys_with_timing(&rhs, None, 1e-10, 10, (0, 0), &old)
        .expect("faer direct LU should solve");
    let solution = solution.to_DVectorType();

    assert!((solution[0] - 2.0).abs() < 1e-12);
    assert!((solution[1] - 3.0).abs() < 1e-12);
    assert!(timing.factorization > Duration::ZERO);
    assert!(timing.rhs_solve >= Duration::ZERO);
}

#[test]
fn typed_banded_linear_solve_uses_native_result_and_reports_timing() {
    let mut assembly = BandedAssembly::zeros(2, 0, 0).expect("valid banded allocation");
    assembly
        .set(0, 0, 2.0)
        .expect("first diagonal entry is valid");
    assembly
        .set(1, 1, 4.0)
        .expect("second diagonal entry is valid");
    let matrix = BandedMatrixType::new(
        assembly,
        NodeMajorLayout::new(2, 1).expect("valid node-major layout"),
        LinearSolverConfig::faithful_banded(),
    );
    let rhs = DVector::from_vec(vec![2.0, 8.0]);
    let old = DVector::zeros(2);

    let (solution, timing) = matrix
        .try_solve_sys_with_timing(&rhs, None, 1e-10, 10, (0, 0), &old)
        .expect("native banded solve should succeed");

    assert!((solution.to_DVectorType()[0] - 1.0).abs() < 1e-12);
    assert!((solution.to_DVectorType()[1] - 2.0).abs() < 1e-12);
    assert!(matrix.factorization_ready());
    assert!(timing.factorization >= Duration::ZERO);
    assert!(timing.rhs_solve >= Duration::ZERO);
}
