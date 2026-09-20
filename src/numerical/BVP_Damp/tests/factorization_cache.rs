//! Correctness gates for native Jacobian factorization reuse.

//! Architecture lane: backend-independent factorization-cache correctness and
//! telemetry semantics.

use crate::numerical::BVP_Damp::BVP_traits::{BandedMatrixType, MatrixType};
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{DampedSolverOptions, NRBVP, SolverParams};
use crate::somelinalg::banded::NodeMajorLayout;
use crate::somelinalg::banded::banded_assembly::BandedAssembly;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use faer::col::Col;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::Duration;

#[test]
fn dense_timed_linear_solve_reports_typed_factor_and_rhs_stages() {
    let matrix = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0]);
    let rhs = DVector::from_vec(vec![5.0, 7.0, 4.0]);
    let old_vec = DVector::zeros(3);

    let (solution, timing) = matrix.solve_sys_with_timing(&rhs, None, 1e-12, 10, (1, 1), &old_vec);

    let dense_solution = solution
        .as_any()
        .downcast_ref::<DVector<f64>>()
        .expect("Dense timing adapter must preserve DVector output");
    assert!((dense_solution[0] - (5.0 / 6.0)).abs() < 1e-12);
    assert!((dense_solution[1] - (5.0 / 3.0)).abs() < 1e-12);
    assert!((dense_solution[2] - 2.0).abs() < 1e-12);
    assert!(timing.factorization > Duration::ZERO);
    assert!(timing.rhs_solve > Duration::ZERO);
}

#[test]
fn faer_sparse_timed_linear_solve_reports_direct_lu_stages() {
    let triplets = [
        Triplet::new(0, 0, 4.0),
        Triplet::new(1, 1, 3.0),
        Triplet::new(2, 2, 2.0),
    ];
    let matrix: SparseColMat<usize, f64> =
        SparseColMat::try_new_from_triplets(3, 3, &triplets).expect("valid sparse matrix");
    let rhs = Col::from_fn(3, |i| [4.0, 6.0, 4.0][i]);
    let old_vec = Col::zeros(3);

    let (solution, timing) = matrix.solve_sys_with_timing(&rhs, None, 1e-12, 10, (0, 0), &old_vec);

    let sparse_solution = solution
        .as_any()
        .downcast_ref::<Col<f64>>()
        .expect("faer timing adapter must preserve Col output");
    assert!((sparse_solution[0] - 1.0).abs() < 1e-12);
    assert!((sparse_solution[1] - 2.0).abs() < 1e-12);
    assert!((sparse_solution[2] - 2.0).abs() < 1e-12);
    assert!(timing.factorization > Duration::ZERO);
    assert!(timing.rhs_solve > Duration::ZERO);
}

#[test]
fn banded_matrix_reuses_factorization_across_rhs_solves_and_clones() {
    let mut assembly = BandedAssembly::zeros(4, 1, 1).expect("valid banded allocation");
    for i in 0..4 {
        assembly.set(i, i, 4.0).expect("diagonal is in band");
        if i > 0 {
            assembly
                .set(i, i - 1, -1.0)
                .expect("lower diagonal is in band");
        }
        if i + 1 < 4 {
            assembly
                .set(i, i + 1, -1.0)
                .expect("upper diagonal is in band");
        }
    }

    let layout = NodeMajorLayout::new(4, 1).expect("valid node-major layout");
    let matrix = BandedMatrixType::new(
        assembly,
        layout,
        crate::somelinalg::banded::LinearSolverConfig::faithful_banded(),
    );
    assert!(!matrix.factorization_ready());

    let rhs = DVector::from_element(4, 1.0);
    let first = matrix.solve_sys(&rhs, None, 1e-12, 10, (1, 1), &rhs);
    assert!(matrix.factorization_ready());
    assert!(matrix.factorization_duration() >= Duration::ZERO);
    let second = matrix.solve_sys(&rhs, None, 1e-12, 10, (1, 1), &rhs);
    assert!(matrix.factorization_ready());
    assert!(matrix.rhs_solve_duration() >= Duration::ZERO);

    for (left, right) in first.iterate().zip(second.iterate()) {
        assert!((left - right).abs() < 1e-12);
    }

    let cloned: BandedMatrixType = matrix.clone();
    assert!(cloned.factorization_ready());
    let cloned_result = cloned.solve_sys(&rhs, None, 1e-12, 10, (1, 1), &rhs);
    for (expected, actual) in first.iterate().zip(cloned_result.iterate()) {
        assert!((expected - actual).abs() < 1e-12);
    }
}

#[test]
fn banded_matrix_invalidates_factorization_when_assembly_is_replaced() {
    let mut first_assembly = BandedAssembly::zeros(2, 0, 0).expect("valid banded allocation");
    first_assembly
        .set(0, 0, 2.0)
        .expect("first diagonal entry is valid");
    first_assembly
        .set(1, 1, 2.0)
        .expect("second diagonal entry is valid");

    let layout = NodeMajorLayout::new(2, 1).expect("valid node-major layout");
    let mut matrix = BandedMatrixType::new(
        first_assembly,
        layout,
        crate::somelinalg::banded::LinearSolverConfig::faithful_banded(),
    );
    let rhs = DVector::from_element(2, 1.0);
    let first = matrix.solve_sys(&rhs, None, 1e-12, 10, (0, 0), &rhs);
    assert!(matrix.factorization_ready());
    assert!(first.iterate().all(|value| (value - 0.5).abs() < 1e-12));

    let mut second_assembly = BandedAssembly::zeros(2, 0, 0).expect("valid banded allocation");
    second_assembly
        .set(0, 0, 4.0)
        .expect("first diagonal entry is valid");
    second_assembly
        .set(1, 1, 4.0)
        .expect("second diagonal entry is valid");
    matrix.set_assembly(second_assembly);
    assert!(!matrix.factorization_ready());
    assert_eq!(matrix.factorization_duration(), Duration::ZERO);
    assert_eq!(matrix.rhs_solve_duration(), Duration::ZERO);

    let second = matrix.solve_sys(&rhs, None, 1e-12, 10, (0, 0), &rhs);
    assert!(matrix.factorization_ready());
    assert!(second.iterate().all(|value| (value - 0.25).abs() < 1e-12));
}

#[test]
fn banded_matrix_invalidates_factorization_when_solver_config_changes() {
    let mut assembly = BandedAssembly::zeros(2, 0, 0).expect("valid banded allocation");
    assembly
        .set(0, 0, 2.0)
        .expect("first diagonal entry is valid");
    assembly
        .set(1, 1, 4.0)
        .expect("second diagonal entry is valid");

    let layout = NodeMajorLayout::new(2, 1).expect("valid node-major layout");
    let mut matrix = BandedMatrixType::new(
        assembly,
        layout,
        crate::somelinalg::banded::LinearSolverConfig::faithful_banded(),
    );
    let rhs = DVector::from_vec(vec![2.0, 8.0]);
    let first = matrix.solve_sys(&rhs, None, 1e-12, 10, (0, 0), &rhs);
    assert!(matrix.factorization_ready());
    let first = first.to_DVectorType();
    assert!((first[0] - 1.0).abs() < 1e-12);
    assert!((first[1] - 2.0).abs() < 1e-12);

    matrix.set_solver_config(
        crate::somelinalg::banded::LinearSolverConfig::faithful_banded_with_refinement(1),
    );
    assert!(!matrix.factorization_ready());
    assert_eq!(matrix.factorization_duration(), Duration::ZERO);
    assert_eq!(matrix.rhs_solve_duration(), Duration::ZERO);

    let second = matrix.solve_sys(&rhs, None, 1e-12, 10, (0, 0), &rhs);
    assert!(matrix.factorization_ready());
    let second = second.to_DVectorType();
    assert!((second[0] - 1.0).abs() < 1e-12);
    assert!((second[1] - 2.0).abs() < 1e-12);
}

#[test]
fn damped_banded_solver_reports_factorization_reuse_at_solver_level() {
    let n_steps = 12;
    let initial_guess = DMatrix::zeros(2, n_steps);
    let options = DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(3),
            max_damp_iter: Some(5),
            damp_factor: Some(0.5),
            adaptive: None,
        }))
        .with_abs_tolerance(1e-8)
        .with_rel_tolerance(HashMap::from([
            ("y".to_string(), 1e-6),
            ("z".to_string(), 1e-6),
        ]))
        .with_max_iterations(20)
        .with_bounds(HashMap::from([
            ("y".to_string(), (-0.2, 1.2)),
            ("z".to_string(), (-2.0, 2.0)),
        ]))
        .with_banded_generated_backend_defaults();

    let mut solver = NRBVP::new_numeric_fd_with_options(
        initial_guess,
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([("y".to_string(), vec![(0, 0.0), (1, 1.0)])]),
        0.0,
        1.0,
        n_steps,
        options,
        |_x, state: &DVector<f64>, _params| DVector::from_vec(vec![state[1], 0.0]),
    );
    solver.dont_save_log(true);
    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
    solver
        .try_solve()
        .expect("numeric Banded BVP should solve through the public API");

    let stats = solver.get_statistics();
    let counters = &stats.telemetry.counters;
    let timings = &stats.telemetry.timings;
    assert!(counters.linear_solves > 0);
    assert_eq!(counters.rhs_solves, counters.linear_solves);
    assert!(counters.factorizations > 0);
    assert!(counters.factorization_cache_hits > 0);
    assert_eq!(
        counters.factorizations + counters.factorization_cache_hits,
        counters.linear_solves
    );
    assert_eq!(
        stats.counters["number of factorization cache hits"] as u64,
        counters.factorization_cache_hits
    );
    assert!(
        timings.factorization > Duration::ZERO,
        "solver telemetry should record native factorization time"
    );
    assert!(
        timings.rhs_solve > Duration::ZERO,
        "solver telemetry should record native RHS solve time"
    );
    assert!(
        timings.factorization + timings.rhs_solve <= timings.linear_system,
        "factor/RHS timings must remain within the aggregate linear stage"
    );
}
