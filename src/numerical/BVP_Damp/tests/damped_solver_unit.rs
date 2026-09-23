//! Unit tests extracted from the production module.
//!
//! Keeping these tests in a separate file keeps solver implementation and
//! test-only story machinery independently navigable.

use super::*;
use crate::numerical::BVP_Damp::BVP_traits::MatrixType;
use crate::numerical::BVP_Damp::BVP_traits::{convert_to_fun, convert_to_jac};
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy, GeneratedBackendConfig,
    SparseGeneratedBackendMode,
};
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedSparseAotBackend, register_linked_sparse_backend, unregister_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_orchestrator::{
    ParallelExecutorConfig, ParallelFallbackPolicy,
};
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
use faer::Col;
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

fn sparse_surface_test_solver() -> NRBVP {
    NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
        DMatrix::from_element(2, 4, 0.1),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        4,
        DampedSolverOptions::sparse_damped(),
    )
}

fn sparse_surface_test_solver_with_tolerances() -> NRBVP {
    let bounds = HashMap::from([
        ("z".to_string(), (-10.0, 10.0)),
        ("y".to_string(), (-10.0, 10.0)),
    ]);
    let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
    let options = DampedSolverOptions {
        strategy_params: Some(SolverParams::default()),
        abs_tolerance: 1e-6,
        rel_tolerance: Some(rel_tolerance),
        max_iterations: 10,
        bounds: Some(bounds),
        ..DampedSolverOptions::sparse_damped()
    };
    NRBVP::new_with_options(
        vec![Expr::parse_expression("y-z"), Expr::parse_expression("-z")],
        DMatrix::from_element(2, 5, 0.25),
        vec!["z".to_string(), "y".to_string()],
        "x".to_string(),
        HashMap::from([
            ("z".to_string(), vec![(0usize, 1.0f64)]),
            ("y".to_string(), vec![(1usize, 1.0f64)]),
        ]),
        0.0,
        1.0,
        5,
        options,
    )
}

#[test]
fn try_step_with_linear_telemetry_surfaces_missing_cached_jacobian_as_typed_error() {
    let solver = sparse_surface_test_solver();

    let error = solver
        .try_step_with_linear_telemetry(solver.p, &*solver.y)
        .expect_err("a Newton step without a cached Jacobian must be fallible");

    assert!(matches!(
        error,
        BvpBackendIntegrationError::PipelinePanicked(message)
            if message.contains("cached Jacobian")
    ));
}

#[test]
fn try_step_preserves_partial_telemetry_on_linear_shape_failure() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    solver.fun = Box::new(FunEnum::Dense(Box::new(|_, y| y.clone())));
    solver.y = Box::new(DVector::from_element(size, 1.0));
    solver.factor_owner.old_jac = Some(Box::new(DMatrix::<f64>::identity(1, 1)));

    let error = solver
        .try_step_with_linear_telemetry(solver.p, &*solver.y)
        .expect_err("a residual/Jacobian shape mismatch must be typed");

    assert!(matches!(
        error,
        BvpBackendIntegrationError::LinearSolveFailed {
            backend,
            matrix_rows: 1,
            matrix_columns: 1,
            rhs_len,
            message,
        } if backend == "legacy-matrix"
            && rhs_len == size
            && message.contains("shape mismatch")
    ));
    let snapshot = solver.get_statistics().telemetry;
    assert_eq!(snapshot.counters.residual_calls, 1);
    assert_eq!(snapshot.counters.linear_solves, 0);
    assert_eq!(snapshot.counters.factorizations, 0);
}

#[test]
fn try_step_preserves_partial_telemetry_on_dense_factorization_failure() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    solver.fun = Box::new(FunEnum::Dense(Box::new(|_, y| y.clone())));
    solver.y = Box::new(DVector::from_element(size, 1.0));
    solver.factor_owner.old_jac = Some(Box::new(DMatrix::<f64>::zeros(size, size)));

    let error = solver
        .try_step_with_linear_telemetry(solver.p, &*solver.y)
        .expect_err("a singular Dense Jacobian must be typed");

    assert!(matches!(
        error,
        BvpBackendIntegrationError::LinearSolveFailed {
            backend,
            matrix_rows,
            matrix_columns,
            rhs_len,
            message,
        } if backend == "matrix-try-api"
            && matrix_rows == size
            && matrix_columns == size
            && rhs_len == size
            && message.contains("factorization failed")
    ));
    let snapshot = solver.get_statistics().telemetry;
    assert_eq!(snapshot.counters.residual_calls, 1);
    assert_eq!(snapshot.counters.linear_solves, 0);
    assert_eq!(snapshot.counters.factorizations, 0);
}

#[test]
fn try_calc_residual_surfaces_callback_shape_mismatch() {
    let mut solver = sparse_surface_test_solver();
    let expected_len = solver.y.len();
    solver.fun = convert_to_fun(Box::new(|_, _| {
        Box::new(DVector::from_element(1, 0.0)) as Box<dyn VectorType>
    }));

    let error = solver
        .try_calc_residual(solver.y.clone_box())
        .expect_err("a malformed residual callback must be typed");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::CallbackShapeMismatch {
            stage,
            expected_rows,
            expected_columns: 1,
            actual_rows: 1,
            actual_columns: 1,
        } if stage == "residual" && expected_rows == expected_len
    ));
}

#[test]
fn try_calc_residual_surfaces_callback_panic_as_typed_error() {
    let mut solver = sparse_surface_test_solver();
    solver.fun = convert_to_fun(Box::new(|_, _| {
        std::panic::panic_any("residual callback failed deliberately")
    }));

    let error = solver
        .try_calc_residual(solver.y.clone_box())
        .expect_err("a callback panic must not escape the typed residual path");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::CallbackExecutionFailed { stage, message }
            if stage == "residual" && message.contains("residual callback failed deliberately")
    ));
}

#[test]
fn grid_refinement_surfaces_sci_residual_callback_failure_as_typed_error() {
    let mut solver = sparse_surface_test_solver();
    solver.result = Some(DVector::from_element(
        solver.values.len() * (solver.n_steps + 1),
        1.0,
    ));
    solver.strategy_params = Some(SolverParams {
        adaptive: Some(AdaptiveGridConfig {
            version: 1,
            max_refinements: 1,
            grid_method: GridRefinementMethod::Sci(),
        }),
        ..SolverParams::default()
    });
    solver.fun = convert_to_fun(Box::new(|_, _| {
        std::panic::panic_any("grid residual failed deliberately")
    }));

    let error = solver
        .create_new_grid()
        .expect_err("Sci grid refinement must expose residual callback errors");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::CallbackExecutionFailed { stage, message }
            if stage == "grid refinement residual"
                && message.contains("grid residual failed deliberately")
    ));
}

#[test]
fn grid_refinement_rejects_missing_method_as_typed_configuration_error() {
    let mut solver = sparse_surface_test_solver();
    solver.result = Some(DVector::from_element(
        solver.values.len() * (solver.n_steps + 1),
        1.0,
    ));
    solver.strategy_params = Some(SolverParams::default());

    let error = solver
        .create_new_grid()
        .expect_err("adaptive grid refinement without a method must be typed");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::InvalidSolverConfiguration {
            field,
            value,
            message,
        } if field == "strategy_params.adaptive.grid_method"
            && value == "missing"
            && message.contains("grid method must be specified")
    ));
}

#[test]
fn try_recalc_jacobian_surfaces_callback_shape_mismatch() {
    let mut solver = sparse_surface_test_solver();
    let expected_len = solver.y.len();
    solver.jac = Some(convert_to_jac(Box::new(|_, _| {
        Box::new(DMatrix::from_element(1, 1, 1.0)) as Box<dyn MatrixType>
    })));
    solver.jac_recalc = true;

    let error = solver
        .try_recalc_jacobian()
        .expect_err("a malformed Jacobian callback must be typed");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::CallbackShapeMismatch {
            stage,
            expected_rows,
            expected_columns,
            actual_rows: 1,
            actual_columns: 1,
        } if stage == "Jacobian"
            && expected_rows == expected_len
            && expected_columns == expected_len
    ));
}

#[test]
fn try_recalc_jacobian_surfaces_callback_panic_as_typed_error() {
    let mut solver = sparse_surface_test_solver();
    solver.jac = Some(convert_to_jac(Box::new(|_, _| {
        std::panic::panic_any("Jacobian callback failed deliberately")
    })));
    solver.jac_recalc = true;

    let error = solver
        .try_recalc_jacobian()
        .expect_err("a callback panic must not escape the typed Jacobian path");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::CallbackExecutionFailed { stage, message }
            if stage == "Jacobian" && message.contains("Jacobian callback failed deliberately")
    ));
}

#[test]
fn damped_try_step_reuses_owned_dense_factor_for_repeated_rhs() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    let matrix = DMatrix::<f64>::identity(size, size);
    let owner = prepare_factor_owner_runtime(&matrix, (0, 0), None)
        .expect("dense matrix should produce an owned factor runtime");
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    *solver.factor_owner.borrow_mut() = Some(owner);
    solver.fun = Box::new(FunEnum::Dense(Box::new(|_, y| y.clone())));
    solver.y = Box::new(DVector::from_element(size, 1.0));

    let first = solver
        .try_step_with_linear_telemetry(solver.p, &*solver.y)
        .expect("first owned-factor RHS solve should succeed");
    let second = solver
        .try_step_with_linear_telemetry(solver.p, &*solver.y)
        .expect("repeated owned-factor RHS solve should succeed");

    assert!(first.2 > std::time::Duration::ZERO);
    assert_eq!(second.2, std::time::Duration::ZERO);
    assert_eq!(first.0.to_DVectorType(), second.0.to_DVectorType());
    assert!(
        solver
            .factor_owner
            .borrow()
            .as_ref()
            .expect("owned factor should remain installed")
            .has_solved_rhs()
    );
}

#[test]
fn damped_continuation_change_invalidates_owned_factor() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    let matrix = DMatrix::<f64>::identity(size, size);
    let owner = prepare_factor_owner_runtime(&matrix, (0, 0), None)
        .expect("dense matrix should produce an owned factor runtime");
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    *solver.factor_owner.borrow_mut() = Some(owner);
    solver.set_p(1.0);

    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.jac_recalc);
}

#[test]
fn damped_mesh_change_invalidates_owned_factor() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    let matrix = DMatrix::<f64>::identity(size, size);
    let owner = prepare_factor_owner_runtime(&matrix, (0, 0), None)
        .expect("dense matrix should produce an owned factor runtime");
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    *solver.factor_owner.borrow_mut() = Some(owner);

    solver.set_mesh(0.0, 2.0, 6);

    assert_eq!(solver.x_mesh.len(), 7);
    assert_eq!(solver.n_steps, 6);
    assert_eq!(solver.initial_guess.shape(), (solver.values.len(), 6));
    assert_eq!(solver.y.len(), solver.values.len() * solver.n_steps);
    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.jac_recalc);
    assert!(!solver.prepared_runtime_revision.is_current());
    assert_eq!(
        solver
            .telemetry_counters
            .snapshot()
            .factorization_invalidations,
        1
    );
}

#[test]
fn damped_try_set_mesh_rejects_invalid_interval_and_step_count() {
    let mut solver = sparse_surface_test_solver();

    let interval_error = solver
        .try_set_mesh(1.0, 1.0, 4)
        .expect_err("zero-length Damped interval must be rejected");
    assert!(matches!(
        interval_error,
        BvpBackendIntegrationError::InvalidProblem { field, .. } if field == "interval"
    ));

    let step_error = solver
        .try_set_mesh(0.0, 1.0, 1)
        .expect_err("one-interval Damped mesh must be rejected");
    assert!(matches!(
        step_error,
        BvpBackendIntegrationError::InvalidProblem { field, .. } if field == "n_steps"
    ));
}

#[test]
fn damped_identical_mesh_preserves_owned_factor() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    let matrix = DMatrix::<f64>::identity(size, size);
    let owner = prepare_factor_owner_runtime(&matrix, (0, 0), None)
        .expect("dense matrix should produce an owned factor runtime");
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    *solver.factor_owner.borrow_mut() = Some(owner);
    solver.jac_recalc = false;

    solver.set_mesh(0.0, 1.0, 4);

    assert!(solver.factor_owner.old_jac.is_some());
    assert!(solver.factor_owner.borrow().is_some());
    assert!(!solver.jac_recalc);
    assert_eq!(
        solver
            .telemetry_counters
            .snapshot()
            .factorization_invalidations,
        0
    );
}

#[test]
fn damped_numeric_callbacks_invalidate_owned_factor() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    let matrix = DMatrix::<f64>::identity(size, size);
    let owner = prepare_factor_owner_runtime(&matrix, (0, 0), None)
        .expect("dense matrix should produce an owned factor runtime");
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    *solver.factor_owner.borrow_mut() = Some(owner);

    solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| y.clone())));

    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.jac_recalc);
    assert_eq!(
        solver
            .telemetry_counters
            .snapshot()
            .factorization_invalidations,
        1
    );

    let matrix = DMatrix::<f64>::identity(size, size);
    *solver.factor_owner.borrow_mut() = Some(
        prepare_factor_owner_runtime(&matrix, (0, 0), None)
            .expect("dense matrix should produce an owned factor runtime"),
    );
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    solver.set_numeric_jacobian(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
        DMatrix::identity(y.len(), y.len())
    })));

    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.jac_recalc);
    assert_eq!(
        solver
            .telemetry_counters
            .snapshot()
            .factorization_invalidations,
        2
    );
}

#[test]
fn damped_parameter_rebind_invalidates_owned_factor() {
    let mut solver = sparse_surface_test_solver();
    solver.set_params(Some(&["alpha"]));
    solver.set_param_values(Some(vec![1.0]));

    let size = solver.values.len() * solver.n_steps;
    let matrix = DMatrix::<f64>::identity(size, size);
    let owner = prepare_factor_owner_runtime(&matrix, (0, 0), None)
        .expect("dense matrix should produce an owned factor runtime");
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    *solver.factor_owner.borrow_mut() = Some(owner);
    solver.set_param_values(Some(vec![2.0]));

    assert!(solver.factor_owner.old_jac.is_none());
    assert!(solver.factor_owner.borrow().is_none());
    assert!(solver.jac_recalc);
}

#[test]
fn damped_statistics_keep_legacy_projection_and_typed_snapshot_in_sync() {
    let solver = NRBVP::default();
    let stats = solver.get_statistics();

    assert_eq!(
        stats.counters["number of iterations"],
        stats.telemetry.counters.iterations as usize
    );
    assert_eq!(
        stats.counters["number of factorizations"],
        stats.telemetry.counters.factorizations as usize
    );
    assert_eq!(
        stats.counters["number of RHS solves"],
        stats.telemetry.counters.rhs_solves as usize
    );
    assert!(stats.telemetry.timings.total >= stats.telemetry.timings.jacobian);
}

#[test]
fn damped_statistics_count_residual_requests_from_shared_self_boundary() {
    let solver = NRBVP::default();
    let _ = solver.calc_residual(solver.y.clone_box());
    let stats = solver.get_statistics();

    assert_eq!(stats.telemetry.counters.residual_calls, 1);
    assert_eq!(stats.counters["number of residual calls"], 1);
}

#[test]
fn prepared_solver_rejects_unprepared_runtime_with_typed_error() {
    let mut solver = NRBVP::default();

    let error = solver
        .try_solver_prepared()
        .expect_err("prepared solve must reject a runtime that was never generated");

    assert!(matches!(
        error,
        BvpBackendIntegrationError::PreparedRuntimeInvalidated { reason }
            if reason.contains("try_eq_generate")
    ));
}

#[test]
fn generated_backend_policy_change_invalidates_prepared_runtime() {
    let mut solver = sparse_surface_test_solver();
    let size = solver.values.len() * solver.n_steps;
    let matrix = DMatrix::<f64>::identity(size, size);
    solver.factor_owner.old_jac = Some(Box::new(matrix));
    solver.prepared_runtime_revision.mark_prepared();

    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

    assert!(!solver.prepared_runtime_revision.is_current());
    assert!(solver.factor_owner.old_jac.is_none());
    let error = solver
        .try_solver_prepared()
        .expect_err("a backend policy change must invalidate prepared callbacks");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::PreparedRuntimeInvalidated { .. }
    ));
}

#[test]
fn damped_statistics_expose_generated_runtime_diagnostics() {
    let mut solver = NRBVP::default();
    let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
        Box::new(|_x, y: &DVector<f64>| y.clone());
    let mut runtime_diagnostics = HashMap::new();
    runtime_diagnostics.insert(
        "aot.runtime.execution_policy".to_string(),
        "Parallel".to_string(),
    );
    runtime_diagnostics.insert(
        "aot.runtime.sparse_jacobian.actual_jobs".to_string(),
        "4".to_string(),
    );

    solver.apply_generated_solver_state(DampedGeneratedSolverState {
        fun: Box::new(FunEnum::Dense(fun0)),
        jac: None,
        bounds_vec: Vec::new(),
        rel_tolerance_vec: Vec::new(),
        variable_string: Vec::new(),
        bandwidth: (0, 0),
        bc_position_and_value: Vec::new(),
        updated_resolver: None,
        selected_backend:
            crate::symbolic::codegen::codegen_backend_selection::SelectedBackendKind::AotCompiled,
        runtime_diagnostics,
        generation_telemetry: None,
        atom_discretization_telemetry: Some(
            crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot {
                total: std::time::Duration::from_millis(3),
                ..Default::default()
            },
        ),
        legacy_lambdify_telemetry: None,
        atom_lambdify_telemetry: None,
        direct_banded_jacobian_telemetry: None,
        aot_telemetry: None,
        parameter_binding: None,
    });

    assert!(solver.prepared_runtime_revision.is_current());

    let stats = solver.get_statistics();
    assert_eq!(
        stats
            .diagnostics
            .get("aot.runtime.execution_policy")
            .map(String::as_str),
        Some("Parallel")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.runtime.sparse_jacobian.actual_jobs")
            .map(String::as_str),
        Some("4")
    );
    assert_eq!(
        stats
            .telemetry
            .atom_discretization
            .expect("Atom preparation telemetry should cross solver handoff")
            .total,
        std::time::Duration::from_millis(3)
    );
    assert_eq!(
        stats
            .diagnostics
            .get("generated.selected_backend")
            .map(String::as_str),
        Some("AotCompiled")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.build_policy")
            .map(String::as_str),
        Some("UseIfAvailable")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.resolver.present")
            .map(String::as_str),
        Some("false")
    );
}

#[test]
fn damped_statistics_include_generated_lifecycle_configuration() {
    let solver = sparse_surface_test_solver()
        .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
        .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
        .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
        ))
        .with_aot_resolver(Some(AotResolver::new(AotRegistry::new())));

    let stats = solver.get_statistics();
    assert_eq!(
        stats
            .diagnostics
            .get("generated.backend_policy")
            .map(String::as_str),
        Some("PreferAotThenLambdify")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("generated.selected_backend")
            .map(String::as_str),
        Some("not_generated")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.build_policy")
            .map(String::as_str),
        Some("RequirePrebuilt")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.execution_policy")
            .map(String::as_str),
        Some("SequentialOnly")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.chunking.residual")
            .map(String::as_str),
        Some("ByTargetChunkCount { target_chunks: 2 }")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.chunking.sparse_jacobian")
            .map(String::as_str),
        Some("ByTargetChunkCount { target_chunks: 3 }")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.resolver.present")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        stats
            .diagnostics
            .get("aot.resolver.entries")
            .map(String::as_str),
        Some("0")
    );
}

#[test]
fn generated_backend_surface_builder_methods_update_solver_config() {
    let solver = sparse_surface_test_solver()
        .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
        .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
        .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
        ))
        .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        solver.aot_execution_policy(),
        &AotExecutionPolicy::SequentialOnly
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
    assert_eq!(
        solver.aot_chunking_policy(),
        AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
        )
    );
    assert_eq!(
        solver.atom_optimization_profile(),
        AtomOptimizationProfile::NoCse
    );
}

#[test]
fn symbolic_assembly_backend_is_exposed_on_solver_surface() {
    let mut solver = sparse_surface_test_solver()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);

    assert_eq!(
        solver.symbolic_assembly_backend(),
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        solver.generated_backend_config().symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );

    solver.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    assert_eq!(
        solver.symbolic_assembly_backend(),
        BvpSymbolicAssemblyBackend::ExprLegacy
    );
    assert_eq!(
        solver.generated_backend_config().symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::ExprLegacy
    );
}

#[test]
fn sparse_generated_backend_presets_are_exposed_on_solver_surface() {
    let solver = sparse_surface_test_solver().with_sparse_aot_build_if_missing_release();

    assert_eq!(
        solver.symbolic_assembly_backend(),
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        solver.aot_build_policy(),
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
}

#[test]
fn sparse_generated_backend_mode_is_exposed_on_solver_surface() {
    let mut solver = sparse_surface_test_solver()
        .with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt);

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);

    solver.set_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease);
    assert_eq!(
        solver.aot_build_policy(),
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
}

#[test]
fn sparse_damped_options_preset_sets_production_defaults() {
    let options = DampedSolverOptions::sparse_damped();

    assert_eq!(options.scheme, "forward");
    assert_eq!(options.strategy, "Damped");
    assert_eq!(options.method, "Sparse");
    assert_eq!(
        options.generated_backend_config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        options.generated_backend_config.backend_policy_override,
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        options.generated_backend_config.aot_build_policy,
        AotBuildPolicy::UseIfAvailable
    );
    assert_eq!(
        options.generated_backend_config.aot_execution_policy,
        AotExecutionPolicy::Auto
    );
    assert_eq!(
        options.generated_backend_config.aot_chunking_policy,
        AotChunkingPolicy::default()
    );
}

#[test]
fn banded_damped_options_preset_uses_auto_aot_chunking_defaults() {
    let options = DampedSolverOptions::banded_damped();

    assert_eq!(
        options.generated_backend_config.matrix_backend_override,
        Some(MatrixBackend::Banded)
    );
    assert_eq!(
        options.generated_backend_config.aot_execution_policy,
        AotExecutionPolicy::Auto
    );
    assert_eq!(
        options.generated_backend_config.aot_chunking_policy,
        AotChunkingPolicy::default()
    );
}

#[test]
fn dense_damped_options_preset_sets_dense_defaults() {
    let options = DampedSolverOptions::dense_damped();

    assert_eq!(options.scheme, "forward");
    assert_eq!(options.strategy, "Damped");
    assert_eq!(options.method, "Dense");
}

#[test]
fn damped_options_scheme_builder_methods_set_legacy_scheme_flag() {
    let options = DampedSolverOptions::sparse_damped().trapezoid_derivative();
    assert_eq!(options.scheme, "trapezoid");

    let options = options.forward_derivative();
    assert_eq!(options.scheme, "forward");

    let options = options.with_scheme(BvpDerivativeScheme::Trapezoid);
    assert_eq!(options.scheme, "trapezoid");

    let options = options.with_scheme_name("custom-experimental");
    assert_eq!(options.scheme, "custom-experimental");
}

#[test]
fn damped_solver_scheme_builder_methods_set_solver_scheme_flag() {
    let solver = sparse_surface_test_solver().trapezoid_derivative();
    assert_eq!(solver.scheme, "trapezoid");

    let solver = solver.forward_derivative();
    assert_eq!(solver.scheme, "forward");

    let solver = solver.with_scheme(BvpDerivativeScheme::Trapezoid);
    assert_eq!(solver.scheme, "trapezoid");

    let solver = solver.with_scheme_name("custom-experimental");
    assert_eq!(solver.scheme, "custom-experimental");
}

#[test]
fn constructor_style_sparse_generated_backend_mode_sets_solver_config() {
    let solver = NRBVP::new_with_sparse_generated_backend_mode(
        vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
        DMatrix::from_element(2, 4, 0.1),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        4,
        "forward".to_string(),
        "Damped".to_string(),
        None,
        None,
        "Sparse".to_string(),
        1e-6,
        None,
        10,
        None,
        None,
        SparseGeneratedBackendMode::RequirePrebuilt,
    );

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
}

#[test]
fn options_style_solver_setup_sets_sparse_generated_backend_mode() {
    let options = DampedSolverOptions::sparse_damped().with_sparse_aot_require_prebuilt();

    let solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
        DMatrix::from_element(2, 4, 0.1),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        4,
        options,
    );

    assert_eq!(
        solver.backend_policy_override(),
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
}

#[test]
fn sparse_eq_generate_uses_bundle_handoff_without_breaking_metadata() {
    let values = vec!["z".to_string(), "y".to_string()];
    let n_steps = 5;
    let mut solver = sparse_surface_test_solver_with_tolerances();

    solver
        .try_eq_generate(None, None)
        .expect("sparse handoff should generate through the fallible API");

    assert!(solver.jac.is_some());
    assert!(!solver.variable_string.is_empty());
    assert!(!solver.BC_position_and_value.is_empty());
    assert_eq!(solver.bounds_vec.len(), values.len() * n_steps);
    assert_eq!(solver.rel_tolerance_vec.len(), values.len() * n_steps);
    let _ = &solver.fun;
}

#[test]
fn prepared_solver_detects_direct_compatibility_field_mutation() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver
        .try_eq_generate(None, None)
        .expect("baseline preparation should succeed");

    // This intentionally bypasses the setter. The fingerprint is checked
    // only at the prepared-solve boundary and must reject stale callbacks.
    solver.abs_tolerance *= 10.0;
    let error = solver
        .try_solver_prepared()
        .expect_err("direct public-field mutation must invalidate the plan");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::PreparedRuntimeInvalidated { reason }
            if reason.contains("public compatibility input")
    ));
}

#[test]
fn prepared_solver_detects_direct_residual_callback_replacement() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver
        .try_eq_generate(None, None)
        .expect("baseline preparation should succeed");

    // Keep the old allocation alive while installing the replacement so
    // the compatibility fingerprint observes two distinct callback
    // identities even when the allocator would otherwise reuse memory.
    let replacement = convert_to_fun(Box::new(|_, y: &dyn VectorType| y.clone_box()));
    let _old_callback = std::mem::replace(&mut solver.fun, replacement);
    let error = solver
        .try_solver_prepared()
        .expect_err("direct callback replacement must invalidate the plan");

    assert!(matches!(
        error,
        BvpBackendIntegrationError::PreparedRuntimeInvalidated { reason }
            if reason.contains("public compatibility input")
    ));
}

#[test]
fn prepared_solver_detects_direct_jacobian_callback_replacement() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver
        .try_eq_generate(None, None)
        .expect("baseline preparation should succeed");

    let replacement = convert_to_jac(Box::new(|_, _| {
        Box::new(DMatrix::<f64>::identity(1, 1)) as Box<dyn MatrixType>
    }));
    let old_callback = solver.jac.take();
    solver.jac = Some(replacement);
    let _old_callback = old_callback;
    let error = solver
        .try_solver_prepared()
        .expect_err("direct Jacobian callback replacement must invalidate the plan");

    assert!(matches!(
        error,
        BvpBackendIntegrationError::PreparedRuntimeInvalidated { reason }
            if reason.contains("public compatibility input")
    ));
}

#[test]
fn typed_parameter_binding_rejects_wrong_arity_without_panic() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver
        .try_set_params(Some(&["alpha", "beta"]))
        .expect("parameter names should be accepted");
    let error = solver
        .try_set_param_values(Some(vec![1.0]))
        .expect_err("wrong parameter arity must be a typed error");
    assert!(matches!(
        error,
        BvpBackendIntegrationError::InvalidSolverConfiguration { field, .. }
            if field == "param_values"
    ));
}

#[test]
fn numeric_only_without_numeric_rhs_is_rejected_instead_of_lambdify_fallback() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

    let err = solver
        .try_eq_generate(None, None)
        .expect_err("NumericOnly without numeric_rhs must not silently fall back to lambdify");
    assert!(matches!(
        err,
        BvpBackendIntegrationError::PipelinePanicked(message)
            if message.contains("requires a numeric_rhs closure")
    ));
}

#[test]
fn sparse_numeric_only_with_pure_numeric_rhs_uses_fd_jacobian_runtime_path() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
    solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
        // y' = z, z' = -y (linear oscillator-like test fixture)
        DVector::from_vec(vec![y[1], -y[0]])
    })));

    solver
        .try_eq_generate(None, None)
        .expect("pure numeric generation should succeed");

    // In pure numeric mode we intentionally rely on FD Jacobian in the solver.
    assert!(solver.jac.is_none());
    assert!(!solver.variable_string.is_empty());
    assert!(!solver.BC_position_and_value.is_empty());

    let runtime_y = Col::from_fn(solver.values.len() * solver.n_steps, |index| {
        0.2 + index as f64 * 0.01
    });
    let residual = solver.fun.call(solver.p, &runtime_y);
    assert_eq!(residual.len(), runtime_y.nrows());

    solver.y = Box::new(runtime_y);
    solver.jac_recalc = true;
    solver.recalc_jacobian();
    assert!(solver.factor_owner.old_jac.is_some());
}

#[test]
fn recalc_jacobian_can_fallback_to_fd_when_symbolic_jacobian_is_missing() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver
        .try_eq_generate(None, None)
        .expect("baseline sparse generation should succeed");

    solver.y = Box::new(Col::from_fn(
        solver.values.len() * solver.n_steps,
        |index| 0.2 + index as f64 * 0.01,
    ));
    solver.jac = None;
    solver.jac_recalc = true;
    solver.recalc_jacobian();

    assert!(solver.factor_owner.old_jac.is_some());
}

#[test]
fn banded_default_atomview_lambdify_generation_produces_runtime_callbacks() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.set_generated_backend_config(GeneratedBackendConfig::from_banded_mode(
        BandedGeneratedBackendMode::Lambdify,
    ));

    assert_eq!(
        solver.symbolic_assembly_backend(),
        BvpSymbolicAssemblyBackend::AtomView
    );
    solver
        .try_eq_generate(None, None)
        .expect("default banded AtomView lambdify generation should succeed");

    assert!(solver.jac.is_some());
    let runtime_y = Col::from_fn(solver.values.len() * solver.n_steps, |index| {
        0.2 + index as f64 * 0.01
    });
    let residual = solver.fun.call(solver.p, &runtime_y);
    assert_eq!(residual.len(), runtime_y.nrows());
}

#[test]
fn build_solver_request_carries_optional_aot_resolver() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));

    let request = solver.build_solver_request(None, None);
    assert!(request.resolver.is_some());
}

#[test]
fn build_solver_request_carries_parameter_names_and_values() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.set_params(Some(&["alpha", "beta"]));
    solver.set_param_values(Some(vec![1.5, -0.25]));

    let request = solver.build_solver_request(None, None);
    assert_eq!(
        request.param_names,
        Some(vec!["alpha".to_string(), "beta".to_string()])
    );
    assert_eq!(request.param_values, Some(vec![1.5, -0.25]));
}

#[test]
#[should_panic(expected = "expected exactly 2 values for declared symbolic parameters")]
fn solver_surface_rejects_parameter_value_length_mismatch() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.set_params(Some(&["alpha", "beta"]));
    solver.set_param_values(Some(vec![1.5]));
}

#[test]
fn build_solver_request_uses_backend_policy_override_when_present() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

    let request = solver.build_solver_request(None, None);
    assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
}

#[test]
fn generated_backend_config_is_exposed_as_user_facing_solver_setting() {
    let mut solver = sparse_surface_test_solver_with_tolerances();

    let config = GeneratedBackendConfig::with_parts(
        Some(BackendSelectionPolicy::NumericOnly),
        Some(AotResolver::new(AotRegistry::new())),
    );
    solver.set_generated_backend_config(config);

    let request = solver.build_solver_request(None, None);
    assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
    assert!(request.resolver.is_some());
    assert_eq!(
        solver.generated_backend_config().backend_policy_override,
        Some(BackendSelectionPolicy::NumericOnly)
    );
}

#[test]
fn generated_backend_config_can_be_applied_during_solver_construction() {
    let solver = sparse_surface_test_solver_with_tolerances().with_generated_backend_config(
        GeneratedBackendConfig::with_parts(
            Some(BackendSelectionPolicy::NumericOnly),
            Some(AotResolver::new(AotRegistry::new())),
        ),
    );

    assert_eq!(
        solver.generated_backend_config().backend_policy_override,
        Some(BackendSelectionPolicy::NumericOnly)
    );
    assert!(solver.generated_backend_config().resolver.is_some());
}

#[test]
fn cleanup_registered_aot_artifacts_is_safe_without_registered_artifacts() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);

    solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));
    assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);
}

#[test]
fn build_solver_request_carries_surface_aot_policies() {
    let config = GeneratedBackendConfig::new()
        .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
        .with_resolver(Some(AotResolver::new(AotRegistry::new())))
        .with_aot_execution_policy(AotExecutionPolicy::Parallel(ParallelExecutorConfig {
            jobs_per_worker: 2,
            max_residual_jobs: Some(4),
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        }))
        .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
        .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 8,
            }),
            Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 4 }),
        ))
        .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

    let mut solver =
        sparse_surface_test_solver_with_tolerances().with_generated_backend_config(config);

    let request = solver.build_solver_request(None, None);
    assert_eq!(
        request.aot_build_policy,
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
    assert_eq!(
        request.aot_chunking_policy.residual,
        Some(ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 8
        })
    );
    assert_eq!(
        request.aot_chunking_policy.sparse_jacobian,
        Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 4 })
    );
    assert_eq!(
        request.atom_optimization_profile,
        AtomOptimizationProfile::NoCse
    );
    match request.aot_execution_policy {
        AotExecutionPolicy::Parallel(inner) => {
            assert_eq!(inner.jobs_per_worker, 2);
            assert_eq!(inner.max_residual_jobs, Some(4));
            assert_eq!(inner.max_sparse_jobs, Some(2));
        }
        other => std::panic!("expected parallel execution policy, got {other:?}"),
    }
}

#[test]
fn eq_generate_build_if_missing_saves_compiled_resolver_for_next_request() {
    let mut solver = sparse_surface_test_solver_with_tolerances().with_generated_backend_config(
        GeneratedBackendConfig::new()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            }),
    );

    solver
        .try_eq_generate(None, None)
        .expect("build-if-missing path should generate through the fallible API");

    let saved_resolver = solver
        .generated_backend_config()
        .resolver
        .as_ref()
        .expect("first build-if-missing run should save updated resolver");
    assert!(
        !saved_resolver.registry().is_empty(),
        "first build-if-missing run should register at least one compiled artifact"
    );

    let updated_config = solver
        .generated_backend_config()
        .clone()
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    solver.set_generated_backend_config(updated_config);

    let next_request = solver.build_solver_request(None, None);
    assert!(next_request.resolver.is_some());

    let next_state = next_request
        .generate()
        .expect("next request should reuse the saved compiled resolver and generate successfully");
    assert!(
        next_state.jac.is_some(),
        "successful generation through the saved resolver should still provide a Jacobian callback"
    );
}

#[test]
fn second_eq_generate_reuses_saved_resolver_and_runs_linked_compiled_backend() {
    let values = vec!["z".to_string(), "y".to_string()];
    let n_steps = 5;
    let mut solver = sparse_surface_test_solver_with_tolerances()
        .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
        .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        });

    solver
        .try_eq_generate(None, None)
        .expect("first sparse generation should succeed through the fallible API");
    let saved_resolver = solver
        .generated_backend_config()
        .resolver
        .clone()
        .expect("first build-if-missing run should save updated resolver");

    let y = Col::from_fn(values.len() * n_steps, |index| 0.2 + index as f64 * 0.01);
    let baseline = solver.fun.call(0.0, &y).to_DVectorType();

    let problem_keys = saved_resolver.registry().problem_keys();
    assert_eq!(
        problem_keys.len(),
        1,
        "build-if-missing should register exactly one artifact for this isolated test"
    );
    let problem_key = problem_keys[0].clone();
    let resolved = saved_resolver.resolve_by_problem_key(&problem_key);
    assert!(
        resolved.is_compiled(),
        "saved resolver should see compiled artifact"
    );

    let baseline_values: Vec<f64> = baseline.iter().copied().collect();
    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        problem_key.clone(),
        resolved.registered.manifest.io.residual_len,
        (
            resolved.registered.manifest.io.jacobian_rows,
            resolved.registered.manifest.io.jacobian_cols,
        ),
        resolved.registered.manifest.io.jacobian_nnz.unwrap_or(0),
        Arc::new(move |_args, out| {
            for (dst, src) in out.iter_mut().zip(baseline_values.iter()) {
                *dst = *src + 123.0;
            }
        }),
        Arc::new(move |_args, out| {
            for (index, value) in out.iter_mut().enumerate() {
                *value = 700.0 + index as f64;
            }
        }),
    ));

    solver.set_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    solver
        .try_eq_generate(None, None)
        .expect("second sparse generation should reuse resolver through the fallible API");
    let residual = solver.fun.call(0.0, &y).to_DVectorType();

    for (actual, expected) in residual.iter().zip(baseline.iter()) {
        assert!((actual - (expected + 123.0)).abs() < 1e-10);
    }

    unregister_linked_sparse_backend(&problem_key);
}

#[test]
fn try_eq_generate_surfaces_missing_prebuilt_aot_as_typed_error() {
    let mut solver =
        sparse_surface_test_solver_with_tolerances().with_sparse_aot_require_prebuilt();

    let err = solver
        .try_eq_generate(None, None)
        .expect_err("try_eq_generate should return a typed AOT availability error");

    assert!(matches!(
        err,
        BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
    ));
}

#[test]
fn try_solve_surfaces_missing_prebuilt_aot_as_typed_error() {
    let mut solver =
        sparse_surface_test_solver_with_tolerances().with_sparse_aot_require_prebuilt();
    solver.dont_save_log(true);

    let err = solver
        .try_solve()
        .expect_err("try_solve should return a typed AOT availability error");

    assert!(matches!(
        err,
        BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
    ));
}

#[test]
fn try_solve_surfaces_invalid_loglevel_as_typed_error() {
    let mut solver = sparse_surface_test_solver_with_tolerances();
    solver.loglevel = Some("trace".to_string());

    let err = solver
        .try_solve()
        .expect_err("invalid loglevel should be returned as a typed error");

    assert!(matches!(
        err,
        BvpBackendIntegrationError::InvalidLogLevel { ref level } if level == "trace"
    ));
}
