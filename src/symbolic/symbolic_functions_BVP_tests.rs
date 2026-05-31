#![cfg(test)]

use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
use crate::numerical::Examples_and_utils::NonlinEquation;
use crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    register_linked_sparse_backend, unregister_linked_sparse_backend, LinkedSparseAotBackend,
};
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_orchestrator::AutoExecutionMode;
use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend, PreparedProblem};
use crate::symbolic::codegen::codegen_runtime_api::{
    recommended_residual_chunking_for_parallelism, recommended_row_chunking_for_parallelism,
    ResidualChunkingStrategy,
};
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{AotBuildProfile, AotBuildRequest};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{
    BvpBackendConfig, BvpBackendKind, BvpMatrixBackend, BvpSparseExecutionPlan,
    BvpSymbolicAssemblyBackend, Jacobian,
};
use crate::symbolic::symbolic_functions_BVP2::{BandedJacobianChunking, BandedLambdifyConfig};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;

fn build_small_symbolic_case() -> Jacobian {
    let mut jacobian = Jacobian::new();
    let y = Expr::Var("y".to_string());
    let z = Expr::Var("z".to_string());
    let eq1 = z.clone() * Expr::Const(10.0) + Expr::exp(y.clone());
    let eq2 = Expr::ln(z) + y;
    jacobian.from_vectors(vec![eq1, eq2], vec!["y".to_string(), "z".to_string()]);
    jacobian.calc_jacobian_parallel();
    jacobian
}

fn build_parameterized_symbolic_case() -> Jacobian {
    let mut jacobian = Jacobian::new();
    let a = Expr::Var("a".to_string());
    let y = Expr::Var("y".to_string());
    let z = Expr::Var("z".to_string());
    let eq1 = a.clone() * z.clone() + y.clone();
    let eq2 = z + a * y;
    jacobian.from_vectors(vec![eq1, eq2], vec!["y".to_string(), "z".to_string()]);
    jacobian.set_params(Some(&["a"]));
    jacobian.set_param_values(Some(vec![2.0]));
    jacobian.variables_for_all_disrete = vec![
        vec!["a".to_string(), "y".to_string(), "z".to_string()],
        vec!["a".to_string(), "y".to_string(), "z".to_string()],
    ];
    jacobian.calc_jacobian_parallel();
    jacobian
}

fn build_bandwidth_symbolic_case() -> Jacobian {
    let mut jacobian = Jacobian::new();
    let a = Expr::Var("a".to_string());
    let b = Expr::Var("b".to_string());
    let c = Expr::Var("c".to_string());
    let eq0 = a.clone() + b.clone();
    let eq1 = a + b.clone() + c.clone();
    let eq2 = b + c;
    jacobian.from_vectors(
        vec![eq0, eq1, eq2],
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    );
    jacobian.variables_for_all_disrete = vec![
        vec!["a".to_string(), "b".to_string()],
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec!["b".to_string(), "c".to_string()],
    ];
    jacobian.bandwidth = Some((1, 1));
    jacobian
}

fn build_node_major_banded_symbolic_case() -> Jacobian {
    let mut jacobian = Jacobian::new();
    jacobian.vector_of_functions = vec![
        Expr::Var("y_0".to_string()) + Expr::Var("z_0".to_string()),
        Expr::Const(2.0) * Expr::Var("z_0".to_string()) - Expr::Var("y_0".to_string()),
        Expr::Var("y_1".to_string()) + Expr::Const(3.0) * Expr::Var("z_1".to_string()),
        Expr::Const(4.0) * Expr::Var("z_1".to_string()) - Expr::Var("y_1".to_string()),
    ];
    jacobian.vector_of_variables = vec![
        Expr::Var("y_0".to_string()),
        Expr::Var("z_0".to_string()),
        Expr::Var("y_1".to_string()),
        Expr::Var("z_1".to_string()),
    ];
    jacobian.variable_string = vec![
        "y_0".to_string(),
        "z_0".to_string(),
        "y_1".to_string(),
        "z_1".to_string(),
    ];
    jacobian.bandwidth = Some((3, 3));
    jacobian.calc_jacobian_parallel();
    jacobian
}

fn real_bvp_inputs() -> (
    Vec<Expr>,
    Vec<String>,
    String,
    HashMap<String, Vec<(usize, f64)>>,
) {
    let eq1 = Expr::parse_expression("y-z");
    let eq2 = Expr::parse_expression("-z^3");
    let eq_system = vec![eq1, eq2];
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();

    let mut border_conditions = HashMap::new();
    border_conditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
    border_conditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

    (eq_system, values, arg, border_conditions)
}

#[test]
fn bvp_matrix_backend_roundtrip_preserves_legacy_sparse_mapping() {
    assert_eq!(
        BvpMatrixBackend::from_legacy_method("Sparse"),
        Some(BvpMatrixBackend::FaerSparseCol)
    );
    assert_eq!(BvpMatrixBackend::FaerSparseCol.legacy_method(), "Sparse");
    assert_eq!(
        BvpBackendConfig::default(),
        BvpBackendConfig::lambdify(BvpMatrixBackend::FaerSparseCol)
    );
    assert_eq!(
        BvpMatrixBackend::from_legacy_method("Banded"),
        Some(BvpMatrixBackend::Banded)
    );
    assert_eq!(BvpMatrixBackend::Banded.legacy_method(), "Banded");
}

#[test]
fn set_backend_config_updates_legacy_method_string() {
    let mut jacobian = Jacobian::new();
    let config = BvpBackendConfig::new(BvpBackendKind::Lambdify, BvpMatrixBackend::Dense);

    jacobian.set_backend_config(config);

    assert_eq!(jacobian.method, "Dense");
    assert_eq!(jacobian.backend_config(), config);
}

#[test]
fn symbolic_assembly_backend_can_switch_to_atom_view() {
    let mut jacobian = Jacobian::new();
    jacobian.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);

    assert_eq!(
        jacobian.symbolic_assembly_backend(),
        BvpSymbolicAssemblyBackend::AtomView
    );
}

#[test]
fn compile_lambdified_problem_with_faer_backend_matches_expected_values() {
    let mut jacobian = build_small_symbolic_case();
    let expected_jacobian = DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 1.0, 1.0]);
    let expected_residual = DVector::from_vec(vec![11.0, 0.0]);
    let dense_variables = DVector::from_vec(vec![0.0, 1.0]);
    let variables = &*Vectors_type_casting(&dense_variables, "Sparse".to_string());

    jacobian.compile_lambdified_problem_with_config(
        "x",
        vec!["y", "z"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::FaerSparseCol),
    );

    let residual = jacobian.residiual_function.call(1.0, variables);
    let jac = jacobian.jac_function.as_mut().unwrap();
    let jacobian_value = jac.call(1.0, variables);

    assert_eq!(
        jacobian.backend_config().matrix_backend,
        BvpMatrixBackend::FaerSparseCol
    );
    assert_eq!(residual.to_DVectorType(), expected_residual);
    assert_eq!(jacobian_value.to_DMatrixType(), expected_jacobian);
}

#[test]
fn compile_lambdified_problem_with_faer_backend_and_params_matches_expected_values() {
    let mut jacobian = build_parameterized_symbolic_case();
    let expected_jacobian = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    let expected_residual = DVector::from_vec(vec![11.0, 10.0]);
    let dense_variables = DVector::from_vec(vec![3.0, 4.0]);
    let variables = &*Vectors_type_casting(&dense_variables, "Sparse".to_string());

    jacobian.compile_lambdified_problem_with_config(
        "x",
        vec!["y", "z"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::FaerSparseCol),
    );

    let residual = jacobian.residiual_function.call(1.0, variables);
    let jac = jacobian.jac_function.as_mut().unwrap();
    let jacobian_value = jac.call(1.0, variables);

    assert_eq!(residual.to_DVectorType(), expected_residual);
    assert_eq!(jacobian_value.to_DMatrixType(), expected_jacobian);
}

#[test]
fn compile_lambdified_problem_with_dense_backend_and_params_matches_expected_values() {
    let mut jacobian = build_parameterized_symbolic_case();
    let expected_jacobian = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    let expected_residual = DVector::from_vec(vec![11.0, 10.0]);
    let variables = DVector::from_vec(vec![3.0, 4.0]);

    jacobian.compile_lambdified_problem_with_config(
        "x",
        vec!["y", "z"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::Dense),
    );

    let residual = jacobian.residiual_function.call(1.0, &variables);
    let jac = jacobian.jac_function.as_mut().unwrap();
    let jacobian_value = jac.call(1.0, &variables);

    assert_eq!(residual.to_DVectorType(), expected_residual);
    assert_eq!(jacobian_value.to_DMatrixType(), expected_jacobian);
}

#[test]
fn compile_lambdified_problem_with_banded_backend_matches_expected_values() {
    let mut jacobian = build_node_major_banded_symbolic_case();
    let variables = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
    let expected_residual = DVector::from_vec(vec![5.0, 4.0, 19.0, 16.0]);
    let expected_jacobian = DMatrix::from_row_slice(
        4,
        4,
        &[
            1.0, 1.0, 0.0, 0.0, //
            -1.0, 2.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 3.0, //
            0.0, 0.0, -1.0, 4.0,
        ],
    );

    jacobian.compile_lambdified_problem_with_config(
        "x",
        vec!["y_0", "z_0", "y_1", "z_1"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::Banded),
    );

    let residual = jacobian.residiual_function.call(1.0, &variables);
    let jac = jacobian.jac_function.as_mut().unwrap();
    let jacobian_value = jac.call(1.0, &variables);

    assert_eq!(
        jacobian.backend_config().matrix_backend,
        BvpMatrixBackend::Banded
    );
    assert_eq!(residual.to_DVectorType(), expected_residual);
    assert_eq!(jacobian_value.to_DMatrixType(), expected_jacobian);
    let binding = jacobian
        .last_lambdify_binding_timer_snapshot
        .as_ref()
        .expect("lambdify callback compilation should expose binding-stage timings");
    for stage in [
        "lambdify jacobian callback compile time",
        "lambdify residual callback compile time",
    ] {
        assert!(
            binding.get(stage).is_some_and(|value| *value >= 0.0),
            "missing non-negative callback compilation stage {stage}"
        );
    }
}

#[test]
fn compile_lambdified_problem_with_banded_backend_entry_chunking_matches_diagonal() {
    let expected_residual = DVector::from_vec(vec![5.0, 4.0, 19.0, 16.0]);
    let expected_jacobian = DMatrix::from_row_slice(
        4,
        4,
        &[
            1.0, 1.0, 0.0, 0.0, //
            -1.0, 2.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 3.0, //
            0.0, 0.0, -1.0, 4.0,
        ],
    );
    let variables = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

    let mut diagonal = build_node_major_banded_symbolic_case();
    diagonal.set_banded_lambdify_config(BandedLambdifyConfig {
        jacobian_chunking: BandedJacobianChunking::Diagonal,
        ..BandedLambdifyConfig::default()
    });
    diagonal.compile_lambdified_problem_with_config(
        "x",
        vec!["y_0", "z_0", "y_1", "z_1"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::Banded),
    );
    let diag_residual = diagonal.residiual_function.call(1.0, &variables);
    let diag_jac = diagonal
        .jac_function
        .as_mut()
        .expect("diagonal banded jacobian callback should exist")
        .call(1.0, &variables);

    let mut entry = build_node_major_banded_symbolic_case();
    entry.set_banded_lambdify_config(BandedLambdifyConfig {
        jacobian_chunking: BandedJacobianChunking::EntryChunks,
        ..BandedLambdifyConfig::default()
    });
    entry.compile_lambdified_problem_with_config(
        "x",
        vec!["y_0", "z_0", "y_1", "z_1"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::Banded),
    );
    let entry_residual = entry.residiual_function.call(1.0, &variables);
    let entry_jac = entry
        .jac_function
        .as_mut()
        .expect("entry-chunk banded jacobian callback should exist")
        .call(1.0, &variables);

    assert_eq!(diag_residual.to_DVectorType(), expected_residual);
    assert_eq!(entry_residual.to_DVectorType(), expected_residual);
    assert_eq!(diag_jac.to_DMatrixType(), expected_jacobian);
    assert_eq!(entry_jac.to_DMatrixType(), expected_jacobian);
}

#[test]
fn compile_lambdified_banded_backend_respects_structural_threshold() {
    let mut jacobian = build_node_major_banded_symbolic_case();
    jacobian
        .symbolic_jacobian_sparse
        .push((0, 2, Expr::Const(1.0e-14)));
    jacobian.set_banded_lambdify_config(BandedLambdifyConfig {
        jacobian_chunking: BandedJacobianChunking::EntryChunks,
        structural_threshold: 1.0e-12,
        ..BandedLambdifyConfig::default()
    });

    let variables = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
    jacobian.compile_lambdified_problem_with_config(
        "x",
        vec!["y_0", "z_0", "y_1", "z_1"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::Banded),
    );

    let jac = jacobian
        .jac_function
        .as_mut()
        .expect("banded jacobian callback should exist");
    let matrix = jac.call(1.0, &variables).to_DMatrixType();
    assert_eq!(matrix[(0, 2)], 0.0);
}

#[test]
fn into_legacy_solver_bundle_collects_callbacks_and_metadata() {
    let mut jacobian = build_parameterized_symbolic_case();
    jacobian.bounds = Some(vec![(0.0, 10.0), (-5.0, 5.0)]);
    jacobian.rel_tolerance_vec = Some(vec![1e-6, 1e-7]);
    jacobian.bandwidth = Some((1, 1));
    jacobian.BC_pos_n_values = vec![(0, 0, 1.0)];
    jacobian.variable_string = vec!["y".to_string(), "z".to_string()];

    jacobian.compile_lambdified_problem_with_config(
        "x",
        vec!["y", "z"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::FaerSparseCol),
    );

    let bundle = jacobian.into_legacy_solver_bundle();

    assert_eq!(
        bundle.variable_string,
        vec!["y".to_string(), "z".to_string()]
    );
    assert_eq!(bundle.bounds_vec, Some(vec![(0.0, 10.0), (-5.0, 5.0)]));
    assert_eq!(bundle.rel_tolerance_vec, Some(vec![1e-6, 1e-7]));
    assert_eq!(bundle.bandwidth, Some((1, 1)));
    assert_eq!(bundle.bc_position_and_value, vec![(0, 0, 1.0)]);
    assert!(bundle.jacobian_function.is_some());
}

#[test]
fn smart_parallel_jacobian_populates_sparse_symbolic_cache() {
    let mut jacobian = build_parameterized_symbolic_case();

    jacobian.calc_jacobian_parallel_smart_optimized();

    let sparse_entries = jacobian.symbolic_jacobian_sparse_entries();
    let coords: Vec<(usize, usize)> = sparse_entries
        .iter()
        .map(|entry| (entry.row, entry.col))
        .collect();

    assert!(
        jacobian.symbolic_jacobian.is_empty(),
        "the default faer sparse backend must not retain a dense zero-filled Jacobian cache"
    );
    assert_eq!(jacobian.symbolic_jacobian_sparse.len(), 4);
    assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
}

#[test]
fn bandwidth_optimized_parallel_jacobian_keeps_sparse_entries_within_band() {
    let mut jacobian = build_bandwidth_symbolic_case();

    jacobian.calc_jacobian_parallel_smart_optimized_with_given_bandwidth();

    let sparse_entries = jacobian.symbolic_jacobian_sparse_entries();
    let coords: Vec<(usize, usize)> = sparse_entries
        .iter()
        .map(|entry| (entry.row, entry.col))
        .collect();

    assert_eq!(
        coords,
        vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
    );
    assert!(coords.iter().all(|(row, col)| row.abs_diff(*col) <= 1));
}

#[test]
fn bandwidth_optimized_banded_jacobian_uses_sparse_first_storage() {
    let mut jacobian = build_bandwidth_symbolic_case();
    jacobian.set_backend_config(BvpBackendConfig::lambdify(BvpMatrixBackend::Banded));

    jacobian.calc_jacobian_parallel_smart_optimized_with_given_bandwidth();

    let coords = jacobian
        .symbolic_jacobian_sparse_entries()
        .iter()
        .map(|entry| (entry.row, entry.col))
        .collect::<Vec<_>>();
    assert_eq!(
        coords,
        vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
    );
    assert!(
        jacobian.symbolic_jacobian.is_empty(),
        "banded callbacks consume sparse entries and must not retain a dense zero-filled cache"
    );
    let timings = jacobian
        .last_symbolic_jacobian_timer_snapshot
        .as_ref()
        .expect("symbolic Jacobian construction should expose internal stage timings");
    for key in [
        "symbolic jacobian variable sets time",
        "symbolic jacobian row differentiation time",
        "symbolic jacobian dense cache materialize time",
        "symbolic jacobian sparse cache flatten time",
    ] {
        assert!(
            timings.get(key).is_some_and(|value| *value >= 0.0),
            "missing non-negative stage timing {key}"
        );
    }
}

#[test]
fn bandwidth_optimized_faer_sparse_jacobian_uses_sparse_first_storage_and_lambdifies() {
    let mut jacobian = build_bandwidth_symbolic_case();
    jacobian.set_backend_config(BvpBackendConfig::lambdify(BvpMatrixBackend::FaerSparseCol));

    jacobian.calc_jacobian_parallel_smart_optimized_with_given_bandwidth();

    let coords = jacobian
        .symbolic_jacobian_sparse_entries()
        .iter()
        .map(|entry| (entry.row, entry.col))
        .collect::<Vec<_>>();
    assert_eq!(
        coords,
        vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
    );
    assert!(
        jacobian.symbolic_jacobian.is_empty(),
        "faer sparse callbacks consume sparse entries and must not retain a dense zero-filled cache"
    );

    jacobian.compile_lambdified_problem_with_config(
        "x",
        vec!["a", "b", "c"],
        BvpBackendConfig::lambdify(BvpMatrixBackend::FaerSparseCol),
    );
    let dense_variables = DVector::from_vec(vec![2.0, 3.0, 4.0]);
    let variables = &*Vectors_type_casting(&dense_variables, "Sparse".to_string());
    let matrix = jacobian
        .jac_function
        .as_mut()
        .expect("faer sparse Jacobian function must exist")
        .call(1.0, variables)
        .to_DMatrixType();
    assert_eq!(
        matrix,
        DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    );
}

#[test]
fn prepare_sparse_aot_problem_uses_sparse_cache_and_bvp_input_order() {
    let mut jacobian = build_parameterized_symbolic_case();
    jacobian.calc_jacobian_parallel_smart_optimized();

    let prepared_bridge = jacobian.prepare_sparse_aot_problem(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
    );
    let prepared = prepared_bridge.as_prepared_problem();

    assert_eq!(prepared.backend_kind, BackendKind::Aot);
    assert_eq!(prepared.matrix_backend, MatrixBackend::SparseCol);
    assert_eq!(prepared.input_names(), &["a", "y", "z"]);
    assert_eq!(prepared.residual_len(), 2);
    assert_eq!(prepared.jacobian_shape(), (2, 2));
    assert_eq!(prepared.jacobian_plan.nnz(), 4);
    assert_eq!(prepared.jacobian_plan.chunks.len(), 1);
    assert_eq!(prepared.jacobian_plan.chunks[0].entries.len(), 4);
}

#[test]
fn prepare_sparse_aot_problem_can_export_banded_prepared_problem() {
    let jacobian = build_node_major_banded_symbolic_case();
    let prepared_bridge = jacobian.prepare_sparse_aot_problem(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
    );

    let prepared = prepared_bridge.as_prepared_problem_for_matrix_backend(MatrixBackend::Banded);
    let PreparedProblem::Banded(prepared) = prepared else {
        panic!("expected banded prepared problem");
    };

    assert_eq!(prepared.backend_kind, BackendKind::Aot);
    assert_eq!(prepared.matrix_backend, MatrixBackend::Banded);
    assert_eq!(prepared.input_names(), &["y_0", "z_0", "y_1", "z_1"]);
    assert_eq!(prepared.residual_len(), 4);
    assert_eq!(prepared.jacobian_shape(), (4, 4));
    assert_eq!(prepared.jacobian_plan.nnz(), 8);
    assert_eq!(prepared.jacobian_plan.structure.kl, 3);
    assert_eq!(prepared.jacobian_plan.structure.ku, 3);
    assert_eq!(
        prepared.jacobian_plan.structure.diagonal_offsets,
        vec![0, 1, -1, 0, 0, 1, -1, 0]
    );
}

#[test]
fn prepared_problem_keys_differ_between_sparse_and_banded_backends() {
    let jacobian = build_node_major_banded_symbolic_case();
    let prepared_bridge = jacobian.prepare_sparse_aot_problem(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
    );

    let sparse_key = prepared_bridge.problem_key_for_matrix_backend(MatrixBackend::SparseCol);
    let banded_key = prepared_bridge.problem_key_for_matrix_backend(MatrixBackend::Banded);

    assert_ne!(sparse_key, banded_key);
}

#[test]
fn prepared_sparse_aot_problem_exposes_machine_aware_auto_parallel_plan() {
    let mut jacobian = build_parameterized_symbolic_case();
    jacobian.calc_jacobian_parallel_smart_optimized();

    let prepared_bridge = jacobian.prepare_sparse_aot_problem(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
    );

    let auto_plan = prepared_bridge.auto_parallel_plan();

    assert!(auto_plan.workers >= 1);
    assert!(auto_plan.min_work_per_job >= 1);
    assert_eq!(
        auto_plan.executor_config,
        prepared_bridge.auto_parallel_executor_config()
    );
    match auto_plan.execution_mode {
        AutoExecutionMode::Sequential => {
            assert!(auto_plan.executor_config.is_none());
            assert_eq!(auto_plan.residual_chunking, ResidualChunkingStrategy::Whole);
            assert_eq!(auto_plan.sparse_chunking, SparseChunkingStrategy::Whole);
        }
        AutoExecutionMode::Parallel => {
            assert!(auto_plan.executor_config.is_some());
            assert!(matches!(
                auto_plan.residual_chunking,
                ResidualChunkingStrategy::ByOutputCount { .. }
            ));
            assert!(matches!(
                auto_plan.sparse_chunking,
                SparseChunkingStrategy::ByRowCount { .. }
            ));
        }
    }
}

#[test]
fn select_sparse_backend_falls_back_to_lambdify_when_aot_is_missing() {
    let mut jacobian = build_parameterized_symbolic_case();
    jacobian.calc_jacobian_parallel_smart_optimized();

    let selection = jacobian.select_sparse_backend(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&AotResolver::new(AotRegistry::new())),
    );

    assert_eq!(selection.requested_backend, BackendKind::Lambdify);
    assert_eq!(selection.effective_backend, SelectedBackendKind::Lambdify);
    assert_eq!(selection.matrix_backend, MatrixBackend::SparseCol);
    assert!(selection.aot_resolution.is_none());
    assert_eq!(selection.prepared_problem.variable_names, vec!["y", "z"]);
}

#[test]
fn select_sparse_backend_prefers_compiled_aot_when_registered_artifact_exists() {
    let mut jacobian = build_parameterized_symbolic_case();
    jacobian.calc_jacobian_parallel_smart_optimized();

    let prepared_bridge = jacobian.prepare_sparse_aot_problem(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
    );
    let prepared = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::sparse(
        prepared_bridge.as_prepared_problem(),
    );
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = tempdir().expect("tempdir should exist");
    let build = AotBuildRequest::new(
        generated_aot_crate_from_prepared_problem(
            "generated_bvp_backend_select_fixture",
            "generated_bvp_backend_select_module",
            &prepared,
        ),
        dir.path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let selection = jacobian.select_sparse_backend(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );

    assert_eq!(selection.requested_backend, BackendKind::Aot);
    assert_eq!(
        selection.effective_backend,
        SelectedBackendKind::AotCompiled
    );
    assert_eq!(selection.matrix_backend, MatrixBackend::SparseCol);
    assert!(selection.is_compiled_aot());
    assert!(selection
        .aot_resolution
        .as_ref()
        .is_some_and(|resolved| resolved.is_compiled()));
}

#[test]
fn prepare_sparse_backend_execution_builds_lambdify_main_path() {
    let mut jacobian = build_parameterized_symbolic_case();
    jacobian.calc_jacobian_parallel_smart_optimized();
    let dense_variables = DVector::from_vec(vec![3.0, 4.0]);
    let variables = &*Vectors_type_casting(&dense_variables, "Sparse".to_string());

    let execution = jacobian.prepare_sparse_backend_execution(
        "x",
        vec!["y", "z"],
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&AotResolver::new(AotRegistry::new())),
    );

    match execution {
        BvpSparseExecutionPlan::LambdifyReady(selected) => {
            assert_eq!(selected.effective_backend, SelectedBackendKind::Lambdify);
        }
        other => panic!("expected LambdifyReady plan, got {other:?}"),
    }

    assert_eq!(
        jacobian.backend_config().backend_kind,
        BvpBackendKind::Lambdify
    );
    assert_eq!(
        jacobian.backend_config().matrix_backend,
        BvpMatrixBackend::FaerSparseCol
    );
    let residual = jacobian.residiual_function.call(1.0, variables);
    let jac = jacobian
        .jac_function
        .as_mut()
        .expect("jacobian function must exist");
    let jacobian_value = jac.call(1.0, variables);

    assert_eq!(
        residual.to_DVectorType(),
        DVector::from_vec(vec![11.0, 10.0])
    );
    assert_eq!(
        jacobian_value.to_DMatrixType(),
        DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0])
    );
}

#[test]
fn prepare_sparse_backend_execution_preserves_compiled_aot_selection_metadata() {
    let mut jacobian = build_parameterized_symbolic_case();
    jacobian.calc_jacobian_parallel_smart_optimized();

    let prepared_bridge = jacobian.prepare_sparse_aot_problem(
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
    );
    let prepared = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::sparse(
        prepared_bridge.as_prepared_problem(),
    );
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = tempdir().expect("tempdir should exist");
    let build = AotBuildRequest::new(
        generated_aot_crate_from_prepared_problem(
            "generated_bvp_backend_execution_fixture",
            "generated_bvp_backend_execution_module",
            &prepared,
        ),
        dir.path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let execution = jacobian.prepare_sparse_backend_execution(
        "x",
        vec!["y", "z"],
        "fixture_residual",
        "fixture_sparse_values",
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );

    match execution {
        BvpSparseExecutionPlan::AotCompiled(selected) => {
            assert_eq!(selected.effective_backend, SelectedBackendKind::AotCompiled);
            assert!(selected
                .aot_resolution
                .as_ref()
                .is_some_and(|resolved| resolved.is_compiled()));
        }
        other => panic!("expected AotCompiled plan, got {other:?}"),
    }

    assert_eq!(jacobian.backend_config().backend_kind, BvpBackendKind::Aot);
    assert_eq!(
        jacobian.backend_config().matrix_backend,
        BvpMatrixBackend::FaerSparseCol
    );
}

#[test]
fn generate_bvp_with_backend_selection_matches_main_lambdify_sparse_path() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut legacy = Jacobian::new();
    legacy.generate_BVP_with_params(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
    );

    let mut selected = Jacobian::new();
    let execution = selected.generate_BVP_with_backend_selection(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&AotResolver::new(AotRegistry::new())),
    );

    match execution {
        BvpSparseExecutionPlan::LambdifyReady(selected_backend) => {
            assert_eq!(
                selected_backend.effective_backend,
                SelectedBackendKind::Lambdify
            );
        }
        other => panic!("expected LambdifyReady plan for real BVP, got {other:?}"),
    }

    let dense_variables = DVector::from_iterator(
        selected.variable_string.len(),
        (0..selected.variable_string.len()).map(|index| 0.2 + index as f64 * 0.01),
    );
    let variables = &*Vectors_type_casting(&dense_variables, "Sparse".to_string());
    let legacy_residual = legacy
        .residiual_function
        .call(1.0, variables)
        .to_DVectorType();
    let selected_residual = selected
        .residiual_function
        .call(1.0, variables)
        .to_DVectorType();
    let legacy_jacobian = legacy
        .jac_function
        .as_mut()
        .expect("legacy jacobian should exist")
        .call(1.0, variables)
        .to_DMatrixType();
    let selected_jacobian = selected
        .jac_function
        .as_mut()
        .expect("selected jacobian should exist")
        .call(1.0, variables)
        .to_DMatrixType();

    assert_eq!(
        selected.backend_config().backend_kind,
        BvpBackendKind::Lambdify
    );
    assert_eq!(
        selected.backend_config().matrix_backend,
        BvpMatrixBackend::FaerSparseCol
    );
    assert_eq!(selected_residual, legacy_residual);
    assert_eq!(selected_jacobian, legacy_jacobian);
}

#[test]
fn generate_bvp_with_atom_discretization_matches_legacy_sparse_path() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut legacy = Jacobian::new();
    legacy.generate_BVP_with_params(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
    );

    let mut atom = Jacobian::new();
    atom.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
    atom.generate_BVP_with_params(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
    );

    let args = DVector::from_vec(
        (0..atom.variable_string.len())
            .map(|index| 0.2 + index as f64 * 0.01)
            .collect(),
    );
    let typed = &*Vectors_type_casting(&args, "Sparse".to_string());

    let legacy_residual = legacy.residiual_function.call(1.0, typed).to_DVectorType();
    let atom_residual = atom.residiual_function.call(1.0, typed).to_DVectorType();
    let legacy_jacobian = legacy
        .jac_function
        .as_mut()
        .expect("legacy jacobian should exist")
        .call(1.0, typed)
        .to_DMatrixType();
    let atom_jacobian = atom
        .jac_function
        .as_mut()
        .expect("atom jacobian should exist")
        .call(1.0, typed)
        .to_DMatrixType();

    assert_eq!(atom.variable_string, legacy.variable_string);
    let mut atom_bc = atom.BC_pos_n_values.clone();
    let mut legacy_bc = legacy.BC_pos_n_values.clone();
    atom_bc.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    legacy_bc.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    assert_eq!(atom_bc, legacy_bc);
    assert_eq!(
        atom.variables_for_all_disrete,
        legacy.variables_for_all_disrete
    );
    assert_eq!(atom_residual.nrows(), legacy_residual.nrows());
    for index in 0..atom_residual.len() {
        assert!(
            (atom_residual[index] - legacy_residual[index]).abs() < 1e-10,
            "residual mismatch at index {index}: atom={}, legacy={}",
            atom_residual[index],
            legacy_residual[index]
        );
    }
    assert_eq!(atom_jacobian.nrows(), legacy_jacobian.nrows());
    assert_eq!(atom_jacobian.ncols(), legacy_jacobian.ncols());
    for row in 0..atom_jacobian.nrows() {
        for col in 0..atom_jacobian.ncols() {
            let atom_value = atom_jacobian[(row, col)];
            let legacy_value = legacy_jacobian[(row, col)];
            assert!(
                (atom_value - legacy_value).abs() < 1e-10,
                "jacobian mismatch at ({row}, {col}): atom={atom_value}, legacy={legacy_value}"
            );
        }
    }
}

#[test]
fn generate_bvp_with_atom_discretization_matches_legacy_banded_path() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let bandwidth = Some((2, 2));
    let mut legacy = Jacobian::new();
    legacy.generate_BVP_with_params(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Banded".to_string(),
        bandwidth,
    );

    let mut atom = Jacobian::new();
    atom.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
    atom.generate_BVP_with_params(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Banded".to_string(),
        bandwidth,
    );

    let args = DVector::from_vec(
        (0..atom.variable_string.len())
            .map(|index| 0.2 + index as f64 * 0.01)
            .collect(),
    );
    let typed = &*Vectors_type_casting(&args, "Banded".to_string());

    let legacy_residual = legacy.residiual_function.call(1.0, typed).to_DVectorType();
    let atom_residual = atom.residiual_function.call(1.0, typed).to_DVectorType();
    let legacy_jacobian = legacy
        .jac_function
        .as_mut()
        .expect("legacy banded jacobian should exist")
        .call(1.0, typed)
        .to_DMatrixType();
    let atom_jacobian = atom
        .jac_function
        .as_mut()
        .expect("atom-native banded jacobian should exist")
        .call(1.0, typed)
        .to_DMatrixType();

    assert_eq!(atom.variable_string, legacy.variable_string);
    assert!(
        atom.symbolic_jacobian.is_empty(),
        "AtomView Banded preparation must remain sparse-first"
    );
    assert_eq!(atom_residual.nrows(), legacy_residual.nrows());
    for index in 0..atom_residual.len() {
        assert!(
            (atom_residual[index] - legacy_residual[index]).abs() < 1e-10,
            "banded residual mismatch at index {index}: atom={}, legacy={}",
            atom_residual[index],
            legacy_residual[index]
        );
    }
    assert_eq!(atom_jacobian.nrows(), legacy_jacobian.nrows());
    assert_eq!(atom_jacobian.ncols(), legacy_jacobian.ncols());
    for row in 0..atom_jacobian.nrows() {
        for col in 0..atom_jacobian.ncols() {
            assert!(
                (atom_jacobian[(row, col)] - legacy_jacobian[(row, col)]).abs() < 1e-10,
                "banded Jacobian mismatch at ({row}, {col}): atom={}, legacy={}",
                atom_jacobian[(row, col)],
                legacy_jacobian[(row, col)]
            );
        }
    }
}

#[test]
fn atom_two_point_residual_row_diagnostics() {
    let ne = NonlinEquation::TwoPointBVP;
    let eq_system = ne.setup();
    let values = ne.values();
    let arg = "x".to_string();
    let border_conditions = ne.boundary_conditions();

    let mut legacy = Jacobian::new();
    legacy.generate_BVP_with_params(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        None,
        -1.0,
        None,
        Some(24),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
    );

    let mut atom = Jacobian::new();
    atom.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
    atom.generate_BVP_with_params(
        eq_system,
        values,
        arg,
        None,
        -1.0,
        None,
        Some(24),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
    );

    let args = DVector::from_element(atom.variable_string.len(), 0.7);
    let typed = &*Vectors_type_casting(&args, "Sparse".to_string());
    let legacy_residual = legacy.residiual_function.call(1.0, typed).to_DVectorType();
    let atom_residual = atom.residiual_function.call(1.0, typed).to_DVectorType();

    let mut max_diff = 0.0f64;
    let mut max_index = 0usize;
    for index in 0..legacy_residual.len() {
        let diff = (legacy_residual[index] - atom_residual[index]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }

    println!(
        "[two-point residual row diagnostics] max_index={}, max_diff={:.6e}, legacy={}, atom={}",
        max_index, max_diff, legacy_residual[max_index], atom_residual[max_index]
    );

    for index in 0..legacy_residual.len().min(8) {
        println!(
            "[two-point residual row diagnostics] row={index}, legacy={:.6e}, atom={:.6e}, diff={:.6e}",
            legacy_residual[index],
            atom_residual[index],
            (legacy_residual[index] - atom_residual[index]).abs()
        );
    }
}

#[test]
fn atom_two_point_sparse_bundle_residual_diagnostics() {
    let ne = NonlinEquation::TwoPointBVP;
    let eq_system = ne.setup();
    let values = ne.values();
    let arg = "x".to_string();
    let border_conditions = ne.boundary_conditions();

    let legacy = Jacobian::new();
    let legacy_bundle = legacy
        .try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system.clone(),
            values.clone(),
            arg.clone(),
            None,
            -1.0,
            None,
            Some(24),
            None,
            None,
            border_conditions.clone(),
            None,
            None,
            "forward".to_string(),
            "Sparse".to_string(),
            None,
            BackendSelectionPolicy::PreferAotThenLambdify,
            None,
        )
        .expect("legacy sparse bundle should build");

    let mut atom = Jacobian::new();
    atom.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
    let atom_bundle = atom
        .try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system,
            values,
            arg,
            None,
            -1.0,
            None,
            Some(24),
            None,
            None,
            border_conditions,
            None,
            None,
            "forward".to_string(),
            "Sparse".to_string(),
            None,
            BackendSelectionPolicy::PreferAotThenLambdify,
            None,
        )
        .expect("atom sparse bundle should build");

    let args = DVector::from_element(atom_bundle.variable_string.len(), 0.7);
    let typed = &*Vectors_type_casting(&args, "Sparse".to_string());
    let legacy_residual = legacy_bundle
        .residual_function
        .as_ref()
        .expect("legacy sparse bundle should have residual callback")
        .call(1.0, typed)
        .to_DVectorType();
    let atom_residual = atom_bundle
        .residual_function
        .as_ref()
        .expect("atom sparse bundle should have residual callback")
        .call(1.0, typed)
        .to_DVectorType();

    let mut max_diff = 0.0f64;
    let mut max_index = 0usize;
    for index in 0..legacy_residual.len() {
        let diff = (legacy_residual[index] - atom_residual[index]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }

    println!(
        "[two-point sparse bundle diagnostics] max_index={}, max_diff={:.6e}, legacy={}, atom={}",
        max_index, max_diff, legacy_residual[max_index], atom_residual[max_index]
    );
}

#[test]
fn atom_two_point_sparse_bundle_residual_diagnostics_large_grid() {
    let ne = NonlinEquation::TwoPointBVP;
    let eq_system = ne.setup();
    let values = ne.values();
    let arg = "x".to_string();
    let border_conditions = ne.boundary_conditions();

    let legacy = Jacobian::new();
    let mut atom = Jacobian::new();
    atom.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);

    let legacy_bundle = legacy
        .try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system.clone(),
            values.clone(),
            arg.clone(),
            None,
            -1.0,
            None,
            Some(72),
            None,
            None,
            border_conditions.clone(),
            None,
            None,
            "forward".to_string(),
            "Sparse".to_string(),
            None,
            BackendSelectionPolicy::PreferAotThenLambdify,
            None,
        )
        .expect("legacy sparse bundle should build");

    let atom_bundle = atom
        .try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system,
            values,
            arg,
            None,
            -1.0,
            None,
            Some(72),
            None,
            None,
            border_conditions,
            None,
            None,
            "forward".to_string(),
            "Sparse".to_string(),
            None,
            BackendSelectionPolicy::PreferAotThenLambdify,
            None,
        )
        .expect("atom sparse bundle should build");

    let args = DVector::from_vec(
        (0..atom_bundle.variable_string.len())
            .map(|index| 0.2 + index as f64 * 0.01)
            .collect(),
    );
    let typed = &*Vectors_type_casting(&args, "Sparse".to_string());
    let legacy_residual = legacy_bundle
        .residual_function
        .as_ref()
        .expect("legacy sparse bundle should have residual callback")
        .call(1.0, typed)
        .to_DVectorType();
    let atom_residual = atom_bundle
        .residual_function
        .as_ref()
        .expect("atom sparse bundle should have residual callback")
        .call(1.0, typed)
        .to_DVectorType();

    let mut max_diff = 0.0f64;
    let mut max_index = 0usize;
    for index in 0..legacy_residual.len() {
        let diff = (legacy_residual[index] - atom_residual[index]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }

    println!(
        "[two-point sparse bundle diagnostics large-grid] max_index={}, max_diff={:.6e}, legacy={}, atom={}",
        max_index,
        max_diff,
        legacy_residual[max_index],
        atom_residual[max_index]
    );
}

#[test]
fn atom_two_point_sparse_bundle_residual_diagnostics_with_explicit_h() {
    let ne = NonlinEquation::TwoPointBVP;
    let eq_system = ne.setup();
    let values = ne.values();
    let arg = "x".to_string();
    let border_conditions = ne.boundary_conditions();
    let t0 = -1.0;
    let t_end = 1.0;
    let n_steps = 72usize;
    let h = (t_end - t0) / n_steps as f64;

    let legacy = Jacobian::new();
    let mut atom = Jacobian::new();
    atom.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);

    let legacy_bundle = legacy
        .try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system.clone(),
            values.clone(),
            arg.clone(),
            None,
            t0,
            None,
            Some(n_steps),
            Some(h),
            None,
            border_conditions.clone(),
            None,
            None,
            "forward".to_string(),
            "Sparse".to_string(),
            None,
            BackendSelectionPolicy::PreferAotThenLambdify,
            None,
        )
        .expect("legacy sparse bundle should build");

    let atom_bundle = atom
        .try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system,
            values,
            arg,
            None,
            t0,
            None,
            Some(n_steps),
            Some(h),
            None,
            border_conditions,
            None,
            None,
            "forward".to_string(),
            "Sparse".to_string(),
            None,
            BackendSelectionPolicy::PreferAotThenLambdify,
            None,
        )
        .expect("atom sparse bundle should build");

    let args = DVector::from_vec(
        (0..atom_bundle.variable_string.len())
            .map(|index| 0.2 + index as f64 * 0.01)
            .collect(),
    );
    let typed = &*Vectors_type_casting(&args, "Sparse".to_string());
    let legacy_residual = legacy_bundle
        .residual_function
        .as_ref()
        .expect("legacy sparse bundle should have residual callback")
        .call(1.0, typed)
        .to_DVectorType();
    let atom_residual = atom_bundle
        .residual_function
        .as_ref()
        .expect("atom sparse bundle should have residual callback")
        .call(1.0, typed)
        .to_DVectorType();

    let mut max_diff = 0.0f64;
    let mut max_index = 0usize;
    for index in 0..legacy_residual.len() {
        let diff = (legacy_residual[index] - atom_residual[index]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }

    println!(
        "[two-point sparse bundle diagnostics explicit-h] max_index={}, max_diff={:.6e}, legacy={}, atom={}",
        max_index,
        max_diff,
        legacy_residual[max_index],
        atom_residual[max_index]
    );
}

#[test]
fn atom_two_point_direct_generate_residual_diagnostics_with_explicit_h() {
    let ne = NonlinEquation::TwoPointBVP;
    let eq_system = ne.setup();
    let values = ne.values();
    let arg = "x".to_string();
    let border_conditions = ne.boundary_conditions();
    let t0 = -1.0;
    let t_end = 1.0;
    let n_steps = 72usize;
    let h = (t_end - t0) / n_steps as f64;

    let mut legacy = Jacobian::new();
    legacy.generate_BVP_with_params(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        None,
        t0,
        None,
        Some(n_steps),
        Some(h),
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
    );

    let mut atom = Jacobian::new();
    atom.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
    atom.generate_BVP_with_params(
        eq_system,
        values,
        arg,
        None,
        t0,
        None,
        Some(n_steps),
        Some(h),
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
    );

    let args = DVector::from_vec(
        (0..atom.variable_string.len())
            .map(|index| 0.2 + index as f64 * 0.01)
            .collect(),
    );
    let typed = &*Vectors_type_casting(&args, "Sparse".to_string());
    let legacy_residual = legacy.residiual_function.call(1.0, typed).to_DVectorType();
    let atom_residual = atom.residiual_function.call(1.0, typed).to_DVectorType();

    let mut max_diff = 0.0f64;
    let mut max_index = 0usize;
    for index in 0..legacy_residual.len() {
        let diff = (legacy_residual[index] - atom_residual[index]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }

    println!(
        "[two-point direct generate explicit-h] max_index={}, max_diff={:.6e}, legacy={}, atom={}",
        max_index, max_diff, legacy_residual[max_index], atom_residual[max_index]
    );
    println!(
        "[two-point direct generate explicit-h] legacy_row_expr={}",
        legacy.vector_of_functions[max_index]
    );
    println!(
        "[two-point direct generate explicit-h] atom_row_expr={}",
        atom.vector_of_functions[max_index]
    );
}

#[test]
fn generate_bvp_with_backend_selection_reports_compiled_aot_for_real_bvp() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        0.0,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let residual_strategy =
        recommended_residual_chunking_for_parallelism(staged.vector_of_functions.len(), 4);
    let jacobian_strategy =
        recommended_row_chunking_for_parallelism(staged.vector_of_functions.len(), 4);

    let prepared_bridge = staged.prepare_sparse_aot_problem(
        "eval_bvp_residual",
        "eval_bvp_sparse_values",
        residual_strategy,
        jacobian_strategy,
    );
    let prepared = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::sparse(
        prepared_bridge.as_prepared_problem(),
    );
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = tempdir().expect("tempdir should exist");
    let build = AotBuildRequest::new(
        generated_aot_crate_from_prepared_problem(
            "generated_bvp_real_selection_fixture",
            "generated_bvp_real_selection_module",
            &prepared,
        ),
        dir.path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let mut selected = Jacobian::new();
    let execution = selected.generate_BVP_with_backend_selection(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );

    match execution {
        BvpSparseExecutionPlan::AotCompiled(selected_backend) => {
            assert_eq!(
                selected_backend.effective_backend,
                SelectedBackendKind::AotCompiled
            );
            assert!(selected_backend.is_compiled_aot());
        }
        other => panic!("expected AotCompiled plan for real BVP, got {other:?}"),
    }

    assert_eq!(selected.backend_config().backend_kind, BvpBackendKind::Aot);
    assert_eq!(
        selected.backend_config().matrix_backend,
        BvpMatrixBackend::FaerSparseCol
    );
}

#[test]
fn sparse_solver_provider_evaluates_real_bvp_lambdify_plan() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut jacobian = Jacobian::new();
    let execution = jacobian.generate_BVP_with_backend_selection(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&AotResolver::new(AotRegistry::new())),
    );
    let args: Vec<f64> = (0..jacobian.variable_string.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();
    let expected_residual = jacobian
        .residiual_function
        .call(
            1.0,
            &*Vectors_type_casting(&DVector::from_column_slice(&args), "Sparse".to_string()),
        )
        .to_DVectorType();
    let expected_dense_jacobian = jacobian
        .jac_function
        .as_mut()
        .expect("jacobian callback should exist")
        .call(
            1.0,
            &*Vectors_type_casting(&DVector::from_column_slice(&args), "Sparse".to_string()),
        )
        .to_DMatrixType();

    let mut provider = jacobian.sparse_solver_provider(execution);
    let mut residual_out = vec![0.0; provider.residual_len()];
    let mut jacobian_values_out = vec![0.0; provider.jacobian_structure().nnz()];
    provider.residual_into(&args, &mut residual_out);
    provider.jacobian_values_into(&args, &mut jacobian_values_out);

    let expected_values: Vec<f64> = provider
        .jacobian_structure()
        .row_indices
        .iter()
        .zip(provider.jacobian_structure().col_indices.iter())
        .map(|(&row, &col)| expected_dense_jacobian[(row, col)])
        .collect();

    assert!(provider.is_runtime_callable());
    assert_eq!(provider.effective_backend(), SelectedBackendKind::Lambdify);
    assert_eq!(residual_out, expected_residual.as_slice().to_vec());
    assert_eq!(jacobian_values_out, expected_values);
}

#[test]
fn sparse_solver_provider_exposes_compiled_aot_metadata_for_real_bvp() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        0.0,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let residual_strategy =
        recommended_residual_chunking_for_parallelism(staged.vector_of_functions.len(), 4);
    let jacobian_strategy =
        recommended_row_chunking_for_parallelism(staged.vector_of_functions.len(), 4);
    let prepared_bridge = staged.prepare_sparse_aot_problem(
        "eval_bvp_residual",
        "eval_bvp_sparse_values",
        residual_strategy,
        jacobian_strategy,
    );
    let prepared = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::sparse(
        prepared_bridge.as_prepared_problem(),
    );
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = tempdir().expect("tempdir should exist");
    let build = AotBuildRequest::new(
        generated_aot_crate_from_prepared_problem(
            "generated_bvp_real_provider_fixture",
            "generated_bvp_real_provider_module",
            &prepared,
        ),
        dir.path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let mut jacobian = Jacobian::new();
    let execution = jacobian.generate_BVP_with_backend_selection(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );

    let provider = jacobian.sparse_solver_provider(execution);
    assert!(!provider.is_runtime_callable());
    assert_eq!(
        provider.effective_backend(),
        SelectedBackendKind::AotCompiled
    );
    assert!(provider
        .resolved_aot_artifact()
        .is_some_and(|resolved| resolved.is_compiled()));
    assert_eq!(provider.jacobian_shape(), (12, 12));
    assert_eq!(provider.jacobian_structure().nnz(), 28);
}

#[test]
fn sparse_solver_provider_executes_linked_compiled_aot_backend_for_real_bvp() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        0.0,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let residual_strategy =
        recommended_residual_chunking_for_parallelism(staged.vector_of_functions.len(), 4);
    let jacobian_strategy =
        recommended_row_chunking_for_parallelism(staged.vector_of_functions.len(), 4);
    let prepared_bridge = staged.prepare_sparse_aot_problem(
        "eval_bvp_residual",
        "eval_bvp_sparse_values",
        residual_strategy,
        jacobian_strategy,
    );
    let problem_key = prepared_bridge.problem_key();
    let input_name_strings = prepared_bridge
        .param_names
        .iter()
        .chain(prepared_bridge.variable_names.iter())
        .cloned()
        .collect::<Vec<_>>();
    let residual_input_name_strings = input_name_strings.clone();
    let jacobian_input_name_strings = input_name_strings.clone();
    let residual_exprs = prepared_bridge.residuals.clone();
    let sparse_exprs = prepared_bridge.sparse_entries.clone();
    let residual_len = prepared_bridge.shape.0;
    let shape = prepared_bridge.shape;
    let nnz = sparse_exprs.len();

    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        problem_key.clone(),
        residual_len,
        shape,
        nnz,
        Arc::new(move |args, out| {
            let input_names = residual_input_name_strings
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>();
            for (slot, expr) in out.iter_mut().zip(residual_exprs.iter()) {
                *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args);
            }
        }),
        Arc::new(move |args, out| {
            let input_names = jacobian_input_name_strings
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>();
            for (slot, (_, _, expr)) in out.iter_mut().zip(sparse_exprs.iter()) {
                *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args);
            }
        }),
    ));

    let prepared = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::sparse(
        prepared_bridge.as_prepared_problem(),
    );
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = tempdir().expect("tempdir should exist");
    let build = AotBuildRequest::new(
        generated_aot_crate_from_prepared_problem(
            "generated_bvp_real_linked_fixture",
            "generated_bvp_real_linked_module",
            &prepared,
        ),
        dir.path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let mut jacobian = Jacobian::new();
    let execution = jacobian.generate_BVP_with_backend_selection(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );

    let args: Vec<f64> = (0..jacobian.variable_string.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();
    let expected_input_names = jacobian
        .variable_string
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    let expected_residual = jacobian
        .vector_of_functions
        .iter()
        .map(|expr| expr.lambdify_borrowed_thread_safe(&expected_input_names)(&args))
        .collect::<Vec<_>>();

    let mut provider = jacobian.sparse_solver_provider(execution);
    let mut residual_out = vec![0.0; provider.residual_len()];
    let mut jacobian_values_out = vec![0.0; provider.jacobian_structure().nnz()];
    provider.residual_into(&args, &mut residual_out);
    provider.jacobian_values_into(&args, &mut jacobian_values_out);

    assert!(provider.is_runtime_callable());
    assert_eq!(
        provider.effective_backend(),
        SelectedBackendKind::AotCompiled
    );
    assert_eq!(residual_out, expected_residual);
    assert_eq!(
        jacobian_values_out.len(),
        provider.jacobian_structure().nnz()
    );

    unregister_linked_sparse_backend(&problem_key);
}

#[test]
fn generate_bvp_with_backend_selection_and_chunking_preserves_explicit_strategies() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut jacobian = Jacobian::new();
    let residual_strategy = ResidualChunkingStrategy::ByOutputCount {
        max_outputs_per_chunk: 3,
    };
    let jacobian_strategy = SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 2 };

    let execution = jacobian.generate_BVP_with_backend_selection_and_chunking(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        Some((2, 2)),
        BackendSelectionPolicy::LambdifyOnly,
        None,
        residual_strategy,
        jacobian_strategy,
    );

    let prepared = &execution.selected().prepared_problem;
    assert_eq!(prepared.residual_strategy, residual_strategy);
    assert_eq!(prepared.jacobian_strategy, jacobian_strategy);
}

#[test]
fn sparse_solver_bundle_collects_real_bvp_lambdify_runtime_and_metadata() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut jacobian = Jacobian::new();
    let execution = jacobian.generate_BVP_with_backend_selection(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&AotResolver::new(AotRegistry::new())),
    );

    let mut bundle = jacobian.into_sparse_solver_bundle(execution);
    let args: Vec<f64> = (0..bundle.variable_string.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();
    let y = &*Vectors_type_casting(&DVector::from_column_slice(&args), "Sparse".to_string());

    let residual = bundle
        .residual_call(1.0, y)
        .expect("lambdify bundle should expose residual callback")
        .to_DVectorType();
    let jacobian_dense = bundle
        .jacobian_call(1.0, y)
        .expect("lambdify bundle should expose jacobian callback")
        .to_DMatrixType();

    assert!(bundle.is_runtime_callable());
    assert_eq!(bundle.effective_backend(), SelectedBackendKind::Lambdify);
    assert_eq!(bundle.jacobian_shape(), (12, 12));
    assert_eq!(bundle.sparse_structure.nnz(), 28);
    assert_eq!(bundle.bandwidth, Some((2, 1)));
    assert_eq!(bundle.residual_len(), 12);
    assert_eq!(residual.len(), 12);
    assert_eq!(jacobian_dense.nrows(), 12);
    assert_eq!(jacobian_dense.ncols(), 12);
}

#[test]
fn sparse_solver_bundle_collects_real_bvp_compiled_aot_metadata() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        0.0,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let residual_strategy =
        recommended_residual_chunking_for_parallelism(staged.vector_of_functions.len(), 4);
    let jacobian_strategy =
        recommended_row_chunking_for_parallelism(staged.vector_of_functions.len(), 4);
    let prepared_bridge = staged.prepare_sparse_aot_problem(
        "eval_bvp_residual",
        "eval_bvp_sparse_values",
        residual_strategy,
        jacobian_strategy,
    );
    let prepared = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::sparse(
        prepared_bridge.as_prepared_problem(),
    );
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = tempdir().expect("tempdir should exist");
    let build = AotBuildRequest::new(
        generated_aot_crate_from_prepared_problem(
            "generated_bvp_real_bundle_fixture",
            "generated_bvp_real_bundle_module",
            &prepared,
        ),
        dir.path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let mut jacobian = Jacobian::new();
    let execution = jacobian.generate_BVP_with_backend_selection(
        eq_system,
        values,
        arg,
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        None,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );
    let bundle = jacobian.into_sparse_solver_bundle(execution);

    assert!(!bundle.is_runtime_callable());
    assert_eq!(bundle.effective_backend(), SelectedBackendKind::AotCompiled);
    assert!(bundle
        .resolved_aot_artifact()
        .is_some_and(|resolved| resolved.is_compiled()));
    assert_eq!(bundle.jacobian_shape(), (12, 12));
    assert_eq!(bundle.sparse_structure.nnz(), 28);
    assert_eq!(bundle.variable_string.len(), 12);
}
