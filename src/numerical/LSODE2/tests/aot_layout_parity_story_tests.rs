//! Debug parity gates for generated Sparse and compact-Banded callbacks.
//!
//! These tests deliberately use one small but genuinely coupled tridiagonal
//! problem. The same prepared equations are evaluated through ExprLegacy-AOT,
//! AtomViewNative-AOT and AtomViewNative-Lambdify, so a layout mismatch cannot
//! be hidden by comparing unrelated fixtures.

use crate::Utils::test_reporting::TestReportCapture;
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::ivp_telemetry::{IvpLambdifyExecutionPolicy, IvpTelemetry};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use crate::symbolic::symbolic_ivp_generated::{
    SelectedSymbolicIvpBackendKind, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_banded_backend, prepare_generated_symbolic_ivp_sparse_backend,
};
use nalgebra::DVector;
use tempfile::tempdir;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn fixture() -> (Vec<Expr>, Vec<String>, String, Vec<String>, DVector<f64>) {
    (
        vec![
            Expr::parse_expression("p*t + 2*y0 - y1 + y0*y0"),
            Expr::parse_expression("q*y0 + 3*y1 - y2 + exp(-t)"),
            Expr::parse_expression("-y1 + 4*y2 + p"),
        ],
        vec!["y0".to_string(), "y1".to_string(), "y2".to_string()],
        "t".to_string(),
        vec!["p".to_string(), "q".to_string()],
        DVector::from_vec(vec![0.5, 1.25]),
    )
}

fn options(
    frontend: IvpSymbolicAssemblyBackend,
    parameter_names: &[String],
    parameter_values: &DVector<f64>,
) -> SymbolicIvpProblemOptions {
    SymbolicIvpProblemOptions::new()
        .with_symbolic_assembly_backend(frontend)
        .with_equation_parameters(parameter_names.to_vec())
        .with_equation_parameter_values(parameter_values.clone())
}

fn build_config(parent: &std::path::Path) -> SymbolicIvpGeneratedBackendConfig {
    SymbolicIvpGeneratedBackendConfig::defaults()
        .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Debug,
        })
        .with_output_parent_dir(Some(parent.to_path_buf()))
}

fn max_vector_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn assert_sparse_values_match_dense(
    rows: &[usize],
    cols: &[usize],
    values: &[f64],
    dense: &nalgebra::DMatrix<f64>,
) -> f64 {
    assert_eq!(rows.len(), cols.len());
    assert_eq!(rows.len(), values.len());
    rows.iter()
        .zip(cols)
        .zip(values)
        .map(|((&row, &col), &value)| (value - dense[(row, col)]).abs())
        .fold(0.0, f64::max)
}

#[test]
fn chunked_aot_sparse_policies_preserve_callback_values() {
    let _report = TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_layout_parity_story_tests::chunked_aot_sparse_policies_preserve_callback_values",
    );
    let (equations, variables, time_arg, parameter_names, parameter_values) = fixture();
    let output = tempdir().expect("AOT output directory should exist");
    let prepared = prepare_generated_symbolic_ivp_sparse_backend(
        equations,
        variables,
        time_arg,
        options(
            IvpSymbolicAssemblyBackend::AtomView,
            &parameter_names,
            &parameter_values,
        ),
        build_config(output.path())
            .with_residual_chunking_strategy(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 1,
            })
            .with_sparse_jacobian_chunking_strategy(SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: 2,
            }),
    )
    .expect("chunked AtomView sparse AOT preparation should succeed");
    let linked = prepared
        .linked_backend
        .as_ref()
        .expect("chunked AOT sparse callbacks should be linked");
    assert!(linked.residual_chunks.len() > 1);
    assert!(linked.jacobian_value_chunks.len() > 1);

    let args = [
        0.375,
        parameter_values[0],
        parameter_values[1],
        0.25,
        -0.5,
        0.75,
    ];
    let mut whole_residual = vec![0.0; linked.residual_len];
    let mut whole_jacobian = vec![0.0; linked.nnz];
    linked
        .try_residual_eval(&args, &mut whole_residual)
        .expect("whole residual callback should evaluate");
    linked
        .try_jacobian_values_eval(&args, &mut whole_jacobian)
        .expect("whole sparse Jacobian callback should evaluate");

    reportln!(
        "[LSODE2 AOT chunk policy parity] chunks=residual:{} jacobian:{}; break-even excluded",
        linked.residual_chunks.len(),
        linked.jacobian_value_chunks.len()
    );
    reportln!(
        "policy | residual_diff | jacobian_diff | aot_dispatches | aot_parallel_dispatches | aot_chunks"
    );
    for (label, policy) in [
        ("Sequential", IvpLambdifyExecutionPolicy::Sequential),
        (
            "Parallel",
            IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        ),
        ("Auto", IvpLambdifyExecutionPolicy::Auto { min_work: 1 }),
    ] {
        let telemetry = IvpTelemetry::detailed();
        let mut residual = vec![0.0; linked.residual_len];
        let mut jacobian = vec![0.0; linked.nnz];
        linked
            .try_residual_eval_with_policy(&args, &mut residual, policy, &telemetry)
            .expect("policy residual callback should evaluate");
        linked
            .try_jacobian_values_eval_with_policy(&args, &mut jacobian, policy, &telemetry)
            .expect("policy sparse Jacobian callback should evaluate");
        let snapshot = telemetry.snapshot();
        let residual_diff = max_vector_diff(&residual, &whole_residual);
        let jacobian_diff = max_vector_diff(&jacobian, &whole_jacobian);
        assert!(residual_diff <= 1.0e-12, "{label} residual drift");
        assert!(jacobian_diff <= 1.0e-12, "{label} Jacobian drift");
        assert_eq!(snapshot.errors, 0, "{label} typed callback errors");
        assert_eq!(
            snapshot.aot_chunk_dispatches, 2,
            "{label} AOT dispatch count"
        );
        assert!(snapshot.aot_chunks >= 2, "{label} AOT chunk count");
        reportln!(
            "{label} | {residual_diff:.3e} | {jacobian_diff:.3e} | {} | {} | {}",
            snapshot.aot_chunk_dispatches,
            snapshot.aot_parallel_dispatches,
            snapshot.aot_chunks,
        );
    }
}

#[test]
fn sparse_and_banded_aot_layouts_match_native_lambdify_after_rebind() {
    let _report = TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_layout_parity_story_tests::sparse_and_banded_aot_layouts_match_native_lambdify_after_rebind",
    );
    let (equations, variables, time_arg, parameter_names, parameter_values) = fixture();
    let output = tempdir().expect("AOT output directory should exist");
    let state = DVector::from_vec(vec![0.25, -0.5, 0.75]);
    let time = 0.375;

    let legacy = prepare_generated_symbolic_ivp_sparse_backend(
        equations.clone(),
        variables.clone(),
        time_arg.clone(),
        options(
            IvpSymbolicAssemblyBackend::ExprLegacy,
            &parameter_names,
            &parameter_values,
        ),
        build_config(output.path()),
    )
    .expect("ExprLegacy sparse AOT preparation should succeed");
    let native = prepare_generated_symbolic_ivp_sparse_backend(
        equations.clone(),
        variables.clone(),
        time_arg.clone(),
        options(
            IvpSymbolicAssemblyBackend::AtomView,
            &parameter_names,
            &parameter_values,
        ),
        build_config(output.path()),
    )
    .expect("AtomView sparse AOT preparation should succeed");
    let lambdify = prepare_symbolic_ivp_problem(
        equations.clone(),
        variables.clone(),
        time_arg.clone(),
        options(
            IvpSymbolicAssemblyBackend::AtomView,
            &parameter_names,
            &parameter_values,
        ),
    )
    .expect("AtomView Lambdify preparation should succeed");

    assert_eq!(
        legacy.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );
    assert_eq!(
        native.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );
    assert_eq!(legacy.jacobian_structure, native.jacobian_structure);
    assert_eq!(legacy.jacobian_structure.nnz(), 7);

    let legacy_linked = legacy
        .linked_backend
        .as_ref()
        .expect("ExprLegacy sparse callback should be linked");
    let native_linked = native
        .linked_backend
        .as_ref()
        .expect("AtomView sparse callback should be linked");
    let args = [
        time,
        parameter_values[0],
        parameter_values[1],
        state[0],
        state[1],
        state[2],
    ];
    let mut legacy_residual = vec![0.0; 3];
    let mut native_residual = vec![0.0; 3];
    let mut legacy_values = vec![0.0; legacy.jacobian_structure.nnz()];
    let mut native_values = vec![0.0; native.jacobian_structure.nnz()];
    legacy_linked
        .try_residual_eval(&args, &mut legacy_residual)
        .expect("ExprLegacy sparse residual callback should evaluate");
    native_linked
        .try_residual_eval(&args, &mut native_residual)
        .expect("AtomView sparse residual callback should evaluate");
    legacy_linked
        .try_jacobian_values_eval(&args, &mut legacy_values)
        .expect("ExprLegacy sparse Jacobian callback should evaluate");
    native_linked
        .try_jacobian_values_eval(&args, &mut native_values)
        .expect("AtomView sparse Jacobian callback should evaluate");

    let reference_residual = lambdify
        .try_evaluate_residual(time, &state)
        .expect("Lambdify residual should evaluate");
    let reference_jacobian = lambdify
        .try_evaluate_jacobian(time, &state)
        .expect("Lambdify Jacobian should evaluate");
    let sparse_dense_drift = assert_sparse_values_match_dense(
        &native.jacobian_structure.row_indices,
        &native.jacobian_structure.col_indices,
        &native_values,
        &reference_jacobian,
    );
    assert!(max_vector_diff(&legacy_residual, &native_residual) <= 1.0e-12);
    assert!(max_vector_diff(&native_residual, reference_residual.as_slice()) <= 1.0e-12);
    assert!(max_vector_diff(&legacy_values, &native_values) <= 1.0e-12);
    assert!(sparse_dense_drift <= 1.0e-12);

    let banded = prepare_generated_symbolic_ivp_banded_backend(
        equations.clone(),
        variables.clone(),
        time_arg.clone(),
        (1, 1),
        options(
            IvpSymbolicAssemblyBackend::AtomView,
            &parameter_names,
            &parameter_values,
        ),
        build_config(output.path()),
    )
    .expect("AtomView compact-Banded AOT preparation should succeed");
    assert_eq!(
        banded.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );
    let banded_linked = banded
        .linked_backend
        .as_ref()
        .expect("compact-Banded callback should be linked");
    let expected_band_len = (1 + 1 + 1) * 3;
    let mut band_values = vec![0.0; expected_band_len];
    banded_linked
        .try_jacobian_values_eval(&args, &mut band_values)
        .expect("compact-Banded callback should evaluate all boundary slots");
    let banded_matrix = Banded::from_vec(3, 1, 1, band_values)
        .expect("compact-Banded callback should produce valid storage");
    let banded_drift = (0usize..3)
        .flat_map(|row| (0usize..3).map(move |col| (row, col)))
        .map(|(row, col)| {
            if col.abs_diff(row) <= 1 {
                (banded_matrix[(row, col)] - reference_jacobian[(row, col)]).abs()
            } else {
                reference_jacobian[(row, col)].abs()
            }
        })
        .fold(0.0, f64::max);
    assert!(banded_drift <= 1.0e-12);

    let banded_key = banded.problem_key.clone();
    let banded_resolver = banded
        .updated_resolver
        .clone()
        .expect("BuildIfMissing should return the compact-Banded resolver");
    crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_sparse_backend(
        banded_key.as_str(),
    );
    crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_residual_backend(
        banded_key.as_str(),
    );
    let require_prebuilt = prepare_generated_symbolic_ivp_banded_backend(
        equations.clone(),
        variables.clone(),
        time_arg.clone(),
        (1, 1),
        options(
            IvpSymbolicAssemblyBackend::AtomView,
            &parameter_names,
            &parameter_values,
        ),
        build_config(output.path())
            .with_resolver(Some(banded_resolver))
            .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt),
    )
    .expect("RequirePrebuilt should reconnect the compact-Banded artifact");
    assert_eq!(
        require_prebuilt.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );
    assert!(
        require_prebuilt.build_result.is_none(),
        "RequirePrebuilt must not compile a second compact-Banded artifact"
    );
    let require_linked = require_prebuilt
        .linked_backend
        .as_ref()
        .expect("RequirePrebuilt should reconnect compact-Banded callbacks");
    let mut require_values = vec![0.0; expected_band_len];
    require_linked
        .try_jacobian_values_eval(&args, &mut require_values)
        .expect("reconnected compact-Banded callback should evaluate");
    assert!(
        max_vector_diff(&require_values, &banded_matrix.as_slice()) <= 1.0e-12,
        "reconnected compact-Banded slots must match the original publication"
    );

    let rebound = DVector::from_vec(vec![1.75, -0.25]);
    lambdify
        .set_parameter_values(rebound.clone())
        .expect("Lambdify numeric rebind should succeed");
    let rebound_state = DVector::from_vec(vec![-0.1, 0.4, 0.9]);
    let rebound_time = 0.625;
    let rebound_args = [
        rebound_time,
        rebound[0],
        rebound[1],
        rebound_state[0],
        rebound_state[1],
        rebound_state[2],
    ];
    let mut rebound_legacy = vec![0.0; 3];
    let mut rebound_native = vec![0.0; 3];
    legacy_linked
        .try_residual_eval(&rebound_args, &mut rebound_legacy)
        .expect("rebound ExprLegacy AOT residual should evaluate");
    native_linked
        .try_residual_eval(&rebound_args, &mut rebound_native)
        .expect("rebound AtomView AOT residual should evaluate");
    let rebound_reference = lambdify
        .try_evaluate_residual(rebound_time, &rebound_state)
        .expect("rebound Lambdify residual should evaluate");
    assert!(max_vector_diff(&rebound_legacy, rebound_native.as_slice()) <= 1.0e-12);
    assert!(max_vector_diff(&rebound_native, rebound_reference.as_slice()) <= 1.0e-12);

    let mut nonfinite = vec![0.0; 3];
    let nonfinite_error = native_linked
        .try_residual_eval(
            &[
                time,
                f64::NAN,
                parameter_values[1],
                state[0],
                state[1],
                state[2],
            ],
            &mut nonfinite,
        )
        .expect_err("AOT callback must reject non-finite input");
    assert!(matches!(
        nonfinite_error,
        crate::symbolic::codegen::codegen_aot_runtime_link::LinkedAotCallbackError::NonFiniteInput { .. }
    ));

    reportln!(
        "[LSODE2 AOT layout parity] fixture=tridiagonal-3; sparse_nnz={}; band=(1,1); rebind=ok; nonfinite=typed-error",
        native.jacobian_structure.nnz()
    );
    reportln!("route | residual_drift | jacobian_drift | layout");
    reportln!(
        "ExprLegacy-AOT vs AtomView-AOT | {:.3e} | {:.3e} | fixed-sparse-order",
        max_vector_diff(&legacy_residual, &native_residual),
        max_vector_diff(&legacy_values, &native_values)
    );
    reportln!(
        "AtomView-AOT vs Lambdify | {:.3e} | {:.3e} | compact-banded-slots {:.3e}",
        max_vector_diff(&native_residual, reference_residual.as_slice()),
        sparse_dense_drift,
        banded_drift
    );
}
