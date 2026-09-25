//! Debug correctness gates for production-shaped Sparse/Banded evaluator work.
//!
//! These are deliberately callback-level tests. They do not claim a linear
//! backend result and therefore cannot accidentally turn Dense into evidence
//! for a large production problem. Release timing belongs to the separate
//! large-system performance story planned in `TODO.md`.

use super::native_jacobian::{NativeJacobianStorage, try_prepare_native_atomview_jacobian_runtime};
use super::story_support::{
    chain_equations, chain_state, legacy_frontend, max_matrix_diff, max_vector_diff,
    native_frontend, prepare_chain,
};
use super::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetrySnapshot, IvpWarmStage,
};
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::symbolic_ivp::SharedIvpParameterValues;
use nalgebra::DVector;
use std::sync::{Arc, RwLock};
use std::time::Instant;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn cold_ms(snapshot: &IvpTelemetrySnapshot, stage: IvpColdStage) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

fn warm_ms(snapshot: &IvpTelemetrySnapshot, stage: IvpWarmStage) -> f64 {
    snapshot.warm_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

#[test]
fn atomview_native_large_chain_matches_exprlegacy_callbacks() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::large_system_story_tests::atomview_native_large_chain_matches_exprlegacy_callbacks",
    );
    reportln!(
        "[LSODE2 large callback correctness/stages] dimensions=[32,128,256]; telemetry=detailed; Dense excluded; callback-only timing"
    );
    reportln!(
        "route | dimension | prepare_ms | residual_wall_ms | jacobian_wall_ms | expr_to_atom_ms | diff_ms | simplify_ms | sparse_pattern_ms | layout_ms | residual_compile_ms | residual_lambdify_ms | jacobian_compile_ms | jacobian_lambdify_ms | binding_ms | residual_eval_ms | residual_output_ms | jacobian_eval_ms | jacobian_output_ms | residual_calls | jacobian_calls | scalar_evals | copies | allocated_bytes | residual_diff | jacobian_diff"
    );
    for dimension in [32usize, 128, 256] {
        let state = chain_state(dimension);
        let expr_telemetry = IvpTelemetry::detailed();
        let expr_prepare_started = Instant::now();
        let expr = prepare_chain(
            dimension,
            legacy_frontend(),
            IvpLambdifyExecutionPolicy::Sequential,
            expr_telemetry.clone(),
        );
        let expr_prepare_ms = expr_prepare_started.elapsed().as_secs_f64() * 1.0e3;
        let expr_residual_started = Instant::now();
        let expr_residual = expr
            .try_evaluate_residual(0.5, &state)
            .expect("ExprLegacy residual should evaluate");
        let expr_residual_wall_ms = expr_residual_started.elapsed().as_secs_f64() * 1.0e3;
        let expr_jacobian_started = Instant::now();
        let expr_jacobian = expr
            .try_evaluate_jacobian(0.5, &state)
            .expect("ExprLegacy Jacobian should evaluate");
        let expr_jacobian_wall_ms = expr_jacobian_started.elapsed().as_secs_f64() * 1.0e3;
        let expr_snapshot = expr_telemetry.snapshot();

        let native_telemetry = IvpTelemetry::detailed();
        let native_prepare_started = Instant::now();
        let native = prepare_chain(
            dimension,
            native_frontend(),
            IvpLambdifyExecutionPolicy::Sequential,
            native_telemetry.clone(),
        );
        let native_prepare_ms = native_prepare_started.elapsed().as_secs_f64() * 1.0e3;
        let native_residual_started = Instant::now();
        let native_residual = native
            .try_evaluate_residual(0.5, &state)
            .expect("AtomViewNative residual should evaluate");
        let native_residual_wall_ms = native_residual_started.elapsed().as_secs_f64() * 1.0e3;
        let native_jacobian_started = Instant::now();
        let native_jacobian = native
            .try_evaluate_jacobian(0.5, &state)
            .expect("AtomViewNative Jacobian should evaluate");
        let native_jacobian_wall_ms = native_jacobian_started.elapsed().as_secs_f64() * 1.0e3;
        let native_snapshot = native_telemetry.snapshot();

        assert_eq!(expr_residual.len(), dimension);
        assert_eq!(native_residual.len(), dimension);
        assert_eq!(expr_jacobian.shape(), (dimension, dimension));
        assert_eq!(native_jacobian.shape(), (dimension, dimension));
        assert!(
            max_vector_diff(&expr_residual, &native_residual) <= 1.0e-10,
            "residual drift at dimension {dimension}"
        );
        assert!(
            max_matrix_diff(&expr_jacobian, &native_jacobian) <= 1.0e-10,
            "Jacobian drift at dimension {dimension}"
        );

        for (
            route,
            snapshot,
            prepare_ms,
            residual_wall_ms,
            jacobian_wall_ms,
            residual_diff,
            jacobian_diff,
        ) in [
            (
                "ExprLegacy",
                &expr_snapshot,
                expr_prepare_ms,
                expr_residual_wall_ms,
                expr_jacobian_wall_ms,
                max_vector_diff(&expr_residual, &native_residual),
                max_matrix_diff(&expr_jacobian, &native_jacobian),
            ),
            (
                "AtomViewNative",
                &native_snapshot,
                native_prepare_ms,
                native_residual_wall_ms,
                native_jacobian_wall_ms,
                max_vector_diff(&expr_residual, &native_residual),
                max_matrix_diff(&expr_jacobian, &native_jacobian),
            ),
        ] {
            reportln!(
                "{route} | {dimension:>9} | {prepare_ms:>11.3} | {residual_wall_ms:>16.3} | {jacobian_wall_ms:>17.3} | {expr_to_atom:>15.3} | {diff:>7.3} | {simplify:>11.3} | {sparse:>17.3} | {layout:>9.3} | {res_compile:>20.3} | {res_lambdify:>21.3} | {jac_compile:>20.3} | {jac_lambdify:>21.3} | {binding:>11.3} | {res_eval:>16.3} | {res_output:>18.3} | {jac_eval:>16.3} | {jac_output:>18.3} | {res_calls:>13} | {jac_calls:>13} | {scalar:>12} | {copies:>6} | {allocated:>15} | {residual_diff:.3e} | {jacobian_diff:.3e}",
                route = route,
                dimension = dimension,
                prepare_ms = prepare_ms,
                residual_wall_ms = residual_wall_ms,
                jacobian_wall_ms = jacobian_wall_ms,
                expr_to_atom = cold_ms(snapshot, IvpColdStage::ExprToAtom),
                diff = cold_ms(snapshot, IvpColdStage::SymbolicDifferentiation),
                simplify = cold_ms(snapshot, IvpColdStage::Simplification),
                sparse = cold_ms(snapshot, IvpColdStage::SparsePattern),
                layout = cold_ms(snapshot, IvpColdStage::LayoutPlanning),
                res_compile = cold_ms(snapshot, IvpColdStage::ResidualCompilation),
                res_lambdify = cold_ms(snapshot, IvpColdStage::ResidualLambdification),
                jac_compile = cold_ms(snapshot, IvpColdStage::JacobianCompilation),
                jac_lambdify = cold_ms(snapshot, IvpColdStage::JacobianLambdification),
                binding = warm_ms(snapshot, IvpWarmStage::ArgumentBinding),
                res_eval = warm_ms(snapshot, IvpWarmStage::ResidualEvaluation),
                res_output = warm_ms(snapshot, IvpWarmStage::ResidualOutputAssembly),
                jac_eval = warm_ms(snapshot, IvpWarmStage::JacobianEvaluation),
                jac_output = warm_ms(snapshot, IvpWarmStage::JacobianOutputAssembly),
                res_calls = snapshot.residual_evaluations,
                jac_calls = snapshot.jacobian_evaluations,
                scalar = snapshot.scalar_evaluations,
                copies = snapshot.copies,
                allocated = snapshot.allocated_bytes,
                residual_diff = residual_diff,
                jacobian_diff = jacobian_diff,
            );
        }
    }
}

#[test]
fn atomview_native_rebind_preserves_large_chain_callback_parity() {
    let dimension = 128;
    let state = chain_state(dimension);
    let expr = prepare_chain(
        dimension,
        legacy_frontend(),
        IvpLambdifyExecutionPolicy::Sequential,
        IvpTelemetry::counters(),
    );
    let native = prepare_chain(
        dimension,
        native_frontend(),
        IvpLambdifyExecutionPolicy::Sequential,
        IvpTelemetry::counters(),
    );

    for parameters in [
        [20.0, 4.0, 0.20, 0.010],
        [12.0, 2.5, 0.35, 0.025],
        [30.0, 7.0, 0.05, 0.005],
    ] {
        expr.set_parameter_values(nalgebra::DVector::from_column_slice(&parameters))
            .expect("ExprLegacy rebind should succeed");
        native
            .set_parameter_values(nalgebra::DVector::from_column_slice(&parameters))
            .expect("AtomViewNative rebind should succeed");
        let expr_residual = expr
            .try_evaluate_residual(0.75, &state)
            .expect("ExprLegacy rebound residual should evaluate");
        let native_residual = native
            .try_evaluate_residual(0.75, &state)
            .expect("AtomViewNative rebound residual should evaluate");
        let expr_jacobian = expr
            .try_evaluate_jacobian(0.75, &state)
            .expect("ExprLegacy rebound Jacobian should evaluate");
        let native_jacobian = native
            .try_evaluate_jacobian(0.75, &state)
            .expect("AtomViewNative rebound Jacobian should evaluate");

        assert!(max_vector_diff(&expr_residual, &native_residual) <= 1.0e-10);
        assert!(max_matrix_diff(&expr_jacobian, &native_jacobian) <= 1.0e-10);
    }
}

#[test]
#[ignore = "large debug gate: native Sparse/Banded layout parity; run explicitly before release baseline"]
fn atomview_native_large_chain_sparse_banded_layout_parity() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::large_system_story_tests::atomview_native_large_chain_sparse_banded_layout_parity",
    );
    let dimensions = std::env::var("LSODE2_LARGE_LAYOUT_DIMENSIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 2)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| vec![512, 1024]);

    reportln!(
        "[LSODE2 large native layout] dimensions={dimensions:?}; Dense excluded; matrix assembly is caller-owned"
    );
    reportln!(
        "dimension | nnz | band_kl | band_ku | sparse_band_diff | sparse_calls | banded_calls"
    );

    for dimension in dimensions {
        let equations = chain_equations(dimension);
        let variables = (0..dimension)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        let sparse_telemetry = IvpTelemetry::counters();
        let banded_telemetry = IvpTelemetry::counters();
        let parameter_handle: SharedIvpParameterValues =
            Arc::new(RwLock::new(DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010])));
        let mut sparse = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            Some(&[
                "k".to_string(),
                "d".to_string(),
                "q".to_string(),
                "nl".to_string(),
            ]),
            Some(parameter_handle.clone()),
            NativeJacobianStorage::SparseTriplets,
            sparse_telemetry.clone(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("large native sparse runtime should prepare");
        let mut banded = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            Some(&[
                "k".to_string(),
                "d".to_string(),
                "q".to_string(),
                "nl".to_string(),
            ]),
            Some(parameter_handle),
            NativeJacobianStorage::Banded { bandwidth: None },
            banded_telemetry.clone(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("large native banded runtime should prepare");
        let state = chain_state(dimension);
        let pattern = sparse.sparse_pattern();
        let mut sparse_values = vec![0.0; pattern.len()];
        sparse
            .try_evaluate_sparse_values_into(0.5, &state, &mut sparse_values)
            .expect("large native sparse callback should evaluate");
        let (band_kl, band_ku, band_len) = banded
            .banded_layout()
            .expect("large native banded runtime should expose compact layout");
        let mut banded_values = vec![0.0; band_len];
        banded
            .try_evaluate_banded_values_into(0.5, &state, &mut banded_values)
            .expect("large native banded callback should evaluate");
        let banded_matrix = Banded::from_vec(dimension, band_kl, band_ku, banded_values)
            .expect("large native compact band should be structurally valid");
        let max_diff = pattern
            .iter()
            .zip(sparse_values.iter())
            .map(|(&(row, col), sparse_value)| (sparse_value - banded_matrix[(row, col)]).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff <= 1.0e-12,
            "Sparse/Banded drift at dimension {dimension}"
        );

        let sparse_snapshot = sparse_telemetry.snapshot();
        let banded_snapshot = banded_telemetry.snapshot();
        reportln!(
            "{dimension:>9} | {:>3} | {:>7} | {:>7} | {:>16.3e} | {:>12} | {:>12}",
            pattern.len(),
            band_kl,
            band_ku,
            max_diff,
            sparse_snapshot.jacobian_evaluations,
            banded_snapshot.jacobian_evaluations,
        );
    }
}
