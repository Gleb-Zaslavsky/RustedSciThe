//! AOT-only correctness gates for the prepared AtomView runtime contract.
//!
//! These tests deliberately do not solve through Lambdify. They verify the
//! fixed input/output ABI, Banded layout ownership and warm telemetry on the
//! Atom plan itself. Cross-route numerical comparisons stay in the separate
//! Lambdify-vs-AOT story modules.

use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
use crate::symbolic::View::parser::parse;
use crate::symbolic::View::state::Symbol;
use crate::symbolic::bvp::aot_telemetry::{
    BvpAotChunking, BvpAotColdStage, BvpAotFrontend, BvpAotMatrixLayout, BvpAotTelemetryMode,
};
use crate::symbolic::bvp::atom_aot::{
    AtomAotMatrixLayout, AtomAotPreparedPlan, AtomAotRuntimeError, AtomAotRuntimeState,
};
use crate::symbolic::codegen::codegen_aot_lifecycle::{
    AotArtifactInspection, AotArtifactState, AotFailureDiagnostics, AotFailureKind,
    AotLifecycleError, AotLifecycleStage, quarantine_generated_tree,
};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;

use crate::numerical::BVP_Damp::test_common::AotStoryProtocol;
use crate::symbolic::codegen::codegen_aot_runtime_link::LinkedSparseAotBackend;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;

macro_rules! aot_contract_report {
    ($name:ident) => {
        let _test_report = crate::Utils::test_reporting::TestReportCapture::new(
            "BVP_Damp_AOT",
            concat!(module_path!(), "::", stringify!($name)),
        );
    };
}

fn atom(source: &str) -> Atom {
    parse(source).expect("AOT-only fixture atom must parse")
}

#[test]
fn atomview_banded_aot_contract_is_self_contained_and_typed() {
    aot_contract_report!(atomview_banded_aot_contract_is_self_contained_and_typed);
    let plan = AtomAotPreparedPlan::from_parts_with_telemetry(
        vec![atom("x + p"), atom("y - p")],
        vec![
            SparseAtomJacobianEntry {
                row: 0,
                col: 0,
                value: atom("1"),
            },
            SparseAtomJacobianEntry {
                row: 1,
                col: 1,
                value: atom("1"),
            },
        ],
        vec!["p".to_string(), "x".to_string(), "y".to_string()],
        vec![
            Symbol::new(crate::wrap_symbol!("p")),
            Symbol::new(crate::wrap_symbol!("x")),
            Symbol::new(crate::wrap_symbol!("y")),
        ],
        1,
        AtomAotMatrixLayout::Banded {
            rows: 2,
            cols: 2,
            kl: 0,
            ku: 0,
            slots: 2,
        },
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
        BvpAotTelemetryMode::Detailed,
    )
    .expect("valid AtomView Banded AOT plan");

    assert_eq!(plan.matrix_layout().value_count(), 2);
    assert_eq!(plan.parameter_count(), 1);
    let mut residual = vec![0.0; 2];
    plan.try_execute_residual_callback(&[2.0, 3.0, 4.0], &mut residual, |_, out| {
        out.copy_from_slice(&[1.0, 2.0]);
        Ok::<(), AtomAotRuntimeError>(())
    })
    .expect("typed residual callback should succeed");
    let mut jacobian = vec![0.0; 2];
    plan.try_execute_jacobian_callback(&[2.0, 3.0, 4.0], &mut jacobian, |_, out| {
        out.copy_from_slice(&[1.0, 1.0]);
        Ok::<(), AtomAotRuntimeError>(())
    })
    .expect("typed Jacobian callback should succeed");
    assert_eq!(residual, [1.0, 2.0]);
    assert_eq!(jacobian, [1.0, 1.0]);
    plan.record_cold_stage_duration(
        BvpAotColdStage::Lowering,
        std::time::Duration::from_nanos(1),
    );

    let snapshot = plan.telemetry_snapshot();
    assert_eq!(snapshot.mode, BvpAotTelemetryMode::Detailed);
    assert_eq!(snapshot.identity.frontend, BvpAotFrontend::AtomView);
    assert_eq!(snapshot.identity.matrix_layout, BvpAotMatrixLayout::Banded);
    assert_eq!(snapshot.identity.residual_chunking, BvpAotChunking::Whole);
    assert_eq!(snapshot.identity.jacobian_chunking, BvpAotChunking::Whole);
    assert_eq!(snapshot.residual_calls, 1);
    assert_eq!(snapshot.jacobian_calls, 1);
    assert_eq!(snapshot.residual_chunks, 1);
    assert_eq!(snapshot.jacobian_chunks, 1);
    assert!(snapshot.lowering > std::time::Duration::ZERO);
}

#[test]
fn atomview_aot_contract_rejects_missing_parameter_binding_before_abi() {
    aot_contract_report!(atomview_aot_contract_rejects_missing_parameter_binding_before_abi);
    let plan = AtomAotPreparedPlan::from_parts_with_telemetry(
        vec![atom("x + p")],
        vec![SparseAtomJacobianEntry {
            row: 0,
            col: 0,
            value: atom("1"),
        }],
        vec!["p".to_string(), "x".to_string()],
        vec![
            Symbol::new(crate::wrap_symbol!("p")),
            Symbol::new(crate::wrap_symbol!("x")),
        ],
        1,
        AtomAotMatrixLayout::SparseCsc {
            rows: 1,
            cols: 1,
            nnz: 1,
        },
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
        BvpAotTelemetryMode::Counters,
    )
    .expect("valid parameterized AtomView AOT plan");

    assert!(matches!(
        plan.validate_runtime_input(&[2.0]),
        Err(AtomAotRuntimeError::InputLength {
            expected: 2,
            actual: 1
        })
    ));
    assert_eq!(plan.telemetry_snapshot().errors, 1);
}

#[test]
fn atomview_compact_banded_owned_callback_preserves_slots_and_invalidation() {
    aot_contract_report!(atomview_compact_banded_owned_callback_preserves_slots_and_invalidation);
    let mut plan = AtomAotPreparedPlan::from_parts_with_telemetry(
        vec![atom("x"), atom("y")],
        vec![
            SparseAtomJacobianEntry {
                row: 0,
                col: 0,
                value: atom("1"),
            },
            SparseAtomJacobianEntry {
                row: 1,
                col: 1,
                value: atom("1"),
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        vec![
            Symbol::new(crate::wrap_symbol!("x")),
            Symbol::new(crate::wrap_symbol!("y")),
        ],
        0,
        AtomAotMatrixLayout::BandedCompact {
            rows: 2,
            cols: 2,
            kl: 0,
            ku: 0,
            slots: 2,
        },
        ResidualChunkingStrategy::Whole,
        SparseChunkingStrategy::Whole,
        BvpAotTelemetryMode::Detailed,
    )
    .expect("valid compact-Banded AtomView AOT plan");

    let backend = LinkedSparseAotBackend::new(
        "compact-banded-owned-fixture",
        2,
        (2, 2),
        2,
        Arc::new(|_, out| out.copy_from_slice(&[2.0, 3.0])),
        Arc::new(|_, out| out.copy_from_slice(&[4.0, 5.0])),
    )
    .with_banded_compact_layout(2, 2, 0, 0);
    plan.bind_linked_runtime(None, backend);
    assert_eq!(plan.runtime_binding().state(), AtomAotRuntimeState::Linked);

    let mut residual = [0.0; 2];
    plan.try_execute_bound_residual_callback(&[1.0, 1.0], &mut residual)
        .expect("owned compact-Banded residual callback should succeed");
    assert_eq!(residual, [2.0, 3.0]);
    let mut compact_jacobian = [0.0; 2];
    plan.try_execute_bound_jacobian_callback(&[1.0, 1.0], &mut compact_jacobian)
        .expect("owned compact-Banded Jacobian callback should preserve slots");
    assert_eq!(compact_jacobian, [4.0, 5.0]);

    let snapshot = plan.telemetry_snapshot();
    assert_eq!(snapshot.residual_calls, 1);
    assert_eq!(snapshot.jacobian_calls, 1);
    plan.invalidate_linked_runtime();
    assert_eq!(plan.runtime_binding().state(), AtomAotRuntimeState::Unbound);
    assert!(matches!(
        plan.try_execute_bound_jacobian_callback(&[1.0, 1.0], &mut compact_jacobian),
        Err(AtomAotRuntimeError::RuntimeNotLinked)
    ));
}

#[test]
fn aot_story_protocol_is_explicit_and_validated() {
    aot_contract_report!(aot_story_protocol_is_explicit_and_validated);
    let protocol = AotStoryProtocol {
        n_steps: 1000,
        cold_repetitions: 3,
        warm_repetitions: 5,
        cold_cooldown_ms: 0,
        warm_cooldown_ms: 1000,
        clean_artifacts: true,
        worker_threads: 1,
    };

    protocol
        .validate()
        .expect("the canonical AOT story protocol should be valid");
    let summary = protocol.summary();
    for field in [
        "n_steps=1000",
        "cold_repetitions=3",
        "warm_repetitions=5",
        "cold_cooldown_ms=0",
        "warm_cooldown_ms=1000",
        "clean_artifacts=true",
        "worker_threads=1",
    ] {
        assert!(summary.contains(field), "protocol summary omitted {field}");
    }

    assert!(
        AotStoryProtocol {
            n_steps: 1,
            ..protocol
        }
        .validate()
        .is_err()
    );
    assert!(
        AotStoryProtocol {
            cold_repetitions: 0,
            ..protocol
        }
        .validate()
        .is_err()
    );
}

#[test]
fn aot_artifact_failure_contract_preserves_state_and_quarantine_diagnostics() {
    aot_contract_report!(aot_artifact_failure_contract_preserves_state_and_quarantine_diagnostics);
    let directory = tempdir().expect("temporary artifact directory should exist");
    let crate_dir = directory.path().join("generated-bvp");
    let marker = crate_dir.join("ready.marker");
    let dynamic_output = crate_dir.join("generated.dll");
    let static_output = crate_dir.join("generated.lib");

    let missing =
        AotArtifactInspection::inspect(&crate_dir, &marker, &static_output, &dynamic_output);
    assert_eq!(missing.state, AotArtifactState::Missing);

    fs::create_dir_all(&crate_dir).expect("generated artifact directory should be writable");
    fs::write(&marker, b"materialized").expect("materialization marker should be writable");
    let materialized =
        AotArtifactInspection::inspect(&crate_dir, &marker, &static_output, &dynamic_output);
    assert_eq!(materialized.state, AotArtifactState::Materialized);
    assert!(!materialized.is_reusable());

    fs::remove_file(&marker).expect("stale marker should be removable");
    fs::write(&dynamic_output, b"stale output").expect("stale output should be writable");
    let stale =
        AotArtifactInspection::inspect(&crate_dir, &marker, &static_output, &dynamic_output);
    assert_eq!(stale.state, AotArtifactState::Stale);

    let quarantined = quarantine_generated_tree(&crate_dir, "bvp-failure-fixture")
        .expect("stale artifact should be quarantined")
        .expect("existing stale artifact should produce a quarantine path");
    assert!(quarantined.exists());
    assert!(!crate_dir.exists());

    let mut diagnostics = AotFailureDiagnostics::new(
        AotLifecycleStage::Build,
        AotFailureKind::PartialArtifact,
        "bvp-failure-fixture",
        "generated output was incomplete before rebuild",
    );
    diagnostics.attempts = 2;
    diagnostics.quarantine_attempted = true;
    diagnostics.cleanup_completed = true;
    diagnostics.inspection = Some(stale);
    let error = AotLifecycleError::new(diagnostics.clone());
    assert_eq!(error.diagnostics, diagnostics);
    assert!(error.to_string().contains("PartialArtifact"));

    fs::remove_dir_all(quarantined).expect("quarantined fixture should be cleaned up");
}
