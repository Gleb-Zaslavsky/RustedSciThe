//! Toolchain-backed AOT stories.
//!
//! These tests are intentionally ignored in the ordinary debug suite.  They
//! invoke an external compiler and therefore belong to the explicit release
//! AOT gate, while the fast callback and lifecycle contracts stay covered by
//! the non-ignored stories in this directory.

use std::path::PathBuf;

use super::super::api::{
    RadauAotConfig, RadauConfig, RadauErrorKind, RadauExecution, RadauExecutionPolicy,
    RadauFrontend, RadauMatrixLayout, RadauOutputPolicy, RadauProblem, RadauSolver,
    RadauTelemetryMode,
};
use super::super::new::aot::AotPlan;
use super::super::new::config::{RadauAssembly, RadauMatrixLayout as InternalMatrixLayout};
use super::super::new::telemetry::{RadauTelemetry, RadauTelemetryMode as InternalTelemetryMode};
use crate::numerical::ivp_workloads::{
    WorkloadKind, build_workload, parameter_continuation_target,
};
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
use crate::symbolic::symbolic_engine::Expr;
use tabled::Tabled;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

#[derive(Debug, Tabled)]
struct AotStageRow {
    frontend: String,
    workload: String,
    layout: String,
    prepare_ms: String,
    cache_ms: String,
    lowering_ms: String,
    source_ms: String,
    materialize_ms: String,
    build_ms: String,
    link_ms: String,
    publication_ms: String,
    cold_callback_ms: String,
    warm_callback_ms: String,
    builds: u64,
    links: u64,
    chunks: u64,
    workers: u64,
}

fn parameterized_decay_problem() -> RadauProblem {
    RadauProblem::new(
        vec![
            Expr::parse_expression("-rate*y"),
            Expr::parse_expression("-rate*z"),
        ],
        vec!["y".to_owned(), "z".to_owned()],
        "t",
    )
    .with_parameters(vec!["rate".to_owned()])
}

fn parameterized_decay_jacobian() -> Vec<Expr> {
    vec![
        Expr::parse_expression("-rate"),
        Expr::parse_expression("0"),
        Expr::parse_expression("0"),
        Expr::parse_expression("-rate"),
    ]
}

fn renamed_parameterized_decay_problem() -> RadauProblem {
    RadauProblem::new(
        vec![
            Expr::parse_expression("-gain*y"),
            Expr::parse_expression("-gain*z"),
        ],
        vec!["y".to_owned(), "z".to_owned()],
        "t",
    )
    .with_parameters(vec!["gain".to_owned()])
}

fn invalidation_producer_problem() -> RadauProblem {
    RadauProblem::new(
        vec![
            Expr::parse_expression("-handoff_rate*y"),
            Expr::parse_expression("-handoff_rate*z"),
        ],
        vec!["y".to_owned(), "z".to_owned()],
        "t",
    )
    .with_parameters(vec!["handoff_rate".to_owned()])
}

fn invalidation_schema_mismatch_problem() -> RadauProblem {
    RadauProblem::new(
        vec![
            Expr::parse_expression("-handoff_gain*y"),
            Expr::parse_expression("-handoff_gain*z"),
        ],
        vec!["y".to_owned(), "z".to_owned()],
        "t",
    )
    .with_parameters(vec!["handoff_gain".to_owned()])
}

fn aot_config(output_dir: PathBuf) -> RadauAotConfig {
    // The test uses tcc explicitly so its lifecycle is reproducible on the
    // same Windows toolchain used by the LSODE2 AOT release stories.
    RadauAotConfig::build_if_missing_release(output_dir).with_c_compiler("tcc")
}

fn rebuild_aot_config(output_dir: PathBuf) -> RadauAotConfig {
    RadauAotConfig::rebuild_always_release(output_dir).with_c_compiler("tcc")
}

fn workload_problem(kind: WorkloadKind, dimension: usize) -> (RadauProblem, Vec<f64>, Vec<f64>) {
    let workload = build_workload(kind, dimension);
    let problem = RadauProblem::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
    )
    .with_parameters(workload.parameter_names);
    (
        problem,
        workload.initial_state.as_slice().to_vec(),
        workload.parameter_values.as_slice().to_vec(),
    )
}

fn workload_t_bound(kind: WorkloadKind) -> f64 {
    match kind {
        WorkloadKind::DiffusionChain | WorkloadKind::StiffScalar => 0.01,
        WorkloadKind::Robertson | WorkloadKind::CombustionLike | WorkloadKind::ThreeBody => 0.002,
    }
}

#[test]
#[ignore = "release AOT callback contract; requires an available tcc toolchain"]
fn aot_dense_callbacks_preserve_decay_sign_and_parameters() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_dense_callbacks_preserve_decay_sign_and_parameters",
    );
    let output_dir = tempfile::tempdir().expect("isolated Radau AOT output directory");
    let mut telemetry = RadauTelemetry::new(InternalTelemetryMode::Timings);
    let prepared = AotPlan::prepare(
        RadauAssembly::ExprLegacy,
        InternalMatrixLayout::Dense,
        vec![Expr::parse_expression("-rate*y")],
        None,
        "t".to_owned(),
        vec!["y".to_owned()],
        vec!["rate".to_owned()],
        aot_config(output_dir.path().to_path_buf()).generated,
        &mut telemetry,
    )
    .expect("AOT callback preparation");
    let plan = prepared.plan;
    let mut workspace = plan.workspace();
    let mut residual = [f64::NAN];
    plan.evaluate_residual(
        0.0,
        &[1.0],
        &[2.0],
        &mut residual,
        &mut workspace,
        &mut telemetry,
    )
    .expect("AOT residual callback");
    let mut jacobian = [f64::NAN];
    plan.evaluate_jacobian(
        0.0,
        &[1.0],
        &[2.0],
        InternalMatrixLayout::Dense,
        &mut jacobian,
        &mut workspace,
        &mut telemetry,
    )
    .expect("AOT Jacobian callback");
    reportln!(
        "[Radau AOT callback contract] residual={:.12e} jacobian={:.12e}",
        residual[0],
        jacobian[0]
    );
    assert!((residual[0] + 2.0).abs() < 1.0e-12);
    assert!((jacobian[0] + 2.0).abs() < 1.0e-12);
}

#[test]
#[ignore = "release AOT matrix; requires an available tcc toolchain"]
fn aot_frontend_layout_matrix_solves_and_reports_provenance() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_frontend_layout_matrix_solves_and_reports_provenance",
    );
    let combinations = [
        (RadauFrontend::ExprLegacy, RadauMatrixLayout::Dense),
        (RadauFrontend::ExprLegacy, RadauMatrixLayout::Sparse),
        (
            RadauFrontend::ExprLegacy,
            RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        ),
        (RadauFrontend::AtomViewNative, RadauMatrixLayout::Dense),
        (RadauFrontend::AtomViewNative, RadauMatrixLayout::Sparse),
        (
            RadauFrontend::AtomViewNative,
            RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        ),
    ];

    for (index, (frontend, matrix_layout)) in combinations.into_iter().enumerate() {
        let output_dir = tempfile::tempdir().expect("isolated Radau AOT output directory");
        let config = RadauConfig {
            t_bound: 0.2,
            first_step: Some(0.05),
            max_step: 0.05,
            rtol: 1.0e-8,
            atol: 1.0e-10,
            execution: RadauExecution::Aot,
            frontend,
            matrix_layout,
            telemetry: RadauTelemetryMode::Timings,
            execution_policy: RadauExecutionPolicy::Parallel { min_work: 1 },
            // This is a cold matrix story. RebuildAlways prevents an earlier
            // ignored test from satisfying BuildIfMissing through the
            // process-local linked registry.
            aot: Some(rebuild_aot_config(output_dir.path().to_path_buf())),
            ..RadauConfig::default()
        };
        let problem = parameterized_decay_problem().with_jacobian(parameterized_decay_jacobian());
        let mut solver = RadauSolver::prepare(problem, config).unwrap_or_else(|error| {
            panic!("AOT preparation failed for {frontend:?}/{matrix_layout:?}: {error}")
        });
        let first = solver
            .solve_with_parameters(&[1.0, 1.0], &[2.0])
            .unwrap_or_else(|error| {
                panic!("AOT solve failed for {frontend:?}/{matrix_layout:?}: {error}")
            });
        let first_expected = (-0.4_f64).exp();
        let first_error = first
            .y
            .iter()
            .map(|value| (value - first_expected).abs())
            .fold(0.0, f64::max);
        assert!(
            first_error < 2.0e-6,
            "AOT numerical drift for {frontend:?}/{matrix_layout:?}: actual={} expected={} error={first_error:e}",
            first.y[0],
            first_expected
        );
        assert_eq!(first.telemetry().counters["aot_runtime_ready"], 1);
        assert_eq!(first.telemetry().counters["aot_build_successes"], 1);
        assert_eq!(first.telemetry().counters["aot_link_successes"], 1);
        assert!(!first.telemetry().aot_artifact_keys.is_empty());

        let continued = solver
            .continue_with_parameters(&[1.0, 1.0], &[1.0])
            .unwrap_or_else(|error| {
                panic!("AOT continuation failed for {frontend:?}/{matrix_layout:?}: {error}")
            });
        let continuation_expected = (-0.2_f64).exp();
        let continuation_error = continued
            .y
            .iter()
            .map(|value| (value - continuation_expected).abs())
            .fold(0.0, f64::max);
        assert!(
            continuation_error < 2.0e-6,
            "AOT continuation drift for {frontend:?}/{matrix_layout:?}: {continuation_error:e}"
        );
        assert_eq!(
            continued.telemetry().counters["aot_build_attempts"],
            first.telemetry().counters["aot_build_attempts"],
            "continuation must not rebuild the prepared AOT route"
        );
        reportln!(
            "[Radau AOT matrix] case={index} frontend={frontend:?} layout={matrix_layout:?} first={:.12e} continued={:.12e} artifact_keys={:?}",
            first.y[0],
            continued.y[0],
            continued.telemetry().aot_artifact_keys
        );
    }
}

#[test]
#[ignore = "release AOT policy matrix; requires an available tcc toolchain"]
fn aot_execution_policy_matrix_preserves_values_and_continuation() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_execution_policy_matrix_preserves_values_and_continuation",
    );
    for execution_policy in [
        RadauExecutionPolicy::Sequential,
        RadauExecutionPolicy::Parallel { min_work: 1 },
        RadauExecutionPolicy::Auto { min_work: 1 },
    ] {
        for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
            let output_dir = tempfile::tempdir().expect("isolated Radau AOT policy directory");
            let config = RadauConfig {
                t_bound: 0.2,
                first_step: Some(0.05),
                max_step: 0.05,
                rtol: 1.0e-8,
                atol: 1.0e-10,
                execution: RadauExecution::Aot,
                frontend,
                matrix_layout: RadauMatrixLayout::Dense,
                telemetry: RadauTelemetryMode::Counters,
                execution_policy,
                aot: Some(rebuild_aot_config(output_dir.path().to_path_buf())),
                ..RadauConfig::default()
            };
            let mut solver = RadauSolver::prepare(parameterized_decay_problem(), config)
                .unwrap_or_else(|error| panic!("AOT policy preparation failed: {error}"));
            let first = solver
                .solve_with_parameters(&[1.0, 1.0], &[2.0])
                .expect("AOT policy solve");
            let continued = solver
                .continue_with_parameters(&[1.0, 1.0], &[1.0])
                .expect("AOT policy continuation");
            assert!((first.y[0] - (-0.4_f64).exp()).abs() < 2.0e-6);
            assert!((continued.y[0] - (-0.2_f64).exp()).abs() < 2.0e-6);
            let counters = &continued.telemetry().counters;
            assert_eq!(counters["aot_build_attempts"], 1);
            assert_eq!(counters["aot_link_attempts"], 1);
            assert!(counters["parallel_dispatches"] + counters["sequential_dispatches"] > 0);
            reportln!(
                "[Radau AOT policy] frontend={frontend:?} policy={execution_policy:?} parallel={} sequential={} chunks={} worker_count={} worker_callbacks={} status=ok",
                counters["parallel_dispatches"],
                counters["sequential_dispatches"],
                counters["aot_chunks"],
                counters["worker_count"],
                counters["aot_worker_callbacks"],
            );
        }
    }
}

#[test]
#[ignore = "release AOT lifecycle; requires an available tcc toolchain"]
fn aot_build_require_prebuilt_rebuild_always_lifecycle_is_consistent() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_build_require_prebuilt_rebuild_always_lifecycle_is_consistent",
    );
    // This fixture has a unique symbolic key so process-local artifacts from
    // the frontend matrix cannot satisfy the mismatch cases below.
    let problem = invalidation_producer_problem();
    let producer_directory = tempfile::tempdir().expect("producer AOT directory");
    let producer_config = RadauConfig {
        t_bound: 0.2,
        first_step: Some(0.05),
        max_step: 0.05,
        execution: RadauExecution::Aot,
        frontend: RadauFrontend::ExprLegacy,
        matrix_layout: RadauMatrixLayout::Dense,
        telemetry: RadauTelemetryMode::Counters,
        aot: Some(aot_config(producer_directory.path().to_path_buf())),
        ..RadauConfig::default()
    };
    let producer = RadauSolver::prepare(problem.clone(), producer_config)
        .expect("BuildIfMissing producer preparation");
    let producer_resolver = producer
        .config()
        .aot
        .and_then(|aot| aot.generated.resolver)
        .expect("producer must publish resolver provenance");
    let handoff = producer_directory.path().join("radau-aot-handoff.txt");
    producer_resolver
        .write_handoff(&handoff)
        .expect("write AOT handoff");
    let consumer_resolver = AotResolver::read_handoff(&handoff).expect("read AOT handoff");

    // The producer and consumer share this test process, while the lifecycle
    // contract is process-isolated.  Remove the producer's process-local
    // registration so RequirePrebuilt must exercise the persisted handoff and
    // dynamic reconnect path instead of accidentally reusing the live entry.
    for problem_key in consumer_resolver.registry().problem_keys() {
        unregister_linked_dense_backend(&problem_key);
    }

    let consumer = RadauSolver::prepare(
        problem.clone(),
        RadauConfig {
            t_bound: 0.2,
            first_step: Some(0.05),
            max_step: 0.05,
            execution: RadauExecution::Aot,
            frontend: RadauFrontend::ExprLegacy,
            matrix_layout: RadauMatrixLayout::Dense,
            telemetry: RadauTelemetryMode::Counters,
            aot: Some(RadauAotConfig::require_prebuilt().with_resolver(Some(consumer_resolver))),
            ..RadauConfig::default()
        },
    )
    .expect("RequirePrebuilt consumer must reconnect published artifact");
    let mut consumer = consumer;
    let consumer_result = consumer
        .solve_with_parameters(&[1.0, 1.0], &[2.0])
        .expect("RequirePrebuilt consumer solve");
    let consumer_counters = &consumer_result.telemetry().counters;
    assert_eq!(consumer_counters["aot_build_attempts"], 0);
    // A process-local registry was cleared above, so the consumer performs
    // one dynamic link while still avoiding every compiler/build attempt.
    assert_eq!(consumer_counters["aot_link_attempts"], 1);
    assert!(consumer_counters["aot_reconnects"] >= 1);
    assert!((consumer_result.y[0] - (-0.4_f64).exp()).abs() < 2.0e-6);

    let rebuild_directory = tempfile::tempdir().expect("rebuild AOT directory");
    let mut rebuilt = RadauSolver::prepare(
        problem.clone(),
        RadauConfig {
            t_bound: 0.2,
            first_step: Some(0.05),
            max_step: 0.05,
            execution: RadauExecution::Aot,
            frontend: RadauFrontend::ExprLegacy,
            matrix_layout: RadauMatrixLayout::Dense,
            telemetry: RadauTelemetryMode::Counters,
            aot: Some(
                RadauAotConfig::rebuild_always_release(rebuild_directory.path().to_path_buf())
                    .with_c_compiler("tcc"),
            ),
            ..RadauConfig::default()
        },
    )
    .expect("RebuildAlways preparation");
    let rebuilt_result = rebuilt
        .solve_with_parameters(&[1.0, 1.0], &[2.0])
        .expect("RebuildAlways solve");
    let rebuilt_counters = &rebuilt_result.telemetry().counters;
    assert_eq!(rebuilt_counters["aot_build_attempts"], 1);
    assert_eq!(rebuilt_counters["aot_link_attempts"], 1);
    assert!((rebuilt_result.y[0] - (-0.4_f64).exp()).abs() < 2.0e-6);

    // Do not let the rebuilt producer's process-local registration make the
    // following empty-resolver negative gate pass accidentally.
    for problem_key in producer_resolver.registry().problem_keys() {
        unregister_linked_dense_backend(&problem_key);
    }

    let empty_resolver = AotResolver::new(AotRegistry::new());
    let missing = match RadauSolver::prepare(
        problem,
        RadauConfig {
            execution: RadauExecution::Aot,
            frontend: RadauFrontend::ExprLegacy,
            matrix_layout: RadauMatrixLayout::Dense,
            aot: Some(RadauAotConfig::require_prebuilt().with_resolver(Some(empty_resolver))),
            ..RadauConfig::default()
        },
    ) {
        Ok(_) => panic!("missing prebuilt artifact must fail"),
        Err(error) => error,
    };
    assert_eq!(missing.kind(), RadauErrorKind::Aot);
    reportln!(
        "[Radau AOT lifecycle] producer=build_if_missing consumer=require_prebuilt consumer_builds={} consumer_links={} reconnects={} rebuild_builds={} rebuild_links={} missing_error={:?} status=ok",
        consumer_counters["aot_build_attempts"],
        consumer_counters["aot_link_attempts"],
        consumer_counters["aot_reconnects"],
        rebuilt_counters["aot_build_attempts"],
        rebuilt_counters["aot_link_attempts"],
        missing.kind(),
    );
}

#[test]
#[ignore = "release AOT compiler failure classification; uses an intentionally missing compiler"]
fn aot_missing_compiler_is_classified_as_typed_lifecycle_error() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_missing_compiler_is_classified_as_typed_lifecycle_error",
    );
    let output_dir = tempfile::tempdir().expect("Radau missing compiler output directory");
    let error = match RadauSolver::prepare(
        // Keep this negative gate on a distinct schema so a linked artifact
        // from another AOT story cannot bypass the missing compiler.
        renamed_parameterized_decay_problem(),
        RadauConfig {
            t_bound: 0.2,
            first_step: Some(0.05),
            max_step: 0.05,
            execution: RadauExecution::Aot,
            frontend: RadauFrontend::ExprLegacy,
            matrix_layout: RadauMatrixLayout::Dense,
            telemetry: RadauTelemetryMode::Counters,
            aot: Some(
                aot_config(output_dir.path().to_path_buf())
                    .with_c_compiler("radau_missing_compiler_for_test"),
            ),
            ..RadauConfig::default()
        },
    ) {
        Ok(_) => panic!("missing compiler must not produce a prepared solver"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), RadauErrorKind::Aot);
    reportln!(
        "[Radau AOT failure] phase=build failure=missing_compiler error_kind={:?} status=ok",
        error.kind()
    );
}

#[test]
#[ignore = "release AOT invalidation; requires an available tcc toolchain"]
fn aot_require_prebuilt_rejects_schema_layout_and_frontend_mismatch() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_require_prebuilt_rejects_schema_layout_and_frontend_mismatch",
    );
    let output_dir = tempfile::tempdir().expect("Radau invalidation output directory");
    let producer = RadauSolver::prepare(
        parameterized_decay_problem(),
        RadauConfig {
            t_bound: 0.2,
            first_step: Some(0.05),
            max_step: 0.05,
            execution: RadauExecution::Aot,
            frontend: RadauFrontend::ExprLegacy,
            matrix_layout: RadauMatrixLayout::Dense,
            telemetry: RadauTelemetryMode::Counters,
            // The invalidation story needs a resolver published by this
            // producer, even if another story warmed the same key earlier.
            aot: Some(rebuild_aot_config(output_dir.path().to_path_buf())),
            ..RadauConfig::default()
        },
    )
    .expect("AOT invalidation producer preparation");
    let resolver = producer
        .config()
        .aot
        .as_ref()
        .and_then(|aot| aot.generated.resolver.clone())
        .expect("producer must publish resolver provenance");

    let cases = [
        (
            "parameter_schema",
            invalidation_schema_mismatch_problem(),
            RadauFrontend::ExprLegacy,
            RadauMatrixLayout::Dense,
        ),
        (
            "layout",
            invalidation_producer_problem(),
            RadauFrontend::ExprLegacy,
            RadauMatrixLayout::Sparse,
        ),
        (
            "frontend",
            invalidation_producer_problem(),
            RadauFrontend::AtomViewNative,
            RadauMatrixLayout::Dense,
        ),
    ];

    for (label, problem, frontend, matrix_layout) in cases {
        let error = match RadauSolver::prepare(
            problem,
            RadauConfig {
                t_bound: 0.2,
                first_step: Some(0.05),
                max_step: 0.05,
                execution: RadauExecution::Aot,
                frontend,
                matrix_layout,
                telemetry: RadauTelemetryMode::Counters,
                aot: Some(RadauAotConfig::require_prebuilt().with_resolver(Some(resolver.clone()))),
                ..RadauConfig::default()
            },
        ) {
            Ok(_) => panic!("RequirePrebuilt unexpectedly accepted {label} mismatch"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), RadauErrorKind::Aot);
        reportln!(
            "[Radau AOT invalidation] variant={label} error_kind={:?} status=ok",
            error.kind()
        );
    }
}

#[test]
#[ignore = "release AOT/Lambdify parity; requires an available tcc toolchain"]
fn aot_matches_lambdify_on_shared_workload_endpoints() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_matches_lambdify_on_shared_workload_endpoints",
    );
    let cases = [
        (WorkloadKind::StiffScalar, 1),
        (WorkloadKind::Robertson, 3),
        (WorkloadKind::CombustionLike, 3),
        (WorkloadKind::ThreeBody, 12),
        (WorkloadKind::DiffusionChain, 16),
    ];

    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        for (kind, dimension) in cases {
            let data = build_workload(kind, dimension);
            let variables: Vec<&str> = data.variables.iter().map(String::as_str).collect();
            let jacobian = match frontend {
                RadauFrontend::ExprLegacy => Some(
                    data.equations
                        .iter()
                        .flat_map(|equation| {
                            variables
                                .iter()
                                .map(move |variable| equation.diff(variable))
                        })
                        .collect(),
                ),
                RadauFrontend::AtomViewNative => None,
            };
            let problem = RadauProblem::new(data.equations, data.variables, data.time_variable)
                .with_parameters(data.parameter_names.clone());
            let problem = jacobian
                .map(|jacobian| problem.clone().with_jacobian(jacobian))
                .unwrap_or(problem);
            let mut lambdify = RadauSolver::prepare(
                problem,
                RadauConfig {
                    t_bound: workload_t_bound(kind),
                    first_step: Some(workload_t_bound(kind) * 0.1),
                    max_step: workload_t_bound(kind) * 0.25,
                    rtol: 1.0e-7,
                    atol: 1.0e-10,
                    execution: RadauExecution::Lambdify,
                    frontend,
                    matrix_layout: RadauMatrixLayout::Dense,
                    telemetry: RadauTelemetryMode::Counters,
                    ..RadauConfig::default()
                },
            )
            .unwrap_or_else(|error| panic!("Lambdify preparation failed: {error}"));

            let output_dir = tempfile::tempdir().expect("isolated AOT parity directory");
            let (aot_problem, initial_state, parameters) = workload_problem(kind, dimension);
            let mut aot = RadauSolver::prepare(
                aot_problem,
                RadauConfig {
                    t_bound: workload_t_bound(kind),
                    first_step: Some(workload_t_bound(kind) * 0.1),
                    max_step: workload_t_bound(kind) * 0.25,
                    rtol: 1.0e-7,
                    atol: 1.0e-10,
                    execution: RadauExecution::Aot,
                    frontend,
                    matrix_layout: RadauMatrixLayout::Dense,
                    telemetry: RadauTelemetryMode::Counters,
                    execution_policy: RadauExecutionPolicy::Sequential,
                    aot: Some(aot_config(output_dir.path().to_path_buf())),
                    ..RadauConfig::default()
                },
            )
            .unwrap_or_else(|error| panic!("AOT preparation failed: {error}"));

            let lambdify_result = if parameters.is_empty() {
                lambdify.solve(&initial_state).expect("Lambdify solve")
            } else {
                lambdify
                    .solve_with_parameters(&initial_state, &parameters)
                    .expect("Lambdify solve")
            };
            let aot_result = if parameters.is_empty() {
                aot.solve(&initial_state).expect("AOT solve")
            } else {
                aot.solve_with_parameters(&initial_state, &parameters)
                    .expect("AOT solve")
            };
            let max_diff = lambdify_result
                .y
                .iter()
                .zip(aot_result.y.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            reportln!(
                "[Radau AOT/Lambdify parity] frontend={frontend:?} workload={kind:?} dimension={dimension} max_final_diff={max_diff:.3e}"
            );
            assert!(
                max_diff < 5.0e-6,
                "AOT/Lambdify endpoint drift for {frontend:?}/{kind:?}: {max_diff:e}"
            );
        }
    }
}

#[test]
#[ignore = "release AOT output parity; requires an available tcc toolchain"]
fn aot_dense_output_samples_match_lambdify_for_both_frontends() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_dense_output_samples_match_lambdify_for_both_frontends",
    );
    let sample_times = [0.0, 0.05, 0.1, 0.15, 0.2];
    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        let problem = parameterized_decay_problem();
        let problem = problem.with_jacobian(parameterized_decay_jacobian());
        let base = RadauConfig {
            t_bound: 0.2,
            first_step: Some(0.05),
            max_step: 0.05,
            rtol: 1.0e-8,
            atol: 1.0e-10,
            frontend,
            matrix_layout: RadauMatrixLayout::Dense,
            output: RadauOutputPolicy::Dense,
            telemetry: RadauTelemetryMode::Counters,
            ..RadauConfig::default()
        };
        let mut lambdify = RadauSolver::prepare(
            problem.clone(),
            RadauConfig {
                execution: RadauExecution::Lambdify,
                ..base.clone()
            },
        )
        .expect("Lambdify output preparation");
        let output_directory = tempfile::tempdir().expect("AOT output parity directory");
        let mut aot = RadauSolver::prepare(
            problem,
            RadauConfig {
                execution: RadauExecution::Aot,
                aot: Some(aot_config(output_directory.path().to_path_buf())),
                ..base
            },
        )
        .expect("AOT output preparation");
        let lambdify_solution = lambdify
            .solve_with_parameters(&[1.0, 1.0], &[2.0])
            .expect("Lambdify output solve");
        let aot_solution = aot
            .solve_with_parameters(&[1.0, 1.0], &[2.0])
            .expect("AOT output solve");
        let lambdify_samples = lambdify_solution
            .sample_many(&sample_times)
            .expect("Lambdify dense output samples");
        let aot_samples = aot_solution
            .sample_many(&sample_times)
            .expect("AOT dense output samples");
        let max_diff = lambdify_samples
            .iter()
            .zip(aot_samples.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        reportln!(
            "[Radau AOT output parity] frontend={frontend:?} samples={} max_sample_diff={max_diff:.3e}",
            sample_times.len()
        );
        assert!(max_diff < 5.0e-6, "dense output drift: {max_diff:e}");
    }
}

#[test]
#[ignore = "release AOT stage matrix; requires an available tcc toolchain"]
fn aot_stage_breakdown_workload_matrix_reports_cold_warm_and_continuation() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT",
        "numerical::Radau::tests::aot::aot_stage_breakdown_workload_matrix_reports_cold_warm_and_continuation",
    );
    let cases = [
        (WorkloadKind::StiffScalar, 1, RadauMatrixLayout::Dense),
        (WorkloadKind::Robertson, 3, RadauMatrixLayout::Dense),
        (WorkloadKind::CombustionLike, 3, RadauMatrixLayout::Dense),
        (WorkloadKind::ThreeBody, 12, RadauMatrixLayout::Dense),
        (WorkloadKind::DiffusionChain, 16, RadauMatrixLayout::Sparse),
        (
            WorkloadKind::DiffusionChain,
            16,
            RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        ),
    ];

    let mut rows = Vec::new();
    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        for (workload_kind, dimension, matrix_layout) in cases {
            let output_dir = tempfile::tempdir().expect("isolated Radau AOT stage directory");
            let (problem, initial_state, parameters) = workload_problem(workload_kind, dimension);
            let config = RadauConfig {
                t_bound: 0.002,
                first_step: Some(0.0005),
                max_step: 0.001,
                rtol: 1.0e-7,
                atol: 1.0e-10,
                execution: RadauExecution::Aot,
                frontend,
                matrix_layout,
                telemetry: RadauTelemetryMode::Timings,
                execution_policy: RadauExecutionPolicy::Sequential,
                aot: Some(rebuild_aot_config(output_dir.path().to_path_buf())),
                ..RadauConfig::default()
            };
            let mut solver = RadauSolver::prepare(problem, config)
                .unwrap_or_else(|error| panic!("AOT stage preparation failed: {error}"));
            let first = if parameters.is_empty() {
                solver.solve(&initial_state).expect("AOT stage solve")
            } else {
                solver
                    .solve_with_parameters(&initial_state, &parameters)
                    .expect("AOT stage solve")
            };
            let updated_parameters = if parameters.is_empty() {
                Vec::new()
            } else {
                parameter_continuation_target(&nalgebra::DVector::from_vec(parameters.clone()), 1)
                    .as_slice()
                    .to_vec()
            };
            let continued = if updated_parameters.is_empty() {
                solver.solve(&initial_state).expect("AOT repeated solve")
            } else {
                solver
                    .continue_with_parameters(&initial_state, &updated_parameters)
                    .expect("AOT continuation")
            };
            let first_timings = &first.telemetry().timings_ms;
            let continued_timings = &continued.telemetry().timings_ms;
            let counters = &continued.telemetry().counters;
            rows.push(AotStageRow {
                frontend: format!("{frontend:?}"),
                workload: format!("{workload_kind:?}"),
                layout: format!("{matrix_layout:?}"),
                prepare_ms: format!("{:.3}", timing(first_timings, "preparation_ms")),
                cache_ms: format!("{:.3}", timing(first_timings, "aot_cache_lookup_ms")),
                lowering_ms: format!("{:.3}", timing(first_timings, "aot_lowering_ms")),
                source_ms: format!("{:.3}", timing(first_timings, "aot_source_generation_ms")),
                materialize_ms: format!("{:.3}", timing(first_timings, "aot_materialize_ms")),
                build_ms: format!("{:.3}", timing(first_timings, "aot_build_ms")),
                link_ms: format!("{:.3}", timing(first_timings, "aot_link_ms")),
                publication_ms: format!("{:.3}", timing(first_timings, "aot_publication_ms")),
                cold_callback_ms: format!("{:.3}", timing(first_timings, "callback_ms")),
                warm_callback_ms: format!("{:.3}", timing(continued_timings, "callback_ms")),
                builds: counters["aot_build_attempts"],
                links: counters["aot_link_attempts"],
                chunks: counters["aot_chunks"],
                workers: counters["worker_count"],
            });
            assert_eq!(counters["aot_build_attempts"], 1);
            assert_eq!(counters["aot_link_attempts"], 1);
            assert!(first.y.iter().all(|value| value.is_finite()));
            assert!(continued.y.iter().all(|value| value.is_finite()));
        }
    }
    crate::Utils::test_reporting::capture_test_table("[Radau AOT stage matrix]", &rows);
}

fn timing(timings: &std::collections::BTreeMap<&'static str, f64>, key: &'static str) -> f64 {
    timings.get(key).copied().unwrap_or_default()
}
