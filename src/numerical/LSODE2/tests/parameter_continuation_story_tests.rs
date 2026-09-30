//! Parameter-continuation correctness and performance gates for LSODE2.
//!
//! These stories distinguish numeric parameter rebinding from rebuilding a
//! symbolic problem. A continuation run must reuse the prepared callback
//! shape while producing the same trajectory as a fresh solver.

use super::workload_fixtures::parameter_continuation_target;
use super::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, Lsode2AotProfile, Lsode2AotToolchain,
    Lsode2ControllerConfig, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::try_linked_aot_runtime_registry_snapshot;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, get_current_pid};
use tempfile::tempdir;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn parameterized_config(
    assembly: Lsode2SymbolicAssemblyBackend,
    banded: bool,
    telemetry: IvpTelemetry,
    parameters: [f64; 2],
) -> Lsode2ProblemConfig {
    let mut config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-a*y1 + b*y2"),
            Expr::parse_expression("a*y1 - 2*b*y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.5]),
        0.5,
        0.025,
        1.0e-9,
        1.0e-11,
    )
    .with_equation_parameters(vec!["a".to_string(), "b".to_string()])
    .with_equation_parameter_values(DVector::from_vec(parameters.to_vec()))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_faithful_bdf_solve(4_000, 4_000)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_telemetry(telemetry);

    config = if banded {
        config.with_native_banded_faithful_backend()
    } else {
        config.with_native_sparse_faer_backend()
    };
    config
}

fn max_matrix_diff(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
    assert_eq!(left.shape(), right.shape());
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn max_vector_diff(left: &DVector<f64>, right: &DVector<f64>) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn cold_calls(snapshot: &super::IvpTelemetrySnapshot, stage: IvpColdStage) -> u64 {
    snapshot.cold_stage(stage).calls
}

fn cold_stage_delta_ms(
    before: &super::IvpTelemetrySnapshot,
    after: &super::IvpTelemetrySnapshot,
    stage: IvpColdStage,
) -> f64 {
    after
        .cold_stage(stage)
        .elapsed
        .saturating_sub(before.cold_stage(stage).elapsed)
        .as_secs_f64()
        * 1.0e3
}

fn warm_stage_delta_ms(
    before: &super::IvpTelemetrySnapshot,
    after: &super::IvpTelemetrySnapshot,
    stage: super::IvpWarmStage,
) -> f64 {
    after
        .warm_stage(stage)
        .elapsed
        .saturating_sub(before.warm_stage(stage).elapsed)
        .as_secs_f64()
        * 1.0e3
}

fn diffusion_diagnostic_target(base: &DVector<f64>, target_index: usize) -> DVector<f64> {
    parameter_continuation_target(base, target_index)
}

fn large_diffusion_diagnostic_config(
    dimension: usize,
    assembly: Lsode2SymbolicAssemblyBackend,
    aot: bool,
    banded: bool,
    telemetry: IvpTelemetry,
    output_parent: &std::path::Path,
) -> Lsode2ProblemConfig {
    let workload = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(dimension);
    let mut config = Lsode2ProblemConfig::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.25,
        0.02,
        1.0e-7,
        1.0e-9,
    )
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_equation_parameters(workload.parameter_names)
    .with_equation_parameter_values(workload.parameter_values)
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: if aot {
            Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            }
        } else {
            Lsode2SymbolicExecutionMode::LambdifyExpr
        },
    })
    .with_telemetry(telemetry);

    config = if banded {
        config.with_native_banded_faithful_backend()
    } else {
        config.with_native_sparse_faer_backend()
    };

    if aot {
        let generated = SymbolicIvpGeneratedBackendConfig::new()
            .with_output_parent_dir(Some(output_parent.to_path_buf()))
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
            .with_c_tcc();
        config = if banded {
            config.with_native_banded_faithful_generated_backend(generated)
        } else {
            config.with_native_sparse_faer_generated_backend(generated)
        };
    }
    config
}

#[test]
#[ignore = "release diagnostic: isolate n=512 Sparse ExprLegacy/AtomView continuation scaling"]
fn lsode2_parameter_continuation_diffusion_sparse_n512_per_target_diagnostic() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_diffusion_sparse_n512_per_target_diagnostic",
    );
    const DIMENSION: usize = 512;
    const MAX_TARGET: usize = 64;
    let base_parameters = DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]);
    reportln!(
        "[LSODE2 continuation anomaly] diffusion Sparse n={DIMENSION}; targets=1..={MAX_TARGET}; prepared solver reused; native engine fresh per solve"
    );
    reportln!(
        "frontend | execution | target | solve_ms | dependency_ms | differentiation_ms | evaluator_plan_ms | residual_ms | jacobian_ms | factor_ms | rhs_ms | allocated_bytes | errors | residual_aux_calls | residual_prep_calls | jacobian_aux_calls | residual_calls | jacobian_calls | linear_solves | accepted | rejected | iterations | final_t | termination | finite"
    );

    for (frontend, assembly, aot) in [
        (
            "ExprLegacy",
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            false,
        ),
        (
            "ExprLegacy",
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            true,
        ),
        (
            "AtomViewNative",
            Lsode2SymbolicAssemblyBackend::AtomView,
            false,
        ),
        (
            "AtomViewNative",
            Lsode2SymbolicAssemblyBackend::AtomView,
            true,
        ),
    ] {
        let output_dir = tempdir().expect("anomaly diagnostic output directory should exist");
        let telemetry = IvpTelemetry::detailed();
        let config = large_diffusion_diagnostic_config(
            DIMENSION,
            assembly,
            aot,
            false,
            telemetry.clone(),
            output_dir.path(),
        );
        let mut solver =
            Lsode2Solver::new(config).expect("anomaly diagnostic solver should construct");
        solver
            .prepare()
            .expect("anomaly diagnostic solver preparation should succeed");
        let mut previous = solver.native_statistics();

        for target_index in 1..=MAX_TARGET {
            let before_telemetry = telemetry.snapshot();
            solver
                .set_parameter_values(diffusion_diagnostic_target(&base_parameters, target_index))
                .expect("anomaly diagnostic parameter rebind should succeed");
            let started = Instant::now();
            let summary = solver
                .solve_with_summary()
                .expect("anomaly diagnostic solve should succeed");
            let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
            let current = solver.native_statistics();
            let after_telemetry = telemetry.snapshot();
            let native = summary
                .native_integration_solve
                .as_ref()
                .expect("anomaly diagnostic must use native integration");
            let finite = summary
                .final_y
                .as_ref()
                .is_some_and(|state| state.iter().all(|value| value.is_finite()));
            reportln!(
                "{frontend} | {} | {target_index} | {elapsed_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.6e} | {} | {finite}",
                if aot { "AOT" } else { "Lambdify" },
                cold_stage_delta_ms(
                    &before_telemetry,
                    &after_telemetry,
                    IvpColdStage::AtomDependencyAnalysis,
                ),
                cold_stage_delta_ms(
                    &before_telemetry,
                    &after_telemetry,
                    IvpColdStage::SymbolicDifferentiation,
                ),
                cold_stage_delta_ms(
                    &before_telemetry,
                    &after_telemetry,
                    IvpColdStage::NativeJacobianEvaluatorPreparation,
                ),
                warm_stage_delta_ms(
                    &before_telemetry,
                    &after_telemetry,
                    super::IvpWarmStage::ResidualCallback,
                ),
                warm_stage_delta_ms(
                    &before_telemetry,
                    &after_telemetry,
                    super::IvpWarmStage::JacobianCallback,
                ),
                warm_stage_delta_ms(
                    &before_telemetry,
                    &after_telemetry,
                    super::IvpWarmStage::Factorization,
                ),
                warm_stage_delta_ms(
                    &before_telemetry,
                    &after_telemetry,
                    super::IvpWarmStage::RhsSolve,
                ),
                after_telemetry
                    .allocated_bytes
                    .saturating_sub(before_telemetry.allocated_bytes),
                after_telemetry
                    .errors
                    .saturating_sub(before_telemetry.errors),
                after_telemetry
                    .residual_auxiliary_evaluations
                    .saturating_sub(before_telemetry.residual_auxiliary_evaluations),
                after_telemetry
                    .residual_preparation_evaluations
                    .saturating_sub(before_telemetry.residual_preparation_evaluations),
                after_telemetry
                    .jacobian_auxiliary_evaluations
                    .saturating_sub(before_telemetry.jacobian_auxiliary_evaluations),
                current
                    .native_residual_calls
                    .saturating_sub(previous.native_residual_calls),
                current
                    .native_jacobian_calls
                    .saturating_sub(previous.native_jacobian_calls),
                current
                    .native_linear_solve_calls
                    .saturating_sub(previous.native_linear_solve_calls),
                native.accepted_steps,
                native.rejected_steps,
                native.total_iterations,
                native.final_t,
                native.termination_kind.label(),
            );
            assert!(
                finite,
                "diagnostic target {target_index} produced a non-finite state"
            );
            previous = current;
        }
    }
}

#[test]
#[ignore = "focused diagnostic: reproduce the exact Criterion n=512 Sparse AOT target sequence"]
fn lsode2_parameter_continuation_sparse_n512_criterion_series_probe() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_sparse_n512_criterion_series_probe",
    );
    const DIMENSION: usize = 512;
    const TARGETS: usize = 64;
    let workload = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(DIMENSION);
    let output_dir = tempdir().expect("focused continuation probe directory should exist");
    let mut solver = Lsode2Solver::new(large_diffusion_diagnostic_config(
        DIMENSION,
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        true,
        false,
        IvpTelemetry::disabled(),
        output_dir.path(),
    ))
    .expect("focused continuation probe should construct");
    solver
        .prepare()
        .expect("focused continuation probe should prepare");

    reportln!(
        "[LSODE2 continuation exact-series probe] workload=diffusion-chain; matrix=Sparse; frontend=ExprLegacy-AOT; dimension={DIMENSION}; targets={TARGETS}; telemetry=off"
    );
    reportln!(
        "target | k | d | q | nl | solve_ms | residual_calls | jacobian_calls | linear_solves | accepted | rejected | finite"
    );
    let checkpoints = [1, 4, 16, 32, 48, 64];
    for target_index in 1..=TARGETS {
        let parameters = parameter_continuation_target(&workload.parameter_values, target_index);
        let before = solver.native_statistics();
        println!("[continuation probe] target={target_index} start");
        solver
            .set_parameter_values(parameters.clone())
            .expect("exact continuation target should rebind");
        let started = Instant::now();
        let summary = solver
            .solve_with_summary()
            .expect("exact continuation target should solve");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
        let finite = summary
            .final_y
            .as_ref()
            .is_some_and(|state| state.iter().all(|value| value.is_finite()));
        assert!(
            summary
                .native_integration_solve
                .as_ref()
                .is_some_and(|native| native.reached_t_bound || native.reached_stop_condition),
            "exact continuation target {target_index} must complete integration"
        );
        let stats = solver.native_statistics();
        println!(
            "[continuation probe] target={target_index} done solve_ms={elapsed_ms:.3} residual_delta={} jacobian_delta={} linear_delta={}",
            stats
                .native_residual_calls
                .saturating_sub(before.native_residual_calls),
            stats
                .native_jacobian_calls
                .saturating_sub(before.native_jacobian_calls),
            stats
                .native_linear_solve_calls
                .saturating_sub(before.native_linear_solve_calls),
        );
        if checkpoints.contains(&target_index) {
            let native = summary
                .native_integration_solve
                .as_ref()
                .expect("probe must use native integration");
            reportln!(
                "{target_index} | {:.8} | {:.8} | {:.8} | {:.8} | {elapsed_ms:.3} | {} | {} | {} | {} | {} | {finite}",
                parameters[0],
                parameters[1],
                parameters[2],
                parameters[3],
                stats
                    .native_residual_calls
                    .saturating_sub(before.native_residual_calls),
                stats
                    .native_jacobian_calls
                    .saturating_sub(before.native_jacobian_calls),
                stats
                    .native_linear_solve_calls
                    .saturating_sub(before.native_linear_solve_calls),
                native.accepted_steps,
                native.rejected_steps,
            );
        }
        assert!(finite, "target {target_index} must remain finite");
    }
}

#[test]
#[ignore = "bounded diagnostic: isolate the slow n=512 Sparse continuation target 41"]
fn lsode2_parameter_continuation_sparse_n512_target_41_budget_probe() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_sparse_n512_target_41_budget_probe",
    );
    const DIMENSION: usize = 512;
    const TARGET: usize = 41;
    const STEP_BUDGET: usize = 2_000;
    let workload = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(DIMENSION);
    let parameters = parameter_continuation_target(&workload.parameter_values, TARGET);
    let output_dir = tempdir().expect("target-41 probe directory should exist");
    let config = large_diffusion_diagnostic_config(
        DIMENSION,
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        true,
        false,
        IvpTelemetry::detailed(),
        output_dir.path(),
    )
    .with_faithful_bdf_solve(STEP_BUDGET, STEP_BUDGET);
    let mut solver = Lsode2Solver::new(config).expect("target-41 probe should construct");
    solver.prepare().expect("target-41 probe should prepare");
    solver
        .set_parameter_values(parameters.clone())
        .expect("target-41 parameters should bind");

    println!(
        "[continuation target-41 probe] k={:.8} d={:.8} q={:.8} nl={:.8} step_budget={STEP_BUDGET} start",
        parameters[0], parameters[1], parameters[2], parameters[3]
    );
    let started = Instant::now();
    let result = solver.solve_with_summary();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let stats = solver.native_statistics();
    let status = result
        .as_ref()
        .map(|summary| {
            summary
                .native_integration_solve
                .as_ref()
                .map(|native| {
                    format!(
                        "ok:{}:{}:{}:t={:.17}:reached={}",
                        native.termination_kind.label(),
                        native.accepted_steps,
                        native.rejected_steps,
                        native.final_t,
                        native.reached_t_bound,
                    )
                })
                .unwrap_or_else(|| "ok:no-native-summary".to_string())
        })
        .unwrap_or_else(|error| format!("error:{error}"));
    reportln!(
        "[LSODE2 continuation target-41 bounded probe] dimension={DIMENSION}; matrix=Sparse; frontend=ExprLegacy-AOT; result={status}; solve_ms={elapsed_ms:.3}; residual_calls={}; jacobian_calls={}; linear_solves={}",
        stats.native_residual_calls,
        stats.native_jacobian_calls,
        stats.native_linear_solve_calls,
    );
    println!(
        "[continuation target-41 probe] done status={status} solve_ms={elapsed_ms:.3} residual_calls={} jacobian_calls={} linear_solves={}",
        stats.native_residual_calls, stats.native_jacobian_calls, stats.native_linear_solve_calls,
    );
    let summary = result.expect("target 41 continuation solve should succeed");
    assert!(
        summary
            .native_integration_solve
            .as_ref()
            .is_some_and(|native| native.reached_t_bound),
        "target 41 must reach t_bound, not merely return a finite partial state"
    );
    assert!(
        summary
            .final_y
            .as_ref()
            .is_some_and(|state| { state.iter().all(|value| value.is_finite()) })
    );
}

#[test]
#[ignore = "bounded release gate: verify the fixed continuation endpoint across symbolic and matrix routes"]
fn lsode2_parameter_continuation_n512_target_41_route_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_n512_target_41_route_matrix",
    );
    const TARGET: usize = 41;
    const STEP_BUDGET: usize = 2_000;

    reportln!(
        "[LSODE2 continuation endpoint route matrix] workload=diffusion-chain; cases=(n=512 Sparse/Banded,n=1024 Banded); target={TARGET}; step_budget={STEP_BUDGET}; telemetry=off"
    );
    reportln!(
        "dimension | matrix | frontend | execution | prepare_ms | solve_ms | final_t | reached_t_bound | accepted | rejected | residual_calls | jacobian_calls | linear_solves | finite"
    );

    for (dimension, matrix_name, banded) in [
        (512, "Sparse", false),
        (512, "Banded", true),
        (1024, "Banded", true),
    ] {
        let workload = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(dimension);
        let parameters = parameter_continuation_target(&workload.parameter_values, TARGET);
        for (frontend, assembly) in [
            ("ExprLegacy", Lsode2SymbolicAssemblyBackend::ExprLegacy),
            ("AtomViewNative", Lsode2SymbolicAssemblyBackend::AtomView),
        ] {
            for (execution, aot) in [("Lambdify", false), ("AOT", true)] {
                let output_dir = tempdir().expect("route-matrix output directory should exist");
                let config = large_diffusion_diagnostic_config(
                    dimension,
                    assembly,
                    aot,
                    banded,
                    IvpTelemetry::disabled(),
                    output_dir.path(),
                )
                .with_faithful_bdf_solve(STEP_BUDGET, STEP_BUDGET);
                let mut solver = Lsode2Solver::new(config)
                    .expect("continuation route-matrix solver should construct");
                let prepare_started = Instant::now();
                solver
                    .prepare()
                    .expect("continuation route-matrix solver should prepare");
                let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1.0e3;
                solver
                    .set_parameter_values(parameters.clone())
                    .expect("route-matrix target parameters should bind");

                let solve_started = Instant::now();
                let summary = solver
                    .solve_with_summary()
                    .expect("route-matrix target solve should succeed");
                let solve_ms = solve_started.elapsed().as_secs_f64() * 1.0e3;
                let native = summary
                    .native_integration_solve
                    .as_ref()
                    .expect("route-matrix solve should use native integration");
                let finite = summary
                    .final_y
                    .as_ref()
                    .is_some_and(|state| state.iter().all(|value| value.is_finite()));
                let stats = solver.native_statistics();

                reportln!(
                    "{dimension} | {matrix_name} | {frontend} | {execution} | {prepare_ms:.3} | {solve_ms:.3} | {:.17} | {} | {} | {} | {} | {} | {} | {finite}",
                    native.final_t,
                    native.reached_t_bound,
                    native.accepted_steps,
                    native.rejected_steps,
                    stats.native_residual_calls,
                    stats.native_jacobian_calls,
                    stats.native_linear_solve_calls,
                );
                assert!(
                    native.reached_t_bound,
                    "n={dimension} {matrix_name}/{frontend}/{execution} target {TARGET} must reach t_bound; final_t={:.17}, termination={}",
                    native.final_t,
                    native.termination_kind.label(),
                );
                assert!(
                    finite,
                    "n={dimension} {matrix_name}/{frontend}/{execution} must stay finite"
                );
            }
        }
    }
}

fn current_process_rss_bytes() -> Option<u64> {
    let pid = get_current_pid().ok()?;
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    system.process(pid).map(|process| process.memory())
}

struct ProcessPeakRssSampler {
    stop: Arc<AtomicBool>,
    peak_bytes: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

impl ProcessPeakRssSampler {
    fn start() -> Option<Self> {
        let pid = get_current_pid().ok()?;
        let stop = Arc::new(AtomicBool::new(false));
        let peak_bytes = Arc::new(AtomicU64::new(current_process_rss_bytes()?));
        let worker_stop = Arc::clone(&stop);
        let worker_peak = Arc::clone(&peak_bytes);
        let worker = thread::Builder::new()
            .name("lsode2-rss-sampler".to_string())
            .spawn(move || {
                let mut system = System::new();
                while !worker_stop.load(Ordering::Relaxed) {
                    system.refresh_processes_specifics(
                        ProcessesToUpdate::Some(&[pid]),
                        true,
                        ProcessRefreshKind::nothing().with_memory(),
                    );
                    if let Some(process) = system.process(pid) {
                        worker_peak.fetch_max(process.memory(), Ordering::Relaxed);
                    }
                    thread::sleep(std::time::Duration::from_millis(10));
                }
            })
            .ok()?;

        Some(Self {
            stop,
            peak_bytes,
            worker: Some(worker),
        })
    }

    fn stop(mut self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take()?.join().ok()?;
        current_process_rss_bytes()
            .map(|rss| self.peak_bytes.fetch_max(rss, Ordering::Relaxed).max(rss))
    }
}

impl Drop for ProcessPeakRssSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[test]
#[ignore = "bounded release diagnostic: track RSS and AOT lifecycle counters over repeated n=1024 Banded continuation series"]
fn lsode2_parameter_continuation_n1024_banded_resource_growth() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_n1024_banded_resource_growth",
    );
    const DIMENSION: usize = 1024;
    const TARGETS: usize = 16;
    const PASSES: usize = 4;
    let workload = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(DIMENSION);
    let output_dir = tempdir().expect("resource diagnostic output directory should exist");
    let telemetry = IvpTelemetry::detailed();
    let registry_before_prepare = try_linked_aot_runtime_registry_snapshot()
        .expect("linked runtime registry should be readable before preparation");
    let mut solver = Lsode2Solver::new(large_diffusion_diagnostic_config(
        DIMENSION,
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        true,
        true,
        telemetry.clone(),
        output_dir.path(),
    ))
    .expect("Banded resource diagnostic solver should construct");
    let rss_before_prepare = current_process_rss_bytes();
    solver
        .prepare()
        .expect("Banded resource diagnostic solver should prepare");
    let after_prepare = telemetry.snapshot();
    let registry_after_prepare = try_linked_aot_runtime_registry_snapshot()
        .expect("linked runtime registry should be readable after preparation");
    let rss_after_prepare = current_process_rss_bytes();
    let peer_telemetry = IvpTelemetry::detailed();
    let mut peer_solver = Lsode2Solver::new(large_diffusion_diagnostic_config(
        DIMENSION,
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        true,
        true,
        peer_telemetry.clone(),
        output_dir.path(),
    ))
    .expect("peer resource diagnostic solver should construct");
    peer_solver
        .prepare()
        .expect("peer resource diagnostic solver should prepare");
    let peer_after_prepare = peer_telemetry.snapshot();
    let registry_after_peer_prepare = try_linked_aot_runtime_registry_snapshot()
        .expect("linked runtime registry should be readable after peer preparation");
    let rss_with_two_solvers = current_process_rss_bytes();
    let peak_rss_sampler = ProcessPeakRssSampler::start()
        .expect("process RSS sampler should start for this release diagnostic");

    reportln!(
        "[LSODE2 continuation resource growth] workload=diffusion-chain; dimension={DIMENSION}; matrix=Banded; route=ExprLegacy-AOT; targets=1..={TARGETS}; passes={PASSES}; two prepared solvers share an output parent; RSS sampled outside solve; telemetry=detailed"
    );
    reportln!(
        "phase | pass | target | target_solve_ms | process_rss_bytes | rss_delta_from_prepared | target_allocated_bytes | artifact_key_count | runtime_ready | build_attempts | link_attempts | target_residual_calls | target_jacobian_calls | target_linear_solves | completed_targets"
    );
    reportln!(
        "prepared | 0 | - | 0.000 | {:?} | 0 | 0 | {} | {} | {} | {} | 0 | 0 | 0 | 0",
        rss_after_prepare,
        after_prepare.aot_artifact_keys.len(),
        after_prepare.aot_runtime_ready,
        after_prepare.aot_build_attempts,
        after_prepare.aot_link_attempts,
    );
    reportln!(
        "[LSODE2 linked AOT registry] total_entries_before={} after_primary={} after_peer={}; primary_keys_registered={}/{}; peer_keys_registered={}/{}",
        registry_before_prepare.total_entries(),
        registry_after_prepare.total_entries(),
        registry_after_peer_prepare.total_entries(),
        after_prepare
            .aot_artifact_keys
            .iter()
            .filter(|key| registry_after_prepare.contains_problem_key(key))
            .count(),
        after_prepare.aot_artifact_keys.len(),
        peer_after_prepare
            .aot_artifact_keys
            .iter()
            .filter(|key| registry_after_peer_prepare.contains_problem_key(key))
            .count(),
        peer_after_prepare.aot_artifact_keys.len(),
    );
    assert!(
        after_prepare
            .aot_artifact_keys
            .iter()
            .all(|key| { registry_after_prepare.contains_problem_key(key) }),
        "every prepared AOT artifact key should have a linked process-local callback"
    );
    reportln!(
        "peer_prepared | 0 | - | 0.000 | {rss_with_two_solvers:?} | {:?} | 0 | {} | {} | {} | {} | 0 | 0 | 0 | 0",
        rss_with_two_solvers
            .zip(rss_after_prepare)
            .map(|(current, prepared)| current as i128 - prepared as i128),
        peer_after_prepare.aot_artifact_keys.len(),
        peer_after_prepare.aot_runtime_ready,
        peer_after_prepare.aot_build_attempts,
        peer_after_prepare.aot_link_attempts,
    );
    assert!(peer_after_prepare.aot_runtime_ready > 0);
    assert!(!peer_after_prepare.aot_artifact_keys.is_empty());
    assert!(
        peer_after_prepare
            .aot_artifact_keys
            .iter()
            .all(|peer_key| after_prepare.aot_artifact_keys.contains(peer_key)),
        "peer solver must resolve artifacts published by the primary solver"
    );

    for pass in 1..=PASSES {
        let before = solver.native_statistics();
        let telemetry_before = telemetry.snapshot();
        let mut target_solve_total_ms = 0.0;
        for target_index in 1..=TARGETS {
            let target_stats_before = solver.native_statistics();
            let target_telemetry_before = telemetry.snapshot();
            let target = parameter_continuation_target(&workload.parameter_values, target_index);
            let target_started = Instant::now();
            solver
                .set_parameter_values(target)
                .expect("resource diagnostic parameter bind should succeed");
            let summary = solver
                .solve_with_summary()
                .expect("resource diagnostic solve should succeed");
            assert!(
                summary
                    .native_integration_solve
                    .as_ref()
                    .is_some_and(|native| native.reached_t_bound || native.reached_stop_condition),
                "resource diagnostic target {target_index} did not complete integration"
            );
            assert!(
                summary
                    .final_y
                    .as_ref()
                    .is_some_and(|state| state.iter().all(|value| value.is_finite()))
            );
            let target_solve_ms = target_started.elapsed().as_secs_f64() * 1.0e3;
            target_solve_total_ms += target_solve_ms;
            let target_stats_after = solver.native_statistics();
            let target_telemetry_after = telemetry.snapshot();
            let target_rss = current_process_rss_bytes();
            let target_rss_delta = target_rss
                .zip(rss_after_prepare)
                .map(|(current, prepared)| current as i128 - prepared as i128);
            reportln!(
                "target | {pass} | {target_index} | {target_solve_ms:.3} | {target_rss:?} | {target_rss_delta:?} | {} | {} | {} | {} | {} | {} | {} | {} | -",
                target_telemetry_after
                    .allocated_bytes
                    .saturating_sub(target_telemetry_before.allocated_bytes),
                target_telemetry_after.aot_artifact_keys.len(),
                target_telemetry_after.aot_runtime_ready,
                target_telemetry_after.aot_build_attempts,
                target_telemetry_after.aot_link_attempts,
                target_stats_after
                    .native_residual_calls
                    .saturating_sub(target_stats_before.native_residual_calls),
                target_stats_after
                    .native_jacobian_calls
                    .saturating_sub(target_stats_before.native_jacobian_calls),
                target_stats_after
                    .native_linear_solve_calls
                    .saturating_sub(target_stats_before.native_linear_solve_calls),
            );
        }
        let after = solver.native_statistics();
        let telemetry_after = telemetry.snapshot();
        let rss = current_process_rss_bytes();
        assert_eq!(
            telemetry_after.aot_artifact_keys, after_prepare.aot_artifact_keys,
            "numeric continuation must keep the prepared artifact-key set stable"
        );
        assert_eq!(
            telemetry_after.aot_build_attempts, after_prepare.aot_build_attempts,
            "numeric continuation must not start additional builds"
        );
        assert_eq!(
            telemetry_after.aot_link_attempts, after_prepare.aot_link_attempts,
            "numeric continuation must not start additional links"
        );
        let registry_after_pass = try_linked_aot_runtime_registry_snapshot()
            .expect("linked runtime registry should be readable after continuation pass");
        let pass_keys_registered = after_prepare
            .aot_artifact_keys
            .iter()
            .filter(|key| registry_after_pass.contains_problem_key(key))
            .count();
        assert_eq!(
            pass_keys_registered,
            after_prepare.aot_artifact_keys.len(),
            "continuation must keep all prepared process-local callback entries resolvable"
        );
        reportln!(
            "[LSODE2 continuation registry pass] pass={pass}; process_entries={}; prepared_keys_registered={}/{}",
            registry_after_pass.total_entries(),
            pass_keys_registered,
            after_prepare.aot_artifact_keys.len(),
        );
        let rss_delta = rss
            .zip(rss_after_prepare)
            .map(|(current, prepared)| current as i128 - prepared as i128);
        reportln!(
            "series | {pass} | - | {target_solve_total_ms:.3} | {rss:?} | {rss_delta:?} | {} | {} | {} | {} | {} | {} | {} | {} | {TARGETS}",
            telemetry_after
                .allocated_bytes
                .saturating_sub(telemetry_before.allocated_bytes),
            telemetry_after.aot_artifact_keys.len(),
            telemetry_after.aot_runtime_ready,
            telemetry_after.aot_build_attempts,
            telemetry_after.aot_link_attempts,
            after
                .native_residual_calls
                .saturating_sub(before.native_residual_calls),
            after
                .native_jacobian_calls
                .saturating_sub(before.native_jacobian_calls),
            after
                .native_linear_solve_calls
                .saturating_sub(before.native_linear_solve_calls),
        );
    }

    let telemetry_before_drop = telemetry.snapshot();
    drop(solver);
    let registry_after_primary_drop = try_linked_aot_runtime_registry_snapshot()
        .expect("linked runtime registry should be readable after primary drop");
    let rss_after_primary_drop = current_process_rss_bytes();
    reportln!(
        "primary_dropped_peer_live | 0 | - | 0.000 | {rss_after_primary_drop:?} | {:?} | 0 | {} | {} | {} | {} | 0 | 0 | 0 | 0",
        rss_after_primary_drop
            .zip(rss_after_prepare)
            .map(|(current, prepared)| current as i128 - prepared as i128),
        peer_after_prepare.aot_artifact_keys.len(),
        peer_after_prepare.aot_runtime_ready,
        peer_after_prepare.aot_build_attempts,
        peer_after_prepare.aot_link_attempts,
    );
    let peer_before_survivor_solve = peer_telemetry.snapshot();
    let survivor_target = parameter_continuation_target(&workload.parameter_values, 1);
    peer_solver
        .set_parameter_values(survivor_target)
        .expect("surviving peer parameter rebind should succeed");
    let survivor_started = Instant::now();
    let survivor_summary = peer_solver
        .solve_with_summary()
        .expect("peer must remain usable after primary solver drop");
    let survivor_solve_ms = survivor_started.elapsed().as_secs_f64() * 1.0e3;
    let peer_after_survivor_solve = peer_telemetry.snapshot();
    assert!(
        survivor_summary
            .native_integration_solve
            .as_ref()
            .is_some_and(|native| native.reached_t_bound || native.reached_stop_condition)
    );
    assert!(
        survivor_summary
            .final_y
            .as_ref()
            .is_some_and(|state| state.iter().all(|value| value.is_finite()))
    );
    assert_eq!(
        peer_after_survivor_solve.aot_artifact_keys,
        peer_after_prepare.aot_artifact_keys
    );
    assert_eq!(
        peer_after_survivor_solve.aot_build_attempts,
        peer_after_prepare.aot_build_attempts
    );
    assert_eq!(
        peer_after_survivor_solve.aot_link_attempts,
        peer_after_prepare.aot_link_attempts
    );
    let rss_after_survivor_solve = current_process_rss_bytes();
    reportln!(
        "peer_survivor_solve | 0 | 1 | {survivor_solve_ms:.3} | {:?} | {:?} | {} | {} | {} | {} | {} | {} | {} | {} | 1",
        rss_after_survivor_solve,
        rss_after_survivor_solve
            .zip(rss_after_prepare)
            .map(|(current, prepared)| current as i128 - prepared as i128),
        peer_after_survivor_solve
            .allocated_bytes
            .saturating_sub(peer_before_survivor_solve.allocated_bytes),
        peer_after_survivor_solve.aot_artifact_keys.len(),
        peer_after_survivor_solve.aot_runtime_ready,
        peer_after_survivor_solve.aot_build_attempts,
        peer_after_survivor_solve.aot_link_attempts,
        survivor_summary.evaluation_telemetry.residual_evaluations,
        survivor_summary.evaluation_telemetry.jacobian_evaluations,
        survivor_summary.evaluation_telemetry.linear_solves,
    );
    drop(peer_solver);
    let registry_after_all_drops = try_linked_aot_runtime_registry_snapshot()
        .expect("linked runtime registry should be readable after all solver drops");
    let rss_after_all_drops = current_process_rss_bytes();
    reportln!(
        "all_dropped | 0 | - | 0.000 | {rss_after_all_drops:?} | {:?} | 0 | {} | {} | {} | {} | 0 | 0 | 0 | 0",
        rss_after_all_drops
            .zip(rss_after_prepare)
            .map(|(current, prepared)| current as i128 - prepared as i128),
        peer_after_survivor_solve.aot_artifact_keys.len(),
        peer_after_survivor_solve.aot_runtime_ready,
        peer_after_survivor_solve.aot_build_attempts,
        peer_after_survivor_solve.aot_link_attempts,
    );
    reportln!(
        "[LSODE2 linked AOT registry retention] after_primary_drop={} after_all_drops={}; retained_primary_keys={}/{}; note=global entries may intentionally outlive solvers for reconnect; this is observable retention, not by itself a leak",
        registry_after_primary_drop.total_entries(),
        registry_after_all_drops.total_entries(),
        after_prepare
            .aot_artifact_keys
            .iter()
            .filter(|key| registry_after_all_drops.contains_problem_key(key))
            .count(),
        after_prepare.aot_artifact_keys.len(),
    );
    assert!(rss_before_prepare.is_some() && rss_after_prepare.is_some());
    assert!(rss_with_two_solvers.is_some() && rss_after_all_drops.is_some());
    let sampled_peak_rss = peak_rss_sampler
        .stop()
        .expect("RSS sampling should produce a peak measurement");
    reportln!(
        "[LSODE2 continuation sampled peak RSS] baseline_after_prepare_bytes={rss_after_prepare:?}; peak_after_prepare_bytes={sampled_peak_rss}; peak_delta_bytes={:?}; sampling_interval_ms=10; sampler_overlaps_diagnostic_work=true",
        rss_after_prepare.map(|baseline| sampled_peak_rss as i128 - baseline as i128),
    );
    assert_eq!(
        telemetry_before_drop.aot_build_attempts, after_prepare.aot_build_attempts,
        "numeric continuation must not rebuild the prepared AOT artifact"
    );
}

#[test]
#[ignore = "release diagnostic: reproduce repeated Criterion warm-pass scaling at n=512 Sparse"]
fn lsode2_parameter_continuation_diffusion_sparse_n512_repeated_warm_pass_diagnostic() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_diffusion_sparse_n512_repeated_warm_pass_diagnostic",
    );
    const DIMENSION: usize = 512;
    const MAX_TARGET: usize = 64;
    const PASSES: usize = 8;
    let base_parameters = DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]);
    reportln!(
        "[LSODE2 continuation repeated-pass anomaly] diffusion Sparse n={DIMENSION}; targets=1..={MAX_TARGET}; passes={PASSES}; one prepared solver per route"
    );
    reportln!(
        "frontend | execution | pass | pass_ms | residual_calls | jacobian_calls | linear_solves | accepted | rejected | iterations | final_t | termination | finite"
    );

    for (frontend, assembly, aot) in [
        (
            "ExprLegacy",
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            false,
        ),
        (
            "ExprLegacy",
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            true,
        ),
        (
            "AtomViewNative",
            Lsode2SymbolicAssemblyBackend::AtomView,
            false,
        ),
        (
            "AtomViewNative",
            Lsode2SymbolicAssemblyBackend::AtomView,
            true,
        ),
    ] {
        let output_dir = tempdir().expect("repeated-pass diagnostic output directory should exist");
        let telemetry = IvpTelemetry::detailed();
        let config = large_diffusion_diagnostic_config(
            DIMENSION,
            assembly,
            aot,
            false,
            telemetry,
            output_dir.path(),
        );
        let mut solver =
            Lsode2Solver::new(config).expect("repeated-pass diagnostic solver should construct");
        solver
            .prepare()
            .expect("repeated-pass diagnostic solver preparation should succeed");

        for pass in 1..=PASSES {
            let before = solver.native_statistics();
            let started = Instant::now();
            let mut last_summary = None;
            for target_index in 1..=MAX_TARGET {
                solver
                    .set_parameter_values(diffusion_diagnostic_target(
                        &base_parameters,
                        target_index,
                    ))
                    .expect("repeated-pass parameter rebind should succeed");
                let summary = solver
                    .solve_with_summary()
                    .expect("repeated-pass solve should succeed");
                assert!(
                    summary
                        .final_y
                        .as_ref()
                        .is_some_and(|state| { state.iter().all(|value| value.is_finite()) })
                );
                last_summary = summary.native_integration_solve.clone();
            }
            let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
            let after = solver.native_statistics();
            let native = last_summary.expect("repeated-pass must produce native summary");
            reportln!(
                "{frontend} | {} | {pass} | {elapsed_ms:.3} | {} | {} | {} | {} | {} | {} | {:.6e} | {} | true",
                if aot { "AOT" } else { "Lambdify" },
                after
                    .native_residual_calls
                    .saturating_sub(before.native_residual_calls),
                after
                    .native_jacobian_calls
                    .saturating_sub(before.native_jacobian_calls),
                after
                    .native_linear_solve_calls
                    .saturating_sub(before.native_linear_solve_calls),
                native.accepted_steps,
                native.rejected_steps,
                native.total_iterations,
                native.final_t,
                native.termination_kind.label(),
            );
        }
    }
}

#[test]
#[ignore = "release diagnostic: isolate per-target diffusion continuation scaling"]
fn lsode2_parameter_continuation_diffusion_per_target_diagnostic() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_diffusion_per_target_diagnostic",
    );
    const DIMENSION: usize = 1024;
    const MAX_TARGET: usize = 256;
    let workload = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(DIMENSION);
    let output_dir = tempdir().expect("continuation diagnostic output directory should exist");
    let config = Lsode2ProblemConfig::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.25,
        0.02,
        1.0e-7,
        1.0e-9,
    )
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_equation_parameters(workload.parameter_names)
    .with_equation_parameter_values(workload.parameter_values)
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
    })
    .with_telemetry(IvpTelemetry::detailed())
    .with_native_banded_faithful_aot_c_tcc(output_dir.path().to_path_buf());

    let base_parameters = DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]);
    let mut solver = Lsode2Solver::new(config).expect("diagnostic solver should construct");
    solver
        .prepare()
        .expect("diagnostic solver preparation should succeed");
    let checkpoints = [1usize, 4, 16, 64, 128, MAX_TARGET];
    let mut previous = solver.summary().native_statistics;
    let started_total = Instant::now();
    for target_index in 1..=MAX_TARGET {
        let target = diffusion_diagnostic_target(&base_parameters, target_index);
        solver
            .set_parameter_values(target)
            .expect("diagnostic numeric rebind should succeed");
        let started = Instant::now();
        let summary = solver
            .solve_with_summary()
            .expect("diagnostic continuation solve should succeed");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
        let current = summary.native_statistics;
        assert!(
            summary
                .final_y
                .as_ref()
                .is_some_and(|state| state.iter().all(|value| value.is_finite())),
            "diagnostic target {target_index} produced non-finite state"
        );
        if checkpoints.contains(&target_index) {
            reportln!(
                "target={target_index} solve_ms={elapsed_ms:.3} residual_calls={} jacobian_calls={} linear_solves={} accepted={} rejected={} cumulative_ms={:.3}",
                current
                    .native_residual_calls
                    .saturating_sub(previous.native_residual_calls),
                current
                    .native_jacobian_calls
                    .saturating_sub(previous.native_jacobian_calls),
                current
                    .native_linear_solve_calls
                    .saturating_sub(previous.native_linear_solve_calls),
                current
                    .native_step_accepts
                    .saturating_sub(previous.native_step_accepts),
                current
                    .native_step_rejects_error_test
                    .saturating_sub(previous.native_step_rejects_error_test)
                    + current
                        .native_step_rejects_nonlinear
                        .saturating_sub(previous.native_step_rejects_nonlinear),
                started_total.elapsed().as_secs_f64() * 1.0e3,
            );
        }
        previous = current;
    }
}

#[test]
#[ignore = "release diagnostic: repeated warm continuation pass and memory/counter stability"]
fn lsode2_parameter_continuation_repeated_warm_pass_diagnostic() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_repeated_warm_pass_diagnostic",
    );
    const DIMENSION: usize = 1024;
    const TARGETS: usize = 256;
    // Criterion performs several warm-up iterations before sampling. Keep
    // enough passes here to expose degradation that only appears after a
    // longer same-process continuation sequence.
    const PASSES: usize = 12;
    let workload = crate::numerical::LSODE2::workload_fixtures::diffusion_chain(DIMENSION);
    let output_dir = tempdir().expect("repeated continuation output directory should exist");
    let config = Lsode2ProblemConfig::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.25,
        0.02,
        1.0e-7,
        1.0e-9,
    )
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_equation_parameters(workload.parameter_names)
    .with_equation_parameter_values(workload.parameter_values)
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
    })
    .with_telemetry(IvpTelemetry::detailed())
    .with_native_banded_faithful_aot_c_tcc(output_dir.path().to_path_buf());
    let base_parameters = DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]);
    let mut solver = Lsode2Solver::new(config).expect("repeated diagnostic should construct");
    solver
        .prepare()
        .expect("repeated diagnostic preparation should succeed");

    reportln!(
        "pass | pass_ms | final_solve_ms | residual_calls | jacobian_calls | linear_solves | final_state_finite"
    );
    for pass in 1..=PASSES {
        let before = solver.native_statistics();
        let started = Instant::now();
        let mut last_solve_ms = 0.0;
        let mut final_state_finite = false;
        for target_index in 1..=TARGETS {
            solver
                .set_parameter_values(diffusion_diagnostic_target(&base_parameters, target_index))
                .expect("repeated diagnostic parameter rebind should succeed");
            let solve_started = Instant::now();
            let summary = solver
                .solve_with_summary()
                .expect("repeated diagnostic solve should succeed");
            last_solve_ms = solve_started.elapsed().as_secs_f64() * 1.0e3;
            final_state_finite = summary
                .final_y
                .as_ref()
                .is_some_and(|state| state.iter().all(|value| value.is_finite()));
            assert!(
                final_state_finite,
                "repeated pass {pass} produced a non-finite state"
            );
        }
        let after = solver.native_statistics();
        reportln!(
            "{pass} | {:.3} | {last_solve_ms:.3} | {} | {} | {} | {final_state_finite}",
            started.elapsed().as_secs_f64() * 1.0e3,
            after
                .native_residual_calls
                .saturating_sub(before.native_residual_calls),
            after
                .native_jacobian_calls
                .saturating_sub(before.native_jacobian_calls),
            after
                .native_linear_solve_calls
                .saturating_sub(before.native_linear_solve_calls),
        );
    }
}

#[test]
fn lsode2_parameter_continuation_matches_fresh_solver_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_matches_fresh_solver_matrix",
    );
    reportln!(
        "[LSODE2 parameter continuation] Lambdify ExprLegacy/AtomViewNative; fresh solver is the correctness oracle"
    );
    reportln!(
        "matrix | frontend | parameter | state_diff | time_diff | continuation_counts | fresh_counts | status"
    );

    let targets = [[1.25, 0.55], [1.75, 0.35], [2.25, 0.80]];
    for banded in [false, true] {
        for assembly in [
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Lsode2SymbolicAssemblyBackend::AtomView,
        ] {
            let matrix = if banded { "Banded" } else { "Sparse" };
            let frontend = match assembly {
                Lsode2SymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                Lsode2SymbolicAssemblyBackend::AtomView => "AtomViewNative",
            };
            let continuation_telemetry = IvpTelemetry::detailed();
            let mut continuation = Lsode2Solver::new(parameterized_config(
                assembly,
                banded,
                continuation_telemetry,
                [1.0, 0.4],
            ))
            .expect("continuation solver should construct");
            continuation
                .solve_with_summary()
                .expect("initial continuation solve should succeed");

            assert!(
                continuation
                    .set_parameter_values(DVector::from_vec(vec![1.0]))
                    .is_err(),
                "{matrix}/{frontend} must reject a parameter-count mismatch"
            );

            for parameters in targets {
                let before_continuation = continuation.summary().native_statistics;
                continuation
                    .set_parameter_values(DVector::from_vec(parameters.to_vec()))
                    .expect("numeric continuation rebind should succeed");
                let continued_summary = continuation
                    .solve_with_summary()
                    .expect("continued solve should succeed");
                let (continued_times, continued_values) = continuation.get_result();

                let mut fresh = Lsode2Solver::new(parameterized_config(
                    assembly,
                    banded,
                    IvpTelemetry::detailed(),
                    parameters,
                ))
                .expect("fresh solver should construct");
                let fresh_summary = fresh
                    .solve_with_summary()
                    .expect("fresh solve should succeed");
                let (fresh_times, fresh_values) = fresh.get_result();
                let fresh_eval = fresh_summary.evaluation_telemetry;
                let continued_native = continued_summary.native_statistics;
                let continued_residuals = continued_native
                    .native_residual_calls
                    .saturating_sub(before_continuation.native_residual_calls);
                let continued_jacobians = continued_native
                    .native_jacobian_calls
                    .saturating_sub(before_continuation.native_jacobian_calls);
                let continued_linear_solves = continued_native
                    .native_linear_solve_calls
                    .saturating_sub(before_continuation.native_linear_solve_calls);
                let state_diff = max_matrix_diff(&continued_values, &fresh_values);
                let time_diff = max_vector_diff(&continued_times, &fresh_times);

                reportln!(
                    "{matrix} | {frontend} | [{:.2},{:.2}] | {state_diff:.3e} | {time_diff:.3e} | {}/{}/{} | {}/{}/{} | ok",
                    parameters[0],
                    parameters[1],
                    continued_residuals,
                    continued_jacobians,
                    continued_linear_solves,
                    fresh_eval.residual_evaluations,
                    fresh_eval.jacobian_evaluations,
                    fresh_eval.linear_solves,
                );
                assert!(state_diff <= 1.0e-9, "{matrix}/{frontend} state drift");
                assert!(time_diff <= 1.0e-12, "{matrix}/{frontend} time drift");
                assert_eq!(
                    continued_residuals, fresh_eval.residual_evaluations,
                    "{matrix}/{frontend} residual trajectory changed"
                );
                assert_eq!(
                    continued_jacobians, fresh_eval.jacobian_evaluations,
                    "{matrix}/{frontend} Jacobian trajectory changed"
                );
                assert_eq!(
                    continued_linear_solves, fresh_eval.linear_solves,
                    "{matrix}/{frontend} linear trajectory changed"
                );
                assert_eq!(continued_summary.algorithm, fresh_summary.algorithm);
            }
        }
    }
}

#[test]
fn lsode2_parameter_continuation_reports_reuse_vs_fresh_preparation() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_reports_reuse_vs_fresh_preparation",
    );
    reportln!(
        "[LSODE2 parameter continuation performance] preparation is separated from repeated solve; debug baseline"
    );
    reportln!(
        "matrix | frontend | continuation_ms | fresh_ms | continuation_expr_to_atom | continuation_symbolic_jacobian | parameter_binds | status"
    );

    let targets = [[1.20, 0.45], [1.40, 0.50], [1.60, 0.55], [1.80, 0.60]];
    for banded in [false, true] {
        for assembly in [
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Lsode2SymbolicAssemblyBackend::AtomView,
        ] {
            let matrix = if banded { "Banded" } else { "Sparse" };
            let frontend = match assembly {
                Lsode2SymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                Lsode2SymbolicAssemblyBackend::AtomView => "AtomViewNative",
            };
            let telemetry = IvpTelemetry::detailed();
            let mut continuation = Lsode2Solver::new(parameterized_config(
                assembly,
                banded,
                telemetry.clone(),
                [1.0, 0.4],
            ))
            .expect("continuation solver should construct");
            continuation
                .solve_with_summary()
                .expect("initial continuation solve should succeed");
            let before = telemetry.snapshot();
            let started = Instant::now();
            for parameters in targets {
                continuation
                    .set_parameter_values(DVector::from_vec(parameters.to_vec()))
                    .expect("continuation rebind should succeed");
                continuation
                    .solve_with_summary()
                    .expect("continued solve should succeed");
            }
            let continuation_ms = started.elapsed().as_secs_f64() * 1.0e3;
            let after = telemetry.snapshot();

            let started = Instant::now();
            for parameters in targets {
                let mut fresh = Lsode2Solver::new(parameterized_config(
                    assembly,
                    banded,
                    IvpTelemetry::disabled(),
                    parameters,
                ))
                .expect("fresh solver should construct");
                fresh
                    .solve_with_summary()
                    .expect("fresh solve should succeed");
            }
            let fresh_ms = started.elapsed().as_secs_f64() * 1.0e3;
            let expr_to_atom = cold_calls(&after, IvpColdStage::ExprToAtom)
                .saturating_sub(cold_calls(&before, IvpColdStage::ExprToAtom));
            let symbolic_jacobian = cold_calls(&after, IvpColdStage::SymbolicJacobian)
                .saturating_sub(cold_calls(&before, IvpColdStage::SymbolicJacobian));
            reportln!(
                "{matrix} | {frontend} | {continuation_ms:.3} | {fresh_ms:.3} | {expr_to_atom} | {symbolic_jacobian} | {} | ok",
                after.parameter_binds.saturating_sub(before.parameter_binds),
            );
            assert_eq!(
                expr_to_atom, 0,
                "{matrix}/{frontend} continuation must not rebuild Expr->Atom"
            );
            assert_eq!(
                symbolic_jacobian, 0,
                "{matrix}/{frontend} continuation must not rebuild symbolic Jacobian"
            );
            assert!(
                after.parameter_binds.saturating_sub(before.parameter_binds)
                    == targets.len() as u64,
                "{matrix}/{frontend} must record one bind per continuation value"
            );
        }
    }
}

#[test]
fn lsode2_parameter_continuation_fair_warm_performance_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_fair_warm_performance_matrix",
    );
    reportln!(
        "[LSODE2 parameter continuation fair performance] identical detailed telemetry; continuation rebind+solve versus fresh prepare+solve"
    );
    reportln!(
        "matrix | frontend | targets | continuation_ms | fresh_prepare_ms | fresh_solve_ms | fresh_total_ms | binds | new_expr_to_atom | new_symbolic_jacobian | status"
    );

    let targets = [[1.20, 0.45], [1.40, 0.50], [1.60, 0.55], [1.80, 0.60]];
    for banded in [false, true] {
        for assembly in [
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Lsode2SymbolicAssemblyBackend::AtomView,
        ] {
            let matrix = if banded { "Banded" } else { "Sparse" };
            let frontend = match assembly {
                Lsode2SymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                Lsode2SymbolicAssemblyBackend::AtomView => "AtomViewNative",
            };

            let continuation_telemetry = IvpTelemetry::detailed();
            let mut continuation = Lsode2Solver::new(parameterized_config(
                assembly,
                banded,
                continuation_telemetry.clone(),
                [1.0, 0.4],
            ))
            .expect("continuation solver should construct");
            continuation
                .solve_with_summary()
                .expect("initial continuation solve should succeed");
            let before = continuation_telemetry.snapshot();
            let continuation_started = Instant::now();
            for parameters in targets {
                continuation
                    .set_parameter_values(DVector::from_vec(parameters.to_vec()))
                    .expect("continuation rebind should succeed");
                continuation
                    .solve_with_summary()
                    .expect("continued solve should succeed");
            }
            let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1.0e3;
            let after = continuation_telemetry.snapshot();

            let fresh_telemetry = IvpTelemetry::detailed();
            let mut fresh_prepare_ms = 0.0;
            let mut fresh_solve_ms = 0.0;
            for parameters in targets {
                let prepare_started = Instant::now();
                let mut fresh = Lsode2Solver::new(parameterized_config(
                    assembly,
                    banded,
                    fresh_telemetry.clone(),
                    parameters,
                ))
                .expect("fresh solver should construct");
                fresh_prepare_ms += prepare_started.elapsed().as_secs_f64() * 1.0e3;

                let solve_started = Instant::now();
                fresh
                    .solve_with_summary()
                    .expect("fresh solve should succeed");
                fresh_solve_ms += solve_started.elapsed().as_secs_f64() * 1.0e3;
            }
            let fresh_total_ms = fresh_prepare_ms + fresh_solve_ms;
            let new_expr_to_atom = cold_calls(&after, IvpColdStage::ExprToAtom)
                .saturating_sub(cold_calls(&before, IvpColdStage::ExprToAtom));
            let new_symbolic_jacobian = cold_calls(&after, IvpColdStage::SymbolicJacobian)
                .saturating_sub(cold_calls(&before, IvpColdStage::SymbolicJacobian));
            let binds = after.parameter_binds.saturating_sub(before.parameter_binds);

            reportln!(
                "{matrix} | {frontend} | {} | {continuation_ms:.3} | {fresh_prepare_ms:.3} | {fresh_solve_ms:.3} | {fresh_total_ms:.3} | {binds} | {new_expr_to_atom} | {new_symbolic_jacobian} | ok",
                targets.len(),
            );
            assert_eq!(binds, targets.len() as u64);
            assert_eq!(new_expr_to_atom, 0);
            assert_eq!(new_symbolic_jacobian, 0);
            assert_eq!(
                after.errors, before.errors,
                "{matrix}/{frontend} continuation must not add telemetry errors"
            );
        }
    }
}
