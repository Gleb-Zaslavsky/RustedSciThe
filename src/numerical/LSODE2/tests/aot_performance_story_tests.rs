//! Release-oriented LSODE2 AOT performance stories.
//!
//! These are deliberately separate from parity tests.  They measure cold
//! preparation/build/link, warm callback throughput, and chunk-policy
//! break-even while retaining enough integer counters to prove that the
//! compared routes did the same amount of numerical work.  They are ignored
//! by default because a single matrix may invoke several external toolchains.

use super::native_jacobian::{
    try_prepare_native_atomview_jacobian_runtime, NativeAtomJacobianRuntime, NativeJacobianStorage,
};
use super::story_support::{chain_equations, chain_solver_config, chain_state, ChainMatrixRoute};
use super::{IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry};
use super::{
    IvpWarmStage, Lsode2AotProfile, Lsode2AotToolchain, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::codegen_aot_driver::GeneratedAotBuildResult;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::ivp_telemetry::IvpTelemetryRoute;
use crate::symbolic::symbolic_ivp::{
    prepare_symbolic_ivp_residual_problem, IvpSymbolicAssemblyBackend,
    PreparedSymbolicIvpResidualProblem, SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    prepare_generated_symbolic_ivp_banded_backend, prepare_generated_symbolic_ivp_sparse_backend,
    SelectedSymbolicIvpBackendKind, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::DVector;
use std::hint::black_box;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;
use tempfile::tempdir;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn dimensions_from_env(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 2)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn usize_from_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn parameter_names() -> Vec<String> {
    ["k", "d", "q", "nl"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn parameter_values() -> DVector<f64> {
    DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010])
}

fn variables(dimension: usize) -> Vec<String> {
    (0..dimension).map(|index| format!("y{index}")).collect()
}

fn options(
    frontend: IvpSymbolicAssemblyBackend,
    telemetry: IvpTelemetry,
) -> SymbolicIvpProblemOptions {
    SymbolicIvpProblemOptions::new()
        .with_equation_parameters(parameter_names())
        .with_equation_parameter_values(parameter_values())
        .with_symbolic_assembly_backend(frontend)
        .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
        .with_telemetry(telemetry)
}

fn generated_config(
    output_parent: &Path,
    compiler: &str,
    residual_chunking: ResidualChunkingStrategy,
    sparse_chunking: SparseChunkingStrategy,
) -> SymbolicIvpGeneratedBackendConfig {
    let config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(output_parent.to_path_buf()))
        .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
        .with_residual_chunking_strategy(residual_chunking)
        .with_sparse_jacobian_chunking_strategy(sparse_chunking);
    match compiler {
        "tcc" => config.with_c_tcc(),
        "gcc" => config.with_c_gcc(),
        "rust" => config.with_rust(),
        "zig" => config.with_zig(),
        other => panic!("unknown AOT compiler {other}"),
    }
}

/// Forces a fresh artifact for frontend performance diagnostics.
///
/// The linked AOT registry is process-local and keyed by prepared-problem
/// identity, not by the temporary output directory. A diagnostic that uses
/// `BuildIfMissing` can therefore compare a newly built Atom artifact with an
/// older linked Expr artifact from an earlier test. Lifecycle stories keep
/// `BuildIfMissing`; callback comparisons must not.
fn fresh_diagnostic_config(
    output_parent: &Path,
    compiler: &str,
    residual_chunking: ResidualChunkingStrategy,
    sparse_chunking: SparseChunkingStrategy,
) -> SymbolicIvpGeneratedBackendConfig {
    generated_config(
        output_parent,
        compiler,
        residual_chunking,
        sparse_chunking,
    )
    .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
        profile: AotBuildProfile::Release,
    })
}

fn cold_ms(snapshot: &super::IvpTelemetrySnapshot, stage: IvpColdStage) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

/// Collects generated-source shape after the measured phase has finished.
/// Filesystem reads are deliberately excluded from both `prepare_ms` and
/// callback timings; these fields only help correlate a raw callback anomaly
/// with the emitted artifact before changing a lowering pass.
#[derive(Clone, Copy, Debug, Default)]
struct GeneratedSourceShape {
    bytes: usize,
    lines: usize,
    compute_lines: usize,
    temp_declarations: usize,
    add_ops: usize,
    sub_ops: usize,
    mul_ops: usize,
    div_ops: usize,
    pow_calls: usize,
    exp_calls: usize,
    output_writes: usize,
}

fn generated_source_shape(build: Option<&GeneratedAotBuildResult>) -> GeneratedSourceShape {
    let path = match build {
        Some(GeneratedAotBuildResult::Rust(result)) => &result.written.generated_rs,
        Some(GeneratedAotBuildResult::C(result)) => &result.written.generated_c,
        Some(GeneratedAotBuildResult::Zig(result)) => &result.written.generated_zig,
        None => return GeneratedSourceShape::default(),
    };
    let Ok(source) = std::fs::read_to_string(path) else {
        return GeneratedSourceShape::default();
    };
    let mut shape = GeneratedSourceShape {
        bytes: source.len(),
        lines: source.lines().count(),
        ..GeneratedSourceShape::default()
    };
    for line in source.lines() {
        let line = line.trim_start();
        let is_compute = line.starts_with("let t")
            || line.starts_with("double t")
            || line.starts_with("var t");
        if is_compute {
            shape.compute_lines += 1;
            shape.temp_declarations += usize::from(
                line.starts_with("double t") || line.starts_with("let t") || line.starts_with("var t"),
            );
            shape.add_ops += line.matches(" + ").count();
            shape.sub_ops += line.matches(" - ").count();
            shape.mul_ops += line.matches(" * ").count();
            shape.div_ops += line.matches(" / ").count();
        }
        shape.pow_calls += line.matches("pow(").count();
        shape.exp_calls += line.matches("exp(").count();
        shape.output_writes += usize::from(
            line.contains("out[") || line.contains("output[") || line.contains("values["),
        );
    }
    shape
}

fn warm_ms(snapshot: &super::IvpTelemetrySnapshot, stage: IvpWarmStage) -> f64 {
    snapshot.warm_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

fn warm_calls(snapshot: &super::IvpTelemetrySnapshot, stage: IvpWarmStage) -> u64 {
    snapshot.warm_stage(stage).calls
}

#[derive(Clone, Copy, Debug)]
struct CallbackTiming {
    residual_ms: f64,
    jacobian_ms: f64,
    calls: usize,
}

fn run_linked_callbacks(
    linked: &crate::symbolic::codegen::codegen_aot_runtime_link::LinkedSparseAotBackend,
    args: &[f64],
    repetitions: usize,
) -> CallbackTiming {
    let mut residual = vec![0.0; linked.residual_len];
    let mut jacobian = vec![0.0; linked.jacobian_output_len().expect("valid AOT layout")];
    let started = Instant::now();
    for _ in 0..repetitions {
        linked
            .try_residual_eval(args, &mut residual)
            .expect("AOT residual callback should succeed");
        black_box(&residual);
    }
    let residual_ms = started.elapsed().as_secs_f64() * 1.0e3;

    let started = Instant::now();
    for _ in 0..repetitions {
        linked
            .try_jacobian_values_eval(args, &mut jacobian)
            .expect("AOT Jacobian callback should succeed");
        black_box(&jacobian);
    }
    let jacobian_ms = started.elapsed().as_secs_f64() * 1.0e3;
    assert!(residual.iter().all(|value| value.is_finite()));
    assert!(jacobian.iter().all(|value| value.is_finite()));
    CallbackTiming {
        residual_ms,
        jacobian_ms,
        calls: repetitions,
    }
}

fn run_native_callbacks(
    residual: &PreparedSymbolicIvpResidualProblem,
    jacobian: &mut NativeAtomJacobianRuntime,
    matrix: &str,
    time: f64,
    state: &DVector<f64>,
    repetitions: usize,
) -> CallbackTiming {
    let mut residual_out = DVector::zeros(state.len());
    let mut args = Vec::with_capacity(1 + parameter_names().len() + state.len());
    let started = Instant::now();
    for _ in 0..repetitions {
        residual
            .try_evaluate_residual_into_with_workspace(time, state, &mut residual_out, &mut args)
            .expect("native residual callback should succeed");
        black_box(&residual_out);
    }
    let residual_ms = started.elapsed().as_secs_f64() * 1.0e3;

    let mut values = match matrix {
        "Sparse" => vec![0.0; jacobian.sparse_pattern().len()],
        "Banded" => vec![0.0; jacobian.banded_layout().expect("banded layout").2],
        other => panic!("unknown native matrix {other}"),
    };
    let started = Instant::now();
    for _ in 0..repetitions {
        match matrix {
            "Sparse" => jacobian
                .try_evaluate_sparse_values_into(time, state, &mut values)
                .expect("native sparse Jacobian callback should succeed"),
            "Banded" => jacobian
                .try_evaluate_banded_values_into(time, state, &mut values)
                .expect("native banded Jacobian callback should succeed"),
            _ => unreachable!(),
        }
        black_box(&values);
    }
    let jacobian_ms = started.elapsed().as_secs_f64() * 1.0e3;
    assert!(residual_out.iter().all(|value| value.is_finite()));
    assert!(values.iter().all(|value| value.is_finite()));
    CallbackTiming {
        residual_ms,
        jacobian_ms,
        calls: repetitions,
    }
}

fn prepare_native_lambdify(
    dimension: usize,
    matrix: &str,
    telemetry: IvpTelemetry,
) -> (
    PreparedSymbolicIvpResidualProblem,
    NativeAtomJacobianRuntime,
) {
    let equations = chain_equations(dimension);
    let variables = variables(dimension);
    let residual = prepare_symbolic_ivp_residual_problem(
        equations.clone(),
        variables.clone(),
        "t".to_string(),
        options(IvpSymbolicAssemblyBackend::AtomView, telemetry.clone()),
    )
    .expect("AtomView Lambdify residual should prepare");
    let storage = match matrix {
        "Sparse" => NativeJacobianStorage::SparseTriplets,
        "Banded" => NativeJacobianStorage::Banded {
            bandwidth: Some((1, 1)),
        },
        other => panic!("unknown matrix {other}"),
    };
    let jacobian = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        Some(&parameter_names()),
        residual.parameter_values_handle(),
        storage,
        telemetry,
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("AtomView native Lambdify Jacobian should prepare");
    (residual, jacobian)
}

fn run_aot_callback_case(
    dimension: usize,
    matrix: &str,
    frontend: IvpSymbolicAssemblyBackend,
    compiler: &str,
    repetitions: usize,
    residual_chunking: ResidualChunkingStrategy,
    sparse_chunking: SparseChunkingStrategy,
) {
    let telemetry = IvpTelemetry::detailed();
    let equations = chain_equations(dimension);
    let variables = variables(dimension);
    let output = tempdir().expect("AOT performance output directory should exist");
    let started = Instant::now();
    let prepared = match matrix {
        "Sparse" => prepare_generated_symbolic_ivp_sparse_backend(
            equations,
            variables,
            "t".to_string(),
            options(frontend, telemetry.clone()),
            fresh_diagnostic_config(
                output.path(),
                compiler,
                residual_chunking,
                sparse_chunking,
            ),
        )
        .expect("large sparse AOT performance preparation should succeed"),
        "Banded" => prepare_generated_symbolic_ivp_banded_backend(
            equations,
            variables,
            "t".to_string(),
            (1, 1),
            options(frontend, telemetry.clone()),
            fresh_diagnostic_config(
                output.path(),
                compiler,
                residual_chunking,
                sparse_chunking,
            ),
        )
        .expect("large banded AOT performance preparation should succeed"),
        other => panic!("unknown AOT matrix {other}"),
    };
    assert_eq!(
        prepared.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );
    let expected_route = match frontend {
        IvpSymbolicAssemblyBackend::ExprLegacy => IvpTelemetryRoute::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomViewExprCompat => IvpTelemetryRoute::AtomViewExprCompat,
        IvpSymbolicAssemblyBackend::AtomView => IvpTelemetryRoute::AtomViewNative,
    };
    assert_eq!(
        prepared.telemetry.snapshot().route,
        expected_route,
        "AOT callback matrix must report the requested symbolic frontend"
    );
    let prepare_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let linked = prepared
        .linked_backend
        .as_ref()
        .expect("AOT performance case should expose linked callbacks");
    let state = chain_state(dimension);
    let mut args = Vec::with_capacity(1 + parameter_names().len() + dimension);
    args.push(0.125);
    args.extend(parameter_values().iter().copied());
    args.extend(state.iter().copied());
    let timing = run_linked_callbacks(linked, &args, repetitions);
    let snapshot = prepared.telemetry.snapshot();
    let source_shape = generated_source_shape(prepared.build_result.as_ref());
    reportln!(
        "AOT | {compiler} | {matrix} | {:?} | {dimension} | {prepare_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
        frontend,
        cold_ms(&snapshot, IvpColdStage::AtomResidualPreparation),
        cold_ms(&snapshot, IvpColdStage::AtomJacobianPreparation),
        timing.residual_ms / repetitions as f64,
        timing.jacobian_ms / repetitions as f64,
        cold_ms(&snapshot, IvpColdStage::ExprToAtom),
        cold_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
        cold_ms(&snapshot, IvpColdStage::Simplification),
        cold_ms(&snapshot, IvpColdStage::SparsePattern),
        cold_ms(&snapshot, IvpColdStage::AotMaterialization),
        cold_ms(&snapshot, IvpColdStage::AotBuild),
        cold_ms(&snapshot, IvpColdStage::AotLink),
        snapshot.aot_build_attempts,
        snapshot.aot_link_attempts,
        timing.calls,
        linked.residual_len,
        linked.jacobian_output_len().expect("valid AOT layout"),
        source_shape.bytes / 1024,
        source_shape.lines,
        source_shape.compute_lines,
    );
}

fn median_sample(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn residual_boundary_samples(
    linked: &crate::symbolic::codegen::codegen_aot_runtime_link::LinkedSparseAotBackend,
    args: &[f64],
    repetitions: usize,
    samples: usize,
) -> (f64, f64, f64, f64) {
    let mut raw_samples = Vec::with_capacity(samples);
    let mut checked_samples = Vec::with_capacity(samples);
    let mut max_diff = 0.0_f64;
    let mut raw_output = vec![0.0; linked.residual_len];
    let mut checked_output = vec![0.0; linked.residual_len];

    for _ in 0..samples {
        for _ in 0..8 {
            (linked.residual_eval)(args, &mut raw_output);
            linked
                .try_residual_eval(args, &mut checked_output)
                .expect("typed residual callback should succeed");
        }

        let started = Instant::now();
        for _ in 0..repetitions {
            (linked.residual_eval)(args, &mut raw_output);
            black_box(&raw_output);
        }
        raw_samples.push(started.elapsed().as_secs_f64() * 1.0e9 / repetitions as f64);

        let started = Instant::now();
        for _ in 0..repetitions {
            linked
                .try_residual_eval(args, &mut checked_output)
                .expect("typed residual callback should succeed");
            black_box(&checked_output);
        }
        checked_samples.push(started.elapsed().as_secs_f64() * 1.0e9 / repetitions as f64);

        max_diff = max_diff.max(
            raw_output
                .iter()
                .zip(checked_output.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max),
        );
    }

    let raw_ns = median_sample(&mut raw_samples);
    let checked_ns = median_sample(&mut checked_samples);
    (raw_ns, checked_ns, checked_ns - raw_ns, max_diff)
}

#[test]
#[ignore = "release-only AOT residual boundary diagnostic; isolates generated code from typed wrapper overhead"]
fn lsode2_aot_residual_boundary_isolation_exprlegacy_vs_atomview() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_performance_story_tests::lsode2_aot_residual_boundary_isolation_exprlegacy_vs_atomview",
    );
    let dimensions = dimensions_from_env(
        "LSODE2_AOT_RESIDUAL_DIAGNOSTIC_DIMENSIONS",
        &[128, 256, 512],
    );
    let repetitions = usize_from_env("LSODE2_AOT_RESIDUAL_DIAGNOSTIC_REPETITIONS", 2_000);
    let samples = usize_from_env("LSODE2_AOT_RESIDUAL_DIAGNOSTIC_SAMPLES", 5);
    reportln!(
        "[LSODE2 AOT residual boundary isolation] dimensions={dimensions:?}; compiler=tcc; samples={samples}; repetitions={repetitions}; Sparse; reused args/output; raw closure bypasses typed validation"
    );
    reportln!(
        "frontend | dimension | raw_ns/call | typed_ns/call | typed_boundary_ns/call | max_diff | build_attempts | link_attempts | source_kb | source_lines | compute_lines | temp_declarations | add | sub | mul | div | pow | exp | output_writes"
    );

    for dimension in dimensions {
        let state = chain_state(dimension);
        let mut args = Vec::with_capacity(1 + parameter_names().len() + dimension);
        args.push(0.125);
        args.extend(parameter_values().iter().copied());
        args.extend(state.iter().copied());

        for (label, frontend) in [
            ("ExprLegacy-AOT", IvpSymbolicAssemblyBackend::ExprLegacy),
            ("AtomView-AOT", IvpSymbolicAssemblyBackend::AtomView),
        ] {
            let telemetry = IvpTelemetry::detailed();
            let output = tempdir().expect("AOT diagnostic output directory should exist");
            let prepared = prepare_generated_symbolic_ivp_sparse_backend(
                chain_equations(dimension),
                variables(dimension),
                "t".to_string(),
                options(frontend, telemetry.clone()),
                fresh_diagnostic_config(
                    output.path(),
                    "tcc",
                    ResidualChunkingStrategy::Whole,
                    SparseChunkingStrategy::Whole,
                ),
            )
            .expect("AOT residual boundary diagnostic preparation should succeed");
            let linked = prepared
                .linked_backend
                .as_ref()
                .expect("AOT residual boundary diagnostic should link a backend");
            let (raw_ns, typed_ns, boundary_ns, max_diff) =
                residual_boundary_samples(linked, &args, repetitions, samples);
            let snapshot = telemetry.snapshot();
            let source_shape = generated_source_shape(prepared.build_result.as_ref());
            assert!(max_diff <= 1.0e-12);
            reportln!(
                "{label} | {dimension} | {raw_ns:.3} | {typed_ns:.3} | {boundary_ns:.3} | {max_diff:.3e} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                snapshot.aot_build_attempts,
                snapshot.aot_link_attempts,
                source_shape.bytes / 1024,
                source_shape.lines,
                source_shape.compute_lines,
                source_shape.temp_declarations,
                source_shape.add_ops,
                source_shape.sub_ops,
                source_shape.mul_ops,
                source_shape.div_ops,
                source_shape.pow_calls,
                source_shape.exp_calls,
                source_shape.output_writes,
            );
        }
    }
}

fn run_lambdify_callback_case(dimension: usize, matrix: &str, repetitions: usize) {
    let telemetry = IvpTelemetry::detailed();
    let started = Instant::now();
    let (residual, mut jacobian) = prepare_native_lambdify(dimension, matrix, telemetry.clone());
    let prepare_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let state = chain_state(dimension);
    let timing = run_native_callbacks(&residual, &mut jacobian, matrix, 0.125, &state, repetitions);
    let snapshot = telemetry.snapshot();
    reportln!(
        "Lambdify | - | {matrix} | AtomViewNative | {dimension} | {prepare_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
        cold_ms(&snapshot, IvpColdStage::AtomResidualPreparation),
        cold_ms(&snapshot, IvpColdStage::AtomJacobianPreparation),
        timing.residual_ms / repetitions as f64,
        timing.jacobian_ms / repetitions as f64,
        cold_ms(&snapshot, IvpColdStage::ExprToAtom),
        cold_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
        cold_ms(&snapshot, IvpColdStage::Simplification),
        cold_ms(&snapshot, IvpColdStage::SparsePattern),
        cold_ms(&snapshot, IvpColdStage::AotMaterialization),
        cold_ms(&snapshot, IvpColdStage::AotBuild),
        cold_ms(&snapshot, IvpColdStage::AotLink),
        0,
        0,
        timing.calls,
        dimension,
        if matrix == "Sparse" {
            dimension * 3 - 2
        } else {
            (1 + 1 + 1) * dimension
        },
        0,
        0,
        0,
    );
}

fn run_full_solver_case(dimension: usize, matrix: ChainMatrixRoute, aot: bool) {
    let telemetry = IvpTelemetry::detailed();
    let output = tempdir().expect("AOT full-solve output directory should exist");
    let mut config = chain_solver_config(
        dimension,
        IvpSymbolicAssemblyBackend::AtomView,
        matrix,
        IvpLambdifyExecutionPolicy::Sequential,
        telemetry.clone(),
    );
    if aot {
        let generated = generated_config(
            output.path(),
            "tcc",
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        );
        config = match matrix {
            ChainMatrixRoute::Sparse => config.with_native_sparse_faer_generated_backend(generated),
            ChainMatrixRoute::Banded => {
                config.with_native_banded_faithful_generated_backend(generated)
            }
        };
        config = config.with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            },
        });
    }
    let route = if aot {
        "AtomViewNative-AOT"
    } else {
        "AtomViewNative-Lambdify"
    };
    let total_started = Instant::now();
    let mut solver =
        Lsode2Solver::new(config).expect("large AOT performance solver should construct");
    let prepare_started = Instant::now();
    solver
        .prepare()
        .expect("large AOT performance solver should prepare");
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1.0e3;
    let solve_started = Instant::now();
    solver
        .solve()
        .expect("large AOT performance solver should solve");
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1.0e3;
    let summary_started = Instant::now();
    let summary = solver.summary();
    let summary_ms = summary_started.elapsed().as_secs_f64() * 1.0e3;
    let snapshot = telemetry.snapshot();
    let final_state = summary
        .final_y
        .as_ref()
        .expect("large AOT performance solve should expose final state");
    assert!(final_state.iter().all(|value| value.is_finite()));
    reportln!(
        "{route} | {} | {dimension} | {prepare_ms:.3} | {solve_ms:.3} | {summary_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {:.3}",
        matrix.label(),
        total_started.elapsed().as_secs_f64() * 1.0e3,
        cold_ms(&snapshot, IvpColdStage::AtomResidualPreparation),
        cold_ms(&snapshot, IvpColdStage::AtomJacobianPreparation),
        cold_ms(&snapshot, IvpColdStage::AotMaterialization),
        cold_ms(&snapshot, IvpColdStage::AotBuild),
        cold_ms(&snapshot, IvpColdStage::AotLink),
        super::IvpTelemetrySnapshot::warm_stage(&snapshot, IvpWarmStage::ResidualCallback)
            .elapsed
            .as_secs_f64()
            * 1.0e3,
        super::IvpTelemetrySnapshot::warm_stage(&snapshot, IvpWarmStage::JacobianCallback)
            .elapsed
            .as_secs_f64()
            * 1.0e3,
        super::IvpTelemetrySnapshot::warm_stage(&snapshot, IvpWarmStage::Factorization)
            .elapsed
            .as_secs_f64()
            * 1.0e3,
        super::IvpTelemetrySnapshot::warm_stage(&snapshot, IvpWarmStage::RhsSolve)
            .elapsed
            .as_secs_f64()
            * 1.0e3,
        snapshot.residual_evaluations,
        snapshot.jacobian_evaluations,
        snapshot.linear_solve_requests,
        snapshot.accepted_steps,
        snapshot.rejected_steps,
        max_vector_norm(final_state),
    );
    reportln!(
        "{route} | {} | {dimension} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}",
        matrix.label(),
        cold_ms(&snapshot, IvpColdStage::SolverPreparation),
        cold_ms(&snapshot, IvpColdStage::BridgePreparation),
        cold_ms(&snapshot, IvpColdStage::NativeCallbackPreparation),
        warm_ms(&snapshot, IvpWarmStage::Solve),
        warm_ms(&snapshot, IvpWarmStage::Summary),
    );
    let controller_ms = warm_ms(&snapshot, IvpWarmStage::Controller);
    let iteration_ms = warm_ms(&snapshot, IvpWarmStage::ControllerIteration);
    let controller_outside_iterations = (controller_ms - iteration_ms).max(0.0);
    let solve_minus_controller_ms = (solve_ms - controller_ms).max(0.0);
    reportln!(
        "{route} | {} | {dimension} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {}",
        matrix.label(),
        controller_ms,
        solve_minus_controller_ms,
        warm_ms(&snapshot, IvpWarmStage::NativeEngineSetup),
        warm_ms(&snapshot, IvpWarmStage::NativeResultAssembly),
        controller_outside_iterations,
        warm_ms(&snapshot, IvpWarmStage::ControllerPredictor),
        warm_ms(&snapshot, IvpWarmStage::ControllerStepSetup),
        iteration_ms,
        warm_ms(&snapshot, IvpWarmStage::ControllerOutcome),
        warm_ms(&snapshot, IvpWarmStage::ControllerStopCondition),
        warm_ms(&snapshot, IvpWarmStage::ControllerMethodPolicy),
        warm_ms(&snapshot, IvpWarmStage::ControllerMethodSwitch),
        warm_ms(&snapshot, IvpWarmStage::ResidualCallback),
        warm_ms(&snapshot, IvpWarmStage::JacobianCallback),
        warm_ms(&snapshot, IvpWarmStage::ResidualEvaluation),
        warm_ms(&snapshot, IvpWarmStage::JacobianEvaluation),
        warm_ms(&snapshot, IvpWarmStage::ResidualOutputAssembly),
        warm_ms(&snapshot, IvpWarmStage::JacobianOutputAssembly),
        warm_ms(&snapshot, IvpWarmStage::Factorization),
        warm_ms(&snapshot, IvpWarmStage::RhsSolve),
        warm_calls(&snapshot, IvpWarmStage::Controller),
        warm_calls(&snapshot, IvpWarmStage::ControllerIteration),
        warm_calls(&snapshot, IvpWarmStage::ResidualCallback),
        warm_calls(&snapshot, IvpWarmStage::JacobianCallback),
    );
}

fn max_vector_norm(vector: &DVector<f64>) -> f64 {
    vector.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[test]
#[ignore = "release-only AOT callback stage matrix; compares cold lifecycle and warm throughput"]
fn lsode2_aot_large_callback_stage_performance_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_performance_story_tests::lsode2_aot_large_callback_stage_performance_matrix",
    );
    let dimensions = dimensions_from_env("LSODE2_AOT_PERF_DIMENSIONS", &[128, 256, 512]);
    let repetitions = usize_from_env("LSODE2_AOT_CALLBACK_REPETITIONS", 200);
    reportln!(
        "[LSODE2 AOT callback performance] dimensions={dimensions:?}; compiler=tcc; repetitions={repetitions}; Dense excluded; build/link included only in prepare_ms; source shape is read after timing; compact-Banded ExprLegacy is reported as unsupported"
    );
    reportln!(
        "route | compiler | matrix | frontend | dimension | prepare_ms | atom_residual_prepare_ms | atom_jacobian_prepare_ms | residual_ms/call | jacobian_ms/call | expr_to_atom_ms | diff_ms | simplify_ms | pattern_ms | materialize_ms | build_ms | link_ms | build_attempts | link_attempts | callback_calls | residual_len | jacobian_output_len | source_kb | source_lines | compute_lines"
    );
    for dimension in dimensions {
        run_lambdify_callback_case(dimension, "Sparse", repetitions);
        run_lambdify_callback_case(dimension, "Banded", repetitions);
        run_aot_callback_case(
            dimension,
            "Sparse",
            IvpSymbolicAssemblyBackend::ExprLegacy,
            "tcc",
            repetitions,
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        );
        run_aot_callback_case(
            dimension,
            "Sparse",
            IvpSymbolicAssemblyBackend::AtomView,
            "tcc",
            repetitions,
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        );
        reportln!(
            "AOT | tcc | Banded | ExprLegacy | unsupported | compact-Banded AOT currently requires AtomView assembly"
        );
        run_aot_callback_case(
            dimension,
            "Banded",
            IvpSymbolicAssemblyBackend::AtomView,
            "tcc",
            repetitions,
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        );
    }
}

#[test]
#[ignore = "release-only AOT toolchain performance matrix; external compilers are optional"]
fn lsode2_aot_toolchain_callback_performance_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_performance_story_tests::lsode2_aot_toolchain_callback_performance_matrix",
    );
    let dimensions = dimensions_from_env("LSODE2_AOT_TOOLCHAIN_DIMENSIONS", &[128, 256]);
    let repetitions = usize_from_env("LSODE2_AOT_TOOLCHAIN_REPETITIONS", 100);
    reportln!(
        "[LSODE2 AOT toolchain callback performance] dimensions={dimensions:?}; repetitions={repetitions}; Sparse/Banded AtomViewNative; source shape is read after timing; unavailable external toolchains are skipped"
    );
    reportln!(
        "route | compiler | matrix | frontend | dimension | prepare_ms | atom_residual_prepare_ms | atom_jacobian_prepare_ms | residual_ms/call | jacobian_ms/call | expr_to_atom_ms | diff_ms | simplify_ms | pattern_ms | materialize_ms | build_ms | link_ms | build_attempts | link_attempts | callback_calls | residual_len | jacobian_output_len | source_kb | source_lines | compute_lines"
    );
    for (compiler, command) in [
        ("tcc", "tcc"),
        ("gcc", "gcc"),
        ("rust", "rustc"),
        ("zig", "zig"),
    ] {
        if compiler != "rust" && !command_available(command) {
            reportln!("{compiler} | skipped | command-unavailable");
            continue;
        }
        for dimension in dimensions.iter().copied() {
            for matrix in ["Sparse", "Banded"] {
                run_aot_callback_case(
                    dimension,
                    matrix,
                    IvpSymbolicAssemblyBackend::AtomView,
                    compiler,
                    repetitions,
                    ResidualChunkingStrategy::Whole,
                    SparseChunkingStrategy::Whole,
                );
            }
        }
    }
}

#[test]
#[ignore = "release-only AOT chunking and Sequential/Parallel/Auto break-even story"]
fn lsode2_aot_chunking_policy_callback_break_even_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_performance_story_tests::lsode2_aot_chunking_policy_callback_break_even_story",
    );
    let dimension = usize_from_env("LSODE2_AOT_CHUNK_DIMENSION", 512);
    let repetitions = usize_from_env("LSODE2_AOT_CHUNK_REPETITIONS", 200);
    let chunk = usize_from_env("LSODE2_AOT_CHUNK_SIZE", 16);
    let telemetry = IvpTelemetry::detailed();
    let output = tempdir().expect("AOT chunking output directory should exist");
    let prepared = prepare_generated_symbolic_ivp_sparse_backend(
        chain_equations(dimension),
        variables(dimension),
        "t".to_string(),
        options(IvpSymbolicAssemblyBackend::AtomView, telemetry),
        generated_config(
            output.path(),
            "tcc",
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: chunk,
            },
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: chunk,
            },
        ),
    )
    .expect("chunked AOT preparation should succeed");
    assert_eq!(
        prepared.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );
    let linked = prepared
        .linked_backend
        .as_ref()
        .expect("chunked AOT runtime should be linked");
    let state = chain_state(dimension);
    let mut args = Vec::with_capacity(1 + parameter_names().len() + dimension);
    args.push(0.125);
    args.extend(parameter_values().iter().copied());
    args.extend(state.iter().copied());
    reportln!(
        "[LSODE2 AOT chunking break-even] dimension={dimension}; chunk_size={chunk}; repetitions={repetitions}; residual_chunks={}; jacobian_chunks={}",
        linked.residual_chunks.len(),
        linked.jacobian_value_chunks.len(),
    );
    reportln!(
        "policy | residual_ms/call | jacobian_ms/call | dispatches | parallel_dispatches | chunks | worker_callbacks | errors"
    );
    for (label, policy) in [
        ("Sequential", IvpLambdifyExecutionPolicy::Sequential),
        (
            "Parallel",
            IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        ),
        ("Auto", IvpLambdifyExecutionPolicy::Auto { min_work: 1 }),
    ] {
        let callback_telemetry = IvpTelemetry::detailed();
        let mut residual = vec![0.0; linked.residual_len];
        let mut jacobian = vec![0.0; linked.jacobian_output_len().expect("valid AOT layout")];
        let started = Instant::now();
        for _ in 0..repetitions {
            linked
                .try_residual_eval_with_policy(&args, &mut residual, policy, &callback_telemetry)
                .expect("chunked residual callback should succeed");
            black_box(&residual);
        }
        let residual_ms = started.elapsed().as_secs_f64() * 1.0e3 / repetitions as f64;
        let started = Instant::now();
        for _ in 0..repetitions {
            linked
                .try_jacobian_values_eval_with_policy(
                    &args,
                    &mut jacobian,
                    policy,
                    &callback_telemetry,
                )
                .expect("chunked Jacobian callback should succeed");
            black_box(&jacobian);
        }
        let jacobian_ms = started.elapsed().as_secs_f64() * 1.0e3 / repetitions as f64;
        let snapshot = callback_telemetry.snapshot();
        assert!(residual.iter().all(|value| value.is_finite()));
        assert!(jacobian.iter().all(|value| value.is_finite()));
        reportln!(
            "{label} | {residual_ms:.6} | {jacobian_ms:.6} | {} | {} | {} | {} | {}",
            snapshot.aot_chunk_dispatches,
            snapshot.aot_parallel_dispatches,
            snapshot.aot_chunks,
            snapshot.aot_worker_callbacks,
            snapshot.errors,
        );
    }
}

#[test]
#[ignore = "release-only large AOT warm full-solve stage matrix"]
fn lsode2_aot_large_warm_solver_stage_performance_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_performance_story_tests::lsode2_aot_large_warm_solver_stage_performance_matrix",
    );
    let dimensions = dimensions_from_env("LSODE2_AOT_SOLVE_DIMENSIONS", &[128, 256]);
    reportln!(
        "[LSODE2 AOT warm solver performance] dimensions={dimensions:?}; compiler=tcc; Dense excluded; preparation/build are reported separately from solve"
    );
    reportln!(
        "route | matrix | dimension | prepare_ms | solve_ms | summary_ms | total_ms | atom_residual_prepare_ms | atom_jacobian_prepare_ms | materialize_ms | build_ms | link_ms | residual_ms | jacobian_ms | factor_ms | rhs_ms | residual_calls | jacobian_calls | linear_solves | accepted | rejected | max_state_abs"
    );
    reportln!(
        "[LSODE2 AOT lifecycle scopes] cold scopes are inclusive; warm scopes are inclusive and must not be summed with their children"
    );
    reportln!(
        "route | matrix | dimension | solver_preparation_ms | bridge_preparation_ms | native_callback_preparation_ms | solve_scope_ms | summary_scope_ms"
    );
    reportln!(
        "[LSODE2 AOT solver overhead] inclusive stages are not additive; solve_minus_controller = solve - controller; controller_outside_iterations = controller - iteration_inclusive"
    );
    reportln!(
        "route | matrix | dimension | controller_ms | solve_minus_controller_ms | native_engine_setup_ms | native_result_assembly_ms | controller_outside_iterations_ms | predictor_ms | step_setup_ms | iteration_inclusive_ms | outcome_ms | stop_ms | method_policy_ms | method_switch_ms | residual_callback_ms | jacobian_callback_ms | residual_eval_ms | jacobian_eval_ms | residual_output_ms | jacobian_output_ms | factorization_ms | rhs_ms | controller_calls | iteration_calls | residual_callback_calls | jacobian_callback_calls"
    );
    for dimension in dimensions {
        for matrix in [ChainMatrixRoute::Sparse, ChainMatrixRoute::Banded] {
            run_full_solver_case(dimension, matrix, false);
            run_full_solver_case(dimension, matrix, true);
        }
    }
}
