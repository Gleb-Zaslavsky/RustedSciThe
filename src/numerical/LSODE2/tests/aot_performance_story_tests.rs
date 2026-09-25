//! Release-oriented LSODE2 AOT performance stories.
//!
//! These are deliberately separate from parity tests.  They measure cold
//! preparation/build/link, warm callback throughput, and chunk-policy
//! break-even while retaining enough integer counters to prove that the
//! compared routes did the same amount of numerical work.  They are ignored
//! by default because a single matrix may invoke several external toolchains.

use super::native_jacobian::{
    NativeAtomJacobianRuntime, NativeJacobianStorage, try_prepare_native_atomview_jacobian_runtime,
};
use super::story_support::{ChainMatrixRoute, chain_equations, chain_solver_config, chain_state};
use super::{IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry};
use super::{
    IvpWarmStage, Lsode2AotProfile, Lsode2AotToolchain, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, PreparedSymbolicIvpResidualProblem, SymbolicIvpProblemOptions,
    prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::{
    SelectedSymbolicIvpBackendKind, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_banded_backend, prepare_generated_symbolic_ivp_sparse_backend,
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

fn cold_ms(snapshot: &super::IvpTelemetrySnapshot, stage: IvpColdStage) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1.0e3
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
            generated_config(output.path(), compiler, residual_chunking, sparse_chunking),
        )
        .expect("large sparse AOT performance preparation should succeed"),
        "Banded" => prepare_generated_symbolic_ivp_banded_backend(
            equations,
            variables,
            "t".to_string(),
            (1, 1),
            options(IvpSymbolicAssemblyBackend::AtomView, telemetry.clone()),
            generated_config(output.path(), compiler, residual_chunking, sparse_chunking),
        )
        .expect("large banded AOT performance preparation should succeed"),
        other => panic!("unknown AOT matrix {other}"),
    };
    assert_eq!(
        prepared.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
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
    reportln!(
        "AOT | {compiler} | {matrix} | {:?} | {dimension} | {prepare_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {}",
        frontend,
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
    );
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
        "Lambdify | - | {matrix} | AtomViewNative | {dimension} | {prepare_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {}",
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
    let summary = solver
        .solve_with_summary()
        .expect("large AOT performance solver should solve");
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1.0e3;
    let snapshot = telemetry.snapshot();
    let final_state = summary
        .final_y
        .as_ref()
        .expect("large AOT performance solve should expose final state");
    assert!(final_state.iter().all(|value| value.is_finite()));
    reportln!(
        "{route} | {} | {dimension} | {prepare_ms:.3} | {solve_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {:.3}",
        matrix.label(),
        total_started.elapsed().as_secs_f64() * 1.0e3,
        cold_ms(&snapshot, IvpColdStage::AtomPreparation),
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
        "[LSODE2 AOT callback performance] dimensions={dimensions:?}; compiler=tcc; repetitions={repetitions}; Dense excluded; build/link included only in prepare_ms"
    );
    reportln!(
        "route | compiler | matrix | frontend | dimension | prepare_ms | residual_ms/call | jacobian_ms/call | expr_to_atom_ms | diff_ms | simplify_ms | pattern_ms | materialize_ms | build_ms | link_ms | build_attempts | link_attempts | callback_calls | residual_len | jacobian_output_len"
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
        "[LSODE2 AOT toolchain callback performance] dimensions={dimensions:?}; repetitions={repetitions}; Sparse/Banded AtomViewNative; unavailable external toolchains are skipped"
    );
    reportln!(
        "route | compiler | matrix | frontend | dimension | prepare_ms | residual_ms/call | jacobian_ms/call | expr_to_atom_ms | diff_ms | simplify_ms | pattern_ms | materialize_ms | build_ms | link_ms | build_attempts | link_attempts | callback_calls | residual_len | jacobian_output_len"
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
        "route | matrix | dimension | prepare_ms | solve_ms | total_ms | atom_prepare_ms | materialize_ms | build_ms | link_ms | residual_ms | jacobian_ms | factor_ms | rhs_ms | residual_calls | jacobian_calls | linear_solves | accepted | rejected | max_state_abs"
    );
    for dimension in dimensions {
        for matrix in [ChainMatrixRoute::Sparse, ChainMatrixRoute::Banded] {
            run_full_solver_case(dimension, matrix, false);
            run_full_solver_case(dimension, matrix, true);
        }
    }
}
