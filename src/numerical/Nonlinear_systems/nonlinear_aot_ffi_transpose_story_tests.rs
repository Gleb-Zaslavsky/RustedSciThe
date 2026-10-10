//! Focused profiling story for the linked dense AOT Jacobian path.
//!
//! The full callback has several costs that should not be conflated:
//! crossing the generated-library boundary, writing a row-major buffer, and
//! adapting that buffer to nalgebra's column-major `DMatrix`. This ignored
//! release story measures those pieces separately.
//!
//! Run with:
//!
//! ```text
//! cargo test --release nonlinear_aot_ffi_callback_and_transpose_stage_story -- --ignored --nocapture --test-threads=1
//! cargo test --release nonlinear_aot_jacobian_layout_strategy_story -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(test)]

use crate::numerical::Nonlinear_systems::problem::JacobianProvider;
use crate::numerical::Nonlinear_systems::symbolic::{
    copy_row_major_jacobian_into_column_major, SymbolicDenseAotOptions, SymbolicNonlinearProblem,
    SymbolicProblemOptions,
};
use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::aot_solver_test_guard;
use crate::numerical::Nonlinear_systems::symbolic_backend::SymbolicBackendSelectionPolicy;
use crate::numerical::Nonlinear_systems::symbolic_generated::{
    SymbolicAotBuildPolicy, SymbolicGeneratedBackendConfig,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    resolve_linked_dense_backend, unregister_linked_dense_backend,
};
use crate::symbolic::codegen::codegen_runtime_api::DenseJacobianChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::hint::black_box;
use std::time::{Duration, Instant};

const DEFAULT_DIMENSION: usize = 128;
const DEFAULT_RUNS: usize = 20;

fn dimension() -> usize {
    std::env::var("NONLINEAR_AOT_FFI_DIMENSION")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value >= 8)
        .unwrap_or(DEFAULT_DIMENSION)
}

fn runs() -> usize {
    std::env::var("NONLINEAR_AOT_FFI_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value >= 3)
        .unwrap_or(DEFAULT_RUNS)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt())
}

fn diagonal_problem(dimension: usize) -> (Vec<Expr>, Vec<String>, DVector<f64>) {
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let equations = variables
        .iter()
        .map(|variable| Expr::parse_expression(&format!("{variable}^2-1.0")))
        .collect::<Vec<_>>();
    (equations, variables, DVector::from_element(dimension, 0.9))
}

#[test]
#[ignore = "release-oriented linked AOT callback versus transpose profiling story"]
fn nonlinear_aot_ffi_callback_and_transpose_stage_story() {
    let _guard = aot_solver_test_guard();
    let dimension = dimension();
    let repetitions = runs();
    let (equations, variables, point) = diagonal_problem(dimension);
    let output_dir = tempfile::tempdir().expect("AOT output directory should exist");

    let built = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
        equations,
        SymbolicProblemOptions::new().with_variables(variables),
        SymbolicGeneratedBackendConfig::new()
            .with_backend_policy_override(Some(SymbolicBackendSelectionPolicy::AotOnly))
            .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
            .with_aot_options(SymbolicDenseAotOptions::default())
            .with_output_parent_dir(Some(output_dir.path().to_path_buf())),
    )
    .expect("AOT profiling fixture should build");
    let problem_key = built
        .preparation_report
        .artifact_key
        .clone()
        .expect("AOT profiling fixture should publish a key");
    let linked = resolve_linked_dense_backend(&problem_key)
        .expect("AOT profiling fixture should register a linked backend");
    let prepared = built.into_prepared();
    let bound = prepared
        .bind_without_parameters()
        .expect("profiling fixture should bind without parameters");

    let args = point.as_slice();
    let mut callback_output = vec![0.0; dimension * dimension];
    (linked.jacobian_eval)(args, &mut callback_output);

    let mut expected = DMatrix::zeros(dimension, dimension);
    bound
        .jacobian_into(&point, &mut expected)
        .expect("full AOT Jacobian path should succeed");
    let mut max_callback_error: f64 = 0.0;
    for row in 0..dimension {
        for column in 0..dimension {
            max_callback_error = max_callback_error
                .max((callback_output[row * dimension + column] - expected[(row, column)]).abs());
        }
    }
    assert!(max_callback_error < 1e-12);

    let callback_samples = (0..repetitions)
        .map(|_| {
            let started = Instant::now();
            (linked.jacobian_eval)(args, &mut callback_output);
            duration_ms(started.elapsed())
        })
        .collect::<Vec<_>>();

    let mut transposed = DMatrix::zeros(dimension, dimension);
    let legacy_transpose_samples = (0..repetitions)
        .map(|_| {
            let started = Instant::now();
            for row in 0..dimension {
                for column in 0..dimension {
                    transposed[(row, column)] = callback_output[row * dimension + column];
                }
            }
            black_box(&transposed);
            duration_ms(started.elapsed())
        })
        .collect::<Vec<_>>();

    let tiled_transpose_samples = (0..repetitions)
        .map(|_| {
            let started = Instant::now();
            copy_row_major_jacobian_into_column_major(
                &callback_output,
                transposed.as_mut_slice(),
                dimension,
                dimension,
            );
            black_box(&transposed);
            duration_ms(started.elapsed())
        })
        .collect::<Vec<_>>();
    assert_eq!(transposed, expected);

    let full_samples = (0..repetitions)
        .map(|_| {
            let started = Instant::now();
            bound
                .jacobian_into(&point, &mut transposed)
                .expect("full AOT Jacobian path should succeed");
            duration_ms(started.elapsed())
        })
        .collect::<Vec<_>>();

    let (callback_mean, callback_std) = mean_std(&callback_samples);
    let (legacy_mean, legacy_std) = mean_std(&legacy_transpose_samples);
    let (tiled_mean, tiled_std) = mean_std(&tiled_transpose_samples);
    let (full_mean, full_std) = mean_std(&full_samples);
    println!(
        "[Nonlinear AOT FFI/transpose] dimension={dimension}; runs={repetitions}; build excluded"
    );
    println!("stage              | mean_ms | std_ms | interpretation");
    println!("--------------------------------------------------------");
    println!(
        "linked callback    | {:>7.3} | {:>7.3} | generated ABI into row-major buffer",
        callback_mean, callback_std
    );
    println!(
        "legacy copy        | {:>7.3} | {:>7.3} | scalar row/column indexing",
        legacy_mean, legacy_std
    );
    println!(
        "tiled copy         | {:>7.3} | {:>7.3} | cache-aware column-major adapter",
        tiled_mean, tiled_std
    );
    println!(
        "full jacobian_into | {:>7.3} | {:>7.3} | input + callback + check + copy",
        full_mean, full_std
    );
    unregister_linked_dense_backend(&problem_key);
}

#[derive(Debug)]
struct LayoutStrategyProfile {
    name: &'static str,
    callback_ms: f64,
    copy_ms: f64,
    full_ms: f64,
}

/// Compare generated Jacobian block layout with the caller-side adaptation cost.
///
/// Each strategy gets its own artifact namespace. The test deliberately keeps
/// the numerical problem and ABI fixed, so a timing difference is attributable
/// to generated block dispatch or the row-major-to-column-major adapter rather
/// than to solver iterations or symbolic preparation.
#[test]
#[ignore = "release-oriented AOT Jacobian layout and adapter profiling story"]
fn nonlinear_aot_jacobian_layout_strategy_story() {
    let _guard = aot_solver_test_guard();
    let dimension = std::env::var("NONLINEAR_AOT_LAYOUT_DIMENSION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(512);
    let repetitions = std::env::var("NONLINEAR_AOT_LAYOUT_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 3)
        .unwrap_or(10);
    let (equations, variables, point) = diagonal_problem(dimension);
    let strategies = [
        ("whole", DenseJacobianChunkingStrategy::Whole),
        (
            "rows-32",
            DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 32 },
        ),
        (
            "rows-64",
            DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 64 },
        ),
    ];

    let mut profiles = Vec::with_capacity(strategies.len());
    for (name, jacobian_strategy) in strategies {
        let output_dir = tempfile::tempdir().expect("AOT output directory should exist");
        let built = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
            equations.clone(),
            SymbolicProblemOptions::new().with_variables(variables.clone()),
            SymbolicGeneratedBackendConfig::new()
                .with_backend_policy_override(Some(SymbolicBackendSelectionPolicy::AotOnly))
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
                .with_aot_options(SymbolicDenseAotOptions {
                    jacobian_strategy,
                    ..SymbolicDenseAotOptions::default()
                })
                .with_output_parent_dir(Some(output_dir.path().to_path_buf())),
        )
        .expect("AOT layout fixture should build");
        let problem_key = built
            .preparation_report
            .artifact_key
            .clone()
            .expect("AOT layout fixture should publish a key");
        let linked = resolve_linked_dense_backend(&problem_key)
            .expect("AOT layout fixture should register a linked backend");
        let prepared = built.into_prepared();
        let bound = prepared
            .bind_without_parameters()
            .expect("AOT layout fixture should bind without parameters");

        let mut expected = DMatrix::zeros(dimension, dimension);
        bound
            .jacobian_into(&point, &mut expected)
            .expect("reference AOT Jacobian should succeed");
        let args = point.as_slice();
        let mut callback_output = vec![0.0; dimension * dimension];
        let mut adapted = DMatrix::zeros(dimension, dimension);
        (linked.jacobian_eval)(args, &mut callback_output);
        copy_row_major_jacobian_into_column_major(
            &callback_output,
            adapted.as_mut_slice(),
            dimension,
            dimension,
        );
        assert_eq!(adapted, expected, "strategy {name} changed Jacobian values");

        let callback_samples = (0..repetitions)
            .map(|_| {
                let started = Instant::now();
                (linked.jacobian_eval)(args, &mut callback_output);
                black_box(&callback_output);
                duration_ms(started.elapsed())
            })
            .collect::<Vec<_>>();
        let copy_samples = (0..repetitions)
            .map(|_| {
                let started = Instant::now();
                copy_row_major_jacobian_into_column_major(
                    &callback_output,
                    adapted.as_mut_slice(),
                    dimension,
                    dimension,
                );
                black_box(&adapted);
                duration_ms(started.elapsed())
            })
            .collect::<Vec<_>>();
        let full_samples = (0..repetitions)
            .map(|_| {
                let started = Instant::now();
                bound
                    .jacobian_into(&point, &mut adapted)
                    .expect("full AOT Jacobian path should succeed");
                black_box(&adapted);
                duration_ms(started.elapsed())
            })
            .collect::<Vec<_>>();
        let (callback_ms, _) = mean_std(&callback_samples);
        let (copy_ms, _) = mean_std(&copy_samples);
        let (full_ms, _) = mean_std(&full_samples);
        profiles.push(LayoutStrategyProfile {
            name,
            callback_ms,
            copy_ms,
            full_ms,
        });
        assert!(unregister_linked_dense_backend(&problem_key).is_some());
    }

    println!(
        "[Nonlinear AOT Jacobian layout] dimension={dimension}; runs={repetitions}; build excluded"
    );
    println!("strategy | callback_ms | copy_ms | full_jacobian_ms");
    println!("-----------------------------------------------------");
    for profile in profiles {
        println!(
            "{:<8} | {:>11.3} | {:>7.3} | {:>16.3}",
            profile.name, profile.callback_ms, profile.copy_ms, profile.full_ms
        );
    }
}
