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
//! ```

#![cfg(test)]

use crate::numerical::Nonlinear_systems::problem::JacobianProvider;
use crate::numerical::Nonlinear_systems::symbolic::{
    SymbolicDenseAotOptions, SymbolicNonlinearProblem, SymbolicProblemOptions,
    copy_row_major_jacobian_into_column_major,
};
use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::aot_solver_test_guard;
use crate::numerical::Nonlinear_systems::symbolic_backend::SymbolicBackendSelectionPolicy;
use crate::numerical::Nonlinear_systems::symbolic_generated::{
    SymbolicAotBuildPolicy, SymbolicGeneratedBackendConfig,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    resolve_linked_dense_backend, unregister_linked_dense_backend,
};
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
