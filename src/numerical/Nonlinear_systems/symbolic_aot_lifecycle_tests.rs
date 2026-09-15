#![cfg(test)]

//! End-to-end dense nonlinear AOT lifecycle tests.
//!
//! These tests mirror the lifecycle proof we already have for sparse BVP:
//! 1. construct a symbolic nonlinear problem,
//! 2. prepare its dense AOT bridge,
//! 3. generate and materialize a standalone crate,
//! 4. compile it in release mode,
//! 5. register and resolve the resulting artifact,
//! 6. and statically link it into a tiny consumer crate that calls the
//!    generated residual/Jacobian functions like ordinary Rust code.

use crate::numerical::Nonlinear_systems::engine::{NewtonMethod, SolveOptions, SolverEngine};
use crate::numerical::Nonlinear_systems::error::TerminationReason;
use crate::numerical::Nonlinear_systems::symbolic::{
    PreparedSymbolicNonlinearProblem, SymbolicDenseAotOptions, SymbolicNonlinearProblem,
    SymbolicProblemOptions,
};
use crate::numerical::Nonlinear_systems::symbolic_aot::materialize_symbolic_nonlinear_aot_build;
use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::aot_solver_test_guard;
use crate::numerical::Nonlinear_systems::symbolic_backend::{
    SelectedSymbolicNonlinearBackendKind, SymbolicBackendSelectionPolicy,
    select_symbolic_nonlinear_backend,
};
use crate::numerical::Nonlinear_systems::symbolic_generated::SymbolicGeneratedBackendConfig;
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::{AotResolutionStatus, AotResolver};
use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use approx::assert_relative_eq;
use nalgebra::DVector;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use tempfile::tempdir;

fn elementary_problem() -> SymbolicNonlinearProblem {
    SymbolicNonlinearProblem::from_expressions_with_options(
        vec![
            Expr::parse_expression("x^2+y^2-10"),
            Expr::parse_expression("x-y-4"),
        ],
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_lambdify_backend(),
    )
    .expect("problem should build")
}

fn parameterized_problem() -> SymbolicNonlinearProblem {
    let symbolic = Expr::Symbols("x, y, a");
    let x = symbolic[0].clone();
    let y = symbolic[1].clone();
    let a = symbolic[2].clone();

    SymbolicNonlinearProblem::from_expressions_with_options(
        vec![a.clone() * x.clone() + y.clone() - Expr::Const(3.0), x - y],
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()]),
    )
    .expect("parameterized problem should build")
}

struct WarmSolveSample {
    x: DVector<f64>,
    total_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    counters: (usize, usize, usize, usize),
}

fn collect_warm_solve_samples<P: crate::numerical::Nonlinear_systems::problem::JacobianProvider>(
    problem: &P,
    options: &SolveOptions,
    runs: usize,
) -> Vec<WarmSolveSample> {
    (0..runs)
        .map(|_| {
            let result = SolverEngine::new(NewtonMethod, options.clone())
                .solve(problem, DVector::from_vec(vec![0.5, 0.5]))
                .expect("warm nonlinear solve should succeed");
            assert_eq!(result.termination, TerminationReason::Converged);
            assert!(result.residual_norm < options.tolerance);
            let statistics = result.statistics;
            WarmSolveSample {
                x: result.x,
                total_ms: duration_ms(statistics.total_duration),
                residual_ms: duration_ms(statistics.residual_duration),
                jacobian_ms: duration_ms(statistics.jacobian_duration),
                linear_ms: duration_ms(statistics.linear_solve_duration),
                counters: (
                    statistics.residual_evaluations,
                    statistics.jacobian_evaluations,
                    statistics.linear_solves,
                    statistics.iterations,
                ),
            }
        })
        .collect()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn mean_duration(samples: &[WarmSolveSample], select: impl Fn(&WarmSolveSample) -> f64) -> f64 {
    samples.iter().map(select).sum::<f64>() / samples.len() as f64
}

fn cargo_program() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

fn run_checked(command: &mut Command, context: &str) {
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("{context}: failed to start process: {err}"));
    assert!(
        output.status.success(),
        "{context} failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn eval_dense_expected(
    problem: &SymbolicNonlinearProblem,
    args: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<String>) {
    let prepared = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
    let input_names = prepared
        .flattened_input_names()
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    let borrowed_names = prepared.flattened_input_names();

    let residual = problem
        .equations()
        .iter()
        .map(|expr| expr.lambdify_borrowed_thread_safe(borrowed_names)(args))
        .collect::<Vec<_>>();
    let jacobian = problem
        .symbolic_jacobian()
        .iter()
        .flat_map(|row| {
            row.iter()
                .map(|expr| expr.lambdify_borrowed_thread_safe(borrowed_names)(args))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    (residual, jacobian, input_names)
}

fn write_dense_consumer_crate(
    parent_dir: &Path,
    dependency_crate_name: &str,
    module_name: &str,
    args: &[f64],
    expected_residual: &[f64],
    expected_jacobian: &[f64],
) -> std::io::Result<PathBuf> {
    let crate_dir = parent_dir.join("nonlinear_dense_aot_consumer");
    let src_dir = crate_dir.join("src");
    let tests_dir = crate_dir.join("tests");
    fs::create_dir_all(&src_dir)?;
    fs::create_dir_all(&tests_dir)?;

    fs::write(
        crate_dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"nonlinear_dense_aot_consumer\"\nversion = \"0.1.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\n{dep} = {{ path = \"../{dep}\" }}\n",
            dep = dependency_crate_name
        ),
    )?;
    fs::write(
        src_dir.join("lib.rs"),
        "//! Dense nonlinear AOT lifecycle consumer.\n",
    )?;

    fs::write(
        tests_dir.join("full_cycle.rs"),
        format!(
            "use {dep}::generated::{module}::eval_nonlinear_jacobian;\n\
             use {dep}::generated::{module}::eval_nonlinear_residual;\n\n\
             fn assert_close(actual: &[f64], expected: &[f64], label: &str) {{\n\
                 assert_eq!(actual.len(), expected.len(), \"{{label}} length mismatch\");\n\
                 for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {{\n\
                     let diff = (a - e).abs();\n\
                     assert!(diff <= 1e-12, \"{{label}}[{{index}}] diff {{diff}} too large: actual={{a}} expected={{e}}\");\n\
                 }}\n\
             }}\n\n\
             #[test]\n\
             fn compiled_generated_dense_backend_matches_expected_values() {{\n\
                 let args = {args:?};\n\
                 let mut residual = vec![0.0_f64; {residual_len}];\n\
                 let mut jacobian = vec![0.0_f64; {jacobian_len}];\n\
                 eval_nonlinear_residual(&args, &mut residual);\n\
                 eval_nonlinear_jacobian(&args, &mut jacobian);\n\
                 assert_close(&residual, &{expected_residual:?}, \"residual\");\n\
                 assert_close(&jacobian, &{expected_jacobian:?}, \"jacobian\");\n\
             }}\n",
            dep = dependency_crate_name,
            module = module_name,
            args = args,
            residual_len = expected_residual.len(),
            jacobian_len = expected_jacobian.len(),
            expected_residual = expected_residual,
            expected_jacobian = expected_jacobian,
        ),
    )?;

    Ok(crate_dir)
}

#[test]
#[ignore = "full lifecycle test that generates, builds, resolves, and statically links a dense nonlinear AOT crate"]
fn dense_nonlinear_aot_full_cycle_builds_resolves_and_links_successfully() {
    let problem = elementary_problem();
    let prepared_bridge = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
    let manifest = prepared_bridge.manifest();
    let (expected_residual, expected_jacobian, _) = eval_dense_expected(&problem, &[3.0, -1.0]);
    let temp = tempdir().expect("tempdir should exist");

    let build = materialize_symbolic_nonlinear_aot_build(
        "generated_nonlinear_dense_full_cycle_fixture",
        "generated_nonlinear_dense_full_cycle_module",
        &problem,
        SymbolicDenseAotOptions::default(),
        temp.path(),
        AotBuildProfile::Release,
    )
    .expect("build request should materialize");

    let mut cargo = Command::new(cargo_program());
    cargo
        .args(&build.cargo_args)
        .current_dir(build.cargo_workdir());
    run_checked(
        &mut cargo,
        "release build for dense nonlinear generated crate",
    );

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let selected = select_symbolic_nonlinear_backend(
        &problem,
        SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
        SymbolicDenseAotOptions::default(),
    );
    assert_eq!(
        selected.effective_backend,
        SelectedSymbolicNonlinearBackendKind::AotCompiled
    );
    let resolved = selected
        .aot_resolution
        .as_ref()
        .expect("compiled selection should include resolution metadata");
    assert_eq!(resolved.status, AotResolutionStatus::Compiled);
    assert!(resolved.registered.expected_rlib.exists());

    let consumer_dir = write_dense_consumer_crate(
        temp.path(),
        "generated_nonlinear_dense_full_cycle_fixture",
        "generated_nonlinear_dense_full_cycle_module",
        &[3.0, -1.0],
        &expected_residual,
        &expected_jacobian,
    )
    .expect("consumer crate should be writable");

    let mut cargo = Command::new(cargo_program());
    cargo
        .arg("test")
        .arg("--release")
        .arg("--")
        .arg("--nocapture")
        .current_dir(&consumer_dir);
    run_checked(
        &mut cargo,
        "consumer crate test with statically linked dense nonlinear generated AOT backend",
    );
}

#[test]
#[ignore = "full lifecycle test for parameterized dense nonlinear AOT input ordering"]
fn parameterized_dense_nonlinear_aot_full_cycle_preserves_parameter_first_input_order() {
    let problem = parameterized_problem();
    let prepared_bridge = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
    assert_eq!(prepared_bridge.flattened_input_names(), &["a", "x", "y"]);
    let manifest = prepared_bridge.manifest();
    let (expected_residual, expected_jacobian, input_names) =
        eval_dense_expected(&problem, &[2.0, 1.0, 1.0]);
    let temp = tempdir().expect("tempdir should exist");

    let build = materialize_symbolic_nonlinear_aot_build(
        "generated_parameterized_nonlinear_dense_fixture",
        "generated_parameterized_nonlinear_dense_module",
        &problem,
        SymbolicDenseAotOptions::default(),
        temp.path(),
        AotBuildProfile::Release,
    )
    .expect("build request should materialize");

    let mut cargo = Command::new(cargo_program());
    cargo
        .args(&build.cargo_args)
        .current_dir(build.cargo_workdir());
    run_checked(
        &mut cargo,
        "release build for parameterized dense nonlinear generated crate",
    );

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let selected = select_symbolic_nonlinear_backend(
        &problem,
        SymbolicBackendSelectionPolicy::AotOnly,
        Some(&resolver),
        SymbolicDenseAotOptions::default(),
    );
    assert_eq!(
        selected.effective_backend,
        SelectedSymbolicNonlinearBackendKind::AotCompiled
    );
    assert_eq!(input_names, vec!["a", "x", "y"]);

    let consumer_dir = write_dense_consumer_crate(
        temp.path(),
        "generated_parameterized_nonlinear_dense_fixture",
        "generated_parameterized_nonlinear_dense_module",
        &[2.0, 1.0, 1.0],
        &expected_residual,
        &expected_jacobian,
    )
    .expect("consumer crate should be writable");

    let mut cargo = Command::new(cargo_program());
    cargo
        .arg("test")
        .arg("--release")
        .arg("--")
        .arg("--nocapture")
        .current_dir(&consumer_dir);
    run_checked(
        &mut cargo,
        "consumer crate test with statically linked parameterized dense nonlinear generated AOT backend",
    );
}

#[test]
#[ignore = "builds, dynamically loads, and solves through a real parameterized dense AOT cdylib"]
fn parameterized_dense_nonlinear_aot_dynamic_load_and_solver_cycle() {
    let _guard = aot_solver_test_guard();
    let source_problem = parameterized_problem();
    let equations = source_problem.equations().to_vec();
    let options = SymbolicProblemOptions::new()
        .with_variables(vec!["x".to_string(), "y".to_string()])
        .with_equation_parameters(vec!["a".to_string()]);
    let temp = tempdir().expect("tempdir should exist");

    let cold = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
        equations.clone(),
        options.clone(),
        SymbolicGeneratedBackendConfig::build_if_missing_release(temp.path()),
    )
    .expect("BuildIfMissing should materialize and compile the AOT artifact");
    assert_eq!(
        cold.selected_backend,
        SelectedSymbolicNonlinearBackendKind::AotCompiled,
        "BuildIfMissing should load and select the generated runtime"
    );
    assert!(cold.build_result.is_some());
    let resolver = cold
        .updated_resolver
        .clone()
        .expect("cold build should return an updated resolver");
    let problem_key = resolver
        .registry()
        .problem_keys()
        .into_iter()
        .next()
        .expect("resolver should contain the generated parameterized artifact");
    let artifact = resolver
        .registry()
        .get_by_problem_key(&problem_key)
        .expect("resolver should expose the generated artifact metadata")
        .clone();
    assert!(artifact.expected_cdylib.exists());

    let warm = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
        equations,
        options,
        SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
    )
    .expect("RequirePrebuilt should select the loaded generated backend");
    assert_eq!(
        warm.selected_backend,
        SelectedSymbolicNonlinearBackendKind::AotCompiled
    );
    assert!(warm.build_result.is_none());

    let prepared = warm.into_prepared();
    let bound = prepared
        .bind_values(DVector::from_vec(vec![2.0]))
        .expect("parameter binding should use the generated ABI order");
    let result = SolverEngine::new(NewtonMethod, SolveOptions::default())
        .solve(&bound, DVector::from_vec(vec![0.5, 0.5]))
        .expect("Newton should solve through the dynamically loaded AOT backend");

    assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-10);
    assert!(result.residual_norm < 1e-10);
    assert!(unregister_linked_dense_backend(&problem_key).is_some());
}

#[test]
#[ignore = "release-oriented warm AOT versus Lambdify stage story"]
fn parameterized_dense_aot_vs_lambdify_warm_stage_story() {
    let _guard = aot_solver_test_guard();
    let runs = std::env::var("NONLINEAR_AOT_STAGE_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|runs| *runs >= 3)
        .unwrap_or(5);
    let source_problem = parameterized_problem();
    let equations = source_problem.equations().to_vec();
    let options = SymbolicProblemOptions::new()
        .with_variables(vec!["x".to_string(), "y".to_string()])
        .with_equation_parameters(vec!["a".to_string()]);
    let temp = tempdir().expect("temporary AOT directory should exist");

    // Cold preparation/build is deliberately outside the warm numerical sample.
    let cold = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
        equations.clone(),
        options.clone(),
        SymbolicGeneratedBackendConfig::build_if_missing_release(temp.path()),
    )
    .expect("BuildIfMissing should produce a usable AOT backend");
    assert_eq!(
        cold.selected_backend,
        SelectedSymbolicNonlinearBackendKind::AotCompiled
    );
    assert!(cold.build_result.is_some());
    let resolver = cold
        .updated_resolver
        .clone()
        .expect("cold AOT preparation should return a resolver");
    let problem_key = resolver
        .registry()
        .problem_keys()
        .into_iter()
        .next()
        .expect("AOT resolver should contain the generated problem");

    let warm = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
        equations,
        options,
        SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
    )
    .expect("RequirePrebuilt should reuse the linked AOT backend");
    assert_eq!(
        warm.selected_backend,
        SelectedSymbolicNonlinearBackendKind::AotCompiled
    );
    assert!(warm.build_result.is_none());
    let aot_prepared = warm.into_prepared();
    let aot_bound = aot_prepared
        .bind_values(DVector::from_vec(vec![2.0]))
        .expect("AOT parameter binding should succeed");

    let lambdify_prepared = PreparedSymbolicNonlinearProblem::from_problem(source_problem);
    let lambdify_bound = lambdify_prepared
        .bind_values(DVector::from_vec(vec![2.0]))
        .expect("Lambdify parameter binding should succeed");
    let solve_options = SolveOptions {
        tolerance: 1e-10,
        max_iterations: 40,
        diagnostics: crate::numerical::Nonlinear_systems::engine::DiagnosticsOptions {
            collect_history: false,
            collect_statistics: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let aot_samples = collect_warm_solve_samples(&aot_bound, &solve_options, runs);
    let lambdify_samples = collect_warm_solve_samples(&lambdify_bound, &solve_options, runs);
    assert_eq!(aot_samples.len(), lambdify_samples.len());
    assert!(
        aot_samples
            .iter()
            .zip(&lambdify_samples)
            .all(|(aot, lambdify)| aot.counters == lambdify.counters)
    );
    let baseline = &lambdify_samples[0].x;
    let max_diff = aot_samples
        .iter()
        .chain(&lambdify_samples)
        .map(|sample| (&sample.x - baseline).norm())
        .fold(0.0, f64::max);
    assert!(max_diff < 1e-9, "warm route solution delta={max_diff:e}");

    println!("[Nonlinear AOT/Lambdify warm stages] runs={runs}; preparation/build excluded");
    println!(
        "route | total_ms | residual_ms | jacobian_ms | linear_ms | counters R/J/L/I | max_diff"
    );
    for (route, samples) in [
        ("Lambdify", &lambdify_samples),
        ("AOT RequirePrebuilt", &aot_samples),
    ] {
        println!(
            "{route:<18} | {:>8.3} | {:>11.3} | {:>11.3} | {:>9.3} | {:>4}/{:<3}/{:<3}/{:<3} | {:>8.1e}",
            mean_duration(samples, |sample| sample.total_ms),
            mean_duration(samples, |sample| sample.residual_ms),
            mean_duration(samples, |sample| sample.jacobian_ms),
            mean_duration(samples, |sample| sample.linear_ms),
            samples[0].counters.0,
            samples[0].counters.1,
            samples[0].counters.2,
            samples[0].counters.3,
            max_diff,
        );
    }

    assert!(unregister_linked_dense_backend(&problem_key).is_some());
}
