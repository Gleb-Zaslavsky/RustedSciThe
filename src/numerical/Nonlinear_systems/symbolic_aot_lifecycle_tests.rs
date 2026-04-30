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

use crate::numerical::Nonlinear_systems::symbolic::{
    SymbolicDenseAotOptions, SymbolicNonlinearProblem, SymbolicProblemOptions,
};
use crate::numerical::Nonlinear_systems::symbolic_aot::materialize_symbolic_nonlinear_aot_build;
use crate::numerical::Nonlinear_systems::symbolic_backend::{
    SelectedSymbolicNonlinearBackendKind, SymbolicBackendSelectionPolicy,
    select_symbolic_nonlinear_backend,
};
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::{AotResolutionStatus, AotResolver};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
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
