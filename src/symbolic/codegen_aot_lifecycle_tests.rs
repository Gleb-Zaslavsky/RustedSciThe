#![cfg(test)]

//! End-to-end AOT lifecycle tests.
//!
//! These tests exercise the full path for a small generated backend:
//! 1. build a symbolic prepared problem,
//! 2. generate a standalone AOT crate,
//! 3. materialize it to disk,
//! 4. compile it in release mode,
//! 5. register and resolve the compiled artifact,
//! 6. and finally link it into a tiny consumer crate that calls the generated
//!    residual/Jacobian functions as ordinary Rust code.
//!
//! The tests are intentionally `#[ignore]` because they spawn nested Cargo
//! builds and are meant as lifecycle proofs rather than quick unit checks.

use crate::symbolic::codegen_aot_build::{AotBuildProfile, AotBuildRequest, AotBuildResult};
use crate::symbolic::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
use crate::symbolic::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen_aot_resolution::{AotResolutionStatus, AotResolver};
use crate::symbolic::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind, select_backend,
};
use crate::symbolic::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedProblem, PreparedSparseProblem,
};
use crate::symbolic::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen_runtime_api::{ResidualRuntimePlan, SparseJacobianRuntimePlan};
use crate::symbolic::codegen_tasks::{
    ResidualTask, SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{BvpPreparedSparseAotProblem, Jacobian};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

#[derive(Debug, Clone)]
struct ConsumerChunkSpec {
    fn_name: String,
    offset: usize,
    len: usize,
}

fn sample_sparse_prepared_problem() -> PreparedProblem<'static> {
    let residuals = Box::leak(Box::new(vec![
        Expr::parse_expression("p + x"),
        Expr::parse_expression("y*z - p"),
        Expr::parse_expression("z - x"),
    ]));
    let vars = Box::leak(Box::new(vec!["x", "y", "z"]));
    let params = Box::leak(Box::new(vec!["p"]));
    let e0 = Box::leak(Box::new(Expr::parse_expression("1")));
    let e1 = Box::leak(Box::new(Expr::parse_expression("z")));
    let e2 = Box::leak(Box::new(Expr::parse_expression("y")));
    let e3 = Box::leak(Box::new(Expr::parse_expression("-1")));
    let e4 = Box::leak(Box::new(Expr::parse_expression("1")));
    let entries = Box::leak(Box::new(vec![
        SparseExprEntry {
            row: 0,
            col: 0,
            expr: e0,
        },
        SparseExprEntry {
            row: 1,
            col: 1,
            expr: e1,
        },
        SparseExprEntry {
            row: 1,
            col: 2,
            expr: e2,
        },
        SparseExprEntry {
            row: 2,
            col: 0,
            expr: e3,
        },
        SparseExprEntry {
            row: 2,
            col: 2,
            expr: e4,
        },
    ]));

    PreparedProblem::sparse(PreparedSparseProblem::new(
        BackendKind::Aot,
        MatrixBackend::SparseCol,
        ResidualTask {
            fn_name: "eval_residual",
            residuals,
            variables: vars,
            params: Some(params),
        }
        .runtime_plan(ResidualChunkingStrategy::Whole),
        SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (3, 3),
            entries,
            variables: vars,
            params: Some(params),
        }
        .runtime_plan(SparseChunkingStrategy::Whole),
    ))
}

fn build_real_bvp_damp1_case(n_steps: usize) -> Jacobian {
    let eq1 = Expr::parse_expression("y-z");
    let eq2 = Expr::parse_expression("-z^3");
    let eq_system = vec![eq1, eq2];
    let values = vec!["z".to_string(), "y".to_string()];

    let mut border_conditions = HashMap::new();
    border_conditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
    border_conditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

    let mut jac = Jacobian::new();
    jac.discretization_system_BVP_par(
        eq_system,
        values,
        "x".to_string(),
        0.0,
        Some(n_steps),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
    );
    jac.calc_jacobian_parallel_smart_optimized();
    jac
}

fn real_bvp_sparse_bridge_problem(
    n_steps: usize,
) -> (BvpPreparedSparseAotProblem, Vec<f64>, Vec<f64>, Vec<f64>) {
    let jac = build_real_bvp_damp1_case(n_steps);
    let bridge = jac.prepare_sparse_aot_problem(
        "eval_bvp_residual",
        "eval_bvp_sparse_values",
        ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 8,
        },
        SparseChunkingStrategy::ByRowCount { rows_per_chunk: 1 },
    );

    let args: Vec<f64> = (0..bridge.variable_names.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();
    let input_names = bridge
        .variable_names
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    let expected_residual = bridge
        .residuals
        .iter()
        .map(|expr| expr.lambdify_borrowed_thread_safe(&input_names)(&args))
        .collect();
    let expected_values = bridge
        .sparse_entries
        .iter()
        .map(|(_, _, expr)| expr.lambdify_borrowed_thread_safe(&input_names)(&args))
        .collect();

    (bridge, args, expected_residual, expected_values)
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

fn build_generated_crate_release(
    crate_name: &str,
    module_name: &str,
    prepared: &PreparedProblem<'_>,
    output_parent_dir: &Path,
) -> AotBuildResult {
    let crate_spec = generated_aot_crate_from_prepared_problem(crate_name, module_name, prepared);
    let request = AotBuildRequest::new(crate_spec, output_parent_dir, AotBuildProfile::Release);
    let result = request
        .materialize()
        .expect("generated AOT crate should materialize");

    let mut cargo = Command::new(cargo_program());
    cargo
        .args(&result.cargo_args)
        .current_dir(result.cargo_workdir());
    run_checked(&mut cargo, "release build for generated crate");

    result
}

fn residual_chunk_specs(plan: &ResidualRuntimePlan<'_>) -> Vec<ConsumerChunkSpec> {
    plan.chunks
        .iter()
        .map(|chunk| ConsumerChunkSpec {
            fn_name: chunk.fn_name.clone(),
            offset: chunk.output_offset,
            len: chunk.residuals.len(),
        })
        .collect()
}

fn sparse_chunk_specs(plan: &SparseJacobianRuntimePlan<'_>) -> Vec<ConsumerChunkSpec> {
    plan.chunks
        .iter()
        .map(|chunk| ConsumerChunkSpec {
            fn_name: chunk.fn_name.clone(),
            offset: chunk.value_offset,
            len: chunk.entries.len(),
        })
        .collect()
}

fn write_consumer_crate(
    parent_dir: &Path,
    dependency_crate_name: &str,
    module_name: &str,
    residual_chunks: &[ConsumerChunkSpec],
    jacobian_chunks: &[ConsumerChunkSpec],
    args: &[f64],
    expected_residual: &[f64],
    expected_values: &[f64],
) -> std::io::Result<std::path::PathBuf> {
    let crate_dir = parent_dir.join("aot_consumer_fixture");
    let src_dir = crate_dir.join("src");
    let tests_dir = crate_dir.join("tests");
    fs::create_dir_all(&src_dir)?;
    fs::create_dir_all(&tests_dir)?;

    fs::write(
        crate_dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"aot_consumer_fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\n{dep} = {{ path = \"../{dep}\" }}\n",
            dep = dependency_crate_name
        ),
    )?;

    fs::write(
        src_dir.join("lib.rs"),
        "//! Consumer fixture crate for generated AOT lifecycle tests.\n",
    )?;

    fs::write(
        tests_dir.join("full_cycle.rs"),
        format!(
            "{residual_calls}\n\
             {jacobian_calls}\n\n\
             #[test]\n\
             fn compiled_generated_backend_matches_expected_values() {{\n\
                 let args = {args:?};\n\
                 let mut residual = vec![0.0_f64; {residual_len}];\n\
                 let mut values = vec![0.0_f64; {values_len}];\n\
                 {residual_body}\
                 {jacobian_body}\
                 assert_eq!(residual, {expected_residual:?});\n\
                 assert_eq!(values, {expected_values:?});\n\
             }}\n",
            residual_calls = residual_chunks
                .iter()
                .map(|chunk| format!(
                    "use {dep}::generated::{module}::{fn_name};",
                    dep = dependency_crate_name,
                    module = module_name,
                    fn_name = chunk.fn_name
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            jacobian_calls = jacobian_chunks
                .iter()
                .map(|chunk| format!(
                    "use {dep}::generated::{module}::{fn_name};",
                    dep = dependency_crate_name,
                    module = module_name,
                    fn_name = chunk.fn_name
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            residual_body = residual_chunks
                .iter()
                .map(|chunk| format!(
                    "    {fn_name}(&args, &mut residual[{start}..{end}]);\n",
                    fn_name = chunk.fn_name,
                    start = chunk.offset,
                    end = chunk.offset + chunk.len
                ))
                .collect::<String>(),
            jacobian_body = jacobian_chunks
                .iter()
                .map(|chunk| format!(
                    "    {fn_name}(&args, &mut values[{start}..{end}]);\n",
                    fn_name = chunk.fn_name,
                    start = chunk.offset,
                    end = chunk.offset + chunk.len
                ))
                .collect::<String>(),
            args = args,
            residual_len = expected_residual.len(),
            values_len = expected_values.len(),
            expected_residual = expected_residual,
            expected_values = expected_values,
        ),
    )?;

    Ok(crate_dir)
}

#[test]
#[ignore = "full lifecycle test that generates, builds, resolves, and statically links a temporary AOT crate"]
fn sparse_aot_full_cycle_builds_resolves_and_links_successfully() {
    let prepared = sample_sparse_prepared_problem();
    let manifest = PreparedProblemManifest::from(&prepared);
    let temp = tempdir().expect("tempdir should exist");
    let (residual_chunk_specs, jacobian_chunk_specs) = match &prepared {
        PreparedProblem::Sparse(problem) => (
            residual_chunk_specs(&problem.residual_plan),
            sparse_chunk_specs(&problem.jacobian_plan),
        ),
        PreparedProblem::Dense(_) => unreachable!("sample sparse prepared problem must be sparse"),
    };

    let build = build_generated_crate_release(
        "generated_sparse_full_cycle_fixture",
        "generated_sparse_full_cycle_module",
        &prepared,
        temp.path(),
    );

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest.clone(), &build);
    let resolver = AotResolver::new(registry);

    let resolved = resolver.resolve_prepared_problem(&prepared);
    assert_eq!(resolved.status, AotResolutionStatus::Compiled);
    assert!(resolved.registered.expected_rlib.exists());

    let selected = select_backend(
        &prepared,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );
    assert_eq!(selected.effective_backend, SelectedBackendKind::AotCompiled);
    assert!(selected.is_compiled_aot());

    let consumer_dir = write_consumer_crate(
        temp.path(),
        "generated_sparse_full_cycle_fixture",
        "generated_sparse_full_cycle_module",
        &residual_chunk_specs,
        &jacobian_chunk_specs,
        &[2.0, 3.0, 4.0, 5.0],
        &[5.0, 18.0, 2.0],
        &[1.0, 5.0, 4.0, -1.0, 1.0],
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
        "consumer crate test with statically linked generated AOT backend",
    );
}

#[test]
#[ignore = "full lifecycle test for a real BVP-derived sparse prepared problem"]
fn bvp_sparse_aot_full_cycle_builds_resolves_and_links_successfully() {
    let (bridge, args, expected_residual, expected_values) = real_bvp_sparse_bridge_problem(8);
    let prepared = PreparedProblem::sparse(bridge.as_prepared_problem());
    let manifest = PreparedProblemManifest::from(&prepared);
    let temp = tempdir().expect("tempdir should exist");
    let (residual_chunk_specs, jacobian_chunk_specs) = match &prepared {
        PreparedProblem::Sparse(problem) => (
            residual_chunk_specs(&problem.residual_plan),
            sparse_chunk_specs(&problem.jacobian_plan),
        ),
        PreparedProblem::Dense(_) => unreachable!("BVP prepared problem must be sparse"),
    };

    let build = build_generated_crate_release(
        "generated_bvp_sparse_full_cycle_fixture",
        "generated_bvp_sparse_full_cycle_module",
        &prepared,
        temp.path(),
    );

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest.clone(), &build);
    let resolver = AotResolver::new(registry);

    let resolved = resolver.resolve_prepared_problem(&prepared);
    assert_eq!(resolved.status, AotResolutionStatus::Compiled);
    assert!(resolved.registered.expected_rlib.exists());

    let selected = select_backend(
        &prepared,
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );
    assert_eq!(selected.effective_backend, SelectedBackendKind::AotCompiled);
    assert!(selected.is_compiled_aot());

    let consumer_dir = write_consumer_crate(
        temp.path(),
        "generated_bvp_sparse_full_cycle_fixture",
        "generated_bvp_sparse_full_cycle_module",
        &residual_chunk_specs,
        &jacobian_chunk_specs,
        &args,
        &expected_residual,
        &expected_values,
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
        "consumer crate test with statically linked BVP-derived generated AOT backend",
    );
}
