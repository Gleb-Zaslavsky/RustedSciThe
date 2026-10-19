//! Shared test helpers for dense nonlinear AOT lifecycle and solver tests.
//!
//! These helpers intentionally centralize three pieces of boilerplate that
//! would otherwise be repeated across acceptance modules:
//! - serializing access to process-global linked dense runtime state,
//! - faking one materialized release build and resolver snapshot,
//! - registering small linked dense backends used by solve-level tests.

#![cfg(test)]

use crate::numerical::Nonlinear_systems::symbolic::{
    SymbolicDenseAotOptions, SymbolicNonlinearProblem,
};
use crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedDenseAotBackend, register_linked_dense_backend,
};
use crate::symbolic::codegen::codegen_provider_api::PreparedProblem;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{AotBuildProfile, AotBuildRequest};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Serializes dense nonlinear AOT tests that share process-global linked runtime state.
pub fn aot_solver_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("AOT solver acceptance mutex should not be poisoned")
}

/// Creates a fake materialized release build and resolver for one dense nonlinear problem.
pub fn linked_dense_resolver_for_problem(
    problem: &SymbolicNonlinearProblem,
) -> (tempfile::TempDir, AotResolver, String) {
    let prepared_bridge = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
    let prepared = PreparedProblem::dense(prepared_bridge.as_prepared_problem());
    let manifest = prepared_bridge.manifest();

    let dir = tempfile::tempdir().expect("tempdir should exist");
    let build = AotBuildRequest::new(
        generated_aot_crate_from_prepared_problem(
            "generated_nonlinear_dense_solver_fixture",
            "generated_nonlinear_dense_solver_module",
            &prepared,
        ),
        dir.path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");

    std::fs::create_dir_all(&build.artifact_dir).expect("artifact dir should exist");
    std::fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    (
        dir,
        AotResolver::new(registry),
        prepared_bridge.problem_key(),
    )
}

/// Registers a linked dense backend for the elementary nonlinear system.
pub fn register_elementary_dense_backend(problem_key: &str) {
    register_linked_dense_backend(LinkedDenseAotBackend::new(
        problem_key.to_string(),
        2,
        (2, 2),
        Arc::new(|args, out| {
            let x = args[0];
            let y = args[1];
            out[0] = x * x + y * y - 10.0;
            out[1] = x - y - 4.0;
        }),
        Arc::new(|args, out| {
            let x = args[0];
            let y = args[1];
            out[0] = 2.0 * x;
            out[1] = 2.0 * y;
            out[2] = 1.0;
            out[3] = -1.0;
        }),
    ));
}

/// Registers a linked dense backend for the parameterized dense nonlinear system.
pub fn register_parameterized_dense_backend(problem_key: &str) {
    register_linked_dense_backend(LinkedDenseAotBackend::new(
        problem_key.to_string(),
        2,
        (2, 2),
        Arc::new(|args, out| {
            let a = args[0];
            let x = args[1];
            let y = args[2];
            out[0] = a * x + y - 3.0;
            out[1] = x - y;
        }),
        Arc::new(|args, out| {
            let a = args[0];
            out[0] = a;
            out[1] = 1.0;
            out[2] = 1.0;
            out[3] = -1.0;
        }),
    ));
}
