//! Registry integration for C AOT builds.
//!
//! This module provides helpers to register C AOT build results in the unified
//! AOT registry, allowing both Rust and C backends to coexist in the same
//! registry and be resolved through the same resolution layer.

use crate::symbolic::codegen::c_backend::codegen_c_aot_build::CAotBuildResult;
use crate::symbolic::codegen::codegen_aot_registry::{AotRegistry, RegisteredAotArtifact};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;

/// Registers a materialized C AOT build result in the unified AOT registry.
///
/// This function adapts the C build result to the registry's expected format,
/// allowing C and Rust backends to be registered and resolved uniformly.
pub fn register_c_build_in_registry<'a>(
    registry: &'a mut AotRegistry,
    manifest: PreparedProblemManifest,
    build: &CAotBuildResult,
) -> &'a RegisteredAotArtifact {
    // Create a temporary AotBuildResult that mimics the C build
    // This allows us to reuse the existing registry infrastructure
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildResult;
    use crate::symbolic::codegen::rust_backend::codegen_aot_crate::WrittenAotCrate;

    let temp_written = WrittenAotCrate {
        crate_dir: build.written.library_dir.clone(),
        cargo_toml: build.written.makefile.clone(),
        src_dir: build.written.library_dir.join("src"),
        lib_rs: build.written.generated_c.clone(),
        generated_rs: build.written.generated_c.clone(),
        manifest_rs: build.written.manifest_h.clone(),
    };

    let temp_build = AotBuildResult {
        written: temp_written,
        cargo_program: build.build_program.clone(),
        cargo_wrapper_program: None,
        cargo_args: build.build_args.clone(),
        cargo_env: build.build_env.clone(),
        artifact_dir: build.artifact_dir.clone(),
        expected_rlib: build.expected_so.clone(),
        expected_cdylib: build.expected_so.clone(),
    };

    registry.register_materialized_build(manifest, &temp_build)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule};
    use crate::symbolic::codegen::c_backend::codegen_c_aot_build::{
        CAotBuildProfile, CAotBuildRequest,
    };
    use crate::symbolic::codegen::c_backend::codegen_c_aot_library::GeneratedCAotLibrary;
    use crate::symbolic::codegen::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem,
    };
    use crate::symbolic::codegen::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::symbolic_engine::Expr;
    use tempfile::tempdir;

    #[test]
    fn c_build_can_be_registered_in_unified_registry() {
        let residuals = vec![Expr::parse_expression("x + 1")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];

        let prepared = PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: None,
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            JacobianTask {
                fn_name: "eval_jacobian",
                jacobian: &jacobian,
                variables: &vars,
                params: None,
            }
            .runtime_plan(DenseJacobianChunkingStrategy::Whole),
        );

        let manifest = PreparedProblemManifest::from(&prepared);
        let mut module = CodegenModule::new("test_module").with_language(CodegenLanguage::C);
        for chunk in &prepared.residual_plan.chunks {
            module.push_residual_block_plan(&chunk.plan);
        }
        for chunk in &prepared.jacobian_plan.chunks {
            module.push_dense_jacobian_plan(&chunk.plan);
        }

        let library_spec = GeneratedCAotLibrary::from_prepared_dense_problem(
            "test_c_registry",
            &prepared,
            &module,
        );
        let dir = tempdir().expect("tempdir should exist");

        let request = CAotBuildRequest::new(library_spec, dir.path(), CAotBuildProfile::Release);
        let build = request
            .materialize()
            .expect("C build request should materialize");

        let mut registry = AotRegistry::new();
        let problem_key_clone = manifest.problem_key();
        let expected_so_clone = build.expected_so.clone();
        let registered = register_c_build_in_registry(&mut registry, manifest.clone(), &build);

        assert_eq!(registered.problem_key, problem_key_clone);
        assert_eq!(registered.crate_name, "test_c_registry");
        assert_eq!(registered.expected_cdylib, expected_so_clone);

        assert_eq!(registry.len(), 1);

        // Verify we can look it up by problem key
        let found = registry
            .get_by_problem_key(&problem_key_clone)
            .expect("should find registered C build");
        assert_eq!(found.crate_name, "test_c_registry");
    }
}
