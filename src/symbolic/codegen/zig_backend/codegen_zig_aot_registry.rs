//! Registry integration for Zig AOT builds.
//!
//! Mirrors `codegen_c_aot_registry.rs` but for Zig compiled libraries.

use crate::symbolic::codegen::codegen_aot_registry::{AotRegistry, RegisteredAotArtifact};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_build::ZigAotBuildResult;

/// Registers a materialized Zig AOT build result in the unified AOT registry.
pub fn register_zig_build_in_registry<'a>(
    registry: &'a mut AotRegistry,
    manifest: PreparedProblemManifest,
    build: &ZigAotBuildResult,
) -> &'a RegisteredAotArtifact {
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildResult;
    use crate::symbolic::codegen::rust_backend::codegen_aot_crate::WrittenAotCrate;

    let temp_written = WrittenAotCrate {
        crate_dir: build.written.library_dir.clone(),
        cargo_toml: build.written.build_zig.clone(),
        src_dir: build.written.library_dir.join("src"),
        lib_rs: build.written.generated_zig.clone(),
        generated_rs: build.written.generated_zig.clone(),
        manifest_rs: build.written.aot_interface_zig.clone(),
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
    use crate::symbolic::codegen::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem,
    };
    use crate::symbolic::codegen::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::codegen::zig_backend::codegen_zig_aot_build::{
        ZigAotBuildProfile, ZigAotBuildRequest,
    };
    use crate::symbolic::codegen::zig_backend::codegen_zig_aot_library::GeneratedZigAotLibrary;
    use crate::symbolic::symbolic_engine::Expr;
    use tempfile::tempdir;

    #[test]
    fn zig_build_can_be_registered_in_unified_registry() {
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
        let module = CodegenModule::new("test_module").with_language(CodegenLanguage::Zig);
        let library_spec = GeneratedZigAotLibrary::from_prepared_dense_problem(
            "test_zig_registry",
            &prepared,
            &module,
        );
        let dir = tempdir().expect("tempdir should exist");
        let request =
            ZigAotBuildRequest::new(library_spec, dir.path(), ZigAotBuildProfile::ReleaseFast);
        let build = request
            .materialize()
            .expect("Zig build request should materialize");

        let mut registry = AotRegistry::new();
        let problem_key_clone = manifest.problem_key();
        let expected_so_clone = build.expected_so.clone();
        let registered =
            register_zig_build_in_registry(&mut registry, manifest.clone(), &build);

        assert_eq!(registered.problem_key, problem_key_clone);
        assert_eq!(registered.crate_name, "test_zig_registry");
        assert_eq!(registered.expected_cdylib, expected_so_clone);

        assert_eq!(registry.len(), 1);

        let found = registry
            .get_by_problem_key(&problem_key_clone)
            .expect("should find registered Zig build");
        assert_eq!(found.crate_name, "test_zig_registry");
    }
}
