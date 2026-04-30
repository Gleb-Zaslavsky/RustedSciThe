//! Thin IVP bridge into the generic AOT lifecycle.
//!
//! This module keeps shared IVP symbolic code from rebuilding the generic AOT
//! stack by hand every time it wants to emit one generated dense IVP artifact.
//!
//! Current layering:
//! - artifact generation is backend-agnostic (`Rust` / `C` / `Zig`);
//! - generic materialized build requests are also backend-agnostic;
//! - the legacy `materialize_symbolic_ivp_aot_build(...)` helper remains
//!   intentionally Rust-shaped as a compatibility wrapper around the older
//!   Rust-only API.

use crate::symbolic::codegen::c_backend::codegen_c_aot_library::GeneratedCAotLibrary;
use crate::symbolic::codegen::codegen_aot_driver::{
    generated_aot_artifact_from_prepared_problem, generated_aot_build_request_from_artifact,
    AotBuildPreset, AotCodegenBackend, GeneratedAotArtifact, GeneratedAotBuildResult,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile, AotBuildRequest, AotBuildResult,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_crate::GeneratedAotCrate;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_library::GeneratedZigAotLibrary;
use crate::symbolic::codegen::CodegenIR::CodegenModule;
use crate::symbolic::symbolic_ivp::{
    PreparedSymbolicIvpAotProblem, PreparedSymbolicIvpProblem,
    PreparedSymbolicIvpResidualAotProblem, PreparedSymbolicIvpResidualProblem,
    SymbolicIvpAotOptions,
};
use log::info;
use std::io;
use std::path::Path;

/// Prepares the dense IVP AOT bridge owned by the shared symbolic IVP layer.
pub fn prepared_problem_from_symbolic_ivp_problem<'a>(
    problem: &'a PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
) -> PreparedSymbolicIvpAotProblem<'a> {
    problem.prepare_dense_aot_problem(options)
}

/// Builds one backend-selected AOT artifact directly from a prepared symbolic IVP problem.
pub fn generated_aot_artifact_from_symbolic_ivp_problem(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    backend: AotCodegenBackend,
) -> GeneratedAotArtifact {
    let prepared = prepared_problem_from_symbolic_ivp_problem(problem, options);
    let prepared_problem = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::dense(
        prepared.as_prepared_problem(),
    );
    info!(
        "Assembling symbolic IVP dense {:?} artifact '{}' from prepared problem",
        backend, artifact_name
    );
    generated_aot_artifact_from_prepared_problem(
        artifact_name,
        module_name,
        &prepared_problem,
        backend,
    )
}

/// Prepares the residual-only IVP AOT bridge owned by the shared symbolic IVP layer.
pub fn prepared_residual_problem_from_symbolic_ivp_residual_problem<'a>(
    problem: &'a PreparedSymbolicIvpResidualProblem,
    options: SymbolicIvpAotOptions,
) -> PreparedSymbolicIvpResidualAotProblem<'a> {
    problem.prepare_residual_aot_problem(options)
}

/// Builds one backend-selected AOT artifact from a residual-only IVP problem.
///
/// The emitted library contains the regular residual symbol plus a no-op
/// Jacobian symbol required by the shared AOT ABI.  Native sparse/banded LSODE2
/// paths ignore that no-op Jacobian and use their own symbolic Jacobian
/// evaluator.
pub fn generated_aot_artifact_from_symbolic_ivp_residual_problem(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpResidualProblem,
    options: SymbolicIvpAotOptions,
    backend: AotCodegenBackend,
) -> GeneratedAotArtifact {
    let prepared = prepared_residual_problem_from_symbolic_ivp_residual_problem(problem, options);
    let residual_plan = prepared.residual_runtime_plan();
    let mut module = CodegenModule::new(module_name).with_language(backend.codegen_language());
    for chunk in &residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    let manifest = prepared.manifest();
    info!(
        "Assembling symbolic IVP residual-only {:?} artifact '{}' from prepared problem",
        backend, artifact_name
    );
    match backend {
        AotCodegenBackend::Rust => GeneratedAotArtifact::Rust(
            GeneratedAotCrate::from_codegen_module(artifact_name, &module, manifest),
        ),
        AotCodegenBackend::C => GeneratedAotArtifact::C(GeneratedCAotLibrary::from_codegen_module(
            artifact_name,
            &module,
            manifest,
        )),
        AotCodegenBackend::Zig => GeneratedAotArtifact::Zig(
            GeneratedZigAotLibrary::from_codegen_module(artifact_name, &module, manifest),
        ),
    }
}

/// Builds a generated Rust AOT crate directly from a prepared symbolic IVP problem.
pub fn generated_aot_crate_from_symbolic_ivp_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
) -> GeneratedAotCrate {
    generated_aot_artifact_from_symbolic_ivp_problem(
        crate_name,
        module_name,
        problem,
        options,
        AotCodegenBackend::Rust,
    )
    .into_rust_crate()
    .expect("Rust backend must return GeneratedAotCrate")
}

/// Materializes a build request for one symbolic IVP AOT crate.
pub fn materialize_symbolic_ivp_aot_build(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    output_parent_dir: &Path,
    profile: AotBuildProfile,
) -> io::Result<AotBuildResult> {
    let crate_spec =
        generated_aot_crate_from_symbolic_ivp_problem(crate_name, module_name, problem, options);
    AotBuildRequest::new(crate_spec, output_parent_dir, profile).materialize()
}

/// Materializes one backend-selected symbolic IVP AOT artifact build.
pub fn materialize_symbolic_ivp_aot_artifact_build(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    backend: AotCodegenBackend,
    output_parent_dir: &Path,
    preset: AotBuildPreset,
) -> io::Result<GeneratedAotBuildResult> {
    let artifact = generated_aot_artifact_from_symbolic_ivp_problem(
        artifact_name,
        module_name,
        problem,
        options,
        backend,
    );
    generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset).materialize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_ivp::{
        prepare_symbolic_ivp_problem, prepare_symbolic_ivp_residual_problem,
        SymbolicIvpProblemOptions,
    };
    use nalgebra::DVector;
    use tempfile::tempdir;

    fn elementary_ivp_problem() -> PreparedSymbolicIvpProblem {
        prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
        )
        .expect("IVP problem should prepare")
    }

    fn elementary_ivp_residual_problem() -> PreparedSymbolicIvpResidualProblem {
        prepare_symbolic_ivp_residual_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
        )
        .expect("IVP residual problem should prepare")
    }

    #[test]
    fn ivp_symbolic_aot_bridge_builds_prepared_problem_with_aot_backend() {
        let problem = elementary_ivp_problem();
        let prepared =
            prepared_problem_from_symbolic_ivp_problem(&problem, SymbolicIvpAotOptions::default());
        let generic = prepared.as_prepared_problem();

        assert_eq!(
            generic.backend_kind,
            crate::symbolic::codegen::codegen_provider_api::BackendKind::Aot
        );
        assert_eq!(
            generic.matrix_backend,
            crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Dense
        );
        assert_eq!(
            prepared.flattened_input_names(),
            &["t", "a", "b", "c", "y", "z"]
        );
    }

    #[test]
    fn ivp_symbolic_aot_bridge_materializes_release_build_request() {
        let problem = elementary_ivp_problem();
        let dir = tempdir().expect("tempdir should exist");
        let result = materialize_symbolic_ivp_aot_build(
            "generated_symbolic_ivp_build",
            "generated_symbolic_ivp_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            dir.path(),
            AotBuildProfile::Release,
        )
        .expect("IVP build request should materialize");

        assert_eq!(result.cargo_command_line(), "cargo build --release");
        assert!(result.written.cargo_toml.exists());
        assert!(result.written.generated_rs.exists());
    }

    #[test]
    fn ivp_symbolic_aot_bridge_can_emit_non_rust_artifacts() {
        let problem = elementary_ivp_problem();

        let c_artifact = generated_aot_artifact_from_symbolic_ivp_problem(
            "generated_symbolic_ivp_c",
            "generated_symbolic_ivp_c_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::C,
        );
        let zig_artifact = generated_aot_artifact_from_symbolic_ivp_problem(
            "generated_symbolic_ivp_zig",
            "generated_symbolic_ivp_zig_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Zig,
        );

        match c_artifact {
            GeneratedAotArtifact::C(library) => {
                assert_eq!(library.library_name, "generated_symbolic_ivp_c");
                assert!(!library.c_source.is_empty());
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::Zig(_) => {
                panic!("C backend should emit GeneratedCAotLibrary")
            }
        }

        match zig_artifact {
            GeneratedAotArtifact::Zig(library) => {
                assert_eq!(library.library_name, "generated_symbolic_ivp_zig");
                assert!(!library.zig_source.is_empty());
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::C(_) => {
                panic!("Zig backend should emit GeneratedZigAotLibrary")
            }
        }
    }

    #[test]
    fn ivp_symbolic_aot_bridge_can_emit_residual_only_artifacts() {
        let problem = elementary_ivp_residual_problem();
        let prepared = prepared_residual_problem_from_symbolic_ivp_residual_problem(
            &problem,
            SymbolicIvpAotOptions::default(),
        );
        assert_eq!(
            prepared.flattened_input_names(),
            &["t", "a", "b", "c", "y", "z"]
        );
        assert_eq!(prepared.manifest().io.jacobian_nnz, Some(0));
        assert!(prepared.manifest().functions.jacobian_fn_name.is_empty());

        let c_artifact = generated_aot_artifact_from_symbolic_ivp_residual_problem(
            "generated_symbolic_ivp_residual_c",
            "generated_symbolic_ivp_residual_c_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::C,
        );
        let zig_artifact = generated_aot_artifact_from_symbolic_ivp_residual_problem(
            "generated_symbolic_ivp_residual_zig",
            "generated_symbolic_ivp_residual_zig_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Zig,
        );

        match c_artifact {
            GeneratedAotArtifact::C(library) => {
                assert!(library.c_source.contains("generated_ivp_residual_eval"));
                assert_eq!(library.manifest.io.jacobian_nnz, Some(0));
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::Zig(_) => {
                panic!("C backend should emit residual-only GeneratedCAotLibrary")
            }
        }

        match zig_artifact {
            GeneratedAotArtifact::Zig(library) => {
                assert!(library.zig_source.contains("generated_ivp_residual_eval"));
                assert_eq!(library.manifest.io.jacobian_nnz, Some(0));
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::C(_) => {
                panic!("Zig backend should emit residual-only GeneratedZigAotLibrary")
            }
        }
    }

    #[test]
    fn ivp_symbolic_aot_bridge_can_materialize_generic_c_and_zig_builds() {
        let problem = elementary_ivp_problem();
        let dir = tempdir().expect("tempdir should exist");

        let c_build = materialize_symbolic_ivp_aot_artifact_build(
            "generated_symbolic_ivp_c_build",
            "generated_symbolic_ivp_c_build_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::C,
            dir.path(),
            AotBuildPreset::DevFastest,
        )
        .expect("C IVP build should materialize");

        let zig_build = materialize_symbolic_ivp_aot_artifact_build(
            "generated_symbolic_ivp_zig_build",
            "generated_symbolic_ivp_zig_build_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Zig,
            dir.path(),
            AotBuildPreset::DevFastest,
        )
        .expect("Zig IVP build should materialize");

        match c_build {
            GeneratedAotBuildResult::C(result) => {
                assert!(result.written.makefile.exists());
                assert!(result.written.generated_c.exists());
            }
            GeneratedAotBuildResult::Rust(_) | GeneratedAotBuildResult::Zig(_) => {
                panic!("C backend should materialize CAotBuildResult")
            }
        }

        match zig_build {
            GeneratedAotBuildResult::Zig(result) => {
                assert!(result.written.build_zig.exists());
                assert!(result.written.generated_zig.exists());
            }
            GeneratedAotBuildResult::Rust(_) | GeneratedAotBuildResult::C(_) => {
                panic!("Zig backend should materialize ZigAotBuildResult")
            }
        }
    }
}
