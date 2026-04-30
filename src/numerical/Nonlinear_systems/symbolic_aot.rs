//! Thin nonlinear-system bridge into the generic AOT lifecycle.
//!
//! This module keeps `Nonlinear_systems` code from manually rebuilding the
//! generic AOT stack every time it wants to use symbolic dense codegen.
//! It provides a narrow set of helpers:
//! - prepare a generic dense `PreparedProblem`,
//! - assemble a generated AOT crate from one symbolic nonlinear problem,
//! - materialize a build request on disk.
//!
//! In other words, this is the lifecycle-facing adapter for the nonlinear
//! family: symbolic code calls into it, but generic codegen/build layers remain
//! unaware of nonlinear-specific setup details.

use crate::numerical::Nonlinear_systems::symbolic::{
    PreparedSymbolicNonlinearAotProblem, SymbolicDenseAotOptions, SymbolicNonlinearProblem,
};
use crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile, AotBuildRequest, AotBuildResult,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_crate::GeneratedAotCrate;
use log::info;
use std::io;
use std::path::Path;

/// Prepares the nonlinear dense AOT bridge owned by the symbolic problem layer.
///
/// The returned bridge owns the flattened input ordering and the runtime-plan
/// buffers that the generic `PreparedDenseProblem` borrows from. Callers that
/// only need the generic shape can use `bridge.as_prepared_problem()`.
pub fn prepared_problem_from_symbolic_nonlinear_problem<'a>(
    problem: &'a SymbolicNonlinearProblem,
    options: SymbolicDenseAotOptions,
) -> PreparedSymbolicNonlinearAotProblem<'a> {
    problem.prepare_dense_aot_problem(options)
}

/// Builds a generated AOT crate directly from a symbolic nonlinear problem.
pub fn generated_aot_crate_from_symbolic_nonlinear_problem(
    crate_name: &str,
    module_name: &str,
    problem: &SymbolicNonlinearProblem,
    options: SymbolicDenseAotOptions,
) -> GeneratedAotCrate {
    let prepared = prepared_problem_from_symbolic_nonlinear_problem(problem, options);
    let prepared_problem = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::dense(
        prepared.as_prepared_problem(),
    );
    info!(
        "Assembling nonlinear symbolic dense AOT crate '{}' from prepared problem",
        crate_name
    );
    generated_aot_crate_from_prepared_problem(crate_name, module_name, &prepared_problem)
}

/// Materializes a build request for one symbolic nonlinear AOT crate.
pub fn materialize_symbolic_nonlinear_aot_build(
    crate_name: &str,
    module_name: &str,
    problem: &SymbolicNonlinearProblem,
    options: SymbolicDenseAotOptions,
    output_parent_dir: &Path,
    profile: AotBuildProfile,
) -> io::Result<AotBuildResult> {
    let crate_spec = generated_aot_crate_from_symbolic_nonlinear_problem(
        crate_name,
        module_name,
        problem,
        options,
    );
    AotBuildRequest::new(crate_spec, output_parent_dir, profile).materialize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;
    use crate::symbolic::symbolic_engine::Expr;
    use tempfile::tempdir;

    fn elementary_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_expressions_with_options(
            vec![
                Expr::parse_expression("x^2+y^2-10"),
                Expr::parse_expression("x-y-4"),
            ],
            SymbolicProblemOptions::new().with_variables(vec!["x".to_string(), "y".to_string()]),
        )
        .expect("problem should build")
    }

    #[test]
    fn nonlinear_symbolic_aot_bridge_builds_prepared_problem_with_aot_backend() {
        let problem = elementary_problem();
        let prepared = prepared_problem_from_symbolic_nonlinear_problem(
            &problem,
            SymbolicDenseAotOptions::default(),
        );
        let generic = prepared.as_prepared_problem();

        assert_eq!(
            generic.backend_kind,
            crate::symbolic::codegen::codegen_provider_api::BackendKind::Aot
        );
        assert_eq!(
            generic.matrix_backend,
            crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Dense
        );
        assert_eq!(prepared.residual_len(), 2);
        assert_eq!(prepared.jacobian_shape(), (2, 2));
    }

    #[test]
    fn nonlinear_symbolic_aot_bridge_builds_generated_crate() {
        let problem = elementary_problem();
        let crate_spec = generated_aot_crate_from_symbolic_nonlinear_problem(
            "generated_nonlinear_symbolic_fixture",
            "generated_nonlinear_symbolic_module",
            &problem,
            SymbolicDenseAotOptions::default(),
        );

        assert_eq!(
            crate_spec.crate_name,
            "generated_nonlinear_symbolic_fixture"
        );
        assert_eq!(crate_spec.manifest.backend_kind.as_str(), "aot");
        assert_eq!(crate_spec.manifest.matrix_backend.as_str(), "dense");
        assert!(
            crate_spec
                .module_source
                .contains("pub mod generated_nonlinear_symbolic_module")
        );
        assert!(
            crate_spec
                .module_source
                .contains("pub fn eval_nonlinear_residual")
        );
        assert!(
            crate_spec
                .module_source
                .contains("pub fn eval_nonlinear_jacobian")
        );
    }

    #[test]
    fn nonlinear_symbolic_aot_bridge_materializes_release_build_request() {
        let problem = elementary_problem();
        let dir = tempdir().expect("tempdir should exist");
        let result = materialize_symbolic_nonlinear_aot_build(
            "generated_nonlinear_symbolic_build",
            "generated_nonlinear_symbolic_build_module",
            &problem,
            SymbolicDenseAotOptions::default(),
            dir.path(),
            AotBuildProfile::Release,
        )
        .expect("build request should materialize");

        assert_eq!(result.cargo_command_line(), "cargo build --release");
        assert!(result.written.cargo_toml.exists());
        assert!(result.written.generated_rs.exists());
        assert_eq!(
            result
                .expected_rlib
                .file_name()
                .and_then(|name| name.to_str()),
            Some("libgenerated_nonlinear_symbolic_build.rlib")
        );
    }
}
