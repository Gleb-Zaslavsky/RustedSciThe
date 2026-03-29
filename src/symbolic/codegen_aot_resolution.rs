//! Resolution layer for reconnecting materialized AOT artifacts to the solver pipeline.
//!
//! This module sits on top of:
//! - generated crate writing,
//! - build request/result metadata,
//! - and the in-memory AOT registry.
//!
//! It does not load compiled code yet. Instead, it answers the next practical
//! question for a higher orchestration layer:
//! - given a prepared problem or its manifest,
//! - do we already know about a generated AOT backend for it,
//! - and does the expected static artifact appear to exist on disk?

use crate::symbolic::codegen_aot_registry::{AotRegistry, RegisteredAotArtifact};
use crate::symbolic::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen_provider_api::PreparedProblem;

/// Resolution status for one registered AOT backend artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotResolutionStatus {
    /// No registered artifact exists for the requested problem key.
    Missing,
    /// A generated crate is known, but the expected compiled static artifact is
    /// not currently present on disk.
    RegisteredButNotBuilt,
    /// The expected compiled static artifact exists on disk.
    Compiled,
}

/// One resolved AOT artifact record returned by lookup helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAotArtifact {
    pub registered: RegisteredAotArtifact,
    pub status: AotResolutionStatus,
}

impl ResolvedAotArtifact {
    /// Returns `true` when the expected compiled artifact exists on disk.
    pub fn is_compiled(&self) -> bool {
        self.status == AotResolutionStatus::Compiled
    }
}

/// Resolver that consults an [`AotRegistry`] and derives artifact readiness.
#[derive(Debug, Clone)]
pub struct AotResolver {
    registry: AotRegistry,
}

impl AotResolver {
    /// Creates a new resolver from an existing registry snapshot.
    pub fn new(registry: AotRegistry) -> Self {
        Self { registry }
    }

    /// Returns the underlying registry snapshot.
    pub fn registry(&self) -> &AotRegistry {
        &self.registry
    }

    /// Resolves an AOT backend by manifest.
    pub fn resolve_by_manifest(&self, manifest: &PreparedProblemManifest) -> ResolvedAotArtifact {
        self.resolve_by_problem_key(&manifest.problem_key())
    }

    /// Resolves an AOT backend by prepared problem.
    pub fn resolve_prepared_problem(&self, problem: &PreparedProblem<'_>) -> ResolvedAotArtifact {
        let manifest = PreparedProblemManifest::from(problem);
        self.resolve_by_manifest(&manifest)
    }

    /// Resolves an AOT backend by generated crate name.
    pub fn resolve_by_crate_name(&self, crate_name: &str) -> ResolvedAotArtifact {
        match self.registry.get_by_crate_name(crate_name) {
            Some(registered) => Self::resolved_from_registered(registered.clone()),
            None => ResolvedAotArtifact {
                registered: RegisteredAotArtifact {
                    problem_key: String::new(),
                    crate_name: crate_name.to_string(),
                    manifest: PreparedProblemManifest {
                        backend_kind: crate::symbolic::codegen_provider_api::BackendKind::Aot,
                        matrix_backend:
                            crate::symbolic::codegen_provider_api::MatrixBackend::ValuesOnly,
                        io: crate::symbolic::codegen_manifest::ProblemIoManifest {
                            input_names: Vec::new(),
                            residual_len: 0,
                            jacobian_rows: 0,
                            jacobian_cols: 0,
                            jacobian_nnz: None,
                        },
                        functions: crate::symbolic::codegen_manifest::GeneratedFunctionsManifest {
                            residual_fn_name: String::new(),
                            residual_chunk_names: Vec::new(),
                            jacobian_fn_name: String::new(),
                            jacobian_chunk_names: Vec::new(),
                        },
                    },
                    crate_dir: Default::default(),
                    artifact_dir: Default::default(),
                    expected_rlib: Default::default(),
                    cargo_program: String::new(),
                    cargo_args: Vec::new(),
                },
                status: AotResolutionStatus::Missing,
            },
        }
    }

    /// Resolves an AOT backend by manifest-derived `problem_key`.
    pub fn resolve_by_problem_key(&self, problem_key: &str) -> ResolvedAotArtifact {
        match self.registry.get_by_problem_key(problem_key) {
            Some(registered) => Self::resolved_from_registered(registered.clone()),
            None => ResolvedAotArtifact {
                registered: RegisteredAotArtifact {
                    problem_key: problem_key.to_string(),
                    crate_name: String::new(),
                    manifest: PreparedProblemManifest {
                        backend_kind: crate::symbolic::codegen_provider_api::BackendKind::Aot,
                        matrix_backend:
                            crate::symbolic::codegen_provider_api::MatrixBackend::ValuesOnly,
                        io: crate::symbolic::codegen_manifest::ProblemIoManifest {
                            input_names: Vec::new(),
                            residual_len: 0,
                            jacobian_rows: 0,
                            jacobian_cols: 0,
                            jacobian_nnz: None,
                        },
                        functions: crate::symbolic::codegen_manifest::GeneratedFunctionsManifest {
                            residual_fn_name: String::new(),
                            residual_chunk_names: Vec::new(),
                            jacobian_fn_name: String::new(),
                            jacobian_chunk_names: Vec::new(),
                        },
                    },
                    crate_dir: Default::default(),
                    artifact_dir: Default::default(),
                    expected_rlib: Default::default(),
                    cargo_program: String::new(),
                    cargo_args: Vec::new(),
                },
                status: AotResolutionStatus::Missing,
            },
        }
    }

    fn resolved_from_registered(registered: RegisteredAotArtifact) -> ResolvedAotArtifact {
        let status = if registered.expected_rlib.exists() {
            AotResolutionStatus::Compiled
        } else {
            AotResolutionStatus::RegisteredButNotBuilt
        };
        ResolvedAotArtifact { registered, status }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_aot_build::{AotBuildProfile, AotBuildRequest};
    use crate::symbolic::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
    use crate::symbolic::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem, PreparedProblem,
    };
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::symbolic_engine::Expr;
    use std::fs;
    use tempfile::tempdir;

    fn sample_prepared_problem() -> PreparedProblem<'static> {
        let residuals = Box::leak(Box::new(vec![Expr::parse_expression("x + 1")]));
        let jacobian = Box::leak(Box::new(vec![vec![Expr::parse_expression("1")]]));
        let vars = Box::leak(Box::new(vec!["x"]));

        PreparedProblem::dense(PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual",
                residuals,
                variables: vars,
                params: None,
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            JacobianTask {
                fn_name: "eval_jacobian",
                jacobian,
                variables: vars,
                params: None,
            }
            .runtime_plan(DenseJacobianChunkingStrategy::Whole),
        ))
    }

    #[test]
    fn resolver_reports_registered_but_not_built_when_expected_rlib_is_missing() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_resolution_fixture",
            "generated_resolution_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Release)
            .materialize()
            .expect("build request should materialize");

        let mut registry = AotRegistry::new();
        registry.register_materialized_build(manifest.clone(), &build);

        let resolver = AotResolver::new(registry);
        let resolved = resolver.resolve_by_manifest(&manifest);

        assert_eq!(resolved.status, AotResolutionStatus::RegisteredButNotBuilt);
        assert!(!resolved.is_compiled());
        assert_eq!(
            resolved.registered.cargo_command_line(),
            "cargo build --release"
        );
    }

    #[test]
    fn resolver_reports_compiled_when_expected_rlib_exists() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_resolution_compiled_fixture",
            "generated_resolution_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");
        fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
        fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

        let mut registry = AotRegistry::new();
        registry.register_materialized_build(manifest.clone(), &build);

        let resolver = AotResolver::new(registry);
        let resolved = resolver.resolve_prepared_problem(&prepared);

        assert_eq!(resolved.status, AotResolutionStatus::Compiled);
        assert!(resolved.is_compiled());
        assert_eq!(resolved.registered.expected_rlib, build.expected_rlib);
    }

    #[test]
    fn resolver_reports_missing_for_unknown_problem_key() {
        let resolver = AotResolver::new(AotRegistry::new());
        let resolved = resolver.resolve_by_problem_key("deadbeef");

        assert_eq!(resolved.status, AotResolutionStatus::Missing);
        assert_eq!(resolved.registered.problem_key, "deadbeef");
        assert!(resolved.registered.crate_name.is_empty());
    }
}
