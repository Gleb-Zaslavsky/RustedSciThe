//! Registry of materialized AOT crate artifacts and their manifests.
//!
//! This is the first reconnect layer on the way back from code generation to
//! solver integration:
//! - codegen/build layers write a tiny generated crate,
//! - the registry remembers where that crate and its expected artifacts live,
//! - later resolution layers can use this metadata to choose and reconnect the
//!   compiled backend requested by a symbolic `Jacobian` orchestration path.
//!
//! The registry is intentionally lightweight. It does not compile crates and it
//! does not load them. Its job is to keep a stable association between:
//! - an owned [`PreparedProblemManifest`](crate::symbolic::codegen_manifest::PreparedProblemManifest),
//! - a derived `problem_key`,
//! - and the on-disk locations returned by `AotBuildRequest::materialize()`.

use crate::symbolic::codegen_aot_build::AotBuildResult;
use crate::symbolic::codegen_manifest::PreparedProblemManifest;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Registered on-disk AOT artifact metadata keyed by prepared-problem manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredAotArtifact {
    pub problem_key: String,
    pub crate_name: String,
    pub manifest: PreparedProblemManifest,
    pub crate_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub expected_rlib: PathBuf,
    pub cargo_program: String,
    pub cargo_args: Vec<String>,
}

impl RegisteredAotArtifact {
    /// Returns the printable Cargo command line associated with this artifact.
    pub fn cargo_command_line(&self) -> String {
        let mut parts = vec![self.cargo_program.clone()];
        parts.extend(self.cargo_args.iter().cloned());
        parts.join(" ")
    }
}

/// In-memory registry of materialized AOT artifacts.
#[derive(Debug, Clone, Default)]
pub struct AotRegistry {
    entries_by_problem_key: BTreeMap<String, RegisteredAotArtifact>,
    crate_name_to_problem_key: BTreeMap<String, String>,
}

impl AotRegistry {
    /// Creates an empty AOT artifact registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of registered problem entries.
    pub fn len(&self) -> usize {
        self.entries_by_problem_key.len()
    }

    /// Returns `true` when no artifacts are registered yet.
    pub fn is_empty(&self) -> bool {
        self.entries_by_problem_key.is_empty()
    }

    /// Returns the currently registered manifest-derived problem keys.
    pub fn problem_keys(&self) -> Vec<String> {
        self.entries_by_problem_key.keys().cloned().collect()
    }

    /// Registers one materialized build result under the manifest-derived
    /// `problem_key`. Re-registering the same problem replaces the old record.
    pub fn register_materialized_build(
        &mut self,
        manifest: PreparedProblemManifest,
        build: &AotBuildResult,
    ) -> &RegisteredAotArtifact {
        let problem_key = manifest.problem_key();
        let crate_name = build
            .written
            .crate_dir
            .file_name()
            .and_then(|name| name.to_str())
            .expect("generated crate directory should end with a valid crate name")
            .to_string();

        if let Some(previous) = self.entries_by_problem_key.insert(
            problem_key.clone(),
            RegisteredAotArtifact {
                problem_key: problem_key.clone(),
                crate_name: crate_name.clone(),
                manifest,
                crate_dir: build.written.crate_dir.clone(),
                artifact_dir: build.artifact_dir.clone(),
                expected_rlib: build.expected_rlib.clone(),
                cargo_program: build.cargo_program.clone(),
                cargo_args: build.cargo_args.clone(),
            },
        ) {
            self.crate_name_to_problem_key.remove(&previous.crate_name);
        }

        self.crate_name_to_problem_key
            .insert(crate_name, problem_key.clone());

        self.entries_by_problem_key
            .get(&problem_key)
            .expect("newly inserted registry entry should exist")
    }

    /// Looks up a registered artifact by its manifest-derived `problem_key`.
    pub fn get_by_problem_key(&self, problem_key: &str) -> Option<&RegisteredAotArtifact> {
        self.entries_by_problem_key.get(problem_key)
    }

    /// Looks up a registered artifact by manifest contents.
    pub fn get_by_manifest(
        &self,
        manifest: &PreparedProblemManifest,
    ) -> Option<&RegisteredAotArtifact> {
        self.get_by_problem_key(&manifest.problem_key())
    }

    /// Looks up a registered artifact by generated crate name.
    pub fn get_by_crate_name(&self, crate_name: &str) -> Option<&RegisteredAotArtifact> {
        let problem_key = self.crate_name_to_problem_key.get(crate_name)?;
        self.entries_by_problem_key.get(problem_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_aot_build::{AotBuildProfile, AotBuildRequest};
    use crate::symbolic::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
    use crate::symbolic::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem, PreparedProblem,
    };
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::symbolic_engine::Expr;
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
    fn registry_registers_and_finds_materialized_builds() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_registry_fixture",
            "generated_registry_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Release)
            .materialize()
            .expect("build request should materialize");

        let mut registry = AotRegistry::new();
        let registered_problem_key = registry
            .register_materialized_build(manifest.clone(), &build)
            .problem_key
            .clone();

        assert_eq!(registry.len(), 1);
        assert_eq!(registered_problem_key, manifest.problem_key());
        let registered = registry
            .get_by_problem_key(&registered_problem_key)
            .expect("problem-key lookup should succeed");
        assert_eq!(registered.crate_name, "generated_registry_fixture");
        assert_eq!(registered.cargo_command_line(), "cargo build --release");
        assert_eq!(
            registry
                .get_by_manifest(&manifest)
                .expect("manifest lookup should succeed")
                .expected_rlib,
            build.expected_rlib
        );
        assert_eq!(
            registry
                .get_by_crate_name("generated_registry_fixture")
                .expect("crate-name lookup should succeed")
                .problem_key,
            manifest.problem_key()
        );
    }

    #[test]
    fn registry_replaces_existing_problem_key_and_updates_crate_name_lookup() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let dir = tempdir().expect("tempdir should exist");

        let build0 = AotBuildRequest::new(
            generated_aot_crate_from_prepared_problem(
                "generated_registry_old",
                "generated_registry_module",
                &prepared,
            ),
            dir.path(),
            AotBuildProfile::Debug,
        )
        .materialize()
        .expect("first build should materialize");

        let build1 = AotBuildRequest::new(
            generated_aot_crate_from_prepared_problem(
                "generated_registry_new",
                "generated_registry_module",
                &prepared,
            ),
            dir.path(),
            AotBuildProfile::Release,
        )
        .materialize()
        .expect("second build should materialize");

        let mut registry = AotRegistry::new();
        registry.register_materialized_build(manifest.clone(), &build0);
        registry.register_materialized_build(manifest.clone(), &build1);

        assert_eq!(registry.len(), 1);
        assert!(
            registry
                .get_by_crate_name("generated_registry_old")
                .is_none()
        );

        let registered = registry
            .get_by_crate_name("generated_registry_new")
            .expect("new crate-name lookup should succeed");
        assert_eq!(registered.expected_rlib, build1.expected_rlib);
        assert_eq!(registered.cargo_command_line(), "cargo build --release");
    }
}
