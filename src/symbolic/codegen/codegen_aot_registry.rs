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

use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildResult;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::PathBuf;

/// Registered on-disk AOT artifact metadata keyed by prepared-problem manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredAotArtifact {
    pub problem_key: String,
    pub crate_name: String,
    pub manifest: PreparedProblemManifest,
    pub crate_dir: PathBuf,
    pub manifest_file: PathBuf,
    pub artifact_dir: PathBuf,
    pub expected_rlib: PathBuf,
    pub expected_cdylib: PathBuf,
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

    /// Recomputes the manifest-derived key stored in this registry entry.
    pub fn manifest_problem_key(&self) -> String {
        self.manifest.problem_key()
    }

    /// Returns true when the registry key still matches the stored manifest.
    pub fn manifest_key_matches(&self) -> bool {
        self.problem_key == self.manifest_problem_key()
    }

    /// Returns true when the generated manifest/header file still exists.
    pub fn manifest_file_exists(&self) -> bool {
        self.manifest_file.exists()
    }

    /// Returns true when at least one compiled output expected by the registry exists.
    pub fn compiled_artifact_exists(&self) -> bool {
        self.expected_cdylib.exists() || self.expected_rlib.exists()
    }

    /// Human-readable lifecycle contract summary for diagnostics and story tables.
    pub fn lifecycle_contract_summary(&self) -> String {
        format!(
            "problem_key={}, manifest_key={}, manifest_key_matches={}, manifest_file={}, manifest_file_exists={}, expected_cdylib={}, expected_cdylib_exists={}, expected_rlib={}, expected_rlib_exists={}, artifact_dir={}",
            self.problem_key,
            self.manifest_problem_key(),
            self.manifest_key_matches(),
            self.manifest_file.display(),
            self.manifest_file_exists(),
            self.expected_cdylib.display(),
            self.expected_cdylib.exists(),
            self.expected_rlib.display(),
            self.expected_rlib.exists(),
            self.artifact_dir.display()
        )
    }

    /// Contract issues that make an artifact suspicious before dynamic loading.
    pub fn lifecycle_contract_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if !self.manifest_key_matches() {
            issues.push(format!(
                "registered problem_key '{}' does not match manifest-derived key '{}'",
                self.problem_key,
                self.manifest_problem_key()
            ));
        }
        if !self.manifest_file_exists() {
            issues.push(format!(
                "generated manifest/header file is missing at '{}'",
                self.manifest_file.display()
            ));
        }
        if !self.compiled_artifact_exists() {
            issues.push(format!(
                "compiled artifact is missing; expected cdylib='{}' or rlib='{}'",
                self.expected_cdylib.display(),
                self.expected_rlib.display()
            ));
        }
        issues
    }

    /// Removes the generated crate/library directory represented by this registry entry.
    ///
    /// This is intentionally conservative: the marker manifest/header file must
    /// exist and must be located inside `crate_dir`. That makes the public cleanup
    /// operation useful for generated AOT artifacts without turning it into a
    /// general recursive delete helper.
    pub fn cleanup_generated_tree(&self) -> io::Result<bool> {
        if !self.crate_dir.exists() {
            return Ok(false);
        }
        if !self.crate_dir.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "AOT cleanup refused: crate_dir is not a directory: '{}'",
                    self.crate_dir.display()
                ),
            ));
        }
        if !self.manifest_file.exists() || !self.manifest_file.starts_with(&self.crate_dir) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "AOT cleanup refused: manifest/header marker '{}' is missing or outside generated directory '{}'",
                    self.manifest_file.display(),
                    self.crate_dir.display()
                ),
            ));
        }

        fs::remove_dir_all(&self.crate_dir)?;
        Ok(true)
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
                manifest_file: build.written.manifest_rs.clone(),
                artifact_dir: build.artifact_dir.clone(),
                expected_rlib: build.expected_rlib.clone(),
                expected_cdylib: build.expected_cdylib.clone(),
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

    /// Removes a registry entry without touching on-disk files.
    pub fn remove_by_problem_key(&mut self, problem_key: &str) -> Option<RegisteredAotArtifact> {
        let removed = self.entries_by_problem_key.remove(problem_key)?;
        self.crate_name_to_problem_key.remove(&removed.crate_name);
        Some(removed)
    }

    /// Safely removes the generated on-disk tree and then unregisters the artifact.
    pub fn cleanup_artifact_by_problem_key(&mut self, problem_key: &str) -> io::Result<bool> {
        let Some(artifact) = self.entries_by_problem_key.get(problem_key).cloned() else {
            return Ok(false);
        };
        artifact.cleanup_generated_tree()?;
        self.remove_by_problem_key(problem_key);
        Ok(true)
    }
}
//================================================================================
//TESTS
//================================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
    use crate::symbolic::codegen::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem, PreparedProblem,
    };
    use crate::symbolic::codegen::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
        AotBuildProfile, AotBuildRequest,
    };
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
        assert_eq!(registered.manifest_file, build.written.manifest_rs);
        assert!(registered.manifest_key_matches());
        assert!(registered.manifest_file_exists());
        assert!(!registered.compiled_artifact_exists());
        assert!(registered
            .lifecycle_contract_issues()
            .iter()
            .any(|issue| issue.contains("compiled artifact is missing")));
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
    fn registered_artifact_lifecycle_contract_tracks_manifest_and_outputs() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_registry_contract_fixture",
            "generated_registry_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");

        let mut registry = AotRegistry::new();
        let registered = registry
            .register_materialized_build(manifest.clone(), &build)
            .clone();

        assert!(registered.manifest_key_matches());
        assert!(registered.manifest_file_exists());
        assert!(!registered.compiled_artifact_exists());

        fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
        fs::write(&build.expected_cdylib, b"fake cdylib")
            .expect("expected cdylib should be writable");

        let refreshed = registry
            .get_by_manifest(&manifest)
            .expect("manifest lookup should still work");
        assert!(refreshed.compiled_artifact_exists());
        assert!(refreshed
            .lifecycle_contract_summary()
            .contains("manifest_key_matches=true"));
    }

    #[test]
    fn registry_cleanup_removes_generated_tree_and_registry_entry() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_registry_cleanup_fixture",
            "generated_registry_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");

        let mut registry = AotRegistry::new();
        let problem_key = registry
            .register_materialized_build(manifest.clone(), &build)
            .problem_key
            .clone();
        assert!(build.written.crate_dir.exists());
        assert!(registry.get_by_problem_key(&problem_key).is_some());

        let removed = registry
            .cleanup_artifact_by_problem_key(&problem_key)
            .expect("cleanup should succeed for a manifest-marked generated tree");

        assert!(removed);
        assert!(!build.written.crate_dir.exists());
        assert!(registry.get_by_problem_key(&problem_key).is_none());
        assert!(registry
            .get_by_crate_name("generated_registry_cleanup_fixture")
            .is_none());
    }

    #[test]
    fn artifact_cleanup_refuses_directory_without_manifest_marker() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_registry_cleanup_refusal_fixture",
            "generated_registry_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");

        fs::remove_file(&build.written.manifest_rs).expect("marker should be removable");
        let mut registry = AotRegistry::new();
        let registered = registry
            .register_materialized_build(manifest, &build)
            .clone();

        let err = registered
            .cleanup_generated_tree()
            .expect_err("cleanup must refuse an unmarked directory");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(build.written.crate_dir.exists());
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
        assert!(registry
            .get_by_crate_name("generated_registry_old")
            .is_none());

        let registered = registry
            .get_by_crate_name("generated_registry_new")
            .expect("new crate-name lookup should succeed");
        assert_eq!(registered.expected_rlib, build1.expected_rlib);
        assert_eq!(registered.cargo_command_line(), "cargo build --release");
    }
}
