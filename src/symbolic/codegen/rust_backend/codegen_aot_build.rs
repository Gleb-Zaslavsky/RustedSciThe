//! Build-step metadata for separately compiled generated AOT crates.
//!
//! This module does not execute Cargo directly by default. Its purpose is to
//! describe the boundary between:
//! - code generation inside RustedSciThe,
//! - and the external orchestration step that compiles the generated crate in
//!   release mode and then reconnects its artifacts to the solver pipeline.
//!
//! In practical terms, the intended flow is:
//! 1. build a `PreparedProblem`,
//! 2. turn it into a `GeneratedAotCrate`,
//! 3. write that crate to disk,
//! 4. hand an `AotBuildRequest` to a higher-level orchestrator,
//! 5. run the returned Cargo command outside the solver hot path,
//! 6. use the resulting artifact paths/metadata to reconnect the compiled
//!    backend to the rest of the application.

use crate::symbolic::codegen::codegen_aot_lifecycle::{
    AotArtifactInspection, AotFailureDiagnostics, AotFailureInjection, AotFailureKind,
    AotLifecycleError, AotLifecycleStage, quarantine_generated_tree,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_crate::{
    GeneratedAotCrate, WrittenAotCrate,
};
use libloading::library_filename;
use log::info;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Requested build profile for a generated AOT crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotBuildProfile {
    Debug,
    Release,
}

impl AotBuildProfile {
    /// Returns the Cargo CLI flag list for this profile.
    pub fn cargo_args(self) -> &'static [&'static str] {
        match self {
            Self::Debug => &["build"],
            Self::Release => &["build", "--release"],
        }
    }

    /// Returns the target subdirectory written by Cargo.
    pub const fn target_dir_component(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

/// Requested optimization level for a generated AOT crate build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AotOptimizationLevel {
    /// Keep Cargo profile defaults unchanged.
    #[default]
    Default,
    O0,
    O1,
    O2,
    O3,
    Os,
    Oz,
}

impl AotOptimizationLevel {
    /// Returns the rustc `-C opt-level=...` payload when an override is requested.
    pub fn rustc_value(self) -> Option<&'static str> {
        match self {
            Self::Default => None,
            Self::O0 => Some("0"),
            Self::O1 => Some("1"),
            Self::O2 => Some("2"),
            Self::O3 => Some("3"),
            Self::Os => Some("s"),
            Self::Oz => Some("z"),
        }
    }
}

/// Optional rustc/codegen overrides for generated AOT crate builds.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AotCompileConfig {
    /// Optional optimization level override. `Default` keeps Cargo profile behavior.
    pub optimization: AotOptimizationLevel,
    /// Optional rustc codegen-units override.
    pub codegen_units: Option<usize>,
}

impl AotCompileConfig {
    /// Creates a compile config that keeps Cargo defaults unchanged.
    pub fn new() -> Self {
        Self::default()
    }

    /// Production-oriented defaults that preserve the current highest-quality release build.
    pub fn production() -> Self {
        Self::new()
    }

    /// Faster cold-build preset intended for practical AOT workflows where compile latency
    /// matters more than extracting every last percent of runtime performance.
    pub fn fast_build() -> Self {
        Self::new()
            .with_optimization(AotOptimizationLevel::O1)
            .with_codegen_units(16)
    }

    /// Fastest developer-oriented preset for diagnostics and iteration speed.
    pub fn dev_fastest() -> Self {
        Self::new()
            .with_optimization(AotOptimizationLevel::O0)
            .with_codegen_units(16)
    }

    /// Sets the optimization level override.
    pub fn with_optimization(mut self, optimization: AotOptimizationLevel) -> Self {
        self.optimization = optimization;
        self
    }

    /// Sets the codegen-units override.
    pub fn with_codegen_units(mut self, codegen_units: usize) -> Self {
        self.codegen_units = Some(codegen_units);
        self
    }

    /// Returns true when this config does not request any rustc overrides.
    pub fn is_default(&self) -> bool {
        self.optimization == AotOptimizationLevel::Default && self.codegen_units.is_none()
    }

    /// Builds rustc flag fragments for this compile configuration.
    pub fn rustflags_fragments(&self) -> Vec<String> {
        let mut flags = Vec::new();
        if let Some(level) = self.optimization.rustc_value() {
            flags.push(format!("-C opt-level={level}"));
        }
        if let Some(codegen_units) = self.codegen_units {
            flags.push(format!("-C codegen-units={codegen_units}"));
        }
        flags
    }

    /// Short stable label for diagnostics and user-facing reports.
    pub fn label(&self) -> String {
        let opt = match self.optimization {
            AotOptimizationLevel::Default => "default".to_string(),
            AotOptimizationLevel::O0 => "O0".to_string(),
            AotOptimizationLevel::O1 => "O1".to_string(),
            AotOptimizationLevel::O2 => "O2".to_string(),
            AotOptimizationLevel::O3 => "O3".to_string(),
            AotOptimizationLevel::Os => "Os".to_string(),
            AotOptimizationLevel::Oz => "Oz".to_string(),
        };
        match self.codegen_units {
            Some(units) => format!("{opt}-cgu{units}"),
            None => opt,
        }
    }
}

/// One build request for a generated AOT crate.
#[derive(Debug, Clone)]
pub struct AotBuildRequest {
    /// Generated crate specification that should be written and built.
    pub crate_spec: GeneratedAotCrate,
    /// Parent directory where the generated crate directory should be created.
    pub output_parent_dir: PathBuf,
    /// Requested Cargo build profile.
    pub profile: AotBuildProfile,
    /// Optional Rust toolchain override, e.g. `nightly`.
    pub toolchain: Option<String>,
    /// Optional extra `RUSTFLAGS` passed to the Cargo build process.
    pub rustflags: Option<String>,
    /// Optional extra Cargo CLI arguments appended after the build profile args.
    pub extra_cargo_args: Vec<String>,
    /// Optional compile-time rustc/codegen overrides.
    pub compile_config: AotCompileConfig,
}

/// Result of materializing a build request into on-disk crate files plus the
/// expected Cargo command and artifact locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotBuildResult {
    /// Materialized generated crate layout on disk.
    pub written: WrittenAotCrate,
    /// Program name expected to perform the build.
    pub cargo_program: String,
    /// Optional wrapper program that should invoke Cargo, e.g. `rustup`.
    pub cargo_wrapper_program: Option<String>,
    /// Command-line arguments expected for the build.
    pub cargo_args: Vec<String>,
    /// Optional extra environment variables for the build command.
    pub cargo_env: Vec<(String, String)>,
    /// Directory where Cargo is expected to place the built artifacts.
    pub artifact_dir: PathBuf,
    /// Expected `rlib` path for the generated crate after a successful build.
    pub expected_rlib: PathBuf,
    /// Expected dynamic library path for the generated crate after a successful build.
    pub expected_cdylib: PathBuf,
}

/// Result of executing the Cargo build step for a materialized generated crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutedAotBuild {
    /// Original materialized build metadata.
    pub build: AotBuildResult,
    /// Exit status code reported by Cargo.
    pub status_code: Option<i32>,
    /// Captured standard output from the build process.
    pub stdout: String,
    /// Captured standard error from the build process.
    pub stderr: String,
}

impl AotBuildRequest {
    /// Creates a new AOT build request.
    pub fn new(
        crate_spec: GeneratedAotCrate,
        output_parent_dir: impl Into<PathBuf>,
        profile: AotBuildProfile,
    ) -> Self {
        Self {
            crate_spec,
            output_parent_dir: output_parent_dir.into(),
            profile,
            toolchain: None,
            rustflags: None,
            extra_cargo_args: Vec::new(),
            compile_config: AotCompileConfig::default(),
        }
    }

    /// Overrides the Cargo toolchain, e.g. `nightly`.
    pub fn with_toolchain(mut self, toolchain: impl Into<String>) -> Self {
        self.toolchain = Some(toolchain.into());
        self
    }

    /// Appends extra `RUSTFLAGS` to the build environment.
    pub fn with_rustflags(mut self, rustflags: impl Into<String>) -> Self {
        self.rustflags = Some(rustflags.into());
        self
    }

    /// Appends extra Cargo arguments such as `--timings`.
    pub fn with_extra_cargo_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_cargo_args.push(arg.into());
        self
    }

    /// Overrides rustc/codegen settings for the generated crate build.
    pub fn with_compile_config(mut self, compile_config: AotCompileConfig) -> Self {
        self.compile_config = compile_config;
        self
    }

    /// Materializes the generated crate and computes the Cargo command plus
    /// expected artifact locations.
    pub fn materialize(&self) -> io::Result<AotBuildResult> {
        info!(
            "Materializing AOT build request for crate '{}' with profile {:?}",
            self.crate_spec.crate_name, self.profile
        );
        let written = self.crate_spec.write_to_dir(&self.output_parent_dir)?;
        let artifact_dir = written
            .crate_dir
            .join("target")
            .join(self.profile.target_dir_component());
        let expected_rlib = artifact_dir.join(format!("lib{}.rlib", self.crate_spec.crate_name));
        let expected_cdylib = artifact_dir.join(library_filename(&self.crate_spec.crate_name));
        let mut cargo_program = "cargo".to_string();
        let mut cargo_wrapper_program = None;
        let mut cargo_args = Vec::new();
        if let Some(toolchain) = &self.toolchain {
            cargo_wrapper_program = Some("rustup".to_string());
            cargo_program = "cargo".to_string();
            cargo_args.push("run".to_string());
            cargo_args.push(toolchain.clone());
            cargo_args.push("cargo".to_string());
        }
        cargo_args.extend(
            self.profile
                .cargo_args()
                .iter()
                .map(|arg| (*arg).to_string()),
        );
        cargo_args.extend(self.extra_cargo_args.iter().cloned());

        let mut cargo_env = Vec::new();
        let mut rustflags_parts = Vec::new();
        if let Some(rustflags) = &self.rustflags {
            rustflags_parts.push(rustflags.clone());
        }
        rustflags_parts.extend(self.compile_config.rustflags_fragments());
        if !rustflags_parts.is_empty() {
            cargo_env.push(("RUSTFLAGS".to_string(), rustflags_parts.join(" ")));
        }

        let result = AotBuildResult {
            written,
            cargo_program,
            cargo_wrapper_program,
            cargo_args,
            cargo_env,
            artifact_dir,
            expected_rlib,
            expected_cdylib,
        };
        info!(
            "AOT build request prepared: command='{}', workdir='{}'",
            result.cargo_command_line(),
            result.cargo_workdir().display()
        );
        Ok(result)
    }

    /// Materializes and executes a build with an explicit quarantine boundary.
    ///
    /// A failed compiler/link step may leave a directory which contains a
    /// marker or partial output but is not reusable. Before a retry this
    /// method moves that tree aside and materializes a fresh crate at the
    /// original path. The operation is intentionally cold-path only.
    pub fn materialize_and_execute_with_retry(
        &self,
        artifact_key: impl Into<String>,
        attempts: u32,
        injection: AotFailureInjection,
    ) -> Result<ExecutedAotBuild, AotLifecycleError> {
        let artifact_key = artifact_key.into();
        let total = attempts.max(1);
        let mut last_error = None;
        let mut quarantine_attempted = false;
        let mut cleanup_completed = false;

        for attempt in 1..=total {
            let build = self.materialize().map_err(|error| {
                let mut diagnostics = AotFailureDiagnostics::new(
                    AotLifecycleStage::Materialized,
                    AotFailureKind::Io,
                    &artifact_key,
                    error.to_string(),
                );
                diagnostics.attempts = attempt;
                AotLifecycleError::new(diagnostics)
            })?;

            match build.execute_with_lifecycle(&artifact_key, injection) {
                Ok(result) => return Ok(result),
                Err(mut error) => {
                    error.diagnostics.attempts = attempt;
                    if attempt < total {
                        let inspection = AotArtifactInspection::inspect(
                            &build.written.crate_dir,
                            &build.written.manifest_rs,
                            &build.expected_rlib,
                            &build.expected_cdylib,
                        );
                        error.diagnostics.inspection = Some(inspection.clone());
                        match quarantine_generated_tree(&build.written.crate_dir, &artifact_key) {
                            Ok(Some(_)) => {
                                quarantine_attempted = true;
                                cleanup_completed = true;
                            }
                            Ok(None) => {}
                            Err(quarantine_error) => return Err(quarantine_error),
                        }
                    }
                    last_error = Some(error);
                }
            }
        }

        let mut error = last_error.expect("at least one typed build attempt exists");
        error.diagnostics.attempts = total;
        error.diagnostics.quarantine_attempted = quarantine_attempted;
        error.diagnostics.cleanup_completed = cleanup_completed;
        error.diagnostics.root_kind = Some(error.diagnostics.kind);
        error.diagnostics.kind = AotFailureKind::RetryExhausted;
        Err(error)
    }
}

impl AotBuildResult {
    /// Returns the recommended Cargo command line as a printable shell command.
    pub fn cargo_command_line(&self) -> String {
        let mut parts = vec![
            self.cargo_wrapper_program
                .clone()
                .unwrap_or_else(|| self.cargo_program.clone()),
        ];
        parts.extend(self.cargo_args.iter().cloned());
        parts.join(" ")
    }

    /// Returns the working directory where the build command should run.
    pub fn cargo_workdir(&self) -> &Path {
        &self.written.crate_dir
    }

    /// Executes the materialized Cargo build command and captures its output.
    pub fn execute(&self) -> io::Result<ExecutedAotBuild> {
        info!(
            "Executing materialized AOT build: command='{}', workdir='{}'",
            self.cargo_command_line(),
            self.cargo_workdir().display()
        );
        let command_program = if self.cargo_wrapper_program.is_some() {
            self.cargo_wrapper_program.clone().unwrap()
        } else {
            std::env::var("CARGO").unwrap_or_else(|_| self.cargo_program.clone())
        };
        let output = Command::new(command_program)
            .args(&self.cargo_args)
            .envs(self.cargo_env.iter().map(|(k, v)| (k, v)))
            .current_dir(self.cargo_workdir())
            .output()?;

        Ok(ExecutedAotBuild {
            build: self.clone(),
            status_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    /// Executes the build through the typed lifecycle boundary.
    ///
    /// The compatibility `execute()` method remains available, but new
    /// orchestration should use this method so compiler exit status, missing
    /// outputs and injected lock/link failures all carry the same partial
    /// artifact inspection. `injection` is also used by debug fault-injection
    /// tests; the default `None` adds no work to callback execution.
    pub fn execute_with_lifecycle(
        &self,
        artifact_key: impl Into<String>,
        injection: AotFailureInjection,
    ) -> Result<ExecutedAotBuild, AotLifecycleError> {
        let artifact_key = artifact_key.into();
        let inspection = || {
            AotArtifactInspection::inspect(
                &self.build_crate_dir(),
                &self.build_manifest_path(),
                &self.expected_rlib,
                &self.expected_cdylib,
            )
        };
        if injection != AotFailureInjection::None {
            let (stage, kind, detail) = match injection {
                AotFailureInjection::Compiler => (
                    AotLifecycleStage::Build,
                    AotFailureKind::Compiler,
                    "compiler failure injected for lifecycle test",
                ),
                AotFailureInjection::PartialArtifact => (
                    AotLifecycleStage::Materialized,
                    AotFailureKind::PartialArtifact,
                    "partial artifact injected for lifecycle test",
                ),
                AotFailureInjection::StaleArtifact => (
                    AotLifecycleStage::Materialized,
                    AotFailureKind::StaleArtifact,
                    "stale artifact injected for lifecycle test",
                ),
                AotFailureInjection::Lock => (
                    AotLifecycleStage::Materialized,
                    AotFailureKind::Lock,
                    "artifact lock failure injected for lifecycle test",
                ),
                AotFailureInjection::Link => (
                    AotLifecycleStage::Link,
                    AotFailureKind::Link,
                    "link failure injected for lifecycle test",
                ),
                AotFailureInjection::None => unreachable!(),
            };
            let mut diagnostics = AotFailureDiagnostics::new(stage, kind, &artifact_key, detail);
            diagnostics.inspection = Some(inspection());
            return Err(AotLifecycleError::new(diagnostics));
        }

        let result = self.execute().map_err(|error| {
            let mut diagnostics = AotFailureDiagnostics::new(
                AotLifecycleStage::Build,
                AotFailureKind::Io,
                &artifact_key,
                error.to_string(),
            );
            diagnostics.inspection = Some(inspection());
            AotLifecycleError::new(diagnostics)
        })?;
        if !result.succeeded() {
            let mut diagnostics = AotFailureDiagnostics::new(
                AotLifecycleStage::Build,
                AotFailureKind::Compiler,
                &artifact_key,
                format!(
                    "compiler exited with status {:?}; stderr: {}",
                    result.status_code,
                    result.stderr.trim()
                ),
            );
            diagnostics.inspection = Some(inspection());
            return Err(AotLifecycleError::new(diagnostics));
        }
        Ok(result)
    }

    /// Retries a typed build without hiding the attempt count on failure.
    /// The generated tree is not overwritten or silently reused by this
    /// helper; callers should quarantine a non-ready inspection before a new
    /// materialization.
    pub fn execute_with_retry(
        &self,
        artifact_key: impl Into<String>,
        attempts: u32,
        injection: AotFailureInjection,
    ) -> Result<ExecutedAotBuild, AotLifecycleError> {
        let artifact_key = artifact_key.into();
        let total = attempts.max(1);
        let mut last_error = None;
        for _ in 0..total {
            match self.execute_with_lifecycle(&artifact_key, injection) {
                Ok(result) => return Ok(result),
                Err(error) => last_error = Some(error),
            }
        }
        let mut error = last_error.expect("at least one typed build attempt exists");
        error.diagnostics.attempts = total;
        error.diagnostics.root_kind = Some(error.diagnostics.kind);
        error.diagnostics.kind = AotFailureKind::RetryExhausted;
        Err(error)
    }

    fn build_crate_dir(&self) -> &Path {
        &self.written.crate_dir
    }

    fn build_manifest_path(&self) -> &Path {
        &self.written.manifest_rs
    }
}

impl ExecutedAotBuild {
    /// Returns `true` when Cargo reported a successful exit status and the
    /// expected `rlib` exists on disk.
    pub fn succeeded(&self) -> bool {
        self.status_code == Some(0)
            && (self.build.expected_rlib.exists() || self.build.expected_cdylib.exists())
    }
}

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
    fn build_request_materializes_generated_crate_and_release_command() {
        let prepared = sample_prepared_problem();
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_build_fixture",
            "generated_build_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");

        let request = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Release);
        let result = request
            .materialize()
            .expect("build request should materialize generated crate");

        assert_eq!(result.cargo_program, "cargo");
        assert_eq!(result.cargo_wrapper_program, None);
        assert_eq!(
            result.cargo_args,
            vec!["build".to_string(), "--release".to_string()]
        );
        assert!(result.cargo_env.is_empty());
        assert_eq!(result.cargo_command_line(), "cargo build --release");
        assert!(result.written.cargo_toml.exists());
        assert_eq!(
            result
                .expected_rlib
                .file_name()
                .and_then(|name| name.to_str()),
            Some("libgenerated_build_fixture.rlib")
        );
        assert_eq!(result.cargo_workdir(), result.written.crate_dir.as_path());
    }

    #[test]
    fn debug_profile_points_to_debug_target_dir() {
        let prepared = sample_prepared_problem();
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_debug_fixture",
            "generated_debug_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");

        let request = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug);
        let result = request
            .materialize()
            .expect("debug build request should materialize generated crate");

        assert_eq!(result.cargo_args, vec!["build".to_string()]);
        assert_eq!(result.cargo_wrapper_program, None);
        assert!(result.cargo_env.is_empty());
        assert!(
            result
                .artifact_dir
                .ends_with(Path::new("target").join("debug"))
        );
        assert_eq!(
            result
                .expected_rlib
                .file_name()
                .and_then(|name| name.to_str()),
            Some("libgenerated_debug_fixture.rlib")
        );
    }

    #[test]
    fn build_request_supports_toolchain_and_rustflags() {
        let prepared = sample_prepared_problem();
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_nightly_fixture",
            "generated_nightly_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");

        let request = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Release)
            .with_toolchain("nightly")
            .with_rustflags("-Z threads=8")
            .with_extra_cargo_arg("--timings");
        let result = request
            .materialize()
            .expect("nightly build request should materialize generated crate");

        assert_eq!(
            result.cargo_args,
            vec![
                "run".to_string(),
                "nightly".to_string(),
                "cargo".to_string(),
                "build".to_string(),
                "--release".to_string(),
                "--timings".to_string()
            ]
        );
        assert_eq!(result.cargo_program, "cargo");
        assert_eq!(result.cargo_wrapper_program, Some("rustup".to_string()));
        assert_eq!(
            result.cargo_env,
            vec![("RUSTFLAGS".to_string(), "-Z threads=8".to_string())]
        );
        assert_eq!(
            result.cargo_command_line(),
            "rustup run nightly cargo build --release --timings"
        );
    }

    #[test]
    fn typed_fault_injection_preserves_partial_build_diagnostics() {
        let prepared = sample_prepared_problem();
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_fault_fixture",
            "generated_fault_module",
            &prepared,
        );
        let dir = tempdir().expect("temporary directory should exist");
        let result = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");

        let error = result
            .execute_with_lifecycle("fault-key", AotFailureInjection::Compiler)
            .expect_err("compiler injection must fail before spawning Cargo");
        assert_eq!(error.diagnostics.kind, AotFailureKind::Compiler);
        assert_eq!(error.diagnostics.stage, AotLifecycleStage::Build);
        assert_eq!(error.diagnostics.artifact_key, "fault-key");
        assert_eq!(
            error
                .diagnostics
                .inspection
                .expect("partial inspection should be retained")
                .state,
            crate::symbolic::codegen::codegen_aot_lifecycle::AotArtifactState::Materialized
        );
    }

    #[test]
    fn stale_artifact_fault_is_distinct_from_partial_artifact() {
        let prepared = sample_prepared_problem();
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_stale_fixture",
            "generated_stale_module",
            &prepared,
        );
        let dir = tempdir().expect("temporary directory should exist");
        let result = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");

        let error = result
            .execute_with_lifecycle("stale-key", AotFailureInjection::StaleArtifact)
            .expect_err("stale artifact injection must fail before spawning Cargo");
        assert_eq!(error.diagnostics.kind, AotFailureKind::StaleArtifact);
        assert_eq!(error.diagnostics.stage, AotLifecycleStage::Materialized);
        assert_eq!(error.diagnostics.attempts, 0);
        assert!(error.diagnostics.inspection.is_some());
    }

    #[test]
    fn typed_retry_reports_attempt_count_without_reusing_a_partial_build() {
        let prepared = sample_prepared_problem();
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_retry_fixture",
            "generated_retry_module",
            &prepared,
        );
        let dir = tempdir().expect("temporary directory should exist");
        let result = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");

        let error = result
            .execute_with_retry("retry-key", 3, AotFailureInjection::Link)
            .expect_err("link injection must exhaust retries");
        assert_eq!(error.diagnostics.kind, AotFailureKind::RetryExhausted);
        assert_eq!(error.diagnostics.root_kind, Some(AotFailureKind::Link));
        assert_eq!(error.diagnostics.attempts, 3);
        assert!(error.diagnostics.inspection.is_some());
    }

    #[test]
    fn materialized_retry_quarantines_the_previous_tree_before_rebuild() {
        let prepared = sample_prepared_problem();
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_quarantine_retry_fixture",
            "generated_quarantine_retry_module",
            &prepared,
        );
        let dir = tempdir().expect("temporary directory should exist");

        let error = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize_and_execute_with_retry(
                "quarantine-retry-key",
                2,
                AotFailureInjection::PartialArtifact,
            )
            .expect_err("fault injection must exhaust the rebuild attempts");

        assert_eq!(error.diagnostics.kind, AotFailureKind::RetryExhausted);
        assert_eq!(
            error.diagnostics.root_kind,
            Some(AotFailureKind::PartialArtifact)
        );
        assert_eq!(error.diagnostics.attempts, 2);
        assert!(error.diagnostics.quarantine_attempted);
        assert!(error.diagnostics.cleanup_completed);
        assert!(
            dir.path()
                .read_dir()
                .expect("temporary directory should be readable")
                .filter_map(Result::ok)
                .any(|entry| entry
                    .file_name()
                    .to_string_lossy()
                    .contains("generated_quarantine_retry_fixture.quarantine-"))
        );
    }
}
