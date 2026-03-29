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

use crate::symbolic::codegen_aot_crate::{GeneratedAotCrate, WrittenAotCrate};
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

/// One build request for a generated AOT crate.
#[derive(Debug, Clone)]
pub struct AotBuildRequest {
    /// Generated crate specification that should be written and built.
    pub crate_spec: GeneratedAotCrate,
    /// Parent directory where the generated crate directory should be created.
    pub output_parent_dir: PathBuf,
    /// Requested Cargo build profile.
    pub profile: AotBuildProfile,
}

/// Result of materializing a build request into on-disk crate files plus the
/// expected Cargo command and artifact locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotBuildResult {
    /// Materialized generated crate layout on disk.
    pub written: WrittenAotCrate,
    /// Program name expected to perform the build.
    pub cargo_program: String,
    /// Command-line arguments expected for the build.
    pub cargo_args: Vec<String>,
    /// Directory where Cargo is expected to place the built artifacts.
    pub artifact_dir: PathBuf,
    /// Expected `rlib` path for the generated crate after a successful build.
    pub expected_rlib: PathBuf,
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
        }
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
        let cargo_args = self
            .profile
            .cargo_args()
            .iter()
            .map(|arg| (*arg).to_string())
            .collect();

        let result = AotBuildResult {
            written,
            cargo_program: "cargo".to_string(),
            cargo_args,
            artifact_dir,
            expected_rlib,
        };
        info!(
            "AOT build request prepared: command='{}', workdir='{}'",
            result.cargo_command_line(),
            result.cargo_workdir().display()
        );
        Ok(result)
    }
}

impl AotBuildResult {
    /// Returns the recommended Cargo command line as a printable shell command.
    pub fn cargo_command_line(&self) -> String {
        let mut parts = vec![self.cargo_program.clone()];
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
        let cargo_program = std::env::var("CARGO").unwrap_or_else(|_| self.cargo_program.clone());
        let output = Command::new(cargo_program)
            .args(&self.cargo_args)
            .current_dir(self.cargo_workdir())
            .output()?;

        Ok(ExecutedAotBuild {
            build: self.clone(),
            status_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

impl ExecutedAotBuild {
    /// Returns `true` when Cargo reported a successful exit status and the
    /// expected `rlib` exists on disk.
    pub fn succeeded(&self) -> bool {
        self.status_code == Some(0) && self.build.expected_rlib.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(
            result.cargo_args,
            vec!["build".to_string(), "--release".to_string()]
        );
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
}
