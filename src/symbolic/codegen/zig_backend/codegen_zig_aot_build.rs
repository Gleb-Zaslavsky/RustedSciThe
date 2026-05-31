//! Build-step metadata for separately compiled generated Zig AOT libraries.
//!
//! Mirrors `codegen_c_aot_build.rs` but targets Zig compilation via `zig build`.

use crate::symbolic::codegen::zig_backend::codegen_zig_aot_library::{
    GeneratedZigAotLibrary, WrittenZigAotLibrary,
};
use log::info;
use std::env;
use std::io;
use std::path::{Component, Path, PathBuf, Prefix};
use std::process::Command;

fn absolute_nonverbatim(path: &Path) -> io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };

    #[cfg(target_os = "windows")]
    {
        let mut normalized = PathBuf::new();
        for component in absolute.components() {
            match component {
                Component::Prefix(prefix) => match prefix.kind() {
                    Prefix::VerbatimDisk(disk) => {
                        normalized.push(format!("{}:", char::from(disk)));
                    }
                    Prefix::VerbatimUNC(server, share) => {
                        normalized.push(format!(
                            "\\\\{}\\{}",
                            server.to_string_lossy(),
                            share.to_string_lossy()
                        ));
                    }
                    _ => normalized.push(component.as_os_str()),
                },
                _ => normalized.push(component.as_os_str()),
            }
        }
        Ok(normalized)
    }

    #[cfg(not(target_os = "windows"))]
    {
        Ok(absolute)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZigAotBuildProfile {
    Debug,
    ReleaseFast,
    ReleaseSafe,
    ReleaseSmall,
}

impl ZigAotBuildProfile {
    pub fn zig_optimize_flag(self) -> &'static str {
        match self {
            Self::Debug => "Debug",
            Self::ReleaseFast => "ReleaseFast",
            Self::ReleaseSafe => "ReleaseSafe",
            Self::ReleaseSmall => "ReleaseSmall",
        }
    }

    pub const fn target_dir_component(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::ReleaseFast => "release_fast",
            Self::ReleaseSafe => "release_safe",
            Self::ReleaseSmall => "release_small",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZigAotBuildRequest {
    pub library_spec: GeneratedZigAotLibrary,
    pub output_parent_dir: PathBuf,
    pub profile: ZigAotBuildProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZigAotBuildResult {
    pub written: WrittenZigAotLibrary,
    pub build_program: String,
    pub build_args: Vec<String>,
    pub build_env: Vec<(String, String)>,
    pub artifact_dir: PathBuf,
    pub expected_so: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutedZigAotBuild {
    pub build: ZigAotBuildResult,
    pub status_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl ZigAotBuildRequest {
    pub fn new(
        library_spec: GeneratedZigAotLibrary,
        output_parent_dir: impl Into<PathBuf>,
        profile: ZigAotBuildProfile,
    ) -> Self {
        Self {
            library_spec,
            output_parent_dir: output_parent_dir.into(),
            profile,
        }
    }

    pub fn materialize(&self) -> io::Result<ZigAotBuildResult> {
        info!(
            "Materializing Zig AOT build request for library '{}' with profile {:?}",
            self.library_spec.library_name, self.profile
        );
        let written = self.library_spec.write_to_dir(&self.output_parent_dir)?;
        let library_dir = absolute_nonverbatim(&written.library_dir)?;
        let written = WrittenZigAotLibrary {
            build_zig: library_dir.join("build.zig"),
            generated_zig: library_dir.join("generated.zig"),
            aot_interface_zig: library_dir.join("aot_interface.zig"),
            library_dir,
        };

        let expected_filename = if cfg!(target_os = "windows") {
            format!("{}.dll", self.library_spec.library_name)
        } else if cfg!(target_os = "macos") {
            format!("lib{}.dylib", self.library_spec.library_name)
        } else {
            format!("lib{}.so", self.library_spec.library_name)
        };

        // `zig build` places dynamic libraries under zig-out/bin on Windows
        // and under zig-out/lib on Unix-like platforms.
        let artifact_dir = if cfg!(target_os = "windows") {
            written.library_dir.join("zig-out").join("bin")
        } else {
            written.library_dir.join("zig-out").join("lib")
        };
        let artifact_dir = absolute_nonverbatim(&artifact_dir)?;
        let expected_so = artifact_dir.join(expected_filename);

        let build_args = vec![
            "build".to_string(),
            format!("-Doptimize={}", self.profile.zig_optimize_flag()),
        ];
        let local_cache_dir = written.library_dir.join(".zig-local-cache");
        let global_cache_dir = written.library_dir.join(".zig-global-cache");
        std::fs::create_dir_all(&local_cache_dir)?;
        std::fs::create_dir_all(&global_cache_dir)?;
        let local_cache_dir = absolute_nonverbatim(&local_cache_dir)?;
        let global_cache_dir = absolute_nonverbatim(&global_cache_dir)?;
        let build_env = vec![
            (
                "ZIG_LOCAL_CACHE_DIR".to_string(),
                local_cache_dir.to_string_lossy().into_owned(),
            ),
            (
                "ZIG_GLOBAL_CACHE_DIR".to_string(),
                global_cache_dir.to_string_lossy().into_owned(),
            ),
        ];

        info!(
            "Zig AOT build request prepared: command='zig {}', workdir='{}'",
            build_args.join(" "),
            written.library_dir.display()
        );

        Ok(ZigAotBuildResult {
            written,
            build_program: "zig".to_string(),
            build_args,
            build_env,
            artifact_dir,
            expected_so,
        })
    }
}

impl ZigAotBuildResult {
    pub fn build_command_line(&self) -> String {
        let mut parts = vec![self.build_program.clone()];
        parts.extend(self.build_args.iter().cloned());
        parts.join(" ")
    }

    pub fn build_workdir(&self) -> &Path {
        &self.written.library_dir
    }

    pub fn execute(&self) -> io::Result<ExecutedZigAotBuild> {
        info!(
            "Executing materialized Zig AOT build: command='{}', workdir='{}'",
            self.build_command_line(),
            self.build_workdir().display()
        );
        let output = Command::new(&self.build_program)
            .args(&self.build_args)
            .envs(self.build_env.iter().map(|(k, v)| (k, v)))
            .current_dir(self.build_workdir())
            .output()?;

        Ok(ExecutedZigAotBuild {
            build: self.clone(),
            status_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

impl ExecutedZigAotBuild {
    pub fn succeeded(&self) -> bool {
        self.status_code == Some(0) && self.build.expected_so.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem,
    };
    use crate::symbolic::codegen::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule};
    use crate::symbolic::symbolic_engine::Expr;
    use tempfile::tempdir;

    #[test]
    fn zig_build_request_materializes_and_produces_release_fast_command() {
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

        let module = CodegenModule::new("test_module").with_language(CodegenLanguage::Zig);
        let library_spec = GeneratedZigAotLibrary::from_prepared_dense_problem(
            "test_zig_build",
            &prepared,
            &module,
        );
        let dir = tempdir().expect("tempdir should exist");
        let request =
            ZigAotBuildRequest::new(library_spec, dir.path(), ZigAotBuildProfile::ReleaseFast);
        let result = request
            .materialize()
            .expect("Zig build request should materialize");

        assert_eq!(result.build_program, "zig");
        assert_eq!(
            result.build_args,
            vec!["build".to_string(), "-Doptimize=ReleaseFast".to_string()]
        );
        assert_eq!(
            result.build_command_line(),
            "zig build -Doptimize=ReleaseFast"
        );
        assert!(result.written.build_zig.exists());
        assert!(result.written.generated_zig.exists());
        assert!(result.written.aot_interface_zig.exists());
        assert!(result.build_workdir().is_absolute());
        assert!(result.expected_so.is_absolute());
        for (key, value) in &result.build_env {
            if key == "ZIG_LOCAL_CACHE_DIR" || key == "ZIG_GLOBAL_CACHE_DIR" {
                assert!(
                    Path::new(value).is_absolute(),
                    "{key} must be absolute so `zig build` does not resolve it relative to the generated crate"
                );
            }
        }
    }

    #[test]
    fn zig_debug_profile_uses_debug_optimize_flag() {
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

        let module = CodegenModule::new("test_module").with_language(CodegenLanguage::Zig);
        let library_spec = GeneratedZigAotLibrary::from_prepared_dense_problem(
            "test_zig_debug",
            &prepared,
            &module,
        );
        let dir = tempdir().expect("tempdir should exist");
        let request = ZigAotBuildRequest::new(library_spec, dir.path(), ZigAotBuildProfile::Debug);
        let result = request
            .materialize()
            .expect("debug Zig build should materialize");

        assert_eq!(
            result.build_args,
            vec!["build".to_string(), "-Doptimize=Debug".to_string()]
        );
    }
}
