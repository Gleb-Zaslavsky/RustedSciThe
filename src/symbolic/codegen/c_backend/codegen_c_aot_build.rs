//! Build-step metadata for separately compiled generated C AOT libraries.
//!
//! This module mirrors the Rust AOT build pipeline but targets C compilation.
//! The intended flow is:
//! 1. build a `PreparedProblem`,
//! 2. turn it into a `GeneratedCAotLibrary`,
//! 3. write that library to disk,
//! 4. hand a `CAotBuildRequest` to a higher-level orchestrator,
//! 5. run the returned Make/compiler command outside the solver hot path,
//! 6. use the resulting artifact paths/metadata to reconnect the compiled
//!    backend to the rest of the application.

use crate::symbolic::codegen::c_backend::codegen_c_aot_library::{
    GeneratedCAotLibrary, WrittenCAotLibrary,
};
use log::info;
use std::env;
use std::io;
use std::path::{Component, Path, PathBuf, Prefix};
use std::process::Command;

fn compiler_override_env_var(program: &str) -> Option<&'static str> {
    match program.to_ascii_lowercase().as_str() {
        "tcc" => Some("RUSTEDSCITHE_TCC"),
        "gcc" => Some("RUSTEDSCITHE_GCC"),
        "clang" => Some("RUSTEDSCITHE_CLANG"),
        "cl" => Some("RUSTEDSCITHE_CL"),
        "cc" => Some("RUSTEDSCITHE_CC"),
        _ => None,
    }
}

fn compiler_override(program: &str) -> Option<String> {
    compiler_override_env_var(program)
        .and_then(|key| env::var(key).ok())
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("RUSTEDSCITHE_C_COMPILER")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
}

fn compiler_command_succeeds(program: &str, probe_arg: &str) -> bool {
    let requested = compiler_override(program).unwrap_or_else(|| program.to_string());
    Command::new(requested)
        .arg(probe_arg)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn select_default_c_compiler() -> String {
    if compiler_command_succeeds("tcc", "-v") {
        return "tcc".to_string();
    }

    if cfg!(target_os = "macos") {
        if compiler_command_succeeds("clang", "-v") {
            return "clang".to_string();
        }
    }

    if cfg!(target_os = "windows") {
        if compiler_command_succeeds("gcc", "-v") {
            return "gcc".to_string();
        }
        if compiler_command_succeeds("cl", "?") {
            return "cl".to_string();
        }
    }

    if cfg!(target_os = "linux") {
        if compiler_command_succeeds("gcc", "-v") {
            return "gcc".to_string();
        }
    }

    if compiler_command_succeeds("cc", "-v") {
        return "cc".to_string();
    }

    "gcc".to_string()
}

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

/// Requested build profile for a generated C AOT library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CAotBuildProfile {
    Debug,
    Release,
}

impl CAotBuildProfile {
    /// Returns the optimization flag for this profile.
    pub fn optimization_flag(self) -> &'static str {
        match self {
            Self::Debug => "-O0",
            Self::Release => "-O3",
        }
    }

    /// Returns the target subdirectory component for this profile.
    pub const fn target_dir_component(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

/// Optional compiler overrides for generated C AOT library builds.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CAotCompileConfig {
    /// Optional C compiler override (default: gcc on Unix, cl on Windows).
    pub compiler: Option<String>,
    /// Optional extra compiler flags.
    pub extra_cflags: Vec<String>,
    /// Optional extra linker flags.
    pub extra_ldflags: Vec<String>,
}

impl CAotCompileConfig {
    /// Creates a compile config that keeps defaults unchanged.
    pub fn new() -> Self {
        Self::default()
    }

    /// Production-oriented defaults.
    pub fn production() -> Self {
        Self::new().with_compiler(select_default_c_compiler())
    }

    /// Fast-build preset for iteration speed.
    pub fn fast_build() -> Self {
        Self::new().with_extra_cflag("-O1")
    }

    /// Fastest developer-oriented preset.
    pub fn dev_fastest() -> Self {
        Self::new().with_extra_cflag("-O0")
    }

    /// Sets the compiler override.
    pub fn with_compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = Some(compiler.into());
        self
    }

    /// Adds an extra compiler flag.
    pub fn with_extra_cflag(mut self, flag: impl Into<String>) -> Self {
        self.extra_cflags.push(flag.into());
        self
    }

    /// Adds an extra linker flag.
    pub fn with_extra_ldflag(mut self, flag: impl Into<String>) -> Self {
        self.extra_ldflags.push(flag.into());
        self
    }

    /// Short stable label for diagnostics.
    pub fn label(&self) -> String {
        let compiler = self
            .compiler
            .as_ref()
            .map(|c| c.as_str())
            .unwrap_or("default");
        format!("{compiler}")
    }
}

/// One build request for a generated C AOT library.
#[derive(Debug, Clone)]
pub struct CAotBuildRequest {
    /// Generated library specification that should be written and built.
    pub library_spec: GeneratedCAotLibrary,
    /// Parent directory where the generated library directory should be created.
    pub output_parent_dir: PathBuf,
    /// Requested build profile.
    pub profile: CAotBuildProfile,
    /// Optional compile-time compiler/flag overrides.
    pub compile_config: CAotCompileConfig,
}

/// Result of materializing a build request into on-disk library files plus the
/// expected Make command and artifact locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CAotBuildResult {
    /// Materialized generated library layout on disk.
    pub written: WrittenCAotLibrary,
    /// Program name expected to perform the build (typically "make").
    pub build_program: String,
    /// Command-line arguments expected for the build.
    pub build_args: Vec<String>,
    /// Optional extra environment variables for the build command.
    pub build_env: Vec<(String, String)>,
    /// Directory where the build system places the built artifacts.
    pub artifact_dir: PathBuf,
    /// Expected shared library path after a successful build.
    pub expected_so: PathBuf,
}

/// Result of executing the build step for a materialized generated C library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutedCAotBuild {
    /// Original materialized build metadata.
    pub build: CAotBuildResult,
    /// Exit status code reported by the build process.
    pub status_code: Option<i32>,
    /// Captured standard output from the build process.
    pub stdout: String,
    /// Captured standard error from the build process.
    pub stderr: String,
}

impl CAotBuildRequest {
    /// Creates a new C AOT build request.
    pub fn new(
        library_spec: GeneratedCAotLibrary,
        output_parent_dir: impl Into<PathBuf>,
        profile: CAotBuildProfile,
    ) -> Self {
        Self {
            library_spec,
            output_parent_dir: output_parent_dir.into(),
            profile,
            compile_config: CAotCompileConfig::default(),
        }
    }

    /// Overrides compiler/flag settings for the generated library build.
    pub fn with_compile_config(mut self, compile_config: CAotCompileConfig) -> Self {
        self.compile_config = compile_config;
        self
    }

    /// Materializes the generated library and computes the build command plus
    /// expected artifact locations.
    pub fn materialize(&self) -> io::Result<CAotBuildResult> {
        info!(
            "Materializing C AOT build request for library '{}' with profile {:?}",
            self.library_spec.library_name, self.profile
        );
        let written = self.library_spec.write_to_dir(&self.output_parent_dir)?;
        let artifact_dir = written
            .library_dir
            .join("build")
            .join(self.profile.target_dir_component());
        std::fs::create_dir_all(&artifact_dir)?;
        let artifact_dir = absolute_nonverbatim(&artifact_dir)?;

        let expected_filename = if cfg!(target_os = "windows") {
            format!("lib{}.dll", self.library_spec.library_name)
        } else if cfg!(target_os = "macos") {
            format!("lib{}.dylib", self.library_spec.library_name)
        } else {
            format!("lib{}.so", self.library_spec.library_name)
        };
        let expected_so = artifact_dir.join(expected_filename);

        let requested_program = self
            .compile_config
            .compiler
            .clone()
            .unwrap_or_else(select_default_c_compiler);
        // Bare compiler names are resolved by process spawning itself. Running
        // `where`/`which` here adds a full subprocess to every cold AOT
        // materialization, which is significant for fast compilers such as
        // tcc. Explicit environment overrides are still honored verbatim.
        let build_program = compiler_override(&requested_program).unwrap_or(requested_program);
        let build_program_path = PathBuf::from(&build_program);
        let mut build_args = vec![
            self.profile.optimization_flag().to_string(),
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-Wall".to_string(),
            "-Wextra".to_string(),
            "-o".to_string(),
            expected_so.to_string_lossy().into_owned(),
            "generated.c".to_string(),
            "aot_interface.c".to_string(),
        ];
        if !cfg!(target_os = "windows") {
            build_args.push("-lm".to_string());
        }

        let mut build_env = Vec::new();
        if let Some(path) = (build_program_path.components().count() > 1)
            .then(|| build_program_path.parent())
            .flatten()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
        {
            let mut merged = path.to_string_lossy().into_owned();
            if let Some(existing) = env::var_os("PATH") {
                let existing = existing.to_string_lossy();
                if !existing.is_empty() {
                    merged.push(if cfg!(target_os = "windows") {
                        ';'
                    } else {
                        ':'
                    });
                    merged.push_str(&existing);
                }
            }
            build_env.push(("PATH".to_string(), merged));
        }
        if !self.compile_config.extra_cflags.is_empty() {
            build_args.extend(self.compile_config.extra_cflags.iter().cloned());
        }
        if !self.compile_config.extra_ldflags.is_empty() {
            build_args.extend(self.compile_config.extra_ldflags.iter().cloned());
        }

        let result = CAotBuildResult {
            written,
            build_program,
            build_args,
            build_env,
            artifact_dir,
            expected_so,
        };
        info!(
            "C AOT build request prepared: command='{}', workdir='{}'",
            result.build_command_line(),
            result.build_workdir().display()
        );
        Ok(result)
    }
}

impl CAotBuildResult {
    /// Returns the recommended build command line as a printable shell command.
    pub fn build_command_line(&self) -> String {
        let mut parts = vec![self.build_program.clone()];
        parts.extend(self.build_args.iter().cloned());
        parts.join(" ")
    }

    /// Returns the working directory where the build command should run.
    pub fn build_workdir(&self) -> &Path {
        &self.written.library_dir
    }

    /// Executes the materialized build command and captures its output.
    pub fn execute(&self) -> io::Result<ExecutedCAotBuild> {
        info!(
            "Executing materialized C AOT build: command='{}', workdir='{}'",
            self.build_command_line(),
            self.build_workdir().display()
        );
        let output = Command::new(&self.build_program)
            .args(&self.build_args)
            .envs(self.build_env.iter().map(|(k, v)| (k, v)))
            .current_dir(self.build_workdir())
            .output()?;

        Ok(ExecutedCAotBuild {
            build: self.clone(),
            status_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

impl ExecutedCAotBuild {
    /// Returns `true` when the build reported a successful exit status and the
    /// expected shared library exists on disk.
    pub fn succeeded(&self) -> bool {
        self.status_code == Some(0) && self.build.expected_so.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule};
    use crate::symbolic::codegen::c_backend::codegen_c_aot_library::GeneratedCAotLibrary;
    use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
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
    fn c_build_request_materializes_generated_library_and_release_command() {
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

        let library_spec =
            GeneratedCAotLibrary::from_prepared_dense_problem("test_c_lib", &prepared, &module);
        let dir = tempdir().expect("tempdir should exist");

        let request = CAotBuildRequest::new(library_spec, dir.path(), CAotBuildProfile::Release);
        let result = request
            .materialize()
            .expect("C build request should materialize generated library");

        assert!(!result.build_program.is_empty());
        assert!(result.build_args.iter().any(|arg| arg == "-shared"));
        assert!(
            result
                .build_args
                .iter()
                .any(|arg| arg.ends_with("generated.c"))
        );
        assert!(result.written.makefile.exists());
        assert!(result.written.generated_c.exists());
        assert!(result.written.generated_h.exists());
        assert_eq!(result.build_workdir(), result.written.library_dir.as_path());
    }

    #[test]
    fn debug_profile_points_to_debug_target_dir() {
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

        let mut module = CodegenModule::new("test_module").with_language(CodegenLanguage::C);
        for chunk in &prepared.residual_plan.chunks {
            module.push_residual_block_plan(&chunk.plan);
        }
        for chunk in &prepared.jacobian_plan.chunks {
            module.push_dense_jacobian_plan(&chunk.plan);
        }

        let library_spec = GeneratedCAotLibrary::from_prepared_dense_problem(
            "test_c_lib_debug",
            &prepared,
            &module,
        );
        let dir = tempdir().expect("tempdir should exist");

        let request = CAotBuildRequest::new(library_spec, dir.path(), CAotBuildProfile::Debug);
        let result = request
            .materialize()
            .expect("debug C build request should materialize generated library");

        assert!(
            result
                .build_args
                .iter()
                .any(|arg| arg == CAotBuildProfile::Debug.optimization_flag())
        );
        assert!(
            result
                .artifact_dir
                .ends_with(Path::new("build").join("debug"))
        );
    }
}
