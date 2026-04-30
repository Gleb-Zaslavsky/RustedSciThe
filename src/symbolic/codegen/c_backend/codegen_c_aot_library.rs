//! C library writer for separately compiled generated AOT backends.
//!
//! This module mirrors `codegen_aot_crate.rs` but generates C code instead of Rust.
//! The generated C library exposes the same FFI interface as the Rust version,
//! making it a drop-in replacement from the solver's perspective.
//!
//! Generated structure:
//! ```text
//! <crate_name>/
//!   ├── Makefile          (build configuration)
//!   ├── generated.c       (emitted C functions)
//!   ├── generated.h       (function declarations)
//!   ├── manifest.h        (metadata constants)
//!   └── aot_interface.c   (FFI wrapper matching Rust interface)
//! ```

use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::{
    PreparedBandedProblem, PreparedDenseProblem, PreparedProblem, PreparedSparseProblem,
};
use log::info;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
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

/// Description of one generated C AOT library ready to be written to disk.
#[derive(Debug, Clone)]
pub struct GeneratedCAotLibrary {
    /// Library name to be written on disk.
    pub library_name: String,
    /// Emitted generated C source code.
    pub c_source: String,
    /// Emitted generated C header code.
    pub c_header: String,
    /// Owned solver/codegen metadata exported alongside the source.
    pub manifest: PreparedProblemManifest,
    /// C compiler to use (gcc, clang, etc.)
    pub compiler: String,
    /// Optimization level (-O0, -O1, -O2, -O3, -Os)
    pub optimization: String,
}

/// Paths written for one generated C AOT library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenCAotLibrary {
    /// Root directory of the generated library.
    pub library_dir: PathBuf,
    /// Path to the generated Makefile.
    pub makefile: PathBuf,
    /// Path to the generated C source file.
    pub generated_c: PathBuf,
    /// Path to the generated C header file.
    pub generated_h: PathBuf,
    /// Path to the generated manifest header.
    pub manifest_h: PathBuf,
    /// Path to the FFI interface source.
    pub aot_interface_c: PathBuf,
    /// Path to the FFI interface header.
    pub aot_interface_h: PathBuf,
}

impl GeneratedCAotLibrary {
    /// Creates a new generated C AOT library description.
    pub fn new(
        library_name: impl Into<String>,
        c_source: impl Into<String>,
        c_header: impl Into<String>,
        manifest: PreparedProblemManifest,
    ) -> Self {
        let library_name = library_name.into();
        validate_library_name(&library_name);

        Self {
            library_name,
            c_source: c_source.into(),
            c_header: c_header.into(),
            manifest,
            compiler: select_default_c_compiler(),
            optimization: "-O3".to_string(),
        }
    }

    /// Builds a generated C library directly from an emitted `CodegenModule`.
    pub fn from_codegen_module(
        library_name: impl Into<String>,
        module: &CodegenModule,
        manifest: PreparedProblemManifest,
    ) -> Self {
        let mut module = module.clone();
        module.set_language(CodegenLanguage::C);
        Self::new(
            library_name,
            module.emit_source(),
            module.emit_c_header(),
            manifest,
        )
    }

    /// Builds a generated C library from a prepared dense problem.
    pub fn from_prepared_dense_problem(
        library_name: impl Into<String>,
        problem: &PreparedDenseProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    /// Builds a generated C library from a prepared sparse problem.
    pub fn from_prepared_sparse_problem(
        library_name: impl Into<String>,
        problem: &PreparedSparseProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    /// Builds a generated C library from a prepared banded problem.
    pub fn from_prepared_banded_problem(
        library_name: impl Into<String>,
        problem: &PreparedBandedProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    /// Builds a generated C library from any prepared problem.
    pub fn from_prepared_problem(
        library_name: impl Into<String>,
        problem: &PreparedProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    /// Sets the C compiler to use.
    pub fn with_compiler(mut self, compiler: impl Into<String>) -> Self {
        self.compiler = compiler.into();
        self
    }

    /// Sets the optimization level.
    pub fn with_optimization(mut self, optimization: impl Into<String>) -> Self {
        self.optimization = optimization.into();
        self
    }

    /// Writes the generated C library under `parent_dir/<library_name>/`.
    pub fn write_to_dir<P: AsRef<Path>>(&self, parent_dir: P) -> io::Result<WrittenCAotLibrary> {
        let library_dir = parent_dir.as_ref().join(&self.library_name);

        info!(
            "Writing generated C AOT library '{}' into '{}'",
            self.library_name,
            library_dir.display()
        );
        fs::create_dir_all(&library_dir)?;

        let makefile = library_dir.join("Makefile");
        let generated_c = library_dir.join("generated.c");
        let generated_h = library_dir.join("generated.h");
        let manifest_h = library_dir.join("manifest.h");
        let aot_interface_c = library_dir.join("aot_interface.c");
        let aot_interface_h = library_dir.join("aot_interface.h");

        fs::write(&makefile, self.emit_makefile())?;
        fs::write(&generated_c, self.emit_generated_c())?;
        fs::write(&generated_h, self.emit_generated_h())?;
        fs::write(&manifest_h, self.emit_manifest_h())?;
        fs::write(&aot_interface_c, self.emit_aot_interface_c())?;
        fs::write(&aot_interface_h, self.emit_aot_interface_h())?;

        let written = WrittenCAotLibrary {
            library_dir,
            makefile,
            generated_c,
            generated_h,
            manifest_h,
            aot_interface_c,
            aot_interface_h,
        };
        info!(
            "Generated C AOT library '{}' written successfully",
            self.library_name
        );
        Ok(written)
    }

    fn emit_makefile(&self) -> String {
        let target_name = if cfg!(target_os = "windows") {
            format!("lib{}.dll", self.library_name)
        } else if cfg!(target_os = "macos") {
            format!("lib{}.dylib", self.library_name)
        } else {
            format!("lib{}.so", self.library_name)
        };
        format!(
            "# Auto-generated Makefile for C AOT library\n\
CC = {}\n\
CFLAGS = {} -fPIC -Wall -Wextra\n\
LDFLAGS = -shared -lm\n\
\n\
TARGET = {}\n\
OBJS = generated.o aot_interface.o\n\
\n\
all: $(TARGET)\n\
\n\
$(TARGET): $(OBJS)\n\
\t$(CC) $(LDFLAGS) -o $@ $^\n\
\n\
generated.o: generated.c generated.h manifest.h\n\
\t$(CC) $(CFLAGS) -c generated.c\n\
\n\
aot_interface.o: aot_interface.c aot_interface.h generated.h manifest.h\n\
\t$(CC) $(CFLAGS) -c aot_interface.c\n\
\n\
clean:\n\
\trm -f $(OBJS) $(TARGET)\n\
\n\
.PHONY: all clean\n",
            self.compiler, self.optimization, target_name
        )
    }

    fn emit_generated_c(&self) -> String {
        let mut source = String::new();
        source.push_str("/* AUTO-GENERATED C AOT SOURCE */\n\n");
        source.push_str("#include \"generated.h\"\n");
        source.push_str("#include \"manifest.h\"\n\n");
        source.push_str(&self.c_source);
        source
    }

    fn emit_generated_h(&self) -> String {
        let mut header = String::new();
        header.push_str("/* AUTO-GENERATED C AOT HEADER */\n\n");
        header.push_str("#ifndef GENERATED_H\n");
        header.push_str("#define GENERATED_H\n\n");
        header.push_str(&self.c_header);
        header.push_str("\n#endif /* GENERATED_H */\n");
        header
    }

    fn emit_manifest_h(&self) -> String {
        let jacobian_nnz = match self.manifest.io.jacobian_nnz {
            Some(nnz) => nnz.to_string(),
            None => "0".to_string(),
        };

        format!(
            "/* AUTO-GENERATED C AOT MANIFEST */\n\n\
#ifndef MANIFEST_H\n\
#define MANIFEST_H\n\n\
#define BACKEND_KIND \"{}\"\n\
#define MATRIX_BACKEND \"{}\"\n\
#define RESIDUAL_LEN {}\n\
#define JACOBIAN_ROWS {}\n\
#define JACOBIAN_COLS {}\n\
#define JACOBIAN_NNZ {}\n\
#define RESIDUAL_FN_NAME \"{}\"\n\
#define JACOBIAN_FN_NAME \"{}\"\n\n\
#endif /* MANIFEST_H */\n",
            self.manifest.backend_kind.as_str(),
            self.manifest.matrix_backend.as_str(),
            self.manifest.io.residual_len,
            self.manifest.io.jacobian_rows,
            self.manifest.io.jacobian_cols,
            jacobian_nnz,
            escape_c_string(&self.manifest.functions.residual_fn_name),
            escape_c_string(&self.manifest.functions.jacobian_fn_name)
        )
    }

    fn emit_aot_interface_c(&self) -> String {
        let residual_dispatch = if self.manifest.functions.residual_chunks.is_empty() {
            format!(
                "    {}(args, out);",
                self.manifest.functions.residual_fn_name
            )
        } else {
            self.manifest
                .functions
                .residual_chunks
                .iter()
                .map(|chunk| format!("    {}(args, &out[{}]);", chunk.fn_name, chunk.offset))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let jacobian_dispatch = if self.manifest.functions.jacobian_fn_name.is_empty() {
            String::new()
        } else if self.manifest.functions.jacobian_chunks.is_empty() {
            format!(
                "    {}(args, out);",
                self.manifest.functions.jacobian_fn_name
            )
        } else {
            self.manifest
                .functions
                .jacobian_chunks
                .iter()
                .map(|chunk| format!("    {}(args, &out[{}]);", chunk.fn_name, chunk.offset))
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!(
            "/* AUTO-GENERATED C AOT FFI INTERFACE */\n\n\
#include \"aot_interface.h\"\n\
#include \"generated.h\"\n\
#include \"manifest.h\"\n\
#include <stddef.h>\n\
#include <stdint.h>\n\n\
RUSTEDSCITHE_AOT_EXPORT int rustedscithe_aot_eval_residual(\n\
    const double* args_ptr,\n\
    size_t args_len,\n\
    double* out_ptr,\n\
    size_t out_len\n\
) {{\n\
    if (args_ptr == NULL || out_ptr == NULL || out_len != RESIDUAL_LEN) {{\n\
        return 0;\n\
    }}\n\
    const double* args = args_ptr;\n\
    double* out = out_ptr;\n\
    (void)args_len;\n\
{}\n\
    return 1;\n\
}}\n\n\
RUSTEDSCITHE_AOT_EXPORT int rustedscithe_aot_eval_jacobian_values(\n\
    const double* args_ptr,\n\
    size_t args_len,\n\
    double* out_ptr,\n\
    size_t out_len\n\
) {{\n\
    if (args_ptr == NULL || out_ptr == NULL) {{\n\
        return 0;\n\
    }}\n\
    size_t expected_out_len = JACOBIAN_NNZ > 0 ? JACOBIAN_NNZ : (JACOBIAN_ROWS * JACOBIAN_COLS);\n\
    if (out_len != expected_out_len) {{\n\
        return 0;\n\
    }}\n\
    const double* args = args_ptr;\n\
    double* out = out_ptr;\n\
    (void)args_len;\n\
{}\n\
    return 1;\n\
}}\n",
            residual_dispatch, jacobian_dispatch
        )
    }

    fn emit_aot_interface_h(&self) -> String {
        "/* AUTO-GENERATED C AOT FFI INTERFACE HEADER */\n\n\
#ifndef AOT_INTERFACE_H\n\
#define AOT_INTERFACE_H\n\n\
#include <stddef.h>\n\
#include <stdint.h>\n\n\
#if defined(_WIN32) || defined(__CYGWIN__)\n\
  #define RUSTEDSCITHE_AOT_EXPORT __declspec(dllexport)\n\
#elif defined(__GNUC__)\n\
  #define RUSTEDSCITHE_AOT_EXPORT __attribute__((visibility(\"default\")))\n\
#else\n\
  #define RUSTEDSCITHE_AOT_EXPORT\n\
#endif\n\n\
#ifdef __cplusplus\n\
extern \"C\" {\n\
#endif\n\n\
RUSTEDSCITHE_AOT_EXPORT int rustedscithe_aot_eval_residual(\n\
    const double* args_ptr,\n\
    size_t args_len,\n\
    double* out_ptr,\n\
    size_t out_len\n\
);\n\n\
RUSTEDSCITHE_AOT_EXPORT int rustedscithe_aot_eval_jacobian_values(\n\
    const double* args_ptr,\n\
    size_t args_len,\n\
    double* out_ptr,\n\
    size_t out_len\n\
);\n\n\
#ifdef __cplusplus\n\
}\n\
#endif\n\n\
#endif /* AOT_INTERFACE_H */\n"
            .to_string()
    }
}

fn validate_library_name(library_name: &str) {
    assert!(
        !library_name.is_empty(),
        "generated C AOT library name must not be empty"
    );
    assert!(
        library_name
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'),
        "generated C AOT library name must be snake_case ASCII"
    );
    assert!(
        library_name
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_lowercase()),
        "generated C AOT library name must start with a lowercase ASCII letter"
    );
}

fn escape_c_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\"', "\\\"")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::CodegenIR::CodegenModule;
    use crate::symbolic::codegen::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedBandedProblem, PreparedDenseProblem, PreparedProblem,
    };
    use crate::symbolic::codegen::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen::codegen_tasks::{
        BandedChunkingStrategy, BandedExprEntry, BandedJacobianTask, JacobianTask, ResidualTask,
    };
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
    fn generated_c_library_writes_expected_file_layout() {
        let prepared = sample_prepared_problem();
        let module = CodegenModule::new("generated_fixture")
            .with_language(CodegenLanguage::C)
            .add_block("eval_residual", &[Expr::parse_expression("x + 1")], &["x"]);

        let library_spec =
            GeneratedCAotLibrary::from_prepared_problem("generated_c_fixture", &prepared, &module);

        let dir = tempdir().expect("tempdir should exist");
        let written = library_spec
            .write_to_dir(dir.path())
            .expect("generated C library should write");

        assert!(written.makefile.exists());
        assert!(written.generated_c.exists());
        assert!(written.generated_h.exists());
        assert!(written.manifest_h.exists());
        assert!(written.aot_interface_c.exists());
        assert!(written.aot_interface_h.exists());

        let makefile = fs::read_to_string(&written.makefile).expect("Makefile should be readable");
        let generated_c =
            fs::read_to_string(&written.generated_c).expect("generated.c should be readable");
        let manifest_h =
            fs::read_to_string(&written.manifest_h).expect("manifest.h should be readable");
        let aot_interface_c = fs::read_to_string(&written.aot_interface_c)
            .expect("aot_interface.c should be readable");

        let expected_target = if cfg!(target_os = "windows") {
            "libgenerated_c_fixture.dll"
        } else if cfg!(target_os = "macos") {
            "libgenerated_c_fixture.dylib"
        } else {
            "libgenerated_c_fixture.so"
        };
        assert!(makefile.contains(expected_target));
        assert!(generated_c.contains("#include <math.h>"));
        assert!(manifest_h.contains("#define BACKEND_KIND \"aot\""));
        assert!(manifest_h.contains("#define RESIDUAL_FN_NAME \"eval_residual\""));
        assert!(aot_interface_c.contains("rustedscithe_aot_eval_residual"));
        assert!(aot_interface_c.contains("rustedscithe_aot_eval_jacobian_values"));
        assert!(aot_interface_c.contains("const double* args = args_ptr;"));
        assert!(aot_interface_c.contains("double* out = out_ptr;"));
    }

    #[test]
    #[should_panic(expected = "snake_case ASCII")]
    fn generated_c_library_rejects_invalid_names() {
        let prepared = sample_prepared_problem();
        let module = CodegenModule::new("test").with_language(CodegenLanguage::C);
        let _ = GeneratedCAotLibrary::from_prepared_problem("Bad-Name", &prepared, &module);
    }

    #[test]
    fn generated_c_library_forces_c_source_even_if_module_defaults_to_rust() {
        let prepared = sample_prepared_problem();
        let module = CodegenModule::new("generated_fixture").add_block(
            "eval_residual",
            &[Expr::parse_expression("x + 1")],
            &["x"],
        );

        let library_spec =
            GeneratedCAotLibrary::from_prepared_problem("generated_c_fixture", &prepared, &module);

        assert!(library_spec.c_source.contains("#include <math.h>"));
        assert!(!library_spec.c_source.contains("pub mod generated_fixture"));
        assert!(!library_spec.c_source.contains("#![allow(clippy::all)]"));
    }

    #[test]
    fn generated_c_library_supports_banded_prepared_problem() {
        let residuals = vec![Expr::parse_expression("x + p")];
        let vars = vec!["x"];
        let params = vec!["p"];
        let jac = Expr::parse_expression("2");
        let entries = vec![BandedExprEntry {
            row: 0,
            col: 0,
            diag_offset: 0,
            diag_position: 0,
            expr: &jac,
        }];

        let prepared = PreparedProblem::banded(PreparedBandedProblem::new(
            BackendKind::Aot,
            MatrixBackend::Banded,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            BandedJacobianTask {
                fn_name: "eval_banded_values",
                shape: (1, 1),
                kl: 0,
                ku: 0,
                entries: &entries,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(BandedChunkingStrategy::Whole),
        ));

        let module = CodegenModule::new("generated_banded_fixture")
            .with_language(CodegenLanguage::C)
            .add_residual_block_plan(match &prepared {
                PreparedProblem::Banded(problem) => &problem.residual_plan.chunks[0].plan,
                PreparedProblem::Dense(_) | PreparedProblem::Sparse(_) => {
                    unreachable!("test uses banded prepared problem")
                }
            })
            .add_sparse_values_plan(match &prepared {
                PreparedProblem::Banded(problem) => &problem.jacobian_plan.chunks[0].plan,
                PreparedProblem::Dense(_) | PreparedProblem::Sparse(_) => {
                    unreachable!("test uses banded prepared problem")
                }
            });

        let library_spec = GeneratedCAotLibrary::from_prepared_problem(
            "generated_banded_c_fixture",
            &prepared,
            &module,
        );

        assert_eq!(library_spec.manifest.matrix_backend, MatrixBackend::Banded);
        assert!(library_spec.c_source.contains("eval_banded_values"));
    }
}
