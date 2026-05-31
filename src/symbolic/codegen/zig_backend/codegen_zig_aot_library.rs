//! Zig library writer for separately compiled generated AOT backends.
//!
//! Mirrors `codegen_c_aot_library.rs` but generates Zig code.
//! Generated structure:
//! ```text
//! <library_name>/
//!   ├── build.zig         (Zig build script)
//!   ├── generated.zig     (emitted Zig functions)
//!   └── aot_interface.zig (FFI wrapper matching Rust interface)
//! ```

use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::{
    PreparedBandedProblem, PreparedDenseProblem, PreparedProblem, PreparedSparseProblem,
};
use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule};
use log::info;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Description of one generated Zig AOT library ready to be written to disk.
#[derive(Debug, Clone)]
pub struct GeneratedZigAotLibrary {
    pub library_name: String,
    pub zig_source: String,
    pub manifest: PreparedProblemManifest,
}

/// Paths written for one generated Zig AOT library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenZigAotLibrary {
    pub library_dir: PathBuf,
    pub build_zig: PathBuf,
    pub generated_zig: PathBuf,
    pub aot_interface_zig: PathBuf,
}

impl GeneratedZigAotLibrary {
    pub fn new(
        library_name: impl Into<String>,
        zig_source: impl Into<String>,
        manifest: PreparedProblemManifest,
    ) -> Self {
        let library_name = library_name.into();
        validate_library_name(&library_name);
        Self {
            library_name,
            zig_source: zig_source.into(),
            manifest,
        }
    }

    pub fn from_codegen_module(
        library_name: impl Into<String>,
        module: &CodegenModule,
        manifest: PreparedProblemManifest,
    ) -> Self {
        let mut module = module.clone();
        module.set_language(CodegenLanguage::Zig);
        Self::new(library_name, module.emit_source(), manifest)
    }

    pub fn from_prepared_dense_problem(
        library_name: impl Into<String>,
        problem: &PreparedDenseProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    pub fn from_prepared_sparse_problem(
        library_name: impl Into<String>,
        problem: &PreparedSparseProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    pub fn from_prepared_banded_problem(
        library_name: impl Into<String>,
        problem: &PreparedBandedProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    pub fn from_prepared_problem(
        library_name: impl Into<String>,
        problem: &PreparedProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(library_name, module, PreparedProblemManifest::from(problem))
    }

    /// Writes the generated Zig library under `parent_dir/<library_name>/`.
    pub fn write_to_dir<P: AsRef<Path>>(&self, parent_dir: P) -> io::Result<WrittenZigAotLibrary> {
        let library_dir = parent_dir.as_ref().join(&self.library_name);
        info!(
            "Writing generated Zig AOT library '{}' into '{}'",
            self.library_name,
            library_dir.display()
        );
        fs::create_dir_all(&library_dir)?;

        let build_zig = library_dir.join("build.zig");
        let generated_zig = library_dir.join("generated.zig");
        let aot_interface_zig = library_dir.join("aot_interface.zig");

        fs::write(&build_zig, self.emit_build_zig())?;
        self.write_generated_zig(&generated_zig)?;
        fs::write(&aot_interface_zig, self.emit_aot_interface_zig())?;

        info!(
            "Generated Zig AOT library '{}' written successfully",
            self.library_name
        );
        Ok(WrittenZigAotLibrary {
            library_dir,
            build_zig,
            generated_zig,
            aot_interface_zig,
        })
    }

    fn emit_build_zig(&self) -> String {
        format!(
            "// Auto-generated build.zig for Zig AOT library\n\
const std = @import(\"std\");\n\
\n\
pub fn build(b: *std.Build) void {{\n\
    const target = b.standardTargetOptions(.{{}});\n\
    const optimize = b.standardOptimizeOption(.{{}});\n\
\n\
    const lib = b.addLibrary(.{{\n\
        .linkage = .dynamic,\n\
        .name = \"{name}\",\n\
        .root_module = b.createModule(.{{\n\
            .root_source_file = b.path(\"aot_interface.zig\"),\n\
            .target = target,\n\
            .optimize = optimize,\n\
        }}),\n\
    }});\n\
\n\
    b.installArtifact(lib);\n\
}}\n",
            name = self.library_name
        )
    }

    fn write_generated_zig(&self, path: &Path) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        file.write_all(b"// AUTO-GENERATED ZIG AOT SOURCE\n\n")?;
        file.write_all(self.zig_source.as_bytes())?;
        Ok(())
    }

    fn emit_aot_interface_zig(&self) -> String {
        let residual_len = self.manifest.io.residual_len;
        let jacobian_nnz = self
            .manifest
            .io
            .jacobian_nnz
            .unwrap_or(self.manifest.io.jacobian_rows * self.manifest.io.jacobian_cols);
        let residual_chunk_exports = self
            .manifest
            .functions
            .residual_chunks
            .iter()
            .map(|chunk| {
                format!(
                    "\nexport fn rustedscithe_aot_chunk_{fn_name}(\n\
    args_ptr: [*]const f64,\n\
    args_len: usize,\n\
    out_ptr: [*]f64,\n\
    out_len: usize,\n\
) bool {{\n\
    _ = args_len;\n\
    if (out_len != {out_len}) return false;\n\
    generated.{fn_name}(args_ptr, out_ptr);\n\
    return true;\n\
}}\n",
                    fn_name = chunk.fn_name,
                    out_len = chunk.len,
                )
            })
            .collect::<Vec<_>>()
            .join("");
        let jacobian_chunk_exports = self
            .manifest
            .functions
            .jacobian_chunks
            .iter()
            .map(|chunk| {
                format!(
                    "\nexport fn rustedscithe_aot_chunk_{fn_name}(\n\
    args_ptr: [*]const f64,\n\
    args_len: usize,\n\
    out_ptr: [*]f64,\n\
    out_len: usize,\n\
) bool {{\n\
    _ = args_len;\n\
    if (out_len != {out_len}) return false;\n\
    generated.{fn_name}(args_ptr, out_ptr);\n\
    return true;\n\
}}\n",
                    fn_name = chunk.fn_name,
                    out_len = chunk.len,
                )
            })
            .collect::<Vec<_>>()
            .join("");

        let residual_dispatch = if self.manifest.functions.residual_chunks.is_empty() {
            format!(
                "    generated.{}(args_ptr, out_ptr);",
                self.manifest.functions.residual_fn_name
            )
        } else {
            self.manifest
                .functions
                .residual_chunks
                .iter()
                .map(|chunk| {
                    format!(
                        "    generated.{}(args_ptr, out_ptr + {});",
                        chunk.fn_name, chunk.offset
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let jacobian_dispatch = if self.manifest.functions.jacobian_fn_name.is_empty() {
            "    _ = args_ptr;\n    _ = out_ptr;".to_string()
        } else if self.manifest.functions.jacobian_chunks.is_empty() {
            format!(
                "    generated.{}(args_ptr, out_ptr);",
                self.manifest.functions.jacobian_fn_name
            )
        } else {
            self.manifest
                .functions
                .jacobian_chunks
                .iter()
                .map(|chunk| {
                    format!(
                        "    generated.{}(args_ptr, out_ptr + {});",
                        chunk.fn_name, chunk.offset
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!(
            "// AUTO-GENERATED ZIG AOT FFI INTERFACE\n\n\
const generated = @import(\"generated.zig\");\n\n\
export fn rustedscithe_aot_eval_residual(\n\
    args_ptr: [*]const f64,\n\
    args_len: usize,\n\
    out_ptr: [*]f64,\n\
    out_len: usize,\n\
) bool {{\n\
    _ = args_len;\n\
    if (out_len != {residual_len}) return false;\n\
{residual_dispatch}\n\
    return true;\n\
}}\n\n\
export fn rustedscithe_aot_eval_jacobian_values(\n\
    args_ptr: [*]const f64,\n\
    args_len: usize,\n\
    out_ptr: [*]f64,\n\
    out_len: usize,\n\
) bool {{\n\
    _ = args_len;\n\
    if (out_len != {jacobian_nnz}) return false;\n\
{jacobian_dispatch}\n\
    return true;\n\
}}\n",
            residual_len = residual_len,
            jacobian_nnz = jacobian_nnz,
            residual_dispatch = residual_dispatch,
            jacobian_dispatch = jacobian_dispatch,
        ) + residual_chunk_exports.as_str()
            + jacobian_chunk_exports.as_str()
    }
}

fn validate_library_name(library_name: &str) {
    assert!(
        !library_name.is_empty(),
        "generated Zig AOT library name must not be empty"
    );
    assert!(
        library_name
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'),
        "generated Zig AOT library name must be snake_case ASCII"
    );
    assert!(
        library_name
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_lowercase()),
        "generated Zig AOT library name must start with a lowercase ASCII letter"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedBandedProblem, PreparedDenseProblem, PreparedProblem,
    };
    use crate::symbolic::codegen::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen::codegen_tasks::{
        BandedChunkingStrategy, BandedExprEntry, BandedJacobianTask, JacobianTask, ResidualTask,
    };
    use crate::symbolic::codegen::CodegenIR::CodegenModule;
    use crate::symbolic::symbolic_engine::Expr;
    use tempfile::tempdir;

    fn chunked_dense_problem() -> PreparedDenseProblem<'static> {
        let residuals = Box::leak(Box::new(vec![
            Expr::parse_expression("x + 1"),
            Expr::parse_expression("y + 2"),
        ]));
        let jacobian = Box::leak(Box::new(vec![
            vec![Expr::parse_expression("1"), Expr::parse_expression("0")],
            vec![Expr::parse_expression("0"), Expr::parse_expression("1")],
        ]));
        let vars = Box::leak(Box::new(vec!["x", "y"]));

        PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual",
                residuals,
                variables: vars,
                params: None,
            }
            .runtime_plan(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 1,
            }),
            JacobianTask {
                fn_name: "eval_jacobian",
                jacobian,
                variables: vars,
                params: None,
            }
            .runtime_plan(DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 1 }),
        )
    }

    #[test]
    fn generated_zig_library_writes_expected_file_layout() {
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
        let library_spec =
            GeneratedZigAotLibrary::from_prepared_dense_problem("test_zig_lib", &prepared, &module);

        let dir = tempdir().expect("tempdir should exist");
        let written = library_spec
            .write_to_dir(dir.path())
            .expect("generated Zig library should write");

        assert!(written.build_zig.exists());
        assert!(written.generated_zig.exists());
        assert!(written.aot_interface_zig.exists());

        let build_zig =
            fs::read_to_string(&written.build_zig).expect("build.zig should be readable");
        let aot_interface = fs::read_to_string(&written.aot_interface_zig)
            .expect("aot_interface.zig should be readable");

        assert!(build_zig.contains("test_zig_lib"));
        assert!(aot_interface.contains("rustedscithe_aot_eval_residual"));
        assert!(aot_interface.contains("rustedscithe_aot_eval_jacobian_values"));
    }

    #[test]
    fn generated_zig_library_exports_chunk_ffi_symbols() {
        let prepared = chunked_dense_problem();
        let module = CodegenModule::new("test_module").with_language(CodegenLanguage::Zig);
        let library_spec =
            GeneratedZigAotLibrary::from_prepared_dense_problem("test_zig_lib", &prepared, &module);

        let aot_interface = library_spec.emit_aot_interface_zig();

        assert!(aot_interface.contains("rustedscithe_aot_chunk_eval_residual_chunk_0"));
        assert!(aot_interface.contains("rustedscithe_aot_chunk_eval_residual_chunk_1"));
        assert!(aot_interface.contains("rustedscithe_aot_chunk_eval_jacobian_chunk_0"));
        assert!(aot_interface.contains("rustedscithe_aot_chunk_eval_jacobian_chunk_1"));
    }

    #[test]
    #[should_panic(expected = "snake_case ASCII")]
    fn generated_zig_library_rejects_invalid_names() {
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
        let module = CodegenModule::new("test").with_language(CodegenLanguage::Zig);
        let _ = GeneratedZigAotLibrary::from_prepared_dense_problem("Bad-Name", &prepared, &module);
    }

    #[test]
    fn generated_zig_library_forces_zig_source_even_if_module_defaults_to_rust() {
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

        let module = CodegenModule::new("generated_fixture").add_block(
            "eval_residual",
            &[Expr::parse_expression("x + 1")],
            &["x"],
        );
        let library_spec = GeneratedZigAotLibrary::from_prepared_dense_problem(
            "generated_zig_fixture",
            &prepared,
            &module,
        );

        assert!(library_spec
            .zig_source
            .contains("const std = @import(\"std\");"));
        assert!(!library_spec
            .zig_source
            .contains("pub mod generated_fixture"));
        assert!(!library_spec.zig_source.contains("#![allow(clippy::all)]"));
    }

    #[test]
    fn generated_zig_library_supports_banded_prepared_problem() {
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
            .with_language(CodegenLanguage::Zig)
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

        let library_spec = GeneratedZigAotLibrary::from_prepared_problem(
            "generated_banded_zig_fixture",
            &prepared,
            &module,
        );

        assert_eq!(library_spec.manifest.matrix_backend, MatrixBackend::Banded);
        assert!(library_spec.zig_source.contains("eval_banded_values"));
    }
}
