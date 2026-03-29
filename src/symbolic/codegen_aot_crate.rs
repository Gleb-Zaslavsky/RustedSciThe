//! Minimal writer for separately compiled generated AOT crates.
//!
//! This module is intentionally small and file-oriented:
//! - it does not decide which symbolic task to generate,
//! - it does not build runtime plans,
//! - and it does not invoke Cargo.
//!
//! Its job is to take already emitted generated Rust source plus an owned
//! [`PreparedProblemManifest`](crate::symbolic::codegen_manifest::PreparedProblemManifest)
//! and write a tiny standalone crate layout that can later be compiled
//! separately and linked statically.

use crate::symbolic::CodegenIR::CodegenModule;
use crate::symbolic::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen_provider_api::{
    PreparedDenseProblem, PreparedProblem, PreparedSparseProblem,
};
use log::info;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Description of one generated AOT crate ready to be written to disk.
#[derive(Debug, Clone)]
pub struct GeneratedAotCrate {
    /// Cargo crate name to be written on disk.
    pub crate_name: String,
    /// Rust edition string written into `Cargo.toml`.
    pub rust_edition: String,
    /// Emitted generated Rust module source.
    pub module_source: String,
    /// Owned solver/codegen metadata exported alongside the module source.
    pub manifest: PreparedProblemManifest,
}

/// Paths written for one generated AOT crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenAotCrate {
    /// Root directory of the generated crate.
    pub crate_dir: PathBuf,
    /// Path to the generated `Cargo.toml`.
    pub cargo_toml: PathBuf,
    /// Path to the crate `src/` directory.
    pub src_dir: PathBuf,
    /// Path to the generated `src/lib.rs`.
    pub lib_rs: PathBuf,
    /// Path to the generated source module containing emitted functions.
    pub generated_rs: PathBuf,
    /// Path to the generated metadata module.
    pub manifest_rs: PathBuf,
}

impl GeneratedAotCrate {
    /// Creates a new generated AOT crate description.
    pub fn new(
        crate_name: impl Into<String>,
        module_source: impl Into<String>,
        manifest: PreparedProblemManifest,
    ) -> Self {
        let crate_name = crate_name.into();
        validate_crate_name(&crate_name);

        Self {
            crate_name,
            rust_edition: "2024".to_string(),
            module_source: module_source.into(),
            manifest,
        }
    }

    /// Builds a generated crate directly from an emitted `CodegenModule`.
    pub fn from_codegen_module(
        crate_name: impl Into<String>,
        module: &CodegenModule,
        manifest: PreparedProblemManifest,
    ) -> Self {
        Self::new(crate_name, module.emit_source(), manifest)
    }

    /// Builds a generated crate from a prepared dense problem and a matching
    /// emitted `CodegenModule`.
    pub fn from_prepared_dense_problem(
        crate_name: impl Into<String>,
        problem: &PreparedDenseProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(crate_name, module, PreparedProblemManifest::from(problem))
    }

    /// Builds a generated crate from a prepared sparse problem and a matching
    /// emitted `CodegenModule`.
    pub fn from_prepared_sparse_problem(
        crate_name: impl Into<String>,
        problem: &PreparedSparseProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(crate_name, module, PreparedProblemManifest::from(problem))
    }

    /// Builds a generated crate from an arbitrary prepared problem and a
    /// matching emitted `CodegenModule`.
    pub fn from_prepared_problem(
        crate_name: impl Into<String>,
        problem: &PreparedProblem<'_>,
        module: &CodegenModule,
    ) -> Self {
        Self::from_codegen_module(crate_name, module, PreparedProblemManifest::from(problem))
    }

    /// Overrides the Rust edition written to `Cargo.toml`.
    pub fn with_rust_edition(mut self, rust_edition: impl Into<String>) -> Self {
        self.rust_edition = rust_edition.into();
        self
    }

    /// Writes the generated crate under `parent_dir/<crate_name>/`.
    pub fn write_to_dir<P: AsRef<Path>>(&self, parent_dir: P) -> io::Result<WrittenAotCrate> {
        let crate_dir = parent_dir.as_ref().join(&self.crate_name);
        let src_dir = crate_dir.join("src");

        info!(
            "Writing generated AOT crate '{}' into '{}'",
            self.crate_name,
            crate_dir.display()
        );
        fs::create_dir_all(&src_dir)?;

        let cargo_toml = crate_dir.join("Cargo.toml");
        let lib_rs = src_dir.join("lib.rs");
        let generated_rs = src_dir.join("generated.rs");
        let manifest_rs = src_dir.join("manifest.rs");

        fs::write(&cargo_toml, self.emit_cargo_toml())?;
        fs::write(&generated_rs, self.emit_generated_rs())?;
        fs::write(&manifest_rs, self.emit_manifest_rs())?;
        fs::write(&lib_rs, self.emit_lib_rs())?;

        let written = WrittenAotCrate {
            crate_dir,
            cargo_toml,
            src_dir,
            lib_rs,
            generated_rs,
            manifest_rs,
        };
        info!(
            "Generated AOT crate '{}' written successfully",
            self.crate_name
        );
        Ok(written)
    }

    fn emit_cargo_toml(&self) -> String {
        format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"{}\"\npublish = false\n\n[lib]\npath = \"src/lib.rs\"\n",
            self.crate_name, self.rust_edition
        )
    }

    fn emit_generated_rs(&self) -> String {
        let mut source = String::new();
        source.push_str("// AUTO-GENERATED AOT MODULE SOURCE.\n");
        source.push_str("// This file is written by codegen_aot_crate.rs.\n\n");
        source.push_str(self.module_source.trim());
        source.push('\n');
        source
    }

    fn emit_manifest_rs(&self) -> String {
        let inputs = self
            .manifest
            .io
            .input_names
            .iter()
            .map(|name| format!("    \"{}\",\n", escape_rust_string(name)))
            .collect::<String>();

        let residual_chunks = self
            .manifest
            .functions
            .residual_chunk_names
            .iter()
            .map(|name| format!("    \"{}\",\n", escape_rust_string(name)))
            .collect::<String>();

        let jacobian_chunks = self
            .manifest
            .functions
            .jacobian_chunk_names
            .iter()
            .map(|name| format!("    \"{}\",\n", escape_rust_string(name)))
            .collect::<String>();

        let jacobian_nnz = match self.manifest.io.jacobian_nnz {
            Some(nnz) => format!("Some({nnz})"),
            None => "None".to_string(),
        };

        format!(
            "// AUTO-GENERATED AOT MANIFEST METADATA.\n\n\
pub const BACKEND_KIND: &str = \"{}\";\n\
pub const MATRIX_BACKEND: &str = \"{}\";\n\
pub const RESIDUAL_LEN: usize = {};\n\
pub const JACOBIAN_ROWS: usize = {};\n\
pub const JACOBIAN_COLS: usize = {};\n\
pub const JACOBIAN_NNZ: Option<usize> = {};\n\
pub const RESIDUAL_FN_NAME: &str = \"{}\";\n\
pub const JACOBIAN_FN_NAME: &str = \"{}\";\n\
pub const INPUT_NAMES: &[&str] = &[\n{}];\n\
pub const RESIDUAL_CHUNK_NAMES: &[&str] = &[\n{}];\n\
pub const JACOBIAN_CHUNK_NAMES: &[&str] = &[\n{}];\n",
            self.manifest.backend_kind.as_str(),
            self.manifest.matrix_backend.as_str(),
            self.manifest.io.residual_len,
            self.manifest.io.jacobian_rows,
            self.manifest.io.jacobian_cols,
            jacobian_nnz,
            escape_rust_string(&self.manifest.functions.residual_fn_name),
            escape_rust_string(&self.manifest.functions.jacobian_fn_name),
            inputs,
            residual_chunks,
            jacobian_chunks
        )
    }

    fn emit_lib_rs(&self) -> String {
        "//! Minimal generated AOT crate.\n\
pub mod generated;\n\
pub mod manifest;\n"
            .to_string()
    }
}

fn validate_crate_name(crate_name: &str) {
    assert!(
        !crate_name.is_empty(),
        "generated AOT crate name must not be empty"
    );
    assert!(
        crate_name
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'),
        "generated AOT crate name must be snake_case ASCII"
    );
    assert!(
        crate_name
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_lowercase()),
        "generated AOT crate name must start with a lowercase ASCII letter"
    );
}

fn escape_rust_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::CodegenIR::CodegenModule;
    use crate::symbolic::codegen_manifest::PreparedProblemManifest;
    use crate::symbolic::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem, PreparedProblem,
    };
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen_tasks::{JacobianTask, ResidualTask};
    use crate::symbolic::symbolic_engine::Expr;
    use tempfile::tempdir;

    fn sample_dense_manifest() -> PreparedProblemManifest {
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

        PreparedProblemManifest::from(&prepared)
    }

    #[test]
    fn generated_aot_crate_writes_expected_file_layout() {
        let module_source = CodegenModule::new("generated_fixture")
            .add_block("eval_residual", &[Expr::parse_expression("x + 1")], &["x"])
            .emit_source();

        let crate_spec = GeneratedAotCrate::new(
            "generated_aot_fixture",
            module_source,
            sample_dense_manifest(),
        );

        let dir = tempdir().expect("tempdir should exist");
        let written = crate_spec
            .write_to_dir(dir.path())
            .expect("generated AOT crate should write");

        assert!(written.cargo_toml.exists());
        assert!(written.lib_rs.exists());
        assert!(written.generated_rs.exists());
        assert!(written.manifest_rs.exists());

        let cargo_toml =
            fs::read_to_string(&written.cargo_toml).expect("Cargo.toml should be readable");
        let lib_rs = fs::read_to_string(&written.lib_rs).expect("lib.rs should be readable");
        let generated_rs =
            fs::read_to_string(&written.generated_rs).expect("generated.rs should be readable");
        let manifest_rs =
            fs::read_to_string(&written.manifest_rs).expect("manifest.rs should be readable");

        assert!(cargo_toml.contains("name = \"generated_aot_fixture\""));
        assert!(lib_rs.contains("pub mod generated;"));
        assert!(generated_rs.contains("pub mod generated_fixture"));
        assert!(manifest_rs.contains("pub const BACKEND_KIND: &str = \"aot\";"));
        assert!(manifest_rs.contains("pub const RESIDUAL_FN_NAME: &str = \"eval_residual\";"));
    }

    #[test]
    fn generated_aot_crate_can_be_built_from_prepared_problem_and_module() {
        let residuals = vec![Expr::parse_expression("x + 1")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];

        let prepared = PreparedProblem::dense(PreparedDenseProblem::new(
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
        ));

        let module = CodegenModule::new("generated_fixture")
            .add_residual_block_plan(match &prepared {
                PreparedProblem::Dense(problem) => &problem.residual_plan.chunks[0].plan,
                PreparedProblem::Sparse(_) => unreachable!("test uses dense prepared problem"),
            })
            .add_dense_jacobian_plan(match &prepared {
                PreparedProblem::Dense(problem) => &problem.jacobian_plan.chunks[0].plan,
                PreparedProblem::Sparse(_) => unreachable!("test uses dense prepared problem"),
            });

        let crate_spec =
            GeneratedAotCrate::from_prepared_problem("generated_from_prepared", &prepared, &module);

        assert_eq!(crate_spec.crate_name, "generated_from_prepared");
        assert!(
            crate_spec
                .module_source
                .contains("pub mod generated_fixture")
        );
        assert_eq!(crate_spec.manifest.backend_kind, BackendKind::Aot);
        assert_eq!(crate_spec.manifest.matrix_backend, MatrixBackend::Dense);
        assert_eq!(crate_spec.manifest.io.residual_len, 1);
        assert_eq!(crate_spec.manifest.io.jacobian_rows, 1);
        assert_eq!(crate_spec.manifest.io.jacobian_cols, 1);
    }

    #[test]
    #[should_panic(expected = "snake_case ASCII")]
    fn generated_aot_crate_rejects_invalid_crate_names() {
        let _ = GeneratedAotCrate::new("Bad-Name", "pub mod x {}", sample_dense_manifest());
    }
}
