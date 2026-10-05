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

use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_aot_lifecycle::{
    AotArtifactInspection, AotArtifactState, AotFailureDiagnostics, AotFailureKind,
    AotLifecycleError, AotLifecycleStage, quarantine_generated_tree,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildResult;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
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
    /// Codegen/runtime loader selected by the producer.
    ///
    /// `None` is retained only for legacy handoff records written before
    /// backend provenance was part of the durable registry contract. New
    /// records always carry this value.
    pub codegen_backend: Option<AotCodegenBackend>,
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

    /// Returns a typed filesystem snapshot used by both RequirePrebuilt and
    /// rebuild orchestration. This is intentionally cheaper and safer than
    /// trying to infer state from diagnostic strings.
    pub fn inspect_artifact(&self) -> AotArtifactInspection {
        AotArtifactInspection::inspect(
            &self.crate_dir,
            &self.manifest_file,
            &self.expected_rlib,
            &self.expected_cdylib,
        )
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

    /// Moves a suspicious generated tree aside before a rebuild.
    ///
    /// We never overwrite a stale or partially written tree in place: a
    /// compiler/linker failure can otherwise leave files that look usable to
    /// a later RequirePrebuilt request. The quarantine is a sibling directory
    /// and is deliberately left for diagnostics/cleanup by the caller.
    pub fn quarantine_generated_tree(&self) -> Result<Option<PathBuf>, AotLifecycleError> {
        let inspection = self.inspect_artifact();
        if matches!(
            inspection.state,
            AotArtifactState::Missing | AotArtifactState::Ready
        ) {
            return Ok(None);
        }
        if !self.crate_dir.is_dir() {
            let mut diagnostics = AotFailureDiagnostics::new(
                AotLifecycleStage::Materialized,
                AotFailureKind::PartialArtifact,
                &self.problem_key,
                format!(
                    "generated crate path is not a directory: {}",
                    self.crate_dir.display()
                ),
            );
            diagnostics.inspection = Some(inspection);
            return Err(AotLifecycleError::new(diagnostics));
        }
        let quarantined = quarantine_generated_tree(&self.crate_dir, &self.problem_key).map_err(
            |mut error| {
                error.diagnostics.quarantine_attempted = true;
                error.diagnostics.inspection = Some(inspection.clone());
                error
            },
        )?;
        Ok(quarantined)
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

    /// Publishes registry metadata for a process-isolated consumer.
    ///
    /// The handoff contains only owned manifest/path metadata. Compiled code
    /// is never copied and no live callback is serialized. A consumer must
    /// still validate the manifest key and inspect the artifact before use.
    pub fn write_handoff(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        writeln!(
            file,
            "RST_AOT_REGISTRY_HANDOFF|version=2|entries={}",
            self.len()
        )?;
        for artifact in self.entries_by_problem_key.values() {
            writeln!(file, "entry|{}", encode_artifact(artifact))?;
        }
        file.flush()
    }

    /// Loads a registry snapshot published by another process.
    pub fn read_handoff(path: impl AsRef<std::path::Path>) -> io::Result<Self> {
        let source = fs::read_to_string(path)?;
        let mut lines = source.lines();
        let header = lines
            .next()
            .ok_or_else(|| invalid_handoff("empty AOT registry handoff"))?;
        let version = header
            .split('|')
            .find_map(|field| field.strip_prefix("version="))
            .ok_or_else(|| invalid_handoff("AOT registry handoff has no version"))?;
        if version != "1" && version != "2" {
            return Err(invalid_handoff(format!(
                "unsupported AOT registry handoff version {version}"
            )));
        }
        let mut registry = Self::new();
        for line in lines.filter(|line| line.starts_with("entry|")) {
            let encoded = line
                .strip_prefix("entry|")
                .ok_or_else(|| invalid_handoff("malformed AOT registry entry prefix"))?;
            let artifact = decode_artifact(encoded, version == "2")?;
            registry.register_existing_artifact(artifact)?;
        }
        Ok(registry)
    }

    /// Inserts metadata reconstructed from a producer handoff.
    pub fn register_existing_artifact(
        &mut self,
        artifact: RegisteredAotArtifact,
    ) -> io::Result<()> {
        if artifact.problem_key.is_empty() || !artifact.manifest_key_matches() {
            return Err(invalid_handoff(
                "AOT handoff artifact key does not match its manifest",
            ));
        }
        if let Some(previous) = self
            .entries_by_problem_key
            .insert(artifact.problem_key.clone(), artifact.clone())
        {
            self.crate_name_to_problem_key.remove(&previous.crate_name);
        }
        self.crate_name_to_problem_key
            .insert(artifact.crate_name.clone(), artifact.problem_key.clone());
        Ok(())
    }

    /// Merges entries from another registry snapshot.
    ///
    /// A generated IVP may prepare residual-only and Jacobian artifacts in
    /// separate orchestration calls.  Each call returns its own resolver
    /// snapshot, so process handoff must combine them rather than let the
    /// later publication overwrite the earlier entry.
    pub fn merge_from(&mut self, other: &AotRegistry) -> io::Result<()> {
        for problem_key in other.problem_keys() {
            let artifact = other
                .get_by_problem_key(&problem_key)
                .ok_or_else(|| invalid_handoff("registry entry disappeared during merge"))?
                .clone();
            self.register_existing_artifact(artifact)?;
        }
        Ok(())
    }

    /// Registers one materialized build result under the manifest-derived
    /// `problem_key`. Re-registering the same problem replaces the old record.
    pub fn register_materialized_build(
        &mut self,
        manifest: PreparedProblemManifest,
        build: &AotBuildResult,
    ) -> &RegisteredAotArtifact {
        self.try_register_materialized_build_with_backend(
            manifest,
            build,
            Some(AotCodegenBackend::Rust),
        )
        .expect("generated crate metadata should be valid at compatibility boundary")
    }

    /// Fallible registry insertion used by new lifecycle code.
    pub fn try_register_materialized_build(
        &mut self,
        manifest: PreparedProblemManifest,
        build: &AotBuildResult,
    ) -> Result<&RegisteredAotArtifact, AotLifecycleError> {
        self.try_register_materialized_build_with_backend(
            manifest,
            build,
            Some(AotCodegenBackend::Rust),
        )
    }

    /// Registers a materialized artifact and preserves the producer's runtime
    /// loader choice for process-isolated consumers.
    pub fn try_register_materialized_build_with_backend(
        &mut self,
        manifest: PreparedProblemManifest,
        build: &AotBuildResult,
        codegen_backend: Option<AotCodegenBackend>,
    ) -> Result<&RegisteredAotArtifact, AotLifecycleError> {
        let problem_key = manifest.problem_key();
        let crate_name = build
            .written
            .crate_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                AotLifecycleError::new(AotFailureDiagnostics::new(
                    AotLifecycleStage::Materialized,
                    AotFailureKind::Manifest,
                    &problem_key,
                    "generated crate directory has no valid UTF-8 crate name",
                ))
            })?
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
                codegen_backend,
            },
        ) {
            self.crate_name_to_problem_key.remove(&previous.crate_name);
        }

        self.crate_name_to_problem_key
            .insert(crate_name, problem_key.clone());

        Ok(self
            .entries_by_problem_key
            .get(&problem_key)
            .expect("newly inserted registry entry should exist"))
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

    /// Quarantines a non-ready artifact while retaining the registry entry.
    /// The caller may then materialize a fresh build under the original path.
    pub fn quarantine_artifact_by_problem_key(
        &self,
        problem_key: &str,
    ) -> Result<Option<PathBuf>, AotLifecycleError> {
        let Some(artifact) = self.entries_by_problem_key.get(problem_key) else {
            return Ok(None);
        };
        artifact.quarantine_generated_tree()
    }
}

fn invalid_handoff(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn encode_text(value: &str) -> String {
    value
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn decode_text(value: &str) -> io::Result<String> {
    if value.len() % 2 != 0 {
        return Err(invalid_handoff("hex text has odd length"));
    }
    let bytes = (0..value.len())
        .step_by(2)
        .map(|offset| {
            u8::from_str_radix(&value[offset..offset + 2], 16)
                .map_err(|_| invalid_handoff("invalid hex text in AOT handoff"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    String::from_utf8(bytes).map_err(|_| invalid_handoff("AOT handoff text is not UTF-8"))
}

fn encode_strings(values: &[String]) -> String {
    values
        .iter()
        .map(|value| encode_text(value))
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_strings(value: &str) -> io::Result<Vec<String>> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value.split(',').map(decode_text).collect()
}

fn encode_chunks(
    chunks: &[crate::symbolic::codegen::codegen_manifest::GeneratedChunkManifest],
) -> String {
    chunks
        .iter()
        .map(|chunk| {
            format!(
                "{}~{}~{}",
                encode_text(&chunk.fn_name),
                chunk.offset,
                chunk.len
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_chunks(
    value: &str,
) -> io::Result<Vec<crate::symbolic::codegen::codegen_manifest::GeneratedChunkManifest>> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|chunk| {
            let mut fields = chunk.split('~');
            let fn_name = decode_text(
                fields
                    .next()
                    .ok_or_else(|| invalid_handoff("missing chunk name"))?,
            )?;
            let offset = fields
                .next()
                .ok_or_else(|| invalid_handoff("missing chunk offset"))?
                .parse()
                .map_err(|_| invalid_handoff("invalid chunk offset"))?;
            let len = fields
                .next()
                .ok_or_else(|| invalid_handoff("missing chunk length"))?
                .parse()
                .map_err(|_| invalid_handoff("invalid chunk length"))?;
            if fields.next().is_some() {
                return Err(invalid_handoff("too many chunk fields"));
            }
            Ok(
                crate::symbolic::codegen::codegen_manifest::GeneratedChunkManifest {
                    fn_name,
                    offset,
                    len,
                },
            )
        })
        .collect()
}

fn encode_layout(
    layout: Option<crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout>,
) -> String {
    use crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout;
    match layout {
        None => "none".to_string(),
        Some(PreparedJacobianLayout::Dense) => "dense".to_string(),
        Some(PreparedJacobianLayout::SparseExplicit) => "sparse_explicit".to_string(),
        Some(PreparedJacobianLayout::BandedExplicit) => "banded_explicit".to_string(),
        Some(PreparedJacobianLayout::BandedCompact { kl, ku }) => {
            format!("banded_compact~{kl}~{ku}")
        }
    }
}

fn encode_codegen_backend(backend: Option<AotCodegenBackend>) -> &'static str {
    match backend {
        None => "unknown",
        Some(AotCodegenBackend::Rust) => "rust",
        Some(AotCodegenBackend::C) => "c",
        Some(AotCodegenBackend::Zig) => "zig",
    }
}

fn decode_codegen_backend(value: &str) -> io::Result<Option<AotCodegenBackend>> {
    match value {
        "unknown" => Ok(None),
        "rust" => Ok(Some(AotCodegenBackend::Rust)),
        "c" => Ok(Some(AotCodegenBackend::C)),
        "zig" => Ok(Some(AotCodegenBackend::Zig)),
        _ => Err(invalid_handoff("unknown codegen backend in AOT handoff")),
    }
}

fn decode_layout(
    value: &str,
) -> io::Result<Option<crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout>> {
    use crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout;
    match value {
        "none" => Ok(None),
        "dense" => Ok(Some(PreparedJacobianLayout::Dense)),
        "sparse_explicit" => Ok(Some(PreparedJacobianLayout::SparseExplicit)),
        "banded_explicit" => Ok(Some(PreparedJacobianLayout::BandedExplicit)),
        value if value.starts_with("banded_compact~") => {
            let mut fields = value.split('~');
            let _ = fields.next();
            let kl = fields
                .next()
                .ok_or_else(|| invalid_handoff("missing compact band kl"))?
                .parse()
                .map_err(|_| invalid_handoff("invalid compact band kl"))?;
            let ku = fields
                .next()
                .ok_or_else(|| invalid_handoff("missing compact band ku"))?
                .parse()
                .map_err(|_| invalid_handoff("invalid compact band ku"))?;
            if fields.next().is_some() {
                return Err(invalid_handoff("too many compact band fields"));
            }
            Ok(Some(PreparedJacobianLayout::BandedCompact { kl, ku }))
        }
        _ => Err(invalid_handoff("unknown Jacobian layout in AOT handoff")),
    }
}

fn encode_artifact(artifact: &RegisteredAotArtifact) -> String {
    use crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute;
    use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend};
    let manifest = &artifact.manifest;
    let backend = match manifest.backend_kind {
        BackendKind::Numeric => "numeric",
        BackendKind::Lambdify => "lambdify",
        BackendKind::Aot => "aot",
    };
    let matrix = match manifest.matrix_backend {
        MatrixBackend::Dense => "dense",
        MatrixBackend::Banded => "banded",
        MatrixBackend::SparseCol => "sparse_col",
        MatrixBackend::CsMat => "cs_mat",
        MatrixBackend::CsMatrix => "cs_matrix",
        MatrixBackend::ValuesOnly => "values_only",
    };
    let route = match manifest.symbolic_route {
        PreparedSymbolicRoute::ExprLegacy => "expr_legacy",
        PreparedSymbolicRoute::AtomViewNative => "atom_view_native",
        PreparedSymbolicRoute::Generic => "generic",
    };
    let io = &manifest.io;
    let functions = &manifest.functions;
    [
        encode_text(&artifact.problem_key),
        encode_text(&artifact.crate_name),
        encode_text(&artifact.crate_dir.to_string_lossy()),
        encode_text(&artifact.manifest_file.to_string_lossy()),
        encode_text(&artifact.artifact_dir.to_string_lossy()),
        encode_text(&artifact.expected_rlib.to_string_lossy()),
        encode_text(&artifact.expected_cdylib.to_string_lossy()),
        encode_text(&artifact.cargo_program),
        encode_strings(&artifact.cargo_args),
        backend.to_string(),
        matrix.to_string(),
        route.to_string(),
        encode_strings(&io.input_names),
        io.residual_len.to_string(),
        io.jacobian_rows.to_string(),
        io.jacobian_cols.to_string(),
        io.jacobian_nnz
            .map_or_else(|| "none".to_string(), |value| value.to_string()),
        encode_layout(io.jacobian_layout),
        encode_text(&functions.residual_fn_name),
        encode_strings(&functions.residual_chunk_names),
        encode_chunks(&functions.residual_chunks),
        encode_text(&functions.jacobian_fn_name),
        encode_strings(&functions.jacobian_chunk_names),
        encode_chunks(&functions.jacobian_chunks),
        manifest.expression_signature.to_string(),
        encode_codegen_backend(artifact.codegen_backend).to_string(),
    ]
    .join("|")
}

fn decode_artifact(value: &str, has_backend_provenance: bool) -> io::Result<RegisteredAotArtifact> {
    use crate::symbolic::codegen::codegen_manifest::{
        GeneratedFunctionsManifest, PreparedProblemManifest, PreparedSymbolicRoute,
        ProblemIoManifest,
    };
    use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend};
    let fields = value.split('|').collect::<Vec<_>>();
    let expected_fields = if has_backend_provenance { 26 } else { 25 };
    if fields.len() != expected_fields {
        return Err(invalid_handoff(
            "AOT handoff entry has an unexpected field count",
        ));
    }
    let decode_num = |field: &str, name: &str| {
        field
            .parse()
            .map_err(|_| invalid_handoff(format!("invalid {name} in AOT handoff")))
    };
    let backend_kind = match fields[9] {
        "numeric" => BackendKind::Numeric,
        "lambdify" => BackendKind::Lambdify,
        "aot" => BackendKind::Aot,
        _ => return Err(invalid_handoff("unknown backend kind in AOT handoff")),
    };
    let matrix_backend = match fields[10] {
        "dense" => MatrixBackend::Dense,
        "banded" => MatrixBackend::Banded,
        "sparse_col" => MatrixBackend::SparseCol,
        "cs_mat" => MatrixBackend::CsMat,
        "cs_matrix" => MatrixBackend::CsMatrix,
        "values_only" => MatrixBackend::ValuesOnly,
        _ => return Err(invalid_handoff("unknown matrix backend in AOT handoff")),
    };
    let symbolic_route = match fields[11] {
        "expr_legacy" => PreparedSymbolicRoute::ExprLegacy,
        "atom_view_native" => PreparedSymbolicRoute::AtomViewNative,
        "generic" => PreparedSymbolicRoute::Generic,
        _ => return Err(invalid_handoff("unknown symbolic route in AOT handoff")),
    };
    let manifest = PreparedProblemManifest {
        backend_kind,
        matrix_backend,
        symbolic_route,
        io: ProblemIoManifest {
            input_names: decode_strings(fields[12])?,
            residual_len: decode_num(fields[13], "residual length")?,
            jacobian_rows: decode_num(fields[14], "Jacobian rows")?,
            jacobian_cols: decode_num(fields[15], "Jacobian columns")?,
            jacobian_nnz: (fields[16] != "none")
                .then(|| decode_num(fields[16], "Jacobian nnz"))
                .transpose()?,
            jacobian_layout: decode_layout(fields[17])?,
        },
        functions: GeneratedFunctionsManifest {
            residual_fn_name: decode_text(fields[18])?,
            residual_chunk_names: decode_strings(fields[19])?,
            residual_chunks: decode_chunks(fields[20])?,
            jacobian_fn_name: decode_text(fields[21])?,
            jacobian_chunk_names: decode_strings(fields[22])?,
            jacobian_chunks: decode_chunks(fields[23])?,
        },
        expression_signature: fields[24]
            .parse::<u64>()
            .map_err(|_| invalid_handoff("invalid expression signature in AOT handoff"))?,
    };
    Ok(RegisteredAotArtifact {
        problem_key: decode_text(fields[0])?,
        crate_name: decode_text(fields[1])?,
        manifest,
        crate_dir: PathBuf::from(decode_text(fields[2])?),
        manifest_file: PathBuf::from(decode_text(fields[3])?),
        artifact_dir: PathBuf::from(decode_text(fields[4])?),
        expected_rlib: PathBuf::from(decode_text(fields[5])?),
        expected_cdylib: PathBuf::from(decode_text(fields[6])?),
        cargo_program: decode_text(fields[7])?,
        cargo_args: decode_strings(fields[8])?,
        codegen_backend: if has_backend_provenance {
            decode_codegen_backend(fields[25])?
        } else {
            None
        },
    })
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
        assert!(
            registered
                .lifecycle_contract_issues()
                .iter()
                .any(|issue| issue.contains("compiled artifact is missing"))
        );
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
    fn registry_handoff_round_trip_preserves_artifact_provenance() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_registry_handoff_fixture",
            "generated_registry_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");

        let mut registry = AotRegistry::new();
        let registered = registry
            .register_materialized_build(manifest, &build)
            .clone();
        let handoff = dir.path().join("registry-handoff.txt");

        registry
            .write_handoff(&handoff)
            .expect("registry handoff should be writable");
        let restored =
            AotRegistry::read_handoff(&handoff).expect("registry handoff should be readable");

        assert_eq!(restored.len(), 1);
        assert_eq!(
            restored
                .get_by_problem_key(&registered.problem_key)
                .expect("restored problem key should resolve"),
            &registered
        );
        assert_eq!(
            restored
                .get_by_problem_key(&registered.problem_key)
                .and_then(|artifact| artifact.codegen_backend),
            Some(AotCodegenBackend::Rust)
        );
        assert_eq!(
            restored
                .get_by_crate_name(&registered.crate_name)
                .expect("restored crate name should resolve")
                .expected_cdylib,
            registered.expected_cdylib
        );
    }

    #[test]
    fn registry_handoff_v1_remains_readable_without_backend_provenance() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_registry_v1_fixture",
            "generated_registry_v1_module",
            &prepared,
        );
        let dir = tempdir().expect("tempdir should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");
        let mut registry = AotRegistry::new();
        registry.register_materialized_build(manifest, &build);

        let v2_path = dir.path().join("registry-v2.txt");
        registry
            .write_handoff(&v2_path)
            .expect("v2 handoff should be writable");
        let v2 = fs::read_to_string(&v2_path).expect("v2 handoff should be readable");
        let v1 = v2
            .lines()
            .map(|line| {
                if line.starts_with("RST_AOT_REGISTRY_HANDOFF|") {
                    line.replace("version=2", "version=1")
                } else if let Some(entry) = line.strip_prefix("entry|") {
                    let mut fields = entry.split('|').collect::<Vec<_>>();
                    fields.pop();
                    format!("entry|{}", fields.join("|"))
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let v1_path = dir.path().join("registry-v1.txt");
        fs::write(&v1_path, v1).expect("v1 handoff should be writable");

        let restored = AotRegistry::read_handoff(&v1_path).expect("v1 handoff should be readable");
        let artifact = restored
            .problem_keys()
            .first()
            .and_then(|key| restored.get_by_problem_key(key))
            .expect("v1 artifact should be restored");
        assert_eq!(artifact.codegen_backend, None);
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
        assert!(
            refreshed
                .lifecycle_contract_summary()
                .contains("manifest_key_matches=true")
        );
    }

    #[test]
    fn registry_quarantines_materialized_tree_before_rebuild() {
        let prepared = sample_prepared_problem();
        let manifest = PreparedProblemManifest::from(&prepared);
        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_registry_quarantine_fixture",
            "generated_registry_module",
            &prepared,
        );
        let dir = tempdir().expect("temporary directory should exist");
        let build = AotBuildRequest::new(crate_spec, dir.path(), AotBuildProfile::Debug)
            .materialize()
            .expect("build request should materialize");
        let mut registry = AotRegistry::new();
        let key = registry
            .register_materialized_build(manifest, &build)
            .problem_key
            .clone();

        let quarantined = registry
            .quarantine_artifact_by_problem_key(&key)
            .expect("quarantine should be typed and successful")
            .expect("materialized tree should be quarantined");
        assert!(!build.written.crate_dir.exists());
        assert!(quarantined.exists());
        assert!(registry.get_by_problem_key(&key).is_some());
        fs::remove_dir_all(quarantined).expect("test quarantine should be removable");
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
        assert!(
            registry
                .get_by_crate_name("generated_registry_cleanup_fixture")
                .is_none()
        );
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
