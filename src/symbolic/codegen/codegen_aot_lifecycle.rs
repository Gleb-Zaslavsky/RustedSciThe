//! Shared typed state and diagnostics for generated AOT artifacts.
//!
//! The lifecycle is deliberately independent of a compiler language. Rust,
//! C and Zig materializers can report the same states and failure classes
//! without putting strings or a `HashMap` on a callback hot path.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Coarse lifecycle stage of one generated artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AotLifecycleStage {
    Planned,
    Materialized,
    Build,
    Link,
    Published,
    RuntimeReady,
}

/// Stable failure class used by all AOT toolchains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AotFailureKind {
    Compiler,
    PartialArtifact,
    StaleArtifact,
    Lock,
    Link,
    Manifest,
    OutputShape,
    RetryExhausted,
    Quarantined,
    Io,
}

/// On-disk state derived from the marker and compiled outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AotArtifactState {
    Missing,
    Materialized,
    Partial,
    Stale,
    Ready,
}

/// Fixed-field inspection result. It is safe to retain in a prepared plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AotArtifactInspection {
    pub state: AotArtifactState,
    pub crate_dir: PathBuf,
    pub marker_exists: bool,
    pub static_output_exists: bool,
    pub dynamic_output_exists: bool,
}

impl AotArtifactInspection {
    /// Inspects only filesystem facts; it never opens or loads a library.
    pub fn inspect(
        crate_dir: impl Into<PathBuf>,
        marker: &Path,
        static_output: &Path,
        dynamic_output: &Path,
    ) -> Self {
        let crate_dir = crate_dir.into();
        let marker_exists = marker.is_file();
        let static_output_exists = static_output.is_file();
        let dynamic_output_exists = dynamic_output.is_file();
        let outputs = static_output_exists || dynamic_output_exists;
        let any_expected = marker_exists || outputs;
        let state = if !crate_dir.exists() && !any_expected {
            AotArtifactState::Missing
        } else if marker_exists && outputs {
            AotArtifactState::Ready
        } else if marker_exists && !outputs {
            AotArtifactState::Materialized
        } else if outputs && !marker_exists {
            AotArtifactState::Stale
        } else {
            AotArtifactState::Partial
        };
        Self {
            state,
            crate_dir,
            marker_exists,
            static_output_exists,
            dynamic_output_exists,
        }
    }

    pub const fn is_reusable(&self) -> bool {
        matches!(self.state, AotArtifactState::Ready)
    }
}

/// Partial diagnostics retained when a lifecycle operation fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AotFailureDiagnostics {
    pub stage: AotLifecycleStage,
    pub kind: AotFailureKind,
    pub artifact_key: String,
    pub attempts: u32,
    /// Original failure class when the public `kind` is `RetryExhausted`.
    pub root_kind: Option<AotFailureKind>,
    pub quarantine_attempted: bool,
    pub cleanup_completed: bool,
    pub inspection: Option<AotArtifactInspection>,
    pub detail: String,
}

impl AotFailureDiagnostics {
    pub fn new(
        stage: AotLifecycleStage,
        kind: AotFailureKind,
        artifact_key: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            kind,
            artifact_key: artifact_key.into(),
            attempts: 0,
            root_kind: None,
            quarantine_attempted: false,
            cleanup_completed: false,
            inspection: None,
            detail: detail.into(),
        }
    }
}

/// Typed lifecycle error. The diagnostic payload is deliberately preserved on
/// every failure so callers can report a useful partial story without parsing
/// compiler stderr or a compatibility map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AotLifecycleError {
    pub diagnostics: AotFailureDiagnostics,
}

impl AotLifecycleError {
    pub fn new(diagnostics: AotFailureDiagnostics) -> Self {
        Self { diagnostics }
    }

    pub fn from_io(
        stage: AotLifecycleStage,
        artifact_key: impl Into<String>,
        error: &io::Error,
    ) -> Self {
        Self::new(AotFailureDiagnostics::new(
            stage,
            AotFailureKind::Io,
            artifact_key,
            error.to_string(),
        ))
    }
}

impl fmt::Display for AotLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let diagnostics = &self.diagnostics;
        write!(
            f,
            "AOT lifecycle {:?} failed ({:?}) for '{}': {}",
            diagnostics.stage, diagnostics.kind, diagnostics.artifact_key, diagnostics.detail
        )
    }
}

impl std::error::Error for AotLifecycleError {}

/// Test-only fault points used by lifecycle tests and by future isolated
/// release harnesses. Production requests use `None` and pay no branching in
/// callback execution because this enum is consulted only before a build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AotFailureInjection {
    #[default]
    None,
    Compiler,
    PartialArtifact,
    StaleArtifact,
    Lock,
    Link,
}

/// Moves a non-ready generated tree aside before a rebuild.
///
/// This helper is shared by Rust, C and Zig orchestration. It only touches a
/// directory explicitly supplied by the materialized build result and never
/// recursively deletes user-selected paths.
pub fn quarantine_generated_tree(
    crate_dir: &Path,
    artifact_key: &str,
) -> Result<Option<PathBuf>, AotLifecycleError> {
    if !crate_dir.exists() {
        return Ok(None);
    }
    if !crate_dir.is_dir() {
        return Err(AotLifecycleError::new(AotFailureDiagnostics::new(
            AotLifecycleStage::Materialized,
            AotFailureKind::PartialArtifact,
            artifact_key,
            format!(
                "generated artifact path is not a directory: {}",
                crate_dir.display()
            ),
        )));
    }
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let name = crate_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("generated-aot");
    let quarantine = crate_dir
        .parent()
        .unwrap_or(crate_dir)
        .join(format!("{name}.quarantine-{suffix}"));
    fs::rename(crate_dir, &quarantine).map_err(|error| {
        AotLifecycleError::new(AotFailureDiagnostics::new(
            AotLifecycleStage::Materialized,
            AotFailureKind::Lock,
            artifact_key,
            format!("failed to quarantine generated tree: {error}"),
        ))
    })?;
    Ok(Some(quarantine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn inspection_distinguishes_partial_stale_and_ready_artifacts() {
        let dir = tempdir().expect("temporary directory should exist");
        let crate_dir = dir.path().join("generated");
        let marker = crate_dir.join("manifest.rs");
        let static_output = crate_dir.join("libgenerated.rlib");
        let dynamic_output = crate_dir.join("generated.dll");

        let missing =
            AotArtifactInspection::inspect(&crate_dir, &marker, &static_output, &dynamic_output);
        assert_eq!(missing.state, AotArtifactState::Missing);

        fs::create_dir_all(&crate_dir).expect("crate directory should exist");
        fs::write(&marker, b"marker").expect("marker should be writable");
        let materialized =
            AotArtifactInspection::inspect(&crate_dir, &marker, &static_output, &dynamic_output);
        assert_eq!(materialized.state, AotArtifactState::Materialized);

        fs::remove_file(&marker).expect("marker should be removable");
        fs::write(&static_output, b"stale").expect("stale output should be writable");
        let stale =
            AotArtifactInspection::inspect(&crate_dir, &marker, &static_output, &dynamic_output);
        assert_eq!(stale.state, AotArtifactState::Stale);

        fs::write(&marker, b"marker").expect("marker should be restorable");
        let ready =
            AotArtifactInspection::inspect(&crate_dir, &marker, &static_output, &dynamic_output);
        assert_eq!(ready.state, AotArtifactState::Ready);
        assert!(ready.is_reusable());
    }

    #[test]
    fn failure_keeps_partial_diagnostics_and_fault_class() {
        let mut diagnostics = AotFailureDiagnostics::new(
            AotLifecycleStage::Build,
            AotFailureKind::Compiler,
            "problem-key",
            "compiler exited with status 101",
        );
        diagnostics.attempts = 2;
        diagnostics.quarantine_attempted = true;
        let error = AotLifecycleError::new(diagnostics.clone());
        assert_eq!(error.diagnostics, diagnostics);
        assert!(error.to_string().contains("Compiler"));
        assert!(error.to_string().contains("problem-key"));
    }
}
