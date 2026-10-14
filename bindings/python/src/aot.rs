//! Shared translation of the Python AOT vocabulary into native lifecycle data.
//!
//! Python bindings must not contain a second code generator or artifact
//! resolver. This module only converts stable option strings and preserves
//! the native typed lifecycle diagnostics at the language boundary.

use std::{collections::HashMap, path::PathBuf};

use rustedscithe_core::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use rustedscithe_core::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind;
use rustedscithe_core::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use rustedscithe_core::symbolic::symbolic_ivp::{IvpBackendError, IvpSymbolicAssemblyBackend};
use rustedscithe_core::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};

use crate::contract::SolverOptions;
use crate::errors::{
    AotArtifactError, AotCompilerError, AotLifecycleError, AotLinkError, AotTimeoutError,
    InvalidConfigurationError, PreparationError,
};

/// Build the shared native AOT configuration from validated Python options.
pub(crate) fn generated_config(
    options: &SolverOptions,
) -> pyo3::PyResult<SymbolicIvpGeneratedBackendConfig> {
    let profile = match options.aot_profile() {
        "debug" => AotBuildProfile::Debug,
        "release" => AotBuildProfile::Release,
        _ => unreachable!("SolverOptions validates the AOT profile"),
    };
    let policy = match options.aot_policy() {
        "require_prebuilt" => SymbolicIvpAotBuildPolicy::RequirePrebuilt,
        "build_if_missing" => SymbolicIvpAotBuildPolicy::BuildIfMissing { profile },
        "rebuild_always" => SymbolicIvpAotBuildPolicy::RebuildAlways { profile },
        _ => unreachable!("SolverOptions validates the AOT policy"),
    };
    let mut config = SymbolicIvpGeneratedBackendConfig::new().with_build_policy(policy);
    config = match options.aot_toolchain() {
        "gcc" => config
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("gcc"),
        "tcc" => config
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("tcc"),
        "zig" => config.with_aot_codegen_backend(AotCodegenBackend::Zig),
        "rust" => config.with_aot_codegen_backend(AotCodegenBackend::Rust),
        _ => unreachable!("SolverOptions validates the AOT toolchain"),
    };
    if let Some(output_dir) = options.aot_output_dir() {
        config = config.with_output_parent_dir(Some(PathBuf::from(output_dir)));
    }
    if let Some(handoff_path) = options.aot_handoff_path() {
        config = config.with_handoff_path(Some(PathBuf::from(handoff_path)));
    }
    Ok(config)
}

/// Return the AOT settings that must travel with solver provenance.
///
/// Artifact paths and lifecycle choices are operational inputs, not merely
/// implementation details: without them a detached report cannot explain how
/// a generated backend was selected or reproduced.
pub(crate) fn provenance_fields(options: &SolverOptions) -> HashMap<String, String> {
    if options.execution() != "aot" {
        return HashMap::new();
    }
    let mut fields = HashMap::from([
        ("aot_policy".to_owned(), options.aot_policy().to_owned()),
        (
            "aot_toolchain".to_owned(),
            options.aot_toolchain().to_owned(),
        ),
        ("aot_profile".to_owned(), options.aot_profile().to_owned()),
    ]);
    if let Some(path) = options.aot_output_dir() {
        fields.insert("aot_output_dir".to_owned(), path.to_owned());
    }
    if let Some(path) = options.aot_handoff_path() {
        fields.insert("aot_handoff_path".to_owned(), path.to_owned());
    }
    fields
}

/// Preserve the native AOT lifecycle class as a Python exception.
pub(crate) fn map_backend_error(error: IvpBackendError) -> pyo3::PyErr {
    match error {
        IvpBackendError::AotArtifactMissing { message }
        | IvpBackendError::AotArtifactNotBuilt { message } => AotArtifactError::new_err(message),
        IvpBackendError::AotRuntimeUnavailable { message } => AotLinkError::new_err(message),
        IvpBackendError::AotBuildOutputDirMissing => InvalidConfigurationError::new_err(
            "AOT execution requires an output directory for generated artifacts",
        ),
        IvpBackendError::AotLifecycleFailure {
            stage,
            kind,
            artifact_key,
            message,
        } => {
            let detail =
                format!("{message} (stage={stage:?}, kind={kind:?}, artifact={artifact_key})");
            match kind {
                AotFailureKind::Compiler => AotCompilerError::new_err(detail),
                AotFailureKind::Link => AotLinkError::new_err(detail),
                AotFailureKind::Timeout => AotTimeoutError::new_err(detail),
                AotFailureKind::PartialArtifact | AotFailureKind::StaleArtifact => {
                    AotArtifactError::new_err(detail)
                }
                _ => AotLifecycleError::new_err(detail),
            }
        }
        other => PreparationError::new_err(other.to_string()),
    }
}

/// The Python facade exposes only the two symbolic routes currently accepted
/// by all initial IVP adapters. This helper keeps route conversion explicit.
pub(crate) fn symbolic_backend(options: &SolverOptions) -> IvpSymbolicAssemblyBackend {
    match options.frontend() {
        "expr_legacy" => IvpSymbolicAssemblyBackend::ExprLegacy,
        "atom_native" => IvpSymbolicAssemblyBackend::AtomView,
        _ => unreachable!("SolverOptions validates the symbolic frontend"),
    }
}
