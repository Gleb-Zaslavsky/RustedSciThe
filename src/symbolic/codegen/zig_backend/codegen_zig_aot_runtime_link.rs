//! Runtime registry for dynamically loaded Zig AOT backends.
//!
//! Mirrors `codegen_c_aot_runtime_link.rs` but for Zig compiled libraries.
//! The FFI interface is identical — Zig exports the same C-ABI symbols.

use crate::symbolic::codegen::codegen_aot_registry::RegisteredAotArtifact;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    try_register_linked_dense_backend, try_register_linked_residual_backend,
    try_register_linked_sparse_backend, LinkedDenseAotBackend, LinkedResidualAotBackend,
    LinkedResidualChunk, LinkedSparseAotBackend, LinkedSparseJacobianChunk,
    validate_compact_banded_manifest,
};
use libloading::Library;
use log::{info, warn};
use std::path::Path;
use std::sync::Arc;

type AbiWholeEval = unsafe extern "C" fn(*const f64, usize, *mut f64, usize) -> bool;

#[derive(Debug)]
struct LoadedZigLibrary {
    _library: Library,
    residual_eval: AbiWholeEval,
    jacobian_eval: AbiWholeEval,
}

#[derive(Debug)]
struct LoadedZigResidualLibrary {
    _library: Library,
    residual_eval: AbiWholeEval,
}

fn load_zig_library(path: &Path) -> Result<Arc<LoadedZigLibrary>, String> {
    info!("Loading Zig AOT library from '{}'", path.display());
    let library = unsafe { Library::new(path) }
        .map_err(|err| format!("failed to load Zig library '{}': {err}", path.display()))?;

    let residual_eval = unsafe {
        *library
            .get::<AbiWholeEval>(b"rustedscithe_aot_eval_residual")
            .map_err(|err| {
                format!(
                    "failed to resolve symbol rustedscithe_aot_eval_residual from '{}': {err}",
                    path.display()
                )
            })?
    };

    let jacobian_eval = unsafe {
        *library
            .get::<AbiWholeEval>(b"rustedscithe_aot_eval_jacobian_values")
            .map_err(|err| {
                format!(
                    "failed to resolve symbol rustedscithe_aot_eval_jacobian_values from '{}': {err}",
                    path.display()
                )
            })?
    };

    Ok(Arc::new(LoadedZigLibrary {
        _library: library,
        residual_eval,
        jacobian_eval,
    }))
}

fn chunk_export_symbol(fn_name: &str) -> String {
    format!("rustedscithe_aot_chunk_{fn_name}")
}

fn load_zig_sparse_residual_chunks(
    loaded: &Arc<LoadedZigLibrary>,
    artifact: &RegisteredAotArtifact,
) -> Vec<LinkedResidualChunk> {
    let mut chunks = Vec::with_capacity(artifact.manifest.functions.residual_chunks.len());
    for chunk in &artifact.manifest.functions.residual_chunks {
        let symbol_name = chunk_export_symbol(&chunk.fn_name);
        let eval = unsafe {
            match loaded._library.get::<AbiWholeEval>(symbol_name.as_bytes()) {
                Ok(symbol) => *symbol,
                Err(err) => {
                    warn!(
                        "Zig sparse AOT residual chunk symbol '{}' is unavailable for problem_key='{}': {err}; falling back to whole residual callback",
                        symbol_name, artifact.problem_key
                    );
                    return Vec::new();
                }
            }
        };
        let chunk_loaded = Arc::clone(loaded);
        let output_len = chunk.len;
        let callback = Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert_eq!(
                out.len(),
                output_len,
                "generated Zig sparse residual chunk output length mismatch"
            );
            let ok = unsafe { (eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len()) };
            assert!(
                ok,
                "generated Zig sparse residual chunk callback returned false"
            );
            let _keep_library_loaded = &chunk_loaded;
        });
        chunks.push(LinkedResidualChunk::new(chunk.offset, chunk.len, callback));
    }
    chunks
}

fn load_zig_sparse_jacobian_chunks(
    loaded: &Arc<LoadedZigLibrary>,
    artifact: &RegisteredAotArtifact,
) -> Vec<LinkedSparseJacobianChunk> {
    let mut chunks = Vec::with_capacity(artifact.manifest.functions.jacobian_chunks.len());
    for chunk in &artifact.manifest.functions.jacobian_chunks {
        let symbol_name = chunk_export_symbol(&chunk.fn_name);
        let eval = unsafe {
            match loaded._library.get::<AbiWholeEval>(symbol_name.as_bytes()) {
                Ok(symbol) => *symbol,
                Err(err) => {
                    warn!(
                        "Zig sparse AOT Jacobian chunk symbol '{}' is unavailable for problem_key='{}': {err}; falling back to whole Jacobian callback",
                        symbol_name, artifact.problem_key
                    );
                    return Vec::new();
                }
            }
        };
        let chunk_loaded = Arc::clone(loaded);
        let value_len = chunk.len;
        let callback = Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert_eq!(
                out.len(),
                value_len,
                "generated Zig sparse Jacobian chunk output length mismatch"
            );
            let ok = unsafe { (eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len()) };
            assert!(
                ok,
                "generated Zig sparse Jacobian chunk callback returned false"
            );
            let _keep_library_loaded = &chunk_loaded;
        });
        chunks.push(LinkedSparseJacobianChunk::new(
            chunk.offset,
            chunk.len,
            callback,
        ));
    }
    chunks
}

fn load_zig_residual_library(path: &Path) -> Result<Arc<LoadedZigResidualLibrary>, String> {
    info!("Loading Zig residual AOT library from '{}'", path.display());
    let library = unsafe { Library::new(path) }
        .map_err(|err| format!("failed to load Zig library '{}': {err}", path.display()))?;
    let residual_eval = unsafe {
        *library
            .get::<AbiWholeEval>(b"rustedscithe_aot_eval_residual")
            .map_err(|err| {
                format!(
                    "failed to resolve symbol rustedscithe_aot_eval_residual from '{}': {err}",
                    path.display()
                )
            })?
    };
    Ok(Arc::new(LoadedZigResidualLibrary {
        _library: library,
        residual_eval,
    }))
}

fn ensure_artifact_manifest_key_matches(artifact: &RegisteredAotArtifact) -> Result<(), String> {
    if artifact.manifest_key_matches() {
        return Ok(());
    }
    Err(format!(
        "Zig AOT artifact manifest key mismatch before dynamic load: {}",
        artifact.lifecycle_contract_summary()
    ))
}

/// Registers a compiled Zig AOT residual-only backend.
pub fn register_generated_zig_residual_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedResidualAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled Zig residual library does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_zig_residual_library(path)?;
    let residual_len = artifact.manifest.io.residual_len;
    let residual_loaded = Arc::clone(&loaded);
    let residual_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (residual_loaded.residual_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(ok, "generated Zig residual library callback returned false");
    });

    let backend =
        LinkedResidualAotBackend::new(artifact.problem_key.clone(), residual_len, residual_eval);
    try_register_linked_residual_backend(backend.clone()).map_err(|error| error.to_string())?;
    info!(
        "Registered Zig residual AOT backend with problem_key='{}'",
        artifact.problem_key
    );
    Ok(backend)
}

/// Registers a compiled Zig AOT sparse backend from a registered artifact.
pub fn register_generated_zig_sparse_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled Zig sparse library does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_zig_library(path)?;
    let residual_len = artifact.manifest.io.residual_len;
    let shape = (
        artifact.manifest.io.jacobian_rows,
        artifact.manifest.io.jacobian_cols,
    );
    let nnz = artifact
        .manifest
        .io
        .jacobian_nnz
        .unwrap_or(shape.0 * shape.1);

    let residual_loaded = Arc::clone(&loaded);
    let residual_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (residual_loaded.residual_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok,
            "generated Zig sparse library residual callback returned false"
        );
    });

    let jacobian_loaded = Arc::clone(&loaded);
    let jacobian_values_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (jacobian_loaded.jacobian_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok,
            "generated Zig sparse library jacobian callback returned false"
        );
    });

    let residual_chunks = load_zig_sparse_residual_chunks(&loaded, artifact);
    let jacobian_value_chunks = load_zig_sparse_jacobian_chunks(&loaded, artifact);

    let backend = LinkedSparseAotBackend::new(
        artifact.problem_key.clone(),
        residual_len,
        shape,
        nnz,
        residual_eval,
        jacobian_values_eval,
    )
    .with_chunked_evaluators(residual_chunks, jacobian_value_chunks);
    try_register_linked_sparse_backend(backend.clone()).map_err(|error| error.to_string())?;
    info!(
        "Registered Zig sparse AOT backend with problem_key='{}'",
        artifact.problem_key
    );
    Ok(backend)
}

/// Registers a compiled Zig AOT banded backend.
///
/// Registers a generated Banded library while preserving its manifest-declared
/// explicit-entry or AtomView-native compact-slot ABI.
pub fn register_generated_zig_banded_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    let compact_layout = validate_compact_banded_manifest(&artifact.manifest)?;
    let backend = register_generated_zig_sparse_backend(artifact)?;
    let Some((rows, cols, kl, ku)) = compact_layout else {
        return Ok(backend);
    };
    let backend = backend.with_banded_compact_layout(rows, cols, kl, ku);
    try_register_linked_sparse_backend(backend.clone()).map_err(|error| error.to_string())?;
    Ok(backend)
}

/// Registers a compiled Zig AOT dense backend from a registered artifact.
pub fn register_generated_zig_dense_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedDenseAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled Zig dense library does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_zig_library(path)?;
    let residual_len = artifact.manifest.io.residual_len;
    let shape = (
        artifact.manifest.io.jacobian_rows,
        artifact.manifest.io.jacobian_cols,
    );

    let residual_loaded = Arc::clone(&loaded);
    let residual_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (residual_loaded.residual_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok,
            "generated Zig dense library residual callback returned false"
        );
    });

    let jacobian_loaded = Arc::clone(&loaded);
    let jacobian_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (jacobian_loaded.jacobian_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok,
            "generated Zig dense library jacobian callback returned false"
        );
    });

    let backend = LinkedDenseAotBackend::new(
        artifact.problem_key.clone(),
        residual_len,
        shape,
        residual_eval,
        jacobian_eval,
    );
    try_register_linked_dense_backend(backend.clone()).map_err(|error| error.to_string())?;
    info!(
        "Registered Zig dense AOT backend with problem_key='{}'",
        artifact.problem_key
    );
    Ok(backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_manifest::{
        GeneratedFunctionsManifest, PreparedProblemManifest, ProblemIoManifest,
    };
    use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend};
    use std::path::PathBuf;

    fn dummy_manifest() -> PreparedProblemManifest {
        PreparedProblemManifest {
            backend_kind: BackendKind::Aot,
            matrix_backend: MatrixBackend::SparseCol,
            symbolic_route: crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::Generic,
            io: ProblemIoManifest {
                input_names: vec!["x".to_string()],
                residual_len: 1,
                jacobian_rows: 1,
                jacobian_cols: 1,
                jacobian_nnz: Some(1),
                jacobian_layout: Some(
                    crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout::SparseExplicit,
                ),
            },
            functions: GeneratedFunctionsManifest {
                residual_fn_name: "eval_residual".to_string(),
                residual_chunk_names: Vec::new(),
                residual_chunks: Vec::new(),
                jacobian_fn_name: "eval_jacobian_values".to_string(),
                jacobian_chunk_names: Vec::new(),
                jacobian_chunks: Vec::new(),
            },
            expression_signature: 11,
        }
    }

    fn mismatched_artifact() -> RegisteredAotArtifact {
        RegisteredAotArtifact {
            problem_key: "stale-zig-problem-key".to_string(),
            crate_name: "generated_zig_mismatch_fixture".to_string(),
            manifest: dummy_manifest(),
            crate_dir: PathBuf::from("generated_zig_mismatch_fixture"),
            manifest_file: PathBuf::from("generated_zig_mismatch_fixture/aot_manifest.zig"),
            artifact_dir: PathBuf::from("generated_zig_mismatch_fixture/zig-out"),
            expected_rlib: PathBuf::from("generated_zig_mismatch_fixture/zig-out/libfixture.a"),
            expected_cdylib: PathBuf::from("generated_zig_mismatch_fixture/zig-out/fixture.dll"),
            cargo_program: "zig".to_string(),
            cargo_args: Vec::new(),
            codegen_backend: Some(crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Zig),
        }
    }

    #[test]
    fn generated_zig_registration_rejects_manifest_key_mismatch_before_loading() {
        let artifact = mismatched_artifact();
        let err = match register_generated_zig_sparse_backend(&artifact) {
            Ok(_) => panic!("mismatched Zig artifact must not be dynamically loaded"),
            Err(err) => err,
        };

        assert!(
            err.contains("manifest key mismatch"),
            "unexpected error: {err}"
        );
        assert!(
            !err.contains("compiled Zig sparse library does not exist"),
            "manifest mismatch must be checked before filesystem/load errors: {err}"
        );
    }

    #[test]
    fn generated_zig_banded_registration_rejects_invalid_compact_manifest_before_loading() {
        let mut artifact = mismatched_artifact();
        artifact.manifest.matrix_backend = MatrixBackend::Banded;
        artifact.manifest.io.jacobian_rows = 3;
        artifact.manifest.io.jacobian_cols = 3;
        artifact.manifest.io.jacobian_nnz = Some(8);
        artifact.manifest.io.jacobian_layout = Some(
            crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout::BandedCompact {
                kl: 1,
                ku: 1,
            },
        );
        artifact.problem_key = artifact.manifest_problem_key();

        let err = match register_generated_zig_banded_backend(&artifact) {
            Ok(_) => panic!("invalid compact manifest must be rejected before loading Zig"),
            Err(err) => err,
        };
        assert!(err.contains("storage length mismatch"), "unexpected error: {err}");
        assert!(
            !err.contains("compiled Zig sparse library does not exist"),
            "manifest validation must precede filesystem/load errors: {err}"
        );
    }

    #[test]
    fn load_zig_library_fails_gracefully_for_missing_file() {
        let result = load_zig_library(Path::new("/nonexistent/path/lib.so"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to load Zig library"));
    }
}
