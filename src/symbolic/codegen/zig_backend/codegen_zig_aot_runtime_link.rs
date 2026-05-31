//! Runtime registry for dynamically loaded Zig AOT backends.
//!
//! Mirrors `codegen_c_aot_runtime_link.rs` but for Zig compiled libraries.
//! The FFI interface is identical — Zig exports the same C-ABI symbols.

use crate::symbolic::codegen::codegen_aot_registry::RegisteredAotArtifact;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    register_linked_dense_backend, register_linked_residual_backend,
    register_linked_sparse_backend, LinkedDenseAotBackend, LinkedResidualAotBackend,
    LinkedResidualChunk, LinkedSparseAotBackend, LinkedSparseJacobianChunk,
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

/// Registers a compiled Zig AOT residual-only backend.
pub fn register_generated_zig_residual_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedResidualAotBackend, String> {
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
    register_linked_residual_backend(backend.clone());
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
    register_linked_sparse_backend(backend.clone());
    info!(
        "Registered Zig sparse AOT backend with problem_key='{}'",
        artifact.problem_key
    );
    Ok(backend)
}

/// Registers a compiled Zig AOT banded backend.
///
/// Banded codegen currently shares the same flat Jacobian-values ABI as the
/// sparse backend; the solver-facing layer reconstructs native banded storage.
pub fn register_generated_zig_banded_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    register_generated_zig_sparse_backend(artifact)
}

/// Registers a compiled Zig AOT dense backend from a registered artifact.
pub fn register_generated_zig_dense_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedDenseAotBackend, String> {
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
    register_linked_dense_backend(backend.clone());
    info!(
        "Registered Zig dense AOT backend with problem_key='{}'",
        artifact.problem_key
    );
    Ok(backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_zig_library_fails_gracefully_for_missing_file() {
        let result = load_zig_library(Path::new("/nonexistent/path/lib.so"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to load Zig library"));
    }
}
