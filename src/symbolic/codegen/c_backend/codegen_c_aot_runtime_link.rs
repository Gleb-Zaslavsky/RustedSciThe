//! Runtime registry for dynamically loaded C AOT backends.
//!
//! This module mirrors codegen_aot_runtime_link.rs but for C compiled libraries.
//! It loads .so/.dll/.dylib files and registers their FFI functions in the
//! process-local registry so solvers can call them.

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

type AbiWholeEval = unsafe extern "C" fn(*const f64, usize, *mut f64, usize) -> i32;

#[derive(Debug)]
struct LoadedCLibrary {
    _library: Library,
    residual_eval: AbiWholeEval,
    jacobian_eval: AbiWholeEval,
}

#[derive(Debug)]
struct LoadedCResidualLibrary {
    _library: Library,
    residual_eval: AbiWholeEval,
}

/// Loads a compiled C AOT shared library from disk.
fn load_c_library(path: &Path) -> Result<Arc<LoadedCLibrary>, String> {
    info!("Loading C AOT library from '{}'", path.display());
    let library = unsafe { Library::new(path) }
        .map_err(|err| format!("failed to load C library '{}': {err}", path.display()))?;

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

    Ok(Arc::new(LoadedCLibrary {
        _library: library,
        residual_eval,
        jacobian_eval,
    }))
}

fn chunk_export_symbol(fn_name: &str) -> String {
    format!("rustedscithe_aot_chunk_{fn_name}")
}

fn load_c_sparse_residual_chunks(
    loaded: &Arc<LoadedCLibrary>,
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
                        "C sparse AOT residual chunk symbol '{}' is unavailable for problem_key='{}': {err}; falling back to whole residual callback",
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
                "generated C sparse residual chunk output length mismatch"
            );
            let ok = unsafe { (eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len()) };
            assert!(
                ok != 0,
                "generated C sparse residual chunk callback returned false"
            );
            let _keep_library_loaded = &chunk_loaded;
        });
        chunks.push(LinkedResidualChunk::new(chunk.offset, chunk.len, callback));
    }
    chunks
}

fn load_c_sparse_jacobian_chunks(
    loaded: &Arc<LoadedCLibrary>,
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
                        "C sparse AOT Jacobian chunk symbol '{}' is unavailable for problem_key='{}': {err}; falling back to whole Jacobian callback",
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
                "generated C sparse Jacobian chunk output length mismatch"
            );
            let ok = unsafe { (eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len()) };
            assert!(
                ok != 0,
                "generated C sparse Jacobian chunk callback returned false"
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

fn load_c_residual_library(path: &Path) -> Result<Arc<LoadedCResidualLibrary>, String> {
    info!("Loading C residual AOT library from '{}'", path.display());
    let library = unsafe { Library::new(path) }
        .map_err(|err| format!("failed to load C library '{}': {err}", path.display()))?;

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

    Ok(Arc::new(LoadedCResidualLibrary {
        _library: library,
        residual_eval,
    }))
}

fn ensure_artifact_manifest_key_matches(artifact: &RegisteredAotArtifact) -> Result<(), String> {
    if artifact.manifest_key_matches() {
        return Ok(());
    }
    Err(format!(
        "C AOT artifact manifest key mismatch before dynamic load: {}",
        artifact.lifecycle_contract_summary()
    ))
}

/// Registers a compiled C AOT residual-only backend from a registered artifact.
pub fn register_generated_c_residual_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedResidualAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled C residual library does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_c_residual_library(path)?;
    let residual_len = artifact.manifest.io.residual_len;
    let residual_loaded = Arc::clone(&loaded);
    let residual_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (residual_loaded.residual_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok != 0,
            "generated C residual library callback returned false"
        );
    });

    let backend =
        LinkedResidualAotBackend::new(artifact.problem_key.clone(), residual_len, residual_eval);
    register_linked_residual_backend(backend.clone());
    info!(
        "Registered C residual AOT backend with problem_key='{}'",
        artifact.problem_key
    );
    Ok(backend)
}

/// Registers a compiled C AOT sparse backend from a registered artifact.
///
/// This function:
/// 1. Loads the .so/.dll/.dylib file
/// 2. Resolves the FFI symbols
/// 3. Wraps them in safe Rust closures
/// 4. Registers the backend in the global runtime registry
pub fn register_generated_c_sparse_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled C sparse library does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_c_library(path)?;
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
            ok != 0,
            "generated C sparse library residual callback returned false"
        );
    });

    let jacobian_loaded = Arc::clone(&loaded);
    let jacobian_values_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (jacobian_loaded.jacobian_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok != 0,
            "generated C sparse library jacobian callback returned false"
        );
    });

    let residual_chunks = load_c_sparse_residual_chunks(&loaded, artifact);
    let jacobian_value_chunks = load_c_sparse_jacobian_chunks(&loaded, artifact);

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
        "Registered C sparse AOT backend with problem_key='{}'",
        artifact.problem_key
    );
    Ok(backend)
}

/// Registers a compiled C AOT banded backend.
///
/// Banded codegen currently shares the same flat Jacobian-values ABI as the
/// sparse backend; the solver-facing layer reconstructs native banded storage.
pub fn register_generated_c_banded_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    register_generated_c_sparse_backend(artifact)
}

/// Registers a compiled C AOT dense backend from a registered artifact.
///
/// This function:
/// 1. Loads the .so/.dll/.dylib file
/// 2. Resolves the FFI symbols
/// 3. Wraps them in safe Rust closures
/// 4. Registers the backend in the global runtime registry
pub fn register_generated_c_dense_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedDenseAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled C dense library does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_c_library(path)?;
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
            ok != 0,
            "generated C dense library residual callback returned false"
        );
    });

    let jacobian_loaded = Arc::clone(&loaded);
    let jacobian_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (jacobian_loaded.jacobian_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok != 0,
            "generated C dense library jacobian callback returned false"
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
        "Registered C dense AOT backend with problem_key='{}'",
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
            io: ProblemIoManifest {
                input_names: vec!["x".to_string()],
                residual_len: 1,
                jacobian_rows: 1,
                jacobian_cols: 1,
                jacobian_nnz: Some(1),
            },
            functions: GeneratedFunctionsManifest {
                residual_fn_name: "eval_residual".to_string(),
                residual_chunk_names: Vec::new(),
                residual_chunks: Vec::new(),
                jacobian_fn_name: "eval_jacobian_values".to_string(),
                jacobian_chunk_names: Vec::new(),
                jacobian_chunks: Vec::new(),
            },
            expression_signature: 7,
        }
    }

    fn mismatched_artifact() -> RegisteredAotArtifact {
        RegisteredAotArtifact {
            problem_key: "stale-c-problem-key".to_string(),
            crate_name: "generated_c_mismatch_fixture".to_string(),
            manifest: dummy_manifest(),
            crate_dir: PathBuf::from("generated_c_mismatch_fixture"),
            manifest_file: PathBuf::from("generated_c_mismatch_fixture/aot_manifest.h"),
            artifact_dir: PathBuf::from("generated_c_mismatch_fixture/build"),
            expected_rlib: PathBuf::from("generated_c_mismatch_fixture/build/libfixture.a"),
            expected_cdylib: PathBuf::from("generated_c_mismatch_fixture/build/fixture.dll"),
            cargo_program: "tcc".to_string(),
            cargo_args: Vec::new(),
        }
    }

    #[test]
    fn generated_c_registration_rejects_manifest_key_mismatch_before_loading() {
        let artifact = mismatched_artifact();
        let err = match register_generated_c_sparse_backend(&artifact) {
            Ok(_) => panic!("mismatched C artifact must not be dynamically loaded"),
            Err(err) => err,
        };

        assert!(
            err.contains("manifest key mismatch"),
            "unexpected error: {err}"
        );
        assert!(
            !err.contains("compiled C sparse library does not exist"),
            "manifest mismatch must be checked before filesystem/load errors: {err}"
        );
    }

    #[test]
    fn load_c_library_fails_gracefully_for_missing_file() {
        let result = load_c_library(Path::new("/nonexistent/path/lib.so"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to load C library"));
    }
}
