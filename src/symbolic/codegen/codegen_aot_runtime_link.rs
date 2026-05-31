//! Runtime registry for statically linked AOT backends.
//!
//! The build/registry/resolution layers can tell us that an AOT artifact exists
//! on disk, but outer solver code still needs an in-process way to call that
//! backend. This module provides a small process-local registry keyed by
//! manifest `problem_key`.
//!
//! In a future production integration, generated AOT crates can register
//! themselves here during program startup or through an explicit initialization
//! hook. The solver-facing symbolic/BVP layers can then turn `AotCompiled`
//! selection into ordinary residual/Jacobian callbacks.

use crate::symbolic::codegen::codegen_aot_registry::RegisteredAotArtifact;
use libloading::Library;
use log::warn;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

/// Shared residual evaluator signature for a linked sparse AOT backend.
pub type LinkedResidualEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared dense Jacobian evaluator signature for a linked dense AOT backend.
pub type LinkedDenseJacobianEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared sparse Jacobian-values evaluator signature for a linked sparse AOT backend.
pub type LinkedSparseJacobianEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared residual chunk evaluator signature for linked chunked AOT execution.
pub type LinkedResidualChunkEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared dense Jacobian chunk evaluator signature for linked chunked AOT execution.
pub type LinkedDenseJacobianChunkEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared sparse Jacobian chunk evaluator signature for linked chunked AOT execution.
pub type LinkedSparseJacobianChunkEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

type AbiWholeEval = unsafe extern "C" fn(*const f64, usize, *mut f64, usize) -> bool;

struct LoadedSparseCdylib {
    _library: Library,
    residual_eval: AbiWholeEval,
    jacobian_values_eval: AbiWholeEval,
}

struct LoadedDenseCdylib {
    _library: Library,
    residual_eval: AbiWholeEval,
    jacobian_eval: AbiWholeEval,
}

struct LoadedResidualCdylib {
    _library: Library,
    residual_eval: AbiWholeEval,
}

/// One linked residual chunk callback writing into a disjoint output slice.
#[derive(Clone)]
pub struct LinkedResidualChunk {
    /// Global offset of the first residual entry written by this chunk.
    pub output_offset: usize,
    /// Number of residual outputs produced by this chunk.
    pub output_len: usize,
    /// Chunk evaluator over flattened `[params..., variables...]` inputs.
    pub eval: Arc<LinkedResidualChunkEval>,
}

impl LinkedResidualChunk {
    /// Creates one linked residual chunk callback.
    pub fn new(
        output_offset: usize,
        output_len: usize,
        eval: Arc<LinkedResidualChunkEval>,
    ) -> Self {
        Self {
            output_offset,
            output_len,
            eval,
        }
    }
}

/// One linked dense Jacobian chunk callback writing into a disjoint row-major slice.
#[derive(Clone)]
pub struct LinkedDenseJacobianChunk {
    /// Global offset of the first dense Jacobian entry written by this chunk.
    pub value_offset: usize,
    /// Number of dense Jacobian entries produced by this chunk.
    pub value_len: usize,
    /// Chunk evaluator over flattened `[params..., variables...]` inputs.
    pub eval: Arc<LinkedDenseJacobianChunkEval>,
}

impl LinkedDenseJacobianChunk {
    /// Creates one linked dense Jacobian chunk callback.
    pub fn new(
        value_offset: usize,
        value_len: usize,
        eval: Arc<LinkedDenseJacobianChunkEval>,
    ) -> Self {
        Self {
            value_offset,
            value_len,
            eval,
        }
    }
}

/// One linked sparse Jacobian-values chunk callback writing into a disjoint explicit-value slice.
#[derive(Clone)]
pub struct LinkedSparseJacobianChunk {
    /// Global explicit-value offset written by this chunk.
    pub value_offset: usize,
    /// Number of sparse explicit values produced by this chunk.
    pub value_len: usize,
    /// Chunk evaluator over flattened `[params..., variables...]` inputs.
    pub eval: Arc<LinkedSparseJacobianChunkEval>,
}

impl LinkedSparseJacobianChunk {
    /// Creates one linked sparse Jacobian chunk callback.
    pub fn new(
        value_offset: usize,
        value_len: usize,
        eval: Arc<LinkedSparseJacobianChunkEval>,
    ) -> Self {
        Self {
            value_offset,
            value_len,
            eval,
        }
    }
}

/// Process-local linked dense backend.
#[derive(Clone)]
pub struct LinkedDenseAotBackend {
    /// Manifest-derived problem key used to reconnect the linked backend.
    pub problem_key: String,
    /// Number of residual outputs produced by `residual_eval`.
    pub residual_len: usize,
    /// Dense Jacobian shape `(rows, cols)`.
    pub shape: (usize, usize),
    /// Residual evaluator over flattened `[params..., variables...]` inputs.
    pub residual_eval: Arc<LinkedResidualEval>,
    /// Dense Jacobian evaluator over flattened `[params..., variables...]` inputs.
    pub jacobian_eval: Arc<LinkedDenseJacobianEval>,
    /// Optional residual chunk evaluators for runtime sequential/parallel orchestration.
    pub residual_chunks: Vec<LinkedResidualChunk>,
    /// Optional dense Jacobian chunk evaluators for runtime sequential/parallel orchestration.
    pub jacobian_chunks: Vec<LinkedDenseJacobianChunk>,
}

impl LinkedDenseAotBackend {
    /// Creates a new linked dense backend entry.
    pub fn new(
        problem_key: impl Into<String>,
        residual_len: usize,
        shape: (usize, usize),
        residual_eval: Arc<LinkedResidualEval>,
        jacobian_eval: Arc<LinkedDenseJacobianEval>,
    ) -> Self {
        Self {
            problem_key: problem_key.into(),
            residual_len,
            shape,
            residual_eval,
            jacobian_eval,
            residual_chunks: Vec::new(),
            jacobian_chunks: Vec::new(),
        }
    }

    /// Adds optional chunked evaluators that can be used by runtime execution policies.
    pub fn with_chunked_evaluators(
        mut self,
        residual_chunks: Vec<LinkedResidualChunk>,
        jacobian_chunks: Vec<LinkedDenseJacobianChunk>,
    ) -> Self {
        self.residual_chunks = residual_chunks;
        self.jacobian_chunks = jacobian_chunks;
        self
    }
}

/// Process-local linked residual-only backend.
#[derive(Clone)]
pub struct LinkedResidualAotBackend {
    /// Manifest-derived problem key used to reconnect the linked backend.
    pub problem_key: String,
    /// Number of residual outputs produced by `residual_eval`.
    pub residual_len: usize,
    /// Residual evaluator over flattened IVP inputs.
    pub residual_eval: Arc<LinkedResidualEval>,
    /// Optional residual chunk evaluators for runtime sequential/parallel orchestration.
    pub residual_chunks: Vec<LinkedResidualChunk>,
}

impl LinkedResidualAotBackend {
    /// Creates a new linked residual-only backend entry.
    pub fn new(
        problem_key: impl Into<String>,
        residual_len: usize,
        residual_eval: Arc<LinkedResidualEval>,
    ) -> Self {
        Self {
            problem_key: problem_key.into(),
            residual_len,
            residual_eval,
            residual_chunks: Vec::new(),
        }
    }

    /// Adds optional chunked evaluators that can be used by runtime execution policies.
    pub fn with_chunked_evaluators(mut self, residual_chunks: Vec<LinkedResidualChunk>) -> Self {
        self.residual_chunks = residual_chunks;
        self
    }
}

/// Process-local linked sparse backend.
#[derive(Clone)]
pub struct LinkedSparseAotBackend {
    /// Manifest-derived problem key used to reconnect the linked backend.
    pub problem_key: String,
    /// Number of residual outputs produced by `residual_eval`.
    pub residual_len: usize,
    /// Sparse Jacobian shape `(rows, cols)`.
    pub shape: (usize, usize),
    /// Number of sparse Jacobian nonzeros expected by `jacobian_values_eval`.
    pub nnz: usize,
    /// Residual evaluator over flattened `[params..., variables...]` inputs.
    pub residual_eval: Arc<LinkedResidualEval>,
    /// Sparse Jacobian values evaluator over flattened `[params..., variables...]` inputs.
    pub jacobian_values_eval: Arc<LinkedSparseJacobianEval>,
    /// Optional residual chunk evaluators for runtime sequential/parallel orchestration.
    pub residual_chunks: Vec<LinkedResidualChunk>,
    /// Optional sparse Jacobian value chunk evaluators for runtime sequential/parallel orchestration.
    pub jacobian_value_chunks: Vec<LinkedSparseJacobianChunk>,
}

impl LinkedSparseAotBackend {
    /// Creates a new linked sparse backend entry.
    pub fn new(
        problem_key: impl Into<String>,
        residual_len: usize,
        shape: (usize, usize),
        nnz: usize,
        residual_eval: Arc<LinkedResidualEval>,
        jacobian_values_eval: Arc<LinkedSparseJacobianEval>,
    ) -> Self {
        Self {
            problem_key: problem_key.into(),
            residual_len,
            shape,
            nnz,
            residual_eval,
            jacobian_values_eval,
            residual_chunks: Vec::new(),
            jacobian_value_chunks: Vec::new(),
        }
    }

    /// Adds optional chunked evaluators that can be used by runtime execution policies.
    pub fn with_chunked_evaluators(
        mut self,
        residual_chunks: Vec<LinkedResidualChunk>,
        jacobian_value_chunks: Vec<LinkedSparseJacobianChunk>,
    ) -> Self {
        self.residual_chunks = residual_chunks;
        self.jacobian_value_chunks = jacobian_value_chunks;
        self
    }
}

fn linked_sparse_registry() -> &'static Mutex<BTreeMap<String, LinkedSparseAotBackend>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<String, LinkedSparseAotBackend>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn linked_residual_registry() -> &'static Mutex<BTreeMap<String, LinkedResidualAotBackend>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<String, LinkedResidualAotBackend>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn linked_dense_registry() -> &'static Mutex<BTreeMap<String, LinkedDenseAotBackend>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<String, LinkedDenseAotBackend>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Registers one linked dense AOT backend in the current process.
pub fn register_linked_dense_backend(backend: LinkedDenseAotBackend) {
    linked_dense_registry()
        .lock()
        .expect("linked dense AOT registry lock poisoned")
        .insert(backend.problem_key.clone(), backend);
}

/// Looks up a linked dense AOT backend by manifest-derived problem key.
pub fn resolve_linked_dense_backend(problem_key: &str) -> Option<LinkedDenseAotBackend> {
    linked_dense_registry()
        .lock()
        .expect("linked dense AOT registry lock poisoned")
        .get(problem_key)
        .cloned()
}

/// Removes one linked dense AOT backend from the current process registry.
pub fn unregister_linked_dense_backend(problem_key: &str) -> Option<LinkedDenseAotBackend> {
    linked_dense_registry()
        .lock()
        .expect("linked dense AOT registry lock poisoned")
        .remove(problem_key)
}

/// Registers one linked residual-only AOT backend in the current process.
pub fn register_linked_residual_backend(backend: LinkedResidualAotBackend) {
    linked_residual_registry()
        .lock()
        .expect("linked residual AOT registry lock poisoned")
        .insert(backend.problem_key.clone(), backend);
}

/// Looks up a linked residual-only AOT backend by manifest-derived problem key.
pub fn resolve_linked_residual_backend(problem_key: &str) -> Option<LinkedResidualAotBackend> {
    linked_residual_registry()
        .lock()
        .expect("linked residual AOT registry lock poisoned")
        .get(problem_key)
        .cloned()
}

/// Removes one linked residual-only AOT backend from the current process registry.
pub fn unregister_linked_residual_backend(problem_key: &str) -> Option<LinkedResidualAotBackend> {
    linked_residual_registry()
        .lock()
        .expect("linked residual AOT registry lock poisoned")
        .remove(problem_key)
}

fn load_residual_cdylib(path: &Path) -> Result<Arc<LoadedResidualCdylib>, String> {
    let library = unsafe { Library::new(path) }
        .map_err(|err| format!("failed to load cdylib '{}': {err}", path.display()))?;
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
    Ok(Arc::new(LoadedResidualCdylib {
        _library: library,
        residual_eval,
    }))
}

/// Registers one linked sparse AOT backend in the current process.
pub fn register_linked_sparse_backend(backend: LinkedSparseAotBackend) {
    linked_sparse_registry()
        .lock()
        .expect("linked sparse AOT registry lock poisoned")
        .insert(backend.problem_key.clone(), backend);
}

/// Looks up a linked sparse AOT backend by manifest-derived problem key.
pub fn resolve_linked_sparse_backend(problem_key: &str) -> Option<LinkedSparseAotBackend> {
    linked_sparse_registry()
        .lock()
        .expect("linked sparse AOT registry lock poisoned")
        .get(problem_key)
        .cloned()
}

/// Removes one linked sparse AOT backend from the current process registry.
pub fn unregister_linked_sparse_backend(problem_key: &str) -> Option<LinkedSparseAotBackend> {
    linked_sparse_registry()
        .lock()
        .expect("linked sparse AOT registry lock poisoned")
        .remove(problem_key)
}

fn load_sparse_cdylib(path: &Path) -> Result<Arc<LoadedSparseCdylib>, String> {
    let library = unsafe { Library::new(path) }
        .map_err(|err| format!("failed to load cdylib '{}': {err}", path.display()))?;
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
    let jacobian_values_eval = unsafe {
        *library
            .get::<AbiWholeEval>(b"rustedscithe_aot_eval_jacobian_values")
            .map_err(|err| {
                format!(
                    "failed to resolve symbol rustedscithe_aot_eval_jacobian_values from '{}': {err}",
                    path.display()
                )
            })?
    };
    Ok(Arc::new(LoadedSparseCdylib {
        _library: library,
        residual_eval,
        jacobian_values_eval,
    }))
}

fn chunk_export_symbol(fn_name: &str) -> String {
    format!("rustedscithe_aot_chunk_{fn_name}")
}

fn load_sparse_residual_chunks(
    loaded: &Arc<LoadedSparseCdylib>,
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
                        "sparse AOT residual chunk symbol '{}' is unavailable for problem_key='{}': {err}; falling back to whole residual callback",
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
                "generated sparse cdylib residual chunk output length mismatch"
            );
            let ok = unsafe { (eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len()) };
            assert!(
                ok,
                "generated sparse cdylib residual chunk callback returned false"
            );
            let _keep_library_loaded = &chunk_loaded;
        });
        chunks.push(LinkedResidualChunk::new(chunk.offset, chunk.len, callback));
    }
    chunks
}

fn load_sparse_jacobian_chunks(
    loaded: &Arc<LoadedSparseCdylib>,
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
                        "sparse AOT Jacobian chunk symbol '{}' is unavailable for problem_key='{}': {err}; falling back to whole Jacobian callback",
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
                "generated sparse cdylib Jacobian chunk output length mismatch"
            );
            let ok = unsafe { (eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len()) };
            assert!(
                ok,
                "generated sparse cdylib Jacobian chunk callback returned false"
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

fn load_dense_cdylib(path: &Path) -> Result<Arc<LoadedDenseCdylib>, String> {
    let library = unsafe { Library::new(path) }
        .map_err(|err| format!("failed to load cdylib '{}': {err}", path.display()))?;
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
    Ok(Arc::new(LoadedDenseCdylib {
        _library: library,
        residual_eval,
        jacobian_eval,
    }))
}

pub fn register_generated_sparse_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled sparse cdylib does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_sparse_cdylib(path)?;
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
            "generated sparse cdylib residual callback returned false"
        );
    });

    let jacobian_loaded = Arc::clone(&loaded);
    let jacobian_values_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (jacobian_loaded.jacobian_values_eval)(
                args.as_ptr(),
                args.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert!(
            ok,
            "generated sparse cdylib jacobian callback returned false"
        );
    });

    let residual_chunks = load_sparse_residual_chunks(&loaded, artifact);
    let jacobian_value_chunks = load_sparse_jacobian_chunks(&loaded, artifact);

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
    Ok(backend)
}

/// Registers a compiled banded cdylib backend.
///
/// Banded generated libraries currently export the same flat Jacobian-values
/// ABI as sparse ones; the outer symbolic/BVP layer decides whether those
/// values are interpreted as sparse triplets or native banded diagonals.
pub fn register_generated_banded_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    register_generated_sparse_cdylib_backend(artifact)
}

pub fn register_generated_dense_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedDenseAotBackend, String> {
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled dense cdylib does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_dense_cdylib(path)?;
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
            "generated dense cdylib residual callback returned false"
        );
    });

    let jacobian_loaded = Arc::clone(&loaded);
    let jacobian_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (jacobian_loaded.jacobian_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(
            ok,
            "generated dense cdylib jacobian callback returned false"
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
    Ok(backend)
}

pub fn register_generated_residual_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedResidualAotBackend, String> {
    let path = &artifact.expected_cdylib;
    if !path.exists() {
        return Err(format!(
            "compiled residual cdylib does not exist at '{}'",
            path.display()
        ));
    }

    let loaded = load_residual_cdylib(path)?;
    let residual_len = artifact.manifest.io.residual_len;
    let residual_loaded = Arc::clone(&loaded);
    let residual_eval = Arc::new(move |args: &[f64], out: &mut [f64]| {
        let ok = unsafe {
            (residual_loaded.residual_eval)(args.as_ptr(), args.len(), out.as_mut_ptr(), out.len())
        };
        assert!(ok, "generated residual cdylib callback returned false");
    });

    let backend =
        LinkedResidualAotBackend::new(artifact.problem_key.clone(), residual_len, residual_eval);
    register_linked_residual_backend(backend.clone());
    Ok(backend)
}
//==========================================================================================
// TESTS
//==========================================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linked_dense_backend_roundtrip_works() {
        let key = "linked_dense_backend_roundtrip";
        let backend = LinkedDenseAotBackend::new(
            key,
            2,
            (2, 2),
            Arc::new(|args, out| {
                out[0] = args[0] + 1.0;
                out[1] = args[1] + 2.0;
            }),
            Arc::new(|args, out| {
                out[0] = args[0];
                out[1] = args[1];
                out[2] = args[0] + args[1];
                out[3] = args[0] - args[1];
            }),
        );

        register_linked_dense_backend(backend.clone());
        let resolved = resolve_linked_dense_backend(key).expect("backend should be registered");
        assert_eq!(resolved.problem_key, key);
        assert_eq!(resolved.residual_len, 2);
        assert_eq!(resolved.shape, (2, 2));

        let mut residual = vec![0.0; 2];
        (resolved.residual_eval)(&[3.0, 4.0], &mut residual);
        assert_eq!(residual, vec![4.0, 6.0]);

        let mut jacobian = vec![0.0; 4];
        (resolved.jacobian_eval)(&[3.0, 4.0], &mut jacobian);
        assert_eq!(jacobian, vec![3.0, 4.0, 7.0, -1.0]);

        unregister_linked_dense_backend(key);
        assert!(resolve_linked_dense_backend(key).is_none());
    }

    #[test]
    fn linked_sparse_backend_roundtrip_works() {
        let key = "linked_sparse_backend_roundtrip";
        let backend = LinkedSparseAotBackend::new(
            key,
            2,
            (2, 2),
            2,
            Arc::new(|args, out| {
                out[0] = args[0] + 1.0;
                out[1] = args[1] + 2.0;
            }),
            Arc::new(|args, out| {
                out[0] = args[0];
                out[1] = args[1];
            }),
        );

        register_linked_sparse_backend(backend.clone());
        let resolved = resolve_linked_sparse_backend(key).expect("backend should be registered");
        assert_eq!(resolved.problem_key, key);
        assert_eq!(resolved.residual_len, 2);
        assert_eq!(resolved.shape, (2, 2));
        assert_eq!(resolved.nnz, 2);

        let mut residual = vec![0.0; 2];
        (resolved.residual_eval)(&[3.0, 4.0], &mut residual);
        assert_eq!(residual, vec![4.0, 6.0]);

        unregister_linked_sparse_backend(key);
        assert!(resolve_linked_sparse_backend(key).is_none());
    }
}
