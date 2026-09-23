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

/// Jacobian value ABI published by one linked generated backend.
///
/// `ExplicitValues` is the historical fixed-structure sparse/banded contract.
/// `BandedCompact` is the AtomView-native full LAPACK-style slot contract. The
/// marker travels with the process-local backend so a solver cannot interpret
/// a complete compact buffer as an explicit coordinate list by accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkedJacobianLayout {
    ExplicitValues,
    BandedCompact {
        rows: usize,
        cols: usize,
        kl: usize,
        ku: usize,
    },
}

/// Validates the manifest contract for a full-slot compact Banded artifact.
///
/// This check is shared by Rust, C and Zig dynamic loaders and intentionally
/// runs before any library is opened. A malformed marker or storage length is
/// a lifecycle error, not a loader/FFI error, so rejecting it early keeps all
/// toolchains on the same diagnostic path.
pub fn validate_compact_banded_manifest(
    manifest: &crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest,
) -> Result<Option<(usize, usize, usize, usize)>, String> {
    let Some(crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout::BandedCompact {
        kl,
        ku,
    }) = manifest.io.jacobian_layout
    else {
        return Ok(None);
    };

    if manifest.matrix_backend
        != crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Banded
    {
        return Err(format!(
            "compact Banded layout requires matrix backend Banded, got {:?}",
            manifest.matrix_backend
        ));
    }

    let rows = manifest.io.jacobian_rows;
    let cols = manifest.io.jacobian_cols;
    let expected_slots =
        crate::symbolic::codegen::codegen_runtime_api::BandedCompactJacobianStructure {
            rows,
            cols,
            kl,
            ku,
        }
        .storage_len()
        .map_err(|error| format!("invalid compact Banded manifest layout: {error}"))?;
    if manifest.io.jacobian_nnz != Some(expected_slots) {
        return Err(format!(
            "compact Banded manifest storage length mismatch: marker requires {expected_slots}, metadata has {:?}",
            manifest.io.jacobian_nnz
        ));
    }

    Ok(Some((rows, cols, kl, ku)))
}

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
    /// Callback storage ABI. Defaults to the retained explicit-values route.
    pub jacobian_layout: LinkedJacobianLayout,
}

/// Typed failure from a linked generated callback.
///
/// The historical callback ABI returns only a boolean inside the dynamic
/// library loader and the process-local registry stores infallible closures.
/// These methods provide the first fallible boundary without changing that ABI
/// or forcing every existing registration site to migrate at once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkedAotCallbackError {
    /// The requested generated chunk does not exist in the linked artifact.
    ChunkIndex {
        stage: &'static str,
        index: usize,
        count: usize,
    },
    /// A generated chunk points outside the caller-provided global buffer.
    ChunkLayout {
        stage: &'static str,
        index: usize,
        offset: usize,
        len: usize,
        total: usize,
    },
    /// The callback was invoked with a buffer of the wrong size.
    OutputLength {
        stage: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The registered callback layout is internally invalid.
    InvalidLayout {
        stage: &'static str,
        message: String,
    },
    /// An input argument was not finite.
    NonFiniteInput { stage: &'static str, index: usize },
    /// The generated callback panicked before returning.
    Panicked { stage: &'static str },
    /// The generated callback returned a non-finite value.
    NonFiniteOutput { stage: &'static str, index: usize },
}

impl std::fmt::Display for LinkedAotCallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChunkIndex {
                stage,
                index,
                count,
            } => write!(
                f,
                "linked AOT {stage} chunk index {index} is out of range; chunk count is {count}"
            ),
            Self::ChunkLayout {
                stage,
                index,
                offset,
                len,
                total,
            } => write!(
                f,
                "linked AOT {stage} chunk {index} has range [{offset}, {}) outside output length {total}",
                offset.saturating_add(*len)
            ),
            Self::OutputLength {
                stage,
                expected,
                actual,
            } => write!(
                f,
                "linked AOT {stage} output has {actual} values; expected {expected}"
            ),
            Self::InvalidLayout { stage, message } => {
                write!(f, "linked AOT {stage} layout is invalid: {message}")
            }
            Self::Panicked { stage } => write!(f, "linked AOT {stage} callback panicked"),
            Self::NonFiniteInput { stage, index } => write!(
                f,
                "linked AOT {stage} input value at index {index} is not finite"
            ),
            Self::NonFiniteOutput { stage, index } => write!(
                f,
                "linked AOT {stage} output value at index {index} is not finite"
            ),
        }
    }
}

impl std::error::Error for LinkedAotCallbackError {}

/// Typed failure from the process-local linked AOT backend registries.
///
/// Registry access is normally infallible, but a poisoned mutex must not be
/// allowed to abort a user process on the fallible production path. The old
/// registration functions below remain compatibility wrappers and preserve
/// their historical panic-on-infrastructure-failure behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkedAotRegistryError {
    /// Another thread panicked while holding one of the linked registries.
    LockPoisoned { registry: &'static str },
}

impl std::fmt::Display for LinkedAotRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LockPoisoned { registry } => {
                write!(f, "linked AOT {registry} registry lock is poisoned")
            }
        }
    }
}

impl std::error::Error for LinkedAotRegistryError {}

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
            jacobian_layout: LinkedJacobianLayout::ExplicitValues,
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

    /// Marks this linked callback as a complete compact Banded callback.
    ///
    /// The dimensions are copied from the artifact manifest and are checked
    /// again by the BVP handoff before the callback is consumed.
    pub fn with_banded_compact_layout(
        mut self,
        rows: usize,
        cols: usize,
        kl: usize,
        ku: usize,
    ) -> Self {
        self.jacobian_layout = LinkedJacobianLayout::BandedCompact { rows, cols, kl, ku };
        self
    }

    /// Returns the callback output length implied by the published layout.
    pub fn jacobian_output_len(
        &self,
    ) -> Result<usize, crate::somelinalg::banded::error::BandedError> {
        match self.jacobian_layout {
            LinkedJacobianLayout::ExplicitValues => Ok(self.nnz),
            LinkedJacobianLayout::BandedCompact { rows, cols, kl, ku } => {
                crate::symbolic::codegen::codegen_runtime_api::BandedCompactJacobianStructure {
                    rows,
                    cols,
                    kl,
                    ku,
                }
                .storage_len()
            }
        }
    }

    /// Invokes the whole residual callback through a typed boundary.
    pub fn try_residual_eval(
        &self,
        args: &[f64],
        out: &mut [f64],
    ) -> Result<(), LinkedAotCallbackError> {
        invoke_linked_callback(
            "residual",
            self.residual_len,
            &*self.residual_eval,
            args,
            out,
        )
    }

    /// Invokes the whole sparse Jacobian-values callback through a typed
    /// boundary. The order remains the registered fixed-CSC order.
    pub fn try_jacobian_values_eval(
        &self,
        args: &[f64],
        out: &mut [f64],
    ) -> Result<(), LinkedAotCallbackError> {
        let expected =
            self.jacobian_output_len()
                .map_err(|error| LinkedAotCallbackError::InvalidLayout {
                    stage: "Jacobian",
                    message: error.to_string(),
                })?;
        invoke_linked_callback("Jacobian", expected, &*self.jacobian_values_eval, args, out)
    }

    /// Invokes one residual chunk through the same typed boundary as the
    /// whole callback. The caller supplies the chunk-local output slice.
    pub fn try_residual_chunk_eval(
        &self,
        chunk_index: usize,
        args: &[f64],
        out: &mut [f64],
    ) -> Result<(), LinkedAotCallbackError> {
        let chunk =
            self.residual_chunks
                .get(chunk_index)
                .ok_or(LinkedAotCallbackError::ChunkIndex {
                    stage: "residual",
                    index: chunk_index,
                    count: self.residual_chunks.len(),
                })?;
        invoke_linked_callback("residual chunk", chunk.output_len, &*chunk.eval, args, out)
    }

    /// Invokes one sparse Jacobian-value chunk through the typed boundary.
    /// The chunk-local order remains the fixed-structure order from the
    /// generated manifest.
    pub fn try_jacobian_chunk_eval(
        &self,
        chunk_index: usize,
        args: &[f64],
        out: &mut [f64],
    ) -> Result<(), LinkedAotCallbackError> {
        let chunk = self.jacobian_value_chunks.get(chunk_index).ok_or(
            LinkedAotCallbackError::ChunkIndex {
                stage: "Jacobian",
                index: chunk_index,
                count: self.jacobian_value_chunks.len(),
            },
        )?;
        invoke_linked_callback("Jacobian chunk", chunk.value_len, &*chunk.eval, args, out)
    }
}

pub(crate) fn invoke_linked_callback(
    stage: &'static str,
    expected_output_len: usize,
    callback: &dyn Fn(&[f64], &mut [f64]),
    args: &[f64],
    out: &mut [f64],
) -> Result<(), LinkedAotCallbackError> {
    if out.len() != expected_output_len {
        return Err(LinkedAotCallbackError::OutputLength {
            stage,
            expected: expected_output_len,
            actual: out.len(),
        });
    }
    if let Some(index) = args.iter().position(|value| !value.is_finite()) {
        return Err(LinkedAotCallbackError::NonFiniteInput { stage, index });
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        callback(args, out);
    }));
    if result.is_err() {
        return Err(LinkedAotCallbackError::Panicked { stage });
    }
    if let Some(index) = out.iter().position(|value| !value.is_finite()) {
        return Err(LinkedAotCallbackError::NonFiniteOutput { stage, index });
    }
    Ok(())
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

/// Fallibly registers one linked dense AOT backend in the current process.
pub fn try_register_linked_dense_backend(
    backend: LinkedDenseAotBackend,
) -> Result<(), LinkedAotRegistryError> {
    linked_dense_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned { registry: "dense" })?
        .insert(backend.problem_key.clone(), backend);
    Ok(())
}

/// Fallibly looks up a linked dense AOT backend by problem key.
pub fn try_resolve_linked_dense_backend(
    problem_key: &str,
) -> Result<Option<LinkedDenseAotBackend>, LinkedAotRegistryError> {
    Ok(linked_dense_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned { registry: "dense" })?
        .get(problem_key)
        .cloned())
}

/// Fallibly removes one linked dense AOT backend from the process registry.
pub fn try_unregister_linked_dense_backend(
    problem_key: &str,
) -> Result<Option<LinkedDenseAotBackend>, LinkedAotRegistryError> {
    Ok(linked_dense_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned { registry: "dense" })?
        .remove(problem_key))
}

/// Registers one linked dense AOT backend through the legacy panic contract.
pub fn register_linked_dense_backend(backend: LinkedDenseAotBackend) {
    try_register_linked_dense_backend(backend)
        .unwrap_or_else(|error| panic!("linked dense AOT registry access failed: {error}"));
}

/// Looks up a linked dense AOT backend through the legacy panic contract.
pub fn resolve_linked_dense_backend(problem_key: &str) -> Option<LinkedDenseAotBackend> {
    try_resolve_linked_dense_backend(problem_key)
        .unwrap_or_else(|error| panic!("linked dense AOT registry access failed: {error}"))
}

/// Removes one linked dense AOT backend through the legacy panic contract.
pub fn unregister_linked_dense_backend(problem_key: &str) -> Option<LinkedDenseAotBackend> {
    try_unregister_linked_dense_backend(problem_key)
        .unwrap_or_else(|error| panic!("linked dense AOT registry access failed: {error}"))
}

/// Fallibly registers one linked residual-only AOT backend.
pub fn try_register_linked_residual_backend(
    backend: LinkedResidualAotBackend,
) -> Result<(), LinkedAotRegistryError> {
    linked_residual_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned {
            registry: "residual",
        })?
        .insert(backend.problem_key.clone(), backend);
    Ok(())
}

/// Fallibly looks up a linked residual-only AOT backend.
pub fn try_resolve_linked_residual_backend(
    problem_key: &str,
) -> Result<Option<LinkedResidualAotBackend>, LinkedAotRegistryError> {
    Ok(linked_residual_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned {
            registry: "residual",
        })?
        .get(problem_key)
        .cloned())
}

/// Fallibly removes one linked residual-only AOT backend.
pub fn try_unregister_linked_residual_backend(
    problem_key: &str,
) -> Result<Option<LinkedResidualAotBackend>, LinkedAotRegistryError> {
    Ok(linked_residual_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned {
            registry: "residual",
        })?
        .remove(problem_key))
}

/// Registers one linked residual-only AOT backend through the legacy contract.
pub fn register_linked_residual_backend(backend: LinkedResidualAotBackend) {
    try_register_linked_residual_backend(backend)
        .unwrap_or_else(|error| panic!("linked residual AOT registry access failed: {error}"));
}

/// Looks up a linked residual-only AOT backend through the legacy contract.
pub fn resolve_linked_residual_backend(problem_key: &str) -> Option<LinkedResidualAotBackend> {
    try_resolve_linked_residual_backend(problem_key)
        .unwrap_or_else(|error| panic!("linked residual AOT registry access failed: {error}"))
}

/// Removes one linked residual-only AOT backend through the legacy contract.
pub fn unregister_linked_residual_backend(problem_key: &str) -> Option<LinkedResidualAotBackend> {
    try_unregister_linked_residual_backend(problem_key)
        .unwrap_or_else(|error| panic!("linked residual AOT registry access failed: {error}"))
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

/// Fallibly registers one linked sparse AOT backend.
pub fn try_register_linked_sparse_backend(
    backend: LinkedSparseAotBackend,
) -> Result<(), LinkedAotRegistryError> {
    linked_sparse_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned { registry: "sparse" })?
        .insert(backend.problem_key.clone(), backend);
    Ok(())
}

/// Fallibly looks up a linked sparse AOT backend.
pub fn try_resolve_linked_sparse_backend(
    problem_key: &str,
) -> Result<Option<LinkedSparseAotBackend>, LinkedAotRegistryError> {
    Ok(linked_sparse_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned { registry: "sparse" })?
        .get(problem_key)
        .cloned())
}

/// Fallibly removes one linked sparse AOT backend.
pub fn try_unregister_linked_sparse_backend(
    problem_key: &str,
) -> Result<Option<LinkedSparseAotBackend>, LinkedAotRegistryError> {
    Ok(linked_sparse_registry()
        .lock()
        .map_err(|_| LinkedAotRegistryError::LockPoisoned { registry: "sparse" })?
        .remove(problem_key))
}

/// Registers one linked sparse AOT backend through the legacy contract.
pub fn register_linked_sparse_backend(backend: LinkedSparseAotBackend) {
    try_register_linked_sparse_backend(backend)
        .unwrap_or_else(|error| panic!("linked sparse AOT registry access failed: {error}"));
}

/// Looks up a linked sparse AOT backend through the legacy contract.
pub fn resolve_linked_sparse_backend(problem_key: &str) -> Option<LinkedSparseAotBackend> {
    try_resolve_linked_sparse_backend(problem_key)
        .unwrap_or_else(|error| panic!("linked sparse AOT registry access failed: {error}"))
}

/// Removes one linked sparse AOT backend through the legacy contract.
pub fn unregister_linked_sparse_backend(problem_key: &str) -> Option<LinkedSparseAotBackend> {
    try_unregister_linked_sparse_backend(problem_key)
        .unwrap_or_else(|error| panic!("linked sparse AOT registry access failed: {error}"))
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

fn ensure_artifact_manifest_key_matches(artifact: &RegisteredAotArtifact) -> Result<(), String> {
    if artifact.manifest_key_matches() {
        return Ok(());
    }
    Err(format!(
        "AOT artifact manifest key mismatch before dynamic load: {}",
        artifact.lifecycle_contract_summary()
    ))
}

pub fn register_generated_sparse_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
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
    try_register_linked_sparse_backend(backend.clone()).map_err(|error| error.to_string())?;
    Ok(backend)
}

/// Registers a compiled banded cdylib backend.
///
/// Registers a generated Banded library while preserving its manifest-declared
/// value ABI. Historical artifacts use explicit band entries; AtomView-native
/// artifacts may publish complete compact LAPACK-style slots.
pub fn register_generated_banded_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedSparseAotBackend, String> {
    let compact_layout = validate_compact_banded_manifest(&artifact.manifest)?;
    let backend = register_generated_sparse_cdylib_backend(artifact)?;
    let Some((rows, cols, kl, ku)) = compact_layout else {
        return Ok(backend);
    };
    let backend = backend.with_banded_compact_layout(rows, cols, kl, ku);
    try_register_linked_sparse_backend(backend.clone()).map_err(|error| error.to_string())?;
    Ok(backend)
}

pub fn register_generated_dense_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedDenseAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
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
    try_register_linked_dense_backend(backend.clone()).map_err(|error| error.to_string())?;
    Ok(backend)
}

pub fn register_generated_residual_cdylib_backend(
    artifact: &RegisteredAotArtifact,
) -> Result<LinkedResidualAotBackend, String> {
    ensure_artifact_manifest_key_matches(artifact)?;
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
    try_register_linked_residual_backend(backend.clone()).map_err(|error| error.to_string())?;
    Ok(backend)
}
//==========================================================================================
// TESTS
//==========================================================================================
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
            matrix_backend: MatrixBackend::ValuesOnly,
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
            expression_signature: 42,
        }
    }

    fn mismatched_artifact() -> RegisteredAotArtifact {
        RegisteredAotArtifact {
            problem_key: "stale-or-wrong-problem-key".to_string(),
            crate_name: "generated_mismatched_fixture".to_string(),
            manifest: dummy_manifest(),
            crate_dir: PathBuf::from("generated_mismatched_fixture"),
            manifest_file: PathBuf::from("generated_mismatched_fixture/aot_manifest.rs"),
            artifact_dir: PathBuf::from("generated_mismatched_fixture/target"),
            expected_rlib: PathBuf::from("generated_mismatched_fixture/target/libfixture.rlib"),
            expected_cdylib: PathBuf::from("generated_mismatched_fixture/target/fixture.dll"),
            cargo_program: "cargo".to_string(),
            cargo_args: vec!["build".to_string()],
        }
    }

    #[test]
    fn generated_cdylib_registration_rejects_manifest_key_mismatch_before_loading() {
        let artifact = mismatched_artifact();
        let err = match register_generated_sparse_cdylib_backend(&artifact) {
            Ok(_) => panic!("mismatched artifact must not be dynamically loaded"),
            Err(err) => err,
        };

        assert!(
            err.contains("manifest key mismatch"),
            "unexpected error: {err}"
        );
        assert!(
            !err.contains("compiled sparse cdylib does not exist"),
            "manifest mismatch must be checked before filesystem/load errors: {err}"
        );
    }

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
    fn fallible_dense_registry_roundtrip_preserves_backend_identity() {
        let key = "fallible_linked_dense_backend_roundtrip";
        let backend = LinkedDenseAotBackend::new(
            key,
            1,
            (1, 1),
            Arc::new(|_, out| out[0] = 1.0),
            Arc::new(|_, out| out[0] = 2.0),
        );

        try_register_linked_dense_backend(backend.clone())
            .expect("fallible registry registration should succeed");
        let resolved = try_resolve_linked_dense_backend(key)
            .expect("fallible registry lookup should succeed")
            .expect("registered backend should be present");
        assert_eq!(resolved.problem_key, backend.problem_key);
        assert_eq!(resolved.residual_len, backend.residual_len);
        assert_eq!(resolved.shape, backend.shape);

        let removed = try_unregister_linked_dense_backend(key)
            .expect("fallible registry removal should succeed")
            .expect("registered backend should be removable");
        assert_eq!(removed.problem_key, key);
        assert!(
            try_resolve_linked_dense_backend(key)
                .expect("fallible registry lookup should succeed")
                .is_none()
        );
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

    #[test]
    fn linked_sparse_backend_preserves_compact_banded_layout_contract() {
        let backend = LinkedSparseAotBackend::new(
            "linked_compact_banded_contract",
            3,
            (3, 3),
            15,
            Arc::new(|_, out| out.fill(0.0)),
            Arc::new(|_, out| {
                out.iter_mut()
                    .enumerate()
                    .for_each(|(i, value)| *value = i as f64)
            }),
        )
        .with_banded_compact_layout(3, 3, 1, 1);

        assert_eq!(
            backend.jacobian_output_len().unwrap(),
            9,
            "compact Banded output uses all slots, not the symbolic nnz count"
        );
        assert!(matches!(
            backend.jacobian_layout,
            LinkedJacobianLayout::BandedCompact {
                rows: 3,
                cols: 3,
                kl: 1,
                ku: 1
            }
        ));
        let mut values = vec![0.0; 9];
        backend
            .try_jacobian_values_eval(&[1.0], &mut values)
            .expect("compact callback should accept complete slot storage");
        assert_eq!(values[8], 8.0);
        assert!(
            backend
                .try_jacobian_values_eval(&[1.0], &mut vec![0.0; 8])
                .is_err()
        );
    }

    #[test]
    fn compact_banded_manifest_validation_is_shared_and_preload() {
        let mut manifest = dummy_manifest();
        manifest.matrix_backend = MatrixBackend::Banded;
        manifest.io.jacobian_rows = 3;
        manifest.io.jacobian_cols = 3;
        manifest.io.jacobian_nnz = Some(9);
        manifest.io.jacobian_layout = Some(
            crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout::BandedCompact {
                kl: 1,
                ku: 1,
            },
        );

        assert_eq!(
            validate_compact_banded_manifest(&manifest).expect("valid compact manifest"),
            Some((3, 3, 1, 1))
        );

        manifest.io.jacobian_nnz = Some(8);
        let error = validate_compact_banded_manifest(&manifest)
            .expect_err("wrong compact storage length must fail before loading");
        assert!(error.contains("storage length mismatch"));

        manifest.io.jacobian_nnz = Some(9);
        manifest.matrix_backend = MatrixBackend::SparseCol;
        let error = validate_compact_banded_manifest(&manifest)
            .expect_err("compact marker on a sparse backend must fail");
        assert!(error.contains("matrix backend Banded"));
    }

    #[test]
    fn linked_sparse_callback_boundary_reports_panics_and_nonfinite_values() {
        let panicking = LinkedSparseAotBackend::new(
            "linked_sparse_panicking_callback",
            1,
            (1, 1),
            1,
            Arc::new(|_, _| panic!("generated callback failure")),
            Arc::new(|_, out| out[0] = 1.0),
        );
        let mut residual = vec![0.0; 1];
        assert!(matches!(
            panicking.try_residual_eval(&[1.0], &mut residual),
            Err(LinkedAotCallbackError::Panicked { stage: "residual" })
        ));

        let nonfinite = LinkedSparseAotBackend::new(
            "linked_sparse_nonfinite_callback",
            1,
            (1, 1),
            1,
            Arc::new(|_, out| out[0] = f64::NAN),
            Arc::new(|_, out| out[0] = 1.0),
        );
        assert!(matches!(
            nonfinite.try_residual_eval(&[1.0], &mut residual),
            Err(LinkedAotCallbackError::NonFiniteOutput {
                stage: "residual",
                index: 0
            })
        ));

        let wrong_shape = LinkedSparseAotBackend::new(
            "linked_sparse_wrong_shape",
            2,
            (2, 2),
            2,
            Arc::new(|_, out| out.fill(0.0)),
            Arc::new(|_, out| out.fill(0.0)),
        );
        let mut short_residual = vec![0.0; 1];
        assert_eq!(
            wrong_shape.try_residual_eval(&[1.0], &mut short_residual),
            Err(LinkedAotCallbackError::OutputLength {
                stage: "residual",
                expected: 2,
                actual: 1,
            })
        );

        let chunked = LinkedSparseAotBackend::new(
            "linked_sparse_chunk_boundary",
            2,
            (2, 2),
            2,
            Arc::new(|_, out| out.fill(0.0)),
            Arc::new(|_, out| out.fill(0.0)),
        )
        .with_chunked_evaluators(
            vec![LinkedResidualChunk::new(
                0,
                1,
                Arc::new(|args, out| out[0] = args[0] + 1.0),
            )],
            vec![LinkedSparseJacobianChunk::new(
                0,
                1,
                Arc::new(|args, out| out[0] = args[0] + 2.0),
            )],
        );
        let mut chunk_output = vec![0.0; 1];
        chunked
            .try_residual_chunk_eval(0, &[3.0], &mut chunk_output)
            .expect("valid residual chunk should execute");
        assert_eq!(chunk_output, vec![4.0]);
        assert!(matches!(
            chunked.try_jacobian_chunk_eval(4, &[3.0], &mut chunk_output),
            Err(LinkedAotCallbackError::ChunkIndex {
                stage: "Jacobian",
                index: 4,
                count: 1
            })
        ));
    }
}
