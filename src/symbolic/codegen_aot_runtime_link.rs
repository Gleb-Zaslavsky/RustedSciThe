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

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Shared residual evaluator signature for a linked sparse AOT backend.
pub type LinkedResidualEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared sparse Jacobian-values evaluator signature for a linked sparse AOT backend.
pub type LinkedSparseJacobianEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared residual chunk evaluator signature for linked chunked AOT execution.
pub type LinkedResidualChunkEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

/// Shared sparse Jacobian chunk evaluator signature for linked chunked AOT execution.
pub type LinkedSparseJacobianChunkEval = dyn Fn(&[f64], &mut [f64]) + Send + Sync;

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

#[cfg(test)]
mod tests {
    use super::*;

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
