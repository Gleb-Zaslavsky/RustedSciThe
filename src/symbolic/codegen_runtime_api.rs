//! Solver-facing runtime API for AOT-generated residuals and Jacobians.
//!
//! This module sits above [`crate::symbolic::codegen_tasks`] and below the
//! eventual Newton/BVP solver integration layer.
//!
//! The goal is to keep two levels of API separate:
//!
//! - low-level generated/AOT-friendly evaluation, where code-generated chunks
//!   write into caller-provided output buffers;
//! - high-level solver-friendly wrappers, where Newton-style code sees the
//!   familiar operations "evaluate residual" and "evaluate Jacobian".
//!
//! Chunking and block-local CSE are intentionally hidden behind runtime plans.
//! A solver should not need to know whether a sparse Jacobian was produced by:
//! - one generated block,
//! - several generated chunks,
//! - or future chunk-local CSE pipelines.
//!
//! Current scope:
//! - residual output chunk planning,
//! - dense Jacobian row-chunk planning,
//! - sparse Jacobian values chunk planning,
//! - argument flattening helpers for `(params, variables)`,
//! - sparse matrix assembly metadata for `faer::sparse::SparseColMat`.
//!
//! Planned next steps:
//! - band-aware sparse chunking (`ByBand`) for strongly banded BVP Jacobians,
//! - compile-time parallel lowering / codegen for large chunk sets,
//! - runtime parallel execution of generated chunk functions into disjoint
//!   output slices.
//!
//! This module does **not** yet execute generated code or own any AOT bundle.
//! It defines the runtime contract that future generated modules should match.

use crate::symbolic::codegen_tasks::{
    CodegenTask, CodegenTaskPlan, IvpJacobianTask, IvpResidualTask, JacobianTask, ResidualTask,
    SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
};
use crate::symbolic::symbolic_engine::Expr;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::DMatrix;
use std::borrow::Cow;

/// Numeric arguments grouped the same way as symbolic task arguments.
///
/// Parameters remain optional because many problems have no external model
/// constants, while others require them on every residual/Jacobian call.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeArguments<'a> {
    /// Extra numeric parameters that affect evaluation but are not
    /// differentiation variables.
    pub params: Option<&'a [f64]>,
    /// State / unknown vector values.
    pub variables: &'a [f64],
}

impl<'a> RuntimeArguments<'a> {
    /// Creates a grouped runtime argument view.
    pub fn new(variables: &'a [f64], params: Option<&'a [f64]>) -> Self {
        Self { params, variables }
    }

    /// Returns the total flat argument count in the default AOT order:
    /// parameters first, then variables.
    pub fn total_len(&self) -> usize {
        self.variables.len() + self.params.map_or(0, |params| params.len())
    }

    /// Flattens grouped runtime arguments into the default AOT order:
    /// parameters first, then variables.
    pub fn flatten(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.total_len());
        if let Some(params) = self.params {
            flat.extend_from_slice(params);
        }
        flat.extend_from_slice(self.variables);
        flat
    }
}

/// Chunking strategy for residual vectors.
///
/// Residuals can be evaluated as one whole generated block or as several
/// output slices. The choice should remain workload-dependent instead of being
/// hard-coded too early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualChunkingStrategy {
    /// Keep the full residual vector as one generated block.
    Whole,
    /// Split outputs into a target number of coarse chunks. This is useful
    /// when runtime execution should use a few larger jobs instead of many
    /// small ones.
    ByTargetChunkCount { target_chunks: usize },
    /// Split residual outputs by a maximum number of equations per chunk.
    ByOutputCount { max_outputs_per_chunk: usize },
}

/// One residual output chunk in solver order.
#[derive(Debug, Clone)]
pub struct ResidualChunkPlan<'a> {
    /// Stable generated function name for the chunk.
    pub fn_name: String,
    /// Global offset of the first residual entry written by this chunk.
    pub output_offset: usize,
    /// Residual expressions evaluated by this chunk.
    pub residuals: &'a [crate::symbolic::symbolic_engine::Expr],
    /// Flattened low-level codegen plan for the chunk.
    pub plan: CodegenTaskPlan<'a>,
}

/// Solver-facing residual runtime plan.
///
/// The solver sees one logical residual evaluator, while generated code can
/// later target one or many chunk functions behind this plan.
#[derive(Debug, Clone)]
pub struct ResidualRuntimePlan<'a> {
    pub fn_name: &'a str,
    pub output_len: usize,
    pub input_names: Vec<&'a str>,
    pub chunks: Vec<ResidualChunkPlan<'a>>,
}

impl<'a> ResidualRuntimePlan<'a> {
    /// Returns the total residual length.
    pub fn len(&self) -> usize {
        self.output_len
    }

    /// Returns `true` if the residual has no outputs.
    pub fn is_empty(&self) -> bool {
        self.output_len == 0
    }
}

impl<'a> ResidualTask<'a> {
    /// Builds a solver-facing residual runtime plan that hides chunking behind
    /// a single logical residual evaluator.
    pub fn runtime_plan(&self, strategy: ResidualChunkingStrategy) -> ResidualRuntimePlan<'a> {
        let chunk_size = match strategy {
            ResidualChunkingStrategy::Whole => self.residuals.len().max(1),
            ResidualChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                self.residuals.len().max(1).div_ceil(target_chunks).max(1)
            }
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk,
            } => {
                assert!(
                    max_outputs_per_chunk > 0,
                    "max_outputs_per_chunk must be positive"
                );
                max_outputs_per_chunk
            }
        };

        let chunks = self
            .residuals
            .chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, residuals)| {
                let output_offset = chunk_index * chunk_size;
                let fn_name = if chunk_index == 0 && residuals.len() == self.residuals.len() {
                    self.fn_name.to_string()
                } else {
                    format!("{}_chunk_{}", self.fn_name, chunk_index)
                };
                let chunk_task = ResidualTask {
                    fn_name: self.fn_name,
                    residuals,
                    variables: self.variables,
                    params: self.params,
                };
                let mut plan = chunk_task.plan();
                plan.fn_name = Cow::Owned(fn_name.clone());

                ResidualChunkPlan {
                    fn_name,
                    output_offset,
                    residuals,
                    plan,
                }
            })
            .collect();

        ResidualRuntimePlan {
            fn_name: self.fn_name,
            output_len: self.residuals.len(),
            input_names: self.flattened_argument_names(),
            chunks,
        }
    }
}

impl<'a> IvpResidualTask<'a> {
    /// Builds a solver-facing IVP residual runtime plan.
    ///
    /// Internally this reuses the same chunk model as algebraic residuals, but
    /// the flattened input order includes the distinguished time argument
    /// first, followed by optional parameters and state variables.
    pub fn runtime_plan(&self, strategy: ResidualChunkingStrategy) -> ResidualRuntimePlan<'a> {
        let chunk_size = match strategy {
            ResidualChunkingStrategy::Whole => self.residuals.len().max(1),
            ResidualChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                self.residuals.len().max(1).div_ceil(target_chunks).max(1)
            }
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk,
            } => {
                assert!(
                    max_outputs_per_chunk > 0,
                    "max_outputs_per_chunk must be positive"
                );
                max_outputs_per_chunk
            }
        };

        let chunks = self
            .residuals
            .chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, residuals)| {
                let output_offset = chunk_index * chunk_size;
                let fn_name = if chunk_index == 0 && residuals.len() == self.residuals.len() {
                    self.fn_name.to_string()
                } else {
                    format!("{}_chunk_{}", self.fn_name, chunk_index)
                };
                let chunk_task = IvpResidualTask {
                    fn_name: self.fn_name,
                    time_arg: self.time_arg,
                    residuals,
                    variables: self.variables,
                    params: self.params,
                };
                let mut plan = chunk_task.plan();
                plan.fn_name = Cow::Owned(fn_name.clone());

                ResidualChunkPlan {
                    fn_name,
                    output_offset,
                    residuals,
                    plan,
                }
            })
            .collect();

        ResidualRuntimePlan {
            fn_name: self.fn_name,
            output_len: self.residuals.len(),
            input_names: self.flattened_argument_names_with_time(),
            chunks,
        }
    }
}

/// One sparse Jacobian values chunk in solver order.
#[derive(Debug, Clone)]
pub struct SparseJacobianValuesChunkPlan<'a> {
    /// Stable generated function name for the chunk.
    pub fn_name: String,
    /// Global offset of the first value written by this chunk.
    pub value_offset: usize,
    /// Sparse entries covered by this chunk in explicit solver order.
    pub entries: Vec<SparseExprEntry<'a>>,
    /// Flattened low-level codegen plan for the chunk.
    pub plan: CodegenTaskPlan<'a>,
}

/// Chunking strategy for dense Jacobian matrices.
///
/// Dense Jacobians are chunked by contiguous row blocks and written in global
/// row-major order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseJacobianChunkingStrategy {
    /// Keep the full dense Jacobian as one generated block.
    Whole,
    /// Split the matrix into a target number of coarse row chunks.
    ByTargetChunkCount { target_chunks: usize },
    /// Split the matrix by a maximum number of rows per chunk.
    ByRowCount { rows_per_chunk: usize },
}

/// One dense Jacobian row chunk in solver row-major order.
#[derive(Debug, Clone)]
pub struct DenseJacobianChunkPlan<'a> {
    /// Stable generated function name for the chunk.
    pub fn_name: String,
    /// Global row offset of the first Jacobian row written by this chunk.
    pub row_offset: usize,
    /// Global row-major offset of the first dense Jacobian value written by
    /// this chunk.
    pub value_offset: usize,
    /// Dense Jacobian rows evaluated by this chunk.
    pub jacobian_rows: &'a [Vec<Expr>],
    /// Flattened low-level codegen plan for the chunk.
    pub plan: CodegenTaskPlan<'a>,
}

impl<'a> DenseJacobianChunkPlan<'a> {
    /// Returns the number of rows covered by this chunk.
    pub fn row_len(&self) -> usize {
        self.jacobian_rows.len()
    }

    /// Returns the global half-open value range written by this chunk.
    pub fn value_range(&self) -> std::ops::Range<usize> {
        let cols = self.jacobian_rows.first().map_or(0usize, |row| row.len());
        self.value_offset..(self.value_offset + self.row_len() * cols)
    }
}

/// Solver-facing dense Jacobian runtime plan.
///
/// Dense Jacobians are exposed as one logical matrix evaluator while codegen
/// remains free to use one or many row chunks internally.
#[derive(Debug, Clone)]
pub struct DenseJacobianRuntimePlan<'a> {
    pub fn_name: &'a str,
    pub rows: usize,
    pub cols: usize,
    pub input_names: Vec<&'a str>,
    pub chunks: Vec<DenseJacobianChunkPlan<'a>>,
}

impl<'a> DenseJacobianRuntimePlan<'a> {
    /// Returns the total number of matrix entries in row-major order.
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Returns `true` if the matrix has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Assembles a dense matrix from a row-major values slice.
    pub fn assemble_dense_matrix(&self, values: &[f64]) -> DMatrix<f64> {
        assert_eq!(
            values.len(),
            self.len(),
            "expected {} dense Jacobian values, got {}",
            self.len(),
            values.len()
        );

        DMatrix::from_row_slice(self.rows, self.cols, values)
    }
}

impl<'a> SparseJacobianValuesChunkPlan<'a> {
    /// Returns the global half-open value range written by this chunk.
    pub fn value_range(&self) -> std::ops::Range<usize> {
        self.value_offset..(self.value_offset + self.entries.len())
    }
}

/// Sparse Jacobian structure metadata independent of values.
///
/// This is the solver-facing representation of sparsity:
/// generated chunks write only `values`, while the solver can still obtain a
/// regular `SparseColMat` from the same explicit entry ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseJacobianStructure {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
}

impl SparseJacobianStructure {
    /// Returns the number of explicitly stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.row_indices.len()
    }

    /// Builds a faer sparse matrix from a value slice that follows the same
    /// explicit entry order as this structure.
    pub fn assemble_sparse_col_mat(&self, values: &[f64]) -> SparseColMat<usize, f64> {
        assert_eq!(
            values.len(),
            self.nnz(),
            "expected {} sparse values, got {}",
            self.nnz(),
            values.len()
        );

        let triplets: Vec<Triplet<usize, usize, f64>> = self
            .row_indices
            .iter()
            .copied()
            .zip(self.col_indices.iter().copied())
            .zip(values.iter().copied())
            .map(|((row, col), value)| Triplet::new(row, col, value))
            .collect();

        SparseColMat::try_new_from_triplets(self.rows, self.cols, &triplets)
            .expect("sparse Jacobian structure should assemble into SparseColMat")
    }
}

/// Solver-facing sparse Jacobian runtime plan.
///
/// The solver should see this as one logical Jacobian provider:
/// - low-level generated chunks write values into disjoint slices,
/// - the wrapper owns the global sparse structure and can assemble the final
///   matrix on demand.
#[derive(Debug, Clone)]
pub struct SparseJacobianRuntimePlan<'a> {
    pub fn_name: &'a str,
    pub input_names: Vec<&'a str>,
    pub structure: SparseJacobianStructure,
    pub chunks: Vec<SparseJacobianValuesChunkPlan<'a>>,
}

impl<'a> SparseJacobianRuntimePlan<'a> {
    /// Returns the number of explicitly stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.structure.nnz()
    }

    /// Assembles a sparse matrix from values written in the global explicit
    /// entry order expected by the chunk plans.
    pub fn assemble_sparse_col_mat(&self, values: &[f64]) -> SparseColMat<usize, f64> {
        self.structure.assemble_sparse_col_mat(values)
    }
}

impl<'a> SparseJacobianTask<'a> {
    /// Builds a solver-facing sparse Jacobian runtime plan.
    ///
    /// Internally this may use one or many sparse values chunks, but the
    /// resulting plan still exposes one logical Jacobian evaluator contract:
    /// arguments in, sparse values out, optional sparse matrix assembly.
    pub fn runtime_plan(&self, strategy: SparseChunkingStrategy) -> SparseJacobianRuntimePlan<'a> {
        let chunks: Vec<SparseJacobianValuesChunkPlan<'a>> = self
            .chunk_with_strategy(strategy)
            .into_iter()
            .map(|chunk| SparseJacobianValuesChunkPlan {
                fn_name: chunk.chunk_fn_name(),
                value_offset: chunk.entry_offset,
                entries: chunk.entries.clone(),
                plan: chunk.plan(),
            })
            .collect();

        SparseJacobianRuntimePlan {
            fn_name: self.fn_name,
            input_names: self.flattened_argument_names(),
            structure: SparseJacobianStructure {
                rows: self.shape.0,
                cols: self.shape.1,
                row_indices: self.entries.iter().map(|entry| entry.row).collect(),
                col_indices: self.entries.iter().map(|entry| entry.col).collect(),
            },
            chunks,
        }
    }
}

impl<'a> JacobianTask<'a> {
    /// Builds a solver-facing dense Jacobian runtime plan.
    ///
    /// Internally this may use one or many row chunks, but the resulting plan
    /// still exposes one logical dense Jacobian evaluator contract:
    /// arguments in, row-major values out, optional dense matrix assembly.
    pub fn runtime_plan(
        &self,
        strategy: DenseJacobianChunkingStrategy,
    ) -> DenseJacobianRuntimePlan<'a> {
        let rows = self.jacobian.len();
        let cols = self.jacobian.first().map_or(0, |row| row.len());
        let rows_per_chunk = match strategy {
            DenseJacobianChunkingStrategy::Whole => rows.max(1),
            DenseJacobianChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                rows.max(1).div_ceil(target_chunks).max(1)
            }
            DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk } => {
                assert!(rows_per_chunk > 0, "rows_per_chunk must be positive");
                rows_per_chunk
            }
        };

        let chunks = self
            .jacobian
            .chunks(rows_per_chunk)
            .enumerate()
            .map(|(chunk_index, jacobian_rows)| {
                let row_offset = chunk_index * rows_per_chunk;
                let value_offset = row_offset * cols;
                let fn_name = if chunk_index == 0 && jacobian_rows.len() == rows {
                    self.fn_name.to_string()
                } else {
                    format!("{}_chunk_{}", self.fn_name, chunk_index)
                };
                let chunk_task = JacobianTask {
                    fn_name: self.fn_name,
                    jacobian: jacobian_rows,
                    variables: self.variables,
                    params: self.params,
                };
                let mut plan = chunk_task.plan();
                plan.fn_name = Cow::Owned(fn_name.clone());

                DenseJacobianChunkPlan {
                    fn_name,
                    row_offset,
                    value_offset,
                    jacobian_rows,
                    plan,
                }
            })
            .collect();

        DenseJacobianRuntimePlan {
            fn_name: self.fn_name,
            rows,
            cols,
            input_names: self.flattened_argument_names(),
            chunks,
        }
    }
}

impl<'a> IvpJacobianTask<'a> {
    /// Builds a solver-facing dense IVP Jacobian runtime plan.
    ///
    /// The chunk representation is the same as for dense algebraic Jacobians,
    /// but the flattened input order includes time as the leading argument.
    pub fn runtime_plan(
        &self,
        strategy: DenseJacobianChunkingStrategy,
    ) -> DenseJacobianRuntimePlan<'a> {
        let rows = self.jacobian.len();
        let cols = self.jacobian.first().map_or(0, |row| row.len());
        let rows_per_chunk = match strategy {
            DenseJacobianChunkingStrategy::Whole => rows.max(1),
            DenseJacobianChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                rows.max(1).div_ceil(target_chunks).max(1)
            }
            DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk } => {
                assert!(rows_per_chunk > 0, "rows_per_chunk must be positive");
                rows_per_chunk
            }
        };

        let chunks = self
            .jacobian
            .chunks(rows_per_chunk)
            .enumerate()
            .map(|(chunk_index, jacobian_rows)| {
                let row_offset = chunk_index * rows_per_chunk;
                let value_offset = row_offset * cols;
                let fn_name = if chunk_index == 0 && jacobian_rows.len() == rows {
                    self.fn_name.to_string()
                } else {
                    format!("{}_chunk_{}", self.fn_name, chunk_index)
                };
                let chunk_task = IvpJacobianTask {
                    fn_name: self.fn_name,
                    time_arg: self.time_arg,
                    jacobian: jacobian_rows,
                    variables: self.variables,
                    params: self.params,
                };
                let mut plan = chunk_task.plan();
                plan.fn_name = Cow::Owned(fn_name.clone());

                DenseJacobianChunkPlan {
                    fn_name,
                    row_offset,
                    value_offset,
                    jacobian_rows,
                    plan,
                }
            })
            .collect();

        DenseJacobianRuntimePlan {
            fn_name: self.fn_name,
            rows,
            cols,
            input_names: self.flattened_argument_names_with_time(),
            chunks,
        }
    }
}

/// Extracts explicit non-zero sparse entries from a legacy dense symbolic
/// Jacobian matrix.
///
/// This is the compatibility bridge from the current dense internal storage
/// (`Vec<Vec<Expr>>`) to the sparse-first AOT/codegen path. Zero entries are
/// skipped using the symbolic engine's zero predicate.
pub fn extract_sparse_entries_from_dense_jacobian<'a>(
    jacobian: &'a [Vec<Expr>],
) -> Vec<SparseExprEntry<'a>> {
    let mut entries = Vec::new();

    for (row, dense_row) in jacobian.iter().enumerate() {
        for (col, expr) in dense_row.iter().enumerate() {
            if !expr.is_zero() {
                entries.push(SparseExprEntry { row, col, expr });
            }
        }
    }

    entries
}

/// Chooses a practical row-based sparse chunking strategy from the currently
/// available parallelism.
///
/// The strategy intentionally creates more than one chunk per worker so a
/// scheduler such as rayon can still balance uneven row workloads.
pub fn recommended_row_chunking_for_parallelism(
    total_rows: usize,
    chunks_per_worker: usize,
) -> SparseChunkingStrategy {
    assert!(total_rows > 0, "total_rows must be positive");
    assert!(chunks_per_worker > 0, "chunks_per_worker must be positive");

    let workers = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let target_chunks = (workers * chunks_per_worker).max(1);
    let rows_per_chunk = total_rows.div_ceil(target_chunks).max(1);

    SparseChunkingStrategy::ByRowCount { rows_per_chunk }
}

/// Chooses a practical residual chunking strategy from the currently available
/// parallelism.
///
/// This is a heuristic for coarse-grained residual execution:
/// - create more than one chunk per worker,
/// - but avoid many tiny chunks on large systems,
/// - keep the final decision adjustable by benchmarks.
pub fn recommended_residual_chunking_for_parallelism(
    total_outputs: usize,
    chunks_per_worker: usize,
) -> ResidualChunkingStrategy {
    assert!(total_outputs > 0, "total_outputs must be positive");
    assert!(chunks_per_worker > 0, "chunks_per_worker must be positive");

    let workers = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let target_chunks = (workers * chunks_per_worker).max(1);
    let max_outputs_per_chunk = total_outputs.div_ceil(target_chunks).max(1);

    ResidualChunkingStrategy::ByOutputCount {
        max_outputs_per_chunk,
    }
}

/// Chooses a practical dense-Jacobian chunking strategy from the currently
/// available parallelism.
///
/// Dense Jacobians are chunked by contiguous row blocks, mirroring the sparse
/// row-based heuristic while preserving row-major output layout.
pub fn recommended_dense_jacobian_chunking_for_parallelism(
    total_rows: usize,
    chunks_per_worker: usize,
) -> DenseJacobianChunkingStrategy {
    assert!(total_rows > 0, "total_rows must be positive");
    assert!(chunks_per_worker > 0, "chunks_per_worker must be positive");

    let workers = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let target_chunks = (workers * chunks_per_worker).max(1);
    let rows_per_chunk = total_rows.div_ceil(target_chunks).max(1);

    DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn runtime_arguments_flatten_params_before_variables() {
        let args = RuntimeArguments::new(&[10.0, 20.0], Some(&[1.5, 2.5]));

        assert_eq!(args.total_len(), 4);
        assert_eq!(args.flatten(), vec![1.5, 2.5, 10.0, 20.0]);
    }

    #[test]
    fn residual_runtime_plan_hides_chunking_behind_output_offsets() {
        let residuals = vec![
            Expr::Var("y0".to_string()),
            Expr::Var("y1".to_string()),
            Expr::Var("y2".to_string()),
        ];
        let task = ResidualTask {
            fn_name: "eval_residual",
            residuals: &residuals,
            variables: &["y0", "y1", "y2"],
            params: Some(&["alpha"]),
        };

        let plan = task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 2,
        });

        assert_eq!(plan.fn_name, "eval_residual");
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.input_names, vec!["alpha", "y0", "y1", "y2"]);
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].output_offset, 0);
        assert_eq!(plan.chunks[0].residuals.len(), 2);
        assert_eq!(plan.chunks[1].output_offset, 2);
        assert_eq!(plan.chunks[1].residuals.len(), 1);
    }

    #[test]
    fn sparse_runtime_plan_assembles_sparse_col_mat_from_global_value_order() {
        let e0 = Expr::Var("y0".to_string());
        let e1 = Expr::Var("y1".to_string());
        let e2 = Expr::Var("y2".to_string());
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 1,
                col: 2,
                expr: &e1,
            },
            SparseExprEntry {
                row: 2,
                col: 1,
                expr: &e2,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (3, 3),
            entries: &entries,
            variables: &["y0", "y1", "y2"],
            params: Some(&["alpha"]),
        };

        let runtime_plan = task.runtime_plan(SparseChunkingStrategy::ByNonZeroCount {
            max_entries_per_chunk: 2,
        });

        assert_eq!(runtime_plan.fn_name, "eval_sparse_values");
        assert_eq!(runtime_plan.nnz(), 3);
        assert_eq!(runtime_plan.input_names, vec!["alpha", "y0", "y1", "y2"]);
        assert_eq!(runtime_plan.chunks.len(), 2);
        assert_eq!(runtime_plan.chunks[0].value_range(), 0..2);
        assert_eq!(runtime_plan.chunks[1].value_range(), 2..3);

        let matrix = runtime_plan.assemble_sparse_col_mat(&[10.0, 20.0, 30.0]);

        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 3);
        assert_eq!(matrix.compute_nnz(), 3);
    }

    #[test]
    fn dense_runtime_plan_assembles_dense_matrix_from_row_major_values() {
        let jacobian = vec![
            vec![Expr::Var("y0".to_string()), Expr::Const(1.0)],
            vec![Expr::Const(2.0), Expr::Var("y1".to_string())],
            vec![Expr::Const(3.0), Expr::Const(4.0)],
        ];
        let task = JacobianTask {
            fn_name: "eval_dense_jacobian",
            jacobian: &jacobian,
            variables: &["y0", "y1"],
            params: Some(&["alpha"]),
        };

        let plan =
            task.runtime_plan(DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 });

        assert_eq!(plan.fn_name, "eval_dense_jacobian");
        assert_eq!(plan.rows, 3);
        assert_eq!(plan.cols, 2);
        assert_eq!(plan.input_names, vec!["alpha", "y0", "y1"]);
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].row_offset, 0);
        assert_eq!(plan.chunks[0].value_range(), 0..4);
        assert_eq!(plan.chunks[1].row_offset, 2);
        assert_eq!(plan.chunks[1].value_range(), 4..6);

        let matrix = plan.assemble_dense_matrix(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);

        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], 10.0);
        assert_eq!(matrix[(2, 1)], 31.0);
    }

    #[test]
    fn extract_sparse_entries_from_dense_jacobian_skips_symbolic_zeros() {
        let jacobian = vec![
            vec![Expr::Const(0.0), Expr::Var("y0".to_string())],
            vec![Expr::Const(2.0), Expr::Const(0.0)],
        ];

        let entries = extract_sparse_entries_from_dense_jacobian(&jacobian);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].row, 0);
        assert_eq!(entries[0].col, 1);
        assert_eq!(entries[1].row, 1);
        assert_eq!(entries[1].col, 0);
    }

    #[test]
    fn recommended_row_chunking_for_parallelism_returns_row_strategy() {
        let strategy = recommended_row_chunking_for_parallelism(128, 4);

        match strategy {
            SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
                assert!(rows_per_chunk >= 1);
                assert!(rows_per_chunk <= 128);
            }
            _ => panic!("expected row-count chunking strategy"),
        }
    }

    #[test]
    fn recommended_residual_chunking_for_parallelism_returns_output_strategy() {
        let strategy = recommended_residual_chunking_for_parallelism(256, 4);

        match strategy {
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk,
            } => {
                assert!(max_outputs_per_chunk >= 1);
                assert!(max_outputs_per_chunk <= 256);
            }
            _ => panic!("expected output-count chunking strategy"),
        }
    }

    #[test]
    fn recommended_dense_jacobian_chunking_for_parallelism_returns_row_strategy() {
        let strategy = recommended_dense_jacobian_chunking_for_parallelism(256, 4);

        match strategy {
            DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk } => {
                assert!(rows_per_chunk >= 1);
                assert!(rows_per_chunk <= 256);
            }
            _ => panic!("expected row-count chunking strategy"),
        }
    }

    #[test]
    fn residual_runtime_plan_supports_target_chunk_count() {
        let residuals = vec![
            Expr::Var("y0".to_string()),
            Expr::Var("y1".to_string()),
            Expr::Var("y2".to_string()),
            Expr::Var("y3".to_string()),
        ];
        let task = ResidualTask {
            fn_name: "eval_residual",
            residuals: &residuals,
            variables: &["y0", "y1", "y2", "y3"],
            params: None,
        };

        let plan =
            task.runtime_plan(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 });

        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].residuals.len(), 2);
        assert_eq!(plan.chunks[1].residuals.len(), 2);
    }
}
