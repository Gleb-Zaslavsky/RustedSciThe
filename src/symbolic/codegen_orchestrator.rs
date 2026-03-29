//! Orchestration layer over generated chunk functions.
//!
//! This module is the bridge between:
//! - low-level generated chunk functions of shape `fn(&[f64], &mut [f64])`,
//! - and solver-facing residual / Jacobian evaluators.
//!
//! Current scope:
//! - sequential orchestration,
//! - parallel orchestration with deterministic chunk joining,
//! - no bundle loading yet.
//!
//! Chunk size matters a lot for runtime performance.
//!
//! Very small chunks are easy to parallelize, but they can still be slow in
//! practice because the runtime spends too much time scheduling work instead of
//! evaluating math. For large residual/Jacobian problems it is usually better
//! to use fewer, heavier chunks:
//! - each chunk does more useful arithmetic,
//! - there are fewer rayon tasks to spawn,
//! - chunk-local CSE has more expressions to reuse,
//! - memory locality is usually better.
//!
//! In other words, parallel work should be coarse enough to amortize runtime
//! overhead, but not so coarse that only one or two workers do all the work.
//! A practical rule of thumb is:
//! - start with several chunks per worker thread,
//! - then increase chunk size for large systems until scheduling overhead stops
//!   dominating,
//! - keep sequential execution available for small systems where parallelism is
//!   more expensive than the math itself.
//!
//! The goal is to replace ad-hoc manual harness code with production-shaped
//! execution objects while keeping the implementation simple and race-free.
//!
//! Planned next steps:
//! - lower-allocation parallel execution into disjoint output slices,
//! - integration with generated-module loading / bundle lifecycle,
//! - orchestration helpers specialized for residual and sparse Jacobian tasks.

use crate::symbolic::codegen_runtime_api::{
    DenseJacobianRuntimePlan, ResidualRuntimePlan, SparseJacobianRuntimePlan,
};
use crate::symbolic::codegen_tasks::SparseExprEntry;
use faer::sparse::SparseColMat;
use nalgebra::DMatrix;
use rayon::{join, scope};
use std::ops::Range;

const DEFAULT_MIN_RESIDUAL_OUTPUTS_FOR_PARALLEL: usize = 128;
const DEFAULT_MIN_SPARSE_VALUES_FOR_PARALLEL: usize = 256;

fn contiguous_job_ranges(len: usize, max_jobs: usize) -> Vec<Range<usize>> {
    if len == 0 {
        return Vec::new();
    }

    let target_jobs = len.min(max_jobs.max(1));
    let items_per_job = len.div_ceil(target_jobs);
    let mut ranges = Vec::with_capacity(target_jobs);
    let mut start = 0usize;

    while start < len {
        let end = (start + items_per_job).min(len);
        ranges.push(start..end);
        start = end;
    }

    ranges
}

fn borrowed_sparse_entries(
    owned_entries: &[(usize, usize, crate::symbolic::symbolic_engine::Expr)],
) -> Vec<SparseExprEntry<'_>> {
    owned_entries
        .iter()
        .map(|(row, col, expr)| SparseExprEntry {
            row: *row,
            col: *col,
            expr,
        })
        .collect()
}

fn contiguous_weighted_job_ranges(weights: &[usize], max_jobs: usize) -> Vec<Range<usize>> {
    if weights.is_empty() {
        return Vec::new();
    }

    let len = weights.len();
    let target_jobs = len.min(max_jobs.max(1));
    if target_jobs == 1 {
        return vec![0..len];
    }

    let normalized_weights: Vec<usize> = weights.iter().map(|weight| (*weight).max(1)).collect();
    let total_weight: usize = normalized_weights.iter().sum();
    if total_weight == 0 {
        return contiguous_job_ranges(len, target_jobs);
    }

    let mut prefix = Vec::with_capacity(len + 1);
    prefix.push(0usize);
    for weight in &normalized_weights {
        let next = prefix.last().copied().unwrap_or(0) + *weight;
        prefix.push(next);
    }

    let mut ranges = Vec::with_capacity(target_jobs);
    let mut start = 0usize;

    for job_index in 1..target_jobs {
        let min_end = start + 1;
        let max_end = len - (target_jobs - job_index);
        let target_prefix = total_weight.saturating_mul(job_index).div_ceil(target_jobs);

        let mut end = min_end;
        while end < max_end && prefix[end] < target_prefix {
            end += 1;
        }

        ranges.push(start..end);
        start = end;
    }

    ranges.push(start..len);
    ranges
}

/// Signature shared by generated chunk functions emitted by the current AOT
/// pipeline.
pub type GeneratedChunkFn = fn(&[f64], &mut [f64]);

/// Policy that decides when a parallel executor should fall back to the
/// sequential path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelFallbackPolicy {
    /// Use built-in heuristics tuned for the current implementation.
    Auto,
    /// Never fall back automatically. This is mainly useful for experiments
    /// and benchmarking the true parallel path.
    Never,
    /// Fall back when the total residual output count or sparse nnz count is
    /// below the user-provided thresholds.
    Thresholds {
        min_residual_outputs: usize,
        min_sparse_values: usize,
    },
}

impl Default for ParallelFallbackPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

/// User-tunable configuration for runtime parallel executors.
///
/// The defaults keep the current production-oriented heuristics, but callers
/// can override them if they want to:
/// - force the true parallel path for benchmarking,
/// - lower or raise fallback thresholds for a specific workload,
/// - or choose how many grouped jobs are created relative to rayon workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelExecutorConfig {
    /// How many grouped jobs to prepare per rayon worker thread.
    pub jobs_per_worker: usize,
    /// Optional explicit residual job count override used instead of the
    /// worker-based heuristic.
    pub max_residual_jobs: Option<usize>,
    /// Optional explicit sparse Jacobian job count override used instead of the
    /// worker-based heuristic.
    pub max_sparse_jobs: Option<usize>,
    /// Sequential fallback policy.
    pub fallback_policy: ParallelFallbackPolicy,
}

impl Default for ParallelExecutorConfig {
    fn default() -> Self {
        Self {
            jobs_per_worker: 1,
            max_residual_jobs: None,
            max_sparse_jobs: None,
            fallback_policy: ParallelFallbackPolicy::Auto,
        }
    }
}

impl ParallelExecutorConfig {
    fn worker_jobs(self) -> usize {
        let workers = rayon::current_num_threads().max(1);
        workers.saturating_mul(self.jobs_per_worker.max(1))
    }

    fn residual_jobs(self) -> usize {
        self.max_residual_jobs
            .unwrap_or_else(|| self.worker_jobs())
            .max(1)
    }

    fn sparse_jobs(self) -> usize {
        self.max_sparse_jobs
            .unwrap_or_else(|| self.worker_jobs())
            .max(1)
    }

    fn residual_fallback(self, output_len: usize, job_count: usize) -> bool {
        if job_count <= 1 {
            return true;
        }

        match self.fallback_policy {
            ParallelFallbackPolicy::Auto => output_len < DEFAULT_MIN_RESIDUAL_OUTPUTS_FOR_PARALLEL,
            ParallelFallbackPolicy::Never => false,
            ParallelFallbackPolicy::Thresholds {
                min_residual_outputs,
                ..
            } => output_len < min_residual_outputs,
        }
    }

    fn sparse_fallback(self, nnz: usize, job_count: usize) -> bool {
        if job_count <= 1 {
            return true;
        }

        match self.fallback_policy {
            ParallelFallbackPolicy::Auto => nnz < DEFAULT_MIN_SPARSE_VALUES_FOR_PARALLEL,
            ParallelFallbackPolicy::Never => false,
            ParallelFallbackPolicy::Thresholds {
                min_sparse_values, ..
            } => nnz < min_sparse_values,
        }
    }
}

/// Binding between one residual chunk plan and one generated function.
#[derive(Clone, Copy, Debug)]
pub struct ResidualChunkBinding<'a> {
    pub fn_name: &'a str,
    pub output_offset: usize,
    pub output_len: usize,
    pub eval: GeneratedChunkFn,
}

/// Binding between one sparse Jacobian-values chunk plan and one generated
/// function.
#[derive(Clone, Copy, Debug)]
pub struct SparseJacobianChunkBinding<'a> {
    pub fn_name: &'a str,
    pub value_offset: usize,
    pub value_len: usize,
    pub eval: GeneratedChunkFn,
}

/// Binding between one dense Jacobian chunk plan and one generated function.
#[derive(Clone, Copy, Debug)]
pub struct DenseJacobianChunkBinding<'a> {
    pub fn_name: &'a str,
    pub value_offset: usize,
    pub value_len: usize,
    pub eval: GeneratedChunkFn,
}

/// Sequential executor for residual chunks.
///
/// This is intentionally the simplest production-shaped execution object:
/// - generated chunks are called in order,
/// - each chunk writes into its own output slice,
/// - there is no shared mutable state between chunks beyond the caller-owned
///   residual buffer.
pub struct SequentialResidualExecutor<'a> {
    plan: &'a ResidualRuntimePlan<'a>,
    chunks: Vec<ResidualChunkBinding<'a>>,
}

impl<'a> SequentialResidualExecutor<'a> {
    /// Creates a sequential residual executor.
    pub fn new(plan: &'a ResidualRuntimePlan<'a>, chunks: Vec<ResidualChunkBinding<'a>>) -> Self {
        assert_eq!(
            plan.chunks.len(),
            chunks.len(),
            "residual chunk binding count must match runtime plan"
        );

        for (planned, bound) in plan.chunks.iter().zip(chunks.iter()) {
            assert_eq!(
                planned.output_offset, bound.output_offset,
                "residual chunk output offset mismatch for {}",
                bound.fn_name
            );
            assert_eq!(
                planned.residuals.len(),
                bound.output_len,
                "residual chunk output length mismatch for {}",
                bound.fn_name
            );
        }

        Self { plan, chunks }
    }

    /// Evaluates all residual chunks into a caller-provided output slice.
    pub fn eval_into(&self, args: &[f64], out: &mut [f64]) {
        assert!(
            out.len() >= self.plan.output_len,
            "expected at least {} residual output slots, got {}",
            self.plan.output_len,
            out.len()
        );

        for chunk in &self.chunks {
            let start = chunk.output_offset;
            let end = start + chunk.output_len;
            (chunk.eval)(args, &mut out[start..end]);
        }
    }

    /// Evaluates all residual chunks and returns the full residual vector.
    pub fn eval(&self, args: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.plan.output_len];
        self.eval_into(args, &mut out);
        out
    }
}

/// Sequential executor for sparse Jacobian value chunks.
///
/// It computes the explicit sparse values in global solver order and can
/// assemble the final `SparseColMat` using the structure stored in the runtime
/// plan.
pub struct SequentialSparseJacobianExecutor<'a> {
    plan: &'a SparseJacobianRuntimePlan<'a>,
    chunks: Vec<SparseJacobianChunkBinding<'a>>,
}

impl<'a> SequentialSparseJacobianExecutor<'a> {
    /// Creates a sequential sparse Jacobian executor.
    pub fn new(
        plan: &'a SparseJacobianRuntimePlan<'a>,
        chunks: Vec<SparseJacobianChunkBinding<'a>>,
    ) -> Self {
        assert_eq!(
            plan.chunks.len(),
            chunks.len(),
            "sparse Jacobian chunk binding count must match runtime plan"
        );

        for (planned, bound) in plan.chunks.iter().zip(chunks.iter()) {
            assert_eq!(
                planned.value_offset, bound.value_offset,
                "sparse Jacobian chunk value offset mismatch for {}",
                bound.fn_name
            );
            assert_eq!(
                planned.entries.len(),
                bound.value_len,
                "sparse Jacobian chunk length mismatch for {}",
                bound.fn_name
            );
        }

        Self { plan, chunks }
    }

    /// Evaluates all sparse Jacobian chunks into a caller-provided values
    /// buffer following the global explicit entry order.
    pub fn eval_values_into(&self, args: &[f64], values_out: &mut [f64]) {
        assert!(
            values_out.len() >= self.plan.nnz(),
            "expected at least {} Jacobian value slots, got {}",
            self.plan.nnz(),
            values_out.len()
        );

        for chunk in &self.chunks {
            let start = chunk.value_offset;
            let end = start + chunk.value_len;
            (chunk.eval)(args, &mut values_out[start..end]);
        }
    }

    /// Evaluates all sparse Jacobian chunks and returns explicit non-zero
    /// values in global solver order.
    pub fn eval_values(&self, args: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.plan.nnz()];
        self.eval_values_into(args, &mut values);
        values
    }

    /// Evaluates all sparse Jacobian chunks and assembles the final
    /// `SparseColMat`.
    pub fn eval_sparse_col_mat(&self, args: &[f64]) -> SparseColMat<usize, f64> {
        let values = self.eval_values(args);
        self.plan.assemble_sparse_col_mat(values.as_slice())
    }
}

/// Sequential executor for dense Jacobian row chunks.
///
/// Dense Jacobian chunks write row-major values into disjoint slices of a
/// caller-provided flat output buffer and can then be assembled into a
/// `nalgebra::DMatrix`.
pub struct SequentialDenseJacobianExecutor<'a> {
    plan: &'a DenseJacobianRuntimePlan<'a>,
    chunks: Vec<DenseJacobianChunkBinding<'a>>,
}

impl<'a> SequentialDenseJacobianExecutor<'a> {
    /// Creates a sequential dense Jacobian executor.
    pub fn new(
        plan: &'a DenseJacobianRuntimePlan<'a>,
        chunks: Vec<DenseJacobianChunkBinding<'a>>,
    ) -> Self {
        assert_eq!(
            plan.chunks.len(),
            chunks.len(),
            "dense Jacobian chunk binding count must match runtime plan"
        );

        for (planned, bound) in plan.chunks.iter().zip(chunks.iter()) {
            assert_eq!(
                planned.value_offset, bound.value_offset,
                "dense Jacobian chunk value offset mismatch for {}",
                bound.fn_name
            );
            assert_eq!(
                planned.value_range().len(),
                bound.value_len,
                "dense Jacobian chunk length mismatch for {}",
                bound.fn_name
            );
        }

        Self { plan, chunks }
    }

    /// Evaluates all dense Jacobian chunks into a caller-provided flat
    /// row-major output buffer.
    pub fn eval_values_into(&self, args: &[f64], values_out: &mut [f64]) {
        assert!(
            values_out.len() >= self.plan.len(),
            "expected at least {} dense Jacobian value slots, got {}",
            self.plan.len(),
            values_out.len()
        );

        for chunk in &self.chunks {
            let start = chunk.value_offset;
            let end = start + chunk.value_len;
            (chunk.eval)(args, &mut values_out[start..end]);
        }
    }

    /// Evaluates all dense Jacobian chunks and returns row-major values.
    pub fn eval_values(&self, args: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.plan.len()];
        self.eval_values_into(args, &mut values);
        values
    }

    /// Evaluates all dense Jacobian chunks and assembles the final matrix.
    pub fn eval_dense_matrix(&self, args: &[f64]) -> DMatrix<f64> {
        let values = self.eval_values(args);
        self.plan.assemble_dense_matrix(values.as_slice())
    }
}

/// Parallel executor for dense Jacobian row chunks.
///
/// Dense Jacobian chunks write row-major values into disjoint slices of one
/// flat output buffer and can then be assembled into a `DMatrix<f64>`.
pub struct ParallelDenseJacobianExecutor<'a> {
    plan: &'a DenseJacobianRuntimePlan<'a>,
    chunks: Vec<DenseJacobianChunkBinding<'a>>,
    job_ranges: Vec<Range<usize>>,
    job_value_ranges: Vec<Range<usize>>,
    use_sequential_fallback: bool,
    config: ParallelExecutorConfig,
}

impl<'a> ParallelDenseJacobianExecutor<'a> {
    pub fn new(
        plan: &'a DenseJacobianRuntimePlan<'a>,
        chunks: Vec<DenseJacobianChunkBinding<'a>>,
    ) -> Self {
        Self::with_config(plan, chunks, ParallelExecutorConfig::default())
    }

    pub fn with_config(
        plan: &'a DenseJacobianRuntimePlan<'a>,
        chunks: Vec<DenseJacobianChunkBinding<'a>>,
        config: ParallelExecutorConfig,
    ) -> Self {
        let sequential = SequentialDenseJacobianExecutor::new(plan, chunks);
        Self::assert_non_overlapping_value_layout(&sequential.chunks, sequential.plan.len());
        let worker_jobs = config.sparse_jobs();
        let chunk_weights: Vec<usize> = sequential
            .chunks
            .iter()
            .map(|chunk| chunk.value_len)
            .collect();
        let job_ranges = contiguous_weighted_job_ranges(chunk_weights.as_slice(), worker_jobs);
        let job_value_ranges = job_ranges
            .iter()
            .map(|chunk_range| {
                let first = &sequential.chunks[chunk_range.start];
                let last = &sequential.chunks[chunk_range.end - 1];
                first.value_offset..(last.value_offset + last.value_len)
            })
            .collect();
        let use_sequential_fallback =
            config.sparse_fallback(sequential.plan.len(), job_ranges.len());

        Self {
            plan: sequential.plan,
            chunks: sequential.chunks,
            job_ranges,
            job_value_ranges,
            use_sequential_fallback,
            config,
        }
    }

    fn assert_non_overlapping_value_layout(chunks: &[DenseJacobianChunkBinding<'_>], len: usize) {
        let mut expected_offset = 0usize;
        for chunk in chunks {
            assert_eq!(
                chunk.value_offset, expected_offset,
                "parallel dense Jacobian chunks must form a contiguous, non-overlapping layout"
            );
            expected_offset += chunk.value_len;
        }
        assert_eq!(
            expected_offset, len,
            "parallel dense Jacobian chunk layout must cover all dense values"
        );
    }

    pub fn eval_values_into(&self, args: &[f64], values_out: &mut [f64]) {
        assert!(
            values_out.len() >= self.plan.len(),
            "expected at least {} dense Jacobian value slots, got {}",
            self.plan.len(),
            values_out.len()
        );

        if self.use_sequential_fallback {
            for chunk in &self.chunks {
                let start = chunk.value_offset;
                let end = start + chunk.value_len;
                (chunk.eval)(args, &mut values_out[start..end]);
            }
            return;
        }

        match self.job_ranges.len() {
            2 => self.eval_two_jobs_with_join(args, values_out),
            4 => self.eval_four_jobs_with_join(args, values_out),
            _ => self.eval_many_jobs_with_scope(args, values_out),
        }
    }

    pub fn eval_values(&self, args: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.plan.len()];
        self.eval_values_into(args, &mut values);
        values
    }

    pub fn eval_dense_matrix(&self, args: &[f64]) -> DMatrix<f64> {
        let values = self.eval_values(args);
        self.plan.assemble_dense_matrix(values.as_slice())
    }

    pub fn uses_sequential_fallback(&self) -> bool {
        self.use_sequential_fallback
    }

    pub fn job_count(&self) -> usize {
        self.job_ranges.len()
    }

    pub fn config(&self) -> ParallelExecutorConfig {
        self.config
    }

    fn eval_job_into(
        &self,
        args: &[f64],
        job_index: usize,
        values_out: &mut [f64],
        base_offset: usize,
    ) {
        let chunk_range = self.job_ranges[job_index].clone();
        for chunk in &self.chunks[chunk_range] {
            let start = chunk.value_offset - base_offset;
            let end = start + chunk.value_len;
            (chunk.eval)(args, &mut values_out[start..end]);
        }
    }

    fn eval_two_jobs_with_join(&self, args: &[f64], values_out: &mut [f64]) {
        let left_range = self.job_value_ranges[0].clone();
        let right_range = self.job_value_ranges[1].clone();
        let split_at = right_range.start;
        let (left_out, right_out) = values_out.split_at_mut(split_at);
        join(
            || self.eval_job_into(args, 0, left_out, left_range.start),
            || self.eval_job_into(args, 1, right_out, right_range.start),
        );
    }

    fn eval_four_jobs_with_join(&self, args: &[f64], values_out: &mut [f64]) {
        let r0 = self.job_value_ranges[0].clone();
        let r1 = self.job_value_ranges[1].clone();
        let r2 = self.job_value_ranges[2].clone();
        let r3 = self.job_value_ranges[3].clone();

        let (left_half, right_half) = values_out.split_at_mut(r2.start);
        let (left0, left1) = left_half.split_at_mut(r1.start - r0.start);
        let (right0, right1) = right_half.split_at_mut(r3.start - r2.start);

        join(
            || {
                join(
                    || self.eval_job_into(args, 0, left0, r0.start),
                    || self.eval_job_into(args, 1, left1, r1.start),
                );
            },
            || {
                join(
                    || self.eval_job_into(args, 2, right0, r2.start),
                    || self.eval_job_into(args, 3, right1, r3.start),
                );
            },
        );
    }

    fn eval_many_jobs_with_scope(&self, args: &[f64], values_out: &mut [f64]) {
        let values_ptr = values_out.as_mut_ptr() as usize;
        scope(|scope| {
            for job_range in &self.job_ranges {
                let args = args;
                let job_range = job_range.clone();
                let chunks = &self.chunks;
                scope.spawn(move |_| {
                    for chunk in &chunks[job_range] {
                        let slice_ptr = (values_ptr as *mut f64).wrapping_add(chunk.value_offset);
                        let out_slice =
                            unsafe { std::slice::from_raw_parts_mut(slice_ptr, chunk.value_len) };
                        (chunk.eval)(args, out_slice);
                    }
                });
            }
        });
    }
}

/// Parallel executor for residual chunks.
///
/// The first version stays maximally safe and deterministic:
/// - each chunk computes into its own temporary buffer in parallel,
/// - results are copied into the final residual slice in solver order.
///
/// This avoids aliasing issues while already matching the future runtime model
/// of "parallel chunk execution + deterministic assembly".
pub struct ParallelResidualExecutor<'a> {
    plan: &'a ResidualRuntimePlan<'a>,
    chunks: Vec<ResidualChunkBinding<'a>>,
    job_ranges: Vec<Range<usize>>,
    job_output_ranges: Vec<Range<usize>>,
    use_sequential_fallback: bool,
    config: ParallelExecutorConfig,
}

impl<'a> ParallelResidualExecutor<'a> {
    pub fn new(plan: &'a ResidualRuntimePlan<'a>, chunks: Vec<ResidualChunkBinding<'a>>) -> Self {
        Self::with_config(plan, chunks, ParallelExecutorConfig::default())
    }

    pub fn with_config(
        plan: &'a ResidualRuntimePlan<'a>,
        chunks: Vec<ResidualChunkBinding<'a>>,
        config: ParallelExecutorConfig,
    ) -> Self {
        let sequential = SequentialResidualExecutor::new(plan, chunks);
        Self::assert_non_overlapping_residual_layout(
            &sequential.chunks,
            sequential.plan.output_len,
        );
        let worker_jobs = config.residual_jobs();
        let chunk_weights: Vec<usize> = sequential
            .chunks
            .iter()
            .map(|chunk| chunk.output_len)
            .collect();
        let job_ranges = contiguous_weighted_job_ranges(chunk_weights.as_slice(), worker_jobs);
        let job_output_ranges = job_ranges
            .iter()
            .map(|chunk_range| {
                let first = &sequential.chunks[chunk_range.start];
                let last = &sequential.chunks[chunk_range.end - 1];
                first.output_offset..(last.output_offset + last.output_len)
            })
            .collect();
        let use_sequential_fallback =
            config.residual_fallback(sequential.plan.output_len, job_ranges.len());
        Self {
            plan: sequential.plan,
            chunks: sequential.chunks,
            job_ranges,
            job_output_ranges,
            use_sequential_fallback,
            config,
        }
    }

    fn assert_non_overlapping_residual_layout(
        chunks: &[ResidualChunkBinding<'_>],
        output_len: usize,
    ) {
        let mut expected_offset = 0usize;
        for chunk in chunks {
            assert_eq!(
                chunk.output_offset, expected_offset,
                "parallel residual chunks must form a contiguous, non-overlapping layout"
            );
            expected_offset += chunk.output_len;
        }
        assert_eq!(
            expected_offset, output_len,
            "parallel residual chunk layout must cover the full output"
        );
    }

    pub fn eval_into(&self, args: &[f64], out: &mut [f64]) {
        assert!(
            out.len() >= self.plan.output_len,
            "expected at least {} residual output slots, got {}",
            self.plan.output_len,
            out.len()
        );

        if self.use_sequential_fallback {
            for chunk in &self.chunks {
                let start = chunk.output_offset;
                let end = start + chunk.output_len;
                (chunk.eval)(args, &mut out[start..end]);
            }
            return;
        }

        match self.job_ranges.len() {
            2 => self.eval_two_jobs_with_join(args, out),
            4 => self.eval_four_jobs_with_join(args, out),
            _ => self.eval_many_jobs_with_scope(args, out),
        }
    }

    pub fn eval(&self, args: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.plan.output_len];
        self.eval_into(args, &mut out);
        out
    }

    /// Returns `true` when this executor intentionally uses the sequential
    /// path because the current workload is too small to amortize parallel
    /// scheduling overhead.
    pub fn uses_sequential_fallback(&self) -> bool {
        self.use_sequential_fallback
    }

    /// Returns the number of grouped worker jobs prepared for parallel
    /// execution.
    pub fn job_count(&self) -> usize {
        self.job_ranges.len()
    }

    /// Returns the effective configuration used by this executor.
    pub fn config(&self) -> ParallelExecutorConfig {
        self.config
    }

    fn eval_job_into(&self, args: &[f64], job_index: usize, out: &mut [f64], base_offset: usize) {
        let chunk_range = self.job_ranges[job_index].clone();
        for chunk in &self.chunks[chunk_range] {
            let start = chunk.output_offset - base_offset;
            let end = start + chunk.output_len;
            (chunk.eval)(args, &mut out[start..end]);
        }
    }

    fn eval_two_jobs_with_join(&self, args: &[f64], out: &mut [f64]) {
        let left_range = self.job_output_ranges[0].clone();
        let right_range = self.job_output_ranges[1].clone();
        let split_at = right_range.start;
        let (left_out, right_out) = out.split_at_mut(split_at);
        join(
            || self.eval_job_into(args, 0, left_out, left_range.start),
            || self.eval_job_into(args, 1, right_out, right_range.start),
        );
    }

    fn eval_four_jobs_with_join(&self, args: &[f64], out: &mut [f64]) {
        let r0 = self.job_output_ranges[0].clone();
        let r1 = self.job_output_ranges[1].clone();
        let r2 = self.job_output_ranges[2].clone();
        let r3 = self.job_output_ranges[3].clone();

        let (left_half, right_half) = out.split_at_mut(r2.start);
        let (left0, left1) = left_half.split_at_mut(r1.start - r0.start);
        let (right0, right1) = right_half.split_at_mut(r3.start - r2.start);

        join(
            || {
                join(
                    || self.eval_job_into(args, 0, left0, r0.start),
                    || self.eval_job_into(args, 1, left1, r1.start),
                );
            },
            || {
                join(
                    || self.eval_job_into(args, 2, right0, r2.start),
                    || self.eval_job_into(args, 3, right1, r3.start),
                );
            },
        );
    }

    fn eval_many_jobs_with_scope(&self, args: &[f64], out: &mut [f64]) {
        let out_ptr = out.as_mut_ptr() as usize;
        scope(|scope| {
            for job_range in &self.job_ranges {
                let args = args;
                let job_range = job_range.clone();
                let chunks = &self.chunks;
                scope.spawn(move |_| {
                    for chunk in &chunks[job_range] {
                        let slice_ptr = (out_ptr as *mut f64).wrapping_add(chunk.output_offset);
                        // SAFETY:
                        // - layout contiguity/non-overlap is validated in `new()`,
                        // - each job processes a disjoint contiguous chunk group,
                        // - therefore each inner chunk still writes to a unique output range,
                        // - `out` lives for the duration of this scope.
                        let out_slice =
                            unsafe { std::slice::from_raw_parts_mut(slice_ptr, chunk.output_len) };
                        (chunk.eval)(args, out_slice);
                    }
                });
            }
        });
    }
}

/// Parallel executor for sparse Jacobian value chunks.
///
/// Just like the residual executor, this version computes each chunk into an
/// independent temporary value buffer in parallel and then joins them into the
/// global explicit sparse ordering.
pub struct ParallelSparseJacobianExecutor<'a> {
    plan: &'a SparseJacobianRuntimePlan<'a>,
    chunks: Vec<SparseJacobianChunkBinding<'a>>,
    job_ranges: Vec<Range<usize>>,
    job_value_ranges: Vec<Range<usize>>,
    use_sequential_fallback: bool,
    config: ParallelExecutorConfig,
}

impl<'a> ParallelSparseJacobianExecutor<'a> {
    pub fn new(
        plan: &'a SparseJacobianRuntimePlan<'a>,
        chunks: Vec<SparseJacobianChunkBinding<'a>>,
    ) -> Self {
        Self::with_config(plan, chunks, ParallelExecutorConfig::default())
    }

    pub fn with_config(
        plan: &'a SparseJacobianRuntimePlan<'a>,
        chunks: Vec<SparseJacobianChunkBinding<'a>>,
        config: ParallelExecutorConfig,
    ) -> Self {
        let sequential = SequentialSparseJacobianExecutor::new(plan, chunks);
        Self::assert_non_overlapping_value_layout(&sequential.chunks, sequential.plan.nnz());
        let worker_jobs = config.sparse_jobs();
        let chunk_weights: Vec<usize> = sequential
            .chunks
            .iter()
            .map(|chunk| chunk.value_len)
            .collect();
        let job_ranges = contiguous_weighted_job_ranges(chunk_weights.as_slice(), worker_jobs);
        let job_value_ranges = job_ranges
            .iter()
            .map(|chunk_range| {
                let first = &sequential.chunks[chunk_range.start];
                let last = &sequential.chunks[chunk_range.end - 1];
                first.value_offset..(last.value_offset + last.value_len)
            })
            .collect();
        let use_sequential_fallback =
            config.sparse_fallback(sequential.plan.nnz(), job_ranges.len());
        Self {
            plan: sequential.plan,
            chunks: sequential.chunks,
            job_ranges,
            job_value_ranges,
            use_sequential_fallback,
            config,
        }
    }

    fn assert_non_overlapping_value_layout(chunks: &[SparseJacobianChunkBinding<'_>], nnz: usize) {
        let mut expected_offset = 0usize;
        for chunk in chunks {
            assert_eq!(
                chunk.value_offset, expected_offset,
                "parallel sparse Jacobian chunks must form a contiguous, non-overlapping layout"
            );
            expected_offset += chunk.value_len;
        }
        assert_eq!(
            expected_offset, nnz,
            "parallel sparse Jacobian chunk layout must cover all explicit values"
        );
    }

    pub fn eval_values_into(&self, args: &[f64], values_out: &mut [f64]) {
        assert!(
            values_out.len() >= self.plan.nnz(),
            "expected at least {} Jacobian value slots, got {}",
            self.plan.nnz(),
            values_out.len()
        );

        if self.use_sequential_fallback {
            for chunk in &self.chunks {
                let start = chunk.value_offset;
                let end = start + chunk.value_len;
                (chunk.eval)(args, &mut values_out[start..end]);
            }
            return;
        }

        match self.job_ranges.len() {
            2 => self.eval_two_jobs_with_join(args, values_out),
            4 => self.eval_four_jobs_with_join(args, values_out),
            _ => self.eval_many_jobs_with_scope(args, values_out),
        }
    }

    pub fn eval_values(&self, args: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.plan.nnz()];
        self.eval_values_into(args, &mut values);
        values
    }

    pub fn eval_sparse_col_mat(&self, args: &[f64]) -> SparseColMat<usize, f64> {
        let values = self.eval_values(args);
        self.plan.assemble_sparse_col_mat(values.as_slice())
    }

    /// Returns `true` when this executor intentionally uses the sequential
    /// path because the current workload is too small to amortize parallel
    /// scheduling overhead.
    pub fn uses_sequential_fallback(&self) -> bool {
        self.use_sequential_fallback
    }

    /// Returns the number of grouped worker jobs prepared for parallel
    /// execution.
    pub fn job_count(&self) -> usize {
        self.job_ranges.len()
    }

    /// Returns the effective configuration used by this executor.
    pub fn config(&self) -> ParallelExecutorConfig {
        self.config
    }

    fn eval_job_into(
        &self,
        args: &[f64],
        job_index: usize,
        values_out: &mut [f64],
        base_offset: usize,
    ) {
        let chunk_range = self.job_ranges[job_index].clone();
        for chunk in &self.chunks[chunk_range] {
            let start = chunk.value_offset - base_offset;
            let end = start + chunk.value_len;
            (chunk.eval)(args, &mut values_out[start..end]);
        }
    }

    fn eval_two_jobs_with_join(&self, args: &[f64], values_out: &mut [f64]) {
        let left_range = self.job_value_ranges[0].clone();
        let right_range = self.job_value_ranges[1].clone();
        let split_at = right_range.start;
        let (left_out, right_out) = values_out.split_at_mut(split_at);
        join(
            || self.eval_job_into(args, 0, left_out, left_range.start),
            || self.eval_job_into(args, 1, right_out, right_range.start),
        );
    }

    fn eval_four_jobs_with_join(&self, args: &[f64], values_out: &mut [f64]) {
        let r0 = self.job_value_ranges[0].clone();
        let r1 = self.job_value_ranges[1].clone();
        let r2 = self.job_value_ranges[2].clone();
        let r3 = self.job_value_ranges[3].clone();

        let (left_half, right_half) = values_out.split_at_mut(r2.start);
        let (left0, left1) = left_half.split_at_mut(r1.start - r0.start);
        let (right0, right1) = right_half.split_at_mut(r3.start - r2.start);

        join(
            || {
                join(
                    || self.eval_job_into(args, 0, left0, r0.start),
                    || self.eval_job_into(args, 1, left1, r1.start),
                );
            },
            || {
                join(
                    || self.eval_job_into(args, 2, right0, r2.start),
                    || self.eval_job_into(args, 3, right1, r3.start),
                );
            },
        );
    }

    fn eval_many_jobs_with_scope(&self, args: &[f64], values_out: &mut [f64]) {
        let values_ptr = values_out.as_mut_ptr() as usize;
        scope(|scope| {
            for job_range in &self.job_ranges {
                let args = args;
                let job_range = job_range.clone();
                let chunks = &self.chunks;
                scope.spawn(move |_| {
                    for chunk in &chunks[job_range] {
                        let slice_ptr = (values_ptr as *mut f64).wrapping_add(chunk.value_offset);
                        // SAFETY:
                        // - layout contiguity/non-overlap is validated in `new()`,
                        // - each job processes a disjoint contiguous chunk group,
                        // - therefore each inner chunk still writes to a unique value range,
                        // - `values_out` lives for the duration of this scope.
                        let out_slice =
                            unsafe { std::slice::from_raw_parts_mut(slice_ptr, chunk.value_len) };
                        (chunk.eval)(args, out_slice);
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DenseJacobianChunkBinding, GeneratedChunkFn, ParallelDenseJacobianExecutor,
        ParallelExecutorConfig, ParallelFallbackPolicy, ParallelResidualExecutor,
        ParallelSparseJacobianExecutor, ResidualChunkBinding, SequentialDenseJacobianExecutor,
        SequentialResidualExecutor, SequentialSparseJacobianExecutor, SparseJacobianChunkBinding,
        borrowed_sparse_entries,
    };
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy, RuntimeArguments,
    };
    use crate::symbolic::codegen_tasks::{
        JacobianTask, ResidualTask, SparseChunkingStrategy, SparseJacobianTask,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::Jacobian;
    use crate::symbolic::test_codegen_generated_bvp_fixtures::generated_bvp_fixture;
    use std::collections::HashMap;

    fn build_real_bvp_damp1_case(n_steps: usize) -> Jacobian {
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];
        let values = vec!["z".to_string(), "y".to_string()];

        let mut border_conditions = HashMap::new();
        border_conditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        border_conditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

        let mut jac = Jacobian::new();
        jac.discretization_system_BVP_par(
            eq_system,
            values,
            "x".to_string(),
            0.0,
            Some(n_steps),
            None,
            None,
            border_conditions,
            None,
            None,
            "forward".to_string(),
        );
        jac.calc_jacobian_parallel_smart_optimized();
        jac
    }

    fn bind_residual_fixture_chunks<'a>(
        plan: &'a crate::symbolic::codegen_runtime_api::ResidualRuntimePlan<'a>,
    ) -> Vec<ResidualChunkBinding<'a>> {
        let chunk_fns: [GeneratedChunkFn; 2] = [
            generated_bvp_fixture::fixture_bvp_residual_chunk_0,
            generated_bvp_fixture::fixture_bvp_residual_chunk_1,
        ];

        plan.chunks
            .iter()
            .zip(chunk_fns)
            .map(|(chunk, eval)| ResidualChunkBinding {
                fn_name: chunk.fn_name.as_str(),
                output_offset: chunk.output_offset,
                output_len: chunk.residuals.len(),
                eval,
            })
            .collect()
    }

    fn bind_sparse_fixture_chunks<'a>(
        plan: &'a crate::symbolic::codegen_runtime_api::SparseJacobianRuntimePlan<'a>,
    ) -> Vec<SparseJacobianChunkBinding<'a>> {
        let chunk_fns: [GeneratedChunkFn; 16] = [
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_0,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_1,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_2,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_3,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_4,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_5,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_6,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_7,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_8,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_9,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_10,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_11,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_12,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_13,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_14,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_15,
        ];

        plan.chunks
            .iter()
            .zip(chunk_fns)
            .map(|(chunk, eval)| SparseJacobianChunkBinding {
                fn_name: chunk.fn_name.as_str(),
                value_offset: chunk.value_offset,
                value_len: chunk.entries.len(),
                eval,
            })
            .collect()
    }

    fn dense_chunk_0(args: &[f64], out: &mut [f64]) {
        out[0] = args[0] + 1.0;
        out[1] = args[1];
        out[2] = 2.0;
        out[3] = args[0] * args[1];
    }

    fn dense_chunk_1(args: &[f64], out: &mut [f64]) {
        out[0] = args[0] - args[1];
        out[1] = 3.0;
    }

    fn bind_dense_fixture_chunks<'a>(
        plan: &'a crate::symbolic::codegen_runtime_api::DenseJacobianRuntimePlan<'a>,
    ) -> Vec<DenseJacobianChunkBinding<'a>> {
        let chunk_fns: [GeneratedChunkFn; 2] = [dense_chunk_0, dense_chunk_1];

        plan.chunks
            .iter()
            .zip(chunk_fns)
            .map(|(chunk, eval)| DenseJacobianChunkBinding {
                fn_name: chunk.fn_name.as_str(),
                value_offset: chunk.value_offset,
                value_len: chunk.value_range().len(),
                eval,
            })
            .collect()
    }

    #[test]
    fn sequential_dense_jacobian_executor_assembles_dense_matrix() {
        let jacobian = vec![
            vec![Expr::parse_expression("x + 1"), Expr::parse_expression("y")],
            vec![Expr::parse_expression("2"), Expr::parse_expression("x*y")],
            vec![Expr::parse_expression("x - y"), Expr::parse_expression("3")],
        ];
        let task = JacobianTask {
            fn_name: "fixture_dense_jacobian",
            jacobian: &jacobian,
            variables: &["x", "y"],
            params: None,
        };
        let runtime_plan =
            task.runtime_plan(DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 });
        let executor = SequentialDenseJacobianExecutor::new(
            &runtime_plan,
            bind_dense_fixture_chunks(&runtime_plan),
        );

        let matrix = executor.eval_dense_matrix(&[2.0, 5.0]);

        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], 3.0);
        assert_eq!(matrix[(0, 1)], 5.0);
        assert_eq!(matrix[(1, 0)], 2.0);
        assert_eq!(matrix[(1, 1)], 10.0);
        assert_eq!(matrix[(2, 0)], -3.0);
        assert_eq!(matrix[(2, 1)], 3.0);
    }

    #[test]
    fn parallel_dense_jacobian_executor_matches_sequential_executor() {
        let jacobian = vec![
            vec![Expr::parse_expression("x + 1"), Expr::parse_expression("y")],
            vec![Expr::parse_expression("2"), Expr::parse_expression("x*y")],
            vec![Expr::parse_expression("x - y"), Expr::parse_expression("3")],
        ];
        let task = JacobianTask {
            fn_name: "fixture_dense_jacobian",
            jacobian: &jacobian,
            variables: &["x", "y"],
            params: None,
        };
        let runtime_plan =
            task.runtime_plan(DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 });
        let bindings = bind_dense_fixture_chunks(&runtime_plan);
        let sequential = SequentialDenseJacobianExecutor::new(&runtime_plan, bindings.clone());
        let parallel = ParallelDenseJacobianExecutor::with_config(
            &runtime_plan,
            bindings,
            ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: None,
                max_sparse_jobs: Some(2),
                fallback_policy: ParallelFallbackPolicy::Never,
            },
        );

        assert_eq!(
            parallel.eval_dense_matrix(&[2.0, 5.0]),
            sequential.eval_dense_matrix(&[2.0, 5.0])
        );
    }

    #[test]
    fn sequential_residual_executor_matches_manual_fixture_collection() {
        let residuals = vec![
            Expr::parse_expression("y0 - y1 - 1"),
            Expr::parse_expression("-y0 + y2 - 1"),
            Expr::parse_expression("-y1^3 - y2 + y4"),
            Expr::parse_expression("-y3 + y5 - y2"),
            Expr::parse_expression("-y3^3 - y4 + y6"),
            Expr::parse_expression("-y5 + y7 - y4"),
            Expr::parse_expression("-y5^3 - y6 + y8"),
            Expr::parse_expression("-y7 + y9 - y6"),
            Expr::parse_expression("-y7^3 - y8 + y10"),
            Expr::parse_expression("-y9 + y11 - y8"),
            Expr::parse_expression("-y9^3 - y10 + y12"),
            Expr::parse_expression("-y11 + y13 - y10"),
            Expr::parse_expression("-y11^3 - y12 + y14"),
            Expr::parse_expression("-y13 + y15 - y12"),
            Expr::parse_expression("-y13^3 - y14 + 1"),
            Expr::parse_expression("-y15 + 1 - y14"),
        ];
        let variables = [
            "y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "y11", "y12", "y13",
            "y14", "y15",
        ];
        let task = ResidualTask {
            fn_name: "fixture_bvp_residual",
            residuals: &residuals,
            variables: &variables,
            params: None,
        };
        let runtime_plan = task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 8,
        });
        let executor = SequentialResidualExecutor::new(
            &runtime_plan,
            bind_residual_fixture_chunks(&runtime_plan),
        );

        let flat_args = RuntimeArguments::new(
            &(0..16)
                .map(|index| 0.2 + index as f64 * 0.01)
                .collect::<Vec<_>>(),
            None,
        )
        .flatten();

        let actual = executor.eval(flat_args.as_slice());

        let mut expected = vec![0.0; 16];
        generated_bvp_fixture::fixture_bvp_residual_chunk_0(
            flat_args.as_slice(),
            &mut expected[0..8],
        );
        generated_bvp_fixture::fixture_bvp_residual_chunk_1(
            flat_args.as_slice(),
            &mut expected[8..16],
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn sequential_sparse_executor_matches_manual_fixture_collection() {
        let jac = build_real_bvp_damp1_case(8);
        let variable_names_owned = jac.variable_string.clone();
        let variables: Vec<&str> = variable_names_owned
            .iter()
            .map(|name| name.as_str())
            .collect();
        let entries_owned = jac.symbolic_jacobian_sparse_entries_owned();
        let entries = borrowed_sparse_entries(&entries_owned);
        let task = SparseJacobianTask {
            fn_name: "fixture_bvp_sparse_values",
            shape: (jac.vector_of_functions.len(), jac.vector_of_variables.len()),
            entries: &entries,
            variables: &variables,
            params: None,
        };
        let runtime_plan =
            task.runtime_plan(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 1 });
        let executor = SequentialSparseJacobianExecutor::new(
            &runtime_plan,
            bind_sparse_fixture_chunks(&runtime_plan),
        );

        let flat_args = RuntimeArguments::new(
            &(0..16)
                .map(|index| 0.2 + index as f64 * 0.01)
                .collect::<Vec<_>>(),
            None,
        )
        .flatten();

        let actual = executor.eval_values(flat_args.as_slice());

        let mut expected = Vec::new();
        let chunk_fns: [GeneratedChunkFn; 16] = [
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_0,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_1,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_2,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_3,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_4,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_5,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_6,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_7,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_8,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_9,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_10,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_11,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_12,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_13,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_14,
            generated_bvp_fixture::fixture_bvp_sparse_values_chunk_15,
        ];
        let chunk_sizes = [2usize, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2];
        for (chunk_fn, chunk_size) in chunk_fns.into_iter().zip(chunk_sizes) {
            let mut chunk_out = vec![0.0; chunk_size];
            chunk_fn(flat_args.as_slice(), &mut chunk_out);
            expected.extend(chunk_out);
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn parallel_residual_executor_matches_sequential_executor() {
        let residuals = vec![
            Expr::parse_expression("y0 - y1 - 1"),
            Expr::parse_expression("-y0 + y2 - 1"),
            Expr::parse_expression("-y1^3 - y2 + y4"),
            Expr::parse_expression("-y3 + y5 - y2"),
            Expr::parse_expression("-y3^3 - y4 + y6"),
            Expr::parse_expression("-y5 + y7 - y4"),
            Expr::parse_expression("-y5^3 - y6 + y8"),
            Expr::parse_expression("-y7 + y9 - y6"),
            Expr::parse_expression("-y7^3 - y8 + y10"),
            Expr::parse_expression("-y9 + y11 - y8"),
            Expr::parse_expression("-y9^3 - y10 + y12"),
            Expr::parse_expression("-y11 + y13 - y10"),
            Expr::parse_expression("-y11^3 - y12 + y14"),
            Expr::parse_expression("-y13 + y15 - y12"),
            Expr::parse_expression("-y13^3 - y14 + 1"),
            Expr::parse_expression("-y15 + 1 - y14"),
        ];
        let variables = [
            "y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "y11", "y12", "y13",
            "y14", "y15",
        ];
        let task = ResidualTask {
            fn_name: "fixture_bvp_residual",
            residuals: &residuals,
            variables: &variables,
            params: None,
        };
        let runtime_plan = task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 8,
        });
        let bindings = bind_residual_fixture_chunks(&runtime_plan);
        let sequential = SequentialResidualExecutor::new(&runtime_plan, bindings.clone());
        let parallel = ParallelResidualExecutor::new(&runtime_plan, bindings);

        let flat_args = RuntimeArguments::new(
            &(0..16)
                .map(|index| 0.2 + index as f64 * 0.01)
                .collect::<Vec<_>>(),
            None,
        )
        .flatten();

        assert_eq!(
            parallel.eval(flat_args.as_slice()),
            sequential.eval(flat_args.as_slice())
        );
    }

    #[test]
    fn parallel_sparse_executor_matches_sequential_executor() {
        let jac = build_real_bvp_damp1_case(8);
        let variable_names_owned = jac.variable_string.clone();
        let variables: Vec<&str> = variable_names_owned
            .iter()
            .map(|name| name.as_str())
            .collect();
        let entries_owned = jac.symbolic_jacobian_sparse_entries_owned();
        let entries = borrowed_sparse_entries(&entries_owned);
        let task = SparseJacobianTask {
            fn_name: "fixture_bvp_sparse_values",
            shape: (jac.vector_of_functions.len(), jac.vector_of_variables.len()),
            entries: &entries,
            variables: &variables,
            params: None,
        };
        let runtime_plan =
            task.runtime_plan(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 1 });
        let bindings = bind_sparse_fixture_chunks(&runtime_plan);
        let sequential = SequentialSparseJacobianExecutor::new(&runtime_plan, bindings.clone());
        let parallel = ParallelSparseJacobianExecutor::new(&runtime_plan, bindings);

        let flat_args = RuntimeArguments::new(
            &(0..16)
                .map(|index| 0.2 + index as f64 * 0.01)
                .collect::<Vec<_>>(),
            None,
        )
        .flatten();

        assert_eq!(
            parallel.eval_values(flat_args.as_slice()),
            sequential.eval_values(flat_args.as_slice())
        );
    }

    #[test]
    fn parallel_residual_executor_allows_disabling_fallback() {
        let residuals = vec![
            Expr::parse_expression("y0 - y1 - 1"),
            Expr::parse_expression("-y0 + y2 - 1"),
            Expr::parse_expression("-y1^3 - y2 + y4"),
            Expr::parse_expression("-y3 + y5 - y2"),
            Expr::parse_expression("-y3^3 - y4 + y6"),
            Expr::parse_expression("-y5 + y7 - y4"),
            Expr::parse_expression("-y5^3 - y6 + y8"),
            Expr::parse_expression("-y7 + y9 - y6"),
            Expr::parse_expression("-y7^3 - y8 + y10"),
            Expr::parse_expression("-y9 + y11 - y8"),
            Expr::parse_expression("-y9^3 - y10 + y12"),
            Expr::parse_expression("-y11 + y13 - y10"),
            Expr::parse_expression("-y11^3 - y12 + y14"),
            Expr::parse_expression("-y13 + y15 - y12"),
            Expr::parse_expression("-y13^3 - y14 + 1"),
            Expr::parse_expression("-y15 + 1 - y14"),
        ];
        let variables = [
            "y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "y11", "y12", "y13",
            "y14", "y15",
        ];
        let task = ResidualTask {
            fn_name: "fixture_bvp_residual",
            residuals: &residuals,
            variables: &variables,
            params: None,
        };
        let runtime_plan = task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 8,
        });
        let parallel = ParallelResidualExecutor::with_config(
            &runtime_plan,
            bind_residual_fixture_chunks(&runtime_plan),
            ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: None,
                max_sparse_jobs: None,
                fallback_policy: ParallelFallbackPolicy::Never,
            },
        );
        let flat_args = RuntimeArguments::new(
            &(0..16)
                .map(|index| 0.2 + index as f64 * 0.01)
                .collect::<Vec<_>>(),
            None,
        )
        .flatten();

        let out = parallel.eval(flat_args.as_slice());
        assert_eq!(out.len(), runtime_plan.output_len);
        assert!(!parallel.uses_sequential_fallback());
    }

    #[test]
    fn parallel_sparse_executor_allows_custom_thresholds() {
        let jac = build_real_bvp_damp1_case(8);
        let variable_names_owned = jac.variable_string.clone();
        let variables: Vec<&str> = variable_names_owned
            .iter()
            .map(|name| name.as_str())
            .collect();
        let entries_owned = jac.symbolic_jacobian_sparse_entries_owned();
        let entries = borrowed_sparse_entries(&entries_owned);
        let task = SparseJacobianTask {
            fn_name: "fixture_bvp_sparse_values",
            shape: (jac.vector_of_functions.len(), jac.vector_of_variables.len()),
            entries: &entries,
            variables: &variables,
            params: None,
        };
        let runtime_plan =
            task.runtime_plan(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 1 });
        let parallel = ParallelSparseJacobianExecutor::with_config(
            &runtime_plan,
            bind_sparse_fixture_chunks(&runtime_plan),
            ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: None,
                max_sparse_jobs: None,
                fallback_policy: ParallelFallbackPolicy::Thresholds {
                    min_residual_outputs: usize::MAX,
                    min_sparse_values: 0,
                },
            },
        );
        let flat_args = RuntimeArguments::new(
            &(0..16)
                .map(|index| 0.2 + index as f64 * 0.01)
                .collect::<Vec<_>>(),
            None,
        )
        .flatten();

        let values = parallel.eval_values(flat_args.as_slice());
        assert_eq!(values.len(), runtime_plan.nnz());
        assert!(!parallel.uses_sequential_fallback());
    }
}
