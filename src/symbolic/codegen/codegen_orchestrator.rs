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

use crate::somelinalg::banded::banded_assembly::BandedAssembly;
use crate::symbolic::codegen::codegen_runtime_api::{
    BandedJacobianRuntimePlan, DenseJacobianRuntimePlan, ResidualChunkingStrategy,
    ResidualRuntimePlan, SparseJacobianRuntimePlan,
    recommended_residual_chunking_for_auto_parallelism,
    recommended_row_chunking_for_auto_parallelism,
};
use crate::symbolic::codegen::codegen_tasks::{SparseChunkingStrategy, SparseExprEntry};
use faer::sparse::SparseColMat;
use nalgebra::DMatrix;
use rayon::{join, scope};
use std::ops::Range;
use std::sync::OnceLock;
use std::time::Instant;

// Runtime parallelism must be chosen from measured scheduling overhead rather
// than from one developer machine.  The unit-cost estimate is intentionally
// small and conservative: if the generated scalar operations are heavier than
// this, Auto will become more willing to parallelize on that workload after the
// join-overhead calibration moves the threshold down.
const DEFAULT_ESTIMATED_UNIT_COST_NS: f64 = 5.0;
const DEFAULT_PARALLEL_OVERHEAD_SAFETY_FACTOR: f64 = 2.0;

/// Cached process-local measurement of raw rayon join overhead on the current machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayonOverheadBaseline {
    /// Number of rayon workers visible to the current process.
    pub workers: usize,
    /// Median overhead of a two-way `rayon::join` with no-op closures.
    pub join2_ns: f64,
    /// Median overhead of a four-way nested `rayon::join` tree with no-op closures.
    pub join4_ns: f64,
}

/// Auto-tuned runtime recommendation for parallel executor configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoParallelRecommendation {
    /// Whether runtime parallel execution is expected to amortize scheduling cost.
    pub should_parallelize: bool,
    /// Recommended grouped residual job count.
    pub residual_jobs: usize,
    /// Recommended grouped sparse-Jacobian job count.
    pub sparse_jobs: usize,
    /// Minimum useful work per parallel job derived from the measured machine overhead.
    pub min_work_per_job: usize,
    /// Number of workers observed during calibration.
    pub workers: usize,
    /// Residual callback workload diagnostics.
    pub residual_stage: AutoParallelStagePlan,
    /// Sparse/Banded Jacobian callback workload diagnostics.
    pub sparse_stage: AutoParallelStagePlan,
}

/// Why `Auto` accepted or rejected parallel execution for one callback stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoParallelDecisionReason {
    /// There is no work for this callback stage.
    NoWork,
    /// The generated artifact has only one chunk or the current machine can use
    /// only one useful grouped job, so there is nothing meaningful to schedule.
    SingleChunkOrJob,
    /// The whole stage is too small to amortize measured rayon scheduling cost.
    TotalWorkTooSmall,
    /// Grouped jobs would be too light.
    WorkPerJobTooSmall,
    /// Individual generated chunks would be too light; this is the usual sign
    /// of over-fragmented generated code.
    WorkPerChunkTooSmall,
    /// The stage is large and coarse enough to run in parallel.
    ParallelCandidate,
}

impl AutoParallelDecisionReason {
    /// Short stable label for logs and story-table diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NoWork => "no_work",
            Self::SingleChunkOrJob => "single_chunk_or_job",
            Self::TotalWorkTooSmall => "total_work_too_small",
            Self::WorkPerJobTooSmall => "work_per_job_too_small",
            Self::WorkPerChunkTooSmall => "work_per_chunk_too_small",
            Self::ParallelCandidate => "parallel_candidate",
        }
    }
}

/// Per-stage workload diagnostics used by the `Auto` execution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoParallelStagePlan {
    /// Total scalar outputs/values represented by this stage.
    pub total_work: usize,
    /// Number of generated chunks available before grouped runtime scheduling.
    pub chunk_count: usize,
    /// Number of grouped runtime jobs selected for this stage.
    pub jobs: usize,
    /// Integer work per grouped runtime job.
    pub work_per_job: usize,
    /// Integer work per generated chunk.
    pub work_per_chunk: usize,
    /// Minimum useful work per job derived from measured machine overhead.
    pub min_work_per_job: usize,
    /// Decision reason for this stage.
    pub reason: AutoParallelDecisionReason,
}

/// High-level machine-aware execution mode chosen by the `Auto` parallel policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoExecutionMode {
    /// Keep runtime execution sequential because the measured machine overhead
    /// is expected to dominate the available work.
    Sequential,
    /// Use runtime parallel execution with the recommended grouped jobs.
    Parallel,
}

/// Full machine-aware recommendation for sparse residual/Jacobian runtime work.
///
/// This is the production-shaped contract for `Auto`:
/// - whether execution should stay sequential or go parallel,
/// - how residuals should be chunked if a new compiled artifact is generated,
/// - how sparse Jacobian values should be chunked,
/// - and which grouped runtime job counts should be used if the problem is
///   already compiled and only runtime binding is being selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseAutoParallelPlan {
    /// Final high-level execution decision.
    pub execution_mode: AutoExecutionMode,
    /// Recommended residual chunking for future code generation.
    pub residual_chunking: ResidualChunkingStrategy,
    /// Recommended sparse-Jacobian chunking for future code generation.
    pub sparse_chunking: SparseChunkingStrategy,
    /// Runtime grouped-job recommendation for already prepared/compiled plans.
    pub executor_config: Option<ParallelExecutorConfig>,
    /// Minimum useful work per job derived from the measured machine overhead.
    pub min_work_per_job: usize,
    /// Number of workers observed during calibration.
    pub workers: usize,
    /// Residual callback workload diagnostics.
    pub residual_stage: AutoParallelStagePlan,
    /// Sparse/Banded Jacobian callback workload diagnostics.
    pub sparse_stage: AutoParallelStagePlan,
}

fn measure_rayon_overhead_baseline() -> RayonOverheadBaseline {
    const ITERS: usize = 10_000;
    const SAMPLES: usize = 7;

    let median_per_iter = |timings: Vec<f64>| -> f64 {
        let mut sorted = timings;
        sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };

    let seq_samples: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..ITERS {
                std::hint::black_box(0u64);
            }
            start.elapsed().as_secs_f64() * 1e9 / ITERS as f64
        })
        .collect();

    let join2_samples: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..ITERS {
                join(|| std::hint::black_box(0u64), || std::hint::black_box(0u64));
            }
            start.elapsed().as_secs_f64() * 1e9 / ITERS as f64
        })
        .collect();

    let join4_samples: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..ITERS {
                join(
                    || join(|| std::hint::black_box(0u64), || std::hint::black_box(0u64)),
                    || join(|| std::hint::black_box(0u64), || std::hint::black_box(0u64)),
                );
            }
            start.elapsed().as_secs_f64() * 1e9 / ITERS as f64
        })
        .collect();

    let seq_ns = median_per_iter(seq_samples);
    let join2_ns = (median_per_iter(join2_samples) - seq_ns).max(0.0);
    let join4_ns = (median_per_iter(join4_samples) - seq_ns).max(join2_ns);

    RayonOverheadBaseline {
        workers: rayon::current_num_threads().max(1),
        join2_ns,
        join4_ns,
    }
}

/// Returns a cached measurement of raw rayon join overhead for the current process.
pub fn rayon_overhead_baseline() -> RayonOverheadBaseline {
    static BASELINE: OnceLock<RayonOverheadBaseline> = OnceLock::new();
    *BASELINE.get_or_init(measure_rayon_overhead_baseline)
}

fn min_work_per_job_from_baseline(baseline: RayonOverheadBaseline) -> usize {
    ((baseline.join4_ns * DEFAULT_PARALLEL_OVERHEAD_SAFETY_FACTOR / DEFAULT_ESTIMATED_UNIT_COST_NS)
        .ceil() as usize)
        .max(1)
}

fn work_per_group(total_work: usize, group_count: usize) -> usize {
    if group_count == 0 {
        0
    } else {
        total_work / group_count
    }
}

fn auto_parallel_fallback_for_workload(
    total_work: usize,
    chunk_count: usize,
    job_count: usize,
) -> bool {
    if job_count <= 1 || chunk_count <= 1 {
        return true;
    }

    let min_work_per_job = min_work_per_job_from_baseline(rayon_overhead_baseline());
    let min_total_work = min_work_per_job.saturating_mul(2);

    total_work < min_total_work
        || work_per_group(total_work, job_count) < min_work_per_job
        || work_per_group(total_work, chunk_count) < min_work_per_job
}

fn auto_parallel_stage_plan(
    total_work: usize,
    chunk_count: usize,
    job_count: usize,
    min_work_per_job: usize,
) -> AutoParallelStagePlan {
    let work_per_job = work_per_group(total_work, job_count);
    let work_per_chunk = work_per_group(total_work, chunk_count);
    let min_total_work = min_work_per_job.saturating_mul(2);
    let reason = if total_work == 0 {
        AutoParallelDecisionReason::NoWork
    } else if job_count <= 1 || chunk_count <= 1 {
        AutoParallelDecisionReason::SingleChunkOrJob
    } else if total_work < min_total_work {
        AutoParallelDecisionReason::TotalWorkTooSmall
    } else if work_per_job < min_work_per_job {
        AutoParallelDecisionReason::WorkPerJobTooSmall
    } else if work_per_chunk < min_work_per_job {
        AutoParallelDecisionReason::WorkPerChunkTooSmall
    } else {
        AutoParallelDecisionReason::ParallelCandidate
    };

    AutoParallelStagePlan {
        total_work,
        chunk_count,
        jobs: job_count,
        work_per_job,
        work_per_chunk,
        min_work_per_job,
        reason,
    }
}

fn recommended_job_count_for_workload(
    total_work: usize,
    chunk_count: usize,
    min_work_per_job: usize,
    workers: usize,
) -> usize {
    if chunk_count <= 1 || total_work < min_work_per_job.saturating_mul(2) {
        return 1;
    }

    let max_useful_jobs = (total_work / min_work_per_job).max(1);
    workers.min(chunk_count).min(max_useful_jobs).max(1)
}

/// Recommends whether runtime execution should stay sequential or go parallel,
/// and, if parallel, how many grouped residual/sparse jobs should be prepared.
pub fn auto_parallel_recommendation(
    residual_work: usize,
    residual_chunk_count: usize,
    sparse_work: usize,
    sparse_chunk_count: usize,
) -> AutoParallelRecommendation {
    let baseline = rayon_overhead_baseline();
    let min_work_per_job = min_work_per_job_from_baseline(baseline);
    let residual_jobs = recommended_job_count_for_workload(
        residual_work,
        residual_chunk_count,
        min_work_per_job,
        baseline.workers,
    );
    let sparse_jobs = recommended_job_count_for_workload(
        sparse_work,
        sparse_chunk_count,
        min_work_per_job,
        baseline.workers,
    );
    let residual_stage = auto_parallel_stage_plan(
        residual_work,
        residual_chunk_count,
        residual_jobs,
        min_work_per_job,
    );
    let sparse_stage = auto_parallel_stage_plan(
        sparse_work,
        sparse_chunk_count,
        sparse_jobs,
        min_work_per_job,
    );

    AutoParallelRecommendation {
        should_parallelize: matches!(
            residual_stage.reason,
            AutoParallelDecisionReason::ParallelCandidate
        ) || matches!(
            sparse_stage.reason,
            AutoParallelDecisionReason::ParallelCandidate
        ),
        residual_jobs,
        sparse_jobs,
        min_work_per_job,
        workers: baseline.workers,
        residual_stage,
        sparse_stage,
    }
}

/// Builds a machine-aware parallel config for compiled/runtime execution.
///
/// Returns `None` when the current machine and workload suggest that the
/// sequential path is still cheaper than runtime parallelization.
pub fn auto_parallel_executor_config(
    residual_work: usize,
    residual_chunk_count: usize,
    sparse_work: usize,
    sparse_chunk_count: usize,
) -> Option<ParallelExecutorConfig> {
    let recommendation = auto_parallel_recommendation(
        residual_work,
        residual_chunk_count,
        sparse_work,
        sparse_chunk_count,
    );
    if !recommendation.should_parallelize {
        return None;
    }

    Some(ParallelExecutorConfig {
        jobs_per_worker: 1,
        max_residual_jobs: Some(recommendation.residual_jobs.max(1)),
        max_sparse_jobs: Some(recommendation.sparse_jobs.max(1)),
        fallback_policy: ParallelFallbackPolicy::Never,
    })
}

fn residual_chunk_count(total_outputs: usize, strategy: ResidualChunkingStrategy) -> usize {
    match strategy {
        ResidualChunkingStrategy::Whole => usize::from(total_outputs > 0),
        ResidualChunkingStrategy::ByTargetChunkCount { target_chunks } => total_outputs
            .max(1)
            .div_ceil(total_outputs.max(1).div_ceil(target_chunks).max(1)),
        ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk,
        } => total_outputs.max(1).div_ceil(max_outputs_per_chunk.max(1)),
    }
}

fn sparse_chunk_count(total_rows: usize, strategy: SparseChunkingStrategy) -> usize {
    match strategy {
        SparseChunkingStrategy::Whole => usize::from(total_rows > 0),
        SparseChunkingStrategy::ByTargetChunkCount { target_chunks } => total_rows
            .max(1)
            .div_ceil(total_rows.max(1).div_ceil(target_chunks).max(1)),
        SparseChunkingStrategy::ByNonZeroCount { .. } => 1,
        SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
            total_rows.max(1).div_ceil(rows_per_chunk.max(1))
        }
    }
}

/// Builds a complete `Auto` recommendation for sparse residual/Jacobian work.
///
/// This helper combines three inputs:
/// - measured machine-specific rayon overhead,
/// - coarse problem-size characteristics,
/// - and conservative per-worker chunking heuristics.
///
/// The returned plan can be used in two different scenarios:
/// - generation time: use `residual_chunking` / `sparse_chunking`;
/// - runtime binding of an already compiled artifact: use `executor_config`.
pub fn recommended_sparse_auto_parallel_plan(
    residual_work: usize,
    sparse_rows: usize,
    sparse_work: usize,
) -> SparseAutoParallelPlan {
    let baseline = rayon_overhead_baseline();
    let min_work_per_job = min_work_per_job_from_baseline(baseline);

    if residual_work == 0 || sparse_rows == 0 || sparse_work == 0 {
        let residual_stage = auto_parallel_stage_plan(residual_work, 0, 1, min_work_per_job);
        let sparse_stage = auto_parallel_stage_plan(sparse_work, 0, 1, min_work_per_job);
        return SparseAutoParallelPlan {
            execution_mode: AutoExecutionMode::Sequential,
            residual_chunking: ResidualChunkingStrategy::Whole,
            sparse_chunking: SparseChunkingStrategy::Whole,
            executor_config: None,
            min_work_per_job,
            workers: baseline.workers,
            residual_stage,
            sparse_stage,
        };
    }

    let residual_chunking = recommended_residual_chunking_for_auto_parallelism(residual_work);
    let sparse_chunking = recommended_row_chunking_for_auto_parallelism(sparse_rows);
    let recommendation = auto_parallel_recommendation(
        residual_work,
        residual_chunk_count(residual_work, residual_chunking),
        sparse_work,
        sparse_chunk_count(sparse_rows, sparse_chunking),
    );

    if !recommendation.should_parallelize {
        return SparseAutoParallelPlan {
            execution_mode: AutoExecutionMode::Sequential,
            residual_chunking: ResidualChunkingStrategy::Whole,
            sparse_chunking: SparseChunkingStrategy::Whole,
            executor_config: None,
            min_work_per_job: recommendation.min_work_per_job,
            workers: recommendation.workers,
            residual_stage: recommendation.residual_stage,
            sparse_stage: recommendation.sparse_stage,
        };
    }

    SparseAutoParallelPlan {
        execution_mode: AutoExecutionMode::Parallel,
        residual_chunking,
        sparse_chunking,
        executor_config: Some(ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: Some(recommendation.residual_jobs.max(1)),
            max_sparse_jobs: Some(recommendation.sparse_jobs.max(1)),
            fallback_policy: ParallelFallbackPolicy::Never,
        }),
        min_work_per_job: recommendation.min_work_per_job,
        workers: recommendation.workers,
        residual_stage: recommendation.residual_stage,
        sparse_stage: recommendation.sparse_stage,
    }
}

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

    fn residual_fallback(self, output_len: usize, chunk_count: usize, job_count: usize) -> bool {
        if job_count <= 1 {
            return true;
        }

        match self.fallback_policy {
            ParallelFallbackPolicy::Never => false,
            ParallelFallbackPolicy::Auto => {
                auto_parallel_fallback_for_workload(output_len, chunk_count, job_count)
            }
            ParallelFallbackPolicy::Thresholds {
                min_residual_outputs,
                ..
            } => output_len < min_residual_outputs,
        }
    }

    fn sparse_fallback(self, nnz: usize, chunk_count: usize, job_count: usize) -> bool {
        if job_count <= 1 {
            return true;
        }

        match self.fallback_policy {
            ParallelFallbackPolicy::Never => false,
            ParallelFallbackPolicy::Auto => {
                auto_parallel_fallback_for_workload(nnz, chunk_count, job_count)
            }
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

/// Binding between one banded Jacobian-values chunk plan and one generated
/// function.
#[derive(Clone, Copy, Debug)]
pub struct BandedJacobianChunkBinding<'a> {
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

/// Sequential executor for banded Jacobian value chunks.
///
/// It computes the banded values in the global diagonal-major order and can
/// assemble the final native `BandedAssembly` using the structure stored in
/// the runtime plan.
pub struct SequentialBandedJacobianExecutor<'a> {
    plan: &'a BandedJacobianRuntimePlan<'a>,
    chunks: Vec<BandedJacobianChunkBinding<'a>>,
}

impl<'a> SequentialBandedJacobianExecutor<'a> {
    /// Creates a sequential banded Jacobian executor.
    pub fn new(
        plan: &'a BandedJacobianRuntimePlan<'a>,
        chunks: Vec<BandedJacobianChunkBinding<'a>>,
    ) -> Self {
        assert_eq!(
            plan.chunks.len(),
            chunks.len(),
            "banded Jacobian chunk binding count must match runtime plan"
        );

        for (planned, bound) in plan.chunks.iter().zip(chunks.iter()) {
            assert_eq!(
                planned.value_offset, bound.value_offset,
                "banded Jacobian chunk value offset mismatch for {}",
                bound.fn_name
            );
            assert_eq!(
                planned.entries.len(),
                bound.value_len,
                "banded Jacobian chunk length mismatch for {}",
                bound.fn_name
            );
        }

        Self { plan, chunks }
    }

    /// Evaluates all banded Jacobian chunks into a caller-provided values
    /// buffer.
    pub fn eval_values_into(&self, args: &[f64], values_out: &mut [f64]) {
        assert!(
            values_out.len() >= self.plan.nnz(),
            "expected at least {} banded Jacobian value slots, got {}",
            self.plan.nnz(),
            values_out.len()
        );

        for chunk in &self.chunks {
            let start = chunk.value_offset;
            let end = start + chunk.value_len;
            (chunk.eval)(args, &mut values_out[start..end]);
        }
    }

    /// Evaluates all banded Jacobian chunks and returns explicit values in
    /// diagonal-major order.
    pub fn eval_values(&self, args: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.plan.nnz()];
        self.eval_values_into(args, &mut values);
        values
    }

    /// Evaluates all banded Jacobian chunks and assembles the final native
    /// `BandedAssembly`.
    pub fn eval_banded_assembly(&self, args: &[f64]) -> BandedAssembly {
        let values = self.eval_values(args);
        self.plan.assemble_banded_assembly(values.as_slice())
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
        let use_sequential_fallback = config.sparse_fallback(
            sequential.plan.len(),
            sequential.chunks.len(),
            job_ranges.len(),
        );

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
            _ => self.eval_many_jobs_with_join_tree(args, values_out),
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

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn work_per_job(&self) -> usize {
        work_per_group(self.plan.len(), self.job_count())
    }

    pub fn work_per_chunk(&self) -> usize {
        work_per_group(self.plan.len(), self.chunk_count())
    }

    pub fn min_work_per_parallel_job(&self) -> usize {
        min_work_per_job_from_baseline(rayon_overhead_baseline())
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

    fn eval_many_jobs_with_join_tree(&self, args: &[f64], values_out: &mut [f64]) {
        self.eval_job_range_recursive(args, 0, self.job_ranges.len(), values_out, 0);
    }

    fn eval_job_range_recursive(
        &self,
        args: &[f64],
        job_start: usize,
        job_end: usize,
        values_out: &mut [f64],
        base_offset: usize,
    ) {
        match job_end - job_start {
            0 => {}
            1 => self.eval_job_into(args, job_start, values_out, base_offset),
            _ => {
                let mid = job_start + (job_end - job_start) / 2;
                let right_range = self.job_value_ranges[mid].clone();
                let split_at = right_range.start - base_offset;
                let (left_out, right_out) = values_out.split_at_mut(split_at);
                join(
                    || self.eval_job_range_recursive(args, job_start, mid, left_out, base_offset),
                    || {
                        self.eval_job_range_recursive(
                            args,
                            mid,
                            job_end,
                            right_out,
                            right_range.start,
                        )
                    },
                );
            }
        }
    }
}

/// Parallel executor for residual chunks.
///
/// Jobs write directly into disjoint slices of one caller-owned residual
/// buffer. This keeps the deterministic chunk order while avoiding the
/// temporary-buffer/copy-back pattern that was too expensive for hot callbacks.
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
        let use_sequential_fallback = config.residual_fallback(
            sequential.plan.output_len,
            sequential.chunks.len(),
            job_ranges.len(),
        );
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
            _ => self.eval_many_jobs_with_join_tree(args, out),
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

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn work_per_job(&self) -> usize {
        work_per_group(self.plan.output_len, self.job_count())
    }

    pub fn work_per_chunk(&self) -> usize {
        work_per_group(self.plan.output_len, self.chunk_count())
    }

    pub fn min_work_per_parallel_job(&self) -> usize {
        min_work_per_job_from_baseline(rayon_overhead_baseline())
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

    fn eval_many_jobs_with_join_tree(&self, args: &[f64], out: &mut [f64]) {
        self.eval_job_range_recursive(args, 0, self.job_ranges.len(), out, 0);
    }

    fn eval_job_range_recursive(
        &self,
        args: &[f64],
        job_start: usize,
        job_end: usize,
        out: &mut [f64],
        base_offset: usize,
    ) {
        match job_end - job_start {
            0 => {}
            1 => self.eval_job_into(args, job_start, out, base_offset),
            _ => {
                let mid = job_start + (job_end - job_start) / 2;
                let right_range = self.job_output_ranges[mid].clone();
                let split_at = right_range.start - base_offset;
                let (left_out, right_out) = out.split_at_mut(split_at);
                join(
                    || self.eval_job_range_recursive(args, job_start, mid, left_out, base_offset),
                    || {
                        self.eval_job_range_recursive(
                            args,
                            mid,
                            job_end,
                            right_out,
                            right_range.start,
                        )
                    },
                );
            }
        }
    }
}

/// Parallel executor for sparse Jacobian value chunks.
///
/// Jobs write directly into disjoint slices of the explicit sparse-values
/// buffer, preserving global sparse ordering without a merge step.
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
        let use_sequential_fallback = config.sparse_fallback(
            sequential.plan.nnz(),
            sequential.chunks.len(),
            job_ranges.len(),
        );
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
            _ => self.eval_many_jobs_with_join_tree(args, values_out),
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

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn work_per_job(&self) -> usize {
        work_per_group(self.plan.nnz(), self.job_count())
    }

    pub fn work_per_chunk(&self) -> usize {
        work_per_group(self.plan.nnz(), self.chunk_count())
    }

    pub fn min_work_per_parallel_job(&self) -> usize {
        min_work_per_job_from_baseline(rayon_overhead_baseline())
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

    fn eval_many_jobs_with_join_tree(&self, args: &[f64], values_out: &mut [f64]) {
        self.eval_job_range_recursive(args, 0, self.job_ranges.len(), values_out, 0);
    }

    fn eval_job_range_recursive(
        &self,
        args: &[f64],
        job_start: usize,
        job_end: usize,
        values_out: &mut [f64],
        base_offset: usize,
    ) {
        match job_end - job_start {
            0 => {}
            1 => self.eval_job_into(args, job_start, values_out, base_offset),
            _ => {
                let mid = job_start + (job_end - job_start) / 2;
                let right_range = self.job_value_ranges[mid].clone();
                let split_at = right_range.start - base_offset;
                let (left_out, right_out) = values_out.split_at_mut(split_at);
                join(
                    || self.eval_job_range_recursive(args, job_start, mid, left_out, base_offset),
                    || {
                        self.eval_job_range_recursive(
                            args,
                            mid,
                            job_end,
                            right_out,
                            right_range.start,
                        )
                    },
                );
            }
        }
    }
}

/// Parallel executor for banded Jacobian value chunks.
///
/// This mirrors the sparse-value executor, but the final assembly target is a
/// native `BandedAssembly` instead of a sparse triplet matrix.
pub struct ParallelBandedJacobianExecutor<'a> {
    plan: &'a BandedJacobianRuntimePlan<'a>,
    chunks: Vec<BandedJacobianChunkBinding<'a>>,
    job_ranges: Vec<Range<usize>>,
    job_value_ranges: Vec<Range<usize>>,
    use_sequential_fallback: bool,
    config: ParallelExecutorConfig,
}

impl<'a> ParallelBandedJacobianExecutor<'a> {
    pub fn new(
        plan: &'a BandedJacobianRuntimePlan<'a>,
        chunks: Vec<BandedJacobianChunkBinding<'a>>,
    ) -> Self {
        Self::with_config(plan, chunks, ParallelExecutorConfig::default())
    }

    pub fn with_config(
        plan: &'a BandedJacobianRuntimePlan<'a>,
        chunks: Vec<BandedJacobianChunkBinding<'a>>,
        config: ParallelExecutorConfig,
    ) -> Self {
        let sequential = SequentialBandedJacobianExecutor::new(plan, chunks);
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
        let use_sequential_fallback = config.sparse_fallback(
            sequential.plan.nnz(),
            sequential.chunks.len(),
            job_ranges.len(),
        );
        Self {
            plan: sequential.plan,
            chunks: sequential.chunks,
            job_ranges,
            job_value_ranges,
            use_sequential_fallback,
            config,
        }
    }

    fn assert_non_overlapping_value_layout(chunks: &[BandedJacobianChunkBinding<'_>], nnz: usize) {
        let mut expected_offset = 0usize;
        for chunk in chunks {
            assert_eq!(
                chunk.value_offset, expected_offset,
                "parallel banded Jacobian chunks must form a contiguous, non-overlapping layout"
            );
            expected_offset += chunk.value_len;
        }
        assert_eq!(
            expected_offset, nnz,
            "parallel banded Jacobian chunk layout must cover all explicit values"
        );
    }

    pub fn eval_values_into(&self, args: &[f64], values_out: &mut [f64]) {
        assert!(
            values_out.len() >= self.plan.nnz(),
            "expected at least {} banded Jacobian value slots, got {}",
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
            _ => self.eval_many_jobs_with_join_tree(args, values_out),
        }
    }

    pub fn eval_values(&self, args: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.plan.nnz()];
        self.eval_values_into(args, &mut values);
        values
    }

    pub fn eval_banded_assembly(&self, args: &[f64]) -> BandedAssembly {
        let values = self.eval_values(args);
        self.plan.assemble_banded_assembly(values.as_slice())
    }

    pub fn uses_sequential_fallback(&self) -> bool {
        self.use_sequential_fallback
    }

    pub fn job_count(&self) -> usize {
        self.job_ranges.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn work_per_job(&self) -> usize {
        work_per_group(self.plan.nnz(), self.job_count())
    }

    pub fn work_per_chunk(&self) -> usize {
        work_per_group(self.plan.nnz(), self.chunk_count())
    }

    pub fn min_work_per_parallel_job(&self) -> usize {
        min_work_per_job_from_baseline(rayon_overhead_baseline())
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

    fn eval_many_jobs_with_join_tree(&self, args: &[f64], values_out: &mut [f64]) {
        self.eval_job_range_recursive(args, 0, self.job_ranges.len(), values_out, 0);
    }

    fn eval_job_range_recursive(
        &self,
        args: &[f64],
        job_start: usize,
        job_end: usize,
        values_out: &mut [f64],
        base_offset: usize,
    ) {
        match job_end - job_start {
            0 => {}
            1 => self.eval_job_into(args, job_start, values_out, base_offset),
            _ => {
                let mid = job_start + (job_end - job_start) / 2;
                let right_range = self.job_value_ranges[mid].clone();
                let split_at = right_range.start - base_offset;
                let (left_out, right_out) = values_out.split_at_mut(split_at);
                join(
                    || self.eval_job_range_recursive(args, job_start, mid, left_out, base_offset),
                    || {
                        self.eval_job_range_recursive(
                            args,
                            mid,
                            job_end,
                            right_out,
                            right_range.start,
                        )
                    },
                );
            }
        }
    }
}
//===================================================================================
// TESTS
//=====================================================================================
#[cfg(test)]
mod tests {
    use super::{
        AutoExecutionMode, BandedJacobianChunkBinding, DenseJacobianChunkBinding, GeneratedChunkFn,
        ParallelBandedJacobianExecutor, ParallelDenseJacobianExecutor, ParallelExecutorConfig,
        ParallelFallbackPolicy, ParallelResidualExecutor, ParallelSparseJacobianExecutor,
        ResidualChunkBinding, SequentialBandedJacobianExecutor, SequentialDenseJacobianExecutor,
        SequentialResidualExecutor, SequentialSparseJacobianExecutor, SparseJacobianChunkBinding,
        auto_parallel_fallback_for_workload, borrowed_sparse_entries,
        min_work_per_job_from_baseline, rayon_overhead_baseline,
        recommended_sparse_auto_parallel_plan, work_per_group,
    };
    use crate::somelinalg::banded::banded_assembly::BandedAssembly;
    use crate::symbolic::codegen::codegen_runtime_api::{
        BandedJacobianRuntimePlan, DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
        RuntimeArguments,
    };
    use crate::symbolic::codegen::codegen_tasks::{
        BandedChunkingStrategy, BandedExprEntry, BandedJacobianTask, JacobianTask, ResidualTask,
        SparseChunkingStrategy, SparseJacobianTask,
    };
    use crate::symbolic::codegen::testing_fixtures::test_codegen_generated_bvp_fixtures::generated_bvp_fixture;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::Jacobian;
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
        plan: &'a crate::symbolic::codegen::codegen_runtime_api::ResidualRuntimePlan<'a>,
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
        plan: &'a crate::symbolic::codegen::codegen_runtime_api::SparseJacobianRuntimePlan<'a>,
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
        plan: &'a crate::symbolic::codegen::codegen_runtime_api::DenseJacobianRuntimePlan<'a>,
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

    fn banded_chunk_0(args: &[f64], out: &mut [f64]) {
        out[0] = args[0] + 1.0;
        out[1] = args[1] + 2.0;
    }

    fn banded_chunk_1(args: &[f64], out: &mut [f64]) {
        out[0] = args[0] * args[1];
        out[1] = args[0] - args[1];
    }

    fn bind_banded_fixture_chunks<'a>(
        plan: &'a BandedJacobianRuntimePlan<'a>,
    ) -> Vec<BandedJacobianChunkBinding<'a>> {
        let chunk_fns: [GeneratedChunkFn; 2] = [banded_chunk_0, banded_chunk_1];

        plan.chunks
            .iter()
            .zip(chunk_fns)
            .map(|(chunk, eval)| BandedJacobianChunkBinding {
                fn_name: chunk.fn_name.as_str(),
                value_offset: chunk.value_offset,
                value_len: chunk.entries.len(),
                eval,
            })
            .collect()
    }

    fn banded_diagonals_snapshot(assembly: &BandedAssembly) -> Vec<(isize, Vec<f64>)> {
        (-(assembly.kl() as isize)..=(assembly.ku() as isize))
            .map(|offset| {
                (
                    offset,
                    assembly
                        .diag(offset)
                        .expect("requested diagonal must exist")
                        .to_vec(),
                )
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
    fn sequential_banded_jacobian_executor_assembles_expected_native_storage() {
        let expressions = vec![
            Expr::parse_expression("x + 1"),
            Expr::parse_expression("y + 2"),
            Expr::parse_expression("x * y"),
            Expr::parse_expression("x - y"),
        ];
        let entries = vec![
            BandedExprEntry {
                row: 1,
                col: 0,
                diag_offset: -1,
                diag_position: 0,
                expr: &expressions[2],
            },
            BandedExprEntry {
                row: 2,
                col: 1,
                diag_offset: -1,
                diag_position: 1,
                expr: &expressions[3],
            },
            BandedExprEntry {
                row: 0,
                col: 0,
                diag_offset: 0,
                diag_position: 0,
                expr: &expressions[0],
            },
            BandedExprEntry {
                row: 1,
                col: 1,
                diag_offset: 0,
                diag_position: 1,
                expr: &expressions[1],
            },
        ];
        let task = BandedJacobianTask {
            fn_name: "fixture_banded_jacobian",
            shape: (3, 3),
            kl: 1,
            ku: 0,
            entries: &entries,
            variables: &["x", "y"],
            params: None,
        };
        let runtime_plan = task.runtime_plan(BandedChunkingStrategy::ByDiagonalCount {
            diagonals_per_chunk: 1,
        });
        let executor = SequentialBandedJacobianExecutor::new(
            &runtime_plan,
            bind_banded_fixture_chunks(&runtime_plan),
        );

        let values = executor.eval_values(&[2.0, 5.0]);
        let assembly = executor.eval_banded_assembly(&[2.0, 5.0]);
        let expected_values = vec![3.0, 7.0, 10.0, -3.0];
        let expected = runtime_plan.assemble_banded_assembly(&expected_values);

        assert_eq!(values, expected_values);
        assert_eq!(
            banded_diagonals_snapshot(&assembly),
            banded_diagonals_snapshot(&expected)
        );
    }

    #[test]
    fn parallel_banded_jacobian_executor_matches_sequential_executor() {
        let expressions = vec![
            Expr::parse_expression("x + 1"),
            Expr::parse_expression("y + 2"),
            Expr::parse_expression("x * y"),
            Expr::parse_expression("x - y"),
        ];
        let entries = vec![
            BandedExprEntry {
                row: 1,
                col: 0,
                diag_offset: -1,
                diag_position: 0,
                expr: &expressions[2],
            },
            BandedExprEntry {
                row: 2,
                col: 1,
                diag_offset: -1,
                diag_position: 1,
                expr: &expressions[3],
            },
            BandedExprEntry {
                row: 0,
                col: 0,
                diag_offset: 0,
                diag_position: 0,
                expr: &expressions[0],
            },
            BandedExprEntry {
                row: 1,
                col: 1,
                diag_offset: 0,
                diag_position: 1,
                expr: &expressions[1],
            },
        ];
        let task = BandedJacobianTask {
            fn_name: "fixture_banded_jacobian",
            shape: (3, 3),
            kl: 1,
            ku: 0,
            entries: &entries,
            variables: &["x", "y"],
            params: None,
        };
        let runtime_plan = task.runtime_plan(BandedChunkingStrategy::ByDiagonalCount {
            diagonals_per_chunk: 1,
        });
        let bindings = bind_banded_fixture_chunks(&runtime_plan);
        let sequential = SequentialBandedJacobianExecutor::new(&runtime_plan, bindings.clone());
        let parallel = ParallelBandedJacobianExecutor::with_config(
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
            parallel.eval_values(&[2.0, 5.0]),
            sequential.eval_values(&[2.0, 5.0])
        );
        assert_eq!(
            banded_diagonals_snapshot(&parallel.eval_banded_assembly(&[2.0, 5.0])),
            banded_diagonals_snapshot(&sequential.eval_banded_assembly(&[2.0, 5.0]))
        );
        assert!(!parallel.uses_sequential_fallback());
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

    #[test]
    fn sparse_auto_parallel_plan_stays_sequential_for_tiny_workloads() {
        let plan = recommended_sparse_auto_parallel_plan(16, 16, 32);

        assert_eq!(plan.execution_mode, AutoExecutionMode::Sequential);
        assert_eq!(plan.residual_chunking, ResidualChunkingStrategy::Whole);
        assert_eq!(plan.sparse_chunking, SparseChunkingStrategy::Whole);
        assert!(plan.executor_config.is_none());
        assert!(plan.min_work_per_job >= 1);
        assert!(plan.workers >= 1);
    }

    #[test]
    fn sparse_auto_parallel_plan_exposes_parallel_shape_for_large_workloads() {
        let baseline = rayon_overhead_baseline();
        let min_work = min_work_per_job_from_baseline(baseline);
        let rows = min_work
            .saturating_mul(baseline.workers.max(2))
            .saturating_mul(4)
            .max(1536);
        let plan = recommended_sparse_auto_parallel_plan(rows, rows, rows.saturating_mul(4));

        assert!(plan.min_work_per_job >= 1);
        assert!(plan.workers >= 1);
        if plan.workers == 1 {
            assert_eq!(plan.execution_mode, AutoExecutionMode::Sequential);
            assert!(plan.executor_config.is_none());
            return;
        }

        assert_eq!(plan.execution_mode, AutoExecutionMode::Parallel);
        assert!(matches!(
            plan.residual_chunking,
            ResidualChunkingStrategy::ByOutputCount { .. }
        ));
        assert!(matches!(
            plan.sparse_chunking,
            SparseChunkingStrategy::ByRowCount { .. }
        ));
        let config = plan
            .executor_config
            .expect("large sparse workloads should produce runtime parallel config");
        assert_eq!(config.jobs_per_worker, 1);
        assert!(config.max_residual_jobs.unwrap_or(0) >= 1);
        assert!(config.max_sparse_jobs.unwrap_or(0) >= 2);
        assert_eq!(config.fallback_policy, ParallelFallbackPolicy::Never);
    }

    #[test]
    fn auto_fallback_rejects_overfragmented_medium_chunks() {
        let min_work = min_work_per_job_from_baseline(rayon_overhead_baseline());
        let total_work = min_work.saturating_mul(8);
        let chunk_count = 16;
        let job_count = 4;

        assert!(
            work_per_group(total_work, job_count) >= min_work,
            "test setup should have enough work per job"
        );
        assert!(
            work_per_group(total_work, chunk_count) < min_work,
            "test setup should expose too little work per chunk"
        );
        assert!(
            auto_parallel_fallback_for_workload(total_work, chunk_count, job_count),
            "Auto should stay sequential when chunks are too fine-grained even if grouped jobs look large enough"
        );
    }

    #[test]
    fn auto_fallback_allows_coarse_large_chunks() {
        let min_work = min_work_per_job_from_baseline(rayon_overhead_baseline());
        let total_work = min_work.saturating_mul(16);
        let chunk_count = 4;
        let job_count = 4;

        assert!(
            work_per_group(total_work, job_count) >= min_work,
            "test setup should have enough work per job"
        );
        assert!(
            work_per_group(total_work, chunk_count) >= min_work,
            "test setup should have enough work per chunk"
        );
        assert!(
            !auto_parallel_fallback_for_workload(total_work, chunk_count, job_count),
            "Auto should permit parallel execution when both jobs and chunks are coarse enough"
        );
    }
}
