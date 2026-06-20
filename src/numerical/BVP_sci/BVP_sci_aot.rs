//! AtomView + AOT backend wiring for `BVP_sci`.
//!
//! This module isolates the compiled-backend lifecycle from the ExprLegacy path:
//! - symbolic preparation for AtomView
//! - sparse runtime planning
//! - AOT build/materialize/link
//! - faer-native residual/Jacobian adapters
//!
//! The goal is to mirror the mature `BVP_Damp` AOT pipeline while keeping the
//! solver core in `BVP_sci_faer` unchanged.

use crate::numerical::BVP_Damp::BVP_utils::record_callback_stage_time;
use crate::numerical::BVP_sci::BVP_sci_faer::{
    BCFunction, ODEBandedJacobian, ODEFunction, ODEJacobian, faer_col, faer_dense_mat, faer_mat,
};
use crate::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciBackendError};
use crate::numerical::BVP_sci::BVP_sci_symbolic_functions::Jacobian_sci_faer;
use crate::somelinalg::banded::banded_assembly::BandedAssembly;
use crate::symbolic::codegen::c_backend::codegen_c_aot_build::CAotCompileConfig;
use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::{
    register_generated_c_banded_backend, register_generated_c_sparse_backend,
};
use crate::symbolic::codegen::codegen_aot_driver::{
    AotBuildPreset, AotCodegenBackend, ExecutedGeneratedAotBuild, GeneratedAotBuildRequest,
    GeneratedAotBuildResult, generated_aot_artifact_from_prepared_problem,
    generated_aot_build_request_from_artifact,
};
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedResidualChunk, LinkedSparseJacobianChunk, register_generated_banded_cdylib_backend,
    register_generated_sparse_cdylib_backend, resolve_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedBandedProblem, PreparedProblem, PreparedSparseProblem,
};
use crate::symbolic::codegen::codegen_runtime_api::{
    BandedJacobianRuntimePlan, BandedJacobianStructure, BandedJacobianValuesChunkPlan,
    ResidualChunkPlan, ResidualChunkingStrategy, ResidualRuntimePlan, SparseJacobianRuntimePlan,
    SparseJacobianStructure, SparseJacobianValuesChunkPlan,
};
use crate::symbolic::codegen::codegen_tasks::{
    BandedChunkingStrategy, BandedExprEntry, CodegenOutputLayout, CodegenTaskKind, CodegenTaskPlan,
    PlannedOutput, SparseChunkingStrategy, SparseExprEntry,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_registry::register_zig_build_in_registry;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::{
    register_generated_zig_banded_backend, register_generated_zig_sparse_backend,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use faer::sparse::{SparseColMat, Triplet};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
/// Backend mode for BVP_sci generated (AOT-compiled) sparse Jacobian/residual evaluation.
///
/// # Constraint
///
/// **All 8 variants operate exclusively on the sparse matrix backend.**
/// There is no dense AOT path in BVP_sci. If you need a dense generated backend,
/// use the BVP_Damp solver infrastructure instead.
///
/// # Variant overview
///
/// | Variant | Description |
/// |---------|-------------|
/// | [`LambdifyOnly`](BvpSciGeneratedBackendMode::LambdifyOnly) | Pure interpreted lambdify — no AOT compilation. |
/// | [`RequirePrebuiltAot`](BvpSciGeneratedBackendMode::RequirePrebuiltAot) | Fail if no pre-built AOT artifact is found. |
/// | [`BuildIfMissingRelease`](BvpSciGeneratedBackendMode::BuildIfMissingRelease) | Build C AOT (default C backend) if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseRust`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust) | Build Rust AOT if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseGcc`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc) | Build C AOT (GCC) if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseTcc`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc) | Build C AOT (TCC) if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseZig`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig) | Build Zig AOT if missing, release profile. |
/// | [`AtomViewForRepeatedSolves`](BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves) | Build once, reuse across repeated solves. |
///
/// # Future variants
///
/// This enum is `#[non_exhaustive]` — adding new backends (e.g. dense AOT,
/// GPU offload) is a non-breaking change.
#[non_exhaustive]
pub enum BvpSciGeneratedBackendMode {
    /// Pure interpreted lambdify — no AOT compilation.
    /// Uses the ExprLegacySmartSparseLambdify workflow.
    #[default]
    LambdifyOnly,
    /// Fail with an error if no pre-built AOT artifact is found.
    /// Useful in CI/deployment where the artifact must already exist.
    RequirePrebuiltAot,
    /// Build C AOT (default C backend) if missing, release profile.
    BuildIfMissingRelease,
    /// Build Rust AOT if missing, release profile.
    AtomViewBuildIfMissingReleaseRust,
    /// Build C AOT (GCC) if missing, release profile.
    AtomViewBuildIfMissingReleaseGcc,
    /// Build C AOT (TCC) if missing, release profile.
    AtomViewBuildIfMissingReleaseTcc,
    /// Build Zig AOT if missing, release profile.
    AtomViewBuildIfMissingReleaseZig,
    /// Build once, reuse across repeated solves.
    /// The artifact is cached and not rebuilt on subsequent solver calls.
    AtomViewForRepeatedSolves,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpSciGeneratedMatrixBackend {
    #[default]
    Sparse,
    Banded,
}

pub(crate) const BVP_SCI_TEST_ARTIFACT_REV: &str = "rev2-bounds";

pub(crate) fn generated_test_artifact_dir(label: &str) -> String {
    format!("target/generated-bvp-sci-tests/{BVP_SCI_TEST_ARTIFACT_REV}/{label}")
}

#[derive(Clone, Debug)]
pub struct BvpSciGeneratedBackendConfig {
    pub mode: BvpSciGeneratedBackendMode,
    pub matrix_backend: BvpSciGeneratedMatrixBackend,
    pub resolver: Option<AotResolver>,
    pub output_parent_dir: Option<PathBuf>,
    pub residual_chunking_strategy: ResidualChunkingStrategy,
    pub sparse_jacobian_chunking_strategy: SparseChunkingStrategy,
    pub banded_jacobian_chunking_strategy: BandedChunkingStrategy,
}

impl BvpSciGeneratedBackendConfig {
    pub fn default_lambdify() -> Self {
        Self::default()
    }

    pub fn workflow(&self) -> super::BVP_sci_symb::BvpSciWorkflow {
        match self.mode {
            BvpSciGeneratedBackendMode::LambdifyOnly => {
                super::BVP_sci_symb::BvpSciWorkflow::ExprLegacySmartSparseLambdify
            }
            BvpSciGeneratedBackendMode::RequirePrebuiltAot
            | BvpSciGeneratedBackendMode::BuildIfMissingRelease
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig
            | BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves => match self.matrix_backend {
                BvpSciGeneratedMatrixBackend::Sparse => {
                    super::BVP_sci_symb::BvpSciWorkflow::AtomViewAotSparse
                }
                BvpSciGeneratedMatrixBackend::Banded => {
                    super::BVP_sci_symb::BvpSciWorkflow::AtomViewAotBanded
                }
            },
        }
    }

    pub fn from_mode(mode: BvpSciGeneratedBackendMode) -> Self {
        Self {
            mode,
            matrix_backend: BvpSciGeneratedMatrixBackend::Sparse,
            resolver: None,
            output_parent_dir: None,
            residual_chunking_strategy: ResidualChunkingStrategy::Whole,
            sparse_jacobian_chunking_strategy: SparseChunkingStrategy::Whole,
            banded_jacobian_chunking_strategy: BandedChunkingStrategy::Whole,
        }
    }

    pub fn with_matrix_backend(mut self, matrix_backend: BvpSciGeneratedMatrixBackend) -> Self {
        self.matrix_backend = matrix_backend;
        self
    }

    pub fn with_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    pub fn with_residual_chunking_strategy(
        mut self,
        residual_chunking_strategy: ResidualChunkingStrategy,
    ) -> Self {
        self.residual_chunking_strategy = residual_chunking_strategy;
        self
    }

    pub fn with_sparse_jacobian_chunking_strategy(
        mut self,
        sparse_jacobian_chunking_strategy: SparseChunkingStrategy,
    ) -> Self {
        self.sparse_jacobian_chunking_strategy = sparse_jacobian_chunking_strategy;
        self
    }

    pub fn with_banded_jacobian_chunking_strategy(
        mut self,
        banded_jacobian_chunking_strategy: BandedChunkingStrategy,
    ) -> Self {
        self.banded_jacobian_chunking_strategy = banded_jacobian_chunking_strategy;
        self
    }

    pub fn sparse_atomview_require_prebuilt() -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::RequirePrebuiltAot)
    }

    pub fn sparse_atomview_build_if_missing_release(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::BuildIfMissingRelease)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_build_if_missing_release_gcc(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_build_if_missing_release_rust(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_build_if_missing_release_tcc(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn banded_atomview_build_if_missing_release_tcc(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::sparse_atomview_build_if_missing_release_tcc(output_parent_dir)
            .with_matrix_backend(BvpSciGeneratedMatrixBackend::Banded)
    }

    pub fn banded_atomview_build_if_missing_release_gcc(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::sparse_atomview_build_if_missing_release_gcc(output_parent_dir)
            .with_matrix_backend(BvpSciGeneratedMatrixBackend::Banded)
    }

    pub fn banded_atomview_build_if_missing_release_rust(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::sparse_atomview_build_if_missing_release_rust(output_parent_dir)
            .with_matrix_backend(BvpSciGeneratedMatrixBackend::Banded)
    }

    pub fn banded_atomview_build_if_missing_release_zig(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::sparse_atomview_build_if_missing_release_zig(output_parent_dir)
            .with_matrix_backend(BvpSciGeneratedMatrixBackend::Banded)
    }

    pub fn sparse_atomview_build_if_missing_release_zig(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_for_repeated_solves(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn with_output_parent_dir(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.output_parent_dir = Some(output_parent_dir.into());
        self
    }

    /// Explicitly removes every registered generated artifact owned by the current resolver
    /// snapshot.
    ///
    /// This is a conservative lifecycle operation for story/debug/cold-build workflows.
    /// It does not unload dynamic libraries or unregister live callbacks.
    pub fn cleanup_registered_aot_artifacts(&mut self) -> std::io::Result<usize> {
        let Some(resolver) = self.resolver.as_mut() else {
            return Ok(0);
        };

        let problem_keys = resolver.registry().problem_keys();
        let mut removed = 0;
        for problem_key in problem_keys {
            if resolver.cleanup_artifact_by_problem_key(&problem_key)? {
                removed += 1;
            }
        }
        Ok(removed)
    }
}

impl Default for BvpSciGeneratedBackendConfig {
    fn default() -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::LambdifyOnly)
    }
}

#[derive(Clone)]
pub(crate) struct PreparedBvpSciSparseProblem {
    residual_eval: Arc<ODEFunction>,
    jacobian_eval: Option<Arc<ODEJacobian>>,
    boundary_conditions: std::collections::HashMap<String, Vec<(usize, f64)>>,
    values: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct PreparedBvpSciBandedProblem {
    pub(crate) residual_eval: Arc<ODEFunction>,
    pub(crate) banded_jacobian_eval: Option<Arc<ODEBandedJacobian>>,
    pub(crate) sparse_compat_jacobian_eval: Option<Arc<ODEJacobian>>,
    pub(crate) boundary_conditions: std::collections::HashMap<String, Vec<(usize, f64)>>,
    pub(crate) values: Vec<String>,
}

#[derive(Clone, Debug)]
struct LinkedRuntimeStageDiagnostics {
    actual_jobs: usize,
    chunk_count: usize,
    work_per_job: usize,
    work_per_chunk: usize,
    fallback_reason: &'static str,
    mesh_parallel: bool,
}

impl LinkedRuntimeStageDiagnostics {
    fn sequential(total_work: usize, chunk_count: usize, reason: &'static str) -> Self {
        Self {
            actual_jobs: 1,
            chunk_count,
            work_per_job: total_work,
            work_per_chunk: work_per_group(total_work, chunk_count.max(1)),
            fallback_reason: reason,
            mesh_parallel: false,
        }
    }

    fn mesh_parallel(total_work: usize, chunk_count: usize, mesh_points: usize) -> Self {
        let actual_jobs = rayon::current_num_threads().min(mesh_points.max(1)).max(1);
        Self {
            actual_jobs,
            chunk_count,
            work_per_job: work_per_group(total_work, actual_jobs),
            work_per_chunk: work_per_group(total_work, chunk_count.max(1)),
            fallback_reason: "none",
            mesh_parallel: true,
        }
    }
}

fn work_per_group(total_work: usize, groups: usize) -> usize {
    if groups == 0 { 0 } else { total_work / groups }
}

fn banded_assembly_to_sparse(matrix: &BandedAssembly) -> faer_mat {
    let mut triplets = Vec::new();
    for offset in matrix.offsets() {
        let diagonal = matrix
            .diag(offset)
            .expect("configured banded diagonal must exist");
        for (position, value) in diagonal.iter().copied().enumerate() {
            if value != 0.0 {
                let (row, col) = matrix
                    .diag_pos_to_ij(offset, position)
                    .expect("banded diagonal position must map to matrix coordinates");
                triplets.push(Triplet::new(row, col, value));
            }
        }
    }
    SparseColMat::try_new_from_triplets(matrix.n(), matrix.n(), &triplets)
        .expect("native banded Jacobian must convert to a sparse compatibility matrix")
}

/// AtomView-oriented symbolic payload for sparse AOT generation.
#[derive(Clone, Debug)]
pub(crate) struct AtomViewPreparedBvpSciProblem {
    pub(crate) equations: Vec<Expr>,
    pub(crate) symbolic_jacobian_sparse: Vec<(usize, usize, Expr)>,
    pub(crate) symbolic_param_jacobian_sparse: Option<Vec<(usize, usize, Expr)>>,
    pub(crate) time_arg: String,
    pub(crate) variables: Vec<String>,
    pub(crate) equation_parameters: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
struct BvpSciPointwiseSparsePrepared {
    residuals: Vec<Expr>,
    variable_names: Vec<String>,
    parameter_names: Vec<String>,
    time_arg: String,
    residual_fn_name: String,
    jacobian_fn_name: String,
    sparse_entries: Vec<(usize, usize, Expr)>,
    shape: (usize, usize),
}

impl BvpSciPointwiseSparsePrepared {
    fn flattened_input_names(&self) -> Vec<&str> {
        let mut names =
            Vec::with_capacity(1 + self.parameter_names.len() + self.variable_names.len());
        names.push(self.time_arg.as_str());
        names.extend(self.variable_names.iter().map(|name| name.as_str()));
        names.extend(self.parameter_names.iter().map(|name| name.as_str()));
        names
    }

    fn borrowed_sparse_entries(&self) -> Vec<SparseExprEntry<'_>> {
        self.sparse_entries
            .iter()
            .map(|(row, col, expr)| SparseExprEntry {
                row: *row,
                col: *col,
                expr,
            })
            .collect()
    }

    fn borrowed_banded_entries(&self) -> Vec<BandedExprEntry<'_>> {
        let mut entries = self
            .sparse_entries
            .iter()
            .map(|(row, col, expr)| {
                let diag_offset = *col as isize - *row as isize;
                BandedExprEntry {
                    row: *row,
                    col: *col,
                    diag_offset,
                    diag_position: if diag_offset >= 0 { *row } else { *col },
                    expr,
                }
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| (entry.diag_offset, entry.diag_position));
        entries
    }

    fn bandwidth(&self) -> (usize, usize) {
        self.sparse_entries
            .iter()
            .fold((0usize, 0usize), |(kl, ku), (row, col, _)| {
                if row >= col {
                    (kl.max(row - col), ku)
                } else {
                    (kl, ku.max(col - row))
                }
            })
    }

    fn residual_runtime_plan(&self, strategy: ResidualChunkingStrategy) -> ResidualRuntimePlan<'_> {
        let input_names = self.flattened_input_names();
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
                let outputs = residuals
                    .iter()
                    .map(|expr| PlannedOutput {
                        expr,
                        coordinate: None,
                    })
                    .collect::<Vec<_>>();
                let fn_name = if chunk_index == 0 && residuals.len() == self.residuals.len() {
                    self.residual_fn_name.clone()
                } else {
                    format!("{}_chunk_{}", self.residual_fn_name, chunk_index)
                };
                ResidualChunkPlan {
                    fn_name: fn_name.clone(),
                    output_offset,
                    residuals,
                    plan: CodegenTaskPlan {
                        fn_name: Cow::Owned(fn_name),
                        kind: CodegenTaskKind::IvpResidual,
                        input_names: input_names.clone(),
                        outputs,
                        layout: CodegenOutputLayout::Vector {
                            len: residuals.len(),
                        },
                    },
                }
            })
            .collect::<Vec<_>>();

        ResidualRuntimePlan {
            fn_name: self.residual_fn_name.as_str(),
            output_len: self.residuals.len(),
            input_names,
            chunks,
        }
    }

    fn sparse_runtime_plan(
        &self,
        strategy: SparseChunkingStrategy,
    ) -> SparseJacobianRuntimePlan<'_> {
        let all_entries = self.borrowed_sparse_entries();
        let input_names = self.flattened_input_names();
        let chunk_groups: Vec<(usize, Vec<SparseExprEntry<'_>>)> = match strategy {
            SparseChunkingStrategy::Whole => vec![(0, all_entries.clone())],
            SparseChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                let max_entries_per_chunk = all_entries.len().max(1).div_ceil(target_chunks).max(1);
                all_entries
                    .chunks(max_entries_per_chunk)
                    .enumerate()
                    .map(|(chunk_index, entries)| {
                        (chunk_index * max_entries_per_chunk, entries.to_vec())
                    })
                    .collect()
            }
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk,
            } => {
                assert!(
                    max_entries_per_chunk > 0,
                    "max_entries_per_chunk must be positive"
                );
                all_entries
                    .chunks(max_entries_per_chunk)
                    .enumerate()
                    .map(|(chunk_index, entries)| {
                        (chunk_index * max_entries_per_chunk, entries.to_vec())
                    })
                    .collect()
            }
            SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
                assert!(rows_per_chunk > 0, "rows_per_chunk must be positive");
                let mut bucket_entries: Vec<Vec<(usize, SparseExprEntry<'_>)>> = Vec::new();
                let mut bucket_offsets: Vec<usize> = Vec::new();
                for (entry_index, entry) in all_entries.iter().copied().enumerate() {
                    let bucket = entry.row / rows_per_chunk;
                    if bucket >= bucket_entries.len() {
                        bucket_entries.resize_with(bucket + 1, Vec::new);
                        bucket_offsets.resize(bucket + 1, 0);
                    }
                    if bucket_entries[bucket].is_empty() {
                        bucket_offsets[bucket] = entry_index;
                    }
                    bucket_entries[bucket].push((entry_index, entry));
                }

                let non_empty_buckets = bucket_entries
                    .iter()
                    .filter(|entries| !entries.is_empty())
                    .collect::<Vec<_>>();
                let all_chunks_are_contiguous = non_empty_buckets.iter().all(|entries| {
                    let start = entries.first().map(|(index, _)| *index).unwrap_or(0);
                    entries
                        .iter()
                        .enumerate()
                        .all(|(local_index, (entry_index, _))| *entry_index == start + local_index)
                });

                if !all_chunks_are_contiguous {
                    let target_chunks = self.shape.0.max(1).div_ceil(rows_per_chunk).max(1);
                    let max_entries_per_chunk =
                        all_entries.len().max(1).div_ceil(target_chunks).max(1);
                    all_entries
                        .chunks(max_entries_per_chunk)
                        .enumerate()
                        .map(|(chunk_index, entries)| {
                            (chunk_index * max_entries_per_chunk, entries.to_vec())
                        })
                        .collect()
                } else {
                    bucket_entries
                        .into_iter()
                        .enumerate()
                        .filter(|(_, entries)| !entries.is_empty())
                        .map(|(chunk_index, entries)| {
                            (
                                bucket_offsets[chunk_index],
                                entries.into_iter().map(|(_, entry)| entry).collect(),
                            )
                        })
                        .collect()
                }
            }
        };
        let chunks = chunk_groups
            .into_iter()
            .enumerate()
            .map(|(chunk_index, (value_offset, entries))| {
                let fn_name = if chunk_index == 0 && entries.len() == all_entries.len() {
                    self.jacobian_fn_name.clone()
                } else {
                    format!("{}_chunk_{}", self.jacobian_fn_name, chunk_index)
                };
                let outputs = entries
                    .iter()
                    .map(|entry| PlannedOutput {
                        expr: entry.expr,
                        coordinate: Some((entry.row, entry.col)),
                    })
                    .collect::<Vec<_>>();
                let mut plan = CodegenTaskPlan {
                    fn_name: Cow::Owned(fn_name.clone()),
                    kind: CodegenTaskKind::SparseJacobianValues,
                    input_names: input_names.clone(),
                    outputs,
                    layout: CodegenOutputLayout::SparseValues {
                        rows: self.shape.0,
                        cols: self.shape.1,
                        nnz: entries.len(),
                    },
                };
                if chunk_index == 0 && entries.len() == all_entries.len() {
                    plan.fn_name = Cow::Owned(self.jacobian_fn_name.clone());
                }
                SparseJacobianValuesChunkPlan {
                    fn_name,
                    value_offset,
                    entries,
                    plan,
                }
            })
            .collect::<Vec<_>>();

        SparseJacobianRuntimePlan {
            fn_name: self.jacobian_fn_name.as_str(),
            input_names,
            structure: SparseJacobianStructure {
                rows: self.shape.0,
                cols: self.shape.1,
                row_indices: all_entries.iter().map(|entry| entry.row).collect(),
                col_indices: all_entries.iter().map(|entry| entry.col).collect(),
            },
            chunks,
        }
    }

    fn as_prepared_problem(
        &self,
        residual_strategy: ResidualChunkingStrategy,
        strategy: SparseChunkingStrategy,
    ) -> PreparedSparseProblem<'_> {
        PreparedSparseProblem::new(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            self.residual_runtime_plan(residual_strategy),
            self.sparse_runtime_plan(strategy),
        )
    }

    fn banded_runtime_plan(
        &self,
        strategy: BandedChunkingStrategy,
    ) -> BandedJacobianRuntimePlan<'_> {
        let all_entries = self.borrowed_banded_entries();
        let (kl, ku) = self.bandwidth();
        let input_names = self.flattened_input_names();
        let total_diagonals = kl + ku + 1;
        let diagonals_per_chunk = match strategy {
            BandedChunkingStrategy::Whole => total_diagonals.max(1),
            BandedChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                total_diagonals.max(1).div_ceil(target_chunks).max(1)
            }
            BandedChunkingStrategy::ByDiagonalCount {
                diagonals_per_chunk,
            } => {
                assert!(
                    diagonals_per_chunk > 0,
                    "diagonals_per_chunk must be positive"
                );
                diagonals_per_chunk
            }
        };

        let mut grouped_entries: Vec<Vec<(usize, BandedExprEntry<'_>)>> = Vec::new();
        for (entry_index, entry) in all_entries.iter().copied().enumerate() {
            let bucket = ((entry.diag_offset + kl as isize) as usize) / diagonals_per_chunk;
            if bucket >= grouped_entries.len() {
                grouped_entries.resize_with(bucket + 1, Vec::new);
            }
            grouped_entries[bucket].push((entry_index, entry));
        }

        let chunks = grouped_entries
            .into_iter()
            .filter(|entries| !entries.is_empty())
            .enumerate()
            .map(|(chunk_index, indexed_entries)| {
                let value_offset = indexed_entries[0].0;
                let entries = indexed_entries
                    .into_iter()
                    .map(|(_, entry)| entry)
                    .collect::<Vec<_>>();
                let fn_name = if chunk_index == 0 && entries.len() == all_entries.len() {
                    self.jacobian_fn_name.clone()
                } else {
                    format!("{}_chunk_{}", self.jacobian_fn_name, chunk_index)
                };
                let outputs = entries
                    .iter()
                    .map(|entry| PlannedOutput {
                        expr: entry.expr,
                        coordinate: Some((entry.row, entry.col)),
                    })
                    .collect::<Vec<_>>();
                let plan = CodegenTaskPlan {
                    fn_name: Cow::Owned(fn_name.clone()),
                    kind: CodegenTaskKind::BandedJacobianValues,
                    input_names: input_names.clone(),
                    outputs,
                    layout: CodegenOutputLayout::SparseValues {
                        rows: self.shape.0,
                        cols: self.shape.1,
                        nnz: entries.len(),
                    },
                };
                BandedJacobianValuesChunkPlan {
                    fn_name,
                    value_offset,
                    entries,
                    plan,
                }
            })
            .collect::<Vec<_>>();

        BandedJacobianRuntimePlan {
            fn_name: self.jacobian_fn_name.as_str(),
            input_names,
            structure: BandedJacobianStructure {
                rows: self.shape.0,
                cols: self.shape.1,
                kl,
                ku,
                diagonal_offsets: all_entries.iter().map(|entry| entry.diag_offset).collect(),
                diagonal_positions: all_entries
                    .iter()
                    .map(|entry| entry.diag_position)
                    .collect(),
            },
            chunks,
        }
    }

    fn as_prepared_banded_problem(
        &self,
        residual_strategy: ResidualChunkingStrategy,
        strategy: BandedChunkingStrategy,
    ) -> PreparedBandedProblem<'_> {
        PreparedBandedProblem::new(
            BackendKind::Aot,
            MatrixBackend::Banded,
            self.residual_runtime_plan(residual_strategy),
            self.banded_runtime_plan(strategy),
        )
    }
}

/// Builds a sparse matrix family evaluator from sparse symbolic entries.
fn compile_bvp_sci_sparse_matrix_family_eval(
    sparse_entries: &[(usize, usize, Expr)],
    shape: (usize, usize),
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
) -> Arc<dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> Vec<faer_mat> + Send + Sync> {
    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |params| params.len()),
    );
    names.push(time_arg.to_string());
    names.extend(variables.iter().cloned());
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().cloned());
    }
    let name_refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();
    let compiled = sparse_entries
        .iter()
        .map(|(_, _, expr)| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
        .collect::<Vec<_>>();
    let structure = SparseJacobianStructure {
        rows: shape.0,
        cols: shape.1,
        row_indices: sparse_entries.iter().map(|(row, _, _)| *row).collect(),
        col_indices: sparse_entries.iter().map(|(_, col, _)| *col).collect(),
    };
    let variable_names = variables.to_vec();

    Arc::new(
        move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> Vec<faer_mat> {
            (0..y.ncols())
                .map(|point_index| {
                    let args = BVPwrap::flattened_args_for_mesh_point(
                        x,
                        y,
                        p,
                        point_index,
                        &variable_names,
                        &bounds,
                    );
                    let sparse_values = compiled.iter().map(|func| func(&args)).collect::<Vec<_>>();
                    structure.assemble_sparse_col_mat(sparse_values.as_slice())
                })
                .collect()
        },
    )
}

impl BVPwrap {
    /// Generates the symbolic payload needed by the AtomView/AOT pipeline.
    pub(crate) fn prepare_atomview_pointwise_problem(
        &self,
    ) -> Result<Arc<AtomViewPreparedBvpSciProblem>, BvpSciBackendError> {
        if self.param.is_empty() {
            if self
                .param_values
                .as_ref()
                .is_some_and(|values| !values.is_empty())
            {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci AtomView prepare received {} parameter values but no parameter names",
                        self.param_values.as_ref().map_or(0, |values| values.len())
                    ),
                });
            }
        } else {
            let values = self.param_values.as_ref().ok_or_else(|| {
                BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci AtomView prepare expected {} parameter values but none were provided",
                        self.param.len()
                    ),
                }
            })?;
            if values.len() != self.param.len() {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci AtomView prepare expected {} parameter values, got {}",
                        self.param.len(),
                        values.len()
                    ),
                });
            }
        }

        let mut legacy = Jacobian_sci_faer::new();
        legacy.from_vectors(self.eq_system.clone(), self.values.clone());
        legacy.calc_jacobian_parallel_smart();

        Ok(Arc::new(AtomViewPreparedBvpSciProblem {
            equations: self.eq_system.clone(),
            symbolic_jacobian_sparse: legacy.symbolic_jacobian_sparse,
            symbolic_param_jacobian_sparse: if self.param.is_empty() {
                None
            } else {
                Some(
                    self.eq_system
                        .iter()
                        .enumerate()
                        .flat_map(|(row, expr)| {
                            self.param
                                .iter()
                                .enumerate()
                                .filter_map(move |(col, parameter)| {
                                    let partial = expr.diff(parameter).simplify();
                                    if partial.is_zero() {
                                        None
                                    } else {
                                        Some((row, col, partial))
                                    }
                                })
                        })
                        .collect(),
                )
            },
            time_arg: self.arg.clone(),
            variables: self.values.clone(),
            equation_parameters: if self.param.is_empty() {
                None
            } else {
                Some(self.param.clone())
            },
        }))
    }

    fn flattened_args_for_mesh_point(
        x: &faer_col,
        y: &faer_dense_mat,
        p: &faer_col,
        point_index: usize,
        values: &[String],
        bounds: &Option<HashMap<String, Vec<(usize, f64)>>>,
    ) -> Vec<f64> {
        let n = y.nrows();
        let mut args = Vec::with_capacity(1 + n + p.nrows());
        args.push(x[point_index]);
        let bounds_ref = bounds.as_ref();
        for row in 0..n {
            args.push(Jacobian_sci_faer::handle_bounds(
                row,
                *y.get(row, point_index),
                bounds_ref,
                values,
            ));
        }
        for param_index in 0..p.nrows() {
            args.push(p[param_index]);
        }
        args
    }

    pub(crate) fn wrap_prepared_sparse_problem(
        &self,
        prepared: PreparedBvpSciSparseProblem,
    ) -> (
        Option<Box<ODEJacobian>>,
        Box<ODEFunction>,
        Option<Box<BCFunction>>,
    ) {
        let residual_eval = Arc::clone(&prepared.residual_eval);
        let residual = Box::new(move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
            (residual_eval)(x, y, p)
        }) as Box<ODEFunction>;

        let jacobian = prepared.jacobian_eval.map(|jacobian_eval| {
            Box::new(move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| (jacobian_eval)(x, y, p))
                as Box<ODEJacobian>
        });

        let bc_func = Self::BC_closure_creater(prepared.boundary_conditions, prepared.values);
        (jacobian, residual, bc_func)
    }

    pub(crate) fn wrap_prepared_banded_problem(
        &self,
        prepared: PreparedBvpSciBandedProblem,
    ) -> (
        Option<Box<ODEJacobian>>,
        Box<ODEFunction>,
        Option<Box<BCFunction>>,
    ) {
        let residual_eval = Arc::clone(&prepared.residual_eval);
        let residual = Box::new(move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
            (residual_eval)(x, y, p)
        }) as Box<ODEFunction>;
        let jacobian = prepared.sparse_compat_jacobian_eval.map(|eval| {
            Box::new(move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| eval(x, y, p))
                as Box<ODEJacobian>
        });
        let bc_func = Self::BC_closure_creater(prepared.boundary_conditions, prepared.values);
        (jacobian, residual, bc_func)
    }

    fn pointwise_sparse_entries(
        pointwise: &AtomViewPreparedBvpSciProblem,
    ) -> Vec<SparseExprEntry<'_>> {
        pointwise
            .symbolic_jacobian_sparse
            .iter()
            .map(|(row, col, expr)| SparseExprEntry {
                row: *row,
                col: *col,
                expr,
            })
            .collect()
    }

    pub(crate) fn symbolic_sparse_structure(
        problem: &AtomViewPreparedBvpSciProblem,
    ) -> SparseJacobianStructure {
        let rows = problem.variables.len();
        let cols = problem.variables.len();
        let row_indices = problem
            .symbolic_jacobian_sparse
            .iter()
            .map(|(row, _, _)| *row)
            .collect();
        let col_indices = problem
            .symbolic_jacobian_sparse
            .iter()
            .map(|(_, col, _)| *col)
            .collect();

        SparseJacobianStructure {
            rows,
            cols,
            row_indices,
            col_indices,
        }
    }

    pub(crate) fn symbolic_banded_structure(
        problem: &AtomViewPreparedBvpSciProblem,
    ) -> BandedJacobianStructure {
        Self::pointwise_sparse_prepared_problem(problem)
            .banded_runtime_plan(BandedChunkingStrategy::Whole)
            .structure
    }

    fn pointwise_sparse_prepared_problem(
        pointwise: &AtomViewPreparedBvpSciProblem,
    ) -> BvpSciPointwiseSparsePrepared {
        let sparse_entries = Self::pointwise_sparse_entries(pointwise)
            .into_iter()
            .map(|entry| (entry.row, entry.col, entry.expr.clone()))
            .collect::<Vec<_>>();

        BvpSciPointwiseSparsePrepared {
            residuals: pointwise.equations.clone(),
            variable_names: pointwise.variables.clone(),
            parameter_names: pointwise.equation_parameters.clone().unwrap_or_default(),
            time_arg: pointwise.time_arg.clone(),
            residual_fn_name: "eval_bvp_sci_pointwise_residual".to_string(),
            jacobian_fn_name: "eval_bvp_sci_pointwise_sparse_values".to_string(),
            sparse_entries,
            shape: (pointwise.variables.len(), pointwise.variables.len()),
        }
    }

    fn generated_backend_names(
        problem_key: &str,
        config: &SymbolicIvpGeneratedBackendConfig,
        matrix_backend: BvpSciGeneratedMatrixBackend,
    ) -> (String, String) {
        let suffix = problem_key
            .chars()
            .take(16)
            .collect::<String>()
            .replace('-', "_");
        let matrix_label = match matrix_backend {
            BvpSciGeneratedMatrixBackend::Sparse => "sparse",
            BvpSciGeneratedMatrixBackend::Banded => "banded",
        };
        let crate_name = config
            .crate_name_override
            .clone()
            .unwrap_or_else(|| format!("generated_bvp_sci_{matrix_label}_{suffix}"));
        let module_name = config
            .module_name_override
            .clone()
            .unwrap_or_else(|| format!("generated_bvp_sci_{matrix_label}_module_{suffix}"));
        (crate_name, module_name)
    }

    fn build_preset(policy: SymbolicIvpAotBuildPolicy) -> Option<AotBuildPreset> {
        match policy {
            SymbolicIvpAotBuildPolicy::BuildIfMissing { profile }
            | SymbolicIvpAotBuildPolicy::RebuildAlways { profile } => match profile {
                AotBuildProfile::Debug => Some(AotBuildPreset::DevFastest),
                AotBuildProfile::Release => Some(AotBuildPreset::Production),
            },
            SymbolicIvpAotBuildPolicy::UseIfAvailable
            | SymbolicIvpAotBuildPolicy::RequirePrebuilt => None,
        }
    }

    fn build_policy_label(policy: SymbolicIvpAotBuildPolicy) -> &'static str {
        match policy {
            SymbolicIvpAotBuildPolicy::UseIfAvailable => "UseIfAvailable",
            SymbolicIvpAotBuildPolicy::RequirePrebuilt => "RequirePrebuilt",
            SymbolicIvpAotBuildPolicy::BuildIfMissing { .. } => "BuildIfMissing",
            SymbolicIvpAotBuildPolicy::RebuildAlways { .. } => "RebuildAlways",
        }
    }

    fn codegen_backend_label(backend: AotCodegenBackend) -> &'static str {
        match backend {
            AotCodegenBackend::Rust => "Rust",
            AotCodegenBackend::C => "C",
            AotCodegenBackend::Zig => "Zig",
        }
    }

    fn linked_runtime_diagnostics(
        residual: &LinkedRuntimeStageDiagnostics,
        jacobian: &LinkedRuntimeStageDiagnostics,
        jacobian_stage: &str,
    ) -> HashMap<String, String> {
        let mut diagnostics = HashMap::new();
        let any_parallel = residual.mesh_parallel || jacobian.mesh_parallel;
        diagnostics.insert(
            "aot.runtime.execution_policy".to_string(),
            if any_parallel {
                "MeshParallel"
            } else {
                "Sequential"
            }
            .to_string(),
        );
        diagnostics.insert(
            "aot.runtime.parallel_requested".to_string(),
            (residual.chunk_count > 1 || jacobian.chunk_count > 1).to_string(),
        );

        for (prefix, stage) in [
            ("aot.runtime.residual", residual),
            (jacobian_stage, jacobian),
        ] {
            diagnostics.insert(
                format!("{prefix}.actual_jobs"),
                stage.actual_jobs.to_string(),
            );
            diagnostics.insert(
                format!("{prefix}.chunk_count"),
                stage.chunk_count.to_string(),
            );
            diagnostics.insert(
                format!("{prefix}.work_per_job"),
                stage.work_per_job.to_string(),
            );
            diagnostics.insert(
                format!("{prefix}.work_per_chunk"),
                stage.work_per_chunk.to_string(),
            );
            diagnostics.insert(
                format!("{prefix}.fallback_reason"),
                stage.fallback_reason.to_string(),
            );
            diagnostics.insert(
                format!("{prefix}.mesh_parallel"),
                stage.mesh_parallel.to_string(),
            );
        }
        diagnostics
    }

    fn eval_linked_residual_chunks(
        chunks: &[LinkedResidualChunk],
        args: &[f64],
        residual_len: usize,
    ) -> Vec<f64> {
        let mut point_out = vec![0.0; residual_len];
        for chunk in chunks {
            let start = chunk.output_offset;
            let end = start + chunk.output_len;
            (chunk.eval)(args, &mut point_out[start..end]);
        }
        point_out
    }

    fn eval_linked_sparse_jacobian_chunks(
        chunks: &[LinkedSparseJacobianChunk],
        args: &[f64],
        nnz: usize,
    ) -> Vec<f64> {
        let mut values = vec![0.0; nnz];
        for chunk in chunks {
            let start = chunk.value_offset;
            let end = start + chunk.value_len;
            (chunk.eval)(args, &mut values[start..end]);
        }
        values
    }

    fn linked_sparse_residual_eval(
        &self,
        linked_problem_key: &str,
    ) -> Result<(Arc<ODEFunction>, LinkedRuntimeStageDiagnostics), BvpSciBackendError> {
        let linked = resolve_linked_sparse_backend(linked_problem_key).ok_or_else(|| {
            BvpSciBackendError::GeneratedBackendFailure {
                message: "compiled sparse BVP_sci artifact exists but no linked sparse runtime is registered"
                    .to_string(),
            }
        })?;
        let values = self.values.clone();
        let bounds = self.Bounds.clone();
        let residual_chunks = linked.residual_chunks.clone();
        let mesh_points = self.x_mesh_col.nrows();
        let diagnostics = if residual_chunks.len() > 1 {
            LinkedRuntimeStageDiagnostics::mesh_parallel(
                linked.residual_len.saturating_mul(mesh_points),
                residual_chunks.len(),
                mesh_points,
            )
        } else {
            LinkedRuntimeStageDiagnostics::sequential(
                linked.residual_len.saturating_mul(mesh_points),
                residual_chunks.len().max(1),
                "single_chunk",
            )
        };
        Ok((
            Arc::new(
                move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> faer_dense_mat {
                    let args_begin = Instant::now();
                    let point_args = (0..y.ncols())
                        .map(|point_index| {
                            BVPwrap::flattened_args_for_mesh_point(
                                x,
                                y,
                                p,
                                point_index,
                                &values,
                                &bounds,
                            )
                        })
                        .collect::<Vec<_>>();
                    record_callback_stage_time("Callback Residual Args", args_begin.elapsed());

                    let values_begin = Instant::now();
                    let point_outputs = if residual_chunks.len() > 1 {
                        point_args
                            .par_iter()
                            .map(|args| {
                                BVPwrap::eval_linked_residual_chunks(
                                    residual_chunks.as_slice(),
                                    args.as_slice(),
                                    linked.residual_len,
                                )
                            })
                            .collect::<Vec<_>>()
                    } else {
                        point_args
                            .iter()
                            .map(|args| {
                                let mut point_out = vec![0.0; linked.residual_len];
                                (linked.residual_eval)(args.as_slice(), &mut point_out);
                                point_out
                            })
                            .collect::<Vec<_>>()
                    };
                    record_callback_stage_time("Callback Residual Values", values_begin.elapsed());

                    let assembly_begin = Instant::now();
                    let mut out = faer_dense_mat::zeros(linked.residual_len, y.ncols());
                    for (point_index, point_out) in point_outputs.iter().enumerate() {
                        for (row, value) in point_out.iter().enumerate() {
                            *out.get_mut(row, point_index) = *value;
                        }
                    }
                    record_callback_stage_time(
                        "Callback Residual Matrix Assembly",
                        assembly_begin.elapsed(),
                    );
                    out
                },
            ),
            diagnostics,
        ))
    }

    fn linked_sparse_jacobian_eval(
        &self,
        linked_problem_key: &str,
        sparse_structure: SparseJacobianStructure,
        param_jacobian_eval: Option<
            Arc<dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> Vec<faer_mat> + Send + Sync>,
        >,
    ) -> Result<(Arc<ODEJacobian>, LinkedRuntimeStageDiagnostics), BvpSciBackendError> {
        let linked = resolve_linked_sparse_backend(linked_problem_key).ok_or_else(|| {
            BvpSciBackendError::GeneratedBackendFailure {
                message: "compiled sparse BVP_sci artifact exists but no linked sparse runtime is registered"
                    .to_string(),
            }
        })?;
        let values = self.values.clone();
        let bounds = self.Bounds.clone();
        let jacobian_chunks = linked.jacobian_value_chunks.clone();
        let mesh_points = self.x_mesh_col.nrows();
        let diagnostics = if jacobian_chunks.len() > 1 {
            LinkedRuntimeStageDiagnostics::mesh_parallel(
                linked.nnz.saturating_mul(mesh_points),
                jacobian_chunks.len(),
                mesh_points,
            )
        } else {
            LinkedRuntimeStageDiagnostics::sequential(
                linked.nnz.saturating_mul(mesh_points),
                jacobian_chunks.len().max(1),
                "single_chunk",
            )
        };
        Ok((
            Arc::new(
                move |x: &faer_col,
                      y: &faer_dense_mat,
                      p: &faer_col|
                      -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
                    let args_begin = Instant::now();
                    let point_args = (0..y.ncols())
                        .map(|point_index| {
                            BVPwrap::flattened_args_for_mesh_point(
                                x,
                                y,
                                p,
                                point_index,
                                &values,
                                &bounds,
                            )
                        })
                        .collect::<Vec<_>>();
                    record_callback_stage_time("Callback Jacobian Args", args_begin.elapsed());

                    let values_begin = Instant::now();
                    let sparse_values_by_point = if jacobian_chunks.len() > 1 {
                        point_args
                            .par_iter()
                            .map(|args| {
                                BVPwrap::eval_linked_sparse_jacobian_chunks(
                                    jacobian_chunks.as_slice(),
                                    args.as_slice(),
                                    linked.nnz,
                                )
                            })
                            .collect::<Vec<_>>()
                    } else {
                        point_args
                            .iter()
                            .map(|args| {
                                let mut sparse_values = vec![0.0; linked.nnz];
                                (linked.jacobian_values_eval)(args.as_slice(), &mut sparse_values);
                                sparse_values
                            })
                            .collect::<Vec<_>>()
                    };
                    record_callback_stage_time("Callback Jacobian Values", values_begin.elapsed());

                    let assembly_begin = Instant::now();
                    let jac_y = sparse_values_by_point
                        .into_iter()
                        .map(|sparse_values| {
                            sparse_structure.assemble_sparse_col_mat(sparse_values.as_slice())
                        })
                        .collect::<Vec<_>>();
                    record_callback_stage_time(
                        "Callback Jacobian Matrix Assembly",
                        assembly_begin.elapsed(),
                    );

                    let param_begin = Instant::now();
                    let jac_p = param_jacobian_eval
                        .as_ref()
                        .map(|param_eval| (param_eval)(x, y, p));
                    if param_jacobian_eval.is_some() {
                        record_callback_stage_time(
                            "Callback Parameter Jacobian Values",
                            param_begin.elapsed(),
                        );
                    }
                    (jac_y, jac_p)
                },
            ),
            diagnostics,
        ))
    }

    fn linked_banded_jacobian_eval(
        &self,
        linked_problem_key: &str,
        banded_structure: BandedJacobianStructure,
        param_jacobian_eval: Option<
            Arc<dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> Vec<faer_mat> + Send + Sync>,
        >,
    ) -> Result<
        (
            Arc<ODEBandedJacobian>,
            Arc<ODEJacobian>,
            LinkedRuntimeStageDiagnostics,
        ),
        BvpSciBackendError,
    > {
        let linked = resolve_linked_sparse_backend(linked_problem_key).ok_or_else(|| {
            BvpSciBackendError::GeneratedBackendFailure {
                message:
                    "compiled banded BVP_sci artifact exists but no linked runtime is registered"
                        .to_string(),
            }
        })?;
        let values = self.values.clone();
        let bounds = self.Bounds.clone();
        let jacobian_chunks = linked.jacobian_value_chunks.clone();
        let mesh_points = self.x_mesh_col.nrows();
        let diagnostics = if jacobian_chunks.len() > 1 {
            LinkedRuntimeStageDiagnostics::mesh_parallel(
                linked.nnz.saturating_mul(mesh_points),
                jacobian_chunks.len(),
                mesh_points,
            )
        } else {
            LinkedRuntimeStageDiagnostics::sequential(
                linked.nnz.saturating_mul(mesh_points),
                jacobian_chunks.len().max(1),
                "single_chunk",
            )
        };

        let eval_values = move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
            let args_begin = Instant::now();
            let point_args = (0..y.ncols())
                .map(|point_index| {
                    BVPwrap::flattened_args_for_mesh_point(x, y, p, point_index, &values, &bounds)
                })
                .collect::<Vec<_>>();
            record_callback_stage_time("Callback Jacobian Args", args_begin.elapsed());

            let values_begin = Instant::now();
            let point_values = if jacobian_chunks.len() > 1 {
                point_args
                    .par_iter()
                    .map(|args| {
                        BVPwrap::eval_linked_sparse_jacobian_chunks(
                            jacobian_chunks.as_slice(),
                            args.as_slice(),
                            linked.nnz,
                        )
                    })
                    .collect::<Vec<_>>()
            } else {
                point_args
                    .iter()
                    .map(|args| {
                        let mut out = vec![0.0; linked.nnz];
                        (linked.jacobian_values_eval)(args.as_slice(), &mut out);
                        out
                    })
                    .collect::<Vec<_>>()
            };
            record_callback_stage_time("Callback Jacobian Values", values_begin.elapsed());
            point_values
        };
        let eval_values = Arc::new(eval_values);

        let banded_structure_native = banded_structure.clone();
        let param_native = param_jacobian_eval.clone();
        let eval_native = Arc::clone(&eval_values);
        let banded = Arc::new(
            move |x: &faer_col,
                  y: &faer_dense_mat,
                  p: &faer_col|
                  -> (Vec<BandedAssembly>, Option<Vec<faer_mat>>) {
                let assembly_begin = Instant::now();
                let matrices = eval_native(x, y, p)
                    .into_iter()
                    .map(|values| {
                        banded_structure_native.assemble_banded_assembly(values.as_slice())
                    })
                    .collect::<Vec<_>>();
                record_callback_stage_time(
                    "Callback Jacobian Banded Assembly",
                    assembly_begin.elapsed(),
                );
                let params = param_native.as_ref().map(|eval| eval(x, y, p));
                (matrices, params)
            },
        ) as Arc<ODEBandedJacobian>;

        let param_sparse = param_jacobian_eval;
        let eval_sparse = Arc::clone(&eval_values);
        let sparse_structure = banded_structure;
        let sparse = Arc::new(
            move |x: &faer_col,
                  y: &faer_dense_mat,
                  p: &faer_col|
                  -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
                let matrices = eval_sparse(x, y, p)
                    .into_iter()
                    .map(|values| {
                        let banded = sparse_structure.assemble_banded_assembly(values.as_slice());
                        banded_assembly_to_sparse(&banded)
                    })
                    .collect::<Vec<_>>();
                let params = param_sparse.as_ref().map(|eval| eval(x, y, p));
                (matrices, params)
            },
        ) as Arc<ODEJacobian>;

        Ok((banded, sparse, diagnostics))
    }

    /// Ensures a compiled sparse backend is linked or builds one if requested.
    pub(crate) fn ensure_sparse_generated_runtime(
        &mut self,
        pointwise: &AtomViewPreparedBvpSciProblem,
        config: &SymbolicIvpGeneratedBackendConfig,
    ) -> Result<String, BvpSciBackendError> {
        let prepared_pointwise = Self::pointwise_sparse_prepared_problem(pointwise);
        let prepared_problem = match self.generated_backend_config.matrix_backend {
            BvpSciGeneratedMatrixBackend::Sparse => {
                PreparedProblem::sparse(prepared_pointwise.as_prepared_problem(
                    config.residual_chunking_strategy,
                    config.sparse_jacobian_chunking_strategy,
                ))
            }
            BvpSciGeneratedMatrixBackend::Banded => PreparedProblem::banded(
                prepared_pointwise.as_prepared_banded_problem(
                    config.residual_chunking_strategy,
                    self.generated_backend_config
                        .banded_jacobian_chunking_strategy,
                ),
            ),
        };
        let manifest = PreparedProblemManifest::from(&prepared_problem);
        let problem_key = manifest.problem_key();
        if resolve_linked_sparse_backend(problem_key.as_str()).is_some() {
            self.record_generated_backend_lifecycle(
                "reused_linked",
                problem_key.clone(),
                Self::build_policy_label(config.build_policy),
                "linked-runtime",
            );
            return Ok(problem_key);
        }

        match config.build_policy {
            SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: "symbolic BVP_sci generated AOT artifact is missing".to_string(),
                });
            }
            SymbolicIvpAotBuildPolicy::UseIfAvailable => {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message:
                        "symbolic BVP_sci generated AOT artifact is not linked in the current process"
                            .to_string(),
                });
            }
            SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
            | SymbolicIvpAotBuildPolicy::RebuildAlways { .. } => {}
        }

        let preset = Self::build_preset(config.build_policy).ok_or_else(|| {
            BvpSciBackendError::GeneratedBackendFailure {
                message: "symbolic BVP_sci generated build policy did not provide a build preset"
                    .to_string(),
            }
        })?;
        let output_parent_dir = config.output_parent_dir.as_deref().ok_or(
            BvpSciBackendError::GeneratedBackendFailure {
                message: "symbolic BVP_sci generated build requested without output directory"
                    .to_string(),
            },
        )?;
        let (artifact_name, module_name) = Self::generated_backend_names(
            problem_key.as_str(),
            config,
            self.generated_backend_config.matrix_backend,
        );
        let artifact = generated_aot_artifact_from_prepared_problem(
            &artifact_name,
            &module_name,
            &prepared_problem,
            config.aot_codegen_backend,
        );
        let mut request =
            generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset);
        if let (GeneratedAotBuildRequest::C(c_request), Some(compiler)) =
            (&mut request, config.aot_c_compiler.as_ref())
        {
            let compile_config = match preset {
                AotBuildPreset::Production => CAotCompileConfig::production(),
                AotBuildPreset::FastBuild => CAotCompileConfig::fast_build(),
                AotBuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
            }
            .with_compiler(compiler.clone());
            *c_request = c_request.clone().with_compile_config(compile_config);
        }

        let build =
            request
                .materialize()
                .map_err(|err| BvpSciBackendError::GeneratedBackendFailure {
                    message: err.to_string(),
                })?;
        let executed =
            build
                .execute()
                .map_err(|err| BvpSciBackendError::GeneratedBackendFailure {
                    message: err.to_string(),
                })?;
        if !executed.succeeded() {
            let (status, stdout, stderr) = match &executed {
                ExecutedGeneratedAotBuild::Rust(result) => (
                    result.status_code,
                    result.stdout.clone(),
                    result.stderr.clone(),
                ),
                ExecutedGeneratedAotBuild::C(result) => (
                    result.status_code,
                    result.stdout.clone(),
                    result.stderr.clone(),
                ),
                ExecutedGeneratedAotBuild::Zig(result) => (
                    result.status_code,
                    result.stdout.clone(),
                    result.stderr.clone(),
                ),
            };
            return Err(BvpSciBackendError::GeneratedBackendFailure {
                message: format!("status={status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}"),
            });
        }

        let mut registry = AotRegistry::new();
        let registered = match &build {
            GeneratedAotBuildResult::Rust(result) => registry
                .register_materialized_build(manifest.clone(), result)
                .clone(),
            GeneratedAotBuildResult::C(result) => {
                register_c_build_in_registry(&mut registry, manifest.clone(), result).clone()
            }
            GeneratedAotBuildResult::Zig(result) => {
                register_zig_build_in_registry(&mut registry, manifest.clone(), result).clone()
            }
        };
        match (
            self.generated_backend_config.matrix_backend,
            config.aot_codegen_backend,
        ) {
            (BvpSciGeneratedMatrixBackend::Sparse, AotCodegenBackend::Rust) => {
                register_generated_sparse_cdylib_backend(&registered)
            }
            (BvpSciGeneratedMatrixBackend::Sparse, AotCodegenBackend::C) => {
                register_generated_c_sparse_backend(&registered)
            }
            (BvpSciGeneratedMatrixBackend::Sparse, AotCodegenBackend::Zig) => {
                register_generated_zig_sparse_backend(&registered)
            }
            (BvpSciGeneratedMatrixBackend::Banded, AotCodegenBackend::Rust) => {
                register_generated_banded_cdylib_backend(&registered)
            }
            (BvpSciGeneratedMatrixBackend::Banded, AotCodegenBackend::C) => {
                register_generated_c_banded_backend(&registered)
            }
            (BvpSciGeneratedMatrixBackend::Banded, AotCodegenBackend::Zig) => {
                register_generated_zig_banded_backend(&registered)
            }
        }
        .map_err(|message| BvpSciBackendError::GeneratedBackendFailure { message })?;
        self.record_generated_backend_lifecycle(
            "built_and_linked",
            problem_key.clone(),
            Self::build_policy_label(config.build_policy),
            Self::codegen_backend_label(config.aot_codegen_backend),
        );
        Ok(problem_key)
    }

    pub(crate) fn prepare_generated_sparse_problem(
        &mut self,
    ) -> Result<PreparedBvpSciSparseProblem, BvpSciBackendError> {
        let backend_config = self
            .bvp_generated_backend_config()?
            .expect("generated backend config should exist");
        let pointwise = self.prepare_atomview_pointwise_problem()?;
        let sparse_structure = Self::symbolic_sparse_structure(&pointwise);
        let linked_problem_key =
            self.ensure_sparse_generated_runtime(&pointwise, &backend_config)?;
        let (residual_eval, residual_diagnostics) =
            self.linked_sparse_residual_eval(linked_problem_key.as_str())?;
        let param_jacobian_eval = pointwise.symbolic_param_jacobian_sparse.as_ref().map(
            |entries: &Vec<(usize, usize, Expr)>| {
                compile_bvp_sci_sparse_matrix_family_eval(
                    entries.as_slice(),
                    (
                        pointwise.equations.len(),
                        pointwise
                            .equation_parameters
                            .as_ref()
                            .map_or(0, |params| params.len()),
                    ),
                    pointwise.time_arg.as_str(),
                    pointwise.variables.as_slice(),
                    pointwise.equation_parameters.as_deref(),
                    self.Bounds.clone(),
                )
            },
        );
        let (jacobian_eval, jacobian_diagnostics) = if self.use_analytical_jacobian {
            let (eval, diagnostics) = self.linked_sparse_jacobian_eval(
                linked_problem_key.as_str(),
                sparse_structure,
                param_jacobian_eval,
            )?;
            (Some(eval), diagnostics)
        } else {
            (
                None,
                LinkedRuntimeStageDiagnostics::sequential(0, 0, "jacobian_disabled"),
            )
        };
        self.record_generated_backend_runtime_diagnostics(Self::linked_runtime_diagnostics(
            &residual_diagnostics,
            &jacobian_diagnostics,
            "aot.runtime.sparse_jacobian",
        ));

        Ok(PreparedBvpSciSparseProblem {
            residual_eval,
            jacobian_eval,
            boundary_conditions: self.BoundaryConditions.clone(),
            values: self.values.clone(),
        })
    }

    pub(crate) fn prepare_generated_banded_problem(
        &mut self,
    ) -> Result<PreparedBvpSciBandedProblem, BvpSciBackendError> {
        let backend_config = self
            .bvp_generated_backend_config()?
            .expect("generated backend config should exist");
        let pointwise = self.prepare_atomview_pointwise_problem()?;
        let banded_structure = Self::symbolic_banded_structure(&pointwise);
        let linked_problem_key =
            self.ensure_sparse_generated_runtime(&pointwise, &backend_config)?;
        let (residual_eval, residual_diagnostics) =
            self.linked_sparse_residual_eval(linked_problem_key.as_str())?;
        let param_jacobian_eval = pointwise.symbolic_param_jacobian_sparse.as_ref().map(
            |entries: &Vec<(usize, usize, Expr)>| {
                compile_bvp_sci_sparse_matrix_family_eval(
                    entries.as_slice(),
                    (
                        pointwise.equations.len(),
                        pointwise
                            .equation_parameters
                            .as_ref()
                            .map_or(0, |params| params.len()),
                    ),
                    pointwise.time_arg.as_str(),
                    pointwise.variables.as_slice(),
                    pointwise.equation_parameters.as_deref(),
                    self.Bounds.clone(),
                )
            },
        );
        let (banded_jacobian_eval, sparse_compat_jacobian_eval, jacobian_diagnostics) =
            if self.use_analytical_jacobian {
                let (banded, sparse, diagnostics) = self.linked_banded_jacobian_eval(
                    linked_problem_key.as_str(),
                    banded_structure,
                    param_jacobian_eval,
                )?;
                (Some(banded), Some(sparse), diagnostics)
            } else {
                (
                    None,
                    None,
                    LinkedRuntimeStageDiagnostics::sequential(0, 0, "jacobian_disabled"),
                )
            };
        self.record_generated_backend_runtime_diagnostics(Self::linked_runtime_diagnostics(
            &residual_diagnostics,
            &jacobian_diagnostics,
            "aot.runtime.banded_jacobian",
        ));

        Ok(PreparedBvpSciBandedProblem {
            residual_eval,
            banded_jacobian_eval,
            sparse_compat_jacobian_eval,
            boundary_conditions: self.BoundaryConditions.clone(),
            values: self.values.clone(),
        })
    }

    pub(crate) fn bvp_generated_backend_config(
        &self,
    ) -> Result<Option<SymbolicIvpGeneratedBackendConfig>, BvpSciBackendError> {
        let output_parent_dir = self.generated_backend_config.output_parent_dir.clone();
        let config = match self.generated_backend_config.mode {
            BvpSciGeneratedBackendMode::LambdifyOnly => return Ok(None),
            BvpSciGeneratedBackendMode::RequirePrebuiltAot => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::RequirePrebuilt,
                )
            }
            BvpSciGeneratedBackendMode::BuildIfMissingRelease => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_tcc()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_rust()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_gcc()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_tcc()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_zig()
            }
            BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_tcc()
            }
        };
        Ok(Some(match output_parent_dir {
            Some(output_parent_dir) => config
                .with_resolver(self.generated_backend_config.resolver.clone())
                .with_output_parent_dir(Some(output_parent_dir))
                .with_residual_chunking_strategy(
                    self.generated_backend_config.residual_chunking_strategy,
                )
                .with_sparse_jacobian_chunking_strategy(
                    self.generated_backend_config
                        .sparse_jacobian_chunking_strategy,
                ),
            None => config
                .with_resolver(self.generated_backend_config.resolver.clone())
                .with_residual_chunking_strategy(
                    self.generated_backend_config.residual_chunking_strategy,
                )
                .with_sparse_jacobian_chunking_strategy(
                    self.generated_backend_config
                        .sparse_jacobian_chunking_strategy,
                ),
        }))
    }

    /// Entry point for AtomView+AOT backend generation.
    pub(crate) fn eq_generate_generated(
        &mut self,
    ) -> Result<
        (
            Option<Box<crate::numerical::BVP_sci::BVP_sci_faer::ODEJacobian>>,
            Box<crate::numerical::BVP_sci::BVP_sci_faer::ODEFunction>,
            Option<Box<crate::numerical::BVP_sci::BVP_sci_faer::BCFunction>>,
        ),
        BvpSciBackendError,
    > {
        match self.generated_backend_config.matrix_backend {
            BvpSciGeneratedMatrixBackend::Sparse => {
                self.banded_jac_function = None;
                let prepared = self.prepare_generated_sparse_problem()?;
                Ok(self.wrap_prepared_sparse_problem(prepared))
            }
            BvpSciGeneratedMatrixBackend::Banded => {
                let prepared = self.prepare_generated_banded_problem()?;
                self.banded_jac_function = prepared.banded_jacobian_eval.clone();
                Ok(self.wrap_prepared_banded_problem(prepared))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;

    #[test]
    fn sparse_runtime_plan_respects_requested_chunking_strategy() {
        let prepared = BvpSciPointwiseSparsePrepared {
            residuals: vec![Expr::parse_expression("y"), Expr::parse_expression("-y")],
            variable_names: vec!["y".to_string()],
            parameter_names: Vec::new(),
            time_arg: "x".to_string(),
            residual_fn_name: "eval_residual".to_string(),
            jacobian_fn_name: "eval_sparse_values".to_string(),
            sparse_entries: vec![
                (0, 0, Expr::parse_expression("1")),
                (1, 0, Expr::parse_expression("-1")),
            ],
            shape: (2, 2),
        };
        let plan = prepared
            .sparse_runtime_plan(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 2 });

        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.input_names[0], "x");
        assert!(plan.chunks.iter().all(|chunk| !chunk.fn_name.is_empty()));
    }

    #[test]
    fn banded_runtime_plan_preserves_diagonal_structure_and_chunking() {
        let prepared = BvpSciPointwiseSparsePrepared {
            residuals: vec![
                Expr::parse_expression("y1"),
                Expr::parse_expression("y0 + y1 + y2"),
                Expr::parse_expression("y1"),
            ],
            variable_names: vec!["y0".to_string(), "y1".to_string(), "y2".to_string()],
            parameter_names: Vec::new(),
            time_arg: "x".to_string(),
            residual_fn_name: "eval_residual".to_string(),
            jacobian_fn_name: "eval_banded_values".to_string(),
            sparse_entries: vec![
                (0, 1, Expr::parse_expression("1")),
                (1, 0, Expr::parse_expression("1")),
                (1, 1, Expr::parse_expression("1")),
                (1, 2, Expr::parse_expression("1")),
                (2, 1, Expr::parse_expression("1")),
            ],
            shape: (3, 3),
        };

        let plan = prepared
            .banded_runtime_plan(BandedChunkingStrategy::ByTargetChunkCount { target_chunks: 3 });

        assert_eq!((plan.structure.kl, plan.structure.ku), (1, 1));
        assert_eq!(plan.nnz(), 5);
        assert_eq!(plan.chunks.len(), 3);
        assert_eq!(
            plan.chunks
                .iter()
                .map(|chunk| chunk.entries.len())
                .sum::<usize>(),
            plan.nnz()
        );
        assert_eq!(
            plan.chunks
                .iter()
                .map(|chunk| chunk.value_offset)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
        assert!(
            plan.chunks
                .iter()
                .all(|chunk| chunk.plan.kind == CodegenTaskKind::BandedJacobianValues)
        );
    }

    #[test]
    fn banded_tcc_preset_selects_true_banded_workflow() {
        let config = BvpSciGeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(
            "target/bvp-sci-banded-config-test",
        );

        assert_eq!(config.matrix_backend, BvpSciGeneratedMatrixBackend::Banded);
        assert_eq!(
            config.workflow(),
            super::super::BVP_sci_symb::BvpSciWorkflow::AtomViewAotBanded
        );
        assert_eq!(
            config.mode,
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc
        );
    }

    #[test]
    fn cleanup_registered_aot_artifacts_is_safe_without_registered_artifacts() {
        let mut config = BvpSciGeneratedBackendConfig::default();
        assert_eq!(config.cleanup_registered_aot_artifacts().unwrap(), 0);

        config = config.with_resolver(Some(
            crate::symbolic::codegen::codegen_aot_resolution::AotResolver::new(
                crate::symbolic::codegen::codegen_aot_registry::AotRegistry::new(),
            ),
        ));
        assert_eq!(config.cleanup_registered_aot_artifacts().unwrap(), 0);
    }
}
