//! Atom-native BVP codegen bridge.
//!
//! This module keeps the symbolic hot path on packed [`Atom`] values all the
//! way up to ordinary [`CodegenModule`] emission:
//! `Atom discretization -> atom sparse Jacobian -> atom lowering -> emitted module`.
//!
//! The generated Rust module is intentionally representation-agnostic. Once a
//! `CodegenModule` has been emitted, the rest of the AOT pipeline no longer
//! needs to know whether the IR came from `Expr` or `AtomView`.

use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::bvp::DiscretizedBvpAtomSystem;
use crate::symbolic::View::jacobian::{PreparedSparseAtomSystem, SparseAtomJacobianEntry};
use crate::symbolic::View::state::Symbol;
use crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode;
use crate::symbolic::bvp::atom_aot::{AtomAotMatrixLayout, AtomAotPlanError, AtomAotPreparedPlan};
use crate::symbolic::codegen::CodegenIR::{
    AtomGeneratedBlockBreakdown, AtomOptimizationProfile, AtomTempReusePolicy, CodegenModule,
    GeneratedBlock,
};
use crate::symbolic::codegen::codegen_manifest::{
    GeneratedChunkManifest, GeneratedFunctionsManifest, PreparedProblemManifest,
};
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::{CodegenOutputLayout, SparseChunkingStrategy};
use rayon::prelude::*;
use std::sync::Arc;

/// Atom-native sparse BVP codegen problem ready for direct IR/module emission.
#[derive(Clone, Debug)]
pub struct PreparedSparseAtomBvpCodegen {
    pub residual_fn_name: String,
    pub jacobian_fn_name: String,
    pub variable_names: Vec<String>,
    pub param_names: Vec<String>,
    pub input_names: Vec<String>,
    pub input_symbols: Vec<Symbol>,
    pub residuals: Vec<Atom>,
    pub sparse_entries: Vec<SparseAtomJacobianEntry>,
    pub shape: (usize, usize),
    matrix_layout: AtomAotMatrixLayout,
    pub residual_strategy: ResidualChunkingStrategy,
    pub jacobian_strategy: SparseChunkingStrategy,
    residual_chunks: Vec<(usize, usize)>,
    sparse_chunks: Vec<(usize, usize)>,
    /// Prepacked full compact band values for the opt-in native ABI.
    ///
    /// Keeping these atoms on the prepared object makes boundary zero slots a
    /// cold-path cost. Warm callback generation never recreates maps or
    /// materializes an `Expr` representation.
    banded_compact_values: Option<Vec<Atom>>,
    aot_telemetry_mode: BvpAotTelemetryMode,
}

/// Fine-grained preparation breakdown for atom-native sparse BVP codegen.
///
/// This isolates the stages that happen before IR lowering proper:
/// - preparing sparse lookup data over already discretized atoms,
/// - building the sparse symbolic Jacobian itself,
/// - and packaging residual/Jacobian data into the codegen bridge object.
#[derive(Clone, Debug, Default)]
pub struct AtomBvpCodegenPrepBreakdown {
    pub sparse_lookup_prepare_ms: f64,
    pub sparse_jacobian_build_ms: f64,
    pub finalize_codegen_plan_ms: f64,
    pub sparse_nnz: usize,
}

/// Fine-grained module-lowering breakdown for atom-native sparse BVP codegen.
#[derive(Clone, Debug, Default)]
pub struct AtomBvpCodegenModuleBreakdown {
    pub input_abi_prepare_ms: f64,
    pub residual_view_collect_ms: f64,
    pub residual_lower_many_ms: f64,
    pub residual_peephole_ms: f64,
    pub residual_reuse_temps_ms: f64,
    pub residual_reuse_temps_blocks: usize,
    pub residual_push_ms: f64,
    pub sparse_view_collect_ms: f64,
    pub sparse_lower_many_ms: f64,
    pub sparse_peephole_ms: f64,
    pub sparse_reuse_temps_ms: f64,
    pub sparse_reuse_temps_blocks: usize,
    pub sparse_push_ms: f64,
}

impl PreparedSparseAtomBvpCodegen {
    /// Marks the flat Jacobian callback as a banded callback without changing
    /// its contiguous value ABI or entry order.
    pub fn with_banded_layout(mut self, kl: usize, ku: usize) -> Result<Self, AtomAotPlanError> {
        let (rows, cols) = self.shape;
        if kl >= rows || ku >= cols {
            return Err(AtomAotPlanError::InvalidBandedLayout { kl, ku, rows, cols });
        }
        for entry in &self.sparse_entries {
            let in_band = entry.row.saturating_add(ku) >= entry.col
                && entry.col.saturating_add(kl) >= entry.row;
            if !in_band {
                return Err(AtomAotPlanError::JacobianEntryOutsideBand {
                    row: entry.row,
                    col: entry.col,
                    kl,
                    ku,
                });
            }
        }
        self.matrix_layout = AtomAotMatrixLayout::Banded {
            rows,
            cols,
            kl,
            ku,
            slots: self.sparse_entries.len(),
        };
        Ok(self)
    }

    /// Enables the native full-slot compact Banded callback ABI.
    ///
    /// The first production-safe integration is deliberately restricted to a
    /// whole Jacobian block. Chunked callbacks have a different outer routing
    /// contract and are kept on the explicit-entry ABI until that routing is
    /// migrated and covered by its own parity tests.
    pub fn with_native_banded_layout(
        mut self,
        kl: usize,
        ku: usize,
    ) -> Result<Self, AtomAotPlanError> {
        let (rows, cols) = self.shape;
        if self.sparse_chunks.len() != 1 {
            return Err(AtomAotPlanError::MatrixLayoutMismatch {
                expected: "whole Jacobian chunking for native Banded compact layout",
            });
        }
        let slot_map =
            crate::symbolic::bvp::atom_aot::AtomAotBandedSlotMap::new(rows, cols, kl, ku)?;
        let mut values = (0..slot_map.storage_len())
            .map(|_| Atom::new_num(0i64))
            .collect::<Vec<_>>();
        let mut occupied = vec![false; values.len()];
        for entry in &self.sparse_entries {
            let storage_index = slot_map.storage_index(entry.row, entry.col).ok_or(
                AtomAotPlanError::JacobianEntryOutsideBand {
                    row: entry.row,
                    col: entry.col,
                    kl,
                    ku,
                },
            )?;
            if occupied[storage_index] {
                return Err(AtomAotPlanError::DuplicateJacobianCoordinate {
                    row: entry.row,
                    col: entry.col,
                });
            }
            occupied[storage_index] = true;
            values[storage_index] = entry.value.clone();
        }
        self.matrix_layout = AtomAotMatrixLayout::BandedCompact {
            rows,
            cols,
            kl,
            ku,
            slots: slot_map.storage_len(),
        };
        self.banded_compact_values = Some(values);
        Ok(self)
    }

    fn jacobian_codegen_layout(&self, slots: usize) -> CodegenOutputLayout {
        match self.matrix_layout {
            AtomAotMatrixLayout::Dense { rows, cols } => CodegenOutputLayout::Matrix { rows, cols },
            AtomAotMatrixLayout::SparseCsc { rows, cols, .. } => {
                CodegenOutputLayout::SparseValues {
                    rows,
                    cols,
                    nnz: slots,
                }
            }
            AtomAotMatrixLayout::Banded {
                rows, cols, kl, ku, ..
            } => CodegenOutputLayout::BandedValues {
                rows,
                cols,
                kl,
                ku,
                slots,
            },
            AtomAotMatrixLayout::BandedCompact {
                rows,
                cols,
                kl,
                ku,
                slots,
            } => CodegenOutputLayout::BandedCompactValues {
                rows,
                cols,
                kl,
                ku,
                slots,
            },
        }
    }

    /// Returns the typed AtomView AOT payload owned by this codegen bridge.
    ///
    /// This is intentionally fallible and does not materialize an `Expr`.
    /// Emitters can validate the symbolic payload and output layout before
    /// selecting a compiler-specific artifact path.
    pub fn prepared_aot_plan(&self) -> Result<AtomAotPreparedPlan, AtomAotPlanError> {
        self.prepared_aot_plan_with_telemetry(self.aot_telemetry_mode)
    }

    /// Sets the typed telemetry policy carried by future AOT prepared plans.
    ///
    /// The policy is metadata on the prepared route. It does not add any
    /// work to Atom lowering and `Off` keeps the plan free of an `Arc` state
    /// allocation.
    pub fn with_aot_telemetry_mode(mut self, mode: BvpAotTelemetryMode) -> Self {
        self.aot_telemetry_mode = mode;
        self
    }

    /// Returns the telemetry policy captured by this codegen payload.
    pub const fn aot_telemetry_mode(&self) -> BvpAotTelemetryMode {
        self.aot_telemetry_mode
    }

    /// Builds a prepared Atom AOT plan with an explicit telemetry policy.
    pub fn prepared_aot_plan_with_telemetry(
        &self,
        mode: BvpAotTelemetryMode,
    ) -> Result<AtomAotPreparedPlan, AtomAotPlanError> {
        AtomAotPreparedPlan::from_parts_with_telemetry(
            self.residuals.clone(),
            self.sparse_entries.clone(),
            self.input_names.clone(),
            self.input_symbols.clone(),
            self.param_names.len(),
            self.matrix_layout,
            self.residual_strategy,
            self.jacobian_strategy,
            mode,
        )
    }

    /// Creates the artifact manifest without materializing an Expr adapter.
    pub fn prepared_aot_manifest(
        &self,
        matrix_backend: MatrixBackend,
    ) -> Result<PreparedProblemManifest, AtomAotPlanError> {
        let plan = self.prepared_aot_plan()?;
        self.prepared_aot_manifest_from_plan(matrix_backend, &plan)
    }

    /// Creates a manifest from an already validated owned plan.
    ///
    /// Route adapters use this entrypoint so manifest construction does not
    /// repeat Atom validation or create a second telemetry stream.
    pub fn prepared_aot_manifest_from_plan(
        &self,
        matrix_backend: MatrixBackend,
        plan: &AtomAotPreparedPlan,
    ) -> Result<PreparedProblemManifest, AtomAotPlanError> {
        let residual_chunk_names = self
            .residual_chunks
            .iter()
            .enumerate()
            .map(|(index, &(start, end))| GeneratedChunkManifest {
                fn_name: if self.residual_chunks.len() == 1 {
                    self.residual_fn_name.clone()
                } else {
                    format!("{}_chunk_{index}", self.residual_fn_name)
                },
                offset: start,
                len: end - start,
            })
            .collect::<Vec<_>>();
        let jacobian_chunk_names = if matches!(
            plan.matrix_layout(),
            AtomAotMatrixLayout::BandedCompact { .. }
        ) {
            vec![GeneratedChunkManifest {
                fn_name: self.jacobian_fn_name.clone(),
                offset: 0,
                len: plan.matrix_layout().value_count(),
            }]
        } else {
            self.sparse_chunks
                .iter()
                .enumerate()
                .map(|(index, &(start, end))| GeneratedChunkManifest {
                    fn_name: if self.sparse_chunks.len() == 1 {
                        self.jacobian_fn_name.clone()
                    } else {
                        format!("{}_chunk_{index}", self.jacobian_fn_name)
                    },
                    offset: start,
                    len: end - start,
                })
                .collect::<Vec<_>>()
        };
        Ok(PreparedProblemManifest::from_atom_aot_plan(
            crate::symbolic::codegen::codegen_provider_api::BackendKind::Aot,
            matrix_backend,
            &plan,
            GeneratedFunctionsManifest {
                residual_fn_name: self.residual_fn_name.clone(),
                residual_chunk_names: residual_chunk_names
                    .iter()
                    .map(|chunk| chunk.fn_name.clone())
                    .collect(),
                residual_chunks: residual_chunk_names,
                jacobian_fn_name: self.jacobian_fn_name.clone(),
                jacobian_chunk_names: jacobian_chunk_names
                    .iter()
                    .map(|chunk| chunk.fn_name.clone())
                    .collect(),
                jacobian_chunks: jacobian_chunk_names,
            },
        ))
    }

    /// Emits a regular `CodegenModule` directly from packed atoms.
    pub fn codegen_module(&self, module_name: &str) -> CodegenModule {
        self.codegen_module_with_breakdown(module_name).0
    }

    /// Emits a regular `CodegenModule` and reports how much time is spent
    /// collecting views vs. each atom-lowering pass.
    pub fn codegen_module_with_breakdown(
        &self,
        module_name: &str,
    ) -> (CodegenModule, AtomBvpCodegenModuleBreakdown) {
        self.codegen_module_with_breakdown_and_optimization_profile(
            module_name,
            AtomOptimizationProfile::Full,
        )
    }

    pub fn codegen_module_with_breakdown_and_optimization_profile(
        &self,
        module_name: &str,
        optimization_profile: AtomOptimizationProfile,
    ) -> (CodegenModule, AtomBvpCodegenModuleBreakdown) {
        self.codegen_module_with_breakdown_with_options(
            module_name,
            optimization_profile,
            AtomTempReusePolicy::Auto,
        )
    }

    pub fn codegen_module_with_breakdown_and_reuse_policy(
        &self,
        module_name: &str,
        reuse_policy: AtomTempReusePolicy,
    ) -> (CodegenModule, AtomBvpCodegenModuleBreakdown) {
        self.codegen_module_with_breakdown_with_options(
            module_name,
            AtomOptimizationProfile::Full,
            reuse_policy,
        )
    }

    fn codegen_module_with_breakdown_with_options(
        &self,
        module_name: &str,
        optimization_profile: AtomOptimizationProfile,
        reuse_policy: AtomTempReusePolicy,
    ) -> (CodegenModule, AtomBvpCodegenModuleBreakdown) {
        let mut module = CodegenModule::new(module_name);
        let mut breakdown = AtomBvpCodegenModuleBreakdown::default();
        let abi_started = std::time::Instant::now();
        let shared_vars: Arc<[String]> = self.input_names.clone().into();
        let shared_var_index = Arc::new(
            self.input_symbols
                .iter()
                .enumerate()
                .map(|(index, symbol)| (symbol.id, index))
                .collect(),
        );
        breakdown.input_abi_prepare_ms = abi_started.elapsed().as_secs_f64() * 1_000.0;

        let residual_blocks = self
            .residual_chunks
            .par_iter()
            .enumerate()
            .map(|(chunk_index, &(start, end))| {
                let fn_name = if self.residual_chunks.len() == 1 {
                    self.residual_fn_name.clone()
                } else {
                    format!("{}_chunk_{chunk_index}", self.residual_fn_name)
                };
                let collect_begin = std::time::Instant::now();
                let views = self.residuals[start..end]
                    .iter()
                    .map(|atom| atom.as_view())
                    .collect::<Vec<_>>();
                let residual_view_collect_ms = collect_begin.elapsed().as_secs_f64() * 1_000.0;
                let (block, atom_breakdown) =
                    GeneratedBlock::from_atom_views_with_shared_abi_and_profile(
                        fn_name,
                        &views,
                        Arc::clone(&shared_vars),
                        Arc::clone(&shared_var_index),
                        Some(CodegenOutputLayout::Vector { len: views.len() }),
                        optimization_profile,
                        reuse_policy,
                    );
                (block, residual_view_collect_ms, atom_breakdown)
            })
            .collect::<Vec<_>>();
        for (block, view_collect_ms, atom_breakdown) in residual_blocks {
            breakdown.residual_view_collect_ms += view_collect_ms;
            accumulate_block_breakdown(&mut breakdown, &atom_breakdown, true);
            let push_begin = std::time::Instant::now();
            module.push_generated_block(block);
            breakdown.residual_push_ms += push_begin.elapsed().as_secs_f64() * 1_000.0;
        }

        let sparse_blocks = if let Some(compact_values) = &self.banded_compact_values {
            let collect_begin = std::time::Instant::now();
            let views = compact_values
                .iter()
                .map(|atom| atom.as_view())
                .collect::<Vec<_>>();
            let sparse_view_collect_ms = collect_begin.elapsed().as_secs_f64() * 1_000.0;
            let (block, atom_breakdown) =
                GeneratedBlock::from_atom_views_with_shared_abi_and_profile(
                    self.jacobian_fn_name.clone(),
                    &views,
                    Arc::clone(&shared_vars),
                    Arc::clone(&shared_var_index),
                    Some(self.jacobian_codegen_layout(views.len())),
                    optimization_profile,
                    reuse_policy,
                );
            vec![(block, sparse_view_collect_ms, atom_breakdown)]
        } else {
            self.sparse_chunks
                .par_iter()
                .enumerate()
                .map(|(chunk_index, &(start, end))| {
                    let entries = &self.sparse_entries[start..end];
                    let fn_name = if self.sparse_chunks.len() == 1 {
                        self.jacobian_fn_name.clone()
                    } else {
                        format!("{}_chunk_{chunk_index}", self.jacobian_fn_name)
                    };
                    let collect_begin = std::time::Instant::now();
                    let views = entries
                        .iter()
                        .map(|entry| entry.value.as_view())
                        .collect::<Vec<_>>();
                    let sparse_view_collect_ms = collect_begin.elapsed().as_secs_f64() * 1_000.0;
                    let (block, atom_breakdown) =
                        GeneratedBlock::from_atom_views_with_shared_abi_and_profile(
                            fn_name,
                            &views,
                            Arc::clone(&shared_vars),
                            Arc::clone(&shared_var_index),
                            Some(self.jacobian_codegen_layout(entries.len())),
                            optimization_profile,
                            reuse_policy,
                        );
                    (block, sparse_view_collect_ms, atom_breakdown)
                })
                .collect::<Vec<_>>()
        };
        for (block, view_collect_ms, atom_breakdown) in sparse_blocks {
            breakdown.sparse_view_collect_ms += view_collect_ms;
            accumulate_block_breakdown(&mut breakdown, &atom_breakdown, false);
            let push_begin = std::time::Instant::now();
            module.push_generated_block(block);
            breakdown.sparse_push_ms += push_begin.elapsed().as_secs_f64() * 1_000.0;
        }

        (module, breakdown)
    }
}

fn accumulate_block_breakdown(
    total: &mut AtomBvpCodegenModuleBreakdown,
    block: &AtomGeneratedBlockBreakdown,
    residual: bool,
) {
    if residual {
        total.residual_lower_many_ms += block.lower_many_ms;
        total.residual_peephole_ms += block.peephole_ms;
        total.residual_reuse_temps_ms += block.reuse_temps_ms;
        total.residual_reuse_temps_blocks += usize::from(block.reuse_temps_applied);
    } else {
        total.sparse_lower_many_ms += block.lower_many_ms;
        total.sparse_peephole_ms += block.peephole_ms;
        total.sparse_reuse_temps_ms += block.reuse_temps_ms;
        total.sparse_reuse_temps_blocks += usize::from(block.reuse_temps_applied);
    }
}

/// Build an atom-native sparse BVP codegen problem from an already
/// discretized atom system.
pub fn prepare_sparse_bvp_codegen_from_discretized_system(
    discretized: &DiscretizedBvpAtomSystem,
    residual_fn_name: impl Into<String>,
    jacobian_fn_name: impl Into<String>,
    param_names: Vec<String>,
    bandwidth: Option<(usize, usize)>,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
) -> PreparedSparseAtomBvpCodegen {
    prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
        discretized,
        residual_fn_name,
        jacobian_fn_name,
        param_names,
        bandwidth,
        residual_strategy,
        jacobian_strategy,
    )
    .0
}

/// Builds an atom-native sparse BVP codegen problem together with a
/// fine-grained preparation breakdown.
pub fn prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
    discretized: &DiscretizedBvpAtomSystem,
    residual_fn_name: impl Into<String>,
    jacobian_fn_name: impl Into<String>,
    param_names: Vec<String>,
    bandwidth: Option<(usize, usize)>,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
) -> (PreparedSparseAtomBvpCodegen, AtomBvpCodegenPrepBreakdown) {
    let lookup_begin = std::time::Instant::now();
    let prepared_system = PreparedSparseAtomSystem::from_atoms(
        &discretized.vector_of_functions,
        &discretized.variable_string,
        &discretized.variables_for_all_discrete,
    );
    let sparse_lookup_prepare_ms = lookup_begin.elapsed().as_secs_f64() * 1_000.0;

    let jacobian_begin = std::time::Instant::now();
    let sparse_entries = prepared_system.calc_sparse_jacobian_with_bandwidth(bandwidth);
    let sparse_jacobian_build_ms = jacobian_begin.elapsed().as_secs_f64() * 1_000.0;

    let finalize_begin = std::time::Instant::now();
    let input_names = param_names
        .iter()
        .chain(discretized.variable_string.iter())
        .cloned()
        .collect::<Vec<_>>();
    let input_symbols = input_names
        .iter()
        .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
        .collect::<Vec<_>>();
    let residual_chunks =
        chunk_residual_ranges(discretized.vector_of_functions.len(), residual_strategy);
    let sparse_chunks = chunk_sparse_ranges_indices(&sparse_entries, jacobian_strategy);
    let sparse_nnz = sparse_entries.len();
    let prepared = PreparedSparseAtomBvpCodegen {
        residual_fn_name: residual_fn_name.into(),
        jacobian_fn_name: jacobian_fn_name.into(),
        variable_names: discretized.variable_string.clone(),
        param_names,
        input_names,
        input_symbols,
        residuals: discretized.vector_of_functions.clone(),
        sparse_entries,
        shape: (
            discretized.vector_of_functions.len(),
            discretized.variable_string.len(),
        ),
        matrix_layout: AtomAotMatrixLayout::SparseCsc {
            rows: discretized.vector_of_functions.len(),
            cols: discretized.variable_string.len(),
            nnz: sparse_nnz,
        },
        residual_strategy,
        jacobian_strategy,
        residual_chunks,
        sparse_chunks,
        banded_compact_values: None,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
    };
    let finalize_codegen_plan_ms = finalize_begin.elapsed().as_secs_f64() * 1_000.0;

    (
        prepared,
        AtomBvpCodegenPrepBreakdown {
            sparse_lookup_prepare_ms,
            sparse_jacobian_build_ms,
            finalize_codegen_plan_ms,
            sparse_nnz,
        },
    )
}

fn chunk_residual_ranges(len: usize, strategy: ResidualChunkingStrategy) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    match strategy {
        ResidualChunkingStrategy::Whole => vec![(0, len)],
        ResidualChunkingStrategy::ByTargetChunkCount { target_chunks } => {
            let target_chunks = target_chunks.max(1).min(len);
            let chunk_size = len.div_ceil(target_chunks);
            (0..len)
                .step_by(chunk_size)
                .map(|start| (start, (start + chunk_size).min(len)))
                .collect()
        }
        ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk,
        } => {
            let chunk_size = max_outputs_per_chunk.max(1);
            (0..len)
                .step_by(chunk_size)
                .map(|start| (start, (start + chunk_size).min(len)))
                .collect()
        }
    }
}

fn chunk_sparse_ranges_indices(
    entries: &[SparseAtomJacobianEntry],
    strategy: SparseChunkingStrategy,
) -> Vec<(usize, usize)> {
    if entries.is_empty() {
        return Vec::new();
    }
    match strategy {
        SparseChunkingStrategy::Whole => vec![(0, entries.len())],
        SparseChunkingStrategy::ByTargetChunkCount { target_chunks } => {
            let target_chunks = target_chunks.max(1).min(entries.len());
            let chunk_size = entries.len().div_ceil(target_chunks);
            (0..entries.len())
                .step_by(chunk_size)
                .map(|start| (start, (start + chunk_size).min(entries.len())))
                .collect()
        }
        SparseChunkingStrategy::ByNonZeroCount {
            max_entries_per_chunk,
        } => {
            let chunk_size = max_entries_per_chunk.max(1);
            (0..entries.len())
                .step_by(chunk_size)
                .map(|start| (start, (start + chunk_size).min(entries.len())))
                .collect()
        }
        SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
            let rows_per_chunk = rows_per_chunk.max(1);
            let mut groups = Vec::new();
            let mut start = 0usize;
            while start < entries.len() {
                let first_row = entries[start].row;
                let max_row_exclusive = first_row + rows_per_chunk;
                let mut end = start;
                while end < entries.len() && entries[end].row < max_row_exclusive {
                    end += 1;
                }
                groups.push((start, end));
                start = end;
            }
            groups
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::View::bvp::discretization_system_bvp_par_atom;
    use crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode;
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    #[test]
    fn atom_sparse_bvp_codegen_module_emits_named_blocks() {
        let eqs = vec![
            Expr::parse_expression("z"),
            Expr::parse_expression("-(1 + 2*ln(y))*y/2"),
        ];
        let values = vec!["y".to_string(), "z".to_string()];
        let boundary_conditions = HashMap::from([
            (
                "y".to_string(),
                vec![(0usize, 1.0), (1usize, (-0.25f64).exp())],
            ),
            ("z".to_string(), vec![(0usize, 0.0)]),
        ]);
        let discretized = discretization_system_bvp_par_atom(
            eqs,
            values,
            "x".to_string(),
            0.0,
            Some(16),
            None,
            Some((0..=16).map(|i| i as f64 / 16.0).collect()),
            boundary_conditions,
            None,
            None,
            "trapezoid".to_string(),
        );
        let prepared = prepare_sparse_bvp_codegen_from_discretized_system(
            &discretized,
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            Vec::new(),
            Some((2, 0)),
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 8,
            },
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: 16,
            },
        );

        let module = prepared.codegen_module("generated_atom_bvp");
        assert!(
            module.blocks().len() > 1,
            "fixture should exercise chunking"
        );
        assert!(
            module
                .blocks()
                .windows(2)
                .all(|blocks| blocks[0].shares_input_abi_with(&blocks[1]))
        );
        let source = module.emit_source();
        assert!(source.contains("pub mod generated_atom_bvp"));
        assert!(source.contains("eval_bvp_residual_chunk_0"));
        assert!(source.contains("eval_bvp_sparse_values_chunk_0"));
    }

    #[test]
    fn atom_bvp_manifest_keeps_explicit_banded_layout_without_expr_adapter() {
        let eqs = vec![Expr::parse_expression("z"), Expr::parse_expression("-y")];
        let values = vec!["y".to_string(), "z".to_string()];
        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 1.0)]),
            ("z".to_string(), vec![(0usize, 0.0)]),
        ]);
        let discretized = discretization_system_bvp_par_atom(
            eqs,
            values,
            "x".to_string(),
            0.0,
            Some(4),
            None,
            Some((0..=4).map(|i| i as f64 / 4.0).collect()),
            boundary_conditions,
            None,
            None,
            "forward".to_string(),
        );
        let prepared = prepare_sparse_bvp_codegen_from_discretized_system(
            &discretized,
            "eval_residual",
            "eval_banded_values",
            Vec::new(),
            None,
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .with_banded_layout(7, 7)
        .expect("fixture should fit the declared band");

        let manifest = prepared
            .prepared_aot_manifest(MatrixBackend::Banded)
            .expect("AtomView manifest should validate");
        assert_eq!(manifest.matrix_backend, MatrixBackend::Banded);
        assert_eq!(manifest.io.jacobian_rows, manifest.io.jacobian_cols);
        assert_eq!(
            manifest.io.jacobian_nnz,
            Some(prepared.sparse_entries.len())
        );
        assert_eq!(
            manifest.io.jacobian_layout,
            Some(
                crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout::BandedExplicit
            )
        );
        assert!(manifest.expression_signature != 0);

        // All language emitters consume this same already-lowered Atom plan.
        // The source syntax differs, but function names and the callback ABI
        // must not silently diverge when the matrix layout is Banded.
        for language in [
            crate::symbolic::codegen::CodegenIR::CodegenLanguage::Rust,
            crate::symbolic::codegen::CodegenIR::CodegenLanguage::C,
            crate::symbolic::codegen::CodegenIR::CodegenLanguage::Zig,
        ] {
            let source = prepared
                .clone()
                .codegen_module("generated_atom_bvp")
                .with_language(language)
                .emit_source();
            assert!(!source.is_empty());
            assert!(
                source.contains("eval_residual"),
                "{language:?} emitter lost the residual callback name"
            );
            assert!(
                source.contains("eval_banded_values"),
                "{language:?} emitter lost the Banded callback name"
            );
        }

        let off_plan = prepared
            .prepared_aot_plan()
            .expect("default AtomView AOT plan should validate");
        assert_eq!(off_plan.telemetry_snapshot().mode, BvpAotTelemetryMode::Off);
        assert_eq!(
            off_plan.telemetry_snapshot().validation,
            std::time::Duration::ZERO
        );

        let detailed_plan = prepared
            .with_aot_telemetry_mode(BvpAotTelemetryMode::Detailed)
            .prepared_aot_plan()
            .expect("detailed AtomView AOT plan should validate");
        assert_eq!(
            detailed_plan.telemetry_snapshot().mode,
            BvpAotTelemetryMode::Detailed
        );
        assert!(detailed_plan.telemetry_snapshot().validation > std::time::Duration::ZERO);
    }

    #[test]
    fn atom_bvp_native_banded_codegen_emits_complete_compact_slots() {
        let eqs = vec![Expr::parse_expression("z"), Expr::parse_expression("-y")];
        let values = vec!["y".to_string(), "z".to_string()];
        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 1.0)]),
            ("z".to_string(), vec![(0usize, 0.0)]),
        ]);
        let discretized = discretization_system_bvp_par_atom(
            eqs,
            values,
            "x".to_string(),
            0.0,
            Some(4),
            None,
            Some((0..=4).map(|i| i as f64 / 4.0).collect()),
            boundary_conditions,
            None,
            None,
            "forward".to_string(),
        );
        let prepared = prepare_sparse_bvp_codegen_from_discretized_system(
            &discretized,
            "eval_residual",
            "eval_native_banded_values",
            Vec::new(),
            None,
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .with_native_banded_layout(7, 7)
        .expect("fixture should fit the declared native band");

        let plan = prepared
            .prepared_aot_plan()
            .expect("native compact AtomView plan should validate");
        let (rows, cols) = plan.matrix_layout().shape();
        let expected_slots = (7 + 7 + 1) * cols;
        assert_eq!(rows, cols);
        assert_eq!(plan.matrix_layout().value_count(), expected_slots);

        let manifest = prepared
            .prepared_aot_manifest(MatrixBackend::Banded)
            .expect("native compact manifest should validate");
        assert_eq!(manifest.io.jacobian_nnz, Some(expected_slots));
        assert_eq!(
            manifest.io.jacobian_layout,
            Some(
                crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout::BandedCompact {
                    kl: 7,
                    ku: 7,
                }
            )
        );
        assert_eq!(manifest.functions.jacobian_chunks.len(), 1);
        assert_eq!(manifest.functions.jacobian_chunks[0].offset, 0);
        assert_eq!(manifest.functions.jacobian_chunks[0].len, expected_slots);

        let module = prepared.codegen_module("generated_native_banded");
        assert_eq!(
            module.total_block_output_count(),
            discretized.vector_of_functions.len() + expected_slots
        );
        for language in [
            crate::symbolic::codegen::CodegenIR::CodegenLanguage::Rust,
            crate::symbolic::codegen::CodegenIR::CodegenLanguage::C,
            crate::symbolic::codegen::CodegenIR::CodegenLanguage::Zig,
        ] {
            let source = prepared
                .clone()
                .codegen_module("generated_native_banded")
                .with_language(language)
                .emit_source();
            assert!(source.contains("eval_native_banded_values"));
        }
    }
}
