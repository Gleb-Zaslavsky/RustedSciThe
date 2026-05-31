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
use crate::symbolic::codegen::CodegenIR::{
    AtomGeneratedBlockBreakdown, AtomOptimizationProfile, AtomTempReusePolicy, CodegenModule,
    GeneratedBlock,
};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::{CodegenOutputLayout, SparseChunkingStrategy};
use rayon::prelude::*;

/// Atom-native sparse BVP codegen problem ready for direct IR/module emission.
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
    pub residual_strategy: ResidualChunkingStrategy,
    pub jacobian_strategy: SparseChunkingStrategy,
    residual_chunks: Vec<(usize, usize)>,
    sparse_chunks: Vec<(usize, usize)>,
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
                    GeneratedBlock::from_atom_views_with_symbols_with_breakdown_and_profile(
                        fn_name,
                        &views,
                        &self.input_names,
                        &self.input_symbols,
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

        let sparse_blocks = self
            .sparse_chunks
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
                    GeneratedBlock::from_atom_views_with_symbols_with_breakdown_and_profile(
                        fn_name,
                        &views,
                        &self.input_names,
                        &self.input_symbols,
                        Some(CodegenOutputLayout::SparseValues {
                            rows: self.shape.0,
                            cols: self.shape.1,
                            nnz: entries.len(),
                        }),
                        optimization_profile,
                        reuse_policy,
                    );
                (block, sparse_view_collect_ms, atom_breakdown)
            })
            .collect::<Vec<_>>();
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
        residual_strategy,
        jacobian_strategy,
        residual_chunks,
        sparse_chunks,
    };
    let sparse_nnz = prepared.sparse_entries.len();
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

        let source = prepared.codegen_module("generated_atom_bvp").emit_source();
        assert!(source.contains("pub mod generated_atom_bvp"));
        assert!(source.contains("eval_bvp_residual_chunk_0"));
        assert!(source.contains("eval_bvp_sparse_values_chunk_0"));
    }
}
