//! Thin IVP bridge into the generic AOT lifecycle.
//!
//! This module keeps shared IVP symbolic code from rebuilding the generic AOT
//! stack by hand every time it wants to emit one generated dense IVP artifact.
//!
//! Current layering:
//! - artifact generation is backend-agnostic (`Rust` / `C` / `Zig`);
//! - generic materialized build requests are also backend-agnostic;
//! - the legacy `materialize_symbolic_ivp_aot_build(...)` helper remains
//!   intentionally Rust-shaped as a compatibility wrapper around the older
//!   Rust-only API.

use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::conversions::expr_to_atom;
use crate::symbolic::View::jacobian::{PreparedSparseAtomSystem, SparseAtomJacobianEntry};
use crate::symbolic::View::state::Symbol;
use crate::symbolic::bvp::atom_aot::{
    AtomAotBandedSlotMap, AtomAotMatrixLayout, AtomAotPreparedPlan,
};
use crate::symbolic::codegen::CodegenIR::{
    AtomOptimizationProfile, AtomTempReusePolicy, CodegenModule, GeneratedBlock,
};
use crate::symbolic::codegen::c_backend::codegen_c_aot_library::GeneratedCAotLibrary;
use crate::symbolic::codegen::codegen_aot_driver::{
    AotBuildPreset, AotCodegenBackend, GeneratedAotArtifact, GeneratedAotBuildResult,
    generated_aot_artifact_from_prepared_problem, generated_aot_build_request_from_artifact,
};
use crate::symbolic::codegen::codegen_manifest::{
    GeneratedChunkManifest, GeneratedFunctionsManifest, PreparedProblemManifest,
};
use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::CodegenOutputLayout;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile, AotBuildRequest, AotBuildResult,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_crate::GeneratedAotCrate;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_library::GeneratedZigAotLibrary;
use crate::symbolic::ivp_telemetry::IvpColdStage;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, PreparedSymbolicIvpAotProblem, PreparedSymbolicIvpProblem,
    PreparedSymbolicIvpResidualAotProblem, PreparedSymbolicIvpResidualProblem,
    SymbolicIvpAotOptions,
};
use log::info;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Owned AtomView-native IVP AOT payload.
///
/// Unlike [`PreparedSymbolicIvpAotProblem`], this type does not borrow or
/// retain an `Expr` Jacobian. It is the explicit handoff between symbolic IVP
/// preparation and the language-neutral AOT codegen layer.
#[derive(Debug, Clone)]
pub struct PreparedSymbolicIvpAtomAotProblem {
    plan: AtomAotPreparedPlan,
    residual_fn_name: String,
    jacobian_fn_name: String,
    telemetry: Option<crate::symbolic::ivp_telemetry::IvpTelemetry>,
}

impl PreparedSymbolicIvpAtomAotProblem {
    pub const fn plan(&self) -> &AtomAotPreparedPlan {
        &self.plan
    }

    pub(crate) fn telemetry(&self) -> Option<&crate::symbolic::ivp_telemetry::IvpTelemetry> {
        self.telemetry.as_ref()
    }

    pub fn flattened_input_names(&self) -> &[String] {
        self.plan.input_names()
    }

    pub fn manifest(&self) -> PreparedProblemManifest {
        let matrix_backend = match self.plan.matrix_layout() {
            AtomAotMatrixLayout::Dense { .. } => MatrixBackend::Dense,
            AtomAotMatrixLayout::SparseCsc { .. } => MatrixBackend::SparseCol,
            AtomAotMatrixLayout::Banded { .. } | AtomAotMatrixLayout::BandedCompact { .. } => {
                MatrixBackend::Banded
            }
        };
        PreparedProblemManifest::from_atom_aot_plan(
            BackendKind::Aot,
            matrix_backend,
            &self.plan,
            self.functions_manifest(),
        )
    }

    pub fn problem_key(&self) -> String {
        self.manifest().problem_key()
    }

    fn functions_manifest(&self) -> GeneratedFunctionsManifest {
        let residual_ranges =
            chunk_ranges(self.plan.residuals().len(), self.plan.residual_strategy());
        let jacobian_len = self.plan.matrix_layout().value_count();
        let jacobian_ranges = chunk_ranges(jacobian_len, self.plan.jacobian_strategy());
        let residual_chunks = chunk_manifests(&self.residual_fn_name, &residual_ranges);
        let jacobian_chunks = chunk_manifests(&self.jacobian_fn_name, &jacobian_ranges);
        GeneratedFunctionsManifest {
            residual_fn_name: self.residual_fn_name.clone(),
            residual_chunk_names: residual_chunks
                .iter()
                .map(|chunk| chunk.fn_name.clone())
                .collect(),
            residual_chunks,
            jacobian_fn_name: self.jacobian_fn_name.clone(),
            jacobian_chunk_names: jacobian_chunks
                .iter()
                .map(|chunk| chunk.fn_name.clone())
                .collect(),
            jacobian_chunks,
        }
    }
}

fn chunk_ranges<T>(len: usize, strategy: T) -> Vec<(usize, usize)>
where
    T: Into<NativeChunking>,
{
    let strategy = strategy.into();
    let max_chunk_len = match strategy {
        NativeChunking::Whole => len.max(1),
        NativeChunking::TargetChunks(target) => len.max(1).div_ceil(target.max(1)),
        NativeChunking::MaxItems(max_items) => max_items.max(1),
    };
    if len == 0 {
        return vec![(0, 0)];
    }
    (0..len)
        .step_by(max_chunk_len)
        .map(|start| (start, (start + max_chunk_len).min(len)))
        .collect()
}

fn chunk_manifests(base_name: &str, ranges: &[(usize, usize)]) -> Vec<GeneratedChunkManifest> {
    ranges
        .iter()
        .enumerate()
        .map(|(index, &(start, end))| GeneratedChunkManifest {
            fn_name: if ranges.len() == 1 {
                base_name.to_string()
            } else {
                format!("{base_name}_chunk_{index}")
            },
            offset: start,
            len: end - start,
        })
        .collect()
}

#[derive(Clone, Copy)]
enum NativeChunking {
    Whole,
    TargetChunks(usize),
    MaxItems(usize),
}

impl From<ResidualChunkingStrategy> for NativeChunking {
    fn from(strategy: ResidualChunkingStrategy) -> Self {
        match strategy {
            ResidualChunkingStrategy::Whole => Self::Whole,
            ResidualChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                Self::TargetChunks(target_chunks)
            }
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk,
            } => Self::MaxItems(max_outputs_per_chunk),
        }
    }
}

impl From<SparseChunkingStrategy> for NativeChunking {
    fn from(strategy: SparseChunkingStrategy) -> Self {
        match strategy {
            SparseChunkingStrategy::Whole => Self::Whole,
            SparseChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                Self::TargetChunks(target_chunks)
            }
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk,
            } => Self::MaxItems(max_entries_per_chunk),
            SparseChunkingStrategy::ByRowCount { rows_per_chunk } => Self::MaxItems(rows_per_chunk),
        }
    }
}

/// Builds the AtomView-native AOT payload from an already prepared IVP.
///
/// The conversion from boxed `Expr` is performed only when the prepared
/// problem does not already retain its Atom payload. Differentiation, layout
/// validation and code generation then stay on packed Atom data. This is kept fallible so malformed symbolic
/// derivatives cannot be exposed as a compiler or callback failure later.
pub fn prepared_atom_aot_problem_from_symbolic_ivp_problem(
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
) -> Result<PreparedSymbolicIvpAtomAotProblem, IvpBackendError> {
    prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout(
        problem,
        options,
        AtomAotMatrixLayout::SparseCsc {
            rows: problem.equations.len(),
            cols: problem.variables.len(),
            nnz: 0,
        },
    )
}

/// Builds the AtomView-native AOT payload with an explicit matrix ABI.
///
/// Dense is intentionally selected only by the dense compatibility/control
/// route. Sparse and Banded callers continue to provide their own layout.
pub fn prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout(
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    requested_layout: AtomAotMatrixLayout,
) -> Result<PreparedSymbolicIvpAtomAotProblem, IvpBackendError> {
    if let Some(atoms) = problem.native_atoms() {
        return prepared_atom_aot_problem_from_atoms_with_layout_and_telemetry(
            atoms,
            &problem.variables,
            problem.time_arg.as_str(),
            problem.equation_parameters.as_deref(),
            options.residual_strategy,
            SparseChunkingStrategy::Whole,
            requested_layout,
            problem.explicit_jacobian.as_deref(),
            Some(&problem.telemetry),
        );
    }
    prepared_atom_aot_problem_from_parts_with_layout(
        &problem.equations,
        &problem.variables,
        problem.time_arg.as_str(),
        problem.equation_parameters.as_deref(),
        options.residual_strategy,
        SparseChunkingStrategy::Whole,
        requested_layout,
    )
}

/// Builds the same native payload from residual-only IVP preparation data.
///
/// LSODE2 deliberately prepares its residual callback separately from the
/// dense Jacobian. Keeping this constructor component-based avoids rebuilding
/// a dense Expr Jacobian merely to produce an Atom-native AOT artifact.
pub fn prepared_atom_aot_problem_from_parts(
    equations: &[crate::symbolic::symbolic_engine::Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    options: SymbolicIvpAotOptions,
) -> Result<PreparedSymbolicIvpAtomAotProblem, IvpBackendError> {
    prepared_atom_aot_problem_from_parts_with_chunking(
        equations,
        variables,
        time_arg,
        equation_parameters,
        options.residual_strategy,
        SparseChunkingStrategy::Whole,
        options,
    )
}

/// Builds the native payload with explicit residual and sparse Jacobian
/// chunking policies. The policies are captured in the prepared plan and
/// therefore participate in the artifact identity through its manifest.
pub fn prepared_atom_aot_problem_from_parts_with_chunking(
    equations: &[crate::symbolic::symbolic_engine::Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
    _options: SymbolicIvpAotOptions,
) -> Result<PreparedSymbolicIvpAtomAotProblem, IvpBackendError> {
    prepared_atom_aot_problem_from_parts_with_layout(
        equations,
        variables,
        time_arg,
        equation_parameters,
        residual_strategy,
        jacobian_strategy,
        AtomAotMatrixLayout::SparseCsc {
            rows: equations.len(),
            cols: variables.len(),
            nnz: 0,
        },
    )
}

/// Builds a native Atom payload for an explicitly selected output layout.
///
/// Sparse entries remain the canonical symbolic Jacobian representation. For
/// compact Banded output the emitter expands those entries into the complete
/// slot order, inserting `Atom::Zero` for boundary and structurally absent
/// slots exactly once during cold preparation.
pub fn prepared_atom_aot_problem_from_parts_with_layout(
    equations: &[crate::symbolic::symbolic_engine::Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
    requested_layout: AtomAotMatrixLayout,
) -> Result<PreparedSymbolicIvpAtomAotProblem, IvpBackendError> {
    let residuals = equations.iter().map(expr_to_atom).collect::<Vec<_>>();
    prepared_atom_aot_problem_from_atoms_with_layout_and_telemetry(
        &residuals,
        variables,
        time_arg,
        equation_parameters,
        residual_strategy,
        jacobian_strategy,
        requested_layout,
        None,
        None,
    )
}

/// Reuses the Atom payload retained by residual-only AtomView preparation.
///
/// This is the canonical LSODE2 AOT handoff: no second `Expr -> Atom` pass is
/// performed when the residual callback and AOT Jacobian are prepared in the
/// same lifecycle.
pub(crate) fn prepared_atom_aot_problem_from_residual_problem(
    problem: &PreparedSymbolicIvpResidualProblem,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
    requested_layout: AtomAotMatrixLayout,
) -> Result<PreparedSymbolicIvpAtomAotProblem, IvpBackendError> {
    let residuals =
        problem
            .native_atoms()
            .ok_or_else(|| IvpBackendError::AtomPreparationFailure {
                stage: "AOT AtomView payload reuse".to_string(),
                message: "residual preparation does not own a native Atom payload".to_string(),
            })?;
    prepared_atom_aot_problem_from_atoms_with_layout_and_telemetry(
        residuals,
        &problem.variables,
        problem.time_arg.as_str(),
        problem.equation_parameters.as_deref(),
        residual_strategy,
        jacobian_strategy,
        requested_layout,
        problem.explicit_jacobian.as_deref(),
        Some(&problem.telemetry),
    )
}

fn prepared_atom_aot_problem_from_atoms_with_layout_and_telemetry(
    residuals: &[Atom],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
    requested_layout: AtomAotMatrixLayout,
    explicit_jacobian: Option<&[Vec<crate::symbolic::symbolic_engine::Expr>]>,
    telemetry: Option<&crate::symbolic::ivp_telemetry::IvpTelemetry>,
) -> Result<PreparedSymbolicIvpAtomAotProblem, IvpBackendError> {
    let dependency_started = telemetry.map(|telemetry| {
        telemetry
            .start_cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::AtomDependencyAnalysis)
    });
    let sparse_system =
        PreparedSparseAtomSystem::from_atoms_discovering_dependencies(residuals, variables);
    if let Some(telemetry) = telemetry {
        telemetry.record_cold_stage(
            crate::symbolic::ivp_telemetry::IvpColdStage::AtomDependencyAnalysis,
            dependency_started.flatten(),
        );
    }
    let jacobian_entries = if let Some(explicit_jacobian) = explicit_jacobian {
        explicit_jacobian
            .iter()
            .enumerate()
            .flat_map(|(row, entries)| {
                entries.iter().enumerate().filter_map(move |(col, expr)| {
                    (!expr.is_zero()).then(|| SparseAtomJacobianEntry {
                        row,
                        col,
                        value: expr_to_atom(expr),
                    })
                })
            })
            .collect::<Vec<_>>()
    } else {
        let differentiation_started = telemetry.map(|telemetry| {
            telemetry.start_cold_stage(
                crate::symbolic::ivp_telemetry::IvpColdStage::SymbolicDifferentiation,
            )
        });
        let jacobian_entries = sparse_system
            .try_calc_sparse_jacobian_with_bandwidth(None)
            .map_err(|error| IvpBackendError::AtomDifferentiationFailure {
                row: error.row,
                col: error.col,
                source: error.source,
            });
        if let Some(telemetry) = telemetry {
            telemetry.record_cold_stage(
                crate::symbolic::ivp_telemetry::IvpColdStage::SymbolicDifferentiation,
                differentiation_started.flatten(),
            );
            if jacobian_entries.is_err() {
                telemetry.record_error();
            }
        }
        jacobian_entries?
    };

    let mut input_names =
        Vec::with_capacity(1 + variables.len() + equation_parameters.map_or(0, <[String]>::len));
    input_names.push(time_arg.to_string());
    if let Some(parameters) = equation_parameters {
        input_names.extend(parameters.iter().cloned());
    }
    input_names.extend(variables.iter().cloned());
    let input_symbols = input_names
        .iter()
        .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
        .collect::<Vec<_>>();

    let layout_started = telemetry.map(|telemetry| {
        telemetry.start_cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::LayoutPlanning)
    });
    let matrix_layout = match requested_layout {
        AtomAotMatrixLayout::Dense { rows, cols } => AtomAotMatrixLayout::Dense {
            rows: if rows == 0 { residuals.len() } else { rows },
            cols: if cols == 0 { variables.len() } else { cols },
        },
        AtomAotMatrixLayout::SparseCsc { .. } => AtomAotMatrixLayout::SparseCsc {
            rows: residuals.len(),
            cols: variables.len(),
            nnz: jacobian_entries.len(),
        },
        AtomAotMatrixLayout::Banded {
            rows,
            cols,
            kl,
            ku,
            slots,
        } => AtomAotMatrixLayout::Banded {
            rows,
            cols,
            kl,
            ku,
            slots: if slots == 0 {
                jacobian_entries.len()
            } else {
                slots
            },
        },
        AtomAotMatrixLayout::BandedCompact {
            rows,
            cols,
            kl,
            ku,
            slots,
        } => {
            let expected_slots = match AtomAotBandedSlotMap::new(rows, cols, kl, ku) {
                Ok(slot_map) => slot_map.storage_len(),
                Err(error) => {
                    if let Some(telemetry) = telemetry {
                        telemetry.record_cold_stage(
                            crate::symbolic::ivp_telemetry::IvpColdStage::LayoutPlanning,
                            layout_started.flatten(),
                        );
                    }
                    return Err(IvpBackendError::AtomPreparationFailure {
                        stage: "AOT AtomView Banded layout".to_string(),
                        message: error.to_string(),
                    });
                }
            };
            if slots != 0 && slots != expected_slots {
                if let Some(telemetry) = telemetry {
                    telemetry.record_cold_stage(
                        crate::symbolic::ivp_telemetry::IvpColdStage::LayoutPlanning,
                        layout_started.flatten(),
                    );
                }
                return Err(IvpBackendError::AtomPreparationFailure {
                    stage: "AOT AtomView Banded layout".to_string(),
                    message: format!(
                        "compact Banded layout declares {slots} slots, expected {expected_slots}"
                    ),
                });
            }
            AtomAotMatrixLayout::BandedCompact {
                rows,
                cols,
                kl,
                ku,
                slots: expected_slots,
            }
        }
    };
    if let Some(telemetry) = telemetry {
        telemetry.record_cold_stage(
            crate::symbolic::ivp_telemetry::IvpColdStage::LayoutPlanning,
            layout_started.flatten(),
        );
    }
    let plan = AtomAotPreparedPlan::from_parts(
        residuals.to_vec(),
        jacobian_entries,
        input_names,
        input_symbols,
        equation_parameters.map_or(0, <[String]>::len),
        matrix_layout,
        residual_strategy,
        jacobian_strategy,
    )
    .map_err(|error| IvpBackendError::AtomPreparationFailure {
        stage: "AOT AtomView plan validation".to_string(),
        message: error.to_string(),
    })?;

    Ok(PreparedSymbolicIvpAtomAotProblem {
        plan,
        residual_fn_name: "generated_ivp_residual_eval".to_string(),
        jacobian_fn_name: "generated_ivp_sparse_jacobian_eval".to_string(),
        telemetry: telemetry.cloned(),
    })
}

/// Emits a Rust/C/Zig AOT artifact directly from the AtomView-native IVP plan.
///
/// The generated callbacks use the same flat argument order as the historical
/// IVP bridge, but their expression lowering starts from `AtomView` and their
/// manifest records `AtomViewNative` as the symbolic route.
pub fn generated_aot_artifact_from_symbolic_ivp_atom_problem(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpAtomAotProblem,
    backend: AotCodegenBackend,
) -> Result<GeneratedAotArtifact, IvpBackendError> {
    let abi_started = problem
        .telemetry
        .as_ref()
        .map(|telemetry| telemetry.start_cold_stage(IvpColdStage::AotInputAbiPreparation));
    let vars: Arc<[String]> = problem.plan.input_names().to_vec().into();
    let symbols = problem.plan.input_symbols().to_vec();
    let var_index_map = Arc::new(
        symbols
            .iter()
            .enumerate()
            .map(|(index, symbol)| (symbol.id, index))
            .collect(),
    );
    if let (Some(telemetry), Some(started)) = (&problem.telemetry, abi_started) {
        telemetry.record_cold_stage(IvpColdStage::AotInputAbiPreparation, started);
    }
    let residual_views = problem
        .plan
        .residuals()
        .iter()
        .map(|atom| atom.as_view())
        .collect::<Vec<_>>();
    let (jacobian_rows, jacobian_cols, jacobian_layout, jacobian_atoms, jacobian_offsets) =
        match *problem.plan.matrix_layout() {
            AtomAotMatrixLayout::Dense { rows, cols } => {
                let (atoms, offsets) = dense_atoms(problem.plan(), rows, cols)?;
                (
                    rows,
                    cols,
                    CodegenOutputLayout::Matrix { rows, cols },
                    atoms,
                    offsets,
                )
            }
            AtomAotMatrixLayout::SparseCsc { rows, cols, nnz } => {
                let _ = nnz;
                (
                    rows,
                    cols,
                    CodegenOutputLayout::SparseValues {
                        rows,
                        cols,
                        nnz: problem.plan.jacobian_entries().len(),
                    },
                    problem
                        .plan
                        .jacobian_entries()
                        .iter()
                        .map(|entry| entry.value.clone())
                        .collect::<Vec<_>>(),
                    None,
                )
            }
            AtomAotMatrixLayout::Banded {
                rows,
                cols,
                kl,
                ku,
                slots,
            } => (
                rows,
                cols,
                CodegenOutputLayout::BandedValues {
                    rows,
                    cols,
                    kl,
                    ku,
                    slots,
                },
                problem
                    .plan
                    .jacobian_entries()
                    .iter()
                    .map(|entry| entry.value.clone())
                    .collect::<Vec<_>>(),
                None,
            ),
            AtomAotMatrixLayout::BandedCompact {
                rows,
                cols,
                kl,
                ku,
                slots,
            } => (
                rows,
                cols,
                CodegenOutputLayout::BandedCompactValues {
                    rows,
                    cols,
                    kl,
                    ku,
                    slots,
                },
                compact_banded_atoms(problem.plan(), rows, cols, kl, ku)?,
                None,
            ),
        };
    let jacobian_views = jacobian_atoms
        .iter()
        .map(|atom| atom.as_view())
        .collect::<Vec<_>>();
    let _ = (jacobian_rows, jacobian_cols);
    let _ = &jacobian_layout;

    // A structured Jacobian may be mathematically empty: a state-independent
    // right-hand side has no explicit sparse or banded values to emit. The
    // generated ABI already represents this as one zero-length chunk, so do
    // not manufacture a fake structural entry or reject a valid problem.
    // Dense uses its own zero-filled matrix contract in `dense_atoms`.
    let mut module = CodegenModule::new(module_name).with_language(backend.codegen_language());
    let residual_ranges = chunk_ranges(residual_views.len(), problem.plan.residual_strategy());
    for (index, (start, end)) in residual_ranges.iter().copied().enumerate() {
        let fn_name = if residual_ranges.len() == 1 {
            problem.residual_fn_name.clone()
        } else {
            format!("{}_chunk_{index}", problem.residual_fn_name)
        };
        module.push_generated_block(
            GeneratedBlock::from_atom_views_with_shared_abi_and_profile(
                fn_name,
                &residual_views[start..end],
                Arc::clone(&vars),
                Arc::clone(&var_index_map),
                Some(CodegenOutputLayout::Vector { len: end - start }),
                AtomOptimizationProfile::Full,
                AtomTempReusePolicy::Auto,
            )
            .0,
        );
    }
    // A dense callback owns one complete row-major matrix buffer. It is not a
    // sparse value slice, so keep the ABI as one full block even if a caller
    // supplied a sparse chunking policy for another layout.
    let jacobian_ranges = if matches!(jacobian_layout, CodegenOutputLayout::Matrix { .. }) {
        vec![(0, jacobian_views.len())]
    } else {
        chunk_ranges(jacobian_views.len(), problem.plan.jacobian_strategy())
    };
    for (index, (start, end)) in jacobian_ranges.iter().copied().enumerate() {
        let fn_name = if jacobian_ranges.len() == 1 {
            problem.jacobian_fn_name.clone()
        } else {
            format!("{}_chunk_{index}", problem.jacobian_fn_name)
        };
        module.push_generated_block(
            GeneratedBlock::from_atom_views_with_shared_abi_and_profile_and_output_offsets(
                fn_name,
                &jacobian_views[start..end],
                Arc::clone(&vars),
                Arc::clone(&var_index_map),
                Some(match jacobian_layout {
                    CodegenOutputLayout::SparseValues { .. } => CodegenOutputLayout::SparseValues {
                        rows: jacobian_rows,
                        cols: jacobian_cols,
                        nnz: end - start,
                    },
                    CodegenOutputLayout::BandedValues {
                        rows, cols, kl, ku, ..
                    } => CodegenOutputLayout::BandedValues {
                        rows,
                        cols,
                        kl,
                        ku,
                        slots: end - start,
                    },
                    CodegenOutputLayout::BandedCompactValues {
                        rows, cols, kl, ku, ..
                    } => CodegenOutputLayout::BandedCompactValues {
                        rows,
                        cols,
                        kl,
                        ku,
                        slots: end - start,
                    },
                    CodegenOutputLayout::Matrix { rows, cols } => {
                        CodegenOutputLayout::Matrix { rows, cols }
                    }
                    CodegenOutputLayout::Vector { .. } => {
                        unreachable!("native IVP Jacobian layout cannot be a vector")
                    }
                }),
                AtomOptimizationProfile::Full,
                AtomTempReusePolicy::Auto,
                jacobian_offsets
                    .as_ref()
                    .map(|offsets| offsets[start..end].to_vec()),
            )
            .0,
        );
    }
    let manifest = problem.manifest();
    info!(
        "Assembling AtomView-native symbolic IVP {:?} artifact '{}'",
        backend, artifact_name
    );
    match backend {
        AotCodegenBackend::Rust => Ok(GeneratedAotArtifact::Rust(
            GeneratedAotCrate::from_codegen_module(artifact_name, &module, manifest),
        )),
        AotCodegenBackend::C => Ok(GeneratedAotArtifact::C(
            GeneratedCAotLibrary::from_codegen_module(artifact_name, &module, manifest),
        )),
        AotCodegenBackend::Zig => Ok(GeneratedAotArtifact::Zig(
            GeneratedZigAotLibrary::from_codegen_module(artifact_name, &module, manifest),
        )),
    }
}

fn compact_banded_atoms(
    plan: &AtomAotPreparedPlan,
    rows: usize,
    cols: usize,
    kl: usize,
    ku: usize,
) -> Result<Vec<crate::symbolic::View::atom::Atom>, IvpBackendError> {
    let slot_map = AtomAotBandedSlotMap::new(rows, cols, kl, ku).map_err(|error| {
        IvpBackendError::AtomPreparationFailure {
            stage: "AOT AtomView Banded slot mapping".to_string(),
            message: error.to_string(),
        }
    })?;
    let entries = plan.jacobian_entries();
    let mut atoms = Vec::with_capacity(slot_map.storage_len());
    for slot in slot_map.slots() {
        let atom = slot
            .matrix_row
            .and_then(|row| {
                entries
                    .binary_search_by_key(&(row, slot.column), |entry| (entry.row, entry.col))
                    .ok()
                    .map(|index| entries[index].value.clone())
            })
            .unwrap_or_else(crate::symbolic::View::atom::Atom::new);
        atoms.push(atom);
    }
    Ok(atoms)
}

fn dense_atoms(
    plan: &AtomAotPreparedPlan,
    rows: usize,
    cols: usize,
) -> Result<(Vec<crate::symbolic::View::atom::Atom>, Option<Vec<usize>>), IvpBackendError> {
    let mut atoms = Vec::with_capacity(plan.jacobian_entries().len());
    let mut offsets = Vec::with_capacity(plan.jacobian_entries().len());
    for entry in plan.jacobian_entries() {
        if entry.row >= rows || entry.col >= cols {
            return Err(IvpBackendError::AtomPreparationFailure {
                stage: "AOT AtomView dense layout".to_string(),
                message: format!(
                    "Jacobian entry ({}, {}) is outside dense shape {}x{}",
                    entry.row, entry.col, rows, cols
                ),
            });
        }
        if !entry.value.is_zero() {
            atoms.push(entry.value.clone());
            offsets.push(entry.row * cols + entry.col);
        }
    }
    // Keep the historical non-empty callback contract for a valid all-zero
    // dense Jacobian. The ABI wrapper clears the full matrix before dispatch,
    // so one zero output preserves the result without restoring zero traffic.
    if atoms.is_empty() && rows > 0 && cols > 0 {
        atoms.push(crate::symbolic::View::atom::Atom::new());
        offsets.push(0);
    }
    Ok((atoms, Some(offsets)))
}

/// Prepares the dense IVP AOT bridge owned by the shared symbolic IVP layer.
pub fn prepared_problem_from_symbolic_ivp_problem<'a>(
    problem: &'a PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
) -> PreparedSymbolicIvpAotProblem<'a> {
    problem.prepare_dense_aot_problem(options)
}

/// Builds one backend-selected AOT artifact directly from a prepared symbolic IVP problem.
pub fn try_generated_aot_artifact_from_symbolic_ivp_problem(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    backend: AotCodegenBackend,
) -> Result<GeneratedAotArtifact, IvpBackendError> {
    if problem.native_atoms().is_some() {
        let prepared = prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout(
            problem,
            options,
            AtomAotMatrixLayout::Dense {
                rows: problem.equations.len(),
                cols: problem.variables.len(),
            },
        )?;
        return generated_aot_artifact_from_symbolic_ivp_atom_problem(
            artifact_name,
            module_name,
            &prepared,
            backend,
        );
    }

    let prepared = prepared_problem_from_symbolic_ivp_problem(problem, options);
    let prepared_problem = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::dense(
        prepared.as_prepared_problem(),
    );
    info!(
        "Assembling symbolic IVP dense {:?} artifact '{}' from prepared Expr problem",
        backend, artifact_name
    );
    Ok(generated_aot_artifact_from_prepared_problem(
        artifact_name,
        module_name,
        &prepared_problem,
        backend,
    ))
}

/// Compatibility wrapper for callers whose historical API is infallible.
/// New lifecycle code should use `try_generated_aot_artifact...` so malformed
/// Atom layouts remain typed preparation errors.
pub fn generated_aot_artifact_from_symbolic_ivp_problem(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    backend: AotCodegenBackend,
) -> GeneratedAotArtifact {
    try_generated_aot_artifact_from_symbolic_ivp_problem(
        artifact_name,
        module_name,
        problem,
        options,
        backend,
    )
    .expect("symbolic IVP AOT artifact generation should succeed")
}

/// Prepares the residual-only IVP AOT bridge owned by the shared symbolic IVP layer.
pub fn prepared_residual_problem_from_symbolic_ivp_residual_problem<'a>(
    problem: &'a PreparedSymbolicIvpResidualProblem,
    options: SymbolicIvpAotOptions,
) -> PreparedSymbolicIvpResidualAotProblem<'a> {
    problem.prepare_residual_aot_problem(options)
}

/// Builds one backend-selected AOT artifact from a residual-only IVP problem.
///
/// The emitted library contains the regular residual symbol plus a no-op
/// Jacobian symbol required by the shared AOT ABI.  Native sparse/banded LSODE2
/// paths ignore that no-op Jacobian and use their own symbolic Jacobian
/// evaluator.
pub fn generated_aot_artifact_from_symbolic_ivp_residual_problem(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpResidualProblem,
    options: SymbolicIvpAotOptions,
    backend: AotCodegenBackend,
) -> GeneratedAotArtifact {
    let prepared = prepared_residual_problem_from_symbolic_ivp_residual_problem(problem, options);
    let residual_plan = prepared.residual_runtime_plan();
    let mut module = CodegenModule::new(module_name).with_language(backend.codegen_language());
    for chunk in &residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    let manifest = prepared.manifest();
    info!(
        "Assembling symbolic IVP residual-only {:?} artifact '{}' from prepared problem",
        backend, artifact_name
    );
    match backend {
        AotCodegenBackend::Rust => GeneratedAotArtifact::Rust(
            GeneratedAotCrate::from_codegen_module(artifact_name, &module, manifest),
        ),
        AotCodegenBackend::C => GeneratedAotArtifact::C(GeneratedCAotLibrary::from_codegen_module(
            artifact_name,
            &module,
            manifest,
        )),
        AotCodegenBackend::Zig => GeneratedAotArtifact::Zig(
            GeneratedZigAotLibrary::from_codegen_module(artifact_name, &module, manifest),
        ),
    }
}

/// Builds a generated Rust AOT crate directly from a prepared symbolic IVP problem.
pub fn generated_aot_crate_from_symbolic_ivp_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
) -> GeneratedAotCrate {
    generated_aot_artifact_from_symbolic_ivp_problem(
        crate_name,
        module_name,
        problem,
        options,
        AotCodegenBackend::Rust,
    )
    .into_rust_crate()
    .expect("Rust backend must return GeneratedAotCrate")
}

/// Materializes a build request for one symbolic IVP AOT crate.
pub fn materialize_symbolic_ivp_aot_build(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    output_parent_dir: &Path,
    profile: AotBuildProfile,
) -> io::Result<AotBuildResult> {
    let crate_spec =
        generated_aot_crate_from_symbolic_ivp_problem(crate_name, module_name, problem, options);
    AotBuildRequest::new(crate_spec, output_parent_dir, profile).materialize()
}

/// Materializes one backend-selected symbolic IVP AOT artifact build.
pub fn materialize_symbolic_ivp_aot_artifact_build(
    artifact_name: &str,
    module_name: &str,
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
    backend: AotCodegenBackend,
    output_parent_dir: &Path,
    preset: AotBuildPreset,
) -> io::Result<GeneratedAotBuildResult> {
    let artifact = generated_aot_artifact_from_symbolic_ivp_problem(
        artifact_name,
        module_name,
        problem,
        options,
        backend,
    );
    generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset).materialize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_ivp::{
        SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
        prepare_symbolic_ivp_residual_problem,
    };
    use nalgebra::DVector;
    use tempfile::tempdir;

    fn elementary_ivp_problem() -> PreparedSymbolicIvpProblem {
        prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
        )
        .expect("IVP problem should prepare")
    }

    fn elementary_atom_ivp_problem() -> PreparedSymbolicIvpProblem {
        prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_symbolic_assembly_backend(
                    crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
                )
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
        )
        .expect("native AtomView IVP problem should prepare")
    }

    fn elementary_ivp_residual_problem() -> PreparedSymbolicIvpResidualProblem {
        prepare_symbolic_ivp_residual_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
        )
        .expect("IVP residual problem should prepare")
    }

    #[test]
    fn ivp_symbolic_aot_bridge_builds_prepared_problem_with_aot_backend() {
        let problem = elementary_ivp_problem();
        let prepared =
            prepared_problem_from_symbolic_ivp_problem(&problem, SymbolicIvpAotOptions::default());
        let generic = prepared.as_prepared_problem();

        assert_eq!(
            generic.backend_kind,
            crate::symbolic::codegen::codegen_provider_api::BackendKind::Aot
        );
        assert_eq!(
            generic.matrix_backend,
            crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Dense
        );
        assert_eq!(
            prepared.flattened_input_names(),
            &["t", "a", "b", "c", "y", "z"]
        );
    }

    #[test]
    fn atomview_dense_aot_uses_native_plan_and_complete_matrix_abi() {
        let problem = elementary_atom_ivp_problem();
        let prepared = prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout(
            &problem,
            SymbolicIvpAotOptions::default(),
            AtomAotMatrixLayout::Dense { rows: 2, cols: 2 },
        )
        .expect("native AtomView dense plan should prepare");

        assert!(matches!(
            prepared.plan().matrix_layout(),
            AtomAotMatrixLayout::Dense { rows: 2, cols: 2 }
        ));
        assert_eq!(prepared.plan().matrix_layout().value_count(), 4);
        assert_eq!(prepared.plan().jacobian_entries().len(), 4);

        let artifact = try_generated_aot_artifact_from_symbolic_ivp_problem(
            "native_dense_fixture",
            "native_dense_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Rust,
        )
        .expect("native AtomView dense artifact should emit");
        let crate_spec = artifact
            .into_rust_crate()
            .expect("Rust backend should produce a crate");
        assert_eq!(
            crate_spec.manifest.symbolic_route,
            crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::AtomViewNative
        );
        assert_eq!(
            crate_spec.manifest.matrix_backend,
            crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Dense
        );
        assert_eq!(
            crate_spec.manifest.io.jacobian_layout,
            Some(PreparedJacobianLayout::Dense)
        );
        assert!(crate_spec.module_source.contains("native_dense_module"));
    }

    #[test]
    fn atomview_dense_aot_preserves_all_zero_jacobian_contract() {
        let problem = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("t"), Expr::parse_expression("2*t")],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new().with_symbolic_assembly_backend(
                crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
            ),
        )
        .expect("all-zero AtomView Jacobian problem should prepare");

        let artifact = try_generated_aot_artifact_from_symbolic_ivp_problem(
            "native_zero_dense_fixture",
            "native_zero_dense_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Rust,
        )
        .expect("all-zero dense AtomView artifact should emit");
        let crate_spec = artifact
            .into_rust_crate()
            .expect("Rust backend should produce a crate");
        assert!(crate_spec.module_source.contains("structural zeros elided"));
        assert!(crate_spec.module_source.contains("out[0]"));
    }

    #[test]
    fn atomview_structured_aot_accepts_all_zero_jacobian_layouts() {
        let problem = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("t"), Expr::parse_expression("2*t")],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new().with_symbolic_assembly_backend(
                crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
            ),
        )
        .expect("all-zero structured AtomView problem should prepare");

        for layout in [
            AtomAotMatrixLayout::SparseCsc {
                rows: 2,
                cols: 2,
                nnz: 0,
            },
            AtomAotMatrixLayout::Banded {
                rows: 2,
                cols: 2,
                kl: 0,
                ku: 0,
                slots: 0,
            },
        ] {
            let prepared = prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout(
                &problem,
                SymbolicIvpAotOptions::default(),
                layout,
            )
            .expect("empty structured Jacobian should prepare");
            assert!(prepared.plan().jacobian_entries().is_empty());

            let artifact = generated_aot_artifact_from_symbolic_ivp_atom_problem(
                "native_zero_structured_fixture",
                "native_zero_structured_module",
                &prepared,
                AotCodegenBackend::Rust,
            )
            .expect("empty structured AtomView artifact should emit");
            let crate_spec = artifact
                .into_rust_crate()
                .expect("Rust backend should produce a crate");
            assert_eq!(crate_spec.manifest.functions.jacobian_chunks.len(), 1);
            assert_eq!(crate_spec.manifest.functions.jacobian_chunks[0].len, 0);
            assert_eq!(crate_spec.manifest.io.jacobian_nnz, Some(0));
            assert!(crate_spec
                .module_source
                .contains("generated_ivp_sparse_jacobian_eval"));
        }
    }

    #[test]
    fn ivp_symbolic_aot_bridge_materializes_release_build_request() {
        let problem = elementary_ivp_problem();
        let dir = tempdir().expect("tempdir should exist");
        let result = materialize_symbolic_ivp_aot_build(
            "generated_symbolic_ivp_build",
            "generated_symbolic_ivp_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            dir.path(),
            AotBuildProfile::Release,
        )
        .expect("IVP build request should materialize");

        assert_eq!(result.cargo_command_line(), "cargo build --release");
        assert!(result.written.cargo_toml.exists());
        assert!(result.written.generated_rs.exists());
    }

    #[test]
    fn ivp_symbolic_aot_bridge_can_emit_non_rust_artifacts() {
        let problem = elementary_ivp_problem();

        let c_artifact = generated_aot_artifact_from_symbolic_ivp_problem(
            "generated_symbolic_ivp_c",
            "generated_symbolic_ivp_c_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::C,
        );
        let zig_artifact = generated_aot_artifact_from_symbolic_ivp_problem(
            "generated_symbolic_ivp_zig",
            "generated_symbolic_ivp_zig_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Zig,
        );

        match c_artifact {
            GeneratedAotArtifact::C(library) => {
                assert_eq!(library.library_name, "generated_symbolic_ivp_c");
                assert!(!library.c_source.is_empty());
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::Zig(_) => {
                panic!("C backend should emit GeneratedCAotLibrary")
            }
        }

        match zig_artifact {
            GeneratedAotArtifact::Zig(library) => {
                assert_eq!(library.library_name, "generated_symbolic_ivp_zig");
                assert!(!library.zig_source.is_empty());
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::C(_) => {
                panic!("Zig backend should emit GeneratedZigAotLibrary")
            }
        }
    }

    #[test]
    fn ivp_symbolic_aot_bridge_can_emit_residual_only_artifacts() {
        let problem = elementary_ivp_residual_problem();
        let prepared = prepared_residual_problem_from_symbolic_ivp_residual_problem(
            &problem,
            SymbolicIvpAotOptions::default(),
        );
        assert_eq!(
            prepared.flattened_input_names(),
            &["t", "a", "b", "c", "y", "z"]
        );
        assert_eq!(prepared.manifest().io.jacobian_nnz, Some(0));
        assert!(prepared.manifest().functions.jacobian_fn_name.is_empty());

        let c_artifact = generated_aot_artifact_from_symbolic_ivp_residual_problem(
            "generated_symbolic_ivp_residual_c",
            "generated_symbolic_ivp_residual_c_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::C,
        );
        let zig_artifact = generated_aot_artifact_from_symbolic_ivp_residual_problem(
            "generated_symbolic_ivp_residual_zig",
            "generated_symbolic_ivp_residual_zig_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Zig,
        );

        match c_artifact {
            GeneratedAotArtifact::C(library) => {
                assert!(library.c_source.contains("generated_ivp_residual_eval"));
                assert_eq!(library.manifest.io.jacobian_nnz, Some(0));
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::Zig(_) => {
                panic!("C backend should emit residual-only GeneratedCAotLibrary")
            }
        }

        match zig_artifact {
            GeneratedAotArtifact::Zig(library) => {
                assert!(library.zig_source.contains("generated_ivp_residual_eval"));
                assert_eq!(library.manifest.io.jacobian_nnz, Some(0));
            }
            GeneratedAotArtifact::Rust(_) | GeneratedAotArtifact::C(_) => {
                panic!("Zig backend should emit residual-only GeneratedZigAotLibrary")
            }
        }
    }

    #[test]
    fn ivp_atom_aot_payload_and_emitters_stay_expr_free_at_codegen_boundary() {
        let problem = prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_symbolic_assembly_backend(
                    crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
                )
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
        )
        .expect("AtomView IVP problem should prepare");

        let prepared = prepared_atom_aot_problem_from_symbolic_ivp_problem(
            &problem,
            SymbolicIvpAotOptions::default(),
        )
        .expect("AtomView AOT payload should prepare");
        assert_eq!(
            prepared.flattened_input_names(),
            &["t", "a", "b", "c", "y", "z"]
        );
        assert_eq!(prepared.plan().residuals().len(), 2);
        assert_eq!(prepared.plan().jacobian_entries().len(), 4);
        assert_eq!(
            prepared.manifest().symbolic_route,
            crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::AtomViewNative
        );
        assert_eq!(
            prepared.manifest().io.jacobian_layout,
            Some(
                crate::symbolic::codegen::codegen_manifest::PreparedJacobianLayout::SparseExplicit
            )
        );

        for backend in [
            AotCodegenBackend::Rust,
            AotCodegenBackend::C,
            AotCodegenBackend::Zig,
        ] {
            let artifact = generated_aot_artifact_from_symbolic_ivp_atom_problem(
                "generated_atom_ivp",
                "generated_atom_ivp_module",
                &prepared,
                backend,
            )
            .expect("AtomView AOT emitter should accept its sparse plan");
            match artifact {
                GeneratedAotArtifact::Rust(crate_spec) => {
                    assert_eq!(
                        crate_spec.manifest.symbolic_route,
                        crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::AtomViewNative
                    );
                    assert!(
                        crate_spec
                            .module_source
                            .contains("generated_ivp_sparse_jacobian_eval")
                    );
                }
                GeneratedAotArtifact::C(library) => {
                    assert_eq!(
                        library.manifest.symbolic_route,
                        crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::AtomViewNative
                    );
                    assert!(
                        library
                            .c_source
                            .contains("generated_ivp_sparse_jacobian_eval")
                    );
                }
                GeneratedAotArtifact::Zig(library) => {
                    assert_eq!(
                        library.manifest.symbolic_route,
                        crate::symbolic::codegen::codegen_manifest::PreparedSymbolicRoute::AtomViewNative
                    );
                    assert!(
                        library
                            .zig_source
                            .contains("generated_ivp_sparse_jacobian_eval")
                    );
                }
            }
        }
    }

    #[test]
    fn ivp_atom_aot_chunking_is_reflected_in_manifest_and_module() {
        let equations = vec![
            Expr::parse_expression("a*t + y + b*z"),
            Expr::parse_expression("c*y - z + b*t"),
        ];
        let variables = vec!["y".to_string(), "z".to_string()];
        let options = SymbolicIvpAotOptions::default();
        let prepared = prepared_atom_aot_problem_from_parts_with_chunking(
            &equations,
            &variables,
            "t",
            Some(&["a".to_string(), "b".to_string(), "c".to_string()]),
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 1,
            },
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: 2,
            },
            options,
        )
        .expect("chunked AtomView AOT payload should prepare");

        let manifest = prepared.manifest();
        assert_eq!(manifest.functions.residual_chunks.len(), 2);
        assert_eq!(manifest.functions.jacobian_chunks.len(), 2);
        assert_eq!(manifest.functions.residual_chunks[1].offset, 1);
        assert_eq!(manifest.functions.jacobian_chunks[1].offset, 2);

        let artifact = generated_aot_artifact_from_symbolic_ivp_atom_problem(
            "generated_chunked_atom_ivp",
            "generated_chunked_atom_ivp_module",
            &prepared,
            AotCodegenBackend::Rust,
        )
        .expect("chunked AtomView AOT artifact should emit");
        let GeneratedAotArtifact::Rust(crate_spec) = artifact else {
            panic!("Rust backend should emit a Rust generated crate")
        };
        assert!(
            crate_spec
                .module_source
                .contains("generated_ivp_residual_eval_chunk_1")
        );
        assert!(
            crate_spec
                .module_source
                .contains("generated_ivp_sparse_jacobian_eval_chunk_1")
        );
    }

    #[test]
    fn ivp_atom_aot_compact_banded_layout_owns_boundary_slots() {
        let equations = vec![
            Expr::parse_expression("y + 2*z"),
            Expr::parse_expression("3*y - z"),
        ];
        let variables = vec!["y".to_string(), "z".to_string()];
        let prepared = prepared_atom_aot_problem_from_parts_with_layout(
            &equations,
            &variables,
            "t",
            None,
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
            AtomAotMatrixLayout::BandedCompact {
                rows: 2,
                cols: 2,
                kl: 1,
                ku: 1,
                slots: 0,
            },
        )
        .expect("compact Banded AtomView payload should prepare");

        let manifest = prepared.manifest();
        assert_eq!(
            manifest.matrix_backend,
            MatrixBackend::Banded,
            "compact Banded identity must not be published as sparse CSC"
        );
        assert_eq!(
            manifest.io.jacobian_layout,
            Some(PreparedJacobianLayout::BandedCompact { kl: 1, ku: 1 })
        );
        assert_eq!(manifest.io.jacobian_nnz, Some(6));
        assert_eq!(manifest.functions.jacobian_chunks.len(), 1);
        assert_eq!(manifest.functions.jacobian_chunks[0].len, 6);

        let artifact = generated_aot_artifact_from_symbolic_ivp_atom_problem(
            "generated_compact_banded_atom_ivp",
            "generated_compact_banded_atom_ivp_module",
            &prepared,
            AotCodegenBackend::Rust,
        )
        .expect("compact Banded AtomView artifact should emit");
        let GeneratedAotArtifact::Rust(crate_spec) = artifact else {
            panic!("Rust backend should emit a Rust generated crate")
        };
        assert!(
            crate_spec
                .module_source
                .contains("generated_ivp_sparse_jacobian_eval")
        );

        let slot_map = prepared
            .plan()
            .native_banded_slot_map()
            .expect("compact Banded plan should expose its slot map");
        let ownership = slot_map
            .chunk_ownership(1)
            .expect("one worker should own the complete compact buffer");
        assert_eq!(slot_map.storage_len(), 6);
        assert_eq!(ownership.owner_of(0), Some(0));
        assert_eq!(ownership.owner_of(5), Some(0));
        assert!(
            slot_map
                .slots()
                .iter()
                .any(|slot| slot.matrix_row.is_none())
        );
    }

    #[test]
    fn ivp_symbolic_aot_bridge_can_materialize_generic_c_and_zig_builds() {
        let problem = elementary_ivp_problem();
        let dir = tempdir().expect("tempdir should exist");

        let c_build = materialize_symbolic_ivp_aot_artifact_build(
            "generated_symbolic_ivp_c_build",
            "generated_symbolic_ivp_c_build_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::C,
            dir.path(),
            AotBuildPreset::DevFastest,
        )
        .expect("C IVP build should materialize");

        let zig_build = materialize_symbolic_ivp_aot_artifact_build(
            "generated_symbolic_ivp_zig_build",
            "generated_symbolic_ivp_zig_build_module",
            &problem,
            SymbolicIvpAotOptions::default(),
            AotCodegenBackend::Zig,
            dir.path(),
            AotBuildPreset::DevFastest,
        )
        .expect("Zig IVP build should materialize");

        match c_build {
            GeneratedAotBuildResult::C(result) => {
                assert!(result.written.makefile.exists());
                assert!(result.written.generated_c.exists());
            }
            GeneratedAotBuildResult::Rust(_) | GeneratedAotBuildResult::Zig(_) => {
                panic!("C backend should materialize CAotBuildResult")
            }
        }

        match zig_build {
            GeneratedAotBuildResult::Zig(result) => {
                assert!(result.written.build_zig.exists());
                assert!(result.written.generated_zig.exists());
            }
            GeneratedAotBuildResult::Rust(_) | GeneratedAotBuildResult::C(_) => {
                panic!("Zig backend should materialize ZigAotBuildResult")
            }
        }
    }
}
