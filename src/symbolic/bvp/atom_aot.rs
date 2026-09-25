//! Typed ownership boundary for the AtomView-native BVP AOT route.
//!
//! This module deliberately does not build or link an artifact yet.  It keeps
//! the symbolic payload and its output layout separate from the historical
//! `Expr`-based AOT adapter so that later compiler integration cannot silently
//! reintroduce an `Atom -> Expr` conversion.

use std::fmt;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

use super::aot_telemetry::{
    BvpAotChunking, BvpAotEvaluatorPolicy, BvpAotFrontend,
    BvpAotMatrixLayout as TelemetryMatrixLayout, BvpAotTelemetry, BvpAotTelemetryIdentity,
    BvpAotTelemetryMode, BvpAotTelemetrySnapshot,
};
use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::bvp::DiscretizedBvpAtomSystem;
use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
use crate::symbolic::View::state::Symbol;
use crate::symbolic::codegen::codegen_aot_resolution::ResolvedAotArtifact;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedAotCallbackError, LinkedSparseAotBackend,
};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use std::sync::Arc;

/// Output storage contract owned by an AtomView AOT plan.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AtomAotMatrixLayout {
    /// Row-major dense callback layout. The prepared symbolic entries remain
    /// sparse, while code generation emits the complete matrix buffer.
    Dense { rows: usize, cols: usize },
    /// Fixed coordinate/value ordering for a CSC-compatible sparse callback.
    SparseCsc {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
    /// Native band-slot callback layout.  The slot ordering is defined by the
    /// entry vector and is not allowed to fall back to dense staging.
    Banded {
        rows: usize,
        cols: usize,
        kl: usize,
        ku: usize,
        slots: usize,
    },
    /// Complete LAPACK compact storage, including zero-valued boundary slots.
    ///
    /// Unlike `Banded`, `slots` is the complete storage length and therefore
    /// does not have to equal the number of symbolic non-zero entries.
    BandedCompact {
        rows: usize,
        cols: usize,
        kl: usize,
        ku: usize,
        slots: usize,
    },
}

/// One slot in the compact LAPACK-style band storage used by the native
/// Banded solver.
///
/// Boundary slots are intentional: a compact band matrix has
/// `(kl + ku + 1) * cols` storage positions, while slots at the beginning or
/// end of a band row may not correspond to a matrix coordinate.  Keeping
/// those slots explicit is what makes the generated callback contract
/// deterministic and avoids a dense staging matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomAotBandedSlot {
    pub storage_index: usize,
    pub band_row: usize,
    pub column: usize,
    pub matrix_row: Option<usize>,
}

/// Explicit ownership of a compact-Banded output buffer by parallel chunks.
///
/// A compact LAPACK buffer contains boundary slots which do not correspond to
/// matrix coordinates. Those slots still belong to exactly one worker: the
/// worker owning the contiguous output range must write zero for them. Keeping
/// ownership as ranges, rather than discovering it from coordinates in the
/// callback, makes the no-overlap/no-hole rule testable before code generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomAotBandedChunkOwnership {
    storage_len: usize,
    ranges: Vec<Range<usize>>,
}

impl AtomAotBandedChunkOwnership {
    /// Splits the complete compact buffer into deterministic contiguous ranges.
    ///
    /// The effective worker count is capped at `storage_len`; therefore every
    /// range is non-empty and every compact slot has one and only one owner.
    pub fn new(storage_len: usize, requested_chunks: usize) -> Result<Self, AtomAotPlanError> {
        if storage_len == 0 || requested_chunks == 0 {
            return Err(AtomAotPlanError::InvalidChunkCount {
                storage_len,
                requested_chunks,
            });
        }
        let chunks = requested_chunks.min(storage_len);
        let base = storage_len / chunks;
        let remainder = storage_len % chunks;
        let mut ranges = Vec::with_capacity(chunks);
        let mut start = 0;
        for index in 0..chunks {
            let len = base + usize::from(index < remainder);
            let end = start + len;
            ranges.push(start..end);
            start = end;
        }
        let ownership = Self {
            storage_len,
            ranges,
        };
        ownership.validate()?;
        Ok(ownership)
    }

    /// Returns the deterministic whole-buffer ownership plan.
    pub fn whole(storage_len: usize) -> Result<Self, AtomAotPlanError> {
        Self::new(storage_len, 1)
    }

    pub const fn storage_len(&self) -> usize {
        self.storage_len
    }

    pub fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    /// Returns the unique chunk owner for a compact storage index.
    pub fn owner_of(&self, storage_index: usize) -> Option<usize> {
        self.ranges
            .iter()
            .position(|range| range.contains(&storage_index))
    }

    /// Revalidates the ownership invariant at an emitter/registration boundary.
    pub fn validate(&self) -> Result<(), AtomAotPlanError> {
        if self.storage_len == 0 || self.ranges.is_empty() {
            return Err(AtomAotPlanError::InvalidChunkCount {
                storage_len: self.storage_len,
                requested_chunks: self.ranges.len(),
            });
        }
        let mut cursor = 0;
        for range in &self.ranges {
            if range.start != cursor || range.start >= range.end || range.end > self.storage_len {
                return Err(AtomAotPlanError::InvalidChunkOwnership {
                    storage_len: self.storage_len,
                });
            }
            cursor = range.end;
        }
        if cursor != self.storage_len {
            return Err(AtomAotPlanError::InvalidChunkOwnership {
                storage_len: self.storage_len,
            });
        }
        Ok(())
    }
}

/// Cold-path mapping from matrix coordinates to native compact band storage.
///
/// This is deliberately independent from the current AOT values callback,
/// which still emits the historical explicit-entry slice.  It is the typed
/// contract required by the next native Banded emitter: symbolic entries can
/// be packed into this map without `HashMap`, `Expr` conversion, or dense
/// staging.  The map is created once per prepared plan, never per callback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomAotBandedSlotMap {
    rows: usize,
    cols: usize,
    kl: usize,
    ku: usize,
    slots: Vec<AtomAotBandedSlot>,
}

impl AtomAotBandedSlotMap {
    /// Creates a deterministic compact-storage map.
    pub fn new(rows: usize, cols: usize, kl: usize, ku: usize) -> Result<Self, AtomAotPlanError> {
        if rows == 0 || cols == 0 || kl >= rows || ku >= cols {
            return Err(AtomAotPlanError::InvalidBandedLayout { kl, ku, rows, cols });
        }
        let band_rows = kl
            .checked_add(ku)
            .and_then(|width| width.checked_add(1))
            .ok_or(AtomAotPlanError::InvalidBandedLayout { kl, ku, rows, cols })?;
        let slot_count = band_rows
            .checked_mul(cols)
            .ok_or(AtomAotPlanError::InvalidBandedLayout { kl, ku, rows, cols })?;
        let mut slots = Vec::with_capacity(slot_count);
        for band_row in 0..band_rows {
            for column in 0..cols {
                let matrix_row = column as isize + band_row as isize - ku as isize;
                let matrix_row = (matrix_row >= 0 && (matrix_row as usize) < rows)
                    .then_some(matrix_row as usize);
                slots.push(AtomAotBandedSlot {
                    storage_index: band_row * cols + column,
                    band_row,
                    column,
                    matrix_row,
                });
            }
        }
        Ok(Self {
            rows,
            cols,
            kl,
            ku,
            slots,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn kl(&self) -> usize {
        self.kl
    }

    pub const fn ku(&self) -> usize {
        self.ku
    }

    /// Full compact storage length, including boundary-zero slots.
    pub fn storage_len(&self) -> usize {
        self.slots.len()
    }

    pub fn slots(&self) -> &[AtomAotBandedSlot] {
        &self.slots
    }

    /// Returns the formal ownership plan for a parallel compact callback.
    /// Boundary slots are included in the same contiguous ranges as all other
    /// storage positions; no worker may write outside its returned range.
    pub fn chunk_ownership(
        &self,
        requested_chunks: usize,
    ) -> Result<AtomAotBandedChunkOwnership, AtomAotPlanError> {
        AtomAotBandedChunkOwnership::new(self.storage_len(), requested_chunks)
    }

    /// Returns the compact storage index for a matrix coordinate.
    pub fn storage_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        if row.saturating_add(self.ku) < col || col.saturating_add(self.kl) < row {
            return None;
        }
        Some((self.ku + row - col) * self.cols + col)
    }

    /// Packs values in explicit Jacobian coordinate order into native compact
    /// band storage.  Missing matrix coordinates remain zero, including
    /// boundary slots that do not belong to the rectangular matrix.
    pub fn pack_values(
        &self,
        coordinates: &[(usize, usize)],
        values: &[f64],
    ) -> Result<Vec<f64>, AtomAotPlanError> {
        if coordinates.len() != values.len() {
            return Err(AtomAotPlanError::JacobianValueCountMismatch {
                entries: values.len(),
                expected: coordinates.len(),
            });
        }
        let mut packed = vec![0.0; self.storage_len()];
        let mut occupied = vec![false; self.storage_len()];
        for ((row, col), value) in coordinates.iter().copied().zip(values.iter().copied()) {
            let Some(storage_index) = self.storage_index(row, col) else {
                if row >= self.rows || col >= self.cols {
                    return Err(AtomAotPlanError::InvalidJacobianCoordinate {
                        row,
                        col,
                        rows: self.rows,
                        cols: self.cols,
                    });
                }
                return Err(AtomAotPlanError::JacobianEntryOutsideBand {
                    row,
                    col,
                    kl: self.kl,
                    ku: self.ku,
                });
            };
            if occupied[storage_index] {
                return Err(AtomAotPlanError::DuplicateJacobianCoordinate { row, col });
            }
            occupied[storage_index] = true;
            packed[storage_index] = value;
        }
        Ok(packed)
    }
}

impl AtomAotMatrixLayout {
    /// Matrix dimensions carried by the generated callback contract.
    pub const fn shape(&self) -> (usize, usize) {
        match *self {
            Self::Dense { rows, cols }
            | Self::SparseCsc { rows, cols, .. }
            | Self::Banded { rows, cols, .. }
            | Self::BandedCompact { rows, cols, .. } => (rows, cols),
        }
    }

    /// Number of values emitted by the Jacobian callback.
    pub const fn value_count(&self) -> usize {
        match *self {
            Self::Dense { rows, cols } => rows * cols,
            Self::SparseCsc { nnz, .. } => nnz,
            Self::Banded { slots, .. } | Self::BandedCompact { slots, .. } => slots,
        }
    }
}

/// Typed preparation failure for the AtomView AOT plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomAotPlanError {
    EmptyResidualSystem,
    EmptyVariableSchema,
    InputSchemaMismatch {
        names: usize,
        symbols: usize,
    },
    ParameterCountOutOfRange {
        parameters: usize,
        inputs: usize,
    },
    ShapeMismatch {
        residuals: usize,
        rows: usize,
        cols: usize,
    },
    InvalidJacobianCoordinate {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    JacobianEntryOutsideBand {
        row: usize,
        col: usize,
        kl: usize,
        ku: usize,
    },
    InvalidBandedLayout {
        kl: usize,
        ku: usize,
        rows: usize,
        cols: usize,
    },
    JacobianValueCountMismatch {
        entries: usize,
        expected: usize,
    },
    DuplicateJacobianCoordinate {
        row: usize,
        col: usize,
    },
    MatrixLayoutMismatch {
        expected: &'static str,
    },
    InvalidChunkCount {
        storage_len: usize,
        requested_chunks: usize,
    },
    InvalidChunkOwnership {
        storage_len: usize,
    },
}

impl fmt::Display for AtomAotPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyResidualSystem => f.write_str("AtomView AOT residual system is empty"),
            Self::EmptyVariableSchema => f.write_str("AtomView AOT variable schema is empty"),
            Self::InputSchemaMismatch { names, symbols } => {
                write!(
                    f,
                    "AtomView AOT input schema has {names} names but {symbols} symbols"
                )
            }
            Self::ParameterCountOutOfRange { parameters, inputs } => write!(
                f,
                "AtomView AOT parameter count {parameters} exceeds input schema size {inputs}"
            ),
            Self::ShapeMismatch {
                residuals,
                rows,
                cols,
            } => write!(
                f,
                "AtomView AOT shape {rows}x{cols} does not match {residuals} residuals"
            ),
            Self::InvalidJacobianCoordinate {
                row,
                col,
                rows,
                cols,
            } => write!(
                f,
                "AtomView AOT Jacobian coordinate ({row}, {col}) is outside {rows}x{cols}"
            ),
            Self::JacobianEntryOutsideBand { row, col, kl, ku } => write!(
                f,
                "AtomView AOT Jacobian coordinate ({row}, {col}) is outside band kl={kl}, ku={ku}"
            ),
            Self::InvalidBandedLayout { kl, ku, rows, cols } => write!(
                f,
                "AtomView AOT Banded layout kl={kl}, ku={ku} is invalid for {rows}x{cols}"
            ),
            Self::JacobianValueCountMismatch { entries, expected } => write!(
                f,
                "AtomView AOT Jacobian has {entries} entries but layout requires {expected}"
            ),
            Self::DuplicateJacobianCoordinate { row, col } => write!(
                f,
                "AtomView AOT Jacobian contains duplicate coordinate ({row}, {col})"
            ),
            Self::MatrixLayoutMismatch { expected } => {
                write!(
                    f,
                    "AtomView AOT operation requires {expected} matrix layout"
                )
            }
            Self::InvalidChunkCount {
                storage_len,
                requested_chunks,
            } => write!(
                f,
                "AtomView AOT compact-Banded chunk count {requested_chunks} is invalid for storage length {storage_len}"
            ),
            Self::InvalidChunkOwnership { storage_len } => write!(
                f,
                "AtomView AOT compact-Banded chunk ownership does not cover storage length {storage_len} exactly"
            ),
        }
    }
}

impl std::error::Error for AtomAotPlanError {}

/// Typed failure reported by a prepared AtomView callback boundary.
///
/// A generated callback receives the flattened input vector
/// `[parameters..., unknowns...]` and writes into a fixed output layout.  The
/// boundary must reject malformed data before a compiler-specific ABI is
/// called; otherwise a bad artifact can turn a user error into an out-of-bounds
/// write or a misleading solver failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomAotRuntimeError {
    /// The flattened callback input has the wrong number of values.
    InputLength { expected: usize, actual: usize },
    /// A callback output buffer does not match the prepared contract.
    OutputLength {
        stage: &'static str,
        expected: usize,
        actual: usize,
    },
    /// An input value is not finite.
    NonFiniteInput { index: usize },
    /// A callback wrote a non-finite value.
    NonFiniteOutput { stage: &'static str, index: usize },
    /// The prepared plan has no linked callback runtime.
    RuntimeNotLinked,
    /// The owned linked callback rejected the request or its layout.
    LinkedCallbackFailure {
        stage: &'static str,
        message: String,
    },
    /// A compiler-specific callback panicked before returning a result.
    CallbackPanicked {
        stage: &'static str,
        message: String,
    },
}

impl fmt::Display for AtomAotRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputLength { expected, actual } => {
                write!(
                    f,
                    "AtomView AOT input has {actual} values; expected {expected}"
                )
            }
            Self::OutputLength {
                stage,
                expected,
                actual,
            } => write!(
                f,
                "AtomView AOT {stage} output has {actual} values; expected {expected}"
            ),
            Self::NonFiniteInput { index } => {
                write!(f, "AtomView AOT input value at index {index} is not finite")
            }
            Self::NonFiniteOutput { stage, index } => write!(
                f,
                "AtomView AOT {stage} output value at index {index} is not finite"
            ),
            Self::RuntimeNotLinked => {
                write!(
                    f,
                    "AtomView AOT prepared plan has no linked callback runtime"
                )
            }
            Self::LinkedCallbackFailure { stage, message } => {
                write!(f, "AtomView AOT linked {stage} callback failed: {message}")
            }
            Self::CallbackPanicked { stage, message } => {
                write!(f, "AtomView AOT {stage} callback panicked: {message}")
            }
        }
    }
}

impl std::error::Error for AtomAotRuntimeError {}

/// Lifecycle state of the owned AOT runtime binding.
///
/// Keeping this state explicit prevents a materialized artifact from being
/// mistaken for an executable callback.  In particular, `ArtifactOnly` is a
/// valid cold-to-warm transition, but it must not be used as proof that the
/// linked runtime is ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomAotRuntimeState {
    /// No artifact or linked callback is owned by the prepared plan.
    Unbound,
    /// An artifact identity is owned, but no linked callback is attached yet.
    ArtifactOnly,
    /// A linked callback runtime is owned and can be used by the adapter.
    Linked,
}

/// Owned warm-runtime binding for one prepared AtomView AOT plan.
///
/// The binding deliberately keeps the artifact identity and the linked
/// callback backend together. Rebinding replaces this value atomically from
/// the plan's point of view; invalidation drops both, so a stale callback
/// cannot survive a parameter/policy/mesh refresh in the BVP bridge.
#[derive(Clone)]
pub struct AtomAotRuntimeBinding {
    artifact: Option<ResolvedAotArtifact>,
    linked_sparse: Option<Arc<LinkedSparseAotBackend>>,
    generation: u64,
}

impl std::fmt::Debug for AtomAotRuntimeBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomAotRuntimeBinding")
            .field(
                "artifact",
                &self
                    .artifact
                    .as_ref()
                    .map(|artifact| &artifact.registered.problem_key),
            )
            .field(
                "linked_sparse",
                &self.linked_sparse.as_ref().map(|_| "linked"),
            )
            .field("generation", &self.generation)
            .finish()
    }
}

impl Default for AtomAotRuntimeBinding {
    fn default() -> Self {
        Self {
            artifact: None,
            linked_sparse: None,
            generation: 0,
        }
    }
}

impl AtomAotRuntimeBinding {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn is_bound(&self) -> bool {
        self.linked_sparse.is_some()
    }

    /// Returns the typed lifecycle state instead of forcing callers to infer
    /// it from two independent optional fields.
    pub const fn state(&self) -> AtomAotRuntimeState {
        if self.linked_sparse.is_some() {
            AtomAotRuntimeState::Linked
        } else if self.artifact.is_some() {
            AtomAotRuntimeState::ArtifactOnly
        } else {
            AtomAotRuntimeState::Unbound
        }
    }

    pub fn artifact(&self) -> Option<&ResolvedAotArtifact> {
        self.artifact.as_ref()
    }

    pub fn linked_sparse(&self) -> Option<&LinkedSparseAotBackend> {
        self.linked_sparse.as_deref()
    }
}

/// Prepared AtomView payload shared by future Rust/C/Zig AOT emitters.
///
/// The plan owns the symbolic values needed for both cold code generation and
/// warm callbacks.  It contains no `Expr`, trait object or compiler-specific
/// state.  Those belong to explicit adapters layered on top of this plan.
#[derive(Clone, Debug)]
pub struct AtomAotPreparedPlan {
    residuals: Vec<Atom>,
    jacobian_entries: Vec<SparseAtomJacobianEntry>,
    input_names: Vec<String>,
    input_symbols: Vec<Symbol>,
    parameter_count: usize,
    matrix_layout: AtomAotMatrixLayout,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: SparseChunkingStrategy,
    telemetry: BvpAotTelemetry,
    runtime_binding: AtomAotRuntimeBinding,
}

fn telemetry_matrix_layout(layout: AtomAotMatrixLayout) -> TelemetryMatrixLayout {
    match layout {
        AtomAotMatrixLayout::Dense { .. } => TelemetryMatrixLayout::Dense,
        AtomAotMatrixLayout::SparseCsc { .. } => TelemetryMatrixLayout::SparseCsc,
        AtomAotMatrixLayout::Banded { .. } => TelemetryMatrixLayout::Banded,
        AtomAotMatrixLayout::BandedCompact { .. } => TelemetryMatrixLayout::BandedCompact,
    }
}

fn telemetry_residual_chunking(strategy: ResidualChunkingStrategy) -> BvpAotChunking {
    match strategy {
        ResidualChunkingStrategy::Whole => BvpAotChunking::Whole,
        ResidualChunkingStrategy::ByTargetChunkCount { .. }
        | ResidualChunkingStrategy::ByOutputCount { .. } => BvpAotChunking::Chunked,
    }
}

fn telemetry_sparse_chunking(strategy: SparseChunkingStrategy) -> BvpAotChunking {
    match strategy {
        SparseChunkingStrategy::Whole => BvpAotChunking::Whole,
        SparseChunkingStrategy::ByTargetChunkCount { .. }
        | SparseChunkingStrategy::ByNonZeroCount { .. }
        | SparseChunkingStrategy::ByRowCount { .. } => BvpAotChunking::Chunked,
    }
}

impl AtomAotPreparedPlan {
    /// Builds a sparse AtomView AOT plan from an already discretized system.
    ///
    /// The only conversion performed here is name-to-`Symbol` interning.  The
    /// residual and Jacobian payload remains packed Atom data throughout.
    pub fn from_discretized_sparse(
        discretized: &DiscretizedBvpAtomSystem,
        parameter_names: &[String],
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> Result<Self, AtomAotPlanError> {
        let input_names = parameter_names
            .iter()
            .chain(discretized.variable_string.iter())
            .cloned()
            .collect::<Vec<_>>();
        let input_symbols = input_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let entries = crate::symbolic::View::jacobian::PreparedSparseAtomSystem::from_atoms(
            &discretized.vector_of_functions,
            &discretized.variable_string,
            &discretized.variables_for_all_discrete,
        )
        .calc_sparse_jacobian_with_bandwidth(None);
        let nnz = entries.len();

        Self::from_parts(
            discretized.vector_of_functions.clone(),
            entries,
            input_names,
            input_symbols,
            parameter_names.len(),
            AtomAotMatrixLayout::SparseCsc {
                rows: discretized.vector_of_functions.len(),
                cols: discretized.variable_string.len(),
                nnz,
            },
            residual_strategy,
            jacobian_strategy,
        )
    }

    /// Builds a plan from owned Atom payload and an explicit output layout.
    ///
    /// This constructor is also used by the codegen bridge so layout checks are
    /// performed before any language-specific source emitter is invoked.
    pub fn from_parts(
        residuals: Vec<Atom>,
        jacobian_entries: Vec<SparseAtomJacobianEntry>,
        input_names: Vec<String>,
        input_symbols: Vec<Symbol>,
        parameter_count: usize,
        matrix_layout: AtomAotMatrixLayout,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> Result<Self, AtomAotPlanError> {
        Self::from_parts_with_telemetry(
            residuals,
            jacobian_entries,
            input_names,
            input_symbols,
            parameter_count,
            matrix_layout,
            residual_strategy,
            jacobian_strategy,
            BvpAotTelemetryMode::Off,
        )
    }

    /// Builds a plan and records the typed validation stage when requested.
    ///
    /// Keeping this at the ownership boundary is important: validation is
    /// paid exactly once for a prepared payload and is not confused with a
    /// later compiler or warm callback measurement.
    pub fn from_parts_with_telemetry(
        residuals: Vec<Atom>,
        jacobian_entries: Vec<SparseAtomJacobianEntry>,
        input_names: Vec<String>,
        input_symbols: Vec<Symbol>,
        parameter_count: usize,
        matrix_layout: AtomAotMatrixLayout,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
        telemetry_mode: BvpAotTelemetryMode,
    ) -> Result<Self, AtomAotPlanError> {
        let telemetry =
            BvpAotTelemetry::with_mode(telemetry_mode).with_identity(BvpAotTelemetryIdentity {
                frontend: BvpAotFrontend::AtomView,
                matrix_layout: telemetry_matrix_layout(matrix_layout),
                evaluator_policy: Default::default(),
                residual_chunking: telemetry_residual_chunking(residual_strategy),
                jacobian_chunking: telemetry_sparse_chunking(jacobian_strategy),
            });
        let validation_started = telemetry.start_timing();
        validate_plan(
            &residuals,
            &jacobian_entries,
            &input_names,
            &input_symbols,
            matrix_layout.shape(),
            &matrix_layout,
        )
        .inspect_err(|_| telemetry.record_error())?;
        let compact_banded = matches!(matrix_layout, AtomAotMatrixLayout::BandedCompact { .. });
        let value_count_matches = if matches!(matrix_layout, AtomAotMatrixLayout::Dense { .. }) {
            // Dense output is a complete buffer; the prepared entries remain
            // the canonical sparse symbolic representation.
            jacobian_entries.len() <= matrix_layout.value_count()
        } else if compact_banded {
            jacobian_entries.len() <= matrix_layout.value_count()
        } else {
            matrix_layout.value_count() == jacobian_entries.len()
        };
        if !value_count_matches {
            telemetry.record_error();
            return Err(AtomAotPlanError::JacobianValueCountMismatch {
                entries: jacobian_entries.len(),
                expected: matrix_layout.value_count(),
            });
        }
        if parameter_count > input_names.len() {
            telemetry.record_error();
            return Err(AtomAotPlanError::ParameterCountOutOfRange {
                parameters: parameter_count,
                inputs: input_names.len(),
            });
        }
        Ok(Self {
            residuals,
            jacobian_entries,
            input_names,
            input_symbols,
            parameter_count,
            matrix_layout,
            residual_strategy,
            jacobian_strategy,
            telemetry,
            runtime_binding: AtomAotRuntimeBinding::default(),
        })
        .map(|plan| {
            let estimated_owned_bytes = plan.residuals.capacity() * std::mem::size_of::<Atom>()
                + plan.jacobian_entries.capacity() * std::mem::size_of::<SparseAtomJacobianEntry>()
                + plan.input_names.capacity() * std::mem::size_of::<String>()
                + plan.input_symbols.capacity() * std::mem::size_of::<Symbol>();
            plan.telemetry.record_allocation(estimated_owned_bytes);
            plan.telemetry.record_cold_stage(
                super::aot_telemetry::BvpAotColdStage::Validation,
                validation_started,
            );
            plan
        })
    }

    /// Enables the typed AOT telemetry stream for this prepared plan.
    pub fn with_telemetry_mode(mut self, mode: BvpAotTelemetryMode) -> Self {
        let identity = self.telemetry.identity();
        self.telemetry = BvpAotTelemetry::with_mode(mode).with_identity(identity);
        self
    }

    /// Records the evaluator policy selected by the resolved runtime plan.
    ///
    /// Preparation keeps the policy as `Unknown` unless a caller explicitly
    /// selects one. This avoids reporting an invented execution mode while
    /// still allowing the solver/codegen handoff to publish `Sequential`,
    /// `Parallel` or `Auto` as typed route identity.
    pub fn with_evaluator_policy(mut self, policy: BvpAotEvaluatorPolicy) -> Self {
        let mut identity = self.telemetry.identity();
        identity.evaluator_policy = policy;
        self.telemetry = self.telemetry.with_identity(identity);
        self
    }

    pub fn residuals(&self) -> &[Atom] {
        &self.residuals
    }

    pub fn jacobian_entries(&self) -> &[SparseAtomJacobianEntry] {
        &self.jacobian_entries
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn input_symbols(&self) -> &[Symbol] {
        &self.input_symbols
    }

    pub const fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub const fn matrix_layout(&self) -> &AtomAotMatrixLayout {
        &self.matrix_layout
    }

    /// Returns the cold-path native band storage map for a Banded plan.
    ///
    /// The explicit and full-slot Banded plans share the same coordinate map.
    /// The distinction is in the callback value count and codegen layout.
    pub fn native_banded_slot_map(&self) -> Option<AtomAotBandedSlotMap> {
        match self.matrix_layout {
            AtomAotMatrixLayout::Dense { .. } => None,
            AtomAotMatrixLayout::Banded {
                rows, cols, kl, ku, ..
            }
            | AtomAotMatrixLayout::BandedCompact {
                rows, cols, kl, ku, ..
            } => AtomAotBandedSlotMap::new(rows, cols, kl, ku).ok(),
            AtomAotMatrixLayout::SparseCsc { .. } => None,
        }
    }

    /// Packs the prepared Banded Jacobian entry order into full native compact
    /// storage. This is a cold/preparation helper, not a warm callback copy.
    pub fn pack_banded_values_for_native_storage(
        &self,
        values: &[f64],
    ) -> Result<Vec<f64>, AtomAotPlanError> {
        let Some(slot_map) = self.native_banded_slot_map() else {
            return Err(AtomAotPlanError::MatrixLayoutMismatch { expected: "Banded" });
        };
        let coordinates = self
            .jacobian_entries
            .iter()
            .map(|entry| (entry.row, entry.col))
            .collect::<Vec<_>>();
        let packed = slot_map.pack_values(&coordinates, values);
        if let Ok(packed_values) = &packed {
            self.telemetry
                .record_copy_bytes(packed_values.len() * std::mem::size_of::<f64>());
        }
        packed
    }

    pub const fn residual_strategy(&self) -> ResidualChunkingStrategy {
        self.residual_strategy
    }

    pub const fn jacobian_strategy(&self) -> SparseChunkingStrategy {
        self.jacobian_strategy
    }

    pub fn telemetry(&self) -> &BvpAotTelemetry {
        &self.telemetry
    }

    pub fn telemetry_snapshot(&self) -> BvpAotTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Returns the artifact/runtime owner without exposing mutable callback
    /// internals to code generators or solver compatibility layers.
    pub fn runtime_binding(&self) -> &AtomAotRuntimeBinding {
        &self.runtime_binding
    }

    /// Publishes a fresh linked runtime for this prepared plan.
    pub(crate) fn bind_linked_runtime(
        &mut self,
        artifact: Option<ResolvedAotArtifact>,
        linked_sparse: LinkedSparseAotBackend,
    ) {
        self.runtime_binding.generation = self.runtime_binding.generation.wrapping_add(1);
        self.runtime_binding.artifact = artifact;
        self.runtime_binding.linked_sparse = Some(Arc::new(linked_sparse));
    }

    /// Records the selected materialized artifact before its linked runtime is
    /// attached. This makes Registered-but-not-built and Compiled states
    /// visible in the same owner without pretending that a callback is ready.
    pub(crate) fn bind_artifact(&mut self, artifact: Option<ResolvedAotArtifact>) {
        self.runtime_binding.generation = self.runtime_binding.generation.wrapping_add(1);
        self.runtime_binding.artifact = artifact;
        self.runtime_binding.linked_sparse = None;
    }

    /// Invalidates the artifact and linked callback together.
    pub(crate) fn invalidate_linked_runtime(&mut self) {
        self.runtime_binding.generation = self.runtime_binding.generation.wrapping_add(1);
        self.runtime_binding.artifact = None;
        self.runtime_binding.linked_sparse = None;
    }

    /// Validates the flattened input ABI before invoking a generated callback.
    pub fn validate_runtime_input(&self, args: &[f64]) -> Result<(), AtomAotRuntimeError> {
        let result = if args.len() != self.input_names.len() {
            Err(AtomAotRuntimeError::InputLength {
                expected: self.input_names.len(),
                actual: args.len(),
            })
        } else if let Some(index) = args.iter().position(|value| !value.is_finite()) {
            Err(AtomAotRuntimeError::NonFiniteInput { index })
        } else {
            Ok(())
        };
        if result.is_err() {
            self.telemetry.record_error();
        }
        result
    }

    /// Validates a residual callback output against the prepared plan.
    pub fn validate_residual_output(&self, out: &[f64]) -> Result<(), AtomAotRuntimeError> {
        self.validate_output("residual", out, self.residuals.len())
    }

    /// Validates a Jacobian-values callback output against its fixed layout.
    pub fn validate_jacobian_output(&self, out: &[f64]) -> Result<(), AtomAotRuntimeError> {
        self.validate_output("Jacobian", out, self.matrix_layout.value_count())
    }

    /// Executes one residual callback through the prepared AtomView ABI.
    ///
    /// The callback receives the already validated flattened input and writes
    /// directly into the caller-owned output buffer.  This keeps validation,
    /// finite-value checks and warm telemetry at the same boundary for Rust,
    /// C and Zig emitters.  `E` lets a compiler-specific callback preserve its
    /// own typed runtime error without forcing this low-level module to depend
    /// on a solver integration error enum.
    pub fn try_execute_residual_callback<F, E>(
        &self,
        args: &[f64],
        out: &mut [f64],
        callback: F,
    ) -> Result<(), E>
    where
        F: FnOnce(&[f64], &mut [f64]) -> Result<(), E>,
        E: From<AtomAotRuntimeError>,
    {
        self.try_execute_residual_callback_with_chunks(args, out, 1, callback)
    }

    /// Executes a residual callback and records the actual generated chunk
    /// count for the warm telemetry snapshot.
    pub fn try_execute_residual_callback_with_chunks<F, E>(
        &self,
        args: &[f64],
        out: &mut [f64],
        chunks: usize,
        callback: F,
    ) -> Result<(), E>
    where
        F: FnOnce(&[f64], &mut [f64]) -> Result<(), E>,
        E: From<AtomAotRuntimeError>,
    {
        self.validate_runtime_input(args).map_err(E::from)?;
        let started = self.telemetry.start_timing();
        let callback_result =
            catch_unwind(AssertUnwindSafe(|| callback(args, out))).map_err(|panic| {
                AtomAotRuntimeError::CallbackPanicked {
                    stage: "residual",
                    message: callback_panic_message(panic),
                }
            });
        match callback_result {
            Err(error) => {
                self.telemetry.record_error();
                return Err(E::from(error));
            }
            Ok(Err(error)) => {
                self.telemetry.record_error();
                return Err(error);
            }
            Ok(Ok(())) => {}
        }
        self.validate_residual_output(out).map_err(E::from)?;
        self.telemetry.record_residual(started, chunks.max(1));
        Ok(())
    }

    /// Executes one sparse-Jacobian callback through the prepared AtomView
    /// ABI and validates its fixed value ordering before publishing telemetry.
    pub fn try_execute_jacobian_callback<F, E>(
        &self,
        args: &[f64],
        out: &mut [f64],
        callback: F,
    ) -> Result<(), E>
    where
        F: FnOnce(&[f64], &mut [f64]) -> Result<(), E>,
        E: From<AtomAotRuntimeError>,
    {
        self.try_execute_jacobian_callback_with_chunks(args, out, 1, callback)
    }

    /// Executes a Jacobian callback and records its actual generated chunk
    /// count.  The callback must preserve the prepared sparse/banded ordering.
    pub fn try_execute_jacobian_callback_with_chunks<F, E>(
        &self,
        args: &[f64],
        out: &mut [f64],
        chunks: usize,
        callback: F,
    ) -> Result<(), E>
    where
        F: FnOnce(&[f64], &mut [f64]) -> Result<(), E>,
        E: From<AtomAotRuntimeError>,
    {
        self.validate_runtime_input(args).map_err(E::from)?;
        let started = self.telemetry.start_timing();
        let callback_result =
            catch_unwind(AssertUnwindSafe(|| callback(args, out))).map_err(|panic| {
                AtomAotRuntimeError::CallbackPanicked {
                    stage: "Jacobian",
                    message: callback_panic_message(panic),
                }
            });
        match callback_result {
            Err(error) => {
                self.telemetry.record_error();
                return Err(E::from(error));
            }
            Ok(Err(error)) => {
                self.telemetry.record_error();
                return Err(error);
            }
            Ok(Ok(())) => {}
        }
        self.validate_jacobian_output(out).map_err(E::from)?;
        self.telemetry.record_jacobian(started, chunks.max(1));
        Ok(())
    }

    /// Executes the residual callback owned by this prepared plan.
    ///
    /// Unlike the lower-level closure adapter above, this method cannot run
    /// without a linked runtime. That distinction is the lifecycle boundary:
    /// after invalidation, an old callback owner is no longer reachable through
    /// the prepared plan.
    pub fn try_execute_bound_residual_callback(
        &self,
        args: &[f64],
        out: &mut [f64],
    ) -> Result<(), AtomAotRuntimeError> {
        self.validate_runtime_input(args)?;
        let backend = self.runtime_binding.linked_sparse().ok_or_else(|| {
            self.telemetry.record_error();
            AtomAotRuntimeError::RuntimeNotLinked
        })?;
        let started = self.telemetry.start_timing();
        backend
            .try_residual_eval(args, out)
            .map_err(map_linked_callback_error)?;
        self.validate_residual_output(out)?;
        self.telemetry.record_residual(started, 1);
        Ok(())
    }

    /// Executes the Jacobian-values callback owned by this prepared plan.
    ///
    /// The linked backend remains responsible for fixed-CSC or compact-Banded
    /// output layout validation; this plan adds the prepared ABI and telemetry
    /// contract around that call.
    pub fn try_execute_bound_jacobian_callback(
        &self,
        args: &[f64],
        out: &mut [f64],
    ) -> Result<(), AtomAotRuntimeError> {
        self.validate_runtime_input(args)?;
        let backend = self.runtime_binding.linked_sparse().ok_or_else(|| {
            self.telemetry.record_error();
            AtomAotRuntimeError::RuntimeNotLinked
        })?;
        let started = self.telemetry.start_timing();
        backend
            .try_jacobian_values_eval(args, out)
            .map_err(map_linked_callback_error)?;
        self.validate_jacobian_output(out)?;
        self.telemetry.record_jacobian(started, 1);
        Ok(())
    }

    fn validate_output(
        &self,
        stage: &'static str,
        out: &[f64],
        expected: usize,
    ) -> Result<(), AtomAotRuntimeError> {
        let result = if out.len() != expected {
            Err(AtomAotRuntimeError::OutputLength {
                stage,
                expected,
                actual: out.len(),
            })
        } else if let Some(index) = out.iter().position(|value| !value.is_finite()) {
            Err(AtomAotRuntimeError::NonFiniteOutput { stage, index })
        } else {
            Ok(())
        };
        if result.is_err() {
            self.telemetry.record_error();
        }
        result
    }

    /// Adds an already measured cold stage to this owned plan.
    pub fn record_cold_stage_duration(
        &self,
        stage: super::aot_telemetry::BvpAotColdStage,
        elapsed: Duration,
    ) {
        self.telemetry.record_cold_stage_duration(stage, elapsed);
    }

    /// Records a typed artifact lifecycle event without exposing telemetry
    /// internals to codegen adapters.
    pub fn record_lifecycle(&self, event: super::aot_telemetry::BvpAotLifecycleEvent) {
        self.telemetry.record_lifecycle(event);
    }

    /// Records a lifecycle event and emits the optional debug log entry.
    pub fn record_lifecycle_with_log(
        &self,
        event: super::aot_telemetry::BvpAotLifecycleEvent,
        artifact_key: &str,
    ) {
        self.telemetry
            .record_lifecycle_with_log(event, artifact_key);
    }

    /// Aggregates actual worker/chunk dispatch without retaining per-thread
    /// records. This is called by the linked runtime only when a parallel
    /// policy was selected, so the disabled path remains untouched.
    pub fn record_worker_batch(&self, workers: usize, chunks: usize) {
        self.telemetry.record_worker_batch(workers, chunks);
    }

    /// Records a known prepared-buffer allocation at a cold lifecycle edge.
    pub fn record_allocation(&self, bytes: usize) {
        self.telemetry.record_allocation(bytes);
    }

    /// Records a known adapter copy and its size at a lifecycle edge.
    pub fn record_copy_bytes(&self, bytes: usize) {
        self.telemetry.record_copy_bytes(bytes);
    }
}

fn callback_panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "callback panicked with a non-string payload".to_string()
    }
}

fn map_linked_callback_error(error: LinkedAotCallbackError) -> AtomAotRuntimeError {
    match error {
        LinkedAotCallbackError::OutputLength {
            stage,
            expected,
            actual,
        } => AtomAotRuntimeError::OutputLength {
            stage,
            expected,
            actual,
        },
        LinkedAotCallbackError::NonFiniteInput { index, .. } => {
            AtomAotRuntimeError::NonFiniteInput { index }
        }
        LinkedAotCallbackError::NonFiniteOutput { stage, index } => {
            AtomAotRuntimeError::NonFiniteOutput { stage, index }
        }
        LinkedAotCallbackError::Panicked { stage } => AtomAotRuntimeError::CallbackPanicked {
            stage,
            message: "linked callback panicked".to_string(),
        },
        other => AtomAotRuntimeError::LinkedCallbackFailure {
            stage: linked_error_stage(&other),
            message: other.to_string(),
        },
    }
}

fn linked_error_stage(error: &LinkedAotCallbackError) -> &'static str {
    match error {
        LinkedAotCallbackError::ChunkIndex { stage, .. }
        | LinkedAotCallbackError::ChunkLayout { stage, .. }
        | LinkedAotCallbackError::OutputLength { stage, .. }
        | LinkedAotCallbackError::InvalidLayout { stage, .. }
        | LinkedAotCallbackError::NonFiniteInput { stage, .. }
        | LinkedAotCallbackError::Panicked { stage }
        | LinkedAotCallbackError::NonFiniteOutput { stage, .. } => *stage,
    }
}

fn validate_plan(
    residuals: &[Atom],
    entries: &[SparseAtomJacobianEntry],
    input_names: &[String],
    input_symbols: &[Symbol],
    (rows, cols): (usize, usize),
    layout: &AtomAotMatrixLayout,
) -> Result<(), AtomAotPlanError> {
    if residuals.is_empty() {
        return Err(AtomAotPlanError::EmptyResidualSystem);
    }
    if cols == 0 || input_names.is_empty() {
        return Err(AtomAotPlanError::EmptyVariableSchema);
    }
    if input_names.len() != input_symbols.len() {
        return Err(AtomAotPlanError::InputSchemaMismatch {
            names: input_names.len(),
            symbols: input_symbols.len(),
        });
    }
    if residuals.len() != rows {
        return Err(AtomAotPlanError::ShapeMismatch {
            residuals: residuals.len(),
            rows,
            cols,
        });
    }
    if let Some(entry) = entries
        .iter()
        .find(|entry| entry.row >= rows || entry.col >= cols)
    {
        return Err(AtomAotPlanError::InvalidJacobianCoordinate {
            row: entry.row,
            col: entry.col,
            rows,
            cols,
        });
    }
    let mut coordinates = entries
        .iter()
        .map(|entry| (entry.row, entry.col))
        .collect::<Vec<_>>();
    coordinates.sort_unstable();
    if let Some(duplicate) = coordinates.windows(2).find(|window| window[0] == window[1]) {
        return Err(AtomAotPlanError::DuplicateJacobianCoordinate {
            row: duplicate[0].0,
            col: duplicate[0].1,
        });
    }
    if let AtomAotMatrixLayout::Banded { kl, ku, .. }
    | AtomAotMatrixLayout::BandedCompact { kl, ku, .. } = layout
    {
        if *kl >= rows || *ku >= cols {
            return Err(AtomAotPlanError::InvalidBandedLayout {
                kl: *kl,
                ku: *ku,
                rows,
                cols,
            });
        }
        if let Some(entry) = entries.iter().find(|entry| {
            entry.row.saturating_add(*ku) < entry.col || entry.col.saturating_add(*kl) < entry.row
        }) {
            return Err(AtomAotPlanError::JacobianEntryOutsideBand {
                row: entry.row,
                col: entry.col,
                kl: *kl,
                ku: *ku,
            });
        }
        if matches!(layout, AtomAotMatrixLayout::BandedCompact { .. }) {
            let expected_slots =
                AtomAotBandedSlotMap::new(rows, cols, *kl, *ku).map(|map| map.storage_len())?;
            let actual_slots = layout.value_count();
            if actual_slots != expected_slots {
                return Err(AtomAotPlanError::JacobianValueCountMismatch {
                    entries: actual_slots,
                    expected: expected_slots,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::View::parser::parse;

    fn atom(source: &str) -> Atom {
        parse(source).expect("test atom must parse")
    }

    fn entry(row: usize, col: usize, source: &str) -> SparseAtomJacobianEntry {
        SparseAtomJacobianEntry {
            row,
            col,
            value: atom(source),
        }
    }

    #[test]
    fn sparse_plan_owns_atom_payload_and_ordered_input_schema() {
        let plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x + p"), atom("y - p")],
            vec![entry(0, 0, "1"), entry(1, 1, "1")],
            vec!["p".to_string(), "x".to_string(), "y".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("p")),
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
            ],
            1,
            AtomAotMatrixLayout::SparseCsc {
                rows: 2,
                cols: 2,
                nnz: 2,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("valid AtomView AOT plan")
        .with_evaluator_policy(BvpAotEvaluatorPolicy::Parallel);

        assert_eq!(plan.parameter_count(), 1);
        assert_eq!(plan.input_names(), &["p", "x", "y"]);
        assert_eq!(plan.residuals().len(), 2);
        assert_eq!(plan.jacobian_entries().len(), 2);
        assert_eq!(plan.matrix_layout().value_count(), 2);
        assert_eq!(
            plan.telemetry_snapshot().identity.evaluator_policy,
            BvpAotEvaluatorPolicy::Parallel
        );
        assert!(!plan.runtime_binding().is_bound());
    }

    #[test]
    fn runtime_binding_replacement_invalidates_stale_linked_owner() {
        let mut plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x")],
            vec![entry(0, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("valid AtomView AOT plan");
        let backend =
            crate::symbolic::codegen::codegen_aot_runtime_link::LinkedSparseAotBackend::new(
                "runtime-binding-fixture",
                1,
                (1, 1),
                1,
                std::sync::Arc::new(|_, out| out.fill(0.0)),
                std::sync::Arc::new(|_, out| out.fill(1.0)),
            );

        let initial_generation = plan.runtime_binding().generation();
        plan.bind_linked_runtime(None, backend);
        assert!(plan.runtime_binding().is_bound());
        assert_eq!(plan.runtime_binding().state(), AtomAotRuntimeState::Linked);
        assert!(plan.runtime_binding().generation() > initial_generation);

        let bound_generation = plan.runtime_binding().generation();
        plan.invalidate_linked_runtime();
        assert!(!plan.runtime_binding().is_bound());
        assert!(plan.runtime_binding().generation() > bound_generation);
        assert!(plan.runtime_binding().artifact().is_none());
        assert_eq!(plan.runtime_binding().state(), AtomAotRuntimeState::Unbound);
    }

    #[test]
    fn runtime_binding_state_distinguishes_artifact_from_linked_callback() {
        let mut plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x")],
            vec![entry(0, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("valid AtomView AOT plan");

        assert_eq!(plan.runtime_binding().state(), AtomAotRuntimeState::Unbound);

        let artifact = crate::symbolic::codegen::codegen_aot_resolution::AotResolver::new(
            crate::symbolic::codegen::codegen_aot_registry::AotRegistry::new(),
        )
        .resolve_by_problem_key("runtime-state-fixture");
        plan.bind_artifact(Some(artifact));
        assert_eq!(
            plan.runtime_binding().state(),
            AtomAotRuntimeState::ArtifactOnly
        );
        assert!(!plan.runtime_binding().is_bound());

        let backend =
            crate::symbolic::codegen::codegen_aot_runtime_link::LinkedSparseAotBackend::new(
                "runtime-state-fixture",
                1,
                (1, 1),
                1,
                std::sync::Arc::new(|_, out| out.fill(0.0)),
                std::sync::Arc::new(|_, out| out.fill(1.0)),
            );
        plan.bind_linked_runtime(plan.runtime_binding().artifact().cloned(), backend);
        assert_eq!(plan.runtime_binding().state(), AtomAotRuntimeState::Linked);

        let mut residual = [0.0];
        plan.try_execute_bound_residual_callback(&[2.0], &mut residual)
            .expect("owned linked residual callback should be callable");
        assert_eq!(residual, [0.0]);
        let mut jacobian = [0.0];
        plan.try_execute_bound_jacobian_callback(&[2.0], &mut jacobian)
            .expect("owned linked Jacobian callback should be callable");
        assert_eq!(jacobian, [1.0]);

        plan.bind_artifact(None);
        assert_eq!(plan.runtime_binding().state(), AtomAotRuntimeState::Unbound);
        assert!(!plan.runtime_binding().is_bound());
        assert!(plan.runtime_binding().artifact().is_none());
        assert!(matches!(
            plan.try_execute_bound_residual_callback(&[2.0], &mut residual),
            Err(AtomAotRuntimeError::RuntimeNotLinked)
        ));
    }

    #[test]
    fn detailed_plan_telemetry_records_owned_buffers_without_hot_path_maps() {
        let plan = AtomAotPreparedPlan::from_parts_with_telemetry(
            vec![atom("x")],
            vec![entry(0, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
            BvpAotTelemetryMode::Detailed,
        )
        .expect("valid detailed AtomView AOT plan");

        let snapshot = plan.telemetry_snapshot();
        assert_eq!(snapshot.mode, BvpAotTelemetryMode::Detailed);
        assert_eq!(snapshot.allocation_events, 1);
        assert!(snapshot.allocation_bytes > 0);
        assert_eq!(snapshot.copies, 0);
    }

    #[test]
    fn plan_rejects_invalid_coordinates_before_codegen() {
        let error = AtomAotPreparedPlan::from_parts(
            vec![atom("x")],
            vec![entry(1, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect_err("out-of-bounds Atom entry must be typed");

        assert!(matches!(
            error,
            AtomAotPlanError::InvalidJacobianCoordinate { .. }
        ));
    }

    #[test]
    fn plan_rejects_jacobian_entry_outside_declared_band() {
        let error = AtomAotPreparedPlan::from_parts(
            vec![atom("x"), atom("y")],
            vec![entry(0, 1, "1")],
            vec!["x".to_string(), "y".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
            ],
            0,
            AtomAotMatrixLayout::Banded {
                rows: 2,
                cols: 2,
                kl: 0,
                ku: 0,
                slots: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect_err("an off-diagonal entry must not fit a diagonal-only layout");

        assert_eq!(
            error,
            AtomAotPlanError::JacobianEntryOutsideBand {
                row: 0,
                col: 1,
                kl: 0,
                ku: 0,
            }
        );
        assert!(error.to_string().contains("outside band"));
    }

    #[test]
    fn plan_rejects_parameter_count_outside_input_schema() {
        let error = AtomAotPreparedPlan::from_parts(
            vec![atom("x")],
            vec![entry(0, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            2,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect_err("parameter prefix cannot exceed the flattened input schema");

        assert_eq!(
            error,
            AtomAotPlanError::ParameterCountOutOfRange {
                parameters: 2,
                inputs: 1,
            }
        );
    }

    #[test]
    fn banded_layout_is_explicit_and_not_sparse_fallback() {
        let layout = AtomAotMatrixLayout::Banded {
            rows: 3,
            cols: 3,
            kl: 1,
            ku: 1,
            slots: 3,
        };

        assert_eq!(layout.shape(), (3, 3));
        assert_eq!(layout.value_count(), 3);
        assert_ne!(
            layout,
            AtomAotMatrixLayout::SparseCsc {
                rows: 3,
                cols: 3,
                nnz: 3,
            }
        );
    }

    #[test]
    fn native_banded_slot_map_matches_lapack_compact_coordinates() {
        let map = AtomAotBandedSlotMap::new(3, 3, 1, 1).expect("valid compact band map");

        assert_eq!(map.storage_len(), 9);
        assert_eq!(
            map.slots()
                .iter()
                .map(|slot| slot.matrix_row)
                .collect::<Vec<_>>(),
            vec![
                None,
                Some(0),
                Some(1),
                Some(0),
                Some(1),
                Some(2),
                Some(1),
                Some(2),
                None,
            ]
        );
        assert_eq!(map.storage_index(0, 0), Some(3));
        assert_eq!(map.storage_index(0, 1), Some(1));
        assert_eq!(map.storage_index(2, 1), Some(7));
        assert_eq!(map.storage_index(0, 2), None);
    }

    #[test]
    fn native_banded_packing_matches_banded_assembly_storage() {
        let map = AtomAotBandedSlotMap::new(3, 3, 1, 1).expect("valid compact band map");
        let coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)];
        let values = [10.0, 11.0, 20.0, 21.0, 22.0, 31.0, 32.0];
        let packed = map
            .pack_values(&coordinates, &values)
            .expect("band values should pack");

        let mut assembly =
            crate::somelinalg::banded::banded_assembly::BandedAssembly::zeros(3, 1, 1)
                .expect("valid band assembly");
        for ((row, col), value) in coordinates.iter().copied().zip(values) {
            assembly
                .set(row, col, value)
                .expect("coordinate is in band");
        }
        let expected = assembly
            .to_banded()
            .expect("compact band conversion should succeed");
        assert_eq!(packed, expected.as_slice());
    }

    #[test]
    fn native_banded_packing_rejects_duplicates_and_out_of_band_coordinates() {
        let map = AtomAotBandedSlotMap::new(3, 3, 1, 1).expect("valid compact band map");

        assert_eq!(
            map.pack_values(&[(1, 1), (1, 1)], &[1.0, 2.0]),
            Err(AtomAotPlanError::DuplicateJacobianCoordinate { row: 1, col: 1 })
        );
        assert_eq!(
            map.pack_values(&[(0, 2)], &[1.0]),
            Err(AtomAotPlanError::JacobianEntryOutsideBand {
                row: 0,
                col: 2,
                kl: 1,
                ku: 1,
            })
        );
        assert_eq!(
            map.pack_values(&[(3, 0)], &[1.0]),
            Err(AtomAotPlanError::InvalidJacobianCoordinate {
                row: 3,
                col: 0,
                rows: 3,
                cols: 3,
            })
        );
    }

    #[test]
    fn compact_banded_chunk_ownership_covers_boundary_slots_exactly_once() {
        let map = AtomAotBandedSlotMap::new(5, 5, 1, 1).expect("valid compact band map");
        let ownership = map
            .chunk_ownership(4)
            .expect("compact storage should split into owned ranges");

        assert_eq!(ownership.storage_len(), 15);
        assert_eq!(ownership.ranges().len(), 4);
        assert_eq!(ownership.ranges()[0], 0..4);
        assert_eq!(ownership.ranges()[3], 12..15);
        for storage_index in 0..ownership.storage_len() {
            assert!(ownership.owner_of(storage_index).is_some());
        }
        assert_eq!(ownership.owner_of(0), Some(0));
        assert_eq!(ownership.owner_of(14), Some(3));
        ownership
            .validate()
            .expect("ownership must have no gaps or overlaps");
    }

    #[test]
    fn compact_banded_chunk_ownership_caps_workers_and_rejects_empty_requests() {
        let map = AtomAotBandedSlotMap::new(2, 2, 1, 1).expect("valid compact band map");
        let ownership = map
            .chunk_ownership(usize::MAX)
            .expect("worker count should be capped by storage length");
        assert_eq!(ownership.ranges(), &[0..1, 1..2, 2..3, 3..4, 4..5, 5..6]);
        assert_eq!(
            map.chunk_ownership(0),
            Err(AtomAotPlanError::InvalidChunkCount {
                storage_len: 6,
                requested_chunks: 0,
            })
        );
    }

    #[test]
    fn prepared_banded_plan_exposes_native_storage_without_changing_callback_abi() {
        let plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x"), atom("y"), atom("z")],
            vec![
                entry(0, 0, "1"),
                entry(0, 1, "2"),
                entry(1, 0, "3"),
                entry(1, 1, "4"),
                entry(1, 2, "5"),
                entry(2, 1, "6"),
                entry(2, 2, "7"),
            ],
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
                Symbol::new(crate::wrap_symbol!("z")),
            ],
            0,
            AtomAotMatrixLayout::Banded {
                rows: 3,
                cols: 3,
                kl: 1,
                ku: 1,
                slots: 7,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("valid explicit-entry Banded plan");

        assert_eq!(plan.matrix_layout().value_count(), 7);
        assert_eq!(plan.native_banded_slot_map().unwrap().storage_len(), 9);
        assert_eq!(
            plan.pack_banded_values_for_native_storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                .expect("native band packing should succeed"),
            vec![0.0, 2.0, 5.0, 1.0, 4.0, 7.0, 3.0, 6.0, 0.0]
        );
    }

    #[test]
    fn prepared_compact_banded_plan_requires_complete_callback_storage() {
        let plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x"), atom("y"), atom("z")],
            vec![
                entry(0, 0, "1"),
                entry(0, 1, "2"),
                entry(1, 0, "3"),
                entry(1, 1, "4"),
                entry(1, 2, "5"),
                entry(2, 1, "6"),
                entry(2, 2, "7"),
            ],
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
                Symbol::new(crate::wrap_symbol!("z")),
            ],
            0,
            AtomAotMatrixLayout::BandedCompact {
                rows: 3,
                cols: 3,
                kl: 1,
                ku: 1,
                slots: 9,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("compact full-slot plan should validate");

        assert_eq!(plan.matrix_layout().value_count(), 9);
        plan.validate_jacobian_output(&[0.0; 9])
            .expect("full compact storage should satisfy the callback ABI");
        assert!(matches!(
            plan.validate_jacobian_output(&[0.0; 7]),
            Err(AtomAotRuntimeError::OutputLength {
                stage: "Jacobian",
                expected: 9,
                actual: 7,
            })
        ));
    }

    #[test]
    fn prepared_plan_rejects_duplicate_fixed_csc_coordinates() {
        let error = AtomAotPreparedPlan::from_parts(
            vec![atom("x"), atom("y")],
            vec![entry(0, 0, "1"), entry(0, 0, "2")],
            vec!["x".to_string(), "y".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
            ],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 2,
                cols: 2,
                nnz: 2,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect_err("fixed CSC layout must not contain duplicate coordinates");

        assert_eq!(
            error,
            AtomAotPlanError::DuplicateJacobianCoordinate { row: 0, col: 0 }
        );
    }

    #[test]
    fn runtime_contract_rejects_bad_input_and_output_shapes() {
        let plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x + p"), atom("y - p")],
            vec![entry(0, 0, "1"), entry(1, 1, "1")],
            vec!["p".to_string(), "x".to_string(), "y".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("p")),
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
            ],
            1,
            AtomAotMatrixLayout::SparseCsc {
                rows: 2,
                cols: 2,
                nnz: 2,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("valid AtomView AOT plan");

        assert!(matches!(
            plan.validate_runtime_input(&[1.0, 2.0]),
            Err(AtomAotRuntimeError::InputLength {
                expected: 3,
                actual: 2
            })
        ));
        assert!(matches!(
            plan.validate_runtime_input(&[1.0, f64::NAN, 2.0]),
            Err(AtomAotRuntimeError::NonFiniteInput { index: 1 })
        ));
        assert!(matches!(
            plan.validate_residual_output(&[0.0]),
            Err(AtomAotRuntimeError::OutputLength {
                stage: "residual",
                expected: 2,
                actual: 1
            })
        ));
        assert!(matches!(
            plan.validate_jacobian_output(&[0.0, f64::INFINITY]),
            Err(AtomAotRuntimeError::NonFiniteOutput {
                stage: "Jacobian",
                index: 1
            })
        ));
    }

    #[test]
    fn runtime_contract_accepts_finite_fixed_layout_buffers() {
        let plan = AtomAotPreparedPlan::from_parts(
            vec![atom("x")],
            vec![entry(0, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
        )
        .expect("valid AtomView AOT plan");

        assert!(plan.validate_runtime_input(&[2.0]).is_ok());
        assert!(plan.validate_residual_output(&[0.0]).is_ok());
        assert!(plan.validate_jacobian_output(&[1.0]).is_ok());
    }

    #[test]
    fn typed_execution_boundary_validates_callbacks_and_records_warm_stages() {
        let plan = AtomAotPreparedPlan::from_parts_with_telemetry(
            vec![atom("x + p"), atom("y - p")],
            vec![entry(0, 0, "1"), entry(1, 1, "1")],
            vec!["p".to_string(), "x".to_string(), "y".to_string()],
            vec![
                Symbol::new(crate::wrap_symbol!("p")),
                Symbol::new(crate::wrap_symbol!("x")),
                Symbol::new(crate::wrap_symbol!("y")),
            ],
            1,
            AtomAotMatrixLayout::SparseCsc {
                rows: 2,
                cols: 2,
                nnz: 2,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
            BvpAotTelemetryMode::Counters,
        )
        .expect("valid AtomView AOT plan");

        let mut residual = vec![0.0; 2];
        plan.try_execute_residual_callback(&[1.0, 2.0, 3.0], &mut residual, |_, out| {
            out.copy_from_slice(&[5.0, 6.0]);
            Ok::<(), AtomAotRuntimeError>(())
        })
        .expect("typed residual callback should succeed");

        let mut jacobian = vec![0.0; 2];
        plan.try_execute_jacobian_callback_with_chunks(
            &[1.0, 2.0, 3.0],
            &mut jacobian,
            2,
            |_, out| {
                out.copy_from_slice(&[7.0, 8.0]);
                Ok::<(), AtomAotRuntimeError>(())
            },
        )
        .expect("typed Jacobian callback should succeed");

        assert_eq!(residual, [5.0, 6.0]);
        assert_eq!(jacobian, [7.0, 8.0]);
        let snapshot = plan.telemetry_snapshot();
        assert_eq!(snapshot.residual_calls, 1);
        assert_eq!(snapshot.jacobian_calls, 1);
        assert_eq!(snapshot.jacobian_chunks, 2);
        assert_eq!(snapshot.errors, 0);
    }

    #[test]
    fn typed_execution_boundary_does_not_publish_failed_callback() {
        let plan = AtomAotPreparedPlan::from_parts_with_telemetry(
            vec![atom("x")],
            vec![entry(0, 0, "1")],
            vec!["x".to_string()],
            vec![Symbol::new(crate::wrap_symbol!("x"))],
            0,
            AtomAotMatrixLayout::SparseCsc {
                rows: 1,
                cols: 1,
                nnz: 1,
            },
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
            BvpAotTelemetryMode::Counters,
        )
        .expect("valid AtomView AOT plan");

        let mut output = vec![0.0; 1];
        let result = plan.try_execute_residual_callback(&[2.0], &mut output, |_, out| {
            out[0] = f64::NAN;
            Ok::<(), AtomAotRuntimeError>(())
        });

        assert!(matches!(
            result,
            Err(AtomAotRuntimeError::NonFiniteOutput {
                stage: "residual",
                index: 0
            })
        ));
        let snapshot = plan.telemetry_snapshot();
        assert_eq!(snapshot.residual_calls, 0);
        assert_eq!(snapshot.errors, 1);

        let panic_result = plan.try_execute_residual_callback(&[2.0], &mut output, |_, _| {
            panic!("synthetic generated callback failure")
        });
        assert!(matches!(
            panic_result,
            Err(AtomAotRuntimeError::CallbackPanicked {
                stage: "residual",
                ..
            })
        ));
        assert_eq!(plan.telemetry_snapshot().errors, 2);
    }
}
