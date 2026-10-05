//! Dense, sparse, and banded shifted linear-system contracts.
//!
//! Radau's order-five Newton solve is not one linear solve: every step needs
//! one real-shifted system and one complex-shifted system.  This module keeps
//! both factorizations in backend-owned storage so the step code does not
//! accidentally rebuild a dense surrogate or allocate a new matrix per
//! Newton iteration.  The backend is selected once during preparation; the
//! hot path then dispatches only on the already-selected enum variant.

use super::coefficients::RadauIia5;
use super::config::RadauMatrixLayout;
use super::error::{RadauConfigError, RadauError, RadauStage, RadauUnsupportedRoute};
use super::telemetry::{RadauLinearStage, RadauTelemetry};
use crate::somelinalg::banded::lapack_style_banded::LapackStyleBandedLuFaithful;
use crate::somelinalg::banded::solver_traits::{DirectLinearSolver, FaerSparseLuSolver};
use faer::sparse::Triplet;

/// Backend-specific reusable storage.
///
/// The numerical step selects one variant during preparation.  Each variant
/// owns the numeric buffers and factorization handles required by its native
/// layout; in particular, Sparse and Banded never pass through Dense storage.
#[derive(Debug, Default)]
/// Reusable dense storage for real and complex shifted Radau systems.
pub(crate) struct DenseLinearWorkspace {
    /// Real shifted matrix/factorization storage.
    pub(crate) real: Vec<f64>,
    /// Real part of the complex shifted matrix/factorization.
    pub(crate) complex_real: Vec<f64>,
    /// Imaginary part of the complex shifted matrix/factorization.
    pub(crate) complex_imag: Vec<f64>,
    /// Real factorization pivots.
    pub(crate) real_pivots: Vec<usize>,
    /// Complex factorization pivots.
    pub(crate) complex_pivots: Vec<usize>,
    /// Reusable real right-hand side.
    pub(crate) real_rhs: Vec<f64>,
    /// Real part of the complex right-hand side.
    pub(crate) complex_rhs_real: Vec<f64>,
    /// Imaginary part of the complex right-hand side.
    pub(crate) complex_rhs_imag: Vec<f64>,
}

impl DenseLinearWorkspace {
    /// Resize all dense matrix, pivot, and right-hand-side buffers together.
    pub(crate) fn resize_for_dimension(&mut self, dimension: usize) -> Result<(), RadauError> {
        let matrix_len = checked_matrix_len(dimension)?;
        self.real.resize(matrix_len, 0.0);
        self.complex_real.resize(matrix_len, 0.0);
        self.complex_imag.resize(matrix_len, 0.0);
        self.real_pivots.resize(dimension, 0);
        self.complex_pivots.resize(dimension, 0);
        self.real_rhs.resize(dimension, 0.0);
        self.complex_rhs_real.resize(dimension, 0.0);
        self.complex_rhs_imag.resize(dimension, 0.0);
        Ok(())
    }
}

#[derive(Debug)]
/// Compact banded storage owned by the prepared banded backend.
pub(crate) struct BandedLinearWorkspace {
    /// Shape metadata used to validate callbacks and compute compact slots.
    pub(crate) dimension: usize,
    pub(crate) lower: usize,
    pub(crate) upper: usize,
    /// Current callback Jacobian in LAPACK-style `(upper + lower + 1) * n`
    /// column-major band storage.
    pub(crate) jacobian: Vec<f64>,
    /// Real shifted matrix in the same compact layout, before factorization.
    pub(crate) real: Vec<f64>,
    /// Real component of the complex shifted matrix.
    pub(crate) complex_real: Vec<f64>,
    /// Imaginary component of the complex shifted matrix.
    pub(crate) complex_imag: Vec<f64>,
    /// Reusable compact storage for the interleaved real complex block.
    /// Keeping this buffer in the workspace avoids a temporary `Banded`
    /// allocation on every numeric factorization.
    pub(crate) complex_block: Vec<f64>,
    pub(crate) real_rhs: Vec<f64>,
    pub(crate) complex_rhs_real: Vec<f64>,
    pub(crate) complex_rhs_imag: Vec<f64>,
    /// Interleaved real block RHS used by the structured complex solve.
    pub(crate) complex_block_rhs: Vec<f64>,
    /// Factorization of the real compact band matrix.
    /// Factorization is optional because assembly invalidates the previous one.
    pub(crate) real_lu: Option<LapackStyleBandedLuFaithful>,
    /// Factorization of the interleaved real block form of the complex system.
    pub(crate) complex_block_lu: Option<LapackStyleBandedLuFaithful>,
}

impl BandedLinearWorkspace {
    /// Allocate compact banded storage for a matrix of the requested shape.
    pub(crate) fn new(dimension: usize, lower: usize, upper: usize) -> Result<Self, RadauError> {
        let slots = lower
            .checked_add(upper)
            .and_then(|value| value.checked_add(1))
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        let storage_len = slots
            .checked_mul(dimension)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        let block_dimension = dimension
            .checked_mul(2)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        let block_lower = lower
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        let block_upper = upper
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        let block_storage_len = block_lower
            .checked_add(block_upper)
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_mul(block_dimension))
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        Ok(Self {
            dimension,
            lower,
            upper,
            jacobian: vec![0.0; storage_len],
            real: vec![0.0; storage_len],
            complex_real: vec![0.0; storage_len],
            complex_imag: vec![0.0; storage_len],
            complex_block: vec![0.0; block_storage_len],
            real_rhs: vec![0.0; dimension],
            complex_rhs_real: vec![0.0; dimension],
            complex_rhs_imag: vec![0.0; dimension],
            complex_block_rhs: vec![
                0.0;
                dimension
                    .checked_mul(2)
                    .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?
            ],
            real_lu: None,
            complex_block_lu: None,
        })
    }

    /// Return the compact-storage slot for a matrix coordinate, if in-band.
    pub(crate) fn slot(&self, row: usize, column: usize) -> Option<usize> {
        if row >= self.dimension || column >= self.dimension {
            return None;
        }
        let row_with_upper = row.checked_add(self.upper)?;
        let column_with_lower = column.checked_add(self.lower)?;
        (row_with_upper >= column && column_with_lower >= row)
            .then(|| (self.upper + row - column) * self.dimension + column)
    }
}

#[derive(Debug)]
/// CSC-pattern storage that avoids a dense staging matrix for sparse routes.
///
/// `column_offsets` and `row_indices` are structural data and survive across
/// steps.  Only the numeric value arrays and factor handles change when a new
/// Jacobian or step size is assembled.
pub(crate) struct SparseLinearWorkspace {
    pub(crate) dimension: usize,
    pub(crate) column_offsets: Vec<usize>,
    pub(crate) row_indices: Vec<usize>,
    pub(crate) jacobian: Vec<f64>,
    pub(crate) real: Vec<f64>,
    pub(crate) complex_real: Vec<f64>,
    pub(crate) complex_imag: Vec<f64>,
    pub(crate) real_rhs: Vec<f64>,
    pub(crate) complex_rhs_real: Vec<f64>,
    pub(crate) complex_rhs_imag: Vec<f64>,
    /// Interleaved real block RHS used by the structured complex solve.
    pub(crate) complex_block_rhs: Vec<f64>,
    /// Reused triplet staging for the numeric real factorization.
    pub(crate) real_triplets: Vec<Triplet<usize, usize, f64>>,
    /// Reused triplet staging for the numeric complex block factorization.
    pub(crate) complex_triplets: Vec<Triplet<usize, usize, f64>>,
    /// Numeric CSC factorization for the real shifted system.
    pub(crate) real_lu: Option<FaerSparseLuSolver>,
    /// Numeric CSC factorization of the interleaved real block complex system.
    pub(crate) complex_block_lu: Option<FaerSparseLuSolver>,
}

impl SparseLinearWorkspace {
    /// Build immutable CSC index storage and reusable numeric value buffers.
    pub(crate) fn from_pattern(
        dimension: usize,
        entries: &[(usize, usize)],
    ) -> Result<Self, RadauError> {
        let mut pattern = entries.to_vec();
        if pattern
            .iter()
            .any(|&(row, column)| row >= dimension || column >= dimension)
        {
            return Err(RadauConfigError::SparsePatternOutOfBounds.into());
        }
        for index in 0..dimension {
            pattern.push((index, index));
        }
        pattern.sort_unstable_by_key(|&(row, column)| (column, row));
        pattern.dedup();
        let offset_len = dimension
            .checked_add(1)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        let mut column_offsets = vec![0usize; offset_len];
        for &(_, column) in &pattern {
            column_offsets[column + 1] += 1;
        }
        for column in 0..dimension {
            column_offsets[column + 1] += column_offsets[column];
        }
        let value_len = pattern.len();
        let row_indices = pattern.into_iter().map(|(row, _)| row).collect::<Vec<_>>();
        Ok(Self {
            dimension,
            column_offsets,
            row_indices,
            jacobian: vec![0.0; value_len],
            real: vec![0.0; value_len],
            complex_real: vec![0.0; value_len],
            complex_imag: vec![0.0; value_len],
            real_rhs: vec![0.0; dimension],
            complex_rhs_real: vec![0.0; dimension],
            complex_rhs_imag: vec![0.0; dimension],
            complex_block_rhs: vec![
                0.0;
                dimension
                    .checked_mul(2)
                    .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?
            ],
            real_triplets: Vec::with_capacity(value_len),
            complex_triplets: Vec::with_capacity(value_len * 4),
            real_lu: None,
            complex_block_lu: None,
        })
    }

    /// Find the numeric value slot for a `(row, column)` entry.
    pub(crate) fn value_index(&self, row: usize, column: usize) -> Option<usize> {
        let start = *self.column_offsets.get(column)?;
        let end = *self.column_offsets.get(column + 1)?;
        self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map(|offset| start + offset)
    }
}

#[derive(Debug)]
/// Backend-specific workspace selected once during model preparation.
pub(crate) enum RadauLinearWorkspace {
    Dense(DenseLinearWorkspace),
    Banded(BandedLinearWorkspace),
    Sparse(SparseLinearWorkspace),
}

/// Operations required by a Radau linear backend.
///
/// Implementations receive values in their native storage format; conversion
/// through a dense temporary is deliberately outside this trait.
pub(crate) trait LinearSystemBackend {
    type Workspace;

    fn layout(&self) -> RadauMatrixLayout;
    fn create_workspace(&self, dimension: usize) -> Result<Self::Workspace, RadauError>;
    /// Form `mu/h * I - J` in the backend's native storage.
    ///
    /// This is deliberately separate from `factor`: the same assembled
    /// matrix is factored once and reused by all Newton iterations of the
    /// current step attempt.
    fn assemble_shifted(
        &self,
        jacobian: JacobianValues<'_>,
        h: f64,
        coefficients: &RadauIia5,
        workspace: &mut Self::Workspace,
    ) -> Result<(), RadauError>;
    /// Replace the numeric factorization owned by `workspace`.
    fn factor(&self, workspace: &mut Self::Workspace) -> Result<(), RadauError>;
    /// Solve the real shifted system using the factorization from `factor`.
    fn solve_real(
        &self,
        workspace: &mut Self::Workspace,
        rhs: &mut [f64],
    ) -> Result<(), RadauError>;
    /// Solve the complex shifted system without exposing a complex scalar
    /// type to the rest of the numerical core.
    fn solve_complex(
        &self,
        workspace: &mut Self::Workspace,
        rhs_real: &mut [f64],
        rhs_imag: &mut [f64],
    ) -> Result<(), RadauError>;
    /// Drop numeric factors while retaining allocated structural capacity.
    fn invalidate(&self, workspace: &mut Self::Workspace);
}

/// Jacobian values are supplied in the storage format selected at preparation.
///
/// The callback contract is part of the performance design: a Banded callback
/// writes compact band slots and a Sparse callback writes values for a fixed
/// `(row, column)` pattern.  A mismatched variant is an error, not a license
/// to silently allocate and convert a dense matrix.
#[derive(Debug, Clone, Copy)]
/// Jacobian values in the storage format selected by the caller.
pub(crate) enum JacobianValues<'a> {
    Dense {
        values: &'a [f64],
    },
    Banded {
        values: &'a [f64],
        lower: usize,
        upper: usize,
    },
    Sparse {
        values: &'a [f64],
        entries: &'a [(usize, usize)],
    },
}

#[derive(Debug, Clone, Copy, Default)]
/// Marker for the dense linear-system implementation.
pub(crate) struct DenseBackend;

#[derive(Debug, Clone, Copy)]
/// Compact banded linear-system implementation and its configured bandwidth.
pub(crate) struct BandedBackend {
    pub(crate) lower: usize,
    pub(crate) upper: usize,
}

#[derive(Debug, Clone)]
/// Sparse linear-system implementation and its prepared structural pattern.
pub(crate) struct SparseBackend {
    pub(crate) pattern: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
/// Backend variant selected once from `RadauMatrixLayout`.
///
/// `PreparedLinearBackend` is intentionally small and cloneable.  It carries
/// immutable layout/pattern metadata, while mutable numeric state lives in
/// `RadauLinearWorkspace`; this separation makes continuation and retries
/// reuse capacity without sharing mutable factorizations.
pub(crate) enum PreparedLinearBackend {
    Dense(DenseBackend),
    Banded(BandedBackend),
    Sparse(SparseBackend),
}

impl PreparedLinearBackend {
    /// Construct the backend variant without converting matrix storage.
    pub(crate) fn from_layout(layout: RadauMatrixLayout) -> Self {
        Self::from_layout_and_pattern(layout, &[])
    }

    /// Construct a backend and retain the prepared sparse structural pattern.
    pub(crate) fn from_layout_and_pattern(
        layout: RadauMatrixLayout,
        pattern: &[(usize, usize)],
    ) -> Self {
        match layout {
            RadauMatrixLayout::Dense => Self::Dense(DenseBackend),
            RadauMatrixLayout::Banded { lower, upper } => {
                Self::Banded(BandedBackend { lower, upper })
            }
            RadauMatrixLayout::Sparse => Self::Sparse(SparseBackend {
                pattern: pattern.to_vec(),
            }),
        }
    }

    /// Return the storage layout owned by this backend.
    pub(crate) fn layout(&self) -> RadauMatrixLayout {
        match self {
            Self::Dense(backend) => backend.layout(),
            Self::Banded(backend) => backend.layout(),
            Self::Sparse(backend) => backend.layout(),
        }
    }

    /// Allocate the backend-specific reusable workspace.
    pub(crate) fn create_workspace(
        &self,
        dimension: usize,
    ) -> Result<RadauLinearWorkspace, RadauError> {
        match self {
            Self::Dense(backend) => Ok(RadauLinearWorkspace::Dense(
                backend.create_workspace(dimension)?,
            )),
            Self::Banded(backend) => Ok(RadauLinearWorkspace::Banded(
                backend.create_workspace(dimension)?,
            )),
            Self::Sparse(backend) => Ok(RadauLinearWorkspace::Sparse(
                backend.create_workspace(dimension)?,
            )),
        }
    }

    /// Assemble a shifted Jacobian directly into backend-owned storage.
    pub(crate) fn assemble_shifted_into(
        &self,
        jacobian: JacobianValues<'_>,
        h: f64,
        coefficients: &RadauIia5,
        workspace: &mut RadauLinearWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_linear(RadauLinearStage::JacobianAssembly, || {
            match (self, workspace) {
                (Self::Dense(backend), RadauLinearWorkspace::Dense(workspace)) => {
                    backend.assemble_shifted(jacobian, h, coefficients, workspace)
                }
                (Self::Banded(backend), RadauLinearWorkspace::Banded(workspace)) => {
                    backend.assemble_shifted(jacobian, h, coefficients, workspace)
                }
                (Self::Sparse(backend), RadauLinearWorkspace::Sparse(workspace)) => {
                    backend.assemble_shifted(jacobian, h, coefficients, workspace)
                }
                _ => Err(RadauConfigError::LinearWorkspaceMismatch.into()),
            }
        })
    }

    /// Factor the assembled shifted system in place when supported.
    pub(crate) fn factor_into(
        &self,
        workspace: &mut RadauLinearWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_linear(RadauLinearStage::Factorization, || {
            match (self, workspace) {
                (Self::Dense(backend), RadauLinearWorkspace::Dense(workspace)) => {
                    backend.factor(workspace)
                }
                (Self::Banded(backend), RadauLinearWorkspace::Banded(workspace)) => {
                    backend.factor(workspace)
                }
                (Self::Sparse(backend), RadauLinearWorkspace::Sparse(workspace)) => {
                    backend.factor(workspace)
                }
                _ => Err(RadauConfigError::LinearWorkspaceMismatch.into()),
            }
        })
    }

    /// Solve a real shifted system using the existing factorization.
    pub(crate) fn solve_real_into(
        &self,
        workspace: &mut RadauLinearWorkspace,
        rhs: &mut [f64],
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_linear(RadauLinearStage::RealSolve, || match (self, workspace) {
            (Self::Dense(backend), RadauLinearWorkspace::Dense(workspace)) => {
                backend.solve_real(workspace, rhs)
            }
            (Self::Banded(backend), RadauLinearWorkspace::Banded(workspace)) => {
                backend.solve_real(workspace, rhs)
            }
            (Self::Sparse(backend), RadauLinearWorkspace::Sparse(workspace)) => {
                backend.solve_real(workspace, rhs)
            }
            _ => Err(RadauConfigError::LinearWorkspaceMismatch.into()),
        })
    }

    /// Solve a complex shifted system using the existing factorization.
    pub(crate) fn solve_complex_into(
        &self,
        workspace: &mut RadauLinearWorkspace,
        rhs_real: &mut [f64],
        rhs_imag: &mut [f64],
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_linear(RadauLinearStage::ComplexSolve, || match (self, workspace) {
            (Self::Dense(backend), RadauLinearWorkspace::Dense(workspace)) => {
                backend.solve_complex(workspace, rhs_real, rhs_imag)
            }
            (Self::Banded(backend), RadauLinearWorkspace::Banded(workspace)) => {
                backend.solve_complex(workspace, rhs_real, rhs_imag)
            }
            (Self::Sparse(backend), RadauLinearWorkspace::Sparse(workspace)) => {
                backend.solve_complex(workspace, rhs_real, rhs_imag)
            }
            _ => Err(RadauConfigError::LinearWorkspaceMismatch.into()),
        })
    }

    /// Invalidate numeric state while retaining allocated workspace capacity.
    pub(crate) fn invalidate_into(
        &self,
        workspace: &mut RadauLinearWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_linear(RadauLinearStage::Invalidate, || match (self, workspace) {
            (Self::Dense(backend), RadauLinearWorkspace::Dense(workspace)) => {
                backend.invalidate(workspace);
                Ok(())
            }
            (Self::Banded(backend), RadauLinearWorkspace::Banded(workspace)) => {
                backend.invalidate(workspace);
                Ok(())
            }
            (Self::Sparse(backend), RadauLinearWorkspace::Sparse(workspace)) => {
                backend.invalidate(workspace);
                Ok(())
            }
            _ => Err(RadauConfigError::LinearWorkspaceMismatch.into()),
        })
    }
}

impl LinearSystemBackend for DenseBackend {
    type Workspace = DenseLinearWorkspace;

    fn layout(&self) -> RadauMatrixLayout {
        RadauMatrixLayout::Dense
    }

    fn create_workspace(&self, dimension: usize) -> Result<Self::Workspace, RadauError> {
        let mut workspace = DenseLinearWorkspace::default();
        workspace.resize_for_dimension(dimension)?;
        Ok(workspace)
    }

    fn assemble_shifted(
        &self,
        jacobian: JacobianValues<'_>,
        h: f64,
        coefficients: &RadauIia5,
        workspace: &mut Self::Workspace,
    ) -> Result<(), RadauError> {
        let values = match jacobian {
            JacobianValues::Dense { values } => values,
            _ => return Err(RadauUnsupportedRoute::DenseValuesRequired.into()),
        };
        let dimension = workspace.real_rhs.len();
        let matrix_len = checked_matrix_len(dimension)?;
        if values.len() != matrix_len {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected: matrix_len,
                actual: values.len(),
            });
        }
        let real_shift = coefficients.mu_real / h;
        let complex_shift_real = coefficients.mu_complex.0 / h;
        let complex_shift_imag = coefficients.mu_complex.1 / h;
        for (index, value) in values.iter().copied().enumerate() {
            let diagonal = index / dimension == index % dimension;
            workspace.real[index] = if diagonal { real_shift - value } else { -value };
            workspace.complex_real[index] = if diagonal {
                complex_shift_real - value
            } else {
                -value
            };
            workspace.complex_imag[index] = if diagonal { complex_shift_imag } else { 0.0 };
        }
        Ok(())
    }

    fn factor(&self, workspace: &mut Self::Workspace) -> Result<(), RadauError> {
        let dimension = workspace.real_rhs.len();
        factor_dense_in_place(&mut workspace.real, &mut workspace.real_pivots, dimension)?;
        factor_complex_in_place(
            &mut workspace.complex_real,
            &mut workspace.complex_imag,
            &mut workspace.complex_pivots,
            dimension,
        )
    }

    fn solve_real(
        &self,
        workspace: &mut Self::Workspace,
        rhs: &mut [f64],
    ) -> Result<(), RadauError> {
        solve_factored_dense_in_place(
            &workspace.real,
            &workspace.real_pivots,
            rhs,
            workspace.real_rhs.len(),
        )
    }

    fn solve_complex(
        &self,
        workspace: &mut Self::Workspace,
        rhs_real: &mut [f64],
        rhs_imag: &mut [f64],
    ) -> Result<(), RadauError> {
        solve_factored_complex_in_place(
            &workspace.complex_real,
            &workspace.complex_imag,
            &workspace.complex_pivots,
            rhs_real,
            rhs_imag,
            workspace.real_rhs.len(),
        )
    }

    fn invalidate(&self, workspace: &mut Self::Workspace) {
        workspace.real.fill(0.0);
        workspace.complex_real.fill(0.0);
        workspace.complex_imag.fill(0.0);
        workspace.real_pivots.fill(0);
        workspace.complex_pivots.fill(0);
    }
}

impl LinearSystemBackend for BandedBackend {
    type Workspace = BandedLinearWorkspace;

    fn layout(&self) -> RadauMatrixLayout {
        RadauMatrixLayout::Banded {
            lower: self.lower,
            upper: self.upper,
        }
    }

    fn create_workspace(&self, dimension: usize) -> Result<Self::Workspace, RadauError> {
        BandedLinearWorkspace::new(dimension, self.lower, self.upper)
    }

    fn assemble_shifted(
        &self,
        jacobian: JacobianValues<'_>,
        h: f64,
        coefficients: &RadauIia5,
        workspace: &mut Self::Workspace,
    ) -> Result<(), RadauError> {
        let (values, lower, upper) = match jacobian {
            JacobianValues::Banded {
                values,
                lower,
                upper,
            } => (values, lower, upper),
            _ => return Err(RadauUnsupportedRoute::BandedValuesRequired.into()),
        };
        if lower != self.lower || upper != self.upper {
            return Err(RadauConfigError::BandedBandwidthMismatch.into());
        }
        let slots = lower
            .checked_add(upper)
            .and_then(|value| value.checked_add(1))
            .ok_or(RadauError::WorkspaceSizeOverflow {
                dimension: workspace.dimension,
            })?;
        let expected =
            slots
                .checked_mul(workspace.dimension)
                .ok_or(RadauError::WorkspaceSizeOverflow {
                    dimension: workspace.dimension,
                })?;
        if values.len() != expected {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected,
                actual: values.len(),
            });
        }
        workspace.real.fill(0.0);
        workspace.complex_real.fill(0.0);
        workspace.complex_imag.fill(0.0);
        let real_shift = coefficients.mu_real / h;
        let complex_shift_real = coefficients.mu_complex.0 / h;
        let complex_shift_imag = coefficients.mu_complex.1 / h;
        if workspace.dimension == 0 {
            return Ok(());
        }
        for column in 0..workspace.dimension {
            // `upper` controls entries above the diagonal (smaller rows),
            // while `lower` controls entries below it (larger rows).
            let row_start = column.saturating_sub(self.upper);
            let row_end = column
                .saturating_add(self.lower)
                .min(workspace.dimension - 1)
                + 1;
            for row in row_start..row_end {
                let slot = workspace
                    .slot(row, column)
                    .ok_or(RadauConfigError::InvalidBandedSlot)?;
                let value = values[slot];
                let diagonal = row == column;
                workspace.real[slot] = if diagonal { real_shift - value } else { -value };
                workspace.complex_real[slot] = if diagonal {
                    complex_shift_real - value
                } else {
                    -value
                };
                workspace.complex_imag[slot] = if diagonal { complex_shift_imag } else { 0.0 };
            }
        }
        Ok(())
    }

    fn factor(&self, workspace: &mut Self::Workspace) -> Result<(), RadauError> {
        let mut real_lu =
            LapackStyleBandedLuFaithful::new(workspace.dimension, workspace.lower, workspace.upper)
                .map_err(|_| RadauError::LinearSolveFailure {
                    dimension: workspace.dimension,
                })?;
        real_lu.factor_from_compact(&workspace.real).map_err(|_| {
            RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            }
        })?;

        // The complex solve is represented as the real block matrix
        // [[Re(A), -Im(A)], [Im(A), Re(A)]].  It preserves compact banded
        // structure and lets the real LAPACK-style solver handle both parts.
        let block_dimension =
            workspace
                .dimension
                .checked_mul(2)
                .ok_or(RadauError::WorkspaceSizeOverflow {
                    dimension: workspace.dimension,
                })?;
        let block_lower = workspace
            .lower
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(RadauError::WorkspaceSizeOverflow {
                dimension: workspace.dimension,
            })?;
        let block_upper = workspace
            .upper
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(RadauError::WorkspaceSizeOverflow {
                dimension: workspace.dimension,
            })?;
        complex_banded_block_into(
            &mut workspace.complex_block,
            workspace.dimension,
            workspace.lower,
            workspace.upper,
            &workspace.complex_real,
            &workspace.complex_imag,
        )?;
        let mut complex_block_lu =
            LapackStyleBandedLuFaithful::new(block_dimension, block_lower, block_upper).map_err(
                |_| RadauError::LinearSolveFailure {
                    dimension: workspace.dimension,
                },
            )?;
        complex_block_lu
            .factor_from_compact(&workspace.complex_block)
            .map_err(|_| RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            })?;
        workspace.real_lu = Some(real_lu);
        workspace.complex_block_lu = Some(complex_block_lu);
        Ok(())
    }

    fn solve_real(
        &self,
        workspace: &mut Self::Workspace,
        rhs: &mut [f64],
    ) -> Result<(), RadauError> {
        if rhs.len() != workspace.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::LinearSolve,
                expected: workspace.dimension,
                actual: rhs.len(),
            });
        }
        let Some(lu) = workspace.real_lu.as_ref() else {
            return Err(RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            });
        };
        lu.solve_in_place(rhs)
            .map_err(|_| RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            })
    }

    fn solve_complex(
        &self,
        workspace: &mut Self::Workspace,
        rhs_real: &mut [f64],
        rhs_imag: &mut [f64],
    ) -> Result<(), RadauError> {
        if rhs_real.len() != workspace.dimension || rhs_imag.len() != workspace.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::LinearSolve,
                expected: workspace.dimension,
                actual: rhs_real.len().min(rhs_imag.len()),
            });
        }
        let Some(lu) = workspace.complex_block_lu.as_ref() else {
            return Err(RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            });
        };
        // Interleaving is confined to this reusable RHS buffer.  No temporary
        // complex vector is created for each Newton iteration.
        let block_rhs = &mut workspace.complex_block_rhs;
        for index in 0..workspace.dimension {
            block_rhs[2 * index] = rhs_real[index];
            block_rhs[2 * index + 1] = rhs_imag[index];
        }
        lu.solve_in_place(block_rhs)
            .map_err(|_| RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            })?;
        for index in 0..workspace.dimension {
            rhs_real[index] = block_rhs[2 * index];
            rhs_imag[index] = block_rhs[2 * index + 1];
        }
        Ok(())
    }

    fn invalidate(&self, workspace: &mut Self::Workspace) {
        workspace.jacobian.fill(0.0);
        workspace.real.fill(0.0);
        workspace.complex_real.fill(0.0);
        workspace.complex_imag.fill(0.0);
        workspace.complex_block.fill(0.0);
        workspace.real_lu = None;
        workspace.complex_block_lu = None;
    }
}

impl LinearSystemBackend for SparseBackend {
    type Workspace = SparseLinearWorkspace;

    fn layout(&self) -> RadauMatrixLayout {
        RadauMatrixLayout::Sparse
    }

    fn create_workspace(&self, dimension: usize) -> Result<Self::Workspace, RadauError> {
        SparseLinearWorkspace::from_pattern(dimension, &self.pattern)
    }

    fn assemble_shifted(
        &self,
        jacobian: JacobianValues<'_>,
        h: f64,
        coefficients: &RadauIia5,
        workspace: &mut Self::Workspace,
    ) -> Result<(), RadauError> {
        let (values, entries) = match jacobian {
            JacobianValues::Sparse { values, entries } => (values, entries),
            _ => return Err(RadauUnsupportedRoute::SparseValuesRequired.into()),
        };
        if values.len() != entries.len() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected: entries.len(),
                actual: values.len(),
            });
        }
        workspace.real.fill(0.0);
        workspace.complex_real.fill(0.0);
        workspace.complex_imag.fill(0.0);
        for (&(row, column), &value) in entries.iter().zip(values) {
            let Some(index) = workspace.value_index(row, column) else {
                return Err(RadauConfigError::SparseEntryMissing.into());
            };
            workspace.real[index] = -value;
            workspace.complex_real[index] = -value;
        }
        let real_shift = coefficients.mu_real / h;
        let complex_shift_real = coefficients.mu_complex.0 / h;
        let complex_shift_imag = coefficients.mu_complex.1 / h;
        for index in 0..workspace.dimension {
            let Some(diagonal) = workspace.value_index(index, index) else {
                return Err(RadauConfigError::SparseDiagonalMissing.into());
            };
            workspace.real[diagonal] += real_shift;
            workspace.complex_real[diagonal] += complex_shift_real;
            workspace.complex_imag[diagonal] = complex_shift_imag;
        }
        Ok(())
    }

    fn factor(&self, workspace: &mut Self::Workspace) -> Result<(), RadauError> {
        sparse_triplets_into(
            &mut workspace.real_triplets,
            workspace.dimension,
            &workspace.column_offsets,
            &workspace.row_indices,
            &workspace.real,
        );
        let real_lu =
            FaerSparseLuSolver::from_triplets(workspace.dimension, &workspace.real_triplets)
                .map_err(|_| RadauError::LinearSolveFailure {
                    dimension: workspace.dimension,
                })?;
        // Faer consumes triplets at factorization time.  The structural CSC
        // pattern remains cached in the workspace; this temporary only packs
        // the current numeric shifted values for the backend API.
        sparse_complex_block_triplets_into(
            &mut workspace.complex_triplets,
            workspace.dimension,
            &workspace.column_offsets,
            &workspace.row_indices,
            &workspace.complex_real,
            &workspace.complex_imag,
        );
        let complex_lu = FaerSparseLuSolver::from_triplets(
            workspace
                .dimension
                .checked_mul(2)
                .ok_or(RadauError::WorkspaceSizeOverflow {
                    dimension: workspace.dimension,
                })?,
            &workspace.complex_triplets,
        )
        .map_err(|_| RadauError::LinearSolveFailure {
            dimension: workspace.dimension,
        })?;
        workspace.real_lu = Some(real_lu);
        workspace.complex_block_lu = Some(complex_lu);
        Ok(())
    }

    fn solve_real(
        &self,
        workspace: &mut Self::Workspace,
        rhs: &mut [f64],
    ) -> Result<(), RadauError> {
        if rhs.len() != workspace.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::LinearSolve,
                expected: workspace.dimension,
                actual: rhs.len(),
            });
        }
        let Some(lu) = workspace.real_lu.as_ref() else {
            return Err(RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            });
        };
        lu.solve_in_place(rhs)
            .map_err(|_| RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            })
    }

    fn solve_complex(
        &self,
        workspace: &mut Self::Workspace,
        rhs_real: &mut [f64],
        rhs_imag: &mut [f64],
    ) -> Result<(), RadauError> {
        if rhs_real.len() != workspace.dimension || rhs_imag.len() != workspace.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::LinearSolve,
                expected: workspace.dimension,
                actual: rhs_real.len().min(rhs_imag.len()),
            });
        }
        let Some(lu) = workspace.complex_block_lu.as_ref() else {
            return Err(RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            });
        };
        let block_rhs = &mut workspace.complex_block_rhs;
        for index in 0..workspace.dimension {
            block_rhs[2 * index] = rhs_real[index];
            block_rhs[2 * index + 1] = rhs_imag[index];
        }
        lu.solve_in_place(block_rhs)
            .map_err(|_| RadauError::LinearSolveFailure {
                dimension: workspace.dimension,
            })?;
        for index in 0..workspace.dimension {
            rhs_real[index] = block_rhs[2 * index];
            rhs_imag[index] = block_rhs[2 * index + 1];
        }
        Ok(())
    }

    fn invalidate(&self, workspace: &mut Self::Workspace) {
        workspace.jacobian.fill(0.0);
        workspace.real.fill(0.0);
        workspace.complex_real.fill(0.0);
        workspace.complex_imag.fill(0.0);
        workspace.real_lu = None;
        workspace.complex_block_lu = None;
    }
}

/// Build the interleaved real representation of a complex band matrix.
///
/// A complex entry `a + i b` becomes the real 2x2 block
/// `[[a, -b], [b, a]]`.  The representation remains banded and is built
/// directly from compact storage; it is never expanded to an `n x n` dense
/// matrix.
fn complex_banded_block_into(
    output: &mut [f64],
    source_dimension: usize,
    source_lower: usize,
    source_upper: usize,
    complex_real: &[f64],
    complex_imag: &[f64],
) -> Result<(), RadauError> {
    let dimension = source_dimension
        .checked_mul(2)
        .ok_or(RadauError::WorkspaceSizeOverflow {
            dimension: source_dimension,
        })?;
    let lower = source_lower
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(RadauError::WorkspaceSizeOverflow {
            dimension: source_dimension,
        })?;
    let upper = source_upper
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(RadauError::WorkspaceSizeOverflow {
            dimension: source_dimension,
        })?;
    let source_slots = source_lower
        .checked_add(source_upper)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_mul(source_dimension))
        .ok_or(RadauError::WorkspaceSizeOverflow {
            dimension: source_dimension,
        })?;
    let output_slots = lower
        .checked_add(upper)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_mul(dimension))
        .ok_or(RadauError::WorkspaceSizeOverflow {
            dimension: source_dimension,
        })?;
    if complex_real.len() != source_slots || complex_imag.len() != source_slots {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::LinearSolve,
            expected: source_slots,
            actual: complex_real.len().min(complex_imag.len()),
        });
    }
    if output.len() != output_slots {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::LinearSolve,
            expected: output_slots,
            actual: output.len(),
        });
    }
    output.fill(0.0);
    for column in 0..source_dimension {
        let row_start = column.saturating_sub(source_upper);
        let row_end = column
            .saturating_add(source_lower)
            .min(source_dimension - 1)
            + 1;
        for row in row_start..row_end {
            let source_slot = (source_upper + row - column) * source_dimension + column;
            let real = complex_real[source_slot];
            let imag = complex_imag[source_slot];
            let set = |output: &mut [f64], row: usize, column: usize, value: f64| {
                output[(upper + row - column) * dimension + column] = value;
            };
            set(output, 2 * row, 2 * column, real);
            set(output, 2 * row, 2 * column + 1, -imag);
            set(output, 2 * row + 1, 2 * column, imag);
            set(output, 2 * row + 1, 2 * column + 1, real);
        }
    }
    Ok(())
}

fn sparse_triplets_into(
    triplets: &mut Vec<Triplet<usize, usize, f64>>,
    dimension: usize,
    column_offsets: &[usize],
    row_indices: &[usize],
    values: &[f64],
) {
    triplets.clear();
    for column in 0..dimension {
        let start = column_offsets[column];
        let end = column_offsets[column + 1];
        for index in start..end {
            triplets.push(Triplet::new(row_indices[index], column, values[index]));
        }
    }
}

fn sparse_complex_block_triplets_into(
    triplets: &mut Vec<Triplet<usize, usize, f64>>,
    dimension: usize,
    column_offsets: &[usize],
    row_indices: &[usize],
    real: &[f64],
    imag: &[f64],
) {
    triplets.clear();
    for column in 0..dimension {
        let start = column_offsets[column];
        let end = column_offsets[column + 1];
        for index in start..end {
            let row = row_indices[index];
            let real_value = real[index];
            let imag_value = imag[index];
            let block_row = 2 * row;
            let block_column = 2 * column;
            triplets.push(Triplet::new(block_row, block_column, real_value));
            triplets.push(Triplet::new(block_row, block_column + 1, -imag_value));
            triplets.push(Triplet::new(block_row + 1, block_column, imag_value));
            triplets.push(Triplet::new(block_row + 1, block_column + 1, real_value));
        }
    }
}

impl Default for RadauLinearWorkspace {
    fn default() -> Self {
        Self::Dense(DenseLinearWorkspace::default())
    }
}

fn checked_matrix_len(dimension: usize) -> Result<usize, RadauError> {
    dimension
        .checked_mul(dimension)
        .ok_or(RadauError::WorkspaceSizeOverflow { dimension })
}

pub(crate) fn factor_dense_in_place(
    matrix: &mut [f64],
    pivots: &mut [usize],
    dimension: usize,
) -> Result<(), RadauError> {
    let matrix_len = dimension
        .checked_mul(dimension)
        .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
    if matrix.len() != matrix_len || pivots.len() != dimension {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::LinearSolve,
            expected: matrix_len,
            actual: matrix.len(),
        });
    }
    for column in 0..dimension {
        let mut pivot = column;
        let mut pivot_abs = matrix[column * dimension + column].abs();
        for row in column + 1..dimension {
            let candidate = matrix[row * dimension + column].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= f64::EPSILON {
            return Err(RadauError::LinearSolveFailure { dimension });
        }
        pivots[column] = pivot;
        if pivot != column {
            for index in 0..dimension {
                matrix.swap(column * dimension + index, pivot * dimension + index);
            }
        }
        for row in column + 1..dimension {
            let factor = matrix[row * dimension + column] / matrix[column * dimension + column];
            matrix[row * dimension + column] = factor;
            for index in column + 1..dimension {
                matrix[row * dimension + index] -= factor * matrix[column * dimension + index];
            }
        }
    }
    Ok(())
}

pub(crate) fn solve_factored_dense_in_place(
    matrix: &[f64],
    pivots: &[usize],
    rhs: &mut [f64],
    dimension: usize,
) -> Result<(), RadauError> {
    let matrix_len = dimension
        .checked_mul(dimension)
        .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
    if matrix.len() != matrix_len || pivots.len() != dimension || rhs.len() != dimension {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::LinearSolve,
            expected: dimension,
            actual: rhs.len(),
        });
    }
    for (column, &pivot) in pivots.iter().enumerate() {
        if pivot != column {
            rhs.swap(column, pivot);
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            rhs[row] -= matrix[row * dimension + column] * rhs[column];
        }
    }
    for row in (0..dimension).rev() {
        for column in row + 1..dimension {
            rhs[row] -= matrix[row * dimension + column] * rhs[column];
        }
        let diagonal = matrix[row * dimension + row];
        if !diagonal.is_finite() || diagonal.abs() <= f64::EPSILON {
            return Err(RadauError::LinearSolveFailure { dimension });
        }
        rhs[row] /= diagonal;
    }
    Ok(())
}

pub(crate) fn factor_complex_in_place(
    real: &mut [f64],
    imag: &mut [f64],
    pivots: &mut [usize],
    dimension: usize,
) -> Result<(), RadauError> {
    let matrix_len = dimension
        .checked_mul(dimension)
        .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
    if real.len() != matrix_len || imag.len() != matrix_len || pivots.len() != dimension {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::LinearSolve,
            expected: matrix_len,
            actual: real.len(),
        });
    }
    for column in 0..dimension {
        let mut pivot = column;
        let mut pivot_abs = complex_abs_squared(
            real[column * dimension + column],
            imag[column * dimension + column],
        );
        for row in column + 1..dimension {
            let candidate = complex_abs_squared(
                real[row * dimension + column],
                imag[row * dimension + column],
            );
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= f64::EPSILON * f64::EPSILON {
            return Err(RadauError::LinearSolveFailure { dimension });
        }
        pivots[column] = pivot;
        if pivot != column {
            for index in 0..dimension {
                real.swap(column * dimension + index, pivot * dimension + index);
                imag.swap(column * dimension + index, pivot * dimension + index);
            }
        }
        let pivot_real = real[column * dimension + column];
        let pivot_imag = imag[column * dimension + column];
        for row in column + 1..dimension {
            let index = row * dimension + column;
            let (factor_real, factor_imag) =
                complex_divide(real[index], imag[index], pivot_real, pivot_imag);
            real[index] = factor_real;
            imag[index] = factor_imag;
            for other in column + 1..dimension {
                let target = row * dimension + other;
                let pivot_row = column * dimension + other;
                let product_real = factor_real * real[pivot_row] - factor_imag * imag[pivot_row];
                let product_imag = factor_real * imag[pivot_row] + factor_imag * real[pivot_row];
                real[target] -= product_real;
                imag[target] -= product_imag;
            }
        }
    }
    Ok(())
}

pub(crate) fn solve_factored_complex_in_place(
    real: &[f64],
    imag: &[f64],
    pivots: &[usize],
    rhs_real: &mut [f64],
    rhs_imag: &mut [f64],
    dimension: usize,
) -> Result<(), RadauError> {
    let matrix_len = dimension
        .checked_mul(dimension)
        .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
    if real.len() != matrix_len
        || imag.len() != matrix_len
        || pivots.len() != dimension
        || rhs_real.len() != dimension
        || rhs_imag.len() != dimension
    {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::LinearSolve,
            expected: dimension,
            actual: rhs_real.len(),
        });
    }
    for (column, &pivot) in pivots.iter().enumerate() {
        if pivot != column {
            rhs_real.swap(column, pivot);
            rhs_imag.swap(column, pivot);
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            let index = row * dimension + column;
            let product_real = real[index] * rhs_real[column] - imag[index] * rhs_imag[column];
            let product_imag = real[index] * rhs_imag[column] + imag[index] * rhs_real[column];
            rhs_real[row] -= product_real;
            rhs_imag[row] -= product_imag;
        }
    }
    for row in (0..dimension).rev() {
        for column in row + 1..dimension {
            let index = row * dimension + column;
            let product_real = real[index] * rhs_real[column] - imag[index] * rhs_imag[column];
            let product_imag = real[index] * rhs_imag[column] + imag[index] * rhs_real[column];
            rhs_real[row] -= product_real;
            rhs_imag[row] -= product_imag;
        }
        let index = row * dimension + row;
        let (value_real, value_imag) =
            complex_divide(rhs_real[row], rhs_imag[row], real[index], imag[index]);
        if !value_real.is_finite() || !value_imag.is_finite() {
            return Err(RadauError::LinearSolveFailure { dimension });
        }
        rhs_real[row] = value_real;
        rhs_imag[row] = value_imag;
    }
    Ok(())
}

fn complex_abs_squared(real: f64, imag: f64) -> f64 {
    real.mul_add(real, imag * imag)
}

fn complex_divide(lhs_real: f64, lhs_imag: f64, rhs_real: f64, rhs_imag: f64) -> (f64, f64) {
    let denominator = complex_abs_squared(rhs_real, rhs_imag);
    (
        (lhs_real * rhs_real + lhs_imag * rhs_imag) / denominator,
        (lhs_imag * rhs_real - lhs_real * rhs_imag) / denominator,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Real or complex shift used to identify a numeric factorization.
pub(crate) enum RadauShift {
    Real { mu_over_h: f64 },
    Complex { real: f64, imag: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Immutable identity of a reusable shifted Jacobian factorization.
pub(crate) struct RadauFactorKey {
    /// Generation of the Jacobian values used by the factorization.
    pub jacobian_generation: u64,
    /// Generation of the step-dependent shift.
    pub step_generation: u64,
    /// Real or complex shift identity.
    pub shift: RadauShift,
    /// Storage layout of the factorized matrix.
    pub matrix_layout: RadauMatrixLayout,
}

impl RadauFactorKey {
    pub(crate) const fn new(
        jacobian_generation: u64,
        step_generation: u64,
        shift: RadauShift,
        matrix_layout: RadauMatrixLayout,
    ) -> Self {
        Self {
            jacobian_generation,
            step_generation,
            shift,
            matrix_layout,
        }
    }

    /// Check whether a cached factorization remains valid for `current`.
    pub(crate) fn remains_valid_for(&self, current: Self) -> bool {
        *self == current
    }
}

/// Row-major real shifted matrix owned by the linear backend.
///
/// The matrix is filled directly from the caller-owned Jacobian and can then
/// be consumed by a factorization object. This avoids a temporary matrix plus
/// a second clone at the ownership handoff.
#[derive(Debug, Default)]
pub(crate) struct RadauDenseShiftedMatrix {
    dimension: usize,
    values: Vec<f64>,
}

impl RadauDenseShiftedMatrix {
    /// Allocate an `n x n` row-major shifted matrix.
    pub(crate) fn new(dimension: usize) -> Result<Self, RadauError> {
        let matrix_len = dimension
            .checked_mul(dimension)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        Ok(Self {
            dimension,
            values: vec![0.0; matrix_len],
        })
    }

    /// Fill `mu/h * I - J` in place from a row-major Jacobian.
    pub(crate) fn load_real_shifted(
        &mut self,
        jacobian: &[f64],
        mu_over_h: f64,
    ) -> Result<(), RadauError> {
        if jacobian.len() != self.values.len() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::LinearSolve,
                expected: self.values.len(),
                actual: jacobian.len(),
            });
        }
        for (index, value) in jacobian.iter().copied().enumerate() {
            let diagonal = index / self.dimension == index % self.dimension;
            self.values[index] = if diagonal { mu_over_h - value } else { -value };
        }
        Ok(())
    }

    /// Return the matrix dimension.
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow the row-major matrix values.
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutably borrow the row-major matrix values.
    pub(crate) fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Transfer matrix ownership to a factorization without cloning.
    pub(crate) fn into_values(self) -> Vec<f64> {
        self.values
    }
}

/// Real/imaginary storage for a complex shifted matrix.
#[derive(Debug, Default)]
pub(crate) struct RadauDenseComplexShiftedMatrix {
    dimension: usize,
    real: Vec<f64>,
    imag: Vec<f64>,
}

impl RadauDenseComplexShiftedMatrix {
    /// Allocate real and imaginary `n x n` shifted storage.
    pub(crate) fn new(dimension: usize) -> Result<Self, RadauError> {
        let matrix_len = dimension
            .checked_mul(dimension)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        Ok(Self {
            dimension,
            real: vec![0.0; matrix_len],
            imag: vec![0.0; matrix_len],
        })
    }

    /// Fill `(real_shift + i * imag_shift) I - J` in place.
    pub(crate) fn load_shifted(
        &mut self,
        jacobian: &[f64],
        real_shift: f64,
        imag_shift: f64,
    ) -> Result<(), RadauError> {
        if jacobian.len() != self.real.len() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::LinearSolve,
                expected: self.real.len(),
                actual: jacobian.len(),
            });
        }
        for (index, value) in jacobian.iter().copied().enumerate() {
            let diagonal = index / self.dimension == index % self.dimension;
            self.real[index] = if diagonal { real_shift - value } else { -value };
            self.imag[index] = if diagonal { imag_shift } else { 0.0 };
        }
        Ok(())
    }

    /// Borrow the real part of the shifted matrix.
    pub(crate) fn real(&self) -> &[f64] {
        &self.real
    }

    /// Borrow the imaginary part of the shifted matrix.
    pub(crate) fn imag(&self) -> &[f64] {
        &self.imag
    }
}
