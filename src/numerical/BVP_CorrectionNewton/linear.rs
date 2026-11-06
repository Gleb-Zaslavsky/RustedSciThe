//! CorrectionNewton-local linear-workspace boundary.
//!
//! The Newton algorithm owns refresh/reuse timing. Backend implementations
//! receive coordinate values directly; Sparse and Banded never build a dense
//! global matrix. The production backend kernels are reused from `BVP_sci`,
//! without importing its collocation/Newton lifecycle.

use crate::numerical::BVP_CorrectionNewton::CorrectionNewtonJacobianPattern;
use crate::numerical::BVP_sci::new::linear::NativeLinearBackend;
use crate::numerical::BVP_sci::new::{BvpSciMatrixLayout, BvpSciNewError};
use nalgebra::{DMatrix, DVectorViewMut, Dyn, linalg::LU};
use thiserror::Error;

/// Linear storage policy for the standalone CorrectionNewton core.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CorrectionNewtonMatrixLayout {
    /// Dense nalgebra LU; appropriate for compact/fully-coupled systems.
    #[default]
    Dense,
    /// Faer CSC sparse LU over the structural Jacobian coordinates.
    Sparse,
    /// Infer exact scalar lower/upper bandwidth from the symbolic pattern.
    BandedAuto,
}

/// Backend identity and resolved storage metadata for one factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionNewtonLinearBackend {
    Dense,
    Sparse,
    Banded { lower: usize, upper: usize },
}

impl CorrectionNewtonLinearBackend {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded { .. } => "Banded",
        }
    }
}

impl CorrectionNewtonMatrixLayout {
    const fn backend_name(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::BandedAuto => "Banded",
        }
    }
}

/// Typed errors crossing the Newton/linear-algebra boundary.
#[derive(Debug, Error)]
pub enum CorrectionNewtonLinearError {
    #[error("{backend} matrix dimension must be positive")]
    ZeroDimension { backend: &'static str },
    #[error("{backend} matrix is singular")]
    Singular { backend: &'static str },
    #[error(
        "{backend} solve failed its backward-residual check: residual={residual_norm:e}, allowed={tolerance:e}"
    )]
    ResidualCheckFailed {
        backend: &'static str,
        residual_norm: f64,
        tolerance: f64,
    },
    #[error("{backend} solve produced a non-finite solution")]
    NonFiniteSolution { backend: &'static str },
    #[error(
        "{backend} Jacobian pattern dimension {pattern_dimension} does not match matrix dimension {matrix_dimension}"
    )]
    PatternDimensionMismatch {
        backend: &'static str,
        pattern_dimension: usize,
        matrix_dimension: usize,
    },
    #[error("{backend} requires a coordinate Jacobian pattern")]
    MissingCoordinatePattern { backend: &'static str },
    #[error("{backend} cannot accept a dense Jacobian callback")]
    DenseJacobianForStructuralBackend { backend: &'static str },
    #[error(
        "{backend} Jacobian has shape {actual_rows}x{actual_columns}, expected {expected}x{expected}"
    )]
    MatrixDimensionMismatch {
        backend: &'static str,
        expected: usize,
        actual_rows: usize,
        actual_columns: usize,
    },
    #[error("{backend} coordinate Jacobian values were supplied without a pattern")]
    CoordinateValuesWithoutPattern { backend: &'static str },
    #[error("{backend} coordinate Jacobian has {actual} values, expected {expected}")]
    CoordinateValueCountMismatch {
        backend: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{backend} Jacobian value at ({row}, {column}) is not finite")]
    NonFiniteJacobianValue {
        backend: &'static str,
        row: usize,
        column: usize,
    },
    #[error("{backend} factorization was requested before matrix assembly")]
    FactorizationNotReady { backend: &'static str },
    #[error("{backend} right-hand side has length {actual}, expected {expected}")]
    RightHandSideLengthMismatch {
        backend: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{backend} native linear backend failed: {source}")]
    NativeBackend {
        backend: &'static str,
        #[source]
        source: BvpSciNewError,
    },
}

impl CorrectionNewtonLinearError {
    pub(crate) const fn backend_name(&self) -> &'static str {
        match self {
            Self::ZeroDimension { backend }
            | Self::Singular { backend }
            | Self::ResidualCheckFailed { backend, .. }
            | Self::NonFiniteSolution { backend }
            | Self::PatternDimensionMismatch { backend, .. }
            | Self::MissingCoordinatePattern { backend }
            | Self::DenseJacobianForStructuralBackend { backend }
            | Self::MatrixDimensionMismatch { backend, .. }
            | Self::CoordinateValuesWithoutPattern { backend }
            | Self::CoordinateValueCountMismatch { backend, .. }
            | Self::NonFiniteJacobianValue { backend, .. }
            | Self::FactorizationNotReady { backend }
            | Self::RightHandSideLengthMismatch { backend, .. }
            | Self::NativeBackend { backend, .. } => backend,
        }
    }
}

enum Storage {
    Dense {
        factorization: Option<LU<f64, Dyn, Dyn>>,
    },
    Native(NativeLinearBackend),
}

/// Reusable factorization and numeric assembly storage for one fixed pattern.
pub(crate) struct CorrectionNewtonLinearWorkspace {
    backend: CorrectionNewtonLinearBackend,
    dimension: usize,
    coordinates: Option<Vec<(usize, usize)>>,
    entries: Vec<(usize, usize, f64)>,
    residual_check_scratch: Vec<f64>,
    row_norm_scratch: Vec<f64>,
    matrix_inf_norm: f64,
    dense_matrix: Option<DMatrix<f64>>,
    storage: Storage,
}

impl CorrectionNewtonLinearWorkspace {
    pub(crate) fn new(
        layout: CorrectionNewtonMatrixLayout,
        dimension: usize,
        pattern: Option<&CorrectionNewtonJacobianPattern>,
    ) -> Result<Self, CorrectionNewtonLinearError> {
        if dimension == 0 {
            return Err(CorrectionNewtonLinearError::ZeroDimension {
                backend: layout.backend_name(),
            });
        }
        if let Some(pattern) = pattern {
            if pattern.dimension() != dimension {
                return Err(CorrectionNewtonLinearError::PatternDimensionMismatch {
                    backend: layout.backend_name(),
                    pattern_dimension: pattern.dimension(),
                    matrix_dimension: dimension,
                });
            }
        }
        let requires_pattern = layout != CorrectionNewtonMatrixLayout::Dense;
        if requires_pattern && pattern.is_none() {
            return Err(CorrectionNewtonLinearError::MissingCoordinatePattern {
                backend: layout.backend_name(),
            });
        }

        let coordinates = pattern.map(|pattern| pattern.coordinates().to_vec());
        let mut entries = Vec::with_capacity(coordinates.as_ref().map_or(0, Vec::len));
        if let Some(coordinates) = &coordinates {
            entries.extend(coordinates.iter().map(|&(row, column)| (row, column, 0.0)));
        }

        let (backend, storage) = match layout {
            CorrectionNewtonMatrixLayout::Dense => (
                CorrectionNewtonLinearBackend::Dense,
                Storage::Dense {
                    factorization: None,
                },
            ),
            CorrectionNewtonMatrixLayout::Sparse => {
                let native =
                    NativeLinearBackend::new(BvpSciMatrixLayout::Sparse, dimension, entries.len())
                        .map_err(|error| {
                            map_native_error(CorrectionNewtonLinearBackend::Sparse, error)
                        })?;
                (
                    CorrectionNewtonLinearBackend::Sparse,
                    Storage::Native(native),
                )
            }
            CorrectionNewtonMatrixLayout::BandedAuto => {
                let (lower, upper) = pattern
                    .expect("BandedAuto pattern checked above")
                    .bandwidth();
                let backend = CorrectionNewtonLinearBackend::Banded { lower, upper };
                let native = NativeLinearBackend::new(
                    BvpSciMatrixLayout::Banded { lower, upper },
                    dimension,
                    entries.len(),
                )
                .map_err(|error| map_native_error(backend, error))?;
                (backend, Storage::Native(native))
            }
        };

        Ok(Self {
            backend,
            dimension,
            coordinates,
            entries,
            residual_check_scratch: vec![0.0; dimension],
            row_norm_scratch: vec![0.0; dimension],
            matrix_inf_norm: 0.0,
            dense_matrix: None,
            storage,
        })
    }

    pub(crate) const fn backend(&self) -> CorrectionNewtonLinearBackend {
        self.backend
    }

    pub(crate) fn factor_dense(
        &mut self,
        matrix: DMatrix<f64>,
    ) -> Result<(), CorrectionNewtonLinearError> {
        if self.backend != CorrectionNewtonLinearBackend::Dense {
            return Err(
                CorrectionNewtonLinearError::DenseJacobianForStructuralBackend {
                    backend: self.backend.name(),
                },
            );
        }
        let (rows, columns) = matrix.shape();
        if rows != self.dimension || columns != self.dimension {
            return Err(CorrectionNewtonLinearError::MatrixDimensionMismatch {
                backend: self.backend.name(),
                expected: self.dimension,
                actual_rows: rows,
                actual_columns: columns,
            });
        }
        let Storage::Dense { factorization } = &mut self.storage else {
            unreachable!("backend and storage variants are constructed together")
        };
        *factorization = Some(matrix.lu());
        Ok(())
    }

    pub(crate) fn assemble_coordinates(
        &mut self,
        values: &[f64],
    ) -> Result<(), CorrectionNewtonLinearError> {
        let coordinates = self.coordinates.as_ref().ok_or_else(|| {
            CorrectionNewtonLinearError::CoordinateValuesWithoutPattern {
                backend: self.backend.name(),
            }
        })?;
        if values.len() != coordinates.len() {
            return Err(CorrectionNewtonLinearError::CoordinateValueCountMismatch {
                backend: self.backend.name(),
                expected: coordinates.len(),
                actual: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            let (row, column) = coordinates[index];
            return Err(CorrectionNewtonLinearError::NonFiniteJacobianValue {
                backend: self.backend.name(),
                row,
                column,
            });
        }

        match &mut self.storage {
            Storage::Dense { .. } => {
                let mut matrix = DMatrix::zeros(self.dimension, self.dimension);
                for (&(row, column), &value) in coordinates.iter().zip(values) {
                    matrix[(row, column)] = value;
                }
                self.dense_matrix = Some(matrix);
            }
            Storage::Native(native) => {
                for (entry, &value) in self.entries.iter_mut().zip(values) {
                    entry.2 = value;
                }
                self.row_norm_scratch.fill(0.0);
                for &(row, _, value) in &self.entries {
                    self.row_norm_scratch[row] += value.abs();
                }
                self.matrix_inf_norm = self
                    .row_norm_scratch
                    .iter()
                    .fold(0.0_f64, |maximum, value| maximum.max(*value));
                native
                    .assemble(&self.entries)
                    .map_err(|error| map_native_error(self.backend, error))?;
            }
        }
        Ok(())
    }

    pub(crate) fn factor_assembled(&mut self) -> Result<(), CorrectionNewtonLinearError> {
        match &mut self.storage {
            Storage::Dense { factorization } => {
                let matrix = self.dense_matrix.take().ok_or_else(|| {
                    CorrectionNewtonLinearError::FactorizationNotReady {
                        backend: self.backend.name(),
                    }
                })?;
                *factorization = Some(matrix.lu());
                Ok(())
            }
            Storage::Native(native) => native
                .factor()
                .map_err(|error| map_native_error(self.backend, error)),
        }
    }

    pub(crate) fn factor_coordinates(
        &mut self,
        values: &[f64],
    ) -> Result<(), CorrectionNewtonLinearError> {
        self.assemble_coordinates(values)?;
        self.factor_assembled()
    }

    pub(crate) fn solve_in_place(
        &mut self,
        rhs: &mut [f64],
    ) -> Result<(), CorrectionNewtonLinearError> {
        if rhs.len() != self.dimension {
            return Err(CorrectionNewtonLinearError::RightHandSideLengthMismatch {
                backend: self.backend.name(),
                expected: self.dimension,
                actual: rhs.len(),
            });
        }
        match &mut self.storage {
            Storage::Dense { factorization } => {
                let factorization = factorization.as_ref().ok_or_else(|| {
                    CorrectionNewtonLinearError::FactorizationNotReady {
                        backend: self.backend.name(),
                    }
                })?;
                let mut rhs = DVectorViewMut::from_slice(rhs, self.dimension);
                if factorization.solve_mut(&mut rhs) {
                    Ok(())
                } else {
                    Err(CorrectionNewtonLinearError::Singular {
                        backend: self.backend.name(),
                    })
                }
            }
            Storage::Native(native) => {
                let rhs_norm = rhs
                    .iter()
                    .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
                for (check, &value) in self.residual_check_scratch.iter_mut().zip(rhs.iter()) {
                    *check = -value;
                }

                native
                    .solve_in_place(rhs)
                    .map_err(|error| map_native_error(self.backend, error))?;
                if rhs.iter().any(|value| !value.is_finite()) {
                    return Err(CorrectionNewtonLinearError::NonFiniteSolution {
                        backend: self.backend.name(),
                    });
                }

                for &(row, column, value) in &self.entries {
                    self.residual_check_scratch[row] += value * rhs[column];
                }
                let residual_norm = self
                    .residual_check_scratch
                    .iter()
                    .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
                let solution_norm = rhs
                    .iter()
                    .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
                let scale = self.matrix_inf_norm * solution_norm + rhs_norm;
                let tolerance =
                    128.0 * f64::EPSILON * self.dimension as f64 * scale.max(f64::MIN_POSITIVE);
                if !residual_norm.is_finite() || residual_norm > tolerance {
                    return Err(CorrectionNewtonLinearError::ResidualCheckFailed {
                        backend: self.backend.name(),
                        residual_norm,
                        tolerance,
                    });
                }
                Ok(())
            }
        }
    }
}

fn map_native_error(
    backend: CorrectionNewtonLinearBackend,
    error: BvpSciNewError,
) -> CorrectionNewtonLinearError {
    match error {
        BvpSciNewError::SingularJacobian => CorrectionNewtonLinearError::Singular {
            backend: backend.name(),
        },
        other => CorrectionNewtonLinearError::NativeBackend {
            backend: backend.name(),
            source: other,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_backend_failures_preserve_the_typed_source() {
        let source = BvpSciNewError::LinearFactorization {
            message: "fixture failure".into(),
        };
        let error = map_native_error(CorrectionNewtonLinearBackend::Sparse, source.clone());
        match error {
            CorrectionNewtonLinearError::NativeBackend {
                backend,
                source: actual,
            } => {
                assert_eq!(backend, "Sparse");
                assert_eq!(actual, source);
                assert!(
                    std::error::Error::source(&CorrectionNewtonLinearError::NativeBackend {
                        backend,
                        source: actual,
                    })
                    .is_some()
                );
            }
            other => panic!("expected typed native backend failure, got {other:?}"),
        }
    }

    fn solve_tridiagonal(layout: CorrectionNewtonMatrixLayout) -> [f64; 3] {
        let pattern = CorrectionNewtonJacobianPattern::try_new(
            3,
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)],
        )
        .expect("tridiagonal pattern should be valid");
        let mut workspace = CorrectionNewtonLinearWorkspace::new(layout, 3, Some(&pattern))
            .expect("linear workspace should prepare");
        workspace
            .factor_coordinates(&[4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0])
            .expect("tridiagonal matrix should factor");
        let mut rhs = [6.0, 12.0, 14.0];
        workspace
            .solve_in_place(&mut rhs)
            .expect("tridiagonal RHS should solve");
        rhs
    }

    #[test]
    fn dense_workspace_reuses_one_factorization_for_multiple_rhs() {
        let pattern =
            CorrectionNewtonJacobianPattern::dense(2).expect("dense pattern should prepare");
        let mut workspace = CorrectionNewtonLinearWorkspace::new(
            CorrectionNewtonMatrixLayout::Dense,
            2,
            Some(&pattern),
        )
        .expect("dense workspace should prepare");
        workspace
            .factor_coordinates(&[3.0, 1.0, 1.0, 2.0])
            .expect("valid square matrix should factor");

        let mut first = [9.0, 8.0];
        workspace
            .solve_in_place(&mut first)
            .expect("first RHS should solve");
        assert!((first[0] - 2.0).abs() < 1e-12);
        assert!((first[1] - 3.0).abs() < 1e-12);

        let mut second = [4.0, 3.0];
        workspace
            .solve_in_place(&mut second)
            .expect("same factorization should solve a second RHS");
        assert!((second[0] - 1.0).abs() < 1e-12);
        assert!((second[1] - 1.0).abs() < 1e-12);
        assert_eq!(workspace.backend(), CorrectionNewtonLinearBackend::Dense);
    }

    #[test]
    fn sparse_and_auto_banded_solve_the_same_coordinate_system() {
        let expected = [1.0, 2.0, 3.0];
        for layout in [
            CorrectionNewtonMatrixLayout::Dense,
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let actual = solve_tridiagonal(layout);
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-10, "layout={layout:?}");
            }
        }
    }

    #[test]
    fn every_backend_reuses_one_factorization_for_multiple_rhs() {
        let expected = [[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]];
        for layout in [
            CorrectionNewtonMatrixLayout::Dense,
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let pattern = CorrectionNewtonJacobianPattern::try_new(
                3,
                vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)],
            )
            .unwrap();
            let mut workspace =
                CorrectionNewtonLinearWorkspace::new(layout, 3, Some(&pattern)).unwrap();
            workspace
                .factor_coordinates(&[4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0])
                .unwrap();
            for solution in expected {
                let rhs = [
                    4.0 * solution[0] + solution[1],
                    solution[0] + 4.0 * solution[1] + solution[2],
                    solution[1] + 4.0 * solution[2],
                ];
                let mut actual = rhs;
                workspace.solve_in_place(&mut actual).unwrap();
                for (actual, expected) in actual.into_iter().zip(solution) {
                    assert!((actual - expected).abs() < 1e-10, "layout={layout:?}");
                }
            }
        }
    }

    #[test]
    fn dense_reports_singular_system_and_all_backends_reject_invalid_inputs() {
        let pattern =
            CorrectionNewtonJacobianPattern::try_new(2, vec![(0, 0), (0, 1), (1, 0), (1, 1)])
                .unwrap();
        let mut dense = CorrectionNewtonLinearWorkspace::new(
            CorrectionNewtonMatrixLayout::Dense,
            2,
            Some(&pattern),
        )
        .unwrap();
        dense.factor_coordinates(&[1.0, 2.0, 2.0, 4.0]).unwrap();
        let mut incompatible_rhs = [1.0, 0.0];
        assert!(matches!(
            dense.solve_in_place(&mut incompatible_rhs),
            Err(CorrectionNewtonLinearError::Singular { .. })
        ));

        for layout in [
            CorrectionNewtonMatrixLayout::Dense,
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let mut workspace =
                CorrectionNewtonLinearWorkspace::new(layout, 2, Some(&pattern)).unwrap();
            assert!(matches!(
                workspace.factor_coordinates(&[1.0]),
                Err(CorrectionNewtonLinearError::CoordinateValueCountMismatch { .. })
            ));
            assert!(matches!(
                workspace.factor_coordinates(&[1.0, f64::NAN, 1.0, 1.0]),
                Err(CorrectionNewtonLinearError::NonFiniteJacobianValue { .. })
            ));
            let mut wrong_rhs = [1.0];
            assert!(matches!(
                workspace.solve_in_place(&mut wrong_rhs),
                Err(CorrectionNewtonLinearError::RightHandSideLengthMismatch { .. })
            ));
        }
    }

    #[test]
    fn every_backend_surfaces_a_typed_singular_failure_with_backend_identity() {
        let pattern =
            CorrectionNewtonJacobianPattern::try_new(2, vec![(0, 0), (0, 1), (1, 0), (1, 1)])
                .unwrap();
        for layout in [
            CorrectionNewtonMatrixLayout::Dense,
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let mut workspace =
                CorrectionNewtonLinearWorkspace::new(layout, 2, Some(&pattern)).unwrap();
            let factor = workspace.factor_coordinates(&[1.0, 2.0, 2.0, 4.0]);
            let failure = match factor {
                Err(error) => error,
                Ok(()) => {
                    let mut rhs = [1.0, 0.0];
                    match workspace.solve_in_place(&mut rhs) {
                        Err(error) => error,
                        Ok(()) => panic!(
                            "singular matrix unexpectedly solved successfully with {:?}",
                            workspace.backend()
                        ),
                    }
                }
            };
            assert!(
                failure.to_string().contains(workspace.backend().name()),
                "failure must identify {:?}: {failure}",
                workspace.backend()
            );
        }
    }

    #[test]
    fn sparse_and_banded_assembly_never_materializes_a_dense_matrix() {
        for layout in [
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let pattern = CorrectionNewtonJacobianPattern::try_new(
                3,
                vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)],
            )
            .unwrap();
            let mut workspace =
                CorrectionNewtonLinearWorkspace::new(layout, 3, Some(&pattern)).unwrap();
            workspace
                .assemble_coordinates(&[4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0])
                .unwrap();
            assert!(workspace.dense_matrix.is_none(), "layout={layout:?}");
            assert!(matches!(&workspace.storage, Storage::Native(_)));
        }
    }

    #[test]
    fn large_structural_workspaces_keep_correction_newton_owned_storage_linear() {
        let dimension = 18_000;
        let mut coordinates = Vec::with_capacity(3 * dimension - 2);
        for row in 0..dimension {
            if row > 0 {
                coordinates.push((row, row - 1));
            }
            coordinates.push((row, row));
            if row + 1 < dimension {
                coordinates.push((row, row + 1));
            }
        }
        let pattern = CorrectionNewtonJacobianPattern::try_new(dimension, coordinates).unwrap();
        let nnz = pattern.coordinates().len();

        for layout in [
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let workspace =
                CorrectionNewtonLinearWorkspace::new(layout, dimension, Some(&pattern)).unwrap();
            assert!(workspace.dense_matrix.is_none(), "layout={layout:?}");
            assert_eq!(workspace.coordinates.as_ref().unwrap().len(), nnz);
            assert_eq!(workspace.entries.len(), nnz);

            // Count only the CorrectionNewton-owned vectors; NativeLinearBackend keeps
            // its allocations private and is deliberately not estimated here.
            let owned_bytes = workspace.coordinates.as_ref().unwrap().capacity()
                * std::mem::size_of::<(usize, usize)>()
                + workspace.entries.capacity() * std::mem::size_of::<(usize, usize, f64)>()
                + workspace.residual_check_scratch.capacity() * std::mem::size_of::<f64>()
                + workspace.row_norm_scratch.capacity() * std::mem::size_of::<f64>();
            println!(
                "CorrectionNewton-owned linear buffers: layout={layout:?}, dimension={dimension}, nnz={nnz}, bytes={owned_bytes}"
            );
            let linear_bound = 128 * (nnz + dimension);
            assert!(
                owned_bytes <= linear_bound,
                "CorrectionNewton-owned structural storage unexpectedly superlinear: layout={layout:?}, bytes={owned_bytes}, bound={linear_bound}"
            );
        }
    }

    #[test]
    fn structural_backend_rejects_dense_matrix_without_a_pattern() {
        let error = match CorrectionNewtonLinearWorkspace::new(
            CorrectionNewtonMatrixLayout::Sparse,
            2,
            None,
        ) {
            Ok(_) => panic!("Sparse requires coordinates"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            CorrectionNewtonLinearError::MissingCoordinatePattern { .. }
        ));
    }
}
