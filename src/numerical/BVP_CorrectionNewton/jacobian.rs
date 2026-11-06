//! Stable coordinate layout and callback contract for CorrectionNewton Jacobians.
//!
//! The pattern is prepared once and values are written into a caller-owned
//! buffer. Sparse and Banded workspaces can consume this representation
//! directly; the current Dense parity workspace scatters it into Dense storage.

use nalgebra::DMatrix;
use std::collections::HashSet;
use thiserror::Error;

/// Fixed structural coordinates for one square Newton Jacobian.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorrectionNewtonJacobianPattern {
    dimension: usize,
    coordinates: Vec<(usize, usize)>,
}

impl CorrectionNewtonJacobianPattern {
    /// Validate and retain coordinates in callback value order.
    pub fn try_new(
        dimension: usize,
        coordinates: Vec<(usize, usize)>,
    ) -> Result<Self, CorrectionNewtonJacobianError> {
        if dimension == 0 {
            return Err(CorrectionNewtonJacobianError::ZeroDimension);
        }
        let mut seen = HashSet::with_capacity(coordinates.len());
        for &(row, column) in &coordinates {
            if row >= dimension || column >= dimension {
                return Err(CorrectionNewtonJacobianError::CoordinateOutOfBounds {
                    row,
                    column,
                    dimension,
                });
            }
            if !seen.insert((row, column)) {
                return Err(CorrectionNewtonJacobianError::DuplicateCoordinate { row, column });
            }
        }
        Ok(Self {
            dimension,
            coordinates,
        })
    }

    /// Construct the full row-major pattern for an explicitly Dense callback.
    pub fn dense(dimension: usize) -> Result<Self, CorrectionNewtonJacobianError> {
        if dimension == 0 {
            return Err(CorrectionNewtonJacobianError::ZeroDimension);
        }
        let count = dimension
            .checked_mul(dimension)
            .ok_or(CorrectionNewtonJacobianError::PatternSizeOverflow { dimension })?;
        let mut coordinates = Vec::with_capacity(count);
        for row in 0..dimension {
            for column in 0..dimension {
                coordinates.push((row, column));
            }
        }
        Self::try_new(dimension, coordinates)
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn nnz(&self) -> usize {
        self.coordinates.len()
    }

    pub fn coordinates(&self) -> &[(usize, usize)] {
        &self.coordinates
    }

    /// Exact scalar lower/upper bandwidth of the fixed structural pattern.
    pub fn bandwidth(&self) -> (usize, usize) {
        self.coordinates
            .iter()
            .fold((0, 0), |(lower, upper), &(row, column)| {
                (
                    lower.max(row.saturating_sub(column)),
                    upper.max(column.saturating_sub(row)),
                )
            })
    }

    pub fn values_buffer(&self) -> Vec<f64> {
        vec![0.0; self.nnz()]
    }

    pub fn validate_values(&self, values: &[f64]) -> Result<(), CorrectionNewtonJacobianError> {
        if values.len() != self.nnz() {
            return Err(CorrectionNewtonJacobianError::ValuesLengthMismatch {
                expected: self.nnz(),
                actual: values.len(),
            });
        }
        for (&(row, column), &value) in self.coordinates.iter().zip(values) {
            if !value.is_finite() {
                return Err(CorrectionNewtonJacobianError::NonFiniteValue { row, column });
            }
        }
        Ok(())
    }

    /// Dense assembly is currently used only by the Dense parity backend.
    pub(crate) fn assemble_dense(
        &self,
        values: &[f64],
    ) -> Result<DMatrix<f64>, CorrectionNewtonJacobianError> {
        self.validate_values(values)?;
        let mut matrix = DMatrix::zeros(self.dimension, self.dimension);
        for (&(row, column), &value) in self.coordinates.iter().zip(values) {
            matrix[(row, column)] = value;
        }
        Ok(matrix)
    }
}

/// Errors found while preparing or validating a coordinate Jacobian.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CorrectionNewtonJacobianError {
    #[error("Jacobian dimension must be positive")]
    ZeroDimension,
    #[error("dense Jacobian pattern size overflows usize for dimension {dimension}")]
    PatternSizeOverflow { dimension: usize },
    #[error("Jacobian coordinate ({row}, {column}) is outside dimension {dimension}")]
    CoordinateOutOfBounds {
        row: usize,
        column: usize,
        dimension: usize,
    },
    #[error("Jacobian pattern contains duplicate coordinate ({row}, {column})")]
    DuplicateCoordinate { row: usize, column: usize },
    #[error("Jacobian values buffer has length {actual}, expected {expected}")]
    ValuesLengthMismatch { expected: usize, actual: usize },
    #[error("Jacobian value at ({row}, {column}) is not finite")]
    NonFiniteValue { row: usize, column: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_preserves_declared_value_order_and_assembles_dense() {
        let pattern = CorrectionNewtonJacobianPattern::try_new(3, vec![(2, 1), (0, 0), (1, 2)])
            .expect("valid sparse pattern should be retained");
        assert_eq!(pattern.coordinates(), &[(2, 1), (0, 0), (1, 2)]);
        assert_eq!(pattern.nnz(), 3);
        let matrix = pattern
            .assemble_dense(&[4.0, 5.0, 6.0])
            .expect("matching values should assemble");
        assert_eq!(matrix[(2, 1)], 4.0);
        assert_eq!(matrix[(0, 0)], 5.0);
        assert_eq!(matrix[(1, 2)], 6.0);
        assert_eq!(matrix[(0, 1)], 0.0);
    }

    #[test]
    fn pattern_rejects_duplicate_and_out_of_range_coordinates() {
        assert!(matches!(
            CorrectionNewtonJacobianPattern::try_new(2, vec![(0, 1), (0, 1)]),
            Err(CorrectionNewtonJacobianError::DuplicateCoordinate { .. })
        ));
        assert!(matches!(
            CorrectionNewtonJacobianPattern::try_new(2, vec![(2, 0)]),
            Err(CorrectionNewtonJacobianError::CoordinateOutOfBounds { .. })
        ));
    }

    #[test]
    fn dense_pattern_checks_size_and_values() {
        assert!(matches!(
            CorrectionNewtonJacobianPattern::dense(0),
            Err(CorrectionNewtonJacobianError::ZeroDimension)
        ));
        let pattern = CorrectionNewtonJacobianPattern::dense(2).expect("2x2 pattern should fit");
        assert_eq!(pattern.coordinates(), &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert!(matches!(
            pattern.assemble_dense(&[1.0]),
            Err(CorrectionNewtonJacobianError::ValuesLengthMismatch { .. })
        ));
        assert!(matches!(
            pattern.assemble_dense(&[1.0, 0.0, f64::NAN, 1.0]),
            Err(CorrectionNewtonJacobianError::NonFiniteValue { row: 1, column: 0 })
        ));
    }
}
