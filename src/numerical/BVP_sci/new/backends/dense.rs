//! Dense collocation backend, intended mainly for small fully coupled systems.

use crate::numerical::BVP_sci::new::{
    BvpSciMatrixLayout, error::BvpSciNewError, linear::LinearSystemBackend,
};
use nalgebra::{DMatrix, DVectorViewMut, Dyn, LU};

/// Dense global collocation storage with nalgebra LU factorization.
///
/// The assembled matrix stays available for a refresh while nalgebra owns the
/// factorization. This is intentionally a simple, predictable path for small
/// fully coupled systems; its matrix clone is an explicit optimization debt,
/// not an invisible conversion.
#[derive(Clone, Debug)]
pub struct DenseBackend {
    dimension: usize,
    matrix: DMatrix<f64>,
    factorization: Option<LU<f64, Dyn, Dyn>>,
}

impl DenseBackend {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            matrix: DMatrix::zeros(dimension, dimension),
            factorization: None,
        }
    }

    pub fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        // Reusing the matrix allocation matters because Newton may refresh
        // numeric values many times without changing the global dimension.
        self.matrix.fill(0.0);
        for &(row, column, value) in entries {
            if row >= self.dimension || column >= self.dimension {
                return Err(BvpSciNewError::LinearBackend {
                    message: format!("dense entry ({row}, {column}) is outside the matrix"),
                });
            }
            // The collocation assembler emits additive contributions from
            // adjacent intervals at shared node columns. Keep the same
            // duplicate-entry semantics as the sparse triplet backend.
            self.matrix[(row, column)] += value;
        }
        self.factorization = None;
        Ok(())
    }

    pub fn factor(&mut self) -> Result<(), BvpSciNewError> {
        // nalgebra's LU owns its input. Keep the assembly matrix reusable;
        // this is the one deliberate dense-backend copy for now.
        self.factorization = Some(self.matrix.clone().lu());
        Ok(())
    }

    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        if rhs.len() != self.dimension {
            return Err(BvpSciNewError::LinearBackend {
                message: "dense solve vector has an invalid length".into(),
            });
        }
        let factorization =
            self.factorization
                .as_ref()
                .ok_or_else(|| BvpSciNewError::LinearBackend {
                    message: "dense backend was solved before factorization".into(),
                })?;
        let mut rhs = DVectorViewMut::from_slice(rhs, self.dimension);
        if !factorization.solve_mut(&mut rhs) {
            return Err(BvpSciNewError::SingularJacobian);
        }
        Ok(())
    }
}

impl LinearSystemBackend for DenseBackend {
    fn layout(&self) -> BvpSciMatrixLayout {
        BvpSciMatrixLayout::Dense
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        Self::assemble(self, entries)
    }

    fn factor(&mut self) -> Result<(), BvpSciNewError> {
        Self::factor(self)
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        Self::solve_in_place(self, rhs)
    }
}
