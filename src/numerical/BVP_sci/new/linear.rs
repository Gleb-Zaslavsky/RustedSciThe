//! Backend-neutral linear-system boundary.
//!
//! The collocation core sees one contract.  Backend implementations assemble
//! directly into their native storage and own factorization/workspace reuse;
//! dense-to-sparse and sparse-to-banded conversions are not part of the new
//! design.

use super::{
    backends::{BandedBackend, DenseBackend, SparseBackend},
    config::BvpSciMatrixLayout,
    error::BvpSciNewError,
    telemetry::BvpSciTelemetry,
};

pub trait LinearSystemBackend {
    /// Return the immutable layout selected for this backend instance.
    fn layout(&self) -> BvpSciMatrixLayout;
    /// Return the global collocation system dimension.
    fn dimension(&self) -> usize;
    /// Assemble numeric entries directly into native backend storage.
    fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError>;
    /// Factor the most recently assembled matrix.
    fn factor(&mut self) -> Result<(), BvpSciNewError>;
    /// Solve in place without allocating a result vector per Newton step.
    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError>;
}

/// Backend selected once for one global collocation system.
///
/// The enum keeps dispatch outside the Newton scalar loops.  Each variant
/// assembles directly into its native matrix representation.
pub enum NativeLinearBackend {
    /// Nalgebra dense matrix/LU path for compact fully coupled systems.
    Dense(DenseBackend),
    /// Faer sparse path; no intermediate dense global matrix is formed.
    Sparse(SparseBackend),
    /// Native band storage selected with explicit lower/upper widths.
    Banded(BandedBackend),
}

impl NativeLinearBackend {
    pub fn new(
        layout: BvpSciMatrixLayout,
        dimension: usize,
        pattern_nnz: usize,
    ) -> Result<Self, BvpSciNewError> {
        if dimension == 0 {
            return Err(BvpSciNewError::InvalidConfiguration(
                "linear system dimension must be positive".into(),
            ));
        }
        // Layout is resolved once per mesh. The Newton loop later calls the
        // enum through one fixed variant, so it never performs conversions or
        // re-evaluates the user's layout policy for individual entries.
        Ok(match layout {
            BvpSciMatrixLayout::Dense => Self::Dense(DenseBackend::new(dimension)),
            BvpSciMatrixLayout::Sparse => Self::Sparse(SparseBackend::new_with_telemetry(
                dimension,
                pattern_nnz,
                BvpSciTelemetry::disabled(),
            )),
            BvpSciMatrixLayout::Banded { lower, upper } => {
                Self::Banded(BandedBackend::new(dimension, lower, upper)?)
            }
        })
    }

    /// Construct a backend with collocation dimensions available to the
    /// bordered-banded route. Dense and Sparse retain the same native paths;
    /// Banded receives the node/state partition and writes directly into
    /// `[T U; V D]` storage.
    pub fn new_for_collocation(
        layout: BvpSciMatrixLayout,
        dimension: usize,
        pattern_nnz: usize,
        state_dimension: usize,
        node_count: usize,
        parameter_dimension: usize,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        if dimension == 0 {
            return Err(BvpSciNewError::InvalidConfiguration(
                "linear system dimension must be positive".into(),
            ));
        }
        Ok(match layout {
            BvpSciMatrixLayout::Dense => Self::Dense(DenseBackend::new(dimension)),
            BvpSciMatrixLayout::Sparse => Self::Sparse(SparseBackend::new_with_telemetry(
                dimension,
                pattern_nnz,
                telemetry,
            )),
            BvpSciMatrixLayout::Banded { lower, upper } => {
                Self::Banded(BandedBackend::new_for_collocation(
                    dimension,
                    lower,
                    upper,
                    state_dimension,
                    node_count,
                    parameter_dimension,
                    telemetry,
                )?)
            }
        })
    }

    pub fn layout(&self) -> BvpSciMatrixLayout {
        match self {
            Self::Dense(backend) => backend.layout(),
            Self::Sparse(backend) => backend.layout(),
            Self::Banded(backend) => backend.layout(),
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::Dense(backend) => backend.dimension(),
            Self::Sparse(backend) => backend.dimension(),
            Self::Banded(backend) => backend.dimension(),
        }
    }

    pub fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        // `entries` is the only backend-neutral boundary. Implementations
        // validate and write directly into their native storage.
        match self {
            Self::Dense(backend) => backend.assemble(entries),
            Self::Sparse(backend) => backend.assemble(entries),
            Self::Banded(backend) => backend.assemble(entries),
        }
    }

    pub fn factor(&mut self) -> Result<(), BvpSciNewError> {
        match self {
            Self::Dense(backend) => backend.factor(),
            Self::Sparse(backend) => backend.factor(),
            Self::Banded(backend) => backend.factor(),
        }
    }

    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        // All backends mutate the caller-owned Newton RHS. This avoids a
        // result vector allocation on every modified-Newton iteration.
        match self {
            Self::Dense(backend) => backend.solve_in_place(rhs),
            Self::Sparse(backend) => backend.solve_in_place(rhs),
            Self::Banded(backend) => backend.solve_in_place(rhs),
        }
    }
}

impl LinearSystemBackend for NativeLinearBackend {
    fn layout(&self) -> BvpSciMatrixLayout {
        NativeLinearBackend::layout(self)
    }

    fn dimension(&self) -> usize {
        NativeLinearBackend::dimension(self)
    }

    fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        NativeLinearBackend::assemble(self, entries)
    }

    fn factor(&mut self) -> Result<(), BvpSciNewError> {
        NativeLinearBackend::factor(self)
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        NativeLinearBackend::solve_in_place(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_sci::new::BvpSciStatus;

    #[test]
    fn invalid_banded_dimensions_return_typed_configuration_error() {
        let result = NativeLinearBackend::new(
            BvpSciMatrixLayout::Banded {
                lower: usize::MAX,
                upper: 0,
            },
            1,
            0,
        );
        assert!(matches!(
            result,
            Err(BvpSciNewError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn singular_banded_factorization_keeps_scipy_status_two() {
        let mut backend =
            NativeLinearBackend::new(BvpSciMatrixLayout::Banded { lower: 2, upper: 2 }, 3, 3)
                .expect("banded backend should construct");
        backend
            .assemble(&[(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)])
            .expect("singular test matrix should assemble");
        let error = backend.factor().expect_err("zero matrix must be singular");
        assert_eq!(error.status(), Some(BvpSciStatus::SingularJacobian));
    }

    #[test]
    fn singular_dense_solve_uses_typed_scipy_status_two() {
        let mut backend = NativeLinearBackend::new(BvpSciMatrixLayout::Dense, 2, 0)
            .expect("dense backend should construct");
        backend
            .assemble(&[(0, 0, 0.0), (1, 1, 0.0)])
            .expect("singular dense matrix should assemble");
        backend.factor().expect("dense factorization should be created");
        let error = backend
            .solve_in_place(&mut [1.0, 1.0])
            .expect_err("singular dense solve must fail");
        assert!(matches!(error, BvpSciNewError::SingularJacobian));
        assert_eq!(error.status(), Some(BvpSciStatus::SingularJacobian));
    }
}
