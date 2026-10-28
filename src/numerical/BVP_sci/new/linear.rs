//! Backend-neutral linear-system boundary.
//!
//! The collocation core sees one contract.  Backend implementations assemble
//! directly into their native storage and own factorization/workspace reuse;
//! dense-to-sparse and sparse-to-banded conversions are not part of the new
//! design.

use super::{
    backends::{BandedBackend, DenseBackend, ReusableSparseExperimentalBackend, SparseBackend},
    config::{BvpSciMatrixLayout, BvpSciSparseFactorization},
    error::BvpSciNewError,
    telemetry::BvpSciTelemetry,
};
use crate::somelinalg::bvp_sparse_lu::ReusableSparseLuStrategy;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

static SUPERLU_FALLBACK_WARNINGS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

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
            BvpSciMatrixLayout::BandedAuto => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "BandedAuto must be resolved from symbolic dependencies before backend creation".into(),
                ));
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
            BvpSciMatrixLayout::BandedAuto => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "BandedAuto must be resolved from symbolic dependencies before backend creation".into(),
                ));
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

/// Private solver dispatch that preserves the public `NativeLinearBackend`
/// variants while supporting optional Sparse factorization policies.
pub(crate) enum BvpSciLinearBackend {
    Standard(NativeLinearBackend),
    ReusableSparseExperimental(ReusableSparseExperimentalBackend),
    #[cfg(feature = "superlu")]
    SuperLu(super::backends::SuperLuBackend),
}

impl BvpSciLinearBackend {
    #[allow(deprecated)]
    pub(crate) fn new_for_collocation(
        layout: BvpSciMatrixLayout,
        dimension: usize,
        pattern_nnz: usize,
        state_dimension: usize,
        node_count: usize,
        parameter_dimension: usize,
        telemetry: BvpSciTelemetry,
        sparse_factorization: BvpSciSparseFactorization,
    ) -> Result<Self, BvpSciNewError> {
        #[allow(deprecated)]
        let wants_superlu = matches!(
            sparse_factorization,
            BvpSciSparseFactorization::SuperLu | BvpSciSparseFactorization::SuperLuExperimental
        );
        if wants_superlu {
            if layout != BvpSciMatrixLayout::Sparse {
                return Err(BvpSciNewError::UnsupportedRoute(
                    "SuperLU factorization requires Sparse matrix layout".into(),
                ));
            }
            #[cfg(feature = "superlu")]
            {
                match super::backends::SuperLuBackend::new(dimension, telemetry.clone()) {
                    Ok(backend) => return Ok(Self::SuperLu(backend)),
                    Err(error) => {
                        return Self::superlu_fallback(
                            layout,
                            dimension,
                            pattern_nnz,
                            state_dimension,
                            node_count,
                            parameter_dimension,
                            telemetry,
                            error.to_string(),
                        );
                    }
                }
            }
            #[cfg(not(feature = "superlu"))]
            {
                return Self::superlu_fallback(
                    layout,
                    dimension,
                    pattern_nnz,
                    state_dimension,
                    node_count,
                    parameter_dimension,
                    telemetry,
                    "crate was compiled without Cargo feature `superlu`".into(),
                );
            }
        }

        let reusable_strategy = match sparse_factorization {
            BvpSciSparseFactorization::ReusableFaerExperimental => {
                Some(ReusableSparseLuStrategy::Auto)
            }
            BvpSciSparseFactorization::ReusableFaerSimplicialExperimental => {
                Some(ReusableSparseLuStrategy::Simplicial)
            }
            BvpSciSparseFactorization::StandardFaer
            | BvpSciSparseFactorization::SuperLu
            | BvpSciSparseFactorization::SuperLuExperimental => None,
        };
        if layout == BvpSciMatrixLayout::Sparse {
            if let Some(reusable_strategy) = reusable_strategy {
                if dimension == 0 {
                    return Err(BvpSciNewError::InvalidConfiguration(
                        "linear system dimension must be positive".into(),
                    ));
                }
                return Ok(Self::ReusableSparseExperimental(
                    ReusableSparseExperimentalBackend::new(
                        dimension,
                        pattern_nnz,
                        telemetry,
                        reusable_strategy,
                    ),
                ));
            }
        }
        Ok(Self::Standard(NativeLinearBackend::new_for_collocation(
            layout,
            dimension,
            pattern_nnz,
            state_dimension,
            node_count,
            parameter_dimension,
            telemetry,
        )?))
    }

    fn superlu_fallback(
        layout: BvpSciMatrixLayout,
        dimension: usize,
        pattern_nnz: usize,
        state_dimension: usize,
        node_count: usize,
        parameter_dimension: usize,
        telemetry: BvpSciTelemetry,
        reason: String,
    ) -> Result<Self, BvpSciNewError> {
        telemetry.record_superlu_fallback();
        let warned = SUPERLU_FALLBACK_WARNINGS.get_or_init(|| Mutex::new(HashSet::new()));
        if let Ok(mut warned) = warned.lock() {
            if warned.insert(reason.clone()) {
                eprintln!(
                    "[BVP_sci] SuperLU was requested but unavailable ({reason}); falling back to StandardFaer Sparse. Enable Cargo feature `superlu` and set SUPERLU_LIBRARY to use SuperLU."
                );
            }
        }
        Ok(Self::Standard(NativeLinearBackend::new_for_collocation(
            layout,
            dimension,
            pattern_nnz,
            state_dimension,
            node_count,
            parameter_dimension,
            telemetry,
        )?))
    }

    pub(crate) fn layout(&self) -> BvpSciMatrixLayout {
        match self {
            Self::Standard(backend) => backend.layout(),
            Self::ReusableSparseExperimental(backend) => backend.layout(),
            #[cfg(feature = "superlu")]
            Self::SuperLu(backend) => backend.layout(),
        }
    }

    pub(crate) fn assemble(
        &mut self,
        entries: &[(usize, usize, f64)],
    ) -> Result<(), BvpSciNewError> {
        match self {
            Self::Standard(backend) => backend.assemble(entries),
            Self::ReusableSparseExperimental(backend) => backend.assemble(entries),
            #[cfg(feature = "superlu")]
            Self::SuperLu(backend) => backend.assemble(entries),
        }
    }

    pub(crate) fn factor(&mut self) -> Result<(), BvpSciNewError> {
        match self {
            Self::Standard(backend) => backend.factor(),
            Self::ReusableSparseExperimental(backend) => backend.factor(),
            #[cfg(feature = "superlu")]
            Self::SuperLu(backend) => backend.factor(),
        }
    }

    pub(crate) fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        match self {
            Self::Standard(backend) => backend.solve_in_place(rhs),
            Self::ReusableSparseExperimental(backend) => backend.solve_in_place(rhs),
            #[cfg(feature = "superlu")]
            Self::SuperLu(backend) => backend.solve_in_place(rhs),
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

    #[cfg(not(feature = "superlu"))]
    #[test]
    fn superlu_policy_without_feature_falls_back_to_standard_faer() {
        let telemetry = BvpSciTelemetry::detailed();
        let result = BvpSciLinearBackend::new_for_collocation(
            BvpSciMatrixLayout::Sparse,
            2,
            2,
            1,
            2,
            0,
            telemetry.clone(),
            BvpSciSparseFactorization::SuperLu,
        )
        .expect("missing optional SuperLU feature should use StandardFaer");
        assert!(matches!(result, BvpSciLinearBackend::Standard(_)));
        assert_eq!(telemetry.snapshot().superlu_fallbacks, 1);
    }

    #[test]
    fn superlu_fallback_is_sparse_and_observable() {
        let telemetry = BvpSciTelemetry::detailed();
        let backend = BvpSciLinearBackend::superlu_fallback(
            BvpSciMatrixLayout::Sparse,
            3,
            3,
            1,
            3,
            0,
            telemetry.clone(),
            "simulated bridge load failure".into(),
        )
        .expect("fallback should construct the StandardFaer sparse route");
        assert!(matches!(&backend, BvpSciLinearBackend::Standard(_)));
        assert_eq!(backend.layout(), BvpSciMatrixLayout::Sparse);
        assert_eq!(telemetry.snapshot().superlu_fallbacks, 1);
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
        backend
            .factor()
            .expect("dense factorization should be created");
        let error = backend
            .solve_in_place(&mut [1.0, 1.0])
            .expect_err("singular dense solve must fail");
        assert!(matches!(error, BvpSciNewError::SingularJacobian));
        assert_eq!(error.status(), Some(BvpSciStatus::SingularJacobian));
    }

    #[test]
    fn reusable_sparse_backend_refactors_same_pattern_and_solves_in_place() {
        let telemetry = BvpSciTelemetry::detailed();
        let mut backend = BvpSciLinearBackend::new_for_collocation(
            BvpSciMatrixLayout::Sparse,
            3,
            5,
            3,
            1,
            0,
            telemetry.clone(),
            BvpSciSparseFactorization::ReusableFaerExperimental,
        )
        .expect("experimental sparse route should construct");

        for scale in [1.0, 2.0] {
            backend
                .assemble(&[
                    (0, 0, 4.0 * scale),
                    (1, 0, 1.0 * scale),
                    (1, 1, 5.0 * scale),
                    (2, 1, 1.0 * scale),
                    (2, 2, 6.0 * scale),
                ])
                .expect("same-pattern sparse matrix should assemble");
            backend
                .factor()
                .expect("numeric factorization should succeed");
            let mut rhs = [1.0 * scale, 2.0 * scale, 3.0 * scale];
            backend
                .solve_in_place(&mut rhs)
                .expect("in-place sparse solve should succeed");
            assert!((rhs[0] - 0.25).abs() < 1e-12);
            assert!((rhs[1] - 0.35).abs() < 1e-12);
            assert!((rhs[2] - 0.44166666666666665).abs() < 1e-12);
        }

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.sparse_symbolic_analyses, 1);
        assert_eq!(snapshot.sparse_symbolic_reuses, 1);
        assert_eq!(snapshot.sparse_numeric_factorizations, 2);
        assert_eq!(snapshot.sparse_csc_assemblies, 2);
        assert!(snapshot.validate_contract().is_ok());
    }
}
