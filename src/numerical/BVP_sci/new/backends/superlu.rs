//! Optional SuperLU backend for explicit BVP_sci Sparse selection.
//!
//! The native library is loaded from `SUPERLU_LIBRARY` (the old
//! `SUPERLU_EXPERIMENTAL_LIBRARY` variable remains accepted for compatibility).
//! It is never selected by default and retains one native LU factor across
//! modified-Newton solves.

use std::env;
use std::path::PathBuf;

use crate::numerical::BVP_sci::new::{
    BvpSciMatrixLayout, BvpSciNewError, BvpSciTelemetry, linear::LinearSystemBackend,
};
use crate::somelinalg::superlu::{CscMatrix, Error as SuperLuError, Factorization, SolverLibrary};
use faer::sparse::{SparseColMat, Triplet};

/// Runtime-loaded SuperLU factors for one BVP Newton matrix.
pub struct SuperLuBackend {
    dimension: usize,
    telemetry: BvpSciTelemetry,
    control: SolverLibrary,
    matrix: Option<SparseColMat<usize, f64>>,
    factor: Option<Factorization>,
}

impl SuperLuBackend {
    pub(crate) fn new(
        dimension: usize,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        if dimension == 0 {
            return Err(BvpSciNewError::InvalidConfiguration(
                "SuperLU sparse system dimension must be positive".into(),
            ));
        }
        let library_path = env::var_os("SUPERLU_LIBRARY")
            .or_else(|| env::var_os("SUPERLU_EXPERIMENTAL_LIBRARY"))
            .map(PathBuf::from)
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration(
                    "SuperLU bridge path is unset (set SUPERLU_LIBRARY)".into(),
                )
            })?;
        let control =
            SolverLibrary::load(&library_path).map_err(|error| BvpSciNewError::LinearBackend {
                message: format!("could not load SuperLU bridge: {error}"),
            })?;
        Ok(Self {
            dimension,
            telemetry,
            control,
            matrix: None,
            factor: None,
        })
    }

    fn map_factor_error(error: SuperLuError) -> BvpSciNewError {
        match error {
            SuperLuError::Factorization(info) if info > 0 => BvpSciNewError::SingularJacobian,
            other => BvpSciNewError::LinearFactorization {
                message: format!("SuperLU factorization failed: {other}"),
            },
        }
    }
}

impl LinearSystemBackend for SuperLuBackend {
    fn layout(&self) -> BvpSciMatrixLayout {
        BvpSciMatrixLayout::Sparse
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        self.factor = None;
        let triplet_started = self.telemetry.start_timing();
        let triplets = entries
            .iter()
            .map(|&(row, column, value)| Triplet::new(row, column, value))
            .collect::<Vec<_>>();
        self.telemetry
            .record_sparse_triplet_materialization(triplet_started);

        let csc_started = self.telemetry.start_timing();
        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(
            self.dimension,
            self.dimension,
            &triplets,
        )
        .map_err(|error| BvpSciNewError::LinearBackend {
            message: format!("SuperLU CSC assembly failed: {error:?}"),
        })?;
        self.telemetry.record_sparse_csc_assembly(csc_started);
        self.matrix = Some(matrix);
        Ok(())
    }

    fn factor(&mut self) -> Result<(), BvpSciNewError> {
        let matrix = self
            .matrix
            .as_ref()
            .ok_or_else(|| BvpSciNewError::LinearBackend {
                message: "SuperLU backend was factored before assembly".into(),
            })?;
        let (symbolic, values) = matrix.as_ref().parts();
        let conversion_started = self.telemetry.start_timing();
        let converted = CscMatrix::try_from_usize(
            self.dimension,
            symbolic.col_ptr(),
            symbolic.row_idx(),
            values,
        );
        self.telemetry
            .record_superlu_abi_conversion(conversion_started);
        let csc32 = converted.map_err(|error| BvpSciNewError::LinearFactorization {
            message: format!("SuperLU CSC ABI conversion failed: {error}"),
        })?;
        let ffi_started = self.telemetry.start_timing();
        match self.control.factorize(&csc32) {
            Ok(factor) => {
                self.telemetry.record_superlu_factorization(
                    ffi_started.map(|started| started.elapsed()),
                    Some(factor.factor_elapsed),
                    Some((
                        factor.factor_l_storage_nnz,
                        factor.factor_u_storage_nnz,
                        factor.factor_l_csc_nnz,
                        factor.factor_u_csc_nnz,
                    )),
                );
                self.factor = Some(factor);
            }
            Err(error) => {
                self.telemetry.record_superlu_factorization(
                    ffi_started.map(|started| started.elapsed()),
                    None,
                    None,
                );
                return Err(Self::map_factor_error(error));
            }
        }
        Ok(())
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        let factor = self
            .factor
            .as_mut()
            .ok_or_else(|| BvpSciNewError::LinearBackend {
                message: "SuperLU backend was solved before factorization".into(),
            })?;
        let ffi_started = self.telemetry.start_timing();
        match factor.solve_in_place(rhs) {
            Ok(native_elapsed) => {
                self.telemetry.record_superlu_solve(
                    ffi_started.map(|started| started.elapsed()),
                    Some(native_elapsed),
                );
                Ok(())
            }
            Err(error) => {
                self.telemetry
                    .record_superlu_solve(ffi_started.map(|started| started.elapsed()), None);
                Err(BvpSciNewError::LinearSolve {
                    message: format!("SuperLU solve failed: {error}"),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the separately built SuperLU bridge DLL"]
    fn retained_factor_solves_multiple_rhs_without_refactorization() {
        let telemetry = BvpSciTelemetry::timings();
        let mut backend =
            SuperLuBackend::new(2, telemetry.clone()).expect("load opt-in SuperLU bridge");
        let entries = [(0, 0, 4.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 3.0)];
        backend.assemble(&entries).expect("assemble CSC matrix");
        backend.factor().expect("factor once");

        let mut rhs_a = [1.0, 1.0];
        backend.solve_in_place(&mut rhs_a).expect("solve first RHS");
        assert!((rhs_a[0] - 0.2).abs() < 1e-12);
        assert!((rhs_a[1] - 0.2).abs() < 1e-12);

        let mut rhs_b = [5.0, 5.0];
        backend
            .solve_in_place(&mut rhs_b)
            .expect("reuse factors for second RHS");
        assert!((rhs_b[0] - 1.0).abs() < 1e-12);
        assert!((rhs_b[1] - 1.0).abs() < 1e-12);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.sparse_triplet_materializations, 1);
        assert_eq!(snapshot.sparse_csc_assemblies, 1);
        assert_eq!(snapshot.superlu.abi_conversions, 1);
        assert_eq!(snapshot.superlu.factor_ffi_calls, 1);
        assert_eq!(snapshot.superlu.native_factorization_calls, 1);
        assert_eq!(snapshot.superlu.solve_ffi_calls, 2);
        assert_eq!(snapshot.superlu.native_solve_calls, 2);
        assert!(snapshot.superlu.factor_l_storage_entries > 0);
        assert!(snapshot.superlu.factor_u_storage_entries > 0);
        assert!(snapshot.superlu.factor_l_csc_nnz > 0);
        assert!(snapshot.superlu.factor_u_csc_nnz > 0);
        assert!(snapshot.superlu.abi_conversion_ms.is_some());
        assert!(snapshot.superlu.native_factorization_ms.is_some());
        assert!(snapshot.superlu.native_solve_ms.is_some());
        assert!(snapshot.validate_contract().is_ok());
    }

    #[test]
    fn superlu_policy_is_sparse_only() {
        let result =
            crate::numerical::BVP_sci::new::linear::BvpSciLinearBackend::new_for_collocation(
                BvpSciMatrixLayout::Dense,
                2,
                0,
                1,
                2,
                0,
                BvpSciTelemetry::disabled(),
                crate::numerical::BVP_sci::new::BvpSciSparseFactorization::SuperLu,
            );
        assert!(matches!(result, Err(BvpSciNewError::UnsupportedRoute(_))));
    }
}
