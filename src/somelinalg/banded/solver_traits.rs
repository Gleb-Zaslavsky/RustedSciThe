use super::error::BandedError;

pub trait DirectLinearSolver {
    fn n(&self) -> usize;

    fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError>;

    fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError>;
}

use super::block_tridiagonal_lu::BlockTridiagonalLu;
use super::block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent;
use super::general_lu_partial_pivot::GeneralBandedLuPartialPivot;
use super::lapack_style_banded::LapackStyleBandedLuFaithful;

impl DirectLinearSolver for BlockTridiagonalLu {
    fn n(&self) -> usize {
        self.n()
    }

    fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        BlockTridiagonalLu::solve_in_place(self, rhs)
    }

    fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError> {
        BlockTridiagonalLu::solve_multiple_in_place(self, rhs, nrhs, ldb)
    }
}

impl DirectLinearSolver for BlockTridiagonalLuConsistent {
    fn n(&self) -> usize {
        self.n()
    }

    fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        BlockTridiagonalLuConsistent::solve_in_place(self, rhs)
    }

    fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError> {
        BlockTridiagonalLuConsistent::solve_multiple_in_place(self, rhs, nrhs, ldb)
    }
}

impl DirectLinearSolver for GeneralBandedLuPartialPivot {
    fn n(&self) -> usize {
        self.n()
    }

    fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        GeneralBandedLuPartialPivot::solve_in_place(self, rhs)
    }

    fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError> {
        GeneralBandedLuPartialPivot::solve_multiple_in_place(self, rhs, nrhs, ldb)
    }
}

impl DirectLinearSolver for LapackStyleBandedLuFaithful {
    fn n(&self) -> usize {
        self.n()
    }

    fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        LapackStyleBandedLuFaithful::solve_in_place(self, rhs)
    }

    fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError> {
        LapackStyleBandedLuFaithful::solve_multiple_in_place(self, rhs, nrhs, ldb)
    }
}

use super::solver_policy::LinearSolveError;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{SparseColMat, Triplet};

#[derive(Debug)]
pub struct FaerSparseLuSolver {
    n: usize,

    lu: Lu<usize, f64>,
}

impl FaerSparseLuSolver {
    pub fn from_triplets(
        n: usize,
        triplets: &[Triplet<usize, usize, f64>],
    ) -> Result<Self, LinearSolveError> {
        let sparse =
            SparseColMat::<usize, f64>::try_new_from_triplets(n, n, triplets).map_err(|e| {
                LinearSolveError::Faer(format!("SparseColMat::try_new_from_triplets failed: {e:?}"))
            })?;

        let lu = sparse
            .sp_lu()
            .map_err(|e| LinearSolveError::Faer(format!("sp_lu failed: {e:?}")))?;

        Ok(Self { n, lu })
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

impl DirectLinearSolver for FaerSparseLuSolver {
    fn n(&self) -> usize {
        self.n
    }

    fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), super::error::BandedError> {
        if rhs.len() != self.n {
            return Err(super::error::BandedError::DimensionMismatch);
        }

        let b = faer::Col::from_fn(self.n, |i| rhs[i]);
        let x = self.lu.solve(&b);

        for i in 0..self.n {
            rhs[i] = x[i];
        }

        Ok(())
    }

    fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), super::error::BandedError> {
        if ldb < self.n || rhs.len() < nrhs.saturating_mul(ldb) {
            return Err(super::error::BandedError::DimensionMismatch);
        }

        for col in 0..nrhs {
            let start = col * ldb;
            let end = start + self.n;
            self.solve_in_place(&mut rhs[start..end])?;
        }

        Ok(())
    }
}
