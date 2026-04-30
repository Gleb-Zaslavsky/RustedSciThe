//! WINDOWS
//!  pacman -S mingw-w64-x86_64-openblas
//!  pacman -S mingw-w64-x86_64-lapacke
//!  pkg-config --cflags --libs lapack
//! add to Path: C:\msys64\mingw64\bin
use super::{Banded, BandedError};

/// LU factors in LAPACK general band storage.
///
/// LAPACK expects:
/// - square matrix n x n
/// - kl subdiagonals
/// - ku superdiagonals
/// - ldab >= 2*kl + ku + 1
///
/// Storage is column-major:
/// ab[row + col * ldab]
///
/// The original matrix entries are placed so that the main diagonal is at row kl + ku.
#[derive(Clone, Debug)]
pub struct BandedLu {
    n: usize,
    kl: usize,
    ku: usize,
    ldab: usize,
    ab: Vec<f64>,
    ipiv: Vec<i32>,
}

impl BandedLu {
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn kl(&self) -> usize {
        self.kl
    }

    #[inline]
    pub fn ku(&self) -> usize {
        self.ku
    }

    #[inline]
    pub fn ldab(&self) -> usize {
        self.ldab
    }

    #[inline]
    pub fn factors(&self) -> &[f64] {
        &self.ab
    }

    #[inline]
    pub fn pivots(&self) -> &[i32] {
        &self.ipiv
    }

    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        if rhs.len() != self.n {
            return Err(BandedError::DimensionMismatch);
        }

        #[cfg(not(feature = "lapack"))]
        {
            let _ = rhs;
            return Err(BandedError::BackendUnavailable);
        }

        #[cfg(feature = "lapack")]
        {
            let mut info = 0_i32;

            unsafe {
                lapack::dgbtrs(
                    b'N',
                    self.n as i32,
                    self.kl as i32,
                    self.ku as i32,
                    1_i32, // nrhs
                    &self.ab,
                    self.ldab as i32,
                    &self.ipiv,
                    rhs,
                    self.n as i32, // ldb
                    &mut info,
                );
            }

            if info < 0 {
                return Err(BandedError::LapackArgument {
                    routine: "dgbtrs",
                    arg_index: -info,
                });
            }

            Ok(())
        }
    }

    pub fn solve_transpose_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        if rhs.len() != self.n {
            return Err(BandedError::DimensionMismatch);
        }

        #[cfg(not(feature = "lapack"))]
        {
            let _ = rhs;
            return Err(BandedError::BackendUnavailable);
        }

        #[cfg(feature = "lapack")]
        {
            let mut info = 0_i32;

            unsafe {
                lapack::dgbtrs(
                    b'T',
                    self.n as i32,
                    self.kl as i32,
                    self.ku as i32,
                    1_i32,
                    &self.ab,
                    self.ldab as i32,
                    &self.ipiv,
                    rhs,
                    self.n as i32,
                    &mut info,
                );
            }

            if info < 0 {
                return Err(BandedError::LapackArgument {
                    routine: "dgbtrs",
                    arg_index: -info,
                });
            }

            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
pub enum GeneralBandedSolver {
    Lapack(BandedLu),
}

impl GeneralBandedSolver {
    pub fn factor(a: &Banded<f64>) -> Result<Self, BandedError> {
        #[cfg(not(feature = "lapack"))]
        {
            let _ = a;
            Err(BandedError::BackendUnavailable)
        }

        #[cfg(feature = "lapack")]
        {
            Ok(Self::Lapack(factor_lapack(a)?))
        }
    }

    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        match self {
            Self::Lapack(lu) => lu.solve_in_place(rhs),
        }
    }

    pub fn solve_transpose_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        match self {
            Self::Lapack(lu) => lu.solve_transpose_in_place(rhs),
        }
    }

    pub fn as_lu(&self) -> &BandedLu {
        match self {
            Self::Lapack(lu) => lu,
        }
    }
}

/// Convert compact banded storage into LAPACK GB storage.
///
/// Compact storage:
///   compact[(ku + i - j) * n + j] = A[i, j]
///
/// LAPACK GB storage:
///   ab[(kl + ku + i - j) + j * ldab] = A[i, j]
///
/// where ldab = 2*kl + ku + 1.
fn compact_to_lapack_gb(a: &Banded<f64>) -> BandedLu {
    let n = a.n();
    let kl = a.kl();
    let ku = a.ku();
    let ldab = 2 * kl + ku + 1;

    let mut ab = vec![0.0_f64; ldab * n];

    for j in 0..n {
        let i0 = j.saturating_sub(ku);
        let i1 = (j + kl + 1).min(n);

        for i in i0..i1 {
            let compact_row = ku + i - j;
            let compact_idx = compact_row * n + j;

            let lapack_row = kl + ku + i - j;
            let lapack_idx = lapack_row + j * ldab;

            ab[lapack_idx] = a.as_slice()[compact_idx];
        }
    }

    BandedLu {
        n,
        kl,
        ku,
        ldab,
        ab,
        ipiv: vec![0_i32; n],
    }
}

#[cfg(feature = "lapack")]
fn factor_lapack(a: &Banded<f64>) -> Result<BandedLu, BandedError> {
    let mut lu = compact_to_lapack_gb(a);
    let mut info = 0_i32;

    unsafe {
        lapack::dgbtrf(
            lu.n as i32,
            lu.n as i32,
            lu.kl as i32,
            lu.ku as i32,
            &mut lu.ab,
            lu.ldab as i32,
            &mut lu.ipiv,
            &mut info,
        );
    }

    if info < 0 {
        return Err(BandedError::LapackArgument {
            routine: "dgbtrf",
            arg_index: -info,
        });
    }

    if info > 0 {
        return Err(BandedError::Singular {
            index: info as usize - 1,
        });
    }

    Ok(lu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::somelinalg::banded::Banded;
    #[cfg(feature = "openblas-system")]
    extern crate openblas_src as _;
    #[test]
    fn compact_layout_indexing() {
        let mut a = Banded::<f64>::zeros(5, 1, 2).unwrap();

        a[(0, 0)] = 10.0;
        a[(0, 1)] = 11.0;
        a[(0, 2)] = 12.0;

        a[(1, 0)] = 20.0;
        a[(1, 1)] = 21.0;
        a[(1, 2)] = 22.0;
        a[(1, 3)] = 23.0;

        assert_eq!(a[(0, 0)], 10.0);
        assert_eq!(a[(0, 1)], 11.0);
        assert_eq!(a[(1, 0)], 20.0);
        assert_eq!(a[(1, 3)], 23.0);
    }

    #[test]
    fn convert_to_lapack_storage_shape() {
        let mut a = Banded::<f64>::zeros(4, 1, 1).unwrap();
        a[(0, 0)] = 4.0;
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 2.0;
        a[(1, 1)] = 5.0;
        a[(1, 2)] = 1.0;
        a[(2, 1)] = 3.0;
        a[(2, 2)] = 6.0;
        a[(2, 3)] = 1.0;
        a[(3, 2)] = 4.0;
        a[(3, 3)] = 7.0;

        let lu = compact_to_lapack_gb(&a);
        assert_eq!(lu.n, 4);
        assert_eq!(lu.kl, 1);
        assert_eq!(lu.ku, 1);
        assert_eq!(lu.ldab, 4);
        assert_eq!(lu.ab.len(), 16);
        assert_eq!(lu.ipiv.len(), 4);
    }

    #[cfg(feature = "lapack")]
    #[test]
    fn factor_and_solve_tridiagonal() {
        // Matrix:
        // [ 4 1 0 0 ]
        // [ 2 5 1 0 ]
        // [ 0 3 6 1 ]
        // [ 0 0 4 7 ]
        //
        // rhs = A * [1,1,1,1]^T = [5,8,10,11]^T

        let mut a = Banded::<f64>::zeros(4, 1, 1).unwrap();

        a[(0, 0)] = 4.0;
        a[(0, 1)] = 1.0;

        a[(1, 0)] = 2.0;
        a[(1, 1)] = 5.0;
        a[(1, 2)] = 1.0;

        a[(2, 1)] = 3.0;
        a[(2, 2)] = 6.0;
        a[(2, 3)] = 1.0;

        a[(3, 2)] = 4.0;
        a[(3, 3)] = 7.0;

        let solver = GeneralBandedSolver::factor(&a).unwrap();
        let mut rhs = vec![5.0, 8.0, 10.0, 11.0];
        solver.solve_in_place(&mut rhs).unwrap();

        for x in rhs {
            assert!((x - 1.0).abs() < 1e-12);
        }
    }
    #[cfg(feature = "lapack")]
    #[test]
    fn lapack_try() {
        println!("Trying LAPACK...");

        let mut info = 0;

        unsafe {
            lapack::dlamch(b'E'); // просто вызов
        }

        println!("OK");
    }
}
