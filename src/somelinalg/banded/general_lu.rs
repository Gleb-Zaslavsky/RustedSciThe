use super::{error::BandedError, lu_storage::BandedLuStorage, storage::Banded};

#[derive(Clone, Debug)]
pub struct GeneralBandedLuNoPivot {
    storage: BandedLuStorage,
    is_factorized: bool,
    pivot_epsilon: f64,
}

impl GeneralBandedLuNoPivot {
    pub fn new(n: usize, kl: usize, ku: usize) -> Result<Self, BandedError> {
        Ok(Self {
            storage: BandedLuStorage::new(n, kl, ku)?,
            is_factorized: false,
            pivot_epsilon: 1e-14,
        })
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.storage.n()
    }

    #[inline]
    pub fn kl(&self) -> usize {
        self.storage.kl()
    }

    #[inline]
    pub fn ku(&self) -> usize {
        self.storage.ku()
    }

    #[inline]
    pub fn ext_ku(&self) -> usize {
        self.storage.ext_ku()
    }

    #[inline]
    pub fn storage(&self) -> &BandedLuStorage {
        &self.storage
    }

    pub fn set_pivot_epsilon(&mut self, eps: f64) {
        self.pivot_epsilon = eps.max(0.0);
    }

    /// Factor A into in-place banded LU without pivoting.
    ///
    /// Storage convention after factorization:
    /// - diagonal and upper part contain U
    /// - strict lower part contains multipliers of L
    /// - unit diagonal of L is implicit
    pub fn factor_from(&mut self, a: &Banded<f64>) -> Result<(), BandedError> {
        if a.n() != self.n() || a.kl() != self.kl() || a.ku() != self.ku() {
            return Err(BandedError::DimensionMismatch);
        }

        self.storage.copy_from_compact(a)?;
        self.is_factorized = false;

        let n = self.n();
        let kl = self.kl();
        let ext_ku = self.ext_ku();

        for k in 0..n {
            let pivot = self.storage.get(k, k).unwrap_or(0.0);
            if pivot.abs() <= self.pivot_epsilon {
                return Err(BandedError::ZeroPivot {
                    index: k,
                    value: pivot,
                });
            }

            let i_max = (k + kl + 1).min(n);
            let j_max = (k + ext_ku + 1).min(n);

            // Eliminate entries below pivot in column k
            for i in (k + 1)..i_max {
                let a_ik = self.storage.get(i, k).unwrap_or(0.0);
                if a_ik == 0.0 {
                    continue;
                }

                let multiplier = a_ik / pivot;

                // Store L multiplier in place of A[i, k]
                {
                    let slot = self.storage.get_mut(i, k).expect("L slot must exist");
                    *slot = multiplier;
                }

                // Update trailing row segment inside expanded band
                for j in (k + 1)..j_max {
                    let u_kj = self.storage.get(k, j).unwrap_or(0.0);
                    if u_kj == 0.0 {
                        continue;
                    }

                    if let Some(a_ij) = self.storage.get_mut(i, j) {
                        *a_ij -= multiplier * u_kj;
                    }
                }
            }
        }

        self.is_factorized = true;
        Ok(())
    }

    /// Solve Ax = b in place, where A has already been factorized into LU.
    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        if rhs.len() != self.n() {
            return Err(BandedError::DimensionMismatch);
        }

        let n = self.n();
        let kl = self.kl();
        let ext_ku = self.ext_ku();

        // Forward solve Ly = b, L has implicit unit diagonal
        for i in 0..n {
            let j0 = i.saturating_sub(kl);
            let mut sum = rhs[i];

            for j in j0..i {
                let lij = self.storage.get(i, j).unwrap_or(0.0);
                if lij != 0.0 {
                    sum -= lij * rhs[j];
                }
            }

            rhs[i] = sum;
        }

        // Back solve Ux = y
        for i in (0..n).rev() {
            let j1 = (i + ext_ku + 1).min(n);
            let mut sum = rhs[i];

            for j in (i + 1)..j1 {
                let uij = self.storage.get(i, j).unwrap_or(0.0);
                if uij != 0.0 {
                    sum -= uij * rhs[j];
                }
            }

            let uii = self.storage.get(i, i).unwrap_or(0.0);
            if uii.abs() <= self.pivot_epsilon {
                return Err(BandedError::ZeroPivot {
                    index: i,
                    value: uii,
                });
            }

            rhs[i] = sum / uii;
        }

        Ok(())
    }

    /// Solve A X = B in place for multiple RHS.
    ///
    /// `rhs` is interpreted as column-major with leading dimension `ldb`.
    /// Column k starts at rhs[k * ldb].
    pub fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }

        let n = self.n();
        if ldb < n || rhs.len() < nrhs.saturating_mul(ldb) {
            return Err(BandedError::InvalidRhsLayout {
                rhs_len: rhs.len(),
                n,
                nrhs,
                ldb,
            });
        }

        for col in 0..nrhs {
            let start = col * ldb;
            let end = start + n;
            self.solve_in_place(&mut rhs[start..end])?;
        }

        Ok(())
    }
    /// Extract dense L from factorized storage.
    /// Unit diagonal is implicit in storage and materialized here.
    pub fn extract_l_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }

        let n = self.n();
        let kl = self.kl();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            l[i][i] = 1.0;

            let j0 = i.saturating_sub(kl);
            for j in j0..i {
                l[i][j] = self.storage.get(i, j).unwrap_or(0.0);
            }
        }

        Ok(l)
    }

    /// Extract dense U from factorized storage.
    pub fn extract_u_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }

        let n = self.n();
        let ext_ku = self.ext_ku();
        let mut u = vec![vec![0.0; n]; n];

        for i in 0..n {
            let j1 = (i + ext_ku + 1).min(n);
            for j in i..j1 {
                u[i][j] = self.storage.get(i, j).unwrap_or(0.0);
            }
        }

        Ok(u)
    }

    /// Reconstruct dense LU product from factorized storage.
    /// Debug/test helper only.
    pub fn reconstruct_lu_product_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        let l = self.extract_l_dense()?;
        let u = self.extract_u_dense()?;
        Ok(super::ops::dense_matmul(&l, &u))
    }
}
