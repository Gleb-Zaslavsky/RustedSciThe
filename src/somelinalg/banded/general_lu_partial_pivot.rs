use super::{
    dense_block_kernels::{dense_lu_pivot_in_place, dense_lu_pivot_solve_in_place, idx},
    error::BandedError,
    lu_storage::BandedLuStorage,
    ops::dense_matmul,
    storage::Banded,
};

#[derive(Clone, Debug)]
pub struct GeneralBandedLuPartialPivot {
    storage: BandedLuStorage,
    dense_lu: Vec<f64>,
    pivots: Vec<usize>,
    is_factorized: bool,
    pivot_epsilon: f64,
}

impl GeneralBandedLuPartialPivot {
    pub fn new(n: usize, kl: usize, ku: usize) -> Result<Self, BandedError> {
        Ok(Self {
            storage: BandedLuStorage::new(n, kl, ku)?,
            dense_lu: vec![0.0; n * n],
            pivots: (0..n).collect(),
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

    #[inline]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    pub fn set_pivot_epsilon(&mut self, eps: f64) {
        self.pivot_epsilon = eps.max(0.0);
    }

    fn reset_pivots(&mut self) {
        for (i, p) in self.pivots.iter_mut().enumerate() {
            *p = i;
        }
    }

    fn project_dense_lu_back_to_storage(&mut self) -> Result<(), BandedError> {
        self.storage.fill_zero();
        let n = self.n();
        for i in 0..n {
            let j0 = i.saturating_sub(self.kl());
            let j1 = (i + self.ext_ku() + 1).min(n);
            for j in j0..j1 {
                self.storage
                    .set_row_value(i, j, self.dense_lu[idx(n, i, j)])?;
            }
        }
        Ok(())
    }

    /// Factor A into in-place banded LU with partial pivoting restricted
    /// to the lower band.
    ///
    /// Storage after factorization:
    /// - strict lower part stores L multipliers
    /// - diagonal and upper part store U
    /// - pivots[k] records the row swapped with k at step k
    pub fn factor_from(&mut self, a: &Banded<f64>) -> Result<(), BandedError> {
        if a.n() != self.n() || a.kl() != self.kl() || a.ku() != self.ku() {
            return Err(BandedError::DimensionMismatch);
        }

        self.storage.copy_from_compact(a)?;
        self.is_factorized = false;
        self.reset_pivots();
        self.dense_lu.fill(0.0);
        let n = self.n();
        for j in 0..n {
            let i0 = j.saturating_sub(self.ku());
            let i1 = (j + self.kl() + 1).min(n);
            for i in i0..i1 {
                self.dense_lu[idx(n, i, j)] = a[(i, j)];
            }
        }

        dense_lu_pivot_in_place(&mut self.dense_lu, n, &mut self.pivots)?;
        self.project_dense_lu_back_to_storage()?;

        self.is_factorized = true;
        Ok(())
    }

    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        if rhs.len() != self.n() {
            return Err(BandedError::DimensionMismatch);
        }

        dense_lu_pivot_solve_in_place(&self.dense_lu, self.n(), &self.pivots, rhs)
    }

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

    pub fn extract_l_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }

        let n = self.n();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            l[i][i] = 1.0;
            for j in 0..i {
                l[i][j] = self.dense_lu[idx(n, i, j)];
            }
        }

        Ok(l)
    }

    pub fn extract_u_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }

        let n = self.n();
        let mut u = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                u[i][j] = self.dense_lu[idx(n, i, j)];
            }
        }

        Ok(u)
    }

    pub fn reconstruct_lu_product_dense(&self) -> Result<Vec<Vec<f64>>, BandedError> {
        let l = self.extract_l_dense()?;
        let u = self.extract_u_dense()?;
        Ok(dense_matmul(&l, &u))
    }

    pub fn apply_pivots_to_dense_rows(&self, a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }

        let mut out = a.to_vec();
        for (k, &p) in self.pivots.iter().enumerate() {
            if p != k {
                out.swap(k, p);
            }
        }
        Ok(out)
    }

    pub fn debug_dump_dense(&self) -> String {
        self.storage.debug_dump_dense()
    }
}
//========================================================================================
#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::somelinalg::banded::error::BandedError;
    use crate::somelinalg::banded::{
        GeneralBandedLuPartialPivot, banded_matvec, banded_to_dense, dense_diff_linf,
        general_lu::GeneralBandedLuNoPivot, residual_linf, storage::Banded,
    };

    fn random_diag_dominant_banded(n: usize, kl: usize, ku: usize, seed: u64) -> Banded<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);

            let mut col_abs_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = rng.gen_range(-1.0..1.0);
                a[(i, j)] = v;
                col_abs_sum += v.abs();
            }

            a[(j, j)] = col_abs_sum + rng.gen_range(1.0..2.0);
        }

        a
    }

    #[test]
    fn no_pivot_fails_but_partial_pivot_succeeds() {
        // [0 2 0]
        // [1 3 4]
        // [0 5 6]
        let mut a = Banded::<f64>::zeros(3, 1, 1).unwrap();
        a[(0, 0)] = 0.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 3.0;
        a[(1, 2)] = 4.0;
        a[(2, 1)] = 5.0;
        a[(2, 2)] = 6.0;

        let mut no_pivot = GeneralBandedLuNoPivot::new(3, 1, 1).unwrap();
        assert!(matches!(
            no_pivot.factor_from(&a),
            Err(BandedError::ZeroPivot { .. })
        ));

        let mut pp = GeneralBandedLuPartialPivot::new(3, 1, 1).unwrap();
        pp.factor_from(&a).unwrap();
    }

    #[test]
    fn pa_matches_lu_for_known_case() {
        let mut a = Banded::<f64>::zeros(3, 1, 1).unwrap();
        a[(0, 0)] = 0.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 3.0;
        a[(1, 2)] = 4.0;
        a[(2, 1)] = 5.0;
        a[(2, 2)] = 6.0;

        let a_dense = banded_to_dense(&a);

        let mut pp = GeneralBandedLuPartialPivot::new(3, 1, 1).unwrap();
        pp.factor_from(&a).unwrap();

        let pa = pp.apply_pivots_to_dense_rows(&a_dense).unwrap();
        let lu = pp.reconstruct_lu_product_dense().unwrap();

        let err = dense_diff_linf(&pa, &lu);
        assert!(err < 1e-10, "P*A vs LU mismatch too large: {err:e}");
    }

    #[test]
    fn partial_pivot_solve_recovers_known_solution() {
        let mut a = Banded::<f64>::zeros(3, 1, 1).unwrap();
        a[(0, 0)] = 0.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 3.0;
        a[(1, 2)] = 4.0;
        a[(2, 1)] = 5.0;
        a[(2, 2)] = 6.0;

        let x_true = vec![1.0, -1.0, 2.0];
        let b = banded_matvec(&a, &x_true).unwrap();

        let mut x = b.clone();
        let mut pp = GeneralBandedLuPartialPivot::new(3, 1, 1).unwrap();
        pp.factor_from(&a).unwrap();
        pp.solve_in_place(&mut x).unwrap();

        let res = residual_linf(&a, &x, &b).unwrap();
        assert!(res < 1e-10, "residual too large: {res:e}");

        for i in 0..x.len() {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-10,
                "x mismatch at {i}: got {}, expected {}",
                x[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn partial_pivot_matches_no_pivot_on_diag_dominant_cases() {
        let cases = [
            (8, 1, 1, 101_u64),
            (12, 2, 2, 202_u64),
            (20, 3, 1, 303_u64),
            (24, 2, 4, 404_u64),
        ];

        for (n, kl, ku, seed) in cases {
            let a = random_diag_dominant_banded(n, kl, ku, seed);

            let mut rng = StdRng::seed_from_u64(seed + 9999);
            let x_true: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let b = banded_matvec(&a, &x_true).unwrap();

            let mut x_np = b.clone();
            let mut np = GeneralBandedLuNoPivot::new(n, kl, ku).unwrap();
            np.factor_from(&a).unwrap();
            np.solve_in_place(&mut x_np).unwrap();

            let mut x_pp = b.clone();
            let mut pp = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
            pp.factor_from(&a).unwrap();
            pp.solve_in_place(&mut x_pp).unwrap();

            let err_np = residual_linf(&a, &x_np, &b).unwrap();
            let err_pp = residual_linf(&a, &x_pp, &b).unwrap();

            assert!(err_np < 1e-8, "no-pivot residual too large: {err_np:e}");
            assert!(
                err_pp < 1e-8,
                "partial-pivot residual too large: {err_pp:e}"
            );

            for i in 0..n {
                assert!(
                    (x_pp[i] - x_true[i]).abs() < 1e-8,
                    "partial-pivot x mismatch at {i}: got {}, expected {}",
                    x_pp[i],
                    x_true[i]
                );
            }
        }
    }

    #[test]
    fn partial_pivot_does_not_swap_history_outside_lower_band() {
        // This reproduces the same structural failure mode we saw on
        // combustion-like banded Jacobians: at step k=1 a deeper pivot row
        // wants to swap with the current row, while the current row still has
        // a left-history entry that would sit outside the deeper row's lower
        // band if we swapped too much of the row fragment.
        let mut a = Banded::<f64>::zeros(6, 2, 1).unwrap();

        a[(0, 0)] = 4.0;
        a[(1, 0)] = 1.0;
        a[(0, 1)] = 0.5;
        a[(1, 1)] = 1.0e-6;
        a[(2, 1)] = 1.0;
        a[(3, 1)] = 5.0;
        a[(1, 2)] = 0.25;
        a[(2, 2)] = 4.0;
        a[(3, 2)] = 0.5;
        a[(2, 3)] = 0.5;
        a[(3, 3)] = 3.0;
        a[(4, 3)] = 0.5;
        a[(3, 4)] = 0.5;
        a[(4, 4)] = 3.0;
        a[(5, 4)] = 0.5;
        a[(4, 5)] = 0.5;
        a[(5, 5)] = 2.5;

        let x_true = vec![1.0, -2.0, 0.5, 1.5, -1.0, 0.25];
        let b = banded_matvec(&a, &x_true).unwrap();

        let mut x = b.clone();
        let mut lu = GeneralBandedLuPartialPivot::new(6, 2, 1).unwrap();
        lu.factor_from(&a)
            .expect("partial pivot factorization should not hit out-of-band swap");
        lu.solve_in_place(&mut x).unwrap();

        let res = residual_linf(&a, &x, &b).unwrap();
        assert!(res < 1e-8, "residual too large after guarded swap: {res:e}");
    }
}
