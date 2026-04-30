use super::{
    block_tridiagonal::BlockTridiagonal,
    dense_block_kernels::{
        block_copy, dense_lu_in_place, dense_lu_pivot_in_place,
        dense_lu_pivot_right_solve_block_in_place, dense_lu_pivot_solve_in_place,
        dense_lu_right_solve_block_in_place, dense_lu_solve_in_place, dense_matmul_sub_assign, idx,
    },
    error::BandedError,
};

#[derive(Clone, Debug)]
pub struct BlockTridiagonalLu {
    n_blocks: usize,
    block_size: usize,
    lower_mult: Vec<Vec<f64>>,    // len = n_blocks - 1
    diag_lu: Vec<Vec<f64>>,       // len = n_blocks
    diag_pivots: Vec<Vec<usize>>, // len = n_blocks, each len = block_size
    upper: Vec<Vec<f64>>,         // len = n_blocks - 1
    is_factorized: bool,
}
impl BlockTridiagonalLu {
    pub fn new(n_blocks: usize, block_size: usize) -> Result<Self, BandedError> {
        if n_blocks == 0 || block_size == 0 {
            return Err(BandedError::DimensionMismatch);
        }

        let block_len = block_size
            .checked_mul(block_size)
            .ok_or(BandedError::DimensionMismatch)?;

        let mut diag_lu = Vec::with_capacity(n_blocks);
        let mut diag_pivots = Vec::with_capacity(n_blocks);

        for _ in 0..n_blocks {
            diag_lu.push(vec![0.0; block_len]);
            diag_pivots.push(vec![0usize; block_size]);
        }

        let off_len = n_blocks.saturating_sub(1);

        let mut lower_mult = Vec::with_capacity(off_len);
        let mut upper = Vec::with_capacity(off_len);

        for _ in 0..off_len {
            lower_mult.push(vec![0.0; block_len]);
            upper.push(vec![0.0; block_len]);
        }

        Ok(Self {
            n_blocks,
            block_size,
            lower_mult,
            diag_lu,
            diag_pivots,
            upper,
            is_factorized: false,
        })
    }

    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n_blocks * self.block_size
    }

    #[inline]
    pub fn lower_mult_blocks(&self) -> &[Vec<f64>] {
        &self.lower_mult
    }

    #[inline]
    pub fn diag_lu_blocks(&self) -> &[Vec<f64>] {
        &self.diag_lu
    }

    #[inline]
    pub fn upper_blocks(&self) -> &[Vec<f64>] {
        &self.upper
    }

    #[inline]
    pub fn diag_pivots(&self) -> &[Vec<usize>] {
        &self.diag_pivots
    }

    pub fn factor_from(&mut self, a: &BlockTridiagonal) -> Result<(), BandedError> {
        if a.n_blocks() != self.n_blocks || a.block_size() != self.block_size {
            return Err(BandedError::DimensionMismatch);
        }

        let bs = self.block_size;
        let nb = self.n_blocks;

        self.is_factorized = false;

        // Copy blocks
        for k in 0..nb {
            block_copy(&mut self.diag_lu[k], a.diag_block(k).unwrap(), bs)?;
            self.diag_pivots[k].fill(0);
        }

        for k in 0..nb.saturating_sub(1) {
            block_copy(&mut self.upper[k], a.upper_block(k).unwrap(), bs)?;
            block_copy(&mut self.lower_mult[k], a.lower_block(k).unwrap(), bs)?;
        }

        // Factor first diagonal block with pivoting
        dense_lu_pivot_in_place(&mut self.diag_lu[0], bs, &mut self.diag_pivots[0])?;

        // Forward block elimination
        for k in 1..nb {
            // lower_mult[k-1] := A_k * inv(D_{k-1})
            dense_lu_pivot_right_solve_block_in_place(
                &self.diag_lu[k - 1],
                bs,
                &self.diag_pivots[k - 1],
                &mut self.lower_mult[k - 1],
            )?;

            // D_k := D_k - lower_mult[k-1] * upper[k-1]
            dense_matmul_sub_assign(
                &mut self.diag_lu[k],
                &self.lower_mult[k - 1],
                &self.upper[k - 1],
                bs,
            )?;

            // Factor updated diagonal block with pivoting
            dense_lu_pivot_in_place(&mut self.diag_lu[k], bs, &mut self.diag_pivots[k])?;
        }

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

        let bs = self.block_size;
        let nb = self.n_blocks;

        // Forward sweep:
        // rhs_k -= L_k * rhs_{k-1}
        for k in 1..nb {
            let prev_start = (k - 1) * bs;
            let prev_end = prev_start + bs;
            let cur_start = k * bs;
            let cur_end = cur_start + bs;

            let prev = rhs[prev_start..prev_end].to_vec();
            let cur = &mut rhs[cur_start..cur_end];
            let l = &self.lower_mult[k - 1];

            for i in 0..bs {
                let mut corr = 0.0;
                for j in 0..bs {
                    corr += l[idx(bs, i, j)] * prev[j];
                }
                cur[i] -= corr;
            }
        }

        // Solve last block
        {
            let start = (nb - 1) * bs;
            let end = start + bs;
            dense_lu_pivot_solve_in_place(
                &self.diag_lu[nb - 1],
                bs,
                &self.diag_pivots[nb - 1],
                &mut rhs[start..end],
            )?;
        }

        // Backward sweep:
        // rhs_k = D_k^{-1} (rhs_k - U_k * rhs_{k+1})
        for k in (0..(nb - 1)).rev() {
            let cur_start = k * bs;
            let cur_end = cur_start + bs;
            let next_start = (k + 1) * bs;
            let next_end = next_start + bs;

            let next = rhs[next_start..next_end].to_vec();
            let cur = &mut rhs[cur_start..cur_end];
            let u = &self.upper[k];

            for i in 0..bs {
                let mut corr = 0.0;
                for j in 0..bs {
                    corr += u[idx(bs, i, j)] * next[j];
                }
                cur[i] -= corr;
            }

            dense_lu_pivot_solve_in_place(&self.diag_lu[k], bs, &self.diag_pivots[k], cur)?;
        }

        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::BlockTridiagonalLu;
    use crate::somelinalg::banded::block_tridiagonal::BlockTridiagonal;

    fn dense_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i][j] * x[j];
            }
        }
        y
    }

    fn vec_diff_linf(x: &[f64], y: &[f64]) -> f64 {
        let mut m = 0.0;
        for i in 0..x.len() {
            let d = (x[i] - y[i]).abs();
            if d > m {
                m = d;
            }
        }
        m
    }

    #[test]
    fn block_tridiagonal_lu_small_manual_case() {
        // 2 blocks, block size 2
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        // B0
        a.set_diag(0, 0, 0, 4.0).unwrap();
        a.set_diag(0, 0, 1, 1.0).unwrap();
        a.set_diag(0, 1, 0, 2.0).unwrap();
        a.set_diag(0, 1, 1, 3.0).unwrap();

        // C0
        a.set_upper(0, 0, 0, 0.5).unwrap();
        a.set_upper(0, 0, 1, 0.0).unwrap();
        a.set_upper(0, 1, 0, 0.0).unwrap();
        a.set_upper(0, 1, 1, 0.5).unwrap();

        // A1
        a.set_lower(0, 0, 0, 0.25).unwrap();
        a.set_lower(0, 0, 1, 0.0).unwrap();
        a.set_lower(0, 1, 0, 0.0).unwrap();
        a.set_lower(0, 1, 1, 0.25).unwrap();

        // B1
        a.set_diag(1, 0, 0, 5.0).unwrap();
        a.set_diag(1, 0, 1, 1.0).unwrap();
        a.set_diag(1, 1, 0, 1.0).unwrap();
        a.set_diag(1, 1, 1, 4.0).unwrap();

        let dense = a.to_dense();

        let x_true = vec![1.0, 2.0, 3.0, 4.0];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = BlockTridiagonalLu::new(2, 2).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();

        assert!(vec_diff_linf(&rhs, &x_true) < 1e-10);
    }

    #[test]
    fn block_tridiagonal_lu_multiple_rhs() {
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        // Simple diagonally dominant blocks
        a.set_diag(0, 0, 0, 4.0).unwrap();
        a.set_diag(0, 0, 1, 1.0).unwrap();
        a.set_diag(0, 1, 0, 1.0).unwrap();
        a.set_diag(0, 1, 1, 3.0).unwrap();

        a.set_upper(0, 0, 0, 0.2).unwrap();
        a.set_upper(0, 1, 1, 0.2).unwrap();

        a.set_lower(0, 0, 0, 0.1).unwrap();
        a.set_lower(0, 1, 1, 0.1).unwrap();

        a.set_diag(1, 0, 0, 5.0).unwrap();
        a.set_diag(1, 0, 1, 0.5).unwrap();
        a.set_diag(1, 1, 0, 0.5).unwrap();
        a.set_diag(1, 1, 1, 4.0).unwrap();

        let dense = a.to_dense();

        let x1 = vec![1.0, 0.0, 2.0, 1.0];
        let x2 = vec![0.5, -1.0, 1.5, 2.0];

        let b1 = dense_matvec(&dense, &x1);
        let b2 = dense_matvec(&dense, &x2);

        let mut rhs = vec![b1[0], b1[1], b1[2], b1[3], b2[0], b2[1], b2[2], b2[3]];

        let mut lu = BlockTridiagonalLu::new(2, 2).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_multiple_in_place(&mut rhs, 2, 4).unwrap();

        let got1 = &rhs[0..4];
        let got2 = &rhs[4..8];

        assert!(vec_diff_linf(got1, &x1) < 1e-10);
        assert!(vec_diff_linf(got2, &x2) < 1e-10);
    }

    #[test]
    fn block_tridiagonal_lu_handles_zero_pivot_inside_diagonal_block() {
        // First diagonal block requires pivoting:
        // [0 2]
        // [1 3]
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        // B0
        a.set_diag(0, 0, 0, 0.0).unwrap();
        a.set_diag(0, 0, 1, 2.0).unwrap();
        a.set_diag(0, 1, 0, 1.0).unwrap();
        a.set_diag(0, 1, 1, 3.0).unwrap();

        // C0
        a.set_upper(0, 0, 0, 0.2).unwrap();
        a.set_upper(0, 1, 1, 0.2).unwrap();

        // A1
        a.set_lower(0, 0, 0, 0.1).unwrap();
        a.set_lower(0, 1, 1, 0.1).unwrap();

        // B1
        a.set_diag(1, 0, 0, 4.0).unwrap();
        a.set_diag(1, 0, 1, 0.5).unwrap();
        a.set_diag(1, 1, 0, 0.5).unwrap();
        a.set_diag(1, 1, 1, 5.0).unwrap();

        let dense = a.to_dense();

        let x_true = vec![1.0, -1.0, 2.0, 0.5];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = BlockTridiagonalLu::new(2, 2).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();

        assert!(vec_diff_linf(&rhs, &x_true) < 1e-10);
    }
}
