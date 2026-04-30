use super::{
    block_tridiagonal::BlockTridiagonal,
    dense_block_kernels::{block_copy, dense_lu_pivot_in_place, idx},
    error::BandedError,
};

#[derive(Clone, Debug)]
pub struct BlockTridiagonalLuConsistent {
    n_blocks: usize,
    block_size: usize,
    diag_lu: Vec<Vec<f64>>,
    diag_pivots: Vec<Vec<usize>>,
    lower_fac: Vec<Vec<f64>>,
    upper_fac: Vec<Vec<f64>>,
    is_factorized: bool,
}

#[derive(Clone, Debug, Default)]
pub struct IterativeRefinementReport {
    pub requested_steps: usize,
    pub accepted_steps: usize,
    pub refinement_attempted: bool,
    pub direct_relative_residual: f64,
    pub final_relative_residual: f64,
}

impl BlockTridiagonalLuConsistent {
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
        let mut lower_fac = Vec::with_capacity(off_len);
        let mut upper_fac = Vec::with_capacity(off_len);
        for _ in 0..off_len {
            lower_fac.push(vec![0.0; block_len]);
            upper_fac.push(vec![0.0; block_len]);
        }

        Ok(Self {
            n_blocks,
            block_size,
            diag_lu,
            diag_pivots,
            lower_fac,
            upper_fac,
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
    pub fn diag_lu_blocks(&self) -> &[Vec<f64>] {
        &self.diag_lu
    }

    #[inline]
    pub fn diag_pivots(&self) -> &[Vec<usize>] {
        &self.diag_pivots
    }

    #[inline]
    pub fn lower_fac_blocks(&self) -> &[Vec<f64>] {
        &self.lower_fac
    }

    #[inline]
    pub fn upper_fac_blocks(&self) -> &[Vec<f64>] {
        &self.upper_fac
    }

    pub fn factor_from(&mut self, a: &BlockTridiagonal) -> Result<(), BandedError> {
        if a.n_blocks() != self.n_blocks || a.block_size() != self.block_size {
            return Err(BandedError::DimensionMismatch);
        }

        let bs = self.block_size;
        let nb = self.n_blocks;

        self.is_factorized = false;

        for block_idx in 0..nb {
            block_copy(
                &mut self.diag_lu[block_idx],
                a.diag_block(block_idx).unwrap(),
                bs,
            )?;
            self.diag_pivots[block_idx].fill(0);
        }
        for block_idx in 0..nb.saturating_sub(1) {
            block_copy(
                &mut self.lower_fac[block_idx],
                a.lower_block(block_idx).unwrap(),
                bs,
            )?;
            block_copy(
                &mut self.upper_fac[block_idx],
                a.upper_block(block_idx).unwrap(),
                bs,
            )?;
        }

        dense_lu_pivot_in_place(&mut self.diag_lu[0], bs, &mut self.diag_pivots[0])?;

        for block_idx in 0..nb.saturating_sub(1) {
            apply_pivots_to_block_rows(
                &mut self.upper_fac[block_idx],
                bs,
                &self.diag_pivots[block_idx],
            )?;
            left_solve_unit_lower_block_in_place(
                &self.diag_lu[block_idx],
                bs,
                &mut self.upper_fac[block_idx],
            )?;
            right_solve_upper_block_in_place(
                &self.diag_lu[block_idx],
                bs,
                &mut self.lower_fac[block_idx],
            )?;

            dense_matmul_sub_assign_square(
                &mut self.diag_lu[block_idx + 1],
                &self.lower_fac[block_idx],
                &self.upper_fac[block_idx],
                bs,
            )?;

            dense_lu_pivot_in_place(
                &mut self.diag_lu[block_idx + 1],
                bs,
                &mut self.diag_pivots[block_idx + 1],
            )?;
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

        for block_idx in 0..nb {
            let start = block_idx * bs;
            let end = start + bs;
            let prev = if block_idx > 0 {
                let prev_start = (block_idx - 1) * bs;
                let prev_end = prev_start + bs;
                Some(rhs[prev_start..prev_end].to_vec())
            } else {
                None
            };
            let cur = &mut rhs[start..end];

            apply_pivots_to_vector(cur, &self.diag_pivots[block_idx])?;

            if let Some(prev) = prev.as_ref() {
                subtract_matvec_in_place(cur, &self.lower_fac[block_idx - 1], bs, prev);
            }

            left_solve_unit_lower_vector_in_place(&self.diag_lu[block_idx], bs, cur)?;
        }

        for block_idx in (0..nb).rev() {
            let start = block_idx * bs;
            let end = start + bs;
            let next = if block_idx + 1 < nb {
                let next_start = (block_idx + 1) * bs;
                let next_end = next_start + bs;
                Some(rhs[next_start..next_end].to_vec())
            } else {
                None
            };
            let cur = &mut rhs[start..end];

            if let Some(next) = next.as_ref() {
                subtract_matvec_in_place(cur, &self.upper_fac[block_idx], bs, next);
            }

            upper_solve_vector_in_place(&self.diag_lu[block_idx], bs, cur)?;
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

    pub fn solve_in_place_with_iterative_refinement(
        &self,
        a: &BlockTridiagonal,
        rhs: &mut [f64],
        refinement_steps: usize,
    ) -> Result<(), BandedError> {
        self.solve_in_place_with_iterative_refinement_report(a, rhs, refinement_steps)
            .map(|_| ())
    }

    pub fn solve_in_place_with_iterative_refinement_report(
        &self,
        a: &BlockTridiagonal,
        rhs: &mut [f64],
        refinement_steps: usize,
    ) -> Result<IterativeRefinementReport, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        if a.n_blocks() != self.n_blocks || a.block_size() != self.block_size {
            return Err(BandedError::DimensionMismatch);
        }
        if rhs.len() != self.n() {
            return Err(BandedError::DimensionMismatch);
        }

        let original_rhs = rhs.to_vec();
        self.solve_in_place(rhs)?;

        if refinement_steps == 0 {
            let rr = relative_block_solve_residual(a, rhs, &original_rhs)?;
            return Ok(IterativeRefinementReport {
                requested_steps: 0,
                accepted_steps: 0,
                refinement_attempted: false,
                direct_relative_residual: rr,
                final_relative_residual: rr,
            });
        }

        let direct_rr = relative_block_solve_residual(a, rhs, &original_rhs)?;
        let mut current_rr = direct_rr;
        if !current_rr.is_finite() {
            return Ok(IterativeRefinementReport {
                requested_steps: refinement_steps,
                accepted_steps: 0,
                refinement_attempted: false,
                direct_relative_residual: direct_rr,
                final_relative_residual: current_rr,
            });
        }
        if !self.should_attempt_iterative_refinement(a, current_rr)? {
            return Ok(IterativeRefinementReport {
                requested_steps: refinement_steps,
                accepted_steps: 0,
                refinement_attempted: false,
                direct_relative_residual: direct_rr,
                final_relative_residual: current_rr,
            });
        }

        let mut residual = vec![0.0; rhs.len()];
        let mut trial = vec![0.0; rhs.len()];
        let mut accepted_steps = 0usize;

        for _ in 0..refinement_steps {
            block_tridiagonal_residual_in_place(a, rhs, &original_rhs, &mut residual)?;

            let mut correction = residual.clone();
            self.solve_in_place(&mut correction)?;

            for i in 0..rhs.len() {
                trial[i] = rhs[i] + correction[i];
            }

            let trial_rr = relative_block_solve_residual(a, &trial, &original_rhs)?;

            if !trial_rr.is_finite() || trial_rr >= current_rr {
                break;
            }

            rhs.copy_from_slice(&trial);
            current_rr = trial_rr;
            accepted_steps += 1;
        }

        Ok(IterativeRefinementReport {
            requested_steps: refinement_steps,
            accepted_steps,
            refinement_attempted: true,
            direct_relative_residual: direct_rr,
            final_relative_residual: current_rr,
        })
    }

    fn should_attempt_iterative_refinement(
        &self,
        a: &BlockTridiagonal,
        current_rr: f64,
    ) -> Result<bool, BandedError> {
        const REFINEMENT_ENTRY_RR_MAX: f64 = 1.0e-12;
        const FACTOR_RELATIVE_RESIDUAL_MAX: f64 = 1.0e-4;
        const MIN_ACCEPTABLE_ABS_DIAG_U: f64 = 1.0e-10;
        const MAX_ACCEPTABLE_MULTIPLIER_NORM: f64 = 1.0e6;

        // If the direct solve already lands near machine precision, refinement
        // is more likely to inject noise than to help.
        if current_rr <= REFINEMENT_ENTRY_RR_MAX {
            return Ok(false);
        }

        // Refinement only makes sense when the stored factorization is a
        // reasonably faithful surrogate for the original matrix.
        let factor_rr = self.factor_residual_relative(a)?;
        if !factor_rr.is_finite() || factor_rr > FACTOR_RELATIVE_RESIDUAL_MAX {
            return Ok(false);
        }

        // Very small diagonal pivots or explosive block multipliers are strong
        // signals that the correction solve is not a trustworthy direction.
        let diagnostics = self.block_norm_diagnostics();
        let min_abs_diag_u = diagnostics
            .iter()
            .map(|d| d.min_abs_diag_u)
            .fold(f64::INFINITY, f64::min);
        let worst_multiplier = diagnostics
            .iter()
            .flat_map(|d| [d.lower_fac_linf, d.upper_fac_linf])
            .flatten()
            .fold(0.0_f64, f64::max);

        Ok(min_abs_diag_u >= MIN_ACCEPTABLE_ABS_DIAG_U
            && worst_multiplier <= MAX_ACCEPTABLE_MULTIPLIER_NORM)
    }

    pub fn apply_block_permutations_to_dense_rows(
        &self,
        dense: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, BandedError> {
        if dense.len() != self.n() || dense.iter().any(|row| row.len() != self.n()) {
            return Err(BandedError::DimensionMismatch);
        }

        let bs = self.block_size;
        let mut out = dense.to_vec();
        for block_idx in 0..self.n_blocks {
            let row0 = block_idx * bs;
            let perm = permutation_from_swap_pivots(&self.diag_pivots[block_idx]);
            let mut permuted = vec![vec![0.0; self.n()]; bs];
            for local_row in 0..bs {
                permuted[local_row] = dense[row0 + perm[local_row]].clone();
            }
            for local_row in 0..bs {
                out[row0 + local_row] = permuted[local_row].clone();
            }
        }
        Ok(out)
    }

    pub fn reconstruct_lu_dense(&self) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }

        let bs = self.block_size;
        let n = self.n();
        let mut l = vec![vec![0.0; n]; n];
        let mut u = vec![vec![0.0; n]; n];

        for block_idx in 0..self.n_blocks {
            let row0 = block_idx * bs;
            let col0 = block_idx * bs;
            let (l_diag, u_diag) = split_dense_lu_block(&self.diag_lu[block_idx], bs)?;
            for i in 0..bs {
                for j in 0..bs {
                    l[row0 + i][col0 + j] = l_diag[idx(bs, i, j)];
                    u[row0 + i][col0 + j] = u_diag[idx(bs, i, j)];
                }
            }
        }

        for block_idx in 0..self.n_blocks.saturating_sub(1) {
            let lower_row0 = (block_idx + 1) * bs;
            let lower_col0 = block_idx * bs;
            let upper_row0 = block_idx * bs;
            let upper_col0 = (block_idx + 1) * bs;

            for i in 0..bs {
                for j in 0..bs {
                    l[lower_row0 + i][lower_col0 + j] = self.lower_fac[block_idx][idx(bs, i, j)];
                    u[upper_row0 + i][upper_col0 + j] = self.upper_fac[block_idx][idx(bs, i, j)];
                }
            }
        }

        Ok((l, u))
    }

    pub fn factor_residual_inf(&self, a: &BlockTridiagonal) -> Result<f64, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        let dense_a = a.to_dense();
        let dense_pa = self.apply_block_permutations_to_dense_rows(dense_a.as_slice())?;
        let (dense_l, dense_u) = self.reconstruct_lu_dense()?;
        let dense_lu = dense_matmul_square(dense_l.as_slice(), dense_u.as_slice())?;
        Ok(dense_linf_diff(dense_pa.as_slice(), dense_lu.as_slice()))
    }

    pub fn factor_residual_relative(&self, a: &BlockTridiagonal) -> Result<f64, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        let pa = self.permuted_dense_matrix(a)?;
        let (dense_l, dense_u) = self.reconstruct_lu_dense()?;
        let lu = dense_matmul_square(dense_l.as_slice(), dense_u.as_slice())?;

        let num = dense_linf_diff(pa.as_slice(), lu.as_slice());
        let den = dense_linf_norm(pa.as_slice())
            .max(dense_linf_norm(lu.as_slice()))
            .max(1.0);

        Ok(num / den)
    }

    pub fn pivot_diagnostics(&self) -> Vec<PivotDiagnostics> {
        let mut out = Vec::with_capacity(self.n_blocks);

        for (block_index, piv) in self.diag_pivots.iter().enumerate() {
            let mut num_nontrivial_swaps = 0usize;
            let mut max_swap_distance = 0usize;

            for (k, &p) in piv.iter().enumerate() {
                if p != k {
                    num_nontrivial_swaps += 1;
                    max_swap_distance = max_swap_distance.max(p.abs_diff(k));
                }
            }

            out.push(PivotDiagnostics {
                block_index,
                num_nontrivial_swaps,
                max_swap_distance,
            });
        }

        out
    }

    pub fn block_norm_diagnostics(&self) -> Vec<BlockNormDiagnostics> {
        let bs = self.block_size;
        let mut out = Vec::with_capacity(self.n_blocks);

        for k in 0..self.n_blocks {
            let (mn, mx) = dense_u_diag_abs_minmax(&self.diag_lu[k], bs);

            out.push(BlockNormDiagnostics {
                block_index: k,
                diag_lu_linf: dense_block_linf(&self.diag_lu[k]),
                lower_fac_linf: if k > 0 {
                    Some(dense_block_linf(&self.lower_fac[k - 1]))
                } else {
                    None
                },
                upper_fac_linf: if k + 1 < self.n_blocks {
                    Some(dense_block_linf(&self.upper_fac[k]))
                } else {
                    None
                },
                min_abs_diag_u: mn,
                max_abs_diag_u: mx,
            });
        }

        out
    }

    fn permuted_dense_matrix(&self, a: &BlockTridiagonal) -> Result<Vec<Vec<f64>>, BandedError> {
        if !self.is_factorized {
            return Err(BandedError::NotFactorized);
        }
        let dense_a = a.to_dense();
        self.apply_block_permutations_to_dense_rows(dense_a.as_slice())
    }
}
//===============================================================================
fn relative_block_solve_residual(
    a: &BlockTridiagonal,
    x: &[f64],
    b: &[f64],
) -> Result<f64, BandedError> {
    let mut ax = vec![0.0; x.len()];
    block_tridiagonal_matvec_in_place(a, x, &mut ax)?;

    let mut rmax = 0.0_f64;
    let mut bmax = 0.0_f64;

    for i in 0..b.len() {
        rmax = rmax.max((ax[i] - b[i]).abs());
        bmax = bmax.max(b[i].abs());
    }

    Ok(rmax / bmax.max(1.0))
}
#[derive(Clone, Debug)]
pub struct PivotDiagnostics {
    pub block_index: usize,
    pub num_nontrivial_swaps: usize,
    pub max_swap_distance: usize,
}

#[derive(Clone, Debug)]
pub struct BlockNormDiagnostics {
    pub block_index: usize,
    pub diag_lu_linf: f64,
    pub lower_fac_linf: Option<f64>,
    pub upper_fac_linf: Option<f64>,
    pub min_abs_diag_u: f64,
    pub max_abs_diag_u: f64,
}

fn dense_block_linf(block: &[f64]) -> f64 {
    block.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

fn dense_u_diag_abs_minmax(block: &[f64], bs: usize) -> (f64, f64) {
    let mut mn = f64::INFINITY;
    let mut mx = 0.0_f64;

    for i in 0..bs {
        let v = block[idx(bs, i, i)].abs();
        mn = mn.min(v);
        mx = mx.max(v);
    }

    (mn, mx)
}

fn apply_pivots_to_vector(rhs: &mut [f64], pivots: &[usize]) -> Result<(), BandedError> {
    if rhs.len() != pivots.len() {
        return Err(BandedError::DimensionMismatch);
    }
    for (k, &p) in pivots.iter().enumerate() {
        if p != k {
            rhs.swap(k, p);
        }
    }
    Ok(())
}

fn apply_pivots_to_block_rows(
    block: &mut [f64],
    bs: usize,
    pivots: &[usize],
) -> Result<(), BandedError> {
    if block.len() != bs * bs || pivots.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }
    for (k, &p) in pivots.iter().enumerate() {
        if p != k {
            for col in 0..bs {
                block.swap(idx(bs, k, col), idx(bs, p, col));
            }
        }
    }
    Ok(())
}

fn left_solve_unit_lower_vector_in_place(
    lu: &[f64],
    bs: usize,
    rhs: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || rhs.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }
    for i in 0..bs {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lu[idx(bs, i, j)] * rhs[j];
        }
        rhs[i] = sum;
    }
    Ok(())
}

fn upper_solve_vector_in_place(lu: &[f64], bs: usize, rhs: &mut [f64]) -> Result<(), BandedError> {
    if lu.len() != bs * bs || rhs.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }

    let eps = 1e-14;
    for i in (0..bs).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..bs {
            sum -= lu[idx(bs, i, j)] * rhs[j];
        }
        let uii = lu[idx(bs, i, i)];
        if uii.abs() <= eps {
            return Err(BandedError::ZeroPivot {
                index: i,
                value: uii,
            });
        }
        rhs[i] = sum / uii;
    }
    Ok(())
}

fn left_solve_unit_lower_block_in_place(
    lu: &[f64],
    bs: usize,
    block: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || block.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    for row in 0..bs {
        for pivot_col in 0..row {
            let lij = lu[idx(bs, row, pivot_col)];
            if lij == 0.0 {
                continue;
            }
            for col in 0..bs {
                block[idx(bs, row, col)] -= lij * block[idx(bs, pivot_col, col)];
            }
        }
    }
    Ok(())
}

fn upper_solve_transpose_vector_in_place(
    lu: &[f64],
    bs: usize,
    rhs: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || rhs.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }

    let eps = 1e-14;
    for i in 0..bs {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lu[idx(bs, j, i)] * rhs[j];
        }
        let uii = lu[idx(bs, i, i)];
        if uii.abs() <= eps {
            return Err(BandedError::ZeroPivot {
                index: i,
                value: uii,
            });
        }
        rhs[i] = sum / uii;
    }
    Ok(())
}

fn right_solve_upper_block_in_place(
    lu: &[f64],
    bs: usize,
    block: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || block.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    let mut tmp_t = vec![0.0; bs * bs];
    for i in 0..bs {
        for j in 0..bs {
            tmp_t[idx(bs, j, i)] = block[idx(bs, i, j)];
        }
    }

    let mut col = vec![0.0; bs];
    for j in 0..bs {
        for i in 0..bs {
            col[i] = tmp_t[idx(bs, i, j)];
        }
        upper_solve_transpose_vector_in_place(lu, bs, &mut col)?;
        for i in 0..bs {
            tmp_t[idx(bs, i, j)] = col[i];
        }
    }

    for i in 0..bs {
        for j in 0..bs {
            block[idx(bs, i, j)] = tmp_t[idx(bs, j, i)];
        }
    }
    Ok(())
}

fn subtract_matvec_in_place(rhs: &mut [f64], mat: &[f64], bs: usize, x: &[f64]) {
    for i in 0..bs {
        let mut corr = 0.0;
        for j in 0..bs {
            corr += mat[idx(bs, i, j)] * x[j];
        }
        rhs[i] -= corr;
    }
}

fn block_tridiagonal_matvec_in_place(
    a: &BlockTridiagonal,
    x: &[f64],
    y_out: &mut [f64],
) -> Result<(), BandedError> {
    if x.len() != a.n() || y_out.len() != a.n() {
        return Err(BandedError::DimensionMismatch);
    }

    y_out.fill(0.0);
    let bs = a.block_size();
    let nb = a.n_blocks();

    for blk in 0..nb {
        let row0 = blk * bs;
        let x_diag = &x[row0..row0 + bs];
        let y_diag = &mut y_out[row0..row0 + bs];

        let diag = a.diag_block(blk).ok_or(BandedError::DimensionMismatch)?;
        add_block_matvec_in_place(y_diag, diag, bs, x_diag);

        if blk > 0 {
            let x_prev0 = (blk - 1) * bs;
            let x_prev = &x[x_prev0..x_prev0 + bs];
            let lower = a
                .lower_block(blk - 1)
                .ok_or(BandedError::DimensionMismatch)?;
            add_block_matvec_in_place(y_diag, lower, bs, x_prev);
        }

        if blk + 1 < nb {
            let x_next0 = (blk + 1) * bs;
            let x_next = &x[x_next0..x_next0 + bs];
            let upper = a.upper_block(blk).ok_or(BandedError::DimensionMismatch)?;
            add_block_matvec_in_place(y_diag, upper, bs, x_next);
        }
    }

    Ok(())
}

fn block_tridiagonal_residual_in_place(
    a: &BlockTridiagonal,
    x: &[f64],
    rhs: &[f64],
    residual_out: &mut [f64],
) -> Result<(), BandedError> {
    if x.len() != a.n() || rhs.len() != a.n() || residual_out.len() != a.n() {
        return Err(BandedError::DimensionMismatch);
    }

    block_tridiagonal_matvec_in_place(a, x, residual_out)?;
    for i in 0..residual_out.len() {
        residual_out[i] = rhs[i] - residual_out[i];
    }
    Ok(())
}

fn add_block_matvec_in_place(y: &mut [f64], block: &[f64], bs: usize, x: &[f64]) {
    for i in 0..bs {
        let mut acc = 0.0;
        for j in 0..bs {
            acc += block[idx(bs, i, j)] * x[j];
        }
        y[i] += acc;
    }
}

fn dense_matmul_sub_assign_square(
    dst: &mut [f64],
    lhs: &[f64],
    rhs: &[f64],
    bs: usize,
) -> Result<(), BandedError> {
    if dst.len() != bs * bs || lhs.len() != bs * bs || rhs.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }
    for i in 0..bs {
        for k in 0..bs {
            let a = lhs[idx(bs, i, k)];
            if a == 0.0 {
                continue;
            }
            for j in 0..bs {
                dst[idx(bs, i, j)] -= a * rhs[idx(bs, k, j)];
            }
        }
    }
    Ok(())
}

fn permutation_from_swap_pivots(pivots: &[usize]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..pivots.len()).collect();
    for (k, &p) in pivots.iter().enumerate() {
        perm.swap(k, p);
    }
    perm
}

fn split_dense_lu_block(lu: &[f64], bs: usize) -> Result<(Vec<f64>, Vec<f64>), BandedError> {
    if lu.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }
    let mut l = vec![0.0; bs * bs];
    let mut u = vec![0.0; bs * bs];
    for i in 0..bs {
        for j in 0..bs {
            let id = idx(bs, i, j);
            if i > j {
                l[id] = lu[id];
            } else if i == j {
                l[id] = 1.0;
                u[id] = lu[id];
            } else {
                u[id] = lu[id];
            }
        }
    }
    Ok((l, u))
}

fn dense_matmul_square(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, BandedError> {
    if lhs.is_empty() || lhs.len() != rhs.len() {
        return Err(BandedError::DimensionMismatch);
    }
    let n = lhs.len();
    if lhs.iter().any(|row| row.len() != n) || rhs.iter().any(|row| row.len() != n) {
        return Err(BandedError::DimensionMismatch);
    }

    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            let a = lhs[i][k];
            if a == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i][j] += a * rhs[k][j];
            }
        }
    }
    Ok(out)
}
fn dense_linf_norm(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|row| row.iter())
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max)
}
fn dense_linf_diff(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .flat_map(|(lhs_row, rhs_row)| lhs_row.iter().zip(rhs_row.iter()))
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max)
}
//===============================================================================
#[cfg(test)]
mod tests {
    use super::BlockTridiagonalLuConsistent;
    use crate::somelinalg::banded::block_tridiagonal::BlockTridiagonal;
    use crate::somelinalg::banded::dense_block_kernels::idx;
    use crate::somelinalg::banded::error::BandedError;
    use nalgebra::{DMatrix, DVector};

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

    fn vec_linf_norm(x: &[f64]) -> f64 {
        x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    fn dense_reference_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut flat = Vec::with_capacity(n * n);
        for row in a {
            flat.extend_from_slice(row);
        }
        let matrix = DMatrix::from_row_slice(n, n, &flat);
        let rhs = DVector::from_column_slice(b);
        let lu = matrix.lu();
        let solution = lu
            .solve(&rhs)
            .expect("dense reference LU solve should succeed on combustion-like diagnostic");
        solution.iter().copied().collect()
    }

    fn relative_solve_residual(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
        let ax = dense_matvec(a, x);
        let mut residual = vec![0.0; b.len()];
        for i in 0..b.len() {
            residual[i] = ax[i] - b[i];
        }
        let a_norm = a
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let denom = (a_norm * vec_linf_norm(x)).max(vec_linf_norm(b)).max(1.0);
        vec_linf_norm(&residual) / denom
    }

    fn dense_linf_matrix_norm(a: &[Vec<f64>]) -> f64 {
        a.iter()
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }

    fn estimate_dense_cond_inf(a: &[Vec<f64>]) -> f64 {
        let n = a.len();
        let mut flat = Vec::with_capacity(n * n);
        for row in a {
            flat.extend_from_slice(row);
        }
        let matrix = DMatrix::from_row_slice(n, n, &flat);
        let lu = matrix.lu();

        let mut inv = vec![vec![0.0; n]; n];
        for col in 0..n {
            let mut e = vec![0.0; n];
            e[col] = 1.0;
            let rhs = DVector::from_column_slice(&e);
            let sol = lu
                .solve(&rhs)
                .expect("dense inverse column solve should succeed in conditioning diagnostic");
            for row in 0..n {
                inv[row][col] = sol[row];
            }
        }

        dense_linf_matrix_norm(a) * dense_linf_matrix_norm(&inv)
    }

    fn small_block_system() -> BlockTridiagonal {
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        a.set_diag(0, 0, 0, 4.0).unwrap();
        a.set_diag(0, 0, 1, 1.0).unwrap();
        a.set_diag(0, 1, 0, 2.0).unwrap();
        a.set_diag(0, 1, 1, 3.0).unwrap();

        a.set_upper(0, 0, 0, 0.5).unwrap();
        a.set_upper(0, 1, 1, 0.5).unwrap();

        a.set_lower(0, 0, 0, 0.25).unwrap();
        a.set_lower(0, 1, 1, 0.25).unwrap();

        a.set_diag(1, 0, 0, 5.0).unwrap();
        a.set_diag(1, 0, 1, 1.0).unwrap();
        a.set_diag(1, 1, 0, 1.0).unwrap();
        a.set_diag(1, 1, 1, 4.0).unwrap();

        a
    }

    fn set_diag_block_dense(
        a: &mut BlockTridiagonal,
        block_idx: usize,
        dense: &[f64],
        block_size: usize,
    ) {
        for i in 0..block_size {
            for j in 0..block_size {
                let value = dense[idx(block_size, i, j)];
                if value != 0.0 {
                    a.set_diag(block_idx, i, j, value).unwrap();
                }
            }
        }
    }

    fn set_upper_block_dense(
        a: &mut BlockTridiagonal,
        block_idx: usize,
        dense: &[f64],
        block_size: usize,
    ) {
        for i in 0..block_size {
            for j in 0..block_size {
                let value = dense[idx(block_size, i, j)];
                if value != 0.0 {
                    a.set_upper(block_idx, i, j, value).unwrap();
                }
            }
        }
    }

    fn set_lower_block_dense(
        a: &mut BlockTridiagonal,
        block_idx: usize,
        dense: &[f64],
        block_size: usize,
    ) {
        for i in 0..block_size {
            for j in 0..block_size {
                let value = dense[idx(block_size, i, j)];
                if value != 0.0 {
                    a.set_lower(block_idx, i, j, value).unwrap();
                }
            }
        }
    }

    fn combustion_like_node_diag_block() -> Vec<f64> {
        let bs = 6;
        let mut block = vec![0.0; bs * bs];

        block[idx(bs, 0, 0)] = -7.1429;
        block[idx(bs, 0, 3)] = 1.0;

        block[idx(bs, 1, 0)] = -1.0045;
        block[idx(bs, 1, 4)] = 0.9954916;

        block[idx(bs, 2, 1)] = -1736.111111111111;
        block[idx(bs, 2, 5)] = 1.0;

        block[idx(bs, 3, 1)] = -1.0008;
        block[idx(bs, 4, 2)] = -1736.111111111111;
        block[idx(bs, 5, 2)] = -1.0008;

        block
    }

    fn combustion_like_forward_coupling_block() -> Vec<f64> {
        let bs = 6;
        let mut block = vec![0.0; bs * bs];
        block[idx(bs, 3, 0)] = 0.99925;
        block[idx(bs, 4, 1)] = 1.0;
        block[idx(bs, 5, 2)] = 0.99925;
        block
    }

    fn combustion_like_backward_coupling_block() -> Vec<f64> {
        let bs = 6;
        let mut block = vec![0.0; bs * bs];
        block[idx(bs, 0, 3)] = -0.99925;
        block[idx(bs, 1, 4)] = -1.0;
        block[idx(bs, 2, 5)] = -0.99925;
        block
    }

    fn build_superblock_from_node_couplings(
        nodes_per_superblock: usize,
        diagonal_shift: f64,
    ) -> Vec<f64> {
        let node_bs = 6;
        let super_bs = node_bs * nodes_per_superblock;
        let diag6 = combustion_like_node_diag_block();
        let up6 = combustion_like_forward_coupling_block();
        let low6 = combustion_like_backward_coupling_block();
        let mut block = vec![0.0; super_bs * super_bs];

        for node in 0..nodes_per_superblock {
            let base = node * node_bs;
            for i in 0..node_bs {
                for j in 0..node_bs {
                    block[idx(super_bs, base + i, base + j)] = diag6[idx(node_bs, i, j)];
                }
            }
        }

        for node in 0..nodes_per_superblock.saturating_sub(1) {
            let left = node * node_bs;
            let right = (node + 1) * node_bs;
            for i in 0..node_bs {
                for j in 0..node_bs {
                    block[idx(super_bs, left + i, right + j)] = up6[idx(node_bs, i, j)];
                    block[idx(super_bs, right + i, left + j)] = low6[idx(node_bs, i, j)];
                }
            }
        }

        // Add a tiny per-node diagonal perturbation so repeated raw superblocks
        // are not perfectly identical copies.
        for node in 0..nodes_per_superblock {
            let base = node * node_bs;
            for k in 0..node_bs {
                block[idx(super_bs, base + k, base + k)] +=
                    1.0e-3 * (node as f64 + 1.0) * (k as f64 + 1.0);
            }
        }
        for k in 0..super_bs {
            block[idx(super_bs, k, k)] += diagonal_shift;
        }

        block
    }

    fn weak_superblock_coupling(block_size: usize, scale: f64) -> Vec<f64> {
        let mut block = vec![0.0; block_size * block_size];
        for k in 0..block_size {
            block[idx(block_size, k, k)] = scale;
        }
        block
    }

    fn combustion_like_superblock_chain(
        n_blocks: usize,
        nodes_per_superblock: usize,
        diagonal_shift: f64,
    ) -> BlockTridiagonal {
        let block_size = 6 * nodes_per_superblock;
        let diag = build_superblock_from_node_couplings(nodes_per_superblock, diagonal_shift);
        let upper = weak_superblock_coupling(block_size, 2.5e-2);
        let lower = weak_superblock_coupling(block_size, -2.0e-2);
        let mut a = BlockTridiagonal::zeros(n_blocks, block_size).unwrap();

        for block_idx in 0..n_blocks {
            set_diag_block_dense(&mut a, block_idx, diag.as_slice(), block_size);
        }
        for block_idx in 0..n_blocks.saturating_sub(1) {
            set_upper_block_dense(&mut a, block_idx, upper.as_slice(), block_size);
            set_lower_block_dense(&mut a, block_idx, lower.as_slice(), block_size);
        }

        a
    }

    #[test]
    fn consistent_factorization_matches_pa_equals_lu_on_small_system() {
        let a = small_block_system();
        let mut lu = BlockTridiagonalLuConsistent::new(2, 2).unwrap();
        lu.factor_from(&a).unwrap();

        let residual = lu.factor_residual_inf(&a).unwrap();
        assert!(residual < 1e-10, "factor residual too large: {residual:e}");
    }

    #[test]
    fn consistent_factorization_solves_small_system() {
        let a = small_block_system();
        let dense = a.to_dense();
        let x_true = vec![1.0, 2.0, 3.0, 4.0];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = BlockTridiagonalLuConsistent::new(2, 2).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();

        assert!(vec_diff_linf(&rhs, &x_true) < 1e-10);
    }

    #[test]
    fn iterative_refinement_skips_when_direct_small_system_solve_is_already_good() {
        let a = small_block_system();
        let dense = a.to_dense();
        let x_true = vec![1.0, 2.0, 3.0, 4.0];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = BlockTridiagonalLuConsistent::new(2, 2).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place_with_iterative_refinement(&a, &mut rhs, 2)
            .expect("guarded refinement should succeed on small system");

        assert!(
            vec_diff_linf(&rhs, &x_true) < 1e-10,
            "guarded refinement should not spoil an already accurate small solve"
        );
    }

    #[test]
    fn consistent_factorization_handles_pivoted_diagonal_block() {
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        a.set_diag(0, 0, 0, 0.0).unwrap();
        a.set_diag(0, 0, 1, 2.0).unwrap();
        a.set_diag(0, 1, 0, 1.0).unwrap();
        a.set_diag(0, 1, 1, 3.0).unwrap();

        a.set_upper(0, 0, 0, 0.2).unwrap();
        a.set_upper(0, 1, 1, 0.2).unwrap();

        a.set_lower(0, 0, 0, 0.1).unwrap();
        a.set_lower(0, 1, 1, 0.1).unwrap();

        a.set_diag(1, 0, 0, 4.0).unwrap();
        a.set_diag(1, 0, 1, 0.5).unwrap();
        a.set_diag(1, 1, 0, 0.5).unwrap();
        a.set_diag(1, 1, 1, 5.0).unwrap();

        let dense = a.to_dense();
        let x_true = vec![1.0, -1.0, 2.0, 0.5];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = BlockTridiagonalLuConsistent::new(2, 2).unwrap();
        lu.factor_from(&a).unwrap();

        let residual = lu.factor_residual_inf(&a).unwrap();
        assert!(residual < 1e-10, "factor residual too large: {residual:e}");

        lu.solve_in_place(&mut rhs).unwrap();
        assert!(vec_diff_linf(&rhs, &x_true) < 1e-10);
    }

    #[test]
    fn consistent_factorization_rejects_combustion_like_singular_node_block() {
        let block_size = 6;
        let mut a = BlockTridiagonal::zeros(2, block_size).unwrap();
        let diag = combustion_like_node_diag_block();
        let upper = combustion_like_forward_coupling_block();
        let lower = combustion_like_backward_coupling_block();

        set_diag_block_dense(&mut a, 0, diag.as_slice(), block_size);
        set_diag_block_dense(&mut a, 1, diag.as_slice(), block_size);
        set_upper_block_dense(&mut a, 0, upper.as_slice(), block_size);
        set_lower_block_dense(&mut a, 0, lower.as_slice(), block_size);

        let mut lu = BlockTridiagonalLuConsistent::new(2, block_size).unwrap();
        let err = lu.factor_from(&a).unwrap_err();
        assert!(
            matches!(err, BandedError::ZeroPivot { .. }),
            "expected a structural zero-pivot on combustion-like 6x6 node block, got {err:?}"
        );
    }

    #[test]
    fn consistent_factorization_matches_pa_equals_lu_on_stabilized_combustion_like_superblocks() {
        let a = combustion_like_superblock_chain(3, 2, 2.5e3);
        let mut lu = BlockTridiagonalLuConsistent::new(3, 12).unwrap();
        lu.factor_from(&a).unwrap();

        let residual = lu.factor_residual_inf(&a).unwrap();
        assert!(
            residual < 1e-8,
            "factor residual too large for combustion-like superblocks: {residual:e}"
        );
    }

    #[test]
    fn consistent_factorization_solves_stabilized_combustion_like_superblock_chain() {
        let a = combustion_like_superblock_chain(3, 2, 2.5e3);
        let dense = a.to_dense();
        let x_true: Vec<f64> = (0..dense.len())
            .map(|i| 0.1 + (i as f64) * 2.5e-3)
            .collect();
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = BlockTridiagonalLuConsistent::new(3, 12).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();

        assert!(
            vec_diff_linf(&rhs, &x_true) < 1e-8,
            "solve drift too large on combustion-like superblock chain"
        );
    }

    #[test]
    fn consistent_factorization_solves_multiple_rhs_on_stabilized_combustion_like_superblocks() {
        let a = combustion_like_superblock_chain(2, 2, 2.5e3);
        let dense = a.to_dense();
        let n = dense.len();

        let x0: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 1.0e-3).collect();
        let x1: Vec<f64> = (0..n).map(|i| -0.5 + (i as f64) * 7.5e-4).collect();
        let b0 = dense_matvec(&dense, &x0);
        let b1 = dense_matvec(&dense, &x1);

        let mut rhs = vec![0.0; n * 2];
        rhs[..n].copy_from_slice(b0.as_slice());
        rhs[n..(2 * n)].copy_from_slice(b1.as_slice());

        let mut lu = BlockTridiagonalLuConsistent::new(2, 12).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_multiple_in_place(&mut rhs, 2, n).unwrap();

        assert!(vec_diff_linf(&rhs[..n], &x0) < 1e-8);
        assert!(vec_diff_linf(&rhs[n..(2 * n)], &x1) < 1e-8);
    }

    #[test]
    fn raw_combustion_like_superblocks_still_show_large_factor_residual() {
        let a = combustion_like_superblock_chain(3, 2, 0.0);
        let mut lu = BlockTridiagonalLuConsistent::new(3, 12).unwrap();
        lu.factor_from(&a).unwrap();

        let residual = lu.factor_residual_inf(&a).unwrap();
        assert!(
            residual < 1.0,
            "raw combustion-like superblocks unsolved: factor residual={residual:e}"
        );
    }

    #[test]
    fn raw_combustion_like_superblocks_still_show_large_solve_drift() {
        let a = combustion_like_superblock_chain(3, 2, 0.0);
        let dense = a.to_dense();
        let x_true: Vec<f64> = (0..dense.len())
            .map(|i| 0.1 + (i as f64) * 2.5e-3)
            .collect();
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = BlockTridiagonalLuConsistent::new(3, 12).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();
        let drift = vec_diff_linf(&rhs, &x_true);

        assert!(
            drift > 1.0e-6,
            "raw combustion-like superblocks unexpectedly solve accurately: drift={drift:e}"
        );
    }

    #[test]
    fn diagnostics_are_well_formed_on_stabilized_combustion_like_superblocks() {
        let a = combustion_like_superblock_chain(3, 2, 2.5e3);
        let mut lu = BlockTridiagonalLuConsistent::new(3, 12).unwrap();
        lu.factor_from(&a).unwrap();

        let rel_fac = lu.factor_residual_relative(&a).unwrap();
        let block_diag = lu.block_norm_diagnostics();
        let pivot_diag = lu.pivot_diagnostics();

        assert!(rel_fac >= 0.0);
        assert_eq!(block_diag.len(), 3);
        assert_eq!(pivot_diag.len(), 3);
        assert!(block_diag.iter().all(|d| d.diag_lu_linf.is_finite()));
        assert!(block_diag.iter().all(|d| d.min_abs_diag_u.is_finite()));
        assert!(block_diag.iter().all(|d| d.max_abs_diag_u.is_finite()));
    }

    #[test]
    fn diagnostic_raw_combustion_like_superblocks() {
        let a = combustion_like_superblock_chain(3, 2, 0.0);
        let mut lu = BlockTridiagonalLuConsistent::new(3, 12).unwrap();
        lu.factor_from(&a).unwrap();

        let abs_fac = lu.factor_residual_inf(&a).unwrap();
        let rel_fac = lu.factor_residual_relative(&a).unwrap();

        eprintln!("factor residual abs = {abs_fac:e}");
        eprintln!("factor residual rel = {rel_fac:e}");

        for d in lu.block_norm_diagnostics() {
            eprintln!(
                "block {}: ||diag_lu||inf={:e}, ||lower||inf={:?}, ||upper||inf={:?}, min|Uii|={:e}, max|Uii|={:e}",
                d.block_index,
                d.diag_lu_linf,
                d.lower_fac_linf,
                d.upper_fac_linf,
                d.min_abs_diag_u,
                d.max_abs_diag_u,
            );
        }

        for p in lu.pivot_diagnostics() {
            eprintln!(
                "block {}: swaps={}, max_swap_distance={}",
                p.block_index, p.num_nontrivial_swaps, p.max_swap_distance,
            );
        }
    }

    #[test]
    fn diagnostic_raw_combustion_like_superblock_scope_growth() {
        // This is a structural diagnostic, not a regression threshold test.
        // We intentionally compare a few wider superblocks to see whether
        // growth/pivot stress decreases when the local pivoting scope grows.
        for &nodes_per_superblock in &[2usize, 8, 16] {
            let a = combustion_like_superblock_chain(2, nodes_per_superblock, 0.0);
            let block_size = 6 * nodes_per_superblock;
            let mut lu = BlockTridiagonalLuConsistent::new(2, block_size).unwrap();

            eprintln!(
                "raw combustion-like superblock diagnostic: g={}, block_size={}",
                nodes_per_superblock, block_size
            );

            match lu.factor_from(&a) {
                Ok(()) => {
                    let abs_fac = lu.factor_residual_inf(&a).unwrap();
                    let rel_fac = lu.factor_residual_relative(&a).unwrap();
                    let block_diag = lu.block_norm_diagnostics();
                    let pivot_diag = lu.pivot_diagnostics();

                    let worst_lower = block_diag
                        .iter()
                        .filter_map(|d| d.lower_fac_linf)
                        .fold(0.0_f64, f64::max);
                    let worst_upper = block_diag
                        .iter()
                        .filter_map(|d| d.upper_fac_linf)
                        .fold(0.0_f64, f64::max);
                    let min_u = block_diag
                        .iter()
                        .map(|d| d.min_abs_diag_u)
                        .fold(f64::INFINITY, f64::min);
                    let max_swap_distance = pivot_diag
                        .iter()
                        .map(|d| d.max_swap_distance)
                        .fold(0usize, usize::max);
                    let total_swaps = pivot_diag
                        .iter()
                        .map(|d| d.num_nontrivial_swaps)
                        .sum::<usize>();

                    eprintln!("  factor residual: abs={:e}, rel={:e}", abs_fac, rel_fac);
                    eprintln!(
                        "  norm diagnostics: worst ||lower||inf={:e}, worst ||upper||inf={:e}, min |Uii|={:e}",
                        worst_lower, worst_upper, min_u
                    );
                    eprintln!(
                        "  pivot diagnostics: total_swaps={}, max_swap_distance={}",
                        total_swaps, max_swap_distance
                    );
                }
                Err(err) => {
                    eprintln!("  factorization failed: {err:?}");
                }
            }
        }
    }

    #[test]
    fn diagnostic_raw_combustion_like_superblock_solve_quality() {
        // Structural solve-quality diagnostic against a dense reference solve.
        // This is the test that tells us whether a moderate superblock size is
        // not only algebraically plausible, but also numerically usable.
        for &nodes_per_superblock in &[2usize, 4, 8, 16] {
            let a = combustion_like_superblock_chain(2, nodes_per_superblock, 0.0);
            let dense = a.to_dense();
            let n = dense.len();
            let x_true: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 2.5e-3).collect();
            let rhs = dense_matvec(&dense, &x_true);
            let x_ref = dense_reference_solve(&dense, &rhs);

            let block_size = 6 * nodes_per_superblock;
            let mut lu = BlockTridiagonalLuConsistent::new(2, block_size).unwrap();

            eprintln!(
                "raw combustion-like solve diagnostic: g={}, block_size={}",
                nodes_per_superblock, block_size
            );

            match lu.factor_from(&a) {
                Ok(()) => {
                    let mut x_block = rhs.clone();
                    lu.solve_in_place(&mut x_block).unwrap();

                    let factor_rel = lu.factor_residual_relative(&a).unwrap();
                    let solve_res_rel = relative_solve_residual(&dense, &x_block, &rhs);
                    let ref_res_rel = relative_solve_residual(&dense, &x_ref, &rhs);
                    let diff_vs_ref = vec_diff_linf(&x_block, &x_ref);
                    let diff_vs_true = vec_diff_linf(&x_block, &x_true);
                    let ref_diff_vs_true = vec_diff_linf(&x_ref, &x_true);

                    eprintln!(
                        "  factor residual rel={:e}, solve residual rel={:e}, reference residual rel={:e}",
                        factor_rel, solve_res_rel, ref_res_rel
                    );
                    eprintln!(
                        "  solution drift: ||x_block-x_ref||inf={:e}, ||x_block-x_true||inf={:e}, ||x_ref-x_true||inf={:e}",
                        diff_vs_ref, diff_vs_true, ref_diff_vs_true
                    );
                }
                Err(err) => {
                    eprintln!("  factorization failed: {err:?}");
                }
            }
        }
    }

    #[test]
    fn diagnostic_raw_combustion_like_superblock_refinement_and_conditioning() {
        // This diagnostic helps separate two possibilities:
        // 1. the block solve is already backward-stable, but the matrix is very ill-conditioned;
        // 2. the block solve itself still injects extra error that refinement can partially remove.
        for &nodes_per_superblock in &[2usize, 4, 8, 16] {
            let a = combustion_like_superblock_chain(2, nodes_per_superblock, 0.0);
            let dense = a.to_dense();
            let n = dense.len();
            let x_true: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 2.5e-3).collect();
            let rhs = dense_matvec(&dense, &x_true);
            let x_ref = dense_reference_solve(&dense, &rhs);
            let cond_inf = estimate_dense_cond_inf(&dense);

            let block_size = 6 * nodes_per_superblock;
            let mut lu = BlockTridiagonalLuConsistent::new(2, block_size).unwrap();

            eprintln!(
                "raw combustion-like refinement diagnostic: g={}, block_size={}",
                nodes_per_superblock, block_size
            );

            match lu.factor_from(&a) {
                Ok(()) => {
                    let mut x0 = rhs.clone();
                    lu.solve_in_place_with_iterative_refinement(&a, &mut x0, 0)
                        .expect("raw combustion-like block solve should succeed");
                    let mut x1 = rhs.clone();
                    lu.solve_in_place_with_iterative_refinement(&a, &mut x1, 1)
                        .expect("raw combustion-like block solve + 1 refinement should succeed");
                    let mut x2 = rhs.clone();
                    lu.solve_in_place_with_iterative_refinement(&a, &mut x2, 2)
                        .expect("raw combustion-like block solve + 2 refinements should succeed");

                    let rr0 = relative_solve_residual(&dense, &x0, &rhs);
                    let rr1 = relative_solve_residual(&dense, &x1, &rhs);
                    let rr2 = relative_solve_residual(&dense, &x2, &rhs);

                    let d0 = vec_diff_linf(&x0, &x_ref);
                    let d1 = vec_diff_linf(&x1, &x_ref);
                    let d2 = vec_diff_linf(&x2, &x_ref);

                    eprintln!(
                        "  cond_inf≈{:e}, cond_inf * rr0≈{:e}",
                        cond_inf,
                        cond_inf * rr0
                    );
                    eprintln!(
                        "  relative residuals: no_refine={:e}, refine1={:e}, refine2={:e}",
                        rr0, rr1, rr2
                    );
                    eprintln!(
                        "  drift vs ref: no_refine={:e}, refine1={:e}, refine2={:e}, ref_vs_true={:e}",
                        d0,
                        d1,
                        d2,
                        vec_diff_linf(&x_ref, &x_true)
                    );
                }
                Err(err) => {
                    eprintln!("  factorization failed: {err:?}");
                }
            }
        }
    }

    #[test]
    fn iterative_refinement_recovers_raw_combustion_like_superblock_solution() {
        let a = combustion_like_superblock_chain(2, 2, 0.0);
        let dense = a.to_dense();
        let n = dense.len();
        let x_true: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 2.5e-3).collect();
        let rhs = dense_matvec(&dense, &x_true);
        let x_ref = dense_reference_solve(&dense, &rhs);

        let mut lu = BlockTridiagonalLuConsistent::new(2, 12).unwrap();
        lu.factor_from(&a).unwrap();

        let mut x = rhs.clone();
        lu.solve_in_place_with_iterative_refinement(&a, &mut x, 1)
            .expect("iterative refinement should succeed on raw combustion-like superblocks");

        assert!(
            vec_diff_linf(&x, &x_ref) < 1e-10,
            "iterative refinement should recover the dense reference solution"
        );
    }
}
