use super::error::BandedError;

/// Block-tridiagonal matrix with dense square blocks.
///
/// Structure:
///   diag[k]  : block on main diagonal, k = 0..n_blocks-1
///   lower[k] : block below diagonal, connecting block row (k+1) to block col k
///   upper[k] : block above diagonal, connecting block row k to block col (k+1)
///
/// Each block is stored as a flat row-major array of length block_size * block_size:
///   block[i * block_size + j] = value at (i, j)
#[derive(Clone, Debug)]
pub struct BlockTridiagonal {
    n_blocks: usize,
    block_size: usize,
    lower: Vec<Vec<f64>>, // len = n_blocks - 1
    diag: Vec<Vec<f64>>,  // len = n_blocks
    upper: Vec<Vec<f64>>, // len = n_blocks - 1
}

impl BlockTridiagonal {
    pub fn zeros(n_blocks: usize, block_size: usize) -> Result<Self, BandedError> {
        if n_blocks == 0 || block_size == 0 {
            return Err(BandedError::DimensionMismatch);
        }

        let block_len = block_size
            .checked_mul(block_size)
            .ok_or(BandedError::DimensionMismatch)?;

        let mut diag = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            diag.push(vec![0.0; block_len]);
        }

        let off_len = n_blocks.saturating_sub(1);

        let mut lower = Vec::with_capacity(off_len);
        let mut upper = Vec::with_capacity(off_len);
        for _ in 0..off_len {
            lower.push(vec![0.0; block_len]);
            upper.push(vec![0.0; block_len]);
        }

        Ok(Self {
            n_blocks,
            block_size,
            lower,
            diag,
            upper,
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
    pub fn block_len(&self) -> usize {
        self.block_size * self.block_size
    }

    #[inline]
    pub fn lower_blocks(&self) -> &[Vec<f64>] {
        &self.lower
    }

    #[inline]
    pub fn diag_blocks(&self) -> &[Vec<f64>] {
        &self.diag
    }

    #[inline]
    pub fn upper_blocks(&self) -> &[Vec<f64>] {
        &self.upper
    }

    #[inline]
    pub fn lower_blocks_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.lower
    }

    #[inline]
    pub fn diag_blocks_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.diag
    }

    #[inline]
    pub fn upper_blocks_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.upper
    }

    #[inline]
    pub fn diag_block(&self, k: usize) -> Option<&[f64]> {
        self.diag.get(k).map(|b| b.as_slice())
    }

    #[inline]
    pub fn diag_block_mut(&mut self, k: usize) -> Option<&mut [f64]> {
        self.diag.get_mut(k).map(|b| b.as_mut_slice())
    }

    /// lower block at position (k+1, k)
    #[inline]
    pub fn lower_block(&self, k: usize) -> Option<&[f64]> {
        self.lower.get(k).map(|b| b.as_slice())
    }

    #[inline]
    pub fn lower_block_mut(&mut self, k: usize) -> Option<&mut [f64]> {
        self.lower.get_mut(k).map(|b| b.as_mut_slice())
    }

    /// upper block at position (k, k+1)
    #[inline]
    pub fn upper_block(&self, k: usize) -> Option<&[f64]> {
        self.upper.get(k).map(|b| b.as_slice())
    }

    #[inline]
    pub fn upper_block_mut(&mut self, k: usize) -> Option<&mut [f64]> {
        self.upper.get_mut(k).map(|b| b.as_mut_slice())
    }

    #[inline]
    pub fn block_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.block_size || j >= self.block_size {
            return None;
        }
        Some(i * self.block_size + j)
    }

    pub fn set_diag(
        &mut self,
        k: usize,
        i: usize,
        j: usize,
        value: f64,
    ) -> Result<(), BandedError> {
        let idx = self
            .block_index(i, j)
            .ok_or(BandedError::OutOfBounds { i, j })?;
        let blk = self.diag.get_mut(k).ok_or(BandedError::DimensionMismatch)?;
        blk[idx] = value;
        Ok(())
    }

    pub fn set_lower(
        &mut self,
        k: usize,
        i: usize,
        j: usize,
        value: f64,
    ) -> Result<(), BandedError> {
        let idx = self
            .block_index(i, j)
            .ok_or(BandedError::OutOfBounds { i, j })?;
        let blk = self
            .lower
            .get_mut(k)
            .ok_or(BandedError::DimensionMismatch)?;
        blk[idx] = value;
        Ok(())
    }

    pub fn set_upper(
        &mut self,
        k: usize,
        i: usize,
        j: usize,
        value: f64,
    ) -> Result<(), BandedError> {
        let idx = self
            .block_index(i, j)
            .ok_or(BandedError::OutOfBounds { i, j })?;
        let blk = self
            .upper
            .get_mut(k)
            .ok_or(BandedError::DimensionMismatch)?;
        blk[idx] = value;
        Ok(())
    }

    pub fn get_diag(&self, k: usize, i: usize, j: usize) -> Result<f64, BandedError> {
        let idx = self
            .block_index(i, j)
            .ok_or(BandedError::OutOfBounds { i, j })?;
        let blk = self.diag.get(k).ok_or(BandedError::DimensionMismatch)?;
        Ok(blk[idx])
    }

    pub fn get_lower(&self, k: usize, i: usize, j: usize) -> Result<f64, BandedError> {
        let idx = self
            .block_index(i, j)
            .ok_or(BandedError::OutOfBounds { i, j })?;
        let blk = self.lower.get(k).ok_or(BandedError::DimensionMismatch)?;
        Ok(blk[idx])
    }

    pub fn get_upper(&self, k: usize, i: usize, j: usize) -> Result<f64, BandedError> {
        let idx = self
            .block_index(i, j)
            .ok_or(BandedError::OutOfBounds { i, j })?;
        let blk = self.upper.get(k).ok_or(BandedError::DimensionMismatch)?;
        Ok(blk[idx])
    }

    /// Convert to dense matrix for debugging/tests.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let n = self.n();
        let bs = self.block_size;
        let mut out = vec![vec![0.0; n]; n];

        for blk in 0..self.n_blocks {
            let r0 = blk * bs;
            let c0 = blk * bs;

            // diagonal
            let d = &self.diag[blk];
            for i in 0..bs {
                for j in 0..bs {
                    out[r0 + i][c0 + j] = d[i * bs + j];
                }
            }

            // lower
            if blk > 0 {
                let l = &self.lower[blk - 1];
                let lr0 = blk * bs;
                let lc0 = (blk - 1) * bs;
                for i in 0..bs {
                    for j in 0..bs {
                        out[lr0 + i][lc0 + j] = l[i * bs + j];
                    }
                }
            }

            // upper
            if blk + 1 < self.n_blocks {
                let u = &self.upper[blk];
                let ur0 = blk * bs;
                let uc0 = (blk + 1) * bs;
                for i in 0..bs {
                    for j in 0..bs {
                        out[ur0 + i][uc0 + j] = u[i * bs + j];
                    }
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::BlockTridiagonal;

    #[test]
    fn block_tridiagonal_zeros_shape() {
        let a = BlockTridiagonal::zeros(4, 3).unwrap();

        assert_eq!(a.n_blocks(), 4);
        assert_eq!(a.block_size(), 3);
        assert_eq!(a.n(), 12);

        assert_eq!(a.diag_blocks().len(), 4);
        assert_eq!(a.lower_blocks().len(), 3);
        assert_eq!(a.upper_blocks().len(), 3);

        for blk in a.diag_blocks() {
            assert_eq!(blk.len(), 9);
        }
    }

    #[test]
    fn block_set_and_get() {
        let mut a = BlockTridiagonal::zeros(3, 2).unwrap();

        a.set_diag(0, 0, 1, 5.0).unwrap();
        a.set_lower(0, 1, 0, 2.0).unwrap();
        a.set_upper(1, 0, 1, 7.0).unwrap();

        assert_eq!(a.get_diag(0, 0, 1).unwrap(), 5.0);
        assert_eq!(a.get_lower(0, 1, 0).unwrap(), 2.0);
        assert_eq!(a.get_upper(1, 0, 1).unwrap(), 7.0);
    }

    #[test]
    fn block_to_dense_layout() {
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        // diag block 0
        a.set_diag(0, 0, 0, 1.0).unwrap();
        a.set_diag(0, 0, 1, 2.0).unwrap();
        a.set_diag(0, 1, 0, 3.0).unwrap();
        a.set_diag(0, 1, 1, 4.0).unwrap();

        // upper block 0
        a.set_upper(0, 0, 0, 5.0).unwrap();
        a.set_upper(0, 0, 1, 6.0).unwrap();
        a.set_upper(0, 1, 0, 7.0).unwrap();
        a.set_upper(0, 1, 1, 8.0).unwrap();

        // lower block 0
        a.set_lower(0, 0, 0, 9.0).unwrap();
        a.set_lower(0, 0, 1, 10.0).unwrap();
        a.set_lower(0, 1, 0, 11.0).unwrap();
        a.set_lower(0, 1, 1, 12.0).unwrap();

        // diag block 1
        a.set_diag(1, 0, 0, 13.0).unwrap();
        a.set_diag(1, 0, 1, 14.0).unwrap();
        a.set_diag(1, 1, 0, 15.0).unwrap();
        a.set_diag(1, 1, 1, 16.0).unwrap();

        let d = a.to_dense();

        let expected = vec![
            vec![1.0, 2.0, 5.0, 6.0],
            vec![3.0, 4.0, 7.0, 8.0],
            vec![9.0, 10.0, 13.0, 14.0],
            vec![11.0, 12.0, 15.0, 16.0],
        ];

        assert_eq!(d, expected);
    }
}
