use super::superblock_layout::SuperBlockLayout;
use super::{block_tridiagonal::BlockTridiagonal, error::BandedError, storage::Banded};

#[derive(Clone, Debug)]
pub struct BandedAssembly {
    n: usize,
    kl: usize,
    ku: usize,
    diagonals: Vec<Vec<f64>>,
}

impl BandedAssembly {
    pub fn zeros(n: usize, kl: usize, ku: usize) -> Result<Self, BandedError> {
        if n == 0 {
            return Err(BandedError::DimensionMismatch);
        }

        let mut diagonals = Vec::with_capacity(kl + ku + 1);

        for d in -(kl as isize)..=(ku as isize) {
            let len = Self::diag_len_static(n, d);
            diagonals.push(vec![0.0; len]);
        }

        Ok(Self {
            n,
            kl,
            ku,
            diagonals,
        })
    }

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
    pub fn num_diagonals(&self) -> usize {
        self.diagonals.len()
    }

    #[inline]
    pub fn diag_index(&self, offset: isize) -> Option<usize> {
        if offset < -(self.kl as isize) || offset > self.ku as isize {
            return None;
        }
        Some((offset + self.kl as isize) as usize)
    }

    #[inline]
    pub fn diag_len(&self, offset: isize) -> Option<usize> {
        self.diag_index(offset).map(|idx| self.diagonals[idx].len())
    }

    #[inline]
    fn diag_len_static(n: usize, offset: isize) -> usize {
        if offset >= 0 {
            n.saturating_sub(offset as usize)
        } else {
            n.saturating_sub((-offset) as usize)
        }
    }

    #[inline]
    pub fn diag(&self, offset: isize) -> Option<&[f64]> {
        self.diag_index(offset)
            .map(|idx| self.diagonals[idx].as_slice())
    }

    #[inline]
    pub fn diag_mut(&mut self, offset: isize) -> Option<&mut [f64]> {
        self.diag_index(offset)
            .map(|idx| self.diagonals[idx].as_mut_slice())
    }

    #[inline]
    pub fn diagonals(&self) -> &[Vec<f64>] {
        &self.diagonals
    }

    #[inline]
    pub fn diagonals_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.diagonals
    }

    /// For an element index within a diagonal, return the corresponding (i, j).
    ///
    /// offset = j - i
    pub fn diag_pos_to_ij(&self, offset: isize, pos: usize) -> Result<(usize, usize), BandedError> {
        let len = self
            .diag_len(offset)
            .ok_or(BandedError::DimensionMismatch)?;

        if pos >= len {
            return Err(BandedError::DimensionMismatch);
        }

        if offset >= 0 {
            let j0 = offset as usize;
            let i = pos;
            let j = j0 + pos;
            Ok((i, j))
        } else {
            let i0 = (-offset) as usize;
            let i = i0 + pos;
            let j = pos;
            Ok((i, j))
        }
    }

    /// For global matrix coordinates (i, j), return (diag_offset, pos_inside_diag).
    pub fn ij_to_diag_pos(&self, i: usize, j: usize) -> Result<(isize, usize), BandedError> {
        if i >= self.n || j >= self.n {
            return Err(BandedError::OutOfBounds { i, j });
        }

        let offset = j as isize - i as isize;
        if self.diag_index(offset).is_none() {
            return Err(BandedError::OutOfBounds { i, j });
        }

        let pos = if offset >= 0 { i } else { j };
        Ok((offset, pos))
    }

    pub fn get(&self, i: usize, j: usize) -> Result<f64, BandedError> {
        let (offset, pos) = self.ij_to_diag_pos(i, j)?;
        let diag = self.diag(offset).unwrap();
        Ok(diag[pos])
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) -> Result<(), BandedError> {
        let (offset, pos) = self.ij_to_diag_pos(i, j)?;
        let diag = self.diag_mut(offset).unwrap();
        diag[pos] = value;
        Ok(())
    }

    pub fn fill_zero(&mut self) {
        for d in &mut self.diagonals {
            d.fill(0.0);
        }
    }

    pub fn to_banded(&self) -> Result<Banded<f64>, BandedError> {
        let mut out = Banded::<f64>::zeros(self.n, self.kl, self.ku)?;

        for offset in -(self.kl as isize)..=(self.ku as isize) {
            let diag = self.diag(offset).unwrap();
            for pos in 0..diag.len() {
                let (i, j) = self.diag_pos_to_ij(offset, pos)?;
                out[(i, j)] = diag[pos];
            }
        }

        Ok(out)
    }

    pub fn to_block_tridiagonal(
        &self,
        n_blocks: usize,
        block_size: usize,
    ) -> Result<BlockTridiagonal, BandedError> {
        if n_blocks == 0 || block_size == 0 || n_blocks * block_size != self.n {
            return Err(BandedError::DimensionMismatch);
        }

        let mut out = BlockTridiagonal::zeros(n_blocks, block_size)?;

        for blk in 0..n_blocks {
            let r0 = blk * block_size;
            let c0 = blk * block_size;

            // diagonal block
            for i in 0..block_size {
                for j in 0..block_size {
                    let gi = r0 + i;
                    let gj = c0 + j;
                    if self.in_band(gi, gj) {
                        let v = self.get(gi, gj)?;
                        out.set_diag(blk, i, j, v)?;
                    }
                }
            }

            // lower block
            if blk > 0 {
                let lr0 = blk * block_size;
                let lc0 = (blk - 1) * block_size;
                for i in 0..block_size {
                    for j in 0..block_size {
                        let gi = lr0 + i;
                        let gj = lc0 + j;
                        if let Ok(v) = self.get(gi, gj) {
                            out.set_lower(blk - 1, i, j, v)?;
                        }
                    }
                }
            }

            // upper block
            if blk + 1 < n_blocks {
                let ur0 = blk * block_size;
                let uc0 = (blk + 1) * block_size;
                for i in 0..block_size {
                    for j in 0..block_size {
                        let gi = ur0 + i;
                        let gj = uc0 + j;
                        if let Ok(v) = self.get(gi, gj) {
                            out.set_upper(blk, i, j, v)?;
                        }
                    }
                }
            }
        }

        Ok(out)
    }

    pub fn for_each_diag_mut<F>(&mut self, mut f: F) -> Result<(), BandedError>
    where
        F: FnMut(isize, &mut [f64]) -> Result<(), BandedError>,
    {
        for d_idx in 0..self.diagonals.len() {
            let offset = self.diag_index_to_offset(d_idx)?;
            f(offset, &mut self.diagonals[d_idx])?;
        }
        Ok(())
    }

    #[inline]
    pub fn in_band(&self, i: usize, j: usize) -> bool {
        if i >= self.n || j >= self.n {
            return false;
        }
        let offset = j as isize - i as isize;
        offset >= -(self.kl as isize) && offset <= self.ku as isize
    }

    #[inline]
    pub fn min_offset(&self) -> isize {
        -(self.kl as isize)
    }

    #[inline]
    pub fn max_offset(&self) -> isize {
        self.ku as isize
    }

    pub fn offsets(&self) -> impl Iterator<Item = isize> + '_ {
        self.min_offset()..=self.max_offset()
    }

    #[inline]
    pub fn diag_index_to_offset(&self, diag_index: usize) -> Result<isize, BandedError> {
        if diag_index >= self.diagonals.len() {
            return Err(BandedError::DimensionMismatch);
        }
        Ok(diag_index as isize - self.kl as isize)
    }

    pub fn fill_diag_with<F>(&mut self, offset: isize, mut f: F) -> Result<(), BandedError>
    where
        F: FnMut(usize, usize) -> f64,
    {
        let len = self
            .diag_len(offset)
            .ok_or(BandedError::DimensionMismatch)?;
        for pos in 0..len {
            let (i, j) = self.diag_pos_to_ij(offset, pos)?;
            let diag = self
                .diag_mut(offset)
                .ok_or(BandedError::DimensionMismatch)?;
            diag[pos] = f(i, j);
        }
        Ok(())
    }

    pub fn fill_all_diagonals_with<F>(&mut self, mut f: F) -> Result<(), BandedError>
    where
        F: FnMut(isize, usize, usize) -> f64,
    {
        let offsets: Vec<isize> = self.offsets().collect();
        for offset in offsets {
            self.fill_diag_with(offset, |i, j| f(offset, i, j))?;
        }
        Ok(())
    }

    pub fn finalize_banded(&self) -> Result<Banded<f64>, BandedError> {
        self.to_banded()
    }

    pub fn finalize_block_tridiagonal(
        &self,
        n_blocks: usize,
        block_size: usize,
    ) -> Result<BlockTridiagonal, BandedError> {
        self.to_block_tridiagonal(n_blocks, block_size)
    }

    pub fn finalize_superblock_tridiagonal(
        &self,
        layout: &SuperBlockLayout,
    ) -> Result<BlockTridiagonal, BandedError> {
        if layout.n_total() != self.n() {
            return Err(BandedError::DimensionMismatch);
        }

        // The current native block-tridiagonal solver expects a single dense
        // block size for every block. We therefore reject tail superblocks for
        // now and keep superblock experiments on the evenly-divisible cases.
        if !layout.is_evenly_divisible() {
            return Err(BandedError::DimensionMismatch);
        }

        self.to_block_tridiagonal(layout.n_blocks(), layout.block_size())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiagInfo {
    pub offset: isize,
    pub len: usize,
}

impl BandedAssembly {
    pub fn diag_infos(&self) -> Vec<DiagInfo> {
        self.offsets()
            .map(|offset| DiagInfo {
                offset,
                len: self.diag_len(offset).unwrap(),
            })
            .collect()
    }
}

pub fn fill_banded_assembly_sequential<F>(
    asm: &mut BandedAssembly,
    mut f: F,
) -> Result<(), BandedError>
where
    F: FnMut(isize, usize, usize) -> f64,
{
    let offsets: Vec<isize> = asm.offsets().collect();
    for offset in offsets {
        asm.fill_diag_with(offset, |i, j| f(offset, i, j))?;
    }
    Ok(())
}

/*
примерно так с якобианом
let infos = asm.diag_infos();
asm.diagonals_mut()
    .par_iter_mut()
    .zip(infos.into_par_iter())
    .for_each(|(diag, info)| {
        for pos in 0..info.len {
            let (i, j) = if info.offset >= 0 {
                (pos, info.offset as usize + pos)
            } else {
                ((-info.offset) as usize + pos, pos)
            };
            diag[pos] = jac_entry(i, j, state);
        }
    });


*/
pub fn fill_banded_assembly_parallel<F>(asm: &mut BandedAssembly, f: F) -> Result<(), BandedError>
where
    F: Fn(isize, usize, usize) -> f64 + Sync + Send,
{
    use rayon::prelude::*;

    let kl = asm.kl();
    let infos: Vec<(isize, usize)> = asm
        .diagonals()
        .iter()
        .enumerate()
        .map(|(d_idx, diag)| (d_idx as isize - kl as isize, diag.len()))
        .collect();

    asm.diagonals_mut()
        .par_iter_mut()
        .zip(infos.into_par_iter())
        .for_each(|(diag, (offset, len))| {
            for pos in 0..len {
                let (i, j) = if offset >= 0 {
                    (pos, offset as usize + pos)
                } else {
                    ((-offset) as usize + pos, pos)
                };

                diag[pos] = f(offset, i, j);
            }
        });

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::BandedAssembly;

    #[test]
    fn diag_lengths_are_correct() {
        let a = BandedAssembly::zeros(5, 2, 1).unwrap();

        assert_eq!(a.diag_len(-2), Some(3));
        assert_eq!(a.diag_len(-1), Some(4));
        assert_eq!(a.diag_len(0), Some(5));
        assert_eq!(a.diag_len(1), Some(4));
    }

    #[test]
    fn ij_to_diag_pos_and_back() {
        let a = BandedAssembly::zeros(6, 2, 2).unwrap();

        let cases = [(0, 0), (0, 1), (1, 0), (3, 5), (5, 3), (4, 4)];

        for (i, j) in cases {
            let (offset, pos) = a.ij_to_diag_pos(i, j).unwrap();
            let (ii, jj) = a.diag_pos_to_ij(offset, pos).unwrap();
            assert_eq!((ii, jj), (i, j));
        }
    }

    #[test]
    fn set_get_roundtrip() {
        let mut a = BandedAssembly::zeros(6, 2, 2).unwrap();

        a.set(0, 0, 1.0).unwrap();
        a.set(0, 1, 2.0).unwrap();
        a.set(1, 0, 3.0).unwrap();
        a.set(4, 4, 4.0).unwrap();
        a.set(5, 3, 5.0).unwrap();

        assert_eq!(a.get(0, 0).unwrap(), 1.0);
        assert_eq!(a.get(0, 1).unwrap(), 2.0);
        assert_eq!(a.get(1, 0).unwrap(), 3.0);
        assert_eq!(a.get(4, 4).unwrap(), 4.0);
        assert_eq!(a.get(5, 3).unwrap(), 5.0);
    }

    #[test]
    fn converts_to_banded() {
        let mut a = BandedAssembly::zeros(4, 1, 1).unwrap();

        a.set(0, 0, 4.0).unwrap();
        a.set(0, 1, 1.0).unwrap();
        a.set(1, 0, 2.0).unwrap();
        a.set(1, 1, 5.0).unwrap();
        a.set(1, 2, 1.0).unwrap();
        a.set(2, 1, 3.0).unwrap();
        a.set(2, 2, 6.0).unwrap();
        a.set(2, 3, 1.0).unwrap();
        a.set(3, 2, 4.0).unwrap();
        a.set(3, 3, 7.0).unwrap();

        let b = a.to_banded().unwrap();

        assert_eq!(b[(0, 0)], 4.0);
        assert_eq!(b[(0, 1)], 1.0);
        assert_eq!(b[(1, 0)], 2.0);
        assert_eq!(b[(3, 3)], 7.0);
    }

    #[test]
    fn converts_to_block_tridiagonal() {
        // 2 blocks of size 2 => scalar half-bandwidth must be at least 2
        let mut a = BandedAssembly::zeros(4, 2, 2).unwrap();

        a.set(0, 0, 1.0).unwrap();
        a.set(0, 1, 2.0).unwrap();
        a.set(1, 0, 3.0).unwrap();
        a.set(1, 1, 4.0).unwrap();

        a.set(0, 2, 5.0).unwrap();
        a.set(1, 3, 6.0).unwrap();

        a.set(2, 0, 7.0).unwrap();
        a.set(3, 1, 8.0).unwrap();

        a.set(2, 2, 9.0).unwrap();
        a.set(2, 3, 10.0).unwrap();
        a.set(3, 2, 11.0).unwrap();
        a.set(3, 3, 12.0).unwrap();

        let blk = a.to_block_tridiagonal(2, 2).unwrap();
        let d = blk.to_dense();

        assert_eq!(d[0][0], 1.0);
        assert_eq!(d[0][2], 5.0);
        assert_eq!(d[2][0], 7.0);
        assert_eq!(d[3][3], 12.0);
    }

    #[test]
    fn fill_diag_with_works() {
        let mut a = BandedAssembly::zeros(5, 2, 2).unwrap();

        a.fill_diag_with(1, |i, j| (10 * i + j) as f64).unwrap();

        assert_eq!(a.get(0, 1).unwrap(), 1.0);
        assert_eq!(a.get(1, 2).unwrap(), 12.0);
        assert_eq!(a.get(2, 3).unwrap(), 23.0);
        assert_eq!(a.get(3, 4).unwrap(), 34.0);
    }
}
