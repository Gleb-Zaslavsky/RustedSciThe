use super::{error::BandedError, storage::Banded};

/// Expanded LU workspace for general banded LU without pivoting.
///
/// Layout:
/// data[(ext_ku + i - j) * n + j] = A[i, j]
///
/// where:
/// - lower bandwidth is still `kl`
/// - upper bandwidth is expanded to `ext_ku`
///
/// For no-pivot LU we can safely choose:
///     ext_ku = kl + ku
///
/// Then the total number of stored band rows is:
///     kl + ext_ku + 1 = 2*kl + ku + 1
#[derive(Clone, Debug)]
pub struct BandedLuStorage {
    n: usize,
    kl: usize,
    ku: usize,
    ext_ku: usize,
    data: Vec<f64>,
}

impl BandedLuStorage {
    pub fn new(n: usize, kl: usize, ku: usize) -> Result<Self, BandedError> {
        let ext_ku = kl
            .checked_add(ku)
            .ok_or(BandedError::InvalidBand { n, kl, ku })?;

        let rows = kl
            .checked_add(ext_ku)
            .and_then(|x| x.checked_add(1))
            .ok_or(BandedError::InvalidBand { n, kl, ku })?;

        let len = rows
            .checked_mul(n)
            .ok_or(BandedError::InvalidBand { n, kl, ku })?;

        Ok(Self {
            n,
            kl,
            ku,
            ext_ku,
            data: vec![0.0; len],
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
    pub fn ext_ku(&self) -> usize {
        self.ext_ku
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.kl + self.ext_ku + 1
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    #[inline]
    pub fn fill_zero(&mut self) {
        self.data.fill(0.0);
    }

    #[inline]
    pub fn in_storage_band(&self, i: usize, j: usize) -> bool {
        i < self.n && j < self.n && i + self.ext_ku >= j && j + self.kl >= i
    }

    #[inline]
    pub fn offset(&self, i: usize, j: usize) -> Option<usize> {
        if !self.in_storage_band(i, j) {
            return None;
        }
        let band_row = self.ext_ku + i - j;
        Some(band_row * self.n + j)
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        self.offset(i, j).map(|idx| self.data[idx])
    }

    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut f64> {
        self.offset(i, j).map(|idx| &mut self.data[idx])
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) -> Result<(), BandedError> {
        let idx = self.offset(i, j).ok_or(BandedError::OutOfBounds { i, j })?;
        self.data[idx] = value;
        Ok(())
    }

    #[inline]
    pub fn add_assign(&mut self, i: usize, j: usize, value: f64) -> Result<(), BandedError> {
        let idx = self.offset(i, j).ok_or(BandedError::OutOfBounds { i, j })?;
        self.data[idx] += value;
        Ok(())
    }

    /// Copy compact banded matrix into expanded LU workspace.
    ///
    /// All entries outside the original compact band remain zero.
    pub fn copy_from_compact(&mut self, a: &Banded<f64>) -> Result<(), BandedError> {
        if self.n != a.n() || self.kl != a.kl() || self.ku != a.ku() {
            return Err(BandedError::DimensionMismatch);
        }

        self.fill_zero();

        let n = self.n;
        for j in 0..n {
            let i0 = j.saturating_sub(self.ku);
            let i1 = (j + self.kl + 1).min(n);

            for i in i0..i1 {
                let value = a[(i, j)];
                let idx = self
                    .offset(i, j)
                    .expect("original band must fit into LU storage");
                self.data[idx] = value;
            }
        }

        Ok(())
    }
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut out = vec![vec![0.0; n]; n];

        for j in 0..n {
            let i0 = j.saturating_sub(self.ext_ku);
            let i1 = (j + self.kl + 1).min(n);

            for i in i0..i1 {
                if let Some(v) = self.get(i, j) {
                    out[i][j] = v;
                }
            }
        }

        out
    }

    pub fn debug_dump_dense(&self) -> String {
        let dense = self.to_dense();
        let mut s = String::new();

        for row in dense {
            for (k, v) in row.iter().enumerate() {
                if k > 0 {
                    s.push(' ');
                }
                s.push_str(&format!("{:>12.5e}", v));
            }
            s.push('\n');
        }

        s
    }
    /// Column range in which logical row `i` may have stored entries
    /// in the expanded LU workspace.
    #[inline]
    pub fn col_range_for_row(&self, i: usize) -> std::ops::Range<usize> {
        let j0 = i.saturating_sub(self.kl);
        let j1 = (i + self.ext_ku + 1).min(self.n);
        j0..j1
    }

    /// Read logical matrix value at (i, j), returning 0.0 if the location
    /// is outside the workspace band.
    #[inline]
    pub fn row_value(&self, i: usize, j: usize) -> f64 {
        self.get(i, j).unwrap_or(0.0)
    }

    /// Write logical matrix value at (i, j) if the location exists in the
    /// workspace band. Writing zero outside the band is ignored.
    ///
    /// Returns an error only when trying to write a nonzero value outside band.
    #[inline]
    pub fn set_row_value(&mut self, i: usize, j: usize, value: f64) -> Result<(), BandedError> {
        match self.offset(i, j) {
            Some(idx) => {
                self.as_mut_slice()[idx] = value;
                Ok(())
            }
            None => {
                if value == 0.0 {
                    Ok(())
                } else {
                    Err(BandedError::OutOfBounds { i, j })
                }
            }
        }
    }

    /// Swap logical rows `i1` and `i2` over the column fragment [j0, j1).
    ///
    /// This is a *logical row swap* inside expanded band storage:
    /// for each column j in [j0, j1), values A[i1, j] and A[i2, j]
    /// are exchanged, with implicit zeros respected outside the stored band.
    ///
    /// This operation is intended for pivoting support.
    pub fn swap_rows_fragment(
        &mut self,
        i1: usize,
        i2: usize,
        j0: usize,
        j1: usize,
    ) -> Result<(), BandedError> {
        if i1 >= self.n || i2 >= self.n {
            return Err(BandedError::DimensionMismatch);
        }
        if j0 > j1 || j1 > self.n {
            return Err(BandedError::DimensionMismatch);
        }
        if i1 == i2 || j0 == j1 {
            return Ok(());
        }

        for j in j0..j1 {
            let v1 = self.row_value(i1, j);
            let v2 = self.row_value(i2, j);

            self.set_row_value(i1, j, v2)?;
            self.set_row_value(i2, j, v1)?;
        }

        Ok(())
    }
}
//========================================================================================
#[cfg(test)]
mod tests {
    use super::BandedLuStorage;
    use super::*;
    use crate::somelinalg::banded::GeneralBandedLuPartialPivot;
    use crate::somelinalg::banded::banded_matvec;
    use crate::somelinalg::banded::storage::Banded;
    fn dense_from_storage(s: &BandedLuStorage) -> Vec<Vec<f64>> {
        s.to_dense()
    }

    #[test]
    fn col_range_for_row_is_reasonable() {
        let s = BandedLuStorage::new(8, 2, 1).unwrap();
        // ext_ku = kl + ku = 3

        assert_eq!(s.col_range_for_row(0), 0..4);
        assert_eq!(s.col_range_for_row(1), 0..5);
        assert_eq!(s.col_range_for_row(2), 0..6);
        assert_eq!(s.col_range_for_row(5), 3..8);
        assert_eq!(s.col_range_for_row(7), 5..8);
    }

    #[test]
    fn row_value_returns_zero_outside_band() {
        let mut a = Banded::<f64>::zeros(5, 1, 1).unwrap();
        a[(0, 0)] = 10.0;
        a[(1, 0)] = 20.0;
        a[(0, 1)] = 11.0;
        a[(1, 1)] = 21.0;
        a[(2, 1)] = 31.0;

        let mut s = BandedLuStorage::new(5, 1, 1).unwrap();
        s.copy_from_compact(&a).unwrap();

        assert_eq!(s.row_value(0, 0), 10.0);
        assert_eq!(s.row_value(1, 0), 20.0);
        assert_eq!(s.row_value(4, 0), 0.0);
        assert_eq!(s.row_value(0, 4), 0.0);
    }

    #[test]
    fn swap_rows_fragment_swaps_visible_values() {
        let mut a = Banded::<f64>::zeros(6, 1, 1).unwrap();

        // Dense picture inside original compact band:
        // [10 11  0  0  0  0]
        // [20 21 22  0  0  0]
        // [ 0 31 32 33  0  0]
        // [ 0  0 42 43 44  0]
        // [ 0  0  0 53 54 55]
        // [ 0  0  0  0 64 65]

        a[(0, 0)] = 10.0;
        a[(0, 1)] = 11.0;

        a[(1, 0)] = 20.0;
        a[(1, 1)] = 21.0;
        a[(1, 2)] = 22.0;

        a[(2, 1)] = 31.0;
        a[(2, 2)] = 32.0;
        a[(2, 3)] = 33.0;

        a[(3, 2)] = 42.0;
        a[(3, 3)] = 43.0;
        a[(3, 4)] = 44.0;

        a[(4, 3)] = 53.0;
        a[(4, 4)] = 54.0;
        a[(4, 5)] = 55.0;

        a[(5, 4)] = 64.0;
        a[(5, 5)] = 65.0;

        let mut s = BandedLuStorage::new(6, 1, 1).unwrap();
        s.copy_from_compact(&a).unwrap();

        let before = dense_from_storage(&s);

        // Swap logical rows 1 and 2 only over columns [1, 4)
        s.swap_rows_fragment(1, 2, 1, 4).unwrap();

        let after = dense_from_storage(&s);

        // Outside fragment, rows remain unchanged
        assert_eq!(after[1][0], before[1][0]);
        assert_eq!(after[2][0], before[2][0]);
        assert_eq!(after[1][4], before[1][4]);
        assert_eq!(after[2][4], before[2][4]);

        // Inside fragment, values are swapped logically
        for j in 1..4 {
            assert!(
                (after[1][j] - before[2][j]).abs() < 1e-14,
                "row 1, col {j}: got {}, expected {}",
                after[1][j],
                before[2][j]
            );
            assert!(
                (after[2][j] - before[1][j]).abs() < 1e-14,
                "row 2, col {j}: got {}, expected {}",
                after[2][j],
                before[1][j]
            );
        }
    }

    #[test]
    fn swap_rows_fragment_handles_implicit_zeros_correctly() {
        let mut s = BandedLuStorage::new(6, 2, 1).unwrap();

        // ext_ku = 3, rows are fairly wide.
        // Put values so that one row has a stored entry where the other row has logical zero.
        s.set(1, 1, 100.0).unwrap();
        s.set(1, 2, 101.0).unwrap();
        s.set(3, 2, 200.0).unwrap();
        s.set(3, 3, 201.0).unwrap();

        let before = s.to_dense();

        s.swap_rows_fragment(1, 3, 1, 4).unwrap();

        let after = s.to_dense();

        for j in 1..4 {
            assert!(
                (after[1][j] - before[3][j]).abs() < 1e-14,
                "row 1, col {j}: got {}, expected {}",
                after[1][j],
                before[3][j]
            );
            assert!(
                (after[3][j] - before[1][j]).abs() < 1e-14,
                "row 3, col {j}: got {}, expected {}",
                after[3][j],
                before[1][j]
            );
        }
    }

    #[test]
    fn swap_rows_fragment_noop_for_same_row() {
        let mut s = BandedLuStorage::new(5, 1, 1).unwrap();
        s.set(1, 1, 3.0).unwrap();
        s.set(1, 2, 4.0).unwrap();

        let before = s.to_dense();
        s.swap_rows_fragment(1, 1, 0, 5).unwrap();
        let after = s.to_dense();

        assert_eq!(before, after);
    }

    #[test]
    fn swap_rows_fragment_empty_interval_is_noop() {
        let mut s = BandedLuStorage::new(5, 1, 1).unwrap();
        s.set(1, 1, 3.0).unwrap();
        s.set(2, 2, 4.0).unwrap();

        let before = s.to_dense();
        s.swap_rows_fragment(1, 2, 3, 3).unwrap();
        let after = s.to_dense();

        assert_eq!(before, after);
    }

    #[test]
    fn partial_pivot_multiple_rhs() {
        let mut a = Banded::<f64>::zeros(4, 1, 1).unwrap();

        a[(0, 0)] = 0.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 3.0;
        a[(1, 2)] = 4.0;
        a[(2, 1)] = 5.0;
        a[(2, 2)] = 6.0;
        a[(2, 3)] = 1.0;
        a[(3, 2)] = 2.0;
        a[(3, 3)] = 7.0;

        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let x2 = vec![0.5, -1.0, 2.0, 1.0];

        let b1 = banded_matvec(&a, &x1).unwrap();
        let b2 = banded_matvec(&a, &x2).unwrap();

        // column-major layout: [b1 | b2]
        let mut rhs = vec![b1[0], b1[1], b1[2], b1[3], b2[0], b2[1], b2[2], b2[3]];

        let mut lu = GeneralBandedLuPartialPivot::new(4, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();

        lu.solve_multiple_in_place(&mut rhs, 2, 4).unwrap();

        let sol1 = &rhs[0..4];
        let sol2 = &rhs[4..8];

        for i in 0..4 {
            assert!((sol1[i] - x1[i]).abs() < 1e-10);
            assert!((sol2[i] - x2[i]).abs() < 1e-10);
        }
    }
}
