use super::error::BandedError;
use faer::sparse::SparseColMat;
use std::ops::{Index, IndexMut};

/// Compact banded storage:
/// data[(ku + i - j) * n + j] = A[i, j]
///
/// Shape in memory is effectively (kl + ku + 1, n), row-major by band row.
/// This is convenient for assembly and easy to convert into LAPACK GB format.
#[derive(Clone, Debug)]
pub struct Banded<T> {
    n: usize,
    kl: usize,
    ku: usize,
    data: Vec<T>,
}

impl<T: Clone + Default> Banded<T> {
    pub fn zeros(n: usize, kl: usize, ku: usize) -> Result<Self, BandedError> {
        if n == 0 {
            return Ok(Self {
                n,
                kl,
                ku,
                data: Vec::new(),
            });
        }

        let rows = kl
            .checked_add(ku)
            .and_then(|x| x.checked_add(1))
            .ok_or(BandedError::InvalidBand { n, kl, ku })?;

        let len = rows
            .checked_mul(n)
            .ok_or(BandedError::InvalidBand { n, kl, ku })?;

        Ok(Self {
            n,
            kl,
            ku,
            data: vec![T::default(); len],
        })
    }
}

impl<T> Banded<T> {
    pub fn from_vec(n: usize, kl: usize, ku: usize, data: Vec<T>) -> Result<Self, BandedError> {
        let rows = kl
            .checked_add(ku)
            .and_then(|x| x.checked_add(1))
            .ok_or(BandedError::InvalidBand { n, kl, ku })?;

        if data.len() != rows.saturating_mul(n) {
            return Err(BandedError::DimensionMismatch);
        }

        Ok(Self { n, kl, ku, data })
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
    pub fn rows(&self) -> usize {
        self.kl + self.ku + 1
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn in_band(&self, i: usize, j: usize) -> bool {
        i < self.n && j < self.n && i + self.ku >= j && j + self.kl >= i
    }

    #[inline]
    pub fn offset(&self, i: usize, j: usize) -> Option<usize> {
        if !self.in_band(i, j) {
            return None;
        }
        let band_row = self.ku + i - j;
        Some(band_row * self.n + j)
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        self.offset(i, j).map(|idx| &self.data[idx])
    }

    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        self.offset(i, j).map(|idx| &mut self.data[idx])
    }

    pub fn set(&mut self, i: usize, j: usize, value: T) -> Result<(), BandedError> {
        let idx = self.offset(i, j).ok_or(BandedError::OutOfBounds { i, j })?;
        self.data[idx] = value;
        Ok(())
    }

    pub fn fill_from_dense<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, usize) -> T,
    {
        for j in 0..self.n {
            let i0 = j.saturating_sub(self.ku);
            let i1 = (j + self.kl + 1).min(self.n);
            for i in i0..i1 {
                let idx = self.offset(i, j).expect("index must be in band");
                self.data[idx] = f(i, j);
            }
        }
    }
}

impl<T> Index<(usize, usize)> for Banded<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        let idx = self
            .offset(i, j)
            .unwrap_or_else(|| panic!("index ({i}, {j}) is outside the band or matrix bounds"));
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for Banded<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        let idx = self
            .offset(i, j)
            .unwrap_or_else(|| panic!("index ({i}, {j}) is outside the band or matrix bounds"));
        &mut self.data[idx]
    }
}
/*
impl Banded<f64> {
    pub fn to_faer_csc(&self) -> SparseColMat<f64> {
        let n = self.n();

        let mut col_ptrs = Vec::with_capacity(n + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        col_ptrs.push(0);

        for j in 0..n {
            let i0 = j.saturating_sub(self.ku());
            let i1 = (j + self.kl() + 1).min(n);

            for i in i0..i1 {
                let v = self[(i, j)];
                if v != 0.0 {
                    row_indices.push(i);
                    values.push(v);
                }
            }

            col_ptrs.push(row_indices.len());
        }

        SparseColMat::new(n, n, col_ptrs, row_indices, values)
    }
}
    */
