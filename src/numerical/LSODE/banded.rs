use super::core::{LinearBackend, OdeError};

/// =========================
/// Banded matrix storage
///
/// Compact storage for Jacobian/system:
///   offset = j * ldab + (mu + i - j)
///   ldab   = ml + mu + 1
///
/// LU workspace storage:
///   offset = j * ldab_lu + (ml + mu + i - j)
///   ldab_lu = 2*ml + mu + 1
///
/// The extra ml rows in LU storage are needed for fill-in / multipliers
/// during band Gaussian elimination.
/// =========================

#[derive(Debug, Clone)]
pub struct BandedMatrixStorage {
    pub n: usize,
    pub ml: usize,
    pub mu: usize,

    /// compact band storage for Jacobian/system
    pub ldab: usize,
    pub jac: Vec<f64>,
    pub system: Vec<f64>,

    /// LU workspace
    pub ldab_lu: usize,
    pub lu: Vec<f64>,
    pub pivots: Vec<usize>,
}

impl BandedMatrixStorage {
    pub fn new(n: usize, ml: usize, mu: usize) -> Self {
        let ldab = ml + mu + 1;
        let ldab_lu = 2 * ml + mu + 1;

        Self {
            n,
            ml,
            mu,
            ldab,
            jac: vec![0.0; ldab * n],
            system: vec![0.0; ldab * n],
            ldab_lu,
            lu: vec![0.0; ldab_lu * n],
            pivots: vec![0; n],
        }
    }

    #[inline(always)]
    pub fn in_band(&self, i: usize, j: usize) -> bool {
        if i >= self.n || j >= self.n {
            return false;
        }
        i <= j + self.ml && j <= i + self.mu
    }

    #[inline(always)]
    pub fn offset(&self, i: usize, j: usize) -> Option<usize> {
        if !self.in_band(i, j) {
            return None;
        }
        let row = self.mu as isize + i as isize - j as isize;
        debug_assert!(row >= 0);
        debug_assert!((row as usize) < self.ldab);
        Some(j * self.ldab + row as usize)
    }

    /// Offset into LU workspace.
    ///
    /// This allows a wider lower region than compact storage.
    #[inline(always)]
    pub fn offset_lu(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.n || j >= self.n {
            return None;
        }

        // In LU workspace, after elimination U can extend to mu + ml above diagonal,
        // and L can extend to ml below diagonal.
        if i + (self.mu + self.ml) < j {
            return None;
        }
        if i > j + self.ml {
            return None;
        }

        let row = (self.ml + self.mu) as isize + i as isize - j as isize;
        debug_assert!(row >= 0);
        debug_assert!((row as usize) < self.ldab_lu);
        Some(j * self.ldab_lu + row as usize)
    }

    #[inline(always)]
    pub fn get_jac(&self, i: usize, j: usize) -> f64 {
        match self.offset(i, j) {
            Some(k) => self.jac[k],
            None => 0.0,
        }
    }

    #[inline(always)]
    pub fn set_jac(&mut self, i: usize, j: usize, val: f64) {
        if let Some(k) = self.offset(i, j) {
            self.jac[k] = val;
        }
    }

    #[inline(always)]
    pub fn get_system(&self, i: usize, j: usize) -> f64 {
        match self.offset(i, j) {
            Some(k) => self.system[k],
            None => 0.0,
        }
    }

    #[inline(always)]
    pub fn set_system(&mut self, i: usize, j: usize, val: f64) {
        if let Some(k) = self.offset(i, j) {
            self.system[k] = val;
        }
    }

    #[inline(always)]
    pub fn get_lu(&self, i: usize, j: usize) -> f64 {
        match self.offset_lu(i, j) {
            Some(k) => self.lu[k],
            None => 0.0,
        }
    }

    #[inline(always)]
    pub fn set_lu(&mut self, i: usize, j: usize, val: f64) {
        if let Some(k) = self.offset_lu(i, j) {
            self.lu[k] = val;
        }
    }

    #[inline(always)]
    pub fn zero_jac(&mut self) {
        self.jac.fill(0.0);
    }

    #[inline(always)]
    pub fn zero_system(&mut self) {
        self.system.fill(0.0);
    }

    #[inline(always)]
    pub fn zero_lu(&mut self) {
        self.lu.fill(0.0);
    }
}

/// =========================
/// Real banded backend
/// =========================

pub struct BandedBackend {
    storage: BandedMatrixStorage,
}

impl BandedBackend {
    pub fn new(n: usize, ml: usize, mu: usize) -> Self {
        Self {
            storage: BandedMatrixStorage::new(n, ml, mu),
        }
    }

    #[inline(always)]
    fn n(&self) -> usize {
        self.storage.n
    }

    fn copy_system_to_lu(&mut self) {
        self.storage.zero_lu();

        let n = self.storage.n;
        for j in 0..n {
            let i_min = j.saturating_sub(self.storage.mu);
            let i_max = (j + self.storage.ml + 1).min(n);

            for i in i_min..i_max {
                if let Some(ks) = self.storage.offset(i, j) {
                    if let Some(klu) = self.storage.offset_lu(i, j) {
                        self.storage.lu[klu] = self.storage.system[ks];
                    }
                }
            }
        }
    }

    fn swap_rows_lu_partial(
        &mut self,
        r1: usize,
        r2: usize,
        j_start: usize,
        j_end_exclusive: usize,
    ) {
        if r1 == r2 {
            return;
        }

        for j in j_start..j_end_exclusive {
            let a = self.storage.offset_lu(r1, j);
            let b = self.storage.offset_lu(r2, j);

            match (a, b) {
                (Some(ka), Some(kb)) => {
                    self.storage.lu.swap(ka, kb);
                }
                (Some(ka), None) => {
                    let tmp = self.storage.lu[ka];
                    self.storage.lu[ka] = 0.0;
                    if let Some(kb) = self.storage.offset_lu(r2, j) {
                        self.storage.lu[kb] = tmp;
                    }
                }
                (None, Some(kb)) => {
                    let tmp = self.storage.lu[kb];
                    self.storage.lu[kb] = 0.0;
                    if let Some(ka) = self.storage.offset_lu(r1, j) {
                        self.storage.lu[ka] = tmp;
                    }
                }
                (None, None) => {}
            }
        }
    }

    fn lu_factorize_in_place(&mut self) -> Result<(), OdeError> {
        let n = self.storage.n;
        let ml = self.storage.ml;
        let mu = self.storage.mu;

        self.storage.pivots.fill(0);

        for k in 0..n {
            // Pivot search only within lower bandwidth
            let i_max_pivot = (k + ml).min(n - 1);
            let mut p = k;
            let mut max_val = self.storage.get_lu(k, k).abs();

            for i in (k + 1)..=i_max_pivot {
                let val = self.storage.get_lu(i, k).abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                }
            }

            if max_val == 0.0 {
                return Err(OdeError::SingularJacobian);
            }

            self.storage.pivots[k] = p;

            if p != k {
                // Only swap rows over the active local column window
                let j_end = (k + mu + ml + 1).min(n);
                self.swap_rows_lu_partial(k, p, k, j_end);
            }

            let akk = self.storage.get_lu(k, k);
            if akk == 0.0 {
                return Err(OdeError::SingularJacobian);
            }

            // Elimination below pivot within lower bandwidth
            let i_max = (k + ml).min(n - 1);
            for i in (k + 1)..=i_max {
                let aik = self.storage.get_lu(i, k);
                if aik == 0.0 {
                    continue;
                }

                let lik = aik / akk;
                self.storage.set_lu(i, k, lik);

                // Update trailing band block
                let j_max = (k + mu + ml).min(n - 1);
                for j in (k + 1)..=j_max {
                    let aij = self.storage.get_lu(i, j);
                    let akj = self.storage.get_lu(k, j);

                    // If both are structurally zero in workspace, skip
                    if aij == 0.0 && akj == 0.0 {
                        continue;
                    }

                    let new_val = aij - lik * akj;
                    self.storage.set_lu(i, j, new_val);
                }
            }
        }

        Ok(())
    }

    fn apply_pivots_to_rhs(&self, rhs: &mut [f64]) {
        for k in 0..self.storage.n {
            let p = self.storage.pivots[k];
            if p != k {
                rhs.swap(k, p);
            }
        }
    }

    fn forward_substitute(&self, rhs: &mut [f64]) {
        let n = self.storage.n;
        let ml = self.storage.ml;

        // L has unit diagonal and is stored below diagonal in LU
        for i in 0..n {
            let j_min = i.saturating_sub(ml);
            let mut sum = rhs[i];

            for j in j_min..i {
                sum -= self.storage.get_lu(i, j) * rhs[j];
            }

            rhs[i] = sum;
        }
    }

    fn backward_substitute(&self, rhs: &mut [f64]) -> Result<(), OdeError> {
        let n = self.storage.n;
        let upper_bw = self.storage.mu + self.storage.ml;

        for ii in 0..n {
            let i = n - 1 - ii;
            let j_max = (i + upper_bw).min(n - 1);

            let mut sum = rhs[i];
            for j in (i + 1)..=j_max {
                sum -= self.storage.get_lu(i, j) * rhs[j];
            }

            let uii = self.storage.get_lu(i, i);
            if uii == 0.0 {
                return Err(OdeError::SingularJacobian);
            }

            rhs[i] = sum / uii;
        }

        Ok(())
    }
}

impl LinearBackend for BandedBackend {
    type Storage = BandedMatrixStorage;

    #[inline(always)]
    fn storage(&self) -> &Self::Storage {
        &self.storage
    }

    #[inline(always)]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        &mut self.storage
    }

    fn assemble_system_matrix(&mut self, gamma: f64) {
        self.storage.system.copy_from_slice(&self.storage.jac);

        let n = self.n();
        for j in 0..n {
            let i_min = j.saturating_sub(self.storage.mu);
            let i_max = (j + self.storage.ml + 1).min(n);

            for i in i_min..i_max {
                if let Some(k) = self.storage.offset(i, j) {
                    self.storage.system[k] *= -gamma;
                }
            }

            if let Some(kd) = self.storage.offset(j, j) {
                self.storage.system[kd] += 1.0;
            }
        }
    }

    fn factorize(&mut self) -> Result<(), OdeError> {
        self.copy_system_to_lu();
        self.lu_factorize_in_place()
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), OdeError> {
        if rhs.len() != self.storage.n {
            return Err(OdeError::IllegalInput(
                "rhs length mismatch in banded linear solve",
            ));
        }

        self.apply_pivots_to_rhs(rhs);
        self.forward_substitute(rhs);
        self.backward_substitute(rhs)
    }
}
