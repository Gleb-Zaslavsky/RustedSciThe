use super::core::{LinearBackend, OdeError};

/// =========================
/// Dense matrix storage
/// Column-major storage:
/// a[j * n + i] = A_{i,j}
/// =========================

#[derive(Debug, Clone)]
pub struct DenseMatrixStorage {
    pub n: usize,
    pub jac: Vec<f64>,
    pub system: Vec<f64>,
}

impl DenseMatrixStorage {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            jac: vec![0.0; n * n],
            system: vec![0.0; n * n],
        }
    }

    #[inline(always)]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.n);
        debug_assert!(j < self.n);
        j * self.n + i
    }

    #[inline(always)]
    pub fn jac_col_mut(&mut self, j: usize) -> &mut [f64] {
        let start = j * self.n;
        &mut self.jac[start..start + self.n]
    }

    #[inline(always)]
    pub fn zero_jac(&mut self) {
        self.jac.fill(0.0);
    }

    #[inline(always)]
    pub fn zero_system(&mut self) {
        self.system.fill(0.0);
    }
}

/// =========================
/// Dense LU backend
/// Naive reference implementation.
/// Replace later with faer / LAPACK / tuned LU.
/// =========================

pub struct DenseBackend {
    storage: DenseMatrixStorage,
    pivots: Vec<usize>,
}

impl DenseBackend {
    pub fn new(n: usize) -> Self {
        Self {
            storage: DenseMatrixStorage::new(n),
            pivots: (0..n).collect(),
        }
    }

    #[inline(always)]
    fn n(&self) -> usize {
        self.storage.n
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize {
        self.storage.idx(i, j)
    }

    fn swap_rows_system(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        let n = self.n();
        for j in 0..n {
            let a = self.idx(r1, j);
            let b = self.idx(r2, j);
            self.storage.system.swap(a, b);
        }
    }
}

impl LinearBackend for DenseBackend {
    type Storage = DenseMatrixStorage;

    #[inline(always)]
    fn storage(&self) -> &Self::Storage {
        &self.storage
    }

    #[inline(always)]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        &mut self.storage
    }

    fn assemble_system_matrix(&mut self, gamma: f64) {
        let n = self.n();
        self.storage.system.copy_from_slice(&self.storage.jac);

        // system = I - gamma * J
        for j in 0..n {
            for i in 0..n {
                let idx = self.idx(i, j);
                self.storage.system[idx] *= -gamma;
            }
        }
        for i in 0..n {
            let idx = self.idx(i, i);
            self.storage.system[idx] += 1.0;
        }
    }

    fn factorize(&mut self) -> Result<(), OdeError> {
        let n = self.n();
        self.pivots.clear();
        self.pivots.resize(n, 0);

        for k in 0..n {
            let mut p = k;
            let mut max_val = self.storage.system[self.idx(k, k)].abs();
            for i in (k + 1)..n {
                let val = self.storage.system[self.idx(i, k)].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                }
            }

            if max_val == 0.0 {
                return Err(OdeError::SingularJacobian);
            }

            self.pivots[k] = p;

            if p != k {
                self.swap_rows_system(k, p);
            }

            let akk = self.storage.system[self.idx(k, k)];
            for i in (k + 1)..n {
                let ik = self.idx(i, k);
                self.storage.system[ik] /= akk;
                let lik = self.storage.system[ik];

                for j in (k + 1)..n {
                    let ij = self.idx(i, j);
                    let kj = self.idx(k, j);
                    self.storage.system[ij] -= lik * self.storage.system[kj];
                }
            }
        }

        Ok(())
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), OdeError> {
        let n = self.n();
        if rhs.len() != n {
            return Err(OdeError::IllegalInput(
                "rhs length mismatch in linear solve",
            ));
        }

        // Apply row permutations to rhs
        for k in 0..n {
            let p = self.pivots[k];
            if p != k {
                rhs.swap(k, p);
            }
        }
        // Forward solve Ly = Pb
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= self.storage.system[self.idx(i, j)] * rhs[j];
            }
            rhs[i] = sum;
        }

        // Backward solve Ux = y
        for ii in 0..n {
            let i = n - 1 - ii;
            let mut sum = rhs[i];
            for j in (i + 1)..n {
                sum -= self.storage.system[self.idx(i, j)] * rhs[j];
            }
            let uii = self.storage.system[self.idx(i, i)];
            if uii == 0.0 {
                return Err(OdeError::SingularJacobian);
            }
            rhs[i] = sum / uii;
        }

        Ok(())
    }
}
