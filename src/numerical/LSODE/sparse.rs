use super::core::{LinearBackend, OdeError};
use super::dense::DenseBackend;

/// =========================
/// Sparse matrix storage (CSC)
///
/// Pattern is fixed:
/// - col_ptrs has length n + 1
/// - row_indices has length nnz
/// - row indices in each column must be sorted
/// - each diagonal entry must be present
///
/// Values are updated in-place:
/// - jac_values
/// - system_values = I - gamma * J, same pattern
/// =========================

#[derive(Debug, Clone)]
pub struct SparseMatrixStorage {
    pub n: usize,

    /// CSC pattern
    pub col_ptrs: Vec<usize>,
    pub row_indices: Vec<usize>,

    /// Jacobian values in CSC nonzero order
    pub jac_values: Vec<f64>,

    /// System values in CSC nonzero order
    pub system_values: Vec<f64>,

    /// diagonal position for each column
    pub diag_pos: Vec<usize>,
}

impl SparseMatrixStorage {
    pub fn new(n: usize, col_ptrs: Vec<usize>, row_indices: Vec<usize>) -> Result<Self, OdeError> {
        if col_ptrs.len() != n + 1 {
            return Err(OdeError::IllegalInput(
                "sparse col_ptrs must have length n + 1",
            ));
        }
        if col_ptrs[0] != 0 {
            return Err(OdeError::IllegalInput("sparse col_ptrs must start at 0"));
        }
        if *col_ptrs.last().unwrap() != row_indices.len() {
            return Err(OdeError::IllegalInput(
                "sparse col_ptrs শেষ element must equal row_indices.len()",
            ));
        }

        for j in 0..n {
            if col_ptrs[j] > col_ptrs[j + 1] {
                return Err(OdeError::IllegalInput(
                    "sparse col_ptrs must be nondecreasing",
                ));
            }
        }

        let mut diag_pos = vec![usize::MAX; n];

        for j in 0..n {
            let start = col_ptrs[j];
            let end = col_ptrs[j + 1];

            let mut prev_row = None;
            let mut found_diag = false;

            for p in start..end {
                let i = row_indices[p];
                if i >= n {
                    return Err(OdeError::IllegalInput("sparse row index out of bounds"));
                }

                if let Some(prev) = prev_row {
                    if i <= prev {
                        return Err(OdeError::IllegalInput(
                            "sparse row indices in each column must be strictly increasing",
                        ));
                    }
                }
                prev_row = Some(i);

                if i == j {
                    diag_pos[j] = p;
                    found_diag = true;
                }
            }

            if !found_diag {
                return Err(OdeError::IllegalInput(
                    "sparse pattern must explicitly contain all diagonal entries",
                ));
            }
        }

        let nnz = row_indices.len();

        Ok(Self {
            n,
            col_ptrs,
            row_indices,
            jac_values: vec![0.0; nnz],
            system_values: vec![0.0; nnz],
            diag_pos,
        })
    }

    #[inline(always)]
    pub fn nnz(&self) -> usize {
        self.row_indices.len()
    }

    #[inline(always)]
    pub fn zero_jac(&mut self) {
        self.jac_values.fill(0.0);
    }

    #[inline(always)]
    pub fn zero_system(&mut self) {
        self.system_values.fill(0.0);
    }

    /// Returns the CSC slot of (i, j) if present in the fixed pattern.
    /// Binary search inside column since rows are sorted.
    #[inline]
    pub fn find_pos(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.n || j >= self.n {
            return None;
        }

        let start = self.col_ptrs[j];
        let end = self.col_ptrs[j + 1];
        match self.row_indices[start..end].binary_search(&i) {
            Ok(k) => Some(start + k),
            Err(_) => None,
        }
    }

    #[inline(always)]
    pub fn set_jac(&mut self, i: usize, j: usize, val: f64) {
        if let Some(p) = self.find_pos(i, j) {
            self.jac_values[p] = val;
        }
    }

    #[inline(always)]
    pub fn add_jac(&mut self, i: usize, j: usize, val: f64) {
        if let Some(p) = self.find_pos(i, j) {
            self.jac_values[p] += val;
        }
    }

    #[inline(always)]
    pub fn get_jac(&self, i: usize, j: usize) -> f64 {
        match self.find_pos(i, j) {
            Some(p) => self.jac_values[p],
            None => 0.0,
        }
    }

    #[inline(always)]
    pub fn set_system(&mut self, i: usize, j: usize, val: f64) {
        if let Some(p) = self.find_pos(i, j) {
            self.system_values[p] = val;
        }
    }

    #[inline(always)]
    pub fn get_system(&self, i: usize, j: usize) -> f64 {
        match self.find_pos(i, j) {
            Some(p) => self.system_values[p],
            None => 0.0,
        }
    }
}

/// =========================
/// Reference sparse backend
///
/// Important:
/// - CSC assembly is truly sparse
/// - factor/solve is delegated to DenseBackend for correctness-first development
///
/// Later:
/// - replace dense_shadow with real sparse factorization path (faer)
/// =========================

pub struct SparseBackend {
    storage: SparseMatrixStorage,
    dense_shadow: DenseBackend,
}

impl SparseBackend {
    pub fn new(storage: SparseMatrixStorage) -> Self {
        let n = storage.n;
        Self {
            storage,
            dense_shadow: DenseBackend::new(n),
        }
    }

    fn copy_system_to_dense_shadow(&mut self) {
        let n = self.storage.n;
        let dense = self.dense_shadow.storage_mut();
        dense.zero_jac();
        dense.zero_system();

        for j in 0..n {
            let start = self.storage.col_ptrs[j];
            let end = self.storage.col_ptrs[j + 1];

            for p in start..end {
                let i = self.storage.row_indices[p];
                let kd = dense.idx(i, j);
                dense.system[kd] = self.storage.system_values[p];
            }
        }
    }
}

impl LinearBackend for SparseBackend {
    type Storage = SparseMatrixStorage;

    #[inline(always)]
    fn storage(&self) -> &Self::Storage {
        &self.storage
    }

    #[inline(always)]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        &mut self.storage
    }

    fn assemble_system_matrix(&mut self, gamma: f64) {
        self.storage
            .system_values
            .copy_from_slice(&self.storage.jac_values);

        for v in &mut self.storage.system_values {
            *v *= -gamma;
        }

        for &p in &self.storage.diag_pos {
            self.storage.system_values[p] += 1.0;
        }
    }

    fn factorize(&mut self) -> Result<(), OdeError> {
        self.copy_system_to_dense_shadow();
        self.dense_shadow.factorize()
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), OdeError> {
        self.dense_shadow.solve_in_place(rhs)
    }
}
