//! Dynamic SuperLU control used only by the isolated sparse-LU experiments.
//!
//! This is deliberately not wired into BVP_sci's production backend enum. The
//! native bridge must be built separately from the pinned SciPy-compatible
//! SuperLU source; set `SUPERLU_LIBRARY` to its shared-library path.

use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use libloading::Library;
use thiserror::Error;

type VersionFn = unsafe extern "C" fn() -> i32;
type SolveFn = unsafe extern "C" fn(
    i32,
    i32,
    *const i32,
    *const i32,
    *const f64,
    *const f64,
    *mut f64,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut u64,
    *mut u64,
) -> i32;
type FactorFn = unsafe extern "C" fn(
    i32,
    i32,
    *const i32,
    *const i32,
    *const f64,
    *mut *mut std::ffi::c_void,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut u64,
) -> i32;
type FactoredSolveFn = unsafe extern "C" fn(*mut std::ffi::c_void, *mut f64, *mut u64) -> i32;
type DestroyFactorFn = unsafe extern "C" fn(*mut std::ffi::c_void);

#[derive(Debug, Error)]
pub enum SuperLuExperimentError {
    #[error("SuperLU bridge path is not configured: {0}")]
    MissingLibraryPath(&'static str),
    #[error("failed to load SuperLU experiment library: {0}")]
    Load(#[from] libloading::Error),
    #[error("invalid CSC input: {0}")]
    InvalidCsc(&'static str),
    #[error("matrix dimension or nnz does not fit SuperLU's 32-bit index ABI")]
    IndexOverflow,
    #[error("SuperLU bridge returned internal status {0}")]
    Bridge(i32),
    #[error("bridge reports an unexpected SuperLU version: encoded version {0}")]
    VersionMismatch(i32),
    #[error("SuperLU reported factorization info={0}")]
    Factorization(i32),
}

/// CSC view converted once to the 32-bit ABI used by SciPy's SuperLU build.
#[derive(Debug, Clone)]
pub struct SuperLuCsc32 {
    n: i32,
    col_ptr: Vec<i32>,
    row_idx: Vec<i32>,
    values: Vec<f64>,
}

impl SuperLuCsc32 {
    /// Validates and converts a canonical CSC matrix without changing its data.
    pub fn try_from_usize(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
    ) -> Result<Self, SuperLuExperimentError> {
        let expected_column_pointer_count = n
            .checked_add(1)
            .ok_or(SuperLuExperimentError::IndexOverflow)?;
        if n == 0 || col_ptr.len() != expected_column_pointer_count {
            return Err(SuperLuExperimentError::InvalidCsc(
                "non-empty square matrix requires n+1 column pointers",
            ));
        }
        if row_idx.len() != values.len() || col_ptr.first() != Some(&0) {
            return Err(SuperLuExperimentError::InvalidCsc(
                "row indices, values, and column pointers have inconsistent lengths",
            ));
        }
        if col_ptr.last() != Some(&values.len())
            || col_ptr.windows(2).any(|pair| pair[0] > pair[1])
            || row_idx.iter().any(|&row| row >= n)
        {
            return Err(SuperLuExperimentError::InvalidCsc(
                "CSC pointers must be monotonic and terminate at nnz; rows must be in range",
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SuperLuExperimentError::InvalidCsc(
                "matrix values must all be finite",
            ));
        }
        for column in 0..n {
            let start = col_ptr[column];
            let end = col_ptr[column + 1];
            if row_idx[start..end]
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(SuperLuExperimentError::InvalidCsc(
                    "row indices must be strictly increasing in each CSC column",
                ));
            }
        }

        let n32 = i32::try_from(n).map_err(|_| SuperLuExperimentError::IndexOverflow)?;
        let nnz32 =
            i32::try_from(values.len()).map_err(|_| SuperLuExperimentError::IndexOverflow)?;
        let col_ptr = col_ptr
            .iter()
            .map(|&index| i32::try_from(index).map_err(|_| SuperLuExperimentError::IndexOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let row_idx = row_idx
            .iter()
            .map(|&index| i32::try_from(index).map_err(|_| SuperLuExperimentError::IndexOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        debug_assert_eq!(row_idx.len(), nnz32 as usize);

        Ok(Self {
            n: n32,
            col_ptr,
            row_idx,
            values: values.to_vec(),
        })
    }

    pub fn dimension(&self) -> usize {
        self.n as usize
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    fn parts(&self) -> (&[i32], &[i32], &[f64]) {
        (&self.col_ptr, &self.row_idx, &self.values)
    }
}

/// Owns the loaded DLL and its stable bridge function pointers.
pub struct SuperLuControl {
    _library: Arc<Library>,
    solve: SolveFn,
    factorize: FactorFn,
    solve_factored: FactoredSolveFn,
    destroy_factor: DestroyFactorFn,
    version: i32,
}

impl SuperLuControl {
    /// Loads a separately built bridge and verifies that it is SuperLU 6.0.1.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SuperLuExperimentError> {
        // SAFETY: the bridge exports the documented C ABI and remains loaded for
        // as long as this owner, including the copied function pointers, lives.
        let library = Arc::new(unsafe { Library::new(path.as_ref())? });
        let (version_fn, solve) = unsafe {
            let version_fn = *library.get::<VersionFn>(b"rs_superlu_version\0")?;
            let solve = *library.get::<SolveFn>(b"rs_superlu_solve_csc\0")?;
            (version_fn, solve)
        };
        let (factorize, solve_factored, destroy_factor) = unsafe {
            (
                *library.get::<FactorFn>(b"rs_superlu_factor_csc\0")?,
                *library.get::<FactoredSolveFn>(b"rs_superlu_solve_factored\0")?,
                *library.get::<DestroyFactorFn>(b"rs_superlu_destroy_factor\0")?,
            )
        };
        // SAFETY: the version function takes no arguments and returns an integer.
        let version = unsafe { version_fn() };
        if version != 60_001 {
            return Err(SuperLuExperimentError::VersionMismatch(version));
        }
        Ok(Self {
            _library: library,
            solve,
            factorize,
            solve_factored,
            destroy_factor,
            version,
        })
    }

    pub fn version(&self) -> (i32, i32, i32) {
        (
            self.version / 10_000,
            (self.version / 100) % 100,
            self.version % 100,
        )
    }

    /// Runs one fresh factorization and one solve; `rhs` is never modified.
    pub fn factor_and_solve(
        &self,
        matrix: &SuperLuCsc32,
        rhs: &[f64],
    ) -> Result<SuperLuSolve, SuperLuExperimentError> {
        if rhs.len() != matrix.dimension() {
            return Err(SuperLuExperimentError::InvalidCsc(
                "RHS length must match the square matrix dimension",
            ));
        }
        let mut solution = rhs.to_vec();
        let mut factor_l_storage_nnz = 0;
        let mut factor_u_storage_nnz = 0;
        let mut factor_l_csc_nnz = 0;
        let mut factor_u_csc_nnz = 0;
        let mut solver_info = 0;
        let mut factor_ns = 0;
        let mut solve_ns = 0;
        let started = Instant::now();
        // SAFETY: all slices are valid for their lengths, indices were checked
        // and converted to SuperLU's int32 ABI, and the bridge borrows inputs.
        let bridge_status = unsafe {
            (self.solve)(
                matrix.n,
                i32::try_from(matrix.nnz()).map_err(|_| SuperLuExperimentError::IndexOverflow)?,
                matrix.col_ptr.as_ptr(),
                matrix.row_idx.as_ptr(),
                matrix.values.as_ptr(),
                rhs.as_ptr(),
                solution.as_mut_ptr(),
                &mut factor_l_storage_nnz,
                &mut factor_u_storage_nnz,
                &mut factor_l_csc_nnz,
                &mut factor_u_csc_nnz,
                &mut solver_info,
                &mut factor_ns,
                &mut solve_ns,
            )
        };
        let ffi_elapsed = started.elapsed();
        if bridge_status != 0 {
            return Err(SuperLuExperimentError::Bridge(bridge_status));
        }
        if solver_info != 0 {
            return Err(SuperLuExperimentError::Factorization(solver_info));
        }

        Ok(SuperLuSolve {
            solution,
            factor_l_storage_nnz: factor_l_storage_nnz as usize,
            factor_u_storage_nnz: factor_u_storage_nnz as usize,
            factor_l_csc_nnz: factor_l_csc_nnz as usize,
            factor_u_csc_nnz: factor_u_csc_nnz as usize,
            ffi_elapsed,
            factor_elapsed: Duration::from_nanos(factor_ns),
            solve_elapsed: Duration::from_nanos(solve_ns),
        })
    }

    /// Factor one CSC matrix and retain native factors for repeated RHS solves.
    pub fn factorize(
        &self,
        matrix: &SuperLuCsc32,
    ) -> Result<SuperLuFactor, SuperLuExperimentError> {
        let (col_ptr, row_idx, values) = matrix.parts();
        let mut raw_handle = std::ptr::null_mut();
        let mut factor_l_storage_nnz = 0;
        let mut factor_u_storage_nnz = 0;
        let mut factor_l_csc_nnz = 0;
        let mut factor_u_csc_nnz = 0;
        let mut solver_info = 0;
        let mut factor_ns = 0;
        let bridge_status = unsafe {
            (self.factorize)(
                matrix.n,
                i32::try_from(matrix.nnz()).map_err(|_| SuperLuExperimentError::IndexOverflow)?,
                col_ptr.as_ptr(),
                row_idx.as_ptr(),
                values.as_ptr(),
                &mut raw_handle,
                &mut factor_l_storage_nnz,
                &mut factor_u_storage_nnz,
                &mut factor_l_csc_nnz,
                &mut factor_u_csc_nnz,
                &mut solver_info,
                &mut factor_ns,
            )
        };
        if bridge_status != 0 {
            return Err(SuperLuExperimentError::Bridge(bridge_status));
        }
        if solver_info != 0 {
            return Err(SuperLuExperimentError::Factorization(solver_info));
        }
        let handle = NonNull::new(raw_handle).ok_or(SuperLuExperimentError::Bridge(-4))?;
        Ok(SuperLuFactor {
            handle,
            dimension: matrix.dimension(),
            factor_l_storage_nnz: factor_l_storage_nnz as usize,
            factor_u_storage_nnz: factor_u_storage_nnz as usize,
            factor_l_csc_nnz: factor_l_csc_nnz as usize,
            factor_u_csc_nnz: factor_u_csc_nnz as usize,
            factor_elapsed: Duration::from_nanos(factor_ns),
            solve: self.solve_factored,
            destroy: self.destroy_factor,
            _library: Arc::clone(&self._library),
        })
    }
}

/// Persistent native SuperLU factorization. Drop releases its C-owned factors.
pub struct SuperLuFactor {
    handle: NonNull<std::ffi::c_void>,
    dimension: usize,
    pub factor_l_storage_nnz: usize,
    pub factor_u_storage_nnz: usize,
    pub factor_l_csc_nnz: usize,
    pub factor_u_csc_nnz: usize,
    pub factor_elapsed: Duration,
    solve: FactoredSolveFn,
    destroy: DestroyFactorFn,
    // Keep function pointers valid until the opaque native factor is destroyed.
    _library: Arc<Library>,
}

impl SuperLuFactor {
    /// Solve in place using retained L/U factors without refactorization.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<Duration, SuperLuExperimentError> {
        if rhs.len() != self.dimension {
            return Err(SuperLuExperimentError::InvalidCsc(
                "RHS length must match the factored matrix dimension",
            ));
        }
        let mut solve_ns = 0;
        let status = unsafe { (self.solve)(self.handle.as_ptr(), rhs.as_mut_ptr(), &mut solve_ns) };
        if status != 0 {
            return Err(SuperLuExperimentError::Bridge(status));
        }
        Ok(Duration::from_nanos(solve_ns))
    }
}

impl Drop for SuperLuFactor {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct SuperLuSolve {
    pub solution: Vec<f64>,
    pub factor_l_storage_nnz: usize,
    pub factor_u_storage_nnz: usize,
    /// Materialized CSC L nonzeros, counted with SciPy's LU-to-CSC semantics.
    pub factor_l_csc_nnz: usize,
    /// Materialized CSC U nonzeros, including upper-triangular L supernode data.
    pub factor_u_csc_nnz: usize,
    /// Rust wall-clock around the FFI call, including the bridge's setup.
    pub ffi_elapsed: Duration,
    /// High-resolution native wall-clock around ordering plus numeric factorization.
    pub factor_elapsed: Duration,
    /// High-resolution native wall-clock around triangular solve.
    pub solve_elapsed: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_abi_conversion_rejects_bad_shapes_and_preserves_explicit_zeros() {
        let valid =
            SuperLuCsc32::try_from_usize(2, &[0, 2, 4], &[0, 1, 0, 1], &[2.0, 0.0, 1.0, 3.0])
                .expect("valid CSC with an explicit zero");
        assert_eq!(valid.dimension(), 2);
        assert_eq!(valid.nnz(), 4);
        assert!(matches!(
            SuperLuCsc32::try_from_usize(2, &[0, 2, 1], &[0, 1], &[1.0, 2.0]),
            Err(SuperLuExperimentError::InvalidCsc(_))
        ));
        assert!(matches!(
            SuperLuCsc32::try_from_usize(2, &[0, 2, 4], &[1, 0, 0, 1], &[1.0; 4]),
            Err(SuperLuExperimentError::InvalidCsc(_))
        ));
        assert!(matches!(
            SuperLuCsc32::try_from_usize(1, &[0, 1], &[0], &[f64::NAN]),
            Err(SuperLuExperimentError::InvalidCsc(_))
        ));
    }

    #[test]
    #[ignore = "requires the separately built SuperLU 6.0.1 bridge DLL"]
    fn superlu_control_solves_the_same_sparse_fixture() {
        let path = std::env::var("SUPERLU_LIBRARY")
            .or_else(|_| std::env::var("SUPERLU_EXPERIMENTAL_LIBRARY"))
            .expect("set path to the built SuperLU bridge DLL");
        let control = SuperLuControl::load(path).expect("load pinned SuperLU bridge");
        let matrix =
            SuperLuCsc32::try_from_usize(2, &[0, 2, 4], &[0, 1, 0, 1], &[4.0, 2.0, 1.0, 3.0])
                .unwrap();
        let rhs = [1.0, 1.0];
        let result = control.factor_and_solve(&matrix, &rhs).unwrap();
        assert!((result.solution[0] - 0.2).abs() < 1e-12);
        assert!((result.solution[1] - 0.2).abs() < 1e-12);
        assert_eq!(rhs, [1.0, 1.0]);
        assert!(result.factor_l_storage_nnz > 0 && result.factor_u_storage_nnz > 0);
        assert!(result.factor_l_csc_nnz > 0 && result.factor_u_csc_nnz > 0);
        assert_eq!(result.factor_l_csc_nnz, 3);
        assert_eq!(result.factor_u_csc_nnz, 3);

        let singular =
            SuperLuCsc32::try_from_usize(2, &[0, 2, 4], &[0, 1, 0, 1], &[1.0, 2.0, 2.0, 4.0])
                .unwrap();
        assert!(matches!(
            control.factor_and_solve(&singular, &[3.0, 6.0]),
            Err(SuperLuExperimentError::Factorization(info)) if info > 0
        ));
    }
}
