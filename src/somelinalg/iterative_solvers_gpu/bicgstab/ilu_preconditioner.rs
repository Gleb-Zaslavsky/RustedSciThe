//! # Incomplete LU (ILU0) Preconditioner
//!
//! ## Aim and General Description
//! This module implements the ILU(0) incomplete LU factorization for banded sparse matrices.
//! ILU(0) provides an approximate factorization A ≈ LU where L and U maintain the same
//! sparsity pattern as the original matrix A. This creates an effective preconditioner
//! M = LU that accelerates iterative solver convergence.
//!
//! ## Mathematical Considerations
//! - **ILU(0) Factorization**: Computes L and U factors with zero fill-in beyond original pattern
//! - **Banded Storage**: Exploits diagonal band structure for efficient memory usage
//! - **Forward/Backward Substitution**: Solves Ly = r, then Uz = y for preconditioning
//! - **Diagonal Pivoting**: Includes optional diagonal shift (eps) to avoid zero pivots
//! - **Numerical Stability**: Uses safeguards against very small pivot elements
//!
//! ## Main Components
//!
//! ### Structures
//! - `ILU0`: Stores factorized L and U matrices in banded diagonal format
//!   - `n`: Matrix dimension
//!   - `offsets`: Diagonal offset positions (negative=lower, 0=main, positive=upper)
//!   - `l_diags`: Lower triangular factor diagonals
//!   - `u_diags`: Upper triangular factor diagonals
//!
//! ### Main Functions
//! - `ILU0::new()`: Standard ILU(0) factorization without diagonal shift
//! - `ILU0::new_eps()`: ILU(0) with diagonal regularization parameter eps
//! - `ilu0_apply()`: Applies preconditioner M⁻¹r = z via forward/backward substitution
//!
//! ## Usage Examples
//! ```rust
//! // Create ILU0 factorization for tridiagonal matrix
//! let offsets = vec![-1, 0, 1];
//! let diags = vec![lower_diag, main_diag, upper_diag];
//! let ilu = ILU0::new(n, &offsets, &diags);
//!
//! // Apply preconditioning: solve Mz = r
//! let mut z = vec![0.0; n];
//! ilu0_apply(&ilu, &residual, &mut z);
//!
//! // With diagonal regularization for ill-conditioned matrices
//! let ilu_reg = ILU0::new_eps(n, &offsets, &diags, 1e-6);
//! ```
//!
//! ## Non-obvious Features and Tips
//! - **Zero Pivot Handling**: `new_eps()` adds regularization to prevent singular factorization
//! - **In-place Updates**: Factorization modifies U diagonals during L computation for efficiency
//! - **Banded Optimization**: Only processes non-zero diagonals, skipping empty bands
//! - **Memory Layout**: L and U stored separately in same diagonal format as input matrix
//! - **Numerical Thresholds**: Uses 1e-30 and 1e-12 thresholds for different numerical checks
//! - **CPU-only Implementation**: Designed for CPU execution, suitable for moderate-sized problems

pub struct ILU0 {
    pub n: usize,
    pub offsets: Vec<isize>,
    pub l_diags: Vec<Vec<f32>>,
    pub u_diags: Vec<Vec<f32>>,
}
/// ILU(0) factorisation of a banded matrix with optional diagonal shift.
/// `eps` is added to each pivot to avoid zero/very small pivots.
impl ILU0 {
    pub fn new_eps(
        n: usize,
        offsets: &[isize],
        diags: &[Vec<f32>],
        eps: f32, // new parameter
    ) -> ILU0 {
        let nbands = offsets.len();
        let mut l_diags = vec![vec![0.0f32; n]; nbands];
        let mut u_diags = vec![vec![0.0f32; n]; nbands];

        // Copy the original diagonals first
        for k in 0..nbands {
            for i in 0..n {
                if diags[k][i] != 0.0 {
                    if offsets[k] < 0 {
                        l_diags[k][i] = diags[k][i];
                    } else {
                        u_diags[k][i] = diags[k][i];
                    }
                }
            }
        }

        // Factorisation
        let main_idx = offsets
            .iter()
            .position(|&o| o == 0)
            .expect("No main diagonal");
        for i in 0..n {
            // For each lower band
            for (k, &off) in offsets.iter().enumerate() {
                if off >= 0 {
                    continue;
                }
                let j = (i as isize + off) as isize;
                if j < 0 || j >= n as isize {
                    continue;
                }
                let j_usize = j as usize;
                // compute L(i,j) = A(i,j)/U(j,j)
                let ujj = u_diags[main_idx][j_usize];
                if ujj.abs() < 1e-12 {
                    // safeguard
                    u_diags[main_idx][j_usize] = eps;
                }
                l_diags[k][i] /= u_diags[main_idx][j_usize];
            }

            // Update U bands
            for (k, &off) in offsets.iter().enumerate() {
                if off < 0 {
                    continue;
                }
                let j = i as isize + off;
                if j < 0 || j >= n as isize {
                    continue;
                }
               // let j_usize = j as usize;

                // subtract contributions from previous L*U
                let mut sum = 0.0f32;
                for (m, &off_m) in offsets.iter().enumerate() {
                    if off_m >= 0 {
                        continue;
                    }
                    let kcol = i as isize + off_m;
                    if kcol < 0 || kcol >= n as isize {
                        continue;
                    }
                    let kcol_usize = kcol as usize;
                    // find corresponding upper-band index
                    let offu = j as isize - kcol as isize;
                    if let Some(ku) = offsets.iter().position(|&o| o == offu) {
                        sum += l_diags[m][i] * u_diags[ku][kcol_usize];
                    }
                }
                u_diags[k][i] -= sum;
            }

            // Add diagonal shift to pivot
            u_diags[main_idx][i] += eps;
        }

        Self {
            n,
            offsets: offsets.to_vec(),
            l_diags,
            u_diags,
        }
    }

    pub fn new(n: usize, offsets: &[isize], diags: &[Vec<f32>]) -> ILU0 {
        let idx_main = offsets
            .iter()
            .position(|&o| o == 0)
            .expect("main diagonal offset not found");
        let mut l_diags = vec![vec![0.0f32; n]; offsets.len()];
        let mut u_diags = vec![vec![0.0f32; n]; offsets.len()];

        // initialise U diagonals with A diagonals
        for (k, diag) in diags.iter().enumerate() {
            u_diags[k].clone_from(diag);
        }

        for i in 0..n {
            // lower part (compute L entries and update U):
            for (k, &off) in offsets.iter().enumerate() {
                if off >= 0 {
                    continue;
                } // only lower
                let j = i as isize + off;
                if j < 0 {
                    continue;
                }
                let j = j as usize;
                let u_jj = u_diags[idx_main][j];
                if u_jj.abs() < 1e-30 {
                    continue;
                }
                let l_ij = u_diags[k][i] / u_jj;
                l_diags[k][i] = l_ij;
                // update U’s upper diagonals
                for (m, &offu) in offsets.iter().enumerate() {
                    if offu < 0 {
                        continue;
                    } // only main/upper
                    let jj = j as isize + offu;
                    let ii = i as isize + offu;
                    if ii >= 0 && ii < n as isize && jj >= 0 && jj < n as isize {
                        u_diags[m][i] -= l_ij * u_diags[m][j];
                    }
                }
            }
        }

        Self {
            n,
            offsets: offsets.to_vec(),
            l_diags,
            u_diags,
        }
    }
}
/// Apply ILU0 preconditioner on CPU: z = M^{-1} * r
pub fn ilu0_apply(ilu: &ILU0, r: &[f32], z: &mut [f32]) {
    let n = ilu.n;
    let idx_main = ilu
        .offsets
        .iter()
        .position(|&o| o == 0)
        .expect("main diagonal offset not found");

    // forward solve L y = r:
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = r[i];
        for (k, &off) in ilu.offsets.iter().enumerate() {
            if off >= 0 {
                continue;
            }
            let j = i as isize + off;
            if j >= 0 {
                sum -= ilu.l_diags[k][i] * y[j as usize];
            }
        }
        y[i] = sum;
    }

    // backward solve U z = y:
    for i in (0..n).rev() {
        let mut sum = y[i];
        for (k, &off) in ilu.offsets.iter().enumerate() {
            if off <= 0 {
                continue;
            }
            let j = i as isize + off;
            if j < n as isize {
                sum -= ilu.u_diags[k][i] * z[j as usize];
            }
        }
        let diag = ilu.u_diags[idx_main][i];
        z[i] = sum / diag;
    }
}

/////////////////////////////////////TESTING////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    // helper: multiply banded matrix by a dense vector on CPU for verification
    fn banded_matvec(n: usize, offsets: &[isize], diags: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for (k, &off) in offsets.iter().enumerate() {
                let j = i as isize + off;
                if j >= 0 && j < n as isize {
                    sum += diags[k][i] * x[j as usize];
                }
            }
            y[i] = sum;
        }
        y
    }

    // Identity matrix: ILU0 of I is I, so z=r exactly.
    #[test]
    fn test_ilu0_apply_identity() {
        // A = I (main diag=1)
        let n = 5;
        let offsets = [0isize];
        let diags = vec![vec![1.0f32; n]];

        // ILU0 of I should be I
        let ilu = ILU0::new(n, &offsets, &diags);

        // apply to r
        let r = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut z = vec![0.0f32; n];
        ilu0_apply(&ilu, &r, &mut z);

        assert!((0..n).all(|i| (z[i] - r[i]).abs() < 1e-6));
    }
    // Tridiagonal matrix: ILU0 produces an approximate LU. If you apply ILU0 to r you should get z such that A z ≈ r.
    // The test multiplies A z and checks it’s close to r.
    #[test]
    #[allow(non_snake_case)]
    fn test_ilu0_apply_tridiagonal() {
        // A = tridiagonal diag=4, off=-1=1, off=+1=1
        let n = 5;
        let offsets = [-1isize, 0isize, 1isize];
        let d_main = vec![4.0f32; n];
        let d_lower = vec![1.0f32; n];
        let d_upper = vec![1.0f32; n];
        let diags = vec![d_lower.clone(), d_main.clone(), d_upper.clone()];

        // Build ILU0
        let ilu = ILU0::new(n, &offsets, &diags);

        // Pick a random vector r
        let r = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        // Solve Mz=r
        let mut z = vec![0.0f32; n];
        ilu0_apply(&ilu, &r, &mut z);

        // For ILU0 of a diagonally dominant tridiagonal, M≈A, so Az≈r
        let Az = banded_matvec(n, &offsets, &diags, &z);

        // check that A*z ≈ r
        for i in 0..n {
            assert!(
                (Az[i] - r[i]).abs() < 1e-3,
                "row {}: Az={} r={}",
                i,
                Az[i],
                r[i]
            );
        }
    }
}

#[cfg(test)]
mod tests_ilu {
    use super::*;

    // Build dense M = L*U from ILU0
    #[allow(non_snake_case)]
    fn build_dense_from_ilu(ilu: &ILU0) -> Vec<Vec<f32>> {
        let n = ilu.n;
        let _idx_main = ilu.offsets.iter().position(|&o| o == 0).unwrap();
        let mut M = vec![vec![0.0f32; n]; n];
        // Fill with L and U product:
        // L has implicit 1.0 on main diag
        for i in 0..n {
            for (_k, &off) in ilu.offsets.iter().enumerate() {
                let j = i as isize + off;
                if j < 0 || j >= n as isize {
                    continue;
                }
                let _j = j as usize;
                // fill L and U separately then multiply
            }
        }
        // Simpler: explicitly construct L and U dense then multiply
        let mut L = vec![vec![0.0f32; n]; n];
        let mut U = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            L[i][i] = 1.0;
        }
        for (k, &off) in ilu.offsets.iter().enumerate() {
            for i in 0..n {
                let j = i as isize + off;
                if j < 0 || j >= n as isize {
                    continue;
                }
                let j = j as usize;
                if off < 0 {
                    L[i][j] = ilu.l_diags[k][i];
                } else {
                    U[i][j] = ilu.u_diags[k][i];
                }
            }
        }
        // now M=L*U
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += L[i][k] * U[k][j];
                }
                M[i][j] = sum;
            }
        }
        M
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_ilu0_preconditioner_identity() {
        let n = 5;
        let offsets = [0isize];
        let diags = vec![vec![1.0f32; n]];
        let ilu = ILU0::new(n, &offsets, &diags);
        let r = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut z = vec![0.0f32; n];
        ilu0_apply(&ilu, &r, &mut z);
        // compute M z:
        let M = build_dense_from_ilu(&ilu);
        let mut Mz = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += M[i][j] * z[j];
            }
            Mz[i] = sum;
        }
        for i in 0..n {
            assert!(
                (Mz[i] - r[i]).abs() < 1e-6,
                "row {}: Mz={} r={}",
                i,
                Mz[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_ilu0_sparsity_pattern() {
        // small tridiagonal
        let n = 5;
        let offsets = [-1isize, 0isize, 1isize];
        let diags = vec![vec![1.0f32; n], vec![4.0f32; n], vec![1.0f32; n]];
        let ilu = ILU0::new(n, &offsets, &diags);

        // count nonzeros in A’s pattern
        let mut nnz_a = 0;
        for (k, &_off) in offsets.iter().enumerate() {
            nnz_a += diags[k].iter().filter(|&&v| v != 0.0).count();
        }

        // count nonzeros in L+U pattern
        let mut nnz_lu = 0;
        for (k, &_off) in ilu.offsets.iter().enumerate() {
            nnz_lu += ilu.l_diags[k].iter().filter(|&&v| v != 0.0).count();
            nnz_lu += ilu.u_diags[k].iter().filter(|&&v| v != 0.0).count();
        }
        println!("nnz_a={} nnz_lu={}", nnz_a, nnz_lu);
        // ILU0 should not create more fill than A
        assert!(nnz_lu <= nnz_a * 2, "nnz_lu={} nnz_a={}", nnz_lu, nnz_a);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_ilu0_preconditioner_tridiagonal() {
        let n = 5;
        let offsets = [-1isize, 0isize, 1isize];
        let diags = vec![vec![1.0f32; n], vec![4.0f32; n], vec![1.0f32; n]];
        let ilu = ILU0::new(n, &offsets, &diags);
        let r = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut z = vec![0.0f32; n];
        ilu0_apply(&ilu, &r, &mut z);
        // compute M z exactly:

        let M = build_dense_from_ilu(&ilu);

        let mut Mz = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += M[i][j] * z[j];
            }
            Mz[i] = sum;
        }
        for i in 0..n {
            assert!(
                (Mz[i] - r[i]).abs() < 1e-5,
                "row {}: Mz={} r={}",
                i,
                Mz[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_ilu0_sparsity_pattern_nnz_exext() {
        let n = 5;
        let offsets = [-1isize, 0isize, 1isize];
        let diags = vec![vec![1.0f32; n], vec![4.0f32; n], vec![1.0f32; n]];
        let ilu = ILU0::new(n, &offsets, &diags);

        // count nonzeros in A’s pattern properly
        let mut nnz_a = 0;
        for (k, &off) in offsets.iter().enumerate() {
            for i in 0..n {
                let j = i as isize + off;
                if j < 0 || j >= n as isize {
                    continue;
                }
                if diags[k][i] != 0.0 {
                    nnz_a += 1;
                }
            }
        }

        // count nonzeros in L and U properly
        let mut nnz_lu = 0;
        for (k, &off) in ilu.offsets.iter().enumerate() {
            for i in 0..n {
                let j = i as isize + off;
                if j < 0 || j >= n as isize {
                    continue;
                }
                if off < 0 {
                    if ilu.l_diags[k][i] != 0.0 {
                        nnz_lu += 1;
                    }
                } else {
                    if ilu.u_diags[k][i] != 0.0 {
                        nnz_lu += 1;
                    }
                }
            }
        }
        println!("nnz_a {} nnz_lu {}", nnz_a, nnz_lu);
        assert_eq!(
            nnz_a, nnz_lu,
            "nnz pattern mismatch: A={} L+U={}",
            nnz_a, nnz_lu
        );
    }
}
