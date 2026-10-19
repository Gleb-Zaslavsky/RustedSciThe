#![cfg(feature = "arrayfire")]

//! # BiCGStab Matrix API and Format Conversion
//!
//! ## Aim and General Description
//! This module provides a high-level API for the BiCGStab solver that works with standard sparse matrix
//! formats. It bridges between the faer sparse matrix library and the optimized banded BiCGStab solver,
//! handling format conversions and providing a convenient interface for general sparse linear systems.
//!
//! ## Mathematical Considerations
//! - **Sparse Matrix Formats**: Converts between CSC (Compressed Sparse Column) and banded diagonal storage
//! - **Diagonal Extraction**: Identifies and extracts diagonal bands from arbitrary sparse patterns
//! - **Format Optimization**: Converts general sparse matrices to efficient banded representation when possible
//! - **Precision Handling**: Manages f32/f64 conversions between different library interfaces
//!
//! ## Main Functions
//!
//! ### Format Conversion Functions
//! - `sparsecol_to_banded()`: Converts faer SparseColMat to banded diagonal format
//!   - Extracts unique diagonal offsets from sparse pattern
//!   - Creates aligned diagonal storage for efficient GPU operations
//!   - Returns (offsets, diags_host) suitable for BiCGStab solver
//!
//! - `banded_to_sparsecol()`: Converts banded format back to standard sparse matrix
//!   - Reconstructs CSC format from diagonal representation
//!   - Maintains numerical precision and sparsity pattern
//!   - Useful for verification and interfacing with other libraries
//!
//! ### High-level Solver Interface
//! - `bicgstab_solver()`: Complete solver interface for general sparse matrices
//!   - Accepts standard SparseColMat input format
//!   - Handles all format conversions internally
//!   - Returns solution vector with iteration count and residual
//!   - Supports all preconditioner types
//!
//! ## Usage Examples
//! ```rust, ignore
//! // Convert sparse matrix to banded format
//! let (offsets, diags_host) = sparsecol_to_banded(&sparse_matrix);
//!
//! // Solve linear system with high-level API
//! let (solution, iterations, residual) = bicgstab_solver(
//!     matrix, rhs_vector, initial_guess, tolerance, max_iterations,
//!     PreconditionerType::GS { sweeps: 2, symmetric: false }
//! )?;
//!
//! // Round-trip conversion for verification
//! let reconstructed = banded_to_sparsecol(&offsets, &diags_host);
//! ```
//!
//! ## Non-obvious Features and Tips
//! - **Automatic Diagonal Detection**: Uses BTreeSet to automatically identify and sort diagonal offsets
//! - **Sparse Pattern Preservation**: Maintains exact sparsity pattern during round-trip conversions
//! - **Memory Efficiency**: Only stores non-zero diagonals, skipping empty bands
//! - **Precision Management**: Handles f32/f64 conversions transparently for optimal GPU performance
//! - **Error Handling**: Returns Result types for robust error propagation
//! - **Integration Testing**: Includes comprehensive tests comparing with direct LU solutions
//! - **Triplet Iteration**: Uses efficient triplet iteration for sparse matrix traversal

use crate::somelinalg::iterative_solvers_gpu::bicgstab::bicgstab_with_preconditioneer::{
    PreconditionerType, solve_banded_bicgstab_flexible_f32,
};
use af::{Array, Dim4};
use arrayfire as af;
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
/// comverts faer sparse matrix to offsets and diagonal host
/// offsets: Vec<isize> = sorted unique diagonal offsets (negative = below main, 0 = main, positive = above main).

///diags_host: Vec<Vec<f32>> = each entry is a vector of length n (nrows), storing the coefficients for that diagonal, aligned with row index
use std::collections::BTreeSet;

/// Convert a `SparseColMat<f32, I>` to banded format used by our BiCGStab solver.
/// Returns `(offsets, diags_host)` where
/// - `offsets[k]` = column - row for diagonal k
/// - `diags_host[k][i]` = value at row i, col i+offset (if exists) else 0.0
pub fn sparsecol_to_banded(mat: &SparseColMat<usize, f64>) -> (Vec<isize>, Vec<Vec<f32>>) {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    assert_eq!(nrows, ncols, "Matrix must be square for banded conversion");

    // 1. Collect unique offsets
    let mut offsets_set = BTreeSet::new();
    for Triplet {
        row: i,
        col: j,
        val: _v,
    } in mat.triplet_iter()
    {
        // i = row, j = col
        let off = (j as isize) - (i as isize);
        offsets_set.insert(off);
    }
    let offsets: Vec<isize> = offsets_set.into_iter().collect();

    // 2. Prepare diags_host
    let mut diags_host: Vec<Vec<f32>> = offsets.iter().map(|_| vec![0.0f32; nrows]).collect();

    // 3. Fill each diagonal
    // Offsets index map for fast lookup
    let mut offset_index = std::collections::HashMap::new();
    for (k, &off) in offsets.iter().enumerate() {
        offset_index.insert(off, k);
    }

    for Triplet {
        row: i,
        col: j,
        val: v,
    } in mat.triplet_iter()
    {
        let off = (j as isize) - (i as isize);
        if let Some(&k) = offset_index.get(&off) {
            // store value at row i
            diags_host[k][i as usize] = *v as f32;
        }
    }

    (offsets, diags_host)
}

/// Convert (offsets, diags_host) back to a SparseColMat<usize,f64>
pub fn banded_to_sparsecol(offsets: &[isize], diags_host: &[Vec<f32>]) -> SparseColMat<usize, f64> {
    let n = diags_host[0].len();
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    // Collect all nonzero entries as (row, col, val)
    for (k, &off) in offsets.iter().enumerate() {
        for i in 0..n {
            let j = (i as isize) + off;
            if j >= 0 && j < n as isize {
                let val = diags_host[k][i];
                if val != 0.0 {
                    triplets.push((i, j as usize, val as f64));
                }
            }
        }
    }

    // Group by column, sort rows inside each column
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    col_ptr.push(0);
    let mut count = 0;
    for col in 0..n {
        // Collect all rows for this col
        let mut col_entries: Vec<_> = triplets
            .iter()
            .filter(|&&(_, j, _)| j == col)
            .map(|&(i, _, val)| (i, val))
            .collect();
        // Sort by row index ascending
        col_entries.sort_by_key(|&(i, _)| i);
        // Push to arrays
        for (i, val) in col_entries {
            row_idx.push(i);
            values.push(val);
            count += 1;
        }
        col_ptr.push(count);
    }

    let symbolic = SymbolicSparseColMat::new_checked(n, n, col_ptr, None, row_idx);
    SparseColMat::new(symbolic, values)
}
#[allow(non_snake_case)]
pub fn bicgstab_solver(
    A: SparseColMat<usize, f64>,
    b: Vec<f64>,
    x0: Vec<f64>,
    tol: f64,
    max_iter: usize,
    preconditioner: PreconditionerType,
) -> Result<(Vec<f64>, usize, f32), Box<dyn std::error::Error>> {
    let (offsets, diags_host) = sparsecol_to_banded(&A);
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let x0_f32: Vec<f32> = x0.iter().map(|&x| x as f32).collect();
    let b_array = Array::new(&b_f32, Dim4::new(&[b_f32.len() as u64, 1, 1, 1]));
    let x0_array = Array::new(&x0_f32, Dim4::new(&[x0_f32.len() as u64, 1, 1, 1]));

    let (x_result, iter, res) = solve_banded_bicgstab_flexible_f32(
        A.nrows(),
        &offsets,
        &diags_host,
        &b_array,
        Some(&x0_array),
        tol as f32,
        max_iter,
        preconditioner,
    );

    // FIX: copy from ArrayFire Array to Vec
    let mut x_host = vec![0.0f32; x_result.elements()];
    x_result.host(&mut x_host);
    let x_f64: Vec<f64> = x_host.iter().map(|&x| x as f64).collect();

    Ok((x_f64, iter, res))
}

#[cfg(all(test, feature = "arrayfire"))]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, SymbolicSparseColMat};

    #[test]
    fn test_diagonal_matrix() {
        let col_ptr = vec![0, 1, 2, 3];
        let row_idx = vec![0, 1, 2];
        let values = vec![1.0, 2.0, 3.0];
        let symbolic = SymbolicSparseColMat::new_checked(3, 3, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);

        let (offsets, diags) = sparsecol_to_banded(&mat);

        assert_eq!(offsets, vec![0]);
        assert_eq!(diags[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tridiagonal_matrix() {
        let col_ptr = vec![0, 2, 5, 7];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0];
        let symbolic = SymbolicSparseColMat::new_checked(3, 3, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);

        let (offsets, diags) = sparsecol_to_banded(&mat);

        assert_eq!(offsets, vec![-1, 0, 1]);
        assert_eq!(diags[0], vec![0.0, 1.0, 1.0]);
        assert_eq!(diags[1], vec![2.0, 2.0, 2.0]);
        assert_eq!(diags[2], vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_roundtrip_sparsecol_banded() {
        // Build a small tridiagonal
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let symbolic = SymbolicSparseColMat::new_checked(4, 4, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);

        let (offsets, diags) = sparsecol_to_banded(&mat);
        let mat2 = banded_to_sparsecol(&offsets, &diags);

        // Check same structure & values
        assert_eq!(mat.nrows(), mat2.nrows());
        assert_eq!(mat.ncols(), mat2.ncols());
        let mut vals1: Vec<_> = mat.triplet_iter().collect();
        let mut vals2: Vec<_> = mat2.triplet_iter().collect();
        vals1.sort_by_key(|t| (t.row, t.col));
        vals2.sort_by_key(|t| (t.row, t.col));
        assert_eq!(vals1.len(), vals2.len());
        for (t1, t2) in vals1.iter().zip(vals2.iter()) {
            assert_eq!(t1.row, t2.row);
            assert_eq!(t1.col, t2.col);
            assert!((t1.val - t2.val).abs() < 1e-6);
        }
    }
    #[test]
    fn debug_banded_conversion() {
        // Test the exact case from small_example
        let offsets = vec![-1isize, 0isize, 1isize];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0], // Lower diagonal
            vec![2.0, 20.0, 20.0, 20.0],    // Main diagonal
            vec![-10.0, -10.0, -10.0, 0.0], // Upper diagonal
        ];

        println!("Original banded format:");
        println!("Offsets: {:?}", offsets);
        for (i, diag) in diags.iter().enumerate() {
            println!("Diagonal {} (offset {}): {:?}", i, offsets[i], diag);
        }

        // Convert to sparse matrix
        let sparse_mat = banded_to_sparsecol(&offsets, &diags);

        println!("\nConverted to sparse matrix:");
        for triplet in sparse_mat.triplet_iter() {
            println!("({}, {}) = {}", triplet.row, triplet.col, triplet.val);
        }

        // Print as dense matrix for visualization
        println!("\nAs dense matrix:");
        for i in 0..4 {
            let mut row = vec![0.0; 4];
            for Triplet {
                row: row_i,
                col: col_j,
                val: value,
            } in sparse_mat.triplet_iter()
            {
                if row_i == i {
                    row[col_j] = *value;
                }
            }
            println!("{:?}", row);
        }

        // Convert back to banded
        let (offsets2, diags2) = sparsecol_to_banded(&sparse_mat);

        println!("\nConverted back to banded:");
        println!("Offsets: {:?}", offsets2);
        for (i, diag) in diags2.iter().enumerate() {
            println!("Diagonal {} (offset {}): {:?}", i, offsets2[i], diag);
        }

        // Check if round-trip preserves data
        assert_eq!(offsets, offsets2);
        for (d1, d2) in diags.iter().zip(diags2.iter()) {
            for (v1, v2) in d1.iter().zip(d2.iter()) {
                assert!((v1 - v2).abs() < 1e-6, "Value mismatch: {} vs {}", v1, v2);
            }
        }
    }

    #[test]
    fn debug_manual_matrix_construction() {
        // Manually construct the expected matrix and convert to banded
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![
            2.0, -10.0, -10.0, 20.0, -10.0, -10.0, 20.0, -10.0, -10.0, 20.0,
        ];
        let symbolic = SymbolicSparseColMat::new_checked(4, 4, col_ptr, None, row_idx);
        let expected_mat = SparseColMat::new(symbolic, values);

        println!("Expected matrix (manually constructed):");
        for triplet in expected_mat.triplet_iter() {
            println!("({}, {}) = {}", triplet.row, triplet.col, triplet.val);
        }

        // Convert to banded format
        let (offsets, diags) = sparsecol_to_banded(&expected_mat);

        println!("\nExpected matrix as banded:");
        println!("Offsets: {:?}", offsets);
        for (i, diag) in diags.iter().enumerate() {
            println!("Diagonal {} (offset {}): {:?}", i, offsets[i], diag);
        }

        // This should match what we expect for the small_example test
        // Expected: offsets [-1, 0, 1] with specific diagonal values
        let expected_offsets = vec![-1isize, 0isize, 1isize];
        let expected_diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];

        assert_eq!(offsets, expected_offsets, "Offsets don't match expected");
        assert_eq!(
            diags.len(),
            expected_diags.len(),
            "Number of diagonals doesn't match"
        );

        for (i, (actual_diag, expected_diag)) in diags.iter().zip(expected_diags.iter()).enumerate()
        {
            for (j, (&actual, &expected)) in
                actual_diag.iter().zip(expected_diag.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "Diagonal {} position {}: got {}, expected {}",
                    i,
                    j,
                    actual,
                    expected
                );
            }
        }
    }

    #[test]
    fn verify_actual_matrix_from_diags() {
        // Use exact same diagonal data as small_example
        let offsets = vec![-1isize, 0isize, 1isize];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];

        // Convert to sparse and print the actual matrix
        let mat = banded_to_sparsecol(&offsets, &diags);

        println!("Matrix represented by the diagonal data:");
        for i in 0..4 {
            let mut row = vec![0.0; 4];
            for Triplet {
                row: row_i,
                col: col_j,
                val: value,
            } in mat.triplet_iter()
            {
                if row_i == i {
                    row[col_j] = *value;
                }
            }
            println!("Row {}: {:?}", i, row);
        }

        // Test matrix-vector multiplication with [1,1,1,1]
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 4];

        for triplet in mat.triplet_iter() {
            y[triplet.row] += triplet.val * x[triplet.col];
        }

        println!("\nMatrix * [1,1,1,1] = {:?}", y);
        println!("This should match the 'Sparse SpMV result' from verify_matrix_reconstruction");

        // Expected result: Row 0: 2*1 + (-10)*1 = -8, Row 3: (-10)*1 + 20*1 = 10
        let expected = vec![-8.0, 0.0, 0.0, 10.0];
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "Mismatch at position {}: got {}, expected {}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_roundtrip2() {
        let _b = vec![1.0, 0.0, 0.0, 1.0];
        // A = tridiag([-1,2,-1]) for n=4
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let symbolic = SymbolicSparseColMat::new_checked(4, 4, col_ptr, None, row_idx);
        let mat: SparseColMat<usize, f64> = SparseColMat::new(symbolic, values);
        let (offsets, diags) = sparsecol_to_banded(&mat);
        let mat2 = banded_to_sparsecol(&offsets, &diags);

        // Check same structure & values
        assert_eq!(mat.nrows(), mat2.nrows());
        assert_eq!(mat.ncols(), mat2.ncols());
        let mut vals1: Vec<_> = mat.triplet_iter().collect();
        let mut vals2: Vec<_> = mat2.triplet_iter().collect();
        vals1.sort_by_key(|t| (t.row, t.col));
        vals2.sort_by_key(|t| (t.row, t.col));
        assert_eq!(vals1.len(), vals2.len());
        for (t1, t2) in vals1.iter().zip(vals2.iter()) {
            assert_eq!(t1.row, t2.row);
            assert_eq!(t1.col, t2.col);
            assert!((t1.val - t2.val).abs() < 1e-6);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod integration_tests {
    use super::*;
    use faer::{
        col::Col,
        prelude::Solve,
        sparse::{SparseColMat, SymbolicSparseColMat},
    };

    #[allow(dead_code)]
    // helper to build ArrayFire Array from Vec<f32>
    fn to_af_array(v: &[f32]) -> arrayfire::Array<f32> {
        use arrayfire::{Array, Dim4};
        Array::new(v, Dim4::new(&[v.len() as u64, 1, 1, 1]))
    }

    #[test]
    fn test_bicgstab_solver_on_diagonal() {
        // A = diag(4,4,4)
        let col_ptr = vec![0, 1, 2, 3]; // CSR style col_ptr
        let row_idx = vec![0, 1, 2];
        let values = vec![4.0, 4.0, 4.0];
        let symbolic = SymbolicSparseColMat::new_checked(3, 3, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);

        // RHS b = (8,8,8)
        let b = vec![8.0f64, 8.0, 8.0];
        // x0 = zeros
        let x0 = vec![0.0f64; 3];

        let (x, iter, res) =
            bicgstab_solver(mat, b, x0, 1e-6, 100, PreconditionerType::Vanilla).expect("solver ok");

        // Should converge to x = b / diag = 2.0
        for xi in x {
            assert!((xi - 2.0).abs() < 1e-4);
        }
        assert!(iter <= 5);
        assert!(res < 1e-3);
    }

    #[test]
    fn test_bicgstab_solver_on_tridiagonal() {
        let b = vec![1.0, 0.0, 0.0, 1.0];
        // A = tridiag([-1,2,-1]) for n=4
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let symbolic = SymbolicSparseColMat::new_checked(4, 4, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);
        //lu solution
        let lu = mat.sp_lu().unwrap();

        let b_col: Col<f64> = Col::from_iter(b.clone().iter().map(|x| *x));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();
        let lu_solution = lu.solve(b_mat);
        let x_lu: Vec<f64> = lu_solution.row_iter().map(|x| x[0]).collect();
        // x_true = (1,1,1,1)

        // compute b = A * x_true
        // for this small matrix manually:
        // row0: 2*1 -1*1 =1
        // row1: -1*1 +2*1 -1*1=0
        // row2: -1*1 +2*1 -1*1=0
        // row3: -1*1 +2*1=1

        let x0 = vec![0.7f64; 4];

        let (x, iter, res) = bicgstab_solver(
            mat.clone(),
            b,
            x0,
            1e-6,
            1500,
            PreconditionerType::GS {
                sweeps: 3,
                symmetric: false,
            },
        )
        .expect("solver ok");
        let x_col: Col<f64> = Col::from_iter(x.clone().iter().map(|x| *x));
        let binding = x_col.clone();
        let x_mat = binding.as_mat();
        let r = mat * x_mat - b_mat;
        println!("x = {:?}", x);
        println!("lu solution {:?}", x_lu);
        println!("iter = {:?}, res = {}", iter, res);
        println!("r = {:?}", r);
        for (xi, x_lui) in x.iter().zip(x_lu.iter()) {
            assert!((xi - x_lui).abs() < 1e-3, "xi={} xt={}", xi, x_lui);
        }
    }

    #[test]
    fn test_bicgstab_solver_on_tridiagonal2() {
        let b = vec![1.0, 0.0, 0.0, 1.0];
        // A = tridiag([-1,2,-1]) for n=4
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![
            2.0, -10.0, -10.0, 20.0, -10.0, -10.0, 20.0, -10.0, -10.0, 20.0,
        ];
        let symbolic = SymbolicSparseColMat::new_checked(4, 4, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);
        let (offsets, diags) = sparsecol_to_banded(&mat);
        println!("offsets = {:?}", offsets);
        println!("diags = {:?}", diags);
        //lu solution
        let lu = mat.sp_lu().unwrap();

        let b_col: Col<f64> = Col::from_iter(b.clone().iter().map(|x| *x));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();
        let lu_solution = lu.solve(b_mat);
        let x_lu: Vec<f64> = lu_solution.row_iter().map(|x| x[0]).collect();
        // x_true = (1,1,1,1)

        // compute b = A * x_true
        // for this small matrix manually:
        // row0: 2*1 -1*1 =1
        // row1: -1*1 +2*1 -1*1=0
        // row2: -1*1 +2*1 -1*1=0
        // row3: -1*1 +2*1=1

        let x0 = vec![0.7f64; 4];

        let (x, iter, res) = bicgstab_solver(
            mat.clone(),
            b,
            x0,
            1e-7,
            1500,
            PreconditionerType::GS {
                sweeps: 3,
                symmetric: false,
            },
        )
        .expect("solver ok");
        let x_col: Col<f64> = Col::from_iter(x.clone().iter().map(|x| *x));
        let binding = x_col.clone();
        let x_mat = binding.as_mat();
        let r = mat * x_mat - b_mat;
        println!("x = {:?}", x);
        println!("lu solution {:?}", x_lu);
        println!("iter = {:?}, res = {}", iter, res);
        println!("r = {:?}", r);
        for (xi, x_lui) in x.iter().zip(x_lu.iter()) {
            assert!((xi - x_lui).abs() < 1e-3, "xi={} xt={}", xi, x_lui);
        }
    }
}
