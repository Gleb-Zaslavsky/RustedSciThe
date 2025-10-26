#![cfg(feature = "arrayfire")]

//! # Banded Matrix Utilities for GPU Operations
//!
//! ## Aim and General Description
//! This module provides essential utilities for handling banded sparse matrices on GPU using ArrayFire.
//! It implements efficient storage and computation patterns for matrices with diagonal band structure,
//! enabling high-performance sparse matrix-vector multiplication and data transfer operations.
//!
//! ## Mathematical Considerations
//! - **Banded Matrix Storage**: Stores matrix as collection of diagonals with offset positions
//! - **Diagonal Masking**: Uses masks to handle boundary conditions where diagonals extend beyond matrix bounds
//! - **Shifted Operations**: Leverages ArrayFire's shift operations for efficient diagonal access
//! - **GPU Memory Layout**: Optimizes data layout for coalesced GPU memory access patterns
//!
//! ## Main Functions
//!
//! ### Data Upload and Storage
//! - `upload_banded_f32()`: Transfers banded matrix from CPU to GPU with boundary masks
//!   - Input: Host diagonal data and offset positions
//!   - Output: GPU arrays for diagonals and corresponding boundary masks
//!   - Creates masks to zero out invalid entries at matrix boundaries
//!
//! ### Matrix Operations
//! - `banded_spmv_f32()`: GPU-accelerated sparse matrix-vector multiplication
//!   - Implements y = A*x for banded matrix A using diagonal representation
//!   - Uses ArrayFire shift operations for efficient diagonal access
//!   - Applies boundary masks to ensure correct results
//!
//! ## Usage Examples
//! ```rust, ignore
//! // Upload tridiagonal matrix to GPU
//! let offsets = vec![-1, 0, 1];
//! let diags_host = vec![lower_diag, main_diag, upper_diag];
//! let (diags_gpu, masks_gpu) = upload_banded_f32(&diags_host, &offsets);
//!
//! // Perform matrix-vector multiplication: y = A*x
//! let y = banded_spmv_f32(&diags_gpu, &masks_gpu, &offsets, &x);
//! ```
//!
//! ## Non-obvious Features and Tips
//! - **Boundary Masking**: Automatically handles matrix boundaries by zeroing invalid diagonal entries
//! - **Memory Coalescing**: Uses ArrayFire's optimized memory access patterns for GPU efficiency
//! - **Lazy Evaluation**: Leverages ArrayFire's lazy evaluation system with explicit `af::eval!()` calls
//! - **Shift Operations**: Uses `af::shift()` for efficient diagonal access without explicit indexing
//! - **Saturating Arithmetic**: Uses `saturating_sub()` to safely handle negative indices in boundary calculations
//! - **Dimension Consistency**: Maintains consistent Dim4 format for all ArrayFire operations

use arrayfire::{self as af, Array, Dim4};

pub fn upload_banded_f32(
    diags_host: &Vec<Vec<f32>>,
    offsets: &Vec<i32>,
) -> (Vec<Array<f32>>, Vec<Array<f32>>) {
    let m = diags_host.len();
    assert_eq!(m, offsets.len());
    let n = diags_host[0].len();
    let dims = Dim4::new(&[n as u64, 1, 1, 1]);

    let mut diags_dev = Vec::with_capacity(m);
    let mut masks_dev = Vec::with_capacity(m);

    for (&offset, diag_host) in offsets.iter().zip(diags_host.iter()) {
        let diag_arr = Array::new(&diag_host[..], dims);
        let mut mask_host = vec![1.0f32; n];

        if offset > 0 {
            let s = offset as usize;
            for i in (n.saturating_sub(s))..n {
                mask_host[i] = 0.0;
            }
        } else if offset < 0 {
            let s = (-offset) as usize;
            for i in 0..s.min(n) {
                mask_host[i] = 0.0;
            }
        }
        /*
        if offset > 0 {
            let s = offset as usize;
            for i in 0..s.min(n) {
                mask_host[i] = 0.0;
            }
        } else if offset < 0 {
            let s = (-offset) as usize;
            for i in (n.saturating_sub(s))..n {
                mask_host[i] = 0.0;
            }
        }
        */
        let mask_arr = Array::new(&mask_host[..], dims);

        diags_dev.push(diag_arr);
        masks_dev.push(mask_arr);
    }
    (diags_dev, masks_dev)
}
/*
old version with masking - has boundary issues
pub fn banded_spmv_f32(
    diags: &Vec<Array<f32>>,
    masks: &Vec<Array<f32>>,
    offsets: &Vec<i32>,
    x: &Array<f32>,
) -> Array<f32> {
    let dims = x.dims();
    let mut y = af::constant(0.0f32, dims);

    for (i, &off) in offsets.iter().enumerate() {
        let x_shift = af::shift(x, &[off, 0i32, 0i32, 0i32]) * &masks[i];
        let term = &diags[i] * &x_shift;
        af::eval!(&term);
        y = &y + &term;
    }
    af::eval!(&y);
    y
}

*/
pub fn banded_spmv_f32(diags: &Vec<Array<f32>>, offsets: &Vec<i32>, x: &Array<f32>) -> Array<f32> {
    let dims = x.dims();
    let n = dims[0] as i64;
    let mut y = af::constant(0.0f32, dims);

    for (diag_idx, &off) in offsets.iter().enumerate() {
        let x_shifted = if off == 0 {
            // No shift needed for main diagonal
            x.clone()
        } else if off > 0 {
            // Positive offset: need x[i+off], so take x[off:] and pad zeros at end
            let x_trimmed = af::rows(x, off as i64, n - 1);
            let zeros = af::constant(0.0f32, Dim4::new(&[off as u64, 1, 1, 1]));
            af::join(0, &x_trimmed, &zeros)
        } else {
            // Negative offset: need x[i+off], so pad zeros at beginning and take x[:n+off]
            let abs_off = (-off) as i64;
            let zeros = af::constant(0.0f32, Dim4::new(&[abs_off as u64, 1, 1, 1]));
            let x_trimmed = af::rows(x, 0, n - abs_off - 1);
            af::join(0, &zeros, &x_trimmed)
        };

        let term = &diags[diag_idx] * &x_shifted;
        af::eval!(&term);
        y = &y + &term;
    }
    af::eval!(&y);
    y
}

#[cfg(all(test, feature = "arrayfire"))]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_masking_diagnosis() {
        // Test case from small_example that shows boundary errors
        let offsets = vec![-1, 0, 1];
        let diags_host = vec![
            vec![0.0, -10.0, -10.0, -10.0], // Lower diagonal
            vec![2.0, 20.0, 20.0, 20.0],    // Main diagonal
            vec![-10.0, -10.0, -10.0, 0.0], // Upper diagonal
        ];

        let (diags_gpu, masks_gpu) = upload_banded_f32(&diags_host, &offsets);

        // Check masks - this will reveal the masking problem
        for (i, &offset) in offsets.iter().enumerate() {
            let mut mask_host = vec![0.0f32; 4];
            masks_gpu[i].host(&mut mask_host);
            println!("Offset {}: mask = {:?}", offset, mask_host);
        }

        // Test SpMV with identity vector to see boundary effects
        let x_host = vec![1.0f32; 4];
        let x = Array::new(&x_host, Dim4::new(&[4, 1, 1, 1]));
        let y = banded_spmv_f32(&diags_gpu, &offsets, &x);

        let mut y_host = vec![0.0f32; 4];
        y.host(&mut y_host);
        println!("SpMV result with ones: {:?}", y_host);

        // Expected result for matrix * [1,1,1,1]:
        // Row 0: 2*1 + (-10)*1 = -8
        // Row 1: (-10)*1 + 20*1 + (-10)*1 = 0
        // Row 2: (-10)*1 + 20*1 + (-10)*1 = 0
        // Row 3: (-10)*1 + 20*1 = 10
        let expected = vec![-8.0, 0.0, 0.0, 10.0];

        for (i, (&actual, &exp)) in y_host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "SpMV result mismatch at position {}: got {}, expected {}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_shift_operation_boundary_behavior() {
        let x_host = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = Array::new(&x_host, Dim4::new(&[4, 1, 1, 1]));

        // Test positive shift (upper diagonal access)
        let x_shift_pos = af::shift(&x, &[1, 0, 0, 0]);
        let mut result_pos = vec![0.0f32; 4];
        x_shift_pos.host(&mut result_pos);
        println!("Shift +1: {:?} -> {:?}", x_host, result_pos);

        // ArrayFire shift +1 should be [4,1,2,3] (circular)
        let expected_pos = vec![4.0, 1.0, 2.0, 3.0];
        for (i, (&actual, &exp)) in result_pos.iter().zip(expected_pos.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "Shift +1 mismatch at position {}: got {}, expected {}",
                i,
                actual,
                exp
            );
        }

        // Test negative shift (lower diagonal access)
        let x_shift_neg = af::shift(&x, &[-1, 0, 0, 0]);
        let mut result_neg = vec![0.0f32; 4];
        x_shift_neg.host(&mut result_neg);
        println!("Shift -1: {:?} -> {:?}", x_host, result_neg);

        // ArrayFire shift -1 should be [2,3,4,1] (circular)
        let expected_neg = vec![2.0, 3.0, 4.0, 1.0];
        for (i, (&actual, &exp)) in result_neg.iter().zip(expected_neg.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "Shift -1 mismatch at position {}: got {}, expected {}",
                i,
                actual,
                exp
            );
        }
    }

    #[test]
    fn test_spmv_with_different_vector() {
        // Use exact same setup as test_spmv_consistency that fails
        let offsets = vec![-1, 0, 1];
        let diags_host = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];
        /*
        Row 0: [2.0, -10.0, 0.0, 0.0]
        Row 1: [-10.0, 20.0, -10.0, 0.0]
        Row 2: [0.0, -10.0, 20.0, -10.0]
        Row 3: [0.0, 0.0, -10.0, 20.0]

         */
        let x_test = vec![1.0f32, 2.0, 3.0, 4.0]; // Same as failing test

        // Manual CPU calculation
        let mut y_manual = vec![0.0f32; 4];
        for i in 0..4 {
            for (k, &off) in offsets.iter().enumerate() {
                let j = i as i32 + off;
                if j >= 0 && j < 4 {
                    y_manual[i] += diags_host[k][i] * x_test[j as usize];
                }
            }
        }

        // GPU calculation
        let (diags_gpu, _masks_gpu) = upload_banded_f32(&diags_host, &offsets);
        let x = Array::new(&x_test, Dim4::new(&[4, 1, 1, 1]));
        let y_gpu_array = banded_spmv_f32(&diags_gpu, &offsets, &x);
        let mut y_gpu = vec![0.0f32; 4];
        y_gpu_array.host(&mut y_gpu);

        println!("Test vector: {:?}", x_test);
        println!("Manual CPU result: {:?}", y_manual);
        println!("GPU result:        {:?}", y_gpu);

        // This should match the failing test results
        // Expected from manual calculation: [2*1 + (-10)*2, (-10)*1 + 20*2 + (-10)*3, etc.]

        for i in 0..4 {
            assert!(
                (y_manual[i] - y_gpu[i]).abs() < 1e-6,
                "GPU SpMV mismatch at position {}: CPU={}, GPU={}",
                i,
                y_manual[i],
                y_gpu[i]
            );
        }
    }

    #[test]
    fn test_manual_vs_gpu_spmv() {
        let offsets = vec![-1, 0, 1];
        let diags_host = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];

        let x_host = vec![1.0f32, 1.0, 1.0, 1.0];

        // Manual CPU calculation
        let mut y_manual = vec![0.0f32; 4];
        for i in 0..4 {
            for (k, &off) in offsets.iter().enumerate() {
                let j = i as i32 + off;
                if j >= 0 && j < 4 {
                    y_manual[i] += diags_host[k][i] * x_host[j as usize];
                }
            }
        }

        // GPU calculation
        let (diags_gpu, _masks_gpu) = upload_banded_f32(&diags_host, &offsets);
        let x = Array::new(&x_host, Dim4::new(&[4, 1, 1, 1]));
        let y_gpu_array = banded_spmv_f32(&diags_gpu, &offsets, &x);
        let mut y_gpu = vec![0.0f32; 4];
        y_gpu_array.host(&mut y_gpu);

        println!("Manual CPU result: {:?}", y_manual);
        println!("GPU result:        {:?}", y_gpu);

        // Compare - GPU should match CPU exactly
        for i in 0..4 {
            let diff = (y_manual[i] - y_gpu[i]).abs();
            if diff > 1e-6 {
                println!(
                    "Boundary error at position {}: CPU={}, GPU={}, diff={}",
                    i, y_manual[i], y_gpu[i], diff
                );
            }
            assert!(
                (y_manual[i] - y_gpu[i]).abs() < 1e-6,
                "GPU SpMV mismatch at position {}: CPU={}, GPU={}",
                i,
                y_manual[i],
                y_gpu[i]
            );
        }
    }
}
