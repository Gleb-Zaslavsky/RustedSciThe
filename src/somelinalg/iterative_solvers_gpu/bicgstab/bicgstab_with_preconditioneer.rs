//! # BiCGStab Solver with Flexible Preconditioning
//!
//! ## Aim and General Description
//! This module implements the Biconjugate Gradient Stabilized (BiCGStab) iterative method for solving
//! large sparse linear systems Ax = b. BiCGStab is particularly effective for non-symmetric matrices
//! and provides faster convergence than basic CG methods through its stabilization mechanism.
//!
//! ## Mathematical Considerations
//! - **BiCGStab Algorithm**: Uses two-step process with intermediate vector s to improve stability
//! - **Preconditioning**: Applies M⁻¹ to accelerate convergence where M ≈ A but easier to invert
//! - **Banded Matrix Storage**: Efficient diagonal-based storage for sparse matrices with band structure
//! - **GPU Acceleration**: Leverages ArrayFire for GPU-accelerated linear algebra operations
//!
//! ## Main Components
//!
//! ### Traits and Enums
//! - `Preconditioner`: Trait defining preconditioner interface with `apply()` method
//! - `PreconditionerType`: Enum selecting preconditioner type (Vanilla, Jacobi, GS, ILU0)
//!
//! ### Preconditioner Implementations
//! - `VanillaPreconditioner`: Identity operation (no preconditioning)
//! - `JacobiPreconditioner`: Diagonal scaling with configurable sweeps
//! - `GSPreconditioner`: GPU-native Gauss-Seidel with multicolor parallelization
//! - `ILU0Preconditioner`: Incomplete LU factorization (CPU-based)
//!
//! ### Main Solver Function
//! - `solve_banded_bicgstab_flexible_f32()`: Core BiCGStab implementation with flexible preconditioning
//!
//! ## Usage Examples
//! ```rust, ignore
//! // Basic usage with Jacobi preconditioning
//! let (x, iters, residual) = solve_banded_bicgstab_flexible_f32(
//!     n, &offsets, &diags_host, &b, None, 1e-6, 500,
//!     PreconditionerType::Jacobi { sweeps: 1 }
//! );
//!
//! // GPU-native Gauss-Seidel for better convergence
//! let (x, iters, residual) = solve_banded_bicgstab_flexible_f32(
//!     n, &offsets, &diags_host, &b, None, 1e-6, 500,
//!     PreconditionerType::GS { sweeps: 2, symmetric: true }
//! );
//! ```
//!
//! ## Non-obvious Features and Tips
//! - **GPU Memory Management**: Uses ArrayFire's lazy evaluation with `af::eval!()` for optimal GPU memory usage
//! - **Preconditioner Timing**: Tracks preconditioner overhead separately for performance analysis
//! - **Multicolor GS**: Uses CUDA kernel for parallel Gauss-Seidel avoiding race conditions
//! - **Residual Checking**: Performs periodic residual norm updates (every 10 iterations) for efficiency
//! - **Breakdown Recovery**: Handles numerical breakdown with epsilon thresholds (1e-30)
//! - **Mixed Precision**: CPU-GPU transfers use f32 for performance while maintaining accuracy

#![cfg(feature = "arrayfire")]

#[cfg(feature = "cuda")]
use crate::somelinalg::iterative_solvers_gpu::bicgstab::cuda_lib_ffi::{
    flatten_diagonals, launch_multicolor_gs_fused,
};
use crate::somelinalg::iterative_solvers_gpu::bicgstab::ilu_preconditioner::{ILU0, ilu0_apply};
use crate::somelinalg::iterative_solvers_gpu::bicgstab::utils::{
    banded_spmv_f32, upload_banded_f32,
};
use af::{Array, Dim4, MatProp};
use arrayfire as af;
use std::time::Instant;

/// Trait for GPU preconditioners
pub trait Preconditioner {
    fn apply(&self, vec_gpu: &Array<f32>) -> Array<f32>;
}

/// Enum for preconditioner selection
pub enum PreconditionerType {
    Vanilla,
    Jacobi {
        sweeps: usize,
    },
    #[cfg(feature = "cuda")]
    GS {
        sweeps: usize,
        symmetric: bool,
    },
    ILU0,
}

/// No preconditioning - identity operation
pub struct VanillaPreconditioner;

impl Preconditioner for VanillaPreconditioner {
    fn apply(&self, vec_gpu: &Array<f32>) -> Array<f32> {
        vec_gpu.clone()
    }
}

/// Jacobi preconditioner using main diagonal
pub struct JacobiPreconditioner {
    inv_diag_gpu: Array<f32>,
    sweeps: usize,
}

impl JacobiPreconditioner {
    pub fn new(n: usize, offsets: &[isize], diags_host: &[Vec<f32>], sweeps: usize) -> Self {
        // Find main diagonal (offset 0)
        let main_diag_idx = offsets
            .iter()
            .position(|&o| o == 0)
            .expect("Main diagonal not found");

        // Extract and invert main diagonal
        let inv_diag: Vec<f32> = diags_host[main_diag_idx]
            .iter()
            .map(|&d| if d.abs() > 1e-12 { 1.0 / d } else { 1.0 })
            .collect();

        let inv_diag_gpu = Array::new(&inv_diag, Dim4::new(&[n as u64, 1, 1, 1]));

        Self {
            inv_diag_gpu,
            sweeps,
        }
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, vec_gpu: &Array<f32>) -> Array<f32> {
        let mut result = vec_gpu * &self.inv_diag_gpu;

        // Multiple Jacobi sweeps if requested
        for _ in 1..self.sweeps {
            result = &result * &self.inv_diag_gpu;
        }

        result
    }
}

/// ILU0 preconditioner (CPU-based)
pub struct ILU0Preconditioner {
    ilu: ILU0,
}

impl ILU0Preconditioner {
    pub fn new(n: usize, offsets: &[isize], diags_host: &[Vec<f32>]) -> Self {
        let ilu = ILU0::new(n, offsets, diags_host);
        Self { ilu }
    }
}

impl Preconditioner for ILU0Preconditioner {
    fn apply(&self, vec_gpu: &Array<f32>) -> Array<f32> {
        // Transfer to CPU
        let mut r_host = vec![0.0f32; self.ilu.n];
        vec_gpu.host(&mut r_host);

        // Apply ILU0 on CPU
        let mut z_host = vec![0.0f32; self.ilu.n];
        ilu0_apply(&self.ilu, &r_host, &mut z_host);

        // Transfer back to GPU
        Array::new(&z_host, vec_gpu.dims())
    }
}

/// GPU-native Gauss-Seidel preconditioner
#[cfg(feature = "cuda")]
pub struct GSPreconditioner {
    n: usize,
    diags_gpu: Array<f32>,
    offsets_gpu: Array<i32>,
    color_count: i32,
    sweeps: usize,
    symmetric: bool,
}

#[cfg(feature = "cuda")]
impl GSPreconditioner {
    pub fn new(
        n: usize,
        offsets: &[isize],
        diags_host: &[Vec<f32>],
        sweeps: usize,
        symmetric: bool,
    ) -> Self {
        let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let flat_diags = flatten_diagonals(diags_host);
        let diags_gpu = Array::new(
            &flat_diags,
            Dim4::new(&[(n * diags_host.len()) as u64, 1, 1, 1]),
        );
        let offsets_gpu = Array::new(
            &offsets_i32,
            Dim4::new(&[offsets_i32.len() as u64, 1, 1, 1]),
        );

        Self {
            n,
            diags_gpu,
            offsets_gpu,
            color_count: 1,
            sweeps,
            symmetric,
        }
    }
}

#[cfg(feature = "cuda")]
impl Preconditioner for GSPreconditioner {
    fn apply(&self, vec_gpu: &Array<f32>) -> Array<f32> {
        if self.sweeps == 0 {
            return vec_gpu.clone();
        }

        let x_gpu = af::constant(0.0f32, vec_gpu.dims());

        unsafe {
            let offsets_ptr = self.offsets_gpu.device_ptr() as *const i32;
            let diags_ptr = self.diags_gpu.device_ptr() as *const f32;
            let b_ptr = vec_gpu.device_ptr() as *const f32;
            let x_ptr = x_gpu.device_ptr() as *mut f32;

            for _ in 0..self.sweeps {
                launch_multicolor_gs_fused(
                    self.n as i32,
                    self.offsets_gpu.elements() as i32,
                    offsets_ptr,
                    diags_ptr,
                    b_ptr,
                    x_ptr,
                    self.color_count,
                    self.symmetric as i32,
                );
            }
        }

        x_gpu
    }
}

/// Flexible BiCGStab solver with configurable preconditioner
pub fn solve_banded_bicgstab_flexible_f32(
    n: usize,
    offsets: &[isize],
    diags_host: &Vec<Vec<f32>>,
    b: &Array<f32>,
    x0: Option<&Array<f32>>,
    tol: f32,
    max_iter: usize,
    preconditioner_type: PreconditionerType,
) -> (Array<f32>, usize, f32) {
    let total_time_begin = Instant::now();
    let mut cuda_time_total: f64 = 0.0;

    // Upload matrix data once
    let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
    let (diags_dev, _masks_dev) = upload_banded_f32(diags_host, &offsets_i32);

    // Create preconditioner based on type
    let preconditioner: Box<dyn Preconditioner> = match preconditioner_type {
        PreconditionerType::Vanilla => Box::new(VanillaPreconditioner),
        PreconditionerType::Jacobi { sweeps } => {
            Box::new(JacobiPreconditioner::new(n, offsets, diags_host, sweeps))
        }
        #[cfg(feature = "cuda")]
        PreconditionerType::GS { sweeps, symmetric } => Box::new(GSPreconditioner::new(
            n, offsets, diags_host, sweeps, symmetric,
        )),
        PreconditionerType::ILU0 => Box::new(ILU0Preconditioner::new(n, offsets, diags_host)),
    };

    // Initial guess
    let mut x = match x0 {
        Some(x0_val) => x0_val.clone(),
        None => af::constant(0.0f32, b.dims()),
    };

    // Initial residual: r = b - A*x
    let ax = banded_spmv_f32(&diags_dev, &offsets_i32, &x);
    let mut r = b - &ax;
    af::eval!(&r);

    let mut resid_norm = {
        let arr = af::sqrt(&af::dot(&r, &r, MatProp::NONE, MatProp::NONE));
        let mut tmp = [0.0f32; 1];
        arr.host(&mut tmp);
        tmp[0]
    };

    if resid_norm <= tol {
        let total_time = total_time_begin.elapsed().as_millis() as f64;
        let precond_perc = (cuda_time_total / total_time) * 100.0;
        println!("total time: {:.3} ms", total_time);
        println!("preconditioner time: {:.2} %", precond_perc);
        return (x, 0, resid_norm);
    }

    // BiCGStab setup
    let r_hat = r.clone();
    let mut rho = 1.0f32;
    let mut alpha = 1.0f32;
    let mut omega = 1.0f32;
    let mut p = af::constant(0.0f32, b.dims());
    let mut v = af::constant(0.0f32, b.dims());

    for k in 1..=max_iter {
        let rho_old = rho;
        rho = {
            let arr = af::dot(&r_hat, &r, MatProp::NONE, MatProp::NONE);
            let mut tmp = [0.0f32; 1];
            arr.host(&mut tmp);
            tmp[0]
        };

        if rho.abs() < 1e-30 {
            break;
        }

        if k == 1 {
            p = r.clone();
        } else {
            let beta = (rho / rho_old) * (alpha / omega);
            p = &r + &(&p - &(&v * omega)) * beta;
        }
        af::eval!(&p);

        let cuda_time_begin = Instant::now();
        let z = preconditioner.apply(&p);
        let cuda_time = cuda_time_begin.elapsed().as_millis() as f64;
        cuda_time_total += cuda_time;

        v = banded_spmv_f32(&diags_dev, &offsets_i32, &z);
        af::eval!(&v);

        let rhat_v = {
            let arr = af::dot(&r_hat, &v, MatProp::NONE, MatProp::NONE);
            let mut tmp = [0.0f32; 1];
            arr.host(&mut tmp);
            tmp[0]
        };

        if rhat_v.abs() < 1e-30 {
            break;
        }

        alpha = rho / rhat_v;
        let s = &r - &(&v * alpha);
        af::eval!(&s);

        let s_norm = {
            let arr = af::sqrt(&af::dot(&s, &s, MatProp::NONE, MatProp::NONE));
            let mut tmp = [0.0f32; 1];
            arr.host(&mut tmp);
            tmp[0]
        };

        if s_norm <= tol {
            x = &x + &(&z * alpha);
            af::eval!(&x);
            let total_time = total_time_begin.elapsed().as_millis() as f64;
            let precond_perc = (cuda_time_total / total_time) * 100.0;
            println!("total time: {:.3} ms", total_time);
            println!("preconditioner time: {:.2} %", precond_perc);
            return (x, k, s_norm);
        }

        let cuda_time_begin = Instant::now();
        let z_s = preconditioner.apply(&s);
        let cuda_time = cuda_time_begin.elapsed().as_millis() as f64;
        cuda_time_total += cuda_time;

        let t_vec = banded_spmv_f32(&diags_dev, &offsets_i32, &z_s);
        af::eval!(&t_vec);

        let (t_dot_s, t_dot_t) = {
            let a = af::dot(&t_vec, &s, MatProp::NONE, MatProp::NONE);
            let b = af::dot(&t_vec, &t_vec, MatProp::NONE, MatProp::NONE);
            let mut ta = [0.0f32; 1];
            let mut tb = [0.0f32; 1];
            a.host(&mut ta);
            b.host(&mut tb);
            (ta[0], tb[0])
        };

        if t_dot_t.abs() < 1e-30 {
            break;
        }

        omega = t_dot_s / t_dot_t;

        x = &x + &(&z * alpha) + &(&z_s * omega);
        af::eval!(&x);

        r = &s - &(&t_vec * omega);
        af::eval!(&r);

        if k % 10 == 0 {
            let arr = af::sqrt(&af::dot(&r, &r, MatProp::NONE, MatProp::NONE));
            let mut tmp = [0.0f32; 1];
            arr.host(&mut tmp);
            resid_norm = tmp[0];
            if resid_norm <= tol {
                let total_time = total_time_begin.elapsed().as_millis() as f64;
                let precond_perc = (cuda_time_total / total_time) * 100.0;
                println!("total time: {:.3} ms", total_time);
                println!("preconditioner time: {:.2} %", precond_perc);
                return (x, k, resid_norm);
            }
        }
    }

    (x, max_iter, resid_norm)
}
/////////////////////////////////////TESTING//////////////////////////////////////////////
#[cfg(all(test, feature = "arrayfire"))]
mod tests {
    use super::*;

    #[test]
    fn test_vanilla_preconditioner() {
        let n = 1000;
        let offsets: Vec<isize> = vec![-1, 0, 1];
        let mut diags_host = Vec::new();
        for off in &offsets {
            let mut d = vec![0.0f32; n];
            for i in 0..n {
                if *off == 0 {
                    d[i] = 3.0;
                } else {
                    d[i] = -1.0;
                }
            }
            diags_host.push(d);
        }

        let b_host: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let b = Array::new(&b_host, Dim4::new(&[n as u64, 1, 1, 1]));

        let (_x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            500,
            PreconditionerType::Vanilla,
        );

        println!("Vanilla: {} iters, residual={}", iters, res);
        assert!(res < 1e-5);
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let n = 1000;
        let offsets: Vec<isize> = vec![-1, 0, 1];
        let mut diags_host = Vec::new();
        for off in &offsets {
            let mut d = vec![0.0f32; n];
            for i in 0..n {
                if *off == 0 {
                    d[i] = 3.0;
                } else {
                    d[i] = -1.0;
                }
            }
            diags_host.push(d);
        }

        let b_host: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let b = Array::new(&b_host, Dim4::new(&[n as u64, 1, 1, 1]));

        let (_x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            500,
            PreconditionerType::Jacobi { sweeps: 1 },
        );

        println!("Jacobi: {} iters, residual={}", iters, res);
        assert!(res < 1e-5);
    }

    #[test]
    fn test_ilu0_preconditioner() {
        let n = 10000;
        let offsets: Vec<isize> = vec![-1, 0, 1];
        let mut diags_host = Vec::new();
        for off in &offsets {
            let mut d = vec![0.0f32; n];
            for i in 0..n {
                if *off == 0 {
                    d[i] = 3.0;
                } else {
                    d[i] = -1.0;
                }
            }
            diags_host.push(d);
        }

        let b_host: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let b = Array::new(&b_host, Dim4::new(&[n as u64, 1, 1, 1]));

        let (_x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            500,
            PreconditionerType::ILU0,
        );

        println!("ILU0: {} iters, residual={}", iters, res);
        assert!(res < 1e-5);
    }
}
#[cfg(all(test, feature = "cuda"))]
#[allow(non_snake_case)]
mod tests_GS {
    use super::*;

    #[test]
    fn test_gpu_native_performance() {
        let now = Instant::now();
        let n = 10000;
        let offsets: Vec<isize> = vec![-1, 0, 1];

        let mut diags_host = Vec::new();
        for off in &offsets {
            let mut d = vec![0.0f32; n];
            for i in 0..n {
                if *off == 0 {
                    d[i] = 3.0;
                } else {
                    d[i] = -1.0;
                }
            }
            diags_host.push(d);
        }

        let b_host: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let b = Array::new(&b_host, Dim4::new(&[n as u64, 1, 1, 1]));

        let (_x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            500,
            PreconditionerType::GS {
                sweeps: 2,
                symmetric: true,
            },
        );

        println!(
            "GPU-native: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-5);
    }

    #[test]
    fn test_large_matrix_gpu_native() {
        let now = Instant::now();
        let n = 5000;
        let offsets: Vec<isize> = vec![-2, -1, 0, 1, 2];

        let mut diags_host = Vec::new();
        for off in &offsets {
            let mut d = vec![0.0f32; n];
            for i in 0..n {
                if *off == 0 {
                    d[i] = 2.0;
                } else {
                    d[i] = 0.8;
                }
            }
            diags_host.push(d);
        }

        let b_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let b = Array::new(&b_host, Dim4::new(&[n as u64, 1, 1, 1]));

        let (_x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            200,
            PreconditionerType::GS {
                sweeps: 1,
                symmetric: false,
            },
        );

        println!(
            "Large GPU-native: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-4);
    }
    #[test]
    fn test_wide_band_bicgstab_f32() {
        let now = Instant::now();

        let n = 300_000usize;
        let bandwidth = 500;
        let half_band = bandwidth / 2;

        let offsets: Vec<isize> = (-half_band as isize..=half_band as isize).collect();
        let mut diags_host: Vec<Vec<f32>> = Vec::new();

        for &off in offsets.iter() {
            let mut diag = vec![0.0f32; n];
            for j in 0..n {
                let target = j as isize + off;
                if target >= 0 && (target as usize) < n {
                    if off == 0 {
                        diag[j] = 2.5;
                    } else {
                        diag[j] = 1.0 / (1.0 + off.abs() as f32);
                    }
                }
            }
            diags_host.push(diag);
        }

        let dims = af::Dim4::new(&[n as u64, 1, 1, 1]);
        let x_true_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let x_true = af::Array::new(&x_true_host, dims);

        let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let (diags_dev, _masks_dev) = upload_banded_f32(&diags_host, &offsets_i32);
        let b = banded_spmv_f32(&diags_dev, &offsets_i32, &x_true);

        let (_x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-4,
            500,
            PreconditionerType::GS {
                sweeps: 1,
                symmetric: false,
            },
        );

        println!(
            "Wide band test: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-3);
    }
}
#[cfg(all(test, feature = "cuda"))]
#[allow(non_snake_case)]
mod tests_GS_compare_with_lu {
    use super::*;
    use crate::somelinalg::iterative_solvers_gpu::bicgstab::bicgstab_matrix_api::banded_to_sparsecol;
    use faer::{col::Col, prelude::Solve};
    #[test]
    fn test_large_matrix_gpu_native_with_comparsion() {
        let now = Instant::now();
        let n = 5000;
        let offsets: Vec<isize> = vec![-2, -1, 0, 1, 2];

        let mut diags_host = Vec::new();
        for off in &offsets {
            let mut d = vec![0.0f32; n];
            for i in 0..n {
                if *off == 0 {
                    d[i] = 2.0;
                } else {
                    d[i] = 0.8;
                }
            }
            diags_host.push(d);
        }

        let b_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let b = Array::new(&b_host, Dim4::new(&[n as u64, 1, 1, 1]));

        let (x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            200,
            PreconditionerType::GS {
                sweeps: 1,
                symmetric: false,
            },
        );
        let mut x_host = vec![0.0f32; x.elements()];
        x.host(&mut x_host);
        let x_f64: Vec<f64> = x_host.iter().map(|&x| x as f64).collect();
        println!(
            "Large GPU-native: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-4);
        ///////////////////compare with LU/////////////////////
        let mat = banded_to_sparsecol(&offsets, &diags_host);
        let lu_time = Instant::now();
        let lu = mat.sp_lu().unwrap();
        let b_col = Col::from_iter(b_host.clone().iter().map(|x| *x as f64));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();

        let lu_solution = lu.solve(b_mat);
        let x_lu: Vec<f64> = lu_solution.row_iter().map(|x| x[0]).collect();
        for (i, (x_i, x_lu_i)) in x_f64.iter().zip(x_lu.iter()).enumerate() {
            if (x_i - x_lu_i).abs() > 1e-3 {
                println!("i: {}, x_i: {}, x_lu_i: {}", i, x_i, x_lu_i);
            }
        }
        println!("LU time: {:?}", lu_time.elapsed().as_millis());
    }

    #[test]
    fn test_gpu_native_performance() {
        let now = Instant::now();
        let n = 10000;
        let offsets: Vec<isize> = vec![-1, 0, 1];

        let mut diags_host = Vec::new();
        for off in &offsets {
            let mut d = vec![0.0f32; n];
            for i in 0..n {
                if *off == 0 {
                    d[i] = 3.0;
                } else {
                    d[i] = -1.0;
                }
            }
            diags_host.push(d);
        }

        let b_host: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let b = Array::new(&b_host, Dim4::new(&[n as u64, 1, 1, 1]));

        let (x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            500,
            PreconditionerType::GS {
                sweeps: 1,
                symmetric: false,
            },
        );

        println!(
            "GPU-native: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-5);

        let mut x_host = vec![0.0f32; x.elements()];
        x.host(&mut x_host);
        let x_f64: Vec<f64> = x_host.iter().map(|&x| x as f64).collect();
        println!(
            "Large GPU-native: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-4);
        ///////////////////compare with LU/////////////////////
        let mat = banded_to_sparsecol(&offsets, &diags_host);
        let lu_time = Instant::now();
        let lu = mat.sp_lu().unwrap();
        let b_col = Col::from_iter(b_host.clone().iter().map(|x| *x as f64));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();

        let lu_solution = lu.solve(b_mat);
        let x_lu: Vec<f64> = lu_solution.row_iter().map(|x| x[0]).collect();
        for (i, (x_i, x_lu_i)) in x_f64.iter().zip(x_lu.iter()).enumerate() {
            if (x_i - x_lu_i).abs() > 1e-3 {
                println!("i: {}, x_i: {}, x_lu_i: {}", i, x_i, x_lu_i);
            }
        }
        println!("LU time: {:?}", lu_time.elapsed().as_millis());
    }

    #[test]
    fn test_wide_band_bicgstab_f32() {
        let now = Instant::now();

        let n = 10_000usize;
        let bandwidth = 50;
        let half_band = bandwidth / 2;

        let offsets: Vec<isize> = (-half_band as isize..=half_band as isize).collect();
        let mut diags_host: Vec<Vec<f32>> = Vec::new();

        for &off in offsets.iter() {
            let mut diag = vec![0.0f32; n];
            for j in 0..n {
                let target = j as isize + off;
                if target >= 0 && (target as usize) < n {
                    if off == 0 {
                        diag[j] = 2.5;
                    } else {
                        diag[j] = 1.0 / (1.0 + off.abs() as f32);
                    }
                }
            }
            diags_host.push(diag);
        }

        let dims = af::Dim4::new(&[n as u64, 1, 1, 1]);
        let x_true_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let x_true = af::Array::new(&x_true_host, dims);

        let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let (diags_dev, _masks_dev) = upload_banded_f32(&diags_host, &offsets_i32);
        let b = banded_spmv_f32(&diags_dev, &offsets_i32, &x_true);

        let (x, iters, res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-7,
            500,
            PreconditionerType::Jacobi { sweeps: 3 }, //GS { sweeps: 1, symmetric: false }
        );

        println!(
            "Wide band test: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-3);

        let mut x_host = vec![0.0f32; x.elements()];
        x.host(&mut x_host);
        let x_f64: Vec<f64> = x_host.iter().map(|&x| x as f64).collect();
        println!(
            "Large GPU-native: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-4);

        ///////////////////compare with LU/////////////////////
        let mat = banded_to_sparsecol(&offsets, &diags_host);
        let lu_time = Instant::now();
        let lu = mat.sp_lu().unwrap();

        let mut b_host = vec![0.0f32; b.elements()];
        b.host(&mut b_host);
        let b_f64: Vec<f64> = b_host.iter().map(|&x| x as f64).collect();

        let b_col = Col::from_iter(b_f64.clone().iter().map(|x| *x as f64));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();

        let lu_solution = lu.solve(b_mat);
        let x_lu: Vec<f64> = lu_solution.row_iter().map(|x| x[0]).collect();
        let mut failures: i32 = 0;
        let mut max_diff = 0.0;
        for (i, (x_i, x_lu_i)) in x_f64.iter().zip(x_lu.iter()).enumerate() {
            let diff = (x_i - x_lu_i).abs();
            if (diff / x_i) > max_diff {
                max_diff = diff / x_i;
            }
            if diff > 1e-3 {
                println!("i: {}, x_i: {}, x_lu_i: {}", i, x_i, x_lu_i);
                failures += 1;
            }
        }
        println!(
            "comparsion failed in {} points, max relative difference {}",
            failures, max_diff
        );
        println!("LU time: {:?}", lu_time.elapsed().as_millis());
        // let us find out how good is solution
        let r_lu = mat.clone() * lu_solution - b_mat;
        let r_lu: Vec<f64> = r_lu.row_iter().map(|x| x[0]).collect();

        let x_col = Col::from_iter(x_f64.clone().iter().map(|x| *x as f64));
        let binding = x_col.clone();
        let x_gpu = binding.as_mat();
        let r_gpu = mat * x_gpu - b_mat;
        let r_gpu: Vec<f64> = r_gpu.row_iter().map(|x| x[0]).collect();
        for (i, (r_lu_i, r_gpu_i)) in r_lu.iter().zip(r_gpu.iter()).enumerate() {
            let diff = (r_lu_i - r_gpu_i).abs();
            if diff > 1e-3 {
                println!("i: {}, r_lu_i: {}, r_gpu_i: {}", i, r_lu_i, r_gpu_i);
            }
        }
    }
    #[test]
    fn bicgstab_step_by_step_debug() {
        let offsets = vec![-1, 0, 1];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];
        let b_f32 = vec![1.0f32, 0.0, 0.0, 1.0];
        let b_array = Array::new(&b_f32, Dim4::new(&[4, 1, 1, 1]));

        // Test with very loose tolerance to see if algorithm converges at all
        let (x, iters, res) = solve_banded_bicgstab_flexible_f32(
            4,
            &offsets,
            &diags,
            &b_array,
            None,
            1e-2,
            500,
            PreconditionerType::Vanilla,
        );

        println!("Loose tolerance - iters: {}, residual: {}", iters, res);

        let mut x_host = vec![0.0f32; 4];
        x.host(&mut x_host);
        println!("Solution: {:?}", x_host);

        // Manual residual check
        let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let (diags_dev, _masks_dev) = upload_banded_f32(&diags, &offsets_i32);
        let ax = banded_spmv_f32(&diags_dev, &offsets_i32, &x);
        let r_manual = &b_array - &ax;

        let mut r_host = vec![0.0f32; 4];
        r_manual.host(&mut r_host);
        println!("Manual residual: {:?}", r_host);

        let manual_norm = r_host.iter().map(|&x| x * x).sum::<f32>().sqrt();
        println!("Manual residual norm: {}", manual_norm);
    }

    #[test]
    fn test_spmv_consistency() {
        let offsets = vec![-1, 0, 1];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];

        let x_test = vec![1.0f32, 1.0, 1.0, 1.0];
        let x_array = Array::new(&x_test, Dim4::new(&[4, 1, 1, 1]));

        // Method 1: GPU banded_spmv_f32
        let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let (diags_dev, _masks_dev) = upload_banded_f32(&diags, &offsets_i32);
        let y_gpu = banded_spmv_f32(&diags_dev, &offsets_i32, &x_array);

        let mut y_gpu_host = vec![0.0f32; 4];
        y_gpu.host(&mut y_gpu_host);

        // Method 2: faer sparse matrix
        let mat = banded_to_sparsecol(&offsets, &diags);
        let x_f64: Vec<f64> = x_test.iter().map(|&x| x as f64).collect();
        let x_col = Col::from_iter(x_f64.iter().map(|&x| x));
        let x_mat = x_col.as_mat();
        let y_faer = mat * x_mat;
        let y_faer_host: Vec<f64> = y_faer.row_iter().map(|x| x[0]).collect();

        println!("GPU banded_spmv result: {:?}", y_gpu_host);
        println!("Faer sparse result:     {:?}", y_faer_host);

        // Compare results - should be identical
        for i in 0..4 {
            let diff = (y_gpu_host[i] as f64 - y_faer_host[i]).abs();
            if diff > 1e-6 {
                println!(
                    "INCONSISTENCY at position {}: GPU={}, Faer={}, diff={}",
                    i, y_gpu_host[i], y_faer_host[i], diff
                );
            }
        }

        // Manual calculation for verification
        let mut y_manual = vec![0.0f32; 4];
        for i in 0..4 {
            for (k, &off) in offsets.iter().enumerate() {
                let j = i as i32 + off as i32;
                if j >= 0 && j < 4 {
                    y_manual[i] += diags[k][i] * x_test[j as usize];
                }
            }
        }
        println!("Manual calculation:     {:?}", y_manual);
    }

    #[test]
    fn debug_bicgstab_residual_calculation() {
        let offsets = vec![-1, 0, 1];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];
        let b_f32 = vec![1.0f32, 0.0, 0.0, 1.0];
        let b_array = Array::new(&b_f32, Dim4::new(&[4, 1, 1, 1]));

        let (x, iters, solver_residual) = solve_banded_bicgstab_flexible_f32(
            4,
            &offsets,
            &diags,
            &b_array,
            None,
            1e-6,
            500,
            PreconditionerType::Vanilla,
        );

        println!(
            "BiCGStab: {} iters, reported residual: {}",
            iters, solver_residual
        );

        let mut x_host = vec![0.0f32; 4];
        x.host(&mut x_host);
        println!("Solution: {:?}", x_host);

        // Check residual using SAME method as BiCGStab internally
        let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let (diags_dev, _masks_dev) = upload_banded_f32(&diags, &offsets_i32);
        let ax_internal = banded_spmv_f32(&diags_dev, &offsets_i32, &x);
        let r_internal = &b_array - &ax_internal;

        let mut r_internal_host = vec![0.0f32; 4];
        r_internal.host(&mut r_internal_host);
        let internal_norm = r_internal_host.iter().map(|&x| x * x).sum::<f32>().sqrt();

        println!(
            "Internal residual (same as BiCGStab): {:?}",
            r_internal_host
        );
        println!("Internal residual norm: {}", internal_norm);

        // Check residual using external verification method
        let mat = banded_to_sparsecol(&offsets, &diags);
        let x_f64: Vec<f64> = x_host.iter().map(|&x| x as f64).collect();
        let b_f64: Vec<f64> = b_f32.iter().map(|&x| x as f64).collect();

        let x_col = Col::from_iter(x_f64.iter().map(|&x| x));
        let b_col = Col::from_iter(b_f64.iter().map(|&x| x));
        let r_external = b_col.as_mat() - mat * x_col.as_mat();
        let r_external_host: Vec<f64> = r_external.row_iter().map(|x| x[0]).collect();
        let external_norm = r_external_host.iter().map(|&x| x * x).sum::<f64>().sqrt();

        println!("External residual (verification): {:?}", r_external_host);
        println!("External residual norm: {}", external_norm);

        // The key question: are these the same?
        println!(
            "Residual norm difference: {}",
            (internal_norm as f64 - external_norm).abs()
        );
    }

    #[test]
    fn verify_matrix_reconstruction() {
        let offsets = vec![-1, 0, 1];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];

        // Test vector
        let x_test = vec![1.0f32, 2.0, 3.0, 4.0];

        // Method 1: Direct banded SpMV
        let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let (diags_dev, _masks_dev) = upload_banded_f32(&diags, &offsets_i32);
        let x_array = Array::new(&x_test, Dim4::new(&[4, 1, 1, 1]));
        let y_banded = banded_spmv_f32(&diags_dev, &offsets_i32, &x_array);
        let mut y_banded_host = vec![0.0f32; 4];
        y_banded.host(&mut y_banded_host);

        // Method 2: Reconstructed sparse matrix
        let mat = banded_to_sparsecol(&offsets, &diags);
        let x_f64: Vec<f64> = x_test.iter().map(|&x| x as f64).collect();
        let x_col = Col::from_iter(x_f64.iter().map(|&x| x));
        let y_sparse = mat.clone() * x_col.as_mat();
        let y_sparse_host: Vec<f64> = y_sparse.row_iter().map(|x| x[0]).collect();

        println!("Test vector: {:?}", x_test);
        println!("Banded SpMV result: {:?}", y_banded_host);
        println!("Sparse SpMV result: {:?}", y_sparse_host);

        // Check if matrices are equivalent
        for i in 0..4 {
            let diff = (y_banded_host[i] as f64 - y_sparse_host[i]).abs();
            if diff > 1e-6 {
                println!(
                    "MATRIX MISMATCH at row {}: banded={}, sparse={}, diff={}",
                    i, y_banded_host[i], y_sparse_host[i], diff
                );
            }
        }

        // Print actual matrix for verification
        println!("\nReconstructed matrix:");
        for i in 0..4 {
            let mut row = vec![0.0; 4];
            let unit = vec![0.0, 0.0, 0.0, 0.0];
            let mut unit_i = unit.clone();
            unit_i[i] = 1.0;
            let unit_col = Col::from_iter(unit_i.iter().map(|&x| x));
            let col_result = mat.clone() * unit_col.as_mat();
            for j in 0..4 {
                row[j] = col_result.row_iter().nth(j).unwrap()[0];
            }
            println!("{:?}", row);
        }
    }

    #[test]
    fn small_example() {
        let offsets = vec![-1, 0, 1];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];
        let mat = banded_to_sparsecol(&offsets, &diags);
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        let b_col = Col::from_iter(b_f32.clone().iter().map(|x| *x as f64));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();
        let b_array = Array::new(&b_f32, Dim4::new(&[b_f32.len() as u64, 1, 1, 1]));
        let n = 4;
        let (x, _iters, _res) = solve_banded_bicgstab_flexible_f32(
            n,
            &offsets,
            &diags,
            &b_array,
            None,
            1e-6, // Use reasonable tolerance
            500,
            PreconditionerType::Vanilla,
        );

        let mut x_host = vec![0.0f32; x.elements()];
        x.host(&mut x_host);
        let x_f64: Vec<f64> = x_host.iter().map(|&x| x as f64).collect();

        let x_col = Col::from_iter(x_f64.clone().iter().map(|x| *x as f64));
        let binding = x_col.clone();
        let x_gpu = binding.as_mat();
        let r_gpu = b_mat - mat * x_gpu;
        println!("_iter {}", _iters);
        println!("_res = {}", _res);
        let r_gpu: Vec<f64> = r_gpu.row_iter().map(|x| x[0]).collect();
        println!("residuals: {:?}", r_gpu);

        println!("solution: {:?}", x_f64);
        // Check that residual norm meets the solver tolerance
        let residual_norm: f64 = r_gpu.iter().map(|&x| x * x).sum::<f64>().sqrt();
        println!("Final residual norm: {}", residual_norm);

        // Assert based on solver tolerance, not arbitrary strict values
        assert!(
            residual_norm < 1e-5,
            "Residual norm {} exceeds expected tolerance",
            residual_norm
        );

        // Verify solution quality by checking individual residuals are reasonable
        for (i, &r_i) in r_gpu.iter().enumerate() {
            assert!(
                r_i.abs() < 1e-4,
                "Residual at position {} is {}, too large",
                i,
                r_i
            );
        }
    }
}
