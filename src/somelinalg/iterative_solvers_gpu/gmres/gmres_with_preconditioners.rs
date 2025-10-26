#![cfg(feature = "arrayfire")]
#[cfg(feature = "cuda")]
use crate::somelinalg::iterative_solvers_gpu::bicgstab::bicgstab_with_preconditioneer::GSPreconditioner;
use crate::somelinalg::iterative_solvers_gpu::bicgstab::bicgstab_with_preconditioneer::{
    ILU0Preconditioner, JacobiPreconditioner, Preconditioner, PreconditionerType,
    VanillaPreconditioner,
};
use crate::somelinalg::iterative_solvers_gpu::bicgstab::utils::{
    banded_spmv_f32, upload_banded_f32,
};
use af::Array;
use arrayfire as af;
use std::time::Instant;
#[allow(non_snake_case)]
pub fn solve_banded_gmres_f32(
    n: usize,
    offsets: &[isize],
    diags_host: &Vec<Vec<f32>>,
    b: &Array<f32>,
    x0: Option<&Array<f32>>,
    tol: f32,
    max_iter: usize,
    restart: usize,
    preconditioner_type: PreconditionerType,
) -> (Array<f32>, usize, f32) {
    let start = Instant::now();
    // Upload matrix data once
    let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
    let (diags_dev, _masks_dev) = upload_banded_f32(diags_host, &offsets_i32);

    // Create preconditioner based on type
    let mut preconditioner: Box<dyn Preconditioner> = match preconditioner_type {
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

    let m = restart.min(n);

    for outer_iter in 0..max_iter {
        // Compute initial residual r0 = b - A*x
        let ax = banded_spmv_f32(&diags_dev, &offsets_i32, &x);
        let r = b - &ax;
        let beta = {
            let arr = af::sqrt(&af::dot(&r, &r, af::MatProp::NONE, af::MatProp::NONE));
            let mut tmp = [0.0f32; 1];
            arr.host(&mut tmp);
            tmp[0]
        };

        if beta < tol {
            println!(
                "Converged at outer iteration {} in {:?}",
                outer_iter,
                start.elapsed()
            );
            return (x, outer_iter * m, beta);
        }

        // Initialize Arnoldi basis
        let mut V: Vec<Array<f32>> = Vec::with_capacity(m + 1);
        V.push(&r / beta);

        // Hessenberg matrix H (stored column-major)
        let mut H = vec![0f32; (m + 1) * m];
        let mut cs = vec![0f32; m];
        let mut sn = vec![0f32; m];
        let mut g = vec![0f32; m + 1];
        g[0] = beta;

        let mut j = 0;
        while j < m {
            // Apply preconditioner and matrix
            let z = preconditioner.apply(&V[j]);
            let mut w = banded_spmv_f32(&diags_dev, &offsets_i32, &z);

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                let hij = af::sum_all(&(&V[i] * &w)).0 as f32;
                H[i + j * (m + 1)] = hij;
                w = &w - hij * &V[i];
            }

            let h_next = {
                let arr = af::sqrt(&af::dot(&w, &w, af::MatProp::NONE, af::MatProp::NONE));
                let mut tmp = [0.0f32; 1];
                arr.host(&mut tmp);
                tmp[0]
            };
            H[(j + 1) + j * (m + 1)] = h_next;

            if h_next < 1e-12 {
                // Lucky breakdown
                break;
            }

            V.push(&w / h_next);

            // Apply previous Givens rotations
            for i in 0..j {
                let temp = cs[i] * H[i + j * (m + 1)] + sn[i] * H[(i + 1) + j * (m + 1)];
                H[(i + 1) + j * (m + 1)] =
                    -sn[i] * H[i + j * (m + 1)] + cs[i] * H[(i + 1) + j * (m + 1)];
                H[i + j * (m + 1)] = temp;
            }

            // Compute new Givens rotation
            let h_jj = H[j + j * (m + 1)];
            let h_j1j = H[(j + 1) + j * (m + 1)];
            let norm = (h_jj * h_jj + h_j1j * h_j1j).sqrt();

            if norm > 1e-12 {
                cs[j] = h_jj / norm;
                sn[j] = h_j1j / norm;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }

            // Apply new rotation to H and g
            H[j + j * (m + 1)] = cs[j] * h_jj + sn[j] * h_j1j;
            H[(j + 1) + j * (m + 1)] = 0.0;

            let temp = cs[j] * g[j];
            g[j + 1] = -sn[j] * g[j];
            g[j] = temp;

            let residual = g[j + 1].abs();
            j += 1;

            if residual < tol {
                println!(
                    "Converged at outer iteration {} in {:?}",
                    outer_iter,
                    start.elapsed()
                );
                // Solve upper triangular system and update solution
                x = update_solution_gmres(&x, &V, &H, &g, j, m + 1, &mut preconditioner);
                return (x, outer_iter * m + j, residual);
            }
        }

        // Update solution at restart
        x = update_solution_gmres(&x, &V, &H, &g, j, m + 1, &mut preconditioner);
    }

    (x, max_iter, tol)
}
#[allow(non_snake_case)]
fn update_solution_gmres(
    x: &Array<f32>,
    V: &Vec<Array<f32>>,
    H: &Vec<f32>,
    g: &Vec<f32>,
    k: usize,
    ldh: usize,
    precond: &mut Box<dyn Preconditioner>,
) -> Array<f32> {
    // Solve upper triangular system H*y = g
    let mut y = vec![0f32; k];
    for i in (0..k).rev() {
        let mut sum = g[i];
        for j in (i + 1)..k {
            sum -= H[i + j * ldh] * y[j];
        }
        y[i] = sum / H[i + i * ldh];
    }

    // Update solution: x = x + M^(-1) * V * y
    let mut x_new = x.clone();
    for i in 0..k {
        let vy = y[i] * &V[i];
        let z = precond.apply(&vy);
        x_new = &x_new + &z;
    }
    x_new
}
// cargo test gmres_with_preconditioners::tests::test_wide_band_bicgstab_f3 --features cuda
#[cfg(all(test, feature = "arrayfire"))]
mod tests {
    use super::*;
    use af::{Array, Dim4};
    use std::time::Instant;

    #[test]
    fn test_wide_band_bicgstab_f32() {
        let now = Instant::now();

        let n = 300_00usize;
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

        let (_x, iters, res) = solve_banded_gmres_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            50,
            50,
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

    #[test]
    fn test_gmres_vanilla_preconditioner() {
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

        let (_x, iters, res) = solve_banded_gmres_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            50,
            50,
            PreconditionerType::Vanilla,
        );

        println!("GMRES Vanilla: {} iters, residual={}", iters, res);
        assert!(res < 1e-5);
    }

    #[test]
    fn test_gmres_jacobi_preconditioner() {
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

        let (_x, iters, res) = solve_banded_gmres_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            500,
            50,
            PreconditionerType::Jacobi { sweeps: 1 },
        );

        println!("GMRES Jacobi: {} iters, residual={}", iters, res);
        assert!(res < 1e-5);
    }

    #[test]
    fn test_gmres_small_example() {
        let offsets = vec![-1, 0, 1];
        let diags = vec![
            vec![0.0, -10.0, -10.0, -10.0],
            vec![2.0, 20.0, 20.0, 20.0],
            vec![-10.0, -10.0, -10.0, 0.0],
        ];
        let b_f32 = vec![1.0f32, 0.0, 0.0, 1.0];
        let b_array = Array::new(&b_f32, Dim4::new(&[4, 1, 1, 1]));
        let n = 4;

        let (x, iters, res) = solve_banded_gmres_f32(
            n,
            &offsets,
            &diags,
            &b_array,
            None,
            1e-6,
            20,
            10,
            PreconditionerType::Vanilla,
        );

        println!("GMRES small: {} iters, residual={}", iters, res);

        let mut x_host = vec![0.0f32; 4];
        x.host(&mut x_host);
        println!("Solution: {:?}", x_host);

        assert!(res < 1e-4);
    }

    #[test]
    fn test_gmres_performance() {
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

        let (_x, iters, res) = solve_banded_gmres_f32(
            n,
            &offsets,
            &diags_host,
            &b,
            None,
            1e-6,
            200,
            30,
            PreconditionerType::Jacobi { sweeps: 1 },
        );

        println!(
            "GMRES performance: {} iters, residual={}, time={:?}",
            iters,
            res,
            now.elapsed()
        );
        assert!(res < 1e-4);
    }
}
