#![cfg(feature = "arrayfire")]
use arrayfire as af;
use af::{Array, Dim4};
use crate::somelinalg::iterative_solvers_gpu::bicgstab::bicgstab_matrix_api::sparsecol_to_banded;
use crate::somelinalg::iterative_solvers_gpu::gmres::gmres_with_preconditioners::solve_banded_gmres_f32;
use crate::somelinalg::iterative_solvers_gpu::bicgstab::bicgstab_with_preconditioneer::PreconditionerType;
use faer::sparse::SparseColMat;

#[allow(non_snake_case)]
pub fn gmres_solver(
    A: SparseColMat<usize, f64>,
    b: Vec<f64>,
    x0: Vec<f64>,
    tol: f64,
    max_iter: usize,
    restart: usize,
    preconditioner: PreconditionerType,
) -> Result<(Vec<f64>, usize, f32), Box<dyn std::error::Error>> {
    let (offsets, diags_host) = sparsecol_to_banded(&A);
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let x0_f32: Vec<f32> = x0.iter().map(|&x| x as f32).collect();
    let b_array = Array::new(&b_f32, Dim4::new(&[b_f32.len() as u64, 1, 1, 1]));
    let x0_array = Array::new(&x0_f32, Dim4::new(&[x0_f32.len() as u64, 1, 1, 1]));

    let (x_result, iter, res) = solve_banded_gmres_f32(
        A.nrows(),
        &offsets,
        &diags_host,
        &b_array,
        Some(&x0_array),
        tol as f32,
        max_iter,
        restart,
        preconditioner,
    );

    let mut x_host = vec![0.0f32; x_result.elements()];
    x_result.host(&mut x_host);
    let x_f64: Vec<f64> = x_host.iter().map(|&x| x as f64).collect();

    Ok((x_f64, iter, res))
}

#[cfg(all(test, feature = "cuda"))]
mod integration_tests {
    use super::*;
    use faer::{
        col::Col,
        prelude::Solve,
        sparse::{SparseColMat, SymbolicSparseColMat},
    };

    #[test]
    fn test_gmres_solver_on_diagonal() {
        let col_ptr = vec![0, 1, 2, 3];
        let row_idx = vec![0, 1, 2];
        let values = vec![4.0, 4.0, 4.0];
        let symbolic = SymbolicSparseColMat::new_checked(3, 3, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);

        let b = vec![8.0f64, 8.0, 8.0];
        let x0 = vec![0.0f64; 3];

        let (x, iter, res) =
            gmres_solver(mat, b, x0, 1e-6, 100, 30, PreconditionerType::Vanilla).expect("solver ok");

        for xi in x {
            assert!((xi - 2.0).abs() < 1e-4);
        }
        assert!(iter <= 5);
        assert!(res < 1e-3);
    }

    #[test]
    fn test_gmres_solver_on_tridiagonal() {
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let symbolic = SymbolicSparseColMat::new_checked(4, 4, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);
        
        let lu = mat.sp_lu().unwrap();
        let b_col: Col<f64> = Col::from_iter(b.clone().iter().map(|x| *x));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();
        let lu_solution = lu.solve(b_mat);
        let x_lu: Vec<f64> = lu_solution.row_iter().map(|x| x[0]).collect();

        let x0 = vec![0.7f64; 4];

        let (x, iter, res) = gmres_solver(
            mat.clone(),
            b,
            x0,
            1e-6,
            1500,
            30,
            PreconditionerType::GS {
                sweeps: 3,
                symmetric: false,
            },
        )
        .expect("solver ok");
        
        println!("GMRES: x = {:?}", x);
        println!("LU solution {:?}", x_lu);
        println!("iter = {:?}, res = {}", iter, res);
        
        for (xi, x_lui) in x.iter().zip(x_lu.iter()) {
            assert!((xi - x_lui).abs() < 1e-3, "xi={} xt={}", xi, x_lui);
        }
    }

    #[test]
    fn test_gmres_solver_on_tridiagonal2() {
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![
            2.0, -10.0, -10.0, 20.0, -10.0, -10.0, 20.0, -10.0, -10.0, 20.0,
        ];
        let symbolic = SymbolicSparseColMat::new_checked(4, 4, col_ptr, None, row_idx);
        let mat = SparseColMat::new(symbolic, values);
        
        let lu = mat.sp_lu().unwrap();
        let b_col: Col<f64> = Col::from_iter(b.clone().iter().map(|x| *x));
        let binding = b_col.clone();
        let b_mat = binding.as_mat();
        let lu_solution = lu.solve(b_mat);
        let x_lu: Vec<f64> = lu_solution.row_iter().map(|x| x[0]).collect();

        let x0 = vec![0.7f64; 4];

        let (x, iter, res) = gmres_solver(
            mat.clone(),
            b,
            x0,
            1e-7,
            1500,
            30,
            PreconditionerType::GS {
                sweeps: 3,
                symmetric: false,
            },
        )
        .expect("solver ok");
        
        println!("GMRES: x = {:?}", x);
        println!("LU solution {:?}", x_lu);
        println!("iter = {:?}, res = {}", iter, res);
        
        for (xi, x_lui) in x.iter().zip(x_lu.iter()) {
            assert!((xi - x_lui).abs() < 1e-3, "xi={} xt={}", xi, x_lui);
        }
    }
}