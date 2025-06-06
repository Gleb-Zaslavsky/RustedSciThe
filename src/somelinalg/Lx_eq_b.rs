use faer::col::{Col, ColRef};

use faer::mat::{Mat, MatRef};

use faer::linalg::solvers::Solve;
use faer::sparse::SparseColMat;
use faer_gmres::JacobiPreconLinOp;
use faer_gmres::restarted_gmres;
use sprs::linalg::bicgstab::BiCGSTAB;
use sprs::{CsMat, CsVec};
pub fn solve_csmat(
    A: &CsMat<f64>,
    b: &CsVec<f64>,
    tol: f64,
    max_iter: usize,
    x0: &CsVec<f64>,
) -> Option<CsVec<f64>> {
    let (n, m) = A.shape();
    assert_eq!(n, m, "matrix must be square");
    let _vector: Vec<usize> = (0..n).collect();
    //  let x0:CsVec<f64> = CsVecI::new(n, _vector, vec![1e-11;n]);
    let res = BiCGSTAB::<'_, f64, _, _>::solve(A.view(), x0.view(), b.view(), tol, max_iter);

    match res {
        Ok(res) => {
            let x = res.x();

            Some(x.to_owned())
        }
        Err(e) => {
            println!("Error: {:?}", e);
            panic!("Error while solving linear system ",);
        }
    }
}

pub fn solve_sys_SparseColMat(
    A: &SparseColMat<usize, f64>,
    b: MatRef<f64>,
    tol: f64,
    _max_iter: usize,
    x_0: &Col<f64>,
) -> Option<Col<f64>> {
    let (n, m) = A.shape();
    assert_eq!(n, m, "matrix must be square");

    let _vec_of_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let jacobi_pre = JacobiPreconLinOp::new(A.as_ref());
    let mut x: Mat<f64> = x_0.as_mat().to_owned();
    //    println!("A, {:?}", &A);
    //    let mut x: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);

    // let res = gmres(A.as_ref(), b.as_ref(), x.as_mut(), _max_iter, tol,  Some(&jacobi_pre) );
    let res = restarted_gmres(
        A.as_ref(),
        b,
        x.as_mut(),
        1500,
        1500,
        tol,
        Some(&jacobi_pre),
    );
    match res {
        Ok(res) => {
            let _err = res.0;
            let res_vec: Vec<f64> = x.row_iter().map(|x| x[0]).collect();

            let res = ColRef::from_slice(res_vec.as_slice()).to_owned();
            Some(res)
        }
        Err(e) => {
            println!("Error: {:?}", e);
            println!("gmres not covered!");
            let A = A;
            let LU = A.sp_lu().unwrap();
            let b: MatRef<f64> = b;
            let _res: Mat<f64> = LU.solve(b); // TODO wh
            let res_vec: Vec<f64> = x.row_iter().map(|x| x[0]).collect();

            let res = ColRef::from_slice(res_vec.as_slice()).to_owned();
            Some(res)
        }
    }
}
