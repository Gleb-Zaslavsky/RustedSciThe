//use std::ops::Mul;\
use crate::somelinalg::BICGSTAB::BiCGSTAB;
use nalgebra::DMatrix;
use nalgebra::sparse::{CsCholesky, CsMatrix};
use sprs::linalg::bicgstab::BiCGSTAB as BiCGSTAB_sprs;
use sprs::{CsMat, CsVec, CsVecI};

/*
fn inverse_via_lapack(mat: &CsMatrix<f64>) -> Option<CsMatrix<f64>> {
    let (n ,m) = mat.shape();
    let a: Vec<f64> = mat.
    unsafe {dgetri(
        n,
        a,
        n,
        ipiv,
        work,
        lwork,
        info
    )}
    Some(work)
}

    */

//_________________________CSMAt________________________________________
fn filter_csmat(mat: &CsMat<f64>, epsilon: f64) -> CsMat<f64> {
    let (rows, cols) = mat.shape();
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for row in 0..rows {
        let mut nnz = 0;
        for (col, &val) in mat.outer_view(row).unwrap().iter() {
            if val.abs() >= epsilon {
                indices.push(col);
                data.push(val);
                nnz += 1;
            }
        }
        indptr.push(indptr.last().unwrap() + nnz);
    }

    CsMat::new((rows, cols), indptr, indices, data)
}

pub fn invers_csmat(mat: CsMat<f64>, tol: f64, max_iter: usize) -> Option<CsMat<f64>> {
    println!("mat = {:?}", mat);
    let mat = mat.to_csr();
    let mat = filter_csmat(&mat, tol);

    let (n, m) = mat.shape();
    assert_eq!(n, m, "matrix must be square");
    let mat = mat.to_csr();

    let _eye: DMatrix<f64> = DMatrix::identity(n, m);
    let vector: Vec<usize> = (0..n).collect();

    // let x0:CsVec<f64> = CsVecI::new(n, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]); // indices and data refers to non-zero elemets, so we create all-zero vector
    let x0: CsVec<f64> = CsVecI::new(n, vector, vec![1e-11; n]);

    let mut inverted_matrix: CsMat<f64> = CsMat::empty(sprs::CompressedStorage::CSR, n);

    for i in 0..n {
        //let b: Vec<f64> = _eye.row(i).transpose();

        let indices: Vec<usize> = (0..n).collect();
        let mut data: Vec<f64> = vec![tol; n];
        data[i] = 1.0;
        let b: CsVec<f64> = CsVecI::new(n, indices, data);
        println!("{}-th col {:?}", i, b);
        let res = BiCGSTAB::<'_, f64, _, _>::solve(mat.view(), x0.view(), b.view(), tol, max_iter);

        match res {
            Ok(res) => {
                let x = res.x();
                let nonzero_indexes = res.nonzero_indexes();
                println!("\n \n  solution: {:?}, {}", x, x.dim());
                println!("\n \n  nonzero indexes: {:?}", nonzero_indexes);

                inverted_matrix = inverted_matrix.append_outer_csvec(x);
                println!("inv mat {:?}", inverted_matrix.clone());
                //   inverted_matrix.append_outer(data, x, nonzero_indexes);
            }
            Err(_e) => {
                panic!("Error while solving linear system ",);
            }
        }
    }
    println!("inverted matrix {:?}", inverted_matrix);
    Some(inverted_matrix) //None
}
//_______________________CsMatrix_inverse____________________________
#[allow(unused)]
fn invers_csmat2(mat: CsMat<f64>, tol: f64, max_iter: usize) -> Option<CsMat<f64>> {
    let (n, m) = mat.shape();
    let mat = mat.to_csc();
    let _eye: DMatrix<f64> = DMatrix::identity(n, m);
    let vector: Vec<usize> = (0..n).collect();

    // let x0:CsVec<f64> = CsVecI::new(n, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]); // indices and data refers to non-zero elemets, so we create all-zero vector
    let x0: CsVec<f64> = CsVecI::new(n, vector, vec![1e+2; n]);
    // let inverted_matrix:CsMat<f64> = CsMatI::new( (n, m), vec![], vec![], vec![]);
    for i in 0..n {
        //let b: Vec<f64> = _eye.row(i).transpose();

        let indices: Vec<usize> = vec![i];
        let data: Vec<f64> = vec![1.0];
        let b: CsVec<f64> = CsVecI::new(n, indices, data);
        println!("{}-th col {:?}", i, b);
        let res =
            BiCGSTAB_sprs::<'_, f64, _, _>::solve(mat.view(), x0.view(), b.view(), tol, max_iter);

        match res {
            Ok(res) => {
                let x = res.x();
                println!("\n \n  solution: {:?}", x);
            }
            Err(_e) => {
                panic!("Error while solving linear system ",);
            }
        }
    }
    None
}
//_______________________CsMatrix_inverse______________________________
#[allow(dead_code)]
fn inverse_via_cholesky(mat: &CsMatrix<f64>) -> Option<CsMatrix<f64>> {
    let sparse = CsCholesky::new(&mat);
    let eye = DMatrix::identity(mat.nrows(), mat.ncols());
    match sparse.l() {
        None => None,
        Some(l) => {
            // println!("{:?}", eye);

            for i in 0..eye.nrows() {
                let b = eye.row(i).to_owned();
                let col_of_invese = l
                    .tr_solve_lower_triangular(&l.solve_lower_triangular(&b).unwrap())
                    .unwrap();
                println!("{:?}", col_of_invese);
            }
            Some(l.to_owned().into())
        }
    }
}

pub fn inverse_CsMatrix(mat: &CsMatrix<f64>) -> Option<CsMatrix<f64>> {
    //Option<CsMatrix<f64>>
    let cholesky = CsCholesky::new(&mat);

    let _flag = false;
    let L_ = cholesky.l().unwrap();

    let L_Dense: DMatrix<f64> = L_.to_owned().into();
    println!("{:?}", L_Dense.transpose());
    let inverse_L: DMatrix<f64> = L_Dense.try_inverse().unwrap();
    let L_transpose: DMatrix<f64> = L_.transpose().into();
    let inverse_L_transpose: DMatrix<f64> = L_transpose.try_inverse().unwrap();

    let result = inverse_L * (inverse_L_transpose);
    let result: CsMatrix<f64> = result.into();
    Some(result)
}
/*

*/
/*

fn main() {
    // Example sparse matrix (3x3)
    let indptr = vec![0, 2, 4, 5];
    let indices = vec![0, 2, 1, 2, 2];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mat = CsMat::new((3, 3), indptr, indices, data);

    // Example right-hand side vector
    let rhs = CsVec::new(3, vec![0, 1, 2], vec![1.0, 2.0, 3.0]);

    // Solve the linear system using BiCGStab
    let mut x = rhs.clone();
    let tol = 1e-6;
    let max_iter = 1000;
    let result = BiCGStab::new(&mat, tol, max_iter).solve(&rhs, &mut x);

    match result {
        Ok(_) => {
            println!("Solution: {:?}", x);
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}
 */

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMatI;
    #[test]
    fn test_inverse_CsMatrix() {
        let mat = DMatrix::from_vec(3, 3, vec![4.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 0.0, 3.0]);

        let mat: CsMatrix<f64> = mat.into();
        println!("{:?}", mat);
        let reult = inverse_CsMatrix(&mat);
        println!("{:?}", reult);
    }
    #[test]
    fn test_inverse_csmat() {
        let mat = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
        );

        println!("{:?} \n \n", mat);
        let tol = 1e-8;
        let max_iter = 1000;
        let res = invers_csmat(mat, tol, max_iter).unwrap();
        assert_eq!((4, 4), res.shape());
    }
    /*
        #[test]
        fn test_inverse_csmat2() {
            let data = vec!  [-0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, -0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, -0.25,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, -0.25, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -1.0, 0.0,
                 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -1.0, 0.0];
            let indptr = vec![0, 8, 16, 24, 32, 40, 48, 56, 64];
            let ind = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0,
            1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] ;
            let mat = CsMatI::new_csc(
                (8, 8), indptr, ind, data.clone() );

            let mat_DMatrix:DMatrix<f64> = DMatrix::from_column_slice(8, 8, data.as_slice());
            let inv_mat_DMatrix:DMatrix<f64> = mat_DMatrix.try_inverse().unwrap();
            println!("inv DMatrix: {:?}", inv_mat_DMatrix);
         //   println!("{:?} \n \n", mat);
            let tol = 1e-8;
            let max_iter = 1000;
            let res = invers_csmat(mat, tol, max_iter).unwrap();
            assert_eq!((4,4), res.shape());
        }
    */
}
