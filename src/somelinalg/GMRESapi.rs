use col::ColBatch;
use faer;
use faer::col;
use faer::mat::Mat;
use faer::prelude::*;
use faer::sparse::SparseColMat;
use faer_gmres::gmres;
use nalgebra::DMatrix;
use log::info;
fn filter_zeros(mat: &Mat<f64>, i: usize, tol: f64) -> Vec<(usize, usize, f64)> {
    let mut vec_of_triplets: Vec<(usize, usize, f64)> = Vec::new();
    for (j, row) in mat.row_iter().enumerate() {
        if row[0].abs() >= tol {
            vec_of_triplets.push((j, i, row[0]));
        }
    }
    vec_of_triplets
}
#[allow(dead_code)]
fn get_i_row_as_Mat(mat: &SparseColMat<usize, f64>, i: usize) -> Mat<f64> {
    let (_R, C) = mat.shape();

    let mut row_data: Vec<f64> = Vec::new();
    for j in 0..C {
        row_data.push(*mat.get(i, j).to_owned().unwrap_or(&0.0));
    }
    let Mat_i = mat::from_column_major_slice::<f64>(row_data.as_slice(), C, 1).to_owned();
    Mat_i
}
pub fn invers_Mat(
    mat: SparseColMat<usize, f64>,
    tol: f64,
    max_iter: usize,
) -> Option<SparseColMat<usize, f64>> {
    //  info("mat = {:?}", mat);
    //let t0 = Instant::now();
    let (n, m) = mat.shape();
    assert_eq!(n, m, "matrix must be square");

    let mut x: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);

    let mut vec_of_triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        //let b: Vec<f64> = eye.row(i).transpose();

        let mut b: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);
        b[(i, 0)] = 1.0;
        //  info("{}-th col {:?}",i, b);
        let res = gmres(mat.as_ref(), b.as_ref(), x.as_mut(), max_iter, tol, None);

        match res {
            Ok(res) => {
                let _err = res.0;
                //  info("{}-th col, _err = {:?}",i,  _err);
                let triplets_i = filter_zeros(&x, i, tol);
                vec_of_triplets.extend(triplets_i);
                //   inverted_matrix.append_outer(data, x, nonzero_indexes);
            }
            Err(e) => {
                info!("Error: {:?}", e);
                panic!("Error while solving linear system ",);
            }
        }
    }

    let inverted_matrix: SparseColMat<usize, f64> =
        SparseColMat::<usize, f64>::try_new_from_triplets(n, m, &vec_of_triplets).unwrap();
    //   info!("inverted matrix {:?}", inverted_matrix);
    Some(inverted_matrix) //None
}

pub fn dense_to_sparse(dense: DMatrix<f64>) -> SparseColMat<usize, f64> {
    let (nrows, ncols) = dense.shape();
    let mut vec_of_triplets: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..nrows {
        for j in 0..ncols {
            let value = dense[(i, j)];
            if value != 0.0 {
                vec_of_triplets.push((i, j, value));
            }
        }
    }

    let sparse: SparseColMat<usize, f64> =
        SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &vec_of_triplets).unwrap();

    sparse
}

#[cfg(test)]
mod tests {
    use col::ColBatch;

    use super::*;

    #[test]
    fn test_gmres() {
        // create faer sparse mat from triplets
        let a_test_triplets: Vec<(usize, usize, f64)> = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let a_test: SparseColMat<usize, f64> =
            SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &a_test_triplets).unwrap();

        // rhs
        let b = faer::mat![[2.0], [2.0], [2.0],];

        // init sol guess
        // Note: x is modified in-place, the result is stored in x
        let mut x = faer::mat![[0.0], [0.0], [0.0],];

        // the final None arg means do not apply left preconditioning
        let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x.as_mut(), 10, 1e-8, None).unwrap();
        info!("Result x: {:?}", x);
        info!("Error x: {:?}", err);
        info!("Iters : {:?}", iters);
        assert_eq!(x.shape(), (3, 1));
    }
    #[test]
    fn test_gmres2() {
        let test_jac = vec![
            -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.9, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.9, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        info!("{}", test_jac.len());
        let jac_DM: DMatrix<f64> = DMatrix::from_row_slice(20, 20, &test_jac.as_slice());
        //info!("JAC: {:?}", &jac_DM);
        let sparse_jac = dense_to_sparse(jac_DM);
        info!("JAC: {:?},{:?}", sparse_jac, sparse_jac.shape());
        let mut b: Mat<f64> = Mat::<f64>::new_owned_zeros(20, 1);
        b[(0, 0)] = 1.0;
        let mut x = Mat::<f64>::new_owned_zeros(20, 1);

        //   b[(0, 0)] = 1.0;
        //  let guess:Mat<f64> = Mat::<f64>::new_owned_zeros(20, 1 )
        //  .row_iter().map(|r|  r.iter().map(|mut x| x=&1e-5) ); // for_each(|r| { r.iter().map(|mut x| x=&1e-5);} ).collect();
        // info!("B: {:?}", &b);
        let (err, iters) =
            gmres(sparse_jac.as_ref(), b.as_ref(), x.as_mut(), 100, 1e-8, None).unwrap();
        info!("Result x: {:?}", x);
        info!("Error x: {:?}", err);
        info!("Iters : {:?}", iters);
        assert_eq!(b.shape(), (20, 1));

        let inverted = invers_Mat(sparse_jac, 1e-13, 100).unwrap();
        for j in 0..inverted.shape().1 {
            let row = inverted.values_of_col(j);
            info!("\n {:?}", row);
        }
        // info!("inverted: {:?}", inverted);
        assert_eq!(inverted.shape(), (20, 20));
    }
}
