use crate::somelinalg::GMRES_mult_api::{dense_to_sparse, filter_zeros};
use col::ColBatch;
use faer::mat::Mat;
use faer::prelude::*;
use faer::sparse::SparseColMat;
use faer::col;

use rayon::prelude::*;
// Import rayon prelude for parallel iterators

pub fn invers_Mat_LU(
    mat: SparseColMat<usize, f64>,
    tol: f64,
    _max_iter: usize,
) -> Option<SparseColMat<usize, f64>> {
    let (n, m) = mat.shape();
    assert_eq!(n, m, "matrix must be square");
    let LU = mat.sp_lu().unwrap();

    let mut vec_of_triplets: Vec<(usize, usize, f64)> = Vec::new();
    // Use rayon's parallel iterator to parallelize the loop
    let triplets: Vec<Vec<(usize, usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let _x: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);
            let mut b: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);
            b[(i, 0)] = 1.0;

            let res = LU.solve(b);
            filter_zeros(&res, i, tol)
        })
        .collect();

    // Flatten the vector of vectors into a single vector
    for triplet in triplets {
        vec_of_triplets.extend(triplet);
    }

    let inverted_matrix: SparseColMat<usize, f64> =
        SparseColMat::<usize, f64>::try_new_from_triplets(n, m, &vec_of_triplets).unwrap();
    Some(inverted_matrix)
}
// only triangular matrix
pub fn solve_with_upper_triangular(
    mat: SparseColMat<usize, f64>,
    tol: f64,
    _max_iter: usize,
) -> Option<SparseColMat<usize, f64>> {
    println!("mat = {:?}", mat.row_indices());
    for j in 0..mat.shape().0 {
        let row_indices_of_col_raw = mat.row_indices_of_col_raw(j);
        println!("{}-th col {:?}", j, row_indices_of_col_raw);
        row_indices_of_col_raw.into_iter().for_each(|k| {
            if *k < j {
                println!("k<j, k={},j={},element = {}", k, j, mat[(*k, j)])
            }
        });
        //  println!("\n \n \n {}th row = {:?}",j, row);
    }
    let (n, m) = mat.shape();

    assert_eq!(n, m, "matrix must be square");

    let mat = mat.to_sorted().unwrap();
    let mut vec_of_triplets: Vec<(usize, usize, f64)> = Vec::new();
    // Use rayon's parallel iterator to parallelize the loop
    let triplets: Vec<Vec<(usize, usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut b: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);
            b[(i, 0)] = 1.0;

            mat.sp_solve_upper_triangular_in_place(&mut b);
            //  println!("{}-th col, b = {:?}, shape = {:?}",i,  &b, &b.shape());
            filter_zeros(&b, i, tol)
        })
        .collect();

    // Flatten the vector of vectors into a single vector
    for triplet in triplets {
        vec_of_triplets.extend(triplet);
    }

    let inverted_matrix: SparseColMat<usize, f64> =
        SparseColMat::<usize, f64>::try_new_from_triplets(n, m, &vec_of_triplets).unwrap();
    Some(inverted_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use col::ColBatch;
    use faer_gmres::gmres;
    use nalgebra::DMatrix;
    #[test]
    fn test_Lusolver() {

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

        println!("{}", test_jac.len());
        let jac_DM: DMatrix<f64> = DMatrix::from_row_slice(20, 20, &test_jac.as_slice());
        let sparse_jac = dense_to_sparse(jac_DM);
        println!("JAC: {:?},{:?}", sparse_jac, sparse_jac.shape());
        let mut b: Mat<f64> = Mat::<f64>::new_owned_zeros(20, 1);
        b[(0, 0)] = 1.0;
        let mut x = Mat::<f64>::new_owned_zeros(20, 1);

        let (_err,_iterss) =
            gmres(sparse_jac.as_ref(), b.as_ref(), x.as_mut(), 100, 1e-8, None).unwrap();
        //    println!("Result x: {:?}", x);
        //    println!("Error x: {:?}", err);
        //   println!("Iters : {:?}", iters);
        assert_eq!(b.shape(), (20, 1));

        let inverted1 = invers_Mat_LU(sparse_jac.clone(), 1e-13, 100).unwrap();
        // println!("unverted by LU ");
        for j in 0..inverted1.shape().1 {
            let _row = inverted1.values_of_col(j);
            //  println!("\n \n \n {}th row = {:?}",j, row);
        }
        assert_eq!(inverted1.shape(), (20, 20));

        //  let inverted2 = solve_with_upper_triangular(sparse_jac, 1e-13, 100).unwrap();
        //  println!("unverted by upper triang ");
        // for j in 0..inverted2.shape().1 {
        //    let row = inverted2.values_of_col(j);
        //  println!("\n \n \n {}th row = {:?}",j, row);
        //   }
        //  assert_eq!(inverted2.shape(), (20, 20));
        //assert_eq!(inverted1, inverted2);
    }
}
