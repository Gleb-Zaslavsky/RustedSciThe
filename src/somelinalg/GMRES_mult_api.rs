use col::ColBatch;
use faer;
use faer::col;
use faer::mat::Mat;
use faer::prelude::*;
use faer::sparse::SparseColMat;
use faer_gmres::gmres;
use faer_gmres::JacobiPreconLinOp;
use nalgebra::DMatrix;
use rayon::prelude::*;

// Filters out elements from a given matrix row that are below a specified tolerance.
//
// # Parameters
//
// * `mat`: A reference to a 2D matrix of type `Mat<f64>`.
// * `i`: An index representing the row to be filtered.
// * `tol`: A tolerance value. Elements with absolute values less than or equal to this value will be filtered out.
//
// # Returns
//
// A vector of tuples, where each tuple represents a non-zero element in the filtered row.
// Each tuple contains three elements: the row index, the column index, and the value of the element.

pub fn filter_zeros(mat: &Mat<f64>, i: usize, tol: f64) -> Vec<(usize, usize, f64)> {
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
    let (n, m) = mat.shape();
    assert_eq!(n, m, "matrix must be square");

    let mut vec_of_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let jacobi_pre = JacobiPreconLinOp::new(mat.as_ref());
    // Use rayon's parallel iterator to parallelize the loop
    let triplets: Vec<Vec<(usize, usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut x: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);
            let mut b: Mat<f64> = Mat::<f64>::new_owned_zeros(n, 1);
            b[(i, 0)] = 1.0;

            let res = gmres(
                mat.as_ref(),
                b.as_ref(),
                x.as_mut(),
                max_iter,
                tol,
                Some(&jacobi_pre),
            );

            match res {
                Ok(res) => {
                    let _err = res.0;
                    filter_zeros(&x, i, tol)
                }
                Err(e) => {
                    println!("Error: {:?}", e);
                    panic!("Error while solving linear system");
                }
            }
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
    use super::*;
    use col::ColBatch;

    #[test]
    fn test_gmres_mult() {
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
        println!("len {}", test_jac.len());
        println!("JAC: {:?},{:?}", sparse_jac, sparse_jac.shape());
        let mut b: Mat<f64> = Mat::<f64>::new_owned_zeros(20, 1);
        b[(0, 0)] = 1.0;
        let mut x = Mat::<f64>::new_owned_zeros(20, 1);
        println!("b={:?}", b);

        let (err, iters) =
            gmres(sparse_jac.as_ref(), b.as_ref(), x.as_mut(), 100, 1e-8, None).unwrap();
        println!("Result x: {:?}", x);
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert_eq!(b.shape(), (20, 1));

        let inverted = invers_Mat(sparse_jac, 1e-13, 100).unwrap();
        for j in 0..inverted.shape().1 {
            let row = inverted.values_of_col(j);
            println!("\n \n \n {:?}", row);
        }
        assert_eq!(inverted.shape(), (20, 20));
    }
}
