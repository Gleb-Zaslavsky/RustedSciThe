#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rayon::prelude::*; //parallel processing library


///////////////////////////////////////////////
///  A COLLECTION OF SOLVERS FOR BANDED LINEAR EQUATIONS
////////////////////////////////////////////////
///           LU decomposition
////////////////////////////////////////////////
// basic easiest solver for LU decomposition: no banded structure, no pivoting
fn lu_decomposition(a: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = a.nrows();
    let mut l = DMatrix::zeros(n, n);
    let mut u = DMatrix::zeros(n, n);
    for i in 0..n {
        // Upper Triangular
        for k in i..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[(i, j)] * u[(j, k)];
            }
            u[(i, k)] = a[(i, k)] - sum;
        }
        // Lower Triangular
        for k in i..n {
            if i == k {
                l[(i, i)] = 1.0; // Diagonal as 1
            } else {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += l[(k, j)] * u[(j, i)];
                }
                l[(k, i)] = (a[(k, i)] - sum) / u[(i, i)];
            }
        }
    }

    (l, u)
}
// slightlu different version of easiest solver for LU decomposition: no banded structure, no pivoting
fn lu_decomposition2(A: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = A.nrows();
    let mut L = DMatrix::zeros(n, n);
    let mut U = A.clone();
    for (k, _col_k) in A.column_iter().enumerate() {
        L[(k, k)] = 1.0;
        for i in k + 1..n {
            println!("{} {}", i, k);
            L[(i, k)] = U[(i, k)] / U[(k, k)];
            for j in k..n {
                U[(i, j)] = U[(i, j)] - L[(i, k)] * U[(k, j)];
            }
        }
    }

    (L, U)
}

//CHECK!
fn banded_lu_decomposition(a: &DMatrix<f64>, kl: usize, ku: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = a.nrows();
    let mut l = DMatrix::zeros(n, kl + 1);
    let mut u = DMatrix::zeros(n, ku + 1);

    for i in 0..n {
        // Upper triangular part
        for j in i..std::cmp::min(n, i + ku + 1) {
            let mut sum = 0.0;
            let k_min = std::cmp::max(0, j as i32 - ku as i32) as usize;
            for k in k_min..i {
                sum += l[(i, k - i + kl)] * u[(k, j - k)];
            }
            u[(i, j - i)] = a[(i, j - i)] - sum;
        }

        // Lower triangular part
        for j in std::cmp::max(0, i as i32 - kl as i32) as usize..=i {
            if i == j {
                l[(i, 0)] = 1.0;
            } else {
                let mut sum = 0.0;
                let k_min = std::cmp::max(0, i as i32 - kl as i32) as usize;
                for k in k_min..j {
                    sum += l[(i, k - i + kl)] * u[(k, j - k)];
                }
                l[(i, j - i + kl)] = (a[(i, j - i + ku)] - sum) / u[(j, 0)];
            }
        }
    }

    (l, u)
}
// easy solver for LU decomposition: no pivoting, but band structure has been taken into account
fn banded_lu_decomposition_no_pivoting(
    A: &DMatrix<f64>,
    kl: usize,
    ku: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = A.nrows();
    let mut L = DMatrix::zeros(n, n);
    let mut U = A.clone();
    //iterate through columns
    for (k, _col_k) in A.column_iter().enumerate() {
        L[(k, k)] = 1.0;
        /* let's iterate through all rows below row k (i.e. below main diag with elements k,k) i.e j = k + 1, but only rows containing
        elements =>  we must take into account only subdiogonal, which has width of kl elements but for the last row we can't exceed the width of the matrix, so
         while iterating through all subdiogonal elements so we establish the condition min(n, k+kl+1);
        │        7  8  0 │
        │  0  0  9 10 11 │
        │  0  0  0 12 13 │ <- elements in the last row
                */
        let low_border = std::cmp::min(n, k + kl + 1);
        for i in k + 1..low_border {
            L[(i, k)] = U[(i, k)] / U[(k, k)];
            /* */
            let border = std::cmp::min(n, k + ku + 1);
            for j in k..border {
                U[(i, j)] = U[(i, j)] - L[(i, k)] * U[(k, j)];
            }
        }
    }

    (L, U)
}
// more complex solver for LU decomposition:  pivoting + band structure has been taken into account
fn banded_lu_decomposition_with_pivoting(
    A: &DMatrix<f64>,
    kl: usize,
    ku: usize,
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let n = A.nrows();
    let mut L = DMatrix::zeros(n, n);
    let mut U = A.clone();
    let mut P: Vec<usize> = (0..n).collect(); // Permutation vector


    for (k, _col_k) in A.column_iter().enumerate() {

        
        let low_border = std::cmp::min(n, k + kl + 1);
              //Find the pivot row by finding the maximum absolute value in the current row starting from the current column.
        let piv = U.view_range(k..low_border, k).icamax() + k;
        //Extract the diagonal element diag from the pivot row and column.
        let diag = U[(piv, k)];
        //Check if the diagonal element is zero. If it is, continue to the next iteration (no non-zero entries on this column).
        if diag == 0.0 {
            // No non-zero entries on this column.
            continue;
        }
      //  println!("{} {}, {}", _col_k[piv], _col_k[k], _col_k);

        if piv != k {
               
            //  p.append_permutation(i, piv);
            P.swap(k, piv);
           // Do for all rows below pivot:
            U.columns_range_mut(0..k).swap_rows(k, piv);
        }

     //   println!("{} ", U);
        L[(k, k)] = 1.0;
        for i in k + 1..low_border {
            L[(i, k)] = U[(i, k)] / U[(k, k)];
           // U[(i, k)]=0.0;
            let border = std::cmp::min(n, k + ku + 1);
            for j in k..border {
                U[(i, j)] = U[(i, j)] - L[(i, k)] * U[(k, j)];
            }
        }
    }


    (L, U, P)
}
// even more complex solver for LU decomposition:  pivoting + band structure + tiny element
fn banded_lu_decomposition_with_pivoting_and_tiny(
    A: &DMatrix<f64>,
    kl: usize,
    ku: usize,
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let n = A.nrows();
    let mut L = DMatrix::zeros(n, n);
    let mut U = A.clone();
    let mut P: Vec<usize> = (0..n).collect(); // Permutation vector
    let tiny = 1e-10;
    for k in 0..n {
        let low_border = std::cmp::min(n, k + kl + 1);
        // Partial pivoting: find the row with the largest element in the current column
        let mut max_row = k;
        for i in (k + 1)..low_border {
            if U[(i, k)].abs() > U[(max_row, k)].abs() {
                max_row = i;
            }
        }
        // Swap rows in U and update permutation vector
        if max_row != k {
            U.swap_rows(k, max_row);
            P.swap(k, max_row);
        }

        L[(k, k)] = 1.0;

        if U[(k, k)].abs() == 0.0 {
            U[(k, k)] = tiny;
        }

        for i in k + 1..low_border {
            L[(i, k)] = U[(i, k)] / U[(k, k)];
            /* */
            let border = std::cmp::min(n, k + ku + 1);
            for j in k..border {
                U[(i, j)] = U[(i, j)] - L[(i, k)] * U[(k, j)];
            }
        }
    }

    (L, U, P)
}
////////////////////////////////
//             LINEAR SYSTEM SOLUTION
////////////////////////////////
// easiest solver of linear system: no banded structure, no pivoting
fn solve_lu(l: &DMatrix<f64>, u: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    let n = l.nrows();
    let mut y = DVector::zeros(n);
    let mut x = DVector::zeros(n);

    // Forward substitution Ly = b
    for i in 0..n {
        y[i] = b[i];
        for j in 0..i {
            y[i] -= l[(i, j)] * y[j];
        }
    }

    // Backward substitution Ux = y
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in (i + 1)..n {
            x[i] -= u[(i, j)] * x[j];
        }
        x[i] /= u[(i, i)];
    }

    x
}

// solver for solving a linear system for given L, U matrices, bandwidth and permutation
fn banded_solve_lu(
    l: &DMatrix<f64>,
    u: &DMatrix<f64>,
    b: &DVector<f64>,
    kl: usize,
    ku: usize,
    P: Vec<usize>,
) -> DVector<f64> {
    let n = b.len();
    let mut y = DVector::zeros(n);
    let mut x = DVector::zeros(n);

    // Apply permutation to b
    let mut Pb = DVector::zeros(n);
    for i in 0..n {
        Pb[i] = b[P[i]];
    }
 //   let lower_border = |i: usize| std::cmp::max(0, i as isize - kl as isize) as usize;
    // Forward substitution Ly = Pb
    for i in 0..n {
        y[i] = Pb[i];
        for j in 0..i {
            y[i] -= l[(i, j)] * y[j];
        }
    }
   // let upper_border = |i: usize| std::cmp::min(n, i + kl + 1);
    // Backward substitution Ux = y
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in i + 1..n {
            x[i] -= u[(i, j)] * x[j];
        }
        x[i] /= u[(i, i)];
    }

    x
}

////////////////////////////////////////////////
/// MISC
/// //////////////////////////////////////////////
/// finds the bandwidth of a matrix
fn find_bandwidths(A: &DMatrix<f64>) -> (usize, usize) {
    let n = A.nrows();
    let mut kl = 0; // Number of subdiagonals
    let mut ku = 0; // Number of superdiagonals
                    /*
                        Matrix Iteration: The function find_bandwidths iterates through each element of the matrix A.
                    Subdiagonal Width (kl): For each non-zero element below the main diagonal (i.e., i > j), it calculates the distance from the diagonal and updates
                    kl if this distance is greater than the current value of kl.
                    Superdiagonal Width (ku): Similarly, for each non-zero element above the main diagonal (i.e., j > i), it calculates the distance from the diagonal
                     and updates ku if this distance is greater than the current value of ku.
                         */
    for i in 0..n {
        for j in 0..n {
            if A[(i, j)] != 0.0 {
                if j > i {
                    ku = std::cmp::max(ku, j - i);
                } else if i > j {
                    kl = std::cmp::max(kl, i - j);
                }
            }
        }
    }

    (kl, ku)
}

// for testng purposes function generate a banded matrix with random values. Function takes the size of the matrix,
//the number of subdiagonals and the number of superdiagonals
fn generate_banded_matrix(n: usize, kl: usize, ku: usize) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    let mut A = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in std::cmp::max(0, i as isize - kl as isize) as usize..std::cmp::min(n, i + ku + 1) {
            A[(i, j)] = rng.gen_range(-10.0..10.0);
        }
    }

    A
}
fn how_many_zeros(matrix: &DMatrix<f64>) -> (usize, f64) {
    let (nrows, ncols) = matrix.shape();
    let mut count = 0;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            if matrix[(i, j)] == 0.0 {
                count += 1;
            }
        }
    }
    let proc_of_zero: f64 = count as f64 / (ncols as f64 * nrows as f64);
    println!("Percentage of zeros: {}", proc_of_zero);

    (count, proc_of_zero)
}
// how many columns and rows are filled with zeros
fn count_zeros(matrix: &DMatrix<f64>) -> (usize, usize) {
    let mut row_zeros = 0;
    let mut col_zeros = 0;

    for i in 0..matrix.nrows() {
        let row = matrix.row(i);
        if row.iter().all(|&x| x == 0.0) {
            row_zeros += 1;
        }
    }

    for j in 0..matrix.ncols() {
        let col = matrix.column(j);
        if col.iter().all(|&x| x == 0.0) {
            col_zeros += 1;
        }
    }
    println!("Number of rows filled with zeros: {}", row_zeros);
    println!("Number of columns filled with zeros: {}", col_zeros);
    (row_zeros, col_zeros)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;
    use nalgebra::{DMatrix, DVector};
    #[test]
    fn test_banded_lu() {
        let a = DMatrix::from_row_slice(3, 3, &[2.0, -1.0, -2.0, -4.0, 6.0, 3.0, -4.0, -2.0, 8.0]);
        let b = DVector::from_column_slice(&[1.0, -1.0, 2.0]);
        let (l, u) = lu_decomposition(&a);
        let assertion = &l * &u - &a;
        for num in assertion.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }

        let x = solve_lu(&l, &u, &b);
        let assertion = (&a * x) - (&b);
        for num in assertion.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }
    }

    #[test]
    fn test_banded() {
        // Example usage for a banded matrix
        let n = 5;
        let kl = 1; // number of subdiagonals
        let ku = 1; // number of superdiagonals

        let data = [
            1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0,
            9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 12.0, 13.0,
        ];
        let a = DMatrix::from_row_slice(n, n, &data);
        let b = DVector::from_iterator(n, (0..n).map(|i| i as f64));
        let (l, u) = banded_lu_decomposition_no_pivoting(&a, kl, ku);
        let assertion = &l * &u - &a;
        for num in assertion.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }

        let x = solve_lu(&l, &u, &b);
        let LU_standard = a.clone().lu();
        let x_standart = LU_standard.solve(&b.clone()).unwrap();
        let assertion = (&a * &x) - (&b);
        for num in assertion.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }
        let assertion2 = x - x_standart;
        for num in assertion2.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }
    }
    #[test]
    fn test_banded_with_pivoting() {
        // Example usage for a banded matrix
        let n = 5;
        let kl = 1; // number of subdiagonals
        let ku = 1; // number of superdiagonals
        let data = [
            1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0,
            9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 12.0, 13.0,
        ];
        let a = DMatrix::from_row_slice(n, n, &data);
        let b = DVector::from_iterator(n, (0..n).map(|i| i as f64));
        let (l, u, p) = banded_lu_decomposition_with_pivoting(&a, kl, ku);
       

        let assertion = &l * &u - &a;
        for num in assertion.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }

        let x = solve_lu(&l, &u, &b);
        let LU_standard = a.clone().lu();
        let x_standart = LU_standard.solve(&b.clone()).unwrap();
        let assertion = (&a * &x) - (&b);
        for num in assertion.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }
        let assertion2 = x - x_standart;
        for num in assertion2.iter() {
            assert!(relative_eq!(*num, 0.0, epsilon = 1e-5));
        }
    }

    #[test]
    fn test_bandwidth_calculation() {
        let data = [
            1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0,
            9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 12.0, 13.0,
        ];
        let A = DMatrix::from_row_slice(5, 5, &data);

        let (kl, ku) = find_bandwidths(&A);
        assert_eq!(kl, 1);
        assert_eq!(ku, 1);
    }
    #[test]
    fn test_random_band_matrix_generator() {
        let random_matrix = generate_banded_matrix(1000, 10, 5);
        let (kl, ku) = find_bandwidths(&random_matrix);
        assert_eq!(kl, 10);
        assert_eq!(ku, 5);
    }
}

