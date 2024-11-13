use nalgebra::{ DMatrix, DVector};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use core::panic;



const SPARSE: f64 =0.01;   
/*
Group columns of a 2-D matrix for sparse finite differencing [1]_.

Two columns are in the same group if in each row at least one of them
has zero. A greedy sequential algorithm is used to construct groups.

Parameters
----------
A : array_like or sparse matrix, shape (m, n)
    Matrix of which to group columns.
order : int, iterable of int with shape (n,) or None
    Permutation array which defines the order of columns enumeration.
    If int or None, a random permutation is used with `order` used as
    a random seed. Default is 0, that is use a random permutation but
    guarantee repeatability.

Returns
-------
groups : ndarray of int, shape (n,)
    Contains values from 0 to n_groups-1, where n_groups is the number
    of found groups. Each value ``groups[i]`` is an index of a group to
    which ith column assigned. The procedure was helpful only if
    n_groups is significantly less than n.
    */


extern crate nalgebra as na;
extern crate rand;
pub fn is_sparse_int(matrix: &DMatrix<i64>, threshold: f64) -> bool {
    let (rows, cols) = matrix.shape();
    let total_elements = (rows * cols) as f64;
    let nonzero_elements = matrix.iter().filter(|&x| x.abs() > 0).count() as f64;
    // A matrix is considered sparse if the number of nonzero elements is less than a certain percentage of the total elements.
    // You can adjust the threshold percentage as needed.
    nonzero_elements / total_elements < threshold
}

// order can be None, vector or scalar
pub enum OrderEnum {
    None,
    Vector(Vec<usize>),
    Scalar(usize),
}

/// randomly shuffle columns 
pub fn group_columns(A: &DMatrix<f64>, order: OrderEnum) -> Vec<usize> {
    //turn A into matrix of only ones and zeros
    let A  = A.map(|x| if x != 0.0 { 1 } else { 0 }).cast::<i64>();

    let (m, n) = A.shape();
    // order can be None, vector or scalar
    let new_order = match order {
        OrderEnum::None => {
            // lets create a random permutation of columns
            let mut rng = thread_rng();
            let mut order = (0..n).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            order 
        }
        OrderEnum::Vector(o) => {        
            if o.len() != n {
                panic!("Vector must have the same length as the number of columns")
            }
            o   
        },
        OrderEnum::Scalar(_o) => {

            let mut rng = thread_rng();
            let mut order = (0..n).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            order
        
        },
    };
    // reorder A according to new_order (random new order of columns)
    let  A = A.clone();
    let mut A_ = DMatrix::zeros( m, n);
    for (i, &o) in new_order.iter().enumerate() {
        // iterate over columns and set to the i-th column - the random one
        A_.set_column(i, &A.column(o));
    }

    let groups = if is_sparse_int(&A,  SPARSE) {
        group_sparse(m, n, &A_)
    } else {
        group_dense(m, n, &A_)
    };

    let mut result = vec![0; n];
    for (i, &o) in new_order.iter().enumerate() {
        result[o] = groups[i];
    }

    result
}


fn group_dense(m: usize, n: usize, A: &DMatrix<i64>) -> Vec<usize> {
    let mut groups = vec![-1; n];
    let mut current_group = 0;
    let mut union = DVector::zeros(m);

    for i in 0..n {
        if groups[i] >= 0 {
            continue;
        }
        groups[i] = current_group;
        let mut all_grouped = true;
        union.copy_from(&A.column(i));

        for j in 0..n {
            if groups[j] >= 0 {
                continue;
            }
            all_grouped = false;
            let mut intersect = false;
            for k in 0..m {
                if union[k] > 0 && A[(k, j)] > 0 {
                    intersect = true;
                    break;
                }
            }
            if !intersect {
                union += &A.column(j);
                groups[j] = current_group;
            }
        }
        if all_grouped {
            break;
        }
        current_group += 1;
    }

    groups.iter().map(|&x| x as usize).collect()
}

fn group_sparse(m: usize, n: usize, A: &DMatrix<i64>) -> Vec<usize> {
    let mut groups = vec![-1; n];
    let mut current_group = 0;
    let mut union = DVector::zeros(m);

    for i in 0..n {
        if groups[i] >= 0 {
            continue;
        }
        groups[i] = current_group;
        let mut all_grouped = true;
        union.fill(0);
        for k in 0..m {
            if A[(k, i)] != 0 {
                union[k] = 1;
            }
        }

        for j in 0..n {
            if groups[j] >= 0 {
                continue;
            }
            all_grouped = false;
            let mut intersect = false;
            for k in 0..m {
                if union[k] == 1 && A[(k, j)] != 0 {
                    intersect = true;
                    break;
                }
            }
            if !intersect {
                for k in 0..m {
                    if A[(k, j)] != 0 {
                        union[k] = 1;
                    }
                }
                groups[j] = current_group;
            }
        }
        if all_grouped {
            break;
        }
        current_group += 1;
    }

    groups.iter().map(|&x| x as usize).collect()
}

    /* 
    fn is_sparse_dense(matrix: &DMatrix<i32>, threshold: f64) -> bool {
        let total_elements = matrix.len();
        let non_zero_elements = matrix.iter().filter(|&&x| x != 0).count();
        let sparsity = non_zero_elements as f64 / total_elements as f64;
        sparsity < threshold
    }

pub fn group_columns(A: DMatrix<i32>, order: Option<&[usize]>) -> Result<DVector<usize>, Box<dyn std::error::Error>> {
    if is_sparse_dense(&A, 0.5) {   
        println!("matrix is sparse");
        
        
    }
    let (m, n) = A.shape();

    let order = if let Some(order) = order {
        if order.len() != n {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "`order` has incorrect shape.",
            )));
        }
        order.to_vec()
    } else {
        let mut rng = thread_rng();
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        order
    };

    let mut A_permuted = DMatrix::zeros(m, n);
    for (i, &col) in order.iter().enumerate() {
        A_permuted.set_column(i, &A.column(col));
    }

    let groups = if A_permuted.is_sparse() {
        group_sparse(m, n, &A_permuted);
    } else {
        group_dense(m, n, &A_permuted)
    };

    let mut result = DVector::zeros(n);
    for (i, &col) in order.iter().enumerate() {
        result[col] = groups[i];
    }

    Ok(result)
}




fn group_dense(m: usize, n: usize, A: &DMatrix<i32>) -> DVector<isize> {
    let B = A.transpose();
    let mut groups = DVector::from_element(n, -1);
    let mut current_group = 0;
    let mut union = DVector::zeros(m);

    for i in 0..n {
        if groups[i] >= 0 {
            continue;
        }

        groups[i] = current_group;
        let mut all_grouped = true;
        union.copy_from(&B.row(i));

        for j in 0..n {
            if groups[j] >= 0 {
                continue;
            }

            all_grouped = false;
            let mut intersect = false;

            for k in 0..m {
                if union[k] > 0 && B[(j, k)] > 0 {
                    intersect = true;
                    break;
                }
            }

            if !intersect {
                union += &B.row(j);
                groups[j] = current_group;
            }
        }

        if all_grouped {
            break;
        }

        current_group += 1;
    }

    groups
}

fn group_sparse(m: usize, n: usize, indices: &[i32], indptr: &[i32]) -> DVector<isize> {
    let mut groups = DVector::from_element(n, -1);
    let mut current_group = 0;
    let mut union = DVector::zeros(m);

    for i in 0..n {
        if groups[i] >= 0 {
            continue;
        }

        groups[i] = current_group;
        let mut all_grouped = true;
        union.fill(0);

        for k in indptr[i]..indptr[i + 1] {
            union[indices[k as usize] as usize] = 1;
        }

        for j in 0..n {
            if groups[j] >= 0 {
                continue;
            }

            all_grouped = false;
            let mut intersect = false;

            for k in indptr[j]..indptr[j + 1] {
                if union[indices[k as usize] as usize] == 1 {
                    intersect = true;
                    break;
                }
            }

            if !intersect {
                for k in indptr[j]..indptr[j + 1] {
                    union[indices[k as usize] as usize] = 1;
                }
                groups[j] = current_group;
            }
        }

        if all_grouped {
            break;
        }

        current_group += 1;
    }

    groups
}


/*  {
    // Example usage of the group_columns function
    let A = DMatrix::from_vec(3, 3, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
    let order = None;

    match group_columns(&A, order) {
        Ok(groups) => println!("Groups: {:?}", groups),
        Err(e) => println!("Error: {}", e),
    }
}

*/

*/