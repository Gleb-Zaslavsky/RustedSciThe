//! Boundary value problem solver - Rust translation from SciPy's _bvp.py
//!
//! This module implements a 4th order collocation algorithm with residual control
//! similar to the MATLAB/SciPy BVP solver, translated from Python to Rust.
use crate::numerical::optimization::PPoly::{Extrapolate, PPoly};
use faer::col::{Col, ColRef};
use faer::linalg::solvers::Solve;
use faer::mat::{Mat, MatRef};
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};

type faer_mat = SparseColMat<usize, f64>;
type faer_col = Col<f64>;
type faer_dense_mat = Mat<f64>;
// demonstrating function or reminder function with main
#[allow(dead_code)]
pub fn test_faer_fn() {
    type faer_mat = SparseColMat<usize, f64>;
    type faer_col = Col<f64>; // Mat<f64>;
    // Example usage of faer types
    type faer_mat_sym = SymbolicSparseColMat<usize, f64>;
    // empty vector
    let mut vec1: faer_col = faer_col::zeros(2);
    // setting values of vector
    vec1[1] = 1.0;
    // 1st method to create sparse matrix
    // col_ptr: [0, 1, 1, 2, 2, 2]
    // - Column 0: 1 entry (row 0)
    // - Column 1: 0 entries
    // - Column 2: 1 entry (row 1)
    // - Column 3: 0 entries
    // - Column 4: 0 entries
    let nrows = 5;
    let ncols = 5;
    let col_ptr = vec![0usize, 1, 1, 2, 2, 2]; // length = ncols + 1
    let row_idx = vec![0usize, 1]; // length = col_ptr[ncols]
    let values = vec![1.0, 2.0]; // same length as row_idx
    // col_nnz in SymbolicSparseColMat (and related types) stands for column nonzero counts.
    // It is an optional vector that, if present, specifies for
    // each column the number of nonzero entries in that column.
    let symbolic = SymbolicSparseColMat::new_checked(nrows, ncols, col_ptr, None, row_idx);
    let matrix = SparseColMat::new(symbolic, values);
    // getting certain element from matrix
    let _element = matrix.get(1, 1).unwrap_or(&0.0);
    //creating sparse mattrix from vector of triplets
    let non_zero_triplet = vec![(0, 0, 1.0), (1, 1, 1.0)];
    let nrows = 2;
    let ncols = 2;

    let triplet: Vec<Triplet<usize, usize, f64>> = non_zero_triplet
        .iter()
        .map(|triplet| Triplet::new(triplet.0, triplet.1, triplet.2))
        .collect::<Vec<_>>();
    let new_matrix: SparseColMat<usize, f64> =
        SparseColMat::try_new_from_triplets(nrows, ncols, triplet.as_slice()).unwrap();
    // solving procedure for linear system
    let lhs: MatRef<f64> = vec1.as_mat();
    let LU = new_matrix.sp_lu().unwrap();
    let res: Mat<f64> = LU.solve(lhs);
    let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();
    // calculation result
    let _res: faer_col = ColRef::from_slice(res_vec.as_slice()).to_owned();
}
/// Machine epsilon for floating point arithmetic
const EPS: f64 = f64::EPSILON;

/// Result structure for BVP solver
#[derive(Debug, Clone)]
pub struct BVPResult {
    /// Solution as cubic spline interpolator (PPoly)
    pub sol: Option<PPoly>,
    /// Found parameters (if any)
    pub p: Option<faer_col>,
    /// Final mesh nodes
    pub x: faer_col,
    /// Solution values at mesh nodes
    pub y: faer_dense_mat,
    /// Solution derivatives at mesh nodes  
    pub yp: faer_dense_mat,
    /// RMS residuals over each mesh interval
    pub rms_residuals: faer_col,
    /// Number of iterations
    pub niter: usize,
    /// Status code (0=success, 1=max_nodes, 2=singular, 3=bc_tol)
    pub status: i32,
    /// Status message
    pub message: String,
    /// Success flag
    pub success: bool,
}

/// Function type for ODE right-hand side evaluation
/// Arguments: (x, y, p) where x is scalar, y is n-dimensional, p is k-dimensional parameters
pub type ODEFunction = dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> faer_dense_mat;

/// Function type for boundary condition evaluation  
/// Arguments: (ya, yb, p) where ya, yb are n-dimensional boundary values, p is parameters
/// Returns: (n+k)-dimensional boundary condition residuals
pub type BCFunction = dyn Fn(&faer_col, &faer_col, &faer_col) -> faer_col;

/// Function type for ODE Jacobian evaluation (optional)
/// Returns: (df_dy, df_dp) where df_dy is (n,n,m), df_dp is (n,k,m) or None
pub type ODEJacobian =
    dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> (Vec<faer_mat>, Option<Vec<faer_mat>>);

/// Function type for boundary condition Jacobian evaluation (optional)  
/// Returns: (dbc_dya, dbc_dyb, dbc_dp) where each is appropriately sized matrix or None
pub type BCJacobian =
    dyn Fn(&faer_col, &faer_col, &faer_col) -> (faer_mat, faer_mat, Option<faer_mat>);

/// Estimate derivatives of an ODE system RHS with forward differences
///
/// # Arguments
/// * `fun` - ODE function to differentiate
/// * `x` - Mesh points (m,)
/// * `y` - Solution values at mesh points (n, m)  
/// * `p` - Parameters (k,)
/// * `f0` - Pre-computed function values (optional)
///
/// # Returns
/// * `df_dy` - Derivatives w.r.t. y: Vec of (n,n) matrices, one per mesh point
/// * `df_dp` - Derivatives w.r.t. p: Vec of (n,k) matrices, one per mesh point (or None)
pub fn estimate_fun_jac(
    fun: &ODEFunction,
    x: &faer_col,
    y: &faer_dense_mat,
    p: &faer_col,
    f0: Option<&faer_dense_mat>,
) -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
    let (n, m) = (y.nrows(), y.ncols());

    let f0_computed;
    let f0_ref = match f0 {
        Some(f) => f,
        None => {
            f0_computed = fun(x, y, p);
            &f0_computed
        }
    };

    let mut df_dy = Vec::with_capacity(m);

    // Compute df/dy for each mesh point
    for col in 0..m {
        let mut triplets = Vec::new();

        for i in 0..n {
            let mut y_perturbed = y.clone();
            let h = EPS.sqrt() * (1.0 + y.get(i, col).abs());
            *y_perturbed.get_mut(i, col) += h;

            let x_slice = x.clone();
            let f_new = fun(&x_slice, &y_perturbed, p);

            for row in 0..n {
                let val = (f_new.get(row, col) - f0_ref.get(row, col)) / h;
                if val.abs() > 1e-15 {
                    triplets.push(Triplet::new(row, i, val));
                }
            }
        }
        let jacobian = SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();
        df_dy.push(jacobian);
    }

    // Compute df/dp if parameters exist
    let df_dp = if p.nrows() == 0 {
        None
    } else {
        let k = p.nrows();
        let mut df_dp_vec = Vec::with_capacity(m);

        for col in 0..m {
            let mut triplets = Vec::new();

            for i in 0..k {
                let mut p_perturbed = p.clone();
                let h = EPS.sqrt() * (1.0 + p[i].abs());
                p_perturbed[i] += h;

                let f_new = fun(x, y, &p_perturbed);

                for row in 0..n {
                    let val = (f_new.get(row, col) - f0_ref.get(row, col)) / h;
                    if val.abs() > 1e-15 {
                        triplets.push(Triplet::new(row, i, val));
                    }
                }
            }
            let param_jacobian = SparseColMat::try_new_from_triplets(n, k, &triplets).unwrap();
            df_dp_vec.push(param_jacobian);
        }
        Some(df_dp_vec)
    };

    (df_dy, df_dp)
}

/// Estimate derivatives of boundary conditions with forward differences
///
/// # Arguments  
/// * `bc` - Boundary condition function
/// * `ya` - Left boundary values (n,)
/// * `yb` - Right boundary values (n,)
/// * `p` - Parameters (k,)
/// * `bc0` - Pre-computed boundary condition values (optional)
///
/// # Returns
/// * `dbc_dya` - Derivatives w.r.t. ya: (n+k, n) matrix
/// * `dbc_dyb` - Derivatives w.r.t. yb: (n+k, n) matrix  
/// * `dbc_dp` - Derivatives w.r.t. p: (n+k, k) matrix (or None)
pub fn estimate_bc_jac(
    bc: &BCFunction,
    ya: &faer_col,
    yb: &faer_col,
    p: &faer_col,
    bc0: Option<&faer_col>,
) -> (faer_mat, faer_mat, Option<faer_mat>) {
    let n = ya.nrows();
    let k = p.nrows();

    let bc0_computed;
    let bc0_ref = match bc0 {
        Some(bc_val) => bc_val,
        None => {
            bc0_computed = bc(ya, yb, p);
            &bc0_computed
        }
    };

    // Compute dbc/dya
    let mut triplets_ya = Vec::new();
    for i in 0..n {
        let mut ya_perturbed = ya.clone();
        let h = EPS.sqrt() * (1.0 + ya[i].abs());
        ya_perturbed[i] += h;

        let bc_new = bc(&ya_perturbed, yb, p);
        for row in 0..(n + k) {
            let val = (bc_new[row] - bc0_ref[row]) / h;
            if val.abs() > 1e-15 {
                triplets_ya.push(Triplet::new(row, i, val));
            }
        }
    }
    let dbc_dya = SparseColMat::try_new_from_triplets(n + k, n, &triplets_ya).unwrap();

    // Compute dbc/dyb
    let mut triplets_yb = Vec::new();
    for i in 0..n {
        let mut yb_perturbed = yb.clone();
        let h = EPS.sqrt() * (1.0 + yb[i].abs());
        yb_perturbed[i] += h;

        let bc_new = bc(ya, &yb_perturbed, p);
        for row in 0..(n + k) {
            let val = (bc_new[row] - bc0_ref[row]) / h;
            if val.abs() > 1e-15 {
                triplets_yb.push(Triplet::new(row, i, val));
            }
        }
    }
    let dbc_dyb = SparseColMat::try_new_from_triplets(n + k, n, &triplets_yb).unwrap();

    // Compute dbc/dp if parameters exist
    let dbc_dp = if k == 0 {
        None
    } else {
        let mut triplets_p = Vec::new();
        for i in 0..k {
            let mut p_perturbed = p.clone();
            let h = EPS.sqrt() * (1.0 + p[i].abs());
            p_perturbed[i] += h;

            let bc_new = bc(ya, yb, &p_perturbed);
            for row in 0..(n + k) {
                let val = (bc_new[row] - bc0_ref[row]) / h;
                if val.abs() > 1e-15 {
                    triplets_p.push(Triplet::new(row, i, val));
                }
            }
        }
        Some(SparseColMat::try_new_from_triplets(n + k, k, &triplets_p).unwrap())
    };

    (dbc_dya, dbc_dyb, dbc_dp)
}

/// Compute indices for the collocation system Jacobian construction
///
/// # Arguments
/// * `n` - Number of equations in ODE system
/// * `m` - Number of mesh nodes  
/// * `k` - Number of unknown parameters
///
/// # Returns
/// * `i_indices` - Row indices for sparse matrix construction
/// * `j_indices` - Column indices for sparse matrix construction  
pub fn compute_jac_indices(n: usize, m: usize, k: usize) -> (Vec<usize>, Vec<usize>) {
    // Pattern matches the Python implementation structure
    let mut i_indices = Vec::new();
    let mut j_indices = Vec::new();

    // Block 1: m-1 diagonal n×n blocks for collocation residuals
    for block in 0..(m - 1) {
        for i in 0..n {
            for j in 0..n {
                i_indices.push(block * n + i);
                j_indices.push(block * n + j);
            }
        }
    }

    // Block 2: m-1 off-diagonal n×n blocks for collocation residuals
    for block in 0..(m - 1) {
        for i in 0..n {
            for j in 0..n {
                i_indices.push(block * n + i);
                j_indices.push((block + 1) * n + j);
            }
        }
    }

    // Block 3: (n+k)×n block for dependency of BC on ya
    for i in 0..(n + k) {
        for j in 0..n {
            i_indices.push((m - 1) * n + i);
            j_indices.push(j);
        }
    }

    // Block 4: (n+k)×n block for dependency of BC on yb
    for i in 0..(n + k) {
        for j in 0..n {
            i_indices.push((m - 1) * n + i);
            j_indices.push((m - 1) * n + j);
        }
    }

    // Block 5: (m-1)*n×k block for dependency of collocation on p
    if k > 0 {
        for block in 0..(m - 1) {
            for i in 0..n {
                for j in 0..k {
                    i_indices.push(block * n + i);
                    j_indices.push(m * n + j);
                }
            }
        }

        // Block 6: (n+k)×k block for dependency of BC on p
        for i in 0..(n + k) {
            for j in 0..k {
                i_indices.push((m - 1) * n + i);
                j_indices.push(m * n + j);
            }
        }
    }

    (i_indices, j_indices)
}

/// Stacked matrix multiplication: out[i,:,:] = a[i,:,:] * b[i,:,:]
/*
pub fn stacked_matmul(a: &[faer_dense_mat], b: &[faer_dense_mat]) -> Vec<faer_dense_mat> {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(a_mat, b_mat)| {
            let mut result = faer_dense_mat::zeros(a_mat.nrows(), b_mat.ncols());
            faer::linalg::matmul::matmul(
                result.as_mut(),
                a_mat.as_ref(),
                b_mat.as_ref(),
                None,
                1.0,
                faer::Parallelism::None,
            );
            result
        })
        .collect()
}
*/
/// Evaluate collocation residuals
///
/// This function implements the core collocation method. The solution is sought
/// as a cubic C1 continuous spline with derivatives matching the ODE RHS at given nodes.
/// Collocation conditions are formed from equality of spline derivatives and RHS
/// of the ODE system at middle points between nodes.
///
/// # Arguments
/// * `fun` - ODE function f(x, y, p)
/// * `y` - Solution values at mesh nodes (n, m)
/// * `p` - Parameters (k,)  
/// * `x` - Mesh nodes (m,)
/// * `h` - Mesh intervals (m-1,)
///
/// # Returns
/// * `col_res` - Collocation residuals at middle points (n, m-1)
/// * `y_middle` - Spline values at middle points (n, m-1)
/// * `f` - RHS values at mesh nodes (n, m)
/// * `f_middle` - RHS values at middle points (n, m-1)
pub fn collocation_fun(
    fun: &ODEFunction,
    y: &faer_dense_mat,
    p: &faer_col,
    x: &faer_col,
    h: &faer_col,
) -> (
    faer_dense_mat,
    faer_dense_mat,
    faer_dense_mat,
    faer_dense_mat,
) {
    let (n, m) = (y.nrows(), y.ncols());

    // Evaluate RHS at mesh nodes
    let f = fun(x, y, p);

    // Compute solution values at middle points using cubic interpolation formula
    let mut y_middle = faer_dense_mat::zeros(n, m - 1);
    for i in 0..n {
        for j in 0..(m - 1) {
            *y_middle.get_mut(i, j) = 0.5 * (y.get(i, j + 1) + y.get(i, j))
                - 0.125 * h[j] * (f.get(i, j + 1) - f.get(i, j));
        }
    }

    // Evaluate RHS at middle points
    // The ODE function expects x and y to have consistent dimensions
    // We evaluate the function at each middle point individually
    let mut f_middle = faer_dense_mat::zeros(n, m - 1);
    for j in 0..(m - 1) {
        let x_mid = x[j] + 0.5 * h[j];
        let x_single = faer_col::from_fn(1, |_| x_mid);

        // Extract the j-th column of y_middle for evaluation as a column vector
        let mut y_single = faer_dense_mat::zeros(n, 1);
        for i in 0..n {
            *y_single.get_mut(i, 0) = *y_middle.get(i, j);
        }

        let f_result = fun(&x_single, &y_single, p);

        // Copy the result to the j-th column of f_middle
        for i in 0..n {
            *f_middle.get_mut(i, j) = *f_result.get(i, 0);
        }
    }

    // Compute collocation residuals
    let mut col_res = faer_dense_mat::zeros(n, m - 1);
    for i in 0..n {
        for j in 0..(m - 1) {
            *col_res.get_mut(i, j) = y.get(i, j + 1)
                - y.get(i, j)
                - h[j] / 6.0 * (f.get(i, j) + f.get(i, j + 1) + 4.0 * f_middle.get(i, j));
        }
    }

    (col_res, y_middle, f, f_middle)
}

/// Construct the Jacobian of the collocation system using sparse matrices
pub fn construct_global_jac(
    n: usize,
    m: usize,
    k: usize,
    h: &faer_col,
    df_dy: &[faer_mat],
    df_dy_middle: &[faer_mat],
    df_dp: Option<&[faer_mat]>,
    df_dp_middle: Option<&[faer_mat]>,
    dbc_dya: &faer_mat,
    dbc_dyb: &faer_mat,
    dbc_dp: Option<&faer_mat>,
) -> faer_mat {
    let total_size = (m - 1) * n + (n + k);
    let mut triplets = Vec::new();

    // Process diagonal and off-diagonal blocks for collocation residuals
    for i in 0..(m - 1) {
        let h_i = h[i];

        // For sparse matrices, we'll compute the blocks directly as triplets
        // Diagonal block: -I - h/6*(df_dy[i] + 2*df_dy_middle[i]) - h²/12*df_dy_middle[i]*df_dy[i]

        // Add diagonal block entries
        let row_start = i * n;
        let col_start = i * n;

        // Identity part: -I
        for idx in 0..n {
            triplets.push(Triplet::new(row_start + idx, col_start + idx, -1.0));
        }

        // Add -h/6 * df_dy[i]
        for triplt in df_dy[i].triplet_iter() {
            let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
            let coeff = -h_i / 6.0 * val;
            if coeff.abs() > 1e-15 {
                triplets.push(Triplet::new(
                    row_start + row_idx,
                    col_start + col_idx,
                    coeff,
                ));
            }
        }

        // Add -h/3 * df_dy_middle[i] (2 * h/6)
        for triplt in df_dy_middle[i].triplet_iter() {
            let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
            let coeff = -h_i / 3.0 * val;
            if coeff.abs() > 1e-15 {
                triplets.push(Triplet::new(
                    row_start + row_idx,
                    col_start + col_idx,
                    coeff,
                ));
            }
        }

        // Off-diagonal block: I - h/6*(df_dy[i+1] + 2*df_dy_middle[i]) + h²/12*df_dy_middle[i]*df_dy[i+1]

        // Add off-diagonal block entries
        let col_start_off = (i + 1) * n;

        // Identity part: I
        for idx in 0..n {
            triplets.push(Triplet::new(row_start + idx, col_start_off + idx, 1.0));
        }

        // Add -h/6 * df_dy[i+1]
        for triplt in df_dy[i + 1].triplet_iter() {
            let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
            let coeff = -h_i / 6.0 * val;
            if coeff.abs() > 1e-15 {
                triplets.push(Triplet::new(
                    row_start + row_idx,
                    col_start_off + col_idx,
                    coeff,
                ));
            }
        }

        // Add -h/3 * df_dy_middle[i]
        for triplt in df_dy_middle[i].triplet_iter() {
            let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
            let coeff = -h_i / 3.0 * val;
            if coeff.abs() > 1e-15 {
                triplets.push(Triplet::new(
                    row_start + row_idx,
                    col_start_off + col_idx,
                    coeff,
                ));
            }
        }

        // Handle parameter derivatives if present
        if let (Some(df_dp_vec), Some(df_dp_middle_vec)) = (df_dp, df_dp_middle) {
            let param_col_start = n * m;

            // Add -h/6 * df_dp[i]
            for triplt in df_dp_vec[i].triplet_iter() {
                let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
                let coeff = -h_i / 6.0 * val;
                if coeff.abs() > 1e-15 {
                    triplets.push(Triplet::new(
                        row_start + row_idx,
                        param_col_start + col_idx,
                        coeff,
                    ));
                }
            }

            // Add -h/6 * df_dp[i+1]
            for triplt in df_dp_vec[i + 1].triplet_iter() {
                let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
                let coeff = -h_i / 6.0 * val;
                if coeff.abs() > 1e-15 {
                    triplets.push(Triplet::new(
                        row_start + row_idx,
                        param_col_start + col_idx,
                        coeff,
                    ));
                }
            }

            // Add -2h/3 * df_dp_middle[i]
            for triplt in df_dp_middle_vec[i].triplet_iter() {
                let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
                let coeff = -2.0 * h_i / 3.0 * val;
                if coeff.abs() > 1e-15 {
                    triplets.push(Triplet::new(
                        row_start + row_idx,
                        param_col_start + col_idx,
                        coeff,
                    ));
                }
            }
        }
    }

    // Insert boundary condition blocks
    let bc_row_start = (m - 1) * n;

    // dbc_dya block - dependency on ya
    for triplt in dbc_dya.triplet_iter() {
        let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
        if val.abs() > 1e-15 {
            triplets.push(Triplet::new(bc_row_start + row_idx, col_idx, *val));
        }
    }

    // dbc_dyb block - dependency on yb
    let yb_col_start = (m - 1) * n;
    for triplt in dbc_dyb.triplet_iter() {
        let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
        if val.abs() > 1e-15 {
            triplets.push(Triplet::new(
                bc_row_start + row_idx,
                yb_col_start + col_idx,
                *val,
            ));
        }
    }

    // dbc_dp block - dependency on parameters
    if let Some(dbc_dp_mat) = dbc_dp {
        let param_col_start = n * m;
        for triplt in dbc_dp_mat.triplet_iter() {
            let (row_idx, col_idx, val) = (triplt.row, triplt.col, triplt.val);
            if val.abs() > 1e-15 {
                triplets.push(Triplet::new(
                    bc_row_start + row_idx,
                    param_col_start + col_idx,
                    *val,
                ));
            }
        }
    }

    // Create sparse matrix from triplets
    SparseColMat::try_new_from_triplets(total_size, total_size, &triplets)
        .expect("Failed to create sparse matrix")
}

/// Solve the nonlinear collocation system by Newton's method with sparse matrices
///
/// This version uses faer's sparse matrix capabilities for better performance.
///
/// # Arguments
/// * `n` - Number of equations in the ODE system
/// * `m` - Number of nodes in the mesh
/// * `h` - Mesh intervals (m-1,)
/// * `fun` - ODE function
/// * `bc` - Boundary condition function
/// * `fun_jac` - ODE Jacobian function (optional)
/// * `bc_jac` - BC Jacobian function (optional)
/// * `y` - Initial guess for solution values at mesh nodes (n, m)
/// * `p` - Initial guess for parameters (k,)
/// * `x` - Mesh nodes (m,)
/// * `bvp_tol` - BVP tolerance
/// * `bc_tol` - Boundary condition tolerance
///
/// # Returns
/// * `y` - Final solution values at mesh nodes
/// * `p` - Final parameter values  
/// * `singular` - True if Jacobian was singular
pub fn solve_newton(
    n: usize,
    m: usize,
    h: &faer_col,
    fun: &ODEFunction,
    bc: &BCFunction,
    fun_jac: Option<&ODEJacobian>,
    bc_jac: Option<&BCJacobian>,
    mut y: faer_dense_mat,
    mut p: faer_col,
    x: &faer_col,
    bvp_tol: f64,
    bc_tol: f64,
) -> (faer_dense_mat, faer_col, bool) {
    let k = p.nrows();

    // Newton iteration parameters
    let max_iter = 8;
    let max_njev = 4;
    let sigma = 0.2; // Armijo constant
    let tau = 0.5; // Step size decrease factor
    let n_trial = 4; // Max backtracking steps

    // Tolerance for collocation residuals
    let tol_r: faer_col = faer_col::from_fn(h.nrows(), |i| 2.0 / 3.0 * h[i] * 5e-2 * bvp_tol);

    let mut njev = 0;
    let mut singular = false;
    let mut recompute_jac = true;
    let mut lu_solver: Option<faer::sparse::linalg::solvers::Lu<_, _>> = None;
    let mut cost = 0.0;

    for _iteration in 0..max_iter {
        // Compute collocation residuals and function values
        let (col_res, y_middle, f, f_middle) = collocation_fun(fun, &y, &p, x, h);

        // Extract first and last columns for boundary conditions
        let ya = faer_col::from_fn(n, |i| *y.get(i, 0));
        let yb = faer_col::from_fn(n, |i| *y.get(i, m - 1));
        let bc_res = bc(&ya, &yb, &p);

        // Stack residuals into single vector (Python uses Fortran order: column-major)
        let mut res: Col<f64> = faer_col::zeros((m - 1) * n + (n + k));

        // Collocation residuals (column-major order like Python)
        for j in 0..(m - 1) {
            for i in 0..n {
                res[j * n + i] = *col_res.get(i, j);
            }
        }

        // Boundary condition residuals
        for i in 0..(n + k) {
            res[(m - 1) * n + i] = bc_res[i];
        }

        if recompute_jac {
            // Compute Jacobians
            let (df_dy, df_dp) = if let Some(jac_fn) = fun_jac {
                jac_fn(x, &y, &p)
            } else {
                estimate_fun_jac(fun, x, &y, &p, Some(&f))
            };

            let mut x_middle = faer_col::zeros(m - 1);
            for j in 0..(m - 1) {
                x_middle[j] = x[j] + 0.5 * h[j];
            }

            let (df_dy_middle, df_dp_middle) = if let Some(jac_fn) = fun_jac {
                jac_fn(&x_middle, &y_middle, &p)
            } else {
                estimate_fun_jac(fun, &x_middle, &y_middle, &p, Some(&f_middle))
            };

            let (dbc_dya, dbc_dyb, dbc_dp) = if let Some(bc_jac_fn) = bc_jac {
                bc_jac_fn(&ya, &yb, &p)
            } else {
                estimate_bc_jac(bc, &ya, &yb, &p, Some(&bc_res))
            };

            // Construct global Jacobian
            let jac_matrix: faer_mat = construct_global_jac(
                n,
                m,
                k,
                h,
                &df_dy,
                &df_dy_middle,
                df_dp.as_ref().map(|v| v.as_slice()),
                df_dp_middle.as_ref().map(|v| v.as_slice()),
                &dbc_dya,
                &dbc_dyb,
                dbc_dp.as_ref(),
            );

            // Attempt sparse LU decomposition
            match jac_matrix.sp_lu() {
                Ok(lu) => {
                    let step = lu.solve(res.as_mat());
                    let step_col = faer_col::from_fn(step.nrows(), |i| *step.get(i, 0));

                    lu_solver = Some(lu);
                    cost = step_col.squared_norm_l2();
                }
                Err(_) => {
                    singular = true;
                    break;
                }
            }
            njev += 1;
        }

        if let Some(ref lu) = lu_solver {
            let step: Mat<f64> = lu.solve(res.as_mat());
            let step_col = faer_col::from_fn(step.nrows(), |i| *step.get(i, 0));

            // Extract step components
            let mut y_step = faer_dense_mat::zeros(n, m);
            for j in 0..m {
                for i in 0..n {
                    *y_step.get_mut(i, j) = step_col[j * n + i];
                }
            }

            let p_step = if k > 0 {
                faer_col::from_fn(k, |i| step_col[(m - 1) * n + (n + k) - k + i])
            } else {
                faer_col::zeros(0)
            };

            // Backtracking line search
            let mut alpha = 1.0;
            let mut best_y = y.clone();
            let mut best_p = p.clone();
            let mut best_cost = cost;

            for trial in 0..=n_trial {
                let mut y_new = y.clone();
                let mut p_new = p.clone();

                // y_new = y - alpha * y_step
                for i in 0..n {
                    for j in 0..m {
                        *y_new.get_mut(i, j) -= alpha * y_step.get(i, j);
                    }
                }

                // p_new = p - alpha * p_step
                for i in 0..k {
                    p_new[i] -= alpha * p_step[i];
                }

                // Compute new residuals
                let (col_res_new, _y_middle_new, _f_new, _f_middle_new) =
                    collocation_fun(fun, &y_new, &p_new, x, h);

                let ya_new = faer_col::from_fn(n, |i| *y_new.get(i, 0));
                let yb_new = faer_col::from_fn(n, |i| *y_new.get(i, m - 1));
                let bc_res_new = bc(&ya_new, &yb_new, &p_new);

                let mut res_new = faer_col::zeros((m - 1) * n + (n + k));
                for j in 0..(m - 1) {
                    for i in 0..n {
                        res_new[j * n + i] = *col_res_new.get(i, j);
                    }
                }
                for i in 0..(n + k) {
                    res_new[(m - 1) * n + i] = bc_res_new[i];
                }

                let step_new = lu.solve(res_new.as_mat());
                let step_new_col = faer_col::from_fn(step_new.nrows(), |i| *step_new.get(i, 0));
                let cost_new = step_new_col.squared_norm_l2();

                if cost_new < (1.0 - 2.0 * alpha * sigma) * cost {
                    best_y = y_new;
                    best_p = p_new;
                    best_cost = cost_new;
                    break;
                }

                if trial < n_trial {
                    alpha *= tau;
                }
            }

            y = best_y;
            p = best_p;
            cost = best_cost;

            // Check convergence
            let (col_res_final, _, _f_final, f_middle_final) = collocation_fun(fun, &y, &p, x, h);
            let ya_final = faer_col::from_fn(n, |i| *y.get(i, 0));
            let yb_final = faer_col::from_fn(n, |i| *y.get(i, m - 1));
            let bc_res_final = bc(&ya_final, &yb_final, &p);

            let mut converged = true;

            // Check collocation residuals
            for j in 0..(m - 1) {
                for i in 0..n {
                    if col_res_final.get(i, j).abs()
                        >= tol_r[j] * (1.0 + f_middle_final.get(i, j).abs())
                    {
                        converged = false;
                        break;
                    }
                }
                if !converged {
                    break;
                }
            }

            // Check boundary condition residuals
            for i in 0..(n + k) {
                if bc_res_final[i].abs() >= bc_tol {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }

            // Decide whether to recompute Jacobian
            recompute_jac = if alpha == 1.0 {
                false // Continue with same Jacobian
            } else {
                true // Recompute Jacobian
            };

            if njev >= max_njev {
                break;
            }
        } else {
            singular = true;
            break;
        }
    }

    (y, p, singular)
}

/// Create a cubic spline given values and derivatives at mesh nodes.
///
/// This function creates a cubic spline representation matching SciPy's PPoly format.
/// The formulas for coefficients are taken from scipy.interpolate.CubicSpline.
///
/// # Arguments
/// * `y` - Function values at mesh nodes (n, m)
/// * `yp` - Function derivatives at mesh nodes (n, m)
/// * `x` - Mesh nodes (m,)
/// * `h` - Mesh intervals (m-1,)
///
/// # Returns
/// PPoly spline with axis=1 for compatibility with SciPy
pub fn create_spline(y: &faer_dense_mat, yp: &faer_dense_mat, x: &faer_col, h: &faer_col) -> PPoly {
    let (n, m) = (y.nrows(), y.ncols());

    // Initialize coefficient array: c[degree][interval][polynomial]
    // For cubic spline: degree = 4 (coefficients for x^3, x^2, x^1, x^0)
    let mut c = vec![vec![vec![0.0; n]; m - 1]; 4];

    // Compute cubic spline coefficients for each interval and each polynomial
    for j in 0..(m - 1) {
        for i in 0..n {
            // Compute slope over interval
            let slope = (y.get(i, j + 1) - y.get(i, j)) / h[j];

            // Compute t parameter: (yp_left + yp_right - 2*slope) / h
            let t = (yp.get(i, j) + yp.get(i, j + 1) - 2.0 * slope) / h[j];

            // Cubic spline coefficients (highest degree first)
            c[0][j][i] = t / h[j]; // x^3 coefficient
            c[1][j][i] = (slope - yp.get(i, j)) / h[j] - t; // x^2 coefficient  
            c[2][j][i] = *yp.get(i, j); // x^1 coefficient
            c[3][j][i] = *y.get(i, j); // x^0 coefficient
        }
    }

    PPoly {
        c,
        x: (0..x.nrows()).map(|i| x[i]).collect(),
        extrapolate: Extrapolate::Bool(true),
        axis: 1, // axis=1 for compatibility with SciPy
    }
}
/// Estimate RMS values of collocation residuals using Lobatto quadrature.
///
/// The residuals are defined as the difference between the derivatives of
/// our solution and RHS of the ODE system. We use relative residuals, i.e.,
/// normalized by 1 + abs(f). RMS values are computed as sqrt from the
/// normalized integrals of the squared relative residuals over each interval.
/// Integrals are estimated using 5-point Lobatto quadrature.
///
/// # Arguments
/// * `fun` - ODE function f(x, y, p)
/// * `sol` - Solution spline (PPoly)
/// * `x` - Mesh nodes (m,)
/// * `h` - Mesh intervals (m-1,)
/// * `p` - Parameters (k,)
/// * `r_middle` - Residuals at middle points (n, m-1)
/// * `f_middle` - Function values at middle points (n, m-1)
///
/// # Returns
/// RMS residuals for each interval (m-1,)
pub fn estimate_rms_residuals(
    fun: &ODEFunction,
    sol: &PPoly,
    x: &faer_col,
    h: &faer_col,
    p: &faer_col,
    r_middle: &faer_dense_mat,
    f_middle: &faer_dense_mat,
) -> faer_col {
    let (n, m1) = (r_middle.nrows(), r_middle.ncols()); // n equations, m-1 intervals

    // Compute middle points of intervals
    let mut x_middle = faer_col::zeros(m1);
    for j in 0..m1 {
        x_middle[j] = x[j] + 0.5 * h[j];
    }

    // Compute Lobatto quadrature points: x_middle ± s
    // where s = 0.5 * h * sqrt(3/7)
    let mut x1 = faer_col::zeros(m1);
    let mut x2 = faer_col::zeros(m1);
    for j in 0..m1 {
        let s = 0.5 * h[j] * ((3.0 / 7.0) as f64).sqrt();
        x1[j] = x_middle[j] + s;
        x2[j] = x_middle[j] - s;
    }

    // Evaluate solution and its derivative at quadrature points
    let x1_vec: Vec<f64> = (0..x1.nrows()).map(|i| x1[i]).collect();
    let x2_vec: Vec<f64> = (0..x2.nrows()).map(|i| x2[i]).collect();

    let y1 = sol.call(&x1_vec, &[m1], Some(0), None);
    let y2 = sol.call(&x2_vec, &[m1], Some(0), None);
    let y1_prime = sol.call(&x1_vec, &[m1], Some(1), None);
    let y2_prime = sol.call(&x2_vec, &[m1], Some(1), None);

    // Convert to faer matrices
    let y1_faer = faer_dense_mat::from_fn(y1.ncols(), y1.nrows(), |i, j| y1[(j, i)]);
    let y2_faer = faer_dense_mat::from_fn(y2.ncols(), y2.nrows(), |i, j| y2[(j, i)]);

    // Evaluate ODE RHS at quadrature points
    let f1 = fun(&x1, &y1_faer, p);
    let f2 = fun(&x2, &y2_faer, p);

    // Compute residuals: r = y' - f
    let mut r1 = faer_dense_mat::zeros(n, m1);
    let mut r2 = faer_dense_mat::zeros(n, m1);
    for j in 0..m1 {
        for i in 0..n {
            *r1.get_mut(i, j) = y1_prime[(j, i)] - f1.get(i, j);
            *r2.get_mut(i, j) = y2_prime[(j, i)] - f2.get(i, j);
        }
    }

    // Normalize residuals by (1 + |f|) and compute squared norms
    let mut r_middle_norm = faer_col::zeros(m1);
    let mut r1_norm = faer_col::zeros(m1);
    let mut r2_norm = faer_col::zeros(m1);

    for j in 0..m1 {
        let mut sum_r_middle = 0.0;
        let mut sum_r1 = 0.0;
        let mut sum_r2 = 0.0;

        for i in 0..n {
            // Normalize residuals
            let r_mid_norm = r_middle.get(i, j) / (1.0 + f_middle.get(i, j).abs());
            let r1_norm_val = r1.get(i, j) / (1.0 + f1.get(i, j).abs());
            let r2_norm_val = r2.get(i, j) / (1.0 + f2.get(i, j).abs());

            // Sum of squares (for complex numbers, this would be |r|^2)
            sum_r_middle += r_mid_norm * r_mid_norm;
            sum_r1 += r1_norm_val * r1_norm_val;
            sum_r2 += r2_norm_val * r2_norm_val;
        }

        r_middle_norm[j] = sum_r_middle;
        r1_norm[j] = sum_r1;
        r2_norm[j] = sum_r2;
    }

    // Apply 5-point Lobatto quadrature formula and take square root
    let mut rms_res = faer_col::zeros(m1);
    for j in 0..m1 {
        let integral =
            0.5 * (32.0 / 45.0 * r_middle_norm[j] + 49.0 / 90.0 * (r1_norm[j] + r2_norm[j]));
        rms_res[j] = integral.sqrt();
    }

    rms_res
}
/// Modify mesh by inserting nodes
///
/// # Arguments
/// * `x` - Current mesh nodes
/// * `insert_1` - Intervals to insert 1 node in middle
/// * `insert_2` - Intervals to insert 2 nodes (divide into 3 parts)
///
/// # Returns
/// New mesh with inserted nodes
pub fn modify_mesh(x: &faer_col, insert_1: &[usize], insert_2: &[usize]) -> faer_col {
    let mut new_points: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();

    // Insert 1 node in middle of intervals
    for &i in insert_1 {
        let mid = 0.5 * (x[i] + x[i + 1]);
        new_points.push(mid);
    }

    // Insert 2 nodes to divide interval into 3 parts
    for &i in insert_2 {
        let p1 = (2.0 * x[i] + x[i + 1]) / 3.0;
        let p2 = (x[i] + 2.0 * x[i + 1]) / 3.0;
        new_points.push(p1);
        new_points.push(p2);
    }

    new_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    faer_col::from_fn(new_points.len(), |i| new_points[i])
}

/// BVP solver with mesh refinement
pub fn solve_bvp(
    fun: &ODEFunction,
    bc: &BCFunction,
    mut x: faer_col,
    mut y: faer_dense_mat,
    p: Option<faer_col>,
    _s: Option<faer_dense_mat>, // Singular term not implemented
    fun_jac: Option<&ODEJacobian>,
    bc_jac: Option<&BCJacobian>,
    tol: f64,
    max_nodes: usize,
    verbose: u8,
    bc_tol: Option<f64>,
) -> Result<BVPResult, String> {
    let n = y.nrows();
    let mut m = x.nrows();

    if y.ncols() != m {
        return Err("y must have same number of columns as x has elements".to_string());
    }

    let mut p = p.unwrap_or_else(|| faer_col::zeros(0));
    let bc_tol = bc_tol.unwrap_or(tol);
    let max_iteration = 10;

    // Initial validation
    let f_test = fun(&x, &y, &p);
    if (f_test.nrows(), f_test.ncols()) != (y.nrows(), y.ncols()) {
        return Err(format!(
            "Function return shape ({}, {}) doesn't match y shape ({}, {})",
            f_test.nrows(),
            f_test.ncols(),
            y.nrows(),
            y.ncols()
        ));
    }

    let ya_test = faer_col::from_fn(n, |i| *y.get(i, 0));
    let yb_test = faer_col::from_fn(n, |i| *y.get(i, m - 1));
    let bc_test = bc(&ya_test, &yb_test, &p);
    let expected_bc_size = n + p.nrows();
    if bc_test.nrows() != expected_bc_size {
        return Err(format!(
            "BC return size {} doesn't match expected size {}",
            bc_test.nrows(),
            expected_bc_size
        ));
    }

    let mut status = 0;
    let mut iteration = 0;

    if verbose == 2 {
        println!(
            "{:^15}{:^15}{:^15}{:^15}{:^15}",
            "Iteration", "Max residual", "Max BC residual", "Total nodes", "Nodes added"
        );
    }

    loop {
        m = x.nrows();

        // Compute mesh intervals
        let mut h = faer_col::zeros(m - 1);
        for i in 0..(m - 1) {
            h[i] = x[i + 1] - x[i];
        }

        // Solve Newton system
        let (y_new, p_new, singular) = solve_newton(
            n,
            m,
            &h,
            fun,
            bc,
            fun_jac,
            bc_jac,
            y.clone(),
            p.clone(),
            &x,
            tol,
            bc_tol,
        );

        y = y_new;
        p = p_new;
        iteration += 1;

        // Compute collocation residuals and boundary condition residuals
        let (col_res, _y_middle, f, f_middle) = collocation_fun(fun, &y, &p, &x, &h);
        let ya_curr = faer_col::from_fn(n, |i| *y.get(i, 0));
        let yb_curr = faer_col::from_fn(n, |i| *y.get(i, m - 1));
        let bc_res = bc(&ya_curr, &yb_curr, &p);
        let max_bc_res = (0..bc_res.nrows())
            .map(|i| bc_res[i].abs())
            .fold(0.0, f64::max);

        if singular {
            status = 2;
            break;
        }

        // This relation is not trivial, but can be verified
        let mut r_middle = faer_dense_mat::zeros(n, m - 1);
        for j in 0..(m - 1) {
            for i in 0..n {
                *r_middle.get_mut(i, j) = 1.5 * col_res.get(i, j) / h[j];
            }
        }

        let sol = create_spline(&y, &f, &x, &h);
        let rms_res = estimate_rms_residuals(fun, &sol, &x, &h, &p, &r_middle, &f_middle);
        let max_rms_res = (0..rms_res.nrows()).map(|i| rms_res[i]).fold(0.0, f64::max);

        // Determine which intervals need refinement
        let mut insert_1 = Vec::new();
        let mut insert_2 = Vec::new();

        for j in 0..(m - 1) {
            if rms_res[j] > tol && rms_res[j] < 100.0 * tol {
                insert_1.push(j);
            } else if rms_res[j] >= 100.0 * tol {
                insert_2.push(j);
            }
        }

        let nodes_added = insert_1.len() + 2 * insert_2.len();

        if m + nodes_added > max_nodes {
            status = 1;
            if verbose == 2 {
                println!(
                    "{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}",
                    iteration,
                    max_rms_res,
                    max_bc_res,
                    m,
                    format!("({})", nodes_added)
                );
            }
            break;
        }

        if verbose == 2 {
            println!(
                "{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}",
                iteration, max_rms_res, max_bc_res, m, nodes_added
            );
        }

        if nodes_added > 0 {
            x = modify_mesh(&x, &insert_1, &insert_2);
            // Evaluate solution at new mesh points
            let x_eval: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
            let y_new_vals = sol.call(&x_eval, &[x.nrows()], Some(0), None);
            y = faer_dense_mat::from_fn(y_new_vals.ncols(), y_new_vals.nrows(), |i, j| {
                y_new_vals[(j, i)]
            });
        } else if max_bc_res <= bc_tol {
            status = 0;
            break;
        } else if iteration >= max_iteration {
            status = 3;
            break;
        }
    }

    if verbose > 0 {
        match status {
            0 => println!(
                "Solved in {} iterations, number of nodes {}.",
                iteration,
                x.nrows()
            ),
            1 => println!("Number of nodes exceeded after iteration {}.", iteration),
            2 => println!("Singular Jacobian encountered on iteration {}.", iteration),
            3 => println!(
                "Unable to satisfy boundary conditions tolerance on iteration {}.",
                iteration
            ),
            _ => {}
        }
    }

    let final_f = fun(&x, &y, &p);
    let final_h = faer_col::from_fn(x.nrows() - 1, |i| x[i + 1] - x[i]);
    let (col_res_final, _y_middle_final, _, f_middle_final) =
        collocation_fun(fun, &y, &p, &x, &final_h);
    let mut r_middle_final = faer_dense_mat::zeros(n, x.nrows() - 1);
    for j in 0..(x.nrows() - 1) {
        for i in 0..n {
            *r_middle_final.get_mut(i, j) = 1.5 * col_res_final.get(i, j) / final_h[j];
        }
    }
    let final_sol = create_spline(&y, &final_f, &x, &final_h);
    let final_rms_res = estimate_rms_residuals(
        fun,
        &final_sol,
        &x,
        &final_h,
        &p,
        &r_middle_final,
        &f_middle_final,
    );

    let message = match status {
        0 => "The algorithm converged to the desired accuracy.",
        1 => "The maximum number of mesh nodes is exceeded.",
        2 => "A singular Jacobian encountered when solving the collocation system.",
        3 => "The solver was unable to satisfy boundary conditions tolerance on iteration 10.",
        _ => "Unknown status",
    };

    Ok(BVPResult {
        sol: Some(final_sol),
        p: if p.nrows() > 0 { Some(p) } else { None },
        x,
        y,
        yp: final_f,
        rms_residuals: final_rms_res,
        niter: iteration,
        status,
        message: message.to_string(),
        success: status == 0,
    })
}

/// BVP solver with mesh refinement using sparse matrices
/// This version uses sparse matrix operations for better performance on large systems
pub fn solve_bvp_sparse(
    fun: &ODEFunction,
    bc: &BCFunction,
    mut x: faer_col,
    mut y: faer_dense_mat,
    p: Option<faer_col>,
    _s: Option<faer_dense_mat>, // Singular term not implemented
    fun_jac: Option<&ODEJacobian>,
    bc_jac: Option<&BCJacobian>,
    tol: f64,
    max_nodes: usize,
    verbose: u8,
    bc_tol: Option<f64>,
) -> Result<BVPResult, String> {
    let n = y.nrows();
    let mut m = x.nrows();

    if y.ncols() != m {
        return Err("y must have same number of columns as x has elements".to_string());
    }

    let mut p = p.unwrap_or_else(|| faer_col::zeros(0));
    let bc_tol = bc_tol.unwrap_or(tol);
    let max_iteration = 10;

    // Initial validation
    let f_test = fun(&x, &y, &p);
    if (f_test.nrows(), f_test.ncols()) != (y.nrows(), y.ncols()) {
        return Err(format!(
            "Function return shape ({}, {}) doesn't match y shape ({}, {})",
            f_test.nrows(),
            f_test.ncols(),
            y.nrows(),
            y.ncols()
        ));
    }

    let ya_test = faer_col::from_fn(n, |i| *y.get(i, 0));
    let yb_test = faer_col::from_fn(n, |i| *y.get(i, m - 1));
    let bc_test = bc(&ya_test, &yb_test, &p);
    let expected_bc_size = n + p.nrows();
    if bc_test.nrows() != expected_bc_size {
        return Err(format!(
            "BC return size {} doesn't match expected size {}",
            bc_test.nrows(),
            expected_bc_size
        ));
    }

    let mut status = 0;
    let mut iteration = 0;

    if verbose == 2 {
        println!(
            "{:^15}{:^15}{:^15}{:^15}{:^15}",
            "Iteration", "Max residual", "Max BC residual", "Total nodes", "Nodes added"
        );
    }

    loop {
        m = x.nrows();

        // Compute mesh intervals
        let mut h = faer_col::zeros(m - 1);
        for i in 0..(m - 1) {
            h[i] = x[i + 1] - x[i];
        }

        // Solve Newton system using sparse solver
        let (y_new, p_new, singular) = solve_newton(
            n,
            m,
            &h,
            fun,
            bc,
            fun_jac,
            bc_jac,
            y.clone(),
            p.clone(),
            &x,
            tol,
            bc_tol,
        );

        y = y_new;
        p = p_new;
        iteration += 1;

        // Compute collocation residuals and boundary condition residuals
        let (col_res, _y_middle, f, f_middle) = collocation_fun(fun, &y, &p, &x, &h);
        let ya_curr = faer_col::from_fn(n, |i| *y.get(i, 0));
        let yb_curr = faer_col::from_fn(n, |i| *y.get(i, m - 1));
        let bc_res = bc(&ya_curr, &yb_curr, &p);
        let max_bc_res = (0..bc_res.nrows())
            .map(|i| bc_res[i].abs())
            .fold(0.0, f64::max);

        if singular {
            status = 2;
            break;
        }

        // This relation is not trivial, but can be verified
        let mut r_middle = faer_dense_mat::zeros(n, m - 1);
        for j in 0..(m - 1) {
            for i in 0..n {
                *r_middle.get_mut(i, j) = 1.5 * col_res.get(i, j) / h[j];
            }
        }

        let sol = create_spline(&y, &f, &x, &h);
        let rms_res = estimate_rms_residuals(fun, &sol, &x, &h, &p, &r_middle, &f_middle);
        let max_rms_res = (0..rms_res.nrows()).map(|i| rms_res[i]).fold(0.0, f64::max);

        // Determine which intervals need refinement
        let mut insert_1 = Vec::new();
        let mut insert_2 = Vec::new();

        for j in 0..(m - 1) {
            if rms_res[j] > tol && rms_res[j] < 100.0 * tol {
                insert_1.push(j);
            } else if rms_res[j] >= 100.0 * tol {
                insert_2.push(j);
            }
        }

        let nodes_added = insert_1.len() + 2 * insert_2.len();

        if m + nodes_added > max_nodes {
            status = 1;
            if verbose == 2 {
                println!(
                    "{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}",
                    iteration,
                    max_rms_res,
                    max_bc_res,
                    m,
                    format!("({})", nodes_added)
                );
            }
            break;
        }

        if verbose == 2 {
            println!(
                "{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}",
                iteration, max_rms_res, max_bc_res, m, nodes_added
            );
        }

        if nodes_added > 0 {
            x = modify_mesh(&x, &insert_1, &insert_2);
            // Evaluate solution at new mesh points
            let x_eval: Vec<f64> = (0..x.nrows()).map(|i| x[i]).collect();
            let y_new_vals = sol.call(&x_eval, &[x.nrows()], Some(0), None);
            y = faer_dense_mat::from_fn(y_new_vals.ncols(), y_new_vals.nrows(), |i, j| {
                y_new_vals[(j, i)]
            });
        } else if max_bc_res <= bc_tol {
            status = 0;
            break;
        } else if iteration >= max_iteration {
            status = 3;
            break;
        }
    }

    if verbose > 0 {
        match status {
            0 => println!(
                "Solved in {} iterations, number of nodes {}.",
                iteration,
                x.nrows()
            ),
            1 => println!("Number of nodes exceeded after iteration {}.", iteration),
            2 => println!("Singular Jacobian encountered on iteration {}.", iteration),
            3 => println!(
                "Unable to satisfy boundary conditions tolerance on iteration {}.",
                iteration
            ),
            _ => {}
        }
    }

    let final_f = fun(&x, &y, &p);
    let final_h = faer_col::from_fn(x.nrows() - 1, |i| x[i + 1] - x[i]);
    let (col_res_final, _y_middle_final, _, f_middle_final) =
        collocation_fun(fun, &y, &p, &x, &final_h);
    let mut r_middle_final = faer_dense_mat::zeros(n, x.nrows() - 1);
    for j in 0..(x.nrows() - 1) {
        for i in 0..n {
            *r_middle_final.get_mut(i, j) = 1.5 * col_res_final.get(i, j) / final_h[j];
        }
    }
    let final_sol = create_spline(&y, &final_f, &x, &final_h);
    let final_rms_res = estimate_rms_residuals(
        fun,
        &final_sol,
        &x,
        &final_h,
        &p,
        &r_middle_final,
        &f_middle_final,
    );

    let message = match status {
        0 => "The algorithm converged to the desired accuracy.",
        1 => "The maximum number of mesh nodes is exceeded.",
        2 => "A singular Jacobian encountered when solving the collocation system.",
        3 => "The solver was unable to satisfy boundary conditions tolerance on iteration 10.",
        _ => "Unknown status",
    };

    Ok(BVPResult {
        sol: Some(final_sol),
        p: if p.nrows() > 0 { Some(p) } else { None },
        x,
        y,
        yp: final_f,
        rms_residuals: final_rms_res,
        niter: iteration,
        status,
        message: message.to_string(),
        success: status == 0,
    })
}
