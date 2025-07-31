//! Boundary value problem solver - Rust translation from SciPy's _bvp.py
//!
//! This module implements a 4th order collocation algorithm with residual control
//! similar to the MATLAB/SciPy BVP solver, translated from Python to Rust.
//! This translation is supposed to be prorotype for further improvements,
//! because it was written using dense matrices which is not good for huge
//! problems and does not use sparse matrices. But it is fully functional
//! (seems to be) and good for small problems (single equations ans small systems).
//!  
//! This module implements a collocation method for solving nonlinear
//!  boundary value problems (BVPs) for ordinary differential equations (ODEs),
//!  inspired by SciPy's and MATLAB's solvers. The solution is approximated by
//!  a cubic spline (piecewise cubic polynomial) that is continuously
//!  differentiable (C¹) and satisfies the ODE at selected collocation points
//! (typically midpoints of mesh intervals). The mesh is adaptively refined
//! based on residual estimates.
//! Problem Statement
//! The module solves nonlinear boundary value problems (BVPs) for systems of
//!  ordinary differential equations (ODEs):

//! [ y'(x) = f(x, y(x), p), \quad x \in [a, b] ] with boundary conditions:
//!  [ g(y(a), y(b), p) = 0 ] where:
//! ( y(x) \in \mathbb{R}^n ) is the solution vector,
//! ( p \in \mathbb{R}^k ) are unknown parameters (possibly zero),
//! ( f ) is the ODE right-hand side,
//! ( g ) is the boundary condition function.
//! Algorithm Overview
//! 1) Mesh Discretization:
//! The interval ([a, b]) is divided into mesh points ( x_0, x_1, ..., x_m ).
//! 2) Collocation Method:
//!   The solution ( y(x) ) is approximated by a piecewise cubic spline.
//!  The spline is constructed so that:
//!         -It is ( C^1 ) continuous (function and first derivative are
//!         continuous).
//!         - It satisfies the ODE at collocation points (typically midpoints
//!          of mesh intervals).
//! 3) Nonlinear System Formation:
//! The collocation and boundary conditions yield a nonlinear system
//!  for the unknowns ( y(x_i) ) and ( p ).
//! 4) Newton's Method:
//! The nonlinear system is solved iteratively using Newton's method:
//! At each iteration, the Jacobian is computed (analytically or by finite differences).
//! The linearized system is solved for updates to ( y ) and ( p ).
//! 5) Mesh Refinement:
//! After each solution, the residuals are estimated. If the error is too large on some intervals, the mesh is refined by adding points.
//! 6) Spline Construction:
//! Once converged, a cubic spline interpolant is constructed for the solution.
//!
//! Main Functions and Their Mathematical Meaning
//! collocation_fun:
//! Computes the collocation residuals, i.e., the difference between the spline
//! derivative and the ODE right-hand side at collocation points.
//!
//! estimate_fun_jac / estimate_bc_jac:
//! Estimate Jacobians (partial derivatives) of the ODE and boundary condition functions using finite differences.
//!
//! construct_global_jac:
//! Assembles the global Jacobian matrix for the nonlinear system (collocation + boundary conditions).
//!
//! solve_newton:
//! Solves the nonlinear system using Newton's method, with line search and backtracking.
//!
//! estimate_rms_residuals:
//! Estimates the root-mean-square of the residuals over each mesh interval using Lobatto quadrature.
//!
//! modify_mesh:
//! Refines the mesh by inserting new nodes where residuals are large.
//!
//!solve_bvp:
//!The main driver: iteratively solves the BVP, refines the mesh, and returns
//!  the solution.
/*
function solve_bvp(f, g, x, y, p, tol, max_nodes, ...)
    repeat
        h = mesh_intervals(x)

        // 1. Solve nonlinear collocation system using Newton's method
        (y, p, singular) = solve_newton(f, g, y, p, x, h, ...)
        if singular: break

        // 2. Compute collocation and boundary residuals
        (colloc_res, _, f_vals, f_middle) = collocation_fun(f, y, p, x, h)
        bc_res = g(y[:,0], y[:,-1], p)

        // 3. Estimate interval-wise RMS residuals
        r_middle = 1.5 * colloc_res / h
        spline = create_spline(y, f_vals, x, h)
        rms_res = estimate_rms_residuals(f, spline, x, h, p, r_middle, f_middle)

        // 4. Decide where to refine mesh
        insert_1 = [j for j where tol < rms_res[j] < 100*tol]
        insert_2 = [j for j where rms_res[j] >= 100*tol]
        nodes_added = len(insert_1) + 2*len(insert_2)

        if len(x) + nodes_added > max_nodes: break

        // 5. Refine mesh if needed
        if nodes_added > 0:
            x = modify_mesh(x, insert_1, insert_2)
            y = evaluate_spline_at_new_mesh(spline, x)
        else if max(abs(bc_res)) <= tol:
            break
    until converged or max iterations reached

    return solution_struct(spline, x, y, f_vals, rms_res, ...)
end

function solve_newton(f, g, y, p, x, h, ...)
    for iter = 1 to max_iter
        // 1. Compute residuals
        (colloc_res, y_middle, f_vals, f_middle) = collocation_fun(f, y, p, x, h)
        bc_res = g(y[:,0], y[:,-1], p)
        res = stack(colloc_res, bc_res)

        // 2. Compute Jacobian (by finite diff or analytic)
        (df_dy, df_dp) = estimate_fun_jac(f, x, y, p, f_vals)
        (df_dy_middle, df_dp_middle) = estimate_fun_jac(f, x_middle, y_middle, p, f_middle)
        (dbc_dya, dbc_dyb, dbc_dp) = estimate_bc_jac(g, y[:,0], y[:,-1], p, bc_res)
        jac = construct_global_jac(df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp, ...)

        // 3. Newton step: solve jac * step = res
        step = solve_linear_system(jac, res)
        if step fails: return (y, p, singular=true)

        // 4. Line search/backtracking
        for alpha in [1, tau, tau^2, ...]:
            y_new = y - alpha * step_y
            p_new = p - alpha * step_p
            if cost(y_new, p_new) < sufficient_decrease: break

        y, p = y_new, p_new
        if converged: break
    end
    return (y, p, singular=false)
end

function collocation_fun(f, y, p, x, h)
    f_vals = f(x, y, p)
    for each interval j:
        y_middle[:,j] = 0.5*(y[:,j+1] + y[:,j]) - 0.125*h[j]*(f_vals[:,j+1] - f_vals[:,j])
        f_middle[:,j] = f(x[j]+0.5*h[j], y_middle[:,j], p)
        colloc_res[:,j] = y[:,j+1] - y[:,j] - h[j]/6*(f_vals[:,j] + f_vals[:,j+1] + 4*f_middle[:,j])
    return (colloc_res, y_middle, f_vals, f_middle)
end

function estimate_rms_residuals(f, spline, x, h, p, r_middle, f_middle)
    for each interval j:
        // Evaluate solution and derivative at Lobatto quadrature points
        x1, x2 = quadrature_points(x[j], x[j+1])
        y1, y2 = spline(x1), spline(x2)
        y1p, y2p = spline'(x1), spline'(x2)
        f1, f2 = f(x1, y1, p), f(x2, y2, p)
        r1 = y1p - f1
        r2 = y2p - f2
        // Compute normalized RMS using quadrature weights
        rms[j] = sqrt(0.5 * (32/45 * norm(r_middle[:,j])^2 + 49/90 * (norm(r1)^2 + norm(r2)^2)))
    return rms
end

*/

use crate::numerical::optimization::PPoly::{Extrapolate, PPoly};
use nalgebra::{DMatrix, DVector, LU};

/// Machine epsilon for floating point arithmetic
const EPS: f64 = f64::EPSILON;

/// Result structure for BVP solver
#[derive(Debug, Clone)]
pub struct BVPResult {
    /// Solution as cubic spline interpolator (PPoly)
    pub sol: Option<PPoly>,
    /// Found parameters (if any)
    pub p: Option<DVector<f64>>,
    /// Final mesh nodes
    pub x: DVector<f64>,
    /// Solution values at mesh nodes
    pub y: DMatrix<f64>,
    /// Solution derivatives at mesh nodes  
    pub yp: DMatrix<f64>,
    /// RMS residuals over each mesh interval
    pub rms_residuals: DVector<f64>,
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
pub type ODEFunction = dyn Fn(&DVector<f64>, &DMatrix<f64>, &DVector<f64>) -> DMatrix<f64>;

/// Function type for boundary condition evaluation  
/// Arguments: (ya, yb, p) where ya, yb are n-dimensional boundary values, p is parameters
/// Returns: (n+k)-dimensional boundary condition residuals
pub type BCFunction = dyn Fn(&DVector<f64>, &DVector<f64>, &DVector<f64>) -> DVector<f64>;

/// Function type for ODE Jacobian evaluation (optional)
/// Returns: (df_dy, df_dp) where df_dy is (n,n,m), df_dp is (n,k,m) or None
pub type ODEJacobian = dyn Fn(
    &DVector<f64>,
    &DMatrix<f64>,
    &DVector<f64>,
) -> (Vec<DMatrix<f64>>, Option<Vec<DMatrix<f64>>>);

/// Function type for boundary condition Jacobian evaluation (optional)  
/// Returns: (dbc_dya, dbc_dyb, dbc_dp) where each is appropriately sized matrix or None
pub type BCJacobian = dyn Fn(
    &DVector<f64>,
    &DVector<f64>,
    &DVector<f64>,
) -> (DMatrix<f64>, DMatrix<f64>, Option<DMatrix<f64>>);

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
    x: &DVector<f64>,
    y: &DMatrix<f64>,
    p: &DVector<f64>,
    f0: Option<&DMatrix<f64>>,
) -> (Vec<DMatrix<f64>>, Option<Vec<DMatrix<f64>>>) {
    let (n, m) = y.shape();

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
        let mut jacobian = DMatrix::zeros(n, n);

        for i in 0..n {
            let mut y_perturbed = y.clone();
            let h = EPS.sqrt() * (1.0 + y[(i, col)].abs());
            y_perturbed[(i, col)] += h;

            let x_slice = x.clone();
            let f_new = fun(&x_slice, &y_perturbed, p);

            for row in 0..n {
                jacobian[(row, i)] = (f_new[(row, col)] - f0_ref[(row, col)]) / h;
            }
        }
        df_dy.push(jacobian);
    }

    // Compute df/dp if parameters exist
    let df_dp = if p.len() == 0 {
        None
    } else {
        let k = p.len();
        let mut df_dp_vec = Vec::with_capacity(m);

        for col in 0..m {
            let mut param_jacobian = DMatrix::zeros(n, k);

            for i in 0..k {
                let mut p_perturbed = p.clone();
                let h = EPS.sqrt() * (1.0 + p[i].abs());
                p_perturbed[i] += h;

                let f_new = fun(x, y, &p_perturbed);

                for row in 0..n {
                    param_jacobian[(row, i)] = (f_new[(row, col)] - f0_ref[(row, col)]) / h;
                }
            }
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
    ya: &DVector<f64>,
    yb: &DVector<f64>,
    p: &DVector<f64>,
    bc0: Option<&DVector<f64>>,
) -> (DMatrix<f64>, DMatrix<f64>, Option<DMatrix<f64>>) {
    let n = ya.len();
    let k = p.len();

    let bc0_computed;
    let bc0_ref = match bc0 {
        Some(bc_val) => bc_val,
        None => {
            bc0_computed = bc(ya, yb, p);
            &bc0_computed
        }
    };

    // Compute dbc/dya
    let mut dbc_dya = DMatrix::zeros(n + k, n);
    for i in 0..n {
        let mut ya_perturbed = ya.clone();
        let h = EPS.sqrt() * (1.0 + ya[i].abs());
        ya_perturbed[i] += h;

        let bc_new = bc(&ya_perturbed, yb, p);
        for row in 0..(n + k) {
            dbc_dya[(row, i)] = (bc_new[row] - bc0_ref[row]) / h;
        }
    }

    // Compute dbc/dyb
    let mut dbc_dyb = DMatrix::zeros(n + k, n);
    for i in 0..n {
        let mut yb_perturbed = yb.clone();
        let h = EPS.sqrt() * (1.0 + yb[i].abs());
        yb_perturbed[i] += h;

        let bc_new = bc(ya, &yb_perturbed, p);
        for row in 0..(n + k) {
            dbc_dyb[(row, i)] = (bc_new[row] - bc0_ref[row]) / h;
        }
    }

    // Compute dbc/dp if parameters exist
    let dbc_dp = if k == 0 {
        None
    } else {
        let mut dbc_dp_mat = DMatrix::zeros(n + k, k);
        for i in 0..k {
            let mut p_perturbed = p.clone();
            let h = EPS.sqrt() * (1.0 + p[i].abs());
            p_perturbed[i] += h;

            let bc_new = bc(ya, yb, &p_perturbed);
            for row in 0..(n + k) {
                dbc_dp_mat[(row, i)] = (bc_new[row] - bc0_ref[row]) / h;
            }
        }
        Some(dbc_dp_mat)
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
/// This is a simplified version - the Python original had optimization for large matrices
pub fn stacked_matmul(a: &[DMatrix<f64>], b: &[DMatrix<f64>]) -> Vec<DMatrix<f64>> {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(a_mat, b_mat)| a_mat * b_mat)
        .collect()
}

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
    y: &DMatrix<f64>,
    p: &DVector<f64>,
    x: &DVector<f64>,
    h: &DVector<f64>,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let (n, m) = y.shape();

    // Evaluate RHS at mesh nodes
    let f = fun(x, y, p);

    // Compute solution values at middle points using cubic interpolation formula
    let mut y_middle = DMatrix::zeros(n, m - 1);
    for i in 0..n {
        for j in 0..(m - 1) {
            y_middle[(i, j)] =
                0.5 * (y[(i, j + 1)] + y[(i, j)]) - 0.125 * h[j] * (f[(i, j + 1)] - f[(i, j)]);
        }
    }

    // Evaluate RHS at middle points
    // The ODE function expects x and y to have consistent dimensions
    // We evaluate the function at each middle point individually
    let mut f_middle = DMatrix::zeros(n, m - 1);
    for j in 0..(m - 1) {
        let x_mid = x[j] + 0.5 * h[j];
        let x_single = DVector::from_vec(vec![x_mid]);

        // Extract the j-th column of y_middle for evaluation as a column vector
        let mut y_single = DMatrix::zeros(n, 1);
        for i in 0..n {
            y_single[(i, 0)] = y_middle[(i, j)];
        }

        let f_result = fun(&x_single, &y_single, p);

        // Copy the result to the j-th column of f_middle
        for i in 0..n {
            f_middle[(i, j)] = f_result[(i, 0)];
        }
    }

    // Compute collocation residuals
    let mut col_res = DMatrix::zeros(n, m - 1);
    for i in 0..n {
        for j in 0..(m - 1) {
            col_res[(i, j)] = y[(i, j + 1)]
                - y[(i, j)]
                - h[j] / 6.0 * (f[(i, j)] + f[(i, j + 1)] + 4.0 * f_middle[(i, j)]);
        }
    }

    (col_res, y_middle, f, f_middle)
}

/// Construct the Jacobian of the collocation system using dense matrices
///
/// This is a simplified version using dense matrices instead of sparse matrices
/// for the minimal working example.
///
/// # Arguments
/// * `n` - Number of equations in the ODE system
/// * `m` - Number of nodes in the mesh
/// * `k` - Number of unknown parameters  
/// * `h` - Mesh intervals (m-1,)
/// * `df_dy` - Jacobian of f w.r.t. y at mesh nodes: Vec of (n,n) matrices
/// * `df_dy_middle` - Jacobian of f w.r.t. y at middle points: Vec of (n,n) matrices  
/// * `df_dp` - Jacobian of f w.r.t. p at mesh nodes: Vec of (n,k) matrices (or None)
/// * `df_dp_middle` - Jacobian of f w.r.t. p at middle points: Vec of (n,k) matrices (or None)
/// * `dbc_dya` - Jacobian of bc w.r.t. ya: (n+k, n) matrix
/// * `dbc_dyb` - Jacobian of bc w.r.t. yb: (n+k, n) matrix
/// * `dbc_dp` - Jacobian of bc w.r.t. p: (n+k, k) matrix (or None)
///
/// # Returns
/// * Dense Jacobian matrix of size (n*m + k, n*m + k)
pub fn construct_global_jac(
    n: usize,
    m: usize,
    k: usize,
    h: &DVector<f64>,
    df_dy: &[DMatrix<f64>],
    df_dy_middle: &[DMatrix<f64>],
    df_dp: Option<&[DMatrix<f64>]>,
    df_dp_middle: Option<&[DMatrix<f64>]>,
    dbc_dya: &DMatrix<f64>,
    dbc_dyb: &DMatrix<f64>,
    dbc_dp: Option<&DMatrix<f64>>,
) -> DMatrix<f64> {
    let total_size = (m - 1) * n + (n + k);
    let mut jac = DMatrix::zeros(total_size, total_size);

    // Process diagonal and off-diagonal blocks for collocation residuals
    for i in 0..(m - 1) {
        let h_i = h[i];

        // Compute diagonal n×n block (dPhi_dy_0)
        let mut dphi_dy_0 = -DMatrix::identity(n, n);
        dphi_dy_0 -= (h_i / 6.0) * (&df_dy[i] + 2.0 * &df_dy_middle[i]);

        // T = df_dy_middle[i] * df_dy[i]
        let t_diag = &df_dy_middle[i] * &df_dy[i];
        dphi_dy_0 -= (h_i * h_i / 12.0) * t_diag;

        // Insert diagonal block
        let row_start = i * n;
        let col_start = i * n;
        for r in 0..n {
            for c in 0..n {
                jac[(row_start + r, col_start + c)] = dphi_dy_0[(r, c)];
            }
        }

        // Compute off-diagonal n×n block (dPhi_dy_1)
        let mut dphi_dy_1 = DMatrix::identity(n, n);
        dphi_dy_1 -= (h_i / 6.0) * (&df_dy[i + 1] + 2.0 * &df_dy_middle[i]);

        // T = df_dy_middle[i] * df_dy[i+1]
        let t_off = &df_dy_middle[i] * &df_dy[i + 1];
        dphi_dy_1 += (h_i * h_i / 12.0) * t_off;

        // Insert off-diagonal block
        let col_start_off = (i + 1) * n;
        for r in 0..n {
            for c in 0..n {
                jac[(row_start + r, col_start_off + c)] = dphi_dy_1[(r, c)];
            }
        }

        // Handle parameter derivatives if present
        if let (Some(df_dp_vec), Some(df_dp_middle_vec)) = (df_dp, df_dp_middle) {
            // T = df_dy_middle[i] * (df_dp[i] - df_dp[i+1])
            let dp_diff = &df_dp_vec[i] - &df_dp_vec[i + 1];
            let t_param = &df_dy_middle[i] * dp_diff;

            let mut dphi_dp =
                -h_i / 6.0 * (&df_dp_vec[i] + &df_dp_vec[i + 1] + 4.0 * &df_dp_middle_vec[i]);
            dphi_dp += 0.125 * h_i * t_param;

            // Insert parameter block
            let param_col_start = n * m;
            for r in 0..n {
                for c in 0..k {
                    jac[(row_start + r, param_col_start + c)] = dphi_dp[(r, c)];
                }
            }
        }
    }

    // Insert boundary condition blocks
    let bc_row_start = (m - 1) * n;

    // dbc_dya block - dependency on ya
    for r in 0..(n + k) {
        for c in 0..n {
            jac[(bc_row_start + r, c)] = dbc_dya[(r, c)];
        }
    }

    // dbc_dyb block - dependency on yb
    let yb_col_start = (m - 1) * n;
    for r in 0..(n + k) {
        for c in 0..n {
            jac[(bc_row_start + r, yb_col_start + c)] = dbc_dyb[(r, c)];
        }
    }

    // dbc_dp block - dependency on parameters
    if let Some(dbc_dp_mat) = dbc_dp {
        let param_col_start = n * m;
        for r in 0..(n + k) {
            for c in 0..k {
                jac[(bc_row_start + r, param_col_start + c)] = dbc_dp_mat[(r, c)];
            }
        }
    }

    jac
}

/// Solve the nonlinear collocation system by Newton's method with dense matrices
///
/// This is a simplified version using dense LU decomposition instead of sparse solver.
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
    h: &DVector<f64>,
    fun: &ODEFunction,
    bc: &BCFunction,
    fun_jac: Option<&ODEJacobian>,
    bc_jac: Option<&BCJacobian>,
    mut y: DMatrix<f64>,
    mut p: DVector<f64>,
    x: &DVector<f64>,
    bvp_tol: f64,
    bc_tol: f64,
) -> (DMatrix<f64>, DVector<f64>, bool) {
    let k = p.len();

    // Newton iteration parameters
    let max_iter = 8;
    let max_njev = 4;
    let sigma = 0.2; // Armijo constant
    let tau = 0.5; // Step size decrease factor
    let n_trial = 4; // Max backtracking steps

    // Tolerance for collocation residuals
    let tol_r: DVector<f64> = h.map(|h_i| 2.0 / 3.0 * h_i * 5e-2 * bvp_tol);

    let mut njev = 0;
    let mut singular = false;
    let mut recompute_jac = true;
    let mut lu_decomp: Option<LU<f64, nalgebra::Dyn, nalgebra::Dyn>> = None;
    let mut cost = 0.0;

    for _iteration in 0..max_iter {
        // Compute collocation residuals and function values
        let (col_res, y_middle, f, f_middle) = collocation_fun(fun, &y, &p, x, h);
        let bc_res = bc(&y.column(0).into(), &y.column(m - 1).into(), &p);

        // Stack residuals into single vector (Python uses Fortran order: column-major)
        let mut res = DVector::zeros((m - 1) * n + (n + k));

        // Collocation residuals (column-major order like Python)
        for j in 0..(m - 1) {
            for i in 0..n {
                res[j * n + i] = col_res[(i, j)];
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

            let mut x_middle = DVector::zeros(m - 1);
            for j in 0..(m - 1) {
                x_middle[j] = x[j] + 0.5 * h[j];
            }

            let (df_dy_middle, df_dp_middle) = if let Some(jac_fn) = fun_jac {
                jac_fn(&x_middle, &y_middle, &p)
            } else {
                estimate_fun_jac(fun, &x_middle, &y_middle, &p, Some(&f_middle))
            };

            let (dbc_dya, dbc_dyb, dbc_dp) = if let Some(bc_jac_fn) = bc_jac {
                bc_jac_fn(&y.column(0).into(), &y.column(m - 1).into(), &p)
            } else {
                estimate_bc_jac(
                    bc,
                    &y.column(0).into(),
                    &y.column(m - 1).into(),
                    &p,
                    Some(&bc_res),
                )
            };

            // Construct global Jacobian
            let jac_matrix = construct_global_jac(
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

            // Attempt LU decomposition
            let lu = LU::new(jac_matrix);
            match lu.solve(&res) {
                Some(step) => {
                    lu_decomp = Some(lu);
                    cost = step.dot(&step);
                }
                None => {
                    singular = true;
                    break;
                }
            }
            njev += 1;
        }

        if let Some(ref lu) = lu_decomp {
            if let Some(step) = lu.solve(&res) {
                // Extract step components
                let mut y_step = DMatrix::zeros(n, m);
                for j in 0..m {
                    for i in 0..n {
                        y_step[(i, j)] = step[j * n + i];
                    }
                }

                let p_step = if k > 0 {
                    step.rows((m - 1) * n + (n + k) - k, k).into()
                } else {
                    DVector::zeros(0)
                };

                // Backtracking line search
                let mut alpha = 1.0;
                let mut best_y = y.clone();
                let mut best_p = p.clone();
                let mut best_cost = cost;

                for trial in 0..=n_trial {
                    let y_new = &y - alpha * &y_step;
                    let p_new = &p - alpha * &p_step;

                    // Compute new residuals
                    let (col_res_new, _y_middle_new, _f_new, _f_middle_new) =
                        collocation_fun(fun, &y_new, &p_new, x, h);
                    let bc_res_new =
                        bc(&y_new.column(0).into(), &y_new.column(m - 1).into(), &p_new);

                    let mut res_new = DVector::zeros((m - 1) * n + (n + k));
                    for j in 0..(m - 1) {
                        for i in 0..n {
                            res_new[j * n + i] = col_res_new[(i, j)];
                        }
                    }
                    for i in 0..(n + k) {
                        res_new[(m - 1) * n + i] = bc_res_new[i];
                    }

                    if let Some(step_new) = lu.solve(&res_new) {
                        let cost_new = step_new.dot(&step_new);
                        if cost_new < (1.0 - 2.0 * alpha * sigma) * cost {
                            best_y = y_new;
                            best_p = p_new;
                            best_cost = cost_new;
                            break;
                        }
                    }

                    if trial < n_trial {
                        alpha *= tau;
                    }
                }

                y = best_y;
                p = best_p;
                cost = best_cost;

                // Check convergence
                let (col_res_final, _, _f_final, f_middle_final) =
                    collocation_fun(fun, &y, &p, x, h);
                let bc_res_final = bc(&y.column(0).into(), &y.column(m - 1).into(), &p);

                let mut converged = true;

                // Check collocation residuals
                for j in 0..(m - 1) {
                    for i in 0..n {
                        if col_res_final[(i, j)].abs()
                            >= tol_r[j] * (1.0 + f_middle_final[(i, j)].abs())
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
pub fn create_spline(
    y: &DMatrix<f64>,
    yp: &DMatrix<f64>,
    x: &DVector<f64>,
    h: &DVector<f64>,
) -> PPoly {
    let (n, m) = y.shape();

    // Initialize coefficient array: c[degree][interval][polynomial]
    // For cubic spline: degree = 4 (coefficients for x^3, x^2, x^1, x^0)
    let mut c = vec![vec![vec![0.0; n]; m - 1]; 4];

    // Compute cubic spline coefficients for each interval and each polynomial
    for j in 0..(m - 1) {
        for i in 0..n {
            // Compute slope over interval
            let slope = (y[(i, j + 1)] - y[(i, j)]) / h[j];

            // Compute t parameter: (yp_left + yp_right - 2*slope) / h
            let t = (yp[(i, j)] + yp[(i, j + 1)] - 2.0 * slope) / h[j];

            // Cubic spline coefficients (highest degree first)
            c[0][j][i] = t / h[j]; // x^3 coefficient
            c[1][j][i] = (slope - yp[(i, j)]) / h[j] - t; // x^2 coefficient  
            c[2][j][i] = yp[(i, j)]; // x^1 coefficient
            c[3][j][i] = y[(i, j)]; // x^0 coefficient
        }
    }

    PPoly {
        c,
        x: x.iter().cloned().collect(),
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
    x: &DVector<f64>,
    h: &DVector<f64>,
    p: &DVector<f64>,
    r_middle: &DMatrix<f64>,
    f_middle: &DMatrix<f64>,
) -> DVector<f64> {
    let (n, m1) = r_middle.shape(); // n equations, m-1 intervals

    // Compute middle points of intervals
    let mut x_middle = DVector::zeros(m1);
    for j in 0..m1 {
        x_middle[j] = x[j] + 0.5 * h[j];
    }

    // Compute Lobatto quadrature points: x_middle ± s
    // where s = 0.5 * h * sqrt(3/7)
    let mut x1 = DVector::zeros(m1);
    let mut x2 = DVector::zeros(m1);
    for j in 0..m1 {
        let s = 0.5 * h[j] * ((3.0 / 7.0) as f64).sqrt();
        x1[j] = x_middle[j] + s;
        x2[j] = x_middle[j] - s;
    }

    // Evaluate solution and its derivative at quadrature points
    let y1 = sol.call(&x1.data.as_vec(), &[m1], Some(0), None);
    let y2 = sol.call(&x2.data.as_vec(), &[m1], Some(0), None);
    let y1_prime = sol.call(&x1.data.as_vec(), &[m1], Some(1), None);
    let y2_prime = sol.call(&x2.data.as_vec(), &[m1], Some(1), None);

    // Evaluate ODE RHS at quadrature points
    let f1 = fun(&x1, &y1.transpose(), p);
    let f2 = fun(&x2, &y2.transpose(), p);

    // Compute residuals: r = y' - f
    let mut r1 = DMatrix::zeros(n, m1);
    let mut r2 = DMatrix::zeros(n, m1);
    for j in 0..m1 {
        for i in 0..n {
            r1[(i, j)] = y1_prime[(j, i)] - f1[(i, j)];
            r2[(i, j)] = y2_prime[(j, i)] - f2[(i, j)];
        }
    }

    // Normalize residuals by (1 + |f|) and compute squared norms
    let mut r_middle_norm = DVector::zeros(m1);
    let mut r1_norm = DVector::zeros(m1);
    let mut r2_norm = DVector::zeros(m1);

    for j in 0..m1 {
        let mut sum_r_middle = 0.0;
        let mut sum_r1 = 0.0;
        let mut sum_r2 = 0.0;

        for i in 0..n {
            // Normalize residuals
            let r_mid_norm = r_middle[(i, j)] / (1.0 + f_middle[(i, j)].abs());
            let r1_norm_val = r1[(i, j)] / (1.0 + f1[(i, j)].abs());
            let r2_norm_val = r2[(i, j)] / (1.0 + f2[(i, j)].abs());

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
    let mut rms_res = DVector::zeros(m1);
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
pub fn modify_mesh(x: &DVector<f64>, insert_1: &[usize], insert_2: &[usize]) -> DVector<f64> {
    let mut new_points = x.iter().cloned().collect::<Vec<f64>>();

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
    DVector::from_vec(new_points)
}

/// BVP solver with mesh refinement
pub fn solve_bvp(
    fun: &ODEFunction,
    bc: &BCFunction,
    mut x: DVector<f64>,
    mut y: DMatrix<f64>,
    p: Option<DVector<f64>>,
    _s: Option<DMatrix<f64>>, // Singular term not implemented
    fun_jac: Option<&ODEJacobian>,
    bc_jac: Option<&BCJacobian>,
    tol: f64,
    max_nodes: usize,
    verbose: u8,
    bc_tol: Option<f64>,
) -> Result<BVPResult, String> {
    let n = y.nrows();
    let mut m = x.len();

    if y.ncols() != m {
        return Err("y must have same number of columns as x has elements".to_string());
    }

    let mut p = p.unwrap_or_else(|| DVector::zeros(0));
    let bc_tol = bc_tol.unwrap_or(tol);
    let max_iteration = 10;

    // Initial validation
    let f_test = fun(&x, &y, &p);
    if f_test.shape() != y.shape() {
        return Err(format!(
            "Function return shape {:?} doesn't match y shape {:?}",
            f_test.shape(),
            y.shape()
        ));
    }

    let bc_test = bc(&y.column(0).into(), &y.column(m - 1).into(), &p);
    let expected_bc_size = n + p.len();
    if bc_test.len() != expected_bc_size {
        return Err(format!(
            "BC return size {} doesn't match expected size {}",
            bc_test.len(),
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
        m = x.len();

        // Compute mesh intervals
        let mut h = DVector::zeros(m - 1);
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
        let bc_res = bc(&y.column(0).into(), &y.column(m - 1).into(), &p);
        let max_bc_res = bc_res.iter().map(|x| x.abs()).fold(0.0, f64::max);

        if singular {
            status = 2;
            break;
        }

        // This relation is not trivial, but can be verified
        let mut r_middle = DMatrix::zeros(n, m - 1);
        for j in 0..(m - 1) {
            for i in 0..n {
                r_middle[(i, j)] = 1.5 * col_res[(i, j)] / h[j];
            }
        }

        let sol = create_spline(&y, &f, &x, &h);
        let rms_res = estimate_rms_residuals(fun, &sol, &x, &h, &p, &r_middle, &f_middle);
        let max_rms_res = rms_res.iter().cloned().fold(0.0, f64::max);

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
            let x_eval: Vec<f64> = x.iter().cloned().collect();
            let y_new_vals = sol.call(&x_eval, &[x.len()], Some(0), None);
            y = y_new_vals.transpose();
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
                x.len()
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
    let final_h = DVector::from_iterator(x.len() - 1, (0..x.len() - 1).map(|i| x[i + 1] - x[i]));
    let (col_res_final, _y_middle_final, _, f_middle_final) =
        collocation_fun(fun, &y, &p, &x, &final_h);
    let mut r_middle_final = DMatrix::zeros(n, x.len() - 1);
    for j in 0..(x.len() - 1) {
        for i in 0..n {
            r_middle_final[(i, j)] = 1.5 * col_res_final[(i, j)] / final_h[j];
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
        p: if p.len() > 0 { Some(p) } else { None },
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
