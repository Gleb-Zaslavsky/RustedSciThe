
#![allow(warnings)]
extern crate nalgebra as na;
extern crate sprs;

use na::{DMatrix, DVector};
use sprs::{CsMat, CsVec};
use std::f64::EPSILON;
use crate::numerical::BVP_sci::BVP_sci_prepare::{wrap_functions, print_iteration_header, task_check, 
    estimate_bc_jac, estimate_fun_jac,  print_iteration_progress, estimate_rms_residuals, calc_F};

/* 
 

fn compute_jac_indices(n: usize, m: usize, k: usize) -> (Vec<usize>, Vec<usize>) {
    let i_col: Vec<usize> = (0..(m - 1) * n).flat_map(|i| vec![i; n]).collect();
    let j_col: Vec<usize> = (0..n)
        .cycle()
        .take(n * (m - 1))
        .zip((0..(m - 1) * n).flat_map(|i| vec![i; n]))
        .map(|(a, b)| a + b)
        .collect();

    let i_bc: Vec<usize> = ((m - 1) * n..m * n + k).flat_map(|i| vec![i; n]).collect();
    let j_bc: Vec<usize> = (0..n).cycle().take(n + k).collect();

    let i_p_col: Vec<usize> = (0..(m - 1) * n).flat_map(|i| vec![i; k]).collect();
    let j_p_col: Vec<usize> = (m * n..m * n + k)
        .cycle()
        .take((m - 1) * n * k)
        .collect();

    let i_p_bc: Vec<usize> = ((m - 1) * n..m * n + k).flat_map(|i| vec![i; k]).collect();
    let j_p_bc: Vec<usize> = (m * n..m * n + k).cycle().take((n + k) * k).collect();

    let i = [i_col.clone(), i_col, i_bc.clone(), i_bc, i_p_col, i_p_bc].concat();
    let j = [
        j_col.clone(),
        j_col.iter().map(|&x| x + n).collect(),
        j_bc.clone(),
        j_bc.iter().map(|&x| x + (m - 1) * n).collect(),
        j_p_col,
        j_p_bc,
    ]
    .concat();

    (i, j)
}

fn stacked_matmul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    if a.ncols() > 50 {
        let mut out = DMatrix::zeros(a.nrows(), b.ncols());
        for i in 0..a.nrows() {
            out.row_mut(i).copy_from(&(a.row(i) * b.row(i)));
        }
        out
    } else {
        a * b
    }
}

fn construct_global_jac(
    n: usize,
    m: usize,
    k: usize,
    i_jac: &[usize],
    j_jac: &[usize],
    h: &DVector<f64>,
    df_dy: &DMatrix<f64>,
    df_dy_middle: &DMatrix<f64>,

    dbc_dya: &DMatrix<f64>,
    dbc_dyb: &DMatrix<f64>,
  
) -> CsMat<f64> {
    let df_dy = df_dy.transpose();
    let df_dy_middle = df_dy_middle.transpose();

    let h = h.map(|x| x * x);

    let mut dPhi_dy_0 = DMatrix::identity(n, n) * -1.0;
    dPhi_dy_0 -= h / 6.0 * (df_dy.rows(0, m - 1) + 2.0 * df_dy_middle.rows(0, m - 1));
    let T = stacked_matmul(&df_dy_middle, &df_dy.rows(0, m - 1));
    dPhi_dy_0 -= h.map(|x| x * x) / 12.0 * T;

    let mut dPhi_dy_1 = DMatrix::identity(n, n);
    dPhi_dy_1 -= h / 6.0 * (df_dy.rows(1, m) + 2.0 * df_dy_middle.rows(0, m - 1));
    let T = stacked_matmul(&df_dy_middle, &df_dy.rows(1, m));
    dPhi_dy_1 += h.map(|x| x * x) / 12.0 * T;

    let mut values = vec![];
    values.extend(dPhi_dy_0.iter());
    values.extend(dPhi_dy_1.iter());
    values.extend(dbc_dya.iter());
    values.extend(dbc_dyb.iter());


    let J = CsMat::new((n * m + k, n * m + k), i_jac.to_vec(), j_jac.to_vec(), values);
    J
}

fn collocation_fun(
    fun:  Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    y: &DMatrix<f64>,

    x: &DVector<f64>,
    h: &DVector<f64>,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>,  DMatrix<f64>)

{
    let f: DMatrix<f64> = calc_F(&fun, &x, &y);
    let y_middle: DMatrix<f64> = 0.5 * (y.columns(1, y.ncols() - 1) + y.columns(0, y.ncols() - 1))
        - 0.125 * h * (f.columns(1, f.ncols() - 1) - f.columns(0, f.ncols() - 1));
    let f_middle: DMatrix<f64> = calc_F( &fun, &(x.rows(0, x.nrows() - 1) + 0.5 * h), &y_middle);
    let col_res: DMatrix<f64> = y.columns(1, y.ncols() - 1)
        - y.columns(0, y.ncols() - 1)
        - h / 6.0 * (f.columns(0, f.ncols() - 1) + f.columns(1, f.ncols() - 1) + 4.0 * &f_middle);

    (col_res, y_middle, f, f_middle)
}



type ColFun = fn(&DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>, DVector<f64>, DVector<f64>);
type BcFun = fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>;
type JacFun = fn(&DMatrix<f64>, &DVector<f64>, &DMatrix<f64>, &DVector<f64>, &DVector<f64>, &DVector<f64>) -> DMatrix<f64>;

fn solve_newton(
    n: usize,
    m: usize,
    h: &DVector<f64>,
    col_fun: ColFun,
    bc: BcFun,
    jac: JacFun,
    mut y: DMatrix<f64>,
 
    B: Option<DMatrix<f64>>,
    bvp_tol: f64,
    bc_tol: f64,
) -> (DMatrix<f64>, DVector<f64>, bool) {
    let tol_r = 2.0 / 3.0 * h * 5e-2 * bvp_tol;
    let max_njev = 4;
    let max_iter = 8;
    let sigma = 0.2;
    let tau = 0.5;
    let n_trial = 4;

    let (mut col_res, mut y_middle, mut f, mut f_middle) = col_fun(&y);
    let mut bc_res = bc(&y.column(0), &y.column(m - 1), );
    let mut res = DVector::from_iterator(col_res.len() + bc_res.len(), col_res.iter().chain(bc_res.iter()).cloned());
    let mut njev = 0;
    let mut singular = false;
    let mut recompute_jac = true;

    for _ in 0..max_iter {
        if recompute_jac {
            let J = jac(&y, &y_middle, &f, &f_middle, &bc_res);
            njev += 1;
            let LU = J.lu();
            if LU.is_singular() {
                singular = true;
                break;
            }
            let step = LU.solve(&res).unwrap();
            let cost = step.dot(&step);
        }

        let y_step = DMatrix::from_iterator(n, m, step.iter().take(n * m).cloned());
      
        let mut alpha = 1.0;

        for trial in 0..=n_trial {
            let y_new = &y - alpha * &y_step;
            if let Some(ref B) = B {
                y_new.set_column(0, &(B * y_new.column(0)));
            }
         
            let (col_res, y_middle, f, f_middle) = col_fun(&y_new);
            let bc_res = bc(&y_new.column(0), &y_new.column(m - 1));
            let res = DVector::from_iterator(col_res.len() + bc_res.len(), col_res.iter().chain(bc_res.iter()).cloned());
            let step_new = LU.solve(&res).unwrap();
            let cost_new = step_new.dot(&step_new);

            if cost_new < (1.0 - 2.0 * alpha * sigma) * cost {
                break;
            }
            if trial < n_trial {
                alpha *= tau;
            }
        }

        y = y_new;
     

        if njev == max_njev {
            break;
        }
        if col_res.iter().all(|&r| r.abs() < tol_r * (1.0 + f_middle.abs())) && bc_res.iter().all(|&r| r.abs() < bc_tol) {
            break;
        }

        if alpha == 1.0 {
            step = step_new;
            cost = cost_new;
            recompute_jac = false;
        } else {
            recompute_jac = true;
        }
    }

    (y, singular)
}




struct BVPResult {
    // Define fields as needed
}

const TERMINATION_MESSAGES: [&str; 4] = [
    "The algorithm converged to the desired accuracy.",
    "The maximum number of mesh nodes is exceeded.",
    "A singular Jacobian encountered when solving the collocation system.",
    "The solver was unable to satisfy boundary conditions tolerance on iteration 10."
];





fn modify_mesh(x: &DVector<f64>, insert_1: &DVector<usize>, insert_2: &DVector<usize>) -> DVector<f64> {
    let mut new_nodes = Vec::new();
    new_nodes.extend_from_slice(x.as_slice());
    for &i in insert_1.iter() {
        new_nodes.push(0.5 * (x[i] + x[i + 1]));
    }
    for &i in insert_2.iter() {
        new_nodes.push((2.0 * x[i] + x[i + 1]) / 3.0);
        new_nodes.push((x[i] + 2.0 * x[i + 1]) / 3.0);
    }
    new_nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    DVector::from_vec(new_nodes)
}


use std::option::Option;

fn prepare_sys(
    n: usize,
    m: usize,
    k: usize,
    fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    bc: Box<dyn Fn(&DVector<f64>, &DVector<f64>) ->   DVector<f64>> ,
    fun_jac:  Option < Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>  >,
    bc_jac:  Option < Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64>  >>,
    x: &DVector<f64>,
    h: &DVector<f64>,
    // col_fun y(&DMatrix<f64>) -> col_res, y_middle, f, f_middle (DMatrix<f64>'s)
) -> (impl Fn(&DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>),
// sys jac y: &DMatrix<f64>, y_middle: &DMatrix<f64>, f: &DMatrix<f64>, f_middle: &DMatrix<f64>, bc0: &DVector<f64> - > CsMat<f64>
     impl Fn(&DMatrix<f64>, &DVector<f64>, &DMatrix<f64>, &DMatrix<f64>, &DMatrix<f64>, &DVector<f64>) -> CsMat<f64>)
//where
 //   F: Fn(&DVector<f64>, &DMatrix<f64>, &DVector<f64>) -> DMatrix<f64> + Copy,
  //  G: Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64> + Copy,
{
    let x_middle = x.rows(0, x.nrows() - 1) + 0.5 * h;
    let (i_jac, j_jac) = compute_jac_indices(n, m, k);

    let col_fun = move |y: &DMatrix<f64>| collocation_fun(fun, y,  x, h);

    let sys_jac = move |y: &DMatrix<f64>, y_middle: &DMatrix<f64>, f: &DMatrix<f64>, f_middle: &DMatrix<f64>, bc0: &DVector<f64>| {
        let (df_dy) = if let Some(fun_jac_) = fun_jac {
            (fun_jac_(x, y), fun_jac_(&x_middle, y_middle))
        } else {
            (estimate_fun_jac(fun, x, y,  Some(f)), estimate_fun_jac(fun, &x_middle, y_middle, Some(f_middle)))
        };

        let (dbc_dya, dbc_dyb, ) = if let Some(bc_jac) = bc_jac {
            bc_jac(y.column(0), y.column(y.ncols() - 1))
        } else {
            estimate_bc_jac(bc, y.column(0), y.column(y.ncols() - 1), p, Some(bc0))
        };

        construct_global_jac(n, m, k, &i_jac, &j_jac, h, &df_dy, &df_dy_middle, &dbc_dya, &dbc_dyb, )
    };

    (col_fun, sys_jac)
}
 
struct BVPResult {
    sol: DMatrix<f64>,
    p: Option<DVector<f64>>,
    x: DVector<f64>,
    y: DMatrix<f64>,
    yp: DMatrix<f64>,
    rms_residuals: DVector<f64>,
    niter: usize,
    status: usize,
    message: String,
    success: bool,
}

fn solve_bvp(
    fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    bc: Box<dyn Fn(&DVector<f64>, &DVector<f64>) ->   DVector<f64>> ,
    x: DVector<f64>,
    y: DMatrix<f64>,
 
    S: Option<DMatrix<f64>>,
    fun_jac: Option < Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>  >,
    bc_jac: Option < Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64>  >>,
    tol: f64,
    max_nodes: usize,
    verbose: usize,
    bc_tol: Option<f64>,
) -> BVPResult {

    let (a, B, D, tol, h) = task_check( x, y, verbose,  tol, S);
    let n = y.nrows();
    let k =0.0;


    let bc_tol = bc_tol.unwrap_or(tol);

    let max_iteration = 10;

    let (fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped) = wrap_functions(
        fun, bc, fun_jac, bc_jac, a, S, D.unwrap(),
    );

    let f = calc_F(&fun_wrapped, &x.clone(), &y.clone());
    if f.shape() != y.shape() {
        panic!("`fun` return is expected to have shape {:?}, but actually has {:?}", y.shape(), f.shape());
    }

    let bc_res = bc_wrapped(&y.column(0), &y.column(y.ncols() - 1),);
    if bc_res.len() != n  {
        panic!("`bc` return is expected to have shape ({},), but actually has ({},)", n , bc_res.len());
    }

    let mut status = 0;
    let mut iteration = 0;
    if verbose == 2 {
        print_iteration_header();
    }

    loop {
        let m = x.len();

        let (col_fun, jac_sys) = prepare_sys(n, m, k, fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped, x.clone(), h.clone());
        let (y_new, p_new, singular) = solve_newton(n, m, h.clone(), col_fun, bc_wrapped, jac_sys, y.clone(), B.clone(), tol, bc_tol);
        y = y_new;
    
        iteration += 1;

        let (col_res, y_middle, f, f_middle) = colloction_fun(fun_wrapped, y.clone(),  x.clone(), h.clone());
        let bc_res = bc_wrapped(y.column(0), y.column(y.ncols() - 1), );
        let max_bc_res = bc_res.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let r_middle = 1.5 * col_res / h.clone();
        let sol = create_spline(y.clone(), f.clone(), x.clone(), h.clone());
        let rms_res = estimate_rms_residuals(fun_wrapped, sol.clone(), &x, &h, r_middle.clone(), f_middle.clone());
        let max_rms_res = rms_res.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if singular {
            status = 2;
            break;
        }

        let insert_1: Vec<usize> = rms_res.iter().enumerate().filter(|&(_, &r)| r > tol && r < 100.0 * tol).map(|(i, _)| i).collect();
        let insert_2: Vec<usize> = rms_res.iter().enumerate().filter(|&(_, &r)| r >= 100.0 * tol).map(|(i, _)| i).collect();
        let nodes_added = insert_1.len() + 2 * insert_2.len();

        if m + nodes_added > max_nodes {
            status = 1;
            if verbose == 2 {
                print_iteration_progress(iteration, max_rms_res, max_bc_res, m, nodes_added);
            }
            break;
        }

        if verbose == 2 {
            print_iteration_progress(iteration, max_rms_res, max_bc_res, m, nodes_added);
        }

        if nodes_added > 0 {
            x = modify_mesh(x.clone(), insert_1, insert_2);
            h = x.iter().zip(x.iter().skip(1)).map(|(a, b)| b - a).collect::<Vec<_>>();

            y = sol(x.clone());
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
            0 => println!("Solved in {} iterations, number of nodes {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}", iteration, x.len(), max_rms_res, max_bc_res),
            1 => println!("Number of nodes is exceeded after iteration {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}", iteration, max_rms_res, max_bc_res),
            2 => println!("Singular Jacobian encountered when solving the collocation system on iteration {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}", iteration, max_rms_res, max_bc_res),
            3 => println!("The solver was unable to satisfy boundary conditions tolerance on iteration {}. \nMaximum relative residual: {:.2e} \nMaximum boundary residual: {:.2e}", iteration, max_rms_res, max_bc_res),
            _ => (),
        }
    }

    BVPResult {
        sol,
   
        x,
        y,
        yp: f,
        rms_residuals: rms_res,
        niter: iteration,
        status,
        message: TERMINATION_MESSAGES[status].to_string(),
        success: status == 0,
    }
}

*/

/* 

fn main() {
    // Example usage of the functions
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0]);
    let y = DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let p = DVector::from_vec(vec![0.1, 0.2]);

    let fun = |x: &DVector<f64>, y: &DMatrix<f64>, p: &DVector<f64>| -> DMatrix<f64> {
        // Define your function here
        y.clone()
    };

    let bc = |ya: &DVector<f64>, yb: &DVector<f64>, p: &DVector<f64>| -> DVector<f64> {
        // Define your boundary condition function here
        ya.clone()
    };

    let (df_dy, df_dp) = estimate_fun_jac(fun, &x, &y, &p, None);
    let (dbc_dya, dbc_dyb, dbc_dp) = estimate_bc_jac(bc, &y.column(0), &y.column(1), &p, None);

    println!("df_dy: {:?}", df_dy);
    println!("df_dp: {:?}", df_dp);
    println!("dbc_dya: {:?}", dbc_dya);
    println!("dbc_dyb: {:?}", dbc_dyb);
    println!("dbc_dp: {:?}", dbc_dp);
}

use nalgebra::{DMatrix, DVector};
use std::f64::EPSILON;

fn estimate_fun_jac<F>(
    fun: F,
    x: &DVector<f64>,
    y: &DMatrix<f64>,
    p: &DVector<f64>,
    f0: Option<&DMatrix<f64>>,
) -> (DMatrix<f64>, Option<DMatrix<f64>>)
where
    F: Fn(&DVector<f64>, &DMatrix<f64>, &DVector<f64>) -> DMatrix<f64>,
{
    let (n, m) = (y.nrows(), y.ncols());
    let f0 = f0.unwrap_or(&fun(x, y, p));

    let mut df_dy = DMatrix::zeros(n, n * m);
    let h = EPSILON.sqrt() * (1.0 + y.abs());
    
    for i in 0..n {
        let mut y_new = y.clone();
        y_new.row_mut(i).add_assign(&h[i]);
        let hi = y_new.row(i) - y.row(i);
        let f_new = fun(x, &y_new, p);
        df_dy.column_mut(i).copy_from(&(&f_new - f0) / hi);
    }

    let k = p.len();
    let df_dp = if k == 0 {
        None
    } else {
        let mut df_dp = DMatrix::zeros(n, k * m);
        let h = EPSILON.sqrt() * (1.0 + p.abs());
        for i in 0..k {
            let mut p_new = p.clone();
            p_new[i] += h[i];
            let hi = p_new[i] - p[i];
            let f_new = fun(x, y, &p_new);
            df_dp.column_mut(i).copy_from(&(&f_new - f0) / hi);
        }
        Some(df_dp)
    };

    (df_dy, df_dp)
}

fn estimate_bc_jac<F>(
    bc: F,
    ya: &DVector<f64>,
    yb: &DVector<f64>,
    p: &DVector<f64>,
    bc0: Option<&DVector<f64>>,
) -> (DMatrix<f64>, DMatrix<f64>, Option<DMatrix<f64>>)
where
    F: Fn(&DVector<f64>, &DVector<f64>, &DVector<f64>) -> DVector<f64>,
{
    let n = ya.len();
    let k = p.len();
    let bc0 = bc0.unwrap_or(&bc(ya, yb, p));

    let mut dbc_dya = DMatrix::zeros(n, n + k);
    let h = EPSILON.sqrt() * (1.0 + ya.abs());
    
    for i in 0..n {
        let mut ya_new = ya.clone();
        ya_new[i] += h[i];
        let hi = ya_new[i] - ya[i];
        let bc_new = bc(&ya_new, yb, p);
        dbc_dya.column_mut(i).copy_from(&(&bc_new - bc0) / hi);
    }

    let mut dbc_dyb = DMatrix::zeros(n, n + k);
    let h = EPSILON.sqrt() * (1.0 + yb.abs());
    
    for i in 0..n {
        let mut yb_new = yb.clone();
        yb_new[i] += h[i];
        let hi = yb_new[i] - yb[i];
        let bc_new = bc(ya, &yb_new, p);
        dbc_dyb.column_mut(i).copy_from(&(&bc_new - bc0) / hi);
    }

    let dbc_dp = if k == 0 {
        None
    } else {
        let mut dbc_dp = DMatrix::zeros(n + k, k);
        let h = EPSILON.sqrt() * (1.0 + p.abs());
        for i in 0..k {
            let mut p_new = p.clone();
            p_new[i] += h[i];
            let hi = p_new[i] - p[i];
            let bc_new = bc(ya, yb, &p_new);
            dbc_dp.column_mut(i).copy_from(&(&bc_new - bc0) / hi);
        }
        Some(dbc_dp)
    };

    (dbc_dya, dbc_dyb, dbc_dp)
}
    
fn main() {
    // Example usage of the functions
    let x = DVector::from_vec(vec![0.0, 1.0, 2.0]);
    let y = DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let p = DVector::from_vec(vec![0.1, 0.2]);

    let fun = |x: &DVector<f64>, y: &DMatrix<f64>, p: &DVector<f64>| -> DMatrix<f64> {
        // Define your function here
        y.clone()
    };

    let bc = |ya: &DVector<f64>, yb: &DVector<f64>, p: &DVector<f64>| -> DVector<f64> {
        // Define your boundary condition function here
        ya.clone()
    };

    let (col_fun, sys_jac) = prepare_sys(2, 3, 2, fun, bc, None, None, &x, &DVector::from_vec(vec![1.0, 1.0]));

    let (col_res, y_middle, f, f_middle) = col_fun(&y, &p);
    let J = sys_jac(&y, &p, &y_middle, &f, &f_middle, &DVector::from_vec(vec![0.0, 0.0]));

    println!("col_res: {:?}", col_res);
    println!("y_middle: {:?}", y_middle);
    println!("f: {:?}", f);
    println!("f_middle: {:?}", f_middle);
    println!("J: {:?}", J);
}
*/