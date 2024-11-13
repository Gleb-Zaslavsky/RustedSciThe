#![allow(warnings)]
extern crate nalgebra as na;
extern crate sprs;
use std::ops::AddAssign;
use na::{DMatrix, DVector, Complex};
use sprs::{CsMat, CsVec};
use std::f64::EPSILON;

use approx::relative_eq;
pub fn task_check( x: DVector<f64>, y: DMatrix<f64>, verbose: usize,  tol: f64, S: Option<DMatrix<f64>>) -> (f64,  Option<DMatrix<f64>>, Option<DMatrix<f64>>, f64, DVector<f64>) {
    let x = x.clone();
    if x.len() < 2 {
        panic!("`x` must be 1 dimensional.");
    }
    let h = x.iter().zip(x.iter().skip(1)).map(|(a, b)| b - a).collect::<Vec<_>>();
    let h: DVector<f64> = DVector::from_vec(h);
    if h.iter().any(|&h| h <= 0.0) {
        panic!("`x` must be strictly increasing.");
    }
    let a = x[0];

    let mut y = y.clone();
    let dtype = if y.iter().any(|&v| v.is_nan()) { "complex" } else { "float" };

    if y.nrows() != x.len() {
        panic!("`y` is expected to have {} columns, but actually has {}.", x.len(), y.nrows());
    }


    let tol = if tol < 100.0 * EPSILON {
        println!("`tol` is too low, setting to {:.2e}", 100.0 * EPSILON);
        100.0 * EPSILON
    } else {
        tol
    };

    if ![0, 1, 2].contains(&verbose) {
        panic!("`verbose` must be in [0, 1, 2].");

    }
    let n = y.nrows();
    let (B, D) = if let Some(S) = S {
        let S = S.clone();
        if S.shape() != (n, n) {
            panic!("`S` is expected to have shape ({}, {}), but actually has {:?}", n, n, S.shape());
        }
        let B = DMatrix::identity(n, n) - S.clone().pseudo_inverse(1e-10).unwrap() * S.clone();
        y.set_column(0, &(B.clone() * y.column(0)));
        let D = (DMatrix::identity(n, n) - S.clone()).pseudo_inverse(1e-10).unwrap();
        (Some(B), Some(D))
    } else {
        (None, None)
    };

    (a, B, D, tol, h)
}

type Fun = Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>;
type Bc = Box<dyn Fn(&DVector<f64>, &DVector<f64>) ->   DVector<f64>> ;
type FunJac =   Option < Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>  >;
type BcJac = Option < Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64>  >>; //fn(&DVector<f64>, &DVector<f64>, &DVector<f64>) -> fn(&DVector<f64>, &DVector<f64>, &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>);

pub fn wrap_functions(
    fun: Fun,
    bc: Bc,
    fun_jac: FunJac,
    bc_jac: BcJac,

    a: f64,
    S: Option<DMatrix<f64>>,
    D: DMatrix<f64>,
) -> (Fun, Bc, FunJac, BcJac) {


    let fun_jac_wrapped: Option<Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>> = if let Some(fun_jac) = fun_jac {
        Some(Box::new(move |x, y| {
            fun_jac(x, y)
        }))
    } else {
        None
    };

    let S1= S.clone();
    let D1 = D.clone();
    let fun_wrapped: Fun = if let Some( S) = S1 {
        Box::new(move |x: f64, y: &DVector<f64>| {
            let mut f = fun(x, y);
            if x == a {
                f.set_column(0, &(D1.clone() * f.column(0)));
                f.columns_mut(1, f.ncols() - 1).add_assign(&(S.clone() * y.columns(1, y.ncols() - 1) / (x - a)));
            } else {
                f.add_assign(&(S.clone() * y / (x - a)));
            }
            f
        })
    } else {
        fun
    };  

    let fun_jac_wrapped: FunJac = if let Some(fun_jac_wrapped) = fun_jac_wrapped {
        if let Some(S) = S.clone() {
            Some(Box::new(move |x: f64, y: &DVector<f64>,           | {
                let mut df_dy = fun_jac_wrapped(x, y);
                if x == a {
                    df_dy.set_column(0, &(D.clone() * df_dy.column(0)));
                    df_dy.columns_mut(1, df_dy.ncols() - 1).add_assign(&(S.clone() / (x - a)));
                } else {
                    df_dy.add_assign(&(S.clone() / (x - a)));
                }
                df_dy
            })) 
        } else {
            Some(fun_jac_wrapped)
        }
    } else {
        None
    };
    let bc_jac_wrapped = bc_jac;
    let bc_wrapped = bc;
    (fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped)
}


pub fn print_iteration_header() {
    println!("{:^15}{:^15}{:^15}{:^15}{:^15}", 
             "Iteration", "Max residual", "Max BC residual", "Total nodes", "Nodes added");
}

pub fn print_iteration_progress(iteration: usize, residual: f64, bc_residual: f64, total_nodes: usize, nodes_added: usize) {
    println!("{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}", 
             iteration, residual, bc_residual, total_nodes, nodes_added);
}

pub fn calc_F(
    fun:  &Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    x: &DVector<f64>,
    y: &DMatrix<f64>,
) -> DMatrix<f64> {
    let mut F = DMatrix::zeros(x.len(), y.ncols());
    for i in 0..x.len() {
      
        let f_i = fun(x[i], &y.row(i).transpose());
  
        F.set_row(i, &f_i.transpose());
    }
    F
}
   

pub fn estimate_fun_jac(
    fun:  Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    x: &DVector<f64>,
    y: &DMatrix<f64>,

    f0: Option< Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>  >,
) -> (Vec<DMatrix<f64>>)

{   
    let (n, m) = y.shape();
    let x_len = x.len();
    assert_eq!(n, x_len, "x and y have different lengths");
    let f0:DMatrix<f64> = if let Some(f0_) = f0 {
          calc_F(&f0_, x, &y)
    } else   {
      
        calc_F(&fun, x, &y)
       
    }; 
    let mut y_abs = y.abs();
    y_abs.add_assign(  DMatrix::from_element(n, m, 1.0) );
    let y_plus_1: DMatrix<f64> = y_abs;
    let h =   y_plus_1 * (EPSILON.sqrt());  
    let mut Jac = Vec::new();
   
    for i in 0..m {
        let mut df_dy = DMatrix::zeros( n,  m);
        let mut y_new = y.clone();
        // y_new[i]=+h[i];
        y_new.column_mut(i).add_assign( &(h.column(i)) );
        let hi = y_new.column(i) - y.column(i);
        let f_new = calc_F(&fun, x, &y_new);
       // print!("\n \n {}, hi {:?},  \n \n f_new {:?}, \n \n f0 {:?} ", i, hi, f_new, f0);
        let dy_dx_j = (&f_new.column(i) - f0.clone().column(i)).component_div(&hi);
       // println!("\n \n dy_dx_j {:?} \n \n",  dy_dx_j);
        df_dy.set_column(i, &dy_dx_j);
       // println!("dy_dx {:?},",  df_dy);

        Jac.push(df_dy);
    }



    Jac
}



pub fn estimate_bc_jac(
    bc:  Box<dyn Fn(&DVector<f64>, &DVector<f64>) ->   DVector<f64>> ,
    ya: &DVector<f64>,
    yb: &DVector<f64>,
    bc0: Option<  Box<dyn Fn(&DVector<f64>, &DVector<f64>) ->   DVector<f64>>  > ,
) -> (DMatrix<f64>, DMatrix<f64>)

{
    let n = ya.len();
    let bc0:DVector<f64> = if let Some(bc0_) = bc0 {
        bc0_(ya, yb)
  } else   {
     bc(ya, yb)
  };



    let mut dbc_dya = DMatrix::zeros(n, n );
    let mut y_abs = ya.abs();
    y_abs.add_assign(  DMatrix::from_element(n, 1, 1.0) );
    let y_plus_1: DVector<f64> = y_abs;
    let h =   y_plus_1 * (EPSILON.sqrt());  

    for i in 0..n {
        let mut ya_new = ya.clone();
        ya_new[i] += h[i];
        let hi = ya_new[i] - ya[i];
        let bc_new = bc(&ya_new, yb);
        let dy_dx_j = (&bc_new - bc0.clone())/(hi);
        dbc_dya.set_row(i, &dy_dx_j.transpose());
    }

    let mut y_abs = yb.abs();
    y_abs.add_assign(  DMatrix::from_element(n, 1, 1.0) );
    let y_plus_1: DVector<f64> = y_abs;
    let h =   y_plus_1 * (EPSILON.sqrt());  

    let mut dbc_dyb = DMatrix::zeros(n, n );
    for i in 0..n {
        let mut yb_new = yb.clone();
        yb_new[i] += h[i];
        let hi = yb_new[i] - yb[i];
        let bc_new = bc(ya, &yb_new);
        let dy_dx_j = (&bc_new - bc0.clone())/(hi);
        dbc_dyb.set_row(i, &dy_dx_j.transpose());
    }

 
    (dbc_dya, dbc_dyb )
}//estimate_bc_jac


fn create_spline(y: &DMatrix<f64>, yp: &DMatrix<f64>, x: &DVector<f64>, h: &DVector<f64>) -> DMatrix<f64> {
    let (n, m) = y.shape();
    let mut c = DMatrix::zeros(4, n * (m - 1));
    let slope = (y.columns(1, m - 1) - y.columns(0, m - 1)).component_div(&h);
    let t = (yp.columns(0, m - 1) + yp.columns(1, m - 1) - 2.0 * &slope).component_div(& h);
    c.row_mut(0).copy_from(&(t.component_div(& h)  ));
    c.row_mut(1).copy_from(&((slope - yp.columns(0, m - 1)).component_div(& h) - t));
    c.row_mut(2).copy_from(&yp.columns(0, m - 1));
    c.row_mut(3).copy_from(&y.columns(0, m - 1));
    c
}

pub fn estimate_rms_residuals(
    fun:  Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    sol: fn(&DVector<f64>, bool) -> DMatrix<f64>,
    x: &DVector<f64>,
    h: &DVector<f64>,

    r_middle: &DMatrix<f64>,
    f_middle: &DMatrix<f64>
) -> DVector<f64> {
    let x_len = x.len();
    let h_len = h.len();
    assert_eq!(x_len-1,h_len, "x_len-1 should be equal to h_len");
    let x_middle = x.rows(0, x.len() - 1) + 0.5 * h;
   
    let s = 0.5 * h * ((3.0 / 7.0)as f64).sqrt();
   
    let x1 = &x_middle + &s;
    let x2 = &x_middle - &s;
    let y1 = sol(&x1, false);
    let y2 = sol(&x2, false);
    let y1_prime = sol(&x1, true); // Assuming sol(x, 1) is the derivative
    let y2_prime = sol(&x2, true); // Assuming sol(x, 1) is the derivative
    let f1 = calc_F(&fun, &x1, &y1);
    let f2 = calc_F(&fun, &x2, &y2,);
    let r1 = &y1_prime - &f1;
    let r2 = &y2_prime - &f2;
    let mut f_middle_abs = f_middle.abs();
    let (n, m ) = f_middle.shape();
    f_middle_abs.add_assign(  DMatrix::from_element(n, m, 1.0) );
    let f_middle_plus_1: DMatrix<f64> = f_middle_abs;
    let r_middle: DMatrix<f64> = r_middle.component_div (& f_middle_plus_1);
    let f1_pl_1: DMatrix<f64> = DMatrix::from_iterator(f1.len(), 1, f1.abs().iter().map(|x| x + 1.0));
    let r1:DMatrix<f64> = r1 .component_div  (  &f1_pl_1  );    
    let f2_pl_1 = DMatrix::from_iterator(f2.len(), 1, f2.abs().iter().map(|x| x + 1.0));
    let r2 = r2  .component_div  (  &( f2.abs()  ));
   
    let r1_conj_square = r1.component_mul(&r1.conjugate());
   // let r1_conj_square_real = r1_conj_square.map(|x| x.re);
    let r1:DVector<f64> = r1_conj_square.column_sum();
    let r2_conj_square = r2.component_mul(&r2.conjugate());
    let r2 = r2_conj_square.column_sum();
    let r_middle = r_middle.component_mul(&r_middle.conjugate()).column_sum();
    /* 


    let h = x.iter().zip(x.iter().skip(1)).map(|(a, b)| b - a).collect::<Vec<_>>();
    let h: DVector<f64> = DVector::from_vec(h);

    
     */

    let res = ((0.5 * (32.0 / 45.0 * r_middle + 49.0 / 90.0 * (r1 + r2)))   ).iter().map(|x| x.sqrt()).collect::<Vec<_>>();
    let res: DVector<f64> = DVector::from_vec(res);
    res
}//fn estimate_rms_residuals
/* 
#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
    use super::*;

    #[test]
    fn test_estimate_fun_jac() {
        let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> = Box::new(|scalar, vector| {
            // Multiply each element of the vector by the scalar
            scalar * vector
        });
        let x = DVector::from_element(3, 2.0);
        let y = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
        ]).transpose();
        let expected_J = vec![vec![2.0, 2.0, 2.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 2.0, 2.0, 2.0]];
        let J: Vec<DMatrix<f64>> = estimate_fun_jac(fun, &x, &y, None);
        print!("\n J {:?}", J);
        for (i, matrix_i) in J.iter().enumerate() {
            let J_i:Vec<f64> = matrix_i.iter().cloned().collect();
            println!("J_i {:?}", J_i);
            let expected_Ji:Vec<f64> = expected_J[i].clone(); 
            println!("expected_Ji {:?}", expected_Ji);
            let epsilon = 1e-6;
            let are_equal = J_i.iter().zip(expected_Ji.iter()).all(|(a, b)| (a - b).abs() < epsilon);
            assert_eq!(are_equal, true, "Row {} is not equal.", i);  
        }
             
    }
/* 
    #[test]
    fn test_estimate_bc_jac() {
        let bc: Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>> = Box::new(|vec0, vector| {
            // Multiply each element of the vector by the scalar
            vec0 * vector
        });
        let ya = DVector::from_element(3, 1.0);
        let yb = DVector::from_element(3, 2.0);

        let expected_dbc_dya = DMatrix::from_element(3, 3, 1.0);
        let expected_dbc_dyb = DMatrix::from_element(3, 3, 1.0);

        let (dbc_dya, dbc_dyb) = estimate_bc_jac(bc, &ya, &yb, None);

    
        for (i, row_i) in dbc_dya.row_iter().enumerate() {
            let df_dy_row_i:DVector<f64> = row_i.transpose().into_owned();
            let df_dy_vec_i:Vec<f64> = df_dy_row_i.data.as_vec().clone();
            let exp_df_dy = expected_dbc_dya.row(i).transpose().into_owned();
            let expected_df_dy:Vec<f64> = exp_df_dy.data.as_vec().clone(); 
            let epsilon = 1e-6;
            let are_equal = df_dy_vec_i.iter().zip(expected_df_dy.iter()).all(|(a, b)| (a - b).abs() < epsilon);
            assert_eq!(are_equal, true, "Row {} is not equal.", i);

    }
}
        */
    #[test]
    fn test_estimate_rms_residuals() {
        let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> = Box::new(|scalar, vector| {
            // Multiply each element of the vector by the scalar
            scalar * vector
        });
        let sol: fn(&DVector<f64>, bool) -> DMatrix<f64> = |x: &DVector<f64>, flag: bool| {
            DMatrix::from_iterator(x.len(), 1, x.iter().map(|x| x * x))
        };
        let x = DVector::from_iterator(3, 0..3).map(|x| x as f64);
        let h = DVector::from_element(2, 1.0);

        let r_middle = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,]);
        let f_middle =  DMatrix::from_row_slice(2, 3, &[
            1.01, 2.01, 3.01,
            1.01, 2.01, 3.01,]);

        let expected_rms_residuals:DVector<f64> = DVector::from_element(1, 0.0);

        let rms_residuals: DVector<f64> = estimate_rms_residuals(fun, sol, &x, &h, &r_middle, &f_middle);

        
        for (i, x_i) in rms_residuals.row_iter().enumerate() {
            let x_i = x_i[0];
            let epsilon = 1e-6;
            relative_eq!(x_i, expected_rms_residuals[i], epsilon = epsilon);
    }
}
    
}



*/