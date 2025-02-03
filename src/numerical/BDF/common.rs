extern crate nalgebra as na;

use na::{DMatrix, DVector};

const EPS: f64 = f64::EPSILON;
use nalgebra_sparse::CsrMatrix;
use std::cmp::{PartialEq, PartialOrd};

extern crate num;
extern crate num_complex;
use log::{error, info};
use num::traits::Float;
use std::error::Error;
use std::fmt::Debug;

pub fn newton_tol(rtol: NumberOrVec) -> f64 {
    let newton_tol: f64 = match rtol {
        // rtol is a number
        NumberOrVec::Number(rtol_i) => f64::max(
            10.0 * f64::EPSILON / rtol_i,
            f64::min(0.03, rtol_i.powf(0.5)),
        ),
        NumberOrVec::Vec(rtol_i) => {
            let rtol_i = *rtol_i
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            f64::max(
                10.0 * f64::EPSILON / rtol_i,
                f64::min(0.03, rtol_i.powf(0.5)),
            )
        } // let newton_tol = f64::max(10.0 * std::f64::EPSILON / rtol, f64::min(0.03, rtol.powf(0.5)));
    };
    newton_tol
}
#[allow(unused)]
fn is_sparseCs(matrix: &CsrMatrix<f64>, threshold: f64) -> bool {
    let total_elements = matrix.nrows() * matrix.ncols();
    let non_zero_elements = matrix.nnz();
    let sparsity = non_zero_elements as f64 / total_elements as f64;
    sparsity < threshold
}

pub fn is_sparse(matrix: &DMatrix<f64>, threshold: f64) -> bool {
    let (rows, cols) = matrix.shape();
    let total_elements = (rows * cols) as f64;
    let nonzero_elements = matrix.iter().filter(|&x| x.abs() > 0.0).count() as f64;
    // A matrix is considered sparse if the number of nonzero elements is less than a certain percentage of the total elements.
    // You can adjust the threshold percentage as needed.
    nonzero_elements / total_elements < threshold
}

pub fn check_arguments<F, T>(
    fun: F,
    y0: &[T],
    support_complex: bool,
) -> Result<(Box<dyn Fn(f64, &DVector<T>) -> DVector<T>>, DVector<T>), Box<dyn Error>>
where
    F: 'static + Fn(f64, &DVector<T>) -> DVector<T>,
    T: Float + num::traits::FromPrimitive + Debug + 'static,
{
    let y0 = DVector::from_column_slice(y0);

    let dtype_is_complex = y0.iter().any(|&x| x.is_nan() || x.is_infinite());

    if dtype_is_complex && !support_complex {
        return Err("`y0` is complex, but the chosen solver does not support integration in a complex domain.".into());
    }
    fn check_if_is_dvector(obj: &dyn std::any::Any) -> bool {
        if let Some(_obj) = obj.downcast_ref::<DVector<f64>>() {
            true
        } else {
            false
        }
    }

    if y0.nrows() != 1 && check_if_is_dvector(&y0) == false {
        return Err("`y0` must be 1-dimensional.".into());
    }

    if y0.iter().any(|&x| !x.is_finite()) {
        return Err("All components of the initial state `y0` must be finite.".into());
    }

    let fun_wrapped = Box::new(move |t: f64, y: &DVector<T>| -> DVector<T> { fun(t, y) });

    Ok((fun_wrapped, y0))
}

pub fn validate_first_step(first_step: f64, t0: f64, t_bound: f64) -> Result<f64, &'static str> {
    if first_step <= 0.0 {
        return Err("`first_step` must be positive.");
    }
    if first_step > (t_bound - t0).abs() {
        return Err("`first_step` exceeds bounds.");
    }
    info!("first step validation: done");
    Ok(first_step)
}

pub fn validate_max_step(max_step: f64) -> Result<f64, &'static str> {
    if max_step <= 0.0 {
        return Err("`max_step` must be positive.");
    }
    Ok(max_step)
}

pub fn norm(vector: &DVector<f64>) -> f64 {
    vector.norm() / (vector.len() as f64).sqrt()
}

// rtol or atol can be a number or a vector
pub enum NumberOrVec {
    Number(f64),
    Vec(Vec<f64>),
}
impl PartialEq for NumberOrVec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NumberOrVec::Number(a), NumberOrVec::Number(b)) => a == b,
            (NumberOrVec::Vec(a), NumberOrVec::Vec(b)) => a == b,
            _ => false,
        }
    }
}
// we must compare NumberOrVec elements
impl PartialOrd for NumberOrVec {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (NumberOrVec::Number(a), NumberOrVec::Number(b)) => a.partial_cmp(b),
            (NumberOrVec::Vec(a), NumberOrVec::Vec(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}
impl Debug for NumberOrVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumberOrVec::Number(n) => write!(f, "{}", n),
            NumberOrVec::Vec(v) => write!(f, "{:?}", v),
        }
    }
}
impl Clone for NumberOrVec {
    fn clone(&self) -> Self {
        match self {
            NumberOrVec::Number(n) => NumberOrVec::Number(*n),
            NumberOrVec::Vec(v) => NumberOrVec::Vec(v.clone()),
        }
    }
}
/// This function calculates the initial step size for an ODE solver.
///
/// # Parameters
///
/// - `fun`: A function representing the ODE system. It takes a time `t` and a state vector `y` as input and returns the derivative of `y`.
/// - `t0`: The initial time.
/// - `y0`: The initial state vector.
/// - `t_bound`: The boundary time.
/// - `max_step`: The maximum allowed step size.
/// - `f0`: The derivative of the initial state vector.
/// - `direction`: The direction of integration (1.0 for forward, -1.0 for backward).
/// - `order`: The order of the ODE solver.
/// - `rtol`: The relative tolerance for the solution. It can be a number or a vector.
/// - `atol`: The absolute tolerance for the solution. It can be a number or a vector.
///
/// # Return
///
/// The function returns the initial step size for the ODE solver.
///

pub fn scale_func(rtol: NumberOrVec, atol: NumberOrVec, y0: &DVector<f64>) -> Vec<f64> {
    let scale: Vec<f64> = match atol {
        // atol is a number
        NumberOrVec::Number(atol) => {
            match rtol {
                // rtol is a number
                NumberOrVec::Number(rtol) => y0
                    .into_iter()
                    .map(|&y_i| atol + (y_i.abs() * rtol))
                    .collect(),

                // rtol is a vector
                NumberOrVec::Vec(rtol) => y0
                    .into_iter()
                    .zip(&rtol)
                    .map(|(y_i, rtol_i)| atol + (y_i.abs() * rtol_i))
                    .collect(),
            }
        }
        // atol is a vector
        NumberOrVec::Vec(atol) => {
            match rtol {
                // rtol is a number
                NumberOrVec::Number(rtol) => y0
                    .into_iter()
                    .zip(&atol)
                    .map(|(y_i, atol_i)| atol_i + (y_i.abs() * rtol))
                    .collect(),
                // rtol is a vector
                NumberOrVec::Vec(rtol) => y0
                    .into_iter()
                    .zip(&atol)
                    .zip(&rtol)
                    .map(|((y_i, atol_i), rtol_i)| atol_i + (y_i.abs() * rtol_i))
                    .collect(),
            }
        }
    };
    //   print!("scale = {:?} \n", &scale);
    scale
}
pub fn select_initial_step(
    fun: &Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    t0: f64,
    y0: &DVector<f64>,
    t_bound: f64,
    max_step: f64,
    f0: &DVector<f64>,
    direction: f64,
    order: f64,
    rtol: NumberOrVec,
    atol: NumberOrVec,
) -> f64 {
    if y0.len() == 0 {
        return f64::INFINITY;
    }

    let interval_length = (t_bound - t0).abs();
    if interval_length == 0.0 {
        return 0.0;
    }
    // atol_i + (y_i.abs() * rtol element-wise

    let scale = scale_func(rtol, atol, y0);
    let scale: DVector<f64> = DVector::from_vec(scale);
    let d0 = norm(&(y0.component_div(&scale)));
    let d1 = norm(&(f0.component_div(&scale)));
    // print!("d0 = {}, d1 = {}", d0, d1);
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };

    let h0 = h0.min(interval_length);
    let y1 = y0 + h0 * direction * f0;
    let f1 = fun(t0 + h0 * direction, &y1);
    //   info!("f {}, arg{}", f1.clone(), &y1);
    let d2 = norm(&((f1 - f0).component_div(&scale))) / h0;
    //  info!("SCALE, {}, d2, {}, h0 {}", scale, d2.clone(), h0  );
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        1e-6.max(h0 * 1e-3)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (order + 1.0))
    };
    //   info!("scale, {}, d2, {}, h1 {}", scale, d2.clone(), h1);
    //  info!("select initial step: done" );
    // info!("h0, {}, h1, {}, interval_length, {}, max_step, {}", h0, h1, interval_length, max_step);
    //let res = h0.min(h1).min(interval_length).min(max_step);
    let res = vec![100.0 * h0, h1, interval_length, max_step]
        .into_iter()
        .fold(1.0, |acc, x| acc.min(x));

    res
}

pub fn validate_tol(
    rtol: NumberOrVec,
    atol: NumberOrVec,
    _n: usize,
) -> Result<(NumberOrVec, NumberOrVec), &'static str> {
    let rtol = match rtol {
        // rtol is a number
        NumberOrVec::Number(rtol_i) => {
            if rtol_i < 100.0 * f64::EPSILON {
                error!("At least one element of `rtol` is too small. Setting `rtol = std::f64::max(rtol, 100.0 * std::f64::EPSILON)`.");
                NumberOrVec::Number(f64::max(rtol_i, 100.0 * f64::EPSILON))
            } else {
                NumberOrVec::Number(rtol_i)
            }
        }
        NumberOrVec::Vec(rtol) => {
            let mut fixed_rtol = rtol.clone();
            for i in 0..rtol.len() {
                if rtol[i] < 100.0 * f64::EPSILON {
                    eprintln!("At least one element of `rtol` is too small. Setting `rtol = std::f64::max(rtol, 100.0 * std::f64::EPSILON)`.");
                    fixed_rtol[i] = f64::max(rtol[i], 100.0 * f64::EPSILON);
                }
            }
            NumberOrVec::Vec(fixed_rtol)
        }
    };
    let atol = match atol {
        // atol is a number
        NumberOrVec::Number(atol) => {
            if atol < 0.0 {
                return Err("At least one element of `atol` is negative.");
            } else {
                NumberOrVec::Number(atol)
            }
        }

        // atol is a vector
        NumberOrVec::Vec(atol) => {
            if atol.iter().any(|&x| x < 0.0) {
                return Err("At least one element of `atol` is negative.");
            } else {
                NumberOrVec::Vec(atol)
            }
        }
    };

    Ok((rtol, atol))
}

/*
    This function computes finite difference approximation to the Jacobian
    matrix of `fun` with respect to `y` using forward differences.
    The Jacobian matrix has shape (n, n) and its element (i, j) is equal to
    ``d f_i / d y_j``.

    A special feature of this function is the ability to correct the step
    size from iteration to iteration. The main idea is to keep the finite
    difference significantly separated from its round-off error which
    approximately equals ``EPS * np.abs(f)``. It reduces a possibility of a
    huge error and assures that the estimated derivative are reasonably close
    to the true values (i.e., the finite difference approximation is at least
    qualitatively reflects the structure of the true Jacobian).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system implemented in a vectorized fashion.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Value of the right hand side at (t, y).
    threshold : float
        Threshold for `y` value used for computing the step size as
        ``factor * np.maximum(np.abs(y), threshold)``. Typically, the value of
        absolute tolerance (atol) for a solver should be passed as `threshold`.
    factor : ndarray with shape (n,) or None
        Factor to use for computing the step size. Pass None for the very
        evaluation, then use the value returned from this function.
    sparsity : tuple (structure, groups) or None
        Sparsity structure of the Jacobian, `structure` must be csc_matrix.

    Returns
    -------
    J : ndarray or csc_matrix, shape (n, n)
        Jacobian matrix.
    factor : ndarray, shape (n,)
        Suggested `factor` for the next evaluation.
*/

pub fn num_jac(
    fun: &Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    t: f64,
    y: &DVector<f64>,
    f: &DVector<f64>,  // Value of the right hand side at (t, y).
    atol: NumberOrVec, // threshold
    factor: Option<DVector<f64>>,
    sparsity: Option<(DMatrix<f64>, Vec<usize>)>,
) -> (DMatrix<f64>, DVector<f64>) {
    let y = y.clone();
    let n = y.len();
    if n == 0 {
        return (DMatrix::zeros(0, 0), DVector::zeros(0));
    }

    let mut factor = if let Some(factor) = factor {
        factor.clone()
    } else {
        DVector::from_element(n, EPS.powf(0.5))
    };
    // Direct the step as ODE dictates, hoping that such a step won't lead to
    // a problematic region. For complex ODEs it makes sense to use the real
    // part of f as we use steps along real axis.
    // here we do something different from python code because in py code atol is alaways a number
    let atol_value = match atol {
        NumberOrVec::Number(x) => x,
        NumberOrVec::Vec(x) => *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap(),
    };

    let mut h = DVector::zeros(n);
    let y_scale = DVector::zeros(n);
    for i in 0..n {
        let f_sign = if f[i] >= 0.0 { 1.0 } else { -1.0 };
        let y_scale_i = f_sign * f64::max(y[i].abs(), atol_value);
        y_scale.push(y_scale_i);
        h[i] = y_scale_i * factor[i];
    }
    let h_ = DVector::zeros(n);
    for i in 0..h.len() {
        if h[i] == 0.0 {
            while h[i] == 0.0 {
                factor[i] *= 10.0;
                h[i] = y_scale[i] * factor[i];
                h_.push(h[i]);
            }
        }
    }

    // let fun = *fun;
    let (J, factor) = if let Some((_structure, _groups)) = sparsity {
        _dense_num_jac(fun, t, &y, f, &h_, &mut factor, y_scale)
        //TODO!!!
        // _sparse_num_jac(fun, t, &y, f, &h_, &factor, y_scale, _structure, groups)
    } else {
        _dense_num_jac(fun, t, &y, f, &h_, &mut factor, y_scale)
    };

    (J, factor)
}
#[allow(dead_code)]
const NUM_JAC_DIFF_REJECT: f64 = 1e-6;
const NUM_JAC_FACTOR_INCREASE: f64 = 10.0;
const NUM_JAC_FACTOR_DECREASE: f64 = 0.1;
const NUM_JAC_MIN_FACTOR: f64 = 1e-7;
const NUM_JAC_DIFF_SMALL: f64 = 1e-6;
const NUM_JAC_DIFF_BIG: f64 = 1e-2;

fn _dense_num_jac(
    fun: &Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    t: f64,
    y: &DVector<f64>,
    f: &DVector<f64>,
    h: &DVector<f64>,
    factor: &mut DVector<f64>,
    y_scale: DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>) {
    fn ret(v: &Vec<f64>, indices: &Vec<usize>) -> Vec<f64> {
        indices
            .iter()
            .filter_map(|&index| v.get(index))
            .cloned()
            .collect()
    }

    let n = y.len();
    let _h_vecs = DMatrix::from_diagonal(h);
    /*
    // create a new matrix
    let mut y_new:Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..n {
            row.push(y[j] + h[i]);
        }
        y_new.push(row);
    }
    */

    let y_new: DMatrix<f64> = DMatrix::from_fn(n, n, |i, j| y[j] + h[i]);

    fn create_f_diff(
        y_new: DMatrix<f64>,
        t: f64,
        fun: &Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        f: &DVector<f64>,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = y_new.nrows();
        let mut fun_vector = Vec::new();
        let mut diff_vec = Vec::new();
        for row in y_new.row_iter() {
            let fun_new_i = fun(t, &(row.transpose()));
            let fun_new_vec_i: Vec<f64> = fun_new_i.data.as_vec().to_owned(); // TODO
            let diff_vec_i: Vec<f64> = fun_new_vec_i
                .clone()
                .iter()
                .zip(f.data.as_vec().to_owned())
                .map(|(fi, f)| fi - f)
                .collect();
            fun_vector.extend(fun_new_vec_i);
            diff_vec.extend(diff_vec_i);
        }
        let f_new = DMatrix::from_vec(n, n, fun_vector);
        let diff = DMatrix::from_vec(n, n, diff_vec);

        (f_new, diff)
    }
    let (f_new, mut diff) = create_f_diff(y_new, t, fun, f);

    //find the index of the maximum absolute difference for each column in the DMatrix
    let mut max_indices: Vec<usize> = Vec::new();
    for column in diff.column_iter() {
        // iterate over each column
        let max_index = column
            .iter()
            .enumerate()
            .filter_map(|(index, &val)| {
                if val == column.amax() {
                    Some(index)
                } else {
                    None
                }
            }) //finds the index of the maximum absolute difference using the enumerate()
            .next() //method to get the index along with the value, and the filter_map() method to filter out the maximum value.
            .map_or(0, |index| index);

        max_indices.push(max_index);
    }

    /*
    the same code as above
    for column in diff.column_iter() {
        let max_value = column.amax();
        for (i, &val) in column.iter().enumerate() {
            if val==max_value {
                max_indices.push(i);
            }
        }
    }*/
    let r: Vec<usize> = (0..n).collect();
    let r = DVector::from_vec(r);
    // extracts the maximum absolute differences corresponding to the indices in max_ind from the diff matrix,
    // and stores them in the vector max_diff.
    let max_diff_vec: Vec<_> = r
        .iter()
        .zip(&max_indices)
        .map(|(r_i, max_ind)| diff[(*max_ind, *r_i)])
        .collect();
    let max_diff = DVector::from_vec(max_diff_vec.clone());
    //  let max_diff:Vec<f64> = max_diff.into();
    // The f_new_max vector is created using the zip method to combine the r and max_indices vectors. For each pair of indices, the corresponding
    // value from the f_new vector is selected using the indices. The collect method is used to gather the selected values into a new vector.
    let f_new_max: Vec<_> = r
        .iter()
        .zip(&max_indices)
        .map(|(r_i, max_ind)| f_new[(*max_ind, *r_i)])
        .collect();

    let f_max: Vec<_> = max_indices.iter().map(|ma_index| f[*ma_index]).collect();
    //  iterate over the pairs of corresponding elements in f_new_max and f_max. For each pair, it calculates the maximum absolute value between the
    // corresponding elements in f_new_max and f_max, a
    let scale_vec: Vec<f64> = f_new_max
        .iter()
        .zip(&f_max)
        .map(|(fi, fi_max)| fi_max.abs().max(fi.abs()))
        .collect();
    let scale = DMatrix::from_vec(n, 1, scale_vec.clone());
    // element-wise compare
    let diff_too_small: Vec<bool> = max_diff
        .iter()
        .zip(scale_vec.clone())
        .map(|(max_diff_i, scale_i)| *max_diff_i < NUM_JAC_DIFF_SMALL * (scale_i))
        .collect();

    if diff_too_small.iter().any(|&x| x) {
        let ind: Vec<usize> = diff_too_small
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if x == true { Some(i) } else { None })
            .collect();
        let new_factor = factor.select_rows(&ind) * NUM_JAC_FACTOR_INCREASE;
        let h_new = y.select_rows(&ind) + new_factor.component_mul(&y_scale.select_rows(&ind))
            - y.select_rows(&ind);

        let h_vecs = DMatrix::from_diagonal(&h_new);
        let h_vecs = h_vecs.select_columns(&ind);
        let y_new: DMatrix<f64> = DMatrix::from_fn(n, n, |i, j| y[j] + h_vecs[i]);

        let (_f_new, diff_new) = create_f_diff(y_new, t, fun, f);

        let mut max_indices_new: Vec<usize> = Vec::new();
        for column in diff_new.column_iter() {
            // iterate over each column
            let max_index = column
                .iter()
                .enumerate()
                .filter_map(|(index, &val)| {
                    if val == column.amax() {
                        Some(index)
                    } else {
                        None
                    }
                }) //finds the index of the maximum absolute difference using the enumerate()
                .next() //method to get the index along with the value, and the filter_map() method to filter out the maximum value.
                .map_or(0, |index| index);

            max_indices_new.push(max_index);
        }
        let r: Vec<usize> = (0..ind.len()).collect();
        let r = DVector::from_vec(r);

        let max_diff_new: Vec<_> = r
            .iter()
            .zip(&max_indices)
            .map(|(r_i, max_ind)| diff_new[(*max_ind, *r_i)])
            .collect();
        let max_diff_new = DVector::from_vec(max_diff_new);
        let scale_new_vec: Vec<f64> = f_new_max
            .iter()
            .zip(&f_max)
            .map(|(fi, fi_max)| fi_max.abs().max(fi.abs()))
            .collect();
        let scale_new = DMatrix::from_vec(n, 1, scale_new_vec.clone());
        let max_diff_scale: Vec<f64> = ret(&max_diff_vec, &ind)
            .iter()
            .zip(&scale_new_vec)
            .map(|(max_diff_i, scale_i)| *max_diff_i * (*scale_i))
            .collect();

        let max_diff_new_scale: Vec<f64> = max_diff_new
            .iter()
            .zip(&ret(&scale_vec, &ind))
            .map(|(max_diff_i, scale_i)| *max_diff_i * (*scale_i))
            .collect();

        let update: Vec<bool> = max_diff_scale
            .iter()
            .zip(&max_diff_new_scale)
            .map(|(max_diff_i, max_diff_new_scale_i)| *max_diff_i < *max_diff_new_scale_i)
            .collect();
        if update.iter().any(|&x| x) {
            let update_ind: Vec<usize> = update
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| if x { Some(i) } else { None })
                .collect();

            factor
                .select_rows(&update_ind)
                .copy_from(&new_factor.select_rows(&update_ind));
            h.select_rows(&update_ind)
                .copy_from(&h_new.select_rows(&update_ind));
            diff.select_columns(&update_ind)
                .copy_from(&diff_new.select_columns(&update_ind));
            scale
                .select_rows(&update_ind)
                .copy_from(&scale_new.select_rows(&update_ind));
            max_diff
                .select_rows(&update_ind)
                .copy_from(&max_diff_new.select_rows(&update_ind));
        }
    }
    //diff/=h
    diff.component_div_assign(h);
    factor.iter_mut().enumerate().for_each(|(i, f)| {
        if *f < NUM_JAC_DIFF_SMALL * scale[i] {
            *f *= NUM_JAC_FACTOR_INCREASE;
        } else if *f > NUM_JAC_DIFF_BIG * scale[i] {
            *f *= NUM_JAC_FACTOR_DECREASE;
        }
    });

    factor
        .iter_mut()
        .for_each(|f| *f = f.max(NUM_JAC_MIN_FACTOR));

    (diff, factor.clone())
}

/*
fn _sparse_num_jac(
    fun:&Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    t: f64,
    y: &DVector<f64>,
    f: &DVector<f64>,
    h: &DVector<f64>,
    factor: &DVector<f64>,
    y_scale: DVector<f64>,
    structure: &DMatrix<f64>,
    groups: Vec<usize>,
) -> (DMatrix<f64>, DVector<f64>) {
    let n = y.len();
    let mut J = DMatrix::from_sparse(structure.clone(), vec![0; n * n], vec![0.0; n * n]);
    let mut factor = factor.clone();

    for group in groups {
        let y_plus = y + h * y_scale;
        let y_minus = y - h * y_scale;
        let f_plus = fun(t, &y_plus);
        let f_minus = fun(t, &y_minus);
        let df_dy = (f_plus - f_minus) / (2.0 * h[group] * y_scale);

        for i in 0..n {
            J[(i, group)] = df_dy[i];
        }

        // Adjust the step size for the next evaluation.
        let error = (df_dy - (f_plus - f) / h[group]).norm();
        let tolerance = EPS * f.norm();
        if error > tolerance {
            factor[group] *= (tolerance / error).powf(0.5);
        } else {
            factor[group] *= 10.0;
        }
    }

    (J, factor)
}// fn sparcy_num_jac
*/

/*
use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, CsVec};
use std::f64::EPSILON;

const NUM_JAC_DIFF_REJECT: f64 = 1e-6;
const NUM_JAC_FACTOR_INCREASE: f64 = 10.0;
const NUM_JAC_FACTOR_DECREASE: f64 = 0.1;
const NUM_JAC_MIN_FACTOR: f64 = 1e-7;
const NUM_JAC_DIFF_SMALL: f64 = 1e-6;
const NUM_JAC_DIFF_BIG: f64 = 1e-2;

fn num_jac<F>(
    fun: F,
    t: f64,
    y: &DVector<f64>,
    f: &DVector<f64>,
    threshold: f64,
    factor: Option<DVector<f64>>,
    sparsity: Option<(CsMat<f64>, DVector<usize>)>,
) -> (DMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    let n = y.len();
    if n == 0 {
        return (DMatrix::zeros(0, 0), factor.unwrap_or_else(|| DVector::zeros(0)));
    }

    let mut factor = factor.unwrap_or_else(|| DVector::from_element(n, EPSILON.sqrt()));
    let f_sign = f.map(|fi| if fi >= 0.0 { 1.0 } else { -1.0 });
    let y_scale = f_sign.component_mul(&y.map(|yi| yi.abs().max(threshold)));
    let mut h = y + factor.component_mul(&y_scale) - y;

    for i in 0..n {
        while h[i] == 0.0 {
            factor[i] *= 10.0;
            h[i] = (y[i] + factor[i] * y_scale[i]) - y[i];
        }
    }

    if let Some((structure, groups)) = sparsity {
        return sparse_num_jac(&fun, t, y, f, &h, &mut factor, &y_scale, &structure, &groups);
    } else {
        return dense_num_jac(&fun, t, y, f, &h, &mut factor, &y_scale);
    }
}



    fn _dense_num_jac(
    fun: &Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    t: f64,
    y: &DVector<f64>,
    f: &DVector<f64>,
    h: &DVector<f64>,
    factor: &DVector<f64>,
    y_scale: DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>) {
    let n = y.len();
    let mut J = DMatrix::zeros(n, n);
    let mut factor = factor.clone();

    for j in 0..n {
        let y_plus = y + h * y_scale;
        let y_minus = y - h * y_scale;
        let f_plus = fun(t, &y_plus);
        let f_minus = fun(t, &y_minus);
        let df_dy = (f_plus.clone() - f_minus) / (2.0 * h[j] * y_scale);

        for i in 0..n {
            J[(i, j)] = df_dy[i];
        }

        // Adjust the step size for the next evaluation.
        let error = (df_dy - (f_plus - f) / h[j]).norm();
        let tolerance = EPS * f.norm();
        if error > tolerance {
            factor[j] *= (tolerance / error).powf(0.5);
        } else {
            factor[j] *= 10.0;
        }
    }

    (J, factor)
}


    fn dense_num_jac<F>(
    fun: &F,
    t: f64,
    y: &DVector<f64>,
    f: &DVector<f64>,
    h: &DVector<f64>,
    factor: &mut DVector<f64>,
    y_scale: &DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    let n = y.len();
    let h_vecs = DMatrix::from_diagonal(h);
    let f_new = fun(t, &(y + h_vecs));
    let mut diff = f_new - f;
    let max_ind = diff.column_iter().map(|col| col.amax()).collect::<DVector<f64>>();
    let r = DVector::from_iterator(n, 0..n);
    let max_diff = diff.column_iter().map(|col| col.amax()).collect::<DVector<f64>>();
    let scale = f.map(|fi| fi.abs()).max(&f_new.map(|fi| fi.abs()));
    let diff_too_small = max_diff.map(|d| d < NUM_JAC_DIFF_REJECT * scale);

    if diff_too_small.iter().any(|&x| x) {
        let ind: Vec<usize> = diff_too_small
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if x { Some(i) } else { None })
            .collect();
        let new_factor = factor.select_rows(&ind) * NUM_JAC_FACTOR_INCREASE;
        let h_new = y.select_rows(&ind) + new_factor.component_mul(&y_scale.select_rows(&ind)) - y.select_rows(&ind);
        let mut h_vecs = DMatrix::from_diagonal(&h_new);
        let f_new = fun(t, &(y + h_vecs));
        let diff_new = f_new - f;
        let max_ind_new = diff_new.column_iter().map(|col| col.amax()).collect::<DVector<f64>>();
        let max_diff_new = diff_new.column_iter().map(|col| col.amax()).collect::<DVector<f64>>();
        let scale_new = f.map(|fi| fi.abs()).max(&f_new.map(|fi| fi.abs()));
        let update = max_diff.select_rows(&ind).component_mul(&scale_new) < max_diff_new.component_mul(&scale.select_rows(&ind));

        if update.iter().any(|&x| x) {
            let update_ind: Vec<usize> = update
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| if x { Some(i) } else { None })
                .collect();
            factor.select_rows_mut(&update_ind).copy_from(&new_factor.select_rows(&update_ind));
            h.select_rows_mut(&update_ind).copy_from(&h_new.select_rows(&update_ind));
            diff.select_columns_mut(&update_ind).copy_from(&diff_new.select_columns(&update_ind));
            scale.select_rows_mut(&update_ind).copy_from(&scale_new.select_rows(&update_ind));
            max_diff.select_rows_mut(&update_ind).copy_from(&max_diff_new.select_rows(&update_ind));
        }
    }

    diff /= h;
    factor.iter_mut().for_each(|f| {
        if *f < NUM_JAC_DIFF_SMALL * scale {
            *f *= NUM_JAC_FACTOR_INCREASE;
        } else if *f > NUM_JAC_DIFF_BIG * scale {
            *f *= NUM_JAC_FACTOR_DECREASE;
        }
    });
    factor.iter_mut().for_each(|f| *f = f.max(NUM_JAC_MIN_FACTOR));

    (diff, factor.clone())
}
*/
/*
*/
