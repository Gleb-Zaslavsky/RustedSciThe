extern crate nalgebra as na;
extern crate num_traits;
extern crate sprs;

use na::{DMatrix, DVector, Dyn, LU};

use std::f64;
use std::ops::AddAssign;

use crate::numerical::BDF::BDF_utils::{group_columns, OrderEnum};

use crate::numerical::BDF::common::{
    check_arguments, is_sparse, newton_tol, norm, scale_func, select_initial_step,
    validate_first_step, validate_max_step, validate_tol, NumberOrVec,
};
use std::fmt::Debug;
use std::fmt::Display;
const MAX_ORDER: usize = 5;
const NEWTON_MAXITER: usize = 4;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const SPARSE: f64 = 0.01;
/// Computes a specific matrix `R` based on the given order and factor.
/// # Parameters
/// * `order`: An unsigned integer representing the order of the matrix.
/// * `factor`: A floating-point number representing a factor used in computing the matrix.
/// # Returns
///
/// A 2D matrix of type `DMatrix<f64>` representing the computed matrix `R`.

fn cumulative_product_along_columns(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let (rows, cols) = matrix.shape();
    let mut result = DMatrix::zeros(rows, cols);

    for col in 0..cols {
        let mut cumprod = 1.0;
        for row in 0..rows {
            cumprod *= matrix[(row, col)];
            result[(row, col)] = cumprod;
        }
    }

    result
}

fn compute_r(order: usize, factor: f64) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(order + 1, order + 1);
    for i in 1..(order + 1) {
        for j in 1..(order + 1) {
            m[(i, j)] = (i as f64 - 1.0 - factor * j as f64) / i as f64;
        }
    }
    m.row_mut(0).fill(1.0);
    // println!("\n M= {}", m);
    let result = cumulative_product_along_columns(&m);
    result
}

/// Updates the matrix `d` in-place by multiplying it with the product of matrices `r` and `u`.
/// # Parameters
/// * `d`: A mutable reference to a mutable 2D matrix of type `DMatrix<f64>`. It will be updated in-place.
/// * `order`: An unsigned integer representing the order of the matrices `r` and `u`.
/// * `factor`: A floating-point number representing a factor used in computing the matrices `r` and `u`.
/// # Returns
/// This function does not return a value, but it updates the matrix `d` in-place.
fn change_D(D: &mut DMatrix<f64>, order: usize, factor: f64) {
    let r = compute_r(order, factor);
    let u = compute_r(order, 1.0);
    //  println!("r={:?}, u={:?}", r, u);
    let ru = r * u;
    let temp = ru.transpose() * D.rows(0, order + 1);
    D.rows_mut(0, order + 1).copy_from(&temp);
}

/// Solves a system of ordinary differential equations using the Backward Differentiation Formula (BDF) method.
///
/// # Parameters
/// * `fun`: A closure representing the system of ordinary differential equations. It takes a time `t` and a vector of
///   dependent variables `y` as input and returns a vector representing the derivatives of `y` with respect to `t`.
/// * `t_new`: The new time at which the system of ODEs needs to be solved.
/// * `y_predict`: An initial guess for the solution at the new time `t_new`.
/// * `c`: A constant used in the BDF method.
/// * `psi`: A vector used in the BDF method.
/// * `lu`: A sparse matrix representing the LU decomposition of the Jacobian matrix.
/// * `solve_lu`: A closure representing the function to solve the linear system using the LU decomposition.
/// * `scale`: A vector representing the scaling factors used in the BDF method.
/// * `tol`: The tolerance for the solution.
/// # Returns
/// A tuple containing:
/// * `converged`: A boolean indicating whether the solution converged or not.
/// * `iterations`: The number of iterations performed during the solution process.
/// * `y`: The solution vector at the new time `t_new`.
/// * `d`: An intermediate vector used in the BDF method.
fn solve_bdf_system<F>(
    fun: F,
    t_new: f64,
    y_predict: &DVector<f64>,
    c: f64,
    psi: &DVector<f64>,
    lumatrx: LU<f64, Dyn, Dyn>,
    solve_lu: &mut Box<dyn FnMut(&LU<f64, Dyn, Dyn>, &DVector<f64>) -> DVector<f64>>, //&dyn Fn(&CsMat<f64>, &DVector<f64>) -> DVector<f64>,
    scale: &DVector<f64>,                                                             // scale
    tol: f64,
) -> (bool, usize, DVector<f64>, DVector<f64>)
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    let mut d = DVector::zeros(y_predict.len()); //??????????
    let mut y = y_predict.clone();
    let mut dy_norm_old: Option<f64> = None;
    let mut converged = false;
    let mut k_: usize = 0;
    for k in 0..NEWTON_MAXITER {
        //   println!("Newton iteration: {}", k);
        let f = fun(t_new, &y);
        // checks if all elements in the vector f are finite. If any element is not finite, the loop is broken.
        if !f.iter().all(|&x| x.is_finite()) {
            break;
        }
        //The dy vector represents the change in the solution vector y at each iteration of the Newton-Raphson method
        // println!("DY c {:?},\n f = {:?}, \n {:?},\n {:?} \n", c, &f , psi, &d);
        let dy = solve_lu(&lumatrx, &(c * &f - psi - &d));
        // let new_fun:f64 = (|x:&f64, y:&f64| -x - (-y).exp() )(&y[0], &y[1]);
        //  print!("new_fun {:?} ", new_fun);
        //   println!("dy {:}, \n &(c * &f - psi - &d) {:?}", dy, &(c * &f - psi - &d));
        // The dy_norm value is then used to determine the convergence of the Newton-Raphson method and to adjust the step size in the BDF method
        let dy_norm = norm(&(dy.component_div(scale)));
        //  calculate the rate of convergence for the Newton-Raphson method used to solve the system of ODEs. The rate of convergence is calculated
        //by dividing the current norm of the change in the solution vector (dy_norm) by the norm of the change in the solution vector from the
        // previous iteration (dy_norm_old).
        let rate: Option<f64> = if let Some(dy_norm_old) = dy_norm_old {
            Some(dy_norm / dy_norm_old)
        } else {
            None
        };
        // println!("dy = {:?}, dy_norm = {:?}, rate = {:?}, {}", dy, dy_norm, rate, tol);
        // if the rate of convergence is greater than or equal to 1.0 or if the rate divided by (1.0 - rate) times the norm of the change in
        //the solution vector is greater than the specified tolerance (tol), the loop breaks, indicating that the Newton-Raphson method has
        //not converged.
        if let Some(rate) = rate {
            if rate >= 1.0
                || (rate.powi((NEWTON_MAXITER - k) as i32) / (1.0 - rate)) * dy_norm > tol
            {
                // powi is integer exponrnt
                break;
            }
        }

        y += &dy;
        d += &dy;
        k_ = k;
        if dy_norm == 0.0 {
            converged = true;
            break;
        }
        // if let Some(rate) = rate { println!("condition {}", rate / (1.0 - rate) * dy_norm)};
        if let Some(rate) = rate {
            if rate / (1.0 - rate) * dy_norm < tol {
                converged = true;
                break;
            }
        }
        dy_norm_old = Some(dy_norm);
    } //for dy_norm
      //  println!("{}, {}, {}, {}", converged, k_+1, y, d);
    (converged, k_ + 1, y, d)
}

//#[derive(Debug)]
pub struct BDF {
    fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    pub t: f64,
    pub y: DVector<f64>,
    t_bound: f64,
    max_step: f64,
    rtol: NumberOrVec,
    atol: NumberOrVec,
    vectorized: bool,
    n: usize,
    pub t_old: Option<f64>,
    h_abs: f64,
    h_abs_old: Option<f64>,
    error_norm_old: Option<f64>,
    newton_tol: f64,
    jac_factor: Option<DVector<f64>>,
    jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>>,
    J: DMatrix<f64>,
    I: DMatrix<f64>,
    lu: Box<dyn FnMut(&DMatrix<f64>) -> LU<f64, Dyn, Dyn>>,
    solve_lu: Box<dyn FnMut(&LU<f64, Dyn, Dyn>, &DVector<f64>) -> DVector<f64>>,
    gamma: DVector<f64>,
    alpha: DVector<f64>,
    error_const: DVector<f64>,
    D: DMatrix<f64>,
    order: usize,
    n_equal_steps: usize,
    LU: Option<LU<f64, Dyn, Dyn>>,
    nlu: usize,
    nfev: usize,
    njev: usize,
    pub direction: f64,
}
/*
impl Clone for BDF {
    fn clone(&self) -> Self {
        BDF {
            fun: self.fun.clone(),
            t: self.t,
            y: self.y.clone(),
            t_bound: self.t_bound,
            max_step: self.max_step,
            rtol: self.rtol.clone(),
            atol: self.atol.clone(),
            vectorized: self.vectorized,
            n: self.n,
            t_old: self.t_old,
            h_abs: self.h_abs,
            h_abs_old: self.h_abs_old,
            error_norm_old: self.error_norm_old,
            newton_tol: self.newton_tol,
            jac_factor: self.jac_factor.clone(),
            jac: self.jac.clone(),
            J: self.J.clone(),
            I: self.I.clone(),
            lu: self.lu.clone(),
            solve_lu: self.solve_lu.clone(),
            gamma: self.gamma.clone(),
            alpha: self.alpha.clone(),
            error_const: self.error_const.clone(),
            D: self.D.clone(),
            order: self.order,
            n_equal_steps: self.n_equal_steps,
            LU: self.LU.clone(),
            nlu: self.nlu,
            nfev: self.nfev,
            njev: self.njev,
            direction: self.direction,
        }
    }
}
    */
impl Display for BDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
impl Debug for BDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BDF")
            .field("t", &self.t)
            .field("y", &self.y)
            .field("t_bound", &self.t_bound)
            .field("max_step", &self.max_step)
            .field("rtol", &self.rtol)
            .field("atol", &self.atol)
            .field("vectorized", &self.vectorized)
            .field("n", &self.n)
            .field("t_old", &self.t_old)
            .field("h_abs", &self.h_abs)
            .field("h_abs_old", &self.h_abs_old)
            .field("error_norm_old", &self.error_norm_old)
            .field("newton_tol", &self.newton_tol)
            .field("jac_factor", &self.jac_factor)
            .field("J", &self.J)
            .field("I", &self.I)
            .field("gamma", &self.gamma)
            .field("alpha", &self.alpha)
            .field("error_const", &self.error_const)
            .field("D", &self.D)
            .field("order", &self.order)
            .field("LU", &self.LU)
            .field("nlu", &self.nlu)
            .field("nfev", &self.nfev)
            .field("njev", &self.njev)
            .field("direction", &self.direction)
            .finish()
    }
}

impl BDF {
    /*
    Initialize the BDF solver.

    Parameters:
    fun (callable): The right-hand side of the ordinary differential equation.
    t0 (float): The initial time.
    y0 (ndarray): The initial state.
    t_bound (float): The final time.
    max_step (float, optional): The maximum allowed step size. Default is np.inf.
    rtol (float, optional): The relative tolerance for the error norm. Default is 1e-3.
    atol (float, optional): The absolute tolerance for the error norm. Default is 1e-6.
    jac (callable, optional): The Jacobian of the right-hand side function. Default is None.
    jac_sparsity (ndarray, optional): The sparsity pattern of the Jacobian. Default is None.
    vectorized (bool, optional): Whether the function is vectorized. Default is False.
    first_step (float, optional): The initial step size. If None, it will be selected automatically. Default is None.
    **extraneous (dict, optional): Additional keyword arguments.

    Raises:
    ValueError: If the Jacobian is provided and its shape is not (self.n, self.n).
    */

    pub fn new(/* 
         fun: Box<dyn Fn(f64, DVector<f64>) -> DVector<f64>>, //?????
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        jac: Option<Box<dyn Fn(f64, DVector<f64>) -> DMatrix<f64>>>,
        jac_sparsity: Option<DMatrix<f64>>,
        vectorized: bool,
        first_step: Option<f64>,
        */) -> Self {
        BDF {
            fun: Box::new(|_t, y| y.clone()),
            t: 0.0,
            y: DVector::zeros(1),
            t_bound: 1.0,
            max_step: 1e-3,
            rtol: NumberOrVec::Number(1e-3),
            atol: NumberOrVec::Number(1e-4),
            vectorized: false,

            h_abs: 0.0,
            h_abs_old: None,
            error_norm_old: None,
            newton_tol: 0.0,
            jac_factor: None,
            jac: None,
            J: DMatrix::zeros(0, 0),
            I: DMatrix::zeros(0, 0),
            lu: Box::new(|J: &DMatrix<f64>| LU::new(J.clone())),

            solve_lu: Box::new(|_lu: &LU<f64, Dyn, Dyn>, dv: &DVector<f64>| {
                // implementation of the function
                dv.clone()
            }),
            gamma: DVector::zeros(0),
            alpha: DVector::zeros(0),
            error_const: DVector::zeros(0),
            D: DMatrix::zeros(0, 0),
            order: 1,
            n_equal_steps: 0,
            LU: None,
            nlu: 0,
            direction: 1.0,
            nfev: 0,
            njev: 0,
            n: 0,
            t_old: None,
        }
    } //fn new

    pub fn set_initial(
        &mut self,
        fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>, //?????
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        _max_step: f64,
        rtol: NumberOrVec,
        atol: NumberOrVec,
        jac: Option<Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>>,
        jac_sparsity: Option<DMatrix<f64>>,
        vectorized: bool,
        first_step: Option<f64>,
    ) {
        // prelude imitates super class in python package
        self.prelude(fun, t0, y0, t_bound, vectorized);
        println!("prelude done");
        // initialize parameters, if some parameters are not provided by the user then we will use special functions
        let max_step = validate_max_step(self.max_step);
        self.max_step = max_step.unwrap();

        let (rtol, atol) = validate_tol(rtol, atol, self.n).unwrap();
        self.rtol = rtol.clone();
        self.atol = atol.clone();
        println!("tolerance validation: done");
        let f = (&self.fun)(self.t, &self.y);
        let h_abs = match first_step {
            None => select_initial_step(
                &self.fun,
                self.t,
                &self.y,
                self.t_bound,
                self.max_step,
                &f,
                self.direction,
                1.0,
                self.rtol.clone(),
                self.atol.clone(),
            ),
            Some(first_step) => {
                validate_first_step(first_step, t0, t_bound).expect("first_step must be positive")
                //
            }
        };
        self.h_abs = h_abs;
        self.h_abs_old = None; // TODO: check this
        self.error_norm_old = None; //
                                    //             println!("h_abs {}", h_abs);
                                    //  let newton_tol = f64::max(10.0 * std::f64::EPSILON / rtol, f64::min(0.03, rtol.powf(0.5)));
        self.newton_tol = newton_tol(rtol);
        //  println!("newton_tol {}", self.newton_tol);
        self.jac_factor = None;

        // kappa: This array contains the coefficients for the BDF method. The values are specific to the BDF method and
        // are used to calculate the differences between the solution and the predicted solution.
        let kappa = DVector::from_vec(vec![0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0]);
        //  The gamma coefficients are related to the order of the BDF method.
        // The gamma vector is created using the DVector::from_iterator function, which takes an iterator as input. The iterator is created using
        // the std::iter::once and std::iter::map functions. The std::iter::once function is used to create an iterator that yields a single value,
        // which is 0.0 in this case. The std::iter::map function is used to create an iterator that maps each element of the range 1..=MAX_ORDER to
        // its reciprocal, which is 1.0 / i as f64. The scan function is used to create an iterator that accumulates the sum of the elements.
        // The resulting gamma vector has a length of MAX_ORDER + 1 and contains the cumulative sum of the reciprocals of the integers from 1
        //to MAX_ORDER, with an initial value of 0.0.

        let gamma = DVector::from_iterator(
            MAX_ORDER + 1,
            std::iter::once(0.0).chain((1..=MAX_ORDER).map(|i| 1.0 / i as f64).scan(
                0.0,
                |acc, x| {
                    *acc += x;
                    Some(*acc)
                },
            )),
        );
        // The alpha coefficients are used to calculate the differences between the predicted solution and the actual solution.
        assert_eq!(
            kappa.len(),
            gamma.len(),
            "kappa and gamma must have the same length"
        );
        let alpha = (DVector::from(vec![1.0; MAX_ORDER + 1]) - kappa.clone()).component_mul(&gamma); // component_mul is element-wise multiplication
        assert_eq!(
            alpha.len(),
            MAX_ORDER + 1,
            "alpha must have length MAX_ORDER + 1"
        );
        let error_const = kappa.component_mul(&gamma)
            + DVector::from_iterator(MAX_ORDER + 1, (1..=MAX_ORDER + 1).map(|i| 1.0 / i as f64));
        //         println!("kappa {:?}, alpha {:?}, gamma {:?},  error_const {:?}", &kappa, &alpha, &gamma, &error_const );
        self.alpha = alpha;
        self.error_const = error_const;
        self.gamma = gamma;

        let mut D = DMatrix::zeros(MAX_ORDER + 3, self.y.len());
        println!("created matrix of size: {:?} x {:?}", D.nrows(), D.ncols());
        //  let row_vector = DVector::from_row_slice([(self.y)  );
        // D.row_mut(0).copy_from(  &DVector::from_row_slice( row_vector));
        D.set_row(0, &self.y.transpose());
        //         println!("{:?}",  h_abs);
        D.set_row(1, &(f * h_abs * self.direction).transpose());
        //           println!("D = {:?},", &D);
        self.D = D;

        self.order = 1;
        self.n_equal_steps = 0;
        self.LU = None;
        self.create_funct();
        self.validate_jac(jac, jac_sparsity);

        // let (jac, J) = binding;
        // let (jac, J) = self.validate_jac(jac, jac_sparsity).as_mut().unwrap();

        /*
         */
    }

    fn create_funct(&mut self) {
        if is_sparse(&self.J, SPARSE) {
            // realization more efficient for sparse matrices
            let lu = |A: &DMatrix<f64>| {
                //   self.nlu += 1;
                A.clone().lu()
            };
            let solve_lu = |LU: &LU<f64, Dyn, Dyn>, b: &DVector<f64>| -> DVector<f64> {
                let linear_solution: DVector<f64> = LU.solve(b).unwrap();
                linear_solution
            };
            #[allow(unused)]
            fn solve_lu_f(LU: &LU<f64, Dyn, Dyn>, b: &DVector<f64>) -> DVector<f64> {
                let linear_solution = LU.solve(b).unwrap();
                linear_solution
            }

            self.lu = Box::new(lu) as Box<dyn FnMut(&DMatrix<f64>) -> LU<f64, Dyn, Dyn>>;
            self.solve_lu = Box::new(solve_lu)
                as Box<dyn FnMut(&LU<f64, Dyn, Dyn>, &DVector<f64>) -> DVector<f64>>;
            self.I = DMatrix::identity(self.n, self.n); // . An identity matrix is a square matrix with ones on the main diagonal
                                                        //(from the top left to the bottom right) and zeros elsewhere.
        } else {
            // realization more efficient for dense matrices

            // The type LU<f64, R, C> represents a LU decomposition of a matrix, where f64 is the element type,
            // R and C are the row and column indices, respectively - they are generic parameters.
            let lu = |A: &DMatrix<f64>| {
                //   self.nlu += 1;
                A.clone().lu()
            };
            let solve_lu = |LU: &LU<f64, Dyn, Dyn>, b: &DVector<f64>| -> DVector<f64> {
                let linear_solution: DVector<f64> = LU.solve(b).unwrap();
                linear_solution
            };
            #[allow(dead_code)]
            fn solve_lu_f(LU: &LU<f64, Dyn, Dyn>, b: &DVector<f64>) -> DVector<f64> {
                let linear_solution = LU.solve(b).unwrap();
                linear_solution
            }

            self.lu = Box::new(lu) as Box<dyn FnMut(&DMatrix<f64>) -> LU<f64, Dyn, Dyn>>;
            self.solve_lu = Box::new(solve_lu)
                as Box<dyn FnMut(&LU<f64, Dyn, Dyn>, &DVector<f64>) -> DVector<f64>>;
            self.I = DMatrix::identity(self.n, self.n); // . An identity matrix is a square matrix with ones on the main diagonal
                                                        //(from the top left to the bottom right) and zeros elsewhere.

            println!("functions creation: done");
        };
    }
    fn prelude(
        &mut self,
        fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        vectorized: bool,
    ) {
        let support_complex: bool = false;
        self.t_old = None;
        self.t = t0;
        self.n = y0.len();
        let (fun, y) = check_arguments(fun, (&y0).into(), support_complex).unwrap();

        self.y = y;

        self.t_bound = t_bound;
        self.vectorized = vectorized;
        /*
        if vectorized:
            def fun_single(t, y):
                return self._fun(t, y[:, None]).ravel()
            fun_vectorized = self._fun
        else:
            fun_single = self._fun

            def fun_vectorized(t, y):
                f = np.empty_like(y)
                for i, yi in enumerate(y.T):
                    f[:, i] = self._fun(t, yi)
                return f
            */
        //  fn fun(t, y):
        //    self.nfev += 1;
        //   return self.fun_single(t, y);

        self.fun = fun;
        //   self.fun_single = fun_single;
        //   self.fun_vectorized = fun_vectorized;

        self.direction = if t_bound != t0 {
            (t_bound - t0).signum()
        } else {
            1.0
        };

        self.nfev = 0;
    } // end prelude

    /*
    If jac is None, it estimates the Jacobian using numerical differentiation (num_jac) and returns a wrapped function (jac_wrapped)
    that computes the Jacobian at a given point.
    If jac is a callable function, it calls the function to compute the Jacobian, increments a counter (self.njev), and returns the
     computed Jacobian along with a wrapped function that computes the Jacobian at a given point.
    If jac is not a callable function, it assumes it's a pre-computed Jacobian matrix and checks its shape. If the shape is incorrect,
     it raises a ValueError.
    In all cases, it returns a tuple containing the wrapped Jacobian function (jac_wrapped) and the computed or pre-computed Jacobian matrix (J).
    The purpose of this code is to ensure that the Jacobian matrix is correctly computed or provided, and to provide a consistent
    interface for working with the Jacobian in the numerical method.

    */

    fn validate_jac(
        // jacobian can be None or a function, in case of None it will be computed using the num_jac function
        &mut self,
        jac: Option<Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>>,
        sparsity: Option<DMatrix<f64>>,
    ) {
        let t0 = self.t;
        let y0 = self.y.clone();
        let _sparsity_ = 0.0;
        // if jac is None, then we calculate the jacobian using the num_jac function and return a wrapped function that computes the jacobian
        // at a given point

        if let Some(jac) = jac {
            println!("analytical jacobian used");
            let J = jac(t0, &y0);
            self.njev += 1;

            let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
                if is_sparse(&J, SPARSE) {
                    Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
                        // self.njev += 1;
                        jac(t, &y)
                    })
                } else {
                    Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
                        //   self.njev += 1;
                        jac(t, &y)
                    })
                };
            /*
            if J.shape() != (self.n, self.n) {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "`jac` is expected to have shape ({}, {}), but actually has {:?}.",
                        self.n, self.n, J.shape()
                    ),
                )));
            }
             */
            //  ( Some(jac_wrapped), J)
            self.jac = Some(jac_wrapped);
            self.J = J.clone();
        } else {
            let _new_sparsity: Option<(DMatrix<f64>, Vec<usize>)> = if let Some(sparsity) = sparsity
            {
                let groups = group_columns(&sparsity, OrderEnum::None);
                Some((sparsity.clone(), groups))
            } else {
                None
            };
            /*
                  let mut jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>  = Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
                      //self.njev += 1;
                      let f = &(self.fun)(t, &y);
                      let (J, jac_factor) = num_jac(
                          &self.fun,
                          t,
                          &y,
                          &f,
                          self.atol.clone(),
                          self.jac_factor.clone(),
                        new_sparsity.clone() ,
                      );
                      self.jac_factor = Some(jac_factor);
                      J
                  });

                  let  J = jac_wrapped(t0, &y0);
            //    let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> = jac_wrapped;
            self.jac = Some(jac_wrapped);
            self.J = J.clone();
               */
        }; // jac is  None
           //  Ok((A, B))

        /*

        let J = if let Some(jac) = jac {
            jac(t0, y0)
        } else {
            DMatrix::zeros(self.n, self.n)
        };

        if J.shape() != (self.n, self.n) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "`jac` is expected to have shape ({}, {}), but actually has {:?}.",
                    self.n, self.n, J.shape()
                ),
            )));
        }

        (None, J)


           */

        // Ok((jac_wrapped, J))

        println!("jac validation: done");
    } // validate_jac

    //

    pub fn _step_impl(&mut self) -> (bool, Option<&'static str>) {
        //    println!("\n start step: t {}, y {:?}, order {:?}", self.t, self.y, self.order);
        let t = self.t;
        let mut D = self.D.clone();
        //       println!("D0 {:?}", D.transpose(),);
        let max_step = self.max_step;
        let min_step = 10.0 * f64::MIN; //(t.next_after(self.direction * INFINITY) - t).abs();

        let mut h_abs = if self.h_abs > max_step {
            //  println!("coind, {:?}, {:?}", self.order, max_step / self.h_abs);
            change_D(&mut D, self.order, max_step / self.h_abs);
            self.n_equal_steps = 0;

            max_step
        } else if self.h_abs < min_step {
            change_D(&mut D, self.order, min_step / self.h_abs);
            self.n_equal_steps = 0;
            min_step
        } else {
            self.h_abs
        };
        //  println!("h_abs = {:?} \n", h_abs);
        //   println!("D1 {:?}", D.transpose());

        let order = self.order;
        let alpha = &self.alpha;
        let gamma = &self.gamma;
        let error_const = &self.error_const;
        let J = self.J.clone();
        let mut LU = self.LU.clone();
        let mut current_jac = self.jac.is_none();
        let mut step_accepted = false;
        // scale preallocation
        let mut scale = DVector::zeros(self.n);
        //n_iter preallocation
        let mut n_iter = 0;
        // y_new preallocation
        let mut y_new = DVector::zeros(self.n);

        let _conv = false;
        // t_new preallocation
        let mut t_new = 0.0;
        // safety preallocation
        let mut safety = 0.0;
        // error_norm preallocation
        let mut error_norm = 0.0;
        // order preallocation
        let mut d = DVector::zeros(self.n);
        //      println!("{}", &J);
        while !step_accepted {
            if h_abs < min_step {
                return (false, "step size too small".into());
            }

            let h = h_abs * self.direction;
            let t_new_ = t + h;

            if self.direction * (t_new - self.t_bound) > 0.0 {
                //?

                t_new = self.t_bound;

                change_D(&mut D, order, (t_new - t).abs() / h_abs);
                self.n_equal_steps = 0;
                LU = None;
            }
            // println!("D cnanged {:?}", D);
            t_new = t_new_;

            let h = t_new - t;
            h_abs = h.abs();
            //     thread::sleep(Duration::from_millis(100));
            // println!("for pre {:?}, {:?}, ,", D.rows(0, order + 1).transpose(), order);
            let y_predict = D.rows(0, order + 1).row_sum().transpose();
            //    println!("y_predict = {:?} \n", y_predict);
            let y_predict_abs = y_predict.abs();
            let scale_ = scale_func(self.rtol.clone(), self.atol.clone(), &y_predict_abs); //  atol + rtol * y_predict.abs();
            let scale_: DVector<f64> = DVector::from_vec(scale_);
            scale = scale_;
            //      println!("for psi {:?}, {:?}, {:?}, {:?}", D.rows(1, order).transpose(), gamma.rows(1, order).transpose(), alpha[order], order);
            let psi = D.rows(1, order).transpose() * gamma.rows(1, order) / alpha[order]; //???
            let mut converged = false;
            let c = h / alpha[order];

            while !converged {
                if LU.clone().is_none() {
                    let eye: DMatrix<f64> = DMatrix::identity(self.n, self.n);
                    assert_eq!(eye.shape(), J.shape(), "J shape is not equal to eye shape");
                    let x = &(eye - c * J.clone());
                    LU = Some((self.lu)(x));
                    // println!("LU = {:?} \n", LU);
                }

                let (conv, n_iter_, y_new_, d_) = solve_bdf_system(
                    &self.fun,
                    t_new,
                    &y_predict.clone(),
                    c,
                    &psi.clone(),
                    LU.clone().unwrap(),
                    &mut self.solve_lu,
                    &scale.clone(),
                    self.newton_tol,
                );
                // println!("y_new_ = {:?} \n, d_ = {:?}", y_new_, d_);
                n_iter = n_iter_;
                y_new = y_new_;
                d = d_;
                converged = conv;
                // println!("converged = {:?}, d = {:?}", converged, d);
                if !converged {
                    //?
                    if current_jac {
                        break;
                    }
                    let _J = self.jac.as_mut().unwrap()(t_new, &y_predict);
                    LU = None;
                    current_jac = true;
                }
            } //end while

            if !converged {
                //?
                let factor = 0.5;
                h_abs *= factor;
                change_D(&mut D, order, factor);
                self.n_equal_steps = 0;
                LU = None;
                continue;
            }

            let safety_ = 0.9 * (2.0 * (NEWTON_MAXITER as f64) + 1.0)
                / (2.0 * (NEWTON_MAXITER as f64) + n_iter as f64);
            safety = safety_;
            let scale = scale_func(self.rtol.clone(), self.atol.clone(), &y_new.abs());
            let scale: DVector<f64> = DVector::from_vec(scale);
            let error = error_const[order] * d.clone();
            let error_norm_ = norm(&(error.component_div(&scale)));
            error_norm = error_norm_;
            //          println!("error_norm = {:?}, safety = {:?}, d = {:?}, order {}", error_norm, safety, d.clone(), order);
            if error_norm > 1.0 {
                //?
                let factor =
                    (safety * error_norm.powf(-1.0 / (order as f64 + 1.0))).max(MIN_FACTOR);
                h_abs *= factor;
                change_D(&mut D, order, factor);
                self.n_equal_steps = 0;
            } else {
                step_accepted = true;
            }
        } //end !step_accepted

        self.n_equal_steps += 1;
        self.t = t_new;
        self.y = y_new;
        self.h_abs = h_abs;
        self.J = J;
        self.LU = LU;
        let D_ = D.clone();
        //let row: DVector<f64> =  D_.row(2).transpose().clone();
        //    println!("d= {:?},", &(d.clone())   );
        //D.row_mut(order + 2).copy_from(&(d.clone() - D_.row(order + 1).transpose()  )   );
        D.set_row(order + 2, &(d.clone().transpose() - D_.row(order + 1)));

        D.set_row(order + 1, &d.transpose());

        //  for i in (0..order).rev() {
        //     let D_ = D.clone();// avoiding borrow checker issues with mutable and immutable references
        //      D.row_mut(i).add_assign(D_.row(i + 1));
        //  }
        // for i in (0..order+1).rev() {
        //  D[i] += D[i + 1];
        //   }
        for i in (0..order + 1).rev() {
            let D_ = D.clone();
            D.row_mut(i).add_assign(D_.row(i + 1));
        }

        //     println!("D1 = {:?} \n, {}", D.transpose(), order);
        if self.n_equal_steps < order + 1 {
            //          println!("end step exit_0: t {}, y {:?}", self.t, self.y);
            self.D = D; // !!!
            return (true, None);
        }

        let error_m_norm = if order > 1 {
            let error_m = error_const[order - 1] * D.row(order);

            norm(&(error_m.transpose().component_div(&scale)))
        } else {
            f64::INFINITY
        };

        let error_p_norm = if order < MAX_ORDER {
            let error_p = error_const[order + 1] * D.row(order + 2);

            norm(&(error_p.transpose().component_div(&scale)))
        } else {
            f64::INFINITY
        };

        let error_norms = DVector::from_vec(vec![error_m_norm, error_norm, error_p_norm]);
        // let factors = error_norms.map(|x| x.powf(-1.0 / (order as f64 + 1.0))); //?
        let factors: Vec<f64> = error_norms
            .iter()
            .enumerate()
            .map(|(i, x)| x.powf(-1.0 / (order as f64 + i as f64)))
            .collect(); //?
        let factors: DVector<f64> = factors.into();
        //        println!("factors = {:?}, error_norms = {:?}", factors, error_norms);
        // argmax returns ( index, value)
        let delta_order = factors.argmax().0 - 1;
        let new_order = order + delta_order;
        self.order = new_order;

        let factor = (safety * factors.max()).min(MAX_FACTOR);
        self.h_abs *= factor;
        //       println!("delta_order = {}, new_order = {}, factor = {}", delta_order, new_order, factor);

        change_D(&mut D, self.order, factor);
        self.n_equal_steps = 0;
        self.LU = None;
        //      println!("end step exit_1: t {}, y {:?}", self.t, self.y);
        //    println!(" D end = {:?} \n", &D.transpose());
        self.D = D; // !!!

        (true, None)
    }

    /*
     */
} //impl BDF

/*
pub struct DenseOutput {
    t_old: f64,
    t: f64,
}

impl DenseOutput {
    pub fn new(t_old: f64, t: f64) -> DenseOutput {
        DenseOutput { t_old, t }
    }
}

pub struct BdfDenseOutput {
    t_old: f64,
    t: f64,
    order: usize,
    t_shift: DVector<f64>,
    denom: DVector<f64>,
    D: DMatrix<f64>,
}

impl BdfDenseOutput {
    pub fn new(t_old: f64, t: f64, h: f64, order: usize, D: DMatrix<f64>) -> BdfDenseOutput {  // self.t - h * np.arange(self.order)
        let t_shift = DVector::from_iterator(order, (0..order).map(|i| t - h * i as f64));
        let denom = DVector::from_iterator(order, (1..=order).map(|i| h * i as f64));

        BdfDenseOutput {
            t_old,
            t,
            order,
            t_shift,
            denom,
            D,
        }
    }

    pub fn call(&self, t: f64) -> DVector<f64> {
        let x = if t.is_nan() {
            DVector::zeros(self.order)
        } else {
            (t - self.t_shift) / &self.denom
        };

        let p = x.iter().scan(1.0, |state, &x| Some(state * x)).collect::<DVector<f64>>();

        let y = self.D.slice((1, 0), (self.order, self.D.ncols())).transpose() * p;

        if y.len() == 1 {
            y + self.D.row(0)
        } else {
            y + &self.D.slice((0, 0), (1, self.D.ncols())).into_owned()
        }
    }
}
*/
