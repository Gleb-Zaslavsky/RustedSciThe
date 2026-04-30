//! # Backward Differentiation Formula (BDF) Solver
//!
//! ## Mathematical Foundation
//!
//! The Backward Differentiation Formula (BDF) is a family of implicit methods for solving
//! stiff ordinary differential equations (ODEs) of the form:
//!
//! ```text
//! dy/dt = f(t, y), y(t₀) = y₀
//! ```
//!
//! ### BDF Formula Derivation
//!
//! BDF methods approximate the derivative using backward differences. For a k-step BDF method,
//! the formula is:
//!
//! ```text
//! Σ(i=0 to k) αᵢ yₙ₋ᵢ = h βₖ f(tₙ, yₙ)
//! ```
//!
//! Where:
//! - `yₙ` is the solution at time `tₙ`
//! - `h` is the step size
//! - `αᵢ` are the BDF coefficients
//! - `βₖ = 1` for BDF methods (implicit)
//!
//! ### Stability and Order
//!
//! BDF methods up to order 6 exist, but only orders 1-5 are A-stable:
//! - **Order 1 (Backward Euler)**: `yₙ - yₙ₋₁ = h f(tₙ, yₙ)`
//! - **Order 2**: `3/2 yₙ - 2yₙ₋₁ + 1/2 yₙ₋₂ = h f(tₙ, yₙ)`
//! - **Higher orders**: Use more previous points for better accuracy
//!
//! ### Newton-Raphson Solution
//!
//! At each step, we solve the nonlinear system:
//! ```text
//! G(yₙ) = yₙ - h/αₖ f(tₙ, yₙ) - ψ = 0
//! ```
//!
//! Where `ψ` contains contributions from previous solution values.
//!
//! Using Newton-Raphson iteration:
//! ```text
//! yₙ⁽ᵐ⁺¹⁾ = yₙ⁽ᵐ⁾ - [I - h/αₖ J]⁻¹ G(yₙ⁽ᵐ⁾)
//! ```
//!
//! Where `J = ∂f/∂y` is the Jacobian matrix.
//!
//! ## Implementation Details
//!
//! ### Variable Order and Step Size Control
//!
//! This implementation uses:
//! - **Adaptive order**: Automatically adjusts between orders 1-5 based on error estimates
//! - **Adaptive step size**: Uses local error estimation to control step size
//! - **Nordsieck form**: Stores solution history as scaled derivatives for efficiency
//!
//! ### Key Components
//!
//! 1. **Difference Matrix (D)**: Stores scaled backward differences
//!    - `D[0,:]` = current solution
//!    - `D[1,:]` = h * f(t,y) (scaled first derivative)
//!    - `D[k,:]` = k-th scaled backward difference
//!
//! 2. **Coefficient Arrays**:
//!    - `alpha`: BDF coefficients for different orders
//!    - `gamma`: Integration coefficients
//!    - `error_const`: Error estimation coefficients
//!
//! 3. **Error Control**: Uses weighted RMS norm for error estimation:
//!    ```text
//!    ||e||ᵣₘₛ = sqrt(1/n Σ(eᵢ/(atol + rtol*|yᵢ|))²)
//!    ```
//!
//! ### Algorithm Flow
//!
//! 1. **Prediction**: Use Nordsieck array to predict solution
//! 2. **Correction**: Solve nonlinear system with Newton-Raphson
//! 3. **Error Estimation**: Compute local truncation error
//! 4. **Step/Order Control**: Accept/reject step and adjust order/step size
//! 5. **Update**: Update Nordsieck array for next step
//!
//! ### Stiffness Handling
//!
//! BDF methods are particularly effective for stiff problems because:
//! - **A-stability**: Unconditionally stable for orders 1-2
//! - **L-stability**: Good damping of high-frequency components
//! - **Implicit nature**: Can handle large step sizes for stiff systems
//!
//! ## References
//!
//! - Byrne, G.D., Hindmarsh, A.C. "A Polyalgorithm for the Numerical Solution of ODEs"
//! - Shampine, L.F., Reichelt, M.W. "The MATLAB ODE Suite"
//! - Hairer, E., Wanner, G. "Solving Ordinary Differential Equations II: Stiff Problems"

extern crate nalgebra as na;
extern crate num_traits;
extern crate sprs;

use na::{DMatrix, DVector, Dyn, LU};

use log::info;
use std::f64;
use std::ops::AddAssign;

use crate::numerical::BDF::BDF_utils::{OrderEnum, group_columns};

use crate::numerical::BDF::common::{
    NumberOrVec, check_arguments, is_sparse, newton_tol, norm, scale_func, select_initial_step,
    validate_first_step, validate_max_step, validate_tol,
};
use crate::somelinalg::banded::storage::Banded;
use faer::sparse::Triplet;
use std::fmt::Debug;
use std::fmt::Display;
const MAX_ORDER: usize = 5;
const NEWTON_MAXITER: usize = 4;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const SPARSE: f64 = 0.01;

/// Factorized Newton matrix used by one BDF correction step.
///
/// This intentionally mirrors the narrow operation BDF needs from linear
/// algebra: solve `[I - cJ] * x = rhs`.  Keeping the interface this small lets
/// LSODE2 later plug in `faer` sparse LU or faithful LAPACK-style banded LU
/// without changing the BDF predictor/corrector mathematics.
pub trait BdfLinearFactorization {
    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>>;
}

/// Backend that factorizes BDF Newton matrices.
pub trait BdfLinearBackend {
    fn factor(&mut self, matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>>;

    /// Factorizes the Newton matrix `[I - cJ]` from the Jacobian representation.
    ///
    /// The default implementation preserves the legacy dense path.  Native
    /// sparse/banded backends can override this method and avoid materializing a
    /// dense Newton matrix.
    fn factor_shifted_jacobian(
        &mut self,
        c: f64,
        jacobian: &BdfJacobian,
    ) -> Option<Box<dyn BdfLinearFactorization>> {
        let matrix = jacobian.to_shifted_dense(c)?;
        self.factor(&matrix)
    }
}

/// Jacobian storage accepted by BDF linear backends.
///
/// This is deliberately a solver-facing representation, not a symbolic API.
/// Existing BDF users still provide dense `DMatrix` Jacobians; LSODE2 can grow
/// native sparse/banded Jacobian routes by producing these variants directly.
#[derive(Clone, Debug)]
pub enum BdfJacobian {
    Dense(DMatrix<f64>),
    SparseTriplets {
        n: usize,
        triplets: Vec<Triplet<usize, usize, f64>>,
    },
    Banded(Banded<f64>),
}

impl BdfJacobian {
    pub fn from_dense(matrix: DMatrix<f64>) -> Self {
        Self::Dense(matrix)
    }

    pub fn n(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.nrows(),
            Self::SparseTriplets { n, .. } => *n,
            Self::Banded(matrix) => matrix.n(),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        let n = self.n();
        (n, n)
    }

    pub fn to_shifted_dense(&self, c: f64) -> Option<DMatrix<f64>> {
        let n = self.n();
        let mut out = DMatrix::identity(n, n);
        match self {
            Self::Dense(matrix) => {
                if matrix.nrows() != matrix.ncols() {
                    return None;
                }
                out -= c * matrix;
            }
            Self::SparseTriplets { triplets, .. } => {
                for triplet in triplets {
                    if triplet.row >= n || triplet.col >= n {
                        return None;
                    }
                    out[(triplet.row, triplet.col)] -= c * triplet.val;
                }
            }
            Self::Banded(matrix) => {
                for j in 0..n {
                    let i0 = j.saturating_sub(matrix.ku());
                    let i1 = (j + matrix.kl() + 1).min(n);
                    for i in i0..i1 {
                        out[(i, j)] -= c * matrix[(i, j)];
                    }
                }
            }
        }
        Some(out)
    }

    pub fn to_shifted_sparse_triplets(&self, c: f64) -> Option<Vec<Triplet<usize, usize, f64>>> {
        let n = self.n();
        let mut triplets = Vec::new();
        let mut diagonal = vec![1.0; n];

        match self {
            Self::Dense(matrix) => {
                if matrix.nrows() != matrix.ncols() {
                    return None;
                }
                for j in 0..n {
                    for i in 0..n {
                        let value = matrix[(i, j)];
                        if value != 0.0 {
                            if i == j {
                                diagonal[i] -= c * value;
                            } else {
                                triplets.push(Triplet::new(i, j, -c * value));
                            }
                        }
                    }
                }
            }
            Self::SparseTriplets {
                n: sparse_n,
                triplets: sparse_triplets,
            } => {
                if *sparse_n != n {
                    return None;
                }
                for triplet in sparse_triplets {
                    if triplet.row >= n || triplet.col >= n {
                        return None;
                    }
                    if triplet.row == triplet.col {
                        diagonal[triplet.row] -= c * triplet.val;
                    } else {
                        triplets.push(Triplet::new(triplet.row, triplet.col, -c * triplet.val));
                    }
                }
            }
            Self::Banded(matrix) => {
                for j in 0..n {
                    let i0 = j.saturating_sub(matrix.ku());
                    let i1 = (j + matrix.kl() + 1).min(n);
                    for i in i0..i1 {
                        let value = matrix[(i, j)];
                        if value != 0.0 {
                            if i == j {
                                diagonal[i] -= c * value;
                            } else {
                                triplets.push(Triplet::new(i, j, -c * value));
                            }
                        }
                    }
                }
            }
        }

        for (i, value) in diagonal.into_iter().enumerate() {
            if value != 0.0 {
                triplets.push(Triplet::new(i, i, value));
            }
        }

        Some(triplets)
    }

    pub fn to_shifted_banded(&self, c: f64) -> Option<Banded<f64>> {
        match self {
            Self::Banded(matrix) => {
                let n = matrix.n();
                let mut out = Banded::<f64>::zeros(n, matrix.kl(), matrix.ku()).ok()?;
                out.fill_from_dense(|i, j| {
                    let identity = if i == j { 1.0 } else { 0.0 };
                    identity - c * matrix[(i, j)]
                });
                Some(out)
            }
            Self::Dense(matrix) => dense_shifted_to_banded(matrix, c),
            Self::SparseTriplets { n, triplets } => {
                let mut kl = 0usize;
                let mut ku = 0usize;
                for triplet in triplets {
                    if triplet.row >= *n || triplet.col >= *n {
                        return None;
                    }
                    kl = kl.max(triplet.row.saturating_sub(triplet.col));
                    ku = ku.max(triplet.col.saturating_sub(triplet.row));
                }
                let mut out = Banded::<f64>::zeros(*n, kl, ku).ok()?;
                for i in 0..*n {
                    out.set(i, i, 1.0).ok()?;
                }
                for triplet in triplets {
                    let current = *out.get(triplet.row, triplet.col).unwrap_or(&0.0);
                    out.set(triplet.row, triplet.col, current - c * triplet.val)
                        .ok()?;
                }
                Some(out)
            }
        }
    }
}

fn dense_shifted_to_banded(matrix: &DMatrix<f64>, c: f64) -> Option<Banded<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return None;
    }

    let n = matrix.nrows();
    let mut kl = 0usize;
    let mut ku = 0usize;
    for j in 0..n {
        for i in 0..n {
            let shifted = if i == j { 1.0 } else { 0.0 } - c * matrix[(i, j)];
            if shifted != 0.0 {
                kl = kl.max(i.saturating_sub(j));
                ku = ku.max(j.saturating_sub(i));
            }
        }
    }

    let mut out = Banded::<f64>::zeros(n, kl, ku).ok()?;
    out.fill_from_dense(|i, j| {
        let identity = if i == j { 1.0 } else { 0.0 };
        identity - c * matrix[(i, j)]
    });
    Some(out)
}

struct DenseNalgebraFactorization {
    lu: LU<f64, Dyn, Dyn>,
}

impl BdfLinearFactorization for DenseNalgebraFactorization {
    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        self.lu.solve(rhs)
    }
}

struct DenseNalgebraLinearBackend;

impl BdfLinearBackend for DenseNalgebraLinearBackend {
    fn factor(&mut self, matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        Some(Box::new(DenseNalgebraFactorization {
            lu: LU::new(matrix.clone()),
        }))
    }
}
/// Computes cumulative product along columns of a matrix.
///
/// For each column j, computes the cumulative product:
/// ```text
/// result[i,j] = ∏(k=0 to i) matrix[k,j]
/// ```
///
/// This is used in the BDF coefficient computation where we need
/// cumulative products of scaling factors.
///
/// # Parameters
/// * `matrix` - Input matrix for cumulative product computation
///
/// # Returns
/// Matrix where each element is the cumulative product from top to current row
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

/// Computes the R matrix for BDF step size and order changes.
///
/// The R matrix is used to transform the Nordsieck array when the step size
/// changes by a factor. The matrix elements are computed as:
/// ```text
/// R[i,j] = (i-1-factor*j)/i  for i,j ≥ 1
/// R[0,j] = 1                 for all j
/// ```
///
/// Then cumulative products are taken along columns to get the final R matrix.
///
/// # Parameters
/// * `order` - Current BDF order (determines matrix size)
/// * `factor` - Step size scaling factor (new_h/old_h)
///
/// # Returns
/// Transformation matrix R of size (order+1) × (order+1)
///
/// # Mathematical Background
/// When step size changes from h to h*factor, the scaled derivatives
/// in the Nordsieck array must be transformed accordingly.
fn compute_r(order: usize, factor: f64) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(order + 1, order + 1);
    for i in 1..(order + 1) {
        for j in 1..(order + 1) {
            m[(i, j)] = (i as f64 - 1.0 - factor * j as f64) / i as f64;
        }
    }
    m.row_mut(0).fill(1.0);
    let result = cumulative_product_along_columns(&m);
    result
}

/// Updates the Nordsieck difference array when step size changes.
///
/// When the step size changes by a factor, the scaled derivatives in the
/// Nordsieck array D must be transformed to maintain consistency.
/// The transformation is: D_new = (R*U)^T * D_old
///
/// Where:
/// - R = transformation matrix for the new step size factor
/// - U = transformation matrix for factor = 1 (identity transformation)
/// - D contains scaled derivatives: [y, h*y', h²*y''/2!, ...]
///
/// # Parameters
/// * `D` - Nordsieck difference matrix to be updated in-place
/// * `order` - Current BDF order
/// * `factor` - Step size scaling factor (h_new/h_old)
///
/// # Mathematical Formula
/// ```text
/// D[0:order+1] = (R(order,factor) * R(order,1))^T * D[0:order+1]
/// ```
fn change_D(D: &mut DMatrix<f64>, order: usize, factor: f64) {
    let r = compute_r(order, factor);
    let u = compute_r(order, 1.0);
    let ru = r * u;
    let temp = ru.transpose() * D.rows(0, order + 1);
    D.rows_mut(0, order + 1).copy_from(&temp);
}

/// Solves the nonlinear BDF system using Newton-Raphson iteration.
///
/// At each BDF step, we must solve the nonlinear system:
/// ```text
/// G(y) = y - c*f(t,y) - ψ = 0
/// ```
///
/// Where:
/// - c = h/α₀ (step size divided by leading BDF coefficient)
/// - ψ = (1/α₀) * Σ(i=1 to k) αᵢ*yₙ₋ᵢ (contribution from previous values)
/// - f(t,y) is the ODE right-hand side
///
/// Newton-Raphson iteration:
/// ```text
/// y^(m+1) = y^(m) - [I - c*J]^(-1) * G(y^(m))
/// ```
///
/// Where J = ∂f/∂y is the Jacobian matrix.
///
/// # Parameters
/// * `fun` - ODE right-hand side function f(t,y)
/// * `t_new` - Target time for the solution
/// * `y_predict` - Initial guess from predictor step
/// * `c` - BDF coefficient c = h/α₀
/// * `psi` - Vector ψ containing previous step contributions
/// * `lumatrx` - LU factorization of [I - c*J]
/// * `solve_lu` - Function to solve linear systems with LU factorization
/// * `scale` - Scaling vector for error norm computation
/// * `tol` - Newton iteration tolerance
///
/// # Returns
/// * `converged` - Whether Newton iteration converged
/// * `iterations` - Number of Newton iterations performed
/// * `y` - Final solution at t_new
/// * `d` - Correction vector (y_final - y_predict)
fn solve_bdf_system<F>(
    fun: F,
    t_new: f64,
    y_predict: &DVector<f64>,
    c: f64,
    psi: &DVector<f64>,
    linear_factorization: &dyn BdfLinearFactorization,
    scale: &DVector<f64>, // scale
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
        let f = fun(t_new, &y);
        // checks if all elements in the vector f are finite. If any element is not finite, the loop is broken.
        if !f.iter().all(|&x| x.is_finite()) {
            break;
        }
        // The dy vector represents the change in the solution vector y at each iteration of the Newton-Raphson method
        let Some(dy) = linear_factorization.solve(&(c * &f - psi - &d)) else {
            break;
        };
        // The dy_norm value is then used to determine the convergence of the Newton-Raphson method and to adjust the step size in the BDF method
        let dy_norm = norm(&(dy.component_div(scale)));
        // Calculate the rate of convergence for the Newton-Raphson method
        let rate: Option<f64> = if let Some(dy_norm_old) = dy_norm_old {
            Some(dy_norm / dy_norm_old)
        } else {
            None
        };
        // If the rate of convergence is too slow, break the Newton iteration
        if let Some(rate) = rate {
            if rate >= 1.0
                || (rate.powi((NEWTON_MAXITER - k) as i32) / (1.0 - rate)) * dy_norm > tol
            {
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
        if let Some(rate) = rate {
            if rate / (1.0 - rate) * dy_norm < tol {
                converged = true;
                break;
            }
        }
        dy_norm_old = Some(dy_norm);
    }
    (converged, k_ + 1, y, d)
}

/// Backward Differentiation Formula (BDF) solver for stiff ODEs.
///
/// This struct implements a variable-order, variable-step-size BDF method
/// for solving initial value problems of the form:
/// ```text
/// dy/dt = f(t,y), y(t₀) = y₀
/// ```
///
/// # Key Features
/// - **Variable order**: Automatically adjusts between orders 1-5
/// - **Variable step size**: Adaptive step size control based on error estimates
/// - **Nordsieck form**: Efficient storage of solution history as scaled derivatives
/// - **Newton-Raphson**: Implicit system solution with Jacobian reuse
/// - **Stiffness handling**: A-stable methods suitable for stiff problems
///
/// # Mathematical Foundation
/// The k-step BDF formula is:
/// ```text
/// Σ(i=0 to k) αᵢ yₙ₋ᵢ = h f(tₙ, yₙ)
/// ```
///
/// # Data Organization
/// - **D matrix**: Nordsieck array storing scaled derivatives [y, h*y', h²*y''/2!, ...]
/// - **Coefficients**: α (BDF), γ (integration), error constants
/// - **Linear algebra**: Cached LU factorization for efficiency
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
    jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>>,
    J: BdfJacobian,
    I: DMatrix<f64>,
    linear_backend: Box<dyn BdfLinearBackend>,
    gamma: DVector<f64>,
    alpha: DVector<f64>,
    error_const: DVector<f64>,
    D: DMatrix<f64>,
    order: usize,
    max_order_cap: usize,
    n_equal_steps: usize,
    linear_factorization: Option<Box<dyn BdfLinearFactorization>>,
    nlu: usize,
    nfev: usize,
    njev: usize,
    pub direction: f64,
}

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
            .field("max_order_cap", &self.max_order_cap)
            .field(
                "linear_factorization_cached",
                &self.linear_factorization.is_some(),
            )
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

    /// Creates a new BDF solver instance with default parameters.
    ///
    /// This constructor initializes the solver with placeholder values.
    /// The actual problem setup is done via `set_initial()`.
    ///
    /// # Returns
    /// A new BDF solver instance ready for initialization
    pub fn new() -> Self {
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
            J: BdfJacobian::from_dense(DMatrix::zeros(0, 0)),
            I: DMatrix::zeros(0, 0),
            linear_backend: Box::new(DenseNalgebraLinearBackend),
            gamma: DVector::zeros(0),
            alpha: DVector::zeros(0),
            error_const: DVector::zeros(0),
            D: DMatrix::zeros(0, 0),
            order: 1,
            max_order_cap: MAX_ORDER,
            n_equal_steps: 0,
            linear_factorization: None,
            nlu: 0,
            direction: 1.0,
            nfev: 0,
            njev: 0,
            n: 0,
            t_old: None,
        }
    }

    pub fn counters(&self) -> (usize, usize, usize) {
        (self.nfev, self.njev, self.nlu)
    }

    /// Caps adaptive BDF order selection.
    ///
    /// The implementation keeps coefficient storage for the full tested BDF
    /// range, but LSODE2 can use this hook to expose LSODE-style algorithm
    /// policy without forking the BDF stepper.
    pub fn set_max_order_cap(&mut self, max_order_cap: usize) {
        assert!(
            (1..=MAX_ORDER).contains(&max_order_cap),
            "BDF max order cap must be in 1..={MAX_ORDER}, got {max_order_cap}"
        );
        self.max_order_cap = max_order_cap;
        if self.order > self.max_order_cap {
            self.order = self.max_order_cap;
            self.linear_factorization = None;
            self.n_equal_steps = 0;
        }
    }

    pub fn max_order_cap(&self) -> usize {
        self.max_order_cap
    }

    pub fn current_order(&self) -> usize {
        self.order
    }

    pub fn equal_step_count(&self) -> usize {
        self.n_equal_steps
    }

    /// Replaces the Newton linear backend and drops any cached factorization.
    ///
    /// This is the intentional extension point for LSODE2.  The BDF stepper
    /// still builds the same Newton matrix `[I - cJ]`; only the factor/solve
    /// implementation changes.
    pub fn set_linear_backend(&mut self, backend: Box<dyn BdfLinearBackend>) {
        self.linear_backend = backend;
        self.linear_factorization = None;
    }

    /// Replaces the Jacobian evaluator with a native-storage evaluator.
    ///
    /// Existing BDF callers still use dense Jacobian closures through
    /// `set_initial`.  LSODE2 can use this hook to install sparse triplet or
    /// banded Jacobian evaluators and let the selected linear backend form
    /// `[I - cJ]` without a dense round-trip.
    pub fn set_native_jacobian(
        &mut self,
        mut jacobian: Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>,
    ) {
        let initial = jacobian(self.t, &self.y);
        assert_eq!(
            initial.shape(),
            (self.n, self.n),
            "native Jacobian shape is not equal to solver dimension"
        );
        self.jac = Some(jacobian);
        self.J = initial;
        self.linear_factorization = None;
        self.njev += 1;
    }

    /// Initializes the BDF solver with problem-specific parameters.
    ///
    /// Sets up the ODE system, tolerances, Jacobian, and BDF coefficients.
    /// Computes initial step size and initializes the Nordsieck array.
    ///
    /// # Parameters
    /// * `fun` - ODE right-hand side function f(t,y) → dy/dt
    /// * `t0` - Initial time
    /// * `y0` - Initial solution vector
    /// * `t_bound` - Final integration time
    /// * `_max_step` - Maximum allowed step size
    /// * `rtol` - Relative error tolerance (scalar or vector)
    /// * `atol` - Absolute error tolerance (scalar or vector)
    /// * `jac` - Optional Jacobian function ∂f/∂y
    /// * `jac_sparsity` - Optional sparsity pattern for numerical Jacobian
    /// * `vectorized` - Whether function supports vectorized evaluation
    /// * `first_step` - Optional initial step size (auto-selected if None)
    ///
    /// # Mathematical Setup
    /// Initializes BDF coefficients:
    /// - α: BDF coefficients for different orders
    /// - γ: Integration coefficients γₖ = Σ(i=1 to k) 1/i
    /// - error_const: Local error estimation coefficients
    ///
    /// Nordsieck array initialization:
    /// ```text
    /// D[0] = y₀
    /// D[1] = h * f(t₀, y₀)
    /// ```
    pub fn set_initial(
        &mut self,
        fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
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
        info!("prelude done");
        // initialize parameters, if some parameters are not provided by the user then we will use special functions
        let max_step = validate_max_step(self.max_step);
        self.max_step = max_step.unwrap();

        let (rtol, atol) = validate_tol(rtol, atol, self.n).unwrap();
        self.rtol = rtol.clone();
        self.atol = atol.clone();
        info!("tolerance validation: done");
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
        self.h_abs_old = None;
        self.error_norm_old = None;
        self.newton_tol = newton_tol(rtol);
        self.jac_factor = None;

        // kappa: This array contains the coefficients for the BDF method. The values are specific to the BDF method and
        // are used to calculate the differences between the solution and the predicted solution.
        let kappa = DVector::from_vec(vec![0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0]);

        // Match Python: self.gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
        let gamma = {
            let mut g = vec![0.0];
            let mut cumsum = 0.0;
            for i in 1..=MAX_ORDER {
                cumsum += 1.0 / (i as f64);
                g.push(cumsum);
            }
            DVector::from_vec(g)
        };

        // Match Python: self.alpha = (1 - kappa) * self.gamma
        let alpha = (DVector::from(vec![1.0; MAX_ORDER + 1]) - kappa.clone()).component_mul(&gamma);

        let error_const = kappa.component_mul(&gamma)
            + DVector::from_iterator(MAX_ORDER + 1, (1..=MAX_ORDER + 1).map(|i| 1.0 / i as f64));

        // Debug assertions to ensure correct initialization
        assert_eq!(alpha.len(), MAX_ORDER + 1);
        assert_eq!(gamma.len(), MAX_ORDER + 1);
        assert_eq!(error_const.len(), MAX_ORDER + 1);
        self.alpha = alpha;
        self.error_const = error_const;
        self.gamma = gamma;

        let mut D = DMatrix::zeros(MAX_ORDER + 3, self.y.len());
        assert!(
            D.nrows() >= MAX_ORDER + 3,
            "D matrix must have at least MAX_ORDER + 3 rows"
        );
        info!("created matrix of size: {:?} x {:?}", D.nrows(), D.ncols());

        D.set_row(0, &self.y.transpose());
        D.set_row(1, &(f * h_abs * self.direction).transpose());
        self.D = D;

        self.order = 1;
        self.n_equal_steps = 0;
        self.linear_factorization = None;
        self.create_funct();
        self.validate_jac(jac, jac_sparsity);
    }

    /// Creates linear algebra functions for sparse or dense matrices.
    ///
    /// Sets up LU factorization and linear system solving functions
    /// optimized for either sparse or dense Jacobian matrices.
    ///
    /// # Implementation Details
    /// - **Sparse**: Uses specialized sparse LU factorization
    /// - **Dense**: Uses standard dense LU factorization
    /// - **Identity matrix**: Creates I for Newton system [I - c*J]
    fn create_funct(&mut self) {
        // Milestone 1 keeps the tested dense nalgebra path as the default
        // backend.  The trait boundary is deliberately here, next to the
        // Newton matrix cache, so LSODE2 can install sparse/banded backends
        // without rewriting the BDF stepper.
        self.linear_backend = Box::new(DenseNalgebraLinearBackend);
        self.I = DMatrix::identity(self.n, self.n);
        info!("linear backend creation: dense nalgebra");
    }
    /// Performs initial setup and validation of solver parameters.
    ///
    /// This method handles the basic initialization that's common to all
    /// BDF solvers, including argument validation and direction determination.
    ///
    /// # Parameters
    /// * `fun` - ODE right-hand side function
    /// * `t0` - Initial time
    /// * `y0` - Initial solution vector
    /// * `t_bound` - Final integration time
    /// * `vectorized` - Whether function supports vectorized calls
    ///
    /// # Sets
    /// - Integration direction: sign(t_bound - t0)
    /// - Problem dimension: n = length(y0)
    /// - Function evaluation counter: nfev = 0
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
        self.fun = fun;

        self.direction = if t_bound != t0 {
            (t_bound - t0).signum()
        } else {
            1.0
        };

        self.nfev = 0;
    }

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

    /// Validates and sets up the Jacobian computation strategy.
    ///
    /// Handles three cases:
    /// 1. **Analytical Jacobian**: User-provided function J = ∂f/∂y
    /// 2. **Numerical Jacobian**: Finite difference approximation (not implemented)
    /// 3. **No Jacobian**: Uses identity matrix (not recommended)
    ///
    /// # Parameters
    /// * `jac` - Optional analytical Jacobian function
    /// * `sparsity` - Optional sparsity pattern for numerical differentiation
    ///
    /// # Mathematical Background
    /// The Jacobian matrix J[i,j] = ∂fᵢ/∂yⱼ is crucial for:
    /// - Newton-Raphson convergence: [I - c*J] Δy = -G(y)
    /// - Stability analysis: eigenvalues of J determine stiffness
    /// - Efficiency: analytical J is much faster than numerical
    ///
    /// # Implementation Notes
    /// - Wraps user Jacobian for consistent interface
    /// - Handles both sparse and dense matrices
    /// - Increments Jacobian evaluation counter (njev)
    fn validate_jac(
        &mut self,
        jac: Option<Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>>,
        sparsity: Option<DMatrix<f64>>,
    ) {
        let t0 = self.t;
        let y0 = self.y.clone();
        let _sparsity_ = 0.0;
        // if jac is None, then we calculate the jacobian using the num_jac function and return a wrapped function that computes the jacobian
        // at a given point

        match jac {
            Some(jac) => {
                info!("analytical jacobian used");
                let J = jac(t0, &y0);
                self.njev += 1;

                let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> =
                    if is_sparse(&J, SPARSE) {
                        Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                            // self.njev += 1;
                            BdfJacobian::from_dense(jac(t, &y))
                        })
                    } else {
                        Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                            //   self.njev += 1;
                            BdfJacobian::from_dense(jac(t, &y))
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
                self.J = BdfJacobian::from_dense(J.clone());
            }
            _ => {
                let _new_sparsity: Option<(DMatrix<f64>, Vec<usize>)> =
                    if let Some(sparsity) = sparsity {
                        let groups = group_columns(&sparsity, OrderEnum::None);
                        Some((sparsity.clone(), groups))
                    } else {
                        None
                    };
            }
        };

        info!("jac validation: done");
    }

    /// Performs one BDF integration step with adaptive order and step size control.
    ///
    /// This is the core BDF algorithm implementing:
    /// 1. **Step size control**: Ensures h ∈ [h_min, h_max]
    /// 2. **Prediction**: Uses Nordsieck array to predict solution
    /// 3. **Correction**: Solves nonlinear system with Newton-Raphson
    /// 4. **Error estimation**: Computes local truncation error
    /// 5. **Acceptance**: Accept/reject step based on error tolerance
    /// 6. **Order selection**: Choose optimal order for next step
    /// 7. **Nordsieck update**: Update scaled derivatives array
    ///
    /// # Returns
    /// * `(true, None)` - Step accepted successfully
    /// * `(false, Some(msg))` - Step failed with error message
    ///
    /// # Mathematical Algorithm
    ///
    /// **Prediction Phase**:
    /// ```text
    /// y_predict = Σ(i=0 to k) D[i]  (sum of Nordsieck array)
    /// ```
    ///
    /// **Correction Phase** (Newton-Raphson):
    /// ```text
    /// Solve: [I - c*J] Δy = c*f(t,y) - ψ - d
    /// Where: c = h/αₖ, ψ = D[1:k]^T * γ[1:k] / αₖ
    /// ```
    ///
    /// **Error Estimation**:
    /// ```text
    /// error = Cₖ * d  (where Cₖ is error constant)
    /// error_norm = ||error / scale||_RMS
    /// ```
    ///
    /// **Step Control**:
    /// ```text
    /// factor = safety * error_norm^(-1/(k+1))
    /// h_new = h * clamp(factor, MIN_FACTOR, MAX_FACTOR)
    /// ```
    pub fn _step_impl(&mut self) -> (bool, Option<&'static str>) {
        let t = self.t;
        let mut D = self.D.clone();
        let max_step = self.max_step;
        let min_step = 10.0 * f64::MIN;

        let mut h_abs = if self.h_abs > max_step {
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

        let order = self.order;
        assert!(
            order <= self.max_order_cap,
            "Order cannot exceed configured max order cap"
        );
        assert!(
            order < self.alpha.len(),
            "Order must be within alpha bounds"
        );

        let alpha = &self.alpha;
        let gamma = &self.gamma;
        let error_const = &self.error_const;
        let mut J = self.J.clone();
        let mut linear_factorization = self.linear_factorization.take();
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
        let mut d = DVector::zeros(self.n);
        while !step_accepted {
            if h_abs < min_step {
                return (false, "step size too small".into());
            }

            let h = h_abs * self.direction;
            let t_new_ = t + h;

            if self.direction * (t_new - self.t_bound) > 0.0 {
                t_new = self.t_bound;

                change_D(&mut D, order, (t_new - t).abs() / h_abs);
                self.n_equal_steps = 0;
                linear_factorization = None;
            }
            t_new = t_new_;

            let h = t_new - t;
            h_abs = h.abs();
            let y_predict = D.rows(0, order + 1).row_sum().transpose();
            let y_predict_abs = y_predict.abs();
            let scale_ = scale_func(self.rtol.clone(), self.atol.clone(), &y_predict_abs);
            let scale_: DVector<f64> = DVector::from_vec(scale_);
            scale = scale_;
            let psi = D.rows(1, order).transpose() * gamma.rows(1, order) / alpha[order];
            let mut converged = false;
            let c = h / alpha[order];

            while !converged {
                if linear_factorization.is_none() {
                    assert_eq!(
                        J.shape(),
                        (self.n, self.n),
                        "J shape is not equal to solver dimension"
                    );
                    linear_factorization = self.linear_backend.factor_shifted_jacobian(c, &J);
                    self.nlu += 1;
                    if linear_factorization.is_none() {
                        break;
                    }
                }

                let (conv, n_iter_, y_new_, d_) = solve_bdf_system(
                    &self.fun,
                    t_new,
                    &y_predict.clone(),
                    c,
                    &psi.clone(),
                    linear_factorization.as_ref().unwrap().as_ref(),
                    &scale.clone(),
                    self.newton_tol,
                );
                n_iter = n_iter_;
                y_new = y_new_;
                d = d_;
                converged = conv;
                if !converged {
                    if current_jac {
                        break;
                    }
                    J = self.jac.as_mut().unwrap()(t_new, &y_predict);
                    self.njev += 1;
                    linear_factorization = None;
                    current_jac = true;
                }
            }

            if !converged {
                let factor = 0.5;
                h_abs *= factor;
                change_D(&mut D, order, factor);
                self.n_equal_steps = 0;
                linear_factorization = None;
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
            if error_norm > 1.0 {
                let factor =
                    (safety * error_norm.powf(-1.0 / (order as f64 + 1.0))).max(MIN_FACTOR);
                h_abs *= factor;
                change_D(&mut D, order, factor);
                self.n_equal_steps = 0;
            } else {
                step_accepted = true;
            }
        }

        self.n_equal_steps += 1;
        self.t = t_new;
        self.y = y_new;
        self.h_abs = h_abs;
        self.J = J;
        self.linear_factorization = linear_factorization;
        let D_ = D.clone();
        D.set_row(order + 2, &(d.clone().transpose() - D_.row(order + 1)));

        D.set_row(order + 1, &d.transpose());

        for i in (0..order + 1).rev() {
            let D_ = D.clone();
            D.row_mut(i).add_assign(D_.row(i + 1));
        }

        if self.n_equal_steps < order + 1 {
            self.D = D;
            return (true, None);
        }

        let error_m_norm = if order > 1 {
            let error_m = error_const[order - 1] * D.row(order);

            norm(&(error_m.transpose().component_div(&scale)))
        } else {
            f64::INFINITY
        };

        let error_p_norm = if order < self.max_order_cap {
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
        // Python: delta_order = np.argmax(factors) - 1
        // This gives delta_order ∈ {-1, 0, 1} since factors has 3 elements
        let argmax_index = factors.argmax().0;
        let delta_order = (argmax_index as i32) - 1; // Can be -1, 0, or 1
        let new_order = ((order as i32) + delta_order)
            .max(1)
            .min(self.max_order_cap as i32) as usize;
        self.order = new_order;

        let factor = (safety * factors.max()).min(MAX_FACTOR);
        self.h_abs *= factor;

        change_D(&mut D, self.order, factor);
        self.n_equal_steps = 0;
        self.linear_factorization = None;
        self.D = D;

        (true, None)
    }
}
