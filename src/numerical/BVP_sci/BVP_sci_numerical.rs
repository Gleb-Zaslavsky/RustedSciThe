//! Pure numerical `BVP_sci` workflow with no symbolic machinery.
//!
//! This module exposes a user-facing trait for pointwise BVP definitions:
//! users implement the ODE RHS and boundary residuals in plain Rust, while the
//! collocation system and mesh refinement are built numerically on top of the
//! `faer` solver core.
//!
//! Two pure numerical workflows are intentionally first-class:
//! - `NumericalJacobianMode::FiniteDifference` is the low-friction mode for
//!   users who provide only `rhs()` and `boundary_residual()`;
//! - `NumericalJacobianMode::AnalyticalPointwise` is the high-performance mode
//!   for users who can also provide pointwise Jacobians.
//!
//! Current compare diagnostics suggest a clear rule of thumb:
//! finite differences are often perfectly adequate for small systems, but on
//! medium and large BVPs the pointwise analytical Jacobian path is the one that
//! scales reliably and preserves the strongest performance.

use crate::numerical::BVP_Damp::BVP_utils::CustomTimer;
use crate::numerical::BVP_sci::BVP_sci_faer::{
    BCFunction, BCJacobian, BVPResult, ODEFunction, ODEJacobian, faer_col, faer_dense_mat,
    faer_mat, solve_bvp,
};
use faer::sparse::Triplet;
use std::sync::Arc;

type NumericalRhsClosure = dyn Fn(f64, &[f64], &[f64], &mut [f64]) + Send + Sync;
type NumericalBoundaryResidualClosure = dyn Fn(&[f64], &[f64], &[f64], &mut [f64]) + Send + Sync;
type NumericalRhsJacobianClosure = dyn Fn(f64, &[f64], &[f64]) -> Option<faer_mat> + Send + Sync;
type NumericalBoundaryJacobianClosure =
    dyn Fn(&[f64], &[f64], &[f64]) -> Option<(faer_mat, faer_mat, Option<faer_mat>)> + Send + Sync;

/// How the pure numerical branch obtains Jacobians.
///
/// This is intentionally explicit in the public API:
/// users either provide only RHS / boundary callbacks and rely on
/// finite-difference Jacobians, or they provide analytical pointwise
/// Jacobians together with those callbacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalJacobianMode {
    /// Build Jacobians numerically with finite differences.
    FiniteDifference,
    /// Use user-provided analytical pointwise Jacobians.
    AnalyticalPointwise,
}

/// Closure-based pure numerical BVP definition.
///
/// This is the ergonomic route for callers that do not want to implement a
/// custom trait type. It mirrors the trait-based API but keeps the problem
/// definition in plain closures and makes the Jacobian mode explicit.
#[derive(Clone)]
pub struct NumericalBvpClosureProblem {
    dimension: usize,
    parameter_dimension: usize,
    rhs: Arc<NumericalRhsClosure>,
    boundary_residual: Arc<NumericalBoundaryResidualClosure>,
    rhs_jacobian: Option<Arc<NumericalRhsJacobianClosure>>,
    rhs_param_jacobian: Option<Arc<NumericalRhsJacobianClosure>>,
    boundary_jacobian: Option<Arc<NumericalBoundaryJacobianClosure>>,
    jacobian_mode: NumericalJacobianMode,
}

/// User-facing alias for the closure-first pure numerical BVP route.
///
/// This name is intentionally short and stable for public examples and guides:
/// users should be able to write `BvpSciNumericalProblem::new_fd(...)` without
/// learning any symbolic API details.
pub type BvpSciNumericalProblem = NumericalBvpClosureProblem;

impl NumericalBvpClosureProblem {
    /// Construct a low-friction numerical BVP problem that relies on finite
    /// differences for all Jacobians.
    pub fn new_fd<Rhs, Bc>(
        dimension: usize,
        parameter_dimension: usize,
        rhs: Rhs,
        boundary_residual: Bc,
    ) -> Self
    where
        Rhs: Fn(f64, &[f64], &[f64], &mut [f64]) + Send + Sync + 'static,
        Bc: Fn(&[f64], &[f64], &[f64], &mut [f64]) + Send + Sync + 'static,
    {
        Self {
            dimension,
            parameter_dimension,
            rhs: Arc::new(rhs),
            boundary_residual: Arc::new(boundary_residual),
            rhs_jacobian: None,
            rhs_param_jacobian: None,
            boundary_jacobian: None,
            jacobian_mode: NumericalJacobianMode::FiniteDifference,
        }
    }

    /// Construct a high-performance numerical BVP problem with explicit
    /// pointwise Jacobians.
    pub fn new_with_jacobian<Rhs, Bc, RhsJ, BcJ>(
        dimension: usize,
        parameter_dimension: usize,
        rhs: Rhs,
        boundary_residual: Bc,
        rhs_jacobian: RhsJ,
        boundary_jacobian: BcJ,
    ) -> Self
    where
        Rhs: Fn(f64, &[f64], &[f64], &mut [f64]) + Send + Sync + 'static,
        Bc: Fn(&[f64], &[f64], &[f64], &mut [f64]) + Send + Sync + 'static,
        RhsJ: Fn(f64, &[f64], &[f64]) -> Option<faer_mat> + Send + Sync + 'static,
        BcJ: Fn(&[f64], &[f64], &[f64]) -> Option<(faer_mat, faer_mat, Option<faer_mat>)>
            + Send
            + Sync
            + 'static,
    {
        Self {
            dimension,
            parameter_dimension,
            rhs: Arc::new(rhs),
            boundary_residual: Arc::new(boundary_residual),
            rhs_jacobian: Some(Arc::new(rhs_jacobian)),
            rhs_param_jacobian: None,
            boundary_jacobian: Some(Arc::new(boundary_jacobian)),
            jacobian_mode: NumericalJacobianMode::AnalyticalPointwise,
        }
    }

    /// Attach a pointwise `df/dp` closure to a `new_with_jacobian` problem.
    pub fn with_rhs_param_jacobian<RhsP>(mut self, rhs_param_jacobian: RhsP) -> Self
    where
        RhsP: Fn(f64, &[f64], &[f64]) -> Option<faer_mat> + Send + Sync + 'static,
    {
        self.rhs_param_jacobian = Some(Arc::new(rhs_param_jacobian));
        self
    }
}

impl NumericalBvpProblem for NumericalBvpClosureProblem {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn parameter_dimension(&self) -> usize {
        self.parameter_dimension
    }

    fn rhs(&self, x: f64, y: &[f64], p: &[f64], out: &mut [f64]) {
        (self.rhs)(x, y, p, out)
    }

    fn boundary_residual(&self, ya: &[f64], yb: &[f64], p: &[f64], out: &mut [f64]) {
        (self.boundary_residual)(ya, yb, p, out)
    }

    fn jacobian_mode(&self) -> NumericalJacobianMode {
        self.jacobian_mode
    }

    fn rhs_jacobian(&self, x: f64, y: &[f64], p: &[f64]) -> Option<faer_mat> {
        self.rhs_jacobian
            .as_ref()
            .and_then(|callback| callback(x, y, p))
    }

    fn rhs_param_jacobian(&self, x: f64, y: &[f64], p: &[f64]) -> Option<faer_mat> {
        self.rhs_param_jacobian
            .as_ref()
            .and_then(|callback| callback(x, y, p))
    }

    fn boundary_jacobian(
        &self,
        ya: &[f64],
        yb: &[f64],
        p: &[f64],
    ) -> Option<(faer_mat, faer_mat, Option<faer_mat>)> {
        self.boundary_jacobian
            .as_ref()
            .and_then(|callback| callback(ya, yb, p))
    }
}

/// Pointwise pure-numerical BVP definition.
///
/// The solver core works on vectorized mesh-wide callbacks, but this trait keeps
/// the user-facing API conventional for numerical Rust code: a single point,
/// a state vector, and optional parameters.
pub trait NumericalBvpProblem {
    /// Number of unknowns in the ODE system.
    fn dimension(&self) -> usize;

    /// Number of unknown scalar parameters to be solved together with the state.
    fn parameter_dimension(&self) -> usize {
        0
    }

    /// Evaluate the RHS `f(x, y, p)` at a single mesh point.
    fn rhs(&self, x: f64, y: &[f64], p: &[f64], out: &mut [f64]);

    /// Evaluate the boundary residual vector `bc(ya, yb, p)`.
    ///
    /// The output length must be `dimension() + parameter_dimension()`.
    fn boundary_residual(&self, ya: &[f64], yb: &[f64], p: &[f64], out: &mut [f64]);

    /// Select whether this problem relies on finite differences or supplies
    /// analytical pointwise Jacobians.
    ///
    /// `FiniteDifference` is the default and is the low-friction mode:
    /// implement only `rhs()` and `boundary_residual()`.
    ///
    /// `AnalyticalPointwise` is the high-performance mode:
    /// implement `rhs_jacobian()` and `boundary_jacobian()` as well.
    fn jacobian_mode(&self) -> NumericalJacobianMode {
        NumericalJacobianMode::FiniteDifference
    }

    /// Pointwise Jacobian `df/dy` at one mesh point.
    ///
    /// Return `Some(matrix)` when analytical derivatives are implemented.
    fn rhs_jacobian(&self, _x: f64, _y: &[f64], _p: &[f64]) -> Option<faer_mat> {
        None
    }

    /// Pointwise parameter Jacobian `df/dp` at one mesh point.
    fn rhs_param_jacobian(&self, _x: f64, _y: &[f64], _p: &[f64]) -> Option<faer_mat> {
        None
    }

    /// Boundary Jacobians `(dbc_dya, dbc_dyb, dbc_dp)`.
    fn boundary_jacobian(
        &self,
        _ya: &[f64],
        _yb: &[f64],
        _p: &[f64],
    ) -> Option<(faer_mat, faer_mat, Option<faer_mat>)> {
        None
    }
}

#[derive(Clone)]
pub struct NumericalBvpSolveOptions {
    pub mesh: faer_col,
    pub initial_guess: faer_dense_mat,
    pub parameters: Option<faer_col>,
    pub singular_term: Option<faer_dense_mat>,
    pub tolerance: f64,
    pub max_nodes: usize,
    pub verbose: u8,
    pub bc_tol: Option<f64>,
    pub custom_timer: Option<CustomTimer>,
}

/// User-facing alias for the pure numerical solve options.
pub type BvpSciNumericalOptions = NumericalBvpSolveOptions;

impl NumericalBvpSolveOptions {
    pub fn new(
        mesh: faer_col,
        initial_guess: faer_dense_mat,
        tolerance: f64,
        max_nodes: usize,
    ) -> Self {
        Self {
            mesh,
            initial_guess,
            parameters: None,
            singular_term: None,
            tolerance,
            max_nodes,
            verbose: 0,
            bc_tol: None,
            custom_timer: None,
        }
    }

    pub fn with_parameters(mut self, parameters: Option<faer_col>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Attach a singular-term matrix `S` in the SciPy-style form
    /// `dy/dx = f(x, y, p) + S * y / (x - a)`.
    ///
    /// This matrix is threaded through the solver core and is expected to be
    /// square with the same dimension as the state vector.
    pub fn with_singular_term(mut self, singular_term: Option<faer_dense_mat>) -> Self {
        self.singular_term = singular_term;
        self
    }

    /// Override the residual/Jacobian tolerance after construction.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Override the maximum allowed mesh nodes after construction.
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Override both mesh-refinement controls in one place.
    pub fn with_mesh_refinement(mut self, tolerance: f64, max_nodes: usize) -> Self {
        self.tolerance = tolerance;
        self.max_nodes = max_nodes;
        self
    }

    pub fn with_verbose(mut self, verbose: u8) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_bc_tol(mut self, bc_tol: Option<f64>) -> Self {
        self.bc_tol = bc_tol;
        self
    }

    pub fn with_custom_timer(mut self, custom_timer: Option<CustomTimer>) -> Self {
        self.custom_timer = custom_timer;
        self
    }
}

pub fn solve_numerical_bvp<P: NumericalBvpProblem + 'static>(
    problem: P,
    options: NumericalBvpSolveOptions,
) -> Result<BVPResult, String> {
    let problem = Arc::new(problem);
    let n = problem.dimension();
    let k = problem.parameter_dimension();
    let jacobian_mode = problem.jacobian_mode();

    if options.initial_guess.nrows() != n {
        return Err(format!(
            "initial guess row count {} does not match problem dimension {}",
            options.initial_guess.nrows(),
            n
        ));
    }
    if options.initial_guess.ncols() != options.mesh.nrows() {
        return Err(format!(
            "initial guess column count {} does not match mesh size {}",
            options.initial_guess.ncols(),
            options.mesh.nrows()
        ));
    }

    if let Some(parameters) = options.parameters.as_ref() {
        if parameters.nrows() != k {
            return Err(format!(
                "parameter vector size {} does not match problem parameter dimension {}",
                parameters.nrows(),
                k
            ));
        }
    } else if k > 0 {
        return Err(format!(
            "problem expects {} parameters but no initial parameter vector was provided",
            k
        ));
    }

    if matches!(jacobian_mode, NumericalJacobianMode::AnalyticalPointwise) {
        let probe_y = vec![0.0; n];
        let probe_p = vec![0.0; k];
        if problem
            .rhs_jacobian(options.mesh[0], &probe_y, &probe_p)
            .is_none()
        {
            return Err(
                "NumericalJacobianMode::AnalyticalPointwise requires rhs_jacobian()".to_string(),
            );
        }
        if problem
            .boundary_jacobian(&probe_y, &probe_y, &probe_p)
            .is_none()
        {
            return Err(
                "NumericalJacobianMode::AnalyticalPointwise requires boundary_jacobian()"
                    .to_string(),
            );
        }
    }

    let problem_for_fun = Arc::clone(&problem);
    let fun: Box<ODEFunction> = Box::new(
        move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> faer_dense_mat {
            let point_count = x.nrows();
            let mut result = faer_dense_mat::zeros(n, point_count);
            let p_values = p.iter().copied().collect::<Vec<_>>();
            let mut y_point = vec![0.0; n];
            let mut rhs_out = vec![0.0; n];

            for col in 0..point_count {
                for row in 0..n {
                    y_point[row] = *y.get(row, col);
                }
                problem_for_fun.rhs(x[col], &y_point, &p_values, &mut rhs_out);
                for row in 0..n {
                    *result.get_mut(row, col) = rhs_out[row];
                }
            }

            result
        },
    );

    let problem_for_bc = Arc::clone(&problem);
    let bc: Box<BCFunction> = Box::new(
        move |ya: &faer_col, yb: &faer_col, p: &faer_col| -> faer_col {
            let p_values = p.iter().copied().collect::<Vec<_>>();
            let ya_values = ya.iter().copied().collect::<Vec<_>>();
            let yb_values = yb.iter().copied().collect::<Vec<_>>();
            let mut residual = vec![0.0; n + k];
            problem_for_bc.boundary_residual(&ya_values, &yb_values, &p_values, &mut residual);
            faer_col::from_fn(n + k, |i| residual[i])
        },
    );

    let problem_for_fun_jac = Arc::clone(&problem);
    let fun_jacobian: Box<ODEJacobian> =
        Box::new(
            move |x: &faer_col,
                  y: &faer_dense_mat,
                  p: &faer_col|
                  -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
                let point_count = x.nrows();
                let p_values = p.iter().copied().collect::<Vec<_>>();
                let mut y_point = vec![0.0; n];
                let mut jacobians = Vec::with_capacity(point_count);
                let mut param_jacobians = if k > 0 {
                    Some(Vec::with_capacity(point_count))
                } else {
                    None
                };

                for col in 0..point_count {
                    for row in 0..n {
                        y_point[row] = *y.get(row, col);
                    }

                    let jacobian = problem_for_fun_jac
                .rhs_jacobian(x[col], &y_point, &p_values)
                .expect("AnalyticalPointwise mode requires rhs_jacobian() to return Some(matrix)");
                    jacobians.push(jacobian);

                    if let Some(param_store) = param_jacobians.as_mut() {
                        let param_jacobian =
                            problem_for_fun_jac.rhs_param_jacobian(x[col], &y_point, &p_values);
                        match param_jacobian {
                            Some(matrix) => param_store.push(matrix),
                            None => {
                                let empty_triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
                                param_store.push(
                                    faer_mat::try_new_from_triplets(n, k, &empty_triplets).expect(
                                        "empty sparse parameter Jacobian should be constructible",
                                    ),
                                );
                            }
                        }
                    }
                }

                (jacobians, param_jacobians)
            },
        );

    let problem_for_bc_jac = Arc::clone(&problem);
    let bc_jacobian: Box<BCJacobian> = Box::new(
        move |ya: &faer_col,
              yb: &faer_col,
              p: &faer_col|
              -> (faer_mat, faer_mat, Option<faer_mat>) {
            let p_values = p.iter().copied().collect::<Vec<_>>();
            let ya_values = ya.iter().copied().collect::<Vec<_>>();
            let yb_values = yb.iter().copied().collect::<Vec<_>>();
            problem_for_bc_jac
            .boundary_jacobian(&ya_values, &yb_values, &p_values)
            .expect(
                "AnalyticalPointwise mode requires boundary_jacobian() to return Some(matrices)",
            )
        },
    );

    let fun_jacobian_ref = if matches!(jacobian_mode, NumericalJacobianMode::AnalyticalPointwise) {
        Some(fun_jacobian.as_ref())
    } else {
        None
    };
    let bc_jacobian_ref = if matches!(jacobian_mode, NumericalJacobianMode::AnalyticalPointwise) {
        Some(bc_jacobian.as_ref())
    } else {
        None
    };

    solve_bvp(
        fun.as_ref(),
        bc.as_ref(),
        options.mesh,
        options.initial_guess,
        options.parameters,
        options.singular_term,
        fun_jacobian_ref,
        bc_jacobian_ref,
        options.tolerance,
        options.max_nodes,
        options.verbose,
        options.bc_tol,
        options.custom_timer,
    )
}

#[derive(Clone)]
struct ForcedJacobianModeProblem<P> {
    inner: P,
    mode: NumericalJacobianMode,
}

impl<P: NumericalBvpProblem> NumericalBvpProblem for ForcedJacobianModeProblem<P> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn parameter_dimension(&self) -> usize {
        self.inner.parameter_dimension()
    }

    fn rhs(&self, x: f64, y: &[f64], p: &[f64], out: &mut [f64]) {
        self.inner.rhs(x, y, p, out)
    }

    fn boundary_residual(&self, ya: &[f64], yb: &[f64], p: &[f64], out: &mut [f64]) {
        self.inner.boundary_residual(ya, yb, p, out)
    }

    fn jacobian_mode(&self) -> NumericalJacobianMode {
        self.mode
    }

    fn rhs_jacobian(&self, x: f64, y: &[f64], p: &[f64]) -> Option<faer_mat> {
        self.inner.rhs_jacobian(x, y, p)
    }

    fn rhs_param_jacobian(&self, x: f64, y: &[f64], p: &[f64]) -> Option<faer_mat> {
        self.inner.rhs_param_jacobian(x, y, p)
    }

    fn boundary_jacobian(
        &self,
        ya: &[f64],
        yb: &[f64],
        p: &[f64],
    ) -> Option<(faer_mat, faer_mat, Option<faer_mat>)> {
        self.inner.boundary_jacobian(ya, yb, p)
    }
}

/// Explicit low-friction numerical route: force finite-difference Jacobians.
pub fn solve_numerical_bvp_fd<P: NumericalBvpProblem + 'static>(
    problem: P,
    options: NumericalBvpSolveOptions,
) -> Result<BVPResult, String> {
    solve_numerical_bvp(
        ForcedJacobianModeProblem {
            inner: problem,
            mode: NumericalJacobianMode::FiniteDifference,
        },
        options,
    )
}

/// Explicit high-performance numerical route: force analytical pointwise Jacobians.
pub fn solve_numerical_bvp_with_jacobian<P: NumericalBvpProblem + 'static>(
    problem: P,
    options: NumericalBvpSolveOptions,
) -> Result<BVPResult, String> {
    solve_numerical_bvp(
        ForcedJacobianModeProblem {
            inner: problem,
            mode: NumericalJacobianMode::AnalyticalPointwise,
        },
        options,
    )
}
