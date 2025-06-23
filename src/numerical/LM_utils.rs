use super::NR::solve_linear_system;

use log::{error, info};
use nalgebra::{DMatrix, DVector};
use std::fmt::Display;
use strum_macros::Display;
#[derive(Debug, Clone, PartialEq, Display)]

pub enum UpdateMethod {
    Nielsen,
    Gsl,
}
///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////SCALING METHODS///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///  Scaling methods for trust region algorithms (GSL scaling.c)
#[derive(Debug, Clone, PartialEq, Display)]
pub enum ScalingMethod {
    /// Levenberg scaling: D = I (identity matrix)
    Levenberg,
    /// Marquardt scaling: D[j,j] = ||J[:,j]||_2 (column norms)
    Marquardt,

    More,
}

/// Scaling operations for trust region methods
pub struct TrustRegionScaling;

impl TrustRegionScaling {
    /// Initialize diagonal scaling matrix based on Jacobian and method
    pub fn init_scaling(jacobian: &DMatrix<f64>, method: &ScalingMethod) -> DVector<f64> {
        match method {
            ScalingMethod::Levenberg => Self::init_levenberg_scaling(jacobian),
            ScalingMethod::Marquardt => Self::init_marquardt_scaling(jacobian),
            ScalingMethod::More => Self::init_more_scaling(jacobian),
        }
    }

    /// Update diagonal scaling matrix based on current Jacobian (source: GSL file scaling.c)
    pub fn update_scaling(
        jacobian: &DMatrix<f64>,
        method: &ScalingMethod,
        current_diag: &mut DVector<f64>,
    ) {
        match method {
            ScalingMethod::Levenberg => {
                // Levenberg scaling never changes - always identity
                // No update needed
            }
            ScalingMethod::Marquardt => {
                Self::update_marquardt_scaling(jacobian, current_diag);
            }
            ScalingMethod::More => {
                Self::update_more_scaling(jacobian, current_diag);
            }
        }
    }

    ////////////////////////LEVENBERG//////////////////////////////////////////////////////////////////
    /// Levenberg scaling: D = I (all diagonal elements = 1.0)  (source: GSL file scaling.c)
    fn init_levenberg_scaling(jacobian: &DMatrix<f64>) -> DVector<f64> {
        let n_params = jacobian.ncols();
        DVector::from_element(n_params, 1.0)
    }
    ///////////////////MARQUARDT//////////////////////////////////////////////////////////////////
    /// Marquardt scaling: D[j,j] = ||J[:,j]||_2  (source: GSL file scaling.c)
    fn init_marquardt_scaling(jacobian: &DMatrix<f64>) -> DVector<f64> {
        let n_params = jacobian.ncols();
        let mut diag = DVector::zeros(n_params);
        Self::update_marquardt_scaling(jacobian, &mut diag);
        info!("\n Diagonal scaling vector created!: {}", diag);
        diag
    }
    /// Update Marquardt scaling with current Jacobian (source: GSL file scaling.c)
    fn update_marquardt_scaling(jacobian: &DMatrix<f64>, diag: &mut DVector<f64>) {
        let n_params = jacobian.ncols();

        for j in 0..n_params {
            // Get j-th column of Jacobian
            let column = jacobian.column(j);

            // Compute Euclidean norm of column
            let norm = column.norm();
            //info!("norm: {}, of column: {}", norm, j);
            // Handle degenerate case
            let scaling_value = if norm < 1e-12 { 1.0 } else { norm };

            diag[j] = scaling_value;
        }
    }
    /////////////////////////////////////////////////////MORE//////////////////////////////////////////////////////////////////
    /// More scaling: D[j,j] = max(||J[:,j]||_2, D[j,j])  (source: GSL file scaling.c)
    fn init_more_scaling(jacobian: &DMatrix<f64>) -> DVector<f64> {
        Self::init_marquardt_scaling(jacobian)
    }
    /// Update More scaling with current Jacobian (source: GSL file scaling.c)
    ///   this method described in More Eq. 6.3 p. 111-112
    fn update_more_scaling(jacobian: &DMatrix<f64>, diag: &mut DVector<f64>) {
        let n_params = jacobian.ncols();

        for j in 0..n_params {
            // Get j-th column of Jacobian
            let column = jacobian.column(j);

            // Compute Euclidean norm of column
            let norm = column.norm();
            //info!("norm: {}, of column: {}", norm, j);
            // Handle degenerate case
            let scaling_value = if norm < 1e-12 { 1.0 } else { norm };

            diag[j] = scaling_value.max(diag[j]);
        }
    }

    /////////////////////////////////////REGULARIZATION////////////////////////////////////////////////////////
    /// Create scaled regularization matrix based on method
    pub fn create_scaled_regularization(
        mu: f64,
        diag: &DVector<f64>,
        method: &ScalingMethod,
    ) -> DMatrix<f64> {
        let n = diag.len();
        let mut regularization = DMatrix::zeros(n, n);

        match method {
            ScalingMethod::Levenberg => {
                // Levenberg: μ * I (identity matrix)
                for i in 0..n {
                    regularization[(i, i)] = mu;
                }
            }
            ScalingMethod::Marquardt | ScalingMethod::More => {
                // Marquardt: μ * D^T * D = μ * diag(d_i^2)
                for i in 0..n {
                    regularization[(i, i)] = mu * diag[i] * diag[i];
                }
            }
        }
        regularization
    }
    ////////////////////////////////////MISC////////////////////////////////////////////////////////////////
    /// Store scaling diagonal in scales_vec (DVector)
    pub fn store_scaling_in_vector(
        scales_vec: &mut DVector<f64>,
        diag: &DVector<f64>,
        iteration: usize,
    ) {
        // Resize scales_vec if necessary
        if scales_vec.len() != diag.len() {
            if iteration != 0 {
                error!("scales_vec is not properly initialized")
            };
            *scales_vec = DVector::zeros(diag.len());
        }

        // Copy scaling values
        scales_vec.copy_from(diag);
    }

    /// Extract scaling diagonal from scales_vec (DVector)
    pub fn extract_scaling_from_vector(scales_vec: &DVector<f64>, n_params: usize) -> DVector<f64> {
        if scales_vec.len() != n_params || scales_vec.len() == 0 {
            error!("scales_vec is not properly initialized");
            // Return identity scaling if not properly initialized
            return DVector::from_element(n_params, 1.0);
        }

        // Return copy of scales_vec
        scales_vec.clone()
    }
}
/////////////////////END OF SCALING////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////TRUST REGION SUBPROBLEM/////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, PartialEq)]
pub enum SubproblemMethod {
    Dogleg_simple,
    Dogleg_gsl,
    Direct,
}
impl Display for SubproblemMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubproblemMethod::Dogleg_simple => write!(f, "Dogleg_simple"),
            SubproblemMethod::Dogleg_gsl => write!(f, "Dogleg_gsl"),
            SubproblemMethod::Direct => write!(f, "Direct"),
        }
    }
}
pub struct TrustRegionSubproblem;
impl TrustRegionSubproblem {
    /// Solve trust region subproblem using specified method
    /// The Levenberg-Marquardt method does NOT use delta directly in solving the subproblem. Bit dogleg methods do.
    pub fn solve(
        jacobian: &DMatrix<f64>,
        residuals: &DVector<f64>,
        gradient: &DVector<f64>,
        scaling: &DVector<f64>,
        mu: f64,
        _delta: f64,
        linear_solver: &str,
        scaling_method: &ScalingMethod,
        method: SubproblemMethod,
    ) -> DVector<f64> {
        match method {
            SubproblemMethod::Direct => Self::direct_method(
                jacobian,
                residuals,
                gradient,
                scaling,
                mu,
                linear_solver,
                scaling_method,
            ),
            SubproblemMethod::Dogleg_simple => {
                unimplemented!()
                // Self::dogleg_simple_method(jacobian, residuals, gradient, scaling, delta, linear_solver, scaling_method)
            }
            SubproblemMethod::Dogleg_gsl => {
                unimplemented!()
                // Self::dogleg_gsl_method(jacobian, residuals, gradient, scaling, delta, linear_solver, scaling_method)
            }
        }
    }
    /// Direct method: Solve regularized normal equations
    /// This is the standard Levenberg-Marquardt approach (source: Nielsen, p. 122)
    ///
    fn direct_method(
        jacobian: &DMatrix<f64>,
        _residuals: &DVector<f64>,
        gradient: &DVector<f64>,
        scaling: &DVector<f64>,
        mu: f64,
        linear_solver: &str,
        scaling_method: &ScalingMethod,
    ) -> DVector<f64> {
        info!("Solving trust region subproblem using Direct method");

        // Compute J^T * J
        let JtJ = jacobian.transpose() * jacobian;

        // Create scaled regularization matrix (method-specific)
        let regularization =
            TrustRegionScaling::create_scaled_regularization(mu, scaling, scaling_method);

        // Form augmented system: A = J^T*J + regularization
        let A = &JtJ + &regularization;

        // Solve: A * p = -gradient
        let p = solve_linear_system(linear_solver.to_string(), &A, gradient)
            .expect("Failed to solve linear system in direct method");

        info!("Direct method step norm: {}", p.norm());
        p
    }
    /// Simple dogleg method: Combines Gauss-Newton and steepest descent directions
    #[allow(dead_code)]
    fn dogleg_simple_method(
        jacobian: &DMatrix<f64>,
        _residuals: &DVector<f64>,
        gradient: &DVector<f64>,
        scaling: &DVector<f64>,
        delta: f64,
        linear_solver: &str,
        scaling_method: &ScalingMethod,
    ) -> DVector<f64> {
        info!("Solving trust region subproblem using Simple Dogleg method");

        // 1. Compute Gauss-Newton step (unregularized)
        let JtJ = jacobian.transpose() * jacobian;
        let p_gn =
            solve_linear_system(linear_solver.to_string(), &JtJ, gradient).unwrap_or_else(|_| {
                // Fallback to regularized system if singular
                let regularization =
                    TrustRegionScaling::create_scaled_regularization(1e-6, scaling, scaling_method);
                let A_reg = &JtJ + &regularization;
                solve_linear_system(linear_solver.to_string(), &A_reg, gradient)
                    .expect("Failed to solve regularized system")
            });

        // 2. Check if Gauss-Newton step is within trust region
        let scaled_gn_norm = scaled_norm_common(scaling, &p_gn);

        if scaled_gn_norm <= delta {
            info!("Gauss-Newton step accepted, norm: {}", scaled_gn_norm);
            return p_gn;
        }

        // 3. Compute steepest descent step
        let grad_norm_sq = gradient.norm_squared();
        let Jg = jacobian * gradient;
        let alpha = grad_norm_sq / Jg.norm_squared();
        let p_sd = -alpha * gradient;

        // 4. Find dogleg path intersection with trust region boundary
        let scaled_sd_norm = scaled_norm_common(scaling, &p_sd);

        if scaled_sd_norm >= delta {
            // Use scaled steepest descent to boundary
            let tau = delta / scaled_sd_norm;
            info!("Using steepest descent step with tau: {}", tau);
            return tau * p_sd;
        }

        // 5. Find intersection of dogleg path with trust region
        // Solve: ||D * (p_sd + t * (p_gn - p_sd))||^2 = delta^2
        let diff = &p_gn - &p_sd;
        let scaled_diff = scaling.component_mul(&diff);
        let scaled_sd = scaling.component_mul(&p_sd);

        let a = scaled_diff.norm_squared();
        let b = 2.0 * scaled_sd.dot(&scaled_diff);
        let c = scaled_sd.norm_squared() - delta * delta;

        let discriminant = b * b - 4.0 * a * c;
        let t = if discriminant >= 0.0 {
            (-b + discriminant.sqrt()) / (2.0 * a)
        } else {
            1.0 // Fallback
        };

        let t = t.clamp(0.0, 1.0);
        let p_dogleg = &p_sd + t * diff;

        info!(
            "Dogleg step with t: {}, norm: {}",
            t,
            scaled_norm_common(scaling, &p_dogleg)
        );
        p_dogleg
    }
}
/////////////////////////END OF TRUST REGION SUBPROBLEM SOLVERS////////////////////////////////////////////////

////////////////////////////////REDUCTION RAIO//////////////////////////////////////////////////////////////
#[derive(Debug, Clone, PartialEq, Display)]
pub enum ReductionRatio {
    Nielsen, // Nielsen, p. 120
    More,    // More, p. 108
}

pub struct ReductionRatioSolver {}
impl ReductionRatioSolver {
    pub fn solve_reduction_ratio(
        Jy: &DMatrix<f64>,
        Fy: &DVector<f64>,
        Fy_new: &DVector<f64>,

        scaling: &DVector<f64>,
        p: &DVector<f64>,
        mu: f64,
        reduction_ratio_method: ReductionRatio,
    ) -> f64 {
        match reduction_ratio_method {
            ReductionRatio::Nielsen => {
                // Compute the reduction ratio using Nielsen's method
                let rr = Self::nielsen_reduction_ratio(Jy, Fy, Fy_new, p);
                rr
            }
            ReductionRatio::More => {
                // Compute the reduction ratio using More's method
                let rr = Self::gsl_reduction_ratio(Jy, Fy, Fy_new, p, scaling, mu);
                rr
            }
        }
    }
    /// Compute the reduction ratio using Nielsen's method (Nielsen, p. 120)
    fn nielsen_reduction_ratio(
        Jy: &DMatrix<f64>,
        Fy: &DVector<f64>,
        Fy_new: &DVector<f64>,
        p: &DVector<f64>,
    ) -> f64 {
        let actual_reduction = Fy.norm_squared() - Fy_new.norm_squared();

        // Method-specific predicted reduction calculation
        let predicted_reduction = Fy.norm_squared() - (Fy + Jy * p).norm_squared(); // ||F||² - ||F + J*p||²;

        let rho = if predicted_reduction.abs() > 1e-12 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        info!(
            "rho = {}, actual_reduction = {}, predicted_reduction = {}",
            rho, actual_reduction, predicted_reduction
        );

        rho
    }
    /// Compute the reduction ratio using More's method (More, p. 108)
    /// This method is implemented in the GSL library
    fn gsl_reduction_ratio(
        Jy: &DMatrix<f64>,
        Fy: &DVector<f64>,
        Fy_new: &DVector<f64>,
        p: &DVector<f64>,

        scaling: &DVector<f64>,
        mu: f64,
    ) -> f64 {
        // Actual reduction
        let norm_f = Fy.norm_squared();
        let norm_f_trial = Fy_new.norm_squared();

        if norm_f_trial >= norm_f {
            info!("Cost increased, rejecting step");
            return -1.0; // reject immediately if cost increased
        }

        // Actual reduction
        let actual_reduction = 1.0 - (norm_f_trial / norm_f).powf(2.0);

        // Predicted reduction (from quadratic model)
        // Uses method-specific calculation from trust region subproblem
        let predicted_reduction = Self::predicted_reduction(Fy, Jy, p, scaling, mu);

        if predicted_reduction > 0.0 {
            let rho = actual_reduction / predicted_reduction;
            info!(
                "rho = {}, actual_reduction = {}, predicted_reduction = {}",
                rho, actual_reduction, predicted_reduction
            );
            return rho;
        } else {
            info!(
                "rho = {}, actual_reduction = {}, predicted_reduction = {}",
                -1.0, actual_reduction, predicted_reduction
            );
            return -1.0; // reject immediately if predicted reduction is negative
        }
    }
    fn predicted_reduction(
        Fy: &DVector<f64>,
        Jy: &DMatrix<f64>,
        p: &DVector<f64>,

        scaling: &DVector<f64>,
        mu: f64,
    ) -> f64 {
        // source More p. 108-109: Moré's formula (Eq 4.4):
        // pred = ||J*p||²/||f||² + 2*mu*||D*p||²/||f||²
        // where p is the velocity component of the step
        let norm_f = Fy.norm();
        let norm_Jp = (Jy * p).norm();
        // Compute scaled norm ||D * velocity||
        let norm_Dp = (scaling.component_mul(p)).norm();

        // Normalize by residual norm
        let u = norm_Jp / norm_f;
        let v = norm_Dp / norm_f;
        // Moré's formula (Eq 4.4)
        let predicted_reduction = u.powf(2.0) + 2.0 * mu * v.powf(2.0);
        predicted_reduction
    }
}

//////////////////////////////CONVERGENCE CRITERIA///////////////////////////////////////////////////////////
/// Convergence test for Levenberg-Marquardt algorithm
/// Based on GSL implementation in convergence.c
///
/// Returns: (converged: bool, info: i32)
/// - converged: true if algorithm has converged
/// - info: convergence type (1=parameter change, 2=gradient, 0=no convergence)
#[derive(Debug, Clone, PartialEq, Display)]
pub enum ConvergenceCriteria {
    GSL,
    SimpleScaled,
}
pub struct ConvergenceCriteriaSolver {}
impl ConvergenceCriteriaSolver {
    pub fn test_convergence(
        x: &DVector<f64>,    // Current parameter vector
        dx: &DVector<f64>,   // Parameter step vector
        g: &DVector<f64>,    // Gradient vector J^T * f
        f: &DVector<f64>,    // Residual vector
        diag: &DVector<f64>, // Diagonal matrix of parameter scaling factors
        xtol: f64,           // Parameter change tolerance
        gtol: Option<f64>,   // Gradient tolerance
        ftol: Option<f64>,   // Function change tolerance (currently unused)
        convergence_criteria: ConvergenceCriteria,
    ) -> (bool, i32) {
        match convergence_criteria {
            ConvergenceCriteria::GSL => test_convergence_gsl(x, dx, g, f, xtol, gtol, ftol),
            ConvergenceCriteria::SimpleScaled => test_convergence_simple_scaled(dx, diag, xtol),
        }
    }
}

/// Simple scaled convergence test
fn test_convergence_simple_scaled(
    dx: &DVector<f64>,
    diag: &DVector<f64>,
    xtol: f64,
) -> (bool, i32) {
    let scaled_norm = scaled_norm_common(diag, dx);
    let converged = scaled_norm < xtol;
    let info = if converged { 1 } else { 0 };
    (converged, info)
}

/// Compute scaled norm: ||D * v||_2
pub fn scaled_norm_common(diag: &DVector<f64>, vector: &DVector<f64>) -> f64 {
    let mut sum_squares = 0.0;

    for i in 0..vector.len() {
        let scaled_component = vector[i] * diag[i];
        sum_squares += scaled_component * scaled_component;
    }

    let scaled_norm = sum_squares.sqrt();
    info!("scaled norm ||step*D|| {}", scaled_norm);
    scaled_norm
}
/////////////////////////////GSL CONVERGENCE CRITERIA///////////////////////////////////////////////////////////
pub fn test_convergence_gsl(
    x: &DVector<f64>,  // Current parameter vector
    dx: &DVector<f64>, // Parameter step vector
    g: &DVector<f64>,  // Gradient vector J^T * f
    f: &DVector<f64>,  // Residual vector
    xtol: f64,         // Parameter change tolerance
    gtol: Option<f64>, // Gradient tolerance
    ftol: Option<f64>, // Function change tolerance (currently unused)
) -> (bool, i32) {
    let gtol = gtol.unwrap_or(1e-6);
    let ftol = ftol.unwrap_or(1e-6);
    info!("Checking convergence conditions:");
    info!("  xtol = {}, gtol = {}, ftol = {}", xtol, gtol, ftol);
    info!(
        "  ||dx|| = {}, ||g|| = {}, ||f|| = {}",
        dx.norm(),
        g.norm(),
        f.norm()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 1: PARAMETER CHANGE TEST (xtol)
    // Check if parameter changes are negligibly small
    // ═══════════════════════════════════════════════════════════════════════

    if test_parameter_change(dx, x, xtol) {
        info!("CONVERGED: Parameter change test passed");
        info!("  ||dx|| components are all < xtol = {}", xtol);
        return (true, 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 2: GRADIENT TEST (gtol)
    // Check if scaled gradient norm is small (near optimal point)
    // ═══════════════════════════════════════════════════════════════════════

    // Compute scaled infinity norm of gradient
    let gnorm = scaled_infnorm(x, g);

    // Compute current objective function value
    let fnorm = f.norm();
    let phi = 0.5 * fnorm * fnorm; // Current objective: Φ = ½||f||²

    // Gradient convergence threshold
    let gradient_threshold = gtol * phi.max(1.0);

    info!("Gradient test:");
    info!("  gnorm = {}", gnorm);
    info!("  phi = 0.5 * ||f||² = {}", phi);
    info!("  threshold = gtol * max(phi, 1) = {}", gradient_threshold);

    if gnorm <= gradient_threshold {
        info!("CONVERGED: Gradient test passed");
        info!(
            "  Scaled gradient norm {} <= threshold {}",
            gnorm, gradient_threshold
        );
        return (true, 2);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TEST 3: FUNCTION CHANGE TEST (ftol) - CURRENTLY DISABLED
    // This test is disabled in GSL, but we include the structure for completeness
    // ═══════════════════════════════════════════════════════════════════════

    // Note: In GSL this test is commented out (#if 0)
    // If you want to enable it in the future, you would need to track
    // the previous function norm and compare changes
    let _ftol_unused = ftol; // Suppress unused parameter warning

    // ═══════════════════════════════════════════════════════════════════════
    // NO CONVERGENCE
    // ═══════════════════════════════════════════════════════════════════════

    info!("No convergence detected:");
    info!("  Parameter change test: FAILED");
    info!(
        "  Gradient test: gnorm = {} > threshold = {}",
        gnorm, gradient_threshold
    );
    info!("  Function change test: DISABLED (following GSL)");

    (false, 0)
}
/// Test if parameter changes are small enough for convergence
/// Based on GSL's test_delta function
fn test_parameter_change(dx: &DVector<f64>, x: &DVector<f64>, xtol: f64) -> bool {
    let n = dx.len();
    let xtol_squared = xtol * xtol;

    info!("\n Parameter change test (component-wise):");

    for i in 0..n {
        let dx_i = dx[i];
        let x_i = x[i];

        // Absolute tolerance test
        if dx_i.abs() > xtol {
            info!(
                "  Component {}: |dx| = {} > xtol = {} (FAIL absolute)",
                i,
                dx_i.abs(),
                xtol
            );
            return false;
        }

        // Relative tolerance test
        if dx_i.abs() > xtol_squared * x_i.abs() {
            info!(
                "  Component {}: |dx| = {} > xtol² * |x| = {} (FAIL relative)",
                i,
                dx_i.abs(),
                xtol_squared * x_i.abs()
            );
            return false;
        }

        info!("  Component {}: |dx| = {} (PASS both tests)", i, dx_i.abs());
    }

    info!("All components passed parameter change test");
    true
}

/// Compute scaled infinity norm of gradient
/// Based on GSL's scaled_infnorm function
///
/// Formula: max_i(|g_i| * max(|x_i|, 1))
/// This scaling accounts for the magnitude of parameters
fn scaled_infnorm(x: &DVector<f64>, g: &DVector<f64>) -> f64 {
    let n = x.len();
    let mut max_scaled_grad: f64 = 0.0;

    info!("\n Computing scaled gradient norm:");

    for i in 0..n {
        let x_i = x[i];
        let g_i = g[i];

        // Scale factor: max(|x_i|, 1.0)
        let scale_factor = x_i.abs().max(1.0);

        // Scaled gradient component: |g_i| * scale_factor
        let scaled_grad_i: f64 = g_i.abs() * scale_factor;

        // Track maximum
        max_scaled_grad = max_scaled_grad.max(scaled_grad_i);

        info!(
            "  Component {}: |g| = {}, scale = {}, scaled = {}",
            i,
            g_i.abs(),
            scale_factor,
            scaled_grad_i
        );
    }

    info!("Maximum scaled gradient component: {}", max_scaled_grad);
    max_scaled_grad
}

/// Convergence information codes (matching GSL)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceInfo {
    NoConvergence = 0,
    ParameterChange = 1,
    GradientSmall = 2,
    FunctionChange = 3, // Reserved for future use
}

impl From<i32> for ConvergenceInfo {
    fn from(code: i32) -> Self {
        match code {
            1 => ConvergenceInfo::ParameterChange,
            2 => ConvergenceInfo::GradientSmall,
            3 => ConvergenceInfo::FunctionChange,
            _ => ConvergenceInfo::NoConvergence,
        }
    }
}

impl std::fmt::Display for ConvergenceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvergenceInfo::NoConvergence => write!(f, "No convergence"),
            ConvergenceInfo::ParameterChange => write!(f, "Converged: parameter change small"),
            ConvergenceInfo::GradientSmall => write!(f, "Converged: gradient small"),
            ConvergenceInfo::FunctionChange => write!(f, "Converged: function change small"),
        }
    }
}

/////
//////////////////////////////////END OF CONVERGENCE CHECKING////////////////////////////////////////////
/// Compute scaled gradient norm for optimality check
#[allow(dead_code)]
fn compute_scaled_gradient_norm(
    jacobian: &DMatrix<f64>,
    residuals: &DVector<f64>,
    residuals_norm: f64,
) -> f64 {
    let mut max_scaled_grad: f64 = 0.0;

    for j in 0..jacobian.ncols() {
        let col = jacobian.column(j);
        let col_norm = col.norm();
        if col_norm > 0.0 && residuals_norm > 0.0 {
            let grad_component = col.dot(residuals).abs();
            let scaled_grad: f64 = grad_component / (col_norm * residuals_norm);
            max_scaled_grad = max_scaled_grad.max(scaled_grad);
        }
    }

    max_scaled_grad
}
/// Compute scaled gradient norm for optimality check for More method
pub fn compute_scaled_gradient_norm2(
    jacobian: &DMatrix<f64>,  // Full Jacobian matrix J
    residuals: &DVector<f64>, // Residual vector r
) -> Option<f64> {
    let residual_norm = residuals.norm();
    if residual_norm == 0.0 {
        return Some(0.0);
    }

    let mut max_scaled_gradient: f64 = 0.0;
    let n = jacobian.ncols();

    // For each column of the Jacobian
    for j in 0..n {
        let jacobian_column = jacobian.column(j);
        let mut column_norm = jacobian_column.norm();

        // Skip zero columns
        if column_norm.abs() < 1e-40 {
            column_norm = 1.0;
        }

        // Compute (J^T * r)_j = dot product of j-th column with residuals
        let gradient_component = jacobian_column.dot(residuals);

        // Scale by column norm and residual norm
        let scaled_gradient = gradient_component.abs() / (column_norm * residual_norm);

        // Check for numerical issues
        if scaled_gradient.is_nan() {
            return None;
        }

        // Track maximum
        max_scaled_gradient = max_scaled_gradient.max(scaled_gradient);
    }

    Some(max_scaled_gradient)
}
