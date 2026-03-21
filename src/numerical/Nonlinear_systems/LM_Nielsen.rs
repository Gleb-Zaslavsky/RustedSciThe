use log::{info, warn};
use nalgebra::{DMatrix, DVector};

use crate::numerical::Nonlinear_systems::engine::{
    IterationState, NonlinearMethod, RuntimeDiagnostics, SolveOptions, StepOutcome, scaled_norm,
    scaling_vector, solve_linear_system,
};
use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
use crate::numerical::Nonlinear_systems::problem::JacobianProvider;
const ONE_THIRD: f64 = 1.0 / 3.0;
use crate::numerical::Nonlinear_systems::LM_utils::*;
/*
================================================================================
LEVENBERG-MARQUARDT ALGORITHM WITH TRUST REGION - RUST IMPLEMENTATION
================================================================================

This module implements the Levenberg-Marquardt (LM) algorithm for nonlinear
least squares optimization with trust region enhancements, following:

1. Nielsen & Madsen: "Introduction to Optimization and Data Fitting" (2010)
2. GSL (GNU Scientific Library) multifit_nlinear implementation
3. Moré: "The Levenberg-Marquardt Algorithm: Implementation and Theory" (1978)

PROBLEM FORMULATION:
    Minimize: F(x) = 0.5 * ||f(x)||²
    Where: f(x) is a vector of residuals, x is parameter vector

ALGORITHM OVERVIEW:
    LM combines Gauss-Newton (fast near solution) with gradient descent
    (stable far from solution) using adaptive damping parameter μ.

    At each iteration, solve: (J^T*J + μ*D^T*D) * p = -J^T*f
    Where: J=Jacobian, D=scaling matrix, p=step vector, μ=damping parameter
================================================================================
*/
/*

MAIN ALGORITHM STRUCTURE:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OUTER LOOP (Main Iterations)                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                     INNER LOOP (Step Attempts)                         ││
│  │  - Try step with current parameters (μ, δ)                             ││
│  │  - If step good (ρ > 0): accept, update x, exit inner loop             ││
│  │  │  If step bad (ρ ≤ 0): reject, increase μ, try again                 ││
│  │  - Max 15 consecutive rejections before giving up                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  Continue until convergence or max iterations                               │
└─────────────────────────────────────────────────────────────────────────────┘

/*
================================================================================
FUNCTION: step_trust_region_Nielsen()
PURPOSE: Perform one iteration of LM algorithm with Nielsen's parameter updates
RETURNS: (status_code, new_x_vector)
    status_code: 1=converged, 0=continue, -1=error, -2=no_progress
================================================================================
*/

FUNCTION step_trust_region_Nielsen():

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: INITIALIZATION AND SETUP
    // ═══════════════════════════════════════════════════════════════════════

    // Initialize parameters on first iteration only
    IF iteration == 0 THEN
        CALL preloop_trust_region_Nielsen()
        // This initializes:
        // - μ₀ (damping parameter)
        // - δ₀ (trust region radius)
        // - ν₀ (Nielsen parameter)
        // - D₀ (scaling matrix)
    END IF

    // Get algorithm configuration
    scaling_method = GET_SCALING_METHOD()  // Levenberg, Marquardt, or Moré
    linear_solver = GET_LINEAR_SOLVER()    // LU, QR, Cholesky, etc.
    bounds = GET_PARAMETER_BOUNDS()        // Optional box constraints

    // Get current algorithm parameters
    μ = GET_DAMPING_PARAMETER()           // Current damping
    ν = GET_NIELSEN_PARAMETER()           // Nielsen update factor
    δ = GET_TRUST_REGION_RADIUS()         // Trust region size

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: FUNCTION AND JACOBIAN EVALUATION
    // ═══════════════════════════════════════════════════════════════════════

    x = GET_CURRENT_POINT()               // Current parameter vector
    f = EVALUATE_FUNCTION(x)              // Residual vector f(x)
    J = EVALUATE_JACOBIAN(x)              // Jacobian matrix J(x)

    // Update scaling matrix D based on current Jacobian
    D = UPDATE_SCALING_MATRIX(J, scaling_method)
    // Levenberg:  D = I (identity)
    // Marquardt:  D[i,i] = ||J[:,i]||₂ (column norms)
    // Moré:       D[i,i] = max(D[i,i], ||J[:,i]||₂) (cumulative max)

    // Compute gradient: g = -J^T * f
    g = -TRANSPOSE(J) * f

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3: INNER LOOP - STEP ATTEMPTS WITH CONSECUTIVE REJECTION HANDLING
    // ═══════════════════════════════════════════════════════════════════════

    consecutive_rejections = 0
    MAX_REJECTIONS = 15
    step_accepted = FALSE

    WHILE NOT step_accepted AND consecutive_rejections <= MAX_REJECTIONS:


        // ───────────────────────────────────────────────────────────────────
        // STEP 3A: SOLVE TRUST REGION SUBPROBLEM
        // ───────────────────────────────────────────────────────────────────

        // Create regularization matrix based on scaling method:
        // Levenberg:  R = μ * I
        // Marquardt:  R = μ * D^T * D = μ * diag(d₁², d₂², ..., dₙ²)
        R = CREATE_REGULARIZATION_MATRIX(μ, D, scaling_method)

        // Form augmented normal equations: A = J^T*J + R
        A = TRANSPOSE(J) * J + R

        // Solve linear system: A * p = -g
        // This is the core LM step computation
        p = SOLVE_LINEAR_SYSTEM(A, -g, linear_solver)

        // Compute scaled step norm for convergence check
        scaled_norm = COMPUTE_SCALED_NORM(D, p, scaling_method)

        // Check for convergence (step too small)
        IF scaled_norm < tolerance THEN
            RETURN (CONVERGED, x)
        END IF

        // ───────────────────────────────────────────────────────────────────
        // STEP 3B: COMPUTE TRIAL POINT AND EVALUATE
        // ───────────────────────────────────────────────────────────────────

        // Compute trial point
        x_trial = x + p

        // Apply parameter bounds if specified
        IF bounds_exist THEN
            x_trial = CLIP_TO_BOUNDS(x_trial, bounds)
        END IF

        // Evaluate function at trial point
        f_trial = EVALUATE_FUNCTION(x_trial)

        PRINT("Step computed: ||p|| =", NORM(p))
        PRINT("Function values: ||f|| =", NORM(f), "||f_trial|| =", NORM(f_trial))

        // ───────────────────────────────────────────────────────────────────
        // STEP 3C: COMPUTE REDUCTION RATIO ρ
        // ───────────────────────────────────────────────────────────────────

        // Actual reduction in objective function
        actual_reduction = NORM_SQUARED(f) - NORM_SQUARED(f_trial)

        // Predicted reduction from quadratic model
        // Different methods use different formulas:
        IF scaling_method == Nielsen THEN
            // Simple first-order approximation
            predicted_reduction = NORM_SQUARED(f) - NORM_SQUARED(f + J*p)
        ELSE IF scaling_method == GSL_More THEN
            // Moré's formula (Eq 4.4): pred = ||J*p||²/||f||² + 2*μ*||D*p||²/||f||²
            u = NORM(J * p) / NORM(f)
            v = NORM(D .* p) / NORM(f)  // Element-wise multiplication
            predicted_reduction = u² + 2*μ*v²
        END IF

        // Reduction ratio: ρ = actual/predicted
        IF ABS(predicted_reduction) > 1e-12 THEN
            ρ = actual_reduction / predicted_reduction
        ELSE
            ρ = 0.0  // Avoid division by zero
        END IF

        PRINT("Reduction ratio ρ =", ρ)
        PRINT("  Actual reduction =", actual_reduction)
        PRINT("  Predicted reduction =", predicted_reduction)

        // ───────────────────────────────────────────────────────────────────
        // STEP 3D: UPDATE TRUST REGION RADIUS δ
        // ───────────────────────────────────────────────────────────────────

        // Update trust region radius based on step quality
        IF ρ > 0.75 THEN
            δ = δ * 2.0                    // Excellent step: expand region
            PRINT("Excellent step: expanding δ to", δ)
        ELSE IF ρ < 0.25 THEN
            δ = δ / 2.0                    // Poor step: shrink region
            PRINT("Poor step: shrinking δ to", δ)
        END IF
        // For 0.25 ≤ ρ ≤ 0.75: keep δ unchanged (adequate step)

        // ───────────────────────────────────────────────────────────────────
        // STEP 3E: ACCEPT OR REJECT STEP
        // ───────────────────────────────────────────────────────────────────

        IF ρ > 0.0 THEN
            // ═══════════════════════════════════════════════════════════════
            // STEP ACCEPTED - UPDATE PARAMETERS AND EXIT INNER LOOP
            // ═══════════════════════════════════════════════════════════════

            step_accepted = TRUE

            // Update damping parameter using Nielsen's smooth accept rule:
            // μ := μ * max{1/3, 1 - (2ρ - 1)³}
            // This provides smooth decrease when step is good
            factor = MAX(1.0/3.0, 1.0 - POWER(2.0*ρ - 1.0, 3))
            μ = μ * factor
            ν = 2.0                        // Reset Nielsen parameter

            PRINT("STEP ACCEPTED:")
            PRINT("  New μ =", μ, "(decreased)")
            PRINT("  Reset ν =", ν)
            PRINT("  Final δ =", δ)

            // Store updated parameters for next iteration
            STORE_PARAMETERS(μ, ν, δ)

            RETURN (CONTINUE, x_trial)

        ELSE
            // ═══════════════════════════════════════════════════════════════
            // STEP REJECTED - UPDATE PARAMETERS AND TRY AGAIN
            // ═══════════════════════════════════════════════════════════════

            consecutive_rejections = consecutive_rejections + 1

            // Check for too many consecutive rejections
            IF consecutive_rejections > MAX_REJECTIONS THEN
                PRINT("ERROR: Too many consecutive rejections (", MAX_REJECTIONS, ")")
                PRINT("No progress possible - algorithm stuck")

                STORE_PARAMETERS(μ, ν, δ)
                STOP_TIMER()
                RETURN (NO_PROGRESS, x)
            END IF

            // Update damping parameter using Nielsen's reject rule:
            // μ := μ * ν, ν := 2 * ν
            // This provides exponential increase when steps are bad
            μ = μ * ν
            ν = ν * 2.0

            // Continue to next inner iteration with updated parameters
            // The loop will recompute step with larger μ (more regularization)

        END IF

    END WHILE  // Inner loop

    // This point should never be reached due to loop logic
    ERROR("Internal error: inner loop exited unexpectedly")

END FUNCTION

/*
================================================================================
FUNCTION: preloop_trust_region_Nielsen()
PURPOSE: Initialize algorithm parameters on first iteration (i=0)
================================================================================
*/

FUNCTION preloop_trust_region_Nielsen():

    IF iteration != 0 THEN
        RETURN  // Only run on first iteration
    END IF

    PRINT("Initializing LM parameters for first iteration...")

    // ═══════════════════════════════════════════════════════════════════════
    // INITIALIZE SCALING MATRIX D
    // ═══════════════════════════════════════════════════════════════════════

    x₀ = GET_INITIAL_POINT()
    f₀ = EVALUATE_FUNCTION(x₀)
    J₀ = EVALUATE_JACOBIAN(x₀)

    // Initialize scaling based on method
    D₀ = INITIALIZE_SCALING(J₀, scaling_method)
    STORE_SCALING_MATRIX(D₀)

    // ═══════════════════════════════════════════════════════════════════════
    // INITIALIZE DAMPING PARAMETER μ₀
    // ═══════════════════════════════════════════════════════════════════════

    τ = GET_TAU_PARAMETER()  // Typically τ = 1e-3
    JᵀJ = TRANSPOSE(J₀) * J₀

    IF scaling_method == Levenberg THEN
        // Original Levenberg: μ₀ = τ * max(diag(JᵀJ))
        μ₀ = τ * MAX(DIAGONAL(JᵀJ))
    ELSE
        // Marquardt/Moré: μ₀ = τ * max(diag(JᵀJ) .* D₀²)
        scaled_diag = DIAGONAL(JᵀJ) .* (D₀ .* D₀)  // Element-wise operations
        μ₀ = τ * MAX(scaled_diag)
    END IF

    // ═══════════════════════════════════════════════════════════════
*/

//=================================================================================
//  NIELSEN LEVENBERG-MARQUARDT METHOD ADVANCED
//=================================================================================
/// Advanced Nielsen LM with full trust region and scaling support.
/// Integrates the step_trust_region_Nielsen logic into the new pipeline.
#[derive(Debug, Clone, Copy)]
pub struct NielsenLevenbergMarquardtMethodAdvanced {
    /// Initial `mu` scale (tau parameter).
    pub tau: f64,
    /// Initial rejection multiplier.
    pub nu_init: f64,
    /// Trust-region growth factor.
    pub factor_up: f64,
    /// Trust-region shrink factor.
    pub factor_down: f64,
    /// Acceptance threshold for `rho`.
    pub rho_threshold: f64,
    /// Optional function tolerance.
    pub f_tolerance: Option<f64>,
    /// Optional gradient tolerance.
    pub g_tolerance: Option<f64>,
    /// Scaling method (Levenberg, Marquardt, or More).
    pub scaling_method: ScalingMethod,
    /// Reduction ratio calculation method.
    pub reduction_ratio_method: ReductionRatio,
    /// Maximum number of consecutive rejections per iteration.
    pub max_rejections: usize,
}

impl Default for NielsenLevenbergMarquardtMethodAdvanced {
    fn default() -> Self {
        Self {
            tau: 1e-6,
            nu_init: 2.0,
            factor_up: 3.0,
            factor_down: 2.0,
            rho_threshold: 1e-4,
            f_tolerance: None,
            g_tolerance: None,
            scaling_method: ScalingMethod::Marquardt,
            reduction_ratio_method: ReductionRatio::Nielsen,
            max_rejections: 15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NielsenLevenbergMarquardtStateAdvanced {
    mu: f64,
    nu: f64,
    delta: f64,
    scaling: DVector<f64>,
}

impl NonlinearMethod for NielsenLevenbergMarquardtMethodAdvanced {
    type MethodState = NielsenLevenbergMarquardtStateAdvanced;

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        x0: &DVector<f64>,
        _options: &SolveOptions,
        residual: &DVector<f64>,
        jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        if self.tau <= 0.0 || self.nu_init <= 0.0 || self.max_rejections == 0 {
            return Err(SolveError::InvalidConfig(
                "invalid Nielsen LM Advanced parameters".to_string(),
            ));
        }

        // Initialize scaling vector based on method
        let scaling = TrustRegionScaling::init_scaling(jacobian, &self.scaling_method);

        // Initialize damping parameter μ
        let jtj = jacobian.transpose() * jacobian;
        let mu = self.tau
            * if self.scaling_method == ScalingMethod::Levenberg {
                jtj.diagonal().max()
            } else {
                jtj.diagonal()
                    .component_mul(&scaling)
                    .component_mul(&scaling)
                    .max()
            };

        if !residual.norm().is_finite() {
            return Err(SolveError::NumericalBreakdown(
                "initial residual norm is not finite".to_string(),
            ));
        }

        // Initialize trust region radius δ
        // source GSL trust.c, line ~200
        let delta = 0.3 * (1.0f64).max(scaled_norm_common(&scaling, x0));

        info!(
            "Initialized Nielsen LM Advanced with {} scaling",
            self.scaling_method
        );
        info!("  mu0 = {}, nu0 = {}, delta0 = {}", mu, self.nu_init, delta);
        info!("  Reduction ratio method: {}", self.reduction_ratio_method);

        Ok(NielsenLevenbergMarquardtStateAdvanced {
            mu: mu.max(1e-16),
            nu: self.nu_init,
            delta,
            scaling,
        })
    }

    fn step<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError> {
        // Update scaling based on current Jacobian
        TrustRegionScaling::update_scaling(
            &state.jacobian,
            &self.scaling_method,
            &mut method_state.scaling,
        );

        // Compute gradient (constant during inner loop)
        let gradient = -state.jacobian.transpose() * &state.residual;

        // Check gradient tolerance
        if let Some(g_tol) = self.g_tolerance {
            if gradient.norm() < g_tol {
                info!(
                    "Gradient tolerance satisfied: ||g|| = {:.6e}",
                    gradient.norm()
                );
                return Ok(StepOutcome::Terminated(TerminationReason::Converged));
            }
        }

        // Check function tolerance
        if let Some(f_tol) = self.f_tolerance {
            if state.residual_norm < f_tol {
                info!(
                    "Function tolerance satisfied: ||f|| = {:.6e}",
                    state.residual_norm
                );
                return Ok(StepOutcome::Terminated(TerminationReason::Converged));
            }
        }

        let jtj = state.jacobian.transpose() * &state.jacobian;

        // INNER LOOP: Try steps until one is accepted or max rejections exceeded
        // source GSL trust.c
        /*
        1. Inner Loop Structure:
        Continues until acceptable step found or max rejections exceeded
        Tracks consecutive_rejections counter
        Updates mu, nu, delta on each rejection
        2. Trust Region Updates:
        rho > 0.75: Expand delta (excellent step)
        rho < 0.25: Shrink delta (poor step)
        0.25 ≤ rho ≤ 0.75: Keep delta unchanged
        3. LM Parameter Updates:
        Accept: Nielsen's smooth update rule
        Reject: Increase mu exponentially, double nu
        4. Step Recomputation:
        After each rejection, solves trust region subproblem again
        Uses updated mu and delta values
        Evaluates new trial point and computes new rho
        5. Termination Conditions:
        Success: rho > 0 (step accepted)
        Failure: More than 15 consecutive rejections
        Error: Returns special code 0 for no progress
        */
        for rejection_count in 0..self.max_rejections {
            info!("=== Inner iteration {} ===", rejection_count);
            info!(
                "Current: mu = {:.6e}, nu = {:.6e}, delta = {:.6e}",
                method_state.mu, method_state.nu, method_state.delta
            );

            // SOLVE TRUST REGION SUBPROBLEM with current parameters
            // Create scaled regularization matrix based on method
            let reg = TrustRegionScaling::create_scaled_regularization(
                method_state.mu,
                &method_state.scaling,
                &self.scaling_method,
            );

            runtime.linear_solves += 1;
            let step = solve_linear_system(options.linear_solver, &(&jtj + reg), &gradient)?;

            info!("Step vector norm: {:.6e}", step.norm());

            // Compute scaled step norm for convergence check
            let scaled_step_norm = scaled_norm_common(&method_state.scaling, &step);
            info!("Scaled step norm = {:.6e}", scaled_step_norm);

            if scaled_step_norm < options.tolerance {
                info!("Solution found (small step)!");
                return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
            }

            // Compute trial point
            let trial_x = if let Some(bounds) = &options.bounds {
                bounds.project(&(&state.x + &step))
            } else {
                &state.x + &step
            };

            // Evaluate function at trial point
            let trial_residual = problem.residual(&trial_x)?;

            // Compute reduction ratio using selected method
            let rho = ReductionRatioSolver::solve_reduction_ratio(
                &state.jacobian,
                &state.residual,
                &trial_residual,
                &method_state.scaling,
                &step,
                method_state.mu,
                self.reduction_ratio_method.clone(),
            );

            info!("Step norm = {:.6e}, rho = {:.6e}", step.norm(), rho);

            // UPDATE TRUST REGION RADIUS based on step quality
            if rho > 0.75 {
                method_state.delta *= self.factor_up;
                info!(
                    "Excellent step (rho > 0.75): expanding delta to {:.6e}",
                    method_state.delta
                );
            } else if rho < 0.25 {
                method_state.delta /= self.factor_down;
                info!(
                    "Poor step (rho < 0.25): shrinking delta to {:.6e}",
                    method_state.delta
                );
            }
            // For 0.25 <= rho <= 0.75: keep delta unchanged

            // CHECK IF STEP IS ACCEPTABLE
            // MINPACK recommends using 10^-4 as threshold
            if rho > self.rho_threshold {
                // STEP ACCEPTED
                // Update trust region parameters (Nielsen's method)
                // ρ = actual_reduction / predicted_reduction
                // When ρ > 0: Good step, decrease damping with smooth function
                // When ρ ≤ 0: Bad step, increase damping exponentially
                // The cubic term (2ρ-1)³ provides smooth transition
                method_state.mu *= (ONE_THIRD).max(1.0 - (2.0 * rho - 1.0).powi(3));
                method_state.mu = method_state.mu.max(1e-16);
                method_state.nu = self.nu_init; // Reset nu

                info!(
                    "Step ACCEPTED: mu updated to {:.6e}, nu reset to {:.6e}",
                    method_state.mu, method_state.nu
                );

                runtime.accepted_steps += 1;
                return Ok(StepOutcome::Continue {
                    next_x: trial_x,
                    accepted: true,
                });
            } else {
                // STEP REJECTED
                info!(
                    "Step REJECTED (rho <= {}): consecutive rejections = {}",
                    self.rho_threshold,
                    rejection_count + 1
                );

                // Update LM parameters using Nielsen's reject rule
                method_state.mu *= method_state.nu;
                method_state.nu *= 2.0;

                info!(
                    "LM parameters updated for next attempt: mu = {:.6e}, nu = {:.6e}",
                    method_state.mu, method_state.nu
                );

                runtime.rejected_steps += 1;

                // Continue to next inner iteration with updated mu, nu, delta
                // The loop will recompute the step with new parameters
            }
        }

        // Too many consecutive rejections
        warn!(
            "Too many consecutive rejections ({}). No progress possible.",
            self.max_rejections
        );
        Ok(StepOutcome::Terminated(
            TerminationReason::RejectedStepLimit,
        ))
    }
}
//=================================================================================
//  NIELSEN LEVENBERG-MARQUARDT METHOD
//=================================================================================
/// Internal state of Nielsen LM.
/// Nielsen-style Levenberg-Marquardt method.
/*
Main sources (In brackets we will indicate the abbreviated name)
1. GSL multifit_nlinear folder (GSL)
2. "THE LEVENBERG-MARQUARDT  ALGORITHM: IMPLEMENTATION  AND THEORY by Jorge  J. More' " in
the book "Lecture notes in Mathematics №630"p. 105 - 116 (More)
3.B.Nielsen, K.Madsen "Introduction to Optimization and Data Fitting" (Nielsen)
4. MINPACK Fortran code https://www.netlib.org/minpack/hybrd.f
or f90 revision (https://github.com/fortran-lang/minpack) (MINPACK)

"Therefore, we recommend to use the equally simple strategy given by
if ̺ ro> 0 then µ := µ ∗ max{1/3, 1 − (2ro - 1)^3}
else µ := µ ∗ 2" Nielsen, p 51
*/
#[derive(Debug, Clone, Copy)]
pub struct NielsenLevenbergMarquardtMethod {
    /// Initial `mu` scale.
    pub tau: f64,
    /// Initial rejection multiplier.
    pub nu_init: f64,
    /// Trust-region growth factor.
    ///  GSL fdf.c uses default parameters:  factor_up = 3.0;
    pub factor_up: f64,
    /// Trust-region shrink factor.
    pub factor_down: f64,
    /// Acceptance threshold for `rho`.
    pub rho_threshold: f64,
    /// Optional function tolerance.
    pub f_tolerance: Option<f64>,
    /// Optional gradient tolerance.
    pub g_tolerance: Option<f64>,
    /// Use Jacobian column scaling.
    pub use_column_scaling: bool,
    /// Maximum number of rejections per iteration.
    pub max_rejections: usize,
}

impl Default for NielsenLevenbergMarquardtMethod {
    fn default() -> Self {
        Self {
            tau: 1e-6,
            nu_init: 2.0,
            factor_up: 3.0,
            factor_down: 2.0,
            rho_threshold: 1e-4,
            f_tolerance: None,
            g_tolerance: None,
            use_column_scaling: true,
            max_rejections: 15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NielsenLevenbergMarquardtState {
    mu: f64,
    nu: f64,
    delta: f64,
}

impl NonlinearMethod for NielsenLevenbergMarquardtMethod {
    type MethodState = NielsenLevenbergMarquardtState;

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        x0: &DVector<f64>,
        _options: &SolveOptions,
        residual: &DVector<f64>,
        jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        if self.tau <= 0.0 || self.nu_init <= 0.0 || self.max_rejections == 0 {
            return Err(SolveError::InvalidConfig(
                "invalid Nielsen LM parameters".to_string(),
            ));
        }
        let scaling = scaling_vector(jacobian, self.use_column_scaling);
        let jtj = jacobian.transpose() * jacobian;
        let mu = self.tau
            * if self.use_column_scaling {
                jtj.diagonal()
                    .component_mul(&scaling)
                    .component_mul(&scaling)
                    .max()
            } else {
                jtj.diagonal().max()
            };
        if !residual.norm().is_finite() {
            return Err(SolveError::NumericalBreakdown(
                "initial residual norm is not finite".to_string(),
            ));
        }
        Ok(NielsenLevenbergMarquardtState {
            mu: mu.max(1e-16),
            nu: self.nu_init,
            delta: 0.3 * (1.0f64).max(scaled_norm(&scaling, x0)),
        })
    }

    fn step<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError> {
        let scaling = scaling_vector(&state.jacobian, self.use_column_scaling);
        let gradient = -state.jacobian.transpose() * &state.residual;
        if let Some(g_tol) = self.g_tolerance {
            if gradient.norm() < g_tol {
                return Ok(StepOutcome::Terminated(TerminationReason::Converged));
            }
        }
        if let Some(f_tol) = self.f_tolerance {
            if state.residual_norm < f_tol {
                return Ok(StepOutcome::Terminated(TerminationReason::Converged));
            }
        }

        let jtj = state.jacobian.transpose() * &state.jacobian;
        for _ in 0..self.max_rejections {
            let reg = DMatrix::from_diagonal(&scaling.map(|v| method_state.mu * v * v));
            runtime.linear_solves += 1;
            let step = solve_linear_system(options.linear_solver, &(&jtj + reg), &gradient)?;
            if scaled_norm(&scaling, &step) < options.tolerance {
                return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
            }

            let trial_x = if let Some(bounds) = &options.bounds {
                bounds.project(&(&state.x + &step))
            } else {
                &state.x + &step
            };
            let trial_residual = problem.residual(&trial_x)?;
            let actual = state.residual.norm_squared() - trial_residual.norm_squared();
            let trial_jacobian = problem.jacobian(&trial_x)?;
            let predicted = 0.5
                * step.dot(
                    &(step.component_mul(&scaling).map(|v| method_state.mu * v)
                        - trial_jacobian.transpose() * &trial_residual),
                );
            let rho = if predicted.abs() > 1e-12 {
                actual / predicted
            } else {
                0.0
            };

            if rho > 0.75 {
                method_state.delta *= self.factor_up;
            } else if rho < 0.25 {
                method_state.delta /= self.factor_down;
            }
            if rho > self.rho_threshold {
                method_state.mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                method_state.mu = method_state.mu.max(1e-16);
                method_state.nu = self.nu_init;
                runtime.accepted_steps += 1;
                return Ok(StepOutcome::Continue {
                    next_x: trial_x,
                    accepted: true,
                });
            }

            runtime.rejected_steps += 1;
            method_state.mu *= method_state.nu;
            method_state.nu *= 2.0;
        }

        Ok(StepOutcome::Terminated(
            TerminationReason::RejectedStepLimit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::error::TerminationReason;
    use crate::numerical::Nonlinear_systems::problem::{
        Bounds, JacobianProvider, NonlinearProblem,
    };
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem;
    use approx::assert_relative_eq;

    struct RosenbrockProblem;

    impl NonlinearProblem for RosenbrockProblem {
        fn dimension(&self) -> usize {
            2
        }
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![
                10.0 * (x[1] - x[0] * x[0]),
                1.0 - x[0],
            ]))
        }
    }

    impl JacobianProvider for RosenbrockProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[-20.0 * x[0], 10.0, -1.0, 0.0],
            ))
        }
    }

    #[test]
    fn nielsen_lm_advanced_converges_rosenbrock() {
        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(NielsenLevenbergMarquardtMethodAdvanced::default(), options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_with_bounds() {
        let bounds = Bounds::new(vec![(0.0, 2.0), (0.0, 2.0)]).expect("bounds");
        let options = SolveOptions {
            bounds: Some(bounds),
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(NielsenLevenbergMarquardtMethodAdvanced::default(), options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert!(result.x[0] >= 0.0 && result.x[0] <= 2.0);
        assert!(result.x[1] >= 0.0 && result.x[1] <= 2.0);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_with_symbolic_problem() {
        let problem = SymbolicNonlinearProblem::from_strings(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .expect("symbolic problem");

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(NielsenLevenbergMarquardtMethodAdvanced::default(), options)
            .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_with_gradient_tolerance() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        method.g_tolerance = Some(1e-8);

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
    }

    #[test]
    fn nielsen_lm_advanced_with_function_tolerance() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        method.f_tolerance = Some(1e-8);

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
    }

    #[test]
    fn nielsen_lm_advanced_without_column_scaling() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        //  method.use_column_scaling = false;

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_custom_parameters() {
        let method = NielsenLevenbergMarquardtMethodAdvanced {
            tau: 1e-3,
            nu_init: 3.0,
            factor_up: 2.5,
            factor_down: 1.5,
            rho_threshold: 1e-3,
            f_tolerance: None,
            g_tolerance: None,

            max_rejections: 20,
            scaling_method: ScalingMethod::More,
            reduction_ratio_method: ReductionRatio::More,
        };

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_handles_difficult_start() {
        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 200,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(NielsenLevenbergMarquardtMethodAdvanced::default(), options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![-1.0, 2.0]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_with_levenberg_scaling() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        method.scaling_method = ScalingMethod::Levenberg;

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_with_more_scaling() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        method.scaling_method = ScalingMethod::More;

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_with_more_reduction_ratio() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        method.reduction_ratio_method = ReductionRatio::More;

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_levenberg_with_more_ratio() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        method.scaling_method = ScalingMethod::Levenberg;
        method.reduction_ratio_method = ReductionRatio::More;

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_more_scaling_nielsen_ratio() {
        let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
        method.scaling_method = ScalingMethod::More;
        method.reduction_ratio_method = ReductionRatio::Nielsen;

        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(method, options)
            .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn nielsen_lm_advanced_all_combinations_with_bounds() {
        let bounds = Bounds::new(vec![(0.0, 2.0), (0.0, 2.0)]).expect("bounds");

        for scaling in [
            ScalingMethod::Levenberg,
            ScalingMethod::Marquardt,
            ScalingMethod::More,
        ] {
            for ratio in [ReductionRatio::Nielsen, ReductionRatio::More] {
                let mut method = NielsenLevenbergMarquardtMethodAdvanced::default();
                method.scaling_method = scaling.clone();
                method.reduction_ratio_method = ratio.clone();

                let options = SolveOptions {
                    bounds: Some(bounds.clone()),
                    tolerance: 1e-6,
                    max_iterations: 150,
                    ..SolveOptions::default()
                };

                let result = SolverEngine::new(method, options)
                    .solve(&RosenbrockProblem, DVector::from_vec(vec![0.5, 0.5]))
                    .expect(&format!(
                        "solve with {:?} scaling and {:?} ratio",
                        scaling, ratio
                    ));

                assert_eq!(
                    result.termination,
                    TerminationReason::Converged,
                    "Failed with {:?} scaling and {:?} ratio",
                    scaling,
                    ratio
                );
                assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
                assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
            }
        }
    }
}
