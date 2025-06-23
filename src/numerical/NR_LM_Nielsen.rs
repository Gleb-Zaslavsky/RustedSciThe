use super::NR::{NR, solve_linear_system};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use crate::numerical::dogleg::{DoglegSolver, Powell_dogleg_method};
use log::{error, info};
use nalgebra::{DMatrix, DVector};

use crate::numerical::LM_utils::{
    ConvergenceCriteria, ConvergenceCriteriaSolver, ReductionRatioSolver, ScalingMethod,
    TrustRegionScaling, TrustRegionSubproblem, scaled_norm_common,
};

use std::time::Instant;
const ONE_THIRD: f64 = 1.0 / 3.0;

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
////////////////////////TRUST REGION METHODS////////////////////////

impl NR {
    pub fn step_trust_region_Nielsen_simple(&mut self) -> (i32, Option<DVector<f64>>) {
        let now = Instant::now();
        let bounds_vec = self.bounds_vec.clone();
        let parameters = self.parameters.clone().unwrap();

        let tau = parameters.get("tau").unwrap();
        let method = self.linear_sys_method.clone().unwrap();

        // mutable parameters
        let mut parameters_mut = self.parameters.clone().unwrap();

        let y = self.y.clone();
        let Fy = self.evaluate_function(y.clone());
        let Jy = self.evaluate_jacobian(y.clone());
        let B = Jy.clone().transpose() * Jy.clone();
        let mu0 = tau * (B.diagonal()).max();
        let mut mu = if self.i == 0 {
            mu0
        } else {
            *parameters_mut.get("mu").unwrap()
        };
        let mut nu = *parameters_mut.get("nu").unwrap();

        info!("mu = {}, nu = {}", mu, nu);
        let g = -Jy.transpose() * Fy.clone();
        let A = B.clone() + mu * DMatrix::identity(B.nrows(), B.ncols());
        // Check convergence
        let norm = Fy.norm_squared();
        info!("norm = {}", norm);
        if norm < self.tolerance {
            info!("Solution found!");
            return (1, Some(y));
        }

        // solve the linear system - find step p
        let p = solve_linear_system(method, &A, &g).expect("Failed to solve linear system");
        let mut y_new = y.clone() + &p;

        // Project onto bounds (if they exist)
        y_new = self.clip(&y_new, &bounds_vec);

        let Fy_new = self.evaluate_function(y_new.clone());
        let Jy_new = self.evaluate_jacobian(y_new.clone());
        let actual_reduction = Fy.norm_squared() - Fy_new.norm_squared();
        //    let predicted_reduction =    p.transpose().dot( &(Jy_new.transpose() * Fy_new.clone()))
        //   - 0.5 *p.clone().dot(&(Jy_new.transpose()*Jy_new.clone()*p.clone()));
        let predicted_reduction = 0.5
            * p.clone()
                .dot(&(p.clone() * mu - Jy_new.transpose() * Fy_new.clone()));
        let rho = if predicted_reduction.abs() > 1e-12 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };
        info!("rho is: {}", rho);
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        // renew trust region
        if rho > 0.0 {
            mu = mu * (ONE_THIRD).max(1.0 - (2.0 * rho - 1.0).powi(3));
            nu = 2.0;
            info!("rho > 0, mu changed to: {:3}, nu changed to: {:3}", mu, nu);
            // Update the parameters HashMap
            parameters_mut.insert("mu".to_string(), mu);
            parameters_mut.insert("nu".to_string(), nu);
            self.parameters = Some(parameters_mut);
            let y_result = y_new;
            return (0, Some(y_result));
        } else {
            mu = mu * nu;
            nu = nu * 2.0;
            info!("rho<0,mu changed to: {:3}, nu changed to: {:3}", mu, nu);
            // Update the parameters HashMap
            parameters_mut.insert("mu".to_string(), mu);
            parameters_mut.insert("nu".to_string(), nu);
            self.parameters = Some(parameters_mut);
            let y_result = y;
            return (0, Some(y_result));
        }
    }
    /////////////////////////////////ADVANCED NIELSEN TRUST REGION ALGORITHM////////////////////////

    /// Enhanced Nielsen trust region step with scaling support
    pub fn step_trust_region_Nielsen(&mut self) -> (i32, Option<DVector<f64>>) {
        let now = Instant::now();
        // STEP 1: PRELOOP - Initialization of the algorithm
        // Run preloop initialization if at iteration 0
        if let Err(e) = self.preloop_trust_region_Nielsen() {
            error!("Preloop initialization failed: {}", e);
            return (-1, None);
        }
        // STEP 2: CALCULTING JACOBIAN ANF RESIDUALS
        let scaling_method = self.scaling_method.clone().unwrap();
        let bounds_vec = self.bounds_vec.clone();
        let method = self.linear_sys_method.clone().unwrap();
        let subproblem_method = self.subproblem_method.clone().unwrap(); // Trust region subproblem method
        let reduction_ratio_method = self.reduction_ratio.clone().unwrap(); // Reduction ratio calculation method
        let update_method = self.update_method.clone().unwrap();
        let convergence_criteria = self.convergence_criteria.clone().unwrap();

        // Get parameters
        let mut parameters_mut = self.parameters.clone().unwrap();
        let mut mu = *parameters_mut.get("mu").unwrap();
        let mut nu = *parameters_mut.get("nu").unwrap();
        let mut delta = *parameters_mut.get("delta").unwrap();
        // GSL fdf.c uses default parameters:  params.factor_up = 3.0;
        // params.factor_down = 2.0;
        let factor_up = *parameters_mut.get("factor_up").unwrap();
        let factor_down = *parameters_mut.get("factor_down").unwrap();
        //
        let rho_threshold = *parameters_mut.get("rho_threshold").unwrap();
        let f_tolerance = parameters_mut.get("f_tolerance");
        let g_tolerance = parameters_mut.get("g_tolerance");

        // Current point (doesn't change during inner loop)
        let y = self.y.clone();
        let Fy = self.evaluate_function(y.clone());
        let Jy = self.evaluate_jacobian(y.clone());

        // Get or update scaling
        let diag = if self.i == 0 {
            // At iteration 0, scaling was already initialized in preloop
            TrustRegionScaling::extract_scaling_from_vector(&self.scales_vec, Jy.ncols())
        } else {
            // For subsequent iterations, update existing scaling
            let mut current_diag =
                TrustRegionScaling::extract_scaling_from_vector(&self.scales_vec, Jy.ncols());
            TrustRegionScaling::update_scaling(&Jy, &scaling_method, &mut current_diag);
            // Store updated scaling
            TrustRegionScaling::store_scaling_in_vector(
                &mut self.scales_vec,
                &current_diag,
                self.i,
            );
            current_diag
        };

        info!("Scaling diagonal: {:?}", diag.as_slice());
        info!("Initial: mu = {}, nu = {}, delta = {}", mu, nu, delta);

        // Compute gradient (constant during inner loop)
        let g = -Jy.transpose() * &Fy;
        let norm = g.norm_squared();
        info!("Function norm = {}", norm);

        // INNER LOOP: Try steps until one is accepted or max rejections exceeded
        //  source GSL trust.c
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
        let max_consecutive_rejections = 15;
        let mut consecutive_rejections = 0;
        let mut found_acceptable_step = false;
        while !found_acceptable_step {
            info!("=== Inner iteration {} ===", consecutive_rejections);
            info!("Current: mu = {}, nu = {}, delta = {}", mu, nu, delta);

            //STEP 3: SOLVE TRUST REGION SUBPROBLEM with current parameters
            let p = TrustRegionSubproblem::solve(
                &Jy,
                &Fy,
                &g,
                &diag,
                mu,
                delta,
                &method,
                &scaling_method,
                subproblem_method.clone(),
            );
            info!("Step vector: {:?},\n y = {:?}", &p, &y);
            // Compute scaled step norm for convergence check
            let (converged, flag) = ConvergenceCriteriaSolver::test_convergence(
                &y,                   // Current parameter vector
                &p,                   // Parameter step vector
                &g,                   // Gradient vector J^T * f
                &Fy,                  // Residual vector
                &diag,                // Diagonal matrix of parameter scaling factors
                self.tolerance,       // Parameter change tolerance
                g_tolerance.copied(), // Gradient tolerance
                f_tolerance.copied(), // Function change tolerance (currently unused)
                convergence_criteria.clone(),
            );
            //   info!("Scaled step norm = {}", scaled_step_norm);

            if Fy.norm_squared() < self.tolerance {
                info!("Solution found (converged)!");
                // Update final parameters
                parameters_mut.insert("mu".to_string(), mu);
                parameters_mut.insert("nu".to_string(), nu);
                parameters_mut.insert("delta".to_string(), delta);
                self.parameters = Some(parameters_mut);
                return (1, Some(y));
            }
            if flag == 2 {
                info!("Solution found (small step)!");
                // Update final parameters
                parameters_mut.insert("mu".to_string(), mu);
                parameters_mut.insert("nu".to_string(), nu);
                parameters_mut.insert("delta".to_string(), delta);
                self.parameters = Some(parameters_mut);
                return (1, Some(y));
            }

            // Compute trial point
            let mut y_new = &y + &p;
            y_new = self.clip(&y_new, &bounds_vec);

            // Evaluate function at trial point
            let Fy_new = self.evaluate_function(y_new.clone());

            //STEP 4: Compute reduction ratio
            let rho = ReductionRatioSolver::solve_reduction_ratio(
                &Jy,
                &Fy,
                &Fy_new,
                &diag,
                &p,
                mu,
                reduction_ratio_method.clone(),
            );

            info!("Step norm = {}, rho = {}", p.norm(), rho);

            //STEP 5: UPDATE TRUST REGION RADIUS based on step quality
            if rho > 0.75 {
                delta *= factor_up;
                info!("Excellent step (rho > 0.75): expanding delta to {}", delta);
            } else if rho < 0.25 {
                delta /= factor_down;
                info!("Poor step (rho < 0.25): shrinking delta to {}", delta);
            }
            // For 0.25 <= rho <= 0.75: keep delta unchanged

            // CHECK IF STEP IS ACCEPTABLE
            // MINPACK recommends using 10^-4 as threshold
            if rho > rho_threshold {
                // STEP ACCEPTED
                found_acceptable_step = true;
                // Update trust region parameters (Nielsen's method)
                // ρ = actual_reduction / predicted_reduction
                // When ρ > 0: Good step, decrease damping with smooth function
                // When ρ ≤ 0: Bad step, increase damping exponentially
                // The cubic term (2ρ-1)³ provides smooth transition
                // Update LM parameters using Nielsen's accept rule
                mu = mu * (ONE_THIRD).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0; // Reset nu

                info!("Step ACCEPTED: mu updated to {}, nu reset to {}", mu, nu);

                // Update and store final parameters
                parameters_mut.insert("mu".to_string(), mu);
                parameters_mut.insert("nu".to_string(), nu);
                parameters_mut.insert("delta".to_string(), delta);
                self.parameters = Some(parameters_mut);

                let elapsed = now.elapsed();
                elapsed_time(elapsed);

                return (0, Some(y_new));
            } else {
                // STEP REJECTED
                consecutive_rejections += 1;
                info!(
                    "Step REJECTED (rho <= 0): consecutive rejections = {}",
                    consecutive_rejections
                );

                // Update LM parameters using Nielsen's reject rule
                mu = mu * nu;
                nu = nu * 2.0;

                // Check if we've exceeded maximum consecutive rejections
                if consecutive_rejections > max_consecutive_rejections {
                    error!(
                        "Too many consecutive rejections ({}). No progress possible.",
                        max_consecutive_rejections
                    );

                    // Update parameters before returning failure
                    parameters_mut.insert("mu".to_string(), mu);
                    parameters_mut.insert("nu".to_string(), nu);
                    parameters_mut.insert("delta".to_string(), delta);
                    self.parameters = Some(parameters_mut);

                    let elapsed = now.elapsed();
                    elapsed_time(elapsed);

                    return (0, Some(y)); // Special error code for no progress
                }
                info!(
                    "LM parameters updated for next attempt: mu = {}, nu = {}",
                    mu, nu
                );

                // Continue to next inner iteration with updated mu, nu, delta
                // The loop will recompute the step with new parameters
            }
        }
        // This should never be reached due to the loop logic above
        unreachable!("Inner loop should always return via accept or max rejections")
    }

    /// Preloop initialization - called only at iteration 0 (source: GSL)
    pub fn preloop_trust_region_Nielsen(&mut self) -> Result<(), String> {
        if self.i != 0 {
            return Ok(()); // Only run at iteration 0
        }

        info!("Initializing trust region parameters at iteration 0");

        let parameters = self.parameters.clone().unwrap();
        let tau = parameters.get("tau").unwrap();
        let scaling_method = self.scaling_method.clone().unwrap();

        let y = self.y.clone();
        let Fy = self.evaluate_function(y.clone());
        let Jy = self.evaluate_jacobian(y.clone());

        // 1. Initialize scaling vector
        let diag = TrustRegionScaling::init_scaling(&Jy, &scaling_method);
        TrustRegionScaling::store_scaling_in_vector(&mut self.scales_vec, &diag, self.i);
        info!("Initial scaling diagonal: {:?}", diag.as_slice());

        // 2. Initialize damping parameter μ
        let JtJ = Jy.transpose() * &Jy;
        let mu0 = if scaling_method == ScalingMethod::Levenberg {
            // Original approach for Levenberg
            tau * JtJ.diagonal().max()
        } else {
            // For Marquardt/More, use scaled diagonal maximum
            let scaled_diag = JtJ.diagonal().component_mul(&diag).component_mul(&diag);
            tau * scaled_diag.max()
        };

        // 3. Initialize trust region radius δ (if using trust region)
        // source GSL trust.c, line ~200
        let scaled_norm = scaled_norm_common(&diag, &y);
        let delta0 = 0.3 * (1.0_f64).max(scaled_norm);

        // 4. Initialize Nielsen parameters
        let nu0 = 2.0;

        // Store all initialized parameters
        let mut parameters_mut = self.parameters.clone().unwrap();
        parameters_mut.insert("mu".to_string(), mu0);
        parameters_mut.insert("nu".to_string(), nu0);
        parameters_mut.insert("delta".to_string(), delta0);
        self.parameters = Some(parameters_mut);

        info!(
            "Initialized: mu0 = {}, nu0 = {}, delta0 = {}",
            mu0, nu0, delta0
        );

        Ok(())
    }
} // end of struct NR
