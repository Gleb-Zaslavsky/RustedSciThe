//! Public solve boundary for the new BVP architecture.

use std::sync::Arc;

use super::{
    callbacks::{BvpSciBoundary, evaluate_boundary},
    collocation::{collocation_rms, evaluate_collocation, max_abs},
    config::{BvpSciMatrixLayout, BvpSciOptions},
    error::{BvpSciNewError, BvpSciStage},
    jacobian::assemble_global_jacobian,
    linear::NativeLinearBackend,
    output::{BvpSciDenseOutput, BvpSciSolution},
    prepared::BvpSciLambdifyPlan,
    singular::SingularTermRuntime,
    telemetry::BvpSciTelemetryStage,
    workspace::BvpSciCollocationWorkspace,
};

/// Newton collocation solver using a prepared Lambdify frontend.
///
/// The prepared plan and the selected linear backend are retained across
/// continuation solves.  Only numerical buffers and parameter values change.
pub struct BvpSciSolver {
    /// Prepared symbolic/native callback plan.  This is never rebuilt by
    /// Newton, continuation or adaptive mesh refinement.
    plan: BvpSciLambdifyPlan,
    /// Boundary residual callback shared by all numerical stages.
    boundary: Arc<dyn BvpSciBoundary>,
    /// Validated controller and backend policies.
    options: BvpSciOptions,
    /// Current strictly increasing adaptive mesh.
    x: Vec<f64>,
    /// Current node-major state values.
    y: Vec<f64>,
    /// Current runtime parameter values; symbolic structure is immutable.
    parameters: Vec<f64>,
    /// Reusable collocation, callback and Jacobian assembly buffers.
    workspace: BvpSciCollocationWorkspace,
    /// Native linear backend selected for the current mesh.
    backend: NativeLinearBackend,
    /// Trial state used by Newton backtracking and mesh defect probes.
    trial_state: Vec<f64>,
    /// Trial parameter block used by parameter finite differences in Newton.
    trial_parameters: Vec<f64>,
    /// Prepared pointwise SciPy singular-term correction, if configured.
    singular: Option<SingularTermRuntime>,
    /// Set after a parameter rebind and consumed by the next solve.
    continuation_pending: bool,
}

/// Result of one SciPy-style Newton pass on the current mesh.
///
/// SciPy does not turn exhaustion of the inner Newton budget into an
/// immediate solver error.  It returns the last finite iterate to the outer
/// controller, which then evaluates defects and may refine the mesh.  Keeping
/// that distinction explicit prevents a recoverable inner budget event from
/// being confused with a terminal numerical failure.
#[derive(Debug, Clone, Copy, PartialEq)]
enum NewtonMeshOutcome {
    /// The collocation and boundary residuals satisfied the current mesh
    /// tolerances during the Newton pass.
    Converged { norm: f64, iterations: usize },
    /// The last finite Newton iterate is available, but the inner budget was
    /// exhausted.  The outer controller must still inspect its defect.
    BudgetExhausted { norm: f64, iterations: usize },
}

impl BvpSciSolver {
    /// Start the fluent public builder for the symbolic Lambdify route.
    pub fn builder(
        equations: Vec<crate::symbolic::symbolic_engine::Expr>,
        state_names: Vec<String>,
    ) -> super::builder::BvpSciSolverBuilder {
        super::builder::BvpSciSolverBuilder::new(equations, state_names)
    }

    /// Create a solver from a prepared ExprLegacy or AtomView plan.
    pub fn new<B>(
        plan: BvpSciLambdifyPlan,
        boundary: B,
        x: Vec<f64>,
        mut y: Vec<f64>,
        parameters: Vec<f64>,
        mut options: BvpSciOptions,
    ) -> Result<Self, BvpSciNewError>
    where
        B: BvpSciBoundary + 'static,
    {
        if matches!(options.execution, super::config::BvpSciExecution::Aot) != plan.is_aot() {
            return Err(BvpSciNewError::UnsupportedRoute(
                "BVP execution selection does not match the prepared frontend plan".into(),
            ));
        }
        if let Some(aot_policy) = plan.aot_execution_policy() {
            if aot_policy != options.execution_policy {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "AOT callback policy must be selected during AOT preparation".into(),
                ));
            }
        }
        if let Some(assembly) = options.assembly {
            if assembly != plan.assembly() {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "selected assembly does not match the prepared Lambdify plan".into(),
                ));
            }
        }
        let plan = plan.with_execution_policy(options.execution_policy);
        let n = plan.dimension();
        if matches!(plan, BvpSciLambdifyPlan::Numerical(_)) {
            if options.execution != super::config::BvpSciExecution::Numerical {
                return Err(BvpSciNewError::UnsupportedRoute(
                    "Numerical callback plans require BvpSciExecution::Numerical".into(),
                ));
            }
            if !matches!(options.matrix_layout, BvpSciMatrixLayout::Dense) {
                return Err(BvpSciNewError::UnsupportedRoute(
                    "Numerical BVP callbacks currently support only the Dense matrix layout".into(),
                ));
            }
        }
        let m = x.len();
        let state_dimension = n.checked_mul(m).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP state dimension overflows usize".into())
        })?;
        if m < 2 || y.len() != state_dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::NumericalCore,
                expected: state_dimension,
                actual: y.len(),
            });
        }
        if parameters.len() != plan.parameter_dimension() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::Continuation,
                expected: plan.parameter_dimension(),
                actual: parameters.len(),
            });
        }
        if y.iter().any(|value| !value.is_finite())
            || parameters.iter().any(|value| !value.is_finite())
        {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::NumericalCore,
            });
        }
        if !x.iter().all(|value| value.is_finite()) || !x.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(BvpSciNewError::InvalidConfiguration(
                "BVP mesh must be finite and strictly increasing".into(),
            ));
        }
        if options.tolerance <= 0.0 || !options.tolerance.is_finite() {
            return Err(BvpSciNewError::InvalidConfiguration(
                "BVP tolerance must be finite and positive".into(),
            ));
        }
        // SciPy clamps unrealistically small tolerances instead of allowing
        // roundoff to drive the mesh controller indefinitely.
        options.tolerance = options.tolerance.max(100.0 * f64::EPSILON);
        if options
            .boundary_tolerance
            .is_some_and(|value| value <= 0.0 || !value.is_finite())
        {
            return Err(BvpSciNewError::InvalidConfiguration(
                "BVP boundary tolerance must be finite and positive".into(),
            ));
        }
        if options.max_nodes < m
            || options.max_newton_iterations == 0
            || options.max_jacobian_refreshes == 0
        {
            return Err(BvpSciNewError::InvalidConfiguration(
                "BVP node/iteration/Jacobian budgets are inconsistent".into(),
            ));
        }
        let singular = options
            .singular_term
            .as_ref()
            .map(|term| term.prepare(x[0]))
            .transpose()?;
        if let Some(singular) = singular.as_ref() {
            singular.project_initial_state(&mut y[..n]);
        }
        let boundary_dimension = n.checked_add(parameters.len()).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP boundary dimension overflows usize".into())
        })?;
        if boundary.residual_dimension() != boundary_dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::BoundaryCallback,
                expected: boundary_dimension,
                actual: boundary.residual_dimension(),
            });
        }
        let boundary: Arc<dyn BvpSciBoundary> = Arc::new(boundary);
        let telemetry = plan.telemetry();
        let callback_jacobian_len = plan.jacobian_callback_scratch_len()?;
        // The global unknown vector is node-major state followed by the
        // parameter block. Every native backend receives this same ordering,
        // so Dense/Sparse/Banded differ only in storage and factorization.
        let workspace = BvpSciCollocationWorkspace::new_with_callback_jacobian_len(
            n,
            m,
            parameters.len(),
            callback_jacobian_len,
            telemetry,
        )?;
        let total_dimension = n
            .checked_mul(m)
            .and_then(|state_dimension| state_dimension.checked_add(parameters.len()))
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration(
                    "BVP global linear-system dimension overflows usize".into(),
                )
            })?;
        let backend = NativeLinearBackend::new_for_collocation(
            options.matrix_layout,
            total_dimension,
            plan.jacobian_nnz(),
            n,
            m,
            parameters.len(),
            telemetry.clone(),
        )?;
        let trial_state = y.clone();
        let trial_parameters = parameters.clone();
        Ok(Self {
            plan,
            boundary,
            options,
            x,
            y,
            parameters,
            workspace,
            backend,
            trial_state,
            trial_parameters,
            singular,
            continuation_pending: false,
        })
    }

    /// Inspect the prepared frontend without exposing mutable symbolic state.
    pub fn plan(&self) -> &BvpSciLambdifyPlan {
        &self.plan
    }

    /// Borrow the current mesh, node-major state and runtime parameters.
    pub fn state(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.x, &self.y, &self.parameters)
    }

    /// Rebind parameter values without rebuilding symbolic structures.
    pub fn set_parameters(&mut self, parameters: Vec<f64>) -> Result<(), BvpSciNewError> {
        if parameters.len() != self.plan.parameter_dimension() {
            return Err(BvpSciNewError::ContinuationMismatch);
        }
        if parameters.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::Continuation,
            });
        }
        self.parameters = parameters;
        self.plan.telemetry().record_parameter_rebind();
        self.continuation_pending = true;
        Ok(())
    }

    /// Restart the prepared model from a new mesh and initial state.
    ///
    /// The symbolic frontend and parameter schema remain intact. A workspace
    /// and native linear backend are reused when the mesh size is unchanged;
    /// changing the mesh rebuilds only numerical storage, never evaluators.
    pub fn restart(&mut self, x: Vec<f64>, mut y: Vec<f64>) -> Result<(), BvpSciNewError> {
        let n = self.plan.dimension();
        let state_dimension = n.checked_mul(x.len()).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration(
                "BVP restart state dimension overflows usize".into(),
            )
        })?;
        if x.len() < 2 || y.len() != state_dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::Continuation,
                expected: state_dimension,
                actual: y.len(),
            });
        }
        if !x.iter().all(|value| value.is_finite()) || !x.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(BvpSciNewError::InvalidConfiguration(
                "BVP restart mesh must be finite and strictly increasing".into(),
            ));
        }
        if let Some(singular) = self.singular.as_ref()
            && x[0] != singular.endpoint()
        {
            return Err(BvpSciNewError::InvalidConfiguration(
                "BVP restart left endpoint must match the prepared singular-term endpoint".into(),
            ));
        }
        if y.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::Continuation,
            });
        }
        if let Some(singular) = self.singular.as_ref() {
            // SciPy enforces S*y(a)=0 after every restart as well as after
            // every Newton trial. Do this before publishing the new state so
            // callbacks never observe an incompatible endpoint guess.
            singular.project_initial_state(&mut y[..n]);
        }

        if x.len() != self.x.len() {
            let layout = self.backend.layout();
            self.workspace
                .resize(n, x.len(), self.parameters.len(), self.plan.telemetry())?;
            let total_dimension = n
                .checked_mul(x.len())
                .and_then(|state_dimension| state_dimension.checked_add(self.parameters.len()))
                .ok_or_else(|| {
                    BvpSciNewError::InvalidConfiguration(
                        "BVP restart linear-system dimension overflows usize".into(),
                    )
                })?;
            self.backend = NativeLinearBackend::new_for_collocation(
                layout,
                total_dimension,
                self.plan.jacobian_nnz(),
                n,
                x.len(),
                self.parameters.len(),
                self.plan.telemetry().clone(),
            )?;
        }
        self.trial_state.resize(y.len(), 0.0);
        self.x = x;
        self.y = y;
        self.plan.telemetry().record_restart();
        Ok(())
    }

    /// Solve from the current state and retain the converged state for reuse.
    ///
    /// The outer loop is the adaptive collocation controller.  A mesh solve
    /// first converges with modified Newton, then samples the cubic Hermite
    /// defect away from collocation points.  Refinement only replaces numeric
    /// storage; the prepared symbolic plan and its evaluators remain intact.
    pub fn solve(&mut self) -> Result<BvpSciSolution, BvpSciNewError> {
        let _full_solve_timer = self.plan.telemetry().start_full_solve();
        if self.continuation_pending {
            self.plan.telemetry().record_continuation_solve();
            self.continuation_pending = false;
        }
        let boundary_tolerance = self
            .options
            .boundary_tolerance
            .unwrap_or(self.options.tolerance);
        let mut total_iterations = 0usize;

        for refinement in 0..=self.options.max_mesh_refinements {
            // A mesh pass has two phases: bounded Newton convergence on the
            // current grid, then a defect estimate that decides whether the
            // grid should grow. Symbolic preparation never belongs here.
            let (newton_outcome, iterations) = self.solve_current_mesh(boundary_tolerance)?;
            let norm = match newton_outcome {
                NewtonMeshOutcome::Converged { norm, .. }
                | NewtonMeshOutcome::BudgetExhausted { norm, .. } => norm,
            };
            total_iterations = total_iterations.checked_add(iterations).ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration(
                    "BVP total iteration count overflows usize".into(),
                )
            })?;
            let interval_errors = self.estimate_interval_defects()?;
            let max_error = interval_errors.iter().copied().fold(0.0, f64::max);
            let boundary_error = max_abs(&self.workspace.boundary_residual);
            if max_error <= self.options.tolerance && boundary_error <= boundary_tolerance {
                return self.solution(norm.max(max_error), total_iterations, interval_errors);
            }
            if refinement == self.options.max_mesh_refinements {
                if max_error <= self.options.tolerance && boundary_error > boundary_tolerance {
                    return Err(BvpSciNewError::BoundaryToleranceNotMet {
                        tolerance: boundary_tolerance,
                    });
                }
                return Err(BvpSciNewError::MeshRefinementLimit {
                    refinements: self.options.max_mesh_refinements,
                });
            }
            // When the defect is already within tolerance but the boundary
            // rows are not, SciPy retries Newton on the same mesh. Avoid
            // rebuilding the backend or resizing the workspace in that case.
            if max_error <= self.options.tolerance {
                continue;
            }
            self.refine_mesh(&interval_errors)?;
        }

        Err(BvpSciNewError::MeshRefinementLimit {
            refinements: self.options.max_mesh_refinements,
        })
    }

    /// Solve the nonlinear collocation equations on the current mesh.
    ///
    /// This is a bounded modified-Newton controller.  A factorized Jacobian
    /// is reused while trial steps reduce the residual sufficiently; poor
    /// reduction or failed backtracking invalidates it and consumes one of the
    /// explicit refreshes.  This avoids the old unconditional symbolic/numeric
    /// Jacobian work on every Newton iteration without making reuse unbounded.
    fn solve_current_mesh(
        &mut self,
        boundary_tolerance: f64,
    ) -> Result<(NewtonMeshOutcome, usize), BvpSciNewError> {
        let _newton_timer = self
            .plan
            .telemetry()
            .start_stage(BvpSciTelemetryStage::Newton);
        let mut norm = self.evaluate_residual()?;
        if !norm.is_finite() {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::NumericalCore,
            });
        }
        let mut jacobian_ready = false;
        let mut correction_ready = false;
        let mut jacobian_refreshes = 0usize;
        if self.scipy_converged(boundary_tolerance) {
            return Ok((
                NewtonMeshOutcome::Converged {
                    norm,
                    iterations: 0,
                },
                0,
            ));
        }

        for iteration in 0..self.options.max_newton_iterations {
            self.plan.telemetry().record_newton_iteration();
            let mut jacobian_refreshed = false;
            if !jacobian_ready {
                // One refresh builds the complete numeric global Jacobian and
                // factorization. Several trial corrections may reuse it;
                // refreshes are explicit because they are the expensive
                // operation this modified-Newton controller is designed to
                // limit.
                if jacobian_refreshes >= self.options.max_jacobian_refreshes {
                    return Ok((
                        NewtonMeshOutcome::BudgetExhausted {
                            norm,
                            iterations: iteration,
                        },
                        iteration,
                    ));
                }
                self.plan.telemetry().record_newton_jacobian_refresh();
                assemble_global_jacobian(
                    &self.plan,
                    self.boundary.as_ref(),
                    &self.x,
                    &self.y,
                    &self.parameters,
                    self.singular.as_ref(),
                    &mut self.workspace,
                )?;
                let linear_started = self.plan.telemetry().start_timing();
                self.backend.assemble(&self.workspace.entries)?;
                self.plan.telemetry().record_linear_assembly(linear_started);

                let factor_started = self.plan.telemetry().start_timing();
                self.backend.factor()?;
                self.plan.telemetry().record_factorization(factor_started);
                jacobian_refreshes += 1;
                jacobian_ready = true;
                correction_ready = false;
                jacobian_refreshed = true;
            }

            if !correction_ready {
                for (rhs, residual) in self.workspace.step.iter_mut().zip(&self.workspace.residual)
                {
                    *rhs = -*residual;
                }
                let solve_started = self.plan.telemetry().start_timing();
                self.backend.solve_in_place(&mut self.workspace.step)?;
                self.plan.telemetry().record_linear_solve(solve_started);
            }
            // SciPy's affine-invariant criterion is F = ||J^-1 r||^2. The
            // sign of the Newton correction is irrelevant to this norm, so
            // the current `-J^-1 r` step and the reference criterion agree.
            let cost = self
                .workspace
                .step
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let step_inf_norm = self
                .workspace
                .step
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);

            self.workspace.rollback_state.copy_from_slice(&self.y);
            self.workspace
                .rollback_parameters
                .copy_from_slice(&self.parameters);
            let previous_norm = norm;
            let mut accepted = false;
            let mut accepted_norm = None;
            let mut backtracking_trials = 0u64;
            let mut accepted_cost = None;
            let mut accepted_alpha = 0.0;
            let mut last_trial_norm = None;
            let mut last_trial_cost = None;
            let mut last_trial_alpha = 1.0;
            // SciPy evaluates the full step plus n_trial backtracking steps.
            // Thus the default n_trial=4 permits alpha=1, 1/2, 1/4, 1/8,
            // and 1/16. Keep the public option name for source compatibility.
            for trial in 0..=self.options.max_backtracking_steps {
                self.plan.telemetry().record_newton_backtracking_trial();
                backtracking_trials += 1;
                // Backtracking must shrink the Newton correction on every
                // retry.  A negative exponent would grow it (1, 2, 4, ...)
                // and can exhaust Jacobian refreshes on stiff workloads.
                let alpha = super::config::SCIPY_BACKTRACKING_TAU.powf(trial as f64);
                for (target, (base, delta)) in self.trial_state.iter_mut().zip(
                    self.workspace
                        .rollback_state
                        .iter()
                        .zip(&self.workspace.step),
                ) {
                    *target = *base + alpha * delta;
                }
                for (target, (base, delta)) in self.trial_parameters.iter_mut().zip(
                    self.workspace
                        .rollback_parameters
                        .iter()
                        .zip(&self.workspace.step[self.y.len()..]),
                ) {
                    *target = *base + alpha * delta;
                }
                if let Some(singular) = self.singular.as_ref() {
                    // The singular constraint is part of the iterate, not
                    // merely an initial-guess convenience. Without this
                    // projection accepted backtracking steps can leave the
                    // SciPy-compatible subspace at x=a.
                    singular.project_initial_state(&mut self.trial_state[..self.plan.dimension()]);
                }
                // A structured linear solve may return a finite but unusable
                // correction on an ill-conditioned Schur system. Treat a
                // non-finite trial as a rejected step so backtracking and a
                // fresh Jacobian get a chance; never let NaN poison the
                // controller's convergence comparison.
                let trial_norm = if self.trial_state.iter().all(|value| value.is_finite())
                    && self.trial_parameters.iter().all(|value| value.is_finite())
                {
                    evaluate_residual_buffers(
                        &self.plan,
                        self.boundary.as_ref(),
                        &self.x,
                        &self.trial_state,
                        &self.trial_parameters,
                        self.singular.as_ref(),
                        &mut self.workspace,
                    )?
                } else {
                    f64::INFINITY
                };
                let cost_new = if trial_norm.is_finite() {
                    self.workspace
                        .trial_step
                        .copy_from_slice(&self.workspace.residual);
                    let solve_started = self.plan.telemetry().start_timing();
                    self.backend
                        .solve_in_place(&mut self.workspace.trial_step)?;
                    self.plan.telemetry().record_linear_solve(solve_started);
                    Some(
                        self.workspace
                            .trial_step
                            .iter()
                            .map(|value| value * value)
                            .sum::<f64>(),
                    )
                } else {
                    None
                };
                let armijo_target = (1.0 - 2.0 * alpha * super::config::SCIPY_ARMIJO_SIGMA) * cost;
                if let Some(cost_new) = cost_new {
                    last_trial_norm = Some(trial_norm);
                    last_trial_cost = Some(cost_new);
                    last_trial_alpha = alpha;
                    if cost_new < armijo_target {
                        // Commit only after the complete residual, including
                        // boundary rows and parameter unknowns, accepts the same
                        // trial. Rollback buffers remain the last accepted state.
                        self.y.copy_from_slice(&self.trial_state);
                        self.parameters.copy_from_slice(&self.trial_parameters);
                        norm = trial_norm;
                        accepted_norm = Some(trial_norm);
                        accepted_cost = Some(cost_new);
                        accepted_alpha = alpha;
                        accepted = true;
                        self.plan.telemetry().record_newton_accepted_step();
                        break;
                    }
                }
            }
            // SciPy's reference loop commits the final trial even when none
            // satisfies Armijo. The next iteration then either converges or
            // recomputes the Jacobian after the damped step. Retaining this
            // behavior is important for stiff continuation cases; rejecting
            // the whole step here is a different, more conservative method.
            if !accepted {
                if let (Some(trial_norm), Some(cost_new)) = (last_trial_norm, last_trial_cost) {
                    self.y.copy_from_slice(&self.trial_state);
                    self.parameters.copy_from_slice(&self.trial_parameters);
                    norm = trial_norm;
                    accepted_norm = Some(trial_norm);
                    accepted_cost = Some(cost_new);
                    accepted_alpha = last_trial_alpha;
                    accepted = true;
                    self.plan.telemetry().record_newton_accepted_step();
                }
            }
            if !accepted {
                self.plan.telemetry().record_newton_rejected_step();
                self.plan.telemetry().record_newton_trace(
                    super::telemetry::BvpSciNewtonTraceEntry {
                        iteration: iteration as u64,
                        residual_before: previous_norm,
                        residual_after: None,
                        step_inf_norm,
                        affine_cost_before: cost,
                        affine_cost_after: last_trial_cost,
                        armijo_alpha: last_trial_alpha,
                        backtracking_trials,
                        jacobian_refreshed,
                        accepted: false,
                    },
                );
                // A rejected correction may have changed all callback scratch
                // buffers. Restore the accepted state and recompute its
                // residual before attempting a fresh Jacobian.
                self.y.copy_from_slice(&self.workspace.rollback_state);
                self.parameters
                    .copy_from_slice(&self.workspace.rollback_parameters);
                // Restore f/y_middle/residual to the accepted state before a
                // possible refresh; otherwise the next Jacobian would mix
                // the restored state with the rejected trial buffers.
                norm = self.evaluate_residual()?;
                if !norm.is_finite() {
                    return Err(BvpSciNewError::NonFinite {
                        stage: BvpSciStage::NumericalCore,
                    });
                }
                jacobian_ready = false;
                correction_ready = false;
                if jacobian_refreshes >= self.options.max_jacobian_refreshes {
                    return Ok((
                        NewtonMeshOutcome::BudgetExhausted {
                            norm,
                            iterations: iteration + 1,
                        },
                        iteration + 1,
                    ));
                }
                continue;
            }
            self.plan
                .telemetry()
                .record_newton_trace(super::telemetry::BvpSciNewtonTraceEntry {
                    iteration: iteration as u64,
                    residual_before: previous_norm,
                    residual_after: accepted_norm,
                    step_inf_norm,
                    affine_cost_before: cost,
                    affine_cost_after: accepted_cost,
                    armijo_alpha: accepted_alpha,
                    backtracking_trials,
                    jacobian_refreshed,
                    accepted: true,
                });
            // SciPy stops immediately after the fourth Jacobian refresh,
            // before starting another fixed-Jacobian iteration. Preserve
            // that lifecycle boundary and let the outer defect controller
            // decide the final status of the returned iterate.
            if jacobian_refreshes >= self.options.max_jacobian_refreshes {
                return Ok((
                    NewtonMeshOutcome::BudgetExhausted {
                        norm,
                        iterations: iteration + 1,
                    },
                    iteration + 1,
                ));
            }
            // A stale Jacobian remains useful for a strong contraction.  A
            // weak contraction is the explicit signal to refresh it.
            if accepted_alpha != 1.0 || accepted_cost.is_none() {
                // Weak contraction indicates that the local linearization is
                // stale whenever SciPy's line search needed damping. A full
                // step keeps the already factorized Jacobian and carries the
                // trial correction forward as the next fixed-Jacobian step.
                jacobian_ready = false;
                correction_ready = false;
            } else {
                for (step, trial_step) in self
                    .workspace
                    .step
                    .iter_mut()
                    .zip(&self.workspace.trial_step)
                {
                    // `trial_step` solves J * step_new = residual_new,
                    // whereas the state update convention stores
                    // `-J^-1 residual` in `step`.
                    *step = -*trial_step;
                }
                correction_ready = true;
            }
            if self.scipy_converged(boundary_tolerance) {
                return Ok((
                    NewtonMeshOutcome::Converged {
                        norm,
                        iterations: iteration + 1,
                    },
                    iteration + 1,
                ));
            }
        }
        Ok((
            NewtonMeshOutcome::BudgetExhausted {
                norm,
                iterations: self.options.max_newton_iterations,
            },
            self.options.max_newton_iterations,
        ))
    }

    /// Apply SciPy `_bvp.solve_newton`'s component-wise stopping contract.
    ///
    /// Collocation residuals are compared against
    /// `2/3 * h * 0.05 * bvp_tol * (1 + abs(f_middle))`; boundary rows use an
    /// independent absolute tolerance. This is intentionally separate from
    /// the RMS defect reported by the outer mesh controller.
    fn scipy_converged(&self, boundary_tolerance: f64) -> bool {
        let residual_ok =
            self.workspace
                .collocation_residual
                .chunks_exact(self.plan.dimension())
                .zip(self.workspace.h.iter())
                .zip(self.workspace.f_middle.chunks_exact(self.plan.dimension()))
                .all(|((residual, h), f_middle)| {
                    let tolerance = (2.0 / 3.0) * *h * 5.0e-2 * self.options.tolerance;
                    residual.iter().zip(f_middle).all(|(residual, f_middle)| {
                        residual.abs() < tolerance * (1.0 + f_middle.abs())
                    })
                });
        residual_ok && max_abs(&self.workspace.boundary_residual) < boundary_tolerance
    }

    /// Estimate SciPy-style relative RMS defects with 5-point Lobatto
    /// quadrature.  The midpoint residual is recovered from the collocation
    /// residual (`3/2 * r_collocation / h`), while the two off-midpoint
    /// Lobatto samples are evaluated through the cubic Hermite interpolant.
    /// This is deliberately outside the Newton hot path.
    fn estimate_interval_defects(&mut self) -> Result<Vec<f64>, BvpSciNewError> {
        let _defect_timer = self
            .plan
            .telemetry()
            .start_stage(BvpSciTelemetryStage::MeshDefectEstimation);
        let n = self.plan.dimension();
        let mut errors: Vec<f64> = vec![0.0; self.x.len() - 1];
        let offset = 0.5 * (3.0 / 7.0_f64).sqrt();
        for interval in 0..self.x.len() - 1 {
            let x_left = self.x[interval];
            let h = self.x[interval + 1] - x_left;
            let midpoint_sum = (0..n)
                .map(|component| {
                    let residual =
                        1.5 * self.workspace.collocation_residual[interval * n + component] / h;
                    let f_middle = self.workspace.f_middle[interval * n + component];
                    let normalized = residual / (1.0 + f_middle.abs());
                    normalized * normalized
                })
                .sum::<f64>();
            self.plan.telemetry().record_mesh_defect_probe();

            let mut off_midpoint_sums = [0.0; 2];
            for (sample, fraction) in [0.5 - offset, 0.5 + offset].into_iter().enumerate() {
                let x_probe = x_left + fraction * h;
                for component in 0..n {
                    let yl = self.y[interval * n + component];
                    let yr = self.y[(interval + 1) * n + component];
                    let fl = self.workspace.f_nodes[interval * n + component];
                    let fr = self.workspace.f_nodes[(interval + 1) * n + component];
                    self.trial_state[component] = hermite_value(fraction, h, yl, yr, fl, fr);
                }
                self.plan.evaluate_rhs(
                    x_probe,
                    &self.trial_state[..n],
                    &self.parameters,
                    &mut self.workspace.callback_arguments,
                    &mut self.workspace.callback_output,
                )?;
                if let Some(singular) = self.singular.as_ref() {
                    self.workspace.callback_rhs[..n]
                        .copy_from_slice(&self.workspace.callback_output[..n]);
                    singular.apply_rhs(
                        x_probe,
                        &self.trial_state[..n],
                        &self.workspace.callback_rhs[..n],
                        &mut self.workspace.callback_output[..n],
                    );
                    self.plan.telemetry().record_singular_term_application();
                }
                let mean = (0..n)
                    .map(|component| {
                        let yl = self.y[interval * n + component];
                        let yr = self.y[(interval + 1) * n + component];
                        let fl = self.workspace.f_nodes[interval * n + component];
                        let fr = self.workspace.f_nodes[(interval + 1) * n + component];
                        let derivative = hermite_derivative(fraction, h, yl, yr, fl, fr);
                        let rhs = self.workspace.callback_output[component];
                        let normalized = (derivative - rhs) / (1.0 + rhs.abs());
                        normalized * normalized
                    })
                    .sum::<f64>();
                self.plan.telemetry().record_mesh_defect_probe();
                off_midpoint_sums[sample] = mean;
            }
            let integral = 0.5
                * (32.0 / 45.0 * midpoint_sum
                    + 49.0 / 90.0 * (off_midpoint_sums[0] + off_midpoint_sums[1]));
            errors[interval] = integral.sqrt();
        }
        Ok(errors)
    }

    /// Insert two or three subintervals according to the defect estimator.
    /// Cubic Hermite interpolation supplies a continuation-quality initial
    /// guess, avoiding zero-filled states after refinement.
    fn refine_mesh(&mut self, errors: &[f64]) -> Result<usize, BvpSciNewError> {
        let started = self.plan.telemetry().start_timing();
        let old_nodes = self.x.len();
        let tolerance = self.options.tolerance;
        let subdivisions: Vec<usize> = errors
            .iter()
            .map(|error| {
                if *error > 100.0 * tolerance {
                    3
                } else if *error > tolerance {
                    2
                } else {
                    1
                }
            })
            .collect();
        let interval_count = subdivisions
            .iter()
            .try_fold(0usize, |sum, value| sum.checked_add(*value))
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration("BVP refined mesh size overflows usize".into())
            })?;
        let new_nodes = interval_count.checked_add(1).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP refined mesh size overflows usize".into())
        })?;
        // Keep this guard at the ownership boundary as well as in the outer
        // controller.  A future caller may pass an already-satisfied defect
        // vector; in that case no mesh, workspace, or linear backend should
        // be rebuilt and any prepared sparse symbolic analysis must survive.
        if new_nodes == old_nodes {
            return Ok(0);
        }
        if new_nodes > self.options.max_nodes {
            return Err(BvpSciNewError::MaxNodesExceeded {
                max_nodes: self.options.max_nodes,
            });
        }
        let n = self.plan.dimension();
        let new_state_len = new_nodes.checked_mul(n).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP refined state size overflows usize".into())
        })?;
        let mut new_x = Vec::with_capacity(new_nodes);
        let mut new_y = Vec::with_capacity(new_state_len);
        for interval in 0..old_nodes - 1 {
            let subdivisions = subdivisions[interval];
            let h = self.x[interval + 1] - self.x[interval];
            new_x.push(self.x[interval]);
            new_y.extend_from_slice(&self.y[interval * n..(interval + 1) * n]);
            for part in 1..subdivisions {
                let fraction = part as f64 / subdivisions as f64;
                new_x.push(self.x[interval] + fraction * h);
                for component in 0..n {
                    new_y.push(hermite_value(
                        fraction,
                        h,
                        self.y[interval * n + component],
                        self.y[(interval + 1) * n + component],
                        self.workspace.f_nodes[interval * n + component],
                        self.workspace.f_nodes[(interval + 1) * n + component],
                    ));
                }
            }
        }
        new_x.push(*self.x.last().ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP mesh is unexpectedly empty".into())
        })?);
        new_y.extend_from_slice(&self.y[(old_nodes - 1) * n..old_nodes * n]);

        let layout = self.backend.layout();
        self.workspace
            .resize(n, new_nodes, self.parameters.len(), self.plan.telemetry())?;
        let total_dimension = n
            .checked_mul(new_nodes)
            .and_then(|state_dimension| state_dimension.checked_add(self.parameters.len()))
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration(
                    "BVP refined linear-system dimension overflows usize".into(),
                )
            })?;
        self.backend = NativeLinearBackend::new_for_collocation(
            layout,
            total_dimension,
            self.plan.jacobian_nnz(),
            n,
            new_nodes,
            self.parameters.len(),
            self.plan.telemetry().clone(),
        )?;
        self.trial_state.resize(new_y.len(), 0.0);
        self.x = new_x;
        self.y = new_y;
        let points_added = new_nodes - old_nodes;
        self.plan
            .telemetry()
            .record_mesh_refinement(started, points_added);
        Ok(points_added)
    }

    fn evaluate_residual(&mut self) -> Result<f64, BvpSciNewError> {
        let plan = &self.plan;
        let boundary = self.boundary.as_ref();
        let x = &self.x;
        let y = &self.y;
        let parameters = &self.parameters;
        evaluate_residual_buffers(
            plan,
            boundary,
            x,
            y,
            parameters,
            self.singular.as_ref(),
            &mut self.workspace,
        )
    }

    fn solution(
        &self,
        residual_norm: f64,
        iterations: usize,
        interval_residuals: Vec<f64>,
    ) -> Result<BvpSciSolution, BvpSciNewError> {
        let _output_timer = self
            .plan
            .telemetry()
            .start_stage(BvpSciTelemetryStage::OutputConstruction);
        let dense_output = match self.options.output_policy {
            super::output::BvpSciOutputPolicy::DenseOutput => {
                let output = BvpSciDenseOutput::new(&self.x, &self.y, &self.workspace.f_nodes)?;
                self.plan.telemetry().record_dense_output_construction();
                Some(output)
            }
            _ => None,
        };
        Ok(BvpSciSolution {
            x: self.x.clone(),
            y: self.y.clone(),
            parameters: self.parameters.clone(),
            residual_norm,
            interval_residuals,
            status: super::error::BvpSciStatus::Success,
            message: "The algorithm converged to the desired accuracy.".into(),
            iterations,
            dense_output,
        })
    }
}

#[inline]
fn hermite_value(t: f64, h: f64, yl: f64, yr: f64, fl: f64, fr: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * yl
        + (t3 - 2.0 * t2 + t) * h * fl
        + (-2.0 * t3 + 3.0 * t2) * yr
        + (t3 - t2) * h * fr
}

#[inline]
fn hermite_derivative(t: f64, h: f64, yl: f64, yr: f64, fl: f64, fr: f64) -> f64 {
    let t2 = t * t;
    ((6.0 * t2 - 6.0 * t) * yl + (-6.0 * t2 + 6.0 * t) * yr) / h
        + (3.0 * t2 - 4.0 * t + 1.0) * fl
        + (3.0 * t2 - 2.0 * t) * fr
}

fn evaluate_residual_buffers(
    plan: &BvpSciLambdifyPlan,
    boundary: &dyn BvpSciBoundary,
    x: &[f64],
    y: &[f64],
    parameters: &[f64],
    singular: Option<&SingularTermRuntime>,
    workspace: &mut BvpSciCollocationWorkspace,
) -> Result<f64, BvpSciNewError> {
    // The residual vector has the same ordering as the global Newton unknown:
    // collocation rows first, then boundary rows. Keeping this ordering in
    // one helper ensures initial, trial and rollback evaluations are identical.
    evaluate_collocation(plan, x, y, parameters, singular, workspace)?;
    let n = plan.dimension();
    let ya = &y[..n];
    let yb = &y[(x.len() - 1) * n..x.len() * n];
    // Boundary evaluation is centralized here so callback telemetry is counted
    // once even when the Jacobian builder later probes boundary columns.
    evaluate_boundary(
        boundary,
        plan.telemetry(),
        ya,
        yb,
        parameters,
        &mut workspace.boundary_residual,
    )?;
    let interval_residual_len = (x.len() - 1) * n;
    workspace.residual[..interval_residual_len].copy_from_slice(&workspace.collocation_residual);
    workspace.residual[interval_residual_len..].copy_from_slice(&workspace.boundary_residual);
    let collocation_norm = collocation_rms(workspace);
    let boundary_norm = max_abs(&workspace.boundary_residual);
    Ok(collocation_norm.max(boundary_norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_sci::new::{
        BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciMatrixLayout, BvpSciNumericalPlan,
        BvpSciOutputPolicy, BvpSciSingularTerm, BvpSciStatus, BvpSciTelemetry,
    };
    use crate::symbolic::symbolic_engine::Expr;

    fn numerical_linear_plan(
        telemetry: BvpSciTelemetry,
        with_jacobian: bool,
    ) -> BvpSciLambdifyPlan {
        let plan = BvpSciNumericalPlan::new(
            1,
            0,
            |_x, _state, _parameters, output| {
                output[0] = 1.0;
                Ok(())
            },
            telemetry,
        )
        .unwrap();
        let plan = if with_jacobian {
            plan.with_rhs_jacobian(|_x, _state, _parameters, output| {
                output[0] = 0.0;
                Ok(())
            })
        } else {
            plan
        };
        BvpSciLambdifyPlan::prepare_numerical(plan)
    }

    fn numerical_linear_boundary(
        telemetry: BvpSciTelemetry,
        with_jacobian: bool,
    ) -> BvpSciBoundaryCallbacks {
        if with_jacobian {
            BvpSciBoundaryCallbacks::new_with_jacobian(
                1,
                |_ya, yb, _parameters, output| {
                    output[0] = yb[0] - 1.0;
                    Ok(())
                },
                |_ya, _yb, _parameters, dya, dyb, _dp| {
                    dya[0] = 0.0;
                    dyb[0] = 1.0;
                    Ok(())
                },
                telemetry,
            )
        } else {
            BvpSciBoundaryCallbacks::new(
                1,
                |_ya, yb, _parameters, output| {
                    output[0] = yb[0] - 1.0;
                    Ok(())
                },
                telemetry,
            )
        }
    }

    fn numerical_linear_solver(
        telemetry: BvpSciTelemetry,
        with_jacobian: bool,
        layout: BvpSciMatrixLayout,
    ) -> Result<BvpSciSolver, BvpSciNewError> {
        numerical_linear_solver_with_mesh(telemetry, with_jacobian, layout, vec![0.0, 0.5, 1.0])
    }

    fn numerical_linear_solver_with_mesh(
        telemetry: BvpSciTelemetry,
        with_jacobian: bool,
        layout: BvpSciMatrixLayout,
        mesh: Vec<f64>,
    ) -> Result<BvpSciSolver, BvpSciNewError> {
        let plan = numerical_linear_plan(telemetry.clone(), with_jacobian);
        let boundary = numerical_linear_boundary(telemetry, with_jacobian);
        let options = BvpSciOptions::default()
            .with_matrix_layout(layout)
            .with_tolerance(1e-8);
        BvpSciSolver::new(
            plan,
            boundary,
            mesh.clone(),
            vec![0.0; mesh.len()],
            vec![],
            options,
        )
    }

    #[test]
    fn numerical_dense_uses_supplied_rhs_and_boundary_jacobians() {
        let telemetry = BvpSciTelemetry::counters();
        let mut solver = numerical_linear_solver(telemetry, true, BvpSciMatrixLayout::Dense)
            .expect("analytical numerical route should construct");
        let solution = solver
            .solve()
            .expect("analytical numerical BVP should converge");
        assert!(solution.residual_norm < 1e-8);
        assert!((solution.y[2] - 1.0).abs() < 1e-8);
        let snapshot = solver.plan().telemetry_snapshot();
        assert_eq!(snapshot.finite_difference_probes, 0);
        assert!(snapshot.jacobian_evaluations > 0);
    }

    #[test]
    fn numerical_dense_falls_back_to_finite_difference_jacobians() {
        let telemetry = BvpSciTelemetry::counters();
        let mut solver = numerical_linear_solver(telemetry, false, BvpSciMatrixLayout::Dense)
            .expect("residual-only numerical route should construct");
        let solution = solver.solve().expect("FD numerical BVP should converge");
        assert!(solution.residual_norm < 1e-8);
        assert!((solution.y[2] - 1.0).abs() < 1e-8);
        let snapshot = solver.plan().telemetry_snapshot();
        assert!(snapshot.finite_difference_probes > 0);
    }

    #[test]
    fn numerical_route_rejects_non_dense_layouts_with_typed_error() {
        let sparse = match numerical_linear_solver(
            BvpSciTelemetry::disabled(),
            false,
            BvpSciMatrixLayout::Sparse,
        ) {
            Ok(_) => panic!("Numerical Sparse route must be rejected for now"),
            Err(error) => error,
        };
        assert!(matches!(sparse, BvpSciNewError::UnsupportedRoute(_)));

        let banded = match numerical_linear_solver(
            BvpSciTelemetry::disabled(),
            false,
            BvpSciMatrixLayout::Banded { lower: 1, upper: 1 },
        ) {
            Ok(_) => panic!("Numerical Banded route must be rejected for now"),
            Err(error) => error,
        };
        assert!(matches!(banded, BvpSciNewError::UnsupportedRoute(_)));
    }

    #[test]
    fn mesh_validation_rejects_non_finite_last_node_on_create_and_restart() {
        let constructor_error = match numerical_linear_solver_with_mesh(
            BvpSciTelemetry::disabled(),
            false,
            BvpSciMatrixLayout::Dense,
            vec![0.0, 0.5, f64::INFINITY],
        ) {
            Ok(_) => panic!("constructor must reject a non-finite final mesh node"),
            Err(error) => error,
        };
        assert!(matches!(
            constructor_error,
            BvpSciNewError::InvalidConfiguration(_)
        ));

        let mut solver = numerical_linear_solver(
            BvpSciTelemetry::disabled(),
            false,
            BvpSciMatrixLayout::Dense,
        )
        .expect("baseline solver should construct");
        let restart_error = solver
            .restart(vec![0.0, 0.5, f64::NEG_INFINITY], vec![0.0; 3])
            .expect_err("restart must reject a non-finite final mesh node");
        assert!(matches!(
            restart_error,
            BvpSciNewError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn numerical_fd_rejects_short_jacobian_scratch_without_panicking() {
        let plan = BvpSciNumericalPlan::new(
            2,
            0,
            |_x, _state, _parameters, output| {
                output.copy_from_slice(&[1.0, 2.0]);
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        let plan = BvpSciLambdifyPlan::prepare_numerical(plan);
        let mut arguments = vec![0.0; 3];
        let mut output = vec![0.0; 4];
        let mut short_scratch = vec![0.0; 2];
        let error = plan
            .evaluate_jacobian_dense_with_scratch(
                0.0,
                &[0.0, 0.0],
                &[],
                &mut arguments,
                &mut output,
                &mut short_scratch,
            )
            .expect_err("FD must reject scratch shorter than two RHS vectors");
        assert!(matches!(
            error,
            BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected: 4,
                actual: 2,
            }
        ));
    }

    fn linear_problem(assembly: BvpSciAssembly, layout: BvpSciMatrixLayout) -> BvpSciSolver {
        let plan = BvpSciLambdifyPlan::prepare(
            assembly,
            &[Expr::parse_expression("1"), Expr::parse_expression("1")],
            &["y0".into(), "y1".into()],
            &[],
            "x",
            BvpSciTelemetry::counters(),
        )
        .unwrap();
        let boundary = BvpSciBoundaryCallbacks::new(
            2,
            |ya, yb, _, output| {
                output[0] = ya[0];
                output[1] = yb[1] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::counters(),
        );
        let mut options = BvpSciOptions::default();
        options.matrix_layout = layout;
        options.tolerance = 1e-10;
        BvpSciSolver::new(
            plan,
            boundary,
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![],
            options,
        )
        .unwrap()
    }

    #[test]
    fn linear_bvp_converges_with_dense_backend() {
        let mut solver = linear_problem(BvpSciAssembly::ExprLegacy, BvpSciMatrixLayout::Dense);
        let solution = solver.solve().unwrap();
        assert!(solution.residual_norm < 1e-10);
        assert!((solution.y[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn linear_bvp_converges_with_sparse_backend() {
        let mut solver = linear_problem(BvpSciAssembly::ExprLegacy, BvpSciMatrixLayout::Sparse);
        let solution = solver.solve().unwrap();
        assert!(solution.residual_norm < 1e-10);
        assert!((solution.y[4] - 1.0).abs() < 1e-10);
        let telemetry = solver.plan().telemetry_snapshot();
        assert_eq!(telemetry.sparse_symbolic_analyses, 1);
        assert!(telemetry.sparse_numeric_factorizations >= 1);
    }

    #[test]
    fn linear_bvp_converges_with_banded_backend() {
        let mut solver = linear_problem(
            BvpSciAssembly::ExprLegacy,
            BvpSciMatrixLayout::Banded { lower: 5, upper: 5 },
        );
        let solution = solver.solve().unwrap();
        assert!(solution.residual_norm < 1e-10);
        assert!((solution.y[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn atom_native_converges_with_all_native_layouts() {
        for layout in [
            BvpSciMatrixLayout::Dense,
            BvpSciMatrixLayout::Sparse,
            BvpSciMatrixLayout::Banded { lower: 5, upper: 5 },
        ] {
            let mut solver = linear_problem(BvpSciAssembly::AtomViewNative, layout);
            let solution = solver.solve().expect("AtomNative BVP should converge");
            assert!(solution.residual_norm < 1e-10);
            assert!((solution.y[4] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn aot_selection_returns_typed_unsupported_route_until_frontend_exists() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[crate::symbolic::symbolic_engine::Expr::parse_expression(
                "1",
            )],
            &["y".into()],
            &[],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        let boundary = BvpSciBoundaryCallbacks::new(
            1,
            |_, yb, _, output| {
                output[0] = yb[0] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let mut options = BvpSciOptions::default();
        options.execution = super::super::config::BvpSciExecution::Aot;
        let result = BvpSciSolver::new(
            plan,
            boundary,
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            vec![],
            options,
        );
        let error = match result {
            Ok(_) => panic!("AOT placeholder must not silently select Lambdify"),
            Err(error) => error,
        };
        assert!(matches!(error, BvpSciNewError::UnsupportedRoute(_)));
    }

    #[test]
    fn parameter_rebind_reuses_atom_native_preparation() {
        use std::sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        };

        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::AtomViewNative,
            &[Expr::parse_expression("y+p")],
            &["y".into()],
            &["p".into()],
            "x",
            BvpSciTelemetry::counters(),
        )
        .expect("parameterized AtomNative plan should prepare");
        let target = Arc::new(AtomicU64::new((1.0f64.exp() - 1.0).to_bits()));
        let target_for_boundary = Arc::clone(&target);
        let boundary = BvpSciBoundaryCallbacks::new(
            2,
            move |ya, yb, _, output| {
                let target = f64::from_bits(target_for_boundary.load(Ordering::Relaxed));
                // The exact family is y(x) = p * (exp(x) - 1), so the
                // left boundary is homogeneous and the parameter controls
                // the right endpoint.  This keeps the continuation fixture
                // consistent with its stated analytic solution.
                output[0] = ya[0];
                output[1] = yb[0] - target;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let mut options = BvpSciOptions::default();
        options.matrix_layout = BvpSciMatrixLayout::Dense;
        // This fixture exercises continuation, not machine-precision mesh
        // refinement.  The adaptive defect is a continuous error estimate;
        // asking it for 1e-10 would legitimately consume the node budget for
        // the exponential solution.
        options.tolerance = 1e-3;
        let mut solver = BvpSciSolver::new(
            plan,
            boundary,
            vec![0.0, 0.5, 1.0],
            vec![0.0, 1.0f64.exp().sqrt() - 1.0, 1.0f64.exp() - 1.0],
            vec![1.0],
            options,
        )
        .expect("parameterized AtomNative solver should construct");

        let first = solver
            .solve()
            .expect("first parameter solve should converge");
        assert!((first.y[first.y.len() - 1] - (1.0f64.exp() - 1.0)).abs() < 1e-10);
        target.store((2.0 * (1.0f64.exp() - 1.0)).to_bits(), Ordering::Relaxed);
        solver
            .set_parameters(vec![2.0])
            .expect("parameter rebind should preserve shape");
        let second = solver
            .solve()
            .expect("continued parameter solve should converge");
        assert!(
            (second.y[second.y.len() - 1] - 2.0 * (1.0f64.exp() - 1.0)).abs() < 1e-8,
            "second endpoint={} expected={}",
            second.y[second.y.len() - 1],
            2.0 * (1.0f64.exp() - 1.0)
        );

        let telemetry = solver.plan().telemetry_snapshot();
        assert_eq!(telemetry.atom_conversions, 1);
        assert_eq!(telemetry.pattern_entries, 1);
        assert_eq!(telemetry.parameter_rebinds, 1);
        assert!(telemetry.newton_iterations > 0);
    }

    #[test]
    fn adaptive_controller_refines_and_reuses_modified_newton_factorization() {
        let telemetry = BvpSciTelemetry::counters();
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[Expr::parse_expression("0.5*y*y")],
            &["y".into()],
            &[],
            "x",
            telemetry,
        )
        .unwrap();
        let boundary = BvpSciBoundaryCallbacks::new(
            1,
            |ya, _yb, _, output| {
                output[0] = ya[0] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let mut options = BvpSciOptions::default();
        options.tolerance = 1e-3;
        options.max_nodes = 128;
        options.max_newton_iterations = 12;
        options.max_jacobian_refreshes = 6;
        let mut solver = BvpSciSolver::new(
            plan,
            boundary,
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![],
            options,
        )
        .unwrap();
        let solution = solver.solve().unwrap();
        assert!(solution.x.len() > 2);
        let snapshot = solver.plan().telemetry_snapshot();
        assert!(snapshot.mesh_refinements > 0);
        assert!(snapshot.mesh_defect_probes > 0);
        assert!(snapshot.collocation_evaluations > 0);
        assert!(snapshot.factorizations <= snapshot.newton_iterations);
    }

    #[test]
    fn modified_newton_trace_preserves_controller_invariants() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[Expr::parse_expression("0.5*y*y")],
            &["y".into()],
            &[],
            "x",
            BvpSciTelemetry::timings(),
        )
        .unwrap();
        let boundary = BvpSciBoundaryCallbacks::new(
            1,
            |ya, _yb, _, output| {
                output[0] = ya[0] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let mut options = BvpSciOptions::default();
        options.tolerance = 1e-3;
        options.max_nodes = 128;
        options.max_newton_iterations = 12;
        options.max_jacobian_refreshes = 6;
        options.max_backtracking_steps = 10;
        let mut solver = BvpSciSolver::new(
            plan,
            boundary,
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![],
            options.clone(),
        )
        .unwrap();

        solver
            .solve()
            .expect("bounded nonlinear fixture should converge");
        let snapshot = solver.plan().telemetry_snapshot();
        assert!(snapshot.validate_contract().is_ok());
        assert!(!snapshot.newton_residual_history.is_empty());
        assert!(snapshot.newton_jacobian_refreshes <= options.max_jacobian_refreshes as u64);
        assert!(snapshot.newton_backtracking_trials >= snapshot.newton_iterations);

        for entry in snapshot.newton_residual_history {
            assert!(entry.residual_before.is_finite());
            assert!(entry.step_inf_norm.is_finite());
            assert!(entry.affine_cost_before.is_finite());
            if let Some(cost_after) = entry.affine_cost_after {
                assert!(cost_after.is_finite());
            }
            assert!((0.0..=1.0).contains(&entry.armijo_alpha));
            assert!(entry.backtracking_trials > 0);
            if entry.accepted {
                let residual_after = entry
                    .residual_after
                    .expect("accepted Newton step must record its residual");
                assert!(residual_after.is_finite());
                assert!(
                    residual_after <= entry.residual_before,
                    "accepted Newton step increased residual: before={} after={}",
                    entry.residual_before,
                    residual_after
                );
            } else {
                assert!(
                    entry.residual_after.is_none(),
                    "rejected Newton step must not publish a committed residual"
                );
            }
        }
    }

    #[test]
    fn scipy_controller_defaults_match_reference_contract() {
        use super::super::config::{
            SCIPY_ARMIJO_SIGMA, SCIPY_BACKTRACKING_TAU, SCIPY_MAX_BACKTRACKING_TRIALS,
            SCIPY_MAX_JACOBIAN_REFRESHES, SCIPY_MAX_MESH_ITERATIONS, SCIPY_MAX_NEWTON_ITERATIONS,
        };

        let options = BvpSciOptions::default();
        assert_eq!(options.max_jacobian_refreshes, SCIPY_MAX_JACOBIAN_REFRESHES);
        assert_eq!(options.max_newton_iterations, SCIPY_MAX_NEWTON_ITERATIONS);
        assert_eq!(options.max_mesh_refinements, SCIPY_MAX_MESH_ITERATIONS);
        assert_eq!(
            options.max_backtracking_steps,
            SCIPY_MAX_BACKTRACKING_TRIALS
        );
        assert_eq!(SCIPY_ARMIJO_SIGMA, 0.2);
        assert_eq!(SCIPY_BACKTRACKING_TAU, 0.5);
    }

    #[test]
    fn adaptive_controller_reports_node_budget_without_panic() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[Expr::parse_expression("0.5*y*y")],
            &["y".into()],
            &[],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        let boundary = BvpSciBoundaryCallbacks::new(
            1,
            |ya, _, _, output| {
                output[0] = ya[0] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let mut options = BvpSciOptions::default();
        options.tolerance = 1e-8;
        options.max_nodes = 2;
        options.max_newton_iterations = 20;
        options.max_jacobian_refreshes = 20;
        options.max_backtracking_steps = 10;
        let result = BvpSciSolver::new(
            plan,
            boundary,
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![],
            options,
        )
        .unwrap()
        .solve();
        assert!(matches!(
            result,
            Err(BvpSciNewError::MaxNodesExceeded { max_nodes: 2 })
                | Err(BvpSciNewError::MeshRefinementLimit { .. })
                | Err(BvpSciNewError::NewtonFailure { .. })
        ));
    }

    #[test]
    fn restart_reuses_prepared_atom_native_model() {
        let mut solver = linear_problem(BvpSciAssembly::AtomViewNative, BvpSciMatrixLayout::Sparse);
        let first = solver.solve().expect("initial solve should converge");
        solver
            .restart(vec![0.0, 0.5, 1.0], vec![0.0; 6])
            .expect("same-size restart should preserve prepared model");
        let second = solver.solve().expect("restarted solve should converge");
        assert!((first.y[4] - second.y[4]).abs() < 1e-10);
        let telemetry = solver.plan().telemetry_snapshot();
        assert_eq!(telemetry.atom_conversions, 2);
        assert_eq!(telemetry.restarts, 1);
    }

    #[test]
    fn timing_telemetry_separates_linear_stages() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[Expr::parse_expression("1"), Expr::parse_expression("1")],
            &["y0".into(), "y1".into()],
            &[],
            "x",
            BvpSciTelemetry::timings(),
        )
        .unwrap();
        let telemetry = plan.telemetry().clone();
        let boundary = BvpSciBoundaryCallbacks::new(
            2,
            |ya, yb, _, output| {
                output[0] = ya[0];
                output[1] = yb[1] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let mut options = BvpSciOptions::default();
        options.matrix_layout = BvpSciMatrixLayout::Dense;
        options.tolerance = 1e-10;
        let mut solver = BvpSciSolver::new(
            BvpSciLambdifyPlan::prepare(
                BvpSciAssembly::ExprLegacy,
                &[Expr::parse_expression("1"), Expr::parse_expression("1")],
                &["y0".into(), "y1".into()],
                &[],
                "x",
                telemetry,
            )
            .unwrap(),
            boundary,
            vec![0.0, 0.5, 1.0],
            vec![0.0; 6],
            vec![],
            options,
        )
        .unwrap();
        solver.solve().unwrap();
        let snapshot = solver.plan().telemetry_snapshot();
        assert!(snapshot.boundary_calls > 0);
        assert!(snapshot.callback_ms.is_some());
        assert!(snapshot.linear_assemblies > 0);
        assert!(snapshot.factorizations > 0);
        assert!(snapshot.linear_solves > 0);
        assert!(snapshot.linear_assembly_ms.is_some());
        assert!(snapshot.factorization_ms.is_some());
        assert!(snapshot.solve_ms.is_some());
        assert!(snapshot.jacobian_output_assembly_ms.is_some());
    }

    #[test]
    fn scipy_status_and_dense_output_are_exposed_without_recomputation() {
        let mut solver = linear_problem(BvpSciAssembly::ExprLegacy, BvpSciMatrixLayout::Dense);
        solver.options.output_policy = BvpSciOutputPolicy::DenseOutput;
        let solution = solver.solve().expect("linear BVP should converge");
        assert_eq!(solution.status, BvpSciStatus::Success);
        assert_eq!(solution.status.code(), 0);
        assert_eq!(solution.interval_residuals.len(), solution.x.len() - 1);
        let output = solution
            .dense_output()
            .expect("dense output should be present");
        let value = output.evaluate(0.25).expect("query should be in domain");
        let derivative = output
            .evaluate_derivative(0.25)
            .expect("derivative query should be in domain");
        assert!((value[0] - 0.25).abs() < 1e-10);
        assert!((value[1] - 0.25).abs() < 1e-10);
        assert!((derivative[0] - 1.0).abs() < 1e-10);
        let mut value_buffer = [0.0; 2];
        let mut derivative_buffer = [0.0; 2];
        output
            .evaluate_into(0.25, &mut value_buffer)
            .expect("buffered value query should be in domain");
        output
            .evaluate_derivative_into(0.25, &mut derivative_buffer)
            .expect("buffered derivative query should be in domain");
        assert_eq!(value_buffer.as_slice(), value.as_slice());
        assert_eq!(derivative_buffer.as_slice(), derivative.as_slice());
        assert!(matches!(
            output.evaluate_into(0.25, &mut [0.0]),
            Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::Output,
                ..
            })
        ));
        assert!(matches!(
            output.evaluate(-1.0),
            Err(BvpSciNewError::OutputOutOfDomain)
        ));
    }

    #[test]
    fn singular_term_is_applied_to_rhs_and_jacobian_for_all_layouts() {
        for layout in [
            BvpSciMatrixLayout::Dense,
            BvpSciMatrixLayout::Sparse,
            BvpSciMatrixLayout::Banded { lower: 3, upper: 3 },
        ] {
            let plan = BvpSciLambdifyPlan::prepare(
                BvpSciAssembly::ExprLegacy,
                &[Expr::parse_expression("0.5")],
                &["y".into()],
                &[],
                "x",
                BvpSciTelemetry::counters(),
            )
            .unwrap();
            let boundary = BvpSciBoundaryCallbacks::new(
                1,
                |ya, yb, _, output| {
                    let _ = ya;
                    output[0] = yb[0] - 1.0;
                    Ok(())
                },
                BvpSciTelemetry::disabled(),
            );
            let mut options = BvpSciOptions::default();
            options.matrix_layout = layout;
            options.tolerance = 1e-7;
            options.max_nodes = 64;
            options.singular_term = Some(BvpSciSingularTerm::new(1, vec![0.5]).unwrap());
            let mut solver = BvpSciSolver::new(
                plan,
                boundary,
                vec![0.0, 0.5, 1.0],
                vec![0.0, 0.5, 1.0],
                vec![],
                options,
            )
            .unwrap();
            let solution = solver.solve().expect("singular BVP should converge");
            assert_eq!(solution.status, BvpSciStatus::Success);
            assert!((solution.y[solution.y.len() - 1] - 1.0).abs() < 1e-7);

            let restart_error = match solver.restart(vec![0.1, 0.55, 1.0], vec![0.0, 0.5, 1.0]) {
                Ok(_) => panic!("singular restart must preserve the prepared endpoint"),
                Err(error) => error,
            };
            assert!(matches!(
                restart_error,
                BvpSciNewError::InvalidConfiguration(_)
            ));
        }
    }
}
