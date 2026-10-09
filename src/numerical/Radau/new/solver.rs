//! Adaptive Radau production facade.
//!
//! The driver owns only integration state and retry policy. Newton iteration,
//! callbacks, linear algebra, and symbolic frontend work remain in the
//! already-tested step/session layers.

use super::callbacks::{
    DenseJacobianCallback, ResidualCallback, SymbolicCallbackSession, validate_callback_output,
};
use super::config::{RadauConfig, RadauMatrixLayout};
use super::controller::{
    clamp_step_to_bound, initial_probe_step, is_finished, min_step, predict_factor, safety_factor,
    select_initial_step_from_probes,
};
use super::error::{RadauError, RadauStage};
use super::linear::PreparedLinearBackend;
use super::output::{RadauOutput, RadauOutputCollector};
use super::session::RadauSession;
use super::state::RadauSolverState;
use super::step::{
    RadauStepResult, try_radau5_step_with_rejection, try_radau5_symbolic_step_with_backend,
};
use super::telemetry::{RadauAdaptiveStepTrace, RadauCallbackStage};
use super::workspace::RadauWorkspace;

/// Final state and basic lifecycle counters for one adaptive solve.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RadauSolveResult {
    pub(crate) t: f64,
    pub(crate) y: Vec<f64>,
    pub(crate) attempts: usize,
    pub(crate) accepted_steps: usize,
    pub(crate) rejected_steps: usize,
}

/// Solve a dense problem through a prepared numeric session.
pub(crate) fn try_solve_dense<R, J>(
    config: &RadauConfig,
    session: &mut RadauSession,
    residual: &mut R,
    jacobian: &mut J,
    y0: &[f64],
) -> Result<RadauSolveResult, RadauError>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    config.validate()?;
    if y0.len() != session.prepared().key().dimension {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: session.prepared().key().dimension,
            actual: y0.len(),
        });
    }
    let mut runner = SessionStepRunner {
        config,
        session,
        residual,
        jacobian,
    };
    run_adaptive_dense(config, y0, &mut runner)
}

/// Solve a dense symbolic problem without rebuilding its prepared frontend.
pub(crate) fn try_solve_symbolic_dense(
    config: &RadauConfig,
    callbacks: &mut SymbolicCallbackSession<'_>,
    y0: &[f64],
) -> Result<RadauSolveResult, RadauError> {
    config.validate()?;
    callbacks.telemetry_mut().set_mode(config.telemetry);
    if y0.is_empty() {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: 1,
            actual: 0,
        });
    }
    // Prepare the symbolic pattern and native linear workspace before the
    // adaptive loop.  This is the continuation/performance boundary: retries
    // and accepted steps reuse these objects instead of rebuilding symbolic
    // expressions or sparse/banded structure.
    let backend = PreparedLinearBackend::from_layout_and_pattern(
        config.matrix_layout,
        callbacks.jacobian_pattern(),
    );
    let mut workspace = RadauWorkspace::default();
    let sparse_pattern = match &backend {
        PreparedLinearBackend::Sparse(backend) => backend.pattern.as_slice(),
        _ => &[],
    };
    workspace.resize_for_layout_with_pattern(y0.len(), backend.layout(), sparse_pattern)?;
    let mut runner = SymbolicStepRunner {
        config,
        callbacks,
        backend,
        workspace,
    };
    let result = run_adaptive_dense(config, y0, &mut runner);
    // Numerical telemetry belongs to the runner workspace while the solve is
    // active.  Move its snapshot out only after the runner is dropped, then
    // publish one complete report through the symbolic session.
    let runtime_telemetry = runner.workspace.telemetry.clone();
    drop(runner);
    callbacks.telemetry_mut().absorb_runtime(&runtime_telemetry);
    result
}

/// Symbolic solve variant that returns the explicitly requested trajectory
/// output without changing the allocation-free final-state path.
pub(crate) fn try_solve_symbolic_dense_with_output(
    config: &RadauConfig,
    callbacks: &mut SymbolicCallbackSession<'_>,
    y0: &[f64],
) -> Result<(RadauSolveResult, RadauOutput), RadauError> {
    config.validate()?;
    callbacks.telemetry_mut().set_mode(config.telemetry);
    if y0.is_empty() {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: 1,
            actual: 0,
        });
    }
    let backend = PreparedLinearBackend::from_layout_and_pattern(
        config.matrix_layout,
        callbacks.jacobian_pattern(),
    );
    let mut workspace = RadauWorkspace::default();
    let sparse_pattern = match &backend {
        PreparedLinearBackend::Sparse(backend) => backend.pattern.as_slice(),
        _ => &[],
    };
    workspace.resize_for_layout_with_pattern(y0.len(), backend.layout(), sparse_pattern)?;
    let mut runner = SymbolicStepRunner {
        config,
        callbacks,
        backend,
        workspace,
    };
    let mut collector = RadauOutputCollector::new(config.output.clone(), y0.len())?;
    let result = run_adaptive_dense_with_output(config, y0, &mut runner, Some(&mut collector));
    let runtime_telemetry = runner.workspace.telemetry.clone();
    drop(runner);
    callbacks.telemetry_mut().absorb_runtime(&runtime_telemetry);
    Ok((result?, collector.finish()?))
}

trait AdaptiveStepRunner {
    fn select_initial_step(
        &mut self,
        config: &RadauConfig,
        t: f64,
        direction: f64,
        y: &[f64],
    ) -> Result<f64, RadauError>;
    fn try_step(
        &mut self,
        t: f64,
        h: f64,
        y: &[f64],
        output: &mut [f64],
        rejected: bool,
    ) -> Result<RadauStepResult, RadauError>;

    fn record_step(&mut self, accepted: bool);
    fn record_initial_step(&mut self, h_abs: f64);
    fn record_adaptive_step(&mut self, trace: RadauAdaptiveStepTrace);
    fn dense_output_segment(
        &mut self,
    ) -> Result<Option<super::dense_output::RadauDenseOutputSegment>, RadauError>;
    fn refresh_jacobian(&mut self, t: f64, y: &[f64]) -> Result<(), RadauError>;
    fn jacobian_current(&mut self) -> bool;
    fn mark_jacobian_stale(&mut self);
    fn invalidate_factor(&mut self);
}

struct SessionStepRunner<'a, R, J> {
    config: &'a RadauConfig,
    session: &'a mut RadauSession,
    residual: &'a mut R,
    jacobian: &'a mut J,
}

impl<R, J> AdaptiveStepRunner for SessionStepRunner<'_, R, J>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    fn select_initial_step(
        &mut self,
        config: &RadauConfig,
        t: f64,
        direction: f64,
        y: &[f64],
    ) -> Result<f64, RadauError> {
        let workspace = self.session.workspace();
        self.residual.eval(t, y, &mut workspace.base_rhs)?;
        validate_callback_output(RadauStage::Residual, y.len(), &workspace.base_rhs)?;
        let h0 = initial_probe_step(config, y, &workspace.base_rhs);
        for index in 0..y.len() {
            workspace.rollback_state[index] = y[index] + direction * h0 * workspace.base_rhs[index];
        }
        self.residual.eval(
            t + direction * h0,
            &workspace.rollback_state,
            &mut workspace.stage_rhs[..y.len()],
        )?;
        validate_callback_output(
            RadauStage::Residual,
            y.len(),
            &workspace.stage_rhs[..y.len()],
        )?;
        let selected = select_initial_step_from_probes(
            config,
            y,
            &workspace.base_rhs,
            &workspace.stage_rhs[..y.len()],
            h0,
        );
        workspace.preload_base_rhs();
        Ok(selected)
    }

    fn try_step(
        &mut self,
        t: f64,
        h: f64,
        y: &[f64],
        output: &mut [f64],
        rejected: bool,
    ) -> Result<RadauStepResult, RadauError> {
        self.session.try_dense_step_with_rejection(
            self.config,
            self.residual,
            self.jacobian,
            t,
            h,
            y,
            output,
            rejected,
        )
    }

    fn record_step(&mut self, accepted: bool) {
        self.session.workspace().telemetry.count_step(accepted);
        if accepted {
            self.session.workspace().clear_base_rhs_preload();
            self.session.workspace().commit_dense_output();
        } else {
            self.session.workspace().discard_dense_output();
            self.session.workspace().invalidate_factor();
        }
    }

    fn record_initial_step(&mut self, h_abs: f64) {
        self.session
            .workspace()
            .telemetry
            .record_initial_step(h_abs);
    }

    fn record_adaptive_step(&mut self, trace: RadauAdaptiveStepTrace) {
        self.session
            .workspace()
            .telemetry
            .record_adaptive_step(trace);
    }

    fn dense_output_segment(
        &mut self,
    ) -> Result<Option<super::dense_output::RadauDenseOutputSegment>, RadauError> {
        self.session.workspace().dense_output_segment()
    }

    fn refresh_jacobian(&mut self, t: f64, y: &[f64]) -> Result<(), RadauError> {
        self.jacobian
            .eval_into(t, y, &mut self.session.workspace().jacobian)?;
        super::callbacks::validate_callback_output(
            RadauStage::Jacobian,
            self.session.workspace().jacobian.len(),
            &self.session.workspace().jacobian,
        )?;
        self.session.workspace().mark_jacobian_evaluated();
        Ok(())
    }

    fn jacobian_current(&mut self) -> bool {
        self.session.workspace().jacobian_current()
    }

    fn mark_jacobian_stale(&mut self) {
        self.session.workspace().mark_jacobian_stale();
    }

    fn invalidate_factor(&mut self) {
        self.session.workspace().invalidate_factor();
    }
}

struct SymbolicStepRunner<'config, 'callbacks> {
    config: &'config RadauConfig,
    callbacks: &'config mut SymbolicCallbackSession<'callbacks>,
    backend: PreparedLinearBackend,
    workspace: RadauWorkspace,
}

impl<'config, 'callbacks> AdaptiveStepRunner for SymbolicStepRunner<'config, 'callbacks> {
    fn select_initial_step(
        &mut self,
        config: &RadauConfig,
        t: f64,
        direction: f64,
        y: &[f64],
    ) -> Result<f64, RadauError> {
        self.callbacks
            .evaluate_residual(t, y, &mut self.workspace.base_rhs)?;
        validate_callback_output(RadauStage::Residual, y.len(), &self.workspace.base_rhs)?;
        let h0 = initial_probe_step(config, y, &self.workspace.base_rhs);
        for index in 0..y.len() {
            self.workspace.rollback_state[index] =
                y[index] + direction * h0 * self.workspace.base_rhs[index];
        }
        self.callbacks.evaluate_residual(
            t + direction * h0,
            &self.workspace.rollback_state,
            &mut self.workspace.stage_rhs[..y.len()],
        )?;
        validate_callback_output(
            RadauStage::Residual,
            y.len(),
            &self.workspace.stage_rhs[..y.len()],
        )?;
        let selected = select_initial_step_from_probes(
            config,
            y,
            &self.workspace.base_rhs,
            &self.workspace.stage_rhs[..y.len()],
            h0,
        );
        self.workspace.preload_base_rhs();
        Ok(selected)
    }

    fn try_step(
        &mut self,
        t: f64,
        h: f64,
        y: &[f64],
        output: &mut [f64],
        rejected: bool,
    ) -> Result<RadauStepResult, RadauError> {
        try_radau5_symbolic_step_with_backend(
            self.config,
            &self.backend,
            self.callbacks,
            t,
            h,
            y,
            output,
            &mut self.workspace,
            rejected,
        )
    }

    fn record_step(&mut self, accepted: bool) {
        self.callbacks.telemetry_mut().count_step(accepted);
        if accepted {
            self.workspace.clear_base_rhs_preload();
            self.workspace.commit_dense_output();
        } else {
            self.workspace.discard_dense_output();
            self.workspace.invalidate_factor();
        }
    }

    fn record_initial_step(&mut self, h_abs: f64) {
        self.callbacks.telemetry_mut().record_initial_step(h_abs);
    }

    fn record_adaptive_step(&mut self, trace: RadauAdaptiveStepTrace) {
        self.callbacks.telemetry_mut().record_adaptive_step(trace);
    }

    fn dense_output_segment(
        &mut self,
    ) -> Result<Option<super::dense_output::RadauDenseOutputSegment>, RadauError> {
        self.workspace.dense_output_segment()
    }

    fn refresh_jacobian(&mut self, t: f64, y: &[f64]) -> Result<(), RadauError> {
        self.callbacks.evaluate_jacobian_layout(
            t,
            y,
            self.backend.layout(),
            &mut self.workspace.jacobian,
        )?;
        super::callbacks::validate_callback_output(
            RadauStage::Jacobian,
            self.workspace.jacobian.len(),
            &self.workspace.jacobian,
        )?;
        self.workspace.mark_jacobian_evaluated();
        Ok(())
    }

    fn jacobian_current(&mut self) -> bool {
        self.workspace.jacobian_current()
    }

    fn mark_jacobian_stale(&mut self) {
        self.workspace.mark_jacobian_stale();
    }

    fn invalidate_factor(&mut self) {
        self.workspace.invalidate_factor();
    }
}

fn run_adaptive_dense<R>(
    config: &RadauConfig,
    y0: &[f64],
    runner: &mut R,
) -> Result<RadauSolveResult, RadauError>
where
    R: AdaptiveStepRunner,
{
    run_adaptive_dense_with_output(config, y0, runner, None)
}

fn run_adaptive_dense_with_output<R>(
    config: &RadauConfig,
    y0: &[f64],
    runner: &mut R,
    mut output: Option<&mut RadauOutputCollector>,
) -> Result<RadauSolveResult, RadauError>
where
    R: AdaptiveStepRunner,
{
    let mut state = RadauSolverState::new(config, y0)?;
    if config.first_step.is_none() {
        state.h_abs = runner.select_initial_step(config, state.t, state.direction, &state.y)?;
    }
    runner.record_initial_step(state.h_abs);
    // `trial` is deliberately allocated once.  Rejected steps overwrite it
    // and never mutate the committed state until `commit` accepts the result.
    let mut trial = vec![0.0; y0.len()];
    let mut retries = 0usize;
    let mut rejected = false;

    while !is_finished(state.t, config) {
        if state.attempts >= config.max_steps {
            return Err(RadauError::StepBudgetExceeded {
                steps: state.attempts,
            });
        }
        let minimum = min_step(state.t, state.direction);
        if !minimum.is_finite() || minimum <= 0.0 {
            return Err(RadauError::StepUnderflow {
                t: state.t,
                h: state.direction * minimum,
            });
        }
        let (h_candidate, h_abs_old, error_norm_old) = if state.h_abs > config.max_step {
            (config.max_step, None, None)
        } else if state.h_abs < minimum {
            (minimum, None, None)
        } else {
            (state.h_abs, state.h_abs_old, state.error_norm_old)
        };
        let h_abs = clamp_step_to_bound(state.t, h_candidate, config);
        let h = state.direction * h_abs;
        if !h_abs.is_finite() || h_abs < minimum || state.t + h == state.t {
            return Err(RadauError::StepUnderflow { t: state.t, h });
        }

        state.attempts += 1;
        let attempt = runner.try_step(state.t, h, &state.y, &mut trial, rejected);
        match attempt {
            Ok(result) if result.error_norm.is_finite() && result.error_norm <= 1.0 => {
                // A successful attempt publishes both time and state.  The
                // next step may reuse all backend factors' capacity, but it
                // must assemble/factor again if its new h changes the shift.
                let next_t = state.t + h;
                let recompute_jac =
                    result.iterations > 2 && result.rate.map(|rate| rate > 1e-3).unwrap_or(false);
                let mut factor =
                    predict_factor(h_abs, h_abs_old, result.error_norm, error_norm_old);
                factor = (safety_factor(result.iterations) * factor).clamp(0.2, 10.0);
                let factor_invalidated = recompute_jac || factor >= 1.2;
                if !factor_invalidated {
                    factor = 1.0;
                } else {
                    runner.invalidate_factor();
                }
                let next_h_abs = (h_abs * factor).min(config.max_step);
                state.commit(next_t, &trial, next_h_abs, result.error_norm);
                runner.record_step(true);
                runner.record_adaptive_step(RadauAdaptiveStepTrace {
                    h_abs,
                    error_norm: result.error_norm,
                    accepted: true,
                    was_retry: rejected,
                    next_h_abs,
                    factor_invalidated,
                    jacobian_refreshed: recompute_jac,
                });
                if let Some(collector) = output.as_mut() {
                    if let Some(segment) = runner.dense_output_segment()? {
                        collector.push(segment)?;
                    }
                }
                if recompute_jac {
                    runner.refresh_jacobian(next_t, &state.y)?;
                } else {
                    runner.mark_jacobian_stale();
                }
                retries = 0;
                rejected = false;
            }
            Ok(result) if result.error_norm.is_finite() => {
                // A finite but too-large embedded error is a normal adaptive
                // rejection, not a callback failure.  Preserve the committed
                // state and retry with a smaller step.
                state.rejected_steps += 1;
                retries += 1;
                runner.record_step(false);
                if retries > config.max_retries {
                    return Err(RadauError::StepControlFailure {
                        error_norm: result.error_norm,
                        retries,
                    });
                }
                let factor = predict_factor(h_abs, h_abs_old, result.error_norm, error_norm_old);
                let next_h_abs = h_abs * factor.min(1.0).max(0.2);
                state.h_abs = next_h_abs;
                runner.record_adaptive_step(RadauAdaptiveStepTrace {
                    h_abs,
                    error_norm: result.error_norm,
                    accepted: false,
                    was_retry: rejected,
                    next_h_abs,
                    factor_invalidated: true,
                    jacobian_refreshed: false,
                });
                rejected = true;
            }
            Ok(_) => {
                runner.record_step(false);
                return Err(RadauError::NonFiniteCallback {
                    stage: RadauStage::StepControl,
                });
            }
            Err(RadauError::NewtonFailure { .. }) => {
                // Newton failure is recoverable at the controller level: the
                // current trial is discarded and the step is retried with a
                // conservative half-size.  Other typed errors are fatal and
                // are returned unchanged below.
                state.rejected_steps += 1;
                retries += 1;
                runner.record_step(false);
                if !runner.jacobian_current() {
                    // SciPy retries the same h with a freshly evaluated J when
                    // the retained Jacobian is stale.  A current J instead
                    // leads directly to the conservative half-step branch.
                    runner.refresh_jacobian(state.t, &state.y)?;
                    runner.record_adaptive_step(RadauAdaptiveStepTrace {
                        h_abs,
                        error_norm: f64::INFINITY,
                        accepted: false,
                        was_retry: rejected,
                        next_h_abs: h_abs,
                        factor_invalidated: true,
                        jacobian_refreshed: true,
                    });
                    continue;
                }
                if retries > config.max_retries {
                    return Err(RadauError::StepControlFailure {
                        error_norm: f64::INFINITY,
                        retries,
                    });
                }
                let next_h_abs = h_abs * 0.5;
                state.h_abs = next_h_abs;
                runner.record_adaptive_step(RadauAdaptiveStepTrace {
                    h_abs,
                    error_norm: f64::INFINITY,
                    accepted: false,
                    was_retry: rejected,
                    next_h_abs,
                    factor_invalidated: true,
                    jacobian_refreshed: false,
                });
            }
            Err(error) => {
                runner.record_step(false);
                return Err(error);
            }
        }
    }

    Ok(RadauSolveResult {
        t: state.t,
        y: state.y,
        attempts: state.attempts,
        accepted_steps: state.accepted_steps,
        rejected_steps: state.rejected_steps,
    })
}

/// Convenience wrapper for callers that do not need to retain a session.
pub(crate) fn try_solve_dense_with_callbacks<R, J>(
    config: &RadauConfig,
    residual: &mut R,
    jacobian: &mut J,
    y0: &[f64],
) -> Result<RadauSolveResult, RadauError>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    config.validate()?;
    if y0.is_empty() {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: 1,
            actual: 0,
        });
    }
    let mut workspace = RadauWorkspace::default();
    workspace.telemetry.set_mode(config.telemetry);
    workspace.resize_for_layout(y0.len(), RadauMatrixLayout::Dense)?;
    let mut runner = CallbackStepRunner {
        config,
        residual,
        jacobian,
        workspace: &mut workspace,
    };
    run_adaptive_dense(config, y0, &mut runner)
}

/// Solve a direct callback problem and retain the selected output policy.
///
/// The public native facade uses this path so analytic and finite-difference
/// Jacobians share exactly the same controller, Newton, factorization, and
/// typed-error behavior as symbolic frontends.
pub(crate) fn try_solve_dense_with_callbacks_with_output<R, J>(
    config: &RadauConfig,
    residual: &mut R,
    jacobian: &mut J,
    y0: &[f64],
) -> Result<
    (
        RadauSolveResult,
        RadauOutput,
        super::telemetry::RadauTelemetry,
    ),
    RadauError,
>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    config.validate()?;
    if y0.is_empty() {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: 1,
            actual: 0,
        });
    }
    let mut workspace = RadauWorkspace::default();
    workspace.telemetry.set_mode(config.telemetry);
    workspace.resize_for_layout(y0.len(), RadauMatrixLayout::Dense)?;
    let mut runner = CallbackStepRunner {
        config,
        residual,
        jacobian,
        workspace: &mut workspace,
    };
    let mut collector = RadauOutputCollector::new(config.output.clone(), y0.len())?;
    let result = run_adaptive_dense_with_output(config, y0, &mut runner, Some(&mut collector))?;
    let telemetry = runner.workspace.telemetry.clone();
    Ok((result, collector.finish()?, telemetry))
}

struct CallbackStepRunner<'a, R, J> {
    config: &'a RadauConfig,
    residual: &'a mut R,
    jacobian: &'a mut J,
    workspace: &'a mut RadauWorkspace,
}

impl<R, J> AdaptiveStepRunner for CallbackStepRunner<'_, R, J>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    fn select_initial_step(
        &mut self,
        config: &RadauConfig,
        t: f64,
        direction: f64,
        y: &[f64],
    ) -> Result<f64, RadauError> {
        self.workspace.telemetry.count_stage(RadauStage::Residual);
        let residual = &mut *self.residual;
        self.workspace
            .telemetry
            .measure_callback(RadauCallbackStage::ResidualEvaluation, || {
                residual.eval(t, y, &mut self.workspace.base_rhs)
            })?;
        validate_callback_output(RadauStage::Residual, y.len(), &self.workspace.base_rhs)?;
        let h0 = initial_probe_step(config, y, &self.workspace.base_rhs);
        for index in 0..y.len() {
            self.workspace.rollback_state[index] =
                y[index] + direction * h0 * self.workspace.base_rhs[index];
        }
        self.workspace.telemetry.count_stage(RadauStage::Residual);
        let residual = &mut *self.residual;
        self.workspace.telemetry.measure_callback(
            RadauCallbackStage::ResidualEvaluation,
            || {
                residual.eval(
                    t + direction * h0,
                    &self.workspace.rollback_state,
                    &mut self.workspace.stage_rhs[..y.len()],
                )
            },
        )?;
        validate_callback_output(
            RadauStage::Residual,
            y.len(),
            &self.workspace.stage_rhs[..y.len()],
        )?;
        let selected = select_initial_step_from_probes(
            config,
            y,
            &self.workspace.base_rhs,
            &self.workspace.stage_rhs[..y.len()],
            h0,
        );
        self.workspace.preload_base_rhs();
        Ok(selected)
    }

    fn try_step(
        &mut self,
        t: f64,
        h: f64,
        y: &[f64],
        output: &mut [f64],
        rejected: bool,
    ) -> Result<RadauStepResult, RadauError> {
        try_radau5_step_with_rejection(
            self.config,
            self.residual,
            self.jacobian,
            t,
            h,
            y,
            output,
            self.workspace,
            rejected,
        )
    }

    fn record_step(&mut self, accepted: bool) {
        self.workspace.telemetry.count_step(accepted);
        if accepted {
            self.workspace.clear_base_rhs_preload();
            self.workspace.commit_dense_output();
        } else {
            self.workspace.discard_dense_output();
            self.workspace.invalidate_factor();
        }
    }

    fn record_initial_step(&mut self, h_abs: f64) {
        self.workspace.telemetry.record_initial_step(h_abs);
    }

    fn record_adaptive_step(&mut self, trace: RadauAdaptiveStepTrace) {
        self.workspace.telemetry.record_adaptive_step(trace);
    }

    fn dense_output_segment(
        &mut self,
    ) -> Result<Option<super::dense_output::RadauDenseOutputSegment>, RadauError> {
        self.workspace.dense_output_segment()
    }

    fn refresh_jacobian(&mut self, t: f64, y: &[f64]) -> Result<(), RadauError> {
        self.workspace.telemetry.count_stage(RadauStage::Jacobian);
        let jacobian = &mut *self.jacobian;
        self.workspace
            .telemetry
            .measure_callback(RadauCallbackStage::JacobianEvaluation, || {
                jacobian.eval_into(t, y, &mut self.workspace.jacobian)
            })?;
        super::callbacks::validate_callback_output(
            RadauStage::Jacobian,
            self.workspace.jacobian.len(),
            &self.workspace.jacobian,
        )?;
        self.workspace.mark_jacobian_evaluated();
        Ok(())
    }

    fn jacobian_current(&mut self) -> bool {
        self.workspace.jacobian_current()
    }

    fn mark_jacobian_stale(&mut self) {
        self.workspace.mark_jacobian_stale();
    }

    fn invalidate_factor(&mut self) {
        self.workspace.invalidate_factor();
    }
}
