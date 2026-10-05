//! One dense Radau IIA order-5 step.
//!
//! The Newton system follows SciPy's eigentransformed formulation: one real
//! and one complex shifted n-by-n system replace a dense 3n-by-3n system.
//! Stage values remain in caller-owned workspace and no symbolic operation is
//! performed inside the step.

use super::{
    callbacks::{
        DenseJacobianCallback, RadauStepCallbacks, ResidualCallback, validate_callback_output,
    },
    coefficients::RadauIia5,
    config::RadauConfig,
    error::{RadauConfigError, RadauError, RadauStage},
    linear::{JacobianValues, PreparedLinearBackend, RadauLinearWorkspace},
    telemetry::{RadauCallbackStage, RadauTelemetry},
    workspace::RadauWorkspace,
};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Result of one Radau IIA order-5 step attempt.
pub(crate) struct RadauStepResult {
    /// Number of Newton iterations used by the step attempt.
    pub(crate) iterations: usize,
    /// Normalized embedded error estimate.
    pub(crate) error_norm: f64,
    /// Newton convergence rate used by the Jacobian refresh policy.
    pub(crate) rate: Option<f64>,
}

/// Execute one generic-callback Radau5 step using the selected backend.
pub(crate) fn try_radau5_step<R, J>(
    config: &RadauConfig,
    residual: &mut R,
    jacobian: &mut J,
    t: f64,
    h: f64,
    y: &[f64],
    output: &mut [f64],
    workspace: &mut RadauWorkspace,
) -> Result<RadauStepResult, RadauError>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    try_radau5_step_with_rejection(
        config, residual, jacobian, t, h, y, output, workspace, false,
    )
}

/// Execute one generic-callback step while informing the error estimator that
/// the current attempt follows an adaptive rejection.
pub(crate) fn try_radau5_step_with_rejection<R, J>(
    config: &RadauConfig,
    residual: &mut R,
    jacobian: &mut J,
    t: f64,
    h: f64,
    y: &[f64],
    output: &mut [f64],
    workspace: &mut RadauWorkspace,
    rejected: bool,
) -> Result<RadauStepResult, RadauError>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    let backend = PreparedLinearBackend::from_layout(config.matrix_layout);
    try_radau5_step_with_backend(
        config, &backend, residual, jacobian, t, h, y, output, workspace, rejected,
    )
}

/// Execute one symbolic callback step through the common callback contract.
pub(crate) fn try_radau5_symbolic_step(
    config: &RadauConfig,
    callbacks: &mut super::callbacks::SymbolicCallbackSession<'_>,
    t: f64,
    h: f64,
    y: &[f64],
    output: &mut [f64],
    workspace: &mut RadauWorkspace,
) -> Result<RadauStepResult, RadauError> {
    let backend = PreparedLinearBackend::from_layout_and_pattern(
        config.matrix_layout,
        callbacks.jacobian_pattern(),
    );
    try_radau5_symbolic_step_with_backend(
        config, &backend, callbacks, t, h, y, output, workspace, false,
    )
}

/// Execute one symbolic step against a backend prepared for the whole solve.
///
/// The adaptive loop calls this function repeatedly. Keeping the backend and
/// its structural pattern outside that loop prevents Sparse/Banded setup work
/// from being repeated for every accepted or rejected attempt.
pub(crate) fn try_radau5_symbolic_step_with_backend(
    config: &RadauConfig,
    backend: &PreparedLinearBackend,
    callbacks: &mut super::callbacks::SymbolicCallbackSession<'_>,
    t: f64,
    h: f64,
    y: &[f64],
    output: &mut [f64],
    workspace: &mut RadauWorkspace,
    rejected: bool,
) -> Result<RadauStepResult, RadauError> {
    workspace.sync_callback_generation(callbacks.generation());
    try_radau5_step_with_callbacks_and_backend(
        config, backend, callbacks, t, h, y, output, workspace, rejected,
    )
}

/// Execute one Radau5 step against a backend selected during preparation.
pub(crate) fn try_radau5_step_with_backend<R, J>(
    config: &RadauConfig,
    backend: &PreparedLinearBackend,
    residual: &mut R,
    jacobian: &mut J,
    t: f64,
    h: f64,
    y: &[f64],
    output: &mut [f64],
    workspace: &mut RadauWorkspace,
    rejected: bool,
) -> Result<RadauStepResult, RadauError>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    // The callback snapshot must use the requested mode before the lower
    // numerical step starts; the workspace is initially unconfigured for a
    // fresh direct-callback solve.
    workspace.telemetry.set_mode(config.telemetry);
    let mut callback_telemetry = RadauTelemetry::new(workspace.telemetry.mode());
    let result = {
        let mut callbacks = SeparateStepCallbacks {
            residual,
            jacobian,
            telemetry: &mut callback_telemetry,
        };
        try_radau5_step_with_callbacks_and_backend(
            config,
            backend,
            &mut callbacks,
            t,
            h,
            y,
            output,
            workspace,
            rejected,
        )
    };
    workspace
        .telemetry
        .absorb_callback_runtime(&callback_telemetry);
    result
}

fn try_radau5_step_with_callbacks_and_backend<C>(
    config: &RadauConfig,
    backend: &PreparedLinearBackend,
    callbacks: &mut C,
    t: f64,
    h: f64,
    y: &[f64],
    output: &mut [f64],
    workspace: &mut RadauWorkspace,
    rejected: bool,
) -> Result<RadauStepResult, RadauError>
where
    C: RadauStepCallbacks,
{
    // This function is the numerical hot path.  Preparation has already
    // selected the backend and, for symbolic routes, cached the structural
    // pattern.  The only permitted reconfiguration here is an explicit
    // dimension/layout change from a caller using the low-level step API.
    config.validate()?;
    workspace.telemetry.set_mode(config.telemetry);
    if y.is_empty() || output.len() != y.len() {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: y.len(),
            actual: output.len(),
        });
    }
    if !t.is_finite() || !h.is_finite() || h == 0.0 {
        return Err(RadauConfigError::InvalidStep.into());
    }

    let dimension = y.len();
    if config.matrix_layout != backend.layout() {
        return Err(RadauConfigError::LinearWorkspaceMismatch.into());
    }
    if workspace.dimension() != dimension || workspace.linear_layout() != backend.layout() {
        // A prepared adaptive solve reaches this branch only on its first
        // step.  Keeping the check here makes the primitive safe to call on
        // its own without putting resize logic into every Newton iteration.
        let sparse_pattern = match backend {
            PreparedLinearBackend::Sparse(backend) => backend.pattern.as_slice(),
            _ => &[],
        };
        workspace.resize_for_layout_with_pattern(dimension, backend.layout(), sparse_pattern)?;
    }
    let coefficients = RadauIia5::new();

    if !workspace.base_rhs_preloaded() {
        callbacks.eval_residual(t, y, &mut workspace.base_rhs)?;
        validate_callback_output(RadauStage::Residual, dimension, &workspace.base_rhs)?;
    }
    for index in 0..dimension {
        workspace.scale[index] = config.atol + config.rtol * y[index].abs();
    }
    // SciPy starts with zero stage increments and then predicts from the last
    // accepted dense output.  This avoids the old Euler-only guess and keeps
    // rejected attempts anchored to the previous accepted trajectory.
    workspace.predict_stage_states(&coefficients, t, h, y);
    transform_stage_increments(
        &coefficients,
        y,
        &workspace.stage_states,
        &mut workspace.transformed_state,
    );

    if !workspace.jacobian_ready() {
        // J is evaluated once initially and then retained until the
        // controller explicitly requests a refresh after slow Newton
        // convergence or parameter/restart invalidation.
        callbacks.eval_jacobian_layout(t, y, backend.layout(), &mut workspace.jacobian)?;
        validate_callback_output(
            RadauStage::Jacobian,
            workspace.jacobian.len(),
            &workspace.jacobian,
        )?;
        workspace.mark_jacobian_evaluated();
    }
    // The callback writes directly into Dense, compact Banded, or fixed CSC
    // value storage.  No dense Jacobian is synthesized for a structured
    // backend, and a callback that cannot honor the requested layout fails
    // with a typed capability error.
    if !workspace.factor_valid_for(h) {
        let jacobian = &workspace.jacobian;
        let linear_storage = &mut workspace.linear;
        let values = jacobian_values_for_backend(backend, jacobian)?;
        // Assembly and factorization happen once per step attempt.  Newton
        // iterations below reuse both factors; only a rejected adaptive step
        // causes the next attempt to assemble/factor again.
        backend.assemble_shifted_into(
            values,
            h,
            &coefficients,
            linear_storage,
            &mut workspace.telemetry,
        )?;
        backend.factor_into(linear_storage, &mut workspace.telemetry)?;
        workspace.mark_factorized(h);
    }

    let newton_tolerance = newton_tolerance(config);
    let mut previous_norm: Option<f64> = None;
    let mut convergence_rate: Option<f64> = None;
    let mut iterations = 0;
    let newton_started = (workspace.telemetry.mode()
        == super::telemetry::RadauTelemetryMode::Timings)
        .then(std::time::Instant::now);
    // The eigentransformed Radau system has one real and one conjugate-pair
    // solve.  The factorization is intentionally outside this loop, matching
    // modified-Newton semantics rather than performing a hidden refactor.
    let converged = loop {
        iterations += 1;
        workspace.telemetry.count_newton_iteration();
        evaluate_stage_rhs(
            &coefficients,
            callbacks,
            t,
            h,
            &workspace.stage_states,
            &mut workspace.stage_rhs,
            dimension,
        )?;
        let current_norm = {
            let stage_rhs = &workspace.stage_rhs;
            let transformed_state = &workspace.transformed_state;
            let scale = &workspace.scale;
            let transformed_correction = &mut workspace.transformed_correction;
            solve_transformed_newton_system(
                &coefficients,
                h,
                stage_rhs,
                transformed_state,
                scale,
                backend,
                &mut workspace.linear,
                &mut workspace.real_rhs,
                &mut workspace.complex_rhs_real,
                &mut workspace.complex_rhs_imag,
                transformed_correction,
                dimension,
                &mut workspace.telemetry,
            )?
        };
        let rate = previous_norm.map(|previous| current_norm / previous);
        convergence_rate = rate;
        if let Some(rate) = rate {
            if rate >= 1.0
                || rate.powi(
                    (config.max_newton_iterations - iterations.min(config.max_newton_iterations))
                        as i32,
                ) / (1.0 - rate)
                    * current_norm
                    > newton_tolerance
            {
                break false;
            }
        }
        update_transformed_stage_state(
            &coefficients,
            &mut workspace.transformed_state,
            &workspace.transformed_correction,
            &mut workspace.stage_states,
            y,
            dimension,
        );
        if current_norm == 0.0
            || rate
                .map(|value| value / (1.0 - value) * current_norm < newton_tolerance)
                .unwrap_or(false)
        {
            break true;
        }
        if iterations >= config.max_newton_iterations {
            break false;
        }
        previous_norm = Some(current_norm);
    };
    if let Some(started) = newton_started {
        workspace
            .telemetry
            .add_newton_timing_ms(started.elapsed().as_secs_f64() * 1_000.0);
    }

    if !converged {
        return Err(RadauError::NewtonFailure { iterations });
    }
    let output_started = (workspace.telemetry.mode()
        == super::telemetry::RadauTelemetryMode::Timings)
        .then(std::time::Instant::now);
    // Only a converged attempt may publish its candidate state.  The adaptive
    // controller owns the committed state and decides later whether this
    // caller-owned trial output is accepted or retried.
    for index in 0..dimension {
        let mut value = y[index];
        for stage in 0..3 {
            value += h * coefficients.b[stage] * workspace.stage_rhs[stage * dimension + index];
        }
        output[index] = value;
    }
    let output_result = validate_callback_output(RadauStage::Output, dimension, output);
    if let Some(started) = output_started {
        workspace
            .telemetry
            .add_output_timing_ms(started.elapsed().as_secs_f64() * 1_000.0);
    }
    output_result?;
    workspace.telemetry.count_output_writes(output.len());
    workspace.telemetry.count_error_estimate();
    let step_control_started = (workspace.telemetry.mode()
        == super::telemetry::RadauTelemetryMode::Timings)
        .then(std::time::Instant::now);
    workspace.prepare_dense_output(&coefficients, t, h, y);
    let error_norm_result = {
        let stage_states = &workspace.stage_states;
        let base_rhs = &workspace.base_rhs;
        let scale = &mut workspace.scale;
        estimate_embedded_error(
            &coefficients,
            h,
            y,
            output,
            stage_states,
            base_rhs,
            backend,
            &mut workspace.linear,
            &mut workspace.real_rhs,
            scale,
            config.atol,
            config.rtol,
            dimension,
            &mut workspace.telemetry,
        )
    };
    if let Some(started) = step_control_started {
        workspace
            .telemetry
            .add_step_control_timing_ms(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let mut error_norm = error_norm_result?;
    if rejected && error_norm > 1.0 {
        // SciPy performs one extra corrected error evaluation after a rejected
        // step.  Reuse the existing state/RHS buffers; this must not allocate
        // a second candidate vector in the adaptive hot path.
        for index in 0..dimension {
            workspace.rollback_state[index] = y[index] + workspace.real_rhs[index];
        }
        callbacks.eval_residual(t, &workspace.rollback_state, &mut workspace.base_rhs)?;
        for index in 0..dimension {
            let embedded_stage_sum = (0..3)
                .map(|stage| {
                    coefficients.e[stage]
                        * (workspace.stage_states[stage * dimension + index] - y[index])
                })
                .sum::<f64>();
            workspace.real_rhs[index] = workspace.base_rhs[index] + embedded_stage_sum / h;
            workspace.scale[index] =
                config.atol + config.rtol * y[index].abs().max(output[index].abs());
        }
        backend.solve_real_into(
            &mut workspace.linear,
            &mut workspace.real_rhs,
            &mut workspace.telemetry,
        )?;
        error_norm = rms_scaled_norm(&workspace.real_rhs, &workspace.scale, dimension);
    }
    Ok(RadauStepResult {
        iterations,
        error_norm,
        rate: convergence_rate,
    })
}

fn initialize_stage_states(
    coefficients: &RadauIia5,
    h: f64,
    y: &[f64],
    base_rhs: &[f64],
    stage_states: &mut [f64],
) {
    let dimension = y.len();
    for stage in 0..3 {
        let offset = stage * dimension;
        for index in 0..dimension {
            stage_states[offset + index] = y[index] + coefficients.c[stage] * h * base_rhs[index];
        }
    }
}

fn transform_stage_increments(
    coefficients: &RadauIia5,
    y: &[f64],
    stage_states: &[f64],
    transformed: &mut [f64],
) {
    let dimension = y.len();
    for index in 0..dimension {
        let z = [
            stage_states[index] - y[index],
            stage_states[dimension + index] - y[index],
            stage_states[2 * dimension + index] - y[index],
        ];
        for component in 0..3 {
            transformed[component * dimension + index] = coefficients.ti[component]
                .iter()
                .zip(z)
                .map(|(coefficient, value)| coefficient * value)
                .sum();
        }
    }
}

fn evaluate_stage_rhs<C: RadauStepCallbacks>(
    coefficients: &RadauIia5,
    callbacks: &mut C,
    t: f64,
    h: f64,
    stage_states: &[f64],
    stage_rhs: &mut [f64],
    dimension: usize,
) -> Result<(), RadauError> {
    for stage in 0..3 {
        let offset = stage * dimension;
        callbacks.eval_residual(
            t + coefficients.c[stage] * h,
            &stage_states[offset..offset + dimension],
            &mut stage_rhs[offset..offset + dimension],
        )?;
        validate_callback_output(
            RadauStage::Residual,
            dimension,
            &stage_rhs[offset..offset + dimension],
        )?;
    }
    Ok(())
}

struct SeparateStepCallbacks<'a, R, J> {
    residual: &'a mut R,
    jacobian: &'a mut J,
    telemetry: &'a mut RadauTelemetry,
}

impl<R, J> RadauStepCallbacks for SeparateStepCallbacks<'_, R, J>
where
    R: ResidualCallback,
    J: DenseJacobianCallback,
{
    fn eval_residual(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        self.telemetry.count_stage(RadauStage::Residual);
        let residual = &mut *self.residual;
        self.telemetry
            .measure_callback(RadauCallbackStage::ResidualEvaluation, || {
                residual.eval(t, y, out)
            })
    }

    fn eval_jacobian(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        self.telemetry.count_stage(RadauStage::Jacobian);
        let jacobian = &mut *self.jacobian;
        self.telemetry
            .measure_callback(RadauCallbackStage::JacobianEvaluation, || {
                jacobian.eval_into(t, y, out)
            })
    }
}

fn solve_transformed_newton_system(
    coefficients: &RadauIia5,
    h: f64,
    stage_rhs: &[f64],
    transformed_state: &[f64],
    scale: &[f64],
    backend: &PreparedLinearBackend,
    linear: &mut RadauLinearWorkspace,
    real_rhs: &mut [f64],
    complex_rhs_real: &mut [f64],
    complex_rhs_imag: &mut [f64],
    transformed_correction: &mut [f64],
    dimension: usize,
    telemetry: &mut RadauTelemetry,
) -> Result<f64, RadauError> {
    // `stage_rhs` is transformed with SciPy's real matrix T into the Radau
    // eigenbasis.  The first block uses the real eigenvalue; the remaining
    // two blocks are the real and imaginary parts of one complex eigenvalue.
    // Keeping these RHS vectors separate avoids allocating a complex matrix
    // or complex heap object in the Newton loop.
    for index in 0..dimension {
        let f_real = coefficients.ti[0]
            .iter()
            .enumerate()
            .map(|(stage, coefficient)| coefficient * stage_rhs[stage * dimension + index])
            .sum::<f64>();
        real_rhs[index] = f_real - coefficients.mu_real / h * transformed_state[index];
        let f_complex_real = coefficients.ti[1][0] * stage_rhs[index]
            + coefficients.ti[1][1] * stage_rhs[dimension + index]
            + coefficients.ti[1][2] * stage_rhs[2 * dimension + index];
        let f_complex_imag = coefficients.ti[2][0] * stage_rhs[index]
            + coefficients.ti[2][1] * stage_rhs[dimension + index]
            + coefficients.ti[2][2] * stage_rhs[2 * dimension + index];
        let complex_state_real = transformed_state[dimension + index];
        let complex_state_imag = transformed_state[2 * dimension + index];
        complex_rhs_real[index] = f_complex_real
            - (coefficients.mu_complex.0 * complex_state_real
                - coefficients.mu_complex.1 * complex_state_imag)
                / h;
        complex_rhs_imag[index] = f_complex_imag
            - (coefficients.mu_complex.0 * complex_state_imag
                + coefficients.mu_complex.1 * complex_state_real)
                / h;
    }
    backend.solve_real_into(linear, real_rhs, telemetry)?;
    backend.solve_complex_into(linear, complex_rhs_real, complex_rhs_imag, telemetry)?;
    for index in 0..dimension {
        transformed_correction[index] = real_rhs[index];
        transformed_correction[dimension + index] = complex_rhs_real[index];
        transformed_correction[2 * dimension + index] = complex_rhs_imag[index];
    }
    Ok(rms_scaled_norm(transformed_correction, scale, dimension))
}

fn update_transformed_stage_state(
    coefficients: &RadauIia5,
    transformed_state: &mut [f64],
    correction: &[f64],
    stage_states: &mut [f64],
    y: &[f64],
    dimension: usize,
) {
    // The transformed correction is accumulated in place, then mapped back to
    // the three physical collocation stages.  `stage_states` is the only
    // representation passed to residual callbacks, so this is the explicit
    // boundary between linear algebra and user code.
    for index in 0..3 * dimension {
        transformed_state[index] += correction[index];
    }
    for index in 0..dimension {
        for stage in 0..3 {
            stage_states[stage * dimension + index] = y[index]
                + (0..3)
                    .map(|component| {
                        coefficients.t[stage][component]
                            * transformed_state[component * dimension + index]
                    })
                    .sum::<f64>();
        }
    }
}

fn rms_scaled_norm(values: &[f64], scale: &[f64], dimension: usize) -> f64 {
    let sum = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let normalized = value / scale[index % dimension];
            normalized * normalized
        })
        .sum::<f64>();
    (sum / values.len() as f64).sqrt()
}

fn newton_tolerance(config: &RadauConfig) -> f64 {
    (10.0 * f64::EPSILON / config.rtol).max(config.rtol.sqrt().min(0.03))
}

fn estimate_embedded_error(
    coefficients: &RadauIia5,
    h: f64,
    y: &[f64],
    output: &[f64],
    stage_states: &[f64],
    base_rhs: &[f64],
    backend: &PreparedLinearBackend,
    linear: &mut RadauLinearWorkspace,
    error_rhs: &mut [f64],
    scale: &mut [f64],
    atol: f64,
    rtol: f64,
    dimension: usize,
    telemetry: &mut RadauTelemetry,
) -> Result<f64, RadauError> {
    // The embedded estimator reuses the already factored real system.  It
    // changes only the RHS, so no second Jacobian assembly or factorization is
    // allowed here.
    for index in 0..dimension {
        let embedded_stage_sum = (0..3)
            .map(|stage| {
                coefficients.e[stage] * (stage_states[stage * dimension + index] - y[index])
            })
            .sum::<f64>();
        error_rhs[index] = base_rhs[index] + embedded_stage_sum / h;
        scale[index] = atol + rtol * y[index].abs().max(output[index].abs());
    }
    backend.solve_real_into(linear, error_rhs, telemetry)?;
    for index in 0..dimension {
        if !error_rhs[index].is_finite() {
            return Err(RadauError::NonFiniteCallback {
                stage: RadauStage::StepControl,
            });
        }
    }
    Ok(rms_scaled_norm(error_rhs, scale, dimension))
}

fn jacobian_values_for_backend<'a>(
    backend: &'a PreparedLinearBackend,
    values: &'a [f64],
) -> Result<JacobianValues<'a>, RadauError> {
    Ok(match backend {
        PreparedLinearBackend::Dense(_) => JacobianValues::Dense { values },
        PreparedLinearBackend::Banded(backend) => JacobianValues::Banded {
            values,
            lower: backend.lower,
            upper: backend.upper,
        },
        PreparedLinearBackend::Sparse(backend) => JacobianValues::Sparse {
            values,
            entries: &backend.pattern,
        },
    })
}
