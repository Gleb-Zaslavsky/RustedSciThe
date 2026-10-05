//! Reusable solver workspace and ownership boundaries.
//!
//! Buffers, stage vectors, Jacobian storage, and rollback snapshots are sized
//! once per dimension rather than allocated inside each Newton iteration.
//! Adaptive rejection reuses the same workspace with a smaller step, while
//! Newton iterations overwrite the stage/RHS buffers in place.  The selected
//! linear backend owns its native storage below this layer, so this type does
//! not force Sparse or Banded routes through a dense representation.

use super::config::RadauMatrixLayout;
use super::error::{RadauError, RadauUnsupportedRoute};
use super::linear::{DenseLinearWorkspace, RadauLinearWorkspace, SparseLinearWorkspace};
use super::telemetry::RadauTelemetry;

#[derive(Debug, Default)]
/// Reusable stage, callback, rollback, and linear-system storage for one session.
pub(crate) struct RadauWorkspace {
    /// Per-session diagnostic counters and timings.
    pub telemetry: RadauTelemetry,
    /// Three collocation stage states, stored as `[stage][component]`.
    pub stage_states: Vec<f64>,
    /// Three collocation stage residuals in the same layout.
    pub stage_rhs: Vec<f64>,
    /// Newton correction stages, reused across iterations and retries.
    pub correction: Vec<f64>,
    /// Per-component error scaling vector used by Newton and step control.
    pub scale: Vec<f64>,
    /// Jacobian scratch in the native storage selected by the active layout.
    /// Dense means `n*n`; Banded and Sparse have their own compact lengths.
    pub jacobian: Vec<f64>,
    /// Backend-specific shifted matrices, RHS buffers, and factor handles.
    pub linear: RadauLinearWorkspace,
    /// Rollback copy of the committed state.
    pub rollback_state: Vec<f64>,
    /// Base residual at the current step.
    pub base_rhs: Vec<f64>,
    /// Whether `base_rhs` was preloaded by the initial-step probe for the
    /// current committed state and may be reused by the first attempt.
    base_rhs_preloaded: bool,
    /// Newton RHS for the real shifted system.
    pub real_rhs: Vec<f64>,
    /// Real part of the Newton RHS for the complex shifted system.
    pub complex_rhs_real: Vec<f64>,
    /// Imaginary part of the Newton RHS for the complex shifted system.
    pub complex_rhs_imag: Vec<f64>,
    /// Radau eigentransformed state: real block, then real/imaginary pair.
    pub transformed_state: Vec<f64>,
    /// Matching transformed Newton correction, reused for every iteration.
    pub transformed_correction: Vec<f64>,
    /// Coefficients of the last accepted collocation interpolant, stored as
    /// `[component][x, x^2, x^3]`.
    pub dense_output_q: Vec<f64>,
    /// State and interval of the last accepted interpolant.
    pub dense_output_y_old: Vec<f64>,
    pub dense_output_t_old: f64,
    pub dense_output_h: f64,
    pub dense_output_valid: bool,
    /// Candidate interpolant built by the current attempt.
    pub pending_dense_output_q: Vec<f64>,
    /// Start-of-step state paired with the candidate interpolant.
    pub pending_dense_output_y_old: Vec<f64>,
    pub pending_dense_output_t_old: f64,
    pub pending_dense_output_h: f64,
    pub pending_dense_output_valid: bool,
    /// Numeric lifecycle state for SciPy-style Jacobian/LU reuse.
    jacobian_ready: bool,
    jacobian_current: bool,
    factor_valid: bool,
    factor_h: f64,
    jacobian_generation: u64,
    factor_generation: u64,
    callback_generation: Option<u64>,
}

impl RadauWorkspace {
    /// Resize the default dense workspace for a state dimension.
    pub(crate) fn resize_for_dimension(&mut self, dimension: usize) -> Result<(), RadauError> {
        self.resize_for_layout(dimension, RadauMatrixLayout::Dense)
    }

    /// Resize buffers and select the matching backend-specific linear storage.
    pub(crate) fn resize_for_layout(
        &mut self,
        dimension: usize,
        layout: RadauMatrixLayout,
    ) -> Result<(), RadauError> {
        self.resize_for_layout_with_pattern(dimension, layout, &[])
    }

    /// Resize buffers while retaining the prepared sparse structural pattern.
    pub(crate) fn resize_for_layout_with_pattern(
        &mut self,
        dimension: usize,
        layout: RadauMatrixLayout,
        sparse_pattern: &[(usize, usize)],
    ) -> Result<(), RadauError> {
        let started = (self.telemetry.mode() == super::telemetry::RadauTelemetryMode::Timings)
            .then(std::time::Instant::now);
        self.telemetry.count_workspace_resize();
        let result = self.resize_for_layout_inner(dimension, layout, sparse_pattern);
        if let Some(started) = started {
            self.telemetry
                .add_workspace_timing_ms(started.elapsed().as_secs_f64() * 1_000.0);
        }
        result
    }

    fn resize_for_layout_inner(
        &mut self,
        dimension: usize,
        layout: RadauMatrixLayout,
        sparse_pattern: &[(usize, usize)],
    ) -> Result<(), RadauError> {
        self.resize_buffers_with_pattern(dimension, layout, sparse_pattern)?;

        match layout {
            RadauMatrixLayout::Dense => {
                if !matches!(self.linear, RadauLinearWorkspace::Dense(_)) {
                    self.linear = RadauLinearWorkspace::Dense(DenseLinearWorkspace::default());
                }
                if let RadauLinearWorkspace::Dense(linear) = &mut self.linear {
                    linear.resize_for_dimension(dimension)?;
                }
            }
            RadauMatrixLayout::Banded { lower, upper } => {
                let replace = match &self.linear {
                    RadauLinearWorkspace::Banded(linear) => {
                        linear.dimension != dimension
                            || linear.lower != lower
                            || linear.upper != upper
                    }
                    _ => true,
                };
                if replace {
                    self.linear = RadauLinearWorkspace::Banded(
                        super::linear::BandedLinearWorkspace::new(dimension, lower, upper)?,
                    );
                }
            }
            RadauMatrixLayout::Sparse => {
                let expected_pattern_len =
                    SparseLinearWorkspace::from_pattern(dimension, sparse_pattern)?
                        .row_indices
                        .len();
                let replace = match &self.linear {
                    RadauLinearWorkspace::Sparse(linear) => {
                        linear.dimension != dimension
                            || linear.row_indices.len() != expected_pattern_len
                    }
                    _ => true,
                };
                if replace {
                    self.linear = RadauLinearWorkspace::Sparse(
                        SparseLinearWorkspace::from_pattern(dimension, sparse_pattern)?,
                    );
                }
            }
        }
        Ok(())
    }

    /// Resize non-linear stage and rollback buffers without selecting a backend.
    pub(crate) fn resize_buffers(
        &mut self,
        dimension: usize,
        layout: RadauMatrixLayout,
    ) -> Result<(), RadauError> {
        self.resize_buffers_with_pattern(dimension, layout, &[])
    }

    /// Resize non-linear buffers and native Jacobian output storage.
    pub(crate) fn resize_buffers_with_pattern(
        &mut self,
        dimension: usize,
        layout: RadauMatrixLayout,
        sparse_pattern: &[(usize, usize)],
    ) -> Result<(), RadauError> {
        // Resizing is a preparation/reconfiguration operation, not part of a
        // normal Newton step.  Avoid even capacity reads when telemetry is
        // Off so the default path remains genuinely low overhead.
        let capacities_before =
            (self.telemetry.mode() != super::telemetry::RadauTelemetryMode::Off).then(|| {
                [
                    self.stage_states.capacity(),
                    self.stage_rhs.capacity(),
                    self.correction.capacity(),
                    self.scale.capacity(),
                    self.jacobian.capacity(),
                    self.rollback_state.capacity(),
                    self.base_rhs.capacity(),
                    self.real_rhs.capacity(),
                    self.complex_rhs_real.capacity(),
                    self.complex_rhs_imag.capacity(),
                    self.transformed_state.capacity(),
                    self.transformed_correction.capacity(),
                ]
            });
        let stage_len = dimension
            .checked_mul(3)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        self.stage_states.resize(stage_len, 0.0);
        self.stage_rhs.resize(stage_len, 0.0);
        self.correction.resize(stage_len, 0.0);
        self.scale.resize(dimension, 0.0);
        let jacobian_len = match layout {
            RadauMatrixLayout::Dense => dimension
                .checked_mul(dimension)
                .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?,
            RadauMatrixLayout::Banded { lower, upper } => lower
                .checked_add(upper)
                .and_then(|value| value.checked_add(1))
                .and_then(|slots| slots.checked_mul(dimension))
                .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?,
            RadauMatrixLayout::Sparse => {
                SparseLinearWorkspace::from_pattern(dimension, sparse_pattern)?
                    .jacobian
                    .len()
            }
        };
        self.jacobian.resize(jacobian_len, 0.0);
        self.rollback_state.resize(dimension, 0.0);
        self.base_rhs.resize(dimension, 0.0);
        self.real_rhs.resize(dimension, 0.0);
        self.complex_rhs_real.resize(dimension, 0.0);
        self.complex_rhs_imag.resize(dimension, 0.0);
        self.transformed_state.resize(stage_len, 0.0);
        self.transformed_correction.resize(stage_len, 0.0);
        let dense_output_len = dimension
            .checked_mul(3)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        self.dense_output_q.resize(dense_output_len, 0.0);
        self.dense_output_y_old.resize(dimension, 0.0);
        self.pending_dense_output_q.resize(dense_output_len, 0.0);
        self.pending_dense_output_y_old.resize(dimension, 0.0);
        self.jacobian_ready = false;
        self.jacobian_current = false;
        self.factor_valid = false;
        self.factor_h = 0.0;
        self.jacobian_generation = 0;
        self.factor_generation = 0;
        self.callback_generation = None;
        self.base_rhs_preloaded = false;
        self.dense_output_valid = false;
        self.pending_dense_output_valid = false;
        if let Some(capacities_before) = capacities_before {
            // Vec::resize does not expose whether an allocator was called.
            // Capacity deltas provide a cheap diagnostic for unexpected
            // workspace growth without instrumenting the global allocator.
            let capacities_after = [
                self.stage_states.capacity(),
                self.stage_rhs.capacity(),
                self.correction.capacity(),
                self.scale.capacity(),
                self.jacobian.capacity(),
                self.rollback_state.capacity(),
                self.base_rhs.capacity(),
                self.real_rhs.capacity(),
                self.complex_rhs_real.capacity(),
                self.complex_rhs_imag.capacity(),
                self.transformed_state.capacity(),
                self.transformed_correction.capacity(),
            ];
            for (before, after) in capacities_before.into_iter().zip(capacities_after) {
                if after > before {
                    self.telemetry.count_allocation();
                }
            }
        }
        Ok(())
    }

    /// Return the dimension represented by the rollback buffer.
    pub(crate) fn dimension(&self) -> usize {
        self.rollback_state.len()
    }

    /// Return the layout currently owning the linear workspace.
    pub(crate) fn linear_layout(&self) -> RadauMatrixLayout {
        // Validate layout before a callback writes values.  Discovering a
        // mismatch inside factorization would leave partially assembled
        // numeric state and make the failure much harder to diagnose.
        match &self.linear {
            RadauLinearWorkspace::Dense(_) => RadauMatrixLayout::Dense,
            RadauLinearWorkspace::Banded(linear) => RadauMatrixLayout::Banded {
                lower: linear.lower,
                upper: linear.upper,
            },
            RadauLinearWorkspace::Sparse(_) => RadauMatrixLayout::Sparse,
        }
    }

    /// Return whether the current LU pair matches this step size and J.
    pub(crate) fn factor_valid_for(&self, h: f64) -> bool {
        self.factor_valid
            && self.factor_generation == self.jacobian_generation
            && self.factor_h == h
    }

    /// Mark a freshly evaluated Jacobian as usable and invalidate its factors.
    pub(crate) fn mark_jacobian_evaluated(&mut self) {
        self.jacobian_ready = true;
        self.jacobian_current = true;
        self.factor_valid = false;
        self.jacobian_generation = self.jacobian_generation.wrapping_add(1);
    }

    /// Keep J values but record that they describe an earlier accepted state.
    pub(crate) fn mark_jacobian_stale(&mut self) {
        if self.jacobian_ready {
            self.jacobian_current = false;
        }
    }

    /// Drop numeric Jacobian state after a symbolic parameter generation change.
    pub(crate) fn invalidate_jacobian(&mut self) {
        self.jacobian_ready = false;
        self.jacobian_current = false;
        self.factor_valid = false;
        self.base_rhs_preloaded = false;
    }

    /// Mark the initial RHS probe as available to the first Newton attempt.
    pub(crate) fn preload_base_rhs(&mut self) {
        self.base_rhs_preloaded = true;
    }

    /// Consume the initial RHS preload only after a step is accepted.
    pub(crate) fn clear_base_rhs_preload(&mut self) {
        self.base_rhs_preloaded = false;
    }

    pub(crate) fn base_rhs_preloaded(&self) -> bool {
        self.base_rhs_preloaded
    }

    /// Synchronize callback-owned parameter generation with numeric workspace.
    ///
    /// A value-only rebind keeps compiled closures and storage, but it must
    /// never reuse a Jacobian or LU factor produced for the previous values.
    pub(crate) fn sync_callback_generation(&mut self, generation: u64) {
        if self.callback_generation != Some(generation) {
            self.invalidate_jacobian();
            self.callback_generation = Some(generation);
        }
    }

    pub(crate) fn jacobian_ready(&self) -> bool {
        self.jacobian_ready
    }

    pub(crate) fn jacobian_current(&self) -> bool {
        self.jacobian_current
    }

    /// Record an in-place factorization for the current step size and J.
    pub(crate) fn mark_factorized(&mut self, h: f64) {
        self.factor_valid = true;
        self.factor_h = h;
        self.factor_generation = self.jacobian_generation;
    }

    /// Invalidate only LU factors while retaining J and all allocations.
    pub(crate) fn invalidate_factor(&mut self) {
        self.factor_valid = false;
    }

    /// Build the candidate continuous-output polynomial from stage increments.
    pub(crate) fn prepare_dense_output(
        &mut self,
        coefficients: &super::coefficients::RadauIia5,
        t: f64,
        h: f64,
        y: &[f64],
    ) {
        let dimension = y.len();
        self.pending_dense_output_y_old.copy_from_slice(y);
        for index in 0..dimension {
            let z = [
                self.stage_states[index] - y[index],
                self.stage_states[dimension + index] - y[index],
                self.stage_states[2 * dimension + index] - y[index],
            ];
            for power in 0..3 {
                self.pending_dense_output_q[power * dimension + index] = (0..3)
                    .map(|stage| z[stage] * coefficients.p[stage][power])
                    .sum();
            }
        }
        self.pending_dense_output_t_old = t;
        self.pending_dense_output_h = h;
        self.pending_dense_output_valid = true;
    }

    /// Publish a candidate interpolant only after adaptive acceptance.
    pub(crate) fn commit_dense_output(&mut self) {
        if !self.pending_dense_output_valid {
            return;
        }
        self.dense_output_q
            .copy_from_slice(&self.pending_dense_output_q);
        self.dense_output_y_old
            .copy_from_slice(&self.pending_dense_output_y_old);
        self.dense_output_t_old = self.pending_dense_output_t_old;
        self.dense_output_h = self.pending_dense_output_h;
        self.dense_output_valid = true;
        self.pending_dense_output_valid = false;
    }

    /// Snapshot the last accepted cubic segment for post-processing.
    pub(crate) fn dense_output_segment(
        &self,
    ) -> Result<Option<super::dense_output::RadauDenseOutputSegment>, RadauError> {
        if !self.dense_output_valid {
            return Ok(None);
        }
        Ok(Some(super::dense_output::RadauDenseOutputSegment::new(
            self.dense_output_t_old,
            self.dense_output_h,
            self.dense_output_y_old.clone(),
            self.dense_output_q.clone(),
        )?))
    }

    /// Discard a rejected candidate without touching the previous predictor.
    pub(crate) fn discard_dense_output(&mut self) {
        self.pending_dense_output_valid = false;
    }

    /// Use the previous accepted interpolant as SciPy's next-step predictor.
    pub(crate) fn predict_stage_states(
        &mut self,
        coefficients: &super::coefficients::RadauIia5,
        t: f64,
        h: f64,
        y: &[f64],
    ) {
        let dimension = y.len();
        if !self.dense_output_valid {
            for stage in 0..3 {
                self.stage_states[stage * dimension..(stage + 1) * dimension].copy_from_slice(y);
            }
            return;
        }
        for stage in 0..3 {
            let x = (t + coefficients.c[stage] * h - self.dense_output_t_old) / self.dense_output_h;
            let powers = [x, x * x, x * x * x];
            for index in 0..dimension {
                let offset = stage * dimension + index;
                self.stage_states[offset] = self.dense_output_y_old[index]
                    + (0..3)
                        .map(|power| self.dense_output_q[power * dimension + index] * powers[power])
                        .sum::<f64>();
            }
        }
    }

    /// Borrow dense linear storage or report the selected structured route.
    pub(crate) fn dense_linear(&self) -> Result<&DenseLinearWorkspace, RadauError> {
        match &self.linear {
            RadauLinearWorkspace::Dense(linear) => Ok(linear),
            _ => Err(RadauUnsupportedRoute::DenseWorkspaceRequired.into()),
        }
    }

    /// Mutably borrow dense linear storage or report the selected structured route.
    pub(crate) fn dense_linear_mut(&mut self) -> Result<&mut DenseLinearWorkspace, RadauError> {
        match &mut self.linear {
            RadauLinearWorkspace::Dense(linear) => Ok(linear),
            _ => Err(RadauUnsupportedRoute::DenseWorkspaceRequired.into()),
        }
    }
}
