//! Reusable collocation and backend workspaces.
//!
//! State is node-major: `y[node * n + component]`.  This is the same block
//! order used by the global collocation Jacobian and avoids a transpose while
//! entering a linear backend.

use super::{error::BvpSciNewError, telemetry::BvpSciTelemetry};

/// Scratch owned by one prepared solve. Resizing is explicit and observable;
/// ordinary Newton iterations only overwrite these buffers. The workspace is
/// intentionally solver-owned: sharing it between concurrent solves would
/// corrupt rollback state and make telemetry attribution ambiguous.
#[derive(Clone, Debug)]
pub struct BvpSciCollocationWorkspace {
    /// State dimension, mesh node count and parameter dimension.
    pub n: usize,
    pub m: usize,
    pub k: usize,
    /// Geometry and reusable RHS/interpolation buffers. `f_nodes` and
    /// `f_middle` are also the accepted-state values used by Hermite defect
    /// probes and mesh interpolation.
    pub h: Vec<f64>,
    pub x_mid: Vec<f64>,
    pub f_nodes: Vec<f64>,
    pub f_middle: Vec<f64>,
    pub y_middle: Vec<f64>,
    pub collocation_residual: Vec<f64>,
    /// Boundary and combined residual buffers consumed by Newton.
    pub boundary_residual: Vec<f64>,
    pub residual: Vec<f64>,
    pub step: Vec<f64>,
    /// Reusable RHS for the affine-invariant SciPy line-search criterion.
    /// It holds `J^-1 r_trial` while `step` remains the current Newton step.
    pub trial_step: Vec<f64>,
    pub trial_parameters: Vec<f64>,
    pub rollback_state: Vec<f64>,
    pub rollback_parameters: Vec<f64>,
    /// Pointwise state and parameter derivative blocks. Their layout is
    /// node-major, then row/column, matching the callback contract and the
    /// global Jacobian assembly helpers.
    pub jacobian_nodes: Vec<f64>,
    pub jacobian_middle: Vec<f64>,
    pub parameter_nodes: Vec<f64>,
    pub parameter_middle: Vec<f64>,
    pub boundary_y_a: Vec<f64>,
    pub boundary_y_b: Vec<f64>,
    pub boundary_parameters: Vec<f64>,
    pub boundary_ya_jacobian: Vec<f64>,
    pub boundary_yb_jacobian: Vec<f64>,
    pub boundary_parameter_jacobian: Vec<f64>,
    pub boundary_trial_output: Vec<f64>,
    /// Caller-owned callback argument/output storage.  These vectors avoid
    /// per-RHS allocation in both collocation and Jacobian assembly.
    pub callback_arguments: Vec<f64>,
    pub callback_output: Vec<f64>,
    /// RHS scratch used when a singular term transforms a callback output.
    pub callback_rhs: Vec<f64>,
    /// Scratch Jacobian block used by singular-term correction without a
    /// temporary allocation in the Newton loop.
    pub callback_jacobian: Vec<f64>,
    /// Required callback scratch length. Structured AOT layouts can expose
    /// compact slot buffers larger than the pointwise dense `n*n` block.
    callback_jacobian_required_len: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl BvpSciCollocationWorkspace {
    /// Allocate a workspace and size every reusable buffer for one mesh.
    ///
    /// The checked arithmetic is part of the public error contract: a very
    /// large requested mesh must return a typed configuration error instead of
    /// overflowing an index or panicking during a later Newton step.
    pub fn new(
        n: usize,
        m: usize,
        k: usize,
        telemetry: &BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        let callback_jacobian_len = n.checked_mul(n).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("callback Jacobian size overflow".into())
        })?;
        Self::new_with_callback_jacobian_len(n, m, k, callback_jacobian_len, telemetry)
    }

    /// Allocate a workspace with the exact callback scratch contract of the
    /// prepared frontend. Dense/Lambdify routes use `n*n`; compact AOT
    /// Banded routes may require additional structural boundary slots.
    pub fn new_with_callback_jacobian_len(
        n: usize,
        m: usize,
        k: usize,
        callback_jacobian_len: usize,
        telemetry: &BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        let boundary_len = n.checked_add(k).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary workspace size overflow".into())
        })?;
        let boundary_state_len = boundary_len.checked_mul(n).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary Jacobian size overflow".into())
        })?;
        let boundary_parameter_len = boundary_len.checked_mul(k).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary parameter Jacobian size overflow".into())
        })?;
        let mut workspace = Self {
            n,
            m,
            k,
            h: Vec::new(),
            x_mid: Vec::new(),
            f_nodes: Vec::new(),
            f_middle: Vec::new(),
            y_middle: Vec::new(),
            collocation_residual: Vec::new(),
            boundary_residual: vec![0.0; boundary_len],
            residual: Vec::new(),
            step: Vec::new(),
            trial_step: Vec::new(),
            trial_parameters: vec![0.0; k],
            rollback_state: Vec::new(),
            rollback_parameters: vec![0.0; k],
            jacobian_nodes: Vec::new(),
            jacobian_middle: Vec::new(),
            parameter_nodes: Vec::new(),
            parameter_middle: Vec::new(),
            boundary_y_a: vec![0.0; n],
            boundary_y_b: vec![0.0; n],
            boundary_parameters: vec![0.0; k],
            boundary_ya_jacobian: vec![0.0; boundary_state_len],
            boundary_yb_jacobian: vec![0.0; boundary_state_len],
            boundary_parameter_jacobian: vec![0.0; boundary_parameter_len],
            boundary_trial_output: vec![0.0; boundary_len],
            callback_arguments: Vec::new(),
            callback_output: vec![0.0; n],
            callback_rhs: vec![0.0; n],
            callback_jacobian: Vec::new(),
            callback_jacobian_required_len: callback_jacobian_len,
            entries: Vec::new(),
        };
        workspace.resize(n, m, k, telemetry)?;
        Ok(workspace)
    }

    pub fn resize(
        &mut self,
        n: usize,
        m: usize,
        k: usize,
        telemetry: &BvpSciTelemetry,
    ) -> Result<(), BvpSciNewError> {
        if n == 0 || m < 2 {
            return Err(BvpSciNewError::InvalidConfiguration(
                "BVP workspace requires n > 0 and at least two mesh nodes".into(),
            ));
        }
        // Capacity is sampled only for telemetry. It is not used to decide
        // numerical behavior and does not pretend to be a process allocator
        // measurement.
        let capacity_before = self.capacity_units();
        let intervals = m - 1;
        let state_len = n.checked_mul(m).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("state workspace size overflow".into())
        })?;
        let interval_len = n.checked_mul(intervals).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("interval workspace size overflow".into())
        })?;
        let jacobian_len = state_len.checked_mul(n).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("Jacobian workspace size overflow".into())
        })?;
        let parameter_len = state_len.checked_mul(k).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("parameter workspace size overflow".into())
        })?;
        let boundary_len = n.checked_add(k).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary workspace size overflow".into())
        })?;
        let boundary_state_len = boundary_len.checked_mul(n).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary Jacobian size overflow".into())
        })?;
        let boundary_parameter_len = boundary_len.checked_mul(k).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary parameter Jacobian size overflow".into())
        })?;
        let midpoint_jacobian_len = interval_len.checked_mul(n).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("midpoint Jacobian size overflow".into())
        })?;
        let midpoint_parameter_len = interval_len.checked_mul(k).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("midpoint parameter size overflow".into())
        })?;
        let callback_argument_len = 1usize.checked_add(boundary_len).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("callback argument size overflow".into())
        })?;
        let residual_len = interval_len.checked_add(boundary_len).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("residual workspace size overflow".into())
        })?;
        let step_len = state_len.checked_add(k).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("Newton step size overflow".into())
        })?;
        self.n = n;
        self.m = m;
        self.k = k;
        self.h.resize(intervals, 0.0);
        self.x_mid.resize(intervals, 0.0);
        self.f_nodes.resize(state_len, 0.0);
        self.f_middle.resize(interval_len, 0.0);
        self.y_middle.resize(interval_len, 0.0);
        self.collocation_residual.resize(interval_len, 0.0);
        self.boundary_residual.resize(boundary_len, 0.0);
        self.residual.resize(residual_len, 0.0);
        self.step.resize(step_len, 0.0);
        self.trial_step.resize(step_len, 0.0);
        self.trial_parameters.resize(k, 0.0);
        self.rollback_state.resize(state_len, 0.0);
        self.rollback_parameters.resize(k, 0.0);
        self.jacobian_nodes.resize(jacobian_len, 0.0);
        self.jacobian_middle.resize(midpoint_jacobian_len, 0.0);
        self.parameter_nodes.resize(parameter_len, 0.0);
        self.parameter_middle.resize(midpoint_parameter_len, 0.0);
        self.boundary_y_a.resize(n, 0.0);
        self.boundary_y_b.resize(n, 0.0);
        self.boundary_parameters.resize(k, 0.0);
        self.boundary_ya_jacobian.resize(boundary_state_len, 0.0);
        self.boundary_yb_jacobian.resize(boundary_state_len, 0.0);
        self.boundary_parameter_jacobian
            .resize(boundary_parameter_len, 0.0);
        self.boundary_trial_output.resize(boundary_len, 0.0);
        self.callback_arguments.resize(callback_argument_len, 0.0);
        self.callback_output.resize(n, 0.0);
        self.callback_rhs.resize(n, 0.0);
        let dense_jacobian_len = n.checked_mul(n).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("callback Jacobian size overflow".into())
        })?;
        self.callback_jacobian.resize(
            self.callback_jacobian_required_len.max(dense_jacobian_len),
            0.0,
        );
        // Entries are cleared but capacity is retained because every Newton
        // refresh uses the same structural collocation pattern for this mesh.
        self.entries.clear();
        let entry_capacity = intervals
            .checked_mul(n)
            .and_then(|value| value.checked_mul(2usize.checked_mul(n)?.checked_add(k)?))
            .and_then(|value| {
                boundary_len
                    .checked_mul(2usize.checked_mul(n)?.checked_add(k)?)
                    .and_then(|boundary_entries| value.checked_add(boundary_entries))
            })
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration("linear entry capacity overflow".into())
            })?;
        self.entries.try_reserve(entry_capacity).map_err(|_| {
            BvpSciNewError::InvalidConfiguration("linear entry capacity cannot be allocated".into())
        })?;
        if self.capacity_units() > capacity_before {
            telemetry.record_allocation();
        }
        telemetry.record_workspace_resize();
        Ok(())
    }

    fn capacity_units(&self) -> usize {
        [
            self.h.capacity(),
            self.x_mid.capacity(),
            self.f_nodes.capacity(),
            self.f_middle.capacity(),
            self.y_middle.capacity(),
            self.collocation_residual.capacity(),
            self.boundary_residual.capacity(),
            self.residual.capacity(),
            self.step.capacity(),
            self.trial_step.capacity(),
            self.trial_parameters.capacity(),
            self.rollback_state.capacity(),
            self.rollback_parameters.capacity(),
            self.jacobian_nodes.capacity(),
            self.jacobian_middle.capacity(),
            self.parameter_nodes.capacity(),
            self.parameter_middle.capacity(),
            self.boundary_y_a.capacity(),
            self.boundary_y_b.capacity(),
            self.boundary_parameters.capacity(),
            self.boundary_ya_jacobian.capacity(),
            self.boundary_yb_jacobian.capacity(),
            self.boundary_parameter_jacobian.capacity(),
            self.boundary_trial_output.capacity(),
            self.callback_arguments.capacity(),
            self.callback_output.capacity(),
            self.callback_rhs.capacity(),
            self.callback_jacobian.capacity(),
            self.entries.capacity(),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add)
    }

    #[inline]
    pub fn state_offset(&self, node: usize, component: usize) -> usize {
        node * self.n + component
    }

    #[inline]
    pub fn jacobian_offset(&self, point: usize, row: usize, column: usize) -> usize {
        (point * self.n + row) * self.n + column
    }

    #[inline]
    pub fn parameter_offset(&self, point: usize, row: usize, parameter: usize) -> usize {
        (point * self.n + row) * self.k + parameter
    }
}
