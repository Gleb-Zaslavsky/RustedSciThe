//! Fixed-parameter continuation with transactional state/parameter recovery.
//!
//! Prepared callbacks, symbolic/AOT plans, and linear workspace allocations are
//! retained. Each `solve` refreshes its numeric Jacobian before using a factor.

use super::{CanteraNewtonSolver, CanteraSolveReport, CanteraSolverError, CanteraTelemetryMode};
use nalgebra::{DMatrix, DVector};

/// Native result of one ordered parameter-loop attempt.
#[derive(Debug)]
pub enum CanteraParameterLoopOutcome {
    /// Endpoint-complete profile, in mesh-by-state orientation.
    Converged {
        mesh: Vec<f64>,
        values: DMatrix<f64>,
        report: CanteraSolveReport,
    },
    /// Typed numerical failure, including native partial telemetry via
    /// [`CanteraSolverError::partial_report`]. Never a successful profile.
    Failed { error: CanteraSolverError },
}

/// One attempted vector in the order established by `bind_parameters`.
#[derive(Debug)]
pub struct CanteraParameterLoopPoint {
    pub index: usize,
    pub parameters: Vec<f64>,
    /// Previous successful point supplying the guess, or `None` for the
    /// configured initial guess (including after failures before first success).
    pub guess_source_index: Option<usize>,
    pub outcome: CanteraParameterLoopOutcome,
}

/// Ordered native results; malformed input is instead a top-level error.
#[derive(Debug, Default)]
pub struct CanteraParameterLoopReport {
    pub points: Vec<CanteraParameterLoopPoint>,
    /// Setup/output-contract failure stopping the loop; completed points remain.
    pub stop_error: Option<CanteraSolverError>,
    /// Off snapshots contain zeros, not measured work counts or durations.
    pub telemetry_mode: CanteraTelemetryMode,
}

#[derive(Clone)]
pub(super) struct ContinuationCheckpoint {
    state: DVector<f64>,
    /// Stored in canonical callback order, never caller binding order.
    parameters: Vec<f64>,
    parameters_bound: bool,
}

impl CanteraNewtonSolver {
    /// Solve new fixed inputs from the last successful solve's state.
    ///
    /// Values follow the most recent `bind_parameters` ordering. Before any
    /// success the configured initial state is used. Invalid inputs do not
    /// mutate anything. A failed attempt restores the starting checkpoint's
    /// state and canonical parameters while its error retains the attempted
    /// state's partial report. Raw `solve` still retains intermediate states
    /// on failure; this shortcut deliberately does not.
    pub fn continue_with_parameter_values(
        &mut self,
        values: Vec<f64>,
    ) -> Result<CanteraSolveReport, CanteraSolverError> {
        self.validate_parameter_values(&values)?;
        let checkpoint = self.last_successful_checkpoint.clone().unwrap_or_else(|| {
            let mut checkpoint = self.capture_checkpoint();
            checkpoint.state.clone_from(&self.initial_state);
            checkpoint
        });
        self.restore_checkpoint(&checkpoint);
        self.set_parameter_values(values)?;
        match self.solve() {
            Ok(report) => Ok(report),
            Err(error) => {
                self.restore_checkpoint(&checkpoint);
                Err(error)
            }
        }
    }

    /// Solve an ordered sequence of fixed model-parameter vectors.
    ///
    /// All rows must be finite and match a nonempty, already-bound schema;
    /// validation finishes before any solver mutation or callback evaluation.
    /// Only symbolic BVP solvers carry the real mesh required by this API.
    /// As in the Damped loop, the first point always uses the configured guess,
    /// even after a previous solve. Later points use the latest successful loop
    /// point, never a failed intermediate iterate. On return, state and inputs
    /// are restored to the latest success, or the exact entry checkpoint when
    /// no point succeeded. No symbolic preparation or AOT build is performed.
    pub fn try_solve_parameter_loop(
        &mut self,
        parameter_vectors: &[Vec<f64>],
    ) -> Result<CanteraParameterLoopReport, CanteraSolverError> {
        if parameter_vectors.is_empty() || self.external_parameter_names.is_empty() {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: "parameter loop requires a nonempty sequence and fixed parameter schema"
                    .into(),
            });
        }
        for parameters in parameter_vectors {
            self.validate_parameter_values(parameters)?;
        }
        let mesh =
            self.symbolic_mesh
                .clone()
                .ok_or_else(|| CanteraSolverError::InvalidConfiguration {
                    message: "parameter loop requires a symbolic BVP with a retained physical mesh"
                        .into(),
                })?;
        let original = self.capture_checkpoint();
        let original_success = self.last_successful_checkpoint.clone();
        let mut checkpoint = original.clone();
        checkpoint.state.clone_from(&self.initial_state);
        let mut guess_source_index = None;
        let mut result = CanteraParameterLoopReport {
            points: Vec::with_capacity(parameter_vectors.len()),
            stop_error: None,
            telemetry_mode: self.config.telemetry_mode,
        };

        for (index, parameters) in parameter_vectors.iter().enumerate() {
            self.restore_checkpoint(&checkpoint);
            // Preflight validated all rows and binding metadata is unchanged.
            if let Err(error) = self.set_parameter_values(parameters.clone()) {
                result.stop_error = Some(error);
                break;
            }
            let source = guess_source_index;
            let outcome = match self.solve() {
                Ok(report) => {
                    let values = match self.full_grid_values(&report.state) {
                        Ok(values) if values.nrows() == mesh.len() => values,
                        Ok(_) => {
                            result.stop_error = Some(CanteraSolverError::FrontendContract {
                                message: "parameter loop mesh/profile row mismatch".into(),
                            });
                            break;
                        }
                        Err(error) => {
                            result.stop_error = Some(error);
                            break;
                        }
                    };
                    checkpoint = self.capture_checkpoint();
                    guess_source_index = Some(index);
                    CanteraParameterLoopOutcome::Converged {
                        mesh: mesh.clone(),
                        values,
                        report,
                    }
                }
                Err(error) => {
                    self.restore_checkpoint(&checkpoint);
                    if error.partial_report().is_none() {
                        result.stop_error = Some(error);
                        break;
                    }
                    CanteraParameterLoopOutcome::Failed { error }
                }
            };
            result.points.push(CanteraParameterLoopPoint {
                index,
                parameters: parameters.clone(),
                guess_source_index: source,
                outcome,
            });
        }

        if guess_source_index.is_some() {
            self.restore_checkpoint(&checkpoint);
            self.last_successful_checkpoint = Some(checkpoint);
        } else {
            self.restore_checkpoint(&original);
            self.last_successful_checkpoint = original_success;
        }
        Ok(result)
    }

    /// Alias of [`Self::try_solve_parameter_loop`], with identical typed errors.
    pub fn solve_parameter_loop(
        &mut self,
        parameter_vectors: &[Vec<f64>],
    ) -> Result<CanteraParameterLoopReport, CanteraSolverError> {
        self.try_solve_parameter_loop(parameter_vectors)
    }

    pub(super) fn capture_checkpoint(&self) -> ContinuationCheckpoint {
        ContinuationCheckpoint {
            state: self.state.clone(),
            parameters: self.external_parameter_values.clone(),
            parameters_bound: self.external_parameters_bound,
        }
    }

    fn restore_checkpoint(&mut self, checkpoint: &ContinuationCheckpoint) {
        self.state.copy_from(&checkpoint.state);
        self.external_parameter_values
            .clone_from(&checkpoint.parameters);
        self.external_parameters_bound = checkpoint.parameters_bound;
        // Binding order belongs to the caller's current schema, not the old
        // solution. Numeric factors are refreshed unconditionally by solve().
    }
}
