//! Ordered solves over fixed external model-parameter vectors.
//!
//! A failed point is reported in place and does not abort later points. The
//! next attempt starts from the last successful solution, or from the original
//! solver state if no point has succeeded yet.

use super::{BvpSciNewError, BvpSciSolution, BvpSciSolver, BvpSciTelemetrySnapshot};

/// Outcome and cumulative telemetry snapshot after one requested input set.
#[derive(Debug)]
pub struct BvpSciParameterLoopPoint {
    pub external_parameters: Vec<f64>,
    pub result: Result<BvpSciSolution, BvpSciNewError>,
    pub telemetry: BvpSciTelemetrySnapshot,
}

/// Ordered loop results. `stop_error` is set only if the solver could not be
/// restored after a point failure; ordinary solve failures remain per-point.
#[derive(Debug, Default)]
pub struct BvpSciParameterLoopReport {
    pub points: Vec<BvpSciParameterLoopPoint>,
    pub stop_error: Option<BvpSciNewError>,
}

/// Solve each external-parameter vector in order while reusing the prepared
/// model and the most recent successful mesh/profile/free-parameter solution.
pub fn solve_ordered(
    solver: &mut BvpSciSolver,
    parameter_vectors: &[Vec<f64>],
) -> Result<BvpSciParameterLoopReport, BvpSciNewError> {
    let expected = solver.plan().external_parameter_dimension();
    if parameter_vectors.is_empty() {
        return Err(BvpSciNewError::InvalidConfiguration(
            "parameter loop requires at least one parameter vector".into(),
        ));
    }
    for values in parameter_vectors {
        if values.len() != expected {
            return Err(BvpSciNewError::ParameterBindingMismatch {
                expected,
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: super::BvpSciStage::ParameterBinding,
            });
        }
    }

    let (initial_mesh, initial_state, initial_free_parameters) = solver.state();
    let mut checkpoint = Checkpoint {
        mesh: initial_mesh.to_vec(),
        state: initial_state.to_vec(),
        free_parameters: initial_free_parameters.to_vec(),
        external_parameters: solver.external_parameter_values().to_vec(),
        has_successful_solve: false,
    };
    let mut report = BvpSciParameterLoopReport {
        points: Vec::with_capacity(parameter_vectors.len()),
        stop_error: None,
    };

    for values in parameter_vectors {
        solver.set_external_parameter_values(values.clone())?;
        let result = solver.solve();
        let telemetry = solver.plan().telemetry_snapshot();
        match &result {
            Ok(solution) => {
                checkpoint = Checkpoint {
                    mesh: solution.x.clone(),
                    state: solution.y.clone(),
                    free_parameters: solution.parameters.clone(),
                    external_parameters: solution.external_parameters.clone(),
                    has_successful_solve: true,
                };
            }
            Err(_) => {
                if let Err(error) = checkpoint.restore(solver) {
                    report.stop_error = Some(error);
                    report.points.push(BvpSciParameterLoopPoint {
                        external_parameters: values.clone(),
                        result,
                        telemetry,
                    });
                    break;
                }
            }
        }
        report.points.push(BvpSciParameterLoopPoint {
            external_parameters: values.clone(),
            result,
            telemetry,
        });
    }
    Ok(report)
}

struct Checkpoint {
    mesh: Vec<f64>,
    state: Vec<f64>,
    free_parameters: Vec<f64>,
    external_parameters: Vec<f64>,
    has_successful_solve: bool,
}

impl Checkpoint {
    fn restore(&self, solver: &mut BvpSciSolver) -> Result<(), BvpSciNewError> {
        solver.restore_parameter_loop_checkpoint(
            self.mesh.clone(),
            self.state.clone(),
            self.free_parameters.clone(),
            self.external_parameters.clone(),
            self.has_successful_solve,
        )
    }
}
