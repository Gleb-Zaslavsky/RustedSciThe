//! Shared data types for ordered IVP parameter sweeps.
//!
//! Every point is a fresh solve from the caller's fixed initial state and
//! interval. A failed point is retained in the report and does not prevent
//! later parameter vectors from being attempted.

use nalgebra::{DMatrix, DVector};

/// Time axis and row-oriented state values from one successful solve.
#[derive(Debug, Clone, PartialEq)]
pub struct IvpParameterLoopSolution {
    /// Solver output times. For solvers with adaptive accepted-step output,
    /// these are the retained mesh points; sampled-output solvers use their
    /// requested sample axis.
    pub x_mesh: DVector<f64>,
    /// State values with shape `(time_points, state_variables)`.
    pub values: DMatrix<f64>,
}

/// One attempted parameter vector and its solver-native outcome/telemetry.
#[derive(Debug)]
pub struct IvpParameterLoopPoint<Telemetry, Error> {
    pub index: usize,
    pub parameter_values: Vec<f64>,
    pub result: Result<IvpParameterLoopSolution, Error>,
    /// Native telemetry snapshot after this point; absent native telemetry is
    /// represented by the solver's native empty/disabled snapshot semantics.
    /// Counter reset and accumulation scope remain solver-native; the loop
    /// does not synthesize per-point deltas.
    pub telemetry: Telemetry,
}

/// Results for all requested parameter vectors in caller-specified order.
#[derive(Debug, Default)]
pub struct IvpParameterLoopReport<Telemetry, Error> {
    pub points: Vec<IvpParameterLoopPoint<Telemetry, Error>>,
}

/// Execute an already validated ordered list while keeping every point local.
pub(crate) fn run_ordered<Telemetry, Error>(
    parameter_vectors: &[Vec<f64>],
    mut solve: impl FnMut(&[f64]) -> (Result<IvpParameterLoopSolution, Error>, Telemetry),
) -> IvpParameterLoopReport<Telemetry, Error> {
    let points = parameter_vectors
        .iter()
        .enumerate()
        .map(|(index, parameter_values)| {
            let (result, telemetry) = solve(parameter_values);
            IvpParameterLoopPoint {
                index,
                parameter_values: parameter_values.clone(),
                result,
                telemetry,
            }
        })
        .collect();
    IvpParameterLoopReport { points }
}
