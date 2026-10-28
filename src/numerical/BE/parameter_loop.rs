//! Independent ordered parameter sweeps for prepared Backward Euler models.

use super::{BE, BeDetailedStatistics, BeError};
use crate::numerical::ivp_parameter_loop::{
    IvpParameterLoopReport, IvpParameterLoopSolution, run_ordered,
};
use crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot;
use crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics;
use nalgebra::DVector;

/// Native Backward Euler telemetry captured after one parameter point.
#[derive(Debug, Clone)]
pub struct BeParameterLoopTelemetry {
    pub statistics: IvpBackendStatistics,
    pub details: BeDetailedStatistics,
    pub symbolic: Option<IvpTelemetrySnapshot>,
}

impl BE {
    /// Solve each parameter vector independently from the same initial state.
    ///
    /// Prepared callbacks are reused, while the time-integration state is
    /// reset for each point. A native [`BeError`] is retained per failed point
    /// and later vectors are still attempted.
    pub fn try_solve_parameter_loop(
        &mut self,
        y0: DVector<f64>,
        parameter_vectors: &[Vec<f64>],
    ) -> IvpParameterLoopReport<BeParameterLoopTelemetry, BeError> {
        run_ordered(parameter_vectors, |parameters| {
            let result = self
                .try_solve_with_parameter_values(y0.clone(), DVector::from_column_slice(parameters))
                .map(|()| {
                    let (x_mesh, values) = self.trajectory();
                    IvpParameterLoopSolution {
                        x_mesh: x_mesh.clone(),
                        values: values.clone(),
                    }
                });
            let telemetry = BeParameterLoopTelemetry {
                statistics: self.get_statistics(),
                details: self.detailed_statistics(),
                symbolic: self.symbolic_ivp_telemetry_snapshot(),
            };
            (result, telemetry)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BE::{BeSolverOptions, BeTelemetryMode};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn backward_euler_loop_records_failure_and_attempts_later_points() {
        let options = BeSolverOptions::new(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t".into(),
            1e-8,
            20,
            Some(0.01),
            0.0,
            0.1,
            DVector::from_element(1, 1.0),
        )
        .with_telemetry_mode(BeTelemetryMode::Counters);
        let mut solver = BE::try_new_with_options(options).unwrap();
        solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
        let report = solver.try_solve_parameter_loop(
            DVector::from_element(1, 1.0),
            &[vec![0.5], vec![0.5, 1.0], vec![1.0]],
        );

        assert_eq!(report.points.len(), 3);
        assert!(report.points[0].result.is_ok());
        assert!(report.points[1].result.is_err());
        assert!(report.points[2].result.is_ok());
        let first = report.points[0].result.as_ref().unwrap();
        let last = report.points[2].result.as_ref().unwrap();
        assert_eq!(first.values[(0, 0)], 1.0);
        assert_eq!(last.values[(0, 0)], 1.0);
        assert_ne!(
            first.values[(first.values.nrows() - 1, 0)],
            last.values[(last.values.nrows() - 1, 0)]
        );
    }
}
