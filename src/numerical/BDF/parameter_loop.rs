//! Independent ordered parameter sweeps for prepared BDF models.

use super::BDF_api::{BdfPreparationTimings, BdfSolveError, ODEsolver};
use crate::numerical::ivp_parameter_loop::{
    IvpParameterLoopReport, IvpParameterLoopSolution, run_ordered,
};
use crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot;
use crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics;
use nalgebra::DVector;

/// Native BDF telemetry captured immediately after one parameter point.
#[derive(Debug, Clone)]
pub struct BdfParameterLoopTelemetry {
    pub statistics: IvpBackendStatistics,
    pub symbolic: Option<IvpTelemetrySnapshot>,
    pub preparation_timings: BdfPreparationTimings,
}

impl ODEsolver {
    /// Solve each parameter vector as an independent IVP from the same `y0`.
    ///
    /// The prepared callbacks are reused, but integration history and initial
    /// state are reset for every point. Failures retain their native
    /// [`BdfSolveError`] and do not stop later points.
    pub fn try_solve_parameter_loop(
        &mut self,
        y0: DVector<f64>,
        parameter_vectors: &[Vec<f64>],
    ) -> IvpParameterLoopReport<BdfParameterLoopTelemetry, BdfSolveError> {
        run_ordered(parameter_vectors, |parameters| {
            let result = self
                .try_solve_with_parameter_values(y0.clone(), DVector::from_column_slice(parameters))
                .map(|()| {
                    let (x_mesh, values) = self.get_result_ref();
                    IvpParameterLoopSolution {
                        x_mesh: x_mesh.clone(),
                        values: values.clone(),
                    }
                });
            let telemetry = BdfParameterLoopTelemetry {
                statistics: self.get_statistics(),
                symbolic: self.preparation_backend_telemetry().cloned(),
                preparation_timings: self.preparation_timings(),
            };
            (result, telemetry)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn bdf_loop_keeps_order_continues_after_typed_failure_and_reuses_same_initial_state() {
        let options = BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t".into(),
            0.0,
            DVector::from_element(1, 1.0),
            0.2,
            0.02,
            1e-8,
            1e-10,
            None,
            false,
            Some(0.01),
        )
        .with_equation_parameters(vec!["rate".into()])
        .with_equation_parameter_values(DVector::from_element(1, 1.0))
        .with_telemetry_mode(BdfTelemetryMode::Counters);
        let mut solver = ODEsolver::new_with_options(options);
        solver.try_generate().unwrap();
        let report = solver.try_solve_parameter_loop(
            DVector::from_element(1, 1.0),
            &[vec![1.0], vec![], vec![2.0]],
        );

        assert_eq!(report.points.len(), 3);
        assert!(
            report.points[0].result.is_ok(),
            "{:?}",
            report.points[0].result
        );
        assert!(report.points[1].result.is_err());
        assert!(report.points[2].result.is_ok());
        for point in [&report.points[0], &report.points[2]] {
            let trajectory = point.result.as_ref().unwrap();
            assert_eq!(trajectory.values.nrows(), trajectory.x_mesh.len());
            assert_eq!(trajectory.values[(0, 0)], 1.0);
            assert!(point.telemetry.statistics.solve_calls > 0);
        }
    }
}
