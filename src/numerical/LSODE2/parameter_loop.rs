//! Independent ordered parameter sweeps for prepared LSODE2 models.

use super::solver::{Lsode2Error, Lsode2EvaluationTelemetry, Lsode2SolveSummary, Lsode2Solver};
use super::statistics::Lsode2NativeStatistics;
use crate::numerical::ivp_parameter_loop::{
    IvpParameterLoopReport, IvpParameterLoopSolution, run_ordered,
};
use crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot;
use crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics;
use nalgebra::DVector;

/// Native LSODE2 telemetry captured immediately after a parameter point.
#[derive(Debug, Clone)]
pub struct Lsode2ParameterLoopTelemetry {
    pub statistics: IvpBackendStatistics,
    pub native_statistics: Lsode2NativeStatistics,
    pub symbolic: IvpTelemetrySnapshot,
    pub evaluation: Option<Lsode2EvaluationTelemetry>,
}

impl Lsode2Solver {
    /// Solve each parameter vector independently from the same initial state.
    ///
    /// The prepared symbolic/AOT backend is reused, but each solve starts from
    /// the configured initial time and the supplied `y0`. Native errors are
    /// retained per point and do not stop later vectors.
    pub fn try_solve_parameter_loop(
        &mut self,
        y0: DVector<f64>,
        parameter_vectors: &[Vec<f64>],
    ) -> IvpParameterLoopReport<Lsode2ParameterLoopTelemetry, Lsode2Error> {
        run_ordered(parameter_vectors, |parameters| {
            let result = self
                .try_solve_with_parameter_values(y0.clone(), DVector::from_column_slice(parameters))
                .map(|_summary: Lsode2SolveSummary| {
                    let (x_mesh, values) = self.get_result_ref();
                    IvpParameterLoopSolution {
                        x_mesh: x_mesh.clone(),
                        values: values.clone(),
                    }
                });
            let telemetry = Lsode2ParameterLoopTelemetry {
                statistics: self.statistics(),
                native_statistics: self.native_statistics(),
                symbolic: self.telemetry_snapshot(),
                evaluation: Some(self.summary().evaluation_telemetry),
            };
            (result, telemetry)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::LSODE2::{
        IvpTelemetry, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
        Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn lsode2_loop_keeps_failed_parameter_point_and_runs_the_next_one() {
        let config = Lsode2ProblemConfig::new(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t".into(),
            0.0,
            DVector::from_element(1, 1.0),
            0.2,
            0.02,
            1e-8,
            1e-10,
        )
        .with_equation_parameters(vec!["rate".into()])
        .with_equation_parameter_values(DVector::from_element(1, 1.0))
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_bridge_solve()
        .with_telemetry(IvpTelemetry::detailed());
        let mut solver = Lsode2Solver::new(config).unwrap();
        let report = solver.try_solve_parameter_loop(
            DVector::from_element(1, 1.0),
            &[vec![0.5], vec![], vec![1.5]],
        );

        assert_eq!(report.points.len(), 3);
        assert!(report.points[0].result.is_ok());
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
