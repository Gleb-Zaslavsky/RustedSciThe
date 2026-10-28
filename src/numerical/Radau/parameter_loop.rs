//! Independent ordered parameter sweeps for prepared Radau models.

use super::api::{RadauError, RadauSolver, RadauTelemetryReport};
use crate::numerical::ivp_parameter_loop::{
    IvpParameterLoopReport, IvpParameterLoopSolution, run_ordered,
};

impl RadauSolver {
    /// Solve each parameter vector independently from the same initial state.
    ///
    /// The prepared callbacks are reused; each solve starts at the configured
    /// initial time and uses the supplied `y0`. Output points follow Radau's
    /// configured `FinalOnly`, `Dense` or `Sampled` policy. Native errors and
    /// each attempt's available telemetry are retained per point.
    pub fn try_solve_parameter_loop(
        &mut self,
        y0: &[f64],
        parameter_vectors: &[Vec<f64>],
    ) -> IvpParameterLoopReport<Option<RadauTelemetryReport>, RadauError> {
        run_ordered(parameter_vectors, |parameters| {
            let result = self
                .try_solve_with_parameter_values(y0, parameters)
                .and_then(|solution| {
                    let (x_mesh, values) = solution.output_trajectory()?;
                    Ok(IvpParameterLoopSolution { x_mesh, values })
                });
            (result, self.last_solve_telemetry().cloned())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Radau::api::{
        RadauConfig, RadauOutputPolicy, RadauProblem, RadauTelemetryMode,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn radau_loop_records_failure_and_respects_output_policy() {
        let problem = RadauProblem::new(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t",
        )
        .with_parameter_names(vec!["rate".into()]);
        let config = RadauConfig {
            t_bound: 0.2,
            first_step: Some(0.02),
            max_step: 0.05,
            output: RadauOutputPolicy::Dense,
            telemetry: RadauTelemetryMode::Counters,
            ..RadauConfig::default()
        };
        let mut solver = RadauSolver::prepare(problem, config).unwrap();
        let report = solver.try_solve_parameter_loop(&[1.0], &[vec![0.5], vec![], vec![1.5]]);

        assert_eq!(report.points.len(), 3);
        assert!(report.points[0].result.is_ok());
        assert!(report.points[1].result.is_err());
        assert!(report.points[2].result.is_ok());
        for point in [&report.points[0], &report.points[2]] {
            let trajectory = point.result.as_ref().unwrap();
            assert!(trajectory.x_mesh.len() > 1);
            assert_eq!(trajectory.values[(0, 0)], 1.0);
            assert!(
                point
                    .telemetry
                    .as_ref()
                    .unwrap()
                    .counters
                    .contains_key("accepted_steps")
            );
        }
    }
}
