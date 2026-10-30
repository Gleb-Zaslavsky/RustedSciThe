//! Ordered fixed-parameter sweeps for the adaptive Damped BVP solver.
//!
//! A successful point checkpoints the refined mesh and its converged free
//! iterate together. Failed points are reported and never become the next
//! initial guess. Mesh-sized callback and factorization state is regenerated
//! when recovery has to move back to a different checkpoint mesh.

use nalgebra::{DMatrix, DVector};
use std::time::Instant;

use super::{BvpBackendIntegrationError, NRBVP};
use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
use crate::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot;
use crate::numerical::logging::SolverLogGuard;

/// Result of one requested parameter point in an adaptive Damped sweep.
#[derive(Debug)]
pub enum DampedParameterLoopOutcome {
    /// The point converged; values use the public mesh-by-state orientation.
    Converged {
        mesh: DVector<f64>,
        values: DMatrix<f64>,
        telemetry: BvpTelemetrySnapshot,
    },
    /// The iteration/refinement policy returned without a converged solution.
    NotConverged { telemetry: BvpTelemetrySnapshot },
    /// A typed native solve failure; following points may still be attempted.
    Failed {
        error: BvpBackendIntegrationError,
        telemetry: BvpTelemetrySnapshot,
    },
}

/// One attempted parameter vector, in caller-specified order.
#[derive(Debug)]
pub struct DampedParameterLoopPoint {
    pub index: usize,
    pub parameters: Vec<f64>,
    pub outcome: DampedParameterLoopOutcome,
}

/// Results of an ordered Damped parameter sweep.
#[derive(Debug, Default)]
pub struct DampedParameterLoopReport {
    pub points: Vec<DampedParameterLoopPoint>,
    /// Unrecoverable setup/restoration failure; completed point records remain.
    pub stop_error: Option<BvpBackendIntegrationError>,
}

#[derive(Clone)]
struct LoopCheckpoint {
    parameters: Option<Vec<f64>>,
    mesh: DVector<f64>,
    iterate: DMatrix<f64>,
    initial_guess: DMatrix<f64>,
    result: Option<DVector<f64>>,
    full_result: Option<DMatrix<f64>>,
    grid_refinements: usize,
    prepared: bool,
    bandwidth: (usize, usize),
}

impl NRBVP {
    /// Solve a caller-ordered list of fixed parameter vectors.
    ///
    /// The first attempt uses the configured guess. Each subsequent attempt
    /// uses the most recent successful solution on its associated (possibly
    /// refined and nonuniform) mesh. Failed attempts are recorded and skipped.
    /// A later point reuses a prepared plan if its checkpoint mesh matches the
    /// current plan; otherwise the mesh-sized symbolic/runtime plan is rebuilt.
    /// Telemetry follows the solver's configured mode and remains solve-local
    /// in its trace identifier while native counters retain native semantics.
    pub fn try_solve_parameter_loop(
        &mut self,
        parameter_vectors: &[Vec<f64>],
    ) -> Result<DampedParameterLoopReport, BvpBackendIntegrationError> {
        validate_input(self, parameter_vectors)?;

        let original_initial_guess = self.initial_guess.clone();
        let original = self.capture_checkpoint(self.has_current_prepared_plan());
        let mut first_start = original.clone();
        // Point zero is defined by the configured guess, not by a result left
        // behind by an earlier solve on this mutable solver instance.
        first_start.iterate = original_initial_guess.clone();
        first_start.initial_guess = original_initial_guess.clone();
        first_start.result = None;
        first_start.full_result = None;
        first_start.grid_refinements = 0;
        let original_y = self.y.clone_box();
        let original_result = self.result.clone();
        let original_full_result = self.full_result.clone();
        let original_grid_refinements = self.grid_refinemens;
        let mut report = DampedParameterLoopReport {
            points: Vec::with_capacity(parameter_vectors.len()),
            stop_error: None,
        };
        let mut successful: Option<LoopCheckpoint> = None;

        for (index, parameters) in parameter_vectors.iter().enumerate() {
            let restore_result = if let Some(checkpoint) = successful.as_ref() {
                self.restore_checkpoint(checkpoint, Some(parameters.clone()))
            } else {
                self.restore_checkpoint(&first_start, Some(parameters.clone()))
            };
            if let Err(error) = restore_result {
                report.stop_error = Some(error);
                break;
            }
            if !self.has_current_prepared_plan() {
                let bandwidth = (self.factor_owner.bandwidth() != (0, 0))
                    .then_some(self.factor_owner.bandwidth());
                if let Err(error) = self.try_eq_generate(None, bandwidth) {
                    report.stop_error = Some(error);
                    break;
                }
            }

            let _logging_scope = match self.effective_logging_level().and_then(|level| {
                SolverLogGuard::try_enter(level)
                    .map_err(|source| BvpBackendIntegrationError::LogSetupFailed { source })
            }) {
                Ok(scope) => scope,
                Err(error) => {
                    report.stop_error = Some(error);
                    break;
                }
            };
            self.telemetry_counters.begin_solve();
            self.custom_timer.start();
            let started = Instant::now();
            let solve_result = self.try_main_loop_damped();
            self.custom_timer.finish();
            self.telemetry_counters.finish_solve(&solve_result);
            self.telemetry_counters.record_termination(
                matches!(&solve_result, Ok(Some(_))),
                if matches!(&solve_result, Ok(Some(_))) {
                    1.0
                } else {
                    0.0
                },
            );
            let _elapsed = started.elapsed();
            let telemetry = self.get_statistics().telemetry;

            let outcome = match solve_result {
                Ok(Some(solution)) => {
                    self.handle_result();
                    self.calc_statistics();
                    self.custom_timer.get_all();
                    let Some(values) = self.full_result.clone() else {
                        report.stop_error = Some(BvpBackendIntegrationError::InvalidProblem {
                            field: "parameter_loop_full_result".to_owned(),
                            message: "converged Damped solve did not produce a full profile".into(),
                        });
                        break;
                    };
                    if values.nrows() != self.x_mesh.len()
                        || values.ncols() != self.values.len()
                        || solution.len() != self.values.len() * self.n_steps
                    {
                        report.stop_error = Some(BvpBackendIntegrationError::InvalidProblem {
                            field: "parameter_loop_solution_shape".to_owned(),
                            message: format!(
                                "mesh/profile/free-iterate shape mismatch: mesh={}, profile={}x{}, iterate={}, expected={}",
                                self.x_mesh.len(),
                                values.nrows(),
                                values.ncols(),
                                solution.len(),
                                self.values.len() * self.n_steps
                            ),
                        });
                        break;
                    }
                    successful = Some(self.capture_checkpoint(true));
                    DampedParameterLoopOutcome::Converged {
                        mesh: self.x_mesh.clone(),
                        values,
                        telemetry,
                    }
                }
                Ok(None) => DampedParameterLoopOutcome::NotConverged { telemetry },
                Err(error) => DampedParameterLoopOutcome::Failed { error, telemetry },
            };
            report.points.push(DampedParameterLoopPoint {
                index,
                parameters: parameters.clone(),
                outcome,
            });
        }

        if let Some(checkpoint) = successful.as_ref() {
            if let Err(error) = self.restore_checkpoint(checkpoint, checkpoint.parameters.clone()) {
                report.stop_error = Some(error);
            }
        } else {
            if let Err(error) = self.restore_checkpoint(&original, original.parameters.clone()) {
                report.stop_error = Some(error);
            } else {
                self.initial_guess = original_initial_guess;
                self.y = original_y;
                self.result = original_result;
                self.full_result = original_full_result;
                self.grid_refinemens = original_grid_refinements;
            }
        }
        Ok(report)
    }

    fn has_current_prepared_plan(&self) -> bool {
        self.prepared_runtime_revision
            .is_current_with_fingerprint(self.prepared_plan_fingerprint())
            && self
                .factor_owner
                .prepared_binding_matches(self.prepared_plan_fingerprint())
    }

    fn capture_checkpoint(&self, prepared: bool) -> LoopCheckpoint {
        LoopCheckpoint {
            parameters: self.param_values.clone(),
            mesh: self.x_mesh.clone(),
            iterate: if prepared {
                self.result
                    .as_ref()
                    .and_then(|result| {
                        (result.len() == self.values.len() * self.n_steps).then(|| {
                            DMatrix::from_column_slice(
                                self.values.len(),
                                self.n_steps,
                                result.as_slice(),
                            )
                        })
                    })
                    .unwrap_or_else(|| self.initial_guess.clone())
            } else {
                self.initial_guess.clone()
            },
            initial_guess: self.initial_guess.clone(),
            result: self.result.clone(),
            full_result: self.full_result.clone(),
            grid_refinements: self.grid_refinemens,
            prepared,
            bandwidth: self.factor_owner.bandwidth(),
        }
    }

    fn restore_checkpoint(
        &mut self,
        checkpoint: &LoopCheckpoint,
        parameters: Option<Vec<f64>>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let mesh_changed = self.x_mesh != checkpoint.mesh;
        let iterate = checkpoint.iterate.clone();
        let flat = DVector::from_column_slice(iterate.as_slice());

        self.try_set_parameter_values(parameters)?;
        self.n_steps = checkpoint.mesh.len().checked_sub(1).ok_or_else(|| {
            BvpBackendIntegrationError::InvalidProblem {
                field: "parameter_loop_mesh".to_owned(),
                message: "checkpoint mesh must contain at least two points".into(),
            }
        })?;
        self.t0 = checkpoint.mesh[0];
        self.t_end = checkpoint.mesh[checkpoint.mesh.len() - 1];
        self.x_mesh = checkpoint.mesh.clone();
        self.initial_guess = checkpoint.initial_guess.clone();
        self.y = Vectors_type_casting(&flat, self.effective_runtime_method());
        self.result = checkpoint.result.clone();
        self.full_result = checkpoint.full_result.clone();
        // Each parameter point gets its own adaptive-refinement budget from
        // the checkpoint mesh, rather than inheriting the previous solve's
        // exhausted counter.
        self.grid_refinemens = 0;
        self.number_of_refined_intervals = 0;
        self.nodes_added.clear();
        self.prepared_iterate = false;

        if mesh_changed {
            self.invalidate_linear_runtime();
            self.prepared_runtime_revision.mesh_changed();
            self.jac = None;
            self.bounds_vec.clear();
            self.rel_tolerance_vec.clear();
            self.factor_owner.clear_layout();
            self.variable_string.clear();
            if checkpoint.prepared {
                self.try_eq_generate(
                    Some(checkpoint.mesh.iter().copied().collect()),
                    Some(checkpoint.bandwidth),
                )?;
            }
        } else if checkpoint.prepared && !self.has_current_prepared_plan() {
            self.try_eq_generate(None, Some(checkpoint.bandwidth))?;
        }

        if checkpoint.prepared {
            self.try_set_prepared_iterate(iterate)?;
        }
        self.result = checkpoint.result.clone();
        self.full_result = checkpoint.full_result.clone();
        if !checkpoint.prepared {
            self.grid_refinemens = checkpoint.grid_refinements;
        }
        Ok(())
    }
}

fn validate_input(
    solver: &NRBVP,
    parameter_vectors: &[Vec<f64>],
) -> Result<(), BvpBackendIntegrationError> {
    if parameter_vectors.is_empty() {
        return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
            field: "parameter_loop".into(),
            value: "empty sequence".into(),
            message: "at least one parameter vector is required".into(),
        });
    }
    if solver.param_names.is_empty() {
        return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
            field: "parameter_loop".into(),
            value: "no fixed parameter schema".into(),
            message: "declare fixed parameter names before running a parameter loop".into(),
        });
    }
    for (index, parameters) in parameter_vectors.iter().enumerate() {
        if parameters.len() != solver.param_names.len() {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "parameter_loop".into(),
                value: format!("point {index} has {} values", parameters.len()),
                message: format!(
                    "expected {} values for the bound parameter schema",
                    solver.param_names.len()
                ),
            });
        }
        if let Some(value_index) = parameters.iter().position(|value| !value.is_finite()) {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "parameter_loop".into(),
                value: format!("point {index}, value {value_index} is non-finite"),
                message: "all fixed parameter values must be finite".into(),
            });
        }
    }
    if solver.initial_guess.shape() != (solver.values.len(), solver.n_steps) {
        return Err(BvpBackendIntegrationError::InvalidProblem {
            field: "initial_guess".into(),
            message: format!(
                "expected shape {}x{}, got {}x{}",
                solver.values.len(),
                solver.n_steps,
                solver.initial_guess.nrows(),
                solver.initial_guess.ncols()
            ),
        });
    }
    if let Some(index) = solver
        .initial_guess
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
            stage: "parameter_loop_initial_guess".into(),
            index,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        AdaptiveGridConfig, DampedSolverOptions, SolverParams,
    };
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    fn adaptive_parameterized_solver() -> NRBVP {
        let mut options = DampedSolverOptions::dense_damped()
            .with_abs_tolerance(1e-9)
            .with_rel_tolerance(HashMap::from([("y".to_owned(), 1e-8)]))
            .with_bounds(HashMap::from([("y".to_owned(), (-10.0, 10.0))]))
            .with_max_iterations(12)
            .with_strategy_params(Some(SolverParams {
                adaptive: Some(AdaptiveGridConfig {
                    version: 1,
                    max_refinements: 1,
                    grid_method: GridRefinementMethod::DoublePoints,
                }),
                ..SolverParams::default()
            }))
            .with_bvp_telemetry_mode(BvpTelemetryMode::Counters);
        let generated = options
            .generated_backend_config
            .clone()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
        options.generated_backend_config = generated;

        let mut solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("alpha^0.5")],
            DMatrix::zeros(1, 4),
            vec!["y".to_owned()],
            "x".to_owned(),
            HashMap::from([("y".to_owned(), vec![(0, 0.0)])]),
            0.0,
            1.0,
            4,
            options,
        );
        solver
            .try_set_parameter_names(Some(&["alpha"]))
            .expect("fixed parameter schema is valid");
        solver
            .try_set_parameter_values(Some(vec![1.0]))
            .expect("initial parameter is valid");
        solver.dont_save_log(true);
        solver
    }

    #[test]
    fn adaptive_checkpoint_restores_refined_mesh_across_success_failure_success() {
        let mut solver = adaptive_parameterized_solver();
        let report = solver
            .try_solve_parameter_loop(&[vec![1.0], vec![-1.0], vec![4.0]])
            .expect("valid fixed-parameter path should produce a report");

        assert_eq!(report.points.len(), 3);
        assert!(report.stop_error.is_none());
        let first_mesh = match &report.points[0].outcome {
            DampedParameterLoopOutcome::Converged {
                mesh,
                values,
                telemetry,
            } => {
                assert_eq!(values.nrows(), mesh.len());
                assert_eq!(values.ncols(), 1);
                assert_eq!(telemetry.telemetry_mode, BvpTelemetryMode::Counters);
                assert!(telemetry.counters.iterations > 0);
                for (node, x) in mesh.iter().enumerate() {
                    assert!((values[(node, 0)] - x).abs() < 1e-7);
                }
                mesh.clone()
            }
            other => panic!("first point should converge: {other:?}"),
        };
        assert!(matches!(
            report.points[1].outcome,
            DampedParameterLoopOutcome::Failed { .. }
        ));
        let (last_mesh, last_values) = match &report.points[2].outcome {
            DampedParameterLoopOutcome::Converged { mesh, values, .. } => {
                assert_eq!(values.nrows(), mesh.len());
                assert_eq!(values.ncols(), 1);
                assert!((values[(mesh.len() - 1, 0)] - 2.0).abs() < 1e-7);
                for (node, x) in mesh.iter().enumerate() {
                    assert!((values[(node, 0)] - 2.0 * x).abs() < 1e-7);
                }
                (mesh.clone(), values.clone())
            }
            other => panic!("last point should recover and converge: {other:?}"),
        };
        assert!(first_mesh.len() > 4, "first solve should refine its mesh");
        assert!(
            last_mesh.len() >= first_mesh.len(),
            "the final solve should retain at least the successful refined mesh"
        );
        assert!(
            last_mesh.len() > 4,
            "the final profile should remain adaptive"
        );
        assert_eq!(solver.param_values, Some(vec![4.0]));
        assert_eq!(solver.x_mesh, last_mesh);
        assert_eq!(solver.full_result.as_ref().unwrap(), &last_values);
    }

    #[test]
    fn malformed_input_is_rejected_before_preparation_or_mutation() {
        let mut solver = adaptive_parameterized_solver();
        let original_guess = solver.initial_guess.clone();
        let error = solver
            .try_solve_parameter_loop(&[vec![1.0], vec![2.0, 3.0]])
            .expect_err("all parameter dimensions are validated before solving");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidSolverConfiguration { field, .. }
                if field == "parameter_loop"
        ));
        assert!(solver.jac.is_none());
        assert_eq!(solver.initial_guess, original_guess);
    }
}
