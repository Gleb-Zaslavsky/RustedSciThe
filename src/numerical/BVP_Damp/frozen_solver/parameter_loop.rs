//! Ordered fixed-parameter solves for the fixed-mesh Frozen/Naive BVP routes.
//!
//! The loop keeps the most recent successful profile as the next Newton guess.
//! Failed points are reported and skipped; a failed iterate is never reused.

use nalgebra::{DMatrix, DVector};

use super::{BvpBackendIntegrationError, NRBVP};
use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
use crate::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot;

/// Result of one requested parameter point in an ordered Frozen/Naive sweep.
#[derive(Debug)]
pub enum FrozenParameterLoopOutcome {
    /// The point converged; values use the solver's public mesh-by-state layout.
    Converged {
        mesh: DVector<f64>,
        values: DMatrix<f64>,
        telemetry: BvpTelemetrySnapshot,
    },
    /// The nonlinear iteration budget was exhausted without a typed runtime error.
    NotConverged { telemetry: BvpTelemetrySnapshot },
    /// The native solve failed at this point; later points may still be attempted.
    Failed {
        error: BvpBackendIntegrationError,
        telemetry: BvpTelemetrySnapshot,
    },
}

/// One attempted parameter vector and its outcome, in caller-supplied order.
#[derive(Debug)]
pub struct FrozenParameterLoopPoint {
    pub index: usize,
    pub parameters: Vec<f64>,
    pub outcome: FrozenParameterLoopOutcome,
}

/// Ordered parameter-loop report. A stop error means native state could not be
/// safely prepared or restored; already completed point outcomes are retained.
#[derive(Debug, Default)]
pub struct FrozenParameterLoopReport {
    pub points: Vec<FrozenParameterLoopPoint>,
    pub stop_error: Option<BvpBackendIntegrationError>,
}

struct SuccessfulCheckpoint {
    parameters: Vec<f64>,
    iterate: DMatrix<f64>,
    result: DVector<f64>,
}

impl NRBVP {
    /// Solve a caller-ordered sequence of fixed model-parameter vectors.
    ///
    /// The first attempt starts from the solver's configured initial guess.
    /// Every later attempt starts from the most recent successful profile. A
    /// failed point is retained in the report but never becomes a guess; if
    /// there has not yet been a success, the original guess is reused. Native
    /// callbacks are prepared at most once for this loop. Telemetry collection
    /// follows the solver's configured mode; snapshots preserve native
    /// cumulative counter semantics and carry the solve-local trace id.
    ///
    /// Input-shape/configuration errors are returned before any numerical work.
    /// Per-point non-convergence and native solve errors are stored in the
    /// report so the remaining parameter vectors can be attempted.
    pub fn try_solve_parameter_loop(
        &mut self,
        parameter_vectors: &[Vec<f64>],
    ) -> Result<FrozenParameterLoopReport, BvpBackendIntegrationError> {
        validate_loop_input(self, parameter_vectors)?;

        let original_parameters = self.param_values.clone();
        let original_guess = self.initial_guess.clone();
        let original_result = self.result.clone();
        let original_y = self.y.clone_box();

        // Bind before preparing so a new solver creates callbacks for the
        // first requested point. Existing prepared bindings update in place.
        self.try_set_parameter_values(Some(parameter_vectors[0].clone()))?;
        let fingerprint = self.prepared_plan_fingerprint();
        let already_prepared = self
            .prepared_runtime_revision
            .is_current_with_fingerprint(fingerprint)
            && self.factor_owner.prepared_binding_matches(fingerprint);
        if !already_prepared {
            if let Err(error) = self.try_eq_generate() {
                let _ = self.try_set_parameter_values(original_parameters.clone());
                return Err(error);
            }
        }

        let mut report = FrozenParameterLoopReport {
            points: Vec::with_capacity(parameter_vectors.len()),
            stop_error: None,
        };
        let mut successful_checkpoint: Option<SuccessfulCheckpoint> = None;

        for (index, parameters) in parameter_vectors.iter().enumerate() {
            let guess = match &successful_checkpoint {
                Some(checkpoint) => &checkpoint.iterate,
                None => &original_guess,
            };

            // Restore the checkpoint before every attempt, including after a
            // failure, so the mutable solver cannot leak a rejected iterate.
            let set_parameters = self.try_set_parameter_values(Some(parameters.clone()));
            let set_guess =
                set_parameters.and_then(|()| self.try_set_prepared_iterate(guess.clone()));
            if let Err(error) = set_guess {
                report.stop_error = Some(error);
                break;
            }

            self.telemetry_counters.begin_solve();
            self.custom_timer.start();
            let solve_result = self.try_solver_prepared();
            self.custom_timer.finish();
            self.telemetry_counters.record_termination(
                matches!(&solve_result, Ok(Some(_))),
                if matches!(&solve_result, Ok(Some(_))) {
                    1.0
                } else {
                    0.0
                },
            );
            let telemetry = self.get_statistics().telemetry;

            let outcome = match solve_result {
                Ok(Some(solution)) => {
                    let expected = self.values.len() * self.n_steps;
                    if solution.len() != expected {
                        report.stop_error = Some(BvpBackendIntegrationError::InvalidProblem {
                            field: "parameter_loop_solution".into(),
                            message: format!(
                                "Frozen solve returned {} values; expected {expected}",
                                solution.len()
                            ),
                        });
                        break;
                    }
                    let iterate = DMatrix::from_column_slice(
                        self.values.len(),
                        self.n_steps,
                        solution.as_slice(),
                    );
                    let values = iterate.transpose();
                    successful_checkpoint = Some(SuccessfulCheckpoint {
                        parameters: parameters.clone(),
                        iterate,
                        result: solution,
                    });
                    FrozenParameterLoopOutcome::Converged {
                        mesh: self.x_mesh.clone(),
                        values,
                        telemetry,
                    }
                }
                Ok(None) => FrozenParameterLoopOutcome::NotConverged { telemetry },
                Err(error) => FrozenParameterLoopOutcome::Failed { error, telemetry },
            };

            report.points.push(FrozenParameterLoopPoint {
                index,
                parameters: parameters.clone(),
                outcome,
            });
        }

        if let Some(checkpoint) = successful_checkpoint {
            if let Err(error) = self.restore_checkpoint(
                Some(checkpoint.parameters),
                &checkpoint.iterate,
                Some(checkpoint.result),
            ) {
                report.stop_error = Some(error);
            }
        } else if let Err(error) =
            self.restore_checkpoint(original_parameters, &original_guess, original_result)
        {
            self.y = original_y;
            report.stop_error = Some(error);
        } else {
            self.y = original_y;
        }

        Ok(report)
    }

    fn restore_checkpoint(
        &mut self,
        parameters: Option<Vec<f64>>,
        iterate: &DMatrix<f64>,
        result: Option<DVector<f64>>,
    ) -> Result<(), BvpBackendIntegrationError> {
        self.try_set_parameter_values(parameters)?;
        self.try_set_prepared_iterate(iterate.clone())?;
        self.result = result;
        let flat = DVector::from_column_slice(iterate.as_slice());
        self.y = Vectors_type_casting(&flat, self.method.clone());
        Ok(())
    }
}

fn validate_loop_input(
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
    for (index, values) in parameter_vectors.iter().enumerate() {
        if values.len() != solver.param_names.len() {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "parameter_loop".into(),
                value: format!("point {index} has {} values", values.len()),
                message: format!(
                    "expected {} values for the bound parameter schema",
                    solver.param_names.len()
                ),
            });
        }
        if let Some(value_index) = values.iter().position(|value| !value.is_finite()) {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "parameter_loop".into(),
                value: format!("point {index}, value {value_index} is non-finite"),
                message: "all fixed parameter values must be finite".into(),
            });
        }
    }
    let expected = (solver.values.len(), solver.n_steps);
    if solver.initial_guess.shape() != expected {
        return Err(BvpBackendIntegrationError::InvalidProblem {
            field: "initial_guess".into(),
            message: format!(
                "expected shape {}x{}, got {}x{}",
                expected.0,
                expected.1,
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
    use crate::numerical::BVP_Damp::BvpTelemetryMode;
    use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    fn parameterized_frozen_solver() -> NRBVP {
        let mut options = FrozenSolverOptions::sparse_frozen().with_tolerance(1e-9);
        let backend = options
            .generated_backend_config
            .clone()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
        options.generated_backend_config = backend;

        let mut solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("alpha^0.5")],
            DMatrix::zeros(1, 6),
            vec!["y".to_owned()],
            "x".to_owned(),
            HashMap::from([("y".to_owned(), vec![(0, 0.0)])]),
            0.0,
            1.0,
            6,
            options,
        );
        solver
            .try_set_parameter_names(Some(&["alpha"]))
            .expect("fixed parameter schema should be accepted");
        solver.dont_save_log(true);
        solver
    }

    #[test]
    fn failed_middle_point_does_not_replace_last_successful_guess() {
        let mut solver = parameterized_frozen_solver();
        solver
            .try_set_parameter_values(Some(vec![1.0]))
            .expect("initial fixed parameter should be accepted");
        solver
            .try_eq_generate()
            .expect("callbacks should prepare before the loop");
        let residual_callback = &*solver.fun as *const _ as *const ();
        let jacobian_callback = solver
            .jac
            .as_ref()
            .map(|callback| &**callback as *const _ as *const ())
            .expect("symbolic preparation should install a Jacobian");
        let report = solver
            .try_solve_parameter_loop(&[vec![1.0], vec![-1.0], vec![4.0]])
            .expect("valid ordered parameter sequence should run");

        assert_eq!(report.points.len(), 3);
        assert_eq!(report.points[0].index, 0);
        assert_eq!(report.points[1].index, 1);
        assert_eq!(report.points[2].index, 2);
        assert!(report.stop_error.is_none());
        assert!(matches!(
            report.points[0].outcome,
            FrozenParameterLoopOutcome::Converged { .. }
        ));
        let FrozenParameterLoopOutcome::Failed { error, .. } = &report.points[1].outcome else {
            panic!("negative square-root parameter must produce a typed callback failure");
        };
        assert!(
            matches!(
                error,
                BvpBackendIntegrationError::CallbackExecutionFailed { .. }
                    | BvpBackendIntegrationError::NonFiniteCallbackValue { .. }
                    | BvpBackendIntegrationError::LinearSolveFailed { .. }
            ),
            "unexpected typed point error: {error:?}"
        );
        assert!(matches!(
            report.points[2].outcome,
            FrozenParameterLoopOutcome::Converged { .. }
        ));

        let FrozenParameterLoopOutcome::Converged {
            mesh: first_mesh,
            values: first_values,
            telemetry: first_telemetry,
        } = &report.points[0].outcome
        else {
            panic!("first parameter point should converge");
        };
        let FrozenParameterLoopOutcome::Converged {
            mesh: last_mesh,
            values: last_values,
            telemetry: last_telemetry,
        } = &report.points[2].outcome
        else {
            panic!("last parameter point should converge after the failed point");
        };
        assert_eq!(first_mesh, last_mesh);
        assert_eq!(first_values.shape(), (6, 1));
        assert_eq!(last_values.shape(), (6, 1));
        assert!(last_values[(5, 0)] > first_values[(5, 0)]);
        assert_eq!(first_telemetry.telemetry_mode, BvpTelemetryMode::Counters);
        assert_eq!(last_telemetry.telemetry_mode, BvpTelemetryMode::Counters);
        assert!(last_telemetry.solve_id > first_telemetry.solve_id);

        assert_eq!(solver.param_values, Some(vec![4.0]));
        assert_eq!(
            residual_callback, &*solver.fun as *const _ as *const (),
            "the callback plan should be reused across all points"
        );
        assert_eq!(
            jacobian_callback,
            solver
                .jac
                .as_ref()
                .map(|callback| &**callback as *const _ as *const ())
                .expect("the Jacobian callback should remain installed")
        );
        let retained = solver.get_result().expect("last success stays inspectable");
        assert_eq!(retained, *last_values);
    }

    #[test]
    fn malformed_loop_input_is_rejected_before_preparation_or_solving() {
        let mut solver = parameterized_frozen_solver();
        let error = solver
            .try_solve_parameter_loop(&[vec![1.0], vec![2.0, 3.0]])
            .expect_err("wrong parameter dimension must be rejected up front");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidSolverConfiguration {
                field,
                ..
            } if field == "parameter_loop"
        ));
        assert!(solver.jac.is_none(), "validation must precede preparation");
        assert!(solver.result.is_none());
    }

    #[test]
    fn first_failure_keeps_the_original_guess_for_the_next_point() {
        let mut solver = parameterized_frozen_solver();
        let report = solver
            .try_solve_parameter_loop(&[vec![-1.0], vec![1.0]])
            .expect("a point failure should not abort later attempts");

        assert_eq!(report.points.len(), 2);
        assert!(matches!(
            report.points[0].outcome,
            FrozenParameterLoopOutcome::Failed { .. }
        ));
        assert!(matches!(
            report.points[1].outcome,
            FrozenParameterLoopOutcome::Converged { .. }
        ));
        assert!(report.stop_error.is_none());
    }

    #[test]
    fn all_failed_points_restore_the_pre_loop_solver_checkpoint() {
        let mut solver = parameterized_frozen_solver();
        solver
            .try_set_parameter_values(Some(vec![1.0]))
            .expect("pre-loop parameter value should be valid");
        solver
            .try_solver()
            .expect("baseline solver should run")
            .expect("baseline solve should converge");
        let prior_result = solver.result.clone();
        let prior_guess = solver.initial_guess.clone();

        let report = solver
            .try_solve_parameter_loop(&[vec![-1.0], vec![-4.0]])
            .expect("numerical failures should be reported, not raised");

        assert_eq!(report.points.len(), 2);
        assert!(
            report
                .points
                .iter()
                .all(|point| matches!(point.outcome, FrozenParameterLoopOutcome::Failed { .. }))
        );
        assert!(report.stop_error.is_none());
        assert_eq!(solver.param_values, Some(vec![1.0]));
        assert_eq!(solver.initial_guess, prior_guess);
        assert_eq!(solver.result, prior_result);
    }

    #[test]
    fn empty_loop_is_rejected_without_mutating_solver_state() {
        let mut solver = parameterized_frozen_solver();
        let original_guess = solver.initial_guess.clone();
        let error = solver
            .try_solve_parameter_loop(&[])
            .expect_err("an empty parameter sequence is not a solve request");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidSolverConfiguration {
                field,
                ..
            } if field == "parameter_loop"
        ));
        assert_eq!(solver.initial_guess, original_guess);
        assert!(solver.jac.is_none());
    }
}
