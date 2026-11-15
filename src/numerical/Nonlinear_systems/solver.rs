//! Top-level selection between root-finding and least-squares solvers.

use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolveResult};
use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresError, LeastSquaresProblem, LevenbergMarquardt, MinimizationReport,
};
use crate::numerical::Nonlinear_systems::prelude::NonlinearSolverMethod;
use crate::numerical::Nonlinear_systems::problem::JacobianProvider;
use nalgebra::DVector;

/// Selects one nonlinear solve contract from the crate's shared nonlinear API.
///
/// Root solvers seek a zero of a square system. The least-squares route
/// minimizes a residual norm and also supports rectangular systems.
#[derive(Debug, Clone)]
pub enum NonlinearSolver {
    /// Solve a root-finding problem through the common nonlinear engine.
    Root(NonlinearSolverMethod),
    /// Minimize a residual vector through the canonical rectangular LM core.
    LeastSquares(LevenbergMarquardt),
}

impl NonlinearSolver {
    /// Solves a root-finding problem when this selector contains a root method.
    pub fn solve_root<P: JacobianProvider>(
        &self,
        problem: &P,
        initial_guess: DVector<f64>,
        options: SolveOptions,
    ) -> Result<SolveResult, SolveError> {
        match self {
            Self::Root(method) => method.clone().solve(problem, initial_guess, options),
            Self::LeastSquares(_) => Err(SolveError::InvalidConfig(
                "least-squares selection cannot solve through the root-finding API".to_string(),
            )),
        }
    }

    /// Root-solve dispatch with opt-in cooperative cancellation.
    pub fn solve_root_with_cancellation_token<P: JacobianProvider>(
        &self,
        problem: &P,
        initial_guess: DVector<f64>,
        options: SolveOptions,
        cancellation: crate::numerical::CancellationToken,
    ) -> Result<SolveResult, SolveError> {
        match self {
            Self::Root(method) => method.clone().solve_with_cancellation_token(
                problem,
                initial_guess,
                options,
                cancellation,
            ),
            Self::LeastSquares(_) => Err(SolveError::InvalidConfig(
                "least-squares selection cannot solve through the root-finding API".to_string(),
            )),
        }
    }

    /// Minimizes a residual problem when this selector contains the LM route.
    pub fn minimize_least_squares<P: LeastSquaresProblem>(
        &self,
        problem: P,
    ) -> Result<(P, MinimizationReport), SolveError> {
        match self {
            Self::LeastSquares(method) => Ok(method.minimize(problem)),
            Self::Root(_) => Err(SolveError::InvalidConfig(
                "root-finding selection cannot minimize through the least-squares API".to_string(),
            )),
        }
    }

    /// Fallible least-squares dispatch that preserves callback and numerical
    /// failures as typed [`LeastSquaresError`] values.
    pub fn try_minimize_least_squares<P: LeastSquaresProblem>(
        &self,
        problem: P,
    ) -> Result<(P, MinimizationReport), LeastSquaresError> {
        match self {
            Self::LeastSquares(method) => method.try_minimize(problem),
            _ => Err(LeastSquaresError::WrongSolverKind),
        }
    }

    /// Fallible least-squares dispatch with opt-in cooperative cancellation.
    pub fn try_minimize_least_squares_with_cancellation_token<P: LeastSquaresProblem>(
        &self,
        problem: P,
        cancellation: crate::numerical::CancellationToken,
    ) -> Result<(P, MinimizationReport), LeastSquaresError> {
        match self {
            Self::LeastSquares(method) => {
                method.try_minimize_with_cancellation_token(problem, cancellation)
            }
            _ => Err(LeastSquaresError::WrongSolverKind),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::least_squares::ClosureLeastSquaresProblem;

    #[test]
    fn least_squares_variant_dispatches_to_rectangular_lm() {
        let problem = ClosureLeastSquaresProblem::new(
            DVector::from_element(1, 0.0),
            |x| DVector::from_vec(vec![x[0] - 2.0, 2.0 * x[0] - 4.0]),
            |_| nalgebra::DMatrix::from_column_slice(2, 1, &[1.0, 2.0]),
        );
        let (solved, report) = NonlinearSolver::LeastSquares(LevenbergMarquardt::new())
            .minimize_least_squares(problem)
            .expect("least-squares variant should dispatch");

        assert!(report.termination.was_successful());
        assert!((solved.params()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn root_variant_dispatches_to_the_shared_nonlinear_engine() {
        let problem =
            crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem::from_strings(
                vec!["x - 1".to_string()],
                Some(vec!["x".to_string()]),
                None,
                None,
            )
            .expect("symbolic root problem");

        let result = NonlinearSolver::Root(NonlinearSolverMethod::Newton(
            crate::numerical::Nonlinear_systems::prelude::NewtonMethod,
        ))
        .solve_root(
            &problem,
            DVector::from_element(1, 0.0),
            SolveOptions::default(),
        )
        .expect("root selector should dispatch");

        assert!((result.x[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solver_selector_rejects_the_wrong_problem_contract() {
        let error = NonlinearSolver::LeastSquares(LevenbergMarquardt::new())
            .solve_root(
                &crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem::from_strings(
                    vec!["x - 1".to_string()],
                    Some(vec!["x".to_string()]),
                    None,
                    None,
                )
                .expect("symbolic root problem"),
                DVector::from_element(1, 0.0),
                SolveOptions::default(),
            )
            .expect_err("least-squares variant must reject root dispatch");

        assert!(matches!(error, SolveError::InvalidConfig(_)));
    }

    #[test]
    fn root_selector_forwards_opt_in_cancellation() {
        use crate::numerical::SolverTelemetryMode;

        let problem =
            crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem::from_strings(
                vec!["x - 1".to_string()],
                Some(vec!["x".to_string()]),
                None,
                None,
            )
            .expect("symbolic root problem");
        let token = crate::numerical::CancellationToken::new();
        token.cancel();
        let mut options = SolveOptions::default();
        options.diagnostics.telemetry_mode = SolverTelemetryMode::Counters;

        let error = NonlinearSolver::Root(NonlinearSolverMethod::Newton(
            crate::numerical::Nonlinear_systems::prelude::NewtonMethod,
        ))
        .solve_root_with_cancellation_token(&problem, DVector::from_element(1, 0.0), options, token)
        .expect_err("pre-cancelled root solve must stop");

        let SolveError::Cancelled { snapshot } = error else {
            panic!("expected typed cancellation error");
        };
        assert_eq!(snapshot.completed_iterations, 0);
        assert_eq!(snapshot.current_state, vec![0.0]);
        assert_eq!(snapshot.residual_evaluations, Some(0));
        assert_eq!(snapshot.jacobian_evaluations, Some(0));
        assert_eq!(snapshot.linear_solves, Some(0));
        assert_eq!(snapshot.elapsed, None);
    }

    #[test]
    fn least_squares_selector_forwards_opt_in_cancellation() {
        use crate::numerical::Nonlinear_systems::least_squares::ClosureLeastSquaresProblem;

        let problem = ClosureLeastSquaresProblem::new(
            DVector::from_vec(vec![0.0]),
            |x: &DVector<f64>| DVector::from_element(1, x[0] - 1.0),
            |_x: &DVector<f64>| nalgebra::DMatrix::identity(1, 1),
        );
        let token = crate::numerical::CancellationToken::new();
        token.cancel();
        let result = NonlinearSolver::LeastSquares(LevenbergMarquardt::new())
            .try_minimize_least_squares_with_cancellation_token(problem, token);
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("pre-cancelled least-squares solve unexpectedly succeeded"),
        };

        assert!(matches!(
            error,
            LeastSquaresError::Cancelled { ref snapshot }
                if snapshot.completed_iterations == 0 && snapshot.parameters == vec![0.0]
        ));
    }
}
