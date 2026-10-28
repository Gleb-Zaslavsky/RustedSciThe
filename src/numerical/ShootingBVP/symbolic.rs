//! Shared Lambdify preparation and RHS evaluation for shooting problems.
//!
//! This adapter prepares equations once, treats the independent coordinate as
//! a numeric equation parameter, and keeps the selected symbolic frontend out
//! of the numerical IVP callback implementation.

use crate::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, PreparationTelemetryMode, PreparedSymbolicNonlinearProblem,
    SymbolicBackendConfig, SymbolicLambdifyFrontend, SymbolicPreparationReport,
    SymbolicProblemOptions,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::sync::Arc;
use std::time::Instant;

/// Configuration for the symbolic route; generated AOT code is intentionally
/// not part of the ShootingBVP modernization.
#[derive(Clone, Copy, Debug)]
pub struct ShootingSymbolicOptions {
    /// Selects the ExprLegacy or AtomViewNative Lambdify frontend.
    pub frontend: SymbolicLambdifyFrontend,
    /// Selects sequential or thresholded parallel symbolic callback execution.
    pub execution_policy: LambdifyExecutionPolicy,
    /// Enables detailed symbolic preparation-stage timings.
    pub preparation_telemetry: PreparationTelemetryMode,
}

impl Default for ShootingSymbolicOptions {
    fn default() -> Self {
        Self {
            frontend: SymbolicLambdifyFrontend::ExprLegacy,
            execution_policy: LambdifyExecutionPolicy::Sequential,
            preparation_telemetry: PreparationTelemetryMode::Disabled,
        }
    }
}

/// Immutable prepared RHS shared by all IVP solves in one shooting search.
pub struct PreparedShootingRhs {
    prepared: Arc<PreparedSymbolicNonlinearProblem>,
    state_dimension: usize,
    preparation_report: SymbolicPreparationReport,
    frontend: SymbolicLambdifyFrontend,
    preparation_wall_time: std::time::Duration,
}

impl PreparedShootingRhs {
    /// Prepares a square first-order state system using the shared Lambdify
    /// infrastructure. The independent coordinate is bound at evaluation time.
    pub fn prepare(
        equations: Vec<Expr>,
        variables: Vec<String>,
        independent_variable: String,
        options: ShootingSymbolicOptions,
    ) -> Result<Self, crate::numerical::Nonlinear_systems::error::SolveError> {
        let started = Instant::now();
        let state_dimension = variables.len();
        let symbolic_options = SymbolicProblemOptions::new()
            .with_variables(variables)
            .with_parameter_names(vec![independent_variable])
            .with_parameter_values(DVector::zeros(1))
            .with_backend_config(SymbolicBackendConfig::lambdify())
            .with_lambdify_frontend(options.frontend)
            .with_lambdify_execution_policy(options.execution_policy)
            .with_preparation_telemetry(options.preparation_telemetry);
        let prepared =
            PreparedSymbolicNonlinearProblem::from_expressions(equations, symbolic_options)?;
        let preparation_wall_time = started.elapsed();
        let preparation_report = prepared.preparation_report().clone();
        Ok(Self {
            prepared: Arc::new(prepared),
            state_dimension,
            preparation_report,
            frontend: options.frontend,
            preparation_wall_time,
        })
    }

    /// Evaluates the prepared RHS at `(coordinate, state)` with typed errors.
    pub fn evaluate(
        &self,
        coordinate: f64,
        state: &DVector<f64>,
    ) -> Result<DVector<f64>, crate::numerical::Nonlinear_systems::error::SolveError> {
        if state.len() != self.state_dimension {
            return Err(
                crate::numerical::Nonlinear_systems::error::SolveError::DimensionMismatch {
                    expected: self.state_dimension,
                    actual: state.len(),
                    context: "shooting RHS state",
                },
            );
        }
        if !coordinate.is_finite() {
            return Err(
                crate::numerical::Nonlinear_systems::error::SolveError::ResidualEvaluation(
                    "independent coordinate is not finite".to_string(),
                ),
            );
        }
        let bound = self.prepared.bind_initial()?;
        bound.residual_with_parameter_values(state, &DVector::from_element(1, coordinate))
    }

    /// Reports which Lambdify frontend was prepared.
    pub fn frontend(&self) -> SymbolicLambdifyFrontend {
        self.frontend
    }

    /// Returns preparation lifecycle and optional detailed stage telemetry.
    pub fn preparation_report(&self) -> &SymbolicPreparationReport {
        &self.preparation_report
    }

    /// Returns total wall time around the shared preparation call.
    pub fn preparation_wall_time(&self) -> std::time::Duration {
        self.preparation_wall_time
    }
}

/// Combined result keeping cold preparation separate from the numerical solve.
#[derive(Debug)]
pub struct ShootingSymbolicSolveReport {
    /// Numerical trajectory and optional solve telemetry.
    pub solve: super::Shooting_simple::ShootingSolveReport,
    /// Frontend used for the prepared symbolic RHS.
    pub frontend: SymbolicLambdifyFrontend,
    /// Preparation report from the shared nonlinear symbolic subsystem.
    pub preparation: SymbolicPreparationReport,
    /// End-to-end preparation wall time, exclusive of solve time.
    pub preparation_wall_time: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_lambdify_frontends_match_on_rhs_and_report_preparation() {
        let equations = vec![
            Expr::parse_expression("y1"),
            Expr::parse_expression("-x*y0"),
        ];
        let state = DVector::from_vec(vec![2.0, 3.0]);
        let mut values = Vec::new();
        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let prepared = PreparedShootingRhs::prepare(
                equations.clone(),
                vec!["y0".to_string(), "y1".to_string()],
                "x".to_string(),
                ShootingSymbolicOptions {
                    frontend,
                    preparation_telemetry: PreparationTelemetryMode::Collect,
                    ..ShootingSymbolicOptions::default()
                },
            )
            .unwrap();
            values.push(prepared.evaluate(0.5, &state).unwrap());
            assert_eq!(prepared.frontend(), frontend);
            assert!(prepared.preparation_report().detailed.is_some());
            assert!(!prepared.preparation_wall_time().is_zero());
        }
        assert!((&values[0] - &values[1]).amax() < 1e-12);
        assert!((values[0][0] - 3.0).abs() < 1e-12);
        assert!((values[0][1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn prepared_rhs_rejects_wrong_state_dimension() {
        let prepared = PreparedShootingRhs::prepare(
            vec![Expr::parse_expression("y1"), Expr::parse_expression("0")],
            vec!["y0".to_string(), "y1".to_string()],
            "x".to_string(),
            ShootingSymbolicOptions::default(),
        )
        .unwrap();
        let error = prepared
            .evaluate(0.0, &DVector::from_element(1, 0.0))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::numerical::Nonlinear_systems::error::SolveError::DimensionMismatch { .. }
        ));
    }
}
