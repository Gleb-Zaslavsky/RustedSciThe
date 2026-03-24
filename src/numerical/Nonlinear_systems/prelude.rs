//! Prelude module for convenient import of all nonlinear solvers and related types.
//!
//! This module re-exports the most commonly used items from the `Nonlinear_systems` submodules.
//! To use, import with:
//! ```
//! use RustedSciThe::numerical::Nonlinear_systems::prelude::*;
//! ```

// Core engine types
pub use crate::numerical::Nonlinear_systems::engine::{
    DiagnosticsOptions, EngineLogLevel, IterationRecord, IterationState, LinearSolverKind,
    MemoryDiagnostics, NewtonMethod, NonlinearMethod, RuntimeDiagnostics, SolveOptions,
    SolveResult, SolveStatistics, SolverEngine, StepOutcome, scaled_norm, scaling_vector,
    solve_linear_system,
};

// Error types
pub use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};

// Problem definitions and bounds
pub use crate::numerical::Nonlinear_systems::problem::{
    Bounds, JacobianProvider, NonlinearProblem,
};

// Levenberg-Marquardt variants (vanilla)
pub use crate::numerical::Nonlinear_systems::LM_vanilla::{
    LMMinpackState, LevenbergMarquardtMethod, LevenbergMarquardtMinpack, LevenbergMarquardtState,
};

// Nielsen-style Levenberg-Marquardt with trust region
pub use crate::numerical::Nonlinear_systems::LM_Nielsen::{
    NielsenLevenbergMarquardtMethod, NielsenLevenbergMarquardtMethodAdvanced,
    NielsenLevenbergMarquardtState, NielsenLevenbergMarquardtStateAdvanced,
};

// Utilities for LM algorithms (scaling, reduction ratio, convergence)
pub use crate::numerical::Nonlinear_systems::LM_utils::{
    ConvergenceCriteria, ConvergenceCriteriaSolver, ConvergenceInfo, ReductionRatio,
    ReductionRatioSolver, ScalingMethod, TrustRegionScaling, compute_scaled_gradient_norm2,
    scaled_norm_common, test_convergence_gsl,
};

// Damped Newton methods
pub use crate::numerical::Nonlinear_systems::NR_damped::{
    DampedNewtonMethod, DampedNewtonMethodAdvanced, bound_step,
};

// Trust region methods
pub use crate::numerical::Nonlinear_systems::trust_region::{
    DoglegError, DoglegSolver, DoglegState, Powell_dogleg_method, PowellDoglegMethod,
    PowellDoglegState, TrustRegionMethod, TrustRegionState,
};
pub use crate::numerical::Nonlinear_systems::trust_region_LM::{
    TrustRegionLMMethod, TrustRegionLMState, TrustRegionResult, solve_trust_region_subproblem,
};

// Symbolic problem adapter
pub use crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem;

/// User-facing wrapper around every nonlinear-system solver exported by the prelude.
///
/// This enum lets the caller choose a method from one place instead of importing
/// every solver module separately.
#[derive(Debug, Clone)]
pub enum NonlinearSolverMethod {
    /// Classical Newton method.
    ///
    /// Fast near the solution, but sensitive to the initial guess and to singular Jacobians.
    Newton(NewtonMethod),
    /// Newton method with backtracking line search.
    ///
    /// More robust than plain Newton when the full step is too aggressive.
    DampedNewton(DampedNewtonMethod),
    /// Bound-aware damped Newton method.
    ///
    /// Useful for constrained problems where clipping and bound-respecting damping matter.
    DampedNewtonAdvanced(DampedNewtonMethodAdvanced),
    /// Classical Levenberg-Marquardt method.
    ///
    /// Blends Gauss-Newton and gradient descent through a damping parameter.
    LevenbergMarquardt(LevenbergMarquardtMethod),
    /// MINPACK-style Levenberg-Marquardt method.
    ///
    /// Uses MINPACK-inspired trust-region logic and is a good general-purpose least-squares choice.
    LevenbergMarquardtMinpack(LevenbergMarquardtMinpack),
    /// Nielsen's Levenberg-Marquardt method.
    ///
    /// Uses smoother damping updates and often behaves well on difficult nonlinear systems.
    NielsenLevenbergMarquardt(NielsenLevenbergMarquardtMethod),
    /// Advanced Nielsen-style Levenberg-Marquardt method.
    ///
    /// Extended Nielsen variant with richer trust-region and scaling controls.
    NielsenLevenbergMarquardtAdvanced(NielsenLevenbergMarquardtMethodAdvanced),
    /// Basic trust-region dogleg method.
    ///
    /// Uses a trust-region radius to keep steps stable when Newton-like steps are risky.
    TrustRegion(TrustRegionMethod),
    /// Powell's dogleg trust-region method.
    ///
    /// A more specialized dogleg implementation with Powell-style step construction.
    PowellDogleg(PowellDoglegMethod),
    /// Trust-region Levenberg-Marquardt method.
    ///
    /// Solves the MINPACK-style trust-region subproblem directly and is well suited to hard systems.
    TrustRegionLM(TrustRegionLMMethod),
}

impl Default for NonlinearSolverMethod {
    fn default() -> Self {
        Self::Newton(NewtonMethod)
    }
}

/// Internal state enum used by [`NonlinearSolverMethod`].
#[derive(Debug, Clone)]
pub enum NonlinearSolverMethodState {
    /// State for the classical Newton method.
    Newton(()),
    /// State for damped Newton.
    DampedNewton(()),
    /// State for advanced damped Newton.
    DampedNewtonAdvanced(()),
    /// State for classical LM.
    LevenbergMarquardt(LevenbergMarquardtState),
    /// State for MINPACK LM.
    LevenbergMarquardtMinpack(LMMinpackState),
    /// State for Nielsen LM.
    NielsenLevenbergMarquardt(NielsenLevenbergMarquardtState),
    /// State for advanced Nielsen LM.
    NielsenLevenbergMarquardtAdvanced(NielsenLevenbergMarquardtStateAdvanced),
    /// State for the basic trust-region method.
    TrustRegion(TrustRegionState),
    /// State for Powell dogleg.
    PowellDogleg(PowellDoglegState),
    /// State for trust-region LM.
    TrustRegionLM(TrustRegionLMState),
}

impl NonlinearSolverMethod {
    /// Returns a short human-readable method name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Newton(_) => "newton",
            Self::DampedNewton(_) => "damped_newton",
            Self::DampedNewtonAdvanced(_) => "damped_newton_advanced",
            Self::LevenbergMarquardt(_) => "levenberg_marquardt",
            Self::LevenbergMarquardtMinpack(_) => "levenberg_marquardt_minpack",
            Self::NielsenLevenbergMarquardt(_) => "nielsen_levenberg_marquardt",
            Self::NielsenLevenbergMarquardtAdvanced(_) => "nielsen_levenberg_marquardt_advanced",
            Self::TrustRegion(_) => "trust_region",
            Self::PowellDogleg(_) => "powell_dogleg",
            Self::TrustRegionLM(_) => "trust_region_lm",
        }
    }

    /// Creates a solver engine for the selected method.
    pub fn engine(self, options: SolveOptions) -> SolverEngine<Self> {
        SolverEngine::new(self, options)
    }

    /// Solves a nonlinear system with the selected method.
    pub fn solve<P>(
        self,
        problem: &P,
        x0: nalgebra::DVector<f64>,
        options: SolveOptions,
    ) -> Result<SolveResult, SolveError>
    where
        P: JacobianProvider,
    {
        self.engine(options).solve(problem, x0)
    }
}

impl NonlinearMethod for NonlinearSolverMethod {
    type MethodState = NonlinearSolverMethodState;

    fn init<P: JacobianProvider>(
        &self,
        problem: &P,
        x0: &nalgebra::DVector<f64>,
        options: &SolveOptions,
        residual: &nalgebra::DVector<f64>,
        jacobian: &nalgebra::DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        match self {
            Self::Newton(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::Newton),
            Self::DampedNewton(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::DampedNewton),
            Self::DampedNewtonAdvanced(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::DampedNewtonAdvanced),
            Self::LevenbergMarquardt(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::LevenbergMarquardt),
            Self::LevenbergMarquardtMinpack(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::LevenbergMarquardtMinpack),
            Self::NielsenLevenbergMarquardt(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::NielsenLevenbergMarquardt),
            Self::NielsenLevenbergMarquardtAdvanced(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::NielsenLevenbergMarquardtAdvanced),
            Self::TrustRegion(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::TrustRegion),
            Self::PowellDogleg(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::PowellDogleg),
            Self::TrustRegionLM(method) => method
                .init(problem, x0, options, residual, jacobian)
                .map(NonlinearSolverMethodState::TrustRegionLM),
        }
    }

    fn step<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError> {
        match (self, method_state) {
            (Self::Newton(method), NonlinearSolverMethodState::Newton(inner)) => {
                method.step(problem, state, inner, options, runtime)
            }
            (Self::DampedNewton(method), NonlinearSolverMethodState::DampedNewton(inner)) => {
                method.step(problem, state, inner, options, runtime)
            }
            (
                Self::DampedNewtonAdvanced(method),
                NonlinearSolverMethodState::DampedNewtonAdvanced(inner),
            ) => method.step(problem, state, inner, options, runtime),
            (
                Self::LevenbergMarquardt(method),
                NonlinearSolverMethodState::LevenbergMarquardt(inner),
            ) => method.step(problem, state, inner, options, runtime),
            (
                Self::LevenbergMarquardtMinpack(method),
                NonlinearSolverMethodState::LevenbergMarquardtMinpack(inner),
            ) => method.step(problem, state, inner, options, runtime),
            (
                Self::NielsenLevenbergMarquardt(method),
                NonlinearSolverMethodState::NielsenLevenbergMarquardt(inner),
            ) => method.step(problem, state, inner, options, runtime),
            (
                Self::NielsenLevenbergMarquardtAdvanced(method),
                NonlinearSolverMethodState::NielsenLevenbergMarquardtAdvanced(inner),
            ) => method.step(problem, state, inner, options, runtime),
            (Self::TrustRegion(method), NonlinearSolverMethodState::TrustRegion(inner)) => {
                method.step(problem, state, inner, options, runtime)
            }
            (Self::PowellDogleg(method), NonlinearSolverMethodState::PowellDogleg(inner)) => {
                method.step(problem, state, inner, options, runtime)
            }
            (Self::TrustRegionLM(method), NonlinearSolverMethodState::TrustRegionLM(inner)) => {
                method.step(problem, state, inner, options, runtime)
            }
            _ => Err(SolveError::InvalidConfig(
                "solver method state does not match the selected enum variant".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod facade_tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    struct PlainProblem;

    impl NonlinearProblem for PlainProblem {
        fn dimension(&self) -> usize {
            2
        }

        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![
                x[0] * x[0] + x[1] * x[1] - 10.0,
                x[0] - x[1] - 4.0,
            ]))
        }
    }

    impl JacobianProvider for PlainProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 2.0 * x[1], 1.0, -1.0],
            ))
        }
    }

    #[test]
    fn enum_facade_reports_name() {
        let method = NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod::default());
        assert_eq!(method.name(), "trust_region_lm");
    }

    #[test]
    fn enum_facade_solves_plain_problem() {
        let method =
            NonlinearSolverMethod::LevenbergMarquardtMinpack(LevenbergMarquardtMinpack::default());
        let result = method
            .solve(
                &PlainProblem,
                DVector::from_vec(vec![1.0, 1.0]),
                SolveOptions {
                    tolerance: 1e-8,
                    max_iterations: 100,
                    ..SolveOptions::default()
                },
            )
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-6);
    }
}

// Additional solvers (if they are included in the main module)
// Note: NR_LM_minpack and NR_trust_region are not exported by default.
// Uncomment if needed:
// pub use crate::numerical::Nonlinear_systems::NR_LM_minpack::*;
// pub use crate::numerical::Nonlinear_systems::NR_trust_region::*;
// pub use crate::numerical::Nonlinear_systems::trust_region_lmpar::*;
