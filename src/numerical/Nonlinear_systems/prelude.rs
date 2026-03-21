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

// Symbolic problem adapter
pub use crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem;

// Additional solvers (if they are included in the main module)
// Note: NR_LM_minpack and NR_trust_region are not exported by default.
// Uncomment if needed:
// pub use crate::numerical::Nonlinear_systems::NR_LM_minpack::*;
// pub use crate::numerical::Nonlinear_systems::NR_trust_region::*;
// pub use crate::numerical::Nonlinear_systems::trust_region_LM::*;
// pub use crate::numerical::Nonlinear_systems::trust_region_lmpar::*;
