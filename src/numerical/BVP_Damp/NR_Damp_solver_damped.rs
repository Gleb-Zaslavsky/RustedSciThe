//! # Damped Newton-Raphson Solver for Boundary Value Problems
//!
//! ## Module Purpose
//! This module implements a robust damped Newton-Raphson method for solving systems of nonlinear
//! boundary value problems (BVPs). It's the core solver for stiff ODEs with adaptive grid refinement
//! capabilities, making it essential for the entire RustedSciThe project.
//!
//! ## Key Features
//! - **Damped Newton Method**: Prevents divergence by controlling step size through damping coefficients
//! - **Adaptive Grid Refinement**: Automatically refines mesh where solution changes rapidly
//! - **Jacobian Reuse Strategy**: Optimizes performance by intelligently reusing Jacobian matrices
//! - **Boundary Constraint Handling**: Ensures solution stays within physical bounds
//! - **Multiple Matrix Backends**: Supports both dense and sparse matrix operations
//!
//! ## Core Structures
//! - [`NRBVP`]: Main solver struct containing all problem parameters and solution state
//! - [`SolverParams`]: Configuration for damping parameters and adaptive grid settings
//! - [`AdaptiveGridConfig`]: Specific settings for grid refinement algorithms
//!
//! ## Algorithm Overview
//! The solver uses a modified Newton method with damping to ensure convergence:
//! 1. Compute undamped Newton step: `J(x_k) * О”x = -F(x_k)`
//! 2. Apply boundary constraints to determine maximum step size
//! 3. Use damping coefficient О» to control step: `x_{k+1} = x_k + О» * О”x`
//! 4. Accept step if residual decreases, otherwise reduce О» and retry
//!
//! ## Interesting Code Features
//! - **Trait-based abstraction**: Uses `VectorType` and `MatrixType` traits for backend flexibility
//! - **Macro-based timing**: Custom macros for performance profiling of different operations
//! - **Adaptive strategy pattern**: Configurable Jacobian recalculation strategies
//! - **Memory-aware design**: Checks system memory before allocating large matrices
//! - **Comprehensive logging**: Detailed logging with configurable levels for debugging
//!
//! ## Performance Tips
//! - Use sparse matrices for large problems (>1000 unknowns)
//! - Enable adaptive grid refinement for problems with boundary layers
//! - Tune damping parameters based on problem stiffness
//! - Monitor Jacobian age to balance accuracy vs performance
//!
//! ## References
//! - Cantera MultiNewton solver (MultiNewton.cpp)
//! - TWOPNT Fortran solver ("The Twopnt Program for Boundary Value Problems" by J. F. Grcar)
//! - Chemkin Theory Manual p.261
use crate::symbolic::symbolic_engine::Expr;
use chrono::Local;

use crate::Utils::logger::{save_matrix_to_csv, save_matrix_to_file};
use crate::Utils::plots::{plots, plots_gnulot, plots_terminal};
use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, VectorType, Vectors_type_casting, finite_difference_jacobian,
    try_finite_difference_jacobian,
};
use crate::numerical::BVP_Damp::BVP_utils::{
    CustomTimer, construct_full_solution, elapsed_time, extract_unknown_variables, task_check_mem,
};
use crate::numerical::BVP_Damp::BVP_utils_damped::{
    bound_step_Cantera2, convergence_condition, if_initial_guess_inside_bounds, jac_recalc,
};
use crate::numerical::BVP_Damp::factor_runtime::{
    OwnedLinearFactorRuntime, prepare_factor_owner_runtime,
};
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, ApplyDampedGeneratedSolverState,
    BandedGeneratedBackendMode, BuildDampedSolverRequest, DampedGeneratedSolverState,
    DampedSolverBuildRequest, GeneratedBackendConfig, SparseGeneratedBackendMode,
    try_generate_and_apply_damped_solver_state,
};
use crate::numerical::BVP_Damp::numeric_discretization::{
    NumericBvpJacobian, NumericBvpRhs, build_numeric_generated_solver_state,
};
use crate::numerical::BVP_Damp::prepared_runtime::{
    BvpPreparedPlan, BvpPreparedResourceSnapshot, BvpPreparedRuntime, BvpPreparedRuntimeSnapshot,
    BvpRuntimeRevision, PreparedPlanFingerprint, fingerprint_bytes, fingerprint_callback_ptr,
    fingerprint_debug,
};
use crate::numerical::BVP_Damp::telemetry::{
    BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode, BvpTelemetryRecorder, BvpTelemetrySnapshot,
};
use crate::somelinalg::banded::LinearSolverConfig;
use crate::symbolic::bvp::aot_telemetry::BvpAotTelemetry;
use crate::symbolic::bvp::telemetry::{
    BvpGenerationTelemetrySnapshot, BvpLambdifyTelemetry, BvpLambdifyTelemetryMode,
};
use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_functions_BVP::{
    BvpBackendIntegrationError, BvpMatrixBackend, BvpSymbolicAssemblyBackend,
};
use core::panic;
use nalgebra::{DMatrix, DVector};
use simplelog::LevelFilter;
use simplelog::*;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};

use crate::numerical::BVP_Damp::grid_api::{GridRefinementMethod, new_grid};
use crate::numerical::BVP_Damp::solver_common::{
    DEFAULT_FORWARD_SCHEME, DEFAULT_MAX_ITERATIONS, cleanup_registered_aot_artifacts,
    damped_interval_mesh, default_dense_method_name, default_forward_scheme_name,
    default_placeholder_y, default_sparse_method_name,
};

include!("damped_solver/statistics.rs");
include!("damped_solver/options.rs");
include!("damped_solver/state.rs");
include!("damped_solver/preparation.rs");
include!("damped_solver/solve.rs");
include!("damped_solver/postprocess.rs");

#[cfg(test)]
#[path = "tests/damped_solver_unit.rs"]
mod tests;
