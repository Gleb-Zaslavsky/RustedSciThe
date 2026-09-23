use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::Instant;

use crate::Utils::logger::save_matrix_to_file;
use crate::Utils::plots::plots;
use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, LinearSolveTiming, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::*;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::BvpDerivativeScheme;
use crate::numerical::BVP_Damp::factor_runtime::prepare_factor_owner_runtime;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, ApplyFrozenGeneratedSolverState,
    BandedGeneratedBackendMode, BuildFrozenSolverRequest, FrozenGeneratedSolverState,
    FrozenSolverBuildRequest, GeneratedBackendConfig, SparseGeneratedBackendMode,
    try_generate_and_apply_frozen_solver_state,
};
use crate::numerical::BVP_Damp::prepared_runtime::{
    BvpPreparedResourceSnapshot, BvpPreparedRuntime, BvpPreparedRuntimeSnapshot,
    BvpRuntimeRevision, PreparedPlanFingerprint, fingerprint_bytes, fingerprint_callback_ptr,
    fingerprint_debug,
};
use crate::numerical::BVP_Damp::solver_common::{
    DEFAULT_MAX_ITERATIONS, cleanup_registered_aot_artifacts, default_dense_method_name,
    default_forward_scheme_name, default_placeholder_y, default_sparse_method_name,
    frozen_point_mesh,
};
use crate::numerical::BVP_Damp::telemetry::{
    BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode, BvpTelemetryRecorder, BvpTelemetrySnapshot,
};
use crate::somelinalg::banded::LinearSolverConfig;
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

use chrono::Local;

use log::info;

use simplelog::*;

use std::fs::File;

include!("frozen_solver/statistics.rs");
include!("frozen_solver/options.rs");
include!("frozen_solver/state.rs");
include!("frozen_solver/preparation.rs");
include!("frozen_solver/solve.rs");
include!("frozen_solver/postprocess.rs");

#[cfg(test)]
#[path = "tests/frozen_solver_unit.rs"]
mod tests;
