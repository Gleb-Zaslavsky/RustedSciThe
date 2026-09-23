use crate::numerical::BVP_Damp::BVP_traits::{Fun, Jac};
use crate::numerical::BVP_Damp::telemetry::{BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode};
use crate::somelinalg::banded::LinearSolverConfig;
use crate::symbolic::bvp::aot_telemetry::{BvpAotColdStage, BvpAotLifecycleEvent};
use crate::symbolic::bvp::aot_telemetry::{BvpAotTelemetry, BvpAotTelemetryMode};
use crate::symbolic::bvp::telemetry::{
    BvpDirectJacobianTelemetry, BvpGenerationTelemetrySnapshot, BvpLambdifyExecutionPolicy,
    BvpLambdifyTelemetry, BvpLambdifyTelemetryMode,
};
use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
use crate::symbolic::codegen::c_backend::codegen_c_aot_build::{
    CAotBuildProfile, CAotBuildRequest, CAotCompileConfig,
};
use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::{
    register_generated_c_banded_backend, register_generated_c_sparse_backend,
};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_aot_registry::{AotRegistry, RegisteredAotArtifact};
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    register_generated_banded_cdylib_backend, register_generated_sparse_cdylib_backend,
    try_resolve_linked_sparse_backend, try_unregister_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::codegen::codegen_orchestrator::ParallelExecutorConfig;
use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile as LifecycleBuildProfile, AotBuildRequest,
    AotCompileConfig as LifecycleAotCompileConfig,
};
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_build::{
    ZigAotBuildProfile, ZigAotBuildRequest,
};
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_registry::register_zig_build_in_registry;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::{
    register_generated_zig_banded_backend, register_generated_zig_sparse_backend,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{
    BvpAotPreparationRoute, BvpBackendIntegrationError, BvpGeneratedAotCrateBreakdown,
    BvpLegacySolverBundle, BvpSparseExecutionPlan, BvpSparseSolverBundle,
    BvpSymbolicAssemblyBackend, Jacobian,
};
use log::{error, info};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

include!("handoff/lifecycle.rs");
include!("handoff/config.rs");
include!("handoff/requests.rs");
include!("handoff/generation.rs");
include!("handoff/backend.rs");
include!("handoff/build.rs");
include!("handoff/state.rs");

#[cfg(test)]
#[path = "tests/generated_solver_handoff_unit.rs"]
mod tests;
