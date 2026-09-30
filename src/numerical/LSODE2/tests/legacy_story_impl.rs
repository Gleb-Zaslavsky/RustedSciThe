use super::aot_three_body_story_tests;
use super::legacy_atomview_lambdify;
use super::native_jacobian::{
    NativeJacobianStorage, compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry,
};
use super::story_support::{BackendRaceRow, RaceStats, short_error, unique_story_short_tag};
use super::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2BackendConfig, Lsode2JacobianBackend,
    Lsode2LinearSolverBackend, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use super::{Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, algorithm};
use crate::numerical::BDF::BDF_solver::BdfJacobian;
use crate::symbolic::View::conversions::{atom_to_expr, expr_to_atom};
use crate::symbolic::View::{ExpressionMetrics, inspect_exprs};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedResidualAotBackend, register_linked_residual_backend, unregister_linked_residual_backend,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::ivp_telemetry::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetrySnapshot, IvpWarmStage,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SharedIvpParameterValues, SymbolicIvpProblemOptions,
    build_symbolic_jacobian, prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Instant;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Keep existing stdout tables while mirroring lines into dated reports.
macro_rules! println {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

mod legacy_story_core {
    use super::*;
    include!("legacy_story_core.rs");
}

mod legacy_story_race {
    use super::legacy_story_core::{exponential_decay_config, is_native_faithful_status};
    use super::*;
    include!("legacy_story_race.rs");
}

mod legacy_story_solver_quality {
    use super::legacy_story_core::ComprehensiveScenario;
    use super::legacy_story_core::{exponential_decay_config, max_abs_diff_vec};
    use super::legacy_story_lifecycle::LargeIvpChunkingRow;
    use super::legacy_story_lifecycle::{push_large_ivp_sample, run_large_ivp_chunking_sample};
    use super::legacy_story_race::BackendRaceMatrix;
    use super::*;
    include!("legacy_story_solver_quality.rs");
}

mod legacy_story_combustion {
    use super::legacy_story_core::is_native_faithful_status;
    use super::legacy_story_race::{
        BackendRaceMatrix, race_aot_config_with_output, race_aot_parallel_config_with_output,
        race_lambdify_config, run_backend_race_sample, unique_story_run_tag,
    };
    use super::*;
    include!("legacy_story_combustion.rs");
}

mod legacy_story_view {
    use super::legacy_story_combustion::combustion_like_story_base_config;
    use super::legacy_story_lifecycle::large_diffusion_chain_config;
    use super::legacy_story_race::BackendRaceMatrix;
    use super::*;
    include!("legacy_story_view.rs");
}

mod legacy_story_lifecycle {
    use super::legacy_story_combustion::CombustionStorySample;
    use super::legacy_story_combustion::{
        combustion_symbolic_matrix_config, print_compact_combustion_story_tables,
        push_combustion_sample, run_combustion_story_sample_result,
    };
    use super::legacy_story_core::AotStoryToolchain;
    use super::legacy_story_race::BackendRaceMatrix;
    use super::*;
    include!("legacy_story_lifecycle.rs");
}

pub(crate) use legacy_story_combustion::run_lsode2_parallel_chunking_cold_stage_story_by_weight_class;
pub(crate) use legacy_story_combustion::{
    run_lsode2_combustion_like_multi_run_story_dashboard,
    run_lsode2_combustion_like_parallel_chunking_cold_stage_story_dashboard,
    run_lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard,
};
pub(crate) use legacy_story_core::{
    run_lsode2_aot_toolchain_stage_story_table, run_lsode2_exponential_decay_backend_story_table,
};
pub(crate) use legacy_story_lifecycle::{
    run_lsode2_cold_aot_story_config_forces_rebuild_always,
    run_lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix,
    run_lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story,
    run_lsode2_combustion_sparse_banded_all_frontends_tcc_build_then_require_prebuilt_story,
    run_lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story,
    run_lsode2_large_chain_tcc_chunking_sparse_banded_warm_story,
};
pub(crate) use legacy_story_race::run_lsode2_parallel_chunking_story_by_weight_class;
