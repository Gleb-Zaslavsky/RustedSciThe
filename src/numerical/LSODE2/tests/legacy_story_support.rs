// Compatibility facade for historical story runners. The implementation now
// lives in independently named thematic child modules, so this path no longer
// owns the legacy dashboards.
pub(super) use super::legacy_story_impl::{
    run_lsode2_aot_toolchain_stage_story_table,
    run_lsode2_cold_aot_story_config_forces_rebuild_always,
    run_lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix,
    run_lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story,
    run_lsode2_combustion_like_multi_run_story_dashboard,
    run_lsode2_combustion_like_parallel_chunking_cold_stage_story_dashboard,
    run_lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard,
    run_lsode2_combustion_sparse_banded_all_frontends_tcc_build_then_require_prebuilt_story,
    run_lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story,
    run_lsode2_exponential_decay_backend_story_table,
    run_lsode2_large_chain_tcc_chunking_sparse_banded_warm_story,
    run_lsode2_parallel_chunking_cold_stage_story_by_weight_class,
    run_lsode2_parallel_chunking_story_by_weight_class,
};
