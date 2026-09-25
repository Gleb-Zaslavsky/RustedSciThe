//! AOT compiler/toolchain stage stories.

#[test]
fn lsode2_aot_toolchain_stage_story_table() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_toolchain_story_tests::lsode2_aot_toolchain_stage_story_table",
    );
    super::story_tests2::run_lsode2_aot_toolchain_stage_story_table();
}

#[test]
fn lsode2_cold_aot_story_config_forces_rebuild_always() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_toolchain_story_tests::lsode2_cold_aot_story_config_forces_rebuild_always",
    );
    super::story_tests2::run_lsode2_cold_aot_story_config_forces_rebuild_always();
}

#[test]
#[ignore = "release story: cold AOT toolchain/chunking matrix compiles many generated artifacts"]
fn lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_toolchain_story_tests::lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix",
    );
    super::story_tests2::run_lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix();
}
