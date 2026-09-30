//! AOT chunking and warm/cold execution stories.

#[test]
fn lsode2_parallel_chunking_story_by_weight_class() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_chunking_story_tests::lsode2_parallel_chunking_story_by_weight_class",
    );
    super::legacy_story_support::run_lsode2_parallel_chunking_story_by_weight_class();
}

#[test]
fn lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_chunking_story_tests::lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard",
    );
    super::legacy_story_support::run_lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard();
}

#[test]
fn lsode2_parallel_chunking_cold_stage_story_by_weight_class() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_chunking_story_tests::lsode2_parallel_chunking_cold_stage_story_by_weight_class",
    );
    super::legacy_story_support::run_lsode2_parallel_chunking_cold_stage_story_by_weight_class();
}

#[test]
fn lsode2_combustion_like_parallel_chunking_cold_stage_story_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_chunking_story_tests::lsode2_combustion_like_parallel_chunking_cold_stage_story_dashboard",
    );
    super::legacy_story_support::run_lsode2_combustion_like_parallel_chunking_cold_stage_story_dashboard();
}

#[test]
#[ignore = "release story: larger LSODE2 generated IVP checks whether tcc callback chunking can amortize"]
fn lsode2_large_chain_tcc_chunking_sparse_banded_warm_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_chunking_story_tests::lsode2_large_chain_tcc_chunking_sparse_banded_warm_story",
    );
    super::legacy_story_support::run_lsode2_large_chain_tcc_chunking_sparse_banded_warm_story();
}
