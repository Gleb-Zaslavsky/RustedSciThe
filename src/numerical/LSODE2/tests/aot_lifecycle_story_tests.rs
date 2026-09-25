//! AOT lifecycle stories separated from the mixed LSODE2 story source.

#[test]
fn lsode2_exponential_decay_backend_story_table() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_lifecycle_story_tests::lsode2_exponential_decay_backend_story_table",
    );
    super::story_tests2::run_lsode2_exponential_decay_backend_story_table();
}

#[test]
fn lsode2_combustion_like_multi_run_story_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_lifecycle_story_tests::lsode2_combustion_like_multi_run_story_dashboard",
    );
    super::story_tests2::run_lsode2_combustion_like_multi_run_story_dashboard();
}

#[test]
#[ignore = "release story: LSODE2 combustion AtomView tcc BuildIfMissing followed by strict RequirePrebuilt reuse"]
fn lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_lifecycle_story_tests::lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story",
    );
    super::story_tests2::run_lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story();
}

#[test]
#[ignore = "release story: warm repeated-solve comparison with cooldown, LSODE2 Banded AtomViewNative Lambdify vs strict tcc RequirePrebuilt"]
fn lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_lifecycle_story_tests::lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story",
    );
    super::story_tests2::run_lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story();
}
