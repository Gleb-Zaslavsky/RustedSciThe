/// Debug-only gate for localizing sparse AOT chunking failures.
///
/// This is intentionally not a story/performance test. It does not solve
/// the BVP and does not report timing. Its only contract is mathematical:
/// a chunked generated sparse callback must produce exactly the same
/// residual and Jacobian values as the whole generated callback on the same
/// state vector before Newton starts.
#[test]
#[ignore = "debug-only AOT callback equivalence gate; builds combustion-1000 artifacts"]
fn debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000() {
    aot_test_report!(
        debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000
    );
    let n_steps = 1000usize;
    let variants = apply_optional_callback_equivalence_filter(&callback_equivalence_variants());
    let mut rows = Vec::with_capacity(variants.len() * 3);
    let mut stats_rows = Vec::with_capacity(variants.len() * 3);
    let mut lambdify_baselines = HashMap::<&'static str, CallbackProbe>::new();

    for variant in variants {
        if !lambdify_baselines.contains_key(variant.matrix) {
            let baseline_label = format!("{} Lambdify baseline", variant.matrix);
            let baseline = callback_probe_for_initial_guess(
                n_steps,
                baseline_label.as_str(),
                callback_vector_method(variant.matrix),
                callback_lambdify_baseline_config(variant.matrix),
            );
            stats_rows.push(callback_probe_stats_row(
                variant.matrix,
                "baseline",
                "lambdify",
                &baseline,
                "ok".to_string(),
            ));
            lambdify_baselines.insert(variant.matrix, baseline);
        }

        let whole_label = format!("{} {} whole", variant.matrix, variant.toolchain);
        let chunk4_label = format!("{} {} chunk4", variant.matrix, variant.toolchain);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let whole_probe = callback_probe_for_initial_guess(
                n_steps,
                whole_label.as_str(),
                callback_vector_method(variant.matrix),
                variant.whole_config.clone(),
            );
            let chunk_probe = callback_probe_for_initial_guess(
                n_steps,
                chunk4_label.as_str(),
                callback_vector_method(variant.matrix),
                variant.chunk4_config.clone(),
            );
            (whole_probe, chunk_probe)
        }));

        match result {
            Ok((whole_probe, chunk_probe)) => {
                let baseline_probe = lambdify_baselines
                    .get(variant.matrix)
                    .expect("Lambdify baseline must be prepared before AOT probes");
                let baseline_vs_whole = callback_diff_row(
                    variant.matrix,
                    variant.toolchain,
                    "lambdify-vs-whole",
                    baseline_probe,
                    &whole_probe,
                );
                let whole_vs_chunk4 = callback_diff_row(
                    variant.matrix,
                    variant.toolchain,
                    "whole-vs-chunk4",
                    &whole_probe,
                    &chunk_probe,
                );
                let baseline_vs_chunk4 = callback_diff_row(
                    variant.matrix,
                    variant.toolchain,
                    "lambdify-vs-chunk4",
                    baseline_probe,
                    &chunk_probe,
                );
                let status = if [
                    baseline_vs_whole.status.as_str(),
                    whole_vs_chunk4.status.as_str(),
                    baseline_vs_chunk4.status.as_str(),
                ]
                .iter()
                .all(|status| *status == "ok")
                {
                    "ok".to_string()
                } else {
                    "diff_exceeded".to_string()
                };
                stats_rows.push(callback_probe_stats_row(
                    variant.matrix,
                    variant.toolchain,
                    "whole",
                    &whole_probe,
                    status.clone(),
                ));
                stats_rows.push(callback_probe_stats_row(
                    variant.matrix,
                    variant.toolchain,
                    "chunk4",
                    &chunk_probe,
                    status.clone(),
                ));
                rows.push(baseline_vs_whole);
                rows.push(whole_vs_chunk4);
                rows.push(baseline_vs_chunk4);
            }
            Err(panic_payload) => {
                let status = if let Some(message) = panic_payload.downcast_ref::<String>() {
                    format!("panicked({message})")
                } else if let Some(message) = panic_payload.downcast_ref::<&str>() {
                    format!("panicked({message})")
                } else {
                    "panicked(non-string payload)".to_string()
                };
                rows.push(CallbackEquivalenceRow {
                    matrix: variant.matrix,
                    toolchain: variant.toolchain,
                    comparison: "callback-probe",
                    residual_diff: f64::NAN,
                    jacobian_diff: f64::NAN,
                    status: status.clone(),
                });
                for mode in ["whole", "chunk4"] {
                    stats_rows.push(CallbackProbeStatsRow {
                        matrix: variant.matrix,
                        toolchain: variant.toolchain,
                        mode,
                        total_probe_ms: f64::NAN,
                        bootstrap_ms: f64::NAN,
                        residual_ms: f64::NAN,
                        jacobian_ms: f64::NAN,
                        residual_calls: 0,
                        jacobian_calls: 0,
                        residual_len: 0,
                        jac_rows: 0,
                        jac_cols: 0,
                        status: status.clone(),
                    });
                }
            }
        }
    }

    print_callback_equivalence_table(&rows);
    print_callback_probe_stats_table(&stats_rows);

    assert!(
        rows.iter().all(|row| row.status == "ok"),
        "all AtomView AOT chunk4 callbacks must match whole callbacks before Newton"
    );
}

#[test]
#[ignore = "heavy combustion-1000 end-to-end Lambdify Sparse vs Banded race table"]
fn combustion_1000_lambdify_sparse_vs_banded_end_to_end_race() {
    aot_test_report!(combustion_1000_lambdify_sparse_vs_banded_end_to_end_race);
    let n_steps = 1000usize;
    let sparse_config = GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    let banded_config = GeneratedBackendConfig::banded_lambdify_defaults()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

    let variants = [
        RaceVariant {
            source: "Lambdify",
            matrix: "Sparse",
            variant: "ExprLegacy",
            bootstrap_hint: "symbolic+lambdify",
            config: sparse_config,
        },
        RaceVariant {
            source: "Lambdify",
            matrix: "Banded",
            variant: "ExprLegacy",
            bootstrap_hint: "symbolic+lambdify",
            config: banded_config,
        },
    ];

    let protocol = story_protocol(n_steps, RACE_REPETITIONS);
    println!("[BVP Damp race] protocol: {}", protocol.summary());
    let samples = run_race_samples(&variants, n_steps, protocol.cold_repetitions);
    let rows = summarize_samples(&variants, &samples);
    print_race_summary_table(
        "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded end-to-end",
        &rows,
    );
    println!();
    print_e2e_callback_stage_table(
        "[BVP Damp race] combustion-1000 Lambdify callback stage breakdown",
        &rows,
    );
    println!();
    print_e2e_lifecycle_table(
        "[BVP Damp race] combustion-1000 Lambdify lifecycle/refinement breakdown",
        &rows,
    );
    println!();
    print_e2e_bootstrap_pass_table(
        "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded symbolic handoff stages",
        &rows,
    );
    println!();
    print_e2e_symbolic_jacobian_detail_table(
        "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded internal symbolic-Jacobian stages",
        &rows,
    );
    println!();
    print_e2e_lambdify_binding_detail_table(
        "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded callback compilation stages",
        &rows,
    );

    assert!(
        rows.iter().all(|row| row.ok_runs == row.runs),
        "both Lambdify race variants should solve successfully"
    );
}

#[test]
#[ignore = "heavy combustion-1000 end-to-end AOT Sparse vs Banded race table"]
fn combustion_1000_aot_sparse_vs_banded_end_to_end_race() {
    aot_test_report!(combustion_1000_aot_sparse_vs_banded_end_to_end_race);
    let n_steps = 1000usize;
    let variants = [
        RaceVariant {
            source: "Compiled",
            matrix: "Sparse",
            variant: "C-gcc",
            bootstrap_hint: "symbolic+aot-build+link",
            config: rebuild_release(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
            ),
        },
        RaceVariant {
            source: "Compiled",
            matrix: "Banded",
            variant: "C-gcc",
            bootstrap_hint: "symbolic+aot-build+link",
            config: rebuild_release(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
            ),
        },
        RaceVariant {
            source: "Compiled",
            matrix: "Sparse",
            variant: "C-tcc",
            bootstrap_hint: "symbolic+aot-build+link",
            config: rebuild_release(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
            ),
        },
        RaceVariant {
            source: "Compiled",
            matrix: "Banded",
            variant: "C-tcc",
            bootstrap_hint: "symbolic+aot-build+link",
            config: rebuild_release(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
            ),
        },
        RaceVariant {
            source: "Compiled",
            matrix: "Sparse",
            variant: "Zig",
            bootstrap_hint: "symbolic+aot-build+link",
            config: rebuild_release(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
            ),
        },
        RaceVariant {
            source: "Compiled",
            matrix: "Banded",
            variant: "Zig",
            bootstrap_hint: "symbolic+aot-build+link",
            config: rebuild_release(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
            ),
        },
    ];

    let protocol = story_protocol(n_steps, RACE_REPETITIONS);
    println!("[BVP Damp race] protocol: {}", protocol.summary());
    let samples = run_race_samples(&variants, n_steps, protocol.cold_repetitions);
    let rows = summarize_samples(&variants, &samples);
    print_race_summary_table(
        "[BVP Damp race] combustion-1000 AOT Sparse vs Banded end-to-end",
        &rows,
    );
    println!();
    print_e2e_callback_stage_table(
        "[BVP Damp race] combustion-1000 AOT callback stage breakdown",
        &rows,
    );
    println!();
    print_e2e_lifecycle_table(
        "[BVP Damp race] combustion-1000 AOT lifecycle/refinement breakdown",
        &rows,
    );
    println!();
    print_e2e_bootstrap_pass_table(
        "[BVP Damp race] combustion-1000 AOT Sparse vs Banded symbolic handoff stages",
        &rows,
    );
    println!();
    print_e2e_symbolic_jacobian_detail_table(
        "[BVP Damp race] combustion-1000 AOT Sparse vs Banded internal symbolic-Jacobian stages",
        &rows,
    );

    assert!(
        rows.iter().any(|row| row.ok_runs > 0),
        "at least one AOT race variant should solve successfully"
    );
}

#[test]
#[ignore = "heavy combustion-1000 AOT toolchain/chunking release matrix for Sparse and Banded"]
fn combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix() {
    aot_test_report!(combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix);
    let n_steps = 1000usize;
    let repetitions = 2usize;
    let variants = [
        RaceVariant {
            source: "Lambdify",
            matrix: "Sparse",
            variant: "AtomView",
            bootstrap_hint: "baseline+symbolic+lambdify",
            config: sparse_atomview_lambdify_baseline(),
        },
        RaceVariant {
            source: "Lambdify",
            matrix: "Banded",
            variant: "AtomView",
            bootstrap_hint: "baseline+symbolic+lambdify",
            config: banded_atomview_lambdify_baseline(),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "gcc/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "gcc/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "gcc/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "gcc/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "tcc/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "tcc/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "tcc/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "tcc/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "zig/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "zig/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "zig/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "zig/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "rust/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                sparse_atomview_rust_aot_release(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "rust/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                sparse_atomview_rust_aot_release(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "rust/whole",
            bootstrap_hint: "rebuild+seq+whole",
            config: release_matrix_config(
                banded_atomview_rust_aot_release(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "rust/chunk4",
            bootstrap_hint: "rebuild+par+chunk4",
            config: release_matrix_config(
                banded_atomview_rust_aot_release(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
    ];

    let variants = apply_optional_release_matrix_filter(&variants);
    let samples = run_race_samples(&variants, n_steps, repetitions);
    let rows = summarize_samples(&variants, &samples);
    print_race_summary_table(
        "[BVP Damp race] combustion-1000 AOT toolchain/chunking release matrix",
        &rows,
    );

    assert!(
        rows.iter()
            .filter(|row| row.source == "Lambdify")
            .all(|row| row.ok_runs == row.runs),
        "Lambdify baseline variants must solve successfully before AOT chunking rows are interpreted"
    );
}

#[test]
#[ignore = "end-to-end combustion-200 AOT/Lambdify matrix across Sparse/Banded, gcc/tcc/zig, whole/chunk4"]
fn combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix() {
    aot_test_report!(combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix);
    let n_steps = 200usize;
    let repetitions = 3usize;
    let variants = combustion_toolchain_chunking_variants();
    let variants = apply_optional_release_matrix_filter(&variants);

    let samples = run_race_samples(&variants, n_steps, repetitions);
    let rows = summarize_samples(&variants, &samples);
    print_e2e_correctness_table(
        "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT correctness matrix",
        &rows,
    );
    println!();
    print_e2e_performance_table(
        "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT timing/counter matrix",
        &rows,
    );
    println!();
    print_e2e_callback_stage_table(
        "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT callback stage matrix",
        &rows,
    );
    println!();
    print_e2e_lifecycle_table(
        "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT lifecycle matrix",
        &rows,
    );

    assert!(
        rows.iter()
            .filter(|row| row.source == "Lambdify")
            .all(|row| row.ok_runs == row.runs),
        "Lambdify baseline variants must solve successfully before AOT rows are interpreted"
    );
    assert!(
        rows.iter().all(|row| row.ok_runs == row.runs),
        "all combustion-200 Sparse/Banded AOT toolchain/chunking variants should solve successfully"
    );
    assert!(
        rows.iter()
            .filter(|row| row.source == "AOT")
            .all(|row| row.solve_diff.mean.is_finite() && row.solve_diff.mean <= 1e-6),
        "AOT rows must remain numerically equivalent to the Lambdify baseline"
    );
}

#[test]
#[ignore = "heavy combustion-1000 AtomView+tcc Auto chunking production diagnostics"]
fn combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story() {
    aot_test_report!(combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story);
    #[derive(Clone)]
    struct AutoVariant {
        source: &'static str,
        matrix: &'static str,
        variant: &'static str,
        config: GeneratedBackendConfig,
    }

    #[derive(Clone)]
    struct AutoSample {
        source: &'static str,
        matrix: &'static str,
        variant: &'static str,
        total_ms: f64,
        solve_diff: f64,
        rel_x_diff: f64,
        selected_backend: String,
        runtime_execution_policy: String,
        runtime_parallel_requested: String,
        auto_execution_mode: String,
        auto_workers: f64,
        auto_min_work_per_job: f64,
        auto_residual_reason: String,
        auto_sparse_reason: String,
        auto_residual_work_per_job: f64,
        auto_sparse_work_per_job: f64,
        auto_residual_work_per_chunk: f64,
        auto_sparse_work_per_chunk: f64,
        residual_actual_jobs: f64,
        sparse_actual_jobs: f64,
        residual_runtime_work_per_job: f64,
        sparse_runtime_work_per_job: f64,
        residual_fallback_reason: String,
        sparse_fallback_reason: String,
        residual_values_ms: f64,
        jacobian_values_ms: f64,
        jacobian_assembly_ms: f64,
        linear_ms: f64,
        symbolic_ms: f64,
        status: String,
    }

    fn auto_release_tcc_config(config: GeneratedBackendConfig) -> GeneratedBackendConfig {
        release_matrix_config(
            config,
            AotChunkingPolicy::default(),
            AotExecutionPolicy::Auto,
        )
    }

    fn run_auto_sample(
        n_steps: usize,
        variant: &AutoVariant,
    ) -> (AutoSample, Option<DMatrix<f64>>) {
        let total_begin = Instant::now();
        let mut solver = make_combustion_solver(n_steps, variant.config.clone());
        let solve_status = catch_unwind(AssertUnwindSafe(|| solver.try_solver()));
        let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
        let statistics = solver.get_statistics();

        let mut sample = AutoSample {
            source: variant.source,
            matrix: variant.matrix,
            variant: variant.variant,
            total_ms,
            solve_diff: f64::NAN,
            rel_x_diff: f64::NAN,
            selected_backend: stats_diagnostic_string(&statistics, "generated.selected_backend"),
            runtime_execution_policy: stats_diagnostic_string(
                &statistics,
                "aot.runtime.execution_policy",
            ),
            runtime_parallel_requested: stats_diagnostic_string(
                &statistics,
                "aot.runtime.parallel_requested",
            ),
            auto_execution_mode: stats_diagnostic_string(&statistics, "aot.auto.execution_mode"),
            auto_workers: stats_diagnostic_usize(&statistics, "aot.auto.workers"),
            auto_min_work_per_job: stats_diagnostic_usize(&statistics, "aot.auto.min_work_per_job"),
            auto_residual_reason: stats_diagnostic_string(&statistics, "aot.auto.residual.reason"),
            auto_sparse_reason: stats_diagnostic_string(
                &statistics,
                "aot.auto.sparse_jacobian.reason",
            ),
            auto_residual_work_per_job: stats_diagnostic_usize(
                &statistics,
                "aot.auto.residual.work_per_job",
            ),
            auto_sparse_work_per_job: stats_diagnostic_usize(
                &statistics,
                "aot.auto.sparse_jacobian.work_per_job",
            ),
            auto_residual_work_per_chunk: stats_diagnostic_usize(
                &statistics,
                "aot.auto.residual.work_per_chunk",
            ),
            auto_sparse_work_per_chunk: stats_diagnostic_usize(
                &statistics,
                "aot.auto.sparse_jacobian.work_per_chunk",
            ),
            residual_actual_jobs: stats_diagnostic_usize(
                &statistics,
                "aot.runtime.residual.actual_jobs",
            ),
            sparse_actual_jobs: stats_diagnostic_usize(
                &statistics,
                "aot.runtime.sparse_jacobian.actual_jobs",
            ),
            residual_runtime_work_per_job: stats_diagnostic_usize(
                &statistics,
                "aot.runtime.residual.work_per_job",
            ),
            sparse_runtime_work_per_job: stats_diagnostic_usize(
                &statistics,
                "aot.runtime.sparse_jacobian.work_per_job",
            ),
            residual_fallback_reason: stats_diagnostic_string(
                &statistics,
                "aot.runtime.residual.fallback_reason",
            ),
            sparse_fallback_reason: stats_diagnostic_string(
                &statistics,
                "aot.runtime.sparse_jacobian.fallback_reason",
            ),
            residual_values_ms: callback_residual_values_ms(&statistics),
            jacobian_values_ms: callback_jacobian_values_ms(&statistics),
            jacobian_assembly_ms: callback_jacobian_assembly_ms(&statistics),
            linear_ms: stats_timer_ms(&statistics, "Linear System"),
            symbolic_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
            status: "not_run".to_string(),
        };

        match solve_status {
            Ok(Ok(_)) => match solver.get_result() {
                Some(solution) => {
                    sample.status = "ok".to_string();
                    (sample, Some(solution))
                }
                None => {
                    sample.status = "no_result".to_string();
                    (sample, None)
                }
            },
            Ok(Err(err)) => {
                sample.status = format!("solve_error({err:?})");
                (sample, None)
            }
            Err(_) => {
                sample.status = "solve_panicked".to_string();
                (sample, None)
            }
        }
    }

    fn fill_auto_diffs(samples: &mut [AutoSample], solutions: &[Option<DMatrix<f64>>]) {
        let baseline = samples
            .iter()
            .zip(solutions.iter())
            .find_map(|(sample, solution)| {
                (sample.source == "Lambdify" && sample.status == "ok")
                    .then_some(solution.as_ref())
                    .flatten()
            });
        let Some(baseline) = baseline else {
            return;
        };

        for (sample, solution) in samples.iter_mut().zip(solutions.iter()) {
            let Some(solution) = solution.as_ref() else {
                continue;
            };
            if solution.shape() != baseline.shape() {
                sample.status = format!(
                    "shape_mismatch({:?}!={:?})",
                    solution.shape(),
                    baseline.shape()
                );
                continue;
            }
            sample.solve_diff = solution_linf_diff(solution, baseline);
            sample.rel_x_diff = solution_rel_diff(solution, baseline);
        }
    }

    fn fmt_value(value: f64) -> String {
        if value.is_finite() {
            format!("{value:.3}")
        } else {
            "-".to_string()
        }
    }

    fn fmt_scientific(value: f64) -> String {
        if value.is_finite() {
            format!("{value:.3e}")
        } else {
            "-".to_string()
        }
    }

    fn fmt_aggregate(value: Aggregate) -> String {
        if value.mean.is_finite() {
            format!(
                "{:.3} +/- {:.3} [{:.3}, {:.3}]",
                value.mean, value.stddev, value.min, value.max
            )
        } else {
            "-".to_string()
        }
    }

    fn print_auto_sample_table(samples: &[AutoSample]) {
        println!("[BVP Damp Auto] combustion-1000 AtomView+tcc Auto per-run table");
        println!(
            "source   | matrix | variant    | total_ms | solve_diff | selected_backend | policy | auto_mode | parallel_requested | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | status"
        );
        println!(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        );
        for sample in samples {
            println!(
                "{:<8} | {:<6} | {:<10} | {:>8} | {:>10} | {:<16} | {:<6} | {:<9} | {:<18} | {:>18} | {:>18} | {:>20} | {}",
                sample.source,
                sample.matrix,
                sample.variant,
                fmt_value(sample.total_ms),
                fmt_scientific(sample.solve_diff),
                sample.selected_backend,
                sample.runtime_execution_policy,
                sample.auto_execution_mode,
                sample.runtime_parallel_requested,
                fmt_value(sample.residual_values_ms),
                fmt_value(sample.jacobian_values_ms),
                fmt_value(sample.jacobian_assembly_ms),
                sample.status
            );
        }
    }

    fn print_auto_summary_table(variants: &[AutoVariant], samples: &[AutoSample]) {
        println!();
        println!("[BVP Damp Auto] combustion-1000 AtomView+tcc Auto summary table");
        println!(
            "source   | matrix | variant    | ok/runs | total_ms mean+/-std [min,max] | solve_diff mean+/-std | symbolic_ms | linear_ms | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | selected | policy | auto_mode | parallel_requested"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        );
        for variant in variants {
            let rows = samples
                .iter()
                .filter(|sample| {
                    sample.source == variant.source
                        && sample.matrix == variant.matrix
                        && sample.variant == variant.variant
                })
                .collect::<Vec<_>>();
            let ok_runs = rows.iter().filter(|sample| sample.status == "ok").count();
            println!(
                "{:<8} | {:<6} | {:<10} | {:>2}/{:<3} | {:<31} | {:>9.3e} +/- {:<9.1e} | {:<11} | {:<9} | {:<18} | {:<18} | {:<20} | {:<8} | {:<6} | {:<9} | {}",
                variant.source,
                variant.matrix,
                variant.variant,
                ok_runs,
                rows.len(),
                fmt_aggregate(aggregate(rows.iter().map(|sample| sample.total_ms))),
                aggregate(rows.iter().map(|sample| sample.solve_diff)).mean,
                aggregate(rows.iter().map(|sample| sample.solve_diff)).stddev,
                fmt_aggregate(aggregate(rows.iter().map(|sample| sample.symbolic_ms))),
                fmt_aggregate(aggregate(rows.iter().map(|sample| sample.linear_ms))),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.residual_values_ms)
                )),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.jacobian_values_ms)
                )),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.jacobian_assembly_ms)
                )),
                summarize_reason(rows.iter().map(|sample| sample.selected_backend.as_str())),
                summarize_reason(
                    rows.iter()
                        .map(|sample| sample.runtime_execution_policy.as_str())
                ),
                summarize_reason(
                    rows.iter()
                        .map(|sample| sample.auto_execution_mode.as_str())
                ),
                summarize_reason(
                    rows.iter()
                        .map(|sample| sample.runtime_parallel_requested.as_str())
                )
            );
        }
    }

    fn print_auto_runtime_table(variants: &[AutoVariant], samples: &[AutoSample]) {
        println!();
        println!("[BVP Damp Auto] combustion-1000 AtomView+tcc Auto planned/runtime jobs");
        println!(
            "source   | matrix | variant    | auto_workers | min_work/job | auto_res_reason | auto_jac_reason | auto_res_work/job | auto_jac_work/job | auto_res_work/chunk | auto_jac_work/chunk | res_actual_jobs | jac_actual_jobs | res_runtime_work/job | jac_runtime_work/job | res_fallback | jac_fallback"
        );
        println!(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        );
        for variant in variants {
            let rows = samples
                .iter()
                .filter(|sample| {
                    sample.source == variant.source
                        && sample.matrix == variant.matrix
                        && sample.variant == variant.variant
                })
                .collect::<Vec<_>>();
            println!(
                "{:<8} | {:<6} | {:<10} | {:<12} | {:<12} | {:<15} | {:<15} | {:<17} | {:<17} | {:<19} | {:<19} | {:<15} | {:<15} | {:<20} | {:<20} | {:<12} | {}",
                variant.source,
                variant.matrix,
                variant.variant,
                fmt_aggregate(aggregate(rows.iter().map(|sample| sample.auto_workers))),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.auto_min_work_per_job)
                )),
                summarize_reason(
                    rows.iter()
                        .map(|sample| sample.auto_residual_reason.as_str())
                ),
                summarize_reason(rows.iter().map(|sample| sample.auto_sparse_reason.as_str())),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.auto_residual_work_per_job)
                )),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.auto_sparse_work_per_job)
                )),
                fmt_aggregate(aggregate(
                    rows.iter()
                        .map(|sample| sample.auto_residual_work_per_chunk)
                )),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.auto_sparse_work_per_chunk)
                )),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.residual_actual_jobs)
                )),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.sparse_actual_jobs)
                )),
                fmt_aggregate(aggregate(
                    rows.iter()
                        .map(|sample| sample.residual_runtime_work_per_job)
                )),
                fmt_aggregate(aggregate(
                    rows.iter().map(|sample| sample.sparse_runtime_work_per_job)
                )),
                summarize_reason(
                    rows.iter()
                        .map(|sample| sample.residual_fallback_reason.as_str())
                ),
                summarize_reason(
                    rows.iter()
                        .map(|sample| sample.sparse_fallback_reason.as_str())
                )
            );
        }
    }

    let n_steps = 1000usize;
    let repetitions = 2usize;
    let variants = [
        AutoVariant {
            source: "Lambdify",
            matrix: "Sparse",
            variant: "AtomView",
            config: sparse_atomview_lambdify_baseline(),
        },
        AutoVariant {
            source: "Lambdify",
            matrix: "Banded",
            variant: "AtomView",
            config: banded_atomview_lambdify_baseline(),
        },
        AutoVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "tcc/Auto",
            config: auto_release_tcc_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
            ),
        },
        AutoVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "tcc/Auto",
            config: auto_release_tcc_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
            ),
        },
    ];

    let mut samples = Vec::with_capacity(variants.len() * repetitions);
    for repetition in 0..repetitions {
        println!(
            "[BVP Damp Auto] starting repetition {}/{}",
            repetition + 1,
            repetitions
        );
        let mut repetition_samples = Vec::with_capacity(variants.len());
        let mut repetition_solutions = Vec::with_capacity(variants.len());
        for variant in &variants {
            println!(
                "[BVP Damp Auto] running source={} matrix={} variant={}",
                variant.source, variant.matrix, variant.variant
            );
            let _ = io::stdout().flush();
            let (sample, solution) = run_auto_sample(n_steps, variant);
            println!(
                "[BVP Damp Auto] finished source={} matrix={} variant={} status={}",
                sample.source, sample.matrix, sample.variant, sample.status
            );
            repetition_samples.push(sample);
            repetition_solutions.push(solution);
        }
        fill_auto_diffs(&mut repetition_samples, &repetition_solutions);
        samples.extend(repetition_samples);
    }

    print_auto_sample_table(&samples);
    print_auto_summary_table(&variants, &samples);
    print_auto_runtime_table(&variants, &samples);

    assert!(
        samples
            .iter()
            .filter(|sample| sample.source == "Lambdify")
            .all(|sample| sample.status == "ok"),
        "Lambdify baselines must solve successfully before Auto AOT diagnostics are interpreted"
    );
    assert!(
        samples
            .iter()
            .filter(|sample| sample.source == "AOT")
            .all(|sample| sample.status == "ok"),
        "Auto AOT rows must solve successfully"
    );
    assert!(
        samples
            .iter()
            .filter(|sample| sample.source == "AOT")
            .all(|sample| sample.solve_diff.is_finite() && sample.solve_diff <= 1e-6),
        "Auto AOT rows must remain numerically equivalent to the Lambdify baseline"
    );
    assert!(
        samples
            .iter()
            .filter(|sample| sample.source == "AOT")
            .all(|sample| {
                sample.selected_backend == "AotCompiled"
                    && sample.runtime_execution_policy == "Auto"
                    && sample.auto_min_work_per_job.is_finite()
                    && sample.auto_residual_reason != "-"
                    && sample.auto_sparse_reason != "-"
                    && sample.residual_actual_jobs.is_finite()
                    && sample.residual_actual_jobs >= 1.0
                    && sample.sparse_actual_jobs.is_finite()
                    && sample.sparse_actual_jobs >= 1.0
            }),
        "Auto AOT rows must expose selected backend, planned work, actual jobs, and fallback diagnostics"
    );
}

#[test]
fn isolated_cold_race_payload_round_trips_metrics_and_solution() {
    aot_test_report!(isolated_cold_race_payload_round_trips_metrics_and_solution);
    let variant = RaceVariant {
        source: "AOT",
        matrix: "Banded",
        variant: "tcc/chunk4",
        bootstrap_hint: "isolated",
        config: GeneratedBackendConfig::banded_lambdify_defaults(),
    };
    let row = RaceRow {
        source: variant.source,
        matrix: variant.matrix,
        variant: variant.variant,
        bootstrap_hint: variant.bootstrap_hint,
        total_ms: 12.5,
        max_abs_solution: 3.5,
        solve_diff: 0.0,
        rel_x_diff: 0.0,
        iterations: 5,
        linear_solves: 10,
        jac_rebuilds: 1,
        grid_refinements: 0,
        final_grid_points: 31,
        total_timer_ms: 11.0,
        symbolic_timer_ms: 7.0,
        linear_timer_ms: 2.0,
        jac_timer_ms: 1.0,
        fun_timer_ms: 0.5,
        cb_residual_values_ms: 0.2,
        cb_jacobian_values_ms: 0.3,
        cb_jacobian_assembly_ms: 0.1,
        residual_actual_jobs: 4.0,
        sparse_jacobian_actual_jobs: 4.0,
        residual_work_per_job: 50.0,
        sparse_jacobian_work_per_job: 70.0,
        residual_fallback_reason: "none".to_string(),
        sparse_jacobian_fallback_reason: "none".to_string(),
        selected_backend: "AotCompiled".to_string(),
        symbolic_assembly_backend: "ExprLegacy".to_string(),
        aot_build_policy: "RebuildAlways".to_string(),
        initial_generate_ms: 6.0,
        initial_discretization_ms: 1.0,
        initial_symbolic_jacobian_ms: 4.0,
        initial_symbolic_variable_sets_ms: 0.1,
        initial_symbolic_row_differentiation_ms: 3.5,
        initial_symbolic_dense_cache_ms: 0.2,
        initial_symbolic_sparse_flatten_ms: 0.1,
        initial_sparse_prepare_ms: 0.5,
        initial_runtime_binding_ms: 0.1,
        initial_lambdify_jacobian_compile_ms: 0.04,
        initial_lambdify_residual_compile_ms: 0.03,
        post_build_generate_ms: f64::NAN,
        post_build_discretization_ms: f64::NAN,
        post_build_symbolic_jacobian_ms: f64::NAN,
        post_build_sparse_prepare_ms: f64::NAN,
        post_build_runtime_binding_ms: f64::NAN,
        post_build_rebind_ms: 0.05,
        aot_artifact_ms: 0.8,
        aot_module_ms: 0.6,
        aot_residual_lower_ms: 0.4,
        aot_jacobian_lower_ms: 0.3,
        aot_source_emit_ms: 0.2,
        aot_packaging_ms: 0.1,
        aot_materialize_ms: 0.05,
        aot_compile_link_ms: 0.7,
        aot_register_link_ms: 0.04,
        status: "ok".to_string(),
    };
    let decoded = decode_isolated_race_row(
        &format!(
            "{ISOLATED_RACE_ROW_MARKER}\t{}",
            encode_isolated_race_row(&row)
        ),
        &variant,
    );
    assert_eq!(decoded.variant, row.variant);
    assert_eq!(decoded.iterations, row.iterations);
    assert_eq!(decoded.residual_actual_jobs, 4.0);
    assert_eq!(decoded.selected_backend, "AotCompiled");
    assert_eq!(decoded.initial_symbolic_row_differentiation_ms, 3.5);
    assert!(decoded.post_build_generate_ms.is_nan());

    let solution = DMatrix::from_column_slice(
        6,
        2,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    );
    let decoded_solution = decode_isolated_solution(&format!(
        "{ISOLATED_RACE_SOLUTION_MARKER}\t{}",
        encode_isolated_solution(&solution)
    ));
    assert_eq!(decoded_solution, solution);
}

fn run_combustion_3000_banded_isolated_stress_story(
    test_name: &str,
    frontend_label: &str,
    variants: Vec<RaceVariant>,
) {
    let n_steps = 3_000usize;
    let repetitions = 2usize;
    let variants = apply_optional_release_matrix_filter(&variants);

    if let Ok(index) = std::env::var(ISOLATED_STRESS_CHILD_INDEX_ENV) {
        let index = index
            .parse::<usize>()
            .expect("isolated stress child index should be an integer");
        let variant = variants
            .get(index)
            .unwrap_or_else(|| panic!("isolated stress child variant index {index} is invalid"));
        let (row, solution) = run_race_variant(
            n_steps,
            variant.source,
            variant.matrix,
            variant.variant,
            variant.bootstrap_hint,
            variant.config.clone(),
        );
        assert_eq!(
            row.status, "ok",
            "isolated stress child {} should solve successfully",
            variant.variant
        );
        let solution = solution
            .as_ref()
            .expect("isolated stress child should provide its converged solution");
        println!("{ISOLATED_RACE_PID_MARKER}\t{}", std::process::id());
        println!(
            "{ISOLATED_RACE_ROW_MARKER}\t{}",
            encode_isolated_race_row(&row)
        );
        println!(
            "{ISOLATED_RACE_SOLUTION_MARKER}\t{}",
            encode_isolated_solution(solution)
        );
        return;
    }

    println!(
        "[BVP Damp isolated cold] protocol cooldown_ms={}, cleanup_child_artifacts={}",
        cold_cooldown_ms(),
        clean_cold_artifacts_enabled()
    );
    let samples = run_isolated_race_samples(test_name, &variants, repetitions);
    let rows = summarize_samples(&variants, &samples);
    print_isolated_cold_sample_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} raw process-isolated cold observations"
        ),
        &samples,
        variants.len(),
    );
    println!();
    print_e2e_correctness_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT correctness"
        ),
        &rows,
    );
    println!();
    print_e2e_performance_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT timing/counters"
        ),
        &rows,
    );
    println!();
    print_e2e_callback_stage_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT callback stages"
        ),
        &rows,
    );
    println!();
    print_e2e_lifecycle_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT lifecycle/refinement stages"
        ),
        &rows,
    );
    println!();
    print_e2e_bootstrap_pass_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT symbolic handoff passes"
        ),
        &rows,
    );
    println!();
    print_e2e_symbolic_jacobian_detail_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} internal initial symbolic-Jacobian stages"
        ),
        &rows,
    );
    println!();
    print_e2e_lambdify_binding_detail_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} Lambdify callback compilation stages"
        ),
        &rows,
    );
    println!();
    print_e2e_aot_bootstrap_table(
        &format!(
            "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT cold-build stages"
        ),
        &rows,
    );

    assert!(
        rows.iter()
            .filter(|row| row.source == "Lambdify")
            .all(|row| row.ok_runs == row.runs),
        "Banded Lambdify baseline must solve before AOT stress rows are interpreted"
    );
    assert!(
        rows.iter().all(|row| row.ok_runs == row.runs),
        "all selected combustion-3000 Banded Lambdify/AOT stress variants should solve successfully"
    );
    assert!(
        rows.iter()
            .filter(|row| row.source == "AOT")
            .all(|row| row.solve_diff.mean.is_finite() && row.solve_diff.mean <= 1e-5),
        "AOT stress rows must remain numerically equivalent to the Banded Lambdify baseline"
    );
    for row in rows.iter().filter(|row| row.variant == "tcc/chunk4") {
        assert!(
            row.residual_actual_jobs.mean > 1.0 && row.sparse_jacobian_actual_jobs.mean > 1.0,
            "process-isolated chunk4 row must retain real parallel execution"
        );
        assert_eq!(
            row.residual_fallback_reason, "none",
            "process-isolated chunk4 residual callback must not fall back to sequential"
        );
        assert_eq!(
            row.sparse_jacobian_fallback_reason, "none",
            "process-isolated chunk4 Jacobian callback must not fall back to sequential"
        );
    }
}

#[test]
#[ignore = "very heavy process-isolated cold combustion-3000 ExprLegacy Banded Lambdify vs tcc whole/chunk4 end-to-end control"]
fn combustion_3000_banded_lambdify_vs_aot_end_to_end_stress() {
    aot_test_report!(combustion_3000_banded_lambdify_vs_aot_end_to_end_stress);
    run_combustion_3000_banded_isolated_stress_story(
        "numerical::BVP_Damp::test_aot_race_stress::tests::combustion_3000_banded_lambdify_vs_aot_end_to_end_stress",
        "ExprLegacy",
        vec![
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "ExprLegacy",
                bootstrap_hint: "exprlegacy+lambdify",
                config: GeneratedBackendConfig::banded_lambdify_defaults()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_build_if_missing_release()
                        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                        .with_aot_codegen_backend(AotCodegenBackend::C)
                        .with_aot_c_compiler("tcc"),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_build_if_missing_release()
                        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                        .with_aot_codegen_backend(AotCodegenBackend::C)
                        .with_aot_c_compiler("tcc"),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
        ],
    );
}

#[test]
#[ignore = "very heavy process-isolated cold combustion-3000 AtomView Banded Lambdify vs tcc whole/chunk4 production end-to-end stress test"]
fn combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress() {
    aot_test_report!(combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress);
    run_combustion_3000_banded_isolated_stress_story(
        "numerical::BVP_Damp::test_aot_race_stress::tests::combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress",
        "AtomView",
        vec![
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "AtomView",
                bootstrap_hint: "atomview+lambdify",
                config: banded_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/whole",
                bootstrap_hint: "atomview+rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/chunk4",
                bootstrap_hint: "atomview+rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
        ],
    );
}

fn run_combustion_1000_symbolic_frontend_honest_wall_clock_table(
    test_name: &str,
    matrix: &str,
    variants: Vec<RaceVariant>,
) {
    let n_steps = 1_000usize;
    let repetitions = 3usize;
    let variants = apply_optional_release_matrix_filter(&variants);

    if let Ok(index) = std::env::var(ISOLATED_STRESS_CHILD_INDEX_ENV) {
        let index = index
            .parse::<usize>()
            .expect("isolated symbolic-frontend child index should be an integer");
        let variant = variants.get(index).unwrap_or_else(|| {
            panic!("isolated symbolic-frontend child variant index {index} is invalid")
        });
        let (row, solution) = run_race_variant(
            n_steps,
            variant.source,
            variant.matrix,
            variant.variant,
            variant.bootstrap_hint,
            variant.config.clone(),
        );
        assert_eq!(
            row.status, "ok",
            "isolated symbolic-frontend child {} should solve successfully",
            variant.variant
        );
        let solution = solution
            .as_ref()
            .expect("isolated symbolic-frontend child should provide its converged solution");
        println!("{ISOLATED_RACE_PID_MARKER}\t{}", std::process::id());
        println!(
            "{ISOLATED_RACE_ROW_MARKER}\t{}",
            encode_isolated_race_row(&row)
        );
        println!(
            "{ISOLATED_RACE_SOLUTION_MARKER}\t{}",
            encode_isolated_solution(solution)
        );
        return;
    }

    println!(
        "[BVP Damp symbolic frontend cold] protocol cooldown_ms={}, cleanup_child_artifacts={}",
        cold_cooldown_ms(),
        clean_cold_artifacts_enabled()
    );
    let samples = run_isolated_race_samples(test_name, &variants, repetitions);
    let rows = summarize_samples(&variants, &samples);
    print_isolated_cold_sample_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} raw process-isolated observations"
        ),
        &samples,
        variants.len(),
    );
    println!();
    print_e2e_correctness_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} ExprLegacy/AtomView correctness"
        ),
        &rows,
    );
    println!();
    print_e2e_performance_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} ExprLegacy/AtomView wall-clock and solver stages"
        ),
        &rows,
    );
    println!();
    print_e2e_callback_stage_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} callback/runtime stages"
        ),
        &rows,
    );
    println!();
    print_e2e_lifecycle_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} backend selection and symbolic totals"
        ),
        &rows,
    );
    println!();
    print_e2e_bootstrap_pass_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} symbolic handoff stages"
        ),
        &rows,
    );
    println!();
    print_e2e_symbolic_jacobian_detail_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} internal initial symbolic-Jacobian stages"
        ),
        &rows,
    );
    println!();
    print_e2e_aot_bootstrap_table(
        &format!(
            "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} tcc cold-build stages"
        ),
        &rows,
    );

    assert!(
        rows.iter().all(|row| row.ok_runs == row.runs),
        "every symbolic-frontend cold comparison row should solve successfully"
    );
    assert!(
        rows.iter()
            .all(|row| row.solve_diff.mean.is_finite() && row.solve_diff.mean <= 1e-5),
        "ExprLegacy, AtomView, Lambdify and tcc AOT rows must remain numerically equivalent"
    );
    assert!(
        rows.iter()
            .filter(|row| row.source == "AOT")
            .all(|row| row.post_build_symbolic_jacobian_ms.mean.is_nan()),
        "fresh AOT rebinding must not rebuild either symbolic frontend after compilation"
    );
}

#[test]
#[ignore = "process-isolated combustion-1000 Banded ExprLegacy/AtomView symbolic frontend comparison for Lambdify and tcc AOT"]
fn combustion_1000_banded_symbolic_frontend_honest_wall_clock_table() {
    aot_test_report!(combustion_1000_banded_symbolic_frontend_honest_wall_clock_table);
    run_combustion_1000_symbolic_frontend_honest_wall_clock_table(
        "numerical::BVP_Damp::test_aot_race_stress::tests::combustion_1000_banded_symbolic_frontend_honest_wall_clock_table",
        "Banded",
        vec![
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "ExprLegacy",
                bootstrap_hint: "exprlegacy+lambdify",
                config: GeneratedBackendConfig::banded_lambdify_defaults()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            },
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "AtomView",
                bootstrap_hint: "atomview+lambdify",
                config: banded_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "ExprLegacy+tcc",
                bootstrap_hint: "exprlegacy+rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_build_if_missing_release()
                        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                        .with_aot_codegen_backend(AotCodegenBackend::C)
                        .with_aot_c_compiler("tcc"),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "AtomView+tcc",
                bootstrap_hint: "atomview+rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
        ],
    );
}

#[test]
#[ignore = "process-isolated combustion-1000 Banded AtomView tcc Full vs NoCse optimization-profile end-to-end story"]
fn combustion_1000_banded_atomview_tcc_cse_profile_end_to_end_story() {
    aot_test_report!(combustion_1000_banded_atomview_tcc_cse_profile_end_to_end_story);
    run_combustion_1000_symbolic_frontend_honest_wall_clock_table(
        "numerical::BVP_Damp::test_aot_race_stress::tests::combustion_1000_banded_atomview_tcc_cse_profile_end_to_end_story",
        "Banded",
        vec![
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "AtomView",
                bootstrap_hint: "atomview+lambdify",
                config: banded_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/full",
                bootstrap_hint: "atomview+full+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
                        .with_atom_optimization_profile(AtomOptimizationProfile::Full),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/no_cse",
                bootstrap_hint: "atomview+no_cse+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
                        .with_atom_optimization_profile(AtomOptimizationProfile::NoCse),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
        ],
    );
}

#[test]
#[ignore = "process-isolated combustion-1000 Sparse ExprLegacy/AtomView symbolic frontend comparison for Lambdify and tcc AOT"]
fn combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table() {
    aot_test_report!(combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table);
    run_combustion_1000_symbolic_frontend_honest_wall_clock_table(
        "numerical::BVP_Damp::test_aot_race_stress::tests::combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table",
        "Sparse",
        vec![
            RaceVariant {
                source: "Lambdify",
                matrix: "Sparse",
                variant: "ExprLegacy",
                bootstrap_hint: "exprlegacy+lambdify",
                config: GeneratedBackendConfig::sparse_defaults()
                    .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
            },
            RaceVariant {
                source: "Lambdify",
                matrix: "Sparse",
                variant: "AtomView",
                bootstrap_hint: "atomview+lambdify",
                config: sparse_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "ExprLegacy+tcc",
                bootstrap_hint: "exprlegacy+rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_build_if_missing_release()
                        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                        .with_aot_codegen_backend(AotCodegenBackend::C)
                        .with_aot_c_compiler("tcc"),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "AtomView+tcc",
                bootstrap_hint: "atomview+rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
        ],
    );
}

struct DampedLifecycleRow {
    matrix: &'static str,
    phase: &'static str,
    total_ms: f64,
    symbolic_ms: f64,
    linear_ms: f64,
    solve_diff: f64,
    initial_generate_ms: f64,
    post_build_rebind_ms: f64,
    compile_link_ms: f64,
    selected_backend: String,
    build_policy: String,
}

fn run_damped_lifecycle_row(
    n_steps: usize,
    matrix: &'static str,
    phase: &'static str,
    config: GeneratedBackendConfig,
    baseline: Option<&DMatrix<f64>>,
) -> (DampedLifecycleRow, DMatrix<f64>, GeneratedBackendConfig) {
    let begin = Instant::now();
    let mut solver = make_combustion_solver(n_steps, config);
    solver
        .try_solver()
        .unwrap_or_else(|err| panic!("{matrix}/{phase} damped lifecycle solve failed: {err:?}"));
    let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
    let solution = solver
        .get_result()
        .expect("successful damped lifecycle solve must expose a result");
    let stats = solver.get_statistics();
    let row = DampedLifecycleRow {
        matrix,
        phase,
        total_ms,
        symbolic_ms: stats_timer_ms(&stats, "Symbolic Operations"),
        linear_ms: stats_timer_ms(&stats, "Linear System"),
        solve_diff: baseline
            .map(|reference| solution_linf_diff(&solution, reference))
            .unwrap_or(0.0),
        initial_generate_ms: stats_diagnostic_ms(
            &stats,
            "generated.handoff.initial_generate_wall_ms",
        ),
        post_build_rebind_ms: stats_diagnostic_ms(
            &stats,
            "generated.handoff.post_build_rebind_wall_ms",
        ),
        compile_link_ms: stats_diagnostic_ms(&stats, "generated.aot.compile_link_ms"),
        selected_backend: stats_diagnostic_string(&stats, "generated.selected_backend"),
        build_policy: stats_diagnostic_string(&stats, "aot.build_policy"),
    };
    (row, solution, solver.generated_backend_config().clone())
}

fn print_damped_lifecycle_table(rows: &[DampedLifecycleRow]) {
    println!(
        "[BVP Damp lifecycle] combustion-1000 AtomView tcc BuildIfMissing -> RequirePrebuilt correctness"
    );
    println!("matrix | phase      | selected_backend | build_policy    | solve_diff");
    println!("{}", "-".repeat(84));
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:<16} | {:<15} | {:.6e}",
            row.matrix, row.phase, row.selected_backend, row.build_policy, row.solve_diff
        );
    }
    println!();
    println!(
        "[BVP Damp lifecycle] wall-clock and artifact stages; all time columns are milliseconds"
    );
    println!(
        "matrix | phase      | total_ms | symbolic_ms | linear_ms | initial_generate | rebind_ms | compile_link"
    );
    println!("{}", "-".repeat(112));
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:>8.3} | {:>11.3} | {:>9.3} | {:>16.3} | {:>9.3} | {:>12.3}",
            row.matrix,
            row.phase,
            row.total_ms,
            row.symbolic_ms,
            row.linear_ms,
            row.initial_generate_ms,
            row.post_build_rebind_ms,
            row.compile_link_ms,
        );
    }
}

fn print_damped_warm_comparison_table(
    build_row: &DampedLifecycleRow,
    samples: &[(usize, usize, DampedLifecycleRow)],
) {
    println!("[BVP Damp warm] Banded AtomView Lambdify vs tcc RequirePrebuilt; setup build row");
    println!(
        "phase | selected_backend | build_policy    | total_ms | symbolic_ms | rebind_ms | compile_link | solve_diff"
    );
    println!("{}", "-".repeat(120));
    println!(
        "{:<5} | {:<16} | {:<15} | {:>8.3} | {:>11.3} | {:>9.3} | {:>12.3} | {:.6e}",
        build_row.phase,
        build_row.selected_backend,
        build_row.build_policy,
        build_row.total_ms,
        build_row.symbolic_ms,
        build_row.post_build_rebind_ms,
        build_row.compile_link_ms,
        build_row.solve_diff,
    );
    println!();
    println!(
        "[BVP Damp warm] measured rows after cooldown_ms={}; milliseconds",
        warm_cooldown_ms()
    );
    println!(
        "rep | pos | phase      | selected_backend | build_policy    | total_ms | symbolic_ms | linear_ms | initial_generate | compile_link | solve_diff"
    );
    println!("{}", "-".repeat(160));
    for (repetition, position, row) in samples {
        println!(
            "{:>3} | {:>3} | {:<10} | {:<16} | {:<15} | {:>8.3} | {:>11.3} | {:>9.3} | {:>16.3} | {:>12.3} | {:.6e}",
            repetition,
            position,
            row.phase,
            row.selected_backend,
            row.build_policy,
            row.total_ms,
            row.symbolic_ms,
            row.linear_ms,
            row.initial_generate_ms,
            row.compile_link_ms,
            row.solve_diff,
        );
    }
    println!();
    println!(
        "[BVP Damp warm] paired summary: build row excluded; each route has the same cooldown and alternating order"
    );
    println!(
        "phase      | runs | total_ms mean+/-std [min,max] | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff"
    );
    println!("{}", "-".repeat(150));
    for phase in ["lambdify", "prebuilt"] {
        let rows = samples
            .iter()
            .filter(|(_, _, row)| row.phase == phase)
            .map(|(_, _, row)| row)
            .collect::<Vec<_>>();
        let total = aggregate(rows.iter().map(|row| row.total_ms));
        let symbolic = aggregate(rows.iter().map(|row| row.symbolic_ms));
        let linear = aggregate(rows.iter().map(|row| row.linear_ms));
        let max_diff = rows
            .iter()
            .map(|row| row.solve_diff)
            .fold(0.0_f64, f64::max);
        println!(
            "{:<10} | {:>4} | {:<32} | {:<21} | {:<21} | {:.6e}",
            phase,
            rows.len(),
            fmt_agg(total),
            fmt_agg_short(symbolic),
            fmt_agg_short(linear),
            max_diff,
        );
    }
}

#[test]
#[ignore = "heavy damped combustion-1000 Sparse/Banded AtomView tcc artifact lifecycle: BuildIfMissing then strict RequirePrebuilt reuse"]
fn combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story() {
    aot_test_report!(combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story);
    let n_steps = 1_000;
    let (baseline_row, baseline, _) = run_damped_lifecycle_row(
        n_steps,
        "Banded",
        "baseline",
        banded_atomview_lambdify_baseline(),
        None,
    );
    let mut rows = vec![baseline_row];
    let build_configs = [
        (
            "Sparse",
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        ),
        (
            "Banded",
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
        ),
    ];
    for (matrix, config) in build_configs {
        let config = config
            .with_aot_compile_dev_fastest()
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_chunking_policy(whole_chunking())
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            });
        let (built_row, _, built_config) =
            run_damped_lifecycle_row(n_steps, matrix, "build", config, Some(&baseline));
        assert!(
            built_config.resolver.is_some(),
            "{matrix} BuildIfMissing must preserve a resolver snapshot for strict reuse"
        );
        rows.push(built_row);
        let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        for _ in 0..3 {
            let (prebuilt_row, _, _) = run_damped_lifecycle_row(
                n_steps,
                matrix,
                "prebuilt",
                strict_config.clone(),
                Some(&baseline),
            );
            rows.push(prebuilt_row);
        }
    }

    print_damped_lifecycle_table(&rows);
    assert!(
        rows.iter().all(|row| row.solve_diff <= 1e-5),
        "Damped lifecycle routes must remain equivalent to the Lambdify baseline"
    );
    assert!(
        rows.iter()
            .filter(|row| row.phase != "baseline")
            .all(|row| row.selected_backend == "AotCompiled"),
        "BuildIfMissing and RequirePrebuilt routes must run compiled callbacks"
    );
    assert!(
        rows.iter()
            .filter(|row| row.phase == "prebuilt")
            .all(|row| row.build_policy == "RequirePrebuilt"),
        "warm rows must be strict RequirePrebuilt executions"
    );
}

#[test]
#[ignore = "heavy warm repeated-solve comparison with cooldown: combustion-1000 Banded AtomView Lambdify vs strict tcc RequirePrebuilt"]
fn combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story() {
    aot_test_report!(combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story);
    let n_steps = 1_000;
    let repetitions = 5;
    let cooldown_ms = warm_cooldown_ms();
    let (_, reference_solution, _) = run_damped_lifecycle_row(
        n_steps,
        "Banded",
        "reference",
        banded_atomview_lambdify_baseline(),
        None,
    );
    let build_config = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
        .with_aot_compile_dev_fastest()
        .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
        .with_aot_chunking_policy(whole_chunking())
        .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        });
    let (build_row, _, built_config) = run_damped_lifecycle_row(
        n_steps,
        "Banded",
        "build",
        build_config,
        Some(&reference_solution),
    );
    assert!(
        built_config.resolver.is_some(),
        "warm comparison requires a resolver snapshot from BuildIfMissing"
    );
    let prebuilt_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    let mut samples = Vec::with_capacity(repetitions * 2);

    for repetition in 1..=repetitions {
        let phases = if repetition % 2 == 1 {
            ["lambdify", "prebuilt"]
        } else {
            ["prebuilt", "lambdify"]
        };
        for (position, phase) in phases.into_iter().enumerate() {
            if cooldown_ms > 0 {
                thread::sleep(Duration::from_millis(cooldown_ms));
            }
            let config = if phase == "lambdify" {
                banded_atomview_lambdify_baseline()
            } else {
                prebuilt_config.clone()
            };
            let (row, _, _) = run_damped_lifecycle_row(
                n_steps,
                "Banded",
                phase,
                config,
                Some(&reference_solution),
            );
            samples.push((repetition, position + 1, row));
        }
    }

    print_damped_warm_comparison_table(&build_row, &samples);
    assert!(
        build_row.selected_backend == "AotCompiled" && build_row.solve_diff <= 1e-5,
        "setup build row must install a correct compiled backend"
    );
    assert!(
        samples.iter().all(|(_, _, row)| row.solve_diff <= 1e-5),
        "all measured warm rows must match the common Lambdify reference"
    );
    assert!(
        samples
            .iter()
            .filter(|(_, _, row)| row.phase == "prebuilt")
            .all(|(_, _, row)| row.selected_backend == "AotCompiled"
                && row.build_policy == "RequirePrebuilt"
                && row.compile_link_ms.is_nan()),
        "measured prebuilt rows must stay compiled without any rebuild/link step"
    );
    assert_eq!(
        samples
            .iter()
            .filter(|(_, _, row)| row.phase == "lambdify")
            .count(),
        repetitions
    );
    assert_eq!(
        samples
            .iter()
            .filter(|(_, _, row)| row.phase == "prebuilt")
            .count(),
        repetitions
    );
}
