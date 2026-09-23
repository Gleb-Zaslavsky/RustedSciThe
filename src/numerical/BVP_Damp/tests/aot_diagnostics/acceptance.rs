#[test]
//  #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
fn aot_rust_default_exact_examples_sequential_cover_tens_and_hundreds_of_steps() {
    println!("[AOT acceptance] default Rust AOT smoke: exact examples, sequential execution");
    let configs = [
        (
            "two-point-64",
            NonlinEquation::TwoPointBVP,
            64usize,
            1.5e-2f64,
        ),
        (
            "two-point-240",
            NonlinEquation::TwoPointBVP,
            240usize,
            6.0e-3f64,
        ),
        ("clairaut-72", NonlinEquation::Clairaut, 72usize, 1.5e-2f64),
        (
            "clairaut-220",
            NonlinEquation::Clairaut,
            220usize,
            1.25e-2f64,
        ),
    ];

    for (label, equation, n_steps, max_tol) in configs {
        let generated_backend_config =
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
        let mut solver = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            generated_backend_config,
        );
        let _guard = solve_with_aot_and_report(&mut solver, label)
            .expect("sequential AOT exact-example acceptance case should solve");

        let error = match equation {
            NonlinEquation::TwoPointBVP => {
                max_abs_error_against_exact(&solver, |x| (-x * x / 4.0).exp())
            }
            NonlinEquation::Clairaut => l2_error_against_exact(&solver, |x| {
                1.0 + (x - 1.0).powi(2) - (x - 1.0).powi(3) / 6.0 + (x - 1.0).powi(4) / 12.0
            }),
            _ => unreachable!("only exact sequential examples are expected here"),
        };

        println!("[AOT exact sequential] {label}: n_steps={n_steps}, error={error:.6e}");
        assert!(
            error < max_tol,
            "{label}: exact-solution error {error} exceeded tolerance {max_tol}"
        );
    }
}

#[test]
//   #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
fn aot_rust_default_parallel_exact_examples_cover_parallel_modes_and_chunking() {
    println!("[AOT acceptance] default Rust AOT smoke: exact examples, parallel/chunked execution");
    let lane_parallel = SolverParams {
        max_jac: Some(5),
        max_damp_iter: Some(5),
        damp_factor: None,
        adaptive: None,
    };
    let parachute_parallel = SolverParams {
        max_jac: Some(5),
        max_damp_iter: Some(5),
        damp_factor: None,
        adaptive: None,
    };

    let cases = [
        (
            "parachute-parallel-48",
            NonlinEquation::ParachuteEquation,
            48usize,
            parachute_parallel.clone(),
            5.0e-3f64,
        ),
        (
            "lane-emden-parallel-180",
            NonlinEquation::LaneEmden5,
            180usize,
            lane_parallel.clone(),
            3.5e-4f64,
        ),
    ];

    for (label, equation, n_steps, strategy_params, l2_tol) in cases {
        let generated_backend_config =
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_aot_execution_policy(sparse_parallel_policy())
                    .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                        Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                        Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    ));
        let mut solver = make_example_solver(
            &equation,
            n_steps,
            Some(strategy_params),
            generated_backend_config,
        );
        let _guard = solve_with_aot_and_report(&mut solver, label)
            .expect("parallel AOT exact-example acceptance case should solve");

        let l2_error = match equation {
            NonlinEquation::ParachuteEquation => {
                l2_error_against_exact(&solver, |x| (((2.0 * x).exp() + 1.0) / 2.0).ln() - x)
            }
            NonlinEquation::LaneEmden5 => {
                l2_error_against_exact(&solver, |x| (1.0 + x * x / 3.0).powf(-0.5))
            }
            _ => unreachable!("only parallel analytical examples are expected here"),
        };

        println!("[AOT exact parallel] {label}: n_steps={n_steps}, l2_error={l2_error:.6e}");
        assert!(
            l2_error < l2_tol,
            "{label}: L2 exact-solution error {l2_error} exceeded tolerance {l2_tol}"
        );
    }
}

#[test]
// #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
fn aot_rust_default_combustion_acceptance_covers_sequential_parallel_and_varied_grids() {
    println!("[AOT acceptance] default Rust AOT smoke: combustion sequential/parallel grids");
    let sequential_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
    let parallel_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_aot_execution_policy(sparse_parallel_policy())
                .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                ));

    let mut sequential = make_combustion_solver(36, sequential_config.clone());
    let _sequential_guard = solve_with_aot_and_report(&mut sequential, "combustion-sequential-36")
        .expect("sequential AOT combustion case should solve");
    let sequential_solution = sequential
        .get_result()
        .expect("sequential AOT combustion case should produce a solution");
    assert!(
        sequential_solution.iter().all(|value| value.is_finite()),
        "sequential AOT combustion solution should remain finite"
    );

    let mut parallel = make_combustion_solver(36, parallel_config.clone());
    let _parallel_guard = solve_with_aot_and_report(&mut parallel, "combustion-parallel-36")
        .expect("parallel AOT combustion case should solve");
    let parallel_solution = parallel
        .get_result()
        .expect("parallel AOT combustion case should produce a solution");
    assert!(
        parallel_solution.iter().all(|value| value.is_finite()),
        "parallel AOT combustion solution should remain finite"
    );

    let max_difference = sequential_solution
        .iter()
        .zip(parallel_solution.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max);
    println!("[AOT combustion compare] n_steps=36, max_difference_seq_vs_par={max_difference:.6e}");
    assert!(
        max_difference < 1.0e-4,
        "AOT combustion sequential/parallel disagreement {max_difference} is too large"
    );

    let mut large_sequential = make_combustion_solver(256, sequential_config);
    let _large_sequential_guard =
        solve_with_aot_and_report(&mut large_sequential, "combustion-sequential-256")
            .expect("large-grid sequential AOT combustion case should solve");
    let large_sequential_solution = large_sequential
        .get_result()
        .expect("large-grid sequential AOT combustion case should produce a solution");
    assert!(
        large_sequential_solution
            .iter()
            .all(|value| value.is_finite()),
        "large-grid sequential AOT combustion solution should remain finite"
    );

    let mut large_parallel = make_combustion_solver(256, parallel_config);
    let _large_parallel_guard =
        solve_with_aot_and_report(&mut large_parallel, "combustion-parallel-256")
            .expect("large-grid parallel AOT combustion case should solve");
    let large_solution = large_parallel
        .get_result()
        .expect("large-grid AOT combustion case should produce a solution");
    assert!(
        large_solution.iter().all(|value| value.is_finite()),
        "large-grid parallel AOT combustion solution should remain finite"
    );

    let large_max_difference = large_sequential_solution
        .iter()
        .zip(large_solution.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max);
    println!(
        "[AOT combustion compare] n_steps=256, max_difference_seq_vs_par={large_max_difference:.6e}"
    );
    assert!(
        large_max_difference < 1.0e-4,
        "AOT combustion sequential/parallel disagreement {large_max_difference} is too large"
    );
}

#[test]
fn aot_tcc_smoke_exact_two_point_small_grid_solves() {
    println!(
        "[AOT acceptance] compact TCC smoke: two-point exact BVP, sparse AtomView, sequential execution"
    );
    let generated_backend_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
    let mut solver = make_example_solver(
        &NonlinEquation::TwoPointBVP,
        40,
        Some(SolverParams::default()),
        generated_backend_config,
    );

    match solve_with_aot_and_report(&mut solver, "tcc-smoke-two-point-40") {
        Ok(_guard) => {
            let error = max_abs_error_against_exact(&solver, |x| (-x * x / 4.0).exp());
            println!("[AOT TCC smoke] two-point-40: error={error:.6e}");
            assert!(
                error < 2.0e-2,
                "TCC smoke exact-solution error {error} exceeded tolerance"
            );
        }
        Err(err) if is_aot_environment_issue(&err) => {
            eprintln!(
                "[AOT TCC smoke] skipped because TCC toolchain/artifact environment is unavailable: {err:?}"
            );
        }
        Err(err) => panic!("TCC smoke should solve or report an environment issue: {err:?}"),
    }
}
// Historical local timing notes kept only as archaeological context. The live
// source of truth is the multi-run, multi-toolchain table printed by
// `aot_combustion_parallel_tuning_reports_runtime_table` and summarized in
// BVP_DAMP_STORY_TESTS.md.
/*
        слишком мало чанков: не хватает загрузки
    слишком много чанков: overhead на orchestration начинает съедать выигрыш
    средний режим вроде 8x8 + jobs8 попадает в sweet spot
    И очень важно:

    max_diff_vs_seq = 0 у всех конфигураций
    это именно то, что и нужно было доказать: parallel AOT меняет скорость, но не математику.
    CPU 4 Cores
    [AOT combustion tuning map] scenario=medium-grid, n_steps=128
    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     128 |     7113.999 |      440.910 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     128 |    15332.448 |      335.703 |          1.313 |     0.000000e0
    par-8x8-jobs8            |     128 |    10177.598 |     1116.167 |          0.395 |     0.000000e0
    par-16x16-jobs16         |     128 |     8069.948 |      261.911 |          1.683 |     0.000000e0
    par-res16-row32-jobs8    |     128 |     6763.878 |      280.503 |          1.572 |     0.000000e0
    [AOT tuning winner] scenario=medium-grid, config=par-16x16-jobs16, n_steps=128, solve_ms=261.911, speedup_vs_seq=1.683, bootstrap_ms=8069.948

    [AOT combustion tuning map] scenario=large-grid, n_steps=256
    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     256 |    15992.826 |      803.811 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     256 |    36969.672 |     1046.569 |          0.768 |     0.000000e0
    par-8x8-jobs8            |     256 |    22350.296 |      510.894 |          1.573 |     0.000000e0
    par-16x16-jobs16         |     256 |    18268.103 |      956.696 |          0.840 |     0.000000e0
    par-res16-row32-jobs8    |     256 |    13534.274 |      593.440 |          1.354 |     0.000000e0
    [AOT tuning winner] scenario=large-grid, config=par-8x8-jobs8, n_steps=256, solve_ms=510.894, speedup_vs_seq=1.573, bootstrap_ms=22350.296
        CPU 8 Core:
        run 1
    [AOT combustion tuning map] scenario=medium-grid, n_steps=128
config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
------------------------------------------------------------------------------------------------
sequential-baseline      |     128 |     2130.826 |      137.532 |          1.000 |     0.000000e0
par-4x4-jobs4            |     128 |     3685.552 |      116.561 |          1.180 |     0.000000e0
par-8x8-jobs8            |     128 |     2446.450 |      104.325 |          1.318 |     0.000000e0
par-16x16-jobs16         |     128 |     1870.474 |      116.204 |          1.184 |     0.000000e0
par-res16-row32-jobs8    |     128 |     1807.076 |      109.260 |          1.259 |     0.000000e0
[AOT tuning winner] scenario=medium-grid, config=par-8x8-jobs8, n_steps=128, solve_ms=104.325, speedup_vs_seq=1.318, bootstrap_ms=2446.450

[AOT combustion tuning map] scenario=large-grid, n_steps=256
config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
------------------------------------------------------------------------------------------------
sequential-baseline      |     256 |     3382.138 |      203.362 |          1.000 |     0.000000e0
par-4x4-jobs4            |     256 |    12063.711 |      172.593 |          1.178 |     0.000000e0
par-8x8-jobs8            |     256 |     7365.829 |      168.625 |          1.206 |     0.000000e0
par-16x16-jobs16         |     256 |     4900.391 |      170.448 |          1.193 |     0.000000e0
par-res16-row32-jobs8    |     256 |     3832.109 |      168.844 |          1.204 |     0.000000e0
[AOT tuning winner] scenario=large-grid, config=par-8x8-jobs8, n_steps=256, solve_ms=168.625, speedup_vs_seq=1.206, bootstrap_ms=7365.829
run 2
[AOT combustion tuning map] scenario=medium-grid, n_steps=128
config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
------------------------------------------------------------------------------------------------
sequential-baseline      |     128 |     2044.555 |      127.818 |          1.000 |     0.000000e0
par-4x4-jobs4            |     128 |     3723.176 |      115.799 |          1.104 |     0.000000e0
par-8x8-jobs8            |     128 |     2492.751 |      108.581 |          1.177 |     0.000000e0
par-16x16-jobs16         |     128 |     1891.942 |      107.237 |          1.192 |     0.000000e0
par-res16-row32-jobs8    |     128 |     1794.337 |      107.782 |          1.186 |     0.000000e0
[AOT tuning winner] scenario=medium-grid, config=par-16x16-jobs16, n_steps=128, solve_ms=107.237, speedup_vs_seq=1.192, bootstrap_ms=1891.942

config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
------------------------------------------------------------------------------------------------
sequential-baseline      |     256 |     3503.787 |      188.826 |          1.000 |     0.000000e0
par-4x4-jobs4            |     256 |    12043.939 |      167.994 |          1.124 |     0.000000e0
par-8x8-jobs8            |     256 |     7444.171 |      167.913 |          1.125 |     0.000000e0
par-16x16-jobs16         |     256 |     5047.130 |      164.290 |          1.149 |     0.000000e0
par-res16-row32-jobs8    |     256 |     3913.400 |      176.590 |          1.069 |     0.000000e0
[AOT tuning winner] scenario=large-grid, config=par-16x16-jobs16, n_steps=256, solve_ms=164.290, speedup_vs_seq=1.149, bootstrap_ms=5047.130

        */
#[test]
fn isolated_tuning_solution_payload_round_trips() {
    let solution = DMatrix::from_column_slice(2, 3, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let encoded = format!(
        "{ISOLATED_TUNING_SOLUTION_MARKER}\t{}",
        encode_isolated_tuning_solution(&solution)
    );
    assert_eq!(
        decode_isolated_tuning_solution(&encoded),
        solution.iter().copied().collect::<Vec<_>>()
    );
    let metrics = IsolatedColdMetrics {
        total_timer_ms: 1.0,
        symbolic_ms: 2.0,
        linear_ms: 3.0,
        jac_ms: 4.0,
        fun_ms: 5.0,
        cb_residual_values_ms: 6.0,
        cb_jacobian_values_ms: 7.0,
        cb_jacobian_assembly_ms: 8.0,
        residual_actual_jobs: 4.0,
        sparse_jacobian_actual_jobs: 4.0,
        initial_symbolic_jacobian_ms: 9.0,
        post_build_rebind_ms: 10.0,
        aot_artifact_ms: 11.0,
        aot_materialize_ms: 12.0,
        aot_compile_link_ms: 13.0,
        aot_register_link_ms: 14.0,
        iterations: 15,
        linear_solves: 16,
        jacobian_rebuilds: 17,
        refinements: 18,
        residual_calls: 19,
        jacobian_calls: 20,
    };
    let encoded = format!(
        "{ISOLATED_TUNING_METRICS_MARKER}\t{}",
        encode_isolated_cold_metrics(&metrics)
    );
    let decoded = decode_isolated_cold_metrics(&encoded);
    assert_eq!(decoded.initial_symbolic_jacobian_ms, 9.0);
    assert_eq!(decoded.aot_compile_link_ms, 13.0);
    assert_eq!(decoded.residual_actual_jobs, 4.0);
}

#[test]
#[ignore = "heavy combustion sparse AOT runtime tuning map across Rust/gcc/tcc/zig and chunking policies; cold wall-clock rows run in isolated child processes"]
fn aot_combustion_parallel_tuning_reports_runtime_table() {
    if let Ok(index) = std::env::var(ISOLATED_TUNING_CHILD_INDEX_ENV) {
        let index = index
            .parse::<usize>()
            .expect("isolated tuning child index should be an integer");
        let n_steps = std::env::var(ISOLATED_TUNING_CHILD_STEPS_ENV)
            .expect("isolated tuning child should receive n_steps")
            .parse::<usize>()
            .expect("isolated tuning n_steps should be an integer");
        run_isolated_tuning_child(index, n_steps);
        return;
    }
    run_combustion_tuning_scenario(1000, 4, "medium-grid-multi-toolchain")
        .expect("medium-grid multi-run AOT combustion tuning scenario should solve");
}

#[test]
#[ignore = "practical isolated cold combustion tuning table for Lambdify versus tcc chunking strategies"]
fn combustion_tcc_chunking_honest_wall_clock_table() {
    let n_steps = 1_000usize;
    let repetitions = 3usize;
    let variants = runtime_tuning_cold_variants();
    let selected = variants
        .iter()
        .enumerate()
        .filter(|(_, (label, _))| label == "lambdify-baseline" || label.starts_with("tcc/"))
        .map(|(index, (label, _))| (index, label.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        selected.len(),
        1 + 1 + runtime_tuning_parallel_cases().len(),
        "narrow cold table should contain Lambdify and every tcc chunking policy"
    );
    println!(
        "[AOT tcc practical cold map] n_steps={n_steps}, repetitions={repetitions}, cooldown_ms={}, cleanup_child_artifacts={}",
        cold_cooldown_ms(),
        clean_cold_artifacts_enabled()
    );

    let mut samples: Vec<(String, IsolatedColdObservation, f64)> = Vec::new();
    for repetition in 0..repetitions {
        let (baseline_index, baseline_label) = &selected[0];
        let baseline = solve_isolated_cold_tuning_variant(*baseline_index, n_steps);
        print_isolated_cold_raw_observation(repetition, baseline_label, &baseline);
        let baseline_solution = baseline.solution.clone();
        samples.push((baseline_label.clone(), baseline, 0.0));
        for (index, label) in selected.iter().skip(1) {
            let observation = solve_isolated_cold_tuning_variant(*index, n_steps);
            print_isolated_cold_raw_observation(repetition, label, &observation);
            let max_diff = baseline_solution
                .iter()
                .zip(observation.solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            assert!(
                max_diff < 1.0e-4,
                "{label}: isolated cold result differs from Lambdify by {max_diff}"
            );
            samples.push((label.clone(), observation, max_diff));
        }
    }

    println!();
    println!("[AOT tcc practical cold map] correctness and wall-clock table");
    println!(
        "{:<26} | {:<22} | {:<18} | {:<18} | {:<18}",
        "config", "honest_e2e_ms [min,max]", "max_diff", "symbolic_ms", "initial_sym_jac"
    );
    println!("{}", "-".repeat(114));
    for (_, label) in &selected {
        let rows = samples
            .iter()
            .filter(|(sample_label, _, _)| sample_label == label)
            .collect::<Vec<_>>();
        let total = runtime_tuning_aggregate(rows.iter().map(|(_, row, _)| row.elapsed_ms));
        let diff = runtime_tuning_aggregate(rows.iter().map(|(_, _, diff)| *diff));
        let symbolic =
            runtime_tuning_aggregate(rows.iter().map(|(_, row, _)| row.metrics.symbolic_ms));
        let sym_jac = runtime_tuning_aggregate(
            rows.iter()
                .map(|(_, row, _)| row.metrics.initial_symbolic_jacobian_ms),
        );
        println!(
            "{:<26} | {:<22} | {:<18} | {:<18} | {:<18}",
            label,
            fmt_tuning_agg(total),
            fmt_tuning_exp(diff),
            fmt_tuning_short(symbolic),
            fmt_tuning_short(sym_jac)
        );
    }

    println!();
    println!("[AOT tcc practical cold map] build and callback stages from the same child solves");
    println!(
        "{:<26} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12}",
        "config",
        "materialize_ms",
        "compile_link_ms",
        "residual_values",
        "jacobian_values",
        "res_jobs",
        "jac_jobs"
    );
    println!("{}", "-".repeat(132));
    for (_, label) in &selected {
        let rows = samples
            .iter()
            .filter(|(sample_label, _, _)| sample_label == label)
            .collect::<Vec<_>>();
        let materialize = runtime_tuning_aggregate(
            rows.iter()
                .map(|(_, row, _)| row.metrics.aot_materialize_ms),
        );
        let compile_link = runtime_tuning_aggregate(
            rows.iter()
                .map(|(_, row, _)| row.metrics.aot_compile_link_ms),
        );
        let residual = runtime_tuning_aggregate(
            rows.iter()
                .map(|(_, row, _)| row.metrics.cb_residual_values_ms),
        );
        let jacobian = runtime_tuning_aggregate(
            rows.iter()
                .map(|(_, row, _)| row.metrics.cb_jacobian_values_ms),
        );
        let res_jobs = runtime_tuning_aggregate(
            rows.iter()
                .map(|(_, row, _)| row.metrics.residual_actual_jobs),
        );
        let jac_jobs = runtime_tuning_aggregate(
            rows.iter()
                .map(|(_, row, _)| row.metrics.sparse_jacobian_actual_jobs),
        );
        println!(
            "{:<26} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12}",
            label,
            fmt_tuning_short(materialize),
            fmt_tuning_short(compile_link),
            fmt_tuning_short(residual),
            fmt_tuning_short(jacobian),
            fmt_tuning_short(res_jobs),
            fmt_tuning_short(jac_jobs)
        );
    }
}

/// Measures what fraction of total solve time is spent in residual+Jacobian eval
/// vs the linear solve and other Newton overhead.
///
/// If eval_fraction is small (< 0.3), parallelising the eval cannot give
/// more than 1/(1 - eval_fraction) speedup by Amdahl's law regardless of
/// how many cores are used.  That is the root cause of the 1.2x ceiling.
///
/// Run with: cargo test diagnose_eval_fraction -- --nocapture
#[test]
fn diagnose_eval_fraction_of_solve_time() {
    /*
     use std::hint::black_box;
    use crate::numerical::BVP_Damp::BVP_traits::Jac;
     let n_steps_list = [128usize, 256, 512, 1000];
     let iters = 50usize;

     println!(
         "\n=== Eval fraction of solve time (combustion, iters={iters}) ==="
     );
     println!(
         "{:<8} {:<10} {:<10} {:<12} {:<12} {:<14} {:<14}",
         "n_steps", "vars", "nnz",
         "full_ms", "eval_ms", "eval_frac", "amdahl_ceil"
     );

     for n_steps in n_steps_list {
         let sequential_config =
             crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                 .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
         let mut solver = make_combustion_solver(n_steps, sequential_config);
         let _guard = bootstrap_callable_aot_backend(
             &mut solver,
             &format!("eval-fraction-{n_steps}"),
         )
         .expect("bootstrap should succeed");

         // Full solve timing.
         let t_full = Instant::now();
         solver.try_solve().expect("solve should succeed");
         let full_ms = t_full.elapsed().as_secs_f64() * 1_000.0;

         // Isolate eval cost: call fun+jac directly iters times on the
         // converged solution so we measure hot-path eval, not convergence.
         let solution = solver
             .get_result()
             .expect("solution should be present after solve");
         let flat: Vec<f64> = solution.as_slice().to_vec();
         let col = faer::col::ColRef::from_slice(&flat).to_owned();

         let fun = &solver.fun;
         let jac: & Box<dyn Jac> = solver.jac.as_ref().expect("jac should be present");

         let t_eval = Instant::now();
         for _ in 0..iters {
             let r = fun.call(0.0, &col);
             black_box(r.len());
             let j = jac.call(0.0, &col);
             black_box(j.shape());
         }
         let eval_ms = t_eval.elapsed().as_secs_f64() * 1_000.0 / iters as f64;

         // Estimate Newton iterations from solve time and per-eval cost.
         let vars = 6 * n_steps;
         // nnz estimate: banded structure ~12 nonzeros per row for this problem
         let nnz_est = vars * 12;
         let eval_frac = eval_ms / (full_ms / iters as f64).max(eval_ms);
         // Amdahl ceiling: max speedup if eval is perfectly parallelised
         let amdahl_ceil = if eval_frac >= 1.0 {
             f64::INFINITY
         } else {
             1.0 / (1.0 - eval_frac)
         };

         println!(
             "{:<8} {:<10} {:<10} {:<12.3} {:<12.3} {:<14.3} {:<14.2}",
             n_steps, vars, nnz_est,
             full_ms, eval_ms, eval_frac, amdahl_ceil
         );

     }
     */
}
/// THE MOST IMPOTANT TEST FOR LINEAR SOLVERS COMPARISON: runs a full combustion-1000 eval+linear-solve and compares
#[test]
#[ignore = "heavy combustion-1000 linear-system story for sparse baseline vs consistent superblock solver"]
fn combustion_1000_linear_system_story_sparse_vs_banded_consistent() {
    #[derive(Debug)]
    struct Row {
        source: &'static str,
        matrix_backend: &'static str,
        variant: String,
        linear_solver: String,
        bootstrap_ms: f64,
        residual_diff: f64,
        jacobian_diff: f64,
        sparse_ms: f64,
        banded_ms: f64,
        layout: String,
        refinement: String,
        direct_rr: f64,
        final_rr: f64,
        solve_rr: f64,
        solve_diff: f64,
        relative_x_diff: f64,
        status: String,
    }

    let n_steps = 1000usize;
    let mut rows = Vec::new();
    println!(
        "[BVP Damp story] starting combustion-1000 linear-system comparison across sparse/banded backends"
    );

    let lambdify_sparse_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
            .with_matrix_backend_override(MatrixBackend::SparseCol);

    println!("[BVP Damp story] bootstrapping lambdify sparse baseline");
    let mut baseline_solver = make_combustion_solver(n_steps, lambdify_sparse_config);
    let baseline_bootstrap_begin = Instant::now();
    baseline_solver
        .try_eq_generate(None, None)
        .expect("lambdify sparse combustion-1000 bootstrap should succeed");
    let baseline_bootstrap_ms = baseline_bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;

    let baseline_args = flattened_initial_guess_state(
        &baseline_solver,
        baseline_solver.values.len() * n_steps,
        "combustion-1000-lambdify-sparse",
    );
    let (baseline_residual, baseline_dense_matrix, baseline_sparse_matrix, _) =
        eval_solver_callback_state(
            &mut baseline_solver,
            &baseline_args,
            MatrixBackend::SparseCol,
            "combustion-1000-lambdify-sparse",
        );
    let baseline_sparse_matrix =
        baseline_sparse_matrix.expect("sparse baseline should produce SparseColMat");
    let baseline_rhs: Vec<f64> = baseline_residual.iter().map(|value| -*value).collect();

    let baseline_sparse_begin = Instant::now();
    let baseline_sparse_solution =
        solve_sparse_lu_for_rhs(&baseline_sparse_matrix, baseline_rhs.as_slice());
    let baseline_sparse_ms = baseline_sparse_begin.elapsed().as_secs_f64() * 1_000.0;
    rows.push(Row {
        source: "Lambdify",
        matrix_backend: "Sparse",
        variant: "ExprLegacy".to_string(),
        linear_solver: "faer_sparse_lu".to_string(),
        bootstrap_ms: baseline_bootstrap_ms,
        residual_diff: 0.0,
        jacobian_diff: 0.0,
        sparse_ms: baseline_sparse_ms,
        banded_ms: 0.0,
        layout: "-".to_string(),
        refinement: "-".to_string(),
        direct_rr: 0.0,
        final_rr: 0.0,
        solve_rr: relative_dense_residual(
            &baseline_dense_matrix,
            &baseline_sparse_solution,
            baseline_rhs.as_slice(),
        ),
        solve_diff: 0.0,
        relative_x_diff: 0.0,
        status: "ok".to_string(),
    });

    let baseline_dense_banded_assembly = banded_assembly_from_dense_matrix(&baseline_dense_matrix);
    for solver_choice in BandedStorySolver::variants() {
        let banded_begin = Instant::now();
        let metrics = solve_banded_story_for_rhs(
            &baseline_dense_banded_assembly,
            n_steps,
            baseline_rhs.as_slice(),
            solver_choice,
        );
        let banded_ms = banded_begin.elapsed().as_secs_f64() * 1_000.0;
        let solve_diff = metrics
            .solution
            .as_ref()
            .map(|solution| {
                solution
                    .iter()
                    .zip(baseline_sparse_solution.iter())
                    .map(|(lhs, rhs)| (lhs - rhs).abs())
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(f64::NAN);
        let rel_x = metrics
            .solution
            .as_ref()
            .map(|solution| relative_x_diff(solution, &baseline_sparse_solution))
            .unwrap_or(f64::NAN);
        let solve_rr = metrics
            .solution
            .as_ref()
            .map(|solution| {
                relative_dense_residual(&baseline_dense_matrix, solution, baseline_rhs.as_slice())
            })
            .unwrap_or(f64::NAN);
        rows.push(Row {
            source: "Derived",
            matrix_backend: "Sparse->Banded",
            variant: "DenseBaseline".to_string(),
            linear_solver: metrics.linear_solver,
            bootstrap_ms: 0.0,
            residual_diff: 0.0,
            jacobian_diff: 0.0,
            sparse_ms: 0.0,
            banded_ms,
            layout: metrics.layout,
            refinement: metrics
                .report
                .as_ref()
                .map(|report| {
                    format!(
                        "{}/{}{}",
                        report.accepted_steps,
                        report.requested_steps,
                        if report.refinement_attempted {
                            ""
                        } else {
                            " skipped"
                        }
                    )
                })
                .unwrap_or_else(|| "-".to_string()),
            direct_rr: metrics
                .report
                .as_ref()
                .map(|report| report.direct_relative_residual)
                .unwrap_or(f64::NAN),
            final_rr: metrics
                .report
                .as_ref()
                .map(|report| report.final_relative_residual)
                .unwrap_or(f64::NAN),
            solve_rr,
            solve_diff,
            relative_x_diff: rel_x,
            status: if metrics.status == "ok" {
                "diag".to_string()
            } else {
                metrics.status
            },
        });
    }

    let lambdify_banded_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
            .with_matrix_backend_override(MatrixBackend::Banded);
    println!("[BVP Damp story] bootstrapping lambdify banded path");
    let mut lambdify_banded_solver = make_combustion_solver(n_steps, lambdify_banded_config);
    let lambdify_banded_bootstrap_begin = Instant::now();
    lambdify_banded_solver
        .try_eq_generate(None, None)
        .expect("lambdify banded combustion-1000 bootstrap should succeed");
    let lambdify_banded_bootstrap_ms =
        lambdify_banded_bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
    let lambdify_banded_args = flattened_initial_guess_state(
        &lambdify_banded_solver,
        lambdify_banded_solver.values.len() * n_steps,
        "combustion-1000-lambdify-banded",
    );
    let (lambdify_banded_residual, lambdify_banded_dense, _, lambdify_banded_matrix) =
        eval_solver_callback_state(
            &mut lambdify_banded_solver,
            &lambdify_banded_args,
            MatrixBackend::Banded,
            "combustion-1000-lambdify-banded",
        );
    let lambdify_banded_matrix =
        lambdify_banded_matrix.expect("banded lambdify path should produce BandedMatrixType");
    let lambdify_banded_rhs: Vec<f64> = lambdify_banded_residual
        .iter()
        .map(|value| -*value)
        .collect();
    for solver_choice in BandedStorySolver::variants() {
        let banded_begin = Instant::now();
        let metrics = solve_banded_story_for_rhs(
            &lambdify_banded_matrix.assembly,
            n_steps,
            lambdify_banded_rhs.as_slice(),
            solver_choice,
        );
        let banded_ms = banded_begin.elapsed().as_secs_f64() * 1_000.0;
        let solve_diff = metrics
            .solution
            .as_ref()
            .map(|solution| {
                solution
                    .iter()
                    .zip(baseline_sparse_solution.iter())
                    .map(|(lhs, rhs)| (lhs - rhs).abs())
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(f64::NAN);
        let rel_x = metrics
            .solution
            .as_ref()
            .map(|solution| relative_x_diff(solution, &baseline_sparse_solution))
            .unwrap_or(f64::NAN);
        let solve_rr = metrics
            .solution
            .as_ref()
            .map(|solution| {
                relative_dense_residual(
                    &lambdify_banded_dense,
                    solution,
                    lambdify_banded_rhs.as_slice(),
                )
            })
            .unwrap_or(f64::NAN);
        rows.push(Row {
            source: "Lambdify",
            matrix_backend: "Banded",
            variant: "ExprLegacy".to_string(),
            linear_solver: metrics.linear_solver,
            bootstrap_ms: lambdify_banded_bootstrap_ms,
            residual_diff: max_abs_vector_diff(&lambdify_banded_residual, &baseline_residual),
            jacobian_diff: max_abs_matrix_diff(&lambdify_banded_dense, &baseline_dense_matrix),
            sparse_ms: 0.0,
            banded_ms,
            layout: metrics.layout,
            refinement: metrics
                .report
                .as_ref()
                .map(|report| {
                    format!(
                        "{}/{}{}",
                        report.accepted_steps,
                        report.requested_steps,
                        if report.refinement_attempted {
                            ""
                        } else {
                            " skipped"
                        }
                    )
                })
                .unwrap_or_else(|| "-".to_string()),
            direct_rr: metrics
                .report
                .as_ref()
                .map(|report| report.direct_relative_residual)
                .unwrap_or(f64::NAN),
            final_rr: metrics
                .report
                .as_ref()
                .map(|report| report.final_relative_residual)
                .unwrap_or(f64::NAN),
            solve_rr,
            solve_diff,
            relative_x_diff: rel_x,
            status: metrics.status,
        });
    }

    let compiled_sparse_variants = [
            (
                "C-gcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("gcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
            ),
            (
                "C-tcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("tcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
            ),
            (
                "Zig",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::Zig)
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
            ),
        ];

    for (variant, config) in compiled_sparse_variants {
        println!("[BVP Damp story] bootstrapping compiled sparse variant `{variant}`");
        let mut solver = make_combustion_solver(n_steps, config);
        let bootstrap_begin = Instant::now();
        solver
            .try_eq_generate(None, None)
            .expect("compiled sparse combustion-1000 bootstrap should succeed");
        let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;

        let args = flattened_initial_guess_state(
            &solver,
            solver.values.len() * n_steps,
            &format!("combustion-1000-sparse-{variant}"),
        );
        let (residual, dense_matrix, sparse_matrix, _) = eval_solver_callback_state(
            &mut solver,
            &args,
            MatrixBackend::SparseCol,
            &format!("combustion-1000-sparse-{variant}"),
        );
        let sparse_matrix =
            sparse_matrix.expect("compiled sparse path should produce SparseColMat");
        let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();

        let sparse_begin = Instant::now();
        let sparse_solution = solve_sparse_lu_for_rhs(&sparse_matrix, rhs.as_slice());
        let sparse_ms = sparse_begin.elapsed().as_secs_f64() * 1_000.0;

        rows.push(Row {
            source: "Compiled",
            matrix_backend: "Sparse",
            variant: variant.to_string(),
            linear_solver: "faer_sparse_lu".to_string(),
            bootstrap_ms,
            residual_diff: max_abs_vector_diff(&residual, &baseline_residual),
            jacobian_diff: max_abs_matrix_diff(&dense_matrix, &baseline_dense_matrix),
            sparse_ms,
            banded_ms: 0.0,
            layout: "-".to_string(),
            refinement: "-".to_string(),
            direct_rr: 0.0,
            final_rr: 0.0,
            solve_rr: relative_dense_residual(&dense_matrix, &sparse_solution, rhs.as_slice()),
            solve_diff: sparse_solution
                .iter()
                .zip(baseline_sparse_solution.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max),
            relative_x_diff: relative_x_diff(&sparse_solution, &baseline_sparse_solution),
            status: "ok".to_string(),
        });
    }

    let compiled_banded_variants = [
            (
                "C-gcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("gcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::Banded),
            ),
            (
                "C-tcc",
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                    .with_aot_codegen_backend(AotCodegenBackend::C)
                    .with_aot_c_compiler("tcc")
                    .with_aot_compile_dev_fastest()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_matrix_backend_override(MatrixBackend::Banded),
            ),
        ];

    for (variant, config) in compiled_banded_variants {
        println!("[BVP Damp story] bootstrapping compiled banded variant `{variant}`");
        let mut solver = make_combustion_solver(n_steps, config);
        let bootstrap_begin = Instant::now();
        solver
            .try_eq_generate(None, None)
            .expect("compiled banded combustion-1000 bootstrap should succeed");
        let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;

        let args = flattened_initial_guess_state(
            &solver,
            solver.values.len() * n_steps,
            &format!("combustion-1000-banded-{variant}"),
        );
        let (residual, dense_matrix, _, banded_matrix) = eval_solver_callback_state(
            &mut solver,
            &args,
            MatrixBackend::Banded,
            &format!("combustion-1000-banded-{variant}"),
        );
        let banded_matrix =
            banded_matrix.expect("compiled banded path should produce BandedMatrixType");
        let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();

        for solver_choice in BandedStorySolver::variants() {
            let banded_begin = Instant::now();
            let metrics = solve_banded_story_for_rhs(
                &banded_matrix.assembly,
                n_steps,
                rhs.as_slice(),
                solver_choice,
            );
            let banded_ms = banded_begin.elapsed().as_secs_f64() * 1_000.0;
            let solve_diff = metrics
                .solution
                .as_ref()
                .map(|solution| {
                    solution
                        .iter()
                        .zip(baseline_sparse_solution.iter())
                        .map(|(lhs, rhs)| (lhs - rhs).abs())
                        .fold(0.0_f64, f64::max)
                })
                .unwrap_or(f64::NAN);
            let rel_x = metrics
                .solution
                .as_ref()
                .map(|solution| relative_x_diff(solution, &baseline_sparse_solution))
                .unwrap_or(f64::NAN);
            let solve_rr = metrics
                .solution
                .as_ref()
                .map(|solution| relative_dense_residual(&dense_matrix, solution, rhs.as_slice()))
                .unwrap_or(f64::NAN);

            rows.push(Row {
                source: "Compiled",
                matrix_backend: "Banded",
                variant: variant.to_string(),
                linear_solver: metrics.linear_solver,
                bootstrap_ms,
                residual_diff: max_abs_vector_diff(&residual, &baseline_residual),
                jacobian_diff: max_abs_matrix_diff(&dense_matrix, &baseline_dense_matrix),
                sparse_ms: 0.0,
                banded_ms,
                layout: metrics.layout,
                refinement: metrics
                    .report
                    .as_ref()
                    .map(|report| {
                        format!(
                            "{}/{}{}",
                            report.accepted_steps,
                            report.requested_steps,
                            if report.refinement_attempted {
                                ""
                            } else {
                                " skipped"
                            }
                        )
                    })
                    .unwrap_or_else(|| "-".to_string()),
                direct_rr: metrics
                    .report
                    .as_ref()
                    .map(|report| report.direct_relative_residual)
                    .unwrap_or(f64::NAN),
                final_rr: metrics
                    .report
                    .as_ref()
                    .map(|report| report.final_relative_residual)
                    .unwrap_or(f64::NAN),
                solve_rr,
                solve_diff,
                relative_x_diff: rel_x,
                status: metrics.status,
            });
        }
    }

    println!("[BVP Damp linear story] combustion-1000 sparse baseline vs banded solver variants");
    println!(
        "{:<10} | {:<13} | {:<10} | {:<34} | {:>12} | {:>12} | {:>12} | {:>10} | {:>10} | {:<8} | {:<13} | {:>10} | {:>10} | {:>10} | {:>12} | {:>12} | {:<24}",
        "source",
        "matrix",
        "variant",
        "linear_solver",
        "bootstrap_ms",
        "res_diff",
        "jac_diff",
        "sparse_ms",
        "banded_ms",
        "layout",
        "refinement",
        "direct_rr",
        "final_rr",
        "solve_rr",
        "solve_diff",
        "rel_x_diff",
        "status"
    );
    println!("{}", "-".repeat(262));
    for row in &rows {
        println!(
            "{:<10} | {:<13} | {:<10} | {:<34} | {:>12.3} | {:>12.3e} | {:>12.3e} | {:>10.3} | {:>10.3} | {:<8} | {:<13} | {:>10} | {:>10} | {:>10} | {:>12.3e} | {:>12.3e} | {:<24}",
            row.source,
            row.matrix_backend,
            row.variant,
            row.linear_solver,
            row.bootstrap_ms,
            row.residual_diff,
            row.jacobian_diff,
            row.sparse_ms,
            row.banded_ms,
            row.layout,
            row.refinement,
            fmt_metric(row.direct_rr),
            fmt_metric(row.final_rr),
            fmt_metric(row.solve_rr),
            row.solve_diff,
            row.relative_x_diff,
            row.status
        );
    }

    for row in &rows {
        assert!(
            row.residual_diff < 1e-6,
            "{} {} {} residual diff too large: {}",
            row.source,
            row.matrix_backend,
            row.variant,
            row.residual_diff
        );
        assert!(
            row.jacobian_diff < 1e-6,
            "{} {} {} jacobian diff too large: {}",
            row.source,
            row.matrix_backend,
            row.variant,
            row.jacobian_diff
        );
        if row.matrix_backend == "Sparse" {
            assert!(
                row.solve_diff < 1e-6,
                "{} {} {} sparse solve drift too large: {}",
                row.source,
                row.matrix_backend,
                row.variant,
                row.solve_diff
            );
        } else if row.matrix_backend.contains("Banded") && row.status == "ok" {
            assert!(
                row.solve_diff.is_finite(),
                "{} {} {} {} should report finite banded solve_diff",
                row.source,
                row.matrix_backend,
                row.variant,
                row.linear_solver
            );
        }
    }
}
