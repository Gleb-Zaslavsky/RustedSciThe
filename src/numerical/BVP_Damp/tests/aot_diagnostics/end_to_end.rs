#[test]
#[ignore = "heavy combustion-1000 end-to-end solve through banded lambdify/AOT backends using lapack-style banded LU + refinement"]
fn combustion_1000_end_to_end_banded_lapack_refine_statistics() {
    aot_test_report!(combustion_1000_end_to_end_banded_lapack_refine_statistics);
    #[derive(Debug)]
    struct EndToEndRow {
        source: &'static str,
        variant: &'static str,
        bootstrap_ms: f64,
        solve_ms: f64,
        total_ms: f64,
        max_abs_solution: f64,
        solve_diff: f64,
        rel_x_diff: f64,
        iterations: usize,
        linear_solves: usize,
        jac_rebuilds: usize,
        linear_timer: String,
        jac_timer: String,
        fun_timer: String,
        symbolic_prepare_ms: f64,
        fixture_generation_ms: f64,
        compile_ms: f64,
        link_ms: f64,
        residual_calls: f64,
        jacobian_calls: f64,
        status: String,
    }

    fn repetitions() -> usize {
        std::env::var("BVP_BANDED_LAPACK_REPETITIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(5)
    }

    fn lapack_refine_cooldown_ms() -> u64 {
        std::env::var("BVP_BANDED_LAPACK_COOLDOWN_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(5_000)
    }

    fn solution_max_abs(solution: &DMatrix<f64>) -> f64 {
        solution
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max)
    }

    fn solution_linf_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0_f64, f64::max)
    }

    fn solution_rel_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        solution_linf_diff(lhs, rhs) / solution_max_abs(rhs).max(1.0)
    }

    fn run_variant(
        n_steps: usize,
        source: &'static str,
        variant: &'static str,
        config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
    ) -> (EndToEndRow, Option<DMatrix<f64>>) {
        fn panic_status(prefix: &str, payload: Box<dyn std::any::Any + Send>) -> String {
            let message = if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_string()
            } else {
                "non-string payload".to_string()
            };
            format!("{prefix}({message})")
        }

        let total_begin = Instant::now();
        let mut solver = make_combustion_solver(n_steps, config);

        let bootstrap_begin = Instant::now();
        // AOT failures can occur while callbacks are materialized, before
        // the solver can return its typed Result. Keep the matrix row and
        // report instead of aborting all remaining toolchains.
        let bootstrap_status =
            catch_unwind(AssertUnwindSafe(|| solver.try_eq_generate(None, None)));
        let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
        match bootstrap_status {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return (
                    EndToEndRow {
                        source,
                        variant,
                        bootstrap_ms,
                        solve_ms: 0.0,
                        total_ms: total_begin.elapsed().as_secs_f64() * 1_000.0,
                        max_abs_solution: f64::NAN,
                        solve_diff: f64::NAN,
                        rel_x_diff: f64::NAN,
                        iterations: 0,
                        linear_solves: 0,
                        jac_rebuilds: 0,
                        linear_timer: "-".to_string(),
                        jac_timer: "-".to_string(),
                        fun_timer: "-".to_string(),
                        symbolic_prepare_ms: 0.0,
                        fixture_generation_ms: 0.0,
                        compile_ms: 0.0,
                        link_ms: 0.0,
                        residual_calls: 0.0,
                        jacobian_calls: 0.0,
                        status: format!("bootstrap_failed({err:?})"),
                    },
                    None,
                );
            }
            Err(panic_payload) => {
                return (
                    EndToEndRow {
                        source,
                        variant,
                        bootstrap_ms,
                        solve_ms: 0.0,
                        total_ms: total_begin.elapsed().as_secs_f64() * 1_000.0,
                        max_abs_solution: f64::NAN,
                        solve_diff: f64::NAN,
                        rel_x_diff: f64::NAN,
                        iterations: 0,
                        linear_solves: 0,
                        jac_rebuilds: 0,
                        linear_timer: "-".to_string(),
                        jac_timer: "-".to_string(),
                        fun_timer: "-".to_string(),
                        symbolic_prepare_ms: 0.0,
                        fixture_generation_ms: 0.0,
                        compile_ms: 0.0,
                        link_ms: 0.0,
                        residual_calls: 0.0,
                        jacobian_calls: 0.0,
                        status: panic_status("bootstrap_panicked", panic_payload),
                    },
                    None,
                );
            }
        }

        let solve_begin = Instant::now();
        let solve_status = catch_unwind(AssertUnwindSafe(|| solver.try_solve()));
        let solve_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
        let statistics = solver.get_statistics();
        let symbolic_prepare_ms =
            stats_diagnostic_ms(&statistics, "generated.aot.typed.symbolic_preparation_ms");
        let fixture_generation_ms =
            stats_diagnostic_ms(&statistics, "generated.aot.typed.fixture_generation_ms");
        let compile_ms = stats_diagnostic_ms(&statistics, "generated.aot.typed.compile_ms").max(
            stats_diagnostic_ms(&statistics, "generated.aot.compile_link_ms"),
        );
        let link_ms = stats_diagnostic_ms(&statistics, "generated.aot.typed.link_ms").max(
            stats_diagnostic_ms(&statistics, "generated.aot.register_link_ms"),
        );
        let residual_calls =
            stats_diagnostic_usize(&statistics, "generated.aot.typed.residual_calls");
        let jacobian_calls =
            stats_diagnostic_usize(&statistics, "generated.aot.typed.jacobian_calls");

        match solve_status {
            Ok(Ok(_)) => match solver.get_result() {
                Some(solution) => {
                    let max_abs_solution = solution_max_abs(&solution);
                    (
                        EndToEndRow {
                            source,
                            variant,
                            bootstrap_ms,
                            solve_ms,
                            total_ms,
                            max_abs_solution,
                            solve_diff: 0.0,
                            rel_x_diff: 0.0,
                            iterations: stats_count(&statistics, "number of iterations"),
                            linear_solves: stats_count(
                                &statistics,
                                "number of solving linear systems",
                            ),
                            jac_rebuilds: stats_count(
                                &statistics,
                                "number of jacobians recalculations",
                            ),
                            linear_timer: stats_timer(&statistics, "Linear System"),
                            jac_timer: stats_timer(&statistics, "Jacobian"),
                            fun_timer: stats_timer(&statistics, "Function"),
                            symbolic_prepare_ms,
                            fixture_generation_ms,
                            compile_ms,
                            link_ms,
                            residual_calls,
                            jacobian_calls,
                            status: "ok".to_string(),
                        },
                        Some(solution),
                    )
                }
                None => (
                    EndToEndRow {
                        source,
                        variant,
                        bootstrap_ms,
                        solve_ms,
                        total_ms,
                        max_abs_solution: f64::NAN,
                        solve_diff: f64::NAN,
                        rel_x_diff: f64::NAN,
                        iterations: stats_count(&statistics, "number of iterations"),
                        linear_solves: stats_count(&statistics, "number of solving linear systems"),
                        jac_rebuilds: stats_count(
                            &statistics,
                            "number of jacobians recalculations",
                        ),
                        linear_timer: stats_timer(&statistics, "Linear System"),
                        jac_timer: stats_timer(&statistics, "Jacobian"),
                        fun_timer: stats_timer(&statistics, "Function"),
                        symbolic_prepare_ms,
                        fixture_generation_ms,
                        compile_ms,
                        link_ms,
                        residual_calls,
                        jacobian_calls,
                        status: "no_result".to_string(),
                    },
                    None,
                ),
            },
            Ok(Err(err)) => (
                EndToEndRow {
                    source,
                    variant,
                    bootstrap_ms,
                    solve_ms,
                    total_ms,
                    max_abs_solution: f64::NAN,
                    solve_diff: f64::NAN,
                    rel_x_diff: f64::NAN,
                    iterations: stats_count(&statistics, "number of iterations"),
                    linear_solves: stats_count(&statistics, "number of solving linear systems"),
                    jac_rebuilds: stats_count(&statistics, "number of jacobians recalculations"),
                    linear_timer: stats_timer(&statistics, "Linear System"),
                    jac_timer: stats_timer(&statistics, "Jacobian"),
                    fun_timer: stats_timer(&statistics, "Function"),
                    symbolic_prepare_ms,
                    fixture_generation_ms,
                    compile_ms,
                    link_ms,
                    residual_calls,
                    jacobian_calls,
                    status: format!("solve_failed({err:?})"),
                },
                None,
            ),
            Err(panic_payload) => {
                let status = panic_status("solve_panicked", panic_payload);
                (
                    EndToEndRow {
                        source,
                        variant,
                        bootstrap_ms,
                        solve_ms,
                        total_ms,
                        max_abs_solution: f64::NAN,
                        solve_diff: f64::NAN,
                        rel_x_diff: f64::NAN,
                        iterations: stats_count(&statistics, "number of iterations"),
                        linear_solves: stats_count(&statistics, "number of solving linear systems"),
                        jac_rebuilds: stats_count(
                            &statistics,
                            "number of jacobians recalculations",
                        ),
                        linear_timer: stats_timer(&statistics, "Linear System"),
                        jac_timer: stats_timer(&statistics, "Jacobian"),
                        fun_timer: stats_timer(&statistics, "Function"),
                        symbolic_prepare_ms,
                        fixture_generation_ms,
                        compile_ms,
                        link_ms,
                        residual_calls,
                        jacobian_calls,
                        status,
                    },
                    None,
                )
            }
        }
    }

    let n_steps = 1000usize;

    // Canonical Lambdify comparison: AtomView preparation plus native
    // faithful banded LU. ExprLegacy remains an explicit compatibility
    // oracle in the dedicated frontend comparison stories.
    let lambdify_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_lambdify_defaults();

    let gcc_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_telemetry_mode(BvpAotTelemetryMode::Detailed)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });

    let tcc_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_telemetry_mode(BvpAotTelemetryMode::Detailed)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });

    let zig_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_telemetry_mode(BvpAotTelemetryMode::Detailed)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });

    let runs = [
        ("Lambdify", "AtomView", lambdify_config),
        ("Compiled", "C-gcc", gcc_config),
        ("Compiled", "C-tcc", tcc_config),
        ("Compiled", "Zig", zig_config),
    ];

    let repetitions = repetitions();
    let cooldown_ms = lapack_refine_cooldown_ms();
    let mut rows: Vec<(usize, EndToEndRow)> = Vec::new();
    let mut baseline_count = 0usize;

    for rep in 1..=repetitions {
        let mut baseline_solution: Option<DMatrix<f64>> = None;

        for (source, variant, config) in runs.clone() {
            if !rows.is_empty() && cooldown_ms > 0 {
                thread::sleep(Duration::from_millis(cooldown_ms));
            }
            let (mut row, solution) = run_variant(n_steps, source, variant, config);
            if source == "Lambdify" && variant == "AtomView" {
                baseline_solution = solution.clone();
                if baseline_solution.is_some() {
                    baseline_count += 1;
                }
            }
            if let (Some(solution), Some(baseline)) =
                (solution.as_ref(), baseline_solution.as_ref())
            {
                row.solve_diff = solution_linf_diff(solution, baseline);
                row.rel_x_diff = solution_rel_diff(solution, baseline);
            }
            rows.push((rep, row));
        }
    }

    println!(
        "[BVP Damp end-to-end] combustion-1000 full solve with banded backends using lapack_style_banded_lu; repetitions={repetitions}, cooldown_ms={cooldown_ms}"
    );
    if baseline_count < repetitions {
        println!(
            "[BVP Damp end-to-end] Lambdify baseline converged in {baseline_count}/{repetitions} repetitions; solve_diff and rel_x_diff may be unavailable for failed repetitions"
        );
    }
    println!(
        "{:>3} | {:<10} | {:<10} | {:>12} | {:>10} | {:>10} | {:>12} | {:>12} | {:>12} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<8}",
        "rep",
        "source",
        "variant",
        "bootstrap_ms",
        "solve_ms",
        "total_ms",
        "max_abs_sol",
        "solve_diff",
        "rel_x_diff",
        "iters",
        "linsys",
        "jac_re",
        "linear_timer",
        "jac_timer",
        "fun_timer",
        "status"
    );
    println!("{}", "-".repeat(228));
    for (rep, row) in &rows {
        println!(
            "{:>3} | {:<10} | {:<10} | {:>12.3} | {:>10.3} | {:>10.3} | {:>12.6e} | {:>12} | {:>12} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<8}",
            rep,
            row.source,
            row.variant,
            row.bootstrap_ms,
            row.solve_ms,
            row.total_ms,
            row.max_abs_solution,
            fmt_metric(row.solve_diff),
            fmt_metric(row.rel_x_diff),
            row.iterations,
            row.linear_solves,
            row.jac_rebuilds,
            row.linear_timer,
            row.jac_timer,
            row.fun_timer,
            row.status
        );
    }

    println!();
    println!(
        "[BVP Damp AOT lifecycle] combustion-1000 cold stages; zero means the route did not perform that AOT stage"
    );
    println!(
        "source     | variant    | symbolic_prepare_ms | fixture_generation_ms | compile_ms | link_ms | residual_calls | jacobian_calls | status"
    );
    println!("{}", "-".repeat(132));
    for row in rows.iter().filter(|(_, row)| row.status == "ok") {
        println!(
            "{:<10} | {:<10} | {:>19.3} | {:>21.3} | {:>10.3} | {:>7.3} | {:>14} | {:>14} | {}",
            row.1.source,
            row.1.variant,
            row.1.symbolic_prepare_ms,
            row.1.fixture_generation_ms,
            row.1.compile_ms,
            row.1.link_ms,
            row.1.residual_calls,
            row.1.jacobian_calls,
            row.1.status,
        );
    }

    println!();
    println!(
        "[BVP Damp end-to-end] combustion-1000 banded lapack-style summary; all time columns are milliseconds"
    );
    println!(
        "{:<10} | {:<10} | {:>7} | {:<36} | {:<36} | {:<36} | {:<18} | {:<18} | {:<18} | {:<18}",
        "source",
        "variant",
        "ok/runs",
        "total_ms mean+/-std [min,max]",
        "bootstrap_ms mean+/-std [min,max]",
        "solve_ms mean+/-std [min,max]",
        "linear_solves",
        "jac_rebuilds",
        "solve_diff",
        "rel_x_diff"
    );
    println!("{}", "-".repeat(220));
    for (source, variant, _) in runs {
        let samples = rows
            .iter()
            .filter_map(|(_, row)| {
                (row.source == source && row.variant == variant && row.status == "ok")
                    .then_some(row)
            })
            .collect::<Vec<_>>();
        let ok = samples.len();
        println!(
            "{:<10} | {:<10} | {:>2}/{:<4} | {:<36} | {:<36} | {:<36} | {:<18} | {:<18} | {:<18} | {:<18}",
            source,
            variant,
            ok,
            repetitions,
            fmt_tuning_agg(runtime_tuning_aggregate(
                samples.iter().map(|row| row.total_ms)
            )),
            fmt_tuning_agg(runtime_tuning_aggregate(
                samples.iter().map(|row| row.bootstrap_ms)
            )),
            fmt_tuning_agg(runtime_tuning_aggregate(
                samples.iter().map(|row| row.solve_ms)
            )),
            fmt_tuning_short(runtime_tuning_aggregate(
                samples.iter().map(|row| row.linear_solves as f64)
            )),
            fmt_tuning_short(runtime_tuning_aggregate(
                samples.iter().map(|row| row.jac_rebuilds as f64)
            )),
            fmt_tuning_exp(runtime_tuning_aggregate(
                samples.iter().map(|row| row.solve_diff)
            )),
            fmt_tuning_exp(runtime_tuning_aggregate(
                samples.iter().map(|row| row.rel_x_diff)
            )),
        );
    }

    assert!(
        !rows.is_empty(),
        "combustion-1000 end-to-end banded diagnostic should produce at least one row"
    );
    assert_eq!(
        baseline_count, repetitions,
        "lambdify banded combustion-1000 baseline should converge in every repetition"
    );
    for (rep, row) in &rows {
        assert_eq!(
            row.status, "ok",
            "rep {rep}: {} {} end-to-end banded row should solve successfully, got {}",
            row.source, row.variant, row.status
        );
        assert!(
            row.max_abs_solution.is_finite(),
            "rep {rep}: {} {} end-to-end banded result should stay finite",
            row.source,
            row.variant
        );
    }
}

#[test]
#[ignore = "focused Zig banded AOT bootstrap diagnostic for combustion-1000"]
fn combustion_1000_compiled_banded_zig_bootstrap_smoke() {
    aot_test_report!(combustion_1000_compiled_banded_zig_bootstrap_smoke);
    let n_steps = 1000usize;
    println!("[BVP Damp Zig banded] starting focused combustion-1000 Zig AOT bootstrap");

    let config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_aot_codegen_backend(AotCodegenBackend::Zig)
                .with_aot_compile_dev_fastest()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_matrix_backend_override(MatrixBackend::Banded);

    let mut solver = make_combustion_solver(n_steps, config);
    solver
        .try_eq_generate(None, None)
        .expect("compiled banded Zig combustion-1000 bootstrap should succeed");

    let args = flattened_initial_guess_state(
        &solver,
        solver.values.len() * n_steps,
        "combustion-1000-banded-zig-smoke",
    );
    let (residual, dense_matrix, _, banded_matrix) = eval_solver_callback_state(
        &mut solver,
        &args,
        MatrixBackend::Banded,
        "combustion-1000-banded-zig-smoke",
    );
    let banded_matrix =
        banded_matrix.expect("compiled banded Zig path should produce BandedMatrixType");
    let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();
    let block_metrics = solve_banded_story_for_rhs(
        &banded_matrix.assembly,
        n_steps,
        rhs.as_slice(),
        BandedStorySolver::ConsistentSuperblock {
            nodes_per_superblock: 2,
            refinement_steps: 1,
        },
    );
    let default_metrics = solve_banded_story_for_rhs(
        &banded_matrix.assembly,
        n_steps,
        rhs.as_slice(),
        BandedStorySolver::LapackStyle {
            refinement_steps: 1,
        },
    );
    let default_rr = default_metrics
        .solution
        .as_ref()
        .map(|solution| relative_dense_residual(&dense_matrix, solution, rhs.as_slice()))
        .unwrap_or(f64::NAN);

    println!(
        "[BVP Damp Zig banded default] residual_len={}, matrix={}x{}, solver={}, status={}, solve_rr={:.3e}",
        residual.len(),
        dense_matrix.nrows(),
        dense_matrix.ncols(),
        default_metrics.linear_solver,
        default_metrics.status,
        default_rr
    );
    println!(
        "[BVP Damp Zig banded structured diagnostic] solver={}, layout={}, refinement={}, direct_rr={:.3e}, final_rr={:.3e}, max|x|={:.3e}, status={}",
        block_metrics.linear_solver,
        block_metrics.layout,
        block_metrics
            .report
            .as_ref()
            .map(|report| format!("{}/{}", report.accepted_steps, report.requested_steps))
            .unwrap_or_else(|| "-".to_string()),
        block_metrics
            .report
            .as_ref()
            .map(|report| report.direct_relative_residual)
            .unwrap_or(f64::NAN),
        block_metrics
            .report
            .as_ref()
            .map(|report| report.final_relative_residual)
            .unwrap_or(f64::NAN),
        block_metrics
            .solution
            .as_ref()
            .map(|solution| solution
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())))
            .unwrap_or(f64::NAN),
        block_metrics.status
    );

    assert!(
        default_metrics
            .solution
            .as_ref()
            .map(|solution| solution.iter().all(|value| value.is_finite()))
            .unwrap_or(false),
        "compiled banded Zig default solve should remain finite"
    );
    assert_eq!(
        default_metrics.status, "ok",
        "the Zig-generated Banded default must use the faithful LAPACK route"
    );
    assert_eq!(
        block_metrics.status, "diag",
        "the structured block-tridiagonal diagnostic must not regress to green while its residual is unacceptable"
    );
    assert!(
        default_rr.is_finite() && default_rr <= 1.0e-8,
        "the faithful LAPACK Banded default should have a small residual, got {default_rr:.6e}"
    );
}

#[test]
#[ignore = "focused combustion-1000 diagnostic for lapack-style banded storage/factor/solve"]
fn diagnose_combustion_1000_lapack_style_banded_path() {
    aot_test_report!(diagnose_combustion_1000_lapack_style_banded_path);
    let n_steps = 1000usize;
    let config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
            .with_matrix_backend_override(MatrixBackend::Banded);

    println!("[Lapack banded diagnostic] bootstrapping combustion-1000 lambdify banded");
    let mut solver = make_combustion_solver(n_steps, config);
    solver
        .try_eq_generate(None, None)
        .expect("combustion-1000 lambdify banded bootstrap should succeed");

    let args = flattened_initial_guess_state(
        &solver,
        solver.values.len() * n_steps,
        "combustion-1000-lapack-style-diagnostic",
    );

    let (residual, dense_matrix, _, banded_matrix) = eval_solver_callback_state(
        &mut solver,
        &args,
        MatrixBackend::Banded,
        "combustion-1000-lapack-style-diagnostic",
    );
    let banded_matrix =
        banded_matrix.expect("combustion-1000 banded path should produce BandedMatrixType");
    let assembly = &banded_matrix.assembly;
    let compact = assembly
        .to_banded()
        .expect("banded assembly should convert to compact storage");

    let dense_from_compact = dense_from_compact_banded(&compact);
    let compact_diff = max_abs_matrix_diff(&dense_matrix, &dense_from_compact);

    let mut lapack = LapackStyleBandedLuFaithful::new(compact.n(), compact.kl(), compact.ku())
        .expect("lapack-style workspace should allocate");
    lapack
        .load_from_banded(&compact)
        .expect("lapack-style workspace should load compact banded matrix");
    let loaded_dense = dense_from_vecvec(&lapack.reconstruct_original_band_dense());
    let load_diff = max_abs_matrix_diff(&dense_matrix, &loaded_dense);

    lapack
        .factor_from(&compact)
        .expect("lapack-style factorization should succeed on combustion-1000 banded Jacobian");
    let factor_rel = lapack
        .factor_residual_relative(&compact)
        .expect("factor residual should be available after factorization");

    let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();
    let mut x_lapack = rhs.clone();
    lapack
        .solve_in_place(&mut x_lapack)
        .expect("lapack-style solve should succeed after factorization");
    let solve_rr = relative_dense_residual(&dense_matrix, &x_lapack, &rhs);

    let sparse = dense_to_sparse_col_mat(&dense_matrix);
    let x_sparse = solve_sparse_lu_for_rhs(&sparse, rhs.as_slice());
    let solve_diff = x_lapack
        .iter()
        .zip(x_sparse.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);

    println!(
        "[Lapack banded diagnostic] combustion-1000: n={}, kl={}, ku={}, compact_diff={:.3e}, load_diff={:.3e}, factor_rel={:.3e}, solve_rr={:.3e}, solve_diff={:.3e}",
        compact.n(),
        compact.kl(),
        compact.ku(),
        compact_diff,
        load_diff,
        factor_rel,
        solve_rr,
        solve_diff
    );
}
