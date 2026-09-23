#[test]
#[ignore = "diagnostic combustion Rust-AOT-only compile-preset table; compares rustc presets, not C/Zig toolchains"]
fn symbolic_assembly_rust_compile_presets_report_build_vs_runtime_1000() {
    aot_test_report!(symbolic_assembly_rust_compile_presets_report_build_vs_runtime_1000);
    #[derive(Debug)]
    struct Row {
        backend: &'static str,
        preset: &'static str,
        bootstrap_ms: f64,
        solve_ms: f64,
        max_abs_solution: f64,
    }

    let n_steps = 200usize;
    let mut rows = Vec::new();

    for (backend_label, symbolic_backend) in [
        ("ExprLegacy", BvpSymbolicAssemblyBackend::ExprLegacy),
        ("AtomView", BvpSymbolicAssemblyBackend::AtomView),
    ] {
        for (preset_label, config) in [
                (
                    "Production",
                    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                        .with_aot_compile_production(),
                ),
                (
                    "FastBuild",
                    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                        .with_aot_compile_fast_build(),
                ),
                (
                    "DevFastest",
                    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                        .with_aot_compile_dev_fastest(),
                ),
            ] {
                let generated_backend_config = config
                    .with_symbolic_assembly_backend(symbolic_backend)
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                    .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                        profile:
                            crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                    });

                let mut solver = make_combustion_solver(n_steps, generated_backend_config);
                let label = format!(
                    "combustion-compile-preset-{}-{}-{}",
                    backend_label, preset_label, n_steps
                );
                let (_guard, bootstrap_ms, solve_ms) = solve_with_aot_and_measure(&mut solver, &label)
                    .expect("compile-preset combustion run should solve");
                let solution = solver
                    .get_result()
                    .expect("compile-preset combustion run should produce a solution");
                let max_abs_solution = solution.iter().copied().map(f64::abs).fold(0.0, f64::max);
                assert!(
                    solution.iter().all(|value| value.is_finite()),
                    "{backend_label} {preset_label} solution should remain finite"
                );

                rows.push(Row {
                    backend: backend_label,
                    preset: preset_label,
                    bootstrap_ms,
                    solve_ms,
                    max_abs_solution,
                });
            }
    }

    println!(
        "[BVP symbolic assembly Rust compile presets] combustion build-vs-runtime, n_steps={n_steps}"
    );
    println!(
        "note: this is intentionally Rust-AOT only; C/Zig toolchain comparisons live in the end-to-end and callback-throughput matrices."
    );
    println!(
        "{:<12} | {:<11} | {:>12} | {:>10} | {:>16}",
        "backend", "preset", "bootstrap_ms", "solve_ms", "max_abs_solution"
    );
    println!("{}", "-".repeat(74));
    for row in rows {
        println!(
            "{:<12} | {:<11} | {:>12.3} | {:>10.3} | {:>16.6e}",
            row.backend, row.preset, row.bootstrap_ms, row.solve_ms, row.max_abs_solution
        );
    }
}

#[test]
#[ignore = "diagnostic combustion Lambdify vs AtomView DevFastest end-to-end matrix across Rust/gcc/tcc/zig"]
fn combustion_lambdify_vs_atomview_devfastest_toolchain_end_to_end_1000() {
    aot_test_report!(combustion_lambdify_vs_atomview_devfastest_toolchain_end_to_end_1000);
    #[derive(Debug)]
    struct Row {
        backend: String,
        setup_ms: f64,
        solve_ms: f64,
        total_ms: f64,
        solution_diff: f64,
        max_abs_solution: f64,
    }

    // Keep this compact: it is an end-to-end toolchain comparison, while large-grid
    // stress belongs to the test_aot_race_stress release matrices. Rust AOT can become a
    // compiler-stress test before the solver path is reached on larger generated crates.
    let n_steps = 200usize;

    let lambdify_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

    let mut lambdify_solver = make_combustion_solver(n_steps, lambdify_config);
    let lambdify_generate_begin = Instant::now();
    lambdify_solver
        .try_eq_generate(None, None)
        .expect("lambdify combustion generate should succeed");
    let lambdify_setup_ms = lambdify_generate_begin.elapsed().as_secs_f64() * 1_000.0;
    let lambdify_solve_begin = Instant::now();
    lambdify_solver
        .try_solve()
        .expect("lambdify combustion solve should succeed");
    let lambdify_solve_ms = lambdify_solve_begin.elapsed().as_secs_f64() * 1_000.0;
    let lambdify_solution = lambdify_solver
        .get_result()
        .expect("lambdify combustion solve should produce a solution");

    let mut rows = vec![Row {
        backend: "Lambdify".to_string(),
        setup_ms: lambdify_setup_ms,
        solve_ms: lambdify_solve_ms,
        total_ms: lambdify_setup_ms + lambdify_solve_ms,
        solution_diff: 0.0,
        max_abs_solution: lambdify_solution
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max),
    }];

    for toolchain in RuntimeTuningToolchain::variants() {
        let config = sparse_atomview_rebuild_release_devfastest_config(toolchain);
        let label = format!("AtomView+{}", toolchain.label());
        let mut atom_solver = make_combustion_solver(n_steps, config);
        let (_atom_guard, atom_setup_ms, atom_solve_ms) = solve_with_aot_and_measure(
            &mut atom_solver,
            &format!(
                "combustion-atomview-{}-devfastest-vs-lambdify-{n_steps}",
                toolchain.label()
            ),
        )
        .unwrap_or_else(|err| {
            panic!("{label} DevFastest combustion solve should succeed: {err:?}")
        });
        let atom_solution = atom_solver.get_result().unwrap_or_else(|| {
            panic!("{label} DevFastest combustion solve should produce a solution")
        });

        let solution_diff = lambdify_solution
            .iter()
            .zip(atom_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(
            solution_diff < 1e-6,
            "{label}: Lambdify vs AtomView DevFastest disagreement {solution_diff} is too large"
        );

        rows.push(Row {
            backend: label,
            setup_ms: atom_setup_ms,
            solve_ms: atom_solve_ms,
            total_ms: atom_setup_ms + atom_solve_ms,
            solution_diff,
            max_abs_solution: atom_solution
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max),
        });
    }

    println!(
        "[BVP end-to-end compare] combustion Lambdify vs AtomView DevFastest toolchain matrix, n_steps={n_steps}"
    );
    println!(
        "{:<16} | {:>12} | {:>10} | {:>10} | {:>14} | {:>16}",
        "backend", "setup_ms", "solve_ms", "total_ms", "diff_vs_base", "max_abs_solution"
    );
    println!("{}", "-".repeat(95));
    for row in rows {
        println!(
            "{:<16} | {:>12.3} | {:>10.3} | {:>10.3} | {:>14.6e} | {:>16.6e}",
            row.backend,
            row.setup_ms,
            row.solve_ms,
            row.total_ms,
            row.solution_diff,
            row.max_abs_solution
        );
    }
}

#[test]
#[ignore = "diagnostic combustion callback-throughput matrix: Lambdify baseline vs linked AtomView Rust/gcc/tcc/zig"]
fn combustion_callback_throughput_lambdify_vs_atomview_linked_runtime_1000() {
    aot_test_report!(combustion_callback_throughput_lambdify_vs_atomview_linked_runtime_1000);
    #[derive(Debug)]
    struct Row {
        backend: String,
        residual_ms: f64,
        jacobian_ms: f64,
        total_ms: f64,
        speedup_vs_lambdify: f64,
        residual_diff: f64,
        jacobian_diff: f64,
    }

    // This diagnostic is a runtime callback-binding guard, not a Rust compiler stress test.
    // Large Rust AOT artifacts can overflow rustc's stack before the test reaches the linked
    // chunk callbacks we want to validate here.
    let n_steps = 200usize;
    let iters = 20usize;
    // Cross-backend callback equivalence compares Lambdify against generated
    // cdylib code. Tiny libm/codegen ordering differences show up around
    // 1e-6 on this combustion fixture, while actual callback wiring bugs
    // were orders of magnitude larger.
    let cross_backend_callback_tol = 5.0e-6;

    let lambdify_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

    let mut lambdify_solver = make_combustion_solver(n_steps, lambdify_config);
    let mut lambdify_bundle = sparse_bundle_from_solver_request(&mut lambdify_solver)
        .expect("lambdify sparse bundle should build for callback throughput compare");
    assert_eq!(
        lambdify_bundle.effective_backend(),
        SelectedBackendKind::Lambdify,
        "lambdify callback throughput compare should stay on lambdify backend"
    );
    assert!(
        lambdify_bundle.is_runtime_callable(),
        "lambdify sparse bundle should be runtime-callable"
    );

    let args = DVector::from_element(lambdify_bundle.jacobian_shape().1, 0.99);
    let typed = Vectors_type_casting(
        &args,
        crate::symbolic::symbolic_functions_BVP::BvpMatrixBackend::FaerSparseCol
            .legacy_method()
            .to_string(),
    );

    let (lambdify_residual_ms, lambdify_jacobian_ms) =
        measure_sparse_runtime_callback_throughput(&mut lambdify_bundle, &*typed, iters);
    let lambdify_total_ms = lambdify_residual_ms + lambdify_jacobian_ms;

    let mut rows = vec![Row {
        backend: "Lambdify".to_string(),
        residual_ms: lambdify_residual_ms,
        jacobian_ms: lambdify_jacobian_ms,
        total_ms: lambdify_total_ms,
        speedup_vs_lambdify: 1.0,
        residual_diff: 0.0,
        jacobian_diff: 0.0,
    }];

    for toolchain in RuntimeTuningToolchain::variants() {
        let label = format!("AtomView+{}", toolchain.label());
        let config = sparse_atomview_rebuild_release_devfastest_config(toolchain);
        let mut atom_solver = make_combustion_solver(n_steps, config);
        let _atom_guard = bootstrap_callable_aot_backend(
            &mut atom_solver,
            &format!(
                "combustion-linked-runtime-callback-throughput-{}-{n_steps}",
                toolchain.label()
            ),
        )
        .unwrap_or_else(|err| {
            panic!("{label} bootstrap should succeed for callback throughput compare: {err:?}")
        });
        let mut atom_bundle =
            sparse_bundle_from_solver_request(&mut atom_solver).unwrap_or_else(|err| {
                panic!("{label} sparse bundle should rebuild after linked bootstrap: {err:?}")
            });
        assert_eq!(
            atom_bundle.effective_backend(),
            SelectedBackendKind::AotCompiled,
            "{label}: callback throughput compare should resolve to compiled AOT selection"
        );
        assert!(
            atom_bundle.is_runtime_callable(),
            "{label}: linked sparse bundle should be runtime-callable after bootstrap"
        );

        let (residual_diff, jacobian_diff) = compare_sparse_bundles_numerically(
            &mut lambdify_bundle,
            &mut atom_bundle,
            &args,
            &format!(
                "combustion-callback-throughput-{}-{n_steps}",
                toolchain.label()
            ),
        );
        assert!(
            residual_diff <= cross_backend_callback_tol
                && jacobian_diff <= cross_backend_callback_tol,
            "{label}: runtime callback compare requires numerically close callbacks within {cross_backend_callback_tol:e}, got residual_diff={residual_diff}, jacobian_diff={jacobian_diff}"
        );

        let (atom_residual_ms, atom_jacobian_ms) =
            measure_sparse_runtime_callback_throughput(&mut atom_bundle, &*typed, iters);
        let atom_total_ms = atom_residual_ms + atom_jacobian_ms;

        rows.push(Row {
            backend: label,
            residual_ms: atom_residual_ms,
            jacobian_ms: atom_jacobian_ms,
            total_ms: atom_total_ms,
            speedup_vs_lambdify: lambdify_total_ms / atom_total_ms.max(f64::EPSILON),
            residual_diff,
            jacobian_diff,
        });
    }

    println!(
        "[BVP callback throughput] combustion Lambdify vs AtomView linked-runtime toolchain matrix, n_steps={n_steps}, iters={iters}"
    );
    println!(
        "[BVP callback throughput] note: AtomView+Linked now measures the generated cdylib runtime path loaded into the current process"
    );
    println!(
        "{:<16} | {:>12} | {:>12} | {:>12} | {:>18} | {:>13} | {:>13}",
        "backend",
        "residual_ms",
        "jacobian_ms",
        "total_ms",
        "speedup_vs_lambdify",
        "res_diff",
        "jac_diff"
    );
    println!("{}", "-".repeat(112));
    for row in rows {
        println!(
            "{:<16} | {:>12.3} | {:>12.3} | {:>12.3} | {:>18.3}x | {:>13.6e} | {:>13.6e}",
            row.backend,
            row.residual_ms,
            row.jacobian_ms,
            row.total_ms,
            row.speedup_vs_lambdify,
            row.residual_diff,
            row.jacobian_diff
        );
    }
}

#[test]
#[ignore = "diagnostic sparse AOT whole-vs-chunked callback throughput across Rust/gcc/tcc/zig; isolates runtime parallelism from Newton/bootstrap cost"]
fn combustion_sparse_aot_callback_chunking_parallelism_diagnostic() {
    aot_test_report!(combustion_sparse_aot_callback_chunking_parallelism_diagnostic);
    #[derive(Debug)]
    struct CallbackChunkRow {
        config: String,
        bootstrap_ms: f64,
        rayon_workers: usize,
        residual_chunks: usize,
        sparse_chunks: usize,
        residual_jobs: usize,
        sparse_jobs: usize,
        residual_ms: RuntimeTuningAggregate,
        jacobian_ms: RuntimeTuningAggregate,
        total_ms: RuntimeTuningAggregate,
        speedup_vs_whole: RuntimeTuningAggregate,
        residual_diff_vs_whole: f64,
        jacobian_diff_vs_whole: f64,
    }

    fn sparse_aot_callback_config(
        toolchain: RuntimeTuningToolchain,
        execution_policy: AotExecutionPolicy,
        chunking_policy: AotChunkingPolicy,
    ) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
        let config = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Debug,
                })
                .with_aot_compile_dev_fastest()
                .with_aot_execution_policy(execution_policy)
                .with_aot_chunking_policy(chunking_policy);
        toolchain.apply_to(config)
    }

    fn linked_job_range_count(chunk_count: usize, max_jobs: usize) -> usize {
        if chunk_count == 0 {
            return 0;
        }
        let target_jobs = chunk_count.min(max_jobs.max(1));
        let items_per_job = chunk_count.div_ceil(target_jobs);
        chunk_count.div_ceil(items_per_job)
    }

    fn policy_max_jobs(policy: &AotExecutionPolicy) -> (usize, usize) {
        match policy {
            AotExecutionPolicy::Parallel(config) => {
                let workers = rayon::current_num_threads().max(1);
                let worker_jobs = workers.saturating_mul(config.jobs_per_worker.max(1));
                (
                    config.max_residual_jobs.unwrap_or(worker_jobs).max(1),
                    config.max_sparse_jobs.unwrap_or(worker_jobs).max(1),
                )
            }
            _ => (1, 1),
        }
    }

    fn linked_sparse_chunk_counts(
        bundle: &BvpSparseSolverBundle,
        residual_max_jobs: usize,
        sparse_max_jobs: usize,
    ) -> (usize, usize, usize, usize) {
        let Some(problem_key) = bundle
            .resolved_aot_artifact()
            .map(|artifact| artifact.registered.problem_key.as_str())
        else {
            return (0, 0, 0, 0);
        };
        let Some(linked) = resolve_linked_sparse_backend(problem_key) else {
            return (0, 0, 0, 0);
        };
        let residual_chunks = linked.residual_chunks.len();
        let sparse_chunks = linked.jacobian_value_chunks.len();
        (
            residual_chunks,
            sparse_chunks,
            linked_job_range_count(residual_chunks, residual_max_jobs),
            linked_job_range_count(sparse_chunks, sparse_max_jobs),
        )
    }

    // This diagnostic is a runtime callback-binding guard, not a Rust compiler stress test.
    // Large Rust AOT artifacts can overflow rustc's stack before the test reaches the linked
    // chunk callbacks we want to validate here.
    let n_steps = 200usize;
    let callback_iters = 30usize;
    let measurement_repeats = 5usize;
    let variants = [
        (
            "whole-sequential",
            AotExecutionPolicy::SequentialOnly,
            AotChunkingPolicy::default(),
        ),
        (
            "par-4x4-jobs4",
            AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(4),
                max_sparse_jobs: Some(4),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
            AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            ),
        ),
        (
            "par-8x8-jobs8",
            AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(8),
                max_sparse_jobs: Some(8),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
            AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
            ),
        ),
        (
            "par-16x16-jobs16",
            AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(16),
                max_sparse_jobs: Some(16),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
            AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
            ),
        ),
    ];

    let mut rows = Vec::new();

    for toolchain in RuntimeTuningToolchain::variants() {
        let toolchain_label = toolchain.label();
        let mut whole_solver = make_combustion_solver(
            n_steps,
            sparse_aot_callback_config(
                toolchain,
                AotExecutionPolicy::SequentialOnly,
                AotChunkingPolicy::default(),
            ),
        );
        let whole_bootstrap_begin = Instant::now();
        let _whole_guard = bootstrap_callable_aot_backend(
            &mut whole_solver,
            &format!("combustion-sparse-aot-callback-{toolchain_label}-whole-{n_steps}"),
        )
        .unwrap_or_else(|err| {
            panic!(
                "{toolchain_label}/whole: sparse AOT callback diagnostic bootstrap failed: {err:?}"
            )
        });
        let whole_bootstrap_ms = whole_bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
        let mut whole_bundle =
            sparse_bundle_from_solver_request(&mut whole_solver).unwrap_or_else(|err| {
                panic!(
                    "{toolchain_label}/whole: sparse AOT callback diagnostic bundle failed: {err:?}"
                )
            });
        assert_eq!(
            whole_bundle.effective_backend(),
            SelectedBackendKind::AotCompiled,
            "{toolchain_label}/whole: sparse AOT callback diagnostic should use compiled backend"
        );

        let args = DVector::from_element(whole_bundle.jacobian_shape().1, 0.99);
        let typed = Vectors_type_casting(
            &args,
            crate::symbolic::symbolic_functions_BVP::BvpMatrixBackend::FaerSparseCol
                .legacy_method()
                .to_string(),
        );

        let mut whole_residual_samples = Vec::with_capacity(measurement_repeats);
        let mut whole_jacobian_samples = Vec::with_capacity(measurement_repeats);
        for _ in 0..measurement_repeats {
            let (residual_ms, jacobian_ms) = measure_sparse_runtime_callback_throughput(
                &mut whole_bundle,
                &*typed,
                callback_iters,
            );
            whole_residual_samples.push(residual_ms);
            whole_jacobian_samples.push(jacobian_ms);
        }
        let whole_total_samples = whole_residual_samples
            .iter()
            .zip(whole_jacobian_samples.iter())
            .map(|(&residual_ms, &jacobian_ms)| residual_ms + jacobian_ms)
            .collect::<Vec<_>>();
        let whole_total_mean = runtime_tuning_aggregate(whole_total_samples.iter().copied()).mean;
        let (whole_residual_chunks, whole_sparse_chunks, whole_residual_jobs, whole_sparse_jobs) =
            linked_sparse_chunk_counts(&whole_bundle, 1, 1);

        rows.push(CallbackChunkRow {
            config: format!("{toolchain_label}/whole-sequential"),
            bootstrap_ms: whole_bootstrap_ms,
            rayon_workers: rayon::current_num_threads(),
            residual_chunks: whole_residual_chunks,
            sparse_chunks: whole_sparse_chunks,
            residual_jobs: whole_residual_jobs,
            sparse_jobs: whole_sparse_jobs,
            residual_ms: runtime_tuning_aggregate(whole_residual_samples.iter().copied()),
            jacobian_ms: runtime_tuning_aggregate(whole_jacobian_samples.iter().copied()),
            total_ms: runtime_tuning_aggregate(whole_total_samples.iter().copied()),
            speedup_vs_whole: runtime_tuning_aggregate([1.0]),
            residual_diff_vs_whole: 0.0,
            jacobian_diff_vs_whole: 0.0,
        });

        for (label, execution_policy, chunking_policy) in variants.iter().skip(1) {
            let (residual_max_jobs, sparse_max_jobs) = policy_max_jobs(execution_policy);
            let full_label = format!("{toolchain_label}/{label}");
            let mut solver = make_combustion_solver(
                n_steps,
                sparse_aot_callback_config(
                    toolchain,
                    execution_policy.clone(),
                    chunking_policy.clone(),
                ),
            );
            let bootstrap_begin = Instant::now();
            let _guard = bootstrap_callable_aot_backend(
                &mut solver,
                &format!("combustion-sparse-aot-callback-{toolchain_label}-{label}-{n_steps}"),
            )
            .unwrap_or_else(|err| {
                panic!("{full_label}: sparse AOT callback diagnostic bootstrap failed: {err:?}")
            });
            let bootstrap_ms = bootstrap_begin.elapsed().as_secs_f64() * 1_000.0;
            let mut bundle = sparse_bundle_from_solver_request(&mut solver).unwrap_or_else(|err| {
                panic!("{full_label}: sparse AOT callback diagnostic bundle failed: {err:?}")
            });
            assert_eq!(
                bundle.effective_backend(),
                SelectedBackendKind::AotCompiled,
                "{full_label}: sparse AOT callback diagnostic should use compiled backend"
            );

            let (residual_diff, jacobian_diff) = compare_sparse_bundles_numerically(
                &mut whole_bundle,
                &mut bundle,
                &args,
                &format!("combustion-sparse-aot-callback-{toolchain_label}-{label}-{n_steps}"),
            );
            assert!(
                residual_diff < 1e-6 && jacobian_diff < 1e-6,
                "{full_label}: chunked callback differs from whole callback, residual_diff={residual_diff}, jacobian_diff={jacobian_diff}"
            );

            let mut residual_samples = Vec::with_capacity(measurement_repeats);
            let mut jacobian_samples = Vec::with_capacity(measurement_repeats);
            for _ in 0..measurement_repeats {
                let (residual_ms, jacobian_ms) = measure_sparse_runtime_callback_throughput(
                    &mut bundle,
                    &*typed,
                    callback_iters,
                );
                residual_samples.push(residual_ms);
                jacobian_samples.push(jacobian_ms);
            }
            let total_samples = residual_samples
                .iter()
                .zip(jacobian_samples.iter())
                .map(|(&residual_ms, &jacobian_ms)| residual_ms + jacobian_ms)
                .collect::<Vec<_>>();
            let speedup_samples = total_samples
                .iter()
                .map(|&total_ms| whole_total_mean / total_ms.max(f64::EPSILON))
                .collect::<Vec<_>>();
            let (residual_chunks, sparse_chunks, residual_jobs, sparse_jobs) =
                linked_sparse_chunk_counts(&bundle, residual_max_jobs, sparse_max_jobs);
            assert!(
                residual_chunks > 1,
                "{full_label}: requested chunked sparse AOT residual execution, but linked backend registered only {residual_chunks} residual chunks"
            );
            assert!(
                sparse_chunks > 1,
                "{full_label}: requested chunked sparse AOT Jacobian execution, but linked backend registered only {sparse_chunks} Jacobian chunks"
            );
            assert!(
                residual_jobs > 1,
                "{full_label}: requested parallel residual execution, but runtime job planner produced only {residual_jobs} job"
            );
            assert!(
                sparse_jobs > 1,
                "{full_label}: requested parallel Jacobian execution, but runtime job planner produced only {sparse_jobs} job"
            );

            rows.push(CallbackChunkRow {
                config: full_label,
                bootstrap_ms,
                rayon_workers: rayon::current_num_threads(),
                residual_chunks,
                sparse_chunks,
                residual_jobs,
                sparse_jobs,
                residual_ms: runtime_tuning_aggregate(residual_samples.iter().copied()),
                jacobian_ms: runtime_tuning_aggregate(jacobian_samples.iter().copied()),
                total_ms: runtime_tuning_aggregate(total_samples.iter().copied()),
                speedup_vs_whole: runtime_tuning_aggregate(speedup_samples.iter().copied()),
                residual_diff_vs_whole: residual_diff,
                jacobian_diff_vs_whole: jacobian_diff,
            });
        }
    }

    println!(
        "[BVP callback parallelism diagnostic] sparse AtomView AOT callbacks, n_steps={n_steps}, callback_iters={callback_iters}, measurement_repeats={measurement_repeats}"
    );
    println!(
        "note: this isolates residual/Jacobian callback evaluation; bootstrap_ms is reported only to expose artifact overhead and is not part of callback throughput."
    );
    println!(
        "{:<28} | {:>12} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<18} | {:>15} | {:>15}",
        "config",
        "bootstrap_ms",
        "workers",
        "res_ch",
        "jac_ch",
        "res_jobs",
        "jac_jobs",
        "residual_ms",
        "jacobian_ms",
        "callback_total_ms",
        "speedup_vs_whole",
        "residual_diff",
        "jacobian_diff",
    );
    println!("{}", "-".repeat(214));
    for row in rows {
        println!(
            "{:<28} | {:>12.3} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:<18} | {:<18} | {:<18} | {:<18} | {:>15.6e} | {:>15.6e}",
            row.config,
            row.bootstrap_ms,
            row.rayon_workers,
            row.residual_chunks,
            row.sparse_chunks,
            row.residual_jobs,
            row.sparse_jobs,
            fmt_tuning_short(row.residual_ms),
            fmt_tuning_short(row.jacobian_ms),
            fmt_tuning_short(row.total_ms),
            fmt_tuning_short(row.speedup_vs_whole),
            row.residual_diff_vs_whole,
            row.jacobian_diff_vs_whole,
        );
    }
}

#[test]
#[ignore = "diagnostic stage breakdown inside generate_ms for combustion symbolic backends"]
fn symbolic_assembly_backends_report_combustion_generate_breakdown_table() {
    aot_test_report!(symbolic_assembly_backends_report_combustion_generate_breakdown_table);
    #[derive(Debug)]
    struct GenerateBreakdownRow {
        backend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
        discretization_ms: f64,
        symbolic_jacobian_ms: f64,
        sparse_aot_prep_ms: f64,
        total_ms: f64,
    }

    let n_steps_list = [200usize, 300usize];
    let base_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();
    let mut rows = Vec::new();

    for &n_steps in &n_steps_list {
        let mut legacy_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_request = legacy_solver.build_solver_request(None, None);
        let legacy_snapshot = measure_symbolic_generation_breakdown_with_symbolic_backend(
            legacy_request,
            BvpSymbolicAssemblyBackend::ExprLegacy,
        )
        .expect("ExprLegacy combustion breakdown should build");

        let mut atom_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_request = atom_solver.build_solver_request(None, None);
        let atom_snapshot = measure_symbolic_generation_breakdown_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        )
        .expect("AtomView combustion breakdown should build");

        for (backend, snapshot) in [
            (BvpSymbolicAssemblyBackend::ExprLegacy, legacy_snapshot),
            (BvpSymbolicAssemblyBackend::AtomView, atom_snapshot),
        ] {
            rows.push(GenerateBreakdownRow {
                backend,
                n_steps,
                discretization_ms: snapshot
                    .get("discretization time")
                    .copied()
                    .unwrap_or_default()
                    * 1_000.0,
                symbolic_jacobian_ms: snapshot
                    .get("symbolic jacobian time")
                    .copied()
                    .unwrap_or_default()
                    * 1_000.0,
                sparse_aot_prep_ms: snapshot
                    .get("sparse AOT preparation time")
                    .copied()
                    .unwrap_or_default()
                    * 1_000.0,
                total_ms: snapshot.get("total time, sec").copied().unwrap_or_default() * 1_000.0,
            });
        }
    }

    println!("[BVP symbolic assembly generate breakdown] combustion ExprLegacy vs AtomView");
    println!(
        "{:<12} | {:>7} | {:>17} | {:>20} | {:>19} | {:>10}",
        "backend",
        "n_steps",
        "discretization_ms",
        "symbolic_jacobian_ms",
        "sparse_aot_prep_ms",
        "total_ms"
    );
    println!("{}", "-".repeat(104));
    for row in &rows {
        let backend = match row.backend {
            BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
            BvpSymbolicAssemblyBackend::AtomView => "AtomView",
        };
        println!(
            "{:<12} | {:>7} | {:>17.3} | {:>20.3} | {:>19.3} | {:>10.3}",
            backend,
            row.n_steps,
            row.discretization_ms,
            row.symbolic_jacobian_ms,
            row.sparse_aot_prep_ms,
            row.total_ms
        );
    }
}

#[test]
#[ignore = "heavy oscillator end-to-end compare for banded lambdify baseline vs AtomView AOT with bootstrap/runtime split"]
fn oscillator_lambdify_vs_atomview_aot_banded_end_to_end_heavy() {
    aot_test_report!(oscillator_lambdify_vs_atomview_aot_banded_end_to_end_heavy);
    #[derive(Debug)]
    struct Row {
        backend: &'static str,
        setup_ms: f64,
        solve_ms: f64,
        total_ms: f64,
        max_abs_solution: f64,
    }

    let n_steps = 1000usize;

    let lambdify_cfg =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
            .with_matrix_backend_override(MatrixBackend::Banded);

    let atom_aot_cfg =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_matrix_backend_override(MatrixBackend::Banded);

    let mut lambdify_solver = make_oscillator_solver(n_steps, lambdify_cfg);
    let lambdify_generate_begin = Instant::now();
    lambdify_solver
        .try_eq_generate(None, None)
        .expect("banded lambdify oscillator generate should succeed");
    let lambdify_setup_ms = lambdify_generate_begin.elapsed().as_secs_f64() * 1_000.0;
    let lambdify_solve_begin = Instant::now();
    lambdify_solver
        .try_solve()
        .expect("banded lambdify oscillator solve should succeed");
    let lambdify_solve_ms = lambdify_solve_begin.elapsed().as_secs_f64() * 1_000.0;
    let lambdify_solution = lambdify_solver
        .get_result()
        .expect("banded lambdify oscillator solve should produce a solution");

    let mut atom_solver = make_oscillator_solver(n_steps, atom_aot_cfg);
    let (atom_setup_ms, atom_solve_ms, atom_solution) = match solve_with_aot_and_measure(
        &mut atom_solver,
        &format!("oscillator-atomview-aot-banded-vs-lambdify-{n_steps}"),
    ) {
        Ok((_guard, setup_ms, solve_ms)) => {
            let solution = atom_solver
                .get_result()
                .expect("banded AtomView AOT oscillator solve should produce a solution")
                .clone();
            (setup_ms, solve_ms, solution)
        }
        Err(err) if is_aot_environment_issue(&err) => {
            eprintln!(
                "Skipping heavy oscillator AOT compare due to environment/toolchain issue: {err:?}"
            );
            return;
        }
        Err(err) => {
            panic!("banded AtomView AOT oscillator compare failed unexpectedly: {err:?}")
        }
    };

    let solution_diff = lambdify_solution
        .iter()
        .zip(atom_solution.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max);
    assert!(
        solution_diff < 1e-5,
        "banded oscillator Lambdify vs AtomView AOT disagreement {solution_diff} is too large"
    );

    let rows = [
        Row {
            backend: "Lambdify",
            setup_ms: lambdify_setup_ms,
            solve_ms: lambdify_solve_ms,
            total_ms: lambdify_setup_ms + lambdify_solve_ms,
            max_abs_solution: lambdify_solution
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max),
        },
        Row {
            backend: "AtomView+AOT",
            setup_ms: atom_setup_ms,
            solve_ms: atom_solve_ms,
            total_ms: atom_setup_ms + atom_solve_ms,
            max_abs_solution: atom_solution
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max),
        },
    ];

    println!(
        "[BVP end-to-end compare] oscillator banded Lambdify vs AtomView AOT, n_steps={n_steps}"
    );
    println!(
        "{:<14} | {:>12} | {:>10} | {:>10} | {:>16}",
        "backend", "setup_ms", "solve_ms", "total_ms", "max_abs_solution"
    );
    println!("{}", "-".repeat(76));
    for row in rows {
        println!(
            "{:<14} | {:>12.3} | {:>10.3} | {:>10.3} | {:>16.6e}",
            row.backend, row.setup_ms, row.solve_ms, row.total_ms, row.max_abs_solution
        );
    }
    println!(
        "[BVP end-to-end compare] max_diff_lambdify_vs_atomview_aot = {:.6e}",
        solution_diff
    );
}

#[test]
#[ignore = "diagnostic AOT crate emission/materialize/build comparison for combustion symbolic backends"]
fn symbolic_assembly_backends_report_combustion_aot_crate_build_table() {
    aot_test_report!(symbolic_assembly_backends_report_combustion_aot_crate_build_table);
    let n_steps_list = [200usize, 300usize];
    let base_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();
    let mut rows = Vec::new();

    for &n_steps in &n_steps_list {
        let mut legacy_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_request = legacy_solver.build_solver_request(None, None);
        rows.push(
            measure_generated_crate_build_with_symbolic_backend(
                legacy_request,
                BvpSymbolicAssemblyBackend::ExprLegacy,
                n_steps,
                &format!("combustion-aot-expr-{n_steps}"),
            )
            .expect("ExprLegacy combustion AOT crate generation should succeed"),
        );

        let mut atom_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_request = atom_solver.build_solver_request(None, None);
        rows.push(
            measure_generated_crate_build_with_symbolic_backend(
                atom_request,
                BvpSymbolicAssemblyBackend::AtomView,
                n_steps,
                &format!("combustion-aot-atom-{n_steps}"),
            )
            .expect("AtomView combustion AOT crate generation should succeed"),
        );
    }

    println!("[BVP symbolic assembly AOT crate build] combustion ExprLegacy vs AtomView");
    println!(
        "{:<12} | {:>7} | {:>11} | {:>11} | {:>11} | {:>8} | {:>11} | {:>10} | {:>14} | {:>10} | {:>10} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:<18}",
        "backend",
        "n_steps",
        "jac_prep_ms",
        "lookup_ms",
        "jac_ms",
        "nnz",
        "finalize_ms",
        "module_ms",
        "source_ms",
        "materialize_ms",
        "build_ms",
        "source_kb",
        "blocks",
        "instr",
        "temps",
        "max_blk",
        "outputs",
        "status"
    );
    println!("{}", "-".repeat(234));
    for row in &rows {
        let backend = match row.backend {
            BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
            BvpSymbolicAssemblyBackend::AtomView => "AtomView",
        };
        println!(
            "{:<12} | {:>7} | {:>11} | {:>11} | {:>11} | {:>8} | {:>11} | {:>10} | {:>14} | {:>10} | {:>10} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:<18}",
            backend,
            row.n_steps,
            row.jacobian_prepare_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_sparse_lookup_prepare_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_sparse_jacobian_build_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_sparse_nnz
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.atom_finalize_codegen_plan_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.module_build_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.source_emit_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.materialize_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.build_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.source_kb
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "-".to_string()),
            row.module_blocks
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.total_block_instructions
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.total_block_temps
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.max_block_instructions
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.total_block_outputs
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            &row.status
        );
    }

    println!();
    println!("[BVP symbolic assembly AOT crate build] atom module pass breakdown");
    println!(
        "{:<12} | {:>7} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}",
        "backend",
        "n_steps",
        "res_view_ms",
        "res_lower_ms",
        "res_ph_ms",
        "res_reuse_ms",
        "sp_view_ms",
        "sp_lower_ms",
        "sp_ph_ms",
        "sp_reuse_ms"
    );
    println!("{}", "-".repeat(143));
    for row in rows
        .iter()
        .filter(|row| matches!(row.backend, BvpSymbolicAssemblyBackend::AtomView))
    {
        println!(
            "{:<12} | {:>7} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}",
            "AtomView",
            row.n_steps,
            row.atom_residual_view_collect_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_residual_lower_many_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_residual_peephole_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_residual_reuse_temps_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_sparse_view_collect_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_sparse_lower_many_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_sparse_peephole_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.atom_sparse_reuse_temps_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string()),
        );
    }

    println!();
    println!(
        "[BVP typed AOT lifecycle] AtomView rows expose the same canonical buckets as the historical AOT story"
    );
    println!(
        "frontend | n_steps | symbolic_prepare_ms | fixture_generation_ms | compile_ms | link_ms | residual_calls | jacobian_calls | errors"
    );
    println!("{}", "-".repeat(132));
    for row in &rows {
        let Some(snapshot) = row.typed_aot_snapshot else {
            continue;
        };
        println!(
            "{:>8} | {:>7} | {:>19.3} | {:>22.3} | {:>10.3} | {:>7.3} | {:>14} | {:>14} | {:>6}",
            match row.backend {
                BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
                BvpSymbolicAssemblyBackend::AtomView => "AtomView",
            },
            row.n_steps,
            snapshot.symbolic_preparation().as_secs_f64() * 1_000.0,
            snapshot.fixture_generation().as_secs_f64() * 1_000.0,
            snapshot.compilation().as_secs_f64() * 1_000.0,
            snapshot.linking().as_secs_f64() * 1_000.0,
            snapshot.residual_calls,
            snapshot.jacobian_calls,
            snapshot.errors,
        );
    }
}

#[test]
#[ignore = "diagnostic chunk-level IR compare for ExprLegacy vs AtomView combustion lowering"]
fn symbolic_assembly_backends_report_combustion_chunk_ir_table() {
    aot_test_report!(symbolic_assembly_backends_report_combustion_chunk_ir_table);
    let n_steps = 300usize;
    let base_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

    let mut legacy_solver = make_combustion_solver(
        n_steps,
        base_config
            .clone()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
    );
    let legacy_request = legacy_solver.build_solver_request(None, None);
    let (legacy_module, legacy_breakdown) = measure_codegen_module_with_symbolic_backend(
        legacy_request,
        BvpSymbolicAssemblyBackend::ExprLegacy,
    )
    .expect("legacy combustion codegen module should build");

    let mut atom_solver = make_combustion_solver(
        n_steps,
        base_config
            .clone()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
    );
    let atom_request = atom_solver.build_solver_request(None, None);
    let (atom_module, atom_breakdown) = measure_codegen_module_with_symbolic_backend(
        atom_request,
        BvpSymbolicAssemblyBackend::AtomView,
    )
    .expect("atom combustion codegen module should build");

    assert_eq!(
        legacy_module.blocks().len(),
        atom_module.blocks().len(),
        "legacy and atom module should produce the same number of blocks"
    );

    let rows = legacy_module
        .blocks()
        .iter()
        .zip(atom_module.blocks().iter())
        .map(|(legacy, atom)| ChunkIrCompareRow {
            fn_name: legacy.fn_name.clone(),
            outputs: legacy.output_count(),
            legacy_instr: legacy.instruction_count(),
            atom_instr: atom.instruction_count(),
            legacy_temps: legacy.temp_count(),
            atom_temps: atom.temp_count(),
        })
        .collect::<Vec<_>>();

    println!(
        "[BVP symbolic assembly chunk IR compare] combustion, n_steps={n_steps}, legacy_instr_total={}, atom_instr_total={}, legacy_temps_total={}, atom_temps_total={}",
        legacy_breakdown.total_block_instructions,
        atom_breakdown.total_block_instructions,
        legacy_breakdown.total_block_temps,
        atom_breakdown.total_block_temps
    );
    println!(
        "{:<36} | {:>7} | {:>12} | {:>10} | {:>12} | {:>10}",
        "fn_name", "outputs", "legacy_instr", "atom_instr", "legacy_temps", "atom_temps"
    );
    println!("{}", "-".repeat(103));
    for row in rows.iter().take(12) {
        println!(
            "{:<36} | {:>7} | {:>12} | {:>10} | {:>12} | {:>10}",
            row.fn_name,
            row.outputs,
            row.legacy_instr,
            row.atom_instr,
            row.legacy_temps,
            row.atom_temps
        );
    }
}
