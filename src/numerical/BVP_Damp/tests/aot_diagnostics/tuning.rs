fn runtime_tuning_lambdify_config_for(
    matrix_backend: MatrixBackend,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
        .with_matrix_backend_override(matrix_backend)
}

#[allow(dead_code)]
fn runtime_tuning_lambdify_config()
-> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    runtime_tuning_lambdify_config_for(MatrixBackend::SparseCol)
}

fn runtime_tuning_aot_base_config_for(
    toolchain: RuntimeTuningToolchain,
    matrix_backend: MatrixBackend,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    let config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_matrix_backend_override(matrix_backend)
                .with_aot_compile_dev_fastest();
    toolchain.apply_to(config)
}

#[allow(dead_code)]
fn runtime_tuning_aot_base_config(
    toolchain: RuntimeTuningToolchain,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    runtime_tuning_aot_base_config_for(toolchain, MatrixBackend::SparseCol)
}

fn sparse_atomview_rebuild_release_devfastest_config(
    toolchain: RuntimeTuningToolchain,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    let config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
                .with_aot_compile_dev_fastest()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                    profile:
                        crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
                });
    toolchain.apply_to(config)
}

#[test]
fn atomview_aot_selection_keeps_expr_compatibility_payload_empty() {
    aot_test_report!(atomview_aot_selection_keeps_expr_compatibility_payload_empty);
    let config = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::AotOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_matrix_backend_override(MatrixBackend::SparseCol)
            .with_aot_telemetry_mode(BvpAotTelemetryMode::Detailed);
    let mut solver = make_combustion_solver(4, config);
    let bundle = sparse_bundle_from_solver_request(&mut solver)
        .expect("AtomView AOT selection should build a prepared bundle without compiling");
    let selected = bundle.execution.selected();

    assert_eq!(
        selected.preparation_route(),
        crate::symbolic::bvp::legacy::BvpAotPreparationRoute::AtomViewNative
    );
    assert!(selected.prepared_problem.residuals.is_empty());
    assert!(selected.prepared_problem.sparse_entries.is_empty());
    assert!(selected.prepared_problem.atom_codegen.is_some());
    assert!(selected.prepared_problem.atom_aot_plan().is_some());
    match selected
        .prepared_problem
        .try_aot_adapter()
        .expect("AtomView AOT payload should expose a complete typed adapter")
    {
        BvpAotAdapter::AtomView(adapter) => {
            assert!(adapter.codegen.residuals.is_empty() == false);
            assert_eq!(
                adapter.plan.matrix_layout().shape(),
                selected.prepared_problem.shape
            );
        }
        BvpAotAdapter::ExprLegacy(_) => panic!("AtomView route exposed ExprLegacy adapter"),
    }
    assert_eq!(selected.matrix_backend, MatrixBackend::SparseCol);
    let sparse_snapshot = selected
        .prepared_problem
        .aot_telemetry_snapshot()
        .expect("AtomView route should expose typed AOT telemetry");
    assert_eq!(sparse_snapshot.mode, BvpAotTelemetryMode::Detailed);
    assert!(sparse_snapshot.validation > Duration::ZERO);

    let (_, sparse_codegen_breakdown) = selected
        .prepared_problem
        .codegen_module_with_breakdown_for_matrix_backend(
            "bvp_atom_aot_telemetry_gate",
            MatrixBackend::SparseCol,
        );
    assert!(sparse_codegen_breakdown.module_blocks > 0);
    let (_, sparse_try_breakdown) = selected
        .prepared_problem
        .try_codegen_module_with_breakdown_for_matrix_backend(
            "bvp_atom_aot_try_gate",
            MatrixBackend::SparseCol,
        )
        .expect("typed AtomView codegen route should validate before emission");
    assert!(sparse_try_breakdown.module_blocks > 0);
    let sparse_after_codegen = selected
        .prepared_problem
        .aot_telemetry_snapshot()
        .expect("AtomView plan should keep telemetry after lowering");
    assert!(sparse_after_codegen.lowering >= sparse_snapshot.lowering);
    assert!(sparse_after_codegen.source_emission >= sparse_snapshot.source_emission);

    let (_, sparse_artifact_breakdown) = selected
        .prepared_problem
        .generated_aot_artifact_with_breakdown_for_matrix_backend(
            "bvp_atom_aot_lifecycle_gate",
            "bvp_atom_aot_lifecycle_gate_module",
            AotCodegenBackend::Rust,
            MatrixBackend::SparseCol,
        );
    assert!(sparse_artifact_breakdown.module_blocks > 0);
    let (_, sparse_try_artifact_breakdown) = selected
        .prepared_problem
        .try_generated_aot_artifact_with_breakdown_for_matrix_backend(
            "bvp_atom_aot_try_artifact_gate",
            "bvp_atom_aot_try_artifact_gate_module",
            AotCodegenBackend::Rust,
            MatrixBackend::SparseCol,
        )
        .expect("typed AtomView artifact route should validate before materialization");
    assert!(sparse_try_artifact_breakdown.module_blocks > 0);
    let sparse_after_artifact = selected
        .prepared_problem
        .aot_telemetry_snapshot()
        .expect("AtomView plan should keep lifecycle telemetry after materialization");
    assert!(sparse_after_artifact.materialization >= sparse_after_codegen.materialization);

    let banded_config = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::AotOnly))
            .with_aot_telemetry_mode(BvpAotTelemetryMode::Detailed);
    let mut banded_solver = make_combustion_solver(4, banded_config);
    let banded_bundle = sparse_bundle_from_solver_request(&mut banded_solver)
        .expect("AtomView Banded AOT selection should build a prepared bundle");
    let banded_selected = banded_bundle.execution.selected();
    assert_eq!(
        banded_selected.preparation_route(),
        crate::symbolic::bvp::legacy::BvpAotPreparationRoute::AtomViewNative
    );
    assert!(banded_selected.prepared_problem.residuals.is_empty());
    assert!(banded_selected.prepared_problem.sparse_entries.is_empty());
    assert!(banded_selected.prepared_problem.atom_codegen.is_some());
    assert!(banded_selected.prepared_problem.atom_aot_plan().is_some());
    assert!(matches!(
        banded_selected.prepared_problem.try_aot_adapter(),
        Ok(BvpAotAdapter::AtomView(_))
    ));
    assert_eq!(banded_selected.matrix_backend, MatrixBackend::Banded);
    let banded_snapshot = banded_selected
        .prepared_problem
        .aot_telemetry_snapshot()
        .expect("Banded AtomView route should expose typed AOT telemetry");
    assert_eq!(banded_snapshot.mode, BvpAotTelemetryMode::Detailed);
    assert!(banded_snapshot.validation > Duration::ZERO);
    let (_, banded_try_breakdown) = banded_selected
        .prepared_problem
        .try_codegen_module_with_breakdown_for_matrix_backend(
            "bvp_atom_aot_banded_try_gate",
            MatrixBackend::Banded,
        )
        .expect("typed AtomView Banded route should validate before emission");
    assert!(banded_try_breakdown.module_blocks > 0);
}

fn runtime_tuning_sequential_case() -> RuntimeTuningChunkCase {
    RuntimeTuningChunkCase {
        label: "seq",
        execution_policy: AotExecutionPolicy::SequentialOnly,
        chunking_policy: AotChunkingPolicy::default(),
    }
}

fn runtime_tuning_parallel_cases() -> Vec<RuntimeTuningChunkCase> {
    vec![
        RuntimeTuningChunkCase {
            label: "par-4x4-jobs4",
            execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(4),
                max_sparse_jobs: Some(4),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
            chunking_policy: AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            ),
        },
        RuntimeTuningChunkCase {
            label: "par-8x8-jobs8",
            execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(8),
                max_sparse_jobs: Some(8),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
            chunking_policy: AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
            ),
        },
        RuntimeTuningChunkCase {
            label: "par-16x16-jobs16",
            execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(16),
                max_sparse_jobs: Some(16),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
            chunking_policy: AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
            ),
        },
        RuntimeTuningChunkCase {
            label: "par-res16-row32-jobs8",
            execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(8),
                max_sparse_jobs: Some(8),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
            chunking_policy: AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 32 }),
            ),
        },
    ]
}

fn runtime_tuning_variant_label(
    toolchain: RuntimeTuningToolchain,
    chunk_case: &RuntimeTuningChunkCase,
) -> String {
    format!("{}/{}", toolchain.label(), chunk_case.label)
}

fn runtime_tuning_aot_config_for(
    toolchain: RuntimeTuningToolchain,
    chunk_case: &RuntimeTuningChunkCase,
    matrix_backend: MatrixBackend,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    runtime_tuning_aot_base_config_for(toolchain, matrix_backend)
        .with_aot_execution_policy(chunk_case.execution_policy.clone())
        .with_aot_chunking_policy(chunk_case.chunking_policy)
}

#[allow(dead_code)]
fn runtime_tuning_aot_config(
    toolchain: RuntimeTuningToolchain,
    chunk_case: &RuntimeTuningChunkCase,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    runtime_tuning_aot_config_for(toolchain, chunk_case, MatrixBackend::SparseCol)
}

fn runtime_tuning_honest_cold_aot_config_for(
    toolchain: RuntimeTuningToolchain,
    chunk_case: &RuntimeTuningChunkCase,
    matrix_backend: MatrixBackend,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    runtime_tuning_aot_config_for(toolchain, chunk_case, matrix_backend).with_aot_build_policy(
        AotBuildPolicy::RebuildAlways {
            profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
        },
    )
}

#[allow(dead_code)]
fn runtime_tuning_honest_cold_aot_config(
    toolchain: RuntimeTuningToolchain,
    chunk_case: &RuntimeTuningChunkCase,
) -> crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig {
    runtime_tuning_honest_cold_aot_config_for(toolchain, chunk_case, MatrixBackend::SparseCol)
}

fn runtime_tuning_cold_variants_for(
    matrix_backend: MatrixBackend,
) -> Vec<(
    String,
    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
)> {
    let sequential_case = runtime_tuning_sequential_case();
    let parallel_cases = runtime_tuning_parallel_cases();
    let mut variants = vec![(
        "lambdify-baseline".to_string(),
        runtime_tuning_lambdify_config_for(matrix_backend),
    )];
    for toolchain in RuntimeTuningToolchain::variants() {
        variants.push((
            runtime_tuning_variant_label(toolchain, &sequential_case),
            runtime_tuning_honest_cold_aot_config_for(toolchain, &sequential_case, matrix_backend),
        ));
        for chunk_case in &parallel_cases {
            variants.push((
                runtime_tuning_variant_label(toolchain, chunk_case),
                runtime_tuning_honest_cold_aot_config_for(toolchain, chunk_case, matrix_backend),
            ));
        }
    }
    variants
}

fn runtime_tuning_cold_variants() -> Vec<(
    String,
    crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
)> {
    runtime_tuning_cold_variants_for(MatrixBackend::SparseCol)
}

fn encode_isolated_tuning_solution(solution: &DMatrix<f64>) -> String {
    solution
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("\t")
}

fn decode_isolated_tuning_solution(line: &str) -> Vec<f64> {
    line.strip_prefix(ISOLATED_TUNING_SOLUTION_MARKER)
        .expect("isolated tuning solution marker should be present")
        .trim_start_matches('\t')
        .split('\t')
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|err| panic!("isolated tuning solution value invalid: {err}"))
        })
        .collect()
}

fn encode_isolated_cold_metrics(metrics: &IsolatedColdMetrics) -> String {
    [
        metrics.total_timer_ms,
        metrics.symbolic_ms,
        metrics.linear_ms,
        metrics.jac_ms,
        metrics.fun_ms,
        metrics.cb_residual_values_ms,
        metrics.cb_jacobian_values_ms,
        metrics.cb_jacobian_assembly_ms,
        metrics.residual_actual_jobs,
        metrics.sparse_jacobian_actual_jobs,
        metrics.initial_symbolic_jacobian_ms,
        metrics.post_build_rebind_ms,
        metrics.aot_artifact_ms,
        metrics.aot_materialize_ms,
        metrics.aot_compile_link_ms,
        metrics.aot_register_link_ms,
    ]
    .into_iter()
    .map(|value| value.to_string())
    .chain(
        [
            metrics.iterations,
            metrics.linear_solves,
            metrics.jacobian_rebuilds,
            metrics.refinements,
            metrics.residual_calls,
            metrics.jacobian_calls,
        ]
        .into_iter()
        .map(|value| value.to_string()),
    )
    .collect::<Vec<_>>()
    .join("\t")
}

fn decode_isolated_cold_metrics(line: &str) -> IsolatedColdMetrics {
    let values = line
        .strip_prefix(ISOLATED_TUNING_METRICS_MARKER)
        .expect("isolated tuning metrics marker should be present")
        .trim_start_matches('\t')
        .split('\t')
        .map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|err| panic!("isolated cold metric invalid: {err}"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        values.len(),
        22,
        "isolated cold metrics payload should contain every stage and trajectory field"
    );
    IsolatedColdMetrics {
        total_timer_ms: values[0],
        symbolic_ms: values[1],
        linear_ms: values[2],
        jac_ms: values[3],
        fun_ms: values[4],
        cb_residual_values_ms: values[5],
        cb_jacobian_values_ms: values[6],
        cb_jacobian_assembly_ms: values[7],
        residual_actual_jobs: values[8],
        sparse_jacobian_actual_jobs: values[9],
        initial_symbolic_jacobian_ms: values[10],
        post_build_rebind_ms: values[11],
        aot_artifact_ms: values[12],
        aot_materialize_ms: values[13],
        aot_compile_link_ms: values[14],
        aot_register_link_ms: values[15],
        iterations: values[16] as usize,
        linear_solves: values[17] as usize,
        jacobian_rebuilds: values[18] as usize,
        refinements: values[19] as usize,
        residual_calls: values[20] as usize,
        jacobian_calls: values[21] as usize,
    }
}

fn story_protocol(n_steps: usize, repetitions: usize) -> AotStoryProtocol {
    let protocol = AotStoryProtocol::from_env(n_steps, repetitions);
    protocol
        .validate()
        .unwrap_or_else(|error| panic!("invalid AOT story protocol: {error}"));
    protocol
}

fn cold_cooldown_ms() -> u64 {
    story_protocol(2, 3).cold_cooldown_ms
}

fn clean_cold_artifacts_enabled() -> bool {
    story_protocol(2, 3).clean_artifacts
}

fn remove_generated_aot_builds_for_child(child_pid: u32) {
    if !clean_cold_artifacts_enabled() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("generated-aot");
    let prefix = format!("build-{child_pid}-");
    let Ok(problem_dirs) = fs::read_dir(&root) else {
        return;
    };
    for problem_dir in problem_dirs.flatten() {
        let Ok(build_dirs) = fs::read_dir(problem_dir.path()) else {
            continue;
        };
        for build_dir in build_dirs.flatten() {
            let name = build_dir.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) {
                fs::remove_dir_all(build_dir.path()).unwrap_or_else(|err| {
                    panic!(
                        "failed to remove isolated child AOT artifact directory {}: {err}",
                        build_dir.path().display()
                    )
                });
            }
        }
    }
}

fn isolated_tuning_matrix_backend() -> MatrixBackend {
    match std::env::var(ISOLATED_TUNING_MATRIX_ENV).as_deref() {
        Ok("banded") => MatrixBackend::Banded,
        Ok("sparse") | Ok("") | Err(_) => MatrixBackend::SparseCol,
        Ok(other) => panic!("isolated tuning matrix backend {other:?} is unsupported"),
    }
}

fn run_isolated_tuning_child(index: usize, n_steps: usize) {
    let matrix_backend = isolated_tuning_matrix_backend();
    let variants = runtime_tuning_cold_variants_for(matrix_backend);
    let (label, config) = variants
        .get(index)
        .unwrap_or_else(|| panic!("isolated tuning child index {index} is invalid"));
    let mut solver = make_combustion_solver(n_steps, config.clone());
    let elapsed_ms =
        solve_honest_user_e2e_and_measure(&mut solver, &format!("isolated-cold-{label}-{n_steps}"))
            .expect("isolated cold tuning variant should solve");
    let solution = solver
        .get_result()
        .expect("isolated cold tuning variant should produce a solution");
    let metrics = isolated_cold_metrics_from_solver(&solver);
    println!("{ISOLATED_TUNING_PID_MARKER}\t{}", std::process::id());
    println!("{ISOLATED_TUNING_TIME_MARKER}\t{elapsed_ms}");
    println!(
        "{ISOLATED_TUNING_METRICS_MARKER}\t{}",
        encode_isolated_cold_metrics(&metrics)
    );
    println!(
        "{ISOLATED_TUNING_SOLUTION_MARKER}\t{}",
        encode_isolated_tuning_solution(&solution)
    );
}

fn solve_isolated_cold_tuning_variant_for(
    index: usize,
    n_steps: usize,
    matrix_backend: MatrixBackend,
    child_test_name: &str,
) -> IsolatedColdObservation {
    let protocol = story_protocol(n_steps, 3);
    let mut command =
        Command::new(std::env::current_exe().expect("test executable should resolve"));
    command
        .arg("--exact")
        .arg(child_test_name)
        .arg("--ignored")
        .arg("--nocapture")
        .env(ISOLATED_TUNING_CHILD_INDEX_ENV, index.to_string())
        .env(ISOLATED_TUNING_CHILD_STEPS_ENV, n_steps.to_string())
        .env(
            ISOLATED_TUNING_MATRIX_ENV,
            match matrix_backend {
                MatrixBackend::Banded => "banded",
                MatrixBackend::SparseCol => "sparse",
                other => panic!("isolated tuning matrix backend {other:?} is unsupported"),
            },
        );
    if protocol.worker_threads > 0 {
        command.env("RAYON_NUM_THREADS", protocol.worker_threads.to_string());
    }
    let output = command
        .output()
        .expect("isolated cold tuning child should launch");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "isolated cold tuning child {index} failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let elapsed_ms = stdout
        .lines()
        .find(|line| line.starts_with(ISOLATED_TUNING_TIME_MARKER))
        .and_then(|line| {
            line.strip_prefix(ISOLATED_TUNING_TIME_MARKER)
                .map(str::trim)
        })
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no time"));
    let metrics = stdout
        .lines()
        .find(|line| line.starts_with(ISOLATED_TUNING_METRICS_MARKER))
        .map(decode_isolated_cold_metrics)
        .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no metrics"));
    let solution = stdout
        .lines()
        .find(|line| line.starts_with(ISOLATED_TUNING_SOLUTION_MARKER))
        .map(decode_isolated_tuning_solution)
        .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no solution"));
    let child_pid = stdout
        .lines()
        .find(|line| line.starts_with(ISOLATED_TUNING_PID_MARKER))
        .and_then(|line| line.strip_prefix(ISOLATED_TUNING_PID_MARKER).map(str::trim))
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or_else(|| panic!("isolated cold tuning child {index} emitted no pid"));
    remove_generated_aot_builds_for_child(child_pid);
    let cooldown_ms = cold_cooldown_ms();
    if cooldown_ms > 0 {
        thread::sleep(Duration::from_millis(cooldown_ms));
    }
    IsolatedColdObservation {
        elapsed_ms,
        solution,
        metrics,
    }
}

fn solve_isolated_cold_tuning_variant(index: usize, n_steps: usize) -> IsolatedColdObservation {
    solve_isolated_cold_tuning_variant_for(
        index,
        n_steps,
        MatrixBackend::SparseCol,
        "numerical::BVP_Damp::test_aot_diagnostics::tests::aot_combustion_parallel_tuning_reports_runtime_table",
    )
}

fn isolated_tuning_child_test_name(matrix_backend: MatrixBackend) -> &'static str {
    match matrix_backend {
        MatrixBackend::Banded => {
            "numerical::BVP_Damp::test_aot_diagnostics::tests::aot_banded_apple_to_apple_release_protocol"
        }
        MatrixBackend::SparseCol => {
            "numerical::BVP_Damp::test_aot_diagnostics::tests::aot_combustion_parallel_tuning_reports_runtime_table"
        }
        other => panic!("isolated tuning matrix backend {other:?} is unsupported"),
    }
}

fn print_isolated_cold_raw_observation(
    repetition: usize,
    label: &str,
    observation: &IsolatedColdObservation,
) {
    println!(
        "[AOT isolated cold raw] rep={} config={} total_ms={:.3} symbolic_ms={:.3} initial_sym_jac_ms={:.3} materialize_ms={:.3} compile_link_ms={:.3} res_jobs={:.3} jac_jobs={:.3} iterations={} linear_solves={} jacobian_rebuilds={} refinements={} residual_calls={} jacobian_calls={}",
        repetition + 1,
        label,
        observation.elapsed_ms,
        observation.metrics.symbolic_ms,
        observation.metrics.initial_symbolic_jacobian_ms,
        observation.metrics.aot_materialize_ms,
        observation.metrics.aot_compile_link_ms,
        observation.metrics.residual_actual_jobs,
        observation.metrics.sparse_jacobian_actual_jobs,
        observation.metrics.iterations,
        observation.metrics.linear_solves,
        observation.metrics.jacobian_rebuilds,
        observation.metrics.refinements,
        observation.metrics.residual_calls,
        observation.metrics.jacobian_calls,
    );
}

fn run_combustion_tuning_scenario(
    n_steps: usize,
    repetitions: usize,
    scenario_label: &str,
) -> Result<(), BvpBackendIntegrationError> {
    run_combustion_tuning_scenario_for_matrix(
        n_steps,
        repetitions,
        scenario_label,
        MatrixBackend::SparseCol,
    )
}

fn run_combustion_tuning_scenario_for_matrix(
    n_steps: usize,
    repetitions: usize,
    scenario_label: &str,
    matrix_backend: MatrixBackend,
) -> Result<(), BvpBackendIntegrationError> {
    let toolchains = RuntimeTuningToolchain::variants();
    let sequential_case = runtime_tuning_sequential_case();
    let parallel_cases = runtime_tuning_parallel_cases();
    let mut labels = Vec::with_capacity(1 + toolchains.len() * (1 + parallel_cases.len()));
    labels.push("lambdify-baseline".to_string());
    for toolchain in toolchains {
        labels.push(runtime_tuning_variant_label(toolchain, &sequential_case));
        for chunk_case in &parallel_cases {
            labels.push(runtime_tuning_variant_label(toolchain, chunk_case));
        }
    }

    let mut samples = Vec::with_capacity(labels.len() * repetitions);
    println!(
        "[AOT combustion tuning map] isolated cold protocol: cooldown_ms={}, cleanup_child_artifacts={}",
        cold_cooldown_ms(),
        clean_cold_artifacts_enabled()
    );
    for repetition in 0..repetitions {
        println!(
            "[AOT combustion tuning map] starting repetition {}/{}",
            repetition + 1,
            repetitions
        );
        let lambdify_config = runtime_tuning_lambdify_config_for(matrix_backend);
        let lambdify_cold = solve_isolated_cold_tuning_variant_for(
            0,
            n_steps,
            matrix_backend,
            isolated_tuning_child_test_name(matrix_backend),
        );
        print_isolated_cold_raw_observation(repetition, "lambdify-baseline", &lambdify_cold);
        let mut lambdify = make_combustion_solver(n_steps, lambdify_config);
        let (lambdify_prepare_ms, lambdify_solve_ms) = solve_with_lambdify_and_measure(
            &mut lambdify,
            &format!("combustion-lambdify-baseline-{n_steps}-rep{repetition}"),
        )?;
        let lambdify_solution = lambdify
            .get_result()
            .expect("lambdify baseline for AOT combustion tuning should produce a solution");
        assert!(
            lambdify_solution.iter().all(|value| value.is_finite()),
            "lambdify baseline for AOT combustion tuning should remain finite"
        );
        let lambdify_honest_diff = lambdify_solution
            .iter()
            .zip(lambdify_cold.solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(
            lambdify_honest_diff < 1.0e-4,
            "isolated Lambdify cold result disagrees with prepared reference by {lambdify_honest_diff}"
        );
        samples.push(runtime_tuning_sample_from_solver(
            "lambdify-baseline",
            n_steps,
            lambdify_cold.elapsed_ms,
            f64::NAN,
            lambdify_prepare_ms,
            lambdify_solve_ms,
            f64::NAN,
            0.0,
            &lambdify,
            lambdify_cold.metrics,
        ));

        let mut cold_variant_index = 1usize;
        for toolchain in toolchains {
            let sequential_label = runtime_tuning_variant_label(toolchain, &sequential_case);
            let sequential_config =
                runtime_tuning_aot_config_for(toolchain, &sequential_case, matrix_backend);
            let mut sequential = make_combustion_solver(n_steps, sequential_config);
            let (_seq_guard, seq_bootstrap_ms, seq_solve_ms) = solve_with_aot_and_measure(
                &mut sequential,
                &format!("combustion-{sequential_label}-{n_steps}-rep{repetition}"),
            )?;
            let sequential_solution = sequential.get_result().unwrap_or_else(|| {
                panic!("{sequential_label}: AOT tuning case should produce a solution")
            });
            assert!(
                sequential_solution.iter().all(|value| value.is_finite()),
                "{sequential_label}: AOT tuning solution should remain finite"
            );
            let seq_max_diff_vs_ref = lambdify_solution
                .iter()
                .zip(sequential_solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            assert!(
                seq_max_diff_vs_ref < 1.0e-4,
                "{sequential_label}: AOT tuning disagreement with lambdify reference {seq_max_diff_vs_ref} is too large"
            );
            let seq_cold = solve_isolated_cold_tuning_variant_for(
                cold_variant_index,
                n_steps,
                matrix_backend,
                isolated_tuning_child_test_name(matrix_backend),
            );
            cold_variant_index += 1;
            print_isolated_cold_raw_observation(repetition, &sequential_label, &seq_cold);
            let seq_honest_diff_vs_ref = lambdify_solution
                .iter()
                .zip(seq_cold.solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            assert!(
                seq_honest_diff_vs_ref < 1.0e-4,
                "{sequential_label}: honest AOT e2e disagreement with lambdify reference {seq_honest_diff_vs_ref} is too large"
            );
            samples.push(runtime_tuning_sample_from_solver(
                sequential_label.clone(),
                n_steps,
                seq_cold.elapsed_ms,
                1.0,
                seq_bootstrap_ms,
                seq_solve_ms,
                1.0,
                seq_max_diff_vs_ref,
                &sequential,
                seq_cold.metrics,
            ));

            for chunk_case in &parallel_cases {
                let label = runtime_tuning_variant_label(toolchain, chunk_case);
                let generated_backend_config =
                    runtime_tuning_aot_config_for(toolchain, chunk_case, matrix_backend);
                let mut solver = make_combustion_solver(n_steps, generated_backend_config);
                let (_guard, bootstrap_ms, solve_ms) = solve_with_aot_and_measure(
                    &mut solver,
                    &format!("combustion-{label}-{n_steps}-rep{repetition}"),
                )?;
                let solution = solver.get_result().unwrap_or_else(|| {
                    panic!("{label}: AOT tuning case should produce a solution")
                });
                assert!(
                    solution.iter().all(|value| value.is_finite()),
                    "{label}: AOT combustion tuning solution should remain finite"
                );
                let max_diff_vs_ref = lambdify_solution
                    .iter()
                    .zip(solution.iter())
                    .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                    .fold(0.0, f64::max);
                assert!(
                    max_diff_vs_ref < 1.0e-4,
                    "{label}: AOT combustion tuning disagreement with lambdify reference {max_diff_vs_ref} is too large"
                );
                let cold = solve_isolated_cold_tuning_variant_for(
                    cold_variant_index,
                    n_steps,
                    matrix_backend,
                    isolated_tuning_child_test_name(matrix_backend),
                );
                cold_variant_index += 1;
                print_isolated_cold_raw_observation(repetition, &label, &cold);
                let honest_max_diff_vs_ref = lambdify_solution
                    .iter()
                    .zip(cold.solution.iter())
                    .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                    .fold(0.0, f64::max);
                assert!(
                    honest_max_diff_vs_ref < 1.0e-4,
                    "{label}: honest AOT e2e disagreement with lambdify reference {honest_max_diff_vs_ref} is too large"
                );
                samples.push(runtime_tuning_sample_from_solver(
                    label,
                    n_steps,
                    cold.elapsed_ms,
                    seq_cold.elapsed_ms / cold.elapsed_ms.max(f64::EPSILON),
                    bootstrap_ms,
                    solve_ms,
                    seq_solve_ms / solve_ms,
                    max_diff_vs_ref,
                    &solver,
                    cold.metrics,
                ));
            }
        }
    }

    let rows = summarize_runtime_tuning_samples(&labels, &samples);
    print_runtime_tuning_summary_table(scenario_label, n_steps, repetitions, &rows);

    let winner = rows
        .iter()
        .min_by(|lhs, rhs| lhs.solve_ms.mean.total_cmp(&rhs.solve_ms.mean))
        .expect("runtime tuning table should contain at least one row");
    println!(
        "[AOT runtime tuning winner] scenario={}, config={}, n_steps={}, runs={}, solve_ms_mean={:.3}, speedup_vs_seq_mean={:.3}, manual_bootstrap_ms_mean={:.3}",
        scenario_label,
        winner.label,
        winner.n_steps,
        winner.runs,
        winner.solve_ms.mean,
        winner.speedup_vs_seq.mean,
        winner.bootstrap_ms.mean,
    );
    let cold_winner = rows
        .iter()
        .min_by(|lhs, rhs| {
            lhs.honest_user_e2e_ms
                .mean
                .total_cmp(&rhs.honest_user_e2e_ms.mean)
        })
        .expect("cold wall-clock tuning table should contain at least one row");
    println!(
        "[AOT isolated cold wall-clock winner] scenario={}, config={}, n_steps={}, runs={}, honest_user_e2e_ms_mean={:.3}",
        scenario_label,
        cold_winner.label,
        cold_winner.n_steps,
        cold_winner.runs,
        cold_winner.honest_user_e2e_ms.mean,
    );
    Ok(())
}

fn make_example_solver(
    equation: &NonlinEquation,
    n_steps: usize,
    strategy_params: Option<SolverParams>,
    generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
) -> NRBVP {
    let eq_system = equation.setup();
    let values = equation.values();
    let mut border_conditions = equation.boundary_conditions();
    let bounds = equation.Bounds();
    let rel_tolerance = equation.rel_tolerance();
    let (t0, t_end) = if matches!(equation, NonlinEquation::LaneEmden5) {
        // The Lane-Emden equation contains the removable singular term
        // `-2*z/x`. Keep this AOT fixture finite until the solver has an
        // explicit limiting-value policy for x=0.
        let x0: f64 = 1.0e-6;
        let y0 = (1.0 + x0 * x0 / 3.0).powf(-0.5);
        let z0 = -x0 / 3.0 * (1.0 + x0 * x0 / 3.0).powf(-1.5);
        border_conditions.insert("y".to_string(), vec![(0usize, y0)]);
        border_conditions.insert("z".to_string(), vec![(0usize, z0)]);
        (x0, equation.span(None, None).1)
    } else {
        equation.span(None, None)
    };
    let initial_guess = uniform_initial_guess(values.len(), n_steps, 0.7);
    let options = DampedSolverOptions::sparse_damped()
        .with_strategy_params(strategy_params)
        .with_abs_tolerance(1e-8)
        .with_rel_tolerance(rel_tolerance)
        .with_max_iterations(40)
        .with_bounds(bounds)
        .with_generated_backend_config(generated_backend_config);

    let mut solver = NRBVP::new_with_options(
        eq_system,
        initial_guess,
        values,
        "x".to_string(),
        border_conditions,
        t0,
        t_end,
        n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn make_combustion_solver(
    n_steps: usize,
    generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
) -> NRBVP {
    let unknowns_str: Vec<&str> = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
    let unknowns: Vec<Expr> = Expr::parse_vector_expression(unknowns_str.clone());
    let teta = unknowns[0].clone();
    let q = unknowns[1].clone();
    let c0 = unknowns[2].clone();
    let j0 = unknowns[3].clone();
    let j1 = unknowns[5].clone();

    let q_heat = 3000.0 * 1e3 * 0.034;
    let dt = 600.0;
    let t_scale = 600.0;
    let l: f64 = 3e-4;
    let m0 = 34.2 / 1000.0;
    let lambda = 0.07;
    let p = 2e6;
    let tm = 1500.0;
    let c1_0 = 1.0;
    let t_initial = 1000.0;
    let pe_q = 0.0090168;
    let d_ro = 2.88e-4;
    let pe_d = 1.50e-3;
    let ro_m_ = m0 * p / (8.314 * tm);

    let dt_sym = Expr::Const(dt);
    let t_scale_sym = Expr::Const(t_scale);
    let lambda_sym = Expr::Const(lambda);
    let q_heat = Expr::Const(q_heat);
    let a = Expr::Const(1.3e5);
    let e = Expr::Const(5000.0 * 4.184);
    let m = Expr::Const(m0);
    let r_g = Expr::Const(8.314);
    let ro_m = Expr::Const(ro_m_);
    let qm = Expr::Const(l.powf(2.0) / t_scale);
    let qs = Expr::Const(l.powf(2.0));
    let pe_q_sym = Expr::Const(pe_q);
    let ro_d = vec![Expr::Const(d_ro), Expr::Const(d_ro)];
    let pe_d = vec![Expr::Const(pe_d), Expr::Const(pe_d)];
    let minus = Expr::Const(-1.0);
    let m_reag = Expr::Const(0.342);

    let rate = a
        * Expr::exp(-e / (r_g * (teta.clone() * t_scale_sym + dt_sym)))
        * c0.clone()
        * (ro_m.clone() / m_reag.clone());
    let eq_t = q.clone() / lambda_sym;
    let eq_q = q * pe_q_sym - q_heat * rate.clone() * qm;
    let eq_c0 = j0.clone() / ro_d[0].clone();
    let eq_j0 = j0 * pe_d[0].clone()
        - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone();
    let eq_c1 = j1.clone() / ro_d[1].clone();
    let eq_j1 = j1 * pe_d[1].clone() - (m.clone() * rate * ro_m / m) * qs;
    let eqs = vec![eq_t, eq_q, eq_c0, eq_j0, eq_c1, eq_j1];

    let boundary_conditions = HashMap::from([
        ("Teta".to_string(), vec![(0, (t_initial - dt) / t_scale)]),
        ("q".to_string(), vec![(1, 1e-10)]),
        ("C0".to_string(), vec![(0, c1_0)]),
        ("J0".to_string(), vec![(1, 1e-7)]),
        ("C1".to_string(), vec![(0, 1e-3)]),
        ("J1".to_string(), vec![(1, 1e-7)]),
    ]);
    let bounds = HashMap::from([
        ("Teta".to_string(), (0.0, 10.0)),
        ("q".to_string(), (-1e20, 1e20)),
        ("C0".to_string(), (0.0, 1.5)),
        ("J0".to_string(), (-1e2, 1e2)),
        ("C1".to_string(), (0.0, 1.5)),
        ("J1".to_string(), (-1e2, 1e2)),
    ]);
    let rel_tolerance = HashMap::from([
        ("Teta".to_string(), 1e-5),
        ("q".to_string(), 1e-5),
        ("C0".to_string(), 1e-5),
        ("J0".to_string(), 1e-5),
        ("C1".to_string(), 1e-5),
        ("J1".to_string(), 1e-5),
    ]);
    let strategy_params = SolverParams {
        max_jac: Some(6),
        max_damp_iter: Some(6),
        damp_factor: Some(0.5),
        adaptive: None,
    };
    let options = DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(strategy_params))
        .with_abs_tolerance(1e-6)
        .with_rel_tolerance(rel_tolerance)
        .with_max_iterations(100)
        .with_bounds(bounds)
        .with_generated_backend_config(generated_backend_config)
        .with_loglevel(Some("none".to_string()));
    let initial_guess = uniform_initial_guess(unknowns_str.len(), n_steps, 0.99);

    let mut solver = NRBVP::new_with_options(
        eqs,
        initial_guess,
        unknowns_str.iter().map(|value| value.to_string()).collect(),
        "x".to_string(),
        boundary_conditions,
        0.0,
        1.0,
        n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn make_oscillator_solver(
    n_steps: usize,
    generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
) -> NRBVP {
    let eqs = vec![Expr::parse_expression("z"), Expr::parse_expression("-y")];
    let values = vec!["y".to_string(), "z".to_string()];
    let t0 = 0.0;
    let t_end = std::f64::consts::FRAC_PI_2;
    let h = (t_end - t0) / n_steps as f64;

    let mut guess = vec![0.0; values.len() * n_steps];
    for i in 0..n_steps {
        let x = t0 + (i as f64) * h;
        guess[i * values.len()] = x.sin();
        guess[i * values.len() + 1] = x.cos();
    }
    let initial_guess =
        DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

    let boundary_conditions = HashMap::from([
        ("y".to_string(), vec![(0usize, 0.0f64)]),
        ("z".to_string(), vec![(0usize, 1.0f64)]),
    ]);
    let bounds = HashMap::from([
        ("y".to_string(), (-1.2, 1.2)),
        ("z".to_string(), (-1.2, 1.2)),
    ]);
    let rel_tolerance = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);

    let options = DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(SolverParams::default()))
        .with_abs_tolerance(1e-8)
        .with_rel_tolerance(rel_tolerance)
        .with_max_iterations(60)
        .with_bounds(bounds)
        .with_generated_backend_config(generated_backend_config)
        .with_loglevel(Some("error".to_string()));

    let mut solver = NRBVP::new_with_options(
        eqs,
        initial_guess,
        values,
        "x".to_string(),
        boundary_conditions,
        t0,
        t_end,
        n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}
