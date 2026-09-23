fn whole_chunking() -> AotChunkingPolicy {
    AotChunkingPolicy::with_parts(
        Some(ResidualChunkingStrategy::Whole),
        Some(SparseChunkingStrategy::Whole),
    )
}

fn four_way_chunking() -> AotChunkingPolicy {
    AotChunkingPolicy::with_parts(
        Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
        Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
    )
}

fn forced_parallel_execution() -> AotExecutionPolicy {
    AotExecutionPolicy::Parallel(ParallelExecutorConfig {
        jobs_per_worker: 1,
        max_residual_jobs: Some(4),
        max_sparse_jobs: Some(4),
        fallback_policy: ParallelFallbackPolicy::Never,
    })
}

fn combustion_toolchain_chunking_variants() -> Vec<RaceVariant> {
    vec![
        RaceVariant {
            source: "Lambdify",
            matrix: "Sparse",
            variant: "AtomView",
            bootstrap_hint: "baseline",
            config: sparse_atomview_lambdify_baseline(),
        },
        RaceVariant {
            source: "Lambdify",
            matrix: "Banded",
            variant: "AtomView",
            bootstrap_hint: "baseline",
            config: banded_atomview_lambdify_baseline(),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "gcc",
            bootstrap_hint: "whole",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "gcc",
            bootstrap_hint: "chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "tcc",
            bootstrap_hint: "whole",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "tcc",
            bootstrap_hint: "chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "zig",
            bootstrap_hint: "whole",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Sparse",
            variant: "zig",
            bootstrap_hint: "chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "gcc",
            bootstrap_hint: "whole",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "gcc",
            bootstrap_hint: "chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "tcc",
            bootstrap_hint: "whole",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "tcc",
            bootstrap_hint: "chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "zig",
            bootstrap_hint: "whole",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
        },
        RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "zig",
            bootstrap_hint: "chunk4",
            config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
    ]
}

fn sparse_atomview_rust_aot_release() -> GeneratedBackendConfig {
    GeneratedBackendConfig::sparse_build_if_missing_release()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
        .with_aot_codegen_backend(AotCodegenBackend::Rust)
}

fn banded_atomview_rust_aot_release() -> GeneratedBackendConfig {
    GeneratedBackendConfig::banded_build_if_missing_release()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
        .with_aot_codegen_backend(AotCodegenBackend::Rust)
}

fn sparse_atomview_lambdify_baseline() -> GeneratedBackendConfig {
    GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
}

fn banded_atomview_lambdify_baseline() -> GeneratedBackendConfig {
    GeneratedBackendConfig::banded_lambdify_defaults()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
}

fn callback_probe_for_initial_guess(
    n_steps: usize,
    label: &str,
    vector_method: &str,
    config: GeneratedBackendConfig,
) -> CallbackProbe {
    println!("[BVP Damp debug] bootstrapping {label}");
    let _ = io::stdout().flush();

    let probe_start = Instant::now();
    let bootstrap_start = Instant::now();
    let mut solver = make_combustion_solver(n_steps, config);
    solver
        .try_eq_generate(None, None)
        .unwrap_or_else(|err| panic!("{label}: generated backend bootstrap failed: {err:?}"));
    let bootstrap_ms = bootstrap_start.elapsed().as_secs_f64() * 1_000.0;

    let args = DVector::from_element(solver.values.len() * n_steps, 0.99);
    let typed_args = crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting(
        &args,
        vector_method.to_string(),
    );
    let residual_start = Instant::now();
    let residual = solver.fun.call(0.0, &*typed_args).to_DVectorType();
    let residual_ms = residual_start.elapsed().as_secs_f64() * 1_000.0;
    let jacobian_start = Instant::now();
    let jacobian = solver
        .jac
        .as_mut()
        .unwrap_or_else(|| panic!("{label}: generated backend did not provide a Jacobian"))
        .call(0.0, &*typed_args)
        .to_DMatrixType();
    let jacobian_ms = jacobian_start.elapsed().as_secs_f64() * 1_000.0;
    let total_probe_ms = probe_start.elapsed().as_secs_f64() * 1_000.0;

    assert!(
        residual.iter().all(|value| value.is_finite()),
        "{label}: residual callback produced non-finite values"
    );
    assert!(
        jacobian.iter().all(|value| value.is_finite()),
        "{label}: Jacobian callback produced non-finite values"
    );

    println!(
        "[BVP Damp debug] {label}: residual_len={}, jacobian={}x{}",
        residual.len(),
        jacobian.nrows(),
        jacobian.ncols()
    );
    CallbackProbe {
        residual: residual.iter().copied().collect(),
        jacobian,
        total_probe_ms,
        bootstrap_ms,
        residual_ms,
        jacobian_ms,
    }
}

fn max_slice_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "callback outputs must have identical lengths"
    );
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

#[derive(Clone)]
struct CallbackEquivalenceVariant {
    matrix: &'static str,
    toolchain: &'static str,
    whole_config: GeneratedBackendConfig,
    chunk4_config: GeneratedBackendConfig,
}

struct CallbackEquivalenceRow {
    matrix: &'static str,
    toolchain: &'static str,
    comparison: &'static str,
    residual_diff: f64,
    jacobian_diff: f64,
    status: String,
}

struct CallbackProbe {
    residual: Vec<f64>,
    jacobian: DMatrix<f64>,
    total_probe_ms: f64,
    bootstrap_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
}

struct CallbackProbeStatsRow {
    matrix: &'static str,
    toolchain: &'static str,
    mode: &'static str,
    total_probe_ms: f64,
    bootstrap_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    residual_calls: usize,
    jacobian_calls: usize,
    residual_len: usize,
    jac_rows: usize,
    jac_cols: usize,
    status: String,
}

fn callback_equivalence_filter_matches(variant: &CallbackEquivalenceVariant, filter: &str) -> bool {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return true;
    }
    format!("{} {}", variant.matrix, variant.toolchain)
        .to_ascii_lowercase()
        .contains(&filter)
}

fn apply_optional_callback_equivalence_filter(
    variants: &[CallbackEquivalenceVariant],
) -> Vec<CallbackEquivalenceVariant> {
    let Ok(filter) = std::env::var("BVP_AOT_CALLBACK_FILTER") else {
        return variants.to_vec();
    };
    let selected = variants
        .iter()
        .filter(|variant| callback_equivalence_filter_matches(variant, &filter))
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "BVP_AOT_CALLBACK_FILTER={filter:?} did not match any callback-equivalence variant"
    );
    println!(
        "[BVP Damp debug] BVP_AOT_CALLBACK_FILTER={filter:?}: running {}/{} variants",
        selected.len(),
        variants.len()
    );
    selected
}

fn callback_equivalence_variants() -> Vec<CallbackEquivalenceVariant> {
    vec![
        CallbackEquivalenceVariant {
            matrix: "Sparse",
            toolchain: "gcc",
            whole_config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        CallbackEquivalenceVariant {
            matrix: "Sparse",
            toolchain: "tcc",
            whole_config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        CallbackEquivalenceVariant {
            matrix: "Sparse",
            toolchain: "zig",
            whole_config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        CallbackEquivalenceVariant {
            matrix: "Sparse",
            toolchain: "rust",
            whole_config: release_matrix_config(
                sparse_atomview_rust_aot_release(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                sparse_atomview_rust_aot_release(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        CallbackEquivalenceVariant {
            matrix: "Banded",
            toolchain: "gcc",
            whole_config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        CallbackEquivalenceVariant {
            matrix: "Banded",
            toolchain: "tcc",
            whole_config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        CallbackEquivalenceVariant {
            matrix: "Banded",
            toolchain: "zig",
            whole_config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
        CallbackEquivalenceVariant {
            matrix: "Banded",
            toolchain: "rust",
            whole_config: release_matrix_config(
                banded_atomview_rust_aot_release(),
                whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            chunk4_config: release_matrix_config(
                banded_atomview_rust_aot_release(),
                four_way_chunking(),
                forced_parallel_execution(),
            ),
        },
    ]
}

fn callback_lambdify_baseline_config(matrix: &str) -> GeneratedBackendConfig {
    match matrix {
        "Sparse" => sparse_atomview_lambdify_baseline(),
        "Banded" => banded_atomview_lambdify_baseline(),
        other => panic!("unsupported callback-equivalence matrix {other}"),
    }
}

fn callback_vector_method(matrix: &str) -> &'static str {
    match matrix {
        "Sparse" => "Sparse",
        "Banded" => "Banded",
        other => panic!("unsupported callback-equivalence matrix {other}"),
    }
}

fn callback_diff_status(residual_diff: f64, jacobian_diff: f64) -> String {
    if residual_diff < 1.0e-10 && jacobian_diff < 1.0e-10 {
        "ok".to_string()
    } else {
        "diff_exceeded".to_string()
    }
}

fn callback_diff_row(
    matrix: &'static str,
    toolchain: &'static str,
    comparison: &'static str,
    lhs: &CallbackProbe,
    rhs: &CallbackProbe,
) -> CallbackEquivalenceRow {
    let residual_diff = max_slice_abs_diff(lhs.residual.as_slice(), rhs.residual.as_slice());
    let jacobian_diff = max_slice_abs_diff(lhs.jacobian.as_slice(), rhs.jacobian.as_slice());
    CallbackEquivalenceRow {
        matrix,
        toolchain,
        comparison,
        residual_diff,
        jacobian_diff,
        status: callback_diff_status(residual_diff, jacobian_diff),
    }
}

fn callback_probe_stats_row(
    matrix: &'static str,
    toolchain: &'static str,
    mode: &'static str,
    probe: &CallbackProbe,
    status: String,
) -> CallbackProbeStatsRow {
    CallbackProbeStatsRow {
        matrix,
        toolchain,
        mode,
        total_probe_ms: probe.total_probe_ms,
        bootstrap_ms: probe.bootstrap_ms,
        residual_ms: probe.residual_ms,
        jacobian_ms: probe.jacobian_ms,
        residual_calls: 1,
        jacobian_calls: 1,
        residual_len: probe.residual.len(),
        jac_rows: probe.jacobian.nrows(),
        jac_cols: probe.jacobian.ncols(),
        status,
    }
}

fn print_callback_equivalence_table(rows: &[CallbackEquivalenceRow]) {
    println!("[BVP Damp debug] AtomView AOT whole-vs-chunk4 callback correctness matrix");
    println!("matrix | toolchain | comparison           | residual_diff | jacobian_diff | status");
    println!("{}", "-".repeat(108));
    for row in rows {
        println!(
            "{:<6} | {:<9} | {:<20} | {:>13.6e} | {:>13.6e} | {}",
            row.matrix,
            row.toolchain,
            row.comparison,
            row.residual_diff,
            row.jacobian_diff,
            row.status
        );
    }
}

fn print_callback_probe_stats_table(rows: &[CallbackProbeStatsRow]) {
    println!();
    println!(
        "[BVP Damp debug] AtomView AOT callback probe statistics; all time columns are milliseconds"
    );
    println!(
        "matrix | toolchain | mode     | total_probe_ms | prepare_ms | residual_ms | jacobian_ms | res_calls | jac_calls | residual_len | jacobian_shape | status"
    );
    println!("{}", "-".repeat(174));
    for row in rows {
        println!(
            "{:<6} | {:<9} | {:<8} | {:>14.3} | {:>10.3} | {:>11.3} | {:>11.3} | {:>9} | {:>9} | {:>12} | {:>6}x{:<6} | {}",
            row.matrix,
            row.toolchain,
            row.mode,
            row.total_probe_ms,
            row.bootstrap_ms,
            row.residual_ms,
            row.jacobian_ms,
            row.residual_calls,
            row.jacobian_calls,
            row.residual_len,
            row.jac_rows,
            row.jac_cols,
            row.status
        );
    }
}
