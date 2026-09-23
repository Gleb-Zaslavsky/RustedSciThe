fn is_aot_environment_issue(err: &BvpBackendIntegrationError) -> bool {
    match err {
        BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
        | BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { .. }
        | BvpBackendIntegrationError::AutomaticAotBuildRequested { .. }
        | BvpBackendIntegrationError::AutomaticAotRebuildRequested { .. } => true,
        BvpBackendIntegrationError::AutomaticAotBuildFailed { message, .. } => {
            let msg = message.to_ascii_lowercase();
            msg.contains("permission denied")
                || msg.contains("not found")
                || msg.contains("failed to spawn")
                || msg.contains("status=some(1)")
                || msg.contains("toolchain")
        }
        BvpBackendIntegrationError::PipelinePanicked(message) => {
            let msg = message.to_ascii_lowercase();
            msg.contains("generatedbackendfailure")
                || msg.contains("permission denied")
                || msg.contains("failed to spawn")
                || msg.contains("status=some(1)")
                || msg.contains("toolchain")
        }
        _ => false,
    }
}

fn measure_sparse_runtime_callback_throughput(
    bundle: &mut BvpSparseSolverBundle,
    typed: &dyn crate::numerical::BVP_Damp::BVP_traits::VectorType,
    iters: usize,
) -> (f64, f64) {
    assert!(
        bundle.is_runtime_callable(),
        "callback throughput benchmark requires runtime-callable sparse bundle"
    );

    let residual_begin = Instant::now();
    for _ in 0..iters {
        let residual = bundle
            .residual_call(1.0, typed)
            .expect("runtime-callable sparse bundle must expose residual callback");
        black_box(residual);
    }
    let residual_ms = residual_begin.elapsed().as_secs_f64() * 1_000.0;

    let jacobian_begin = Instant::now();
    for _ in 0..iters {
        let jacobian = bundle
            .jacobian_call(1.0, typed)
            .expect("runtime-callable sparse bundle must expose jacobian callback");
        black_box(jacobian);
    }
    let jacobian_ms = jacobian_begin.elapsed().as_secs_f64() * 1_000.0;

    (residual_ms, jacobian_ms)
}

fn max_abs_vector_diff(lhs: &DVector<f64>, rhs: &DVector<f64>) -> f64 {
    assert_eq!(lhs.len(), rhs.len(), "vector lengths should match");
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs_matrix_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
    assert_eq!(lhs.shape(), rhs.shape(), "matrix shapes should match");
    let mut max_diff: f64 = 0.0;
    for row in 0..lhs.nrows() {
        for col in 0..lhs.ncols() {
            max_diff = max_diff.max((lhs[(row, col)] - rhs[(row, col)]).abs());
        }
    }
    max_diff
}

fn infer_bandwidth_from_dense_matrix(matrix: &DMatrix<f64>) -> (usize, usize) {
    let mut kl = 0usize;
    let mut ku = 0usize;
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let value = matrix[(row, col)];
            if value == 0.0 {
                continue;
            }
            if row >= col {
                kl = kl.max(row - col);
            } else {
                ku = ku.max(col - row);
            }
        }
    }
    (kl, ku)
}

fn banded_assembly_from_dense_matrix(dense: &DMatrix<f64>) -> BandedAssembly {
    let (kl, ku) = infer_bandwidth_from_dense_matrix(dense);
    let mut assembly = BandedAssembly::zeros(dense.nrows(), kl, ku)
        .expect("dense callback matrix should define a valid banded allocation");
    for row in 0..dense.nrows() {
        for col in 0..dense.ncols() {
            let value = dense[(row, col)];
            if value == 0.0 {
                continue;
            }
            assembly
                .set(row, col, value)
                .expect("dense callback entry should fit inside inferred band");
        }
    }
    assembly
}

fn dense_from_compact_banded(a: &Banded<f64>) -> DMatrix<f64> {
    let mut dense = DMatrix::zeros(a.n(), a.n());
    for col in 0..a.n() {
        let row_lo = col.saturating_sub(a.ku());
        let row_hi = (col + a.kl() + 1).min(a.n());
        for row in row_lo..row_hi {
            dense[(row, col)] = a[(row, col)];
        }
    }
    dense
}

fn dense_from_vecvec(matrix: &[Vec<f64>]) -> DMatrix<f64> {
    let n = matrix.len();
    let mut dense = DMatrix::zeros(n, n);
    for row in 0..n {
        for col in 0..n {
            dense[(row, col)] = matrix[row][col];
        }
    }
    dense
}

fn dense_to_sparse_col_mat(dense: &DMatrix<f64>) -> SparseColMat<usize, f64> {
    let mut triplets = Vec::new();
    for row in 0..dense.nrows() {
        for col in 0..dense.ncols() {
            let value = dense[(row, col)];
            if value != 0.0 {
                triplets.push(faer::sparse::Triplet::new(row, col, value));
            }
        }
    }
    SparseColMat::<usize, f64>::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
        .expect("dense matrix should convert to sparse triplets")
}

fn relative_dense_residual(a: &DMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut rmax = 0.0_f64;
    let mut bmax = 0.0_f64;
    for row in 0..a.nrows() {
        let mut ax = 0.0;
        for col in 0..a.ncols() {
            ax += a[(row, col)] * x[col];
        }
        rmax = rmax.max((ax - b[row]).abs());
        bmax = bmax.max(b[row].abs());
    }
    rmax / bmax.max(1.0)
}

fn vector_linf_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn relative_x_diff(x: &[f64], x_ref: &[f64]) -> f64 {
    let abs = x
        .iter()
        .zip(x_ref.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    abs / vector_linf_norm(x_ref).max(1.0)
}

fn fmt_metric(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.3e}")
    } else {
        "-".to_string()
    }
}

fn stats_count(stats: &DampedBvpStatistics, key: &str) -> usize {
    stats.counters.get(key).copied().unwrap_or(0)
}

fn stats_timer(stats: &DampedBvpStatistics, prefix: &str) -> String {
    stats
        .timers
        .iter()
        .find(|(key, _)| key.starts_with(prefix))
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| "-".to_string())
}

fn stats_timer_ms(stats: &DampedBvpStatistics, prefix: &str) -> f64 {
    stats
        .timers
        .iter()
        .find(|(key, _)| key.starts_with(prefix))
        .and_then(|(key, value)| timer_value_to_ms(key, value))
        .unwrap_or(f64::NAN)
}

fn stats_diagnostic_ms(stats: &DampedBvpStatistics, key: &str) -> f64 {
    stats
        .diagnostics
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn stats_diagnostic_usize(stats: &DampedBvpStatistics, key: &str) -> f64 {
    stats
        .diagnostics
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value as f64)
        .unwrap_or(0.0)
}

fn isolated_cold_metrics_from_solver(solver: &NRBVP) -> IsolatedColdMetrics {
    let stats = solver.get_statistics();
    let counters = stats.telemetry.counters.clone();
    IsolatedColdMetrics {
        total_timer_ms: stats_timer_ms(&stats, "time elapsed"),
        symbolic_ms: stats_timer_ms(&stats, "Symbolic Operations"),
        linear_ms: stats_timer_ms(&stats, "Linear System"),
        jac_ms: stats_timer_ms(&stats, "Jacobian"),
        fun_ms: stats_timer_ms(&stats, "Function"),
        cb_residual_values_ms: stats_timer_ms(&stats, "Callback Residual Values"),
        cb_jacobian_values_ms: stats_timer_ms(&stats, "Callback Jacobian Values"),
        cb_jacobian_assembly_ms: stats_timer_ms(&stats, "Callback Jacobian Matrix Assembly"),
        residual_actual_jobs: stats_diagnostic_usize(&stats, "aot.runtime.residual.actual_jobs"),
        sparse_jacobian_actual_jobs: stats_diagnostic_usize(
            &stats,
            "aot.runtime.sparse_jacobian.actual_jobs",
        ),
        initial_symbolic_jacobian_ms: stats_diagnostic_ms(
            &stats,
            "generated.handoff.initial.symbolic_jacobian_time_ms",
        ),
        post_build_rebind_ms: stats_diagnostic_ms(
            &stats,
            "generated.handoff.post_build_rebind_wall_ms",
        ),
        aot_artifact_ms: stats_diagnostic_ms(&stats, "generated.aot.artifact_wall_ms"),
        aot_materialize_ms: stats_diagnostic_ms(&stats, "generated.aot.materialize_ms"),
        aot_compile_link_ms: stats_diagnostic_ms(&stats, "generated.aot.compile_link_ms"),
        aot_register_link_ms: stats_diagnostic_ms(&stats, "generated.aot.register_link_ms"),
        iterations: counters.iterations as usize,
        linear_solves: counters.linear_solves as usize,
        jacobian_rebuilds: counters.jacobian_recalculations as usize,
        refinements: counters.grid_refinements as usize,
        residual_calls: counters.residual_calls as usize,
        jacobian_calls: counters.jacobian_requests as usize,
    }
}

fn timer_value_to_ms(key: &str, value: &str) -> Option<f64> {
    let duration = value
        .split(',')
        .next_back()
        .and_then(|part| part.trim().parse::<f64>().ok())?;
    let multiplier = if key.contains("ms") {
        1.0
    } else if key.contains("min") {
        60_000.0
    } else if key.contains('h') {
        3_600_000.0
    } else {
        1_000.0
    };
    Some(duration * multiplier)
}

fn runtime_tuning_aggregate(values: impl IntoIterator<Item = f64>) -> RuntimeTuningAggregate {
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return RuntimeTuningAggregate {
            mean: f64::NAN,
            stddev: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        };
    }

    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / count;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    RuntimeTuningAggregate {
        mean,
        stddev: variance.sqrt(),
        min,
        max,
    }
}

fn fmt_tuning_agg(value: RuntimeTuningAggregate) -> String {
    if value.mean.is_finite() {
        format!(
            "{:.3} +/- {:.3} [{:.3}, {:.3}]",
            value.mean, value.stddev, value.min, value.max
        )
    } else {
        "-".to_string()
    }
}

fn fmt_tuning_short(value: RuntimeTuningAggregate) -> String {
    if value.mean.is_finite() {
        format!("{:.3} +/- {:.3}", value.mean, value.stddev)
    } else {
        "-".to_string()
    }
}

fn fmt_tuning_exp(value: RuntimeTuningAggregate) -> String {
    if value.mean.is_finite() {
        format!("{:.3e} +/- {:.1e}", value.mean, value.stddev)
    } else {
        "-".to_string()
    }
}

fn solve_sparse_lu_for_rhs(matrix: &SparseColMat<usize, f64>, rhs: &[f64]) -> Vec<f64> {
    let lu = matrix
        .sp_lu()
        .expect("sparse LU factorization should succeed");
    let rhs_col = faer::Col::from_fn(rhs.len(), |index| rhs[index]);
    let solution = lu.solve(&rhs_col);
    solution.iter().copied().collect()
}

fn solve_banded_story_for_rhs(
    assembly: &BandedAssembly,
    n_steps: usize,
    rhs: &[f64],
    solver_choice: BandedStorySolver,
) -> BandedStorySolveMetrics {
    match solver_choice {
        BandedStorySolver::LegacyAuto => {
            let layout =
                NodeMajorLayout::new(n_steps, 6).expect("combustion node-major layout valid");
            let layout_label = format!("{}x{}", layout.n_blocks(), layout.block_size());
            match build_solver_for_system(
                LinearSystemRef::NodeMajorAssembly { assembly, layout },
                LinearSolverConfig {
                    policy: LinearSolverPolicy::Auto,
                    fallback: FallbackPolicy::ToFaerSparse,
                    iterative_refinement_steps: 0,
                },
            ) {
                Ok(solver) => {
                    let linear_solver = solver.backend_name().to_string();
                    let mut rhs_owned = rhs.to_vec();
                    match solver.solve_in_place(rhs_owned.as_mut_slice()) {
                        Ok(()) => BandedStorySolveMetrics {
                            linear_solver,
                            solution: Some(rhs_owned),
                            report: None,
                            layout: layout_label,
                            status: "ok".to_string(),
                        },
                        Err(err) => BandedStorySolveMetrics {
                            linear_solver,
                            solution: None,
                            report: None,
                            layout: layout_label,
                            status: format!("solve_failed({err:?})"),
                        },
                    }
                }
                Err(err) => BandedStorySolveMetrics {
                    linear_solver: solver_choice.label(),
                    solution: None,
                    report: None,
                    layout: layout_label,
                    status: format!("factorization_failed({err:?})"),
                },
            }
        }
        BandedStorySolver::ConsistentSuperblock {
            nodes_per_superblock,
            refinement_steps,
        } => {
            let layout = SuperBlockLayout::new(n_steps, 6, nodes_per_superblock)
                .expect("combustion superblock layout should be valid");
            assert!(
                layout.is_evenly_divisible(),
                "combustion superblock diagnostic requires an even node grouping"
            );
            let block = assembly.finalize_superblock_tridiagonal(&layout).expect(
                "combustion banded assembly should finalize into a uniform superblock chain",
            );
            let mut solver =
                match BlockTridiagonalLuConsistent::new(block.n_blocks(), block.block_size()) {
                    Ok(solver) => solver,
                    Err(err) => {
                        return BandedStorySolveMetrics {
                            linear_solver: solver_choice.label(),
                            solution: None,
                            report: None,
                            layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                            status: format!("factorization_failed({err:?})"),
                        };
                    }
                };
            if let Err(err) = solver.factor_from(&block) {
                return BandedStorySolveMetrics {
                    linear_solver: solver_choice.label(),
                    solution: None,
                    report: None,
                    layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                    status: format!("factorization_failed({err:?})"),
                };
            }
            let mut rhs_owned = rhs.to_vec();
            match solver.solve_in_place_with_iterative_refinement_report(
                &block,
                rhs_owned.as_mut_slice(),
                refinement_steps,
            ) {
                Ok(report) => {
                    let residual_ok = report.direct_relative_residual.is_finite()
                        && report.final_relative_residual.is_finite()
                        && report.final_relative_residual <= 1.0e-8;
                    BandedStorySolveMetrics {
                        linear_solver: solver_choice.label(),
                        solution: Some(rhs_owned),
                        report: Some(report),
                        layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                        // A successful factor/solve call is not enough to
                        // call an experimental structured solver correct.
                        // Keep numerically unacceptable reports visible as
                        // diagnostics instead of letting them become green.
                        status: if residual_ok {
                            "ok".to_string()
                        } else {
                            "diag".to_string()
                        },
                    }
                }
                Err(err) => BandedStorySolveMetrics {
                    linear_solver: solver_choice.label(),
                    solution: None,
                    report: None,
                    layout: format!("{}x{}", layout.n_blocks(), layout.block_size()),
                    status: format!("solve_failed({err:?})"),
                },
            }
        }
        BandedStorySolver::LapackStyle { refinement_steps } => {
            match build_solver_for_system(
                LinearSystemRef::BandedAssembly(assembly),
                LinearSolverConfig {
                    policy: LinearSolverPolicy::ForceBanded,
                    fallback: FallbackPolicy::Never,
                    iterative_refinement_steps: refinement_steps,
                },
            ) {
                Ok(solver) => {
                    let linear_solver = solver.backend_name().to_string();
                    let mut rhs_owned = rhs.to_vec();
                    match solver.solve_in_place(rhs_owned.as_mut_slice()) {
                        Ok(()) => BandedStorySolveMetrics {
                            linear_solver,
                            solution: Some(rhs_owned),
                            report: None,
                            layout: format!(
                                "n{} kl{} ku{}",
                                assembly.n(),
                                assembly.kl(),
                                assembly.ku()
                            ),
                            status: "ok".to_string(),
                        },
                        Err(err) => BandedStorySolveMetrics {
                            linear_solver,
                            solution: None,
                            report: None,
                            layout: format!(
                                "n{} kl{} ku{}",
                                assembly.n(),
                                assembly.kl(),
                                assembly.ku()
                            ),
                            status: format!("solve_failed({err:?})"),
                        },
                    }
                }
                Err(err) => BandedStorySolveMetrics {
                    linear_solver: solver_choice.label(),
                    solution: None,
                    report: None,
                    layout: format!("n{} kl{} ku{}", assembly.n(), assembly.kl(), assembly.ku()),
                    status: format!("factorization_failed({err:?})"),
                },
            }
        }
    }
}

fn flattened_initial_guess_state(solver: &NRBVP, expected_len: usize, label: &str) -> DVector<f64> {
    let dense = DVector::from_vec(solver.initial_guess.iter().cloned().collect());
    assert_eq!(
        dense.len(),
        expected_len,
        "{label}: flattened initial_guess length must match bundle variable count"
    );
    dense
}

fn runtime_vector_backend_for_matrix_backend(matrix_backend: MatrixBackend) -> &'static str {
    match matrix_backend {
        MatrixBackend::Banded => "Dense",
        _ => "Sparse",
    }
}

fn eval_solver_callback_state(
    solver: &mut NRBVP,
    args: &DVector<f64>,
    matrix_backend: MatrixBackend,
    label: &str,
) -> (
    DVector<f64>,
    DMatrix<f64>,
    Option<SparseColMat<usize, f64>>,
    Option<BandedMatrixType>,
) {
    let backend = runtime_vector_backend_for_matrix_backend(matrix_backend);
    let typed = &*Vectors_type_casting(args, backend.to_string());
    let residual = solver.fun.call(1.0, typed).to_DVectorType();
    let jacobian = solver
        .jac
        .as_mut()
        .unwrap_or_else(|| panic!("{label}: jacobian callback should be available"))
        .call(1.0, typed);
    let dense = jacobian.to_DMatrixType();

    match matrix_backend {
        MatrixBackend::Banded => {
            let banded = jacobian
                .as_any()
                .downcast_ref::<BandedMatrixType>()
                .unwrap_or_else(|| {
                    panic!("{label}: banded callback should produce BandedMatrixType")
                })
                .clone();
            (residual, dense, None, Some(banded))
        }
        _ => {
            let sparse = jacobian
                .as_any()
                .downcast_ref::<SparseColMat<usize, f64>>()
                .unwrap_or_else(|| panic!("{label}: sparse callback should produce SparseColMat"))
                .to_owned();
            (residual, dense, Some(sparse), None)
        }
    }
}

fn runtime_tuning_sample_from_solver(
    label: impl Into<String>,
    n_steps: usize,
    honest_user_e2e_ms: f64,
    honest_speedup_vs_seq: f64,
    bootstrap_ms: f64,
    solve_ms: f64,
    speedup_vs_seq: f64,
    max_diff_vs_ref: f64,
    solver: &NRBVP,
    cold: IsolatedColdMetrics,
) -> RuntimeTuningSample {
    let statistics = solver.get_statistics();
    RuntimeTuningSample {
        label: label.into(),
        n_steps,
        honest_user_e2e_ms,
        honest_speedup_vs_seq,
        bootstrap_ms,
        solve_ms,
        speedup_vs_seq,
        max_diff_vs_ref,
        total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
        symbolic_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
        linear_ms: stats_timer_ms(&statistics, "Linear System"),
        jac_ms: stats_timer_ms(&statistics, "Jacobian"),
        fun_ms: stats_timer_ms(&statistics, "Function"),
        cb_residual_values_ms: stats_timer_ms(&statistics, "Callback Residual Values"),
        cb_jacobian_values_ms: stats_timer_ms(&statistics, "Callback Jacobian Values"),
        cb_jacobian_assembly_ms: stats_timer_ms(&statistics, "Callback Jacobian Matrix Assembly"),
        iterations: stats_count(&statistics, "number of iterations"),
        linear_solves: stats_count(&statistics, "number of solving linear systems"),
        jac_rebuilds: stats_count(&statistics, "number of jacobians recalculations"),
        cold,
    }
}

fn summarize_runtime_tuning_samples(
    labels: &[String],
    samples: &[RuntimeTuningSample],
) -> Vec<RuntimeTuningSummary> {
    labels
        .iter()
        .map(|label| {
            let rows = samples
                .iter()
                .filter(|sample| sample.label.as_str() == label.as_str())
                .collect::<Vec<_>>();
            RuntimeTuningSummary {
                label: label.clone(),
                n_steps: rows.first().map(|sample| sample.n_steps).unwrap_or(0),
                runs: rows.len(),
                honest_user_e2e_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.honest_user_e2e_ms),
                ),
                honest_speedup_vs_seq: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.honest_speedup_vs_seq),
                ),
                bootstrap_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.bootstrap_ms),
                ),
                solve_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.solve_ms)),
                speedup_vs_seq: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.speedup_vs_seq),
                ),
                max_diff_vs_ref: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.max_diff_vs_ref),
                ),
                total_timer_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.total_timer_ms),
                ),
                symbolic_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.symbolic_ms)),
                linear_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.linear_ms)),
                jac_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.jac_ms)),
                fun_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.fun_ms)),
                cb_residual_values_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cb_residual_values_ms),
                ),
                cb_jacobian_values_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cb_jacobian_values_ms),
                ),
                cb_jacobian_assembly_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cb_jacobian_assembly_ms),
                ),
                iterations: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.iterations as f64),
                ),
                linear_solves: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.linear_solves as f64),
                ),
                jac_rebuilds: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.jac_rebuilds as f64),
                ),
                cold_total_timer_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.total_timer_ms),
                ),
                cold_symbolic_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.symbolic_ms),
                ),
                cold_linear_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.linear_ms),
                ),
                cold_jac_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.cold.jac_ms)),
                cold_fun_ms: runtime_tuning_aggregate(rows.iter().map(|sample| sample.cold.fun_ms)),
                cold_cb_residual_values_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.cb_residual_values_ms),
                ),
                cold_cb_jacobian_values_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.cb_jacobian_values_ms),
                ),
                cold_cb_jacobian_assembly_ms: runtime_tuning_aggregate(
                    rows.iter()
                        .map(|sample| sample.cold.cb_jacobian_assembly_ms),
                ),
                cold_residual_actual_jobs: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.residual_actual_jobs),
                ),
                cold_sparse_jacobian_actual_jobs: runtime_tuning_aggregate(
                    rows.iter()
                        .map(|sample| sample.cold.sparse_jacobian_actual_jobs),
                ),
                cold_initial_symbolic_jacobian_ms: runtime_tuning_aggregate(
                    rows.iter()
                        .map(|sample| sample.cold.initial_symbolic_jacobian_ms),
                ),
                cold_post_build_rebind_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.post_build_rebind_ms),
                ),
                cold_aot_artifact_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.aot_artifact_ms),
                ),
                cold_aot_materialize_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.aot_materialize_ms),
                ),
                cold_aot_compile_link_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.aot_compile_link_ms),
                ),
                cold_aot_register_link_ms: runtime_tuning_aggregate(
                    rows.iter().map(|sample| sample.cold.aot_register_link_ms),
                ),
            }
        })
        .collect()
}

fn print_runtime_tuning_summary_table(
    scenario_label: &str,
    n_steps: usize,
    repetitions: usize,
    rows: &[RuntimeTuningSummary],
) {
    println!();
    println!(
        "[AOT combustion tuning map] scenario={scenario_label}, n_steps={n_steps}, repetitions={repetitions}"
    );
    println!(
        "[AOT combustion tuning map] runner=manual_prelinked_runtime_tuning; solve_ms is the runtime Newton solve after callbacks are prepared."
    );
    println!(
        "[AOT combustion tuning map] manual_bootstrap_ms is diagnostic setup for this test only; do not read it as normal solver end-to-end time."
    );
    println!(
        "[AOT combustion tuning map] honest_user_e2e_ms is measured in a fresh child process around solver.try_solve(); AOT rows force RebuildAlways Release so codegen/build/link/Newton are included without registry/DLL carry-over."
    );
    println!("[AOT combustion tuning map] correctness summary");
    println!(
        "{:<30} | {:>7} | {:>4} | {:<18} | {:<18} | {:<18} | {:<18}",
        "config",
        "n_steps",
        "runs",
        "honest_user_e2e_ms",
        "solve_ms",
        "speedup_vs_seq",
        "max_diff_vs_ref"
    );
    println!("{}", "-".repeat(143));
    for row in rows {
        println!(
            "{:<30} | {:>7} | {:>4} | {:<18} | {:<18} | {:<18} | {:<18}",
            row.label,
            row.n_steps,
            row.runs,
            fmt_tuning_short(row.honest_user_e2e_ms),
            fmt_tuning_short(row.solve_ms),
            fmt_tuning_short(row.speedup_vs_seq),
            fmt_tuning_exp(row.max_diff_vs_ref),
        );
    }

    println!();
    println!(
        "[AOT combustion tuning map] honest wall-clock summary; all time columns are milliseconds"
    );
    println!(
        "note: this is the closest table to stopwatch timing from button press to finished result."
    );
    println!(
        "{:<30} | {:<18} | {:<18} | {:<18} | {:<31}",
        "config",
        "honest_user_e2e_ms",
        "honest_speedup",
        "runtime_solve_ms",
        "manual_bootstrap_ms mean+/-std [min,max]"
    );
    println!("{}", "-".repeat(132));
    for row in rows {
        println!(
            "{:<30} | {:<18} | {:<18} | {:<18} | {:<31}",
            row.label,
            fmt_tuning_short(row.honest_user_e2e_ms),
            fmt_tuning_short(row.honest_speedup_vs_seq),
            fmt_tuning_short(row.solve_ms),
            fmt_tuning_agg(row.bootstrap_ms),
        );
    }

    println!();
    println!(
        "[AOT combustion tuning map] isolated cold stage breakdown; every column in this table comes from the same child solve as honest_user_e2e_ms."
    );
    println!(
        "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18}",
        "config",
        "honest_e2e_ms",
        "solver_total_ms",
        "symbolic_ms",
        "initial_sym_jac",
        "artifact_ms",
        "materialize_ms",
        "compile_link_ms"
    );
    println!("{}", "-".repeat(188));
    for row in rows {
        println!(
            "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18}",
            row.label,
            fmt_tuning_short(row.honest_user_e2e_ms),
            fmt_tuning_short(row.cold_total_timer_ms),
            fmt_tuning_short(row.cold_symbolic_ms),
            fmt_tuning_short(row.cold_initial_symbolic_jacobian_ms),
            fmt_tuning_short(row.cold_aot_artifact_ms),
            fmt_tuning_short(row.cold_aot_materialize_ms),
            fmt_tuning_short(row.cold_aot_compile_link_ms),
        );
    }

    println!();
    println!(
        "[AOT combustion tuning map] isolated cold numerical/runtime stages; all columns are from the fresh child solve."
    );
    println!(
        "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12} | {:<18} | {:<18}",
        "config",
        "linear_ms",
        "jac_ms",
        "fun_ms",
        "residual_values",
        "jacobian_values",
        "jacobian_assembly",
        "res_jobs",
        "jac_jobs",
        "rebind_ms",
        "register_link_ms"
    );
    println!("{}", "-".repeat(222));
    for row in rows {
        println!(
            "{:<30} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<12} | {:<12} | {:<18} | {:<18}",
            row.label,
            fmt_tuning_short(row.cold_linear_ms),
            fmt_tuning_short(row.cold_jac_ms),
            fmt_tuning_short(row.cold_fun_ms),
            fmt_tuning_short(row.cold_cb_residual_values_ms),
            fmt_tuning_short(row.cold_cb_jacobian_values_ms),
            fmt_tuning_short(row.cold_cb_jacobian_assembly_ms),
            fmt_tuning_short(row.cold_residual_actual_jobs),
            fmt_tuning_short(row.cold_sparse_jacobian_actual_jobs),
            fmt_tuning_short(row.cold_post_build_rebind_ms),
            fmt_tuning_short(row.cold_aot_register_link_ms),
        );
    }

    println!();
    println!(
        "[AOT combustion tuning map] runtime timing/counter summary; all time columns are milliseconds"
    );
    println!(
        "note: manual_bootstrap_ms is callback preparation/linking performed by this diagnostic runner. Use solve_ms, callback stages, and counters for chunking decisions."
    );
    println!(
        "{:<30} | {:<18} | {:<31} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<14} | {:<14} | {:<14}",
        "config",
        "solve_ms",
        "manual_bootstrap_ms mean+/-std [min,max]",
        "solver_total_ms",
        "symbolic_ms",
        "linear_ms",
        "jac_ms",
        "fun_ms",
        "iters",
        "linsys",
        "jac_re"
    );
    println!("{}", "-".repeat(258));
    for row in rows {
        println!(
            "{:<30} | {:<18} | {:<31} | {:<18} | {:<18} | {:<18} | {:<18} | {:<18} | {:<14} | {:<14} | {:<14}",
            row.label,
            fmt_tuning_short(row.solve_ms),
            fmt_tuning_agg(row.bootstrap_ms),
            fmt_tuning_short(row.total_timer_ms),
            fmt_tuning_short(row.symbolic_ms),
            fmt_tuning_short(row.linear_ms),
            fmt_tuning_short(row.jac_ms),
            fmt_tuning_short(row.fun_ms),
            fmt_tuning_short(row.iterations),
            fmt_tuning_short(row.linear_solves),
            fmt_tuning_short(row.jac_rebuilds),
        );
    }

    println!();
    println!(
        "[AOT combustion tuning map] linked callback stage summary; all time columns are milliseconds"
    );
    println!(
        "note: these columns are populated by linked AOT callbacks; Lambdify rows may be blank."
    );
    println!(
        "{:<30} | {:<18} | {:<18} | {:<20}",
        "config", "residual_values", "jacobian_values", "jacobian_assembly"
    );
    println!("{}", "-".repeat(98));
    for row in rows {
        println!(
            "{:<30} | {:<18} | {:<18} | {:<20}",
            row.label,
            fmt_tuning_short(row.cb_residual_values_ms),
            fmt_tuning_short(row.cb_jacobian_values_ms),
            fmt_tuning_short(row.cb_jacobian_assembly_ms),
        );
    }
}
