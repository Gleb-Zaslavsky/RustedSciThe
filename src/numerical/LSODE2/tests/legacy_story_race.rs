#[derive(Clone, Copy)]
pub(super) enum BackendRaceMatrix {
    Dense,
    Sparse,
    Banded,
}

impl BackendRaceMatrix {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
    }
}

pub(super) fn race_lambdify_config(matrix: BackendRaceMatrix) -> Lsode2ProblemConfig {
    match matrix {
        BackendRaceMatrix::Dense => exponential_decay_config()
            .with_linear_system_structure(super::Lsode2LinearSystemStructure::Dense)
            .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto),
        BackendRaceMatrix::Sparse => exponential_decay_config().with_native_sparse_faer_backend(),
        BackendRaceMatrix::Banded => {
            exponential_decay_config().with_native_banded_faithful_backend()
        }
    }
}

pub(super) fn unique_story_run_tag(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}_pid{}_{}", std::process::id(), nanos)
}

pub(super) fn race_aot_config_with_output(
    matrix: BackendRaceMatrix,
    output_dir: impl Into<PathBuf>,
) -> Lsode2ProblemConfig {
    let generated =
        SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir).with_c_tcc();
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
    };
    match matrix {
        BackendRaceMatrix::Dense => {
            let mut config = exponential_decay_config()
                .with_linear_system_structure(super::Lsode2LinearSystemStructure::Dense)
                .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto)
                .with_residual_jacobian_source(source);
            config.backend.generated_backend = generated;
            config
        }
        BackendRaceMatrix::Sparse => exponential_decay_config()
            .with_native_sparse_faer_generated_backend(generated)
            .with_residual_jacobian_source(source),
        BackendRaceMatrix::Banded => exponential_decay_config()
            .with_native_banded_faithful_generated_backend(generated)
            .with_residual_jacobian_source(source),
    }
}

fn race_aot_config(matrix: BackendRaceMatrix) -> Lsode2ProblemConfig {
    let out = PathBuf::from(format!(
        "target/lsode2-story-race/{}/aot_c_tcc",
        matrix.label().to_lowercase()
    ));
    race_aot_config_with_output(matrix, out)
}

fn race_aot_parallel_config(
    matrix: BackendRaceMatrix,
    chunks_per_worker: usize,
) -> Lsode2ProblemConfig {
    race_aot_config(matrix).with_aot_parallel_chunking(chunks_per_worker)
}

pub(super) fn race_aot_parallel_config_with_output(
    matrix: BackendRaceMatrix,
    output_dir: impl Into<PathBuf>,
    chunks_per_worker: usize,
) -> Lsode2ProblemConfig {
    race_aot_config_with_output(matrix, output_dir).with_aot_parallel_chunking(chunks_per_worker)
}

fn race_analytical_config(matrix: BackendRaceMatrix) -> Option<Lsode2ProblemConfig> {
    let callbacks = (
        |_t: f64, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        |_t: f64, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
    );
    match matrix {
        BackendRaceMatrix::Dense => None,
        BackendRaceMatrix::Sparse => Some(
            exponential_decay_config()
                .with_native_sparse_faer_backend()
                .with_analytical_callbacks(callbacks.0, callbacks.1)
                .with_faithful_bdf_solve(4096, 4096),
        ),
        BackendRaceMatrix::Banded => Some(
            exponential_decay_config()
                .with_native_banded_faithful_backend()
                .with_analytical_callbacks(callbacks.0, callbacks.1)
                .with_faithful_bdf_solve(4096, 4096),
        ),
    }
}

pub(super) fn run_backend_race_sample(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_y: f64,
) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
    let started_total = Instant::now();
    let mut solver = Lsode2Solver::new(config).ok()?;
    let mut prepare_ms = 0.0;
    if route != "Analytical-Native" {
        let prep_started = Instant::now();
        solver.prepare().ok()?;
        prepare_ms = prep_started.elapsed().as_secs_f64() * 1_000.0;
    }
    let solve_started = Instant::now();
    let summary = solver.solve_with_summary().ok()?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = started_total.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary.final_y.as_ref()?.get(0).copied()?;
    let final_diff = (final_y - baseline_final_y).abs();
    let is_native_faithful = is_native_faithful_status(&summary.status);
    let residual_calls = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_residual_calls as f64
    } else {
        summary.statistics.residual_calls as f64
    };
    let jacobian_calls = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_jacobian_calls as f64
    } else {
        summary.statistics.jacobian_calls as f64
    };
    let nlu_or_native_linear = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_linear_solve_calls as f64
    } else {
        summary.statistics.bdf_nlu_total as f64
    };
    let residual_ms = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_residual_ms_total
    } else {
        summary.statistics.residual_ms_total
    };
    let jacobian_ms = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_jacobian_ms_total
    } else {
        summary.statistics.jacobian_ms_total
    };
    let linear_ms = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_linear_solve_ms_total
    } else {
        0.0
    };
    let accepted_steps = summary
        .native_statistics
        .native_step_accepts
        .max(summary.native_statistics.bridge_accepted_steps) as f64;
    let rejected_steps = (summary.native_statistics.native_step_rejects_error_test
        + summary.native_statistics.native_step_rejects_nonlinear) as f64;
    Some((
        total_ms,
        prepare_ms,
        solve_ms,
        final_diff,
        residual_calls,
        jacobian_calls,
        nlu_or_native_linear,
        residual_ms,
        jacobian_ms,
        linear_ms,
        accepted_steps,
        rejected_steps,
    ))
}

#[test]
fn lsode2_multi_run_backend_race_by_weight_class() {
    const REPEATS: usize = 5;
    let matrices = [
        BackendRaceMatrix::Dense,
        BackendRaceMatrix::Sparse,
        BackendRaceMatrix::Banded,
    ];
    let routes = ["Lambdify", "Analytical-Native", "AOT-Ctcc"];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let mut solver = Lsode2Solver::new(race_lambdify_config(matrix))
            .expect("race baseline config should build");
        let summary = solver
            .solve_with_summary()
            .expect("race baseline solve should finish");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("race baseline should expose final_y")[0];
        baselines.insert(matrix.label(), final_y);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for route in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let baseline = *baselines
                    .get(matrix.label())
                    .expect("baseline final_y should exist");
                let config = match route {
                    "Lambdify" => Some(race_lambdify_config(matrix)),
                    "AOT-Ctcc" => Some(race_aot_config(matrix)),
                    "Analytical-Native" => race_analytical_config(matrix),
                    _ => None,
                };
                let Some(config) = config else {
                    continue;
                };
                if let Some(sample) = run_backend_race_sample(route, config, baseline) {
                    row.runs_ok += 1;
                    row.total_ms.push(sample.0);
                    row.prepare_ms.push(sample.1);
                    row.solve_ms.push(sample.2);
                    row.final_diff.push(sample.3);
                    row.residual_calls.push(sample.4);
                    row.jacobian_calls.push(sample.5);
                    row.nlu_or_native_linear.push(sample.6);
                    row.residual_ms.push(sample.7);
                    row.jacobian_ms.push(sample.8);
                    row.linear_ms.push(sample.9);
                    row.accepted_steps.push(sample.10);
                    row.rejected_steps.push(sample.11);
                }
            }
            rows.push(row);
        }
    }

    println!(
        "[LSODE2 story] multi-run backend race by weight class; all time columns are milliseconds"
    );
    println!(
        "matrix | route             | ok/runs | total_ms mean+/-std [min,max] | final_diff mean+/-std [min,max] | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3e}+/-{s:.1e} [{n:.3e},{x:.3e}]"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "not_supported_or_failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<17} | {:>7} | {:<31} | {:<34} | {}",
            row.matrix,
            row.route,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            diff,
            status
        );
    }

    println!(
        "[LSODE2 story] backend race diagnostics; counters are per-solve counts (aggregated as mean+/-std)"
    );
    println!(
        "matrix | route             | prepare_ms mean+/-std | solve_ms mean+/-std | residual_calls mean+/-std | jacobian_calls mean+/-std | nlu_or_native_linear mean+/-std | accepted mean+/-std | rejected mean+/-std"
    );
    println!(
        "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let residual = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let nlu = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let accepted = row
            .accepted_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let rejected = row
            .rejected_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<17} | {:<21} | {:<19} | {:<24} | {:<24} | {:<31} | {:<18} | {}",
            row.matrix, row.route, prep, solve, residual, jacobian, nlu, accepted, rejected
        );
    }

    println!(
        "[LSODE2 story] backend race stage timers; all time columns are milliseconds (mean+/-std)"
    );
    println!("matrix | route             | residual_ms | jacobian_ms | linear_ms");
    println!("--------------------------------------------------------------------------");
    for row in &rows {
        let residual_ms = row
            .residual_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_ms = row
            .jacobian_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let linear_ms = row
            .linear_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<17} | {:<11} | {:<11} | {}",
            row.matrix, row.route, residual_ms, jacobian_ms, linear_ms
        );
    }

    assert!(
        rows.iter().any(|r| r.runs_ok > 0),
        "at least one backend race route should complete successfully"
    );
    for row in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful route should have final_diff samples");
            let tol = if row.route == "Analytical-Native" {
                1.0e-4
            } else {
                1.0e-7
            };
            assert!(
                mean_diff <= tol,
                "{} {} mean final_diff too large: {:e} (tol={:e})",
                row.matrix,
                row.route,
                mean_diff,
                tol
            );
        }
    }
}

pub(crate) fn run_lsode2_parallel_chunking_story_by_weight_class() {
    const REPEATS: usize = 5;
    const CHUNKS_PER_WORKER: usize = 2;
    let matrices = [
        BackendRaceMatrix::Dense,
        BackendRaceMatrix::Sparse,
        BackendRaceMatrix::Banded,
    ];
    let routes = [
        ("Lambdify", "baseline(no_chunk_knobs)"),
        ("AOT-Ctcc-Whole", "whole"),
        ("AOT-Ctcc-Parallel", "parallel(auto,x2)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let mut solver = Lsode2Solver::new(race_lambdify_config(matrix))
            .expect("parallel story baseline config should build");
        let summary = solver
            .solve_with_summary()
            .expect("parallel story baseline solve should finish");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("parallel story baseline should expose final_y")[0];
        baselines.insert(matrix.label(), final_y);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, chunking) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let baseline = *baselines
                .get(matrix.label())
                .expect("parallel story baseline final_y should exist");

            // Prewarm AOT routes once so multi-run aggregates reflect warm runtime
            // behavior instead of mixing in cold codegen/build noise.
            if route.starts_with("AOT-") {
                let warmup_cfg = match route {
                    "AOT-Ctcc-Whole" => Some(race_aot_config(matrix)),
                    "AOT-Ctcc-Parallel" => {
                        Some(race_aot_parallel_config(matrix, CHUNKS_PER_WORKER))
                    }
                    _ => None,
                };
                if let Some(cfg) = warmup_cfg {
                    let _ = run_backend_race_sample(route, cfg, baseline);
                }
            }

            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match route {
                    "Lambdify" => Some(race_lambdify_config(matrix)),
                    "AOT-Ctcc-Whole" => Some(race_aot_config(matrix)),
                    "AOT-Ctcc-Parallel" => {
                        Some(race_aot_parallel_config(matrix, CHUNKS_PER_WORKER))
                    }
                    _ => None,
                };
                let Some(config) = config else {
                    continue;
                };
                if let Some(sample) = run_backend_race_sample(route, config, baseline) {
                    row.runs_ok += 1;
                    row.total_ms.push(sample.0);
                    row.prepare_ms.push(sample.1);
                    row.solve_ms.push(sample.2);
                    row.final_diff.push(sample.3);
                    row.residual_calls.push(sample.4);
                    row.jacobian_calls.push(sample.5);
                    row.nlu_or_native_linear.push(sample.6);
                    row.residual_ms.push(sample.7);
                    row.jacobian_ms.push(sample.8);
                    row.linear_ms.push(sample.9);
                    row.accepted_steps.push(sample.10);
                    row.rejected_steps.push(sample.11);
                }
            }
            rows.push((row, chunking));
        }
    }

    println!(
        "[LSODE2 story] parallel chunking race by weight class; all time columns are milliseconds"
    );
    println!(
        "note: `Lambdify` is a baseline route and currently does not use generated-backend chunking knobs."
    );
    println!(
        "matrix | route             | chunking              | ok/runs | total_ms mean+/-std [min,max] | solve_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, chunking) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<17} | {:<21} | {:>7} | {:<31} | {:<18} | {:<20} | {}",
            row.matrix,
            row.route,
            chunking,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            solve,
            diff,
            status
        );
    }

    println!("[LSODE2 story] parallel chunking diagnostics; counters are counts (mean+/-std)");
    println!(
        "matrix | route             | chunking              | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | accepted | rejected"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, chunking) in &rows {
        let residual_calls = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_calls = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let linear_calls = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let residual_ms = row
            .residual_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_ms = row
            .jacobian_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let linear_ms = row
            .linear_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let accepted = row
            .accepted_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let rejected = row
            .rejected_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<17} | {:<21} | {:<14} | {:<14} | {:<12} | {:<11} | {:<11} | {:<9} | {:<8} | {}",
            row.matrix,
            row.route,
            chunking,
            residual_calls,
            jacobian_calls,
            linear_calls,
            residual_ms,
            jacobian_ms,
            linear_ms,
            accepted,
            rejected
        );
    }

    assert!(
        rows.iter().any(|(r, _)| r.runs_ok > 0),
        "at least one parallel chunking race route should complete successfully"
    );
    for (row, _) in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful parallel chunking route should have diff samples");
            assert!(
                mean_diff <= 1.0e-5,
                "{} {} final_diff too large in parallel chunking race: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
        }
    }
}
