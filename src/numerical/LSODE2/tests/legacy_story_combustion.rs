pub(super) fn combustion_like_story_base_config() -> Lsode2ProblemConfig {
    // Simplified stiff combustion-like IVP:
    // A -> B with Arrhenius-driven heat release + linear cooling.
    let eqs = vec![
        Expr::parse_expression("-k*exp(-E/(R*T))*A*A"),
        Expr::parse_expression("0.5*k*exp(-E/(R*T))*A*A - kloss*B"),
        Expr::parse_expression("Qcrho*k*exp(-E/(R*T))*A*A - cooling*(T - T0)"),
    ];
    Lsode2ProblemConfig::new(
        eqs,
        vec!["A".to_string(), "B".to_string(), "T".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0, 300.0]),
        120.0,
        0.5,
        1e-8,
        1e-8,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "T0".to_string(),
        "Qcrho".to_string(),
        "kloss".to_string(),
        "cooling".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![
        1.0e7, // k
        5.0e4, // E
        8.314, // R
        300.0, // T0
        5.0e2, // Qcrho
        0.0,   // kloss
        0.5,   // cooling
    ]))
    .with_controller(
        super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
            .with_method_switch_probe_steps(1),
    )
    .with_faithful_bdf_solve(20_000, 20_000)
}

fn combustion_story_route_config_with_base_dir(
    matrix: BackendRaceMatrix,
    route: &'static str,
    base_dir: &str,
) -> Option<Lsode2ProblemConfig> {
    let source = match route {
        "Lambdify" => Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
        "AOT-Ctcc" | "AOT-Ctcc-Whole" | "AOT-Ctcc-Parallel" => {
            Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
                execution: Lsode2SymbolicExecutionMode::Aot {
                    toolchain: Lsode2AotToolchain::CTcc,
                    profile: Lsode2AotProfile::Release,
                },
            }
        }
        _ => return None,
    };

    let base = combustion_like_story_base_config().with_residual_jacobian_source(source);
    let cfg = match (matrix, route) {
        (BackendRaceMatrix::Sparse, "Lambdify") => base.with_native_sparse_faer_backend(),
        (BackendRaceMatrix::Banded, "Lambdify") => base.with_native_banded_faithful_backend(),
        (BackendRaceMatrix::Sparse, "AOT-Ctcc") | (BackendRaceMatrix::Sparse, "AOT-Ctcc-Whole") => {
            let out = PathBuf::from(format!("{base_dir}/sparse/aot_c_tcc"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_sparse_faer_generated_backend(backend)
        }
        (BackendRaceMatrix::Banded, "AOT-Ctcc") | (BackendRaceMatrix::Banded, "AOT-Ctcc-Whole") => {
            let out = PathBuf::from(format!("{base_dir}/banded/aot_c_tcc"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_banded_faithful_generated_backend(backend)
        }
        (BackendRaceMatrix::Sparse, "AOT-Ctcc-Parallel") => {
            let out = PathBuf::from(format!("{base_dir}/sparse/aot_c_tcc_parallel"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_sparse_faer_generated_backend(backend)
                .with_aot_parallel_chunking(2)
        }
        (BackendRaceMatrix::Banded, "AOT-Ctcc-Parallel") => {
            let out = PathBuf::from(format!("{base_dir}/banded/aot_c_tcc_parallel"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_banded_faithful_generated_backend(backend)
                .with_aot_parallel_chunking(2)
        }
        _ => return None,
    };
    Some(cfg)
}

fn combustion_story_route_config(
    matrix: BackendRaceMatrix,
    route: &'static str,
) -> Option<Lsode2ProblemConfig> {
    combustion_story_route_config_with_base_dir(matrix, route, "target/lsode2-story-combustion")
}

pub(super) type CombustionStorySample = (
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
);

pub(super) fn run_combustion_story_sample_result(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_a: f64,
) -> Result<CombustionStorySample, String> {
    let skip_explicit_prepare = matches!(
        config.backend.generated_backend.build_policy,
        SymbolicIvpAotBuildPolicy::RebuildAlways { .. }
    );
    let started_total = Instant::now();
    let mut solver = Lsode2Solver::new(config)
        .map_err(|err| format!("new_error({})", short_error(&err.to_string())))?;
    let prepare_ms_wall = if skip_explicit_prepare {
        // `solve_with_summary` builds the native generated engine itself.
        // With RebuildAlways, an eager `prepare()` would build and load the
        // same DLL once for the bridge path, then native solve would try to
        // rebuild that loaded artifact again and fail on Windows file locks.
        0.0
    } else {
        let prep_started = Instant::now();
        solver
            .prepare()
            .map_err(|err| format!("prepare_error({})", short_error(&err.to_string())))?;
        prep_started.elapsed().as_secs_f64() * 1_000.0
    };
    let solve_started = Instant::now();
    let summary = solver
        .solve_with_summary()
        .map_err(|err| format!("solve_error({})", short_error(&err.to_string())))?;
    let solve_ms_wall = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let total_ms_wall = started_total.elapsed().as_secs_f64() * 1_000.0;

    let final_a = summary
        .final_y
        .as_ref()
        .and_then(|y| y.get(0).copied())
        .ok_or_else(|| "solve_error(missing final state)".to_string())?;
    let final_diff = (final_a - baseline_final_a).abs();
    let is_native_faithful = is_native_faithful_status(&summary.status);

    let (
        stage_prepare_ms,
        stage_solve_ms,
        residual_calls,
        jacobian_calls,
        linear_calls,
        residual_ms,
        jacobian_ms,
        linear_ms,
    ) = if is_native_faithful {
        (
            summary.native_statistics.backend_prepare_ms_total,
            summary.native_statistics.solve_ms_total,
            summary.native_statistics.native_residual_calls as f64,
            summary.native_statistics.native_jacobian_calls as f64,
            summary.native_statistics.native_linear_solve_calls as f64,
            summary.native_statistics.native_residual_ms_total,
            summary.native_statistics.native_jacobian_ms_total,
            summary.native_statistics.native_linear_solve_ms_total,
        )
    } else {
        (
            summary.statistics.backend_prepare_ms_total,
            summary.statistics.solve_ms_total,
            summary.statistics.residual_calls as f64,
            summary.statistics.jacobian_calls as f64,
            summary.statistics.bdf_nlu_total as f64,
            summary.statistics.residual_ms_total,
            summary.statistics.jacobian_ms_total,
            0.0,
        )
    };

    let accepted_steps = summary
        .native_statistics
        .native_step_accepts
        .max(summary.native_statistics.bridge_accepted_steps) as f64;
    let rejected_steps = (summary.native_statistics.native_step_rejects_error_test
        + summary.native_statistics.native_step_rejects_nonlinear) as f64;
    let preferred_bdf = summary.native_statistics.preferred_bdf_count as f64;
    let executed_bdf = summary.native_statistics.executed_bdf_count as f64;

    let _ = route;
    let reported_prepare_ms = if skip_explicit_prepare {
        stage_prepare_ms
    } else {
        prepare_ms_wall
    };
    Ok((
        total_ms_wall,
        reported_prepare_ms,
        solve_ms_wall,
        final_diff,
        residual_calls,
        jacobian_calls,
        linear_calls,
        accepted_steps,
        rejected_steps,
        preferred_bdf,
        executed_bdf,
        stage_prepare_ms,
        stage_solve_ms,
        residual_ms,
        jacobian_ms,
        linear_ms,
    ))
}

fn run_combustion_story_sample(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_a: f64,
) -> Option<CombustionStorySample> {
    run_combustion_story_sample_result(route, config, baseline_final_a).ok()
}

pub(crate) fn run_lsode2_combustion_like_multi_run_story_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::tests::lsode2_combustion_like_multi_run_story_dashboard",
    );
    const REPEATS: usize = 5;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = ["Lambdify", "AOT-Ctcc"];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let baseline_cfg =
            combustion_story_route_config(matrix, "Lambdify").expect("lambdify baseline config");
        let mut solver = Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
        let summary = solver
            .solve_with_summary()
            .expect("baseline combustion solve should finish");
        let final_a = summary
            .final_y
            .as_ref()
            .expect("baseline should provide final_y")[0];
        baselines.insert(matrix.label(), final_a);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for route in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let mut preferred_bdf = RaceStats::default();
            let mut executed_bdf = RaceStats::default();
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let Some(cfg) = combustion_story_route_config(matrix, route) else {
                    continue;
                };
                let baseline = *baselines
                    .get(matrix.label())
                    .expect("combustion baseline should exist");
                if let Some(sample) = run_combustion_story_sample(route, cfg, baseline) {
                    row.runs_ok += 1;
                    row.total_ms.push(sample.0);
                    row.prepare_ms.push(sample.1);
                    row.solve_ms.push(sample.2);
                    row.final_diff.push(sample.3);
                    row.residual_calls.push(sample.4);
                    row.jacobian_calls.push(sample.5);
                    row.nlu_or_native_linear.push(sample.6);
                    row.accepted_steps.push(sample.7);
                    row.rejected_steps.push(sample.8);
                    preferred_bdf.push(sample.9);
                    executed_bdf.push(sample.10);
                    row.residual_ms.push(sample.13);
                    row.jacobian_ms.push(sample.14);
                    row.linear_ms.push(sample.15);
                }
            }
            rows.push((row, preferred_bdf, executed_bdf));
        }
    }

    println!(
        "[LSODE2 story] combustion-like backend summary (multi-run); all time columns are milliseconds"
    );
    println!(
        "matrix | route     | ok/runs | total_ms mean+/-std [min,max] | final_diff(A) mean+/-std [min,max] | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------"
    );
    for (row, _, _) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2e}+/-{s:.1e} [{n:.2e},{x:.2e}]"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<9} | {:>7} | {:<31} | {:<36} | {}",
            row.matrix,
            row.route,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            diff,
            status
        );
    }

    println!(
        "[LSODE2 story] combustion-like diagnostics (multi-run); prepare/solve are stage times, counters are counts"
    );
    println!(
        "matrix | route     | prepare_ms mean+/-std | solve_ms mean+/-std | residual_calls mean+/-std | jacobian_calls mean+/-std | linear_calls mean+/-std | accepted mean+/-std | rejected mean+/-std | preferred_bdf mean+/-std | executed_bdf mean+/-std"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, preferred_bdf, executed_bdf) in &rows {
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let residual = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let linear = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
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
        let pref_bdf = preferred_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let exec_bdf = executed_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<9} | {:<21} | {:<19} | {:<24} | {:<24} | {:<21} | {:<18} | {:<18} | {:<24} | {}",
            row.matrix,
            row.route,
            prep,
            solve,
            residual,
            jacobian,
            linear,
            accepted,
            rejected,
            pref_bdf,
            exec_bdf
        );
    }

    println!(
        "[LSODE2 story] combustion-like stage timers (multi-run); all time columns are milliseconds"
    );
    println!(
        "matrix | route     | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std"
    );
    println!(
        "-----------------------------------------------------------------------------------------------"
    );
    for (row, _, _) in &rows {
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
            "{:<6} | {:<9} | {:<21} | {:<21} | {}",
            row.matrix, row.route, residual_ms, jacobian_ms, linear_ms
        );
    }

    assert!(
        rows.iter().any(|(row, _, _)| row.runs_ok > 0),
        "at least one combustion story route should complete successfully"
    );
    for (row, _, _) in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful combustion route should have diff samples");
            assert!(
                mean_diff <= 2.0e-4,
                "{} {} combustion final_diff too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
        }
    }
}

pub(crate) fn run_lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard() {
    const REPEATS: usize = 5;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = [
        ("Lambdify", "baseline(no_chunk_knobs)"),
        ("AOT-Ctcc-Whole", "whole"),
        ("AOT-Ctcc-Parallel", "parallel(auto,x2)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let baseline_cfg =
            combustion_story_route_config(matrix, "Lambdify").expect("lambdify baseline config");
        let mut solver = Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
        let summary = solver
            .solve_with_summary()
            .expect("baseline combustion solve should finish");
        let final_a = summary
            .final_y
            .as_ref()
            .expect("baseline should provide final_y")[0];
        baselines.insert(matrix.label(), final_a);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, chunking) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let mut preferred_bdf = RaceStats::default();
            let mut executed_bdf = RaceStats::default();

            let baseline = *baselines
                .get(matrix.label())
                .expect("combustion baseline should exist");

            // Prewarm AOT routes once so statistics focus on warm runtime and
            // chunking effects, while cold build cost is covered by dedicated
            // AOT stage/cold-warm stories.
            if route.starts_with("AOT-") {
                if let Some(cfg) = combustion_story_route_config(matrix, route) {
                    let _ = run_combustion_story_sample(route, cfg, baseline);
                }
            }

            for _ in 0..REPEATS {
                row.runs_total += 1;
                let Some(cfg) = combustion_story_route_config(matrix, route) else {
                    continue;
                };
                if let Some(sample) = run_combustion_story_sample(route, cfg, baseline) {
                    row.runs_ok += 1;
                    row.total_ms.push(sample.0);
                    row.prepare_ms.push(sample.1);
                    row.solve_ms.push(sample.2);
                    row.final_diff.push(sample.3);
                    row.residual_calls.push(sample.4);
                    row.jacobian_calls.push(sample.5);
                    row.nlu_or_native_linear.push(sample.6);
                    row.accepted_steps.push(sample.7);
                    row.rejected_steps.push(sample.8);
                    preferred_bdf.push(sample.9);
                    executed_bdf.push(sample.10);
                    row.residual_ms.push(sample.13);
                    row.jacobian_ms.push(sample.14);
                    row.linear_ms.push(sample.15);
                }
            }
            rows.push((row, preferred_bdf, executed_bdf, chunking));
        }
    }

    println!(
        "[LSODE2 story] combustion-like parallel chunking summary (multi-run); all time columns are milliseconds"
    );
    println!(
        "matrix | route              | chunking              | ok/runs | total_ms mean+/-std [min,max] | solve_ms mean+/-std | final_diff(A) mean+/-std | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, _, _, chunking) in &rows {
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
            "{:<6} | {:<18} | {:<21} | {:>7} | {:<31} | {:<18} | {:<24} | {}",
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

    println!(
        "[LSODE2 story] combustion-like parallel chunking diagnostics (multi-run); counters are counts"
    );
    println!(
        "matrix | route              | chunking              | residual_calls | jacobian_calls | linear_calls | accepted | rejected | preferred_bdf | executed_bdf"
    );
    println!(
        "-------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, preferred_bdf, executed_bdf, chunking) in &rows {
        let residual = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let linear = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
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
        let pref_bdf = preferred_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let exec_bdf = executed_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<18} | {:<21} | {:<14} | {:<14} | {:<12} | {:<8} | {:<8} | {:<13} | {}",
            row.matrix,
            row.route,
            chunking,
            residual,
            jacobian,
            linear,
            accepted,
            rejected,
            pref_bdf,
            exec_bdf
        );
    }

    assert!(
        rows.iter().any(|(row, _, _, _)| row.runs_ok > 0),
        "at least one combustion parallel chunking route should complete successfully"
    );
    for (row, _, _, _) in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful combustion parallel route should have diff samples");
            assert!(
                mean_diff <= 2.0e-4,
                "{} {} combustion parallel final_diff too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
        }
    }
}

pub(crate) fn run_lsode2_parallel_chunking_cold_stage_story_by_weight_class() {
    const REPEATS: usize = 3;
    const CHUNKS_PER_WORKER: usize = 2;
    let matrices = [
        BackendRaceMatrix::Dense,
        BackendRaceMatrix::Sparse,
        BackendRaceMatrix::Banded,
    ];
    let routes = [
        ("Lambdify", "baseline(no_build_stage)"),
        ("AOT-Ctcc-Whole", "cold_build(whole)"),
        ("AOT-Ctcc-Parallel", "cold_build(parallel)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let mut solver = Lsode2Solver::new(race_lambdify_config(matrix))
            .expect("cold-stage baseline config should build");
        let summary = solver
            .solve_with_summary()
            .expect("cold-stage baseline solve should finish");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("cold-stage baseline should expose final_y")[0];
        baselines.insert(matrix.label(), final_y);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, scenario) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let baseline = *baselines
                .get(matrix.label())
                .expect("cold-stage baseline final_y should exist");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match route {
                    "Lambdify" => Some(race_lambdify_config(matrix)),
                    "AOT-Ctcc-Whole" => {
                        let tag = unique_story_run_tag("lsode2_story_race_cold_whole");
                        let out = PathBuf::from(format!(
                            "target/lsode2-story-race-cold/{}/{}/whole",
                            matrix.label().to_lowercase(),
                            tag
                        ));
                        Some(race_aot_config_with_output(matrix, out))
                    }
                    "AOT-Ctcc-Parallel" => {
                        let tag = unique_story_run_tag("lsode2_story_race_cold_parallel");
                        let out = PathBuf::from(format!(
                            "target/lsode2-story-race-cold/{}/{}/parallel",
                            matrix.label().to_lowercase(),
                            tag
                        ));
                        Some(race_aot_parallel_config_with_output(
                            matrix,
                            out,
                            CHUNKS_PER_WORKER,
                        ))
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
                }
            }
            rows.push((row, scenario));
        }
    }

    println!(
        "[LSODE2 story] parallel chunking cold-stage story by weight class; all time columns are milliseconds"
    );
    println!(
        "note: this table intentionally measures cold build+prepare cost for AOT (unique artifact dir per run)."
    );
    println!(
        "matrix | route             | scenario              | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, scenario) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
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
            "{:<6} | {:<17} | {:<21} | {:>7} | {:<31} | {:<21} | {:<18} | {:<20} | {}",
            row.matrix,
            row.route,
            scenario,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            prep,
            solve,
            diff,
            status
        );
    }

    assert!(
        rows.iter().any(|(row, _)| row.runs_ok > 0),
        "at least one cold-stage route should complete successfully"
    );
}

pub(crate) fn run_lsode2_combustion_like_parallel_chunking_cold_stage_story_dashboard() {
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = [
        ("Lambdify", "baseline(no_build_stage)"),
        ("AOT-Ctcc-Whole", "cold_build(whole)"),
        ("AOT-Ctcc-Parallel", "cold_build(parallel)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let baseline_cfg =
            combustion_story_route_config(matrix, "Lambdify").expect("lambdify baseline config");
        let mut solver = Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
        let summary = solver
            .solve_with_summary()
            .expect("baseline combustion solve should finish");
        let final_a = summary
            .final_y
            .as_ref()
            .expect("baseline should provide final_y")[0];
        baselines.insert(matrix.label(), final_a);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, scenario) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let baseline = *baselines
                .get(matrix.label())
                .expect("combustion baseline should exist");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match route {
                    "Lambdify" => combustion_story_route_config(matrix, route),
                    "AOT-Ctcc-Whole" | "AOT-Ctcc-Parallel" => {
                        let tag = unique_story_short_tag();
                        let route_short = if route == "AOT-Ctcc-Whole" { "w" } else { "p" };
                        let matrix_short = if matrix.label() == "Sparse" { "s" } else { "b" };
                        let base_dir =
                            format!("target/l2cold/{}/{}/{}", matrix_short, route_short, tag);
                        combustion_story_route_config_with_base_dir(
                            matrix,
                            route,
                            base_dir.as_str(),
                        )
                    }
                    _ => None,
                };
                let Some(config) = config else {
                    continue;
                };
                if let Some(sample) = run_combustion_story_sample(route, config, baseline) {
                    row.runs_ok += 1;
                    row.total_ms.push(sample.0);
                    row.prepare_ms.push(sample.1);
                    row.solve_ms.push(sample.2);
                    row.final_diff.push(sample.3);
                }
            }
            rows.push((row, scenario));
        }
    }

    println!(
        "[LSODE2 story] combustion-like parallel chunking cold-stage summary; all time columns are milliseconds"
    );
    println!(
        "note: this table intentionally includes cold AOT build/prepare by forcing unique artifact dirs."
    );
    println!(
        "matrix | route              | scenario              | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff(A) mean+/-std | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, scenario) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
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
            "{:<6} | {:<18} | {:<21} | {:>7} | {:<31} | {:<21} | {:<18} | {:<24} | {}",
            row.matrix,
            row.route,
            scenario,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            prep,
            solve,
            diff,
            status
        );
    }

    assert!(
        rows.iter().any(|(row, _)| row.runs_ok > 0),
        "at least one combustion cold-stage route should complete successfully"
    );
}

pub(super) fn combustion_symbolic_matrix_config(
    matrix: BackendRaceMatrix,
    assembly: Lsode2SymbolicAssemblyBackend,
    execution: Lsode2SymbolicExecutionMode,
    generated: Option<SymbolicIvpGeneratedBackendConfig>,
) -> Lsode2ProblemConfig {
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution,
    };
    let base = combustion_like_story_base_config().with_residual_jacobian_source(source);
    match (matrix, generated) {
        (BackendRaceMatrix::Sparse, Some(backend)) => {
            base.with_native_sparse_faer_generated_backend(backend)
        }
        (BackendRaceMatrix::Banded, Some(backend)) => {
            base.with_native_banded_faithful_generated_backend(backend)
        }
        (BackendRaceMatrix::Sparse, None) => base.with_native_sparse_faer_backend(),
        (BackendRaceMatrix::Banded, None) => base.with_native_banded_faithful_backend(),
        (BackendRaceMatrix::Dense, _) => {
            unreachable!("combustion symbolic frontend matrix intentionally tests sparse/banded")
        }
    }
}

fn combustion_symbolic_matrix_config_with_evaluator_policy(
    matrix: BackendRaceMatrix,
    assembly: Lsode2SymbolicAssemblyBackend,
    evaluator_policy: IvpLambdifyExecutionPolicy,
    telemetry: IvpTelemetry,
) -> Lsode2ProblemConfig {
    combustion_symbolic_matrix_config(
        matrix,
        assembly,
        Lsode2SymbolicExecutionMode::LambdifyExpr,
        None,
    )
    .with_lambdify_execution_policy(evaluator_policy)
    .with_telemetry(telemetry)
}

pub(super) fn push_combustion_sample(row: &mut BackendRaceRow, sample: CombustionStorySample) {
    row.runs_ok += 1;
    row.total_ms.push(sample.0);
    row.prepare_ms.push(sample.1);
    row.solve_ms.push(sample.2);
    row.final_diff.push(sample.3);
    row.residual_calls.push(sample.4);
    row.jacobian_calls.push(sample.5);
    row.nlu_or_native_linear.push(sample.6);
    row.accepted_steps.push(sample.7);
    row.rejected_steps.push(sample.8);
    row.residual_ms.push(sample.13);
    row.jacobian_ms.push(sample.14);
    row.linear_ms.push(sample.15);
}

pub(super) fn print_compact_combustion_story_tables(title: &str, rows: &[BackendRaceRow]) {
    println!("[LSODE2 story] {title} correctness/wall-clock; all time columns are milliseconds");
    println!(
        "matrix | route                    | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let prepare = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
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
        let status = row.status_label();
        println!(
            "{:<6} | {:<24} | {:>7} | {:<31} | {:<21} | {:<19} | {:<21} | {}",
            row.matrix,
            row.route,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            prepare,
            solve,
            diff,
            status
        );
    }

    println!("[LSODE2 story] {title} numerical work; counters are counts (mean+/-std)");
    println!(
        "matrix | route                    | residual_calls | jacobian_calls | linear_calls | accepted | rejected"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let fmt_count = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<24} | {:<14} | {:<14} | {:<12} | {:<8} | {}",
            row.matrix,
            row.route,
            fmt_count(&row.residual_calls),
            fmt_count(&row.jacobian_calls),
            fmt_count(&row.nlu_or_native_linear),
            fmt_count(&row.accepted_steps),
            fmt_count(&row.rejected_steps),
        );
    }

    println!("[LSODE2 story] {title} hot-stage timers; all time columns are milliseconds");
    println!(
        "matrix | route                    | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let fmt_time = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<24} | {:<21} | {:<21} | {}",
            row.matrix,
            row.route,
            fmt_time(&row.residual_ms),
            fmt_time(&row.jacobian_ms),
            fmt_time(&row.linear_ms),
        );
    }
}

#[test]
#[ignore = "release story: multi-run symbolic frontend comparison on the combustion workload"]
fn lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard",
    );
    const REPEATS: usize = 5;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let frontends = [
        (
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            "Lambdify-ExprLegacy",
        ),
        (
            Lsode2SymbolicAssemblyBackend::AtomView,
            "Lambdify-AtomViewNative",
        ),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let config = combustion_symbolic_matrix_config(
            matrix,
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        );
        let mut solver = Lsode2Solver::new(config).expect("ExprLegacy baseline should build");
        let summary = solver
            .solve_with_summary()
            .expect("ExprLegacy baseline should solve");
        baselines.insert(
            matrix.label(),
            summary.final_y.expect("baseline final state should exist")[0],
        );
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (assembly, label) in frontends {
            let mut row = BackendRaceRow::new(matrix.label(), label);
            let baseline = *baselines
                .get(matrix.label())
                .expect("baseline should exist");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = combustion_symbolic_matrix_config(
                    matrix,
                    assembly,
                    Lsode2SymbolicExecutionMode::LambdifyExpr,
                    None,
                );
                if let Some(sample) = run_combustion_story_sample(label, config, baseline) {
                    push_combustion_sample(&mut row, sample);
                }
            }
            rows.push(row);
        }
    }

    print_compact_combustion_story_tables(
        "combustion symbolic frontend Sparse/Banded (Lambdify)",
        &rows,
    );

    for row in rows {
        assert_eq!(
            row.runs_ok, row.runs_total,
            "{} {} should complete all frontend comparison runs",
            row.matrix, row.route
        );
        let (mean_diff, _, _, _) = row.final_diff.summary().expect("completed row has diffs");
        assert!(
            mean_diff <= 2.0e-4,
            "{} {} frontend drift too large: {:e}",
            row.matrix,
            row.route,
            mean_diff
        );
    }
}

#[test]
#[ignore = "release canonical Lambdify evaluator-policy baseline on the archived combustion fixture"]
fn lsode2_combustion_lambdify_evaluator_policy_canonical_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_combustion_lambdify_evaluator_policy_canonical_story",
    );
    let repeats = std::env::var("LSODE2_COMBUSTION_POLICY_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3);
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let frontends = [
        (Lsode2SymbolicAssemblyBackend::ExprLegacy, "ExprLegacy"),
        (Lsode2SymbolicAssemblyBackend::AtomView, "AtomViewNative"),
    ];
    let policies = [
        ("Sequential", IvpLambdifyExecutionPolicy::Sequential),
        (
            "Parallel",
            IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        ),
        ("Auto", IvpLambdifyExecutionPolicy::Auto { min_work: 1 }),
    ];

    println!(
        "[LSODE2 canonical Lambdify policy] same archived combustion fixture; repeats={repeats}; preparation, warm callbacks, solver wall-clock and integer trajectory counters are reported separately"
    );
    println!(
        "matrix | frontend          | policy     | rep | prepare_ms | solve_ms | residual_ms | jacobian_ms | workers | parallel_dispatches | sequential_dispatches | solver_residual_calls | solver_jacobian_calls | evaluator_residual_calls | evaluator_jacobian_calls | jacobian_rebuilds | linear_solves | accepted | rejected | max_state_diff"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    let mut stage_rows = Vec::new();
    let mut counter_rows = Vec::new();
    for matrix in matrices {
        let mut reference_state: Option<DVector<f64>> = None;
        for (assembly, frontend_label) in frontends {
            for (policy_label, policy) in policies {
                for repetition in 1..=repeats {
                    let telemetry = IvpTelemetry::detailed();
                    let config = combustion_symbolic_matrix_config_with_evaluator_policy(
                        matrix,
                        assembly,
                        policy,
                        telemetry.clone(),
                    );
                    let prepare_started = Instant::now();
                    let mut solver = Lsode2Solver::new(config)
                        .expect("canonical combustion policy config should build");
                    solver
                        .prepare()
                        .expect("canonical combustion policy preparation should succeed");
                    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
                    let solve_started = Instant::now();
                    let summary = solver
                        .solve_with_summary()
                        .expect("canonical combustion policy solve should succeed");
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
                    let final_state = summary
                        .final_y
                        .clone()
                        .expect("canonical combustion policy should return final state");
                    let max_state_diff = reference_state
                        .as_ref()
                        .map(|reference| {
                            final_state
                                .iter()
                                .zip(reference.iter())
                                .map(|(actual, expected)| (actual - expected).abs())
                                .fold(0.0_f64, f64::max)
                        })
                        .unwrap_or(0.0);
                    let snapshot: IvpTelemetrySnapshot = solver.telemetry_snapshot();
                    let cold_ms = |stage: IvpColdStage| {
                        snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1_000.0
                    };
                    let warm_ms = |stage: IvpWarmStage| {
                        snapshot.warm_stage(stage).elapsed.as_secs_f64() * 1_000.0
                    };
                    let solver_residual_calls: u64 = if is_native_faithful_status(&summary.status) {
                        summary.native_statistics.native_residual_calls as u64
                    } else {
                        summary.statistics.residual_calls as u64
                    };
                    let solver_jacobian_calls: u64 = if is_native_faithful_status(&summary.status) {
                        summary.native_statistics.native_jacobian_calls as u64
                    } else {
                        summary.statistics.jacobian_calls as u64
                    };
                    assert!(
                        snapshot.residual_evaluations >= solver_residual_calls,
                        "evaluator residual evaluations cannot be below solver requests"
                    );
                    assert!(
                        snapshot.jacobian_evaluations >= solver_jacobian_calls,
                        "evaluator Jacobian evaluations cannot be below solver requests"
                    );
                    assert_eq!(
                        snapshot.residual_evaluations,
                        solver_residual_calls
                            .saturating_add(snapshot.residual_preparation_evaluations),
                        "residual evaluator calls must equal solver calls plus cold preparation probes"
                    );
                    assert_eq!(
                        snapshot.jacobian_evaluations, solver_jacobian_calls,
                        "canonical Jacobian evaluator calls must equal solver calls"
                    );
                    let unattributed_residual_calls = snapshot
                        .residual_evaluations
                        .checked_sub(
                            solver_residual_calls
                                .saturating_add(snapshot.residual_preparation_evaluations),
                        )
                        .expect("residual evaluator ownership must not exceed the total");
                    let unattributed_jacobian_calls = snapshot
                        .jacobian_evaluations
                        .checked_sub(solver_jacobian_calls)
                        .expect("Jacobian evaluator ownership must not exceed the total");
                    assert_eq!(
                        unattributed_residual_calls, 0,
                        "all canonical residual evaluator calls should have a typed owner"
                    );
                    assert_eq!(
                        unattributed_jacobian_calls, 0,
                        "all canonical Jacobian evaluator calls should have a typed owner"
                    );
                    println!(
                        "{:<6} | {:<17} | {:<10} | {:>3} | {:>10.3} | {:>8.3} | {:>11.3} | {:>11.3} | {:>7} | {:>19} | {:>21} | {:>21} | {:>22} | {:>24} | {:>24} | {:>17} | {:>13} | {:>8} | {:>8} | {:.3e}",
                        matrix.label(),
                        frontend_label,
                        policy_label,
                        repetition,
                        prepare_ms,
                        solve_ms,
                        snapshot
                            .warm_stage(IvpWarmStage::ResidualCallback)
                            .elapsed
                            .as_secs_f64()
                            * 1_000.0,
                        snapshot
                            .warm_stage(IvpWarmStage::JacobianCallback)
                            .elapsed
                            .as_secs_f64()
                            * 1_000.0,
                        snapshot.lambdify_worker_count,
                        snapshot.parallel_dispatches,
                        snapshot.sequential_dispatches,
                        solver_residual_calls,
                        solver_jacobian_calls,
                        snapshot.residual_evaluations,
                        snapshot.jacobian_evaluations,
                        snapshot.jacobian_rebuilds,
                        snapshot.linear_solve_requests,
                        snapshot.accepted_steps,
                        snapshot.rejected_steps,
                        max_state_diff,
                    );
                    counter_rows.push(format!(
                        "{:<6} | {:<17} | {:<10} | {:>3} | {:>7} | {:>7} | {:>8} | {:>8} | {:>8} | {:>16} | {:>9} | {:>9} | {:>9} | {:>9} | {:>15}",
                        matrix.label(),
                        frontend_label,
                        policy_label,
                        repetition,
                        solver_residual_calls,
                        snapshot.residual_requests,
                        snapshot.residual_evaluations,
                        snapshot.residual_auxiliary_evaluations,
                        snapshot.residual_preparation_evaluations,
                        unattributed_residual_calls,
                        solver_jacobian_calls,
                        snapshot.jacobian_requests,
                        snapshot.jacobian_evaluations,
                        snapshot.jacobian_auxiliary_evaluations,
                        unattributed_jacobian_calls,
                    ));
                    stage_rows.push(format!(
                        "{:<6} | {:<17} | {:<10} | {:>3} | {:>7.3} | {:>11.3} | {:>15.3} | {:>15.3} | {:>17.3} | {:>19.3} | {:>19.3} | {:>19.3} | {:>15.3} | {:>17.3} | {:>15.3} | {:>16.3}",
                        matrix.label(),
                        frontend_label,
                        policy_label,
                        repetition,
                        cold_ms(IvpColdStage::SymbolicDifferentiation),
                        cold_ms(IvpColdStage::Simplification),
                        cold_ms(IvpColdStage::ExprToAtom),
                        cold_ms(IvpColdStage::AtomToExpr),
                        cold_ms(IvpColdStage::SparsePattern),
                        cold_ms(IvpColdStage::ResidualLambdification),
                        cold_ms(IvpColdStage::JacobianLambdification),
                        warm_ms(IvpWarmStage::ArgumentBinding),
                        warm_ms(IvpWarmStage::ResidualEvaluation),
                        warm_ms(IvpWarmStage::ResidualOutputAssembly),
                        warm_ms(IvpWarmStage::JacobianEvaluation),
                        warm_ms(IvpWarmStage::JacobianOutputAssembly),
                    ));
                    assert!(final_state.iter().all(|value| value.is_finite()));
                    assert!(snapshot.residual_evaluations > 0);
                    assert!(snapshot.jacobian_evaluations > 0);
                    assert!(snapshot.linear_solve_requests > 0);
                    assert!(
                        snapshot.warm_stage(IvpWarmStage::ResidualCallback).calls > 0,
                        "detailed Lambdify telemetry must close the residual callback scope"
                    );
                    assert!(
                        snapshot.warm_stage(IvpWarmStage::JacobianCallback).calls > 0,
                        "detailed Lambdify telemetry must close the Jacobian callback scope"
                    );
                    assert_eq!(snapshot.lambdify_execution_policy, policy);
                    if matches!(policy, IvpLambdifyExecutionPolicy::Sequential) {
                        assert_eq!(snapshot.parallel_dispatches, 0);
                    }
                    assert!(max_state_diff <= 5.0e-6);
                    if reference_state.is_none() {
                        reference_state = Some(final_state);
                    }
                }
            }
        }
    }
    println!("");
    println!(
        "[LSODE2 canonical Lambdify policy] stage decomposition; argument_binding is combined residual+Jacobian binding"
    );
    println!(
        "matrix | frontend          | policy     | rep | diff_ms | simplify_ms | expr_to_atom_ms | atom_to_expr_ms | sparse_pattern_ms | residual_lambdify_ms | jacobian_lambdify_ms | argument_binding_ms | residual_eval_ms | residual_output_ms | jacobian_eval_ms | jacobian_output_ms"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in stage_rows {
        println!("{row}");
    }
    println!("");
    println!(
        "[LSODE2 canonical Lambdify policy] counter ownership; runtime and preparation probes are attributed"
    );
    println!(
        "matrix | frontend          | policy     | rep | solver_res | res_requests | res_evals | aux_res | prep_res | unattributed_res | solver_jac | jac_requests | jac_evals | aux_jac | unattributed_jac"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in counter_rows {
        println!("{row}");
    }
}
