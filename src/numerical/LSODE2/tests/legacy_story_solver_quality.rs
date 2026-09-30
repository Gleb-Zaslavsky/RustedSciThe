fn stiff_scalar_tracking_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-1000*(y-cos(t))-sin(t)")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.02))
}

fn stiff_switch_acceptance_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-10000*(y-cos(t))-sin(t)")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.25,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.25))
    .with_controller(
        super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
            .with_method_switch_probe_steps(1)
            .with_stiffness_ratio_threshold(10.0)
            .with_convergence_failure_threshold(1)
            .with_rejection_threshold(1),
    )
    .with_faithful_bdf_solve(65_536, 65_536)
}

fn mixed_regime_ramp_config() -> Lsode2ProblemConfig {
    let stiffness = |t: f64| 1.0 + 9_999.0 / (1.0 + (-80.0 * (t - 0.45)).exp());
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("0")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.005,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.005))
    .with_controller(
        super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
            .with_method_switch_probe_steps(1)
            .with_stiffness_ratio_threshold(10.0)
            .with_convergence_failure_threshold(1)
            .with_rejection_threshold(1),
    )
    .with_analytical_callbacks(
        move |t, y: &DVector<f64>| {
            let k = stiffness(t);
            DVector::from_vec(vec![-k * (y[0] - t.cos()) - t.sin()])
        },
        move |t, _y: &DVector<f64>| {
            let k = stiffness(t);
            DMatrix::from_row_slice(1, 1, &[-k])
        },
    )
    .with_faithful_bdf_solve(65_536, 65_536)
}

#[test]
fn lsode2_quality_dashboard_stiff_vs_nonstiff_auto_switch() {
    const REPEATS: usize = 3;
    let scenarios = [("nonstiff-decay", false), ("stiff-tracking", true)];
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    println!(
        "[LSODE2 story] quality dashboard (algorithm focus); counters are counts, time is milliseconds"
    );
    println!(
        "scenario        | matrix | runs | preferred_family | executed_family | switch_reason         | accepted mean+/-std | rejected mean+/-std | nlu/native_linear mean+/-std | jac_refresh mean+/-std | total_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for (scenario_label, is_stiff) in scenarios {
        for matrix in matrices {
            let mut accepted = RaceStats::default();
            let mut rejected = RaceStats::default();
            let mut nlu = RaceStats::default();
            let mut jac_refresh = RaceStats::default();
            let mut total_ms = RaceStats::default();
            let mut final_diff = RaceStats::default();
            let mut preferred: Option<String> = None;
            let mut executed: Option<String> = None;
            let mut switch_reason: Option<String> = None;
            let mut ok = 0usize;

            for _ in 0..REPEATS {
                let base = if is_stiff {
                    stiff_scalar_tracking_config()
                } else {
                    exponential_decay_config()
                };
                let mut config = match matrix {
                    BackendRaceMatrix::Sparse => base.with_native_sparse_faer_backend(),
                    BackendRaceMatrix::Banded => base.with_native_banded_faithful_backend(),
                    BackendRaceMatrix::Dense => unreachable!(),
                };
                config = config.with_faithful_bdf_solve(4096, 4096).with_controller(
                    super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
                        .with_method_switch_probe_steps(1),
                );

                let started = Instant::now();
                let mut solver = match Lsode2Solver::new(config) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let summary = match solver.solve_with_summary() {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                ok += 1;
                total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

                preferred = Some(summary.algorithm.preferred_family.to_string());
                executed = Some(summary.algorithm.executed_family.unwrap_or("-").to_string());
                switch_reason = Some(summary.algorithm.switch_reason.to_string());
                let accepted_steps = summary
                    .native_statistics
                    .native_step_accepts
                    .max(summary.native_statistics.bridge_accepted_steps);
                let rejected_steps = summary.native_statistics.native_step_rejects_error_test
                    + summary.native_statistics.native_step_rejects_nonlinear;
                accepted.push(accepted_steps as f64);
                rejected.push(rejected_steps as f64);
                nlu.push(
                    summary
                        .statistics
                        .bdf_nlu_total
                        .max(summary.native_statistics.native_linear_solve_calls)
                        as f64,
                );
                jac_refresh.push(summary.native_statistics.native_jacobian_refresh_requests as f64);

                if let Some(y) = summary.final_y.as_ref() {
                    let expected = if is_stiff {
                        summary.final_t.unwrap_or(1.0).cos()
                    } else {
                        (-1.0_f64).exp()
                    };
                    final_diff.push((y[0] - expected).abs());
                }
            }

            let status = if ok == REPEATS {
                format!("ok {ok}/{REPEATS}")
            } else if ok == 0 {
                "failed".to_string()
            } else {
                format!("partial {ok}/{REPEATS}")
            };

            let fmt = |s: &RaceStats, p: usize| -> String {
                s.summary()
                    .map(|(m, sd, _, _)| match p {
                        0 => format!("{m:.0}+/-{sd:.0}"),
                        2 => format!("{m:.2}+/-{sd:.2}"),
                        _ => format!("{m:.3}+/-{sd:.3}"),
                    })
                    .unwrap_or_else(|| "-".to_string())
            };

            println!(
                "{:<15} | {:<6} | {:>4} | {:<16} | {:<15} | {:<20} | {:<19} | {:<19} | {:<28} | {:<22} | {:<18} | {:<20} | {}",
                scenario_label,
                matrix.label(),
                format!("{ok}/{REPEATS}"),
                preferred.clone().unwrap_or_else(|| "-".to_string()),
                executed.clone().unwrap_or_else(|| "-".to_string()),
                switch_reason.clone().unwrap_or_else(|| "-".to_string()),
                fmt(&accepted, 2),
                fmt(&rejected, 2),
                fmt(&nlu, 2),
                fmt(&jac_refresh, 2),
                fmt(&total_ms, 3),
                fmt(&final_diff, 3),
                status
            );

            if ok > 0 {
                let p = preferred.clone().unwrap_or_default();
                let r = switch_reason.clone().unwrap_or_default();
                if is_stiff {
                    assert!(
                        p == "adams" || p == "bdf",
                        "{} {} stiff run should expose a valid family, got={}",
                        scenario_label,
                        matrix.label(),
                        p
                    );
                    assert!(
                        r == "switch_probe_warmup"
                            || r == "stiffness_suspected"
                            || r == "convergence_trouble"
                            || (p == "adams" && r == "switch_advantage_not_met")
                            || (p == "bdf" && r == "switch_advantage_not_met"),
                        "{} {} stiff run should report a LSODA-style stiff/warmup reason, got family={} reason={}",
                        scenario_label,
                        matrix.label(),
                        p,
                        r
                    );
                } else {
                    assert!(
                        p == "adams" || p == "bdf",
                        "{} {} non-stiff run should expose a valid family, got={}",
                        scenario_label,
                        matrix.label(),
                        p
                    );
                }
            }
        }
    }
}

#[test]
fn lsode2_nonstiff_adams_corpus_sparse_banded_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::tests::lsode2_nonstiff_adams_corpus_sparse_banded_dashboard",
    );
    const REPEATS: usize = 3;
    let scenarios = [
        ComprehensiveScenario::NonStiffScalarDecay,
        ComprehensiveScenario::NonStiffSystemDecay2,
    ];
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let controllers = [("adams_only", true), ("automatic_adams_bdf", false)];

    println!("[LSODE2 story] non-stiff Adams corpus: fixed Adams and automatic controller routes");
    println!(
        "scenario                  | matrix | controller          | ok/runs | preferred | executed | reason                 | preferred_adams | executed_adams | preferred_bdf | executed_bdf | accepted | rejected | total_ms | max_abs_err | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for scenario in scenarios {
        for matrix in matrices {
            for (controller_label, fixed_adams) in controllers {
                let mut ok = 0usize;
                let mut preferred_adams = RaceStats::default();
                let mut executed_adams = RaceStats::default();
                let mut preferred_bdf = RaceStats::default();
                let mut executed_bdf = RaceStats::default();
                let mut accepted = RaceStats::default();
                let mut rejected = RaceStats::default();
                let mut total_ms = RaceStats::default();
                let mut max_abs_err = RaceStats::default();
                let mut preferred_family = String::new();
                let mut executed_family = String::new();
                let mut switch_reason = String::new();
                let mut first_failure: Option<String> = None;

                for _ in 0..REPEATS {
                    let mut config = scenario.config();
                    config = match matrix {
                        BackendRaceMatrix::Sparse => config.with_native_sparse_faer_backend(),
                        BackendRaceMatrix::Banded => config.with_native_banded_faithful_backend(),
                        BackendRaceMatrix::Dense => unreachable!(),
                    };
                    config = if fixed_adams {
                        config.with_adams_only_controller()
                    } else {
                        config.with_automatic_adams_bdf_controller()
                    }
                    .with_native_solve(65_536, 65_536);

                    let started = Instant::now();
                    let result = Lsode2Solver::new(config)
                        .and_then(|mut solver| solver.solve_with_summary());
                    match result {
                        Ok(summary) => {
                            let final_t = summary.final_t.unwrap_or(1.0);
                            let final_y = summary
                                .final_y
                                .as_ref()
                                .expect("non-stiff Adams story should expose final state");
                            let expected = scenario.expected(final_t);
                            let err = max_abs_diff_vec(final_y, &expected);
                            let native = &summary.native_statistics;

                            ok += 1;
                            preferred_adams.push(native.preferred_adams_count as f64);
                            executed_adams.push(native.executed_adams_count as f64);
                            preferred_bdf.push(native.preferred_bdf_count as f64);
                            executed_bdf.push(native.executed_bdf_count as f64);
                            accepted.push(native.native_step_accepts as f64);
                            rejected.push(
                                (native.native_step_rejects_error_test
                                    + native.native_step_rejects_nonlinear)
                                    as f64,
                            );
                            total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                            max_abs_err.push(err);
                            preferred_family = summary.algorithm.preferred_family.to_string();
                            executed_family = summary
                                .algorithm
                                .executed_family
                                .clone()
                                .unwrap_or("-")
                                .to_string();
                            switch_reason = summary.algorithm.switch_reason.to_string();

                            let story_tolerance = scenario.tolerance().max(5.0e-6);
                            assert!(
                                err <= story_tolerance,
                                "{} {} {} max_abs_err too large: {:e} (tol={:e})",
                                scenario.label(),
                                matrix.label(),
                                controller_label,
                                err,
                                story_tolerance
                            );
                            if fixed_adams {
                                assert_eq!(
                                    summary.algorithm.controller_mode,
                                    "adams_only",
                                    "{} {} must stay in fixed Adams mode",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert_eq!(
                                    summary.algorithm.preferred_family,
                                    "adams",
                                    "{} {} fixed Adams must prefer Adams",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert!(
                                    native.executed_adams_count > 0,
                                    "{} {} fixed Adams should execute Adams steps",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert_eq!(
                                    native.executed_bdf_count,
                                    0,
                                    "{} {} fixed Adams should not execute BDF steps",
                                    scenario.label(),
                                    matrix.label()
                                );
                            } else {
                                assert_eq!(
                                    summary.algorithm.controller_mode,
                                    "automatic_adams_bdf",
                                    "{} {} automatic row must use automatic controller",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert!(
                                    summary.algorithm.preferred_family == "adams"
                                        || summary.algorithm.preferred_family == "bdf",
                                    "{} {} automatic row should expose a valid family, got {}",
                                    scenario.label(),
                                    matrix.label(),
                                    summary.algorithm.preferred_family
                                );
                            }
                        }
                        Err(err) => {
                            if first_failure.is_none() {
                                first_failure = Some(short_error(&err.to_string()));
                            }
                        }
                    }
                }

                let fmt_count = |stats: &RaceStats| {
                    stats
                        .summary()
                        .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                        .unwrap_or_else(|| "-".to_string())
                };
                let fmt_time = |stats: &RaceStats| {
                    stats
                        .summary()
                        .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
                        .unwrap_or_else(|| "-".to_string())
                };
                let fmt_err = |stats: &RaceStats| {
                    stats
                        .summary()
                        .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
                        .unwrap_or_else(|| "-".to_string())
                };
                let status = match first_failure {
                    Some(message) if ok < REPEATS => {
                        format!("partial {ok}/{REPEATS}, first_failure={message}")
                    }
                    _ if ok == REPEATS => format!("ok {ok}/{REPEATS}"),
                    _ => format!("failed {ok}/{REPEATS}"),
                };
                println!(
                    "{:<25} | {:<6} | {:<19} | {:>7} | {:<9} | {:<8} | {:<22} | {:<15} | {:<14} | {:<13} | {:<12} | {:<8} | {:<8} | {:<8} | {:<11} | {}",
                    scenario.label(),
                    matrix.label(),
                    controller_label,
                    format!("{ok}/{REPEATS}"),
                    preferred_family,
                    executed_family,
                    switch_reason,
                    fmt_count(&preferred_adams),
                    fmt_count(&executed_adams),
                    fmt_count(&preferred_bdf),
                    fmt_count(&executed_bdf),
                    fmt_count(&accepted),
                    fmt_count(&rejected),
                    fmt_time(&total_ms),
                    fmt_err(&max_abs_err),
                    status
                );
                assert_eq!(
                    ok,
                    REPEATS,
                    "{} {} {} should complete every run",
                    scenario.label(),
                    matrix.label(),
                    controller_label
                );
            }
        }
    }
}

fn numerical_closure_story_base_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-y1 + 0.1*y2"),
            Expr::parse_expression("-2*y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.5]),
        1.0,
        0.02,
        1e-7,
        1e-9,
    )
    .with_bdf_only_controller()
    .with_faithful_bdf_solve(65_536, 65_536)
}

fn numerical_closure_lambdify_config(matrix: BackendRaceMatrix) -> Lsode2ProblemConfig {
    let config = numerical_closure_story_base_config().with_residual_jacobian_source(
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
    );
    match matrix {
        BackendRaceMatrix::Sparse => config.with_native_sparse_faer_backend(),
        BackendRaceMatrix::Banded => config.with_native_banded_faithful_backend(),
        BackendRaceMatrix::Dense => unreachable!("numerical closure story is sparse/banded only"),
    }
}

fn numerical_closure_native_config(
    matrix: BackendRaceMatrix,
    jacobian_backend: Lsode2JacobianBackend,
) -> Lsode2ProblemConfig {
    let linear_backend = match matrix {
        BackendRaceMatrix::Sparse => Lsode2LinearSolverBackend::SparseFaer,
        BackendRaceMatrix::Banded => Lsode2LinearSolverBackend::BandedFaithful,
        BackendRaceMatrix::Dense => unreachable!("numerical closure story is sparse/banded only"),
    };
    let structure = match matrix {
        BackendRaceMatrix::Sparse => super::Lsode2LinearSystemStructure::Sparse,
        BackendRaceMatrix::Banded => super::Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 },
        BackendRaceMatrix::Dense => unreachable!("numerical closure story is sparse/banded only"),
    };
    numerical_closure_story_base_config()
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0] + 0.1 * y[1], -2.0 * y[1]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(2, 2, &[-1.0, 0.1, 0.0, -2.0]),
        )
        .with_linear_system_structure(structure)
        .with_backend(
            Lsode2BackendConfig::default()
                .with_jacobian_backend(jacobian_backend)
                .with_linear_solver_backend(linear_backend),
        )
}

#[test]
fn lsode2_symbolic_vs_numerical_closure_sparse_banded_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::tests::lsode2_symbolic_vs_numerical_closure_sparse_banded_dashboard",
    );
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = [
        ("Lambdify-AtomViewNative", None),
        (
            "Numerical-AnalyticalJac",
            Some(Lsode2JacobianBackend::AnalyticClosure),
        ),
        (
            "Numerical-FDJac",
            Some(Lsode2JacobianBackend::FiniteDifference),
        ),
    ];

    println!(
        "[LSODE2 story] symbolic Lambdify vs pure numerical closure routes; all time columns are milliseconds"
    );
    println!(
        "matrix | route                   | ok/runs | total_ms mean+/-std [min,max] | final_linf mean+/-std | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | status"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for matrix in matrices {
        let mut baseline_solver = Lsode2Solver::new(numerical_closure_lambdify_config(matrix))
            .expect("numerical closure baseline should build");
        let baseline = baseline_solver
            .solve_with_summary()
            .expect("numerical closure baseline should solve")
            .final_y
            .expect("numerical closure baseline should expose final state");

        for (route, jacobian_backend) in routes {
            let mut row = LargeIvpChunkingRow::new(matrix.label(), route, "native_solve");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match jacobian_backend {
                    None => numerical_closure_lambdify_config(matrix),
                    Some(backend) => numerical_closure_native_config(matrix, backend),
                };
                match run_large_ivp_chunking_sample(config, &baseline) {
                    Ok(sample) => push_large_ivp_sample(&mut row, sample),
                    Err(err) => row.record_failure(err),
                }
            }

            let total = row
                .total_ms
                .summary()
                .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
                .unwrap_or_else(|| "-".to_string());
            let fmt = |stats: &RaceStats| {
                stats
                    .summary()
                    .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
                    .unwrap_or_else(|| "-".to_string())
            };
            let fmt_count = |stats: &RaceStats| {
                stats
                    .summary()
                    .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                    .unwrap_or_else(|| "-".to_string())
            };
            let diff = row
                .final_linf
                .summary()
                .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "{:<6} | {:<23} | {:>7} | {:<31} | {:<21} | {:<14} | {:<14} | {:<12} | {:<11} | {:<11} | {:<9} | {}",
                row.matrix,
                row.route,
                format!("{}/{}", row.runs_ok, row.runs_total),
                total,
                diff,
                fmt_count(&row.residual_calls),
                fmt_count(&row.jacobian_calls),
                fmt_count(&row.linear_calls),
                fmt(&row.residual_ms),
                fmt(&row.jacobian_ms),
                fmt(&row.linear_ms),
                row.status_label()
            );

            assert_eq!(
                row.runs_ok, row.runs_total,
                "{} {} numerical closure story should complete every run",
                row.matrix, row.route
            );
            let (mean_diff, _, _, _) = row
                .final_linf
                .summary()
                .expect("successful row should have final diffs");
            assert!(
                mean_diff <= 2.0e-5,
                "{} {} numerical closure drift too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
            assert!(
                row.residual_calls
                    .summary()
                    .map(|(mean, _, _, _)| mean > 0.0)
                    .unwrap_or(false),
                "{} {} must execute residual callbacks",
                row.matrix,
                row.route
            );
            assert!(
                row.jacobian_calls
                    .summary()
                    .map(|(mean, _, _, _)| mean > 0.0)
                    .unwrap_or(false),
                "{} {} must execute Jacobian work",
                row.matrix,
                row.route
            );
        }
    }
}

#[test]
fn lsode2_mixed_regime_ramp_auto_switch_diagnostic_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::tests::lsode2_mixed_regime_ramp_auto_switch_diagnostic_story",
    );
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    println!("[LSODE2 story] mixed-regime ramp: one IVP starts Adams-capable and becomes stiff");
    println!(
        "matrix | ok/runs | preferred_adams | executed_adams | preferred_bdf | executed_bdf | accepted | rejected | max_stiff | max_rh1 | max_rh2 | total_ms | final_diff | final_family | reason | switch_observed | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for matrix in matrices {
        let mut preferred_adams = RaceStats::default();
        let mut executed_adams = RaceStats::default();
        let mut preferred_bdf = RaceStats::default();
        let mut executed_bdf = RaceStats::default();
        let mut accepted = RaceStats::default();
        let mut rejected = RaceStats::default();
        let mut max_stiff = RaceStats::default();
        let mut max_rh1 = RaceStats::default();
        let mut max_rh2 = RaceStats::default();
        let mut total_ms = RaceStats::default();
        let mut final_diff = RaceStats::default();
        let mut ok = 0usize;
        let mut final_family = "-".to_string();
        let mut reason = "-".to_string();
        let mut first_failure: Option<String> = None;

        for _ in 0..REPEATS {
            let config = match matrix {
                BackendRaceMatrix::Sparse => {
                    mixed_regime_ramp_config().with_native_sparse_faer_backend()
                }
                BackendRaceMatrix::Banded => {
                    mixed_regime_ramp_config().with_native_banded_faithful_backend()
                }
                BackendRaceMatrix::Dense => unreachable!(),
            };
            let started = Instant::now();
            let result =
                Lsode2Solver::new(config).and_then(|mut solver| solver.solve_with_summary());
            match result {
                Ok(summary) => {
                    let native = &summary.native_statistics;
                    let final_t = summary.final_t.unwrap_or(1.0);
                    let final_y = summary
                        .final_y
                        .as_ref()
                        .expect("mixed-regime result should expose final state")[0];
                    let diff = (final_y - final_t.cos()).abs();

                    assert!(
                        summary.algorithm.method_switching_enabled,
                        "{} mixed-regime story must use automatic method selection",
                        matrix.label()
                    );
                    assert!(
                        native.executed_adams_count + native.executed_bdf_count > 0,
                        "{} mixed-regime story should execute at least one method family",
                        matrix.label()
                    );
                    ok += 1;
                    preferred_adams.push(native.preferred_adams_count as f64);
                    executed_adams.push(native.executed_adams_count as f64);
                    preferred_bdf.push(native.preferred_bdf_count as f64);
                    executed_bdf.push(native.executed_bdf_count as f64);
                    accepted.push(native.native_step_accepts as f64);
                    rejected.push(
                        (native.native_step_rejects_error_test
                            + native.native_step_rejects_nonlinear) as f64,
                    );
                    if let Some(solve) = summary.native_integration_solve.as_ref() {
                        let max_or_zero = |values: Vec<Option<f64>>| {
                            values
                                .into_iter()
                                .flatten()
                                .filter(|value| value.is_finite())
                                .fold(0.0_f64, f64::max)
                        };
                        max_stiff.push(max_or_zero(
                            solve
                                .attempt_reports
                                .iter()
                                .map(|report| report.telemetry.stiffness_ratio)
                                .collect(),
                        ));
                        max_rh1.push(max_or_zero(
                            solve
                                .attempt_reports
                                .iter()
                                .map(|report| report.telemetry.adams_step_size_cap_estimate)
                                .collect(),
                        ));
                        max_rh2.push(max_or_zero(
                            solve
                                .attempt_reports
                                .iter()
                                .map(|report| report.telemetry.bdf_step_size_cap_estimate)
                                .collect(),
                        ));
                    }
                    total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                    final_diff.push(diff);
                    final_family = summary
                        .algorithm
                        .executed_family
                        .unwrap_or(summary.algorithm.active_family)
                        .to_string();
                    reason = summary.algorithm.switch_reason.to_string();
                }
                Err(err) => {
                    if first_failure.is_none() {
                        first_failure = Some(short_error(&err.to_string()));
                    }
                }
            }
        }

        let fmt_count = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let fmt_time = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let fmt_err = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let status = match first_failure {
            Some(message) if ok < REPEATS => {
                format!("partial {ok}/{REPEATS}, first_failure={message}")
            }
            _ if ok == REPEATS => format!("ok {ok}/{REPEATS}"),
            _ => format!("failed {ok}/{REPEATS}"),
        };
        let switch_observed = match (executed_adams.summary(), executed_bdf.summary()) {
            (Some((adams, _, _, _)), Some((bdf, _, _, _))) if adams > 0.0 && bdf > 0.0 => {
                "adams+bdf"
            }
            (Some((adams, _, _, _)), _) if adams > 0.0 => "adams_only_current_limit",
            (_, Some((bdf, _, _, _))) if bdf > 0.0 => "bdf_only_current_limit",
            _ => "none",
        };
        println!(
            "{:<6} | {:>7} | {:<15} | {:<14} | {:<13} | {:<12} | {:<8} | {:<8} | {:<9} | {:<7} | {:<7} | {:<8} | {:<10} | {:<12} | {:<24} | {:<26} | {}",
            matrix.label(),
            format!("{ok}/{REPEATS}"),
            fmt_count(&preferred_adams),
            fmt_count(&executed_adams),
            fmt_count(&preferred_bdf),
            fmt_count(&executed_bdf),
            fmt_count(&accepted),
            fmt_count(&rejected),
            fmt_count(&max_stiff),
            fmt_count(&max_rh1),
            fmt_count(&max_rh2),
            fmt_time(&total_ms),
            fmt_err(&final_diff),
            final_family,
            reason,
            switch_observed,
            status
        );
        assert_eq!(
            ok,
            REPEATS,
            "{} should complete every mixed-regime run",
            matrix.label()
        );
    }
}

#[test]
fn lsode2_mixed_regime_ramp_native_switches_adams_to_bdf_acceptance() {
    for matrix in [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded] {
        let config = match matrix {
            BackendRaceMatrix::Sparse => {
                mixed_regime_ramp_config().with_native_sparse_faer_backend()
            }
            BackendRaceMatrix::Banded => {
                mixed_regime_ramp_config().with_native_banded_faithful_backend()
            }
            BackendRaceMatrix::Dense => unreachable!(),
        };
        let summary = Lsode2Solver::new(config)
            .and_then(|mut solver| solver.solve_with_summary())
            .unwrap_or_else(|err| {
                panic!(
                    "{} mixed-regime native switch solve failed: {err}",
                    matrix.label()
                )
            });
        let native = &summary.native_statistics;
        assert!(
            native.executed_adams_count > 0,
            "{} mixed-regime solve must execute Adams before stiffness appears",
            matrix.label()
        );
        assert!(
            native.executed_bdf_count > 0,
            "{} mixed-regime solve must switch to BDF after stiffness appears",
            matrix.label()
        );
        assert!(
            summary.algorithm.method_switching_enabled,
            "{} mixed-regime solve must keep automatic switching enabled",
            matrix.label()
        );
        let final_t = summary
            .final_t
            .expect("mixed-regime solve should expose final_t");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("mixed-regime solve should expose final_y")[0];
        let final_diff = (final_y - final_t.cos()).abs();
        assert!(
            final_diff <= 1.0e-6,
            "{} mixed-regime final drift too large after Adams/BDF switch: {final_diff:e}",
            matrix.label()
        );
    }
}

#[test]
fn lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::tests::lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf",
    );
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    println!("[LSODE2 story] stiff-switch acceptance: automatic controller must execute BDF");
    println!(
        "matrix | ok/runs | preferred_bdf mean+/-std | executed_bdf mean+/-std | accepted mean+/-std | rejected mean+/-std | total_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for matrix in matrices {
        let mut preferred_bdf = RaceStats::default();
        let mut executed_bdf = RaceStats::default();
        let mut accepted = RaceStats::default();
        let mut rejected = RaceStats::default();
        let mut total_ms = RaceStats::default();
        let mut final_diff = RaceStats::default();
        let mut ok = 0usize;

        for _ in 0..REPEATS {
            let config = match matrix {
                BackendRaceMatrix::Sparse => {
                    stiff_switch_acceptance_config().with_native_sparse_faer_backend()
                }
                BackendRaceMatrix::Banded => {
                    stiff_switch_acceptance_config().with_native_banded_faithful_backend()
                }
                BackendRaceMatrix::Dense => unreachable!(),
            };
            let started = Instant::now();
            let mut solver = Lsode2Solver::new(config).expect("stiff-switch config should build");
            let summary = solver
                .solve_with_summary()
                .expect("stiff-switch acceptance solve should finish");

            let native = &summary.native_statistics;
            assert!(
                native.preferred_bdf_count > 0,
                "{} automatic stiff-switch run should prefer BDF at least once",
                matrix.label()
            );
            assert!(
                native.executed_bdf_count > 0,
                "{} automatic stiff-switch run should execute BDF at least once",
                matrix.label()
            );
            assert!(
                summary.algorithm.method_switching_enabled,
                "{} run must use automatic method selection",
                matrix.label()
            );

            ok += 1;
            preferred_bdf.push(native.preferred_bdf_count as f64);
            executed_bdf.push(native.executed_bdf_count as f64);
            accepted.push(native.native_step_accepts as f64);
            rejected.push(
                (native.native_step_rejects_error_test + native.native_step_rejects_nonlinear)
                    as f64,
            );
            total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            let final_t = summary.final_t.unwrap_or(1.0);
            let final_y = summary
                .final_y
                .as_ref()
                .expect("stiff-switch result should expose final state")[0];
            final_diff.push((final_y - final_t.cos()).abs());
        }

        let fmt = |stats: &RaceStats, digits: usize| {
            stats
                .summary()
                .map(|(mean, std, _, _)| match digits {
                    0 => format!("{mean:.0}+/-{std:.0}"),
                    2 => format!("{mean:.2}+/-{std:.2}"),
                    _ => format!("{mean:.3e}+/-{std:.1e}"),
                })
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:>7} | {:<24} | {:<23} | {:<19} | {:<19} | {:<19} | {:<21} | ok {}/{}",
            matrix.label(),
            format!("{ok}/{REPEATS}"),
            fmt(&preferred_bdf, 2),
            fmt(&executed_bdf, 2),
            fmt(&accepted, 2),
            fmt(&rejected, 2),
            fmt(&total_ms, 2),
            fmt(&final_diff, 3),
            ok,
            REPEATS,
        );
        assert_eq!(
            ok,
            REPEATS,
            "{} should complete every stiff-switch acceptance run",
            matrix.label()
        );
    }
}
