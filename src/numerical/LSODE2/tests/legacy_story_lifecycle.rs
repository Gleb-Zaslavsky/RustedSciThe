fn combustion_aot_matrix_config(
    matrix: BackendRaceMatrix,
    toolchain: AotStoryToolchain,
    parallel: bool,
    output_dir: PathBuf,
) -> Lsode2ProblemConfig {
    let generated = toolchain.apply_generated(
        SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir).with_build_policy(
            SymbolicIvpAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release,
            },
        ),
    );
    let mut config = combustion_symbolic_matrix_config(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        toolchain.as_source(),
        Some(generated),
    )
    .with_bdf_only_controller();
    if parallel {
        config = config.with_aot_parallel_chunking(2);
    }
    config
}

fn combustion_tcc_lifecycle_config(
    matrix: BackendRaceMatrix,
    output_dir: PathBuf,
    build_policy: SymbolicIvpAotBuildPolicy,
) -> Lsode2ProblemConfig {
    combustion_tcc_lifecycle_config_for_frontend(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        output_dir,
        build_policy,
    )
}

fn combustion_tcc_lifecycle_config_for_frontend(
    matrix: BackendRaceMatrix,
    frontend: Lsode2SymbolicAssemblyBackend,
    output_dir: PathBuf,
    build_policy: SymbolicIvpAotBuildPolicy,
) -> Lsode2ProblemConfig {
    let generated = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir)
        .with_c_tcc()
        .with_build_policy(build_policy);
    combustion_symbolic_matrix_config(
        matrix,
        frontend,
        Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
        Some(generated),
    )
    .with_bdf_only_controller()
}

struct Lsode2LifecycleRow {
    matrix: &'static str,
    phase: &'static str,
    build_policy: &'static str,
    total_ms: f64,
    prepare_ms: f64,
    solve_ms: f64,
    final_diff: f64,
    residual_calls: f64,
    jacobian_calls: f64,
    linear_calls: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    status: String,
}

impl Lsode2LifecycleRow {
    fn from_sample(
        matrix: &'static str,
        phase: &'static str,
        build_policy: &'static str,
        sample: CombustionStorySample,
    ) -> Self {
        Self {
            matrix,
            phase,
            build_policy,
            total_ms: sample.0,
            prepare_ms: sample.1,
            solve_ms: sample.2,
            final_diff: sample.3,
            residual_calls: sample.4,
            jacobian_calls: sample.5,
            linear_calls: sample.6,
            residual_ms: sample.13,
            jacobian_ms: sample.14,
            linear_ms: sample.15,
            status: "ok".to_string(),
        }
    }

    fn failed(
        matrix: &'static str,
        phase: &'static str,
        build_policy: &'static str,
        err: String,
    ) -> Self {
        Self {
            matrix,
            phase,
            build_policy,
            total_ms: f64::NAN,
            prepare_ms: f64::NAN,
            solve_ms: f64::NAN,
            final_diff: f64::NAN,
            residual_calls: f64::NAN,
            jacobian_calls: f64::NAN,
            linear_calls: f64::NAN,
            residual_ms: f64::NAN,
            jacobian_ms: f64::NAN,
            linear_ms: f64::NAN,
            status: format!("failed({})", short_error(&err)),
        }
    }

    fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

fn run_lsode2_lifecycle_row(
    matrix: BackendRaceMatrix,
    phase: &'static str,
    build_policy: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_a: f64,
) -> Lsode2LifecycleRow {
    match run_combustion_story_sample_result(phase, config, baseline_final_a) {
        Ok(sample) => Lsode2LifecycleRow::from_sample(matrix.label(), phase, build_policy, sample),
        Err(err) => Lsode2LifecycleRow::failed(matrix.label(), phase, build_policy, err),
    }
}

fn fmt_story_value(value: f64, decimals: usize) -> String {
    if value.is_finite() {
        format!("{value:.decimals$}")
    } else {
        "-".to_string()
    }
}

fn print_lsode2_lifecycle_table(title: &str, rows: &[Lsode2LifecycleRow]) {
    println!("[LSODE2 lifecycle] {title}: correctness/backend policy");
    println!("matrix | phase      | build_policy    | final_diff | status");
    println!("--------------------------------------------------------------------------");
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:<15} | {:>10} | {}",
            row.matrix,
            row.phase,
            row.build_policy,
            if row.final_diff.is_finite() {
                format!("{:.3e}", row.final_diff)
            } else {
                "-".to_string()
            },
            row.status
        );
    }

    println!("[LSODE2 lifecycle] {title}: wall-clock and hot stages; milliseconds");
    println!(
        "matrix | phase      | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms"
    );
    println!(
        "------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:>8} | {:>10} | {:>8} | {:>11} | {:>11} | {:>9}",
            row.matrix,
            row.phase,
            fmt_story_value(row.total_ms, 3),
            fmt_story_value(row.prepare_ms, 3),
            fmt_story_value(row.solve_ms, 3),
            fmt_story_value(row.residual_ms, 3),
            fmt_story_value(row.jacobian_ms, 3),
            fmt_story_value(row.linear_ms, 3),
        );
    }

    println!("[LSODE2 lifecycle] {title}: numerical work; counters are counts");
    println!("matrix | phase      | residual_calls | jacobian_calls | linear_calls");
    println!("------------------------------------------------------------------------");
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:>14} | {:>14} | {:>12}",
            row.matrix,
            row.phase,
            fmt_story_value(row.residual_calls, 0),
            fmt_story_value(row.jacobian_calls, 0),
            fmt_story_value(row.linear_calls, 0),
        );
    }
}

fn warm_cooldown_ms() -> u64 {
    std::env::var("LSODE2_WARM_COOLDOWN_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_000)
}

fn lifecycle_repetitions(env_name: &str, default: usize) -> usize {
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

pub(crate) fn run_lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story() {
    let strict_repeats = lifecycle_repetitions("LSODE2_PREBUILT_REPEATS", 3);
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let mut rows = Vec::new();

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let reference_config = combustion_symbolic_matrix_config(
            matrix,
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        )
        .with_bdf_only_controller();
        let mut reference_solver =
            Lsode2Solver::new(reference_config).expect("Lambdify reference should build");
        let reference = reference_solver
            .solve_with_summary()
            .expect("Lambdify reference should solve");
        let baseline_final_a = reference
            .final_y
            .expect("reference final state should exist")[0];
        baselines.insert(matrix.label(), baseline_final_a);
    }

    for matrix in matrices {
        let baseline = *baselines
            .get(matrix.label())
            .expect("matrix baseline should exist");
        let output_dir = PathBuf::from(format!(
            "target/l2-aot-prebuilt-lifecycle/{}/{}",
            matrix.label().to_ascii_lowercase(),
            unique_story_short_tag()
        ));
        let build_config = combustion_tcc_lifecycle_config(
            matrix,
            output_dir.clone(),
            SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            },
        );
        rows.push(run_lsode2_lifecycle_row(
            matrix,
            "build",
            "BuildIfMissing",
            build_config,
            baseline,
        ));

        let strict_config = combustion_tcc_lifecycle_config(
            matrix,
            output_dir,
            SymbolicIvpAotBuildPolicy::RequirePrebuilt,
        );
        for _ in 0..strict_repeats {
            rows.push(run_lsode2_lifecycle_row(
                matrix,
                "prebuilt",
                "RequirePrebuilt",
                strict_config.clone(),
                baseline,
            ));
        }
    }

    print_lsode2_lifecycle_table(
        "combustion AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
        &rows,
    );

    assert!(
        rows.iter().all(Lsode2LifecycleRow::is_ok),
        "all LSODE2 BuildIfMissing and RequirePrebuilt lifecycle rows must solve"
    );
    assert!(
        rows.iter().all(|row| row.final_diff <= 2.0e-4),
        "LSODE2 prebuilt lifecycle rows must remain numerically equivalent"
    );
    assert!(
        rows.iter()
            .filter(|row| row.phase == "prebuilt")
            .all(|row| row.build_policy == "RequirePrebuilt" && row.prepare_ms < 20.0),
        "strict prebuilt rows should reuse already linked compiled callbacks without a cold rebuild"
    );
}

pub(crate) fn run_lsode2_combustion_sparse_banded_all_frontends_tcc_build_then_require_prebuilt_story(
) {
    let strict_repeats = lifecycle_repetitions("LSODE2_PREBUILT_REPEATS", 3);
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let frontends = [
        (
            "ExprLegacy",
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            "expr-build",
            "expr-prebuilt",
        ),
        (
            "AtomViewNative",
            Lsode2SymbolicAssemblyBackend::AtomView,
            "atom-build",
            "atom-prebuilt",
        ),
    ];
    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let reference_config = combustion_symbolic_matrix_config(
            matrix,
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        )
        .with_bdf_only_controller();
        let mut reference_solver =
            Lsode2Solver::new(reference_config).expect("Lambdify reference should build");
        let reference = reference_solver
            .solve_with_summary()
            .expect("Lambdify reference should solve");
        baselines.insert(
            matrix.label(),
            reference
                .final_y
                .expect("reference final state should exist")[0],
        );
    }

    let mut rows = Vec::new();
    for (frontend_label, frontend, build_phase, prebuilt_phase) in frontends {
        for matrix in matrices {
            let baseline = *baselines
                .get(matrix.label())
                .expect("matrix baseline should exist");
            let route_dir = format!(
                "target/l2-aot-all-route-lifecycle/{}/{}/{}",
                frontend_label.to_ascii_lowercase(),
                matrix.label().to_ascii_lowercase(),
                unique_story_short_tag()
            );
            let output_dir = PathBuf::from(route_dir);
            let build_config = combustion_tcc_lifecycle_config_for_frontend(
                matrix,
                frontend,
                output_dir.clone(),
                SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
            );
            rows.push(run_lsode2_lifecycle_row(
                matrix,
                build_phase,
                "BuildIfMissing",
                build_config,
                baseline,
            ));

            let strict_config = combustion_tcc_lifecycle_config_for_frontend(
                matrix,
                frontend,
                output_dir,
                SymbolicIvpAotBuildPolicy::RequirePrebuilt,
            );
            for _ in 0..strict_repeats {
                rows.push(run_lsode2_lifecycle_row(
                    matrix,
                    prebuilt_phase,
                    "RequirePrebuilt",
                    strict_config.clone(),
                    baseline,
                ));
            }
        }
    }

    print_lsode2_lifecycle_table(
        "combustion ExprLegacy/AtomView Sparse/Banded BuildIfMissing -> RequirePrebuilt lifecycle",
        &rows,
    );
    assert!(
        rows.iter().all(Lsode2LifecycleRow::is_ok),
        "all ExprLegacy and AtomView lifecycle rows must solve"
    );
    assert!(
        rows.iter().all(|row| row.final_diff <= 2.0e-4),
        "all frontend lifecycle rows must remain numerically equivalent"
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.build_policy == "BuildIfMissing")
            .count(),
        matrices.len() * frontends.len(),
        "each frontend and matrix must have one cold build row"
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.build_policy == "RequirePrebuilt")
            .count(),
        matrices.len() * frontends.len() * strict_repeats,
        "each frontend and matrix must have all strict reuse rows"
    );
    assert!(
        rows.iter()
            .filter(|row| row.build_policy == "RequirePrebuilt")
            .all(|row| row.prepare_ms < 20.0),
        "strict rows should reuse linked artifacts without a cold rebuild"
    );
}

fn print_lsode2_warm_prebuilt_table(
    build_row: &Lsode2LifecycleRow,
    rows: &[(usize, usize, Lsode2LifecycleRow)],
) {
    println!("[LSODE2 warm] Banded AtomViewNative Lambdify vs tcc RequirePrebuilt setup row");
    println!(
        "phase | build_policy    | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms | final_diff | status"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<5} | {:<15} | {:>8} | {:>10} | {:>8} | {:>11} | {:>11} | {:>9} | {:>10} | {}",
        build_row.phase,
        build_row.build_policy,
        fmt_story_value(build_row.total_ms, 3),
        fmt_story_value(build_row.prepare_ms, 3),
        fmt_story_value(build_row.solve_ms, 3),
        fmt_story_value(build_row.residual_ms, 3),
        fmt_story_value(build_row.jacobian_ms, 3),
        fmt_story_value(build_row.linear_ms, 3),
        if build_row.final_diff.is_finite() {
            format!("{:.3e}", build_row.final_diff)
        } else {
            "-".to_string()
        },
        build_row.status
    );

    println!(
        "[LSODE2 warm] measured rows after cooldown_ms={}; build row excluded",
        warm_cooldown_ms()
    );
    println!(
        "rep | pos | phase      | build_policy    | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms | final_diff | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (rep, pos, row) in rows {
        println!(
            "{:>3} | {:>3} | {:<10} | {:<15} | {:>8} | {:>10} | {:>8} | {:>11} | {:>11} | {:>9} | {:>10} | {}",
            rep,
            pos,
            row.phase,
            row.build_policy,
            fmt_story_value(row.total_ms, 3),
            fmt_story_value(row.prepare_ms, 3),
            fmt_story_value(row.solve_ms, 3),
            fmt_story_value(row.residual_ms, 3),
            fmt_story_value(row.jacobian_ms, 3),
            fmt_story_value(row.linear_ms, 3),
            if row.final_diff.is_finite() {
                format!("{:.3e}", row.final_diff)
            } else {
                "-".to_string()
            },
            row.status
        );
    }

    println!("[LSODE2 warm] paired summary; milliseconds");
    println!(
        "phase      | runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | jacobian_ms mean+/-std | max_final_diff"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------"
    );
    for phase in ["lambdify", "prebuilt"] {
        let phase_rows = rows
            .iter()
            .filter_map(|(_, _, row)| (row.phase == phase && row.is_ok()).then_some(row))
            .collect::<Vec<_>>();
        let mut total = RaceStats::default();
        let mut prepare = RaceStats::default();
        let mut solve = RaceStats::default();
        let mut jac = RaceStats::default();
        for row in &phase_rows {
            total.push(row.total_ms);
            prepare.push(row.prepare_ms);
            solve.push(row.solve_ms);
            jac.push(row.jacobian_ms);
        }
        let fmt_agg = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
                .unwrap_or_else(|| "-".to_string())
        };
        let fmt_agg_short = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let max_diff = phase_rows
            .iter()
            .map(|row| row.final_diff)
            .fold(0.0_f64, f64::max);
        println!(
            "{:<10} | {:>4} | {:<31} | {:<21} | {:<19} | {:<21} | {:.3e}",
            phase,
            phase_rows.len(),
            fmt_agg(&total),
            fmt_agg_short(&prepare),
            fmt_agg_short(&solve),
            fmt_agg_short(&jac),
            max_diff
        );
    }
}

pub(crate) fn run_lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story() {
    let repetitions = lifecycle_repetitions("LSODE2_WARM_REPEATS", 5);
    let cooldown_ms = warm_cooldown_ms();
    let matrix = BackendRaceMatrix::Banded;

    let reference_config = combustion_symbolic_matrix_config(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        Lsode2SymbolicExecutionMode::LambdifyExpr,
        None,
    )
    .with_bdf_only_controller();
    let mut reference_solver =
        Lsode2Solver::new(reference_config).expect("Lambdify reference should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("Lambdify reference should solve");
    let baseline_final_a = reference
        .final_y
        .expect("reference final state should exist")[0];

    let output_dir = PathBuf::from(format!(
        "target/l2-aot-warm-prebuilt/banded/{}",
        unique_story_short_tag()
    ));
    let build_config = combustion_tcc_lifecycle_config(
        matrix,
        output_dir.clone(),
        SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        },
    );
    let build_row = run_lsode2_lifecycle_row(
        matrix,
        "build",
        "BuildIfMissing",
        build_config,
        baseline_final_a,
    );
    assert!(
        build_row.is_ok() && build_row.final_diff <= 2.0e-4,
        "setup BuildIfMissing row must install a correct compiled backend"
    );

    let prebuilt_config = combustion_tcc_lifecycle_config(
        matrix,
        output_dir,
        SymbolicIvpAotBuildPolicy::RequirePrebuilt,
    );
    let lambdify_config = combustion_symbolic_matrix_config(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        Lsode2SymbolicExecutionMode::LambdifyExpr,
        None,
    )
    .with_bdf_only_controller();

    let mut rows = Vec::with_capacity(repetitions * 2);
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
            let (policy, config) = if phase == "lambdify" {
                ("UseIfAvailable", lambdify_config.clone())
            } else {
                ("RequirePrebuilt", prebuilt_config.clone())
            };
            let row = run_lsode2_lifecycle_row(matrix, phase, policy, config, baseline_final_a);
            rows.push((repetition, position + 1, row));
        }
    }

    print_lsode2_warm_prebuilt_table(&build_row, &rows);

    assert!(
        rows.iter().all(|(_, _, row)| row.is_ok()),
        "all LSODE2 warm Lambdify/prebuilt rows must solve"
    );
    assert!(
        rows.iter().all(|(_, _, row)| row.final_diff <= 2.0e-4),
        "all LSODE2 warm Lambdify/prebuilt rows must match the common reference"
    );
    assert_eq!(
        rows.iter()
            .filter(|(_, _, row)| row.phase == "lambdify")
            .count(),
        repetitions,
        "warm comparison must collect one Lambdify row per repetition"
    );
    assert_eq!(
        rows.iter()
            .filter(|(_, _, row)| row.phase == "prebuilt")
            .count(),
        repetitions,
        "warm comparison must collect one prebuilt row per repetition"
    );
    assert!(
        rows.iter()
            .filter(|(_, _, row)| row.phase == "prebuilt")
            .all(|(_, _, row)| row.build_policy == "RequirePrebuilt" && row.prepare_ms < 20.0),
        "measured prebuilt rows must stay strict and avoid a cold rebuild"
    );
}

fn large_ivp_chunking_dim() -> usize {
    lifecycle_repetitions("LSODE2_LARGE_CHUNK_DIM", 96).max(8)
}

fn large_ivp_chunking_dims() -> Vec<usize> {
    std::env::var("LSODE2_LARGE_CHUNK_DIMS")
        .ok()
        .map(|raw| {
            raw.split(|ch| ch == ',' || ch == ';' || ch == ' ')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|n| *n >= 8)
                .collect::<Vec<_>>()
        })
        .filter(|dims| !dims.is_empty())
        .unwrap_or_else(|| vec![large_ivp_chunking_dim()])
}

fn large_ivp_chunking_repeats() -> usize {
    lifecycle_repetitions("LSODE2_LARGE_CHUNK_REPEATS", 3)
}

fn large_ivp_chunking_chunks() -> usize {
    lifecycle_repetitions("LSODE2_LARGE_CHUNK_TARGET", 4)
}

pub(super) fn large_diffusion_chain_config(n: usize) -> Lsode2ProblemConfig {
    let diffusion = 6.0_f64;
    let mut equations = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut y0 = Vec::with_capacity(n);

    for i in 0..n {
        let var = format!("y{i}");
        values.push(var.clone());
        y0.push(((i as f64 + 1.0) * 0.071).sin() * 0.2 + 1.0);

        let lambda = 35.0 + (i % 11) as f64 * 4.0;
        let mut rhs = format!("-{lambda:.8}*{var}");
        let mut neighbor_count = 0usize;
        if i > 0 {
            rhs.push_str(&format!(" + {diffusion:.8}*y{}", i - 1));
            neighbor_count += 1;
        }
        if i + 1 < n {
            rhs.push_str(&format!(" + {diffusion:.8}*y{}", i + 1));
            neighbor_count += 1;
        }
        if neighbor_count > 0 {
            rhs.push_str(&format!(
                " - {:.8}*{var}",
                diffusion * neighbor_count as f64
            ));
        }
        rhs.push_str(" + 0.001*cos(t)");
        equations.push(Expr::parse_expression(&rhs));
    }

    Lsode2ProblemConfig::new(
        equations,
        values,
        "t".to_string(),
        0.0,
        DVector::from_vec(y0),
        0.12,
        0.004,
        1e-5,
        1e-8,
    )
    .with_first_step(Some(0.001))
    .with_bdf_only_controller()
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
}

fn large_ivp_lambdify_config(matrix: BackendRaceMatrix, n: usize) -> Lsode2ProblemConfig {
    match matrix {
        BackendRaceMatrix::Sparse => {
            large_diffusion_chain_config(n).with_native_sparse_faer_backend()
        }
        BackendRaceMatrix::Banded => {
            large_diffusion_chain_config(n).with_native_banded_faithful_backend()
        }
        BackendRaceMatrix::Dense => unreachable!("large chunking story is sparse/banded only"),
    }
}

fn large_ivp_tcc_config(
    matrix: BackendRaceMatrix,
    n: usize,
    output_dir: PathBuf,
    build_policy: SymbolicIvpAotBuildPolicy,
    target_chunks: usize,
) -> Lsode2ProblemConfig {
    let generated = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir)
        .with_c_tcc()
        .with_build_policy(build_policy);
    let generated = Lsode2BackendConfig::native_sparse_faer()
        .with_generated_backend(generated)
        .with_generated_backend_target_chunks(target_chunks.max(1), target_chunks.max(1))
        .generated_backend;
    let base = large_diffusion_chain_config(n).with_residual_jacobian_source(
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            },
        },
    );
    match matrix {
        BackendRaceMatrix::Sparse => base.with_native_sparse_faer_generated_backend(generated),
        BackendRaceMatrix::Banded => base.with_native_banded_faithful_generated_backend(generated),
        BackendRaceMatrix::Dense => unreachable!("large chunking story is sparse/banded only"),
    }
}

pub(super) struct LargeIvpChunkingRow {
    pub(super) matrix: &'static str,
    pub(super) route: &'static str,
    pub(super) build_policy: &'static str,
    pub(super) runs_ok: usize,
    pub(super) runs_total: usize,
    pub(super) first_failure: Option<String>,
    pub(super) total_ms: RaceStats,
    pub(super) prepare_ms: RaceStats,
    pub(super) solve_ms: RaceStats,
    pub(super) final_linf: RaceStats,
    pub(super) residual_calls: RaceStats,
    pub(super) jacobian_calls: RaceStats,
    pub(super) linear_calls: RaceStats,
    pub(super) residual_ms: RaceStats,
    pub(super) jacobian_ms: RaceStats,
    pub(super) linear_ms: RaceStats,
}

impl LargeIvpChunkingRow {
    pub(super) fn new(
        matrix: &'static str,
        route: &'static str,
        build_policy: &'static str,
    ) -> Self {
        Self {
            matrix,
            route,
            build_policy,
            runs_ok: 0,
            runs_total: 0,
            first_failure: None,
            total_ms: RaceStats::default(),
            prepare_ms: RaceStats::default(),
            solve_ms: RaceStats::default(),
            final_linf: RaceStats::default(),
            residual_calls: RaceStats::default(),
            jacobian_calls: RaceStats::default(),
            linear_calls: RaceStats::default(),
            residual_ms: RaceStats::default(),
            jacobian_ms: RaceStats::default(),
            linear_ms: RaceStats::default(),
        }
    }

    pub(super) fn record_failure(&mut self, err: impl AsRef<str>) {
        if self.first_failure.is_none() {
            self.first_failure = Some(short_error(err.as_ref()));
        }
    }

    pub(super) fn status_label(&self) -> String {
        let base = if self.runs_ok == self.runs_total {
            format!("ok {}/{}", self.runs_ok, self.runs_total)
        } else if self.runs_ok == 0 {
            format!("failed {}/{}", self.runs_ok, self.runs_total)
        } else {
            format!("partial {}/{}", self.runs_ok, self.runs_total)
        };
        match &self.first_failure {
            Some(first_failure) if self.runs_ok < self.runs_total => {
                format!("{base}, first_failure={first_failure}")
            }
            _ => base,
        }
    }
}

pub(super) fn run_large_ivp_chunking_sample(
    config: Lsode2ProblemConfig,
    baseline: &DVector<f64>,
) -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let total_started = Instant::now();
    let mut solver = Lsode2Solver::new(config)
        .map_err(|err| format!("new_error({})", short_error(&err.to_string())))?;
    let prepare_started = Instant::now();
    solver
        .prepare()
        .map_err(|err| format!("prepare_error({})", short_error(&err.to_string())))?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    let summary = solver
        .solve_with_summary()
        .map_err(|err| format!("solve_error({})", short_error(&err.to_string())))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary
        .final_y
        .ok_or_else(|| "solve_error(missing final state)".to_string())?;
    if final_y.len() != baseline.len() {
        return Err(format!(
            "solve_error(final size mismatch {} != {})",
            final_y.len(),
            baseline.len()
        ));
    }
    let final_linf = final_y
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let stats = summary.native_statistics;
    Ok((
        total_ms,
        prepare_ms,
        solve_ms,
        final_linf,
        stats.native_residual_calls as f64,
        stats.native_jacobian_calls as f64,
        stats.native_linear_solve_calls as f64,
        stats.native_residual_ms_total,
        stats.native_jacobian_ms_total,
        stats.native_linear_solve_ms_total,
    ))
}

pub(super) fn push_large_ivp_sample(
    row: &mut LargeIvpChunkingRow,
    sample: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64),
) {
    row.runs_ok += 1;
    row.total_ms.push(sample.0);
    row.prepare_ms.push(sample.1);
    row.solve_ms.push(sample.2);
    row.final_linf.push(sample.3);
    row.residual_calls.push(sample.4);
    row.jacobian_calls.push(sample.5);
    row.linear_calls.push(sample.6);
    row.residual_ms.push(sample.7);
    row.jacobian_ms.push(sample.8);
    row.linear_ms.push(sample.9);
}

fn print_large_ivp_chunking_tables(title: &str, n: usize, rows: &[LargeIvpChunkingRow]) {
    println!("[LSODE2 large chunking] {title}: n={n}; correctness/wall-clock");
    println!(
        "matrix | route           | policy          | ok/runs | total_ms mean+/-std [min,max] | prepare_ms | solve_ms | final_linf | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let fmt = |stats: &RaceStats, precision: usize| {
            stats
                .summary()
                .map(|(m, s, _, _)| {
                    if precision == 3 {
                        format!("{m:.3}+/-{s:.3}")
                    } else {
                        format!("{m:.3e}+/-{s:.1e}")
                    }
                })
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<15} | {:<15} | {:>7} | {:<31} | {:<10} | {:<10} | {:<12} | {}",
            row.matrix,
            row.route,
            row.build_policy,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            fmt(&row.prepare_ms, 3),
            fmt(&row.solve_ms, 3),
            fmt(&row.final_linf, 0),
            row.status_label()
        );
    }

    println!("[LSODE2 large chunking] {title}: hot-stage timers and counters");
    println!(
        "matrix | route           | residual_ms | jacobian_ms | linear_ms | residual_calls | jacobian_calls | linear_calls"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
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
        println!(
            "{:<6} | {:<15} | {:<11} | {:<11} | {:<9} | {:<14} | {:<14} | {}",
            row.matrix,
            row.route,
            fmt(&row.residual_ms),
            fmt(&row.jacobian_ms),
            fmt(&row.linear_ms),
            fmt_count(&row.residual_calls),
            fmt_count(&row.jacobian_calls),
            fmt_count(&row.linear_calls),
        );
    }
}

pub(crate) fn run_lsode2_large_chain_tcc_chunking_sparse_banded_warm_story() {
    let dims = large_ivp_chunking_dims();
    let repeats = large_ivp_chunking_repeats();
    let target_chunks = large_ivp_chunking_chunks();
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    for n in dims {
        let mut rows = Vec::new();

        for matrix in matrices {
            let mut baseline_solver =
                Lsode2Solver::new(large_ivp_lambdify_config(matrix, n)).expect("baseline config");
            let baseline_summary = baseline_solver
                .solve_with_summary()
                .expect("baseline solve should finish");
            let baseline = baseline_summary
                .final_y
                .expect("baseline final state should exist");

            let whole_dir = PathBuf::from(format!(
                "target/l2-large-chain-chunking/n{}/{}/whole/{}",
                n,
                matrix.label().to_ascii_lowercase(),
                unique_story_short_tag()
            ));
            let chunk_dir = PathBuf::from(format!(
                "target/l2-large-chain-chunking/n{}/{}/chunk{}/{}",
                n,
                matrix.label().to_ascii_lowercase(),
                target_chunks,
                unique_story_short_tag()
            ));

            for (route, dir, chunks) in [
                ("tcc-whole", whole_dir.clone(), 1usize),
                ("tcc-chunk", chunk_dir.clone(), target_chunks),
            ] {
                let setup = large_ivp_tcc_config(
                    matrix,
                    n,
                    dir.clone(),
                    SymbolicIvpAotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Release,
                    },
                    chunks,
                );
                let setup_sample = run_large_ivp_chunking_sample(setup, &baseline)
                    .unwrap_or_else(|err| panic!("{} {route} setup failed: {err}", matrix.label()));
                assert!(
                    setup_sample.3 <= 5.0e-4,
                    "{} {} setup drift too large: {:e}",
                    matrix.label(),
                    route,
                    setup_sample.3
                );
            }

            let route_configs = [
                (
                    "lambdify",
                    "UseIfAvailable",
                    large_ivp_lambdify_config(matrix, n),
                ),
                (
                    "tcc-whole",
                    "RequirePrebuilt",
                    large_ivp_tcc_config(
                        matrix,
                        n,
                        whole_dir,
                        SymbolicIvpAotBuildPolicy::RequirePrebuilt,
                        1,
                    ),
                ),
                (
                    "tcc-chunk",
                    "RequirePrebuilt",
                    large_ivp_tcc_config(
                        matrix,
                        n,
                        chunk_dir,
                        SymbolicIvpAotBuildPolicy::RequirePrebuilt,
                        target_chunks,
                    ),
                ),
            ];

            for (route, policy, config) in route_configs {
                let mut row = LargeIvpChunkingRow::new(matrix.label(), route, policy);
                for _ in 0..repeats {
                    row.runs_total += 1;
                    match run_large_ivp_chunking_sample(config.clone(), &baseline) {
                        Ok(sample) => push_large_ivp_sample(&mut row, sample),
                        Err(err) => row.record_failure(err),
                    }
                }
                rows.push(row);
            }
        }

        print_large_ivp_chunking_tables(
            &format!("AtomViewNative Lambdify vs tcc whole/chunk{target_chunks} warm prebuilt"),
            n,
            &rows,
        );

        assert!(
            rows.iter().all(|row| row.runs_ok == row.runs_total),
            "all large LSODE2 chunking rows must solve for n={n}"
        );
        assert!(
            rows.iter().all(|row| {
                row.final_linf
                    .summary()
                    .map(|(mean, _, _, _)| mean <= 5.0e-4)
                    .unwrap_or(false)
            }),
            "large LSODE2 chunking rows must remain numerically equivalent for n={n}"
        );
    }
}

pub(crate) fn run_lsode2_cold_aot_story_config_forces_rebuild_always() {
    for matrix in [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded] {
        for toolchain in [
            AotStoryToolchain::CTcc,
            AotStoryToolchain::CGcc,
            AotStoryToolchain::Zig,
            AotStoryToolchain::Rust,
        ] {
            let config = combustion_aot_matrix_config(
                matrix,
                toolchain,
                false,
                PathBuf::from("target/lsode2-cold-contract-test"),
            );
            assert_eq!(
                config.backend.generated_backend.build_policy,
                SymbolicIvpAotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                },
                "{} {} cold story rows must force a new build rather than reuse a problem-keyed backend",
                matrix.label(),
                toolchain.label()
            );
        }
    }
}

pub(crate) fn run_lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix() {
    const DEFAULT_REPEATS: usize = 3;
    let repeats = std::env::var("LSODE2_AOT_COLD_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_REPEATS);
    let row_filter = std::env::var("LSODE2_AOT_COLD_FILTER")
        .ok()
        .map(|value| value.to_ascii_lowercase());
    let allow_aot_failures = std::env::var("LSODE2_AOT_COLD_ALLOW_FAILURES")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let variants = [
        (AotStoryToolchain::CTcc, false, "tcc/whole"),
        (AotStoryToolchain::CTcc, true, "tcc/parallel"),
        (AotStoryToolchain::CGcc, false, "gcc/whole"),
        (AotStoryToolchain::CGcc, true, "gcc/parallel"),
        (AotStoryToolchain::Zig, false, "zig/whole"),
        (AotStoryToolchain::Zig, true, "zig/parallel"),
        (AotStoryToolchain::Rust, false, "rust/whole"),
        (AotStoryToolchain::Rust, true, "rust/parallel"),
    ];

    if let Some(filter) = &row_filter {
        println!(
            "[LSODE2 story] cold AOT matrix filter active: LSODE2_AOT_COLD_FILTER={filter:?}, repeats={repeats}"
        );
    }
    if allow_aot_failures {
        println!(
            "[LSODE2 story] cold AOT matrix will report AOT row failures without failing the test because LSODE2_AOT_COLD_ALLOW_FAILURES is set"
        );
    }

    let mut baselines = std::collections::BTreeMap::new();
    let mut rows = Vec::new();
    let mut lifecycle_rows = Vec::new();
    for matrix in matrices {
        let reference_config = combustion_symbolic_matrix_config(
            matrix,
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        )
        .with_bdf_only_controller();
        let mut reference_solver =
            Lsode2Solver::new(reference_config).expect("Lambdify reference should build");
        let reference = reference_solver
            .solve_with_summary()
            .expect("Lambdify reference should solve");
        let baseline_final_a = reference
            .final_y
            .expect("reference final state should exist")[0];

        let mut baseline_row = BackendRaceRow::new(matrix.label(), "Lambdify-AtomViewNative");
        for _ in 0..repeats {
            baseline_row.runs_total += 1;
            let config = combustion_symbolic_matrix_config(
                matrix,
                Lsode2SymbolicAssemblyBackend::AtomView,
                Lsode2SymbolicExecutionMode::LambdifyExpr,
                None,
            )
            .with_bdf_only_controller();
            match run_combustion_story_sample_result(
                "Lambdify-AtomViewNative",
                config,
                baseline_final_a,
            ) {
                Ok(sample) => push_combustion_sample(&mut baseline_row, sample),
                Err(err) => baseline_row.record_failure(err),
            }
        }
        baselines.insert(matrix.label(), baseline_final_a);
        rows.push(baseline_row);
    }

    let mut matched_aot_rows = 0usize;
    for matrix in matrices {
        let baseline = *baselines
            .get(matrix.label())
            .expect("baseline should exist");
        for (toolchain, parallel, label) in variants {
            if let Some(filter) = &row_filter {
                let row_id = format!("{} {label} {}", matrix.label(), toolchain.label())
                    .to_ascii_lowercase();
                if !row_id.contains(filter) {
                    continue;
                }
            }
            matched_aot_rows += 1;
            let mut row = BackendRaceRow::new(matrix.label(), label);
            for repeat in 0..repeats {
                row.runs_total += 1;
                let run_tag = unique_story_short_tag();
                let output_dir = PathBuf::from(format!(
                    "target/l2-aot-cold-matrix/{}/{}/{}/{}",
                    matrix.label().to_lowercase(),
                    label.replace('/', "_"),
                    repeat,
                    run_tag
                ));
                let config =
                    combustion_aot_matrix_config(matrix, toolchain, parallel, output_dir.clone());
                match run_combustion_story_sample_result(label, config, baseline) {
                    Ok(sample) => {
                        push_combustion_sample(&mut row, sample);
                        lifecycle_rows.push((
                            matrix.label(),
                            label,
                            repeat + 1,
                            "rebuild_always",
                            output_dir.exists(),
                            "ok".to_string(),
                        ));
                    }
                    Err(err) => {
                        let status = format!("failed: {}", short_error(&err));
                        row.record_failure(&err);
                        lifecycle_rows.push((
                            matrix.label(),
                            label,
                            repeat + 1,
                            "rebuild_always",
                            output_dir.exists(),
                            status,
                        ));
                    }
                }
            }
            rows.push(row);
        }
    }
    if row_filter.is_some() {
        assert!(
            matched_aot_rows > 0,
            "LSODE2_AOT_COLD_FILTER did not match any AOT matrix row"
        );
    }

    print_compact_combustion_story_tables(
        "combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix",
        &rows,
    );
    println!(
        "note: AOT total_ms includes symbolic preparation, artifact build/link and native integration; hot-stage timers isolate repeated callback/linear work."
    );
    println!(
        "[LSODE2 story] cold AOT lifecycle observations; successful AOT rows require a fresh materialization directory"
    );
    println!(
        "matrix | route                    | rep | cold_action    | artifact_dir_written | status"
    );
    println!(
        "---------------------------------------------------------------------------------------------"
    );
    for (matrix, route, repeat, action, artifact_written, status) in &lifecycle_rows {
        println!(
            "{:<6} | {:<24} | {:>3} | {:<14} | {:<20} | {}",
            matrix, route, repeat, action, artifact_written, status
        );
        assert!(
            status != "ok" || *artifact_written,
            "{} {} repetition {} completed without writing its isolated cold artifact directory",
            matrix,
            route,
            repeat
        );
    }

    for row in rows {
        if row.route == "Lambdify-AtomViewNative" {
            assert_eq!(
                row.runs_ok, row.runs_total,
                "{} baseline should complete all runs",
                row.matrix
            );
        } else if !allow_aot_failures {
            assert_eq!(
                row.runs_ok, row.runs_total,
                "{} {} cold AOT matrix row should complete all runs; set LSODE2_AOT_COLD_ALLOW_FAILURES=1 only for exploratory runs on machines without this toolchain",
                row.matrix, row.route
            );
        }
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row.final_diff.summary().expect("successful row has diffs");
            assert!(
                mean_diff <= 2.0e-4,
                "{} {} AOT matrix drift too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
        }
    }
}
