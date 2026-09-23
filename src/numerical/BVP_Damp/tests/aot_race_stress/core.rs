#[derive(Clone)]
struct RaceVariant {
    source: &'static str,
    matrix: &'static str,
    variant: &'static str,
    bootstrap_hint: &'static str,
    config: GeneratedBackendConfig,
}

#[derive(Clone, Debug)]
struct RaceRow {
    source: &'static str,
    matrix: &'static str,
    variant: &'static str,
    bootstrap_hint: &'static str,
    total_ms: f64,
    max_abs_solution: f64,
    solve_diff: f64,
    rel_x_diff: f64,
    iterations: usize,
    linear_solves: usize,
    jac_rebuilds: usize,
    grid_refinements: usize,
    final_grid_points: usize,
    total_timer_ms: f64,
    symbolic_timer_ms: f64,
    linear_timer_ms: f64,
    jac_timer_ms: f64,
    fun_timer_ms: f64,
    cb_residual_values_ms: f64,
    cb_jacobian_values_ms: f64,
    cb_jacobian_assembly_ms: f64,
    residual_actual_jobs: f64,
    sparse_jacobian_actual_jobs: f64,
    residual_work_per_job: f64,
    sparse_jacobian_work_per_job: f64,
    residual_fallback_reason: String,
    sparse_jacobian_fallback_reason: String,
    selected_backend: String,
    symbolic_assembly_backend: String,
    aot_build_policy: String,
    initial_generate_ms: f64,
    initial_discretization_ms: f64,
    initial_symbolic_jacobian_ms: f64,
    initial_symbolic_variable_sets_ms: f64,
    initial_symbolic_row_differentiation_ms: f64,
    initial_symbolic_dense_cache_ms: f64,
    initial_symbolic_sparse_flatten_ms: f64,
    initial_sparse_prepare_ms: f64,
    initial_runtime_binding_ms: f64,
    initial_lambdify_jacobian_compile_ms: f64,
    initial_lambdify_residual_compile_ms: f64,
    post_build_generate_ms: f64,
    post_build_discretization_ms: f64,
    post_build_symbolic_jacobian_ms: f64,
    post_build_sparse_prepare_ms: f64,
    post_build_runtime_binding_ms: f64,
    post_build_rebind_ms: f64,
    aot_artifact_ms: f64,
    aot_module_ms: f64,
    aot_residual_lower_ms: f64,
    aot_jacobian_lower_ms: f64,
    aot_source_emit_ms: f64,
    aot_packaging_ms: f64,
    aot_materialize_ms: f64,
    aot_compile_link_ms: f64,
    aot_register_link_ms: f64,
    status: String,
}

#[derive(Clone, Copy, Debug)]
struct Aggregate {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

#[derive(Clone, Debug)]
struct RaceSummaryRow {
    source: &'static str,
    matrix: &'static str,
    variant: &'static str,
    bootstrap_hint: &'static str,
    runs: usize,
    ok_runs: usize,
    total_ms: Aggregate,
    max_abs_solution: Aggregate,
    solve_diff: Aggregate,
    rel_x_diff: Aggregate,
    iterations: Aggregate,
    linear_solves: Aggregate,
    jac_rebuilds: Aggregate,
    grid_refinements: Aggregate,
    final_grid_points: Aggregate,
    total_timer_ms: Aggregate,
    symbolic_timer_ms: Aggregate,
    linear_timer_ms: Aggregate,
    jac_timer_ms: Aggregate,
    fun_timer_ms: Aggregate,
    cb_residual_values_ms: Aggregate,
    cb_jacobian_values_ms: Aggregate,
    cb_jacobian_assembly_ms: Aggregate,
    residual_actual_jobs: Aggregate,
    sparse_jacobian_actual_jobs: Aggregate,
    residual_work_per_job: Aggregate,
    sparse_jacobian_work_per_job: Aggregate,
    residual_fallback_reason: String,
    sparse_jacobian_fallback_reason: String,
    selected_backend: String,
    symbolic_assembly_backend: String,
    aot_build_policy: String,
    initial_generate_ms: Aggregate,
    initial_discretization_ms: Aggregate,
    initial_symbolic_jacobian_ms: Aggregate,
    initial_symbolic_variable_sets_ms: Aggregate,
    initial_symbolic_row_differentiation_ms: Aggregate,
    initial_symbolic_dense_cache_ms: Aggregate,
    initial_symbolic_sparse_flatten_ms: Aggregate,
    initial_sparse_prepare_ms: Aggregate,
    initial_runtime_binding_ms: Aggregate,
    initial_lambdify_jacobian_compile_ms: Aggregate,
    initial_lambdify_residual_compile_ms: Aggregate,
    post_build_generate_ms: Aggregate,
    post_build_discretization_ms: Aggregate,
    post_build_symbolic_jacobian_ms: Aggregate,
    post_build_sparse_prepare_ms: Aggregate,
    post_build_runtime_binding_ms: Aggregate,
    post_build_rebind_ms: Aggregate,
    aot_artifact_ms: Aggregate,
    aot_module_ms: Aggregate,
    aot_residual_lower_ms: Aggregate,
    aot_jacobian_lower_ms: Aggregate,
    aot_source_emit_ms: Aggregate,
    aot_packaging_ms: Aggregate,
    aot_materialize_ms: Aggregate,
    aot_compile_link_ms: Aggregate,
    aot_register_link_ms: Aggregate,
    status: String,
}

fn make_combustion_solver(
    n_steps: usize,
    generated_backend_config: GeneratedBackendConfig,
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

fn stats_count(stats: &DampedBvpStatistics, key: &str) -> usize {
    stats.counters.get(key).copied().unwrap_or(0)
}

fn stats_timer_ms(stats: &DampedBvpStatistics, prefix: &str) -> f64 {
    stats
        .timers
        .iter()
        .find(|(key, _)| key.starts_with(prefix))
        .and_then(|(key, value)| timer_value_to_ms(key, value))
        .unwrap_or(f64::NAN)
}

fn callback_residual_values_ms(stats: &DampedBvpStatistics) -> f64 {
    stats_timer_ms(stats, "Callback Residual Values")
}

fn callback_jacobian_values_ms(stats: &DampedBvpStatistics) -> f64 {
    stats_timer_ms(stats, "Callback Jacobian Values")
}

fn callback_jacobian_assembly_ms(stats: &DampedBvpStatistics) -> f64 {
    stats_timer_ms(stats, "Callback Jacobian Matrix Assembly")
}

fn stats_diagnostic_usize(stats: &DampedBvpStatistics, key: &str) -> f64 {
    stats
        .diagnostics
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value as f64)
        .unwrap_or(f64::NAN)
}

fn stats_diagnostic_ms(stats: &DampedBvpStatistics, key: &str) -> f64 {
    stats
        .diagnostics
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn stats_diagnostic_string(stats: &DampedBvpStatistics, key: &str) -> String {
    stats
        .diagnostics
        .get(key)
        .cloned()
        .unwrap_or_else(|| "-".to_string())
}

fn row_runtime_diagnostics(
    statistics: &DampedBvpStatistics,
) -> (f64, f64, f64, f64, String, String) {
    (
        stats_diagnostic_usize(statistics, "aot.runtime.residual.actual_jobs"),
        stats_diagnostic_usize(statistics, "aot.runtime.sparse_jacobian.actual_jobs"),
        stats_diagnostic_usize(statistics, "aot.runtime.residual.work_per_job"),
        stats_diagnostic_usize(statistics, "aot.runtime.sparse_jacobian.work_per_job"),
        stats_diagnostic_string(statistics, "aot.runtime.residual.fallback_reason"),
        stats_diagnostic_string(statistics, "aot.runtime.sparse_jacobian.fallback_reason"),
    )
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

fn run_race_variant(
    n_steps: usize,
    source: &'static str,
    matrix: &'static str,
    variant: &'static str,
    bootstrap_hint: &'static str,
    config: GeneratedBackendConfig,
) -> (RaceRow, Option<DMatrix<f64>>) {
    let total_begin = Instant::now();
    let mut solver = make_combustion_solver(n_steps, config);
    let solve_status = catch_unwind(AssertUnwindSafe(|| solver.try_solver()));
    let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
    let statistics = solver.get_statistics();
    let (
        residual_actual_jobs,
        sparse_jacobian_actual_jobs,
        residual_work_per_job,
        sparse_jacobian_work_per_job,
        residual_fallback_reason,
        sparse_jacobian_fallback_reason,
    ) = row_runtime_diagnostics(&statistics);

    match solve_status {
        Ok(Ok(_)) => match solver.get_result() {
            Some(solution) => {
                let max_abs_solution = solution_max_abs(&solution);
                (
                    RaceRow {
                        source,
                        matrix,
                        variant,
                        bootstrap_hint,
                        total_ms,
                        max_abs_solution,
                        solve_diff: 0.0,
                        rel_x_diff: 0.0,
                        iterations: stats_count(&statistics, "number of iterations"),
                        linear_solves: stats_count(&statistics, "number of solving linear systems"),
                        jac_rebuilds: stats_count(
                            &statistics,
                            "number of jacobians recalculations",
                        ),
                        grid_refinements: stats_count(&statistics, "number of grid refinements"),
                        final_grid_points: stats_count(&statistics, "number of grid points"),
                        total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                        symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                        linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                        jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                        fun_timer_ms: stats_timer_ms(&statistics, "Function"),
                        cb_residual_values_ms: callback_residual_values_ms(&statistics),
                        cb_jacobian_values_ms: callback_jacobian_values_ms(&statistics),
                        cb_jacobian_assembly_ms: callback_jacobian_assembly_ms(&statistics),
                        residual_actual_jobs,
                        sparse_jacobian_actual_jobs,
                        residual_work_per_job,
                        sparse_jacobian_work_per_job,
                        residual_fallback_reason,
                        sparse_jacobian_fallback_reason,
                        selected_backend: stats_diagnostic_string(
                            &statistics,
                            "generated.selected_backend",
                        ),
                        symbolic_assembly_backend: stats_diagnostic_string(
                            &statistics,
                            "generated.symbolic_assembly_backend",
                        ),
                        aot_build_policy: stats_diagnostic_string(&statistics, "aot.build_policy"),
                        initial_generate_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial_generate_wall_ms",
                        ),
                        initial_discretization_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.discretization_time_ms",
                        ),
                        initial_symbolic_jacobian_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.symbolic_jacobian_time_ms",
                        ),
                        initial_symbolic_variable_sets_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.symbolic_jacobian_variable_sets_time_ms",
                        ),
                        initial_symbolic_row_differentiation_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.symbolic_jacobian_row_differentiation_time_ms",
                        ),
                        initial_symbolic_dense_cache_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.symbolic_jacobian_dense_cache_materialize_time_ms",
                        ),
                        initial_symbolic_sparse_flatten_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.symbolic_jacobian_sparse_cache_flatten_time_ms",
                        ),
                        initial_sparse_prepare_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.sparse_AOT_preparation_time_ms",
                        ),
                        initial_runtime_binding_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.runtime_binding_time_ms",
                        ),
                        initial_lambdify_jacobian_compile_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.lambdify_jacobian_callback_compile_time_ms",
                        ),
                        initial_lambdify_residual_compile_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.initial.lambdify_residual_callback_compile_time_ms",
                        ),
                        post_build_generate_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.post_build_regenerate_wall_ms",
                        ),
                        post_build_discretization_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.post_build.discretization_time_ms",
                        ),
                        post_build_symbolic_jacobian_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.post_build.symbolic_jacobian_time_ms",
                        ),
                        post_build_sparse_prepare_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.post_build.sparse_AOT_preparation_time_ms",
                        ),
                        post_build_runtime_binding_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.post_build.runtime_binding_time_ms",
                        ),
                        post_build_rebind_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.handoff.post_build_rebind_wall_ms",
                        ),
                        aot_artifact_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.artifact_wall_ms",
                        ),
                        aot_module_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.artifact.module_ms",
                        ),
                        aot_residual_lower_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.artifact.residual_lower_ms",
                        ),
                        aot_jacobian_lower_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.artifact.jacobian_lower_ms",
                        ),
                        aot_source_emit_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.artifact.source_emit_ms",
                        ),
                        aot_packaging_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.artifact.packaging_ms",
                        ),
                        aot_materialize_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.materialize_ms",
                        ),
                        aot_compile_link_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.compile_link_ms",
                        ),
                        aot_register_link_ms: stats_diagnostic_ms(
                            &statistics,
                            "generated.aot.register_link_ms",
                        ),
                        status: "ok".to_string(),
                    },
                    Some(solution),
                )
            }
            None => (
                RaceRow {
                    source,
                    matrix,
                    variant,
                    bootstrap_hint,
                    total_ms,
                    max_abs_solution: f64::NAN,
                    solve_diff: f64::NAN,
                    rel_x_diff: f64::NAN,
                    iterations: stats_count(&statistics, "number of iterations"),
                    linear_solves: stats_count(&statistics, "number of solving linear systems"),
                    jac_rebuilds: stats_count(&statistics, "number of jacobians recalculations"),
                    grid_refinements: stats_count(&statistics, "number of grid refinements"),
                    final_grid_points: stats_count(&statistics, "number of grid points"),
                    total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                    symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                    linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                    jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                    fun_timer_ms: stats_timer_ms(&statistics, "Function"),
                    cb_residual_values_ms: callback_residual_values_ms(&statistics),
                    cb_jacobian_values_ms: callback_jacobian_values_ms(&statistics),
                    cb_jacobian_assembly_ms: callback_jacobian_assembly_ms(&statistics),
                    residual_actual_jobs,
                    sparse_jacobian_actual_jobs,
                    residual_work_per_job,
                    sparse_jacobian_work_per_job,
                    residual_fallback_reason,
                    sparse_jacobian_fallback_reason,
                    selected_backend: stats_diagnostic_string(
                        &statistics,
                        "generated.selected_backend",
                    ),
                    symbolic_assembly_backend: stats_diagnostic_string(
                        &statistics,
                        "generated.symbolic_assembly_backend",
                    ),
                    aot_build_policy: stats_diagnostic_string(&statistics, "aot.build_policy"),
                    initial_generate_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial_generate_wall_ms",
                    ),
                    initial_discretization_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.discretization_time_ms",
                    ),
                    initial_symbolic_jacobian_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_time_ms",
                    ),
                    initial_symbolic_variable_sets_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_variable_sets_time_ms",
                    ),
                    initial_symbolic_row_differentiation_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_row_differentiation_time_ms",
                    ),
                    initial_symbolic_dense_cache_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_dense_cache_materialize_time_ms",
                    ),
                    initial_symbolic_sparse_flatten_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_sparse_cache_flatten_time_ms",
                    ),
                    initial_sparse_prepare_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.sparse_AOT_preparation_time_ms",
                    ),
                    initial_runtime_binding_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.runtime_binding_time_ms",
                    ),
                    initial_lambdify_jacobian_compile_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.lambdify_jacobian_callback_compile_time_ms",
                    ),
                    initial_lambdify_residual_compile_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.lambdify_residual_callback_compile_time_ms",
                    ),
                    post_build_generate_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build_regenerate_wall_ms",
                    ),
                    post_build_discretization_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.discretization_time_ms",
                    ),
                    post_build_symbolic_jacobian_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.symbolic_jacobian_time_ms",
                    ),
                    post_build_sparse_prepare_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.sparse_AOT_preparation_time_ms",
                    ),
                    post_build_runtime_binding_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.runtime_binding_time_ms",
                    ),
                    post_build_rebind_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build_rebind_wall_ms",
                    ),
                    aot_artifact_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact_wall_ms",
                    ),
                    aot_module_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.module_ms",
                    ),
                    aot_residual_lower_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.residual_lower_ms",
                    ),
                    aot_jacobian_lower_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.jacobian_lower_ms",
                    ),
                    aot_source_emit_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.source_emit_ms",
                    ),
                    aot_packaging_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.packaging_ms",
                    ),
                    aot_materialize_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.materialize_ms",
                    ),
                    aot_compile_link_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.compile_link_ms",
                    ),
                    aot_register_link_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.register_link_ms",
                    ),
                    status: "no_result".to_string(),
                },
                None,
            ),
        },
        Ok(Err(err)) => (
            RaceRow {
                source,
                matrix,
                variant,
                bootstrap_hint,
                total_ms,
                max_abs_solution: f64::NAN,
                solve_diff: f64::NAN,
                rel_x_diff: f64::NAN,
                iterations: stats_count(&statistics, "number of iterations"),
                linear_solves: stats_count(&statistics, "number of solving linear systems"),
                jac_rebuilds: stats_count(&statistics, "number of jacobians recalculations"),
                grid_refinements: stats_count(&statistics, "number of grid refinements"),
                final_grid_points: stats_count(&statistics, "number of grid points"),
                total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                fun_timer_ms: stats_timer_ms(&statistics, "Function"),
                cb_residual_values_ms: callback_residual_values_ms(&statistics),
                cb_jacobian_values_ms: callback_jacobian_values_ms(&statistics),
                cb_jacobian_assembly_ms: callback_jacobian_assembly_ms(&statistics),
                residual_actual_jobs,
                sparse_jacobian_actual_jobs,
                residual_work_per_job,
                sparse_jacobian_work_per_job,
                residual_fallback_reason,
                sparse_jacobian_fallback_reason,
                selected_backend: stats_diagnostic_string(
                    &statistics,
                    "generated.selected_backend",
                ),
                symbolic_assembly_backend: stats_diagnostic_string(
                    &statistics,
                    "generated.symbolic_assembly_backend",
                ),
                aot_build_policy: stats_diagnostic_string(&statistics, "aot.build_policy"),
                initial_generate_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial_generate_wall_ms",
                ),
                initial_discretization_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.discretization_time_ms",
                ),
                initial_symbolic_jacobian_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.symbolic_jacobian_time_ms",
                ),
                initial_symbolic_variable_sets_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.symbolic_jacobian_variable_sets_time_ms",
                ),
                initial_symbolic_row_differentiation_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.symbolic_jacobian_row_differentiation_time_ms",
                ),
                initial_symbolic_dense_cache_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.symbolic_jacobian_dense_cache_materialize_time_ms",
                ),
                initial_symbolic_sparse_flatten_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.symbolic_jacobian_sparse_cache_flatten_time_ms",
                ),
                initial_sparse_prepare_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.sparse_AOT_preparation_time_ms",
                ),
                initial_runtime_binding_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.runtime_binding_time_ms",
                ),
                initial_lambdify_jacobian_compile_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.lambdify_jacobian_callback_compile_time_ms",
                ),
                initial_lambdify_residual_compile_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.initial.lambdify_residual_callback_compile_time_ms",
                ),
                post_build_generate_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.post_build_regenerate_wall_ms",
                ),
                post_build_discretization_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.post_build.discretization_time_ms",
                ),
                post_build_symbolic_jacobian_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.post_build.symbolic_jacobian_time_ms",
                ),
                post_build_sparse_prepare_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.post_build.sparse_AOT_preparation_time_ms",
                ),
                post_build_runtime_binding_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.post_build.runtime_binding_time_ms",
                ),
                post_build_rebind_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.handoff.post_build_rebind_wall_ms",
                ),
                aot_artifact_ms: stats_diagnostic_ms(&statistics, "generated.aot.artifact_wall_ms"),
                aot_module_ms: stats_diagnostic_ms(&statistics, "generated.aot.artifact.module_ms"),
                aot_residual_lower_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.aot.artifact.residual_lower_ms",
                ),
                aot_jacobian_lower_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.aot.artifact.jacobian_lower_ms",
                ),
                aot_source_emit_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.aot.artifact.source_emit_ms",
                ),
                aot_packaging_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.aot.artifact.packaging_ms",
                ),
                aot_materialize_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.aot.materialize_ms",
                ),
                aot_compile_link_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.aot.compile_link_ms",
                ),
                aot_register_link_ms: stats_diagnostic_ms(
                    &statistics,
                    "generated.aot.register_link_ms",
                ),
                status: format!("solve_failed({err:?})"),
            },
            None,
        ),
        Err(panic_payload) => {
            let status = if let Some(message) = panic_payload.downcast_ref::<String>() {
                format!("solve_panicked({message})")
            } else if let Some(message) = panic_payload.downcast_ref::<&str>() {
                format!("solve_panicked({message})")
            } else {
                "solve_panicked(non-string payload)".to_string()
            };
            (
                RaceRow {
                    source,
                    matrix,
                    variant,
                    bootstrap_hint,
                    total_ms,
                    max_abs_solution: f64::NAN,
                    solve_diff: f64::NAN,
                    rel_x_diff: f64::NAN,
                    iterations: stats_count(&statistics, "number of iterations"),
                    linear_solves: stats_count(&statistics, "number of solving linear systems"),
                    jac_rebuilds: stats_count(&statistics, "number of jacobians recalculations"),
                    grid_refinements: stats_count(&statistics, "number of grid refinements"),
                    final_grid_points: stats_count(&statistics, "number of grid points"),
                    total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                    symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                    linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                    jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                    fun_timer_ms: stats_timer_ms(&statistics, "Function"),
                    cb_residual_values_ms: callback_residual_values_ms(&statistics),
                    cb_jacobian_values_ms: callback_jacobian_values_ms(&statistics),
                    cb_jacobian_assembly_ms: callback_jacobian_assembly_ms(&statistics),
                    residual_actual_jobs,
                    sparse_jacobian_actual_jobs,
                    residual_work_per_job,
                    sparse_jacobian_work_per_job,
                    residual_fallback_reason,
                    sparse_jacobian_fallback_reason,
                    selected_backend: stats_diagnostic_string(
                        &statistics,
                        "generated.selected_backend",
                    ),
                    symbolic_assembly_backend: stats_diagnostic_string(
                        &statistics,
                        "generated.symbolic_assembly_backend",
                    ),
                    aot_build_policy: stats_diagnostic_string(&statistics, "aot.build_policy"),
                    initial_generate_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial_generate_wall_ms",
                    ),
                    initial_discretization_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.discretization_time_ms",
                    ),
                    initial_symbolic_jacobian_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_time_ms",
                    ),
                    initial_symbolic_variable_sets_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_variable_sets_time_ms",
                    ),
                    initial_symbolic_row_differentiation_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_row_differentiation_time_ms",
                    ),
                    initial_symbolic_dense_cache_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_dense_cache_materialize_time_ms",
                    ),
                    initial_symbolic_sparse_flatten_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.symbolic_jacobian_sparse_cache_flatten_time_ms",
                    ),
                    initial_sparse_prepare_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.sparse_AOT_preparation_time_ms",
                    ),
                    initial_runtime_binding_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.runtime_binding_time_ms",
                    ),
                    initial_lambdify_jacobian_compile_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.lambdify_jacobian_callback_compile_time_ms",
                    ),
                    initial_lambdify_residual_compile_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.initial.lambdify_residual_callback_compile_time_ms",
                    ),
                    post_build_generate_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build_regenerate_wall_ms",
                    ),
                    post_build_discretization_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.discretization_time_ms",
                    ),
                    post_build_symbolic_jacobian_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.symbolic_jacobian_time_ms",
                    ),
                    post_build_sparse_prepare_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.sparse_AOT_preparation_time_ms",
                    ),
                    post_build_runtime_binding_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build.runtime_binding_time_ms",
                    ),
                    post_build_rebind_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.handoff.post_build_rebind_wall_ms",
                    ),
                    aot_artifact_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact_wall_ms",
                    ),
                    aot_module_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.module_ms",
                    ),
                    aot_residual_lower_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.residual_lower_ms",
                    ),
                    aot_jacobian_lower_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.jacobian_lower_ms",
                    ),
                    aot_source_emit_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.source_emit_ms",
                    ),
                    aot_packaging_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.artifact.packaging_ms",
                    ),
                    aot_materialize_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.materialize_ms",
                    ),
                    aot_compile_link_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.compile_link_ms",
                    ),
                    aot_register_link_ms: stats_diagnostic_ms(
                        &statistics,
                        "generated.aot.register_link_ms",
                    ),
                    status,
                },
                None,
            )
        }
    }
}

fn run_race_samples(variants: &[RaceVariant], n_steps: usize, repetitions: usize) -> Vec<RaceRow> {
    let mut samples = Vec::with_capacity(variants.len() * repetitions);
    for repetition in 0..repetitions {
        println!(
            "[BVP Damp race] starting repetition {}/{}",
            repetition + 1,
            repetitions
        );
        let mut rows = Vec::with_capacity(variants.len());
        let mut solutions = Vec::with_capacity(variants.len());
        for variant in variants {
            println!(
                "[BVP Damp race] running source={} matrix={} variant={} bootstrap_hint={}",
                variant.source, variant.matrix, variant.variant, variant.bootstrap_hint
            );
            let _ = io::stdout().flush();
            let (row, solution) = run_race_variant(
                n_steps,
                variant.source,
                variant.matrix,
                variant.variant,
                variant.bootstrap_hint,
                variant.config.clone(),
            );
            println!(
                "[BVP Damp race] finished source={} matrix={} variant={} status={}",
                row.source, row.matrix, row.variant, row.status
            );
            let _ = io::stdout().flush();
            rows.push(row);
            solutions.push(solution);
        }
        fill_solution_diffs(&mut rows, &solutions);
        samples.extend(rows);
    }
    samples
}

fn fill_solution_diffs(rows: &mut [RaceRow], solutions: &[Option<DMatrix<f64>>]) {
    let baseline = solutions.iter().find_map(|solution| solution.as_ref());
    if let Some(baseline) = baseline {
        for (row, solution) in rows.iter_mut().zip(solutions.iter()) {
            if let Some(solution) = solution {
                row.solve_diff = solution_linf_diff(solution, baseline);
                row.rel_x_diff = solution_rel_diff(solution, baseline);
            }
        }
    }
}
