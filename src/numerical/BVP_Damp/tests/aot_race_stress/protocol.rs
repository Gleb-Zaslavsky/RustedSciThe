fn encode_isolated_race_row(row: &RaceRow) -> String {
    [
        row.total_ms.to_string(),
        row.max_abs_solution.to_string(),
        row.iterations.to_string(),
        row.linear_solves.to_string(),
        row.jac_rebuilds.to_string(),
        row.grid_refinements.to_string(),
        row.final_grid_points.to_string(),
        row.total_timer_ms.to_string(),
        row.symbolic_timer_ms.to_string(),
        row.linear_timer_ms.to_string(),
        row.jac_timer_ms.to_string(),
        row.fun_timer_ms.to_string(),
        row.cb_residual_values_ms.to_string(),
        row.cb_jacobian_values_ms.to_string(),
        row.cb_jacobian_assembly_ms.to_string(),
        row.residual_actual_jobs.to_string(),
        row.sparse_jacobian_actual_jobs.to_string(),
        row.residual_work_per_job.to_string(),
        row.sparse_jacobian_work_per_job.to_string(),
        row.residual_fallback_reason.clone(),
        row.sparse_jacobian_fallback_reason.clone(),
        row.selected_backend.clone(),
        row.symbolic_assembly_backend.clone(),
        row.aot_build_policy.clone(),
        row.initial_generate_ms.to_string(),
        row.initial_discretization_ms.to_string(),
        row.initial_symbolic_jacobian_ms.to_string(),
        row.initial_symbolic_variable_sets_ms.to_string(),
        row.initial_symbolic_row_differentiation_ms.to_string(),
        row.initial_symbolic_dense_cache_ms.to_string(),
        row.initial_symbolic_sparse_flatten_ms.to_string(),
        row.initial_sparse_prepare_ms.to_string(),
        row.initial_runtime_binding_ms.to_string(),
        row.initial_lambdify_jacobian_compile_ms.to_string(),
        row.initial_lambdify_residual_compile_ms.to_string(),
        row.post_build_generate_ms.to_string(),
        row.post_build_discretization_ms.to_string(),
        row.post_build_symbolic_jacobian_ms.to_string(),
        row.post_build_sparse_prepare_ms.to_string(),
        row.post_build_runtime_binding_ms.to_string(),
        row.post_build_rebind_ms.to_string(),
        row.aot_artifact_ms.to_string(),
        row.aot_module_ms.to_string(),
        row.aot_residual_lower_ms.to_string(),
        row.aot_jacobian_lower_ms.to_string(),
        row.aot_source_emit_ms.to_string(),
        row.aot_packaging_ms.to_string(),
        row.aot_materialize_ms.to_string(),
        row.aot_compile_link_ms.to_string(),
        row.aot_register_link_ms.to_string(),
        row.status.clone(),
    ]
    .join("\t")
}

fn parse_isolated_field<T: std::str::FromStr>(
    fields: &mut impl Iterator<Item = String>,
    name: &str,
) -> T
where
    T::Err: std::fmt::Debug,
{
    fields
        .next()
        .unwrap_or_else(|| panic!("isolated race row missing field {name}"))
        .parse::<T>()
        .unwrap_or_else(|err| panic!("isolated race row field {name} could not parse: {err:?}"))
}

fn parse_isolated_string(fields: &mut impl Iterator<Item = String>, name: &str) -> String {
    fields
        .next()
        .unwrap_or_else(|| panic!("isolated race row missing field {name}"))
}

fn decode_isolated_race_row(line: &str, variant: &RaceVariant) -> RaceRow {
    let payload = line
        .strip_prefix(ISOLATED_RACE_ROW_MARKER)
        .expect("isolated race row marker should be present")
        .trim_start_matches('\t');
    let mut fields = payload.split('\t').map(str::to_string);
    let row = RaceRow {
        source: variant.source,
        matrix: variant.matrix,
        variant: variant.variant,
        bootstrap_hint: variant.bootstrap_hint,
        total_ms: parse_isolated_field(&mut fields, "total_ms"),
        max_abs_solution: parse_isolated_field(&mut fields, "max_abs_solution"),
        solve_diff: 0.0,
        rel_x_diff: 0.0,
        iterations: parse_isolated_field(&mut fields, "iterations"),
        linear_solves: parse_isolated_field(&mut fields, "linear_solves"),
        jac_rebuilds: parse_isolated_field(&mut fields, "jac_rebuilds"),
        grid_refinements: parse_isolated_field(&mut fields, "grid_refinements"),
        final_grid_points: parse_isolated_field(&mut fields, "final_grid_points"),
        total_timer_ms: parse_isolated_field(&mut fields, "total_timer_ms"),
        symbolic_timer_ms: parse_isolated_field(&mut fields, "symbolic_timer_ms"),
        linear_timer_ms: parse_isolated_field(&mut fields, "linear_timer_ms"),
        jac_timer_ms: parse_isolated_field(&mut fields, "jac_timer_ms"),
        fun_timer_ms: parse_isolated_field(&mut fields, "fun_timer_ms"),
        cb_residual_values_ms: parse_isolated_field(&mut fields, "cb_residual_values_ms"),
        cb_jacobian_values_ms: parse_isolated_field(&mut fields, "cb_jacobian_values_ms"),
        cb_jacobian_assembly_ms: parse_isolated_field(&mut fields, "cb_jacobian_assembly_ms"),
        residual_actual_jobs: parse_isolated_field(&mut fields, "residual_actual_jobs"),
        sparse_jacobian_actual_jobs: parse_isolated_field(
            &mut fields,
            "sparse_jacobian_actual_jobs",
        ),
        residual_work_per_job: parse_isolated_field(&mut fields, "residual_work_per_job"),
        sparse_jacobian_work_per_job: parse_isolated_field(
            &mut fields,
            "sparse_jacobian_work_per_job",
        ),
        residual_fallback_reason: parse_isolated_string(&mut fields, "residual_fallback_reason"),
        sparse_jacobian_fallback_reason: parse_isolated_string(
            &mut fields,
            "sparse_jacobian_fallback_reason",
        ),
        selected_backend: parse_isolated_string(&mut fields, "selected_backend"),
        symbolic_assembly_backend: parse_isolated_string(&mut fields, "symbolic_assembly_backend"),
        aot_build_policy: parse_isolated_string(&mut fields, "aot_build_policy"),
        initial_generate_ms: parse_isolated_field(&mut fields, "initial_generate_ms"),
        initial_discretization_ms: parse_isolated_field(&mut fields, "initial_discretization_ms"),
        initial_symbolic_jacobian_ms: parse_isolated_field(
            &mut fields,
            "initial_symbolic_jacobian_ms",
        ),
        initial_symbolic_variable_sets_ms: parse_isolated_field(
            &mut fields,
            "initial_symbolic_variable_sets_ms",
        ),
        initial_symbolic_row_differentiation_ms: parse_isolated_field(
            &mut fields,
            "initial_symbolic_row_differentiation_ms",
        ),
        initial_symbolic_dense_cache_ms: parse_isolated_field(
            &mut fields,
            "initial_symbolic_dense_cache_ms",
        ),
        initial_symbolic_sparse_flatten_ms: parse_isolated_field(
            &mut fields,
            "initial_symbolic_sparse_flatten_ms",
        ),
        initial_sparse_prepare_ms: parse_isolated_field(&mut fields, "initial_sparse_prepare_ms"),
        initial_runtime_binding_ms: parse_isolated_field(&mut fields, "initial_runtime_binding_ms"),
        initial_lambdify_jacobian_compile_ms: parse_isolated_field(
            &mut fields,
            "initial_lambdify_jacobian_compile_ms",
        ),
        initial_lambdify_residual_compile_ms: parse_isolated_field(
            &mut fields,
            "initial_lambdify_residual_compile_ms",
        ),
        post_build_generate_ms: parse_isolated_field(&mut fields, "post_build_generate_ms"),
        post_build_discretization_ms: parse_isolated_field(
            &mut fields,
            "post_build_discretization_ms",
        ),
        post_build_symbolic_jacobian_ms: parse_isolated_field(
            &mut fields,
            "post_build_symbolic_jacobian_ms",
        ),
        post_build_sparse_prepare_ms: parse_isolated_field(
            &mut fields,
            "post_build_sparse_prepare_ms",
        ),
        post_build_runtime_binding_ms: parse_isolated_field(
            &mut fields,
            "post_build_runtime_binding_ms",
        ),
        post_build_rebind_ms: parse_isolated_field(&mut fields, "post_build_rebind_ms"),
        aot_artifact_ms: parse_isolated_field(&mut fields, "aot_artifact_ms"),
        aot_module_ms: parse_isolated_field(&mut fields, "aot_module_ms"),
        aot_residual_lower_ms: parse_isolated_field(&mut fields, "aot_residual_lower_ms"),
        aot_jacobian_lower_ms: parse_isolated_field(&mut fields, "aot_jacobian_lower_ms"),
        aot_source_emit_ms: parse_isolated_field(&mut fields, "aot_source_emit_ms"),
        aot_packaging_ms: parse_isolated_field(&mut fields, "aot_packaging_ms"),
        aot_materialize_ms: parse_isolated_field(&mut fields, "aot_materialize_ms"),
        aot_compile_link_ms: parse_isolated_field(&mut fields, "aot_compile_link_ms"),
        aot_register_link_ms: parse_isolated_field(&mut fields, "aot_register_link_ms"),
        status: parse_isolated_string(&mut fields, "status"),
    };
    assert!(
        fields.next().is_none(),
        "isolated race row contains unexpected trailing fields"
    );
    row
}

fn encode_isolated_solution(solution: &DMatrix<f64>) -> String {
    solution
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("\t")
}

fn decode_isolated_solution(line: &str) -> DMatrix<f64> {
    let payload = line
        .strip_prefix(ISOLATED_RACE_SOLUTION_MARKER)
        .expect("isolated solution marker should be present")
        .trim_start_matches('\t');
    let values = payload
        .split('\t')
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|err| panic!("isolated solution value could not parse: {err}"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        values.len() % 6,
        0,
        "isolated combustion solution should contain six state rows"
    );
    DMatrix::from_column_slice(6, values.len() / 6, &values)
}

fn story_protocol(n_steps: usize, repetitions: usize) -> AotStoryProtocol {
    let protocol = AotStoryProtocol::from_env(n_steps, repetitions);
    protocol
        .validate()
        .unwrap_or_else(|error| panic!("invalid AOT story protocol: {error}"));
    protocol
}

fn cold_cooldown_ms() -> u64 {
    story_protocol(2, RACE_REPETITIONS).cold_cooldown_ms
}

fn warm_cooldown_ms() -> u64 {
    story_protocol(2, RACE_REPETITIONS).warm_cooldown_ms
}

fn clean_cold_artifacts_enabled() -> bool {
    story_protocol(2, RACE_REPETITIONS).clean_artifacts
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

fn run_isolated_race_samples(
    test_name: &str,
    variants: &[RaceVariant],
    repetitions: usize,
) -> Vec<RaceRow> {
    let executable = std::env::current_exe().expect("current test executable should resolve");
    let protocol = story_protocol(2, repetitions);
    let mut samples = Vec::with_capacity(variants.len() * repetitions);
    for repetition in 0..repetitions {
        let mut rows = Vec::with_capacity(variants.len());
        let mut solutions = Vec::with_capacity(variants.len());
        for (index, variant) in variants.iter().enumerate() {
            println!(
                "[BVP Damp isolated cold] launching repetition {}/{} source={} variant={}",
                repetition + 1,
                repetitions,
                variant.source,
                variant.variant
            );
            let mut command = Command::new(&executable);
            command
                .arg("--exact")
                .arg(test_name)
                .arg("--ignored")
                .arg("--nocapture")
                .env(ISOLATED_STRESS_CHILD_INDEX_ENV, index.to_string())
                .env(ISOLATED_STRESS_CHILD_REPETITION_ENV, repetition.to_string());
            if protocol.worker_threads > 0 {
                command.env("RAYON_NUM_THREADS", protocol.worker_threads.to_string());
            }
            let output = command
                .output()
                .expect("isolated cold child process should launch");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "isolated cold child failed for {}:\nstdout:\n{}\nstderr:\n{}",
                variant.variant,
                stdout,
                stderr
            );
            let row_line = stdout
                .lines()
                .find(|line| line.starts_with(ISOLATED_RACE_ROW_MARKER))
                .unwrap_or_else(|| {
                    panic!(
                        "isolated cold child did not emit metrics for {}:\n{}",
                        variant.variant, stdout
                    )
                });
            let solution_line = stdout
                .lines()
                .find(|line| line.starts_with(ISOLATED_RACE_SOLUTION_MARKER))
                .unwrap_or_else(|| {
                    panic!(
                        "isolated cold child did not emit a solution for {}:\n{}",
                        variant.variant, stdout
                    )
                });
            let row = decode_isolated_race_row(row_line, variant);
            let solution = decode_isolated_solution(solution_line);
            let child_pid = stdout
                .lines()
                .find(|line| line.starts_with(ISOLATED_RACE_PID_MARKER))
                .and_then(|line| line.strip_prefix(ISOLATED_RACE_PID_MARKER).map(str::trim))
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_else(|| {
                    panic!("isolated cold child emitted no pid for {}", variant.variant)
                });
            remove_generated_aot_builds_for_child(child_pid);
            let cooldown_ms = cold_cooldown_ms();
            if cooldown_ms > 0 {
                thread::sleep(Duration::from_millis(cooldown_ms));
            }
            println!(
                "[BVP Damp isolated cold] finished source={} variant={} total_ms={:.3} symbolic_ms={:.3} status={}",
                row.source, row.variant, row.total_ms, row.symbolic_timer_ms, row.status
            );
            rows.push(row);
            solutions.push(Some(solution));
        }
        fill_solution_diffs(&mut rows, &solutions);
        samples.extend(rows);
    }
    samples
}

fn aggregate(values: impl IntoIterator<Item = f64>) -> Aggregate {
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Aggregate {
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
    Aggregate {
        mean,
        stddev: variance.sqrt(),
        min,
        max,
    }
}

fn summarize_reason<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut reasons = values
        .into_iter()
        .filter(|value| !value.is_empty() && *value != "-")
        .collect::<Vec<_>>();
    reasons.sort_unstable();
    reasons.dedup();
    match reasons.as_slice() {
        [] => "-".to_string(),
        [single] => (*single).to_string(),
        _ => "mixed".to_string(),
    }
}

fn summarize_variant(variant: &RaceVariant, samples: &[RaceRow]) -> RaceSummaryRow {
    let rows = samples
        .iter()
        .filter(|row| {
            row.source == variant.source
                && row.matrix == variant.matrix
                && row.variant == variant.variant
                && row.bootstrap_hint == variant.bootstrap_hint
        })
        .collect::<Vec<_>>();
    let ok_runs = rows.iter().filter(|row| row.status == "ok").count();
    let status = if ok_runs == rows.len() {
        format!("ok {ok_runs}/{}", rows.len())
    } else {
        let first_failure = rows
            .iter()
            .find(|row| row.status != "ok")
            .map(|row| row.status.as_str())
            .unwrap_or("unknown");
        format!("ok {ok_runs}/{}, first_failure={first_failure}", rows.len())
    };

    RaceSummaryRow {
        source: variant.source,
        matrix: variant.matrix,
        variant: variant.variant,
        bootstrap_hint: variant.bootstrap_hint,
        runs: rows.len(),
        ok_runs,
        total_ms: aggregate(rows.iter().map(|row| row.total_ms)),
        max_abs_solution: aggregate(rows.iter().map(|row| row.max_abs_solution)),
        solve_diff: aggregate(rows.iter().map(|row| row.solve_diff)),
        rel_x_diff: aggregate(rows.iter().map(|row| row.rel_x_diff)),
        iterations: aggregate(rows.iter().map(|row| row.iterations as f64)),
        linear_solves: aggregate(rows.iter().map(|row| row.linear_solves as f64)),
        jac_rebuilds: aggregate(rows.iter().map(|row| row.jac_rebuilds as f64)),
        grid_refinements: aggregate(rows.iter().map(|row| row.grid_refinements as f64)),
        final_grid_points: aggregate(rows.iter().map(|row| row.final_grid_points as f64)),
        total_timer_ms: aggregate(rows.iter().map(|row| row.total_timer_ms)),
        symbolic_timer_ms: aggregate(rows.iter().map(|row| row.symbolic_timer_ms)),
        linear_timer_ms: aggregate(rows.iter().map(|row| row.linear_timer_ms)),
        jac_timer_ms: aggregate(rows.iter().map(|row| row.jac_timer_ms)),
        fun_timer_ms: aggregate(rows.iter().map(|row| row.fun_timer_ms)),
        cb_residual_values_ms: aggregate(rows.iter().map(|row| row.cb_residual_values_ms)),
        cb_jacobian_values_ms: aggregate(rows.iter().map(|row| row.cb_jacobian_values_ms)),
        cb_jacobian_assembly_ms: aggregate(rows.iter().map(|row| row.cb_jacobian_assembly_ms)),
        residual_actual_jobs: aggregate(rows.iter().map(|row| row.residual_actual_jobs)),
        sparse_jacobian_actual_jobs: aggregate(
            rows.iter().map(|row| row.sparse_jacobian_actual_jobs),
        ),
        residual_work_per_job: aggregate(rows.iter().map(|row| row.residual_work_per_job)),
        sparse_jacobian_work_per_job: aggregate(
            rows.iter().map(|row| row.sparse_jacobian_work_per_job),
        ),
        residual_fallback_reason: summarize_reason(
            rows.iter().map(|row| row.residual_fallback_reason.as_str()),
        ),
        sparse_jacobian_fallback_reason: summarize_reason(
            rows.iter()
                .map(|row| row.sparse_jacobian_fallback_reason.as_str()),
        ),
        selected_backend: summarize_reason(rows.iter().map(|row| row.selected_backend.as_str())),
        symbolic_assembly_backend: summarize_reason(
            rows.iter()
                .map(|row| row.symbolic_assembly_backend.as_str()),
        ),
        aot_build_policy: summarize_reason(rows.iter().map(|row| row.aot_build_policy.as_str())),
        initial_generate_ms: aggregate(rows.iter().map(|row| row.initial_generate_ms)),
        initial_discretization_ms: aggregate(rows.iter().map(|row| row.initial_discretization_ms)),
        initial_symbolic_jacobian_ms: aggregate(
            rows.iter().map(|row| row.initial_symbolic_jacobian_ms),
        ),
        initial_symbolic_variable_sets_ms: aggregate(
            rows.iter().map(|row| row.initial_symbolic_variable_sets_ms),
        ),
        initial_symbolic_row_differentiation_ms: aggregate(
            rows.iter()
                .map(|row| row.initial_symbolic_row_differentiation_ms),
        ),
        initial_symbolic_dense_cache_ms: aggregate(
            rows.iter().map(|row| row.initial_symbolic_dense_cache_ms),
        ),
        initial_symbolic_sparse_flatten_ms: aggregate(
            rows.iter()
                .map(|row| row.initial_symbolic_sparse_flatten_ms),
        ),
        initial_sparse_prepare_ms: aggregate(rows.iter().map(|row| row.initial_sparse_prepare_ms)),
        initial_runtime_binding_ms: aggregate(
            rows.iter().map(|row| row.initial_runtime_binding_ms),
        ),
        initial_lambdify_jacobian_compile_ms: aggregate(
            rows.iter()
                .map(|row| row.initial_lambdify_jacobian_compile_ms),
        ),
        initial_lambdify_residual_compile_ms: aggregate(
            rows.iter()
                .map(|row| row.initial_lambdify_residual_compile_ms),
        ),
        post_build_generate_ms: aggregate(rows.iter().map(|row| row.post_build_generate_ms)),
        post_build_discretization_ms: aggregate(
            rows.iter().map(|row| row.post_build_discretization_ms),
        ),
        post_build_symbolic_jacobian_ms: aggregate(
            rows.iter().map(|row| row.post_build_symbolic_jacobian_ms),
        ),
        post_build_sparse_prepare_ms: aggregate(
            rows.iter().map(|row| row.post_build_sparse_prepare_ms),
        ),
        post_build_runtime_binding_ms: aggregate(
            rows.iter().map(|row| row.post_build_runtime_binding_ms),
        ),
        post_build_rebind_ms: aggregate(rows.iter().map(|row| row.post_build_rebind_ms)),
        aot_artifact_ms: aggregate(rows.iter().map(|row| row.aot_artifact_ms)),
        aot_module_ms: aggregate(rows.iter().map(|row| row.aot_module_ms)),
        aot_residual_lower_ms: aggregate(rows.iter().map(|row| row.aot_residual_lower_ms)),
        aot_jacobian_lower_ms: aggregate(rows.iter().map(|row| row.aot_jacobian_lower_ms)),
        aot_source_emit_ms: aggregate(rows.iter().map(|row| row.aot_source_emit_ms)),
        aot_packaging_ms: aggregate(rows.iter().map(|row| row.aot_packaging_ms)),
        aot_materialize_ms: aggregate(rows.iter().map(|row| row.aot_materialize_ms)),
        aot_compile_link_ms: aggregate(rows.iter().map(|row| row.aot_compile_link_ms)),
        aot_register_link_ms: aggregate(rows.iter().map(|row| row.aot_register_link_ms)),
        status,
    }
}

fn summarize_samples(variants: &[RaceVariant], samples: &[RaceRow]) -> Vec<RaceSummaryRow> {
    variants
        .iter()
        .map(|variant| summarize_variant(variant, samples))
        .collect()
}

fn fmt_agg(value: Aggregate) -> String {
    if value.mean.is_finite() {
        format!(
            "{:.3} +/- {:.3} [{:.3}, {:.3}]",
            value.mean, value.stddev, value.min, value.max
        )
    } else {
        "-".to_string()
    }
}

fn fmt_agg_short(value: Aggregate) -> String {
    if value.mean.is_finite() {
        format!("{:.3} +/- {:.3}", value.mean, value.stddev)
    } else {
        "-".to_string()
    }
}

fn fmt_agg_exp(value: Aggregate) -> String {
    if value.mean.is_finite() {
        format!("{:.3e} +/- {:.1e}", value.mean, value.stddev)
    } else {
        "-".to_string()
    }
}

fn fmt_sample_ms(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.3}")
    } else {
        "-".to_string()
    }
}

fn print_race_summary_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!("[BVP Damp race] summary table: all time columns are milliseconds.");
    println!(
        "source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status"
    );
    println!("{}", "-".repeat(190));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<8} | {:>2}/{:<2} | {:<32} | {:<18} | {:<18} | {:<18} | {}",
            row.source,
            row.matrix,
            row.variant,
            row.ok_runs,
            row.runs,
            fmt_agg(row.total_ms),
            fmt_agg_exp(row.solve_diff),
            fmt_agg_exp(row.rel_x_diff),
            fmt_agg_exp(row.max_abs_solution),
            row.status
        );
    }

    println!();
    println!(
        "[BVP Damp race] diagnostics table: all timer columns are milliseconds; counters are counts."
    );
    println!(
        "source   | matrix | variant | bootstrap_hint | solver_total_ms | symbolic/bootstrap_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re"
    );
    println!("{}", "-".repeat(220));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<8} | {:<24} | {:<18} | {:<21} | {:<18} | {:<18} | {:<18} | {:<14} | {:<14} | {:<14}",
            row.source,
            row.matrix,
            row.variant,
            row.bootstrap_hint,
            fmt_agg_short(row.total_timer_ms),
            fmt_agg_short(row.symbolic_timer_ms),
            fmt_agg_short(row.linear_timer_ms),
            fmt_agg_short(row.jac_timer_ms),
            fmt_agg_short(row.fun_timer_ms),
            fmt_agg_short(row.iterations),
            fmt_agg_short(row.linear_solves),
            fmt_agg_short(row.jac_rebuilds),
        );
    }
}

fn print_e2e_correctness_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition."
    );
    println!(
        "source   | matrix | variant    | chunking         | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status"
    );
    println!("{}", "-".repeat(190));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<16} | {:>2}/{:<2}  | {:<20} | {:<21} | {:<22} | {}",
            row.source,
            row.matrix,
            row.variant,
            row.bootstrap_hint,
            row.ok_runs,
            row.runs,
            fmt_agg_exp(row.solve_diff),
            fmt_agg_exp(row.rel_x_diff),
            fmt_agg_exp(row.max_abs_solution),
            row.status
        );
    }
}

fn print_e2e_performance_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] timing/counter table: all time columns are milliseconds; counters are counts."
    );
    println!(
        "source   | matrix | variant    | chunking         | total_ms mean+/-std [min,max] | solver_total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re"
    );
    println!("{}", "-".repeat(230));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<16} | {:<32} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<12} | {:<12}",
            row.source,
            row.matrix,
            row.variant,
            row.bootstrap_hint,
            fmt_agg(row.total_ms),
            fmt_agg_short(row.total_timer_ms),
            fmt_agg_short(row.symbolic_timer_ms),
            fmt_agg_short(row.linear_timer_ms),
            fmt_agg_short(row.jac_timer_ms),
            fmt_agg_short(row.fun_timer_ms),
            fmt_agg_short(row.iterations),
            fmt_agg_short(row.linear_solves),
            fmt_agg_short(row.jac_rebuilds),
        );
    }
}

fn print_isolated_cold_sample_table(title: &str, samples: &[RaceRow], rows_per_repetition: usize) {
    println!("{title}");
    println!(
        "[BVP Damp stress] raw process-isolated table: every row was executed in a fresh child process, while callback-internal parallel workers remain enabled."
    );
    println!(
        "rep | pos | source   | variant    | total_ms | symbolic_ms | initial_sym_jac_ms | compile_link_ms | residual_values_ms | jacobian_values_ms | res_jobs | jac_jobs | status"
    );
    println!("{}", "-".repeat(210));
    for (index, row) in samples.iter().enumerate() {
        println!(
            "{:>3} | {:>3} | {:<8} | {:<10} | {:>8} | {:>11} | {:>18} | {:>15} | {:>18} | {:>18} | {:>8} | {:>8} | {}",
            index / rows_per_repetition + 1,
            index % rows_per_repetition + 1,
            row.source,
            row.variant,
            fmt_sample_ms(row.total_ms),
            fmt_sample_ms(row.symbolic_timer_ms),
            fmt_sample_ms(row.initial_symbolic_jacobian_ms),
            fmt_sample_ms(row.aot_compile_link_ms),
            fmt_sample_ms(row.cb_residual_values_ms),
            fmt_sample_ms(row.cb_jacobian_values_ms),
            fmt_sample_ms(row.residual_actual_jobs),
            fmt_sample_ms(row.sparse_jacobian_actual_jobs),
            row.status
        );
    }
}

fn print_e2e_callback_stage_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] callback/runtime table: AOT linked callbacks report hot values, matrix assembly, actual jobs, fallback reason, and workload per job; Lambdify rows may be blank."
    );
    println!(
        "source   | matrix | variant    | chunking         | residual_values_ms | jacobian_values_ms | jacobian_assembly_ms | res_jobs | jac_jobs | res_work/job | jac_work/job | res_fallback | jac_fallback"
    );
    println!("{}", "-".repeat(230));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<16} | {:<18} | {:<18} | {:<20} | {:<8} | {:<8} | {:<12} | {:<12} | {:<12} | {:<12}",
            row.source,
            row.matrix,
            row.variant,
            row.bootstrap_hint,
            fmt_agg_short(row.cb_residual_values_ms),
            fmt_agg_short(row.cb_jacobian_values_ms),
            fmt_agg_short(row.cb_jacobian_assembly_ms),
            fmt_agg_short(row.residual_actual_jobs),
            fmt_agg_short(row.sparse_jacobian_actual_jobs),
            fmt_agg_short(row.residual_work_per_job),
            fmt_agg_short(row.sparse_jacobian_work_per_job),
            row.residual_fallback_reason,
            row.sparse_jacobian_fallback_reason,
        );
    }
}

fn print_e2e_lifecycle_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] lifecycle table: refinement-triggered regeneration is part of real wall-clock time."
    );
    println!(
        "source   | matrix | variant    | chunking         | assembly | selected_backend | build_policy | refinements | final_grid_points | symbolic_ms"
    );
    println!("{}", "-".repeat(180));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<16} | {:<8} | {:<16} | {:<12} | {:<18} | {:<18} | {:<15}",
            row.source,
            row.matrix,
            row.variant,
            row.bootstrap_hint,
            row.symbolic_assembly_backend,
            row.selected_backend,
            row.aot_build_policy,
            fmt_agg_short(row.grid_refinements),
            fmt_agg_short(row.final_grid_points),
            fmt_agg_short(row.symbolic_timer_ms),
        );
    }
}

fn print_e2e_bootstrap_pass_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] generated handoff pass table: post_build symbolic columns must stay blank after direct rebinding of a freshly built AOT artifact."
    );
    println!(
        "source   | matrix | variant    | initial_total | initial_discretize | initial_sym_jac | initial_prepare | initial_bind | post_build_total | post_discretize | post_sym_jac | post_prepare | post_bind | rebind_ms"
    );
    println!("{}", "-".repeat(236));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<15} | {:<18} | {:<15} | {:<15} | {:<12} | {:<16} | {:<15} | {:<12} | {:<12} | {:<12} | {:<12}",
            row.source,
            row.matrix,
            row.variant,
            fmt_agg_short(row.initial_generate_ms),
            fmt_agg_short(row.initial_discretization_ms),
            fmt_agg_short(row.initial_symbolic_jacobian_ms),
            fmt_agg_short(row.initial_sparse_prepare_ms),
            fmt_agg_short(row.initial_runtime_binding_ms),
            fmt_agg_short(row.post_build_generate_ms),
            fmt_agg_short(row.post_build_discretization_ms),
            fmt_agg_short(row.post_build_symbolic_jacobian_ms),
            fmt_agg_short(row.post_build_sparse_prepare_ms),
            fmt_agg_short(row.post_build_runtime_binding_ms),
            fmt_agg_short(row.post_build_rebind_ms),
        );
    }
}

fn print_e2e_symbolic_jacobian_detail_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] internal initial symbolic-Jacobian stages: dense_cache is expected to be near zero for Faer Sparse and Banded sparse-first routes."
    );
    println!(
        "source   | matrix | variant    | initial_sym_jac | variable_sets | row_diff | dense_cache | sparse_flatten"
    );
    println!("{}", "-".repeat(145));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15}",
            row.source,
            row.matrix,
            row.variant,
            fmt_agg_short(row.initial_symbolic_jacobian_ms),
            fmt_agg_short(row.initial_symbolic_variable_sets_ms),
            fmt_agg_short(row.initial_symbolic_row_differentiation_ms),
            fmt_agg_short(row.initial_symbolic_dense_cache_ms),
            fmt_agg_short(row.initial_symbolic_sparse_flatten_ms),
        );
    }
}

fn print_e2e_lambdify_binding_detail_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] Lambdify initial binding stages: callback compilation is setup work; AOT rows intentionally remain blank."
    );
    println!("source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile");
    println!("{}", "-".repeat(116));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<12} | {:<16} | {:<16}",
            row.source,
            row.matrix,
            row.variant,
            fmt_agg_short(row.initial_runtime_binding_ms),
            fmt_agg_short(row.initial_lambdify_jacobian_compile_ms),
            fmt_agg_short(row.initial_lambdify_residual_compile_ms),
        );
    }
}

fn print_e2e_aot_bootstrap_table(title: &str, rows: &[RaceSummaryRow]) {
    println!("{title}");
    println!(
        "[BVP Damp e2e] AOT cold-build table: compile_link is the external compiler/linker interval; artifact/module/lowering/source/packaging are nested codegen diagnostics and must not be summed; rows without AOT build are blank."
    );
    println!(
        "source   | matrix | variant    | artifact_total | module | residual_lower | jacobian_lower | source_emit | packaging | materialize | compile_link | register_link"
    );
    println!("{}", "-".repeat(215));
    for row in rows {
        println!(
            "{:<8} | {:<6} | {:<10} | {:<16} | {:<12} | {:<14} | {:<14} | {:<12} | {:<12} | {:<12} | {:<13} | {:<13}",
            row.source,
            row.matrix,
            row.variant,
            fmt_agg_short(row.aot_artifact_ms),
            fmt_agg_short(row.aot_module_ms),
            fmt_agg_short(row.aot_residual_lower_ms),
            fmt_agg_short(row.aot_jacobian_lower_ms),
            fmt_agg_short(row.aot_source_emit_ms),
            fmt_agg_short(row.aot_packaging_ms),
            fmt_agg_short(row.aot_materialize_ms),
            fmt_agg_short(row.aot_compile_link_ms),
            fmt_agg_short(row.aot_register_link_ms),
        );
    }
}

fn rebuild_release(config: GeneratedBackendConfig) -> GeneratedBackendConfig {
    config
        .with_aot_compile_dev_fastest()
        .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
        .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        })
}

fn release_matrix_config(
    config: GeneratedBackendConfig,
    chunking_policy: AotChunkingPolicy,
    execution_policy: AotExecutionPolicy,
) -> GeneratedBackendConfig {
    config
        .with_aot_compile_dev_fastest()
        .with_aot_execution_policy(execution_policy)
        .with_aot_chunking_policy(chunking_policy)
        .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        })
}

fn variant_filter_matches(variant: &RaceVariant, filter: &str) -> bool {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return true;
    }
    let haystack = format!(
        "{} {} {} {} {}/{}",
        variant.source,
        variant.matrix,
        variant.variant,
        variant.bootstrap_hint,
        variant.variant,
        variant.bootstrap_hint
    )
    .to_ascii_lowercase();
    haystack.contains(&filter)
}

fn apply_optional_release_matrix_filter(variants: &[RaceVariant]) -> Vec<RaceVariant> {
    let Ok(filter) = std::env::var("BVP_AOT_MATRIX_FILTER") else {
        return variants.to_vec();
    };
    let selected = variants
        .iter()
        .filter(|variant| variant_filter_matches(variant, &filter))
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "BVP_AOT_MATRIX_FILTER={filter:?} did not match any AOT matrix variant"
    );

    let mut filtered = variants
        .iter()
        .filter(|variant| variant.source == "Lambdify")
        .cloned()
        .collect::<Vec<_>>();
    for variant in selected {
        if !filtered.iter().any(|existing| {
            existing.source == variant.source
                && existing.matrix == variant.matrix
                && existing.variant == variant.variant
                && existing.bootstrap_hint == variant.bootstrap_hint
        }) {
            filtered.push(variant);
        }
    }
    assert!(
        !filtered.is_empty(),
        "BVP_AOT_MATRIX_FILTER={filter:?} did not match any AOT matrix variant"
    );
    println!(
        "[BVP Damp race] BVP_AOT_MATRIX_FILTER={filter:?}: running {}/{} variants, including Lambdify baselines",
        filtered.len(),
        variants.len()
    );
    filtered
}
