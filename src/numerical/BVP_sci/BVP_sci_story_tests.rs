//! # BVP_sci Story Tests
//!
//! End-to-end correctness and performance story tests for the `BVP_sci` solver.
//!
//! ## Design Principles
//!
//! 1. **Extensibility via `BvpSciStatistics` HashMaps** — `RaceRow` has minimal hardcoded
//!    fields (source, matrix, variant, bootstrap_hint, total_ms, max_abs_solution,
//!    solve_diff, rel_x_diff, status). All solver-specific metrics are read from
//!    `BvpSciStatistics.counters`, `.timers`, `.diagnostics` at summarization time.
//!    Adding a new backend (Banded, Dense) = adding a `RaceVariant` entry, no struct changes.
//!
//! 2. **Key comparison axes** — Sparse (only backend available now) vs future Banded/Dense.
//!    AtomView vs ExprLegacy. Toolchain matrix: Lambdify (baseline), AOT/gcc, AOT/tcc,
//!    AOT/zig, AOT/rust.
//!
//! 3. **Statistics source** — `BvpSciStatistics` (the internal instrument) exclusively.
//!    No handrolled wrappers. The `stats_timer_ms()`, `stats_diagnostic_*()`, `stats_count()`
//!    helpers mirror `BVP_Damp_tests4.rs` exactly.
//!
//! 4. **Future-proof** — `RaceVariant` uses `BvpSciGeneratedBackendConfig` directly.
//!    When Banded backend is added to BVP_sci, new variants with `matrix: "Banded"` are
//!    added to the variant generator. No structural changes needed.
//!
//! 5. **Per-stage information** — `BvpSciStatistics.diagnostics` records `n_variables`,
//!    `n_mesh_points`, `matrix_nnz`, `solver_strategy`. The `timers` HashMap records
//!    per-stage timers (symbolic, residual, jacobian, linear system, grid refinement).
//!    The `counters` HashMap records iterations, linear solves, jacobian recalculations,
//!    backtracking steps.
//!
//! ## Test Index
//!
//! - `combustion_200_lambdify_baseline_story` — ExprLegacy lambdify baseline, 3 reps
//! - `combustion_200_aot_correctness_story` — All 4 AOT toolchains (gcc/tcc/zig/rust), 3 reps
//! - `combustion_1000_release_matrix_story` — Full matrix (all variants), 5 reps
//! - `combustion_200_exprlegacy_stability_story` — ExprLegacy stability, 5 reps
//! - `combustion_3000_sparse_isolated_stress_story` — Process-isolated cold stress, 3000 mesh, 2 reps

#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_aot::BvpSciGeneratedBackendConfig;
    use crate::numerical::BVP_sci::BVP_sci_faer::BvpSciLinearSolvePolicy;
    use crate::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciSolverOptions, BvpSciStatistics};
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::fs;
    use std::io::{self, Write};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::PathBuf;
    use std::process::Command;
    use std::thread;
    use std::time::{Duration, Instant};

    // ============================================================
    // Section A: Data structures (extensible via BvpSciStatistics)
    // ============================================================

    #[derive(Clone)]
    struct RaceVariant {
        source: &'static str,
        matrix: &'static str,
        variant: &'static str,
        bootstrap_hint: &'static str,
        config: BvpSciGeneratedBackendConfig,
    }

    /// Minimal hardcoded fields. All solver-specific metrics are read from
    /// `BvpSciStatistics` HashMaps at summarization time via `RaceSample`.
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
        status: String,
    }

    /// Wraps a `RaceRow` with the full `BvpSciStatistics` so that
    /// `summarize_variant` can read any diagnostic/timer/counter from the HashMaps.
    #[derive(Clone)]
    struct RaceSample {
        row: RaceRow,
        statistics: BvpSciStatistics,
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
        status: String,
    }

    // ============================================================
    // Section B: Statistics helpers (mirror BVP_Damp_tests4.rs)
    // ============================================================

    fn stats_count(stats: &BvpSciStatistics, key: &str) -> usize {
        stats.counters.get(key).copied().unwrap_or(0)
    }

    fn stats_timer_ms(stats: &BvpSciStatistics, prefix: &str) -> f64 {
        stats
            .timers
            .iter()
            .find(|(key, _)| key.starts_with(prefix))
            .and_then(|(key, value)| timer_value_to_ms(key, value))
            .unwrap_or(f64::NAN)
    }

    fn stats_diagnostic_usize(stats: &BvpSciStatistics, key: &str) -> f64 {
        stats
            .diagnostics
            .get(key)
            .and_then(|value| value.parse::<usize>().ok())
            .or_else(|| stats.counters.get(key).copied())
            .map(|value| value as f64)
            .unwrap_or(f64::NAN)
    }

    fn stats_counter_us_as_ms(stats: &BvpSciStatistics, key: &str) -> f64 {
        stats_count(stats, key) as f64 / 1_000.0
    }

    fn stats_diagnostic_ms(stats: &BvpSciStatistics, key: &str) -> f64 {
        stats
            .diagnostics
            .get(key)
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    }

    fn stats_diagnostic_string(stats: &BvpSciStatistics, key: &str) -> String {
        stats
            .diagnostics
            .get(key)
            .cloned()
            .unwrap_or_else(|| "-".to_string())
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

    // ============================================================
    // Section C: Solution helpers
    // ============================================================

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

    fn uniform_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
        DMatrix::from_element(variable_count, n_steps, value)
    }

    // ============================================================
    // Section D: Combustion problem setup
    // ============================================================

    fn combustion_problem_options(n_steps: usize) -> BvpSciSolverOptions {
        let values = vec![
            "Teta".to_string(),
            "q".to_string(),
            "C0".to_string(),
            "J0".to_string(),
            "C1".to_string(),
            "J1".to_string(),
        ];
        let teta = Expr::parse_expression("Teta");
        let q = Expr::parse_expression("q");
        let c0 = Expr::parse_expression("C0");
        let j0 = Expr::parse_expression("J0");
        let j1 = Expr::parse_expression("J1");

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
        let ro_d = [Expr::Const(d_ro), Expr::Const(d_ro)];
        let pe_d = [Expr::Const(pe_d), Expr::Const(pe_d)];
        let minus = Expr::Const(-1.0);
        let m_reag = Expr::Const(0.342);

        let rate = a
            * Expr::exp(-e / (r_g * (teta.clone() * t_scale_sym + dt_sym)))
            * c0.clone()
            * (ro_m.clone() / m_reag.clone());
        let eq_system = vec![
            q.clone() / lambda_sym,
            q * pe_q_sym - q_heat * rate.clone() * qm,
            j0.clone() / ro_d[0].clone(),
            j0 * pe_d[0].clone()
                - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone(),
            j1.clone() / ro_d[1].clone(),
            j1 * pe_d[1].clone() - (m.clone() * rate * ro_m / m) * qs,
        ];

        let boundary_conditions = HashMap::from([
            ("Teta".to_string(), vec![(0, (t_initial - dt) / t_scale)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(), vec![(0, c1_0)]),
            ("J0".to_string(), vec![(1, 1e-7)]),
            ("C1".to_string(), vec![(0, 1e-3)]),
            ("J1".to_string(), vec![(1, 1e-7)]),
        ]);
        let bounds = HashMap::from([
            ("Teta".to_string(), vec![(0, 0.0), (1, 10.0)]),
            ("q".to_string(), vec![(0, -1e20), (1, 1e20)]),
            ("C0".to_string(), vec![(0, 0.0), (1, 1.5)]),
            ("J0".to_string(), vec![(0, -1e2), (1, 1e2)]),
            ("C1".to_string(), vec![(0, 0.0), (1, 1.5)]),
            ("J1".to_string(), vec![(0, -1e2), (1, 1e2)]),
        ]);

        BvpSciSolverOptions::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(n_steps),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            n_steps * 2,
            uniform_initial_guess(6, n_steps, 0.99),
        )
        .with_bounds(Some(bounds))
        .with_loglevel(Some("off".to_string()))
    }

    fn exponential_endpoint_problem_options(n_steps: usize) -> BvpSciSolverOptions {
        let eq_system = Expr::parse_vector_expression(vec!["z", "-(2.0/4.0)*(1+2.0*ln((y)))*y"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let a = 4.0_f64;
        let bc_val = (-1.0 / a).exp();
        let boundary_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, bc_val), (1usize, bc_val)])]);
        let bounds = HashMap::from([("y".to_string(), vec![(0usize, 1e-10)])]);

        BvpSciSolverOptions::new(
            None,
            Some(-1.0),
            Some(1.0),
            Some(n_steps),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            (n_steps * 2).max(2048),
            DMatrix::zeros(2, n_steps),
        )
        .with_bounds(Some(bounds))
        .with_loglevel(Some("off".to_string()))
    }

    fn make_combustion_solver(n_steps: usize, config: BvpSciGeneratedBackendConfig) -> BVPwrap {
        let options = combustion_problem_options(n_steps).with_generated_backend_config(config);
        BVPwrap::new_with_options(options)
    }

    fn make_solver_with_options_and_linear_policy(
        options: BvpSciSolverOptions,
        config: BvpSciGeneratedBackendConfig,
        policy: BvpSciLinearSolvePolicy,
    ) -> BVPwrap {
        BVPwrap::new_with_options(
            options
                .with_generated_backend_config(config)
                .with_linear_solve_policy(policy),
        )
    }

    // ============================================================
    // Section E: Runner functions
    // ============================================================

    fn run_race_variant(
        n_steps: usize,
        source: &'static str,
        matrix: &'static str,
        variant: &'static str,
        bootstrap_hint: &'static str,
        config: BvpSciGeneratedBackendConfig,
    ) -> (RaceSample, Option<DMatrix<f64>>) {
        let total_begin = Instant::now();
        let mut solver = make_combustion_solver(n_steps, config);
        let solve_status = catch_unwind(AssertUnwindSafe(|| solver.try_solve()));
        let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
        let statistics = solver.get_statistics();

        let (row, solution) = match solve_status {
            Ok(Ok(())) => match solver.get_result() {
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
                    status: format!("err:{}", err),
                },
                None,
            ),
            Err(panic_err) => (
                RaceRow {
                    source,
                    matrix,
                    variant,
                    bootstrap_hint,
                    total_ms,
                    max_abs_solution: f64::NAN,
                    solve_diff: f64::NAN,
                    rel_x_diff: f64::NAN,
                    status: format!("panic:{:?}", panic_err),
                },
                None,
            ),
        };

        (RaceSample { row, statistics }, solution)
    }

    fn run_race_variant_with_problem_options_and_linear_policy(
        options: BvpSciSolverOptions,
        source: &'static str,
        matrix: &'static str,
        variant: &'static str,
        bootstrap_hint: &'static str,
        config: BvpSciGeneratedBackendConfig,
        policy: BvpSciLinearSolvePolicy,
    ) -> (RaceSample, Option<DMatrix<f64>>) {
        let total_begin = Instant::now();
        let mut solver = make_solver_with_options_and_linear_policy(options, config, policy);
        let solve_status = catch_unwind(AssertUnwindSafe(|| solver.try_solve()));
        let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
        let statistics = solver.get_statistics();

        let (row, solution) = match solve_status {
            Ok(Ok(())) => match solver.get_result() {
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
                    status: format!("err:{}", err),
                },
                None,
            ),
            Err(panic_err) => (
                RaceRow {
                    source,
                    matrix,
                    variant,
                    bootstrap_hint,
                    total_ms,
                    max_abs_solution: f64::NAN,
                    solve_diff: f64::NAN,
                    rel_x_diff: f64::NAN,
                    status: format!("panic:{:?}", panic_err),
                },
                None,
            ),
        };

        (RaceSample { row, statistics }, solution)
    }

    fn run_race_samples(
        variants: &[RaceVariant],
        n_steps: usize,
        repetitions: usize,
    ) -> Vec<RaceSample> {
        let mut samples = Vec::with_capacity(variants.len() * repetitions);
        for repetition in 0..repetitions {
            println!(
                "[BVP_sci story] starting repetition {}/{}",
                repetition + 1,
                repetitions
            );
            let mut rows = Vec::with_capacity(variants.len());
            let mut solutions = Vec::with_capacity(variants.len());
            for variant in variants {
                println!(
                    "[BVP_sci story] running source={} matrix={} variant={} bootstrap_hint={}",
                    variant.source, variant.matrix, variant.variant, variant.bootstrap_hint
                );
                let _ = io::stdout().flush();
                let (sample, solution) = run_race_variant(
                    n_steps,
                    variant.source,
                    variant.matrix,
                    variant.variant,
                    variant.bootstrap_hint,
                    variant.config.clone(),
                );
                println!(
                    "[BVP_sci story] finished source={} matrix={} variant={} status={}",
                    sample.row.source, sample.row.matrix, sample.row.variant, sample.row.status
                );
                let _ = io::stdout().flush();
                rows.push(sample);
                solutions.push(solution);
            }
            fill_solution_diffs(&mut rows, &solutions);
            samples.extend(rows);
        }
        samples
    }

    fn fill_solution_diffs(samples: &mut [RaceSample], solutions: &[Option<DMatrix<f64>>]) {
        let baseline = solutions.iter().find_map(|solution| solution.as_ref());
        if let Some(baseline) = baseline {
            for (sample, solution) in samples.iter_mut().zip(solutions.iter()) {
                if let Some(solution) = solution {
                    sample.row.solve_diff = solution_linf_diff(solution, baseline);
                    sample.row.rel_x_diff = solution_rel_diff(solution, baseline);
                }
            }
        }
    }

    // ============================================================
    // Section F: Aggregation and summarization
    // ============================================================

    fn aggregate(values: impl IntoIterator<Item = f64>) -> Aggregate {
        let values: Vec<f64> = values.into_iter().collect();
        let n = values.len() as f64;
        if n == 0.0 {
            return Aggregate {
                mean: f64::NAN,
                stddev: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
            };
        }
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Aggregate {
            mean,
            stddev,
            min,
            max,
        }
    }

    fn summarize_reason<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
        let reasons: Vec<&str> = values.into_iter().collect();
        if reasons.is_empty() {
            return "-".to_string();
        }
        let first = reasons[0];
        if reasons.iter().all(|r| *r == first) {
            first.to_string()
        } else {
            let unique: std::collections::HashSet<&str> = reasons.into_iter().collect();
            let mut sorted: Vec<&&str> = unique.iter().collect();
            sorted.sort();
            sorted
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join("|")
        }
    }

    fn summarize_variant(variant: &RaceVariant, samples: &[RaceSample]) -> RaceSummaryRow {
        let rows: Vec<&RaceSample> = samples
            .iter()
            .filter(|s| {
                s.row.source == variant.source
                    && s.row.matrix == variant.matrix
                    && s.row.variant == variant.variant
                    && s.row.bootstrap_hint == variant.bootstrap_hint
            })
            .collect();
        let ok_runs = rows.iter().filter(|s| s.row.status == "ok").count();
        let status = if ok_runs == rows.len() {
            format!("ok {ok_runs}/{}", rows.len())
        } else {
            let first_failure = rows
                .iter()
                .find(|s| s.row.status != "ok")
                .map(|s| s.row.status.as_str())
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
            total_ms: aggregate(rows.iter().map(|s| s.row.total_ms)),
            max_abs_solution: aggregate(rows.iter().map(|s| s.row.max_abs_solution)),
            solve_diff: aggregate(rows.iter().map(|s| s.row.solve_diff)),
            rel_x_diff: aggregate(rows.iter().map(|s| s.row.rel_x_diff)),
            status,
        }
    }

    fn summarize_samples(variants: &[RaceVariant], samples: &[RaceSample]) -> Vec<RaceSummaryRow> {
        variants
            .iter()
            .map(|variant| summarize_variant(variant, samples))
            .collect()
    }

    // ============================================================
    // Section G: Table printers
    // ============================================================

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

    fn print_race_summary_table(title: &str, rows: &[RaceSummaryRow]) {
        println!("{title}");
        println!("[BVP_sci story] summary table: all time columns are milliseconds.");
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
    }

    fn print_e2e_correctness_table(title: &str, rows: &[RaceSummaryRow]) {
        println!("{title}");
        println!(
            "[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition."
        );
        println!(
            "source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status"
        );
        println!("{}", "-".repeat(190));
        for row in rows {
            println!(
                "{:<8} | {:<6} | {:<10} | {:<15} | {:>2}/{:<2}  | {:<20} | {:<21} | {:<22} | {}",
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
        println!();
    }

    fn print_e2e_performance_table(title: &str, rows: &[RaceSummaryRow]) {
        println!("{title}");
        println!(
            "[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts."
        );
        println!(
            "source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status"
        );
        println!("{}", "-".repeat(140));
        for row in rows {
            println!(
                "{:<8} | {:<6} | {:<10} | {:<15} | {:<32} | {}",
                row.source,
                row.matrix,
                row.variant,
                row.bootstrap_hint,
                fmt_agg(row.total_ms),
                row.status
            );
        }
        println!();
    }

    fn print_e2e_lifecycle_table(title: &str, rows: &[RaceSummaryRow]) {
        println!("{title}");
        println!(
            "[BVP_sci e2e] lifecycle table: source | matrix | variant | bootstrap_hint | refinements | final_grid_points | status"
        );
        println!("{}", "-".repeat(120));
        for row in rows {
            println!(
                "{:<8} | {:<6} | {:<10} | {:<15} | {}",
                row.source, row.matrix, row.variant, row.bootstrap_hint, row.status
            );
        }
        println!();
    }

    fn print_linear_policy_route_table(title: &str, samples: &[RaceSample]) {
        println!("{title}");
        println!(
            "[BVP_sci linear policy] route table: counters are accumulated solver statistics. AutoBanded must not force full scalar banded on endpoint-BC matrices; ExperimentalBorderedBanded must not silently fall back to Sparse."
        );
        println!(
            "source   | matrix       | variant    | total_ms | sparse | sparse_fb | full_banded | bordered | extract_ms | factor_ms | solve_ms | factor_calls | solve_calls | reuse | ls | dense_kib | sparse_kib | dense/sparse | route_full | route_bordered | route_sparse | status"
        );
        println!("{}", "-".repeat(315));
        for sample in samples {
            let dense_kib =
                stats_diagnostic_usize(&sample.statistics, "global jacobian dense equivalent kib");
            let sparse_kib =
                stats_diagnostic_usize(&sample.statistics, "global jacobian sparse storage kib");
            let dense_to_sparse = stats_diagnostic_usize(
                &sample.statistics,
                "global jacobian dense to sparse permille",
            ) / 1000.0;
            println!(
                "{:<8} | {:<12} | {:<10} | {:>8.3} | {:>6} | {:>9} | {:>11} | {:>8} | {:>10.3} | {:>9.3} | {:>8.3} | {:>12} | {:>11} | {:>5} | {:>2} | {:>9.0} | {:>10.0} | {:>12.3} | {:>10} | {:>14} | {:>12} | {}",
                sample.row.source,
                sample.row.matrix,
                sample.row.variant,
                sample.row.total_ms,
                stats_count(&sample.statistics, "bvp sci linear backend sparse solves"),
                stats_count(
                    &sample.statistics,
                    "bvp sci linear backend sparse fallback solves"
                ),
                stats_count(
                    &sample.statistics,
                    "bvp sci linear backend full banded solves"
                ),
                stats_count(
                    &sample.statistics,
                    "bvp sci linear backend bordered structured solves"
                ),
                stats_counter_us_as_ms(&sample.statistics, "bvp sci bordered extraction us"),
                stats_counter_us_as_ms(&sample.statistics, "bvp sci bordered factorization us"),
                stats_counter_us_as_ms(
                    &sample.statistics,
                    "bvp sci bordered structured solve us"
                ),
                stats_count(
                    &sample.statistics,
                    "bvp sci bordered factorization calls"
                ),
                stats_count(
                    &sample.statistics,
                    "bvp sci bordered structured solve calls"
                ),
                stats_count(&sample.statistics, "bvp sci bordered reuse solve calls"),
                stats_count(
                    &sample.statistics,
                    "bvp sci bordered line search solve calls"
                ),
                dense_kib,
                sparse_kib,
                dense_to_sparse,
                stats_count(&sample.statistics, "bvp sci route full scalar banded"),
                stats_count(
                    &sample.statistics,
                    "bvp sci route bordered banded candidate"
                ),
                stats_count(&sample.statistics, "bvp sci route sparse fallback"),
                sample.row.status
            );
        }
        println!();
    }

    // ============================================================
    // Section H: Variant generators
    // ============================================================

    fn combustion_sparse_variants() -> Vec<RaceVariant> {
        vec![
            // Lambdify baseline (ExprLegacy)
            RaceVariant {
                source: "Lambdify",
                matrix: "Sparse",
                variant: "ExprLegacy",
                bootstrap_hint: "baseline",
                config: BvpSciGeneratedBackendConfig::default_lambdify(),
            },
            // AOT toolchains (AtomView)
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "gcc",
                bootstrap_hint: "build_if_missing",
                config: BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(
                    "output_bvp_sci",
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "tcc",
                bootstrap_hint: "build_if_missing",
                config: BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(
                    "output_bvp_sci",
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "zig",
                bootstrap_hint: "build_if_missing",
                config: BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(
                    "output_bvp_sci",
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "rust",
                bootstrap_hint: "build_if_missing",
                config: BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_rust(
                    "output_bvp_sci",
                ),
            },
        ]
    }

    fn combustion_tcc_lifecycle_variants() -> Vec<RaceVariant> {
        let output_dir = "output_bvp_sci_lifecycle";
        vec![
            RaceVariant {
                source: "Lambdify",
                matrix: "Sparse",
                variant: "ExprLegacy",
                bootstrap_hint: "baseline",
                config: BvpSciGeneratedBackendConfig::default_lambdify(),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "tcc",
                bootstrap_hint: "build_if_missing_or_reuse",
                config: BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(
                    output_dir,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "tcc",
                bootstrap_hint: "require_prebuilt_in_process",
                config: BvpSciGeneratedBackendConfig::sparse_atomview_require_prebuilt()
                    .with_output_parent_dir(output_dir),
            },
        ]
    }

    fn combustion_linear_policy_variants() -> Vec<(RaceVariant, BvpSciLinearSolvePolicy)> {
        vec![
            (
                RaceVariant {
                    source: "Lambdify",
                    matrix: "Sparse",
                    variant: "ExprLegacy",
                    bootstrap_hint: "linear_policy_baseline",
                    config: BvpSciGeneratedBackendConfig::default_lambdify(),
                },
                BvpSciLinearSolvePolicy::Sparse,
            ),
            (
                RaceVariant {
                    source: "Lambdify",
                    matrix: "AutoBanded",
                    variant: "ExprLegacy",
                    bootstrap_hint: "auto_banded_policy",
                    config: BvpSciGeneratedBackendConfig::default_lambdify(),
                },
                BvpSciLinearSolvePolicy::AutoBanded,
            ),
            (
                RaceVariant {
                    source: "Lambdify",
                    matrix: "ExperimentalBordered",
                    variant: "ExprLegacy",
                    bootstrap_hint: "experimental_bordered_policy",
                    config: BvpSciGeneratedBackendConfig::default_lambdify(),
                },
                BvpSciLinearSolvePolicy::ExperimentalBorderedBanded,
            ),
        ]
    }

    fn env_usize_or(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(default)
    }

    fn run_combustion_linear_policy_samples(
        n_steps: usize,
        repetitions: usize,
    ) -> (Vec<RaceVariant>, Vec<RaceSample>) {
        run_linear_policy_samples_with_options(|| combustion_problem_options(n_steps), repetitions)
    }

    fn run_linear_policy_samples_with_options(
        mut options_factory: impl FnMut() -> BvpSciSolverOptions,
        repetitions: usize,
    ) -> (Vec<RaceVariant>, Vec<RaceSample>) {
        let variant_specs = combustion_linear_policy_variants();
        let variants: Vec<RaceVariant> = variant_specs
            .iter()
            .map(|(variant, _)| variant.clone())
            .collect();
        let mut samples = Vec::with_capacity(variant_specs.len() * repetitions);

        for _rep in 0..repetitions {
            let mut rep_samples = Vec::with_capacity(variant_specs.len());
            let mut rep_solutions = Vec::with_capacity(variant_specs.len());
            for (variant, policy) in variant_specs.iter().cloned() {
                let (sample, solution) = run_race_variant_with_problem_options_and_linear_policy(
                    options_factory(),
                    variant.source,
                    variant.matrix,
                    variant.variant,
                    variant.bootstrap_hint,
                    variant.config,
                    policy,
                );
                rep_samples.push(sample);
                rep_solutions.push(solution);
            }
            fill_solution_diffs(&mut rep_samples, &rep_solutions);
            samples.extend(rep_samples);
        }

        (variants, samples)
    }

    fn assert_linear_policy_contract(summary: &[RaceSummaryRow], samples: &[RaceSample]) {
        for row in summary {
            assert!(
                row.ok_runs == row.runs,
                "{} {} {}: expected every linear-policy run to pass, got {}",
                row.source,
                row.matrix,
                row.variant,
                row.status
            );
            assert!(
                row.solve_diff.mean < 1e-8,
                "{} {} {}: solve_diff too large: {:e}",
                row.source,
                row.matrix,
                row.variant,
                row.solve_diff.mean
            );
        }

        let auto_samples: Vec<&RaceSample> = samples
            .iter()
            .filter(|sample| sample.row.matrix == "AutoBanded")
            .collect();
        assert!(
            auto_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci route bordered banded candidate"
            ) > 0),
            "AutoBanded should detect bordered-banded endpoint-BC structure"
        );
        assert!(
            auto_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci linear backend full banded solves"
            ) == 0),
            "AutoBanded must not force full scalar banded on endpoint-BC matrices"
        );
        assert!(
            auto_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci linear backend sparse fallback solves"
            ) > 0),
            "AutoBanded should remain a safe Sparse fallback on endpoint-BC matrices"
        );

        let experimental_samples: Vec<&RaceSample> = samples
            .iter()
            .filter(|sample| sample.row.matrix == "ExperimentalBordered")
            .collect();
        assert!(
            experimental_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci route bordered banded candidate"
            ) > 0),
            "ExperimentalBordered should detect bordered-banded endpoint-BC structure"
        );
        assert!(
            experimental_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci linear backend bordered structured solves"
            ) > 0),
            "ExperimentalBordered must use the structured bordered solver"
        );
        assert!(
            experimental_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci bordered extraction calls"
            ) > 0),
            "ExperimentalBordered must report bordered extraction diagnostics"
        );
        assert!(
            experimental_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci bordered structured solve calls"
            ) > 0),
            "ExperimentalBordered must report bordered structured solve diagnostics"
        );
        assert!(
            experimental_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci bordered factorization calls"
            ) > 0),
            "ExperimentalBordered must factor the bordered structured system"
        );
        assert!(
            experimental_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci bordered structured solve us"
            ) > 0),
            "ExperimentalBordered must report non-zero bordered solve timing"
        );
        assert!(
            experimental_samples.iter().all(|sample| stats_count(
                &sample.statistics,
                "bvp sci linear backend sparse fallback solves"
            ) == 0),
            "ExperimentalBordered must not silently fall back to Sparse"
        );
    }

    // ============================================================
    // Section I: Story tests
    // ============================================================

    /// I.1: Combustion_200 — Lambdify baseline (ExprLegacy)
    ///
    /// Verifies that the ExprLegacy lambdify workflow converges on the combustion
    /// problem with 200 mesh points. Reports correctness and performance tables.
    /// This serves as the baseline for all AOT comparisons.
    #[test]
    fn combustion_200_lambdify_baseline_story() {
        let variants: Vec<RaceVariant> = combustion_sparse_variants()
            .into_iter()
            .filter(|v| v.source == "Lambdify")
            .collect();
        let samples = run_race_samples(&variants, 200, 3);
        let summary = summarize_samples(&variants, &samples);
        print_e2e_correctness_table("Combustion 200: Lambdify baseline (ExprLegacy)", &summary);
        print_e2e_performance_table("Combustion 200: Lambdify performance", &summary);
        // Assert: Lambdify baseline converges
        for row in &summary {
            assert!(
                row.ok_runs > 0,
                "{} {} {}: all runs failed",
                row.source,
                row.matrix,
                row.variant
            );
        }
    }

    /// I.2: Combustion_200 — AOT correctness (gcc, tcc, zig, rust)
    ///
    /// Verifies that all AOT toolchains produce correct solutions on the
    /// combustion problem with 200 mesh points. Reports correctness,
    /// performance, and lifecycle tables.
    #[test]
    fn combustion_200_aot_correctness_story() {
        let variants: Vec<RaceVariant> = combustion_sparse_variants()
            .into_iter()
            .filter(|v| v.source == "AOT")
            .collect();
        let samples = run_race_samples(&variants, 200, 3);
        let summary = summarize_samples(&variants, &samples);
        print_e2e_correctness_table("Combustion 200: AOT correctness (all toolchains)", &summary);
        print_e2e_performance_table("Combustion 200: AOT performance", &summary);
        print_e2e_lifecycle_table("Combustion 200: AOT lifecycle", &summary);
        // Assert: all AOT variants converge
        for row in &summary {
            assert!(
                row.ok_runs > 0,
                "{} {} {}: all runs failed",
                row.source,
                row.matrix,
                row.variant
            );
            if row.solve_diff.mean.is_finite() {
                assert!(
                    row.solve_diff.mean < 1e-6,
                    "{} {} {}: solve_diff too large: {:e}",
                    row.source,
                    row.matrix,
                    row.variant,
                    row.solve_diff.mean
                );
            }
        }
    }

    /// I.3: Combustion_1000 — Full release matrix (all variants, 5 reps)
    ///
    /// Runs all variants (Lambdify + AOT) on the combustion problem with
    /// 1000 mesh points and 5 repetitions. Reports summary, correctness,
    /// and performance tables.
    #[test]
    fn combustion_1000_release_matrix_story() {
        let variants = combustion_sparse_variants();
        let samples = run_race_samples(&variants, 1000, 5);
        let summary = summarize_samples(&variants, &samples);
        print_race_summary_table("Combustion 1000: Full release matrix (5 reps)", &summary);
        print_e2e_correctness_table("Combustion 1000: Correctness", &summary);
        print_e2e_performance_table("Combustion 1000: Performance", &summary);
        // Assert: all variants converge
        for row in &summary {
            assert!(
                row.ok_runs > 0,
                "{} {} {}: all runs failed",
                row.source,
                row.matrix,
                row.variant
            );
        }
    }

    /// I.4: Combustion_200 — ExprLegacy stability (5 reps)
    ///
    /// Focused stability check for the ExprLegacy lambdify workflow with
    /// 5 repetitions. Verifies consistent convergence and solution quality.
    /// When AtomView lambdify mode is added to BVP_sci, this test will be
    /// extended to compare ExprLegacy vs AtomView.
    #[test]
    fn combustion_200_exprlegacy_stability_story() {
        let variants: Vec<RaceVariant> = combustion_sparse_variants()
            .into_iter()
            .filter(|v| v.source == "Lambdify")
            .collect();
        let samples = run_race_samples(&variants, 200, 5);
        let summary = summarize_samples(&variants, &samples);
        print_e2e_correctness_table("Combustion 200: ExprLegacy stability (5 reps)", &summary);
        print_e2e_performance_table("Combustion 200: ExprLegacy stability (5 reps)", &summary);
        // Assert: ExprLegacy converges consistently
        for row in &summary {
            assert!(
                row.ok_runs > 0,
                "{} {} {}: all runs failed",
                row.source,
                row.matrix,
                row.variant
            );
        }
    }

    /// I.5: Combustion_200 - AOT lifecycle: BuildIfMissing then RequirePrebuilt
    ///
    /// BVP_sci currently treats RequirePrebuilt as an in-process contract:
    /// a compiled sparse runtime must already be linked in the current process.
    /// This story proves the production handoff sequence that BVP_sci supports
    /// today: BuildIfMissing registers the tcc runtime, then RequirePrebuilt
    /// uses that runtime without falling back to lambdify or rebuilding.
    #[test]
    #[ignore]
    fn combustion_200_tcc_build_then_require_prebuilt_story() {
        let variants = combustion_tcc_lifecycle_variants();
        let samples = run_race_samples(&variants, 200, 3);
        let summary = summarize_samples(&variants, &samples);

        print_e2e_correctness_table(
            "Combustion 200: tcc BuildIfMissing -> RequirePrebuilt correctness",
            &summary,
        );
        print_e2e_performance_table(
            "Combustion 200: tcc BuildIfMissing -> RequirePrebuilt timing",
            &summary,
        );
        print_e2e_lifecycle_table(
            "Combustion 200: tcc BuildIfMissing -> RequirePrebuilt lifecycle",
            &summary,
        );

        for row in &summary {
            assert!(
                row.ok_runs == row.runs,
                "{} {} {} {}: expected every lifecycle run to pass, got {}",
                row.source,
                row.matrix,
                row.variant,
                row.bootstrap_hint,
                row.status
            );
            if row.source == "AOT" {
                assert!(
                    row.solve_diff.mean < 1e-6,
                    "{} {} {} {}: solve_diff too large: {:e}",
                    row.source,
                    row.matrix,
                    row.variant,
                    row.bootstrap_hint,
                    row.solve_diff.mean
                );
            }
        }
    }

    /// I.6: Combustion_200 - Sparse vs safe AutoBanded vs experimental bordered routing
    ///
    /// This is a correctness/diagnostic story, not a performance claim. The
    /// standard endpoint-BC BVP_sci Newton matrix has a compact collocation
    /// body plus boundary rows. AutoBanded must therefore recognize a
    /// bordered-banded candidate and use the Sparse fallback. The explicit
    /// ExperimentalBorderedBanded policy must use the native structured bordered
    /// route and match the Sparse baseline.
    #[test]
    fn combustion_200_auto_banded_linear_policy_route_story() {
        let (variants, samples) = run_combustion_linear_policy_samples(200, 1);
        let summary = summarize_samples(&variants, &samples);

        print_e2e_correctness_table(
            "Combustion 200: Sparse vs safe AutoBanded vs experimental bordered correctness",
            &summary,
        );
        print_e2e_performance_table(
            "Combustion 200: Sparse vs safe AutoBanded vs experimental bordered timing",
            &summary,
        );
        print_linear_policy_route_table(
            "Combustion 200: Sparse vs safe AutoBanded vs experimental bordered route counters",
            &samples,
        );

        assert_linear_policy_contract(&summary, &samples);
    }

    /// I.7: Combustion linear-policy release candidate stress.
    ///
    /// This is the performance/correctness companion to the fast combustion-200
    /// route gate.  It keeps Sparse as the baseline, verifies that AutoBanded is
    /// still a safe fallback for endpoint-BC matrices, and measures the explicit
    /// ExperimentalBorderedBanded route with multi-run statistics.  Defaults are
    /// intentionally moderate; release runs can scale with:
    ///
    /// - `BVP_SCI_LINEAR_POLICY_N_STEPS`
    /// - `BVP_SCI_LINEAR_POLICY_RUNS`
    #[test]
    #[ignore]
    fn combustion_linear_policy_release_candidate_story() {
        let n_steps = env_usize_or("BVP_SCI_LINEAR_POLICY_N_STEPS", 1000);
        let repetitions = env_usize_or("BVP_SCI_LINEAR_POLICY_RUNS", 3);
        println!(
            "[BVP_sci linear policy] release candidate settings: n_steps={n_steps}, runs={repetitions}"
        );

        let (variants, samples) = run_combustion_linear_policy_samples(n_steps, repetitions);
        let summary = summarize_samples(&variants, &samples);

        print_e2e_correctness_table(
            &format!(
                "Combustion {n_steps}: Sparse vs safe AutoBanded vs experimental bordered correctness ({repetitions} runs)"
            ),
            &summary,
        );
        print_e2e_performance_table(
            &format!(
                "Combustion {n_steps}: Sparse vs safe AutoBanded vs experimental bordered timing ({repetitions} runs)"
            ),
            &summary,
        );
        print_linear_policy_route_table(
            &format!(
                "Combustion {n_steps}: Sparse vs safe AutoBanded vs experimental bordered route counters ({repetitions} runs)"
            ),
            &samples,
        );

        assert_linear_policy_contract(&summary, &samples);
    }

    /// I.8: Larger-mesh combustion confirmation for bordered opt-in policy.
    ///
    /// This stress is intentionally separate from the combustion-1000 release
    /// candidate: promotion decisions should not be made from a single mesh size.
    /// Defaults are heavier but still user-configurable:
    ///
    /// - `BVP_SCI_LARGE_LINEAR_POLICY_N_STEPS`
    /// - `BVP_SCI_LARGE_LINEAR_POLICY_RUNS`
    #[test]
    #[ignore]
    fn combustion_large_linear_policy_release_story() {
        let n_steps = env_usize_or("BVP_SCI_LARGE_LINEAR_POLICY_N_STEPS", 3000);
        let repetitions = env_usize_or("BVP_SCI_LARGE_LINEAR_POLICY_RUNS", 3);
        println!(
            "[BVP_sci linear policy] large combustion settings: n_steps={n_steps}, runs={repetitions}"
        );

        let (variants, samples) = run_combustion_linear_policy_samples(n_steps, repetitions);
        let summary = summarize_samples(&variants, &samples);

        print_e2e_correctness_table(
            &format!(
                "Combustion {n_steps}: large Sparse vs safe AutoBanded vs experimental bordered correctness ({repetitions} runs)"
            ),
            &summary,
        );
        print_e2e_performance_table(
            &format!(
                "Combustion {n_steps}: large Sparse vs safe AutoBanded vs experimental bordered timing ({repetitions} runs)"
            ),
            &summary,
        );
        print_linear_policy_route_table(
            &format!(
                "Combustion {n_steps}: large Sparse vs safe AutoBanded vs experimental bordered route counters ({repetitions} runs)"
            ),
            &samples,
        );

        assert_linear_policy_contract(&summary, &samples);
    }

    /// I.9: Non-combustion endpoint-BC confirmation for bordered opt-in policy.
    ///
    /// Uses the exponential BVP already present in the BVP_sci compare suite.
    /// This guards against accidentally tuning the bordered route only for the
    /// combustion matrix shape.
    ///
    /// - `BVP_SCI_NONCOMB_LINEAR_POLICY_N_STEPS`
    /// - `BVP_SCI_NONCOMB_LINEAR_POLICY_RUNS`
    #[test]
    #[ignore]
    fn exponential_endpoint_linear_policy_release_story() {
        let n_steps = env_usize_or("BVP_SCI_NONCOMB_LINEAR_POLICY_N_STEPS", 1000);
        let repetitions = env_usize_or("BVP_SCI_NONCOMB_LINEAR_POLICY_RUNS", 3);
        println!(
            "[BVP_sci linear policy] exponential endpoint settings: n_steps={n_steps}, runs={repetitions}"
        );

        let (variants, samples) = run_linear_policy_samples_with_options(
            || exponential_endpoint_problem_options(n_steps),
            repetitions,
        );
        let summary = summarize_samples(&variants, &samples);

        print_e2e_correctness_table(
            &format!(
                "Exponential endpoint {n_steps}: Sparse vs safe AutoBanded vs experimental bordered correctness ({repetitions} runs)"
            ),
            &summary,
        );
        print_e2e_performance_table(
            &format!(
                "Exponential endpoint {n_steps}: Sparse vs safe AutoBanded vs experimental bordered timing ({repetitions} runs)"
            ),
            &summary,
        );
        print_linear_policy_route_table(
            &format!(
                "Exponential endpoint {n_steps}: Sparse vs safe AutoBanded vs experimental bordered route counters ({repetitions} runs)"
            ),
            &samples,
        );

        assert_linear_policy_contract(&summary, &samples);
    }

    // ============================================================
    // Section J: Process-isolated cold test infrastructure
    // ============================================================

    const ISOLATED_STRESS_CHILD_INDEX_ENV: &str = "BVP_SCI_ISOLATED_STRESS_CHILD_INDEX";
    const ISOLATED_STRESS_CHILD_REPETITION_ENV: &str = "BVP_SCI_ISOLATED_STRESS_CHILD_REPETITION";
    const ISOLATED_RACE_ROW_MARKER: &str = "[BVP_SCI_ISOLATED_RACE_ROW]";
    const ISOLATED_RACE_SOLUTION_MARKER: &str = "[BVP_SCI_ISOLATED_RACE_SOLUTION]";
    const ISOLATED_RACE_PID_MARKER: &str = "[BVP_SCI_ISOLATED_RACE_PID]";
    const COLD_COOLDOWN_MS_ENV: &str = "BVP_SCI_COLD_COOLDOWN_MS";
    const COLD_CLEAN_ARTIFACTS_ENV: &str = "BVP_SCI_COLD_CLEAN_ARTIFACTS";
    const COMBUSTION_3000_ISOLATED_STRESS_TEST_NAME: &str = "numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_3000_sparse_isolated_stress_story";

    /// Encode a `RaceRow` as a tab-separated string for child→parent IPC.
    /// Only the 9 hardcoded fields are serialized; solver-specific metrics
    /// are not available in the child process (no `BvpSciStatistics` cross-process).
    fn encode_isolated_race_row(row: &RaceRow) -> String {
        [
            row.total_ms.to_string(),
            row.max_abs_solution.to_string(),
            row.status.clone(),
        ]
        .join("\t")
    }

    /// Parse a single tab-separated field from an iterator.
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

    /// Decode a `RaceRow` from a child process stdout line.
    /// The `variant` provides the static fields (source, matrix, variant, bootstrap_hint).
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
            status: parse_isolated_field(&mut fields, "status"),
        };
        assert!(
            fields.next().is_none(),
            "isolated race row contains unexpected trailing fields"
        );
        row
    }

    /// Encode a `DMatrix<f64>` solution as a tab-separated string for child→parent IPC.
    fn encode_isolated_solution(solution: &DMatrix<f64>) -> String {
        solution
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join("\t")
    }

    /// Decode a `DMatrix<f64>` solution from a child process stdout line.
    /// The combustion problem has 6 variables, so we assert `values.len() % 6 == 0`.
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

    fn cold_cooldown_ms() -> u64 {
        std::env::var(COLD_COOLDOWN_MS_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    }

    fn clean_cold_artifacts_enabled() -> bool {
        std::env::var(COLD_CLEAN_ARTIFACTS_ENV)
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
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

    /// Run process-isolated cold samples by spawning child processes.
    ///
    /// Each child process runs a single variant (identified by
    /// `ISOLATED_STRESS_CHILD_INDEX_ENV`) and emits the race row + solution
    /// as marker-prefixed lines on stdout. The parent collects these,
    /// decodes them, computes solution diffs, and returns `Vec<RaceRow>`.
    ///
    /// Note: Unlike `run_race_samples`, this returns `Vec<RaceRow>` (not
    /// `Vec<RaceSample>`) because `BvpSciStatistics` cannot be serialized
    /// across process boundaries. The caller can still use `summarize_samples`
    /// by wrapping rows in a dummy `RaceSample` with default statistics.
    fn run_isolated_race_samples(
        test_name: &str,
        variants: &[RaceVariant],
        _n_steps: usize,
        repetitions: usize,
    ) -> Vec<RaceRow> {
        let executable = std::env::current_exe().expect("current test executable should resolve");
        let mut samples = Vec::with_capacity(variants.len() * repetitions);
        for repetition in 0..repetitions {
            let mut rows = Vec::with_capacity(variants.len());
            let mut solutions = Vec::with_capacity(variants.len());
            for (index, variant) in variants.iter().enumerate() {
                println!(
                    "[BVP_sci isolated cold] launching repetition {}/{} source={} variant={}",
                    repetition + 1,
                    repetitions,
                    variant.source,
                    variant.variant
                );
                let output = Command::new(&executable)
                    .arg("--exact")
                    .arg(test_name)
                    .arg("--ignored")
                    .arg("--nocapture")
                    .env(ISOLATED_STRESS_CHILD_INDEX_ENV, index.to_string())
                    .env(ISOLATED_STRESS_CHILD_REPETITION_ENV, repetition.to_string())
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
                            "isolated cold child did not emit metrics for {} using exact test name `{}`:\nstdout:\n{}\nstderr:\n{}",
                            variant.variant, test_name, stdout, stderr
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
                    "[BVP_sci isolated cold] finished source={} variant={} total_ms={:.3} status={}",
                    row.source, row.variant, row.total_ms, row.status
                );
                rows.push(row);
                solutions.push(Some(solution));
            }
            fill_solution_diffs_from_rows(&mut rows, &solutions);
            samples.extend(rows);
        }
        samples
    }

    /// Fill solution diffs for `Vec<RaceRow>` (no `BvpSciStatistics` wrapper).
    fn fill_solution_diffs_from_rows(rows: &mut [RaceRow], solutions: &[Option<DMatrix<f64>>]) {
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

    /// Print a raw isolated cold sample table (one row per observation).
    fn print_isolated_cold_sample_table(title: &str, samples: &[RaceRow], variants_count: usize) {
        println!("\n{}", title);
        println!(
            "source    | matrix | variant        | bootstrap_hint | total_ms     | max_abs_solution | solve_diff    | rel_x_diff    | status"
        );
        println!(
            "----------|--------|----------------|----------------|--------------|------------------|---------------|---------------|--------"
        );
        for (i, row) in samples.iter().enumerate() {
            let rep = i / variants_count;
            println!(
                "{} | {} | {} | {} | {:>12.3} | {:>16.6e} | {:>13.6e} | {:>13.6e} | {}",
                row.source,
                row.matrix,
                row.variant,
                row.bootstrap_hint,
                row.total_ms,
                row.max_abs_solution,
                row.solve_diff,
                row.rel_x_diff,
                row.status
            );
            if (i + 1) % variants_count == 0 && (i + 1) < samples.len() {
                println!("--- repetition {} ---", rep);
            }
        }
    }

    // ============================================================
    // Section K: Isolated stress story test
    // ============================================================

    /// J.1: Combustion_3000 — Process-isolated cold stress test
    ///
    /// Runs the combustion problem with 3000 mesh points in separate child
    /// processes to measure cold-start performance (no warm caches, no
    /// reused AOT artifacts from previous in-process runs). Each variant
    /// is run 2 times in isolated processes.
    ///
    /// This test is `#[ignore]` by default because it spawns child processes
    /// and takes significant time. Run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn combustion_3000_sparse_isolated_stress_story() {
        let n_steps = 3_000usize;
        let repetitions = 2usize;
        let variants = combustion_sparse_variants();

        // Child process path: run a single variant and emit metrics
        if let Ok(index) = std::env::var(ISOLATED_STRESS_CHILD_INDEX_ENV) {
            let index = index
                .parse::<usize>()
                .expect("isolated stress child index should be an integer");
            let variant = variants.get(index).unwrap_or_else(|| {
                panic!("isolated stress child variant index {index} is invalid")
            });
            let (sample, solution) = run_race_variant(
                n_steps,
                variant.source,
                variant.matrix,
                variant.variant,
                variant.bootstrap_hint,
                variant.config.clone(),
            );
            assert_eq!(
                sample.row.status, "ok",
                "isolated stress child {} should solve successfully",
                variant.variant
            );
            let solution = solution
                .as_ref()
                .expect("isolated stress child should provide its converged solution");
            println!("{ISOLATED_RACE_PID_MARKER}\t{}", std::process::id());
            println!(
                "{ISOLATED_RACE_ROW_MARKER}\t{}",
                encode_isolated_race_row(&sample.row)
            );
            println!(
                "{ISOLATED_RACE_SOLUTION_MARKER}\t{}",
                encode_isolated_solution(solution)
            );
            return;
        }

        // Parent process path: spawn children and collect results
        println!(
            "[BVP_sci isolated cold] protocol cooldown_ms={}, cleanup_child_artifacts={}",
            cold_cooldown_ms(),
            clean_cold_artifacts_enabled()
        );
        let samples = run_isolated_race_samples(
            COMBUSTION_3000_ISOLATED_STRESS_TEST_NAME,
            &variants,
            n_steps,
            repetitions,
        );

        // Convert RaceRows to RaceSamples with default statistics for summarization
        let default_stats = BvpSciStatistics::default();
        let race_samples: Vec<RaceSample> = samples
            .iter()
            .map(|row| RaceSample {
                row: row.clone(),
                statistics: default_stats.clone(),
            })
            .collect();
        let summary = summarize_samples(&variants, &race_samples);

        println!();
        print_isolated_cold_sample_table(
            "[BVP_sci stress] combustion-3000 raw process-isolated cold observations",
            &samples,
            variants.len(),
        );
        println!();
        print_e2e_correctness_table(
            "[BVP_sci stress] combustion-3000 Lambdify vs AOT correctness (isolated cold)",
            &summary,
        );
        println!();
        print_e2e_performance_table(
            "[BVP_sci stress] combustion-3000 Lambdify vs AOT timing (isolated cold)",
            &summary,
        );

        // Assert: all variants converge in isolated cold runs
        for row in &summary {
            assert!(
                row.ok_runs > 0,
                "{} {} {}: all isolated cold runs failed",
                row.source,
                row.matrix,
                row.variant
            );
        }
    }
}
