#![cfg(test)]

mod tests {
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedBvpStatistics, DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
    };
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::time::Instant;

    const RACE_REPETITIONS: usize = 5;

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
        total_timer_ms: f64,
        symbolic_timer_ms: f64,
        linear_timer_ms: f64,
        jac_timer_ms: f64,
        fun_timer_ms: f64,
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
        total_timer_ms: Aggregate,
        symbolic_timer_ms: Aggregate,
        linear_timer_ms: Aggregate,
        jac_timer_ms: Aggregate,
        fun_timer_ms: Aggregate,
        status: String,
    }

    fn uniform_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
        DMatrix::from_column_slice(
            variable_count,
            n_steps,
            DVector::from_element(variable_count * n_steps, value).as_slice(),
        )
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
                            linear_solves: stats_count(
                                &statistics,
                                "number of solving linear systems",
                            ),
                            jac_rebuilds: stats_count(
                                &statistics,
                                "number of jacobians recalculations",
                            ),
                            total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                            symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                            linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                            jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                            fun_timer_ms: stats_timer_ms(&statistics, "Function"),
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
                        jac_rebuilds: stats_count(
                            &statistics,
                            "number of jacobians recalculations",
                        ),
                        total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                        symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                        linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                        jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                        fun_timer_ms: stats_timer_ms(&statistics, "Function"),
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
                    total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                    symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                    linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                    jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                    fun_timer_ms: stats_timer_ms(&statistics, "Function"),
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
                        jac_rebuilds: stats_count(
                            &statistics,
                            "number of jacobians recalculations",
                        ),
                        total_timer_ms: stats_timer_ms(&statistics, "time elapsed"),
                        symbolic_timer_ms: stats_timer_ms(&statistics, "Symbolic Operations"),
                        linear_timer_ms: stats_timer_ms(&statistics, "Linear System"),
                        jac_timer_ms: stats_timer_ms(&statistics, "Jacobian"),
                        fun_timer_ms: stats_timer_ms(&statistics, "Function"),
                        status,
                    },
                    None,
                )
            }
        }
    }

    fn run_race_samples(
        variants: &[RaceVariant],
        n_steps: usize,
        repetitions: usize,
    ) -> Vec<RaceRow> {
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
                let (row, solution) = run_race_variant(
                    n_steps,
                    variant.source,
                    variant.matrix,
                    variant.variant,
                    variant.bootstrap_hint,
                    variant.config.clone(),
                );
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
            total_timer_ms: aggregate(rows.iter().map(|row| row.total_timer_ms)),
            symbolic_timer_ms: aggregate(rows.iter().map(|row| row.symbolic_timer_ms)),
            linear_timer_ms: aggregate(rows.iter().map(|row| row.linear_timer_ms)),
            jac_timer_ms: aggregate(rows.iter().map(|row| row.jac_timer_ms)),
            fun_timer_ms: aggregate(rows.iter().map(|row| row.fun_timer_ms)),
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

    fn rebuild_release(config: GeneratedBackendConfig) -> GeneratedBackendConfig {
        config
            .with_aot_compile_dev_fastest()
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release,
            })
    }

    #[test]
    #[ignore = "heavy combustion-1000 end-to-end Lambdify Sparse vs Banded race table"]
    fn combustion_1000_lambdify_sparse_vs_banded_end_to_end_race() {
        let n_steps = 1000usize;
        let sparse_config = GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
        let banded_config = GeneratedBackendConfig::banded_lambdify_defaults()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

        let variants = [
            RaceVariant {
                source: "Lambdify",
                matrix: "Sparse",
                variant: "ExprLegacy",
                bootstrap_hint: "symbolic+lambdify",
                config: sparse_config,
            },
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "ExprLegacy",
                bootstrap_hint: "symbolic+lambdify",
                config: banded_config,
            },
        ];

        let samples = run_race_samples(&variants, n_steps, RACE_REPETITIONS);
        let rows = summarize_samples(&variants, &samples);
        print_race_summary_table(
            "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded end-to-end",
            &rows,
        );

        assert!(
            rows.iter().all(|row| row.ok_runs == row.runs),
            "both Lambdify race variants should solve successfully"
        );
    }

    #[test]
    #[ignore = "heavy combustion-1000 end-to-end AOT Sparse vs Banded race table"]
    fn combustion_1000_aot_sparse_vs_banded_end_to_end_race() {
        let n_steps = 1000usize;
        let variants = [
            RaceVariant {
                source: "Compiled",
                matrix: "Sparse",
                variant: "C-gcc",
                bootstrap_hint: "symbolic+aot-build+link",
                config: rebuild_release(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                ),
            },
            RaceVariant {
                source: "Compiled",
                matrix: "Banded",
                variant: "C-gcc",
                bootstrap_hint: "symbolic+aot-build+link",
                config: rebuild_release(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                ),
            },
            RaceVariant {
                source: "Compiled",
                matrix: "Sparse",
                variant: "C-tcc",
                bootstrap_hint: "symbolic+aot-build+link",
                config: rebuild_release(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                ),
            },
            RaceVariant {
                source: "Compiled",
                matrix: "Banded",
                variant: "C-tcc",
                bootstrap_hint: "symbolic+aot-build+link",
                config: rebuild_release(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                ),
            },
            RaceVariant {
                source: "Compiled",
                matrix: "Sparse",
                variant: "Zig",
                bootstrap_hint: "symbolic+aot-build+link",
                config: rebuild_release(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                ),
            },
            RaceVariant {
                source: "Compiled",
                matrix: "Banded",
                variant: "Zig",
                bootstrap_hint: "symbolic+aot-build+link",
                config: rebuild_release(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                ),
            },
        ];

        let samples = run_race_samples(&variants, n_steps, RACE_REPETITIONS);
        let rows = summarize_samples(&variants, &samples);
        print_race_summary_table(
            "[BVP Damp race] combustion-1000 AOT Sparse vs Banded end-to-end",
            &rows,
        );

        assert!(
            rows.iter().any(|row| row.ok_runs > 0),
            "at least one AOT race variant should solve successfully"
        );
    }
}
