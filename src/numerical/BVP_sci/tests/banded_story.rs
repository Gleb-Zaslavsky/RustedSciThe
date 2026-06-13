#![cfg(test)]

mod tests {
    use crate::numerical::BVP_sci::BVP_sci_faer::BvpSciLinearSolvePolicy;
    use crate::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciSolverOptions, BvpSciStatistics};
    use crate::numerical::BVP_sci::test_common::{mean, stddev};
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::time::Instant;

    #[derive(Clone)]
    struct BandedVariant {
        matrix: &'static str,
        variant: &'static str,
        bootstrap_hint: &'static str,
        policy: BvpSciLinearSolvePolicy,
    }

    #[derive(Clone)]
    struct BandedRow {
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

    #[derive(Clone)]
    struct BandedSample {
        row: BandedRow,
        statistics: BvpSciStatistics,
        solution: Option<DMatrix<f64>>,
    }

    #[derive(Clone, Copy)]
    struct Aggregate {
        mean: f64,
        stddev: f64,
        min: f64,
        max: f64,
    }

    #[derive(Clone)]
    struct BandedSummaryRow {
        source: &'static str,
        matrix: &'static str,
        variant: &'static str,
        bootstrap_hint: &'static str,
        total_ms: Aggregate,
        symbolic_ms: Aggregate,
        residual_ms: Aggregate,
        jacobian_ms: Aggregate,
        linear_ms: Aggregate,
        grid_refine_ms: Aggregate,
        solve_diff: Aggregate,
        rel_x_diff: Aggregate,
        max_abs_solution: Aggregate,
        sparse_fallbacks: Aggregate,
        full_banded_solves: Aggregate,
        bordered_routes: Aggregate,
        dense_kib: Aggregate,
        sparse_kib: Aggregate,
        nnz: Aggregate,
        bordered_solves: Aggregate,
        ok_runs: usize,
        runs: usize,
        status: String,
    }

    fn aggregate(values: &[f64]) -> Aggregate {
        let mean = mean(values);
        let stddev = stddev(values, mean);
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Aggregate {
            mean,
            stddev,
            min,
            max,
        }
    }

    fn fmt_agg(agg: Aggregate) -> String {
        format!(
            "{:.3} +/- {:.3} [{:.3}, {:.3}]",
            agg.mean, agg.stddev, agg.min, agg.max
        )
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

    fn stats_count(stats: &BvpSciStatistics, key: &str) -> f64 {
        stats.counters.get(key).copied().unwrap_or(0) as f64
    }

    fn stats_usize(stats: &BvpSciStatistics, key: &str) -> f64 {
        stats
            .counters
            .get(key)
            .copied()
            .or_else(|| {
                stats
                    .diagnostics
                    .get(key)
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap_or(0) as f64
    }

    fn combustion_problem_options(n_steps: usize) -> BvpSciSolverOptions {
        let values = vec![
            "Teta".to_string(),
            "q".to_string(),
            "C0".to_string(),
            "J0".to_string(),
            "C1".to_string(),
            "J1".to_string(),
        ];
        let unknowns = Expr::parse_vector_expression(
            values
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
        );
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

        let initial_guess = DMatrix::from_element(values.len(), n_steps, 0.99);
        let x_mesh = DVector::from_fn(n_steps, |i, _| {
            i as f64 / (n_steps.saturating_sub(1)) as f64
        });

        BvpSciSolverOptions::new(
            Some(x_mesh),
            Some(0.0),
            Some(1.0),
            Some(n_steps),
            eqs,
            values,
            Vec::new(),
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            n_steps.saturating_mul(3),
            initial_guess,
        )
    }

    fn build_banded_solver(n_steps: usize, policy: BvpSciLinearSolvePolicy) -> BVPwrap {
        let options = combustion_problem_options(n_steps);
        let mut solver = BVPwrap::new_with_options(options).with_expr_legacy_smart_sparse();
        solver.set_linear_solve_policy(policy);
        solver.set_additional_parameters(Some(true), None, Some("off".to_string()));
        solver
    }

    fn combustion_banded_variants() -> Vec<BandedVariant> {
        vec![
            BandedVariant {
                matrix: "Sparse",
                variant: "ExprLegacy",
                bootstrap_hint: "sparse_baseline",
                policy: BvpSciLinearSolvePolicy::Sparse,
            },
            BandedVariant {
                matrix: "AutoBanded",
                variant: "ExprLegacy",
                bootstrap_hint: "auto_banded_policy",
                policy: BvpSciLinearSolvePolicy::AutoBanded,
            },
            BandedVariant {
                matrix: "ExperimentalBordered",
                variant: "ExprLegacy",
                bootstrap_hint: "experimental_bordered_policy",
                policy: BvpSciLinearSolvePolicy::ExperimentalBorderedBanded,
            },
        ]
    }

    fn run_variant(
        n_steps: usize,
        variant: &BandedVariant,
    ) -> (BandedSample, Option<DMatrix<f64>>) {
        let total_begin = Instant::now();
        let mut solver = build_banded_solver(n_steps, variant.policy);
        let solve_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solver.try_solve()));
        let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;

        let (status, solution) = match solve_result {
            Ok(Ok(())) => {
                let solution = solver.get_result();
                let status = if solution.is_some() {
                    "ok"
                } else {
                    "missing_result"
                };
                (status.to_string(), solution)
            }
            Ok(Err(err)) => (format!("solve_err:{err}"), None),
            Err(_) => ("panic".to_string(), None),
        };

        let stats = solver.get_statistics();
        if status == "ok" {
            assert!(
                stats.linear_system_ms_total <= total_ms + 1.0,
                "{}: linear timer {:.3} ms exceeds total wall clock {:.3} ms",
                variant.matrix,
                stats.linear_system_ms_total,
                total_ms
            );
        }
        let row = BandedRow {
            source: "BVP_sci",
            matrix: variant.matrix,
            variant: variant.variant,
            bootstrap_hint: variant.bootstrap_hint,
            total_ms,
            max_abs_solution: solution.as_ref().map(solution_max_abs).unwrap_or(f64::NAN),
            solve_diff: f64::NAN,
            rel_x_diff: f64::NAN,
            status,
        };

        (
            BandedSample {
                row,
                statistics: stats,
                solution,
            },
            None,
        )
    }

    fn run_samples(n_steps: usize, repetitions: usize) -> Vec<BandedSample> {
        let variants = combustion_banded_variants();
        let mut samples = Vec::with_capacity(variants.len() * repetitions);
        for repetition in 0..repetitions {
            let mut rep_samples = Vec::with_capacity(variants.len());
            let mut rep_solutions = Vec::with_capacity(variants.len());
            for variant in &variants {
                let (sample, _) = run_variant(n_steps, variant);
                rep_solutions.push(sample.solution.clone());
                rep_samples.push(sample);
            }
            if let Some(baseline) = rep_solutions.iter().find_map(|solution| solution.as_ref()) {
                for (sample, solution) in rep_samples.iter_mut().zip(rep_solutions.iter()) {
                    if let Some(solution) = solution {
                        sample.row.solve_diff = solution_linf_diff(solution, baseline);
                        sample.row.rel_x_diff = solution_rel_diff(solution, baseline);
                    }
                }
            }
            samples.extend(rep_samples);
            let _ = repetition;
        }
        samples
    }

    fn summarize_samples(samples: &[BandedSample]) -> Vec<BandedSummaryRow> {
        let variants = combustion_banded_variants();
        let mut rows = Vec::with_capacity(variants.len());
        for variant in variants {
            let subset: Vec<&BandedSample> = samples
                .iter()
                .filter(|sample| sample.row.matrix == variant.matrix)
                .collect();

            let total_ms: Vec<f64> = subset.iter().map(|sample| sample.row.total_ms).collect();
            let solve_diff: Vec<f64> = subset.iter().map(|sample| sample.row.solve_diff).collect();
            let rel_x_diff: Vec<f64> = subset.iter().map(|sample| sample.row.rel_x_diff).collect();
            let max_abs_solution: Vec<f64> = subset
                .iter()
                .map(|sample| sample.row.max_abs_solution)
                .collect();
            let symbolic_ms: Vec<f64> = subset
                .iter()
                .map(|sample| sample.statistics.symbolic_prepare_ms_total)
                .collect();
            let residual_ms: Vec<f64> = subset
                .iter()
                .map(|sample| sample.statistics.residual_ms_total)
                .collect();
            let jacobian_ms: Vec<f64> = subset
                .iter()
                .map(|sample| sample.statistics.jacobian_ms_total)
                .collect();
            let linear_ms: Vec<f64> = subset
                .iter()
                .map(|sample| sample.statistics.linear_system_ms_total)
                .collect();
            let grid_refine_ms: Vec<f64> = subset
                .iter()
                .map(|sample| sample.statistics.grid_refinement_ms_total)
                .collect();
            let sparse_fb: Vec<f64> = subset
                .iter()
                .map(|sample| {
                    stats_count(
                        &sample.statistics,
                        "bvp sci linear backend sparse fallback solves",
                    )
                })
                .collect();
            let full_banded_solves: Vec<f64> = subset
                .iter()
                .map(|sample| {
                    stats_count(
                        &sample.statistics,
                        "bvp sci linear backend full banded solves",
                    )
                })
                .collect();
            let bordered_routes: Vec<f64> = subset
                .iter()
                .map(|sample| {
                    stats_count(
                        &sample.statistics,
                        "bvp sci route bordered banded candidate",
                    )
                })
                .collect();
            let bordered_solves: Vec<f64> = subset
                .iter()
                .map(|sample| {
                    stats_count(
                        &sample.statistics,
                        "bvp sci linear backend bordered structured solves",
                    )
                })
                .collect();
            let dense_kib: Vec<f64> = subset
                .iter()
                .map(|sample| {
                    stats_usize(&sample.statistics, "global jacobian dense equivalent kib")
                })
                .collect();
            let sparse_kib: Vec<f64> = subset
                .iter()
                .map(|sample| stats_usize(&sample.statistics, "global jacobian sparse storage kib"))
                .collect();
            let nnz: Vec<f64> = subset
                .iter()
                .map(|sample| stats_usize(&sample.statistics, "global jacobian nnz"))
                .collect();
            let ok_runs = subset
                .iter()
                .filter(|sample| sample.row.status == "ok")
                .count();
            let status = if ok_runs == subset.len() {
                format!("ok {}/{}", ok_runs, subset.len())
            } else {
                let first_failure = subset
                    .iter()
                    .find(|sample| sample.row.status != "ok")
                    .map(|sample| sample.row.status.as_str())
                    .unwrap_or("unknown_failure");
                format!(
                    "fail {}/{} first_failure={first_failure}",
                    ok_runs,
                    subset.len()
                )
            };

            rows.push(BandedSummaryRow {
                source: "BVP_sci",
                matrix: variant.matrix,
                variant: variant.variant,
                bootstrap_hint: variant.bootstrap_hint,
                total_ms: aggregate(&total_ms),
                symbolic_ms: aggregate(&symbolic_ms),
                residual_ms: aggregate(&residual_ms),
                jacobian_ms: aggregate(&jacobian_ms),
                linear_ms: aggregate(&linear_ms),
                grid_refine_ms: aggregate(&grid_refine_ms),
                solve_diff: aggregate(&solve_diff),
                rel_x_diff: aggregate(&rel_x_diff),
                max_abs_solution: aggregate(&max_abs_solution),
                sparse_fallbacks: aggregate(&sparse_fb),
                full_banded_solves: aggregate(&full_banded_solves),
                bordered_routes: aggregate(&bordered_routes),
                dense_kib: aggregate(&dense_kib),
                sparse_kib: aggregate(&sparse_kib),
                nnz: aggregate(&nnz),
                bordered_solves: aggregate(&bordered_solves),
                ok_runs,
                runs: subset.len(),
                status,
            });
        }
        rows
    }

    fn print_correctness_table(title: &str, rows: &[BandedSummaryRow]) {
        println!("{title}");
        println!(
            "source   | matrix                | variant    | bootstrap_hint            | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status"
        );
        println!("{}", "-".repeat(150));
        for row in rows {
            println!(
                "{:<8} | {:<21} | {:<10} | {:<25} | {:>6} | {:<21} | {:<21} | {:<23} | {}",
                row.source,
                row.matrix,
                row.variant,
                row.bootstrap_hint,
                format!("{}/{}", row.ok_runs, row.runs),
                fmt_agg(row.solve_diff),
                fmt_agg(row.rel_x_diff),
                fmt_agg(row.max_abs_solution),
                row.status
            );
        }
        println!();
    }

    fn print_timing_table(title: &str, rows: &[BandedSummaryRow]) {
        println!("{title}");
        println!(
            "[BVP_sci banded] timing table: all time columns are milliseconds; counters are counts."
        );
        println!(
            "source   | matrix                | variant    | bootstrap_hint            | total_ms mean+/-std [min,max] | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms"
        );
        println!("{}", "-".repeat(190));
        for row in rows {
            println!(
                "{:<8} | {:<21} | {:<10} | {:<25} | {:<32} | {:<11} | {:<11} | {:<11} | {:<9} | {:<14}",
                row.source,
                row.matrix,
                row.variant,
                row.bootstrap_hint,
                fmt_agg(row.total_ms),
                fmt_agg(row.symbolic_ms),
                fmt_agg(row.residual_ms),
                fmt_agg(row.jacobian_ms),
                fmt_agg(row.linear_ms),
                fmt_agg(row.grid_refine_ms),
            );
        }
        println!();
    }

    fn print_route_table(title: &str, rows: &[BandedSummaryRow]) {
        println!("{title}");
        println!(
            "[BVP_sci banded] route table: route counters and Jacobian footprint are accumulated solver diagnostics."
        );
        println!(
            "source   | matrix                | variant    | sparse_fb | full_banded | bordered_candidate | bordered_solves | dense_kib | sparse_kib | nnz    | status"
        );
        println!("{}", "-".repeat(160));
        for row in rows {
            println!(
                "{:<8} | {:<21} | {:<10} | {:>9} | {:>11} | {:>18} | {:>15} | {:>9} | {:>10} | {:>6} | {}",
                row.source,
                row.matrix,
                row.variant,
                row.sparse_fallbacks.mean.round() as usize,
                row.full_banded_solves.mean.round() as usize,
                row.bordered_routes.mean.round() as usize,
                row.bordered_solves.mean.round() as usize,
                row.dense_kib.mean.round() as usize,
                row.sparse_kib.mean.round() as usize,
                row.nnz.mean.round() as usize,
                row.status
            );
        }
        println!();
    }

    fn assert_contract(rows: &[BandedSummaryRow]) {
        for row in rows {
            assert_eq!(
                row.ok_runs, row.runs,
                "{} {}: expected every banded story run to pass, got {}",
                row.matrix, row.variant, row.status
            );
        }

        let sparse = rows
            .iter()
            .find(|row| row.matrix == "Sparse")
            .expect("banded story must include Sparse baseline");
        assert_eq!(
            sparse.sparse_fallbacks.mean.round() as usize,
            0,
            "Sparse baseline should use direct Sparse solves, not AutoBanded fallback"
        );

        let auto = rows
            .iter()
            .find(|row| row.matrix == "AutoBanded")
            .expect("banded story must include AutoBanded");
        assert!(
            auto.bordered_routes.mean > 0.0,
            "AutoBanded must detect the endpoint-BC bordered-banded candidate"
        );
        assert_eq!(
            auto.full_banded_solves.mean.round() as usize,
            0,
            "AutoBanded must not force full scalar banded on endpoint-BC matrices"
        );
        assert!(
            auto.bordered_solves.mean > 0.0,
            "AutoBanded should promote parameter-free endpoint-BC matrices to the bordered solver"
        );
        assert_eq!(
            auto.sparse_fallbacks.mean.round() as usize,
            0,
            "successful AutoBanded bordered solves should not fall back to Sparse"
        );

        let bordered = rows
            .iter()
            .find(|row| row.matrix == "ExperimentalBordered")
            .expect("banded story must include ExperimentalBordered");
        assert!(
            bordered.bordered_routes.mean > 0.0,
            "ExperimentalBordered must detect the bordered-banded route"
        );
        assert!(
            bordered.bordered_solves.mean > 0.0,
            "ExperimentalBordered must use the structured bordered solver"
        );
        assert_eq!(
            bordered.sparse_fallbacks.mean.round() as usize,
            0,
            "ExperimentalBordered must not silently fall back to Sparse"
        );
    }

    #[test]
    #[ignore]
    fn combustion_1000_banded_production_story_split() {
        let samples = run_samples(1_000, 5);
        let summary = summarize_samples(&samples);
        print_correctness_table(
            "Combustion 1000: Sparse vs promoted AutoBanded vs strict bordered correctness (5 runs)",
            &summary,
        );
        print_timing_table(
            "Combustion 1000: Sparse vs promoted AutoBanded vs strict bordered timing (5 runs)",
            &summary,
        );
        print_route_table(
            "Combustion 1000: Sparse vs promoted AutoBanded vs strict bordered route / memory (5 runs)",
            &summary,
        );
        assert_contract(&summary);
    }

    #[test]
    #[ignore]
    fn combustion_3000_banded_production_story_split() {
        let samples = run_samples(3_000, 3);
        let summary = summarize_samples(&samples);
        print_correctness_table(
            "Combustion 3000: Sparse vs promoted AutoBanded vs strict bordered correctness (3 runs)",
            &summary,
        );
        print_timing_table(
            "Combustion 3000: Sparse vs promoted AutoBanded vs strict bordered timing (3 runs)",
            &summary,
        );
        print_route_table(
            "Combustion 3000: Sparse vs promoted AutoBanded vs strict bordered route / memory (3 runs)",
            &summary,
        );
        assert_contract(&summary);
    }

    #[test]
    #[ignore]
    fn combustion_10000_banded_stress_story_split() {
        let samples = run_samples(10_000, 2);
        let summary = summarize_samples(&samples);
        print_correctness_table(
            "Combustion 10000: Sparse vs promoted AutoBanded vs strict bordered correctness (2 runs)",
            &summary,
        );
        print_timing_table(
            "Combustion 10000: Sparse vs promoted AutoBanded vs strict bordered timing (2 runs)",
            &summary,
        );
        print_route_table(
            "Combustion 10000: Sparse vs promoted AutoBanded vs strict bordered route / memory (2 runs)",
            &summary,
        );
        assert_contract(&summary);
    }
}
