#![cfg(test)]

mod tests {
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedBvpStatistics, DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy,
        GeneratedBackendConfig,
    };
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::codegen::codegen_orchestrator::{
        ParallelExecutorConfig, ParallelFallbackPolicy,
    };
    use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::fs;
    use std::io::{self, Write};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::PathBuf;
    use std::process::Command;
    use std::thread;
    use std::time::{Duration, Instant};

    const RACE_REPETITIONS: usize = 5;
    const ISOLATED_STRESS_CHILD_INDEX_ENV: &str = "BVP_DAMP_ISOLATED_STRESS_CHILD_INDEX";
    const ISOLATED_STRESS_CHILD_REPETITION_ENV: &str = "BVP_DAMP_ISOLATED_STRESS_CHILD_REPETITION";
    const ISOLATED_RACE_ROW_MARKER: &str = "[BVP_DAMP_ISOLATED_RACE_ROW]";
    const ISOLATED_RACE_SOLUTION_MARKER: &str = "[BVP_DAMP_ISOLATED_RACE_SOLUTION]";
    const ISOLATED_RACE_PID_MARKER: &str = "[BVP_DAMP_ISOLATED_RACE_PID]";
    const COLD_COOLDOWN_MS_ENV: &str = "BVP_AOT_COLD_COOLDOWN_MS";
    const COLD_CLEAN_ARTIFACTS_ENV: &str = "BVP_AOT_COLD_CLEAN_ARTIFACTS";
    const WARM_COOLDOWN_MS_ENV: &str = "BVP_AOT_WARM_COOLDOWN_MS";

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
                            linear_solves: stats_count(
                                &statistics,
                                "number of solving linear systems",
                            ),
                            jac_rebuilds: stats_count(
                                &statistics,
                                "number of jacobians recalculations",
                            ),
                            grid_refinements: stats_count(
                                &statistics,
                                "number of grid refinements",
                            ),
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
                            aot_build_policy: stats_diagnostic_string(
                                &statistics,
                                "aot.build_policy",
                            ),
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
            residual_fallback_reason: parse_isolated_string(
                &mut fields,
                "residual_fallback_reason",
            ),
            sparse_jacobian_fallback_reason: parse_isolated_string(
                &mut fields,
                "sparse_jacobian_fallback_reason",
            ),
            selected_backend: parse_isolated_string(&mut fields, "selected_backend"),
            symbolic_assembly_backend: parse_isolated_string(
                &mut fields,
                "symbolic_assembly_backend",
            ),
            aot_build_policy: parse_isolated_string(&mut fields, "aot_build_policy"),
            initial_generate_ms: parse_isolated_field(&mut fields, "initial_generate_ms"),
            initial_discretization_ms: parse_isolated_field(
                &mut fields,
                "initial_discretization_ms",
            ),
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
            initial_sparse_prepare_ms: parse_isolated_field(
                &mut fields,
                "initial_sparse_prepare_ms",
            ),
            initial_runtime_binding_ms: parse_isolated_field(
                &mut fields,
                "initial_runtime_binding_ms",
            ),
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

    fn cold_cooldown_ms() -> u64 {
        std::env::var(COLD_COOLDOWN_MS_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    }

    fn warm_cooldown_ms() -> u64 {
        std::env::var(WARM_COOLDOWN_MS_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(5_000)
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

    fn run_isolated_race_samples(
        test_name: &str,
        variants: &[RaceVariant],
        repetitions: usize,
    ) -> Vec<RaceRow> {
        let executable = std::env::current_exe().expect("current test executable should resolve");
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
            selected_backend: summarize_reason(
                rows.iter().map(|row| row.selected_backend.as_str()),
            ),
            symbolic_assembly_backend: summarize_reason(
                rows.iter()
                    .map(|row| row.symbolic_assembly_backend.as_str()),
            ),
            aot_build_policy: summarize_reason(
                rows.iter().map(|row| row.aot_build_policy.as_str()),
            ),
            initial_generate_ms: aggregate(rows.iter().map(|row| row.initial_generate_ms)),
            initial_discretization_ms: aggregate(
                rows.iter().map(|row| row.initial_discretization_ms),
            ),
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
            initial_sparse_prepare_ms: aggregate(
                rows.iter().map(|row| row.initial_sparse_prepare_ms),
            ),
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

    fn print_isolated_cold_sample_table(
        title: &str,
        samples: &[RaceRow],
        rows_per_repetition: usize,
    ) {
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
        println!(
            "source   | matrix | variant    | initial_bind | jacobian_compile | residual_compile"
        );
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

    fn whole_chunking() -> AotChunkingPolicy {
        AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::Whole),
            Some(SparseChunkingStrategy::Whole),
        )
    }

    fn four_way_chunking() -> AotChunkingPolicy {
        AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
        )
    }

    fn forced_parallel_execution() -> AotExecutionPolicy {
        AotExecutionPolicy::Parallel(ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: Some(4),
            max_sparse_jobs: Some(4),
            fallback_policy: ParallelFallbackPolicy::Never,
        })
    }

    fn combustion_toolchain_chunking_variants() -> Vec<RaceVariant> {
        vec![
            RaceVariant {
                source: "Lambdify",
                matrix: "Sparse",
                variant: "AtomView",
                bootstrap_hint: "baseline",
                config: sparse_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "AtomView",
                bootstrap_hint: "baseline",
                config: banded_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "gcc",
                bootstrap_hint: "whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "gcc",
                bootstrap_hint: "chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "tcc",
                bootstrap_hint: "whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "tcc",
                bootstrap_hint: "chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "zig",
                bootstrap_hint: "whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "zig",
                bootstrap_hint: "chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "gcc",
                bootstrap_hint: "whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "gcc",
                bootstrap_hint: "chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc",
                bootstrap_hint: "whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc",
                bootstrap_hint: "chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "zig",
                bootstrap_hint: "whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "zig",
                bootstrap_hint: "chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
        ]
    }

    fn sparse_atomview_rust_aot_release() -> GeneratedBackendConfig {
        GeneratedBackendConfig::sparse_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::Rust)
    }

    fn banded_atomview_rust_aot_release() -> GeneratedBackendConfig {
        GeneratedBackendConfig::banded_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::Rust)
    }

    fn sparse_atomview_lambdify_baseline() -> GeneratedBackendConfig {
        GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
    }

    fn banded_atomview_lambdify_baseline() -> GeneratedBackendConfig {
        GeneratedBackendConfig::banded_lambdify_defaults()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
    }

    fn callback_probe_for_initial_guess(
        n_steps: usize,
        label: &str,
        vector_method: &str,
        config: GeneratedBackendConfig,
    ) -> CallbackProbe {
        println!("[BVP Damp debug] bootstrapping {label}");
        let _ = io::stdout().flush();

        let probe_start = Instant::now();
        let bootstrap_start = Instant::now();
        let mut solver = make_combustion_solver(n_steps, config);
        solver
            .try_eq_generate(None, None)
            .unwrap_or_else(|err| panic!("{label}: generated backend bootstrap failed: {err:?}"));
        let bootstrap_ms = bootstrap_start.elapsed().as_secs_f64() * 1_000.0;

        let args = DVector::from_element(solver.values.len() * n_steps, 0.99);
        let typed_args = crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting(
            &args,
            vector_method.to_string(),
        );
        let residual_start = Instant::now();
        let residual = solver.fun.call(0.0, &*typed_args).to_DVectorType();
        let residual_ms = residual_start.elapsed().as_secs_f64() * 1_000.0;
        let jacobian_start = Instant::now();
        let jacobian = solver
            .jac
            .as_mut()
            .unwrap_or_else(|| panic!("{label}: generated backend did not provide a Jacobian"))
            .call(0.0, &*typed_args)
            .to_DMatrixType();
        let jacobian_ms = jacobian_start.elapsed().as_secs_f64() * 1_000.0;
        let total_probe_ms = probe_start.elapsed().as_secs_f64() * 1_000.0;

        assert!(
            residual.iter().all(|value| value.is_finite()),
            "{label}: residual callback produced non-finite values"
        );
        assert!(
            jacobian.iter().all(|value| value.is_finite()),
            "{label}: Jacobian callback produced non-finite values"
        );

        println!(
            "[BVP Damp debug] {label}: residual_len={}, jacobian={}x{}",
            residual.len(),
            jacobian.nrows(),
            jacobian.ncols()
        );
        CallbackProbe {
            residual: residual.iter().copied().collect(),
            jacobian,
            total_probe_ms,
            bootstrap_ms,
            residual_ms,
            jacobian_ms,
        }
    }

    fn max_slice_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "callback outputs must have identical lengths"
        );
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    #[derive(Clone)]
    struct CallbackEquivalenceVariant {
        matrix: &'static str,
        toolchain: &'static str,
        whole_config: GeneratedBackendConfig,
        chunk4_config: GeneratedBackendConfig,
    }

    struct CallbackEquivalenceRow {
        matrix: &'static str,
        toolchain: &'static str,
        comparison: &'static str,
        residual_diff: f64,
        jacobian_diff: f64,
        status: String,
    }

    struct CallbackProbe {
        residual: Vec<f64>,
        jacobian: DMatrix<f64>,
        total_probe_ms: f64,
        bootstrap_ms: f64,
        residual_ms: f64,
        jacobian_ms: f64,
    }

    struct CallbackProbeStatsRow {
        matrix: &'static str,
        toolchain: &'static str,
        mode: &'static str,
        total_probe_ms: f64,
        bootstrap_ms: f64,
        residual_ms: f64,
        jacobian_ms: f64,
        residual_calls: usize,
        jacobian_calls: usize,
        residual_len: usize,
        jac_rows: usize,
        jac_cols: usize,
        status: String,
    }

    fn callback_equivalence_filter_matches(
        variant: &CallbackEquivalenceVariant,
        filter: &str,
    ) -> bool {
        let filter = filter.trim().to_ascii_lowercase();
        if filter.is_empty() {
            return true;
        }
        format!("{} {}", variant.matrix, variant.toolchain)
            .to_ascii_lowercase()
            .contains(&filter)
    }

    fn apply_optional_callback_equivalence_filter(
        variants: &[CallbackEquivalenceVariant],
    ) -> Vec<CallbackEquivalenceVariant> {
        let Ok(filter) = std::env::var("BVP_AOT_CALLBACK_FILTER") else {
            return variants.to_vec();
        };
        let selected = variants
            .iter()
            .filter(|variant| callback_equivalence_filter_matches(variant, &filter))
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            !selected.is_empty(),
            "BVP_AOT_CALLBACK_FILTER={filter:?} did not match any callback-equivalence variant"
        );
        println!(
            "[BVP Damp debug] BVP_AOT_CALLBACK_FILTER={filter:?}: running {}/{} variants",
            selected.len(),
            variants.len()
        );
        selected
    }

    fn callback_equivalence_variants() -> Vec<CallbackEquivalenceVariant> {
        vec![
            CallbackEquivalenceVariant {
                matrix: "Sparse",
                toolchain: "gcc",
                whole_config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            CallbackEquivalenceVariant {
                matrix: "Sparse",
                toolchain: "tcc",
                whole_config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            CallbackEquivalenceVariant {
                matrix: "Sparse",
                toolchain: "zig",
                whole_config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            CallbackEquivalenceVariant {
                matrix: "Sparse",
                toolchain: "rust",
                whole_config: release_matrix_config(
                    sparse_atomview_rust_aot_release(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    sparse_atomview_rust_aot_release(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            CallbackEquivalenceVariant {
                matrix: "Banded",
                toolchain: "gcc",
                whole_config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            CallbackEquivalenceVariant {
                matrix: "Banded",
                toolchain: "tcc",
                whole_config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            CallbackEquivalenceVariant {
                matrix: "Banded",
                toolchain: "zig",
                whole_config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            CallbackEquivalenceVariant {
                matrix: "Banded",
                toolchain: "rust",
                whole_config: release_matrix_config(
                    banded_atomview_rust_aot_release(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                chunk4_config: release_matrix_config(
                    banded_atomview_rust_aot_release(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
        ]
    }

    fn callback_lambdify_baseline_config(matrix: &str) -> GeneratedBackendConfig {
        match matrix {
            "Sparse" => sparse_atomview_lambdify_baseline(),
            "Banded" => banded_atomview_lambdify_baseline(),
            other => panic!("unsupported callback-equivalence matrix {other}"),
        }
    }

    fn callback_vector_method(matrix: &str) -> &'static str {
        match matrix {
            "Sparse" => "Sparse",
            "Banded" => "Banded",
            other => panic!("unsupported callback-equivalence matrix {other}"),
        }
    }

    fn callback_diff_status(residual_diff: f64, jacobian_diff: f64) -> String {
        if residual_diff < 1.0e-10 && jacobian_diff < 1.0e-10 {
            "ok".to_string()
        } else {
            "diff_exceeded".to_string()
        }
    }

    fn callback_diff_row(
        matrix: &'static str,
        toolchain: &'static str,
        comparison: &'static str,
        lhs: &CallbackProbe,
        rhs: &CallbackProbe,
    ) -> CallbackEquivalenceRow {
        let residual_diff = max_slice_abs_diff(lhs.residual.as_slice(), rhs.residual.as_slice());
        let jacobian_diff = max_slice_abs_diff(lhs.jacobian.as_slice(), rhs.jacobian.as_slice());
        CallbackEquivalenceRow {
            matrix,
            toolchain,
            comparison,
            residual_diff,
            jacobian_diff,
            status: callback_diff_status(residual_diff, jacobian_diff),
        }
    }

    fn callback_probe_stats_row(
        matrix: &'static str,
        toolchain: &'static str,
        mode: &'static str,
        probe: &CallbackProbe,
        status: String,
    ) -> CallbackProbeStatsRow {
        CallbackProbeStatsRow {
            matrix,
            toolchain,
            mode,
            total_probe_ms: probe.total_probe_ms,
            bootstrap_ms: probe.bootstrap_ms,
            residual_ms: probe.residual_ms,
            jacobian_ms: probe.jacobian_ms,
            residual_calls: 1,
            jacobian_calls: 1,
            residual_len: probe.residual.len(),
            jac_rows: probe.jacobian.nrows(),
            jac_cols: probe.jacobian.ncols(),
            status,
        }
    }

    fn print_callback_equivalence_table(rows: &[CallbackEquivalenceRow]) {
        println!("[BVP Damp debug] AtomView AOT whole-vs-chunk4 callback correctness matrix");
        println!(
            "matrix | toolchain | comparison           | residual_diff | jacobian_diff | status"
        );
        println!("{}", "-".repeat(108));
        for row in rows {
            println!(
                "{:<6} | {:<9} | {:<20} | {:>13.6e} | {:>13.6e} | {}",
                row.matrix,
                row.toolchain,
                row.comparison,
                row.residual_diff,
                row.jacobian_diff,
                row.status
            );
        }
    }

    fn print_callback_probe_stats_table(rows: &[CallbackProbeStatsRow]) {
        println!();
        println!(
            "[BVP Damp debug] AtomView AOT callback probe statistics; all time columns are milliseconds"
        );
        println!(
            "matrix | toolchain | mode     | total_probe_ms | prepare_ms | residual_ms | jacobian_ms | res_calls | jac_calls | residual_len | jacobian_shape | status"
        );
        println!("{}", "-".repeat(174));
        for row in rows {
            println!(
                "{:<6} | {:<9} | {:<8} | {:>14.3} | {:>10.3} | {:>11.3} | {:>11.3} | {:>9} | {:>9} | {:>12} | {:>6}x{:<6} | {}",
                row.matrix,
                row.toolchain,
                row.mode,
                row.total_probe_ms,
                row.bootstrap_ms,
                row.residual_ms,
                row.jacobian_ms,
                row.residual_calls,
                row.jacobian_calls,
                row.residual_len,
                row.jac_rows,
                row.jac_cols,
                row.status
            );
        }
    }

    /// Debug-only gate for localizing sparse AOT chunking failures.
    ///
    /// This is intentionally not a story/performance test. It does not solve
    /// the BVP and does not report timing. Its only contract is mathematical:
    /// a chunked generated sparse callback must produce exactly the same
    /// residual and Jacobian values as the whole generated callback on the same
    /// state vector before Newton starts.
    #[test]
    #[ignore = "debug-only AOT callback equivalence gate; builds combustion-1000 artifacts"]
    fn debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000() {
        let n_steps = 1000usize;
        let variants = apply_optional_callback_equivalence_filter(&callback_equivalence_variants());
        let mut rows = Vec::with_capacity(variants.len() * 3);
        let mut stats_rows = Vec::with_capacity(variants.len() * 3);
        let mut lambdify_baselines = HashMap::<&'static str, CallbackProbe>::new();

        for variant in variants {
            if !lambdify_baselines.contains_key(variant.matrix) {
                let baseline_label = format!("{} Lambdify baseline", variant.matrix);
                let baseline = callback_probe_for_initial_guess(
                    n_steps,
                    baseline_label.as_str(),
                    callback_vector_method(variant.matrix),
                    callback_lambdify_baseline_config(variant.matrix),
                );
                stats_rows.push(callback_probe_stats_row(
                    variant.matrix,
                    "baseline",
                    "lambdify",
                    &baseline,
                    "ok".to_string(),
                ));
                lambdify_baselines.insert(variant.matrix, baseline);
            }

            let whole_label = format!("{} {} whole", variant.matrix, variant.toolchain);
            let chunk4_label = format!("{} {} chunk4", variant.matrix, variant.toolchain);
            let result = catch_unwind(AssertUnwindSafe(|| {
                let whole_probe = callback_probe_for_initial_guess(
                    n_steps,
                    whole_label.as_str(),
                    callback_vector_method(variant.matrix),
                    variant.whole_config.clone(),
                );
                let chunk_probe = callback_probe_for_initial_guess(
                    n_steps,
                    chunk4_label.as_str(),
                    callback_vector_method(variant.matrix),
                    variant.chunk4_config.clone(),
                );
                (whole_probe, chunk_probe)
            }));

            match result {
                Ok((whole_probe, chunk_probe)) => {
                    let baseline_probe = lambdify_baselines
                        .get(variant.matrix)
                        .expect("Lambdify baseline must be prepared before AOT probes");
                    let baseline_vs_whole = callback_diff_row(
                        variant.matrix,
                        variant.toolchain,
                        "lambdify-vs-whole",
                        baseline_probe,
                        &whole_probe,
                    );
                    let whole_vs_chunk4 = callback_diff_row(
                        variant.matrix,
                        variant.toolchain,
                        "whole-vs-chunk4",
                        &whole_probe,
                        &chunk_probe,
                    );
                    let baseline_vs_chunk4 = callback_diff_row(
                        variant.matrix,
                        variant.toolchain,
                        "lambdify-vs-chunk4",
                        baseline_probe,
                        &chunk_probe,
                    );
                    let status = if [
                        baseline_vs_whole.status.as_str(),
                        whole_vs_chunk4.status.as_str(),
                        baseline_vs_chunk4.status.as_str(),
                    ]
                    .iter()
                    .all(|status| *status == "ok")
                    {
                        "ok".to_string()
                    } else {
                        "diff_exceeded".to_string()
                    };
                    stats_rows.push(callback_probe_stats_row(
                        variant.matrix,
                        variant.toolchain,
                        "whole",
                        &whole_probe,
                        status.clone(),
                    ));
                    stats_rows.push(callback_probe_stats_row(
                        variant.matrix,
                        variant.toolchain,
                        "chunk4",
                        &chunk_probe,
                        status.clone(),
                    ));
                    rows.push(baseline_vs_whole);
                    rows.push(whole_vs_chunk4);
                    rows.push(baseline_vs_chunk4);
                }
                Err(panic_payload) => {
                    let status = if let Some(message) = panic_payload.downcast_ref::<String>() {
                        format!("panicked({message})")
                    } else if let Some(message) = panic_payload.downcast_ref::<&str>() {
                        format!("panicked({message})")
                    } else {
                        "panicked(non-string payload)".to_string()
                    };
                    rows.push(CallbackEquivalenceRow {
                        matrix: variant.matrix,
                        toolchain: variant.toolchain,
                        comparison: "callback-probe",
                        residual_diff: f64::NAN,
                        jacobian_diff: f64::NAN,
                        status: status.clone(),
                    });
                    for mode in ["whole", "chunk4"] {
                        stats_rows.push(CallbackProbeStatsRow {
                            matrix: variant.matrix,
                            toolchain: variant.toolchain,
                            mode,
                            total_probe_ms: f64::NAN,
                            bootstrap_ms: f64::NAN,
                            residual_ms: f64::NAN,
                            jacobian_ms: f64::NAN,
                            residual_calls: 0,
                            jacobian_calls: 0,
                            residual_len: 0,
                            jac_rows: 0,
                            jac_cols: 0,
                            status: status.clone(),
                        });
                    }
                }
            }
        }

        print_callback_equivalence_table(&rows);
        print_callback_probe_stats_table(&stats_rows);

        assert!(
            rows.iter().all(|row| row.status == "ok"),
            "all AtomView AOT chunk4 callbacks must match whole callbacks before Newton"
        );
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
        println!();
        print_e2e_callback_stage_table(
            "[BVP Damp race] combustion-1000 Lambdify callback stage breakdown",
            &rows,
        );
        println!();
        print_e2e_lifecycle_table(
            "[BVP Damp race] combustion-1000 Lambdify lifecycle/refinement breakdown",
            &rows,
        );
        println!();
        print_e2e_bootstrap_pass_table(
            "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded symbolic handoff stages",
            &rows,
        );
        println!();
        print_e2e_symbolic_jacobian_detail_table(
            "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded internal symbolic-Jacobian stages",
            &rows,
        );
        println!();
        print_e2e_lambdify_binding_detail_table(
            "[BVP Damp race] combustion-1000 Lambdify Sparse vs Banded callback compilation stages",
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
        println!();
        print_e2e_callback_stage_table(
            "[BVP Damp race] combustion-1000 AOT callback stage breakdown",
            &rows,
        );
        println!();
        print_e2e_lifecycle_table(
            "[BVP Damp race] combustion-1000 AOT lifecycle/refinement breakdown",
            &rows,
        );
        println!();
        print_e2e_bootstrap_pass_table(
            "[BVP Damp race] combustion-1000 AOT Sparse vs Banded symbolic handoff stages",
            &rows,
        );
        println!();
        print_e2e_symbolic_jacobian_detail_table(
            "[BVP Damp race] combustion-1000 AOT Sparse vs Banded internal symbolic-Jacobian stages",
            &rows,
        );

        assert!(
            rows.iter().any(|row| row.ok_runs > 0),
            "at least one AOT race variant should solve successfully"
        );
    }

    #[test]
    #[ignore = "heavy combustion-1000 AOT toolchain/chunking release matrix for Sparse and Banded"]
    fn combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix() {
        let n_steps = 1000usize;
        let repetitions = 2usize;
        let variants = [
            RaceVariant {
                source: "Lambdify",
                matrix: "Sparse",
                variant: "AtomView",
                bootstrap_hint: "baseline+symbolic+lambdify",
                config: sparse_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "Lambdify",
                matrix: "Banded",
                variant: "AtomView",
                bootstrap_hint: "baseline+symbolic+lambdify",
                config: banded_atomview_lambdify_baseline(),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "gcc/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "gcc/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "gcc/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "gcc/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "tcc/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "tcc/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "tcc/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "zig/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "zig/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "zig/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "zig/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "rust/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    sparse_atomview_rust_aot_release(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Sparse",
                variant: "rust/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    sparse_atomview_rust_aot_release(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "rust/whole",
                bootstrap_hint: "rebuild+seq+whole",
                config: release_matrix_config(
                    banded_atomview_rust_aot_release(),
                    whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
            },
            RaceVariant {
                source: "AOT",
                matrix: "Banded",
                variant: "rust/chunk4",
                bootstrap_hint: "rebuild+par+chunk4",
                config: release_matrix_config(
                    banded_atomview_rust_aot_release(),
                    four_way_chunking(),
                    forced_parallel_execution(),
                ),
            },
        ];

        let variants = apply_optional_release_matrix_filter(&variants);
        let samples = run_race_samples(&variants, n_steps, repetitions);
        let rows = summarize_samples(&variants, &samples);
        print_race_summary_table(
            "[BVP Damp race] combustion-1000 AOT toolchain/chunking release matrix",
            &rows,
        );

        assert!(
            rows.iter()
                .filter(|row| row.source == "Lambdify")
                .all(|row| row.ok_runs == row.runs),
            "Lambdify baseline variants must solve successfully before AOT chunking rows are interpreted"
        );
    }

    #[test]
    #[ignore = "end-to-end combustion-200 AOT/Lambdify matrix across Sparse/Banded, gcc/tcc/zig, whole/chunk4"]
    fn combustion_200_aot_toolchain_chunking_sparse_banded_end_to_end_matrix() {
        let n_steps = 200usize;
        let repetitions = 3usize;
        let variants = combustion_toolchain_chunking_variants();
        let variants = apply_optional_release_matrix_filter(&variants);

        let samples = run_race_samples(&variants, n_steps, repetitions);
        let rows = summarize_samples(&variants, &samples);
        print_e2e_correctness_table(
            "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT correctness matrix",
            &rows,
        );
        println!();
        print_e2e_performance_table(
            "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT timing/counter matrix",
            &rows,
        );
        println!();
        print_e2e_callback_stage_table(
            "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT callback stage matrix",
            &rows,
        );
        println!();
        print_e2e_lifecycle_table(
            "[BVP Damp e2e] combustion-200 Sparse/Banded Lambdify+AOT lifecycle matrix",
            &rows,
        );

        assert!(
            rows.iter()
                .filter(|row| row.source == "Lambdify")
                .all(|row| row.ok_runs == row.runs),
            "Lambdify baseline variants must solve successfully before AOT rows are interpreted"
        );
        assert!(
            rows.iter().all(|row| row.ok_runs == row.runs),
            "all combustion-200 Sparse/Banded AOT toolchain/chunking variants should solve successfully"
        );
        assert!(
            rows.iter().filter(|row| row.source == "AOT").all(|row| row
                .solve_diff
                .mean
                .is_finite()
                && row.solve_diff.mean <= 1e-6),
            "AOT rows must remain numerically equivalent to the Lambdify baseline"
        );
    }

    #[test]
    fn isolated_cold_race_payload_round_trips_metrics_and_solution() {
        let variant = RaceVariant {
            source: "AOT",
            matrix: "Banded",
            variant: "tcc/chunk4",
            bootstrap_hint: "isolated",
            config: GeneratedBackendConfig::banded_lambdify_defaults(),
        };
        let row = RaceRow {
            source: variant.source,
            matrix: variant.matrix,
            variant: variant.variant,
            bootstrap_hint: variant.bootstrap_hint,
            total_ms: 12.5,
            max_abs_solution: 3.5,
            solve_diff: 0.0,
            rel_x_diff: 0.0,
            iterations: 5,
            linear_solves: 10,
            jac_rebuilds: 1,
            grid_refinements: 0,
            final_grid_points: 31,
            total_timer_ms: 11.0,
            symbolic_timer_ms: 7.0,
            linear_timer_ms: 2.0,
            jac_timer_ms: 1.0,
            fun_timer_ms: 0.5,
            cb_residual_values_ms: 0.2,
            cb_jacobian_values_ms: 0.3,
            cb_jacobian_assembly_ms: 0.1,
            residual_actual_jobs: 4.0,
            sparse_jacobian_actual_jobs: 4.0,
            residual_work_per_job: 50.0,
            sparse_jacobian_work_per_job: 70.0,
            residual_fallback_reason: "none".to_string(),
            sparse_jacobian_fallback_reason: "none".to_string(),
            selected_backend: "AotCompiled".to_string(),
            symbolic_assembly_backend: "ExprLegacy".to_string(),
            aot_build_policy: "RebuildAlways".to_string(),
            initial_generate_ms: 6.0,
            initial_discretization_ms: 1.0,
            initial_symbolic_jacobian_ms: 4.0,
            initial_symbolic_variable_sets_ms: 0.1,
            initial_symbolic_row_differentiation_ms: 3.5,
            initial_symbolic_dense_cache_ms: 0.2,
            initial_symbolic_sparse_flatten_ms: 0.1,
            initial_sparse_prepare_ms: 0.5,
            initial_runtime_binding_ms: 0.1,
            initial_lambdify_jacobian_compile_ms: 0.04,
            initial_lambdify_residual_compile_ms: 0.03,
            post_build_generate_ms: f64::NAN,
            post_build_discretization_ms: f64::NAN,
            post_build_symbolic_jacobian_ms: f64::NAN,
            post_build_sparse_prepare_ms: f64::NAN,
            post_build_runtime_binding_ms: f64::NAN,
            post_build_rebind_ms: 0.05,
            aot_artifact_ms: 0.8,
            aot_module_ms: 0.6,
            aot_residual_lower_ms: 0.4,
            aot_jacobian_lower_ms: 0.3,
            aot_source_emit_ms: 0.2,
            aot_packaging_ms: 0.1,
            aot_materialize_ms: 0.05,
            aot_compile_link_ms: 0.7,
            aot_register_link_ms: 0.04,
            status: "ok".to_string(),
        };
        let decoded = decode_isolated_race_row(
            &format!(
                "{ISOLATED_RACE_ROW_MARKER}\t{}",
                encode_isolated_race_row(&row)
            ),
            &variant,
        );
        assert_eq!(decoded.variant, row.variant);
        assert_eq!(decoded.iterations, row.iterations);
        assert_eq!(decoded.residual_actual_jobs, 4.0);
        assert_eq!(decoded.selected_backend, "AotCompiled");
        assert_eq!(decoded.initial_symbolic_row_differentiation_ms, 3.5);
        assert!(decoded.post_build_generate_ms.is_nan());

        let solution = DMatrix::from_column_slice(
            6,
            2,
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let decoded_solution = decode_isolated_solution(&format!(
            "{ISOLATED_RACE_SOLUTION_MARKER}\t{}",
            encode_isolated_solution(&solution)
        ));
        assert_eq!(decoded_solution, solution);
    }

    fn run_combustion_3000_banded_isolated_stress_story(
        test_name: &str,
        frontend_label: &str,
        variants: Vec<RaceVariant>,
    ) {
        let n_steps = 3_000usize;
        let repetitions = 2usize;
        let variants = apply_optional_release_matrix_filter(&variants);

        if let Ok(index) = std::env::var(ISOLATED_STRESS_CHILD_INDEX_ENV) {
            let index = index
                .parse::<usize>()
                .expect("isolated stress child index should be an integer");
            let variant = variants.get(index).unwrap_or_else(|| {
                panic!("isolated stress child variant index {index} is invalid")
            });
            let (row, solution) = run_race_variant(
                n_steps,
                variant.source,
                variant.matrix,
                variant.variant,
                variant.bootstrap_hint,
                variant.config.clone(),
            );
            assert_eq!(
                row.status, "ok",
                "isolated stress child {} should solve successfully",
                variant.variant
            );
            let solution = solution
                .as_ref()
                .expect("isolated stress child should provide its converged solution");
            println!("{ISOLATED_RACE_PID_MARKER}\t{}", std::process::id());
            println!(
                "{ISOLATED_RACE_ROW_MARKER}\t{}",
                encode_isolated_race_row(&row)
            );
            println!(
                "{ISOLATED_RACE_SOLUTION_MARKER}\t{}",
                encode_isolated_solution(solution)
            );
            return;
        }

        println!(
            "[BVP Damp isolated cold] protocol cooldown_ms={}, cleanup_child_artifacts={}",
            cold_cooldown_ms(),
            clean_cold_artifacts_enabled()
        );
        let samples = run_isolated_race_samples(test_name, &variants, repetitions);
        let rows = summarize_samples(&variants, &samples);
        print_isolated_cold_sample_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} raw process-isolated cold observations"
            ),
            &samples,
            variants.len(),
        );
        println!();
        print_e2e_correctness_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT correctness"
            ),
            &rows,
        );
        println!();
        print_e2e_performance_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT timing/counters"
            ),
            &rows,
        );
        println!();
        print_e2e_callback_stage_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT callback stages"
            ),
            &rows,
        );
        println!();
        print_e2e_lifecycle_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT lifecycle/refinement stages"
            ),
            &rows,
        );
        println!();
        print_e2e_bootstrap_pass_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT symbolic handoff passes"
            ),
            &rows,
        );
        println!();
        print_e2e_symbolic_jacobian_detail_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} internal initial symbolic-Jacobian stages"
            ),
            &rows,
        );
        println!();
        print_e2e_lambdify_binding_detail_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} Lambdify callback compilation stages"
            ),
            &rows,
        );
        println!();
        print_e2e_aot_bootstrap_table(
            &format!(
                "[BVP Damp stress] combustion-3000 {frontend_label} Banded Lambdify vs AOT cold-build stages"
            ),
            &rows,
        );

        assert!(
            rows.iter()
                .filter(|row| row.source == "Lambdify")
                .all(|row| row.ok_runs == row.runs),
            "Banded Lambdify baseline must solve before AOT stress rows are interpreted"
        );
        assert!(
            rows.iter().all(|row| row.ok_runs == row.runs),
            "all selected combustion-3000 Banded Lambdify/AOT stress variants should solve successfully"
        );
        assert!(
            rows.iter().filter(|row| row.source == "AOT").all(|row| row
                .solve_diff
                .mean
                .is_finite()
                && row.solve_diff.mean <= 1e-5),
            "AOT stress rows must remain numerically equivalent to the Banded Lambdify baseline"
        );
        for row in rows.iter().filter(|row| row.variant == "tcc/chunk4") {
            assert!(
                row.residual_actual_jobs.mean > 1.0 && row.sparse_jacobian_actual_jobs.mean > 1.0,
                "process-isolated chunk4 row must retain real parallel execution"
            );
            assert_eq!(
                row.residual_fallback_reason, "none",
                "process-isolated chunk4 residual callback must not fall back to sequential"
            );
            assert_eq!(
                row.sparse_jacobian_fallback_reason, "none",
                "process-isolated chunk4 Jacobian callback must not fall back to sequential"
            );
        }
    }

    #[test]
    #[ignore = "very heavy process-isolated cold combustion-3000 ExprLegacy Banded Lambdify vs tcc whole/chunk4 end-to-end control"]
    fn combustion_3000_banded_lambdify_vs_aot_end_to_end_stress() {
        run_combustion_3000_banded_isolated_stress_story(
            "numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_3000_banded_lambdify_vs_aot_end_to_end_stress",
            "ExprLegacy",
            vec![
                RaceVariant {
                    source: "Lambdify",
                    matrix: "Banded",
                    variant: "ExprLegacy",
                    bootstrap_hint: "exprlegacy+lambdify",
                    config: GeneratedBackendConfig::banded_lambdify_defaults()
                        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Banded",
                    variant: "tcc/whole",
                    bootstrap_hint: "rebuild+seq+whole",
                    config: release_matrix_config(
                        GeneratedBackendConfig::banded_build_if_missing_release()
                            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                            .with_aot_codegen_backend(AotCodegenBackend::C)
                            .with_aot_c_compiler("tcc"),
                        whole_chunking(),
                        AotExecutionPolicy::SequentialOnly,
                    ),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Banded",
                    variant: "tcc/chunk4",
                    bootstrap_hint: "rebuild+par+chunk4",
                    config: release_matrix_config(
                        GeneratedBackendConfig::banded_build_if_missing_release()
                            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                            .with_aot_codegen_backend(AotCodegenBackend::C)
                            .with_aot_c_compiler("tcc"),
                        four_way_chunking(),
                        forced_parallel_execution(),
                    ),
                },
            ],
        );
    }

    #[test]
    #[ignore = "very heavy process-isolated cold combustion-3000 AtomView Banded Lambdify vs tcc whole/chunk4 production end-to-end stress test"]
    fn combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress() {
        run_combustion_3000_banded_isolated_stress_story(
            "numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress",
            "AtomView",
            vec![
                RaceVariant {
                    source: "Lambdify",
                    matrix: "Banded",
                    variant: "AtomView",
                    bootstrap_hint: "atomview+lambdify",
                    config: banded_atomview_lambdify_baseline(),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Banded",
                    variant: "tcc/whole",
                    bootstrap_hint: "atomview+rebuild+seq+whole",
                    config: release_matrix_config(
                        GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                        whole_chunking(),
                        AotExecutionPolicy::SequentialOnly,
                    ),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Banded",
                    variant: "tcc/chunk4",
                    bootstrap_hint: "atomview+rebuild+par+chunk4",
                    config: release_matrix_config(
                        GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                        four_way_chunking(),
                        forced_parallel_execution(),
                    ),
                },
            ],
        );
    }

    fn run_combustion_1000_symbolic_frontend_honest_wall_clock_table(
        test_name: &str,
        matrix: &str,
        variants: Vec<RaceVariant>,
    ) {
        let n_steps = 1_000usize;
        let repetitions = 3usize;
        let variants = apply_optional_release_matrix_filter(&variants);

        if let Ok(index) = std::env::var(ISOLATED_STRESS_CHILD_INDEX_ENV) {
            let index = index
                .parse::<usize>()
                .expect("isolated symbolic-frontend child index should be an integer");
            let variant = variants.get(index).unwrap_or_else(|| {
                panic!("isolated symbolic-frontend child variant index {index} is invalid")
            });
            let (row, solution) = run_race_variant(
                n_steps,
                variant.source,
                variant.matrix,
                variant.variant,
                variant.bootstrap_hint,
                variant.config.clone(),
            );
            assert_eq!(
                row.status, "ok",
                "isolated symbolic-frontend child {} should solve successfully",
                variant.variant
            );
            let solution = solution
                .as_ref()
                .expect("isolated symbolic-frontend child should provide its converged solution");
            println!("{ISOLATED_RACE_PID_MARKER}\t{}", std::process::id());
            println!(
                "{ISOLATED_RACE_ROW_MARKER}\t{}",
                encode_isolated_race_row(&row)
            );
            println!(
                "{ISOLATED_RACE_SOLUTION_MARKER}\t{}",
                encode_isolated_solution(solution)
            );
            return;
        }

        println!(
            "[BVP Damp symbolic frontend cold] protocol cooldown_ms={}, cleanup_child_artifacts={}",
            cold_cooldown_ms(),
            clean_cold_artifacts_enabled()
        );
        let samples = run_isolated_race_samples(test_name, &variants, repetitions);
        let rows = summarize_samples(&variants, &samples);
        print_isolated_cold_sample_table(
            &format!(
                "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} raw process-isolated observations"
            ),
            &samples,
            variants.len(),
        );
        println!();
        print_e2e_correctness_table(
            &format!(
                "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} ExprLegacy/AtomView correctness"
            ),
            &rows,
        );
        println!();
        print_e2e_performance_table(
            &format!(
                "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} ExprLegacy/AtomView wall-clock and solver stages"
            ),
            &rows,
        );
        println!();
        print_e2e_lifecycle_table(
            &format!(
                "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} backend selection and symbolic totals"
            ),
            &rows,
        );
        println!();
        print_e2e_bootstrap_pass_table(
            &format!(
                "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} symbolic handoff stages"
            ),
            &rows,
        );
        println!();
        print_e2e_symbolic_jacobian_detail_table(
            &format!(
                "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} internal initial symbolic-Jacobian stages"
            ),
            &rows,
        );
        println!();
        print_e2e_aot_bootstrap_table(
            &format!(
                "[BVP Damp symbolic frontend cold] combustion-1000 {matrix} tcc cold-build stages"
            ),
            &rows,
        );

        assert!(
            rows.iter().all(|row| row.ok_runs == row.runs),
            "every symbolic-frontend cold comparison row should solve successfully"
        );
        assert!(
            rows.iter()
                .all(|row| row.solve_diff.mean.is_finite() && row.solve_diff.mean <= 1e-5),
            "ExprLegacy, AtomView, Lambdify and tcc AOT rows must remain numerically equivalent"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.post_build_symbolic_jacobian_ms.mean.is_nan()),
            "fresh AOT rebinding must not rebuild either symbolic frontend after compilation"
        );
    }

    #[test]
    #[ignore = "process-isolated combustion-1000 Banded ExprLegacy/AtomView symbolic frontend comparison for Lambdify and tcc AOT"]
    fn combustion_1000_banded_symbolic_frontend_honest_wall_clock_table() {
        run_combustion_1000_symbolic_frontend_honest_wall_clock_table(
            "numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_1000_banded_symbolic_frontend_honest_wall_clock_table",
            "Banded",
            vec![
                RaceVariant {
                    source: "Lambdify",
                    matrix: "Banded",
                    variant: "ExprLegacy",
                    bootstrap_hint: "exprlegacy+lambdify",
                    config: GeneratedBackendConfig::banded_lambdify_defaults()
                        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
                },
                RaceVariant {
                    source: "Lambdify",
                    matrix: "Banded",
                    variant: "AtomView",
                    bootstrap_hint: "atomview+lambdify",
                    config: banded_atomview_lambdify_baseline(),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Banded",
                    variant: "ExprLegacy+tcc",
                    bootstrap_hint: "exprlegacy+rebuild+seq+whole",
                    config: release_matrix_config(
                        GeneratedBackendConfig::banded_build_if_missing_release()
                            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                            .with_aot_codegen_backend(AotCodegenBackend::C)
                            .with_aot_c_compiler("tcc"),
                        whole_chunking(),
                        AotExecutionPolicy::SequentialOnly,
                    ),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Banded",
                    variant: "AtomView+tcc",
                    bootstrap_hint: "atomview+rebuild+seq+whole",
                    config: release_matrix_config(
                        GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
                        whole_chunking(),
                        AotExecutionPolicy::SequentialOnly,
                    ),
                },
            ],
        );
    }

    #[test]
    #[ignore = "process-isolated combustion-1000 Sparse ExprLegacy/AtomView symbolic frontend comparison for Lambdify and tcc AOT"]
    fn combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table() {
        run_combustion_1000_symbolic_frontend_honest_wall_clock_table(
            "numerical::BVP_Damp::BVP_Damp_tests4::tests::combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table",
            "Sparse",
            vec![
                RaceVariant {
                    source: "Lambdify",
                    matrix: "Sparse",
                    variant: "ExprLegacy",
                    bootstrap_hint: "exprlegacy+lambdify",
                    config: GeneratedBackendConfig::sparse_defaults()
                        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
                },
                RaceVariant {
                    source: "Lambdify",
                    matrix: "Sparse",
                    variant: "AtomView",
                    bootstrap_hint: "atomview+lambdify",
                    config: sparse_atomview_lambdify_baseline(),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Sparse",
                    variant: "ExprLegacy+tcc",
                    bootstrap_hint: "exprlegacy+rebuild+seq+whole",
                    config: release_matrix_config(
                        GeneratedBackendConfig::sparse_build_if_missing_release()
                            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
                            .with_aot_codegen_backend(AotCodegenBackend::C)
                            .with_aot_c_compiler("tcc"),
                        whole_chunking(),
                        AotExecutionPolicy::SequentialOnly,
                    ),
                },
                RaceVariant {
                    source: "AOT",
                    matrix: "Sparse",
                    variant: "AtomView+tcc",
                    bootstrap_hint: "atomview+rebuild+seq+whole",
                    config: release_matrix_config(
                        GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
                        whole_chunking(),
                        AotExecutionPolicy::SequentialOnly,
                    ),
                },
            ],
        );
    }

    struct DampedLifecycleRow {
        matrix: &'static str,
        phase: &'static str,
        total_ms: f64,
        symbolic_ms: f64,
        linear_ms: f64,
        solve_diff: f64,
        initial_generate_ms: f64,
        post_build_rebind_ms: f64,
        compile_link_ms: f64,
        selected_backend: String,
        build_policy: String,
    }

    fn run_damped_lifecycle_row(
        n_steps: usize,
        matrix: &'static str,
        phase: &'static str,
        config: GeneratedBackendConfig,
        baseline: Option<&DMatrix<f64>>,
    ) -> (DampedLifecycleRow, DMatrix<f64>, GeneratedBackendConfig) {
        let begin = Instant::now();
        let mut solver = make_combustion_solver(n_steps, config);
        solver.try_solver().unwrap_or_else(|err| {
            panic!("{matrix}/{phase} damped lifecycle solve failed: {err:?}")
        });
        let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
        let solution = solver
            .get_result()
            .expect("successful damped lifecycle solve must expose a result");
        let stats = solver.get_statistics();
        let row = DampedLifecycleRow {
            matrix,
            phase,
            total_ms,
            symbolic_ms: stats_timer_ms(&stats, "Symbolic Operations"),
            linear_ms: stats_timer_ms(&stats, "Linear System"),
            solve_diff: baseline
                .map(|reference| solution_linf_diff(&solution, reference))
                .unwrap_or(0.0),
            initial_generate_ms: stats_diagnostic_ms(
                &stats,
                "generated.handoff.initial_generate_wall_ms",
            ),
            post_build_rebind_ms: stats_diagnostic_ms(
                &stats,
                "generated.handoff.post_build_rebind_wall_ms",
            ),
            compile_link_ms: stats_diagnostic_ms(&stats, "generated.aot.compile_link_ms"),
            selected_backend: stats_diagnostic_string(&stats, "generated.selected_backend"),
            build_policy: stats_diagnostic_string(&stats, "aot.build_policy"),
        };
        (row, solution, solver.generated_backend_config().clone())
    }

    fn print_damped_lifecycle_table(rows: &[DampedLifecycleRow]) {
        println!(
            "[BVP Damp lifecycle] combustion-1000 AtomView tcc BuildIfMissing -> RequirePrebuilt correctness"
        );
        println!("matrix | phase      | selected_backend | build_policy    | solve_diff");
        println!("{}", "-".repeat(84));
        for row in rows {
            println!(
                "{:<6} | {:<10} | {:<16} | {:<15} | {:.6e}",
                row.matrix, row.phase, row.selected_backend, row.build_policy, row.solve_diff
            );
        }
        println!();
        println!(
            "[BVP Damp lifecycle] wall-clock and artifact stages; all time columns are milliseconds"
        );
        println!(
            "matrix | phase      | total_ms | symbolic_ms | linear_ms | initial_generate | rebind_ms | compile_link"
        );
        println!("{}", "-".repeat(112));
        for row in rows {
            println!(
                "{:<6} | {:<10} | {:>8.3} | {:>11.3} | {:>9.3} | {:>16.3} | {:>9.3} | {:>12.3}",
                row.matrix,
                row.phase,
                row.total_ms,
                row.symbolic_ms,
                row.linear_ms,
                row.initial_generate_ms,
                row.post_build_rebind_ms,
                row.compile_link_ms,
            );
        }
    }

    fn print_damped_warm_comparison_table(
        build_row: &DampedLifecycleRow,
        samples: &[(usize, usize, DampedLifecycleRow)],
    ) {
        println!(
            "[BVP Damp warm] Banded AtomView Lambdify vs tcc RequirePrebuilt; setup build row"
        );
        println!(
            "phase | selected_backend | build_policy    | total_ms | symbolic_ms | rebind_ms | compile_link | solve_diff"
        );
        println!("{}", "-".repeat(120));
        println!(
            "{:<5} | {:<16} | {:<15} | {:>8.3} | {:>11.3} | {:>9.3} | {:>12.3} | {:.6e}",
            build_row.phase,
            build_row.selected_backend,
            build_row.build_policy,
            build_row.total_ms,
            build_row.symbolic_ms,
            build_row.post_build_rebind_ms,
            build_row.compile_link_ms,
            build_row.solve_diff,
        );
        println!();
        println!(
            "[BVP Damp warm] measured rows after cooldown_ms={}; milliseconds",
            warm_cooldown_ms()
        );
        println!(
            "rep | pos | phase      | selected_backend | build_policy    | total_ms | symbolic_ms | linear_ms | initial_generate | compile_link | solve_diff"
        );
        println!("{}", "-".repeat(160));
        for (repetition, position, row) in samples {
            println!(
                "{:>3} | {:>3} | {:<10} | {:<16} | {:<15} | {:>8.3} | {:>11.3} | {:>9.3} | {:>16.3} | {:>12.3} | {:.6e}",
                repetition,
                position,
                row.phase,
                row.selected_backend,
                row.build_policy,
                row.total_ms,
                row.symbolic_ms,
                row.linear_ms,
                row.initial_generate_ms,
                row.compile_link_ms,
                row.solve_diff,
            );
        }
        println!();
        println!(
            "[BVP Damp warm] paired summary: build row excluded; each route has the same cooldown and alternating order"
        );
        println!(
            "phase      | runs | total_ms mean+/-std [min,max] | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff"
        );
        println!("{}", "-".repeat(150));
        for phase in ["lambdify", "prebuilt"] {
            let rows = samples
                .iter()
                .filter(|(_, _, row)| row.phase == phase)
                .map(|(_, _, row)| row)
                .collect::<Vec<_>>();
            let total = aggregate(rows.iter().map(|row| row.total_ms));
            let symbolic = aggregate(rows.iter().map(|row| row.symbolic_ms));
            let linear = aggregate(rows.iter().map(|row| row.linear_ms));
            let max_diff = rows
                .iter()
                .map(|row| row.solve_diff)
                .fold(0.0_f64, f64::max);
            println!(
                "{:<10} | {:>4} | {:<32} | {:<21} | {:<21} | {:.6e}",
                phase,
                rows.len(),
                fmt_agg(total),
                fmt_agg_short(symbolic),
                fmt_agg_short(linear),
                max_diff,
            );
        }
    }

    #[test]
    #[ignore = "heavy damped combustion-1000 Sparse/Banded AtomView tcc artifact lifecycle: BuildIfMissing then strict RequirePrebuilt reuse"]
    fn combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story() {
        let n_steps = 1_000;
        let (baseline_row, baseline, _) = run_damped_lifecycle_row(
            n_steps,
            "Banded",
            "baseline",
            banded_atomview_lambdify_baseline(),
            None,
        );
        let mut rows = vec![baseline_row];
        let build_configs = [
            (
                "Sparse",
                GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
            ),
            (
                "Banded",
                GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
            ),
        ];
        for (matrix, config) in build_configs {
            let config = config
                .with_aot_compile_dev_fastest()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
                .with_aot_chunking_policy(whole_chunking())
                .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                });
            let (built_row, _, built_config) =
                run_damped_lifecycle_row(n_steps, matrix, "build", config, Some(&baseline));
            assert!(
                built_config.resolver.is_some(),
                "{matrix} BuildIfMissing must preserve a resolver snapshot for strict reuse"
            );
            rows.push(built_row);
            let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
            for _ in 0..3 {
                let (prebuilt_row, _, _) = run_damped_lifecycle_row(
                    n_steps,
                    matrix,
                    "prebuilt",
                    strict_config.clone(),
                    Some(&baseline),
                );
                rows.push(prebuilt_row);
            }
        }

        print_damped_lifecycle_table(&rows);
        assert!(
            rows.iter().all(|row| row.solve_diff <= 1e-5),
            "Damped lifecycle routes must remain equivalent to the Lambdify baseline"
        );
        assert!(
            rows.iter()
                .filter(|row| row.phase != "baseline")
                .all(|row| row.selected_backend == "AotCompiled"),
            "BuildIfMissing and RequirePrebuilt routes must run compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.phase == "prebuilt")
                .all(|row| row.build_policy == "RequirePrebuilt"),
            "warm rows must be strict RequirePrebuilt executions"
        );
    }

    #[test]
    #[ignore = "heavy warm repeated-solve comparison with cooldown: combustion-1000 Banded AtomView Lambdify vs strict tcc RequirePrebuilt"]
    fn combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story() {
        let n_steps = 1_000;
        let repetitions = 5;
        let cooldown_ms = warm_cooldown_ms();
        let (_, reference_solution, _) = run_damped_lifecycle_row(
            n_steps,
            "Banded",
            "reference",
            banded_atomview_lambdify_baseline(),
            None,
        );
        let build_config = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
            .with_aot_compile_dev_fastest()
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_chunking_policy(whole_chunking())
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            });
        let (build_row, _, built_config) = run_damped_lifecycle_row(
            n_steps,
            "Banded",
            "build",
            build_config,
            Some(&reference_solution),
        );
        assert!(
            built_config.resolver.is_some(),
            "warm comparison requires a resolver snapshot from BuildIfMissing"
        );
        let prebuilt_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        let mut samples = Vec::with_capacity(repetitions * 2);

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
                let config = if phase == "lambdify" {
                    banded_atomview_lambdify_baseline()
                } else {
                    prebuilt_config.clone()
                };
                let (row, _, _) = run_damped_lifecycle_row(
                    n_steps,
                    "Banded",
                    phase,
                    config,
                    Some(&reference_solution),
                );
                samples.push((repetition, position + 1, row));
            }
        }

        print_damped_warm_comparison_table(&build_row, &samples);
        assert!(
            build_row.selected_backend == "AotCompiled" && build_row.solve_diff <= 1e-5,
            "setup build row must install a correct compiled backend"
        );
        assert!(
            samples.iter().all(|(_, _, row)| row.solve_diff <= 1e-5),
            "all measured warm rows must match the common Lambdify reference"
        );
        assert!(
            samples
                .iter()
                .filter(|(_, _, row)| row.phase == "prebuilt")
                .all(|(_, _, row)| row.selected_backend == "AotCompiled"
                    && row.build_policy == "RequirePrebuilt"
                    && row.compile_link_ms.is_nan()),
            "measured prebuilt rows must stay compiled without any rebuild/link step"
        );
        assert_eq!(
            samples
                .iter()
                .filter(|(_, _, row)| row.phase == "lambdify")
                .count(),
            repetitions
        );
        assert_eq!(
            samples
                .iter()
                .filter(|(_, _, row)| row.phase == "prebuilt")
                .count(),
            repetitions
        );
    }
}
