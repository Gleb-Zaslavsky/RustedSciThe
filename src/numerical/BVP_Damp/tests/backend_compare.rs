#![cfg(test)]

//! Solver-facing backend comparison diagnostics for BVP generated pipelines.
//!
//! This module complements the lower-level codegen backend comparison tests by
//! asking the same practical questions one layer higher, through the public BVP
//! solver API:
//! - how long does it take to get callable residual/Jacobian functions,
//! - how fast do those installed callbacks run,
//! - and does the full solve remain numerically aligned with the lambdify path.
//!
//! Main diagnostic:
//! - `combustion_lambdify_vs_atomview_c_leaders_compare_1000`
//!   Compares the practical leaders for large BVPs:
//!   `Lambdify (ExprLegacy)` vs `AtomView + C-gcc` vs `AtomView + C-tcc`.
//!   Hypothesis: `Lambdify` remains strongest for one-shot bootstrap, while
//!   `AtomView + C-tcc/gcc` wins decisively once callable native callbacks are ready.
//! - `combustion_production_like_end_to_end_compare_1000`
//!   Asks the production-style question with no extra diagnostics:
//!   "given solver options and a problem, which path gets to a solved result fastest?"
//!   Hypothesis: `Lambdify` is a strong one-shot baseline, while `AtomView + C-tcc`
//!   and `AtomView + C-gcc` may win once native build latency is low enough.
//! - `combustion_break_even_lambdify_vs_atomview_ctcc_2000`
//!   Asks: after one setup/bootstrap, how many repeated solves are needed for
//!   `AtomView + C-tcc` to amortize its higher startup cost against `Lambdify`?
//!   Hypothesis: for very large BVPs, `Lambdify` wins one-shot runs, but repeated
//!   solves can still make the `C-tcc` path worthwhile.

//! Architecture lane: explicit ExprLegacy oracle versus AtomView parity and
//! isolated performance comparison. This remains an intermediate migration
//! module; the final physical test layout is intentionally not fixed yet.

mod tests {
    use crate::Utils::test_reporting::write_test_report;
    use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
    use crate::numerical::BVP_Damp::MatrixBackend;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotExecutionPolicy, GeneratedBackendConfig,
    };
    use crate::numerical::BVP_Damp::test_common::{
        AotStoryProtocol, SymbolicTestRoute, TEST_SUITE_ARCHITECTURE, sleep_ms, story_repetitions,
        uniform_initial_guess,
    };
    use crate::symbolic::bvp::telemetry::{
        BvpDirectJacobianTelemetrySnapshot, BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode,
    };
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::fmt::Write as FmtWrite;
    use std::hint::black_box;
    use std::time::Instant;

    macro_rules! println {
        () => {
            crate::Utils::test_reporting::capture_test_line(format_args!(""));
        };
        ($($arg:tt)*) => {
            crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
        };
    }

    macro_rules! aot_compare_test_report {
        ($name:ident) => {
            let _test_report = crate::Utils::test_reporting::TestReportCapture::new(
                "BVP_Damp_AOT_Compare",
                concat!(module_path!(), "::", stringify!($name)),
            );
        };
    }

    #[derive(Debug)]
    struct EndToEndRow {
        variant: &'static str,
        sym_backend: &'static str,
        preset: &'static str,
        setup_ms: f64,
        residual_ms: f64,
        jacobian_ms: f64,
        callback_total_ms: f64,
        solve_ms: f64,
        total_ms: f64,
        speedup_vs_lambdify: f64,
        max_abs_solution: f64,
        residual_diff: f64,
        jacobian_diff: f64,
        solution_diff: f64,
        status: &'static str,
    }

    fn make_combustion_solver(n_steps: usize, options: DampedSolverOptions) -> NRBVP {
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

    fn combustion_options_base() -> DampedSolverOptions {
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
        DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(strategy_params))
            .with_abs_tolerance(1e-6)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(100)
            .with_bounds(bounds)
            .with_loglevel(Some("error".to_string()))
    }

    fn lambdify_options() -> DampedSolverOptions {
        let config = GeneratedBackendConfig::default()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
        combustion_options_base().with_generated_backend_config(config)
    }

    fn banded_lambdify_options(assembly: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
        combustion_options_base()
            .with_banded_lambdify()
            .with_symbolic_assembly_backend(assembly)
    }

    fn combustion_lambdify_options_for_route(
        route: &'static str,
        assembly: BvpSymbolicAssemblyBackend,
    ) -> DampedSolverOptions {
        let config = match route {
            "Sparse" => GeneratedBackendConfig::sparse_defaults()
                .with_matrix_backend_override(MatrixBackend::SparseCol),
            "Banded" => GeneratedBackendConfig::banded_lambdify_defaults(),
            other => panic!("unsupported combustion baseline route: {other}"),
        }
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(assembly)
        // Keep the ExprLegacy/AtomView stage baseline on identical runtime
        // dispatch. Sequential-vs-parallel is measured by a separate story.
        .with_lambdify_execution_policy(BvpLambdifyExecutionPolicy::Parallel { min_work: 0 });
        combustion_options_base().with_generated_backend_config(config)
    }

    fn atomview_gcc_options() -> DampedSolverOptions {
        let mut options = combustion_options_base().with_sparse_atomview_c_gcc();
        options.generated_backend_config = options
            .generated_backend_config
            .clone()
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                profile:
                    crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
            });
        options
    }

    fn atomview_tcc_options() -> DampedSolverOptions {
        let mut options = combustion_options_base().with_sparse_atomview_c_tcc();
        options.generated_backend_config = options
            .generated_backend_config
            .clone()
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_build_policy(AotBuildPolicy::RebuildAlways {
                profile:
                    crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
            });
        options
    }

    fn compare_installed_callbacks(lhs: &mut NRBVP, rhs: &mut NRBVP) -> (f64, f64) {
        let args = DVector::from_element(lhs.values.len() * lhs.n_steps, 0.99);
        let typed = Vectors_type_casting(&args, lhs.method.clone());
        let lhs_residual = lhs.fun.call(1.0, &*typed).to_DVectorType();
        let rhs_residual = rhs.fun.call(1.0, &*typed).to_DVectorType();
        let residual_diff = lhs_residual
            .iter()
            .zip(rhs_residual.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        let lhs_jacobian = lhs
            .jac
            .as_mut()
            .expect("lhs solver should expose Jacobian callback")
            .call(1.0, &*typed)
            .to_DMatrixType();
        let rhs_jacobian = rhs
            .jac
            .as_mut()
            .expect("rhs solver should expose Jacobian callback")
            .call(1.0, &*typed)
            .to_DMatrixType();
        let mut jacobian_diff: f64 = 0.0;
        for row in 0..lhs_jacobian.nrows() {
            for col in 0..lhs_jacobian.ncols() {
                jacobian_diff =
                    jacobian_diff.max((lhs_jacobian[(row, col)] - rhs_jacobian[(row, col)]).abs());
            }
        }
        (residual_diff, jacobian_diff)
    }

    fn measure_installed_callback_runtime(
        solver: &mut NRBVP,
        iters: usize,
        samples: usize,
    ) -> (f64, f64) {
        let args = DVector::from_element(solver.values.len() * solver.n_steps, 0.99);
        let typed = Vectors_type_casting(&args, solver.method.clone());
        let mut residual_samples = Vec::with_capacity(samples);
        let mut jacobian_samples = Vec::with_capacity(samples);

        for _ in 0..samples {
            let residual_begin = Instant::now();
            for _ in 0..iters {
                let residual = solver.fun.call(1.0, &*typed);
                black_box(residual);
            }
            residual_samples.push(residual_begin.elapsed().as_secs_f64() * 1_000.0);

            let jacobian_begin = Instant::now();
            for _ in 0..iters {
                let jacobian = solver
                    .jac
                    .as_mut()
                    .expect("solver should expose Jacobian callback")
                    .call(1.0, &*typed);
                black_box(jacobian);
            }
            jacobian_samples.push(jacobian_begin.elapsed().as_secs_f64() * 1_000.0);
        }

        let residual_avg = residual_samples.iter().sum::<f64>() / samples as f64;
        let jacobian_avg = jacobian_samples.iter().sum::<f64>() / samples as f64;
        (residual_avg, jacobian_avg)
    }

    fn measure_callback_cold_and_warm(solver: &mut NRBVP, warm_samples: usize) -> (f64, f64) {
        let args = DVector::from_element(solver.values.len() * solver.n_steps, 0.99);
        let typed = Vectors_type_casting(&args, solver.method.clone());

        let cold_begin = Instant::now();
        black_box(solver.fun.call(1.0, &*typed));
        black_box(
            solver
                .jac
                .as_mut()
                .expect("solver should expose Jacobian callback")
                .call(1.0, &*typed),
        );
        let cold_ms = cold_begin.elapsed().as_secs_f64() * 1_000.0;

        let warm_begin = Instant::now();
        for _ in 0..warm_samples {
            black_box(solver.fun.call(1.0, &*typed));
            black_box(
                solver
                    .jac
                    .as_mut()
                    .expect("solver should expose Jacobian callback")
                    .call(1.0, &*typed),
            );
        }
        let warm_ms = warm_begin.elapsed().as_secs_f64() * 1_000.0 / warm_samples as f64;
        (cold_ms, warm_ms)
    }

    fn solve_and_collect_solution(solver: &mut NRBVP) -> (f64, f64) {
        let solve_begin = Instant::now();
        solver.try_solve().expect("solver solve should succeed");
        let solve_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let solution = solver
            .get_result()
            .expect("solver should expose a solution after solve");
        let max_abs_solution = solution.iter().copied().map(f64::abs).fold(0.0, f64::max);
        (solve_ms, max_abs_solution)
    }

    fn solve_prepared_and_collect_solution(solver: &mut NRBVP) -> (f64, f64) {
        let solve_begin = Instant::now();
        solver
            .try_solver_prepared()
            .expect("prepared solver solve should succeed");
        let solve_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let solution = solver
            .get_result()
            .expect("solver should expose a solution after solve");
        let max_abs_solution = solution.iter().copied().map(f64::abs).fold(0.0, f64::max);
        (solve_ms, max_abs_solution)
    }

    fn measure_prepared_warm_solves(
        solver: &mut NRBVP,
        repetitions: usize,
        cooldown_ms: u64,
    ) -> f64 {
        let started = Instant::now();
        for repetition in 0..repetitions {
            solver
                .try_solver_prepared()
                .expect("prepared warm solve should succeed");
            if repetition + 1 < repetitions {
                sleep_ms(cooldown_ms);
            }
        }
        started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64
    }

    fn oscillator_options_for_route(
        route: &'static str,
        assembly: BvpSymbolicAssemblyBackend,
    ) -> DampedSolverOptions {
        let config = match route {
            "Sparse" => GeneratedBackendConfig::sparse_defaults()
                .with_matrix_backend_override(MatrixBackend::SparseCol),
            "Banded" => GeneratedBackendConfig::banded_lambdify_defaults(),
            "Dense-control" => {
                GeneratedBackendConfig::default().with_matrix_backend_override(MatrixBackend::Dense)
            }
            other => panic!("unknown pure Lambdify baseline route: {other}"),
        }
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(assembly)
        // The cross-frontend baseline must not mix evaluator cost with a
        // different dispatch policy; the break-even story covers that axis.
        .with_lambdify_execution_policy(BvpLambdifyExecutionPolicy::Parallel { min_work: 0 });
        let base = match route {
            "Sparse" => DampedSolverOptions::sparse_damped(),
            "Banded" => DampedSolverOptions::banded_damped().with_banded_lambdify(),
            "Dense-control" => DampedSolverOptions::dense_damped(),
            _ => unreachable!(),
        };
        base.with_generated_backend_config(config)
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(HashMap::from([
                ("y".to_string(), 1e-7),
                ("z".to_string(), 1e-7),
            ]))
            .with_bounds(HashMap::from([
                ("y".to_string(), (-2.0, 2.0)),
                ("z".to_string(), (-2.0, 2.0)),
            ]))
            .with_max_iterations(30)
            .with_loglevel(Some("error".to_string()))
    }

    fn make_oscillator_solver(n_steps: usize, options: DampedSolverOptions) -> NRBVP {
        let t_end = std::f64::consts::FRAC_PI_2;
        let h = t_end / n_steps as f64;
        let initial_guess = DMatrix::from_fn(2, n_steps, |row, column| {
            let x = column as f64 * h;
            match row {
                0 => 1.01 * x.sin(),
                _ => 0.99 * x.cos(),
            }
        });
        let mut solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            initial_guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, 0.0f64)]),
                ("z".to_string(), vec![(0usize, 1.0f64)]),
            ]),
            0.0,
            t_end,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    #[derive(Debug)]
    struct LambdifyStageBaselineRow {
        case_label: &'static str,
        n_steps: usize,
        matrix_backend: &'static str,
        frontend: &'static str,
        setup_ms: f64,
        generation_discretization_ms: f64,
        generation_symbolic_jacobian_ms: f64,
        generation_binding_ms: f64,
        callback_residual_ms: f64,
        callback_jacobian_ms: f64,
        solve_wall_ms: f64,
        solver_total_ms: f64,
        solver_residual_ms: f64,
        solver_jacobian_ms: f64,
        linear_ms: f64,
        factorization_ms: f64,
        rhs_ms: f64,
        direct_banded_before_solve: Option<BvpDirectJacobianTelemetrySnapshot>,
        direct_banded_after_solve: Option<BvpDirectJacobianTelemetrySnapshot>,
        counters: crate::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
        solution: Vec<f64>,
    }

    fn assert_integer_trajectory_parity(
        left: &crate::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
        right: &crate::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
        case_label: &str,
        matrix_backend: &str,
    ) {
        let fields = [
            ("iterations", left.iterations, right.iterations),
            ("residual_calls", left.residual_calls, right.residual_calls),
            (
                "residual_requests",
                left.residual_requests,
                right.residual_requests,
            ),
            (
                "jacobian_requests",
                left.jacobian_requests,
                right.jacobian_requests,
            ),
            (
                "jacobian_recalculations",
                left.jacobian_recalculations,
                right.jacobian_recalculations,
            ),
            ("damping_trials", left.damping_trials, right.damping_trials),
            (
                "damping_rejections",
                left.damping_rejections,
                right.damping_rejections,
            ),
            ("linear_solves", left.linear_solves, right.linear_solves),
            ("rhs_solves", left.rhs_solves, right.rhs_solves),
            (
                "grid_refinements",
                left.grid_refinements,
                right.grid_refinements,
            ),
        ];
        for (field, legacy, atom) in fields {
            assert_eq!(
                legacy, atom,
                "{case_label} {matrix_backend} ExprLegacy/AtomView trajectory drift in {field}"
            );
        }
    }

    fn run_lambdify_stage_baseline_case<F>(
        case_label: &'static str,
        n_steps: usize,
        matrix_backend: &'static str,
        make_solver: F,
        runs: usize,
    ) -> Vec<LambdifyStageBaselineRow>
    where
        F: Fn(BvpSymbolicAssemblyBackend) -> NRBVP,
    {
        let callback_iters = 3;
        [
            ("ExprLegacy", BvpSymbolicAssemblyBackend::ExprLegacy),
            ("AtomView", BvpSymbolicAssemblyBackend::AtomView),
        ]
        .into_iter()
        .map(|(frontend, assembly)| {
            let mut solver = make_solver(assembly);
            // This is a diagnostic baseline, not a production solve. The
            // explicit Detailed mode keeps direct Banded evaluator, storage
            // and dispatch stages visible after the production default became
            // telemetry Off.
            solver.set_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Detailed);
            let setup_begin = Instant::now();
            solver
                .try_eq_generate(None, None)
                .expect("Lambdify baseline generation should succeed");
            let setup_ms = setup_begin.elapsed().as_secs_f64() * 1_000.0;
            let (callback_residual_ms, callback_jacobian_ms) =
                measure_installed_callback_runtime(&mut solver, callback_iters, runs);
            let direct_banded_before_solve =
                solver.get_statistics().telemetry.direct_banded_jacobian;

            let solve_begin = Instant::now();
            solver
                .try_solver_prepared()
                .expect("Lambdify baseline prepared solve should succeed");
            let solve_wall_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
            let solution = solver
                .get_result()
                .expect("Lambdify baseline should publish a solution")
                .as_slice()
                .to_vec();
            let stats = solver.get_statistics();
            let telemetry = stats.telemetry;
            let timings = telemetry.timings;
            let generation = telemetry.generation.unwrap_or_default();
            LambdifyStageBaselineRow {
                case_label,
                n_steps,
                matrix_backend,
                frontend,
                setup_ms,
                generation_discretization_ms: generation.discretization.as_secs_f64() * 1_000.0,
                generation_symbolic_jacobian_ms: generation.symbolic_jacobian.as_secs_f64()
                    * 1_000.0,
                generation_binding_ms: generation.runtime_binding.as_secs_f64() * 1_000.0,
                callback_residual_ms: callback_residual_ms / callback_iters as f64,
                callback_jacobian_ms: callback_jacobian_ms / callback_iters as f64,
                solve_wall_ms,
                solver_total_ms: timings.total.as_secs_f64() * 1_000.0,
                solver_residual_ms: timings.residual.as_secs_f64() * 1_000.0,
                solver_jacobian_ms: timings.jacobian.as_secs_f64() * 1_000.0,
                linear_ms: timings.linear_system.as_secs_f64() * 1_000.0,
                factorization_ms: timings.factorization.as_secs_f64() * 1_000.0,
                rhs_ms: timings.rhs_solve.as_secs_f64() * 1_000.0,
                direct_banded_before_solve,
                direct_banded_after_solve: telemetry.direct_banded_jacobian,
                counters: telemetry.counters,
                solution,
            }
        })
        .collect()
    }

    fn print_cold_generation_snapshot(
        solver: &NRBVP,
        symbolic_frontend: &str,
        runtime_route: &str,
        setup_ms: f64,
    ) {
        let stats = solver.get_statistics();
        println!(
            "[BVP Banded Lambdify] cold preparation finished: symbolic_frontend={symbolic_frontend}; runtime_route={runtime_route}; setup_ms={setup_ms:.3}; timers={:?}",
            stats.timers
        );
    }

    /// Release story for the production Banded Lambdify route.
    ///
    /// This deliberately excludes AOT. It answers the narrower migration
    /// question: does direct AtomView evaluation preserve ExprLegacy values
    /// and solver behavior, and what is the cold/warm cost on a combustion
    /// workload? The exact/analytic linear fixtures remain in
    /// `basic_correctness.rs`; this story is the larger representative case.
    #[test]
    #[ignore = "release story for ExprLegacy versus AtomView Banded Lambdify"]
    fn combustion_lambdify_exprlegacy_vs_atomview_banded_release_story() {
        let n_steps = std::env::var("BVP_LAMBDIFY_BANDED_STEPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1_000);
        let runs = story_repetitions("BVP_LAMBDIFY_BANDED_RUNS", 3);
        let callback_iters = 3;

        let mut legacy = make_combustion_solver(
            n_steps,
            banded_lambdify_options(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let mut atom = make_combustion_solver(
            n_steps,
            banded_lambdify_options(BvpSymbolicAssemblyBackend::AtomView),
        );

        println!(
            "[BVP Banded Lambdify] cold preparation start: symbolic_frontend=ExprLegacy; runtime_route=ExprLegacy+Mutex; n_steps={n_steps}"
        );
        let legacy_setup_begin = Instant::now();
        legacy
            .try_eq_generate(None, None)
            .expect("ExprLegacy Banded Lambdify generation should succeed");
        let legacy_setup_ms = legacy_setup_begin.elapsed().as_secs_f64() * 1_000.0;
        print_cold_generation_snapshot(&legacy, "ExprLegacy", "ExprLegacy+Mutex", legacy_setup_ms);

        println!(
            "[BVP Banded Lambdify] cold preparation start: symbolic_frontend=AtomView; runtime_route=AtomView+direct-no-Mutex; n_steps={n_steps}"
        );
        let atom_setup_begin = Instant::now();
        atom.try_eq_generate(None, None)
            .expect("AtomView Banded Lambdify generation should succeed");
        let atom_setup_ms = atom_setup_begin.elapsed().as_secs_f64() * 1_000.0;
        print_cold_generation_snapshot(
            &atom,
            "AtomView",
            "AtomView+direct-no-Mutex",
            atom_setup_ms,
        );

        let (residual_diff, jacobian_diff) = compare_installed_callbacks(&mut legacy, &mut atom);
        assert!(
            residual_diff < 1e-8 && jacobian_diff < 1e-8,
            "Banded Lambdify callbacks diverged: residual_diff={residual_diff:.3e}, jacobian_diff={jacobian_diff:.3e}"
        );

        let (legacy_residual_ms, legacy_jacobian_ms) =
            measure_installed_callback_runtime(&mut legacy, callback_iters, runs);
        let (atom_residual_ms, atom_jacobian_ms) =
            measure_installed_callback_runtime(&mut atom, callback_iters, runs);

        let (legacy_solve_ms, _) = solve_prepared_and_collect_solution(&mut legacy);
        let (atom_solve_ms, _) = solve_prepared_and_collect_solution(&mut atom);
        let legacy_solution = legacy
            .get_result()
            .expect("ExprLegacy Banded solve should produce a solution")
            .clone();
        let atom_solution = atom
            .get_result()
            .expect("AtomView Banded solve should produce a solution")
            .clone();
        let solution_diff = crate::numerical::BVP_Damp::test_common::max_abs_slice_diff(
            legacy_solution.as_slice(),
            atom_solution.as_slice(),
        );
        assert!(
            solution_diff < 1e-6,
            "Banded Lambdify solutions diverged: max_diff={solution_diff:.3e}"
        );

        let legacy_stats = legacy.get_statistics();
        let atom_stats = atom.get_statistics();
        println!(
            "[BVP Banded Lambdify migration] architecture={TEST_SUITE_ARCHITECTURE}; n_steps={n_steps}; runs={runs}; callback_iters={callback_iters}; preparation excluded from warm callbacks"
        );
        println!(
            "symbolic_frontend | runtime_route             | setup_ms | residual_ms | jacobian_ms | solve_ms | total_ms | iterations | jacobian_rebuilds | linear_solves"
        );
        println!("{}", "-".repeat(161));
        println!(
            "{:16} | {:24} | {:>8.3} | {:>11.3} | {:>11.3} | {:>8.3} | {:>8.3} | {:>10} | {:>17} | {:>13}",
            SymbolicTestRoute::ExprLegacy.symbolic_frontend(),
            SymbolicTestRoute::ExprLegacy.runtime_route(),
            legacy_setup_ms,
            legacy_residual_ms / callback_iters as f64,
            legacy_jacobian_ms / callback_iters as f64,
            legacy_solve_ms,
            legacy_setup_ms + legacy_solve_ms,
            legacy_stats
                .counters
                .get("number of iterations")
                .copied()
                .unwrap_or_default(),
            legacy_stats
                .counters
                .get("number of jacobians recalculations")
                .copied()
                .unwrap_or_default(),
            legacy_stats
                .counters
                .get("number of solving linear systems")
                .copied()
                .unwrap_or_default(),
        );
        println!(
            "{:16} | {:24} | {:>8.3} | {:>11.3} | {:>11.3} | {:>8.3} | {:>8.3} | {:>10} | {:>17} | {:>13}",
            SymbolicTestRoute::AtomView.symbolic_frontend(),
            SymbolicTestRoute::AtomView.runtime_route(),
            atom_setup_ms,
            atom_residual_ms / callback_iters as f64,
            atom_jacobian_ms / callback_iters as f64,
            atom_solve_ms,
            atom_setup_ms + atom_solve_ms,
            atom_stats
                .counters
                .get("number of iterations")
                .copied()
                .unwrap_or_default(),
            atom_stats
                .counters
                .get("number of jacobians recalculations")
                .copied()
                .unwrap_or_default(),
            atom_stats
                .counters
                .get("number of solving linear systems")
                .copied()
                .unwrap_or_default(),
        );
        println!(
            "{:16} | residual_diff={residual_diff:.3e} | jacobian_diff={jacobian_diff:.3e} | solution_diff={solution_diff:.3e}",
            SymbolicTestRoute::Parity.symbolic_frontend(),
        );
        println!(
            "telemetry   | ExprLegacy timers={:?} | AtomView timers={:?}",
            legacy_stats.timers, atom_stats.timers
        );

        let mut report = format!(
            "status: passed\n\narchitecture: {TEST_SUITE_ARCHITECTURE}\nn_steps: {n_steps}\nruns: {runs}\ncallback_iters: {callback_iters}\n\n\
             | symbolic_frontend | runtime_route | setup_ms | residual_ms | jacobian_ms | solve_ms | total_ms | iterations | jacobian_rebuilds | linear_solves |\n\
             |---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        );
        writeln!(
            report,
            "| ExprLegacy | ExprLegacy+Mutex | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} |",
            legacy_setup_ms,
            legacy_residual_ms / callback_iters as f64,
            legacy_jacobian_ms / callback_iters as f64,
            legacy_solve_ms,
            legacy_setup_ms + legacy_solve_ms,
            legacy_stats
                .counters
                .get("number of iterations")
                .copied()
                .unwrap_or_default(),
            legacy_stats
                .counters
                .get("number of jacobians recalculations")
                .copied()
                .unwrap_or_default(),
            legacy_stats
                .counters
                .get("number of solving linear systems")
                .copied()
                .unwrap_or_default(),
        )
        .expect("writing an in-memory test report cannot fail");
        writeln!(
            report,
            "| AtomView | AtomView+direct-no-Mutex | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} |",
            atom_setup_ms,
            atom_residual_ms / callback_iters as f64,
            atom_jacobian_ms / callback_iters as f64,
            atom_solve_ms,
            atom_setup_ms + atom_solve_ms,
            atom_stats
                .counters
                .get("number of iterations")
                .copied()
                .unwrap_or_default(),
            atom_stats
                .counters
                .get("number of jacobians recalculations")
                .copied()
                .unwrap_or_default(),
            atom_stats
                .counters
                .get("number of solving linear systems")
                .copied()
                .unwrap_or_default(),
        )
        .expect("writing an in-memory test report cannot fail");
        writeln!(
            report,
            "\n- residual_max_diff: {residual_diff:.6e}\n- jacobian_max_diff: {jacobian_diff:.6e}\n- solution_max_diff: {solution_diff:.6e}\n\n\
             ExprLegacy timers: {:?}\n\nAtomView timers: {:?}\n",
            legacy_stats.timers, atom_stats.timers
        )
        .expect("writing an in-memory test report cannot fail");
        if let Err(error) = write_test_report(
            "bvp_damp",
            "combustion_lambdify_exprlegacy_vs_atomview_banded_release_story",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write Banded Lambdify report: {error}");
        }
    }

    #[test]
    #[ignore = "diagnostic solver-facing compare for Lambdify vs AtomView C leaders on combustion-1000"]
    fn combustion_lambdify_vs_atomview_c_leaders_compare_1000() {
        aot_compare_test_report!(combustion_lambdify_vs_atomview_c_leaders_compare_1000);
        let n_steps = 2_000usize;
        let iters = 6usize;
        let samples = 3usize;

        let mut lambdify = make_combustion_solver(n_steps, lambdify_options());
        let mut gcc = make_combustion_solver(n_steps, atomview_gcc_options());
        let mut tcc = make_combustion_solver(n_steps, atomview_tcc_options());

        let lambdify_setup_begin = Instant::now();
        lambdify
            .try_eq_generate(None, None)
            .expect("lambdify generate should succeed");
        let lambdify_setup_ms = lambdify_setup_begin.elapsed().as_secs_f64() * 1_000.0;

        let gcc_setup_begin = Instant::now();
        gcc.try_eq_generate(None, None)
            .expect("AtomView + gcc generate should succeed");
        let gcc_setup_ms = gcc_setup_begin.elapsed().as_secs_f64() * 1_000.0;

        let tcc_setup_begin = Instant::now();
        tcc.try_eq_generate(None, None)
            .expect("AtomView + tcc generate should succeed");
        let tcc_setup_ms = tcc_setup_begin.elapsed().as_secs_f64() * 1_000.0;

        let (gcc_residual_diff, gcc_jacobian_diff) =
            compare_installed_callbacks(&mut lambdify, &mut gcc);
        let (tcc_residual_diff, tcc_jacobian_diff) =
            compare_installed_callbacks(&mut lambdify, &mut tcc);

        assert!(
            gcc_residual_diff < 1e-6 && gcc_jacobian_diff < 1e-6,
            "AtomView + gcc callbacks must remain numerically close to lambdify baseline"
        );
        assert!(
            tcc_residual_diff < 1e-6 && tcc_jacobian_diff < 1e-6,
            "AtomView + tcc callbacks must remain numerically close to lambdify baseline"
        );

        let (lambdify_residual_ms, lambdify_jacobian_ms) =
            measure_installed_callback_runtime(&mut lambdify, iters, samples);
        let (gcc_residual_ms, gcc_jacobian_ms) =
            measure_installed_callback_runtime(&mut gcc, iters, samples);
        let (tcc_residual_ms, tcc_jacobian_ms) =
            measure_installed_callback_runtime(&mut tcc, iters, samples);

        let lambdify_callback_total_ms = lambdify_residual_ms + lambdify_jacobian_ms;
        let gcc_callback_total_ms = gcc_residual_ms + gcc_jacobian_ms;
        let tcc_callback_total_ms = tcc_residual_ms + tcc_jacobian_ms;

        let (lambdify_solve_ms, lambdify_max_abs_solution) =
            solve_and_collect_solution(&mut lambdify);
        let (gcc_solve_ms, gcc_max_abs_solution) = solve_and_collect_solution(&mut gcc);
        let (tcc_solve_ms, tcc_max_abs_solution) = solve_and_collect_solution(&mut tcc);

        let lambdify_solution = lambdify
            .get_result()
            .expect("lambdify solution should still be present")
            .clone();
        let gcc_solution = gcc
            .get_result()
            .expect("gcc solution should still be present")
            .clone();
        let tcc_solution = tcc
            .get_result()
            .expect("tcc solution should still be present")
            .clone();

        let gcc_solution_diff = lambdify_solution
            .iter()
            .zip(gcc_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        let tcc_solution_diff = lambdify_solution
            .iter()
            .zip(tcc_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);

        assert!(
            gcc_solution_diff < 1e-6,
            "AtomView + gcc solution must stay close to lambdify"
        );
        assert!(
            tcc_solution_diff < 1e-6,
            "AtomView + tcc solution must stay close to lambdify"
        );

        let lambdify_total_ms = lambdify_setup_ms + lambdify_solve_ms;
        let gcc_total_ms = gcc_setup_ms + gcc_solve_ms;
        let tcc_total_ms = tcc_setup_ms + tcc_solve_ms;

        let rows = [
            EndToEndRow {
                variant: "Lambdify",
                sym_backend: "ExprLegacy",
                preset: "n/a",
                setup_ms: lambdify_setup_ms,
                residual_ms: lambdify_residual_ms,
                jacobian_ms: lambdify_jacobian_ms,
                callback_total_ms: lambdify_callback_total_ms,
                solve_ms: lambdify_solve_ms,
                total_ms: lambdify_total_ms,
                speedup_vs_lambdify: 1.0,
                max_abs_solution: lambdify_max_abs_solution,
                residual_diff: 0.0,
                jacobian_diff: 0.0,
                solution_diff: 0.0,
                status: "ok",
            },
            EndToEndRow {
                variant: "C-gcc",
                sym_backend: "AtomView",
                preset: "DevFastest",
                setup_ms: gcc_setup_ms,
                residual_ms: gcc_residual_ms,
                jacobian_ms: gcc_jacobian_ms,
                callback_total_ms: gcc_callback_total_ms,
                solve_ms: gcc_solve_ms,
                total_ms: gcc_total_ms,
                speedup_vs_lambdify: lambdify_callback_total_ms / gcc_callback_total_ms,
                max_abs_solution: gcc_max_abs_solution,
                residual_diff: gcc_residual_diff,
                jacobian_diff: gcc_jacobian_diff,
                solution_diff: gcc_solution_diff,
                status: "ok",
            },
            EndToEndRow {
                variant: "C-tcc",
                sym_backend: "AtomView",
                preset: "DevFastest",
                setup_ms: tcc_setup_ms,
                residual_ms: tcc_residual_ms,
                jacobian_ms: tcc_jacobian_ms,
                callback_total_ms: tcc_callback_total_ms,
                solve_ms: tcc_solve_ms,
                total_ms: tcc_total_ms,
                speedup_vs_lambdify: lambdify_callback_total_ms / tcc_callback_total_ms,
                max_abs_solution: tcc_max_abs_solution,
                residual_diff: tcc_residual_diff,
                jacobian_diff: tcc_jacobian_diff,
                solution_diff: tcc_solution_diff,
                status: "ok",
            },
        ];

        println!(
            "[BVP solver-facing compare] combustion Lambdify vs pure AtomView+C leaders, n_steps={n_steps}"
        );
        println!(
            "{:<14} | {:<11} | {:<11} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:<6}",
            "variant",
            "sym_backend",
            "preset",
            "setup_ms",
            "residual_ms",
            "jacobian_ms",
            "cb_total_ms",
            "solve_ms",
            "total_ms",
            "max_abs_sol",
            "status"
        );
        println!("{}", "-".repeat(147));
        for row in &rows {
            println!(
                "{:<14} | {:<11} | {:<11} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10.3} | {:>12.6e} | {:<6}",
                row.variant,
                row.sym_backend,
                row.preset,
                row.setup_ms,
                row.residual_ms,
                row.jacobian_ms,
                row.callback_total_ms,
                row.solve_ms,
                row.total_ms,
                row.max_abs_solution,
                row.status
            );
        }

        println!(
            "{:<14} | {:>16} | {:>16} | {:>16} | {:>18}",
            "variant", "speedup_vs_lambdify", "residual_diff", "jacobian_diff", "solution_diff"
        );
        println!("{}", "-".repeat(94));
        for row in &rows {
            println!(
                "{:<14} | {:>16.3}x | {:>16.6e} | {:>16.6e} | {:>18.6e}",
                row.variant,
                row.speedup_vs_lambdify,
                row.residual_diff,
                row.jacobian_diff,
                row.solution_diff
            );
        }
        println!(
            "[BVP solver-facing compare] combustion leaders compare finished n_steps={n_steps}"
        );
    }

    #[test]
    #[ignore = "diagnostic production-like end-to-end compare for Lambdify vs AtomView C leaders on combustion-1000"]
    fn combustion_production_like_end_to_end_compare_1000() {
        aot_compare_test_report!(combustion_production_like_end_to_end_compare_1000);
        #[derive(Debug)]
        struct Row {
            variant: &'static str,
            total_ms: f64,
            max_abs_solution: f64,
            solution_diff_vs_lambdify: f64,
            status: &'static str,
        }

        fn solve_total(n_steps: usize, options: DampedSolverOptions) -> (f64, DMatrix<f64>) {
            let total_begin = Instant::now();
            let mut solver = make_combustion_solver(n_steps, options);
            solver
                .try_eq_generate(None, None)
                .expect("production-like compare generate should succeed");
            solver
                .try_solve()
                .expect("production-like compare solve should succeed");
            let total_ms = total_begin.elapsed().as_secs_f64() * 1_000.0;
            let result = solver
                .get_result()
                .expect("production-like compare should expose a solution")
                .clone();
            (total_ms, result)
        }

        let n_steps = 2000usize;

        let (lambdify_total_ms, lambdify_solution) = solve_total(n_steps, lambdify_options());
        let (gcc_total_ms, gcc_solution) = solve_total(n_steps, atomview_gcc_options());
        let (tcc_total_ms, tcc_solution) = solve_total(n_steps, atomview_tcc_options());

        let gcc_solution_diff = lambdify_solution
            .iter()
            .zip(gcc_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        let tcc_solution_diff = lambdify_solution
            .iter()
            .zip(tcc_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);

        assert!(
            gcc_solution_diff < 1e-6,
            "production-like gcc result must stay close to lambdify baseline"
        );
        assert!(
            tcc_solution_diff < 1e-6,
            "production-like tcc result must stay close to lambdify baseline"
        );

        let rows = [
            Row {
                variant: "Lambdify",
                total_ms: lambdify_total_ms,
                max_abs_solution: lambdify_solution
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max),
                solution_diff_vs_lambdify: 0.0,
                status: "ok",
            },
            Row {
                variant: "C-gcc",
                total_ms: gcc_total_ms,
                max_abs_solution: gcc_solution
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max),
                solution_diff_vs_lambdify: gcc_solution_diff,
                status: "ok",
            },
            Row {
                variant: "C-tcc",
                total_ms: tcc_total_ms,
                max_abs_solution: tcc_solution
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max),
                solution_diff_vs_lambdify: tcc_solution_diff,
                status: "ok",
            },
        ];

        println!(
            "[BVP production-like end-to-end] combustion Lambdify vs AtomView C leaders, n_steps={n_steps}"
        );
        println!(
            "{:<12} | {:>12} | {:>16} | {:>24} | {:<6}",
            "variant", "total_ms", "max_abs_solution", "solution_diff_vs_lambdify", "status"
        );
        println!("{}", "-".repeat(86));
        for row in rows {
            println!(
                "{:<12} | {:>12.3} | {:>16.6e} | {:>24.6e} | {:<6}",
                row.variant,
                row.total_ms,
                row.max_abs_solution,
                row.solution_diff_vs_lambdify,
                row.status
            );
        }
        println!("[BVP production-like end-to-end] finished combustion compare n_steps={n_steps}");
    }

    #[test]
    #[ignore = "diagnostic break-even compare for Lambdify vs AtomView C-tcc on combustion-2000"]
    fn combustion_break_even_lambdify_vs_atomview_ctcc_2000() {
        aot_compare_test_report!(combustion_break_even_lambdify_vs_atomview_ctcc_2000);
        #[derive(Debug)]
        struct Row {
            variant: &'static str,
            setup_ms: f64,
            solve_ms: f64,
            warm_solve_ms: f64,
            total_one_solve_ms: f64,
            runtime_share: f64,
        }

        let protocol = AotStoryProtocol::from_env(2000, 5);
        protocol
            .validate()
            .expect("AOT/Lambdify break-even protocol should be valid");
        let n_steps = protocol.n_steps;
        println!(
            "[BVP break-even] protocol: {}; cold E2E and warm prepared solves are reported separately",
            protocol.summary()
        );
        let mut lambdify = make_combustion_solver(n_steps, lambdify_options());
        let mut tcc = make_combustion_solver(n_steps, atomview_tcc_options());

        let lambdify_setup_begin = Instant::now();
        lambdify
            .try_eq_generate(None, None)
            .expect("break-even lambdify generate should succeed");
        let lambdify_setup_ms = lambdify_setup_begin.elapsed().as_secs_f64() * 1_000.0;
        let (lambdify_solve_ms, _) = solve_and_collect_solution(&mut lambdify);
        let lambdify_warm_solve_ms = measure_prepared_warm_solves(
            &mut lambdify,
            protocol.warm_repetitions,
            protocol.warm_cooldown_ms,
        );
        let lambdify_solution = lambdify
            .get_result()
            .expect("lambdify break-even test should produce a solution")
            .clone();

        let tcc_setup_begin = Instant::now();
        tcc.try_eq_generate(None, None)
            .expect("break-even AtomView+tcc generate should succeed");
        let tcc_setup_ms = tcc_setup_begin.elapsed().as_secs_f64() * 1_000.0;
        let (tcc_solve_ms, _) = solve_and_collect_solution(&mut tcc);
        let tcc_warm_solve_ms = measure_prepared_warm_solves(
            &mut tcc,
            protocol.warm_repetitions,
            protocol.warm_cooldown_ms,
        );
        let tcc_solution = tcc
            .get_result()
            .expect("AtomView+tcc break-even test should produce a solution")
            .clone();

        let solution_diff = lambdify_solution
            .iter()
            .zip(tcc_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(
            solution_diff < 1e-6,
            "break-even compare requires numerically close solutions"
        );

        let lambdify_total_one_solve_ms = lambdify_setup_ms + lambdify_solve_ms;
        let tcc_total_one_solve_ms = tcc_setup_ms + tcc_solve_ms;
        let extra_bootstrap_ms = (tcc_setup_ms - lambdify_setup_ms).max(0.0);
        let runtime_gain_ms_per_solve = (lambdify_warm_solve_ms - tcc_warm_solve_ms).max(0.0);
        let break_even_solves = if runtime_gain_ms_per_solve > 0.0 {
            (extra_bootstrap_ms / runtime_gain_ms_per_solve).ceil()
        } else {
            f64::INFINITY
        };

        let rows = [
            Row {
                variant: "Lambdify",
                setup_ms: lambdify_setup_ms,
                solve_ms: lambdify_solve_ms,
                warm_solve_ms: lambdify_warm_solve_ms,
                total_one_solve_ms: lambdify_total_one_solve_ms,
                runtime_share: lambdify_solve_ms / lambdify_total_one_solve_ms,
            },
            Row {
                variant: "C-tcc",
                setup_ms: tcc_setup_ms,
                solve_ms: tcc_solve_ms,
                warm_solve_ms: tcc_warm_solve_ms,
                total_one_solve_ms: tcc_total_one_solve_ms,
                runtime_share: tcc_solve_ms / tcc_total_one_solve_ms,
            },
        ];

        println!("[BVP break-even] combustion Lambdify vs AtomView+C-tcc, n_steps={n_steps}");
        println!(
            "{:<12} | {:>12} | {:>14} | {:>16} | {:>18} | {:>14}",
            "variant",
            "setup_ms",
            "cold_solve_ms",
            "warm_solve_ms",
            "total_one_solve_ms",
            "runtime_share"
        );
        println!("{}", "-".repeat(104));
        for row in rows {
            println!(
                "{:<12} | {:>12.3} | {:>14.3} | {:>16.3} | {:>18.3} | {:>13.3}%",
                row.variant,
                row.setup_ms,
                row.solve_ms,
                row.warm_solve_ms,
                row.total_one_solve_ms,
                row.runtime_share * 100.0
            );
        }

        println!(
            "{:<24} | {:>16.3}",
            "extra_bootstrap_ms_vs_lambdify", extra_bootstrap_ms
        );
        println!(
            "{:<24} | {:>16.3}",
            "warm_runtime_gain_ms_per_solve", runtime_gain_ms_per_solve
        );
        if break_even_solves.is_finite() {
            println!("{:<24} | {:>16.0}", "break_even_solves", break_even_solves);
        } else {
            println!("{:<24} | {:>16}", "break_even_solves", "never");
        }
        println!("{:<24} | {:>16.6e}", "solution_diff", solution_diff);
        println!("[BVP break-even] finished combustion break-even compare n_steps={n_steps}");
    }

    /// Measures the current pure-Lambdify Sequential/Parallel break-even.
    ///
    /// The cold callback pair deliberately remains separate from the warm
    /// callback average: the first Parallel dispatch may pay worker-pool
    /// startup, while repeated calls measure the steady-state work. The
    /// solver counters are part of the report so a timing difference cannot
    /// be mistaken for an algorithmic trajectory change. `Auto` is now a
    /// public BVP Lambdify policy, but this story intentionally measures only
    /// the explicit Sequential/Parallel break-even pair.
    #[test]
    #[ignore = "release-only Sequential/Parallel Lambdify break-even story"]
    fn lambdify_sequential_parallel_break_even_story() {
        let warm_samples = std::env::var("BVP_LAMBDIFY_BREAK_EVEN_SAMPLES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(5);
        let workloads = [
            ("oscillator-128", "oscillator", 128usize),
            ("combustion-1000", "combustion", 1_000usize),
        ];
        let mut report = format!(
            "status: passed\n\nrecorded_on: 2026-09-20\nfrontend: AtomView\npolicies: Sequential vs Parallel {{ min_work: 0 }}\nwarm_callback_samples: {warm_samples}\nAuto: exposed as a deterministic scalar-work policy; this report does not benchmark its threshold.\n\n"
        );
        report.push_str(
            "| workload | n_steps | matrix | setup_seq_ms | setup_par_ms | cold_pair_seq_ms | cold_pair_par_ms | warm_pair_seq_ms | warm_pair_par_ms | solve_seq_ms | solve_par_ms | steady_gain_ms | cold_overhead_ms | break_even_pairs | iterations | jacobian_recalculations | refinements | damping_rejections | max_solution_diff |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        println!(
            "[BVP Lambdify break-even] frontend=AtomView; warm_samples={warm_samples}; Auto=exposed-not-benchmarked"
        );
        println!(
            "workload | n_steps | matrix | setup_seq_ms | setup_par_ms | cold_seq_ms | cold_par_ms | warm_seq_ms | warm_par_ms | solve_seq_ms | solve_par_ms | steady_gain_ms | cold_overhead_ms | break_even_pairs | iterations | jac_recalculations | refinements | damping_rejections | max_solution_diff"
        );
        println!("{}", "-".repeat(240));

        for (workload_label, workload_kind, n_steps) in workloads {
            for route in ["Sparse", "Banded"] {
                let build = |policy| {
                    let mut options = match workload_kind {
                        "combustion" => combustion_lambdify_options_for_route(
                            route,
                            BvpSymbolicAssemblyBackend::AtomView,
                        ),
                        "oscillator" => oscillator_options_for_route(
                            route,
                            BvpSymbolicAssemblyBackend::AtomView,
                        ),
                        _ => unreachable!(),
                    };
                    let config = options
                        .generated_backend_config
                        .clone()
                        .with_lambdify_execution_policy(policy);
                    options = options.with_generated_backend_config(config);
                    let mut solver = match workload_kind {
                        "combustion" => make_combustion_solver(n_steps, options),
                        "oscillator" => make_oscillator_solver(n_steps, options),
                        _ => unreachable!(),
                    };
                    solver.dont_save_log(true);
                    solver
                };

                let mut sequential = build(BvpLambdifyExecutionPolicy::Sequential);
                let mut parallel = build(BvpLambdifyExecutionPolicy::Parallel { min_work: 0 });

                let sequential_setup_begin = Instant::now();
                sequential
                    .try_eq_generate(None, None)
                    .expect("Sequential AtomView preparation should succeed");
                let sequential_setup_ms = sequential_setup_begin.elapsed().as_secs_f64() * 1_000.0;
                let parallel_setup_begin = Instant::now();
                parallel
                    .try_eq_generate(None, None)
                    .expect("Parallel AtomView preparation should succeed");
                let parallel_setup_ms = parallel_setup_begin.elapsed().as_secs_f64() * 1_000.0;

                let (sequential_cold_ms, sequential_warm_ms) =
                    measure_callback_cold_and_warm(&mut sequential, warm_samples);
                let (parallel_cold_ms, parallel_warm_ms) =
                    measure_callback_cold_and_warm(&mut parallel, warm_samples);
                let (sequential_solve_ms, _) = solve_prepared_and_collect_solution(&mut sequential);
                let (parallel_solve_ms, _) = solve_prepared_and_collect_solution(&mut parallel);
                let sequential_solution = sequential
                    .get_result()
                    .expect("Sequential break-even solve should publish a solution")
                    .as_slice()
                    .to_vec();
                let parallel_solution = parallel
                    .get_result()
                    .expect("Parallel break-even solve should publish a solution")
                    .as_slice()
                    .to_vec();
                let max_solution_diff = crate::numerical::BVP_Damp::test_common::max_abs_slice_diff(
                    &sequential_solution,
                    &parallel_solution,
                );
                assert!(
                    max_solution_diff < 1e-6,
                    "{workload_label} {route} Sequential/Parallel solution drift: {max_solution_diff:.3e}"
                );

                let sequential_counters = sequential.get_statistics().telemetry.counters;
                let parallel_counters = parallel.get_statistics().telemetry.counters;
                assert_integer_trajectory_parity(
                    &sequential_counters,
                    &parallel_counters,
                    workload_label,
                    route,
                );
                let steady_gain_ms = sequential_warm_ms - parallel_warm_ms;
                let cold_overhead_ms = parallel_cold_ms - sequential_cold_ms;
                let break_even_pairs = if cold_overhead_ms <= 0.0 {
                    1.0
                } else if steady_gain_ms > 0.0 {
                    (cold_overhead_ms / steady_gain_ms).ceil()
                } else {
                    f64::INFINITY
                };

                println!(
                    "{workload_label} | {n_steps:>7} | {route:14} | {sequential_setup_ms:>13.3} | {parallel_setup_ms:>13.3} | {sequential_cold_ms:>12.3} | {parallel_cold_ms:>12.3} | {sequential_warm_ms:>12.3} | {parallel_warm_ms:>12.3} | {sequential_solve_ms:>12.3} | {parallel_solve_ms:>12.3} | {steady_gain_ms:>14.3} | {cold_overhead_ms:>15.3} | {break_even_pairs:>15.0} | {:>10} | {:>17} | {:>11} | {:>18} | {max_solution_diff:.3e}",
                    sequential_counters.iterations,
                    sequential_counters.jacobian_recalculations,
                    sequential_counters.grid_refinements,
                    sequential_counters.damping_rejections,
                );
                writeln!(
                    report,
                    "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.0} | {} | {} | {} | {} | {:.6e} |",
                    workload_label,
                    n_steps,
                    route,
                    sequential_setup_ms,
                    parallel_setup_ms,
                    sequential_cold_ms,
                    parallel_cold_ms,
                    sequential_warm_ms,
                    parallel_warm_ms,
                    sequential_solve_ms,
                    parallel_solve_ms,
                    steady_gain_ms,
                    cold_overhead_ms,
                    break_even_pairs,
                    sequential_counters.iterations,
                    sequential_counters.jacobian_recalculations,
                    sequential_counters.grid_refinements,
                    sequential_counters.damping_rejections,
                    max_solution_diff,
                )
                .expect("writing break-even report cannot fail");
            }
        }

        report.push_str(
            "\nInterpretation: break_even_pairs counts callback pairs required to amortize the first Parallel cold-dispatch overhead using the measured warm callback gain. It is not a claim about Auto; Auto requires a separately calibrated policy and must be rechecked against full solver wall-clock.\n",
        );
        if let Err(error) = write_test_report(
            "bvp_damp",
            "lambdify_sequential_parallel_break_even_story",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write Lambdify break-even report: {error}");
        }
    }

    /// Pre-optimization pure-Lambdify stage baseline.
    ///
    /// This runs several workloads before changing callback, factor-owner or
    /// buffer ownership. It preserves a dated map of where time and calls are
    /// spent so later optimization claims can be localized.
    #[test]
    #[ignore = "release-only pre-optimization Lambdify stage baseline corpus"]
    fn lambdify_stage_baseline_corpus() {
        let runs = std::env::var("BVP_LAMBDIFY_BASELINE_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(3);
        let mut rows = Vec::new();
        rows.extend(run_lambdify_stage_baseline_case(
            "combustion-1000",
            1_000,
            "Sparse",
            |assembly| {
                make_combustion_solver(
                    1_000,
                    combustion_lambdify_options_for_route("Sparse", assembly),
                )
            },
            runs,
        ));
        rows.extend(run_lambdify_stage_baseline_case(
            "combustion-1000",
            1_000,
            "Banded",
            |assembly| {
                make_combustion_solver(
                    1_000,
                    combustion_lambdify_options_for_route("Banded", assembly),
                )
            },
            runs,
        ));
        rows.extend(run_lambdify_stage_baseline_case(
            "combustion-3000",
            3_000,
            "Sparse",
            |assembly| {
                make_combustion_solver(
                    3_000,
                    combustion_lambdify_options_for_route("Sparse", assembly),
                )
            },
            runs,
        ));
        rows.extend(run_lambdify_stage_baseline_case(
            "combustion-3000",
            3_000,
            "Banded",
            |assembly| {
                make_combustion_solver(
                    3_000,
                    combustion_lambdify_options_for_route("Banded", assembly),
                )
            },
            runs,
        ));
        rows.extend(run_lambdify_stage_baseline_case(
            "oscillator-128",
            128,
            "Dense-control",
            |assembly| {
                make_oscillator_solver(128, oscillator_options_for_route("Dense-control", assembly))
            },
            runs,
        ));
        rows.extend(run_lambdify_stage_baseline_case(
            "oscillator-1000",
            1_000,
            "Sparse",
            |assembly| {
                make_oscillator_solver(1_000, oscillator_options_for_route("Sparse", assembly))
            },
            runs,
        ));
        rows.extend(run_lambdify_stage_baseline_case(
            "oscillator-1000",
            1_000,
            "Banded",
            |assembly| {
                make_oscillator_solver(1_000, oscillator_options_for_route("Banded", assembly))
            },
            runs,
        ));

        let mut report = format!(
            "status: passed\n\nrecorded_on: 2026-09-20\nruns: {runs}\ncallback_iters: 3\nroute: pure Lambdify Sparse/Banded; Dense-control on oscillator only\nexecution_policy: Parallel {{ min_work: 0 }} for ExprLegacy and AtomView; Sequential/Auto are covered by the separate break-even story\ntelemetry_mode: Detailed for diagnostic stage breakdown only; production default remains Off\ncomparison_policy: ExprLegacy must not regress against its archived release baseline; AtomView must match ExprLegacy for correctness and must not regress against its own archived AtomView release baseline\n\n"
        );
        report.push_str(
            "| case | n_steps | matrix | frontend | cold_setup_ms | discretization_ms | symbolic_jacobian_ms | binding_ms | callback_residual_ms | callback_jacobian_ms | solve_wall_ms | solver_total_ms | solver_residual_ms | solver_jacobian_ms | linear_ms | factor_ms | rhs_ms | iterations | residual_calls | jacobian_calls | factorizations | cache_hits | conversions | copies |\n|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        println!(
            "[BVP Lambdify stage baseline] recorded_on=2026-09-20; runs={runs}; callback_iters=3; AOT excluded"
        );
        println!(
            "case | n_steps | matrix | frontend | cold_setup_ms | callback_R_ms | callback_J_ms | solve_wall_ms | solver_total_ms | solver_R_ms | solver_J_ms | linear_ms | factor_ms | rhs_ms | iterations | residual_calls | jacobian_calls | factorizations | cache_hits | conversions | copies"
        );
        println!("{}", "-".repeat(250));

        for row in &rows {
            let counters = &row.counters;
            println!(
                "{} | {:>7} | {:14} | {:10} | {:>14.3} | {:>13.3} | {:>13.3} | {:>14.3} | {:>15.3} | {:>11.3} | {:>11.3} | {:>9.3} | {:>9.3} | {:>7.3} | {:>10} | {:>14} | {:>14} | {:>14} | {:>10} | {:>11} | {:>6}",
                row.case_label,
                row.n_steps,
                row.matrix_backend,
                row.frontend,
                row.setup_ms,
                row.callback_residual_ms,
                row.callback_jacobian_ms,
                row.solve_wall_ms,
                row.solver_total_ms,
                row.solver_residual_ms,
                row.solver_jacobian_ms,
                row.linear_ms,
                row.factorization_ms,
                row.rhs_ms,
                counters.iterations,
                counters.residual_calls,
                counters.jacobian_requests,
                counters.factorizations,
                counters.factorization_cache_hits,
                counters.conversions,
                counters.copies,
            );
            writeln!(
                report,
                "| {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} |",
                row.case_label,
                row.n_steps,
                row.matrix_backend,
                row.frontend,
                row.setup_ms,
                row.generation_discretization_ms,
                row.generation_symbolic_jacobian_ms,
                row.generation_binding_ms,
                row.callback_residual_ms,
                row.callback_jacobian_ms,
                row.solve_wall_ms,
                row.solver_total_ms,
                row.solver_residual_ms,
                row.solver_jacobian_ms,
                row.linear_ms,
                row.factorization_ms,
                row.rhs_ms,
                counters.iterations,
                counters.residual_calls,
                counters.jacobian_requests,
                counters.factorizations,
                counters.factorization_cache_hits,
                counters.conversions,
                counters.copies,
            )
            .expect("writing an in-memory baseline report cannot fail");
        }

        report.push_str(
            "\n## Integer trajectory counters\n\n| case | n_steps | matrix | frontend | iterations | residual_calls | residual_requests | jacobian_requests | jacobian_recalculations | residual_chunks | jacobian_chunks | damping_trials | damping_rejections | linear_solves | factorizations | factor_cache_hits | factor_invalidations | rhs_solves | grid_refinements | conversions | copies |\n|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        println!(
            "\n[BVP Lambdify stage baseline] integer trajectory counters; one solver run per row"
        );
        println!(
            "case | n_steps | matrix | frontend | iterations | residual_calls | residual_requests | jacobian_requests | jacobian_recalculations | residual_chunks | jacobian_chunks | damping_trials | damping_rejections | linear_solves | factorizations | cache_hits | invalidations | rhs_solves | refinements | conversions | copies"
        );
        println!("{}", "-".repeat(250));
        for row in &rows {
            let counters = &row.counters;
            println!(
                "{} | {:>7} | {:14} | {:10} | {:>10} | {:>14} | {:>17} | {:>17} | {:>23} | {:>15} | {:>15} | {:>14} | {:>18} | {:>13} | {:>14} | {:>16} | {:>18} | {:>10} | {:>17} | {:>11} | {:>6}",
                row.case_label,
                row.n_steps,
                row.matrix_backend,
                row.frontend,
                counters.iterations,
                counters.residual_calls,
                counters.residual_requests,
                counters.jacobian_requests,
                counters.jacobian_recalculations,
                counters.residual_chunks,
                counters.jacobian_chunks,
                counters.damping_trials,
                counters.damping_rejections,
                counters.linear_solves,
                counters.factorizations,
                counters.factorization_cache_hits,
                counters.factorization_invalidations,
                counters.rhs_solves,
                counters.grid_refinements,
                counters.conversions,
                counters.copies,
            );
            writeln!(
                report,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                row.case_label,
                row.n_steps,
                row.matrix_backend,
                row.frontend,
                counters.iterations,
                counters.residual_calls,
                counters.residual_requests,
                counters.jacobian_requests,
                counters.jacobian_recalculations,
                counters.residual_chunks,
                counters.jacobian_chunks,
                counters.damping_trials,
                counters.damping_rejections,
                counters.linear_solves,
                counters.factorizations,
                counters.factorization_cache_hits,
                counters.factorization_invalidations,
                counters.rhs_solves,
                counters.grid_refinements,
                counters.conversions,
                counters.copies,
            )
            .expect("writing integer baseline counters cannot fail");
        }

        report.push_str(
            "\n## Componentwise solution parity\n\n| case | matrix | max_abs_diff | limit |\n|---|---|---:|---:|\n",
        );
        for pair in rows.chunks_exact(2) {
            assert_eq!(pair[0].case_label, pair[1].case_label);
            assert_eq!(pair[0].matrix_backend, pair[1].matrix_backend);
            assert_integer_trajectory_parity(
                &pair[0].counters,
                &pair[1].counters,
                pair[0].case_label,
                pair[0].matrix_backend,
            );
            println!(
                "[BVP Lambdify stage baseline trajectory parity] case={} matrix={} iterations={} jacobian_recalculations={} refinements={} damping_rejections={}",
                pair[0].case_label,
                pair[0].matrix_backend,
                pair[0].counters.iterations,
                pair[0].counters.jacobian_recalculations,
                pair[0].counters.grid_refinements,
                pair[0].counters.damping_rejections,
            );
            let diff = crate::numerical::BVP_Damp::test_common::max_abs_slice_diff(
                &pair[0].solution,
                &pair[1].solution,
            );
            // This ignored story measures complete solver trajectories at the
            // configured production tolerance. Different factorization/update
            // orders can leave a tolerance-scale final-state drift even when
            // callback values are identical. Keep the strict callback,
            // Jacobian and small-problem solution gate in the cross-product
            // test; this baseline uses a separate diagnostic guard.
            let parity_limit = 1e-6;
            assert!(
                diff < parity_limit,
                "{} {} ExprLegacy/AtomView solution drift: {diff:.3e} (limit {parity_limit:.1e})",
                pair[0].case_label,
                pair[0].matrix_backend,
            );
            println!(
                "[BVP Lambdify stage baseline parity] case={} matrix={} max_abs_diff={diff:.3e} limit={parity_limit:.1e}",
                pair[0].case_label, pair[0].matrix_backend,
            );
            writeln!(
                report,
                "| {} | {} | {:.6e} | {:.1e} |",
                pair[0].case_label, pair[0].matrix_backend, diff, parity_limit
            )
            .expect("writing an in-memory parity report cannot fail");
        }

        report.push_str(
            "\n## Direct Banded Jacobian callback breakdown\n\n\
             The `before_solve` snapshot covers the explicit callback sample;\n\
             `solver_delta` is the difference between the post-solve and\n\
             pre-solve snapshots. This separates callback work from the\n\
             Newton solve and localizes evaluator/storage/allocation regressions.\n\n\
             | case | matrix | frontend | sample_calls | solver_calls | sample_argument_prepare_ms | solver_argument_prepare_ms | sample_evaluator_ms | solver_evaluator_ms | sample_storage_write_ms | solver_storage_write_ms | sample_assembly_alloc_ms | solver_assembly_alloc_ms | sample_parallel | solver_parallel | sample_sequential | solver_sequential | evaluator_calls_delta | storage_writes_delta |\n\
             |---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        for row in &rows {
            let (Some(before), Some(after)) = (
                row.direct_banded_before_solve,
                row.direct_banded_after_solve,
            ) else {
                continue;
            };
            let solver_calls = after.calls.saturating_sub(before.calls);
            let solver_evaluator = after
                .evaluator_elapsed
                .saturating_sub(before.evaluator_elapsed)
                .as_secs_f64()
                * 1_000.0;
            let solver_storage_write = after
                .storage_write_elapsed
                .saturating_sub(before.storage_write_elapsed)
                .as_secs_f64()
                * 1_000.0;
            let solver_assembly_alloc = after
                .assembly_alloc_elapsed
                .saturating_sub(before.assembly_alloc_elapsed)
                .as_secs_f64()
                * 1_000.0;
            writeln!(
                report,
                "| {} | {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} |",
                row.case_label,
                row.matrix_backend,
                row.frontend,
                before.calls,
                solver_calls,
                before.argument_prepare_elapsed.as_secs_f64() * 1_000.0,
                after
                    .argument_prepare_elapsed
                    .saturating_sub(before.argument_prepare_elapsed)
                    .as_secs_f64()
                    * 1_000.0,
                before.evaluator_elapsed.as_secs_f64() * 1_000.0,
                solver_evaluator,
                before.storage_write_elapsed.as_secs_f64() * 1_000.0,
                solver_storage_write,
                before.assembly_alloc_elapsed.as_secs_f64() * 1_000.0,
                solver_assembly_alloc,
                before.parallel_dispatches,
                after.parallel_dispatches.saturating_sub(before.parallel_dispatches),
                before.sequential_dispatches,
                after.sequential_dispatches.saturating_sub(before.sequential_dispatches),
                after.evaluator_calls.saturating_sub(before.evaluator_calls),
                after.storage_writes.saturating_sub(before.storage_writes),
            )
            .expect("writing an in-memory direct Banded report cannot fail");
        }

        report.push_str(
            "\nInterpretation: this is the pre-optimization control map. Do not treat it as a final performance ranking; rerun it after PreparedPlan ownership, fixed-CSC, reusable buffers and AtomView chunking changes.\n",
        );
        if let Err(error) = write_test_report("bvp_damp", "lambdify_stage_baseline_corpus", &report)
        {
            eprintln!("[BVP test report] unable to write Lambdify baseline report: {error}");
        }
    }
}
