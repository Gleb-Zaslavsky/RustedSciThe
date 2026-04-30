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

mod tests {
    use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotExecutionPolicy, GeneratedBackendConfig,
    };
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::hint::black_box;
    use std::time::Instant;

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

    fn uniform_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
        DMatrix::from_column_slice(
            variable_count,
            n_steps,
            DVector::from_element(variable_count * n_steps, value).as_slice(),
        )
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

    #[test]
    #[ignore = "diagnostic solver-facing compare for Lambdify vs AtomView C leaders on combustion-1000"]
    fn combustion_lambdify_vs_atomview_c_leaders_compare_1000() {
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
        #[derive(Debug)]
        struct Row {
            variant: &'static str,
            setup_ms: f64,
            solve_ms: f64,
            total_one_solve_ms: f64,
            runtime_share: f64,
        }

        let n_steps = 2000usize;
        let mut lambdify = make_combustion_solver(n_steps, lambdify_options());
        let mut tcc = make_combustion_solver(n_steps, atomview_tcc_options());

        let lambdify_setup_begin = Instant::now();
        lambdify
            .try_eq_generate(None, None)
            .expect("break-even lambdify generate should succeed");
        let lambdify_setup_ms = lambdify_setup_begin.elapsed().as_secs_f64() * 1_000.0;
        let (lambdify_solve_ms, _) = solve_and_collect_solution(&mut lambdify);
        let lambdify_solution = lambdify
            .get_result()
            .expect("lambdify break-even test should produce a solution")
            .clone();

        let tcc_setup_begin = Instant::now();
        tcc.try_eq_generate(None, None)
            .expect("break-even AtomView+tcc generate should succeed");
        let tcc_setup_ms = tcc_setup_begin.elapsed().as_secs_f64() * 1_000.0;
        let (tcc_solve_ms, _) = solve_and_collect_solution(&mut tcc);
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
        let runtime_gain_ms_per_solve = (lambdify_solve_ms - tcc_solve_ms).max(0.0);
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
                total_one_solve_ms: lambdify_total_one_solve_ms,
                runtime_share: lambdify_solve_ms / lambdify_total_one_solve_ms,
            },
            Row {
                variant: "C-tcc",
                setup_ms: tcc_setup_ms,
                solve_ms: tcc_solve_ms,
                total_one_solve_ms: tcc_total_one_solve_ms,
                runtime_share: tcc_solve_ms / tcc_total_one_solve_ms,
            },
        ];

        println!("[BVP break-even] combustion Lambdify vs AtomView+C-tcc, n_steps={n_steps}");
        println!(
            "{:<12} | {:>12} | {:>12} | {:>18} | {:>14}",
            "variant", "setup_ms", "solve_ms", "total_one_solve_ms", "runtime_share"
        );
        println!("{}", "-".repeat(82));
        for row in rows {
            println!(
                "{:<12} | {:>12.3} | {:>12.3} | {:>18.3} | {:>13.3}%",
                row.variant,
                row.setup_ms,
                row.solve_ms,
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
            "runtime_gain_ms_per_solve", runtime_gain_ms_per_solve
        );
        if break_even_solves.is_finite() {
            println!("{:<24} | {:>16.0}", "break_even_solves", break_even_solves);
        } else {
            println!("{:<24} | {:>16}", "break_even_solves", "never");
        }
        println!("{:<24} | {:>16.6e}", "solution_diff", solution_diff);
        println!("[BVP break-even] finished combustion break-even compare n_steps={n_steps}");
    }
}
