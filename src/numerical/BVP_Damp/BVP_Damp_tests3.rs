#![cfg(test)]

mod tests {
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, BuildDampedSolverRequest,
    };
    use crate::numerical::Examples_and_utils::NonlinEquation;
    use crate::symbolic::codegen_aot_runtime_link::{
        LinkedResidualChunk, LinkedSparseAotBackend, LinkedSparseJacobianChunk,
        register_linked_sparse_backend, unregister_linked_sparse_backend,
    };
    use crate::symbolic::codegen_backend_selection::SelectedBackendKind;
    use crate::symbolic::codegen_orchestrator::{ParallelExecutorConfig, ParallelFallbackPolicy};
    use crate::symbolic::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen_tasks::SparseChunkingStrategy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::{
        BvpBackendIntegrationError, BvpSparseSolverBundle, Jacobian,
    };
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    #[derive(Debug)]
    struct RuntimeTuningRow {
        label: &'static str,
        n_steps: usize,
        bootstrap_ms: f64,
        solve_ms: f64,
        speedup_vs_seq: f64,
        max_diff_vs_seq: f64,
    }

    struct LinkedBackendGuard {
        problem_key: String,
    }

    impl Drop for LinkedBackendGuard {
        fn drop(&mut self) {
            let _ = unregister_linked_sparse_backend(&self.problem_key);
        }
    }

    fn uniform_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
        DMatrix::from_column_slice(
            variable_count,
            n_steps,
            DVector::from_element(variable_count * n_steps, value).as_slice(),
        )
    }

    fn sparse_parallel_policy() -> AotExecutionPolicy {
        AotExecutionPolicy::Parallel(ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: Some(8),
            max_sparse_jobs: Some(8),
            fallback_policy: ParallelFallbackPolicy::Never,
        })
    }

    fn register_linked_backend_from_sparse_bundle(
        bundle: &BvpSparseSolverBundle,
    ) -> LinkedBackendGuard {
        let prepared = bundle.execution.selected().prepared_problem.clone();
        let prepared_sparse = prepared.as_prepared_problem();
        let problem_key = prepared.problem_key();
        let input_name_strings = prepared
            .param_names
            .iter()
            .chain(prepared.variable_names.iter())
            .cloned()
            .collect::<Vec<_>>();
        let residual_names = input_name_strings.clone();
        let jacobian_names = input_name_strings.clone();
        let residual_exprs = prepared.residuals.clone();
        let sparse_exprs = prepared.sparse_entries.clone();
        let residual_len = prepared.shape.0;
        let shape = prepared.shape;
        let nnz = sparse_exprs.len();

        let residual_chunks = prepared_sparse
            .residual_plan
            .chunks
            .iter()
            .map(|chunk| {
                let names = input_name_strings.clone();
                let exprs = chunk.residuals.iter().cloned().collect::<Vec<_>>();
                LinkedResidualChunk::new(
                    chunk.output_offset,
                    exprs.len(),
                    Arc::new(move |args, out| {
                        let input_names =
                            names.iter().map(|name| name.as_str()).collect::<Vec<_>>();
                        for (slot, expr) in out.iter_mut().zip(exprs.iter()) {
                            *slot = expr.eval_expression(&input_names, args);
                        }
                    }),
                )
            })
            .collect::<Vec<_>>();

        let jacobian_chunks = prepared_sparse
            .jacobian_plan
            .chunks
            .iter()
            .map(|chunk| {
                let names = input_name_strings.clone();
                let entries = chunk
                    .entries
                    .iter()
                    .map(|entry| (entry.row, entry.col, entry.expr.clone()))
                    .collect::<Vec<_>>();
                LinkedSparseJacobianChunk::new(
                    chunk.value_offset,
                    entries.len(),
                    Arc::new(move |args, out| {
                        let input_names =
                            names.iter().map(|name| name.as_str()).collect::<Vec<_>>();
                        for (slot, (_, _, expr)) in out.iter_mut().zip(entries.iter()) {
                            *slot = expr.eval_expression(&input_names, args);
                        }
                    }),
                )
            })
            .collect::<Vec<_>>();

        let backend = LinkedSparseAotBackend::new(
            problem_key.clone(),
            residual_len,
            shape,
            nnz,
            Arc::new(move |args, out| {
                let input_names = residual_names
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>();
                for (slot, expr) in out.iter_mut().zip(residual_exprs.iter()) {
                    *slot = expr.eval_expression(&input_names, args);
                }
            }),
            Arc::new(move |args, out| {
                let input_names = jacobian_names
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>();
                for (slot, (_, _, expr)) in out.iter_mut().zip(sparse_exprs.iter()) {
                    *slot = expr.eval_expression(&input_names, args);
                }
            }),
        )
        .with_chunked_evaluators(residual_chunks, jacobian_chunks);

        register_linked_sparse_backend(backend);
        LinkedBackendGuard { problem_key }
    }

    fn sparse_bundle_from_solver_request(
        solver: &mut NRBVP,
    ) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
        let request = solver.build_solver_request(None, None);
        let jacobian = Jacobian::new();
        match (
            request.aot_chunking_policy.residual,
            request.aot_chunking_policy.sparse_jacobian,
        ) {
            (None, None) => jacobian.try_generate_sparse_solver_bundle_with_backend_selection(
                request.eq_system,
                request.values,
                request.arg,
                None,
                request.t0,
                None,
                request.n_steps,
                request.h,
                request.mesh,
                request.border_conditions,
                request.bounds,
                request.rel_tolerance,
                request.scheme,
                request.method,
                request.bandwidth,
                request.backend_policy,
                request.resolver.as_ref(),
            ),
            (residual, sparse_jacobian) => jacobian
                .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                    request.eq_system,
                    request.values,
                    request.arg,
                    None,
                    request.t0,
                    None,
                    request.n_steps,
                    request.h,
                    request.mesh,
                    request.border_conditions,
                    request.bounds,
                    request.rel_tolerance,
                    request.scheme,
                    request.method,
                    request.bandwidth,
                    request.backend_policy,
                    request.resolver.as_ref(),
                    residual.unwrap_or(ResidualChunkingStrategy::Whole),
                    sparse_jacobian.unwrap_or(SparseChunkingStrategy::Whole),
                ),
        }
    }

    fn bootstrap_callable_aot_backend(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<LinkedBackendGuard, BvpBackendIntegrationError> {
        let build_begin = Instant::now();
        solver.try_eq_generate(None, None)?;
        println!(
            "[AOT bootstrap] {label}: build/materialize stage took {:?}",
            build_begin.elapsed()
        );

        let bundle = sparse_bundle_from_solver_request(solver)?;
        assert_eq!(
            bundle.effective_backend(),
            SelectedBackendKind::AotCompiled,
            "{label}: sparse bundle should resolve to compiled AOT after bootstrap"
        );
        assert!(
            bundle.resolved_aot_artifact().is_some(),
            "{label}: compiled AOT artifact metadata should be present"
        );

        let linked_guard = register_linked_backend_from_sparse_bundle(&bundle);
        let updated_config = solver
            .generated_backend_config()
            .clone()
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        solver.set_generated_backend_config(updated_config);
        solver.try_eq_generate(None, None)?;
        Ok(linked_guard)
    }

    fn max_abs_error_against_exact<F>(solver: &NRBVP, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result = solver
            .get_result()
            .expect("AOT acceptance check requires a computed solution matrix");
        let y = result.column(0);
        solver
            .x_mesh
            .iter()
            .zip(y.iter())
            .map(|(&x, &y_num)| (y_num - exact(x)).abs())
            .fold(0.0, f64::max)
    }

    fn l2_error_against_exact<F>(solver: &NRBVP, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result = solver
            .get_result()
            .expect("AOT acceptance check requires a computed solution matrix");
        let y = result.column(0);
        let mse = solver
            .x_mesh
            .iter()
            .zip(y.iter())
            .map(|(&x, &y_num)| {
                let diff = y_num - exact(x);
                diff * diff
            })
            .sum::<f64>()
            / solver.x_mesh.len() as f64;
        mse.sqrt()
    }

    fn solve_with_aot_and_report(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<LinkedBackendGuard, BvpBackendIntegrationError> {
        let guard = bootstrap_callable_aot_backend(solver, label)?;
        let solve_begin = Instant::now();
        solver.try_solve()?;
        println!(
            "[AOT solve] {label}: solve took {:?}",
            solve_begin.elapsed()
        );
        Ok(guard)
    }

    fn solve_with_aot_and_measure(
        solver: &mut NRBVP,
        label: &str,
    ) -> Result<(LinkedBackendGuard, f64, f64), BvpBackendIntegrationError> {
        let build_begin = Instant::now();
        let guard = bootstrap_callable_aot_backend(solver, label)?;
        let bootstrap_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
        let solve_begin = Instant::now();
        solver.try_solve()?;
        let solve_ms = solve_begin.elapsed().as_secs_f64() * 1_000.0;
        println!("[AOT measure] {label}: bootstrap={bootstrap_ms:.3} ms, solve={solve_ms:.3} ms");
        Ok((guard, bootstrap_ms, solve_ms))
    }

    fn run_combustion_tuning_scenario(
        n_steps: usize,
        scenario_label: &str,
    ) -> Result<(), BvpBackendIntegrationError> {
        let sequential_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);

        let mut sequential = make_combustion_solver(n_steps, sequential_config);
        let (_seq_guard, seq_bootstrap_ms, seq_solve_ms) = solve_with_aot_and_measure(
            &mut sequential,
            &format!("combustion-sequential-baseline-{n_steps}"),
        )?;
        let sequential_solution = sequential
            .get_result()
            .expect("sequential baseline for AOT combustion tuning should produce a solution");
        assert!(
            sequential_solution.iter().all(|value| value.is_finite()),
            "sequential baseline for AOT combustion tuning should remain finite"
        );

        let parallel_cases = [
            (
                "par-4x4-jobs4",
                ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(4),
                    max_sparse_jobs: Some(4),
                    fallback_policy: ParallelFallbackPolicy::Never,
                },
                AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                ),
            ),
            (
                "par-8x8-jobs8",
                ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(8),
                    max_sparse_jobs: Some(8),
                    fallback_policy: ParallelFallbackPolicy::Never,
                },
                AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                ),
            ),
            (
                "par-16x16-jobs16",
                ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(16),
                    max_sparse_jobs: Some(16),
                    fallback_policy: ParallelFallbackPolicy::Never,
                },
                AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                ),
            ),
            (
                "par-res16-row32-jobs8",
                ParallelExecutorConfig {
                    jobs_per_worker: 1,
                    max_residual_jobs: Some(8),
                    max_sparse_jobs: Some(8),
                    fallback_policy: ParallelFallbackPolicy::Never,
                },
                AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 16 }),
                    Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 32 }),
                ),
            ),
        ];

        let mut rows = vec![RuntimeTuningRow {
            label: "sequential-baseline",
            n_steps,
            bootstrap_ms: seq_bootstrap_ms,
            solve_ms: seq_solve_ms,
            speedup_vs_seq: 1.0,
            max_diff_vs_seq: 0.0,
        }];

        for (label, execution_policy, chunking_policy) in parallel_cases {
            let generated_backend_config =
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_aot_execution_policy(AotExecutionPolicy::Parallel(execution_policy))
                    .with_aot_chunking_policy(chunking_policy);
            let mut solver = make_combustion_solver(n_steps, generated_backend_config);
            let (_guard, bootstrap_ms, solve_ms) =
                solve_with_aot_and_measure(&mut solver, &format!("{label}-{n_steps}"))?;
            let solution = solver
                .get_result()
                .expect("parallel AOT combustion tuning case should produce a solution");
            assert!(
                solution.iter().all(|value| value.is_finite()),
                "{label}: parallel AOT combustion tuning solution should remain finite"
            );
            let max_diff_vs_seq = sequential_solution
                .iter()
                .zip(solution.iter())
                .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            assert!(
                max_diff_vs_seq < 1.0e-4,
                "{label}: AOT combustion tuning disagreement with sequential baseline {max_diff_vs_seq} is too large"
            );
            rows.push(RuntimeTuningRow {
                label,
                n_steps,
                bootstrap_ms,
                solve_ms,
                speedup_vs_seq: seq_solve_ms / solve_ms,
                max_diff_vs_seq,
            });
        }

        println!();
        println!("[AOT combustion tuning map] scenario={scenario_label}, n_steps={n_steps}");
        println!(
            "{:<24} | {:>7} | {:>12} | {:>12} | {:>14} | {:>14}",
            "config", "n_steps", "bootstrap_ms", "solve_ms", "speedup_vs_seq", "max_diff_vs_seq"
        );
        println!("{}", "-".repeat(96));
        for row in &rows {
            println!(
                "{:<24} | {:>7} | {:>12.3} | {:>12.3} | {:>14.3} | {:>14.6e}",
                row.label,
                row.n_steps,
                row.bootstrap_ms,
                row.solve_ms,
                row.speedup_vs_seq,
                row.max_diff_vs_seq
            );
        }

        let winner = rows
            .iter()
            .min_by(|lhs, rhs| lhs.solve_ms.total_cmp(&rhs.solve_ms))
            .expect("runtime tuning table should contain at least one row");
        println!(
            "[AOT tuning winner] scenario={}, config={}, n_steps={}, solve_ms={:.3}, speedup_vs_seq={:.3}, bootstrap_ms={:.3}",
            scenario_label,
            winner.label,
            winner.n_steps,
            winner.solve_ms,
            winner.speedup_vs_seq,
            winner.bootstrap_ms
        );
        Ok(())
    }

    fn make_example_solver(
        equation: &NonlinEquation,
        n_steps: usize,
        strategy_params: Option<SolverParams>,
        generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
    ) -> NRBVP {
        let eq_system = equation.setup();
        let values = equation.values();
        let border_conditions = equation.boundary_conditions();
        let bounds = equation.Bounds();
        let rel_tolerance = equation.rel_tolerance();
        let (t0, t_end) = equation.span(None, None);
        let initial_guess = uniform_initial_guess(values.len(), n_steps, 0.7);
        let options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(strategy_params)
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(40)
            .with_bounds(bounds)
            .with_generated_backend_config(generated_backend_config);

        let mut solver = NRBVP::new_with_options(
            eq_system,
            initial_guess,
            values,
            "x".to_string(),
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn make_combustion_solver(
        n_steps: usize,
        generated_backend_config: crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig,
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
            .with_loglevel(Some("error".to_string()));
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

    #[test]
    //  #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
    fn aot_exact_examples_sequential_cover_tens_and_hundreds_of_steps() {
        let configs = [
            (
                "two-point-64",
                NonlinEquation::TwoPointBVP,
                64usize,
                1.5e-2f64,
            ),
            (
                "two-point-240",
                NonlinEquation::TwoPointBVP,
                240usize,
                6.0e-3f64,
            ),
            ("clairaut-72", NonlinEquation::Clairaut, 72usize, 1.5e-2f64),
            (
                "clairaut-220",
                NonlinEquation::Clairaut,
                220usize,
                1.25e-2f64,
            ),
        ];

        for (label, equation, n_steps, max_tol) in configs {
            let generated_backend_config =
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
            let mut solver = make_example_solver(
                &equation,
                n_steps,
                Some(SolverParams::default()),
                generated_backend_config,
            );
            let _guard = solve_with_aot_and_report(&mut solver, label)
                .expect("sequential AOT exact-example acceptance case should solve");

            let error = match equation {
                NonlinEquation::TwoPointBVP => {
                    max_abs_error_against_exact(&solver, |x| (-x * x / 4.0).exp())
                }
                NonlinEquation::Clairaut => l2_error_against_exact(&solver, |x| {
                    1.0 + (x - 1.0).powi(2) - (x - 1.0).powi(3) / 6.0 + (x - 1.0).powi(4) / 12.0
                }),
                _ => unreachable!("only exact sequential examples are expected here"),
            };

            println!("[AOT exact sequential] {label}: n_steps={n_steps}, error={error:.6e}");
            assert!(
                error < max_tol,
                "{label}: exact-solution error {error} exceeded tolerance {max_tol}"
            );
        }
    }

    #[test]
    //   #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
    fn aot_parallel_exact_examples_cover_parallel_modes_and_chunking() {
        let lane_parallel = SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: None,
        };
        let parachute_parallel = SolverParams {
            max_jac: Some(5),
            max_damp_iter: Some(5),
            damp_factor: None,
            adaptive: None,
        };

        let cases = [
            (
                "parachute-parallel-48",
                NonlinEquation::ParachuteEquation,
                48usize,
                parachute_parallel.clone(),
                5.0e-3f64,
            ),
            (
                "lane-emden-parallel-180",
                NonlinEquation::LaneEmden5,
                180usize,
                lane_parallel.clone(),
                3.5e-4f64,
            ),
        ];

        for (label, equation, n_steps, strategy_params, l2_tol) in cases {
            let generated_backend_config =
                crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                    .with_aot_execution_policy(sparse_parallel_policy())
                    .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                        Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                        Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    ));
            let mut solver = make_example_solver(
                &equation,
                n_steps,
                Some(strategy_params),
                generated_backend_config,
            );
            let _guard = solve_with_aot_and_report(&mut solver, label)
                .expect("parallel AOT exact-example acceptance case should solve");

            let l2_error = match equation {
                NonlinEquation::ParachuteEquation => {
                    l2_error_against_exact(&solver, |x| (((2.0 * x).exp() + 1.0) / 2.0).ln() - x)
                }
                NonlinEquation::LaneEmden5 => {
                    l2_error_against_exact(&solver, |x| (1.0 + x * x / 3.0).powf(-0.5))
                }
                _ => unreachable!("only parallel analytical examples are expected here"),
            };

            println!("[AOT exact parallel] {label}: n_steps={n_steps}, l2_error={l2_error:.6e}");
            assert!(
                l2_error < l2_tol,
                "{label}: L2 exact-solution error {l2_error} exceeded tolerance {l2_tol}"
            );
        }
    }

    #[test]
    // #[ignore = "production-style AOT acceptance suite with real build/bootstrap"]
    fn aot_combustion_acceptance_covers_sequential_parallel_and_varied_grids() {
        let sequential_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly);
        let parallel_config =
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_build_if_missing_release()
                .with_aot_execution_policy(sparse_parallel_policy())
                .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                    Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                    Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 8 }),
                ));

        let mut sequential = make_combustion_solver(36, sequential_config.clone());
        let _sequential_guard =
            solve_with_aot_and_report(&mut sequential, "combustion-sequential-36")
                .expect("sequential AOT combustion case should solve");
        let sequential_solution = sequential
            .get_result()
            .expect("sequential AOT combustion case should produce a solution");
        assert!(
            sequential_solution.iter().all(|value| value.is_finite()),
            "sequential AOT combustion solution should remain finite"
        );

        let mut parallel = make_combustion_solver(36, parallel_config.clone());
        let _parallel_guard = solve_with_aot_and_report(&mut parallel, "combustion-parallel-36")
            .expect("parallel AOT combustion case should solve");
        let parallel_solution = parallel
            .get_result()
            .expect("parallel AOT combustion case should produce a solution");
        assert!(
            parallel_solution.iter().all(|value| value.is_finite()),
            "parallel AOT combustion solution should remain finite"
        );

        let max_difference = sequential_solution
            .iter()
            .zip(parallel_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        println!(
            "[AOT combustion compare] n_steps=36, max_difference_seq_vs_par={max_difference:.6e}"
        );
        assert!(
            max_difference < 1.0e-4,
            "AOT combustion sequential/parallel disagreement {max_difference} is too large"
        );

        let mut large_sequential = make_combustion_solver(256, sequential_config);
        let _large_sequential_guard =
            solve_with_aot_and_report(&mut large_sequential, "combustion-sequential-256")
                .expect("large-grid sequential AOT combustion case should solve");
        let large_sequential_solution = large_sequential
            .get_result()
            .expect("large-grid sequential AOT combustion case should produce a solution");
        assert!(
            large_sequential_solution
                .iter()
                .all(|value| value.is_finite()),
            "large-grid sequential AOT combustion solution should remain finite"
        );

        let mut large_parallel = make_combustion_solver(256, parallel_config);
        let _large_parallel_guard =
            solve_with_aot_and_report(&mut large_parallel, "combustion-parallel-256")
                .expect("large-grid parallel AOT combustion case should solve");
        let large_solution = large_parallel
            .get_result()
            .expect("large-grid AOT combustion case should produce a solution");
        assert!(
            large_solution.iter().all(|value| value.is_finite()),
            "large-grid parallel AOT combustion solution should remain finite"
        );

        let large_max_difference = large_sequential_solution
            .iter()
            .zip(large_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        println!(
            "[AOT combustion compare] n_steps=256, max_difference_seq_vs_par={large_max_difference:.6e}"
        );
        assert!(
            large_max_difference < 1.0e-4,
            "AOT combustion sequential/parallel disagreement {large_max_difference} is too large"
        );
    }
    /*
        слишком мало чанков: не хватает загрузки
    слишком много чанков: overhead на orchestration начинает съедать выигрыш
    средний режим вроде 8x8 + jobs8 попадает в sweet spot
    И очень важно:

    max_diff_vs_seq = 0 у всех конфигураций
    это именно то, что и нужно было доказать: parallel AOT меняет скорость, но не математику.
    CPU 4 Cores
    [AOT combustion tuning map] scenario=medium-grid, n_steps=128
    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     128 |     7113.999 |      440.910 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     128 |    15332.448 |      335.703 |          1.313 |     0.000000e0
    par-8x8-jobs8            |     128 |    10177.598 |     1116.167 |          0.395 |     0.000000e0
    par-16x16-jobs16         |     128 |     8069.948 |      261.911 |          1.683 |     0.000000e0
    par-res16-row32-jobs8    |     128 |     6763.878 |      280.503 |          1.572 |     0.000000e0
    [AOT tuning winner] scenario=medium-grid, config=par-16x16-jobs16, n_steps=128, solve_ms=261.911, speedup_vs_seq=1.683, bootstrap_ms=8069.948

    [AOT combustion tuning map] scenario=large-grid, n_steps=256
    config                   | n_steps | bootstrap_ms |     solve_ms | speedup_vs_seq | max_diff_vs_seq
    ------------------------------------------------------------------------------------------------
    sequential-baseline      |     256 |    15992.826 |      803.811 |          1.000 |     0.000000e0
    par-4x4-jobs4            |     256 |    36969.672 |     1046.569 |          0.768 |     0.000000e0
    par-8x8-jobs8            |     256 |    22350.296 |      510.894 |          1.573 |     0.000000e0
    par-16x16-jobs16         |     256 |    18268.103 |      956.696 |          0.840 |     0.000000e0
    par-res16-row32-jobs8    |     256 |    13534.274 |      593.440 |          1.354 |     0.000000e0
    [AOT tuning winner] scenario=large-grid, config=par-8x8-jobs8, n_steps=256, solve_ms=510.894, speedup_vs_seq=1.573, bootstrap_ms=22350.296
         */
    #[test]
    fn aot_combustion_parallel_tuning_reports_runtime_table() {
        run_combustion_tuning_scenario(128, "medium-grid")
            .expect("medium-grid AOT combustion tuning scenario should solve");
        run_combustion_tuning_scenario(256, "large-grid")
            .expect("large-grid AOT combustion tuning scenario should solve");
    }
}
