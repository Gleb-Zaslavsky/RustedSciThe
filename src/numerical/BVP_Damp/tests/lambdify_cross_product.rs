//! Correctness cross-product for the pure Lambdify BVP routes.
//!
//! The existing parity corpus proves the production Banded route on a few
//! end-to-end fixtures. This module deliberately widens that gate without
//! measuring performance: the same discretized problem is checked across
//! `ExprLegacy` and `AtomView`, `Sparse` and `Banded`, with `Dense` retained as
//! a control backend only. The low-level policy test also proves that the
//! no-Mutex Banded callback is invariant under sequential/parallel execution
//! and under both supported work decompositions.
//!
//! ```text
//! cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_cross_product -- --nocapture --test-threads=1
//! ```

mod tests {
    use crate::Utils::test_reporting::write_test_report;
    use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
    use crate::numerical::BVP_Damp::MatrixBackend;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
    use crate::symbolic::bvp::direct::{
        BandedJacobianChunking, BandedLambdifyConfig, DirectBandedProblem,
    };
    use crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    #[derive(Clone, Copy, Debug)]
    enum MatrixRoute {
        Dense,
        Sparse,
        Banded,
    }

    impl MatrixRoute {
        fn label(self) -> &'static str {
            match self {
                Self::Dense => "Dense-control",
                Self::Sparse => "Sparse-faer",
                Self::Banded => "Banded",
            }
        }

        fn options(self, frontend: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
            let config = match self {
                Self::Dense => GeneratedBackendConfig::default()
                    .with_matrix_backend_override(MatrixBackend::Dense),
                Self::Sparse => GeneratedBackendConfig::sparse_defaults()
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
                Self::Banded => GeneratedBackendConfig::banded_lambdify_defaults(),
            }
            .with_backend_policy_override(Some(
                crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy::LambdifyOnly,
            ))
            .with_symbolic_assembly_backend(frontend);

            let base = match self {
                Self::Dense => DampedSolverOptions::dense_damped(),
                Self::Sparse => DampedSolverOptions::sparse_damped(),
                Self::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
            };
            base.with_generated_backend_config(config)
                .with_strategy_params(Some(SolverParams::default()))
                .with_abs_tolerance(1e-9)
                .with_rel_tolerance(HashMap::from([
                    ("y".to_string(), 1e-7),
                    ("z".to_string(), 1e-7),
                ]))
                .with_max_iterations(30)
                .with_bounds(HashMap::from([
                    ("y".to_string(), (-2.0, 2.0)),
                    ("z".to_string(), (-2.0, 2.0)),
                ]))
                .with_loglevel(Some("error".to_string()))
        }
    }

    fn oscillator_solver(
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
    ) -> NRBVP {
        let t_end = std::f64::consts::FRAC_PI_2;
        let h = t_end / n_steps as f64;
        let guess = DMatrix::from_fn(2, n_steps, |row, column| {
            let x = column as f64 * h;
            if row == 0 { x.sin() } else { x.cos() }
        });
        let mut solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([("y".to_string(), vec![(0usize, 0.0), (1usize, 1.0)])]),
            0.0,
            t_end,
            n_steps,
            route.options(frontend),
        );
        solver.dont_save_log(true);
        solver
    }

    fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    fn callback_snapshot(solver: &mut NRBVP) -> (Vec<f64>, DMatrix<f64>) {
        solver
            .try_eq_generate(None, None)
            .expect("cross-product callback generation should succeed");
        let args = DVector::from_element(solver.values.len() * solver.n_steps, 0.7);
        let typed = Vectors_type_casting(&args, solver.method.clone());
        let residual = solver.fun.call(1.0, &*typed).to_DVectorType();
        let jacobian = solver
            .jac
            .as_mut()
            .expect("generated Lambdify route must install a Jacobian")
            .call(1.0, &*typed)
            .to_DMatrixType();
        (residual.as_slice().to_vec(), jacobian)
    }

    #[test]
    fn lambdify_frontend_matrix_cross_product_has_callback_and_solution_parity() {
        let routes = [MatrixRoute::Dense, MatrixRoute::Sparse, MatrixRoute::Banded];
        let frontends = [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ];
        let mut snapshots = Vec::new();

        for route in routes {
            for frontend in frontends {
                let mut solver = oscillator_solver(route, frontend, 12);
                let (residual, jacobian) = callback_snapshot(&mut solver);
                solver
                    .try_solver_prepared()
                    .expect("cross-product solver should converge");
                let solution = solver
                    .get_result()
                    .expect("cross-product solver should publish a result")
                    .as_slice()
                    .to_vec();
                println!(
                    "[BVP Lambdify cross-product] frontend={frontend:?} matrix={} residual_len={} jacobian={}x{}",
                    route.label(),
                    residual.len(),
                    jacobian.nrows(),
                    jacobian.ncols()
                );
                snapshots.push((route, frontend, residual, jacobian, solution));
            }
        }

        let reference = &snapshots[0];
        for (route, frontend, residual, jacobian, solution) in snapshots.iter().skip(1) {
            assert!(
                max_abs_diff(&reference.2, residual) < 1e-9,
                "callback residual drift for {frontend:?}/{}",
                route.label()
            );
            assert_eq!(reference.3.shape(), jacobian.shape());
            let jacobian_diff = reference
                .3
                .iter()
                .zip(jacobian.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                jacobian_diff < 1e-8,
                "callback Jacobian drift for {frontend:?}/{}: {jacobian_diff:.3e}",
                route.label()
            );
            assert!(
                max_abs_diff(&reference.4, solution) < 1e-7,
                "solution drift for {frontend:?}/{}",
                route.label()
            );
        }

        let report = "status: passed\n\nfixture: oscillator, n_steps=12\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Dense-control, Sparse-faer, Banded\nexecution_policy: solver-level default for the selected pure-Lambdify route\nchecks: residual callback parity, dense Jacobian parity, final solution parity\ntolerance: residual < 1e-9; Jacobian < 1e-8; solution < 1e-7\ninterpretation: Dense is a control route only; Sparse/faer and Banded are production routes.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "lambdify_frontend_matrix_cross_product_has_callback_and_solution_parity",
            report,
        ) {
            eprintln!("[BVP test report] unable to write cross-product report: {error}");
        }
    }

    fn nonlinear_direct_problem() -> DirectBandedProblem {
        let mut problem = DirectBandedProblem::default();
        problem.vector_of_functions = vec![
            Expr::parse_expression("y_0^2 + z_0"),
            Expr::parse_expression("y_0 - z_0^2"),
            Expr::parse_expression("y_1^2 + z_1"),
            Expr::parse_expression("y_1 - z_1^2"),
        ];
        problem.vector_of_variables = vec![
            Expr::Var("y_0".to_string()),
            Expr::Var("z_0".to_string()),
            Expr::Var("y_1".to_string()),
            Expr::Var("z_1".to_string()),
        ];
        problem.variable_string = vec![
            "y_0".to_string(),
            "z_0".to_string(),
            "y_1".to_string(),
            "z_1".to_string(),
        ];
        problem.bandwidth = Some((1, 1));
        problem.symbolic_jacobian_sparse = vec![
            (0, 0, Expr::parse_expression("2*y_0")),
            (0, 1, Expr::Const(1.0)),
            (1, 0, Expr::Const(1.0)),
            (1, 1, Expr::parse_expression("-2*z_0")),
            (2, 2, Expr::parse_expression("2*y_1")),
            (2, 3, Expr::Const(1.0)),
            (3, 2, Expr::Const(1.0)),
            (3, 3, Expr::parse_expression("-2*z_1")),
        ];
        problem
    }

    #[test]
    fn banded_lambdify_parallel_policies_and_chunk_layouts_match_on_nonlinear_corpus() {
        let policies = [
            (
                "sequential-diagonal",
                BvpLambdifyExecutionPolicy::Sequential,
                BandedJacobianChunking::Diagonal,
            ),
            (
                "parallel-diagonal",
                BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
                BandedJacobianChunking::Diagonal,
            ),
            (
                "sequential-entry",
                BvpLambdifyExecutionPolicy::Sequential,
                BandedJacobianChunking::EntryChunks,
            ),
            (
                "parallel-entry",
                BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
                BandedJacobianChunking::EntryChunks,
            ),
        ];
        let unknowns = [1.25, -0.75, 0.5, 1.5];
        let mut reference: Option<(Vec<f64>, Vec<f64>)> = None;

        for (label, execution_policy, jacobian_chunking) in policies {
            let problem = nonlinear_direct_problem();
            let config = BandedLambdifyConfig {
                execution_policy,
                jacobian_chunking,
                ..BandedLambdifyConfig::default()
            };
            let residual = problem
                .generate_banded_residual_with_config(&config)
                .expect("nonlinear residual callback should compile");
            let jacobian = problem
                .generate_banded_jacobian_runtime_parallel(&config)
                .expect("nonlinear Jacobian callback should compile");
            let residual_values = residual(&unknowns).expect("nonlinear residual should evaluate");
            let assembly =
                (jacobian.callback())(&unknowns).expect("nonlinear Jacobian should evaluate");
            let mut jacobian_values = Vec::with_capacity(16);
            for row in 0..4 {
                for col in 0..4 {
                    // Values outside the declared band are mathematically
                    // zero; the native storage reports them as out of bounds.
                    jacobian_values.push(assembly.get(row, col).unwrap_or(0.0));
                }
            }
            if let Some((reference_residual, reference_jacobian)) = &reference {
                assert_eq!(
                    reference_residual, &residual_values,
                    "residual drift in {label}"
                );
                assert_eq!(
                    reference_jacobian, &jacobian_values,
                    "Jacobian layout drift in {label}"
                );
            } else {
                reference = Some((residual_values, jacobian_values));
            }
        }

        let report = "status: passed\n\nfixture: nonlinear four-variable direct Banded corpus\npolicies: Sequential/Parallel(min_work=0)\nlayouts: Diagonal/EntryChunks\nchecks: residual values, Banded slot values and out-of-band zero semantics\ninterpretation: execution policy and work decomposition are numerically interchangeable; this is a correctness gate, not a performance claim.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "banded_lambdify_parallel_policies_and_chunk_layouts_match_on_nonlinear_corpus",
            report,
        ) {
            eprintln!("[BVP test report] unable to write policy report: {error}");
        }
    }
}
