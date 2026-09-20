//! Debug correctness gates for the production pure-Lambdify routes.
//!
//! These tests intentionally do not measure wall-clock time. Their purpose is
//! to keep the frontend, matrix backend and nonlinear solve contracts aligned
//! before a later release benchmark. Dense is not included: it remains a
//! small-problem control route, while Sparse/faer and Banded are the intended
//! production routes for large BVP systems.
//!
//! ```text
//! cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_acceptance -- --nocapture --test-threads=1
//! ```

#[cfg(test)]
mod tests {
    use crate::Utils::test_reporting::write_test_report;
    use crate::numerical::BVP_Damp::BVP_traits::{VectorType, Vectors_type_casting};
    use crate::numerical::BVP_Damp::MatrixBackend;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        AdaptiveGridConfig, DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode;
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    #[derive(Clone, Copy, Debug)]
    enum MatrixRoute {
        Sparse,
        Banded,
    }

    impl MatrixRoute {
        fn label(self) -> &'static str {
            match self {
                Self::Sparse => "Sparse-faer",
                Self::Banded => "Banded",
            }
        }

        fn options(self, frontend: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
            let generated = match self {
                Self::Sparse => GeneratedBackendConfig::sparse_defaults()
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
                Self::Banded => GeneratedBackendConfig::banded_lambdify_defaults(),
            }
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(frontend);

            let base = match self {
                Self::Sparse => DampedSolverOptions::sparse_damped(),
                Self::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
            };
            base.with_generated_backend_config(generated)
                .with_strategy_params(Some(SolverParams::default()))
                .with_abs_tolerance(1e-10)
                .with_rel_tolerance(HashMap::from([
                    ("y".to_string(), 1e-8),
                    ("z".to_string(), 1e-8),
                ]))
                .with_bounds(HashMap::from([
                    ("y".to_string(), (-2.0, 2.0)),
                    ("z".to_string(), (-2.0, 2.0)),
                ]))
                .with_max_iterations(40)
                .with_loglevel(Some("error".to_string()))
        }
    }

    fn nonlinear_exact_fixture(
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
    ) -> NRBVP {
        // y' = z, z' = 2*y^3, y(0)=1, y(1)=1/2.
        // Exact solution: y=1/(1+x), z=-1/(1+x)^2.
        let mut guess = DMatrix::zeros(2, n_steps);
        for node in 0..n_steps {
            let x = node as f64 / n_steps as f64;
            guess[(0, node)] = 1.0 / (1.0 + x);
            guess[(1, node)] = -1.0 / (1.0 + x).powi(2);
        }

        let mut solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("2*y^3")],
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([("y".to_string(), vec![(0usize, 1.0), (1usize, 0.5)])]),
            0.0,
            1.0,
            n_steps,
            route.options(frontend),
        );
        solver.dont_save_log(true);
        solver
    }

    fn variable_coefficient_fixture(
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
    ) -> NRBVP {
        // y' = z, z' = -2*x*z/(1+x^2), y(0)=0, y(1)=atan(1).
        // Exact solution: y=atan(x), z=1/(1+x^2). This checks that the
        // runtime argument x survives both symbolic frontends and layouts.
        let mut guess = DMatrix::zeros(2, n_steps);
        for node in 0..n_steps {
            let x = node as f64 / n_steps as f64;
            guess[(0, node)] = x.atan();
            guess[(1, node)] = 1.0 / (1.0 + x * x);
        }

        let mut solver = NRBVP::new_with_options(
            vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("-2*x*z/(1+x^2)"),
            ],
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([(
                "y".to_string(),
                vec![(0usize, 0.0), (1usize, 0.25 * std::f64::consts::PI)],
            )]),
            0.0,
            1.0,
            n_steps,
            route.options(frontend),
        );
        solver.dont_save_log(true);
        solver
    }

    fn nonuniform_linear_fixture(
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
        mesh: &[f64],
    ) -> NRBVP {
        // y' = z, z' = 0 with y(0)=0 and y(1)=1. The exact solution is
        // y=x, z=1 even on a nonuniform mesh, so this isolates mesh plumbing
        // from nonlinear convergence and frontend arithmetic.
        let n_steps = mesh.len() - 1;
        let guess = DMatrix::from_fn(
            2,
            n_steps,
            |row, column| {
                if row == 0 { mesh[column] } else { 1.0 }
            },
        );
        let mut solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::Const(0.0)],
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([("y".to_string(), vec![(0usize, 0.0), (1usize, 1.0)])]),
            0.0,
            1.0,
            n_steps,
            route.options(frontend),
        );
        solver.dont_save_log(true);
        solver
    }

    fn assert_exact_solution(
        solver: &NRBVP,
        n_steps: usize,
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
    ) {
        let solution = solver
            .get_result()
            .expect("pure Lambdify nonlinear BVP should publish a result");
        assert_eq!(solution.ncols(), 2);
        assert_eq!(solution.nrows(), n_steps + 1);
        assert!(
            solution.iter().all(|value| value.is_finite()),
            "exact nonlinear fixture returned a non-finite solution for {frontend:?}/{}",
            route.label()
        );

        let mut max_y_error: f64 = 0.0;
        let mut max_z_error: f64 = 0.0;
        for node in 0..=n_steps {
            let x = node as f64 / n_steps as f64;
            max_y_error = max_y_error.max((solution[(node, 0)] - 1.0 / (1.0 + x)).abs());
            max_z_error = max_z_error.max((solution[(node, 1)] + 1.0 / (1.0 + x).powi(2)).abs());
        }
        assert!(
            max_y_error < 1e-2 && max_z_error < 5e-2,
            "exact nonlinear fixture drift for {frontend:?}/{}: y={max_y_error:e}, z={max_z_error:e}",
            route.label()
        );
        assert!((solution[(0, 0)] - 1.0).abs() < 1e-8);
        assert!((solution[(n_steps, 0)] - 0.5).abs() < 1e-8);
        let reduced = solver
            .result
            .as_ref()
            .expect("exact nonlinear solve should retain its reduced state");
        let typed_reduced = Vectors_type_casting(reduced, solver.method.clone());
        let residual = solver.fun.call(solver.p, &*typed_reduced).to_DVectorType();
        assert!(
            residual.iter().all(|value| value.is_finite()),
            "exact nonlinear callback returned a non-finite residual for {frontend:?}/{}",
            route.label()
        );
        let max_residual = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(
            max_residual < 1e-1,
            "exact nonlinear residual too large for {frontend:?}/{}: {max_residual:e}",
            route.label()
        );
    }

    fn assert_variable_coefficient_solution(
        solver: &NRBVP,
        n_steps: usize,
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
    ) {
        let solution = solver
            .get_result()
            .expect("variable-coefficient BVP should publish a result");
        assert_eq!(solution.nrows(), n_steps + 1);
        assert_eq!(solution.ncols(), 2);
        assert!(
            solution.iter().all(|value| value.is_finite()),
            "variable-coefficient fixture returned a non-finite solution for {frontend:?}/{}",
            route.label()
        );

        let mut max_y_error: f64 = 0.0;
        let mut max_z_error: f64 = 0.0;
        for node in 0..=n_steps {
            let x = node as f64 / n_steps as f64;
            max_y_error = max_y_error.max((solution[(node, 0)] - x.atan()).abs());
            max_z_error = max_z_error.max((solution[(node, 1)] - 1.0 / (1.0 + x * x)).abs());
        }
        assert!(
            max_y_error < 2e-2 && max_z_error < 5e-2,
            "variable-coefficient drift for {frontend:?}/{}: y={max_y_error:e}, z={max_z_error:e}",
            route.label()
        );
        assert!(solution[(0, 0)].abs() < 1e-8);
        assert!((solution[(n_steps, 0)] - 0.25 * std::f64::consts::PI).abs() < 1e-8);
        let reduced = solver
            .result
            .as_ref()
            .expect("variable-coefficient solve should retain its reduced state");
        let typed_reduced = Vectors_type_casting(reduced, solver.method.clone());
        let residual = solver.fun.call(solver.p, &*typed_reduced).to_DVectorType();
        assert!(
            residual.iter().all(|value| value.is_finite()),
            "variable-coefficient callback returned a non-finite residual for {frontend:?}/{}",
            route.label()
        );
        let max_residual = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(
            max_residual < 1e-1,
            "variable-coefficient residual too large for {frontend:?}/{}: {max_residual:e}",
            route.label()
        );
    }

    #[test]
    fn nonlinear_exact_solution_is_preserved_across_production_lambdify_routes() {
        let n_steps = 24;
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = nonlinear_exact_fixture(route, frontend, n_steps);
                solver.try_eq_generate(None, None).unwrap_or_else(|error| {
                    panic!(
                        "pure Lambdify preparation failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });
                solver.try_solver_prepared().unwrap_or_else(|error| {
                    panic!(
                        "pure Lambdify solve failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });
                assert_exact_solution(&solver, n_steps, route, frontend);
            }
        }
        let report = "status: passed\n\nfixture: nonlinear exact BVP y=1/(1+x), z=-1/(1+x)^2\nmesh: uniform, n_steps=24\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: explicit preparation, typed solver completion, finite result, endpoint BC values, bounded finite reduced-state residual and exact-solution max-error bounds\nlimits: max_y_error < 1e-2; max_z_error < 5e-2; discrete residual < 1e-1\ninterpretation: production pure-Lambdify routes preserve a nonlinear analytical fixture; this is a debug correctness gate, not a timing claim.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "nonlinear_exact_solution_is_preserved_across_production_lambdify_routes",
            report,
        ) {
            eprintln!("[BVP test report] unable to write exact-solution report: {error}");
        }
    }

    #[test]
    fn nonlinear_exact_solution_frontends_have_matching_final_state() {
        let n_steps = 16;
        let mut solutions = Vec::new();
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            let mut solver = nonlinear_exact_fixture(MatrixRoute::Banded, frontend, n_steps);
            solver.try_eq_generate(None, None).unwrap_or_else(|error| {
                panic!("Banded preparation failed for {frontend:?}: {error:?}")
            });
            solver
                .try_solver_prepared()
                .unwrap_or_else(|error| panic!("Banded solve failed for {frontend:?}: {error:?}"));
            solutions.push(
                solver
                    .get_result()
                    .expect("solution should be stored")
                    .clone(),
            );
        }

        let max_diff = solutions[0]
            .iter()
            .zip(solutions[1].iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff < 1e-8,
            "Banded frontend solution drift: {max_diff:e}"
        );
        let report = format!(
            "status: passed\n\nfixture: nonlinear exact BVP, Banded, n_steps={n_steps}\nfrontends: ExprLegacy, AtomView\ncheck: final-state componentwise parity\nmax_solution_diff: {max_diff:.3e}\nlimit: 1e-8\n"
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "nonlinear_exact_solution_frontends_have_matching_final_state",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write frontend parity report: {error}");
        }
    }

    #[test]
    fn variable_coefficient_solution_is_preserved_across_production_lambdify_routes() {
        let n_steps = 20;
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = variable_coefficient_fixture(route, frontend, n_steps);
                solver.try_eq_generate(None, None).unwrap_or_else(|error| {
                    panic!(
                        "variable-coefficient preparation failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });
                solver.try_solver_prepared().unwrap_or_else(|error| {
                    panic!(
                        "variable-coefficient solve failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });
                assert_variable_coefficient_solution(&solver, n_steps, route, frontend);
            }
        }
        let report = "status: passed\n\nfixture: variable-coefficient BVP y=atan(x), z=1/(1+x^2)\nmesh: uniform, n_steps=20\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: x-dependent RHS argument, mixed left/right y boundary conditions, finite result, endpoint BC values, bounded finite reduced-state residual and analytical max-error bounds\nlimits: max_y_error < 2e-2; max_z_error < 5e-2; discrete residual < 1e-1\ninterpretation: variable-coefficient pure-Lambdify callbacks preserve the explicit independent-variable contract; this is a debug correctness gate, not a timing claim.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "variable_coefficient_solution_is_preserved_across_production_lambdify_routes",
            report,
        ) {
            eprintln!("[BVP test report] unable to write variable-coefficient report: {error}");
        }
    }

    fn adaptive_bratu_fixture(
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
    ) -> NRBVP {
        // The high initial profile intentionally exercises damping and one
        // adaptive DoublePoints refinement, matching the parity corpus while
        // running through every pure-Lambdify production matrix route.
        let guess = DMatrix::from_fn(2, n_steps, |row, _| if row == 0 { 4.0 } else { 0.0 });
        let strategy = SolverParams {
            adaptive: Some(AdaptiveGridConfig {
                version: 1,
                max_refinements: 1,
                grid_method: GridRefinementMethod::DoublePoints,
            }),
            ..SolverParams::default()
        };
        let mut solver = NRBVP::new_with_options(
            vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("-2*exp(y)"),
            ],
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([("y".to_string(), vec![(0usize, 0.0), (1usize, 0.0)])]),
            0.0,
            1.0,
            n_steps,
            route
                .options(frontend)
                .with_strategy_params(Some(strategy))
                .with_bounds(HashMap::from([
                    ("y".to_string(), (-10.0, 10.0)),
                    ("z".to_string(), (-20.0, 20.0)),
                ])),
        );
        solver.dont_save_log(true);
        solver
    }

    #[test]
    fn adaptive_nonlinear_lambdify_routes_preserve_refinement_and_solution_contract() {
        let n_steps = 24;
        let mut reference: Option<DMatrix<f64>> = None;
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = adaptive_bratu_fixture(route, frontend, n_steps);
                solver.try_solve().unwrap_or_else(|error| {
                    panic!(
                        "adaptive pure Lambdify solve failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });
                let solution = solver
                    .get_result()
                    .expect("adaptive pure Lambdify route should publish a result")
                    .clone();
                assert_eq!(solution.nrows(), 2 * n_steps + 1);
                assert!(solution.iter().all(|value| value.is_finite()));
                let telemetry = solver.get_statistics().telemetry;
                assert_eq!(telemetry.counters.grid_refinements, 1);
                assert!(telemetry.counters.iterations > 0);
                if let Some(reference) = &reference {
                    let max_diff = reference
                        .iter()
                        .zip(solution.iter())
                        .map(|(lhs, rhs)| (lhs - rhs).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        max_diff < 1e-7,
                        "adaptive frontend/backend drift for {frontend:?}/{}: {max_diff:e}",
                        route.label()
                    );
                } else {
                    reference = Some(solution);
                }
            }
        }
        let report = "status: passed\n\nfixture: nonlinear Bratu-like BVP y'=z, z'=-2*exp(y), y(0)=y(1)=0\nmesh: uniform n_steps=24, one DoublePoints refinement\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: finite published result, exactly one refinement, nonzero Newton trace, final-state parity\nlimit: max cross-route/frontend drift < 1e-7\ninterpretation: adaptive mesh lifecycle is covered independently from the fixed-mesh analytical gates; this is a debug correctness gate, not a timing claim.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "adaptive_nonlinear_lambdify_routes_preserve_refinement_and_solution_contract",
            report,
        ) {
            eprintln!("[BVP test report] unable to write adaptive Lambdify report: {error}");
        }
    }

    #[test]
    fn nonuniform_mesh_analytical_solution_is_preserved_across_lambdify_routes() {
        let mesh = vec![0.0, 0.03, 0.1, 0.23, 0.48, 0.72, 1.0];
        let mut reference: Option<DMatrix<f64>> = None;
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = nonuniform_linear_fixture(route, frontend, &mesh);
                solver
                    .try_eq_generate(Some(mesh.clone()), None)
                    .unwrap_or_else(|error| {
                        panic!(
                            "nonuniform preparation failed for {frontend:?}/{}: {error:?}",
                            route.label()
                        )
                    });
                solver.try_solver_prepared().unwrap_or_else(|error| {
                    panic!(
                        "nonuniform solve failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });
                assert_eq!(solver.x_mesh.as_slice(), mesh.as_slice());
                let solution = solver
                    .get_result()
                    .expect("nonuniform route should publish a result")
                    .clone();
                assert_eq!(solution.nrows(), mesh.len());
                assert!(solution.iter().all(|value| value.is_finite()));
                for (row, x) in mesh.iter().copied().enumerate() {
                    assert!((solution[(row, 0)] - x).abs() < 1e-8);
                    assert!((solution[(row, 1)] - 1.0).abs() < 1e-8);
                }
                if let Some(reference) = &reference {
                    let max_diff = reference
                        .iter()
                        .zip(solution.iter())
                        .map(|(lhs, rhs)| (lhs - rhs).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        max_diff < 1e-8,
                        "nonuniform frontend/backend drift for {frontend:?}/{}: {max_diff:e}",
                        route.label()
                    );
                } else {
                    reference = Some(solution);
                }
            }
        }
        let report = "status: passed\n\nfixture: linear analytical BVP y'=z, z'=0, y(0)=0, y(1)=1\nmesh: nonuniform [0.0, 0.03, 0.1, 0.23, 0.48, 0.72, 1.0]\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: custom mesh preservation, finite result, exact node values and frontend/backend parity\nlimit: node errors and parity < 1e-8\ninterpretation: the pure-Lambdify mesh plumbing preserves a nonuniform analytical solution; this is a debug correctness gate, not a timing claim.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "nonuniform_mesh_analytical_solution_is_preserved_across_lambdify_routes",
            report,
        ) {
            eprintln!("[BVP test report] unable to write nonuniform-mesh report: {error}");
        }
    }

    #[test]
    fn pure_lambdify_routes_publish_typed_generation_and_callback_telemetry() {
        let mut rows = Vec::new();
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = nonlinear_exact_fixture(route, frontend, 12);
                solver.set_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Detailed);
                solver.try_eq_generate(None, None).unwrap_or_else(|error| {
                    panic!(
                        "telemetry regeneration failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });
                solver.try_solver_prepared().unwrap_or_else(|error| {
                    panic!(
                        "telemetry route failed for {frontend:?}/{}: {error:?}",
                        route.label()
                    )
                });

                let stats = solver.get_statistics();
                let generation = stats
                    .telemetry
                    .generation
                    .expect("prepared Lambdify route should publish generation telemetry");
                let callback = match route {
                    MatrixRoute::Sparse => Some(match frontend {
                        BvpSymbolicAssemblyBackend::ExprLegacy => stats
                            .telemetry
                            .legacy_lambdify
                            .expect("Sparse ExprLegacy route should publish callback telemetry"),
                        BvpSymbolicAssemblyBackend::AtomView => stats
                            .telemetry
                            .atom_lambdify
                            .expect("Sparse AtomView route should publish callback telemetry"),
                    }),
                    MatrixRoute::Banded => None,
                };
                let direct_banded = stats.telemetry.direct_banded_jacobian;
                if let Some(callback) = callback {
                    assert_eq!(callback.mode, BvpLambdifyTelemetryMode::Detailed);
                    assert!(callback.residual_calls > 0);
                    assert!(callback.jacobian_calls > 0);
                } else {
                    assert!(stats.telemetry.timings.jacobian > std::time::Duration::ZERO);
                    assert!(stats.telemetry.counters.jacobian_requests > 0);
                    let direct = direct_banded
                        .expect("Banded route should publish direct callback telemetry");
                    assert!(direct.calls > 0);
                    assert!(direct.evaluator_calls > 0);
                    assert!(direct.storage_writes > 0);
                    assert!(direct.elapsed > std::time::Duration::ZERO);
                    assert!(direct.parallel_dispatches + direct.sequential_dispatches > 0);
                    assert!(direct.diagonal_dispatches + direct.entry_dispatches > 0);
                }
                assert!(generation.total > std::time::Duration::ZERO);
                if let Some(direct) = direct_banded {
                    rows.push(format!(
                        "frontend={frontend:?}; route={}; generation_total_ms={:.3}; callback_telemetry=direct-banded; direct_calls={}; evaluator_calls={}; storage_writes={}; effective_task_count={}; parallel_dispatches={}; sequential_dispatches={}; diagonal_dispatches={}; entry_dispatches={}; evaluator_ms={:.3}; storage_write_ms={:.3}",
                        route.label(),
                        generation.total.as_secs_f64() * 1_000.0,
                        direct.calls,
                        direct.evaluator_calls,
                        direct.storage_writes,
                        direct.effective_task_count,
                        direct.parallel_dispatches,
                        direct.sequential_dispatches,
                        direct.diagonal_dispatches,
                        direct.entry_dispatches,
                        direct.evaluator_elapsed.as_secs_f64() * 1_000.0,
                        direct.storage_write_elapsed.as_secs_f64() * 1_000.0,
                    ));
                } else {
                    rows.push(format!(
                        "frontend={frontend:?}; route={}; generation_total_ms={:.3}; callback_telemetry=frontend-stream",
                        route.label(),
                        generation.total.as_secs_f64() * 1_000.0,
                    ));
                }
            }
        }

        let report = format!(
            "status: passed\n\nfixture: nonlinear exact BVP y'=z, z'=2*y^3, n_steps=12\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\ntelemetry_mode: Detailed\nchecks: typed generation snapshot for every route; frontend-specific callback snapshot for Sparse; solver-level direct no-Mutex callback snapshot for Banded\nrows:\n{}\ninterpretation: pure-Lambdify preparation telemetry is available after a prepared solve. Sparse exposes frontend callback streams; Banded now projects direct evaluator, storage-write, dispatch and elapsed-time stages into the solver-level typed snapshot. This is a debug diagnostics gate, not a timing benchmark.\n",
            rows.join("\n")
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "pure_lambdify_routes_publish_typed_generation_and_callback_telemetry",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write Lambdify telemetry report: {error}");
        }
    }
}
