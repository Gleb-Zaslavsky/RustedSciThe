//! Prepared numeric-parameter correctness gates for the pure Lambdify route.
//!
//! The symbolic problem, discretization and callbacks are prepared once. The
//! test then replaces values for a parameter that is not a Newton unknown and
//! verifies that Dense, faer Sparse and native Banded callbacks observe the
//! new values without another `try_eq_generate` call.
//!
//! ```text
//! cargo test --lib --no-default-features numerical::BVP_Damp::test_parameter_rebind -- --nocapture --test-threads=1
//! ```

#[cfg(test)]
mod tests {
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use faer::Col;
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    #[derive(Clone, Copy)]
    enum Route {
        Dense,
        Sparse,
        Banded,
    }

    impl Route {
        fn name(self) -> &'static str {
            match self {
                Self::Dense => "Dense",
                Self::Sparse => "faer-Sparse",
                Self::Banded => "Banded",
            }
        }

        fn options(self, frontend: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
            let options = match self {
                Self::Dense => DampedSolverOptions::dense_damped(),
                Self::Sparse => DampedSolverOptions::sparse_damped(),
                Self::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
            };
            let config = options
                .generated_backend_config
                .clone()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(frontend);
            options
                .with_generated_backend_config(config)
                .with_strategy_params(Some(SolverParams::default()))
                .with_abs_tolerance(1e-8)
                .with_rel_tolerance(HashMap::from([
                    ("y".to_string(), 1e-6),
                    ("z".to_string(), 1e-6),
                ]))
                .with_bounds(HashMap::from([
                    ("y".to_string(), (-10.0, 10.0)),
                    ("z".to_string(), (-10.0, 10.0)),
                ]))
                .with_max_iterations(8)
        }
    }

    fn build_solver(route: Route, frontend: BvpSymbolicAssemblyBackend) -> NRBVP {
        let mut solver = NRBVP::new_with_options(
            vec![
                Expr::parse_expression("alpha*y-z"),
                Expr::parse_expression("-z"),
            ],
            DMatrix::from_element(2, 5, 0.25),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, 0.0f64)]),
                ("z".to_string(), vec![(0usize, 1.0f64)]),
            ]),
            0.0,
            1.0,
            5,
            route.options(frontend),
        );
        solver.dont_save_log(true);
        solver
    }

    fn residual_at(solver: &NRBVP) -> Vec<f64> {
        solver
            .fun
            .call(0.0, &*solver.y)
            .to_DVectorType()
            .iter()
            .copied()
            .collect()
    }

    #[test]
    fn prepared_numeric_rebind_updates_dense_sparse_and_banded_callbacks() {
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [Route::Dense, Route::Sparse, Route::Banded] {
                let mut solver = build_solver(route, frontend);
                solver
                    .try_set_params(Some(&["alpha"]))
                    .expect("parameter name should be accepted");
                solver
                    .try_set_param_values(Some(vec![2.0]))
                    .expect("initial numeric parameter should be accepted");
                solver.try_eq_generate(None, None).unwrap_or_else(|error| {
                    panic!("{} preparation failed: {error:?}", route.name())
                });
                let state_len = solver.values.len() * solver.n_steps;
                solver.y = match route {
                    Route::Sparse => {
                        Box::new(Col::from_fn(state_len, |index| 0.25 + index as f64 * 0.001))
                    }
                    Route::Dense | Route::Banded => {
                        Box::new(nalgebra::DVector::from_element(state_len, 0.25))
                    }
                };

                let first = residual_at(&solver);
                assert!(solver.prepared_plan_for_diagnostics().is_some());
                solver.try_solver_prepared().unwrap_or_else(|error| {
                    panic!("{} first prepared solve failed: {error:?}", route.name())
                });
                let first_stats = solver.get_statistics();

                solver
                    .try_set_param_values(Some(vec![5.0]))
                    .expect("numeric rebind should be accepted");
                let rebound_stats = solver.get_statistics();
                assert!(
                    rebound_stats.telemetry.counters.factorization_invalidations
                        > first_stats.telemetry.counters.factorization_invalidations,
                    "{frontend:?}/{} numeric rebind must invalidate the old factor before the next solve",
                    route.name()
                );
                let second = residual_at(&solver);
                assert!(solver.prepared_plan_for_diagnostics().is_some());
                solver.try_solver_prepared().unwrap_or_else(|error| {
                    panic!(
                        "{frontend:?}/{} rebound prepared solve failed: {error:?}",
                        route.name()
                    )
                });
                let second_stats = solver.get_statistics();
                assert!(
                    second_stats.telemetry.counters.factorizations
                        > first_stats.telemetry.counters.factorizations,
                    "{} rebound solve must build/use a factor for the new numeric Jacobian",
                    route.name()
                );

                assert_ne!(
                    first,
                    second,
                    "{frontend:?}/{} callback ignored numeric rebind",
                    route.name()
                );
            }
        }
    }
}
