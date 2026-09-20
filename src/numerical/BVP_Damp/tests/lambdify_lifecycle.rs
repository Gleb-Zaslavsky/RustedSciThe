//! P0 lifecycle and typed-error gates for the pure Lambdify BVP route.
//!
//! The historical parity corpus checks numerical answers. This module checks
//! a different contract: a prepared callback/factor runtime may be reused for
//! numeric parameter rebinding, but it must not survive structural changes to
//! the mesh, boundary conditions, backend policy, or public compatibility
//! inputs. The tests intentionally cover only Sparse/faer and native Banded;
//! Dense remains a small control backend elsewhere in the suite.
//!
//! ```text
//! cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle -- --nocapture --test-threads=1
//! ```

#[cfg(test)]
mod tests {
    use crate::Utils::test_reporting::write_test_report;
    use crate::numerical::BVP_Damp::BVP_traits::{
        BandedMatrixType, MatrixType, VectorType, convert_to_fun, convert_to_jac,
    };
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::{
        FrozenSolverOptions, NRBVP as FrozenNRBVP,
    };
    use crate::symbolic::bvp::telemetry::{BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode};
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use faer::sparse::SparseColMat;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum MatrixRoute {
        Sparse,
        Banded,
    }

    impl MatrixRoute {
        const fn label(self) -> &'static str {
            match self {
                Self::Sparse => "Sparse-faer",
                Self::Banded => "Banded",
            }
        }

        fn options(self, frontend: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
            let base = match self {
                Self::Sparse => DampedSolverOptions::sparse_damped(),
                Self::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
            };
            let config = base
                .generated_backend_config
                .clone()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
                .with_symbolic_assembly_backend(frontend);
            base.with_generated_backend_config(config)
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

    fn build_solver(route: MatrixRoute, frontend: BvpSymbolicAssemblyBackend) -> NRBVP {
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

    fn build_solver_with_policy(
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
        policy: BvpLambdifyExecutionPolicy,
    ) -> NRBVP {
        let mut solver = build_solver(route, frontend);
        let config = solver
            .generated_backend_config()
            .clone()
            .with_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Counters)
            .with_lambdify_execution_policy(policy);
        solver.set_generated_backend_config(config);
        solver
    }

    fn sparse_csc_snapshot(solver: &mut NRBVP) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let args = DVector::from_element(solver.values.len() * solver.n_steps, 0.25);
        let typed = crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting(
            &args,
            solver.method.clone(),
        );
        let matrix = solver
            .jac
            .as_mut()
            .expect("prepared Sparse Jacobian should be installed")
            .call(0.5, &*typed);
        let sparse = matrix
            .as_any()
            .downcast_ref::<SparseColMat<usize, f64>>()
            .expect("Sparse route must publish faer SparseColMat");
        let symbolic = sparse.symbolic();
        (
            symbolic.col_ptr().to_vec(),
            symbolic.row_idx().to_vec(),
            sparse.val().to_vec(),
        )
    }

    fn banded_slot_snapshot(solver: &mut NRBVP) -> Vec<(isize, Vec<f64>)> {
        let args = DVector::from_element(solver.values.len() * solver.n_steps, 0.25);
        let typed = crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting(
            &args,
            solver.method.clone(),
        );
        let matrix = solver
            .jac
            .as_mut()
            .expect("prepared Banded Jacobian should be installed")
            .call(0.5, &*typed);
        let banded = matrix
            .as_any()
            .downcast_ref::<BandedMatrixType>()
            .expect("Banded route must publish native BandedMatrixType");
        banded
            .assembly
            .offsets()
            .map(|offset| {
                (
                    offset,
                    banded
                        .assembly
                        .diag(offset)
                        .expect("declared Banded diagonal should exist")
                        .to_vec(),
                )
            })
            .collect()
    }

    fn prepare(solver: &mut NRBVP) {
        solver
            .try_set_params(Some(&["alpha"]))
            .expect("parameter name should be accepted");
        solver
            .try_set_param_values(Some(vec![2.0]))
            .expect("parameter value should be accepted");
        solver
            .try_eq_generate(None, None)
            .expect("pure Lambdify preparation should succeed");
    }

    fn expect_stale(solver: &mut NRBVP, reason: &str) {
        let error = solver
            .try_solver_prepared()
            .expect_err("a structural mutation must reject the stale prepared runtime");
        assert!(
            matches!(
                error,
                BvpBackendIntegrationError::PreparedRuntimeInvalidated { .. }
            ),
            "{reason} returned an unexpected error: {error:?}"
        );
    }

    fn build_frozen_solver(
        route: MatrixRoute,
        frontend: BvpSymbolicAssemblyBackend,
    ) -> FrozenNRBVP {
        let base = match route {
            MatrixRoute::Sparse => FrozenSolverOptions::sparse_frozen(),
            MatrixRoute::Banded => FrozenSolverOptions::banded_frozen().with_banded_lambdify(),
        };
        let config = base
            .generated_backend_config
            .clone()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(frontend);
        let options = base.with_generated_backend_config(config);
        let mut solver = FrozenNRBVP::new_with_options(
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
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn prepare_frozen(solver: &mut FrozenNRBVP) {
        solver
            .try_set_params(Some(&["alpha"]))
            .expect("Frozen parameter name should be accepted");
        solver
            .try_set_param_values(Some(vec![2.0]))
            .expect("Frozen parameter value should be accepted");
        solver
            .try_eq_generate()
            .expect("Frozen pure Lambdify preparation should succeed");
    }

    fn expect_frozen_stale(solver: &mut FrozenNRBVP, reason: &str) {
        let error = solver
            .try_solver_prepared()
            .expect_err("a Frozen structural mutation must reject stale runtime");
        assert!(
            matches!(
                error,
                BvpBackendIntegrationError::PreparedRuntimeInvalidated { .. }
            ),
            "Frozen {reason} returned an unexpected error: {error:?}"
        );
    }

    #[test]
    fn prepared_lambdify_rebind_and_structural_invalidation_matrix() {
        let mut rows = Vec::new();

        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = build_solver(route, frontend);
                prepare(&mut solver);
                solver
                    .try_solver_prepared()
                    .expect("initial prepared solve should succeed");
                let initial_stats = solver.get_statistics();
                let initial_invalidations =
                    initial_stats.telemetry.counters.factorization_invalidations;

                solver
                    .try_set_param_values(Some(vec![5.0]))
                    .expect("numeric parameter rebind should be accepted");
                solver
                    .try_solver_prepared()
                    .expect("numeric rebind must preserve prepared callbacks");
                let rebound_stats = solver.get_statistics();
                assert!(
                    rebound_stats.telemetry.counters.factorization_invalidations
                        > initial_invalidations,
                    "{frontend:?}/{} must invalidate numeric factors after rebind",
                    route.label()
                );

                solver.set_mesh(0.0, 1.0, 6);
                expect_stale(&mut solver, "mesh");
                solver
                    .try_eq_generate(None, None)
                    .expect("mesh regeneration should restore the prepared runtime");
                solver
                    .try_solver_prepared()
                    .expect("regenerated mesh should be solvable");

                solver.set_boundary_conditions(HashMap::from([
                    ("y".to_string(), vec![(0usize, 0.1f64)]),
                    ("z".to_string(), vec![(0usize, 1.0f64)]),
                ]));
                expect_stale(&mut solver, "boundary conditions");
                solver
                    .try_eq_generate(None, None)
                    .expect("boundary-condition regeneration should succeed");

                solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
                expect_stale(&mut solver, "backend policy");
                solver.set_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
                solver
                    .try_eq_generate(None, None)
                    .expect("restoring Lambdify policy should rebuild callbacks");

                solver.values[0] = "renamed_y".to_string();
                expect_stale(&mut solver, "public values mutation");
                solver.values[0] = "y".to_string();
                solver
                    .try_eq_generate(None, None)
                    .expect("restoring public compatibility input should succeed");
                solver
                    .try_solver_prepared()
                    .expect("final regenerated runtime should be usable");

                rows.push(format!(
                    "frontend={frontend:?}; route={}; numeric_rebind=accepted; stale_mesh=typed; stale_bc=typed; stale_policy=typed; stale_public_values=typed",
                    route.label()
                ));
            }
        }

        let report = format!(
            "status: passed\n\nfixture: alpha*y-z, -z; initial n_steps=5; regenerated mesh n_steps=6\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: numeric rebind keeps prepared callbacks; mesh, boundary conditions, backend policy and direct public values mutation reject stale prepared solves with PreparedRuntimeInvalidated; regeneration restores solve path\nrows:\n{}\ninterpretation: numeric parameter values are runtime bindings and invalidate factors but not symbolic callbacks; structural compatibility changes require explicit regeneration.\n",
            rows.join("\n")
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "prepared_lambdify_rebind_and_structural_invalidation_matrix",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write lifecycle report: {error}");
        }
    }

    #[test]
    fn solver_try_calc_residual_returns_typed_shape_and_nonfinite_errors() {
        let mut solver = build_solver(MatrixRoute::Sparse, BvpSymbolicAssemblyBackend::AtomView);
        prepare(&mut solver);
        let expected_len = solver.y.len();

        solver.fun = convert_to_fun(Box::new(|_, _| {
            Box::new(DVector::from_element(1, 0.0)) as Box<dyn VectorType>
        }));
        let shape_error = solver
            .try_calc_residual(solver.y.clone_box())
            .expect_err("shape mismatch must be returned as a typed error");
        assert!(matches!(
            shape_error,
            BvpBackendIntegrationError::CallbackShapeMismatch {
                stage,
                expected_rows,
                actual_rows: 1,
                ..
            } if stage == "residual" && expected_rows == expected_len
        ));

        solver.fun = convert_to_fun(Box::new(move |_, _| {
            Box::new(DVector::from_element(expected_len, f64::NAN)) as Box<dyn VectorType>
        }));
        let nonfinite_error = solver
            .try_calc_residual(solver.y.clone_box())
            .expect_err("non-finite callback output must be typed");
        assert!(matches!(
            nonfinite_error,
            BvpBackendIntegrationError::NonFiniteCallbackValue { stage, .. }
                if stage == "residual"
        ));

        let report = "status: passed\n\nchecks: solver-level residual shape mismatch and non-finite callback values are returned as BvpBackendIntegrationError without panic\ninterpretation: public try_* boundary is safe for malformed callback outputs; compatibility calc_residual remains the only panic wrapper.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "solver_try_calc_residual_returns_typed_shape_and_nonfinite_errors",
            report,
        ) {
            eprintln!("[BVP test report] unable to write typed-error report: {error}");
        }
    }

    #[test]
    fn solver_try_recalculate_jacobian_returns_typed_shape_error() {
        let mut solver = build_solver(MatrixRoute::Sparse, BvpSymbolicAssemblyBackend::AtomView);
        prepare(&mut solver);
        let expected_len = solver.y.len();
        solver.jac = Some(convert_to_jac(Box::new(|_, _| {
            Box::new(DMatrix::from_element(1, 1, 1.0)) as Box<dyn MatrixType>
        })));
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));

        let error = solver
            .try_recalculate_jacobian()
            .expect_err("malformed Jacobian output must be returned as a typed error");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::CallbackShapeMismatch {
                stage,
                expected_rows,
                expected_columns,
                actual_rows: 1,
                actual_columns: 1,
            } if stage == "Jacobian"
                && expected_rows == expected_len
                && expected_columns == expected_len
        ));

        let report = "status: passed\n\nchecks: public try_recalculate_jacobian returns CallbackShapeMismatch for malformed Jacobian output without panic\ninterpretation: residual and Jacobian callback shape failures now have symmetric public typed boundaries.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "solver_try_recalculate_jacobian_returns_typed_shape_error",
            report,
        ) {
            eprintln!("[BVP test report] unable to write Jacobian error report: {error}");
        }
    }

    #[test]
    fn frozen_prepared_lambdify_rebind_and_structural_invalidation_matrix() {
        let mut rows = Vec::new();

        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = build_frozen_solver(route, frontend);
                prepare_frozen(&mut solver);
                solver
                    .try_solver_prepared()
                    .expect("Frozen initial prepared solve should succeed");

                solver
                    .try_set_param_values(Some(vec![5.0]))
                    .expect("Frozen numeric rebind should be accepted");
                solver
                    .try_solver_prepared()
                    .expect("Frozen numeric rebind must preserve callbacks");

                solver.set_boundary_conditions(HashMap::from([
                    ("y".to_string(), vec![(0usize, 0.1f64)]),
                    ("z".to_string(), vec![(0usize, 1.0f64)]),
                ]));
                expect_frozen_stale(&mut solver, "boundary conditions");
                solver
                    .try_eq_generate()
                    .expect("Frozen BC regeneration should succeed");

                solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
                expect_frozen_stale(&mut solver, "backend policy");
                solver.set_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
                solver
                    .try_eq_generate()
                    .expect("restoring Frozen Lambdify policy should succeed");

                solver.values[0] = "renamed_y".to_string();
                expect_frozen_stale(&mut solver, "public values mutation");
                solver.values[0] = "y".to_string();
                solver
                    .try_eq_generate()
                    .expect("restoring Frozen public compatibility input should succeed");
                solver
                    .try_solver_prepared()
                    .expect("final Frozen prepared runtime should be usable");

                rows.push(format!(
                    "frontend={frontend:?}; route={}; numeric_rebind=accepted; stale_bc=typed; stale_policy=typed; stale_public_values=typed",
                    route.label()
                ));
            }
        }

        let report = format!(
            "status: passed\n\nfixture: alpha*y-z, -z; Frozen strategy; n_steps=5\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: numeric rebind preserves prepared callbacks; BC, backend policy and direct public values mutation reject stale Frozen solves; regeneration restores the prepared path\nrows:\n{}\ninterpretation: Frozen shares the typed prepared invalidation boundary with Damped. Frozen mesh mutation remains a separate API gap because no public set_mesh contract exists yet.\n",
            rows.join("\n")
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "frozen_prepared_lambdify_rebind_and_structural_invalidation_matrix",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write Frozen lifecycle report: {error}");
        }
    }

    #[test]
    fn solver_level_lambdify_execution_policy_preserves_parity_and_dispatch() {
        let mut rows = Vec::new();

        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut sequential = build_solver_with_policy(
                    route,
                    frontend,
                    BvpLambdifyExecutionPolicy::Sequential,
                );
                let mut parallel = build_solver_with_policy(
                    route,
                    frontend,
                    BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
                );
                prepare(&mut sequential);
                prepare(&mut parallel);
                assert_eq!(
                    sequential
                        .generated_backend_config()
                        .lambdify_execution_policy,
                    BvpLambdifyExecutionPolicy::Sequential
                );
                assert_eq!(
                    parallel
                        .generated_backend_config()
                        .lambdify_execution_policy,
                    BvpLambdifyExecutionPolicy::Parallel { min_work: 0 }
                );
                sequential
                    .try_solver_prepared()
                    .expect("sequential Lambdify solve should succeed");
                parallel
                    .try_solver_prepared()
                    .expect("parallel Lambdify solve should succeed");

                let sequential_values = sequential.y.to_DVectorType();
                let parallel_values = parallel.y.to_DVectorType();
                assert_eq!(sequential_values.len(), parallel_values.len());
                let max_diff = sequential_values
                    .iter()
                    .zip(parallel_values.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_diff < 1e-10,
                    "{frontend:?}/{} execution policies changed the solution: diff={max_diff:e}",
                    route.label()
                );

                let sequential_stats = sequential.get_statistics();
                let parallel_stats = parallel.get_statistics();
                if route == MatrixRoute::Banded {
                    let sequential_dispatch = sequential_stats
                        .telemetry
                        .direct_banded_jacobian
                        .expect("Banded solver should publish direct telemetry");
                    let parallel_dispatch = parallel_stats
                        .telemetry
                        .direct_banded_jacobian
                        .expect("Banded solver should publish direct telemetry");
                    assert_eq!(sequential_dispatch.parallel_dispatches, 0);
                    assert!(sequential_dispatch.sequential_dispatches > 0);
                    assert!(parallel_dispatch.parallel_dispatches > 0);
                    assert_eq!(parallel_dispatch.sequential_dispatches, 0);
                } else {
                    let sequential_stream = match frontend {
                        BvpSymbolicAssemblyBackend::ExprLegacy => sequential_stats
                            .telemetry
                            .legacy_lambdify
                            .expect("ExprLegacy telemetry should be published"),
                        BvpSymbolicAssemblyBackend::AtomView => sequential_stats
                            .telemetry
                            .atom_lambdify
                            .expect("AtomView telemetry should be published"),
                    };
                    let parallel_stream = match frontend {
                        BvpSymbolicAssemblyBackend::ExprLegacy => parallel_stats
                            .telemetry
                            .legacy_lambdify
                            .expect("ExprLegacy telemetry should be published"),
                        BvpSymbolicAssemblyBackend::AtomView => parallel_stats
                            .telemetry
                            .atom_lambdify
                            .expect("AtomView telemetry should be published"),
                    };
                    assert!(sequential_stream.sequential_dispatches > 0);
                    assert_eq!(sequential_stream.parallel_dispatches, 0);
                    assert!(parallel_stream.parallel_dispatches > 0);
                    assert_eq!(parallel_stream.sequential_dispatches, 0);
                }
                rows.push(format!(
                    "frontend={frontend:?}; route={}; sequential=sequential; parallel=parallel; max_solution_diff={max_diff:e}",
                    route.label()
                ));
            }
        }

        let report = format!(
            "status: passed\n\nfixture: alpha*y-z, -z; n_steps=5\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\npolicies: Sequential vs Parallel {{ min_work: 0 }}\nchecks: solver-level policy reaches prepared callback families; dispatch telemetry identifies the selected branch; final solutions remain componentwise equivalent\nrows:\n{}\ninterpretation: execution policy is an explicit pure-Lambdify solver option, independent of AOT policy. No performance claim is made in this debug gate.\n",
            rows.join("\n")
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "solver_level_lambdify_execution_policy_preserves_parity_and_dispatch",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write execution-policy report: {error}");
        }
    }

    #[test]
    fn prepared_lambdify_rebind_rebuilds_factor_for_all_routes() {
        let mut rows = Vec::new();

        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut solver = build_solver_with_policy(
                    route,
                    frontend,
                    BvpLambdifyExecutionPolicy::Sequential,
                );
                prepare(&mut solver);

                solver
                    .try_solver_prepared()
                    .expect("initial prepared solve should succeed");
                let first = solver.get_statistics().telemetry.counters;

                solver
                    .try_solver_prepared()
                    .expect("unchanged prepared solve should succeed");
                let repeated = solver.get_statistics().telemetry.counters;
                assert!(
                    repeated.factorizations >= first.factorizations,
                    "{frontend:?}/{} must keep factorization counters monotonic",
                    route.label()
                );

                solver
                    .try_set_param_values(Some(vec![3.0]))
                    .expect("numeric rebind should be accepted");
                solver
                    .try_solver_prepared()
                    .expect("rebound prepared solve should succeed");
                let rebound = solver.get_statistics().telemetry.counters;
                assert!(
                    rebound.factorizations > repeated.factorizations,
                    "{frontend:?}/{} must refactor after numeric parameter rebind",
                    route.label()
                );
                assert!(
                    rebound.factorization_invalidations > first.factorization_invalidations,
                    "{frontend:?}/{} must record factor invalidation after rebind",
                    route.label()
                );

                rows.push(format!(
                    "frontend={frontend:?}; route={}; first_factorizations={}; repeated_factorizations={}; repeated_cache_hits={}; rebound_factorizations={}; invalidations={}",
                    route.label(),
                    first.factorizations,
                    repeated.factorizations,
                    repeated.factorization_cache_hits,
                    rebound.factorizations,
                    rebound.factorization_invalidations
                ));
            }
        }

        let report = format!(
            "status: passed\n\nfixture: alpha*y-z, -z; n_steps=5\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: repeated prepared solves remain valid; numeric parameter rebind invalidates and rebuilds the factor; cache-hit counters remain monotonic\nrows:\n{}\ninterpretation: prepared callbacks survive repeated solver-level use, while numeric rebinding cannot consume a stale factor. Factor reuse across separate solver-level solves is intentionally a separate P1 optimization question; cache reuse inside one solve is covered by the factorization-cache suite.\n",
            rows.join("\n")
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "prepared_lambdify_rebind_rebuilds_factor_for_all_routes",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write factor-reuse report: {error}");
        }
    }

    #[test]
    fn solver_level_lambdify_parallel_threshold_falls_back_without_drift() {
        let mut rows = Vec::new();

        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            for route in [MatrixRoute::Sparse, MatrixRoute::Banded] {
                let mut forced_sequential = build_solver_with_policy(
                    route,
                    frontend,
                    BvpLambdifyExecutionPolicy::Parallel {
                        min_work: usize::MAX,
                    },
                );
                let mut parallel = build_solver_with_policy(
                    route,
                    frontend,
                    BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
                );
                prepare(&mut forced_sequential);
                prepare(&mut parallel);
                forced_sequential
                    .try_solver_prepared()
                    .expect("threshold fallback solve should succeed");
                parallel
                    .try_solver_prepared()
                    .expect("parallel solve should succeed");

                let threshold_values = forced_sequential.y.to_DVectorType();
                let parallel_values = parallel.y.to_DVectorType();
                let max_diff = threshold_values
                    .iter()
                    .zip(parallel_values.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_diff < 1e-10,
                    "{frontend:?}/{} threshold fallback changed the solution: diff={max_diff:e}",
                    route.label()
                );

                if route == MatrixRoute::Banded {
                    let threshold_dispatch = forced_sequential
                        .get_statistics()
                        .telemetry
                        .direct_banded_jacobian
                        .expect("Banded threshold solve should publish direct telemetry");
                    let parallel_dispatch = parallel
                        .get_statistics()
                        .telemetry
                        .direct_banded_jacobian
                        .expect("Banded parallel solve should publish direct telemetry");
                    assert_eq!(threshold_dispatch.parallel_dispatches, 0);
                    assert!(threshold_dispatch.sequential_dispatches > 0);
                    assert!(parallel_dispatch.parallel_dispatches > 0);
                } else {
                    let threshold_stream = match frontend {
                        BvpSymbolicAssemblyBackend::ExprLegacy => forced_sequential
                            .get_statistics()
                            .telemetry
                            .legacy_lambdify
                            .expect("ExprLegacy threshold telemetry should be published"),
                        BvpSymbolicAssemblyBackend::AtomView => forced_sequential
                            .get_statistics()
                            .telemetry
                            .atom_lambdify
                            .expect("AtomView threshold telemetry should be published"),
                    };
                    let parallel_stream = match frontend {
                        BvpSymbolicAssemblyBackend::ExprLegacy => parallel
                            .get_statistics()
                            .telemetry
                            .legacy_lambdify
                            .expect("ExprLegacy parallel telemetry should be published"),
                        BvpSymbolicAssemblyBackend::AtomView => parallel
                            .get_statistics()
                            .telemetry
                            .atom_lambdify
                            .expect("AtomView parallel telemetry should be published"),
                    };
                    assert_eq!(threshold_stream.parallel_dispatches, 0);
                    assert!(threshold_stream.sequential_dispatches > 0);
                    assert!(parallel_stream.parallel_dispatches > 0);
                }

                rows.push(format!(
                    "frontend={frontend:?}; route={}; min_work=usize::MAX->sequential; min_work=0->parallel; max_solution_diff={max_diff:e}",
                    route.label()
                ));
            }
        }

        let report = format!(
            "status: passed\n\nfixture: alpha*y-z, -z; n_steps=5\nfrontends: ExprLegacy, AtomView\nmatrix_routes: Sparse-faer, Banded\nchecks: solver-level Parallel min_work fallback, explicit parallel dispatch, solution parity\nrows:\n{}\ninterpretation: the current execution-policy contract has an observable sequential fallback for work below min_work. Auto calibration/chunk strategy selection remains a separate performance task.\n",
            rows.join("\n")
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "solver_level_lambdify_parallel_threshold_falls_back_without_drift",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write threshold report: {error}");
        }
    }

    #[test]
    fn sparse_lambdify_fixed_csc_pattern_is_frontend_stable_after_rebind() {
        let mut snapshots = Vec::new();
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            let mut solver = build_solver(MatrixRoute::Sparse, frontend);
            prepare(&mut solver);
            let before = sparse_csc_snapshot(&mut solver);

            solver
                .try_set_param_values(Some(vec![3.0]))
                .expect("numeric rebind should be accepted");
            solver
                .try_eq_generate(None, None)
                .expect("rebind should regenerate numeric Sparse values");
            let after = sparse_csc_snapshot(&mut solver);

            assert_eq!(
                before.0, after.0,
                "numeric rebind must preserve fixed CSC column pointers for {frontend:?}"
            );
            assert_eq!(
                before.1, after.1,
                "numeric rebind must preserve fixed CSC row order for {frontend:?}"
            );
            assert!(
                before
                    .2
                    .iter()
                    .zip(after.2.iter())
                    .any(|(old, new)| (old - new).abs() > 1e-12),
                "numeric rebind must change at least one Jacobian value for {frontend:?}"
            );
            snapshots.push((frontend, after));
        }

        assert_eq!(
            snapshots[0].1.0, snapshots[1].1.0,
            "ExprLegacy and AtomView must publish identical CSC column pointers"
        );
        assert_eq!(
            snapshots[0].1.1, snapshots[1].1.1,
            "ExprLegacy and AtomView must publish identical CSC row order"
        );
        let max_value_diff = snapshots[0]
            .1
            .2
            .iter()
            .zip(snapshots[1].1.2.iter())
            .map(|(legacy, atom)| (legacy - atom).abs())
            .fold(0.0, f64::max);
        assert!(
            max_value_diff < 1e-10,
            "ExprLegacy/AtomView fixed-CSC values drifted: {max_value_diff:.3e}"
        );

        let report = "status: passed\n\nfixture: parameterized alpha*y-z, -z; n_steps=5\nfrontends: ExprLegacy, AtomView\nroute: Sparse-faer\nchecks: fixed CSC col_ptr/row_idx parity, deterministic row order after numeric rebind, changed numeric values, cross-frontend value parity\ninterpretation: numeric parameter rebinding preserves the prepared sparse pattern and does not silently reuse old numeric values; this is a debug lifecycle gate, not a performance claim.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "sparse_lambdify_fixed_csc_pattern_is_frontend_stable_after_rebind",
            report,
        ) {
            eprintln!("[BVP test report] unable to write fixed-CSC report: {error}");
        }
    }

    #[test]
    fn banded_lambdify_slots_are_frontend_stable_and_rebind_refactors() {
        let mut frontend_snapshots = Vec::new();
        for frontend in [
            BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSymbolicAssemblyBackend::AtomView,
        ] {
            let mut solver = build_solver(MatrixRoute::Banded, frontend);
            prepare(&mut solver);
            let before = banded_slot_snapshot(&mut solver);
            solver
                .try_set_param_values(Some(vec![3.0]))
                .expect("numeric rebind should be accepted");
            solver
                .try_eq_generate(None, None)
                .expect("rebind should regenerate native Banded values");
            let after = banded_slot_snapshot(&mut solver);

            assert_eq!(
                before.iter().map(|(offset, _)| offset).collect::<Vec<_>>(),
                after.iter().map(|(offset, _)| offset).collect::<Vec<_>>(),
                "numeric rebind must preserve Banded diagonal offsets for {frontend:?}"
            );
            assert_eq!(
                before
                    .iter()
                    .map(|(_, values)| values.len())
                    .collect::<Vec<_>>(),
                after
                    .iter()
                    .map(|(_, values)| values.len())
                    .collect::<Vec<_>>(),
                "numeric rebind must preserve Banded slot lengths for {frontend:?}"
            );
            assert!(
                before
                    .iter()
                    .zip(after.iter())
                    .flat_map(|((_, old), (_, new))| old.iter().zip(new.iter()))
                    .any(|(old, new)| (old - new).abs() > 1e-12),
                "numeric rebind must change at least one Banded slot for {frontend:?}"
            );
            frontend_snapshots.push((frontend, after));
        }

        assert_eq!(
            frontend_snapshots[0]
                .1
                .iter()
                .map(|(offset, _)| offset)
                .collect::<Vec<_>>(),
            frontend_snapshots[1]
                .1
                .iter()
                .map(|(offset, _)| offset)
                .collect::<Vec<_>>(),
            "ExprLegacy and AtomView must publish identical Banded offsets"
        );
        let max_value_diff = frontend_snapshots[0]
            .1
            .iter()
            .zip(frontend_snapshots[1].1.iter())
            .flat_map(|((_, legacy), (_, atom))| legacy.iter().zip(atom.iter()))
            .map(|(legacy, atom)| (legacy - atom).abs())
            .fold(0.0, f64::max);
        assert!(
            max_value_diff < 1e-10,
            "ExprLegacy/AtomView Banded slot values drifted: {max_value_diff:.3e}"
        );

        let mut solver = build_solver(MatrixRoute::Banded, BvpSymbolicAssemblyBackend::AtomView);
        prepare(&mut solver);
        solver
            .try_solver_prepared()
            .expect("initial Banded solve should succeed");
        let before_rebind = solver.get_statistics().telemetry.counters;
        solver
            .try_set_param_values(Some(vec![3.0]))
            .expect("factor invalidation rebind should be accepted");
        let after_rebind = solver.get_statistics().telemetry.counters;
        assert!(
            after_rebind.factorization_invalidations > before_rebind.factorization_invalidations,
            "Banded rebind must invalidate the previous factor"
        );
        solver
            .try_eq_generate(None, None)
            .expect("Banded factor-invalidated state should regenerate");
        solver
            .try_solver_prepared()
            .expect("Banded solve after rebind should succeed");
        let after_second_solve = solver.get_statistics().telemetry.counters;
        assert!(
            after_second_solve.factorizations > before_rebind.factorizations,
            "Banded solve after rebind must build a new factor"
        );

        let report = "status: passed\n\nfixture: parameterized alpha*y-z, -z; n_steps=5\nfrontends: ExprLegacy, AtomView\nroute: native Banded\nchecks: diagonal offsets and slot lengths remain stable after numeric rebind, cross-frontend slot value parity, factorization invalidation and rebuild counters\ninterpretation: native Banded storage remains structurally stable while numeric rebinding changes values and forces a new factor; this is a debug lifecycle gate, not a performance claim.\n";
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "banded_lambdify_slots_are_frontend_stable_and_rebind_refactors",
            report,
        ) {
            eprintln!("[BVP test report] unable to write Banded slot report: {error}");
        }
    }
}
