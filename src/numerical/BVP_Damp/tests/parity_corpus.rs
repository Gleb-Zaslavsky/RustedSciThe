//! Architecture lane: deterministic ExprLegacy versus AtomView correctness.
//!
//! These tests intentionally stay small and debug-friendly. They use the same
//! equations, mesh, boundary conditions, initial guess, tolerances, and
//! Banded Lambdify route for both symbolic frontends. Runtime speed belongs in
//! the release-only story ledgers; this module only guards numerical parity.

mod tests {
    use crate::Utils::test_reporting::write_test_report;
    use crate::numerical::BVP_Damp::BVP_traits::{
        MatrixType, VectorType, convert_to_fun, convert_to_jac,
    };
    use crate::numerical::BVP_Damp::MatrixBackend;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::numerical::BVP_Damp::telemetry::{
        BvpLogEventKind, BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode, BvpTelemetrySnapshot,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    #[derive(Clone, Copy, Debug)]
    enum MatrixRoute {
        Sparse,
        Banded,
    }

    impl MatrixRoute {
        fn config(self, frontend: BvpSymbolicAssemblyBackend) -> GeneratedBackendConfig {
            let config = match self {
                Self::Sparse => GeneratedBackendConfig::sparse_defaults()
                    .with_matrix_backend_override(MatrixBackend::SparseCol),
                Self::Banded => GeneratedBackendConfig::banded_lambdify_defaults(),
            };
            config
                .with_backend_policy_override(Some(
                    crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy::LambdifyOnly,
                ))
                .with_symbolic_assembly_backend(frontend)
        }

        fn base_options(self) -> DampedSolverOptions {
            match self {
                Self::Sparse => DampedSolverOptions::sparse_damped(),
                Self::Banded => DampedSolverOptions::banded_damped().with_banded_lambdify(),
            }
        }
    }

    fn options(backend: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
        let bounds = HashMap::from([
            ("y".to_string(), (-2.0, 2.0)),
            ("z".to_string(), (-2.0, 2.0)),
        ]);
        let rel_tolerance = HashMap::from([("y".to_string(), 1e-7), ("z".to_string(), 1e-7)]);

        DampedSolverOptions::banded_damped()
            .with_banded_lambdify()
            .with_symbolic_assembly_backend(backend)
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-9)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(30)
            .with_bounds(bounds)
    }

    fn initial_guess<F>(n_steps: usize, t_end: f64, profile: F) -> DMatrix<f64>
    where
        F: Fn(f64) -> (f64, f64),
    {
        let h = t_end / n_steps as f64;
        let values = (0..n_steps)
            .map(|i| {
                let x = i as f64 * h;
                let (y, z) = profile(x);
                [y, z]
            })
            .flatten()
            .collect::<Vec<_>>();
        DMatrix::from_column_slice(2, n_steps, DVector::from_vec(values).as_slice())
    }

    fn build_solver(
        equations: Vec<Expr>,
        guess: DMatrix<f64>,
        t_end: f64,
        boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
        backend: BvpSymbolicAssemblyBackend,
    ) -> NRBVP {
        let mut solver = NRBVP::new_with_options(
            equations,
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            boundary_conditions,
            0.0,
            t_end,
            24,
            options(backend),
        );
        solver.dont_save_log(true);
        solver
    }

    fn solve_pair(
        equations: Vec<Expr>,
        guess: DMatrix<f64>,
        t_end: f64,
        boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let mut legacy = build_solver(
            equations.clone(),
            guess.clone(),
            t_end,
            boundary_conditions.clone(),
            BvpSymbolicAssemblyBackend::ExprLegacy,
        );
        let mut atom = build_solver(
            equations,
            guess,
            t_end,
            boundary_conditions,
            BvpSymbolicAssemblyBackend::AtomView,
        );

        legacy
            .try_solve()
            .expect("ExprLegacy parity fixture should solve");
        atom.try_solve()
            .expect("AtomView parity fixture should solve");

        (
            legacy
                .get_result()
                .expect("ExprLegacy should publish a solution"),
            atom.get_result()
                .expect("AtomView should publish a solution"),
        )
    }

    fn assert_matrix_parity(legacy: &DMatrix<f64>, atom: &DMatrix<f64>, tolerance: f64) {
        assert_eq!(legacy.shape(), atom.shape());
        let max_diff = legacy
            .iter()
            .zip(atom.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff <= tolerance,
            "ExprLegacy/AtomView solution drift: max_diff={max_diff:.3e}, tolerance={tolerance:.3e}"
        );
    }

    fn trace_options(
        backend: BvpSymbolicAssemblyBackend,
        matrix: MatrixRoute,
        adaptive: bool,
    ) -> DampedSolverOptions {
        let bounds = HashMap::from([
            ("y".to_string(), (-10.0, 10.0)),
            ("z".to_string(), (-20.0, 20.0)),
        ]);
        let strategy = SolverParams {
            max_jac: Some(2),
            max_damp_iter: Some(8),
            damp_factor: Some(0.5),
            adaptive: adaptive.then_some(
                crate::numerical::BVP_Damp::NR_Damp_solver_damped::AdaptiveGridConfig {
                    version: 1,
                    max_refinements: 1,
                    grid_method: GridRefinementMethod::DoublePoints,
                },
            ),
        };
        matrix
            .base_options()
            .with_generated_backend_config(matrix.config(backend))
            .with_strategy_params(Some(strategy))
            .with_abs_tolerance(1e-9)
            .with_rel_tolerance(HashMap::from([
                ("y".to_string(), 1e-7),
                ("z".to_string(), 1e-7),
            ]))
            .with_max_iterations(30)
            .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
            .with_bvp_logging_config(
                BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(512),
            )
            .with_bounds(bounds)
    }

    fn solve_with_trace(
        equations: Vec<Expr>,
        guess: DMatrix<f64>,
        backend: BvpSymbolicAssemblyBackend,
        matrix: MatrixRoute,
    ) -> (DMatrix<f64>, BvpTelemetrySnapshot) {
        let boundary_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0f64), (1usize, 0.0f64)])]);
        let mut solver = NRBVP::new_with_options(
            equations,
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            boundary_conditions,
            0.0,
            1.0,
            24,
            trace_options(backend, matrix, true),
        );
        solver.dont_save_log(true);
        solver
            .try_solve()
            .expect("symbolic parity trace fixture should solve");
        let result = solver
            .get_result()
            .expect("symbolic parity trace fixture should publish a result");
        (result, solver.get_statistics().telemetry)
    }

    fn event_trace(snapshot: &BvpTelemetrySnapshot) -> Vec<(BvpLogEventKind, u64, u64, u64)> {
        snapshot
            .log_events
            .iter()
            .map(|event| {
                (
                    event.kind,
                    event.iteration,
                    event.damping_trial,
                    event.mesh_revision,
                )
            })
            .collect()
    }

    fn decision_event_trace(
        snapshot: &BvpTelemetrySnapshot,
    ) -> Vec<(BvpLogEventKind, u64, u64, u64)> {
        snapshot
            .log_events
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    BvpLogEventKind::DampingTrial
                        | BvpLogEventKind::DampingRejected
                        | BvpLogEventKind::BoundLimitedStep
                        | BvpLogEventKind::MeshRevision
                        | BvpLogEventKind::Termination
                )
            })
            .map(|event| {
                (
                    event.kind,
                    event.iteration,
                    event.damping_trial,
                    event.mesh_revision,
                )
            })
            .collect()
    }

    #[test]
    fn nonlinear_refinement_trace_matches_across_frontends_and_sparse_layouts() {
        let equations = vec![
            Expr::parse_expression("z"),
            Expr::parse_expression("-2*exp(y)"),
        ];
        let guess = DMatrix::from_fn(2, 24, |row, _column| if row == 0 { 4.0 } else { 0.0 });
        let mut cases = Vec::new();

        for matrix in [MatrixRoute::Sparse, MatrixRoute::Banded] {
            for frontend in [
                BvpSymbolicAssemblyBackend::ExprLegacy,
                BvpSymbolicAssemblyBackend::AtomView,
            ] {
                let (solution, telemetry) =
                    solve_with_trace(equations.clone(), guess.clone(), frontend, matrix);
                println!(
                    "[BVP layout/frontend trace] matrix={matrix:?} frontend={frontend:?} iterations={} trials={} rejected={} refinements={} events={}",
                    telemetry.counters.iterations,
                    telemetry.counters.damping_trials,
                    telemetry.counters.damping_rejections,
                    telemetry.counters.grid_refinements,
                    telemetry.log_events.len(),
                );
                cases.push((matrix, frontend, solution, telemetry));
            }
        }

        let (_, _, reference_solution, reference) = &cases[0];
        assert!(
            reference.counters.grid_refinements > 0,
            "fixture must exercise adaptive mesh refinement"
        );
        for (matrix, frontend, solution, telemetry) in cases.iter().skip(1) {
            let max_diff = reference_solution
                .iter()
                .zip(solution.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                max_diff < 1e-7,
                "solution mismatch for {matrix:?}/{frontend:?}: {max_diff:.3e}"
            );
            assert_eq!(
                reference.counters.iterations, telemetry.counters.iterations,
                "Newton iteration mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                reference.counters.damping_trials, telemetry.counters.damping_trials,
                "damping trial mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                reference.counters.damping_rejections, telemetry.counters.damping_rejections,
                "damping rejection mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                reference.counters.grid_refinements, telemetry.counters.grid_refinements,
                "refinement count mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                decision_event_trace(reference),
                decision_event_trace(telemetry),
                "decision-event sequence mismatch for {matrix:?}/{frontend:?}"
            );
        }
        assert!(
            reference
                .log_events
                .iter()
                .any(|event| event.kind == BvpLogEventKind::MeshRevision)
        );
    }

    fn solve_symbolic_cubic_rejected_trace(
        frontend: BvpSymbolicAssemblyBackend,
        matrix: MatrixRoute,
    ) -> (DMatrix<f64>, Vec<f64>, BvpTelemetrySnapshot) {
        let root = 1.7692923542386314;
        let mut solver = NRBVP::new_with_options(
            vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("y^3-2*y-2"),
            ],
            DMatrix::zeros(2, 24),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([("y".to_string(), vec![(0usize, root), (1usize, root)])]),
            0.0,
            1.0,
            24,
            trace_options(frontend, matrix, false),
        );
        solver.dont_save_log(true);
        let result = solver.try_solve();
        result.unwrap_or_else(|error| {
            panic!("symbolic cubic fixture failed for {matrix:?}/{frontend:?}: {error:?}")
        });
        let solution = solver
            .get_result()
            .expect("converged symbolic cubic solve should publish a trajectory");
        let residual = solver
            .fun
            .call(0.0, &*solver.y)
            .to_DVectorType()
            .iter()
            .copied()
            .collect();
        (solution, residual, solver.get_statistics().telemetry)
    }

    #[test]
    fn symbolic_rejected_trajectory_matches_frontends_and_layouts() {
        let mut cases = Vec::new();
        for matrix in [MatrixRoute::Sparse, MatrixRoute::Banded] {
            for frontend in [
                BvpSymbolicAssemblyBackend::ExprLegacy,
                BvpSymbolicAssemblyBackend::AtomView,
            ] {
                let (solution, residual, telemetry) =
                    solve_symbolic_cubic_rejected_trace(frontend, matrix);
                let termination = telemetry
                    .log_events
                    .iter()
                    .find(|event| event.kind == BvpLogEventKind::Termination)
                    .expect("symbolic route must publish a typed termination event");
                assert_eq!(
                    termination.value_a, 1.0,
                    "route did not converge: {matrix:?}/{frontend:?}"
                );
                assert!(
                    telemetry.counters.damping_rejections > 0,
                    "fixture did not exercise a rejected damping trial for {matrix:?}/{frontend:?}"
                );
                cases.push((matrix, frontend, solution, residual, telemetry));
            }
        }

        let (_, _, reference_solution, reference_residual, reference) = &cases[0];
        for (matrix, frontend, solution, residual, telemetry) in cases.iter().skip(1) {
            let max_solution_diff = reference_solution
                .iter()
                .zip(solution.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            let max_residual_diff = reference_residual
                .iter()
                .zip(residual.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max);
            let residual_inf = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
            assert!(
                max_solution_diff < 1e-7,
                "symbolic rejected trajectory differs for {matrix:?}/{frontend:?}: {max_solution_diff:.3e}"
            );
            assert!(
                max_residual_diff < 1e-7,
                "final residual route mismatch for {matrix:?}/{frontend:?}: {max_residual_diff:.3e}"
            );
            assert!(
                residual_inf < 1e-5,
                "final residual exceeds the fixture tolerance for {matrix:?}/{frontend:?}: {residual_inf:.3e}"
            );
            assert_eq!(
                reference.counters.iterations, telemetry.counters.iterations,
                "Newton iteration mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                reference.counters.damping_trials, telemetry.counters.damping_trials,
                "damping trial mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                reference.counters.damping_rejections, telemetry.counters.damping_rejections,
                "damping rejection mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                decision_event_trace(reference),
                decision_event_trace(telemetry),
                "symbolic rejected-step decision trace mismatch for {matrix:?}/{frontend:?}"
            );
        }

        println!(
            "[BVP symbolic rejected trajectory] iterations={} trials={} rejections={} max_state_diff={:.3e} max_final_residual_diff={:.3e} max_final_residual_inf={:.3e}",
            reference.counters.iterations,
            reference.counters.damping_trials,
            reference.counters.damping_rejections,
            cases
                .iter()
                .skip(1)
                .flat_map(|case| reference_solution.iter().zip(case.2.iter()))
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max),
            cases
                .iter()
                .skip(1)
                .flat_map(|case| reference_residual.iter().zip(case.3.iter()))
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0, f64::max),
            cases
                .iter()
                .flat_map(|case| case.3.iter().map(|value| value.abs()))
                .fold(0.0, f64::max),
        );
    }

    #[test]
    fn linear_two_point_exprlegacy_and_atomview_have_solution_parity() {
        // y' = z, z' = 0 with y(0)=0 and y(1)=1, hence y(x)=x and z(x)=1.
        let equations = vec![Expr::parse_expression("z"), Expr::Const(0.0)];
        let guess = initial_guess(24, 1.0, |x| (x, 1.0));
        let boundary_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0f64), (1usize, 1.0f64)])]);

        let (legacy, atom) = solve_pair(equations, guess, 1.0, boundary_conditions);
        assert_matrix_parity(&legacy, &atom, 1e-8);
    }

    #[test]
    fn oscillator_exprlegacy_and_atomview_have_solution_parity() {
        // y' = z, z' = -y with y(0)=0 and y(pi/2)=1, hence y=sin(x).
        let t_end = std::f64::consts::FRAC_PI_2;
        let equations = vec![Expr::parse_expression("z"), Expr::parse_expression("-y")];
        let guess = initial_guess(24, t_end, |x| (x.sin(), x.cos()));
        let boundary_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0f64), (1usize, 1.0f64)])]);

        let (legacy, atom) = solve_pair(equations, guess, t_end, boundary_conditions);
        assert_matrix_parity(&legacy, &atom, 1e-7);
    }

    #[test]
    fn atomview_aot_matches_lambdify_on_sparse_and_banded_oscillator() {
        use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
        use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
        use std::process::Command;

        if Command::new("tcc").arg("-version").output().is_err() {
            eprintln!("[BVP Lambdify/AOT parity] skipped: tcc is not available on PATH");
            return;
        }

        fn run(
            matrix: MatrixRoute,
            config: GeneratedBackendConfig,
        ) -> (DMatrix<f64>, BvpTelemetrySnapshot, String) {
            let t_end = std::f64::consts::FRAC_PI_2;
            let n_steps = 24usize;
            let h = t_end / n_steps as f64;
            let guess = DMatrix::from_fn(2, n_steps, |row, column| {
                let x = column as f64 * h;
                if row == 0 { x.sin() } else { x.cos() }
            });
            let options = matrix
                .base_options()
                .with_generated_backend_config(config)
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
                .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed);
            let mut solver = NRBVP::new_with_options(
                vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
                guess,
                vec!["y".to_string(), "z".to_string()],
                "x".to_string(),
                HashMap::from([("y".to_string(), vec![(0usize, 0.0), (1usize, 1.0)])]),
                0.0,
                t_end,
                n_steps,
                options,
            );
            solver.dont_save_log(true);
            solver
                .try_solve()
                .expect("oscillator frontend/backend route should solve");
            let stats = solver.get_statistics();
            (
                solver.get_result().expect("solution should be published"),
                stats.telemetry,
                stats
                    .diagnostics
                    .get("generated.selected_backend")
                    .cloned()
                    .unwrap_or_default(),
            )
        }

        for matrix in [MatrixRoute::Sparse, MatrixRoute::Banded] {
            let lambdify_config = matrix
                .config(BvpSymbolicAssemblyBackend::ExprLegacy)
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
            let (lambdify_solution, lambdify_stats, _) = run(matrix, lambdify_config);

            let aot_config = match matrix {
                MatrixRoute::Sparse => {
                    GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
                        .with_matrix_backend_override(MatrixBackend::SparseCol)
                }
                MatrixRoute::Banded => {
                    GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
                }
            }
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_backend_policy_override(Some(BackendSelectionPolicy::AotOnly));
            let (aot_solution, aot_stats, selected_backend) = run(matrix, aot_config);
            assert_eq!(
                selected_backend, "AotCompiled",
                "test must prove an actual AOT callback ran for {matrix:?}"
            );
            let max_diff = lambdify_solution
                .iter()
                .zip(aot_solution.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                max_diff < 1e-7,
                "Lambdify/AOT trajectory mismatch on {matrix:?}: {max_diff:.3e}"
            );
            assert_eq!(
                lambdify_stats.counters.iterations, aot_stats.counters.iterations,
                "Lambdify/AOT Newton decisions differ on {matrix:?}"
            );
            assert_eq!(
                lambdify_stats.counters.damping_trials, aot_stats.counters.damping_trials,
                "Lambdify/AOT damping decisions differ on {matrix:?}"
            );
            println!(
                "[BVP Lambdify/AOT parity] matrix={matrix:?} max_solution_diff={max_diff:.3e} iterations={} AOT={:?}",
                aot_stats.counters.iterations, selected_backend
            );
        }
    }

    #[test]
    fn nonlinear_exprlegacy_and_atomview_preserve_newton_and_refinement_trace() {
        // Bratu-like two-point problem: y' = z, z' = -2 exp(y), y(0)=y(1)=0.
        // The deliberately high initial profile exercises accepted damping
        // trials and, when the error estimator marks the mesh, refinement.
        let equations = vec![
            Expr::parse_expression("z"),
            Expr::parse_expression("-2*exp(y)"),
        ];
        let guess = DMatrix::from_fn(2, 24, |row, _column| if row == 0 { 4.0 } else { 0.0 });

        let (legacy_result, legacy_trace) = solve_with_trace(
            equations.clone(),
            guess.clone(),
            BvpSymbolicAssemblyBackend::ExprLegacy,
            MatrixRoute::Banded,
        );
        let (atom_result, atom_trace) = solve_with_trace(
            equations,
            guess,
            BvpSymbolicAssemblyBackend::AtomView,
            MatrixRoute::Banded,
        );

        let rejected_probe = legacy_trace
            .log_events
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    BvpLogEventKind::DampingTrial | BvpLogEventKind::DampingRejected
                )
            })
            .map(|event| (event.kind, event.value_a, event.value_b))
            .collect::<Vec<_>>();
        println!("[BVP rejected damping probe] legacy_trials={rejected_probe:?}");

        assert_matrix_parity(&legacy_result, &atom_result, 1e-7);
        assert_eq!(
            legacy_trace.counters.iterations, atom_trace.counters.iterations,
            "ExprLegacy and AtomView must take the same Newton iteration count"
        );
        assert_eq!(
            legacy_trace.counters.damping_trials, atom_trace.counters.damping_trials,
            "ExprLegacy and AtomView must take the same damping-trial count"
        );
        assert_eq!(
            legacy_trace.counters.damping_rejections, atom_trace.counters.damping_rejections,
            "ExprLegacy and AtomView must take the same accepted/rejected damping decisions"
        );
        assert_eq!(
            legacy_trace.counters.grid_refinements, atom_trace.counters.grid_refinements,
            "ExprLegacy and AtomView must take the same refinement count"
        );
        assert_eq!(event_trace(&legacy_trace), event_trace(&atom_trace));
        assert!(
            legacy_trace
                .log_events
                .iter()
                .any(|event| event.kind == BvpLogEventKind::DampingTrial)
        );
        assert!(
            legacy_trace
                .log_events
                .iter()
                .any(|event| event.kind == BvpLogEventKind::Termination)
        );

        println!(
            "[BVP symbolic parity trace] iterations={} damping_trials={} damping_rejections={} refinements={} events={}",
            legacy_trace.counters.iterations,
            legacy_trace.counters.damping_trials,
            legacy_trace.counters.damping_rejections,
            legacy_trace.counters.grid_refinements,
            legacy_trace.log_events.len(),
        );

        let report = format!(
            "status: passed\n\nlegacy_iterations: {}\nlegacy_damping_trials: {}\nlegacy_damping_rejections: {}\nlegacy_refinements: {}\nlegacy_event_count: {}\natom_iterations: {}\natom_damping_trials: {}\natom_damping_rejections: {}\natom_refinements: {}\nrejected_probe: {rejected_probe:?}\n",
            legacy_trace.counters.iterations,
            legacy_trace.counters.damping_trials,
            legacy_trace.counters.damping_rejections,
            legacy_trace.counters.grid_refinements,
            legacy_trace.log_events.len(),
            atom_trace.counters.iterations,
            atom_trace.counters.damping_trials,
            atom_trace.counters.damping_rejections,
            atom_trace.counters.grid_refinements,
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "nonlinear_exprlegacy_and_atomview_preserve_newton_and_refinement_trace",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write parity trace report: {error}");
        }
    }

    fn solve_rejected_damping_fixture(
        backend: BvpSymbolicAssemblyBackend,
        matrix: MatrixRoute,
    ) -> BvpTelemetrySnapshot {
        // The symbolic route is still exercised before installing the common
        // numeric callbacks. The callback fixture is deliberately elementwise:
        // at y=0, F=-2 and J=-2, so the full Newton proposal reaches y=-1;
        // there F=-1 and J=1, making the next step have the same norm. The
        // strict acceptance rule must reject lambda=1 and retry with damping.
        let equations = vec![Expr::parse_expression("z"), Expr::Const(0.0)];
        let guess = DMatrix::from_element(2, 2, 0.0);
        let boundary_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0f64), (1usize, 0.0f64)])]);
        let mut solver = NRBVP::new_with_options(
            equations,
            guess,
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            boundary_conditions,
            0.0,
            1.0,
            2,
            trace_options(backend, matrix, false),
        );
        solver.dont_save_log(true);
        solver
            .try_eq_generate(None, None)
            .expect("symbolic frontend should prepare the damping fixture");

        let fun = convert_to_fun(Box::new(|_, vector: &dyn VectorType| {
            let values = vector.to_DVectorType();
            let mut residual = vector.zeros(values.len());
            for (index, value) in values.iter().enumerate() {
                residual = residual.assign_value(index, value.powi(3) - 2.0 * value - 2.0);
            }
            residual
        }));
        let jac = Some(convert_to_jac(Box::new(|_, vector: &dyn VectorType| {
            let values = vector.to_DVectorType();
            let diagonal = values.map(|value| 3.0 * value * value - 2.0);
            let n = diagonal.len();
            let dense = DMatrix::from_diagonal(&diagonal);
            let triplets = (0..n).map(|i| (i, i, diagonal[i])).collect();
            vector.from_vector(n, n, &dense.as_slice().to_vec(), triplets) as Box<dyn MatrixType>
        })));
        solver
            .try_replace_prepared_callbacks(fun, jac)
            .expect("explicit callback replacement should refresh the prepared fixture");
        solver
            .try_solver_prepared()
            .expect("rejected damping fixture should solve through the prepared path");
        solver.get_statistics().telemetry
    }

    #[test]
    fn rejected_damping_decisions_match_across_frontends_and_layout_configs() {
        let cases = [
            (MatrixRoute::Sparse, BvpSymbolicAssemblyBackend::ExprLegacy),
            (MatrixRoute::Sparse, BvpSymbolicAssemblyBackend::AtomView),
            (MatrixRoute::Banded, BvpSymbolicAssemblyBackend::ExprLegacy),
            (MatrixRoute::Banded, BvpSymbolicAssemblyBackend::AtomView),
        ]
        .map(|(matrix, frontend)| {
            (
                matrix,
                frontend,
                solve_rejected_damping_fixture(frontend, matrix),
            )
        });
        let legacy_trace = &cases[0].2;

        assert!(
            legacy_trace.counters.damping_rejections > 0,
            "fixture produced no rejected trial: iterations={} trials={} rejections={}",
            legacy_trace.counters.iterations,
            legacy_trace.counters.damping_trials,
            legacy_trace.counters.damping_rejections
        );
        for (matrix, frontend, trace) in cases.iter().skip(1) {
            assert_eq!(
                legacy_trace.counters.iterations, trace.counters.iterations,
                "iteration mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                legacy_trace.counters.damping_trials, trace.counters.damping_trials,
                "trial mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                legacy_trace.counters.damping_rejections, trace.counters.damping_rejections,
                "rejection mismatch for {matrix:?}/{frontend:?}"
            );
            assert_eq!(
                event_trace(legacy_trace),
                event_trace(trace),
                "event trace mismatch for {matrix:?}/{frontend:?}"
            );
        }
        assert!(
            legacy_trace
                .log_events
                .iter()
                .any(|event| { event.kind == BvpLogEventKind::DampingRejected })
        );
        println!(
            "[BVP rejected damping parity] trials={} rejections={} iterations={}",
            legacy_trace.counters.damping_trials,
            legacy_trace.counters.damping_rejections,
            legacy_trace.counters.iterations,
        );

        let report = format!(
            "status: passed\n\nreference_trials: {}\nreference_rejections: {}\nreference_iterations: {}\nsparse_atom_trials: {}\nsparse_atom_rejections: {}\nsparse_atom_iterations: {}\nbanded_expr_trials: {}\nbanded_expr_rejections: {}\nbanded_expr_iterations: {}\nbanded_atom_trials: {}\nbanded_atom_rejections: {}\nbanded_atom_iterations: {}\n",
            legacy_trace.counters.damping_trials,
            legacy_trace.counters.damping_rejections,
            legacy_trace.counters.iterations,
            cases[1].2.counters.damping_trials,
            cases[1].2.counters.damping_rejections,
            cases[1].2.counters.iterations,
            cases[2].2.counters.damping_trials,
            cases[2].2.counters.damping_rejections,
            cases[2].2.counters.iterations,
            cases[3].2.counters.damping_trials,
            cases[3].2.counters.damping_rejections,
            cases[3].2.counters.iterations,
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "rejected_damping_decisions_match_across_frontends_and_layout_configs",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write rejected-damping report: {error}");
        }
    }
}
