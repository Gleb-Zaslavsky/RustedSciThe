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
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::numerical::BVP_Damp::telemetry::{
        BvpLogEventKind, BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode, BvpTelemetrySnapshot,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

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

    fn trace_options(backend: BvpSymbolicAssemblyBackend, adaptive: bool) -> DampedSolverOptions {
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
        options(backend)
            .with_strategy_params(Some(strategy))
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
            trace_options(backend, true),
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
        );
        let (atom_result, atom_trace) =
            solve_with_trace(equations, guess, BvpSymbolicAssemblyBackend::AtomView);

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

    fn solve_rejected_damping_fixture(backend: BvpSymbolicAssemblyBackend) -> BvpTelemetrySnapshot {
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
            trace_options(backend, false),
        );
        solver.dont_save_log(true);
        solver
            .try_eq_generate(None, None)
            .expect("symbolic frontend should prepare the damping fixture");

        let fun = convert_to_fun(Box::new(|_, vector: &dyn VectorType| {
            let values = vector
                .as_any()
                .downcast_ref::<DVector<f64>>()
                .expect("damping fixture uses a dense vector");
            Box::new(values.map(|value| value.powi(3) - 2.0 * value - 2.0)) as Box<dyn VectorType>
        }));
        let jac = Some(convert_to_jac(Box::new(|_, vector: &dyn VectorType| {
            let values = vector
                .as_any()
                .downcast_ref::<DVector<f64>>()
                .expect("damping fixture uses a dense vector");
            let diagonal = values.map(|value| 3.0 * value * value - 2.0);
            Box::new(DMatrix::from_diagonal(&diagonal)) as Box<dyn MatrixType>
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
    fn rejected_damping_fixture_is_reproducible_across_symbolic_frontends() {
        let legacy_trace = solve_rejected_damping_fixture(BvpSymbolicAssemblyBackend::ExprLegacy);
        let atom_trace = solve_rejected_damping_fixture(BvpSymbolicAssemblyBackend::AtomView);

        assert!(
            legacy_trace.counters.damping_rejections > 0,
            "fixture produced no rejected trial: iterations={} trials={} rejections={}",
            legacy_trace.counters.iterations,
            legacy_trace.counters.damping_trials,
            legacy_trace.counters.damping_rejections
        );
        assert_eq!(
            legacy_trace.counters.damping_rejections,
            atom_trace.counters.damping_rejections
        );
        assert_eq!(event_trace(&legacy_trace), event_trace(&atom_trace));
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
            "status: passed\n\nlegacy_trials: {}\nlegacy_rejections: {}\nlegacy_iterations: {}\natom_trials: {}\natom_rejections: {}\natom_iterations: {}\n",
            legacy_trace.counters.damping_trials,
            legacy_trace.counters.damping_rejections,
            legacy_trace.counters.iterations,
            atom_trace.counters.damping_trials,
            atom_trace.counters.damping_rejections,
            atom_trace.counters.iterations,
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "rejected_damping_fixture_is_reproducible_across_symbolic_frontends",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write rejected-damping report: {error}");
        }
    }
}
