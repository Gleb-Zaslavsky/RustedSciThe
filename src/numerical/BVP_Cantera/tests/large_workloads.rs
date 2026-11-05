use super::*;

pub(super) fn combustion_six_state_problem(nodes: usize) -> CanteraExprBvpProblem {
    CanteraExprBvpProblem {
        equations: [
            "q",
            "-0.5*q + 0.2*exp(Teta)*C0",
            "J0",
            "-0.3*J0 - 0.5*C0 + 0.1*Teta",
            "J1",
            "-0.2*J1 - 0.3*C1 + 0.05*C0^2",
        ]
        .into_iter()
        .map(Expr::parse_expression)
        .collect(),
        initial_guess: DMatrix::from_fn(6, nodes, |state, _| if state == 2 { 1.0 } else { 0.0 }),
        state_names: ["Teta", "q", "C0", "J0", "C1", "J1"]
            .map(str::to_owned)
            .to_vec(),
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("Teta".into(), vec![(0, 0.0)]),
            ("q".into(), vec![(1, 0.0)]),
            ("C0".into(), vec![(0, 1.0)]),
            ("J0".into(), vec![(1, 0.0)]),
            ("C1".into(), vec![(0, 0.0)]),
            ("J1".into(), vec![(1, 0.0)]),
        ]),
        interval: (0.0, 1.0),
        steps: nodes,
    }
}

fn combustion_six_state_options() -> CanteraExprBvpOptions {
    CanteraExprBvpOptions {
        absolute_tolerance: 1e-6,
        relative_tolerances: ["Teta", "q", "C0", "J0", "C1", "J1"]
            .into_iter()
            .map(|name| (name.into(), 1e-5))
            .collect(),
        bounds: std::collections::HashMap::from([
            ("Teta".into(), (-10.0, 10.0)),
            ("q".into(), (-10.0, 10.0)),
            ("C0".into(), (-2.0, 2.0)),
            ("J0".into(), (-10.0, 10.0)),
            ("C1".into(), (-2.0, 2.0)),
            ("J1".into(), (-10.0, 10.0)),
        ]),
        max_iterations: 100,
        telemetry_mode: CanteraTelemetryMode::Counters,
        ..CanteraExprBvpOptions::default()
    }
}

fn kithe_stiff_six_state_problem(nodes: usize) -> CanteraExprBvpProblem {
    CanteraExprBvpProblem {
        equations: [
            "300.0*u1",
            "30.0*u1",
            "300000.0*u3",
            "30.0*u3",
            "300000.0*u5",
            "30.0*u5",
        ]
        .into_iter()
        .map(Expr::parse_expression)
        .collect(),
        initial_guess: DMatrix::from_fn(6, nodes, |state, _| match state {
            0 => 0.0,
            1 => 0.025,
            2 => 0.6,
            3 => 0.0,
            4 => 0.4,
            5 => 0.0,
            _ => unreachable!(),
        }),
        state_names: ["u0", "u1", "u2", "u3", "u4", "u5"]
            .map(str::to_owned)
            .to_vec(),
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("u0".into(), vec![(0, 0.0)]),
            ("u1".into(), vec![(1, 0.025)]),
            ("u2".into(), vec![(0, 0.6)]),
            ("u3".into(), vec![(1, 1.5e-5)]),
            ("u4".into(), vec![(0, 0.4)]),
            ("u5".into(), vec![(1, -1.5e-5)]),
        ]),
        interval: (0.0, 1.0),
        steps: nodes,
    }
}

fn kithe_stiff_six_state_options() -> CanteraExprBvpOptions {
    CanteraExprBvpOptions {
        absolute_tolerance: 1e-8,
        relative_tolerances: ["u0", "u1", "u2", "u3", "u4", "u5"]
            .into_iter()
            .map(|name| (name.into(), 1e-8))
            .collect(),
        bounds: std::collections::HashMap::from([
            ("u0".into(), (-10.0, 10.0)),
            ("u1".into(), (-10.0, 10.0)),
            ("u2".into(), (-1.0, 2.0)),
            ("u3".into(), (-10.0, 10.0)),
            ("u4".into(), (-1.0, 2.0)),
            ("u5".into(), (-10.0, 10.0)),
        ]),
        max_iterations: 100,
        logging_level: SolverLogLevel::Debug,
        telemetry_mode: CanteraTelemetryMode::Timings,
        ..CanteraExprBvpOptions::default()
    }
}

#[test]
#[ignore = "KiThe AtomView/grid-refinement failure fixture, run as a fixed-mesh Cantera numerical gate"]
fn kithe_grid_refinement_failure_fixture_solves_on_fixed_mesh() {
    // This exact six-state BVP fails during AtomView Jacobian preparation in
    // the Damped adaptive-grid route, while the ExprLegacy route is a known
    // BVP_sci-positive control. The Cantera gate isolates its own fixed-mesh
    // Newton core from Damped refinement and AtomView conversion.
    let nodes = 33;
    let problem = kithe_stiff_six_state_problem(nodes);
    let mut solver =
        CanteraNewtonSolver::from_expr_legacy_dense(problem, kithe_stiff_six_state_options())
            .expect("KiThe fixed-mesh ExprLegacy/Dense problem should prepare");
    let report = solver
        .solve()
        .expect("Cantera core should solve the KiThe six-state fixed-mesh fixture");

    assert_eq!(report.state.len(), 6 * nodes);
    assert!(report.state.iter().all(|value| value.is_finite()));
    for (value, &(lower, upper)) in report.state.iter().zip(&solver.config.bounds) {
        assert!(*value >= lower - 1e-10 && *value <= upper + 1e-10);
    }
    assert!(report.correction_norm.is_some_and(|norm| norm < 1.0));
    assert!(report.residual_norm.is_some_and(|norm| norm < 1e-8));
    assert!(report.telemetry.jacobian_evaluations > 0);
    assert!(report.telemetry.factorizations > 0);
    assert!(report.telemetry.linear_solves > 0);
    assert!(report.telemetry.accepted_steps > 0);
    assert!(report.telemetry.callback_time > std::time::Duration::ZERO);
    assert_eq!(
        report.telemetry.callback_time,
        report.telemetry.residual_callback_time + report.telemetry.jacobian_callback_time
    );
    println!(
        "Cantera Dense KiThe six-state combustion fixture ({nodes} nodes): iterations={}, correction={:?}, residual={:?}, telemetry={:?}",
        report.iterations, report.correction_norm, report.residual_norm, report.telemetry
    );
}

#[test]
#[ignore = "KiThe AtomView regression fixture, fixed-mesh Cantera route"]
fn kithe_atomview_six_state_fixed_mesh_sparse_solves() {
    let nodes = 33;
    let mut options = kithe_stiff_six_state_options();
    options.symbolic_frontend = CanteraSymbolicFrontend::AtomView;
    options.matrix_layout = CanteraMatrixLayout::Sparse;
    let mut solver =
        CanteraNewtonSolver::from_symbolic(kithe_stiff_six_state_problem(nodes), options)
            .expect("KiThe AtomView fixed-mesh preparation should succeed");
    let report = solver
        .solve()
        .expect("Cantera core should solve the KiThe AtomView six-state fixture");
    assert!(report.state.iter().all(|value| value.is_finite()));
    for (value, &(lower, upper)) in report.state.iter().zip(&solver.config.bounds) {
        assert!(*value >= lower - 1e-10 && *value <= upper + 1e-10);
    }
    assert!(report.correction_norm.is_some_and(|norm| norm < 1.0));
    assert!(report.residual_norm.is_some_and(|norm| norm < 1e-8));
    assert!(report.telemetry.jacobian_evaluations > 0);
    assert!(report.telemetry.linear_solves > 0);
}

#[test]
#[ignore = "large 6-state combustion workload ported from the BVP_sci cross-solver gate"]
fn large_six_state_combustion_exprlegacy_dense_core() {
    for nodes in [64, 128] {
        let problem = combustion_six_state_problem(nodes);
        let mut solver =
            CanteraNewtonSolver::from_expr_legacy_dense(problem, combustion_six_state_options())
                .expect("six-state ExprLegacy/Dense preparation should succeed");
        let report = solver
            .solve()
            .expect("standalone Cantera core should converge on the six-state combustion fixture");
        println!(
            "Cantera Dense six-state combustion ({nodes} nodes): iterations={}, correction={:?}, residual={:?}, telemetry={:?}",
            report.iterations, report.correction_norm, report.residual_norm, report.telemetry
        );
        assert!(report.state.iter().all(|value| value.is_finite()));
        for (value, &(lower, upper)) in report.state.iter().zip(&solver.config.bounds) {
            assert!(*value >= lower - 1e-10 && *value <= upper + 1e-10);
        }
        assert!(report.correction_norm.is_some_and(|norm| norm < 1.0));
        assert!(report.residual_norm.is_some_and(f64::is_finite));
        assert!(report.telemetry.jacobian_evaluations > 0);
        assert!(report.telemetry.linear_solves >= report.telemetry.jacobian_evaluations);
    }
}

#[test]
#[ignore = "large 26-state combustion validation for the experimental Dense Cantera core"]
fn large_fixed_mesh_combustion_exprlegacy_dense_core() {
    let problem =
        crate::numerical::BVP_Damp::test_large_native_regressions::cantera_fixed_grid_expr_problem(
        );
    let mut relative_tolerances = std::collections::HashMap::new();
    let mut bounds = std::collections::HashMap::new();
    for state in 0..problem.state_names.len() {
        let name = format!("u{state}");
        relative_tolerances.insert(name.clone(), 1e-5);
        let bound = if state == 0 {
            (-1.0, 3.0)
        } else if state % 2 == 1 {
            (-1e4, 1e4)
        } else {
            (-0.1, 1.1)
        };
        bounds.insert(name, bound);
    }
    let options = CanteraExprBvpOptions {
        absolute_tolerance: 1e-5,
        relative_tolerances,
        bounds,
        max_iterations: 150,
        telemetry_mode: CanteraTelemetryMode::Counters,
        ..CanteraExprBvpOptions::default()
    };
    let mut solver = CanteraNewtonSolver::from_expr_legacy_dense(problem, options)
        .expect("combustion ExprLegacy/Dense preparation should succeed");
    let report = solver
        .solve()
        .expect("standalone Cantera core should converge on the fixed-mesh combustion fixture");
    println!(
        "Cantera Dense combustion: iterations={}, correction={:?}, residual={:?}, telemetry={:?}",
        report.iterations, report.correction_norm, report.residual_norm, report.telemetry
    );
    assert!(report.state.iter().all(|value| value.is_finite()));
    for (value, &(lower, upper)) in report.state.iter().zip(&solver.config.bounds) {
        assert!(*value >= lower - 1e-10 && *value <= upper + 1e-10);
    }
    assert!(report.telemetry.jacobian_evaluations > 0);
    assert!(report.telemetry.linear_solves >= report.telemetry.jacobian_evaluations);

    assert!(report.correction_norm.is_some_and(|norm| norm < 1.0));
    assert!(report.residual_norm.is_some_and(f64::is_finite));
}

#[test]
#[ignore = "large AtomView/Sparse combustion validation for the experimental Cantera core"]
fn large_fixed_mesh_combustion_atomview_sparse_core() {
    let problem =
        crate::numerical::BVP_Damp::test_large_native_regressions::cantera_fixed_grid_expr_problem(
        );
    let relative_tolerances = problem
        .state_names
        .iter()
        .map(|name| (name.clone(), 1e-5))
        .collect();
    let bounds = problem
        .state_names
        .iter()
        .enumerate()
        .map(|(state, name)| {
            let bound = if state == 0 {
                (-1.0, 3.0)
            } else if state % 2 == 1 {
                (-1e4, 1e4)
            } else {
                (-0.1, 1.1)
            };
            (name.clone(), bound)
        })
        .collect();
    let options = CanteraExprBvpOptions {
        symbolic_frontend: CanteraSymbolicFrontend::AtomView,
        matrix_layout: CanteraMatrixLayout::Sparse,
        absolute_tolerance: 1e-5,
        relative_tolerances,
        bounds,
        max_iterations: 150,
        telemetry_mode: CanteraTelemetryMode::Counters,
        ..CanteraExprBvpOptions::default()
    };
    let mut solver = CanteraNewtonSolver::from_symbolic(problem, options)
        .expect("large combustion AtomView/Sparse preparation should succeed");
    let report = solver
        .solve()
        .expect("standalone Cantera core should converge on the AtomView combustion fixture");
    println!(
        "Cantera AtomView/Sparse combustion: iterations={}, correction={:?}, residual={:?}, telemetry={:?}",
        report.iterations, report.correction_norm, report.residual_norm, report.telemetry
    );
    assert!(report.state.iter().all(|value| value.is_finite()));
    for (value, &(lower, upper)) in report.state.iter().zip(&solver.config.bounds) {
        assert!(*value >= lower - 1e-10 && *value <= upper + 1e-10);
    }
    assert!(report.correction_norm.is_some_and(|norm| norm < 1.0));
    assert!(report.residual_norm.is_some_and(f64::is_finite));
    assert!(report.telemetry.jacobian_evaluations > 0);
    assert!(report.telemetry.linear_solves >= report.telemetry.jacobian_evaluations);
}
