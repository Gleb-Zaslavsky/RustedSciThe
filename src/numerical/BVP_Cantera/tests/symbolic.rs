use super::*;

#[derive(tabled::Tabled)]
struct AnalyticGateRow {
    fixture: &'static str,
    steps: usize,
    frontend: String,
    layout: String,
    profile_max_error: String,
    trapezoid_residual_max: String,
    iterations: usize,
    residual_calls: usize,
    jacobian_calls: usize,
}

fn write_analytic_gate_report(name: &str, rows: Vec<AnalyticGateRow>) {
    let table = tabled::Table::new(rows).to_string();
    let report = format!(
        "Cantera continuous-analytic correctness gate\n\nChecks: exact endpoint values, positive profile where required, trapezoid residual, cross-route parity, and second-order mesh convergence. These measurements are correctness diagnostics, not performance timings.\n\n{table}\n"
    );
    if let Err(error) =
        crate::Utils::test_reporting::write_test_report("BVP_Cantera", name, &report)
    {
        eprintln!("[BVP_Cantera analytic report] unable to write report: {error}");
    }
}

fn scalar_solver(initial: f64, target: f64, jacobian_value: f64) -> CanteraNewtonSolver {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    CanteraNewtonSolver::new(
        DVector::from_element(1, initial),
        move |state| {
            Ok(DVector::from_element(
                1,
                jacobian_value * (state[0] - target),
            ))
        },
        move |_| Ok(DMatrix::from_element(1, 1, jacobian_value)),
        config,
    )
    .unwrap()
}

#[test]
fn buffered_residual_reuses_solver_owned_output_storage() {
    use std::cell::Cell;

    let output_address = Cell::new(None);
    let pattern = CanteraJacobianPattern::dense(1).unwrap();
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new_with_buffered_callbacks(
        DVector::from_element(1, 0.0),
        move |state, output| {
            let address = output.as_ptr() as usize;
            if let Some(previous) = output_address.get() {
                assert_eq!(address, previous, "residual output must be reused");
            } else {
                output_address.set(Some(address));
            }
            output[0] = state[0] - 1.0;
            Ok(())
        },
        pattern,
        |_, output| {
            output[0] = 1.0;
            Ok(())
        },
        config,
    )
    .unwrap();

    let report = solver
        .solve()
        .expect("scalar buffered solve should converge");
    assert!((report.state[0] - 1.0).abs() < 1e-12);
    assert!(report.telemetry.residual_evaluations >= 2);
}

#[test]
fn frontend_preparation_error_preserves_typed_source_chain() {
    let source = BvpSciNewError::InvalidConfiguration("fixture preparation failure".into());
    let error = CanteraSolverError::SymbolicPreparation { source };
    let source = std::error::Error::source(&error).expect("typed source should be retained");
    assert!(matches!(
        source.downcast_ref::<BvpSciNewError>(),
        Some(BvpSciNewError::InvalidConfiguration(message)) if message == "fixture preparation failure"
    ));
    assert!(error.partial_report().is_none());
}

#[test]
fn selecting_aot_without_configuration_is_a_typed_configuration_failure() {
    let problem = CanteraExprBvpProblem {
        equations: vec![Expr::parse_expression("-y")],
        initial_guess: DMatrix::from_element(1, 2, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 2,
    };
    let options = CanteraExprBvpOptions {
        symbolic_execution: CanteraSymbolicExecution::Aot,
        ..CanteraExprBvpOptions::default()
    };
    assert!(matches!(
        CanteraNewtonSolver::from_symbolic(problem, options),
        Err(CanteraSolverError::InvalidConfiguration { message })
            if message.contains("explicit generated-backend configuration")
    ));
}

#[test]
fn symbolic_fixed_inputs_bind_and_rebind_without_repreparing_callbacks() {
    let make_problem = || CanteraExprBvpProblem {
        equations: vec![Expr::parse_expression("-(rate + offset)*y")],
        initial_guess: DMatrix::from_element(1, 16, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 16,
    };
    let mut solver = CanteraNewtonSolver::from_symbolic(
        make_problem(),
        CanteraExprBvpOptions {
            matrix_layout: CanteraMatrixLayout::BandedAuto,
            ..CanteraExprBvpOptions::default()
        },
    )
    .expect("parameterized symbolic callbacks should prepare");
    assert_eq!(solver.external_parameter_names(), ["offset", "rate"]);
    assert!(matches!(
        solver.solve(),
        Err(CanteraSolverError::MissingParameterValues { names })
            if names == ["offset", "rate"]
    ));

    solver
        .bind_parameters(vec!["rate".into(), "offset".into()], vec![0.4, 0.1])
        .expect("the inferred fixed-input schema should bind");
    let first = solver.solve().expect("first bound solve should converge");
    let first_endpoint = solver.full_grid_values(&first.state).unwrap()[(16, 0)];
    assert!((first_endpoint - (-0.5_f64).exp()).abs() < 2e-3);

    solver
        .set_parameter_values(vec![0.5, 0.5])
        .expect("new values should reuse prepared callbacks");
    let second = solver.solve().expect("rebound solve should converge");
    let second_endpoint = solver.full_grid_values(&second.state).unwrap()[(16, 0)];
    assert!((second_endpoint - (-1.0_f64).exp()).abs() < 2e-3);
    assert!(second_endpoint < first_endpoint);
}

#[test]
#[ignore = "compiles generated artifacts for the Cantera AOT frontend/backend parity matrix"]
fn aot_frontends_match_lambdify_across_linear_layouts() {
    let root = std::env::var_os("BVP_CANTERA_AOT_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("target/bvp-cantera-aot-tests"));
    for frontend in [
        CanteraSymbolicFrontend::ExprLegacy,
        CanteraSymbolicFrontend::AtomView,
    ] {
        for layout in [
            CanteraMatrixLayout::Dense,
            CanteraMatrixLayout::Sparse,
            CanteraMatrixLayout::BandedAuto,
        ] {
            let make_problem = || CanteraExprBvpProblem {
                equations: vec![Expr::parse_expression("-rate*y")],
                initial_guess: DMatrix::from_element(1, 4, 1.0),
                state_names: vec!["y".into()],
                independent_variable: "x".into(),
                boundary_conditions: std::collections::HashMap::from([(
                    "y".into(),
                    vec![(0, 1.0)],
                )]),
                interval: (0.0, 1.0),
                steps: 4,
            };
            let mut lambdify = CanteraNewtonSolver::from_symbolic(
                make_problem(),
                CanteraExprBvpOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    telemetry_mode: CanteraTelemetryMode::Counters,
                    ..CanteraExprBvpOptions::default()
                },
            )
            .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} Lambdify prepare: {error}"));
            lambdify
                .bind_parameters(vec!["rate".into()], vec![0.5])
                .expect("Lambdify fixed input should bind");
            let lambdify_report = lambdify
                .solve()
                .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} Lambdify solve: {error}"));

            let aot_config =
                crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig::
                    build_if_missing_release(root.join(format!("{frontend:?}-{layout:?}")))
                    .with_c_tcc();
            let mut aot = CanteraNewtonSolver::from_symbolic(
                make_problem(),
                CanteraExprBvpOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    telemetry_mode: CanteraTelemetryMode::Counters,
                    ..CanteraExprBvpOptions::default()
                }
                .with_aot_config(aot_config),
            )
            .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} AOT prepare: {error}"));
            aot.bind_parameters(vec!["rate".into()], vec![0.5])
                .expect("AOT fixed input should bind");
            let aot_report = aot
                .solve()
                .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} AOT solve: {error}"));
            assert_eq!(aot_report.linear_backend, lambdify_report.linear_backend);
            aot.set_parameter_values(vec![1.0])
                .expect("AOT artifact should support rebinding");
            let rebound = aot.solve().unwrap_or_else(|error| {
                panic!("{frontend:?}/{layout:?} rebound AOT solve: {error}")
            });
            let endpoint = aot.full_grid_values(&rebound.state).unwrap()[(4, 0)];
            assert!((endpoint - (-1.0_f64).exp()).abs() < 2e-3);
            for (actual, expected) in aot_report.state.iter().zip(lambdify_report.state.iter()) {
                assert!((actual - expected).abs() < 1e-12, "{frontend:?}/{layout:?}");
            }
            assert_eq!(
                aot_report.telemetry.jacobian_evaluations,
                lambdify_report.telemetry.jacobian_evaluations
            );
            assert_eq!(
                aot_report.telemetry.factorizations,
                lambdify_report.telemetry.factorizations
            );
            assert_eq!(
                aot_report.telemetry.linear_solves,
                lambdify_report.telemetry.linear_solves
            );
        }
    }
}

#[test]
fn symbolic_trapezoid_plan_matches_frontends_and_linear_layouts() {
    let expected_backend = [
        CanteraLinearBackend::Dense,
        CanteraLinearBackend::Sparse,
        CanteraLinearBackend::Banded { lower: 1, upper: 0 },
    ];
    let mut reference: Option<f64> = None;
    for frontend in [
        CanteraSymbolicFrontend::ExprLegacy,
        CanteraSymbolicFrontend::AtomView,
    ] {
        for (layout, backend) in [
            CanteraMatrixLayout::Dense,
            CanteraMatrixLayout::Sparse,
            CanteraMatrixLayout::BandedAuto,
        ]
        .into_iter()
        .zip(expected_backend)
        {
            let problem = CanteraExprBvpProblem {
                equations: vec![Expr::parse_expression("-y")],
                initial_guess: DMatrix::from_element(1, 4, 1.0),
                state_names: vec!["y".into()],
                independent_variable: "x".into(),
                boundary_conditions: std::collections::HashMap::from([(
                    "y".into(),
                    vec![(0, 1.0)],
                )]),
                interval: (0.0, 1.0),
                steps: 4,
            };
            let options = CanteraExprBvpOptions {
                symbolic_frontend: frontend,
                matrix_layout: layout,
                telemetry_mode: CanteraTelemetryMode::Counters,
                ..CanteraExprBvpOptions::default()
            };
            let mut solver = CanteraNewtonSolver::from_symbolic(problem, options)
                .expect("direct symbolic preparation should succeed");
            let report = solver
                .solve()
                .expect("linear trapezoid problem should converge");
            let full = solver
                .full_grid_values(&report.state)
                .expect("the reduced solution should expand to the complete mesh");
            let expected_ratio: f64 = (1.0 - 0.5 / 4.0) / (1.0 + 0.5 / 4.0);
            let expected_final = expected_ratio.powi(4);
            assert!((full[(4, 0)] - expected_final).abs() < 1e-12);
            assert_eq!(report.linear_backend, backend);
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.factorizations, 1);
            assert_eq!(report.telemetry.linear_solves, 2);
            if frontend == CanteraSymbolicFrontend::ExprLegacy {
                if let Some(reference) = reference {
                    assert!((full[(4, 0)] - reference).abs() < 1e-13);
                } else {
                    reference = Some(full[(4, 0)]);
                }
            }
        }
    }
}

fn nonlinear_exact_problem(steps: usize) -> CanteraExprBvpProblem {
    let h = 1.0 / steps as f64;
    let initial_guess = DMatrix::from_fn(2, steps, |state, free_node| {
        let node = if state == 0 {
            free_node + 1 // y(0) is prescribed.
        } else {
            free_node // z(1) is prescribed.
        };
        let x = node as f64 * h;
        let exact = if state == 0 {
            1.0 / (1.0 + x)
        } else {
            -1.0 / (1.0 + x).powi(2)
        };
        exact * if state == 0 { 1.08 } else { 0.92 }
    });

    CanteraExprBvpProblem {
        equations: ["z", "2*y^3"]
            .into_iter()
            .map(Expr::parse_expression)
            .collect(),
        initial_guess,
        state_names: vec!["y".into(), "z".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("y".into(), vec![(0, 1.0)]),
            ("z".into(), vec![(1, -0.25)]),
        ]),
        interval: (0.0, 1.0),
        steps,
    }
}

fn nonlinear_exact_errors(values: &DMatrix<f64>, steps: usize) -> (f64, f64) {
    let h = 1.0 / steps as f64;
    let mut profile_error: f64 = 0.0;
    let mut residual_error: f64 = 0.0;
    for node in 0..=steps {
        let x = node as f64 * h;
        profile_error = profile_error
            .max((values[(node, 0)] - 1.0 / (1.0 + x)).abs())
            .max((values[(node, 1)] + 1.0 / (1.0 + x).powi(2)).abs());
    }
    for interval in 0..steps {
        let y0 = values[(interval, 0)];
        let y1 = values[(interval + 1, 0)];
        let z0 = values[(interval, 1)];
        let z1 = values[(interval + 1, 1)];
        residual_error = residual_error
            .max((y1 - y0 - 0.5 * h * (z0 + z1)).abs())
            .max((z1 - z0 - h * (y0.powi(3) + y1.powi(3))).abs());
    }
    (profile_error, residual_error)
}

#[test]
fn nonlinear_analytic_bvp_has_second_order_profile_convergence_across_routes() {
    let mut errors_by_steps = Vec::new();
    let mut report_rows = Vec::new();
    for steps in [16, 32, 64] {
        let mut reference_error: Option<f64> = None;
        for frontend in [
            CanteraSymbolicFrontend::ExprLegacy,
            CanteraSymbolicFrontend::AtomView,
        ] {
            for layout in [
                CanteraMatrixLayout::Dense,
                CanteraMatrixLayout::Sparse,
                CanteraMatrixLayout::BandedAuto,
            ] {
                let options = CanteraExprBvpOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    absolute_tolerance: 1e-10,
                    default_relative_tolerance: 1e-9,
                    telemetry_mode: CanteraTelemetryMode::Counters,
                    ..CanteraExprBvpOptions::default()
                };
                let mut solver =
                    CanteraNewtonSolver::from_symbolic(nonlinear_exact_problem(steps), options)
                        .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} prepare: {error}"));
                let report = solver
                    .solve()
                    .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} solve: {error}"));
                assert!(report.correction_norm.unwrap_or(f64::INFINITY) <= 1.0);
                let values = solver
                    .full_grid_values(&report.state)
                    .expect("analytic profile should expand to the full mesh");
                assert!((values[(0, 0)] - 1.0).abs() <= 1e-14);
                assert!((values[(steps, 1)] + 0.25).abs() <= 1e-14);
                let (profile_error, residual_error) = nonlinear_exact_errors(&values, steps);
                assert!(residual_error <= 1e-8, "residual={residual_error:e}");
                report_rows.push(AnalyticGateRow {
                    fixture: "y=1/(1+x)",
                    steps,
                    frontend: format!("{frontend:?}"),
                    layout: format!("{layout:?}"),
                    profile_max_error: format!("{profile_error:.6e}"),
                    trapezoid_residual_max: format!("{residual_error:.6e}"),
                    iterations: report.iterations,
                    residual_calls: report.telemetry.residual_evaluations,
                    jacobian_calls: report.telemetry.jacobian_evaluations,
                });
                if let Some(reference) = reference_error {
                    assert!(
                        (profile_error - reference).abs() <= 1e-8,
                        "{frontend:?}/{layout:?}, profile error {profile_error:e} != {reference:e}"
                    );
                } else {
                    reference_error = Some(profile_error);
                }
                assert!(report.telemetry.residual_evaluations > 0);
                assert!(report.telemetry.jacobian_evaluations > 0);
            }
        }
        errors_by_steps.push((steps, reference_error.expect("routes were exercised")));
    }

    for pair in errors_by_steps.windows(2) {
        let ratio = pair[0].1 / pair[1].1;
        assert!(
            ratio > 3.0 && ratio < 5.0,
            "expected second-order profile convergence when refining {} to {} steps; error ratio={ratio}",
            pair[0].0,
            pair[1].0
        );
    }
    write_analytic_gate_report("nonlinear_analytic_grid_convergence", report_rows);
}

fn gaussian_two_point_problem(steps: usize) -> CanteraExprBvpProblem {
    let h = 2.0 / steps as f64;
    let initial_guess = DMatrix::from_fn(2, steps, |state, free_node| {
        let node = if state == 0 {
            free_node + 1 // y(-1) is prescribed.
        } else {
            free_node // z(1) is prescribed.
        };
        let x = -1.0 + node as f64 * h;
        let y = (-x * x / 4.0).exp();
        if state == 0 {
            1.01 * y
        } else {
            0.99 * (-0.5 * x * y)
        }
    });
    let endpoint_y = (-0.25_f64).exp();
    CanteraExprBvpProblem {
        equations: ["z", "-0.5*(1+2*ln(y))*y"]
            .into_iter()
            .map(Expr::parse_expression)
            .collect(),
        initial_guess,
        state_names: vec!["y".into(), "z".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("y".into(), vec![(0, endpoint_y)]),
            ("z".into(), vec![(1, -0.5 * endpoint_y)]),
        ]),
        interval: (-1.0, 1.0),
        steps,
    }
}

fn gaussian_two_point_errors(values: &DMatrix<f64>, steps: usize) -> (f64, f64) {
    let h = 2.0 / steps as f64;
    let mut profile_error: f64 = 0.0;
    let mut residual_error: f64 = 0.0;
    for node in 0..=steps {
        let x = -1.0 + node as f64 * h;
        let exact_y = (-x * x / 4.0).exp();
        profile_error = profile_error
            .max((values[(node, 0)] - exact_y).abs())
            .max((values[(node, 1)] + 0.5 * x * exact_y).abs());
    }
    for interval in 0..steps {
        let y0 = values[(interval, 0)];
        let y1 = values[(interval + 1, 0)];
        let z0 = values[(interval, 1)];
        let z1 = values[(interval + 1, 1)];
        let rhs = |y: f64| -0.5 * (1.0 + 2.0 * y.ln()) * y;
        residual_error = residual_error
            .max((y1 - y0 - 0.5 * h * (z0 + z1)).abs())
            .max((z1 - z0 - 0.5 * h * (rhs(y0) + rhs(y1))).abs());
    }
    (profile_error, residual_error)
}

#[test]
fn examples_utils_two_point_exact_bvp_converges_across_symbolic_routes() {
    let mut errors_by_steps = Vec::new();
    let mut report_rows = Vec::new();
    for steps in [16, 32, 64] {
        let mut reference_error: Option<f64> = None;
        for frontend in [
            CanteraSymbolicFrontend::ExprLegacy,
            CanteraSymbolicFrontend::AtomView,
        ] {
            for layout in [
                CanteraMatrixLayout::Dense,
                CanteraMatrixLayout::Sparse,
                CanteraMatrixLayout::BandedAuto,
            ] {
                let options = CanteraExprBvpOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    absolute_tolerance: 1e-10,
                    default_relative_tolerance: 1e-9,
                    bounds: std::collections::HashMap::from([("y".into(), (1e-6, 2.0))]),
                    telemetry_mode: CanteraTelemetryMode::Counters,
                    ..CanteraExprBvpOptions::default()
                };
                let mut solver =
                    CanteraNewtonSolver::from_symbolic(gaussian_two_point_problem(steps), options)
                        .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} prepare: {error}"));
                let report = solver
                    .solve()
                    .unwrap_or_else(|error| panic!("{frontend:?}/{layout:?} solve: {error}"));
                assert!(report.correction_norm.unwrap_or(f64::INFINITY) <= 1.0);
                let values = solver
                    .full_grid_values(&report.state)
                    .expect("exact profile should expand to the full mesh");
                let endpoint_y = (-0.25_f64).exp();
                assert!((values[(0, 0)] - endpoint_y).abs() <= 1e-14);
                assert!((values[(steps, 1)] + 0.5 * endpoint_y).abs() <= 1e-14);
                assert!(values.column(0).iter().all(|value| *value > 0.0));
                let (profile_error, residual_error) = gaussian_two_point_errors(&values, steps);
                assert!(residual_error <= 1e-8, "residual={residual_error:e}");
                report_rows.push(AnalyticGateRow {
                    fixture: "exp(-x^2/4), Examples_and_utils",
                    steps,
                    frontend: format!("{frontend:?}"),
                    layout: format!("{layout:?}"),
                    profile_max_error: format!("{profile_error:.6e}"),
                    trapezoid_residual_max: format!("{residual_error:.6e}"),
                    iterations: report.iterations,
                    residual_calls: report.telemetry.residual_evaluations,
                    jacobian_calls: report.telemetry.jacobian_evaluations,
                });
                if let Some(reference) = reference_error {
                    assert!(
                        (profile_error - reference).abs() <= 1e-8,
                        "{frontend:?}/{layout:?}, profile error {profile_error:e} != {reference:e}"
                    );
                } else {
                    reference_error = Some(profile_error);
                }
                assert!(report.telemetry.residual_evaluations > 0);
                assert!(report.telemetry.jacobian_evaluations > 0);
            }
        }
        errors_by_steps.push((steps, reference_error.expect("routes were exercised")));
    }

    for pair in errors_by_steps.windows(2) {
        let ratio = pair[0].1 / pair[1].1;
        assert!(
            ratio > 3.0 && ratio < 5.0,
            "expected second-order profile convergence when refining {} to {} steps; error ratio={ratio}",
            pair[0].0,
            pair[1].0
        );
    }
    write_analytic_gate_report("examples_utils_two_point_grid_convergence", report_rows);
}

#[derive(Clone, Copy, Debug)]
enum CanteraAotToolchain {
    CcGcc,
    CcTcc,
    Rust,
    Zig,
}

impl CanteraAotToolchain {
    fn label(self) -> &'static str {
        match self {
            Self::CcGcc => "C/GCC",
            Self::CcTcc => "C/TCC",
            Self::Rust => "Rust",
            Self::Zig => "Zig",
        }
    }

    fn config(self) -> crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig {
        use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
        match self {
            Self::CcGcc => SymbolicIvpGeneratedBackendConfig::new().with_c_gcc(),
            Self::CcTcc => SymbolicIvpGeneratedBackendConfig::new().with_c_tcc(),
            Self::Rust => SymbolicIvpGeneratedBackendConfig::new().with_rust(),
            Self::Zig => SymbolicIvpGeneratedBackendConfig::new().with_zig(),
        }
    }

    fn compiler_version(self) -> String {
        use std::process::Command;
        let (program, args): (&str, &[&str]) = match self {
            Self::CcGcc => ("gcc", &["--version"]),
            Self::CcTcc => ("tcc", &["-v"]),
            Self::Rust => ("rustc", &["--version"]),
            Self::Zig => ("zig", &["version"]),
        };
        let output = Command::new(program)
            .args(args)
            .output()
            .unwrap_or_else(|error| panic!("failed to query {} version: {error}", self.label()));
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        stdout
            .lines()
            .chain(stderr.lines())
            .find(|line| !line.trim().is_empty())
            .unwrap_or("version unavailable")
            .trim()
            .to_owned()
    }
}

#[derive(tabled::Tabled)]
struct AotCompilerGateRow {
    toolchain: String,
    compiler_version: String,
    profile: String,
    frontend: String,
    layout: String,
    initial_state_max_diff: String,
    rebound_state_max_diff: String,
    iterations: usize,
    jacobian_calls: usize,
}

#[test]
#[ignore = "builds Cantera AOT artifacts for all supported compilers and optimization presets"]
fn aot_compiler_and_profile_matrix_matches_lambdify() {
    use crate::symbolic::codegen::codegen_aot_driver::AotBuildPreset;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::symbolic_ivp_generated::{
        SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    };
    use std::path::PathBuf;

    let output_root = std::env::var_os("BVP_CANTERA_AOT_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/bvp-cantera-aot-tests"));
    let presets = [
        AotBuildPreset::Production,
        AotBuildPreset::FastBuild,
        AotBuildPreset::DevFastest,
    ];
    let toolchains = [
        CanteraAotToolchain::CcGcc,
        CanteraAotToolchain::CcTcc,
        CanteraAotToolchain::Rust,
        CanteraAotToolchain::Zig,
    ];
    let mut rows = Vec::new();

    for frontend in [
        CanteraSymbolicFrontend::ExprLegacy,
        CanteraSymbolicFrontend::AtomView,
    ] {
        for toolchain in toolchains {
            for preset in presets {
                let preset_label = format!("{preset:?}");
                let route_dir = output_root
                    .join(format!("{frontend:?}"))
                    .join(toolchain.label().replace('/', "-"))
                    .join(&preset_label);
                let config = toolchain
                    .config()
                    .with_output_parent_dir(Some(route_dir))
                    .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
                        profile: AotBuildProfile::Release,
                    })
                    .with_aot_build_preset(preset);

                let make_problem = || CanteraExprBvpProblem {
                    equations: vec![Expr::parse_expression("-rate*y")],
                    initial_guess: DMatrix::from_element(1, 16, 0.8),
                    state_names: vec!["y".into()],
                    independent_variable: "x".into(),
                    boundary_conditions: std::collections::HashMap::from([(
                        "y".into(),
                        vec![(0, 1.0)],
                    )]),
                    interval: (0.0, 1.0),
                    steps: 16,
                };
                let mut lambdify = CanteraNewtonSolver::from_symbolic(
                    make_problem(),
                    CanteraExprBvpOptions {
                        symbolic_frontend: frontend,
                        matrix_layout: CanteraMatrixLayout::Sparse,
                        telemetry_mode: CanteraTelemetryMode::Counters,
                        ..CanteraExprBvpOptions::default()
                    },
                )
                .unwrap_or_else(|error| panic!("{frontend:?} Lambdify prepare: {error}"));
                lambdify
                    .bind_parameters(vec!["rate".into()], vec![0.7])
                    .expect("Lambdify input should bind");
                let lambda_first = lambdify
                    .solve()
                    .unwrap_or_else(|error| panic!("{frontend:?} Lambdify solve: {error}"));

                let aot_config: SymbolicIvpGeneratedBackendConfig = config;
                let mut aot = CanteraNewtonSolver::from_symbolic(
                    make_problem(),
                    CanteraExprBvpOptions {
                        symbolic_frontend: frontend,
                        symbolic_execution: CanteraSymbolicExecution::Aot,
                        aot_config: Some(aot_config),
                        matrix_layout: CanteraMatrixLayout::Sparse,
                        telemetry_mode: CanteraTelemetryMode::Counters,
                        ..CanteraExprBvpOptions::default()
                    },
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "{} / {preset:?} / {frontend:?} AOT prepare: {error}",
                        toolchain.label()
                    )
                });
                aot.bind_parameters(vec!["rate".into()], vec![0.7])
                    .expect("AOT input should bind");
                let aot_first = aot.solve().unwrap_or_else(|error| {
                    panic!(
                        "{} / {preset:?} / {frontend:?} AOT solve: {error}",
                        toolchain.label()
                    )
                });
                let initial_state_max_diff = aot_first
                    .state
                    .iter()
                    .zip(lambda_first.state.iter())
                    .map(|(aot, lambda)| (aot - lambda).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    initial_state_max_diff <= 1e-10,
                    "{toolchain:?}/{preset:?}/{frontend:?}: {initial_state_max_diff:e}"
                );

                lambdify
                    .set_parameter_values(vec![1.0])
                    .expect("Lambdify rebind should succeed");
                aot.set_parameter_values(vec![1.0])
                    .expect("AOT rebind should reuse the built artifact");
                let lambda_rebound = lambdify
                    .solve()
                    .unwrap_or_else(|error| panic!("Lambdify rebound solve: {error}"));
                let aot_rebound = aot
                    .solve()
                    .unwrap_or_else(|error| panic!("AOT rebound solve: {error}"));
                let rebound_state_max_diff = aot_rebound
                    .state
                    .iter()
                    .zip(lambda_rebound.state.iter())
                    .map(|(aot, lambda)| (aot - lambda).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    rebound_state_max_diff <= 1e-10,
                    "{toolchain:?}/{preset:?}/{frontend:?}: {rebound_state_max_diff:e}"
                );
                assert_eq!(
                    aot_rebound.telemetry.jacobian_evaluations,
                    lambda_rebound.telemetry.jacobian_evaluations
                );

                rows.push(AotCompilerGateRow {
                    toolchain: toolchain.label().into(),
                    compiler_version: toolchain.compiler_version(),
                    profile: preset_label,
                    frontend: format!("{frontend:?}"),
                    layout: "Sparse".into(),
                    initial_state_max_diff: format!("{initial_state_max_diff:.6e}"),
                    rebound_state_max_diff: format!("{rebound_state_max_diff:.6e}"),
                    iterations: aot_rebound.iterations,
                    jacobian_calls: aot_rebound.telemetry.jacobian_evaluations,
                });
            }
        }
    }

    let table = tabled::Table::new(rows).to_string();
    let report = format!(
        "Cantera AOT compiler/profile correctness matrix\n\nCompiler routes: GCC, TCC, Rust and Zig. Presets: Production, FastBuild and DevFastest. Layout: Sparse. Each row checks ExprLegacy/AtomView parity with Lambdify before and after fixed-input rebinding. Build and cache lifecycle are forced into a unique route directory.\n\n{table}\n"
    );
    if let Err(error) = crate::Utils::test_reporting::write_test_report(
        "BVP_Cantera_AOT",
        "aot_compiler_and_profile_matrix",
        &report,
    ) {
        eprintln!("[BVP_Cantera AOT report] unable to write report: {error}");
    }
}
