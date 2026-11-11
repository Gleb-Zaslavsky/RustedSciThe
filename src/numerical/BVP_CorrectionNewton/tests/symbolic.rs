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
        "CorrectionNewton continuous-analytic correctness gate\n\nChecks: exact endpoint values, positive profile where required, trapezoid residual, cross-route parity, and second-order mesh convergence. These measurements are correctness diagnostics, not performance timings.\n\n{table}\n"
    );
    if let Err(error) =
        crate::Utils::test_reporting::write_test_report("BVP_CorrectionNewton", name, &report)
    {
        eprintln!("[BVP_CorrectionNewton analytic report] unable to write report: {error}");
    }
}

fn scalar_solver(initial: f64, target: f64, jacobian_value: f64) -> CorrectionNewtonSolver {
    let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    CorrectionNewtonSolver::new(
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
    let pattern = CorrectionNewtonJacobianPattern::dense(1).unwrap();
    let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let mut solver = CorrectionNewtonSolver::new_with_buffered_callbacks(
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
    let error = CorrectionNewtonSolverError::SymbolicPreparation { source };
    let source = std::error::Error::source(&error).expect("typed source should be retained");
    assert!(matches!(
        source.downcast_ref::<BvpSciNewError>(),
        Some(BvpSciNewError::InvalidConfiguration(message)) if message == "fixture preparation failure"
    ));
    assert!(error.partial_report().is_none());
}

#[test]
fn selecting_aot_without_configuration_is_a_typed_configuration_failure() {
    let problem = CorrectionNewtonProblem {
        equations: vec![Expr::parse_expression("-y")],
        initial_guess: DMatrix::from_element(1, 2, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 2,
    };
    let options = CorrectionNewtonOptions {
        symbolic_execution: CorrectionNewtonSymbolicExecution::Aot,
        ..CorrectionNewtonOptions::default()
    };
    assert!(matches!(
        CorrectionNewtonSolver::from_symbolic(problem, options),
        Err(CorrectionNewtonSolverError::InvalidConfiguration {
            source: super::super::CorrectionNewtonConfigurationError::MissingAotConfiguration,
        })
    ));
}

#[test]
fn explicit_nonuniform_mesh_validation_returns_a_typed_mesh_error() {
    let problem = CorrectionNewtonProblem {
        equations: vec![Expr::parse_expression("-y")],
        initial_guess: DMatrix::from_element(1, 2, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 2,
    };
    for mesh in [
        vec![0.0, 1.0],
        vec![0.0, 0.8, 0.7],
        vec![0.0, f64::NAN, 1.0],
        vec![0.0, 0.5, 1.01],
    ] {
        assert!(matches!(
            CorrectionNewtonSolver::from_symbolic_on_mesh(
                problem.clone(),
                mesh,
                CorrectionNewtonOptions::default(),
            ),
            Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::InvalidExplicitMesh,
            })
        ));
    }
}

#[test]
fn symbolic_fixed_inputs_bind_and_rebind_without_repreparing_callbacks() {
    let make_problem = || CorrectionNewtonProblem {
        equations: vec![Expr::parse_expression("-(rate + offset)*y")],
        initial_guess: DMatrix::from_element(1, 16, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 16,
    };
    let mut solver = CorrectionNewtonSolver::from_symbolic(
        make_problem(),
        CorrectionNewtonOptions {
            matrix_layout: CorrectionNewtonMatrixLayout::BandedAuto,
            ..CorrectionNewtonOptions::default()
        },
    )
    .expect("parameterized symbolic callbacks should prepare");
    assert_eq!(solver.external_parameter_names(), ["offset", "rate"]);
    assert!(matches!(
        solver.solve(),
        Err(CorrectionNewtonSolverError::MissingParameterValues { names })
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
fn symbolic_boundary_values_rebind_without_repreparing_callbacks() {
    let problem = CorrectionNewtonProblem {
        equations: vec![Expr::parse_expression("-y")],
        initial_guess: DMatrix::from_element(1, 16, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 16,
    };
    let options = CorrectionNewtonOptions {
        bounds: std::collections::HashMap::from([("y".into(), (0.0, 2.0))]),
        ..CorrectionNewtonOptions::default()
    };
    let mut solver = CorrectionNewtonSolver::from_symbolic(problem, options)
        .expect("symbolic callbacks should prepare");
    let first = solver
        .solve()
        .expect("initial boundary solve should converge");
    let first_profile = solver.full_grid_values(&first.state).unwrap();
    assert!((first_profile[(0, 0)] - 1.0).abs() < 1e-14);

    solver
        .set_boundary_values(vec![2.0])
        .expect("numeric endpoint values should rebind");
    let second = solver
        .solve()
        .expect("rebound boundary solve should converge");
    let second_profile = solver.full_grid_values(&second.state).unwrap();
    assert!((second_profile[(0, 0)] - 2.0).abs() < 1e-14);
    assert!((second_profile[(16, 0)] - 2.0 * (-1.0_f64).exp()).abs() < 2e-3);

    let error = solver
        .set_boundary_values(vec![3.0])
        .expect_err("out-of-bounds boundary values must be rejected");
    assert!(matches!(
        error,
        CorrectionNewtonSolverError::InvalidConfiguration {
            source: super::super::CorrectionNewtonConfigurationError::BoundaryValueOutsideBounds { .. }
        }
    ));
    let retained = solver
        .solve()
        .expect("rejected rebinding must preserve the previous boundary value");
    let retained_profile = solver.full_grid_values(&retained.state).unwrap();
    assert!((retained_profile[(0, 0)] - 2.0).abs() < 1e-14);
}

#[test]
fn symbolic_fixed_boundary_values_must_respect_component_bounds() {
    let problem = CorrectionNewtonProblem {
        equations: vec![Expr::parse_expression("0")],
        initial_guess: DMatrix::from_row_slice(1, 2, &[0.5, 0.5]),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 2.0)])]),
        interval: (0.0, 1.0),
        steps: 2,
    };
    let options = CorrectionNewtonOptions {
        bounds: std::collections::HashMap::from([("y".into(), (0.0, 1.0))]),
        ..CorrectionNewtonOptions::default()
    };

    let error = match CorrectionNewtonSolver::from_symbolic(problem.clone(), options) {
        Ok(_) => panic!("an out-of-bounds prescribed endpoint must fail preparation"),
        Err(error) => error,
    };
    match error {
        CorrectionNewtonSolverError::InvalidConfiguration {
            source:
                super::super::CorrectionNewtonConfigurationError::FixedBoundaryOutsideBounds {
                    state,
                    value,
                    lower,
                    upper,
                },
        } => {
            assert_eq!(state, "y");
            assert_eq!(value, 2.0);
            assert_eq!(lower, 0.0);
            assert_eq!(upper, 1.0);
        }
        other => panic!("expected typed invalid-configuration error, got {other:?}"),
    }

    for bounds in [(f64::NEG_INFINITY, f64::INFINITY), (0.0, 2.0)] {
        CorrectionNewtonSolver::from_symbolic(
            problem.clone(),
            CorrectionNewtonOptions {
                bounds: std::collections::HashMap::from([("y".into(), bounds)]),
                ..CorrectionNewtonOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("valid inclusive bounds {bounds:?}: {error}"));
    }
}

#[test]
fn bratu_lower_and_upper_branches_match_independent_analytic_profiles() {
    let lambda = 1.0_f64;
    let turning_point = 1.2_f64;
    let branch_parameter = |target_lambda: f64, mut low: f64, mut high: f64| {
        let low_is_positive = 8.0 * low.powi(2) / low.cosh().powi(2) - target_lambda > 0.0;
        for _ in 0..80 {
            let middle = 0.5 * (low + high);
            let value = 8.0 * middle.powi(2) / middle.cosh().powi(2) - target_lambda;
            if (value > 0.0) == low_is_positive {
                low = middle;
            } else {
                high = middle;
            }
        }
        0.5 * (low + high)
    };
    let branches = [
        ("lower", branch_parameter(lambda, 0.0, turning_point)),
        ("upper", branch_parameter(lambda, turning_point, 5.0)),
    ];
    let steps = 64;
    let half_domain = 0.5;
    let mut center_values = Vec::new();

    for (branch, parameter) in branches {
        let initial_guess = DMatrix::from_fn(2, steps, |state, column| {
            let node = if state == 0 { column + 1 } else { column };
            let x = half_domain * node as f64 / steps as f64;
            let shifted = parameter * (1.0 - 2.0 * x);
            if state == 0 {
                2.0 * (parameter.cosh() / shifted.cosh()).ln()
            } else {
                4.0 * parameter * shifted.tanh()
            }
        });
        let problem = CorrectionNewtonProblem {
            equations: vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("-lambda*exp(y)"),
            ],
            initial_guess,
            state_names: vec!["y".into(), "z".into()],
            independent_variable: "x".into(),
            boundary_conditions: std::collections::HashMap::from([
                ("y".into(), vec![(0, 0.0)]),
                ("z".into(), vec![(1, 0.0)]),
            ]),
            interval: (0.0, half_domain),
            steps,
        };
        let mut solver = CorrectionNewtonSolver::from_symbolic(
            problem,
            CorrectionNewtonOptions {
                matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
                relative_tolerances: std::collections::HashMap::from([
                    ("y".into(), 1e-9),
                    ("z".into(), 1e-9),
                ]),
                bounds: std::collections::HashMap::from([
                    ("y".into(), (0.0, 10.0)),
                    ("z".into(), (-30.0, 30.0)),
                ]),
                ..CorrectionNewtonOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{branch}-branch Bratu prepare: {error}"));
        solver
            .bind_parameters(vec!["lambda".into()], vec![lambda])
            .expect("Bratu parameter should bind to the prepared equation");
        let report = solver
            .solve()
            .unwrap_or_else(|error| panic!("{branch}-branch Bratu solve: {error}"));
        assert!(report.residual_norm.unwrap_or(f64::INFINITY) <= 1e-8);
        let full = solver
            .full_grid_values(&report.state)
            .expect("Bratu profile expands to its fixed mesh");
        let expected_center = 2.0 * parameter.cosh().ln();
        let actual_center = full[(steps, 0)];
        assert!(
            (actual_center - expected_center).abs() < 2.0e-2,
            "{branch} branch center={actual_center}, expected={expected_center}"
        );
        assert!(full[(0, 0)].abs() < 1e-14);
        assert!(full[(steps, 1)].abs() < 1e-14);
        center_values.push(actual_center);
    }

    assert!(center_values[1] > center_values[0] + 1.0);

    let parameter = 0.8_f64;
    let initial_branch_parameter = branch_parameter(parameter, 0.0, turning_point);
    let initial_guess = DMatrix::from_fn(2, steps, |state, column| {
        let node = if state == 0 { column + 1 } else { column };
        let x = half_domain * node as f64 / steps as f64;
        let shifted = initial_branch_parameter * (1.0 - 2.0 * x);
        if state == 0 {
            2.0 * (initial_branch_parameter.cosh() / shifted.cosh()).ln()
        } else {
            4.0 * initial_branch_parameter * shifted.tanh()
        }
    });
    let problem = CorrectionNewtonProblem {
        equations: vec![
            Expr::parse_expression("z"),
            Expr::parse_expression("-lambda*exp(y)"),
        ],
        initial_guess,
        state_names: vec!["y".into(), "z".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("y".into(), vec![(0, 0.0)]),
            ("z".into(), vec![(1, 0.0)]),
        ]),
        interval: (0.0, half_domain),
        steps,
    };
    let mut solver = CorrectionNewtonSolver::from_symbolic(
        problem,
        CorrectionNewtonOptions {
            matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
            relative_tolerances: std::collections::HashMap::from([
                ("y".into(), 1e-9),
                ("z".into(), 1e-9),
            ]),
            bounds: std::collections::HashMap::from([
                ("y".into(), (0.0, 10.0)),
                ("z".into(), (-30.0, 30.0)),
            ]),
            ..CorrectionNewtonOptions::default()
        },
    )
    .expect("lower-branch Bratu continuation should prepare");
    solver
        .bind_parameters(vec!["lambda".into()], vec![parameter])
        .expect("initial Bratu parameter should bind");
    solver
        .solve()
        .expect("initial lower-branch point should converge");
    for next_parameter in [0.9, 1.0] {
        let report = solver
            .continue_with_parameter_values(vec![next_parameter])
            .unwrap_or_else(|error| panic!("Bratu continuation to {next_parameter}: {error}"));
        let full = solver
            .full_grid_values(&report.state)
            .expect("continued Bratu profile expands to the full mesh");
        let lower_parameter = branch_parameter(next_parameter, 0.0, turning_point);
        let expected_center = 2.0 * lower_parameter.cosh().ln();
        assert!(
            (full[(steps, 0)] - expected_center).abs() < 2.0e-2,
            "continuation jumped branches at lambda={next_parameter}: center={} expected={expected_center}",
            full[(steps, 0)]
        );
    }
}

#[test]
fn nonlinear_reaction_diffusion_preserves_enthalpy_invariant() {
    const STEPS: usize = 64;
    const REFERENCE_SUBSTEPS: usize = 256;
    const REACTION_RATE: f64 = 0.8;
    const HEAT_RELEASE: f64 = 0.5;
    let initial = [0.8_f64, 0.0, HEAT_RELEASE * 0.2, 0.0];
    let derivative = |state: [f64; 4]| {
        let source = REACTION_RATE * state[2].exp() * state[0];
        [state[1], -source, state[3], HEAT_RELEASE * source]
    };
    let advance_rk4 = |state: [f64; 4], step: f64| {
        let k1 = derivative(state);
        let k2 = derivative(std::array::from_fn(|i| state[i] + 0.5 * step * k1[i]));
        let k3 = derivative(std::array::from_fn(|i| state[i] + 0.5 * step * k2[i]));
        let k4 = derivative(std::array::from_fn(|i| state[i] + step * k3[i]));
        std::array::from_fn(|i| state[i] + step * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
    };
    let reference_step = 1.0 / (STEPS * REFERENCE_SUBSTEPS) as f64;
    let mut reference = Vec::with_capacity(STEPS + 1);
    let mut state = initial;
    reference.push(state);
    for _ in 0..STEPS {
        for _ in 0..REFERENCE_SUBSTEPS {
            state = advance_rk4(state, reference_step);
        }
        reference.push(state);
    }
    let right_j = reference[STEPS][1];
    let right_q = reference[STEPS][3];
    let problem = CorrectionNewtonProblem {
        equations: ["J", "-0.8*exp(theta)*Y", "q", "0.4*exp(theta)*Y"]
            .into_iter()
            .map(Expr::parse_expression)
            .collect(),
        initial_guess: DMatrix::from_fn(4, STEPS, |component, column| {
            let node = if component == 0 || component == 2 {
                column + 1
            } else {
                column
            };
            reference[node][component]
        }),
        state_names: vec!["Y".into(), "J".into(), "theta".into(), "q".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("Y".into(), vec![(0, initial[0])]),
            ("J".into(), vec![(1, right_j)]),
            ("theta".into(), vec![(0, initial[2])]),
            ("q".into(), vec![(1, right_q)]),
        ]),
        interval: (0.0, 1.0),
        steps: STEPS,
    };

    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        let mut solver = CorrectionNewtonSolver::from_symbolic(
            problem.clone(),
            CorrectionNewtonOptions {
                symbolic_frontend: frontend,
                matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
                absolute_tolerance: 1.0e-9,
                default_relative_tolerance: 1.0e-9,
                bounds: std::collections::HashMap::from([
                    ("Y".into(), (0.0, 1.0)),
                    ("J".into(), (-1.0, 1.0)),
                    ("theta".into(), (-1.0, 1.0)),
                    ("q".into(), (-1.0, 1.0)),
                ]),
                telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                ..CorrectionNewtonOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{frontend:?} reaction-diffusion prepare: {error}"));
        let report = solver
            .solve()
            .unwrap_or_else(|error| panic!("{frontend:?} reaction-diffusion solve: {error}"));
        assert!(report.residual_norm.unwrap_or(f64::INFINITY) <= 1.0e-8);
        assert!(report.telemetry.jacobian_evaluations > 0);
        let full = solver
            .full_grid_values(&report.state)
            .expect("reaction-diffusion profile expands to the full mesh");
        let mut max_reference_error = 0.0_f64;
        let mut max_discrete_residual = 0.0_f64;
        for node in 0..=STEPS {
            assert!(full[(node, 0)] > 0.0 && full[(node, 0)] <= 1.0);
            let invariant = full[(node, 2)] + HEAT_RELEASE * full[(node, 0)];
            assert!((invariant - HEAT_RELEASE).abs() <= 1.0e-7);
            for component in 0..4 {
                max_reference_error = max_reference_error
                    .max((full[(node, component)] - reference[node][component]).abs());
            }
        }
        let mesh_step = 1.0 / STEPS as f64;
        for interval in 0..STEPS {
            let left: [f64; 4] = std::array::from_fn(|component| full[(interval, component)]);
            let right: [f64; 4] = std::array::from_fn(|component| full[(interval + 1, component)]);
            let left_rhs = derivative(left);
            let right_rhs = derivative(right);
            for component in 0..4 {
                let defect = right[component]
                    - left[component]
                    - 0.5 * mesh_step * (left_rhs[component] + right_rhs[component]);
                max_discrete_residual = max_discrete_residual.max(defect.abs());
            }
        }
        assert!(
            max_reference_error < 2.0e-4,
            "{frontend:?} profile differs from high-resolution RK4 reference by {max_reference_error:e}"
        );
        assert!(
            max_discrete_residual < 1.0e-7,
            "{frontend:?} trapezoid defect is {max_discrete_residual:e}"
        );
    }
}

#[test]
fn difficult_seed_reaction_diffusion_converges_on_sparse_and_banded_meshes() {
    const COARSE_STEPS: usize = 128;
    const FINE_STEPS: usize = 256;
    const REFERENCE_SUBSTEPS: usize = 256;
    const REACTION_RATE: f64 = 2.0;
    const HEAT_RELEASE: f64 = 0.5;
    const ABS_TOL: f64 = 1.0e-8;
    let initial = [0.8_f64, 0.0, HEAT_RELEASE * 0.2, 0.0];
    let derivative = |state: [f64; 4]| {
        let source = REACTION_RATE * state[2].exp() * state[0];
        [state[1], -source, state[3], HEAT_RELEASE * source]
    };
    let advance_rk4 = |state: [f64; 4], step: f64| {
        let k1 = derivative(state);
        let k2 = derivative(std::array::from_fn(|i| state[i] + 0.5 * step * k1[i]));
        let k3 = derivative(std::array::from_fn(|i| state[i] + 0.5 * step * k2[i]));
        let k4 = derivative(std::array::from_fn(|i| state[i] + step * k3[i]));
        std::array::from_fn(|i| state[i] + step * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
    };
    let reference_step = 1.0 / (FINE_STEPS * REFERENCE_SUBSTEPS) as f64;
    let mut fine_reference = Vec::with_capacity(FINE_STEPS + 1);
    let mut state = initial;
    fine_reference.push(state);
    for _ in 0..FINE_STEPS {
        for _ in 0..REFERENCE_SUBSTEPS {
            state = advance_rk4(state, reference_step);
        }
        fine_reference.push(state);
    }
    let right_j = fine_reference[FINE_STEPS][1];
    let right_q = fine_reference[FINE_STEPS][3];
    let mut errors_by_layout = Vec::new();

    for steps in [COARSE_STEPS, FINE_STEPS] {
        let reference_stride = FINE_STEPS / steps;
        let reference: Vec<[f64; 4]> = (0..=steps)
            .map(|node| fine_reference[node * reference_stride])
            .collect();
        let mesh_step = 1.0 / steps as f64;
        for (seed_name, seed_kind) in [
            ("constant", 0_u8),
            ("linear", 1_u8),
            ("shifted_zone", 2_u8),
            ("perturbed_reaction_profile", 3_u8),
            ("reference_control", 4_u8),
        ] {
            for layout in [
                CorrectionNewtonMatrixLayout::Sparse,
                CorrectionNewtonMatrixLayout::BandedAuto,
            ] {
                if seed_name == "reference_control" && steps == FINE_STEPS {
                    continue;
                }
                let mut profile: Vec<[f64; 4]> = (0..=steps)
                    .map(|node| {
                        let x = node as f64 / steps as f64;
                        match seed_kind {
                            0 => [initial[0], right_j, initial[2], right_q],
                            1 => [
                                initial[0] * (1.0 - 0.2 * x),
                                -0.16,
                                initial[2] + 0.1 * x,
                                0.1,
                            ],
                            2 => {
                                let zone = 1.0 / (1.0 + (-6.0 * (x - 0.65)).exp());
                                let zone_derivative = 6.0 * zone * (1.0 - zone);
                                [
                                    initial[0] * (1.0 - 0.3 * zone),
                                    -initial[0] * 0.3 * zone_derivative,
                                    initial[2] + 0.6 * zone,
                                    0.6 * zone_derivative,
                                ]
                            }
                            3 => [
                                initial[0] * (1.0 - 0.05 * x),
                                -initial[0] * 0.05,
                                initial[2] + 0.2 * x,
                                0.2,
                            ],
                            _ => reference[node],
                        }
                    })
                    .collect();
                profile[0][0] = initial[0];
                profile[steps][1] = right_j;
                profile[0][2] = initial[2];
                profile[steps][3] = right_q;
                let initial_guess = DMatrix::from_fn(4, steps, |component, column| {
                    let fixed_side = if component == 0 || component == 2 {
                        0
                    } else {
                        1
                    };
                    let node = if fixed_side == 0 { column + 1 } else { column };
                    profile[node][component]
                });
                let problem = CorrectionNewtonProblem {
                    equations: [
                        "J".to_string(),
                        format!("-{REACTION_RATE}*exp(theta)*Y"),
                        "q".to_string(),
                        format!("{}*exp(theta)*Y", HEAT_RELEASE * REACTION_RATE),
                    ]
                    .into_iter()
                    .map(|equation| Expr::parse_expression(&equation))
                    .collect(),
                    initial_guess,
                    state_names: vec!["Y".into(), "J".into(), "theta".into(), "q".into()],
                    independent_variable: "x".into(),
                    boundary_conditions: std::collections::HashMap::from([
                        ("Y".into(), vec![(0, initial[0])]),
                        ("J".into(), vec![(1, right_j)]),
                        ("theta".into(), vec![(0, initial[2])]),
                        ("q".into(), vec![(1, right_q)]),
                    ]),
                    interval: (0.0, 1.0),
                    steps,
                };
                let mut solver = CorrectionNewtonSolver::from_symbolic(
                    problem,
                    CorrectionNewtonOptions {
                        matrix_layout: layout,
                        absolute_tolerance: ABS_TOL,
                        default_relative_tolerance: ABS_TOL,
                        max_iterations: 100,
                        max_jacobian_age: 5,
                        max_damping_trials: 7,
                        bounds: std::collections::HashMap::from([
                            ("Y".into(), (0.0, 1.0)),
                            ("J".into(), (-10.0, 10.0)),
                            ("theta".into(), (-1.0, 2.0)),
                            ("q".into(), (-10.0, 10.0)),
                        ]),
                        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                        ..CorrectionNewtonOptions::default()
                    },
                )
                .unwrap_or_else(|error| panic!("{seed_name}/{layout:?}/{steps} prepare: {error}"));
                let initial_residual = (0..steps)
                    .flat_map(|interval| {
                        let left = profile[interval];
                        let right = profile[interval + 1];
                        let left_rhs = derivative(left);
                        let right_rhs = derivative(right);
                        (0..4).map(move |component| {
                            (right[component]
                                - left[component]
                                - 0.5 * mesh_step * (left_rhs[component] + right_rhs[component]))
                                .abs()
                        })
                    })
                    .fold(0.0_f64, f64::max);

                let report = match solver.solve() {
                    Ok(report) => report,
                    Err(error) => {
                        let partial = error.partial_report().map(|report| {
                            (
                                report.iterations,
                                report.telemetry.accepted_steps,
                                report.telemetry.damping_trials,
                                report.telemetry.damping_rejections,
                                report.telemetry.domain_rejections,
                                report.telemetry.jacobian_evaluations,
                                report.telemetry.factorizations,
                                report.telemetry.linear_solves,
                                report.correction_norm,
                            )
                        });
                        println!(
                            "DIFFICULT_SEED seed={seed_name} backend={layout:?} steps={steps} status=FAIL initial_discrete_residual={initial_residual:.6e} error={error} partial={partial:?}"
                        );
                        if seed_name == "perturbed_reaction_profile" {
                            panic!("primary difficult seed failed: {error}");
                        }
                        continue;
                    }
                };
                let full = solver
                    .full_grid_values(&report.state)
                    .expect("converged profile expands to the full mesh");
                let mut max_profile_error = 0.0_f64;
                let mut max_discrete_residual = 0.0_f64;
                let mut max_invariant_error = 0.0_f64;
                let mut max_source = f64::NEG_INFINITY;
                let mut max_source_x = 0.0_f64;
                for node in 0..=steps {
                    let actual: [f64; 4] = std::array::from_fn(|component| full[(node, component)]);
                    for component in 0..4 {
                        max_profile_error = max_profile_error
                            .max((actual[component] - reference[node][component]).abs());
                    }
                    max_invariant_error = max_invariant_error
                        .max((actual[2] + HEAT_RELEASE * actual[0] - HEAT_RELEASE).abs());
                    let source = REACTION_RATE * actual[2].exp() * actual[0];
                    if source > max_source {
                        max_source = source;
                        max_source_x = node as f64 * mesh_step;
                    }
                    if node < steps {
                        let right: [f64; 4] =
                            std::array::from_fn(|component| full[(node + 1, component)]);
                        let left_rhs = derivative(actual);
                        let right_rhs = derivative(right);
                        for component in 0..4 {
                            let defect = right[component]
                                - actual[component]
                                - 0.5 * mesh_step * (left_rhs[component] + right_rhs[component]);
                            max_discrete_residual = max_discrete_residual.max(defect.abs());
                        }
                    }
                }
                let boundary_residual = [
                    (full[(0, 0)] - initial[0]).abs(),
                    (full[(steps, 1)] - right_j).abs(),
                    (full[(0, 2)] - initial[2]).abs(),
                    (full[(steps, 3)] - right_q).abs(),
                ]
                .into_iter()
                .fold(0.0_f64, f64::max);
                println!(
                    "DIFFICULT_SEED seed={seed_name} backend={layout:?} steps={steps} status=PASS initial_residual={initial_residual:.6e} iterations={} accepted={} damping_trials={} damping_rejections={} domain_rejections={} jacobians={} factorizations={} linear_solves={} final_correction={:.6e} discrete_residual={max_discrete_residual:.6e} boundary_residual={boundary_residual:.6e} invariant_error={max_invariant_error:.6e} max_profile_error={max_profile_error:.6e} source_peak=({max_source:.6e}@{max_source_x:.4})",
                    report.iterations,
                    report.telemetry.accepted_steps,
                    report.telemetry.damping_trials,
                    report.telemetry.damping_rejections,
                    report.telemetry.domain_rejections,
                    report.telemetry.jacobian_evaluations,
                    report.telemetry.factorizations,
                    report.telemetry.linear_solves,
                    report.correction_norm.unwrap_or(f64::INFINITY)
                );
                assert!(boundary_residual <= 1.0e-14);
                assert!(max_invariant_error <= 1.0e-7);
                assert!(max_discrete_residual <= 1.0e-7);
                assert!(max_profile_error <= 2.0e-4);
                assert!(max_source.is_finite() && max_source_x.is_finite());
                if seed_name == "perturbed_reaction_profile" {
                    assert!(report.iterations > 1, "difficult seed must exercise Newton");
                    errors_by_layout.push((layout, steps, max_profile_error));
                }
            }
        }
    }

    for layout in [
        CorrectionNewtonMatrixLayout::Sparse,
        CorrectionNewtonMatrixLayout::BandedAuto,
    ] {
        let coarse = errors_by_layout
            .iter()
            .find(|(candidate, steps, _)| *candidate == layout && *steps == COARSE_STEPS)
            .expect("coarse perturbed-seed report exists")
            .2;
        let fine = errors_by_layout
            .iter()
            .find(|(candidate, steps, _)| *candidate == layout && *steps == FINE_STEPS)
            .expect("fine perturbed-seed report exists")
            .2;
        assert!(
            fine < coarse,
            "{layout:?} refinement should reduce profile error: coarse={coarse:e}, fine={fine:e}"
        );
    }
}

#[test]
#[ignore = "compiles generated artifacts for the CorrectionNewton AOT frontend/backend parity matrix"]
fn aot_frontends_match_lambdify_across_linear_layouts() {
    let root = std::env::var_os("BVP_CORRECTION_NEWTON_AOT_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rst-cn-aot"));
    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        for layout in [
            CorrectionNewtonMatrixLayout::Dense,
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let make_problem = || CorrectionNewtonProblem {
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
            let mut lambdify = CorrectionNewtonSolver::from_symbolic(
                make_problem(),
                CorrectionNewtonOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                    ..CorrectionNewtonOptions::default()
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
            let mut aot = CorrectionNewtonSolver::from_symbolic(
                make_problem(),
                CorrectionNewtonOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                    ..CorrectionNewtonOptions::default()
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
        CorrectionNewtonLinearBackend::Dense,
        CorrectionNewtonLinearBackend::Sparse,
        CorrectionNewtonLinearBackend::Banded { lower: 1, upper: 0 },
    ];
    let mut reference: Option<f64> = None;
    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        for (layout, backend) in [
            CorrectionNewtonMatrixLayout::Dense,
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ]
        .into_iter()
        .zip(expected_backend)
        {
            let problem = CorrectionNewtonProblem {
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
            let options = CorrectionNewtonOptions {
                symbolic_frontend: frontend,
                matrix_layout: layout,
                telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                ..CorrectionNewtonOptions::default()
            };
            let mut solver = CorrectionNewtonSolver::from_symbolic(problem, options)
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
            if frontend == CorrectionNewtonSymbolicFrontend::ExprLegacy {
                if let Some(reference) = reference {
                    assert!((full[(4, 0)] - reference).abs() < 1e-13);
                } else {
                    reference = Some(full[(4, 0)]);
                }
            }
        }
    }
}

fn nonlinear_exact_problem(steps: usize) -> CorrectionNewtonProblem {
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

    CorrectionNewtonProblem {
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
            CorrectionNewtonSymbolicFrontend::ExprLegacy,
            CorrectionNewtonSymbolicFrontend::AtomView,
        ] {
            for layout in [
                CorrectionNewtonMatrixLayout::Dense,
                CorrectionNewtonMatrixLayout::Sparse,
                CorrectionNewtonMatrixLayout::BandedAuto,
            ] {
                let options = CorrectionNewtonOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    absolute_tolerance: 1e-10,
                    default_relative_tolerance: 1e-9,
                    telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                    ..CorrectionNewtonOptions::default()
                };
                let mut solver =
                    CorrectionNewtonSolver::from_symbolic(nonlinear_exact_problem(steps), options)
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

fn gaussian_two_point_problem(steps: usize) -> CorrectionNewtonProblem {
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
    CorrectionNewtonProblem {
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
            CorrectionNewtonSymbolicFrontend::ExprLegacy,
            CorrectionNewtonSymbolicFrontend::AtomView,
        ] {
            for layout in [
                CorrectionNewtonMatrixLayout::Dense,
                CorrectionNewtonMatrixLayout::Sparse,
                CorrectionNewtonMatrixLayout::BandedAuto,
            ] {
                let options = CorrectionNewtonOptions {
                    symbolic_frontend: frontend,
                    matrix_layout: layout,
                    absolute_tolerance: 1e-10,
                    default_relative_tolerance: 1e-9,
                    bounds: std::collections::HashMap::from([("y".into(), (1e-6, 2.0))]),
                    telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                    ..CorrectionNewtonOptions::default()
                };
                let mut solver = CorrectionNewtonSolver::from_symbolic(
                    gaussian_two_point_problem(steps),
                    options,
                )
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
enum CorrectionNewtonAotToolchain {
    CcGcc,
    CcTcc,
    Rust,
    Zig,
}

impl CorrectionNewtonAotToolchain {
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
#[ignore = "builds CorrectionNewton AOT artifacts for all supported compilers and optimization presets"]
fn aot_compiler_and_profile_matrix_matches_lambdify() {
    use crate::symbolic::codegen::codegen_aot_driver::AotBuildPreset;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::symbolic_ivp_generated::{
        SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    };
    use std::path::PathBuf;

    let output_root = std::env::var_os("BVP_CORRECTION_NEWTON_AOT_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rst-cn-aot"));
    let presets = [
        AotBuildPreset::Production,
        AotBuildPreset::FastBuild,
        AotBuildPreset::DevFastest,
    ];
    let toolchains = [
        CorrectionNewtonAotToolchain::CcGcc,
        CorrectionNewtonAotToolchain::CcTcc,
        CorrectionNewtonAotToolchain::Rust,
        CorrectionNewtonAotToolchain::Zig,
    ];
    let mut rows = Vec::new();

    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
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

                let make_problem = || CorrectionNewtonProblem {
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
                let mut lambdify = CorrectionNewtonSolver::from_symbolic(
                    make_problem(),
                    CorrectionNewtonOptions {
                        symbolic_frontend: frontend,
                        matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
                        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                        ..CorrectionNewtonOptions::default()
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
                let mut aot = CorrectionNewtonSolver::from_symbolic(
                    make_problem(),
                    CorrectionNewtonOptions {
                        symbolic_frontend: frontend,
                        symbolic_execution: CorrectionNewtonSymbolicExecution::Aot,
                        aot_config: Some(aot_config),
                        matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
                        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                        ..CorrectionNewtonOptions::default()
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
        "CorrectionNewton AOT compiler/profile correctness matrix\n\nCompiler routes: GCC, TCC, Rust and Zig. Presets: Production, FastBuild and DevFastest. Layout: Sparse. Each row checks ExprLegacy/AtomView parity with Lambdify before and after fixed-input rebinding. Build and cache lifecycle are forced into a unique route directory.\n\n{table}\n"
    );
    if let Err(error) = crate::Utils::test_reporting::write_test_report(
        "BVP_CorrectionNewton_AOT",
        "aot_compiler_and_profile_matrix",
        &report,
    ) {
        eprintln!("[BVP_CorrectionNewton AOT report] unable to write report: {error}");
    }
}
