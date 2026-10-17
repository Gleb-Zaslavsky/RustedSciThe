use crate::Utils::test_reporting::{TestReportCapture, capture_test_table};
use crate::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;
use crate::numerical::optimization::lm_gavin::{
    ExponentialSinusoidalModel, GavinLmTermination, GavinLmUpdatePolicy, LevenbergMarquardt,
};
use crate::numerical::optimization::universal_fitting::{
    Method, UniversalFitting, UniversalFittingResult,
};
use tabled::Tabled;

#[derive(Tabled)]
struct GavinStoryRow {
    frontend: &'static str,
    update_policy: &'static str,
    samples: usize,
    prepare_ms: String,
    solve_ms: String,
    full_ms: String,
    iterations: usize,
    objective_calls: usize,
    jacobian_calls: usize,
    objective_ms: String,
    jacobian_ms: String,
    linearization_ms: String,
    linear_solve_ms: String,
    fd_jacobians: usize,
    accepted: usize,
    rejected: usize,
    termination: String,
    max_parameter_error: String,
}

#[derive(Tabled)]
struct GavinPythonParityRow {
    fixture: &'static str,
    policy: &'static str,
    iterations: usize,
    python_objective_calls: usize,
    rust_objective_calls: usize,
    python_linear_solves: usize,
    rust_linear_solves: usize,
    accepted: usize,
    rejected: usize,
    fd_jacobians: usize,
    broyden_updates: usize,
    termination: &'static str,
    max_history_error: String,
    max_parameter_error: String,
    reduced_chi_square: String,
    python_refresh_objective_delta: String,
}

struct GavinPythonGolden {
    fixture: &'static str,
    policy: GavinLmUpdatePolicy,
    initial: [f64; 4],
    expected_history: &'static [[f64; 5]],
    accepted: usize,
    rejected: usize,
    python_objective_calls: usize,
    rust_objective_calls: usize,
    python_linear_solves: usize,
    fd_jacobians: usize,
    broyden_updates: usize,
    termination: GavinLmTermination,
    corrected_refresh_x2: Option<f64>,
}

#[derive(Tabled)]
struct GavinPythonBoundsRow {
    solver: &'static str,
    iterations: usize,
    accepted: usize,
    rejected: usize,
    rust_objective_calls: usize,
    fd_jacobians: usize,
    broyden_updates: usize,
    termination: &'static str,
    final_parameters: String,
    max_checkpoint_parameter_error: String,
}

const MARQUARDT_BASELINE: &[[f64; 5]] = &[
    [
        9.18332080958e-6,
        1.96737877666,
        0.531341446239,
        0.953316542136,
        2.00322533328,
    ],
    [
        1.46792717673e-6,
        1.95136957896,
        0.509913236413,
        1.00480004532,
        2.00514645862,
    ],
    [
        2.09557761324e-7,
        1.96142517027,
        0.507482128098,
        1.00419510158,
        1.99990585122,
    ],
    [
        9.48435322361e-8,
        1.97498022474,
        0.508601832355,
        0.996553263748,
        2.00026733742,
    ],
    [
        2.15379422553e-9,
        1.99846520254,
        0.500449967249,
        0.999312764688,
        1.99996748516,
    ],
];

const QUADRATIC_BASELINE: &[[f64; 5]] = &[
    [
        6.04790314211e-5,
        1.90762018771,
        0.550014739064,
        0.932195373215,
        1.9688037575,
    ],
    [
        6.84988550144e-6,
        1.91934358603,
        0.537406940253,
        0.968676434517,
        1.9925650623,
    ],
    [
        8.56241322045e-7,
        1.93120053499,
        0.525057956821,
        0.990417152788,
        1.99829744425,
    ],
    [
        2.57595810222e-7,
        1.95243272914,
        0.514662626546,
        0.996975552255,
        1.99901757482,
    ],
    [
        7.6429664362e-8,
        1.97237644505,
        0.508112532036,
        0.998449164501,
        1.99960298504,
    ],
    [
        1.57390717188e-8,
        1.9864877975,
        0.503712290345,
        0.999541247108,
        1.99990274236,
    ],
    [
        2.77367407374e-9,
        1.99424805565,
        0.501533140092,
        0.999857824097,
        1.99996570141,
    ],
];

const NIELSEN_BASELINE: &[[f64; 5]] = &[
    [
        1.14293170165e-5,
        1.91142654785,
        0.550022101374,
        0.948277284824,
        2.00317192398,
    ],
    [
        1.93772247923e-6,
        1.93037162683,
        0.518686127045,
        1.00602215776,
        2.005540305,
    ],
    [
        2.17482900603e-7,
        1.97443125418,
        0.504164475748,
        1.00170121527,
        1.99814991436,
    ],
    [
        2.37989828403e-8,
        1.99114205876,
        0.503776769486,
        0.997918577338,
        2.000169442,
    ],
    [
        8.23273147427e-10,
        1.99883890255,
        0.499786152742,
        1.00037125963,
        1.99998387089,
    ],
];

const MARQUARDT_REJECTION_REFRESH: &[[f64; 5]] = &[
    [
        0.00362439353131,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
    [
        0.0106739854527,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
    [
        0.0106739854527,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
    [
        0.0106739854527,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
    [
        0.0106739854527,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
    [
        0.0106739854527,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
    [
        0.0106739854527,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
    [
        0.0106739854527,
        1.23512435148,
        1.42094490417,
        -0.265197413391,
        1.38850702362,
    ],
];

const QUADRATIC_REJECTION: &[[f64; 5]] = &[
    [
        0.00416255409406,
        1.18316368029,
        1.41770386732,
        0.32549168676,
        2.83778512582,
    ],
    [
        0.00344061907473,
        1.38040585661,
        1.95652949712,
        0.102469233498,
        2.62888255548,
    ],
    [
        0.00416255409406,
        1.38040585661,
        1.95652949712,
        0.102469233498,
        2.62888255548,
    ],
    [
        0.00355926274191,
        1.42090948173,
        2.00627989248,
        0.00300604210352,
        2.44737396482,
    ],
    [
        0.00416255409406,
        1.42090948173,
        2.00627989248,
        0.00300604210352,
        2.44737396482,
    ],
    [
        0.00361807406632,
        1.31334800124,
        1.79640278109,
        0.0150566938421,
        2.41255250946,
    ],
    [
        0.00416255409406,
        1.31334800124,
        1.79640278109,
        0.0150566938421,
        2.41255250946,
    ],
    [
        0.00416255409406,
        1.31334800124,
        1.79640278109,
        0.0150566938421,
        2.41255250946,
    ],
];

const NIELSEN_REJECTION: &[[f64; 5]] = &[
    [
        9.29800599472e-5,
        1.03364321134,
        0.811598289927,
        0.980266980516,
        2.00104873854,
    ],
    [
        3.62270123444e-5,
        1.37097050776,
        0.732609697852,
        0.998904337341,
        2.00060755914,
    ],
    [
        1.10394017959e-5,
        1.70628355098,
        0.511216660674,
        0.992490561012,
        1.99931019883,
    ],
    [
        2.5018469181e-6,
        1.96561403413,
        0.562676429802,
        1.00037468517,
        2.00019430596,
    ],
    [
        1.10394017959e-5,
        1.96561403413,
        0.562676429802,
        1.00037468517,
        2.00019430596,
    ],
    [
        2.26777066244e-6,
        1.97175044446,
        0.45795276482,
        0.999970906625,
        2.00020562692,
    ],
    [
        2.06442554512e-9,
        2.00099524382,
        0.498374103563,
        0.999878548118,
        1.99995851786,
    ],
];

fn python_controller_goldens() -> [GavinPythonGolden; 6] {
    [
        GavinPythonGolden {
            fixture: "baseline",
            policy: GavinLmUpdatePolicy::Marquardt,
            initial: [1.9, 0.55, 0.9, 1.9],
            expected_history: MARQUARDT_BASELINE,
            python_objective_calls: 16,
            rust_objective_calls: 10,
            python_linear_solves: 5,
            accepted: 5,
            rejected: 0,
            fd_jacobians: 1,
            broyden_updates: 6,
            termination: GavinLmTermination::GradientTolerance,
            corrected_refresh_x2: None,
        },
        GavinPythonGolden {
            fixture: "baseline",
            policy: GavinLmUpdatePolicy::Quadratic,
            initial: [1.9, 0.55, 0.9, 1.9],
            expected_history: QUADRATIC_BASELINE,
            python_objective_calls: 27,
            rust_objective_calls: 19,
            python_linear_solves: 7,
            accepted: 7,
            rejected: 0,
            fd_jacobians: 1,
            broyden_updates: 8,
            termination: GavinLmTermination::GradientTolerance,
            corrected_refresh_x2: None,
        },
        GavinPythonGolden {
            fixture: "baseline",
            policy: GavinLmUpdatePolicy::Nielsen,
            initial: [1.9, 0.55, 0.9, 1.9],
            expected_history: NIELSEN_BASELINE,
            python_objective_calls: 16,
            rust_objective_calls: 10,
            python_linear_solves: 5,
            accepted: 5,
            rejected: 0,
            fd_jacobians: 1,
            broyden_updates: 6,
            termination: GavinLmTermination::GradientTolerance,
            corrected_refresh_x2: None,
        },
        GavinPythonGolden {
            fixture: "rejection_refresh",
            policy: GavinLmUpdatePolicy::Marquardt,
            initial: [1.94, 0.55, 1.0, 1.38],
            expected_history: MARQUARDT_REJECTION_REFRESH,
            python_objective_calls: 24,
            rust_objective_calls: 22,
            python_linear_solves: 8,
            accepted: 1,
            rejected: 7,
            fd_jacobians: 3,
            broyden_updates: 1,
            termination: GavinLmTermination::ParameterStepTolerance,
            corrected_refresh_x2: Some(0.00362439353131),
        },
        GavinPythonGolden {
            fixture: "rejection_quadratic",
            policy: GavinLmUpdatePolicy::Quadratic,
            initial: [0.63, 0.74, 0.98, 3.06],
            expected_history: QUADRATIC_REJECTION,
            python_objective_calls: 35,
            rust_objective_calls: 30,
            python_linear_solves: 8,
            accepted: 4,
            rejected: 4,
            fd_jacobians: 3,
            broyden_updates: 4,
            termination: GavinLmTermination::ParameterStepTolerance,
            corrected_refresh_x2: Some(0.00361807406632),
        },
        GavinPythonGolden {
            fixture: "rejection_nielsen",
            policy: GavinLmUpdatePolicy::Nielsen,
            initial: [0.76, 0.77, 0.94, 1.95],
            expected_history: NIELSEN_REJECTION,
            python_objective_calls: 19,
            rust_objective_calls: 12,
            python_linear_solves: 7,
            accepted: 6,
            rejected: 1,
            fd_jacobians: 1,
            broyden_updates: 7,
            termination: GavinLmTermination::GradientTolerance,
            corrected_refresh_x2: None,
        },
    ]
}

fn execute(
    frontend: SymbolicLambdifyFrontend,
    update_policy: GavinLmUpdatePolicy,
    sample_count: usize,
) -> GavinStoryRow {
    let x = (0..sample_count)
        .map(|i| 20.0 * i as f64 / (sample_count - 1) as f64)
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|&x| 2.0 * (-x / 0.5_f64).exp() + 1.0 * (x / 2.0_f64).sin())
        .collect::<Vec<_>>();
    let started = std::time::Instant::now();
    let result = UniversalFitting::new()
        .with_method(Method::GAVIN)
        .with_data(x, y)
        .with_equation_str("a*exp(-x/tau)+c*sin(x/omega)".to_owned())
        .with_arg("x".to_owned())
        .with_unknowns(vec![
            "a".to_owned(),
            "tau".to_owned(),
            "c".to_owned(),
            "omega".to_owned(),
        ])
        .with_initial_guess(vec![1.9, 0.55, 0.9, 1.9])
        .with_gavin_frontend(frontend)
        .with_gavin_update_policy(update_policy)
        .with_gavin_telemetry(crate::numerical::SolverTelemetryMode::Detailed)
        .build()
        .expect("Gavin fit should complete");
    let full_ms = started.elapsed().as_secs_f64() * 1e3;
    let UniversalFittingResult::GAVIN(fit) = result else {
        panic!("Gavin method returned a different result variant");
    };
    let expected = [2.0, 0.5, 1.0, 2.0];
    let max_parameter_error = fit
        .fit
        .parameters
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(fit.fit.termination.is_converged());
    assert!(
        max_parameter_error < 2e-2,
        "parameter error={max_parameter_error}"
    );
    let telemetry = fit.telemetry.expect("story enables solve telemetry");
    GavinStoryRow {
        frontend: match frontend {
            SymbolicLambdifyFrontend::ExprLegacy => "ExprLegacy",
            SymbolicLambdifyFrontend::AtomViewNative => "AtomViewNative",
        },
        update_policy: match update_policy {
            GavinLmUpdatePolicy::Marquardt => "Marquardt",
            GavinLmUpdatePolicy::Quadratic => "Quadratic",
            GavinLmUpdatePolicy::Nielsen => "Nielsen",
        },
        samples: sample_count,
        prepare_ms: format!(
            "{:.3}",
            fit.preparation.preparation_duration.as_secs_f64() * 1e3
        ),
        solve_ms: format!("{:.3}", telemetry.total_time.as_secs_f64() * 1e3),
        full_ms: format!("{full_ms:.3}"),
        iterations: telemetry.iterations,
        objective_calls: telemetry.objective_evaluations,
        jacobian_calls: telemetry.analytic_jacobian_evaluations,
        objective_ms: format!("{:.3}", telemetry.objective_time.as_secs_f64() * 1e3),
        jacobian_ms: format!(
            "{:.3}",
            telemetry.analytic_jacobian_time.as_secs_f64() * 1e3
        ),
        linearization_ms: format!("{:.3}", telemetry.linearization_time.as_secs_f64() * 1e3),
        linear_solve_ms: format!("{:.3}", telemetry.linear_solve_time.as_secs_f64() * 1e3),
        fd_jacobians: telemetry.finite_difference_jacobians,
        accepted: telemetry.accepted_steps,
        rejected: telemetry.rejected_steps,
        termination: format!("{:?}", fit.fit.termination),
        max_parameter_error: format!("{max_parameter_error:.3e}"),
    }
}

#[test]
#[ignore = "release story: matched Gavin preparation/solve frontend matrix"]
fn gavin_universal_fitting_frontend_performance_story() {
    let _report = TestReportCapture::new(
        "Optimization_Gavin",
        "gavin_universal_fitting_frontend_performance_story",
    );
    let mut rows = Vec::new();
    for samples in [201, 1001] {
        for policy in [
            GavinLmUpdatePolicy::Marquardt,
            GavinLmUpdatePolicy::Quadratic,
            GavinLmUpdatePolicy::Nielsen,
        ] {
            for frontend in [
                SymbolicLambdifyFrontend::ExprLegacy,
                SymbolicLambdifyFrontend::AtomViewNative,
            ] {
                rows.push(execute(frontend, policy, samples));
            }
        }
    }
    capture_test_table(
        "Gavin matched frontend/policy matrix; preparation, solver-internal, and external full time are separate; inner solver stages overlap",
        &rows,
    );
}

#[test]
#[ignore = "controller parity story: Python golden histories and rejection/refresh branches"]
fn gavin_python_controller_history_parity_story() {
    let _report = TestReportCapture::new(
        "Optimization_Gavin",
        "gavin_python_controller_history_parity_story",
    );
    let sample_count = 201;
    let t = nalgebra::DVector::from_iterator(
        sample_count,
        (0..sample_count).map(|index| index as f64 * 0.1),
    );
    let observations = t.map(|sample| 2.0 * (-sample / 0.5_f64).exp() + (sample / 2.0_f64).sin());
    let mut rows = Vec::new();

    for golden in python_controller_goldens() {
        let mut solver = LevenbergMarquardt::new(ExponentialSinusoidalModel)
            .with_update_policy(golden.policy)
            .with_telemetry(crate::numerical::SolverTelemetryMode::Detailed);
        let result = solver
            .lm_detailed(
                nalgebra::DVector::from_column_slice(&golden.initial),
                &t,
                &observations,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", golden.fixture));
        let telemetry = solver.telemetry().expect("telemetry enabled");
        assert_eq!(
            result.convergence_history.nrows(),
            golden.expected_history.len()
        );
        assert_eq!(telemetry.iterations, golden.expected_history.len());
        assert_eq!(
            telemetry.accepted_steps, golden.accepted,
            "{}",
            golden.fixture
        );
        assert_eq!(
            telemetry.rejected_steps, golden.rejected,
            "{}",
            golden.fixture
        );
        assert_eq!(
            telemetry.finite_difference_jacobians, golden.fd_jacobians,
            "{}",
            golden.fixture
        );
        assert_eq!(
            telemetry.broyden_updates, golden.broyden_updates,
            "{}",
            golden.fixture
        );
        assert_eq!(telemetry.linear_solves, golden.python_linear_solves);
        assert_eq!(
            telemetry.objective_evaluations, golden.rust_objective_calls,
            "{}",
            golden.fixture
        );
        assert_eq!(result.termination, golden.termination, "{}", golden.fixture);

        let mut max_history_error = 0.0_f64;
        let mut max_parameter_error = 0.0_f64;
        let mut python_refresh_objective_delta = String::from("-");
        for (iteration, expected_row) in golden.expected_history.iter().enumerate() {
            let actual = result.convergence_history.row(iteration);
            for (column, expected) in expected_row.iter().enumerate() {
                let actual = actual[column + 1]; // Python history starts with callback count.
                let error = (actual - expected).abs();
                if column > 0 {
                    max_history_error = max_history_error.max(error);
                    max_parameter_error = max_parameter_error.max(error);
                    assert!(
                        error <= 2e-8 + 2e-7 * expected.abs(),
                        "{} iteration {} parameter column {}: Rust={actual:.12e}, Python={expected:.12e}",
                        golden.fixture,
                        iteration + 1,
                        column
                    );
                } else if let Some(corrected_x2) = golden.corrected_refresh_x2
                    && iteration + 1 == golden.expected_history.len()
                {
                    let correction_error = (actual - corrected_x2).abs();
                    assert!(
                        correction_error <= 2e-12 + 2e-5 * corrected_x2.abs(),
                        "{} periodic-refresh objective: Rust={actual:.12e}, corrected={corrected_x2:.12e}",
                        golden.fixture
                    );
                    python_refresh_objective_delta = format!("{error:.3e}");
                } else {
                    max_history_error = max_history_error.max(error);
                    assert!(
                        error <= 2e-12 + 2e-5 * expected.abs(),
                        "{} iteration {} reduced chi-square: Rust={actual:.12e}, Python={expected:.12e}",
                        golden.fixture,
                        iteration + 1
                    );
                }
            }
        }

        rows.push(GavinPythonParityRow {
            fixture: golden.fixture,
            policy: match golden.policy {
                GavinLmUpdatePolicy::Marquardt => "Marquardt",
                GavinLmUpdatePolicy::Quadratic => "Quadratic",
                GavinLmUpdatePolicy::Nielsen => "Nielsen",
            },
            iterations: telemetry.iterations,
            python_objective_calls: golden.python_objective_calls,
            rust_objective_calls: telemetry.objective_evaluations,
            python_linear_solves: golden.python_linear_solves,
            rust_linear_solves: telemetry.linear_solves,
            accepted: telemetry.accepted_steps,
            rejected: telemetry.rejected_steps,
            fd_jacobians: telemetry.finite_difference_jacobians,
            broyden_updates: telemetry.broyden_updates,
            termination: match result.termination {
                GavinLmTermination::GradientTolerance => "gradient_tolerance",
                GavinLmTermination::ParameterStepTolerance => "parameter_step_tolerance",
                GavinLmTermination::MaxIterations => "max_iterations",
                GavinLmTermination::NonFiniteTrial => "non_finite_trial",
            },
            max_history_error: format!("{max_history_error:.3e}"),
            max_parameter_error: format!("{max_parameter_error:.3e}"),
            reduced_chi_square: format!("{:.12e}", result.reduced_chi_square),
            python_refresh_objective_delta,
        });
    }

    capture_test_table(
        "Rust vs Python Gavin controller parity; histories, branch counters, and parameters are checked; the known Python periodic-refresh objective-cache bug is reported separately",
        &rows,
    );
}

#[test]
#[ignore = "controller parity story: active Gavin coefficient bounds against Python"]
fn gavin_python_active_coefficient_bounds_parity_story() {
    let _report = TestReportCapture::new(
        "Optimization_Gavin",
        "gavin_python_active_coefficient_bounds_parity_story",
    );
    let t = nalgebra::DVector::from_iterator(201, (0..201).map(|index| index as f64 * 0.1));
    let observations = t.map(|sample| 2.0 * (-sample / 0.5_f64).exp() + (sample / 2.0).sin());
    let mut solver = LevenbergMarquardt::new(ExponentialSinusoidalModel)
        .with_telemetry(crate::numerical::SolverTelemetryMode::Detailed);
    let result = solver
        .lm_detailed(
            nalgebra::DVector::from_column_slice(&[0.01, 0.55, 0.9, 1.9]),
            &t,
            &observations,
        )
        .expect("active-bound fit should return its constrained solution");
    let telemetry = solver.telemetry().expect("telemetry enabled");

    // Gavin's source clamps each coefficient to +/-100 times its initial
    // magnitude. The tiny initial amplitude therefore makes p0=1 an active
    // upper bound while the remaining parameters continue to adapt.
    let expected_checkpoints: [(usize, [f64; 4]); 7] = [
        (1, [0.01, 0.55, 0.9, 1.9]),
        (
            4,
            [
                0.14169025302958485,
                7.37280650432259,
                0.9043061557633918,
                1.9071623215502398,
            ],
        ),
        (
            100,
            [
                1.0,
                0.9202185731378543,
                0.9853185154083796,
                2.0003251135043048,
            ],
        ),
        (
            200,
            [
                1.0,
                0.9257882684849157,
                0.9865383458701159,
                2.0002804837929706,
            ],
        ),
        (
            300,
            [
                1.0,
                0.9282099237721158,
                0.9868522245533135,
                2.0002649754181228,
            ],
        ),
        (
            384,
            [
                1.0,
                0.9293436278561849,
                0.9869594156981102,
                2.0002585261305117,
            ],
        ),
        (
            385,
            [
                1.0,
                0.9293556346552438,
                0.9869601207993766,
                2.0002584878064233,
            ],
        ),
    ];
    assert_eq!(telemetry.iterations, 385);
    assert_eq!(telemetry.accepted_steps, 199);
    assert_eq!(telemetry.rejected_steps, 186);
    assert_eq!(telemetry.finite_difference_jacobians, 49);
    assert_eq!(telemetry.broyden_updates, 175);
    assert_eq!(telemetry.linear_solves, 385);
    assert_eq!(
        result.termination,
        GavinLmTermination::ParameterStepTolerance
    );
    assert_eq!(result.convergence_history.nrows(), 385);

    let mut max_checkpoint_parameter_error = 0.0_f64;
    for (iteration, expected) in expected_checkpoints {
        let actual = result.convergence_history.row(iteration - 1);
        for parameter in 0..4 {
            let value = actual[parameter + 2];
            let error = (value - expected[parameter]).abs();
            max_checkpoint_parameter_error = max_checkpoint_parameter_error.max(error);
            assert!(
                error <= 2e-7 + 2e-7 * expected[parameter].abs(),
                "iteration {iteration} p{parameter}: Rust={value:.12e}, Python={:.12e}",
                expected[parameter]
            );
        }
    }
    assert_eq!(
        result.parameters[0], 1.0,
        "the coefficient upper bound is active"
    );

    let row = GavinPythonBoundsRow {
        solver: "Gavin Rust vs Python reference",
        iterations: telemetry.iterations,
        accepted: telemetry.accepted_steps,
        rejected: telemetry.rejected_steps,
        rust_objective_calls: telemetry.objective_evaluations,
        fd_jacobians: telemetry.finite_difference_jacobians,
        broyden_updates: telemetry.broyden_updates,
        termination: "parameter_step_tolerance",
        final_parameters: format!("{:.10?}", result.parameters.as_slice()),
        max_checkpoint_parameter_error: format!("{max_checkpoint_parameter_error:.3e}"),
    };
    capture_test_table(
        "Active coefficient-bound parity; Python checkpoints cover the initial, bound-entry, long-run, and final states; objective histories are not compared across the documented cache defect",
        &[row],
    );
}
