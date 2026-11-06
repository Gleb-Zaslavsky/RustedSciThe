use super::*;

pub(crate) const COMB01_REFERENCE: &str = include_str!("../reference/comb01_scipy.csv");
pub(crate) const COMB02_REFERENCE: &str = include_str!("../reference/comb02_scipy.csv");

#[derive(tabled::Tabled)]
struct CombustionRow {
    fixture: &'static str,
    seed: &'static str,
    steps: usize,
    route: String,
    iterations: usize,
    jacobians: usize,
    factorizations: usize,
    residual: String,
    profile_error: String,
    invariant_error: String,
    physical_summary: String,
}

pub(crate) fn parse_reference(csv: &str, state_count: usize) -> Vec<(f64, Vec<f64>)> {
    csv.lines()
        .skip(1)
        .map(|line| {
            let values = line
                .split(',')
                .map(|field| field.parse::<f64>().expect("reference CSV is numeric"))
                .collect::<Vec<_>>();
            assert_eq!(values.len(), state_count + 1);
            (values[0], values[1..].to_vec())
        })
        .collect()
}

fn reference_at(reference: &[(f64, Vec<f64>)], x: f64) -> Vec<f64> {
    let scaled = x * (reference.len() - 1) as f64;
    let left = scaled.floor() as usize;
    let right = (left + 1).min(reference.len() - 1);
    let fraction = scaled - left as f64;
    reference[left]
        .1
        .iter()
        .zip(&reference[right].1)
        .map(|(a, b)| a + fraction * (b - a))
        .collect()
}

pub(crate) fn comb01_problem(steps: usize) -> CorrectionNewtonProblem {
    let pe = 5.0;
    let da = 2.0;
    let heat = 2.0;
    let beta = 5.0;
    let gamma = 0.5;
    let approximate = |x: f64| {
        let fuel = (-da * x).exp();
        let flux = -da * fuel;
        [heat * (1.0 - fuel), -heat * flux, fuel, flux]
    };
    let initial_guess = DMatrix::from_fn(4, steps, |state, column| {
        let node = if state == 0 || state == 2 {
            column + 1
        } else {
            column
        };
        approximate(node as f64 / steps as f64)[state]
    });
    let rate = format!("{da} * Y * exp({beta} * (1.0 - 1.0 / (1.0 + {gamma} * theta)))");
    CorrectionNewtonProblem {
        equations: vec![
            Expr::parse_expression("q"),
            Expr::parse_expression(&format!("{pe} * q - {heat} * ({rate})")),
            Expr::parse_expression("J"),
            Expr::parse_expression(&format!("{pe} * J + ({rate})")),
        ],
        initial_guess,
        state_names: ["theta", "q", "Y", "J"].map(str::to_owned).to_vec(),
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("theta".into(), vec![(0, 0.0)]),
            ("Y".into(), vec![(0, 1.0)]),
            ("q".into(), vec![(1, 0.0)]),
            ("J".into(), vec![(1, 0.0)]),
        ]),
        interval: (0.0, 1.0),
        steps,
    }
}

pub(crate) fn comb02_problem(steps: usize) -> CorrectionNewtonProblem {
    let (pe, da1, da2) = (2.0, 3.0, 12.0);
    let (q1, q2, beta1, beta2, gamma) = (1.0, 2.0, 12.0, 1.0, 0.5);
    let approximate = |x: f64| {
        let a = (-da1 * x).exp();
        let b = da1 / (da2 - da1) * ((-da1 * x).exp() - (-da2 * x).exp());
        let c = 1.0 - a - b;
        let ja = -da1 * a;
        let jb = da1 / (da2 - da1) * (-da1 * (-da1 * x).exp() + da2 * (-da2 * x).exp());
        let jc = -ja - jb;
        let theta = q1 * (1.0 - a) + q2 * c;
        let flux = -(q1 + q2) * ja - q2 * jb;
        [theta, flux, a, ja, b, jb, c, jc]
    };
    let initial_guess = DMatrix::from_fn(8, steps, |state, column| {
        let node = if matches!(state, 0 | 2 | 4 | 6) {
            column + 1
        } else {
            column
        };
        approximate(node as f64 / steps as f64)[state]
    });
    let thermal = format!("(1.0 - 1.0 / (1.0 + {gamma} * theta))");
    let r1 = format!("{da1} * A * exp({beta1} * {thermal})");
    let r2 = format!("{da2} * B * exp({beta2} * {thermal})");
    CorrectionNewtonProblem {
        equations: vec![
            Expr::parse_expression("q"),
            Expr::parse_expression(&format!("{pe} * q - {q1} * ({r1}) - {q2} * ({r2})")),
            Expr::parse_expression("JA"),
            Expr::parse_expression(&format!("{pe} * JA + ({r1})")),
            Expr::parse_expression("JB"),
            Expr::parse_expression(&format!("{pe} * JB - ({r1}) + ({r2})")),
            Expr::parse_expression("JC"),
            Expr::parse_expression(&format!("{pe} * JC - ({r2})")),
        ],
        initial_guess,
        state_names: ["theta", "q", "A", "JA", "B", "JB", "C", "JC"]
            .map(str::to_owned)
            .to_vec(),
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("theta".into(), vec![(0, 0.0)]),
            ("A".into(), vec![(0, 1.0)]),
            ("B".into(), vec![(0, 0.0)]),
            ("C".into(), vec![(0, 0.0)]),
            ("q".into(), vec![(1, 0.0)]),
            ("JA".into(), vec![(1, 0.0)]),
            ("JB".into(), vec![(1, 0.0)]),
            ("JC".into(), vec![(1, 0.0)]),
        ]),
        interval: (0.0, 1.0),
        steps,
    }
}

pub(crate) fn comb02_perturbed_reference_problem(
    steps: usize,
    reference: &[(f64, Vec<f64>)],
) -> CorrectionNewtonProblem {
    let mut full = (0..=steps)
        .map(|node| {
            let x = node as f64 / steps as f64;
            let mut state = reference_at(reference, x);
            let smooth_offset = (std::f64::consts::PI * x).sin();
            state[2] += 0.005 * smooth_offset;
            state[4] += 0.015 * smooth_offset;
            state[6] = 1.0 - state[2] - state[4];
            state[0] = 3.0 - 3.0 * state[2] - 2.0 * state[4];
            state
        })
        .collect::<Vec<_>>();
    let dx = 1.0 / steps as f64;
    for node in 0..=steps {
        let (left, right) = match node {
            0 => (0, 1),
            n if n == steps => (steps - 1, steps),
            n => (n - 1, n + 1),
        };
        let divisor = if node == 0 || node == steps {
            dx
        } else {
            2.0 * dx
        };
        let ja = (full[right][2] - full[left][2]) / divisor;
        let jb = (full[right][4] - full[left][4]) / divisor;
        let jc = (full[right][6] - full[left][6]) / divisor;
        full[node][3] = ja;
        full[node][5] = jb;
        full[node][7] = jc;
        full[node][1] = -3.0 * ja - 2.0 * jb;
    }
    let mut problem = comb02_problem(steps);
    for state in 0..8 {
        for column in 0..steps {
            let node = if matches!(state, 0 | 2 | 4 | 6) {
                column + 1
            } else {
                column
            };
            problem.initial_guess[(state, column)] = full[node][state];
        }
    }
    problem
}

pub(crate) fn options(
    state_names: &[String],
    frontend: CorrectionNewtonSymbolicFrontend,
    layout: CorrectionNewtonMatrixLayout,
) -> CorrectionNewtonOptions {
    let bounds = state_names
        .iter()
        .map(|name| {
            let bound = match name.as_str() {
                "theta" => (-0.5, 4.0),
                "Y" | "A" | "B" | "C" => (-0.2, 1.2),
                _ => (-30.0, 30.0),
            };
            (name.clone(), bound)
        })
        .collect();
    CorrectionNewtonOptions {
        symbolic_frontend: frontend,
        matrix_layout: layout,
        absolute_tolerance: 1.0e-8,
        default_relative_tolerance: 1.0e-8,
        bounds,
        max_iterations: 100,
        max_damping_trials: 12,
        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
        ..CorrectionNewtonOptions::default()
    }
}

fn max_discrete_defect(full: &DMatrix<f64>, steps: usize, rhs: impl Fn(&[f64]) -> Vec<f64>) -> f64 {
    let h = 1.0 / steps as f64;
    let mut max_defect = 0.0_f64;
    for interval in 0..steps {
        let left_state = (0..full.ncols())
            .map(|state| full[(interval, state)])
            .collect::<Vec<_>>();
        let right_state = (0..full.ncols())
            .map(|state| full[(interval + 1, state)])
            .collect::<Vec<_>>();
        let left_values = rhs(&left_state);
        let right_values = rhs(&right_state);
        for state in 0..full.ncols() {
            let defect = full[(interval + 1, state)]
                - full[(interval, state)]
                - 0.5 * h * (left_values[state] + right_values[state]);
            max_defect = max_defect.max(defect.abs());
        }
    }
    max_defect
}

fn comb01_rhs(state: &[f64]) -> Vec<f64> {
    let (theta, q, fuel, flux) = (state[0], state[1], state[2], state[3]);
    let rate = 2.0 * fuel * (5.0 * (1.0 - 1.0 / (1.0 + 0.5 * theta))).exp();
    vec![q, 5.0 * q - 2.0 * rate, flux, 5.0 * flux + rate]
}

fn comb02_rhs(state: &[f64]) -> Vec<f64> {
    let (theta, q, a, ja, b, jb, _c, jc) = (
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
    );
    let thermal = 1.0 - 1.0 / (1.0 + 0.5 * theta);
    let rate1 = 3.0 * a * (12.0 * thermal).exp();
    let rate2 = 12.0 * b * thermal.exp();
    vec![
        q,
        2.0 * q - rate1 - 2.0 * rate2,
        ja,
        2.0 * ja + rate1,
        jb,
        2.0 * jb - rate1 + rate2,
        jc,
        2.0 * jc - rate2,
    ]
}

fn write_report(name: &str, rows: &[CombustionRow], details: &str) {
    let report = format!(
        "# BVP_CorrectionNewton {name} correctness gate\n\n{details}\n\n{}\n",
        tabled::Table::new(rows)
    );
    if let Err(error) = crate::Utils::test_reporting::write_test_report(
        "BVP_CorrectionNewton_combustion",
        name,
        &report,
    ) {
        eprintln!("[BVP_CorrectionNewton combustion report] unable to write report: {error}");
    }
}

fn probe_seed(problem: CorrectionNewtonProblem, seed_name: &str) -> String {
    let solver_options = options(
        &problem.state_names,
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonMatrixLayout::Sparse,
    );
    let mut solver = CorrectionNewtonSolver::from_symbolic(problem, solver_options)
        .unwrap_or_else(|error| panic!("COMB-02 {seed_name} preparation failed: {error}"));
    match solver.solve() {
        Ok(report) => format!(
            "{seed_name}: converged, iterations={}, residual={:?}, telemetry={:?}",
            report.iterations, report.residual_norm, report.telemetry
        ),
        Err(error) => {
            let partial = error.partial_report();
            let summary = partial.map_or_else(
                || "no partial Newton report".to_owned(),
                |report| {
                    format!(
                        "iterations={}, correction={:?}, residual={:?}, telemetry={:?}",
                        report.iterations,
                        report.correction_norm,
                        report.residual_norm,
                        report.telemetry
                    )
                },
            );
            format!("{seed_name}: typed failure `{error}`; {summary}")
        }
    }
}

fn displaced_intermediate_seed(
    steps: usize,
    reference: &[(f64, Vec<f64>)],
) -> CorrectionNewtonProblem {
    let mut problem = comb02_perturbed_reference_problem(steps, reference);
    for column in 0..steps {
        let x = (column + 1) as f64 / steps as f64;
        let offset = 0.15 * (std::f64::consts::PI * x).sin();
        let slope = 0.15 * std::f64::consts::PI * (std::f64::consts::PI * x).cos();
        problem.initial_guess[(4, column)] += offset;
        problem.initial_guess[(6, column)] -= offset;
        problem.initial_guess[(0, column)] -= 2.0 * offset;
        problem.initial_guess[(5, column)] += slope;
        problem.initial_guess[(7, column)] -= slope;
        problem.initial_guess[(1, column)] -= 2.0 * slope;
    }
    problem
}

fn displaced_temperature_seed(
    steps: usize,
    reference: &[(f64, Vec<f64>)],
) -> CorrectionNewtonProblem {
    let mut problem = comb02_perturbed_reference_problem(steps, reference);
    for column in 0..steps {
        let x = (column + 1) as f64 / steps as f64;
        let displaced = reference_at(reference, (x - 0.08).clamp(0.0, 1.0));
        problem.initial_guess[(0, column)] = displaced[0];
        problem.initial_guess[(1, column)] = displaced[1];
    }
    problem
}

fn run_fixture(
    fixture_name: &'static str,
    seed_name: &'static str,
    steps: usize,
    reference: &[(f64, Vec<f64>)],
    problem: CorrectionNewtonProblem,
    frontend: CorrectionNewtonSymbolicFrontend,
    layout: CorrectionNewtonMatrixLayout,
    rhs: impl Fn(&[f64]) -> Vec<f64>,
    invariant: impl Fn(&[f64]) -> f64,
) -> (CombustionRow, DMatrix<f64>) {
    let options = options(&problem.state_names, frontend, layout);
    let mut solver = CorrectionNewtonSolver::from_symbolic(problem.clone(), options)
        .unwrap_or_else(|error| panic!("{fixture_name} preparation failed: {error}"));
    let report = solver
        .solve()
        .unwrap_or_else(|error| panic!("{fixture_name} Newton solve failed: {error}"));
    let full = solver
        .full_grid_values(&report.state)
        .unwrap_or_else(|error| panic!("{fixture_name} full-grid expansion failed: {error}"));
    let mut profile_error = 0.0_f64;
    for node in 0..=steps {
        let expected = reference_at(reference, node as f64 / steps as f64);
        for state in 0..full.ncols() {
            profile_error = profile_error.max((full[(node, state)] - expected[state]).abs());
        }
    }
    let invariant_error = (0..=steps)
        .map(|node| {
            let state = (0..full.ncols())
                .map(|component| full[(node, component)])
                .collect::<Vec<_>>();
            invariant(&state).abs()
        })
        .fold(0.0_f64, f64::max);
    let discrete_defect = max_discrete_defect(&full, steps, &rhs);
    let physical_summary = if full.ncols() == 4 {
        let rate_peak = (0..=steps)
            .map(|node| {
                let state = (0..4)
                    .map(|component| full[(node, component)])
                    .collect::<Vec<_>>();
                let rate = 2.0 * state[2] * (5.0 * (1.0 - 1.0 / (1.0 + 0.5 * state[0]))).exp();
                (rate, node)
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap();
        format!(
            "theta_max={:.4}, fuel_out={:.4}, rate_max={:.4}@x={:.4}",
            (0..=steps)
                .map(|node| full[(node, 0)])
                .fold(0.0_f64, f64::max),
            full[(steps, 2)],
            rate_peak.0,
            rate_peak.1 as f64 / steps as f64
        )
    } else {
        let peak_b = (0..=steps)
            .map(|node| (full[(node, 4)], node))
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap();
        let mut peak_r1 = (0.0, 0usize);
        let mut peak_r2 = (0.0, 0usize);
        for node in 0..=steps {
            let theta = full[(node, 0)];
            let thermal = 1.0 - 1.0 / (1.0 + 0.5 * theta);
            let rate1 = 3.0 * full[(node, 2)] * (12.0 * thermal).exp();
            let rate2 = 12.0 * full[(node, 4)] * thermal.exp();
            if rate1 > peak_r1.0 {
                peak_r1 = (rate1, node);
            }
            if rate2 > peak_r2.0 {
                peak_r2 = (rate2, node);
            }
        }
        format!(
            "B_max={:.4}@x={:.4}, R1_max={:.4}@x={:.4}, R2_max={:.4}@x={:.4}",
            peak_b.0,
            peak_b.1 as f64 / steps as f64,
            peak_r1.0,
            peak_r1.1 as f64 / steps as f64,
            peak_r2.0,
            peak_r2.1 as f64 / steps as f64
        )
    };

    assert!(report.state.iter().all(|value| value.is_finite()));
    assert!(report.residual_norm.is_some_and(f64::is_finite));
    assert!(report.correction_norm.is_some_and(f64::is_finite));
    assert!(
        discrete_defect < 2.0e-5,
        "discrete trapezoid defect={discrete_defect:e}"
    );
    assert!(
        invariant_error < 2.0e-6,
        "first-integral error={invariant_error:e}"
    );
    assert!(report.telemetry.jacobian_evaluations > 0);
    assert!(report.telemetry.factorizations > 0);
    assert!(report.telemetry.linear_solves > 0);

    let row = CombustionRow {
        fixture: fixture_name,
        seed: seed_name,
        steps,
        route: format!("{frontend:?}/{layout:?}"),
        iterations: report.iterations,
        jacobians: report.telemetry.jacobian_evaluations,
        factorizations: report.telemetry.factorizations,
        residual: format!(
            "{:.4e} / {:.4e}",
            report.residual_norm.unwrap(),
            discrete_defect
        ),
        profile_error: format!("{profile_error:.4e}"),
        invariant_error: format!("{invariant_error:.4e}"),
        physical_summary,
    };
    println!(
        "{fixture_name} {steps} {frontend:?}/{layout:?}: iters={}, profile_error={profile_error:.4e}, discrete_defect={discrete_defect:.4e}, invariant_error={invariant_error:.4e}, telemetry={:?}",
        report.iterations, report.telemetry
    );
    (row, full)
}

#[test]
#[ignore = "stiff nonlinear combustion/reference/mesh-convergence correctness gate"]
fn comb01_strong_exothermic_flame_matches_scipy_and_refines() {
    let reference = parse_reference(COMB01_REFERENCE, 4);
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    let mut profiles = Vec::new();
    for steps in [128, 256] {
        for (frontend, layout) in [
            (
                CorrectionNewtonSymbolicFrontend::ExprLegacy,
                CorrectionNewtonMatrixLayout::Sparse,
            ),
            (
                CorrectionNewtonSymbolicFrontend::AtomView,
                CorrectionNewtonMatrixLayout::BandedAuto,
            ),
        ] {
            let (row, full) = run_fixture(
                "COMB-01",
                "smooth analytic",
                steps,
                &reference,
                comb01_problem(steps),
                frontend,
                layout,
                comb01_rhs,
                |state| state[0] + 2.0 * state[2] - 2.0,
            );
            let error = row.profile_error.parse::<f64>().unwrap();
            errors.push((steps, frontend, error));
            profiles.push((steps, frontend, full.clone()));
            rows.push(row);
            for node in 0..full.nrows() {
                assert!((0.0..=2.0).contains(&full[(node, 0)]));
                assert!((0.0..=1.0).contains(&full[(node, 2)]));
                assert!((full[(node, 0)] + 2.0 * full[(node, 2)] - 2.0).abs() < 2.0e-6);
                assert!((full[(node, 1)] + 2.0 * full[(node, 3)]).abs() < 2.0e-6);
            }
            assert!(full[(0, 0)].abs() < 1.0e-12 && (full[(0, 2)] - 1.0).abs() < 1.0e-12);
            assert!(full[(steps, 1)].abs() < 1.0e-12 && full[(steps, 3)].abs() < 1.0e-12);
        }
    }
    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        assert!(
            errors
                .iter()
                .find(|(n, f, _)| *n == 256 && *f == frontend)
                .unwrap()
                .2
                < errors
                    .iter()
                    .find(|(n, f, _)| *n == 128 && *f == frontend)
                    .unwrap()
                    .2,
            "COMB-01 finer-grid profile error must decrease for {frontend:?}: {errors:?}"
        );
    }
    assert!(
        errors
            .iter()
            .filter(|(steps, _, _)| *steps == 256)
            .all(|(_, _, error)| *error < 1.0e-4)
    );
    for steps in [128, 256] {
        let expr = &profiles
            .iter()
            .find(|(n, f, _)| *n == steps && *f == CorrectionNewtonSymbolicFrontend::ExprLegacy)
            .unwrap()
            .2;
        let atom = &profiles
            .iter()
            .find(|(n, f, _)| *n == steps && *f == CorrectionNewtonSymbolicFrontend::AtomView)
            .unwrap()
            .2;
        assert!((expr - atom).amax() < 1.0e-9, "COMB-01 route parity failed");
    }
    let details = "Single-step A -> B model; Pe=5, Da=2, Q=2, beta=5, gamma=0.5. The analytic non-reference seed uses an exponential fuel profile. Independent frozen reference: SciPy solve_bvp 1.18.1, tolerance 2e-9, 2049 samples. Checked theta+Q*Y=Q, q+Q*J=0, BC, bounds, trapezoid defect, profile error, and 128/256 mesh refinement.";
    write_report("comb01_stiff_flame", &rows, details);
}

#[test]
#[ignore = "stiff two-stage combustion/reference/mesh-convergence correctness gate"]
fn comb02_two_stage_flame_preserves_intermediate_and_invariants() {
    let reference = parse_reference(COMB02_REFERENCE, 8);
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    let mut profiles = Vec::new();
    let seed_observations = [
        probe_seed(
            comb02_problem(128),
            "smooth analytic (no reference profile)",
        ),
        probe_seed(
            displaced_intermediate_seed(128, &reference),
            "incorrect intermediate concentration",
        ),
        probe_seed(
            displaced_temperature_seed(128, &reference),
            "displaced temperature front",
        ),
    ];
    for observation in &seed_observations {
        println!("COMB-02 seed probe: {observation}");
    }
    for steps in [128, 256] {
        for (frontend, layout) in [
            (
                CorrectionNewtonSymbolicFrontend::ExprLegacy,
                CorrectionNewtonMatrixLayout::Sparse,
            ),
            (
                CorrectionNewtonSymbolicFrontend::AtomView,
                CorrectionNewtonMatrixLayout::BandedAuto,
            ),
        ] {
            let (row, full) = run_fixture(
                "COMB-02",
                "smooth perturbed reference",
                steps,
                &reference,
                comb02_perturbed_reference_problem(steps, &reference),
                frontend,
                layout,
                comb02_rhs,
                |state| state[0] + 3.0 * state[2] + 2.0 * state[4] - 3.0,
            );
            errors.push((steps, frontend, row.profile_error.parse::<f64>().unwrap()));
            profiles.push((steps, frontend, full.clone()));
            rows.push(row);
            for node in 0..full.nrows() {
                let a = full[(node, 2)];
                let b = full[(node, 4)];
                let c = full[(node, 6)];
                assert!(a >= -1.0e-7 && b >= -1.0e-7 && c >= -1.0e-7);
                assert!((a + b + c - 1.0).abs() < 2.0e-6);
                assert!(
                    (full[(node, 1)] + 3.0 * full[(node, 3)] + 2.0 * full[(node, 5)]).abs()
                        < 2.0e-5
                );
                assert!((0.0..=4.0).contains(&full[(node, 0)]));
            }
            assert!(
                full[(0, 0)].abs() < 1.0e-12
                    && (full[(0, 2)] - 1.0).abs() < 1.0e-12
                    && full[(0, 4)].abs() < 1.0e-12
                    && full[(0, 6)].abs() < 1.0e-12
            );
            assert!(
                full[(steps, 1)].abs() < 1.0e-12
                    && full[(steps, 3)].abs() < 1.0e-12
                    && full[(steps, 5)].abs() < 1.0e-12
                    && full[(steps, 7)].abs() < 1.0e-12
            );
        }
    }
    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        let fine_error = errors
            .iter()
            .find(|(n, f, _)| *n == 256 && *f == frontend)
            .unwrap()
            .2;
        let coarse_error = errors
            .iter()
            .find(|(n, f, _)| *n == 128 && *f == frontend)
            .unwrap()
            .2;
        assert!(fine_error < coarse_error, "COMB-02 errors={errors:?}");
        let full = &profiles
            .iter()
            .find(|(n, f, _)| *n == 256 && *f == frontend)
            .unwrap()
            .2;
        let steps = full.nrows() - 1;
        let mut max_b = (0.0, 0usize);
        let mut max_r1 = (0.0, 0usize);
        let mut max_r2 = (0.0, 0usize);
        for node in 0..=steps {
            let theta = full[(node, 0)];
            let thermal = 1.0 - 1.0 / (1.0 + 0.5 * theta);
            let r1 = 3.0 * full[(node, 2)] * (12.0 * thermal).exp();
            let r2 = 12.0 * full[(node, 4)] * thermal.exp();
            if full[(node, 4)] > max_b.0 {
                max_b = (full[(node, 4)], node);
            }
            if r1 > max_r1.0 {
                max_r1 = (r1, node);
            }
            if r2 > max_r2.0 {
                max_r2 = (r2, node);
            }
        }
        let peak_separation = max_r1.1.abs_diff(max_r2.1) as f64 / steps as f64;
        assert!(max_b.0 > 0.4, "intermediate B peak too small: {max_b:?}");
        assert!(
            peak_separation > 0.03,
            "reaction peaks not spatially distinct: {max_r1:?}, {max_r2:?}"
        );
        println!(
            "COMB-02 {frontend:?}: Bmax={}, x_R1={}, x_R2={}",
            max_b.0,
            max_r1.1 as f64 / steps as f64,
            max_r2.1 as f64 / steps as f64
        );
    }
    assert!(
        errors
            .iter()
            .filter(|(steps, _, _)| *steps == 256)
            .all(|(_, _, error)| *error < 2.0e-3)
    );
    for steps in [128, 256] {
        let expr = &profiles
            .iter()
            .find(|(n, f, _)| *n == steps && *f == CorrectionNewtonSymbolicFrontend::ExprLegacy)
            .unwrap()
            .2;
        let atom = &profiles
            .iter()
            .find(|(n, f, _)| *n == steps && *f == CorrectionNewtonSymbolicFrontend::AtomView)
            .unwrap()
            .2;
        assert!((expr - atom).amax() < 1.0e-9, "COMB-02 route parity failed");
    }
    let details = format!(
        "Two-stage A -> B -> C model; Pe=2, Da1=3, Da2=12, Q1=1, Q2=2, beta1=12, beta2=1, gamma=0.5. The convergent seed is a smooth, perturbed profile, not the frozen reference itself. The exponential smooth seed and deliberately displaced seeds are attempted and retained below as diagnostics; arbitrary-seed failure is not treated as solver failure. Independent frozen reference: SciPy solve_bvp 1.18.1, tolerance 2e-9, 2049 samples. Checked mass/energy invariants, BC, species admissibility, trapezoid defect, intermediate maximum, separated reaction-rate peaks, profile error, and 128/256 mesh refinement.\n\nSeed outcomes:\n{}",
        seed_observations
            .iter()
            .map(|line| format!("- {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    write_report("comb02_two_stage_flame", &rows, &details);
}
