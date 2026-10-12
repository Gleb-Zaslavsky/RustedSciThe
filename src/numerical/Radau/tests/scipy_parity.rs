//! Matched-workload counter semantics and numerical parity against vendored SciPy.

use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;
use std::process::Command;
use tabled::{Table, Tabled};

use super::super::api::{
    RadauConfig, RadauErrorKind, RadauNativeSolver, RadauOutputPolicy, RadauSolution,
    RadauTelemetryMode,
};

#[derive(Debug, Tabled)]
struct ParityRow {
    workload: String,
    scaled_final_error: String,
    scaled_trajectory_error: String,
    rust_residual_calls: u64,
    scipy_nfev: u64,
    rust_newton_iterations: u64,
    scipy_newton_iterations: u64,
    rust_error_estimates: u64,
    scipy_error_estimates: u64,
    scipy_rhs_callbacks: u64,
    scipy_fd_probes: u64,
    rust_jacobian_calls: u64,
    scipy_njev: u64,
    rust_factor_pairs: u64,
    rust_lu_equiv: u64,
    scipy_nlu: u64,
    rust_linear_solves: u64,
    rust_real_solves: u64,
    rust_complex_solves: u64,
    scipy_solve_calls: u64,
    rust_accepted: u64,
    scipy_accepted: u64,
    rust_rejected: u64,
    scipy_rejected: u64,
    rust_solve_ms: String,
    scipy_solve_ms: String,
}

#[derive(Debug, Tabled)]
struct AcceptedStepTraceRow {
    implementation: String,
    attempt: usize,
    h_abs: String,
    error_norm: String,
    newton_iterations: usize,
    accepted: bool,
    next_h_abs: String,
    jacobian_refreshed: bool,
}

#[derive(Debug, Tabled)]
struct CoreTimingRow {
    workload: String,
    rust_core_median_ms: String,
    rust_core_p10_ms: String,
    rust_core_p90_ms: String,
    scipy_wall_median_ms: String,
    scipy_wall_min_ms: String,
    scipy_wall_p90_ms: String,
    scipy_over_rust: String,
}

#[derive(Debug, Tabled)]
struct ExtendedParityRow {
    scenario: String,
    rust_status: String,
    scipy_status: String,
    rust_initial_step: String,
    scipy_initial_step: String,
    scaled_final_error: String,
    scaled_sample_error: String,
    rust_accepted: String,
    scipy_accepted: String,
    rust_rejected: String,
    scipy_rejected: String,
    rust_jacobian_calls: String,
    scipy_jacobian_calls: String,
}

fn matched_config(t_bound: f64) -> RadauConfig {
    let sample_times = [0.25, 0.5, 0.75]
        .map(|fraction| fraction * t_bound)
        .to_vec();
    RadauConfig {
        t_bound,
        first_step: Some(t_bound * 0.1),
        max_step: t_bound * 0.25,
        rtol: 1.0e-7,
        atol: 1.0e-10,
        max_steps: 2_000,
        max_newton_iterations: 6,
        max_retries: 24,
        telemetry: RadauTelemetryMode::Timings,
        output: RadauOutputPolicy::Sampled(sample_times),
        ..RadauConfig::default()
    }
}

fn core_benchmark_config(t_bound: f64) -> RadauConfig {
    let mut config = matched_config(t_bound);
    config.telemetry = RadauTelemetryMode::Off;
    config
}

fn timing_quantile(samples: &[f64], fraction: f64) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let index = ((ordered.len() - 1) as f64 * fraction).round() as usize;
    ordered[index]
}

fn measure_prepared_core(
    solver: &mut RadauNativeSolver,
    y0: &[f64],
    warmups: usize,
    repeats: usize,
) -> Vec<f64> {
    for _ in 0..warmups {
        let solution = solver.solve(y0).expect("Rust Radau warmup solve");
        assert!(solution.y.iter().all(|value| value.is_finite()));
    }
    (0..repeats)
        .map(|_| {
            let started = std::time::Instant::now();
            let solution = solver.solve(y0).expect("Rust Radau timed solve");
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            assert!(solution.y.iter().all(|value| value.is_finite()));
            elapsed_ms
        })
        .collect()
}

fn run_python_core_benchmark(repeats: usize, warmups: usize) -> BTreeMap<String, Vec<String>> {
    let python = std::env::var_os("IVP_SCIPY_PYTHON").unwrap_or_else(|| "python".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("numerical");
    let output = Command::new(python)
        .args([
            "-m",
            "ivp_scipy_reference.radau_parity",
            "--format",
            "tsv",
            "--benchmark-repeats",
            &repeats.to_string(),
            "--benchmark-warmups",
            &warmups.to_string(),
        ])
        .current_dir(&root)
        .output()
        .unwrap_or_else(|error| panic!("could not start Python benchmark: {error}"));
    assert!(
        output.status.success(),
        "Python Radau benchmark failed. Set IVP_SCIPY_PYTHON to the pinned interpreter.\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("benchmark TSV must be UTF-8");
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let headers: Vec<_> = lines
        .next()
        .expect("benchmark TSV header")
        .split('\t')
        .collect();
    let indices: BTreeMap<_, _> = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, index))
        .collect();
    lines
        .map(|line| {
            let columns: Vec<_> = line.split('\t').map(str::to_owned).collect();
            (columns[indices["workload"]].clone(), columns)
        })
        .collect()
}

fn run_python_extended() -> BTreeMap<String, BTreeMap<String, String>> {
    let python = std::env::var_os("IVP_SCIPY_PYTHON").unwrap_or_else(|| "python".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("numerical");
    let output = Command::new(python)
        .args([
            "-m",
            "ivp_scipy_reference.radau_parity",
            "--extended",
            "--format",
            "tsv",
        ])
        .current_dir(&root)
        .output()
        .unwrap_or_else(|error| panic!("could not start extended Python parity: {error}"));
    assert!(
        output.status.success(),
        "Extended Python Radau parity failed. Set IVP_SCIPY_PYTHON to the pinned interpreter.\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("extended TSV must be UTF-8");
    let mut lines = text.lines();
    let headers: Vec<_> = lines
        .next()
        .expect("extended TSV header")
        .split('\t')
        .map(str::to_owned)
        .collect();
    lines
        .map(|line| {
            let values: Vec<_> = line.split('\t').map(str::to_owned).collect();
            let row: BTreeMap<_, _> = headers.iter().cloned().zip(values).collect();
            (row["scenario"].clone(), row)
        })
        .collect()
}

fn extended_config(
    t0: f64,
    t_bound: f64,
    first_step: Option<f64>,
    max_step: f64,
    rtol: f64,
) -> RadauConfig {
    let mut config = matched_config(t_bound.abs().max(t0.abs()));
    config.t0 = t0;
    config.t_bound = t_bound;
    config.first_step = first_step;
    config.max_step = max_step;
    config.rtol = rtol;
    config.telemetry = RadauTelemetryMode::Timings;
    config.output = RadauOutputPolicy::Sampled(
        [0.25, 0.5, 0.75]
            .map(|fraction| t0 + fraction * (t_bound - t0))
            .to_vec(),
    );
    config
}

fn parse_csv_f64(value: &str) -> Vec<f64> {
    if value.is_empty() {
        return Vec::new();
    }
    value
        .split(',')
        .map(|item| item.parse().expect("finite numeric TSV value"))
        .collect()
}

fn scaled_max_error(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs() / (atol + rtol * expected.abs()))
        .fold(0.0, f64::max)
}

fn compare_extended_success<R, J>(
    python: &BTreeMap<String, BTreeMap<String, String>>,
    name: &str,
    t0: f64,
    t_bound: f64,
    max_step: f64,
    first_step: Option<f64>,
    rtol: f64,
    atol: f64,
    y0: &[f64],
    residual: R,
    jacobian: J,
) -> ExtendedParityRow
where
    R: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
    J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
{
    let py = &python[name];
    assert_eq!(py["status"], "finished", "SciPy {name}");
    let sample_times = [0.25, 0.5, 0.75].map(|fraction| t0 + fraction * (t_bound - t0));
    let config = extended_config(t0, t_bound, first_step, max_step, rtol);
    let mut solver = RadauNativeSolver::prepare(config, residual, Some(jacobian))
        .unwrap_or_else(|error| panic!("Rust {name} preparation failed: {error}"));
    let solution = solver
        .solve(y0)
        .unwrap_or_else(|error| panic!("Rust {name} solve failed: {error}"));
    let py_y = parse_csv_f64(&py["y"]);
    let final_error = scaled_max_error(&solution.y, &py_y, atol, rtol);
    assert!(
        final_error < 20.0,
        "{name} scaled final error={final_error:e}"
    );
    let py_samples: Vec<Vec<f64>> = py["samples"].split(';').map(parse_csv_f64).collect();
    let rust_samples = solution
        .sample_many(&sample_times)
        .unwrap_or_else(|error| panic!("Rust {name} output sampling failed: {error}"));
    let sample_error = scaled_max_error(
        &rust_samples,
        &py_samples.into_iter().flatten().collect::<Vec<_>>(),
        atol,
        rtol,
    );
    assert!(
        sample_error < 20.0,
        "{name} scaled dense error={sample_error:e}"
    );
    let rust_initial = solution
        .telemetry()
        .initial_h_abs
        .expect("timed Rust solve records selected first step");
    let scipy_initial: f64 = py["initial_h_abs"].parse().expect("SciPy first trial step");
    let initial_step_relative_difference =
        (rust_initial - scipy_initial).abs() / scipy_initial.abs().max(f64::MIN_POSITIVE);
    assert!(
        initial_step_relative_difference < 1.0e-10,
        "{name} initial-step mismatch: Rust={rust_initial:.17e}, SciPy={scipy_initial:.17e}"
    );
    ExtendedParityRow {
        scenario: name.to_owned(),
        rust_status: "finished".to_owned(),
        scipy_status: py["status"].clone(),
        rust_initial_step: format!("{rust_initial:.6e}"),
        scipy_initial_step: format!("{scipy_initial:.6e}"),
        scaled_final_error: format!("{final_error:.3e}"),
        scaled_sample_error: format!("{sample_error:.3e}"),
        rust_accepted: solution.accepted_steps.to_string(),
        scipy_accepted: py["accepted_steps"].clone(),
        rust_rejected: solution.rejected_steps.to_string(),
        scipy_rejected: py["rejected_steps"].clone(),
        rust_jacobian_calls: solution
            .telemetry()
            .counters
            .get("jacobian_calls")
            .copied()
            .unwrap_or_default()
            .to_string(),
        scipy_jacobian_calls: py["njev"].clone(),
    }
}

fn run_python_reference() -> (String, BTreeMap<String, Vec<String>>) {
    let python = std::env::var_os("IVP_SCIPY_PYTHON").unwrap_or_else(|| "python".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("numerical");
    let output = Command::new(python)
        .args(["-m", "ivp_scipy_reference.radau_parity", "--format", "tsv"])
        .current_dir(&root)
        .output()
        .unwrap_or_else(|error| panic!("could not start Python reference: {error}"));
    assert!(
        output.status.success(),
        "Python Radau reference failed. Set IVP_SCIPY_PYTHON to the interpreter with NumPy/SciPy installed.\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("reference TSV must be UTF-8");
    let environment = text
        .lines()
        .find(|line| line.starts_with('#'))
        .unwrap_or("# Python/SciPy environment metadata missing")
        .replace('\t', " ");
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let headers: Vec<_> = lines
        .next()
        .expect("reference TSV header")
        .split('\t')
        .collect();
    let indices: BTreeMap<_, _> = headers
        .iter()
        .enumerate()
        .map(|(index, header)| (*header, index))
        .collect();
    let cases = lines
        .map(|line| {
            let columns: Vec<_> = line.split('\t').map(str::to_owned).collect();
            let name = columns[indices["workload"]].clone();
            (name, columns)
        })
        .collect();
    (environment, cases)
}

fn field<'a>(columns: &'a [String], name: &str) -> &'a str {
    // Keep the small TSV contract explicit so schema drift fails loudly.
    let headers = [
        "workload",
        "t",
        "y",
        "samples",
        "status",
        "nfev",
        "njev",
        "nlu",
        "rhs_calls",
        "rhs_state_evaluations",
        "fd_rhs_calls",
        "fd_state_probes",
        "jacobian_calls",
        "jacobian_refreshes",
        "lu_calls",
        "factor_pair_attempts",
        "linear_solve_calls",
        "step_attempts",
        "newton_iterations",
        "accepted_steps",
        "rejected_steps",
        "error_estimates",
        "controller_trace",
        "total_solve_ms",
    ];
    columns[headers
        .iter()
        .position(|item| *item == name)
        .expect("known TSV field")]
    .as_str()
}

fn compare_case(
    name: &str,
    y0: &[f64],
    t_bound: f64,
    residual: impl Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
    jacobian: impl Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    python: &[String],
) -> (ParityRow, RadauSolution) {
    let mut solver = RadauNativeSolver::prepare(matched_config(t_bound), residual, Some(jacobian))
        .unwrap_or_else(|error| panic!("Rust {name} prepare failed: {error}"));
    let started = std::time::Instant::now();
    let solution = solver
        .solve(y0)
        .unwrap_or_else(|error| panic!("Rust {name} solve failed: {error}"));
    let rust_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let py_y: Vec<f64> = field(python, "y")
        .split(',')
        .map(|value| value.parse().expect("Python state is numeric"))
        .collect();
    let py_samples: Vec<Vec<f64>> = field(python, "samples")
        .split(';')
        .map(|sample| {
            sample
                .split(',')
                .map(|value| value.parse().expect("Python sample is numeric"))
                .collect()
        })
        .collect();
    assert_eq!(field(python, "status"), "finished");
    let scipy_t: f64 = field(python, "t")
        .parse()
        .expect("Python final time is numeric");
    assert!((solution.t - t_bound).abs() <= f64::EPSILON * t_bound.abs().max(1.0));
    assert!((scipy_t - t_bound).abs() <= f64::EPSILON * t_bound.abs().max(1.0));
    assert_eq!(solution.y.len(), py_y.len());
    let scaled_error = solution
        .y
        .iter()
        .zip(&py_y)
        .map(|(rust, scipy)| (rust - scipy).abs() / (1.0e-10 + scipy.abs() * 1.0e-7))
        .fold(0.0, f64::max);
    assert!(
        scaled_error < 20.0,
        "{name} scaled endpoint difference={scaled_error:e}"
    );
    let sample_times = [0.25, 0.5, 0.75].map(|fraction| fraction * t_bound);
    let rust_samples = solution
        .sample_many(&sample_times)
        .unwrap_or_else(|error| panic!("Rust {name} dense-output sampling failed: {error}"));
    assert_eq!(rust_samples.len(), py_samples.len() * py_y.len());
    let scaled_trajectory_error = rust_samples
        .chunks_exact(py_y.len())
        .zip(&py_samples)
        .flat_map(|(rust, scipy)| {
            rust.iter()
                .zip(scipy)
                .map(|(rust, scipy)| (rust - scipy).abs() / (1.0e-10 + scipy.abs() * 1.0e-7))
        })
        .fold(0.0, f64::max);
    assert!(
        scaled_trajectory_error < 20.0,
        "{name} scaled dense-trajectory difference={scaled_trajectory_error:e}"
    );

    let counters = &solution.telemetry().counters;
    let scipy_nfev = field(python, "nfev").parse().unwrap();
    let scipy_rhs_callbacks = field(python, "rhs_calls").parse().unwrap();
    let scipy_fd_probes = field(python, "fd_state_probes").parse().unwrap();
    let scipy_njev = field(python, "njev").parse().unwrap();
    let scipy_nlu = field(python, "nlu").parse().unwrap();
    let scipy_factor_pairs = field(python, "factor_pair_attempts")
        .parse::<u64>()
        .unwrap();
    let scipy_solves = field(python, "linear_solve_calls").parse().unwrap();
    let scipy_accepted = field(python, "accepted_steps").parse().unwrap();
    let scipy_rejected = field(python, "rejected_steps").parse().unwrap();
    let rust_lu_equiv = counters["factorizations"] * 2;

    // Executable counter contracts: SciPy counts each shifted LU separately;
    // its pair-attempt and call-level counters must preserve that relation.
    assert_eq!(
        field(python, "lu_calls").parse::<u64>().unwrap(),
        scipy_nlu,
        "SciPy nlu must count each actual LU call"
    );
    assert_eq!(
        scipy_nlu,
        scipy_factor_pairs * 2,
        "each Radau shifted-factor pair consists of one real and one complex LU"
    );
    assert_eq!(
        counters["linear_solves"],
        counters["real_solves"] + counters["complex_solves"],
        "Rust total solve count is the sum of real and complex shifted solves"
    );

    let row = ParityRow {
        workload: name.to_owned(),
        scaled_final_error: format!("{scaled_error:.3e}"),
        scaled_trajectory_error: format!("{scaled_trajectory_error:.3e}"),
        rust_residual_calls: counters["residual_calls"],
        scipy_nfev,
        rust_newton_iterations: counters["newton_iterations"],
        scipy_newton_iterations: field(python, "newton_iterations").parse().unwrap(),
        rust_error_estimates: counters["error_estimates"],
        scipy_error_estimates: field(python, "error_estimates").parse().unwrap(),
        scipy_rhs_callbacks,
        scipy_fd_probes,
        rust_jacobian_calls: counters["jacobian_calls"],
        scipy_njev,
        rust_factor_pairs: counters["factorizations"],
        rust_lu_equiv,
        scipy_nlu,
        rust_linear_solves: counters["linear_solves"],
        rust_real_solves: counters["real_solves"],
        rust_complex_solves: counters["complex_solves"],
        scipy_solve_calls: scipy_solves,
        rust_accepted: solution.accepted_steps as u64,
        scipy_accepted,
        rust_rejected: solution.rejected_steps as u64,
        scipy_rejected,
        rust_solve_ms: format!("{rust_ms:.3}"),
        scipy_solve_ms: field(python, "total_solve_ms").to_owned(),
    };
    (row, solution)
}

#[test]
#[ignore = "cross-language story; requires IVP_SCIPY_PYTHON with NumPy and SciPy"]
fn radau_rust_vs_scipy_matched_workload_and_counter_semantics_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_SciPy_Parity",
        "numerical::Radau::tests::scipy_parity::radau_rust_vs_scipy_matched_workload_and_counter_semantics_story",
    );
    let (environment, py_cases) = run_python_reference();
    let mut rows = Vec::new();

    let (row, _) = compare_case(
        "stiff-scalar",
        &[1.0],
        0.01,
        |_t, y| DVector::from_vec(vec![-1000.0 * (y[0] - _t.cos()) - _t.sin()]),
        |_t, _y| DMatrix::from_element(1, 1, -1000.0),
        &py_cases["stiff-scalar"],
    );
    rows.push(row);

    let (row, robertson_solution) = compare_case(
        "robertson",
        &[1.0, 0.0, 0.0],
        0.002,
        |_t, y| {
            DVector::from_vec(vec![
                -0.04 * y[0] + 1.0e4 * y[1] * y[2],
                0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
                3.0e7 * y[1] * y[1],
            ])
        },
        |_t, y| {
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    -0.04,
                    1.0e4 * y[2],
                    1.0e4 * y[1],
                    0.04,
                    -1.0e4 * y[2] - 6.0e7 * y[1],
                    -1.0e4 * y[1],
                    0.0,
                    6.0e7 * y[1],
                    0.0,
                ],
            )
        },
        &py_cases["robertson"],
    );
    rows.push(row);

    let (row, _) = compare_case(
        "combustion-like",
        &[1.0, 0.0, 300.0],
        0.002,
        |_t, y| {
            let rate = 1.0e7 * (-5.0e4 / (8.314 * y[2])).exp() * y[0] * y[0];
            DVector::from_vec(vec![-rate, 0.5 * rate, 5.0e2 * rate - 0.5 * (y[2] - 300.0)])
        },
        |_t, y| {
            let q = 1.0e7 * (-5.0e4 / (8.314 * y[2])).exp();
            let dr_da = 2.0 * q * y[0];
            let dr_dt = q * y[0] * y[0] * 5.0e4 / (8.314 * y[2] * y[2]);
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    -dr_da,
                    0.0,
                    -dr_dt,
                    0.5 * dr_da,
                    0.0,
                    0.5 * dr_dt,
                    5.0e2 * dr_da,
                    0.0,
                    5.0e2 * dr_dt - 0.5,
                ],
            )
        },
        &py_cases["combustion-like"],
    );
    rows.push(row);

    let rust_trace = robertson_solution
        .telemetry()
        .adaptive_steps
        .iter()
        .map(|step| {
            format!(
                "{:.17e}:{:.17e}:{:.17e}:{}:{}:{}",
                step.h_abs,
                step.error_norm,
                step.next_h_abs,
                step.newton_iterations,
                u8::from(step.accepted),
                u8::from(step.jacobian_refreshed)
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    let python_trace = field(&py_cases["robertson"], "controller_trace");
    let mut trace_rows = Vec::new();
    for (attempt, trace) in rust_trace.split(';').enumerate() {
        let values: Vec<_> = trace.split(':').collect();
        trace_rows.push(AcceptedStepTraceRow {
            implementation: "Rust".to_owned(),
            attempt: attempt + 1,
            h_abs: values[0].to_owned(),
            error_norm: values[1].to_owned(),
            next_h_abs: values[2].to_owned(),
            newton_iterations: values[3].parse().unwrap(),
            accepted: values[4] == "1",
            jacobian_refreshed: values[5] == "1",
        });
    }
    for (attempt, trace) in python_trace.split(';').enumerate() {
        let values: Vec<_> = trace.split(':').collect();
        trace_rows.push(AcceptedStepTraceRow {
            implementation: "SciPy".to_owned(),
            attempt: attempt + 1,
            h_abs: values[0].to_owned(),
            error_norm: values[1].to_owned(),
            next_h_abs: values[2].to_owned(),
            newton_iterations: values[3].parse().unwrap(),
            accepted: values[4] == "1",
            jacobian_refreshed: values[5] == "1",
        });
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "Reference environment: {environment}\nCounter mapping: Rust residual_calls vs SciPy nfev (analytic-Jacobian fixtures); Rust jacobian_calls vs SciPy njev; 2 * Rust factorizations vs SciPy nlu; Rust linear_solves vs instrumented SciPy solve_lu calls.\nRust/SciPy timings use different runtime scopes and are diagnostic only, not a portable performance comparison.\n{}",
        Table::new(rows)
    ));
    crate::Utils::test_reporting::capture_test_block(format!(
        "Robertson controller-attempt trace (compare sequence rows by attempt):\n{}",
        Table::new(trace_rows)
    ));
}

#[test]
#[ignore = "release-only Python wall-clock vs prepared Rust numerical-core benchmark"]
fn radau_scipy_release_core_timing_benchmark_story() {
    const WARMUPS: usize = 3;
    const REPEATS: usize = 21;
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_SciPy_Parity",
        "numerical::Radau::tests::scipy_parity::radau_scipy_release_core_timing_benchmark_story",
    );
    let python = run_python_core_benchmark(REPEATS, WARMUPS);
    let python_field = |workload: &str, field_name: &str| -> f64 {
        let columns = &python[workload];
        let headers = [
            "workload",
            "warmups",
            "repeats",
            "wall_min_ms",
            "wall_median_ms",
            "wall_p90_ms",
        ];
        columns[headers
            .iter()
            .position(|item| *item == field_name)
            .expect("known benchmark field")]
        .parse()
        .expect("Python timing is numeric")
    };
    let mut rows = Vec::new();

    macro_rules! measure_workload {
        ($name:literal, $bound:expr, $y0:expr, $residual:expr, $jacobian:expr) => {{
            let mut solver = RadauNativeSolver::prepare(
                core_benchmark_config($bound),
                $residual,
                Some($jacobian),
            )
            .unwrap_or_else(|error| panic!("Rust {} prepare failed: {error}", $name));
            let rust_samples = measure_prepared_core(&mut solver, $y0, WARMUPS, REPEATS);
            let rust_median = timing_quantile(&rust_samples, 0.5);
            let python_median = python_field($name, "wall_median_ms");
            rows.push(CoreTimingRow {
                workload: $name.to_owned(),
                rust_core_median_ms: format!("{rust_median:.4}"),
                rust_core_p10_ms: format!("{:.4}", timing_quantile(&rust_samples, 0.1)),
                rust_core_p90_ms: format!("{:.4}", timing_quantile(&rust_samples, 0.9)),
                scipy_wall_median_ms: format!("{python_median:.4}"),
                scipy_wall_min_ms: format!("{:.4}", python_field($name, "wall_min_ms")),
                scipy_wall_p90_ms: format!("{:.4}", python_field($name, "wall_p90_ms")),
                scipy_over_rust: format!("{:.2}x", python_median / rust_median),
            });
        }};
    }

    measure_workload!(
        "stiff-scalar",
        0.01,
        &[1.0],
        |_t: f64, y: &DVector<f64>| DVector::from_vec(vec![-1000.0 * (y[0] - _t.cos()) - _t.sin()]),
        |_t: f64, _y: &DVector<f64>| DMatrix::from_element(1, 1, -1000.0)
    );
    measure_workload!(
        "robertson",
        0.002,
        &[1.0, 0.0, 0.0],
        |_t: f64, y: &DVector<f64>| {
            DVector::from_vec(vec![
                -0.04 * y[0] + 1.0e4 * y[1] * y[2],
                0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
                3.0e7 * y[1] * y[1],
            ])
        },
        |_t: f64, y: &DVector<f64>| {
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    -0.04,
                    1.0e4 * y[2],
                    1.0e4 * y[1],
                    0.04,
                    -1.0e4 * y[2] - 6.0e7 * y[1],
                    -1.0e4 * y[1],
                    0.0,
                    6.0e7 * y[1],
                    0.0,
                ],
            )
        }
    );
    measure_workload!(
        "combustion-like",
        0.002,
        &[1.0, 0.0, 300.0],
        |_t: f64, y: &DVector<f64>| {
            let rate = 1.0e7 * (-5.0e4 / (8.314 * y[2])).exp() * y[0] * y[0];
            DVector::from_vec(vec![-rate, 0.5 * rate, 5.0e2 * rate - 0.5 * (y[2] - 300.0)])
        },
        |_t: f64, y: &DVector<f64>| {
            let q = 1.0e7 * (-5.0e4 / (8.314 * y[2])).exp();
            let dr_da = 2.0 * q * y[0];
            let dr_dt = q * y[0] * y[0] * 5.0e4 / (8.314 * y[2] * y[2]);
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    -dr_da,
                    0.0,
                    -dr_dt,
                    0.5 * dr_da,
                    0.0,
                    0.5 * dr_dt,
                    5.0e2 * dr_da,
                    0.0,
                    5.0e2 * dr_dt - 0.5,
                ],
            )
        }
    );

    crate::Utils::test_reporting::capture_test_block(format!(
        "Release timing only; Python scope is Radau construction + integration + the same three output samples with telemetry disabled. Rust scope is solve() on a pre-prepared native callback model, excluding symbolic preparation. Both use the same equations, analytic Jacobian, interval, tolerances, first/max steps and sampled times. warmups={WARMUPS}, measured_repeats={REPEATS}; rows report median and p10/p90. This is a practical numerical-core comparison, not a symbolic full-lifecycle comparison.\n{}",
        Table::new(rows)
    ));
}

#[test]
#[ignore = "cross-language extended parity; requires IVP_SCIPY_PYTHON with pinned NumPy/SciPy"]
fn radau_scipy_extended_direction_initial_step_and_failure_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_SciPy_Parity",
        "numerical::Radau::tests::scipy_parity::radau_scipy_extended_direction_initial_step_and_failure_story",
    );
    let python = run_python_extended();
    let mut rows = Vec::new();

    rows.push(compare_extended_success(
        &python,
        "backward-auto-step",
        0.01,
        0.0,
        0.0025,
        None,
        1.0e-7,
        1.0e-10,
        &[0.01_f64.cos()],
        |t, y| DVector::from_vec(vec![-1000.0 * (y[0] - t.cos()) - t.sin()]),
        |_, _| DMatrix::from_element(1, 1, -1000.0),
    ));

    rows.push(compare_extended_success(
        &python,
        "hires-auto-step",
        0.0,
        1.0,
        0.25,
        None,
        1.0e-7,
        1.0e-10,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057],
        |_, y| {
            DVector::from_vec(vec![
                -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] - 7.0e-4,
                1.71 * y[0] - 8.75 * y[1],
                -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
                8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
                -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
                -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
                280.0 * y[5] * y[7] - 1.81 * y[6],
                -1.81 * y[6] + 1.81 * y[7],
            ])
        },
        |_, y| {
            DMatrix::from_row_slice(
                8,
                8,
                &[
                    -1.71,
                    0.43,
                    8.32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.71,
                    -8.75,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -10.03,
                    0.43,
                    0.035,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    8.32,
                    1.71,
                    -1.12,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.745,
                    0.43,
                    0.43,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.69,
                    1.71,
                    -280.0 * y[7] - 0.43,
                    0.69,
                    -280.0 * y[5],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    280.0 * y[7],
                    -1.81,
                    280.0 * y[5],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.81,
                    1.81,
                ],
            )
        },
    ));

    rows.push(compare_extended_success(
        &python,
        "robertson-nearby-tolerance",
        0.0,
        0.002,
        0.0005,
        Some(0.0002),
        8.0e-8,
        1.0e-10,
        &[1.0, 0.0, 0.0],
        |_, y| {
            DVector::from_vec(vec![
                -0.04 * y[0] + 1.0e4 * y[1] * y[2],
                0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
                3.0e7 * y[1] * y[1],
            ])
        },
        |_, y| {
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    -0.04,
                    1.0e4 * y[2],
                    1.0e4 * y[1],
                    0.04,
                    -1.0e4 * y[2] - 6.0e7 * y[1],
                    -1.0e4 * y[1],
                    0.0,
                    6.0e7 * y[1],
                    0.0,
                ],
            )
        },
    ));

    // SciPy's validate_first_step rejects an explicit step larger than the
    // complete interval; Rust must report the same invalid configuration.
    let oversize_config = extended_config(0.0, 0.1, Some(0.2), 0.1, 1.0e-7);
    let rust_oversize = match RadauNativeSolver::prepare(
        oversize_config,
        |_, y: &DVector<f64>| -y,
        Some(|_: f64, _: &DVector<f64>| DMatrix::from_element(1, 1, -1.0)),
    ) {
        Ok(_) => panic!("Rust accepted a first step larger than the interval"),
        Err(error) => error,
    };
    assert_eq!(rust_oversize.kind(), RadauErrorKind::Configuration);
    let scipy_oversize = &python["oversize-first-step"];
    assert_eq!(scipy_oversize["status"], "configuration_error");
    assert!(scipy_oversize["message"].contains("exceeds bounds"));
    rows.push(ExtendedParityRow {
        scenario: "oversize-first-step".to_owned(),
        rust_status: format!("{:?}", rust_oversize.kind()),
        scipy_status: scipy_oversize["status"].clone(),
        rust_initial_step: "rejected before solve".to_owned(),
        scipy_initial_step: "rejected before solve".to_owned(),
        scaled_final_error: "n/a".to_owned(),
        scaled_sample_error: "n/a".to_owned(),
        rust_accepted: "0".to_owned(),
        scipy_accepted: "0".to_owned(),
        rust_rejected: "0".to_owned(),
        scipy_rejected: "0".to_owned(),
        rust_jacobian_calls: "0".to_owned(),
        scipy_jacobian_calls: "0".to_owned(),
    });

    // SciPy's stepwise interface reports TOO_SMALL_STEP as a failed solver
    // status. The Rust one-shot API preserves the corresponding typed cause.
    let underflow_t0 = 1.0e16;
    let underflow_config =
        extended_config(underflow_t0, underflow_t0 + 2.0, Some(1.0), 2.0, 1.0e-7);
    let mut underflow_solver = RadauNativeSolver::prepare(
        underflow_config,
        |_, y: &DVector<f64>| -y,
        Some(|_: f64, _: &DVector<f64>| DMatrix::from_element(1, 1, -1.0)),
    )
    .expect("underflow fixture prepares");
    let rust_underflow = underflow_solver.solve(&[1.0]).unwrap_err();
    assert_eq!(rust_underflow.kind(), RadauErrorKind::StepUnderflow);
    let scipy_underflow = &python["step-underflow"];
    assert_eq!(scipy_underflow["status"], "failed");
    assert!(scipy_underflow["message"].contains("spacing between numbers"));
    rows.push(ExtendedParityRow {
        scenario: "step-underflow".to_owned(),
        rust_status: format!("{:?}", rust_underflow.kind()),
        scipy_status: scipy_underflow["status"].clone(),
        rust_initial_step: "1.000000e0".to_owned(),
        scipy_initial_step: "1.000000e0".to_owned(),
        scaled_final_error: "n/a".to_owned(),
        scaled_sample_error: "n/a".to_owned(),
        rust_accepted: "0".to_owned(),
        scipy_accepted: scipy_underflow["accepted_steps"].clone(),
        rust_rejected: "0".to_owned(),
        scipy_rejected: scipy_underflow["rejected_steps"].clone(),
        rust_jacobian_calls: "n/a".to_owned(),
        scipy_jacobian_calls: scipy_underflow["njev"].clone(),
    });

    // max_steps is a Rust one-shot API extension. SciPy has no equivalent
    // solver option, so the reference adapter caps accepted driver steps.
    let mut budget_config = extended_config(0.0, 0.01, Some(0.005), 0.005, 1.0e-7);
    budget_config.max_steps = 1;
    let mut budget_solver = RadauNativeSolver::prepare(
        budget_config,
        |_, y: &DVector<f64>| -y,
        Some(|_: f64, _: &DVector<f64>| DMatrix::from_element(1, 1, -1.0)),
    )
    .expect("budget fixture prepares");
    let rust_budget = budget_solver.solve(&[1.0]).unwrap_err();
    assert_eq!(rust_budget.kind(), RadauErrorKind::StepBudget);
    let scipy_budget = &python["accepted-step-budget"];
    assert_eq!(scipy_budget["status"], "runner_step_budget_exhausted");
    assert_eq!(scipy_budget["accepted_steps"], "1");
    rows.push(ExtendedParityRow {
        scenario: "accepted-step-budget (wrapper)".to_owned(),
        rust_status: format!("{:?}", rust_budget.kind()),
        scipy_status: scipy_budget["status"].clone(),
        rust_initial_step: "5.000000e-3".to_owned(),
        scipy_initial_step: scipy_budget["initial_h_abs"].clone(),
        scaled_final_error: "n/a".to_owned(),
        scaled_sample_error: "n/a".to_owned(),
        rust_accepted: "1 before typed exhaustion".to_owned(),
        scipy_accepted: scipy_budget["accepted_steps"].clone(),
        rust_rejected: "0".to_owned(),
        scipy_rejected: scipy_budget["rejected_steps"].clone(),
        rust_jacobian_calls: "n/a".to_owned(),
        scipy_jacobian_calls: scipy_budget["njev"].clone(),
    });

    crate::Utils::test_reporting::capture_test_block(format!(
        "Extended SciPy v1.18.1 parity; atol=1e-10, rtol=1e-7 except nearby Robertson rtol=8e-8. Auto first-step rows compare first attempted h; Robertson boundary probe uses first_step=2e-4/max_step=5e-4. Failure rows compare typed/API status, not a nonexistent SciPy max_steps option.\n{}",
        Table::new(rows)
    ));
}
