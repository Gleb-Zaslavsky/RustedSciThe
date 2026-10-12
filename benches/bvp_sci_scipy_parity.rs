//! Matched Python/SciPy wall-clock versus Rust BVP numerical-core benchmark.
//!
//! Python callback construction and Rust symbolic preparation are outside
//! their respective solve timers. Rust plan preparation and per-solve solver
//! construction are reported separately so the differing scopes stay visible.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use tabled::{Table, Tabled};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan,
    BvpSciMatrixLayout, BvpSciOptions, BvpSciOutputPolicy, BvpSciSolver, BvpSciStatus,
    BvpSciTelemetry,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::Utils::test_reporting::write_test_report;

const CASES: &[(&str, BvpSciMatrixLayout)] = &[
    ("exponential-analytic", BvpSciMatrixLayout::Dense),
    ("big-parameterized-analytic", BvpSciMatrixLayout::Dense),
    ("big-parameterized-analytic", BvpSciMatrixLayout::Sparse),
    (
        "big-parameterized-analytic",
        BvpSciMatrixLayout::Banded {
            lower: 256,
            upper: 256,
        },
    ),
];

#[derive(Debug, Tabled)]
struct BenchmarkRow {
    workload: &'static str,
    rust_layout: &'static str,
    dimension: usize,
    parameters: usize,
    scipy: String,
    python_wall_median_ms: String,
    python_range_ms: String,
    rust_expr_prepare_ms: String,
    rust_solver_construct_median_ms: String,
    rust_numerical_core_median_ms: String,
    rust_construct_plus_core_median_ms: String,
    python_over_rust_core: String,
    python_over_rust_construct_core: String,
    python_status_nodes: String,
    rust_status_nodes: String,
    rust_layout_max_diff: String,
    parity_gate: &'static str,
}

struct CaseResult {
    row: BenchmarkRow,
    final_state: Vec<f64>,
}

struct Problem {
    dimension: usize,
    parameter_count: usize,
    plan: BvpSciLambdifyPlan,
    mesh: Vec<f64>,
    initial_state: Vec<f64>,
    parameters: Vec<f64>,
}

struct PythonMeasurement {
    scipy_version: String,
    numpy_version: String,
    status: u8,
    nodes: usize,
    iterations: usize,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn parse_env_usize(name: &str, default: usize, minimum: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .max(minimum)
}

fn parse_python_row(line: &str) -> Result<PythonMeasurement, String> {
    let fields = line.trim().split('\t').collect::<Vec<_>>();
    if fields.len() != 9 {
        return Err(format!(
            "Python benchmark emitted {} TSV columns, expected 9: {line:?}",
            fields.len()
        ));
    }
    let parse = |index: usize, name: &str| {
        fields[index]
            .parse::<f64>()
            .map_err(|error| format!("invalid Python {name} {:?}: {error}", fields[index]))
    };
    Ok(PythonMeasurement {
        scipy_version: fields[1].into(),
        numpy_version: fields[2].into(),
        status: fields[3]
            .parse()
            .map_err(|error| format!("invalid Python status: {error}"))?,
        nodes: fields[4]
            .parse()
            .map_err(|error| format!("invalid Python node count: {error}"))?,
        iterations: fields[5]
            .parse()
            .map_err(|error| format!("invalid Python iteration count: {error}"))?,
        median_ms: parse(6, "median time")?,
        min_ms: parse(7, "minimum time")?,
        max_ms: parse(8, "maximum time")?,
    })
}

fn python_executable() -> PathBuf {
    if let Some(path) = std::env::var_os("BVP_SCI_SCIPY_PYTHON") {
        return PathBuf::from(path);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    #[cfg(windows)]
    let candidates = [root.join(".venv-gavin-win/Scripts/python.exe")];
    #[cfg(not(windows))]
    let candidates = [root.join(".venv-gavin/bin/python")];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }))
}

fn benchmark_python(
    case: &str,
    warmups: usize,
    repeats: usize,
) -> Result<PythonMeasurement, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("src/numerical/BVP_sci/scipy_reference/benchmark.py");
    let output = Command::new(python_executable())
        .arg(script)
        .arg("--case")
        .arg(case)
        .arg("--warmups")
        .arg(warmups.to_string())
        .arg("--repeats")
        .arg(repeats.to_string())
        .output()
        .map_err(|error| format!("could not start pinned Python benchmark: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Python benchmark failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("Python benchmark output is not UTF-8: {error}"))?;
    let row = parse_python_row(stdout.trim())?;
    if !matches!(row.scipy_version.as_str(), "1.18.1")
        || !matches!(row.numpy_version.as_str(), "2.5.4")
    {
        return Err(format!(
            "benchmark requires SciPy 1.18.1 / NumPy 2.5.4, got {} / {}",
            row.scipy_version, row.numpy_version
        ));
    }
    Ok(row)
}

fn symbolic_expressions(
    case: &str,
) -> Result<(Vec<Expr>, Vec<String>, Vec<String>, usize), String> {
    match case {
        "exponential-analytic" => Ok((
            vec![Expr::parse_expression("z"), Expr::parse_expression("y")],
            vec!["y".into(), "z".into()],
            Vec::new(),
            2,
        )),
        "big-parameterized-analytic" => {
            let dimension = 60;
            let mut states = Vec::with_capacity(dimension);
            let mut equations = Vec::with_capacity(dimension);
            for pair in 0..dimension / 2 {
                let even = 2 * pair;
                let odd = even + 1;
                let even_name = format!("y{even}");
                let odd_name = format!("y{odd}");
                let parameter = pair % 2;
                states.extend([even_name.clone(), odd_name.clone()]);
                equations.push(Expr::parse_expression(&odd_name));
                equations.push(Expr::parse_expression(&format!(
                    "-p{parameter}*p{parameter}*{even_name}"
                )));
            }
            Ok((equations, states, vec!["p0".into(), "p1".into()], dimension))
        }
        _ => Err(format!("unsupported workload {case}")),
    }
}

fn prepare_problem(case: &str) -> Result<(Problem, f64), String> {
    let started = Instant::now();
    let (equations, states, parameters, dimension) = symbolic_expressions(case)?;
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &equations,
        &states,
        &parameters,
        "x",
        BvpSciTelemetry::disabled(),
    )
    .map_err(|error| format!("Rust symbolic preparation failed: {error}"))?;
    let (mesh, initial_state, parameter_values) = match case {
        "exponential-analytic" => (
            (0..5).map(|index| index as f64 / 4.0).collect(),
            vec![0.0; 10],
            Vec::new(),
        ),
        "big-parameterized-analytic" => (
            (0..5)
                .map(|index| std::f64::consts::PI * index as f64 / 4.0)
                .collect(),
            vec![1.0; dimension * 5],
            vec![0.5, 0.5],
        ),
        _ => unreachable!(),
    };
    Ok((
        Problem {
            dimension,
            parameter_count: parameter_values.len(),
            plan,
            mesh,
            initial_state,
            parameters: parameter_values,
        },
        started.elapsed().as_secs_f64() * 1e3,
    ))
}

fn boundary(case: &str, dimension: usize) -> BvpSciBoundaryCallbacks {
    match case {
        "exponential-analytic" => BvpSciBoundaryCallbacks::new_with_jacobian(
            2,
            |ya, yb, _, output| {
                output[0] = ya[0] - 1.0;
                output[1] = yb[0];
                Ok(())
            },
            |_, _, _, dya, dyb, _| {
                dya.fill(0.0);
                dyb.fill(0.0);
                dya[0] = 1.0;
                dyb[2] = 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        "big-parameterized-analytic" => BvpSciBoundaryCallbacks::new_with_jacobian(
            dimension + 2,
            move |ya, yb, parameters, output| {
                let systems = dimension / 2;
                for system in 0..systems {
                    output[system] = ya[2 * system];
                    output[systems + system] = yb[2 * system];
                }
                output[dimension] = ya[1] - parameters[0];
                output[dimension + 1] = ya[3] - parameters[1];
                Ok(())
            },
            move |_ya, _yb, _parameters, dya, dyb, dp| {
                let systems = dimension / 2;
                dya.fill(0.0);
                dyb.fill(0.0);
                dp.fill(0.0);
                for system in 0..systems {
                    dya[system * dimension + 2 * system] = 1.0;
                    dyb[(systems + system) * dimension + 2 * system] = 1.0;
                }
                dya[dimension * dimension + 1] = 1.0;
                dya[(dimension + 1) * dimension + 3] = 1.0;
                dp[dimension * 2] = -1.0;
                dp[(dimension + 1) * 2 + 1] = -1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        _ => unreachable!(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn fmt_ms(value: f64) -> String {
    format!("{value:.3}")
}

fn fmt_ratio(numerator: f64, denominator: f64) -> String {
    if denominator > 0.0 {
        format!("{:.2}x", numerator / denominator)
    } else {
        "n/a".into()
    }
}

fn benchmark_rust(
    case: &str,
    problem: &Problem,
    layout: BvpSciMatrixLayout,
    warmups: usize,
    repeats: usize,
) -> Result<(f64, f64, f64, u8, usize, Vec<f64>), String> {
    let mut construction_ms = Vec::with_capacity(repeats);
    let mut solve_ms = Vec::with_capacity(repeats);
    let mut construction_plus_solve_ms = Vec::with_capacity(repeats);
    let mut outcomes = Vec::with_capacity(repeats);
    let mut final_state = None;

    for sample in 0..(warmups + repeats) {
        let mut options = BvpSciOptions::default();
        options.execution = BvpSciExecution::Lambdify;
        options.assembly = Some(BvpSciAssembly::ExprLegacy);
        options.matrix_layout = layout;
        options.tolerance = 1e-3;
        options.max_nodes = 1000;
        options.output_policy = BvpSciOutputPolicy::MeshAndValues;

        let construction_started = Instant::now();
        let mut solver = BvpSciSolver::new(
            problem.plan.clone(),
            boundary(case, problem.dimension),
            problem.mesh.clone(),
            problem.initial_state.clone(),
            problem.parameters.clone(),
            options,
        )
        .map_err(|error| format!("Rust solver construction failed: {error}"))?;
        let construction = construction_started.elapsed().as_secs_f64() * 1e3;

        let solve_started = Instant::now();
        let solution = solver
            .solve()
            .map_err(|error| format!("Rust numerical core failed: {error}"))?;
        let solve = solve_started.elapsed().as_secs_f64() * 1e3;
        if solution.status != BvpSciStatus::Success {
            return Err(format!(
                "Rust {case} returned non-success status {}",
                solution.status.code()
            ));
        }
        outcomes.push((solution.status.code(), solution.x.len()));
        if sample >= warmups {
            construction_ms.push(construction);
            solve_ms.push(solve);
            construction_plus_solve_ms.push(construction + solve);
            final_state = Some(solution.y.clone());
        }
    }

    if outcomes.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(format!("Rust outcomes varied across repeats: {outcomes:?}"));
    }
    let (status, nodes) = outcomes[0];
    let construction_median = median(&mut construction_ms);
    let solve_median = median(&mut solve_ms);
    let total_median = median(&mut construction_plus_solve_ms);
    Ok((
        construction_median,
        solve_median,
        total_median,
        status,
        nodes,
        final_state.expect("positive repeat count produces a measured state"),
    ))
}

fn benchmark_case(
    case: &'static str,
    layout: BvpSciMatrixLayout,
    warmups: usize,
    repeats: usize,
) -> Result<CaseResult, String> {
    let python = benchmark_python(case, warmups, repeats)?;
    let (problem, prepare_ms) = prepare_problem(case)?;
    let (construct_ms, core_ms, total_ms, rust_status, rust_nodes, final_state) =
        benchmark_rust(case, &problem, layout, warmups, repeats)?;
    if python.status != rust_status || python.nodes != rust_nodes {
        return Err(format!(
            "status/mesh mismatch for {case}: Python {}/{}, Rust {rust_status}/{rust_nodes}",
            python.status, python.nodes
        ));
    }
    let workload = match case {
        "exponential-analytic" => "exponential, 2 states",
        "big-parameterized-analytic" => "official parameterized, 60 states / 2 params",
        _ => case,
    };
    Ok(CaseResult {
        row: BenchmarkRow {
            workload,
            rust_layout: match layout {
                BvpSciMatrixLayout::Dense => "Dense/nalgebra",
                BvpSciMatrixLayout::Sparse => "Sparse/faer",
                BvpSciMatrixLayout::Banded { .. } => "Banded",
            },
            dimension: problem.dimension,
            parameters: problem.parameter_count,
            scipy: format!("{} / NumPy {}", python.scipy_version, python.numpy_version),
            python_wall_median_ms: fmt_ms(python.median_ms),
            python_range_ms: format!("{:.3}..{:.3}", python.min_ms, python.max_ms),
            rust_expr_prepare_ms: fmt_ms(prepare_ms),
            rust_solver_construct_median_ms: fmt_ms(construct_ms),
            rust_numerical_core_median_ms: fmt_ms(core_ms),
            rust_construct_plus_core_median_ms: fmt_ms(total_ms),
            python_over_rust_core: fmt_ratio(python.median_ms, core_ms),
            python_over_rust_construct_core: fmt_ratio(python.median_ms, total_ms),
            python_status_nodes: format!(
                "{}/{} ({} iters)",
                python.status, python.nodes, python.iterations
            ),
            rust_status_nodes: format!("{rust_status}/{rust_nodes}"),
            rust_layout_max_diff: "pending".into(),
            parity_gate: "status/nodes + Rust layout parity",
        },
        final_state,
    })
}

fn main() {
    let warmups = parse_env_usize("BVP_SCI_SCIPY_BENCH_WARMUPS", 2, 0);
    let repeats = parse_env_usize("BVP_SCI_SCIPY_BENCH_REPEATS", 9, 1);
    let mut rows = Vec::with_capacity(CASES.len());
    let mut failures = Vec::new();
    let mut layout_reference = std::collections::HashMap::<&str, Vec<f64>>::new();
    for (case, layout) in CASES {
        match benchmark_case(case, *layout, warmups, repeats) {
            Ok(mut result) => {
                if let Some(reference) = layout_reference.get(case) {
                    let max_diff = result
                        .final_state
                        .iter()
                        .zip(reference)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0_f64, f64::max);
                    result.row.rust_layout_max_diff = format!("{max_diff:.3e}");
                    if result.final_state.len() != reference.len() || max_diff > 1e-7 {
                        failures.push(format!(
                            "{case}/{layout:?}: Rust layout solution drift {max_diff:.3e}"
                        ));
                    }
                } else {
                    result.row.rust_layout_max_diff = "reference".into();
                    layout_reference.insert(case, result.final_state.clone());
                }
                rows.push(result.row);
            }
            Err(error) => failures.push(format!("{case}/{layout:?}: {error}")),
        }
    }

    let table = Table::new(&rows);
    let report = format!(
        "# BVP_sci Python versus Rust numerical-core benchmark\n\n\
         - pinned reference: official SciPy 1.18.1 / NumPy 2.5.4; uninstrumented Python reference source\n\
         - matched workloads use analytical RHS and boundary Jacobians in both implementations\n\
         - Python timer covers each `solve_bvp` call; callback/problem construction and process startup are excluded\n\
         - Rust `numerical_core` covers `BvpSciSolver::solve`; solver construction and one-time ExprLegacy preparation are separate\n\
         - `construct + core` is the closer lifecycle comparison; neither Rust column includes ExprLegacy preparation\n\
         - Rust uses Dense/nalgebra for the small case, and Dense/nalgebra, Sparse/faer and bordered-Banded for the 60-state case; SciPy uses CSC/SuperLU\n\
         - warmups: {warmups}; measured repeats: {repeats}; cells are medians unless the column says range\n\
         - each Rust layout must match SciPy's successful status/node count and match the first Rust layout's final state within 1e-7\n\
         - full SciPy value/derivative/defect and counter semantics are gated by the dedicated debug parity stories\n\n\
         {}\n\n\
         {}",
        table,
        if failures.is_empty() {
            "All matched timing rows completed successfully.".to_string()
        } else {
            format!("Benchmark row failures:\n\n- {}", failures.join("\n- "))
        }
    );
    let name = std::env::var("BVP_SCI_SCIPY_BENCH_REPORT")
        .unwrap_or_else(|_| "python_vs_rust_numerical_core".into());
    match write_test_report("BVP_sci_SciPy_Parity_Bench", &name, &report) {
        Ok(path) => println!("report={}", path.display()),
        Err(error) => {
            eprintln!("could not write compact benchmark report: {error}");
            std::process::exit(2);
        }
    }
    if !failures.is_empty() {
        for failure in failures {
            eprintln!("{failure}");
        }
        std::process::exit(1);
    }
}
