//! Matched Python/SciPy wall-clock versus Rust BVP numerical-core benchmark.
//!
//! Python callback construction and Rust symbolic preparation are outside
//! their respective solve timers. Rust plan preparation and per-solve solver
//! construction are reported separately so the differing scopes stay visible.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BVP_sci::new::linear::NativeLinearBackend;
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan,
    BvpSciLinearSystemCapture, BvpSciMatrixLayout, BvpSciNumericalPlan, BvpSciOptions,
    BvpSciOutputPolicy, BvpSciSolver, BvpSciStatus, BvpSciTelemetry, BvpSciTelemetrySnapshot,
};
#[cfg(feature = "superlu")]
use RustedSciThe::somelinalg::superlu::{CscMatrix as SuperLuCscMatrix, SolverLibrary};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use tabled::{Table, Tabled};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustRoute {
    ExactNumerical,
    #[cfg(feature = "superlu")]
    ExprLegacySuperLu,
    ExprLegacy,
}

impl RustRoute {
    fn is_exact_numerical(self) -> bool {
        match self {
            Self::ExactNumerical => true,
            Self::ExprLegacy => false,
            #[cfg(feature = "superlu")]
            Self::ExprLegacySuperLu => false,
        }
    }

    fn uses_symbolic_frontend(self) -> bool {
        match self {
            Self::ExprLegacy => true,
            Self::ExactNumerical => false,
            #[cfg(feature = "superlu")]
            Self::ExprLegacySuperLu => true,
        }
    }

    #[cfg(feature = "superlu")]
    fn uses_superlu(self) -> bool {
        matches!(self, Self::ExprLegacySuperLu)
    }

    fn label(self) -> &'static str {
        match self {
            Self::ExactNumerical => "Numerical/exact-Jacobian",
            #[cfg(feature = "superlu")]
            Self::ExprLegacySuperLu => "ExprLegacy/exact-symbolic + Sparse/SuperLU",
            Self::ExprLegacy => "ExprLegacy/Lambdify",
        }
    }
}

const CASES: &[(&str, BvpSciMatrixLayout, RustRoute)] = &[
    (
        "exponential-analytic",
        BvpSciMatrixLayout::Dense,
        RustRoute::ExactNumerical,
    ),
    (
        "big-parameterized-analytic",
        BvpSciMatrixLayout::Dense,
        RustRoute::ExactNumerical,
    ),
    #[cfg(feature = "superlu")]
    (
        "big-parameterized-analytic",
        BvpSciMatrixLayout::Sparse,
        RustRoute::ExprLegacySuperLu,
    ),
    (
        "big-parameterized-analytic",
        BvpSciMatrixLayout::Dense,
        RustRoute::ExprLegacy,
    ),
    (
        "big-parameterized-analytic",
        BvpSciMatrixLayout::Sparse,
        RustRoute::ExprLegacy,
    ),
    (
        "big-parameterized-analytic",
        BvpSciMatrixLayout::Banded {
            lower: 256,
            upper: 256,
        },
        RustRoute::ExprLegacy,
    ),
];

#[derive(Debug, Tabled)]
struct BenchmarkRow {
    workload: &'static str,
    rust_route: &'static str,
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
    rust_solution_max_diff: String,
    parity_gate: &'static str,
}

struct CaseResult {
    row: BenchmarkRow,
    final_state: Vec<f64>,
}

struct Problem {
    dimension: usize,
    parameter_count: usize,
    expr_plan: Option<BvpSciLambdifyPlan>,
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

#[derive(Debug, Tabled)]
struct StageDiagnosticRow {
    rust_route: &'static str,
    rust_layout: &'static str,
    mesh_nodes: usize,
    linear_system_dimension: usize,
    stage: &'static str,
    scipy_calls: String,
    scipy_work_units: String,
    scipy_ms: String,
    rust_calls: String,
    rust_work_units: String,
    rust_ms: String,
}

#[derive(Debug, Tabled)]
struct MeshPassDiagnosticRow {
    rust_route: &'static str,
    rust_layout: &'static str,
    pass: usize,
    scipy_nodes: usize,
    rust_nodes: usize,
    scipy_linear_dimension: usize,
    rust_linear_dimension: usize,
    scipy_newton_passes: String,
    rust_newton_passes: u64,
    scipy_jacobian_builds: String,
    rust_jacobian_builds: u64,
    scipy_factorizations: String,
    rust_factorizations: u64,
    scipy_linear_solves: String,
    rust_linear_solves: u64,
    scipy_rhs_calls: String,
    rust_rhs_calls: u64,
    scipy_rhs_state_points: String,
    rust_rhs_state_points: u64,
    rust_rhs_jacobian_calls: u64,
    rust_parameter_jacobian_calls: u64,
    scipy_rhs_callback_ms: String,
    rust_rhs_callback_ms: String,
    scipy_rhs_jacobian_ms: String,
    rust_rhs_jacobian_ms: String,
    scipy_boundary_callback_ms: String,
    rust_boundary_callback_ms: String,
    scipy_collocation_ms: String,
    rust_collocation_ms: String,
    scipy_pass_ms: String,
    rust_pass_ms: String,
    scipy_newton_ms: String,
    rust_newton_ms: String,
    scipy_factorization_ms: String,
    rust_factorization_ms: String,
    scipy_linear_solve_ms: String,
    rust_linear_solve_ms: String,
    scipy_defect_ms: String,
    rust_defect_ms: String,
    scipy_mesh_update_ms: String,
    rust_mesh_update_ms: String,
}

#[derive(Debug, Tabled)]
struct LinearKernelRow {
    implementation: &'static str,
    layout: &'static str,
    dimension: usize,
    mesh_nodes: usize,
    input_triplets: usize,
    canonical_coordinates: usize,
    numerical_nonzeros: usize,
    factor_l_nnz: String,
    factor_u_nnz: String,
    assembly_ms: String,
    cold_factor_ms: String,
    warm_factor_median_ms: String,
    fresh_factor_median_ms: String,
    solve_median_ms: String,
    phase_factor_samples: String,
    phase_sparse_symbolic_analyses: String,
    phase_sparse_symbolic_reuses: String,
    phase_sparse_numeric_factorizations: String,
    phase_banded_core_factorizations: String,
    phase_banded_core_to_border_solves: String,
    phase_banded_schur_assemblies: String,
    phase_banded_border_factorizations: String,
    phase_sparse_triplet_ms: String,
    phase_sparse_csc_ms: String,
    phase_sparse_symbolic_ms: String,
    phase_sparse_symbolic_reuse_ms: String,
    phase_sparse_numeric_factor_ms: String,
    phase_banded_core_factor_ms: String,
    phase_banded_core_to_border_ms: String,
    phase_banded_schur_assembly_ms: String,
    phase_banded_border_factor_ms: String,
    residual_inf: String,
    relative_residual_inf: String,
}

#[derive(Debug, Tabled)]
struct LinearBackendMeshRow {
    rust_route: &'static str,
    rust_layout: &'static str,
    pass: u64,
    mesh_nodes: usize,
    linear_dimension: usize,
    sparse_symbolic_analyses: u64,
    sparse_symbolic_reuses: u64,
    sparse_numeric_factorizations: u64,
    sparse_triplet_materializations: u64,
    sparse_csc_assemblies: u64,
    sparse_triplet_materialization_ms: String,
    sparse_csc_assembly_ms: String,
    sparse_symbolic_analysis_ms: String,
    sparse_symbolic_reuse_ms: String,
    sparse_numeric_factorization_ms: String,
    banded_core_factorizations: u64,
    banded_core_to_border_solves: u64,
    banded_schur_assemblies: u64,
    banded_border_factorizations: u64,
    banded_core_factorization_ms: String,
    banded_core_to_border_solve_ms: String,
    banded_schur_assembly_ms: String,
    banded_border_factorization_ms: String,
    banded_structured_factorizations: u64,
    banded_scalar_fallback_factorizations: u64,
    banded_sparse_fallback_factorizations: u64,
    banded_structured_solves: u64,
    banded_scalar_fallback_solves: u64,
    banded_sparse_fallback_solves: u64,
    banded_structured_factorization_ms: String,
    banded_scalar_fallback_factorization_ms: String,
    banded_sparse_fallback_factorization_ms: String,
    banded_structured_solve_ms: String,
    banded_scalar_fallback_solve_ms: String,
    banded_sparse_fallback_solve_ms: String,
    banded_residual_guard_ms: String,
    banded_rhs_permutation_ms: String,
}

#[cfg(feature = "superlu")]
#[derive(Debug, Tabled)]
struct SuperLuStageRow {
    rust_layout: &'static str,
    csc_assemblies: u64,
    csc_assembly_ms: String,
    abi_conversions: u64,
    abi_conversion_ms: String,
    factor_ffi_calls: u64,
    factor_ffi_ms: String,
    native_factorizations: u64,
    native_factorization_ms: String,
    solve_ffi_calls: u64,
    solve_ffi_ms: String,
    native_solves: u64,
    native_solve_ms: String,
    fallbacks: u64,
}

#[cfg(feature = "superlu")]
fn superlu_stage_row(snapshot: &BvpSciTelemetrySnapshot) -> SuperLuStageRow {
    SuperLuStageRow {
        rust_layout: "Sparse/SuperLU",
        csc_assemblies: snapshot.sparse_csc_assemblies,
        csc_assembly_ms: snapshot_time(snapshot.sparse_csc_assembly_ms),
        abi_conversions: snapshot.superlu.abi_conversions,
        abi_conversion_ms: snapshot_time(snapshot.superlu.abi_conversion_ms),
        factor_ffi_calls: snapshot.superlu.factor_ffi_calls,
        factor_ffi_ms: snapshot_time(snapshot.superlu.factor_ffi_ms),
        native_factorizations: snapshot.superlu.native_factorization_calls,
        native_factorization_ms: snapshot_time(snapshot.superlu.native_factorization_ms),
        solve_ffi_calls: snapshot.superlu.solve_ffi_calls,
        solve_ffi_ms: snapshot_time(snapshot.superlu.solve_ffi_ms),
        native_solves: snapshot.superlu.native_solve_calls,
        native_solve_ms: snapshot_time(snapshot.superlu.native_solve_ms),
        fallbacks: snapshot.superlu_fallbacks,
    }
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

fn python_diagnostic(case: &str) -> Result<HashMap<String, String>, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("src/numerical/BVP_sci/scipy_reference/bvp_parity.py");
    let output = Command::new(python_executable())
        .arg(script)
        .arg(case)
        .arg("--format")
        .arg("diagnostic-tsv")
        .output()
        .map_err(|error| format!("could not start SciPy diagnostic: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "SciPy diagnostic failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("SciPy diagnostic output is not UTF-8: {error}"))?;
    stdout
        .trim()
        .split('\t')
        .map(|field| {
            field
                .split_once('=')
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .ok_or_else(|| format!("invalid SciPy diagnostic field {field:?}"))
        })
        .collect()
}

fn numeric_field(fields: &HashMap<String, String>, key: &str) -> Result<u64, String> {
    fields
        .get(key)
        .ok_or_else(|| format!("SciPy diagnostic omitted {key}"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid SciPy counter {key}: {error}"))
}

fn optional_time(fields: &HashMap<String, String>, key: &str) -> String {
    fields
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .map(fmt_ms)
        .unwrap_or_else(|| "n/a".into())
}

fn snapshot_time(value: Option<f64>) -> String {
    value.map(fmt_ms).unwrap_or_else(|| "n/a".into())
}

fn layout_name(layout: BvpSciMatrixLayout) -> &'static str {
    match layout {
        BvpSciMatrixLayout::Dense => "Dense/nalgebra",
        BvpSciMatrixLayout::Sparse => "Sparse/faer",
        BvpSciMatrixLayout::Banded { .. } => "Banded",
    }
}

fn route_layout_name(route: RustRoute, layout: BvpSciMatrixLayout) -> &'static str {
    #[cfg(not(feature = "superlu"))]
    let _ = route;
    #[cfg(feature = "superlu")]
    if route.uses_superlu() {
        return "Sparse/SuperLU";
    }
    layout_name(layout)
}

fn sum_snapshot_times(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    Some(left? + right?)
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

fn prepare_problem(case: &str, prepare_expr: bool) -> Result<(Problem, f64), String> {
    let started = Instant::now();
    let (dimension, parameter_values) = match case {
        "exponential-analytic" => (2, Vec::new()),
        "big-parameterized-analytic" => (60, vec![0.5, 0.5]),
        _ => return Err(format!("unsupported workload {case}")),
    };
    let (mesh, initial_state) = match case {
        "exponential-analytic" => (
            (0..5).map(|index| index as f64 / 4.0).collect(),
            vec![0.0; 10],
        ),
        "big-parameterized-analytic" => (
            (0..5)
                .map(|index| std::f64::consts::PI * index as f64 / 4.0)
                .collect(),
            vec![1.0; dimension * 5],
        ),
        _ => unreachable!(),
    };
    let expr_plan = if prepare_expr {
        let (equations, states, parameters, _) = symbolic_expressions(case)?;
        Some(
            BvpSciLambdifyPlan::prepare(
                BvpSciAssembly::ExprLegacy,
                &equations,
                &states,
                &parameters,
                "x",
                BvpSciTelemetry::disabled(),
            )
            .map_err(|error| format!("Rust symbolic preparation failed: {error}"))?,
        )
    } else {
        None
    };
    Ok((
        Problem {
            dimension,
            parameter_count: parameter_values.len(),
            expr_plan,
            mesh,
            initial_state,
            parameters: parameter_values,
        },
        started.elapsed().as_secs_f64() * 1e3,
    ))
}

fn make_numerical_plan(
    case: &str,
    telemetry: BvpSciTelemetry,
) -> Result<BvpSciLambdifyPlan, String> {
    match case {
        "exponential-analytic" => {
            let plan = BvpSciNumericalPlan::new(
                2,
                0,
                |_x, state, _parameters, output| {
                    output[0] = state[1];
                    output[1] = state[0];
                    Ok(())
                },
                telemetry,
            )
            .map_err(|error| format!("Rust numerical plan failed: {error}"))?
            .with_rhs_jacobian(|_x, _state, _parameters, output| {
                output.fill(0.0);
                output[1] = 1.0;
                output[2] = 1.0;
                Ok(())
            });
            Ok(BvpSciLambdifyPlan::prepare_numerical(plan))
        }
        "big-parameterized-analytic" => {
            let dimension = 60;
            let systems = dimension / 2;
            let plan = BvpSciNumericalPlan::new(
                dimension,
                2,
                move |_x, state, parameters, output| {
                    for system in 0..systems {
                        let parameter = system % 2;
                        output[2 * system] = state[2 * system + 1];
                        output[2 * system + 1] =
                            -(parameters[parameter] * parameters[parameter]) * state[2 * system];
                    }
                    Ok(())
                },
                telemetry,
            )
            .map_err(|error| format!("Rust numerical plan failed: {error}"))?
            .with_rhs_jacobian(move |_x, _state, parameters, output| {
                output.fill(0.0);
                for system in 0..systems {
                    let parameter = system % 2;
                    output[(2 * system) * dimension + (2 * system + 1)] = 1.0;
                    output[(2 * system + 1) * dimension + 2 * system] =
                        -(parameters[parameter] * parameters[parameter]);
                }
                Ok(())
            })
            .with_rhs_parameter_jacobian(move |_x, state, parameters, output| {
                output.fill(0.0);
                for system in 0..systems {
                    let parameter = system % 2;
                    output[(2 * system + 1) * 2 + parameter] =
                        -2.0 * parameters[parameter] * state[2 * system];
                }
                Ok(())
            });
            Ok(BvpSciLambdifyPlan::prepare_numerical(plan))
        }
        _ => Err(format!("unsupported numerical workload {case}")),
    }
}

fn boundary(case: &str, dimension: usize, telemetry: BvpSciTelemetry) -> BvpSciBoundaryCallbacks {
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
            telemetry,
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
            telemetry,
        ),
        _ => unreachable!(),
    }
}

fn run_route_diagnostic(
    case: &str,
    layout: BvpSciMatrixLayout,
    route: RustRoute,
) -> Result<
    (
        BvpSciTelemetrySnapshot,
        u8,
        usize,
        Option<BvpSciLinearSystemCapture>,
    ),
    String,
> {
    let (problem, _) = prepare_problem(case, route.uses_symbolic_frontend())?;
    let telemetry = BvpSciTelemetry::detailed();
    if matches!(route, RustRoute::ExactNumerical) && layout == BvpSciMatrixLayout::Dense {
        telemetry.enable_linear_system_capture(1000);
    }
    let plan = match route {
        RustRoute::ExactNumerical => make_numerical_plan(case, telemetry.clone())?,
        #[cfg(feature = "superlu")]
        RustRoute::ExprLegacySuperLu => {
            let (equations, state_names, parameter_names, _) = symbolic_expressions(case)?;
            BvpSciLambdifyPlan::prepare(
                BvpSciAssembly::ExprLegacy,
                &equations,
                &state_names,
                &parameter_names,
                "x",
                telemetry.clone(),
            )
            .map_err(|error| {
                format!("Rust SuperLU symbolic diagnostic preparation failed: {error}")
            })?
        }
        RustRoute::ExprLegacy => {
            let (equations, state_names, parameter_names, _) = symbolic_expressions(case)?;
            BvpSciLambdifyPlan::prepare(
                BvpSciAssembly::ExprLegacy,
                &equations,
                &state_names,
                &parameter_names,
                "x",
                telemetry.clone(),
            )
            .map_err(|error| format!("Rust ExprLegacy diagnostic preparation failed: {error}"))?
        }
    };
    let mut options = BvpSciOptions::default();
    options.execution = if route.is_exact_numerical() {
        BvpSciExecution::Numerical
    } else {
        BvpSciExecution::Lambdify
    };
    options.assembly = Some(plan.assembly());
    options.matrix_layout = layout;
    options.tolerance = 1e-3;
    options.max_nodes = 1000;
    options.output_policy = BvpSciOutputPolicy::MeshAndValues;
    let boundary = boundary(case, problem.dimension, telemetry.clone());
    #[cfg(feature = "superlu")]
    let solver_result = if route.uses_superlu() {
        BvpSciSolver::new_with_sparse_factorization(
            plan,
            boundary,
            problem.mesh,
            problem.initial_state,
            problem.parameters,
            options,
            RustedSciThe::numerical::BVP_sci::new::BvpSciSparseFactorization::SuperLu,
        )
    } else {
        BvpSciSolver::new(
            plan,
            boundary,
            problem.mesh,
            problem.initial_state,
            problem.parameters,
            options,
        )
    };
    #[cfg(not(feature = "superlu"))]
    let solver_result = BvpSciSolver::new(
        plan,
        boundary,
        problem.mesh,
        problem.initial_state,
        problem.parameters,
        options,
    );
    let mut solver = solver_result
        .map_err(|error| format!("Rust diagnostic solver construction failed: {error}"))?;
    let solution = solver
        .solve()
        .map_err(|error| format!("Rust diagnostic solve failed: {error}"))?;
    if solution.status != BvpSciStatus::Success {
        return Err(format!(
            "Rust diagnostic returned non-success status {}",
            solution.status.code()
        ));
    }
    #[cfg(feature = "superlu")]
    if route.uses_superlu() {
        let snapshot = solver.plan().telemetry_snapshot();
        if snapshot.superlu_fallbacks != 0 || snapshot.superlu.native_factorization_calls == 0 {
            return Err(format!(
                "SuperLU diagnostic did not use the native backend: fallback_count={}, native_factorizations={}",
                snapshot.superlu_fallbacks, snapshot.superlu.native_factorization_calls
            ));
        }
    }
    Ok((
        solver.plan().telemetry_snapshot(),
        solution.status.code(),
        solution.x.len(),
        telemetry.linear_system_capture(),
    ))
}

fn build_stage_rows(
    case: &str,
    python: &HashMap<String, String>,
    layout: BvpSciMatrixLayout,
    route: RustRoute,
    state_dimension: usize,
    parameter_count: usize,
    mesh_nodes: usize,
    rust: &BvpSciTelemetrySnapshot,
) -> Result<Vec<StageDiagnosticRow>, String> {
    let layout_name = route_layout_name(route, layout);
    let py_count = |name: &str| numeric_field(python, &format!("counter.{name}"));
    let py_units = |name: &str| numeric_field(python, &format!("counter.{name}"));
    let py_count_or_zero = |name: &str| {
        python
            .get(&format!("counter.{name}"))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    };
    let py_ms = |name: &str| optional_time(python, &format!("time_ms.{name}"));
    let row = |stage: &'static str,
               scipy_calls: Result<u64, String>,
               scipy_units: Result<u64, String>,
               scipy_ms: String,
               rust_calls: u64,
               rust_units: u64,
               rust_ms: Option<f64>| {
        StageDiagnosticRow {
            rust_route: route.label(),
            rust_layout: layout_name,
            mesh_nodes,
            linear_system_dimension: state_dimension * mesh_nodes + parameter_count,
            stage,
            scipy_calls: scipy_calls
                .map(|value| value.to_string())
                .unwrap_or_else(|_| "n/a".into()),
            scipy_work_units: scipy_units
                .map(|value| value.to_string())
                .unwrap_or_else(|_| "n/a".into()),
            scipy_ms,
            rust_calls: rust_calls.to_string(),
            rust_work_units: rust_units.to_string(),
            rust_ms: snapshot_time(rust_ms),
        }
    };

    if python.get("case").map(String::as_str) != Some(case)
        || python.get("success").map(String::as_str) != Some("1")
        || python.get("status").map(String::as_str) != Some("0")
    {
        return Err(format!("unexpected SciPy diagnostic result for {case}"));
    }

    let mut rows = vec![
        row(
            "RHS residual callback",
            py_count("rhs_callback_calls"),
            py_units("rhs_callback_state_points"),
            py_ms("rhs_callback"),
            rust.rhs_calls,
            rust.rhs_calls,
            rust.rhs_callback_ms,
        ),
        row(
            "RHS Jacobians (df/dy + df/dp)",
            py_count("rhs_jacobian_callback_calls"),
            py_units("rhs_jacobian_callback_state_points"),
            py_ms("rhs_jacobian_callback"),
            rust.jacobian_calls + rust.parameter_jacobian_calls,
            rust.jacobian_calls.min(rust.parameter_jacobian_calls),
            sum_snapshot_times(
                rust.jacobian_callback_ms,
                rust.parameter_jacobian_callback_ms,
            ),
        ),
        row(
            "Boundary residual callback",
            py_count("boundary_callback_calls"),
            py_units("boundary_callback_calls"),
            py_ms("boundary_callback"),
            rust.boundary_calls,
            rust.boundary_calls,
            rust.boundary_callback_ms,
        ),
        row(
            "Boundary Jacobian callback",
            py_count("boundary_jacobian_callback_calls"),
            py_units("boundary_jacobian_callback_calls"),
            py_ms("boundary_jacobian_callback"),
            rust.boundary_jacobian_calls,
            rust.boundary_jacobian_calls,
            rust.boundary_jacobian_callback_ms,
        ),
        row(
            "Finite-difference callback probes",
            Ok(py_count_or_zero("rhs_finite_difference_callback_calls")
                + py_count_or_zero("boundary_finite_difference_callback_calls")),
            Ok(py_count_or_zero("rhs_finite_difference_callback_calls")
                + py_count_or_zero("boundary_finite_difference_callback_calls")),
            "n/a".into(),
            rust.finite_difference_probes,
            rust.finite_difference_probes,
            None,
        ),
        row(
            "Newton backtracking trials",
            py_count("newton_trials"),
            py_count("newton_trials"),
            "n/a".into(),
            rust.newton_residual_history
                .iter()
                .map(|entry| entry.backtracking_trials)
                .sum(),
            rust.newton_backtracking_trials,
            None,
        ),
        row(
            "Newton solve passes",
            py_count("solve_newton_calls"),
            py_count("solve_newton_calls"),
            py_ms("solve_newton"),
            rust.newton_scope_calls,
            rust.newton_scope_calls,
            rust.newton_ms,
        ),
        row(
            "Global Jacobian builds",
            py_count("construct_global_jac_calls"),
            py_count("construct_global_jac_calls"),
            py_ms("construct_global_jac"),
            rust.newton_jacobian_refreshes,
            rust.newton_jacobian_refreshes,
            rust.jacobian_output_assembly_ms,
        ),
        row(
            "LU factorizations",
            py_count("factorization_calls"),
            py_count("factorization_calls"),
            py_ms("factorization"),
            rust.factorizations,
            rust.factorizations,
            rust.factorization_ms,
        ),
        row(
            "Linear solves",
            py_count("linear_solve_calls"),
            py_count("linear_solve_calls"),
            py_ms("linear_solve"),
            rust.linear_solves,
            rust.linear_solves,
            rust.linear_solve_ms,
        ),
        row(
            "Linear-system preparation/assembly",
            py_count("prepare_sys_calls"),
            py_count("prepare_sys_calls"),
            py_ms("prepare_sys"),
            rust.linear_assemblies,
            rust.linear_assemblies,
            rust.linear_assembly_ms,
        ),
        row(
            "Collocation evaluations",
            py_count("collocation_fun_calls"),
            py_count("collocation_fun_calls"),
            py_ms("collocation_fun"),
            rust.collocation_evaluations,
            rust.collocation_evaluations,
            rust.collocation_ms,
        ),
        row(
            "Defect estimates",
            py_count("estimate_rms_residuals_calls"),
            py_count("estimate_rms_residuals_calls"),
            py_ms("estimate_rms_residuals"),
            rust.mesh_defect_estimation_calls,
            rust.mesh_defect_estimation_calls,
            rust.mesh_defect_estimation_ms,
        ),
        row(
            "Mesh modifications",
            py_count("modify_mesh_calls"),
            py_count("modify_mesh_calls"),
            py_ms("modify_mesh"),
            rust.mesh_refinements,
            rust.mesh_points_added,
            rust.mesh_refinement_ms,
        ),
        row(
            "Spline/output construction",
            py_count("create_spline_calls"),
            py_count("create_spline_calls"),
            py_ms("create_spline"),
            rust.output_construction_calls,
            rust.output_construction_calls,
            rust.output_construction_ms,
        ),
        row(
            "Full solve",
            py_count("solve_bvp_calls"),
            py_count("solve_bvp_calls"),
            py_ms("solve_bvp"),
            rust.full_solve_calls,
            rust.full_solve_calls,
            rust.full_solve_ms,
        ),
    ];
    for item in &mut rows {
        item.rust_layout = layout_name;
    }
    Ok(rows)
}

fn build_mesh_pass_rows(
    python: &HashMap<String, String>,
    rust_route: RustRoute,
    rust_layout: &'static str,
    rust: &BvpSciTelemetrySnapshot,
    failures: &mut Vec<String>,
    gate_core_counts: bool,
) -> Result<Vec<MeshPassDiagnosticRow>, String> {
    let python_passes = numeric_field(python, "mesh_pass_count")? as usize;
    if gate_core_counts && python_passes != rust.mesh_passes.len() {
        failures.push(format!(
            "mesh-pass count differs for {rust_layout}: SciPy {python_passes}, Rust {}",
            rust.mesh_passes.len()
        ));
    }
    let mut rows = Vec::with_capacity(python_passes.max(rust.mesh_passes.len()));
    for pass in 0..python_passes.max(rust.mesh_passes.len()) {
        let prefix = format!("mesh.{pass}");
        let scipy_nodes = numeric_field(python, &format!("{prefix}.nodes"))? as usize;
        let scipy_dimension = numeric_field(python, &format!("{prefix}.dimension"))? as usize;
        let Some(native) = rust.mesh_passes.get(pass) else {
            continue;
        };
        if gate_core_counts
            && (scipy_nodes != native.mesh_nodes
                || scipy_dimension != native.linear_system_dimension)
        {
            failures.push(format!(
                "mesh pass {pass} layout {rust_layout}: SciPy mesh/dimension {scipy_nodes}/{scipy_dimension}, Rust {}/{}",
                native.mesh_nodes, native.linear_system_dimension
            ));
        }
        let py_counter = |name: &str| {
            python
                .get(&format!("{prefix}.counter.{name}"))
                .and_then(|value| value.parse::<u64>().ok())
        };
        let py_time = |name: &str| {
            python
                .get(&format!("{prefix}.time_ms.{name}"))
                .and_then(|value| value.parse::<f64>().ok())
        };
        let compare = [
            ("solve_newton_calls", native.newton_passes),
            ("construct_global_jac_calls", native.jacobian_builds),
            ("factorization_calls", native.factorizations),
            ("linear_solve_calls", native.linear_solves),
        ];
        if gate_core_counts {
            for (name, rust_count) in compare {
                if let Some(python_count) = py_counter(name) {
                    if python_count != rust_count {
                        failures.push(format!(
                            "mesh pass {pass} {name} differs for {rust_layout}: SciPy {python_count}, Rust {rust_count}"
                        ));
                    }
                }
            }
        }
        rows.push(MeshPassDiagnosticRow {
            rust_route: rust_route.label(),
            rust_layout,
            pass,
            scipy_nodes,
            rust_nodes: native.mesh_nodes,
            scipy_linear_dimension: scipy_dimension,
            rust_linear_dimension: native.linear_system_dimension,
            scipy_newton_passes: py_counter("solve_newton_calls")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into()),
            rust_newton_passes: native.newton_passes,
            scipy_jacobian_builds: py_counter("construct_global_jac_calls")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into()),
            rust_jacobian_builds: native.jacobian_builds,
            scipy_factorizations: py_counter("factorization_calls")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into()),
            rust_factorizations: native.factorizations,
            scipy_linear_solves: py_counter("linear_solve_calls")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into()),
            rust_linear_solves: native.linear_solves,
            scipy_rhs_calls: py_counter("rhs_callback_calls")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into()),
            rust_rhs_calls: native.rhs_calls,
            scipy_rhs_state_points: py_counter("rhs_callback_state_points")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into()),
            rust_rhs_state_points: native.rhs_calls,
            rust_rhs_jacobian_calls: native.rhs_jacobian_calls,
            rust_parameter_jacobian_calls: native.parameter_jacobian_calls,
            scipy_rhs_callback_ms: py_time("rhs_callback")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_rhs_callback_ms: snapshot_time(native.rhs_callback_ms),
            scipy_rhs_jacobian_ms: py_time("rhs_jacobian_callback")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_rhs_jacobian_ms: snapshot_time(sum_snapshot_times(
                native.jacobian_callback_ms,
                native.parameter_jacobian_callback_ms,
            )),
            scipy_boundary_callback_ms: py_time("boundary_callback")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_boundary_callback_ms: snapshot_time(native.boundary_callback_ms),
            scipy_collocation_ms: py_time("collocation_fun")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_collocation_ms: snapshot_time(native.collocation_ms),
            scipy_pass_ms: py_time("elapsed")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_pass_ms: fmt_ms(native.elapsed_ms),
            scipy_newton_ms: py_time("solve_newton")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_newton_ms: snapshot_time(native.newton_ms),
            scipy_factorization_ms: py_time("factorization")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_factorization_ms: snapshot_time(native.factorization_ms),
            scipy_linear_solve_ms: py_time("linear_solve")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_linear_solve_ms: snapshot_time(native.linear_solve_ms),
            scipy_defect_ms: py_time("estimate_rms_residuals")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_defect_ms: snapshot_time(native.defect_estimation_ms),
            scipy_mesh_update_ms: py_time("modify_mesh")
                .map(fmt_ms)
                .unwrap_or_else(|| "n/a".into()),
            rust_mesh_update_ms: snapshot_time(native.mesh_refinement_ms),
        });
    }
    Ok(rows)
}

fn build_backend_mesh_rows(
    route: RustRoute,
    layout: BvpSciMatrixLayout,
    rust: &BvpSciTelemetrySnapshot,
) -> Vec<LinearBackendMeshRow> {
    if layout == BvpSciMatrixLayout::Dense {
        return Vec::new();
    }
    rust.mesh_passes
        .iter()
        .map(|pass| LinearBackendMeshRow {
            rust_route: route.label(),
            rust_layout: route_layout_name(route, layout),
            pass: pass.mesh_pass,
            mesh_nodes: pass.mesh_nodes,
            linear_dimension: pass.linear_system_dimension,
            sparse_symbolic_analyses: pass.sparse_symbolic_analyses,
            sparse_symbolic_reuses: pass.sparse_symbolic_reuses,
            sparse_numeric_factorizations: pass.sparse_numeric_factorizations,
            sparse_triplet_materializations: pass.sparse_triplet_materializations,
            sparse_csc_assemblies: pass.sparse_csc_assemblies,
            sparse_triplet_materialization_ms: snapshot_time(
                pass.sparse_triplet_materialization_ms,
            ),
            sparse_csc_assembly_ms: snapshot_time(pass.sparse_csc_assembly_ms),
            sparse_symbolic_analysis_ms: snapshot_time(pass.sparse_symbolic_analysis_ms),
            sparse_symbolic_reuse_ms: snapshot_time(pass.sparse_symbolic_reuse_ms),
            sparse_numeric_factorization_ms: snapshot_time(pass.sparse_numeric_factorization_ms),
            banded_core_factorizations: pass.banded_core_factorizations,
            banded_core_to_border_solves: pass.banded_core_to_border_solves,
            banded_schur_assemblies: pass.banded_schur_assemblies,
            banded_border_factorizations: pass.banded_border_factorizations,
            banded_core_factorization_ms: snapshot_time(pass.banded_core_factorization_ms),
            banded_core_to_border_solve_ms: snapshot_time(pass.banded_core_to_border_solve_ms),
            banded_schur_assembly_ms: snapshot_time(pass.banded_schur_assembly_ms),
            banded_border_factorization_ms: snapshot_time(pass.banded_border_factorization_ms),
            banded_structured_factorizations: pass.banded_structured_factorizations,
            banded_scalar_fallback_factorizations: pass.banded_scalar_fallback_factorizations,
            banded_sparse_fallback_factorizations: pass.banded_sparse_fallback_factorizations,
            banded_structured_solves: pass.banded_structured_solves,
            banded_scalar_fallback_solves: pass.banded_scalar_fallback_solves,
            banded_sparse_fallback_solves: pass.banded_sparse_fallback_solves,
            banded_structured_factorization_ms: snapshot_time(
                pass.banded_structured_factorization_ms,
            ),
            banded_scalar_fallback_factorization_ms: snapshot_time(
                pass.banded_scalar_fallback_factorization_ms,
            ),
            banded_sparse_fallback_factorization_ms: snapshot_time(
                pass.banded_sparse_fallback_factorization_ms,
            ),
            banded_structured_solve_ms: snapshot_time(pass.banded_structured_solve_ms),
            banded_scalar_fallback_solve_ms: snapshot_time(pass.banded_scalar_fallback_solve_ms),
            banded_sparse_fallback_solve_ms: snapshot_time(pass.banded_sparse_fallback_solve_ms),
            banded_residual_guard_ms: snapshot_time(pass.banded_residual_guard_ms),
            banded_rhs_permutation_ms: snapshot_time(pass.banded_rhs_permutation_ms),
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn residual_norms(capture: &BvpSciLinearSystemCapture, solution: &[f64]) -> (f64, f64) {
    let mut residual = capture.rhs.iter().map(|value| -*value).collect::<Vec<_>>();
    for &(row, column, value) in &capture.entries {
        residual[row] += value * solution[column];
    }
    let residual_inf = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let rhs_inf = capture
        .rhs
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    (residual_inf, residual_inf / rhs_inf.max(1.0))
}

fn canonical_entry_counts(entries: &[(usize, usize, f64)]) -> (usize, usize) {
    let mut combined = HashMap::<(usize, usize), f64>::with_capacity(entries.len());
    for &(row, column, value) in entries {
        *combined.entry((row, column)).or_default() += value;
    }
    (
        combined.len(),
        combined.values().filter(|value| **value != 0.0).count(),
    )
}

fn write_linear_capture(capture: &BvpSciLinearSystemCapture) -> Result<PathBuf, String> {
    let path = std::env::temp_dir().join(format!(
        "bvp_sci_linear_capture_{}_{}.tsv",
        std::process::id(),
        capture.dimension
    ));
    let mut text = format!(
        "{} {} {}\n",
        capture.dimension,
        capture.mesh_nodes,
        capture.entries.len()
    );
    for (index, value) in capture.rhs.iter().enumerate() {
        if index > 0 {
            text.push(' ');
        }
        text.push_str(&format!("{value:.17e}"));
    }
    text.push('\n');
    for &(row, column, value) in &capture.entries {
        text.push_str(&format!("{row} {column} {value:.17e}\n"));
    }
    fs::write(&path, text).map_err(|error| format!("could not export captured system: {error}"))?;
    Ok(path)
}

fn run_python_linear_kernel(
    capture: &BvpSciLinearSystemCapture,
    repeats: usize,
) -> Result<HashMap<String, String>, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("src/numerical/BVP_sci/scipy_reference/linear_kernel_benchmark.py");
    let input = write_linear_capture(capture)?;
    let output = Command::new(python_executable())
        .arg(script)
        .arg(&input)
        .arg("--repeats")
        .arg(repeats.to_string())
        .output()
        .map_err(|error| format!("could not start SciPy linear-kernel benchmark: {error}"));
    let _ = fs::remove_file(input);
    let output = output?;
    if !output.status.success() {
        return Err(format!(
            "SciPy linear-kernel benchmark failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("SciPy kernel output is not UTF-8: {error}"))?
        .trim()
        .split('\t')
        .map(|field| {
            field
                .split_once('=')
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .ok_or_else(|| format!("invalid SciPy kernel field {field:?}"))
        })
        .collect()
}

fn run_rust_linear_kernel(
    capture: &BvpSciLinearSystemCapture,
    layout: BvpSciMatrixLayout,
    repeats: usize,
) -> Result<LinearKernelRow, String> {
    let telemetry = BvpSciTelemetry::detailed();
    let mut backend = NativeLinearBackend::new_for_collocation(
        layout,
        capture.dimension,
        capture.entries.len(),
        60,
        capture.mesh_nodes,
        2,
        telemetry.clone(),
    )
    .map_err(|error| {
        format!(
            "could not construct {} backend: {error}",
            layout_name(layout)
        )
    })?;

    let started = Instant::now();
    backend
        .assemble(&capture.entries)
        .map_err(|error| format!("{} assembly failed: {error}", layout_name(layout)))?;
    let assembly_ms = started.elapsed().as_secs_f64() * 1e3;

    let started = Instant::now();
    backend
        .factor()
        .map_err(|error| format!("{} cold factorization failed: {error}", layout_name(layout)))?;
    let cold_factor_ms = started.elapsed().as_secs_f64() * 1e3;

    let mut warm_factor_times = Vec::with_capacity(repeats);
    let mut assembly_times = Vec::with_capacity(repeats + 1);
    assembly_times.push(assembly_ms);
    let mut solve_times = Vec::with_capacity(repeats);
    let mut latest_residual = f64::INFINITY;
    let mut latest_relative_residual = f64::INFINITY;
    for _ in 0..repeats {
        let started = Instant::now();
        backend
            .assemble(&capture.entries)
            .map_err(|error| format!("{} warm assembly failed: {error}", layout_name(layout)))?;
        assembly_times.push(started.elapsed().as_secs_f64() * 1e3);
        let started = Instant::now();
        backend.factor().map_err(|error| {
            format!("{} warm factorization failed: {error}", layout_name(layout))
        })?;
        warm_factor_times.push(started.elapsed().as_secs_f64() * 1e3);

        let mut solution = capture.rhs.clone();
        let started = Instant::now();
        backend
            .solve_in_place(&mut solution)
            .map_err(|error| format!("{} solve failed: {error}", layout_name(layout)))?;
        solve_times.push(started.elapsed().as_secs_f64() * 1e3);
        (latest_residual, latest_relative_residual) = residual_norms(capture, &solution);
    }
    let assembly_ms = median(&mut assembly_times);
    let warm_factor_ms = median(&mut warm_factor_times);
    let solve_ms = median(&mut solve_times);
    let phase_snapshot = telemetry.snapshot();
    Ok(LinearKernelRow {
        implementation: "Rust native backend",
        layout: layout_name(layout),
        dimension: capture.dimension,
        mesh_nodes: capture.mesh_nodes,
        input_triplets: capture.entries.len(),
        canonical_coordinates: canonical_entry_counts(&capture.entries).0,
        numerical_nonzeros: canonical_entry_counts(&capture.entries).1,
        factor_l_nnz: "n/a (faer API)".into(),
        factor_u_nnz: "n/a (faer API)".into(),
        assembly_ms: fmt_ms(assembly_ms),
        cold_factor_ms: fmt_ms(cold_factor_ms),
        warm_factor_median_ms: fmt_ms(warm_factor_ms),
        fresh_factor_median_ms: "n/a".into(),
        solve_median_ms: fmt_ms(solve_ms),
        phase_factor_samples: (repeats + 1).to_string(),
        phase_sparse_symbolic_analyses: phase_snapshot.sparse_symbolic_analyses.to_string(),
        phase_sparse_symbolic_reuses: phase_snapshot.sparse_symbolic_reuses.to_string(),
        phase_sparse_numeric_factorizations: phase_snapshot
            .sparse_numeric_factorizations
            .to_string(),
        phase_banded_core_factorizations: phase_snapshot.banded_core_factorizations.to_string(),
        phase_banded_core_to_border_solves: phase_snapshot.banded_core_to_border_solves.to_string(),
        phase_banded_schur_assemblies: phase_snapshot.banded_schur_assemblies.to_string(),
        phase_banded_border_factorizations: phase_snapshot.banded_border_factorizations.to_string(),
        phase_sparse_triplet_ms: snapshot_time(phase_snapshot.sparse_triplet_materialization_ms),
        phase_sparse_csc_ms: snapshot_time(phase_snapshot.sparse_csc_assembly_ms),
        phase_sparse_symbolic_ms: snapshot_time(phase_snapshot.sparse_symbolic_analysis_ms),
        phase_sparse_symbolic_reuse_ms: snapshot_time(phase_snapshot.sparse_symbolic_reuse_ms),
        phase_sparse_numeric_factor_ms: snapshot_time(
            phase_snapshot.sparse_numeric_factorization_ms,
        ),
        phase_banded_core_factor_ms: snapshot_time(phase_snapshot.banded_core_factorization_ms),
        phase_banded_core_to_border_ms: snapshot_time(
            phase_snapshot.banded_core_to_border_solve_ms,
        ),
        phase_banded_schur_assembly_ms: snapshot_time(phase_snapshot.banded_schur_assembly_ms),
        phase_banded_border_factor_ms: snapshot_time(phase_snapshot.banded_border_factorization_ms),
        residual_inf: format!("{latest_residual:.3e}"),
        relative_residual_inf: format!("{latest_relative_residual:.3e}"),
    })
}

#[cfg(feature = "superlu")]
fn run_superlu_linear_kernel(
    capture: &BvpSciLinearSystemCapture,
    repeats: usize,
) -> Result<LinearKernelRow, String> {
    use faer::sparse::{SparseColMat, Triplet};

    let library_path = std::env::var_os("SUPERLU_LIBRARY")
        .ok_or_else(|| "SuperLU SciPy parity row requires SUPERLU_LIBRARY".to_string())?;
    let control = SolverLibrary::load(library_path)
        .map_err(|error| format!("could not load SuperLU bridge for SciPy parity row: {error}"))?;

    let started = Instant::now();
    let triplets = capture
        .entries
        .iter()
        .map(|&(row, column, value)| Triplet::new(row, column, value))
        .collect::<Vec<_>>();
    let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(
        capture.dimension,
        capture.dimension,
        &triplets,
    )
    .map_err(|error| format!("SuperLU parity CSC assembly failed: {error:?}"))?;
    let (symbolic, values) = matrix.as_ref().parts();
    let csc = SuperLuCscMatrix::try_from_usize(
        capture.dimension,
        symbolic.col_ptr(),
        symbolic.row_idx(),
        values,
    )
    .map_err(|error| format!("SuperLU parity ABI conversion failed: {error}"))?;
    let assembly_ms = started.elapsed().as_secs_f64() * 1e3;

    let started = Instant::now();
    let mut factor = control
        .factorize(&csc)
        .map_err(|error| format!("SuperLU parity cold factorization failed: {error}"))?;
    let cold_factor_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut latest_solution = capture.rhs.clone();
    let mut solve_times = Vec::with_capacity(repeats + 1);
    let solve_started = Instant::now();
    factor
        .solve_in_place(&mut latest_solution)
        .map_err(|error| format!("SuperLU parity initial solve failed: {error}"))?;
    solve_times.push(solve_started.elapsed().as_secs_f64() * 1e3);

    let mut fresh_factor_times = Vec::with_capacity(repeats);
    let mut latest_factor = factor;
    for _ in 0..repeats {
        let started = Instant::now();
        latest_factor = control
            .factorize(&csc)
            .map_err(|error| format!("SuperLU parity repeated factorization failed: {error}"))?;
        fresh_factor_times.push(started.elapsed().as_secs_f64() * 1e3);
        let mut solution = capture.rhs.clone();
        let started = Instant::now();
        latest_factor
            .solve_in_place(&mut solution)
            .map_err(|error| format!("SuperLU parity repeated solve failed: {error}"))?;
        solve_times.push(started.elapsed().as_secs_f64() * 1e3);
        latest_solution = solution;
    }
    let (_, relative_residual_inf) = residual_norms(capture, &latest_solution);
    let solve_ms = median(&mut solve_times);
    let fresh_factor_ms = median(&mut fresh_factor_times);

    Ok(LinearKernelRow {
        implementation: "Rust SuperLU bridge",
        layout: "CSC/SuperLU",
        dimension: capture.dimension,
        mesh_nodes: capture.mesh_nodes,
        input_triplets: capture.entries.len(),
        canonical_coordinates: csc.nnz(),
        numerical_nonzeros: canonical_entry_counts(&capture.entries).1,
        factor_l_nnz: format!(
            "storage={} csc={}",
            latest_factor.factor_l_storage_nnz, latest_factor.factor_l_csc_nnz
        ),
        factor_u_nnz: format!(
            "storage={} csc={}",
            latest_factor.factor_u_storage_nnz, latest_factor.factor_u_csc_nnz
        ),
        assembly_ms: fmt_ms(assembly_ms),
        cold_factor_ms: fmt_ms(cold_factor_ms),
        warm_factor_median_ms: "n/a (fresh symbolic+numeric)".into(),
        fresh_factor_median_ms: fmt_ms(fresh_factor_ms),
        solve_median_ms: fmt_ms(solve_ms),
        phase_factor_samples: (repeats + 1).to_string(),
        phase_sparse_symbolic_analyses: "n/a".into(),
        phase_sparse_symbolic_reuses: "n/a".into(),
        phase_sparse_numeric_factorizations: "n/a".into(),
        phase_banded_core_factorizations: "n/a".into(),
        phase_banded_core_to_border_solves: "n/a".into(),
        phase_banded_schur_assemblies: "n/a".into(),
        phase_banded_border_factorizations: "n/a".into(),
        phase_sparse_triplet_ms: "n/a".into(),
        phase_sparse_csc_ms: "n/a".into(),
        phase_sparse_symbolic_ms: "n/a".into(),
        phase_sparse_symbolic_reuse_ms: "n/a".into(),
        phase_sparse_numeric_factor_ms: "n/a".into(),
        phase_banded_core_factor_ms: "n/a".into(),
        phase_banded_core_to_border_ms: "n/a".into(),
        phase_banded_schur_assembly_ms: "n/a".into(),
        phase_banded_border_factor_ms: "n/a".into(),
        residual_inf: "n/a".into(),
        relative_residual_inf: format!("{relative_residual_inf:.3e}"),
    })
}

fn write_captured_csc_fixture(
    path: &std::path::Path,
    capture: &BvpSciLinearSystemCapture,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut content = format!(
        "{} {} {}\n",
        capture.dimension,
        capture.mesh_nodes,
        capture.entries.len()
    );
    for (index, value) in capture.rhs.iter().enumerate() {
        if index > 0 {
            content.push(' ');
        }
        content.push_str(&format!("{value:.17e}"));
    }
    content.push('\n');
    for (row, column, value) in &capture.entries {
        content.push_str(&format!("{row} {column} {value:.17e}\n"));
    }
    fs::write(path, content)
}

fn append_linear_kernel_rows(
    capture: Option<&BvpSciLinearSystemCapture>,
    rows: &mut Vec<LinearKernelRow>,
    failures: &mut Vec<String>,
) {
    let Some(capture) = capture else {
        failures.push("Dense exact-Jacobian solve did not capture the requested linear system (dimension >= 1000)".into());
        return;
    };
    if let Some(path) = std::env::var_os("BVP_SCI_CAPTURE_CSC_FIXTURE") {
        if let Err(error) = write_captured_csc_fixture(std::path::Path::new(&path), capture) {
            failures.push(format!("could not export captured CSC fixture: {error}"));
            return;
        }
    }
    let repeats = parse_env_usize("BVP_SCI_SCIPY_KERNEL_REPEATS", 5, 1);
    let python = match run_python_linear_kernel(capture, repeats) {
        Ok(fields) => fields,
        Err(error) => {
            failures.push(error);
            return;
        }
    };
    let rust_numerical_nonzeros = canonical_entry_counts(&capture.entries).1;
    if let Some(python_numerical_nonzeros) = python
        .get("csc_numerical_nonzeros")
        .and_then(|value| value.parse::<usize>().ok())
    {
        if python_numerical_nonzeros != rust_numerical_nonzeros {
            failures.push(format!(
                "canonical numerical-nonzero count differs for identical input: Rust {rust_numerical_nonzeros}, SciPy CSC {python_numerical_nonzeros}"
            ));
        }
    }
    for (key, value) in [
        ("dimension", capture.dimension.to_string()),
        ("mesh_nodes", capture.mesh_nodes.to_string()),
    ] {
        if python.get(key) != Some(&value) {
            failures.push(format!(
                "SciPy captured-kernel {key} mismatch: expected {value}, got {:?}",
                python.get(key)
            ));
        }
    }
    for layout in [
        BvpSciMatrixLayout::Dense,
        BvpSciMatrixLayout::Sparse,
        BvpSciMatrixLayout::Banded {
            lower: 256,
            upper: 256,
        },
    ] {
        match run_rust_linear_kernel(capture, layout, repeats) {
            Ok(row) => {
                if row
                    .relative_residual_inf
                    .parse::<f64>()
                    .unwrap_or(f64::INFINITY)
                    > 1e-8
                {
                    failures.push(format!(
                        "{} kernel residual is too large: {}",
                        row.layout, row.relative_residual_inf
                    ));
                }
                rows.push(row);
            }
            Err(error) => failures.push(error),
        }
    }
    #[cfg(feature = "superlu")]
    match run_superlu_linear_kernel(capture, repeats) {
        Ok(row) => {
            if row
                .relative_residual_inf
                .parse::<f64>()
                .unwrap_or(f64::INFINITY)
                > 1e-8
            {
                failures.push(format!(
                    "Rust SuperLU captured-kernel residual is too large: {}",
                    row.relative_residual_inf
                ));
            }
            rows.push(row);
        }
        Err(error) => failures.push(error),
    }
    let py_residual = python
        .get("relative_residual_inf")
        .and_then(|value| value.parse::<f64>().ok());
    if py_residual.is_none_or(|value| value > 1e-8) {
        failures.push(format!(
            "SciPy CSC kernel residual is not acceptable: {py_residual:?}"
        ));
    }
    rows.push(LinearKernelRow {
        implementation: "SciPy CSC/SuperLU",
        layout: "CSC/SuperLU",
        dimension: capture.dimension,
        mesh_nodes: capture.mesh_nodes,
        input_triplets: capture.entries.len(),
        canonical_coordinates: python
            .get("csc_stored_entries")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
        numerical_nonzeros: python
            .get("csc_numerical_nonzeros")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
        factor_l_nnz: python
            .get("factor_L_nnz")
            .cloned()
            .unwrap_or_else(|| "n/a".into()),
        factor_u_nnz: python
            .get("factor_U_nnz")
            .cloned()
            .unwrap_or_else(|| "n/a".into()),
        assembly_ms: python
            .get("assembly_median_ms")
            .cloned()
            .unwrap_or_else(|| "n/a".into()),
        cold_factor_ms: "n/a".into(),
        warm_factor_median_ms: "n/a".into(),
        fresh_factor_median_ms: python
            .get("factor_median_ms")
            .cloned()
            .unwrap_or_else(|| "n/a".into()),
        solve_median_ms: python
            .get("solve_median_ms")
            .cloned()
            .unwrap_or_else(|| "n/a".into()),
        phase_factor_samples: repeats.to_string(),
        phase_sparse_symbolic_analyses: "n/a".into(),
        phase_sparse_symbolic_reuses: "n/a".into(),
        phase_sparse_numeric_factorizations: "n/a".into(),
        phase_banded_core_factorizations: "n/a".into(),
        phase_banded_core_to_border_solves: "n/a".into(),
        phase_banded_schur_assemblies: "n/a".into(),
        phase_banded_border_factorizations: "n/a".into(),
        phase_sparse_triplet_ms: "n/a".into(),
        phase_sparse_csc_ms: "n/a".into(),
        phase_sparse_symbolic_ms: "n/a".into(),
        phase_sparse_symbolic_reuse_ms: "n/a".into(),
        phase_sparse_numeric_factor_ms: "n/a".into(),
        phase_banded_core_factor_ms: "n/a".into(),
        phase_banded_core_to_border_ms: "n/a".into(),
        phase_banded_schur_assembly_ms: "n/a".into(),
        phase_banded_border_factor_ms: "n/a".into(),
        residual_inf: python
            .get("residual_inf")
            .cloned()
            .unwrap_or_else(|| "n/a".into()),
        relative_residual_inf: python
            .get("relative_residual_inf")
            .cloned()
            .unwrap_or_else(|| "n/a".into()),
    });
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
    plan: &BvpSciLambdifyPlan,
    layout: BvpSciMatrixLayout,
    route: RustRoute,
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
        options.execution = if route.is_exact_numerical() {
            BvpSciExecution::Numerical
        } else {
            BvpSciExecution::Lambdify
        };
        options.assembly = Some(plan.assembly());
        options.matrix_layout = layout;
        options.tolerance = 1e-3;
        options.max_nodes = 1000;
        options.output_policy = BvpSciOutputPolicy::MeshAndValues;

        let construction_started = Instant::now();
        let boundary = boundary(case, problem.dimension, plan.telemetry().clone());
        #[cfg(feature = "superlu")]
        let solver_result = if route.uses_superlu() {
            BvpSciSolver::new_with_sparse_factorization(
                plan.clone(),
                boundary,
                problem.mesh.clone(),
                problem.initial_state.clone(),
                problem.parameters.clone(),
                options,
                RustedSciThe::numerical::BVP_sci::new::BvpSciSparseFactorization::SuperLu,
            )
        } else {
            BvpSciSolver::new(
                plan.clone(),
                boundary,
                problem.mesh.clone(),
                problem.initial_state.clone(),
                problem.parameters.clone(),
                options,
            )
        };
        #[cfg(not(feature = "superlu"))]
        let solver_result = BvpSciSolver::new(
            plan.clone(),
            boundary,
            problem.mesh.clone(),
            problem.initial_state.clone(),
            problem.parameters.clone(),
            options,
        );
        let mut solver =
            solver_result.map_err(|error| format!("Rust solver construction failed: {error}"))?;
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
    route: RustRoute,
    warmups: usize,
    repeats: usize,
) -> Result<CaseResult, String> {
    let python = benchmark_python(case, warmups, repeats)?;
    let (problem, prepare_ms) = prepare_problem(case, route.uses_symbolic_frontend())?;
    let exact_plan;
    let plan = match route {
        RustRoute::ExactNumerical => {
            exact_plan = make_numerical_plan(case, BvpSciTelemetry::disabled())?;
            &exact_plan
        }
        #[cfg(feature = "superlu")]
        RustRoute::ExprLegacySuperLu => problem
            .expr_plan
            .as_ref()
            .expect("SuperLU route prepares its symbolic plan"),
        RustRoute::ExprLegacy => problem
            .expr_plan
            .as_ref()
            .expect("ExprLegacy route prepares its symbolic plan"),
    };
    let (construct_ms, core_ms, total_ms, rust_status, rust_nodes, final_state) =
        benchmark_rust(case, &problem, plan, layout, route, warmups, repeats)?;
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
            rust_route: route.label(),
            rust_layout: route_layout_name(route, layout),
            dimension: problem.dimension,
            parameters: problem.parameter_count,
            scipy: format!("{} / NumPy {}", python.scipy_version, python.numpy_version),
            python_wall_median_ms: fmt_ms(python.median_ms),
            python_range_ms: format!("{:.3}..{:.3}", python.min_ms, python.max_ms),
            rust_expr_prepare_ms: if route.uses_symbolic_frontend() {
                fmt_ms(prepare_ms)
            } else {
                "n/a".into()
            },
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
            rust_solution_max_diff: "pending".into(),
            parity_gate: "status/nodes + solution vs exact Rust baseline",
        },
        final_state,
    })
}

fn main() {
    let warmups = parse_env_usize("BVP_SCI_SCIPY_BENCH_WARMUPS", 2, 0);
    let repeats = parse_env_usize("BVP_SCI_SCIPY_BENCH_REPEATS", 9, 1);
    let mut rows = Vec::with_capacity(CASES.len());
    let mut failures = Vec::new();
    let mut exact_reference = std::collections::HashMap::<&str, Vec<f64>>::new();
    for (case, layout, route) in CASES {
        match benchmark_case(case, *layout, *route, warmups, repeats) {
            Ok(mut result) => {
                if let Some(reference) = exact_reference.get(case) {
                    let max_diff = result
                        .final_state
                        .iter()
                        .zip(reference)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0_f64, f64::max);
                    result.row.rust_solution_max_diff = format!("{max_diff:.3e}");
                    if result.final_state.len() != reference.len() || max_diff > 1e-6 {
                        failures.push(format!(
                            "{case}/{route:?}/{layout:?}: Rust solution drift {max_diff:.3e}"
                        ));
                    }
                } else {
                    result.row.rust_solution_max_diff = "exact-reference".into();
                    exact_reference.insert(case, result.final_state.clone());
                }
                rows.push(result.row);
            }
            Err(error) => failures.push(format!("{case}/{route:?}/{layout:?}: {error}")),
        }
    }

    let mut diagnostic_rows = Vec::new();
    let mut mesh_pass_rows = Vec::new();
    let mut backend_mesh_rows = Vec::new();
    #[cfg(feature = "superlu")]
    let mut superlu_stage_rows = Vec::new();
    let mut captured_system = None;
    match python_diagnostic("big-parameterized-analytic") {
        Ok(python) => {
            for (layout, route) in [
                (BvpSciMatrixLayout::Dense, RustRoute::ExactNumerical),
                #[cfg(feature = "superlu")]
                (BvpSciMatrixLayout::Sparse, RustRoute::ExprLegacySuperLu),
                (BvpSciMatrixLayout::Sparse, RustRoute::ExprLegacy),
                (
                    BvpSciMatrixLayout::Banded {
                        lower: 256,
                        upper: 256,
                    },
                    RustRoute::ExprLegacy,
                ),
            ] {
                match run_route_diagnostic("big-parameterized-analytic", layout, route) {
                    Ok((rust, status, nodes, capture)) => {
                        backend_mesh_rows.extend(build_backend_mesh_rows(route, layout, &rust));
                        #[cfg(feature = "superlu")]
                        if route.uses_superlu() {
                            superlu_stage_rows.push(superlu_stage_row(&rust));
                        }
                        if capture.is_some() {
                            captured_system = capture;
                        }
                        let scipy_status = python
                            .get("status")
                            .and_then(|value| value.parse::<u8>().ok());
                        let scipy_nodes = python
                            .get("nodes")
                            .and_then(|value| value.parse::<usize>().ok());
                        if scipy_status != Some(status) || scipy_nodes != Some(nodes) {
                            failures.push(format!(
                                "diagnostic {layout:?}: status/nodes SciPy {scipy_status:?}/{scipy_nodes:?}, Rust {status}/{nodes}"
                            ));
                        }
                        if route.is_exact_numerical() {
                            for (python_counter, rust_counter, label) in [
                                (
                                    "counter.solve_newton_calls",
                                    rust.newton_scope_calls,
                                    "Newton passes",
                                ),
                                (
                                    "counter.construct_global_jac_calls",
                                    rust.newton_jacobian_refreshes,
                                    "global Jacobian builds",
                                ),
                                (
                                    "counter.factorization_calls",
                                    rust.factorizations,
                                    "factorizations",
                                ),
                                (
                                    "counter.linear_solve_calls",
                                    rust.linear_solves,
                                    "linear solves",
                                ),
                                (
                                    "counter.modify_mesh_calls",
                                    rust.mesh_refinements,
                                    "mesh modifications",
                                ),
                            ] {
                                let python_count = python
                                    .get(python_counter)
                                    .and_then(|value| value.parse::<u64>().ok());
                                if python_count != Some(rust_counter) {
                                    failures.push(format!(
                                    "diagnostic {layout:?}: {label} differ, SciPy {python_count:?}, Rust {rust_counter}"
                                ));
                                }
                            }
                            let rust_backtracking_trials = rust
                                .newton_residual_history
                                .iter()
                                .map(|entry| entry.backtracking_trials)
                                .sum::<u64>();
                            for (python_counter, rust_counter, label) in [
                                (
                                    "counter.boundary_callback_calls",
                                    rust.boundary_calls,
                                    "boundary residual callback calls",
                                ),
                                (
                                    "counter.boundary_jacobian_callback_calls",
                                    rust.boundary_jacobian_calls,
                                    "boundary Jacobian callback calls",
                                ),
                                (
                                    "counter.newton_trials",
                                    rust_backtracking_trials,
                                    "Newton backtracking trials",
                                ),
                            ] {
                                let python_count = python
                                    .get(python_counter)
                                    .and_then(|value| value.parse::<u64>().ok());
                                if python_count != Some(rust_counter) {
                                    failures.push(format!(
                                    "diagnostic {layout:?}: {label} differ, SciPy {python_count:?}, Rust {rust_counter}"
                                ));
                                }
                            }
                            let python_fd_probes = [
                                "counter.rhs_finite_difference_callback_calls",
                                "counter.boundary_finite_difference_callback_calls",
                            ]
                            .iter()
                            .filter_map(|key| python.get(*key))
                            .filter_map(|value| value.parse::<u64>().ok())
                            .sum::<u64>();
                            if python_fd_probes != rust.finite_difference_probes {
                                failures.push(format!(
                                "diagnostic {layout:?}: finite-difference probes differ, SciPy {python_fd_probes}, Rust {}",
                                rust.finite_difference_probes
                            ));
                            }
                        }
                        match build_stage_rows(
                            "big-parameterized-analytic",
                            &python,
                            layout,
                            route,
                            60,
                            2,
                            nodes,
                            &rust,
                        ) {
                            Ok(mut stages) => diagnostic_rows.append(&mut stages),
                            Err(error) => failures.push(format!(
                                "diagnostic {layout:?}: could not build stage rows: {error}"
                            )),
                        }
                        match build_mesh_pass_rows(
                            &python,
                            route,
                            layout_name(layout),
                            &rust,
                            &mut failures,
                            route.is_exact_numerical(),
                        ) {
                            Ok(mut passes) => mesh_pass_rows.append(&mut passes),
                            Err(error) => failures.push(format!(
                                "diagnostic {layout:?}: could not build mesh-pass rows: {error}"
                            )),
                        }
                    }
                    Err(error) => {
                        failures.push(format!("diagnostic {route:?}/{layout:?}: {error}"))
                    }
                }
            }
        }
        Err(error) => failures.push(format!("Python stage diagnostic: {error}")),
    }

    let mut linear_kernel_rows = Vec::new();
    append_linear_kernel_rows(
        captured_system.as_ref(),
        &mut linear_kernel_rows,
        &mut failures,
    );

    let table = Table::new(&rows);
    let stage_table = Table::new(&diagnostic_rows);
    let mesh_pass_table = Table::new(&mesh_pass_rows);
    let backend_mesh_table = Table::new(&backend_mesh_rows);
    let linear_kernel_table = Table::new(&linear_kernel_rows);
    let report = format!(
        "# BVP_sci Python versus Rust numerical-core benchmark\n\n\
         - pinned reference: official SciPy 1.18.1 / NumPy 2.5.4; uninstrumented Python reference source\n\
         - the Dense Numerical row uses analytical state, parameter and boundary Jacobians in both implementations; ExprLegacy uses exact symbolic state and parameter derivatives with analytic boundary callbacks\n\
         - Dense Numerical is the strict callback-route comparison against Python. ExprLegacy/SuperLU is a separate symbolic frontend route with mathematically exact derivatives; its Rust numerical-core timing is useful context, not an apples-to-apples callback-implementation claim\n\
         - Python timer covers each `solve_bvp` call; callback/problem construction and process startup are excluded\n\
         - Rust `numerical_core` covers `BvpSciSolver::solve`; solver construction and one-time ExprLegacy preparation are separate\n\
         - `construct + core` is the closer lifecycle comparison; neither Rust column includes ExprLegacy preparation\n\
         - Rust is measured through Dense Numerical and ExprLegacy routes; with Cargo feature `superlu`, the official 60-state ExprLegacy/exact-symbolic Sparse/SuperLU full solve is added and fallback is rejected. ExprLegacy diagnostics also cover Sparse/faer and bordered-Banded; SciPy uses CSC/SuperLU\n\
         - warmups: {warmups}; measured repeats: {repeats}; cells are medians unless the column says range\n\
         - every Rust route/layout must match SciPy's successful status/node count and match the first exact Rust final state within 1e-6\n\
         - full SciPy value/derivative/defect and counter semantics are gated by the dedicated debug parity stories\n\n\
         ## Uninstrumented Release Timing\n\n\
         {}\n\n\
         ## Instrumented Stage Diagnostics\n\n\
         This separate single-solve diagnostic uses opt-in Python/Rust telemetry; its timings include instrumentation overhead and must not be compared to the release timing table. It reports the exact-Jacobian Dense Numerical route, ExprLegacy/SuperLU, and ExprLegacy Sparse/Banded routes. Only Dense Numerical is strict callback/counter parity; symbolic routes provide stage attribution because callback construction and dispatch differ. Parent/child times are inclusive and non-additive; dedicated callback timings are attribution views and overlap aggregate evaluation scopes. RHS/Jacobian work units count evaluated state points so Python vectorized calls can be compared with Rust point callbacks. Callback work differences are reported, not treated as parity failures, because each implementation may evaluate or cache callback values at different points in the algorithm. `calls` intentionally exposes dispatch granularity.\n\n\
         {}\n\n\
         ## Per-Mesh-Pass Work and Timing\n\n\
         Rows are indexed by outer solve pass and report the mesh/system dimension seen by that pass, so the first work-count divergence is visible instead of being hidden by equal totals. Stage timings are matched by operation where semantics permit; pass elapsed scopes are diagnostic only (SciPy starts at `prepare_sys`, Rust at the current-mesh solve boundary), so do not compare those two columns as equivalent wall-clock scopes. All timings are inclusive attribution and non-additive.\n\n\
         {}\n\n\
         ## Same-Matrix Linear Kernel\n\n\
         One captured Dense Numerical Newton matrix and its actual RHS are replayed through Rust Dense/Sparse/Banded and SciPy CSC/SuperLU. Assembly and factorization are separated; `cold_factor_ms` is the first Rust factorization, `warm_factor_median_ms` is repeated Rust factorization after backend setup, and SciPy's fresh CSC/SuperLU factor median is shown separately. `phase_*` fields are opt-in internal telemetry accumulated over the Rust cold plus warm factor calls; their sample counts are printed and their timings include instrumentation overhead. They isolate CSC construction, symbolic reuse, and bordered-Banded Schur phases, while the outer medians remain the backend wall timings. This isolates linear kernels from differences in symbolic Jacobian/frontends. Residuals are checked against the original triplets/RHS, with duplicate triplets summed by each native representation. `canonical_coordinates` may include explicit zeros; `numerical_nonzeros` counts only nonzero sums and is cross-checked across implementations.\n\n\
         {}\n\n\
         ## Sparse/Banded Backend Per-Mesh Details\n\n\
         Backend-specific reuse and fallback counters are separated from the general SciPy parity table to keep the per-pass report readable. `n/a` means that the stage does not apply to the selected native layout or was not timed.\n\n\
         {}\n\n\
         {}",
        table,
        stage_table,
        mesh_pass_table,
        linear_kernel_table,
        backend_mesh_table,
        if failures.is_empty() {
            "All matched timing rows completed successfully.".to_string()
        } else {
            format!("Benchmark row failures:\n\n- {}", failures.join("\n- "))
        }
    );
    #[cfg(feature = "superlu")]
    let report = format!(
        "{report}\n\n## Sparse/SuperLU Full-Solve Stage Attribution\n\nThis is one instrumented solve, separate from the repeated uninstrumented timing table. It records the Rust CSC assembly, ABI conversion, FFI and native factorization/solve scopes, and fallback count. The SciPy per-mesh stage table above provides corresponding total assembly/factor/solve scopes; the boundary between Rust bridge/assembly and SciPy's CSC construction is not identical.\n\n{}\n\nWith Cargo feature `superlu`, the official 60-state ExprLegacy route uses exact symbolic state and parameter derivatives with the Sparse/SuperLU backend. The preflight rejects fallback and gates SciPy status, mesh and common Newton/factor/solve counters. The symbolic Rust callback route is not claimed to match Python callback implementation, and the result does not establish a universal performance win.\n",
        Table::new(&superlu_stage_rows)
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
