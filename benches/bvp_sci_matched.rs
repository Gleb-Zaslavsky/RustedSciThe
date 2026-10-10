//! Matched AOT/Lambdify continuation dashboard for BVP_sci.
//!
//! This is a manual compact report, not a Criterion warm-up loop. Each row
//! prepares one frontend/layout once, then repeats the same nonzero solve and
//! parameter-continuation series. Preparation is therefore visible separately
//! from the warm numerical work and cannot be mistaken for callback speed.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan,
    BvpSciMatrixLayout, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
};
use RustedSciThe::symbolic::ivp_telemetry::IvpColdStage;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use tabled::{Table, Tabled};

#[derive(Clone, Copy)]
enum Route {
    LambdifyExpr,
    LambdifyAtom,
    AotExpr,
    AotAtom,
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::LambdifyExpr => "Lambdify-ExprLegacy",
            Self::LambdifyAtom => "Lambdify-AtomViewNative",
            Self::AotExpr => "AOT-ExprLegacy",
            Self::AotAtom => "AOT-AtomViewNative",
        }
    }

    fn assembly(self) -> BvpSciAssembly {
        match self {
            Self::LambdifyExpr | Self::AotExpr => BvpSciAssembly::ExprLegacy,
            Self::LambdifyAtom | Self::AotAtom => BvpSciAssembly::AtomViewNative,
        }
    }

    fn is_aot(self) -> bool {
        matches!(self, Self::AotExpr | Self::AotAtom)
    }

    fn preparation_scope(self) -> &'static str {
        if self.is_aot() {
            "aot:BuildIfMissing+compile+link+publish"
        } else {
            "lambdify:symbolic+evaluator"
        }
    }
}

#[derive(Clone, Copy)]
enum Layout {
    Dense,
    Sparse,
    Banded,
}

impl Layout {
    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
    }

    fn value(self) -> BvpSciMatrixLayout {
        match self {
            Self::Dense => BvpSciMatrixLayout::Dense,
            Self::Sparse => BvpSciMatrixLayout::Sparse,
            Self::Banded => BvpSciMatrixLayout::Banded { lower: 1, upper: 1 },
        }
    }
}

#[derive(Debug, Tabled)]
struct Row {
    route: String,
    layout: String,
    dimension: usize,
    nodes: usize,
    continuation_count: usize,
    repeats: usize,
    prepare_scope: String,
    prepare_ms: String,
    aot_prepare_ms: String,
    warm_series_ms: String,
    continuation_ms: String,
    amortized_solve_ms: String,
    full_solve_ms: String,
    callback_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    factorization_ms: String,
    residual_calls: u64,
    jacobian_calls: u64,
    factorizations: u64,
    cache_hits: u64,
    cache_misses: u64,
    build_attempts: u64,
    link_attempts: u64,
    runtime_ready: u64,
    parity_diff: String,
    status: String,
}

struct RunResult {
    row: Row,
    final_state: Vec<f64>,
}

fn parse_usizes(name: &str, default: &str, minimum: usize) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value >= minimum)
        .collect()
}

fn parse_routes() -> Vec<Route> {
    std::env::var("BVP_SCI_MATCHED_FRONTENDS")
        .unwrap_or_else(|_| {
            "lambdify-expr-legacy,lambdify-atom-native,aot-expr-legacy,aot-atom-native".into()
        })
        .split(',')
        .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "lambdify-expr-legacy" => Some(Route::LambdifyExpr),
            "lambdify-atom-native" => Some(Route::LambdifyAtom),
            "aot-expr-legacy" => Some(Route::AotExpr),
            "aot-atom-native" => Some(Route::AotAtom),
            _ => None,
        })
        .collect()
}

fn parse_layouts() -> Vec<Layout> {
    std::env::var("BVP_SCI_MATCHED_LAYOUTS")
        .unwrap_or_else(|_| "dense,sparse,banded".into())
        .split(',')
        .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "dense" => Some(Layout::Dense),
            "sparse" => Some(Layout::Sparse),
            "banded" => Some(Layout::Banded),
            _ => None,
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn format_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".into())
}

fn equations(dimension: usize) -> (Vec<Expr>, Vec<String>) {
    let states = (0..dimension)
        .map(|index| format!("y{index}"))
        .collect::<Vec<_>>();
    let equations = states
        .iter()
        .map(|state| Expr::parse_expression(&format!("p*({state} + 1)")))
        .collect();
    (equations, states)
}

fn initial_state(dimension: usize, nodes: usize, parameter: f64) -> Vec<f64> {
    (0..nodes)
        .flat_map(|node| {
            let x = node as f64 / (nodes - 1) as f64;
            let exact = (parameter * x).exp() - 1.0;
            (0..dimension).map(move |_| 0.90 * exact + 0.01 * (std::f64::consts::PI * x).sin())
        })
        .collect()
}

fn boundary(dimension: usize) -> BvpSciBoundaryCallbacks {
    BvpSciBoundaryCallbacks::new(
        dimension + 1,
        move |ya, yb, parameters, output| {
            output[..dimension].copy_from_slice(&ya[..dimension]);
            let parameter = parameters.first().copied().unwrap_or(0.0);
            output[dimension] = yb[0] - (parameter.exp() - 1.0);
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    )
}

fn prepare(
    route: Route,
    layout: Layout,
    dimension: usize,
    output_root: &PathBuf,
    compiler: &str,
) -> Result<(BvpSciLambdifyPlan, f64), String> {
    let (equations, states) = equations(dimension);
    let telemetry = BvpSciTelemetry::timings();
    let started = Instant::now();
    let plan = if route.is_aot() {
        let mut config =
            RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                output_root
                    .join(route.label())
                    .join(layout.label())
                    .join(dimension.to_string()),
            );
        config = if compiler.eq_ignore_ascii_case("gcc") {
            config.with_c_gcc()
        } else {
            config.with_c_tcc()
        };
        BvpSciLambdifyPlan::prepare_aot(
            route.assembly(),
            layout.value(),
            equations,
            states,
            vec!["p".into()],
            "x",
            config,
            telemetry,
        )
    } else {
        BvpSciLambdifyPlan::prepare(
            route.assembly(),
            &equations,
            &states,
            &["p".into()],
            "x",
            telemetry,
        )
    }
    .map_err(|error| error.to_string())?;
    Ok((plan, started.elapsed().as_secs_f64() * 1e3))
}

fn run_row(
    route: Route,
    layout: Layout,
    dimension: usize,
    nodes: usize,
    continuation_count: usize,
    repeats: usize,
    output_root: &PathBuf,
    compiler: &str,
) -> RunResult {
    let (plan, prepare_ms) = match prepare(route, layout, dimension, output_root, compiler) {
        Ok(value) => value,
        Err(error) => {
            return RunResult {
                row: error_row(
                    route,
                    layout,
                    dimension,
                    nodes,
                    continuation_count,
                    repeats,
                    error,
                ),
                final_state: Vec::new(),
            };
        }
    };
    let mut warm_series = Vec::with_capacity(repeats);
    let mut continuation_times = Vec::with_capacity(repeats);
    let mut last_snapshot = None;
    let mut final_state = Vec::new();
    let initial_parameter = 0.5;
    let mesh = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect::<Vec<_>>();
    let initial_state = initial_state(dimension, nodes, initial_parameter);
    let mut status = "ok".to_string();

    for _ in 0..repeats.max(1) {
        let mut options = BvpSciOptions::default();
        options.execution = if route.is_aot() {
            BvpSciExecution::Aot
        } else {
            BvpSciExecution::Lambdify
        };
        options.assembly = Some(route.assembly());
        options.matrix_layout = layout.value();
        options.max_nodes = nodes.saturating_mul(16).max(128);
        options.max_mesh_refinements = 8;
        options.max_newton_iterations = 20;
        options.max_jacobian_refreshes = 30;
        options.tolerance = 1e-5;
        let mut solver = match BvpSciSolver::new(
            plan.clone(),
            boundary(dimension),
            mesh.clone(),
            initial_state.clone(),
            vec![initial_parameter],
            options,
        ) {
            Ok(solver) => solver,
            Err(error) => {
                status = format!("error: construct: {error}");
                break;
            }
        };
        let series_started = Instant::now();
        match solver.solve() {
            Ok(solution) => final_state = solution.y,
            Err(error) => {
                status = format!("error: initial solve: {error}");
                break;
            }
        }
        let continuation_started = Instant::now();
        for index in 1..continuation_count {
            let parameter = if continuation_count <= 1 {
                initial_parameter
            } else {
                initial_parameter + 0.25 * index as f64 / (continuation_count - 1) as f64
            };
            if let Err(error) = solver.set_parameters(vec![parameter]) {
                status = format!("error: rebind: {error}");
                break;
            }
            match solver.solve() {
                Ok(solution) => final_state = solution.y,
                Err(error) => {
                    status = format!("error: continuation solve: {error}");
                    break;
                }
            }
        }
        if status != "ok" {
            break;
        }
        warm_series.push(series_started.elapsed().as_secs_f64() * 1e3);
        continuation_times.push(continuation_started.elapsed().as_secs_f64() * 1e3);
        last_snapshot = Some(solver.plan().telemetry_snapshot());
    }

    let snapshot = last_snapshot;
    let mut warm_series_for_median = warm_series.clone();
    let mut continuation_for_median = continuation_times.clone();
    let warm_ms = if warm_series_for_median.is_empty() {
        None
    } else {
        Some(median(&mut warm_series_for_median))
    };
    let continuation_ms = if continuation_for_median.is_empty() {
        None
    } else {
        Some(median(&mut continuation_for_median))
    };
    let aot = snapshot.as_ref().and_then(|value| value.aot.as_ref());
    let row = Row {
        route: route.label().into(),
        layout: layout.label().into(),
        dimension,
        nodes,
        continuation_count,
        repeats: repeats.max(1),
        prepare_scope: route.preparation_scope().into(),
        prepare_ms: format!("{prepare_ms:.3}"),
        aot_prepare_ms: snapshot
            .as_ref()
            .and_then(|value| value.aot.as_ref())
            .map(|value| {
                format!(
                    "{:.3}",
                    value
                        .cold_stage(IvpColdStage::SolverPreparation)
                        .elapsed
                        .as_secs_f64()
                        * 1e3
                )
            })
            .unwrap_or_else(|| "-".into()),
        warm_series_ms: format_ms(warm_ms),
        continuation_ms: format_ms(continuation_ms),
        amortized_solve_ms: warm_ms
            .map(|value| {
                format!(
                    "{:.3}",
                    (prepare_ms + value) / continuation_count.max(1) as f64
                )
            })
            .unwrap_or_else(|| "-".into()),
        full_solve_ms: format_ms(snapshot.as_ref().and_then(|value| value.full_solve_ms)),
        callback_ms: format_ms(snapshot.as_ref().and_then(|value| value.callback_ms)),
        residual_ms: format_ms(
            snapshot
                .as_ref()
                .and_then(|value| value.residual_evaluation_ms),
        ),
        jacobian_ms: format_ms(
            snapshot
                .as_ref()
                .and_then(|value| value.jacobian_evaluation_ms),
        ),
        factorization_ms: format_ms(snapshot.as_ref().and_then(|value| value.factorization_ms)),
        residual_calls: snapshot
            .as_ref()
            .map_or(0, |value| value.residual_evaluations),
        jacobian_calls: snapshot
            .as_ref()
            .map_or(0, |value| value.jacobian_evaluations),
        factorizations: snapshot.as_ref().map_or(0, |value| value.factorizations),
        cache_hits: aot.map_or(0, |value| value.aot_resolution_hits),
        cache_misses: aot.map_or(0, |value| value.aot_resolution_misses),
        build_attempts: aot.map_or(0, |value| value.aot_build_attempts),
        link_attempts: aot.map_or(0, |value| value.aot_link_attempts),
        runtime_ready: aot.map_or(0, |value| value.aot_runtime_ready),
        parity_diff: "pending".into(),
        status,
    };
    RunResult { row, final_state }
}

fn error_row(
    route: Route,
    layout: Layout,
    dimension: usize,
    nodes: usize,
    continuation_count: usize,
    repeats: usize,
    error: String,
) -> Row {
    Row {
        route: route.label().into(),
        layout: layout.label().into(),
        dimension,
        nodes,
        continuation_count,
        repeats: repeats.max(1),
        prepare_scope: route.preparation_scope().into(),
        prepare_ms: "-".into(),
        aot_prepare_ms: "-".into(),
        warm_series_ms: "-".into(),
        continuation_ms: "-".into(),
        amortized_solve_ms: "-".into(),
        full_solve_ms: "-".into(),
        callback_ms: "-".into(),
        residual_ms: "-".into(),
        jacobian_ms: "-".into(),
        factorization_ms: "-".into(),
        residual_calls: 0,
        jacobian_calls: 0,
        factorizations: 0,
        cache_hits: 0,
        cache_misses: 0,
        build_attempts: 0,
        link_attempts: 0,
        runtime_ready: 0,
        parity_diff: "-".into(),
        status: format!("error: {error}"),
    }
}

fn main() {
    let dimensions = parse_usizes("BVP_SCI_MATCHED_DIMENSIONS", "8,32", 2);
    let nodes = parse_usizes("BVP_SCI_MATCHED_NODES", "16,64", 3);
    let counts = parse_usizes("BVP_SCI_MATCHED_COUNTS", "1,4,16", 1);
    let repeats = std::env::var("BVP_SCI_MATCHED_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);
    let routes = parse_routes();
    let layouts = parse_layouts();
    let compiler = std::env::var("BVP_SCI_MATCHED_COMPILER").unwrap_or_else(|_| "tcc".into());
    let output_root = std::env::var_os("BVP_SCI_MATCHED_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/bvp-sci-matched"));
    let mut rows = Vec::new();
    let mut references = HashMap::<(usize, usize, String, usize), Vec<f64>>::new();

    for &dimension in &dimensions {
        for &node_count in &nodes {
            for &layout in &layouts {
                for &count in &counts {
                    for &route in &routes {
                        let mut result = run_row(
                            route,
                            layout,
                            dimension,
                            node_count,
                            count,
                            repeats,
                            &output_root,
                            &compiler,
                        );
                        let key = (dimension, node_count, layout.label().into(), count);
                        if result.row.status == "ok" {
                            if let Some(reference) = references.get(&key) {
                                let diff = result
                                    .final_state
                                    .iter()
                                    .zip(reference)
                                    .map(|(left, right)| (left - right).abs())
                                    .fold(0.0, f64::max);
                                result.row.parity_diff = format!("{diff:.3e}");
                            } else {
                                references.insert(key, result.final_state.clone());
                                result.row.parity_diff = "0.000e0".into();
                            }
                        }
                        rows.push(result.row);
                    }
                }
            }
        }
    }
    assert!(!rows.is_empty(), "matched benchmark selected no rows");
    let table = Table::new(&rows).to_string();
    let report = format!(
        "# BVP_sci matched AOT/Lambdify full-solve continuation\n\n- dimensions: {dimensions:?}\n- nodes: {nodes:?}\n- continuation counts: {counts:?}\n- repeats per prepared route: {repeats}\n- nonzero manufactured family: `y' = p*(y + 1)`, `y(0)=0`, `y(1)=exp(p)-1`\n- each row prepares one route/layout/count; AOT cache hit/miss columns distinguish reused artifacts\n- `warm_series_ms` includes the initial solve and the continuation solves\n- `amortized_solve_ms = (prepare_ms + warm_series_ms) / continuation_count`; it is a decision aid, not a statistical threshold\n- all timing scopes are diagnostic and non-additive; `full_solve_ms` is the inclusive latest public solve\n\n{table}\n",
    );
    let name = std::env::var("BVP_SCI_MATCHED_REPORT")
        .unwrap_or_else(|_| "matched_full_solve_continuation".into());
    let path = write_test_report("BVP_sci_Matched_Bench", &name, &report)
        .expect("write compact matched BVP_sci report");
    println!("report={}", path.display());
    if rows.iter().any(|row| row.status.starts_with("error:"))
        && std::env::var("BVP_SCI_MATCHED_FAIL_ON_ERROR").as_deref() == Ok("1")
    {
        std::process::exit(1);
    }
}
