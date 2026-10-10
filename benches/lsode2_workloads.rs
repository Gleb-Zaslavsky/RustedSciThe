//! Compact, non-Criterion LSODE2 release matrix.
//!
//! This target intentionally emits one Tabled report instead of Criterion's
//! warm-up/sample chatter. The existing Criterion targets remain the detailed
//! statistical instruments; this binary is the bounded release smoke/matrix.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use tabled::{Table, Tabled};
use tempfile::TempDir;

use RustedSciThe::numerical::LSODE2::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, Lsode2AotProfile, Lsode2AotToolchain,
    Lsode2ControllerConfig, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ivp_workloads::{
    SymbolicWorkload, WorkloadKind, build_workload, parameter_continuation_target,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};

#[derive(Debug, Tabled)]
struct Lsode2CompactRow {
    workload: String,
    dimension: String,
    frontend: String,
    execution: String,
    linear_backend: String,
    policy: String,
    prepare_ms: String,
    solve_ms: String,
    continuation_ms: String,
    symbolic_jacobian_ms: String,
    atom_jacobian_ms: String,
    aot_build_ms: String,
    aot_link_ms: String,
    factorization_ms: String,
    residual_calls: usize,
    jacobian_calls: usize,
    factorizations: usize,
    accepted_steps: usize,
    workers: usize,
    parallel_dispatches: u64,
    sequential_dispatches: u64,
    aot_chunks: u64,
    max_final_abs: String,
    parity_diff: String,
    status: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixRoute {
    Dense,
    Sparse,
    Banded,
}

impl MatrixRoute {
    const ALL: [Self; 3] = [Self::Dense, Self::Sparse, Self::Banded];

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense_lu",
            Self::Sparse => "faer_sparse_lu",
            Self::Banded => "lapack_faithful_banded_lu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrontendRoute {
    ExprLegacy,
    AtomView,
}

impl FrontendRoute {
    const ALL: [Self; 2] = [Self::ExprLegacy, Self::AtomView];

    fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "expr-legacy",
            Self::AtomView => "atom-view",
        }
    }

    fn assembly(self) -> Lsode2SymbolicAssemblyBackend {
        match self {
            Self::ExprLegacy => Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Self::AtomView => Lsode2SymbolicAssemblyBackend::AtomView,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionRoute {
    Lambdify(IvpLambdifyExecutionPolicy),
    AotWhole,
    AotParallel,
}

impl ExecutionRoute {
    fn label(self) -> &'static str {
        match self {
            Self::Lambdify(IvpLambdifyExecutionPolicy::Sequential) => "lambdify-sequential",
            Self::Lambdify(IvpLambdifyExecutionPolicy::Parallel { .. }) => "lambdify-parallel",
            Self::Lambdify(IvpLambdifyExecutionPolicy::Auto { .. }) => "lambdify-auto",
            Self::AotWhole => "aot-whole",
            Self::AotParallel => "aot-parallel2",
        }
    }

    fn is_aot(self) -> bool {
        matches!(self, Self::AotWhole | Self::AotParallel)
    }

    fn policy(self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::Lambdify(policy) => policy,
            Self::AotWhole | Self::AotParallel => IvpLambdifyExecutionPolicy::Sequential,
        }
    }

    fn execution(self) -> Lsode2SymbolicExecutionMode {
        if self.is_aot() {
            Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            }
        } else {
            Lsode2SymbolicExecutionMode::LambdifyExpr
        }
    }
}

fn labels(variable: &str, defaults: &[&str]) -> Vec<String> {
    let Some(value) = std::env::var_os(variable) else {
        return defaults.iter().map(|value| (*value).to_owned()).collect();
    };
    let values = value
        .to_string_lossy()
        .split(',')
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        defaults.iter().map(|value| (*value).to_owned()).collect()
    } else {
        values
    }
}

fn workloads() -> Vec<WorkloadKind> {
    let selected = labels(
        "LSODE2_BENCH_COMPACT_WORKLOADS",
        &[
            "stiff-scalar",
            "robertson",
            "combustion-like",
            "three-body",
            "diffusion-chain",
        ],
    );
    WorkloadKind::ALL
        .into_iter()
        .filter(|kind| selected.iter().any(|value| value == kind.label()))
        .collect()
}

fn dimensions(kind: WorkloadKind) -> Vec<usize> {
    let defaults = match kind {
        WorkloadKind::DiffusionChain => vec![32, 128],
        WorkloadKind::CombustionLike | WorkloadKind::Robertson => vec![3],
        WorkloadKind::StiffScalar => vec![1],
        WorkloadKind::ThreeBody => vec![12],
    };
    let variable = if matches!(kind, WorkloadKind::DiffusionChain) {
        "LSODE2_BENCH_COMPACT_DIFFUSION_DIMENSIONS"
    } else {
        "LSODE2_BENCH_COMPACT_DIMENSIONS"
    };
    std::env::var(variable)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or(defaults)
}

fn matrices() -> Vec<MatrixRoute> {
    let selected = labels(
        "LSODE2_BENCH_COMPACT_MATRICES",
        &["dense", "sparse", "banded"],
    );
    MatrixRoute::ALL
        .into_iter()
        .filter(|route| {
            selected.iter().any(|value| {
                (*route == MatrixRoute::Dense && value == "dense")
                    || (*route == MatrixRoute::Sparse && value == "sparse")
                    || (*route == MatrixRoute::Banded && value == "banded")
            })
        })
        .collect()
}

fn routes() -> Vec<ExecutionRoute> {
    labels(
        "LSODE2_BENCH_COMPACT_ROUTES",
        &["lambdify-sequential", "lambdify-auto", "aot-whole"],
    )
    .into_iter()
    .filter_map(|value| match value.as_str() {
        "lambdify" | "lambdify-sequential" => Some(ExecutionRoute::Lambdify(
            IvpLambdifyExecutionPolicy::Sequential,
        )),
        "lambdify-parallel" => Some(ExecutionRoute::Lambdify(
            IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        )),
        "lambdify-auto" => Some(ExecutionRoute::Lambdify(IvpLambdifyExecutionPolicy::Auto {
            min_work: 1,
        })),
        "aot" | "aot-whole" => Some(ExecutionRoute::AotWhole),
        "aot-parallel" | "aot-parallel2" => Some(ExecutionRoute::AotParallel),
        other => panic!("unknown LSODE2_BENCH_COMPACT_ROUTES item {other:?}"),
    })
    .collect()
}

fn continuation_counts() -> Vec<usize> {
    labels("LSODE2_BENCH_COMPACT_CONTINUATION_COUNTS", &["1", "4"])
        .into_iter()
        .filter_map(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .collect()
}

fn generated_backend(output: &Path, route: ExecutionRoute) -> SymbolicIvpGeneratedBackendConfig {
    let policy = SymbolicIvpAotBuildPolicy::RebuildAlways {
        profile: AotBuildProfile::Release,
    };
    let mut config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(output.to_path_buf()))
        .with_build_policy(policy)
        .with_c_tcc();
    if matches!(route, ExecutionRoute::AotParallel) {
        config = config.with_aot_options(
            RustedSciThe::symbolic::symbolic_ivp::SymbolicIvpAotOptions {
                residual_strategy: RustedSciThe::symbolic::codegen::codegen_runtime_api::
                    recommended_residual_chunking_for_parallelism(128, 2),
                jacobian_strategy: RustedSciThe::symbolic::codegen::codegen_runtime_api::
                    recommended_dense_jacobian_chunking_for_parallelism(128, 2),
            },
        );
    }
    config
}

fn make_config(
    workload: SymbolicWorkload,
    matrix: MatrixRoute,
    frontend: FrontendRoute,
    route: ExecutionRoute,
    output: Option<&TempDir>,
) -> Lsode2ProblemConfig {
    let dimension = workload.equations.len();
    let mut config = Lsode2ProblemConfig::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.25,
        0.02,
        1.0e-7,
        1.0e-9,
    )
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_lambdify_execution_policy(route.policy())
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: frontend.assembly(),
        execution: route.execution(),
    })
    .with_telemetry(IvpTelemetry::detailed());

    if !workload.parameter_names.is_empty() {
        config = config
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }

    config = match matrix {
        MatrixRoute::Dense => config.with_dense_symbolic_backend(),
        MatrixRoute::Sparse => config.with_native_sparse_faer_backend(),
        MatrixRoute::Banded => config
            .with_native_banded_faithful_backend()
            .with_linear_system_structure(Lsode2LinearSystemStructure::Banded {
                kl: if workload.kind == WorkloadKind::DiffusionChain {
                    1
                } else {
                    dimension.saturating_sub(1)
                },
                ku: if workload.kind == WorkloadKind::DiffusionChain {
                    1
                } else {
                    dimension.saturating_sub(1)
                },
            }),
    };

    if route.is_aot() {
        let output = output.expect("AOT compact route requires an isolated output directory");
        config = match matrix {
            MatrixRoute::Dense => config.with_dense_aot_c_tcc(output.path()),
            MatrixRoute::Sparse => config
                .with_native_sparse_faer_generated_backend(generated_backend(output.path(), route)),
            MatrixRoute::Banded => config.with_native_banded_faithful_generated_backend(
                generated_backend(output.path(), route),
            ),
        };
        if matches!(route, ExecutionRoute::AotParallel) {
            config = config.with_aot_parallel_chunking(2);
        }
    }
    config
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn stage_ms(
    snapshot: &RustedSciThe::numerical::LSODE2::IvpTelemetrySnapshot,
    stage: IvpColdStage,
) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1_000.0
}

fn fmt_ms(value: f64) -> String {
    format!("{value:.3}")
}

fn state_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn run_case(
    workload: WorkloadKind,
    dimension: usize,
    matrix: MatrixRoute,
    frontend: FrontendRoute,
    route: ExecutionRoute,
    counts: &[usize],
) -> (Lsode2CompactRow, Option<Vec<f64>>) {
    let output = route
        .is_aot()
        .then(|| tempfile::tempdir().expect("LSODE2 AOT output"));
    let workload_data = build_workload(workload, dimension);
    let parameter_values = workload_data.parameter_values.clone();
    let parameterized = !parameter_values.is_empty();
    let label_dimension = if workload == WorkloadKind::DiffusionChain {
        dimension.to_string()
    } else {
        "fixed".to_owned()
    };
    let base = |status: String| Lsode2CompactRow {
        workload: workload.label().to_owned(),
        dimension: label_dimension.clone(),
        frontend: frontend.label().to_owned(),
        execution: route.label().to_owned(),
        linear_backend: matrix.label().to_owned(),
        policy: route.policy().label().to_owned(),
        prepare_ms: "-".to_owned(),
        solve_ms: "-".to_owned(),
        continuation_ms: "-".to_owned(),
        symbolic_jacobian_ms: "-".to_owned(),
        atom_jacobian_ms: "-".to_owned(),
        aot_build_ms: "-".to_owned(),
        aot_link_ms: "-".to_owned(),
        factorization_ms: "-".to_owned(),
        residual_calls: 0,
        jacobian_calls: 0,
        factorizations: 0,
        accepted_steps: 0,
        workers: 0,
        parallel_dispatches: 0,
        sequential_dispatches: 0,
        aot_chunks: 0,
        max_final_abs: "-".to_owned(),
        parity_diff: "-".to_owned(),
        status,
    };

    let mut solver = match Lsode2Solver::new(make_config(
        workload_data,
        matrix,
        frontend,
        route,
        output.as_ref(),
    )) {
        Ok(solver) => solver,
        Err(error) => return (base(format!("construct-error: {error}")), None),
    };
    let prepare_started = Instant::now();
    if let Err(error) = solver.prepare() {
        return (base(format!("prepare-error: {error}")), None);
    }
    let prepare_ms = ms(prepare_started);
    let solve_started = Instant::now();
    let summary = match solver.solve_with_summary() {
        Ok(summary) => summary,
        Err(error) => return (base(format!("solve-error: {error}")), None),
    };
    let solve_ms = ms(solve_started);
    let solve_final_state = summary
        .final_y
        .as_ref()
        .map(|state| state.iter().copied().collect::<Vec<_>>());
    let snapshot = solver.telemetry_snapshot();
    let mut continuation = Vec::new();
    let mut status = "ok".to_owned();
    if parameterized {
        for count in counts {
            let started = Instant::now();
            for index in 1..=*count {
                if let Err(error) = solver
                    .set_parameter_values(parameter_continuation_target(&parameter_values, index))
                    .and_then(|_| solver.solve())
                {
                    status = format!("continuation-error: {error}");
                    break;
                }
            }
            continuation.push(format!("{count}={:.3}", ms(started)));
            if status != "ok" {
                break;
            }
        }
    }
    let stats = &summary.statistics;
    let row = Lsode2CompactRow {
        workload: workload.label().to_owned(),
        dimension: label_dimension,
        frontend: frontend.label().to_owned(),
        execution: route.label().to_owned(),
        linear_backend: matrix.label().to_owned(),
        policy: route.policy().label().to_owned(),
        prepare_ms: fmt_ms(prepare_ms),
        solve_ms: fmt_ms(solve_ms),
        continuation_ms: if continuation.is_empty() {
            "-".to_owned()
        } else {
            continuation.join(";")
        },
        symbolic_jacobian_ms: fmt_ms(stage_ms(&snapshot, IvpColdStage::SymbolicJacobian)),
        atom_jacobian_ms: fmt_ms(stage_ms(&snapshot, IvpColdStage::AtomJacobianPreparation)),
        aot_build_ms: fmt_ms(stage_ms(&snapshot, IvpColdStage::AotBuild)),
        aot_link_ms: fmt_ms(stage_ms(&snapshot, IvpColdStage::AotLink)),
        factorization_ms: fmt_ms(stats.linear_factorization_ms_total),
        residual_calls: stats.residual_calls,
        jacobian_calls: stats.jacobian_calls,
        factorizations: stats.bdf_nlu_total,
        accepted_steps: stats.accepted_steps_total,
        workers: snapshot.lambdify_worker_count,
        parallel_dispatches: snapshot.parallel_dispatches,
        sequential_dispatches: snapshot.sequential_dispatches,
        aot_chunks: snapshot.aot_chunks,
        max_final_abs: format!("{:.6e}", summary.max_abs_solution),
        parity_diff: "pending".to_owned(),
        status,
    };
    (row, solve_final_state)
}

fn main() {
    let mut rows = Vec::new();
    let mut states = Vec::new();
    let counts = continuation_counts();
    for workload in workloads() {
        for dimension in dimensions(workload) {
            for matrix in matrices() {
                for frontend in FrontendRoute::ALL {
                    for route in routes() {
                        let (row, state) =
                            run_case(workload, dimension, matrix, frontend, route, &counts);
                        states.push((workload, dimension, matrix, route, frontend, state));
                        rows.push(row);
                    }
                }
            }
        }
    }

    for (index, row) in rows.iter_mut().enumerate() {
        let (workload, dimension, matrix, route, frontend, candidate) = &states[index];
        let reference = states.iter().find_map(|(w, d, m, r, f, state)| {
            (*w == *workload
                && *d == *dimension
                && *m == *matrix
                && *r == *route
                && *f != *frontend)
                .then_some(state.as_ref())
                .flatten()
        });
        if let (Some(candidate), Some(reference)) = (candidate.as_ref(), reference) {
            row.parity_diff = format!("{:.3e}", state_diff(candidate, reference));
        }
    }

    let mut body = String::new();
    let _ = writeln!(
        body,
        "- compact=true; workloads={:?}; matrices={:?}; routes={:?}; continuation_counts={:?}",
        workloads()
            .iter()
            .map(|workload| workload.label())
            .collect::<Vec<_>>(),
        matrices()
            .iter()
            .map(|matrix| matrix.label())
            .collect::<Vec<_>>(),
        routes()
            .iter()
            .map(|route| route.label())
            .collect::<Vec<_>>(),
        counts
    );
    let _ = writeln!(body, "- AOT compiler: C+tcc; telemetry: detailed");
    let _ = writeln!(
        body,
        "- parent/child stage timings are diagnostic and non-additive\n"
    );
    body.push_str(&Table::new(rows).to_string());
    let name = std::env::var("LSODE2_COMPACT_REPORT_NAME")
        .unwrap_or_else(|_| "lsode2_workload_matrix".to_owned());
    let path = RustedSciThe::Utils::test_reporting::write_test_report("LSODE2_Bench", &name, &body)
        .expect("write LSODE2 compact report");
    println!("[LSODE2 compact benchmark] report={}", path.display());
}
