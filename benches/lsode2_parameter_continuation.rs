//! Repeated-parameter continuation and preparation-amortization benchmark.
//!
//! The warm route prepares one solver and rebinds parameters for a target
//! series. The fresh route prepares a new solver for every target. Both use
//! the same cache-aware policy, so the result exposes the number of repeats
//! needed to amortize preparation without pretending that compilation is
//! repeated for every numeric parameter value.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};
use tempfile::tempdir;

#[path = "support/lsode2_bench_support.rs"]
mod lsode2_bench_support;

use RustedSciThe::numerical::LSODE2::workload_fixtures::{
    SymbolicWorkload, WorkloadKind, build_workload, parameter_continuation_target,
};
use RustedSciThe::numerical::LSODE2::{
    IvpLambdifyExecutionPolicy, IvpTelemetry, Lsode2AotProfile, Lsode2AotToolchain,
    Lsode2ControllerConfig, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::DVector;

const DEFAULT_DIFFUSION_DIMENSIONS: &[usize] = &[256];
const DEFAULT_TARGET_COUNTS: &[usize] = &[1, 4, 16];
const DEFAULT_WORKLOADS: &[WorkloadKind] = &[
    WorkloadKind::DiffusionChain,
    WorkloadKind::CombustionLike,
    WorkloadKind::ThreeBody,
];

#[derive(Clone, Copy, Debug)]
enum MatrixRoute {
    Sparse,
    Banded,
}

#[derive(Clone, Copy, Debug)]
enum ContinuationSlice {
    All,
    DiffusionSparse,
    DiffusionBanded,
    Small,
}

impl ContinuationSlice {
    fn from_env() -> Self {
        match std::env::var("LSODE2_BENCH_CONTINUATION_SLICE")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            // Keep an ordinary `cargo bench` bounded. The full matrix is an
            // explicit release sweep selected with `slice=all`.
            None | Some("small") => Self::Small,
            Some("all") => Self::All,
            Some("diffusion-sparse") => Self::DiffusionSparse,
            Some("diffusion-banded") => Self::DiffusionBanded,
            Some(value) => panic!(
                "invalid LSODE2_BENCH_CONTINUATION_SLICE value {value:?}; expected all, diffusion-sparse, diffusion-banded or small"
            ),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::DiffusionSparse => "diffusion-sparse",
            Self::DiffusionBanded => "diffusion-banded",
            Self::Small => "small",
        }
    }

    const fn allows_matrix(self, matrix: MatrixRoute) -> bool {
        match self {
            Self::All | Self::Small => true,
            Self::DiffusionSparse => matches!(matrix, MatrixRoute::Sparse),
            Self::DiffusionBanded => matches!(matrix, MatrixRoute::Banded),
        }
    }

    const fn allows_workload(self, workload: WorkloadKind) -> bool {
        match self {
            Self::All => true,
            Self::DiffusionSparse | Self::DiffusionBanded => {
                matches!(workload, WorkloadKind::DiffusionChain)
            }
            Self::Small => !matches!(workload, WorkloadKind::DiffusionChain),
        }
    }
}

impl MatrixRoute {
    const ALL: [Self; 2] = [Self::Sparse, Self::Banded];

    const fn label(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Banded => "banded",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum FrontendRoute {
    ExprLegacy,
    AtomViewNative,
}

fn labels_from_env(variable: &str, defaults: &[&str]) -> Vec<String> {
    let Some(value) = std::env::var_os(variable) else {
        return defaults.iter().map(|label| (*label).to_string()).collect();
    };

    let labels = value
        .to_string_lossy()
        .split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    if labels.is_empty() {
        defaults.iter().map(|label| (*label).to_string()).collect()
    } else {
        labels
    }
}

fn selected_matrix_routes() -> Vec<MatrixRoute> {
    let slice = ContinuationSlice::from_env();
    let labels = labels_from_env(
        "LSODE2_BENCH_CONTINUATION_MATRIX_ROUTES",
        &["sparse", "banded"],
    );
    let routes = MatrixRoute::ALL
        .into_iter()
        .filter(|route| slice.allows_matrix(*route))
        .filter(|route| labels.iter().any(|label| label == route.label()))
        .collect::<Vec<_>>();
    assert!(
        !routes.is_empty(),
        "LSODE2_BENCH_CONTINUATION_MATRIX_ROUTES selected no routes; expected sparse and/or banded"
    );
    routes
}

fn selected_frontend_routes() -> Vec<FrontendRoute> {
    let labels = labels_from_env(
        "LSODE2_BENCH_CONTINUATION_FRONTEND_ROUTES",
        &["expr-legacy", "atom-native"],
    );
    let routes = FrontendRoute::ALL
        .into_iter()
        .filter(|route| labels.iter().any(|label| label == route.label()))
        .collect::<Vec<_>>();
    assert!(
        !routes.is_empty(),
        "LSODE2_BENCH_CONTINUATION_FRONTEND_ROUTES selected no routes; expected expr-legacy and/or atom-native"
    );
    routes
}

fn selected_execution_modes() -> Vec<bool> {
    let labels = labels_from_env("LSODE2_BENCH_CONTINUATION_EXECUTIONS", &["lambdify", "aot"]);
    let mut modes = Vec::new();
    if labels.iter().any(|label| label == "lambdify") {
        modes.push(false);
    }
    if labels.iter().any(|label| label == "aot") {
        modes.push(true);
    }
    assert!(
        !modes.is_empty(),
        "LSODE2_BENCH_CONTINUATION_EXECUTIONS selected no routes; expected lambdify and/or aot"
    );
    modes
}

fn selected_phases() -> (bool, bool) {
    let labels = labels_from_env("LSODE2_BENCH_CONTINUATION_PHASES", &["warm"]);
    let phases = (
        labels.iter().any(|label| label == "warm"),
        labels.iter().any(|label| label == "fresh"),
    );
    assert!(
        phases.0 || phases.1,
        "LSODE2_BENCH_CONTINUATION_PHASES selected no phases; expected warm and/or fresh"
    );
    phases
}

impl FrontendRoute {
    const ALL: [Self; 2] = [Self::ExprLegacy, Self::AtomViewNative];

    const fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "expr-legacy",
            Self::AtomViewNative => "atom-native",
        }
    }

    const fn assembly(self) -> Lsode2SymbolicAssemblyBackend {
        match self {
            Self::ExprLegacy => Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Self::AtomViewNative => Lsode2SymbolicAssemblyBackend::AtomView,
        }
    }
}

fn dimensions(kind: WorkloadKind) -> Vec<usize> {
    match kind {
        WorkloadKind::DiffusionChain => lsode2_bench_support::dimensions_from_env(
            "LSODE2_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS",
            DEFAULT_DIFFUSION_DIMENSIONS,
        ),
        WorkloadKind::CombustionLike | WorkloadKind::Robertson => vec![3],
        WorkloadKind::StiffScalar => vec![1],
        WorkloadKind::ThreeBody => vec![12],
    }
}

fn workloads() -> Vec<WorkloadKind> {
    let slice = ContinuationSlice::from_env();
    let selected = lsode2_bench_support::workloads_from_env(
        "LSODE2_BENCH_CONTINUATION_WORKLOADS",
        DEFAULT_WORKLOADS,
    )
    .into_iter()
    .filter(|kind| slice.allows_workload(*kind))
    .filter(|kind| build_workload(*kind, dimensions(*kind)[0]).is_parameterized())
    .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "continuation benchmark selected no parameterized workloads; use combustion-like, three-body or diffusion-chain and check the slice"
    );
    selected
}

fn target_counts() -> Vec<usize> {
    lsode2_bench_support::positive_usizes_from_env(
        "LSODE2_BENCH_CONTINUATION_COUNTS",
        DEFAULT_TARGET_COUNTS,
    )
}

fn allow_long_continuation() -> bool {
    matches!(
        std::env::var("LSODE2_BENCH_CONTINUATION_ALLOW_LONG")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1" | "true" | "yes")
    )
}

fn continuation_case_needs_long_opt_in(dimension: usize, targets: usize, fresh: bool) -> bool {
    let work = dimension.saturating_mul(targets);
    targets > 64 || work > if fresh { 4_096 } else { 16_384 }
}

fn check_continuation_case_budget(
    kind: WorkloadKind,
    dimension: usize,
    targets: usize,
    fresh: bool,
    allow_long: bool,
) {
    if !allow_long && continuation_case_needs_long_opt_in(dimension, targets, fresh) {
        let phase = if fresh { "fresh-cache-aware" } else { "warm" };
        panic!(
            "refusing expensive LSODE2 continuation case by default: workload={}, dimension={dimension}, targets={targets}, phase={phase}; set LSODE2_BENCH_CONTINUATION_ALLOW_LONG=1 to opt in",
            kind.label(),
        );
    }
}

fn validate_continuation_plan(
    workload_list: &[WorkloadKind],
    counts: &[usize],
    phases: (bool, bool),
    allow_long: bool,
) {
    for kind in workload_list {
        for dimension in dimensions(*kind) {
            for count in counts {
                if phases.0 {
                    check_continuation_case_budget(*kind, dimension, *count, false, allow_long);
                }
                if phases.1 {
                    check_continuation_case_budget(*kind, dimension, *count, true, allow_long);
                }
            }
        }
    }
}

fn planned_case_counts(
    workload_list: &[WorkloadKind],
    counts: &[usize],
    matrices: &[MatrixRoute],
    frontends: &[FrontendRoute],
    executions: &[bool],
    phases: (bool, bool),
) -> (usize, usize) {
    let phase_count = usize::from(phases.0) + usize::from(phases.1);
    let route_count = matrices.len() * frontends.len() * executions.len() * phase_count;
    let cases = workload_list
        .iter()
        .map(|kind| dimensions(*kind).len() * counts.len() * route_count)
        .sum();
    let target_solves = workload_list
        .iter()
        .map(|kind| {
            dimensions(*kind).len()
                * counts.iter().sum::<usize>()
                * matrices.len()
                * frontends.len()
                * executions.len()
                * phase_count
        })
        .sum();
    (cases, target_solves)
}

fn execution(aot: bool) -> Lsode2SymbolicExecutionMode {
    if aot {
        Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        }
    } else {
        Lsode2SymbolicExecutionMode::LambdifyExpr
    }
}

fn generated_backend(output_parent: &Path) -> SymbolicIvpGeneratedBackendConfig {
    SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(output_parent.to_path_buf()))
        .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
        .with_c_tcc()
}

fn solver_config(
    workload: SymbolicWorkload,
    matrix: MatrixRoute,
    frontend: FrontendRoute,
    aot: bool,
    output_parent: &Path,
) -> Lsode2ProblemConfig {
    let parameter_names = workload.parameter_names.clone();
    let parameter_values = workload.parameter_values.clone();
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
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: frontend.assembly(),
        execution: execution(aot),
    })
    .with_telemetry(IvpTelemetry::disabled());

    if !parameter_names.is_empty() {
        config = config
            .with_equation_parameters(parameter_names)
            .with_equation_parameter_values(parameter_values);
    }

    config = match matrix {
        MatrixRoute::Sparse => config.with_native_sparse_faer_backend(),
        MatrixRoute::Banded => config.with_native_banded_faithful_backend(),
    };

    if aot {
        let generated = generated_backend(output_parent);
        config = match matrix {
            MatrixRoute::Sparse => config.with_native_sparse_faer_generated_backend(generated),
            MatrixRoute::Banded => config.with_native_banded_faithful_generated_backend(generated),
        };
    }

    config
}

fn parameter_series(workload: &SymbolicWorkload, count: usize) -> Vec<DVector<f64>> {
    (1..=count)
        .map(|step| parameter_continuation_target(&workload.parameter_values, step))
        .collect()
}

fn prepare_solver(
    kind: WorkloadKind,
    dimension: usize,
    matrix: MatrixRoute,
    frontend: FrontendRoute,
    aot: bool,
    output_parent: &Path,
) -> Lsode2Solver {
    let mut solver = Lsode2Solver::new(solver_config(
        build_workload(kind, dimension),
        matrix,
        frontend,
        aot,
        output_parent,
    ))
    .expect("continuation solver construction should succeed");
    solver
        .prepare()
        .expect("continuation solver preparation should succeed");
    solver
}

fn run_continuation(solver: &mut Lsode2Solver, targets: &[DVector<f64>]) {
    for (index, target) in targets.iter().enumerate() {
        solver
            .set_parameter_values(target.clone())
            .expect("numeric parameter rebind should succeed");
        let summary = solver
            .solve_with_summary()
            .expect("continued solve should succeed");
        let native = summary.native_integration_solve.as_ref();
        assert!(
            native.is_some_and(|native| native.reached_t_bound || native.reached_stop_condition),
            "continuation target {} did not complete integration: status={}, final_t={:?}, termination={:?}",
            index + 1,
            summary.status,
            native.map(|native| native.final_t),
            summary.native_termination_kind,
        );
        assert!(
            summary
                .final_y
                .as_ref()
                .is_some_and(|state| { state.iter().all(|value| value.is_finite()) })
        );
        black_box(summary);
    }
}

fn benchmark_warm_continuation(c: &mut Criterion) {
    let phases = selected_phases();
    if !phases.0 {
        return;
    }
    let workload_list = workloads();
    let counts = target_counts();
    let matrices = selected_matrix_routes();
    let frontends = selected_frontend_routes();
    let executions = selected_execution_modes();
    let slice = ContinuationSlice::from_env();
    let allow_long = allow_long_continuation();
    let (planned_cases, planned_target_solves) = planned_case_counts(
        &workload_list,
        &counts,
        &matrices,
        &frontends,
        &executions,
        phases,
    );
    lsode2_bench_support::print_metadata(
        "lsode2_parameter_continuation",
        &dimensions(WorkloadKind::DiffusionChain),
        &format!(
            "slice={}; routes={frontends:?}; matrix_routes={matrices:?}; execution={executions:?}; phases={phases:?}; target_counts={counts:?}; planned_cases={planned_cases}; planned_target_solves={planned_target_solves}; allow_long={allow_long}; telemetry=off; warm_lifecycle=one_prepared_solver_per_id; workloads={workload_list:?}",
            slice.label()
        ),
    );
    validate_continuation_plan(&workload_list, &counts, phases, allow_long);
    let mut group = c.benchmark_group("lsode2_parameter_continuation_warm");
    group.sample_size(lsode2_bench_support::sample_size());
    if let Some(measurement_time) = lsode2_bench_support::measurement_time() {
        group.measurement_time(measurement_time);
    }
    let cache_root = tempdir().expect("continuation cache directory should exist");

    for kind in workload_list {
        for dimension in dimensions(kind) {
            let workload = build_workload(kind, dimension);
            let target_series = counts
                .iter()
                .map(|count| (*count, parameter_series(&workload, *count)))
                .collect::<Vec<_>>();
            for matrix in matrices.iter().copied() {
                for frontend in frontends.iter().copied() {
                    for aot in executions.iter().copied() {
                        let route = if aot { "aot" } else { "lambdify" };
                        for (count, targets) in &target_series {
                            let id = BenchmarkId::new(
                                format!(
                                    "{}/{}/{}/{}/warm-continuation",
                                    kind.label(),
                                    matrix.label(),
                                    frontend.label(),
                                    route
                                ),
                                format!("n={dimension}/targets={count}"),
                            );
                            group.bench_function(id, |bencher| {
                                bencher.iter_custom(|iterations| {
                                    let mut measured = Duration::ZERO;
                                    for _ in 0..iterations {
                                        // Isolate Criterion iterations. Preparation is
                                        // intentionally outside the timed interval, while
                                        // each measured series starts from a fresh prepared
                                        // solver rather than carrying mutable state from the
                                        // previous sample.
                                        let mut solver = prepare_solver(
                                            kind,
                                            dimension,
                                            matrix,
                                            frontend,
                                            aot,
                                            cache_root.path(),
                                        );
                                        let started = Instant::now();
                                        run_continuation(&mut solver, targets);
                                        measured += started.elapsed();
                                    }
                                    measured
                                });
                            });
                        }
                    }
                }
            }
        }
    }
    group.finish();
}

fn benchmark_fresh_cache_aware(c: &mut Criterion) {
    let phases = selected_phases();
    if !phases.1 {
        return;
    }
    let workload_list = workloads();
    let counts = target_counts();
    let matrices = selected_matrix_routes();
    let frontends = selected_frontend_routes();
    let executions = selected_execution_modes();
    let slice = ContinuationSlice::from_env();
    let allow_long = allow_long_continuation();
    let (planned_cases, planned_target_solves) = planned_case_counts(
        &workload_list,
        &counts,
        &matrices,
        &frontends,
        &executions,
        phases,
    );
    lsode2_bench_support::print_metadata(
        "lsode2_parameter_continuation_fresh",
        &dimensions(WorkloadKind::DiffusionChain),
        &format!(
            "slice={}; routes={frontends:?}; matrix_routes={matrices:?}; execution={executions:?}; phases={phases:?}; target_counts={counts:?}; planned_cases={planned_cases}; planned_target_solves={planned_target_solves}; allow_long={allow_long}; telemetry=off; workloads={workload_list:?}",
            slice.label()
        ),
    );
    validate_continuation_plan(&workload_list, &counts, phases, allow_long);
    let mut group = c.benchmark_group("lsode2_parameter_continuation_fresh");
    group.sample_size(lsode2_bench_support::sample_size());
    if let Some(measurement_time) = lsode2_bench_support::measurement_time() {
        group.measurement_time(measurement_time);
    }
    let cache_root = tempdir().expect("fresh continuation cache directory should exist");

    for kind in workload_list {
        for dimension in dimensions(kind) {
            let workload = build_workload(kind, dimension);
            for matrix in matrices.iter().copied() {
                for frontend in frontends.iter().copied() {
                    for aot in executions.iter().copied() {
                        let route = if aot { "aot" } else { "lambdify" };
                        for count in &counts {
                            let targets = parameter_series(&workload, *count);
                            let id = BenchmarkId::new(
                                format!(
                                    "{}/{}/{}/{}/fresh-cache-aware",
                                    kind.label(),
                                    matrix.label(),
                                    frontend.label(),
                                    route
                                ),
                                format!("n={dimension}/targets={count}"),
                            );
                            group.bench_function(id, |bencher| {
                                bencher.iter(|| {
                                    for target in &targets {
                                        let mut solver = prepare_solver(
                                            kind,
                                            dimension,
                                            matrix,
                                            frontend,
                                            aot,
                                            cache_root.path(),
                                        );
                                        solver.set_parameter_values(target.clone()).expect(
                                            "fresh numeric parameter rebind should succeed",
                                        );
                                        let summary = solver
                                            .solve_with_summary()
                                            .expect("fresh solve should succeed");
                                        assert!(summary.final_y.as_ref().is_some_and(|state| {
                                            state.iter().all(|value| value.is_finite())
                                        }));
                                        black_box(summary);
                                    }
                                });
                            });
                        }
                    }
                }
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_warm_continuation,
    benchmark_fresh_cache_aware
);
criterion_main!(benches);
