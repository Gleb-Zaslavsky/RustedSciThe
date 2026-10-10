//! Compact BVP_sci AOT/Lambdify dashboard.
//!
//! This is intentionally a manual dashboard rather than a Criterion group.
//! AOT preparation is a lifecycle operation and must not be repeated by a
//! statistical warm-up loop. The report therefore separates cold preparation,
//! warm callbacks, parameter continuation and callback policy dispatch.

use std::path::PathBuf;
use std::time::Instant;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciExecutionPolicy,
    BvpSciLambdifyPlan, BvpSciMatrixLayout, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
};
use RustedSciThe::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetrySnapshot};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use tabled::{Table, Tabled};

fn validate_markdown_table(table: &str) {
    let mut expected_delimiters = None;
    for line in table.lines().filter(|line| line.starts_with('|')) {
        let delimiters = line.chars().filter(|character| *character == '|').count();
        match expected_delimiters {
            Some(expected) => assert_eq!(
                delimiters, expected,
                "Tabled report row has an inconsistent Markdown delimiter count"
            ),
            None => expected_delimiters = Some(delimiters),
        }
    }
}

#[derive(Clone, Debug, Tabled)]
struct Row {
    frontend: String,
    layout: String,
    dimension: usize,
    policy: String,
    preparation_ms: String,
    preparation_scope: String,
    bvp_expr_to_atom_ms: String,
    bvp_symbolic_jacobian_ms: String,
    bvp_pattern_ms: String,
    bvp_evaluator_compilation_ms: String,
    bvp_binding_ms: String,
    aot_symbolic_jacobian_ms: String,
    aot_residual_compilation_ms: String,
    aot_jacobian_compilation_ms: String,
    aot_atom_residual_ms: String,
    aot_atom_jacobian_ms: String,
    aot_shared_symbolic_problem_ms: String,
    aot_runtime_validation_ms: String,
    aot_generated_prepare_ms: String,
    aot_cache_lookup_ms: String,
    aot_problem_key_ms: String,
    aot_atom_plan_ms: String,
    aot_input_abi_ms: String,
    aot_lowering_ms: String,
    aot_source_generation_ms: String,
    aot_parallel_calibration_ms: String,
    aot_solver_preparation_ms: String,
    aot_bridge_preparation_ms: String,
    aot_materialization_ms: String,
    aot_build_ms: String,
    aot_link_ms: String,
    aot_publication_ms: String,
    aot_cache_hits: u64,
    aot_cache_misses: u64,
    aot_reconnects: u64,
    callback_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    residual_callback_ms: String,
    jacobian_callback_ms: String,
    continuation_ms: String,
    continuation_vs_lambdify: String,
    residual_calls: u64,
    jacobian_calls: u64,
    parallel_dispatches: u64,
    sequential_dispatches: u64,
    chunk_dispatches: u64,
    dispatch_requests: u64,
    eligible_dispatches: u64,
    sequential_fallbacks: u64,
    completed_dispatches: u64,
    failed_dispatches: u64,
    chunks: u64,
    worker_callbacks: u64,
    observed_workers: u64,
    runtime_ready: u64,
    build_attempts: u64,
    link_attempts: u64,
    max_rhs_diff: String,
    max_jacobian_diff: String,
    status: String,
}

fn aot_stage_ms(aot: Option<&IvpTelemetrySnapshot>, stage: IvpColdStage) -> String {
    aot.map(|snapshot| {
        format!(
            "{:.3}",
            snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1_000.0
        )
    })
    .unwrap_or_else(|| "-".into())
}

fn optional_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".into())
}

#[derive(Clone, Copy)]
enum Frontend {
    ExprLegacy,
    AtomViewNative,
    AotExprLegacy,
    AotAtomViewNative,
}

impl Frontend {
    fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "lambdify-expr-legacy",
            Self::AtomViewNative => "lambdify-atom-native",
            Self::AotExprLegacy => "aot-expr-legacy",
            Self::AotAtomViewNative => "aot-atom-native",
        }
    }

    fn assembly(self) -> BvpSciAssembly {
        match self {
            Self::ExprLegacy | Self::AotExprLegacy => BvpSciAssembly::ExprLegacy,
            Self::AtomViewNative | Self::AotAtomViewNative => BvpSciAssembly::AtomViewNative,
        }
    }

    fn is_aot(self) -> bool {
        matches!(self, Self::AotExprLegacy | Self::AotAtomViewNative)
    }

    fn preparation_scope(self) -> &'static str {
        if self.is_aot() {
            "aot:BuildIfMissing+compile+link+publish+plan"
        } else {
            "lambdify:symbolic+evaluator+plan"
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
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::Banded => "banded",
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

fn parse_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn dimensions() -> Vec<usize> {
    parse_list("BVP_SCI_AOT_BENCH_DIMENSIONS", "2,8,32")
        .into_iter()
        .filter_map(|value| value.parse().ok())
        .filter(|value: &usize| *value > 0)
        .collect()
}

fn equations(dimension: usize) -> (Vec<Expr>, Vec<String>) {
    let names = (0..dimension)
        .map(|index| format!("y{index}"))
        .collect::<Vec<_>>();
    let equations = (0..dimension)
        .map(|index| {
            if index == 0 {
                Expr::parse_expression("p*y0 + x")
            } else {
                Expr::parse_expression(&format!("y{} - y{} + p*x", index - 1, index))
            }
        })
        .collect();
    (equations, names)
}

fn policy(label: &str) -> BvpSciExecutionPolicy {
    let min_work = std::env::var("BVP_SCI_AOT_BENCH_MIN_WORK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    match label {
        "parallel" => BvpSciExecutionPolicy::Parallel { min_work },
        "auto" => BvpSciExecutionPolicy::Auto { min_work },
        _ => BvpSciExecutionPolicy::Sequential,
    }
}

fn selected_frontends() -> Vec<Frontend> {
    parse_list(
        "BVP_SCI_AOT_BENCH_FRONTENDS",
        "lambdify-expr-legacy,lambdify-atom-native,aot-expr-legacy,aot-atom-native",
    )
    .into_iter()
    .filter_map(|value| match value.as_str() {
        "lambdify-expr-legacy" => Some(Frontend::ExprLegacy),
        "lambdify-atom-native" => Some(Frontend::AtomViewNative),
        "aot-expr-legacy" => Some(Frontend::AotExprLegacy),
        "aot-atom-native" => Some(Frontend::AotAtomViewNative),
        _ => None,
    })
    .collect()
}

fn selected_layouts() -> Vec<Layout> {
    parse_list("BVP_SCI_AOT_BENCH_LAYOUTS", "dense,sparse,banded")
        .into_iter()
        .filter_map(|value| match value.as_str() {
            "dense" => Some(Layout::Dense),
            "sparse" => Some(Layout::Sparse),
            "banded" => Some(Layout::Banded),
            _ => None,
        })
        .collect()
}

fn selected_policies() -> Vec<String> {
    parse_list("BVP_SCI_AOT_BENCH_POLICIES", "sequential")
}

fn output_root() -> PathBuf {
    std::env::var_os("BVP_SCI_AOT_BENCH_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/bvp-sci-aot-bench"))
}

fn run_row(frontend: Frontend, layout: Layout, dimension: usize, policy_name: &str) -> Row {
    let (equations, states) = equations(dimension);
    let parameters = vec!["p".to_owned()];
    let telemetry = BvpSciTelemetry::timings();
    let prepare_started = Instant::now();
    let plan = if frontend.is_aot() {
        let mut config = RustedSciThe::numerical::BVP_sci::new::SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
            output_root()
                .join(frontend.label())
                .join(layout.label())
                .join(dimension.to_string()),
        );
        config = match std::env::var("BVP_SCI_AOT_COMPILER")
            .unwrap_or_else(|_| "tcc".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "gcc" => config.with_c_gcc(),
            _ => config.with_c_tcc(),
        };
        BvpSciLambdifyPlan::prepare_aot_with_policy(
            frontend.assembly(),
            layout.value(),
            equations,
            states,
            parameters.clone(),
            "x",
            config,
            telemetry,
            policy(policy_name),
        )
    } else {
        BvpSciLambdifyPlan::prepare(
            frontend.assembly(),
            &equations,
            &states,
            &parameters,
            "x",
            telemetry,
        )
        .map(|plan| plan.with_execution_policy(policy(policy_name)))
    };
    let preparation_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
    let plan = match plan {
        Ok(plan) => plan,
        Err(error) => {
            return failed_row(
                frontend,
                layout,
                dimension,
                policy_name,
                preparation_ms,
                error.to_string(),
            );
        }
    };

    let mut arguments = vec![0.0; dimension + 2];
    let mut rhs = vec![0.0; dimension];
    let mut jacobian = vec![0.0; dimension * dimension];
    let state = vec![1.0; dimension];
    // Banded callbacks expose the complete compact slot buffer, which can be
    // larger than the structural nonzero list. Keep this caller-owned buffer
    // independent of the route so its capacity is visible in the benchmark.
    let mut scratch = vec![0.0; dimension.saturating_mul(dimension).max(3 * dimension)];
    let callback_started = Instant::now();
    let mut max_rhs_diff: f64 = 0.0;
    let mut max_jacobian_diff: f64 = 0.0;
    let residual_started = Instant::now();
    let residual_result = (0..32).try_for_each(|_| {
        plan.evaluate_rhs(0.25, &state, &[3.0], &mut arguments, &mut rhs)?;
        max_rhs_diff = max_rhs_diff.max((rhs[0] - 3.25).abs());
        Ok::<_, RustedSciThe::numerical::BVP_sci::new::BvpSciNewError>(())
    });
    let residual_ms = residual_started.elapsed().as_secs_f64() * 1e3;
    if let Err(error) = residual_result {
        return failed_row(
            frontend,
            layout,
            dimension,
            policy_name,
            preparation_ms,
            error.to_string(),
        );
    }

    let jacobian_started = Instant::now();
    let jacobian_result = (0..32).try_for_each(|_| {
        plan.evaluate_jacobian_dense_with_scratch(
            0.25,
            &state,
            &[3.0],
            &mut arguments,
            &mut jacobian,
            &mut scratch,
        )?;
        max_jacobian_diff = max_jacobian_diff.max((jacobian[0] - 3.0).abs());
        Ok::<_, RustedSciThe::numerical::BVP_sci::new::BvpSciNewError>(())
    });
    let jacobian_ms = jacobian_started.elapsed().as_secs_f64() * 1e3;
    if let Err(error) = jacobian_result {
        return failed_row(
            frontend,
            layout,
            dimension,
            policy_name,
            preparation_ms,
            error.to_string(),
        );
    }
    let callback_ms = callback_started.elapsed().as_secs_f64() * 1e3;
    // Use the same evaluator-only telemetry scopes as the Lambdify dashboard
    // for the matched residual/Jacobian comparison. The external callback
    // wall clock remains available separately in `callback_ms`.
    let callback_telemetry = plan.telemetry_snapshot();

    let continuation_started = Instant::now();
    for index in 0..16 {
        let parameter = 1.0 + index as f64 * 0.1;
        plan.evaluate_rhs(0.5, &state, &[parameter], &mut arguments, &mut rhs)
            .expect("continuation callback should remain valid");
        plan.evaluate_jacobian_dense_with_scratch(
            0.5,
            &state,
            &[parameter],
            &mut arguments,
            &mut jacobian,
            &mut scratch,
        )
        .expect("continuation Jacobian callback should remain valid");
    }
    let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
    let telemetry = plan.telemetry_snapshot();
    let aot = telemetry.aot.as_ref();
    Row {
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        dimension,
        policy: policy_name.into(),
        preparation_ms: format!("{preparation_ms:.3}"),
        preparation_scope: frontend.preparation_scope().into(),
        bvp_expr_to_atom_ms: optional_ms(telemetry.expr_to_atom_ms),
        bvp_symbolic_jacobian_ms: optional_ms(telemetry.symbolic_jacobian_ms),
        bvp_pattern_ms: optional_ms(telemetry.pattern_ms),
        bvp_evaluator_compilation_ms: optional_ms(telemetry.evaluator_compilation_ms),
        bvp_binding_ms: optional_ms(telemetry.binding_ms),
        aot_symbolic_jacobian_ms: aot_stage_ms(aot, IvpColdStage::SymbolicJacobian),
        aot_residual_compilation_ms: aot_stage_ms(aot, IvpColdStage::ResidualCompilation),
        aot_jacobian_compilation_ms: aot_stage_ms(aot, IvpColdStage::JacobianCompilation),
        aot_atom_residual_ms: aot_stage_ms(aot, IvpColdStage::AtomResidualPreparation),
        aot_atom_jacobian_ms: aot_stage_ms(aot, IvpColdStage::AtomJacobianPreparation),
        aot_shared_symbolic_problem_ms: aot_stage_ms(
            aot,
            IvpColdStage::SharedSymbolicProblemPreparation,
        ),
        aot_runtime_validation_ms: aot_stage_ms(aot, IvpColdStage::BackendBinding),
        aot_generated_prepare_ms: aot_stage_ms(aot, IvpColdStage::SolverPreparation),
        aot_cache_lookup_ms: aot_stage_ms(aot, IvpColdStage::AotCacheLookup),
        aot_problem_key_ms: aot_stage_ms(aot, IvpColdStage::AotProblemKeyConstruction),
        aot_atom_plan_ms: aot_stage_ms(aot, IvpColdStage::AotAtomPlanPreparation),
        aot_input_abi_ms: aot_stage_ms(aot, IvpColdStage::AotInputAbiPreparation),
        aot_lowering_ms: aot_stage_ms(aot, IvpColdStage::AotLowering),
        aot_source_generation_ms: aot_stage_ms(aot, IvpColdStage::AotSourceGeneration),
        aot_parallel_calibration_ms: aot_stage_ms(aot, IvpColdStage::ParallelCalibration),
        aot_solver_preparation_ms: aot_stage_ms(aot, IvpColdStage::SolverPreparation),
        aot_bridge_preparation_ms: aot_stage_ms(aot, IvpColdStage::BridgePreparation),
        aot_materialization_ms: aot_stage_ms(aot, IvpColdStage::AotMaterialization),
        aot_build_ms: aot_stage_ms(aot, IvpColdStage::AotBuild),
        aot_link_ms: aot_stage_ms(aot, IvpColdStage::AotLink),
        aot_publication_ms: aot_stage_ms(aot, IvpColdStage::AotPublication),
        aot_cache_hits: aot.map_or(0, |value| value.aot_resolution_hits),
        aot_cache_misses: aot.map_or(0, |value| value.aot_resolution_misses),
        aot_reconnects: aot.map_or(0, |value| value.aot_reconnects),
        callback_ms: format!("{callback_ms:.3}"),
        residual_ms: optional_ms(callback_telemetry.residual_evaluation_ms),
        jacobian_ms: optional_ms(callback_telemetry.jacobian_evaluation_ms),
        residual_callback_ms: format!("{residual_ms:.3}"),
        jacobian_callback_ms: format!("{jacobian_ms:.3}"),
        continuation_ms: format!("{continuation_ms:.3}"),
        continuation_vs_lambdify: "pending-baseline".into(),
        residual_calls: telemetry.residual_evaluations,
        jacobian_calls: telemetry.jacobian_evaluations,
        parallel_dispatches: aot.map_or(telemetry.parallel_dispatches, |value| {
            value.aot_parallel_dispatches
        }),
        sequential_dispatches: aot.map_or(telemetry.sequential_dispatches, |value| {
            value.sequential_dispatches
        }),
        chunk_dispatches: aot.map_or(0, |value| value.aot_chunk_dispatches),
        dispatch_requests: aot.map_or(0, |value| value.aot_dispatch_requests),
        eligible_dispatches: aot.map_or(0, |value| value.aot_eligible_dispatches),
        sequential_fallbacks: aot.map_or(0, |value| value.aot_sequential_fallbacks),
        completed_dispatches: aot.map_or(0, |value| value.aot_completed_dispatches),
        failed_dispatches: aot.map_or(0, |value| value.aot_failed_dispatches),
        chunks: aot.map_or(0, |value| value.aot_chunks),
        worker_callbacks: aot.map_or(0, |value| value.aot_worker_callbacks),
        observed_workers: aot.map_or(telemetry.max_worker_threads, |value| {
            value.lambdify_worker_count as u64
        }),
        runtime_ready: aot.map_or(0, |value| value.aot_runtime_ready),
        build_attempts: aot.map_or(0, |value| value.aot_build_attempts),
        link_attempts: aot.map_or(0, |value| value.aot_link_attempts),
        max_rhs_diff: format!("{max_rhs_diff:.3e}"),
        max_jacobian_diff: format!("{max_jacobian_diff:.3e}"),
        status: "ok".into(),
    }
}

fn failed_row(
    frontend: Frontend,
    layout: Layout,
    dimension: usize,
    policy_name: &str,
    preparation_ms: f64,
    error: String,
) -> Row {
    Row {
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        dimension,
        policy: policy_name.into(),
        preparation_ms: format!("{preparation_ms:.3}"),
        preparation_scope: frontend.preparation_scope().into(),
        bvp_expr_to_atom_ms: "-".into(),
        bvp_symbolic_jacobian_ms: "-".into(),
        bvp_pattern_ms: "-".into(),
        bvp_evaluator_compilation_ms: "-".into(),
        bvp_binding_ms: "-".into(),
        aot_symbolic_jacobian_ms: "-".into(),
        aot_residual_compilation_ms: "-".into(),
        aot_jacobian_compilation_ms: "-".into(),
        aot_atom_residual_ms: "-".into(),
        aot_atom_jacobian_ms: "-".into(),
        aot_shared_symbolic_problem_ms: "-".into(),
        aot_runtime_validation_ms: "-".into(),
        aot_generated_prepare_ms: "-".into(),
        aot_cache_lookup_ms: "-".into(),
        aot_problem_key_ms: "-".into(),
        aot_atom_plan_ms: "-".into(),
        aot_input_abi_ms: "-".into(),
        aot_lowering_ms: "-".into(),
        aot_source_generation_ms: "-".into(),
        aot_parallel_calibration_ms: "-".into(),
        aot_solver_preparation_ms: "-".into(),
        aot_bridge_preparation_ms: "-".into(),
        aot_materialization_ms: "-".into(),
        aot_build_ms: "-".into(),
        aot_link_ms: "-".into(),
        aot_publication_ms: "-".into(),
        aot_cache_hits: 0,
        aot_cache_misses: 0,
        aot_reconnects: 0,
        callback_ms: "-".into(),
        residual_ms: "-".into(),
        jacobian_ms: "-".into(),
        residual_callback_ms: "-".into(),
        jacobian_callback_ms: "-".into(),
        continuation_ms: "-".into(),
        continuation_vs_lambdify: "not-applicable".into(),
        residual_calls: 0,
        jacobian_calls: 0,
        parallel_dispatches: 0,
        sequential_dispatches: 0,
        chunk_dispatches: 0,
        dispatch_requests: 0,
        eligible_dispatches: 0,
        sequential_fallbacks: 0,
        completed_dispatches: 0,
        failed_dispatches: 0,
        chunks: 0,
        worker_callbacks: 0,
        observed_workers: 0,
        runtime_ready: 0,
        build_attempts: 0,
        link_attempts: 0,
        max_rhs_diff: "-".into(),
        max_jacobian_diff: "-".into(),
        status: format!("error: {error}"),
    }
}

#[derive(Debug, Tabled)]
struct FullSolvePolicyRow {
    frontend: String,
    layout: String,
    dimension: usize,
    nodes: usize,
    policy: String,
    workers: usize,
    repeats: usize,
    prepare_ms: String,
    preparation_scope: String,
    full_solve_ms: String,
    full_solve_calls: u64,
    newton_ms: String,
    mesh_defect_ms: String,
    mesh_refinement_ms: String,
    output_construction_ms: String,
    callback_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    factorization_ms: String,
    bvp_parallel_dispatches: u64,
    bvp_sequential_dispatches: u64,
    aot_chunk_dispatches: u64,
    aot_dispatch_requests: u64,
    aot_eligible_dispatches: u64,
    aot_parallel_dispatches: u64,
    aot_sequential_fallbacks: u64,
    aot_completed_dispatches: u64,
    aot_failed_dispatches: u64,
    aot_chunks: u64,
    aot_worker_callbacks: u64,
    observed_workers: u64,
    break_even_vs_sequential: String,
    status: String,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn format_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".into())
}

fn aot_full_solve_row(
    frontend: Frontend,
    layout: Layout,
    dimension: usize,
    nodes: usize,
    policy_name: &str,
    workers: usize,
    repeats: usize,
) -> (FullSolvePolicyRow, Option<f64>) {
    let repeats = repeats.max(1);
    let mut prepare_samples = Vec::with_capacity(repeats);
    let mut solve_samples = Vec::with_capacity(repeats);
    let mut callback_samples = Vec::with_capacity(repeats);
    let mut residual_samples = Vec::with_capacity(repeats);
    let mut jacobian_samples = Vec::with_capacity(repeats);
    let mut factorization_samples = Vec::with_capacity(repeats);
    let mut last_snapshot = None;
    let (equations, states) = full_solve_equations(dimension);

    for _ in 0..repeats {
        let output_root = output_root()
            .join("policy-full-solve")
            .join(frontend.label())
            .join(layout.label())
            .join(policy_name)
            .join(format!("{dimension}-{nodes}"));
        let mut config =
            RustedSciThe::numerical::BVP_sci::new::SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                output_root,
            );
        config = match std::env::var("BVP_SCI_AOT_COMPILER")
            .unwrap_or_else(|_| "tcc".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "gcc" => config.with_c_gcc(),
            _ => config.with_c_tcc(),
        };
        let telemetry = BvpSciTelemetry::timings();
        let prepare_started = Instant::now();
        let plan = match BvpSciLambdifyPlan::prepare_aot_with_policy(
            frontend.assembly(),
            layout.value(),
            equations.clone(),
            states.clone(),
            vec!["p".into()],
            "x",
            config,
            telemetry,
            policy(policy_name),
        ) {
            Ok(plan) => plan,
            Err(error) => {
                return (
                    aot_full_solve_error_row(
                        frontend,
                        layout,
                        dimension,
                        nodes,
                        policy_name,
                        workers,
                        repeats,
                        format!("prepare: {error}"),
                    ),
                    None,
                );
            }
        };
        prepare_samples.push(prepare_started.elapsed().as_secs_f64() * 1e3);

        let target_dimension = dimension;
        let boundary = BvpSciBoundaryCallbacks::new(
            dimension + 1,
            move |ya, yb, parameters, output| {
                output[..target_dimension].copy_from_slice(ya);
                let parameter = parameters.first().copied().unwrap_or(0.0);
                output[target_dimension] = yb[0] - (parameter.exp() - 1.0);
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let mut options = BvpSciOptions::default();
        options.execution = BvpSciExecution::Aot;
        options.assembly = Some(frontend.assembly());
        options.matrix_layout = layout.value();
        options.execution_policy = policy(policy_name);
        options.max_nodes = nodes.saturating_mul(16).max(128);
        options.max_mesh_refinements = 8;
        options.max_newton_iterations = 20;
        options.max_jacobian_refreshes = 30;
        options.tolerance = 1e-5;
        let mesh = (0..nodes)
            .map(|index| index as f64 / (nodes - 1) as f64)
            .collect::<Vec<_>>();
        let initial_parameter = 0.5;
        let initial_state = mesh
            .iter()
            .flat_map(|&x| {
                let exact = (initial_parameter * x).exp() - 1.0;
                (0..dimension).map(move |_| 0.90 * exact + 0.01 * (std::f64::consts::PI * x).sin())
            })
            .collect::<Vec<_>>();
        let mut solver = match BvpSciSolver::new(
            plan,
            boundary,
            mesh,
            initial_state,
            vec![initial_parameter],
            options,
        ) {
            Ok(solver) => solver,
            Err(error) => {
                return (
                    aot_full_solve_error_row(
                        frontend,
                        layout,
                        dimension,
                        nodes,
                        policy_name,
                        workers,
                        repeats,
                        format!("construct: {error}"),
                    ),
                    None,
                );
            }
        };
        let solve_started = Instant::now();
        if let Err(error) = solver.solve() {
            return (
                aot_full_solve_error_row(
                    frontend,
                    layout,
                    dimension,
                    nodes,
                    policy_name,
                    workers,
                    repeats,
                    format!("solve: {error}"),
                ),
                None,
            );
        }
        let fallback_solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
        let snapshot = solver.plan().telemetry_snapshot();
        solve_samples.push(snapshot.full_solve_ms.unwrap_or(fallback_solve_ms));
        callback_samples.push(snapshot.callback_ms.unwrap_or_default());
        residual_samples.push(snapshot.residual_evaluation_ms.unwrap_or_default());
        jacobian_samples.push(snapshot.jacobian_evaluation_ms.unwrap_or_default());
        factorization_samples.push(snapshot.factorization_ms.unwrap_or_default());
        last_snapshot = Some(snapshot);
    }

    let snapshot = last_snapshot.expect("at least one AOT full-solve sample");
    let solve_median = median(&mut solve_samples);
    let aot = snapshot.aot.as_ref();
    let row = FullSolvePolicyRow {
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        dimension,
        nodes,
        policy: policy_name.into(),
        workers,
        repeats,
        prepare_ms: format!("{:.3}", median(&mut prepare_samples)),
        preparation_scope: frontend.preparation_scope().into(),
        full_solve_ms: format!("{solve_median:.3}"),
        full_solve_calls: snapshot.full_solve_calls,
        newton_ms: format_opt(snapshot.newton_ms),
        mesh_defect_ms: format_opt(snapshot.mesh_defect_estimation_ms),
        mesh_refinement_ms: format_opt(snapshot.mesh_refinement_ms),
        output_construction_ms: format_opt(snapshot.output_construction_ms),
        callback_ms: format!("{:.3}", median(&mut callback_samples)),
        residual_ms: format!("{:.3}", median(&mut residual_samples)),
        jacobian_ms: format!("{:.3}", median(&mut jacobian_samples)),
        factorization_ms: format!("{:.3}", median(&mut factorization_samples)),
        bvp_parallel_dispatches: snapshot.parallel_dispatches,
        bvp_sequential_dispatches: snapshot.sequential_dispatches,
        aot_chunk_dispatches: aot.map_or(0, |value| value.aot_chunk_dispatches),
        aot_dispatch_requests: aot.map_or(0, |value| value.aot_dispatch_requests),
        aot_eligible_dispatches: aot.map_or(0, |value| value.aot_eligible_dispatches),
        aot_parallel_dispatches: aot.map_or(0, |value| value.aot_parallel_dispatches),
        aot_sequential_fallbacks: aot.map_or(0, |value| value.aot_sequential_fallbacks),
        aot_completed_dispatches: aot.map_or(0, |value| value.aot_completed_dispatches),
        aot_failed_dispatches: aot.map_or(0, |value| value.aot_failed_dispatches),
        aot_chunks: aot.map_or(0, |value| value.aot_chunks),
        aot_worker_callbacks: aot.map_or(0, |value| value.aot_worker_callbacks),
        observed_workers: aot.map_or(snapshot.max_worker_threads, |value| {
            value.lambdify_worker_count as u64
        }),
        break_even_vs_sequential: "baseline".into(),
        status: "ok".into(),
    };
    (row, Some(solve_median))
}

fn aot_full_solve_error_row(
    frontend: Frontend,
    layout: Layout,
    dimension: usize,
    nodes: usize,
    policy_name: &str,
    workers: usize,
    repeats: usize,
    error: String,
) -> FullSolvePolicyRow {
    FullSolvePolicyRow {
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        dimension,
        nodes,
        policy: policy_name.into(),
        workers,
        repeats,
        prepare_ms: "-".into(),
        preparation_scope: frontend.preparation_scope().into(),
        full_solve_ms: "-".into(),
        full_solve_calls: 0,
        newton_ms: "-".into(),
        mesh_defect_ms: "-".into(),
        mesh_refinement_ms: "-".into(),
        output_construction_ms: "-".into(),
        callback_ms: "-".into(),
        residual_ms: "-".into(),
        jacobian_ms: "-".into(),
        factorization_ms: "-".into(),
        bvp_parallel_dispatches: 0,
        bvp_sequential_dispatches: 0,
        aot_chunk_dispatches: 0,
        aot_dispatch_requests: 0,
        aot_eligible_dispatches: 0,
        aot_parallel_dispatches: 0,
        aot_sequential_fallbacks: 0,
        aot_completed_dispatches: 0,
        aot_failed_dispatches: 0,
        aot_chunks: 0,
        aot_worker_callbacks: 0,
        observed_workers: 0,
        break_even_vs_sequential: "not-applicable".into(),
        status: format!("error: {error}"),
    }
}

fn full_solve_equations(dimension: usize) -> (Vec<Expr>, Vec<String>) {
    let states = (0..dimension)
        .map(|index| format!("y{index}"))
        .collect::<Vec<_>>();
    let equations = states
        .iter()
        .map(|state| Expr::parse_expression(&format!("p*({state} + 1)")))
        .collect();
    (equations, states)
}

fn aot_full_solve_policy_report(
    workers: usize,
    dimensions: &[usize],
    nodes: &[usize],
    frontends: &[Frontend],
    policies: &[String],
    layouts: &[Layout],
    repeats: usize,
) -> String {
    let mut rows = Vec::new();
    for &dimension in dimensions {
        for &node_count in nodes {
            for &layout in layouts {
                for &frontend in frontends {
                    if !frontend.is_aot() {
                        continue;
                    }
                    for policy_name in policies {
                        let (row, solve_ms) = aot_full_solve_row(
                            frontend,
                            layout,
                            dimension,
                            node_count,
                            policy_name,
                            workers,
                            repeats,
                        );
                        rows.push((row, solve_ms));
                    }
                }
            }
        }
    }
    let mut rendered = Vec::with_capacity(rows.len());
    for (mut row, solve_ms) in rows {
        if let (Some(value), Some(baseline)) = (
            solve_ms,
            rendered.iter().find_map(|candidate: &FullSolvePolicyRow| {
                (candidate.policy == "sequential"
                    && candidate.frontend == row.frontend
                    && candidate.layout == row.layout
                    && candidate.dimension == row.dimension
                    && candidate.nodes == row.nodes
                    && candidate.status == "ok")
                    .then(|| candidate.full_solve_ms.parse::<f64>().ok())
                    .flatten()
            }),
        ) {
            let delta = (value / baseline - 1.0) * 100.0;
            row.break_even_vs_sequential = if delta <= -5.0 && baseline - value >= 0.01 {
                format!("win:{delta:.1}%")
            } else if delta >= 5.0 && value - baseline >= 0.01 {
                format!("loss:{delta:+.1}%")
            } else {
                format!("neutral:{delta:+.1}%")
            };
        }
        rendered.push(row);
    }
    let table = Table::new(rendered).to_string();
    validate_markdown_table(&table);
    table
}

fn continuation_comparison(rows: &[Row], row: &Row) -> String {
    let lambdify_frontend = match row.frontend.as_str() {
        "aot-expr-legacy" => "lambdify-expr-legacy",
        "aot-atom-native" => "lambdify-atom-native",
        _ => return "baseline".into(),
    };
    let Some(reference) = rows.iter().find(|candidate| {
        candidate.frontend == lambdify_frontend
            && candidate.layout == row.layout
            && candidate.dimension == row.dimension
            && candidate.policy == row.policy
            && candidate.status == "ok"
    }) else {
        return "missing-lambdify-row".into();
    };
    let Ok(aot_ms) = row.continuation_ms.parse::<f64>() else {
        return "not-applicable".into();
    };
    let Ok(lambdify_ms) = reference.continuation_ms.parse::<f64>() else {
        return "not-applicable".into();
    };
    if lambdify_ms <= 0.0 {
        return "not-applicable".into();
    }
    let delta = (aot_ms / lambdify_ms - 1.0) * 100.0;
    format!("vs {lambdify_frontend}: {delta:+.1}%")
}

fn main() {
    let phase = std::env::var("BVP_SCI_AOT_BENCH_PHASE").unwrap_or_else(|_| "matrix".into());
    let compiler = std::env::var("BVP_SCI_AOT_COMPILER").unwrap_or_else(|_| "tcc".into());
    let policies = if phase == "policy" || phase == "policy-full-solve" {
        selected_policies()
    } else {
        vec!["sequential".into()]
    };
    if phase == "policy-full-solve" {
        let dimensions = parse_list("BVP_SCI_AOT_FULL_SOLVE_DIMENSIONS", "8,32")
            .into_iter()
            .filter_map(|value| value.parse().ok())
            .filter(|value: &usize| *value > 0)
            .collect::<Vec<_>>();
        let nodes = parse_list("BVP_SCI_AOT_FULL_SOLVE_NODES", "16,64")
            .into_iter()
            .filter_map(|value| value.parse().ok())
            .filter(|value: &usize| *value >= 3)
            .collect::<Vec<_>>();
        let workers = std::env::var("BVP_SCI_AOT_POLICY_FULL_SOLVE_WORKER_COUNT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(rayon::current_num_threads);
        let repeats = std::env::var("BVP_SCI_AOT_POLICY_FULL_SOLVE_REPEATS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3);
        let layouts = selected_layouts();
        let frontends = selected_frontends();
        let body = format!(
            "# BVP_sci AOT full-solve execution-policy matrix\n\n- phase: `policy-full-solve`\n- workers_requested: {workers}\n- dimensions: {dimensions:?}\n- nodes: {nodes:?}\n- policies: {policies:?}\n- layouts: {:?}\n- rows report medians over {repeats} independent prepared solves\n- break-even uses inclusive `full_solve_ms` only, against Sequential for the same frontend/layout/dimension/nodes\n- `residual_ms` and `jacobian_ms` are evaluator-only telemetry scopes, matched with the Lambdify policy matrix\n- callback/chunk speed is diagnostic and must not be interpreted as full-solve speedup\n- numerical collocation workspace remains single-owner\n- timing scopes are diagnostic and non-additive\n\n{}",
            layouts
                .iter()
                .map(|layout| layout.label())
                .collect::<Vec<_>>(),
            aot_full_solve_policy_report(
                workers,
                &dimensions,
                &nodes,
                &frontends,
                &policies,
                &layouts,
                repeats,
            )
        );
        let report_name = std::env::var("BVP_SCI_AOT_BENCH_REPORT")
            .unwrap_or_else(|_| format!("aot_policy_full_solve_workers_{workers}"));
        let path = write_test_report("BVP_sci_AOT_Bench", &report_name, &body)
            .expect("write compact AOT full-solve policy report");
        println!("report={}", path.display());
        return;
    }
    let mut rows = Vec::new();
    for dimension in dimensions() {
        for frontend in selected_frontends() {
            for layout in selected_layouts() {
                for policy_name in &policies {
                    rows.push(run_row(frontend, layout, dimension, policy_name));
                }
            }
        }
    }
    let comparison_rows = rows.clone();
    for row in &mut rows {
        row.continuation_vs_lambdify = continuation_comparison(&comparison_rows, row);
    }
    let table = Table::new(&rows).to_string();
    validate_markdown_table(&table);
    let body = format!(
        "# BVP_sci AOT/Lambdify callback matrix\n\n- phase: `{phase}`\n- AOT lifecycle: BuildIfMissing + C/{compiler}\n- `preparation_scope` identifies `lambdify:symbolic+evaluator+plan` versus `aot:BuildIfMissing+compile+link+publish+plan`; `prepare_ms` is not comparable across scope labels or cache policies\n- `residual_ms` and `jacobian_ms` are evaluator-only telemetry scopes; `residual_callback_ms` and `jacobian_callback_ms` include the public callback boundary and caller-owned output work\n- `continuation_ms` is a matched 16-step residual+Jacobian callback series; `continuation_vs_lambdify` compares only the warm series, not cold preparation\n- AOT telemetry is shared IVP lifecycle data and is not additive with BVP scopes\n\n{}",
        table
    );
    let report_name =
        std::env::var("BVP_SCI_AOT_BENCH_REPORT").unwrap_or_else(|_| format!("aot_{phase}"));
    let path = write_test_report("BVP_sci_AOT_Bench", &report_name, &body)
        .expect("write compact BVP_sci AOT report");
    println!("report={}", path.display());
}
