//! Compact Lambdify-only BVP_sci matrix dashboard.
//!
//! This is deliberately a bounded dashboard, not a Criterion group: it emits
//! one table with useful lifecycle numbers and never forwards warm-up chatter
//! into the report. AOT is intentionally absent until its frontend contract is
//! implemented.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecutionPolicy, BvpSciLambdifyPlan,
    BvpSciMatrixLayout, BvpSciOptions, BvpSciSolver, BvpSciTelemetry, BvpSciTelemetrySnapshot,
};
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

#[derive(Clone, Copy)]
enum Workload {
    Linear,
    Parameterized,
    Oscillator,
    StiffDecay,
    Bratu,
    CombustionLike,
    StiffCoupled,
}

impl Workload {
    fn label(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Parameterized => "parameterized-linear",
            Self::Oscillator => "oscillator",
            Self::StiffDecay => "stiff-decay",
            Self::Bratu => "bratu-like",
            Self::CombustionLike => "combustion-like",
            Self::StiffCoupled => "stiff-coupled",
        }
    }

    fn equations(self) -> Vec<Expr> {
        match self {
            Self::Linear => vec![Expr::parse_expression("1")],
            Self::Parameterized => vec![Expr::parse_expression("p")],
            Self::Oscillator => vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            Self::StiffDecay => vec![Expr::parse_expression("-20*y")],
            Self::Bratu => vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("-2*exp(y)"),
            ],
            Self::CombustionLike => vec![
                Expr::parse_expression("q"),
                Expr::parse_expression("-0.5*q + 0.2*exp(Teta)*C0"),
                Expr::parse_expression("J0"),
                Expr::parse_expression("-0.3*J0 - 0.5*C0 + 0.1*Teta"),
                Expr::parse_expression("J1"),
                Expr::parse_expression("-0.2*J1 - 0.3*C1 + 0.05*C0^2"),
            ],
            Self::StiffCoupled => vec![
                Expr::parse_expression("-20*y0+10*y1+9"),
                Expr::parse_expression("-40*y1+20*y2+18-20*x"),
                Expr::parse_expression("-80*y2+77-240*x"),
            ],
        }
    }

    fn state_names(self) -> Vec<String> {
        match self {
            Self::Linear | Self::Parameterized | Self::StiffDecay => vec!["y".into()],
            Self::Oscillator => vec!["y".into(), "z".into()],
            Self::Bratu => vec!["y".into(), "z".into()],
            Self::CombustionLike => vec![
                "Teta".into(),
                "q".into(),
                "C0".into(),
                "J0".into(),
                "C1".into(),
                "J1".into(),
            ],
            Self::StiffCoupled => vec!["y0".into(), "y1".into(), "y2".into()],
        }
    }

    fn parameter_names(self) -> Vec<String> {
        match self {
            Self::Linear => Vec::new(),
            Self::Parameterized => vec!["p".into()],
            Self::Oscillator | Self::StiffDecay => Vec::new(),
            Self::Bratu | Self::CombustionLike | Self::StiffCoupled => Vec::new(),
        }
    }

    fn parameter(self, index: usize) -> Vec<f64> {
        match self {
            Self::Linear => Vec::new(),
            Self::Parameterized => vec![1.0 + index as f64],
            Self::Oscillator
            | Self::StiffDecay
            | Self::Bratu
            | Self::CombustionLike
            | Self::StiffCoupled => Vec::new(),
        }
    }

    fn dimension(self) -> usize {
        self.state_names().len()
    }
}

#[derive(Clone, Copy)]
enum Frontend {
    ExprLegacy,
    AtomNative,
}

impl Frontend {
    fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomNative => "AtomViewNative",
        }
    }

    fn assembly(self) -> BvpSciAssembly {
        match self {
            Self::ExprLegacy => BvpSciAssembly::ExprLegacy,
            Self::AtomNative => BvpSciAssembly::AtomViewNative,
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

    fn value(self, _nodes: usize) -> BvpSciMatrixLayout {
        match self {
            Self::Dense => BvpSciMatrixLayout::Dense,
            Self::Sparse => BvpSciMatrixLayout::Sparse,
            Self::Banded => BvpSciMatrixLayout::Banded {
                // The collocation backend repartitions the global system into
                // a block-tridiagonal core plus a dense endpoint/parameter
                // border. These widths remain part of the public layout
                // contract, while the structured path avoids scalar-band
                // growth with the mesh.
                lower: 256,
                upper: 256,
            },
        }
    }
}

#[derive(Debug, Tabled)]
struct Row {
    workload: String,
    nodes: usize,
    frontend: String,
    layout: String,
    continuation: usize,
    prepare_ms: String,
    solve_ms: String,
    continuation_ms: String,
    continuation_solve_ms: String,
    telemetry_full_solve_ms: String,
    telemetry_full_solve_total_ms: String,
    telemetry_full_solve_calls: u64,
    newton_ms: String,
    mesh_defect_ms: String,
    mesh_refinement_ms: String,
    output_construction_ms: String,
    symbolic_jacobian_ms: String,
    pattern_ms: String,
    lowering_ms: String,
    evaluator_ms: String,
    residual_evaluator_ms: String,
    jacobian_evaluator_ms: String,
    binding_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    collocation_ms: String,
    linear_assembly_ms: String,
    factorization_ms: String,
    linear_solve_ms: String,
    newton_jacobian_refreshes: u64,
    newton_backtracking_trials: u64,
    newton_accepted_steps: u64,
    newton_rejected_steps: u64,
    newton_trace: String,
    banded_route: String,
    banded_structured_factorizations: u64,
    banded_scalar_fallback_factorizations: u64,
    banded_scalar_fallback_assemblies: u64,
    banded_structured_solves: u64,
    banded_scalar_fallback_solves: u64,
    banded_residual_checks: u64,
    banded_fallback_switches: u64,
    banded_rhs_permutations: u64,
    banded_sparse_fallback_factorizations: u64,
    banded_sparse_fallback_solves: u64,
    banded_sparse_fallback_assemblies: u64,
    banded_structured_factorization_ms: String,
    banded_scalar_fallback_factorization_ms: String,
    banded_scalar_fallback_assembly_ms: String,
    banded_structured_solve_ms: String,
    banded_scalar_fallback_solve_ms: String,
    banded_residual_guard_ms: String,
    banded_rhs_permutation_ms: String,
    banded_sparse_fallback_factorization_ms: String,
    banded_sparse_fallback_solve_ms: String,
    banded_sparse_fallback_assembly_ms: String,
    allocations: u64,
    atom_conversions: u64,
    symbolic_derivations: u64,
    pattern_entries: u64,
    evaluator_compilations: u64,
    residual_evaluator_compilations: u64,
    jacobian_evaluator_compilations: u64,
    mesh_refinements: u64,
    factorizations: u64,
    residual_calls: u64,
    jacobian_calls: u64,
    parameter_rebinds: u64,
    continuation_solves: u64,
    parallel_dispatches: u64,
    sequential_dispatches: u64,
    max_worker_threads: u64,
    status: String,
}

#[derive(Debug, Tabled)]
struct FrontendCallbackRow {
    workload: String,
    frontend: String,
    preparation_mode: String,
    repeats: usize,
    prepare_ms: String,
    expr_to_atom_ms: String,
    symbolic_jacobian_ms: String,
    pattern_ms: String,
    lowering_ms: String,
    evaluator_ms: String,
    residual_evaluator_ms: String,
    jacobian_evaluator_ms: String,
    binding_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    atom_conversions: u64,
    symbolic_derivations: u64,
    pattern_entries: u64,
    evaluator_compilations: u64,
    residual_evaluator_compilations: u64,
    jacobian_evaluator_compilations: u64,
    status: String,
}

fn parse_usizes(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .filter(|value: &usize| *value >= 2)
        .collect()
}

fn parse_counts(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .filter(|value: &usize| *value >= 1)
        .collect()
}

fn frontend_callback_row(
    workload: Workload,
    frontend: Frontend,
    repeats: usize,
    warm_rayon: bool,
) -> FrontendCallbackRow {
    let telemetry = BvpSciTelemetry::timings();
    let state_names = workload.state_names();
    let parameters = workload.parameter(0);
    let state = vec![0.25; state_names.len()];
    // Parse/build the symbolic input before the preparation clock. This keeps
    // the frontend comparison about preparation, not workload construction.
    let equations = workload.equations();
    let prepare_started = Instant::now();
    let plan = match BvpSciLambdifyPlan::prepare(
        frontend.assembly(),
        &equations,
        &state_names,
        &workload.parameter_names(),
        "x",
        telemetry,
    ) {
        Ok(plan) => plan,
        Err(error) => {
            return FrontendCallbackRow {
                workload: workload.label().into(),
                frontend: frontend.label().into(),
                preparation_mode: if warm_rayon { "warm" } else { "cold" }.into(),
                repeats,
                prepare_ms: "-".into(),
                expr_to_atom_ms: "-".into(),
                symbolic_jacobian_ms: "-".into(),
                pattern_ms: "-".into(),
                lowering_ms: "-".into(),
                evaluator_ms: "-".into(),
                residual_evaluator_ms: "-".into(),
                jacobian_evaluator_ms: "-".into(),
                binding_ms: "-".into(),
                residual_ms: "-".into(),
                jacobian_ms: "-".into(),
                atom_conversions: 0,
                symbolic_derivations: 0,
                pattern_entries: 0,
                evaluator_compilations: 0,
                residual_evaluator_compilations: 0,
                jacobian_evaluator_compilations: 0,
                status: format!("error: {error}"),
            };
        }
    };
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
    let mut arguments = vec![0.0; 1 + state.len() + parameters.len()];
    let mut residual = vec![0.0; state.len()];
    let mut jacobian = vec![0.0; state.len() * state.len()];
    for _ in 0..repeats {
        if let Err(error) =
            plan.evaluate_rhs(0.5, &state, &parameters, &mut arguments, &mut residual)
        {
            return FrontendCallbackRow {
                workload: workload.label().into(),
                frontend: frontend.label().into(),
                preparation_mode: if warm_rayon { "warm" } else { "cold" }.into(),
                repeats,
                prepare_ms: format!("{prepare_ms:.3}"),
                expr_to_atom_ms: "-".into(),
                symbolic_jacobian_ms: "-".into(),
                pattern_ms: "-".into(),
                lowering_ms: "-".into(),
                evaluator_ms: "-".into(),
                residual_evaluator_ms: "-".into(),
                jacobian_evaluator_ms: "-".into(),
                binding_ms: "-".into(),
                residual_ms: "-".into(),
                jacobian_ms: "-".into(),
                atom_conversions: 0,
                symbolic_derivations: 0,
                pattern_entries: 0,
                evaluator_compilations: 0,
                residual_evaluator_compilations: 0,
                jacobian_evaluator_compilations: 0,
                status: format!("error: {error}"),
            };
        }
        if let Err(error) =
            plan.evaluate_jacobian_dense(0.5, &state, &parameters, &mut arguments, &mut jacobian)
        {
            return FrontendCallbackRow {
                workload: workload.label().into(),
                frontend: frontend.label().into(),
                preparation_mode: if warm_rayon { "warm" } else { "cold" }.into(),
                repeats,
                prepare_ms: format!("{prepare_ms:.3}"),
                expr_to_atom_ms: "-".into(),
                symbolic_jacobian_ms: "-".into(),
                pattern_ms: "-".into(),
                lowering_ms: "-".into(),
                evaluator_ms: "-".into(),
                residual_evaluator_ms: "-".into(),
                jacobian_evaluator_ms: "-".into(),
                binding_ms: "-".into(),
                residual_ms: "-".into(),
                jacobian_ms: "-".into(),
                atom_conversions: 0,
                symbolic_derivations: 0,
                pattern_entries: 0,
                evaluator_compilations: 0,
                residual_evaluator_compilations: 0,
                jacobian_evaluator_compilations: 0,
                status: format!("error: {error}"),
            };
        }
    }
    let snapshot = plan.telemetry_snapshot();
    FrontendCallbackRow {
        workload: workload.label().into(),
        frontend: frontend.label().into(),
        preparation_mode: if warm_rayon { "warm" } else { "cold" }.into(),
        repeats,
        prepare_ms: format!("{prepare_ms:.3}"),
        expr_to_atom_ms: format_opt(snapshot.expr_to_atom_ms),
        symbolic_jacobian_ms: format_opt(snapshot.symbolic_jacobian_ms),
        pattern_ms: format_opt(snapshot.pattern_ms),
        lowering_ms: format_opt(snapshot.lowering_ms),
        evaluator_ms: format_opt(snapshot.evaluator_compilation_ms),
        residual_evaluator_ms: format_opt(snapshot.residual_evaluator_compilation_ms),
        jacobian_evaluator_ms: format_opt(snapshot.jacobian_evaluator_compilation_ms),
        binding_ms: format_opt(snapshot.binding_ms),
        residual_ms: format_opt(snapshot.residual_evaluation_ms),
        jacobian_ms: format_opt(snapshot.jacobian_evaluation_ms),
        atom_conversions: snapshot.atom_conversions,
        symbolic_derivations: snapshot.symbolic_jacobian_derivations,
        pattern_entries: snapshot.pattern_entries,
        evaluator_compilations: snapshot.evaluator_compilations,
        residual_evaluator_compilations: snapshot.residual_evaluator_compilations,
        jacobian_evaluator_compilations: snapshot.jacobian_evaluator_compilations,
        status: "ok".into(),
    }
}

fn frontend_callback_report(workloads: &[Workload], repeats: usize, warm_rayon: bool) -> String {
    if warm_rayon {
        // Atom structural differentiation uses Rayon internally. Keep its
        // one-time global pool bootstrap out of the frontend comparison when
        // the caller requests a warm-process preparation slice.
        let _ = rayon::current_num_threads();
    }
    let mut rows = Vec::new();
    for &workload in workloads {
        for frontend in [Frontend::ExprLegacy, Frontend::AtomNative] {
            rows.push(frontend_callback_row(
                workload, frontend, repeats, warm_rayon,
            ));
        }
    }
    let table = Table::new(rows).to_string();
    validate_markdown_table(&table);
    table
}

#[derive(Clone, Copy)]
enum ContinuationMode {
    Fresh,
    Prepared,
    Warm,
}

impl ContinuationMode {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fresh" => Some(Self::Fresh),
            "prepared" => Some(Self::Prepared),
            "warm" => Some(Self::Warm),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Prepared => "prepared",
            Self::Warm => "warm",
        }
    }
}

#[derive(Debug, Tabled)]
struct ContinuationLifecycleRow {
    mode: String,
    nodes: usize,
    count: usize,
    frontend: String,
    layout: String,
    prepare_ms: String,
    total_ms: String,
    per_solve_ms: String,
    continuation_ms: String,
    factorizations: u64,
    continuation_factorizations: u64,
    sparse_symbolic_analyses: u64,
    sparse_numeric_factorizations: u64,
    parameter_rebinds: u64,
    continuation_solves: u64,
    workspace_resizes: u64,
    logical_allocations: u64,
    retention: String,
    status: String,
}

#[derive(Default)]
struct ContinuationCounters {
    factorizations: u64,
    sparse_symbolic_analyses: u64,
    sparse_numeric_factorizations: u64,
    parameter_rebinds: u64,
    continuation_solves: u64,
    workspace_resizes: u64,
    logical_allocations: u64,
}

#[derive(Clone, Copy)]
enum PolicyCase {
    Sequential,
    Parallel,
    Auto,
}

impl PolicyCase {
    fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::Auto => "auto",
        }
    }

    fn policy(self) -> BvpSciExecutionPolicy {
        match self {
            Self::Sequential => BvpSciExecutionPolicy::Sequential,
            Self::Parallel => BvpSciExecutionPolicy::Parallel { min_work: 0 },
            Self::Auto => BvpSciExecutionPolicy::Auto { min_work: 0 },
        }
    }
}

#[derive(Debug, Tabled)]
struct PolicyFullSolveRow {
    policy: String,
    workers: usize,
    workload: String,
    nodes: usize,
    frontend: String,
    layout: String,
    repeats: usize,
    prepare_ms: String,
    full_solve_ms: String,
    full_solve_calls: u64,
    newton_ms: String,
    mesh_defect_ms: String,
    mesh_refinement_ms: String,
    output_construction_ms: String,
    wall_clock_ms: String,
    callback_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    linear_assembly_ms: String,
    factorization_ms: String,
    linear_solve_ms: String,
    factorizations: u64,
    banded_route: String,
    banded_structured_factorizations: u64,
    banded_scalar_fallback_factorizations: u64,
    banded_scalar_fallback_assemblies: u64,
    banded_structured_solves: u64,
    banded_scalar_fallback_solves: u64,
    banded_residual_checks: u64,
    banded_fallback_switches: u64,
    banded_rhs_permutations: u64,
    banded_sparse_fallback_factorizations: u64,
    banded_sparse_fallback_solves: u64,
    banded_sparse_fallback_assemblies: u64,
    banded_structured_factorization_ms: String,
    banded_scalar_fallback_factorization_ms: String,
    banded_scalar_fallback_assembly_ms: String,
    banded_structured_solve_ms: String,
    banded_scalar_fallback_solve_ms: String,
    banded_residual_guard_ms: String,
    banded_rhs_permutation_ms: String,
    banded_sparse_fallback_factorization_ms: String,
    banded_sparse_fallback_solve_ms: String,
    banded_sparse_fallback_assembly_ms: String,
    parallel_dispatches: u64,
    sequential_dispatches: u64,
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

fn format_median_optional(values: &[Option<f64>]) -> String {
    let mut observed = values.iter().flatten().copied().collect::<Vec<_>>();
    if observed.is_empty() {
        "-".into()
    } else {
        format!("{:.3}", median(&mut observed))
    }
}

fn failed_policy_full_solve_row(
    policy: PolicyCase,
    workers: usize,
    workload: Workload,
    nodes: usize,
    frontend: Frontend,
    layout: Layout,
    repeats: usize,
    error: impl Into<String>,
) -> PolicyFullSolveRow {
    PolicyFullSolveRow {
        policy: policy.label().into(),
        workers,
        workload: workload.label().into(),
        nodes,
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        repeats,
        prepare_ms: "-".into(),
        full_solve_ms: "-".into(),
        full_solve_calls: 0,
        newton_ms: "-".into(),
        mesh_defect_ms: "-".into(),
        mesh_refinement_ms: "-".into(),
        output_construction_ms: "-".into(),
        wall_clock_ms: "-".into(),
        callback_ms: "-".into(),
        residual_ms: "-".into(),
        jacobian_ms: "-".into(),
        linear_assembly_ms: "-".into(),
        factorization_ms: "-".into(),
        linear_solve_ms: "-".into(),
        factorizations: 0,
        banded_route: "not-observed".into(),
        banded_structured_factorizations: 0,
        banded_scalar_fallback_factorizations: 0,
        banded_scalar_fallback_assemblies: 0,
        banded_structured_solves: 0,
        banded_scalar_fallback_solves: 0,
        banded_residual_checks: 0,
        banded_fallback_switches: 0,
        banded_rhs_permutations: 0,
        banded_sparse_fallback_factorizations: 0,
        banded_sparse_fallback_solves: 0,
        banded_sparse_fallback_assemblies: 0,
        banded_structured_factorization_ms: "-".into(),
        banded_scalar_fallback_factorization_ms: "-".into(),
        banded_scalar_fallback_assembly_ms: "-".into(),
        banded_structured_solve_ms: "-".into(),
        banded_scalar_fallback_solve_ms: "-".into(),
        banded_residual_guard_ms: "-".into(),
        banded_rhs_permutation_ms: "-".into(),
        banded_sparse_fallback_factorization_ms: "-".into(),
        banded_sparse_fallback_solve_ms: "-".into(),
        banded_sparse_fallback_assembly_ms: "-".into(),
        parallel_dispatches: 0,
        sequential_dispatches: 0,
        observed_workers: 0,
        break_even_vs_sequential: "not-applicable".into(),
        status: format!("error: {}", error.into()),
    }
}

fn policy_full_solve_row(
    workload: Workload,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    workers: usize,
    policy: PolicyCase,
    repeats: usize,
) -> (PolicyFullSolveRow, Option<f64>) {
    let repeats = repeats.max(1);
    let mut prepare_samples = Vec::with_capacity(repeats);
    let mut solve_samples = Vec::with_capacity(repeats);
    let mut wall_samples = Vec::with_capacity(repeats);
    let mut callback_samples = Vec::with_capacity(repeats);
    let mut residual_samples = Vec::with_capacity(repeats);
    let mut jacobian_samples = Vec::with_capacity(repeats);
    let mut linear_assembly_samples = Vec::with_capacity(repeats);
    let mut factorization_samples = Vec::with_capacity(repeats);
    let mut linear_solve_samples = Vec::with_capacity(repeats);
    let mut banded_structured_factorization_samples = Vec::with_capacity(repeats);
    let mut banded_scalar_fallback_factorization_samples = Vec::with_capacity(repeats);
    let mut banded_scalar_fallback_assembly_samples = Vec::with_capacity(repeats);
    let mut banded_structured_solve_samples = Vec::with_capacity(repeats);
    let mut banded_scalar_fallback_solve_samples = Vec::with_capacity(repeats);
    let mut banded_residual_guard_samples = Vec::with_capacity(repeats);
    let mut banded_rhs_permutation_samples = Vec::with_capacity(repeats);
    let mut banded_sparse_fallback_factorization_samples = Vec::with_capacity(repeats);
    let mut banded_sparse_fallback_solve_samples = Vec::with_capacity(repeats);
    let mut banded_sparse_fallback_assembly_samples = Vec::with_capacity(repeats);
    let mut last_snapshot = None;

    for _ in 0..repeats {
        let row_started = Instant::now();
        let prepare_started = Instant::now();
        let (mut solver, _) =
            match build_solver(workload, frontend, layout, nodes, 0, policy.policy()) {
                Ok(value) => value,
                Err(error) => {
                    return (
                        failed_policy_full_solve_row(
                            policy,
                            workers,
                            workload,
                            nodes,
                            frontend,
                            layout,
                            repeats,
                            error.to_string(),
                        ),
                        None,
                    );
                }
            };
        prepare_samples.push(prepare_started.elapsed().as_secs_f64() * 1e3);
        let solve_started = Instant::now();
        if let Err(error) = solver.solve() {
            return (
                failed_policy_full_solve_row(
                    policy,
                    workers,
                    workload,
                    nodes,
                    frontend,
                    layout,
                    repeats,
                    error.to_string(),
                ),
                None,
            );
        }
        let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
        let snapshot = solver.plan().telemetry_snapshot();
        solve_samples.push(snapshot.full_solve_ms.unwrap_or(solve_ms));
        wall_samples.push(row_started.elapsed().as_secs_f64() * 1e3);
        callback_samples.push(snapshot.callback_ms.unwrap_or_default());
        residual_samples.push(snapshot.residual_evaluation_ms.unwrap_or_default());
        jacobian_samples.push(snapshot.jacobian_evaluation_ms.unwrap_or_default());
        linear_assembly_samples.push(snapshot.linear_assembly_ms);
        factorization_samples.push(snapshot.factorization_ms);
        linear_solve_samples.push(snapshot.solve_ms);
        banded_structured_factorization_samples.push(snapshot.banded_structured_factorization_ms);
        banded_scalar_fallback_factorization_samples
            .push(snapshot.banded_scalar_fallback_factorization_ms);
        banded_scalar_fallback_assembly_samples.push(snapshot.banded_scalar_fallback_assembly_ms);
        banded_structured_solve_samples.push(snapshot.banded_structured_solve_ms);
        banded_scalar_fallback_solve_samples.push(snapshot.banded_scalar_fallback_solve_ms);
        banded_residual_guard_samples.push(snapshot.banded_residual_guard_ms);
        banded_rhs_permutation_samples.push(snapshot.banded_rhs_permutation_ms);
        banded_sparse_fallback_factorization_samples
            .push(snapshot.banded_sparse_fallback_factorization_ms);
        banded_sparse_fallback_solve_samples.push(snapshot.banded_sparse_fallback_solve_ms);
        banded_sparse_fallback_assembly_samples.push(snapshot.banded_sparse_fallback_assembly_ms);
        last_snapshot = Some(snapshot);
    }

    let snapshot = last_snapshot.expect("at least one policy sample");
    let solve_median = median(&mut solve_samples);
    let row = PolicyFullSolveRow {
        policy: policy.label().into(),
        workers,
        workload: workload.label().into(),
        nodes,
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        repeats,
        prepare_ms: format!("{:.3}", median(&mut prepare_samples)),
        full_solve_ms: format!("{solve_median:.3}"),
        full_solve_calls: snapshot.full_solve_calls,
        newton_ms: format_opt(snapshot.newton_ms),
        mesh_defect_ms: format_opt(snapshot.mesh_defect_estimation_ms),
        mesh_refinement_ms: format_opt(snapshot.mesh_refinement_ms),
        output_construction_ms: format_opt(snapshot.output_construction_ms),
        wall_clock_ms: format!("{:.3}", median(&mut wall_samples)),
        callback_ms: format!("{:.3}", median(&mut callback_samples)),
        residual_ms: format!("{:.3}", median(&mut residual_samples)),
        jacobian_ms: format!("{:.3}", median(&mut jacobian_samples)),
        linear_assembly_ms: format_median_optional(&linear_assembly_samples),
        factorization_ms: format_median_optional(&factorization_samples),
        linear_solve_ms: format_median_optional(&linear_solve_samples),
        factorizations: snapshot.factorizations,
        banded_route: if snapshot.banded_sparse_fallback_solves > 0
            || snapshot.banded_sparse_fallback_factorizations > 0
        {
            "structured+sparse-fallback"
        } else if snapshot.banded_scalar_fallback_solves > 0
            || snapshot.banded_scalar_fallback_factorizations > 0
        {
            "structured+scalar-fallback"
        } else if snapshot.banded_structured_solves > 0
            || snapshot.banded_structured_factorizations > 0
        {
            "structured"
        } else {
            "not-observed"
        }
        .into(),
        banded_structured_factorizations: snapshot.banded_structured_factorizations,
        banded_scalar_fallback_factorizations: snapshot.banded_scalar_fallback_factorizations,
        banded_scalar_fallback_assemblies: snapshot.banded_scalar_fallback_assemblies,
        banded_structured_solves: snapshot.banded_structured_solves,
        banded_scalar_fallback_solves: snapshot.banded_scalar_fallback_solves,
        banded_residual_checks: snapshot.banded_residual_checks,
        banded_fallback_switches: snapshot.banded_fallback_switches,
        banded_rhs_permutations: snapshot.banded_rhs_permutations,
        banded_sparse_fallback_factorizations: snapshot.banded_sparse_fallback_factorizations,
        banded_sparse_fallback_solves: snapshot.banded_sparse_fallback_solves,
        banded_sparse_fallback_assemblies: snapshot.banded_sparse_fallback_assemblies,
        banded_structured_factorization_ms: format_median_optional(
            &banded_structured_factorization_samples,
        ),
        banded_scalar_fallback_factorization_ms: format_median_optional(
            &banded_scalar_fallback_factorization_samples,
        ),
        banded_scalar_fallback_assembly_ms: format_median_optional(
            &banded_scalar_fallback_assembly_samples,
        ),
        banded_structured_solve_ms: format_median_optional(&banded_structured_solve_samples),
        banded_scalar_fallback_solve_ms: format_median_optional(
            &banded_scalar_fallback_solve_samples,
        ),
        banded_residual_guard_ms: format_median_optional(&banded_residual_guard_samples),
        banded_rhs_permutation_ms: format_median_optional(&banded_rhs_permutation_samples),
        banded_sparse_fallback_factorization_ms: format_median_optional(
            &banded_sparse_fallback_factorization_samples,
        ),
        banded_sparse_fallback_solve_ms: format_median_optional(
            &banded_sparse_fallback_solve_samples,
        ),
        banded_sparse_fallback_assembly_ms: format_median_optional(
            &banded_sparse_fallback_assembly_samples,
        ),
        parallel_dispatches: snapshot.parallel_dispatches,
        sequential_dispatches: snapshot.sequential_dispatches,
        observed_workers: snapshot.max_worker_threads,
        break_even_vs_sequential: "baseline".into(),
        status: "ok".into(),
    };
    (row, Some(solve_median))
}

fn policy_full_solve_report(
    workers: usize,
    nodes: &[usize],
    workloads: &[Workload],
    frontends: &[Frontend],
    layouts: &[Layout],
    repeats: usize,
) -> String {
    let mut rows = Vec::new();
    for &workload in workloads {
        for &node_count in nodes {
            for &layout in layouts {
                for &frontend in frontends {
                    for policy in [
                        PolicyCase::Sequential,
                        PolicyCase::Parallel,
                        PolicyCase::Auto,
                    ] {
                        let (row, solve_ms) = policy_full_solve_row(
                            workload, frontend, layout, node_count, workers, policy, repeats,
                        );
                        rows.push((row, solve_ms));
                    }
                }
            }
        }
    }

    let mut rendered = Vec::with_capacity(rows.len());
    for (mut row, solve_ms) in rows {
        if let (Some(value), Some(baseline)) = (solve_ms, find_baseline(&rendered, &row)) {
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

fn selected_policy_frontends() -> Vec<Frontend> {
    let selected = std::env::var("BVP_SCI_BENCH_POLICY_FULL_SOLVE_FRONTENDS")
        .unwrap_or_else(|_| "expr-legacy,atom-native".into())
        .split(',')
        .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "expr-legacy" | "exprlegacy" => Some(Frontend::ExprLegacy),
            "atom-native" | "atomview" | "atomview-native" => Some(Frontend::AtomNative),
            _ => None,
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        vec![Frontend::ExprLegacy, Frontend::AtomNative]
    } else {
        selected
    }
}

fn find_baseline(rows: &[PolicyFullSolveRow], row: &PolicyFullSolveRow) -> Option<f64> {
    rows.iter()
        .find(|candidate| {
            candidate.policy == "sequential"
                && candidate.workers == row.workers
                && candidate.workload == row.workload
                && candidate.nodes == row.nodes
                && candidate.frontend == row.frontend
                && candidate.layout == row.layout
                && candidate.status == "ok"
        })
        .and_then(|candidate| candidate.full_solve_ms.parse().ok())
}

fn parameterized_plan(frontend: Frontend) -> Result<BvpSciLambdifyPlan, String> {
    BvpSciLambdifyPlan::prepare(
        frontend.assembly(),
        // This is an exactly solvable continuation family:
        // y' = p * (y + 1), y(0) = 0, y(1) = exp(p) - 1.
        &[Expr::parse_expression("p*(y + 1)")],
        &["y".into()],
        &["p".into()],
        "x",
        BvpSciTelemetry::timings(),
    )
    .map_err(|error| error.to_string())
}

fn parameterized_solver(
    plan: BvpSciLambdifyPlan,
    layout: Layout,
    nodes: usize,
    parameter: f64,
) -> Result<(BvpSciSolver, Arc<AtomicU64>), String> {
    let target = Arc::new(AtomicU64::new(parameter.to_bits()));
    let target_for_boundary = Arc::clone(&target);
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        move |ya, yb, _, output| {
            let target = f64::from_bits(target_for_boundary.load(Ordering::Relaxed));
            output[0] = ya[0];
            output[1] = yb[0] - (target.exp() - 1.0);
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let x: Vec<f64> = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect();
    // Do not seed the lifecycle benchmark with the exact solution.  That
    // makes fresh/prepared rows measure an already-converged residual while a
    // warm row measures a real continuation solve, which is not an apples-to-
    // apples comparison of Newton/Jacobian/LU work.  Keep the same bounded
    // perturbation for every lifecycle mode; warm still starts subsequent
    // members from the preceding converged solution by design.
    let y: Vec<f64> = x
        .iter()
        .map(|&value| {
            let exact = (parameter * value).exp() - 1.0;
            0.85 * exact + 0.03 * (std::f64::consts::PI * value).sin()
        })
        .collect();
    let mut options = BvpSciOptions::default();
    options.matrix_layout = layout.value(nodes);
    options.tolerance = 1e-5;
    // Keep the requested mesh small while giving the exact continuation
    // family enough adaptive headroom to converge. A failed row must remain a
    // visible diagnostic, but it should not be caused by an arbitrary
    // nodes*4/2 refinement ceiling.
    options.max_nodes = nodes.saturating_mul(16).max(128);
    options.max_mesh_refinements = 8;
    options.max_newton_iterations = 20;
    options.max_jacobian_refreshes = 10;
    BvpSciSolver::new(plan, boundary, x, y, vec![parameter], options)
        .map(|solver| (solver, target))
        .map_err(|error| error.to_string())
}

fn continuation_parameter(index: usize, count: usize) -> f64 {
    if count <= 1 {
        1.0
    } else {
        1.0 + 7.0 * index as f64 / (count - 1) as f64
    }
}

fn run_continuation_lifecycle_row(
    mode: ContinuationMode,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    count: usize,
) -> Result<ContinuationLifecycleRow, String> {
    let started = Instant::now();
    let mut prepare_ms = 0.0;
    let mut continuation_ms = None;
    let mut continuation_factorizations = 0;
    let mut retention = "not-applicable";
    let mut counters = ContinuationCounters::default();

    match mode {
        ContinuationMode::Fresh => {
            for index in 0..count {
                let prepare_started = Instant::now();
                let plan = parameterized_plan(frontend)?;
                prepare_ms += prepare_started.elapsed().as_secs_f64() * 1e3;
                let (mut solver, _) = parameterized_solver(
                    plan,
                    layout,
                    nodes,
                    continuation_parameter(index, count),
                )?;
                solver.solve().map_err(|error| error.to_string())?;
                let snapshot = solver.plan().telemetry_snapshot();
                counters.factorizations += snapshot.factorizations;
                counters.sparse_symbolic_analyses += snapshot.sparse_symbolic_analyses;
                counters.sparse_numeric_factorizations += snapshot.sparse_numeric_factorizations;
                counters.workspace_resizes += snapshot.workspace_resizes;
                counters.logical_allocations += snapshot.allocations;
            }
        }
        ContinuationMode::Prepared => {
            let prepare_started = Instant::now();
            let template = parameterized_plan(frontend)?;
            prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
            for index in 0..count {
                let (mut solver, _) = parameterized_solver(
                    template.clone(),
                    layout,
                    nodes,
                    continuation_parameter(index, count),
                )?;
                solver.solve().map_err(|error| error.to_string())?;
            }
            let snapshot = template.telemetry_snapshot();
            counters.factorizations = snapshot.factorizations;
            counters.sparse_symbolic_analyses = snapshot.sparse_symbolic_analyses;
            counters.sparse_numeric_factorizations = snapshot.sparse_numeric_factorizations;
            counters.workspace_resizes = snapshot.workspace_resizes;
            counters.logical_allocations = snapshot.allocations;
        }
        ContinuationMode::Warm => {
            let prepare_started = Instant::now();
            let plan = parameterized_plan(frontend)?;
            prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
            let (mut solver, target) =
                parameterized_solver(plan, layout, nodes, continuation_parameter(0, count))?;
            solver.solve().map_err(|error| error.to_string())?;
            let initial = solver.plan().telemetry_snapshot();
            let continuation_started = Instant::now();
            for index in 1..count {
                let parameter = continuation_parameter(index, count);
                target.store(parameter.to_bits(), Ordering::Relaxed);
                solver
                    .set_parameters(vec![parameter])
                    .map_err(|error| error.to_string())?;
                solver.solve().map_err(|error| error.to_string())?;
            }
            continuation_ms = Some(continuation_started.elapsed().as_secs_f64() * 1e3);
            let final_snapshot = solver.plan().telemetry_snapshot();
            continuation_factorizations = final_snapshot
                .factorizations
                .saturating_sub(initial.factorizations);
            let retained = final_snapshot.workspace_resizes == initial.workspace_resizes
                && final_snapshot.allocations == initial.allocations;
            retention = if retained { "bounded" } else { "grew" };
            counters.factorizations = final_snapshot.factorizations;
            counters.sparse_symbolic_analyses = final_snapshot.sparse_symbolic_analyses;
            counters.sparse_numeric_factorizations = final_snapshot.sparse_numeric_factorizations;
            counters.parameter_rebinds = final_snapshot.parameter_rebinds;
            counters.continuation_solves = final_snapshot.continuation_solves;
            counters.workspace_resizes = final_snapshot.workspace_resizes;
            counters.logical_allocations = final_snapshot.allocations;
        }
    }

    let total_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(ContinuationLifecycleRow {
        mode: mode.label().into(),
        nodes,
        count,
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        prepare_ms: format!("{prepare_ms:.3}"),
        total_ms: format!("{total_ms:.3}"),
        per_solve_ms: format!("{:.3}", total_ms / count.max(1) as f64),
        continuation_ms: continuation_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".into()),
        factorizations: counters.factorizations,
        continuation_factorizations,
        sparse_symbolic_analyses: counters.sparse_symbolic_analyses,
        sparse_numeric_factorizations: counters.sparse_numeric_factorizations,
        parameter_rebinds: counters.parameter_rebinds,
        continuation_solves: counters.continuation_solves,
        workspace_resizes: counters.workspace_resizes,
        logical_allocations: counters.logical_allocations,
        retention: retention.into(),
        status: "ok".into(),
    })
}

fn failed_continuation_lifecycle_row(
    mode: ContinuationMode,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    count: usize,
    error: impl Into<String>,
) -> ContinuationLifecycleRow {
    ContinuationLifecycleRow {
        mode: mode.label().into(),
        nodes,
        count,
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        prepare_ms: "-".into(),
        total_ms: "-".into(),
        per_solve_ms: "-".into(),
        continuation_ms: "-".into(),
        factorizations: 0,
        continuation_factorizations: 0,
        sparse_symbolic_analyses: 0,
        sparse_numeric_factorizations: 0,
        parameter_rebinds: 0,
        continuation_solves: 0,
        workspace_resizes: 0,
        logical_allocations: 0,
        retention: "unknown".into(),
        status: format!("error: {}", error.into()),
    }
}

fn continuation_lifecycle_report(
    nodes: &[usize],
    counts: &[usize],
    frontends: &[Frontend],
    layouts: &[Layout],
    modes: &[ContinuationMode],
) -> Result<String, String> {
    let mut rows = Vec::new();
    for &node_count in nodes {
        for &count in counts {
            for &frontend in frontends {
                for &layout in layouts {
                    for &mode in modes {
                        rows.push(
                            match run_continuation_lifecycle_row(
                                mode, frontend, layout, node_count, count,
                            ) {
                                Ok(row) => row,
                                Err(error) => failed_continuation_lifecycle_row(
                                    mode, frontend, layout, node_count, count, error,
                                ),
                            },
                        );
                    }
                }
            }
        }
    }
    let table = Table::new(rows).to_string();
    validate_markdown_table(&table);
    Ok(table)
}

fn selected_workloads() -> Vec<String> {
    std::env::var("BVP_SCI_BENCH_WORKLOADS")
        .unwrap_or_else(|_| {
            "linear,parameterized-linear,oscillator,stiff-decay,bratu-like,combustion-like,stiff-coupled".into()
        })
        .split(',')
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn policy_workload(value: &str) -> Option<Workload> {
    match value.trim().to_ascii_lowercase().as_str() {
        "stiff-coupled" => Some(Workload::StiffCoupled),
        "combustion-like" => Some(Workload::CombustionLike),
        "stiff-decay" => Some(Workload::StiffDecay),
        "bratu-like" => Some(Workload::Bratu),
        "linear" => Some(Workload::Linear),
        "parameterized-linear" => Some(Workload::Parameterized),
        "oscillator" => Some(Workload::Oscillator),
        _ => None,
    }
}

fn selected_policy_workloads() -> Vec<Workload> {
    std::env::var("BVP_SCI_BENCH_POLICY_FULL_SOLVE_WORKLOADS")
        .unwrap_or_else(|_| "stiff-coupled,combustion-like".into())
        .split(',')
        .filter_map(policy_workload)
        .collect()
}

fn policy_layout(value: &str) -> Option<Layout> {
    match value.trim().to_ascii_lowercase().as_str() {
        "dense" => Some(Layout::Dense),
        "sparse" => Some(Layout::Sparse),
        "banded" => Some(Layout::Banded),
        _ => None,
    }
}

fn selected_policy_layouts() -> Vec<Layout> {
    std::env::var("BVP_SCI_BENCH_POLICY_FULL_SOLVE_LAYOUTS")
        .unwrap_or_else(|_| "sparse,banded".into())
        .split(',')
        .filter_map(policy_layout)
        .collect()
}

fn selected_continuation_layouts() -> Vec<Layout> {
    std::env::var("BVP_SCI_BENCH_CONTINUATION_LAYOUTS")
        .unwrap_or_else(|_| "dense,sparse,banded".into())
        .split(',')
        .filter_map(policy_layout)
        .collect()
}

fn selected_policy() -> BvpSciExecutionPolicy {
    let min_work = std::env::var("BVP_SCI_BENCH_POLICY_MIN_WORK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    match std::env::var("BVP_SCI_BENCH_POLICY")
        .unwrap_or_else(|_| "sequential".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "parallel" => BvpSciExecutionPolicy::Parallel { min_work },
        "auto" => BvpSciExecutionPolicy::Auto { min_work },
        _ => BvpSciExecutionPolicy::Sequential,
    }
}

fn build_solver(
    workload: Workload,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    parameter_index: usize,
    execution_policy: BvpSciExecutionPolicy,
) -> Result<(BvpSciSolver, Arc<AtomicU64>), String> {
    let parameter_names = workload.parameter_names();
    let parameter = workload.parameter(parameter_index);
    let telemetry = BvpSciTelemetry::timings();
    let plan = BvpSciLambdifyPlan::prepare(
        frontend.assembly(),
        &workload.equations(),
        &workload.state_names(),
        &parameter_names,
        "x",
        telemetry,
    )
    .map_err(|error| error.to_string())?;
    let target = Arc::new(AtomicU64::new(
        parameter.first().copied().unwrap_or(1.0).to_bits(),
    ));
    let target_for_boundary = Arc::clone(&target);
    let boundary = match workload {
        Workload::Linear => BvpSciBoundaryCallbacks::new(
            1,
            |_, yb, _, output| {
                output[0] = yb[0] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        Workload::Parameterized => BvpSciBoundaryCallbacks::new(
            2,
            move |ya, yb, _, output| {
                let target = f64::from_bits(target_for_boundary.load(Ordering::Relaxed));
                output[0] = ya[0];
                output[1] = yb[0] - target;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        Workload::Oscillator => BvpSciBoundaryCallbacks::new(
            2,
            |ya, yb, _, output| {
                output[0] = ya[0];
                output[1] = yb[0] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        Workload::StiffDecay => BvpSciBoundaryCallbacks::new(
            1,
            |ya, _, _, output| {
                output[0] = ya[0] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        Workload::Bratu => BvpSciBoundaryCallbacks::new(
            2,
            |ya, yb, _, output| {
                output[0] = ya[0];
                output[1] = yb[0];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        Workload::CombustionLike => BvpSciBoundaryCallbacks::new(
            6,
            |ya, yb, _, output| {
                output[0] = ya[0];
                output[1] = yb[1];
                output[2] = ya[2] - 1.0;
                output[3] = yb[3];
                output[4] = ya[4];
                output[5] = yb[5];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        Workload::StiffCoupled => BvpSciBoundaryCallbacks::new(
            3,
            |ya, _, _, output| {
                output[0] = ya[0] - 1.0;
                output[1] = ya[1] - 1.0;
                output[2] = ya[2] - 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
    };
    let mut options = BvpSciOptions::default();
    options.matrix_layout = layout.value(nodes);
    options.execution_policy = execution_policy;
    options.tolerance = 1e-6;
    options.max_nodes = if matches!(workload, Workload::Bratu) {
        // Bratu is a nonlinear controller workload, not a fixed-mesh smoke
        // fixture. Keep the requested initial mesh in the report while
        // allowing a bounded refinement budget for coarse matrix rows.
        nodes.saturating_mul(4).max(16)
    } else {
        nodes.saturating_mul(4).max(nodes)
    };
    if matches!(workload, Workload::Bratu) {
        options.tolerance = 1e-4;
        options.max_mesh_refinements = 2;
    }
    if matches!(workload, Workload::StiffCoupled) {
        // Keep the performance fixture aligned with the exact story gate:
        // stiff Newton work needs an explicit budget rather than the small
        // default intended for cheap smoke workloads.
        options.max_mesh_refinements = 0;
        options.max_newton_iterations = 30;
        // This workload can require more than twelve modified-Newton
        // refreshes on an intermediate mesh even while the residual contracts
        // monotonically. Keep the benchmark a convergence/performance gate,
        // rather than turning the refresh budget into a false row failure.
        options.max_jacobian_refreshes = 30;
    }
    let right = if matches!(workload, Workload::Oscillator) {
        std::f64::consts::FRAC_PI_2
    } else {
        1.0
    };
    let x: Vec<f64> = (0..nodes)
        .map(|index| right * index as f64 / (nodes - 1) as f64)
        .collect();
    // Deliberately avoid the exact analytical solution.  The dashboard must
    // exercise Newton, Jacobian assembly and factorization rather than only
    // measuring the already-converged residual path.
    let y = if matches!(workload, Workload::CombustionLike) {
        x.iter()
            .flat_map(|_| [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            .collect()
    } else if matches!(workload, Workload::StiffCoupled) {
        x.iter()
            .flat_map(|&x| {
                [
                    0.9 * (1.0 - x),
                    0.9 * (1.0 - 2.0 * x),
                    0.9 * (1.0 - 3.0 * x),
                ]
            })
            .collect()
    } else if matches!(workload, Workload::Oscillator) {
        x.iter()
            .flat_map(|&x| [0.9 * x.sin(), 0.9 * x.cos()])
            .collect()
    } else if matches!(workload, Workload::StiffDecay) {
        x.iter().map(|&x| 0.9 * (-20.0 * x).exp()).collect()
    } else {
        vec![0.0; x.len() * workload.dimension()]
    };
    BvpSciSolver::new(plan, boundary, x, y, parameter, options)
        .map(|solver| (solver, target))
        .map_err(|error| error.to_string())
}

fn run_row(
    workload: Workload,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    continuation: usize,
) -> Row {
    // The manufactured stiff system needs enough collocation intervals to
    // resolve its coupled profile.  Do not turn a too-coarse user selection
    // into a false solver failure; report the effective mesh in the table.
    let nodes = if matches!(workload, Workload::StiffDecay) {
        nodes.max(64)
    } else if matches!(workload, Workload::StiffCoupled) {
        nodes.max(20)
    } else {
        nodes
    };
    let prepare_started = Instant::now();
    let (mut solver, target) =
        match build_solver(workload, frontend, layout, nodes, 0, selected_policy()) {
            Ok(solver) => solver,
            Err(error) => {
                return failed_row(workload, frontend, layout, nodes, continuation, error);
            }
        };
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
    let solve_started = Instant::now();
    if let Err(error) = solver.solve() {
        return failed_solver_row(
            workload,
            frontend,
            layout,
            nodes,
            continuation,
            &solver,
            error,
        );
    }
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
    let effective_continuation = if workload.parameter_names().is_empty() {
        1
    } else {
        continuation
    };
    let continuation_started = Instant::now();
    for index in 1..effective_continuation {
        target.store(
            workload
                .parameter(index)
                .first()
                .copied()
                .unwrap_or(1.0)
                .to_bits(),
            Ordering::Relaxed,
        );
        if let Err(error) = solver.set_parameters(workload.parameter(index)) {
            return failed_solver_row(
                workload,
                frontend,
                layout,
                nodes,
                continuation,
                &solver,
                error,
            );
        }
        if let Err(error) = solver.solve() {
            return failed_solver_row(
                workload,
                frontend,
                layout,
                nodes,
                continuation,
                &solver,
                error,
            );
        }
    }
    let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
    let continuation_steps = effective_continuation.saturating_sub(1);
    let telemetry = solver.plan().telemetry_snapshot();
    let fmt = |value: Option<f64>| {
        value
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".into())
    };
    Row {
        workload: workload.label().into(),
        nodes,
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        continuation: effective_continuation,
        prepare_ms: format!("{prepare_ms:.3}"),
        solve_ms: format!("{solve_ms:.3}"),
        continuation_ms: format!("{continuation_ms:.3}"),
        continuation_solve_ms: if continuation_steps > 0 {
            format!("{:.3}", continuation_ms / continuation_steps as f64)
        } else {
            "-".into()
        },
        telemetry_full_solve_ms: fmt(telemetry.full_solve_ms),
        telemetry_full_solve_total_ms: fmt(telemetry.full_solve_total_ms),
        telemetry_full_solve_calls: telemetry.full_solve_calls,
        newton_ms: fmt(telemetry.newton_ms),
        mesh_defect_ms: fmt(telemetry.mesh_defect_estimation_ms),
        mesh_refinement_ms: fmt(telemetry.mesh_refinement_ms),
        output_construction_ms: fmt(telemetry.output_construction_ms),
        symbolic_jacobian_ms: fmt(telemetry.symbolic_jacobian_ms),
        pattern_ms: fmt(telemetry.pattern_ms),
        lowering_ms: fmt(telemetry.lowering_ms),
        evaluator_ms: fmt(telemetry.evaluator_compilation_ms),
        residual_evaluator_ms: fmt(telemetry.residual_evaluator_compilation_ms),
        jacobian_evaluator_ms: fmt(telemetry.jacobian_evaluator_compilation_ms),
        binding_ms: fmt(telemetry.binding_ms),
        residual_ms: fmt(telemetry.residual_evaluation_ms),
        jacobian_ms: fmt(telemetry.jacobian_evaluation_ms),
        collocation_ms: fmt(telemetry.collocation_ms),
        linear_assembly_ms: fmt(telemetry.linear_assembly_ms),
        factorization_ms: fmt(telemetry.factorization_ms),
        linear_solve_ms: fmt(telemetry.linear_solve_ms),
        newton_jacobian_refreshes: telemetry.newton_jacobian_refreshes,
        newton_backtracking_trials: telemetry.newton_backtracking_trials,
        newton_accepted_steps: telemetry.newton_accepted_steps,
        newton_rejected_steps: telemetry.newton_rejected_steps,
        newton_trace: format_newton_trace(&telemetry),
        banded_route: if telemetry.banded_sparse_fallback_solves > 0
            || telemetry.banded_sparse_fallback_factorizations > 0
        {
            "structured+sparse-fallback".into()
        } else if telemetry.banded_scalar_fallback_solves > 0
            || telemetry.banded_scalar_fallback_factorizations > 0
        {
            "scalar-fallback".into()
        } else if telemetry.banded_structured_solves > 0
            || telemetry.banded_structured_factorizations > 0
        {
            "structured".into()
        } else {
            "-".into()
        },
        banded_structured_factorizations: telemetry.banded_structured_factorizations,
        banded_scalar_fallback_factorizations: telemetry.banded_scalar_fallback_factorizations,
        banded_scalar_fallback_assemblies: telemetry.banded_scalar_fallback_assemblies,
        banded_structured_solves: telemetry.banded_structured_solves,
        banded_scalar_fallback_solves: telemetry.banded_scalar_fallback_solves,
        banded_residual_checks: telemetry.banded_residual_checks,
        banded_fallback_switches: telemetry.banded_fallback_switches,
        banded_rhs_permutations: telemetry.banded_rhs_permutations,
        banded_sparse_fallback_factorizations: telemetry.banded_sparse_fallback_factorizations,
        banded_sparse_fallback_solves: telemetry.banded_sparse_fallback_solves,
        banded_sparse_fallback_assemblies: telemetry.banded_sparse_fallback_assemblies,
        banded_structured_factorization_ms: fmt(telemetry.banded_structured_factorization_ms),
        banded_scalar_fallback_factorization_ms: fmt(
            telemetry.banded_scalar_fallback_factorization_ms
        ),
        banded_scalar_fallback_assembly_ms: fmt(telemetry.banded_scalar_fallback_assembly_ms),
        banded_structured_solve_ms: fmt(telemetry.banded_structured_solve_ms),
        banded_scalar_fallback_solve_ms: fmt(telemetry.banded_scalar_fallback_solve_ms),
        banded_residual_guard_ms: fmt(telemetry.banded_residual_guard_ms),
        banded_rhs_permutation_ms: fmt(telemetry.banded_rhs_permutation_ms),
        banded_sparse_fallback_factorization_ms: fmt(
            telemetry.banded_sparse_fallback_factorization_ms
        ),
        banded_sparse_fallback_solve_ms: fmt(telemetry.banded_sparse_fallback_solve_ms),
        banded_sparse_fallback_assembly_ms: fmt(telemetry.banded_sparse_fallback_assembly_ms),
        allocations: telemetry.allocations,
        atom_conversions: telemetry.atom_conversions,
        symbolic_derivations: telemetry.symbolic_jacobian_derivations,
        pattern_entries: telemetry.pattern_entries,
        evaluator_compilations: telemetry.evaluator_compilations,
        residual_evaluator_compilations: telemetry.residual_evaluator_compilations,
        jacobian_evaluator_compilations: telemetry.jacobian_evaluator_compilations,
        mesh_refinements: telemetry.mesh_refinements,
        factorizations: telemetry.factorizations,
        residual_calls: telemetry.residual_evaluations,
        jacobian_calls: telemetry.jacobian_evaluations,
        parameter_rebinds: telemetry.parameter_rebinds,
        continuation_solves: telemetry.continuation_solves,
        parallel_dispatches: telemetry.parallel_dispatches,
        sequential_dispatches: telemetry.sequential_dispatches,
        max_worker_threads: telemetry.max_worker_threads,
        status: "ok".into(),
    }
}

fn failed_row(
    workload: Workload,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    continuation: usize,
    error: impl std::fmt::Display,
) -> Row {
    empty_row(
        workload,
        frontend,
        layout,
        nodes,
        continuation,
        format!("error: {error}"),
    )
}

fn format_newton_trace(telemetry: &BvpSciTelemetrySnapshot) -> String {
    if telemetry.newton_residual_history.is_empty() {
        return "-".into();
    }
    telemetry
        .newton_residual_history
        .iter()
        .map(|entry| {
            let after = entry
                .residual_after
                .map(|value| format!("{value:.3e}"))
                .unwrap_or_else(|| "-".into());
            format!(
                "{}:{:.3e}->{after};step={:.3e};bt={};jr={};{}",
                entry.iteration,
                entry.residual_before,
                entry.step_inf_norm,
                entry.backtracking_trials,
                entry.jacobian_refreshed as u8,
                if entry.accepted { "ok" } else { "reject" }
            )
        })
        // A pipe is the Markdown column delimiter. Keep the full numeric
        // trace in one cell without making the generated table ambiguous.
        .collect::<Vec<_>>()
        .join(" / ")
}

fn failed_solver_row(
    workload: Workload,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    continuation: usize,
    solver: &BvpSciSolver,
    error: impl std::fmt::Display,
) -> Row {
    let telemetry = solver.plan().telemetry_snapshot();
    let mut row = empty_row(
        workload,
        frontend,
        layout,
        nodes,
        continuation,
        format!("error: {error}"),
    );
    row.newton_jacobian_refreshes = telemetry.newton_jacobian_refreshes;
    row.newton_backtracking_trials = telemetry.newton_backtracking_trials;
    row.newton_accepted_steps = telemetry.newton_accepted_steps;
    row.newton_rejected_steps = telemetry.newton_rejected_steps;
    row.newton_trace = format_newton_trace(&telemetry);
    row.banded_structured_factorizations = telemetry.banded_structured_factorizations;
    row.banded_scalar_fallback_factorizations = telemetry.banded_scalar_fallback_factorizations;
    row.banded_scalar_fallback_assemblies = telemetry.banded_scalar_fallback_assemblies;
    row.banded_structured_solves = telemetry.banded_structured_solves;
    row.banded_scalar_fallback_solves = telemetry.banded_scalar_fallback_solves;
    row.banded_residual_checks = telemetry.banded_residual_checks;
    row.banded_fallback_switches = telemetry.banded_fallback_switches;
    row.residual_evaluator_ms = format_opt(telemetry.residual_evaluator_compilation_ms);
    row.jacobian_evaluator_ms = format_opt(telemetry.jacobian_evaluator_compilation_ms);
    row.atom_conversions = telemetry.atom_conversions;
    row.symbolic_derivations = telemetry.symbolic_jacobian_derivations;
    row.pattern_entries = telemetry.pattern_entries;
    row.evaluator_compilations = telemetry.evaluator_compilations;
    row.residual_evaluator_compilations = telemetry.residual_evaluator_compilations;
    row.jacobian_evaluator_compilations = telemetry.jacobian_evaluator_compilations;
    row.factorizations = telemetry.factorizations;
    row.residual_calls = telemetry.residual_evaluations;
    row.jacobian_calls = telemetry.jacobian_evaluations;
    row.banded_route = if telemetry.banded_scalar_fallback_solves > 0
        || telemetry.banded_scalar_fallback_factorizations > 0
    {
        "scalar-fallback".into()
    } else if telemetry.banded_structured_solves > 0
        || telemetry.banded_structured_factorizations > 0
    {
        "structured".into()
    } else {
        "-".into()
    };
    row
}

fn empty_row(
    workload: Workload,
    frontend: Frontend,
    layout: Layout,
    nodes: usize,
    continuation: usize,
    status: impl Into<String>,
) -> Row {
    Row {
        workload: workload.label().into(),
        nodes,
        frontend: frontend.label().into(),
        layout: layout.label().into(),
        continuation: if workload.parameter_names().is_empty() {
            1
        } else {
            continuation
        },
        prepare_ms: "-".into(),
        solve_ms: "-".into(),
        continuation_ms: "-".into(),
        continuation_solve_ms: "-".into(),
        telemetry_full_solve_ms: "-".into(),
        telemetry_full_solve_total_ms: "-".into(),
        telemetry_full_solve_calls: 0,
        newton_ms: "-".into(),
        mesh_defect_ms: "-".into(),
        mesh_refinement_ms: "-".into(),
        output_construction_ms: "-".into(),
        symbolic_jacobian_ms: "-".into(),
        pattern_ms: "-".into(),
        lowering_ms: "-".into(),
        evaluator_ms: "-".into(),
        residual_evaluator_ms: "-".into(),
        jacobian_evaluator_ms: "-".into(),
        binding_ms: "-".into(),
        residual_ms: "-".into(),
        jacobian_ms: "-".into(),
        collocation_ms: "-".into(),
        linear_assembly_ms: "-".into(),
        factorization_ms: "-".into(),
        linear_solve_ms: "-".into(),
        newton_jacobian_refreshes: 0,
        newton_backtracking_trials: 0,
        newton_accepted_steps: 0,
        newton_rejected_steps: 0,
        newton_trace: "-".into(),
        banded_route: "-".into(),
        banded_structured_factorizations: 0,
        banded_scalar_fallback_factorizations: 0,
        banded_scalar_fallback_assemblies: 0,
        banded_structured_solves: 0,
        banded_scalar_fallback_solves: 0,
        banded_residual_checks: 0,
        banded_fallback_switches: 0,
        banded_rhs_permutations: 0,
        banded_sparse_fallback_factorizations: 0,
        banded_sparse_fallback_solves: 0,
        banded_sparse_fallback_assemblies: 0,
        banded_structured_factorization_ms: "-".into(),
        banded_scalar_fallback_factorization_ms: "-".into(),
        banded_scalar_fallback_assembly_ms: "-".into(),
        banded_structured_solve_ms: "-".into(),
        banded_scalar_fallback_solve_ms: "-".into(),
        banded_residual_guard_ms: "-".into(),
        banded_rhs_permutation_ms: "-".into(),
        banded_sparse_fallback_factorization_ms: "-".into(),
        banded_sparse_fallback_solve_ms: "-".into(),
        banded_sparse_fallback_assembly_ms: "-".into(),
        allocations: 0,
        atom_conversions: 0,
        symbolic_derivations: 0,
        pattern_entries: 0,
        evaluator_compilations: 0,
        residual_evaluator_compilations: 0,
        jacobian_evaluator_compilations: 0,
        mesh_refinements: 0,
        factorizations: 0,
        residual_calls: 0,
        jacobian_calls: 0,
        parameter_rebinds: 0,
        continuation_solves: 0,
        parallel_dispatches: 0,
        sequential_dispatches: 0,
        max_worker_threads: 0,
        status: status.into(),
    }
}

fn main() {
    let phase = std::env::var("BVP_SCI_BENCH_PHASE")
        .unwrap_or_else(|_| "matrix".into())
        .to_ascii_lowercase();
    let continuation_lifecycle = phase == "continuation-lifecycle";
    let policy_full_solve = phase == "policy-full-solve";
    let frontend_callback = phase == "frontend-callback";
    if frontend_callback {
        let workloads = std::env::var("BVP_SCI_BENCH_FRONTEND_CALLBACK_WORKLOADS")
            .unwrap_or_else(|_| "stiff-coupled,combustion-like".into())
            .split(',')
            .filter_map(policy_workload)
            .collect::<Vec<_>>();
        let repeats = parse_counts("BVP_SCI_BENCH_FRONTEND_CALLBACK_REPEATS", "1000")
            .first()
            .copied()
            .unwrap_or(1000);
        let warm_rayon = std::env::var("BVP_SCI_BENCH_FRONTEND_CALLBACK_WARM_RAYON")
            .map(|value| value != "0")
            .unwrap_or(false);
        let body = format!(
            "# BVP_sci Lambdify frontend callback microbench\n\n- phase: frontend-callback\n- workloads: {:?}\n- repeats per row: {repeats}\n- rayon_warmup: {warm_rayon}\n- no mesh, Newton or linear backend work is included\n- callback buffers are caller-owned and reused\n- timing scopes are diagnostic and non-additive\n\n{}",
            workloads
                .iter()
                .map(|workload| workload.label())
                .collect::<Vec<_>>(),
            frontend_callback_report(&workloads, repeats, warm_rayon),
        );
        let report_name = std::env::var("BVP_SCI_BENCH_REPORT")
            .unwrap_or_else(|_| "lambdify_frontend_callback_microbench".into());
        let path = write_test_report("BVP_sci_Lambdify_Bench", &report_name, &body)
            .expect("write compact BVP_sci callback report");
        println!("report={}", path.display());
        return;
    }
    if policy_full_solve {
        let nodes = parse_usizes("BVP_SCI_BENCH_POLICY_FULL_SOLVE_NODES", "64,256");
        let workloads = selected_policy_workloads();
        let layouts = selected_policy_layouts();
        let repeats = parse_counts("BVP_SCI_BENCH_POLICY_FULL_SOLVE_REPEATS", "3")
            .first()
            .copied()
            .unwrap_or(3);
        let frontends = selected_policy_frontends();
        let workers = std::env::var("BVP_SCI_BENCH_POLICY_FULL_SOLVE_WORKER_COUNT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(rayon::current_num_threads);
        let body = format!(
            "# BVP_sci Lambdify full-solve execution-policy matrix\n\n- phase: policy-full-solve\n- workers_requested: {workers}\n- workers_observed: dispatch-dependent; `0` means no parallel callback was selected\n- workloads: {:?}\n- nodes: {:?}\n- frontends: {:?}\n- layouts: {:?}\n- repeats: {repeats}; rows report medians\n- break-even criterion: at least 5% and at least 0.01 ms faster than the same frontend/workload/layout sequential row\n- callback policy affects independent residual/Jacobian entries; numerical workspace remains single-owner\n- timing scopes are diagnostic and non-additive\n\n{}",
            workloads
                .iter()
                .map(|workload| workload.label())
                .collect::<Vec<_>>(),
            nodes,
            frontends
                .iter()
                .map(|frontend| frontend.label())
                .collect::<Vec<_>>(),
            layouts
                .iter()
                .map(|layout| layout.label())
                .collect::<Vec<_>>(),
            policy_full_solve_report(workers, &nodes, &workloads, &frontends, &layouts, repeats,),
        );
        let report_name = std::env::var("BVP_SCI_BENCH_REPORT")
            .unwrap_or_else(|_| format!("lambdify_policy_full_solve_workers_{workers}"));
        let path = write_test_report("BVP_sci_Lambdify_Bench", &report_name, &body)
            .expect("write compact BVP_sci policy report");
        println!("report={}", path.display());
        return;
    }
    if continuation_lifecycle {
        let nodes = parse_usizes("BVP_SCI_BENCH_CONTINUATION_NODES", "64,256");
        let counts = parse_counts("BVP_SCI_BENCH_CONTINUATION_COUNTS", "1,4,16");
        let layouts = selected_continuation_layouts();
        let modes = std::env::var("BVP_SCI_BENCH_CONTINUATION_MODES")
            .unwrap_or_else(|_| "fresh,prepared,warm".into())
            .split(',')
            .filter_map(ContinuationMode::parse)
            .collect::<Vec<_>>();
        let body = continuation_lifecycle_report(
            &nodes,
            &counts,
            &[Frontend::ExprLegacy, Frontend::AtomNative],
            &layouts,
            &modes,
        )
        .map(|table| {
            format!(
                "# BVP_sci Lambdify continuation lifecycle\n\n- modes: {:?}\n- nodes: {:?}\n- counts: {:?}\n- layouts: {:?}\n- initial profile: bounded non-exact perturbation of the manufactured solution\n- prepared excludes one-time symbolic preparation from per-solver rows\n- warm retention is logical workspace growth, not process-wide heap allocation\n- same-mesh retention and mesh-growth retention must be interpreted separately\n- timing scopes are diagnostic and non-additive\n- Sparse symbolic counters distinguish one-time pattern analysis from repeated numeric factorization\n\n{}",
                modes.iter().map(|mode| mode.label()).collect::<Vec<_>>(),
                nodes,
                counts,
                layouts.iter().map(|layout| layout.label()).collect::<Vec<_>>(),
                table
            )
        })
        .unwrap_or_else(|error| format!("# BVP_sci continuation lifecycle\n\nstatus: error\nerror: {error}\n"));
        let report_name = std::env::var("BVP_SCI_BENCH_REPORT")
            .unwrap_or_else(|_| "lambdify_continuation_lifecycle".into());
        let path = write_test_report("BVP_sci_Lambdify_Bench", &report_name, &body)
            .expect("write compact BVP_sci continuation report");
        println!("report={}", path.display());
        return;
    }
    let nodes = parse_usizes(
        if phase == "continuation" {
            "BVP_SCI_BENCH_CONTINUATION_NODES"
        } else {
            "BVP_SCI_BENCH_NODES"
        },
        if phase == "continuation" {
            "64,256,1024"
        } else {
            "8,32,128"
        },
    );
    let workloads = if phase == "continuation" {
        std::env::var("BVP_SCI_BENCH_CONTINUATION_WORKLOADS")
            .unwrap_or_else(|_| "parameterized-linear".into())
            .split(',')
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .collect()
    } else {
        selected_workloads()
    };
    let continuation = std::env::var("BVP_SCI_BENCH_CONTINUATION")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or(4);
    let counts = if phase == "continuation" {
        parse_counts("BVP_SCI_BENCH_CONTINUATION_COUNTS", "1,4,16")
    } else {
        vec![continuation]
    };
    let mut rows = Vec::new();
    for workload in [
        Workload::Linear,
        Workload::Parameterized,
        Workload::Oscillator,
        Workload::StiffDecay,
        Workload::Bratu,
        Workload::CombustionLike,
        Workload::StiffCoupled,
    ] {
        if !workloads.iter().any(|name| name == workload.label()) {
            continue;
        }
        for &dimension in &nodes {
            for frontend in [Frontend::ExprLegacy, Frontend::AtomNative] {
                for layout in [Layout::Dense, Layout::Sparse, Layout::Banded] {
                    for &count in &counts {
                        rows.push(run_row(workload, frontend, layout, dimension, count));
                    }
                }
            }
        }
    }
    let table = Table::new(&rows).to_string();
    validate_markdown_table(&table);
    let body = format!(
        "# BVP_sci Lambdify matrix\n\n- phase: {}\n- workloads: {:?}\n- nodes: {:?}\n- continuation: {}\n- policy: {:?}\n- AOT: not included\n- timing scopes: diagnostic and non-additive\n\n{}",
        phase,
        workloads,
        nodes,
        continuation,
        selected_policy(),
        table
    );
    let report_name = std::env::var("BVP_SCI_BENCH_REPORT").unwrap_or_else(|_| {
        if phase == "continuation" {
            "lambdify_continuation".into()
        } else {
            "lambdify_matrix".into()
        }
    });
    let path = write_test_report("BVP_sci_Lambdify_Bench", &report_name, &body)
        .expect("write compact BVP_sci report");
    println!("report={}", path.display());
}
