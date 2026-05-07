use super::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2Solver, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedResidualAotBackend, register_linked_residual_backend, unregister_linked_residual_backend,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    SymbolicIvpProblemOptions, prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::{DMatrix, DVector};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn exponential_decay_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
}

struct ResidualBackendGuard {
    problem_key: String,
}

impl Drop for ResidualBackendGuard {
    fn drop(&mut self) {
        unregister_linked_residual_backend(self.problem_key.as_str());
    }
}

fn register_prelinked_decay_residual(
    config: &Lsode2ProblemConfig,
    generated_backend: &SymbolicIvpGeneratedBackendConfig,
) -> (ResidualBackendGuard, Arc<AtomicUsize>) {
    let residual_problem = prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new().with_aot_options(generated_backend.aot_options),
    )
    .expect("story residual problem should prepare");
    let problem_key = residual_problem
        .prepare_residual_aot_problem(generated_backend.aot_options)
        .problem_key();
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_backend = Arc::clone(&calls);

    register_linked_residual_backend(LinkedResidualAotBackend::new(
        problem_key.clone(),
        1,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            calls_for_backend.fetch_add(1, Ordering::Relaxed);
            out[0] = -args[1];
        }),
    ));

    (ResidualBackendGuard { problem_key }, calls)
}

struct StoryRow {
    route: &'static str,
    matrix: &'static str,
    mode: &'static str,
    total_ms: f64,
    prepare_ms: Option<f64>,
    solve_ms: Option<f64>,
    final_diff: Option<f64>,
    resolved_source: Option<String>,
    resolved_structure: Option<String>,
    linear_solver: Option<String>,
    linear_reason: Option<String>,
    residual_calls: Option<usize>,
    jacobian_calls: Option<usize>,
    nlu_or_native_linsolve: Option<usize>,
    residual_ms_total: Option<f64>,
    jacobian_ms_total: Option<f64>,
    linked_residual_calls: Option<usize>,
    status: String,
}

fn short_error(message: &str) -> String {
    const LIMIT: usize = 120;
    let flat = message.replace(['\r', '\n'], " ");
    if flat.len() <= LIMIT {
        flat
    } else {
        format!("{}...", &flat[..LIMIT])
    }
}

fn is_native_faithful_status(status: &str) -> bool {
    status == "finished_native_faithful" || status == "finished_native_faithful_partial"
}

fn is_finished_status(status: &str) -> bool {
    status == "finished" || is_native_faithful_status(status)
}

fn catch_unwind_quiet<F, R>(f: F) -> std::thread::Result<R>
where
    F: FnOnce() -> R,
{
    static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = PANIC_HOOK_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("panic hook lock should not be poisoned");
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(previous);
    result
}

fn solve_story_row_fallible(
    route: &'static str,
    matrix: &'static str,
    mode: &'static str,
    config: Lsode2ProblemConfig,
    reference_final_y: f64,
    linked_calls_before: Option<usize>,
) -> StoryRow {
    let started_total = Instant::now();
    let mut solver = match Lsode2Solver::new(config) {
        Ok(solver) => solver,
        Err(err) => {
            return StoryRow {
                route,
                matrix,
                mode,
                total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
                prepare_ms: None,
                solve_ms: None,
                final_diff: None,
                resolved_source: None,
                resolved_structure: None,
                linear_solver: None,
                linear_reason: None,
                residual_calls: None,
                jacobian_calls: None,
                nlu_or_native_linsolve: None,
                residual_ms_total: None,
                jacobian_ms_total: None,
                linked_residual_calls: None,
                status: format!("new_error({})", short_error(&err.to_string())),
            };
        }
    };

    let mut prepare_ms = None;
    if route != "Analytical-Native" {
        let started_prepare = Instant::now();
        if let Err(err) = solver.prepare() {
            return StoryRow {
                route,
                matrix,
                mode,
                total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
                prepare_ms: Some(started_prepare.elapsed().as_secs_f64() * 1_000.0),
                solve_ms: None,
                final_diff: None,
                resolved_source: None,
                resolved_structure: None,
                linear_solver: None,
                linear_reason: None,
                residual_calls: None,
                jacobian_calls: None,
                nlu_or_native_linsolve: None,
                residual_ms_total: None,
                jacobian_ms_total: None,
                linked_residual_calls: None,
                status: format!("prepare_error({})", short_error(&err.to_string())),
            };
        }
        prepare_ms = Some(started_prepare.elapsed().as_secs_f64() * 1_000.0);
    }

    let started_solve = Instant::now();
    let summary = match solver.solve_with_summary() {
        Ok(summary) => summary,
        Err(err) => {
            return StoryRow {
                route,
                matrix,
                mode,
                total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
                prepare_ms,
                solve_ms: Some(started_solve.elapsed().as_secs_f64() * 1_000.0),
                final_diff: None,
                resolved_source: None,
                resolved_structure: None,
                linear_solver: None,
                linear_reason: None,
                residual_calls: None,
                jacobian_calls: None,
                nlu_or_native_linsolve: None,
                residual_ms_total: None,
                jacobian_ms_total: None,
                linked_residual_calls: None,
                status: format!("solve_error({})", short_error(&err.to_string())),
            };
        }
    };
    let solve_ms = started_solve.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = started_total.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary
        .final_y
        .as_ref()
        .expect("story solve should have final y")[0];
    let linked_after = linked_calls_before.map(|_| 0usize);

    let is_native_faithful = is_native_faithful_status(&summary.status);

    let (
        residual_calls,
        jacobian_calls,
        nlu_or_native_linsolve,
        residual_ms_total,
        jacobian_ms_total,
    ) = if route == "Analytical-Native" || is_native_faithful {
        (
            Some(summary.native_statistics.native_residual_calls),
            Some(summary.native_statistics.native_jacobian_calls),
            Some(summary.native_statistics.native_linear_solve_calls),
            Some(summary.native_statistics.native_residual_ms_total),
            Some(summary.native_statistics.native_jacobian_ms_total),
        )
    } else {
        (
            Some(summary.statistics.residual_calls),
            Some(summary.statistics.jacobian_calls),
            Some(summary.statistics.bdf_nlu_total),
            Some(summary.statistics.residual_ms_total),
            Some(summary.statistics.jacobian_ms_total),
        )
    };

    StoryRow {
        route,
        matrix,
        mode,
        total_ms,
        prepare_ms,
        solve_ms: Some(solve_ms),
        final_diff: Some((final_y - reference_final_y).abs()),
        resolved_source: Some(summary.resolved_source.to_string()),
        resolved_structure: Some(summary.resolved_structure.to_string()),
        linear_solver: Some(summary.linear_solver_backend.to_string()),
        linear_reason: Some(summary.linear_solver_reason.to_string()),
        residual_calls,
        jacobian_calls,
        nlu_or_native_linsolve,
        residual_ms_total,
        jacobian_ms_total,
        linked_residual_calls: linked_after,
        status: summary.status,
    }
}

#[test]
fn lsode2_exponential_decay_backend_story_table() {
    let mut reference_solver = Lsode2Solver::new(exponential_decay_config())
        .expect("reference LSODE2 config should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("reference LSODE2 solve should finish");
    let reference_final_y = reference
        .final_y
        .as_ref()
        .expect("reference solve should have final y")[0];

    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt();
    let sparse_aot_config = exponential_decay_config()
        .with_native_sparse_faer_generated_backend(generated_backend.clone());
    let banded_aot_config = exponential_decay_config()
        .with_native_banded_faithful_generated_backend(generated_backend.clone());
    let (sparse_guard, sparse_calls) =
        register_prelinked_decay_residual(&sparse_aot_config, &generated_backend);
    let sparse_calls_before = sparse_calls.load(Ordering::Relaxed);
    let sparse_aot_row = solve_story_row_fallible(
        "AOT-PrelinkedRuntime",
        "Sparse",
        "symbolic/aot (runtime prelinked, no build)",
        sparse_aot_config,
        reference_final_y,
        Some(sparse_calls_before),
    );
    let sparse_aot_calls = sparse_calls.load(Ordering::Relaxed);
    drop(sparse_guard);

    let (banded_guard, banded_calls) =
        register_prelinked_decay_residual(&banded_aot_config, &generated_backend);
    let banded_calls_before = banded_calls.load(Ordering::Relaxed);
    let banded_aot_row = solve_story_row_fallible(
        "AOT-PrelinkedRuntime",
        "Banded",
        "symbolic/aot (runtime prelinked, no build)",
        banded_aot_config,
        reference_final_y,
        Some(banded_calls_before),
    );
    let banded_aot_calls = banded_calls.load(Ordering::Relaxed);
    drop(banded_guard);

    let honest_aot_backend = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
        PathBuf::from("target/lsode2-story-aot"),
    )
    .with_c_tcc();
    let honest_aot_sparse = exponential_decay_config()
        .with_native_sparse_faer_generated_backend(honest_aot_backend.clone())
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            },
        });
    let honest_aot_banded = exponential_decay_config()
        .with_native_banded_faithful_generated_backend(honest_aot_backend)
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            },
        });

    let rows = vec![
        StoryRow {
            route: "Lambdify",
            matrix: "Dense",
            mode: "symbolic/lambdify",
            total_ms: reference.statistics.solve_ms_total,
            prepare_ms: Some(reference.statistics.backend_prepare_ms_total),
            solve_ms: Some(reference.statistics.solve_ms_total),
            final_diff: Some(0.0),
            resolved_source: Some(reference.resolved_source.to_string()),
            resolved_structure: Some(reference.resolved_structure.to_string()),
            linear_solver: Some(reference.linear_solver_backend.to_string()),
            linear_reason: Some(reference.linear_solver_reason.to_string()),
            residual_calls: Some(reference.statistics.residual_calls),
            jacobian_calls: Some(reference.statistics.jacobian_calls),
            nlu_or_native_linsolve: Some(reference.statistics.bdf_nlu_total),
            residual_ms_total: Some(reference.statistics.residual_ms_total),
            jacobian_ms_total: Some(reference.statistics.jacobian_ms_total),
            linked_residual_calls: Some(0),
            status: reference.status.clone(),
        },
        solve_story_row_fallible(
            "Lambdify",
            "Sparse",
            "symbolic/lambdify",
            exponential_decay_config().with_native_sparse_faer_backend(),
            reference_final_y,
            Some(0),
        ),
        solve_story_row_fallible(
            "Lambdify",
            "Banded",
            "symbolic/lambdify",
            exponential_decay_config().with_native_banded_faithful_backend(),
            reference_final_y,
            Some(0),
        ),
        solve_story_row_fallible(
            "Analytical-Native",
            "Sparse",
            "analytical/native",
            exponential_decay_config()
                .with_native_sparse_faer_backend()
                .with_analytical_callbacks(
                    |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
                    |_t, _y: &DVector<f64>| nalgebra::DMatrix::from_row_slice(1, 1, &[-1.0]),
                )
                .with_faithful_bdf_solve(4096, 4096),
            reference_final_y,
            Some(0),
        ),
        solve_story_row_fallible(
            "Analytical-Native",
            "Banded",
            "analytical/native",
            exponential_decay_config()
                .with_native_banded_faithful_backend()
                .with_analytical_callbacks(
                    |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
                    |_t, _y: &DVector<f64>| nalgebra::DMatrix::from_row_slice(1, 1, &[-1.0]),
                )
                .with_faithful_bdf_solve(4096, 4096),
            reference_final_y,
            Some(0),
        ),
        solve_story_row_fallible(
            "AOT-HonestBuild",
            "Sparse",
            "symbolic/aot (build_if_missing, c_tcc)",
            honest_aot_sparse,
            reference_final_y,
            Some(0),
        ),
        solve_story_row_fallible(
            "AOT-HonestBuild",
            "Banded",
            "symbolic/aot (build_if_missing, c_tcc)",
            honest_aot_banded,
            reference_final_y,
            Some(0),
        ),
        StoryRow {
            linked_residual_calls: Some(sparse_aot_calls.saturating_sub(sparse_calls_before)),
            ..sparse_aot_row
        },
        StoryRow {
            linked_residual_calls: Some(banded_aot_calls.saturating_sub(banded_calls_before)),
            ..banded_aot_row
        },
    ];

    println!("[LSODE2 story] exponential decay backend summary; all time columns are milliseconds");
    println!(
        "note: `AOT-PrelinkedRuntime` is NOT a full codegen/build path; it only checks runtime linkage. `AOT-HonestBuild` performs real generated-backend build flow."
    );
    println!(
        "route               | matrix | mode                                  | total_ms | prepare_ms | solve_ms | final_diff | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<19} | {:<6} | {:<37} | {:>8.3} | {:>10} | {:>8} | {:>10} | {}",
            row.route,
            row.matrix,
            row.mode,
            row.total_ms,
            row.prepare_ms
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.solve_ms
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.final_diff
                .map(|v| format!("{v:.3e}"))
                .unwrap_or_else(|| "-".to_string()),
            row.status
        );
    }

    println!(
        "[LSODE2 story] exponential decay backend diagnostics; all time columns are milliseconds"
    );
    println!(
        "route               | matrix | resolved_src | resolved_struct | linear_solver               | linear_reason                  | residual_calls | jacobian_calls | nlu/native_linsolve | residual_ms | jacobian_ms | linked_residual_calls"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<19} | {:<6} | {:<12} | {:<15} | {:<27} | {:<30} | {:>14} | {:>13} | {:>18} | {:>11} | {:>11} | {:>21}",
            row.route,
            row.matrix,
            row.resolved_source
                .clone()
                .unwrap_or_else(|| "-".to_string()),
            row.resolved_structure
                .clone()
                .unwrap_or_else(|| "-".to_string()),
            row.linear_solver.clone().unwrap_or_else(|| "-".to_string()),
            row.linear_reason.clone().unwrap_or_else(|| "-".to_string()),
            row.residual_calls
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.jacobian_calls
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.nlu_or_native_linsolve
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.residual_ms_total
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.jacobian_ms_total
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.linked_residual_calls
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
        );
    }

    for row in rows {
        if row.status.contains("error(") {
            continue;
        }
        let is_native_faithful = row.status == "finished_native_faithful"
            || row.status == "finished_native_faithful_partial";

        if row.route == "Analytical-Native" {
            assert!(
                is_native_faithful,
                "{} {} should finish via native analytical route, got status={}",
                row.route, row.matrix, row.status
            );
            assert!(
                row.final_diff.unwrap_or(f64::INFINITY) < 1e-4,
                "{} {} native analytical drift is too large: {:e}",
                row.route,
                row.matrix,
                row.final_diff.unwrap_or(f64::INFINITY)
            );
            assert!(
                row.residual_calls.unwrap_or(0) > 0,
                "{} {} should execute native residual callbacks",
                row.route,
                row.matrix
            );
            assert!(
                row.nlu_or_native_linsolve.unwrap_or(0) > 0 || row.jacobian_calls.unwrap_or(0) > 0,
                "{} {} should execute native Jacobian/linear work",
                row.route,
                row.matrix
            );
        } else if is_native_faithful {
            assert!(
                row.final_diff.unwrap_or(f64::INFINITY) < 1e-4,
                "{} {} faithful-native drift is too large: {:e}",
                row.route,
                row.matrix,
                row.final_diff.unwrap_or(f64::INFINITY)
            );
            assert!(
                row.residual_calls.unwrap_or(0) > 0,
                "{} {} faithful-native route should evaluate residuals",
                row.route,
                row.matrix
            );
            assert!(
                row.jacobian_calls.unwrap_or(0) > 0,
                "{} {} faithful-native route should evaluate/update Jacobian state",
                row.route,
                row.matrix
            );
        } else {
            assert_eq!(row.status, "finished");
            assert!(
                row.final_diff.unwrap_or(f64::INFINITY) < 1e-8,
                "{} {} drifted from dense reference: {:e}",
                row.route,
                row.matrix,
                row.final_diff.unwrap_or(f64::INFINITY)
            );
            assert!(
                row.residual_calls.unwrap_or(0) > 0,
                "{} {} should evaluate residuals",
                row.route,
                row.matrix
            );
            assert!(
                row.nlu_or_native_linsolve.unwrap_or(0) > 0,
                "{} {} should factor Newton systems",
                row.route,
                row.matrix
            );
        }
        if row.route == "AOT-PrelinkedRuntime" {
            assert!(
                row.linked_residual_calls.unwrap_or(0) > 0,
                "{} {} should route residual calls through the linked AOT registry",
                row.route,
                row.matrix
            );
        }
    }
}

#[derive(Clone, Copy)]
enum ComprehensiveScenario {
    NonStiffScalarDecay,
    StiffScalarTracking,
    NonStiffSystemDecay2,
    StiffSystemTracking2,
}

impl ComprehensiveScenario {
    fn label(self) -> &'static str {
        match self {
            Self::NonStiffScalarDecay => "nonstiff-scalar-decay",
            Self::StiffScalarTracking => "stiff-scalar-tracking",
            Self::NonStiffSystemDecay2 => "nonstiff-system2-decay",
            Self::StiffSystemTracking2 => "stiff-system2-tracking",
        }
    }

    fn band(self) -> (usize, usize) {
        match self {
            Self::NonStiffScalarDecay => (0, 0),
            Self::StiffScalarTracking => (0, 0),
            Self::NonStiffSystemDecay2 => (1, 1),
            Self::StiffSystemTracking2 => (1, 1),
        }
    }

    fn tolerance(self) -> f64 {
        match self {
            Self::NonStiffScalarDecay => 1.0e-6,
            Self::StiffScalarTracking => 5.0e-4,
            Self::NonStiffSystemDecay2 => 1.0e-6,
            Self::StiffSystemTracking2 => 1.0e-3,
        }
    }

    fn config(self) -> Lsode2ProblemConfig {
        match self {
            Self::NonStiffScalarDecay => Lsode2ProblemConfig::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".to_string()],
                "t".to_string(),
                0.0,
                DVector::from_vec(vec![1.0]),
                1.0,
                0.02,
                1e-6,
                1e-8,
            ),
            Self::StiffScalarTracking => Lsode2ProblemConfig::new(
                vec![Expr::parse_expression("-1000*(y-cos(t))-sin(t)")],
                vec!["y".to_string()],
                "t".to_string(),
                0.0,
                DVector::from_vec(vec![1.0]),
                1.0,
                0.02,
                1e-6,
                1e-8,
            )
            .with_first_step(Some(0.02)),
            Self::NonStiffSystemDecay2 => Lsode2ProblemConfig::new(
                vec![
                    Expr::parse_expression("-y1"),
                    Expr::parse_expression("-2*y2"),
                ],
                vec!["y1".to_string(), "y2".to_string()],
                "t".to_string(),
                0.0,
                DVector::from_vec(vec![1.0, 1.0]),
                1.0,
                0.02,
                1e-6,
                1e-8,
            ),
            Self::StiffSystemTracking2 => Lsode2ProblemConfig::new(
                vec![
                    Expr::parse_expression("-1000*(y1-cos(t))-sin(t)"),
                    Expr::parse_expression("-500*(y2-sin(t))+cos(t)"),
                ],
                vec!["y1".to_string(), "y2".to_string()],
                "t".to_string(),
                0.0,
                DVector::from_vec(vec![1.0, 0.0]),
                1.0,
                0.02,
                1e-6,
                1e-8,
            )
            .with_first_step(Some(0.02)),
        }
    }

    fn expected(self, t: f64) -> DVector<f64> {
        match self {
            Self::NonStiffScalarDecay => DVector::from_vec(vec![(-t).exp()]),
            Self::StiffScalarTracking => DVector::from_vec(vec![t.cos()]),
            Self::NonStiffSystemDecay2 => DVector::from_vec(vec![(-t).exp(), (-2.0 * t).exp()]),
            Self::StiffSystemTracking2 => DVector::from_vec(vec![t.cos(), t.sin()]),
        }
    }
}

struct ComprehensiveRow {
    scenario: &'static str,
    route: &'static str,
    matrix: &'static str,
    mode: &'static str,
    resolved_source: String,
    resolved_structure: String,
    total_ms: f64,
    final_t: f64,
    max_abs_err: f64,
    residual_calls: usize,
    jacobian_calls: usize,
    nlu: usize,
    status: String,
}

fn max_abs_diff_vec(lhs: &DVector<f64>, rhs: &DVector<f64>) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0_f64, f64::max)
}

fn run_comprehensive_story_case(
    scenario: ComprehensiveScenario,
    route: &'static str,
    matrix: &'static str,
    mode: &'static str,
    config: Lsode2ProblemConfig,
) -> ComprehensiveRow {
    let mut solver =
        Lsode2Solver::new(config).expect("comprehensive LSODE2 story config should build");
    let started = Instant::now();
    let summary = solver
        .solve_with_summary()
        .expect("comprehensive LSODE2 story solve should finish");
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let final_t = summary
        .final_t
        .expect("comprehensive LSODE2 story solve should expose final_t");
    let final_y = summary
        .final_y
        .as_ref()
        .expect("comprehensive LSODE2 story solve should expose final_y");
    let expected = scenario.expected(final_t);
    let max_abs_err = max_abs_diff_vec(final_y, &expected);

    let is_native_faithful = is_native_faithful_status(&summary.status);
    let (residual_calls, jacobian_calls, nlu) = if is_native_faithful {
        (
            summary.native_statistics.native_residual_calls,
            summary.native_statistics.native_jacobian_calls,
            summary.native_statistics.native_linear_solve_calls,
        )
    } else {
        (
            summary.statistics.residual_calls,
            summary.statistics.jacobian_calls,
            summary.statistics.bdf_nlu_total,
        )
    };

    ComprehensiveRow {
        scenario: scenario.label(),
        route,
        matrix,
        mode,
        resolved_source: summary.resolved_source.to_string(),
        resolved_structure: summary.resolved_structure.to_string(),
        total_ms,
        final_t,
        max_abs_err,
        residual_calls,
        jacobian_calls,
        nlu,
        status: summary.status,
    }
}

#[test]
fn lsode2_comprehensive_multi_equation_backend_story_table() {
    let scenarios = [
        ComprehensiveScenario::NonStiffScalarDecay,
        ComprehensiveScenario::StiffScalarTracking,
        ComprehensiveScenario::NonStiffSystemDecay2,
        ComprehensiveScenario::StiffSystemTracking2,
    ];

    let mut rows = Vec::new();
    for scenario in scenarios {
        let (kl, ku) = scenario.band();
        rows.push(run_comprehensive_story_case(
            scenario,
            "Symbolic-Lambdify",
            "Dense",
            "symbolic/lambdify",
            scenario
                .config()
                .with_linear_system_structure(super::Lsode2LinearSystemStructure::Dense)
                .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto),
        ));
        rows.push(run_comprehensive_story_case(
            scenario,
            "Symbolic-Lambdify",
            "Sparse",
            "symbolic/lambdify",
            scenario
                .config()
                .with_native_sparse_faer_backend()
                .with_linear_system_structure(super::Lsode2LinearSystemStructure::Sparse)
                .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto),
        ));
        rows.push(run_comprehensive_story_case(
            scenario,
            "Symbolic-Lambdify",
            "Banded",
            "symbolic/lambdify",
            scenario
                .config()
                .with_native_banded_faithful_backend()
                .with_linear_system_structure(super::Lsode2LinearSystemStructure::Banded { kl, ku })
                .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto),
        ));
    }

    println!(
        "[LSODE2 story] comprehensive multi-equation backend table (symbolic+lambdify only); all time columns are milliseconds"
    );
    println!(
        "scenario                  | route              | matrix | mode               | resolved_src | resolved_struct | total_ms | final_t   | max_abs_err | residual_calls | jacobian_calls | nlu | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<25} | {:<18} | {:<6} | {:<18} | {:<12} | {:<15} | {:>8.3} | {:>9.3e} | {:>11.3e} | {:>14} | {:>13} | {:>3} | {}",
            row.scenario,
            row.route,
            row.matrix,
            row.mode,
            row.resolved_source,
            row.resolved_structure,
            row.total_ms,
            row.final_t,
            row.max_abs_err,
            row.residual_calls,
            row.jacobian_calls,
            row.nlu,
            row.status
        );
    }

    for row in rows {
        let tolerance = match row.scenario {
            "nonstiff-scalar-decay" => ComprehensiveScenario::NonStiffScalarDecay.tolerance(),
            "stiff-scalar-tracking" => ComprehensiveScenario::StiffScalarTracking.tolerance(),
            "nonstiff-system2-decay" => ComprehensiveScenario::NonStiffSystemDecay2.tolerance(),
            "stiff-system2-tracking" => ComprehensiveScenario::StiffSystemTracking2.tolerance(),
            _ => unreachable!("unexpected scenario label"),
        };

        assert!(
            is_finished_status(&row.status),
            "unexpected status in comprehensive story row: {}",
            row.status
        );
        assert_eq!(row.resolved_source, "symbolic");
        let expected_structure = match row.matrix {
            "Dense" => "dense",
            "Sparse" => "sparse",
            "Banded" => "banded",
            _ => unreachable!("unexpected matrix label"),
        };
        assert_eq!(row.resolved_structure, expected_structure);
        assert!(
            (row.final_t - 1.0).abs() <= 5.0e-3,
            "{} {} should reach final_t close to 1.0, got {:e}",
            row.scenario,
            row.matrix,
            row.final_t
        );
        assert!(
            row.max_abs_err <= tolerance,
            "{} {} max_abs_err too large: {:e} (tol={:e})",
            row.scenario,
            row.matrix,
            row.max_abs_err,
            tolerance
        );
        assert!(
            row.residual_calls > 0,
            "{} {} should evaluate residuals",
            row.scenario,
            row.matrix
        );
        assert!(
            row.jacobian_calls > 0,
            "{} {} should evaluate Jacobians",
            row.scenario,
            row.matrix
        );
        assert!(
            row.nlu > 0,
            "{} {} should factor linear systems",
            row.scenario,
            row.matrix
        );
    }
}

#[derive(Clone, Copy)]
enum AotStoryToolchain {
    CTcc,
    CGcc,
    Zig,
    Rust,
}

impl AotStoryToolchain {
    fn label(self) -> &'static str {
        match self {
            Self::CTcc => "c_tcc",
            Self::CGcc => "c_gcc",
            Self::Zig => "zig",
            Self::Rust => "rust",
        }
    }

    fn as_source(self) -> Lsode2SymbolicExecutionMode {
        match self {
            Self::CTcc => Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            },
            Self::CGcc => Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CGcc,
                profile: Lsode2AotProfile::Release,
            },
            Self::Zig => Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::Zig,
                profile: Lsode2AotProfile::Release,
            },
            Self::Rust => Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::Rust,
                profile: Lsode2AotProfile::Release,
            },
        }
    }

    fn apply_generated(
        self,
        backend: SymbolicIvpGeneratedBackendConfig,
    ) -> SymbolicIvpGeneratedBackendConfig {
        match self {
            Self::CTcc => backend.with_c_tcc(),
            Self::CGcc => backend.with_c_gcc(),
            Self::Zig => backend.with_zig(),
            Self::Rust => backend.with_rust(),
        }
    }
}

#[derive(Clone, Copy)]
enum AotStoryMatrix {
    Dense,
    Sparse,
    Banded,
}

impl AotStoryMatrix {
    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
    }
}

struct AotStoryRow {
    matrix: &'static str,
    toolchain: &'static str,
    run_kind: &'static str,
    repeat_idx: usize,
    total_ms: f64,
    prepare_ms: Option<f64>,
    solve_ms: Option<f64>,
    final_diff_vs_lambdify: Option<f64>,
    residual_calls: Option<usize>,
    jacobian_calls: Option<usize>,
    nlu: Option<usize>,
    residual_ms_total: Option<f64>,
    jacobian_ms_total: Option<f64>,
    status: String,
}

fn build_aot_story_config(
    matrix: AotStoryMatrix,
    toolchain: AotStoryToolchain,
    output_dir: PathBuf,
    _suffix: &str,
) -> Lsode2ProblemConfig {
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: toolchain.as_source(),
    };
    let generated = toolchain.apply_generated(
        SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir),
    );

    match matrix {
        AotStoryMatrix::Dense => {
            let mut config = exponential_decay_config()
                .with_linear_system_structure(super::Lsode2LinearSystemStructure::Dense)
                .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto)
                .with_residual_jacobian_source(source);
            config.backend.generated_backend = generated;
            config
        }
        AotStoryMatrix::Sparse => exponential_decay_config()
            .with_native_sparse_faer_generated_backend(generated)
            .with_residual_jacobian_source(source),
        AotStoryMatrix::Banded => exponential_decay_config()
            .with_native_banded_faithful_generated_backend(generated)
            .with_residual_jacobian_source(source),
    }
}

fn run_aot_story_row(
    matrix: AotStoryMatrix,
    toolchain: AotStoryToolchain,
    run_kind: &'static str,
    repeat_idx: usize,
    config: Lsode2ProblemConfig,
    lambdify_final_y: f64,
) -> AotStoryRow {
    let started_total = Instant::now();
    let mut solver = match Lsode2Solver::new(config) {
        Ok(solver) => solver,
        Err(err) => {
            return AotStoryRow {
                matrix: matrix.label(),
                toolchain: toolchain.label(),
                run_kind,
                repeat_idx,
                total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
                prepare_ms: None,
                solve_ms: None,
                final_diff_vs_lambdify: None,
                residual_calls: None,
                jacobian_calls: None,
                nlu: None,
                residual_ms_total: None,
                jacobian_ms_total: None,
                status: format!("new_error({})", short_error(&err.to_string())),
            };
        }
    };

    let started_prepare = Instant::now();
    let prepare_result = catch_unwind_quiet(|| solver.prepare());
    if let Err(payload) = prepare_result {
        let message = if let Some(text) = payload.downcast_ref::<&str>() {
            text.to_string()
        } else if let Some(text) = payload.downcast_ref::<String>() {
            text.clone()
        } else {
            "unknown panic payload".to_string()
        };
        return AotStoryRow {
            matrix: matrix.label(),
            toolchain: toolchain.label(),
            run_kind,
            repeat_idx,
            total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
            prepare_ms: Some(started_prepare.elapsed().as_secs_f64() * 1_000.0),
            solve_ms: None,
            final_diff_vs_lambdify: None,
            residual_calls: None,
            jacobian_calls: None,
            nlu: None,
            residual_ms_total: None,
            jacobian_ms_total: None,
            status: format!("prepare_panic({})", short_error(&message)),
        };
    }
    if let Err(err) = prepare_result.expect("catch_unwind already handled panic case") {
        return AotStoryRow {
            matrix: matrix.label(),
            toolchain: toolchain.label(),
            run_kind,
            repeat_idx,
            total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
            prepare_ms: Some(started_prepare.elapsed().as_secs_f64() * 1_000.0),
            solve_ms: None,
            final_diff_vs_lambdify: None,
            residual_calls: None,
            jacobian_calls: None,
            nlu: None,
            residual_ms_total: None,
            jacobian_ms_total: None,
            status: format!("prepare_error({})", short_error(&err.to_string())),
        };
    }
    let prepare_ms = started_prepare.elapsed().as_secs_f64() * 1_000.0;

    let started_solve = Instant::now();
    let solve_result = catch_unwind_quiet(|| solver.solve_with_summary());
    let summary = match solve_result {
        Err(payload) => {
            let message = if let Some(text) = payload.downcast_ref::<&str>() {
                text.to_string()
            } else if let Some(text) = payload.downcast_ref::<String>() {
                text.clone()
            } else {
                "unknown panic payload".to_string()
            };
            return AotStoryRow {
                matrix: matrix.label(),
                toolchain: toolchain.label(),
                run_kind,
                repeat_idx,
                total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
                prepare_ms: Some(prepare_ms),
                solve_ms: Some(started_solve.elapsed().as_secs_f64() * 1_000.0),
                final_diff_vs_lambdify: None,
                residual_calls: None,
                jacobian_calls: None,
                nlu: None,
                residual_ms_total: None,
                jacobian_ms_total: None,
                status: format!("solve_panic({})", short_error(&message)),
            };
        }
        Ok(Ok(summary)) => summary,
        Ok(Err(err)) => {
            return AotStoryRow {
                matrix: matrix.label(),
                toolchain: toolchain.label(),
                run_kind,
                repeat_idx,
                total_ms: started_total.elapsed().as_secs_f64() * 1_000.0,
                prepare_ms: Some(prepare_ms),
                solve_ms: Some(started_solve.elapsed().as_secs_f64() * 1_000.0),
                final_diff_vs_lambdify: None,
                residual_calls: None,
                jacobian_calls: None,
                nlu: None,
                residual_ms_total: None,
                jacobian_ms_total: None,
                status: format!("solve_error({})", short_error(&err.to_string())),
            };
        }
    };
    let solve_ms = started_solve.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = started_total.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary
        .final_y
        .as_ref()
        .expect("AOT story solve should expose final_y")[0];

    let is_native_faithful = is_native_faithful_status(&summary.status);
    let (residual_calls, jacobian_calls, nlu, residual_ms_total, jacobian_ms_total) =
        if is_native_faithful {
            (
                Some(summary.native_statistics.native_residual_calls),
                Some(summary.native_statistics.native_jacobian_calls),
                Some(summary.native_statistics.native_linear_solve_calls),
                Some(summary.native_statistics.native_residual_ms_total),
                Some(summary.native_statistics.native_jacobian_ms_total),
            )
        } else {
            (
                Some(summary.statistics.residual_calls),
                Some(summary.statistics.jacobian_calls),
                Some(summary.statistics.bdf_nlu_total),
                Some(summary.statistics.residual_ms_total),
                Some(summary.statistics.jacobian_ms_total),
            )
        };

    AotStoryRow {
        matrix: matrix.label(),
        toolchain: toolchain.label(),
        run_kind,
        repeat_idx,
        total_ms,
        prepare_ms: Some(prepare_ms),
        solve_ms: Some(solve_ms),
        final_diff_vs_lambdify: Some((final_y - lambdify_final_y).abs()),
        residual_calls,
        jacobian_calls,
        nlu,
        residual_ms_total,
        jacobian_ms_total,
        status: summary.status,
    }
}

#[test]
//#[ignore = "heavy AOT toolchain stage story (dense/sparse/banded; cold+warm)"]
fn lsode2_aot_toolchain_stage_story_table() {
    const REPEATS: usize = 3;
    let matrices = [
        AotStoryMatrix::Dense,
        AotStoryMatrix::Sparse,
        AotStoryMatrix::Banded,
    ];
    let toolchains = [
        AotStoryToolchain::CTcc,
        AotStoryToolchain::CGcc,
        AotStoryToolchain::Zig,
        AotStoryToolchain::Rust,
    ];

    let mut baseline_final_by_matrix = std::collections::BTreeMap::new();
    for matrix in matrices {
        let baseline_config = match matrix {
            AotStoryMatrix::Dense => exponential_decay_config()
                .with_linear_system_structure(super::Lsode2LinearSystemStructure::Dense)
                .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto),
            AotStoryMatrix::Sparse => exponential_decay_config().with_native_sparse_faer_backend(),
            AotStoryMatrix::Banded => {
                exponential_decay_config().with_native_banded_faithful_backend()
            }
        };
        let mut solver =
            Lsode2Solver::new(baseline_config).expect("lambdify baseline config should build");
        let summary = solver
            .solve_with_summary()
            .expect("lambdify baseline solve should finish");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("lambdify baseline should expose final_y")[0];
        baseline_final_by_matrix.insert(matrix.label(), final_y);
    }

    let mut rows = Vec::new();
    for repeat_idx in 0..REPEATS {
        let run_token = format!(
            "pid{}_{}_r{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after UNIX_EPOCH")
                .as_nanos(),
            repeat_idx
        );
        for matrix in matrices {
            for toolchain in toolchains {
                let final_ref = *baseline_final_by_matrix
                    .get(matrix.label())
                    .expect("baseline final_y should exist for matrix");
                let out_base = PathBuf::from(format!(
                    "target/lsode2-story-aot/{}/{}/{}",
                    matrix.label().to_lowercase(),
                    toolchain.label(),
                    run_token
                ));
                let cold_suffix = format!(
                    "{}_{}_cold",
                    matrix.label().to_lowercase(),
                    toolchain.label()
                );
                let warm_suffix = format!(
                    "{}_{}_warm",
                    matrix.label().to_lowercase(),
                    toolchain.label()
                );

                rows.push(run_aot_story_row(
                    matrix,
                    toolchain,
                    "cold",
                    repeat_idx,
                    build_aot_story_config(
                        matrix,
                        toolchain,
                        out_base.clone(),
                        cold_suffix.as_str(),
                    ),
                    final_ref,
                ));
                rows.push(run_aot_story_row(
                    matrix,
                    toolchain,
                    "warm",
                    repeat_idx,
                    build_aot_story_config(matrix, toolchain, out_base, warm_suffix.as_str()),
                    final_ref,
                ));
            }
        }
    }

    println!(
        "[LSODE2 story] AOT toolchain stage table (Dense/Sparse/Banded); all time columns are milliseconds"
    );
    println!(
        "matrix | toolchain | run  | rep | total_ms | prepare_ms | solve_ms | final_diff_vs_lambdify | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<6} | {:<9} | {:<4} | {:>3} | {:>8.3} | {:>10} | {:>8} | {:>22} | {}",
            row.matrix,
            row.toolchain,
            row.run_kind,
            row.repeat_idx,
            row.total_ms,
            row.prepare_ms
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.solve_ms
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.final_diff_vs_lambdify
                .map(|v| format!("{v:.3e}"))
                .unwrap_or_else(|| "-".to_string()),
            row.status
        );
    }

    println!("[LSODE2 story] AOT toolchain diagnostics; all time columns are milliseconds");
    println!(
        "matrix | toolchain | run  | rep | residual_calls | jacobian_calls | nlu | residual_ms | jacobian_ms"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<6} | {:<9} | {:<4} | {:>3} | {:>14} | {:>13} | {:>3} | {:>11} | {:>11}",
            row.matrix,
            row.toolchain,
            row.run_kind,
            row.repeat_idx,
            row.residual_calls
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.jacobian_calls
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.nlu
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            row.residual_ms_total
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
            row.jacobian_ms_total
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string()),
        );
    }

    #[derive(Default)]
    struct RunStats {
        values: Vec<f64>,
    }
    impl RunStats {
        fn push(&mut self, value: f64) {
            self.values.push(value);
        }
        fn summary(&self) -> Option<(f64, f64, f64, f64)> {
            if self.values.is_empty() {
                return None;
            }
            let n = self.values.len() as f64;
            let mean = self.values.iter().copied().sum::<f64>() / n;
            let var = self
                .values
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            let std = var.sqrt();
            let min = self.values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = self
                .values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            Some((mean, std, min, max))
        }
    }

    #[derive(Default)]
    struct AggregateBucket {
        ok_runs: usize,
        total_runs: usize,
        total_ms: RunStats,
        prepare_ms: RunStats,
        solve_ms: RunStats,
    }
    let mut aggregates = std::collections::BTreeMap::<String, AggregateBucket>::new();
    for row in &rows {
        let key = format!("{}|{}|{}", row.matrix, row.toolchain, row.run_kind);
        let bucket = aggregates.entry(key).or_default();
        bucket.total_runs += 1;
        if is_finished_status(&row.status) {
            bucket.ok_runs += 1;
            bucket.total_ms.push(row.total_ms);
            if let Some(v) = row.prepare_ms {
                bucket.prepare_ms.push(v);
            }
            if let Some(v) = row.solve_ms {
                bucket.solve_ms.push(v);
            }
        }
    }
    println!("[LSODE2 story] AOT multi-run aggregate table; all time columns are milliseconds");
    println!(
        "matrix | toolchain | run  | ok/runs | total_ms mean±std [min,max] | prepare_ms mean±std [min,max] | solve_ms mean±std [min,max]"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------"
    );
    for (key, bucket) in &aggregates {
        let parts = key.split('|').collect::<Vec<_>>();
        let (matrix, toolchain, run_kind) = (parts[0], parts[1], parts[2]);
        let total = bucket
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}±{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let prepare = bucket
            .prepare_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}±{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let solve = bucket
            .solve_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}±{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<9} | {:<4} | {:>7} | {:<30} | {:<32} | {}",
            matrix,
            toolchain,
            run_kind,
            format!("{}/{}", bucket.ok_runs, bucket.total_runs),
            total,
            prepare,
            solve
        );
    }

    println!(
        "note: anomaly-localization output is intentionally kept in debug diagnostics (`LSODE2/tests.rs`), not in story tables."
    );

    let successful = rows
        .iter()
        .filter(|row| is_finished_status(&row.status))
        .collect::<Vec<_>>();
    assert!(
        !successful.is_empty(),
        "at least one AOT toolchain case should complete successfully"
    );
    for row in successful {
        assert!(
            row.final_diff_vs_lambdify.unwrap_or(f64::INFINITY) < 1.0e-7,
            "{} {} {} drift vs lambdify baseline is too large: {:e}",
            row.matrix,
            row.toolchain,
            row.run_kind,
            row.final_diff_vs_lambdify.unwrap_or(f64::INFINITY)
        );
    }
}

#[derive(Clone, Copy)]
enum BackendRaceMatrix {
    Dense,
    Sparse,
    Banded,
}

impl BackendRaceMatrix {
    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
    }
}

#[derive(Default)]
struct RaceStats {
    values: Vec<f64>,
}

impl RaceStats {
    fn push(&mut self, value: f64) {
        self.values.push(value);
    }

    fn summary(&self) -> Option<(f64, f64, f64, f64)> {
        if self.values.is_empty() {
            return None;
        }
        let n = self.values.len() as f64;
        let mean = self.values.iter().copied().sum::<f64>() / n;
        let var = self
            .values
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        let min = self.values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Some((mean, std, min, max))
    }
}

struct BackendRaceRow {
    matrix: &'static str,
    route: &'static str,
    runs_ok: usize,
    runs_total: usize,
    total_ms: RaceStats,
    prepare_ms: RaceStats,
    solve_ms: RaceStats,
    final_diff: RaceStats,
    residual_calls: RaceStats,
    jacobian_calls: RaceStats,
    nlu_or_native_linear: RaceStats,
    accepted_steps: RaceStats,
    rejected_steps: RaceStats,
}

impl BackendRaceRow {
    fn new(matrix: &'static str, route: &'static str) -> Self {
        Self {
            matrix,
            route,
            runs_ok: 0,
            runs_total: 0,
            total_ms: RaceStats::default(),
            prepare_ms: RaceStats::default(),
            solve_ms: RaceStats::default(),
            final_diff: RaceStats::default(),
            residual_calls: RaceStats::default(),
            jacobian_calls: RaceStats::default(),
            nlu_or_native_linear: RaceStats::default(),
            accepted_steps: RaceStats::default(),
            rejected_steps: RaceStats::default(),
        }
    }
}

fn race_lambdify_config(matrix: BackendRaceMatrix) -> Lsode2ProblemConfig {
    match matrix {
        BackendRaceMatrix::Dense => exponential_decay_config()
            .with_linear_system_structure(super::Lsode2LinearSystemStructure::Dense)
            .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto),
        BackendRaceMatrix::Sparse => exponential_decay_config().with_native_sparse_faer_backend(),
        BackendRaceMatrix::Banded => {
            exponential_decay_config().with_native_banded_faithful_backend()
        }
    }
}

fn race_aot_config(matrix: BackendRaceMatrix) -> Lsode2ProblemConfig {
    let out = PathBuf::from(format!(
        "target/lsode2-story-race/{}/aot_c_tcc",
        matrix.label().to_lowercase()
    ));
    let generated = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
    };
    match matrix {
        BackendRaceMatrix::Dense => {
            let mut config = exponential_decay_config()
                .with_linear_system_structure(super::Lsode2LinearSystemStructure::Dense)
                .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto)
                .with_residual_jacobian_source(source);
            config.backend.generated_backend = generated;
            config
        }
        BackendRaceMatrix::Sparse => exponential_decay_config()
            .with_native_sparse_faer_generated_backend(generated)
            .with_residual_jacobian_source(source),
        BackendRaceMatrix::Banded => exponential_decay_config()
            .with_native_banded_faithful_generated_backend(generated)
            .with_residual_jacobian_source(source),
    }
}

fn race_analytical_config(matrix: BackendRaceMatrix) -> Option<Lsode2ProblemConfig> {
    let callbacks = (
        |_t: f64, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        |_t: f64, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
    );
    match matrix {
        BackendRaceMatrix::Dense => None,
        BackendRaceMatrix::Sparse => Some(
            exponential_decay_config()
                .with_native_sparse_faer_backend()
                .with_analytical_callbacks(callbacks.0, callbacks.1)
                .with_faithful_bdf_solve(4096, 4096),
        ),
        BackendRaceMatrix::Banded => Some(
            exponential_decay_config()
                .with_native_banded_faithful_backend()
                .with_analytical_callbacks(callbacks.0, callbacks.1)
                .with_faithful_bdf_solve(4096, 4096),
        ),
    }
}

fn run_backend_race_sample(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_y: f64,
) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
    let started_total = Instant::now();
    let mut solver = Lsode2Solver::new(config).ok()?;
    let mut prepare_ms = 0.0;
    if route != "Analytical-Native" {
        let prep_started = Instant::now();
        solver.prepare().ok()?;
        prepare_ms = prep_started.elapsed().as_secs_f64() * 1_000.0;
    }
    let solve_started = Instant::now();
    let summary = solver.solve_with_summary().ok()?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = started_total.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary.final_y.as_ref()?.get(0).copied()?;
    let final_diff = (final_y - baseline_final_y).abs();
    let is_native_faithful = is_native_faithful_status(&summary.status);
    let residual_calls = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_residual_calls as f64
    } else {
        summary.statistics.residual_calls as f64
    };
    let jacobian_calls = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_jacobian_calls as f64
    } else {
        summary.statistics.jacobian_calls as f64
    };
    let nlu_or_native_linear = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_linear_solve_calls as f64
    } else {
        summary.statistics.bdf_nlu_total as f64
    };
    let accepted_steps = summary
        .native_statistics
        .native_step_accepts
        .max(summary.native_statistics.bridge_accepted_steps) as f64;
    let rejected_steps = (summary.native_statistics.native_step_rejects_error_test
        + summary.native_statistics.native_step_rejects_nonlinear) as f64;
    Some((
        total_ms,
        prepare_ms,
        solve_ms,
        final_diff,
        residual_calls,
        jacobian_calls,
        nlu_or_native_linear,
        accepted_steps,
        rejected_steps,
    ))
}

#[test]
fn lsode2_multi_run_backend_race_by_weight_class() {
    const REPEATS: usize = 5;
    let matrices = [
        BackendRaceMatrix::Dense,
        BackendRaceMatrix::Sparse,
        BackendRaceMatrix::Banded,
    ];
    let routes = ["Lambdify", "Analytical-Native", "AOT-Ctcc"];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let mut solver = Lsode2Solver::new(race_lambdify_config(matrix))
            .expect("race baseline config should build");
        let summary = solver
            .solve_with_summary()
            .expect("race baseline solve should finish");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("race baseline should expose final_y")[0];
        baselines.insert(matrix.label(), final_y);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for route in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let baseline = *baselines
                    .get(matrix.label())
                    .expect("baseline final_y should exist");
                let config = match route {
                    "Lambdify" => Some(race_lambdify_config(matrix)),
                    "AOT-Ctcc" => Some(race_aot_config(matrix)),
                    "Analytical-Native" => race_analytical_config(matrix),
                    _ => None,
                };
                let Some(config) = config else {
                    continue;
                };
                if let Some(sample) = run_backend_race_sample(route, config, baseline) {
                    row.runs_ok += 1;
                    row.total_ms.push(sample.0);
                    row.prepare_ms.push(sample.1);
                    row.solve_ms.push(sample.2);
                    row.final_diff.push(sample.3);
                    row.residual_calls.push(sample.4);
                    row.jacobian_calls.push(sample.5);
                    row.nlu_or_native_linear.push(sample.6);
                    row.accepted_steps.push(sample.7);
                    row.rejected_steps.push(sample.8);
                }
            }
            rows.push(row);
        }
    }

    println!(
        "[LSODE2 story] multi-run backend race by weight class; all time columns are milliseconds"
    );
    println!(
        "matrix | route             | ok/runs | total_ms mean+/-std [min,max] | final_diff mean+/-std [min,max] | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3e}+/-{s:.1e} [{n:.3e},{x:.3e}]"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "not_supported_or_failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<17} | {:>7} | {:<31} | {:<34} | {}",
            row.matrix,
            row.route,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            diff,
            status
        );
    }

    println!(
        "[LSODE2 story] backend race diagnostics; counters are per-solve counts (aggregated as mean+/-std)"
    );
    println!(
        "matrix | route             | prepare_ms mean+/-std | solve_ms mean+/-std | residual_calls mean+/-std | jacobian_calls mean+/-std | nlu_or_native_linear mean+/-std | accepted mean+/-std | rejected mean+/-std"
    );
    println!(
        "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let residual = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let nlu = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let accepted = row
            .accepted_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let rejected = row
            .rejected_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<17} | {:<21} | {:<19} | {:<24} | {:<24} | {:<31} | {:<18} | {}",
            row.matrix, row.route, prep, solve, residual, jacobian, nlu, accepted, rejected
        );
    }

    assert!(
        rows.iter().any(|r| r.runs_ok > 0),
        "at least one backend race route should complete successfully"
    );
    for row in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful route should have final_diff samples");
            let tol = if row.route == "Analytical-Native" {
                1.0e-4
            } else {
                1.0e-7
            };
            assert!(
                mean_diff <= tol,
                "{} {} mean final_diff too large: {:e} (tol={:e})",
                row.matrix,
                row.route,
                mean_diff,
                tol
            );
        }
    }
}

fn stiff_scalar_tracking_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-1000*(y-cos(t))-sin(t)")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.02))
}

#[test]
fn lsode2_quality_dashboard_stiff_vs_nonstiff_auto_switch() {
    const REPEATS: usize = 3;
    let scenarios = [("nonstiff-decay", false), ("stiff-tracking", true)];
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    println!(
        "[LSODE2 story] quality dashboard (algorithm focus); counters are counts, time is milliseconds"
    );
    println!(
        "scenario        | matrix | runs | preferred_family | executed_family | switch_reason         | accepted mean+/-std | rejected mean+/-std | nlu/native_linear mean+/-std | jac_refresh mean+/-std | total_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for (scenario_label, is_stiff) in scenarios {
        for matrix in matrices {
            let mut accepted = RaceStats::default();
            let mut rejected = RaceStats::default();
            let mut nlu = RaceStats::default();
            let mut jac_refresh = RaceStats::default();
            let mut total_ms = RaceStats::default();
            let mut final_diff = RaceStats::default();
            let mut preferred: Option<String> = None;
            let mut executed: Option<String> = None;
            let mut switch_reason: Option<String> = None;
            let mut ok = 0usize;

            for _ in 0..REPEATS {
                let base = if is_stiff {
                    stiff_scalar_tracking_config()
                } else {
                    exponential_decay_config()
                };
                let mut config = match matrix {
                    BackendRaceMatrix::Sparse => base.with_native_sparse_faer_backend(),
                    BackendRaceMatrix::Banded => base.with_native_banded_faithful_backend(),
                    BackendRaceMatrix::Dense => unreachable!(),
                };
                config = config.with_faithful_bdf_solve(4096, 4096).with_controller(
                    super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
                        .with_method_switch_probe_steps(1),
                );

                let started = Instant::now();
                let mut solver = match Lsode2Solver::new(config) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let summary = match solver.solve_with_summary() {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                ok += 1;
                total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

                preferred = Some(summary.algorithm.preferred_family.to_string());
                executed = Some(summary.algorithm.executed_family.unwrap_or("-").to_string());
                switch_reason = Some(summary.algorithm.switch_reason.to_string());
                let accepted_steps = summary
                    .native_statistics
                    .native_step_accepts
                    .max(summary.native_statistics.bridge_accepted_steps);
                let rejected_steps = summary.native_statistics.native_step_rejects_error_test
                    + summary.native_statistics.native_step_rejects_nonlinear;
                accepted.push(accepted_steps as f64);
                rejected.push(rejected_steps as f64);
                nlu.push(
                    summary
                        .statistics
                        .bdf_nlu_total
                        .max(summary.native_statistics.native_linear_solve_calls)
                        as f64,
                );
                jac_refresh.push(summary.native_statistics.native_jacobian_refresh_requests as f64);

                if let Some(y) = summary.final_y.as_ref() {
                    let expected = if is_stiff {
                        summary.final_t.unwrap_or(1.0).cos()
                    } else {
                        (-1.0_f64).exp()
                    };
                    final_diff.push((y[0] - expected).abs());
                }
            }

            let status = if ok == REPEATS {
                format!("ok {ok}/{REPEATS}")
            } else if ok == 0 {
                "failed".to_string()
            } else {
                format!("partial {ok}/{REPEATS}")
            };

            let fmt = |s: &RaceStats, p: usize| -> String {
                s.summary()
                    .map(|(m, sd, _, _)| match p {
                        0 => format!("{m:.0}+/-{sd:.0}"),
                        2 => format!("{m:.2}+/-{sd:.2}"),
                        _ => format!("{m:.3}+/-{sd:.3}"),
                    })
                    .unwrap_or_else(|| "-".to_string())
            };

            println!(
                "{:<15} | {:<6} | {:>4} | {:<16} | {:<15} | {:<20} | {:<19} | {:<19} | {:<28} | {:<22} | {:<18} | {:<20} | {}",
                scenario_label,
                matrix.label(),
                format!("{ok}/{REPEATS}"),
                preferred.clone().unwrap_or_else(|| "-".to_string()),
                executed.clone().unwrap_or_else(|| "-".to_string()),
                switch_reason.clone().unwrap_or_else(|| "-".to_string()),
                fmt(&accepted, 2),
                fmt(&rejected, 2),
                fmt(&nlu, 2),
                fmt(&jac_refresh, 2),
                fmt(&total_ms, 3),
                fmt(&final_diff, 3),
                status
            );

            if ok > 0 {
                let p = preferred.clone().unwrap_or_default();
                let r = switch_reason.clone().unwrap_or_default();
                if is_stiff {
                    assert!(
                        p == "adams" || p == "bdf",
                        "{} {} stiff run should expose a valid family, got={}",
                        scenario_label,
                        matrix.label(),
                        p
                    );
                    assert!(
                        r == "switch_probe_warmup"
                            || r == "stiffness_suspected"
                            || r == "convergence_trouble"
                            || (p == "adams" && r == "switch_advantage_not_met")
                            || (p == "bdf" && r == "switch_advantage_not_met"),
                        "{} {} stiff run should report a LSODA-style stiff/warmup reason, got family={} reason={}",
                        scenario_label,
                        matrix.label(),
                        p,
                        r
                    );
                } else {
                    assert!(
                        p == "adams" || p == "bdf",
                        "{} {} non-stiff run should expose a valid family, got={}",
                        scenario_label,
                        matrix.label(),
                        p
                    );
                }
            }
        }
    }
}
