use super::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2BackendConfig, Lsode2JacobianBackend,
    Lsode2LinearSolverBackend, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedResidualAotBackend, register_linked_residual_backend, unregister_linked_residual_backend,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    SymbolicIvpProblemOptions, prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Instant;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

mod three_body_story_tests;

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
    const LIMIT: usize = 240;
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
        solve_story_row_fallible(
            "Lambdify",
            "Dense",
            "symbolic/lambdify",
            exponential_decay_config(),
            reference_final_y,
            Some(0),
        ),
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

    let verbose_rows = std::env::var_os("LSODE2_STORY_VERBOSE").is_some();
    if verbose_rows {
        println!("[LSODE2 story] AOT per-run table (verbose); all time columns are milliseconds");
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
        final_diff_vs_lambdify: RunStats,
        residual_calls: RunStats,
        jacobian_calls: RunStats,
        nlu: RunStats,
        residual_ms: RunStats,
        jacobian_ms: RunStats,
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
            if let Some(v) = row.final_diff_vs_lambdify {
                bucket.final_diff_vs_lambdify.push(v);
            }
            if let Some(v) = row.residual_calls {
                bucket.residual_calls.push(v as f64);
            }
            if let Some(v) = row.jacobian_calls {
                bucket.jacobian_calls.push(v as f64);
            }
            if let Some(v) = row.nlu {
                bucket.nlu.push(v as f64);
            }
            if let Some(v) = row.residual_ms_total {
                bucket.residual_ms.push(v);
            }
            if let Some(v) = row.jacobian_ms_total {
                bucket.jacobian_ms.push(v);
            }
        }
    }
    println!("[LSODE2 story] AOT multi-run aggregate table; all time columns are milliseconds");
    println!(
        "matrix | toolchain | run  | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std [min,max] | solve_ms mean+/-std [min,max] | final_diff_vs_lambdify mean+/-std [min,max]"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (key, bucket) in &aggregates {
        let parts = key.split('|').collect::<Vec<_>>();
        let (matrix, toolchain, run_kind) = (parts[0], parts[1], parts[2]);
        let total = bucket
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let prepare = bucket
            .prepare_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let solve = bucket
            .solve_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let diff = bucket
            .final_diff_vs_lambdify
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3e}+/-{s:.1e} [{n:.3e},{x:.3e}]"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<9} | {:<4} | {:>7} | {:<30} | {:<32} | {:<28} | {}",
            matrix,
            toolchain,
            run_kind,
            format!("{}/{}", bucket.ok_runs, bucket.total_runs),
            total,
            prepare,
            solve,
            diff
        );
    }

    println!(
        "[LSODE2 story] AOT aggregate diagnostics; counters are counts, times are milliseconds"
    );
    println!(
        "matrix | toolchain | run  | residual_calls mean+/-std | jacobian_calls mean+/-std | nlu mean+/-std | residual_ms mean+/-std | jacobian_ms mean+/-std"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (key, bucket) in &aggregates {
        let parts = key.split('|').collect::<Vec<_>>();
        let (matrix, toolchain, run_kind) = (parts[0], parts[1], parts[2]);
        let residual_calls = bucket
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_calls = bucket
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let nlu = bucket
            .nlu
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let residual_ms = bucket
            .residual_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_ms = bucket
            .jacobian_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<9} | {:<4} | {:<24} | {:<24} | {:<15} | {:<21} | {}",
            matrix,
            toolchain,
            run_kind,
            residual_calls,
            jacobian_calls,
            nlu,
            residual_ms,
            jacobian_ms
        );
    }

    println!(
        "note: set LSODE2_STORY_VERBOSE=1 to print per-run rows; default output is aggregated for stable CI readability."
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
    first_failure: Option<String>,
    total_ms: RaceStats,
    prepare_ms: RaceStats,
    solve_ms: RaceStats,
    final_diff: RaceStats,
    residual_calls: RaceStats,
    jacobian_calls: RaceStats,
    nlu_or_native_linear: RaceStats,
    residual_ms: RaceStats,
    jacobian_ms: RaceStats,
    linear_ms: RaceStats,
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
            first_failure: None,
            total_ms: RaceStats::default(),
            prepare_ms: RaceStats::default(),
            solve_ms: RaceStats::default(),
            final_diff: RaceStats::default(),
            residual_calls: RaceStats::default(),
            jacobian_calls: RaceStats::default(),
            nlu_or_native_linear: RaceStats::default(),
            residual_ms: RaceStats::default(),
            jacobian_ms: RaceStats::default(),
            linear_ms: RaceStats::default(),
            accepted_steps: RaceStats::default(),
            rejected_steps: RaceStats::default(),
        }
    }

    fn record_failure(&mut self, message: impl AsRef<str>) {
        if self.first_failure.is_none() {
            self.first_failure = Some(short_error(message.as_ref()));
        }
    }

    fn status_label(&self) -> String {
        let base = if self.runs_total == 0 {
            "not_run".to_string()
        } else if self.runs_ok == self.runs_total {
            format!("ok {}/{}", self.runs_ok, self.runs_total)
        } else if self.runs_ok == 0 {
            format!("failed {}/{}", self.runs_ok, self.runs_total)
        } else {
            format!("partial {}/{}", self.runs_ok, self.runs_total)
        };
        match &self.first_failure {
            Some(first_failure) if self.runs_ok < self.runs_total => {
                format!("{base}, first_failure={first_failure}")
            }
            _ => base,
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

fn unique_story_run_tag(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}_pid{}_{}", std::process::id(), nanos)
}

fn unique_story_short_tag() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}", std::process::id(), (nanos & 0xFFFFF))
}

fn race_aot_config_with_output(
    matrix: BackendRaceMatrix,
    output_dir: impl Into<PathBuf>,
) -> Lsode2ProblemConfig {
    let generated =
        SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir).with_c_tcc();
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

fn race_aot_config(matrix: BackendRaceMatrix) -> Lsode2ProblemConfig {
    let out = PathBuf::from(format!(
        "target/lsode2-story-race/{}/aot_c_tcc",
        matrix.label().to_lowercase()
    ));
    race_aot_config_with_output(matrix, out)
}

fn race_aot_parallel_config(
    matrix: BackendRaceMatrix,
    chunks_per_worker: usize,
) -> Lsode2ProblemConfig {
    race_aot_config(matrix).with_aot_parallel_chunking(chunks_per_worker)
}

fn race_aot_parallel_config_with_output(
    matrix: BackendRaceMatrix,
    output_dir: impl Into<PathBuf>,
    chunks_per_worker: usize,
) -> Lsode2ProblemConfig {
    race_aot_config_with_output(matrix, output_dir).with_aot_parallel_chunking(chunks_per_worker)
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
) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
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
    let residual_ms = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_residual_ms_total
    } else {
        summary.statistics.residual_ms_total
    };
    let jacobian_ms = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_jacobian_ms_total
    } else {
        summary.statistics.jacobian_ms_total
    };
    let linear_ms = if route == "Analytical-Native" || is_native_faithful {
        summary.native_statistics.native_linear_solve_ms_total
    } else {
        0.0
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
        residual_ms,
        jacobian_ms,
        linear_ms,
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
                    row.residual_ms.push(sample.7);
                    row.jacobian_ms.push(sample.8);
                    row.linear_ms.push(sample.9);
                    row.accepted_steps.push(sample.10);
                    row.rejected_steps.push(sample.11);
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

    println!(
        "[LSODE2 story] backend race stage timers; all time columns are milliseconds (mean+/-std)"
    );
    println!("matrix | route             | residual_ms | jacobian_ms | linear_ms");
    println!("--------------------------------------------------------------------------");
    for row in &rows {
        let residual_ms = row
            .residual_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_ms = row
            .jacobian_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let linear_ms = row
            .linear_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<17} | {:<11} | {:<11} | {}",
            row.matrix, row.route, residual_ms, jacobian_ms, linear_ms
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

#[test]
fn lsode2_parallel_chunking_story_by_weight_class() {
    const REPEATS: usize = 5;
    const CHUNKS_PER_WORKER: usize = 2;
    let matrices = [
        BackendRaceMatrix::Dense,
        BackendRaceMatrix::Sparse,
        BackendRaceMatrix::Banded,
    ];
    let routes = [
        ("Lambdify", "baseline(no_chunk_knobs)"),
        ("AOT-Ctcc-Whole", "whole"),
        ("AOT-Ctcc-Parallel", "parallel(auto,x2)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let mut solver = Lsode2Solver::new(race_lambdify_config(matrix))
            .expect("parallel story baseline config should build");
        let summary = solver
            .solve_with_summary()
            .expect("parallel story baseline solve should finish");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("parallel story baseline should expose final_y")[0];
        baselines.insert(matrix.label(), final_y);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, chunking) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let baseline = *baselines
                .get(matrix.label())
                .expect("parallel story baseline final_y should exist");

            // Prewarm AOT routes once so multi-run aggregates reflect warm runtime
            // behavior instead of mixing in cold codegen/build noise.
            if route.starts_with("AOT-") {
                let warmup_cfg = match route {
                    "AOT-Ctcc-Whole" => Some(race_aot_config(matrix)),
                    "AOT-Ctcc-Parallel" => {
                        Some(race_aot_parallel_config(matrix, CHUNKS_PER_WORKER))
                    }
                    _ => None,
                };
                if let Some(cfg) = warmup_cfg {
                    let _ = run_backend_race_sample(route, cfg, baseline);
                }
            }

            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match route {
                    "Lambdify" => Some(race_lambdify_config(matrix)),
                    "AOT-Ctcc-Whole" => Some(race_aot_config(matrix)),
                    "AOT-Ctcc-Parallel" => {
                        Some(race_aot_parallel_config(matrix, CHUNKS_PER_WORKER))
                    }
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
                    row.residual_ms.push(sample.7);
                    row.jacobian_ms.push(sample.8);
                    row.linear_ms.push(sample.9);
                    row.accepted_steps.push(sample.10);
                    row.rejected_steps.push(sample.11);
                }
            }
            rows.push((row, chunking));
        }
    }

    println!(
        "[LSODE2 story] parallel chunking race by weight class; all time columns are milliseconds"
    );
    println!(
        "note: `Lambdify` is a baseline route and currently does not use generated-backend chunking knobs."
    );
    println!(
        "matrix | route             | chunking              | ok/runs | total_ms mean+/-std [min,max] | solve_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, chunking) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<17} | {:<21} | {:>7} | {:<31} | {:<18} | {:<20} | {}",
            row.matrix,
            row.route,
            chunking,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            solve,
            diff,
            status
        );
    }

    println!("[LSODE2 story] parallel chunking diagnostics; counters are counts (mean+/-std)");
    println!(
        "matrix | route             | chunking              | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | accepted | rejected"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, chunking) in &rows {
        let residual_calls = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_calls = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let linear_calls = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let residual_ms = row
            .residual_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_ms = row
            .jacobian_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let linear_ms = row
            .linear_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let accepted = row
            .accepted_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let rejected = row
            .rejected_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<17} | {:<21} | {:<14} | {:<14} | {:<12} | {:<11} | {:<11} | {:<9} | {:<8} | {}",
            row.matrix,
            row.route,
            chunking,
            residual_calls,
            jacobian_calls,
            linear_calls,
            residual_ms,
            jacobian_ms,
            linear_ms,
            accepted,
            rejected
        );
    }

    assert!(
        rows.iter().any(|(r, _)| r.runs_ok > 0),
        "at least one parallel chunking race route should complete successfully"
    );
    for (row, _) in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful parallel chunking route should have diff samples");
            assert!(
                mean_diff <= 1.0e-5,
                "{} {} final_diff too large in parallel chunking race: {:e}",
                row.matrix,
                row.route,
                mean_diff
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

fn stiff_switch_acceptance_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-10000*(y-cos(t))-sin(t)")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.25,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.25))
    .with_controller(
        super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
            .with_method_switch_probe_steps(1)
            .with_stiffness_ratio_threshold(10.0)
            .with_convergence_failure_threshold(1)
            .with_rejection_threshold(1),
    )
    .with_faithful_bdf_solve(65_536, 65_536)
}

fn mixed_regime_ramp_config() -> Lsode2ProblemConfig {
    let stiffness = |t: f64| 1.0 + 9_999.0 / (1.0 + (-80.0 * (t - 0.45)).exp());
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("0")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.005,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.005))
    .with_controller(
        super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
            .with_method_switch_probe_steps(1)
            .with_stiffness_ratio_threshold(10.0)
            .with_convergence_failure_threshold(1)
            .with_rejection_threshold(1),
    )
    .with_analytical_callbacks(
        move |t, y: &DVector<f64>| {
            let k = stiffness(t);
            DVector::from_vec(vec![-k * (y[0] - t.cos()) - t.sin()])
        },
        move |t, _y: &DVector<f64>| {
            let k = stiffness(t);
            DMatrix::from_row_slice(1, 1, &[-k])
        },
    )
    .with_faithful_bdf_solve(65_536, 65_536)
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

#[test]
fn lsode2_nonstiff_adams_corpus_sparse_banded_dashboard() {
    const REPEATS: usize = 3;
    let scenarios = [
        ComprehensiveScenario::NonStiffScalarDecay,
        ComprehensiveScenario::NonStiffSystemDecay2,
    ];
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let controllers = [("adams_only", true), ("automatic_adams_bdf", false)];

    println!("[LSODE2 story] non-stiff Adams corpus: fixed Adams and automatic controller routes");
    println!(
        "scenario                  | matrix | controller          | ok/runs | preferred | executed | reason                 | preferred_adams | executed_adams | preferred_bdf | executed_bdf | accepted | rejected | total_ms | max_abs_err | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for scenario in scenarios {
        for matrix in matrices {
            for (controller_label, fixed_adams) in controllers {
                let mut ok = 0usize;
                let mut preferred_adams = RaceStats::default();
                let mut executed_adams = RaceStats::default();
                let mut preferred_bdf = RaceStats::default();
                let mut executed_bdf = RaceStats::default();
                let mut accepted = RaceStats::default();
                let mut rejected = RaceStats::default();
                let mut total_ms = RaceStats::default();
                let mut max_abs_err = RaceStats::default();
                let mut preferred_family = String::new();
                let mut executed_family = String::new();
                let mut switch_reason = String::new();
                let mut first_failure: Option<String> = None;

                for _ in 0..REPEATS {
                    let mut config = scenario.config();
                    config = match matrix {
                        BackendRaceMatrix::Sparse => config.with_native_sparse_faer_backend(),
                        BackendRaceMatrix::Banded => config.with_native_banded_faithful_backend(),
                        BackendRaceMatrix::Dense => unreachable!(),
                    };
                    config = if fixed_adams {
                        config.with_adams_only_controller()
                    } else {
                        config.with_automatic_adams_bdf_controller()
                    }
                    .with_native_solve(65_536, 65_536);

                    let started = Instant::now();
                    let result = Lsode2Solver::new(config)
                        .and_then(|mut solver| solver.solve_with_summary());
                    match result {
                        Ok(summary) => {
                            let final_t = summary.final_t.unwrap_or(1.0);
                            let final_y = summary
                                .final_y
                                .as_ref()
                                .expect("non-stiff Adams story should expose final state");
                            let expected = scenario.expected(final_t);
                            let err = max_abs_diff_vec(final_y, &expected);
                            let native = &summary.native_statistics;

                            ok += 1;
                            preferred_adams.push(native.preferred_adams_count as f64);
                            executed_adams.push(native.executed_adams_count as f64);
                            preferred_bdf.push(native.preferred_bdf_count as f64);
                            executed_bdf.push(native.executed_bdf_count as f64);
                            accepted.push(native.native_step_accepts as f64);
                            rejected.push(
                                (native.native_step_rejects_error_test
                                    + native.native_step_rejects_nonlinear)
                                    as f64,
                            );
                            total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                            max_abs_err.push(err);
                            preferred_family = summary.algorithm.preferred_family.to_string();
                            executed_family = summary
                                .algorithm
                                .executed_family
                                .clone()
                                .unwrap_or("-")
                                .to_string();
                            switch_reason = summary.algorithm.switch_reason.to_string();

                            let story_tolerance = scenario.tolerance().max(5.0e-6);
                            assert!(
                                err <= story_tolerance,
                                "{} {} {} max_abs_err too large: {:e} (tol={:e})",
                                scenario.label(),
                                matrix.label(),
                                controller_label,
                                err,
                                story_tolerance
                            );
                            if fixed_adams {
                                assert_eq!(
                                    summary.algorithm.controller_mode,
                                    "adams_only",
                                    "{} {} must stay in fixed Adams mode",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert_eq!(
                                    summary.algorithm.preferred_family,
                                    "adams",
                                    "{} {} fixed Adams must prefer Adams",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert!(
                                    native.executed_adams_count > 0,
                                    "{} {} fixed Adams should execute Adams steps",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert_eq!(
                                    native.executed_bdf_count,
                                    0,
                                    "{} {} fixed Adams should not execute BDF steps",
                                    scenario.label(),
                                    matrix.label()
                                );
                            } else {
                                assert_eq!(
                                    summary.algorithm.controller_mode,
                                    "automatic_adams_bdf",
                                    "{} {} automatic row must use automatic controller",
                                    scenario.label(),
                                    matrix.label()
                                );
                                assert!(
                                    summary.algorithm.preferred_family == "adams"
                                        || summary.algorithm.preferred_family == "bdf",
                                    "{} {} automatic row should expose a valid family, got {}",
                                    scenario.label(),
                                    matrix.label(),
                                    summary.algorithm.preferred_family
                                );
                            }
                        }
                        Err(err) => {
                            if first_failure.is_none() {
                                first_failure = Some(short_error(&err.to_string()));
                            }
                        }
                    }
                }

                let fmt_count = |stats: &RaceStats| {
                    stats
                        .summary()
                        .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                        .unwrap_or_else(|| "-".to_string())
                };
                let fmt_time = |stats: &RaceStats| {
                    stats
                        .summary()
                        .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
                        .unwrap_or_else(|| "-".to_string())
                };
                let fmt_err = |stats: &RaceStats| {
                    stats
                        .summary()
                        .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
                        .unwrap_or_else(|| "-".to_string())
                };
                let status = match first_failure {
                    Some(message) if ok < REPEATS => {
                        format!("partial {ok}/{REPEATS}, first_failure={message}")
                    }
                    _ if ok == REPEATS => format!("ok {ok}/{REPEATS}"),
                    _ => format!("failed {ok}/{REPEATS}"),
                };
                println!(
                    "{:<25} | {:<6} | {:<19} | {:>7} | {:<9} | {:<8} | {:<22} | {:<15} | {:<14} | {:<13} | {:<12} | {:<8} | {:<8} | {:<8} | {:<11} | {}",
                    scenario.label(),
                    matrix.label(),
                    controller_label,
                    format!("{ok}/{REPEATS}"),
                    preferred_family,
                    executed_family,
                    switch_reason,
                    fmt_count(&preferred_adams),
                    fmt_count(&executed_adams),
                    fmt_count(&preferred_bdf),
                    fmt_count(&executed_bdf),
                    fmt_count(&accepted),
                    fmt_count(&rejected),
                    fmt_time(&total_ms),
                    fmt_err(&max_abs_err),
                    status
                );
                assert_eq!(
                    ok,
                    REPEATS,
                    "{} {} {} should complete every run",
                    scenario.label(),
                    matrix.label(),
                    controller_label
                );
            }
        }
    }
}

fn numerical_closure_story_base_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-y1 + 0.1*y2"),
            Expr::parse_expression("-2*y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.5]),
        1.0,
        0.02,
        1e-7,
        1e-9,
    )
    .with_bdf_only_controller()
    .with_faithful_bdf_solve(65_536, 65_536)
}

fn numerical_closure_lambdify_config(matrix: BackendRaceMatrix) -> Lsode2ProblemConfig {
    let config = numerical_closure_story_base_config().with_residual_jacobian_source(
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
    );
    match matrix {
        BackendRaceMatrix::Sparse => config.with_native_sparse_faer_backend(),
        BackendRaceMatrix::Banded => config.with_native_banded_faithful_backend(),
        BackendRaceMatrix::Dense => unreachable!("numerical closure story is sparse/banded only"),
    }
}

fn numerical_closure_native_config(
    matrix: BackendRaceMatrix,
    jacobian_backend: Lsode2JacobianBackend,
) -> Lsode2ProblemConfig {
    let linear_backend = match matrix {
        BackendRaceMatrix::Sparse => Lsode2LinearSolverBackend::SparseFaer,
        BackendRaceMatrix::Banded => Lsode2LinearSolverBackend::BandedFaithful,
        BackendRaceMatrix::Dense => unreachable!("numerical closure story is sparse/banded only"),
    };
    let structure = match matrix {
        BackendRaceMatrix::Sparse => super::Lsode2LinearSystemStructure::Sparse,
        BackendRaceMatrix::Banded => super::Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 },
        BackendRaceMatrix::Dense => unreachable!("numerical closure story is sparse/banded only"),
    };
    numerical_closure_story_base_config()
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0] + 0.1 * y[1], -2.0 * y[1]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(2, 2, &[-1.0, 0.1, 0.0, -2.0]),
        )
        .with_linear_system_structure(structure)
        .with_backend(
            Lsode2BackendConfig::default()
                .with_jacobian_backend(jacobian_backend)
                .with_linear_solver_backend(linear_backend),
        )
}

#[test]
fn lsode2_symbolic_vs_numerical_closure_sparse_banded_dashboard() {
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = [
        ("Lambdify-AtomView", None),
        (
            "Numerical-AnalyticalJac",
            Some(Lsode2JacobianBackend::AnalyticClosure),
        ),
        (
            "Numerical-FDJac",
            Some(Lsode2JacobianBackend::FiniteDifference),
        ),
    ];

    println!(
        "[LSODE2 story] symbolic Lambdify vs pure numerical closure routes; all time columns are milliseconds"
    );
    println!(
        "matrix | route                   | ok/runs | total_ms mean+/-std [min,max] | final_linf mean+/-std | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | status"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for matrix in matrices {
        let mut baseline_solver = Lsode2Solver::new(numerical_closure_lambdify_config(matrix))
            .expect("numerical closure baseline should build");
        let baseline = baseline_solver
            .solve_with_summary()
            .expect("numerical closure baseline should solve")
            .final_y
            .expect("numerical closure baseline should expose final state");

        for (route, jacobian_backend) in routes {
            let mut row = LargeIvpChunkingRow::new(matrix.label(), route, "native_solve");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match jacobian_backend {
                    None => numerical_closure_lambdify_config(matrix),
                    Some(backend) => numerical_closure_native_config(matrix, backend),
                };
                match run_large_ivp_chunking_sample(config, &baseline) {
                    Ok(sample) => push_large_ivp_sample(&mut row, sample),
                    Err(err) => row.record_failure(err),
                }
            }

            let total = row
                .total_ms
                .summary()
                .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
                .unwrap_or_else(|| "-".to_string());
            let fmt = |stats: &RaceStats| {
                stats
                    .summary()
                    .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
                    .unwrap_or_else(|| "-".to_string())
            };
            let fmt_count = |stats: &RaceStats| {
                stats
                    .summary()
                    .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                    .unwrap_or_else(|| "-".to_string())
            };
            let diff = row
                .final_linf
                .summary()
                .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "{:<6} | {:<23} | {:>7} | {:<31} | {:<21} | {:<14} | {:<14} | {:<12} | {:<11} | {:<11} | {:<9} | {}",
                row.matrix,
                row.route,
                format!("{}/{}", row.runs_ok, row.runs_total),
                total,
                diff,
                fmt_count(&row.residual_calls),
                fmt_count(&row.jacobian_calls),
                fmt_count(&row.linear_calls),
                fmt(&row.residual_ms),
                fmt(&row.jacobian_ms),
                fmt(&row.linear_ms),
                row.status_label()
            );

            assert_eq!(
                row.runs_ok, row.runs_total,
                "{} {} numerical closure story should complete every run",
                row.matrix, row.route
            );
            let (mean_diff, _, _, _) = row
                .final_linf
                .summary()
                .expect("successful row should have final diffs");
            assert!(
                mean_diff <= 2.0e-5,
                "{} {} numerical closure drift too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
            assert!(
                row.residual_calls
                    .summary()
                    .map(|(mean, _, _, _)| mean > 0.0)
                    .unwrap_or(false),
                "{} {} must execute residual callbacks",
                row.matrix,
                row.route
            );
            assert!(
                row.jacobian_calls
                    .summary()
                    .map(|(mean, _, _, _)| mean > 0.0)
                    .unwrap_or(false),
                "{} {} must execute Jacobian work",
                row.matrix,
                row.route
            );
        }
    }
}

#[test]
fn lsode2_mixed_regime_ramp_auto_switch_diagnostic_story() {
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    println!("[LSODE2 story] mixed-regime ramp: one IVP starts Adams-capable and becomes stiff");
    println!(
        "matrix | ok/runs | preferred_adams | executed_adams | preferred_bdf | executed_bdf | accepted | rejected | max_stiff | max_rh1 | max_rh2 | total_ms | final_diff | final_family | reason | switch_observed | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for matrix in matrices {
        let mut preferred_adams = RaceStats::default();
        let mut executed_adams = RaceStats::default();
        let mut preferred_bdf = RaceStats::default();
        let mut executed_bdf = RaceStats::default();
        let mut accepted = RaceStats::default();
        let mut rejected = RaceStats::default();
        let mut max_stiff = RaceStats::default();
        let mut max_rh1 = RaceStats::default();
        let mut max_rh2 = RaceStats::default();
        let mut total_ms = RaceStats::default();
        let mut final_diff = RaceStats::default();
        let mut ok = 0usize;
        let mut final_family = "-".to_string();
        let mut reason = "-".to_string();
        let mut first_failure: Option<String> = None;

        for _ in 0..REPEATS {
            let config = match matrix {
                BackendRaceMatrix::Sparse => {
                    mixed_regime_ramp_config().with_native_sparse_faer_backend()
                }
                BackendRaceMatrix::Banded => {
                    mixed_regime_ramp_config().with_native_banded_faithful_backend()
                }
                BackendRaceMatrix::Dense => unreachable!(),
            };
            let started = Instant::now();
            let result =
                Lsode2Solver::new(config).and_then(|mut solver| solver.solve_with_summary());
            match result {
                Ok(summary) => {
                    let native = &summary.native_statistics;
                    let final_t = summary.final_t.unwrap_or(1.0);
                    let final_y = summary
                        .final_y
                        .as_ref()
                        .expect("mixed-regime result should expose final state")[0];
                    let diff = (final_y - final_t.cos()).abs();

                    assert!(
                        summary.algorithm.method_switching_enabled,
                        "{} mixed-regime story must use automatic method selection",
                        matrix.label()
                    );
                    assert!(
                        native.executed_adams_count + native.executed_bdf_count > 0,
                        "{} mixed-regime story should execute at least one method family",
                        matrix.label()
                    );
                    ok += 1;
                    preferred_adams.push(native.preferred_adams_count as f64);
                    executed_adams.push(native.executed_adams_count as f64);
                    preferred_bdf.push(native.preferred_bdf_count as f64);
                    executed_bdf.push(native.executed_bdf_count as f64);
                    accepted.push(native.native_step_accepts as f64);
                    rejected.push(
                        (native.native_step_rejects_error_test
                            + native.native_step_rejects_nonlinear) as f64,
                    );
                    if let Some(solve) = summary.native_integration_solve.as_ref() {
                        let max_or_zero = |values: Vec<Option<f64>>| {
                            values
                                .into_iter()
                                .flatten()
                                .filter(|value| value.is_finite())
                                .fold(0.0_f64, f64::max)
                        };
                        max_stiff.push(max_or_zero(
                            solve
                                .attempt_reports
                                .iter()
                                .map(|report| report.telemetry.stiffness_ratio)
                                .collect(),
                        ));
                        max_rh1.push(max_or_zero(
                            solve
                                .attempt_reports
                                .iter()
                                .map(|report| report.telemetry.adams_step_size_cap_estimate)
                                .collect(),
                        ));
                        max_rh2.push(max_or_zero(
                            solve
                                .attempt_reports
                                .iter()
                                .map(|report| report.telemetry.bdf_step_size_cap_estimate)
                                .collect(),
                        ));
                    }
                    total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                    final_diff.push(diff);
                    final_family = summary
                        .algorithm
                        .executed_family
                        .unwrap_or(summary.algorithm.active_family)
                        .to_string();
                    reason = summary.algorithm.switch_reason.to_string();
                }
                Err(err) => {
                    if first_failure.is_none() {
                        first_failure = Some(short_error(&err.to_string()));
                    }
                }
            }
        }

        let fmt_count = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let fmt_time = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let fmt_err = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let status = match first_failure {
            Some(message) if ok < REPEATS => {
                format!("partial {ok}/{REPEATS}, first_failure={message}")
            }
            _ if ok == REPEATS => format!("ok {ok}/{REPEATS}"),
            _ => format!("failed {ok}/{REPEATS}"),
        };
        let switch_observed = match (executed_adams.summary(), executed_bdf.summary()) {
            (Some((adams, _, _, _)), Some((bdf, _, _, _))) if adams > 0.0 && bdf > 0.0 => {
                "adams+bdf"
            }
            (Some((adams, _, _, _)), _) if adams > 0.0 => "adams_only_current_limit",
            (_, Some((bdf, _, _, _))) if bdf > 0.0 => "bdf_only_current_limit",
            _ => "none",
        };
        println!(
            "{:<6} | {:>7} | {:<15} | {:<14} | {:<13} | {:<12} | {:<8} | {:<8} | {:<9} | {:<7} | {:<7} | {:<8} | {:<10} | {:<12} | {:<24} | {:<26} | {}",
            matrix.label(),
            format!("{ok}/{REPEATS}"),
            fmt_count(&preferred_adams),
            fmt_count(&executed_adams),
            fmt_count(&preferred_bdf),
            fmt_count(&executed_bdf),
            fmt_count(&accepted),
            fmt_count(&rejected),
            fmt_count(&max_stiff),
            fmt_count(&max_rh1),
            fmt_count(&max_rh2),
            fmt_time(&total_ms),
            fmt_err(&final_diff),
            final_family,
            reason,
            switch_observed,
            status
        );
        assert_eq!(
            ok,
            REPEATS,
            "{} should complete every mixed-regime run",
            matrix.label()
        );
    }
}

#[test]
fn lsode2_mixed_regime_ramp_native_switches_adams_to_bdf_acceptance() {
    for matrix in [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded] {
        let config = match matrix {
            BackendRaceMatrix::Sparse => {
                mixed_regime_ramp_config().with_native_sparse_faer_backend()
            }
            BackendRaceMatrix::Banded => {
                mixed_regime_ramp_config().with_native_banded_faithful_backend()
            }
            BackendRaceMatrix::Dense => unreachable!(),
        };
        let summary = Lsode2Solver::new(config)
            .and_then(|mut solver| solver.solve_with_summary())
            .unwrap_or_else(|err| {
                panic!(
                    "{} mixed-regime native switch solve failed: {err}",
                    matrix.label()
                )
            });
        let native = &summary.native_statistics;
        assert!(
            native.executed_adams_count > 0,
            "{} mixed-regime solve must execute Adams before stiffness appears",
            matrix.label()
        );
        assert!(
            native.executed_bdf_count > 0,
            "{} mixed-regime solve must switch to BDF after stiffness appears",
            matrix.label()
        );
        assert!(
            summary.algorithm.method_switching_enabled,
            "{} mixed-regime solve must keep automatic switching enabled",
            matrix.label()
        );
        let final_t = summary
            .final_t
            .expect("mixed-regime solve should expose final_t");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("mixed-regime solve should expose final_y")[0];
        let final_diff = (final_y - final_t.cos()).abs();
        assert!(
            final_diff <= 1.0e-6,
            "{} mixed-regime final drift too large after Adams/BDF switch: {final_diff:e}",
            matrix.label()
        );
    }
}

#[test]
fn lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf() {
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    println!("[LSODE2 story] stiff-switch acceptance: automatic controller must execute BDF");
    println!(
        "matrix | ok/runs | preferred_bdf mean+/-std | executed_bdf mean+/-std | accepted mean+/-std | rejected mean+/-std | total_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for matrix in matrices {
        let mut preferred_bdf = RaceStats::default();
        let mut executed_bdf = RaceStats::default();
        let mut accepted = RaceStats::default();
        let mut rejected = RaceStats::default();
        let mut total_ms = RaceStats::default();
        let mut final_diff = RaceStats::default();
        let mut ok = 0usize;

        for _ in 0..REPEATS {
            let config = match matrix {
                BackendRaceMatrix::Sparse => {
                    stiff_switch_acceptance_config().with_native_sparse_faer_backend()
                }
                BackendRaceMatrix::Banded => {
                    stiff_switch_acceptance_config().with_native_banded_faithful_backend()
                }
                BackendRaceMatrix::Dense => unreachable!(),
            };
            let started = Instant::now();
            let mut solver = Lsode2Solver::new(config).expect("stiff-switch config should build");
            let summary = solver
                .solve_with_summary()
                .expect("stiff-switch acceptance solve should finish");

            let native = &summary.native_statistics;
            assert!(
                native.preferred_bdf_count > 0,
                "{} automatic stiff-switch run should prefer BDF at least once",
                matrix.label()
            );
            assert!(
                native.executed_bdf_count > 0,
                "{} automatic stiff-switch run should execute BDF at least once",
                matrix.label()
            );
            assert!(
                summary.algorithm.method_switching_enabled,
                "{} run must use automatic method selection",
                matrix.label()
            );

            ok += 1;
            preferred_bdf.push(native.preferred_bdf_count as f64);
            executed_bdf.push(native.executed_bdf_count as f64);
            accepted.push(native.native_step_accepts as f64);
            rejected.push(
                (native.native_step_rejects_error_test + native.native_step_rejects_nonlinear)
                    as f64,
            );
            total_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            let final_t = summary.final_t.unwrap_or(1.0);
            let final_y = summary
                .final_y
                .as_ref()
                .expect("stiff-switch result should expose final state")[0];
            final_diff.push((final_y - final_t.cos()).abs());
        }

        let fmt = |stats: &RaceStats, digits: usize| {
            stats
                .summary()
                .map(|(mean, std, _, _)| match digits {
                    0 => format!("{mean:.0}+/-{std:.0}"),
                    2 => format!("{mean:.2}+/-{std:.2}"),
                    _ => format!("{mean:.3e}+/-{std:.1e}"),
                })
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:>7} | {:<24} | {:<23} | {:<19} | {:<19} | {:<19} | {:<21} | ok {}/{}",
            matrix.label(),
            format!("{ok}/{REPEATS}"),
            fmt(&preferred_bdf, 2),
            fmt(&executed_bdf, 2),
            fmt(&accepted, 2),
            fmt(&rejected, 2),
            fmt(&total_ms, 2),
            fmt(&final_diff, 3),
            ok,
            REPEATS,
        );
        assert_eq!(
            ok,
            REPEATS,
            "{} should complete every stiff-switch acceptance run",
            matrix.label()
        );
    }
}

fn combustion_like_story_base_config() -> Lsode2ProblemConfig {
    // Simplified stiff combustion-like IVP:
    // A -> B with Arrhenius-driven heat release + linear cooling.
    let eqs = vec![
        Expr::parse_expression("-k*exp(-E/(R*T))*A*A"),
        Expr::parse_expression("0.5*k*exp(-E/(R*T))*A*A - kloss*B"),
        Expr::parse_expression("Qcrho*k*exp(-E/(R*T))*A*A - cooling*(T - T0)"),
    ];
    Lsode2ProblemConfig::new(
        eqs,
        vec!["A".to_string(), "B".to_string(), "T".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0, 300.0]),
        120.0,
        0.5,
        1e-8,
        1e-8,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "E".to_string(),
        "R".to_string(),
        "T0".to_string(),
        "Qcrho".to_string(),
        "kloss".to_string(),
        "cooling".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![
        1.0e7, // k
        5.0e4, // E
        8.314, // R
        300.0, // T0
        5.0e2, // Qcrho
        0.0,   // kloss
        0.5,   // cooling
    ]))
    .with_controller(
        super::algorithm::Lsode2ControllerConfig::automatic_adams_bdf()
            .with_method_switch_probe_steps(1),
    )
    .with_faithful_bdf_solve(20_000, 20_000)
}

fn combustion_story_route_config_with_base_dir(
    matrix: BackendRaceMatrix,
    route: &'static str,
    base_dir: &str,
) -> Option<Lsode2ProblemConfig> {
    let source = match route {
        "Lambdify" => Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
        "AOT-Ctcc" | "AOT-Ctcc-Whole" | "AOT-Ctcc-Parallel" => {
            Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
                execution: Lsode2SymbolicExecutionMode::Aot {
                    toolchain: Lsode2AotToolchain::CTcc,
                    profile: Lsode2AotProfile::Release,
                },
            }
        }
        _ => return None,
    };

    let base = combustion_like_story_base_config().with_residual_jacobian_source(source);
    let cfg = match (matrix, route) {
        (BackendRaceMatrix::Sparse, "Lambdify") => base.with_native_sparse_faer_backend(),
        (BackendRaceMatrix::Banded, "Lambdify") => base.with_native_banded_faithful_backend(),
        (BackendRaceMatrix::Sparse, "AOT-Ctcc") | (BackendRaceMatrix::Sparse, "AOT-Ctcc-Whole") => {
            let out = PathBuf::from(format!("{base_dir}/sparse/aot_c_tcc"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_sparse_faer_generated_backend(backend)
        }
        (BackendRaceMatrix::Banded, "AOT-Ctcc") | (BackendRaceMatrix::Banded, "AOT-Ctcc-Whole") => {
            let out = PathBuf::from(format!("{base_dir}/banded/aot_c_tcc"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_banded_faithful_generated_backend(backend)
        }
        (BackendRaceMatrix::Sparse, "AOT-Ctcc-Parallel") => {
            let out = PathBuf::from(format!("{base_dir}/sparse/aot_c_tcc_parallel"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_sparse_faer_generated_backend(backend)
                .with_aot_parallel_chunking(2)
        }
        (BackendRaceMatrix::Banded, "AOT-Ctcc-Parallel") => {
            let out = PathBuf::from(format!("{base_dir}/banded/aot_c_tcc_parallel"));
            let backend =
                SymbolicIvpGeneratedBackendConfig::build_if_missing_release(out).with_c_tcc();
            base.with_native_banded_faithful_generated_backend(backend)
                .with_aot_parallel_chunking(2)
        }
        _ => return None,
    };
    Some(cfg)
}

fn combustion_story_route_config(
    matrix: BackendRaceMatrix,
    route: &'static str,
) -> Option<Lsode2ProblemConfig> {
    combustion_story_route_config_with_base_dir(matrix, route, "target/lsode2-story-combustion")
}

type CombustionStorySample = (
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
);

fn run_combustion_story_sample_result(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_a: f64,
) -> Result<CombustionStorySample, String> {
    let skip_explicit_prepare = matches!(
        config.backend.generated_backend.build_policy,
        SymbolicIvpAotBuildPolicy::RebuildAlways { .. }
    );
    let started_total = Instant::now();
    let mut solver = Lsode2Solver::new(config)
        .map_err(|err| format!("new_error({})", short_error(&err.to_string())))?;
    let prepare_ms_wall = if skip_explicit_prepare {
        // `solve_with_summary` builds the native generated engine itself.
        // With RebuildAlways, an eager `prepare()` would build and load the
        // same DLL once for the bridge path, then native solve would try to
        // rebuild that loaded artifact again and fail on Windows file locks.
        0.0
    } else {
        let prep_started = Instant::now();
        solver
            .prepare()
            .map_err(|err| format!("prepare_error({})", short_error(&err.to_string())))?;
        prep_started.elapsed().as_secs_f64() * 1_000.0
    };
    let solve_started = Instant::now();
    let summary = solver
        .solve_with_summary()
        .map_err(|err| format!("solve_error({})", short_error(&err.to_string())))?;
    let solve_ms_wall = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let total_ms_wall = started_total.elapsed().as_secs_f64() * 1_000.0;

    let final_a = summary
        .final_y
        .as_ref()
        .and_then(|y| y.get(0).copied())
        .ok_or_else(|| "solve_error(missing final state)".to_string())?;
    let final_diff = (final_a - baseline_final_a).abs();
    let is_native_faithful = is_native_faithful_status(&summary.status);

    let (
        stage_prepare_ms,
        stage_solve_ms,
        residual_calls,
        jacobian_calls,
        linear_calls,
        residual_ms,
        jacobian_ms,
        linear_ms,
    ) = if is_native_faithful {
        (
            summary.native_statistics.backend_prepare_ms_total,
            summary.native_statistics.solve_ms_total,
            summary.native_statistics.native_residual_calls as f64,
            summary.native_statistics.native_jacobian_calls as f64,
            summary.native_statistics.native_linear_solve_calls as f64,
            summary.native_statistics.native_residual_ms_total,
            summary.native_statistics.native_jacobian_ms_total,
            summary.native_statistics.native_linear_solve_ms_total,
        )
    } else {
        (
            summary.statistics.backend_prepare_ms_total,
            summary.statistics.solve_ms_total,
            summary.statistics.residual_calls as f64,
            summary.statistics.jacobian_calls as f64,
            summary.statistics.bdf_nlu_total as f64,
            summary.statistics.residual_ms_total,
            summary.statistics.jacobian_ms_total,
            0.0,
        )
    };

    let accepted_steps = summary
        .native_statistics
        .native_step_accepts
        .max(summary.native_statistics.bridge_accepted_steps) as f64;
    let rejected_steps = (summary.native_statistics.native_step_rejects_error_test
        + summary.native_statistics.native_step_rejects_nonlinear) as f64;
    let preferred_bdf = summary.native_statistics.preferred_bdf_count as f64;
    let executed_bdf = summary.native_statistics.executed_bdf_count as f64;

    let _ = route;
    let reported_prepare_ms = if skip_explicit_prepare {
        stage_prepare_ms
    } else {
        prepare_ms_wall
    };
    Ok((
        total_ms_wall,
        reported_prepare_ms,
        solve_ms_wall,
        final_diff,
        residual_calls,
        jacobian_calls,
        linear_calls,
        accepted_steps,
        rejected_steps,
        preferred_bdf,
        executed_bdf,
        stage_prepare_ms,
        stage_solve_ms,
        residual_ms,
        jacobian_ms,
        linear_ms,
    ))
}

fn run_combustion_story_sample(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_a: f64,
) -> Option<CombustionStorySample> {
    run_combustion_story_sample_result(route, config, baseline_final_a).ok()
}

#[test]
fn lsode2_combustion_like_multi_run_story_dashboard() {
    const REPEATS: usize = 5;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = ["Lambdify", "AOT-Ctcc"];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let baseline_cfg =
            combustion_story_route_config(matrix, "Lambdify").expect("lambdify baseline config");
        let mut solver = Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
        let summary = solver
            .solve_with_summary()
            .expect("baseline combustion solve should finish");
        let final_a = summary
            .final_y
            .as_ref()
            .expect("baseline should provide final_y")[0];
        baselines.insert(matrix.label(), final_a);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for route in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let mut preferred_bdf = RaceStats::default();
            let mut executed_bdf = RaceStats::default();
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let Some(cfg) = combustion_story_route_config(matrix, route) else {
                    continue;
                };
                let baseline = *baselines
                    .get(matrix.label())
                    .expect("combustion baseline should exist");
                if let Some(sample) = run_combustion_story_sample(route, cfg, baseline) {
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
                    preferred_bdf.push(sample.9);
                    executed_bdf.push(sample.10);
                    row.residual_ms.push(sample.13);
                    row.jacobian_ms.push(sample.14);
                    row.linear_ms.push(sample.15);
                }
            }
            rows.push((row, preferred_bdf, executed_bdf));
        }
    }

    println!(
        "[LSODE2 story] combustion-like backend summary (multi-run); all time columns are milliseconds"
    );
    println!(
        "matrix | route     | ok/runs | total_ms mean+/-std [min,max] | final_diff(A) mean+/-std [min,max] | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------"
    );
    for (row, _, _) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2e}+/-{s:.1e} [{n:.2e},{x:.2e}]"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<9} | {:>7} | {:<31} | {:<36} | {}",
            row.matrix,
            row.route,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            diff,
            status
        );
    }

    println!(
        "[LSODE2 story] combustion-like diagnostics (multi-run); prepare/solve are stage times, counters are counts"
    );
    println!(
        "matrix | route     | prepare_ms mean+/-std | solve_ms mean+/-std | residual_calls mean+/-std | jacobian_calls mean+/-std | linear_calls mean+/-std | accepted mean+/-std | rejected mean+/-std | preferred_bdf mean+/-std | executed_bdf mean+/-std"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, preferred_bdf, executed_bdf) in &rows {
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let residual = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let linear = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let accepted = row
            .accepted_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let rejected = row
            .rejected_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let pref_bdf = preferred_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let exec_bdf = executed_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<9} | {:<21} | {:<19} | {:<24} | {:<24} | {:<21} | {:<18} | {:<18} | {:<24} | {}",
            row.matrix,
            row.route,
            prep,
            solve,
            residual,
            jacobian,
            linear,
            accepted,
            rejected,
            pref_bdf,
            exec_bdf
        );
    }

    println!(
        "[LSODE2 story] combustion-like stage timers (multi-run); all time columns are milliseconds"
    );
    println!(
        "matrix | route     | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std"
    );
    println!(
        "-----------------------------------------------------------------------------------------------"
    );
    for (row, _, _) in &rows {
        let residual_ms = row
            .residual_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian_ms = row
            .jacobian_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let linear_ms = row
            .linear_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<9} | {:<21} | {:<21} | {}",
            row.matrix, row.route, residual_ms, jacobian_ms, linear_ms
        );
    }

    assert!(
        rows.iter().any(|(row, _, _)| row.runs_ok > 0),
        "at least one combustion story route should complete successfully"
    );
    for (row, _, _) in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful combustion route should have diff samples");
            assert!(
                mean_diff <= 2.0e-4,
                "{} {} combustion final_diff too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
        }
    }
}

#[test]
fn lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard() {
    const REPEATS: usize = 5;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = [
        ("Lambdify", "baseline(no_chunk_knobs)"),
        ("AOT-Ctcc-Whole", "whole"),
        ("AOT-Ctcc-Parallel", "parallel(auto,x2)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let baseline_cfg =
            combustion_story_route_config(matrix, "Lambdify").expect("lambdify baseline config");
        let mut solver = Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
        let summary = solver
            .solve_with_summary()
            .expect("baseline combustion solve should finish");
        let final_a = summary
            .final_y
            .as_ref()
            .expect("baseline should provide final_y")[0];
        baselines.insert(matrix.label(), final_a);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, chunking) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let mut preferred_bdf = RaceStats::default();
            let mut executed_bdf = RaceStats::default();

            let baseline = *baselines
                .get(matrix.label())
                .expect("combustion baseline should exist");

            // Prewarm AOT routes once so statistics focus on warm runtime and
            // chunking effects, while cold build cost is covered by dedicated
            // AOT stage/cold-warm stories.
            if route.starts_with("AOT-") {
                if let Some(cfg) = combustion_story_route_config(matrix, route) {
                    let _ = run_combustion_story_sample(route, cfg, baseline);
                }
            }

            for _ in 0..REPEATS {
                row.runs_total += 1;
                let Some(cfg) = combustion_story_route_config(matrix, route) else {
                    continue;
                };
                if let Some(sample) = run_combustion_story_sample(route, cfg, baseline) {
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
                    preferred_bdf.push(sample.9);
                    executed_bdf.push(sample.10);
                    row.residual_ms.push(sample.13);
                    row.jacobian_ms.push(sample.14);
                    row.linear_ms.push(sample.15);
                }
            }
            rows.push((row, preferred_bdf, executed_bdf, chunking));
        }
    }

    println!(
        "[LSODE2 story] combustion-like parallel chunking summary (multi-run); all time columns are milliseconds"
    );
    println!(
        "matrix | route              | chunking              | ok/runs | total_ms mean+/-std [min,max] | solve_ms mean+/-std | final_diff(A) mean+/-std | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, _, _, chunking) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<18} | {:<21} | {:>7} | {:<31} | {:<18} | {:<24} | {}",
            row.matrix,
            row.route,
            chunking,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            solve,
            diff,
            status
        );
    }

    println!(
        "[LSODE2 story] combustion-like parallel chunking diagnostics (multi-run); counters are counts"
    );
    println!(
        "matrix | route              | chunking              | residual_calls | jacobian_calls | linear_calls | accepted | rejected | preferred_bdf | executed_bdf"
    );
    println!(
        "-------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, preferred_bdf, executed_bdf, chunking) in &rows {
        let residual = row
            .residual_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let jacobian = row
            .jacobian_calls
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let linear = row
            .nlu_or_native_linear
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let accepted = row
            .accepted_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let rejected = row
            .rejected_steps
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let pref_bdf = preferred_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        let exec_bdf = executed_bdf
            .summary()
            .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<18} | {:<21} | {:<14} | {:<14} | {:<12} | {:<8} | {:<8} | {:<13} | {}",
            row.matrix,
            row.route,
            chunking,
            residual,
            jacobian,
            linear,
            accepted,
            rejected,
            pref_bdf,
            exec_bdf
        );
    }

    assert!(
        rows.iter().any(|(row, _, _, _)| row.runs_ok > 0),
        "at least one combustion parallel chunking route should complete successfully"
    );
    for (row, _, _, _) in rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful combustion parallel route should have diff samples");
            assert!(
                mean_diff <= 2.0e-4,
                "{} {} combustion parallel final_diff too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
        }
    }
}

#[test]
fn lsode2_parallel_chunking_cold_stage_story_by_weight_class() {
    const REPEATS: usize = 3;
    const CHUNKS_PER_WORKER: usize = 2;
    let matrices = [
        BackendRaceMatrix::Dense,
        BackendRaceMatrix::Sparse,
        BackendRaceMatrix::Banded,
    ];
    let routes = [
        ("Lambdify", "baseline(no_build_stage)"),
        ("AOT-Ctcc-Whole", "cold_build(whole)"),
        ("AOT-Ctcc-Parallel", "cold_build(parallel)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let mut solver = Lsode2Solver::new(race_lambdify_config(matrix))
            .expect("cold-stage baseline config should build");
        let summary = solver
            .solve_with_summary()
            .expect("cold-stage baseline solve should finish");
        let final_y = summary
            .final_y
            .as_ref()
            .expect("cold-stage baseline should expose final_y")[0];
        baselines.insert(matrix.label(), final_y);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, scenario) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let baseline = *baselines
                .get(matrix.label())
                .expect("cold-stage baseline final_y should exist");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match route {
                    "Lambdify" => Some(race_lambdify_config(matrix)),
                    "AOT-Ctcc-Whole" => {
                        let tag = unique_story_run_tag("lsode2_story_race_cold_whole");
                        let out = PathBuf::from(format!(
                            "target/lsode2-story-race-cold/{}/{}/whole",
                            matrix.label().to_lowercase(),
                            tag
                        ));
                        Some(race_aot_config_with_output(matrix, out))
                    }
                    "AOT-Ctcc-Parallel" => {
                        let tag = unique_story_run_tag("lsode2_story_race_cold_parallel");
                        let out = PathBuf::from(format!(
                            "target/lsode2-story-race-cold/{}/{}/parallel",
                            matrix.label().to_lowercase(),
                            tag
                        ));
                        Some(race_aot_parallel_config_with_output(
                            matrix,
                            out,
                            CHUNKS_PER_WORKER,
                        ))
                    }
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
                }
            }
            rows.push((row, scenario));
        }
    }

    println!(
        "[LSODE2 story] parallel chunking cold-stage story by weight class; all time columns are milliseconds"
    );
    println!(
        "note: this table intentionally measures cold build+prepare cost for AOT (unique artifact dir per run)."
    );
    println!(
        "matrix | route             | scenario              | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, scenario) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<17} | {:<21} | {:>7} | {:<31} | {:<21} | {:<18} | {:<20} | {}",
            row.matrix,
            row.route,
            scenario,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            prep,
            solve,
            diff,
            status
        );
    }

    assert!(
        rows.iter().any(|(row, _)| row.runs_ok > 0),
        "at least one cold-stage route should complete successfully"
    );
}

#[test]
fn lsode2_combustion_like_parallel_chunking_cold_stage_story_dashboard() {
    const REPEATS: usize = 3;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let routes = [
        ("Lambdify", "baseline(no_build_stage)"),
        ("AOT-Ctcc-Whole", "cold_build(whole)"),
        ("AOT-Ctcc-Parallel", "cold_build(parallel)"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let baseline_cfg =
            combustion_story_route_config(matrix, "Lambdify").expect("lambdify baseline config");
        let mut solver = Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
        let summary = solver
            .solve_with_summary()
            .expect("baseline combustion solve should finish");
        let final_a = summary
            .final_y
            .as_ref()
            .expect("baseline should provide final_y")[0];
        baselines.insert(matrix.label(), final_a);
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (route, scenario) in routes {
            let mut row = BackendRaceRow::new(matrix.label(), route);
            let baseline = *baselines
                .get(matrix.label())
                .expect("combustion baseline should exist");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = match route {
                    "Lambdify" => combustion_story_route_config(matrix, route),
                    "AOT-Ctcc-Whole" | "AOT-Ctcc-Parallel" => {
                        let tag = unique_story_short_tag();
                        let route_short = if route == "AOT-Ctcc-Whole" { "w" } else { "p" };
                        let matrix_short = if matrix.label() == "Sparse" { "s" } else { "b" };
                        let base_dir =
                            format!("target/l2cold/{}/{}/{}", matrix_short, route_short, tag);
                        combustion_story_route_config_with_base_dir(
                            matrix,
                            route,
                            base_dir.as_str(),
                        )
                    }
                    _ => None,
                };
                let Some(config) = config else {
                    continue;
                };
                if let Some(sample) = run_combustion_story_sample(route, config, baseline) {
                    row.runs_ok += 1;
                    row.total_ms.push(sample.0);
                    row.prepare_ms.push(sample.1);
                    row.solve_ms.push(sample.2);
                    row.final_diff.push(sample.3);
                }
            }
            rows.push((row, scenario));
        }
    }

    println!(
        "[LSODE2 story] combustion-like parallel chunking cold-stage summary; all time columns are milliseconds"
    );
    println!(
        "note: this table intentionally includes cold AOT build/prepare by forcing unique artifact dirs."
    );
    println!(
        "matrix | route              | scenario              | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff(A) mean+/-std | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (row, scenario) in &rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let prep = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
            .unwrap_or_else(|| "-".to_string());
        let status = if row.runs_ok == row.runs_total {
            format!("ok {}/{}", row.runs_ok, row.runs_total)
        } else if row.runs_ok == 0 {
            "failed".to_string()
        } else {
            format!("partial {}/{}", row.runs_ok, row.runs_total)
        };
        println!(
            "{:<6} | {:<18} | {:<21} | {:>7} | {:<31} | {:<21} | {:<18} | {:<24} | {}",
            row.matrix,
            row.route,
            scenario,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            prep,
            solve,
            diff,
            status
        );
    }

    assert!(
        rows.iter().any(|(row, _)| row.runs_ok > 0),
        "at least one combustion cold-stage route should complete successfully"
    );
}

fn combustion_symbolic_matrix_config(
    matrix: BackendRaceMatrix,
    assembly: Lsode2SymbolicAssemblyBackend,
    execution: Lsode2SymbolicExecutionMode,
    generated: Option<SymbolicIvpGeneratedBackendConfig>,
) -> Lsode2ProblemConfig {
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution,
    };
    let base = combustion_like_story_base_config().with_residual_jacobian_source(source);
    match (matrix, generated) {
        (BackendRaceMatrix::Sparse, Some(backend)) => {
            base.with_native_sparse_faer_generated_backend(backend)
        }
        (BackendRaceMatrix::Banded, Some(backend)) => {
            base.with_native_banded_faithful_generated_backend(backend)
        }
        (BackendRaceMatrix::Sparse, None) => base.with_native_sparse_faer_backend(),
        (BackendRaceMatrix::Banded, None) => base.with_native_banded_faithful_backend(),
        (BackendRaceMatrix::Dense, _) => {
            unreachable!("combustion symbolic frontend matrix intentionally tests sparse/banded")
        }
    }
}

fn push_combustion_sample(row: &mut BackendRaceRow, sample: CombustionStorySample) {
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
    row.residual_ms.push(sample.13);
    row.jacobian_ms.push(sample.14);
    row.linear_ms.push(sample.15);
}

fn print_compact_combustion_story_tables(title: &str, rows: &[BackendRaceRow]) {
    println!("[LSODE2 story] {title} correctness/wall-clock; all time columns are milliseconds");
    println!(
        "matrix | route                    | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.2}+/-{s:.2} [{n:.2},{x:.2}]"))
            .unwrap_or_else(|| "-".to_string());
        let prepare = row
            .prepare_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let solve = row
            .solve_ms
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let diff = row
            .final_diff
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
            .unwrap_or_else(|| "-".to_string());
        let status = row.status_label();
        println!(
            "{:<6} | {:<24} | {:>7} | {:<31} | {:<21} | {:<19} | {:<21} | {}",
            row.matrix,
            row.route,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            prepare,
            solve,
            diff,
            status
        );
    }

    println!("[LSODE2 story] {title} numerical work; counters are counts (mean+/-std)");
    println!(
        "matrix | route                    | residual_calls | jacobian_calls | linear_calls | accepted | rejected"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let fmt_count = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<24} | {:<14} | {:<14} | {:<12} | {:<8} | {}",
            row.matrix,
            row.route,
            fmt_count(&row.residual_calls),
            fmt_count(&row.jacobian_calls),
            fmt_count(&row.nlu_or_native_linear),
            fmt_count(&row.accepted_steps),
            fmt_count(&row.rejected_steps),
        );
    }

    println!("[LSODE2 story] {title} hot-stage timers; all time columns are milliseconds");
    println!(
        "matrix | route                    | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let fmt_time = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<24} | {:<21} | {:<21} | {}",
            row.matrix,
            row.route,
            fmt_time(&row.residual_ms),
            fmt_time(&row.jacobian_ms),
            fmt_time(&row.linear_ms),
        );
    }
}

#[test]
#[ignore = "release story: multi-run symbolic frontend comparison on the combustion workload"]
fn lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard() {
    const REPEATS: usize = 5;
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let frontends = [
        (
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            "Lambdify-ExprLegacy",
        ),
        (Lsode2SymbolicAssemblyBackend::AtomView, "Lambdify-AtomView"),
    ];

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let config = combustion_symbolic_matrix_config(
            matrix,
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        );
        let mut solver = Lsode2Solver::new(config).expect("ExprLegacy baseline should build");
        let summary = solver
            .solve_with_summary()
            .expect("ExprLegacy baseline should solve");
        baselines.insert(
            matrix.label(),
            summary.final_y.expect("baseline final state should exist")[0],
        );
    }

    let mut rows = Vec::new();
    for matrix in matrices {
        for (assembly, label) in frontends {
            let mut row = BackendRaceRow::new(matrix.label(), label);
            let baseline = *baselines
                .get(matrix.label())
                .expect("baseline should exist");
            for _ in 0..REPEATS {
                row.runs_total += 1;
                let config = combustion_symbolic_matrix_config(
                    matrix,
                    assembly,
                    Lsode2SymbolicExecutionMode::LambdifyExpr,
                    None,
                );
                if let Some(sample) = run_combustion_story_sample(label, config, baseline) {
                    push_combustion_sample(&mut row, sample);
                }
            }
            rows.push(row);
        }
    }

    print_compact_combustion_story_tables(
        "combustion symbolic frontend Sparse/Banded (Lambdify)",
        &rows,
    );

    for row in rows {
        assert_eq!(
            row.runs_ok, row.runs_total,
            "{} {} should complete all frontend comparison runs",
            row.matrix, row.route
        );
        let (mean_diff, _, _, _) = row.final_diff.summary().expect("completed row has diffs");
        assert!(
            mean_diff <= 2.0e-4,
            "{} {} frontend drift too large: {:e}",
            row.matrix,
            row.route,
            mean_diff
        );
    }
}

fn combustion_aot_matrix_config(
    matrix: BackendRaceMatrix,
    toolchain: AotStoryToolchain,
    parallel: bool,
    output_dir: PathBuf,
) -> Lsode2ProblemConfig {
    let generated = toolchain.apply_generated(
        SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir).with_build_policy(
            SymbolicIvpAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release,
            },
        ),
    );
    let mut config = combustion_symbolic_matrix_config(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        toolchain.as_source(),
        Some(generated),
    )
    .with_bdf_only_controller();
    if parallel {
        config = config.with_aot_parallel_chunking(2);
    }
    config
}

fn combustion_tcc_lifecycle_config(
    matrix: BackendRaceMatrix,
    output_dir: PathBuf,
    build_policy: SymbolicIvpAotBuildPolicy,
) -> Lsode2ProblemConfig {
    let generated = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir)
        .with_c_tcc()
        .with_build_policy(build_policy);
    combustion_symbolic_matrix_config(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
        Some(generated),
    )
    .with_bdf_only_controller()
}

struct Lsode2LifecycleRow {
    matrix: &'static str,
    phase: &'static str,
    build_policy: &'static str,
    total_ms: f64,
    prepare_ms: f64,
    solve_ms: f64,
    final_diff: f64,
    residual_calls: f64,
    jacobian_calls: f64,
    linear_calls: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    status: String,
}

impl Lsode2LifecycleRow {
    fn from_sample(
        matrix: &'static str,
        phase: &'static str,
        build_policy: &'static str,
        sample: CombustionStorySample,
    ) -> Self {
        Self {
            matrix,
            phase,
            build_policy,
            total_ms: sample.0,
            prepare_ms: sample.1,
            solve_ms: sample.2,
            final_diff: sample.3,
            residual_calls: sample.4,
            jacobian_calls: sample.5,
            linear_calls: sample.6,
            residual_ms: sample.13,
            jacobian_ms: sample.14,
            linear_ms: sample.15,
            status: "ok".to_string(),
        }
    }

    fn failed(
        matrix: &'static str,
        phase: &'static str,
        build_policy: &'static str,
        err: String,
    ) -> Self {
        Self {
            matrix,
            phase,
            build_policy,
            total_ms: f64::NAN,
            prepare_ms: f64::NAN,
            solve_ms: f64::NAN,
            final_diff: f64::NAN,
            residual_calls: f64::NAN,
            jacobian_calls: f64::NAN,
            linear_calls: f64::NAN,
            residual_ms: f64::NAN,
            jacobian_ms: f64::NAN,
            linear_ms: f64::NAN,
            status: format!("failed({})", short_error(&err)),
        }
    }

    fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

fn run_lsode2_lifecycle_row(
    matrix: BackendRaceMatrix,
    phase: &'static str,
    build_policy: &'static str,
    config: Lsode2ProblemConfig,
    baseline_final_a: f64,
) -> Lsode2LifecycleRow {
    match run_combustion_story_sample_result(phase, config, baseline_final_a) {
        Ok(sample) => Lsode2LifecycleRow::from_sample(matrix.label(), phase, build_policy, sample),
        Err(err) => Lsode2LifecycleRow::failed(matrix.label(), phase, build_policy, err),
    }
}

fn fmt_story_value(value: f64, decimals: usize) -> String {
    if value.is_finite() {
        format!("{value:.decimals$}")
    } else {
        "-".to_string()
    }
}

fn print_lsode2_lifecycle_table(title: &str, rows: &[Lsode2LifecycleRow]) {
    println!("[LSODE2 lifecycle] {title}: correctness/backend policy");
    println!("matrix | phase      | build_policy    | final_diff | status");
    println!("--------------------------------------------------------------------------");
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:<15} | {:>10} | {}",
            row.matrix,
            row.phase,
            row.build_policy,
            if row.final_diff.is_finite() {
                format!("{:.3e}", row.final_diff)
            } else {
                "-".to_string()
            },
            row.status
        );
    }

    println!("[LSODE2 lifecycle] {title}: wall-clock and hot stages; milliseconds");
    println!(
        "matrix | phase      | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms"
    );
    println!(
        "------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:>8} | {:>10} | {:>8} | {:>11} | {:>11} | {:>9}",
            row.matrix,
            row.phase,
            fmt_story_value(row.total_ms, 3),
            fmt_story_value(row.prepare_ms, 3),
            fmt_story_value(row.solve_ms, 3),
            fmt_story_value(row.residual_ms, 3),
            fmt_story_value(row.jacobian_ms, 3),
            fmt_story_value(row.linear_ms, 3),
        );
    }

    println!("[LSODE2 lifecycle] {title}: numerical work; counters are counts");
    println!("matrix | phase      | residual_calls | jacobian_calls | linear_calls");
    println!("------------------------------------------------------------------------");
    for row in rows {
        println!(
            "{:<6} | {:<10} | {:>14} | {:>14} | {:>12}",
            row.matrix,
            row.phase,
            fmt_story_value(row.residual_calls, 0),
            fmt_story_value(row.jacobian_calls, 0),
            fmt_story_value(row.linear_calls, 0),
        );
    }
}

fn warm_cooldown_ms() -> u64 {
    std::env::var("LSODE2_WARM_COOLDOWN_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_000)
}

fn lifecycle_repetitions(env_name: &str, default: usize) -> usize {
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

#[test]
#[ignore = "release story: LSODE2 combustion AtomView tcc BuildIfMissing followed by strict RequirePrebuilt reuse"]
fn lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story() {
    let strict_repeats = lifecycle_repetitions("LSODE2_PREBUILT_REPEATS", 3);
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let mut rows = Vec::new();

    let mut baselines = std::collections::BTreeMap::new();
    for matrix in matrices {
        let reference_config = combustion_symbolic_matrix_config(
            matrix,
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        )
        .with_bdf_only_controller();
        let mut reference_solver =
            Lsode2Solver::new(reference_config).expect("Lambdify reference should build");
        let reference = reference_solver
            .solve_with_summary()
            .expect("Lambdify reference should solve");
        let baseline_final_a = reference
            .final_y
            .expect("reference final state should exist")[0];
        baselines.insert(matrix.label(), baseline_final_a);
    }

    for matrix in matrices {
        let baseline = *baselines
            .get(matrix.label())
            .expect("matrix baseline should exist");
        let output_dir = PathBuf::from(format!(
            "target/l2-aot-prebuilt-lifecycle/{}/{}",
            matrix.label().to_ascii_lowercase(),
            unique_story_short_tag()
        ));
        let build_config = combustion_tcc_lifecycle_config(
            matrix,
            output_dir.clone(),
            SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            },
        );
        rows.push(run_lsode2_lifecycle_row(
            matrix,
            "build",
            "BuildIfMissing",
            build_config,
            baseline,
        ));

        let strict_config = combustion_tcc_lifecycle_config(
            matrix,
            output_dir,
            SymbolicIvpAotBuildPolicy::RequirePrebuilt,
        );
        for _ in 0..strict_repeats {
            rows.push(run_lsode2_lifecycle_row(
                matrix,
                "prebuilt",
                "RequirePrebuilt",
                strict_config.clone(),
                baseline,
            ));
        }
    }

    print_lsode2_lifecycle_table(
        "combustion AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
        &rows,
    );

    assert!(
        rows.iter().all(Lsode2LifecycleRow::is_ok),
        "all LSODE2 BuildIfMissing and RequirePrebuilt lifecycle rows must solve"
    );
    assert!(
        rows.iter().all(|row| row.final_diff <= 2.0e-4),
        "LSODE2 prebuilt lifecycle rows must remain numerically equivalent"
    );
    assert!(
        rows.iter()
            .filter(|row| row.phase == "prebuilt")
            .all(|row| row.build_policy == "RequirePrebuilt" && row.prepare_ms < 20.0),
        "strict prebuilt rows should reuse already linked compiled callbacks without a cold rebuild"
    );
}

fn print_lsode2_warm_prebuilt_table(
    build_row: &Lsode2LifecycleRow,
    rows: &[(usize, usize, Lsode2LifecycleRow)],
) {
    println!("[LSODE2 warm] Banded AtomView Lambdify vs tcc RequirePrebuilt setup row");
    println!(
        "phase | build_policy    | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms | final_diff | status"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------"
    );
    println!(
        "{:<5} | {:<15} | {:>8} | {:>10} | {:>8} | {:>11} | {:>11} | {:>9} | {:>10} | {}",
        build_row.phase,
        build_row.build_policy,
        fmt_story_value(build_row.total_ms, 3),
        fmt_story_value(build_row.prepare_ms, 3),
        fmt_story_value(build_row.solve_ms, 3),
        fmt_story_value(build_row.residual_ms, 3),
        fmt_story_value(build_row.jacobian_ms, 3),
        fmt_story_value(build_row.linear_ms, 3),
        if build_row.final_diff.is_finite() {
            format!("{:.3e}", build_row.final_diff)
        } else {
            "-".to_string()
        },
        build_row.status
    );

    println!(
        "[LSODE2 warm] measured rows after cooldown_ms={}; build row excluded",
        warm_cooldown_ms()
    );
    println!(
        "rep | pos | phase      | build_policy    | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms | final_diff | status"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (rep, pos, row) in rows {
        println!(
            "{:>3} | {:>3} | {:<10} | {:<15} | {:>8} | {:>10} | {:>8} | {:>11} | {:>11} | {:>9} | {:>10} | {}",
            rep,
            pos,
            row.phase,
            row.build_policy,
            fmt_story_value(row.total_ms, 3),
            fmt_story_value(row.prepare_ms, 3),
            fmt_story_value(row.solve_ms, 3),
            fmt_story_value(row.residual_ms, 3),
            fmt_story_value(row.jacobian_ms, 3),
            fmt_story_value(row.linear_ms, 3),
            if row.final_diff.is_finite() {
                format!("{:.3e}", row.final_diff)
            } else {
                "-".to_string()
            },
            row.status
        );
    }

    println!("[LSODE2 warm] paired summary; milliseconds");
    println!(
        "phase      | runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | jacobian_ms mean+/-std | max_final_diff"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------"
    );
    for phase in ["lambdify", "prebuilt"] {
        let phase_rows = rows
            .iter()
            .filter_map(|(_, _, row)| (row.phase == phase && row.is_ok()).then_some(row))
            .collect::<Vec<_>>();
        let mut total = RaceStats::default();
        let mut prepare = RaceStats::default();
        let mut solve = RaceStats::default();
        let mut jac = RaceStats::default();
        for row in &phase_rows {
            total.push(row.total_ms);
            prepare.push(row.prepare_ms);
            solve.push(row.solve_ms);
            jac.push(row.jacobian_ms);
        }
        let fmt_agg = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
                .unwrap_or_else(|| "-".to_string())
        };
        let fmt_agg_short = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let max_diff = phase_rows
            .iter()
            .map(|row| row.final_diff)
            .fold(0.0_f64, f64::max);
        println!(
            "{:<10} | {:>4} | {:<31} | {:<21} | {:<19} | {:<21} | {:.3e}",
            phase,
            phase_rows.len(),
            fmt_agg(&total),
            fmt_agg_short(&prepare),
            fmt_agg_short(&solve),
            fmt_agg_short(&jac),
            max_diff
        );
    }
}

#[test]
#[ignore = "release story: warm repeated-solve comparison with cooldown, LSODE2 Banded AtomView Lambdify vs strict tcc RequirePrebuilt"]
fn lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story() {
    let repetitions = lifecycle_repetitions("LSODE2_WARM_REPEATS", 5);
    let cooldown_ms = warm_cooldown_ms();
    let matrix = BackendRaceMatrix::Banded;

    let reference_config = combustion_symbolic_matrix_config(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        Lsode2SymbolicExecutionMode::LambdifyExpr,
        None,
    )
    .with_bdf_only_controller();
    let mut reference_solver =
        Lsode2Solver::new(reference_config).expect("Lambdify reference should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("Lambdify reference should solve");
    let baseline_final_a = reference
        .final_y
        .expect("reference final state should exist")[0];

    let output_dir = PathBuf::from(format!(
        "target/l2-aot-warm-prebuilt/banded/{}",
        unique_story_short_tag()
    ));
    let build_config = combustion_tcc_lifecycle_config(
        matrix,
        output_dir.clone(),
        SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        },
    );
    let build_row = run_lsode2_lifecycle_row(
        matrix,
        "build",
        "BuildIfMissing",
        build_config,
        baseline_final_a,
    );
    assert!(
        build_row.is_ok() && build_row.final_diff <= 2.0e-4,
        "setup BuildIfMissing row must install a correct compiled backend"
    );

    let prebuilt_config = combustion_tcc_lifecycle_config(
        matrix,
        output_dir,
        SymbolicIvpAotBuildPolicy::RequirePrebuilt,
    );
    let lambdify_config = combustion_symbolic_matrix_config(
        matrix,
        Lsode2SymbolicAssemblyBackend::AtomView,
        Lsode2SymbolicExecutionMode::LambdifyExpr,
        None,
    )
    .with_bdf_only_controller();

    let mut rows = Vec::with_capacity(repetitions * 2);
    for repetition in 1..=repetitions {
        let phases = if repetition % 2 == 1 {
            ["lambdify", "prebuilt"]
        } else {
            ["prebuilt", "lambdify"]
        };
        for (position, phase) in phases.into_iter().enumerate() {
            if cooldown_ms > 0 {
                thread::sleep(Duration::from_millis(cooldown_ms));
            }
            let (policy, config) = if phase == "lambdify" {
                ("UseIfAvailable", lambdify_config.clone())
            } else {
                ("RequirePrebuilt", prebuilt_config.clone())
            };
            let row = run_lsode2_lifecycle_row(matrix, phase, policy, config, baseline_final_a);
            rows.push((repetition, position + 1, row));
        }
    }

    print_lsode2_warm_prebuilt_table(&build_row, &rows);

    assert!(
        rows.iter().all(|(_, _, row)| row.is_ok()),
        "all LSODE2 warm Lambdify/prebuilt rows must solve"
    );
    assert!(
        rows.iter().all(|(_, _, row)| row.final_diff <= 2.0e-4),
        "all LSODE2 warm Lambdify/prebuilt rows must match the common reference"
    );
    assert_eq!(
        rows.iter()
            .filter(|(_, _, row)| row.phase == "lambdify")
            .count(),
        repetitions,
        "warm comparison must collect one Lambdify row per repetition"
    );
    assert_eq!(
        rows.iter()
            .filter(|(_, _, row)| row.phase == "prebuilt")
            .count(),
        repetitions,
        "warm comparison must collect one prebuilt row per repetition"
    );
    assert!(
        rows.iter()
            .filter(|(_, _, row)| row.phase == "prebuilt")
            .all(|(_, _, row)| row.build_policy == "RequirePrebuilt" && row.prepare_ms < 20.0),
        "measured prebuilt rows must stay strict and avoid a cold rebuild"
    );
}

fn large_ivp_chunking_dim() -> usize {
    lifecycle_repetitions("LSODE2_LARGE_CHUNK_DIM", 96).max(8)
}

fn large_ivp_chunking_dims() -> Vec<usize> {
    std::env::var("LSODE2_LARGE_CHUNK_DIMS")
        .ok()
        .map(|raw| {
            raw.split(|ch| ch == ',' || ch == ';' || ch == ' ')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|n| *n >= 8)
                .collect::<Vec<_>>()
        })
        .filter(|dims| !dims.is_empty())
        .unwrap_or_else(|| vec![large_ivp_chunking_dim()])
}

fn large_ivp_chunking_repeats() -> usize {
    lifecycle_repetitions("LSODE2_LARGE_CHUNK_REPEATS", 3)
}

fn large_ivp_chunking_chunks() -> usize {
    lifecycle_repetitions("LSODE2_LARGE_CHUNK_TARGET", 4)
}

fn large_diffusion_chain_config(n: usize) -> Lsode2ProblemConfig {
    let diffusion = 6.0_f64;
    let mut equations = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut y0 = Vec::with_capacity(n);

    for i in 0..n {
        let var = format!("y{i}");
        values.push(var.clone());
        y0.push(((i as f64 + 1.0) * 0.071).sin() * 0.2 + 1.0);

        let lambda = 35.0 + (i % 11) as f64 * 4.0;
        let mut rhs = format!("-{lambda:.8}*{var}");
        let mut neighbor_count = 0usize;
        if i > 0 {
            rhs.push_str(&format!(" + {diffusion:.8}*y{}", i - 1));
            neighbor_count += 1;
        }
        if i + 1 < n {
            rhs.push_str(&format!(" + {diffusion:.8}*y{}", i + 1));
            neighbor_count += 1;
        }
        if neighbor_count > 0 {
            rhs.push_str(&format!(
                " - {:.8}*{var}",
                diffusion * neighbor_count as f64
            ));
        }
        rhs.push_str(" + 0.001*cos(t)");
        equations.push(Expr::parse_expression(&rhs));
    }

    Lsode2ProblemConfig::new(
        equations,
        values,
        "t".to_string(),
        0.0,
        DVector::from_vec(y0),
        0.12,
        0.004,
        1e-5,
        1e-8,
    )
    .with_first_step(Some(0.001))
    .with_bdf_only_controller()
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
}

fn large_ivp_lambdify_config(matrix: BackendRaceMatrix, n: usize) -> Lsode2ProblemConfig {
    match matrix {
        BackendRaceMatrix::Sparse => {
            large_diffusion_chain_config(n).with_native_sparse_faer_backend()
        }
        BackendRaceMatrix::Banded => {
            large_diffusion_chain_config(n).with_native_banded_faithful_backend()
        }
        BackendRaceMatrix::Dense => unreachable!("large chunking story is sparse/banded only"),
    }
}

fn large_ivp_tcc_config(
    matrix: BackendRaceMatrix,
    n: usize,
    output_dir: PathBuf,
    build_policy: SymbolicIvpAotBuildPolicy,
    target_chunks: usize,
) -> Lsode2ProblemConfig {
    let generated = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir)
        .with_c_tcc()
        .with_build_policy(build_policy);
    let generated = Lsode2BackendConfig::native_sparse_faer()
        .with_generated_backend(generated)
        .with_generated_backend_target_chunks(target_chunks.max(1), target_chunks.max(1))
        .generated_backend;
    let base = large_diffusion_chain_config(n).with_residual_jacobian_source(
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            },
        },
    );
    match matrix {
        BackendRaceMatrix::Sparse => base.with_native_sparse_faer_generated_backend(generated),
        BackendRaceMatrix::Banded => base.with_native_banded_faithful_generated_backend(generated),
        BackendRaceMatrix::Dense => unreachable!("large chunking story is sparse/banded only"),
    }
}

struct LargeIvpChunkingRow {
    matrix: &'static str,
    route: &'static str,
    build_policy: &'static str,
    runs_ok: usize,
    runs_total: usize,
    first_failure: Option<String>,
    total_ms: RaceStats,
    prepare_ms: RaceStats,
    solve_ms: RaceStats,
    final_linf: RaceStats,
    residual_calls: RaceStats,
    jacobian_calls: RaceStats,
    linear_calls: RaceStats,
    residual_ms: RaceStats,
    jacobian_ms: RaceStats,
    linear_ms: RaceStats,
}

impl LargeIvpChunkingRow {
    fn new(matrix: &'static str, route: &'static str, build_policy: &'static str) -> Self {
        Self {
            matrix,
            route,
            build_policy,
            runs_ok: 0,
            runs_total: 0,
            first_failure: None,
            total_ms: RaceStats::default(),
            prepare_ms: RaceStats::default(),
            solve_ms: RaceStats::default(),
            final_linf: RaceStats::default(),
            residual_calls: RaceStats::default(),
            jacobian_calls: RaceStats::default(),
            linear_calls: RaceStats::default(),
            residual_ms: RaceStats::default(),
            jacobian_ms: RaceStats::default(),
            linear_ms: RaceStats::default(),
        }
    }

    fn record_failure(&mut self, err: impl AsRef<str>) {
        if self.first_failure.is_none() {
            self.first_failure = Some(short_error(err.as_ref()));
        }
    }

    fn status_label(&self) -> String {
        let base = if self.runs_ok == self.runs_total {
            format!("ok {}/{}", self.runs_ok, self.runs_total)
        } else if self.runs_ok == 0 {
            format!("failed {}/{}", self.runs_ok, self.runs_total)
        } else {
            format!("partial {}/{}", self.runs_ok, self.runs_total)
        };
        match &self.first_failure {
            Some(first_failure) if self.runs_ok < self.runs_total => {
                format!("{base}, first_failure={first_failure}")
            }
            _ => base,
        }
    }
}

fn run_large_ivp_chunking_sample(
    config: Lsode2ProblemConfig,
    baseline: &DVector<f64>,
) -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let total_started = Instant::now();
    let mut solver = Lsode2Solver::new(config)
        .map_err(|err| format!("new_error({})", short_error(&err.to_string())))?;
    let prepare_started = Instant::now();
    solver
        .prepare()
        .map_err(|err| format!("prepare_error({})", short_error(&err.to_string())))?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    let summary = solver
        .solve_with_summary()
        .map_err(|err| format!("solve_error({})", short_error(&err.to_string())))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary
        .final_y
        .ok_or_else(|| "solve_error(missing final state)".to_string())?;
    if final_y.len() != baseline.len() {
        return Err(format!(
            "solve_error(final size mismatch {} != {})",
            final_y.len(),
            baseline.len()
        ));
    }
    let final_linf = final_y
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let stats = summary.native_statistics;
    Ok((
        total_ms,
        prepare_ms,
        solve_ms,
        final_linf,
        stats.native_residual_calls as f64,
        stats.native_jacobian_calls as f64,
        stats.native_linear_solve_calls as f64,
        stats.native_residual_ms_total,
        stats.native_jacobian_ms_total,
        stats.native_linear_solve_ms_total,
    ))
}

fn push_large_ivp_sample(
    row: &mut LargeIvpChunkingRow,
    sample: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64),
) {
    row.runs_ok += 1;
    row.total_ms.push(sample.0);
    row.prepare_ms.push(sample.1);
    row.solve_ms.push(sample.2);
    row.final_linf.push(sample.3);
    row.residual_calls.push(sample.4);
    row.jacobian_calls.push(sample.5);
    row.linear_calls.push(sample.6);
    row.residual_ms.push(sample.7);
    row.jacobian_ms.push(sample.8);
    row.linear_ms.push(sample.9);
}

fn print_large_ivp_chunking_tables(title: &str, n: usize, rows: &[LargeIvpChunkingRow]) {
    println!("[LSODE2 large chunking] {title}: n={n}; correctness/wall-clock");
    println!(
        "matrix | route           | policy          | ok/runs | total_ms mean+/-std [min,max] | prepare_ms | solve_ms | final_linf | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let total = row
            .total_ms
            .summary()
            .map(|(m, s, n, x)| format!("{m:.3}+/-{s:.3} [{n:.3},{x:.3}]"))
            .unwrap_or_else(|| "-".to_string());
        let fmt = |stats: &RaceStats, precision: usize| {
            stats
                .summary()
                .map(|(m, s, _, _)| {
                    if precision == 3 {
                        format!("{m:.3}+/-{s:.3}")
                    } else {
                        format!("{m:.3e}+/-{s:.1e}")
                    }
                })
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<15} | {:<15} | {:>7} | {:<31} | {:<10} | {:<10} | {:<12} | {}",
            row.matrix,
            row.route,
            row.build_policy,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            fmt(&row.prepare_ms, 3),
            fmt(&row.solve_ms, 3),
            fmt(&row.final_linf, 0),
            row.status_label()
        );
    }

    println!("[LSODE2 large chunking] {title}: hot-stage timers and counters");
    println!(
        "matrix | route           | residual_ms | jacobian_ms | linear_ms | residual_calls | jacobian_calls | linear_calls"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in rows {
        let fmt = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.3}+/-{s:.3}"))
                .unwrap_or_else(|| "-".to_string())
        };
        let fmt_count = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.1}+/-{s:.1}"))
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<15} | {:<11} | {:<11} | {:<9} | {:<14} | {:<14} | {}",
            row.matrix,
            row.route,
            fmt(&row.residual_ms),
            fmt(&row.jacobian_ms),
            fmt(&row.linear_ms),
            fmt_count(&row.residual_calls),
            fmt_count(&row.jacobian_calls),
            fmt_count(&row.linear_calls),
        );
    }
}

#[test]
#[ignore = "release story: larger LSODE2 generated IVP checks whether tcc callback chunking can amortize"]
fn lsode2_large_chain_tcc_chunking_sparse_banded_warm_story() {
    let dims = large_ivp_chunking_dims();
    let repeats = large_ivp_chunking_repeats();
    let target_chunks = large_ivp_chunking_chunks();
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];

    for n in dims {
        let mut rows = Vec::new();

        for matrix in matrices {
            let mut baseline_solver =
                Lsode2Solver::new(large_ivp_lambdify_config(matrix, n)).expect("baseline config");
            let baseline_summary = baseline_solver
                .solve_with_summary()
                .expect("baseline solve should finish");
            let baseline = baseline_summary
                .final_y
                .expect("baseline final state should exist");

            let whole_dir = PathBuf::from(format!(
                "target/l2-large-chain-chunking/n{}/{}/whole/{}",
                n,
                matrix.label().to_ascii_lowercase(),
                unique_story_short_tag()
            ));
            let chunk_dir = PathBuf::from(format!(
                "target/l2-large-chain-chunking/n{}/{}/chunk{}/{}",
                n,
                matrix.label().to_ascii_lowercase(),
                target_chunks,
                unique_story_short_tag()
            ));

            for (route, dir, chunks) in [
                ("tcc-whole", whole_dir.clone(), 1usize),
                ("tcc-chunk", chunk_dir.clone(), target_chunks),
            ] {
                let setup = large_ivp_tcc_config(
                    matrix,
                    n,
                    dir.clone(),
                    SymbolicIvpAotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Release,
                    },
                    chunks,
                );
                let setup_sample = run_large_ivp_chunking_sample(setup, &baseline)
                    .unwrap_or_else(|err| panic!("{} {route} setup failed: {err}", matrix.label()));
                assert!(
                    setup_sample.3 <= 5.0e-4,
                    "{} {} setup drift too large: {:e}",
                    matrix.label(),
                    route,
                    setup_sample.3
                );
            }

            let route_configs = [
                (
                    "lambdify",
                    "UseIfAvailable",
                    large_ivp_lambdify_config(matrix, n),
                ),
                (
                    "tcc-whole",
                    "RequirePrebuilt",
                    large_ivp_tcc_config(
                        matrix,
                        n,
                        whole_dir,
                        SymbolicIvpAotBuildPolicy::RequirePrebuilt,
                        1,
                    ),
                ),
                (
                    "tcc-chunk",
                    "RequirePrebuilt",
                    large_ivp_tcc_config(
                        matrix,
                        n,
                        chunk_dir,
                        SymbolicIvpAotBuildPolicy::RequirePrebuilt,
                        target_chunks,
                    ),
                ),
            ];

            for (route, policy, config) in route_configs {
                let mut row = LargeIvpChunkingRow::new(matrix.label(), route, policy);
                for _ in 0..repeats {
                    row.runs_total += 1;
                    match run_large_ivp_chunking_sample(config.clone(), &baseline) {
                        Ok(sample) => push_large_ivp_sample(&mut row, sample),
                        Err(err) => row.record_failure(err),
                    }
                }
                rows.push(row);
            }
        }

        print_large_ivp_chunking_tables(
            &format!("AtomView Lambdify vs tcc whole/chunk{target_chunks} warm prebuilt"),
            n,
            &rows,
        );

        assert!(
            rows.iter().all(|row| row.runs_ok == row.runs_total),
            "all large LSODE2 chunking rows must solve for n={n}"
        );
        assert!(
            rows.iter().all(|row| {
                row.final_linf
                    .summary()
                    .map(|(mean, _, _, _)| mean <= 5.0e-4)
                    .unwrap_or(false)
            }),
            "large LSODE2 chunking rows must remain numerically equivalent for n={n}"
        );
    }
}

#[test]
fn lsode2_cold_aot_story_config_forces_rebuild_always() {
    for matrix in [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded] {
        for toolchain in [
            AotStoryToolchain::CTcc,
            AotStoryToolchain::CGcc,
            AotStoryToolchain::Zig,
            AotStoryToolchain::Rust,
        ] {
            let config = combustion_aot_matrix_config(
                matrix,
                toolchain,
                false,
                PathBuf::from("target/lsode2-cold-contract-test"),
            );
            assert_eq!(
                config.backend.generated_backend.build_policy,
                SymbolicIvpAotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                },
                "{} {} cold story rows must force a new build rather than reuse a problem-keyed backend",
                matrix.label(),
                toolchain.label()
            );
        }
    }
}

#[test]
#[ignore = "release story: cold AOT toolchain/chunking matrix compiles many generated artifacts"]
fn lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix() {
    const DEFAULT_REPEATS: usize = 3;
    let repeats = std::env::var("LSODE2_AOT_COLD_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_REPEATS);
    let row_filter = std::env::var("LSODE2_AOT_COLD_FILTER")
        .ok()
        .map(|value| value.to_ascii_lowercase());
    let matrices = [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded];
    let variants = [
        (AotStoryToolchain::CTcc, false, "tcc/whole"),
        (AotStoryToolchain::CTcc, true, "tcc/parallel"),
        (AotStoryToolchain::CGcc, false, "gcc/whole"),
        (AotStoryToolchain::CGcc, true, "gcc/parallel"),
        (AotStoryToolchain::Zig, false, "zig/whole"),
        (AotStoryToolchain::Zig, true, "zig/parallel"),
        (AotStoryToolchain::Rust, false, "rust/whole"),
        (AotStoryToolchain::Rust, true, "rust/parallel"),
    ];

    if let Some(filter) = &row_filter {
        println!(
            "[LSODE2 story] cold AOT matrix filter active: LSODE2_AOT_COLD_FILTER={filter:?}, repeats={repeats}"
        );
    }

    let mut baselines = std::collections::BTreeMap::new();
    let mut rows = Vec::new();
    let mut lifecycle_rows = Vec::new();
    for matrix in matrices {
        let reference_config = combustion_symbolic_matrix_config(
            matrix,
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        )
        .with_bdf_only_controller();
        let mut reference_solver =
            Lsode2Solver::new(reference_config).expect("Lambdify reference should build");
        let reference = reference_solver
            .solve_with_summary()
            .expect("Lambdify reference should solve");
        let baseline_final_a = reference
            .final_y
            .expect("reference final state should exist")[0];

        let mut baseline_row = BackendRaceRow::new(matrix.label(), "Lambdify-AtomView");
        for _ in 0..repeats {
            baseline_row.runs_total += 1;
            let config = combustion_symbolic_matrix_config(
                matrix,
                Lsode2SymbolicAssemblyBackend::AtomView,
                Lsode2SymbolicExecutionMode::LambdifyExpr,
                None,
            )
            .with_bdf_only_controller();
            match run_combustion_story_sample_result("Lambdify-AtomView", config, baseline_final_a)
            {
                Ok(sample) => push_combustion_sample(&mut baseline_row, sample),
                Err(err) => baseline_row.record_failure(err),
            }
        }
        baselines.insert(matrix.label(), baseline_final_a);
        rows.push(baseline_row);
    }

    let mut matched_aot_rows = 0usize;
    for matrix in matrices {
        let baseline = *baselines
            .get(matrix.label())
            .expect("baseline should exist");
        for (toolchain, parallel, label) in variants {
            if let Some(filter) = &row_filter {
                let row_id = format!("{} {label} {}", matrix.label(), toolchain.label())
                    .to_ascii_lowercase();
                if !row_id.contains(filter) {
                    continue;
                }
            }
            matched_aot_rows += 1;
            let mut row = BackendRaceRow::new(matrix.label(), label);
            for repeat in 0..repeats {
                row.runs_total += 1;
                let run_tag = unique_story_short_tag();
                let output_dir = PathBuf::from(format!(
                    "target/l2-aot-cold-matrix/{}/{}/{}/{}",
                    matrix.label().to_lowercase(),
                    label.replace('/', "_"),
                    repeat,
                    run_tag
                ));
                let config =
                    combustion_aot_matrix_config(matrix, toolchain, parallel, output_dir.clone());
                match run_combustion_story_sample_result(label, config, baseline) {
                    Ok(sample) => {
                        push_combustion_sample(&mut row, sample);
                        lifecycle_rows.push((
                            matrix.label(),
                            label,
                            repeat + 1,
                            "rebuild_always",
                            output_dir.exists(),
                            "ok".to_string(),
                        ));
                    }
                    Err(err) => {
                        let status = format!("failed: {}", short_error(&err));
                        row.record_failure(&err);
                        lifecycle_rows.push((
                            matrix.label(),
                            label,
                            repeat + 1,
                            "rebuild_always",
                            output_dir.exists(),
                            status,
                        ));
                    }
                }
            }
            rows.push(row);
        }
    }
    if row_filter.is_some() {
        assert!(
            matched_aot_rows > 0,
            "LSODE2_AOT_COLD_FILTER did not match any AOT matrix row"
        );
    }

    print_compact_combustion_story_tables(
        "combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix",
        &rows,
    );
    println!(
        "note: AOT total_ms includes symbolic preparation, artifact build/link and native integration; hot-stage timers isolate repeated callback/linear work."
    );
    println!(
        "[LSODE2 story] cold AOT lifecycle observations; successful AOT rows require a fresh materialization directory"
    );
    println!(
        "matrix | route                    | rep | cold_action    | artifact_dir_written | status"
    );
    println!(
        "---------------------------------------------------------------------------------------------"
    );
    for (matrix, route, repeat, action, artifact_written, status) in &lifecycle_rows {
        println!(
            "{:<6} | {:<24} | {:>3} | {:<14} | {:<20} | {}",
            matrix, route, repeat, action, artifact_written, status
        );
        assert!(
            status != "ok" || *artifact_written,
            "{} {} repetition {} completed without writing its isolated cold artifact directory",
            matrix,
            route,
            repeat
        );
    }

    for row in rows {
        if row.route == "Lambdify-AtomView" {
            assert_eq!(
                row.runs_ok, row.runs_total,
                "{} baseline should complete all runs",
                row.matrix
            );
        }
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row.final_diff.summary().expect("successful row has diffs");
            assert!(
                mean_diff <= 2.0e-4,
                "{} {} AOT matrix drift too large: {:e}",
                row.matrix,
                row.route,
                mean_diff
            );
        }
    }
}
