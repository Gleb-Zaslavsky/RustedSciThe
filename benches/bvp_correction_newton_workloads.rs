use nalgebra::DMatrix;
use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;
use tabled::{Table, Tabled};

use RustedSciThe::Utils::test_reporting::{capture_test_block, write_test_report};
use RustedSciThe::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonMatrixLayout, CorrectionNewtonOptions, CorrectionNewtonProblem,
    CorrectionNewtonSolver, CorrectionNewtonSymbolicExecution, CorrectionNewtonSymbolicFrontend,
    CorrectionNewtonTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;

const DENSE_CONTROL_STEPS: usize = 128;
const LARGE_PERF_STEPS: usize = 24000;
const RATES: [f64; 6] = [0.7, 1.1, 1.8, 2.4, 3.2, 4.0];

#[derive(Tabled)]
struct Row {
    workload: &'static str,
    frontend: &'static str,
    execution: &'static str,
    layout: &'static str,
    status: &'static str,
    iterations: usize,
    residual_calls: usize,
    jacobian_calls: usize,
    factorizations: usize,
    prepare_ms: String,
    solve_ms: String,
    total_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    assembly_ms: String,
    factorization_ms: String,
    linear_solve_ms: String,
    max_error: String,
}

fn main() {
    let mut rows = Vec::new();
    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        for (layout, label, steps, workload) in [
            (
                CorrectionNewtonMatrixLayout::Dense,
                "Dense",
                DENSE_CONTROL_STEPS,
                "control",
            ),
            (
                CorrectionNewtonMatrixLayout::Sparse,
                "Sparse",
                LARGE_PERF_STEPS,
                "large",
            ),
            (
                CorrectionNewtonMatrixLayout::BandedAuto,
                "BandedAuto",
                LARGE_PERF_STEPS,
                "large",
            ),
        ] {
            for execution in [
                CorrectionNewtonSymbolicExecution::Lambdify,
                CorrectionNewtonSymbolicExecution::Aot,
            ] {
                let result = run(frontend, execution, layout, steps);
                match result {
                    Ok((report, prepare_elapsed, solve_elapsed, total_elapsed, max_error)) => rows
                        .push(Row {
                            workload,
                            frontend: match frontend {
                                CorrectionNewtonSymbolicFrontend::ExprLegacy => "ExprLegacy",
                                CorrectionNewtonSymbolicFrontend::AtomView => "AtomView",
                            },
                            execution: match execution {
                                CorrectionNewtonSymbolicExecution::Lambdify => "Lambdify",
                                CorrectionNewtonSymbolicExecution::Aot => "AOT",
                            },
                            layout: label,
                            status: "PASS",
                            iterations: report.iterations,
                            residual_calls: report.telemetry.residual_evaluations,
                            jacobian_calls: report.telemetry.jacobian_evaluations,
                            factorizations: report.telemetry.factorizations,
                            prepare_ms: format!("{prepare_elapsed:.3}"),
                            solve_ms: format!("{solve_elapsed:.3}"),
                            total_ms: format!("{total_elapsed:.3}"),
                            residual_ms: format!(
                                "{:.3}",
                                report.telemetry.residual_callback_time.as_secs_f64() * 1_000.0
                            ),
                            jacobian_ms: format!(
                                "{:.3}",
                                report.telemetry.jacobian_callback_time.as_secs_f64() * 1_000.0
                            ),
                            assembly_ms: format!(
                                "{:.3}",
                                report.telemetry.jacobian_assembly_time.as_secs_f64() * 1_000.0
                            ),
                            factorization_ms: format!(
                                "{:.3}",
                                report.telemetry.factorization_time.as_secs_f64() * 1_000.0
                            ),
                            linear_solve_ms: format!(
                                "{:.3}",
                                report.telemetry.linear_solve_time.as_secs_f64() * 1_000.0
                            ),
                            max_error: format!("{max_error:.3e}"),
                        }),
                    Err(error) => {
                        eprintln!("{frontend:?}/{label}: {error}");
                        rows.push(Row {
                            workload,
                            frontend: match frontend {
                                CorrectionNewtonSymbolicFrontend::ExprLegacy => "ExprLegacy",
                                CorrectionNewtonSymbolicFrontend::AtomView => "AtomView",
                            },
                            execution: match execution {
                                CorrectionNewtonSymbolicExecution::Lambdify => "Lambdify",
                                CorrectionNewtonSymbolicExecution::Aot => "AOT",
                            },
                            layout: label,
                            status: "FAIL",
                            iterations: 0,
                            residual_calls: 0,
                            jacobian_calls: 0,
                            factorizations: 0,
                            prepare_ms: "-".into(),
                            solve_ms: "-".into(),
                            total_ms: "-".into(),
                            residual_ms: "-".into(),
                            jacobian_ms: "-".into(),
                            assembly_ms: "-".into(),
                            factorization_ms: "-".into(),
                            linear_solve_ms: "-".into(),
                            max_error: "-".into(),
                        });
                    }
                }
            }
        }
    }
    let report = format!(
        "BVP_CorrectionNewton symbolic workload matrix: Dense is a small correctness control ({DENSE_CONTROL_STEPS} intervals); Sparse/BandedAuto are large single-solve performance workloads ({LARGE_PERF_STEPS} intervals). AOT artifact build is warmed/excluded.\n{}",
        Table::new(&rows)
    );
    println!("{report}");
    capture_test_block(&report);
    if let Err(error) = write_test_report(
        "BVP_CorrectionNewton_release",
        "bvp_correction_newton_workloads",
        &report,
    ) {
        eprintln!("could not archive BVP_CorrectionNewton report: {error}");
        std::process::exit(2);
    }
    if rows.iter().any(|row| row.status != "PASS") {
        std::process::exit(1);
    }
}

fn run(
    frontend: CorrectionNewtonSymbolicFrontend,
    execution: CorrectionNewtonSymbolicExecution,
    layout: CorrectionNewtonMatrixLayout,
    steps: usize,
) -> Result<
    (
        RustedSciThe::numerical::BVP_CorrectionNewton::CorrectionNewtonSolveReport,
        f64,
        f64,
        f64,
        f64,
    ),
    String,
> {
    let states = (0..RATES.len())
        .map(|index| format!("y{index}"))
        .collect::<Vec<_>>();
    let equations = states
        .iter()
        .zip(RATES)
        .map(|(state, rate)| Expr::parse_expression(&format!("-{rate}*{state}")))
        .collect::<Vec<_>>();
    let initial_guess = DMatrix::from_fn(RATES.len(), steps, |state, node| {
        let x = (node + 1) as f64 / steps as f64;
        (-RATES[state] * x).exp()
    });
    let problem = CorrectionNewtonProblem {
        equations,
        initial_guess,
        state_names: states.clone(),
        independent_variable: "x".into(),
        boundary_conditions: states
            .into_iter()
            .map(|state| (state, vec![(0, 1.0)]))
            .collect::<HashMap<_, _>>(),
        interval: (0.0, 1.0),
        steps,
    };
    let mut options = CorrectionNewtonOptions {
        symbolic_frontend: frontend,
        symbolic_execution: execution,
        matrix_layout: layout,
        telemetry_mode: CorrectionNewtonTelemetryMode::Timings,
        absolute_tolerance: 1.0e-7,
        default_relative_tolerance: 1.0e-7,
        max_iterations: 20,
        ..CorrectionNewtonOptions::default()
    };
    if execution == CorrectionNewtonSymbolicExecution::Aot {
        let frontend_name = match frontend {
            CorrectionNewtonSymbolicFrontend::ExprLegacy => "exprlegacy",
            CorrectionNewtonSymbolicFrontend::AtomView => "atomview",
        };
        let layout_name = match layout {
            CorrectionNewtonMatrixLayout::Dense => "dense",
            CorrectionNewtonMatrixLayout::Sparse => "sparse",
            CorrectionNewtonMatrixLayout::BandedAuto => "banded",
        };
        let config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
            PathBuf::from("target/bvp-correction-newton-bench-aot")
                .join(format!("{frontend_name}-{layout_name}")),
        )
        .with_c_tcc();
        options = options.with_aot_config(config);
    }
    let mut warmup = CorrectionNewtonSolver::from_symbolic(problem.clone(), options.clone())
        .map_err(|error| format!("warmup prepare failed: {error}"))?;
    let warmup_report = warmup
        .solve()
        .map_err(|error| format!("warmup solve failed: {error}"))?;
    black_box(&warmup_report);

    let total_started = Instant::now();
    let prepare_started = Instant::now();
    let mut solver = CorrectionNewtonSolver::from_symbolic(problem, options)
        .map_err(|error| format!("prepare failed: {error}"))?;
    let prepare_elapsed = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    let report = solver
        .solve()
        .map_err(|error| format!("solve failed: {error}"))?;
    let solve_elapsed = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let total_elapsed = total_started.elapsed().as_secs_f64() * 1_000.0;
    let full = solver
        .full_grid_values(&report.state)
        .map_err(|error| format!("profile expansion failed: {error}"))?;
    let mut max_error = 0.0_f64;
    for node in 0..=steps {
        for state in 0..RATES.len() {
            let half_step = 0.5 * RATES[state] / steps as f64;
            let ratio = (1.0 - half_step) / (1.0 + half_step);
            let expected = ratio.powi(node as i32);
            max_error = max_error.max((full[(node, state)] - expected).abs());
        }
    }
    if report.residual_norm.is_none_or(|value| value > 1.0e-7) || max_error > 1.0e-10 {
        return Err(format!(
            "correctness failed: residual={:?}, max error={max_error:e}",
            report.residual_norm
        ));
    }
    black_box(&full);
    Ok((
        report,
        prepare_elapsed,
        solve_elapsed,
        total_elapsed,
        max_error,
    ))
}
