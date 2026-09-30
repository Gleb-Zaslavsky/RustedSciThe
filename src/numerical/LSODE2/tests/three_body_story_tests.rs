use super::story_support::{BackendRaceRow, RaceStats, short_error, unique_story_short_tag};
use super::*;
use crate::numerical::LSODE2::{Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure};
use crate::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

pub(super) fn three_body_story_base_config() -> Lsode2ProblemConfig {
    three_body_story_base_config_with_horizon(500.0, 0.001)
}

fn three_body_story_base_config_with_horizon(
    final_time: f64,
    max_step: f64,
) -> Lsode2ProblemConfig {
    let workload = crate::numerical::LSODE2::workload_fixtures::three_body();
    let params: HashMap<String, f64> = workload
        .parameter_names
        .iter()
        .cloned()
        .zip(workload.parameter_values.iter().copied())
        .collect();
    let eq_sys = workload
        .equations
        .into_iter()
        .map(|equation| equation.set_variable_from_map(&params))
        .collect();

    Lsode2ProblemConfig::new(
        eq_sys,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        final_time,
        max_step,
        1e-10,
        1e-12,
    )
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_faithful_bdf_solve(250_000, 250_000)
}

fn three_body_story_physics_checks(solution: &DMatrix<f64>, times: &DVector<f64>) {
    assert_eq!(solution.nrows(), 12, "three-body state should have 12 rows");
    assert_eq!(
        solution.ncols(),
        times.len(),
        "solution columns must match time samples"
    );

    let k = 39.47841760435743;
    let m0 = 1.0;
    let m1 = 0.5;
    let m2 = 0.75;
    let m_sum = m0 + m1 + m2;

    let x0 = solution.row(0);
    let vx0 = solution.row(1);
    let y0 = solution.row(2);
    let vy0 = solution.row(3);
    let x1 = solution.row(4);
    let vx1 = solution.row(5);
    let y1 = solution.row(6);
    let vy1 = solution.row(7);
    let x2 = solution.row(8);
    let vx2 = solution.row(9);
    let y2 = solution.row(10);
    let vy2 = solution.row(11);

    let mut initial_energy = 0.0_f64;
    let mut initial_cm_x = 0.0_f64;
    let mut initial_cm_y = 0.0_f64;
    let mut initial_cm_vx = 0.0_f64;
    let mut initial_cm_vy = 0.0_f64;
    let mut max_energy_drift = 0.0_f64;
    let mut max_cm_velocity_drift = 0.0_f64;
    let mut max_cm_position_drift = 0.0_f64;

    for i in 0..solution.ncols() {
        let r01 = ((x0[i] - x1[i]).powi(2) + (y0[i] - y1[i]).powi(2)).sqrt();
        let r02 = ((x0[i] - x2[i]).powi(2) + (y0[i] - y2[i]).powi(2)).sqrt();
        let r12 = ((x1[i] - x2[i]).powi(2) + (y1[i] - y2[i]).powi(2)).sqrt();

        let kinetic = 0.5 * m0 * (vx0[i].powi(2) + vy0[i].powi(2))
            + 0.5 * m1 * (vx1[i].powi(2) + vy1[i].powi(2))
            + 0.5 * m2 * (vx2[i].powi(2) + vy2[i].powi(2));
        let potential = -k * (m0 * m1 / r01 + m0 * m2 / r02 + m1 * m2 / r12);
        let energy = kinetic + potential;

        let cm_x = (m0 * x0[i] + m1 * x1[i] + m2 * x2[i]) / m_sum;
        let cm_y = (m0 * y0[i] + m1 * y1[i] + m2 * y2[i]) / m_sum;
        let cm_vx = (m0 * vx0[i] + m1 * vx1[i] + m2 * vx2[i]) / m_sum;
        let cm_vy = (m0 * vy0[i] + m1 * vy1[i] + m2 * vy2[i]) / m_sum;

        if i == 0 {
            initial_energy = energy;
            initial_cm_x = cm_x;
            initial_cm_y = cm_y;
            initial_cm_vx = cm_vx;
            initial_cm_vy = cm_vy;
        }

        max_energy_drift = max_energy_drift.max((energy - initial_energy).abs());
        max_cm_velocity_drift = max_cm_velocity_drift
            .max(((cm_vx - initial_cm_vx).powi(2) + (cm_vy - initial_cm_vy).powi(2)).sqrt());

        let time = times[i];
        let expected_cm_x = initial_cm_x + initial_cm_vx * time;
        let expected_cm_y = initial_cm_y + initial_cm_vy * time;
        max_cm_position_drift = max_cm_position_drift
            .max(((cm_x - expected_cm_x).powi(2) + (cm_y - expected_cm_y).powi(2)).sqrt());
    }

    assert!(max_energy_drift.is_finite());
    assert!(max_cm_velocity_drift.is_finite());
    assert!(max_cm_position_drift.is_finite());
}

fn three_body_story_trajectory_drift(
    solution: &DMatrix<f64>,
    times: &DVector<f64>,
    baseline: &DMatrix<f64>,
    baseline_times: &DVector<f64>,
) -> f64 {
    let rows = solution.nrows().min(baseline.nrows());
    if rows == 0 || times.is_empty() || baseline_times.is_empty() {
        return f64::NAN;
    }

    let mut max_drift = 0.0_f64;
    let mut solution_col = 0;
    let mut compared = 0;
    for baseline_col in 0..baseline_times.len().min(baseline.ncols()) {
        let target_time = baseline_times[baseline_col];
        while solution_col + 1 < times.len()
            && solution_col + 1 < solution.ncols()
            && times[solution_col + 1] < target_time
        {
            solution_col += 1;
        }
        if solution_col + 1 >= times.len() || solution_col + 1 >= solution.ncols() {
            break;
        }
        if times[solution_col] > target_time {
            continue;
        }
        let left_time = times[solution_col];
        let right_time = times[solution_col + 1];
        let span = right_time - left_time;
        if span <= 0.0 {
            continue;
        }
        let alpha = ((target_time - left_time) / span).clamp(0.0, 1.0);
        let mut sum_sq = 0.0_f64;
        for row in 0..rows {
            let interpolated = solution[(row, solution_col)]
                + alpha * (solution[(row, solution_col + 1)] - solution[(row, solution_col)]);
            let delta = interpolated - baseline[(row, baseline_col)];
            sum_sq += delta * delta;
        }
        max_drift = max_drift.max(sum_sq.sqrt());
        compared += 1;
    }
    if compared == 0 { f64::NAN } else { max_drift }
}

fn three_body_story_config(
    matrix: &'static str,
    route: &'static str,
    output_dir: &str,
) -> Option<Lsode2ProblemConfig> {
    three_body_story_config_with_horizon(matrix, route, output_dir, 500.0, 0.001)
}

fn three_body_story_config_with_horizon(
    matrix: &'static str,
    route: &'static str,
    output_dir: &str,
    final_time: f64,
    max_step: f64,
) -> Option<Lsode2ProblemConfig> {
    let base = three_body_story_base_config_with_horizon(final_time, max_step);
    let source = match route {
        "Lambdify" => Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
        "AOT-Ctcc-Whole" | "AOT-Ctcc-Chunk4" | "AOT-Ctcc-Chunk12" => {
            Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::AtomView,
                execution: Lsode2SymbolicExecutionMode::Aot {
                    toolchain: Lsode2AotToolchain::CTcc,
                    profile: Lsode2AotProfile::Release,
                },
            }
        }
        _ => return None,
    };

    let mut config = match (matrix, route) {
        ("Sparse", "Lambdify") => base.with_native_sparse_faer_backend(),
        ("Sparse", "AOT-Ctcc-Whole") => base.with_native_sparse_faer_aot_c_tcc(output_dir),
        ("Sparse", "AOT-Ctcc-Chunk4") => base
            .with_native_sparse_faer_aot_c_tcc(output_dir)
            .with_aot_parallel_chunking(4),
        ("Sparse", "AOT-Ctcc-Chunk12") => base
            .with_native_sparse_faer_aot_c_tcc(output_dir)
            .with_aot_parallel_chunking(12),
        ("Banded", "Lambdify") => base.with_native_banded_faithful_backend(),
        ("Banded", "AOT-Ctcc-Whole") => base.with_native_banded_faithful_aot_c_tcc(output_dir),
        ("Banded", "AOT-Ctcc-Chunk4") => base
            .with_native_banded_faithful_aot_c_tcc(output_dir)
            .with_aot_parallel_chunking(4),
        ("Banded", "AOT-Ctcc-Chunk12") => base
            .with_native_banded_faithful_aot_c_tcc(output_dir)
            .with_aot_parallel_chunking(12),
        _ => return None,
    };

    config = config
        .with_residual_jacobian_source(source)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
        .with_faithful_bdf_solve(250_000, 250_000);

    Some(config)
}

#[test]
#[ignore = "release correctness: short-horizon three-body trajectory parity"]
fn lsode2_three_body_short_horizon_trajectory_parity() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_three_body_story_tests::lsode2_three_body_short_horizon_trajectory_parity",
    );
    const FINAL_TIME: f64 = 0.5;
    const MAX_STEP: f64 = 0.001;
    const TOLERANCE: f64 = 1.0e-4;
    let baseline_cfg = three_body_story_config_with_horizon(
        "Sparse",
        "Lambdify",
        "target/lsode2-three-body-short/lambdify",
        FINAL_TIME,
        MAX_STEP,
    )
    .expect("short-horizon baseline config should build");
    let mut baseline_solver =
        Lsode2Solver::new(baseline_cfg).expect("short-horizon baseline should construct");
    baseline_solver
        .prepare()
        .expect("short-horizon baseline prepare should succeed");
    baseline_solver
        .solve_with_summary()
        .expect("short-horizon baseline solve should succeed");
    let (baseline_times, baseline_solution) = baseline_solver.get_result();
    let baseline_solution = baseline_solution.transpose();
    three_body_story_physics_checks(&baseline_solution, &baseline_times);

    let routes = [
        ("Sparse", "AOT-Ctcc-Whole"),
        ("Sparse", "AOT-Ctcc-Chunk4"),
        ("Banded", "Lambdify"),
        ("Banded", "AOT-Ctcc-Whole"),
        ("Banded", "AOT-Ctcc-Chunk4"),
    ];
    println!(
        "[LSODE2 three-body short parity] final_time={FINAL_TIME}; max_step={MAX_STEP}; baseline_samples={}",
        baseline_times.len()
    );
    println!("matrix | route | samples | max_trajectory_drift | status");
    println!("-------------------------------------------------------------");
    for (matrix, route) in routes {
        let output_dir = format!("target/lsode2-three-body-short/{matrix}/{route}");
        let config =
            three_body_story_config_with_horizon(matrix, route, &output_dir, FINAL_TIME, MAX_STEP)
                .expect("short-horizon route config should build");
        let sample =
            run_three_body_story_sample_result(route, config, &baseline_solution, &baseline_times)
                .expect("short-horizon route should solve");
        let drift = sample.trajectory_drift;
        println!(
            "{matrix} | {route} | {} | {drift:.3e} | ok",
            baseline_times.len()
        );
        assert!(
            drift <= TOLERANCE,
            "short-horizon trajectory drift too large for {matrix}/{route}: {drift:e}"
        );
    }
}

fn three_body_story_chunking_summary(config: &Lsode2ProblemConfig) -> String {
    let generated = &config.backend.generated_backend;
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let residual_outputs = config.eq_system.len().max(1);
    let jacobian_rows = residual_outputs;
    let residual_chunks =
        story_residual_chunk_count(residual_outputs, generated.aot_options.residual_strategy);
    let jacobian_chunks =
        story_dense_jacobian_chunk_count(jacobian_rows, generated.aot_options.jacobian_strategy);
    let sparse_chunks =
        story_sparse_chunk_count(jacobian_rows, generated.sparse_jacobian_chunking_strategy);
    let residual_work_per_chunk = residual_outputs.div_ceil(residual_chunks.max(1)).max(1);
    let jacobian_work_per_chunk = jacobian_rows.div_ceil(jacobian_chunks.max(1)).max(1);
    let sparse_work_per_chunk = jacobian_rows.div_ceil(sparse_chunks.max(1)).max(1);
    let auto_choice = if residual_chunks == 1 && jacobian_chunks == 1 && sparse_chunks == 1 {
        "whole"
    } else {
        "parallel"
    };
    format!(
        "workers={workers} auto_choice={auto_choice} residual_outputs={residual_outputs} jacobian_rows={jacobian_rows} residual_chunks={residual_chunks} jacobian_chunks={jacobian_chunks} sparse_chunks={sparse_chunks} residual_work/chunk={residual_work_per_chunk} jacobian_work/chunk={jacobian_work_per_chunk} sparse_work/chunk={sparse_work_per_chunk} build_policy={:?} aot_backend={:?} residual_strategy={:?} jacobian_strategy={:?} sparse_strategy={:?}",
        generated.build_policy,
        generated.aot_codegen_backend,
        generated.aot_options.residual_strategy,
        generated.aot_options.jacobian_strategy,
        generated.sparse_jacobian_chunking_strategy,
    )
}

fn assert_three_body_whole_route_is_unchunked(config: &Lsode2ProblemConfig) {
    let generated = &config.backend.generated_backend;
    let residual_is_whole = matches!(
        generated.aot_options.residual_strategy,
        ResidualChunkingStrategy::Whole
    );
    let jacobian_is_whole = matches!(
        generated.aot_options.jacobian_strategy,
        DenseJacobianChunkingStrategy::Whole
    );
    let sparse_is_whole = matches!(
        generated.sparse_jacobian_chunking_strategy,
        SparseChunkingStrategy::Whole
    );
    assert!(
        residual_is_whole && jacobian_is_whole && sparse_is_whole,
        "three-body whole route must stay unchunked, got residual={:?}, jacobian={:?}, sparse={:?}",
        generated.aot_options.residual_strategy,
        generated.aot_options.jacobian_strategy,
        generated.sparse_jacobian_chunking_strategy,
    );
}

fn story_residual_chunk_count(total_outputs: usize, strategy: ResidualChunkingStrategy) -> usize {
    match strategy {
        ResidualChunkingStrategy::Whole => 1,
        ResidualChunkingStrategy::ByTargetChunkCount { target_chunks } => {
            let chunk_size = total_outputs.max(1).div_ceil(target_chunks.max(1)).max(1);
            total_outputs.max(1).div_ceil(chunk_size).max(1)
        }
        ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk,
        } => total_outputs
            .max(1)
            .div_ceil(max_outputs_per_chunk.max(1))
            .max(1),
    }
}

fn story_dense_jacobian_chunk_count(rows: usize, strategy: DenseJacobianChunkingStrategy) -> usize {
    match strategy {
        DenseJacobianChunkingStrategy::Whole => 1,
        DenseJacobianChunkingStrategy::ByTargetChunkCount { target_chunks } => {
            let chunk_size = rows.max(1).div_ceil(target_chunks.max(1)).max(1);
            rows.max(1).div_ceil(chunk_size).max(1)
        }
        DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk } => {
            rows.max(1).div_ceil(rows_per_chunk.max(1)).max(1)
        }
    }
}

fn story_sparse_chunk_count(rows: usize, strategy: SparseChunkingStrategy) -> usize {
    match strategy {
        SparseChunkingStrategy::Whole => 1,
        SparseChunkingStrategy::ByTargetChunkCount { target_chunks } => {
            let chunk_size = rows.max(1).div_ceil(target_chunks.max(1)).max(1);
            rows.max(1).div_ceil(chunk_size).max(1)
        }
        SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
            rows.max(1).div_ceil(rows_per_chunk.max(1)).max(1)
        }
        SparseChunkingStrategy::ByNonZeroCount { .. } => 0,
    }
}

struct ThreeBodyStorySample {
    total_ms: f64,
    prepare_ms: f64,
    solve_ms: f64,
    trajectory_drift: f64,
    counter_scope: &'static str,
    residual_calls: f64,
    jacobian_calls: f64,
    linear_calls: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    accepted_steps: f64,
    rejected_steps: f64,
}

fn run_three_body_story_sample_result(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_solution: &DMatrix<f64>,
    baseline_times: &DVector<f64>,
) -> Result<ThreeBodyStorySample, String> {
    let started_total = Instant::now();
    let mut solver = Lsode2Solver::new(config).map_err(|err| short_error(&err.to_string()))?;
    let started_prepare = Instant::now();
    solver
        .prepare()
        .map_err(|err| short_error(&err.to_string()))?;
    let prepare_ms = started_prepare.elapsed().as_secs_f64() * 1_000.0;
    let started_solve = Instant::now();
    let summary = solver
        .solve_with_summary()
        .map_err(|err| short_error(&err.to_string()))?;
    let solve_ms = started_solve.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = started_total.elapsed().as_secs_f64() * 1_000.0;

    let (times, solution) = solver.get_result();
    let solution = solution.transpose();
    three_body_story_physics_checks(&solution, &times);
    let trajectory_drift =
        three_body_story_trajectory_drift(&solution, &times, baseline_solution, baseline_times);
    if solution.ncols() != baseline_solution.ncols() {
        eprintln!(
            "[LSODE2 three-body] diagnostic note: {} produced {} samples vs baseline {} samples; trajectory_drift is time-aligned by interpolation",
            route,
            solution.ncols(),
            baseline_solution.ncols()
        );
    }

    let telemetry = summary.evaluation_telemetry;

    Ok(ThreeBodyStorySample {
        total_ms,
        prepare_ms,
        solve_ms,
        trajectory_drift,
        counter_scope: telemetry.scope.label(),
        residual_calls: telemetry.residual_evaluations as f64,
        jacobian_calls: telemetry.jacobian_evaluations as f64,
        linear_calls: telemetry.linear_solves as f64,
        residual_ms: telemetry.residual_ms_total,
        jacobian_ms: telemetry.jacobian_ms_total,
        linear_ms: telemetry.linear_ms_total,
        accepted_steps: telemetry.accepted_steps as f64,
        rejected_steps: telemetry.rejected_steps as f64,
    })
}

#[test]
#[ignore = "release story: three-body LSODE2 compares Lambdify vs tcc whole and chunked runtime"]
fn lsode2_three_body_problem_backend_story_dashboard() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_three_body_story_tests::lsode2_three_body_problem_backend_story_dashboard",
    );
    const REPEATS: usize = 4;
    let routes = [
        (
            "Sparse",
            "Lambdify",
            "lambdify",
            false,
            "with_native_sparse_faer_backend()",
        ),
        (
            "Sparse",
            "AOT-Ctcc-Whole",
            "whole",
            true,
            "with_native_sparse_faer_aot_c_tcc(output_dir)",
        ),
        (
            "Sparse",
            "AOT-Ctcc-Chunk4",
            "chunk4",
            true,
            "with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(4)",
        ),
        (
            "Sparse",
            "AOT-Ctcc-Chunk12",
            "chunk12",
            true,
            "with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(12)",
        ),
        (
            "Banded",
            "Lambdify",
            "lambdify",
            false,
            "with_native_banded_faithful_backend()",
        ),
        (
            "Banded",
            "AOT-Ctcc-Whole",
            "whole",
            true,
            "with_native_banded_faithful_aot_c_tcc(output_dir)",
        ),
        (
            "Banded",
            "AOT-Ctcc-Chunk4",
            "chunk4",
            true,
            "with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(4)",
        ),
        (
            "Banded",
            "AOT-Ctcc-Chunk12",
            "chunk12",
            true,
            "with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(12)",
        ),
    ];

    let baseline_cfg = three_body_story_config(
        "Sparse",
        "Lambdify",
        "target/lsode2-three-body-story/lambdify",
    )
    .expect("lambdify baseline config should build");
    let mut baseline_solver =
        Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
    baseline_solver
        .prepare()
        .expect("baseline prepare should succeed");
    let baseline_summary = baseline_solver
        .solve_with_summary()
        .expect("baseline three-body solve should finish");
    let (baseline_times, baseline_solution) = baseline_solver.get_result();
    let baseline_solution = baseline_solution.transpose();
    three_body_story_physics_checks(&baseline_solution, &baseline_times);

    let run_tag = unique_story_short_tag();
    let mut rows = Vec::new();
    for (matrix, route, suffix, needs_tcc, builder_hint) in routes {
        let mut row = BackendRaceRow::new(matrix, route);
        let output_dir = format!("target/lsode2-three-body-story/{run_tag}/{matrix}/{suffix}");
        if needs_tcc && !command_available("tcc") {
            row.record_failure("tcc not available on PATH; AOT row skipped");
            row.runs_total = REPEATS;
            rows.push(row);
            continue;
        }
        println!(
            "[LSODE2 three-body] matrix={matrix} route={route} builder={builder_hint} output_dir={output_dir} repeats={REPEATS}"
        );
        if needs_tcc {
            if let Some(cfg) = three_body_story_config(matrix, route, &output_dir) {
                if route == "AOT-Ctcc-Whole" {
                    assert_three_body_whole_route_is_unchunked(&cfg);
                }
                println!(
                    "[LSODE2 three-body] matrix={matrix} route={route} chunking_plan={}",
                    three_body_story_chunking_summary(&cfg)
                );
                let _ = run_three_body_story_sample_result(
                    route,
                    cfg,
                    &baseline_solution,
                    &baseline_times,
                );
            }
        }
        for rep in 0..REPEATS {
            row.runs_total += 1;
            println!(
                "[LSODE2 three-body] matrix={matrix} route={route} builder={builder_hint} rep={}/{} output_dir={output_dir}",
                rep + 1,
                REPEATS,
            );
            let Some(cfg) = three_body_story_config(matrix, route, &output_dir) else {
                row.record_failure("three-body route config is unavailable");
                continue;
            };
            if route == "AOT-Ctcc-Whole" {
                assert_three_body_whole_route_is_unchunked(&cfg);
            }
            println!(
                "[LSODE2 three-body] matrix={matrix} route={route} chunking_plan={}",
                three_body_story_chunking_summary(&cfg)
            );
            match run_three_body_story_sample_result(
                route,
                cfg,
                &baseline_solution,
                &baseline_times,
            ) {
                Ok(sample) => {
                    row.runs_ok += 1;
                    row.counter_scope = Some(sample.counter_scope);
                    row.total_ms.push(sample.total_ms);
                    row.prepare_ms.push(sample.prepare_ms);
                    row.solve_ms.push(sample.solve_ms);
                    row.final_diff.push(sample.trajectory_drift);
                    row.residual_calls.push(sample.residual_calls);
                    row.jacobian_calls.push(sample.jacobian_calls);
                    row.nlu_or_native_linear.push(sample.linear_calls);
                    row.residual_ms.push(sample.residual_ms);
                    row.jacobian_ms.push(sample.jacobian_ms);
                    row.linear_ms.push(sample.linear_ms);
                    row.accepted_steps.push(sample.accepted_steps);
                    row.rejected_steps.push(sample.rejected_steps);
                }
                Err(err) => row.record_failure(err),
            }
        }
        if row.runs_ok > 0 {
            if let Some((mean_drift, _, _, _)) = row.final_diff.summary() {
                if mean_drift > 1.0e-4 {
                    row.record_failure(format!("trajectory_drift too large: {mean_drift:e}"));
                }
            }
        }
        rows.push(row);
    }

    println!(
        "[LSODE2 story] three-body problem backend dashboard; all time columns are milliseconds"
    );
    println!(
        "note: the example physics checks are preserved on every successful solve (energy and center-of-mass invariants)"
    );
    println!(
        "matrix | route            | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | trajectory_drift mean+/-std | status"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
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
        let drift = row
            .final_diff
            .summary()
            .map(|(m, s, _, _)| format!("{m:.2e}+/-{s:.1e}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<6} | {:<16} | {:>7} | {:<31} | {:<21} | {:<19} | {:<21} | {}",
            row.matrix,
            row.route,
            format!("{}/{}", row.runs_ok, row.runs_total),
            total,
            prepare,
            solve,
            drift,
            row.status_label()
        );
    }

    println!(
        "[LSODE2 story] three-body problem chunking-plan diagnostics; chunk counts are derived from the selected strategy and the current problem size"
    );
    println!(
        "matrix | route            | workers | residual_chunks | jacobian_chunks | sparse_chunks | residual_strategy | jacobian_strategy | sparse_strategy"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let Some(cfg) = three_body_story_config(
            row.matrix,
            row.route,
            "target/lsode2-three-body-story/diagnostic",
        ) else {
            println!(
                "{:<6} | {:<16} | {:>7} | {:<15} | {:<15} | {:<13} | {:<16} | {:<16} | {:<15}",
                row.matrix, row.route, "-", "-", "-", "-", "-", "-", "-"
            );
            continue;
        };
        let generated = &cfg.backend.generated_backend;
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let residual_outputs = cfg.eq_system.len().max(1);
        let jacobian_rows = residual_outputs;
        let residual_chunks =
            story_residual_chunk_count(residual_outputs, generated.aot_options.residual_strategy);
        let jacobian_chunks = story_dense_jacobian_chunk_count(
            jacobian_rows,
            generated.aot_options.jacobian_strategy,
        );
        let sparse_chunks =
            story_sparse_chunk_count(jacobian_rows, generated.sparse_jacobian_chunking_strategy);
        let residual_work_per_chunk = residual_outputs.div_ceil(residual_chunks.max(1)).max(1);
        let jacobian_work_per_chunk = jacobian_rows.div_ceil(jacobian_chunks.max(1)).max(1);
        let sparse_work_per_chunk = jacobian_rows.div_ceil(sparse_chunks.max(1)).max(1);
        println!(
            "{:<6} | {:<16} | {:>7} | {:<15} | {:<15} | {:<13} | {:<18} | {:<18} | {:<17}",
            row.matrix,
            row.route,
            workers,
            residual_chunks,
            jacobian_chunks,
            sparse_chunks,
            format!(
                "{} work/chunk={}",
                format!("{:?}", generated.aot_options.residual_strategy),
                residual_work_per_chunk
            ),
            format!(
                "{} work/chunk={}",
                format!("{:?}", generated.aot_options.jacobian_strategy),
                jacobian_work_per_chunk
            ),
            format!(
                "{} work/chunk={}",
                format!("{:?}", generated.sparse_jacobian_chunking_strategy),
                sparse_work_per_chunk
            ),
        );
    }

    println!(
        "[LSODE2 story] three-body problem stage diagnostics; all time columns are milliseconds"
    );
    println!(
        "note: counter_scope makes residual/jacobian semantics explicit: bridge_bdf_callbacks are BDF-level callback evaluations; native_faithful_inner_loop are faithful native nonlinear inner-loop evaluations"
    );
    println!(
        "matrix | route            | counter_scope                 | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | accepted_steps | rejected_steps"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let fmt = |stats: &RaceStats| {
            stats
                .summary()
                .map(|(m, s, _, _)| format!("{m:.2}+/-{s:.2}"))
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<6} | {:<16} | {:<29} | {:<14} | {:<14} | {:<12} | {:<11} | {:<11} | {:<9} | {:<14} | {:<14}",
            row.matrix,
            row.route,
            row.counter_scope.unwrap_or("-"),
            fmt(&row.residual_calls),
            fmt(&row.jacobian_calls),
            fmt(&row.nlu_or_native_linear),
            fmt(&row.residual_ms),
            fmt(&row.jacobian_ms),
            fmt(&row.linear_ms),
            fmt(&row.accepted_steps),
            fmt(&row.rejected_steps),
        );
    }

    assert!(
        rows.iter().any(|row| row.runs_ok > 0),
        "at least one three-body story route should complete successfully"
    );
    for row in &rows {
        if row.runs_ok > 0 {
            let (mean_diff, _, _, _) = row
                .final_diff
                .summary()
                .expect("successful three-body route should have diff samples");
            if mean_diff > 1.0e-4 {
                eprintln!(
                    "[LSODE2 three-body] diagnostic warning: {} {} final_diff={:e}",
                    row.matrix, row.route, mean_diff
                );
            }
        }
    }
}
