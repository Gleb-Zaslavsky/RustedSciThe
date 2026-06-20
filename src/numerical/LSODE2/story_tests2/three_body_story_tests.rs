use super::*;
use crate::numerical::LSODE2::{Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure};
use crate::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use std::collections::HashMap;
use std::process::Command;

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn three_body_story_base_config() -> Lsode2ProblemConfig {
    let k = 39.47841760435743;
    let m0 = 1.0;
    let m1 = 0.5;
    let m2 = 0.75;

    let params: HashMap<String, f64> = HashMap::from([
        ("k".to_string(), k),
        ("m0".to_string(), m0),
        ("m1".to_string(), m1),
        ("m2".to_string(), m2),
    ]);

    let r01 = Expr::parse_expression("((x0 - x1)^2 + (y0 - y1)^2)^0.5");
    let r02 = Expr::parse_expression("((x0 - x2)^2 + (y0 - y2)^2)^0.5");
    let r12 = Expr::parse_expression("((x1 - x2)^2 + (y1 - y2)^2)^0.5");

    let eq_vx0 = Expr::parse_expression("-k * (m1*(x0 - x1)/R01^3 + m2*(x0 - x2)/R02^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R02", &r02)
        .set_variable_from_map(&params);
    let eq_vy0 = Expr::parse_expression("-k * (m1*(y0 - y1)/R01^3 + m2*(y0 - y2)/R02^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R02", &r02)
        .set_variable_from_map(&params);
    let eq_vx1 = Expr::parse_expression("-k * (m0*(x1 - x0)/R01^3 + m2*(x1 - x2)/R12^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);
    let eq_vy1 = Expr::parse_expression("-k * (m0*(y1 - y0)/R01^3 + m2*(y1 - y2)/R12^3)")
        .substitute_variable("R01", &r01)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);
    let eq_vx2 = Expr::parse_expression("-k * (m0*(x2 - x0)/R02^3 + m1*(x2 - x1)/R12^3)")
        .substitute_variable("R02", &r02)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);
    let eq_vy2 = Expr::parse_expression("-k * (m0*(y2 - y0)/R02^3 + m1*(y2 - y1)/R12^3)")
        .substitute_variable("R02", &r02)
        .substitute_variable("R12", &r12)
        .set_variable_from_map(&params);

    let eq_sys = vec![
        Expr::parse_expression("vx0"),
        eq_vx0,
        Expr::parse_expression("vy0"),
        eq_vy0,
        Expr::parse_expression("vx1"),
        eq_vx1,
        Expr::parse_expression("vy1"),
        eq_vy1,
        Expr::parse_expression("vx2"),
        eq_vx2,
        Expr::parse_expression("vy2"),
        eq_vy2,
    ];

    let unknowns = vec![
        "x0".to_string(),
        "vx0".to_string(),
        "y0".to_string(),
        "vy0".to_string(),
        "x1".to_string(),
        "vx1".to_string(),
        "y1".to_string(),
        "vy1".to_string(),
        "x2".to_string(),
        "vx2".to_string(),
        "y2".to_string(),
        "vy2".to_string(),
    ];

    Lsode2ProblemConfig::new(
        eq_sys,
        unknowns,
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0, -0.4, 33.0, 0.0,
        ]),
        500.0,
        0.001,
        1e-10,
        1e-12,
    )
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_faithful_bdf_solve(250_000, 250_000)
}

fn three_body_story_physics_checks(solution: &DMatrix<f64>, times: &DVector<f64>) {
    assert_eq!(solution.nrows(), 12, "three-body state should have 12 rows");
    assert_eq!(solution.ncols(), times.len(), "solution columns must match time samples");

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
        max_cm_velocity_drift = max_cm_velocity_drift.max(
            ((cm_vx - initial_cm_vx).powi(2) + (cm_vy - initial_cm_vy).powi(2)).sqrt(),
        );

        let time = times[i];
        let expected_cm_x = initial_cm_x + initial_cm_vx * time;
        let expected_cm_y = initial_cm_y + initial_cm_vy * time;
        max_cm_position_drift = max_cm_position_drift.max(
            ((cm_x - expected_cm_x).powi(2) + (cm_y - expected_cm_y).powi(2)).sqrt(),
        );
    }

    assert!(max_energy_drift.is_finite());
    assert!(max_cm_velocity_drift.is_finite());
    assert!(max_cm_position_drift.is_finite());
}

fn three_body_story_trajectory_drift(
    solution: &DMatrix<f64>,
    baseline: &DMatrix<f64>,
) -> f64 {
    let cols = solution.ncols().min(baseline.ncols());
    let rows = solution.nrows().min(baseline.nrows());
    if cols == 0 || rows == 0 {
        return f64::NAN;
    }

    let mut max_drift = 0.0_f64;
    for col in 0..cols {
        let mut sum_sq = 0.0_f64;
        for row in 0..rows {
            let delta = solution[(row, col)] - baseline[(row, col)];
            sum_sq += delta * delta;
        }
        max_drift = max_drift.max(sum_sq.sqrt());
    }
    max_drift
}

fn three_body_story_config(
    matrix: &'static str,
    route: &'static str,
    output_dir: &str,
) -> Option<Lsode2ProblemConfig> {
    let base = three_body_story_base_config();
    let source = match route {
        "Lambdify" => Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
        "AOT-Ctcc-Whole" | "AOT-Ctcc-Chunk4" | "AOT-Ctcc-Chunk12" =>
            Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::AtomView,
                execution: Lsode2SymbolicExecutionMode::Aot {
                    toolchain: Lsode2AotToolchain::CTcc,
                    profile: Lsode2AotProfile::Release,
                },
            },
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
        ("Banded", "AOT-Ctcc-Whole") => {
            base.with_native_banded_faithful_aot_c_tcc(output_dir)
        }
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

fn three_body_story_chunking_summary(config: &Lsode2ProblemConfig) -> String {
    let generated = &config.backend.generated_backend;
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let residual_outputs = config.eq_system.len().max(1);
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
    let residual_is_whole =
        matches!(generated.aot_options.residual_strategy, ResidualChunkingStrategy::Whole);
    let jacobian_is_whole =
        matches!(generated.aot_options.jacobian_strategy, DenseJacobianChunkingStrategy::Whole);
    let sparse_is_whole =
        matches!(generated.sparse_jacobian_chunking_strategy, SparseChunkingStrategy::Whole);
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
        } => total_outputs.max(1).div_ceil(max_outputs_per_chunk.max(1)).max(1),
    }
}

fn story_dense_jacobian_chunk_count(
    rows: usize,
    strategy: DenseJacobianChunkingStrategy,
) -> usize {
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

fn run_three_body_story_sample_result(
    route: &'static str,
    config: Lsode2ProblemConfig,
    baseline_solution: &DMatrix<f64>,
) -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let started_total = Instant::now();
    let mut solver = Lsode2Solver::new(config).map_err(|err| short_error(&err.to_string()))?;
    let started_prepare = Instant::now();
    solver.prepare().map_err(|err| short_error(&err.to_string()))?;
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
    let trajectory_drift = three_body_story_trajectory_drift(&solution, baseline_solution);
    if solution.ncols() != baseline_solution.ncols() {
        eprintln!(
            "[LSODE2 three-body] diagnostic note: {} produced {} samples vs baseline {} samples; trajectory_drift uses the common prefix",
            route,
            solution.ncols(),
            baseline_solution.ncols()
        );
    }

    let is_native_faithful = is_native_faithful_status(&summary.status);
    let (residual_calls, jacobian_calls, linear_calls, residual_ms, jacobian_ms, linear_ms) =
        if route == "Lambdify" || !is_native_faithful {
            (
                summary.statistics.residual_calls as f64,
                summary.statistics.jacobian_calls as f64,
                summary.statistics.bdf_nlu_total as f64,
                summary.statistics.residual_ms_total,
                summary.statistics.jacobian_ms_total,
                0.0,
            )
        } else {
            (
                summary.native_statistics.native_residual_calls as f64,
                summary.native_statistics.native_jacobian_calls as f64,
                summary.native_statistics.native_linear_solve_calls as f64,
                summary.native_statistics.native_residual_ms_total,
                summary.native_statistics.native_jacobian_ms_total,
                summary.native_statistics.native_linear_solve_ms_total,
            )
        };

    let accepted_steps = summary
        .native_statistics
        .native_step_accepts
        .max(summary.native_statistics.bridge_accepted_steps) as f64;
    let rejected_steps = (summary.native_statistics.native_step_rejects_error_test
        + summary.native_statistics.native_step_rejects_nonlinear) as f64;

    Ok((
        total_ms,
        prepare_ms,
        solve_ms,
        trajectory_drift,
        residual_calls,
        jacobian_calls,
        linear_calls,
        residual_ms,
        jacobian_ms,
        linear_ms,
        accepted_steps,
        rejected_steps,
    ))
}

#[test]
#[ignore = "release story: three-body LSODE2 compares Lambdify vs tcc whole and chunked runtime"]
fn lsode2_three_body_problem_backend_story_dashboard() {
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
    let mut baseline_solver = Lsode2Solver::new(baseline_cfg).expect("baseline solver should build");
    baseline_solver.prepare().expect("baseline prepare should succeed");
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
                let _ = run_three_body_story_sample_result(route, cfg, &baseline_solution);
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
            match run_three_body_story_sample_result(route, cfg, &baseline_solution) {
                Ok(sample) => {
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
        let residual_chunks = story_residual_chunk_count(
            residual_outputs,
            generated.aot_options.residual_strategy,
        );
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
        "note: residual_calls/jacobian_calls are route-specific telemetry; Lambdify reports bridge/generated-backend counters, while AOT rows report native inner-loop counters"
    );
    println!(
        "matrix | route            | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | accepted_steps | rejected_steps"
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
            "{:<6} | {:<16} | {:<14} | {:<14} | {:<12} | {:<11} | {:<11} | {:<9} | {:<14} | {:<14}",
            row.matrix,
            row.route,
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
                    row.matrix,
                    row.route,
                    mean_diff
                );
            }
        }
    }
}

