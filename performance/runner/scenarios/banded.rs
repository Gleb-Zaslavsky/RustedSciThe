use crate::error::RunnerError;
use crate::registry::ScenarioOutput;
use RustedSciThe::somelinalg::banded::{
    Banded, LapackStyleBandedLuFaithful, banded_matvec, residual_linf,
};
use std::hint::black_box;
use std::time::Instant;

/// Fixed-seed, compact tridiagonal system. No dense matrix is constructed.
pub fn run_tridiagonal() -> Result<ScenarioOutput, RunnerError> {
    const N: usize = 4096;
    let mut matrix = Banded::<f64>::zeros(N, 1, 1)
        .map_err(|error| RunnerError::Scenario(format!("band allocation: {error:?}")))?;
    for row in 0..N {
        matrix[(row, row)] = 4.0;
        if row > 0 {
            matrix[(row, row - 1)] = -1.0;
        }
        if row + 1 < N {
            matrix[(row, row + 1)] = -1.0;
        }
    }
    let expected: Vec<f64> = (0..N)
        .map(|index| 1.0 + (index % 17) as f64 / 17.0)
        .collect();
    let rhs = banded_matvec(&matrix, &expected)
        .map_err(|error| RunnerError::Scenario(format!("RHS construction: {error:?}")))?;

    let mut factor = LapackStyleBandedLuFaithful::new(N, 1, 1)
        .map_err(|error| RunnerError::Scenario(format!("band factor allocation: {error:?}")))?;
    let full_solve_started = Instant::now();
    let factor_started = Instant::now();
    factor
        .factor_from(&matrix)
        .map_err(|error| RunnerError::Scenario(format!("band factorization: {error:?}")))?;
    let factor_ms = factor_started.elapsed().as_secs_f64() * 1000.0;

    let mut solution = rhs.clone();
    let solve_started = Instant::now();
    factor
        .solve_in_place(&mut solution)
        .map_err(|error| RunnerError::Scenario(format!("band solve: {error:?}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
    let scenario_wall_ms = full_solve_started.elapsed().as_secs_f64() * 1000.0;

    let residual = residual_linf(&matrix, &solution, &rhs)
        .map_err(|error| RunnerError::Scenario(format!("residual: {error:?}")))?;
    let max_diff = solution
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if residual > 1e-10 || max_diff > 1e-10 {
        return Err(RunnerError::Scenario(format!(
            "correctness gate failed: residual={residual:e}, max_diff={max_diff:e}"
        )));
    }
    black_box(&solution);

    Ok(ScenarioOutput {
        metrics: vec![
            ("factorization_ms".into(), "ms".into(), factor_ms),
            ("solve_ms".into(), "ms".into(), solve_ms),
            ("scenario_wall_ms".into(), "ms".into(), scenario_wall_ms),
        ],
        native_scopes: Vec::new(),
        counters: vec![
            ("dimension".into(), N as u64),
            ("lower_bandwidth".into(), 1),
            ("upper_bandwidth".into(), 1),
        ],
        diagnostics: vec![
            ("relative_residual".into(), "ratio".into(), residual),
            (
                "known_solution_max_diff".into(),
                "absolute".into(),
                max_diff,
            ),
        ],
    })
}
