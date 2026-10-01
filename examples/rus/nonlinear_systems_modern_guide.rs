//! Современный типизированный пример для нелинейных систем.
//!
//! Подготовка символьной системы отделена от численного решения: уравнения
//! подготавливаются один раз, затем задаются несколько наборов параметров и
//! читается диагностика каждого решения.
//!
//! Запуск:
//!
//! ```text
//! cargo run --example rus_nonlinear_systems_modern_guide
//! ```

use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    Bounds, DampedNewtonMethod, DiagnosticsOptions, NonlinearSolverMethod,
    PreparedSymbolicNonlinearProblem, SolveOptions, SymbolicProblemOptions,
};
use nalgebra::DVector;

fn solve_options(bounds: Option<Bounds>) -> SolveOptions {
    SolveOptions {
        tolerance: 1e-10,
        max_iterations: 64,
        bounds,
        diagnostics: DiagnosticsOptions {
            collect_history: false,
            collect_statistics: true,
            enable_logging: false,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Современный пример нелинейной системы ===");
    parameter_sweep()?;
    bounded_system()?;
    Ok(())
}

fn parameter_sweep() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Одна подготовка, несколько наборов параметров");
    println!("   уравнения: a*x + y - 3 = 0, x - y = 0");

    let prepared = PreparedSymbolicNonlinearProblem::from_strings(
        vec!["a*x + y - 3".to_string(), "x - y".to_string()],
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()]),
    )?;
    println!("   подготовленный backend: {:?}", prepared.backend_kind());
    println!(
        "   схема параметров: {:?}",
        prepared.parameter_schema().map(|schema| schema.names())
    );

    let method = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default());
    for (a, initial) in [(1.0, [1.0, 1.0]), (2.0, [1.0, 1.0]), (0.5, [1.0, 1.0])] {
        let bound = prepared.bind_values(DVector::from_vec(vec![a]))?;
        let result = method.clone().solve(
            &bound,
            DVector::from_vec(initial.to_vec()),
            solve_options(None),
        )?;
        println!(
            "   a={a:.2}: x={:.8}, y={:.8}, итерации={}, residual_calls={}, jacobian_calls={}, attempts={}",
            result.x[0],
            result.x[1],
            result.statistics.iterations,
            result.statistics.residual_evaluations,
            result.statistics.jacobian_evaluations,
            result.statistics.attempts.len(),
        );
        if let Some(attempt) = result.statistics.attempts.last() {
            println!(
                "      последняя попытка: iteration={}, R/J={}/{}, refresh/reuse={}/{}, factor/linear={}/{}, accepted/rejected={}/{}",
                attempt.iteration,
                attempt.residual_evaluations,
                attempt.jacobian_evaluations,
                attempt.jacobian_refreshes,
                attempt.jacobian_reuses,
                attempt.linear_factorizations,
                attempt.linear_solves,
                attempt.accepted_steps,
                attempt.rejected_steps,
            );
        }
    }
    println!("   символьная подготовка не повторяется для этих наборов параметров");
    Ok(())
}

fn bounded_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Решение с ограничениями и символьным якобианом");
    println!("   уравнения: x^2 - 4 = 0, y^2 - 1 = 0");

    let problem = PreparedSymbolicNonlinearProblem::from_strings(
        vec!["x^2 - 4".to_string(), "y^2 - 1".to_string()],
        SymbolicProblemOptions::new().with_variables(vec!["x".to_string(), "y".to_string()]),
    )?;
    let bound = problem.bind_without_parameters()?;
    let bounds = Bounds::new(vec![(0.0, 3.0), (0.0, 2.0)])?;
    let result = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()).solve(
        &bound,
        DVector::from_vec(vec![0.5, 0.5]),
        solve_options(Some(bounds)),
    )?;
    println!(
        "   решение: x={:.8}, y={:.8}; норма конечной невязки={:.3e}",
        result.x[0],
        result.x[1],
        result.residual.norm(),
    );
    println!("   ограничения задаются через SolveOptions и учитываются солвером");
    Ok(())
}
