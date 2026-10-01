//! Руководство по жизненному циклу AOT-артефакта для нелинейной системы.
//!
//! Сначала пример выполняет холодную подготовку `BuildIfMissing`, затем
//! использует полученный resolver в строгом режиме `RequirePrebuilt` и меняет
//! значения параметров без пересборки символьной задачи.
//!
//! Запуск:
//!
//! ```text
//! cargo run --example rus_nonlinear_aot_lifecycle_guide
//! ```

use RustedSciThe::numerical::Nonlinear_systems::prelude::*;
use nalgebra::DVector;

fn problem_options() -> SymbolicProblemOptions {
    SymbolicProblemOptions::new()
        .with_variables(vec!["x".into(), "y".into()])
        .with_equation_parameters(vec!["a".into(), "b".into()])
}

fn solve_options() -> SolveOptions {
    SolveOptions {
        tolerance: 1e-11,
        max_iterations: 50,
        diagnostics: DiagnosticsOptions {
            collect_statistics: true,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn solve_bound(
    problem: &PreparedSymbolicNonlinearProblem,
    parameters: [f64; 2],
    initial: [f64; 2],
) -> Result<(), SolveError> {
    let bound = problem.bind_values(DVector::from_vec(parameters.to_vec()))?;
    let result = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()).solve(
        &bound,
        DVector::from_vec(initial.to_vec()),
        solve_options(),
    )?;
    println!(
        "parameters={parameters:?}; solution=[{:.8}, {:.8}]; residual_norm={:.3e}; attempts={}",
        result.x[0],
        result.x[1],
        result.residual_norm,
        result.statistics.attempts.len(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let equations = vec!["x^2 - a".to_string(), "y - b*x".to_string()];
    let run_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let output_dir = std::env::temp_dir().join(format!(
        "rustedscithe-nonlinear-aot-guide-{}-{run_id}",
        std::process::id(),
    ));

    println!("=== Руководство по жизненному циклу нелинейного AOT ===");
    println!("каталог артефакта: {}", output_dir.display());

    // Холодный путь: материализация и сборка совместимого артефакта.
    let cold = SymbolicNonlinearProblem::from_strings_with_generated_backend(
        equations.clone(),
        problem_options(),
        SymbolicGeneratedBackendConfig::build_if_missing_release(&output_dir),
    )?;
    println!(
        "cold: backend={:?}; action={:?}; key={:?}; build={:?}",
        cold.selected_backend,
        cold.preparation_report.artifact_action,
        cold.preparation_report.artifact_key,
        cold.preparation_report.build_duration,
    );
    let resolver = cold.updated_resolver.clone();
    let prepared = cold.into_prepared();
    solve_bound(&prepared, [4.0, 3.0], [1.0, 3.0])?;

    // Теплый строгий путь: второй compile запрещен и не требуется.
    let warm = SymbolicNonlinearProblem::from_strings_with_generated_backend(
        equations,
        problem_options(),
        SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(resolver),
    )?;
    assert!(warm.build_result.is_none());
    assert_eq!(
        warm.preparation_report.artifact_action,
        SymbolicArtifactAction::Reused
    );
    println!(
        "warm: backend={:?}; action={:?}; key={:?}; build={:?}",
        warm.selected_backend,
        warm.preparation_report.artifact_action,
        warm.preparation_report.artifact_key,
        warm.preparation_report.build_duration,
    );
    let prepared = warm.into_prepared();
    solve_bound(&prepared, [9.0, 2.0], [2.0, 4.0])?;

    // Каталог можно удалить после завершения процесса; Windows может
    // удерживать загруженную DLL до самого выхода из примера.
    Ok(())
}
