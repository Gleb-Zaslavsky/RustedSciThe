//! Русский пример parameter continuation для новой архитектуры `BVP_sci`.
//!
//! Подготовленный числовой plan сохраняется, а значения параметра меняются
//! через `set_parameters`. Поэтому повторный solve не повторяет символьную
//! подготовку. Задача имеет точный ответ `p = 1` и `y(x) = x`.

use RustedSciThe::numerical::BVP_sci::{
    BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan, BvpSciMatrixLayout,
    BvpSciNumericalPlan, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let telemetry = BvpSciTelemetry::counters();
    let plan = BvpSciNumericalPlan::new(
        1,
        1,
        |_x, _state, parameters, output| {
            output[0] = parameters[0];
            Ok(())
        },
        telemetry.clone(),
    )?;
    let plan = BvpSciLambdifyPlan::prepare_numerical(plan);
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, _yb, parameters, output| {
            output[0] = ya[0];
            output[1] = parameters[0] - 1.0;
            Ok(())
        },
        telemetry,
    );
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Numerical)
        .with_matrix_layout(BvpSciMatrixLayout::Dense)
        .with_tolerance(1e-7);
    let mut solver = BvpSciSolver::new(
        plan,
        boundary,
        vec![0.0, 0.5, 1.0],
        vec![0.0, 0.25, 0.5],
        vec![0.25],
        options,
    )?;

    println!("BVP_sci numerical continuation");
    println!("parameter | y(0) | y(1) | residual_norm | continuation_solves");
    for parameter in [0.25, 1.5] {
        if parameter != 0.25 {
            // Здесь задается новый элемент семейства BVP, а не повторное
            // получение p=1 при неизменном условии p-1=0. Подготовка frontend
            // переиспользуется, численная аппроксимация строится заново.
            // Меняется только численное состояние; подготовленный plan остаётся.
            solver.set_parameters(vec![parameter])?;
        }
        let solution = solver.solve()?;
        let snapshot = solver.plan().telemetry_snapshot();
        println!(
            "{:.6} | {:.6} | {:.6} | {:.3e} | {}",
            solution.parameters[0],
            solution.y[0],
            solution.y[solution.y.len() - 1],
            solution.residual_norm,
            snapshot.continuation_solves,
        );
    }
    Ok(())
}
