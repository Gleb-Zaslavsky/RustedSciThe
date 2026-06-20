use RustedSciThe::numerical::ODE_api2::{SolverType, UniversalODESolver};
use RustedSciThe::numerical::Radau::Radau_main::RadauOrder;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // Задаём ODE-систему: y' = -y, y(0) = 1
    // Точное решение: y(t) = exp(-t)
    let eq1 = Expr::parse_expression("-y");
    let eq_system = vec![eq1];
    let values = vec!["y".to_string()];
    let arg = "t".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0]);
    let t_bound = 1.0;

    println!("=== Universal ODE Solver Examples ===\n");

    // Пример 1: солвер RK45
    println!("1. Using RK45 solver:");
    let mut rk45_solver = UniversalODESolver::rk45(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-4, // шаг интегрирования
    );

    rk45_solver.solve();
    let (t_result, y_result) = rk45_solver.get_result();

    if let (Some(_t_res), Some(y_res)) = (t_result, y_result) {
        let final_value = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        println!("   Final value: {:.6}", final_value);
        println!("   Expected: {:.6}", expected);
        println!("   Error: {:.2e}\n", (final_value - expected).abs());
    }

    // Пример 2: солвер DOPRI
    println!("2. Using DOPRI solver:");
    let mut dopri_solver = UniversalODESolver::dopri(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-4, // шаг интегрирования
    );

    dopri_solver.solve();
    let (t_result, y_result) = dopri_solver.get_result();

    if let (Some(_t_res), Some(y_res)) = (t_result, y_result) {
        let final_value = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        println!("   Final value: {:.6}", final_value);
        println!("   Expected: {:.6}", expected);
        println!("   Error: {:.2e}\n", (final_value - expected).abs());
    }

    // Пример 3: солвер Radau
    println!("3. Using Radau solver (Order 3):");
    let mut radau_solver = UniversalODESolver::radau(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        RadauOrder::Order3,
        t0,
        y0.clone(),
        t_bound,
        1e-6,       // допуск
        50,         // максимум итераций
        Some(1e-3), // шаг интегрирования
    );

    radau_solver.solve();
    let (t_result, y_result) = radau_solver.get_result();

    if let (Some(_t_res), Some(y_res)) = (t_result, y_result) {
        let final_value = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        println!("   Final value: {:.6}", final_value);
        println!("   Expected: {:.6}", expected);
        println!("   Error: {:.2e}\n", (final_value - expected).abs());
    }

    // Пример 4: солвер BDF
    println!("4. Using BDF solver:");
    let mut bdf_solver = UniversalODESolver::bdf(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-3, // максимальный шаг
        1e-5, // rtol
        1e-5, // atol
    );

    bdf_solver.solve();
    let (t_result, y_result) = bdf_solver.get_result();

    if let (Some(_t_res), Some(y_res)) = (t_result, y_result) {
        let final_value = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        println!("   Final value: {:.6}", final_value);
        println!("   Expected: {:.6}", expected);
        println!("   Error: {:.2e}\n", (final_value - expected).abs());
    }

    // Пример 5: солвер Backward Euler
    println!("5. Using Backward Euler solver:");
    let mut be_solver = UniversalODESolver::backward_euler(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-6,       // допуск
        50,         // максимум итераций
        Some(1e-3), // шаг интегрирования
    );

    be_solver.solve();
    let (t_result, y_result) = be_solver.get_result();

    if let (Some(_t_res), Some(y_res)) = (t_result, y_result) {
        let final_value = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        println!("   Final value: {:.6}", final_value);
        println!("   Expected: {:.6}", expected);
        println!("   Error: {:.2e}\n", (final_value - expected).abs());
    }

    // Пример 6: используем generic-конструктор с пользовательскими параметрами
    println!("6. Using generic constructor with custom parameters:");
    let mut custom_solver = UniversalODESolver::new(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        SolverType::NonStiff("RK45".to_string()),
        t0,
        y0.clone(),
        t_bound,
    );

    // Задаём пользовательские параметры
    custom_solver.set_step_size(5e-5);

    custom_solver.solve();
    let (t_result, y_result) = custom_solver.get_result();

    if let (Some(t_res), Some(y_res)) = (t_result, y_result) {
        let final_value = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        println!("   Final value: {:.6}", final_value);
        println!("   Expected: {:.6}", expected);
        println!("   Error: {:.2e}", (final_value - expected).abs());
        println!("   Number of steps: {}", t_res.len());
    }

    println!("\n=== All examples completed ===");
}
