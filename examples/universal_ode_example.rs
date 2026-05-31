use RustedSciThe::numerical::ODE_api2::{SolverType, UniversalODESolver};
use RustedSciThe::numerical::Radau::Radau_main::RadauOrder;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // Define the ODE system: y' = -y, y(0) = 1
    // Exact solution: y(t) = exp(-t)
    let eq1 = Expr::parse_expression("-y");
    let eq_system = vec![eq1];
    let values = vec!["y".to_string()];
    let arg = "t".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0]);
    let t_bound = 1.0;

    println!("=== Universal ODE Solver Examples ===\n");

    // Example 1: RK45 solver
    println!("1. Using RK45 solver:");
    let mut rk45_solver = UniversalODESolver::rk45(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-4, // step size
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

    // Example 2: DOPRI solver
    println!("2. Using DOPRI solver:");
    let mut dopri_solver = UniversalODESolver::dopri(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-4, // step size
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

    // Example 3: Radau solver
    println!("3. Using Radau solver (Order 3):");
    let mut radau_solver = UniversalODESolver::radau(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        RadauOrder::Order3,
        t0,
        y0.clone(),
        t_bound,
        1e-6,       // tolerance
        50,         // max iterations
        Some(1e-3), // step size
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

    // Example 4: BDF solver
    println!("4. Using BDF solver:");
    let mut bdf_solver = UniversalODESolver::bdf(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-3, // max step
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

    // Example 5: Backward Euler solver
    println!("5. Using Backward Euler solver:");
    let mut be_solver = UniversalODESolver::backward_euler(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-6,       // tolerance
        50,         // max iterations
        Some(1e-3), // step size
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

    // Example 6: Using the generic constructor with custom parameters
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

    // Set custom parameters
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
