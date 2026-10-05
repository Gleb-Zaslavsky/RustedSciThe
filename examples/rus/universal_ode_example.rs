use RustedSciThe::numerical::ODE_api2::{NonStiffMethod, SolverType, UniversalODESolver};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // Russischsprachige companion example: y' = -y, y(0) = 1.
    // The public Radau API is demonstrated by the dedicated Radau examples.
    let equation = Expr::parse_expression("-y");
    let equations = vec![equation];
    let variables = vec!["y".to_string()];
    let argument = "t".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0]);
    let t_bound = 1.0;
    let expected = (-1.0_f64).exp();

    println!("=== Universal ODE solver examples ===\n");

    println!("1. RK45:");
    let mut rk45 = UniversalODESolver::rk45(
        equations.clone(),
        variables.clone(),
        argument.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-4,
    );
    rk45.solve();
    print_result(&rk45, expected);

    println!("2. DOPRI:");
    let mut dopri = UniversalODESolver::dopri(
        equations.clone(),
        variables.clone(),
        argument.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-4,
    );
    dopri.solve();
    print_result(&dopri, expected);

    println!("3. Radau:");
    let mut radau = UniversalODESolver::radau(
        equations.clone(),
        variables.clone(),
        argument.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-6,
        50,
        Some(1e-3),
    );
    radau.solve();
    print_result(&radau, expected);

    println!("4. BDF:");
    let mut bdf = UniversalODESolver::bdf(
        equations.clone(),
        variables.clone(),
        argument.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-3,
        1e-5,
        1e-5,
    );
    bdf.solve();
    print_result(&bdf, expected);

    println!("5. Backward Euler:");
    let mut backward_euler = UniversalODESolver::backward_euler(
        equations.clone(),
        variables.clone(),
        argument.clone(),
        t0,
        y0.clone(),
        t_bound,
        1e-6,
        50,
        Some(1e-3),
    );
    backward_euler.solve();
    print_result(&backward_euler, expected);

    println!("6. Generic constructor:");
    let mut custom = UniversalODESolver::new(
        equations,
        variables,
        argument,
        SolverType::NonStiff(NonStiffMethod::Rk45),
        t0,
        y0,
        t_bound,
    );
    custom.set_step_size(5e-5);
    custom.solve();
    print_result(&custom, expected);
}

fn print_result(solver: &UniversalODESolver, expected: f64) {
    let (times, values) = solver.get_result();
    if let (Some(times), Some(values)) = (times, values) {
        let final_value = values[(values.nrows() - 1, 0)];
        println!(
            "   final={final_value:.6}, expected={expected:.6}, error={:.2e}, steps={}\n",
            (final_value - expected).abs(),
            times.len()
        );
    }
}
