use RustedSciThe::numerical::BDF::BDF_api::ODEsolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::collections::HashMap;

fn main() {
    println!("=== BDF Solver Examples ===\n");

    // Example 1: Riccati Equation
    println!("1. Riccati Equation: y' = y² - t²");
    riccati_equation();

    // Example 2: Van der Pol Oscillator
    println!("\n2. Van der Pol Oscillator (Stiff System)");
    van_der_pol_oscillator();

    // Example 3: Logistic Equation with Analytical Comparison
    println!("\n3. Logistic Equation with Analytical Solution");
    logistic_equation();

    // Example 4: Nonlinear Pendulum
    println!("\n4. Nonlinear Pendulum (Energy Conservation)");
    nonlinear_pendulum();

    // Example 5: Lorenz System (Chaotic Dynamics)
    println!("\n5. Lorenz System (Chaotic Attractor)");
    lorenz_system();

    // Example 6: Stiff Chemical Reaction
    println!("\n6. Stiff Chemical Kinetics A→B→C");
    stiff_chemical_reaction();

    // Example 7: Stop Conditions
    println!("\n7. Stop Conditions Example");
    stop_condition_example();
}

fn riccati_equation() {
    // Riccati equation: y' = y² - t², y(0) = 1
    let eq1 = Expr::parse_expression("y*y - t*t");
    let eq_system = vec![eq1];
    let values = vec!["y".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0]);
    let t_bound = 0.5;
    let max_step = 0.001;
    let rtol = 1e-8;
    let atol = 1e-10;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    println!("Status: {}", solver.get_status());

    let (t_result, y_result) = solver.get_result();
    println!("Initial value: y(0) = {}", y_result[(0, 0)]);
    println!("Final value: y({}) = {}", t_bound, y_result[(y_result.nrows() - 1, 0)]);
    println!("Solution points: {}", t_result.len());
}

fn van_der_pol_oscillator() {
    // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1²)y2 - y1 with μ = 5 (stiff)
    let eq1 = Expr::parse_expression("y2");
    let eq2 = Expr::parse_expression("5*(1-y1*y1)*y2 - y1");
    let eq_system = vec![eq1, eq2];
    let values = vec!["y1".to_string(), "y2".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![2.0, 0.0]);
    let t_bound = 5.0;
    let max_step = 0.01;
    let rtol = 1e-6;
    let atol = 1e-8;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    println!("Status: {}", solver.get_status());

    let (_, y_result) = solver.get_result();
    println!("Initial: y1(0) = {}, y2(0) = {}", y_result[(0, 0)], y_result[(0, 1)]);
    println!("Final: y1({}) = {}, y2({}) = {}", 
             t_bound, y_result[(y_result.nrows() - 1, 0)], 
             t_bound, y_result[(y_result.nrows() - 1, 1)]);
    println!("Van der Pol exhibits limit cycle behavior (bounded oscillation)");
}

fn logistic_equation() {
    // Logistic equation: y' = r*y*(1-y/K) with r=2, K=10, y(0)=1
    // Analytical solution: y = K*y0*exp(r*t)/(K + y0*(exp(r*t) - 1))
    let eq1 = Expr::parse_expression("2*y*(1-y/10)");
    let eq_system = vec![eq1];
    let values = vec!["y".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0]);
    let t_bound = 3.0;
    let max_step = 0.01;
    let rtol = 1e-8;
    let atol = 1e-10;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    println!("Status: {}", solver.get_status());

    let (t_result, y_result) = solver.get_result();

    // Compare with analytical solution at final time
    let r = 2.0;
    let k = 10.0;
    let y0_val = 1.0;
    let t_final = t_result[t_result.len() - 1];
    let y_analytical = k * y0_val * (r * t_final).exp() / (k + y0_val * ((r * t_final).exp() - 1.0));
    let y_numerical = y_result[(y_result.nrows() - 1, 0)];

    println!("Final time: {}", t_final);
    println!("Numerical solution: {}", y_numerical);
    println!("Analytical solution: {}", y_analytical);
    println!("Error: {}", (y_numerical - y_analytical).abs());
    println!("Logistic growth approaches carrying capacity K = {}", k);
}

fn nonlinear_pendulum() {
    // Nonlinear pendulum: θ'' + sin(θ) = 0
    // Rewritten as system: θ' = ω, ω' = -sin(θ)
    let eq1 = Expr::parse_expression("omega");
    let eq2 = Expr::parse_expression("-sin(theta)");
    let eq_system = vec![eq1, eq2];
    let values = vec!["theta".to_string(), "omega".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0, 0.0]); // θ(0)=1, ω(0)=0
    let t_bound = 1.0;
    let max_step = 0.001;
    let rtol = 1e-6;
    let atol = 1e-8;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    println!("Status: {}", solver.get_status());

    let (_, y_result) = solver.get_result();

    // Check energy conservation: E = 0.5*ω² - cos(θ) + cos(θ₀)
    let initial_theta = y_result[(0, 0)];
    let initial_omega = y_result[(0, 1)];
    let final_theta = y_result[(y_result.nrows() - 1, 0)];
    let final_omega = y_result[(y_result.nrows() - 1, 1)];

    let initial_energy = 0.5 * initial_omega.powi(2) - initial_theta.cos();
    let final_energy = 0.5 * final_omega.powi(2) - final_theta.cos();

    println!("Initial: θ = {:.4}, ω = {:.4}", initial_theta, initial_omega);
    println!("Final: θ = {:.4}, ω = {:.4}", final_theta, final_omega);
    println!("Initial energy: {:.6}", initial_energy);
    println!("Final energy: {:.6}", final_energy);
    println!("Energy conservation error: {:.2e}", (final_energy - initial_energy).abs());
}

fn lorenz_system() {
    // Lorenz system: x' = σ(y-x), y' = x(ρ-z)-y, z' = xy-βz
    // σ=10, ρ=28, β=8/3 (classic chaotic parameters)
    let eq1 = Expr::parse_expression("10*(y-x)");
    let eq2 = Expr::parse_expression("x*(28-z)-y");
    let eq3 = Expr::parse_expression("x*y-8*z/3");
    let eq_system = vec![eq1, eq2, eq3];
    let values = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
    let t_bound = 5.0;
    let max_step = 0.001;
    let rtol = 1e-8;
    let atol = 1e-10;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    println!("Status: {}", solver.get_status());

    let (_, y_result) = solver.get_result();

    println!("Initial: x={:.2}, y={:.2}, z={:.2}", 
             y_result[(0, 0)], y_result[(0, 1)], y_result[(0, 2)]);
    println!("Final: x={:.2}, y={:.2}, z={:.2}", 
             y_result[(y_result.nrows() - 1, 0)], 
             y_result[(y_result.nrows() - 1, 1)], 
             y_result[(y_result.nrows() - 1, 2)]);

    // Check bounds for chaotic attractor
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;

    for i in 0..y_result.nrows() {
        x_min = x_min.min(y_result[(i, 0)]);
        x_max = x_max.max(y_result[(i, 0)]);
        z_min = z_min.min(y_result[(i, 2)]);
        z_max = z_max.max(y_result[(i, 2)]);
    }

    println!("Chaotic attractor bounds:");
    println!("x ∈ [{:.2}, {:.2}]", x_min, x_max);
    println!("z ∈ [{:.2}, {:.2}]", z_min, z_max);
    println!("Solution exhibits bounded chaotic behavior");
}

fn stiff_chemical_reaction() {
    // Stiff chemical kinetics: A → B → C
    // y1' = -k1*y1, y2' = k1*y1 - k2*y2, y3' = k2*y2
    // with k1 = 1, k2 = 1000 (very stiff!)
    let eq1 = Expr::parse_expression("-y1");
    let eq2 = Expr::parse_expression("y1 - 1000*y2");
    let eq3 = Expr::parse_expression("1000*y2");
    let eq_system = vec![eq1, eq2, eq3];
    let values = vec!["y1".to_string(), "y2".to_string(), "y3".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0, 0.0, 0.0]); // All A initially
    let t_bound = 2.0;
    let max_step = 0.01;
    let rtol = 1e-6;
    let atol = 1e-8;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    solver.solve();
    println!("Status: {}", solver.get_status());

    let (t_result, y_result) = solver.get_result();

    // Check mass conservation: y1 + y2 + y3 = 1
    let final_sum = y_result[(y_result.nrows() - 1, 0)]
        + y_result[(y_result.nrows() - 1, 1)]
        + y_result[(y_result.nrows() - 1, 2)];

    println!("Initial concentrations: A={:.3}, B={:.3}, C={:.3}", 
             y_result[(0, 0)], y_result[(0, 1)], y_result[(0, 2)]);
    println!("Final concentrations: A={:.3}, B={:.3}, C={:.3}", 
             y_result[(y_result.nrows() - 1, 0)], 
             y_result[(y_result.nrows() - 1, 1)], 
             y_result[(y_result.nrows() - 1, 2)]);
    println!("Mass conservation: {:.6} (should be 1.0)", final_sum);
    println!("Conservation error: {:.2e}", (final_sum - 1.0).abs());

    // Compare A with analytical solution: y1 = exp(-t)
    let t_final = t_result[t_result.len() - 1];
    let y1_analytical = (-t_final).exp();
    let y1_numerical = y_result[(y_result.nrows() - 1, 0)];
    println!("A analytical: {:.6}, numerical: {:.6}, error: {:.2e}", 
             y1_analytical, y1_numerical, (y1_numerical - y1_analytical).abs());
}

fn stop_condition_example() {
    // Simple exponential growth: y' = y, y(0) = 1
    // Stop when y reaches 2.0
    let eq1 = Expr::parse_expression("y");
    let eq_system = vec![eq1];
    let values = vec!["y".to_string()];
    let arg = "t".to_string();
    let method = "BDF".to_string();
    let t0 = 0.0;
    let y0 = DVector::from_vec(vec![1.0]);
    let t_bound = 10.0; // Would normally integrate to t=10
    let max_step = 0.01;
    let rtol = 1e-6;
    let atol = 1e-3;

    let mut solver = ODEsolver::new(
        eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false, None,
    );

    // Set stop condition: stop when y reaches 2.0
    let mut stop_condition = HashMap::new();
    stop_condition.insert("y".to_string(), 2.0);
    solver.set_stop_condition(stop_condition);

    solver.solve();
    println!("Status: {}", solver.get_status());

    let (t_result, y_result) = solver.get_result();
    let final_t = t_result[t_result.len() - 1];
    let final_y = y_result[(y_result.nrows() - 1, 0)];

    println!("Integration stopped at t = {:.4} (instead of t = {})", final_t, t_bound);
    println!("Final value: y = {:.4} (target was 2.0)", final_y);
    println!("Analytical stop time: t = ln(2) = {:.4}", 2.0_f64.ln());
    println!("Error in stop time: {:.2e}", (final_t - 2.0_f64.ln()).abs());

    // Demonstrate multiple variable stop condition
    println!("\nMultiple variable stop condition example:");
    
    // Harmonic oscillator: y1' = y2, y2' = -y1
    let eq1 = Expr::parse_expression("y2");
    let eq2 = Expr::parse_expression("-y1");
    let eq_system = vec![eq1, eq2];
    let values = vec!["y1".to_string(), "y2".to_string()];
    
    let mut solver = ODEsolver::new(
        eq_system, values, "t".to_string(), "BDF".to_string(),
        0.0, DVector::from_vec(vec![1.0, 0.0]), 10.0,
        0.01, 1e-6, 1e-3, None, false, None,
    );

    // Stop when y1 crosses zero (first time)
    let mut stop_condition = HashMap::new();
    stop_condition.insert("y1".to_string(), 0.0);
    solver.set_stop_condition(stop_condition);

    solver.solve();
    let (t_result, y_result) = solver.get_result();
    let final_t = t_result[t_result.len() - 1];
    let final_y1 = y_result[(y_result.nrows() - 1, 0)];

    println!("Harmonic oscillator stopped at t = {:.4}", final_t);
    println!("y1 = {:.6} (target was 0.0)", final_y1);
    println!("Analytical crossing time: t = π/2 = {:.4}", std::f64::consts::PI / 2.0);
}