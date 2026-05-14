//! Comprehensive guide for solving nonlinear systems of equations
//!
//! This example demonstrates how to use the LM (Levenberg-Marquardt) solver
//! to solve various nonlinear systems using both the traditional API and
//! the new builder pattern.
//!
//! The solver automatically generates analytical Jacobians from symbolic
//! expressions, providing efficient and accurate solutions.

use RustedSciThe::numerical::optimization::sym_wrapper::LM;
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== Nonlinear Systems Solver Guide ===");
    println!("This guide demonstrates both string-based and native symbolic construction\n");

    // Example 1: Basic usage with builder pattern (string equations)
    example_1_builder_string();

    // Example 2: Builder pattern with Expr objects
    example_2_builder_expr();

    // Example 3: Native symbolic construction (no string parsing)
    example_3_native_symbolic();

    // Example 4: Traditional API (for comparison)
    example_4_traditional_api();

    // Example 5: Rosenbrock function
    example_4_rosenbrock();

    // Example 6: Native symbolic exponential system
    example_6_native_exponential();

    // Example 7: Native symbolic trigonometric system
    example_7_native_trigonometric();

    // Example 8: 3D system with native symbolic
    example_8_native_three_dimensional();

    // Example 9: Parametric system with native symbolic
    example_9_native_parametric();

    // Example 10: Complex native symbolic expression
    example_10_complex_native();

    println!("\n=== All examples completed ===");
}

fn example_1_builder_string() {
    println!("Example 1: Basic Builder Pattern with String Equations");
    println!("-------------------------------------------------------");
    println!("System: x² + y² = 1, x = y");
    println!("Expected solution: x = y = ±√2/2 ≈ ±0.707\n");

    let solver = LM::new()
        .with_equations_str(vec!["x^2 + y^2 - 1".to_string(), "x - y".to_string()])
        .with_unknowns(vec!["x".to_string(), "y".to_string()])
        .with_initial_guess(vec![0.5, 0.5])
        .with_tolerance(1e-8)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!(
            "  Verification: x² + y² = {:.6}",
            map["x"].powi(2) + map["y"].powi(2)
        );
    }
    println!();
}

fn example_2_builder_expr() {
    println!("Example 2: Builder Pattern with Expr Objects");
    println!("---------------------------------------------");
    println!("System: x² + y² = 1, x = y\n");

    let eq1 = Expr::parse_expression("x^2 + y^2 - 1");
    let eq2 = Expr::parse_expression("x - y");

    let solver = LM::new()
        .with_equations(vec![eq1, eq2])
        .with_unknowns(vec!["x".to_string(), "y".to_string()])
        .with_initial_guess(vec![0.5, 0.5])
        .with_tolerance(1e-8)
        .with_f_tolerance(1e-10)
        .with_g_tolerance(1e-10)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution: x = {:.6}, y = {:.6}\n", map["x"], map["y"]);
    }
}

fn example_3_native_symbolic() {
    println!("Example 3: Native Symbolic Construction (No String Parsing)");
    println!("-----------------------------------------------------------");
    println!("System: x² + y² = 1, x = y");
    println!("Using Expr::Symbols() and native operations\n");

    // Create symbolic variables
    let vars = Expr::Symbols("x, y");
    let x = vars[0].clone();
    let y = vars[1].clone();

    // Build equations using native symbolic operations
    let eq1 = x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0)) - Expr::Const(1.0);
    let eq2 = x.clone() - y.clone();

    let solver = LM::new()
        .with_equations(vec![eq1, eq2])
        .with_unknowns(vec!["x".to_string(), "y".to_string()])
        .with_initial_guess(vec![0.5, 0.5])
        .with_tolerance(1e-8)
        .with_f_tolerance(1e-10)
        .with_g_tolerance(1e-10)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!(
            "  Verification: x² + y² = {:.6}",
            map["x"].powi(2) + map["y"].powi(2)
        );
    }
    println!();
}

fn example_4_traditional_api() {
    println!("Example 3: Traditional API (for comparison)");
    println!("--------------------------------------------");
    println!("System: x² + y² = 1, x = y\n");

    let mut solver = LM::new();
    solver.eq_generate_from_str(
        vec!["x^2 + y^2 - 1".to_string(), "x - y".to_string()],
        Some(vec!["x".to_string(), "y".to_string()]),
        None,
        vec![0.5, 0.5],
        Some(1e-8),
        None,
        None,
        None,
        None,
    );
    solver.set_loglevel("none".to_string());
    solver.eq_generate();
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution: x = {:.6}, y = {:.6}\n", map["x"], map["y"]);
    }
}

fn example_4_rosenbrock() {
    println!("Example 4: Rosenbrock Function");
    println!("-------------------------------");
    println!("System: 10(y - x²) = 0, 1 - x = 0");
    println!("Expected solution: x = 1, y = 1\n");

    let solver = LM::new()
        .with_equations_str(vec!["10*(y - x^2)".to_string(), "1 - x".to_string()])
        .with_initial_guess(vec![-1.2, 1.0])
        .with_tolerance(1e-8)
        .with_max_iterations(200)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!(
            "  Error from (1,1): {:.2e}",
            ((map["x"] - 1.0).powi(2) + (map["y"] - 1.0).powi(2)).sqrt()
        );
    }
    println!();
}

fn example_6_native_exponential() {
    println!("Example 6: Native Symbolic Exponential System");
    println!("----------------------------------------------");
    println!("System: exp(x) + y = 3, x + exp(y) = 3");
    println!("Using native Expr::exp() construction\n");

    let vars = Expr::Symbols("x, y");
    let x = vars[0].clone();
    let y = vars[1].clone();

    // Build equations with exponentials
    let eq1 = Expr::exp(x.clone()) + y.clone() - Expr::Const(3.0);
    let eq2 = x.clone() + Expr::exp(y.clone()) - Expr::Const(3.0);

    let solver = LM::new()
        .with_equations(vec![eq1, eq2])
        .with_initial_guess(vec![0.5, 0.5])
        .with_f_tolerance(1e-8)
        .with_g_tolerance(1e-8)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!(
            "  Symmetry check |x - y| = {:.2e}",
            (map["x"] - map["y"]).abs()
        );
        println!(
            "  Verification: exp(x) + y = {:.6}",
            map["x"].exp() + map["y"]
        );
    }
    println!();
}

fn example_7_native_trigonometric() {
    println!("Example 7: Native Symbolic Trigonometric System");
    println!("------------------------------------------------");
    println!("System: sin(x) + cos(y) = 1, cos(x) - sin(y) = 0");
    println!("Using native Expr::sin() and Expr::cos()\n");

    let vars = Expr::Symbols("x, y");
    let x = vars[0].clone();
    let y = vars[1].clone();

    // Build equations with trig functions
    let eq1 = Expr::sin(Box::new(x.clone())) + Expr::cos(Box::new(y.clone())) - Expr::Const(1.0);
    let eq2 = Expr::cos(Box::new(x.clone())) - Expr::sin(Box::new(y.clone()));

    let mut solver = LM::new()
        .with_equations(vec![eq1, eq2])
        .with_initial_guess(vec![0.5, 0.5])
        .with_tolerance(1e-7)
        .with_f_tolerance(1e-8)
        .with_g_tolerance(1e-8)
        .with_max_iterations(150)
        .with_loglevel("none".to_string())
        .build();
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!("  Verification:");
        println!(
            "    sin(x) + cos(y) = {:.6}",
            map["x"].sin() + map["y"].cos()
        );
        println!(
            "    cos(x) - sin(y) = {:.6}",
            map["x"].cos() - map["y"].sin()
        );
    }
    println!();
}

fn example_8_native_three_dimensional() {
    println!("Example 8: Native Symbolic Three-Dimensional System");
    println!("----------------------------------------------------");
    println!("System: x² + y² + z² = 1, x·y + z = 0.5, x + y + z = 1\n");

    let vars = Expr::Symbols("x, y, z");
    let x = vars[0].clone();
    let y = vars[1].clone();
    let z = vars[2].clone();

    // Build 3D system
    let eq1 = x.clone().pow(Expr::Const(2.0))
        + y.clone().pow(Expr::Const(2.0))
        + z.clone().pow(Expr::Const(2.0))
        - Expr::Const(1.0);
    let eq2 = x.clone() * y.clone() + z.clone() - Expr::Const(0.5);
    let eq3 = x.clone() + y.clone() + z.clone() - Expr::Const(1.0);

    let solver = LM::new()
        .with_equations(vec![eq1, eq2, eq3])
        .with_unknowns(vec!["x".to_string(), "y".to_string(), "z".to_string()])
        .with_initial_guess(vec![0.3, 0.3, 0.4])
        .with_tolerance(1e-7)
        .with_max_iterations(200)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!("  z = {:.6}", map["z"]);
        println!("  Verification:");
        println!(
            "    x² + y² + z² = {:.6}",
            map["x"].powi(2) + map["y"].powi(2) + map["z"].powi(2)
        );
        println!("    x·y + z = {:.6}", map["x"] * map["y"] + map["z"]);
        println!("    x + y + z = {:.6}", map["x"] + map["y"] + map["z"]);
    }
    println!();
}

fn example_9_native_parametric() {
    println!("Example 9: Native Symbolic Parametric System");
    println!("---------------------------------------------");
    println!("System: a·x² + b·y² = 1, x = y");
    println!("Using native symbolic parameters\n");

    let vars = Expr::Symbols("x, y");
    let params = Expr::Symbols("a, b");
    let x = vars[0].clone();
    let y = vars[1].clone();
    let a = params[0].clone();
    let b = params[1].clone();

    // Build parametric equations
    let eq1 = a * x.clone().pow(Expr::Const(2.0)) + b * y.clone().pow(Expr::Const(2.0))
        - Expr::Const(1.0);
    let eq2 = x.clone() - y.clone();

    let solver = LM::new()
        .with_equations(vec![eq1, eq2])
        .with_unknowns(vec!["x".to_string(), "y".to_string()])
        .with_parameters(vec!["a".to_string(), "b".to_string()])
        .with_initial_guess(vec![0.5, 0.5])
        .with_tolerance(1e-8)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve_with_params(vec![1.0, 1.0]);

    if let Some(map) = solver.map_of_solutions {
        println!("Solution with a=1, b=1:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
    }

    // Solve again with different parameters
    let vars2 = Expr::Symbols("x, y");
    let params2 = Expr::Symbols("a, b");
    let x2 = vars2[0].clone();
    let y2 = vars2[1].clone();
    let a2 = params2[0].clone();
    let b2 = params2[1].clone();

    let eq1_2 = a2 * x2.clone().pow(Expr::Const(2.0)) + b2 * y2.clone().pow(Expr::Const(2.0))
        - Expr::Const(1.0);
    let eq2_2 = x2.clone() - y2.clone();

    let solver2 = LM::new()
        .with_equations(vec![eq1_2, eq2_2])
        .with_unknowns(vec!["x".to_string(), "y".to_string()])
        .with_parameters(vec!["a".to_string(), "b".to_string()])
        .with_initial_guess(vec![0.5, 0.5])
        .with_tolerance(1e-8)
        .with_loglevel("none".to_string())
        .build();

    let mut solver2 = solver2;
    solver2.solve_with_params(vec![2.0, 0.5]);

    if let Some(map) = solver2.map_of_solutions {
        println!("\nSolution with a=2, b=0.5:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!(
            "  Verification: 2x² + 0.5y² = {:.6}",
            2.0 * map["x"].powi(2) + 0.5 * map["y"].powi(2)
        );
    }
    println!();
}

fn example_10_complex_native() {
    println!("Example 10: Complex Native Symbolic Expression");
    println!("-----------------------------------------------");
    println!("System with logarithms, exponentials, and divisions\n");

    let vars = Expr::Symbols("x, y");
    let x = vars[0].clone();
    let y = vars[1].clone();

    // ln(x) + y = 2, x + ln(y) = 2
    let eq1 = Expr::ln(x.clone()) + y.clone() - Expr::Const(2.0);
    let eq2 = x.clone() + Expr::ln(y.clone()) - Expr::Const(2.0);

    let solver = LM::new()
        .with_equations(vec![eq1, eq2])
        .with_initial_guess(vec![1.0, 1.0])
        .with_tolerance(1e-8)
        .with_loglevel("none".to_string())
        .build();

    let mut solver = solver;
    solver.solve();

    if let Some(map) = solver.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!(
            "  Symmetry check |x - y| = {:.2e}",
            (map["x"] - map["y"]).abs()
        );
        println!(
            "  Verification: ln(x) + y = {:.6}",
            map["x"].ln() + map["y"]
        );
    }

    println!("\n--- Division Example ---");
    let vars2 = Expr::Symbols("x, y");
    let x2 = vars2[0].clone();
    let y2 = vars2[1].clone();

    // x/y = 2, x + y = 3
    let eq1_div = x2.clone() / y2.clone() - Expr::Const(2.0);
    let eq2_div = x2.clone() + y2.clone() - Expr::Const(3.0);

    let solver_div = LM::new()
        .with_equations(vec![eq1_div, eq2_div])
        .with_initial_guess(vec![1.5, 1.0])
        .with_tolerance(1e-8)
        .with_loglevel("none".to_string())
        .build();

    let mut solver_div = solver_div;
    solver_div.solve();

    if let Some(map) = solver_div.map_of_solutions {
        println!("Solution found:");
        println!("  x = {:.6}", map["x"]);
        println!("  y = {:.6}", map["y"]);
        println!(
            "  Verification: x/y = {:.6}, x + y = {:.6}",
            map["x"] / map["y"],
            map["x"] + map["y"]
        );
    }
    println!();
}
