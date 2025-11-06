#[cfg(test)]
mod performance_tests {
    use crate::symbolic::symbolic_engine::Expr;
    use std::time::Instant;

    fn create_complex_expression() -> Expr {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());

        Expr::sin(Box::new(x.clone() * y.clone()))
            + Expr::Exp(Box::new(z.clone() / x.clone()))
            + Expr::cos(Box::new(
                x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0)),
            ))
            + Expr::Ln(Box::new(x.clone() + y.clone() + z.clone()))
            + x.clone() * y.clone() * z.clone()
    }

    fn create_very_complex_expression() -> Expr {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        let w = Expr::Var("w".to_string());

        let part1 = Expr::sin(Box::new(Expr::Exp(Box::new(x.clone() * y.clone()))));
        let part2 = Expr::cos(Box::new(Expr::Ln(Box::new(z.clone() + w.clone()))));
        let part3 = Expr::arctg(Box::new(
            x.clone().pow(Expr::Const(3.0)) / (y.clone() + Expr::Const(1.0)),
        ));
        let part4 = Expr::Exp(Box::new(Expr::sin(Box::new(z.clone() * w.clone()))));
        let part5 = x.clone() * y.clone() * z.clone() * w.clone();

        part1 + part2 + part3 + part4 + part5
    }

    #[test]
    fn performance_comparison_lambdify_methods() {
        println!("\n=== Performance Comparison: lambdify1 vs lambdify2 ===");

        let expressions = vec![
            (
                "Simple",
                Expr::Var("x".to_string()) * Expr::Const(2.0) + Expr::Const(1.0),
            ),
            ("Complex", create_complex_expression()),
            ("Very Complex", create_very_complex_expression()),
        ];

        let test_args = vec![1.5, 2.0, 0.5, 1.0];
        let iterations = 100_000;

        for (name, expr) in expressions {
            println!("\n--- {} Expression ---", name);

            let start = Instant::now();
            let func1 = expr.lambdify1(&["x", "y", "z", "w"]);
            let compile_time1 = start.elapsed();

            let start = Instant::now();
            let func2 = expr.lambdify2(&["x", "y", "z", "w"]);
            let compile_time2 = start.elapsed();

            println!(
                "Compilation time - lambdify1: {:?}, lambdify2: {:?}",
                compile_time1, compile_time2
            );

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = func1(&test_args);
            }
            let exec_time1 = start.elapsed();

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = func2(&test_args);
            }
            let exec_time2 = start.elapsed();

            println!(
                "Execution time ({} iterations) - lambdify1: {:?}, lambdify2: {:?}",
                iterations, exec_time1, exec_time2
            );

            let result1 = func1(&test_args);
            let result2 = func2(&test_args);
            println!("Simplified expression for lambdify1 {:.2}.", result1);
            println!("Simplified expression for lambdify2 {:.2}.", result2);
            assert!(
                (result1 - result2).abs() < 1e-10,
                "Results differ: {} vs {}",
                result1,
                result2
            );

            println!("Result verification: {} (both methods identical)", result1);

            let compile_ratio = compile_time1.as_nanos() as f64 / compile_time2.as_nanos() as f64;
            let exec_ratio = exec_time1.as_nanos() as f64 / exec_time2.as_nanos() as f64;

            println!(
                "Performance ratios (lambdify1/lambdify2) - Compile: {:.2}x, Execute: {:.2}x",
                compile_ratio, exec_ratio
            );
        }
    }

    #[test]
    fn benchmark_compilation_overhead() {
        println!("\n=== Compilation Overhead Benchmark ===");

        let expr = create_very_complex_expression();
        let vars = ["x", "y", "z", "w"];
        let compilations = 1000;

        let start = Instant::now();
        for _ in 0..compilations {
            let _ = expr.lambdify1(&vars);
        }
        let total_time1 = start.elapsed();

        let start = Instant::now();
        for _ in 0..compilations {
            let _ = expr.lambdify2(&vars);
        }
        let total_time2 = start.elapsed();

        println!("Total compilation time ({} compilations):", compilations);
        println!(
            "lambdify1: {:?} (avg: {:?})",
            total_time1,
            total_time1 / compilations
        );
        println!(
            "lambdify2: {:?} (avg: {:?})",
            total_time2,
            total_time2 / compilations
        );

        let ratio = total_time1.as_nanos() as f64 / total_time2.as_nanos() as f64;
        println!(
            "Compilation speed ratio (lambdify1/lambdify2): {:.2}x",
            ratio
        );
    }

    #[test]
    fn benchmark_execution_patterns() {
        println!("\n=== Execution Pattern Benchmark ===");

        let expr = create_complex_expression();
        let vars = ["x", "y", "z"];

        let func1 = expr.lambdify1(&vars);
        let func2 = expr.lambdify2(&vars);

        let patterns = vec![
            ("Small values", vec![0.1, 0.2, 0.3]),
            ("Unit values", vec![1.0, 1.0, 1.0]),
            ("Large values", vec![10.0, 20.0, 30.0]),
            ("Mixed values", vec![0.1, 5.0, 100.0]),
        ];

        let iterations = 50_000;

        for (pattern_name, args) in patterns {
            println!("\n--- {} ---", pattern_name);

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = func1(&args);
            }
            let time1 = start.elapsed();

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = func2(&args);
            }
            let time2 = start.elapsed();

            let result1 = func1(&args);
            let result2 = func2(&args);

            println!("lambdify1: {:?}, lambdify2: {:?}", time1, time2);
            println!(
                "Results: {:.6} vs {:.6} (diff: {:.2e})",
                result1,
                result2,
                (result1 - result2).abs()
            );

            let ratio = time1.as_nanos() as f64 / time2.as_nanos() as f64;
            println!("Speed ratio (lambdify1/lambdify2): {:.2}x", ratio);
        }
    }

    fn build_expression() -> Expr {
        use Expr::*;
        Add(
            Box::new(Mul(
                Box::new(sin(Box::new(Var("x".into())))),
                Box::new(cos(Box::new(Var("y".into())))),
            )),
            Box::new(Mul(Box::new(Var("x".into())), Box::new(Var("y".into())))),
        )
    }

    #[test]
    fn performance_test_lambdify1_vs_lambdify2() {
        println!("\n=== Performance Test: lambdify1 vs lambdify2 with build_expression ===\n");

        let expr = build_expression();
        let vars = ["x", "y"];
        let test_args = [1.5, 2.0];
        let iterations = 1_000_000;

        // Compilation time comparison
        let start = Instant::now();
        let func1 = expr.lambdify1(&vars);
        let compile_time1 = start.elapsed();

        let start = Instant::now();
        let func2 = expr.lambdify2(&vars);
        let compile_time2 = start.elapsed();

        println!("Compilation times:");
        println!("  lambdify1: {:?}", compile_time1);
        println!("  lambdify2: {:?}", compile_time2);
        println!(
            "  Ratio (1/2): {:.2}x\n",
            compile_time1.as_nanos() as f64 / compile_time2.as_nanos() as f64
        );

        // Execution time comparison
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = func1(&test_args);
        }
        let exec_time1 = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = func2(&test_args);
        }
        let exec_time2 = start.elapsed();

        println!("Execution times ({} iterations):", iterations);
        println!(
            "  lambdify1: {:?} ({:.2} ns/call)",
            exec_time1,
            exec_time1.as_nanos() as f64 / iterations as f64
        );
        println!(
            "  lambdify2: {:?} ({:.2} ns/call)",
            exec_time2,
            exec_time2.as_nanos() as f64 / iterations as f64
        );
        println!(
            "  Ratio (1/2): {:.2}x\n",
            exec_time1.as_nanos() as f64 / exec_time2.as_nanos() as f64
        );

        // Verify results are identical
        let result1 = func1(&test_args);
        let result2 = func2(&test_args);
        assert!(
            (result1 - result2).abs() < 1e-12,
            "Results differ: {} vs {}",
            result1,
            result2
        );
        println!(
            "Result verification: {:.6} (both methods identical)",
            result1
        );

        // Overall performance summary
        let total_time1 = compile_time1 + exec_time1;
        let total_time2 = compile_time2 + exec_time2;
        println!("\nTotal time (compile + execute):");
        println!("  lambdify1: {:?}", total_time1);
        println!("  lambdify2: {:?}", total_time2);
        println!(
            "  Overall ratio (1/2): {:.2}x",
            total_time1.as_nanos() as f64 / total_time2.as_nanos() as f64
        );
    }
}
