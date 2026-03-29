#[cfg(test)]
mod performance_tests {
    use crate::symbolic::symbolic_engine::Expr;
    use rayon::prelude::*;
    use std::time::Instant;

    fn create_small_expression() -> Expr {
        Expr::Var("x".to_string()) * Expr::Const(2.0) + Expr::Const(1.0)
    }

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

    fn create_batch_expressions() -> Vec<Expr> {
        vec![
            Expr::parse_expression("2*x + 1"),
            Expr::parse_expression("x*y + sin(x)"),
            Expr::parse_expression("exp(x + y) - ln(z + 1)"),
            Expr::parse_expression("x^2 + y^2 + z^2"),
            Expr::parse_expression("cos(x*y) + z/x"),
        ]
    }

    fn print_ratio(
        label: &str,
        old_duration: std::time::Duration,
        new_duration: std::time::Duration,
    ) {
        let ratio = old_duration.as_nanos() as f64 / new_duration.as_nanos().max(1) as f64;
        println!(
            "{} - old/new ratio: {:.2}x (old: {:?}, new: {:?})",
            label, ratio, old_duration, new_duration
        );
    }

    fn handwritten_build_expression(args: &[f64]) -> f64 {
        let x = args[0];
        let y = args[1];
        x.sin() * y.cos() + x * y
    }

    fn handwritten_complex_expression(args: &[f64]) -> f64 {
        let x = args[0];
        let y = args[1];
        let z = args[2];

        (x * y).sin()
            + (z / x).exp()
            + (x.powf(2.0) + y.powf(2.0)).cos()
            + (x + y + z).ln()
            + x * y * z
    }

    #[test]
    fn performance_comparison_lambdify_methods() {
        println!("\n=== Performance Comparison: lambdify1 vs lambdify2 ===");

        let expressions = vec![
            ("Simple", create_small_expression()),
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
    fn performance_compare_small_and_large_ir_backend() {
        println!("\n=== Performance Comparison: small vs large expressions ===");

        let cases = vec![
            ("Small", create_small_expression(), vec![1.5]),
            (
                "Large",
                create_very_complex_expression(),
                vec![1.5, 2.0, 0.5, 1.0],
            ),
        ];

        for (name, expr, args) in cases {
            println!("\n--- {} ---", name);

            let start = Instant::now();
            let func1 = expr.lambdify1(&["x", "y", "z", "w"][..args.len()]);
            let compile_old = start.elapsed();

            let start = Instant::now();
            let func2 = expr.lambdify2(&["x", "y", "z", "w"][..args.len()]);
            let compile_new = start.elapsed();

            let iterations = if name == "Small" { 500_000 } else { 75_000 };

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = func1(&args);
            }
            let exec_old = start.elapsed();

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = func2(&args);
            }
            let exec_new = start.elapsed();

            assert!((func1(&args) - func2(&args)).abs() < 1e-10);
            print_ratio("compile", compile_old, compile_new);
            print_ratio("execute", exec_old, exec_new);
        }
    }

    #[test]
    fn performance_compare_batch_ir_vs_individual_closures() {
        println!("\n=== Performance Comparison: batch IR vs individual closures ===");

        let exprs = create_batch_expressions();
        let vars = ["x", "y", "z"];
        let args = [1.25, 2.5, 0.75];
        let iterations = 100_000;

        let start = Instant::now();
        let individual: Vec<_> = exprs
            .iter()
            .map(|expr| expr.lambdify_borrowed_thread_safe(&vars))
            .collect();
        let compile_individual = start.elapsed();

        let start = Instant::now();
        let compiled_batch = Expr::compile_many_ir(&exprs, &vars);
        let compile_batch = start.elapsed();

        let start = Instant::now();
        let mut individual_values = vec![0.0; exprs.len()];
        for _ in 0..iterations {
            for (slot, func) in individual_values.iter_mut().zip(individual.iter()) {
                *slot = func(&args);
            }
        }
        let exec_individual = start.elapsed();

        let start = Instant::now();
        let mut batch_values = Vec::new();
        for _ in 0..iterations {
            batch_values = compiled_batch.eval(&args);
        }
        let exec_batch = start.elapsed();

        assert_eq!(individual_values.len(), batch_values.len());
        for (lhs, rhs) in individual_values.iter().zip(batch_values.iter()) {
            assert!((lhs - rhs).abs() < 1e-10);
        }

        print_ratio("batch compile", compile_individual, compile_batch);
        print_ratio("batch execute", exec_individual, exec_batch);
    }

    #[test]
    fn parallel_ir_backend_matches_parallel_closure_results() {
        println!("\n=== Parallel Evaluation Check: thread-safe IR backend ===");

        let expr = create_complex_expression();
        let vars = ["x", "y", "z"];
        let old_backend = expr.lambdify_borrowed_thread_safe(&vars);
        let new_backend = expr.lambdify2(&vars);
        let inputs: Vec<Vec<f64>> = (0..10_000)
            .map(|i| {
                let x = 0.5 + i as f64 * 1e-4;
                let y = 1.0 + i as f64 * 2e-4;
                let z = 1.5 + i as f64 * 3e-4;
                vec![x, y, z]
            })
            .collect();

        let start = Instant::now();
        let old_results: Vec<f64> = inputs.par_iter().map(|args| old_backend(args)).collect();
        let old_parallel = start.elapsed();

        let start = Instant::now();
        let new_results: Vec<f64> = inputs.par_iter().map(|args| new_backend(args)).collect();
        let new_parallel = start.elapsed();

        for (lhs, rhs) in old_results.iter().zip(new_results.iter()) {
            assert!((lhs - rhs).abs() < 1e-10);
        }

        print_ratio("parallel execute", old_parallel, new_parallel);
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

    #[test]
    fn performance_compare_lambdify_vs_handwritten_small_and_complex() {
        println!("\n=== Performance Test: lambdify vs handwritten functions ===\n");

        let small_expr = build_expression();
        let small_vars = ["x", "y"];
        let small_args = [1.5, 2.0];
        let small_iterations = 1_000_000;

        let lambdify1_small = small_expr.lambdify1(&small_vars);
        let lambdify2_small = small_expr.lambdify2(&small_vars);
        let handwritten_small_closure =
            |args: &[f64]| args[0].sin() * args[1].cos() + args[0] * args[1];

        let start = Instant::now();
        for _ in 0..small_iterations {
            let _ = lambdify1_small(&small_args);
        }
        let lambdify1_small_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..small_iterations {
            let _ = lambdify2_small(&small_args);
        }
        let lambdify2_small_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..small_iterations {
            let _ = handwritten_build_expression(&small_args);
        }
        let handwritten_small_fn_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..small_iterations {
            let _ = handwritten_small_closure(&small_args);
        }
        let handwritten_small_closure_time = start.elapsed();

        println!("Small expression:");
        println!("  lambdify1: {:?}", lambdify1_small_time);
        println!("  lambdify2: {:?}", lambdify2_small_time);
        println!("  handwritten fn: {:?}", handwritten_small_fn_time);
        println!(
            "  handwritten closure: {:?}",
            handwritten_small_closure_time
        );

        let complex_expr = create_complex_expression();
        let complex_vars = ["x", "y", "z"];
        let complex_args = [1.5, 0.75, 2.0];
        let complex_iterations = 250_000;

        let lambdify1_complex = complex_expr.lambdify1(&complex_vars);
        let lambdify2_complex = complex_expr.lambdify2(&complex_vars);
        let handwritten_complex_closure = |args: &[f64]| handwritten_complex_expression(args);

        let start = Instant::now();
        for _ in 0..complex_iterations {
            let _ = lambdify1_complex(&complex_args);
        }
        let lambdify1_complex_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..complex_iterations {
            let _ = lambdify2_complex(&complex_args);
        }
        let lambdify2_complex_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..complex_iterations {
            let _ = handwritten_complex_expression(&complex_args);
        }
        let handwritten_complex_fn_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..complex_iterations {
            let _ = handwritten_complex_closure(&complex_args);
        }
        let handwritten_complex_closure_time = start.elapsed();

        println!("Complex expression:");
        println!("  lambdify1: {:?}", lambdify1_complex_time);
        println!("  lambdify2: {:?}", lambdify2_complex_time);
        println!("  handwritten fn: {:?}", handwritten_complex_fn_time);
        println!(
            "  handwritten closure: {:?}",
            handwritten_complex_closure_time
        );

        let expected_small = handwritten_build_expression(&small_args);
        assert!((lambdify1_small(&small_args) - expected_small).abs() < 1e-12);
        assert!((lambdify2_small(&small_args) - expected_small).abs() < 1e-12);

        let expected_complex = handwritten_complex_expression(&complex_args);
        assert!((lambdify1_complex(&complex_args) - expected_complex).abs() < 1e-12);
        assert!((lambdify2_complex(&complex_args) - expected_complex).abs() < 1e-12);
    }

    pub fn parse_very_complex_expression() -> Expr {
        let s = " (0.000002669 * (28.0 * T)^0.5) /
        (13.3225 * ((1.16145 / ((T / 98.1) ^ 0.14874))
                  + (0.52487 / exp(0.7732 * (T / 98.1)))
                  + (2.16178 / exp(2.43787 * (T / 98.1)))
                  + ((0.2 * 1.0 ^ 2) / (T / 98.1))))";
        let s = Expr::parse_expression(s);
        let s1 = s.diff("T");
        let s2 = s1.diff("T");
        let s3 = s2.diff("T");
        s3
    }

    #[test]
    fn performance_test_lambdify1_vs_lambdify3() {
        let expr = parse_very_complex_expression();
        let vars = ["T"];
        let args = [298.15];
        let iterations = 1000_000;

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

        println!("lambdify1: {} ns/call", time1.as_nanos() / iterations);
        println!("lambdify2: {} ns/call", time2.as_nanos() / iterations);
    }
}
