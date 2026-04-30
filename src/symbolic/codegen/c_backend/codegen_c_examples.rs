//! Examples demonstrating C code generation from symbolic expressions.
//!
//! This module provides practical examples of using the C code generation
//! pipeline for various mathematical scenarios.

#[cfg(test)]
mod examples {
    use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule};
    use crate::symbolic::symbolic_engine::Expr;
    use std::fs;
    use tempfile::tempdir;

    /// Example 1: Generate a simple C function for a quadratic expression.
    #[test]
    fn example_simple_quadratic() {
        let x = Expr::Var("x".to_string());
        // f(x) = x^2 + 2*x + 1
        let expr =
            x.clone().pow(Expr::Const(2.0)) + Expr::Const(2.0) * x.clone() + Expr::Const(1.0);

        let module = CodegenModule::new("quadratic")
            .with_language(CodegenLanguage::C)
            .add_function("evaluate_quadratic", &expr, &["x"]);

        let c_source = module.emit_source();
        let c_header = module.emit_c_header();

        println!("=== C Header ===");
        println!("{}", c_header);
        println!("\n=== C Source ===");
        println!("{}", c_source);

        // Verify the generated code contains expected elements
        assert!(c_source.contains("double evaluate_quadratic(const double* inputs)"));
        assert!(c_source.contains("pow("));
        assert!(c_header.contains("double evaluate_quadratic(const double* inputs);"));
    }

    /// Example 2: Generate C code for a system of equations (residual block).
    #[test]
    fn example_residual_system() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        // System: F1 = x^2 + y^2 - 10
        //         F2 = x - y - 4
        let residuals = vec![
            x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0)) - Expr::Const(10.0),
            x.clone() - y.clone() - Expr::Const(4.0),
        ];

        let module = CodegenModule::new("nonlinear_system")
            .with_language(CodegenLanguage::C)
            .add_block("compute_residuals", &residuals, &["x", "y"]);

        let c_source = module.emit_source();
        let c_header = module.emit_c_header();

        println!("=== Nonlinear System C Code ===");
        println!("{}", c_source);

        assert!(c_source.contains("void compute_residuals(const double* inputs, double* outputs)"));
        assert!(c_source.contains("outputs[0] = "));
        assert!(c_source.contains("outputs[1] = "));
    }

    /// Example 3: Generate C code for trigonometric functions.
    #[test]
    fn example_trigonometric_functions() {
        let theta = Expr::Var("theta".to_string());

        let trig_exprs = vec![
            Expr::sin(Box::new(theta.clone())),
            Expr::cos(Box::new(theta.clone())),
            Expr::tg(Box::new(theta.clone())),
            Expr::sin(Box::new(theta.clone())).pow(Expr::Const(2.0))
                + Expr::cos(Box::new(theta.clone())).pow(Expr::Const(2.0)), // Should equal 1
        ];

        let module = CodegenModule::new("trigonometry")
            .with_language(CodegenLanguage::C)
            .add_block("trig_functions", &trig_exprs, &["theta"]);

        let c_source = module.emit_source();

        println!("=== Trigonometric Functions C Code ===");
        println!("{}", c_source);

        assert!(c_source.contains("sin("));
        assert!(c_source.contains("cos("));
        assert!(c_source.contains("tan("));
    }

    /// Example 4: Generate C code for exponential and logarithmic functions.
    #[test]
    fn example_exp_log_functions() {
        let x = Expr::Var("x".to_string());

        let exprs = vec![
            Expr::Exp(Box::new(x.clone())),
            Expr::Ln(Box::new(x.clone())),
            Expr::Exp(Box::new(Expr::Ln(Box::new(x.clone())))), // Should equal x
            x.clone().pow(Expr::Const(2.0)),
        ];

        let module = CodegenModule::new("exp_log")
            .with_language(CodegenLanguage::C)
            .add_block("exp_log_functions", &exprs, &["x"]);

        let c_source = module.emit_source();

        println!("=== Exponential and Logarithmic Functions C Code ===");
        println!("{}", c_source);

        // Note: C uses log() for natural logarithm
        assert!(c_source.contains("exp("));
        assert!(c_source.contains("log("));
        assert!(!c_source.contains("ln("));
    }

    /// Example 5: Generate both Rust and C code for comparison.
    #[test]
    fn example_rust_vs_c_comparison() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        // Distance formula: sqrt(x^2 + y^2)
        let expr = (x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0)))
            .pow(Expr::Const(0.5));

        let rust_module = CodegenModule::new("distance")
            .with_language(CodegenLanguage::Rust)
            .add_function("euclidean_distance", &expr, &["x", "y"]);

        let c_module = CodegenModule::new("distance")
            .with_language(CodegenLanguage::C)
            .add_function("euclidean_distance", &expr, &["x", "y"]);

        let rust_source = rust_module.emit_source();
        let c_source = c_module.emit_source();

        println!("=== Rust Version ===");
        println!("{}", rust_source);
        println!("\n=== C Version ===");
        println!("{}", c_source);

        // Both should compute the same mathematical operation
        assert!(rust_source.contains("euclidean_distance"));
        assert!(c_source.contains("euclidean_distance"));
    }

    /// Example 6: Write C code to files (header and source).
    #[test]
    fn example_write_to_files() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let exprs = vec![
            x.clone() + y.clone(),
            x.clone() * y.clone(),
            x.clone() - y.clone(),
            x.clone() / y.clone(),
        ];

        let module = CodegenModule::new("arithmetic")
            .with_language(CodegenLanguage::C)
            .add_block("basic_operations", &exprs, &["x", "y"]);

        let dir = tempdir().expect("Failed to create temp directory");
        let header_path = dir.path().join("arithmetic.h");
        let source_path = dir.path().join("arithmetic.c");

        // Write header file
        fs::write(&header_path, module.emit_c_header()).expect("Failed to write header file");

        // Write source file
        fs::write(&source_path, module.emit_source()).expect("Failed to write source file");

        // Verify files were created
        assert!(header_path.exists());
        assert!(source_path.exists());

        let header_content = fs::read_to_string(&header_path).unwrap();
        let source_content = fs::read_to_string(&source_path).unwrap();

        println!("=== Generated arithmetic.h ===");
        println!("{}", header_content);
        println!("\n=== Generated arithmetic.c ===");
        println!("{}", source_content);

        assert!(header_content.contains("#ifndef GENERATED_FUNCTIONS_H"));
        assert!(source_content.contains("#include <math.h>"));
    }

    /// Example 7: Complex mathematical expression with optimization.
    #[test]
    fn example_optimized_expression() {
        let x = Expr::Var("x".to_string());

        // Expression with redundant operations that will be optimized
        let expr = (x.clone() + Expr::Const(0.0)) * Expr::Const(1.0)
            + x.clone().pow(Expr::Const(1.0))
            + Expr::Const(0.0);

        let ir = expr.lower_to_linear(&["x"]);
        let optimized_ir = ir.peephole_optimize();

        println!("Original IR instructions: {}", ir.instructions.len());
        println!(
            "Optimized IR instructions: {}",
            optimized_ir.instructions.len()
        );

        let module = CodegenModule::new("optimized")
            .with_language(CodegenLanguage::C)
            .add_function("optimized_func", &expr, &["x"]);

        let c_source = module.emit_source();

        println!("=== Optimized C Code ===");
        println!("{}", c_source);

        // The optimized code should be simpler
        assert!(optimized_ir.instructions.len() < ir.instructions.len());
    }

    /// Example 8: Generate C code for a Jacobian-like structure.
    #[test]
    fn example_jacobian_structure() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        // Partial derivatives of f(x,y) = x^2 + y^2
        let f = x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0));
        let df_dx = f.diff("x");
        let df_dy = f.diff("y");

        let jacobian_entries = vec![df_dx, df_dy];

        let module = CodegenModule::new("jacobian")
            .with_language(CodegenLanguage::C)
            .add_block("compute_gradient", &jacobian_entries, &["x", "y"]);

        let c_source = module.emit_source();

        println!("=== Jacobian/Gradient C Code ===");
        println!("{}", c_source);

        assert!(c_source.contains("void compute_gradient"));
    }

    /// Example 9: Multiple functions in one module.
    #[test]
    fn example_multiple_functions_module() {
        let x = Expr::Var("x".to_string());

        let square = x.clone().pow(Expr::Const(2.0));
        let cube = x.clone().pow(Expr::Const(3.0));
        let sqrt = x.clone().pow(Expr::Const(0.5));

        let module = CodegenModule::new("powers")
            .with_language(CodegenLanguage::C)
            .add_function("square", &square, &["x"])
            .add_function("cube", &cube, &["x"])
            .add_function("sqrt_func", &sqrt, &["x"]);

        let c_source = module.emit_source();
        let c_header = module.emit_c_header();

        println!("=== Multiple Functions Module ===");
        println!("{}", c_source);

        assert!(c_source.contains("double square"));
        assert!(c_source.contains("double cube"));
        assert!(c_source.contains("double sqrt_func"));

        assert!(c_header.contains("double square(const double* inputs);"));
        assert!(c_header.contains("double cube(const double* inputs);"));
        assert!(c_header.contains("double sqrt_func(const double* inputs);"));
    }

    /// Example 10: Demonstrate IR evaluation matches symbolic evaluation.
    #[test]
    fn example_verify_numerical_correctness() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let expr = (x.clone() + y.clone()).pow(Expr::Const(2.0))
            + Expr::sin(Box::new(x.clone() * y.clone()));

        // Evaluate using symbolic lambdify
        let symbolic_func = expr.lambdify1(&["x", "y"]);
        let args = [1.5, 2.5];
        let symbolic_result = symbolic_func(&args);

        // Evaluate using IR
        let ir = expr.lower_to_linear(&["x", "y"]);
        let ir_result = ir.eval(&args);

        println!("Symbolic result: {}", symbolic_result);
        println!("IR result: {}", ir_result);
        println!("Difference: {}", (symbolic_result - ir_result).abs());

        // They should match within numerical precision
        assert!((symbolic_result - ir_result).abs() < 1e-12);

        // Generate C code from the same IR
        let module = CodegenModule::new("verified")
            .with_language(CodegenLanguage::C)
            .add_function("verified_func", &expr, &["x", "y"]);

        let c_source = module.emit_source();

        println!("\n=== Generated C Code (numerically verified) ===");
        println!("{}", c_source);
    }
}
