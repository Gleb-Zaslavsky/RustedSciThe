//! Examples demonstrating Zig code generation from symbolic expressions.
//!
//! This module provides practical examples of using the Zig code generation
//! pipeline for various mathematical scenarios.

#[cfg(test)]
mod examples {
    use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule};
    use crate::symbolic::symbolic_engine::Expr;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn example_simple_quadratic() {
        let x = Expr::Var("x".to_string());
        let expr =
            x.clone().pow(Expr::Const(2.0)) + Expr::Const(2.0) * x.clone() + Expr::Const(1.0);

        let module = CodegenModule::new("quadratic")
            .with_language(CodegenLanguage::Zig)
            .add_function("evaluate_quadratic", &expr, &["x"]);

        let zig_source = module.emit_source();

        println!("=== Zig Source ===");
        println!("{}", zig_source);

        assert!(zig_source.contains("pub fn evaluate_quadratic(inputs: [*]const f64) f64"));
        assert!(zig_source.contains("std.math.pow(f64,"));
    }

    #[test]
    fn example_residual_system() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let residuals = vec![
            x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0)) - Expr::Const(10.0),
            x.clone() - y.clone() - Expr::Const(4.0),
        ];

        let module = CodegenModule::new("nonlinear_system")
            .with_language(CodegenLanguage::Zig)
            .add_block("compute_residuals", &residuals, &["x", "y"]);

        let zig_source = module.emit_source();

        println!("=== Nonlinear System Zig Code ===");
        println!("{}", zig_source);

        assert!(zig_source
            .contains("pub fn compute_residuals(inputs: [*]const f64, outputs: [*]f64) void"));
        assert!(zig_source.contains("outputs[0] = "));
        assert!(zig_source.contains("outputs[1] = "));
    }

    #[test]
    fn example_trigonometric_functions() {
        let theta = Expr::Var("theta".to_string());

        let trig_exprs = vec![
            Expr::sin(Box::new(theta.clone())),
            Expr::cos(Box::new(theta.clone())),
            Expr::tg(Box::new(theta.clone())),
            Expr::sin(Box::new(theta.clone())).pow(Expr::Const(2.0))
                + Expr::cos(Box::new(theta.clone())).pow(Expr::Const(2.0)),
        ];

        let module = CodegenModule::new("trigonometry")
            .with_language(CodegenLanguage::Zig)
            .add_block("trig_functions", &trig_exprs, &["theta"]);

        let zig_source = module.emit_source();

        println!("=== Trigonometric Functions Zig Code ===");
        println!("{}", zig_source);

        assert!(zig_source.contains("@sin("));
        assert!(zig_source.contains("@cos("));
        assert!(zig_source.contains("@tan("));
    }

    #[test]
    fn example_exp_log_functions() {
        let x = Expr::Var("x".to_string());

        let exprs = vec![
            Expr::Exp(Box::new(x.clone())),
            Expr::Ln(Box::new(x.clone())),
            Expr::Exp(Box::new(Expr::Ln(Box::new(x.clone())))),
            x.clone().pow(Expr::Const(2.0)),
        ];

        let module = CodegenModule::new("exp_log")
            .with_language(CodegenLanguage::Zig)
            .add_block("exp_log_functions", &exprs, &["x"]);

        let zig_source = module.emit_source();

        println!("=== Exponential and Logarithmic Functions Zig Code ===");
        println!("{}", zig_source);

        assert!(zig_source.contains("@exp("));
        assert!(zig_source.contains("@log("));
        assert!(!zig_source.contains("ln("));
    }

    #[test]
    fn example_rust_vs_zig_comparison() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let expr = (x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0)))
            .pow(Expr::Const(0.5));

        let rust_module = CodegenModule::new("distance")
            .with_language(CodegenLanguage::Rust)
            .add_function("euclidean_distance", &expr, &["x", "y"]);

        let zig_module = CodegenModule::new("distance")
            .with_language(CodegenLanguage::Zig)
            .add_function("euclidean_distance", &expr, &["x", "y"]);

        let rust_source = rust_module.emit_source();
        let zig_source = zig_module.emit_source();

        println!("=== Rust Version ===");
        println!("{}", rust_source);
        println!("\n=== Zig Version ===");
        println!("{}", zig_source);

        assert!(rust_source.contains("euclidean_distance"));
        assert!(zig_source.contains("euclidean_distance"));
    }

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
            .with_language(CodegenLanguage::Zig)
            .add_block("basic_operations", &exprs, &["x", "y"]);

        let dir = tempdir().expect("Failed to create temp directory");
        let source_path = dir.path().join("arithmetic.zig");

        fs::write(&source_path, module.emit_source()).expect("Failed to write source file");

        assert!(source_path.exists());

        let source_content = fs::read_to_string(&source_path).unwrap();

        println!("=== Generated arithmetic.zig ===");
        println!("{}", source_content);

        assert!(source_content.contains("const std = @import(\"std\");"));
        assert!(source_content.contains("pub fn basic_operations"));
    }

    #[test]
    fn example_optimized_expression() {
        let x = Expr::Var("x".to_string());

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
            .with_language(CodegenLanguage::Zig)
            .add_function("optimized_func", &expr, &["x"]);

        let zig_source = module.emit_source();

        println!("=== Optimized Zig Code ===");
        println!("{}", zig_source);

        assert!(optimized_ir.instructions.len() < ir.instructions.len());
    }

    #[test]
    fn example_jacobian_structure() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let f = x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0));
        let df_dx = f.diff("x");
        let df_dy = f.diff("y");

        let jacobian_entries = vec![df_dx, df_dy];

        let module = CodegenModule::new("jacobian")
            .with_language(CodegenLanguage::Zig)
            .add_block("compute_gradient", &jacobian_entries, &["x", "y"]);

        let zig_source = module.emit_source();

        println!("=== Jacobian/Gradient Zig Code ===");
        println!("{}", zig_source);

        assert!(zig_source.contains("pub fn compute_gradient"));
    }

    #[test]
    fn example_multiple_functions_module() {
        let x = Expr::Var("x".to_string());

        let square = x.clone().pow(Expr::Const(2.0));
        let cube = x.clone().pow(Expr::Const(3.0));
        let sqrt = x.clone().pow(Expr::Const(0.5));

        let module = CodegenModule::new("powers")
            .with_language(CodegenLanguage::Zig)
            .add_function("square", &square, &["x"])
            .add_function("cube", &cube, &["x"])
            .add_function("sqrt_func", &sqrt, &["x"]);

        let zig_source = module.emit_source();

        println!("=== Multiple Functions Module ===");
        println!("{}", zig_source);

        assert!(zig_source.contains("pub fn square"));
        assert!(zig_source.contains("pub fn cube"));
        assert!(zig_source.contains("pub fn sqrt_func"));
    }

    #[test]
    fn example_verify_numerical_correctness() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let expr = (x.clone() + y.clone()).pow(Expr::Const(2.0))
            + Expr::sin(Box::new(x.clone() * y.clone()));

        let symbolic_func = expr.lambdify1(&["x", "y"]);
        let args = [1.5, 2.5];
        let symbolic_result = symbolic_func(&args);

        let ir = expr.lower_to_linear(&["x", "y"]);
        let ir_result = ir.eval(&args);

        println!("Symbolic result: {}", symbolic_result);
        println!("IR result: {}", ir_result);
        println!("Difference: {}", (symbolic_result - ir_result).abs());

        assert!((symbolic_result - ir_result).abs() < 1e-12);

        let module = CodegenModule::new("verified")
            .with_language(CodegenLanguage::Zig)
            .add_function("verified_func", &expr, &["x", "y"]);

        let zig_source = module.emit_source();

        println!("\n=== Generated Zig Code (numerically verified) ===");
        println!("{}", zig_source);
    }

    #[test]
    fn example_inverse_trigonometric() {
        let x = Expr::Var("x".to_string());

        let exprs = vec![
            Expr::arcsin(Box::new(x.clone())),
            Expr::arccos(Box::new(x.clone())),
            Expr::arctg(Box::new(x.clone())),
            Expr::arcctg(Box::new(x.clone())),
        ];

        let module = CodegenModule::new("inverse_trig")
            .with_language(CodegenLanguage::Zig)
            .add_block("inverse_trig_funcs", &exprs, &["x"]);

        let zig_source = module.emit_source();

        println!("=== Inverse Trigonometric Functions Zig Code ===");
        println!("{}", zig_source);

        assert!(zig_source.contains("std.math.asin("));
        assert!(zig_source.contains("std.math.acos("));
        assert!(zig_source.contains("std.math.atan("));
    }

    #[test]
    fn example_special_values() {
        let exprs = vec![
            Expr::Const(f64::NAN),
            Expr::Const(f64::INFINITY),
            Expr::Const(f64::NEG_INFINITY),
        ];

        let module = CodegenModule::new("special")
            .with_language(CodegenLanguage::Zig)
            .add_block("special_values", &exprs, &[]);

        let zig_source = module.emit_source();

        println!("=== Special Values Zig Code ===");
        println!("{}", zig_source);

        assert!(zig_source.contains("std.math.nan(f64)"));
        assert!(zig_source.contains("std.math.inf(f64)"));
    }
}
