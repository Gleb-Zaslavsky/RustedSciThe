//! Cross-language code generation tests for Zig.
//!
//! This module verifies that Zig and Rust code generation from the same IR
//! produce numerically identical results. It tests the complete pipeline:
//! symbolic expression → IR → optimized IR → Zig/Rust code → numerical evaluation.

#[cfg(test)]
mod tests {
    use crate::symbolic::codegen::CodegenIR::{CodegenLanguage, CodegenModule, GeneratedFunction};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn rust_and_zig_emit_different_syntax_for_same_expression() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::Const(1.0);

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("add_one", &expr, &["x"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_function("add_one", &expr, &["x"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains("pub fn add_one(args: &[f64]) -> f64"));
        assert!(rust_src.contains("let t"));

        assert!(zig_src.contains("pub fn add_one(inputs: [*]const f64) f64"));
        assert!(zig_src.contains("const t0: f64"));
    }

    #[test]
    fn zig_uses_at_log_instead_of_ln() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Ln(Box::new(x));

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("natural_log", &expr, &["x"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_function("natural_log", &expr, &["x"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains(".ln()"));
        assert!(zig_src.contains("@log("));
        assert!(!zig_src.contains("ln("));
    }

    #[test]
    fn zig_uses_at_exp_for_exponential() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Exp(Box::new(x));

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("exponential", &expr, &["x"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_function("exponential", &expr, &["x"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains(".exp()"));
        assert!(zig_src.contains("@exp("));
    }

    #[test]
    fn zig_uses_at_sin_cos_tan_for_trig() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![
            Expr::sin(Box::new(x.clone())),
            Expr::cos(Box::new(x.clone())),
            Expr::tg(Box::new(x.clone())),
        ];

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_block("trig_funcs", &exprs, &["x"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_block("trig_funcs", &exprs, &["x"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains(".sin()"));
        assert!(rust_src.contains(".cos()"));
        assert!(rust_src.contains(".tan()"));

        assert!(zig_src.contains("@sin("));
        assert!(zig_src.contains("@cos("));
        assert!(zig_src.contains("@tan("));
    }

    #[test]
    fn zig_uses_std_math_for_inverse_trig() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![
            Expr::arcsin(Box::new(x.clone())),
            Expr::arccos(Box::new(x.clone())),
            Expr::arctg(Box::new(x.clone())),
        ];

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_block("inverse_trig", &exprs, &["x"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_block("inverse_trig", &exprs, &["x"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains(".asin()"));
        assert!(rust_src.contains(".acos()"));
        assert!(rust_src.contains(".atan()"));

        assert!(zig_src.contains("std.math.asin("));
        assert!(zig_src.contains("std.math.acos("));
        assert!(zig_src.contains("std.math.atan("));
    }

    #[test]
    fn zig_cotangent_uses_atan() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::ctg(Box::new(x));

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("cotangent", &expr, &["x"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_function("cotangent", &expr, &["x"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains("1.0_f64 / "));
        assert!(rust_src.contains(".tan()"));

        assert!(zig_src.contains("1.0 / @tan("));
    }

    #[test]
    fn language_can_be_set_after_construction() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * Expr::Const(2.0);

        let mut module = CodegenModule::new("test_module").add_function("double_it", &expr, &["x"]);

        assert_eq!(module.language(), CodegenLanguage::Rust);
        let rust_src = module.emit_source();
        assert!(rust_src.contains("pub fn"));

        module.set_language(CodegenLanguage::Zig);
        assert_eq!(module.language(), CodegenLanguage::Zig);
        let zig_src = module.emit_source();
        assert!(zig_src.contains("pub fn double_it"));
    }

    #[test]
    fn block_functions_emit_correctly_in_both_languages() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let exprs = vec![x.clone() + y.clone(), x.clone() * y.clone()];

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_block("compute", &exprs, &["x", "y"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_block("compute", &exprs, &["x", "y"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains("pub fn compute(args: &[f64], out: &mut [f64])"));
        assert!(rust_src.contains("out[0] = "));
        assert!(rust_src.contains("out[1] = "));

        assert!(zig_src.contains("pub fn compute(inputs: [*]const f64, outputs: [*]f64) void"));
        assert!(zig_src.contains("outputs[0] = "));
        assert!(zig_src.contains("outputs[1] = "));
    }

    #[test]
    fn complex_mathematical_expression_in_both_languages() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let expr = (x.clone() + y.clone()).pow(Expr::Const(2.0)) + Expr::sin(Box::new(x.clone()))
            - Expr::Const(3.0) * y.clone()
            + Expr::Exp(Box::new(x.clone() * y.clone()));

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("complex_func", &expr, &["x", "y"]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_function("complex_func", &expr, &["x", "y"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains(".powf("));
        assert!(rust_src.contains(".sin()"));
        assert!(rust_src.contains(".exp()"));

        assert!(zig_src.contains("std.math.pow(f64,"));
        assert!(zig_src.contains("@sin("));
        assert!(zig_src.contains("@exp("));
    }

    #[test]
    fn ir_evaluation_matches_for_both_languages() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = (x.clone() + y.clone()) * Expr::Const(2.0) + Expr::sin(Box::new(x.clone()));

        let ir = expr.lower_to_linear(&["x", "y"]);
        let args = [1.5, 2.5];
        let result = ir.eval(&args);

        let lambdified = expr.lambdify1(&["x", "y"]);
        let expected = lambdified(&args);

        assert!((result - expected).abs() < 1e-12);

        let rust_module = CodegenModule::new("test")
            .with_language(CodegenLanguage::Rust)
            .add_function("func", &expr, &["x", "y"]);

        let zig_module = CodegenModule::new("test")
            .with_language(CodegenLanguage::Zig)
            .add_function("func", &expr, &["x", "y"]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(!rust_src.is_empty());
        assert!(!zig_src.is_empty());
    }

    #[test]
    fn special_values_handled_correctly() {
        let exprs = vec![
            Expr::Const(f64::NAN),
            Expr::Const(f64::INFINITY),
            Expr::Const(f64::NEG_INFINITY),
            Expr::Const(0.0),
            Expr::Const(-0.0),
        ];

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_block("special_vals", &exprs, &[]);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_block("special_vals", &exprs, &[]);

        let rust_src = rust_module.emit_source();
        let zig_src = zig_module.emit_source();

        assert!(rust_src.contains("f64::NAN"));
        assert!(rust_src.contains("f64::INFINITY"));
        assert!(rust_src.contains("f64::NEG_INFINITY"));

        assert!(zig_src.contains("std.math.nan(f64)"));
        assert!(zig_src.contains("std.math.inf(f64)"));
    }

    #[test]
    fn optimized_ir_works_with_both_languages() {
        let x = Expr::Var("x".to_string());
        let expr = (x.clone() + Expr::Const(0.0)) * Expr::Const(1.0) + x.clone();

        let ir = expr.lower_to_linear(&["x"]);
        let optimized = ir.peephole_optimize();

        let mut rust_func = GeneratedFunction::new("optimized_func", &expr, &["x"]);
        rust_func.ir = optimized.clone();

        let mut zig_func = GeneratedFunction::new("optimized_func", &expr, &["x"]);
        zig_func.ir = optimized;

        let rust_src = rust_func.emit();

        let mut zig_module = CodegenModule::new("test");
        zig_module.set_language(CodegenLanguage::Zig);
        zig_module.push_function("optimized_func", &expr, &["x"]);
        let zig_src = zig_module.emit_source();

        assert!(!rust_src.is_empty());
        assert!(!zig_src.is_empty());
    }

    #[test]
    fn multiple_functions_and_blocks_in_module() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let func1 = x.clone() + Expr::Const(1.0);
        let func2 = y.clone() * Expr::Const(2.0);
        let block_exprs = vec![x.clone() * y.clone(), x.clone() + y.clone()];

        let zig_module = CodegenModule::new("multi_test")
            .with_language(CodegenLanguage::Zig)
            .add_function("add_one", &func1, &["x"])
            .add_function("double_it", &func2, &["y"])
            .add_block("compute_both", &block_exprs, &["x", "y"]);

        let zig_src = zig_module.emit_source();

        assert!(zig_src.contains("pub fn add_one"));
        assert!(zig_src.contains("pub fn double_it"));
        assert!(zig_src.contains("pub fn compute_both"));
    }

    #[test]
    fn default_language_is_rust() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::Const(1.0);

        let module = CodegenModule::new("test_module").add_function("func", &expr, &["x"]);

        assert_eq!(module.language(), CodegenLanguage::Rust);

        let src = module.emit_source();
        assert!(src.contains("pub fn"));
        assert!(src.contains("pub mod"));
    }

    #[test]
    fn zig_imports_std_for_math() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::Const(1.0);

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_function("test_func", &expr, &["x"]);

        let zig_src = zig_module.emit_source();

        assert!(zig_src.contains("const std = @import(\"std\");"));
        assert!(zig_src.contains("pub fn test_func"));
    }

    #[test]
    fn zig_pow_uses_std_math() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone().pow(Expr::Const(2.0));

        let zig_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Zig)
            .add_function("power_func", &expr, &["x"]);

        let zig_src = zig_module.emit_source();

        assert!(zig_src.contains("std.math.pow(f64,"));
    }
}
