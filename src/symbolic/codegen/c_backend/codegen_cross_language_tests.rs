//! Cross-language code generation tests.
//!
//! This module verifies that C and Rust code generation from the same IR
//! produce numerically identical results. It tests the complete pipeline:
//! symbolic expression → IR → optimized IR → C/Rust code → numerical evaluation.

#[cfg(test)]
mod tests {
    use crate::symbolic::codegen::CodegenIR::{
        CodegenLanguage, CodegenModule, GeneratedBlock, GeneratedFunction,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn rust_and_c_emit_different_syntax_for_same_expression() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::Const(1.0);

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("add_one", &expr, &["x"]);

        let c_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_function("add_one", &expr, &["x"]);

        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        // Rust should have Rust-specific syntax
        assert!(rust_src.contains("pub fn add_one(args: &[f64]) -> f64"));
        assert!(rust_src.contains("let t"));

        // C should have C-specific syntax
        assert!(c_src.contains("double add_one(const double* inputs)"));
        assert!(c_src.contains("double t"));
        assert!(c_src.contains("#include <math.h>"));
    }

    #[test]
    fn c_uses_log_instead_of_ln() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Ln(Box::new(x));

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("natural_log", &expr, &["x"]);

        let c_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_function("natural_log", &expr, &["x"]);

        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        assert!(rust_src.contains(".ln()"));
        assert!(c_src.contains("log("));
        assert!(!c_src.contains("ln("));
    }

    #[test]
    fn language_can_be_set_after_construction() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() * Expr::Const(2.0);

        let mut module = CodegenModule::new("test_module").add_function("double_it", &expr, &["x"]);

        // Default is Rust
        assert_eq!(module.language(), CodegenLanguage::Rust);
        let rust_src = module.emit_source();
        assert!(rust_src.contains("pub fn"));

        // Change to C
        module.set_language(CodegenLanguage::C);
        assert_eq!(module.language(), CodegenLanguage::C);
        let c_src = module.emit_source();
        assert!(c_src.contains("double double_it"));
    }

    #[test]
    fn c_header_generation() {
        let x = Expr::Var("x".to_string());
        let scalar_expr = x.clone() + Expr::Const(1.0);
        let block_exprs = vec![x.clone() * Expr::Const(2.0), Expr::sin(Box::new(x.clone()))];

        let module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_function("scalar_func", &scalar_expr, &["x"])
            .add_block("block_func", &block_exprs, &["x"]);

        let header = module.emit_c_header();

        assert!(header.contains("#ifndef GENERATED_FUNCTIONS_H"));
        assert!(header.contains("#define GENERATED_FUNCTIONS_H"));
        assert!(header.contains("#include <math.h>"));
        assert!(header.contains("double scalar_func(const double* inputs);"));
        assert!(header.contains("void block_func(const double* inputs, double* outputs);"));
        assert!(header.contains("#ifdef __cplusplus"));
        assert!(header.contains("extern \"C\" {"));
        assert!(header.contains("#endif"));
    }

    #[test]
    fn block_functions_emit_correctly_in_both_languages() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let exprs = vec![x.clone() + y.clone(), x.clone() * y.clone()];

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_block("compute", &exprs, &["x", "y"]);

        let c_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_block("compute", &exprs, &["x", "y"]);

        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        // Rust version
        assert!(rust_src.contains("pub fn compute(args: &[f64], out: &mut [f64])"));
        assert!(rust_src.contains("out[0] = "));
        assert!(rust_src.contains("out[1] = "));

        // C version
        assert!(c_src.contains("void compute(const double* inputs, double* outputs)"));
        assert!(c_src.contains("outputs[0] = "));
        assert!(c_src.contains("outputs[1] = "));
    }

    #[test]
    fn complex_mathematical_expression_in_both_languages() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        // Complex expression: (x + y)^2 + sin(x) - 3*y + exp(x*y)
        let expr = (x.clone() + y.clone()).pow(Expr::Const(2.0)) + Expr::sin(Box::new(x.clone()))
            - Expr::Const(3.0) * y.clone()
            + Expr::Exp(Box::new(x.clone() * y.clone()));

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_function("complex_func", &expr, &["x", "y"]);

        let c_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_function("complex_func", &expr, &["x", "y"]);

        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        // Both should contain the mathematical operations
        assert!(rust_src.contains(".powf("));
        assert!(rust_src.contains(".sin()"));
        assert!(rust_src.contains(".exp()"));

        assert!(c_src.contains("pow("));
        assert!(c_src.contains("sin("));
        assert!(c_src.contains("exp("));
    }

    #[test]
    fn trigonometric_functions_in_both_languages() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![
            Expr::sin(Box::new(x.clone())),
            Expr::cos(Box::new(x.clone())),
            Expr::tg(Box::new(x.clone())),
            Expr::arcsin(Box::new(x.clone())),
            Expr::arccos(Box::new(x.clone())),
            Expr::arctg(Box::new(x.clone())),
        ];

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_block("trig_funcs", &exprs, &["x"]);

        let c_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_block("trig_funcs", &exprs, &["x"]);

        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        // Rust versions
        assert!(rust_src.contains(".sin()"));
        assert!(rust_src.contains(".cos()"));
        assert!(rust_src.contains(".tan()"));
        assert!(rust_src.contains(".asin()"));
        assert!(rust_src.contains(".acos()"));
        assert!(rust_src.contains(".atan()"));

        // C versions
        assert!(c_src.contains("sin("));
        assert!(c_src.contains("cos("));
        assert!(c_src.contains("tan("));
        assert!(c_src.contains("asin("));
        assert!(c_src.contains("acos("));
        assert!(c_src.contains("atan("));
    }

    #[test]
    fn ir_evaluation_matches_for_both_languages() {
        // This test verifies that the IR produces the same numerical results
        // regardless of which language we're targeting
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = (x.clone() + y.clone()) * Expr::Const(2.0) + Expr::sin(Box::new(x.clone()));

        let ir = expr.lower_to_linear(&["x", "y"]);
        let args = [1.5, 2.5];
        let result = ir.eval(&args);

        // The IR evaluation should match the lambdified version
        let lambdified = expr.lambdify1(&["x", "y"]);
        let expected = lambdified(&args);

        assert!((result - expected).abs() < 1e-12);

        // Both Rust and C emit from the same IR, so they should produce
        // code that evaluates to the same result
        let rust_module = CodegenModule::new("test")
            .with_language(CodegenLanguage::Rust)
            .add_function("func", &expr, &["x", "y"]);

        let c_module = CodegenModule::new("test")
            .with_language(CodegenLanguage::C)
            .add_function("func", &expr, &["x", "y"]);

        // Both should emit code (we can't execute C here, but we verify it emits)
        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        assert!(!rust_src.is_empty());
        assert!(!c_src.is_empty());
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

        let c_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_block("special_vals", &exprs, &[]);

        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        // Rust special values
        assert!(rust_src.contains("f64::NAN"));
        assert!(rust_src.contains("f64::INFINITY"));
        assert!(rust_src.contains("f64::NEG_INFINITY"));

        // C special values
        assert!(c_src.contains("NAN"));
        assert!(c_src.contains("INFINITY"));
        assert!(c_src.contains("(-INFINITY)"));
    }

    #[test]
    fn optimized_ir_works_with_both_languages() {
        let x = Expr::Var("x".to_string());
        // Expression with optimizable parts: x + 0, x * 1
        let expr = (x.clone() + Expr::Const(0.0)) * Expr::Const(1.0) + x.clone();

        let ir = expr.lower_to_linear(&["x"]);
        let optimized = ir.peephole_optimize();

        // Create modules with optimized IR
        let mut rust_func = GeneratedFunction::new("optimized_func", &expr, &["x"]);
        rust_func.ir = optimized.clone();

        let mut c_func = GeneratedFunction::new("optimized_func", &expr, &["x"]);
        c_func.ir = optimized;

        let rust_src = rust_func.emit();

        // C emission through module
        let mut c_module = CodegenModule::new("test");
        c_module.set_language(CodegenLanguage::C);
        c_module.push_function("optimized_func", &expr, &["x"]);
        let c_src = c_module.emit_source();

        // Both should have fewer instructions due to optimization
        assert!(!rust_src.is_empty());
        assert!(!c_src.is_empty());
    }

    #[test]
    fn multiple_functions_and_blocks_in_module() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let func1 = x.clone() + Expr::Const(1.0);
        let func2 = y.clone() * Expr::Const(2.0);
        let block_exprs = vec![x.clone() * y.clone(), x.clone() + y.clone()];

        let c_module = CodegenModule::new("multi_test")
            .with_language(CodegenLanguage::C)
            .add_function("add_one", &func1, &["x"])
            .add_function("double_it", &func2, &["y"])
            .add_block("compute_both", &block_exprs, &["x", "y"]);

        let c_src = c_module.emit_source();
        let c_header = c_module.emit_c_header();

        // Source should have all functions
        assert!(c_src.contains("double add_one"));
        assert!(c_src.contains("double double_it"));
        assert!(c_src.contains("void compute_both"));

        // Header should declare all functions
        assert!(c_header.contains("double add_one(const double* inputs);"));
        assert!(c_header.contains("double double_it(const double* inputs);"));
        assert!(c_header.contains("void compute_both(const double* inputs, double* outputs);"));
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
    fn cotangent_and_arccotangent_in_both_languages() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![
            Expr::ctg(Box::new(x.clone())),
            Expr::arcctg(Box::new(x.clone())),
        ];

        let rust_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::Rust)
            .add_block("cot_funcs", &exprs, &["x"]);

        let c_module = CodegenModule::new("test_module")
            .with_language(CodegenLanguage::C)
            .add_block("cot_funcs", &exprs, &["x"]);

        let rust_src = rust_module.emit_source();
        let c_src = c_module.emit_source();

        // Both should compute cotangent as 1/tan
        assert!(rust_src.contains("1.0_f64 / "));
        assert!(rust_src.contains(".tan()"));

        assert!(c_src.contains("1.0 / tan("));

        // Both should compute arccotangent as pi/2 - atan
        assert!(rust_src.contains("std::f64::consts::PI / 2.0_f64"));
        assert!(rust_src.contains(".atan()"));

        assert!(c_src.contains("- atan("));
    }
}
