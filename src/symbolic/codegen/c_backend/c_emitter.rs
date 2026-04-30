//! C code emission from lowered IR.
//!
//! This module mirrors the Rust emitter but generates C99-compatible source code.
//! It reuses the same IR infrastructure (`Instr`, `LinearExpr`, `LinearBlock`)
//! and applies the same optimization passes, only differing in the final
//! code emission stage.
//!
//! Key differences from Rust emission:
//! - C requires explicit variable declarations
//! - C uses `log()` instead of `ln()`
//! - C requires `#include <math.h>` for mathematical functions
//! - C uses array indexing for inputs/outputs
//! - C requires const correctness for input pointers

use super::super::CodegenIR::{Instr, LinearBlock, LinearExpr, Temp};
use std::f64::consts::PI;

/// Target language for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenLanguage {
    Rust,
    C,
}

/// Emits C source code from lowered IR.
pub struct CEmitter;

impl CEmitter {
    /// Validates that an identifier is a valid C identifier.
    fn validate_identifier(identifier: &str, kind: &str) {
        let mut chars = identifier.chars();
        let Some(first) = chars.next() else {
            panic!("{kind} identifier must not be empty");
        };

        let valid_start = first == '_' || first.is_ascii_alphabetic();
        let valid_rest = chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric());

        assert!(
            valid_start && valid_rest,
            "invalid {kind} identifier: {identifier}"
        );
    }

    /// Returns the C variable name for a temporary register.
    fn temp_name(t: Temp) -> String {
        format!("t{}", t.0)
    }

    /// Formats a floating-point value as a C literal.
    fn fmt_f64(x: f64) -> String {
        if x.is_nan() {
            "NAN".to_string()
        } else if x == f64::INFINITY {
            "INFINITY".to_string()
        } else if x == f64::NEG_INFINITY {
            "(-INFINITY)".to_string()
        } else {
            format!("{:.17e}", x)
        }
    }

    /// Returns true when emitted IR actually reads from the `inputs` pointer.
    fn ir_uses_inputs(instructions: &[Instr]) -> bool {
        instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Input { .. }))
    }

    /// Emits a C function that returns a single scalar value.
    ///
    /// Signature: `double function_name(const double* inputs)`
    pub fn emit_function(ir: &LinearExpr, fn_name: &str, arity: usize) -> String {
        Self::validate_identifier(fn_name, "function");

        let mut out = String::new();

        // Function signature
        out.push_str(&format!("double {}(const double* inputs) {{\n", fn_name));

        // Variable declarations
        if ir.num_temps > 0 {
            out.push_str("    double ");
            for i in 0..ir.num_temps {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Self::temp_name(Temp(i)));
            }
            out.push_str(";\n");
        }

        if arity > 0 && !Self::ir_uses_inputs(&ir.instructions) {
            out.push_str("    (void)inputs;\n");
        }

        // Instructions
        for instr in &ir.instructions {
            Self::emit_instruction(instr, &mut out);
        }

        // Return statement
        out.push_str(&format!("    return {};\n", Self::temp_name(ir.output)));
        out.push_str("}\n");
        out
    }

    /// Emits a C function that writes multiple outputs to an array.
    ///
    /// Signature: `void function_name(const double* inputs, double* outputs)`
    pub fn emit_block_function(ir: &LinearBlock, fn_name: &str, arity: usize) -> String {
        Self::validate_identifier(fn_name, "function");

        let mut out = String::new();

        // Function signature
        out.push_str(&format!(
            "void {}(const double* inputs, double* outputs) {{\n",
            fn_name
        ));

        // Variable declarations
        if ir.num_temps > 0 {
            out.push_str("    double ");
            for i in 0..ir.num_temps {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Self::temp_name(Temp(i)));
            }
            out.push_str(";\n");
        }

        if arity > 0 && !Self::ir_uses_inputs(&ir.instructions) {
            out.push_str("    (void)inputs;\n");
        }

        // Instructions
        for instr in &ir.instructions {
            Self::emit_instruction(instr, &mut out);
        }

        // Output assignments
        for (index, output) in ir.outputs.iter().enumerate() {
            out.push_str(&format!(
                "    outputs[{}] = {};\n",
                index,
                Self::temp_name(*output)
            ));
        }

        out.push_str("}\n");
        out
    }

    /// Emits a residual block function with metadata comment.
    pub fn emit_residual_block_function(
        ir: &LinearBlock,
        fn_name: &str,
        arity: usize,
        len: usize,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!("/* Residual block: {} outputs */\n", len));
        out.push_str(&Self::emit_block_function(ir, fn_name, arity));
        out
    }

    /// Emits a dense Jacobian block function with metadata comment.
    pub fn emit_dense_jacobian_block_function(
        ir: &LinearBlock,
        fn_name: &str,
        arity: usize,
        rows: usize,
        cols: usize,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "/* Dense Jacobian block: {} rows x {} cols */\n",
            rows, cols
        ));
        out.push_str(&Self::emit_block_function(ir, fn_name, arity));
        out
    }

    /// Emits a sparse Jacobian values block function with metadata comment.
    pub fn emit_sparse_values_block_function(
        ir: &LinearBlock,
        fn_name: &str,
        arity: usize,
        rows: usize,
        cols: usize,
        nnz: usize,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "/* Sparse Jacobian values block: {} rows x {} cols, {} non-zero values */\n",
            rows, cols, nnz
        ));
        out.push_str(&Self::emit_block_function(ir, fn_name, arity));
        out
    }

    /// Emits a single IR instruction as C code.
    fn emit_instruction(instr: &Instr, out: &mut String) {
        match instr {
            Instr::Const { dst, value } => {
                out.push_str(&format!(
                    "    {} = {};\n",
                    Self::temp_name(*dst),
                    Self::fmt_f64(*value)
                ));
            }
            Instr::Input { dst, index } => {
                out.push_str(&format!(
                    "    {} = inputs[{}];\n",
                    Self::temp_name(*dst),
                    index
                ));
            }

            Instr::Add { dst, a, b } => {
                out.push_str(&format!(
                    "    {} = {} + {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Sub { dst, a, b } => {
                out.push_str(&format!(
                    "    {} = {} - {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Mul { dst, a, b } => {
                out.push_str(&format!(
                    "    {} = {} * {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Div { dst, a, b } => {
                out.push_str(&format!(
                    "    {} = {} / {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Pow { dst, base, exp } => {
                out.push_str(&format!(
                    "    {} = pow({}, {});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*base),
                    Self::temp_name(*exp),
                ));
            }

            Instr::Exp { dst, x } => {
                out.push_str(&format!(
                    "    {} = exp({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Ln { dst, x } => {
                // C uses log() for natural logarithm
                out.push_str(&format!(
                    "    {} = log({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Sin { dst, x } => {
                out.push_str(&format!(
                    "    {} = sin({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Cos { dst, x } => {
                out.push_str(&format!(
                    "    {} = cos({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Tg { dst, x } => {
                out.push_str(&format!(
                    "    {} = tan({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Ctg { dst, x } => {
                out.push_str(&format!(
                    "    {} = 1.0 / tan({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcSin { dst, x } => {
                out.push_str(&format!(
                    "    {} = asin({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcCos { dst, x } => {
                out.push_str(&format!(
                    "    {} = acos({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcTg { dst, x } => {
                out.push_str(&format!(
                    "    {} = atan({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcCtg { dst, x } => {
                out.push_str(&format!(
                    "    {} = {} - atan({});\n",
                    Self::temp_name(*dst),
                    Self::fmt_f64(PI / 2.0),
                    Self::temp_name(*x)
                ));
            }
        }
    }

    /// Generates a complete C header file with function declarations.
    pub fn emit_header(function_names: &[(String, bool)]) -> String {
        let mut out = String::new();

        out.push_str("#ifndef GENERATED_FUNCTIONS_H\n");
        out.push_str("#define GENERATED_FUNCTIONS_H\n\n");
        out.push_str("#include <math.h>\n\n");
        out.push_str("#ifdef __cplusplus\n");
        out.push_str("extern \"C\" {\n");
        out.push_str("#endif\n\n");

        for (name, is_block) in function_names {
            if *is_block {
                out.push_str(&format!(
                    "void {}(const double* inputs, double* outputs);\n",
                    name
                ));
            } else {
                out.push_str(&format!("double {}(const double* inputs);\n", name));
            }
        }

        out.push_str("\n#ifdef __cplusplus\n");
        out.push_str("}\n");
        out.push_str("#endif\n\n");
        out.push_str("#endif /* GENERATED_FUNCTIONS_H */\n");
        out
    }

    /// Generates a complete C source file with includes.
    pub fn emit_source_file(functions: &str) -> String {
        let mut out = String::new();

        out.push_str("/* ========================================= */\n");
        out.push_str("/* AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. */\n");
        out.push_str("/* ========================================= */\n\n");
        out.push_str("#include <math.h>\n\n");
        out.push_str(functions);

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn emit_simple_constant() {
        let expr = Expr::Const(42.0);
        let ir = expr.lower_to_linear(&[]);
        let src = CEmitter::emit_function(&ir, "get_constant", 0);

        assert!(src.contains("double get_constant(const double* inputs)"));
        assert!(src.contains("t0 = 4.2"));
        assert!(src.contains("return t0;"));
    }

    #[test]
    fn emit_simple_variable() {
        let expr = Expr::Var("x".to_string());
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "identity", 1);

        assert!(src.contains("double identity(const double* inputs)"));
        assert!(src.contains("t0 = inputs[0];"));
        assert!(src.contains("return t0;"));
    }

    #[test]
    fn emit_addition() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::Const(1.0);
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "add_one", 1);

        assert!(src.contains("double add_one(const double* inputs)"));
        assert!(src.contains("t0 = inputs[0];"));
        assert!(src.contains("t1 = 1."));
        assert!(src.contains("t2 = t0 + t1;"));
        assert!(src.contains("return t2;"));
    }

    #[test]
    fn emit_multiplication() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = x.clone() * y.clone();
        let ir = expr.lower_to_linear(&["x", "y"]);
        let src = CEmitter::emit_function(&ir, "multiply", 2);

        assert!(src.contains("t0 = inputs[0];"));
        assert!(src.contains("t1 = inputs[1];"));
        assert!(src.contains("t2 = t0 * t1;"));
    }

    #[test]
    fn emit_power_function() {
        let x = Expr::Var("x".to_string());
        let expr = x.pow(Expr::Const(2.0));
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "square", 1);

        assert!(src.contains("pow(t0, t1)"));
    }

    #[test]
    fn emit_natural_logarithm_uses_log() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Ln(Box::new(x));
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "natural_log", 1);

        // C uses log() not ln()
        assert!(src.contains("log(t0)"));
        assert!(!src.contains("ln("));
    }

    #[test]
    fn emit_exponential() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Exp(Box::new(x));
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "exponential", 1);

        assert!(src.contains("exp(t0)"));
    }

    #[test]
    fn emit_trigonometric_functions() {
        let x = Expr::Var("x".to_string());

        let sin_expr = Expr::sin(Box::new(x.clone()));
        let sin_ir = sin_expr.lower_to_linear(&["x"]);
        let sin_src = CEmitter::emit_function(&sin_ir, "sine", 1);
        assert!(sin_src.contains("sin(t0)"));

        let cos_expr = Expr::cos(Box::new(x.clone()));
        let cos_ir = cos_expr.lower_to_linear(&["x"]);
        let cos_src = CEmitter::emit_function(&cos_ir, "cosine", 1);
        assert!(cos_src.contains("cos(t0)"));

        let tan_expr = Expr::tg(Box::new(x.clone()));
        let tan_ir = tan_expr.lower_to_linear(&["x"]);
        let tan_src = CEmitter::emit_function(&tan_ir, "tangent", 1);
        assert!(tan_src.contains("tan(t0)"));
    }

    #[test]
    fn emit_inverse_trigonometric_functions() {
        let x = Expr::Var("x".to_string());

        let asin_expr = Expr::arcsin(Box::new(x.clone()));
        let asin_ir = asin_expr.lower_to_linear(&["x"]);
        let asin_src = CEmitter::emit_function(&asin_ir, "arcsine", 1);
        assert!(asin_src.contains("asin(t0)"));

        let acos_expr = Expr::arccos(Box::new(x.clone()));
        let acos_ir = acos_expr.lower_to_linear(&["x"]);
        let acos_src = CEmitter::emit_function(&acos_ir, "arccosine", 1);
        assert!(acos_src.contains("acos(t0)"));

        let atan_expr = Expr::arctg(Box::new(x.clone()));
        let atan_ir = atan_expr.lower_to_linear(&["x"]);
        let atan_src = CEmitter::emit_function(&atan_ir, "arctangent", 1);
        assert!(atan_src.contains("atan(t0)"));
    }

    #[test]
    fn emit_cotangent() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::ctg(Box::new(x));
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "cotangent", 1);

        assert!(src.contains("1.0 / tan(t0)"));
    }

    #[test]
    fn emit_arccotangent() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::arcctg(Box::new(x));
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "arccotangent", 1);

        assert!(src.contains("- atan(t0)"));
    }

    #[test]
    fn emit_special_float_values() {
        assert_eq!(CEmitter::fmt_f64(f64::NAN), "NAN");
        assert_eq!(CEmitter::fmt_f64(f64::INFINITY), "INFINITY");
        assert_eq!(CEmitter::fmt_f64(f64::NEG_INFINITY), "(-INFINITY)");
    }

    #[test]
    fn emit_block_function_with_multiple_outputs() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone() + Expr::Const(1.0), x.clone() * Expr::Const(2.0)];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let src = CEmitter::emit_block_function(&block, "compute_block", 1);

        assert!(src.contains("void compute_block(const double* inputs, double* outputs)"));
        assert!(src.contains("outputs[0] = "));
        assert!(src.contains("outputs[1] = "));
    }

    #[test]
    fn emit_residual_block_includes_metadata() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone(), x.clone() + Expr::Const(1.0)];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let src = CEmitter::emit_residual_block_function(&block, "residual", 1, 2);

        assert!(src.contains("/* Residual block: 2 outputs */"));
        assert!(src.contains("void residual(const double* inputs, double* outputs)"));
    }

    #[test]
    fn emit_dense_jacobian_includes_metadata() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone(), x.clone() * Expr::Const(2.0)];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let src = CEmitter::emit_dense_jacobian_block_function(&block, "jacobian", 1, 2, 1);

        assert!(src.contains("/* Dense Jacobian block: 2 rows x 1 cols */"));
    }

    #[test]
    fn emit_sparse_values_includes_metadata() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone()];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let src = CEmitter::emit_sparse_values_block_function(&block, "sparse_vals", 1, 3, 3, 1);

        assert!(
            src.contains("/* Sparse Jacobian values block: 3 rows x 3 cols, 1 non-zero values */")
        );
    }

    #[test]
    fn emit_header_file() {
        let functions = vec![
            ("scalar_func".to_string(), false),
            ("block_func".to_string(), true),
        ];
        let header = CEmitter::emit_header(&functions);

        assert!(header.contains("#ifndef GENERATED_FUNCTIONS_H"));
        assert!(header.contains("#define GENERATED_FUNCTIONS_H"));
        assert!(header.contains("#include <math.h>"));
        assert!(header.contains("double scalar_func(const double* inputs);"));
        assert!(header.contains("void block_func(const double* inputs, double* outputs);"));
        assert!(header.contains("#ifdef __cplusplus"));
        assert!(header.contains("extern \"C\" {"));
    }

    #[test]
    fn emit_source_file_includes_header() {
        let functions = "double test(const double* inputs) { return inputs[0]; }\n";
        let source = CEmitter::emit_source_file(functions);

        assert!(source.contains("/* AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. */"));
        assert!(source.contains("#include <math.h>"));
        assert!(source.contains("double test(const double* inputs)"));
    }

    #[test]
    fn variable_declarations_grouped_on_one_line() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + x.clone() + x.clone();
        let ir = expr.lower_to_linear(&["x"]);
        let src = CEmitter::emit_function(&ir, "triple", 1);

        // Should have one declaration line with multiple variables
        assert!(src.contains("double t0, t1, t2"));
    }

    #[test]
    #[should_panic(expected = "invalid function identifier")]
    fn validate_identifier_rejects_invalid_names() {
        let expr = Expr::Const(1.0);
        let ir = expr.lower_to_linear(&[]);
        CEmitter::emit_function(&ir, "123invalid", 0);
    }

    #[test]
    fn complex_expression_emits_valid_c() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = (x.clone() + y.clone()).pow(Expr::Const(2.0)) + Expr::sin(Box::new(x.clone()))
            - Expr::Const(3.0) * y.clone();
        let ir = expr.lower_to_linear(&["x", "y"]);
        let src = CEmitter::emit_function(&ir, "complex_eval", 2);

        assert!(src.contains("double complex_eval(const double* inputs)"));
        assert!(src.contains("pow("));
        assert!(src.contains("sin("));
        assert!(src.contains("return"));
    }
}
