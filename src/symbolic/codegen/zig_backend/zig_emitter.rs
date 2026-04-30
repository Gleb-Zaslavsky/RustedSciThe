//! Zig code emission from lowered IR.
//!
//! Mirrors `c_emitter.rs` but generates Zig source code.
//! Key differences from C emission:
//! - Zig uses `@import("std").math` for math functions
//! - Zig uses `@log` / `@exp` / `@sin` etc. builtins where available
//! - Temporary declarations use `const name: f64 = ...;`
//! - Function signatures use Zig syntax
//! - Zig uses `@import("std")` for math constants

use super::super::CodegenIR::{Instr, LinearBlock, LinearExpr, Temp};
use std::collections::HashSet;
use std::f64::consts::PI;

/// Emits Zig source code from lowered IR.
pub struct ZigEmitter;

impl ZigEmitter {
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

    fn temp_name(t: Temp) -> String {
        format!("t{}", t.0)
    }

    fn fmt_f64(x: f64) -> String {
        if x.is_nan() {
            "std.math.nan(f64)".to_string()
        } else if x == f64::INFINITY {
            "std.math.inf(f64)".to_string()
        } else if x == f64::NEG_INFINITY {
            "-std.math.inf(f64)".to_string()
        } else {
            format!("{:.17e}", x)
        }
    }

    /// Emits a Zig function returning a single scalar.
    ///
    /// Signature: `pub fn function_name(inputs: [*]const f64) f64`
    pub fn emit_function(ir: &LinearExpr, fn_name: &str, _arity: usize) -> String {
        Self::validate_identifier(fn_name, "function");
        let live = Self::live_instructions_for_expr(ir);
        let mut out = String::new();
        out.push_str(&format!(
            "pub fn {}(inputs: [*]const f64) f64 {{\n",
            fn_name
        ));
        if !Self::live_instructions_use_inputs(&live) {
            out.push_str("    _ = inputs;\n");
        }
        for instr in live {
            Self::emit_instruction(instr, &mut out);
        }
        out.push_str(&format!("    return {};\n", Self::temp_name(ir.output)));
        out.push_str("}\n");
        out
    }

    /// Emits a Zig function writing multiple outputs to a slice.
    ///
    /// Signature: `pub fn function_name(inputs: [*]const f64, outputs: [*]f64) void`
    pub fn emit_block_function(ir: &LinearBlock, fn_name: &str, _arity: usize) -> String {
        Self::validate_identifier(fn_name, "function");
        let live = Self::live_instructions_for_block(ir);
        let mut out = String::new();
        out.push_str(&format!(
            "pub fn {}(inputs: [*]const f64, outputs: [*]f64) void {{\n",
            fn_name
        ));
        if !Self::live_instructions_use_inputs(&live) {
            out.push_str("    _ = inputs;\n");
        }
        if ir.outputs.is_empty() {
            out.push_str("    _ = outputs;\n");
        }
        for instr in live {
            Self::emit_instruction(instr, &mut out);
        }
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

    fn live_instructions_for_expr<'a>(ir: &'a LinearExpr) -> Vec<&'a Instr> {
        Self::live_instructions(&ir.instructions, [ir.output])
    }

    fn live_instructions_for_block<'a>(ir: &'a LinearBlock) -> Vec<&'a Instr> {
        Self::live_instructions(&ir.instructions, ir.outputs.iter().copied())
    }

    fn live_instructions<'a>(
        instructions: &'a [Instr],
        roots: impl IntoIterator<Item = Temp>,
    ) -> Vec<&'a Instr> {
        let mut needed: HashSet<Temp> = roots.into_iter().collect();
        let mut kept_rev = Vec::new();

        for instr in instructions.iter().rev() {
            let dst = Self::instr_dst(instr);
            if needed.contains(&dst) {
                kept_rev.push(instr);
                for input in Self::instr_inputs(instr) {
                    needed.insert(input);
                }
            }
        }

        kept_rev.reverse();
        kept_rev
    }

    fn live_instructions_use_inputs(instructions: &[&Instr]) -> bool {
        instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Input { .. }))
    }

    fn instr_dst(instr: &Instr) -> Temp {
        match instr {
            Instr::Const { dst, .. }
            | Instr::Input { dst, .. }
            | Instr::Add { dst, .. }
            | Instr::Sub { dst, .. }
            | Instr::Mul { dst, .. }
            | Instr::Div { dst, .. }
            | Instr::Pow { dst, .. }
            | Instr::Exp { dst, .. }
            | Instr::Ln { dst, .. }
            | Instr::Sin { dst, .. }
            | Instr::Cos { dst, .. }
            | Instr::Tg { dst, .. }
            | Instr::Ctg { dst, .. }
            | Instr::ArcSin { dst, .. }
            | Instr::ArcCos { dst, .. }
            | Instr::ArcTg { dst, .. }
            | Instr::ArcCtg { dst, .. } => *dst,
        }
    }

    fn instr_inputs(instr: &Instr) -> Vec<Temp> {
        match instr {
            Instr::Const { .. } | Instr::Input { .. } => Vec::new(),
            Instr::Add { a, b, .. }
            | Instr::Sub { a, b, .. }
            | Instr::Mul { a, b, .. }
            | Instr::Div { a, b, .. }
            | Instr::Pow { base: a, exp: b, .. } => vec![*a, *b],
            Instr::Exp { x, .. }
            | Instr::Ln { x, .. }
            | Instr::Sin { x, .. }
            | Instr::Cos { x, .. }
            | Instr::Tg { x, .. }
            | Instr::Ctg { x, .. }
            | Instr::ArcSin { x, .. }
            | Instr::ArcCos { x, .. }
            | Instr::ArcTg { x, .. }
            | Instr::ArcCtg { x, .. } => vec![*x],
        }
    }

    pub fn emit_residual_block_function(
        ir: &LinearBlock,
        fn_name: &str,
        arity: usize,
        len: usize,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!("// Residual block: {} outputs\n", len));
        out.push_str(&Self::emit_block_function(ir, fn_name, arity));
        out
    }

    pub fn emit_dense_jacobian_block_function(
        ir: &LinearBlock,
        fn_name: &str,
        arity: usize,
        rows: usize,
        cols: usize,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "// Dense Jacobian block: {} rows x {} cols\n",
            rows, cols
        ));
        out.push_str(&Self::emit_block_function(ir, fn_name, arity));
        out
    }

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
            "// Sparse Jacobian values block: {} rows x {} cols, {} non-zero values\n",
            rows, cols, nnz
        ));
        out.push_str(&Self::emit_block_function(ir, fn_name, arity));
        out
    }

    fn emit_instruction(instr: &Instr, out: &mut String) {
        match instr {
            Instr::Const { dst, value } => {
                out.push_str(&format!(
                    "    const {}: f64 = {};\n",
                    Self::temp_name(*dst),
                    Self::fmt_f64(*value)
                ));
            }
            Instr::Input { dst, index } => {
                out.push_str(&format!(
                    "    const {}: f64 = inputs[{}];\n",
                    Self::temp_name(*dst),
                    index
                ));
            }
            Instr::Add { dst, a, b } => {
                out.push_str(&format!(
                    "    const {}: f64 = {} + {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Sub { dst, a, b } => {
                out.push_str(&format!(
                    "    const {}: f64 = {} - {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Mul { dst, a, b } => {
                out.push_str(&format!(
                    "    const {}: f64 = {} * {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Div { dst, a, b } => {
                out.push_str(&format!(
                    "    const {}: f64 = {} / {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Pow { dst, base, exp } => {
                out.push_str(&format!(
                    "    const {}: f64 = std.math.pow(f64, {}, {});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*base),
                    Self::temp_name(*exp),
                ));
            }
            Instr::Exp { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = @exp({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Ln { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = @log({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Sin { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = @sin({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Cos { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = @cos({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Tg { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = @tan({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Ctg { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = 1.0 / @tan({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcSin { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = std.math.asin({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcCos { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = std.math.acos({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcTg { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = std.math.atan({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcCtg { dst, x } => {
                out.push_str(&format!(
                    "    const {}: f64 = {} - std.math.atan({});\n",
                    Self::temp_name(*dst),
                    Self::fmt_f64(PI / 2.0),
                    Self::temp_name(*x)
                ));
            }
        }
    }

    /// Generates a complete Zig source file with exports.
    pub fn emit_source_file(functions: &str) -> String {
        let mut out = String::new();
        out.push_str("// =========================================\n");
        out.push_str("// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.\n");
        out.push_str("// =========================================\n\n");
        out.push_str("const std = @import(\"std\");\n\n");
        out.push_str(functions);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn emit_simple_variable() {
        let expr = Expr::Var("x".to_string());
        let ir = expr.lower_to_linear(&["x"]);
        let src = ZigEmitter::emit_function(&ir, "identity", 1);
        assert!(src.contains("pub fn identity(inputs: [*]const f64) f64"));
        assert!(src.contains("inputs[0]"));
        assert!(src.contains("return t0;"));
    }

    #[test]
    fn emit_natural_logarithm_uses_at_log() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Ln(Box::new(x));
        let ir = expr.lower_to_linear(&["x"]);
        let src = ZigEmitter::emit_function(&ir, "natural_log", 1);
        assert!(src.contains("@log(t0)"));
    }

    #[test]
    fn emit_exponential_uses_at_exp() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Exp(Box::new(x));
        let ir = expr.lower_to_linear(&["x"]);
        let src = ZigEmitter::emit_function(&ir, "exponential", 1);
        assert!(src.contains("@exp(t0)"));
    }

    #[test]
    fn emit_power_uses_std_math_pow() {
        let x = Expr::Var("x".to_string());
        let expr = x.pow(Expr::Const(2.0));
        let ir = expr.lower_to_linear(&["x"]);
        let src = ZigEmitter::emit_function(&ir, "square", 1);
        assert!(src.contains("std.math.pow(f64,"));
    }

    #[test]
    fn emit_trig_uses_at_builtins() {
        let x = Expr::Var("x".to_string());
        let sin_src = ZigEmitter::emit_function(
            &Expr::sin(Box::new(x.clone())).lower_to_linear(&["x"]),
            "sine",
            1,
        );
        assert!(sin_src.contains("@sin("));
        let cos_src = ZigEmitter::emit_function(
            &Expr::cos(Box::new(x.clone())).lower_to_linear(&["x"]),
            "cosine",
            1,
        );
        assert!(cos_src.contains("@cos("));
        let tan_src = ZigEmitter::emit_function(
            &Expr::tg(Box::new(x.clone())).lower_to_linear(&["x"]),
            "tangent",
            1,
        );
        assert!(tan_src.contains("@tan("));
    }

    #[test]
    fn emit_block_function_with_multiple_outputs() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone() + Expr::Const(1.0), x.clone() * Expr::Const(2.0)];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let src = ZigEmitter::emit_block_function(&block, "compute_block", 1);
        assert!(src.contains("pub fn compute_block(inputs: [*]const f64, outputs: [*]f64) void"));
        assert!(src.contains("outputs[0] = "));
        assert!(src.contains("outputs[1] = "));
    }

    #[test]
    fn emit_constant_scalar_marks_unused_inputs() {
        let expr = Expr::Const(3.0);
        let ir = expr.lower_to_linear(&["x"]);
        let src = ZigEmitter::emit_function(&ir, "constant_value", 1);
        assert!(src.contains("_ = inputs;"));
    }

    #[test]
    fn emit_constant_block_marks_unused_inputs() {
        let exprs = vec![Expr::Const(1.0), Expr::Const(2.0)];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let src = ZigEmitter::emit_block_function(&block, "constant_block", 1);
        assert!(src.contains("_ = inputs;"));
    }

    #[test]
    #[should_panic(expected = "invalid function identifier")]
    fn validate_identifier_rejects_invalid_names() {
        let expr = Expr::Const(1.0);
        let ir = expr.lower_to_linear(&[]);
        ZigEmitter::emit_function(&ir, "123invalid", 0);
    }
}
