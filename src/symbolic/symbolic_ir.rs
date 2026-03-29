//! Internal instruction-based compiler for symbolic lambdification.
//!
//! This module starts Phase 2A of the lambdify refactor. It provides a small,
//! batch-oriented intermediate representation (IR) that compiles several
//! symbolic expressions into one instruction tape.
//!
//! The current implementation is intentionally conservative:
//! - no common subexpression elimination yet;
//! - no sparse specialization yet;
//! - no public API outside the symbolic subsystem.
//!
//! Even in this minimal form it already gives us a stable execution model for
//! future optimizations such as batch compilation, CSE, and sparse Jacobian
//! evaluators.

use crate::symbolic::symbolic_engine::Expr;
use std::f64::consts::PI;

/// One instruction in the symbolic evaluation tape.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Instruction {
    LoadVar(usize),
    LoadConst(f64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Pow(usize, usize),
    Exp(usize),
    Ln(usize),
    Sin(usize),
    Cos(usize),
    Tg(usize),
    Ctg(usize),
    ArcSin(usize),
    ArcCos(usize),
    ArcTg(usize),
    ArcCtg(usize),
}

/// A compiled batch of symbolic expressions that share one instruction tape.
#[derive(Clone, Debug)]
pub(crate) struct CompiledBatch {
    /// Variable names in the expected input order.
    pub(crate) variable_names: Vec<String>,
    /// Flat instruction tape. Each instruction writes to its own register index.
    pub(crate) instructions: Vec<Instruction>,
    /// Register indices that store the final outputs.
    pub(crate) output_registers: Vec<usize>,
}

impl CompiledBatch {
    /// Evaluate all compiled outputs for one argument slice.
    pub(crate) fn eval(&self, args: &[f64]) -> Vec<f64> {
        let mut registers = Vec::new();
        let mut outputs = Vec::new();
        self.eval_into(args, &mut registers, &mut outputs);
        outputs
    }

    /// Evaluate all compiled outputs using caller-provided scratch buffers.
    ///
    /// This avoids per-call allocations and is intended for hot numerical paths.
    pub(crate) fn eval_into(&self, args: &[f64], registers: &mut Vec<f64>, outputs: &mut Vec<f64>) {
        self.evaluate_registers(args, registers);

        outputs.clear();
        outputs.reserve(self.output_registers.len());
        for &register in &self.output_registers {
            outputs.push(registers[register]);
        }
    }

    /// Evaluate a single-output batch.
    pub(crate) fn eval_single(&self, args: &[f64]) -> f64 {
        let mut registers = Vec::new();
        self.eval_single_with_scratch(args, &mut registers)
    }

    /// Evaluate a single-output batch using caller-provided register scratch.
    ///
    /// This is the allocation-free hot path used by `lambdify2()`.
    pub(crate) fn eval_single_with_scratch(&self, args: &[f64], registers: &mut Vec<f64>) -> f64 {
        debug_assert_eq!(
            self.output_registers.len(),
            1,
            "eval_single_with_scratch() expects exactly one compiled output"
        );
        self.evaluate_registers(args, registers);
        registers[self.output_registers[0]]
    }

    fn evaluate_registers(&self, args: &[f64], registers: &mut Vec<f64>) {
        if registers.len() < self.instructions.len() {
            registers.resize(self.instructions.len(), 0.0);
        }

        for (register, instruction) in self.instructions.iter().enumerate() {
            registers[register] = match *instruction {
                Instruction::LoadVar(index) => args[index],
                Instruction::LoadConst(value) => value,
                Instruction::Add(lhs, rhs) => registers[lhs] + registers[rhs],
                Instruction::Sub(lhs, rhs) => registers[lhs] - registers[rhs],
                Instruction::Mul(lhs, rhs) => registers[lhs] * registers[rhs],
                Instruction::Div(lhs, rhs) => registers[lhs] / registers[rhs],
                Instruction::Pow(lhs, rhs) => registers[lhs].powf(registers[rhs]),
                Instruction::Exp(value) => registers[value].exp(),
                Instruction::Ln(value) => registers[value].ln(),
                Instruction::Sin(value) => registers[value].sin(),
                Instruction::Cos(value) => registers[value].cos(),
                Instruction::Tg(value) => registers[value].tan(),
                Instruction::Ctg(value) => 1.0 / registers[value].tan(),
                Instruction::ArcSin(value) => registers[value].asin(),
                Instruction::ArcCos(value) => registers[value].acos(),
                Instruction::ArcTg(value) => registers[value].atan(),
                Instruction::ArcCtg(value) => (PI / 2.0) - registers[value].atan(),
            };
        }
    }
}

/// Compile several expressions into one instruction tape.
pub(crate) fn compile_many(exprs: &[Expr], vars: &[&str]) -> CompiledBatch {
    let mut instructions = Vec::new();
    let mut output_registers = Vec::with_capacity(exprs.len());

    for expr in exprs {
        let output_register = compile_expr(expr, vars, &mut instructions);
        output_registers.push(output_register);
    }

    CompiledBatch {
        variable_names: vars.iter().map(|name| (*name).to_string()).collect(),
        instructions,
        output_registers,
    }
}

fn compile_expr(expr: &Expr, vars: &[&str], instructions: &mut Vec<Instruction>) -> usize {
    let instruction = match expr {
        Expr::Var(name) => {
            let index = vars
                .iter()
                .position(|&var_name| var_name == name)
                .unwrap_or_else(|| panic!("variable `{}` was not provided to IR compiler", name));
            Instruction::LoadVar(index)
        }
        Expr::Const(value) => Instruction::LoadConst(*value),
        Expr::Add(lhs, rhs) => {
            let lhs_register = compile_expr(lhs, vars, instructions);
            let rhs_register = compile_expr(rhs, vars, instructions);
            Instruction::Add(lhs_register, rhs_register)
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_register = compile_expr(lhs, vars, instructions);
            let rhs_register = compile_expr(rhs, vars, instructions);
            Instruction::Sub(lhs_register, rhs_register)
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_register = compile_expr(lhs, vars, instructions);
            let rhs_register = compile_expr(rhs, vars, instructions);
            Instruction::Mul(lhs_register, rhs_register)
        }
        Expr::Div(lhs, rhs) => {
            let lhs_register = compile_expr(lhs, vars, instructions);
            let rhs_register = compile_expr(rhs, vars, instructions);
            Instruction::Div(lhs_register, rhs_register)
        }
        Expr::Pow(base, exponent) => {
            let base_register = compile_expr(base, vars, instructions);
            let exponent_register = compile_expr(exponent, vars, instructions);
            Instruction::Pow(base_register, exponent_register)
        }
        Expr::Exp(value) => Instruction::Exp(compile_expr(value, vars, instructions)),
        Expr::Ln(value) => Instruction::Ln(compile_expr(value, vars, instructions)),
        Expr::sin(value) => Instruction::Sin(compile_expr(value, vars, instructions)),
        Expr::cos(value) => Instruction::Cos(compile_expr(value, vars, instructions)),
        Expr::tg(value) => Instruction::Tg(compile_expr(value, vars, instructions)),
        Expr::ctg(value) => Instruction::Ctg(compile_expr(value, vars, instructions)),
        Expr::arcsin(value) => Instruction::ArcSin(compile_expr(value, vars, instructions)),
        Expr::arccos(value) => Instruction::ArcCos(compile_expr(value, vars, instructions)),
        Expr::arctg(value) => Instruction::ArcTg(compile_expr(value, vars, instructions)),
        Expr::arcctg(value) => Instruction::ArcCtg(compile_expr(value, vars, instructions)),
    };

    let register = instructions.len();
    instructions.push(instruction);
    register
}

#[cfg(test)]
mod tests {
    use super::compile_many;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn test_compile_many_evaluates_multiple_outputs() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());

        let exprs = vec![
            x.clone() + y.clone(),
            x.clone() * y.clone(),
            (x.clone() + Expr::Const(1.0)).pow(Expr::Const(2.0)),
        ];

        let compiled = compile_many(&exprs, &["x", "y"]);
        let values = compiled.eval(&[2.0, 3.0]);

        assert_eq!(values.len(), 3);
        assert!((values[0] - 5.0).abs() < 1e-12);
        assert!((values[1] - 6.0).abs() < 1e-12);
        assert!((values[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_compile_many_tracks_variable_order() {
        let x = Expr::Var("x".to_string());
        let z = Expr::Var("z".to_string());
        let compiled = compile_many(&[x + z], &["z", "x"]);

        assert_eq!(
            compiled.variable_names,
            vec!["z".to_string(), "x".to_string()]
        );
        assert!((compiled.eval_single(&[2.0, 5.0]) - 7.0).abs() < 1e-12);
    }
}
