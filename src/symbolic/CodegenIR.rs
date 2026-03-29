//! Code generation infrastructure for large symbolic systems.
//!
//! This module is an experimental compile-time oriented backend for the
//! symbolic engine. It is intentionally separate from the runtime
//! `lambdify` path: the goal here is to lower symbolic expressions into a
//! simple linear IR, inspect or transform that IR, and then emit ordinary
//! Rust source code for large residual and Jacobian blocks.
//!
//! Main data structures:
//! - `Instr`: one straight-line IR instruction.
//! - `Temp`: index of a temporary register used by the IR.
//! - `LinearExpr`: lowered IR for one output expression.
//! - `LinearBlock`: lowered IR for many outputs sharing one instruction stream.
//! - `TempPlanEntry`: use/liveness metadata for one temporary.
//! - `LinearBlockPlan`: shared-temporary planning data derived from `LinearBlock`.
//! - `Lowerer`: converts `Expr` trees into `LinearExpr` or `LinearBlock`.
//! - `RustEmitter`: turns lowered IR into Rust source code.
//! - `GeneratedFunction`: named single-output codegen unit.
//! - `GeneratedBlock`: named multi-output codegen unit.
//! - `CodegenModule`: assembles many generated units into one Rust module.
//!
//! Current scope:
//! - shared input temporaries,
//! - shared constant temporaries,
//! - block-level structural CSE during lowering,
//! - lightweight peephole simplification on linear IR,
//! - single-output and multi-output emission,
//! - verification interpreters for tests.
//!
//! High-level mathematical task descriptions such as residual, Jacobian,
//! IVP, and sparse Jacobian scenarios live in
//! [`crate::symbolic::codegen_tasks`]. This module stays focused on the lower
//! IR/code-emission pipeline.
//!
//! Future intended scope:
//! - shared temporary planning across many outputs,
//! - CSE and other IR passes,
//! - residual/Jacobian-specific batch code generation.

use crate::symbolic::codegen_tasks::{CodegenOutputLayout, CodegenTaskPlan};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_metadata::TraversalCache;

use std::collections::HashMap;
use std::f64::consts::PI;

/// Temporary register identifier used by the linear IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Temp(pub usize);

/// A single straight-line IR instruction.
#[derive(Debug, Clone)]
pub enum Instr {
    Const { dst: Temp, value: f64 },
    Input { dst: Temp, index: usize },

    Add { dst: Temp, a: Temp, b: Temp },
    Sub { dst: Temp, a: Temp, b: Temp },
    Mul { dst: Temp, a: Temp, b: Temp },
    Div { dst: Temp, a: Temp, b: Temp },
    Pow { dst: Temp, base: Temp, exp: Temp },

    Exp { dst: Temp, x: Temp },
    Ln { dst: Temp, x: Temp },
    Sin { dst: Temp, x: Temp },
    Cos { dst: Temp, x: Temp },
    Tg { dst: Temp, x: Temp },
    Ctg { dst: Temp, x: Temp },
    ArcSin { dst: Temp, x: Temp },
    ArcCos { dst: Temp, x: Temp },
    ArcTg { dst: Temp, x: Temp },
    ArcCtg { dst: Temp, x: Temp },
}

/// Lowered single-output expression.
#[derive(Debug, Clone)]
pub struct LinearExpr {
    /// Straight-line instruction stream that computes the output temporary.
    pub instructions: Vec<Instr>,
    /// Temporary holding the final scalar result of this lowered expression.
    pub output: Temp,
    /// Number of temporaries required to execute `instructions`.
    pub num_temps: usize,
}

/// Lowered block of multiple output expressions.
///
/// This stage is intentionally separate from symbolic `Expr` and from final
/// source assembly so we can later add IR passes such as CSE, dead-temp
/// cleanup, and shared temporary planning for residual and Jacobian blocks.
#[derive(Debug, Clone)]
pub struct LinearBlock {
    /// Shared straight-line instruction stream for all block outputs.
    pub instructions: Vec<Instr>,
    /// Temporaries that should be written into the final output buffer.
    pub outputs: Vec<Temp>,
    /// Number of temporaries required to execute `instructions`.
    pub num_temps: usize,
}

/// Liveness and usage metadata for a single temporary register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TempPlanEntry {
    /// Temporary described by this planning entry.
    pub temp: Temp,
    /// Instruction index where the temporary is first defined.
    pub defined_at: usize,
    /// Number of later instructions that read this temporary.
    pub use_count: usize,
    /// Last instruction index that reads this temporary.
    pub last_use_at: usize,
    /// Whether this temporary is part of the externally visible outputs.
    pub is_output: bool,
}

/// Planning information derived from a `LinearBlock`.
///
/// This is the first step toward shared temporary reuse. It does not rewrite
/// the IR yet. Instead, it records how each temporary is used and after which
/// instruction a temporary becomes dead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearBlockPlan {
    /// One liveness/usage record per temporary in the block.
    pub entries: Vec<TempPlanEntry>,
    /// Temporaries that become dead immediately after each instruction index.
    pub released_after_instruction: Vec<Vec<Temp>>,
}

impl LinearExpr {
    /// Evaluate the lowered expression for verification tests only.
    ///
    /// This interpreter is not intended to be the final fast backend.
    pub fn eval(&self, args: &[f64]) -> f64 {
        LinearBlock::from(self.clone()).eval_single(args)
    }

    /// Apply a lightweight local simplification pass to the lowered IR.
    pub fn peephole_optimize(&self) -> LinearExpr {
        let optimized = LinearBlock::from(self.clone()).peephole_optimize();
        debug_assert!(
            optimized.outputs.len() == 1,
            "single-output peephole pass must preserve single output"
        );
        LinearExpr {
            instructions: optimized.instructions,
            output: optimized.outputs[0],
            num_temps: optimized.num_temps,
        }
    }
}

impl LinearBlock {
    /// Evaluate the lowered block for verification tests only.
    ///
    /// This interpreter is not intended to be the final fast backend.
    pub fn eval(&self, args: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.outputs.len()];
        self.eval_into(args, &mut out);
        out
    }

    /// Evaluate a single-output block for verification tests only.
    pub fn eval_single(&self, args: &[f64]) -> f64 {
        debug_assert!(
            self.outputs.len() == 1,
            "eval_single() expects exactly one output"
        );
        self.eval(args)[0]
    }

    /// Evaluate into a caller-provided output slice for verification tests only.
    pub fn eval_into(&self, args: &[f64], out: &mut [f64]) {
        assert!(
            out.len() >= self.outputs.len(),
            "output buffer is too small: expected at least {}, got {}",
            self.outputs.len(),
            out.len()
        );

        let mut temps = vec![0.0_f64; self.num_temps];

        for instr in &self.instructions {
            match *instr {
                Instr::Const { dst, value } => {
                    temps[dst.0] = value;
                }
                Instr::Input { dst, index } => {
                    assert!(
                        index < args.len(),
                        "input index {} is out of bounds for {} arguments",
                        index,
                        args.len()
                    );
                    temps[dst.0] = args[index];
                }

                Instr::Add { dst, a, b } => {
                    temps[dst.0] = temps[a.0] + temps[b.0];
                }
                Instr::Sub { dst, a, b } => {
                    temps[dst.0] = temps[a.0] - temps[b.0];
                }
                Instr::Mul { dst, a, b } => {
                    temps[dst.0] = temps[a.0] * temps[b.0];
                }
                Instr::Div { dst, a, b } => {
                    temps[dst.0] = temps[a.0] / temps[b.0];
                }
                Instr::Pow { dst, base, exp } => {
                    temps[dst.0] = temps[base.0].powf(temps[exp.0]);
                }

                Instr::Exp { dst, x } => {
                    temps[dst.0] = temps[x.0].exp();
                }
                Instr::Ln { dst, x } => {
                    temps[dst.0] = temps[x.0].ln();
                }
                Instr::Sin { dst, x } => {
                    temps[dst.0] = temps[x.0].sin();
                }
                Instr::Cos { dst, x } => {
                    temps[dst.0] = temps[x.0].cos();
                }
                Instr::Tg { dst, x } => {
                    temps[dst.0] = temps[x.0].tan();
                }
                Instr::Ctg { dst, x } => {
                    temps[dst.0] = 1.0_f64 / temps[x.0].tan();
                }
                Instr::ArcSin { dst, x } => {
                    temps[dst.0] = temps[x.0].asin();
                }
                Instr::ArcCos { dst, x } => {
                    temps[dst.0] = temps[x.0].acos();
                }
                Instr::ArcTg { dst, x } => {
                    temps[dst.0] = temps[x.0].atan();
                }
                Instr::ArcCtg { dst, x } => {
                    temps[dst.0] = (PI / 2.0_f64) - temps[x.0].atan();
                }
            }
        }

        for (dst, output_temp) in out.iter_mut().zip(&self.outputs) {
            *dst = temps[output_temp.0];
        }
    }

    /// Build a temporary-usage plan for later IR passes.
    pub fn temp_plan(&self) -> LinearBlockPlan {
        let mut defined_at = vec![None; self.num_temps];
        let mut use_count = vec![0_usize; self.num_temps];
        let mut last_use_at = vec![0_usize; self.num_temps];
        let mut is_output = vec![false; self.num_temps];

        for (instr_index, instr) in self.instructions.iter().enumerate() {
            let dst = instr.dst();
            defined_at[dst.0] = Some(instr_index);

            for operand in instr.operands() {
                use_count[operand.0] += 1;
                last_use_at[operand.0] = instr_index;
            }
        }

        for &output in &self.outputs {
            is_output[output.0] = true;
            let last_use = self.instructions.len();
            if last_use > last_use_at[output.0] {
                last_use_at[output.0] = last_use;
            }
        }

        let mut entries = Vec::with_capacity(self.num_temps);
        let mut released_after_instruction = vec![Vec::new(); self.instructions.len()];

        for temp_index in 0..self.num_temps {
            let defined_at = defined_at[temp_index]
                .unwrap_or_else(|| panic!("temporary t{} was never defined", temp_index));
            let entry = TempPlanEntry {
                temp: Temp(temp_index),
                defined_at,
                use_count: use_count[temp_index],
                last_use_at: last_use_at[temp_index],
                is_output: is_output[temp_index],
            };

            if entry.use_count > 0
                && !entry.is_output
                && entry.last_use_at < released_after_instruction.len()
            {
                released_after_instruction[entry.last_use_at].push(entry.temp);
            }

            entries.push(entry);
        }

        LinearBlockPlan {
            entries,
            released_after_instruction,
        }
    }

    /// Apply a lightweight local simplification pass to the lowered IR.
    ///
    /// This pass is intentionally small and predictable. It only performs
    /// constant folding and cheap algebraic identities such as `x + 0`, `x * 1`,
    /// `x * 0`, `x / 1`, and `x^1`.
    pub fn peephole_optimize(&self) -> LinearBlock {
        let mut optimized_instructions = Vec::with_capacity(self.instructions.len());
        let mut aliases: Vec<Temp> = (0..self.num_temps).map(Temp).collect();
        let mut known_consts = vec![None; self.num_temps];

        for instr in &self.instructions {
            let remap = |temp: Temp, aliases: &[Temp]| resolve_alias(temp, aliases);
            let const_of = |temp: Temp, aliases: &[Temp], known_consts: &[Option<f64>]| {
                known_consts[remap(temp, aliases).0]
            };

            match *instr {
                Instr::Const { dst, value } => {
                    aliases[dst.0] = dst;
                    known_consts[dst.0] = Some(value);
                    optimized_instructions.push(Instr::Const { dst, value });
                }
                Instr::Input { dst, index } => {
                    aliases[dst.0] = dst;
                    known_consts[dst.0] = None;
                    optimized_instructions.push(Instr::Input { dst, index });
                }
                Instr::Add { dst, a, b } => {
                    let a = remap(a, &aliases);
                    let b = remap(b, &aliases);
                    match (
                        const_of(a, &aliases, &known_consts),
                        const_of(b, &aliases, &known_consts),
                    ) {
                        (Some(lhs), Some(rhs)) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(lhs + rhs);
                            optimized_instructions.push(Instr::Const {
                                dst,
                                value: lhs + rhs,
                            });
                        }
                        (Some(0.0), _) => {
                            aliases[dst.0] = b;
                            known_consts[dst.0] = known_consts[b.0];
                        }
                        (_, Some(0.0)) => {
                            aliases[dst.0] = a;
                            known_consts[dst.0] = known_consts[a.0];
                        }
                        _ => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = None;
                            optimized_instructions.push(Instr::Add { dst, a, b });
                        }
                    }
                }
                Instr::Sub { dst, a, b } => {
                    let a = remap(a, &aliases);
                    let b = remap(b, &aliases);
                    match (
                        const_of(a, &aliases, &known_consts),
                        const_of(b, &aliases, &known_consts),
                    ) {
                        (Some(lhs), Some(rhs)) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(lhs - rhs);
                            optimized_instructions.push(Instr::Const {
                                dst,
                                value: lhs - rhs,
                            });
                        }
                        (_, Some(0.0)) => {
                            aliases[dst.0] = a;
                            known_consts[dst.0] = known_consts[a.0];
                        }
                        _ if a == b => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(0.0);
                            optimized_instructions.push(Instr::Const { dst, value: 0.0 });
                        }
                        _ => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = None;
                            optimized_instructions.push(Instr::Sub { dst, a, b });
                        }
                    }
                }
                Instr::Mul { dst, a, b } => {
                    let a = remap(a, &aliases);
                    let b = remap(b, &aliases);
                    match (
                        const_of(a, &aliases, &known_consts),
                        const_of(b, &aliases, &known_consts),
                    ) {
                        (Some(lhs), Some(rhs)) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(lhs * rhs);
                            optimized_instructions.push(Instr::Const {
                                dst,
                                value: lhs * rhs,
                            });
                        }
                        (Some(0.0), _) | (_, Some(0.0)) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(0.0);
                            optimized_instructions.push(Instr::Const { dst, value: 0.0 });
                        }
                        (Some(1.0), _) => {
                            aliases[dst.0] = b;
                            known_consts[dst.0] = known_consts[b.0];
                        }
                        (_, Some(1.0)) => {
                            aliases[dst.0] = a;
                            known_consts[dst.0] = known_consts[a.0];
                        }
                        _ => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = None;
                            optimized_instructions.push(Instr::Mul { dst, a, b });
                        }
                    }
                }
                Instr::Div { dst, a, b } => {
                    let a = remap(a, &aliases);
                    let b = remap(b, &aliases);
                    match (
                        const_of(a, &aliases, &known_consts),
                        const_of(b, &aliases, &known_consts),
                    ) {
                        (Some(lhs), Some(rhs)) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(lhs / rhs);
                            optimized_instructions.push(Instr::Const {
                                dst,
                                value: lhs / rhs,
                            });
                        }
                        (Some(0.0), _) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(0.0);
                            optimized_instructions.push(Instr::Const { dst, value: 0.0 });
                        }
                        (_, Some(1.0)) => {
                            aliases[dst.0] = a;
                            known_consts[dst.0] = known_consts[a.0];
                        }
                        _ => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = None;
                            optimized_instructions.push(Instr::Div { dst, a, b });
                        }
                    }
                }
                Instr::Pow { dst, base, exp } => {
                    let base = remap(base, &aliases);
                    let exp = remap(exp, &aliases);
                    match (
                        const_of(base, &aliases, &known_consts),
                        const_of(exp, &aliases, &known_consts),
                    ) {
                        (Some(lhs), Some(rhs)) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(lhs.powf(rhs));
                            optimized_instructions.push(Instr::Const {
                                dst,
                                value: lhs.powf(rhs),
                            });
                        }
                        (_, Some(0.0)) => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = Some(1.0);
                            optimized_instructions.push(Instr::Const { dst, value: 1.0 });
                        }
                        (_, Some(1.0)) => {
                            aliases[dst.0] = base;
                            known_consts[dst.0] = known_consts[base.0];
                        }
                        _ => {
                            aliases[dst.0] = dst;
                            known_consts[dst.0] = None;
                            optimized_instructions.push(Instr::Pow { dst, base, exp });
                        }
                    }
                }
                Instr::Exp { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.exp(),
                        |dst, x| Instr::Exp { dst, x },
                    );
                }
                Instr::Ln { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.ln(),
                        |dst, x| Instr::Ln { dst, x },
                    );
                }
                Instr::Sin { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.sin(),
                        |dst, x| Instr::Sin { dst, x },
                    );
                }
                Instr::Cos { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.cos(),
                        |dst, x| Instr::Cos { dst, x },
                    );
                }
                Instr::Tg { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.tan(),
                        |dst, x| Instr::Tg { dst, x },
                    );
                }
                Instr::Ctg { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| 1.0_f64 / x.tan(),
                        |dst, x| Instr::Ctg { dst, x },
                    );
                }
                Instr::ArcSin { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.asin(),
                        |dst, x| Instr::ArcSin { dst, x },
                    );
                }
                Instr::ArcCos { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.acos(),
                        |dst, x| Instr::ArcCos { dst, x },
                    );
                }
                Instr::ArcTg { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| x.atan(),
                        |dst, x| Instr::ArcTg { dst, x },
                    );
                }
                Instr::ArcCtg { dst, x } => {
                    optimize_unary_instr(
                        dst,
                        remap(x, &aliases),
                        &mut aliases,
                        &mut known_consts,
                        &mut optimized_instructions,
                        |x| (PI / 2.0_f64) - x.atan(),
                        |dst, x| Instr::ArcCtg { dst, x },
                    );
                }
            }
        }

        let outputs = self
            .outputs
            .iter()
            .map(|&output| resolve_alias(output, &aliases))
            .collect();

        LinearBlock {
            instructions: optimized_instructions,
            outputs,
            num_temps: self.num_temps,
        }
    }

    /// Reassign temporary registers using liveness information from
    /// `temp_plan()`.
    ///
    /// This is a first temp-reuse pass only. It does not attempt CSE or
    /// algebraic rewrites. It simply reuses registers whose last use has
    /// already happened.
    pub fn reuse_temps(&self) -> LinearBlock {
        let plan = self.temp_plan();
        let mut remapped_instructions = Vec::with_capacity(self.instructions.len());
        let mut active_map: Vec<Option<Temp>> = vec![None; self.num_temps];
        let mut free_temps = Vec::<Temp>::new();
        let mut next_temp = 0_usize;

        for (instr_index, instr) in self.instructions.iter().enumerate() {
            let remap_operand = |temp: Temp, active_map: &[Option<Temp>]| {
                active_map[temp.0].unwrap_or_else(|| panic!("temporary t{} is not active", temp.0))
            };

            let new_instr = match *instr {
                Instr::Const { dst, value } => {
                    let new_dst = free_temps.pop().unwrap_or_else(|| {
                        let temp = Temp(next_temp);
                        next_temp += 1;
                        temp
                    });
                    active_map[dst.0] = Some(new_dst);
                    Instr::Const {
                        dst: new_dst,
                        value,
                    }
                }
                Instr::Input { dst, index } => {
                    let new_dst = free_temps.pop().unwrap_or_else(|| {
                        let temp = Temp(next_temp);
                        next_temp += 1;
                        temp
                    });
                    active_map[dst.0] = Some(new_dst);
                    Instr::Input {
                        dst: new_dst,
                        index,
                    }
                }
                Instr::Add { dst, a, b } => {
                    let new_instr = Instr::Add {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        a: remap_operand(a, &active_map),
                        b: remap_operand(b, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Sub { dst, a, b } => {
                    let new_instr = Instr::Sub {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        a: remap_operand(a, &active_map),
                        b: remap_operand(b, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Mul { dst, a, b } => {
                    let new_instr = Instr::Mul {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        a: remap_operand(a, &active_map),
                        b: remap_operand(b, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Div { dst, a, b } => {
                    let new_instr = Instr::Div {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        a: remap_operand(a, &active_map),
                        b: remap_operand(b, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Pow { dst, base, exp } => {
                    let new_instr = Instr::Pow {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        base: remap_operand(base, &active_map),
                        exp: remap_operand(exp, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Exp { dst, x } => {
                    let new_instr = Instr::Exp {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Ln { dst, x } => {
                    let new_instr = Instr::Ln {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Sin { dst, x } => {
                    let new_instr = Instr::Sin {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Cos { dst, x } => {
                    let new_instr = Instr::Cos {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Tg { dst, x } => {
                    let new_instr = Instr::Tg {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::Ctg { dst, x } => {
                    let new_instr = Instr::Ctg {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::ArcSin { dst, x } => {
                    let new_instr = Instr::ArcSin {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::ArcCos { dst, x } => {
                    let new_instr = Instr::ArcCos {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::ArcTg { dst, x } => {
                    let new_instr = Instr::ArcTg {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
                Instr::ArcCtg { dst, x } => {
                    let new_instr = Instr::ArcCtg {
                        dst: free_temps.pop().unwrap_or_else(|| {
                            let temp = Temp(next_temp);
                            next_temp += 1;
                            temp
                        }),
                        x: remap_operand(x, &active_map),
                    };
                    active_map[dst.0] = Some(new_instr.dst());
                    new_instr
                }
            };

            remapped_instructions.push(new_instr);

            for &released_original in &plan.released_after_instruction[instr_index] {
                if let Some(physical) = active_map[released_original.0].take() {
                    free_temps.push(physical);
                }
            }
        }

        let outputs = self
            .outputs
            .iter()
            .map(|output| {
                active_map[output.0]
                    .unwrap_or_else(|| panic!("output temporary t{} is not active", output.0))
            })
            .collect();

        LinearBlock {
            instructions: remapped_instructions,
            outputs,
            num_temps: next_temp,
        }
    }
}

impl From<LinearExpr> for LinearBlock {
    fn from(expr: LinearExpr) -> Self {
        Self {
            instructions: expr.instructions,
            outputs: vec![expr.output],
            num_temps: expr.num_temps,
        }
    }
}

impl Instr {
    fn dst(&self) -> Temp {
        match *self {
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
            | Instr::ArcCtg { dst, .. } => dst,
        }
    }

    fn operands(&self) -> Vec<Temp> {
        match *self {
            Instr::Const { .. } | Instr::Input { .. } => Vec::new(),
            Instr::Add { a, b, .. }
            | Instr::Sub { a, b, .. }
            | Instr::Mul { a, b, .. }
            | Instr::Div { a, b, .. } => vec![a, b],
            Instr::Pow { base, exp, .. } => vec![base, exp],
            Instr::Exp { x, .. }
            | Instr::Ln { x, .. }
            | Instr::Sin { x, .. }
            | Instr::Cos { x, .. }
            | Instr::Tg { x, .. }
            | Instr::Ctg { x, .. }
            | Instr::ArcSin { x, .. }
            | Instr::ArcCos { x, .. }
            | Instr::ArcTg { x, .. }
            | Instr::ArcCtg { x, .. } => vec![x],
        }
    }
}

fn resolve_alias(temp: Temp, aliases: &[Temp]) -> Temp {
    let mut current = temp;
    while aliases[current.0] != current {
        current = aliases[current.0];
    }
    current
}

fn optimize_unary_instr<F, G>(
    dst: Temp,
    x: Temp,
    aliases: &mut [Temp],
    known_consts: &mut [Option<f64>],
    optimized_instructions: &mut Vec<Instr>,
    eval_const: F,
    build_instr: G,
) where
    F: FnOnce(f64) -> f64,
    G: FnOnce(Temp, Temp) -> Instr,
{
    let x = resolve_alias(x, aliases);
    if let Some(value) = known_consts[x.0] {
        aliases[dst.0] = dst;
        known_consts[dst.0] = Some(eval_const(value));
        optimized_instructions.push(Instr::Const {
            dst,
            value: known_consts[dst.0].unwrap(),
        });
    } else {
        aliases[dst.0] = dst;
        known_consts[dst.0] = None;
        optimized_instructions.push(build_instr(dst, x));
    }
}

/// Lowers symbolic expressions into linear IR.
pub struct Lowerer<'a> {
    instructions: Vec<Instr>,
    next_temp: usize,
    var_index_map: HashMap<&'a str, usize>,
    input_cache: HashMap<usize, Temp>,
    const_cache: HashMap<u64, Temp>,
    expr_cache: HashMap<u64, Vec<(Expr, Temp)>>,
    traversal_cache: TraversalCache,
}

impl<'a> Lowerer<'a> {
    pub fn new(vars: &'a [&'a str]) -> Self {
        let var_index_map = vars
            .iter()
            .enumerate()
            .map(|(index, &name)| (name, index))
            .collect();

        Self {
            instructions: Vec::new(),
            next_temp: 0,
            var_index_map,
            input_cache: HashMap::new(),
            const_cache: HashMap::new(),
            expr_cache: HashMap::new(),
            traversal_cache: TraversalCache::new(),
        }
    }

    fn fresh_temp(&mut self) -> Temp {
        let t = Temp(self.next_temp);
        self.next_temp += 1;
        t
    }

    fn find_var_index(&self, name: &str) -> usize {
        self.var_index_map
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("Variable '{name}' not found in vars list"))
    }

    fn lower_var(&mut self, name: &str) -> Temp {
        let index = self.find_var_index(name);

        if let Some(&t) = self.input_cache.get(&index) {
            return t;
        }

        let dst = self.fresh_temp();
        self.instructions.push(Instr::Input { dst, index });
        self.input_cache.insert(index, dst);
        dst
    }

    fn lower_const(&mut self, value: f64) -> Temp {
        let key = value.to_bits();

        if let Some(&t) = self.const_cache.get(&key) {
            return t;
        }

        let dst = self.fresh_temp();
        self.instructions.push(Instr::Const { dst, value });
        self.const_cache.insert(key, dst);
        dst
    }

    fn lookup_expr_cache(&mut self, expr: &Expr) -> Option<Temp> {
        let signature = self.traversal_cache.signature(expr);
        self.expr_cache.get(&signature).and_then(|bucket| {
            bucket
                .iter()
                .find(|(candidate, _)| candidate == expr)
                .map(|(_, temp)| *temp)
        })
    }

    fn cache_expr(&mut self, expr: &Expr, temp: Temp) {
        let signature = self.traversal_cache.signature(expr);
        self.expr_cache
            .entry(signature)
            .or_default()
            .push((expr.clone(), temp));
    }

    fn lower_expr(&mut self, expr: &Expr) -> Temp {
        if !matches!(expr, Expr::Var(_) | Expr::Const(_)) {
            if let Some(existing) = self.lookup_expr_cache(expr) {
                return existing;
            }
        }

        let lowered = match expr {
            Expr::Var(name) => self.lower_var(name),
            Expr::Const(value) => self.lower_const(*value),

            Expr::Add(a, b) => {
                let ta = self.lower_expr(a);
                let tb = self.lower_expr(b);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Add { dst, a: ta, b: tb });
                dst
            }
            Expr::Sub(a, b) => {
                let ta = self.lower_expr(a);
                let tb = self.lower_expr(b);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Sub { dst, a: ta, b: tb });
                dst
            }
            Expr::Mul(a, b) => {
                let ta = self.lower_expr(a);
                let tb = self.lower_expr(b);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Mul { dst, a: ta, b: tb });
                dst
            }
            Expr::Div(a, b) => {
                let ta = self.lower_expr(a);
                let tb = self.lower_expr(b);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Div { dst, a: ta, b: tb });
                dst
            }
            Expr::Pow(base, exp) => {
                let tbase = self.lower_expr(base);
                let texp = self.lower_expr(exp);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Pow {
                    dst,
                    base: tbase,
                    exp: texp,
                });
                dst
            }

            Expr::Exp(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Exp { dst, x: tx });
                dst
            }
            Expr::Ln(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Ln { dst, x: tx });
                dst
            }
            Expr::sin(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Sin { dst, x: tx });
                dst
            }
            Expr::cos(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Cos { dst, x: tx });
                dst
            }
            Expr::tg(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Tg { dst, x: tx });
                dst
            }
            Expr::ctg(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::Ctg { dst, x: tx });
                dst
            }
            Expr::arcsin(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::ArcSin { dst, x: tx });
                dst
            }
            Expr::arccos(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::ArcCos { dst, x: tx });
                dst
            }
            Expr::arctg(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::ArcTg { dst, x: tx });
                dst
            }
            Expr::arcctg(x) => {
                let tx = self.lower_expr(x);
                let dst = self.fresh_temp();
                self.instructions.push(Instr::ArcCtg { dst, x: tx });
                dst
            }
        };

        if !matches!(expr, Expr::Var(_) | Expr::Const(_)) {
            self.cache_expr(expr, lowered);
        }

        lowered
    }

    pub fn lower(mut self, expr: &Expr) -> LinearExpr {
        let output = self.lower_expr(expr);

        LinearExpr {
            instructions: self.instructions,
            output,
            num_temps: self.next_temp,
        }
    }

    pub fn lower_many(mut self, exprs: &[Expr]) -> LinearBlock {
        let outputs = exprs.iter().map(|expr| self.lower_expr(expr)).collect();

        LinearBlock {
            instructions: self.instructions,
            outputs,
            num_temps: self.next_temp,
        }
    }
}

impl Expr {
    /// Lower a single symbolic expression into linear verification IR.
    pub fn lower_to_linear(&self, vars: &[&str]) -> LinearExpr {
        Lowerer::new(vars).lower(self)
    }

    /// Lower many symbolic expressions together so they can share input and
    /// constant temporaries before heavier IR passes are introduced.
    pub fn lower_many_to_linear(exprs: &[Expr], vars: &[&str]) -> LinearBlock {
        Lowerer::new(vars).lower_many(exprs)
    }
}

/// Emits Rust source code from lowered IR.
pub struct RustEmitter;

impl RustEmitter {
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
            "f64::NAN".to_string()
        } else if x == f64::INFINITY {
            "f64::INFINITY".to_string()
        } else if x == f64::NEG_INFINITY {
            "f64::NEG_INFINITY".to_string()
        } else {
            format!("{:.17e}_f64", x)
        }
    }

    pub fn emit_function(ir: &LinearExpr, fn_name: &str, arity: usize) -> String {
        Self::validate_identifier(fn_name, "function");

        let mut out = String::new();
        out.push_str(&format!("pub fn {}(args: &[f64]) -> f64 {{\n", fn_name));
        out.push_str(&format!(
            "    debug_assert!(args.len() >= {}, \"expected at least {} arguments\");\n",
            arity, arity
        ));

        for instr in &ir.instructions {
            Self::emit_instruction(instr, &mut out);
        }

        out.push_str(&format!("    {}\n", Self::temp_name(ir.output)));
        out.push_str("}\n");
        out
    }

    pub fn emit_block_function(ir: &LinearBlock, fn_name: &str, arity: usize) -> String {
        Self::validate_identifier(fn_name, "function");

        let mut out = String::new();
        out.push_str(&format!(
            "pub fn {}(args: &[f64], out: &mut [f64]) {{\n",
            fn_name
        ));
        out.push_str(&format!(
            "    debug_assert!(args.len() >= {}, \"expected at least {} arguments\");\n",
            arity, arity
        ));
        out.push_str(&format!(
            "    debug_assert!(out.len() >= {}, \"expected at least {} output slots\");\n",
            ir.outputs.len(),
            ir.outputs.len()
        ));

        for instr in &ir.instructions {
            Self::emit_instruction(instr, &mut out);
        }

        for (index, output) in ir.outputs.iter().enumerate() {
            out.push_str(&format!(
                "    out[{}] = {};\n",
                index,
                Self::temp_name(*output)
            ));
        }

        out.push_str("}\n");
        out
    }

    /// Emit a residual block function with explicit vector-length metadata.
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

    /// Emit a dense Jacobian block function with explicit matrix-shape metadata.
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

    /// Emit a sparse-Jacobian-values block with explicit sparsity metadata.
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
                    "    let {} = {};\n",
                    Self::temp_name(*dst),
                    Self::fmt_f64(*value)
                ));
            }
            Instr::Input { dst, index } => {
                out.push_str(&format!(
                    "    let {} = args[{}];\n",
                    Self::temp_name(*dst),
                    index
                ));
            }

            Instr::Add { dst, a, b } => {
                out.push_str(&format!(
                    "    let {} = {} + {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Sub { dst, a, b } => {
                out.push_str(&format!(
                    "    let {} = {} - {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Mul { dst, a, b } => {
                out.push_str(&format!(
                    "    let {} = {} * {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Div { dst, a, b } => {
                out.push_str(&format!(
                    "    let {} = {} / {};\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*a),
                    Self::temp_name(*b),
                ));
            }
            Instr::Pow { dst, base, exp } => {
                out.push_str(&format!(
                    "    let {} = {}.powf({});\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*base),
                    Self::temp_name(*exp),
                ));
            }

            Instr::Exp { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.exp();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Ln { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.ln();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Sin { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.sin();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Cos { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.cos();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Tg { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.tan();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::Ctg { dst, x } => {
                out.push_str(&format!(
                    "    let {} = 1.0_f64 / {}.tan();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcSin { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.asin();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcCos { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.acos();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcTg { dst, x } => {
                out.push_str(&format!(
                    "    let {} = {}.atan();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
            Instr::ArcCtg { dst, x } => {
                out.push_str(&format!(
                    "    let {} = (std::f64::consts::PI / 2.0_f64) - {}.atan();\n",
                    Self::temp_name(*dst),
                    Self::temp_name(*x)
                ));
            }
        }
    }
}

/// Single generated function with its source-level metadata.
#[derive(Debug, Clone)]
pub struct GeneratedFunction {
    /// Public Rust function name emitted for this generated scalar function.
    pub fn_name: String,
    /// Flattened input names expected by the generated function.
    pub vars: Vec<String>,
    /// Lowered single-output IR backing the generated function.
    pub ir: LinearExpr,
}

impl GeneratedFunction {
    pub fn new(fn_name: impl Into<String>, expr: &Expr, vars: &[&str]) -> Self {
        Self {
            fn_name: fn_name.into(),
            vars: vars.iter().map(|s| s.to_string()).collect(),
            ir: expr.lower_to_linear(vars),
        }
    }

    pub fn emit(&self) -> String {
        RustEmitter::emit_function(&self.ir, &self.fn_name, self.vars.len())
    }
}

/// Named multi-output code generation unit.
#[derive(Debug, Clone)]
pub struct GeneratedBlock {
    /// Public Rust function name emitted for this generated multi-output block.
    pub fn_name: String,
    /// Flattened input names expected by the generated block function.
    pub vars: Vec<String>,
    /// Lowered multi-output IR backing the generated block.
    pub ir: LinearBlock,
    /// Optional typed output-layout metadata used for specialized emitters.
    pub layout: Option<CodegenOutputLayout>,
}

impl GeneratedBlock {
    pub fn new(fn_name: impl Into<String>, exprs: &[Expr], vars: &[&str]) -> Self {
        Self {
            fn_name: fn_name.into(),
            vars: vars.iter().map(|s| s.to_string()).collect(),
            ir: Expr::lower_many_to_linear(exprs, vars),
            layout: None,
        }
    }

    /// Builds a named multi-output code generation unit from a flattened task
    /// plan produced by `codegen_tasks`.
    pub fn from_task_plan(plan: &CodegenTaskPlan<'_>) -> Self {
        let exprs: Vec<Expr> = plan.output_exprs().into_iter().cloned().collect();
        Self {
            fn_name: plan.fn_name.to_string(),
            vars: plan.input_names.iter().map(|s| s.to_string()).collect(),
            ir: Expr::lower_many_to_linear(&exprs, &plan.input_names),
            layout: Some(plan.layout),
        }
    }

    /// Builds a typed residual block from a flattened task plan.
    pub fn from_residual_plan(plan: &CodegenTaskPlan<'_>) -> Self {
        match plan.layout {
            CodegenOutputLayout::Vector { .. } => Self::from_task_plan(plan),
            _ => panic!("residual plan must have vector layout"),
        }
    }

    /// Builds a typed dense Jacobian block from a flattened task plan.
    pub fn from_dense_jacobian_plan(plan: &CodegenTaskPlan<'_>) -> Self {
        match plan.layout {
            CodegenOutputLayout::Matrix { .. } => Self::from_task_plan(plan),
            _ => panic!("dense Jacobian plan must have matrix layout"),
        }
    }

    /// Builds a typed sparse-values block from a flattened task plan.
    pub fn from_sparse_values_plan(plan: &CodegenTaskPlan<'_>) -> Self {
        match plan.layout {
            CodegenOutputLayout::SparseValues { .. } => Self::from_task_plan(plan),
            _ => panic!("sparse-values plan must have sparse-values layout"),
        }
    }

    pub fn emit(&self) -> String {
        match self.layout {
            Some(CodegenOutputLayout::Vector { len }) => RustEmitter::emit_residual_block_function(
                &self.ir,
                &self.fn_name,
                self.vars.len(),
                len,
            ),
            Some(CodegenOutputLayout::Matrix { rows, cols }) => {
                RustEmitter::emit_dense_jacobian_block_function(
                    &self.ir,
                    &self.fn_name,
                    self.vars.len(),
                    rows,
                    cols,
                )
            }
            Some(CodegenOutputLayout::SparseValues { rows, cols, nnz }) => {
                RustEmitter::emit_sparse_values_block_function(
                    &self.ir,
                    &self.fn_name,
                    self.vars.len(),
                    rows,
                    cols,
                    nnz,
                )
            }
            None => RustEmitter::emit_block_function(&self.ir, &self.fn_name, self.vars.len()),
        }
    }
}

/// A source assembly unit for many generated functions.
#[derive(Debug, Default)]
pub struct CodegenModule {
    /// Name of the Rust submodule that will wrap all emitted units.
    module_name: String,
    /// Generated scalar functions collected into this module.
    functions: Vec<GeneratedFunction>,
    /// Generated multi-output blocks collected into this module.
    blocks: Vec<GeneratedBlock>,
}

impl CodegenModule {
    pub fn new(module_name: impl Into<String>) -> Self {
        let module_name = module_name.into();
        RustEmitter::validate_identifier(&module_name, "module");

        Self {
            module_name,
            functions: Vec::new(),
            blocks: Vec::new(),
        }
    }

    pub fn add_function(mut self, fn_name: impl Into<String>, expr: &Expr, vars: &[&str]) -> Self {
        self.functions
            .push(GeneratedFunction::new(fn_name, expr, vars));
        self
    }

    pub fn add_block(mut self, fn_name: impl Into<String>, exprs: &[Expr], vars: &[&str]) -> Self {
        self.blocks.push(GeneratedBlock::new(fn_name, exprs, vars));
        self
    }

    /// Adds a multi-output block from a flattened task plan.
    pub fn add_task_plan(mut self, plan: &CodegenTaskPlan<'_>) -> Self {
        self.blocks.push(GeneratedBlock::from_task_plan(plan));
        self
    }

    /// Adds a typed residual block from a task plan.
    pub fn add_residual_block_plan(mut self, plan: &CodegenTaskPlan<'_>) -> Self {
        self.blocks.push(GeneratedBlock::from_residual_plan(plan));
        self
    }

    /// Adds a typed dense Jacobian block from a task plan.
    pub fn add_dense_jacobian_plan(mut self, plan: &CodegenTaskPlan<'_>) -> Self {
        self.blocks
            .push(GeneratedBlock::from_dense_jacobian_plan(plan));
        self
    }

    /// Adds a typed sparse-values block from a task plan.
    pub fn add_sparse_values_plan(mut self, plan: &CodegenTaskPlan<'_>) -> Self {
        self.blocks
            .push(GeneratedBlock::from_sparse_values_plan(plan));
        self
    }

    pub fn push_function(&mut self, fn_name: impl Into<String>, expr: &Expr, vars: &[&str]) {
        self.functions
            .push(GeneratedFunction::new(fn_name, expr, vars));
    }

    pub fn push_block(&mut self, fn_name: impl Into<String>, exprs: &[Expr], vars: &[&str]) {
        self.blocks.push(GeneratedBlock::new(fn_name, exprs, vars));
    }

    /// Pushes a multi-output block from a flattened task plan.
    pub fn push_task_plan(&mut self, plan: &CodegenTaskPlan<'_>) {
        self.blocks.push(GeneratedBlock::from_task_plan(plan));
    }

    /// Pushes a typed residual block from a task plan.
    pub fn push_residual_block_plan(&mut self, plan: &CodegenTaskPlan<'_>) {
        self.blocks.push(GeneratedBlock::from_residual_plan(plan));
    }

    /// Pushes a typed dense Jacobian block from a task plan.
    pub fn push_dense_jacobian_plan(&mut self, plan: &CodegenTaskPlan<'_>) {
        self.blocks
            .push(GeneratedBlock::from_dense_jacobian_plan(plan));
    }

    /// Pushes a typed sparse-values block from a task plan.
    pub fn push_sparse_values_plan(&mut self, plan: &CodegenTaskPlan<'_>) {
        self.blocks
            .push(GeneratedBlock::from_sparse_values_plan(plan));
    }

    pub fn emit_source(&self) -> String {
        let mut out = String::new();

        out.push_str("// =========================================\n");
        out.push_str("// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.\n");
        out.push_str("// =========================================\n\n");
        out.push_str("#![allow(clippy::all)]\n");
        out.push_str("#![allow(non_snake_case)]\n");
        out.push_str("#![allow(unused_parens)]\n\n");
        out.push_str(&format!("pub mod {} {{\n", self.module_name));

        for func in &self.functions {
            let emitted = func.emit();
            for line in emitted.lines() {
                out.push_str("    ");
                out.push_str(line);
                out.push('\n');
            }
            out.push('\n');
        }

        for block in &self.blocks {
            let emitted = block.emit();
            for line in emitted.lines() {
                out.push_str("    ");
                out.push_str(line);
                out.push('\n');
            }
            out.push('\n');
        }

        out.push_str("}\n");
        out
    }

    pub fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        std::fs::write(path, self.emit_source())
    }

    /// Write the generated module as a checked-in fixture file.
    ///
    /// This is a thin convenience wrapper for the workflow where emitted Rust
    /// source is stored inside the project and then verified by ordinary unit
    /// tests after compilation.
    pub fn write_fixture_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        fixture_note: &str,
    ) -> std::io::Result<()> {
        let mut source = String::new();
        source.push_str("// Fixture file generated by CodegenIR.\n");
        source.push_str("// It is intended to be checked into the repository and\n");
        source.push_str("// validated by wrapper tests after normal Rust compilation.\n");
        if !fixture_note.trim().is_empty() {
            source.push_str("// ");
            source.push_str(fixture_note.trim());
            source.push('\n');
        }
        source.push('\n');
        source.push_str(&self.emit_source());
        std::fs::write(path, source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_generated_fixtures::{fixture_block_eval, fixture_scalar_eval};
    use crate::symbolic::codegen_tasks::{
        CodegenOutputLayout, IvpResidualTask, ResidualTask, SparseExprEntry, SparseJacobianTask,
    };
    use tempfile::tempdir;

    fn count_input_instrs(instrs: &[Instr]) -> usize {
        instrs
            .iter()
            .filter(|instr| matches!(instr, Instr::Input { .. }))
            .count()
    }

    fn count_const_instrs(instrs: &[Instr], value: f64) -> usize {
        instrs
            .iter()
            .filter(|instr| matches!(instr, Instr::Const { value: v, .. } if *v == value))
            .count()
    }

    fn count_add_instrs(instrs: &[Instr]) -> usize {
        instrs
            .iter()
            .filter(|instr| matches!(instr, Instr::Add { .. }))
            .count()
    }

    fn count_mul_instrs(instrs: &[Instr]) -> usize {
        instrs
            .iter()
            .filter(|instr| matches!(instr, Instr::Mul { .. }))
            .count()
    }

    fn build_checked_fixture_module() -> CodegenModule {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let scalar = (x.clone() + y.clone()).pow(Expr::Const(2.0)) + Expr::sin(Box::new(x.clone()))
            - Expr::Const(3.0) * y.clone();
        let block_exprs = vec![
            x.clone() + Expr::Const(1.0),
            x.clone() * y.clone(),
            Expr::cos(Box::new(x.clone() - y.clone())),
        ];

        CodegenModule::new("generated_fixture_snapshot")
            .add_function("fixture_scalar_eval", &scalar, &["x", "y"])
            .add_block("fixture_block_eval", &block_exprs, &["x", "y"])
    }

    #[test]
    fn lowering_single_expr_matches_lambdify1() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = (x.clone() + y.clone()).pow(Expr::Const(2.0)) + Expr::sin(Box::new(x.clone()))
            - Expr::Const(3.0) * y.clone();

        let ir = expr.lower_to_linear(&["x", "y"]);
        let compiled = expr.lambdify1(&["x", "y"]);
        let args = [1.25, -0.75];

        assert!((ir.eval(&args) - compiled(&args)).abs() < 1e-12);
    }

    #[test]
    fn emit_function_generates_expected_source_shape() {
        let expr = Expr::Var("x".to_string()) + Expr::Const(1.0);
        let ir = expr.lower_to_linear(&["x"]);
        let src = RustEmitter::emit_function(&ir, "foo_eval", 1);

        assert!(src.contains("pub fn foo_eval(args: &[f64]) -> f64"));
        assert!(src.contains("debug_assert!(args.len() >= 1"));
        assert!(src.contains("let t0 = args[0];"));
    }

    #[test]
    fn lower_many_preserves_output_order() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![
            x.clone() + Expr::Const(1.0),
            x.clone() * Expr::Const(2.0),
            Expr::sin(Box::new(x.clone())),
        ];

        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let values = block.eval(&[0.5]);

        assert!((values[0] - 1.5).abs() < 1e-12);
        assert!((values[1] - 1.0).abs() < 1e-12);
        assert!((values[2] - 0.5_f64.sin()).abs() < 1e-12);
    }

    #[test]
    fn repeated_variables_reuse_same_input_temp() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + x.clone() + x.clone();
        let ir = expr.lower_to_linear(&["x"]);

        assert_eq!(count_input_instrs(&ir.instructions), 1);
    }

    #[test]
    fn repeated_constants_reuse_same_const_temp() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + Expr::Const(2.0) + Expr::Const(2.0);
        let ir = expr.lower_to_linear(&["x"]);

        assert_eq!(count_const_instrs(&ir.instructions, 2.0), 1);
    }

    #[test]
    fn repeated_subexpression_within_expr_reuses_one_temp() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let shared = x.clone() + y.clone();
        let expr = shared.clone() * shared;
        let ir = expr.lower_to_linear(&["x", "y"]);

        assert_eq!(count_add_instrs(&ir.instructions), 1);
    }

    #[test]
    fn repeated_subexpression_across_outputs_reuses_one_temp() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let shared = x.clone() + y.clone();
        let exprs = vec![
            shared.clone() * Expr::Const(2.0),
            Expr::sin(Box::new(shared)),
        ];
        let block = Expr::lower_many_to_linear(&exprs, &["x", "y"]);

        assert_eq!(count_add_instrs(&block.instructions), 1);
    }

    #[test]
    fn block_eval_matches_evaluating_outputs_one_by_one() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let exprs = vec![
            x.clone() + y.clone(),
            x.clone() * y.clone(),
            Expr::cos(Box::new(x.clone() - y.clone())),
        ];
        let block = Expr::lower_many_to_linear(&exprs, &["x", "y"]);
        let args = [1.2, -0.3];

        let batch = block.eval(&args);
        let single: Vec<f64> = exprs
            .iter()
            .map(|expr| expr.lambdify1(&["x", "y"])(&args))
            .collect();

        assert_eq!(batch.len(), single.len());
        for (a, b) in batch.iter().zip(single.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn emit_block_function_generates_expected_source_shape() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone() + Expr::Const(1.0), x.clone() * Expr::Const(2.0)];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let src = RustEmitter::emit_block_function(&block, "eval_block", 1);

        assert!(src.contains("pub fn eval_block(args: &[f64], out: &mut [f64])"));
        assert!(src.contains("debug_assert!(out.len() >= 2"));
        assert!(src.contains("out[0] = "));
        assert!(src.contains("out[1] = "));
    }

    #[test]
    fn generated_block_emits_named_block_function() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone() + Expr::Const(1.0), Expr::sin(Box::new(x.clone()))];
        let block = GeneratedBlock::new("residual_block", &exprs, &["x"]);
        let src = block.emit();

        assert!(src.contains("pub fn residual_block(args: &[f64], out: &mut [f64])"));
        assert!(src.contains("out[0] = "));
        assert!(src.contains("out[1] = "));
    }

    #[test]
    fn codegen_module_emits_functions_and_blocks_together() {
        let x = Expr::Var("x".to_string());
        let scalar = x.clone() + Expr::Const(1.0);
        let block_exprs = vec![x.clone() * Expr::Const(2.0), Expr::cos(Box::new(x.clone()))];

        let module = CodegenModule::new("generated_math")
            .add_function("eval_scalar", &scalar, &["x"])
            .add_block("eval_block", &block_exprs, &["x"]);

        let src = module.emit_source();

        assert!(src.contains("pub mod generated_math"));
        assert!(src.contains("pub fn eval_scalar(args: &[f64]) -> f64"));
        assert!(src.contains("pub fn eval_block(args: &[f64], out: &mut [f64])"));
    }

    #[test]
    fn temp_plan_tracks_use_counts_and_outputs() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![x.clone() + Expr::Const(1.0), x.clone() * Expr::Const(2.0)];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let plan = block.temp_plan();

        let input_entry = plan
            .entries
            .iter()
            .find(|entry| entry.defined_at == 0)
            .expect("expected shared input temp");
        assert_eq!(input_entry.use_count, 2);

        let output_entries = plan.entries.iter().filter(|entry| entry.is_output).count();
        assert_eq!(output_entries, 2);
    }

    #[test]
    fn temp_plan_marks_non_output_temps_for_release() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let exprs = vec![(x.clone() + y.clone()) * (x.clone() - y.clone())];
        let block = Expr::lower_many_to_linear(&exprs, &["x", "y"]);
        let plan = block.temp_plan();

        let released_count: usize = plan.released_after_instruction.iter().map(Vec::len).sum();

        assert!(released_count > 0, "expected at least one releasable temp");
        assert!(
            plan.entries
                .iter()
                .any(|entry| !entry.is_output && entry.use_count > 0),
            "expected at least one non-output temp with recorded uses"
        );
    }

    #[test]
    fn temp_reuse_preserves_block_evaluation_results() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let exprs = vec![
            (x.clone() + y.clone()) * (x.clone() - y.clone()),
            Expr::sin(Box::new(x.clone() + y.clone())),
            Expr::cos(Box::new(x.clone() - y.clone())),
        ];
        let block = Expr::lower_many_to_linear(&exprs, &["x", "y"]);
        let reused = block.reuse_temps();
        let args = [1.75, -0.25];

        let original_values = block.eval(&args);
        let reused_values = reused.eval(&args);

        assert_eq!(original_values.len(), reused_values.len());
        for (a, b) in original_values.iter().zip(reused_values.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn temp_reuse_can_reduce_number_of_temps() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let exprs = vec![
            (x.clone() + y.clone()) * (x.clone() - y.clone()),
            (x.clone() + Expr::Const(1.0)) * (y.clone() + Expr::Const(2.0)),
        ];
        let block = Expr::lower_many_to_linear(&exprs, &["x", "y"]);
        let reused = block.reuse_temps();

        assert!(
            reused.num_temps <= block.num_temps,
            "temp reuse should not increase temp count"
        );
        assert!(
            reused.num_temps < block.num_temps,
            "expected temp reuse to reduce temp count for this block"
        );
    }

    #[test]
    fn peephole_eliminates_neutral_operations() {
        let x = Expr::Var("x".to_string());
        let expr = Expr::Mul(
            Box::new(Expr::Add(Box::new(x.clone()), Box::new(Expr::Const(0.0)))),
            Box::new(Expr::Const(1.0)),
        );
        let ir = expr.lower_to_linear(&["x"]);
        let optimized = ir.peephole_optimize();

        assert_eq!(count_add_instrs(&optimized.instructions), 0);
        assert_eq!(count_mul_instrs(&optimized.instructions), 0);
        assert_eq!(optimized.eval(&[2.5]), ir.eval(&[2.5]));
    }

    #[test]
    fn peephole_preserves_block_values_and_reduces_instruction_count() {
        let x = Expr::Var("x".to_string());
        let exprs = vec![
            Expr::Add(Box::new(x.clone()), Box::new(Expr::Const(0.0))),
            Expr::Mul(Box::new(x.clone()), Box::new(Expr::Const(1.0))),
            Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(1.0))),
            Expr::sin(Box::new(Expr::Const(0.0))),
        ];
        let block = Expr::lower_many_to_linear(&exprs, &["x"]);
        let optimized = block.peephole_optimize();
        let args = [1.75];

        assert_eq!(block.eval(&args), optimized.eval(&args));
        assert!(
            optimized.instructions.len() < block.instructions.len(),
            "peephole pass should remove at least one instruction for this block"
        );
    }

    #[test]
    fn generated_fixture_scalar_matches_symbolic_expression() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = (x.clone() + y.clone()).pow(Expr::Const(2.0)) + Expr::sin(Box::new(x.clone()))
            - Expr::Const(3.0) * y.clone();
        let expected = expr.lambdify1(&["x", "y"]);
        let args = [1.25, -0.75];

        assert!((fixture_scalar_eval(&args) - expected(&args)).abs() < 1e-12);
    }

    #[test]
    fn generated_fixture_block_matches_symbolic_outputs() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let exprs = [
            x.clone() + Expr::Const(1.0),
            x.clone() * y.clone(),
            Expr::cos(Box::new(x.clone() - y.clone())),
        ];
        let expected: Vec<_> = exprs
            .iter()
            .map(|expr| expr.lambdify1(&["x", "y"]))
            .collect();
        let args = [0.75, -0.5];
        let mut out = [0.0_f64; 3];

        fixture_block_eval(&args, &mut out);

        for (actual, expected_fn) in out.iter().zip(expected.iter()) {
            assert!((*actual - expected_fn(&args)).abs() < 1e-12);
        }
    }

    #[test]
    fn write_fixture_file_creates_rust_source_with_fixture_header() {
        let x = Expr::Var("x".to_string());
        let module = CodegenModule::new("generated_fixture_test").add_function(
            "eval_scalar",
            &(x.clone() + Expr::Const(1.0)),
            &["x"],
        );
        let dir = tempdir().expect("temporary directory should be created");
        let path = dir.path().join("generated_fixture.rs");

        module
            .write_fixture_file(&path, "fixture for codegen regression tests")
            .expect("fixture file should be written");

        let source = std::fs::read_to_string(&path).expect("fixture file should be readable");
        assert!(source.contains("Fixture file generated by CodegenIR."));
        assert!(source.contains("fixture for codegen regression tests"));
        assert!(source.contains("pub mod generated_fixture_test"));
        assert!(source.contains("pub fn eval_scalar(args: &[f64]) -> f64"));
    }

    #[test]
    fn generated_module_matches_checked_in_snapshot() {
        let module = build_checked_fixture_module();
        let generated = module.emit_source().replace("\r\n", "\n");
        let snapshot_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("symbolic")
            .join("codegen_generated_snapshot.rs");
        let snapshot = std::fs::read_to_string(snapshot_path)
            .expect("checked-in codegen snapshot should be readable")
            .replace("\r\n", "\n");

        assert_eq!(generated, snapshot);
    }

    #[test]
    fn generated_block_from_task_plan_matches_direct_block_eval() {
        let residuals = vec![
            Expr::Var("y".to_string()) + Expr::Const(1.0),
            Expr::Var("alpha".to_string()) * Expr::Var("y".to_string()),
        ];
        let task = ResidualTask {
            fn_name: "eval_residual",
            residuals: &residuals,
            variables: &["y"],
            params: Some(&["alpha"]),
        };
        let plan = task.plan();
        let from_plan = GeneratedBlock::from_task_plan(&plan);
        let direct = GeneratedBlock::new("eval_residual", &residuals, &["alpha", "y"]);
        let args = [2.0, 3.0];

        assert_eq!(plan.layout, CodegenOutputLayout::Vector { len: 2 });
        assert_eq!(from_plan.vars, vec!["alpha".to_string(), "y".to_string()]);
        assert_eq!(from_plan.ir.eval(&args), direct.ir.eval(&args));
    }

    #[test]
    fn codegen_module_accepts_ivp_task_plan() {
        let residuals = vec![
            Expr::Var("t".to_string()) + Expr::Var("y".to_string()),
            Expr::Var("beta".to_string()) * Expr::Var("y".to_string()),
        ];
        let task = IvpResidualTask {
            fn_name: "eval_ivp_residual",
            time_arg: "t",
            residuals: &residuals,
            variables: &["y"],
            params: Some(&["beta"]),
        };
        let plan = task.plan();
        let module = CodegenModule::new("generated_ivp").add_task_plan(&plan);
        let source = module.emit_source();

        assert_eq!(plan.input_names, vec!["t", "beta", "y"]);
        assert!(source.contains("pub mod generated_ivp"));
        assert!(source.contains("pub fn eval_ivp_residual(args: &[f64], out: &mut [f64])"));
    }

    #[test]
    fn typed_residual_block_emit_includes_residual_metadata_comment() {
        let residuals = vec![Expr::Var("y".to_string()), Expr::Const(1.0)];
        let task = ResidualTask {
            fn_name: "eval_residual",
            residuals: &residuals,
            variables: &["y"],
            params: Some(&["alpha"]),
        };
        let block = GeneratedBlock::from_residual_plan(&task.plan());
        let source = block.emit();

        assert!(source.contains("// Residual block: 2 outputs"));
        assert!(source.contains("pub fn eval_residual(args: &[f64], out: &mut [f64])"));
    }

    #[test]
    fn typed_sparse_values_emit_includes_sparse_layout_metadata() {
        let e0 = Expr::Var("y0".to_string()) + Expr::Const(1.0);
        let e1 = Expr::Var("alpha".to_string()) * Expr::Var("y1".to_string());
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 1,
                col: 2,
                expr: &e1,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (3, 4),
            entries: &entries,
            variables: &["y0", "y1"],
            params: Some(&["alpha"]),
        };
        let block = GeneratedBlock::from_sparse_values_plan(&task.plan());
        let source = block.emit();

        assert!(
            source.contains("// Sparse Jacobian values block: 3 rows x 4 cols, 2 non-zero values")
        );
        assert!(source.contains("pub fn eval_sparse_values(args: &[f64], out: &mut [f64])"));
    }

    #[test]
    fn sparse_chunk_plan_builds_generated_block_with_chunk_name() {
        let e0 = Expr::Var("y0".to_string()) + Expr::Const(1.0);
        let e1 = Expr::Var("alpha".to_string()) * Expr::Var("y1".to_string());
        let e2 = Expr::Var("y1".to_string()) + Expr::Const(2.0);
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 1,
                col: 2,
                expr: &e1,
            },
            SparseExprEntry {
                row: 2,
                col: 2,
                expr: &e2,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (3, 4),
            entries: &entries,
            variables: &["y0", "y1"],
            params: Some(&["alpha"]),
        };

        let chunks = task.chunk_by_nnz(2);
        let block = GeneratedBlock::from_sparse_values_plan(&chunks[1].plan());
        let source = block.emit();

        assert_eq!(block.fn_name, "eval_sparse_values_chunk_1");
        assert!(
            source.contains("pub fn eval_sparse_values_chunk_1(args: &[f64], out: &mut [f64])")
        );
        assert!(
            source.contains("// Sparse Jacobian values block: 3 rows x 4 cols, 1 non-zero values")
        );
    }
}
