//! IR lowering for [`AtomView`] / [`Atom`] symbolic expressions.
//!
//! Mirrors `CodegenIR` but operates on the packed byte representation instead of
//! the boxed `Expr` tree.  Because `AtomView` is `Copy` and equality is a plain
//! byte-slice comparison, the CSE cache can key directly on the raw data pointer
//! range — no cloning of expression trees is required.

use ahash::{AHasher, HashMap, HashMapExt};
use std::hash::Hasher;

use super::{
    atom::AtomView,
    state::{State, Symbol},
};

// Re-export the instruction set unchanged — it is representation-agnostic.
pub use super::super::codegen::CodegenIR::{Instr, LinearBlock, LinearExpr, Temp};

// ── TraversalCache ────────────────────────────────────────────────────────────

// ── Lowerer ───────────────────────────────────────────────────────────────────

/// Lowers [`AtomView`] expressions into a flat [`LinearExpr`] / [`LinearBlock`].
///
/// Variables are identified by their interned [`Symbol`] id and mapped to
/// positional input slots, matching the original `Lowerer` contract.
pub struct Lowerer {
    instructions: Vec<Instr>,
    next_temp: usize,
    cse_policy: AtomCsePolicy,
    /// Symbol id → input slot index.
    var_index_map: HashMap<u32, usize>,
    input_cache: HashMap<usize, Temp>,
    const_cache: HashMap<u64, Temp>,
    exact_view_cache: HashMap<(*const u8, usize), Temp>,
    /// CSE cache keyed by `(content_hash, byte_len)`.
    ///
    /// This keeps correctness by still confirming exact byte equality inside
    /// the final bucket while keeping the hot path free of an extra recursive
    /// signature traversal.
    expr_cache: HashMap<(u64, usize), Vec<(ExprCacheRef, Temp)>>,
}

enum CseLookup {
    Hit(Temp),
    Miss(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomCsePolicy {
    Enabled,
    Disabled,
}

impl Default for AtomCsePolicy {
    fn default() -> Self {
        Self::Enabled
    }
}

type NodeId = usize;

#[derive(Debug, Clone, Copy, Default)]
pub struct BlockDagStats {
    pub outputs: usize,
    pub intern_calls: usize,
    pub unique_nodes: usize,
    pub exact_hits: usize,
    pub content_hits: usize,
}

#[derive(Clone)]
enum DagNodeKind {
    Var(Symbol),
    Num(f64),
    Pow(NodeId, NodeId),
    Mul(Vec<NodeId>),
    Add(Vec<NodeId>),
    Fun(Symbol, NodeId),
}

#[derive(Clone)]
struct DagNode {
    kind: DagNodeKind,
}

// SAFETY: Lowerer is used single-threaded.
unsafe impl Send for Lowerer {}

#[derive(Copy, Clone)]
struct ExprCacheRef {
    ptr: *const u8,
    len: usize,
}

impl ExprCacheRef {
    fn from_view(view: AtomView<'_>) -> Self {
        let data = view.get_data();
        Self {
            ptr: data.as_ptr(),
            len: data.len(),
        }
    }

    fn equals(self, other: &[u8]) -> bool {
        if self.len != other.len() {
            return false;
        }
        // SAFETY: entries in `expr_cache` are only used during one lowering
        // session, so the underlying atom buffers outlive the cache itself.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) == other }
    }
}

impl Lowerer {
    /// Build a lowerer where `vars` lists the [`Symbol`]s that map to input slots
    /// in order.
    pub fn new(vars: &[Symbol]) -> Self {
        Self::new_with_cse_policy(vars, AtomCsePolicy::Enabled)
    }

    /// Build a lowerer with explicit structural CSE control.
    ///
    /// Disabling CSE is meant for diagnostics and benchmarking. The default
    /// production route keeps CSE enabled.
    pub fn new_with_cse_policy(vars: &[Symbol], cse_policy: AtomCsePolicy) -> Self {
        let var_index_map = vars.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
        Self {
            instructions: Vec::new(),
            next_temp: 0,
            cse_policy,
            var_index_map,
            input_cache: HashMap::new(),
            const_cache: HashMap::new(),
            exact_view_cache: HashMap::new(),
            expr_cache: HashMap::new(),
        }
    }

    fn fresh(&mut self) -> Temp {
        let t = Temp(self.next_temp);
        self.next_temp += 1;
        t
    }

    fn lower_var(&mut self, sym: Symbol) -> Temp {
        let idx = *self
            .var_index_map
            .get(&sym.id)
            .unwrap_or_else(|| panic!("Symbol '{}' not in vars list", sym));
        if let Some(&t) = self.input_cache.get(&idx) {
            return t;
        }
        let dst = self.fresh();
        self.instructions.push(Instr::Input { dst, index: idx });
        self.input_cache.insert(idx, dst);
        dst
    }

    fn lower_num(&mut self, view: AtomView<'_>) -> Temp {
        // Use the raw bytes as the cache key (handles all coefficient types).
        let key = hash_bytes(0, view.get_data());
        if let Some(&t) = self.const_cache.get(&key) {
            return t;
        }
        // Extract f64 from the coefficient.
        let value = coeff_to_f64(view);
        let dst = self.fresh();
        self.instructions.push(Instr::Const { dst, value });
        self.const_cache.insert(key, dst);
        dst
    }

    fn cse_lookup(&mut self, view: AtomView<'_>) -> CseLookup {
        let data = view.get_data();
        if let Some(&t) = self.exact_view_cache.get(&(data.as_ptr(), data.len())) {
            return CseLookup::Hit(t);
        }
        let content_hash = hash_bytes(1, data);
        if let Some(t) = self
            .expr_cache
            .get(&(content_hash, data.len()))
            .and_then(|bucket| {
                bucket
                    .iter()
                    .find(|(bytes, _)| bytes.equals(data))
                    .map(|(_, t)| *t)
            })
        {
            CseLookup::Hit(t)
        } else {
            CseLookup::Miss(content_hash)
        }
    }

    fn cse_insert(&mut self, view: AtomView<'_>, content_hash: u64, t: Temp) {
        let data = view.get_data();
        self.exact_view_cache.insert((data.as_ptr(), data.len()), t);
        self.expr_cache
            .entry((content_hash, data.len()))
            .or_default()
            .push((ExprCacheRef::from_view(view), t));
    }

    fn lower_view(&mut self, view: AtomView<'_>) -> Temp {
        // Vars and nums have their own dedicated caches; skip CSE for them.
        match view {
            AtomView::Var(v) => return self.lower_var(v.get_symbol()),
            AtomView::Num(_) => return self.lower_num(view),
            _ => {}
        }

        if self.cse_policy == AtomCsePolicy::Disabled {
            return self.lower_composite(view);
        }

        match self.cse_lookup(view) {
            CseLookup::Hit(t) => t,
            CseLookup::Miss(content_hash) => {
                let t = self.lower_composite(view);
                self.cse_insert(view, content_hash, t);
                t
            }
        }
    }

    fn lower_composite(&mut self, view: AtomView<'_>) -> Temp {
        match view {
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let tb = self.lower_view(base);
                let te = self.lower_view(exp);
                let dst = self.fresh();
                self.instructions.push(Instr::Pow {
                    dst,
                    base: tb,
                    exp: te,
                });
                dst
            }
            AtomView::Mul(m) => {
                let mut it = m.iter();
                let first = it.next().expect("empty Mul");
                let second = it.next();
                let Some(second) = second else {
                    return self.lower_view(first);
                };
                let third = it.next();
                let t_first = self.lower_view(first);
                let t_second = self.lower_view(second);
                let Some(third) = third else {
                    let dst = self.fresh();
                    self.instructions.push(Instr::Mul {
                        dst,
                        a: t_first,
                        b: t_second,
                    });
                    return dst;
                };

                let mut acc = {
                    let dst = self.fresh();
                    self.instructions.push(Instr::Mul {
                        dst,
                        a: t_first,
                        b: t_second,
                    });
                    dst
                };
                {
                    let tc = self.lower_view(third);
                    let dst = self.fresh();
                    self.instructions.push(Instr::Mul { dst, a: acc, b: tc });
                    acc = dst;
                }
                for child in it {
                    let tc = self.lower_view(child);
                    let dst = self.fresh();
                    self.instructions.push(Instr::Mul { dst, a: acc, b: tc });
                    acc = dst;
                }
                acc
            }
            AtomView::Add(a) => {
                let mut it = a.iter();
                let first = it.next().expect("empty Add");
                let second = it.next();
                let Some(second) = second else {
                    return self.lower_view(first);
                };
                let third = it.next();
                let t_first = self.lower_view(first);
                let t_second = self.lower_view(second);
                let Some(third) = third else {
                    let dst = self.fresh();
                    self.instructions.push(Instr::Add {
                        dst,
                        a: t_first,
                        b: t_second,
                    });
                    return dst;
                };

                let mut acc = {
                    let dst = self.fresh();
                    self.instructions.push(Instr::Add {
                        dst,
                        a: t_first,
                        b: t_second,
                    });
                    dst
                };
                {
                    let tc = self.lower_view(third);
                    let dst = self.fresh();
                    self.instructions.push(Instr::Add { dst, a: acc, b: tc });
                    acc = dst;
                }
                for child in it {
                    let tc = self.lower_view(child);
                    let dst = self.fresh();
                    self.instructions.push(Instr::Add { dst, a: acc, b: tc });
                    acc = dst;
                }
                acc
            }
            AtomView::Fun(f) => self.lower_fun(f.get_symbol(), view),
            // Var / Num handled above
            _ => unreachable!(),
        }
    }

    fn lower_fun(&mut self, sym: Symbol, view: AtomView<'_>) -> Temp {
        // Single-argument built-in functions.
        let AtomView::Fun(f) = view else {
            unreachable!()
        };
        let mut args = f.iter();
        let arg = args.next().expect("function with no argument");
        let tx = self.lower_view(arg);
        let dst = self.fresh();

        let instr = match sym {
            s if s == State::EXP => Instr::Exp { dst, x: tx },
            s if s == State::LOG => Instr::Ln { dst, x: tx },
            s if s == State::SIN => Instr::Sin { dst, x: tx },
            s if s == State::COS => Instr::Cos { dst, x: tx },
            s if s == State::TAN => Instr::Tg { dst, x: tx },
            s if s == State::COT => Instr::Ctg { dst, x: tx },
            s if s == State::ASIN => Instr::ArcSin { dst, x: tx },
            s if s == State::ACOS => Instr::ArcCos { dst, x: tx },
            s if s == State::ATAN => Instr::ArcTg { dst, x: tx },
            s if s == State::ACOT => Instr::ArcCtg { dst, x: tx },
            other => panic!("Unsupported function symbol: {}", other),
        };
        self.instructions.push(instr);
        dst
    }

    /// Lower a single expression.
    pub fn lower(mut self, view: AtomView<'_>) -> LinearExpr {
        let output = self.lower_view(view);
        LinearExpr {
            instructions: self.instructions,
            output,
            num_temps: self.next_temp,
        }
    }

    /// Lower multiple expressions sharing a single instruction stream (CSE across outputs).
    pub fn lower_many(mut self, views: &[AtomView<'_>]) -> LinearBlock {
        if self.cse_policy == AtomCsePolicy::Disabled {
            let outputs = views.iter().map(|&view| self.lower_view(view)).collect();
            return LinearBlock {
                instructions: self.instructions,
                outputs,
                num_temps: self.next_temp,
            };
        }

        let dag = BlockDag::build(views);
        let outputs = dag
            .outputs
            .iter()
            .map(|&node_id| self.lower_dag_node(&dag, node_id))
            .collect();
        LinearBlock {
            instructions: self.instructions,
            outputs,
            num_temps: self.next_temp,
        }
    }

    fn lower_dag_node(&mut self, dag: &BlockDag, node_id: NodeId) -> Temp {
        if let Some(temp) = dag.lowered[node_id].get() {
            return temp;
        }

        let temp = match &dag.nodes[node_id].kind {
            DagNodeKind::Var(sym) => self.lower_var(*sym),
            DagNodeKind::Num(value) => self.lower_const_value(*value),
            DagNodeKind::Pow(base, exp) => {
                let tb = self.lower_dag_node(dag, *base);
                let te = self.lower_dag_node(dag, *exp);
                let dst = self.fresh();
                self.instructions.push(Instr::Pow {
                    dst,
                    base: tb,
                    exp: te,
                });
                dst
            }
            DagNodeKind::Mul(children) => self.lower_chain_from_nodes(dag, children, true),
            DagNodeKind::Add(children) => self.lower_chain_from_nodes(dag, children, false),
            DagNodeKind::Fun(sym, arg) => {
                let tx = self.lower_dag_node(dag, *arg);
                let dst = self.fresh();
                let instr = match *sym {
                    s if s == State::EXP => Instr::Exp { dst, x: tx },
                    s if s == State::LOG => Instr::Ln { dst, x: tx },
                    s if s == State::SIN => Instr::Sin { dst, x: tx },
                    s if s == State::COS => Instr::Cos { dst, x: tx },
                    s if s == State::TAN => Instr::Tg { dst, x: tx },
                    s if s == State::COT => Instr::Ctg { dst, x: tx },
                    s if s == State::ASIN => Instr::ArcSin { dst, x: tx },
                    s if s == State::ACOS => Instr::ArcCos { dst, x: tx },
                    s if s == State::ATAN => Instr::ArcTg { dst, x: tx },
                    s if s == State::ACOT => Instr::ArcCtg { dst, x: tx },
                    other => panic!("Unsupported function symbol: {}", other),
                };
                self.instructions.push(instr);
                dst
            }
        };

        dag.lowered[node_id].set(Some(temp));
        temp
    }

    fn lower_const_value(&mut self, value: f64) -> Temp {
        let key = value.to_bits();
        if let Some(&t) = self.const_cache.get(&key) {
            return t;
        }
        let dst = self.fresh();
        self.instructions.push(Instr::Const { dst, value });
        self.const_cache.insert(key, dst);
        dst
    }

    fn lower_chain_from_nodes(
        &mut self,
        dag: &BlockDag,
        children: &[NodeId],
        is_mul: bool,
    ) -> Temp {
        let mut iter = children.iter().copied();
        let first = iter.next().expect("empty composite");
        let second = iter.next();
        let Some(second) = second else {
            return self.lower_dag_node(dag, first);
        };

        let mut acc = {
            let ta = self.lower_dag_node(dag, first);
            let tb = self.lower_dag_node(dag, second);
            let dst = self.fresh();
            if is_mul {
                self.instructions.push(Instr::Mul { dst, a: ta, b: tb });
            } else {
                self.instructions.push(Instr::Add { dst, a: ta, b: tb });
            }
            dst
        };

        for child in iter {
            let tc = self.lower_dag_node(dag, child);
            let dst = self.fresh();
            if is_mul {
                self.instructions.push(Instr::Mul { dst, a: acc, b: tc });
            } else {
                self.instructions.push(Instr::Add { dst, a: acc, b: tc });
            }
            acc = dst;
        }

        acc
    }
}

struct BlockDag {
    nodes: Vec<DagNode>,
    outputs: Vec<NodeId>,
    lowered: Vec<std::cell::Cell<Option<Temp>>>,
}

impl BlockDag {
    fn build(views: &[AtomView<'_>]) -> Self {
        let mut builder = BlockDagBuilder {
            nodes: Vec::new(),
            exact_index: HashMap::new(),
            index: HashMap::new(),
            stats: BlockDagStats {
                outputs: views.len(),
                ..Default::default()
            },
        };
        let outputs = views
            .iter()
            .map(|&view| builder.intern(view))
            .collect::<Vec<_>>();
        let lowered = (0..builder.nodes.len())
            .map(|_| std::cell::Cell::new(None))
            .collect();
        Self {
            nodes: builder.nodes,
            outputs,
            lowered,
        }
    }

    pub fn stats_for_views(views: &[AtomView<'_>]) -> BlockDagStats {
        let mut builder = BlockDagBuilder {
            nodes: Vec::new(),
            exact_index: HashMap::new(),
            index: HashMap::new(),
            stats: BlockDagStats {
                outputs: views.len(),
                ..Default::default()
            },
        };
        for &view in views {
            let _ = builder.intern(view);
        }
        builder.stats.unique_nodes = builder.nodes.len();
        builder.stats
    }
}

impl BlockDagStats {
    pub fn for_views(views: &[AtomView<'_>]) -> Self {
        BlockDag::stats_for_views(views)
    }
}

struct BlockDagBuilder {
    nodes: Vec<DagNode>,
    exact_index: HashMap<(*const u8, usize), NodeId>,
    index: HashMap<(u64, usize), Vec<(ExprCacheRef, NodeId)>>,
    stats: BlockDagStats,
}

impl BlockDagBuilder {
    fn intern(&mut self, view: AtomView<'_>) -> NodeId {
        self.stats.intern_calls += 1;
        let data = view.get_data();
        if let Some(&node_id) = self.exact_index.get(&(data.as_ptr(), data.len())) {
            self.stats.exact_hits += 1;
            return node_id;
        }
        let content_hash = hash_bytes(7, data);
        if let Some(node_id) = self
            .index
            .get(&(content_hash, data.len()))
            .and_then(|bucket| {
                bucket
                    .iter()
                    .find(|(bytes, _)| bytes.equals(data))
                    .map(|(_, node_id)| *node_id)
            })
        {
            self.stats.content_hits += 1;
            return node_id;
        }

        let kind = match view {
            AtomView::Var(v) => DagNodeKind::Var(v.get_symbol()),
            AtomView::Num(_) => DagNodeKind::Num(coeff_to_f64(view)),
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                DagNodeKind::Pow(self.intern(base), self.intern(exp))
            }
            AtomView::Mul(m) => {
                DagNodeKind::Mul(m.iter().map(|child| self.intern(child)).collect())
            }
            AtomView::Add(a) => {
                DagNodeKind::Add(a.iter().map(|child| self.intern(child)).collect())
            }
            AtomView::Fun(f) => {
                let mut args = f.iter();
                let arg = args.next().expect("function with no argument");
                DagNodeKind::Fun(f.get_symbol(), self.intern(arg))
            }
            _ => unreachable!(),
        };

        let node_id = self.nodes.len();
        self.nodes.push(DagNode { kind });
        self.exact_index
            .insert((data.as_ptr(), data.len()), node_id);
        self.index
            .entry((content_hash, data.len()))
            .or_default()
            .push((ExprCacheRef::from_view(view), node_id));
        node_id
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn combine_sig(tag: u64, parts: &[u64]) -> u64 {
    let mut acc = 0x9E37_79B9_7F4A_7C15u64 ^ tag;
    for p in parts {
        acc = combine_sig_step(acc, *p);
    }
    acc
}

fn combine_sig_step(acc: u64, part: u64) -> u64 {
    acc ^ part
        .wrapping_add(0x9E37_79B9_7F4A_7C15u64)
        .wrapping_add(acc << 6)
        .wrapping_add(acc >> 2)
}

fn hash_bytes(seed: usize, data: &[u8]) -> u64 {
    let mut hasher = AHasher::default();
    hasher.write_usize(seed);
    hasher.write(data);
    hasher.finish()
}

/// Convert a numeric [`AtomView`] to `f64` via its coefficient.
fn coeff_to_f64(view: AtomView<'_>) -> f64 {
    use super::coefficient::CoefficientView;
    if let AtomView::Num(n) = view {
        match n.get_coeff_view() {
            CoefficientView::Natural(num, den) => num as f64 / den as f64,
            CoefficientView::Large(_) => f64::NAN,
        }
    } else {
        f64::NAN
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::View::atom::Atom;
    use crate::{function, parse, symbol};
    fn eval(ir: &LinearExpr, inputs: &[f64]) -> f64 {
        let mut regs = vec![0f64; ir.num_temps];
        for instr in &ir.instructions {
            match *instr {
                Instr::Const { dst, value } => regs[dst.0] = value,
                Instr::Input { dst, index } => regs[dst.0] = inputs[index],
                Instr::Add { dst, a, b } => regs[dst.0] = regs[a.0] + regs[b.0],
                Instr::Sub { dst, a, b } => regs[dst.0] = regs[a.0] - regs[b.0],
                Instr::Mul { dst, a, b } => regs[dst.0] = regs[a.0] * regs[b.0],
                Instr::Div { dst, a, b } => regs[dst.0] = regs[a.0] / regs[b.0],
                Instr::Pow { dst, base, exp } => regs[dst.0] = regs[base.0].powf(regs[exp.0]),
                Instr::Exp { dst, x } => regs[dst.0] = regs[x.0].exp(),
                Instr::Ln { dst, x } => regs[dst.0] = regs[x.0].ln(),
                Instr::Sin { dst, x } => regs[dst.0] = regs[x.0].sin(),
                Instr::Cos { dst, x } => regs[dst.0] = regs[x.0].cos(),
                Instr::Tg { dst, x } => regs[dst.0] = regs[x.0].tan(),
                Instr::Ctg { dst, x } => regs[dst.0] = 1.0 / regs[x.0].tan(),
                Instr::ArcSin { dst, x } => regs[dst.0] = regs[x.0].asin(),
                Instr::ArcCos { dst, x } => regs[dst.0] = regs[x.0].acos(),
                Instr::ArcTg { dst, x } => regs[dst.0] = regs[x.0].atan(),
                Instr::ArcCtg { dst, x } => {
                    regs[dst.0] = std::f64::consts::FRAC_PI_2 - regs[x.0].atan()
                }
            }
        }
        regs[ir.output.0]
    }

    #[test]
    fn test_constant() {
        let expr = Atom::new_num(42);
        let ir = Lowerer::new(&[]).lower(expr.as_view());
        assert_eq!(ir.num_temps, 1);
        assert!((eval(&ir, &[]) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_variable_input() {
        let x = symbol!("x");
        let expr = Atom::new_var(x);
        let ir = Lowerer::new(&[x]).lower(expr.as_view());
        assert!((eval(&ir, &[7.0]) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_add_mul() {
        // x * 2 + 3  at x=5 → 13
        let expr = parse!("x*2+3").unwrap();
        let x = symbol!("x");
        let ir = Lowerer::new(&[x]).lower(expr.as_view());
        assert!((eval(&ir, &[5.0]) - 13.0).abs() < 1e-12);
    }

    #[test]
    fn test_pow() {
        // x^3 at x=2 → 8
        let expr = parse!("x^3").unwrap();
        let x = symbol!("x");
        let ir = Lowerer::new(&[x]).lower(expr.as_view());
        assert!((eval(&ir, &[2.0]) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_trig_functions() {
        let expr = parse!("sin(x)+cos(x)").unwrap();
        let x = symbol!("x");
        let ir = Lowerer::new(&[x]).lower(expr.as_view());
        let v = std::f64::consts::FRAC_PI_4;
        let expected = v.sin() + v.cos();
        assert!((eval(&ir, &[v]) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_exp_log() {
        // exp(log(x)) = x
        let expr = parse!("exp(log(x))").unwrap();
        let x = symbol!("x");
        let ir = Lowerer::new(&[x]).lower(expr.as_view());
        assert!((eval(&ir, &[3.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cse_shared_subexpr() {
        // (x+1)*(x+1) — the subexpression x+1 should be computed once.
        let expr = parse!("(x+1)*(x+1)").unwrap();
        let x = symbol!("x");
        let ir = Lowerer::new(&[x]).lower(expr.as_view());
        // After normalization Symbolica may fold this to (x+1)^2; either way
        // the result at x=3 must be 16.
        assert!((eval(&ir, &[3.0]) - 16.0).abs() < 1e-12);
    }

    #[test]
    fn cse_policy_preserves_values_when_disabled() {
        let expr = parse!("(x+y)*(x+y)+(x+y)").unwrap();
        let (x, y) = symbol!("x", "y");
        let enabled =
            Lowerer::new_with_cse_policy(&[x, y], AtomCsePolicy::Enabled).lower(expr.as_view());
        let disabled =
            Lowerer::new_with_cse_policy(&[x, y], AtomCsePolicy::Disabled).lower(expr.as_view());

        for args in [[1.0, 2.0], [-0.5, 3.25], [4.0, -1.0]] {
            let a = eval(&enabled, &args);
            let b = eval(&disabled, &args);
            assert!(
                (a - b).abs() < 1e-12,
                "CSE policy must not change values: enabled={a}, disabled={b}"
            );
        }
    }

    #[test]
    fn cse_policy_reduces_repeated_subexpression_ir() {
        let e1 = parse!("sin(x+y)+exp(x+y)").unwrap();
        let e2 = parse!("sin(x+y)+exp(x+y)").unwrap();
        let (x, y) = symbol!("x", "y");
        let enabled = Lowerer::new_with_cse_policy(&[x, y], AtomCsePolicy::Enabled)
            .lower_many(&[e1.as_view(), e2.as_view()]);
        let disabled = Lowerer::new_with_cse_policy(&[x, y], AtomCsePolicy::Disabled)
            .lower_many(&[e1.as_view(), e2.as_view()]);

        assert_eq!(enabled.eval(&[2.0, 3.0]), disabled.eval(&[2.0, 3.0]));
        assert!(
            enabled.instructions.len() < disabled.instructions.len(),
            "enabled CSE should reduce repeated-subexpression IR size: enabled={}, disabled={}",
            enabled.instructions.len(),
            disabled.instructions.len()
        );
    }

    #[test]
    fn test_lower_many_shared_stream() {
        // Two outputs sharing the same instruction stream.
        let e1 = parse!("x+y").unwrap();
        let e2 = parse!("x*y").unwrap();
        let (x, y) = symbol!("x", "y");
        let block = Lowerer::new(&[x, y]).lower_many(&[e1.as_view(), e2.as_view()]);
        let mut regs = vec![0f64; block.num_temps];
        for instr in &block.instructions {
            match *instr {
                Instr::Const { dst, value } => regs[dst.0] = value,
                Instr::Input { dst, index } => regs[dst.0] = [2.0_f64, 3.0][index],
                Instr::Add { dst, a, b } => regs[dst.0] = regs[a.0] + regs[b.0],
                Instr::Mul { dst, a, b } => regs[dst.0] = regs[a.0] * regs[b.0],
                _ => {}
            }
        }
        assert!((regs[block.outputs[0].0] - 5.0).abs() < 1e-12); // x+y
        assert!((regs[block.outputs[1].0] - 6.0).abs() < 1e-12); // x*y
    }

    #[test]
    fn test_two_vars() {
        // (x - y) / (x + y) at x=5, y=3 → 0.25
        let expr = parse!("(x-y)/(x+y)").unwrap();
        let (x, y) = symbol!("x", "y");
        let ir = Lowerer::new(&[x, y]).lower(expr.as_view());
        let result = eval(&ir, &[5.0, 3.0]);
        assert!((result - 0.25).abs() < 1e-12);
    }
}
