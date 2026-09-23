//! Numerical evaluation for the simplified crate.
//!
//! The evaluator keeps the original recursive interpretation model but narrows it to
//! `f64`. Expressions are evaluated directly from packed [`AtomView`] data, and custom
//! function calls are memoized by exact atom value so repeated subexpressions do not
//! recompute. This keeps the evaluation code small while still preserving the zero-copy
//! packed-expression design of the rest of the crate.

use std::{
    borrow::Borrow,
    cell::RefCell,
    hash::{Hash, Hasher},
    sync::Arc,
};

use ahash::HashMap;
use once_cell::sync::Lazy;

use super::{
    atom::{
        Atom, AtomView,
        representation::{BorrowedRawAtom, KeyLookup},
    },
    coefficient::{Coefficient, CoefficientView},
    state::Symbol,
};

/// Cache of previously evaluated function atoms.
#[derive(Clone, Default)]
pub struct EvaluationCache {
    map: HashMap<PackedAtomKey, f64>,
}

#[derive(Clone, Eq, PartialEq)]
struct PackedAtomKey(Box<[u8]>);

impl Borrow<BorrowedRawAtom> for PackedAtomKey {
    fn borrow(&self) -> &BorrowedRawAtom {
        &self.0
    }
}

impl Hash for PackedAtomKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PackedAtomKey {
    #[inline]
    fn from_view(view: AtomView<'_>) -> Self {
        Self(view.get_data().into())
    }
}

impl EvaluationCache {
    #[inline]
    pub fn get_view(&self, key: AtomView<'_>) -> Option<&f64> {
        self.map.get(key.get_data() as &BorrowedRawAtom)
    }

    #[inline]
    pub fn insert_view(&mut self, key: AtomView<'_>, value: f64) -> Option<f64> {
        self.map.insert(PackedAtomKey::from_view(key), value)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    #[inline]
    fn clear(&mut self) {
        self.map.clear();
    }
}

/// Constant bindings for exact rational evaluation keyed by full atoms.
pub type ExactConstMap = HashMap<Atom, Coefficient>;
/// Constant bindings for floating-point evaluation keyed by symbols.
pub type FloatSymbolMap = HashMap<Symbol, f64>;
/// Constant bindings for exact rational evaluation keyed by symbols.
pub type ExactSymbolMap = HashMap<Symbol, Coefficient>;
/// Callback type used to evaluate user-defined functions.
pub type EvaluationFn = Arc<
    dyn Fn(&[f64], &HashMap<Atom, f64>, &FunctionMap, &mut EvaluationCache) -> Result<f64, String>
        + Send
        + Sync,
>;
/// Callback type used to evaluate user-defined functions with symbol-keyed bindings.
pub type EvaluationSymbolFn = Arc<
    dyn Fn(&[f64], &FloatSymbolMap, &FunctionMap, &mut EvaluationCache) -> Result<f64, String>
        + Send
        + Sync,
>;

static EMPTY_ATOM_CONST_MAP: Lazy<HashMap<Atom, f64>> = Lazy::new(HashMap::default);

/// Registry of numeric callbacks for symbolic function names.
#[derive(Clone, Default)]
pub struct FunctionMap {
    functions: HashMap<Symbol, EvaluationFn>,
    symbol_functions: HashMap<Symbol, EvaluationSymbolFn>,
}

/// A compiled numeric evaluation plan for repeated evaluation of the same expression.
#[derive(Clone)]
pub struct PreparedEvaluator {
    nodes: Vec<PreparedNode>,
    root: usize,
    vars: Arc<[Symbol]>,
    var_atoms: Arc<[Atom]>,
    function_map: FunctionMap,
    // BVP residual/Jacobian expressions normally contain only numeric nodes and
    // builtins.  Keep that fact so their hot callback path can skip the custom
    // function maps and cache bookkeeping entirely.
    plain_numeric: bool,
}

/// Reusable scratch storage for one prepared evaluation on one worker thread.
///
/// `PreparedEvaluator::evaluate` remains allocation-safe and keeps its public
/// behavior, but the hot Lambdify closure uses this workspace through a
/// thread-local.  AtomView BVP callbacks evaluate many scalar expressions in
/// parallel; allocating a fresh `results` vector for every scalar entry made
/// that route substantially slower than the Expr callback even though both
/// had the same symbolic sparsity.
#[derive(Default)]
struct EvaluationWorkspace {
    results: Vec<f64>,
    cache: EvaluationCache,
    symbol_map: FloatSymbolMap,
    atom_map: HashMap<Atom, f64>,
    arg_buffer: Vec<f64>,
}

thread_local! {
    static EVALUATION_WORKSPACE: RefCell<EvaluationWorkspace> =
        RefCell::new(EvaluationWorkspace::default());
}

/// Immutable variable ABI shared by a batch of prepared evaluators.
///
/// BVP residual and Jacobian entries all use the same flattened argument
/// ordering. Keeping the index and variable atoms in one shared context avoids
/// rebuilding O(number_of_variables) metadata for every scalar callback.
pub(crate) struct PreparedVariableContext {
    pub(crate) vars: Arc<[Symbol]>,
    pub(crate) var_atoms: Arc<[Atom]>,
    pub(crate) var_index: HashMap<Symbol, usize>,
}

impl PreparedVariableContext {
    pub(crate) fn new(vars: &[Symbol]) -> Self {
        let vars: Arc<[Symbol]> = vars.to_vec().into();
        let var_atoms: Arc<[Atom]> = vars.iter().copied().map(Atom::new_var).collect();
        let var_index = vars
            .iter()
            .copied()
            .enumerate()
            .map(|(i, symbol)| (symbol, i))
            .collect();
        Self {
            vars,
            var_atoms,
            var_index,
        }
    }
}

#[derive(Clone)]
enum PreparedNode {
    Const(f64),
    Var(usize),
    Add(Box<[usize]>),
    Mul(Box<[usize]>),
    PowI { base: usize, exponent: i32 },
    Pow { base: usize, exp: usize },
    Builtin { symbol: Symbol, arg: usize },
    Custom { symbol: Symbol, args: Box<[usize]> },
}

impl FunctionMap {
    /// Create an empty function registry.
    pub fn new() -> Self {
        Self {
            functions: HashMap::default(),
            symbol_functions: HashMap::default(),
        }
    }

    /// Register or replace the callback for a symbolic function.
    pub fn insert(&mut self, symbol: Symbol, function: EvaluationFn) -> Option<EvaluationFn> {
        self.functions.insert(symbol, function)
    }

    /// Register or replace the callback for a symbolic function in symbol-keyed evaluation.
    pub fn insert_symbol(
        &mut self,
        symbol: Symbol,
        function: EvaluationSymbolFn,
    ) -> Option<EvaluationSymbolFn> {
        self.symbol_functions.insert(symbol, function)
    }

    /// Register a closure as a function callback without manually wrapping it in [`Arc`].
    pub fn insert_fn<F>(&mut self, symbol: Symbol, function: F) -> Option<EvaluationFn>
    where
        F: Fn(
                &[f64],
                &HashMap<Atom, f64>,
                &FunctionMap,
                &mut EvaluationCache,
            ) -> Result<f64, String>
            + Send
            + Sync
            + 'static,
    {
        self.insert(symbol, Arc::new(function))
    }

    /// Register a closure for symbol-keyed floating-point evaluation.
    pub fn insert_symbol_fn<F>(&mut self, symbol: Symbol, function: F) -> Option<EvaluationSymbolFn>
    where
        F: Fn(&[f64], &FloatSymbolMap, &FunctionMap, &mut EvaluationCache) -> Result<f64, String>
            + Send
            + Sync
            + 'static,
    {
        self.insert_symbol(symbol, Arc::new(function))
    }

    /// Look up a callback by function symbol.
    pub fn get(&self, symbol: &Symbol) -> Option<&EvaluationFn> {
        self.functions.get(symbol)
    }

    /// Look up a symbol-aware callback by function symbol.
    pub fn get_symbol(&self, symbol: &Symbol) -> Option<&EvaluationSymbolFn> {
        self.symbol_functions.get(symbol)
    }

    /// Return whether the registry contains any custom functions.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.symbol_functions.is_empty()
    }
}

impl PreparedEvaluator {
    /// Compile `atom` into a reusable numeric evaluation plan.
    pub fn new(atom: &Atom, vars: &[Symbol], function_map: &FunctionMap) -> Result<Self, String> {
        let context = PreparedVariableContext::new(vars);
        Self::new_with_context(atom, &context, function_map)
    }

    /// Compile an atom while reusing a caller-owned variable ABI.
    ///
    /// Large BVP batches compile many scalar residual/Jacobian entries against
    /// the same argument ABI. Rebuilding the symbol-to-index map for every
    /// entry makes cold preparation quadratic in the number of mesh unknowns.
    /// The map is immutable during compilation, so it can safely be shared by
    /// all Rayon workers.
    pub(crate) fn new_with_context(
        atom: &Atom,
        context: &PreparedVariableContext,
        function_map: &FunctionMap,
    ) -> Result<Self, String> {
        let mut compiler = PreparedCompiler {
            nodes: Vec::new(),
            cache: HashMap::default(),
            var_index: &context.var_index,
        };
        let root = compiler.compile_view(atom.as_view())?;
        let plain_numeric = function_map.is_empty()
            && compiler
                .nodes
                .iter()
                .all(|node| !matches!(node, PreparedNode::Custom { .. }));

        Ok(Self {
            nodes: compiler.nodes,
            root,
            vars: Arc::clone(&context.vars),
            var_atoms: Arc::clone(&context.var_atoms),
            function_map: function_map.clone(),
            plain_numeric,
        })
    }

    /// Return the ordered variables expected by [`Self::evaluate`].
    pub fn variables(&self) -> &[Symbol] {
        &self.vars
    }

    /// Evaluate the prepared plan using the function map captured at compile time.
    pub fn evaluate(&self, values: &[f64]) -> Result<f64, String> {
        self.evaluate_with_function_map(values, &self.function_map)
    }

    /// Evaluate the prepared plan with an explicit function map override.
    pub fn evaluate_with_function_map(
        &self,
        values: &[f64],
        function_map: &FunctionMap,
    ) -> Result<f64, String> {
        let mut workspace = EvaluationWorkspace::default();
        self.evaluate_with_workspace(values, function_map, &mut workspace)
    }

    /// Evaluate using scratch storage local to the current worker thread.
    ///
    /// This is intentionally crate-visible: public callers retain the simple
    /// allocating `evaluate` API, while prepared callback builders can opt into
    /// the reuse contract without exposing mutable evaluator state or a Mutex.
    pub(crate) fn evaluate_thread_local(&self, values: &[f64]) -> Result<f64, String> {
        EVALUATION_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            if self.plain_numeric {
                self.evaluate_plain_numeric(values, &mut workspace)
            } else {
                self.evaluate_with_workspace(values, &self.function_map, &mut workspace)
            }
        })
    }

    /// Evaluate the common AtomView numeric subset without initializing the
    /// general custom-function evaluation state.  This is the path used by
    /// BVP scalar callbacks; custom symbolic functions continue to use the
    /// fully general evaluator below.
    fn evaluate_plain_numeric(
        &self,
        values: &[f64],
        workspace: &mut EvaluationWorkspace,
    ) -> Result<f64, String> {
        if values.len() != self.vars.len() {
            return Err(format!(
                "Prepared evaluator expected {} argument(s), got {}",
                self.vars.len(),
                values.len()
            ));
        }

        workspace.results.resize(self.nodes.len(), 0.0);
        let results = &mut workspace.results;
        for (index, node) in self.nodes.iter().enumerate() {
            results[index] = match node {
                PreparedNode::Const(value) => *value,
                PreparedNode::Var(var_index) => values[*var_index],
                PreparedNode::Add(args) => args.iter().map(|i| results[*i]).sum(),
                PreparedNode::Mul(args) => args.iter().map(|i| results[*i]).product(),
                PreparedNode::PowI { base, exponent } => results[*base].powi(*exponent),
                PreparedNode::Pow { base, exp } => results[*base].powf(results[*exp]),
                PreparedNode::Builtin { symbol, arg } => {
                    evaluate_builtin_function(*symbol, results[*arg])?
                }
                PreparedNode::Custom { symbol, .. } => {
                    return Err(format!(
                        "Custom function {} requires the general evaluator",
                        symbol
                    ));
                }
            };
        }

        Ok(results[self.root])
    }

    fn evaluate_with_workspace(
        &self,
        values: &[f64],
        function_map: &FunctionMap,
        workspace: &mut EvaluationWorkspace,
    ) -> Result<f64, String> {
        if values.len() != self.vars.len() {
            return Err(format!(
                "Prepared evaluator expected {} argument(s), got {}",
                self.vars.len(),
                values.len()
            ));
        }

        workspace.results.resize(self.nodes.len(), 0.0);
        workspace.cache.clear();
        workspace.symbol_map.clear();
        workspace.atom_map.clear();
        workspace.arg_buffer.clear();

        let results = &mut workspace.results;
        let cache = &mut workspace.cache;
        let symbol_map = &mut workspace.symbol_map;
        let atom_map = &mut workspace.atom_map;
        let arg_buffer = &mut workspace.arg_buffer;
        let mut symbol_map_ready = false;
        let mut atom_map_ready = false;

        for (index, node) in self.nodes.iter().enumerate() {
            results[index] = match node {
                PreparedNode::Const(value) => *value,
                PreparedNode::Var(var_index) => values[*var_index],
                PreparedNode::Add(args) => args.iter().map(|i| results[*i]).sum(),
                PreparedNode::Mul(args) => args.iter().map(|i| results[*i]).product(),
                PreparedNode::PowI { base, exponent } => results[*base].powi(*exponent),
                PreparedNode::Pow { base, exp } => {
                    let base_eval = results[*base];
                    let exp_eval = results[*exp];
                    base_eval.powf(exp_eval)
                }
                PreparedNode::Builtin { symbol, arg } => {
                    evaluate_builtin_function(*symbol, results[*arg])?
                }
                PreparedNode::Custom { symbol, args } => {
                    arg_buffer.clear();
                    arg_buffer.extend(args.iter().map(|i| results[*i]));

                    if let Some(fun) = function_map.get_symbol(symbol) {
                        if !symbol_map_ready {
                            self.vars
                                .iter()
                                .copied()
                                .zip(values.iter().copied())
                                .for_each(|(symbol, value)| {
                                    symbol_map.insert(symbol, value);
                                });
                            symbol_map_ready = true;
                        }
                        fun(arg_buffer, symbol_map, function_map, cache)?
                    } else if let Some(fun) = function_map.get(symbol) {
                        if !atom_map_ready {
                            self.var_atoms
                                .iter()
                                .cloned()
                                .zip(values.iter().copied())
                                .for_each(|(atom, value)| {
                                    atom_map.insert(atom, value);
                                });
                            atom_map_ready = true;
                        }
                        fun(arg_buffer, atom_map, function_map, cache)?
                    } else {
                        return Err(format!("Missing function {}", symbol));
                    }
                }
            };
        }

        Ok(results[self.root])
    }
}

struct PreparedCompiler<'a> {
    nodes: Vec<PreparedNode>,
    cache: HashMap<AtomView<'a>, usize>,
    var_index: &'a HashMap<Symbol, usize>,
}

impl<'a> PreparedCompiler<'a> {
    fn compile_view(&mut self, view: AtomView<'a>) -> Result<usize, String> {
        if let Some(index) = self.cache.get(&view) {
            return Ok(*index);
        }

        let node = match view {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(num, den) => PreparedNode::Const(num as f64 / den as f64),
                CoefficientView::Large(_) => {
                    return Err(
                        "Large coefficients are not supported in the prepared evaluator".into(),
                    );
                }
            },
            AtomView::Var(v) => {
                let symbol = v.get_symbol();
                if let Some(index) = self.var_index.get(&symbol) {
                    PreparedNode::Var(*index)
                } else {
                    match symbol.get_stripped_name() {
                        "e" => PreparedNode::Const(std::f64::consts::E),
                        "pi" => PreparedNode::Const(std::f64::consts::PI),
                        "i" => {
                            return Err(
                                "The prepared evaluator does not support the imaginary unit".into(),
                            );
                        }
                        _ => {
                            return Err(format!(
                                "Variable {} not in prepared evaluator variable list",
                                symbol.get_stripped_name()
                            ));
                        }
                    }
                }
            }
            AtomView::Fun(f) => {
                let symbol = f.get_symbol();
                if is_builtin_function(symbol) {
                    if f.get_nargs() != 1 {
                        return Err(format!(
                            "Builtin function {} requires exactly one argument",
                            symbol
                        ));
                    }
                    let arg = self.compile_view(f.iter().next().unwrap())?;
                    PreparedNode::Builtin { symbol, arg }
                } else {
                    let mut args = Vec::with_capacity(f.get_nargs());
                    for arg in f.iter() {
                        args.push(self.compile_view(arg)?);
                    }
                    PreparedNode::Custom {
                        symbol,
                        args: args.into_boxed_slice(),
                    }
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let base = self.compile_view(base)?;
                if let Some(exponent) = packed_integer_exponent(exp) {
                    PreparedNode::PowI { base, exponent }
                } else {
                    PreparedNode::Pow {
                        base,
                        exp: self.compile_view(exp)?,
                    }
                }
            }
            AtomView::Mul(m) => {
                let mut args = Vec::with_capacity(m.get_nargs());
                for arg in m.iter() {
                    args.push(self.compile_view(arg)?);
                }
                PreparedNode::Mul(args.into_boxed_slice())
            }
            AtomView::Add(a) => {
                let mut args = Vec::with_capacity(a.get_nargs());
                for arg in a.iter() {
                    args.push(self.compile_view(arg)?);
                }
                PreparedNode::Add(args.into_boxed_slice())
            }
        };

        let index = self.nodes.len();
        self.nodes.push(node);
        self.cache.insert(view, index);
        Ok(index)
    }
}

/// Decode only the integer exponent range supported by `f64::powi`.
///
/// Values outside `i32` remain on the generic `powf` path. This keeps the
/// lowering conservative for packed coefficients while removing the common
/// reciprocal/small-integer exponent node from prepared callback evaluation.
fn packed_integer_exponent(view: AtomView<'_>) -> Option<i32> {
    let AtomView::Num(num) = view else {
        return None;
    };
    let CoefficientView::Natural(numerator, denominator) = num.get_coeff_view() else {
        return None;
    };
    if denominator != 1
        || numerator < i32::MIN as i64
        || numerator > i32::MAX as i64
    {
        return None;
    }
    Some(numerator as i32)
}

/// Evaluate an owned atom using variable bindings and custom functions.
pub fn evaluate(
    atom: &Atom,
    const_map: &HashMap<Atom, f64>,
    function_map: &FunctionMap,
) -> Result<f64, String> {
    atom.evaluate(const_map, function_map)
}

/// Evaluate an owned atom using symbol-keyed floating-point bindings and custom functions.
pub fn evaluate_with_symbols(
    atom: &Atom,
    const_map: &FloatSymbolMap,
    function_map: &FunctionMap,
) -> Result<f64, String> {
    atom.evaluate_with_symbols(const_map, function_map)
}

/// Compile an owned atom into a reusable prepared evaluator.
pub fn prepare_evaluator(
    atom: &Atom,
    vars: &[Symbol],
    function_map: &FunctionMap,
) -> Result<PreparedEvaluator, String> {
    PreparedEvaluator::new(atom, vars, function_map)
}

/// Evaluate an owned atom exactly over rational coefficients.
pub fn evaluate_exact(atom: &Atom, const_map: &ExactConstMap) -> Result<Coefficient, String> {
    atom.evaluate_exact(const_map)
}

/// Evaluate an owned atom exactly using a symbol-keyed constant map.
pub fn evaluate_exact_with_symbols(
    atom: &Atom,
    const_map: &ExactSymbolMap,
) -> Result<Coefficient, String> {
    atom.evaluate_exact_with_symbols(const_map)
}

impl Atom {
    /// Evaluate this owned atom to `f64`.
    pub fn evaluate(
        &self,
        const_map: &HashMap<Atom, f64>,
        function_map: &FunctionMap,
    ) -> Result<f64, String> {
        let mut cache = EvaluationCache::default();
        self.as_view()
            .evaluate_impl(const_map, function_map, &mut cache)
    }

    /// Evaluate this owned atom to `f64` using symbol-keyed bindings.
    pub fn evaluate_with_symbols(
        &self,
        const_map: &FloatSymbolMap,
        function_map: &FunctionMap,
    ) -> Result<f64, String> {
        let mut cache = EvaluationCache::default();
        self.as_view()
            .evaluate_with_symbols_impl(const_map, function_map, &mut cache)
    }

    /// Compile this atom into a reusable prepared evaluator.
    pub fn prepare_evaluator(
        &self,
        vars: &[Symbol],
        function_map: &FunctionMap,
    ) -> Result<PreparedEvaluator, String> {
        PreparedEvaluator::new(self, vars, function_map)
    }

    /// Evaluate this owned atom exactly when every intermediate stays rational.
    pub fn evaluate_exact(&self, const_map: &ExactConstMap) -> Result<Coefficient, String> {
        self.as_view().evaluate_exact_impl(const_map)
    }

    /// Evaluate this owned atom exactly using a symbol-keyed constant map.
    pub fn evaluate_exact_with_symbols(
        &self,
        const_map: &ExactSymbolMap,
    ) -> Result<Coefficient, String> {
        self.as_view().evaluate_exact_with_symbols_impl(const_map)
    }
}

impl<'a> AtomView<'a> {
    /// Evaluate this borrowed atom view to `f64`.
    pub fn evaluate(
        &self,
        const_map: &HashMap<Atom, f64>,
        function_map: &FunctionMap,
    ) -> Result<f64, String> {
        let mut cache = EvaluationCache::default();
        self.evaluate_impl(const_map, function_map, &mut cache)
    }

    /// Evaluate this borrowed atom view to `f64` using symbol-keyed bindings.
    pub fn evaluate_with_symbols(
        &self,
        const_map: &FloatSymbolMap,
        function_map: &FunctionMap,
    ) -> Result<f64, String> {
        let mut cache = EvaluationCache::default();
        self.evaluate_with_symbols_impl(const_map, function_map, &mut cache)
    }

    /// Evaluate this borrowed atom view exactly when every intermediate stays rational.
    pub fn evaluate_exact(&self, const_map: &ExactConstMap) -> Result<Coefficient, String> {
        self.evaluate_exact_impl(const_map)
    }

    /// Evaluate this borrowed atom view exactly using a symbol-keyed constant map.
    pub fn evaluate_exact_with_symbols(
        &self,
        const_map: &ExactSymbolMap,
    ) -> Result<Coefficient, String> {
        self.evaluate_exact_with_symbols_impl(const_map)
    }

    /// Recursive exact evaluator for rational-only expressions.
    fn evaluate_exact_impl(&self, const_map: &ExactConstMap) -> Result<Coefficient, String> {
        if let Some(value) = const_lookup(const_map, *self) {
            return Ok(value.clone());
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(num, den) => Ok(Coefficient::reduce(num, den)),
                CoefficientView::Large(_) => {
                    Err("Large coefficients are not supported in exact evaluation".into())
                }
            },
            AtomView::Var(v) => Err(format!(
                "Variable {} not in constant map",
                v.get_symbol().get_stripped_name()
            )),
            AtomView::Fun(f) => Err(format!(
                "Exact evaluation does not support function {}",
                f.get_symbol().get_stripped_name()
            )),
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let base_eval = base.evaluate_exact_impl(const_map)?;
                match exp {
                    AtomView::Num(n) => match n.get_coeff_view() {
                        CoefficientView::Natural(num, den) if den == 1 => {
                            pow_coefficient(base_eval, num)
                        }
                        CoefficientView::Natural(_, _) => {
                            Err("Exact evaluation only supports integer exponents".into())
                        }
                        CoefficientView::Large(_) => {
                            Err("Large exponents are not supported in exact evaluation".into())
                        }
                    },
                    _ => Err("Exact evaluation only supports numeric exponents".into()),
                }
            }
            AtomView::Mul(m) => {
                let mut result = Coefficient::one();
                for arg in m.iter() {
                    result = result * arg.evaluate_exact_impl(const_map)?;
                }
                Ok(result)
            }
            AtomView::Add(a) => {
                let mut result = Coefficient::zero();
                for arg in a.iter() {
                    result = result + arg.evaluate_exact_impl(const_map)?;
                }
                Ok(result)
            }
        }
    }

    /// Recursive exact evaluator for symbol-keyed rational bindings.
    fn evaluate_exact_with_symbols_impl(
        &self,
        const_map: &ExactSymbolMap,
    ) -> Result<Coefficient, String> {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(num, den) => Ok(Coefficient::reduce(num, den)),
                CoefficientView::Large(_) => {
                    Err("Large coefficients are not supported in exact evaluation".into())
                }
            },
            AtomView::Var(v) => const_map.get(&v.get_symbol()).cloned().ok_or_else(|| {
                format!(
                    "Variable {} not in constant map",
                    v.get_symbol().get_stripped_name()
                )
            }),
            AtomView::Fun(f) => Err(format!(
                "Exact evaluation does not support function {}",
                f.get_symbol().get_stripped_name()
            )),
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let base_eval = base.evaluate_exact_with_symbols_impl(const_map)?;
                match exp {
                    AtomView::Num(n) => match n.get_coeff_view() {
                        CoefficientView::Natural(num, den) if den == 1 => {
                            pow_coefficient(base_eval, num)
                        }
                        CoefficientView::Natural(_, _) => {
                            Err("Exact evaluation only supports integer exponents".into())
                        }
                        CoefficientView::Large(_) => {
                            Err("Large exponents are not supported in exact evaluation".into())
                        }
                    },
                    _ => Err("Exact evaluation only supports numeric exponents".into()),
                }
            }
            AtomView::Mul(m) => {
                let mut result = Coefficient::one();
                for arg in m.iter() {
                    result = result * arg.evaluate_exact_with_symbols_impl(const_map)?;
                }
                Ok(result)
            }
            AtomView::Add(a) => {
                let mut result = Coefficient::zero();
                for arg in a.iter() {
                    result = result + arg.evaluate_exact_with_symbols_impl(const_map)?;
                }
                Ok(result)
            }
        }
    }

    /// Recursive interpreter for packed atoms.
    fn evaluate_impl(
        &self,
        const_map: &HashMap<Atom, f64>,
        function_map: &FunctionMap,
        cache: &mut EvaluationCache,
    ) -> Result<f64, String> {
        if let Some(value) = const_lookup(const_map, *self) {
            return Ok(*value);
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(num, den) => Ok(num as f64 / den as f64),
                CoefficientView::Large(_) => {
                    Err("Large coefficients are not supported in the simplified evaluator".into())
                }
            },
            AtomView::Var(v) => {
                let symbol = v.get_symbol();
                match symbol.get_stripped_name() {
                    "e" => Ok(std::f64::consts::E),
                    "pi" => Ok(std::f64::consts::PI),
                    "i" => {
                        Err("The simplified evaluator does not support the imaginary unit".into())
                    }
                    _ => Err(format!(
                        "Variable {} not in constant map",
                        symbol.get_stripped_name()
                    )),
                }
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Atom::EXP,
                    Atom::LOG,
                    Atom::SIN,
                    Atom::COS,
                    Atom::TAN,
                    Atom::COT,
                    Atom::ASIN,
                    Atom::ACOS,
                    Atom::ATAN,
                    Atom::ACOT,
                    Atom::SQRT,
                ]
                .contains(&name)
                {
                    let arg = f.iter().next().ok_or_else(|| {
                        format!("Builtin function {} requires exactly one argument", name)
                    })?;
                    let arg_eval = arg.evaluate_impl(const_map, function_map, cache)?;
                    return Ok(match name {
                        s if s == Atom::EXP => arg_eval.exp(),
                        s if s == Atom::LOG => arg_eval.ln(),
                        s if s == Atom::SIN => arg_eval.sin(),
                        s if s == Atom::COS => arg_eval.cos(),
                        s if s == Atom::TAN => arg_eval.tan(),
                        s if s == Atom::COT => arg_eval.tan().recip(),
                        s if s == Atom::ASIN => arg_eval.asin(),
                        s if s == Atom::ACOS => arg_eval.acos(),
                        s if s == Atom::ATAN => arg_eval.atan(),
                        s if s == Atom::ACOT => std::f64::consts::FRAC_PI_2 - arg_eval.atan(),
                        s if s == Atom::SQRT => arg_eval.sqrt(),
                        _ => unreachable!(),
                    });
                }

                if let Some(value) = cache.get_view(*self) {
                    return Ok(*value);
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f.iter() {
                    args.push(arg.evaluate_impl(const_map, function_map, cache)?);
                }

                let Some(fun) = function_map.get(&name) else {
                    return Err(format!("Missing function {}", name));
                };

                let value = fun(&args, const_map, function_map, cache)?;
                cache.insert_view(*self, value);
                Ok(value)
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let base_eval = base.evaluate_impl(const_map, function_map, cache)?;

                if let AtomView::Num(n) = exp {
                    if let CoefficientView::Natural(num, den) = n.get_coeff_view() {
                        if den == 1 {
                            if num >= 0 {
                                return Ok(base_eval.powi(num as i32));
                            }
                            return Ok(base_eval.powi(-(num as i32)).recip());
                        }
                    }
                }

                let exp_eval = exp.evaluate_impl(const_map, function_map, cache)?;
                Ok(base_eval.powf(exp_eval))
            }
            AtomView::Mul(m) => {
                let mut result = 1.0;
                for arg in m.iter() {
                    result *= arg.evaluate_impl(const_map, function_map, cache)?;
                }
                Ok(result)
            }
            AtomView::Add(a) => {
                let mut result = 0.0;
                for arg in a.iter() {
                    result += arg.evaluate_impl(const_map, function_map, cache)?;
                }
                Ok(result)
            }
        }
    }

    /// Recursive interpreter for packed atoms using symbol-keyed floating-point bindings.
    fn evaluate_with_symbols_impl(
        &self,
        const_map: &FloatSymbolMap,
        function_map: &FunctionMap,
        cache: &mut EvaluationCache,
    ) -> Result<f64, String> {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(num, den) => Ok(num as f64 / den as f64),
                CoefficientView::Large(_) => {
                    Err("Large coefficients are not supported in the simplified evaluator".into())
                }
            },
            AtomView::Var(v) => {
                let symbol = v.get_symbol();
                if let Some(value) = const_map.get(&symbol) {
                    Ok(*value)
                } else {
                    match symbol.get_stripped_name() {
                        "e" => Ok(std::f64::consts::E),
                        "pi" => Ok(std::f64::consts::PI),
                        "i" => Err(
                            "The simplified evaluator does not support the imaginary unit".into(),
                        ),
                        _ => Err(format!(
                            "Variable {} not in constant map",
                            symbol.get_stripped_name()
                        )),
                    }
                }
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Atom::EXP,
                    Atom::LOG,
                    Atom::SIN,
                    Atom::COS,
                    Atom::TAN,
                    Atom::COT,
                    Atom::ASIN,
                    Atom::ACOS,
                    Atom::ATAN,
                    Atom::ACOT,
                    Atom::SQRT,
                ]
                .contains(&name)
                {
                    let arg = f.iter().next().ok_or_else(|| {
                        format!("Builtin function {} requires exactly one argument", name)
                    })?;
                    let arg_eval =
                        arg.evaluate_with_symbols_impl(const_map, function_map, cache)?;
                    return Ok(match name {
                        s if s == Atom::EXP => arg_eval.exp(),
                        s if s == Atom::LOG => arg_eval.ln(),
                        s if s == Atom::SIN => arg_eval.sin(),
                        s if s == Atom::COS => arg_eval.cos(),
                        s if s == Atom::TAN => arg_eval.tan(),
                        s if s == Atom::COT => arg_eval.tan().recip(),
                        s if s == Atom::ASIN => arg_eval.asin(),
                        s if s == Atom::ACOS => arg_eval.acos(),
                        s if s == Atom::ATAN => arg_eval.atan(),
                        s if s == Atom::ACOT => std::f64::consts::FRAC_PI_2 - arg_eval.atan(),
                        s if s == Atom::SQRT => arg_eval.sqrt(),
                        _ => unreachable!(),
                    });
                }

                if let Some(value) = cache.get_view(*self) {
                    return Ok(*value);
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f.iter() {
                    args.push(arg.evaluate_with_symbols_impl(const_map, function_map, cache)?);
                }

                let value = if let Some(fun) = function_map.get_symbol(&name) {
                    fun(&args, const_map, function_map, cache)?
                } else if let Some(fun) = function_map.get(&name) {
                    fun(&args, &EMPTY_ATOM_CONST_MAP, function_map, cache)?
                } else {
                    return Err(format!("Missing function {}", name));
                };
                cache.insert_view(*self, value);
                Ok(value)
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let base_eval = base.evaluate_with_symbols_impl(const_map, function_map, cache)?;

                if let AtomView::Num(n) = exp {
                    if let CoefficientView::Natural(num, den) = n.get_coeff_view() {
                        if den == 1 {
                            if num >= 0 {
                                return Ok(base_eval.powi(num as i32));
                            }
                            return Ok(base_eval.powi(-(num as i32)).recip());
                        }
                    }
                }

                let exp_eval = exp.evaluate_with_symbols_impl(const_map, function_map, cache)?;
                Ok(base_eval.powf(exp_eval))
            }
            AtomView::Mul(m) => {
                let mut result = 1.0;
                for arg in m.iter() {
                    result *= arg.evaluate_with_symbols_impl(const_map, function_map, cache)?;
                }
                Ok(result)
            }
            AtomView::Add(a) => {
                let mut result = 0.0;
                for arg in a.iter() {
                    result += arg.evaluate_with_symbols_impl(const_map, function_map, cache)?;
                }
                Ok(result)
            }
        }
    }
}

#[inline]
fn const_lookup<'a, V, K>(map: &'a HashMap<K, V>, key: AtomView<'_>) -> Option<&'a V>
where
    K: KeyLookup,
{
    map.get(key.get_data() as &BorrowedRawAtom)
}

#[inline]
fn is_builtin_function(symbol: Symbol) -> bool {
    [
        Atom::EXP,
        Atom::LOG,
        Atom::SIN,
        Atom::COS,
        Atom::TAN,
        Atom::COT,
        Atom::ASIN,
        Atom::ACOS,
        Atom::ATAN,
        Atom::ACOT,
        Atom::SQRT,
    ]
    .contains(&symbol)
}

#[inline]
fn evaluate_builtin_function(symbol: Symbol, arg_eval: f64) -> Result<f64, String> {
    Ok(match symbol {
        s if s == Atom::EXP => arg_eval.exp(),
        s if s == Atom::LOG => arg_eval.ln(),
        s if s == Atom::SIN => arg_eval.sin(),
        s if s == Atom::COS => arg_eval.cos(),
        s if s == Atom::TAN => arg_eval.tan(),
        s if s == Atom::COT => arg_eval.tan().recip(),
        s if s == Atom::ASIN => arg_eval.asin(),
        s if s == Atom::ACOS => arg_eval.acos(),
        s if s == Atom::ATAN => arg_eval.atan(),
        s if s == Atom::ACOT => std::f64::consts::FRAC_PI_2 - arg_eval.atan(),
        s if s == Atom::SQRT => arg_eval.sqrt(),
        _ => return Err(format!("Unsupported builtin function {}", symbol)),
    })
}

/// Raise a rational coefficient to an integer power.
fn pow_coefficient(base: Coefficient, exp: i64) -> Result<Coefficient, String> {
    if exp == 0 {
        return Ok(Coefficient::one());
    }

    let abs = exp.unsigned_abs();
    let mut num = 1_i128;
    let mut den = 1_i128;
    for _ in 0..abs {
        num = num.checked_mul(base.num as i128).ok_or_else(|| {
            "Exact evaluation overflowed i64-backed rational arithmetic".to_string()
        })?;
        den = den.checked_mul(base.den as i128).ok_or_else(|| {
            "Exact evaluation overflowed i64-backed rational arithmetic".to_string()
        })?;
    }

    let (num, den) = if exp < 0 {
        if num == 0 {
            return Err("Division by zero during exact evaluation".into());
        }
        (den, num)
    } else {
        (num, den)
    };

    let num_i64 = i64::try_from(num)
        .map_err(|_| "Exact evaluation overflowed i64-backed rational arithmetic".to_string())?;
    let den_i64 = i64::try_from(den)
        .map_err(|_| "Exact evaluation overflowed i64-backed rational arithmetic".to_string())?;
    Ok(Coefficient::reduce(num_i64, den_i64))
}

#[cfg(test)]
mod test {
    use ahash::HashMap;

    use super::{
        ExactSymbolMap, FunctionMap, evaluate_exact, evaluate_exact_with_symbols,
        prepare_evaluator,
    };
    use crate::symbolic::View::{atom::Atom, coefficient::Coefficient};
    use crate::{function, parse, symbol};

    #[test]
    fn evaluate_like_original_example() {
        let x = symbol!("x");
        let f = symbol!("f");
        let g = symbol!("g");
        let p0 = parse!("p(0)").unwrap();
        let expr = parse!("x*cos(x) + f(x, 1)^2 + g(g(x)) + p(0)").unwrap();

        let mut const_map = HashMap::default();
        let mut fn_map = FunctionMap::new();

        const_map.insert(Atom::new_var(x), 6.0);
        const_map.insert(p0, 7.0);

        fn_map.insert_fn(f, |args: &[f64], _, _, _| Ok(args[0] * args[0] + args[1]));

        fn_map.insert_fn(g, move |args: &[f64], const_map, fn_map, cache| {
            let f_eval = fn_map
                .get(&f)
                .ok_or_else(|| "Missing function f".to_string())?;
            f_eval(&[args[0], 3.0], const_map, fn_map, cache)
        });

        let result = expr.evaluate(&const_map, &fn_map).unwrap();
        let expected = 6.0_f64 * 6.0_f64.cos()
            + (37.0_f64).powi(2)
            + (39.0_f64 * 39.0_f64 + 3.0_f64)
            + 7.0_f64;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn evaluate_builtin_constants_and_functions() {
        let expr = parse!("sin(pi/2)+log(exp(2))+sqrt(9)+tan(0)+asin(0)+acos(1)+atan(0)").unwrap();
        let result = expr
            .evaluate(&HashMap::default(), &FunctionMap::new())
            .unwrap();
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn prepared_evaluator_preserves_integer_and_fractional_power_values() {
        let expr = parse!("x^(-1)+x^3+sqrt(x)").unwrap();
        let x = symbol!("x");
        let evaluator = prepare_evaluator(&expr, &[x], &FunctionMap::new()).unwrap();
        let value = evaluator.evaluate(&[4.0]).unwrap();
        let expected = 4.0_f64.powi(-1) + 4.0_f64.powi(3) + 4.0_f64.sqrt();
        assert!((value - expected).abs() < 1e-12);
    }

    #[test]
    fn evaluate_exact_rational_expression() {
        let x = symbol!("x");
        let expr = parse!("(x+1/2)^2").unwrap();
        let mut const_map = HashMap::default();
        const_map.insert(Atom::new_var(x), Coefficient::from((1, 2)));
        let value = evaluate_exact(&expr, &const_map).unwrap();
        assert_eq!(value, Coefficient::one());
    }

    #[test]
    fn evaluate_exact_with_symbol_map() {
        let x = symbol!("x");
        let expr = parse!("x^2+1/2").unwrap();
        let mut const_map = ExactSymbolMap::default();
        const_map.insert(x, Coefficient::from((3, 2)));
        let value = evaluate_exact_with_symbols(&expr, &const_map).unwrap();
        assert_eq!(value, Coefficient::from((11, 4)));
    }

    #[test]
    fn evaluate_exact_rejects_transcendentals() {
        let expr = parse!("sin(1)").unwrap();
        let err = expr.evaluate_exact(&HashMap::default()).unwrap_err();
        assert!(err.contains("does not support function sin"));
    }

    #[test]
    fn evaluate_trig_aliases_and_inverse_functions() {
        let expr = parse!("tg(pi/4)+ctg(pi/4)+arcsin(1)+arccos(0)+arctg(1)+arcctg(1)").unwrap();
        let result = expr
            .evaluate(&HashMap::default(), &FunctionMap::new())
            .unwrap();
        assert!((result - (2.0 + 3.0 * std::f64::consts::PI / 2.0)).abs() < 1e-10);
    }

    #[test]
    fn evaluate_missing_variable_errors() {
        let x = symbol!("x");
        let expr = Atom::new_var(x);
        let err = expr
            .evaluate(&HashMap::default(), &FunctionMap::new())
            .unwrap_err();
        assert!(err.contains("Variable x not in constant map"));
    }
}
