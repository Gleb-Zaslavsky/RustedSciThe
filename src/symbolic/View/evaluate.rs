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
        representation::{BorrowedRawAtom, KeyLookup},
        Atom, AtomView,
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
    fast_path: Option<PreparedFastPath>,
    // BVP residual/Jacobian expressions normally contain only numeric nodes and
    // builtins.  Keep that fact so their hot callback path can skip the custom
    // function maps and cache bookkeeping entirely.
    plain_numeric: bool,
}

/// Direct callback forms that do not need the reusable IR workspace.
///
/// These are deliberately conservative. Constant Jacobian entries are common
/// in large linearized chains, and paying for the general interpreter for a
/// single constant node can dominate the callback itself. More involved
/// expressions continue through the existing workspace-backed evaluator.
#[derive(Clone, Copy)]
pub(crate) enum PreparedFastPath {
    Constant(f64),
    Variable(usize),
}

#[derive(Clone, Copy)]
enum PreparedInput<'a> {
    Flat(&'a [f64]),
    Ivp {
        time: f64,
        parameters: &'a [f64],
        state: &'a [f64],
    },
}

impl PreparedInput<'_> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Flat(values) => values.len(),
            Self::Ivp {
                parameters, state, ..
            } => 1 + parameters.len() + state.len(),
        }
    }

    #[inline]
    fn get(&self, index: usize) -> f64 {
        match self {
            Self::Flat(values) => values[index],
            Self::Ivp {
                time,
                parameters,
                state,
            } => match index {
                0 => *time,
                index if index <= parameters.len() => parameters[index - 1],
                index => state[index - parameters.len() - 1],
            },
        }
    }
}

/// Compile-time shape of the prepared Atom evaluator.
///
/// These counters are deliberately collected once, while the evaluator is
/// being prepared. They are used to explain callback-cost differences between
/// Expr-derived closures and Atom-derived closures; they never run on the
/// numeric evaluation hot path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PreparedEvaluatorMetrics {
    pub(crate) nodes: usize,
    pub(crate) add_nodes: usize,
    pub(crate) mul_nodes: usize,
    pub(crate) powi_nodes: usize,
    pub(crate) pow_nodes: usize,
    pub(crate) builtin_nodes: usize,
    pub(crate) custom_nodes: usize,
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

/// Detect constant and identity plans once during preparation.
///
/// Constant folding here is intentionally local to the prepared evaluator. It
/// does not rewrite the symbolic Atom and therefore cannot change symbolic
/// parity or the general lowering rules used by other backends.
fn prepared_fast_path(nodes: &[PreparedNode], root: usize) -> Option<PreparedFastPath> {
    if nodes.len() == 1 {
        if let PreparedNode::Var(index) = nodes[0] {
            return Some(PreparedFastPath::Variable(index));
        }
    }

    Some(PreparedFastPath::Constant(constant_value(nodes, root)?))
}

fn constant_value(nodes: &[PreparedNode], index: usize) -> Option<f64> {
    match &nodes[index] {
        PreparedNode::Const(value) => Some(*value),
        PreparedNode::Var(_) | PreparedNode::Custom { .. } => None,
        PreparedNode::Add(args) => {
            let mut value = 0.0;
            for arg in args {
                value += constant_value(nodes, *arg)?;
            }
            Some(value)
        }
        PreparedNode::Mul(args) => {
            let mut value = 1.0;
            for arg in args {
                value *= constant_value(nodes, *arg)?;
            }
            Some(value)
        }
        PreparedNode::PowI { base, exponent } => Some(evaluate_integer_power(
            constant_value(nodes, *base)?,
            *exponent,
        )),
        PreparedNode::Pow { base, exp } => {
            Some(constant_value(nodes, *base)?.powf(constant_value(nodes, *exp)?))
        }
        PreparedNode::Builtin { symbol, arg } => {
            evaluate_builtin_function(*symbol, constant_value(nodes, *arg)?).ok()
        }
    }
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
        let fast_path = prepared_fast_path(&compiler.nodes, root);

        Ok(Self {
            nodes: compiler.nodes,
            root,
            vars: Arc::clone(&context.vars),
            var_atoms: Arc::clone(&context.var_atoms),
            function_map: function_map.clone(),
            fast_path,
            plain_numeric,
        })
    }

    /// Return the ordered variables expected by [`Self::evaluate`].
    pub fn variables(&self) -> &[Symbol] {
        &self.vars
    }

    /// Returns the immutable operation shape of this prepared evaluator.
    pub(crate) fn metrics(&self) -> PreparedEvaluatorMetrics {
        let mut metrics = PreparedEvaluatorMetrics {
            nodes: self.nodes.len(),
            ..PreparedEvaluatorMetrics::default()
        };
        for node in &self.nodes {
            match node {
                PreparedNode::Add(_) => metrics.add_nodes += 1,
                PreparedNode::Mul(_) => metrics.mul_nodes += 1,
                PreparedNode::PowI { .. } => metrics.powi_nodes += 1,
                PreparedNode::Pow { .. } => metrics.pow_nodes += 1,
                PreparedNode::Builtin { .. } => metrics.builtin_nodes += 1,
                PreparedNode::Custom { .. } => metrics.custom_nodes += 1,
                PreparedNode::Const(_) | PreparedNode::Var(_) => {}
            }
        }
        metrics
    }

    pub(crate) fn fast_path(&self) -> Option<PreparedFastPath> {
        self.fast_path
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
        if let Some(value) = self.evaluate_fast_flat(values)? {
            return Ok(value);
        }
        let mut workspace = EvaluationWorkspace::default();
        self.evaluate_with_workspace(PreparedInput::Flat(values), function_map, &mut workspace)
    }

    /// Evaluate using scratch storage local to the current worker thread.
    ///
    /// This is intentionally crate-visible: public callers retain the simple
    /// allocating `evaluate` API, while prepared callback builders can opt into
    /// the reuse contract without exposing mutable evaluator state or a Mutex.
    pub(crate) fn evaluate_thread_local(&self, values: &[f64]) -> Result<f64, String> {
        if let Some(value) = self.evaluate_fast_flat(values)? {
            return Ok(value);
        }
        EVALUATION_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            if self.plain_numeric {
                self.evaluate_plain_numeric(PreparedInput::Flat(values), &mut workspace)
            } else {
                self.evaluate_with_workspace(
                    PreparedInput::Flat(values),
                    &self.function_map,
                    &mut workspace,
                )
            }
        })
    }

    /// Evaluate an IVP callback without flattening time, parameters and state
    /// into a temporary argument vector. The prepared node indices retain the
    /// public flat ABI, while this adapter resolves those indices directly
    /// against the three borrowed input segments.
    pub(crate) fn evaluate_thread_local_ivp(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
    ) -> Result<f64, String> {
        if let Some(value) = self.evaluate_fast_ivp(time, parameters, state)? {
            return Ok(value);
        }
        EVALUATION_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let input = PreparedInput::Ivp {
                time,
                parameters,
                state,
            };
            if self.plain_numeric {
                self.evaluate_plain_numeric_ivp(time, parameters, state, &mut workspace)
            } else {
                self.evaluate_with_workspace(input, &self.function_map, &mut workspace)
            }
        })
    }

    #[inline]
    fn evaluate_fast_flat(&self, values: &[f64]) -> Result<Option<f64>, String> {
        if values.len() != self.vars.len() {
            return Err(format!(
                "Prepared evaluator expected {} argument(s), got {}",
                self.vars.len(),
                values.len()
            ));
        }
        Ok(match self.fast_path {
            Some(PreparedFastPath::Constant(value)) => Some(value),
            Some(PreparedFastPath::Variable(index)) => Some(values[index]),
            None => None,
        })
    }

    #[inline]
    fn evaluate_fast_ivp(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
    ) -> Result<Option<f64>, String> {
        let input_len = 1 + parameters.len() + state.len();
        if input_len != self.vars.len() {
            return Err(format!(
                "Prepared evaluator expected {} argument(s), got {}",
                self.vars.len(),
                input_len
            ));
        }
        Ok(self.evaluate_fast_ivp_unchecked(time, parameters, state))
    }

    #[inline]
    fn evaluate_fast_ivp_unchecked(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
    ) -> Option<f64> {
        match self.fast_path {
            Some(PreparedFastPath::Constant(value)) => Some(value),
            Some(PreparedFastPath::Variable(index)) => Some(match index {
                0 => time,
                index if index <= parameters.len() => parameters[index - 1],
                index => state[index - parameters.len() - 1],
            }),
            None => None,
        }
    }

    /// Evaluate a batch of IVP scalar plans while borrowing the worker-local
    /// workspace only once.
    ///
    /// A Jacobian contains many independent scalar evaluators. Calling
    /// `evaluate_thread_local_ivp` for every entry repeatedly entered the
    /// `thread_local!`/`RefCell` boundary. The batch API keeps the same prepared
    /// evaluator semantics but makes that boundary one-per-Jacobian callback.
    pub(crate) fn evaluate_many_thread_local_ivp<'a, I>(
        evaluators: I,
        time: f64,
        parameters: &'a [f64],
        state: &'a [f64],
        expected_input_len: usize,
        values: &mut [f64],
    ) -> Result<(), (usize, String)>
    where
        I: IntoIterator<Item = &'a PreparedEvaluator>,
    {
        EVALUATION_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let input = PreparedInput::Ivp {
                time,
                parameters,
                state,
            };
            let input_len = input.len();
            if input_len != expected_input_len {
                return Err((
                    0,
                    format!(
                        "Prepared evaluator batch expected {} argument(s), got {}",
                        expected_input_len, input_len
                    ),
                ));
            }
            for (index, evaluator) in evaluators.into_iter().enumerate() {
                // Keep the batch path semantically identical to the single
                // evaluator path. Constant and identity Jacobian entries are
                // common in sparse diffusion/reaction systems; sending them
                // through the general node interpreter defeats the prepared
                // fast path and can dominate the whole callback.
                let result = match evaluator.evaluate_fast_ivp_unchecked(time, parameters, state) {
                    Some(value) => Ok(value),
                    None if evaluator.plain_numeric => evaluator
                        .evaluate_plain_numeric_ivp_unchecked(
                            time,
                            parameters,
                            state,
                            &mut workspace,
                        ),
                    None => evaluator.evaluate_with_workspace(
                        input,
                        &evaluator.function_map,
                        &mut workspace,
                    ),
                };
                match result {
                    Ok(value) => values[index] = value,
                    Err(error) => return Err((index, error)),
                }
            }
            Ok(())
        })
    }

    /// Evaluate a batch of flat-ABI plans while borrowing the worker-local
    /// workspace only once.
    ///
    /// BVP callbacks already own a reusable `[x, state..., parameters...]`
    /// argument buffer.  The old AtomView adapter nevertheless called
    /// `evaluate_thread_local` once per scalar entry, repeatedly crossing the
    /// thread-local boundary.  Keep the flat ABI, but batch the entries so
    /// residual and sparse/banded Jacobian callbacks get the same reuse
    /// contract as the IVP-native path above.
    pub(crate) fn evaluate_many_thread_local_flat<'a, I>(
        evaluators: I,
        values: &[f64],
        output: &mut [f64],
    ) -> Result<(), (usize, String)>
    where
        I: IntoIterator<Item = &'a PreparedEvaluator>,
    {
        EVALUATION_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let mut count = 0usize;
            for evaluator in evaluators {
                if count >= output.len() {
                    return Err((
                        count,
                        format!(
                            "Prepared evaluator batch produced more than {} value(s)",
                            output.len()
                        ),
                    ));
                }

                // Keep constants and identity variables out of the general
                // interpreter.  This is especially important for sparse
                // Jacobians where such entries are common.
                let result = match evaluator.evaluate_fast_flat(values) {
                    Ok(Some(value)) => Ok(value),
                    Ok(None) if evaluator.plain_numeric => evaluator
                        .evaluate_plain_numeric(PreparedInput::Flat(values), &mut workspace),
                    Ok(None) => evaluator.evaluate_with_workspace(
                        PreparedInput::Flat(values),
                        &evaluator.function_map,
                        &mut workspace,
                    ),
                    Err(error) => Err(error),
                };
                match result {
                    Ok(value) => output[count] = value,
                    Err(error) => return Err((count, error)),
                }
                count += 1;
            }

            if count != output.len() {
                return Err((
                    count,
                    format!(
                        "Prepared evaluator batch produced {} value(s), expected {}",
                        count,
                        output.len()
                    ),
                ));
            }
            Ok(())
        })
    }

    /// Evaluate flat-ABI plans once and scatter their values into caller
    /// storage.  This is used when a structurally sparse Jacobian is exposed
    /// through a dense public layout: it avoids both a per-entry workspace
    /// transition and a temporary dense-value buffer.
    pub(crate) fn evaluate_many_thread_local_flat_scatter<'a, I>(
        evaluators: I,
        values: &[f64],
        output: &mut [f64],
    ) -> Result<(), (usize, String)>
    where
        I: IntoIterator<Item = (usize, &'a PreparedEvaluator)>,
    {
        EVALUATION_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            for (entry_index, (output_index, evaluator)) in evaluators.into_iter().enumerate() {
                if output_index >= output.len() {
                    return Err((
                        entry_index,
                        format!(
                            "Prepared evaluator scatter index {output_index} exceeds output length {}",
                            output.len()
                        ),
                    ));
                }

                let result = match evaluator.evaluate_fast_flat(values) {
                    Ok(Some(value)) => Ok(value),
                    Ok(None) if evaluator.plain_numeric => evaluator
                        .evaluate_plain_numeric(PreparedInput::Flat(values), &mut workspace),
                    Ok(None) => evaluator.evaluate_with_workspace(
                        PreparedInput::Flat(values),
                        &evaluator.function_map,
                        &mut workspace,
                    ),
                    Err(error) => Err(error),
                };
                match result {
                    Ok(value) => output[output_index] = value,
                    Err(error) => return Err((entry_index, error)),
                }
            }
            Ok(())
        })
    }

    /// Evaluate the common AtomView numeric subset without initializing the
    /// general custom-function evaluation state.  This is the path used by
    /// BVP scalar callbacks; custom symbolic functions continue to use the
    /// fully general evaluator below.
    fn evaluate_plain_numeric(
        &self,
        input: PreparedInput<'_>,
        workspace: &mut EvaluationWorkspace,
    ) -> Result<f64, String> {
        if input.len() != self.vars.len() {
            return Err(format!(
                "Prepared evaluator expected {} argument(s), got {}",
                self.vars.len(),
                input.len()
            ));
        }

        workspace.results.resize(self.nodes.len(), 0.0);
        let results = &mut workspace.results;
        for (index, node) in self.nodes.iter().enumerate() {
            results[index] = match node {
                PreparedNode::Const(value) => *value,
                PreparedNode::Var(var_index) => input.get(*var_index),
                PreparedNode::Add(args) => {
                    let mut value = 0.0;
                    for &arg in args.iter() {
                        value += results[arg];
                    }
                    value
                }
                PreparedNode::Mul(args) => {
                    let mut value = 1.0;
                    for &arg in args.iter() {
                        value *= results[arg];
                    }
                    value
                }
                PreparedNode::PowI { base, exponent } => {
                    evaluate_integer_power(results[*base], *exponent)
                }
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

    /// Evaluate the plain numeric subset against the borrowed IVP segments.
    ///
    /// The prepared node ABI still uses one flat index space, but the hot IVP
    /// path does not need to materialize that flat input. Keeping the segment
    /// dispatch here avoids matching on `PreparedInput` for every variable
    /// node while preserving the same time/parameter/state ordering.
    #[inline]
    fn evaluate_plain_numeric_ivp(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        workspace: &mut EvaluationWorkspace,
    ) -> Result<f64, String> {
        let input_len = 1 + parameters.len() + state.len();
        if input_len != self.vars.len() {
            return Err(format!(
                "Prepared evaluator expected {} argument(s), got {}",
                self.vars.len(),
                input_len
            ));
        }

        self.evaluate_plain_numeric_ivp_unchecked(time, parameters, state, workspace)
    }

    #[inline]
    fn evaluate_plain_numeric_ivp_unchecked(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        workspace: &mut EvaluationWorkspace,
    ) -> Result<f64, String> {
        workspace.results.resize(self.nodes.len(), 0.0);
        let results = &mut workspace.results;
        for (index, node) in self.nodes.iter().enumerate() {
            results[index] = match node {
                PreparedNode::Const(value) => *value,
                PreparedNode::Var(var_index) => match *var_index {
                    0 => time,
                    index if index <= parameters.len() => parameters[index - 1],
                    index => state[index - parameters.len() - 1],
                },
                PreparedNode::Add(args) => {
                    let mut value = 0.0;
                    for &arg in args.iter() {
                        value += results[arg];
                    }
                    value
                }
                PreparedNode::Mul(args) => {
                    let mut value = 1.0;
                    for &arg in args.iter() {
                        value *= results[arg];
                    }
                    value
                }
                PreparedNode::PowI { base, exponent } => {
                    evaluate_integer_power(results[*base], *exponent)
                }
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
        input: PreparedInput<'_>,
        function_map: &FunctionMap,
        workspace: &mut EvaluationWorkspace,
    ) -> Result<f64, String> {
        if input.len() != self.vars.len() {
            return Err(format!(
                "Prepared evaluator expected {} argument(s), got {}",
                self.vars.len(),
                input.len()
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
            results[index] =
                match node {
                    PreparedNode::Const(value) => *value,
                    PreparedNode::Var(var_index) => input.get(*var_index),
                    PreparedNode::Add(args) => {
                        let mut value = 0.0;
                        for &arg in args.iter() {
                            value += results[arg];
                        }
                        value
                    }
                    PreparedNode::Mul(args) => {
                        let mut value = 1.0;
                        for &arg in args.iter() {
                            value *= results[arg];
                        }
                        value
                    }
                    PreparedNode::PowI { base, exponent } => {
                        evaluate_integer_power(results[*base], *exponent)
                    }
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
                                self.vars.iter().copied().enumerate().for_each(
                                    |(index, symbol)| {
                                        symbol_map.insert(symbol, input.get(index));
                                    },
                                );
                                symbol_map_ready = true;
                            }
                            fun(arg_buffer, symbol_map, function_map, cache)?
                        } else if let Some(fun) = function_map.get(symbol) {
                            if !atom_map_ready {
                                self.var_atoms.iter().cloned().enumerate().for_each(
                                    |(index, atom)| {
                                        atom_map.insert(atom, input.get(index));
                                    },
                                );
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
    if denominator != 1 || numerator < i32::MIN as i64 || numerator > i32::MAX as i64 {
        return None;
    }
    Some(numerator as i32)
}

#[inline]
fn evaluate_integer_power(base: f64, exponent: i32) -> f64 {
    match exponent {
        0 => 1.0,
        1 => base,
        -1 => base.recip(),
        exponent => base.powi(exponent),
    }
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
        evaluate_exact, evaluate_exact_with_symbols, prepare_evaluator, ExactSymbolMap, FunctionMap,
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
    fn prepared_evaluator_uses_direct_constant_and_identity_paths() {
        let x = symbol!("x");
        let constant =
            prepare_evaluator(&parse!("2 + 3 * 4").unwrap(), &[x], &FunctionMap::new()).unwrap();
        assert!(matches!(
            constant.fast_path,
            Some(super::PreparedFastPath::Constant(value)) if value == 14.0
        ));
        assert_eq!(constant.evaluate_thread_local(&[99.0]).unwrap(), 14.0);

        let identity = prepare_evaluator(&parse!("x").unwrap(), &[x], &FunctionMap::new()).unwrap();
        assert!(matches!(
            identity.fast_path,
            Some(super::PreparedFastPath::Variable(0))
        ));
        assert_eq!(identity.evaluate_thread_local(&[2.5]).unwrap(), 2.5);
    }

    #[test]
    fn prepared_evaluator_ivp_path_matches_flat_abi_and_rejects_bad_shape() {
        let expr = parse!("t + a*y + exp(z)").unwrap();
        let t = symbol!("t");
        let a = symbol!("a");
        let y = symbol!("y");
        let z = symbol!("z");
        let evaluator = prepare_evaluator(&expr, &[t, a, y, z], &FunctionMap::new()).unwrap();

        let time = 0.5;
        let parameters = [2.0];
        let state = [3.0, 4.0];
        let ivp_value = evaluator
            .evaluate_thread_local_ivp(time, &parameters, &state)
            .unwrap();
        let flat_value = evaluator
            .evaluate(&[time, parameters[0], state[0], state[1]])
            .unwrap();
        assert!((ivp_value - flat_value).abs() <= 1.0e-12);

        let error = evaluator
            .evaluate_thread_local_ivp(time, &parameters, &[state[0]])
            .unwrap_err();
        assert!(error.contains("expected 4 argument(s), got 3"));
    }

    #[test]
    fn prepared_evaluator_batch_keeps_constant_and_identity_fast_paths() {
        let t = symbol!("t");
        let y = symbol!("y");
        let context = super::PreparedVariableContext::new(&[t, y]);
        let constant =
            prepare_evaluator(&parse!("6").unwrap(), &[t, y], &FunctionMap::new()).unwrap();
        let identity =
            prepare_evaluator(&parse!("y").unwrap(), &[t, y], &FunctionMap::new()).unwrap();
        let general =
            prepare_evaluator(&parse!("t + 2*y").unwrap(), &[t, y], &FunctionMap::new()).unwrap();

        let mut values = [0.0; 3];
        super::PreparedEvaluator::evaluate_many_thread_local_ivp(
            [&constant, &identity, &general],
            0.25,
            &[],
            &[3.0],
            2,
            &mut values,
        )
        .unwrap();

        assert_eq!(values[0], 6.0);
        assert_eq!(values[1], 3.0);
        assert!((values[2] - 6.25).abs() <= 1.0e-12);

        // The shared-context constructor must produce the same values as the
        // public constructor; this also guards the fast-path classification
        // used by the production native Jacobian batch.
        let context_constant = super::PreparedEvaluator::new_with_context(
            &parse!("6").unwrap(),
            &context,
            &FunctionMap::new(),
        )
        .unwrap();
        assert_eq!(
            context_constant
                .evaluate_thread_local_ivp(0.25, &[], &[3.0])
                .unwrap(),
            6.0
        );

        let error = super::PreparedEvaluator::evaluate_many_thread_local_ivp(
            [&constant, &identity, &general],
            0.25,
            &[],
            &[3.0],
            3,
            &mut values,
        )
        .unwrap_err();
        assert_eq!(error.0, 0);
        assert!(error.1.contains("batch expected 3 argument(s), got 2"));
    }

    #[test]
    fn prepared_evaluator_flat_batch_reuses_one_workspace_scope() {
        let x = symbol!("x");
        let constant = prepare_evaluator(&parse!("6").unwrap(), &[x], &FunctionMap::new()).unwrap();
        let identity = prepare_evaluator(&parse!("x").unwrap(), &[x], &FunctionMap::new()).unwrap();
        let general =
            prepare_evaluator(&parse!("x^2 + 1").unwrap(), &[x], &FunctionMap::new()).unwrap();

        let mut output = [0.0; 3];
        super::PreparedEvaluator::evaluate_many_thread_local_flat(
            [&constant, &identity, &general],
            &[3.0],
            &mut output,
        )
        .unwrap();

        assert_eq!(output[0], 6.0);
        assert_eq!(output[1], 3.0);
        assert_eq!(output[2], 10.0);

        let error = super::PreparedEvaluator::evaluate_many_thread_local_flat(
            [&constant, &identity, &general],
            &[3.0],
            &mut [0.0; 2],
        )
        .unwrap_err();
        assert_eq!(error.0, 2);
        assert!(error.1.contains("produced more than 2 value(s)"));
    }

    #[test]
    fn prepared_evaluator_flat_scatter_batch_writes_selected_slots() {
        let x = symbol!("x");
        let first =
            prepare_evaluator(&parse!("x + 1").unwrap(), &[x], &FunctionMap::new()).unwrap();
        let second = prepare_evaluator(&parse!("x^2").unwrap(), &[x], &FunctionMap::new()).unwrap();
        let mut output = [0.0; 4];

        super::PreparedEvaluator::evaluate_many_thread_local_flat_scatter(
            [(1, &first), (3, &second)],
            &[2.0],
            &mut output,
        )
        .unwrap();

        assert_eq!(output, [0.0, 3.0, 0.0, 4.0]);
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
