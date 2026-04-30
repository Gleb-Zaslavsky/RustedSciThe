//! Global symbol registry and thread-local workspace cache.
//!
//! The registry assigns compact numeric identifiers to symbol names once and keeps their
//! metadata in append-only storage, which makes symbol lookup cheap and stable. The
//! workspace is thread-local and recycles owned atoms, avoiding repeated allocation of
//! temporary buffers during parsing, normalization, differentiation, and evaluation.

use std::{
    borrow::Cow,
    cell::RefCell,
    collections::hash_map::Entry,
    ops::{Deref, DerefMut},
    sync::{
        RwLock,
        atomic::{AtomicUsize, Ordering},
    },
    thread::LocalKey,
};

use ahash::{HashMap, HashMapExt};
use append_only_vec::AppendOnlyVec;
use once_cell::sync::Lazy;
use smartstring::alias::String;

use super::{
    atom::{FunctionAttribute, NamespacedSymbol, NormalizationFunction},
    coefficient::Coefficient,
};

// Re-export Atom here to avoid a circular import in the Drop impl.
use super::atom::Atom;

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
static ID_TO_STR: AppendOnlyVec<(Symbol, SymbolData)> = AppendOnlyVec::new();
static SYMBOL_OFFSET: AtomicUsize = AtomicUsize::new(0);

thread_local!(
    static WORKSPACE: Workspace = const { Workspace::new() }
);

// ── SymbolData ────────────────────────────────────────────────────────────────

/// Stored metadata for a symbol id.
pub(crate) struct SymbolData {
    /// Fully-qualified symbol name.
    pub(crate) name: String,
    /// Namespace portion of the fully-qualified name.
    pub(crate) namespace: Cow<'static, str>,
    /// Source file that created the symbol, when available.
    pub(crate) file: Cow<'static, str>,
    /// Source line that created the symbol, when available.
    pub(crate) line: usize,
    /// Optional custom normalization hook for function symbols.
    pub(crate) function: Option<NormalizationFunction>,
}

// ── Symbol ────────────────────────────────────────────────────────────────────

/// A compact symbol handle shared by variables and function names.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Symbol {
    /// Dense id used inside packed atoms.
    pub(crate) id: u32,
    /// Trailing wildcard marker count for pattern-like symbols.
    pub(crate) wildcard_level: u8,
    /// Whether the symbol should normalize as symmetric.
    pub(crate) is_symmetric: bool,
    /// Whether the symbol should normalize as antisymmetric.
    pub(crate) is_antisymmetric: bool,
    /// Whether the symbol should normalize modulo cyclic rotations.
    pub(crate) is_cyclesymmetric: bool,
    /// Whether the symbol should distribute over addition.
    pub(crate) is_linear: bool,
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)?;
        for _ in 0..self.wildcard_level {
            f.write_str("_")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(State::get_name(*self))
    }
}

impl Symbol {
    /// Intern or retrieve a symbol from its namespaced textual form.
    pub fn new(name: NamespacedSymbol) -> Symbol {
        State::get_symbol(name)
    }

    /// Intern or retrieve a function symbol with normalization attributes.
    pub fn new_with_attributes(
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
    ) -> Result<Symbol, String> {
        State::get_symbol_with_attributes(name, attributes)
    }

    /// Intern or retrieve a function symbol with attributes and a custom normalization hook.
    pub fn new_with_attributes_and_function(
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
        f: NormalizationFunction,
    ) -> Result<Symbol, String> {
        State::get_symbol_with_attributes_and_function(name, attributes, f)
    }

    /// Return the fully-qualified symbol name.
    pub fn get_name(&self) -> &str {
        State::get_name(*self)
    }

    /// Return the symbol name without its namespace prefix.
    pub fn get_stripped_name(&self) -> &str {
        let d = State::get_symbol_data(*self);
        &d.name[d.namespace.len() + 2..]
    }

    pub fn get_id(&self) -> u32 {
        self.id
    }

    pub fn get_namespace(&self) -> &'static str {
        State::get_symbol_namespace(*self)
    }

    pub fn get_wildcard_level(&self) -> u8 {
        self.wildcard_level
    }
    pub fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }
    pub fn is_antisymmetric(&self) -> bool {
        self.is_antisymmetric
    }
    pub fn is_cyclesymmetric(&self) -> bool {
        self.is_cyclesymmetric
    }
    pub fn is_linear(&self) -> bool {
        self.is_linear
    }

    pub fn is_builtin(id: Symbol) -> bool {
        State::is_builtin(id)
    }

    pub const fn raw_var(id: u32, wildcard_level: u8) -> Self {
        Symbol {
            id,
            wildcard_level,
            is_symmetric: false,
            is_antisymmetric: false,
            is_cyclesymmetric: false,
            is_linear: false,
        }
    }

    pub const fn raw_fn(
        id: u32,
        wildcard_level: u8,
        is_symmetric: bool,
        is_antisymmetric: bool,
        is_cyclesymmetric: bool,
        is_linear: bool,
    ) -> Self {
        Symbol {
            id,
            wildcard_level,
            is_symmetric,
            is_antisymmetric,
            is_cyclesymmetric,
            is_linear,
        }
    }
}

// ── State ─────────────────────────────────────────────────────────────────────

/// Global mapping between textual names and compact [`Symbol`] handles.
pub struct State {
    str_to_id: HashMap<String, Symbol>,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    pub(crate) const ARG: Symbol = Symbol::raw_fn(0, 0, false, false, false, false);
    pub(crate) const EXP: Symbol = Symbol::raw_fn(1, 0, false, false, false, false);
    pub(crate) const LOG: Symbol = Symbol::raw_fn(2, 0, false, false, false, false);
    pub(crate) const SIN: Symbol = Symbol::raw_fn(3, 0, false, false, false, false);
    pub(crate) const COS: Symbol = Symbol::raw_fn(4, 0, false, false, false, false);
    pub(crate) const TAN: Symbol = Symbol::raw_fn(5, 0, false, false, false, false);
    pub(crate) const COT: Symbol = Symbol::raw_fn(6, 0, false, false, false, false);
    pub(crate) const ASIN: Symbol = Symbol::raw_fn(7, 0, false, false, false, false);
    pub(crate) const ACOS: Symbol = Symbol::raw_fn(8, 0, false, false, false, false);
    pub(crate) const ATAN: Symbol = Symbol::raw_fn(9, 0, false, false, false, false);
    pub(crate) const ACOT: Symbol = Symbol::raw_fn(10, 0, false, false, false, false);
    pub(crate) const SQRT: Symbol = Symbol::raw_fn(11, 0, false, false, false, false);
    pub(crate) const DERIVATIVE: Symbol = Symbol::raw_fn(12, 0, false, false, false, false);
    pub(crate) const E: Symbol = Symbol::raw_var(13, 0);
    pub(crate) const I: Symbol = Symbol::raw_var(14, 0);
    pub(crate) const PI: Symbol = Symbol::raw_var(15, 0);

    pub const BUILTIN_SYMBOL_NAMES: [&'static str; 16] = [
        "arg", "exp", "log", "sin", "cos", "tan", "cot", "asin", "acos", "atan", "acot", "sqrt",
        "der", "e", "i", "pi",
    ];

    /// Map supported aliases like `tg` and `arctg` onto canonical builtin names.
    pub fn canonical_builtin_name(s: &str) -> Option<&'static str> {
        match s {
            "arg" => Some("arg"),
            "exp" => Some("exp"),
            "log" => Some("log"),
            "sin" => Some("sin"),
            "cos" => Some("cos"),
            "tan" | "tg" => Some("tan"),
            "cot" | "ctg" => Some("cot"),
            "asin" | "arcsin" => Some("asin"),
            "acos" | "arccos" => Some("acos"),
            "atan" | "arctan" | "arctg" => Some("atan"),
            "acot" | "arcctan" | "arcctg" => Some("acot"),
            "sqrt" => Some("sqrt"),
            "der" => Some("der"),
            "e" => Some("e"),
            "i" => Some("i"),
            "pi" => Some("pi"),
            _ => None,
        }
    }

    /// Return whether `s` is the textual name of a builtin symbol or supported alias.
    pub fn is_builtin_name<S: AsRef<str>>(s: S) -> bool {
        Self::canonical_builtin_name(s.as_ref()).is_some()
    }

    fn new() -> State {
        let mut state = State {
            str_to_id: HashMap::new(),
        };
        for x in Self::BUILTIN_SYMBOL_NAMES {
            state.get_symbol_impl(crate::wrap_symbol!(x));
        }
        state
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn get_global_state() -> &'static RwLock<State> {
        &STATE
    }

    pub(crate) fn get_symbol(name: NamespacedSymbol) -> Symbol {
        STATE.write().unwrap().get_symbol_impl(name)
    }

    pub(crate) fn get_symbol_impl(&mut self, name: NamespacedSymbol) -> Symbol {
        match self.str_to_id.entry(name.symbol.into()) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);
                if ID_TO_STR.len() - offset == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let mut wildcard_level = 0u8;
                for x in v.key().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                let id = (ID_TO_STR.len() - offset) as u32;
                let new_symbol = Symbol::raw_var(id, wildcard_level);
                let id_ret = ID_TO_STR.push((
                    new_symbol,
                    SymbolData {
                        name: v.key().clone(),
                        file: name.file,
                        namespace: name.namespace,
                        line: name.line,
                        function: None,
                    },
                )) - offset;
                assert_eq!(id as usize, id_ret);
                v.insert(new_symbol);
                new_symbol
            }
        }
    }

    pub(crate) fn get_symbol_with_attributes(
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
    ) -> Result<Symbol, String> {
        STATE
            .write()
            .unwrap()
            .get_symbol_with_attributes_impl(name, attributes, None)
    }

    pub(crate) fn get_symbol_with_attributes_and_function(
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
        f: NormalizationFunction,
    ) -> Result<Symbol, String> {
        STATE
            .write()
            .unwrap()
            .get_symbol_with_attributes_impl(name, attributes, Some(f))
    }

    pub(crate) fn get_symbol_with_attributes_impl(
        &mut self,
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
        normalization_function: Option<NormalizationFunction>,
    ) -> Result<Symbol, String> {
        match self.str_to_id.entry(name.symbol.into()) {
            Entry::Occupied(o) => {
                let r = *o.get();
                let new_id = Symbol::raw_fn(
                    r.get_id(),
                    r.get_wildcard_level(),
                    attributes.contains(&FunctionAttribute::Symmetric),
                    attributes.contains(&FunctionAttribute::Antisymmetric),
                    attributes.contains(&FunctionAttribute::Cyclesymmetric),
                    attributes.contains(&FunctionAttribute::Linear),
                );
                if r == new_id && normalization_function.is_none() {
                    Ok(r)
                } else {
                    let data = &ID_TO_STR[r.get_id() as usize].1;
                    if data.file.is_empty() {
                        Err(format!("Symbol {} redefined with new attributes.", data.name).into())
                    } else {
                        Err(format!(
                            "Symbol {} redefined with new attributes. First definition: {}:{}.",
                            data.name, data.file, data.line
                        )
                        .into())
                    }
                }
            }
            Entry::Vacant(v) => {
                let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);
                if ID_TO_STR.len() - offset == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let mut wildcard_level = 0u8;
                for x in v.key().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                let id = (ID_TO_STR.len() - offset) as u32;
                let new_symbol = Symbol::raw_fn(
                    id,
                    wildcard_level,
                    attributes.contains(&FunctionAttribute::Symmetric),
                    attributes.contains(&FunctionAttribute::Antisymmetric),
                    attributes.contains(&FunctionAttribute::Cyclesymmetric),
                    attributes.contains(&FunctionAttribute::Linear),
                );

                let id_ret = ID_TO_STR.push((
                    new_symbol,
                    SymbolData {
                        name: v.key().clone(),
                        file: name.file,
                        namespace: name.namespace,
                        line: name.line,
                        function: normalization_function,
                    },
                )) - offset;
                assert_eq!(id as usize, id_ret);
                v.insert(new_symbol);
                Ok(new_symbol)
            }
        }
    }

    #[inline]
    pub(crate) fn get_name(id: Symbol) -> &'static str {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE;
        }
        &ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .name
    }

    #[inline]
    pub(crate) fn get_symbol_namespace(id: Symbol) -> &'static str {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE;
        }
        ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .namespace
            .as_ref()
    }

    #[inline]
    pub(crate) fn get_symbol_data(id: Symbol) -> &'static SymbolData {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE;
        }
        &ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)].1
    }

    #[inline]
    pub(crate) fn get_normalization_function(id: Symbol) -> Option<&'static NormalizationFunction> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE;
        }
        ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .function
            .as_ref()
    }

    pub(crate) fn is_builtin(id: Symbol) -> bool {
        id.get_id() < Self::BUILTIN_SYMBOL_NAMES.len() as u32
    }

    /// Iterate over all interned symbols and their stored names.
    pub fn symbol_iter() -> impl Iterator<Item = (Symbol, &'static str)> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE;
        }
        ID_TO_STR
            .iter()
            .skip(SYMBOL_OFFSET.load(Ordering::Relaxed))
            .map(|s| (s.0, s.1.name.as_str()))
    }
}

// ── Workspace ─────────────────────────────────────────────────────────────────

/// Thread-local cache of reusable owned atoms.
pub struct Workspace {
    atom_buffer: RefCell<Vec<Atom>>,
}

impl Workspace {
    const ATOM_BUFFER_MAX: usize = 30;
    const ATOM_CACHE_SIZE_MAX: usize = 20_000_000;

    const fn new() -> Self {
        Workspace {
            atom_buffer: RefCell::new(Vec::new()),
        }
    }

    #[inline]
    /// Return the thread-local workspace used by high-level operations.
    pub fn get_local() -> &'static LocalKey<Workspace> {
        &WORKSPACE
    }

    #[inline]
    /// Borrow a reusable atom buffer or create a fresh empty atom.
    pub fn new_atom(&self) -> RecycledAtom {
        if let Ok(mut a) = self.atom_buffer.try_borrow_mut() {
            if let Some(b) = a.pop() {
                return b.into();
            }
        }
        Atom::default().into()
    }

    #[inline]
    /// Allocate a recycled variable atom.
    pub fn new_var(&self, id: Symbol) -> RecycledAtom {
        let mut owned = self.new_atom();
        owned.to_var(id);
        owned
    }

    #[inline]
    /// Allocate a recycled numeric atom.
    pub fn new_num<T: Into<Coefficient>>(&self, num: T) -> RecycledAtom {
        let mut owned = self.new_atom();
        owned.to_num(num.into());
        owned
    }

    /// Return an owned atom buffer to the workspace cache.
    pub fn return_atom(&self, atom: Atom) {
        if let Ok(mut a) = self.atom_buffer.try_borrow_mut() {
            a.push(atom);
        }
    }
}

// ── RecycledAtom ──────────────────────────────────────────────────────────────

/// Owned atom wrapper that automatically returns its buffer to the workspace on drop.
#[derive(PartialEq, Eq, Debug, Hash, Clone)]
pub struct RecycledAtom(Atom);

impl From<Atom> for RecycledAtom {
    fn from(a: Atom) -> Self {
        RecycledAtom(a)
    }
}

impl std::fmt::Display for RecycledAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Default for RecycledAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl RecycledAtom {
    #[inline]
    /// Create an empty recycled atom from the thread-local workspace.
    pub fn new() -> RecycledAtom {
        Workspace::get_local().with(|ws| ws.new_atom())
    }

    /// Wrap an already-owned atom without changing its contents.
    pub fn wrap(atom: Atom) -> RecycledAtom {
        RecycledAtom(atom)
    }

    #[inline]
    /// Create a recycled variable atom.
    pub fn new_var(id: Symbol) -> RecycledAtom {
        let mut owned = Self::new();
        owned.to_var(id);
        owned
    }

    #[inline]
    /// Create a recycled numeric atom.
    pub fn new_num<T: Into<Coefficient>>(num: T) -> RecycledAtom {
        let mut owned = Self::new();
        owned.to_num(num.into());
        owned
    }

    /// Extract the owned atom and skip automatic buffer recycling for this value.
    pub fn into_inner(mut self) -> Atom {
        std::mem::replace(&mut self.0, Atom::Zero)
    }
}

impl Deref for RecycledAtom {
    type Target = Atom;
    fn deref(&self) -> &Atom {
        &self.0
    }
}

impl DerefMut for RecycledAtom {
    fn deref_mut(&mut self) -> &mut Atom {
        &mut self.0
    }
}

impl AsRef<Atom> for RecycledAtom {
    fn as_ref(&self) -> &Atom {
        self.deref()
    }
}

impl Drop for RecycledAtom {
    #[inline]
    fn drop(&mut self) {
        if let Atom::Zero = self.0 {
            return;
        }
        if self.0.get_capacity() > Workspace::ATOM_CACHE_SIZE_MAX {
            return;
        }

        let _ = WORKSPACE.try_with(
            #[inline(always)]
            |ws| {
                if let Ok(mut a) = ws.atom_buffer.try_borrow_mut() {
                    if a.len() < Workspace::ATOM_BUFFER_MAX {
                        a.push(std::mem::replace(&mut self.0, Atom::Zero));
                    }
                }
            },
        );
    }
}
