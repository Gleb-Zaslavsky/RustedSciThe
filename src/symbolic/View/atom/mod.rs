//! High-level atom API built on top of the packed representation.
//!
//! This module is the bridge between ergonomic symbolic manipulation and the compact
//! byte-oriented storage in [`representation`]. Parsing, arithmetic, normalization,
//! derivatives, evaluation, and printing all flow through [`Atom`] and [`AtomView`].
//!
//! # Performance model
//! - [`Atom`] owns a packed byte buffer.
//! - [`AtomView`] borrows that buffer without copying.
//! - Inline helpers such as [`InlineVar`] and [`InlineNum`] let builders create tiny
//!   temporary atoms on the stack.
//! - [`FunctionBuilder`] and arithmetic helpers allocate through [`Workspace`] so short-
//!   lived intermediate nodes reuse storage instead of hitting the allocator repeatedly.
//!
//! The result is a public API that still feels symbolic while keeping the fast storage
//! strategy from the original crate.

mod coefficient;
mod core;
pub mod representation;

use std::{borrow::Cow, cmp::Ordering, hash::Hash, ops::DerefMut};

use representation::InlineVar;

use super::{
    coefficient::Coefficient,
    state::{RecycledAtom, State, Workspace},
};

pub use self::core::AtomCore;
pub use self::representation::{
    Add, AddView, Fun, FunView, KeyLookup, ListIterator, ListSlice, Mul, MulView, Num, NumView,
    Pow, PowView, RawAtom, Var, VarView,
};
use crate::symbolic::View::state::Symbol;
/// A symbol name together with namespace and source-location metadata.
pub struct NamespacedSymbol {
    /// Namespace portion used when interning the symbol.
    pub namespace: Cow<'static, str>,
    /// Fully-qualified symbol text.
    pub symbol: Cow<'static, str>,
    /// Source file where the symbol was created, when known.
    pub file: Cow<'static, str>,
    /// Source line where the symbol was created, when known.
    pub line: usize,
}

impl NamespacedSymbol {
    pub fn parse(s: &str) -> NamespacedSymbol {
        let mut parts = s.split("::");
        let namespace = parts.next().unwrap();
        let symbol = parts.next().unwrap();
        NamespacedSymbol {
            namespace: namespace.to_string().into(),
            symbol: symbol.to_string().into(),
            file: "".into(),
            line: 0,
        }
    }

    pub fn try_parse<S: AsRef<str>>(s: S) -> Option<NamespacedSymbol> {
        let mut parts = s.as_ref().split("::");
        let namespace = parts.next()?;
        let _symbol = parts.next()?;
        Some(NamespacedSymbol {
            namespace: namespace.to_string().into(),
            symbol: s.as_ref().to_string().into(),
            file: "".into(),
            line: 0,
        })
    }

    pub fn try_parse_lit(s: &'static str) -> Option<NamespacedSymbol> {
        let mut parts = s.split("::");
        let namespace = parts.next()?;
        let _symbol = parts.next()?;
        Some(NamespacedSymbol {
            namespace: namespace.into(),
            symbol: s.into(),
            file: "".into(),
            line: 0,
        })
    }
}

impl TryFrom<&str> for NamespacedSymbol {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(NamespacedSymbol::parse(value))
    }
}

#[macro_export]
macro_rules! wrap_symbol {
    ($e:literal) => {{
        if let Some(mut s) = $crate::symbolic::View::atom::NamespacedSymbol::try_parse_lit($e) {
            s.file = file!().into();
            s.line = line!() as usize;
            s
        } else if let Some(canonical) = $crate::symbolic::View::state::State::canonical_builtin_name($e) {
            $crate::symbolic::View::atom::NamespacedSymbol {
                symbol: format!("symbolica::{}", canonical).into(),
                namespace: "symbolica".into(),
                file: file!().into(),
                line: line!() as usize,
            }
        } else {
            let ns = $crate::namespace!();
            $crate::symbolic::View::atom::NamespacedSymbol {
                symbol: format!("{}::{}", ns, $e).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            }
        }
    }};
    ($e:expr) => {{
        if let Some(mut s) = $crate::symbolic::View::atom::NamespacedSymbol::try_parse($e) {
            s.file = file!().into();
            s.line = line!() as usize;
            s
        } else {
            let ns = if $crate::symbolic::View::state::State::is_builtin_name(&$e) {
                "symbolica"
            } else {
                $crate::namespace!()
            };
            $crate::symbolic::View::atom::NamespacedSymbol {
                symbol: format!("{}::{}", ns, $e).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            }
        }
    }};
}

/// Parsing context used to attach a default namespace to bare identifiers.
pub struct DefaultNamespace<'a> {
    /// Namespace used for identifiers that are not already qualified.
    pub namespace: Cow<'static, str>,
    /// Raw expression text being parsed.
    pub data: &'a str,
    /// Source file of the parsed expression, when known.
    pub file: Cow<'static, str>,
    /// Source line of the parsed expression, when known.
    pub line: usize,
}

impl DefaultNamespace<'_> {
    pub fn attach_namespace(&self, s: &str) -> NamespacedSymbol {
        if let Some(mut qualified) = NamespacedSymbol::try_parse(s) {
            if qualified.namespace == "symbolica" {
                let stripped = qualified.symbol.split("::").nth(1).unwrap();
                if let Some(canonical) = State::canonical_builtin_name(stripped) {
                    qualified.symbol = format!("symbolica::{}", canonical).into();
                }
            }
            qualified.file = self.file.clone();
            qualified.line = self.line;
            qualified
        } else if let Some(canonical) = State::canonical_builtin_name(s) {
            NamespacedSymbol {
                symbol: format!("symbolica::{}", canonical).into(),
                namespace: "symbolica".into(),
                file: "".into(),
                line: 0,
            }
        } else {
            NamespacedSymbol {
                symbol: format!("{}::{}", self.namespace, s).into(),
                namespace: self.namespace.clone(),
                file: self.file.clone(),
                line: self.line,
            }
        }
    }
}

#[macro_export]
macro_rules! wrap_input {
    ($e:expr) => {{
        let ns = $crate::namespace!();
        $crate::symbolic::View::atom::DefaultNamespace {
            data: $e.as_ref(),
            namespace: ns.into(),
            file: file!().into(),
            line: line!() as usize,
        }
    }};
}

#[macro_export]
macro_rules! with_default_namespace {
    ($e:expr, $namespace: expr) => {{
        $crate::atom::DefaultNamespace {
            data: $e.as_ref(),
            namespace: $namespace.into(),
            file: file!().into(),
            line: line!() as usize,
        }
    }};
}

#[macro_export]
macro_rules! namespace {
    () => {{ env!("CARGO_CRATE_NAME") }};
}

/// Callback type for symbol-specific normalization hooks.
pub type NormalizationFunction = Box<dyn Fn(AtomView<'_>, &mut Atom) -> bool + Send + Sync>;

/// Normalization attributes attached to function symbols.
#[derive(Clone, Copy, PartialEq)]
pub enum FunctionAttribute {
    /// Arguments may be reordered freely.
    Symmetric,
    /// Swapping arguments flips the sign.
    Antisymmetric,
    /// Arguments are compared modulo cyclic rotation.
    Cyclesymmetric,
    /// Function distributes over addition in each argument.
    Linear,
}

/// Logical atom categories stored in packed form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomType {
    /// Numeric coefficient.
    Num,
    /// Symbolic variable.
    Var,
    /// Sum node.
    Add,
    /// Product node.
    Mul,
    /// Power node.
    Pow,
    /// Function application.
    Fun,
}

/// Logical list kinds used when iterating packed child slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceType {
    /// Additive children.
    Add,
    /// Multiplicative children.
    Mul,
    /// Function arguments.
    Arg,
    /// A synthetic one-element slice.
    One,
    /// Base/exponent pair.
    Pow,
    /// Empty slice.
    Empty,
}

/// Borrowed zero-copy view into packed atom data.
#[derive(Debug)]
pub enum AtomView<'a> {
    /// Numeric coefficient view.
    Num(NumView<'a>),
    /// Variable view.
    Var(VarView<'a>),
    /// Function application view.
    Fun(FunView<'a>),
    /// Power view.
    Pow(PowView<'a>),
    /// Product view.
    Mul(MulView<'a>),
    /// Sum view.
    Add(AddView<'a>),
}

impl Clone for AtomView<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for AtomView<'_> {}
impl Eq for AtomView<'_> {}

impl PartialOrd for AtomView<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AtomView<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.get_data().cmp(other.get_data())
    }
}

impl Hash for AtomView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AtomView::Num(a) => a.hash(state),
            AtomView::Var(a) => a.hash(state),
            AtomView::Fun(a) => a.hash(state),
            AtomView::Pow(a) => a.hash(state),
            AtomView::Mul(a) => a.hash(state),
            AtomView::Add(a) => a.hash(state),
        }
    }
}

impl std::fmt::Display for AtomView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        crate::symbolic::View::printer::print_view_to_string(
            self,
            &crate::symbolic::View::printer::PrintOptions::default(),
            &mut out,
        );
        f.write_str(&out)
    }
}

/// Utility enum that can hold either an owned atom or a borrowed view.
#[derive(Clone)]
pub enum AtomOrView<'a> {
    /// Owned packed atom.
    Atom(Atom),
    /// Borrowed packed view.
    View(AtomView<'a>),
}

impl<'a> AtomOrView<'a> {
    pub fn as_view(&'a self) -> AtomView<'a> {
        match self {
            AtomOrView::Atom(a) => a.as_view(),
            AtomOrView::View(a) => *a,
        }
    }
}

/// Owned symbolic expression backed by packed bytes.
#[derive(Clone)]
pub enum Atom {
    /// Numeric coefficient.
    Num(Num),
    /// Variable.
    Var(Var),
    /// Function application.
    Fun(Fun),
    /// Power.
    Pow(Pow),
    /// Product.
    Mul(Mul),
    /// Sum.
    Add(Add),
    /// Canonical zero shortcut.
    Zero,
}

impl Atom {
    pub const ARG: crate::symbolic::View::state::Symbol = State::ARG;
    pub const EXP: crate::symbolic::View::state::Symbol = State::EXP;
    pub const LOG: crate::symbolic::View::state::Symbol = State::LOG;
    pub const SIN: crate::symbolic::View::state::Symbol = State::SIN;
    pub const COS: crate::symbolic::View::state::Symbol = State::COS;
    pub const TAN: crate::symbolic::View::state::Symbol = State::TAN;
    pub const COT: crate::symbolic::View::state::Symbol = State::COT;
    pub const ASIN: crate::symbolic::View::state::Symbol = State::ASIN;
    pub const ACOS: crate::symbolic::View::state::Symbol = State::ACOS;
    pub const ATAN: crate::symbolic::View::state::Symbol = State::ATAN;
    pub const ACOT: crate::symbolic::View::state::Symbol = State::ACOT;
    pub const SQRT: crate::symbolic::View::state::Symbol = State::SQRT;
    pub const DERIVATIVE: crate::symbolic::View::state::Symbol = State::DERIVATIVE;
    pub const E: crate::symbolic::View::state::Symbol = State::E;
    pub const I: crate::symbolic::View::state::Symbol = State::I;
    pub const PI: crate::symbolic::View::state::Symbol = State::PI;

    pub fn new() -> Atom {
        Atom::default()
    }

    /// Parse an expression directly into an owned atom.
    pub fn parse(input: DefaultNamespace<'_>) -> Result<Atom, String> {
        crate::symbolic::View::parser::parse_with_default_namespace(input)
    }

    /// Construct a variable atom.
    pub fn new_var(id: crate::symbolic::View::state::Symbol) -> Atom {
        Var::new(id).into()
    }

    /// Construct a numeric atom, collapsing exact zero into [`Atom::Zero`].
    pub fn new_num<T: Into<Coefficient>>(num: T) -> Atom {
        let c = num.into();
        if c.is_zero() {
            Atom::Zero
        } else {
            Num::new(c).into()
        }
    }

    /// Build `exp(self)`.
    pub fn exp(&self) -> Atom {
        FunctionBuilder::new(Atom::EXP)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `log(self)`.
    pub fn log(&self) -> Atom {
        FunctionBuilder::new(Atom::LOG)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `sin(self)`.
    pub fn sin(&self) -> Atom {
        FunctionBuilder::new(Atom::SIN)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `cos(self)`.
    pub fn cos(&self) -> Atom {
        FunctionBuilder::new(Atom::COS)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `tan(self)`.
    pub fn tan(&self) -> Atom {
        FunctionBuilder::new(Atom::TAN)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `cot(self)`.
    pub fn cot(&self) -> Atom {
        FunctionBuilder::new(Atom::COT)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `asin(self)`.
    pub fn asin(&self) -> Atom {
        FunctionBuilder::new(Atom::ASIN)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `acos(self)`.
    pub fn acos(&self) -> Atom {
        FunctionBuilder::new(Atom::ACOS)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `atan(self)`.
    pub fn atan(&self) -> Atom {
        FunctionBuilder::new(Atom::ATAN)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `acot(self)`.
    pub fn acot(&self) -> Atom {
        FunctionBuilder::new(Atom::ACOT)
            .add_arg(self.as_view())
            .finish()
    }
    /// Build `sqrt(self)`.
    pub fn sqrt(&self) -> Atom {
        FunctionBuilder::new(Atom::SQRT)
            .add_arg(self.as_view())
            .finish()
    }

    pub fn is_zero(&self) -> bool {
        self.as_view().is_zero()
    }
    pub fn is_one(&self) -> bool {
        self.as_view().is_one()
    }
    pub fn nterms(&self) -> usize {
        self.as_view().nterms()
    }

    pub fn to_num(&mut self, coeff: Coefficient) -> &mut Num {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Num(Num::new_into(coeff, buffer));
        match self {
            Atom::Num(n) => n,
            _ => unreachable!(),
        }
    }

    pub fn to_var(&mut self, id: crate::symbolic::View::state::Symbol) -> &mut Var {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Var(Var::new_into(id, buffer));
        match self {
            Atom::Var(v) => v,
            _ => unreachable!(),
        }
    }

    pub fn to_fun(&mut self, id: crate::symbolic::View::state::Symbol) -> &mut Fun {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Fun(Fun::new_into(id, buffer));
        match self {
            Atom::Fun(fun) => fun,
            _ => unreachable!(),
        }
    }

    pub fn to_pow(&mut self, base: AtomView<'_>, exp: AtomView<'_>) -> &mut Pow {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Pow(Pow::new_into(base, exp, buffer));
        match self {
            Atom::Pow(pow) => pow,
            _ => unreachable!(),
        }
    }

    pub fn to_mul(&mut self) -> &mut Mul {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Mul(Mul::new_into(buffer));
        match self {
            Atom::Mul(mul) => mul,
            _ => unreachable!(),
        }
    }

    pub fn to_add(&mut self) -> &mut Add {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Add(Add::new_into(buffer));
        match self {
            Atom::Add(add) => add,
            _ => unreachable!(),
        }
    }

    pub fn into_raw(self) -> RawAtom {
        match self {
            Atom::Num(n) => n.into_raw(),
            Atom::Var(v) => v.into_raw(),
            Atom::Fun(f) => f.into_raw(),
            Atom::Pow(p) => p.into_raw(),
            Atom::Mul(m) => m.into_raw(),
            Atom::Add(a) => a.into_raw(),
            Atom::Zero => RawAtom::new(),
        }
    }

    pub fn set_from_view(&mut self, view: &AtomView<'_>) {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        match view {
            AtomView::Num(n) => *self = Atom::Num(Num::from_view_into(n, buffer)),
            AtomView::Var(v) => *self = Atom::Var(Var::from_view_into(v, buffer)),
            AtomView::Fun(fun) => *self = Atom::Fun(Fun::from_view_into(fun, buffer)),
            AtomView::Pow(pow) => *self = Atom::Pow(Pow::from_view_into(pow, buffer)),
            AtomView::Mul(mul) => *self = Atom::Mul(Mul::from_view_into(mul, buffer)),
            AtomView::Add(add) => *self = Atom::Add(Add::from_view_into(add, buffer)),
        }
    }

    pub fn as_view(&self) -> AtomView<'_> {
        match self {
            Atom::Num(n) => AtomView::Num(n.to_num_view()),
            Atom::Var(v) => AtomView::Var(v.to_var_view()),
            Atom::Fun(f) => AtomView::Fun(f.to_fun_view()),
            Atom::Pow(p) => AtomView::Pow(p.to_pow_view()),
            Atom::Mul(m) => AtomView::Mul(m.to_mul_view()),
            Atom::Add(a) => AtomView::Add(a.to_add_view()),
            Atom::Zero => AtomView::ZERO,
        }
    }

    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        match self {
            Atom::Fun(a) => a.set_normalized(normalized),
            Atom::Pow(a) => a.set_normalized(normalized),
            Atom::Mul(a) => a.set_normalized(normalized),
            Atom::Add(a) => a.set_normalized(normalized),
            _ => {}
        }
    }

    /// Raise the atom to an exact numeric power and normalize the result.
    pub fn npow<T: Into<Coefficient>>(&self, exp: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(exp);
            let mut t = ws.new_atom();
            self.as_view()
                .pow_no_norm(ws, n.as_view())
                .as_view()
                .normalize(&mut t);
            t.into_inner()
        })
    }

    /// Raise the atom to a symbolic power and normalize the result.
    pub fn pow<T: AtomCore>(&self, exp: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view()
                .pow_no_norm(ws, exp.as_atom_view())
                .as_view()
                .normalize(&mut t);
            t.into_inner()
        })
    }

    pub fn rpow<T: AtomCore>(&self, base: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            base.as_atom_view()
                .pow_no_norm(ws, self.as_view())
                .as_view()
                .normalize(&mut t);
            t.into_inner()
        })
    }

    /// Build and normalize an n-ary sum.
    pub fn add_many<T: AtomCore + Copy>(args: &[T]) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            let add = t.to_add();
            for a in args {
                add.extend(a.as_atom_view());
            }
            let mut out = Atom::new();
            t.as_view().normalize(&mut out);
            out
        })
    }

    /// Build and normalize an n-ary product.
    pub fn mul_many<T: AtomCore + Copy>(args: &[T]) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            let mul = t.to_mul();
            for a in args {
                mul.extend(a.as_atom_view());
            }
            let mut out = Atom::new();
            t.as_view().normalize(&mut out);
            out
        })
    }
}

impl Default for Atom {
    fn default() -> Self {
        Atom::Zero
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.as_view(), f)
    }
}

impl std::fmt::Debug for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.as_view(), f)
    }
}

impl PartialEq for Atom {
    fn eq(&self, other: &Self) -> bool {
        self.as_view() == other.as_view()
    }
}
impl Eq for Atom {}
impl Hash for Atom {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_view().hash(state)
    }
}
impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Atom {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_view().cmp(&other.as_view())
    }
}

impl From<Num> for Atom {
    fn from(n: Num) -> Atom {
        Atom::Num(n)
    }
}
impl From<Var> for Atom {
    fn from(v: Var) -> Atom {
        Atom::Var(v)
    }
}
impl From<Fun> for Atom {
    fn from(f: Fun) -> Atom {
        Atom::Fun(f)
    }
}
impl From<Pow> for Atom {
    fn from(p: Pow) -> Atom {
        Atom::Pow(p)
    }
}
impl From<Mul> for Atom {
    fn from(m: Mul) -> Atom {
        Atom::Mul(m)
    }
}
impl From<Add> for Atom {
    fn from(a: Add) -> Atom {
        Atom::Add(a)
    }
}
impl From<Symbol> for Atom {
    fn from(symbol: Symbol) -> Atom {
        Atom::new_var(symbol)
    }
}

impl<'a> AtomView<'a> {
    /// Clone this borrowed view into an owned atom.
    pub fn to_owned(&self) -> Atom {
        let mut a = Atom::default();
        a.set_from_view(self);
        a
    }

    pub fn clone_into(&self, target: &mut Atom) {
        target.set_from_view(self);
    }
    pub fn is_zero(&self) -> bool {
        matches!(self, AtomView::Num(n) if n.is_zero())
    }
    pub fn is_one(&self) -> bool {
        matches!(self, AtomView::Num(n) if n.is_one())
    }
    pub fn nterms(&self) -> usize {
        if let AtomView::Add(a) = self {
            a.get_nargs()
        } else {
            1
        }
    }

    fn sub_no_norm(&self, workspace: &Workspace, rhs: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        let add = e.to_add();
        add.extend(*self);
        add.extend(rhs.neg_no_norm(workspace).as_view());
        e
    }

    fn mul_no_norm(&self, workspace: &Workspace, rhs: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        let mul = e.to_mul();
        mul.extend(*self);
        mul.extend(rhs);
        e
    }

    fn pow_no_norm(&self, workspace: &Workspace, exp: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        e.to_pow(*self, exp);
        e
    }

    fn div_no_norm(&self, workspace: &Workspace, div: AtomView<'_>) -> RecycledAtom {
        self.mul_no_norm(
            workspace,
            div.pow_no_norm(workspace, workspace.new_num(-1).as_view())
                .as_view(),
        )
    }

    fn neg_no_norm(&self, workspace: &Workspace) -> RecycledAtom {
        self.mul_no_norm(workspace, workspace.new_num(-1).as_view())
    }

    pub fn add_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        let mut add = workspace.new_atom();
        let a = add.to_add();
        a.extend(*self);
        a.extend(rhs);
        add.as_view().normalize(out);
    }

    pub fn sub_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.sub_no_norm(workspace, rhs).as_view().normalize(out);
    }

    pub fn mul_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.mul_no_norm(workspace, rhs).as_view().normalize(out);
    }

    pub fn pow_with_ws_into(&self, workspace: &Workspace, exp: AtomView<'_>, out: &mut Atom) {
        self.pow_no_norm(workspace, exp).as_view().normalize(out);
    }

    pub fn div_with_ws_into(&self, workspace: &Workspace, div: AtomView<'_>, out: &mut Atom) {
        self.div_no_norm(workspace, div).as_view().normalize(out);
    }

    pub fn neg_with_ws_into(&self, workspace: &Workspace, out: &mut Atom) {
        self.neg_no_norm(workspace).as_view().normalize(out);
    }

    pub fn get_byte_size(&self) -> usize {
        match self {
            AtomView::Num(n) => n.get_byte_size(),
            AtomView::Var(v) => v.get_byte_size(),
            AtomView::Fun(f) => f.get_byte_size(),
            AtomView::Pow(p) => p.get_byte_size(),
            AtomView::Mul(m) => m.get_byte_size(),
            AtomView::Add(a) => a.get_byte_size(),
        }
    }
}

/// Incremental builder for packed function applications.
#[derive(Clone)]
pub struct FunctionBuilder {
    handle: RecycledAtom,
}

impl FunctionBuilder {
    /// Start building a function application with the given symbol.
    pub fn new(name: Symbol) -> FunctionBuilder {
        let mut a = RecycledAtom::new();
        a.to_fun(name);
        FunctionBuilder { handle: a }
    }

    /// Append one argument to the function under construction.
    pub fn add_arg<T: AtomCore>(mut self, arg: T) -> FunctionBuilder {
        if let Atom::Fun(f) = self.handle.deref_mut() {
            f.add_arg(arg.as_atom_view());
        }
        self
    }

    /// Normalize and return the finished function application.
    pub fn finish(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut f = ws.new_atom();
            self.handle.as_view().normalize(&mut f);
            f.into_inner()
        })
    }
}

/// Helper trait for values that can be appended to a [`FunctionBuilder`].
pub trait FunctionArgument {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder;
}

impl FunctionArgument for Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}
impl FunctionArgument for &Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}
impl<'a> FunctionArgument for AtomView<'a> {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(*self)
    }
}
impl FunctionArgument for Symbol {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        let v = InlineVar::new(*self);
        f.add_arg(v.as_view())
    }
}
impl<T: Into<Coefficient> + Clone> FunctionArgument for T {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(Atom::new_num(self.clone()))
    }
}

#[macro_export]
macro_rules! function {
    ($name: expr) => {{
        $crate::atom::FunctionBuilder::new($name).finish()
    }};
    ($name: expr, $($id: expr),*) => {{
        let mut f = $crate::symbolic::View::atom::FunctionBuilder::new($name);
        $(f = $crate::symbolic::View::atom::FunctionArgument::add_arg_to_function_builder(&$id, f);)*
        f.finish()
    }};
}

#[macro_export]
macro_rules! symbol {
    ($id: expr) => {
        $crate::symbolic::View::state::Symbol::new($crate::wrap_symbol!($id))
    };
    ($id: expr; $($attr: ident),* $(,)?) => {
        $crate::state::Symbol::new_with_attributes($crate::wrap_symbol!($id), &[$($crate::atom::FunctionAttribute::$attr,)*])
    };
    ($first: expr, $second: expr $(, $rest: expr)* $(,)?) => {
        (
            $crate::symbolic::View::state::Symbol::new($crate::wrap_symbol!($first)),
            $crate::symbolic::View::state::Symbol::new($crate::wrap_symbol!($second))
            $(, $crate::symbolic::View::state::Symbol::new($crate::wrap_symbol!($rest)) )*
        )
    };
}

#[macro_export]
macro_rules! parse {
    ($s: expr) => {{ $crate::symbolic::View::atom::Atom::parse($crate::wrap_input!($s)) }};
    ($s: expr, $ns: expr) => {{ $crate::atom::Atom::parse($crate::with_default_namespace!($s, $ns)) }};
}

#[macro_export]
macro_rules! parse_lit {
    ($s: expr) => {{ $crate::atom::Atom::parse($crate::wrap_input!(stringify!($s))) }};
    ($s: expr, $ns: expr) => {{ $crate::atom::Atom::parse($crate::with_default_namespace!(stringify!($s), $ns)) }};
}

impl std::ops::Add<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;
    fn add(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.add_with_ws_into(ws, rhs, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Sub<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;
    fn sub(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.sub_with_ws_into(ws, rhs, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Mul<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;
    fn mul(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.mul_with_ws_into(ws, rhs, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Div<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;
    fn div(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.div_with_ws_into(ws, rhs, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Neg for AtomView<'_> {
    type Output = Atom;
    fn neg(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.neg_with_ws_into(ws, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Add<AtomView<'_>> for Atom {
    type Output = Atom;
    fn add(mut self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().add_with_ws_into(ws, rhs, &mut t);
            std::mem::swap(&mut self, &mut t);
        });
        self
    }
}
impl std::ops::Sub<AtomView<'_>> for Atom {
    type Output = Atom;
    fn sub(mut self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().sub_with_ws_into(ws, rhs, &mut t);
            std::mem::swap(&mut self, &mut t);
        });
        self
    }
}
impl std::ops::Mul<AtomView<'_>> for Atom {
    type Output = Atom;
    fn mul(mut self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().mul_with_ws_into(ws, rhs, &mut t);
            std::mem::swap(&mut self, &mut t);
        });
        self
    }
}
impl std::ops::Div<AtomView<'_>> for Atom {
    type Output = Atom;
    fn div(mut self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().div_with_ws_into(ws, rhs, &mut t);
            std::mem::swap(&mut self, &mut t);
        });
        self
    }
}
impl std::ops::Neg for Atom {
    type Output = Atom;
    fn neg(mut self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().neg_with_ws_into(ws, &mut t);
            std::mem::swap(&mut self, &mut t);
        });
        self
    }
}

impl std::ops::Add<Atom> for Atom {
    type Output = Atom;
    fn add(self, rhs: Atom) -> Atom {
        self + rhs.as_view()
    }
}
impl std::ops::Sub<Atom> for Atom {
    type Output = Atom;
    fn sub(self, rhs: Atom) -> Atom {
        self - rhs.as_view()
    }
}
impl std::ops::Mul<Atom> for Atom {
    type Output = Atom;
    fn mul(self, rhs: Atom) -> Atom {
        self * rhs.as_view()
    }
}
impl std::ops::Div<Atom> for Atom {
    type Output = Atom;
    fn div(self, rhs: Atom) -> Atom {
        self / rhs.as_view()
    }
}

impl<T: Into<Coefficient>> std::ops::Add<T> for Atom {
    type Output = Atom;
    fn add(self, rhs: T) -> Atom {
        self + Atom::new_num(rhs).as_view()
    }
}
impl<T: Into<Coefficient>> std::ops::Sub<T> for Atom {
    type Output = Atom;
    fn sub(self, rhs: T) -> Atom {
        self - Atom::new_num(rhs).as_view()
    }
}
impl<T: Into<Coefficient>> std::ops::Mul<T> for Atom {
    type Output = Atom;
    fn mul(self, rhs: T) -> Atom {
        self * Atom::new_num(rhs).as_view()
    }
}
impl<T: Into<Coefficient>> std::ops::Div<T> for Atom {
    type Output = Atom;
    fn div(self, rhs: T) -> Atom {
        self / Atom::new_num(rhs).as_view()
    }
}

impl AsRef<Atom> for Atom {
    fn as_ref(&self) -> &Atom {
        self
    }
}
