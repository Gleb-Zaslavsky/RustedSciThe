//! A small symbolic-algebra crate built around the original packed atom core.
//!
//! # Dataflow
//! The main dataflow is:
//! `parse` -> packed [`Atom`] construction -> `normalize` -> `derivative` / `evaluate` -> `print`.
//!
//! Parsing produces packed atoms directly, so later stages can work mostly with
//! borrowed [`AtomView`] values instead of allocating temporary trees. Normalization
//! is the semantic center of the crate: constructors and algebraic operations are
//! intentionally thin, and correctness comes from normalizing the packed structure.
//!
//! # Why this stays fast
//! - Expressions are stored in append-only byte buffers instead of pointer-heavy node graphs.
//! - [`AtomView`] is a zero-copy view into those bytes, so many traversals avoid cloning.
//! - Normalization merges factors and terms in-place with small temporary buffers.
//! - [`Workspace`] and [`RecycledAtom`] reuse allocations across operations.
//! - Coefficients stay in small fixed-size rationals, which removes big-number overhead.
//!
//! Those design choices are why the simplified crate can remove features without replacing
//! the underlying data structures with slower owned or boxed forms.

pub mod CodegenIR;
pub mod CodegenIR_atom;
pub mod atom;
pub mod boxed_interactions;
pub mod coefficient;
pub mod conversions;
pub mod derivative;
pub mod evaluate;
pub mod lambdify;
pub mod normalize;
pub mod parser;
pub mod printer;
pub mod state;

pub use atom::{Atom, AtomCore, AtomView};
pub use coefficient::Coefficient;
pub use derivative::{try_derivative, DerivativeError};
pub use evaluate::{
    evaluate, evaluate_exact, evaluate_exact_with_symbols, evaluate_with_symbols, prepare_evaluator,
    EvaluationCache, EvaluationFn, EvaluationSymbolFn, ExactConstMap, ExactSymbolMap,
    FloatSymbolMap, FunctionMap, PreparedEvaluator,
};
pub use parser::parse;
pub use printer::{print, print_debug, print_with_options, PrintOptions};
pub use state::{State, Symbol, Workspace};
