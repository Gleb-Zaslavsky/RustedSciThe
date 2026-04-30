//! Lightweight abstraction over values that can be viewed as atoms.
//!
//! The trait keeps generic APIs flexible without forcing callers to allocate owned atoms.
//! It is used heavily by builders and arithmetic helpers so they can accept owned atoms,
//! borrowed views, and inline packed values through one entry point.

use super::{
    Atom, AtomOrView, AtomView,
    representation::{InlineNum, InlineVar},
};

/// Common interface for anything that can cheaply expose an [`AtomView`].
pub trait AtomCore {
    /// Borrow this value as an atom view.
    fn as_atom_view(&self) -> AtomView<'_>;

    /// Materialize an owned atom by cloning the underlying view.
    fn to_atom(&self) -> Atom {
        self.as_atom_view().to_owned()
    }
}

impl AtomCore for Atom {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

impl<'a> AtomCore for AtomView<'a> {
    fn as_atom_view(&self) -> AtomView<'_> {
        *self
    }
}

impl AtomCore for InlineVar {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

impl AtomCore for InlineNum {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

impl<'a> AtomCore for AtomOrView<'a> {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

impl AtomCore for &Atom {
    fn as_atom_view(&self) -> AtomView<'_> {
        (*self).as_view()
    }
}
impl AtomCore for &mut Atom {
    fn as_atom_view(&self) -> AtomView<'_> {
        (**self).as_view()
    }
}
