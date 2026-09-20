//! Runtime binding for symbolic parameters that are not Newton unknowns.
//!
//! The symbolic parameter order is fixed when a residual/Jacobian callback is
//! compiled. Numeric values may then be replaced between solves without
//! rebuilding the compiled closures. A shared-read snapshot is taken once per
//! request; worker evaluation never locks or mutates Jacobian output.

use std::sync::Arc;
use std::sync::RwLock;

/// Shared numeric binding for one prepared parameter ABI.
#[derive(Clone, Debug)]
pub(crate) struct BvpParameterBindingHandle {
    values: Arc<RwLock<Option<Arc<[f64]>>>>,
}

impl BvpParameterBindingHandle {
    /// Creates a binding from the values captured at callback preparation.
    pub(crate) fn new(values: Option<Vec<f64>>) -> Self {
        Self {
            values: Arc::new(RwLock::new(values.map(Arc::<[f64]>::from))),
        }
    }

    /// Replaces only numeric values; the symbolic parameter order is external
    /// metadata and must not change for an existing prepared callback.
    pub(crate) fn replace(&self, values: Option<Vec<f64>>) {
        *self
            .values
            .write()
            .expect("parameter binding lock poisoned") = values.map(Arc::<[f64]>::from);
    }

    /// Takes a cheap snapshot for one residual/Jacobian request.
    #[inline]
    pub(crate) fn snapshot(&self) -> Option<Arc<[f64]>> {
        self.values
            .read()
            .expect("parameter binding lock poisoned")
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::BvpParameterBindingHandle;

    #[test]
    fn numeric_rebind_replaces_arc_without_changing_handle_identity() {
        let handle = BvpParameterBindingHandle::new(Some(vec![1.0, 2.0]));
        let same_handle = handle.clone();
        assert_eq!(handle.snapshot().as_deref(), Some(&[1.0, 2.0][..]));

        same_handle.replace(Some(vec![3.0, 4.0]));
        assert_eq!(handle.snapshot().as_deref(), Some(&[3.0, 4.0][..]));
    }
}
