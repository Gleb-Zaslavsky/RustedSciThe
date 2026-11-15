//! Shared, opt-in cooperative cancellation primitive for numerical solvers.
//!
//! Cloning a token shares one atomic cancellation flag. Solvers should poll it
//! only at their documented safe checkpoints, never from residual/Jacobian
//! callbacks. Passing no token remains the zero-work default.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Thread-safe cancellation signal that can be shared with a running solver.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new, active cancellation token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Repeated calls are harmless.
    #[inline]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::CancellationToken;

    #[test]
    fn clones_observe_cancellation_across_threads() {
        let token = CancellationToken::new();
        let worker_token = token.clone();
        let worker = std::thread::spawn(move || worker_token.cancel());
        worker.join().unwrap();
        assert!(token.is_cancelled());
    }

    #[test]
    fn fresh_tokens_are_independent_and_repeated_cancel_is_safe() {
        let first = CancellationToken::new();
        let second = CancellationToken::new();
        first.cancel();
        first.cancel();
        assert!(first.is_cancelled());
        assert!(!second.is_cancelled());
    }
}
