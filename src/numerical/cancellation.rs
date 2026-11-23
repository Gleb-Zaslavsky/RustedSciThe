//! Shared, opt-in cooperative cancellation primitive for numerical solvers.
//!
//! Cloning a token shares one atomic cancellation flag. Solvers should poll it
//! only at their documented safe checkpoints, never from residual/Jacobian
//! callbacks. Passing no token remains the zero-work default.

use std::sync::Arc;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Thread-safe cancellation signal that can be shared with a running solver.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    requested_at: Arc<OnceLock<Instant>>,
    #[cfg(test)]
    test_cancel_on_poll: Arc<AtomicUsize>,
    #[cfg(test)]
    test_poll_count: Arc<AtomicUsize>,
}

impl CancellationToken {
    /// Create a new, active cancellation token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Repeated calls are harmless.
    #[inline]
    pub fn cancel(&self) {
        let _ = self.requested_at.set(Instant::now());
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        #[cfg(test)]
        {
            let target = self.test_cancel_on_poll.load(Ordering::Relaxed);
            if target != 0 && self.test_poll_count.fetch_add(1, Ordering::Relaxed) + 1 == target {
                self.cancel();
            }
        }
        self.cancelled.load(Ordering::Acquire)
    }

    /// Monotonic timestamp of the first cancellation request, if any.
    /// Reading this metadata is intended for reports, not solver checkpoints.
    pub fn requested_at(&self) -> Option<Instant> {
        self.requested_at.get().copied()
    }

    #[cfg(test)]
    pub(crate) fn cancel_on_nth_poll_for_test(&self, poll: usize) {
        self.test_cancel_on_poll.store(poll, Ordering::Relaxed);
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

    #[test]
    fn first_request_timestamp_is_shared_and_stable() {
        let token = CancellationToken::new();
        let peer = token.clone();
        assert!(token.requested_at().is_none());
        peer.cancel();
        let first = token.requested_at().expect("cancel stores request time");
        token.cancel();
        assert_eq!(token.requested_at(), Some(first));
    }

    #[test]
    fn test_poll_trigger_cancels_at_exact_checkpoint() {
        let token = CancellationToken::new();
        token.cancel_on_nth_poll_for_test(2);
        assert!(!token.is_cancelled());
        assert!(token.is_cancelled());
        assert!(token.requested_at().is_some());
    }
}
