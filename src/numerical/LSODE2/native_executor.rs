//! Timed native callback executor for LSODE2 Newton iterations.
//!
//! This layer wraps three native ingredients:
//! - residual evaluation,
//! - Jacobian evaluation,
//! - linear factor/solve on `[I - cJ]`.
//!
//! It records stage timings into [`crate::numerical::LSODE2::Lsode2NativeStatistics`]
//! so the future native LSODE2 loop does not need ad-hoc wrappers for story and
//! performance tests.

use super::statistics::Lsode2NativeStatistics;
use crate::numerical::BDF::BDF_solver::{BdfJacobian, BdfLinearBackend, BdfLinearFactorization};
use nalgebra::DVector;
use std::time::Instant;

type ResidualCallback = dyn FnMut(f64, &DVector<f64>) -> DVector<f64>;
type JacobianCallback = dyn FnMut(f64, &DVector<f64>) -> BdfJacobian;

pub(crate) fn jacobian_abs_max(jacobian: &BdfJacobian) -> Option<f64> {
    let max_abs = match jacobian {
        BdfJacobian::Dense(matrix) => matrix.iter().map(|value| value.abs()).fold(0.0, f64::max),
        BdfJacobian::SparseTriplets { triplets, .. } => triplets
            .iter()
            .map(|triplet| triplet.val.abs())
            .fold(0.0, f64::max),
        BdfJacobian::Banded(matrix) => {
            let n = matrix.n();
            let mut max_abs = 0.0_f64;
            for j in 0..n {
                let i0 = j.saturating_sub(matrix.ku());
                let i1 = (j + matrix.kl() + 1).min(n);
                for i in i0..i1 {
                    max_abs = max_abs.max(matrix[(i, j)].abs());
                }
            }
            max_abs
        }
    };
    (max_abs.is_finite() && max_abs > 0.0).then_some(max_abs)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2NativeExecutorError {
    ResidualDimensionMismatch { expected: usize, actual: usize },
    LinearFactorizationFailed,
    LinearSolveFailed,
}

impl std::fmt::Display for Lsode2NativeExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResidualDimensionMismatch { expected, actual } => write!(
                f,
                "LSODE2 native residual dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::LinearFactorizationFailed => {
                write!(f, "LSODE2 native linear factorization failed")
            }
            Self::LinearSolveFailed => write!(f, "LSODE2 native linear solve failed"),
        }
    }
}

impl std::error::Error for Lsode2NativeExecutorError {}

pub struct Lsode2NativeCallbackExecutor<L> {
    residual: Box<ResidualCallback>,
    jacobian: Box<JacobianCallback>,
    linear_backend: L,
    cached_t: Option<f64>,
    cached_c: Option<f64>,
    cached_factorization: Option<Box<dyn BdfLinearFactorization>>,
    last_jacobian_abs_max: Option<f64>,
}

impl<L> std::fmt::Debug for Lsode2NativeCallbackExecutor<L>
where
    L: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lsode2NativeCallbackExecutor")
            .field("linear_backend", &self.linear_backend)
            .finish_non_exhaustive()
    }
}

impl<L> Lsode2NativeCallbackExecutor<L> {
    pub fn new<R, J>(residual: R, jacobian: J, linear_backend: L) -> Self
    where
        R: FnMut(f64, &DVector<f64>) -> DVector<f64> + 'static,
        J: FnMut(f64, &DVector<f64>) -> BdfJacobian + 'static,
    {
        Self {
            residual: Box::new(residual),
            jacobian: Box::new(jacobian),
            linear_backend,
            cached_t: None,
            cached_c: None,
            cached_factorization: None,
            last_jacobian_abs_max: None,
        }
    }
}

impl<L> Lsode2NativeCallbackExecutor<L>
where
    L: BdfLinearBackend,
{
    pub fn eval_residual(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        statistics: &mut Lsode2NativeStatistics,
    ) -> Result<DVector<f64>, Lsode2NativeExecutorError> {
        let started = Instant::now();
        let residual = (self.residual)(t, y);
        statistics.record_native_residual_duration(started.elapsed());
        if residual.len() != y.len() {
            return Err(Lsode2NativeExecutorError::ResidualDimensionMismatch {
                expected: y.len(),
                actual: residual.len(),
            });
        }
        Ok(residual)
    }

    pub fn eval_jacobian(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        statistics: &mut Lsode2NativeStatistics,
    ) -> BdfJacobian {
        let started = Instant::now();
        let jacobian = (self.jacobian)(t, y);
        statistics.record_native_jacobian_duration(started.elapsed());
        self.last_jacobian_abs_max = jacobian_abs_max(&jacobian);
        jacobian
    }

    pub fn last_jacobian_abs_max(&self) -> Option<f64> {
        self.last_jacobian_abs_max
    }

    pub fn invalidate_linearization(&mut self) {
        self.cached_t = None;
        self.cached_c = None;
        self.cached_factorization = None;
    }

    pub fn has_current_linearization(&self, t: f64, c: f64) -> bool {
        self.cached_factorization.is_some() && self.cached_t == Some(t) && self.cached_c == Some(c)
    }

    fn refresh_linearization(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        c: f64,
        statistics: &mut Lsode2NativeStatistics,
    ) -> Result<(), Lsode2NativeExecutorError> {
        let jacobian = self.eval_jacobian(t, y, statistics);
        let factorization = self
            .linear_backend
            .factor_shifted_jacobian(c, &jacobian)
            .ok_or(Lsode2NativeExecutorError::LinearFactorizationFailed)?;
        self.cached_t = Some(t);
        self.cached_c = Some(c);
        self.cached_factorization = Some(factorization);
        Ok(())
    }

    pub fn solve_shifted_system(
        &mut self,
        c: f64,
        jacobian: &BdfJacobian,
        rhs: &DVector<f64>,
        statistics: &mut Lsode2NativeStatistics,
    ) -> Result<DVector<f64>, Lsode2NativeExecutorError> {
        self.cached_t = None;
        self.cached_c = Some(c);
        self.cached_factorization = Some(
            self.linear_backend
                .factor_shifted_jacobian(c, jacobian)
                .ok_or(Lsode2NativeExecutorError::LinearFactorizationFailed)?,
        );
        let started = Instant::now();
        let solution = self
            .cached_factorization
            .as_ref()
            .expect("cached factorization should exist after successful factorization")
            .solve(rhs)
            .ok_or(Lsode2NativeExecutorError::LinearSolveFailed)?;
        statistics.record_native_linear_solve_duration(started.elapsed());
        Ok(solution)
    }

    pub fn compute_newton_correction(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        c: f64,
        statistics: &mut Lsode2NativeStatistics,
    ) -> Result<DVector<f64>, Lsode2NativeExecutorError> {
        self.compute_newton_correction_with_refresh_policy(t, y, c, statistics, false)
    }

    pub fn compute_newton_correction_with_refresh_policy(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        c: f64,
        statistics: &mut Lsode2NativeStatistics,
        force_refresh: bool,
    ) -> Result<DVector<f64>, Lsode2NativeExecutorError> {
        let residual = self.eval_residual(t, y, statistics)?;
        let rhs = -residual;
        if force_refresh {
            self.invalidate_linearization();
        }
        if !self.has_current_linearization(t, c) {
            self.refresh_linearization(t, y, c, statistics)?;
        }
        let started = Instant::now();
        let solution = self
            .cached_factorization
            .as_ref()
            .expect("cached factorization should exist after refresh")
            .solve(&rhs)
            .ok_or(Lsode2NativeExecutorError::LinearSolveFailed)?;
        statistics.record_native_linear_solve_duration(started.elapsed());
        Ok(solution)
    }

    /// ODEPACK-style functional-iteration (`MITER = 0`) correction:
    /// use only residual work and skip Jacobian/factor/solve.
    ///
    /// For the native step residual `g(y)`, this returns `-g(y)`, which is
    /// exactly the additive correction used by the DSTODA `M=0` update path.
    pub fn compute_functional_correction(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        statistics: &mut Lsode2NativeStatistics,
    ) -> Result<DVector<f64>, Lsode2NativeExecutorError> {
        let residual = self.eval_residual(t, y, statistics)?;
        Ok(-residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::LSODE2::FaerSparseBdfLinearBackend;
    use faer::sparse::Triplet;

    #[test]
    fn native_executor_computes_newton_correction_and_records_stage_timings() {
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 1.0]),
            |_, _| BdfJacobian::SparseTriplets {
                n: 1,
                triplets: vec![Triplet::new(0, 0, 0.0)],
            },
            FaerSparseBdfLinearBackend::default(),
        );
        let mut statistics = Lsode2NativeStatistics::default();

        let correction = executor
            .compute_newton_correction(
                0.1,
                &DVector::from_vec(vec![1.0 + 1.0e-5]),
                0.0,
                &mut statistics,
            )
            .unwrap();

        assert!((correction[0] + 1.0e-5).abs() < 1e-12);
        assert_eq!(statistics.native_residual_calls, 1);
        assert_eq!(statistics.native_jacobian_calls, 1);
        assert_eq!(statistics.native_linear_solve_calls, 1);
    }

    #[test]
    fn native_executor_reuses_cached_linearization_across_newton_iterations() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let jacobian_calls = Rc::new(RefCell::new(0usize));
        let jacobian_calls_handle = Rc::clone(&jacobian_calls);
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 1.0]),
            move |_, _| {
                *jacobian_calls_handle.borrow_mut() += 1;
                BdfJacobian::SparseTriplets {
                    n: 1,
                    triplets: vec![Triplet::new(0, 0, 0.0)],
                }
            },
            FaerSparseBdfLinearBackend::default(),
        );
        let mut statistics = Lsode2NativeStatistics::default();

        let _first = executor
            .compute_newton_correction(
                0.1,
                &DVector::from_vec(vec![1.0 + 1.0e-2]),
                0.0,
                &mut statistics,
            )
            .unwrap();
        let _second = executor
            .compute_newton_correction(
                0.1,
                &DVector::from_vec(vec![1.0 + 1.0e-4]),
                0.0,
                &mut statistics,
            )
            .unwrap();

        assert_eq!(*jacobian_calls.borrow(), 1);
        assert_eq!(statistics.native_residual_calls, 2);
        assert_eq!(statistics.native_jacobian_calls, 1);
        assert_eq!(statistics.native_linear_solve_calls, 2);
        assert!(executor.has_current_linearization(0.1, 0.0));
    }

    #[test]
    fn native_executor_can_force_jacobian_refresh_between_iterations() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let jacobian_calls = Rc::new(RefCell::new(0usize));
        let jacobian_calls_handle = Rc::clone(&jacobian_calls);
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 1.0]),
            move |_, _| {
                *jacobian_calls_handle.borrow_mut() += 1;
                BdfJacobian::SparseTriplets {
                    n: 1,
                    triplets: vec![Triplet::new(0, 0, 0.0)],
                }
            },
            FaerSparseBdfLinearBackend::default(),
        );
        let mut statistics = Lsode2NativeStatistics::default();

        let _first = executor
            .compute_newton_correction(
                0.1,
                &DVector::from_vec(vec![1.0 + 1.0e-2]),
                0.0,
                &mut statistics,
            )
            .unwrap();
        let _second = executor
            .compute_newton_correction_with_refresh_policy(
                0.1,
                &DVector::from_vec(vec![1.0 + 1.0e-4]),
                0.0,
                &mut statistics,
                true,
            )
            .unwrap();

        assert_eq!(*jacobian_calls.borrow(), 2);
        assert_eq!(statistics.native_residual_calls, 2);
        assert_eq!(statistics.native_jacobian_calls, 2);
        assert_eq!(statistics.native_linear_solve_calls, 2);
        assert!(executor.has_current_linearization(0.1, 0.0));
    }

    #[test]
    fn native_executor_refreshes_linearization_when_trial_time_changes() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let jacobian_calls = Rc::new(RefCell::new(0usize));
        let jacobian_calls_handle = Rc::clone(&jacobian_calls);
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 1.0]),
            move |_, _| {
                *jacobian_calls_handle.borrow_mut() += 1;
                BdfJacobian::SparseTriplets {
                    n: 1,
                    triplets: vec![Triplet::new(0, 0, 0.0)],
                }
            },
            FaerSparseBdfLinearBackend::default(),
        );
        let mut statistics = Lsode2NativeStatistics::default();

        let _first = executor
            .compute_newton_correction(
                0.1,
                &DVector::from_vec(vec![1.0 + 1.0e-2]),
                0.5,
                &mut statistics,
            )
            .unwrap();
        let _second = executor
            .compute_newton_correction(
                0.2,
                &DVector::from_vec(vec![1.0 + 1.0e-4]),
                0.5,
                &mut statistics,
            )
            .unwrap();

        assert_eq!(*jacobian_calls.borrow(), 2);
        assert_eq!(statistics.native_residual_calls, 2);
        assert_eq!(statistics.native_jacobian_calls, 2);
        assert_eq!(statistics.native_linear_solve_calls, 2);
        assert!(!executor.has_current_linearization(0.1, 0.5));
        assert!(executor.has_current_linearization(0.2, 0.5));
    }

    #[test]
    fn native_executor_functional_correction_uses_residual_only() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let jacobian_calls = Rc::new(RefCell::new(0usize));
        let jacobian_calls_handle = Rc::clone(&jacobian_calls);
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 1.0]),
            move |_, _| {
                *jacobian_calls_handle.borrow_mut() += 1;
                BdfJacobian::SparseTriplets {
                    n: 1,
                    triplets: vec![Triplet::new(0, 0, 0.0)],
                }
            },
            FaerSparseBdfLinearBackend::default(),
        );
        let mut statistics = Lsode2NativeStatistics::default();

        let correction = executor
            .compute_functional_correction(
                0.1,
                &DVector::from_vec(vec![1.0 + 3.0e-4]),
                &mut statistics,
            )
            .expect("functional correction should be computed from residual");

        assert!((correction[0] + 3.0e-4).abs() < 1.0e-12);
        assert_eq!(statistics.native_residual_calls, 1);
        assert_eq!(statistics.native_jacobian_calls, 0);
        assert_eq!(statistics.native_linear_solve_calls, 0);
        assert_eq!(*jacobian_calls.borrow(), 0);
    }

    #[test]
    fn native_executor_rejects_bad_residual_dimension() {
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, _| DVector::from_vec(vec![1.0, 2.0]),
            |_, _| BdfJacobian::SparseTriplets {
                n: 1,
                triplets: vec![],
            },
            FaerSparseBdfLinearBackend::default(),
        );
        let mut statistics = Lsode2NativeStatistics::default();
        let err = executor
            .eval_residual(0.0, &DVector::from_vec(vec![1.0]), &mut statistics)
            .unwrap_err();

        assert!(matches!(
            err,
            Lsode2NativeExecutorError::ResidualDimensionMismatch {
                expected: 1,
                actual: 2
            }
        ));
        assert_eq!(statistics.native_residual_calls, 1);
    }
}
