use crate::numerical::Nonlinear_systems::error::SolveError;
use nalgebra::{DMatrix, DVector};

/// Box constraints for a nonlinear system.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    /// Lower and upper limits for each variable.
    limits: Vec<(f64, f64)>,
}

impl Bounds {
    /// Creates validated bounds from `(lower, upper)` pairs.
    pub fn new(limits: Vec<(f64, f64)>) -> Result<Self, SolveError> {
        if limits.is_empty() {
            return Err(SolveError::InvalidConfig(
                "bounds must not be empty".to_string(),
            ));
        }
        for (index, (lower, upper)) in limits.iter().copied().enumerate() {
            if lower > upper {
                return Err(SolveError::InvalidConfig(format!(
                    "invalid bounds at index {index}: lower bound {lower} exceeds upper bound {upper}"
                )));
            }
        }
        Ok(Self { limits })
    }

    /// Converts `Option<Vec<_>>` into validated optional bounds.
    pub fn from_optional(limits: Option<Vec<(f64, f64)>>) -> Result<Option<Self>, SolveError> {
        limits.map(Self::new).transpose()
    }

    /// Returns the problem dimension covered by the bounds.
    pub fn len(&self) -> usize {
        self.limits.len()
    }

    /// Returns `true` when the bounds collection is empty.
    pub fn is_empty(&self) -> bool {
        self.limits.is_empty()
    }

    /// Returns the internal slice of `(lower, upper)` pairs.
    pub fn as_slice(&self) -> &[(f64, f64)] {
        &self.limits
    }

    /// Checks that a point is inside the bounds.
    pub fn validate(&self, x: &DVector<f64>) -> Result<(), SolveError> {
        if x.len() != self.len() {
            return Err(SolveError::DimensionMismatch {
                expected: self.len(),
                actual: x.len(),
                context: "bounds validation",
            });
        }

        for (index, value) in x.iter().copied().enumerate() {
            let (lower, upper) = self.limits[index];
            if value < lower || value > upper {
                return Err(SolveError::InfeasibleInitialGuess {
                    index,
                    value,
                    lower,
                    upper,
                });
            }
        }
        Ok(())
    }

    /// Projects a point back to the box.
    pub fn project(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut projected = x.clone();
        let limit = projected.len().min(self.limits.len());
        for index in 0..limit {
            let (lower, upper) = self.limits[index];
            projected[index] = projected[index].clamp(lower, upper);
        }
        projected
    }

    /// Returns the largest scalar `alpha` such that `x + alpha * step` stays inside the box.
    pub fn max_step_scale(&self, x: &DVector<f64>, step: &DVector<f64>) -> f64 {
        let mut scale: f64 = 1.0;
        let limit = x.len().min(step.len()).min(self.limits.len());
        for index in 0..limit {
            let value = x[index];
            let delta = step[index];
            let (lower, upper) = self.limits[index];
            if delta > 0.0 {
                scale = scale.min((upper - value) / delta);
            } else if delta < 0.0 {
                scale = scale.min((lower - value) / delta);
            }
        }
        if scale.is_finite() { scale } else { 1.0 }
    }
}

/// Problem definition based on residual evaluation.
pub trait NonlinearProblem {
    /// Returns the number of unknowns.
    fn dimension(&self) -> usize;

    /// Evaluates the residual vector at `x`.
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError>;
}

/// Extension of [`NonlinearProblem`] with an analytic Jacobian.
pub trait JacobianProvider: NonlinearProblem {
    /// Evaluates the Jacobian matrix at `x`.
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError>;
}
