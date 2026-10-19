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

    /// Projects a point back to the box in place.
    ///
    /// This is the allocation-free counterpart of [`Self::project`]. It is
    /// useful on solver trial points, where the vector has already been
    /// allocated by the method's step arithmetic.
    pub fn project_in_place(&self, x: &mut DVector<f64>) {
        let limit = x.len().min(self.limits.len());
        for index in 0..limit {
            let (lower, upper) = self.limits[index];
            x[index] = x[index].clamp(lower, upper);
        }
    }

    /// Projects a point back to the box.
    pub fn project(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut projected = x.clone();
        self.project_in_place(&mut projected);
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

    /// Returns whether [`Self::residual_into`] is an allocation-aware path.
    ///
    /// This is opt-in so adding the compatibility method cannot make existing
    /// providers slower. Implementors overriding `residual_into` should also
    /// override this method and return `true`.
    fn supports_residual_into(&self) -> bool {
        false
    }

    /// Evaluates the residual into caller-owned storage.
    ///
    /// The default keeps existing implementations source-compatible by
    /// delegating to [`Self::residual`] and copying the returned vector. A
    /// provider with reusable storage can override this method to write
    /// directly into `out` and avoid a temporary allocation per evaluation.
    fn residual_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), SolveError> {
        let residual = self.residual(x)?;
        if residual.len() != out.len() {
            return Err(SolveError::DimensionMismatch {
                expected: self.dimension(),
                actual: residual.len(),
                context: "residual evaluation",
            });
        }
        out.copy_from(&residual);
        Ok(())
    }
}

/// Extension of [`NonlinearProblem`] with an analytic Jacobian.
pub trait JacobianProvider: NonlinearProblem {
    /// Evaluates the Jacobian matrix at `x`.
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError>;

    /// Returns whether [`Self::jacobian_into`] is an allocation-aware path.
    ///
    /// This is opt-in for the same compatibility reason as
    /// [`NonlinearProblem::supports_residual_into`].
    fn supports_jacobian_into(&self) -> bool {
        false
    }

    /// Evaluates the Jacobian into caller-owned storage.
    ///
    /// The default preserves the owned-returning API. Providers that can
    /// reuse matrix storage may override it to fill `out` in place.
    fn jacobian_into(&self, x: &DVector<f64>, out: &mut DMatrix<f64>) -> Result<(), SolveError> {
        let jacobian = self.jacobian(x)?;
        if jacobian.nrows() != out.nrows() || jacobian.ncols() != out.ncols() {
            return Err(SolveError::InvalidConfig(format!(
                "Jacobian evaluation returned {}x{}, expected {}x{}",
                jacobian.nrows(),
                jacobian.ncols(),
                out.nrows(),
                out.ncols()
            )));
        }
        out.copy_from(&jacobian);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_in_place_matches_owned_projection() {
        let bounds = Bounds::new(vec![(0.0, 1.0), (-2.0, 2.0), (10.0, 20.0)]).expect("bounds");
        let input = DVector::from_vec(vec![-3.0, 0.5, 25.0]);
        let expected = bounds.project(&input);
        let mut actual = input;

        bounds.project_in_place(&mut actual);

        assert_eq!(actual, expected);
        assert_eq!(actual.as_slice(), &[0.0, 0.5, 20.0]);
    }

    #[test]
    fn method_workspace_reuses_affine_trial_storage() {
        let base = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let direction = DVector::from_vec(vec![0.5, -1.0, 2.0]);
        let mut workspace = crate::numerical::Nonlinear_systems::engine::MethodWorkspace::new(3);
        let initial_pointer = workspace.trial_x().as_slice().as_ptr();

        workspace
            .set_affine_trial(&base, -2.0, &direction)
            .expect("matching dimensions");

        assert_eq!(workspace.trial_x().as_slice(), &[0.0, 4.0, -1.0]);
        assert_eq!(workspace.trial_x().as_slice().as_ptr(), initial_pointer);
    }
}
