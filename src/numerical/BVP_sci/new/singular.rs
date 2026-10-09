//! SciPy-compatible singular-term support.
//!
//! The optional term has the form `S*y/(x-a)`. It is pointwise plumbing and
//! therefore does not change the selected global Dense/Sparse/Banded layout.

use nalgebra::{DMatrix, DVector};

use super::error::{BvpSciNewError, BvpSciStage};

/// Row-major square singular matrix supplied by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct BvpSciSingularTerm {
    dimension: usize,
    values: Vec<f64>,
}

impl BvpSciSingularTerm {
    pub fn new(dimension: usize, values: Vec<f64>) -> Result<Self, BvpSciNewError> {
        let expected = dimension.checked_mul(dimension).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("singular term size overflows usize".into())
        })?;
        if dimension == 0 || values.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::NumericalCore,
                expected,
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::NumericalCore,
            });
        }
        Ok(Self { dimension, values })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub(crate) fn prepare(&self, a: f64) -> Result<SingularTermRuntime, BvpSciNewError> {
        let s = DMatrix::from_row_slice(self.dimension, self.dimension, &self.values);
        let identity = DMatrix::identity(self.dimension, self.dimension);
        // SciPy deliberately uses a pseudoinverse here.  Requiring an
        // ordinary inverse rejects valid singular-term systems exactly when
        // `I - S` has a null space, even though the transformed endpoint RHS
        // is still well-defined on the compatible subspace.
        let endpoint_inverse = (identity.clone() - &s)
            .svd(true, true)
            .pseudo_inverse(1e-12)
            .map_err(|_| {
                BvpSciNewError::InvalidConfiguration(
                    "singular term endpoint pseudoinverse could not be constructed".into(),
                )
            })?;
        let pseudo_inverse = s
            .clone()
            .svd(true, true)
            .pseudo_inverse(1e-12)
            .map_err(|_| {
                BvpSciNewError::InvalidConfiguration(
                    "singular term null-space projection could not be constructed".into(),
                )
            })?;
        let projection = identity - pseudo_inverse * s;
        Ok(SingularTermRuntime {
            a,
            dimension: self.dimension,
            values: self.values.clone(),
            endpoint_inverse,
            projection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::BvpSciSingularTerm;

    #[test]
    fn endpoint_transform_matches_scipy_for_singular_i_minus_s() {
        // I-S is singular in the first component. SciPy's pinv keeps the
        // compatible second component instead of rejecting the configuration.
        let term = BvpSciSingularTerm::new(2, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let runtime = term.prepare(0.0).unwrap();

        let mut state = [3.0, 4.0];
        runtime.project_initial_state(&mut state);
        assert_eq!(state, [0.0, 4.0]);

        let mut rhs = [7.0, 9.0];
        runtime.apply_rhs(0.0, &state, &[7.0, 9.0], &mut rhs);
        assert_eq!(rhs, [0.0, 9.0]);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SingularTermRuntime {
    a: f64,
    dimension: usize,
    values: Vec<f64>,
    endpoint_inverse: DMatrix<f64>,
    projection: DMatrix<f64>,
}

impl SingularTermRuntime {
    pub(crate) fn endpoint(&self) -> f64 {
        self.a
    }

    pub(crate) fn project_initial_state(&self, state: &mut [f64]) {
        let projected = &self.projection * DVector::from_column_slice(state);
        state.copy_from_slice(projected.as_slice());
    }

    pub(crate) fn apply_rhs(&self, x: f64, state: &[f64], base: &[f64], output: &mut [f64]) {
        debug_assert_eq!(state.len(), self.dimension);
        debug_assert_eq!(base.len(), self.dimension);
        debug_assert_eq!(output.len(), self.dimension);
        if x == self.a {
            let transformed = &self.endpoint_inverse * DVector::from_column_slice(base);
            output.copy_from_slice(transformed.as_slice());
            return;
        }
        output.copy_from_slice(base);
        let denominator = x - self.a;
        for row in 0..self.dimension {
            let correction = (0..self.dimension)
                .map(|column| self.values[row * self.dimension + column] * state[column])
                .sum::<f64>();
            output[row] += correction / denominator;
        }
    }

    pub(crate) fn apply_jacobian(&self, x: f64, base: &[f64], output: &mut [f64]) {
        debug_assert_eq!(base.len(), self.dimension * self.dimension);
        debug_assert_eq!(output.len(), base.len());
        if x == self.a {
            let base_matrix = DMatrix::from_row_slice(self.dimension, self.dimension, base);
            let transformed = &self.endpoint_inverse * base_matrix;
            for row in 0..self.dimension {
                for column in 0..self.dimension {
                    output[row * self.dimension + column] = transformed[(row, column)];
                }
            }
            return;
        }
        output.copy_from_slice(base);
        let denominator = x - self.a;
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                output[row * self.dimension + column] +=
                    self.values[row * self.dimension + column] / denominator;
            }
        }
    }
}
