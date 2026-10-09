//! Solution ownership and SciPy-compatible piecewise-cubic dense output.

use super::error::{BvpSciNewError, BvpSciStage, BvpSciStatus};

/// Controls which optional trajectory representation is retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciOutputPolicy {
    /// Keep only convergence metadata and the final state vectors.
    FinalOnly,
    /// Keep the accepted mesh, values and interval residuals.
    MeshAndValues,
    /// Also construct a C1 piecewise-cubic Hermite spline from endpoint RHS data.
    DenseOutput,
}

/// C1 cubic Hermite interpolator over the accepted BVP mesh.
#[derive(Clone, Debug)]
pub struct BvpSciDenseOutput {
    x: Vec<f64>,
    y: Vec<f64>,
    derivatives: Vec<f64>,
    dimension: usize,
}

impl BvpSciDenseOutput {
    pub(crate) fn new(x: &[f64], y: &[f64], derivatives: &[f64]) -> Result<Self, BvpSciNewError> {
        if x.len() < 2
            || x.windows(2)
                .any(|pair| !pair[1].is_finite() || pair[1] <= pair[0])
        {
            return Err(BvpSciNewError::InvalidConfiguration(
                "dense output mesh must be finite and strictly increasing".into(),
            ));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::Output,
            });
        }
        if y.len() != derivatives.len() || y.len() % x.len() != 0 {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::Output,
                expected: y.len(),
                actual: derivatives.len(),
            });
        }
        if y.iter().chain(derivatives).any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::Output,
            });
        }
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            derivatives: derivatives.to_vec(),
            dimension: y.len() / x.len(),
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn domain(&self) -> (f64, f64) {
        (self.x[0], *self.x.last().expect("dense output has nodes"))
    }

    pub fn evaluate(&self, x: f64) -> Result<Vec<f64>, BvpSciNewError> {
        let mut output = vec![0.0; self.dimension];
        self.evaluate_into(x, &mut output)?;
        Ok(output)
    }

    /// Evaluate into caller-owned storage without allocating per query.
    ///
    /// This is the preferred path for post-processing a dense trajectory in a
    /// sampling loop. The allocating `evaluate` method remains a convenience
    /// wrapper around this contract.
    pub fn evaluate_into(&self, x: f64, output: &mut [f64]) -> Result<(), BvpSciNewError> {
        if output.len() != self.dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::Output,
                expected: self.dimension,
                actual: output.len(),
            });
        }
        let interval = self.interval(x)?;
        let h = self.x[interval + 1] - self.x[interval];
        let t = (x - self.x[interval]) / h;
        for component in 0..self.dimension {
            output[component] = hermite_value(
                t,
                h,
                self.y[interval * self.dimension + component],
                self.y[(interval + 1) * self.dimension + component],
                self.derivatives[interval * self.dimension + component],
                self.derivatives[(interval + 1) * self.dimension + component],
            );
        }
        Ok(())
    }

    pub fn evaluate_derivative(&self, x: f64) -> Result<Vec<f64>, BvpSciNewError> {
        let mut output = vec![0.0; self.dimension];
        self.evaluate_derivative_into(x, &mut output)?;
        Ok(output)
    }

    /// Evaluate the spline derivative into caller-owned storage.
    pub fn evaluate_derivative_into(
        &self,
        x: f64,
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        if output.len() != self.dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::Output,
                expected: self.dimension,
                actual: output.len(),
            });
        }
        let interval = self.interval(x)?;
        let h = self.x[interval + 1] - self.x[interval];
        let t = (x - self.x[interval]) / h;
        for component in 0..self.dimension {
            output[component] = hermite_derivative(
                t,
                h,
                self.y[interval * self.dimension + component],
                self.y[(interval + 1) * self.dimension + component],
                self.derivatives[interval * self.dimension + component],
                self.derivatives[(interval + 1) * self.dimension + component],
            );
        }
        Ok(())
    }

    fn interval(&self, x: f64) -> Result<usize, BvpSciNewError> {
        let (left, right) = self.domain();
        if !x.is_finite() || x < left || x > right {
            return Err(BvpSciNewError::OutputOutOfDomain);
        }
        if x == right {
            return Ok(self.x.len() - 2);
        }
        Ok(self.x.partition_point(|node| *node <= x).saturating_sub(1))
    }
}

/// Result of one converged collocation solve.
#[derive(Clone, Debug)]
pub struct BvpSciSolution {
    /// Accepted adaptive mesh, in strictly increasing order.
    pub x: Vec<f64>,
    /// State values on `x`, stored node-major.
    pub y: Vec<f64>,
    /// Parameter values used by the final solve.
    pub parameters: Vec<f64>,
    /// Maximum of the normalized collocation and boundary residuals.
    pub residual_norm: f64,
    /// Per-interval SciPy-style Lobatto RMS residuals.
    pub interval_residuals: Vec<f64>,
    /// SciPy-compatible successful termination status.
    pub status: BvpSciStatus,
    /// Human-readable status explanation.
    pub message: String,
    /// Total Newton iterations across all accepted meshes.
    pub iterations: usize,
    /// Optional owned cubic Hermite spline used as dense output.
    pub dense_output: Option<BvpSciDenseOutput>,
}

impl BvpSciSolution {
    pub fn dimension(&self) -> usize {
        self.y.len() / self.x.len()
    }

    pub fn values_at_node(&self, node: usize) -> Option<&[f64]> {
        let n = self.dimension();
        (node < self.x.len()).then(|| &self.y[node * n..(node + 1) * n])
    }

    pub fn dense_output(&self) -> Result<&BvpSciDenseOutput, BvpSciNewError> {
        self.dense_output
            .as_ref()
            .ok_or(BvpSciNewError::OutputUnavailable)
    }
}

fn hermite_value(t: f64, h: f64, yl: f64, yr: f64, fl: f64, fr: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * yl
        + (t3 - 2.0 * t2 + t) * h * fl
        + (-2.0 * t3 + 3.0 * t2) * yr
        + (t3 - t2) * h * fr
}

fn hermite_derivative(t: f64, h: f64, yl: f64, yr: f64, fl: f64, fr: f64) -> f64 {
    let t2 = t * t;
    ((6.0 * t2 - 6.0 * t) * yl + (-6.0 * t2 + 6.0 * t) * yr) / h
        + (3.0 * t2 - 4.0 * t + 1.0) * fl
        + (3.0 * t2 - 2.0 * t) * fr
}
