//! Reusable evaluation of a successfully fitted symbolic model.
//!
//! The scalar Lambdify callbacks are compiled lazily on the first evaluation
//! and shared by clones of the fitted model. Fit results can therefore be
//! queried repeatedly without rerunning fitting or symbolic compilation.

use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, OnceLock};
use thiserror::Error;

type ScalarEvaluator = Arc<dyn Fn(f64) -> f64 + Send + Sync>;

struct FittedModelInner {
    equations: Vec<Expr>,
    independent_variable: String,
    observed_x: Vec<f64>,
    evaluators: OnceLock<Vec<ScalarEvaluator>>,
}

impl std::fmt::Debug for FittedModelInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedModelInner")
            .field("equation_count", &self.equations.len())
            .field("independent_variable", &self.independent_variable)
            .field("observed_x_count", &self.observed_x.len())
            .field("evaluators_compiled", &self.evaluators.get().is_some())
            .finish()
    }
}

/// Reusable evaluator retained alongside a successful universal fitting result.
#[derive(Clone, Debug)]
pub struct FittedModel(Arc<FittedModelInner>);

/// Values of a fitted model at requested coordinates.
///
/// `values` is sample-major: rows correspond to `x`, columns to model equations.
#[derive(Clone, Debug)]
pub struct FittedValues {
    /// Coordinates in the same order as requested.
    pub x: Vec<f64>,
    /// Model values with shape `(x.len(), equation_count)`.
    pub values: DMatrix<f64>,
}

/// One internal interval whose missing samples should be reconstructed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MissingInterval {
    /// Left boundary of the gap.
    pub start: f64,
    /// Right boundary of the gap.
    pub end: f64,
    /// Number of new interior samples; interval boundaries are not returned.
    pub missing_points: usize,
}

/// Errors from post-fit model evaluation and grid construction.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FittedModelError {
    /// A model requires at least one fitted expression and observed coordinate.
    #[error("fitted model expressions and training coordinates must be non-empty")]
    EmptyModel,
    /// A backend completed fitting but did not expose its fitted parameter map.
    #[error("successful fit did not expose fitted parameters")]
    MissingFittedParameters,
    /// The independent-variable name was not supplied.
    #[error("fitted model independent-variable name must be non-empty")]
    MissingIndependentVariable,
    /// A training coordinate is not finite.
    #[error("training coordinate at index {index} is not finite")]
    NonFiniteTrainingCoordinate { index: usize },
    /// No coordinates were supplied for evaluation.
    #[error("evaluation coordinates must be non-empty")]
    EmptyCoordinates,
    /// A requested coordinate is not finite.
    #[error("evaluation coordinate at index {index} is not finite")]
    NonFiniteCoordinate { index: usize },
    /// A fitted callback returned a non-finite value.
    #[error("fitted equation {equation} returned a non-finite value at sample {sample}")]
    NonFinitePrediction { sample: usize, equation: usize },
    /// A uniform range must have finite, strictly increasing endpoints.
    #[error("uniform evaluation range requires finite endpoints with start < end")]
    InvalidUniformRange,
    /// A uniform range must contain at least two points.
    #[error("uniform evaluation requires at least two total points, got {actual}")]
    TooFewUniformPoints { actual: usize },
    /// The requested output dimensions cannot be represented by `usize`.
    #[error("fitted evaluation output size overflows usize")]
    OutputSizeOverflow,
    /// The output buffer could not reserve the requested capacity.
    #[error("could not reserve {requested} fitted output elements")]
    AllocationFailed { requested: usize },
    /// Floating-point precision cannot represent all requested grid points.
    #[error("uniform grid points are not distinct at the requested scale")]
    UnrepresentableUniformGrid,
    /// Floating-point precision cannot represent all interior points in a gap.
    #[error("missing interval {index} has more points than its floating-point range can represent")]
    UnrepresentableMissingInterval { index: usize },
    /// A missing-point count cannot be extended to include both interval ends.
    #[error("missing-point count overflows while constructing interval {index}")]
    MissingPointCountOverflow { index: usize },
    /// At least one explicitly requested missing interval is required.
    #[error("at least one missing interval must be specified")]
    EmptyMissingIntervals,
    /// A missing interval has invalid boundaries or no interior points.
    #[error("missing interval {index} requires finite start < end and at least one interior point")]
    InvalidMissingInterval { index: usize },
    /// A gap must lie within the fitted-data domain.
    #[error("missing interval {index} lies outside the training-data domain")]
    MissingIntervalOutsideDomain { index: usize },
    /// A requested gap overlaps a coordinate that was included in fitting.
    #[error("missing interval {interval} contains observed training coordinate {coordinate}")]
    MissingIntervalContainsObserved { interval: usize, coordinate: f64 },
    /// Requested missing intervals overlap and would duplicate reconstructed samples.
    #[error("missing intervals {first} and {second} overlap")]
    OverlappingMissingIntervals { first: usize, second: usize },
    /// An explicit extrapolation request included a point inside the training domain.
    #[error(
        "coordinate {coordinate} at index {index} is inside the training domain [{min}, {max}], not outside it"
    )]
    CoordinateInsideTrainingDomain {
        index: usize,
        coordinate: f64,
        min: f64,
        max: f64,
    },
}

impl FittedModel {
    pub(crate) fn from_parameterized_expressions(
        equations: &[Expr],
        independent_variable: impl Into<String>,
        parameters: &HashMap<String, f64>,
        observed_x: Vec<f64>,
    ) -> Result<Self, FittedModelError> {
        let equations = equations
            .iter()
            .map(|equation| equation.clone().set_variable_from_map(parameters))
            .collect();
        Self::from_bound_expressions(equations, independent_variable, observed_x)
    }

    /// Construct an evaluator from expressions with fitted parameters already substituted.
    pub(crate) fn from_bound_expressions(
        equations: Vec<Expr>,
        independent_variable: impl Into<String>,
        observed_x: Vec<f64>,
    ) -> Result<Self, FittedModelError> {
        let independent_variable = independent_variable.into();
        if equations.is_empty() || observed_x.is_empty() {
            return Err(FittedModelError::EmptyModel);
        }
        if independent_variable.trim().is_empty() {
            return Err(FittedModelError::MissingIndependentVariable);
        }
        if let Some(index) = observed_x.iter().position(|value| !value.is_finite()) {
            return Err(FittedModelError::NonFiniteTrainingCoordinate { index });
        }
        Ok(Self(Arc::new(FittedModelInner {
            equations,
            independent_variable,
            observed_x,
            evaluators: OnceLock::new(),
        })))
    }

    /// Evaluate the fitted model at arbitrary coordinates, including interpolation and
    /// extrapolation. Requested coordinate order is preserved.
    pub fn evaluate_at(&self, x: &[f64]) -> Result<FittedValues, FittedModelError> {
        validate_coordinates(x)?;
        let evaluators = self.evaluators();
        let output_len = checked_output_len(x.len(), evaluators.len())?;
        let mut row_major_values = Vec::new();
        row_major_values
            .try_reserve_exact(output_len)
            .map_err(|_| FittedModelError::AllocationFailed {
                requested: output_len,
            })?;
        for (sample, &coordinate) in x.iter().enumerate() {
            for (equation, evaluate) in evaluators.iter().enumerate() {
                let value = evaluate(coordinate);
                if !value.is_finite() {
                    return Err(FittedModelError::NonFinitePrediction { sample, equation });
                }
                row_major_values.push(value);
            }
        }
        let values = DMatrix::from_row_slice(x.len(), evaluators.len(), &row_major_values);
        Ok(FittedValues {
            x: x.to_vec(),
            values,
        })
    }

    /// Evaluate a strictly increasing uniform range with both endpoints included.
    pub fn evaluate_uniform_range(
        &self,
        start: f64,
        end: f64,
        total_points: usize,
    ) -> Result<FittedValues, FittedModelError> {
        let x = uniform_coordinates(start, end, total_points)?;
        self.evaluate_at(&x)
    }

    /// Evaluate only coordinates strictly outside the observed training-data domain.
    pub fn extrapolate_at(&self, x: &[f64]) -> Result<FittedValues, FittedModelError> {
        validate_coordinates(x)?;
        let (min, max) = self.training_domain();
        if let Some((index, &coordinate)) = x
            .iter()
            .enumerate()
            .find(|(_, coordinate)| **coordinate >= min && **coordinate <= max)
        {
            return Err(FittedModelError::CoordinateInsideTrainingDomain {
                index,
                coordinate,
                min,
                max,
            });
        }
        self.evaluate_at(x)
    }

    /// Evaluate a uniform range lying entirely to one side of the training-data domain.
    pub fn extrapolate_uniform_range(
        &self,
        start: f64,
        end: f64,
        total_points: usize,
    ) -> Result<FittedValues, FittedModelError> {
        let x = uniform_coordinates(start, end, total_points)?;
        self.extrapolate_at(&x)
    }

    /// Reconstruct interior samples in one or more explicitly supplied internal gaps.
    pub fn fill_missing_intervals(
        &self,
        intervals: &[MissingInterval],
    ) -> Result<Vec<FittedValues>, FittedModelError> {
        if intervals.is_empty() {
            return Err(FittedModelError::EmptyMissingIntervals);
        }
        let (domain_min, domain_max) = self.training_domain();
        for (index, interval) in intervals.iter().enumerate() {
            if !interval.start.is_finite()
                || !interval.end.is_finite()
                || interval.start >= interval.end
                || interval.missing_points == 0
            {
                return Err(FittedModelError::InvalidMissingInterval { index });
            }
            if interval.start < domain_min || interval.end > domain_max {
                return Err(FittedModelError::MissingIntervalOutsideDomain { index });
            }
            if let Some(&coordinate) = self
                .0
                .observed_x
                .iter()
                .find(|&&x| x > interval.start && x < interval.end)
            {
                return Err(FittedModelError::MissingIntervalContainsObserved {
                    interval: index,
                    coordinate,
                });
            }
        }
        for first in 0..intervals.len() {
            for second in first + 1..intervals.len() {
                if intervals[first].start < intervals[second].end
                    && intervals[second].start < intervals[first].end
                {
                    return Err(FittedModelError::OverlappingMissingIntervals { first, second });
                }
            }
        }

        intervals
            .iter()
            .enumerate()
            .map(|(index, interval)| {
                let subdivisions = interval
                    .missing_points
                    .checked_add(1)
                    .ok_or(FittedModelError::MissingPointCountOverflow { index })?;
                let x = interior_coordinates(
                    interval.start,
                    interval.end,
                    interval.missing_points,
                    subdivisions,
                    index,
                )?;
                self.evaluate_at(&x)
            })
            .collect()
    }

    /// Return the minimum and maximum observed training coordinates.
    pub fn training_domain(&self) -> (f64, f64) {
        self.0
            .observed_x
            .iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
                (min.min(x), max.max(x))
            })
    }

    fn evaluators(&self) -> &[ScalarEvaluator] {
        self.0.evaluators.get_or_init(|| {
            let independent_variable = self.0.independent_variable.as_str();
            self.0
                .equations
                .iter()
                .map(|equation| {
                    let evaluator = equation
                        .clone()
                        .lambdify_borrowed_thread_safe(&[independent_variable]);
                    Arc::new(move |x| evaluator(&[x])) as ScalarEvaluator
                })
                .collect()
        })
    }
}

/// Backend result paired with its reusable fitted-model evaluator.
pub struct FittedBackend<T> {
    backend: T,
    fitted_model: FittedModel,
}

impl<T> FittedBackend<T> {
    pub(crate) fn new(backend: T, fitted_model: FittedModel) -> Self {
        Self {
            backend,
            fitted_model,
        }
    }

    /// Borrow the original backend-specific fit result.
    pub fn backend(&self) -> &T {
        &self.backend
    }

    /// Mutably borrow the backend-specific fit result.
    pub fn backend_mut(&mut self) -> &mut T {
        &mut self.backend
    }

    /// Consume the wrapper and return the backend-specific fit result.
    pub fn into_backend(self) -> T {
        self.backend
    }

    /// Borrow the shared fitted-model evaluator.
    pub fn fitted_model(&self) -> &FittedModel {
        &self.fitted_model
    }

    /// Evaluate this fitted result at arbitrary coordinates.
    pub fn evaluate_at(&self, x: &[f64]) -> Result<FittedValues, FittedModelError> {
        self.fitted_model.evaluate_at(x)
    }

    /// Evaluate a uniform range including both endpoints.
    pub fn evaluate_uniform_range(
        &self,
        start: f64,
        end: f64,
        total_points: usize,
    ) -> Result<FittedValues, FittedModelError> {
        self.fitted_model
            .evaluate_uniform_range(start, end, total_points)
    }

    /// Evaluate coordinates outside the training-data domain.
    pub fn extrapolate_at(&self, x: &[f64]) -> Result<FittedValues, FittedModelError> {
        self.fitted_model.extrapolate_at(x)
    }

    /// Evaluate a uniform range entirely outside the training-data domain.
    pub fn extrapolate_uniform_range(
        &self,
        start: f64,
        end: f64,
        total_points: usize,
    ) -> Result<FittedValues, FittedModelError> {
        self.fitted_model
            .extrapolate_uniform_range(start, end, total_points)
    }

    /// Reconstruct strictly interior samples for explicit gaps.
    pub fn fill_missing_intervals(
        &self,
        intervals: &[MissingInterval],
    ) -> Result<Vec<FittedValues>, FittedModelError> {
        self.fitted_model.fill_missing_intervals(intervals)
    }
}

impl<T> Deref for FittedBackend<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

impl<T> DerefMut for FittedBackend<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.backend
    }
}

fn validate_coordinates(x: &[f64]) -> Result<(), FittedModelError> {
    if x.is_empty() {
        return Err(FittedModelError::EmptyCoordinates);
    }
    if let Some(index) = x.iter().position(|coordinate| !coordinate.is_finite()) {
        return Err(FittedModelError::NonFiniteCoordinate { index });
    }
    Ok(())
}

fn checked_output_len(samples: usize, equations: usize) -> Result<usize, FittedModelError> {
    samples
        .checked_mul(equations)
        .ok_or(FittedModelError::OutputSizeOverflow)
}

fn uniform_coordinates(
    start: f64,
    end: f64,
    total_points: usize,
) -> Result<Vec<f64>, FittedModelError> {
    if !start.is_finite() || !end.is_finite() || start >= end {
        return Err(FittedModelError::InvalidUniformRange);
    }
    if total_points < 2 {
        return Err(FittedModelError::TooFewUniformPoints {
            actual: total_points,
        });
    }
    let mut x = Vec::new();
    x.try_reserve_exact(total_points)
        .map_err(|_| FittedModelError::AllocationFailed {
            requested: total_points,
        })?;
    let denominator = (total_points - 1) as f64;
    for index in 0..total_points {
        let fraction = index as f64 / denominator;
        let coordinate = interpolate(start, end, fraction);
        if !coordinate.is_finite() || x.last().is_some_and(|previous| *previous >= coordinate) {
            return Err(FittedModelError::UnrepresentableUniformGrid);
        }
        x.push(coordinate);
    }
    x[0] = start;
    x[total_points - 1] = end;
    Ok(x)
}

fn interior_coordinates(
    start: f64,
    end: f64,
    point_count: usize,
    subdivisions: usize,
    interval_index: usize,
) -> Result<Vec<f64>, FittedModelError> {
    let mut coordinates = Vec::new();
    coordinates
        .try_reserve_exact(point_count)
        .map_err(|_| FittedModelError::AllocationFailed {
            requested: point_count,
        })?;
    for point in 1..=point_count {
        let fraction = point as f64 / subdivisions as f64;
        let coordinate = interpolate(start, end, fraction);
        if !coordinate.is_finite()
            || coordinate <= start
            || coordinate >= end
            || coordinates
                .last()
                .is_some_and(|previous| *previous >= coordinate)
        {
            return Err(FittedModelError::UnrepresentableMissingInterval {
                index: interval_index,
            });
        }
        coordinates.push(coordinate);
    }
    Ok(coordinates)
}

fn interpolate(start: f64, end: f64, fraction: f64) -> f64 {
    let width = end - start;
    if width.is_finite() {
        start + width * fraction
    } else {
        start * (1.0 - fraction) + end * fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn linear_model() -> FittedModel {
        FittedModel::from_bound_expressions(
            vec![Expr::parse_expression("2*x+1")],
            "x",
            vec![0.0, 1.0, 3.0, 4.0],
        )
        .unwrap()
    }

    #[test]
    fn evaluates_in_request_order_and_uniform_range_includes_endpoints() {
        let model = linear_model();
        let evaluated = model.evaluate_at(&[4.0, -1.0, 2.0]).unwrap();
        assert_eq!(evaluated.values.shape(), (3, 1));
        assert_eq!(evaluated.x, vec![4.0, -1.0, 2.0]);
        assert_relative_eq!(evaluated.values[(0, 0)], 9.0, epsilon = 1e-12);
        assert_relative_eq!(evaluated.values[(1, 0)], -1.0, epsilon = 1e-12);
        assert_relative_eq!(evaluated.values[(2, 0)], 5.0, epsilon = 1e-12);

        let grid = model.evaluate_uniform_range(-1.0, 1.0, 3).unwrap();
        assert_eq!(grid.x, vec![-1.0, 0.0, 1.0]);
        assert_relative_eq!(grid.values[(2, 0)], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn explicit_extrapolation_and_gap_fill_enforce_their_domains() {
        let model = linear_model();
        let extrapolated = model.extrapolate_at(&[-2.0, 5.0]).unwrap();
        assert_relative_eq!(extrapolated.values[(0, 0)], -3.0, epsilon = 1e-12);
        assert_relative_eq!(extrapolated.values[(1, 0)], 11.0, epsilon = 1e-12);
        assert!(matches!(
            model.extrapolate_at(&[2.0]),
            Err(FittedModelError::CoordinateInsideTrainingDomain { .. })
        ));

        let gaps = model
            .fill_missing_intervals(&[MissingInterval {
                start: 1.0,
                end: 3.0,
                missing_points: 2,
            }])
            .unwrap();
        assert_eq!(gaps[0].x.len(), 2);
        assert_relative_eq!(gaps[0].x[0], 1.0 + 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(
            gaps[0].values[(1, 0)],
            1.0 + 2.0 * 7.0 / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn invalid_grids_gaps_and_nonfinite_predictions_are_typed() {
        let model = linear_model();
        assert!(matches!(
            model.evaluate_at(&[]),
            Err(FittedModelError::EmptyCoordinates)
        ));
        assert!(matches!(
            model.evaluate_at(&[f64::NAN]),
            Err(FittedModelError::NonFiniteCoordinate { index: 0 })
        ));
        assert!(matches!(
            model.evaluate_uniform_range(0.0, 1.0, 1),
            Err(FittedModelError::TooFewUniformPoints { actual: 1 })
        ));
        assert!(matches!(
            model.evaluate_uniform_range(1.0e16, 1.0e16 + 2.0, 3),
            Err(FittedModelError::UnrepresentableUniformGrid)
        ));
        assert!(matches!(
            model.fill_missing_intervals(&[MissingInterval {
                start: 1.0,
                end: 3.0,
                missing_points: usize::MAX,
            }]),
            Err(FittedModelError::MissingPointCountOverflow { index: 0 })
        ));
        assert!(matches!(
            model.fill_missing_intervals(&[MissingInterval {
                start: 0.0,
                end: 4.0,
                missing_points: 2,
            }]),
            Err(FittedModelError::MissingIntervalContainsObserved { .. })
        ));

        let singular = FittedModel::from_bound_expressions(
            vec![Expr::parse_expression("1/x")],
            "x",
            vec![-1.0, 1.0],
        )
        .unwrap();
        assert!(matches!(
            singular.evaluate_at(&[0.0]),
            Err(FittedModelError::NonFinitePrediction {
                sample: 0,
                equation: 0
            })
        ));
    }

    #[test]
    fn output_dimensions_are_checked_before_allocation() {
        assert_eq!(usize::MAX.checked_mul(2), None);
        assert!(matches!(
            checked_output_len(usize::MAX, 2),
            Err(FittedModelError::OutputSizeOverflow)
        ));
    }
}
