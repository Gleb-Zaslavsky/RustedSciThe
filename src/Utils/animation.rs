//! Shared validation for trajectory animation inputs.
//!
//! Validation is deliberately separate from Bevy startup so malformed arrays
//! fail before a window is created or matrix indexing begins.

use std::fmt;

use nalgebra::{DMatrix, DVector};

/// Invalid trajectory data or playback configuration for an animation helper.
#[derive(Debug, Clone, PartialEq)]
pub enum AnimationInputError {
    /// The point matrix does not have the requested coordinate dimension.
    InvalidDimension { expected: usize, actual: usize },
    /// A trajectory needs at least one point.
    EmptyTrajectory,
    /// Every point must have a matching timestamp.
    SampleCountMismatch { points: usize, times: usize },
    /// A point coordinate is not finite or cannot be represented by Bevy's f32 data.
    InvalidCoordinate { row: usize, column: usize },
    /// The coordinate range on an axis overflows the renderer's f32 geometry.
    InvalidCoordinateRange { axis: usize },
    /// A timestamp is not finite.
    NonFiniteTime { index: usize },
    /// Timestamps must be strictly monotone, either forward or backward.
    NonMonotoneTime { index: usize },
    /// A time interval is too large for a safe visualization timer.
    UnrepresentableTimeInterval { index: usize },
    /// Playback speed must be finite and positive.
    InvalidSpeed,
}

impl fmt::Display for AnimationInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { expected, actual } => {
                write!(f, "trajectory must have {expected} coordinate rows, got {actual}")
            }
            Self::EmptyTrajectory => write!(f, "trajectory must contain at least one point"),
            Self::SampleCountMismatch { points, times } => write!(
                f,
                "trajectory has {points} points but {times} timestamps"
            ),
            Self::InvalidCoordinate { row, column } => write!(
                f,
                "coordinate at row {row}, sample {column} is non-finite or outside f32 range"
            ),
            Self::InvalidCoordinateRange { axis } => write!(
                f,
                "coordinate range on axis {axis} is outside finite f32 range"
            ),
            Self::NonFiniteTime { index } => {
                write!(f, "timestamp at index {index} is not finite")
            }
            Self::NonMonotoneTime { index } => write!(
                f,
                "timestamps must be strictly monotone; order changes or repeats at index {index}"
            ),
            Self::UnrepresentableTimeInterval { index } => write!(
                f,
                "time interval ending at index {index} is too large for playback"
            ),
            Self::InvalidSpeed => write!(f, "animation speed must be finite and greater than zero"),
        }
    }
}

impl std::error::Error for AnimationInputError {}

/// Validate shape, numeric range and time ordering, returning per-sample frame
/// durations in seconds after applying the playback speed.
///
/// The final point uses the last observed interval when playback loops. A
/// one-point trajectory uses a short display interval. Sub-microsecond samples
/// are clamped because Bevy playback is a visualization, not a numerical clock.
pub fn validate_trajectory<const D: usize>(
    points: &DMatrix<f64>,
    times: &DVector<f64>,
    speed: f32,
) -> Result<Vec<f32>, AnimationInputError> {
    if points.nrows() != D {
        return Err(AnimationInputError::InvalidDimension {
            expected: D,
            actual: points.nrows(),
        });
    }
    if points.ncols() == 0 {
        return Err(AnimationInputError::EmptyTrajectory);
    }
    if points.ncols() != times.len() {
        return Err(AnimationInputError::SampleCountMismatch {
            points: points.ncols(),
            times: times.len(),
        });
    }
    if !speed.is_finite() || speed <= 0.0 {
        return Err(AnimationInputError::InvalidSpeed);
    }

    for column in 0..points.ncols() {
        for row in 0..D {
            let value = points[(row, column)];
            if !value.is_finite() || !(value as f32).is_finite() {
                return Err(AnimationInputError::InvalidCoordinate { row, column });
            }
        }
    }
    for row in 0..D {
        let (min, max) = (0..points.ncols()).fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), column| {
                let value = points[(row, column)];
                (min.min(value), max.max(value))
            },
        );
        if !((max - min) as f32).is_finite() {
            return Err(AnimationInputError::InvalidCoordinateRange { axis: row });
        }
    }
    for (index, time) in times.iter().enumerate() {
        if !time.is_finite() {
            return Err(AnimationInputError::NonFiniteTime { index });
        }
    }

    let n = times.len();
    if n > 1 {
        let first_delta = times[1] - times[0];
        if first_delta == 0.0 {
            return Err(AnimationInputError::NonMonotoneTime { index: 1 });
        }
        let forward = first_delta > 0.0;
        for index in 1..n {
            let delta = times[index] - times[index - 1];
            if (forward && delta <= 0.0) || (!forward && delta >= 0.0) {
                return Err(AnimationInputError::NonMonotoneTime { index });
            }
        }
    }

    let mut intervals = Vec::with_capacity(n);
    for index in 0..n {
        let raw_delta = if n == 1 {
            0.05
        } else if index + 1 < n {
            (times[index + 1] - times[index]).abs()
        } else {
            (times[index] - times[index - 1]).abs()
        };
        let seconds = raw_delta / f64::from(speed);
        // Keep Duration conversion far from its u64-second limit; such an
        // interval would also make an interactive preview appear frozen.
        if !seconds.is_finite() || seconds > 1.0e9 {
            return Err(AnimationInputError::UnrepresentableTimeInterval { index });
        }
        intervals.push((seconds as f32).max(1.0e-6));
    }
    Ok(intervals)
}

/// Advance a frame cursor by elapsed wall time without losing render-frame
/// overshoot. The returned index is the sample active at the new playback
/// time; display may naturally skip samples when several fit in one frame.
pub(crate) fn advance_animation_clock(
    current_index: usize,
    elapsed_in_interval: f64,
    frame_delta: f64,
    intervals: &[f32],
) -> (usize, f64) {
    debug_assert!(!intervals.is_empty());
    debug_assert!(current_index < intervals.len());
    debug_assert!(elapsed_in_interval.is_finite() && elapsed_in_interval >= 0.0);
    debug_assert!(frame_delta.is_finite() && frame_delta >= 0.0);

    let cycle_seconds: f64 = intervals.iter().map(|&value| f64::from(value)).sum();
    let mut remaining = (elapsed_in_interval + frame_delta) % cycle_seconds;
    let mut index = current_index;
    for _ in 0..intervals.len() {
        let interval = f64::from(intervals[index]);
        if remaining < interval {
            return (index, remaining);
        }
        remaining -= interval;
        index = (index + 1) % intervals.len();
    }

    // Rounding at the exact cycle boundary can leave a tiny residual.
    (index, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_forward_backward_and_single_sample_timing() {
        let points = DMatrix::from_row_slice(2, 3, &[0.0, 1.0, 2.0, 0.0, 1.0, 0.0]);
        let forward = DVector::from_vec(vec![0.0, 0.25, 1.0]);
        assert_eq!(validate_trajectory::<2>(&points, &forward, 2.0).unwrap(), vec![0.125, 0.375, 0.375]);

        let backward = DVector::from_vec(vec![1.0, 0.25, 0.0]);
        assert_eq!(validate_trajectory::<2>(&points, &backward, 1.0).unwrap(), vec![0.75, 0.25, 0.25]);

        let one_point = DMatrix::from_row_slice(2, 1, &[1.0, 2.0]);
        assert_eq!(
            validate_trajectory::<2>(&one_point, &DVector::from_vec(vec![3.0]), 1.0).unwrap(),
            vec![0.05]
        );
    }

    #[test]
    fn playback_clock_preserves_overshoot_and_wraps_irregular_samples() {
        let intervals = [0.125, 0.25, 0.125];
        assert_eq!(advance_animation_clock(0, 0.0, 0.1875, &intervals), (1, 0.0625));
        assert_eq!(advance_animation_clock(2, 0.0, 0.25, &intervals), (1, 0.0));
        assert_eq!(advance_animation_clock(0, 0.0, 0.6875, &intervals), (1, 0.0625));
    }

    #[test]
    fn rejects_unsafe_shapes_times_coordinates_and_speed() {
        let points = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 1.0]);
        let times = DVector::from_vec(vec![0.0, 1.0]);
        assert!(matches!(
            validate_trajectory::<3>(&points, &times, 1.0),
            Err(AnimationInputError::InvalidDimension { .. })
        ));
        assert!(matches!(
            validate_trajectory::<2>(&points, &DVector::from_vec(vec![0.0]), 1.0),
            Err(AnimationInputError::SampleCountMismatch { .. })
        ));
        assert!(matches!(
            validate_trajectory::<2>(&points, &times, f32::NAN),
            Err(AnimationInputError::InvalidSpeed)
        ));
        assert!(matches!(
            validate_trajectory::<2>(
                &DMatrix::from_row_slice(2, 1, &[f64::INFINITY, 0.0]),
                &DVector::from_vec(vec![0.0]),
                1.0
            ),
            Err(AnimationInputError::InvalidCoordinate { .. })
        ));
        assert!(matches!(
            validate_trajectory::<2>(
                &points,
                &DVector::from_vec(vec![0.0, 0.0]),
                1.0
            ),
            Err(AnimationInputError::NonMonotoneTime { .. })
        ));
        assert!(matches!(
            validate_trajectory::<2>(
                &points,
                &DVector::from_vec(vec![0.0, 2.0e9]),
                1.0
            ),
            Err(AnimationInputError::UnrepresentableTimeInterval { .. })
        ));
        assert!(matches!(
            validate_trajectory::<2>(
                &DMatrix::from_row_slice(
                    2,
                    2,
                    &[f64::from(f32::MAX), -f64::from(f32::MAX), 0.0, 0.0]
                ),
                &times,
                1.0
            ),
            Err(AnimationInputError::InvalidCoordinateRange { axis: 0 })
        ));
    }
}
