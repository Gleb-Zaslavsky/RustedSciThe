//! Cubic Radau continuous-output interpolation.
//!
//! A segment owns the state at the beginning of an accepted step and the
//! three cubic coefficients produced by the collocation step.  Keeping the
//! segment self-contained makes output evaluation independent from the
//! mutable Newton workspace and prevents a later retry from changing an
//! already accepted trajectory.

use super::error::RadauError;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RadauDenseOutputSegment {
    t_old: f64,
    h: f64,
    y_old: Vec<f64>,
    q: Vec<f64>,
}

impl RadauDenseOutputSegment {
    pub(crate) fn new(
        t_old: f64,
        h: f64,
        y_old: Vec<f64>,
        q: Vec<f64>,
    ) -> Result<Self, RadauError> {
        let dimension = y_old.len();
        if dimension == 0
            || !t_old.is_finite()
            || !h.is_finite()
            || h == 0.0
            || q.len() != dimension.saturating_mul(3)
            || y_old.iter().any(|value| !value.is_finite())
            || q.iter().any(|value| !value.is_finite())
        {
            return Err(RadauError::InvalidDenseOutput);
        }
        Ok(Self { t_old, h, y_old, q })
    }

    pub(crate) fn interval(&self) -> (f64, f64) {
        (self.t_old, self.t_old + self.h)
    }

    pub(crate) fn dimension(&self) -> usize {
        self.y_old.len()
    }

    pub(crate) fn evaluate_into(&self, t: f64, output: &mut [f64]) -> Result<(), RadauError> {
        if output.len() != self.dimension() {
            return Err(RadauError::ShapeMismatch {
                stage: super::error::RadauStage::Output,
                expected: self.dimension(),
                actual: output.len(),
            });
        }
        let (start, end) = self.interval();
        let lo = start.min(end);
        let hi = start.max(end);
        if !t.is_finite() || t < lo || t > hi {
            return Err(RadauError::OutputTimeOutsideInterval { t });
        }
        let x = (t - self.t_old) / self.h;
        let x2 = x * x;
        let x3 = x2 * x;
        for (index, value) in output.iter_mut().enumerate() {
            *value = self.y_old[index]
                + self.q[index] * x
                + self.q[self.dimension() + index] * x2
                + self.q[2 * self.dimension() + index] * x3;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RadauDenseOutputHistory {
    dimension: usize,
    segments: Vec<RadauDenseOutputSegment>,
}

impl RadauDenseOutputHistory {
    pub(crate) fn new(dimension: usize) -> Self {
        Self {
            dimension,
            segments: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, segment: RadauDenseOutputSegment) -> Result<(), RadauError> {
        if segment.dimension() != self.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: super::error::RadauStage::Output,
                expected: self.dimension,
                actual: segment.dimension(),
            });
        }
        self.segments.push(segment);
        Ok(())
    }

    pub(crate) fn segments(&self) -> &[RadauDenseOutputSegment] {
        &self.segments
    }

    pub(crate) fn evaluate_into(&self, t: f64, output: &mut [f64]) -> Result<(), RadauError> {
        if output.len() != self.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: super::error::RadauStage::Output,
                expected: self.dimension,
                actual: output.len(),
            });
        }
        let segment = self
            .segments
            .iter()
            .find(|segment| {
                let (start, end) = segment.interval();
                t >= start.min(end) && t <= start.max(end)
            })
            .ok_or(RadauError::OutputTimeOutsideInterval { t })?;
        segment.evaluate_into(t, output)
    }

    pub(crate) fn sample(&self, times: &[f64]) -> Result<Vec<f64>, RadauError> {
        let mut values = vec![0.0; times.len().saturating_mul(self.dimension)];
        for (row, &time) in times.iter().enumerate() {
            self.evaluate_into(
                time,
                &mut values[row * self.dimension..(row + 1) * self.dimension],
            )?;
        }
        Ok(values)
    }
}
