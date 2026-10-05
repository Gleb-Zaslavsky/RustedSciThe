//! Output collection policies for an adaptive Radau solve.

use super::dense_output::{RadauDenseOutputHistory, RadauDenseOutputSegment};
use super::error::RadauError;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RadauOutputPolicy {
    /// Retain only the final state and do not allocate trajectory storage.
    FinalOnly,
    /// Retain every accepted cubic segment for continuous interpolation.
    Dense,
    /// Retain requested samples evaluated against accepted segments.
    Sampled(Vec<f64>),
}

impl Default for RadauOutputPolicy {
    fn default() -> Self {
        Self::FinalOnly
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RadauOutput {
    FinalOnly,
    Dense(RadauDenseOutputHistory),
    Sampled {
        times: Vec<f64>,
        values: Vec<f64>,
        dimension: usize,
    },
}

impl RadauOutput {
    pub(crate) fn sample_into(&self, t: f64, output: &mut [f64]) -> Result<(), RadauError> {
        match self {
            Self::FinalOnly => Err(RadauError::OutputNotAvailable),
            Self::Dense(history) => history.evaluate_into(t, output),
            Self::Sampled {
                times,
                values,
                dimension,
            } => {
                if output.len() != *dimension {
                    return Err(RadauError::ShapeMismatch {
                        stage: super::error::RadauStage::Output,
                        expected: *dimension,
                        actual: output.len(),
                    });
                }
                let index = times
                    .iter()
                    .position(|candidate| *candidate == t)
                    .ok_or(RadauError::OutputNotAvailable)?;
                output.copy_from_slice(&values[index * dimension..(index + 1) * dimension]);
                Ok(())
            }
        }
    }

    pub(crate) fn sample_many(
        &self,
        times: &[f64],
        dimension: usize,
    ) -> Result<Vec<f64>, RadauError> {
        match self {
            Self::FinalOnly => Err(RadauError::OutputNotAvailable),
            Self::Dense(history) => history.sample(times),
            Self::Sampled { .. } => {
                let mut values = vec![0.0; times.len().saturating_mul(dimension)];
                for (index, &time) in times.iter().enumerate() {
                    self.sample_into(
                        time,
                        &mut values[index * dimension..(index + 1) * dimension],
                    )?;
                }
                Ok(values)
            }
        }
    }
}

pub(crate) struct RadauOutputCollector {
    policy: RadauOutputPolicy,
    dimension: usize,
    history: Option<RadauDenseOutputHistory>,
}

impl RadauOutputCollector {
    pub(crate) fn new(policy: RadauOutputPolicy, dimension: usize) -> Result<Self, RadauError> {
        if dimension == 0 {
            return Err(RadauError::InvalidDenseOutput);
        }
        if let RadauOutputPolicy::Sampled(times) = &policy {
            if times.iter().any(|time| !time.is_finite()) {
                return Err(RadauError::InvalidDenseOutput);
            }
        }
        let history = match policy {
            RadauOutputPolicy::FinalOnly => None,
            RadauOutputPolicy::Dense | RadauOutputPolicy::Sampled(_) => {
                Some(RadauDenseOutputHistory::new(dimension))
            }
        };
        Ok(Self {
            policy,
            dimension,
            history,
        })
    }

    pub(crate) fn push(&mut self, segment: RadauDenseOutputSegment) -> Result<(), RadauError> {
        if let Some(history) = &mut self.history {
            history.push(segment)?;
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<RadauOutput, RadauError> {
        match self.policy {
            RadauOutputPolicy::FinalOnly => Ok(RadauOutput::FinalOnly),
            RadauOutputPolicy::Dense => Ok(RadauOutput::Dense(
                self.history.ok_or(RadauError::InvalidDenseOutput)?,
            )),
            RadauOutputPolicy::Sampled(times) => {
                let history = self.history.ok_or(RadauError::InvalidDenseOutput)?;
                let values = history.sample(&times)?;
                Ok(RadauOutput::Sampled {
                    times,
                    values,
                    dimension: self.dimension,
                })
            }
        }
    }
}
