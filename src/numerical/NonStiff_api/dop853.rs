//! Adaptive explicit DOP853 integrator.
//!
//! The method coefficients follow the published DOP853 tableau also used by
//! SciPy's `integrate._ivp.dop853_coefficients` (BSD-3-Clause, SciPy 1.18.1).
//! This implementation intentionally exposes accepted mesh points only; dense
//! output is a separate feature and is not fabricated by interpolation here.

use crate::symbolic::ivp_telemetry::{IvpTelemetry, IvpWarmStage};
use nalgebra::DVector;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Instant;

const N: usize = 12;
const A: [[f64; N]; N] = [
    [0.0; N],
    [
        0.05260015195876773,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0197250569845379,
        0.059175170953613701,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.029587585476806851,
        0.0,
        0.088762756430420545,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.24136513415926669,
        0.0,
        -0.88454947932828609,
        0.92483400326179199,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.037037037037037035,
        0.0,
        0.0,
        0.17082860872947386,
        0.12546768756682242,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.037109375,
        0.0,
        0.0,
        0.17025221101954405,
        0.060216538980455959,
        -0.017578125,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.037092000118504789,
        0.0,
        0.0,
        0.17038392571223998,
        0.10726203044637328,
        -0.015319437748624402,
        0.0082737891638140233,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.62411095871607569,
        0.0,
        0.0,
        -3.3608926294469414,
        -0.86821934684172597,
        27.59209969944671,
        20.154067550477894,
        -43.489884181069961,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.47766253643826434,
        0.0,
        0.0,
        -2.4881146199716677,
        -0.59029082683684297,
        21.230051448181193,
        15.279233632882423,
        -33.288210968984863,
        -0.020331201708508627,
        0.0,
        0.0,
        0.0,
    ],
    [
        -0.9371424300859873,
        0.0,
        0.0,
        5.1863724288440638,
        1.0914373489967295,
        -8.1497870107469268,
        -18.520065659996959,
        22.739487099350505,
        2.4936055526796523,
        -3.0467644718982196,
        0.0,
        0.0,
    ],
    [
        2.273310147516538,
        0.0,
        0.0,
        -10.534495466737249,
        -2.0008720582248625,
        -17.958931863118799,
        27.94888452941996,
        -2.8589982771350235,
        -8.8728569335306293,
        12.360567175794303,
        0.64339274601576357,
        0.0,
    ],
];
const B: [f64; N] = [
    0.054293734116568765,
    0.0,
    0.0,
    0.0,
    0.0,
    4.4503128927524092,
    1.8915178993145003,
    -5.8012039600105849,
    0.3111643669578199,
    -0.15216094966251609,
    0.20136540080403034,
    0.044710615727772587,
];
const E3: [f64; N + 1] = [
    -0.18980075407240762,
    0.0,
    0.0,
    0.0,
    0.0,
    4.4503128927524092,
    1.8915178993145003,
    -5.8012039600105849,
    -0.42268232132379191,
    -0.15216094966251609,
    0.20136540080403034,
    0.022651792198360821,
    0.0,
];
const E5: [f64; N + 1] = [
    0.01312004499419488,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.2251564463762044,
    -0.4957589496572502,
    1.6643771824549864,
    -0.35032884874997366,
    0.33417911871301748,
    0.08192320648511571,
    -0.022355307863886294,
    0.0,
];

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const ERROR_EXPONENT: f64 = -1.0 / 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dop853Options {
    pub rtol: f64,
    pub atol: f64,
    /// Maximum absolute step. Positive infinity means no user-imposed cap.
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub max_steps: usize,
}

impl Default for Dop853Options {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-10,
            max_step: f64::INFINITY,
            first_step: None,
            max_steps: 1_000_000,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dop853Statistics {
    pub function_evaluations: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub attempted_steps: usize,
    pub minimum_accepted_step: Option<f64>,
    pub maximum_accepted_step: Option<f64>,
    pub solve_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dop853Solution {
    pub times: Vec<f64>,
    /// One state vector per time sample.
    pub states: Vec<Vec<f64>>,
    pub status: Dop853Status,
    pub statistics: Dop853Statistics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dop853Status {
    Finished,
    StoppedByCondition,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Dop853Error {
    InvalidConfiguration(&'static str),
    EmptyState,
    NonFiniteInitialState {
        index: usize,
    },
    RhsDimension {
        expected: usize,
        actual: usize,
        evaluation: usize,
    },
    NonFiniteRhs {
        index: usize,
        evaluation: usize,
    },
    StepSizeUnderflow {
        time: f64,
        step: f64,
    },
    MaximumStepsExceeded {
        limit: usize,
        time: f64,
    },
    NonFiniteErrorEstimate {
        time: f64,
    },
}

impl Display for Dop853Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "invalid DOP853 configuration: {message}")
            }
            Self::EmptyState => f.write_str("DOP853 requires at least one state variable"),
            Self::NonFiniteInitialState { index } => {
                write!(f, "DOP853 initial state at index {index} is not finite")
            }
            Self::RhsDimension {
                expected,
                actual,
                evaluation,
            } => write!(
                f,
                "DOP853 RHS evaluation {evaluation} returned dimension {actual}, expected {expected}"
            ),
            Self::NonFiniteRhs { index, evaluation } => write!(
                f,
                "DOP853 RHS evaluation {evaluation} returned a non-finite value at index {index}"
            ),
            Self::StepSizeUnderflow { time, step } => {
                write!(f, "DOP853 step size {step:e} is too small at t={time:e}")
            }
            Self::MaximumStepsExceeded { limit, time } => write!(
                f,
                "DOP853 exceeded the maximum of {limit} accepted steps at t={time:e}"
            ),
            Self::NonFiniteErrorEstimate { time } => write!(
                f,
                "DOP853 produced a non-finite error estimate at t={time:e}"
            ),
        }
    }
}

impl Error for Dop853Error {}

pub struct Dop853 {
    rhs: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    y0: DVector<f64>,
    t0: f64,
    t_bound: f64,
    options: Dop853Options,
    telemetry: IvpTelemetry,
    statistics: Dop853Statistics,
}

impl Dop853 {
    pub fn new(
        rhs: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        y0: DVector<f64>,
        t0: f64,
        t_bound: f64,
        options: Dop853Options,
        telemetry: IvpTelemetry,
    ) -> Result<Self, Dop853Error> {
        validate_options(options, t0, t_bound)?;
        if y0.is_empty() {
            return Err(Dop853Error::EmptyState);
        }
        if let Some(index) = y0.iter().position(|value| !value.is_finite()) {
            return Err(Dop853Error::NonFiniteInitialState { index });
        }
        Ok(Self {
            rhs,
            y0,
            t0,
            t_bound,
            options,
            telemetry,
            statistics: Dop853Statistics::default(),
        })
    }

    pub fn statistics(&self) -> &Dop853Statistics {
        &self.statistics
    }

    pub fn solve(
        &mut self,
        stop_condition: Option<&dyn Fn(&DVector<f64>) -> bool>,
    ) -> Result<Dop853Solution, Dop853Error> {
        let started = Some(Instant::now());
        let direction = (self.t_bound - self.t0).signum();
        let interval = (self.t_bound - self.t0).abs();
        let mut times = vec![self.t0];
        let mut states = vec![self.y0.as_slice().to_vec()];
        self.statistics = Dop853Statistics::default();

        let mut y = self.y0.clone();
        let mut t = self.t0;
        let mut k: Vec<DVector<f64>> = (0..=N).map(|_| DVector::zeros(y.len())).collect();
        k[0] = match self.evaluate(t, &y) {
            Ok(value) => value,
            Err(error) => {
                self.finish_timing(started);
                return Err(error);
            }
        };
        let mut h_abs = if interval == 0.0 {
            0.0
        } else if let Some(first_step) = self.options.first_step {
            first_step.min(interval).min(self.options.max_step)
        } else {
            match self.select_initial_step(t, &y, &k[0], direction, interval) {
                Ok(step) => step,
                Err(error) => {
                    self.finish_timing(started);
                    return Err(error);
                }
            }
        };
        let mut status = Dop853Status::Finished;
        let mut rejected_before_accept = false;

        while direction * (self.t_bound - t) > 0.0 {
            if self.statistics.accepted_steps >= self.options.max_steps {
                self.finish_timing(started);
                return Err(Dop853Error::MaximumStepsExceeded {
                    limit: self.options.max_steps,
                    time: t,
                });
            }
            let min_step = 10.0 * ulp(t);
            h_abs = h_abs.min(self.options.max_step);
            if h_abs < min_step {
                self.finish_timing(started);
                return Err(Dop853Error::StepSizeUnderflow {
                    time: t,
                    step: h_abs,
                });
            }
            let mut h = direction * h_abs;
            if direction * (t + h - self.t_bound) > 0.0 {
                h = self.t_bound - t;
                h_abs = h.abs();
            }

            self.statistics.attempted_steps += 1;
            let t_new = t + h;
            for stage in 1..N {
                let mut y_stage = y.clone();
                for component in 0..y.len() {
                    let mut weighted = 0.0;
                    for prior in 0..stage {
                        weighted += A[stage][prior] * k[prior][component];
                    }
                    y_stage[component] += h * weighted;
                }
                k[stage] = match self.evaluate(t + h * C[stage], &y_stage) {
                    Ok(value) => value,
                    Err(error) => {
                        self.finish_timing(started);
                        return Err(error);
                    }
                };
            }
            let mut y_new = y.clone();
            for component in 0..y.len() {
                let mut weighted = 0.0;
                for stage in 0..N {
                    weighted += B[stage] * k[stage][component];
                }
                y_new[component] += h * weighted;
            }
            // The endpoint derivative is also the FSAL derivative for the next step.
            k[N] = match self.evaluate(t_new, &y_new) {
                Ok(value) => value,
                Err(error) => {
                    self.finish_timing(started);
                    return Err(error);
                }
            };

            let error_norm = self.error_norm(h, &y, &y_new, &k);
            if !error_norm.is_finite() {
                self.finish_timing(started);
                return Err(Dop853Error::NonFiniteErrorEstimate { time: t });
            }
            if error_norm < 1.0 {
                t = t_new;
                y = y_new;
                k[0] = k[N].clone();
                self.statistics.accepted_steps += 1;
                self.statistics.minimum_accepted_step = Some(
                    self.statistics
                        .minimum_accepted_step
                        .map_or(h_abs, |old| old.min(h_abs)),
                );
                self.statistics.maximum_accepted_step = Some(
                    self.statistics
                        .maximum_accepted_step
                        .map_or(h_abs, |old| old.max(h_abs)),
                );
                if self.telemetry.mode().collects_counters() {
                    self.telemetry.record_accepted_step();
                }
                times.push(t);
                states.push(y.as_slice().to_vec());
                if stop_condition.is_some_and(|condition| condition(&y)) {
                    status = Dop853Status::StoppedByCondition;
                    break;
                }
                let mut factor = if error_norm == 0.0 {
                    MAX_FACTOR
                } else {
                    (SAFETY * error_norm.powf(ERROR_EXPONENT)).clamp(MIN_FACTOR, MAX_FACTOR)
                };
                if rejected_before_accept {
                    factor = factor.min(1.0);
                }
                rejected_before_accept = false;
                h_abs *= factor;
            } else {
                rejected_before_accept = true;
                self.statistics.rejected_steps += 1;
                if self.telemetry.mode().collects_counters() {
                    self.telemetry.record_rejected_step();
                }
                h_abs *= (SAFETY * error_norm.powf(ERROR_EXPONENT)).max(MIN_FACTOR);
            }
        }

        self.finish_timing(started);
        Ok(Dop853Solution {
            times,
            states,
            status,
            statistics: self.statistics.clone(),
        })
    }

    fn evaluate(&mut self, t: f64, y: &DVector<f64>) -> Result<DVector<f64>, Dop853Error> {
        let value = (self.rhs)(t, y);
        self.statistics.function_evaluations += 1;
        if value.len() != y.len() {
            return Err(Dop853Error::RhsDimension {
                expected: y.len(),
                actual: value.len(),
                evaluation: self.statistics.function_evaluations,
            });
        }
        if let Some(index) = value.iter().position(|entry| !entry.is_finite()) {
            return Err(Dop853Error::NonFiniteRhs {
                index,
                evaluation: self.statistics.function_evaluations,
            });
        }
        Ok(value)
    }

    fn select_initial_step(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        f0: &DVector<f64>,
        direction: f64,
        interval: f64,
    ) -> Result<f64, Dop853Error> {
        let scale: Vec<f64> = y
            .iter()
            .map(|value| self.options.atol + self.options.rtol * value.abs())
            .collect();
        let d0 = rms(y.iter().zip(&scale).map(|(value, scale)| value / scale));
        let d1 = rms(f0.iter().zip(&scale).map(|(value, scale)| value / scale));
        let h0 = if d0 < 1e-5 || d1 < 1e-5 {
            1e-6
        } else {
            0.01 * d0 / d1
        };
        let h0 = h0.min(interval);
        let y1 = DVector::from_iterator(
            y.len(),
            y.iter().zip(f0).map(|(y, f)| y + direction * h0 * f),
        );
        let f1 = self.evaluate(t + direction * h0, &y1)?;
        let d2 = rms(f1
            .iter()
            .zip(f0)
            .zip(&scale)
            .map(|((f1, f0), scale)| (f1 - f0) / (h0 * scale)));
        let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
            (1e-6_f64).max(h0 * 1e-3)
        } else {
            (0.01 / d1.max(d2)).powf(1.0 / 8.0)
        };
        Ok((100.0 * h0)
            .min(h1)
            .min(interval)
            .min(self.options.max_step))
    }

    fn error_norm(
        &self,
        h: f64,
        y: &DVector<f64>,
        y_new: &DVector<f64>,
        k: &[DVector<f64>],
    ) -> f64 {
        let mut e5_sq = 0.0;
        let mut e3_sq = 0.0;
        for component in 0..y.len() {
            let scale = self.options.atol
                + self.options.rtol * y[component].abs().max(y_new[component].abs());
            let mut e5 = 0.0;
            let mut e3 = 0.0;
            for stage in 0..=N {
                e5 += E5[stage] * k[stage][component];
                e3 += E3[stage] * k[stage][component];
            }
            e5_sq += (e5 / scale).powi(2);
            e3_sq += (e3 / scale).powi(2);
        }
        if e5_sq == 0.0 && e3_sq == 0.0 {
            0.0
        } else {
            h.abs() * e5_sq / ((e5_sq + 0.01 * e3_sq) * y.len() as f64).sqrt()
        }
    }

    fn finish_timing(&mut self, started: Option<Instant>) {
        if let Some(started) = started {
            let elapsed = started.elapsed();
            self.statistics.solve_ms = elapsed.as_secs_f64() * 1e3;
            if self.telemetry.mode().collects_timings() {
                self.telemetry
                    .record_warm_stage_duration(IvpWarmStage::Solve, elapsed);
            }
        }
    }
}

fn validate_options(options: Dop853Options, t0: f64, t_bound: f64) -> Result<(), Dop853Error> {
    if !t0.is_finite() || !t_bound.is_finite() {
        return Err(Dop853Error::InvalidConfiguration(
            "time bounds must be finite",
        ));
    }
    if !options.rtol.is_finite() || options.rtol <= 0.0 {
        return Err(Dop853Error::InvalidConfiguration(
            "rtol must be finite and positive",
        ));
    }
    if !options.atol.is_finite() || options.atol <= 0.0 {
        return Err(Dop853Error::InvalidConfiguration(
            "atol must be finite and positive",
        ));
    }
    if options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(Dop853Error::InvalidConfiguration(
            "max_step must be positive",
        ));
    }
    if options
        .first_step
        .is_some_and(|step| !step.is_finite() || step <= 0.0)
    {
        return Err(Dop853Error::InvalidConfiguration(
            "first_step must be finite and positive",
        ));
    }
    if options
        .first_step
        .is_some_and(|step| step > (t_bound - t0).abs())
    {
        return Err(Dop853Error::InvalidConfiguration(
            "first_step exceeds the integration interval",
        ));
    }
    if options.max_steps == 0 {
        return Err(Dop853Error::InvalidConfiguration(
            "max_steps must be non-zero",
        ));
    }
    Ok(())
}

fn rms(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn ulp(value: f64) -> f64 {
    if value == 0.0 {
        f64::from_bits(1)
    } else {
        let next = if value > 0.0 {
            f64::from_bits(value.to_bits() + 1)
        } else {
            f64::from_bits(value.to_bits() - 1)
        };
        (next - value).abs()
    }
}

const C: [f64; N] = [
    0.0,
    0.05260015195876773,
    0.078900227938151601,
    0.1183503419072274,
    0.28164965809277259,
    1.0 / 3.0,
    0.25,
    4.0 / 13.0,
    0.6512820512820513,
    0.6,
    0.8571428571428571,
    1.0,
];

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn solve_exp(options: Dop853Options, t_bound: f64) -> Dop853Solution {
        Dop853::new(
            Box::new(|_, y| y.clone()),
            DVector::from_element(1, 1.0),
            0.0,
            t_bound,
            options,
            IvpTelemetry::disabled(),
        )
        .unwrap()
        .solve(None)
        .unwrap()
    }

    #[test]
    fn dop853_matches_exponential_reference_and_hits_final_time_exactly() {
        let solution = solve_exp(Dop853Options::default(), 2.0);
        assert_relative_eq!(
            *solution.states.last().unwrap().first().unwrap(),
            2.0_f64.exp(),
            epsilon = 2e-8
        );
        assert_eq!(*solution.times.last().unwrap(), 2.0);
        assert!(solution.statistics.accepted_steps > 0);
        assert_eq!(
            solution.statistics.function_evaluations,
            2 + 12 * solution.statistics.attempted_steps
        );
    }

    #[test]
    fn dop853_adapts_tolerance_and_rejects_trial_steps() {
        let loose = solve_exp(
            Dop853Options {
                rtol: 1e-3,
                atol: 1e-6,
                ..Dop853Options::default()
            },
            10.0,
        );
        let tight = solve_exp(
            Dop853Options {
                rtol: 1e-11,
                atol: 1e-13,
                first_step: Some(2.0),
                ..Dop853Options::default()
            },
            2.0,
        );
        let exact = 2.0_f64.exp();
        let loose_error = (*loose.states.last().unwrap().first().unwrap() - exact).abs();
        let tight_error = (*tight.states.last().unwrap().first().unwrap() - exact).abs();
        assert!(tight_error < loose_error);
        assert!(tight.statistics.accepted_steps >= loose.statistics.accepted_steps);
        assert!(tight.statistics.rejected_steps > 0);
    }

    #[test]
    fn dop853_supports_backward_integration_and_stop_condition() {
        let backward = Dop853::new(
            Box::new(|_, y| y.clone()),
            DVector::from_element(1, 1.0),
            1.0,
            0.0,
            Dop853Options::default(),
            IvpTelemetry::disabled(),
        )
        .unwrap()
        .solve(None)
        .unwrap();
        assert_relative_eq!(
            *backward.states.last().unwrap().first().unwrap(),
            1.0 / 1.0_f64.exp(),
            epsilon = 1e-9
        );
        assert_eq!(*backward.times.last().unwrap(), 0.0);

        let mut stopped = Dop853::new(
            Box::new(|_, y| y.clone()),
            DVector::from_element(1, 1.0),
            0.0,
            10.0,
            Dop853Options::default(),
            IvpTelemetry::disabled(),
        )
        .unwrap();
        let stopped = stopped
            .solve(Some(&|y: &DVector<f64>| y[0] >= 2.0))
            .unwrap();
        assert_eq!(stopped.status, Dop853Status::StoppedByCondition);
        assert!(stopped.times.last().unwrap() < &10.0);
    }

    #[test]
    fn dop853_returns_typed_rhs_shape_errors() {
        let mut solver = Dop853::new(
            Box::new(|_, _| DVector::zeros(2)),
            DVector::from_element(1, 1.0),
            0.0,
            1.0,
            Dop853Options::default(),
            IvpTelemetry::disabled(),
        )
        .unwrap();
        assert!(matches!(
            solver.solve(None),
            Err(Dop853Error::RhsDimension {
                expected: 1,
                actual: 2,
                ..
            })
        ));
    }
}
