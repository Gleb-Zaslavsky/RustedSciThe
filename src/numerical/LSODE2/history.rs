//! Shared LSODE-style history and tolerance helpers.
//!
//! This module is deliberately independent from the current BDF bridge.  It is
//! the small common substrate we can use for both the existing BDF-backed path
//! and the future Adams engine: raw solution history, Nordsieck columns,
//! prediction, and ODEPACK-style weighted norms.

const LSODE2_MAX_HISTORY_ORDER: usize = 5;
const DIFF_TO_NORD_1: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
const DIFF_TO_NORD_2: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, -0.5], [0.0, 0.0, 0.5]];
const DIFF_TO_NORD_3: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -0.5, 1.0 / 3.0],
    [0.0, 0.0, 0.5, -0.5],
    [0.0, 0.0, 0.0, 1.0 / 6.0],
];
const DIFF_TO_NORD_4: [[f64; 5]; 5] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -0.5, 1.0 / 3.0, -0.25],
    [0.0, 0.0, 0.5, -0.5, 11.0 / 24.0],
    [0.0, 0.0, 0.0, 1.0 / 6.0, -0.25],
    [0.0, 0.0, 0.0, 0.0, 1.0 / 24.0],
];
const DIFF_TO_NORD_5: [[f64; 6]; 6] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -0.5, 1.0 / 3.0, -0.25, 0.2],
    [0.0, 0.0, 0.5, -0.5, 11.0 / 24.0, -5.0 / 12.0],
    [0.0, 0.0, 0.0, 1.0 / 6.0, -0.25, 7.0 / 24.0],
    [0.0, 0.0, 0.0, 0.0, 1.0 / 24.0, -1.0 / 12.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 120.0],
];

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2HistoryError {
    EmptySystem,
    InvalidOrder { order: usize, max_order: usize },
    DimensionMismatch { expected: usize, actual: usize },
    NonPositiveErrorWeight { index: usize, value: f64 },
}

impl std::fmt::Display for Lsode2HistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySystem => write!(f, "LSODE2 history requires at least one equation"),
            Self::InvalidOrder { order, max_order } => {
                write!(
                    f,
                    "LSODE2 history order {order} exceeds max order {max_order}"
                )
            }
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "LSODE2 history dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::NonPositiveErrorWeight { index, value } => write!(
                f,
                "LSODE2 error weight at index {index} must be positive, got {value}"
            ),
        }
    }
}

impl std::error::Error for Lsode2HistoryError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2NordsieckHistory {
    n: usize,
    max_order: usize,
    /// Column-major by history block: `data[order * n + variable]`.
    data: Vec<f64>,
}

impl Lsode2NordsieckHistory {
    pub fn new(n: usize, max_order: usize) -> Result<Self, Lsode2HistoryError> {
        if n == 0 {
            return Err(Lsode2HistoryError::EmptySystem);
        }
        Ok(Self {
            n,
            max_order,
            data: vec![0.0; (max_order + 1) * n],
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn max_order(&self) -> usize {
        self.max_order
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn col(&self, order: usize) -> Result<&[f64], Lsode2HistoryError> {
        self.check_order(order)?;
        let start = order * self.n;
        Ok(&self.data[start..start + self.n])
    }

    pub fn col_mut(&mut self, order: usize) -> Result<&mut [f64], Lsode2HistoryError> {
        self.check_order(order)?;
        let start = order * self.n;
        Ok(&mut self.data[start..start + self.n])
    }

    pub fn set_col(&mut self, order: usize, values: &[f64]) -> Result<(), Lsode2HistoryError> {
        if values.len() != self.n {
            return Err(Lsode2HistoryError::DimensionMismatch {
                expected: self.n,
                actual: values.len(),
            });
        }
        self.col_mut(order)?.copy_from_slice(values);
        Ok(())
    }

    pub fn zero_from(&mut self, first_order: usize) -> Result<(), Lsode2HistoryError> {
        if first_order > self.max_order + 1 {
            return Err(Lsode2HistoryError::InvalidOrder {
                order: first_order,
                max_order: self.max_order,
            });
        }
        for order in first_order..=self.max_order {
            self.col_mut(order)?.fill(0.0);
        }
        Ok(())
    }

    pub fn predict_into(
        &self,
        out: &mut Lsode2NordsieckHistory,
        order: usize,
    ) -> Result<(), Lsode2HistoryError> {
        self.check_order(order)?;
        if out.n != self.n {
            return Err(Lsode2HistoryError::DimensionMismatch {
                expected: self.n,
                actual: out.n,
            });
        }
        if out.max_order < order {
            return Err(Lsode2HistoryError::InvalidOrder {
                order,
                max_order: out.max_order,
            });
        }

        out.data.fill(0.0);
        for j in 0..=order {
            for k in j..=order {
                let coeff = binomial(k, j) as f64;
                let source = self.col(k)?;
                let target = out.col_mut(j)?;
                for i in 0..self.n {
                    target[i] += coeff * source[i];
                }
            }
        }
        Ok(())
    }

    fn check_order(&self, order: usize) -> Result<(), Lsode2HistoryError> {
        if order > self.max_order {
            return Err(Lsode2HistoryError::InvalidOrder {
                order,
                max_order: self.max_order,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2YHistory {
    n: usize,
    max_order: usize,
    /// Blocks: 0 -> y_n, 1 -> y_{n-1}, ...
    data: Vec<f64>,
}

impl Lsode2YHistory {
    pub fn new(y0: &[f64], max_order: usize) -> Result<Self, Lsode2HistoryError> {
        if y0.is_empty() {
            return Err(Lsode2HistoryError::EmptySystem);
        }
        let n = y0.len();
        let mut data = vec![0.0; (max_order + 1) * n];
        data[..n].copy_from_slice(y0);
        Ok(Self { n, max_order, data })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn max_order(&self) -> usize {
        self.max_order
    }

    pub fn block(&self, age: usize) -> Result<&[f64], Lsode2HistoryError> {
        self.check_age(age)?;
        let start = age * self.n;
        Ok(&self.data[start..start + self.n])
    }

    pub fn block_mut(&mut self, age: usize) -> Result<&mut [f64], Lsode2HistoryError> {
        self.check_age(age)?;
        let start = age * self.n;
        Ok(&mut self.data[start..start + self.n])
    }

    pub fn push_front(&mut self, y_new: &[f64]) -> Result<(), Lsode2HistoryError> {
        if y_new.len() != self.n {
            return Err(Lsode2HistoryError::DimensionMismatch {
                expected: self.n,
                actual: y_new.len(),
            });
        }
        for age in (1..=self.max_order).rev() {
            let dst = age * self.n;
            let src = (age - 1) * self.n;
            self.data.copy_within(src..src + self.n, dst);
        }
        self.data[..self.n].copy_from_slice(y_new);
        Ok(())
    }

    pub fn backward_differences(&self, order: usize) -> Result<Vec<f64>, Lsode2HistoryError> {
        self.check_age(order)?;
        let mut work = vec![0.0; (order + 1) * self.n];
        let mut out = vec![0.0; (order + 1) * self.n];

        for age in 0..=order {
            let dst = age * self.n;
            work[dst..dst + self.n].copy_from_slice(self.block(age)?);
        }
        out[..self.n].copy_from_slice(self.block(0)?);

        for diff_order in 1..=order {
            for age in 0..=(order - diff_order) {
                let a = age * self.n;
                let b = (age + 1) * self.n;
                for i in 0..self.n {
                    work[a + i] -= work[b + i];
                }
            }
            let dst = diff_order * self.n;
            out[dst..dst + self.n].copy_from_slice(&work[..self.n]);
        }

        Ok(out)
    }

    fn check_age(&self, age: usize) -> Result<(), Lsode2HistoryError> {
        if age > self.max_order {
            return Err(Lsode2HistoryError::InvalidOrder {
                order: age,
                max_order: self.max_order,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2Tolerance {
    Scalar { rtol: f64, atol: f64 },
    Vector { rtol: Vec<f64>, atol: Vec<f64> },
}

impl Lsode2Tolerance {
    pub fn scalar(rtol: f64, atol: f64) -> Self {
        Self::Scalar { rtol, atol }
    }

    pub fn vector(rtol: Vec<f64>, atol: Vec<f64>) -> Self {
        Self::Vector { rtol, atol }
    }
}

pub fn error_weights(
    y: &[f64],
    tolerance: &Lsode2Tolerance,
) -> Result<Vec<f64>, Lsode2HistoryError> {
    let mut weights = vec![0.0; y.len()];
    match tolerance {
        Lsode2Tolerance::Scalar { rtol, atol } => {
            for (i, yi) in y.iter().enumerate() {
                weights[i] = atol + rtol * yi.abs();
            }
        }
        Lsode2Tolerance::Vector { rtol, atol } => {
            if rtol.len() != y.len() {
                return Err(Lsode2HistoryError::DimensionMismatch {
                    expected: y.len(),
                    actual: rtol.len(),
                });
            }
            if atol.len() != y.len() {
                return Err(Lsode2HistoryError::DimensionMismatch {
                    expected: y.len(),
                    actual: atol.len(),
                });
            }
            for i in 0..y.len() {
                weights[i] = atol[i] + rtol[i] * y[i].abs();
            }
        }
    }

    for (index, value) in weights.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(Lsode2HistoryError::NonPositiveErrorWeight { index, value });
        }
    }
    Ok(weights)
}

pub fn weighted_rms_norm(values: &[f64], weights: &[f64]) -> Result<f64, Lsode2HistoryError> {
    if values.len() != weights.len() {
        return Err(Lsode2HistoryError::DimensionMismatch {
            expected: values.len(),
            actual: weights.len(),
        });
    }
    if values.is_empty() {
        return Err(Lsode2HistoryError::EmptySystem);
    }

    let mut sum = 0.0;
    for (index, (&value, &weight)) in values.iter().zip(weights.iter()).enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(Lsode2HistoryError::NonPositiveErrorWeight {
                index,
                value: weight,
            });
        }
        let scaled = value / weight;
        sum += scaled * scaled;
    }
    Ok((sum / values.len() as f64).sqrt())
}

pub fn backward_differences_to_nordsieck(
    diffs: &[f64],
    order: usize,
    target: &mut Lsode2NordsieckHistory,
) -> Result<(), Lsode2HistoryError> {
    if order > target.max_order() {
        return Err(Lsode2HistoryError::InvalidOrder {
            order,
            max_order: target.max_order(),
        });
    }
    if order > LSODE2_MAX_HISTORY_ORDER {
        return Err(Lsode2HistoryError::InvalidOrder {
            order,
            max_order: LSODE2_MAX_HISTORY_ORDER,
        });
    }
    if diffs.len() != (order + 1) * target.n() {
        return Err(Lsode2HistoryError::DimensionMismatch {
            expected: (order + 1) * target.n(),
            actual: diffs.len(),
        });
    }

    target.zero_from(0)?;
    let n = target.n();
    match order {
        0 => {
            target.col_mut(0)?.copy_from_slice(&diffs[..n]);
        }
        1 => apply_diff_to_nordsieck_matrix::<2>(n, diffs, target, &DIFF_TO_NORD_1)?,
        2 => apply_diff_to_nordsieck_matrix::<3>(n, diffs, target, &DIFF_TO_NORD_2)?,
        3 => apply_diff_to_nordsieck_matrix::<4>(n, diffs, target, &DIFF_TO_NORD_3)?,
        4 => apply_diff_to_nordsieck_matrix::<5>(n, diffs, target, &DIFF_TO_NORD_4)?,
        5 => apply_diff_to_nordsieck_matrix::<6>(n, diffs, target, &DIFF_TO_NORD_5)?,
        _ => unreachable!("validated above"),
    }
    Ok(())
}

pub fn reconcile_first_nordsieck_derivative(
    scaled_derivative: &[f64],
    target: &mut Lsode2NordsieckHistory,
) -> Result<(), Lsode2HistoryError> {
    if target.max_order() == 0 {
        return Ok(());
    }
    if scaled_derivative.len() != target.n() {
        return Err(Lsode2HistoryError::DimensionMismatch {
            expected: target.n(),
            actual: scaled_derivative.len(),
        });
    }
    target.col_mut(1)?.copy_from_slice(scaled_derivative);
    Ok(())
}

fn apply_diff_to_nordsieck_matrix<const M: usize>(
    n: usize,
    diffs: &[f64],
    target: &mut Lsode2NordsieckHistory,
    matrix: &[[f64; M]; M],
) -> Result<(), Lsode2HistoryError> {
    for (j, row) in matrix.iter().enumerate() {
        let zj = target.col_mut(j)?;
        for (k, coeff) in row.iter().enumerate().skip(j) {
            let dk = &diffs[k * n..(k + 1) * n];
            for i in 0..n {
                zj[i] += *coeff * dk[i];
            }
        }
    }
    Ok(())
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 1..=k {
        result = result * (n + 1 - i) / i;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_history_push_front_preserves_recent_solution_blocks() {
        let mut history = Lsode2YHistory::new(&[1.0, 2.0], 3).unwrap();
        history.block_mut(1).unwrap().copy_from_slice(&[10.0, 20.0]);
        history
            .block_mut(2)
            .unwrap()
            .copy_from_slice(&[100.0, 200.0]);

        history.push_front(&[7.0, 8.0]).unwrap();

        assert_eq!(history.block(0).unwrap(), &[7.0, 8.0]);
        assert_eq!(history.block(1).unwrap(), &[1.0, 2.0]);
        assert_eq!(history.block(2).unwrap(), &[10.0, 20.0]);
        assert_eq!(history.block(3).unwrap(), &[100.0, 200.0]);
    }

    #[test]
    fn y_history_builds_backward_differences_for_quadratic_sequence() {
        let mut history = Lsode2YHistory::new(&[25.0], 3).unwrap();
        history.block_mut(1).unwrap()[0] = 16.0;
        history.block_mut(2).unwrap()[0] = 9.0;
        history.block_mut(3).unwrap()[0] = 4.0;

        let diffs = history.backward_differences(3).unwrap();

        assert_eq!(diffs, vec![25.0, 9.0, 2.0, 0.0]);
    }

    #[test]
    fn nordsieck_predictor_advances_polynomial_columns_with_pascal_transform() {
        let mut current = Lsode2NordsieckHistory::new(1, 3).unwrap();
        let mut predicted = Lsode2NordsieckHistory::new(1, 3).unwrap();
        current.set_col(0, &[1.0]).unwrap();
        current.set_col(1, &[2.0]).unwrap();
        current.set_col(2, &[3.0]).unwrap();
        current.set_col(3, &[4.0]).unwrap();

        current.predict_into(&mut predicted, 3).unwrap();

        assert_eq!(predicted.col(0).unwrap(), &[10.0]);
        assert_eq!(predicted.col(1).unwrap(), &[20.0]);
        assert_eq!(predicted.col(2).unwrap(), &[15.0]);
        assert_eq!(predicted.col(3).unwrap(), &[4.0]);
    }

    #[test]
    fn scalar_error_weights_and_weighted_rms_norm_match_odepack_shape() {
        let y = [2.0, -4.0];
        let weights = error_weights(&y, &Lsode2Tolerance::scalar(0.1, 0.01)).unwrap();
        assert_eq!(weights, vec![0.21000000000000002, 0.41000000000000003]);

        let norm = weighted_rms_norm(&[0.21, 0.82], &weights).unwrap();
        let expected = ((1.0_f64 + 4.0) / 2.0).sqrt();
        assert!((norm - expected).abs() < 1e-12);
    }

    #[test]
    fn vector_error_weights_reject_dimension_mismatch_and_nonpositive_entries() {
        let mismatch = error_weights(
            &[1.0, 2.0],
            &Lsode2Tolerance::vector(vec![1e-3], vec![1e-6, 1e-6]),
        )
        .unwrap_err();
        assert!(matches!(
            mismatch,
            Lsode2HistoryError::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        ));

        let nonpositive = error_weights(&[0.0], &Lsode2Tolerance::scalar(0.0, 0.0)).unwrap_err();
        assert!(matches!(
            nonpositive,
            Lsode2HistoryError::NonPositiveErrorWeight { index: 0, .. }
        ));
    }

    #[test]
    fn backward_differences_convert_to_nordsieck_columns() {
        let mut target = Lsode2NordsieckHistory::new(1, 3).unwrap();
        let diffs = vec![25.0, 9.0, 2.0, 0.0];

        backward_differences_to_nordsieck(&diffs, 3, &mut target).unwrap();

        assert_eq!(target.col(0).unwrap(), &[25.0]);
        assert_eq!(target.col(1).unwrap(), &[8.0]);
        assert_eq!(target.col(2).unwrap(), &[1.0]);
        assert_eq!(target.col(3).unwrap(), &[0.0]);
    }

    #[test]
    fn reconcile_first_nordsieck_derivative_overwrites_first_derivative_column() {
        let mut target = Lsode2NordsieckHistory::new(1, 3).unwrap();
        let diffs = vec![25.0, 9.0, 2.0, 0.0];
        backward_differences_to_nordsieck(&diffs, 3, &mut target).unwrap();

        reconcile_first_nordsieck_derivative(&[7.5], &mut target).unwrap();

        assert_eq!(target.col(0).unwrap(), &[25.0]);
        assert_eq!(target.col(1).unwrap(), &[7.5]);
        assert_eq!(target.col(2).unwrap(), &[1.0]);
        assert_eq!(target.col(3).unwrap(), &[0.0]);
    }
}
