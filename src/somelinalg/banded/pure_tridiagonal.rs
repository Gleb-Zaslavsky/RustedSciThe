use thiserror::Error;

/// The logical band from which a tridiagonal input value originated.
///
/// Keeping this as an enum, rather than embedding the band name in an error
/// string, lets callers classify invalid input without parsing diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TridiagonalBand {
    Lower,
    Diagonal,
    Upper,
    Rhs,
}

impl std::fmt::Display for TridiagonalBand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Lower => "lower",
            Self::Diagonal => "diagonal",
            Self::Upper => "upper",
            Self::Rhs => "rhs",
        };
        f.write_str(name)
    }
}

/// Failure modes for the pivoted tridiagonal factorization and solve.
///
/// The variants deliberately retain dimensions, locations and offending
/// values.  This is important for a future Python binding and for native
/// solver diagnostics: callers can branch on the variant without parsing a
/// formatted error message.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TridiagonalPpError {
    #[error(
        "dimension mismatch: lower_len={lower_len}, diag_len={diag_len}, upper_len={upper_len}"
    )]
    DimensionMismatch {
        lower_len: usize,
        diag_len: usize,
        upper_len: usize,
    },

    #[error("non-finite {band} value at index={index}, value={value}")]
    NonFiniteInput {
        band: TridiagonalBand,
        index: usize,
        value: f64,
    },

    #[error("invalid pivot tolerance: {value}")]
    InvalidPivotTolerance { value: f64 },

    #[error("zero or numerically tiny pivot at index={index}, value={value}")]
    ZeroPivot { index: usize, value: f64 },

    #[error("non-finite factor value in {band} at index={index}, value={value}")]
    NonFiniteFactor {
        band: TridiagonalBand,
        index: usize,
        value: f64,
    },

    #[error("RHS dimension mismatch: expected={expected}, actual={actual}")]
    RhsDimensionMismatch { expected: usize, actual: usize },

    #[error("invalid multiple-RHS layout: rhs_len={rhs_len}, nrhs={nrhs}, stride={stride}, n={n}")]
    InvalidRhsLayout {
        rhs_len: usize,
        nrhs: usize,
        stride: usize,
        n: usize,
    },

    #[error("non-finite solution value at index={index}, value={value}")]
    NonFiniteSolution { index: usize, value: f64 },
}

/// Default absolute pivot threshold used by the faithful tridiagonal route.
pub const DEFAULT_TRIDIAGONAL_PIVOT_EPSILON: f64 = 1e-14;

#[derive(Debug, Clone)]
pub struct TridiagonalLu {
    n: usize,

    // LU storage:
    // lower[i] stores multiplier l_{i+1,i}
    lower: Vec<f64>, // len n-1
    diag: Vec<f64>,  // U diagonal, len n
    upper: Vec<f64>, // U superdiagonal, len n-1

    pivot_eps: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TridiagonalError {
    DimensionMismatch,
    ZeroPivot { index: usize, value: f64 },
}

impl TridiagonalLu {
    pub fn factor(lower: &[f64], diag: &[f64], upper: &[f64]) -> Result<Self, TridiagonalError> {
        let n = diag.len();

        if n == 0 || lower.len() + 1 != n || upper.len() + 1 != n {
            return Err(TridiagonalError::DimensionMismatch);
        }

        let pivot_eps = 1e-14;

        let mut l = lower.to_vec();
        let mut d = diag.to_vec();
        let u = upper.to_vec();

        for i in 0..(n - 1) {
            let pivot = d[i];

            if pivot.abs() <= pivot_eps {
                return Err(TridiagonalError::ZeroPivot {
                    index: i,
                    value: pivot,
                });
            }

            let m = l[i] / pivot;
            l[i] = m;
            d[i + 1] -= m * u[i];
        }

        let last = d[n - 1];
        if last.abs() <= pivot_eps {
            return Err(TridiagonalError::ZeroPivot {
                index: n - 1,
                value: last,
            });
        }

        Ok(Self {
            n,
            lower: l,
            diag: d,
            upper: u,
            pivot_eps,
        })
    }

    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), TridiagonalError> {
        if rhs.len() != self.n {
            return Err(TridiagonalError::DimensionMismatch);
        }

        let n = self.n;

        // Forward solve: L y = b
        // L has unit diagonal and lower multipliers.
        for i in 1..n {
            rhs[i] -= self.lower[i - 1] * rhs[i - 1];
        }

        // Backward solve: U x = y
        rhs[n - 1] /= self.diag[n - 1];

        for i in (0..(n - 1)).rev() {
            rhs[i] = (rhs[i] - self.upper[i] * rhs[i + 1]) / self.diag[i];
        }

        Ok(())
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, TridiagonalError> {
        let mut x = rhs.to_vec();
        self.solve_in_place(&mut x)?;
        Ok(x)
    }
}
//==============================================================================================================

/// Reusable pivoted LU factorization for a tridiagonal `f64` matrix.
///
/// The factorization keeps the compact LAPACK-style bands (`dl`, `d`, `du`,
/// `du2`) and zero-based pivot decisions.  It never materializes an `n x n`
/// dense matrix, and the source coefficient slices are copied during
/// [`Self::factor`].
#[derive(Debug, Clone)]
pub struct TridiagonalLuPp {
    n: usize,
    dl: Vec<f64>,  // subdiagonal / L multipliers, len n-1
    d: Vec<f64>,   // U diagonal, len n
    du: Vec<f64>,  // U first superdiagonal, len n-1
    du2: Vec<f64>, // U second superdiagonal, len n-2
    ipiv: Vec<usize>,
    pivot_eps: f64,
}

impl TridiagonalLuPp {
    pub fn factor(lower: &[f64], diag: &[f64], upper: &[f64]) -> Result<Self, TridiagonalPpError> {
        Self::factor_with_pivot_epsilon(lower, diag, upper, DEFAULT_TRIDIAGONAL_PIVOT_EPSILON)
    }

    /// Factor a tridiagonal matrix with a caller-selected absolute pivot
    /// threshold.
    ///
    /// This is the pivoted analogue of LAPACK's compact tridiagonal LU
    /// contract.  The input bands are copied, so subsequent caller mutation
    /// cannot change the retained factorization.  A zero tolerance is allowed
    /// when the caller explicitly wants exact-zero pivot detection only.
    pub fn factor_with_pivot_epsilon(
        lower: &[f64],
        diag: &[f64],
        upper: &[f64],
        pivot_eps: f64,
    ) -> Result<Self, TridiagonalPpError> {
        let n = diag.len();

        if n == 0 || lower.len() + 1 != n || upper.len() + 1 != n {
            return Err(TridiagonalPpError::DimensionMismatch {
                lower_len: lower.len(),
                diag_len: diag.len(),
                upper_len: upper.len(),
            });
        }

        if !pivot_eps.is_finite() || pivot_eps < 0.0 {
            return Err(TridiagonalPpError::InvalidPivotTolerance { value: pivot_eps });
        }

        validate_band(TridiagonalBand::Lower, lower)?;
        validate_band(TridiagonalBand::Diagonal, diag)?;
        validate_band(TridiagonalBand::Upper, upper)?;

        let mut dl = lower.to_vec();
        let mut d = diag.to_vec();
        let mut du = upper.to_vec();
        let mut du2 = vec![0.0; n.saturating_sub(2)];
        let mut ipiv: Vec<usize> = (0..n).collect();

        if n == 1 {
            if d[0].abs() <= pivot_eps {
                return Err(TridiagonalPpError::ZeroPivot {
                    index: 0,
                    value: d[0],
                });
            }

            return Ok(Self {
                n,
                dl,
                d,
                du,
                du2,
                ipiv,
                pivot_eps,
            });
        }

        for k in 0..(n - 1) {
            if d[k].abs() >= dl[k].abs() {
                // No row interchange.
                ipiv[k] = k;

                if d[k].abs() <= pivot_eps {
                    return Err(TridiagonalPpError::ZeroPivot {
                        index: k,
                        value: d[k],
                    });
                }

                let mult = dl[k] / d[k];
                ensure_factor_finite(TridiagonalBand::Lower, k, mult)?;
                dl[k] = mult;
                d[k + 1] -= mult * du[k];
                ensure_factor_finite(TridiagonalBand::Diagonal, k + 1, d[k + 1])?;
            } else {
                // Interchange rows k and k+1.
                ipiv[k] = k + 1;

                if dl[k].abs() <= pivot_eps {
                    return Err(TridiagonalPpError::ZeroPivot {
                        index: k,
                        value: dl[k],
                    });
                }

                let mult = d[k] / dl[k];
                ensure_factor_finite(TridiagonalBand::Lower, k, mult)?;

                let old_dk = d[k];
                let old_du_k = du[k];
                let old_dkp1 = d[k + 1];

                d[k] = dl[k];
                dl[k] = mult;

                du[k] = old_dkp1;
                d[k + 1] = old_du_k - mult * old_dkp1;
                ensure_factor_finite(TridiagonalBand::Diagonal, k, d[k])?;
                ensure_factor_finite(TridiagonalBand::Lower, k, dl[k])?;
                ensure_factor_finite(TridiagonalBand::Upper, k, du[k])?;
                ensure_factor_finite(TridiagonalBand::Diagonal, k + 1, d[k + 1])?;

                if k + 1 < n - 1 {
                    let old_du_kp1 = du[k + 1];
                    du2[k] = old_du_kp1;
                    du[k + 1] = -mult * old_du_kp1;
                    ensure_factor_finite(TridiagonalBand::Upper, k, du2[k])?;
                    ensure_factor_finite(TridiagonalBand::Upper, k + 1, du[k + 1])?;
                }

                let _ = old_dk;
            }
        }

        ipiv[n - 1] = n - 1;

        ensure_factor_finite(TridiagonalBand::Diagonal, n - 1, d[n - 1])?;
        if d[n - 1].abs() <= pivot_eps {
            return Err(TridiagonalPpError::ZeroPivot {
                index: n - 1,
                value: d[n - 1],
            });
        }

        Ok(Self {
            n,
            dl,
            d,
            du,
            du2,
            ipiv,
            pivot_eps,
        })
    }
}

impl TridiagonalLuPp {
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn pivot_epsilon(&self) -> f64 {
        self.pivot_eps
    }

    /// Return the LAPACK-compatible row pivot decisions using zero-based
    /// indices.  The returned slice is owned by the factorization.
    #[inline]
    pub fn ipiv(&self) -> &[usize] {
        &self.ipiv
    }

    /// Solve one RHS in place without refactorizing the matrix.
    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), TridiagonalPpError> {
        if rhs.len() != self.n {
            return Err(TridiagonalPpError::RhsDimensionMismatch {
                expected: self.n,
                actual: rhs.len(),
            });
        }

        validate_band(TridiagonalBand::Rhs, rhs)?;

        let n = self.n;

        if n == 1 {
            rhs[0] /= self.d[0];
            return validate_solution(rhs);
        }

        // Apply row interchanges and solve L*y = P*b.
        for k in 0..(n - 1) {
            if self.ipiv[k] == k {
                rhs[k + 1] -= self.dl[k] * rhs[k];
            } else {
                let tmp = rhs[k];
                rhs[k] = rhs[k + 1];
                rhs[k + 1] = tmp - self.dl[k] * rhs[k];
            }
        }

        // Solve U*x = y.
        rhs[n - 1] /= self.d[n - 1];

        if n >= 2 {
            rhs[n - 2] = (rhs[n - 2] - self.du[n - 2] * rhs[n - 1]) / self.d[n - 2];
        }

        if n >= 3 {
            for k in (0..=(n - 3)).rev() {
                rhs[k] = (rhs[k] - self.du[k] * rhs[k + 1] - self.du2[k] * rhs[k + 2]) / self.d[k];
            }
        }

        validate_solution(rhs)?;

        Ok(())
    }

    /// Solve one RHS while returning an owned result.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, TridiagonalPpError> {
        let mut x = rhs.to_vec();
        self.solve_in_place(&mut x)?;
        Ok(x)
    }

    /// Solve row-major RHS blocks separated by `stride` elements.
    ///
    /// Only the first `n` entries of each strided block are active; padding is
    /// left untouched.  All active RHS values are validated before the first
    /// block is modified.
    pub fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        stride: usize,
    ) -> Result<(), TridiagonalPpError> {
        if nrhs == 0 {
            return Ok(());
        }

        let required_len =
            nrhs.checked_mul(stride)
                .ok_or(TridiagonalPpError::InvalidRhsLayout {
                    rhs_len: rhs.len(),
                    nrhs,
                    stride,
                    n: self.n,
                })?;
        if stride < self.n || rhs.len() < required_len {
            return Err(TridiagonalPpError::InvalidRhsLayout {
                rhs_len: rhs.len(),
                nrhs,
                stride,
                n: self.n,
            });
        }

        // Validate every active RHS before mutating the first one.  A failed
        // later RHS must not leave earlier columns partially solved.
        for r in 0..nrhs {
            validate_band(TridiagonalBand::Rhs, &rhs[r * stride..r * stride + self.n])?;
        }

        for r in 0..nrhs {
            let offset = r * stride;
            let b = &mut rhs[offset..offset + self.n];
            self.solve_in_place(b)?;
        }

        Ok(())
    }
}

fn validate_band(band: TridiagonalBand, values: &[f64]) -> Result<(), TridiagonalPpError> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TridiagonalPpError::NonFiniteInput { band, index, value });
    }
    Ok(())
}

fn validate_solution(values: &[f64]) -> Result<(), TridiagonalPpError> {
    validate_band(TridiagonalBand::Rhs, values).map_err(|error| match error {
        TridiagonalPpError::NonFiniteInput { index, value, .. } => {
            TridiagonalPpError::NonFiniteSolution { index, value }
        }
        other => other,
    })
}

fn ensure_factor_finite(
    band: TridiagonalBand,
    index: usize,
    value: f64,
) -> Result<(), TridiagonalPpError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TridiagonalPpError::NonFiniteFactor { band, index, value })
    }
}
//=================================================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn tridiag_matvec(lower: &[f64], diag: &[f64], upper: &[f64], x: &[f64]) -> Vec<f64> {
        let n = diag.len();
        let mut y = vec![0.0; n];

        for i in 0..n {
            y[i] += diag[i] * x[i];

            if i > 0 {
                y[i] += lower[i - 1] * x[i - 1];
            }

            if i + 1 < n {
                y[i] += upper[i] * x[i + 1];
            }
        }

        y
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn tridiagonal_lu_solves_known_system() {
        let lower = vec![-1.0, -1.0, -1.0];
        let diag = vec![2.0, 2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0, -1.0];

        let x_true = vec![1.0, 2.0, 3.0, 4.0];
        let b = tridiag_matvec(&lower, &diag, &upper, &x_true);

        let lu = TridiagonalLu::factor(&lower, &diag, &upper).unwrap();
        let x = lu.solve(&b).unwrap();

        assert!(max_abs_diff(&x, &x_true) < 1e-12);
    }

    #[test]
    fn tridiagonal_lu_reuses_factorization_for_multiple_rhs() {
        let lower = vec![1.0, 1.0, 1.0];
        let diag = vec![4.0, 4.0, 4.0, 4.0];
        let upper = vec![1.0, 1.0, 1.0];

        let lu = TridiagonalLu::factor(&lower, &diag, &upper).unwrap();

        let x1_true = vec![1.0, 0.0, -1.0, 2.0];
        let x2_true = vec![0.5, -0.5, 1.5, -2.0];

        let b1 = tridiag_matvec(&lower, &diag, &upper, &x1_true);
        let b2 = tridiag_matvec(&lower, &diag, &upper, &x2_true);

        let x1 = lu.solve(&b1).unwrap();
        let x2 = lu.solve(&b2).unwrap();

        assert!(max_abs_diff(&x1, &x1_true) < 1e-12);
        assert!(max_abs_diff(&x2, &x2_true) < 1e-12);
    }
}
//=========================================================================
#[cfg(test)]
mod tridiagonal_pp_tests {
    use super::*;

    fn tridiag_matvec(lower: &[f64], diag: &[f64], upper: &[f64], x: &[f64]) -> Vec<f64> {
        let n = diag.len();
        let mut y = vec![0.0; n];

        for i in 0..n {
            y[i] += diag[i] * x[i];

            if i > 0 {
                y[i] += lower[i - 1] * x[i - 1];
            }

            if i + 1 < n {
                y[i] += upper[i] * x[i + 1];
            }
        }

        y
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn pp_solves_no_pivot_case() {
        let lower = vec![-1.0, -1.0, -1.0];
        let diag = vec![2.0, 2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0, -1.0];

        let x_true = vec![1.0, 2.0, 3.0, 4.0];
        let b = tridiag_matvec(&lower, &diag, &upper, &x_true);

        let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();
        let x = lu.solve(&b).unwrap();

        assert!(max_abs_diff(&x, &x_true) < 1e-12);
    }

    #[test]
    fn pp_solves_pivot_required_case() {
        let lower = vec![2.0, 1.0, 1.0];
        let diag = vec![1e-12, 2.0, 2.0, 2.0];
        let upper = vec![1.0, 1.0, 1.0];

        let x_true = vec![1.0, -2.0, 0.5, 3.0];
        let b = tridiag_matvec(&lower, &diag, &upper, &x_true);

        let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();
        let x = lu.solve(&b).unwrap();

        assert!(max_abs_diff(&x, &x_true) < 1e-10);
    }

    #[test]
    fn pp_reuses_factorization_for_multiple_rhs() {
        let lower = vec![2.0, 1.0, 1.0, 1.0];
        let diag = vec![1e-10, 3.0, 3.0, 3.0, 3.0];
        let upper = vec![1.0, -0.5, 0.25, 1.0];

        let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();

        let x1_true = vec![1.0, 0.0, -1.0, 2.0, 0.5];
        let x2_true = vec![0.5, -0.5, 1.5, -2.0, 1.0];

        let b1 = tridiag_matvec(&lower, &diag, &upper, &x1_true);
        let b2 = tridiag_matvec(&lower, &diag, &upper, &x2_true);

        let x1 = lu.solve(&b1).unwrap();
        let x2 = lu.solve(&b2).unwrap();

        assert!(max_abs_diff(&x1, &x1_true) < 1e-10);
        assert!(max_abs_diff(&x2, &x2_true) < 1e-10);
    }
    #[test]
    fn pp_solve_multiple_in_place_recovers_solutions() {
        let lower = vec![2.0, 1.0, 1.0, 1.0];
        let diag = vec![1e-10, 3.0, 3.0, 3.0, 3.0];
        let upper = vec![1.0, -0.5, 0.25, 1.0];

        let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();

        let x1_true = vec![1.0, 0.0, -1.0, 2.0, 0.5];
        let x2_true = vec![0.5, -0.5, 1.5, -2.0, 1.0];

        let b1 = tridiag_matvec(&lower, &diag, &upper, &x1_true);
        let b2 = tridiag_matvec(&lower, &diag, &upper, &x2_true);

        let n = diag.len();
        let stride = n;

        let mut rhs = Vec::with_capacity(2 * stride);
        rhs.extend_from_slice(&b1);
        rhs.extend_from_slice(&b2);

        lu.solve_multiple_in_place(&mut rhs, 2, stride).unwrap();

        assert!(max_abs_diff(&rhs[0..n], &x1_true) < 1e-10);
        assert!(max_abs_diff(&rhs[n..2 * n], &x2_true) < 1e-10);
    }

    #[test]
    fn pp_solve_multiple_in_place_supports_padded_stride() {
        let lower = vec![2.0, 1.0, 1.0];
        let diag = vec![1e-10, 3.0, 3.0, 3.0];
        let upper = vec![1.0, -0.5, 0.25];

        let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();

        let x1_true = vec![1.0, -1.0, 0.5, 2.0];
        let x2_true = vec![0.25, 1.5, -0.5, 1.0];

        let b1 = tridiag_matvec(&lower, &diag, &upper, &x1_true);
        let b2 = tridiag_matvec(&lower, &diag, &upper, &x2_true);

        let n = diag.len();
        let stride = n + 2;
        let mut rhs = vec![999.0; 2 * stride];

        rhs[0..n].copy_from_slice(&b1);
        rhs[stride..stride + n].copy_from_slice(&b2);

        lu.solve_multiple_in_place(&mut rhs, 2, stride).unwrap();

        assert!(max_abs_diff(&rhs[0..n], &x1_true) < 1e-10);
        assert!(max_abs_diff(&rhs[stride..stride + n], &x2_true) < 1e-10);

        assert_eq!(rhs[n], 999.0);
        assert_eq!(rhs[n + 1], 999.0);
        assert_eq!(rhs[stride + n], 999.0);
        assert_eq!(rhs[stride + n + 1], 999.0);
    }

    #[test]
    fn pp_metadata_and_source_ownership_are_stable() {
        let mut lower = vec![2.0, -1.0];
        let mut diag = vec![1.0, 3.0, 4.0];
        let mut upper = vec![0.5, 2.0];

        let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();
        lower.fill(0.0);
        diag.fill(0.0);
        upper.fill(0.0);

        assert_eq!(lu.n(), 3);
        assert_eq!(lu.pivot_epsilon(), DEFAULT_TRIDIAGONAL_PIVOT_EPSILON);
        assert_eq!(lu.ipiv().len(), 3);
        assert!(
            lu.ipiv()
                .iter()
                .enumerate()
                .all(|(index, &pivot)| pivot >= index)
        );

        let x_true = vec![1.0, -2.0, 0.5];
        let original_lower = vec![2.0, -1.0];
        let original_diag = vec![1.0, 3.0, 4.0];
        let original_upper = vec![0.5, 2.0];
        let rhs = tridiag_matvec(&original_lower, &original_diag, &original_upper, &x_true);
        assert!(max_abs_diff(&lu.solve(&rhs).unwrap(), &x_true) < 1e-12);
    }

    #[test]
    fn pp_rejects_invalid_dimensions_tolerance_and_nonfinite_coefficients() {
        assert!(matches!(
            TridiagonalLuPp::factor(&[1.0], &[2.0, 3.0, 4.0], &[1.0]),
            Err(TridiagonalPpError::DimensionMismatch {
                lower_len: 1,
                diag_len: 3,
                upper_len: 1
            })
        ));

        assert!(matches!(
            TridiagonalLuPp::factor_with_pivot_epsilon(&[1.0], &[2.0, 3.0], &[1.0], f64::NAN),
            Err(TridiagonalPpError::InvalidPivotTolerance { value }) if value.is_nan()
        ));
        assert!(matches!(
            TridiagonalLuPp::factor_with_pivot_epsilon(&[1.0], &[2.0, 3.0], &[1.0], -1.0),
            Err(TridiagonalPpError::InvalidPivotTolerance { value: -1.0 })
        ));

        assert!(matches!(
            TridiagonalLuPp::factor(&[f64::NAN], &[2.0, 3.0], &[1.0]),
            Err(TridiagonalPpError::NonFiniteInput {
                band: TridiagonalBand::Lower,
                index: 0,
                value
            }) if value.is_nan()
        ));
        assert!(matches!(
            TridiagonalLuPp::factor(&[1.0], &[f64::INFINITY, 3.0], &[1.0]),
            Err(TridiagonalPpError::NonFiniteInput {
                band: TridiagonalBand::Diagonal,
                index: 0,
                value: f64::INFINITY
            })
        ));
        assert!(matches!(
            TridiagonalLuPp::factor(&[1.0], &[2.0, 3.0], &[f64::NEG_INFINITY]),
            Err(TridiagonalPpError::NonFiniteInput {
                band: TridiagonalBand::Upper,
                index: 0,
                value: f64::NEG_INFINITY
            })
        ));

        assert!(matches!(
            TridiagonalLuPp::factor(&[1e308], &[1e308, 1e308], &[-1e308]),
            Err(TridiagonalPpError::NonFiniteFactor {
                band: TridiagonalBand::Diagonal,
                index: 1,
                value: f64::INFINITY
            })
        ));
    }

    #[test]
    fn pp_rejects_bad_rhs_without_partial_mutation() {
        let lu = TridiagonalLuPp::factor(&[1.0, 1.0], &[4.0, 4.0, 4.0], &[1.0, 1.0]).unwrap();

        let mut short = vec![1.0, 2.0];
        assert!(matches!(
            lu.solve_in_place(&mut short),
            Err(TridiagonalPpError::RhsDimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));

        let mut rhs = vec![1.0, f64::NAN, 3.0];
        let before = rhs.clone();
        assert!(matches!(
            lu.solve_in_place(&mut rhs),
            Err(TridiagonalPpError::NonFiniteInput {
                band: TridiagonalBand::Rhs,
                index: 1,
                value
            }) if value.is_nan()
        ));
        assert_eq!(rhs[0], before[0]);
        assert!(rhs[1].is_nan() && before[1].is_nan());
        assert_eq!(rhs[2], before[2]);

        let mut multiple = vec![1.0, 2.0, 3.0, 4.0, f64::INFINITY, 6.0];
        let before = multiple.clone();
        assert!(matches!(
            lu.solve_multiple_in_place(&mut multiple, 2, 3),
            Err(TridiagonalPpError::NonFiniteInput {
                band: TridiagonalBand::Rhs,
                index: 1,
                value: f64::INFINITY
            })
        ));
        assert_eq!(multiple, before);

        assert!(matches!(
            lu.solve_multiple_in_place(&mut [0.0; 3], 2, usize::MAX),
            Err(TridiagonalPpError::InvalidRhsLayout { .. })
        ));
    }

    #[test]
    fn pp_reports_nonfinite_solution_after_arithmetic_overflow() {
        let lu = TridiagonalLuPp::factor_with_pivot_epsilon(&[], &[1e-100], &[], 0.0).unwrap();
        let error = lu.solve(&[1e308]).unwrap_err();
        assert!(matches!(
            error,
            TridiagonalPpError::NonFiniteSolution { index: 0, value }
                if value.is_infinite()
        ));
    }

    #[test]
    fn pp_solves_deterministic_family_with_small_residuals() {
        for n in 1usize..=32 {
            let lower: Vec<f64> = (0..n.saturating_sub(1))
                .map(|i| 0.2 + ((i * 7) % 5) as f64 * 0.07)
                .collect();
            let diag: Vec<f64> = (0..n).map(|i| 2.5 + (i % 4) as f64 * 0.2).collect();
            let upper: Vec<f64> = (0..n.saturating_sub(1))
                .map(|i| -0.15 - ((i * 11) % 3) as f64 * 0.04)
                .collect();
            let x_true: Vec<f64> = (0..n)
                .map(|i| ((i * 13 + 3) % 17) as f64 / 7.0 - 1.0)
                .collect();
            let rhs = tridiag_matvec(&lower, &diag, &upper, &x_true);
            let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();
            let x = lu.solve(&rhs).unwrap();

            assert!(max_abs_diff(&x, &x_true) < 1e-12, "n={n}, x={x:?}");
            assert!(
                max_abs_diff(&tridiag_matvec(&lower, &diag, &upper, &x), &rhs) < 1e-12,
                "n={n}, residual={:?}",
                tridiag_matvec(&lower, &diag, &upper, &x)
            );
        }
    }

    #[test]
    fn pp_matches_dense_reference_on_pivoting_family() {
        use nalgebra::{DMatrix, DVector};

        let mut state = 0x9e37_79b9_u64;
        let mut next_unit = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        let mut pivot_cases = 0;

        for case in 0..128 {
            let n = 2 + case % 15;
            let lower: Vec<f64> = (0..n - 1).map(|_| 1.5 * next_unit() - 0.75).collect();
            let diag: Vec<f64> = (0..n).map(|_| 0.4 + 1.2 * next_unit()).collect();
            let upper: Vec<f64> = (0..n - 1).map(|_| 1.5 * next_unit() - 0.75).collect();
            let x_true: Vec<f64> = (0..n).map(|_| 1.5 * next_unit() - 0.75).collect();
            let rhs = tridiag_matvec(&lower, &diag, &upper, &x_true);

            let dense = DMatrix::from_fn(n, n, |row, col| {
                if row == col {
                    diag[row]
                } else if row == col + 1 {
                    lower[col]
                } else if col == row + 1 {
                    upper[row]
                } else {
                    0.0
                }
            });
            let dense_solution = dense
                .lu()
                .solve(&DVector::from_vec(rhs.clone()))
                .expect("deterministic reference system should be nonsingular");
            let lu = TridiagonalLuPp::factor(&lower, &diag, &upper).unwrap();
            if lu
                .ipiv()
                .iter()
                .enumerate()
                .any(|(index, &pivot)| pivot != index)
            {
                pivot_cases += 1;
            }
            let x = lu.solve(&rhs).unwrap();

            assert!(
                max_abs_diff(&x, dense_solution.as_slice()) < 1e-10,
                "case={case}, n={n}, x={x:?}, dense={dense_solution:?}"
            );
        }

        assert!(
            pivot_cases >= 8,
            "pivoting was not exercised enough: {pivot_cases}"
        );
    }
}
