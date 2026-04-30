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

#[derive(Debug, Clone, PartialEq)]
pub enum TridiagonalPpError {
    DimensionMismatch,
    ZeroPivot { index: usize, value: f64 },
}

impl TridiagonalLuPp {
    pub fn factor(lower: &[f64], diag: &[f64], upper: &[f64]) -> Result<Self, TridiagonalPpError> {
        let n = diag.len();

        if n == 0 || lower.len() + 1 != n || upper.len() + 1 != n {
            return Err(TridiagonalPpError::DimensionMismatch);
        }

        let pivot_eps = 1e-14;

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
                dl[k] = mult;
                d[k + 1] -= mult * du[k];
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

                let old_dk = d[k];
                let old_du_k = du[k];
                let old_dkp1 = d[k + 1];

                d[k] = dl[k];
                dl[k] = mult;

                du[k] = old_dkp1;
                d[k + 1] = old_du_k - mult * old_dkp1;

                if k + 1 < n - 1 {
                    let old_du_kp1 = du[k + 1];
                    du2[k] = old_du_kp1;
                    du[k + 1] = -mult * old_du_kp1;
                }

                let _ = old_dk;
            }
        }

        ipiv[n - 1] = n - 1;

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
    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), TridiagonalPpError> {
        if rhs.len() != self.n {
            return Err(TridiagonalPpError::DimensionMismatch);
        }

        let n = self.n;

        if n == 1 {
            rhs[0] /= self.d[0];
            return Ok(());
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

        Ok(())
    }

    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, TridiagonalPpError> {
        let mut x = rhs.to_vec();
        self.solve_in_place(&mut x)?;
        Ok(x)
    }

    pub fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        stride: usize,
    ) -> Result<(), TridiagonalPpError> {
        if nrhs == 0 {
            return Ok(());
        }

        if stride < self.n || rhs.len() < nrhs * stride {
            return Err(TridiagonalPpError::DimensionMismatch);
        }

        for r in 0..nrhs {
            let offset = r * stride;
            let b = &mut rhs[offset..offset + self.n];
            self.solve_in_place(b)?;
        }

        Ok(())
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
}
