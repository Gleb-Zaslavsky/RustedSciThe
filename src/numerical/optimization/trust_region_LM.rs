//! Solver for the trust-region sub-problem in the LM algorithm.
#![allow(clippy::excessive_precision)]

use crate::numerical::optimization::qr_LM::LinearLeastSquaresDiagonalProblem;
use crate::numerical::optimization::utils::{dwarf, enorm};
use nalgebra::DVector;
use log::info;
pub struct LMParameter {
    pub step: DVector<f64>,
    pub lambda: f64,
    pub dp_norm: f64,
}

/// Approximately solve the LM trust-region subproblem.
///
/// Given `F` and a non-singular diagonal matrix `D`,
/// this routine approximately solves the problem
/// ```math
///   \min_{\vec{p}\in\R^n}\|\mathbf{J}\vec{p} - \vec{r}\|^2\text{ subject to }\|\mathbf{D}\vec{p}\|\leq\Delta.
/// ```
/// or min||Jp - r||^2 subject to ||Dp|| <= Delta.where J is an m by n matrix and r is an m vector.
/// It can be shown that `\vec{p}` with `\|\mathbf{D}\vec{p}\|\leq\Delta` is
/// a solution if and only if there exists `\lambda\geq 0` such that
/// ```math
/// \begin{aligned}
/// (\mathbf{J}^\top\mathbf{J} + \lambda \mathbf{D}\mathbf{D})\vec{p} &= \mathbf{J}^\top\vec{r}, \\
/// \lambda(\Delta - \|\mathbf{D}\vec{p}\|) &= 0.
/// \end{aligned}
/// ```
/// or (J'J + lambda D'D)p = J'r, lambda(Delta - ||Dp||) = 0.
/// # Inputs
///
/// The matrix `F` and vector `r` correspond to `A` and
/// `b`
///
/// # Reference
///
/// This method resembles `LMPAR` from `MINPACK`. See the following paper
/// on how it works:
///
/// > Moré J.J. (1978) The Levenberg-Marquardt algorithm: Implementation and theory. In: Watson G.A. (eds) Numerical Analysis. Lecture Notes in Mathematics, vol 630. Springer, Berlin, Heidelberg.
///
/// Chapter 4.3 of "Numerical Optimization" by Nocedal and Wright also contains
/// information about this algorithm but it misses a few details.
pub fn determine_lambda_and_parameter_update(
    lls: &mut LinearLeastSquaresDiagonalProblem,
    diag: &DVector<f64>,
    delta: f64,
    initial_lambda: f64,
) -> LMParameter {
    const P1: f64 = 0.1;
    debug_assert!(delta > 0.0);
    debug_assert!(initial_lambda >= 0.0);
    debug_assert!(!diag.iter().any(|&x| x == 0.0));

    let is_non_singular = lls.is_non_singular();
    let (mut p, mut l) = lls.solve_with_zero_diagonal();
    //  println!("p: {:?}", p);
    let mut diag_p = p.component_mul(diag);
    let mut diag_p_norm = enorm(&diag_p);
    let mut fp = diag_p_norm - delta;
    if fp <= delta * P1 {
        info!("fp <= delta * convert(P1), diag_p_norm = {}", diag_p_norm);
        // we have a feasible p with lambda = 0
        return LMParameter {
            step: p,
            lambda: 0.0,
            dp_norm: diag_p_norm,
        };
    }
    info!("fp => delta * convert(P1) diag_p_norm = {}", diag_p_norm);
    // we now look for lambda > 0 with ||D p|| = delta
    // by using an approximate Newton iteration.

    let mut lambda_lower = if is_non_singular {
        p.copy_from(&diag_p);
        p /= diag_p_norm;
        for (p_val, d_val) in p.iter_mut().zip(diag.iter()) {
            *p_val *= *d_val;
        }
        p = l.solve(p);
        let norm = enorm(&p);
        ((fp / delta) / norm) / norm
    } else {
        0.0
    };

    let gnorm;
    let mut lambda_upper = {
        // Upper bound is given by ||(J * D^T)^T r|| / delta, see paper cited above.
        p = l.mul_qt_b(p);
        for j in 0..p.nrows() {
            p[j] /= diag[l.permutation[j]];
        }
        gnorm = enorm(&p);
        let upper = gnorm / delta;
        if upper == 0.0 {
            dwarf::<f64>() / f64::min(delta, P1)
        } else {
            upper
        }
    };

    let mut lambda = f64::min(f64::max(initial_lambda, lambda_lower), lambda_upper);
    if lambda == 0.0 {
        lambda = gnorm / diag_p_norm;
    }

    for iteration in 1.. {
        if lambda == 0.0 {
            lambda = f64::max(dwarf(), lambda_upper * 0.001);
        }
        let l_sqrt = f64::sqrt(lambda);
        diag_p.axpy(l_sqrt, diag, 0.0);
        let (p_new, l_new) = lls.solve_with_diagonal(&diag_p, p);
        p = p_new;
        l = l_new;
        diag_p = p.component_mul(diag);
        diag_p_norm = enorm(&diag_p);
        if iteration == 10 {
            break;
        }
        let fp_old = fp;
        fp = diag_p_norm - delta;
        if f64::abs(fp) <= delta * P1 || (lambda_lower == 0.0 && fp <= fp_old && fp_old < 0.0) {
            break;
        }

        let newton_correction = {
            p.copy_from(&diag_p);
            p /= diag_p_norm;
            for (p_val, d_val) in p.iter_mut().zip(diag.iter()) {
                *p_val *= *d_val;
            }
            p = l.solve(p);
            let norm = enorm(&p);
            ((fp / delta) / norm) / norm
        };

        if fp > 0.0 {
            lambda_lower = f64::max(lambda_lower, lambda);
        } else {
            lambda_upper = f64::min(lambda_upper, lambda);
        }
        lambda = f64::max(lambda_lower, lambda + newton_correction);
    }

    LMParameter {
        step: p,
        lambda,
        dp_norm: diag_p_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::determine_lambda_and_parameter_update;
    use crate::numerical::optimization::qr_LM::*;
    use approx::assert_relative_eq;
    use nalgebra::*;

    #[test]
    fn test_case1() {
        let j = DMatrix::from_column_slice(
            4,
            3,
            &[
                33., -40., 44., -43., -37., -1., -40., 48., 43., -11., -40., 43.,
            ],
        );
        let residual = DVector::from_vec(vec![7., -1., 0., -1.]);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);

        let diag = DVector::from_vec(vec![18.2, 18.2, 3.2]);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2);

        assert_relative_eq!(param.lambda, 34.628643558156341f64);
        let p_r = DVector::from_vec(vec![
            0.017591648698939,
            -0.020395135814051,
            0.059285196018896,
        ]);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case2() {
        let j = DMatrix::from_column_slice(
            4,
            3,
            &[
                -7., 28., -40., 29., 7., -49., -39., 43., -25., -47., -11., 34.,
            ],
        );
        let residual = DVector::from_vec(vec![-7., -8., -8., -10.]);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = DVector::from_vec(vec![10.2, 13.2, 1.2]);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2f64);

        assert_eq!(param.lambda.classify(), ::core::num::FpCategory::Zero);
        let p_r = DVector::from_vec(vec![
            -0.048474221517806,
            -0.007207732068190,
            0.083138659283539,
        ]);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case3() {
        let j = DMatrix::from_column_slice(
            4,
            3,
            &[
                8., -42., -34., -31., -30., -15., -36., -1., 27., 22., 44., 6.,
            ],
        );
        let residual = DVector::from_vec(vec![1., -5., 2., 7.]);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = DVector::from_vec(vec![4.2, 8.2, 11.2]);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2);

        assert_relative_eq!(param.lambda, 0.017646940861467262f64, epsilon = 1e-14);
        let p_r = DVector::from_vec(vec![
            -0.008462374169585,
            0.033658082419054,
            0.037230479167632,
        ]);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case4() {
        let j = DMatrix::from_column_slice(
            4,
            3,
            &[
                14., -12., 20., -11., 19., 38., -4., -11., -14., 12., -20., 11.,
            ],
        );
        let residual = DVector::from_vec(vec![-5., 3., -2., 7.]);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = DVector::from_vec(vec![6.2, 1.2, 0.2]);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2);

        assert_relative_eq!(param.lambda, 0.);
        let p_r = DVector::from_vec(vec![
            -0.000277548738904,
            -0.046232379576219,
            0.266724338086713,
        ]);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }
}
