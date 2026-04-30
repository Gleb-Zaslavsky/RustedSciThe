use crate::numerical::LSODE::core::*;
struct TridiagStiff {
    alpha: f64,
    n: usize,
}

impl OdeRhs for TridiagStiff {
    fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
        assert_eq!(y.len(), self.n);
        assert_eq!(fy.len(), self.n);

        let a = self.alpha;

        for i in 0..self.n {
            let mut v = -2.0 * a * y[i];
            if i > 0 {
                v += a * y[i - 1];
            }
            if i + 1 < self.n {
                v += a * y[i + 1];
            }
            fy[i] = v;
        }
    }
}

fn tridiag_stiff_dense_jac(alpha: f64) -> impl FnMut(f64, &[f64], &mut [f64], usize) {
    move |_t: f64, _y: &[f64], jac: &mut [f64], n: usize| {
        jac.fill(0.0);

        let idx = |i: usize, j: usize| -> usize { j * n + i };

        for j in 0..n {
            jac[idx(j, j)] = -2.0 * alpha;

            if j > 0 {
                jac[idx(j, j - 1)] = alpha;
            }
            if j + 1 < n {
                jac[idx(j, j + 1)] = alpha;
            }
        }
    }
}

fn tridiag_stiff_banded_jac(
    alpha: f64,
) -> impl FnMut(f64, &[f64], &mut [f64], usize, usize, usize, usize) {
    move |_t: f64, _y: &[f64], jac: &mut [f64], n: usize, ml: usize, mu: usize, ldab: usize| {
        assert_eq!(ml, 1);
        assert_eq!(mu, 1);

        jac.fill(0.0);

        let idx = |i: usize, j: usize| -> usize {
            let row = mu + i - j;
            j * ldab + row
        };

        for j in 0..n {
            jac[idx(j, j)] = -2.0 * alpha;

            if j > 0 {
                jac[idx(j, j - 1)] = alpha;
            }
            if j + 1 < n {
                jac[idx(j, j + 1)] = alpha;
            }
        }
    }
}

#[inline]
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0_f64;
    for i in 0..a.len() {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}

#[cfg(test)]
mod test {
    use crate::numerical::LSODE::core::*;

    use crate::numerical::LSODE::dense::DenseMatrixStorage;
    use crate::numerical::LSODE::jacobian::{DenseAotJac, DenseFdJacobian, JacobianEvaluator};
    #[test]
    fn test_bdf_affine_q2_scalar() {
        let n = 1;
        let qmax = 5;
        let mut yh = YHistory::new(n, qmax, &[10.0]);
        yh.block_mut(0)[0] = 10.0; // y_n
        yh.block_mut(1)[0] = 7.0; // y_{n-1}

        let mut out = vec![0.0; 1];
        build_affine_part(&yh, 2, &mut out);

        let expected = (4.0 / 3.0) * 10.0 - (1.0 / 3.0) * 7.0;
        assert!((out[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn test_bdf_residual_q1_scalar() {
        let y_trial = [1.2];
        let f_trial = [0.5];
        let affine = [1.0];
        let h = 0.1;
        let q = 1;
        let mut residual = [0.0];

        build_bdf_residual(&y_trial, &f_trial, &affine, h, q, &mut residual);

        let expected = 1.2 - 0.1 * 0.5 - 1.0;
        assert!((residual[0] - expected).abs() < 1e-14);
    }
    #[test]
    fn test_backward_differences_scalar() {
        let n = 1;
        let qmax = 5;
        let mut yh = YHistory::new(n, qmax, &[0.0]);

        yh.block_mut(0)[0] = 10.0;
        yh.block_mut(1)[0] = 7.0;
        yh.block_mut(2)[0] = 5.0;
        yh.block_mut(3)[0] = 4.0;

        let mut work = vec![0.0; (qmax + 1) * n];
        let mut out = vec![0.0; (qmax + 1) * n];

        build_backward_differences_in_place(&yh, 3, &mut work, &mut out);

        assert!((out[0] - 10.0).abs() < 1e-14); // d0
        assert!((out[1] - 3.0).abs() < 1e-14); // d1 = 10 - 7
        assert!((out[2] - 1.0).abs() < 1e-14); // d2 = (10-7) - (7-5)
        assert!((out[3] - 0.0).abs() < 1e-14); // d3 = ...
    }
    #[test]
    fn test_predict_nordsieck_quadratic_scalar() {
        let n = 1;
        let qmax = 5;
        let mut zn = NordsieckHistory::new(n, qmax);
        let mut zpred = NordsieckHistory::new(n, qmax);

        zn.col_mut(0)[0] = 1.0;
        zn.col_mut(1)[0] = 0.2;
        zn.col_mut(2)[0] = 0.03;

        predict_nordsieck(&zn, &mut zpred, 2);

        let y_pred = zpred.col(0)[0];
        assert!((y_pred - 1.23).abs() < 1e-14);
    }

    #[test]
    fn test_dense_fd_matches_linear_jacobian() {
        struct LinearRhs {
            a: [[f64; 2]; 2],
        }

        impl OdeRhs for LinearRhs {
            fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
                fy[0] = self.a[0][0] * y[0] + self.a[0][1] * y[1];
                fy[1] = self.a[1][0] * y[0] + self.a[1][1] * y[1];
            }
        }

        let mut rhs = LinearRhs {
            a: [[-2.0, 1.0], [0.5, -3.0]],
        };

        let y = [1.2, -0.7];
        let t = 0.3;
        let mut storage = DenseMatrixStorage::new(2);
        let mut fd = DenseFdJacobian::new(2, FdOptions::default());

        fd.eval(&mut rhs, t, &y, &mut storage);

        let idx = |i: usize, j: usize| j * 2 + i;
        assert!((storage.jac[idx(0, 0)] + 2.0).abs() < 1e-6);
        assert!((storage.jac[idx(0, 1)] - 1.0).abs() < 1e-6);
        assert!((storage.jac[idx(1, 0)] - 0.5).abs() < 1e-6);
        assert!((storage.jac[idx(1, 1)] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_bdf_formula_q1() {
        let f = &BDF_FORMULAS[1];
        assert!((f.beta - 1.0).abs() < 1e-15);
        assert!((f.affine[1] - 1.0).abs() < 1e-15);
    }
    #[test]
    fn test_build_affine_part_q3_scalar() {
        let mut yh = YHistory::new(1, 5, &[10.0]);
        yh.block_mut(0)[0] = 10.0;
        yh.block_mut(1)[0] = 7.0;
        yh.block_mut(2)[0] = 4.0;

        let mut out = vec![0.0; 1];
        build_affine_part(&yh, 3, &mut out);

        let expected = (18.0 / 11.0) * 10.0 - (9.0 / 11.0) * 7.0 + (2.0 / 11.0) * 4.0;
        assert!((out[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn test_predict_nordsieck_constant_solution() {
        let n = 3;
        let q = 4;
        let mut zn = NordsieckHistory::new(n, 5);
        let mut zpred = NordsieckHistory::new(n, 5);

        zn.col_mut(0).copy_from_slice(&[2.0, -1.0, 7.0]);
        zn.fill_zero_from(1);

        predict_nordsieck(&zn, &mut zpred, q);

        assert_eq!(zpred.col(0), &[2.0, -1.0, 7.0]);
        for j in 1..=q {
            assert!(zpred.col(j).iter().all(|x| x.abs() < 1e-15));
        }
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use crate::numerical::LSODE::core::*;
    use crate::numerical::LSODE::dense::DenseBackend;
    use crate::numerical::LSODE::dense::DenseMatrixStorage;
    use crate::numerical::LSODE::jacobian::{DenseAotJac, DenseFdJacobian, JacobianEvaluator};
    use crate::numerical::LSODE::solver::{new_bdf_dense_analytic, new_bdf_dense_fd};

    /// =========================
    /// Example RHS
    /// =========================

    struct Robertson;

    impl OdeRhs for Robertson {
        fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
            let y1 = y[0];
            let y2 = y[1];
            let y3 = y[2];

            fy[0] = -0.04 * y1 + 1.0e4 * y2 * y3;
            fy[1] = 0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2;
            fy[2] = 3.0e7 * y2 * y2;
        }
    }

    /// Optional analytic Jacobian callback
    fn robertson_jac(_t: f64, y: &[f64], jac: &mut [f64], n: usize) {
        assert_eq!(n, 3);
        jac.fill(0.0);

        let y2 = y[1];
        let y3 = y[2];

        let idx = |i: usize, j: usize| -> usize { j * n + i };

        jac[idx(0, 0)] = -0.04;
        jac[idx(0, 1)] = 1.0e4 * y3;
        jac[idx(0, 2)] = 1.0e4 * y2;

        jac[idx(1, 0)] = 0.04;
        jac[idx(1, 1)] = -1.0e4 * y3 - 6.0e7 * y2;
        jac[idx(1, 2)] = -1.0e4 * y2;

        jac[idx(2, 0)] = 0.0;
        jac[idx(2, 1)] = 6.0e7 * y2;
        jac[idx(2, 2)] = 0.0;
    }

    fn _example_usage() -> Result<(), OdeError> {
        let rhs = Robertson;
        let y0 = vec![1.0, 0.0, 0.0];
        let tol = TolMode::Scalar {
            rtol: 1.0e-6,
            atol: 1.0e-10,
        };

        let opts = BdfOptions {
            h0: Some(1.0e-8),
            ..Default::default()
        };

        let mut solver = new_bdf_dense_analytic(rhs, robertson_jac, 0.0, y0, tol, opts)?;
        solver.integrate_to(1.0)?;
        Ok(())
    }

    #[inline]
    fn is_finite_vec(x: &[f64]) -> bool {
        x.iter().all(|v| v.is_finite())
    }

    /// =========================
    /// Unit test RHS examples
    /// =========================

    struct ZeroRhs;

    impl OdeRhs for ZeroRhs {
        fn eval(&mut self, _t: f64, _y: &[f64], fy: &mut [f64]) {
            fy.fill(0.0);
        }
    }

    struct ExpDecay {
        lambda: f64,
    }

    impl OdeRhs for ExpDecay {
        fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
            fy[0] = -self.lambda * y[0];
        }
    }

    fn exp_decay_jac(_t: f64, _y: &[f64], jac: &mut [f64], n: usize) {
        assert_eq!(n, 1);
        jac.fill(0.0);
        jac[0] = -1.0;
    }

    struct StiffDecay {
        lambda: f64,
    }

    impl OdeRhs for StiffDecay {
        fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
            fy[0] = -self.lambda * y[0];
        }
    }

    fn stiff_decay_jac(lambda: f64) -> impl FnMut(f64, &[f64], &mut [f64], usize) {
        move |_t: f64, _y: &[f64], jac: &mut [f64], n: usize| {
            assert_eq!(n, 1);
            jac.fill(0.0);
            jac[0] = -lambda;
        }
    }

    struct Linear2 {
        a11: f64,
        a12: f64,
        a21: f64,
        a22: f64,
    }

    impl OdeRhs for Linear2 {
        fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
            fy[0] = self.a11 * y[0] + self.a12 * y[1];
            fy[1] = self.a21 * y[0] + self.a22 * y[1];
        }
    }

    fn linear2_jac(
        a11: f64,
        a12: f64,
        a21: f64,
        a22: f64,
    ) -> impl FnMut(f64, &[f64], &mut [f64], usize) {
        move |_t: f64, _y: &[f64], jac: &mut [f64], n: usize| {
            assert_eq!(n, 2);
            jac.fill(0.0);
            // column-major
            // col 0
            jac[0] = a11;
            jac[1] = a21;
            // col 1
            jac[2] = a12;
            jac[3] = a22;
        }
    }

    struct DiagonalStiff3;

    impl OdeRhs for DiagonalStiff3 {
        fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
            fy[0] = -1.0 * y[0];
            fy[1] = -100.0 * y[1];
            fy[2] = -10_000.0 * y[2];
        }
    }

    fn diagonal_stiff3_jac(_t: f64, _y: &[f64], jac: &mut [f64], n: usize) {
        assert_eq!(n, 3);
        jac.fill(0.0);
        let idx = |i: usize, j: usize| -> usize { j * n + i };
        jac[idx(0, 0)] = -1.0;
        jac[idx(1, 1)] = -100.0;
        jac[idx(2, 2)] = -10_000.0;
    }

    /// =========================
    /// Unit tests: formulas / history
    /// =========================

    #[test]
    fn test_bdf_formula_q1() {
        let f = &BDF_FORMULAS[1];
        assert_eq!(f.q, 1);
        assert!((f.beta - 1.0).abs() < 1e-15);
        assert!((f.affine[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_bdf_formula_q2() {
        let f = &BDF_FORMULAS[2];
        assert_eq!(f.q, 2);
        assert!((f.beta - 2.0 / 3.0).abs() < 1e-15);
        assert!((f.affine[1] - 4.0 / 3.0).abs() < 1e-15);
        assert!((f.affine[2] + 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_bdf_formula_q5() {
        let f = &BDF_FORMULAS[5];
        assert_eq!(f.q, 5);
        assert!((f.beta - 60.0 / 137.0).abs() < 1e-15);
        assert!((f.affine[1] - 300.0 / 137.0).abs() < 1e-15);
        assert!((f.affine[2] + 300.0 / 137.0).abs() < 1e-15);
        assert!((f.affine[3] - 200.0 / 137.0).abs() < 1e-15);
        assert!((f.affine[4] + 75.0 / 137.0).abs() < 1e-15);
        assert!((f.affine[5] - 12.0 / 137.0).abs() < 1e-15);
    }

    #[test]
    fn test_build_affine_part_q3_scalar() {
        let mut yh = YHistory::new(1, 5, &[10.0]);
        yh.block_mut(0)[0] = 10.0; // y_n
        yh.block_mut(1)[0] = 7.0; // y_{n-1}
        yh.block_mut(2)[0] = 4.0; // y_{n-2}

        let mut out = vec![0.0; 1];
        build_affine_part(&yh, 3, &mut out);

        let expected = (18.0 / 11.0) * 10.0 - (9.0 / 11.0) * 7.0 + (2.0 / 11.0) * 4.0;
        assert!((out[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn test_build_bdf_residual_q2_scalar() {
        let y_trial = [1.25];
        let f_trial = [-0.75];
        let affine = [1.1];
        let h = 0.2;
        let q = 2;
        let mut residual = [0.0];

        build_bdf_residual(&y_trial, &f_trial, &affine, h, q, &mut residual);

        let expected = y_trial[0] - h * (2.0 / 3.0) * f_trial[0] - affine[0];
        assert!((residual[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn test_build_backward_differences_polynomial_square() {
        // Sequence from y_k = k^2 with current n corresponding to k=5:
        // y_n = 25, y_{n-1} = 16, y_{n-2} = 9, y_{n-3} = 4
        let n = 1;
        let qmax = 5;
        let mut yh = YHistory::new(n, qmax, &[25.0]);
        yh.block_mut(0)[0] = 25.0;
        yh.block_mut(1)[0] = 16.0;
        yh.block_mut(2)[0] = 9.0;
        yh.block_mut(3)[0] = 4.0;

        let mut work = vec![0.0; (qmax + 1) * n];
        let mut out = vec![0.0; (qmax + 1) * n];

        build_backward_differences_in_place(&yh, 3, &mut work, &mut out);

        let d0 = out[0];
        let d1 = out[1];
        let d2 = out[2];
        let d3 = out[3];

        assert!((d0 - 25.0).abs() < 1e-14);
        assert!((d1 - 9.0).abs() < 1e-14);
        assert!((d2 - 2.0).abs() < 1e-14);
        assert!(d3.abs() < 1e-14);
    }

    #[test]
    fn test_predict_nordsieck_constant_solution() {
        let n = 3;
        let qmax = 5;
        let q = 4;

        let mut zn = NordsieckHistory::new(n, qmax);
        let mut zpred = NordsieckHistory::new(n, qmax);

        zn.col_mut(0).copy_from_slice(&[2.0, -1.0, 7.0]);
        zn.fill_zero_from(1);

        predict_nordsieck(&zn, &mut zpred, q);

        assert_eq!(zpred.col(0), &[2.0, -1.0, 7.0]);
        for j in 1..=q {
            assert!(zpred.col(j).iter().all(|x| x.abs() < 1e-15));
        }
    }

    #[test]
    fn test_yhistory_push_front() {
        let mut yh = YHistory::new(2, 3, &[1.0, 2.0]);
        yh.block_mut(1).copy_from_slice(&[10.0, 20.0]);
        yh.block_mut(2).copy_from_slice(&[100.0, 200.0]);

        yh.push_front(&[7.0, 8.0]);

        assert_eq!(yh.block(0), &[7.0, 8.0]);
        assert_eq!(yh.block(1), &[1.0, 2.0]);
        assert_eq!(yh.block(2), &[10.0, 20.0]);
        assert_eq!(yh.block(3), &[100.0, 200.0]);
    }

    /// =========================
    /// Unit tests: dense backend
    /// =========================

    #[test]
    fn test_dense_backend_small_linear_system() {
        // A = [[4,1],[2,3]], b = [1,1], x = [0.2,0.2]
        let n = 2;
        let mut backend = DenseBackend::new(n);

        {
            let s = backend.storage_mut();
            s.jac.fill(0.0);

            let idx = |i: usize, j: usize| -> usize { j * n + i };

            // We want system matrix directly, so set gamma = -1 and jac = A - I is awkward.
            // Easier: write A into jac and assemble P = I - gamma*J with gamma = -1, J = A - I.
            // Instead, bypass via jac := -(A - I), gamma = 1 also awkward.
            //
            // Simpler for this test:
            // encode J so that P = I - 1*J = A => J = I - A.
            s.jac[idx(0, 0)] = 1.0 - 4.0; // -3
            s.jac[idx(1, 0)] = 0.0 - 2.0; // -2
            s.jac[idx(0, 1)] = 0.0 - 1.0; // -1
            s.jac[idx(1, 1)] = 1.0 - 3.0; // -2
        }

        backend.assemble_system_matrix(1.0);
        backend.factorize().unwrap();

        let mut rhs = vec![1.0, 1.0];
        backend.solve_in_place(&mut rhs).unwrap();

        assert!((rhs[0] - 0.2).abs() < 1e-12);
        assert!((rhs[1] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_dense_backend_singular_matrix() {
        // Singular A = [[1,2],[2,4]]
        let n = 2;
        let mut backend = DenseBackend::new(n);

        {
            let s = backend.storage_mut();
            let idx = |i: usize, j: usize| -> usize { j * n + i };

            // J = I - A
            s.jac[idx(0, 0)] = 1.0 - 1.0; // 0
            s.jac[idx(1, 0)] = 0.0 - 2.0; // -2
            s.jac[idx(0, 1)] = 0.0 - 2.0; // -2
            s.jac[idx(1, 1)] = 1.0 - 4.0; // -3
        }

        backend.assemble_system_matrix(1.0);
        let res = backend.factorize();
        assert!(matches!(res, Err(OdeError::SingularJacobian)));
    }

    /// =========================
    /// Unit tests: FD Jacobian
    /// =========================

    #[test]
    fn test_dense_fd_matches_linear_jacobian() {
        let n = 2;
        let mut rhs = Linear2 {
            a11: 2.0,
            a12: -1.0,
            a21: 3.5,
            a22: 0.25,
        };

        let y = [1.2, -0.7];
        let t = 0.0;

        let mut storage_fd = DenseMatrixStorage::new(n);
        let mut fd = DenseFdJacobian::new(n, FdOptions::default());
        fd.eval(&mut rhs, t, &y, &mut storage_fd);

        let mut storage_exact = DenseMatrixStorage::new(n);
        let mut jac_cb = linear2_jac(2.0, -1.0, 3.5, 0.25);
        jac_cb(t, &y, &mut storage_exact.jac, n);

        let err = max_abs_diff(&storage_fd.jac, &storage_exact.jac);
        assert!(err < 1e-6, "FD Jacobian too far from exact: err={err:e}");
    }

    #[test]
    fn test_dense_fd_matches_robertson_jacobian_at_nontrivial_point() {
        let n = 3;
        let mut rhs = Robertson;
        let y = [0.7, 1.0e-4, 0.2999];
        let t = 0.0;

        let mut storage_fd = DenseMatrixStorage::new(n);
        let mut fd = DenseFdJacobian::new(
            n,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-8,
            },
        );
        fd.eval(&mut rhs, t, &y, &mut storage_fd);

        let mut storage_exact = DenseMatrixStorage::new(n);
        robertson_jac(t, &y, &mut storage_exact.jac, n);

        let err = max_abs_diff(&storage_fd.jac, &storage_exact.jac);
        assert!(
            err < 5e2,
            "FD Robertson Jacobian mismatch too large: err={err:e}"
        );
        // The system has large coefficients, so use a tolerant smoke threshold here.
    }

    /// =========================
    /// Integration tests: basic
    /// =========================

    #[test]
    fn test_constant_solution() {
        let rhs = ZeroRhs;
        let y0 = vec![3.5];
        let tol = TolMode::Scalar {
            rtol: 1e-8,
            atol: 1e-10,
        };
        let opts = BdfOptions {
            h0: Some(1e-3),
            max_steps: 10_000,
            ..Default::default()
        };

        let jac = |_t: f64, _y: &[f64], jac: &mut [f64], n: usize| {
            assert_eq!(n, 1);
            jac[0] = 0.0;
        };

        let mut solver = new_bdf_dense_analytic(rhs, jac, 0.0, y0, tol, opts).unwrap();
        solver.integrate_to(1.0).unwrap();

        assert!((solver.y()[0] - 3.5).abs() < 1e-8);
        assert!(is_finite_vec(solver.y()));
        assert!(solver.stats().n_accepted > 0);
    }

    #[test]
    fn test_exp_decay_nonstiff() {
        let rhs = ExpDecay { lambda: 1.0 };
        let y0 = vec![1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };
        let opts = BdfOptions {
            h0: Some(1e-3),
            max_steps: 50_000,
            ..Default::default()
        };

        let mut solver = new_bdf_dense_analytic(rhs, exp_decay_jac, 0.0, y0, tol, opts).unwrap();
        solver.integrate_to(1.0).unwrap();

        let expected = (-1.0f64).exp();
        let err = (solver.y()[0] - expected).abs();

        assert!(err < 5e-2, "exp decay error too large: err={err:e}");
        assert!(is_finite_vec(solver.y()));
    }

    #[test]
    fn test_scalar_stiff_decay() {
        let lambda = 1000.0;
        let rhs = StiffDecay { lambda };
        let y0 = vec![1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-12,
        };
        let opts = BdfOptions {
            h0: Some(1e-6),
            max_steps: 200_000,
            ..Default::default()
        };

        let mut solver =
            new_bdf_dense_analytic(rhs, stiff_decay_jac(lambda), 0.0, y0, tol, opts).unwrap();

        solver.integrate_to(1.0).unwrap();

        assert!(solver.y()[0].abs() < 1e-3);
        assert!(is_finite_vec(solver.y()));
        assert!(solver.stats().n_linear_solves > 0);
    }

    #[test]
    fn test_linear_diagonal_stiff_system() {
        let rhs = DiagonalStiff3;
        let y0 = vec![1.0, 1.0, 1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };
        let opts = BdfOptions {
            h0: Some(1e-6),
            max_steps: 300_000,
            ..Default::default()
        };

        let mut solver =
            new_bdf_dense_analytic(rhs, diagonal_stiff3_jac, 0.0, y0, tol, opts).unwrap();

        solver.integrate_to(0.1).unwrap();

        let expected = [(-0.1f64).exp(), (-10.0f64).exp(), (-1000.0f64).exp()];

        assert!((solver.y()[0] - expected[0]).abs() < 1e-2);
        assert!((solver.y()[1] - expected[1]).abs() < 1e-2);
        assert!(solver.y()[2].abs() < 1e-6);
        assert!(is_finite_vec(solver.y()));
    }

    /// =========================
    /// Integration tests: startup / ramp-up
    /// =========================

    #[test]
    fn test_startup_keeps_order_one_initially() {
        let rhs = ExpDecay { lambda: 1.0 };
        let y0 = vec![1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };
        let opts = BdfOptions {
            h0: Some(1e-4),
            max_steps: 10_000,
            ..Default::default()
        };

        let mut solver = new_bdf_dense_analytic(rhs, exp_decay_jac, 0.0, y0, tol, opts).unwrap();

        assert_eq!(solver.core.order, 1);
        assert_eq!(solver.core.order_cap, 1);

        // first accepted step
        loop {
            if solver.try_step().is_ok() {
                break;
            }
        }
        assert_eq!(solver.core.order, 1);

        // second accepted step
        loop {
            if solver.try_step().is_ok() {
                break;
            }
        }
        assert_eq!(solver.core.order, 1);
    }

    #[test]
    fn test_order_cap_respects_available_history() {
        let rhs = ExpDecay { lambda: 1.0 };
        let y0 = vec![1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };
        let opts = BdfOptions {
            h0: Some(1e-4),
            max_steps: 20_000,
            ..Default::default()
        };

        let mut solver = new_bdf_dense_analytic(rhs, exp_decay_jac, 0.0, y0, tol, opts).unwrap();

        for _ in 0..5 {
            loop {
                match solver.try_step() {
                    Ok(()) => break,
                    Err(OdeError::StepRejected) => continue,
                    Err(e) => panic!("unexpected error during stepping: {e}"),
                }
            }

            assert!(solver.core.order_cap <= solver.core.max_order);
            assert!(solver.core.order <= solver.core.order_cap);
            assert!(solver.core.order_cap <= solver.core.n_accept_total + 1);
        }
    }

    /// =========================
    /// Integration tests: Robertson
    /// =========================

    #[test]
    fn test_robertson_mass_conservation_short() {
        let rhs = Robertson;
        let y0 = vec![1.0, 0.0, 0.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-12,
        };
        let opts = BdfOptions {
            h0: Some(1e-8),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut solver = new_bdf_dense_analytic(rhs, robertson_jac, 0.0, y0, tol, opts).unwrap();
        solver.integrate_to(1.0e-2).unwrap();

        let s: f64 = solver.y().iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-4,
            "mass conservation violated: sum={s:.16e}"
        );
        assert!(is_finite_vec(solver.y()));
        assert!(solver.y().iter().all(|v| *v > -1e-8));
    }

    #[test]
    fn test_robertson_analytic_vs_fd_short() {
        let y0 = vec![1.0, 0.0, 0.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-12,
        };

        let opts1 = BdfOptions {
            h0: Some(1e-8),
            max_steps: 500_000,
            ..Default::default()
        };

        let opts2 = BdfOptions {
            h0: Some(1e-8),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut solver_analytic = new_bdf_dense_analytic(
            Robertson,
            robertson_jac,
            0.0,
            y0.clone(),
            tol.clone(),
            opts1,
        )
        .unwrap();

        let mut solver_fd = new_bdf_dense_fd(
            Robertson,
            0.0,
            y0,
            tol,
            opts2,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-8,
            },
        )
        .unwrap();

        solver_analytic.integrate_to(1.0e-2).unwrap();
        solver_fd.integrate_to(1.0e-2).unwrap();

        let err = max_abs_diff(solver_analytic.y(), solver_fd.y());
        assert!(
            err < 5e-3,
            "analytic vs FD Robertson mismatch too large: err={err:e}"
        );

        let s1: f64 = solver_analytic.y().iter().sum();
        let s2: f64 = solver_fd.y().iter().sum();

        assert!((s1 - 1.0).abs() < 1e-4);
        assert!((s2 - 1.0).abs() < 1e-4);
    }

    /// =========================
    /// Slightly longer stiff smoke
    /// =========================

    #[test]
    fn test_robertson_to_one_smoke() {
        let rhs = Robertson;
        let y0 = vec![1.0, 0.0, 0.0];
        let tol = TolMode::Scalar {
            rtol: 1e-5,
            atol: 1e-10,
        };
        let opts = BdfOptions {
            h0: Some(1e-8),
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = new_bdf_dense_analytic(rhs, robertson_jac, 0.0, y0, tol, opts).unwrap();
        solver.integrate_to(1.0).unwrap();

        let y = solver.y();
        let sum: f64 = y.iter().sum();

        assert!(is_finite_vec(y));
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "mass conservation drift too large: {sum:e}"
        );
        assert!(solver.stats().n_accepted > 0);
        assert!(solver.stats().n_linear_solves > 0);
    }

    #[test]
    fn test_robertson_analytic_vs_fd_t1() {
        let y0 = vec![1.0, 0.0, 0.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-12,
        };

        let opts1 = BdfOptions {
            h0: Some(1e-8),
            max_steps: 500_000,
            ..Default::default()
        };

        let opts2 = BdfOptions {
            h0: Some(1e-8),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut solver_analytic = new_bdf_dense_analytic(
            Robertson,
            robertson_jac,
            0.0,
            y0.clone(),
            tol.clone(),
            opts1,
        )
        .unwrap();

        let mut solver_fd = new_bdf_dense_fd(
            Robertson,
            0.0,
            y0,
            tol,
            opts2,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-8,
            },
        )
        .unwrap();

        solver_analytic.integrate_to(1.0).unwrap();
        solver_fd.integrate_to(1.0).unwrap();

        let err = max_abs_diff(solver_analytic.y(), solver_fd.y());
        assert!(
            err < 1e-3,
            "analytic vs FD Robertson mismatch too large: err={err:e}"
        );

        let s1: f64 = solver_analytic.y().iter().sum();
        let s2: f64 = solver_fd.y().iter().sum();
        assert!((s1 - 1.0).abs() < 1e-4);
        assert!((s2 - 1.0).abs() < 1e-4);

        assert!(solver_analytic.stats().n_linear_solves > 0);
        assert!(solver_fd.stats().n_linear_solves > 0);
    }

    #[test]
    fn test_scalar_stiff_decay_analytic_vs_fd() {
        let lambda = 1000.0;
        let y0 = vec![1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-12,
        };

        let opts1 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 200_000,
            ..Default::default()
        };

        let opts2 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 200_000,
            ..Default::default()
        };

        let mut solver_analytic = new_bdf_dense_analytic(
            StiffDecay { lambda },
            stiff_decay_jac(lambda),
            0.0,
            y0.clone(),
            tol.clone(),
            opts1,
        )
        .unwrap();

        let mut solver_fd = new_bdf_dense_fd(
            StiffDecay { lambda },
            0.0,
            y0,
            tol,
            opts2,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-10,
            },
        )
        .unwrap();

        solver_analytic.integrate_to(0.1).unwrap();
        solver_fd.integrate_to(0.1).unwrap();

        let err = max_abs_diff(solver_analytic.y(), solver_fd.y());
        assert!(
            err < 1e-5,
            "analytic vs FD stiff scalar mismatch too large: err={err:e}"
        );
    }

    #[test]
    fn test_diagonal_stiff3_analytic_vs_fd() {
        let y0 = vec![1.0, 1.0, 1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts1 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 300_000,
            ..Default::default()
        };

        let opts2 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 300_000,
            ..Default::default()
        };

        let mut solver_analytic = new_bdf_dense_analytic(
            DiagonalStiff3,
            diagonal_stiff3_jac,
            0.0,
            y0.clone(),
            tol.clone(),
            opts1,
        )
        .unwrap();

        let mut solver_fd = new_bdf_dense_fd(
            DiagonalStiff3,
            0.0,
            y0,
            tol,
            opts2,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-10,
            },
        )
        .unwrap();

        solver_analytic.integrate_to(0.1).unwrap();
        solver_fd.integrate_to(0.1).unwrap();

        let err = max_abs_diff(solver_analytic.y(), solver_fd.y());
        assert!(
            err < 1e-4,
            "analytic vs FD diagonal stiff mismatch too large: err={err:e}"
        );
    }

    #[test]
    fn test_robertson_fd_not_cheaper_than_analytic_in_rhs_calls() {
        let y0 = vec![1.0, 0.0, 0.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-12,
        };

        let opts1 = BdfOptions {
            h0: Some(1e-8),
            max_steps: 500_000,
            ..Default::default()
        };

        let opts2 = BdfOptions {
            h0: Some(1e-8),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut solver_analytic = new_bdf_dense_analytic(
            Robertson,
            robertson_jac,
            0.0,
            y0.clone(),
            tol.clone(),
            opts1,
        )
        .unwrap();

        let mut solver_fd = new_bdf_dense_fd(
            Robertson,
            0.0,
            y0,
            tol,
            opts2,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-8,
            },
        )
        .unwrap();

        solver_analytic.integrate_to(1.0e-2).unwrap();
        solver_fd.integrate_to(1.0e-2).unwrap();

        assert!(
            solver_fd.stats().n_f_evals >= solver_analytic.stats().n_f_evals,
            "FD Jacobian path unexpectedly used fewer RHS evaluations"
        );
    }
}

#[cfg(test)]
mod tests_banded {
    use super::*;
    use crate::numerical::LSODE::banded::{BandedBackend, BandedMatrixStorage};
    use crate::numerical::LSODE::core::*;
    use crate::numerical::LSODE::dense::DenseBackend;

    use crate::numerical::LSODE::jacobian::{DenseAotJac, DenseFdJacobian, JacobianEvaluator};
    use crate::numerical::LSODE::solver::{
        new_bdf_banded_analytic, new_bdf_banded_fd, new_bdf_dense_analytic, new_bdf_dense_fd,
    };
    #[inline]
    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut m = 0.0_f64;
        for i in 0..a.len() {
            m = m.max((a[i] - b[i]).abs());
        }
        m
    }

    struct DiagonalStiff3;

    impl OdeRhs for DiagonalStiff3 {
        fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
            fy[0] = -1.0 * y[0];
            fy[1] = -100.0 * y[1];
            fy[2] = -10_000.0 * y[2];
        }
    }

    fn diagonal_stiff3_jac(_t: f64, _y: &[f64], jac: &mut [f64], n: usize) {
        assert_eq!(n, 3);
        jac.fill(0.0);
        let idx = |i: usize, j: usize| -> usize { j * n + i };
        jac[idx(0, 0)] = -1.0;
        jac[idx(1, 1)] = -100.0;
        jac[idx(2, 2)] = -10_000.0;
    }

    fn diagonal_stiff3_banded_jac(
        _t: f64,
        _y: &[f64],
        jac: &mut [f64],
        n: usize,
        ml: usize,
        mu: usize,
        ldab: usize,
    ) {
        assert_eq!(n, 3);
        assert_eq!(ml, 0);
        assert_eq!(mu, 0);

        jac.fill(0.0);

        let idx = |i: usize, j: usize| -> usize {
            let row = mu + i - j;
            j * ldab + row
        };

        jac[idx(0, 0)] = -1.0;
        jac[idx(1, 1)] = -100.0;
        jac[idx(2, 2)] = -10_000.0;
    }
    #[test]
    fn test_diagonal_stiff3_dense_vs_banded() {
        let y0 = vec![1.0, 1.0, 1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts1 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 300_000,
            ..Default::default()
        };

        let opts2 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 300_000,
            ..Default::default()
        };

        let mut dense_solver = new_bdf_dense_analytic(
            DiagonalStiff3,
            diagonal_stiff3_jac,
            0.0,
            y0.clone(),
            tol.clone(),
            opts1,
        )
        .unwrap();

        let mut band_solver = new_bdf_banded_analytic(
            DiagonalStiff3,
            diagonal_stiff3_banded_jac,
            0,
            0,
            0.0,
            y0,
            tol,
            opts2,
        )
        .unwrap();

        dense_solver.integrate_to(0.1).unwrap();
        band_solver.integrate_to(0.1).unwrap();

        let err = max_abs_diff(dense_solver.y(), band_solver.y());
        assert!(
            err < 1e-10,
            "dense vs banded mismatch too large: err={err:e}"
        );
    }

    #[test]
    fn test_diagonal_stiff3_banded_analytic_vs_fd() {
        let y0 = vec![1.0, 1.0, 1.0];
        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts1 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 300_000,
            ..Default::default()
        };

        let opts2 = BdfOptions {
            h0: Some(1e-6),
            max_steps: 300_000,
            ..Default::default()
        };

        let mut analytic_solver = new_bdf_banded_analytic(
            DiagonalStiff3,
            diagonal_stiff3_banded_jac,
            0,
            0,
            0.0,
            y0.clone(),
            tol.clone(),
            opts1,
        )
        .unwrap();

        let mut fd_solver = new_bdf_banded_fd(
            DiagonalStiff3,
            0,
            0,
            0.0,
            y0,
            tol,
            opts2,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-10,
            },
        )
        .unwrap();

        analytic_solver.integrate_to(0.1).unwrap();
        fd_solver.integrate_to(0.1).unwrap();

        let err = max_abs_diff(analytic_solver.y(), fd_solver.y());
        assert!(
            err < 1e-6,
            "banded analytic vs fd mismatch too large: err={err:e}"
        );
    }

    #[test]
    fn test_tridiag_stiff_dense_vs_banded() {
        let n = 10;
        let alpha = 1000.0;

        let y0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();

        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts_dense = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let opts_banded = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut dense_solver = new_bdf_dense_analytic(
            TridiagStiff { alpha, n },
            tridiag_stiff_dense_jac(alpha),
            0.0,
            y0.clone(),
            tol.clone(),
            opts_dense,
        )
        .unwrap();

        let mut banded_solver = new_bdf_banded_analytic(
            TridiagStiff { alpha, n },
            tridiag_stiff_banded_jac(alpha),
            1, // ml
            1, // mu
            0.0,
            y0,
            tol,
            opts_banded,
        )
        .unwrap();

        dense_solver.integrate_to(1e-2).unwrap();
        banded_solver.integrate_to(1e-2).unwrap();

        let err = max_abs_diff(dense_solver.y(), banded_solver.y());
        assert!(
            err < 1e-8,
            "dense vs banded tridiag mismatch too large: err={err:e}"
        );
    }
    #[test]
    fn test_tridiag_stiff_banded_analytic_vs_fd() {
        let n = 10;
        let alpha = 1000.0;

        let y0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();

        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts_analytic = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let opts_fd = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut analytic_solver = new_bdf_banded_analytic(
            TridiagStiff { alpha, n },
            tridiag_stiff_banded_jac(alpha),
            1,
            1,
            0.0,
            y0.clone(),
            tol.clone(),
            opts_analytic,
        )
        .unwrap();

        let mut fd_solver = new_bdf_banded_fd(
            TridiagStiff { alpha, n },
            1,
            1,
            0.0,
            y0,
            tol,
            opts_fd,
            FdOptions {
                rel_step: 1e-7,
                abs_step: 1e-10,
            },
        )
        .unwrap();

        analytic_solver.integrate_to(1e-2).unwrap();
        fd_solver.integrate_to(1e-2).unwrap();

        let err = max_abs_diff(analytic_solver.y(), fd_solver.y());
        assert!(
            err < 1e-5,
            "banded analytic vs fd tridiag mismatch too large: err={err:e}"
        );
    }
    #[test]
    fn test_tridiag_stiff_dense_vs_banded_larger() {
        let n = 50;
        let alpha = 500.0;

        let y0: Vec<f64> = (0..n).map(|i| ((i + 1) as f64) / (n as f64)).collect();

        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts_dense = BdfOptions {
            h0: Some(1e-6),
            max_steps: 1_000_000,
            ..Default::default()
        };

        let opts_banded = BdfOptions {
            h0: Some(1e-6),
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut dense_solver = new_bdf_dense_analytic(
            TridiagStiff { alpha, n },
            tridiag_stiff_dense_jac(alpha),
            0.0,
            y0.clone(),
            tol.clone(),
            opts_dense,
        )
        .unwrap();

        let mut banded_solver = new_bdf_banded_analytic(
            TridiagStiff { alpha, n },
            tridiag_stiff_banded_jac(alpha),
            1,
            1,
            0.0,
            y0,
            tol,
            opts_banded,
        )
        .unwrap();

        dense_solver.integrate_to(5e-3).unwrap();
        banded_solver.integrate_to(5e-3).unwrap();

        let err = max_abs_diff(dense_solver.y(), banded_solver.y());
        assert!(
            err < 1e-7,
            "dense vs banded larger tridiag mismatch too large: err={err:e}"
        );
    }
    #[test]
    fn test_banded_storage_offsets_tridiagonal() {
        let storage = BandedMatrixStorage::new(5, 1, 1);

        // In-band entries
        assert!(storage.offset(0, 0).is_some());
        assert!(storage.offset(1, 0).is_some());
        assert!(storage.offset(0, 1).is_some());
        assert!(storage.offset(3, 2).is_some());

        // Out-of-band entries
        assert!(storage.offset(2, 0).is_none());
        assert!(storage.offset(0, 2).is_none());
        assert!(storage.offset(4, 2).is_none());
    }

    #[test]
    fn test_banded_lu_recovers_known_solution_tridiagonal() {
        let n = 8;
        let ml = 1;
        let mu = 1;

        let mut backend = BandedBackend::new(n, ml, mu);

        // Build tridiagonal A
        {
            let s = backend.storage_mut();
            s.zero_jac();

            for i in 0..n {
                s.set_jac(i, i, 4.0);
                if i > 0 {
                    s.set_jac(i, i - 1, -1.0);
                }
                if i + 1 < n {
                    s.set_jac(i, i + 1, -1.0);
                }
            }
        }

        backend.assemble_system_matrix(1.0); // system = I - J, so J here is just placeholder algebraically
        {
            // overwrite system directly with A for this unit test
            let s = backend.storage_mut();
            s.zero_system();
            for i in 0..n {
                s.set_system(i, i, 4.0);
                if i > 0 {
                    s.set_system(i, i - 1, -1.0);
                }
                if i + 1 < n {
                    s.set_system(i, i + 1, -1.0);
                }
            }
        }

        let x_true = vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.25];
        let mut b = vec![0.0; n];

        // b = A x_true
        {
            let s = backend.storage();
            for i in 0..n {
                let j_min = i.saturating_sub(ml);
                let j_max = (i + mu).min(n - 1);
                for j in j_min..=j_max {
                    b[i] += s.get_system(i, j) * x_true[j];
                }
            }
        }

        backend.factorize().unwrap();
        backend.solve_in_place(&mut b).unwrap();

        for i in 0..n {
            assert!(
                (b[i] - x_true[i]).abs() < 1e-10,
                "i={i}, got={}, expected={}",
                b[i],
                x_true[i]
            );
        }
    }
}
#[cfg(test)]
mod tests_sparse {
    use super::*;
    use crate::numerical::LSODE::core::*;
    use crate::numerical::LSODE::dense::DenseBackend;
    use crate::numerical::LSODE::sparse::SparseMatrixStorage;

    use crate::numerical::LSODE::jacobian::{DenseAotJac, DenseFdJacobian, JacobianEvaluator};
    use crate::numerical::LSODE::solver::{
        new_bdf_banded_analytic, new_bdf_banded_fd, new_bdf_dense_analytic, new_bdf_dense_fd,
        new_bdf_sparse_analytic,
    };
    pub fn build_tridiag_csc_pattern(n: usize) -> Result<SparseMatrixStorage, OdeError> {
        let mut col_ptrs = Vec::with_capacity(n + 1);
        let mut row_indices = Vec::with_capacity(3 * n.saturating_sub(2) + 2);

        col_ptrs.push(0);

        for j in 0..n {
            if j > 0 {
                row_indices.push(j - 1);
            }

            row_indices.push(j);

            if j + 1 < n {
                row_indices.push(j + 1);
            }

            col_ptrs.push(row_indices.len());
        }

        SparseMatrixStorage::new(n, col_ptrs, row_indices)
    }

    pub fn tridiag_sparse_values(alpha: f64, n: usize) -> impl FnMut(f64, &[f64], &mut [f64]) {
        move |_t: f64, _y: &[f64], vals: &mut [f64]| {
            let mut p = 0;

            for j in 0..n {
                if j > 0 {
                    vals[p] = alpha;
                    p += 1;
                }

                vals[p] = -2.0 * alpha;
                p += 1;

                if j + 1 < n {
                    vals[p] = alpha;
                    p += 1;
                }
            }

            debug_assert_eq!(p, vals.len());
        }
    }

    #[test]
    fn test_sparse_storage_tridiag_pattern_valid() {
        let n = 5;
        let storage = build_tridiag_csc_pattern(n).unwrap();

        assert_eq!(storage.n, 5);
        assert_eq!(storage.col_ptrs.len(), n + 1);
        assert_eq!(storage.diag_pos.len(), n);

        for j in 0..n {
            let p = storage.diag_pos[j];
            assert_eq!(storage.row_indices[p], j);
        }
    }
    #[test]
    fn test_tridiag_stiff_dense_vs_sparse() {
        let n = 10;
        let alpha = 1000.0;

        let y0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();

        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts_dense = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let opts_sparse = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut dense_solver = new_bdf_dense_analytic(
            TridiagStiff { alpha, n },
            tridiag_stiff_dense_jac(alpha),
            0.0,
            y0.clone(),
            tol.clone(),
            opts_dense,
        )
        .unwrap();

        let sparse_storage = build_tridiag_csc_pattern(n).unwrap();

        let mut sparse_solver = new_bdf_sparse_analytic(
            TridiagStiff { alpha, n },
            tridiag_sparse_values(alpha, n),
            sparse_storage,
            0.0,
            y0,
            tol,
            opts_sparse,
        )
        .unwrap();

        dense_solver.integrate_to(1e-2).unwrap();
        sparse_solver.integrate_to(1e-2).unwrap();

        let err = max_abs_diff(dense_solver.y(), sparse_solver.y());
        assert!(
            err < 1e-8,
            "dense vs sparse mismatch too large: err={err:e}"
        );
    }

    #[test]
    fn test_tridiag_stiff_banded_vs_sparse() {
        let n = 10;
        let alpha = 1000.0;

        let y0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();

        let tol = TolMode::Scalar {
            rtol: 1e-6,
            atol: 1e-10,
        };

        let opts_banded = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let opts_sparse = BdfOptions {
            h0: Some(1e-6),
            max_steps: 500_000,
            ..Default::default()
        };

        let mut banded_solver = new_bdf_banded_analytic(
            TridiagStiff { alpha, n },
            tridiag_stiff_banded_jac(alpha),
            1,
            1,
            0.0,
            y0.clone(),
            tol.clone(),
            opts_banded,
        )
        .unwrap();

        let sparse_storage = build_tridiag_csc_pattern(n).unwrap();

        let mut sparse_solver = new_bdf_sparse_analytic(
            TridiagStiff { alpha, n },
            tridiag_sparse_values(alpha, n),
            sparse_storage,
            0.0,
            y0,
            tol,
            opts_sparse,
        )
        .unwrap();

        banded_solver.integrate_to(1e-2).unwrap();
        sparse_solver.integrate_to(1e-2).unwrap();

        let err = max_abs_diff(banded_solver.y(), sparse_solver.y());
        assert!(
            err < 1e-8,
            "banded vs sparse mismatch too large: err={err:e}"
        );
    }
}
