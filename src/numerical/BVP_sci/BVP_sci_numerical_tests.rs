#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat, faer_mat};
    use crate::numerical::BVP_sci::BVP_sci_numerical::{
        NumericalBvpProblem, NumericalBvpSolveOptions, NumericalJacobianMode, solve_numerical_bvp,
    };
    use faer::sparse::Triplet;

    struct HarmonicProblem;

    impl NumericalBvpProblem for HarmonicProblem {
        fn dimension(&self) -> usize {
            2
        }

        fn rhs(&self, _x: f64, y: &[f64], _p: &[f64], out: &mut [f64]) {
            out[0] = y[1];
            out[1] = -y[0];
        }

        fn boundary_residual(&self, ya: &[f64], _yb: &[f64], _p: &[f64], out: &mut [f64]) {
            out[0] = ya[0];
            out[1] = ya[1] - 1.0;
        }
    }

    struct HarmonicProblemWithJacobian;

    impl NumericalBvpProblem for HarmonicProblemWithJacobian {
        fn dimension(&self) -> usize {
            2
        }

        fn rhs(&self, _x: f64, y: &[f64], _p: &[f64], out: &mut [f64]) {
            out[0] = y[1];
            out[1] = -y[0];
        }

        fn boundary_residual(&self, ya: &[f64], _yb: &[f64], _p: &[f64], out: &mut [f64]) {
            out[0] = ya[0];
            out[1] = ya[1] - 1.0;
        }

        fn jacobian_mode(&self) -> NumericalJacobianMode {
            NumericalJacobianMode::AnalyticalPointwise
        }

        fn rhs_jacobian(&self, _x: f64, _y: &[f64], _p: &[f64]) -> Option<faer_mat> {
            let triplets = vec![
                Triplet::new(0usize, 1usize, 1.0),
                Triplet::new(1usize, 0usize, -1.0),
            ];
            Some(faer_mat::try_new_from_triplets(2, 2, &triplets).unwrap())
        }

        fn boundary_jacobian(
            &self,
            _ya: &[f64],
            _yb: &[f64],
            _p: &[f64],
        ) -> Option<(faer_mat, faer_mat, Option<faer_mat>)> {
            let dya = vec![
                Triplet::new(0usize, 0usize, 1.0),
                Triplet::new(1usize, 1usize, 1.0),
            ];
            let dyb: Vec<Triplet<usize, usize, f64>> = Vec::new();
            Some((
                faer_mat::try_new_from_triplets(2, 2, &dya).unwrap(),
                faer_mat::try_new_from_triplets(2, 2, &dyb).unwrap(),
                None,
            ))
        }
    }

    fn harmonic_mesh(n_steps: usize) -> faer_col {
        faer_col::from_fn(n_steps, |i| {
            std::f64::consts::PI * i as f64 / (n_steps.saturating_sub(1)) as f64
        })
    }

    fn harmonic_initial_guess(mesh: &faer_col) -> faer_dense_mat {
        faer_dense_mat::from_fn(2, mesh.nrows(), |i, j| match i {
            0 => mesh[j].sin(),
            1 => mesh[j].cos(),
            _ => 0.0,
        })
    }

    struct ParametricLinearProblem;

    impl NumericalBvpProblem for ParametricLinearProblem {
        fn dimension(&self) -> usize {
            1
        }

        fn parameter_dimension(&self) -> usize {
            1
        }

        fn rhs(&self, _x: f64, _y: &[f64], p: &[f64], out: &mut [f64]) {
            out[0] = p[0];
        }

        fn boundary_residual(&self, ya: &[f64], yb: &[f64], p: &[f64], out: &mut [f64]) {
            out[0] = ya[0];
            let _ = yb;
            out[1] = p[0] - 1.0;
        }
    }

    #[test]
    fn numerical_bvp_solve_without_symbolics_succeeds() {
        let mesh = harmonic_mesh(16);
        let guess = harmonic_initial_guess(&mesh);
        let result = solve_numerical_bvp(
            HarmonicProblem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512),
        )
        .expect("pure numerical BVP solve should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        assert!(result.y.get(0, 0).abs() < 1e-6);
        assert!((result.y.get(1, 0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn numerical_bvp_solve_with_pointwise_jacobians_succeeds() {
        let mesh = harmonic_mesh(16);
        let guess = harmonic_initial_guess(&mesh);
        let result = solve_numerical_bvp(
            HarmonicProblemWithJacobian,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512),
        )
        .expect("pure numerical BVP solve with analytical pointwise Jacobians should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        assert!(result.rms_residuals.nrows() > 0);
    }

    #[test]
    fn numerical_bvp_solve_with_parameters_succeeds() {
        let mesh = faer_col::from_fn(12, |i| i as f64 / 11.0);
        let guess = faer_dense_mat::from_fn(1, mesh.nrows(), |_, j| mesh[j]);
        let parameters = faer_col::from_fn(1, |_| 1.0);

        let result = solve_numerical_bvp(
            ParametricLinearProblem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-7, 256).with_parameters(Some(parameters)),
        )
        .expect("pure numerical BVP solve with parameters should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        assert!((result.y.get(0, 0) - 0.0).abs() < 1e-8);
        assert!((result.y.get(0, result.y.ncols() - 1) - 1.0).abs() < 1e-5);
        let solved_parameter = result
            .p
            .as_ref()
            .expect("parameter vector should be returned");
        assert!((solved_parameter[0] - 1.0).abs() < 1e-6);
    }
}
