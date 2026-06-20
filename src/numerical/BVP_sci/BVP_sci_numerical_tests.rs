#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_aot::BvpSciGeneratedBackendConfig;
    use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat, faer_mat};
    use crate::numerical::BVP_sci::BVP_sci_numerical::{
        BvpSciNumericalOptions, BvpSciNumericalProblem, NumericalBvpClosureProblem,
        NumericalBvpProblem, NumericalBvpSolveOptions, NumericalJacobianMode, solve_numerical_bvp,
        solve_numerical_bvp_fd, solve_numerical_bvp_with_jacobian,
    };
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
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

    #[test]
    fn numerical_bvp_fd_wrapper_succeeds() {
        let mesh = harmonic_mesh(16);
        let guess = harmonic_initial_guess(&mesh);
        let result = solve_numerical_bvp_fd(
            HarmonicProblem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512),
        )
        .expect("explicit finite-difference wrapper should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
    }

    #[test]
    fn numerical_bvp_analytical_wrapper_succeeds() {
        let mesh = harmonic_mesh(16);
        let guess = harmonic_initial_guess(&mesh);
        let result = solve_numerical_bvp_with_jacobian(
            HarmonicProblemWithJacobian,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512),
        )
        .expect("explicit analytical wrapper should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        assert!(result.rms_residuals.nrows() > 0);
    }

    #[test]
    fn numerical_bvp_sparse_chunking_strategy_can_be_configured() {
        let config = BvpSciGeneratedBackendConfig::default()
            .with_sparse_jacobian_chunking_strategy(SparseChunkingStrategy::ByTargetChunkCount {
                target_chunks: 4,
            });
        assert_eq!(
            config.sparse_jacobian_chunking_strategy,
            SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }
        );
    }

    #[test]
    fn numerical_bvp_closure_fd_adapter_succeeds() {
        let mesh = harmonic_mesh(16);
        let guess = harmonic_initial_guess(&mesh);
        let problem = NumericalBvpClosureProblem::new_fd(
            2,
            0,
            |_, y, _, out| {
                out[0] = y[1];
                out[1] = -y[0];
            },
            |ya, _yb, _p, out| {
                out[0] = ya[0];
                out[1] = ya[1] - 1.0;
            },
        );

        let result = solve_numerical_bvp(
            problem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512),
        )
        .expect("closure-based FD numerical BVP solve should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
    }

    #[test]
    fn numerical_bvp_closure_analytical_adapter_succeeds() {
        let mesh = harmonic_mesh(16);
        let guess = harmonic_initial_guess(&mesh);
        let problem = NumericalBvpClosureProblem::new_with_jacobian(
            2,
            0,
            |_, y, _, out| {
                out[0] = y[1];
                out[1] = -y[0];
            },
            |ya, _yb, _p, out| {
                out[0] = ya[0];
                out[1] = ya[1] - 1.0;
            },
            |_x, _y, _p| {
                let triplets = vec![
                    Triplet::new(0usize, 1usize, 1.0),
                    Triplet::new(1usize, 0usize, -1.0),
                ];
                Some(faer_mat::try_new_from_triplets(2, 2, &triplets).unwrap())
            },
            |_ya, _yb, _p| {
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
            },
        );

        let result = solve_numerical_bvp(
            problem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512),
        )
        .expect("closure-based analytical numerical BVP solve should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
    }

    #[test]
    fn numerical_bvp_closure_parameterized_fd_adapter_succeeds() {
        let mesh = faer_col::from_fn(12, |i| i as f64 / 11.0);
        let guess = faer_dense_mat::from_fn(1, mesh.nrows(), |_, j| mesh[j]);
        let parameters = faer_col::from_fn(1, |_| 1.0);

        let problem = NumericalBvpClosureProblem::new_fd(
            1,
            1,
            |_x, _y, p, out| {
                out[0] = p[0];
            },
            |ya, _yb, p, out| {
                out[0] = ya[0];
                out[1] = p[0] - 1.0;
            },
        )
        .with_rhs_param_jacobian(|_x, _y, _p| None);

        let result = solve_numerical_bvp(
            problem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-7, 256).with_parameters(Some(parameters)),
        )
        .expect("closure-based parameterized FD numerical BVP solve should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        let solved_parameter = result
            .p
            .as_ref()
            .expect("parameter vector should be returned for closure-based parametric solve");
        assert!((solved_parameter[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn numerical_bvp_closure_parameterized_analytical_adapter_succeeds() {
        let mesh = faer_col::from_fn(12, |i| i as f64 / 11.0);
        let guess = faer_dense_mat::from_fn(1, mesh.nrows(), |_, j| mesh[j]);
        let parameters = faer_col::from_fn(1, |_| 1.0);

        let problem = NumericalBvpClosureProblem::new_with_jacobian(
            1,
            1,
            |_x, _y, p, out| {
                out[0] = p[0];
            },
            |ya, _yb, p, out| {
                out[0] = ya[0];
                out[1] = p[0] - 1.0;
            },
            |_x, _y, _p| {
                let triplets = vec![Triplet::new(0usize, 0usize, 0.0)];
                Some(faer_mat::try_new_from_triplets(1, 1, &triplets).unwrap())
            },
            |_ya, _yb, _p| {
                let dya = vec![Triplet::new(0usize, 0usize, 1.0)];
                let dyb = vec![Triplet::new(0usize, 0usize, 0.0)];
                let dp = vec![Triplet::new(1usize, 0usize, 1.0)];
                Some((
                    faer_mat::try_new_from_triplets(1, 1, &dya).unwrap(),
                    faer_mat::try_new_from_triplets(1, 1, &dyb).unwrap(),
                    Some(faer_mat::try_new_from_triplets(2, 1, &dp).unwrap()),
                ))
            },
        );

        let result = solve_numerical_bvp(
            problem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-7, 256).with_parameters(Some(parameters)),
        )
        .expect("closure-based analytical parametric numerical BVP solve should succeed");

        assert!(result.success);
        assert_eq!(result.status, 0);
        let solved_parameter = result.p.as_ref().expect(
            "parameter vector should be returned for closure-based analytical parametric solve",
        );
        assert!((solved_parameter[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn numerical_bvp_public_alias_and_builder_style_succeeds() {
        let mesh = harmonic_mesh(16);
        let guess = harmonic_initial_guess(&mesh);
        let problem = BvpSciNumericalProblem::new_fd(
            2,
            0,
            |_, y, _, out| {
                out[0] = y[1];
                out[1] = -y[0];
            },
            |ya, _yb, _p, out| {
                out[0] = ya[0];
                out[1] = ya[1] - 1.0;
            },
        );

        let options = BvpSciNumericalOptions::new(mesh, guess, 1e-6, 512)
            .with_tolerance(1e-6)
            .with_max_nodes(512)
            .with_mesh_refinement(1e-6, 512)
            .with_verbose(0)
            .with_bc_tol(None);

        let result =
            solve_numerical_bvp(problem, options).expect("alias-based numerical solve should work");

        assert!(result.success);
        assert_eq!(result.status, 0);
    }

    #[test]
    fn singular_term_numerical_route_solves_linear_profile() {
        let mesh = faer_col::from_fn(5, |i| i as f64 / 4.0);
        let guess = faer_dense_mat::zeros(1, mesh.nrows());
        let singular_term = faer_dense_mat::from_fn(1, 1, |_, _| 1.0);

        let problem = BvpSciNumericalProblem::new_fd(
            1,
            0,
            |_x, _y, _p, out| {
                out[0] = 0.0;
            },
            |_ya, yb, _p, out| {
                out[0] = yb[0];
            },
        );

        let result = solve_numerical_bvp(
            problem,
            NumericalBvpSolveOptions::new(mesh.clone(), guess, 1e-8, 64)
                .with_singular_term(Some(singular_term))
                .with_verbose(0),
        )
        .expect("singular-term numerical route should converge");

        assert!(
            result.success,
            "solver reported failure: {}",
            result.message
        );
        assert_eq!(
            result.calc_statistics.get("bvp sci singular term enabled"),
            Some(&1usize)
        );
        for i in 0..result.x.nrows() {
            let actual = result.y[(0, i)];
            assert!(
                actual.abs() < 1e-6,
                "solution mismatch at node {}: expected 0, got {}",
                i,
                actual
            );
        }
    }
}
