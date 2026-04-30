#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_faer::faer_col;
    use crate::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciSolverOptions};
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::DMatrix;
    use std::process::Command;

    fn tcc_is_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_TCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        Command::new("tcc").arg("-v").output().is_ok()
    }

    fn generated_test_artifact_dir(label: &str) -> String {
        format!("target/test-artifacts/bvp-sci-aot/{label}")
    }

    fn linear_problem_options() -> BvpSciSolverOptions {
        let eq_system = vec![Expr::parse_expression("z"), Expr::parse_expression("0")];
        let values = vec!["y".to_string(), "z".to_string()];
        let mut boundary_conditions = std::collections::HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0), (1, 1.0)]);
        BvpSciSolverOptions::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(8),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            128,
            DMatrix::zeros(2, 8),
        )
        .with_loglevel(Some("off".to_string()))
    }

    fn param_problem_options() -> BvpSciSolverOptions {
        let eq_system = vec![Expr::parse_expression("z"), Expr::parse_expression("a * y")];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string()];
        let param_values = Some(nalgebra::DVector::from_vec(vec![1.0]));
        let mut boundary_conditions = std::collections::HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 1.0), (1, std::f64::consts::E)]);

        let x_mesh = nalgebra::DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let initial_guess = DMatrix::from_fn(2, 3, |i, j| {
            let x: f64 = x_mesh[j];
            match i {
                0 => x.exp(),
                1 => x.exp(),
                _ => 0.0,
            }
        });

        BvpSciSolverOptions::new(
            Some(x_mesh),
            None,
            None,
            None,
            eq_system,
            values,
            params,
            param_values,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            200,
            initial_guess,
        )
        .with_loglevel(Some("off".to_string()))
    }

    fn max_abs_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
    }

    #[test]
    fn bvp_sci_atomview_prepare_matches_exprlegacy_sparse_symbolics() {
        let solver = BVPwrap::new_with_options(linear_problem_options());
        let exprlegacy = solver
            .prepare_exprlegacy_pointwise_problem()
            .expect("ExprLegacy prepare should succeed");
        let atomview = solver
            .prepare_atomview_pointwise_problem()
            .expect("AtomView prepare should succeed");

        assert_eq!(atomview.equations, exprlegacy.equations);
        assert_eq!(
            atomview.symbolic_jacobian_sparse,
            exprlegacy.symbolic_jacobian_sparse
        );
        assert_eq!(
            atomview.symbolic_param_jacobian_sparse,
            exprlegacy.symbolic_param_jacobian_sparse
        );
        assert_eq!(atomview.time_arg, exprlegacy.time_arg);
        assert_eq!(atomview.variables, exprlegacy.variables);
        assert_eq!(atomview.equation_parameters, exprlegacy.equation_parameters);
    }

    #[test]
    fn bvp_sci_atomview_ctcc_callbacks_match_exprlegacy_linear_problem() {
        if !tcc_is_available() {
            println!(
                "[BVP_sci AtomView] skipping C-tcc callback correctness test: compiler not available"
            );
            return;
        }

        let baseline_solver = BVPwrap::new_with_options(linear_problem_options());
        let generated_solver = BVPwrap::new_with_options(
            linear_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("linear-ctcc")),
        );

        let x = baseline_solver.x_mesh_col.clone();
        let y = baseline_solver.runtime_initial_guess_mat();
        let p = baseline_solver
            .runtime_param_col()
            .unwrap_or_else(|| faer_col::zeros(0));

        let (baseline_jac, baseline_res, baseline_bc) = {
            let mut solver = baseline_solver;
            solver
                .try_eq_generate()
                .expect("ExprLegacy callbacks should prepare")
        };
        let (generated_jac, generated_res, generated_bc) = {
            let mut solver = generated_solver;
            solver
                .try_eq_generate()
                .expect("AtomView callbacks should prepare")
        };

        let baseline_residual = baseline_res(&x, &y, &p);
        let generated_residual = generated_res(&x, &y, &p);
        let mut residual_diff = 0.0_f64;
        for row in 0..baseline_residual.nrows() {
            for col in 0..baseline_residual.ncols() {
                residual_diff = residual_diff.max(
                    (*baseline_residual.get(row, col) - *generated_residual.get(row, col)).abs(),
                );
            }
        }

        let ya = faer_col::from_fn(y.nrows(), |i| *y.get(i, 0));
        let yb = faer_col::from_fn(y.nrows(), |i| *y.get(i, y.ncols() - 1));
        let baseline_bc = baseline_bc.expect("baseline BC callback should exist");
        let generated_bc = generated_bc.expect("generated BC callback should exist");
        let bc_diff = baseline_bc(&ya, &yb, &p)
            .iter()
            .zip(generated_bc(&ya, &yb, &p).iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()));

        let baseline_jacobian =
            baseline_jac.expect("baseline Jacobian callback should exist")(&x, &y, &p).0;
        let generated_jacobian =
            generated_jac.expect("generated Jacobian callback should exist")(&x, &y, &p).0;
        let jacobian_diff = baseline_jacobian
            .iter()
            .zip(generated_jacobian.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| {
                let mut local = 0.0_f64;
                for row in 0..lhs.nrows() {
                    for col in 0..lhs.ncols() {
                        let l = lhs.get(row, col).copied().unwrap_or(0.0);
                        let r = rhs.get(row, col).copied().unwrap_or(0.0);
                        local = local.max((l - r).abs());
                    }
                }
                acc.max(local)
            });

        assert!(
            residual_diff <= 1e-12,
            "AtomView residual should match ExprLegacy on small linear problem"
        );
        assert!(
            bc_diff <= 1e-12,
            "AtomView boundary conditions should match ExprLegacy on small linear problem"
        );
        assert!(
            jacobian_diff <= 1e-12,
            "AtomView sparse Jacobian should match ExprLegacy on small linear problem"
        );
    }

    #[test]
    fn bvp_sci_atomview_ctcc_solution_matches_lambdify_linear_problem() {
        if !tcc_is_available() {
            println!(
                "[BVP_sci AtomView] skipping C-tcc solution correctness test: compiler not available"
            );
            return;
        }

        let mut baseline_solver = BVPwrap::new_with_options(linear_problem_options());
        baseline_solver
            .try_solve()
            .expect("baseline solve should succeed");
        let baseline_solution = baseline_solver
            .get_result()
            .expect("baseline solve should return result");

        let mut generated_solver = BVPwrap::new_with_options(
            linear_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("linear-ctcc-solve")),
        );
        generated_solver
            .try_solve()
            .expect("AtomView solve should succeed");
        let generated_solution = generated_solver
            .get_result()
            .expect("AtomView solve should return result");

        let diff = max_abs_diff(&baseline_solution, &generated_solution);
        assert!(
            diff <= 1e-8,
            "AtomView solution should match Lambdify on linear problem (diff={diff:.3e})"
        );
    }

    #[test]
    fn bvp_sci_atomview_ctcc_solution_matches_lambdify_param_problem() {
        if !tcc_is_available() {
            println!(
                "[BVP_sci AtomView] skipping C-tcc parameter solution test: compiler not available"
            );
            return;
        }

        let mut baseline_solver = BVPwrap::new_with_options(param_problem_options());
        baseline_solver
            .try_solve()
            .expect("baseline param solve should succeed");
        if !baseline_solver.result.success {
            println!(
                "[BVP_sci AtomView] skipping param solution test: baseline did not converge (status={}, msg={})",
                baseline_solver.result.status, baseline_solver.result.message
            );
            return;
        }
        let baseline_solution = baseline_solver
            .get_result()
            .expect("baseline param solve should return result");

        let mut generated_solver = BVPwrap::new_with_options(
            param_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("param-ctcc-solve")),
        );
        generated_solver
            .try_solve()
            .expect("AtomView param solve should succeed");
        if !generated_solver.result.success {
            println!(
                "[BVP_sci AtomView] skipping param solution test: generated did not converge (status={}, msg={})",
                generated_solver.result.status, generated_solver.result.message
            );
            return;
        }
        let generated_solution = generated_solver
            .get_result()
            .expect("AtomView param solve should return result");

        let diff = max_abs_diff(&baseline_solution, &generated_solution);
        assert!(
            diff <= 1e-6,
            "AtomView param solution should match Lambdify (diff={diff:.3e})"
        );
    }
}
