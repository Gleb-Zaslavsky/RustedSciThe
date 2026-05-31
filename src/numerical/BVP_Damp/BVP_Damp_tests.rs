#[cfg(test)]
mod tests {

    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        AdaptiveGridConfig, DampedSolverOptions, NRBVP as NRBDVPd, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildPolicy;
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::numerical::Examples_and_utils::NonlinEquation;
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::{
        BvpBackendIntegrationError, BvpSymbolicAssemblyBackend,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    use strum::IntoEnumIterator;

    use nalgebra::{DMatrix, DVector};

    fn linear_bvp_fixture(
        n_steps: usize,
        options: DampedSolverOptions,
    ) -> (NRBDVPd, f64, f64, usize) {
        // y'' = 0 with y(0)=0, y(1)=1  =>  y(x)=x, z(x)=1
        let eq_system = vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")];
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;

        // Start from the analytical profile to remove initialization artifacts from backend checks.
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            let y_exact = x;
            let z_exact = 1.0;
            guess[i * values.len()] = y_exact;
            guess[i * values.len() + 1] = z_exact;
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let border_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0f64), (1usize, 1.0f64)])]);

        let solver = NRBDVPd::new_with_options(
            eq_system,
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
        );
        (solver, t0, t_end, n_steps)
    }

    fn linear_bvp_numeric_fixture(
        n_steps: usize,
        options: DampedSolverOptions,
    ) -> (NRBDVPd, f64, f64, usize) {
        // y'' = 0 represented as y' = z, z' = 0 with one BC per variable:
        // y(0)=0, z(1)=1  =>  y(x)=x, z(x)=1
        // Numeric-only route: runtime residual source of truth is the RHS closure.
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;

        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = x;
            guess[i * values.len() + 1] = 1.0;
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(1usize, 1.0f64)]),
        ]);

        let solver = NRBDVPd::new_numeric_fd_with_options(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            |_x, y: &DVector<f64>, _params| DVector::from_vec(vec![y[1], 0.0]),
        );
        (solver, t0, t_end, n_steps)
    }

    fn oscillator_bvp_numeric_fixture(
        n_steps: usize,
        options: DampedSolverOptions,
    ) -> (NRBDVPd, f64, f64, usize) {
        // y'' + y = 0 represented as y' = z, z' = -y
        // with y(0)=0 and z(0)=1 -> y(x)=sin(x), z(x)=cos(x)
        // Numeric-only route: runtime residual source of truth is the RHS closure.
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        let t_end = std::f64::consts::FRAC_PI_2;
        let h = (t_end - t0) / n_steps as f64;

        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = x.sin();
            guess[i * values.len() + 1] = x.cos();
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(0usize, 1.0f64)]),
        ]);

        let solver = NRBDVPd::new_numeric_fd_with_options(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            |_x, y: &DVector<f64>, _params| DVector::from_vec(vec![y[1], -y[0]]),
        );
        (solver, t0, t_end, n_steps)
    }

    fn oscillator_bvp_numeric_two_y_bc_fixture(
        n_steps: usize,
        options: DampedSolverOptions,
    ) -> (NRBDVPd, f64, f64, usize) {
        // y'' + y = 0 represented as y' = z, z' = -y.
        // This is the natural two-point BVP form: y(0)=0, y(pi/2)=1.
        // The derivative variable z has no direct boundary condition.
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        let t_end = std::f64::consts::FRAC_PI_2;
        let h = (t_end - t0) / n_steps as f64;

        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = x.sin();
            guess[i * values.len() + 1] = x.cos();
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let border_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0f64), (1usize, 1.0f64)])]);

        let solver = NRBDVPd::new_numeric_with_jacobian_options(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            |_x, y: &DVector<f64>, _params| DVector::from_vec(vec![y[1], -y[0]]),
            |_x, _y: &DVector<f64>, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]),
        );
        (solver, t0, t_end, n_steps)
    }

    fn oscillator_bvp_symbolic_fixture(
        n_steps: usize,
        options: DampedSolverOptions,
    ) -> (NRBDVPd, f64, f64, usize) {
        // Symbolic route: y'' + y = 0 represented as y' = z, z' = -y
        // with y(0)=0 and z(0)=1 -> y(x)=sin(x), z(x)=cos(x)
        let eq_system = vec![Expr::parse_expression("z"), Expr::parse_expression("-y")];
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        let t_end = std::f64::consts::FRAC_PI_2;
        let h = (t_end - t0) / n_steps as f64;

        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = x.sin();
            guess[i * values.len() + 1] = x.cos();
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(0usize, 1.0f64)]),
        ]);

        let solver = NRBDVPd::new_with_options(
            eq_system,
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
        );
        (solver, t0, t_end, n_steps)
    }

    fn stiff_decay_bvp_numeric_fixture(
        n_steps: usize,
        lambda_fast: f64,
        lambda_slow: f64,
        options: DampedSolverOptions,
    ) -> (NRBDVPd, f64, f64, usize) {
        // Stiff diagonal two-rate system:
        // y' = -lambda_fast * y, z' = -lambda_slow * z
        // with y(0)=1, z(0)=1
        // exact: y(x)=exp(-lambda_fast x), z(x)=exp(-lambda_slow x)
        // Numeric-only route: runtime residual source of truth is the RHS closure.

        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;

        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            let y_exact = (-lambda_fast * x).exp();
            let z_exact = (-lambda_slow * x).exp();
            guess[i * values.len()] = y_exact;
            guess[i * values.len() + 1] = z_exact;
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 1.0f64)]),
            ("z".to_string(), vec![(0usize, 1.0f64)]),
        ]);

        let solver = NRBDVPd::new_numeric_fd_with_options(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            move |_x, state: &DVector<f64>, _params| {
                DVector::from_vec(vec![-lambda_fast * state[0], -lambda_slow * state[1]])
            },
        );
        (solver, t0, t_end, n_steps)
    }

    fn stiff_decay_bvp_numeric_jacobian_fixture(
        n_steps: usize,
        lambda_fast: f64,
        lambda_slow: f64,
        options: DampedSolverOptions,
    ) -> (NRBDVPd, f64, f64, usize) {
        // Same problem as `stiff_decay_bvp_numeric_fixture`, but with an
        // analytical continuous RHS Jacobian supplied by the user.
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;

        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = (-lambda_fast * x).exp();
            guess[i * values.len() + 1] = (-lambda_slow * x).exp();
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());

        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 1.0f64)]),
            ("z".to_string(), vec![(0usize, 1.0f64)]),
        ]);

        let solver = NRBDVPd::new_numeric_with_jacobian_options(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            move |_x, state: &DVector<f64>, _params| {
                DVector::from_vec(vec![-lambda_fast * state[0], -lambda_slow * state[1]])
            },
            move |_x, _state: &DVector<f64>, _params| {
                DMatrix::from_row_slice(2, 2, &[-lambda_fast, 0.0, 0.0, -lambda_slow])
            },
        );
        (solver, t0, t_end, n_steps)
    }

    fn base_linear_bvp_options() -> DampedSolverOptions {
        let bounds = HashMap::from([
            ("y".to_string(), (-0.2, 1.2)),
            ("z".to_string(), (-2.0, 2.0)),
        ]);
        let rel_tolerance = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);
        DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(40)
            .with_bounds(bounds)
    }

    fn assert_linear_bvp_solution_quality(
        solver: &NRBDVPd,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        rms_tol: f64,
        max_abs_tol: f64,
    ) {
        let solution = solver
            .get_result()
            .expect("BVP solver should store full solution matrix")
            .clone();
        let (rows, cols) = solution.shape();
        assert_eq!(rows, n_steps + 1);
        assert!(cols >= 1);

        let h = (t_end - t0) / n_steps as f64;
        let mut sq_sum = 0.0;
        let mut max_abs = 0.0;
        for i in 0..rows {
            let x = t0 + (i as f64) * h;
            let y_exact = x;
            let y_num = solution[(i, 0)];
            assert!(y_num.is_finite(), "solution contains non-finite values");
            let err = (y_num - y_exact).abs();
            sq_sum += err * err;
            if err > max_abs {
                max_abs = err;
            }
        }
        let rms = (sq_sum / rows as f64).sqrt();
        assert!(
            rms <= rms_tol,
            "RMS error too large: rms={rms:e}, tol={rms_tol:e}"
        );
        assert!(
            max_abs <= max_abs_tol,
            "Max abs error too large: max_abs={max_abs:e}, tol={max_abs_tol:e}"
        );
    }

    fn assert_oscillator_bvp_solution_quality(
        solver: &NRBDVPd,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        rms_tol: f64,
        max_abs_tol: f64,
    ) {
        let solution = solver
            .get_result()
            .expect("BVP solver should store full solution matrix")
            .clone();
        let rows = solution.nrows();
        let h = (t_end - t0) / n_steps as f64;
        let mut sq_sum = 0.0;
        let mut max_abs = 0.0;
        for i in 0..rows {
            let x = t0 + (i as f64) * h;
            let y_exact = x.sin();
            let y_num = solution[(i, 0)];
            assert!(y_num.is_finite(), "solution contains non-finite values");
            let err = (y_num - y_exact).abs();
            sq_sum += err * err;
            if err > max_abs {
                max_abs = err;
            }
        }
        let rms = (sq_sum / rows as f64).sqrt();
        assert!(
            rms <= rms_tol,
            "oscillator RMS error too large: rms={rms:e}, tol={rms_tol:e}"
        );
        assert!(
            max_abs <= max_abs_tol,
            "oscillator max abs error too large: max_abs={max_abs:e}, tol={max_abs_tol:e}"
        );
    }

    fn assert_stiff_decay_solution_quality(
        solver: &NRBDVPd,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        lambda_fast: f64,
        lambda_slow: f64,
        rms_tol: f64,
        max_abs_tol: f64,
    ) {
        let solution = solver
            .get_result()
            .expect("BVP solver should store full solution matrix")
            .clone();
        let rows = solution.nrows();
        let h = (t_end - t0) / n_steps as f64;
        let mut sq_sum = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for i in 0..rows {
            let x = t0 + (i as f64) * h;
            let y_exact = (-lambda_fast * x).exp();
            let z_exact = (-lambda_slow * x).exp();
            let y_num = solution[(i, 0)];
            let z_num = solution[(i, 1)];
            assert!(y_num.is_finite(), "solution contains non-finite values");
            assert!(z_num.is_finite(), "solution contains non-finite values");
            let y_err = (y_num - y_exact).abs();
            let z_err = (z_num - z_exact).abs();
            sq_sum += y_err * y_err + z_err * z_err;
            max_abs = max_abs.max(y_err.max(z_err));
        }
        let rms = (sq_sum / (2.0 * rows as f64)).sqrt();
        assert!(
            rms <= rms_tol,
            "stiff decay RMS error too large: rms={rms:e}, tol={rms_tol:e}"
        );
        assert!(
            max_abs <= max_abs_tol,
            "stiff decay max abs error too large: max_abs={max_abs:e}, tol={max_abs_tol:e}"
        );
    }

    fn assert_stiff_coupled_solution_quality(
        solver: &NRBDVPd,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        lambda_fast: f64,
        lambda_slow: f64,
        coupling: f64,
        rms_tol: f64,
        max_abs_tol: f64,
    ) {
        let solution = solver
            .get_result()
            .expect("BVP solver should store full solution matrix")
            .clone();
        let rows = solution.nrows();
        let h = (t_end - t0) / n_steps as f64;
        let mut sq_sum = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for i in 0..rows {
            let x = t0 + (i as f64) * h;
            let y_exact = (-lambda_fast * x).exp();
            let z_exact = (-lambda_slow * x).exp()
                + coupling * ((-lambda_fast * x).exp() - (-lambda_slow * x).exp())
                    / (lambda_slow - lambda_fast);
            let y_num = solution[(i, 0)];
            let z_num = solution[(i, 1)];
            assert!(y_num.is_finite(), "solution contains non-finite values");
            assert!(z_num.is_finite(), "solution contains non-finite values");
            let y_err = (y_num - y_exact).abs();
            let z_err = (z_num - z_exact).abs();
            sq_sum += y_err * y_err + z_err * z_err;
            max_abs = max_abs.max(y_err.max(z_err));
        }
        let rms = (sq_sum / (2.0 * rows as f64)).sqrt();
        assert!(
            rms <= rms_tol,
            "stiff coupled RMS error too large: rms={rms:e}, tol={rms_tol:e}"
        );
        assert!(
            max_abs <= max_abs_tol,
            "stiff coupled max abs error too large: max_abs={max_abs:e}, tol={max_abs_tol:e}"
        );
    }

    fn is_aot_environment_issue(err: &BvpBackendIntegrationError) -> bool {
        match err {
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
            | BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { .. } => true,
            BvpBackendIntegrationError::AutomaticAotBuildFailed { message, .. } => {
                let msg = message.to_ascii_lowercase();
                msg.contains("permission denied")
                    || msg.contains("not found")
                    || msg.contains("failed to spawn")
                    || msg.contains("status=some(1)")
                    || msg.contains("toolchain")
            }
            BvpBackendIntegrationError::PipelinePanicked(message) => {
                let msg = message.to_ascii_lowercase();
                msg.contains("generatedbackendfailure")
                    || msg.contains("permission denied")
                    || msg.contains("failed to spawn")
                    || msg.contains("status=some(1)")
                    || msg.contains("toolchain")
            }
            _ => false,
        }
    }
    #[test]
    fn test_BVP_Damp1() {
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-5;
        let max_iterations = 20;

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let strategy_params = Some(SolverParams::default());
        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
        let Bounds = HashMap::from([
            ("z".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-7.0, 7.0)),
        ]);
        let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
        assert_eq!(&eq_system.len(), &2);
        let options = DampedSolverOptions::dense_damped()
            .with_strategy_params(strategy_params)
            .with_abs_tolerance(tolerance)
            .with_rel_tolerance(rel_tolerance)
            .with_max_iterations(max_iterations)
            .with_bounds(Bounds);
        let mut nr = NRBDVPd::new_with_options(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            options,
        );

        println!("solving system");
        nr.try_solve()
            .expect("dense damped BVP should solve through the fallible API");
        let solution = nr.get_result().unwrap();
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps + 1);
        // println!("result = {:?}", solution);
        nr.gnuplot_result();
    }

    #[test]
    fn bvp_banded_atomview_lambdify_two_point_end_to_end_correctness() {
        let options = base_linear_bvp_options()
            .with_banded_lambdify()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
        let (mut solver, t0, t_end, n_steps) = linear_bvp_fixture(80, options);
        solver.dont_save_log(true);

        solver
            .try_solve()
            .expect("banded AtomView lambdify BVP should solve end-to-end");
        assert_linear_bvp_solution_quality(&solver, t0, t_end, n_steps, 1e-5, 5e-5);
    }

    #[test]
    fn bvp_banded_atomview_aot_two_point_end_to_end_correctness() {
        let options = base_linear_bvp_options().with_banded_atomview_c_tcc();
        let (mut solver, t0, t_end, n_steps) = linear_bvp_fixture(80, options);
        solver.dont_save_log(true);

        match solver.try_solve() {
            Ok(_) => {
                assert_linear_bvp_solution_quality(&solver, t0, t_end, n_steps, 1e-5, 5e-5);
            }
            Err(err) if is_aot_environment_issue(&err) => {
                eprintln!(
                    "Skipping AOT correctness assertion due to environment/toolchain issue: {err:?}"
                );
            }
            Err(err) => panic!("banded AtomView AOT BVP solve failed unexpectedly: {err:?}"),
        }
    }

    #[test]
    fn bvp_pure_numerical_backend_two_point_end_to_end_correctness() {
        let options = base_linear_bvp_options().with_sparse_generated_backend_defaults();
        let (mut solver, t0, t_end, n_steps) = linear_bvp_numeric_fixture(80, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
            DVector::from_vec(vec![y[1], 0.0])
        })));

        solver.try_solve().expect(
            "numeric-only sparse BVP path with closure discretization should solve end-to-end",
        );
        assert_linear_bvp_solution_quality(&solver, t0, t_end, n_steps, 1e-5, 5e-5);
    }

    #[test]
    fn bvp_banded_pure_numerical_backend_two_point_end_to_end_correctness() {
        let options = base_linear_bvp_options().with_banded_generated_backend_defaults();
        let (mut solver, t0, t_end, n_steps) = linear_bvp_numeric_fixture(80, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
            DVector::from_vec(vec![y[1], 0.0])
        })));

        solver.try_solve().expect(
            "numeric-only banded BVP path with closure discretization should solve end-to-end",
        );
        assert_linear_bvp_solution_quality(&solver, t0, t_end, n_steps, 1e-5, 5e-5);
    }

    #[test]
    fn bvp_pure_numerical_wrapper_with_user_jacobian_solves_end_to_end() {
        let n_steps = 80;
        let options = base_linear_bvp_options().with_sparse_generated_backend_defaults();
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + i as f64 * h;
            guess[i * values.len()] = x;
            guess[i * values.len() + 1] = 1.0;
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());
        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(1usize, 1.0f64)]),
        ]);

        let mut solver = NRBDVPd::new_numeric_with_jacobian_options(
            initial_guess,
            values,
            "x".to_string(),
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            |_x, y: &DVector<f64>, _params| DVector::from_vec(vec![y[1], 0.0]),
            |_x, _y: &DVector<f64>, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]),
        );
        solver.dont_save_log(true);

        assert!(solver.has_numeric_rhs());
        assert!(solver.has_numeric_jacobian());
        solver
            .try_eq_generate(None, None)
            .expect("numeric wrapper with user Jacobian should generate callbacks");
        assert!(
            solver.jac.is_some(),
            "user numeric Jacobian should install a global discretized Jacobian callback"
        );

        solver
            .try_solve()
            .expect("numeric wrapper with user Jacobian should solve end-to-end");
        assert_linear_bvp_solution_quality(&solver, t0, t_end, n_steps, 1e-8, 1e-7);
    }

    #[test]
    fn bvp_banded_pure_numerical_wrapper_with_user_jacobian_solves_end_to_end() {
        let n_steps = 80;
        let options = base_linear_bvp_options().with_banded_generated_backend_defaults();
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + i as f64 * h;
            guess[i * values.len()] = x;
            guess[i * values.len() + 1] = 1.0;
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());
        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(1usize, 1.0f64)]),
        ]);

        let mut solver = NRBDVPd::new_numeric_with_jacobian_options(
            initial_guess,
            values,
            "x".to_string(),
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            |_x, y: &DVector<f64>, _params| DVector::from_vec(vec![y[1], 0.0]),
            |_x, _y: &DVector<f64>, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]),
        );
        solver.dont_save_log(true);

        assert!(solver.has_numeric_rhs());
        assert!(solver.has_numeric_jacobian());
        solver
            .try_eq_generate(None, None)
            .expect("banded numeric wrapper with user Jacobian should generate callbacks");
        assert!(
            solver.jac.is_some(),
            "banded user numeric Jacobian should install a global discretized Jacobian callback"
        );

        solver
            .try_solve()
            .expect("banded numeric wrapper with user Jacobian should solve end-to-end");
        assert_linear_bvp_solution_quality(&solver, t0, t_end, n_steps, 1e-8, 1e-7);
    }

    #[test]
    fn bvp_pure_numerical_short_wrapper_with_user_jacobian_solves_end_to_end() {
        let n_steps = 80;
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + i as f64 * h;
            guess[i * values.len()] = x;
            guess[i * values.len() + 1] = 1.0;
        }
        let initial_guess =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice());
        let border_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(1usize, 1.0f64)]),
        ]);
        let bounds = HashMap::from([
            ("y".to_string(), (-0.2, 1.2)),
            ("z".to_string(), (-2.0, 2.0)),
        ]);
        let rel_tolerance = HashMap::from([("y".to_string(), 1e-8), ("z".to_string(), 1e-8)]);

        let mut solver = NRBDVPd::new_numeric_with_jacobian(
            initial_guess,
            values,
            "x".to_string(),
            border_conditions,
            t0,
            t_end,
            n_steps,
            bounds,
            rel_tolerance,
            |_x, y: &DVector<f64>, _params| DVector::from_vec(vec![y[1], 0.0]),
            |_x, _y: &DVector<f64>, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]),
        );
        solver.dont_save_log(true);

        assert!(solver.has_numeric_rhs());
        assert!(solver.has_numeric_jacobian());
        solver
            .try_solve()
            .expect("short numeric wrapper with user Jacobian should solve end-to-end");
        assert_linear_bvp_solution_quality(&solver, t0, t_end, n_steps, 1e-8, 1e-7);
    }

    #[test]
    fn bvp_banded_atomview_lambdify_oscillator_end_to_end_correctness() {
        let mut options = base_linear_bvp_options()
            .with_banded_lambdify()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) = oscillator_bvp_symbolic_fixture(120, options);
        solver.dont_save_log(true);

        solver
            .try_solve()
            .expect("banded AtomView lambdify oscillator BVP should solve end-to-end");
        assert_oscillator_bvp_solution_quality(&solver, t0, t_end, n_steps, 2e-4, 2e-3);
    }

    #[test]
    fn bvp_banded_atomview_aot_oscillator_end_to_end_correctness() {
        let mut options = base_linear_bvp_options().with_banded_atomview_c_tcc();
        options.scheme = "trapezoid".to_string();
        let (mut bootstrap_solver, t0, t_end, n_steps) =
            oscillator_bvp_symbolic_fixture(120, options.clone());
        bootstrap_solver.dont_save_log(true);

        match bootstrap_solver.try_solve() {
            Ok(_) => {
                assert_oscillator_bvp_solution_quality(
                    &bootstrap_solver,
                    t0,
                    t_end,
                    n_steps,
                    2e-4,
                    2e-3,
                );

                // Strict AOT phase:
                // - reuse resolver snapshot from bootstrap run
                // - require prebuilt compiled backend (fallback is not allowed)
                let bootstrap_resolver = bootstrap_solver
                    .aot_resolver()
                    .cloned()
                    .expect("bootstrap AOT run should persist resolver snapshot");

                let (mut strict_solver, t0_strict, t_end_strict, n_steps_strict) =
                    oscillator_bvp_symbolic_fixture(120, options);
                strict_solver.dont_save_log(true);
                strict_solver.set_aot_resolver(Some(bootstrap_resolver));
                strict_solver.set_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
                strict_solver.try_solve().expect(
                    "strict AOT run should succeed with RequirePrebuilt (no lambdify fallback)",
                );
                assert_oscillator_bvp_solution_quality(
                    &strict_solver,
                    t0_strict,
                    t_end_strict,
                    n_steps_strict,
                    2e-4,
                    2e-3,
                );
            }
            Err(err) if is_aot_environment_issue(&err) => {
                eprintln!(
                    "Skipping strict AOT oscillator correctness assertion due to environment/toolchain issue: {err:?}"
                );
            }
            Err(err) => {
                panic!("banded AtomView AOT oscillator BVP solve failed unexpectedly: {err:?}")
            }
        }
    }

    #[test]
    fn bvp_sparse_pure_numerical_backend_oscillator_end_to_end_correctness() {
        let mut options = base_linear_bvp_options().with_sparse_generated_backend_defaults();
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) = oscillator_bvp_numeric_fixture(120, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
            DVector::from_vec(vec![y[1], -y[0]])
        })));

        solver
            .try_solve()
            .expect("numeric-only sparse oscillator BVP should solve end-to-end");
        assert_oscillator_bvp_solution_quality(&solver, t0, t_end, n_steps, 2e-4, 2e-3);
    }

    #[test]
    fn bvp_banded_pure_numerical_backend_oscillator_end_to_end_correctness() {
        let mut options = base_linear_bvp_options().with_banded_generated_backend_defaults();
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) = oscillator_bvp_numeric_fixture(120, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
            DVector::from_vec(vec![y[1], -y[0]])
        })));

        solver
            .try_solve()
            .expect("numeric-only banded oscillator BVP should solve end-to-end");
        assert_oscillator_bvp_solution_quality(&solver, t0, t_end, n_steps, 2e-4, 2e-3);
    }

    #[test]
    fn bvp_sparse_pure_numerical_two_point_oscillator_user_jacobian_end_to_end_correctness() {
        let mut options = base_linear_bvp_options().with_sparse_generated_backend_defaults();
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            oscillator_bvp_numeric_two_y_bc_fixture(120, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        solver
            .try_solve()
            .expect("numeric-only sparse oscillator with two y-boundaries should solve end-to-end");
        assert_oscillator_bvp_solution_quality(&solver, t0, t_end, n_steps, 2e-4, 2e-3);
    }

    #[test]
    fn bvp_banded_pure_numerical_two_point_oscillator_user_jacobian_end_to_end_correctness() {
        let mut options = base_linear_bvp_options().with_banded_generated_backend_defaults();
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            oscillator_bvp_numeric_two_y_bc_fixture(120, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        solver
            .try_solve()
            .expect("numeric-only banded oscillator with two y-boundaries should solve end-to-end");
        assert_oscillator_bvp_solution_quality(&solver, t0, t_end, n_steps, 2e-4, 2e-3);
    }

    #[test]
    fn bvp_sparse_pure_numerical_backend_stiff_decay_end_to_end_correctness() {
        let lambda_fast = 25.0f64;
        let lambda_slow = 1.0f64;
        let stiff_bounds = HashMap::from([
            ("y".to_string(), (-0.1, 1.1)),
            ("z".to_string(), (-0.1, 1.1)),
        ]);
        let mut options = base_linear_bvp_options()
            .with_sparse_generated_backend_defaults()
            .with_bounds(stiff_bounds);
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            stiff_decay_bvp_numeric_fixture(200, lambda_fast, lambda_slow, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(move |_x, state: &DVector<f64>, _params| {
            DVector::from_vec(vec![-lambda_fast * state[0], -lambda_slow * state[1]])
        })));

        solver
            .try_solve()
            .expect("numeric-only sparse stiff decay BVP should solve end-to-end");
        assert_stiff_decay_solution_quality(
            &solver,
            t0,
            t_end,
            n_steps,
            lambda_fast,
            lambda_slow,
            2e-3,
            1.5e-2,
        );
    }

    #[test]
    fn bvp_banded_pure_numerical_backend_stiff_decay_end_to_end_correctness() {
        let lambda_fast = 25.0f64;
        let lambda_slow = 1.0f64;
        let stiff_bounds = HashMap::from([
            ("y".to_string(), (-0.1, 1.1)),
            ("z".to_string(), (-0.1, 1.1)),
        ]);
        let mut options = base_linear_bvp_options()
            .with_banded_generated_backend_defaults()
            .with_bounds(stiff_bounds);
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            stiff_decay_bvp_numeric_fixture(200, lambda_fast, lambda_slow, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(move |_x, state: &DVector<f64>, _params| {
            DVector::from_vec(vec![-lambda_fast * state[0], -lambda_slow * state[1]])
        })));

        solver
            .try_solve()
            .expect("numeric-only banded stiff decay BVP should solve end-to-end");
        assert_stiff_decay_solution_quality(
            &solver,
            t0,
            t_end,
            n_steps,
            lambda_fast,
            lambda_slow,
            2e-3,
            1.5e-2,
        );
    }

    #[test]
    fn bvp_sparse_pure_numerical_user_jacobian_stiff_decay_end_to_end_correctness() {
        let lambda_fast = 25.0f64;
        let lambda_slow = 1.0f64;
        let stiff_bounds = HashMap::from([
            ("y".to_string(), (-0.1, 1.1)),
            ("z".to_string(), (-0.1, 1.1)),
        ]);
        let mut options = base_linear_bvp_options()
            .with_sparse_generated_backend_defaults()
            .with_bounds(stiff_bounds);
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            stiff_decay_bvp_numeric_jacobian_fixture(200, lambda_fast, lambda_slow, options);
        solver.dont_save_log(true);

        assert!(solver.has_numeric_rhs());
        assert!(solver.has_numeric_jacobian());
        solver
            .try_solve()
            .expect("sparse numeric stiff decay with user Jacobian should solve end-to-end");
        assert!(
            solver.jac.is_some(),
            "user Jacobian path should keep an installed global Jacobian callback"
        );
        assert_stiff_decay_solution_quality(
            &solver,
            t0,
            t_end,
            n_steps,
            lambda_fast,
            lambda_slow,
            2e-3,
            1.5e-2,
        );
    }

    #[test]
    fn bvp_banded_pure_numerical_user_jacobian_stiff_decay_end_to_end_correctness() {
        let lambda_fast = 25.0f64;
        let lambda_slow = 1.0f64;
        let stiff_bounds = HashMap::from([
            ("y".to_string(), (-0.1, 1.1)),
            ("z".to_string(), (-0.1, 1.1)),
        ]);
        let mut options = base_linear_bvp_options()
            .with_banded_generated_backend_defaults()
            .with_bounds(stiff_bounds);
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            stiff_decay_bvp_numeric_jacobian_fixture(200, lambda_fast, lambda_slow, options);
        solver.dont_save_log(true);

        assert!(solver.has_numeric_rhs());
        assert!(solver.has_numeric_jacobian());
        solver
            .try_solve()
            .expect("banded numeric stiff decay with user Jacobian should solve end-to-end");
        assert!(
            solver.jac.is_some(),
            "banded user Jacobian path should keep an installed global Jacobian callback"
        );
        assert_stiff_decay_solution_quality(
            &solver,
            t0,
            t_end,
            n_steps,
            lambda_fast,
            lambda_slow,
            2e-3,
            1.5e-2,
        );
    }

    #[test]
    fn bvp_sparse_pure_numerical_backend_stiff_coupled_end_to_end_correctness() {
        let lambda_fast = 40.0f64;
        let lambda_slow = 1.0f64;
        let coupling = 3.0f64;
        let stiff_bounds = HashMap::from([
            ("y".to_string(), (-0.2, 1.2)),
            ("z".to_string(), (-0.2, 1.2)),
        ]);
        let mut options = base_linear_bvp_options()
            .with_sparse_generated_backend_defaults()
            .with_bounds(stiff_bounds);
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            stiff_decay_bvp_numeric_fixture(220, lambda_fast, lambda_slow, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(move |_x, state: &DVector<f64>, _params| {
            DVector::from_vec(vec![
                -lambda_fast * state[0],
                -lambda_slow * state[1] + coupling * state[0],
            ])
        })));

        solver
            .try_solve()
            .expect("numeric-only sparse stiff coupled BVP should solve end-to-end");
        assert_stiff_coupled_solution_quality(
            &solver,
            t0,
            t_end,
            n_steps,
            lambda_fast,
            lambda_slow,
            coupling,
            2.5e-3,
            2.0e-2,
        );
    }

    #[test]
    fn bvp_banded_pure_numerical_backend_stiff_coupled_end_to_end_correctness() {
        let lambda_fast = 40.0f64;
        let lambda_slow = 1.0f64;
        let coupling = 3.0f64;
        let stiff_bounds = HashMap::from([
            ("y".to_string(), (-0.2, 1.2)),
            ("z".to_string(), (-0.2, 1.2)),
        ]);
        let mut options = base_linear_bvp_options()
            .with_banded_generated_backend_defaults()
            .with_bounds(stiff_bounds);
        options.scheme = "trapezoid".to_string();
        let (mut solver, t0, t_end, n_steps) =
            stiff_decay_bvp_numeric_fixture(220, lambda_fast, lambda_slow, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(move |_x, state: &DVector<f64>, _params| {
            DVector::from_vec(vec![
                -lambda_fast * state[0],
                -lambda_slow * state[1] + coupling * state[0],
            ])
        })));

        solver
            .try_solve()
            .expect("numeric-only banded stiff coupled BVP should solve end-to-end");
        assert_stiff_coupled_solution_quality(
            &solver,
            t0,
            t_end,
            n_steps,
            lambda_fast,
            lambda_slow,
            coupling,
            2.5e-3,
            2.0e-2,
        );
    }

    #[test]
    fn bvp_sparse_pure_numerical_backend_adaptive_refinement_end_to_end_correctness() {
        let adaptive = AdaptiveGridConfig {
            version: 1,
            max_refinements: 1,
            grid_method: GridRefinementMethod::DoublePoints,
        };
        let strategy = SolverParams {
            adaptive: Some(adaptive),
            ..SolverParams::default()
        };
        let options = base_linear_bvp_options()
            .with_sparse_generated_backend_defaults()
            .with_strategy_params(Some(strategy));
        let coarse_steps = 20usize;
        let (mut solver, t0, t_end, _n_steps) = linear_bvp_numeric_fixture(coarse_steps, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
            DVector::from_vec(vec![y[1], 0.0])
        })));

        solver
            .try_solve()
            .expect("numeric-only sparse adaptive BVP should solve end-to-end on refined mesh");

        let solution = solver
            .get_result()
            .expect("solver should provide full result matrix after adaptive refinement")
            .clone();
        let expected_rows_after_one_refinement = 2 * coarse_steps + 1;
        assert_eq!(
            solution.nrows(),
            expected_rows_after_one_refinement,
            "adaptive DoublePoints refinement should double intervals exactly once"
        );

        let refined_steps = solution.nrows() - 1;
        let h = (t_end - t0) / refined_steps as f64;
        let mut sq_sum = 0.0;
        let mut max_abs = 0.0;
        for i in 0..solution.nrows() {
            let x = t0 + (i as f64) * h;
            let y_exact = x;
            let y_num = solution[(i, 0)];
            let err = (y_num - y_exact).abs();
            sq_sum += err * err;
            if err > max_abs {
                max_abs = err;
            }
        }
        let rms = (sq_sum / solution.nrows() as f64).sqrt();
        assert!(
            rms <= 1e-5,
            "adaptive numeric-only RMS error too large: {rms:e}"
        );
        assert!(
            max_abs <= 5e-5,
            "adaptive numeric-only max abs error too large: {max_abs:e}"
        );

        let stats = solver.get_statistics();
        let refinements = *stats
            .counters
            .get("number of grid refinements")
            .unwrap_or(&0usize);
        assert_eq!(
            refinements, 1usize,
            "adaptive run should perform exactly one refinement step"
        );
    }

    #[test]
    fn bvp_banded_pure_numerical_backend_adaptive_refinement_end_to_end_correctness() {
        let adaptive = AdaptiveGridConfig {
            version: 1,
            max_refinements: 1,
            grid_method: GridRefinementMethod::DoublePoints,
        };
        let strategy = SolverParams {
            adaptive: Some(adaptive),
            ..SolverParams::default()
        };
        let options = base_linear_bvp_options()
            .with_banded_generated_backend_defaults()
            .with_strategy_params(Some(strategy));
        let coarse_steps = 20usize;
        let (mut solver, t0, t_end, _n_steps) = linear_bvp_numeric_fixture(coarse_steps, options);
        solver.dont_save_log(true);
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        solver.set_numeric_rhs(Some(Arc::new(|_x, y: &DVector<f64>, _params| {
            DVector::from_vec(vec![y[1], 0.0])
        })));

        solver
            .try_solve()
            .expect("numeric-only banded adaptive BVP should solve end-to-end on refined mesh");

        let solution = solver
            .get_result()
            .expect("solver should provide full result matrix after adaptive refinement")
            .clone();
        let expected_rows_after_one_refinement = 2 * coarse_steps + 1;
        assert_eq!(
            solution.nrows(),
            expected_rows_after_one_refinement,
            "adaptive DoublePoints refinement should double intervals exactly once"
        );

        let refined_steps = solution.nrows() - 1;
        let h = (t_end - t0) / refined_steps as f64;
        let mut sq_sum = 0.0;
        let mut max_abs = 0.0;
        for i in 0..solution.nrows() {
            let x = t0 + (i as f64) * h;
            let y_exact = x;
            let y_num = solution[(i, 0)];
            let err = (y_num - y_exact).abs();
            sq_sum += err * err;
            if err > max_abs {
                max_abs = err;
            }
        }
        let rms = (sq_sum / solution.nrows() as f64).sqrt();
        assert!(
            rms <= 1e-5,
            "adaptive banded numeric-only RMS error too large: {rms:e}"
        );
        assert!(
            max_abs <= 5e-5,
            "adaptive banded numeric-only max abs error too large: {max_abs:e}"
        );

        let stats = solver.get_statistics();
        let refinements = *stats
            .counters
            .get("number of grid refinements")
            .unwrap_or(&0usize);
        assert_eq!(
            refinements, 1usize,
            "adaptive run should perform exactly one refinement step"
        );
    }

    /// Tests the boundary value problem (BVP) solver for the Clairaut equation.
    ///
    /// This test sets up a non-linear equation using the Clairaut configuration
    /// and solves it numerically using a damped strategy with sparse method
    /// for a specified number of steps. It verifies the numerical solution
    /// against the exact solution by calculating the norm of the difference
    /// between the numerical and exact solutions. The test asserts that this
    /// norm is below a specified tolerance to ensure the accuracy of the solution.
    //  #[test]
    fn test_BVP_Damp2() {
        // let ne=  (NonlinEquation::  Clairaut  ); //  Clairaut  LaneEmden5  ParachuteEquation
        for ne in NonlinEquation::iter() {
            println!("problem {:?}", ne);
            let eq_system = ne.setup();
            let values = ne.values();
            let arg = "x".to_string();
            let tolerance = 1e-5;
            let max_iterations = 20;
            let t0 = ne.span(None, None).0;
            let t_end = ne.span(None, None).1;
            let n_steps = 180; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
            let strategy = "Damped".to_string(); //
            let strategy_params = Some(SolverParams::default());
            let scheme = "forward".to_string();
            let method = "Dense".to_string(); // or  "Dense"
            let linear_sys_method = None;
            let ones = vec![0.99; values.len() * n_steps];
            let initial_guess: DMatrix<f64> = DMatrix::from_column_slice(
                values.len(),
                n_steps,
                DVector::from_vec(ones).as_slice(),
            );
            let BorderConditions = ne.boundary_conditions();
            let Bounds = ne.Bounds();
            let rel_tolerance = ne.rel_tolerance();

            let mut nr = NRBDVPd::new(
                eq_system,
                initial_guess,
                values.clone(),
                arg,
                BorderConditions.clone(),
                t0,
                t_end,
                n_steps,
                scheme,
                strategy,
                strategy_params,
                linear_sys_method,
                method,
                tolerance,
                Some(rel_tolerance),
                max_iterations,
                Some(Bounds),
                None,
            );

            println!("solving system");
            nr.try_solve()
                .expect("enumerated dense damped BVP should solve through the fallible API");
            let solution = nr.get_result().unwrap().clone();
            let y_numer = solution.column(0);
            let y_numer: Vec<f64> = y_numer.iter().map(|x| *x).collect();

            nr.plot_result();
            //compare with exact solution
            let y_exact = ne.exact_solution(None, None, Some(n_steps));
            let n = &y_exact.len();
            // println!("numerical result = {:?}",  y_numer);
            println!("\n \n y exact{:?}, {}", &y_exact, &y_exact.len());
            println!("\n \n y numer{:?}, {}", &y_numer, &y_numer.len());
            let comparsion: Vec<f64> = y_numer
                .into_iter()
                .zip(y_exact.clone())
                .map(|(y_n_i, y_e_i)| (y_n_i - y_e_i).powf(2.0))
                .collect();
            let norm = comparsion.iter().sum::<f64>().sqrt() / (*n as f64);
            let max_residual = comparsion
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let position = comparsion.iter().position(|&x| x == *max_residual).unwrap();
            let relativ_residual = max_residual.abs() / y_exact[position];
            println!(
                "maximum relative residual of numerical solution wioth respect to exact solution = {}",
                relativ_residual
            );

            assert!(norm < 1e-2, "norm = {}", norm);
            assert!(relativ_residual < 1e-1, "norm = {}", norm);
            println!("norm = {}", norm);
            //   let extract_unknown = extract_unknown_variables( solution.clone().transpose(), &BorderConditions.clone(), &values.clone() );
            //   let solution_reconstructed = construct_full_solution( extract_unknown.clone(), &BorderConditions.clone(), &values.clone() ).transpose();
            //   assert_eq!(solution, solution_reconstructed, "solution and reconstruction are not equal");
        }
    }
}
