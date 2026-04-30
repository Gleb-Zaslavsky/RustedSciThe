//! Solve-level acceptance tests for dense nonlinear AOT backends.
//!
//! These tests sit one layer above the lifecycle smoke tests:
//! they do not only verify that a generated dense backend can be built and
//! linked, but that real nonlinear methods can solve through the compiled
//! backend branch selected by the symbolic setup layer.

#[cfg(test)]
mod tests {
    use crate::numerical::Nonlinear_systems::LM_vanilla::LevenbergMarquardtMethod;
    use crate::numerical::Nonlinear_systems::NR_damped::DampedNewtonMethod;
    use crate::numerical::Nonlinear_systems::engine::{NewtonMethod, SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::error::TerminationReason;
    use crate::numerical::Nonlinear_systems::problem::Bounds;
    use crate::numerical::Nonlinear_systems::symbolic::{
        SymbolicDenseAotOptions, SymbolicNonlinearProblem, SymbolicProblemOptions,
    };
    use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::{
        aot_solver_test_guard, linked_dense_resolver_for_problem,
        register_elementary_dense_backend, register_parameterized_dense_backend,
    };
    use crate::numerical::Nonlinear_systems::symbolic_backend::SymbolicBackendSelectionPolicy;
    use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    fn elementary_problem_options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_lambdify_backend()
    }

    fn parameterized_problem_options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()])
            .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
            .with_lambdify_backend()
    }

    fn elementary_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_strings_with_options(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            elementary_problem_options(),
        )
        .expect("elementary symbolic problem should build")
    }

    fn assert_good_termination(reason: TerminationReason) {
        assert!(
            matches!(
                reason,
                TerminationReason::Converged | TerminationReason::StepTooSmall
            ),
            "unexpected nonlinear termination: {:?}",
            reason
        );
    }

    #[test]
    fn dense_compiled_aot_newton_acceptance_solves_elementary_problem() {
        let _guard = aot_solver_test_guard();
        let baseline = elementary_problem();
        let (_dir, resolver, problem_key) = linked_dense_resolver_for_problem(&baseline);
        register_elementary_dense_backend(&problem_key);

        let compiled = SymbolicNonlinearProblem::from_strings_with_backend_selection(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            elementary_problem_options(),
            SymbolicBackendSelectionPolicy::AotOnly,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .expect("compiled nonlinear problem should build");

        let result = SolverEngine::new(NewtonMethod, SolveOptions::default())
            .solve(&compiled, DVector::from_vec(vec![2.5, -0.5]))
            .expect("compiled Newton solve should succeed");

        assert_good_termination(result.termination);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-8);
        assert!(result.residual_norm < 1e-6);

        unregister_linked_dense_backend(&problem_key);
    }

    #[test]
    fn dense_compiled_aot_damped_newton_acceptance_solves_elementary_problem() {
        let _guard = aot_solver_test_guard();
        let baseline = elementary_problem();
        let (_dir, resolver, problem_key) = linked_dense_resolver_for_problem(&baseline);
        register_elementary_dense_backend(&problem_key);

        let compiled = SymbolicNonlinearProblem::from_strings_with_backend_selection(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            elementary_problem_options(),
            SymbolicBackendSelectionPolicy::AotOnly,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .expect("compiled nonlinear problem should build");

        let options = SolveOptions {
            bounds: Some(Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]).expect("bounds")),
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(DampedNewtonMethod::default(), options)
            .solve(&compiled, DVector::from_vec(vec![1.0, 1.0]))
            .expect("compiled damped Newton solve should succeed");

        assert_good_termination(result.termination);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-6);
        assert!(result.residual_norm < 1e-6);

        unregister_linked_dense_backend(&problem_key);
    }

    #[test]
    fn dense_compiled_aot_lm_acceptance_solves_elementary_problem() {
        let _guard = aot_solver_test_guard();
        let baseline = elementary_problem();
        let (_dir, resolver, problem_key) = linked_dense_resolver_for_problem(&baseline);
        register_elementary_dense_backend(&problem_key);

        let compiled = SymbolicNonlinearProblem::from_strings_with_backend_selection(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            elementary_problem_options(),
            SymbolicBackendSelectionPolicy::AotOnly,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .expect("compiled nonlinear problem should build");

        let options = SolveOptions {
            bounds: Some(Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]).expect("bounds")),
            tolerance: 1e-8,
            max_iterations: 80,
            ..SolveOptions::default()
        };
        let method = LevenbergMarquardtMethod::default();
        let result = SolverEngine::new(method, options)
            .solve(&compiled, DVector::from_vec(vec![1.0, 1.0]))
            .expect("compiled LM solve should succeed");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-6);
        assert!(result.residual_norm < 1e-8);

        unregister_linked_dense_backend(&problem_key);
    }

    #[test]
    fn dense_compiled_aot_parameterized_newton_acceptance_solves_problem() {
        let _guard = aot_solver_test_guard();
        let symbolic = Expr::Symbols("x, y, a");
        let x = symbolic[0].clone();
        let y = symbolic[1].clone();
        let a = symbolic[2].clone();
        let options = parameterized_problem_options();
        let baseline = SymbolicNonlinearProblem::from_expressions_with_options(
            vec![
                a.clone() * x.clone() + y.clone() - Expr::Const(3.0),
                x.clone() - y.clone(),
            ],
            options.clone(),
        )
        .expect("parameterized symbolic problem should build");
        let (_dir, resolver, problem_key) = linked_dense_resolver_for_problem(&baseline);
        register_parameterized_dense_backend(&problem_key);

        let compiled = SymbolicNonlinearProblem::from_expressions_with_backend_selection(
            vec![
                a.clone() * x.clone() + y.clone() - Expr::Const(3.0),
                x.clone() - y.clone(),
            ],
            options,
            SymbolicBackendSelectionPolicy::AotOnly,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .expect("compiled parameterized nonlinear problem should build");

        let result = SolverEngine::new(NewtonMethod, SolveOptions::default())
            .solve(&compiled, DVector::from_vec(vec![0.5, 0.5]))
            .expect("compiled parameterized Newton solve should succeed");

        assert_good_termination(result.termination);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-8);
        assert!(result.residual_norm < 1e-6);

        unregister_linked_dense_backend(&problem_key);
    }
}
