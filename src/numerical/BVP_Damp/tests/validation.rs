//! Typed validation gates for the fallible Damped/Frozen preparation boundary.

#[cfg(test)]
mod tests {
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        DampedSolverOptions, NRBVP as DampedBvp, SolverParams,
    };
    use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::{
        FrozenSolverOptions, NRBVP as FrozenBvp,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    fn damped_solver(options: DampedSolverOptions) -> DampedBvp {
        DampedBvp::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.0),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, 0.0)]),
                ("z".to_string(), vec![(0usize, 0.0)]),
            ]),
            0.0,
            1.0,
            4,
            options,
        )
    }

    fn frozen_solver(options: FrozenSolverOptions) -> FrozenBvp {
        FrozenBvp::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.0),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, 0.0)]),
                ("z".to_string(), vec![(0usize, 0.0)]),
            ]),
            0.0,
            1.0,
            4,
            options,
        )
    }

    #[test]
    fn damped_typed_validation_rejects_unknown_method() {
        let mut options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_bounds(HashMap::from([
                ("y".to_string(), (-10.0, 10.0)),
                ("z".to_string(), (-10.0, 10.0)),
            ]))
            .with_rel_tolerance(HashMap::from([
                ("y".to_string(), 1e-6),
                ("z".to_string(), 1e-6),
            ]));
        options.method = "not-a-matrix-backend".to_string();

        let error = damped_solver(options)
            .try_task_check()
            .expect_err("unknown method must be a typed validation error");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidSolverConfiguration { field, .. }
                if field == "method"
        ));
    }

    #[test]
    fn damped_typed_validation_rejects_initial_guess_outside_bounds() {
        let options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_bounds(HashMap::from([
                ("y".to_string(), (-1.0, 1.0)),
                ("z".to_string(), (-1.0, 1.0)),
            ]))
            .with_rel_tolerance(HashMap::from([
                ("y".to_string(), 1e-6),
                ("z".to_string(), 1e-6),
            ]));
        let mut solver = damped_solver(options);
        solver.initial_guess[(0, 0)] = 2.0;

        let error = solver
            .try_task_check()
            .expect_err("out-of-bounds initial guess must be typed");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidProblem { field, .. }
                if field == "initial_guess"
        ));
    }

    #[test]
    fn frozen_typed_validation_rejects_unknown_strategy_without_process_exit() {
        let mut options = FrozenSolverOptions::default();
        options.strategy = "not-a-frozen-strategy".to_string();

        let error = frozen_solver(options)
            .try_task_check()
            .expect_err("unknown Frozen strategy must be typed");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidSolverConfiguration { field, .. }
                if field == "strategy"
        ));
    }

    #[test]
    fn frozen_typed_validation_rejects_malformed_strategy_parameters() {
        let mut options = FrozenSolverOptions::default();
        options.strategy_params = Some(HashMap::from([(
            "every_m".to_string(),
            Some(vec![0.0, 1.0]),
        )]));

        let error = frozen_solver(options)
            .try_task_check()
            .expect_err("malformed Frozen parameters must be typed");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidSolverConfiguration { field, .. }
                if field == "strategy_params"
        ));
    }
}
