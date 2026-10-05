//! Fast correctness stories and independent stiff reference workloads.

use super::super::new::atom_native::AtomNativePlan;
use super::super::new::callbacks::validate_callback_output;
use super::super::new::callbacks::{PreparedSymbolicCallbacks, SymbolicCallbackWorkspace};
use super::super::new::config::{RadauAssembly, RadauConfig, RadauExecution, RadauMatrixLayout};
use super::super::new::error::{RadauError, RadauStage};
use super::super::new::lambdify::LambdifyPlan;
use super::super::new::linear::{
    factor_complex_in_place, factor_dense_in_place, solve_factored_complex_in_place,
    solve_factored_dense_in_place,
};
use super::super::new::prepared::RadauPreparedModel;
use super::super::new::solver::{
    try_solve_dense, try_solve_dense_with_callbacks, try_solve_symbolic_dense,
};
use super::super::new::step::{try_radau5_step, try_radau5_symbolic_step};
use super::super::new::workspace::RadauWorkspace;
use crate::symbolic::symbolic_engine::Expr;

#[test]
fn adaptive_dense_solver_advances_forward_and_reuses_step_contract() {
    let config = RadauConfig {
        t_bound: 1.0,
        first_step: Some(0.1),
        max_step: 0.25,
        rtol: 1.0e-8,
        atol: 1.0e-10,
        max_newton_iterations: 8,
        ..RadauConfig::default()
    };
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok::<(), RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = -1.0;
        Ok::<(), RadauError>(())
    };

    let result =
        try_solve_dense_with_callbacks(&config, &mut residual, &mut jacobian, &[1.0]).unwrap();

    assert_eq!(result.t, config.t_bound);
    assert!(result.accepted_steps > 0);
    assert!(result.attempts >= result.accepted_steps);
    assert!((result.y[0] - (-1.0_f64).exp()).abs() < 1.0e-7);
}

#[test]
fn adaptive_dense_solver_supports_reverse_time_and_step_budget_errors() {
    let config = RadauConfig {
        t0: 1.0,
        t_bound: 0.0,
        first_step: Some(0.1),
        rtol: 1.0e-8,
        atol: 1.0e-10,
        max_newton_iterations: 8,
        ..RadauConfig::default()
    };
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok::<(), RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = -1.0;
        Ok::<(), RadauError>(())
    };
    let result =
        try_solve_dense_with_callbacks(&config, &mut residual, &mut jacobian, &[(-1.0_f64).exp()])
            .unwrap();
    assert_eq!(result.t, config.t_bound);
    assert!((result.y[0] - 1.0).abs() < 1.0e-7);

    let budget = RadauConfig {
        max_steps: 1,
        ..config
    };
    let error =
        try_solve_dense_with_callbacks(&budget, &mut residual, &mut jacobian, &[1.0]).unwrap_err();
    assert!(matches!(error, RadauError::StepBudgetExceeded { steps: 1 }));
}

#[test]
fn adaptive_dense_solver_handles_nonautonomous_rhs_with_time_argument() {
    let config = RadauConfig {
        t_bound: 0.4,
        first_step: Some(0.05),
        max_step: 0.1,
        rtol: 1.0e-9,
        atol: 1.0e-11,
        max_newton_iterations: 8,
        ..RadauConfig::default()
    };
    let mut residual = |time: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = time;
        Ok::<(), RadauError>(())
    };
    let mut jacobian = |_time: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = 0.0;
        Ok::<(), RadauError>(())
    };
    let result =
        try_solve_dense_with_callbacks(&config, &mut residual, &mut jacobian, &[0.0]).unwrap();

    assert_eq!(result.t, config.t_bound);
    assert!((result.y[0] - 0.5 * config.t_bound.powi(2)).abs() < 1.0e-8);
}

#[test]
fn tolerance_refinement_improves_or_preserves_scalar_fidelity() {
    let solve = |rtol: f64, atol: f64| {
        let config = RadauConfig {
            t_bound: 0.5,
            first_step: Some(0.05),
            max_step: 0.1,
            rtol,
            atol,
            max_newton_iterations: 8,
            ..RadauConfig::default()
        };
        let mut residual = |_time: f64, state: &[f64], output: &mut [f64]| {
            output[0] = -state[0];
            Ok::<(), RadauError>(())
        };
        let mut jacobian = |_time: f64, _state: &[f64], output: &mut [f64]| {
            output[0] = -1.0;
            Ok::<(), RadauError>(())
        };
        try_solve_dense_with_callbacks(&config, &mut residual, &mut jacobian, &[1.0])
            .unwrap()
            .y[0]
    };
    let exact = (-0.5_f64).exp();
    let coarse_error = (solve(1.0e-5, 1.0e-7) - exact).abs();
    let fine_error = (solve(1.0e-9, 1.0e-11) - exact).abs();

    assert!(coarse_error < 1.0e-4, "coarse error={coarse_error:.3e}");
    assert!(
        fine_error <= coarse_error * 1.5 + 1.0e-11,
        "coarse error={coarse_error:.3e}; fine error={fine_error:.3e}"
    );
}

#[test]
fn coupled_decay_transfer_preserves_mass_like_invariant() {
    let config = RadauConfig {
        t_bound: 0.5,
        first_step: Some(0.05),
        max_step: 0.1,
        rtol: 1.0e-9,
        atol: 1.0e-11,
        max_newton_iterations: 8,
        ..RadauConfig::default()
    };
    let mut residual = |_time: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        output[1] = state[0];
        Ok::<(), RadauError>(())
    };
    let mut jacobian = |_time: f64, _state: &[f64], output: &mut [f64]| {
        output.copy_from_slice(&[-1.0, 0.0, 1.0, 0.0]);
        Ok::<(), RadauError>(())
    };
    let result =
        try_solve_dense_with_callbacks(&config, &mut residual, &mut jacobian, &[1.0, 0.0]).unwrap();

    assert!((result.y[0] + result.y[1] - 1.0).abs() < 1.0e-8);
    assert!((result.y[0] - (-0.5_f64).exp()).abs() < 1.0e-8);
}

#[test]
fn adaptive_session_records_accepted_and_rejected_steps_in_telemetry() {
    let config = RadauConfig {
        t_bound: 0.2,
        first_step: Some(0.1),
        max_step: 0.1,
        telemetry: super::super::new::telemetry::RadauTelemetryMode::Counters,
        ..RadauConfig::default()
    };
    let prepared = RadauPreparedModel::prepare(1, &config).unwrap();
    let mut session = super::super::new::session::RadauSession::new(prepared).unwrap();
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok::<(), RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = -1.0;
        Ok::<(), RadauError>(())
    };

    let result =
        try_solve_dense(&config, &mut session, &mut residual, &mut jacobian, &[1.0]).unwrap();
    let telemetry = &session.workspace().telemetry.counters;
    assert_eq!(telemetry.accepted_steps as usize, result.accepted_steps);
    assert_eq!(telemetry.rejected_steps as usize, result.rejected_steps);
}

#[test]
fn adaptive_symbolic_solver_uses_both_selectable_lambdify_frontends() {
    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        let residual = vec![Expr::Mul(
            Box::new(Expr::Const(-1.0)),
            Box::new(Expr::Mul(
                Box::new(Expr::Var("p".to_string())),
                Box::new(Expr::Var("y1".to_string())),
            )),
        )];
        let jacobian = vec![Expr::Mul(
            Box::new(Expr::Const(-1.0)),
            Box::new(Expr::Var("p".to_string())),
        )];
        let selected_jacobian = match assembly {
            RadauAssembly::ExprLegacy => Some(jacobian),
            RadauAssembly::AtomViewNative => None,
        };
        let callbacks = PreparedSymbolicCallbacks::prepare(
            assembly,
            residual,
            selected_jacobian,
            "t",
            &["y1"],
            &["p"],
        )
        .unwrap();
        let mut session = callbacks.session();
        session.rebind_parameters(&[1.0]).unwrap();
        let config = RadauConfig {
            execution: RadauExecution::Lambdify,
            assembly: Some(assembly),
            t_bound: 0.5,
            first_step: Some(0.1),
            rtol: 1.0e-8,
            atol: 1.0e-10,
            max_newton_iterations: 8,
            ..RadauConfig::default()
        };
        let result = try_solve_symbolic_dense(&config, &mut session, &[1.0]).unwrap();
        assert_eq!(result.t, config.t_bound);
        assert!((result.y[0] - (-0.5_f64).exp()).abs() < 1.0e-7);
    }
}

#[test]
fn adaptive_symbolic_solver_uses_native_sparse_and_banded_linear_routes() {
    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        for layout in [
            RadauMatrixLayout::Sparse,
            RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        ] {
            let residual = vec![
                Expr::Mul(
                    Box::new(Expr::Const(-1.0)),
                    Box::new(Expr::Var("y1".to_string())),
                ),
                Expr::Mul(
                    Box::new(Expr::Const(-2.0)),
                    Box::new(Expr::Var("y2".to_string())),
                ),
            ];
            let jacobian = vec![
                Expr::Const(-1.0),
                Expr::Const(0.0),
                Expr::Const(0.0),
                Expr::Const(-2.0),
            ];
            let selected_jacobian = match assembly {
                RadauAssembly::ExprLegacy => Some(jacobian),
                RadauAssembly::AtomViewNative => None,
            };
            let callbacks = PreparedSymbolicCallbacks::prepare(
                assembly,
                residual,
                selected_jacobian,
                "t",
                &["y1", "y2"],
                &[],
            )
            .unwrap();
            let mut session = callbacks.session();
            let config = RadauConfig {
                execution: RadauExecution::Lambdify,
                assembly: Some(assembly),
                matrix_layout: layout,
                t_bound: 0.2,
                first_step: Some(0.1),
                rtol: 1.0e-7,
                atol: 1.0e-9,
                max_newton_iterations: 8,
                telemetry: super::super::new::telemetry::RadauTelemetryMode::Counters,
                ..RadauConfig::default()
            };
            let result = try_solve_symbolic_dense(&config, &mut session, &[1.0, 1.0]).unwrap();
            assert_eq!(result.t, config.t_bound);
            assert!((result.y[0] - (-0.2_f64).exp()).abs() < 1.0e-5);
            assert!((result.y[1] - (-0.4_f64).exp()).abs() < 1.0e-5);
            assert!(session.telemetry().counters.factorizations > 0);
            assert!(session.telemetry().counters.real_solves > 0);
            assert!(session.telemetry().counters.complex_solves > 0);
            assert!(session.telemetry().counters.newton_iterations > 0);
        }
    }
}

#[test]
fn new_config_default_is_valid_and_native() {
    let config = RadauConfig::default();
    assert_eq!(config.execution, RadauExecution::Native);
    assert!(config.validate().is_ok());
}

#[test]
fn radau5_dense_modified_newton_advances_linear_decay() {
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = -1.0;
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let config = RadauConfig {
        rtol: 1.0e-10,
        atol: 1.0e-12,
        max_newton_iterations: 8,
        ..RadauConfig::default()
    };
    let mut workspace = RadauWorkspace::default();
    let mut output = [0.0];
    let iterations = try_radau5_step(
        &config,
        &mut residual,
        &mut jacobian,
        0.0,
        0.1,
        &[1.0],
        &mut output,
        &mut workspace,
    )
    .unwrap();

    assert!(iterations.iterations <= config.max_newton_iterations);
    assert!(iterations.error_norm.is_finite());
    assert!((output[0] - (-0.1_f64).exp()).abs() < 1.0e-9);
    assert_eq!(workspace.dimension(), 1);
}

#[test]
fn radau5_embedded_error_norm_decreases_with_step_size() {
    let run = |h: f64| {
        let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
            output[0] = -state[0];
            Ok::<(), super::super::new::error::RadauError>(())
        };
        let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
            output[0] = -1.0;
            Ok::<(), super::super::new::error::RadauError>(())
        };
        let config = RadauConfig {
            rtol: 1.0e-10,
            atol: 1.0e-12,
            max_newton_iterations: 8,
            ..RadauConfig::default()
        };
        let mut workspace = RadauWorkspace::default();
        try_radau5_step(
            &config,
            &mut residual,
            &mut jacobian,
            0.0,
            h,
            &[1.0],
            &mut [0.0],
            &mut workspace,
        )
        .unwrap()
        .error_norm
    };
    let coarse = run(0.1);
    let fine = run(0.05);
    assert!(coarse.is_finite() && fine.is_finite());
    assert!(fine < coarse / 4.0, "coarse={coarse}, fine={fine}");
}

#[test]
fn symbolic_parameter_rebind_changes_dense_step_without_repreparation() {
    let callbacks = PreparedSymbolicCallbacks::prepare(
        RadauAssembly::ExprLegacy,
        vec![Expr::Mul(
            Box::new(Expr::Const(-1.0)),
            Box::new(Expr::Mul(
                Box::new(Expr::Var("p".to_string())),
                Box::new(Expr::Var("y1".to_string())),
            )),
        )],
        Some(vec![Expr::Mul(
            Box::new(Expr::Const(-1.0)),
            Box::new(Expr::Var("p".to_string())),
        )]),
        "t",
        &["y1"],
        &["p"],
    )
    .unwrap();
    let preparation = callbacks.session();
    let mut session = preparation;
    let config = RadauConfig {
        rtol: 1.0e-9,
        atol: 1.0e-11,
        max_newton_iterations: 8,
        ..RadauConfig::default()
    };
    let mut workspace = RadauWorkspace::default();
    let mut first = [0.0];
    session.rebind_parameters(&[1.0]).unwrap();
    try_radau5_symbolic_step(
        &config,
        &mut session,
        0.0,
        0.1,
        &[1.0],
        &mut first,
        &mut workspace,
    )
    .unwrap();
    let generation = session.generation();
    let mut second = [0.0];
    session.rebind_parameters(&[2.0]).unwrap();
    try_radau5_symbolic_step(
        &config,
        &mut session,
        0.0,
        0.1,
        &[1.0],
        &mut second,
        &mut workspace,
    )
    .unwrap();

    assert!(first[0] > second[0]);
    assert!(session.generation() > generation);
    assert!((first[0] - (-0.1_f64).exp()).abs() < 1.0e-8);
    assert!((second[0] - (-0.2_f64).exp()).abs() < 1.0e-8);
}

#[test]
fn radau5_dense_modified_newton_handles_nonlinear_scalar_decay() {
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -(state[0] * state[0]);
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let mut jacobian = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -2.0 * state[0];
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let config = RadauConfig {
        rtol: 1.0e-9,
        atol: 1.0e-11,
        max_newton_iterations: 8,
        ..RadauConfig::default()
    };
    let mut workspace = RadauWorkspace::default();
    let mut output = [0.0];
    let first_capacity = workspace.stage_states.capacity();
    let iterations = try_radau5_step(
        &config,
        &mut residual,
        &mut jacobian,
        0.0,
        0.1,
        &[1.0],
        &mut output,
        &mut workspace,
    )
    .unwrap();

    assert!(iterations.iterations <= config.max_newton_iterations);
    assert!(iterations.error_norm.is_finite());
    assert!((output[0] - 1.0 / 1.1).abs() < 1.0e-8);
    assert!(workspace.stage_states.capacity() >= first_capacity);

    let mut second_output = [0.0];
    try_radau5_step(
        &config,
        &mut residual,
        &mut jacobian,
        0.1,
        0.1,
        &output,
        &mut second_output,
        &mut workspace,
    )
    .unwrap();
    assert!((second_output[0] - 1.0 / 1.2).abs() < 1.0e-8);
}

#[test]
fn radau5_dense_step_rejects_non_finite_generic_callback_output() {
    let mut residual = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = f64::NAN;
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = 1.0;
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let mut workspace = RadauWorkspace::default();
    let error = try_radau5_step(
        &RadauConfig::default(),
        &mut residual,
        &mut jacobian,
        0.0,
        0.1,
        &[1.0],
        &mut [0.0],
        &mut workspace,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        super::super::new::error::RadauError::NonFiniteCallback {
            stage: RadauStage::Residual
        }
    ));
}

#[test]
fn dense_in_place_factorization_solves_pivoted_system() {
    let mut matrix = vec![0.0, 2.0, 1.0, 2.0];
    let mut pivots = vec![0; 2];
    factor_dense_in_place(&mut matrix, &mut pivots, 2).unwrap();
    let mut rhs = vec![4.0, 6.0];
    solve_factored_dense_in_place(&matrix, &pivots, &mut rhs, 2).unwrap();
    assert!((rhs[0] - 2.0).abs() < 1.0e-12);
    assert!((rhs[1] - 2.0).abs() < 1.0e-12);
}

#[test]
fn complex_in_place_factorization_solves_shifted_system() {
    let mut real = vec![2.0];
    let mut imag = vec![1.0];
    let mut pivots = vec![0];
    factor_complex_in_place(&mut real, &mut imag, &mut pivots, 1).unwrap();
    let mut rhs_real = [0.0];
    let mut rhs_imag = [5.0];
    solve_factored_complex_in_place(&real, &imag, &pivots, &mut rhs_real, &mut rhs_imag, 1)
        .unwrap();
    assert!((rhs_real[0] - 1.0).abs() < 1.0e-12);
    assert!((rhs_imag[0] - 2.0).abs() < 1.0e-12);
}

#[test]
fn prepared_model_rejects_zero_dimension_and_preserves_key_for_rebind() {
    let config = RadauConfig::default();
    assert!(RadauPreparedModel::prepare(0, &config).is_err());
    let prepared = RadauPreparedModel::prepare(3, &config).unwrap();
    assert_eq!(prepared.preparation_generation(), 1);
    assert!(prepared.can_rebind_values(prepared.key()));
}

#[test]
fn symbolic_frontends_are_explicitly_selectable_independent_routes() {
    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        let config = RadauConfig {
            execution: RadauExecution::Lambdify,
            assembly: Some(assembly),
            ..RadauConfig::default()
        };
        let prepared = RadauPreparedModel::prepare(2, &config).unwrap();
        assert_eq!(prepared.key().assembly, Some(assembly));
        assert_eq!(prepared.key().execution, RadauExecution::Lambdify);
    }
}

#[test]
fn callback_output_contract_reports_shape_and_non_finite_values() {
    assert!(validate_callback_output(RadauStage::Residual, 2, &[1.0]).is_err());
    assert!(validate_callback_output(RadauStage::Residual, 2, &[1.0, f64::NAN]).is_err());
    assert!(validate_callback_output(RadauStage::Residual, 2, &[1.0, 2.0]).is_ok());
}

#[test]
fn lambdify_callbacks_reuse_arguments_and_write_residual_and_jacobian_in_place() {
    let y = Expr::Var("y1".to_string());
    let p = Expr::Var("p".to_string());
    let residual = vec![Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Var("t".to_string())),
            Box::new(y.clone()),
        )),
        Box::new(p.clone()),
    )];
    let callbacks = LambdifyPlan::from_expressions(
        residual,
        Some(vec![Expr::Const(1.0)]),
        "t",
        &["y1"],
        &["p"],
    )
    .unwrap();
    let mut workspace = callbacks.workspace();
    let capacity = workspace.argument_capacity();
    let mut residual_output = [0.0];
    let mut jacobian_output = [0.0];

    callbacks
        .evaluate_residual(1.0, &[2.0], &[3.0], &mut residual_output, &mut workspace)
        .unwrap();
    callbacks
        .evaluate_jacobian(1.0, &[2.0], &[3.0], &mut jacobian_output, &mut workspace)
        .unwrap();

    assert_eq!(residual_output, [6.0]);
    assert_eq!(jacobian_output, [1.0]);
    assert_eq!(workspace.argument_capacity(), capacity);
    assert!(
        callbacks
            .evaluate_residual(1.0, &[], &[3.0], &mut residual_output, &mut workspace)
            .is_err()
    );
}

#[test]
fn lambdify_parameter_series_reuses_prepared_plan_and_workspace() {
    let y = Expr::Var("y1".to_string());
    let p = Expr::Var("p".to_string());
    let plan = LambdifyPlan::from_expressions(
        vec![Expr::Mul(Box::new(p.clone()), Box::new(y.clone()))],
        Some(vec![p]),
        "t",
        &["y1"],
        &["p"],
    )
    .unwrap();
    let pattern = plan.jacobian_pattern().to_vec();
    let mut workspace = plan.workspace();
    let capacity = workspace.argument_capacity();
    let mut residual = [0.0];
    let mut jacobian = [0.0];

    for (parameter, expected_residual) in [(2.0, 6.0), (3.0, 9.0), (-1.5, -4.5)] {
        plan.evaluate_residual(0.25, &[3.0], &[parameter], &mut residual, &mut workspace)
            .unwrap();
        plan.evaluate_jacobian(0.25, &[3.0], &[parameter], &mut jacobian, &mut workspace)
            .unwrap();

        assert_eq!(residual, [expected_residual]);
        assert_eq!(jacobian, [parameter]);
        assert_eq!(workspace.argument_capacity(), capacity);
    }

    assert_eq!(pattern, [(0, 0)]);
}

#[test]
fn lambdify_callback_session_rebinds_values_without_rebuilding_or_growing_workspace() {
    let callbacks = PreparedSymbolicCallbacks::prepare(
        RadauAssembly::ExprLegacy,
        vec![Expr::Mul(
            Box::new(Expr::Var("p".to_string())),
            Box::new(Expr::Var("y1".to_string())),
        )],
        Some(vec![Expr::Var("p".to_string())]),
        "t",
        &["y1"],
        &["p"],
    )
    .unwrap();
    let mut session = callbacks.session();
    let capacity = session.workspace_capacity();
    assert_eq!(session.parameters(), &[0.0]);
    assert_eq!(session.generation(), 1);

    let mut residual = [0.0];
    let mut jacobian = [0.0];
    for (parameter, expected) in [(2.0, 6.0), (3.0, 9.0)] {
        session.rebind_parameters(&[parameter]).unwrap();
        session
            .evaluate_residual(0.0, &[3.0], &mut residual)
            .unwrap();
        session
            .evaluate_jacobian(0.0, &[3.0], &mut jacobian)
            .unwrap();
        assert_eq!(residual, [expected]);
        assert_eq!(jacobian, [parameter]);
        assert_eq!(session.workspace_capacity(), capacity);
    }
    assert_eq!(session.generation(), 3);
    assert!(session.rebind_parameters(&[]).is_err());
    assert_eq!(session.parameters(), &[3.0]);
}

#[test]
fn lambdify_rejects_missing_jacobian_and_out_of_band_layout() {
    let y1 = Expr::Var("y1".to_string());
    let y2 = Expr::Var("y2".to_string());
    let residual = vec![y1.clone(), y2.clone()];
    let no_jacobian =
        LambdifyPlan::from_expressions(residual.clone(), None, "t", &["y1", "y2"], &[]).unwrap();
    let mut no_jacobian_workspace = no_jacobian.workspace();
    let mut no_jacobian_output = [0.0; 4];
    assert!(matches!(
        no_jacobian.evaluate_jacobian(
            0.0,
            &[1.0, 2.0],
            &[],
            &mut no_jacobian_output,
            &mut no_jacobian_workspace,
        ),
        Err(super::super::new::error::RadauError::UnsupportedRoute(
            super::super::new::error::RadauUnsupportedRoute::LambdifyJacobianNotPrepared,
        ))
    ));

    let out_of_band = LambdifyPlan::from_expressions(
        residual,
        Some(vec![
            Expr::Const(1.0),
            Expr::Const(0.0),
            Expr::Const(2.0),
            Expr::Const(1.0),
        ]),
        "t",
        &["y1", "y2"],
        &[],
    )
    .unwrap();
    let mut workspace = out_of_band.workspace();
    let mut output = [0.0; 4];
    assert!(matches!(
        out_of_band.evaluate_jacobian_layout(
            0.0,
            &[1.0, 2.0],
            &[],
            RadauMatrixLayout::Banded { lower: 0, upper: 1 },
            &mut output,
            &mut workspace,
        ),
        Err(super::super::new::error::RadauError::UnsupportedRoute(
            super::super::new::error::RadauUnsupportedRoute::JacobianOutsideBandedLayout,
        ))
    ));
}

#[test]
fn atom_native_matches_lambdify_and_reuses_callback_workspace_across_rebinds() {
    let y1 = Expr::Var("y1".to_string());
    let y2 = Expr::Var("y2".to_string());
    let p = Expr::Var("p".to_string());
    let residual = vec![
        Expr::Add(
            Box::new(Expr::Add(
                Box::new(Expr::Var("t".to_string())),
                Box::new(y1.clone()),
            )),
            Box::new(p.clone()),
        ),
        Expr::Add(
            Box::new(Expr::Mul(Box::new(y1.clone()), Box::new(y1.clone()))),
            Box::new(y2.clone()),
        ),
    ];
    let jacobian = vec![
        Expr::Const(1.0),
        Expr::Const(0.0),
        Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(y1)),
        Expr::Const(1.0),
    ];
    let lambdify = LambdifyPlan::from_expressions(
        residual.clone(),
        Some(jacobian.clone()),
        "t",
        &["y1", "y2"],
        &["p"],
    )
    .unwrap();
    let atom =
        AtomNativePlan::from_expressions(residual, None, "t", &["y1", "y2"], &["p"]).unwrap();
    let mut lambdify_workspace = lambdify.workspace();
    let mut atom_workspace = atom.workspace();
    let atom_capacity = atom_workspace.value_capacity();
    let mut lambdify_residual = [0.0; 2];
    let mut atom_residual = [0.0; 2];
    let mut lambdify_jacobian = [0.0; 4];
    let mut atom_jacobian = [0.0; 4];

    lambdify
        .evaluate_residual(
            1.0,
            &[2.0, 3.0],
            &[4.0],
            &mut lambdify_residual,
            &mut lambdify_workspace,
        )
        .unwrap();
    atom.evaluate_residual(
        1.0,
        &[2.0, 3.0],
        &[4.0],
        &mut atom_residual,
        &mut atom_workspace,
    )
    .unwrap();
    lambdify
        .evaluate_jacobian(
            1.0,
            &[2.0, 3.0],
            &[4.0],
            &mut lambdify_jacobian,
            &mut lambdify_workspace,
        )
        .unwrap();
    atom.evaluate_jacobian(
        1.0,
        &[2.0, 3.0],
        &[4.0],
        &mut atom_jacobian,
        &mut atom_workspace,
    )
    .unwrap();

    assert_eq!(atom_residual, lambdify_residual);
    assert_eq!(atom_jacobian, lambdify_jacobian);
    assert_eq!(atom_residual, [7.0, 7.0]);
    assert_eq!(atom_jacobian, [1.0, 0.0, 4.0, 1.0]);
    assert_eq!(atom_workspace.value_capacity(), atom_capacity);

    atom.evaluate_residual(
        1.0,
        &[2.0, 3.0],
        &[5.0],
        &mut atom_residual,
        &mut atom_workspace,
    )
    .unwrap();
    assert_eq!(atom_residual, [8.0, 7.0]);
    assert_eq!(atom_workspace.value_capacity(), atom_capacity);
}

#[test]
fn atom_native_derives_only_state_dependent_jacobian_entries() {
    let y1 = Expr::Var("y1".to_string());
    let y2 = Expr::Var("y2".to_string());
    let plan = AtomNativePlan::from_expressions(
        vec![
            Expr::Add(Box::new(y1.clone()), Box::new(Expr::Const(1.0))),
            Expr::Mul(Box::new(y2.clone()), Box::new(y2)),
        ],
        None,
        "t",
        &["y1", "y2"],
        &[],
    )
    .unwrap();
    assert_eq!(plan.jacobian_entry_count(), 2);
}

#[test]
fn symbolic_frontend_dispatch_keeps_one_callback_contract() {
    let residual = vec![Expr::Add(
        Box::new(Expr::Var("y1".to_string())),
        Box::new(Expr::Var("p".to_string())),
    )];
    let jacobian = vec![Expr::Const(1.0)];
    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        let selected_jacobian = match assembly {
            RadauAssembly::ExprLegacy => Some(jacobian.clone()),
            RadauAssembly::AtomViewNative => None,
        };
        let callbacks = PreparedSymbolicCallbacks::prepare(
            assembly,
            residual.clone(),
            selected_jacobian,
            "t",
            &["y1"],
            &["p"],
        )
        .unwrap();
        let mut workspace = callbacks.workspace();
        let mut residual_output = [0.0];
        let mut jacobian_output = [0.0];
        callbacks
            .evaluate_residual(0.0, &[2.0], &[3.0], &mut residual_output, &mut workspace)
            .unwrap();
        callbacks
            .evaluate_jacobian(0.0, &[2.0], &[3.0], &mut jacobian_output, &mut workspace)
            .unwrap();
        assert_eq!(residual_output, [5.0]);
        assert_eq!(jacobian_output, [1.0]);
        assert!(matches!(
            (&callbacks, &workspace),
            (
                PreparedSymbolicCallbacks::ExprLegacy(_),
                SymbolicCallbackWorkspace::ExprLegacy(_)
            ) | (
                PreparedSymbolicCallbacks::AtomViewNative(_),
                SymbolicCallbackWorkspace::AtomViewNative(_)
            )
        ));
    }
}

#[test]
fn symbolic_frontends_emit_dense_sparse_and_banded_jacobian_layouts() {
    let y1 = Expr::Var("y1".to_string());
    let y2 = Expr::Var("y2".to_string());
    let p = Expr::Var("p".to_string());
    let residual = vec![
        Expr::Add(
            Box::new(Expr::Add(
                Box::new(Expr::Var("t".to_string())),
                Box::new(y1.clone()),
            )),
            Box::new(p),
        ),
        Expr::Add(
            Box::new(Expr::Mul(Box::new(y1), Box::new(Expr::Const(2.0)))),
            Box::new(y2),
        ),
    ];
    let explicit_jacobian = vec![
        Expr::Const(1.0),
        Expr::Const(0.0),
        Expr::Const(2.0),
        Expr::Const(1.0),
    ];

    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        let jacobian = match assembly {
            RadauAssembly::ExprLegacy => Some(explicit_jacobian.clone()),
            RadauAssembly::AtomViewNative => None,
        };
        let callbacks = PreparedSymbolicCallbacks::prepare(
            assembly,
            residual.clone(),
            jacobian,
            "t",
            &["y1", "y2"],
            &["p"],
        )
        .unwrap();
        let mut workspace = callbacks.workspace();

        let mut dense = [0.0; 4];
        callbacks
            .evaluate_jacobian_layout(
                0.0,
                &[3.0, 4.0],
                &[5.0],
                RadauMatrixLayout::Dense,
                &mut dense,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(dense, [1.0, 0.0, 2.0, 1.0]);

        let pattern = callbacks.jacobian_pattern();
        let mut sparse = vec![0.0; pattern.len()];
        callbacks
            .evaluate_jacobian_layout(
                0.0,
                &[3.0, 4.0],
                &[5.0],
                RadauMatrixLayout::Sparse,
                &mut sparse,
                &mut workspace,
            )
            .unwrap();
        for ((row, col), value) in pattern.iter().copied().zip(sparse.iter().copied()) {
            assert_eq!(value, dense[row * 2 + col]);
        }

        let mut banded = [0.0; 4];
        callbacks
            .evaluate_jacobian_layout(
                0.0,
                &[3.0, 4.0],
                &[5.0],
                RadauMatrixLayout::Banded { lower: 1, upper: 0 },
                &mut banded,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(banded, [1.0, 1.0, 2.0, 0.0]);
    }
}
