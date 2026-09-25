//! Debug correctness gates for LSODE2 Lambdify routes.
//!
//! These tests intentionally avoid release timing claims. They protect the
//! contracts that performance work must not change: solver trajectory,
//! parameter invalidation, non-finite callback behavior, fixed Sparse order,
//! and compact Banded slot mapping.

use super::native_jacobian::{NativeJacobianStorage, try_prepare_native_atomview_jacobian_runtime};
use super::solver::Lsode2TelemetryScope;
use super::{
    IvpLambdifyExecutionPolicy, IvpTelemetry, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2Solver, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::ivp_telemetry::IvpWarmStage;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions,
    prepare_symbolic_ivp_problem,
};
use nalgebra::DVector;
use std::sync::{Arc, RwLock};

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn trajectory_config(
    assembly: Lsode2SymbolicAssemblyBackend,
    parameter: f64,
    telemetry: IvpTelemetry,
) -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-a*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.5,
        0.025,
        1.0e-9,
        1.0e-11,
    )
    .with_equation_parameters(vec!["a".to_string()])
    .with_equation_parameter_values(DVector::from_vec(vec![parameter]))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_native_banded_faithful_backend()
    .with_faithful_bdf_solve(2_000, 2_000)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_telemetry(telemetry)
}

fn bridge_counter_config(
    assembly: Lsode2SymbolicAssemblyBackend,
    telemetry: IvpTelemetry,
) -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-a*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.5,
        0.025,
        1.0e-9,
        1.0e-11,
    )
    .with_equation_parameters(vec!["a".to_string()])
    .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_bridge_solve()
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_telemetry(telemetry)
}

#[test]
fn lsode2_debug_evaluation_counter_scope_is_explicit_and_not_mixed() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_evaluation_counter_scope_is_explicit_and_not_mixed",
    );

    let bridge_telemetry = IvpTelemetry::detailed();
    let native_telemetry = IvpTelemetry::detailed();
    let mut bridge = Lsode2Solver::new(bridge_counter_config(
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        bridge_telemetry.clone(),
    ))
    .expect("bridge counter fixture should construct");
    let mut native = Lsode2Solver::new(trajectory_config(
        Lsode2SymbolicAssemblyBackend::AtomView,
        2.0,
        native_telemetry.clone(),
    ))
    .expect("native counter fixture should construct");

    let bridge_summary = bridge
        .solve_with_summary()
        .expect("bridge counter fixture should solve");
    let native_summary = native
        .solve_with_summary()
        .expect("native counter fixture should solve");
    let bridge_evaluations = bridge_summary.evaluation_telemetry;
    let native_evaluations = native_summary.evaluation_telemetry;
    let bridge_callbacks = bridge_telemetry.snapshot();
    let native_callbacks = native_telemetry.snapshot();

    assert_eq!(
        bridge_evaluations.scope,
        Lsode2TelemetryScope::BridgeBdfCallbacks
    );
    assert_eq!(
        native_evaluations.scope,
        Lsode2TelemetryScope::NativeFaithfulInnerLoop
    );
    assert!(bridge_evaluations.residual_evaluations > 0);
    assert!(bridge_evaluations.jacobian_evaluations > 0);
    assert!(bridge_evaluations.linear_solves > 0);
    assert!(native_evaluations.residual_evaluations > 0);
    assert!(native_evaluations.jacobian_evaluations > 0);
    assert!(native_evaluations.linear_solves > 0);
    assert!(bridge_callbacks.residual_requests > 0);
    assert!(native_callbacks.residual_requests > 0);

    reportln!(
        "[LSODE2 evaluation counter scope] bridge_scope={}; native_scope={}; bridge_note={}; native_note={}",
        bridge_evaluations.scope.label(),
        native_evaluations.scope.label(),
        bridge_evaluations.scope.counter_note(),
        native_evaluations.scope.counter_note(),
    );
    reportln!(
        "route | solver_residuals | solver_jacobians | solver_linear_solves | accepted | rejected | callback_residual_requests | callback_jacobian_requests",
    );
    reportln!(
        "bridge-bdf | {} | {} | {} | {} | {} | {} | {}",
        bridge_evaluations.residual_evaluations,
        bridge_evaluations.jacobian_evaluations,
        bridge_evaluations.linear_solves,
        bridge_evaluations.accepted_steps,
        bridge_evaluations.rejected_steps,
        bridge_callbacks.residual_requests,
        bridge_callbacks.jacobian_requests,
    );
    reportln!(
        "native-faithful | {} | {} | {} | {} | {} | {} | {}",
        native_evaluations.residual_evaluations,
        native_evaluations.jacobian_evaluations,
        native_evaluations.linear_solves,
        native_evaluations.accepted_steps,
        native_evaluations.rejected_steps,
        native_callbacks.residual_requests,
        native_callbacks.jacobian_requests,
    );
}

#[test]
fn lsode2_debug_trajectory_parity_exprlegacy_vs_atomview_native() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_trajectory_parity_exprlegacy_vs_atomview_native",
    );

    let mut expr = Lsode2Solver::new(trajectory_config(
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        2.0,
        IvpTelemetry::counters(),
    ))
    .expect("ExprLegacy trajectory fixture should construct");
    let mut atom = Lsode2Solver::new(trajectory_config(
        Lsode2SymbolicAssemblyBackend::AtomView,
        2.0,
        IvpTelemetry::counters(),
    ))
    .expect("AtomViewNative trajectory fixture should construct");

    let expr_summary = expr
        .solve_with_summary()
        .expect("ExprLegacy trajectory fixture should solve");
    let atom_summary = atom
        .solve_with_summary()
        .expect("AtomViewNative trajectory fixture should solve");
    let (expr_times, expr_values) = expr.get_result();
    let (atom_times, atom_values) = atom.get_result();

    assert_eq!(expr_times.len(), atom_times.len());
    assert_eq!(expr_values.shape(), atom_values.shape());
    let max_time_diff = expr_times
        .iter()
        .zip(atom_times.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    let max_value_diff = expr_values
        .iter()
        .zip(atom_values.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    let expr_telemetry = expr_summary.evaluation_telemetry;
    let atom_telemetry = atom_summary.evaluation_telemetry;

    reportln!(
        "[LSODE2 trajectory parity] fixture=scalar exponential; matrix=Banded; tolerance=debug"
    );
    reportln!(
        "route | points | max_time_diff | max_state_diff | residual_calls | jacobian_calls | linear_solves | accepted | rejected"
    );
    reportln!(
        "ExprLegacy | {} | {:.3e} | {:.3e} | {} | {} | {} | {} | {}",
        expr_times.len(),
        max_time_diff,
        max_value_diff,
        expr_telemetry.residual_evaluations,
        expr_telemetry.jacobian_evaluations,
        expr_telemetry.linear_solves,
        expr_telemetry.accepted_steps,
        expr_telemetry.rejected_steps,
    );
    reportln!(
        "AtomViewNative | {} | {:.3e} | {:.3e} | {} | {} | {} | {} | {}",
        atom_times.len(),
        max_time_diff,
        max_value_diff,
        atom_telemetry.residual_evaluations,
        atom_telemetry.jacobian_evaluations,
        atom_telemetry.linear_solves,
        atom_telemetry.accepted_steps,
        atom_telemetry.rejected_steps,
    );
    reportln!(
        "algorithm | controller | active | mused | mcur | preferred | executed | reason | bdf_order | bdf_max_order",
    );
    reportln!(
        "ExprLegacy | {} | {} | {} | {} | {} | {:?} | {} | {:?} | {:?}",
        expr_summary.algorithm.controller_mode,
        expr_summary.algorithm.active_family,
        expr_summary.algorithm.mused_family,
        expr_summary.algorithm.mcur_family,
        expr_summary.algorithm.preferred_family,
        expr_summary.algorithm.executed_family,
        expr_summary.algorithm.switch_reason,
        expr_summary.algorithm.bdf_current_order,
        expr_summary.algorithm.bdf_max_order_cap,
    );
    reportln!(
        "AtomViewNative | {} | {} | {} | {} | {} | {:?} | {} | {:?} | {:?}",
        atom_summary.algorithm.controller_mode,
        atom_summary.algorithm.active_family,
        atom_summary.algorithm.mused_family,
        atom_summary.algorithm.mcur_family,
        atom_summary.algorithm.preferred_family,
        atom_summary.algorithm.executed_family,
        atom_summary.algorithm.switch_reason,
        atom_summary.algorithm.bdf_current_order,
        atom_summary.algorithm.bdf_max_order_cap,
    );

    assert!(max_time_diff <= 1.0e-12);
    assert!(max_value_diff <= 1.0e-9);
    assert_eq!(
        expr_summary.algorithm, atom_summary.algorithm,
        "symbolic frontend must not alter the public algorithm snapshot"
    );
    assert_eq!(
        expr_telemetry.residual_evaluations,
        atom_telemetry.residual_evaluations
    );
    assert_eq!(
        expr_telemetry.jacobian_evaluations,
        atom_telemetry.jacobian_evaluations
    );
    assert_eq!(expr_telemetry.linear_solves, atom_telemetry.linear_solves);
    assert_eq!(expr_telemetry.accepted_steps, atom_telemetry.accepted_steps);
    assert_eq!(expr_telemetry.rejected_steps, atom_telemetry.rejected_steps);
}

#[test]
fn lsode2_debug_parameter_rebind_invalidates_prepared_solver_state() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_parameter_rebind_invalidates_prepared_solver_state",
    );

    let telemetry = IvpTelemetry::counters();
    let mut solver = Lsode2Solver::new(trajectory_config(
        Lsode2SymbolicAssemblyBackend::AtomView,
        1.0,
        telemetry,
    ))
    .expect("parameter invalidation fixture should construct");
    solver
        .prepare()
        .expect("initial preparation should succeed");
    assert!(solver.is_prepared());

    let invalid = solver.set_parameter_values(DVector::from_vec(vec![2.0, 3.0]));
    assert!(matches!(
        invalid,
        Err(super::Lsode2Error::GeneratedBackend(
            IvpBackendError::ParameterCountMismatch {
                expected: 1,
                actual: 2
            }
        ))
    ));
    assert!(
        solver.is_prepared(),
        "rejected rebind must not invalidate state"
    );

    solver
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .expect("valid parameter rebind should succeed");
    assert!(
        !solver.is_prepared(),
        "valid rebind must invalidate prepared state"
    );
    solver
        .prepare()
        .expect("rebound preparation should succeed");
    assert!(solver.is_prepared());
    let rebound_summary = solver
        .solve_with_summary()
        .expect("rebound solver should solve");
    let (rebound_times, rebound_values) = solver.get_result();

    let mut fresh = Lsode2Solver::new(trajectory_config(
        Lsode2SymbolicAssemblyBackend::AtomView,
        2.0,
        IvpTelemetry::counters(),
    ))
    .expect("fresh rebound fixture should construct");
    let fresh_summary = fresh
        .solve_with_summary()
        .expect("fresh rebound fixture should solve");
    let (fresh_times, fresh_values) = fresh.get_result();
    let max_time_diff = rebound_times
        .iter()
        .zip(fresh_times.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    let max_value_diff = rebound_values
        .iter()
        .zip(fresh_values.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);

    reportln!(
        "[LSODE2 invalidation] rejected_rebind=typed; valid_rebind=invalidates; stale_factor_or_callback_diff={max_value_diff:.3e}; time_diff={max_time_diff:.3e}; rebound_status={}; fresh_status={}",
        rebound_summary.status,
        fresh_summary.status,
    );
    assert!(max_time_diff <= 1.0e-12);
    assert!(max_value_diff <= 1.0e-9);
}

#[test]
fn lsode2_debug_nonfinite_callbacks_and_typed_shape_errors() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_nonfinite_callbacks_and_typed_shape_errors",
    );
    let telemetry = IvpTelemetry::counters();
    let problem = prepare_symbolic_ivp_problem(
        vec![
            Expr::parse_expression("a*y + exp(t)"),
            Expr::parse_expression("y*y - a"),
        ],
        vec!["y".to_string(), "z".to_string()],
        "t".to_string(),
        SymbolicIvpProblemOptions::new()
            .with_equation_parameters(vec!["a".to_string()])
            .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
            .with_telemetry(telemetry),
    )
    .expect("non-finite callback fixture should prepare");

    let nonfinite_state = DVector::from_vec(vec![f64::NAN, 1.0]);
    let residual = problem
        .try_evaluate_residual(0.5, &nonfinite_state)
        .expect("non-finite state should remain a typed callback result");
    let jacobian = problem
        .try_evaluate_jacobian(0.5, &nonfinite_state)
        .expect("Jacobian callback should not panic on non-finite state");
    assert!(residual.iter().all(|value| value.is_nan()));
    assert!(
        jacobian
            .iter()
            .all(|value| value.is_finite() || value.is_nan())
    );
    assert!(jacobian[(1, 0)].is_nan());

    for state_value in [f64::INFINITY, f64::NEG_INFINITY, 1.0e308, 1.0e-308] {
        let state = DVector::from_vec(vec![state_value, 1.0]);
        let residual = problem
            .try_evaluate_residual(0.5, &state)
            .expect("infinite, overflow and underflow inputs must remain typed results");
        let jacobian = problem
            .try_evaluate_jacobian(0.5, &state)
            .expect("domain-extreme Jacobian input must not panic");
        assert_eq!(residual.len(), 2);
        assert_eq!(jacobian.shape(), (2, 2));
    }

    let mut wrong_residual = DVector::zeros(1);
    assert!(matches!(
        problem.try_evaluate_residual_into(0.5, &nonfinite_state, &mut wrong_residual),
        Err(IvpBackendError::InvalidOutputShape { stage, expected: 2, actual: 1 })
            if stage == "residual"
    ));
    let wrong_state = DVector::zeros(1);
    assert!(matches!(
        problem.try_evaluate_residual_into(0.5, &wrong_state, &mut DVector::zeros(2)),
        Err(IvpBackendError::InvalidStateShape {
            expected: 2,
            actual: 1
        })
    ));
    reportln!(
        "[LSODE2 non-finite contract] nan=propagates; +/-inf=typed_result; overflow_underflow=typed_result; shape_errors=typed; panic_free=true"
    );
}

#[test]
fn lsode2_debug_sparse_order_and_banded_slots_are_stable() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_sparse_order_and_banded_slots_are_stable",
    );
    let equations = vec![
        Expr::parse_expression("-2*y1 + y2"),
        Expr::parse_expression("3*y1 - 4*y2"),
    ];
    let variables = vec!["y1".to_string(), "y2".to_string()];
    let state = DVector::from_vec(vec![1.0, 2.0]);
    let mut sparse = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::SparseTriplets,
        IvpTelemetry::counters(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("Sparse layout fixture should prepare");
    let mut banded = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::Banded { bandwidth: None },
        IvpTelemetry::counters(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("Banded layout fixture should prepare");
    assert_eq!(
        sparse.sparse_pattern(),
        vec![(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    assert_eq!(banded.banded_layout(), Some((1, 1, 6)));

    let mut sparse_values = vec![f64::NAN; 4];
    sparse
        .try_evaluate_sparse_values_into(0.0, &state, &mut sparse_values)
        .expect("Sparse values should fill in fixed pattern order");
    let mut banded_values = vec![f64::NAN; 6];
    banded
        .try_evaluate_banded_values_into(0.0, &state, &mut banded_values)
        .expect("Banded values should fill in compact slot order");
    let banded_matrix = Banded::from_vec(2, 1, 1, banded_values)
        .expect("compact Banded values should have the declared layout");
    assert_eq!(sparse_values, vec![-2.0, 1.0, 3.0, -4.0]);
    assert_eq!(banded_matrix[(0, 0)], -2.0);
    assert_eq!(banded_matrix[(0, 1)], 1.0);
    assert_eq!(banded_matrix[(1, 0)], 3.0);
    assert_eq!(banded_matrix[(1, 1)], -4.0);

    reportln!(
        "[LSODE2 layout correctness] sparse_pattern=[(0,0),(0,1),(1,0),(1,1)]; band_kl=1; band_ku=1; compact_slots=6; values_match=true"
    );
}

#[test]
fn lsode2_debug_high_cardinality_parameter_rebind_is_parity_safe() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_high_cardinality_parameter_rebind_is_parity_safe",
    );

    for parameter_count in [32_usize, 128, 256] {
        let parameter_names = (0..parameter_count)
            .map(|index| format!("p{index}"))
            .collect::<Vec<_>>();
        // Keep each expression shallow. A single 256-term sum would test the
        // parser's recursion depth instead of the prepared parameter contract.
        let equations = (0..parameter_count)
            .map(|index| Expr::parse_expression(&format!("y{index} + p{index}*exp(-t)")))
            .collect::<Vec<_>>();
        let variables = (0..parameter_count)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        let initial_values = DVector::from_iterator(
            parameter_count,
            (0..parameter_count).map(|index| 0.01 + index as f64 * 0.001),
        );
        let rebound_values = DVector::from_iterator(
            parameter_count,
            (0..parameter_count).map(|index| 0.02 + index as f64 * 0.002),
        );
        let state = DVector::from_element(parameter_count, 0.75);

        for backend in [
            IvpSymbolicAssemblyBackend::ExprLegacy,
            IvpSymbolicAssemblyBackend::AtomView,
        ] {
            let options = SymbolicIvpProblemOptions::new()
                .with_equation_parameters(parameter_names.clone())
                .with_equation_parameter_values(initial_values.clone())
                .with_symbolic_assembly_backend(backend)
                .with_telemetry(IvpTelemetry::counters());
            let problem = prepare_symbolic_ivp_problem(
                equations.clone(),
                variables.clone(),
                "t".to_string(),
                options,
            )
            .expect("high-cardinality parameter fixture should prepare");

            let before = problem
                .try_evaluate_residual(0.25, &state)
                .expect("initial high-cardinality residual should evaluate");
            let invalid =
                problem.set_parameter_values(DVector::from_element(parameter_count - 1, 1.0));
            assert!(matches!(
                invalid,
                Err(IvpBackendError::ParameterCountMismatch {
                    expected,
                    actual
                }) if expected == parameter_count && actual == parameter_count - 1
            ));
            let after_failed_rebind = problem
                .try_evaluate_residual(0.25, &state)
                .expect("failed rebind must preserve the previous callback state");
            assert_eq!(before, after_failed_rebind);

            problem
                .set_parameter_values(rebound_values.clone())
                .expect("valid high-cardinality rebind should succeed");
            let rebound = problem
                .try_evaluate_residual(0.25, &state)
                .expect("rebound high-cardinality residual should evaluate");

            let fresh = prepare_symbolic_ivp_problem(
                equations.clone(),
                variables.clone(),
                "t".to_string(),
                SymbolicIvpProblemOptions::new()
                    .with_equation_parameters(parameter_names.clone())
                    .with_equation_parameter_values(rebound_values.clone())
                    .with_symbolic_assembly_backend(backend)
                    .with_telemetry(IvpTelemetry::counters()),
            )
            .expect("fresh high-cardinality fixture should prepare");
            let fresh_value = fresh
                .try_evaluate_residual(0.25, &state)
                .expect("fresh high-cardinality residual should evaluate");
            assert!((rebound[0] - fresh_value[0]).abs() <= 1.0e-12);
            assert!((rebound[0] - before[0]).abs() > 1.0e-6);

            reportln!(
                "[LSODE2 high-cardinality parameters] backend={backend:?}; parameter_count={parameter_count}; failed_rebind=typed; failed_rebind_preserves_value=true; rebound_vs_fresh_diff={:.3e}",
                (rebound[0] - fresh_value[0]).abs(),
            );
        }
    }

    let missing_names = (0..32).map(|index| format!("p{index}")).collect();
    let missing = prepare_symbolic_ivp_problem(
        vec![Expr::parse_expression("y0 + p0")],
        vec!["y0".to_string()],
        "t".to_string(),
        SymbolicIvpProblemOptions::new()
            .with_equation_parameters(missing_names)
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
            .with_telemetry(IvpTelemetry::counters()),
    );
    assert!(matches!(
        missing,
        Err(IvpBackendError::MissingParameterValues { expected: 32 })
    ));
}

#[test]
fn lsode2_debug_structural_jacobian_layout_corpus_is_componentwise_stable() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_structural_jacobian_layout_corpus_is_componentwise_stable",
    );

    let cases = [
        (
            "diagonal",
            vec!["2*y0", "3*y1", "4*y2"],
            vec![(0, 0), (1, 1), (2, 2)],
            (0, 0),
        ),
        (
            "structural-zero-row",
            vec!["y0", "y0-y0", "y2"],
            vec![(0, 0), (2, 2)],
            (0, 0),
        ),
        (
            "maximum-bandwidth",
            vec!["y2", "y1", "y0"],
            vec![(0, 2), (1, 1), (2, 0)],
            (2, 2),
        ),
    ];

    for (name, equation_strings, expected_pattern, expected_bandwidth) in cases {
        let equations = equation_strings
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        let variables = vec!["y0".to_string(), "y1".to_string(), "y2".to_string()];
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut sparse = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::SparseTriplets,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("structural Sparse fixture should prepare");
        assert_eq!(sparse.sparse_pattern(), expected_pattern);
        let mut sparse_values = vec![f64::NAN; expected_pattern.len()];
        sparse
            .try_evaluate_sparse_values_into(0.0, &state, &mut sparse_values)
            .expect("structural Sparse fixture should evaluate");

        let mut dense = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::Dense,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("structural Dense fixture should prepare");
        let mut dense_values = nalgebra::DMatrix::from_element(3, 3, f64::NAN);
        dense
            .try_evaluate_dense_into(0.0, &state, &mut dense_values)
            .expect("structural Dense fixture should evaluate");
        for ((row, col), value) in expected_pattern.iter().zip(sparse_values.iter()) {
            assert!((dense_values[(*row, *col)] - value).abs() <= 1.0e-12);
        }

        let mut banded = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::Banded { bandwidth: None },
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("structural Banded fixture should prepare");
        assert_eq!(
            banded.banded_layout().map(|(kl, ku, _)| (kl, ku)),
            Some(expected_bandwidth)
        );
        let banded_slots = banded
            .banded_layout()
            .expect("structural Banded layout should be available")
            .2;
        let mut banded_values = vec![f64::NAN; banded_slots];
        banded
            .try_evaluate_banded_values_into(0.0, &state, &mut banded_values)
            .expect("structural Banded fixture should evaluate");
        let (band_kl, band_ku, _) = banded
            .banded_layout()
            .expect("structural Banded layout should be available");
        let banded_matrix = Banded::from_vec(3, band_kl, band_ku, banded_values)
            .expect("structural Banded values should have the prepared layout");
        for row in 0..3 {
            for col in 0..3 {
                if (row as isize - col as isize) <= band_kl as isize
                    && (col as isize - row as isize) <= band_ku as isize
                {
                    assert!(
                        (dense_values[(row, col)] - banded_matrix[(row, col)]).abs() <= 1.0e-12,
                        "Banded value differs from Dense at ({row}, {col}) in {name}"
                    );
                }
            }
        }

        reportln!(
            "[LSODE2 structural layout corpus] case={name}; sparse_nnz={}; band_kl_ku={:?}; values_parity=true",
            expected_pattern.len(),
            expected_bandwidth,
        );
    }
}

#[test]
fn lsode2_debug_wider_boundary_sparse_and_banded_layouts_match_dense() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_wider_boundary_sparse_and_banded_layouts_match_dense",
    );

    let equations = ["2*y0 + y1", "y0 + 3*y1 + y2", "y1 + 4*y2 + y3", "y2 + 5*y3"]
        .into_iter()
        .map(Expr::parse_expression)
        .collect::<Vec<_>>();
    let variables = (0..4).map(|index| format!("y{index}")).collect::<Vec<_>>();
    let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let expected_pattern = vec![
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ];
    let telemetry = IvpTelemetry::counters();
    let mut sparse = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::SparseTriplets,
        telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("wider-boundary Sparse fixture should prepare");
    assert_eq!(sparse.sparse_pattern(), expected_pattern);

    let mut sparse_values = vec![f64::NAN; expected_pattern.len()];
    sparse
        .try_evaluate_sparse_values_into(0.0, &state, &mut sparse_values)
        .expect("wider-boundary Sparse fixture should evaluate");

    let mut dense = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::Dense,
        telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("wider-boundary Dense fixture should prepare");
    let mut dense_values = nalgebra::DMatrix::from_element(4, 4, f64::NAN);
    dense
        .try_evaluate_dense_into(0.0, &state, &mut dense_values)
        .expect("wider-boundary Dense fixture should evaluate");

    for ((row, col), value) in expected_pattern.iter().zip(sparse_values.iter()) {
        assert!((dense_values[(*row, *col)] - value).abs() <= 1.0e-12);
    }

    let mut banded = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        None,
        None,
        NativeJacobianStorage::Banded { bandwidth: None },
        telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("wider-boundary Banded fixture should prepare");
    assert_eq!(
        banded.banded_layout().map(|(kl, ku, _)| (kl, ku)),
        Some((1, 1))
    );
    let (band_kl, band_ku, band_slots) = banded
        .banded_layout()
        .expect("wider-boundary Banded layout should be available");
    let mut banded_values = vec![f64::NAN; band_slots];
    banded
        .try_evaluate_banded_values_into(0.0, &state, &mut banded_values)
        .expect("wider-boundary Banded fixture should evaluate");
    let banded_matrix = Banded::from_vec(4, band_kl, band_ku, banded_values)
        .expect("wider-boundary Banded values should have the prepared layout");
    for row in 0..4 {
        for col in 0..4 {
            if (row as isize - col as isize) <= band_kl as isize
                && (col as isize - row as isize) <= band_ku as isize
            {
                assert!((dense_values[(row, col)] - banded_matrix[(row, col)]).abs() <= 1.0e-12);
            }
        }
    }

    reportln!(
        "[LSODE2 wider-boundary layout] dimension=4; sparse_nnz={}; band_kl_ku=(1,1); compact_slots={}; dense_sparse_banded_parity=true",
        expected_pattern.len(),
        band_slots,
    );
}

#[test]
fn lsode2_debug_native_callback_failure_injection_closes_scopes_and_recovers() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_native_callback_failure_injection_closes_scopes_and_recovers",
    );

    let equations = vec![Expr::parse_expression("a*y0"), Expr::parse_expression("y1")];
    let variables = vec!["y0".to_string(), "y1".to_string()];
    let state = DVector::from_vec(vec![3.0, 4.0]);
    let parameters = vec!["a".to_string()];
    let handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0])));
    let telemetry = IvpTelemetry::detailed();
    let mut sparse = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        Some(&parameters),
        Some(handle),
        NativeJacobianStorage::SparseTriplets,
        telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("failure-injection Sparse fixture should prepare");

    let state_error = sparse
        .try_evaluate_sparse_values_into(0.0, &DVector::zeros(1), &mut [0.0, 0.0])
        .expect_err("wrong state length must be a typed callback error");
    assert!(matches!(
        state_error,
        IvpBackendError::InvalidStateShape {
            expected: 2,
            actual: 1
        }
    ));

    let output_error = sparse
        .try_evaluate_sparse_values_into(0.0, &state, &mut [0.0])
        .expect_err("wrong Sparse output length must be a typed callback error");
    assert!(matches!(
        output_error,
        IvpBackendError::InvalidOutputShape { stage, expected: 2, actual: 1 }
            if stage == "sparse Jacobian values"
    ));

    let mut sparse_values = [f64::NAN; 2];
    sparse
        .try_evaluate_sparse_values_into(0.0, &state, &mut sparse_values)
        .expect("a recoverable callback error must not poison the next valid call");
    assert_eq!(sparse_values, [2.0, 1.0]);

    let banded_telemetry = IvpTelemetry::detailed();
    let mut banded = try_prepare_native_atomview_jacobian_runtime(
        &equations,
        &variables,
        "t",
        Some(&parameters),
        Some(Arc::new(RwLock::new(DVector::from_vec(vec![2.0])))),
        NativeJacobianStorage::Banded { bandwidth: None },
        banded_telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("failure-injection Banded fixture should prepare");
    let banded_len = banded
        .banded_layout()
        .expect("Banded fixture should expose its compact layout")
        .2;
    let banded_output_error = banded
        .try_evaluate_banded_values_into(0.0, &state, &mut [0.0])
        .expect_err("wrong Banded output length must be a typed callback error");
    assert!(matches!(
        banded_output_error,
        IvpBackendError::InvalidOutputShape {
            stage,
            expected: 2,
            actual: 1
        } if stage == "banded Jacobian values"
    ));
    let mut banded_values = vec![f64::NAN; banded_len];
    banded
        .try_evaluate_banded_values_into(0.0, &state, &mut banded_values)
        .expect("valid Banded callback should remain usable");
    assert_eq!(banded_values, vec![2.0, 1.0]);

    let poisoned_telemetry = IvpTelemetry::detailed();
    let poison_handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0])));
    let mut poisoned = try_prepare_native_atomview_jacobian_runtime(
        &[Expr::parse_expression("a*y0")],
        &["y0".to_string()],
        "t",
        Some(&["a".to_string()]),
        Some(poison_handle.clone()),
        NativeJacobianStorage::SparseTriplets,
        poisoned_telemetry.clone(),
        IvpLambdifyExecutionPolicy::Sequential,
    )
    .expect("poison-injection fixture should prepare");
    let poison_thread = std::thread::spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = poison_handle
                .write()
                .expect("poison fixture lock should open");
            panic!("intentional parameter lock poison for typed callback test");
        }));
    });
    poison_thread
        .join()
        .expect("catch_unwind should keep the poison fixture thread joinable");
    let poison_error = poisoned
        .try_evaluate_sparse_values_into(0.0, &DVector::from_vec(vec![3.0]), &mut [0.0])
        .expect_err("poisoned parameter state must cross the typed callback boundary");
    assert_eq!(poison_error, IvpBackendError::ParameterStatePoisoned);

    let sparse_snapshot = telemetry.snapshot();
    let banded_snapshot = banded_telemetry.snapshot();
    let poisoned_snapshot = poisoned_telemetry.snapshot();
    assert_eq!(sparse_snapshot.errors, 2);
    assert_eq!(banded_snapshot.errors, 1);
    assert_eq!(poisoned_snapshot.errors, 1);
    assert_eq!(
        sparse_snapshot
            .warm_stage(crate::symbolic::ivp_telemetry::IvpWarmStage::JacobianCallback)
            .calls,
        2
    );
    assert_eq!(
        poisoned_snapshot
            .warm_stage(crate::symbolic::ivp_telemetry::IvpWarmStage::JacobianCallback)
            .calls,
        1
    );
    reportln!(
        "[LSODE2 native callback failure injection] recoverable_state_and_output_errors=typed; sparse_valid_call_after_failure=true; banded_output_error=typed; banded_valid_call=true; poisoned_parameter_error=typed; sparse_errors={}; banded_errors={}; poisoned_errors={}; callback_scope_calls={}",
        sparse_snapshot.errors,
        banded_snapshot.errors,
        poisoned_snapshot.errors,
        sparse_snapshot
            .warm_stage(crate::symbolic::ivp_telemetry::IvpWarmStage::JacobianCallback)
            .calls,
    );
}

#[test]
fn lsode2_debug_exprlegacy_and_native_binding_scopes_close_on_poison() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::correctness_story_tests::lsode2_debug_exprlegacy_and_native_binding_scopes_close_on_poison",
    );

    for backend in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let telemetry = IvpTelemetry::detailed();
        let problem = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("a*y0")],
            vec!["y0".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_symbolic_assembly_backend(backend)
                .with_telemetry(telemetry.clone()),
        )
        .expect("binding scope fixture should prepare");

        let state = DVector::from_vec(vec![3.0]);
        let initial = problem
            .try_evaluate_residual(0.0, &state)
            .expect("valid residual callback should evaluate");
        assert_eq!(initial, DVector::from_vec(vec![6.0]));

        let handle = problem
            .parameter_values_handle()
            .expect("parameterized fixture should expose its shared handle");
        let poison_thread = std::thread::spawn(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = handle
                    .write()
                    .expect("binding scope poison lock should open");
                panic!("intentional parameter lock poison for binding scope test");
            }));
        });
        poison_thread
            .join()
            .expect("catch_unwind should keep the poison fixture joinable");

        let error = problem
            .try_evaluate_residual(0.0, &state)
            .expect_err("poisoned binding must remain a typed callback error");
        assert_eq!(error, IvpBackendError::ParameterStatePoisoned);

        if matches!(backend, IvpSymbolicAssemblyBackend::AtomView) {
            let wrong_state = problem
                .try_evaluate_residual(0.0, &DVector::zeros(0))
                .expect_err("native wrong state must remain typed");
            assert!(matches!(
                wrong_state,
                IvpBackendError::InvalidStateShape {
                    expected: 1,
                    actual: 0
                }
            ));
        }

        let snapshot = telemetry.snapshot();
        let expected_callback_calls = if matches!(backend, IvpSymbolicAssemblyBackend::AtomView) {
            3
        } else {
            2
        };
        let expected_errors = if matches!(backend, IvpSymbolicAssemblyBackend::AtomView) {
            2
        } else {
            1
        };
        assert_eq!(snapshot.errors, expected_errors);
        assert_eq!(
            snapshot.warm_stage(IvpWarmStage::ResidualCallback).calls,
            expected_callback_calls
        );
        assert_eq!(
            snapshot.warm_stage(IvpWarmStage::ArgumentBinding).calls,
            2,
            "successful and poisoned parameter reads each close binding scope"
        );
        reportln!(
            "[LSODE2 binding scope parity] backend={backend:?}; callback_scope_calls={}; binding_scope_calls={}; errors={}; poison=typed",
            snapshot.warm_stage(IvpWarmStage::ResidualCallback).calls,
            snapshot.warm_stage(IvpWarmStage::ArgumentBinding).calls,
            snapshot.errors,
        );
    }
}
