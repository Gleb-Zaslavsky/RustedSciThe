//! Stage-level performance stories. Expensive release benchmarks belong in
//! `benches/` and should reuse the shared Radau workload fixtures.

use std::hint::black_box;
use std::time::Instant;

use super::super::new::callbacks::PreparedSymbolicCallbacks;
use super::super::new::config::{RadauAssembly, RadauConfig, RadauMatrixLayout};
use super::super::new::error::RadauStage;
use super::super::new::step::try_radau5_symbolic_step;
use super::super::new::telemetry::{RadauTelemetry, RadauTelemetryMode};
use super::super::new::workspace::RadauWorkspace;
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::symbolic_engine::Expr;

#[test]
fn telemetry_off_keeps_counters_and_timings_empty() {
    let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Off);
    telemetry.count_stage(super::super::new::error::RadauStage::Residual);
    telemetry.add_timing_ms(super::super::new::error::RadauStage::Residual, 1.0);
    assert_eq!(telemetry.counters.residual_calls, 0);
    assert_eq!(telemetry.timings.callback_ms, 0.0);
}

#[test]
fn telemetry_marks_parallel_applicability_without_fabricating_dispatches() {
    let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Counters);

    telemetry.record_policy_dispatch(IvpLambdifyExecutionPolicy::Sequential, 8, 1);
    assert_eq!(telemetry.counters.parallel_dispatch_applicable, 0);
    assert_eq!(telemetry.counters.parallel_dispatches, 0);
    assert_eq!(telemetry.counters.sequential_dispatches, 1);

    // Applicability is sticky, while dispatch counters still describe what
    // actually happened for the observed callback.
    telemetry.record_policy_dispatch(IvpLambdifyExecutionPolicy::Auto { min_work: 64 }, 8, 2);
    assert_eq!(telemetry.counters.parallel_dispatch_applicable, 1);
    assert_eq!(telemetry.counters.parallel_dispatches, 0);
    assert_eq!(telemetry.counters.sequential_dispatches, 2);
}

#[test]
fn workspace_resize_reports_one_optional_scope() {
    let mut workspace = RadauWorkspace::default();
    workspace.telemetry.set_mode(RadauTelemetryMode::Timings);
    workspace
        .resize_for_layout(8, RadauMatrixLayout::Dense)
        .unwrap();

    assert_eq!(workspace.telemetry.counters.workspace_resizes, 1);
    assert!(workspace.telemetry.counters.allocations > 0);
    assert!(workspace.telemetry.timings.workspace_ms.is_finite());
    assert!(workspace.telemetry.timings.workspace_ms >= 0.0);
}

#[test]
fn lambdify_preparation_is_reported_only_when_timing_is_enabled() {
    let residual = vec![Expr::Add(
        Box::new(Expr::Var("y1".to_string())),
        Box::new(Expr::Var("p".to_string())),
    )];
    let jacobian = vec![Expr::Const(1.0)];
    let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Timings);
    let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry(
        RadauAssembly::ExprLegacy,
        residual,
        Some(jacobian),
        "t",
        &["y1"],
        &["p"],
        &mut telemetry,
    )
    .unwrap();

    assert!(matches!(
        callbacks,
        PreparedSymbolicCallbacks::ExprLegacy(_)
    ));
    assert!(telemetry.timings.expr_legacy_prepare_ms.is_finite());
    assert!(telemetry.timings.expr_legacy_prepare_ms >= 0.0);
    assert_eq!(telemetry.counters.residual_calls, 0);
    assert_eq!(telemetry.counters.jacobian_calls, 0);
    telemetry.count_stage(RadauStage::Residual);
    assert_eq!(telemetry.counters.residual_calls, 1);
}

#[test]
fn lambdify_warm_callback_story_reports_residual_and_jacobian_cost() {
    let dimension = 8;
    let parameter = Expr::Var("p".to_string());
    let variables: Vec<String> = (0..dimension).map(|index| format!("y{index}")).collect();
    let variable_refs: Vec<&str> = variables.iter().map(String::as_str).collect();
    let residual = variables
        .iter()
        .map(|variable| {
            Expr::Add(
                Box::new(Expr::Mul(
                    Box::new(parameter.clone()),
                    Box::new(Expr::Var(variable.clone())),
                )),
                Box::new(Expr::Const(1.0)),
            )
        })
        .collect::<Vec<_>>();
    let jacobian = (0..dimension * dimension)
        .map(|index| {
            if index / dimension == index % dimension {
                parameter.clone()
            } else {
                Expr::Const(0.0)
            }
        })
        .collect::<Vec<_>>();
    let mut preparation_telemetry = RadauTelemetry::new(RadauTelemetryMode::Timings);
    let callbacks =
        super::super::new::callbacks::PreparedSymbolicCallbacks::prepare_with_telemetry(
            super::super::new::config::RadauAssembly::ExprLegacy,
            residual,
            Some(jacobian),
            "t",
            &variable_refs,
            &["p"],
            &mut preparation_telemetry,
        )
        .unwrap();
    let mut session = callbacks.session_with_telemetry(RadauTelemetryMode::Timings);
    let state = vec![2.0; dimension];
    let parameters = [3.0];
    session.rebind_parameters(&parameters).unwrap();
    let capacity = session.workspace_capacity();
    let mut residual_output = vec![0.0; dimension];
    let mut jacobian_output = vec![0.0; dimension * dimension];
    let repetitions = 2_000;

    let residual_start = Instant::now();
    for _ in 0..repetitions {
        session
            .evaluate_residual(black_box(0.5), black_box(&state), &mut residual_output)
            .unwrap();
        black_box(&residual_output);
    }
    let residual_elapsed = residual_start.elapsed();

    let jacobian_start = Instant::now();
    for _ in 0..repetitions {
        session
            .evaluate_jacobian(black_box(0.5), black_box(&state), &mut jacobian_output)
            .unwrap();
        black_box(&jacobian_output);
    }
    let jacobian_elapsed = jacobian_start.elapsed();

    assert_eq!(residual_output, vec![7.0; dimension]);
    for (index, value) in jacobian_output.iter().enumerate() {
        assert_eq!(
            *value,
            if index / dimension == index % dimension {
                3.0
            } else {
                0.0
            }
        );
    }
    assert_eq!(session.workspace_capacity(), capacity);
    assert_eq!(preparation_telemetry.counters.frontend_preparations, 1);
    assert!(
        preparation_telemetry
            .timings
            .expr_legacy_prepare_ms
            .is_finite()
    );
    assert!(session.telemetry().counters.argument_bindings >= repetitions as u64 * 2);
    assert_eq!(
        session.telemetry().counters.residual_evaluations,
        repetitions
    );
    assert_eq!(
        session.telemetry().counters.jacobian_evaluations,
        repetitions
    );
    assert_eq!(session.telemetry().counters.parameter_rebinds, 1);
    assert!(
        session.telemetry().counters.output_writes >= (repetitions as u64 * dimension as u64 * 2)
    );
    assert!(session.telemetry().timings.binding_ms.is_finite());
    assert!(
        session
            .telemetry()
            .timings
            .residual_evaluation_ms
            .is_finite()
    );
    assert!(
        session
            .telemetry()
            .timings
            .jacobian_evaluation_ms
            .is_finite()
    );
    assert!(session.telemetry().timings.parameter_rebind_ms.is_finite());
    println!(
        "[Radau Lambdify warm callbacks] dimension={dimension}; repetitions={repetitions}; residual_ns_per_call={:.3}; jacobian_ns_per_call={:.3}; argument_capacity={capacity}",
        residual_elapsed.as_secs_f64() * 1.0e9 / repetitions as f64,
        jacobian_elapsed.as_secs_f64() * 1.0e9 / repetitions as f64,
    );
}

#[test]
fn symbolic_structured_projection_reports_output_assembly_for_both_frontends() {
    let y1 = Expr::Var("y1".to_string());
    let y2 = Expr::Var("y2".to_string());
    let parameter = Expr::Var("p".to_string());
    let residual = vec![
        Expr::Add(Box::new(y1.clone()), Box::new(parameter.clone())),
        Expr::Add(
            Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(y1))),
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
        for layout in [
            RadauMatrixLayout::Sparse,
            RadauMatrixLayout::Banded { lower: 1, upper: 0 },
        ] {
            let jacobian = match assembly {
                RadauAssembly::ExprLegacy => Some(explicit_jacobian.clone()),
                RadauAssembly::AtomViewNative => None,
            };
            let mut preparation_telemetry = RadauTelemetry::new(RadauTelemetryMode::Counters);
            let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry(
                assembly,
                residual.clone(),
                jacobian,
                "t",
                &["y1", "y2"],
                &["p"],
                &mut preparation_telemetry,
            )
            .unwrap();
            let mut session = callbacks.session_with_telemetry(RadauTelemetryMode::Timings);
            session.rebind_parameters(&[3.0]).unwrap();
            let output_len = match layout {
                RadauMatrixLayout::Sparse => callbacks.jacobian_pattern().len(),
                RadauMatrixLayout::Banded { lower, upper } => (lower + upper + 1) * 2,
                RadauMatrixLayout::Dense => unreachable!(),
            };
            let mut output = vec![0.0; output_len];
            session
                .evaluate_jacobian_layout(0.0, &[2.0, 4.0], layout, &mut output)
                .unwrap();

            assert_eq!(preparation_telemetry.counters.frontend_preparations, 1);
            assert_eq!(session.telemetry().counters.jacobian_output_assemblies, 1);
            assert_eq!(
                session.telemetry().counters.output_writes,
                output_len as u64
            );
            assert!(
                session
                    .telemetry()
                    .timings
                    .jacobian_output_assembly_ms
                    .is_finite()
            );
        }
    }
}

#[test]
fn symbolic_step_reports_callback_linear_and_workspace_stages() {
    let parameter = Expr::Var("p".to_string());
    let state = Expr::Var("y1".to_string());
    let residual = vec![Expr::Mul(
        Box::new(Expr::Const(-1.0)),
        Box::new(Expr::Mul(Box::new(parameter.clone()), Box::new(state))),
    )];
    let jacobian = vec![Expr::Mul(Box::new(Expr::Const(-1.0)), Box::new(parameter))];
    let config = RadauConfig {
        rtol: 1.0e-9,
        atol: 1.0e-11,
        max_newton_iterations: 8,
        telemetry: RadauTelemetryMode::Timings,
        ..RadauConfig::default()
    };

    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        let selected_jacobian = match assembly {
            RadauAssembly::ExprLegacy => Some(jacobian.clone()),
            RadauAssembly::AtomViewNative => None,
        };
        let mut preparation_telemetry = RadauTelemetry::new(RadauTelemetryMode::Timings);
        let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry(
            assembly,
            residual.clone(),
            selected_jacobian,
            "t",
            &["y1"],
            &["p"],
            &mut preparation_telemetry,
        )
        .unwrap();
        let mut callback_session = callbacks.session_with_telemetry(RadauTelemetryMode::Timings);
        callback_session.rebind_parameters(&[1.0]).unwrap();
        let mut workspace = RadauWorkspace::default();
        let mut output = [0.0];
        try_radau5_symbolic_step(
            &config,
            &mut callback_session,
            0.0,
            0.1,
            &[1.0],
            &mut output,
            &mut workspace,
        )
        .unwrap();

        assert!(preparation_telemetry.counters.frontend_preparations > 0);
        assert!(callback_session.telemetry().counters.residual_evaluations > 0);
        assert!(callback_session.telemetry().counters.jacobian_evaluations > 0);
        assert!(
            callback_session
                .telemetry()
                .timings
                .residual_evaluation_ms
                .is_finite()
        );
        assert!(
            callback_session
                .telemetry()
                .timings
                .jacobian_evaluation_ms
                .is_finite()
        );
        assert!(workspace.telemetry.counters.workspace_resizes > 0);
        assert!(workspace.telemetry.counters.jacobian_assemblies > 0);
        assert!(workspace.telemetry.counters.factorizations > 0);
        assert!(workspace.telemetry.counters.real_solves > 0);
        assert!(workspace.telemetry.counters.complex_solves > 0);
        assert!(workspace.telemetry.counters.newton_iterations > 0);
        assert_eq!(workspace.telemetry.counters.error_estimates, 1);
        assert_eq!(workspace.telemetry.counters.output_writes, 1);
        assert!(callback_session.telemetry().timings.callback_ms.is_finite());
        assert!(workspace.telemetry.timings.linear_ms.is_finite());
        assert!(workspace.telemetry.timings.newton_ms.is_finite());
        assert!(workspace.telemetry.timings.output_ms.is_finite());
        assert!(workspace.telemetry.timings.step_control_ms.is_finite());
        assert!(workspace.telemetry.timings.workspace_ms.is_finite());
        assert!(workspace.telemetry.timings.factorization_ms.is_finite());
    }
}

#[test]
fn atom_native_warm_callback_story_reports_native_preparation_and_callback_cost() {
    let dimension = 4;
    let parameter = Expr::Var("p".to_string());
    let variables: Vec<String> = (0..dimension).map(|index| format!("y{index}")).collect();
    let variable_refs: Vec<&str> = variables.iter().map(String::as_str).collect();
    let residual = variables
        .iter()
        .map(|variable| {
            Expr::Add(
                Box::new(Expr::Mul(
                    Box::new(parameter.clone()),
                    Box::new(Expr::Var(variable.clone())),
                )),
                Box::new(Expr::Const(1.0)),
            )
        })
        .collect::<Vec<_>>();
    let mut preparation_telemetry = RadauTelemetry::new(RadauTelemetryMode::Timings);
    let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry(
        RadauAssembly::AtomViewNative,
        residual,
        None,
        "t",
        &variable_refs,
        &["p"],
        &mut preparation_telemetry,
    )
    .unwrap();
    let mut session = callbacks.session_with_telemetry(RadauTelemetryMode::Timings);
    session.rebind_parameters(&[3.0]).unwrap();

    let state = vec![2.0; dimension];
    let mut residual_output = vec![0.0; dimension];
    let mut jacobian_output = vec![0.0; dimension * dimension];
    for _ in 0..16 {
        session
            .evaluate_residual(0.5, &state, &mut residual_output)
            .unwrap();
        session
            .evaluate_jacobian(0.5, &state, &mut jacobian_output)
            .unwrap();
    }

    assert_eq!(residual_output, vec![7.0; dimension]);
    for (index, value) in jacobian_output.iter().enumerate() {
        assert_eq!(
            *value,
            if index / dimension == index % dimension {
                3.0
            } else {
                0.0
            }
        );
    }
    assert_eq!(preparation_telemetry.counters.frontend_preparations, 1);
    assert!(preparation_telemetry.timings.atom_conversion_ms.is_finite());
    assert!(
        preparation_telemetry
            .timings
            .atom_jacobian_prepare_ms
            .is_finite()
    );
    assert!(preparation_telemetry.timings.atom_dependency_ms.is_finite());
    assert!(session.telemetry().counters.argument_bindings >= 32);
    assert_eq!(session.telemetry().counters.residual_evaluations, 16);
    assert_eq!(session.telemetry().counters.jacobian_evaluations, 16);
    assert_eq!(session.telemetry().counters.parameter_rebinds, 1);
    assert!(session.telemetry().counters.output_writes >= (16 * dimension * 2) as u64);
    assert!(
        session
            .telemetry()
            .timings
            .residual_evaluation_ms
            .is_finite()
    );
    assert!(
        session
            .telemetry()
            .timings
            .jacobian_evaluation_ms
            .is_finite()
    );
    assert!(session.telemetry().timings.workspace_ms.is_finite());
}
