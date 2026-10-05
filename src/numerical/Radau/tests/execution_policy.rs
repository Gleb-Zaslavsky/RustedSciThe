//! Scheduling-policy correctness stories.
//!
//! These are deliberately small and non-ignored. They validate that changing
//! callback scheduling does not change values or layout contracts. Expensive
//! AOT timing and break-even evidence remains in ignored tests and benches.

use super::super::new::callbacks::PreparedSymbolicCallbacks;
use super::super::new::config::{RadauAssembly, RadauMatrixLayout};
use super::super::new::telemetry::{RadauTelemetry, RadauTelemetryMode};
use crate::numerical::ivp_workloads::{WorkloadKind, build_workload};
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;

fn prepare(
    assembly: RadauAssembly,
    policy: IvpLambdifyExecutionPolicy,
) -> (PreparedSymbolicCallbacks, Vec<f64>, Vec<f64>) {
    let workload = build_workload(WorkloadKind::DiffusionChain, 8);
    let variables: Vec<&str> = workload.variables.iter().map(String::as_str).collect();
    let jacobian = match assembly {
        RadauAssembly::ExprLegacy => Some(
            workload
                .equations
                .iter()
                .flat_map(|equation| {
                    variables
                        .iter()
                        .map(move |variable| equation.diff(variable))
                })
                .collect(),
        ),
        RadauAssembly::AtomViewNative => None,
    };
    let parameters: Vec<&str> = workload
        .parameter_names
        .iter()
        .map(String::as_str)
        .collect();
    let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Counters);
    let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry_and_policy(
        assembly,
        workload.equations,
        jacobian,
        &workload.time_variable,
        &variables,
        &parameters,
        &mut telemetry,
        policy,
    )
    .expect("policy callback preparation");
    (
        callbacks,
        workload.initial_state.as_slice().to_vec(),
        workload.parameter_values.as_slice().to_vec(),
    )
}

#[test]
fn lambdify_policy_matrix_preserves_residual_and_layout_values() {
    let policies = [
        IvpLambdifyExecutionPolicy::Sequential,
        IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        IvpLambdifyExecutionPolicy::Auto { min_work: 1 },
    ];
    let layouts = [
        RadauMatrixLayout::Dense,
        RadauMatrixLayout::Sparse,
        RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    ];

    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        for policy in policies {
            let (callbacks, state, parameters) = prepare(assembly, policy);
            let mut session = callbacks.session_with_telemetry(RadauTelemetryMode::Counters);
            session
                .rebind_parameters(&parameters)
                .expect("parameter rebind");
            let mut residual = vec![0.0; state.len()];
            session
                .evaluate_residual(0.0, &state, &mut residual)
                .expect("residual callback");
            let residual_reference = residual.clone();

            for layout in layouts {
                let output_len = match layout {
                    RadauMatrixLayout::Dense => state.len() * state.len(),
                    RadauMatrixLayout::Sparse => callbacks.jacobian_pattern().len(),
                    RadauMatrixLayout::Banded { lower, upper } => (lower + upper + 1) * state.len(),
                };
                let mut jacobian = vec![0.0; output_len];
                session
                    .evaluate_jacobian_layout(0.0, &state, layout, &mut jacobian)
                    .expect("Jacobian layout callback");
                assert!(jacobian.iter().all(|value| value.is_finite()));
            }

            let mut repeated = vec![0.0; state.len()];
            session
                .evaluate_residual(0.0, &state, &mut repeated)
                .expect("repeated residual callback");
            assert_eq!(repeated, residual_reference);
            let counters = session.telemetry().counters;
            assert!(counters.residual_evaluations >= 2);
            match policy {
                IvpLambdifyExecutionPolicy::Sequential => {
                    assert!(counters.sequential_dispatches > 0)
                }
                IvpLambdifyExecutionPolicy::Parallel { .. } => {
                    assert!(counters.parallel_dispatches > 0)
                }
                IvpLambdifyExecutionPolicy::Auto { .. } => {
                    assert!(counters.parallel_dispatches + counters.sequential_dispatches > 0)
                }
            }
        }
    }
}
