//! Debug correctness gates for evaluator policy selection.
//!
//! Performance and break-even conclusions belong to release stories. These
//! tests only prove that policy selection cannot change callback values.

use super::story_support::{chain_state, max_matrix_diff, max_vector_diff, prepare_chain};
use super::{IvpLambdifyExecutionPolicy, IvpTelemetry};

#[test]
fn sequential_parallel_and_auto_callbacks_are_value_identical() {
    let dimension = 128;
    let state = chain_state(dimension);
    let policies = [
        IvpLambdifyExecutionPolicy::Sequential,
        IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        IvpLambdifyExecutionPolicy::Auto { min_work: 1 },
    ];
    let reference = prepare_chain(
        dimension,
        super::story_support::native_frontend(),
        IvpLambdifyExecutionPolicy::Sequential,
        IvpTelemetry::counters(),
    );
    let reference_residual = reference
        .try_evaluate_residual(0.5, &state)
        .expect("reference residual should evaluate");
    let reference_jacobian = reference
        .try_evaluate_jacobian(0.5, &state)
        .expect("reference Jacobian should evaluate");

    for policy in policies {
        let telemetry = IvpTelemetry::counters();
        let prepared = prepare_chain(
            dimension,
            super::story_support::native_frontend(),
            policy,
            telemetry.clone(),
        );
        let residual = prepared
            .try_evaluate_residual(0.5, &state)
            .expect("policy residual should evaluate");
        let jacobian = prepared
            .try_evaluate_jacobian(0.5, &state)
            .expect("policy Jacobian should evaluate");
        assert!(max_vector_diff(&reference_residual, &residual) <= 1.0e-12);
        assert!(max_matrix_diff(&reference_jacobian, &jacobian) <= 1.0e-12);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.residual_evaluations, 1);
        assert_eq!(snapshot.jacobian_evaluations, 1);
        assert_eq!(snapshot.errors, 0);
        assert_eq!(
            snapshot.parallel_dispatches + snapshot.sequential_dispatches,
            2
        );
    }
}
