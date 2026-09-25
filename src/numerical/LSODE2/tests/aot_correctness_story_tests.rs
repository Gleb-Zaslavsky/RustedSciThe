//! Debug-only component parity gates for the three primary IVP routes.
//!
//! These tests intentionally stop below the solver/controller level. They
//! compare the prepared residual/Jacobian components first, so a later
//! trajectory mismatch can be attributed to the runtime route rather than to
//! an already divergent callback value.

use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use crate::symbolic::symbolic_ivp_generated::{
    SelectedSymbolicIvpBackendKind, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_problem,
};
use nalgebra::DVector;
use tempfile::tempdir;

fn fixture() -> (Vec<Expr>, Vec<String>, String, Vec<String>, DVector<f64>) {
    (
        vec![
            Expr::parse_expression("a*t + y + b*z"),
            Expr::parse_expression("c*y - z + b*t"),
        ],
        vec!["y".to_string(), "z".to_string()],
        "t".to_string(),
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        DVector::from_vec(vec![2.0, -0.5, 3.0]),
    )
}

fn options(
    backend: IvpSymbolicAssemblyBackend,
    parameter_names: &[String],
    parameter_values: &DVector<f64>,
) -> SymbolicIvpProblemOptions {
    SymbolicIvpProblemOptions::new()
        .with_symbolic_assembly_backend(backend)
        .with_equation_parameters(parameter_names.to_vec())
        .with_equation_parameter_values(parameter_values.clone())
}

fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn assert_component_parity(
    label: &str,
    legacy: &crate::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem,
    native_aot: &crate::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem,
    native_lambdify: &crate::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem,
    time: f64,
    state: &DVector<f64>,
) {
    let legacy_residual = legacy
        .try_evaluate_residual(time, state)
        .expect("ExprLegacy AOT residual should evaluate");
    let native_aot_residual = native_aot
        .try_evaluate_residual(time, state)
        .expect("AtomViewNative AOT residual should evaluate");
    let native_lambdify_residual = native_lambdify
        .try_evaluate_residual(time, state)
        .expect("AtomViewNative Lambdify residual should evaluate");
    assert_eq!(
        legacy_residual.len(),
        native_aot_residual.len(),
        "{label} residual shape"
    );
    assert_eq!(
        legacy_residual.len(),
        native_lambdify_residual.len(),
        "{label} residual shape"
    );
    assert!(
        max_abs_diff(legacy_residual.as_slice(), native_aot_residual.as_slice()) <= 1.0e-12,
        "{label} ExprLegacy-AOT vs AtomViewNative-AOT residual drift"
    );
    assert!(
        max_abs_diff(
            legacy_residual.as_slice(),
            native_lambdify_residual.as_slice()
        ) <= 1.0e-12,
        "{label} ExprLegacy-AOT vs AtomViewNative-Lambdify residual drift"
    );

    let legacy_jacobian = legacy
        .try_evaluate_jacobian(time, state)
        .expect("ExprLegacy AOT Jacobian should evaluate");
    let native_aot_jacobian = native_aot
        .try_evaluate_jacobian(time, state)
        .expect("AtomViewNative AOT Jacobian should evaluate");
    let native_lambdify_jacobian = native_lambdify
        .try_evaluate_jacobian(time, state)
        .expect("AtomViewNative Lambdify Jacobian should evaluate");
    assert_eq!(
        legacy_jacobian.shape(),
        native_aot_jacobian.shape(),
        "{label} Jacobian shape"
    );
    assert_eq!(
        legacy_jacobian.shape(),
        native_lambdify_jacobian.shape(),
        "{label} Jacobian shape"
    );
    assert!(
        max_abs_diff(legacy_jacobian.as_slice(), native_aot_jacobian.as_slice()) <= 1.0e-12,
        "{label} ExprLegacy-AOT vs AtomViewNative-AOT Jacobian drift"
    );
    assert!(
        max_abs_diff(
            legacy_jacobian.as_slice(),
            native_lambdify_jacobian.as_slice()
        ) <= 1.0e-12,
        "{label} ExprLegacy-AOT vs AtomViewNative-Lambdify Jacobian drift"
    );
}

#[test]
fn dense_aot_and_native_lambdify_component_parity_survives_rebind_and_reuse() {
    let (equations, variables, time_arg, parameter_names, initial_parameters) = fixture();
    let output_parent = tempdir().expect("AOT output directory should exist");
    let build_config = || {
        SymbolicIvpGeneratedBackendConfig::defaults()
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug,
            })
            .with_output_parent_dir(Some(output_parent.path().to_path_buf()))
    };

    let legacy_aot = prepare_generated_symbolic_ivp_problem(
        equations.clone(),
        variables.clone(),
        time_arg.clone(),
        options(
            IvpSymbolicAssemblyBackend::ExprLegacy,
            &parameter_names,
            &initial_parameters,
        ),
        build_config(),
    )
    .expect("ExprLegacy AOT preparation should succeed");
    let native_aot = prepare_generated_symbolic_ivp_problem(
        equations.clone(),
        variables.clone(),
        time_arg.clone(),
        options(
            IvpSymbolicAssemblyBackend::AtomView,
            &parameter_names,
            &initial_parameters,
        ),
        build_config(),
    )
    .expect("AtomViewNative AOT preparation should succeed");
    assert_eq!(
        legacy_aot.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );
    assert_eq!(
        native_aot.selected_backend,
        SelectedSymbolicIvpBackendKind::AotCompiled
    );

    let native_lambdify = prepare_symbolic_ivp_problem(
        equations,
        variables,
        time_arg,
        options(
            IvpSymbolicAssemblyBackend::AtomView,
            &parameter_names,
            &initial_parameters,
        ),
    )
    .expect("AtomViewNative Lambdify preparation should succeed");

    let state = DVector::from_vec(vec![0.75, -1.25]);
    assert_component_parity(
        "initial binding",
        &legacy_aot.problem,
        &native_aot.problem,
        &native_lambdify,
        0.25,
        &state,
    );

    let rebound_parameters = DVector::from_vec(vec![1.5, 0.25, -2.0]);
    legacy_aot
        .problem
        .set_parameter_values(rebound_parameters.clone())
        .expect("ExprLegacy AOT rebind should succeed");
    native_aot
        .problem
        .set_parameter_values(rebound_parameters.clone())
        .expect("AtomViewNative AOT rebind should succeed");
    native_lambdify
        .set_parameter_values(rebound_parameters)
        .expect("AtomViewNative Lambdify rebind should succeed");

    for repeat in 0..3 {
        let repeated_state = DVector::from_vec(vec![0.75 + repeat as f64 * 0.1, -1.25]);
        assert_component_parity(
            "rebound repeated warm callback",
            &legacy_aot.problem,
            &native_aot.problem,
            &native_lambdify,
            0.5 + repeat as f64 * 0.25,
            &repeated_state,
        );
    }

    let invalid_state = DVector::from_vec(vec![f64::NAN, -1.25]);
    assert!(
        legacy_aot
            .problem
            .try_evaluate_residual(0.25, &invalid_state)
            .is_err()
    );
    assert!(
        native_aot
            .problem
            .try_evaluate_residual(0.25, &invalid_state)
            .is_err()
    );
    let native_lambdify_nonfinite = native_lambdify
        .try_evaluate_residual(0.25, &invalid_state)
        .expect("native Lambdify keeps its historical NaN propagation contract");
    assert!(native_lambdify_nonfinite.iter().all(|value| value.is_nan()));
}
