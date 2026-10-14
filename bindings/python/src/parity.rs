//! Native adapter parity tests.
//!
//! These tests intentionally live in the binding crate rather than in Python:
//! both sides of each comparison are Rust values, so a failure identifies a
//! translation/lifecycle defect instead of mixing it with Python packaging or
//! numerical tolerance issues.

use nalgebra::DVector;
use pyo3::prelude::*;

use rustedscithe_core::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode, ODEsolver};
use rustedscithe_core::numerical::BE::{
    BE, BeSolverOptions, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use rustedscithe_core::numerical::Radau::api::{
    RadauConfig, RadauExecution, RadauFrontend, RadauJacobianSource, RadauMatrixLayout,
    RadauOutputPolicy, RadauProblem, RadauSolver as NativeRadauSolver, RadauTelemetryMode,
};
use rustedscithe_core::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use rustedscithe_core::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;

use crate::bdf::BdfSolver;
use crate::be::BackwardEulerSolver;
use crate::contract::SolverOptions;
use crate::expr::PyExpr;
use crate::radau::RadauSolver;

fn initialize_python() {
    Python::initialize();
}

fn parse(expression: &str) -> PyExpr {
    PyExpr::parse(expression).expect("parity fixture expression must parse")
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
    let scale = 1.0_f64.max(left.abs()).max(right.abs());
    assert!(
        (left - right).abs() <= tolerance * scale,
        "values differ: left={left:.16e}, right={right:.16e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn bdf_wrapper_matches_native_dense_expr_legacy_trajectory_and_counters() {
    initialize_python();

    let expression = parse("-y");
    let mut wrapper = Python::attach(|py| {
        let options = Py::new(
            py,
            SolverOptions::new(
                None,
                None,
                None,
                None,
                Some("counters"),
                None,
                None,
                None,
                None,
            )
            .expect("parity options"),
        )
        .expect("Python options object");
        BdfSolver::new(
            vec![expression.clone()],
            vec!["y".to_owned()],
            "t".to_owned(),
            vec![1.0],
            0.0,
            0.5,
            0.1,
            1e-5,
            1e-7,
            Some(options.borrow(py)),
            None,
            None,
            None,
            5,
            1_000_000,
        )
        .expect("wrapper construction")
    });
    wrapper.prepare().expect("wrapper preparation");
    let wrapper_result = wrapper.solve().expect("wrapper solve");

    let native_options = BdfSolverOptions::for_bdf(
        vec![expression.inner.clone()],
        vec!["y".to_owned()],
        "t".to_owned(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.5,
        0.1,
        1e-5,
        1e-7,
        None,
        false,
        None,
    )
    .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::ExprLegacy)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut native = ODEsolver::new_with_options(native_options);
    native.try_generate().expect("native preparation");
    native.try_solve().expect("native solve");

    let (native_times, native_states) = native.get_result_ref();
    let wrapper_times = wrapper_result.times();
    let wrapper_states = wrapper_result.states();
    assert_eq!(wrapper_times.len(), native_times.len());
    assert_eq!(wrapper_states.len(), native_states.nrows());
    for (index, time) in wrapper_times.iter().enumerate() {
        assert_close(*time, native_times[index], 1e-12);
        assert_close(wrapper_states[index][0], native_states[(index, 0)], 1e-11);
    }
    assert_close(wrapper_states.last().unwrap()[0], (-0.5_f64).exp(), 2e-3);
    assert_eq!(
        wrapper_result.telemetry_counters()["accepted_steps"],
        native.get_statistics().accepted_steps_total as u64
    );
}

#[test]
fn radau_wrapper_matches_native_dense_expr_legacy_solution_and_counters() {
    initialize_python();

    let expression = parse("-y");
    let jacobian = parse("-1");
    let mut wrapper = Python::attach(|py| {
        let options = Py::new(
            py,
            SolverOptions::new(
                None,
                None,
                None,
                None,
                Some("counters"),
                None,
                None,
                None,
                None,
            )
            .expect("parity options"),
        )
        .expect("Python options object");
        RadauSolver::new(
            vec![expression.clone()],
            vec!["y".to_owned()],
            "t".to_owned(),
            0.0,
            0.5,
            0.1,
            1e-6,
            1e-8,
            Some(options.borrow(py)),
            None,
            Some(vec![jacobian.clone()]),
            "final",
            None,
            1_000_000,
            6,
            12,
        )
        .expect("wrapper construction")
    });
    let wrapper_result = wrapper.solve(vec![1.0], None).expect("wrapper solve");

    let mut config = RadauConfig::default();
    config.t0 = 0.0;
    config.t_bound = 0.5;
    config.max_step = 0.1;
    config.rtol = 1e-6;
    config.atol = 1e-8;
    config.execution = RadauExecution::Lambdify;
    config.frontend = RadauFrontend::ExprLegacy;
    config.matrix_layout = RadauMatrixLayout::Dense;
    config.jacobian_source = RadauJacobianSource::Analytic;
    config.telemetry = RadauTelemetryMode::Counters;
    config.output = RadauOutputPolicy::FinalOnly;
    let problem = RadauProblem::new(vec![expression.inner.clone()], vec!["y".to_owned()], "t")
        .with_jacobian(vec![jacobian.inner.clone()]);
    let mut native = NativeRadauSolver::prepare(problem, config).expect("native preparation");
    let native_result = native
        .solve_with_parameters(&[1.0], &[])
        .expect("native solve");

    assert_close(wrapper_result.final_time(), native_result.t, 1e-12);
    assert_close(wrapper_result.final_state()[0], native_result.y[0], 1e-11);
    assert_eq!(
        wrapper_result.accepted_steps(),
        native_result.accepted_steps
    );
    assert_eq!(
        wrapper_result.rejected_steps(),
        native_result.rejected_steps
    );
    assert_eq!(
        wrapper_result.telemetry_counters()["residual_calls"],
        *native_result
            .telemetry()
            .counters
            .get("residual_calls")
            .unwrap()
    );
    assert_close(wrapper_result.final_state()[0], (-0.5_f64).exp(), 2e-5);
}

#[test]
fn backward_euler_wrapper_matches_native_dense_expr_legacy_trajectory() {
    initialize_python();

    let expression = parse("-y");
    let mut wrapper = Python::attach(|py| {
        let options = Py::new(
            py,
            SolverOptions::new(
                None,
                None,
                None,
                None,
                Some("counters"),
                None,
                None,
                None,
                None,
            )
            .expect("parity options"),
        )
        .expect("Python options object");
        BackwardEulerSolver::new(
            vec![expression.clone()],
            vec!["y".to_owned()],
            "t".to_owned(),
            vec![1.0],
            0.0,
            0.5,
            Some(0.1),
            1e-8,
            100,
            Some(options.borrow(py)),
            None,
            None,
        )
        .expect("wrapper construction")
    });
    let wrapper_result = wrapper.solve().expect("wrapper solve");

    let native_options = BeSolverOptions::new(
        vec![expression.inner.clone()],
        vec!["y".to_owned()],
        "t".to_owned(),
        1e-8,
        100,
        Some(0.1),
        0.0,
        0.5,
        DVector::from_vec(vec![1.0]),
    )
    .with_symbolic_assembly_backend(BeSymbolicAssemblyBackend::ExprLegacy)
    .with_telemetry_mode(BeTelemetryMode::Counters);
    let mut native = BE::try_new_with_options(native_options).expect("native construction");
    native.try_solve().expect("native solve");

    let (native_times, native_states) = native.trajectory();
    let wrapper_times = wrapper_result.times();
    let wrapper_states = wrapper_result.states();
    assert_eq!(wrapper_times.len(), native_times.len());
    assert_eq!(wrapper_states.len(), native_states.nrows());
    for (index, time) in wrapper_times.iter().enumerate() {
        assert_close(*time, native_times[index], 1e-12);
        assert_close(wrapper_states[index][0], native_states[(index, 0)], 1e-12);
    }
    assert_eq!(
        wrapper_result.telemetry_counters()["accepted_steps"],
        native.detailed_statistics().accepted_steps as u64
    );
}
