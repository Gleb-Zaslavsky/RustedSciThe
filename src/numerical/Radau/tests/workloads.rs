//! Shared IVP workload correctness stories.
//!
//! These tests intentionally reuse the same equations as LSODE2 and BDF so
//! Radau frontend/layout regressions cannot hide behind a solver-specific toy
//! problem.

use super::super::new::benchmark::{BenchmarkAssembly, BenchmarkLayout, PreparedBenchmark};
use super::super::new::callbacks::PreparedSymbolicCallbacks;
use super::super::new::config::{RadauAssembly, RadauConfig, RadauExecution, RadauMatrixLayout};
use super::super::new::solver::try_solve_symbolic_dense;
use super::super::new::telemetry::{RadauFrontendStage, RadauTelemetry, RadauTelemetryMode};
use crate::numerical::LSODE2::{Lsode2ProblemConfig, Lsode2Solver};
use crate::numerical::ivp_workloads::{
    WorkloadKind, build_workload, parameter_continuation_target,
};
use crate::symbolic::ivp_telemetry::{IvpLambdifyExecutionPolicy, IvpTelemetry};
use crate::symbolic::symbolic_ivp::{
    build_expr_legacy_banded_jacobian, build_expr_legacy_sparse_jacobian,
};
use tabled::{Table, Tabled};

#[derive(Debug, Tabled)]
struct CrossSolverRow {
    workload: &'static str,
    reference_solver: &'static str,
    dimension: usize,
    interval: String,
    rtol: String,
    atol: String,
    scaled_final_state_error: String,
    gate: &'static str,
}

struct Case {
    callbacks: PreparedSymbolicCallbacks,
    config: RadauConfig,
    initial_state: Vec<f64>,
    parameters: Vec<f64>,
}

fn case(
    kind: WorkloadKind,
    dimension: usize,
    assembly: RadauAssembly,
    layout: RadauMatrixLayout,
) -> Case {
    let workload = build_workload(kind, dimension);
    let variables: Vec<&str> = workload.variables.iter().map(String::as_str).collect();
    let mut preparation = RadauTelemetry::new(RadauTelemetryMode::Counters);
    let parameter_names = workload
        .parameter_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let callbacks = match assembly {
        RadauAssembly::ExprLegacy => {
            preparation.measure_frontend(RadauFrontendStage::ExprLegacyPrepare, || {
                let symbolic_telemetry = IvpTelemetry::disabled();
                let jacobian = match layout {
                    RadauMatrixLayout::Banded { lower, upper } => {
                        build_expr_legacy_banded_jacobian(
                            &workload.equations,
                            &workload.variables,
                            (lower, upper),
                            &symbolic_telemetry,
                        )
                    }
                    RadauMatrixLayout::Dense | RadauMatrixLayout::Sparse => {
                        build_expr_legacy_sparse_jacobian(
                            &workload.equations,
                            &workload.variables,
                            &symbolic_telemetry,
                        )
                    }
                };
                PreparedSymbolicCallbacks::prepare_expr_sparse(
                    workload.equations,
                    jacobian,
                    &workload.time_variable,
                    &variables,
                    &parameter_names,
                    IvpLambdifyExecutionPolicy::Sequential,
                )
            })
        }
        RadauAssembly::AtomViewNative => PreparedSymbolicCallbacks::prepare_with_telemetry(
            assembly,
            workload.equations,
            None,
            &workload.time_variable,
            &variables,
            &parameter_names,
            &mut preparation,
        ),
    }
    .unwrap();
    assert_eq!(preparation.counters.frontend_preparations, 1);

    let horizon = match kind {
        WorkloadKind::CombustionLike => 0.001,
        WorkloadKind::Robertson => 0.002,
        WorkloadKind::ThreeBody => 0.001,
        WorkloadKind::DiffusionChain | WorkloadKind::StiffScalar => 0.01,
    };
    Case {
        callbacks,
        config: RadauConfig {
            execution: RadauExecution::Lambdify,
            assembly: Some(assembly),
            matrix_layout: layout,
            t_bound: horizon,
            first_step: Some(horizon * 0.1),
            max_step: horizon * 0.25,
            rtol: 1.0e-6,
            atol: 1.0e-9,
            max_steps: 2_000,
            max_newton_iterations: 8,
            max_retries: 24,
            ..RadauConfig::default()
        },
        initial_state: workload.initial_state.as_slice().to_vec(),
        parameters: workload.parameter_values.as_slice().to_vec(),
    }
}

fn solve(case: &mut Case) -> Vec<f64> {
    let mut session = case.callbacks.session();
    session.rebind_parameters(&case.parameters).unwrap();
    try_solve_symbolic_dense(&case.config, &mut session, &case.initial_state)
        .unwrap()
        .y
}

fn max_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn shared_stiff_workloads_match_expr_and_atom_frontends() {
    for kind in [
        WorkloadKind::StiffScalar,
        WorkloadKind::Robertson,
        WorkloadKind::CombustionLike,
        WorkloadKind::ThreeBody,
    ] {
        let mut expr = case(
            kind,
            kind_dimension(kind),
            RadauAssembly::ExprLegacy,
            RadauMatrixLayout::Dense,
        );
        let mut atom = case(
            kind,
            kind_dimension(kind),
            RadauAssembly::AtomViewNative,
            RadauMatrixLayout::Dense,
        );
        let expr_state = solve(&mut expr);
        let atom_state = solve(&mut atom);
        assert!(expr_state.iter().all(|value| value.is_finite()), "{kind:?}");
        assert!(atom_state.iter().all(|value| value.is_finite()), "{kind:?}");
        assert!(
            max_diff(&expr_state, &atom_state) < 1.0e-5,
            "{kind:?} frontend drift={:.3e}",
            max_diff(&expr_state, &atom_state)
        );
    }
}

#[test]
fn diffusion_layouts_match_dense_for_both_frontends() {
    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        let mut dense = case(
            WorkloadKind::DiffusionChain,
            16,
            assembly,
            RadauMatrixLayout::Dense,
        );
        let dense_state = solve(&mut dense);
        for layout in [
            RadauMatrixLayout::Sparse,
            RadauMatrixLayout::Banded { lower: 1, upper: 1 },
            RadauMatrixLayout::Banded { lower: 2, upper: 1 },
        ] {
            let mut structured = case(WorkloadKind::DiffusionChain, 16, assembly, layout);
            let structured_state = solve(&mut structured);
            assert!(
                max_diff(&dense_state, &structured_state) < 1.0e-5,
                "{assembly:?} {layout:?} drift={:.3e}",
                max_diff(&dense_state, &structured_state)
            );
        }
    }
}

#[test]
fn parameter_series_reuses_prepared_frontend_and_callback_capacity() {
    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        let workload = build_workload(WorkloadKind::CombustionLike, 3);
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
        let mut preparation = RadauTelemetry::new(RadauTelemetryMode::Counters);
        let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry(
            assembly,
            workload.equations,
            jacobian,
            &workload.time_variable,
            &variables,
            &workload
                .parameter_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            &mut preparation,
        )
        .unwrap();
        let mut session = callbacks.session_with_telemetry(RadauTelemetryMode::Counters);
        let initial_capacity = session.workspace_capacity();
        let parameter_ptr = session.parameters().as_ptr();
        let base = workload.parameter_values.as_slice().to_vec();

        for index in 0..16 {
            let target =
                parameter_continuation_target(&nalgebra::DVector::from_vec(base.clone()), index);
            session.rebind_parameters(target.as_slice()).unwrap();
            assert_eq!(session.parameters().as_ptr(), parameter_ptr);
            assert_eq!(session.workspace_capacity(), initial_capacity);
        }

        assert_eq!(preparation.counters.frontend_preparations, 1);
        assert_eq!(session.telemetry().counters.parameter_rebinds, 16);
        assert_eq!(session.telemetry().counters.allocations, 0);
        assert_eq!(session.telemetry().counters.workspace_resizes, 0);
    }
}

#[test]
fn linear_kernel_harness_runs_each_native_layout_without_conversion_errors() {
    for assembly in [
        BenchmarkAssembly::ExprLegacy,
        BenchmarkAssembly::AtomViewNative,
    ] {
        for layout in [
            BenchmarkLayout::Dense,
            BenchmarkLayout::Sparse,
            BenchmarkLayout::Banded { lower: 1, upper: 1 },
        ] {
            let prepared =
                PreparedBenchmark::prepare(WorkloadKind::DiffusionChain, 8, assembly, layout)
                    .unwrap();
            let checksum = prepared.linear_kernel(2).unwrap();
            assert!(checksum.is_finite(), "{assembly:?} {layout:?}");
        }
    }
}

#[test]
fn expr_default_and_atom_native_fixtures_share_structural_jacobian_pattern() {
    for layout in [
        RadauMatrixLayout::Dense,
        RadauMatrixLayout::Sparse,
        RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    ] {
        let expr = case(
            WorkloadKind::DiffusionChain,
            16,
            RadauAssembly::ExprLegacy,
            layout,
        );
        let atom = case(
            WorkloadKind::DiffusionChain,
            16,
            RadauAssembly::AtomViewNative,
            layout,
        );
        assert_eq!(
            expr.callbacks.jacobian_pattern(),
            atom.callbacks.jacobian_pattern(),
            "default frontend fixtures disagree on the Jacobian pattern for {layout:?}"
        );
        assert_eq!(
            expr.callbacks.jacobian_pattern().len(),
            46,
            "unexpected structural nnz for a 16-state diffusion chain"
        );
    }
}

#[test]
fn parameter_continuation_matches_fresh_reference_without_frontend_reprepare() {
    for (kind, dimension) in [
        (WorkloadKind::CombustionLike, 3),
        (WorkloadKind::DiffusionChain, 8),
    ] {
        for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
            let mut continued = case(kind, dimension, assembly, RadauMatrixLayout::Dense);
            let target = parameter_continuation_target(
                &nalgebra::DVector::from_vec(continued.parameters.clone()),
                4,
            );
            let mut session = continued.callbacks.session();
            session.rebind_parameters(target.as_slice()).unwrap();
            let continued_state =
                try_solve_symbolic_dense(&continued.config, &mut session, &continued.initial_state)
                    .unwrap()
                    .y;

            let mut fresh = case(kind, dimension, assembly, RadauMatrixLayout::Dense);
            fresh.parameters = target.as_slice().to_vec();
            let fresh_state = solve(&mut fresh);
            assert!(
                max_diff(&continued_state, &fresh_state) < 1.0e-8,
                "{kind:?} {assembly:?} continuation drift={:.3e}",
                max_diff(&continued_state, &fresh_state)
            );
        }
    }
}

#[test]
fn structured_parameter_continuation_matches_fresh_reference_for_both_frontends() {
    let layouts = [
        RadauMatrixLayout::Sparse,
        RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        RadauMatrixLayout::Banded { lower: 2, upper: 1 },
    ];

    for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
        for layout in layouts {
            let mut continued = case(WorkloadKind::DiffusionChain, 8, assembly, layout);
            let target = parameter_continuation_target(
                &nalgebra::DVector::from_vec(continued.parameters.clone()),
                3,
            );
            let mut session = continued.callbacks.session();
            session.rebind_parameters(target.as_slice()).unwrap();
            let continued_state =
                try_solve_symbolic_dense(&continued.config, &mut session, &continued.initial_state)
                    .unwrap()
                    .y;

            let mut fresh = case(WorkloadKind::DiffusionChain, 8, assembly, layout);
            fresh.parameters = target.as_slice().to_vec();
            let fresh_state = solve(&mut fresh);
            assert!(
                max_diff(&continued_state, &fresh_state) < 1.0e-8,
                "{assembly:?} {layout:?} continuation drift={:.3e}",
                max_diff(&continued_state, &fresh_state)
            );
        }
    }
}

#[test]
fn radau_robertson_matches_independent_lsode2_reference() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_CrossSolver",
        "numerical::Radau::tests::workloads::radau_robertson_matches_independent_lsode2_reference",
    );
    let workload = build_workload(WorkloadKind::Robertson, 3);
    let t_bound = 0.002;
    let max_step = 0.00025;

    let mut bdf_config = Lsode2ProblemConfig::new(
        workload.equations.clone(),
        workload.variables.clone(),
        workload.time_variable.clone(),
        0.0,
        workload.initial_state.clone(),
        t_bound,
        max_step,
        1.0e-7,
        1.0e-10,
    )
    .with_equation_parameters(workload.parameter_names.clone())
    .with_equation_parameter_values(workload.parameter_values.clone());
    bdf_config.first_step = Some(max_step * 0.2);
    let mut bdf = Lsode2Solver::new(bdf_config).expect("LSODE2 Robertson config");
    bdf.solve().expect("LSODE2 Robertson solve");
    let (_, bdf_states) = bdf.get_result();
    assert!(bdf_states.nrows() > 0, "LSODE2 returned no states");
    let bdf_final = bdf_states.row(bdf_states.nrows() - 1);

    let mut radau = case(
        WorkloadKind::Robertson,
        3,
        RadauAssembly::ExprLegacy,
        RadauMatrixLayout::Dense,
    );
    radau.config.t_bound = t_bound;
    radau.config.first_step = Some(max_step * 0.2);
    radau.config.max_step = max_step;
    radau.config.rtol = 1.0e-7;
    radau.config.atol = 1.0e-10;
    let radau_final = solve(&mut radau);

    let scaled_error = radau_final
        .iter()
        .zip(bdf_final.iter())
        .map(|(radau_value, bdf_value)| {
            (radau_value - bdf_value).abs() / (1.0e-10 + 1.0e-7 * bdf_value.abs())
        })
        .fold(0.0_f64, f64::max);
    assert!(
        scaled_error < 20.0,
        "Radau/LSODE2 Robertson scaled final-state difference={scaled_error:.3e}"
    );
    crate::Utils::test_reporting::capture_test_block(format!(
        "Shared Robertson equations and initial state; both solvers use t=[0, {t_bound}], max_step={max_step}, rtol=1e-7, atol=1e-10. Independent solver check complements, but does not replace, the SciPy fidelity fixtures.\n{}",
        Table::new([CrossSolverRow {
            workload: "Robertson",
            reference_solver: "LSODE2/BDF",
            dimension: 3,
            interval: format!("[0, {t_bound}]"),
            rtol: "1e-7".to_owned(),
            atol: "1e-10".to_owned(),
            scaled_final_state_error: format!("{scaled_error:.3e}"),
            gate: "< 20",
        }])
    ));
}

fn kind_dimension(kind: WorkloadKind) -> usize {
    match kind {
        WorkloadKind::StiffScalar => 1,
        WorkloadKind::Robertson => 3,
        WorkloadKind::CombustionLike => 3,
        WorkloadKind::ThreeBody => 12,
        WorkloadKind::DiffusionChain => 8,
    }
}
