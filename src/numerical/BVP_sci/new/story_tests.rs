//! Compact Lambdify-only story gates for the new BVP architecture.
//!
//! These tests deliberately use small mathematical fixtures so they remain
//! useful in debug mode. The release dashboard in `benches/` expands the same
//! matrix without putting Criterion/compiler chatter into the report table.

use super::*;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{DampedSolverOptions, NRBVP, SolverParams};
use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use tabled::{Table, Tabled};

#[derive(Debug, Tabled)]
struct MatrixRow {
    frontend: String,
    layout: String,
    prepare_ms: String,
    solve_ms: String,
    continuation_ms: String,
    continuation_diff: String,
    symbolic_jacobian_ms: String,
    callback_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    factorization_ms: String,
    residual_norm: String,
    max_fidelity_diff: String,
    status: String,
}

#[derive(Debug, Tabled)]
struct WorkloadRow {
    workload: String,
    frontend: String,
    layout: String,
    solve_ms: String,
    residual_norm: String,
    max_fidelity_diff: String,
    jacobian_calls: u64,
    factorizations: u64,
    status: String,
}

#[derive(Debug, Tabled)]
struct CrossSolverRow {
    workload: String,
    control_points: usize,
    bvp_sci_nodes: usize,
    bvp_damp_nodes: usize,
    bvp_sci_residual: String,
    max_interpolated_diff: String,
    status: String,
}

fn layout_name(layout: BvpSciMatrixLayout) -> String {
    match layout {
        BvpSciMatrixLayout::Dense => "Dense".into(),
        BvpSciMatrixLayout::Sparse => "Sparse".into(),
        BvpSciMatrixLayout::Banded { .. } => "Banded".into(),
    }
}

fn timing(value: Option<f64>) -> String {
    value.map(measured_ms).unwrap_or_else(|| "n/a".into())
}

fn sparse_factor_timing_ms(
    snapshot: &BvpSciTelemetrySnapshot,
    policy: BvpSciSparseFactorization,
) -> Option<f64> {
    #[allow(deprecated)]
    let is_superlu = matches!(
        policy,
        BvpSciSparseFactorization::SuperLu | BvpSciSparseFactorization::SuperLuExperimental
    );
    if is_superlu {
        snapshot.factorization_ms
    } else {
        snapshot.sparse_numeric_factorization_ms
    }
}

fn sparse_policy_effective_backend(
    snapshot: &BvpSciTelemetrySnapshot,
    policy: BvpSciSparseFactorization,
) -> &'static str {
    if snapshot.superlu_fallbacks > 0 {
        "StandardFaer-fallback"
    } else {
        #[allow(deprecated)]
        match policy {
            BvpSciSparseFactorization::SuperLu | BvpSciSparseFactorization::SuperLuExperimental => {
                "SuperLU"
            }
            _ => "faer",
        }
    }
}

fn superlu_stage_summary(snapshot: &BvpSciTelemetrySnapshot) -> String {
    let metrics = &snapshot.superlu;
    if metrics.factor_ffi_calls == 0 && metrics.solve_ffi_calls == 0 {
        return "-".into();
    }
    format!(
        "calls abi/factor-ffi/native-factor/solve-ffi/native-solve={}/{}/{}/{}/{}; ms abi/factor-ffi/native-factor/solve-ffi/native-solve={}/{}/{}/{}/{}",
        metrics.abi_conversions,
        metrics.factor_ffi_calls,
        metrics.native_factorization_calls,
        metrics.solve_ffi_calls,
        metrics.native_solve_calls,
        timing(metrics.abi_conversion_ms),
        timing(metrics.factor_ffi_ms),
        timing(metrics.native_factorization_ms),
        timing(metrics.solve_ffi_ms),
        timing(metrics.native_solve_ms),
    )
}

fn superlu_factor_fill_summary(snapshot: &BvpSciTelemetrySnapshot) -> String {
    let metrics = &snapshot.superlu;
    if metrics.factor_ffi_calls == 0 {
        return "-".into();
    }
    format!(
        "L storage/CSC={}/{}; U storage/CSC={}/{}",
        metrics.factor_l_storage_entries,
        metrics.factor_l_csc_nnz,
        metrics.factor_u_storage_entries,
        metrics.factor_u_csc_nnz,
    )
}

fn measured_ms(value: f64) -> String {
    if value.abs() < 0.0005 {
        "<0.001".into()
    } else {
        format!("{value:.3}")
    }
}

fn aot_stage_ms(
    snapshot: &crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot,
    stage: crate::symbolic::ivp_telemetry::IvpColdStage,
) -> String {
    let timing = snapshot.cold_stage(stage);
    if timing.calls == 0 {
        "n/a".into()
    } else {
        measured_ms(timing.elapsed.as_secs_f64() * 1e3)
    }
}

/// Six-state nonlinear combustion-shaped system shared by the two solvers.
///
/// The fixture intentionally has no closed-form oracle. It is used only for a
/// cross-solver gate: both implementations solve the same equations and
/// boundary conditions, then their trajectories are compared on a third-party
/// control mesh. Keeping the expressions in one helper prevents the gate from
/// accidentally comparing two different physical problems.
fn combustion_like_equations() -> Vec<Expr> {
    vec![
        Expr::parse_expression("q"),
        Expr::parse_expression("-0.5*q + 0.2*exp(Teta)*C0"),
        Expr::parse_expression("J0"),
        Expr::parse_expression("-0.3*J0 - 0.5*C0 + 0.1*Teta"),
        Expr::parse_expression("J1"),
        Expr::parse_expression("-0.2*J1 - 0.3*C1 + 0.05*C0^2"),
    ]
}

#[test]
fn combustion_bvp_componentwise_frontend_parity_pins_compiler_input() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "combustion_bvp_componentwise_frontend_parity",
    );
    let equations = combustion_like_equations();
    let state_names = ["Teta", "q", "C0", "J0", "C1", "J1"].map(str::to_owned);
    let parameter_names: [String; 0] = [];
    let x = 0.37_f64;
    let state: [f64; 6] = [0.15, -0.05, 0.8, 0.2, 0.1, -0.1];
    let equations_text = equations
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let state_bits = state
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    let canonical_input = format!(
        "equations={equations_text}\nstates={}\nparameters=\nindependent=x\nx={:016x}\nstate={state_bits}\n",
        state_names.join(","),
        x.to_bits(),
    );
    let fingerprint = canonical_input
        .bytes()
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    assert_eq!(
        fingerprint, 0x163ad2c0db7000c1,
        "review the BVP_sci fixture/compiler-input change before updating its fingerprint"
    );

    let mut plans = [
        BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &equations,
            &state_names,
            &parameter_names,
            "x",
            BvpSciTelemetry::disabled(),
        )
        .expect("ExprLegacy combustion fixture should prepare"),
        BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::AtomViewNative,
            &equations,
            &state_names,
            &parameter_names,
            "x",
            BvpSciTelemetry::disabled(),
        )
        .expect("AtomViewNative combustion fixture should prepare"),
    ];
    assert_eq!(
        plans[0].jacobian_pattern(),
        plans[1].jacobian_pattern(),
        "ExprLegacy and AtomView must retain the same fixed sparse coordinate order"
    );
    assert_eq!(plans[0].jacobian_pattern().len(), 12);

    let symbols = std::iter::once("x")
        .chain(state_names.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let rhs = equations
        .iter()
        .map(|equation| Expr::lambdify_borrowed_thread_safe(equation, &symbols))
        .collect::<Vec<_>>();
    let mut residuals: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; equations.len()]);
    let mut dense_jacobians: [Vec<f64>; 2] =
        std::array::from_fn(|_| vec![0.0; equations.len() * state_names.len()]);
    let mut sparse_values: [Vec<f64>; 2] =
        std::array::from_fn(|_| vec![0.0; plans[0].jacobian_pattern().len()]);
    for (route, plan) in plans.iter_mut().enumerate() {
        let mut arguments = vec![0.0; 1 + state.len()];
        plan.evaluate_rhs(x, &state, &[], &mut arguments, &mut residuals[route])
            .expect("combustion RHS should evaluate");
        plan.evaluate_jacobian_dense(x, &state, &[], &mut arguments, &mut dense_jacobians[route])
            .expect("combustion state Jacobian should evaluate");
        plan.evaluate_jacobian_values(x, &state, &[], &mut arguments, &mut sparse_values[route])
            .expect("combustion sparse Jacobian values should evaluate");
    }

    let mut arguments = vec![x];
    arguments.extend(state);
    for row in 0..equations.len() {
        let reference = rhs[row](&arguments);
        for route in 0..plans.len() {
            assert!(
                (residuals[route][row] - reference).abs() <= 2.0e-12,
                "route {route}, residual row {row}: got {:e}, expected {reference:e}",
                residuals[route][row]
            );
        }
    }
    for (entry_index, &(row, col)) in plans[0].jacobian_pattern().iter().enumerate() {
        for route in 0..plans.len() {
            let dense_value = dense_jacobians[route][row * state_names.len() + col];
            assert!(
                (sparse_values[route][entry_index] - dense_value).abs() <= 2.0e-12,
                "route {route}, sparse entry {entry_index} ({row}, {col}) does not match dense output"
            );
        }
        assert!(
            (dense_jacobians[0][row * state_names.len() + col]
                - dense_jacobians[1][row * state_names.len() + col])
                .abs()
                <= 2.0e-12,
            "ExprLegacy/AtomView Jacobian mismatch at ({row}, {col})"
        );
    }

    for col in 0..state.len() {
        let step = 1.0e-6 * state[col].abs().max(1.0);
        let mut plus_args = arguments.clone();
        let mut minus_args = arguments.clone();
        plus_args[col + 1] += step;
        minus_args[col + 1] -= step;
        for row in 0..equations.len() {
            let finite_difference = (rhs[row](&plus_args) - rhs[row](&minus_args)) / (2.0 * step);
            let atom_value = dense_jacobians[1][row * state_names.len() + col];
            assert!(
                (finite_difference - atom_value).abs() <= 2.0e-6 * atom_value.abs().max(1.0),
                "AtomView Jacobian ({row}, {col})={atom_value:e}, finite difference={finite_difference:e}"
            );
        }
    }
}

fn combustion_like_boundary_values() -> HashMap<String, Vec<(usize, f64)>> {
    HashMap::from([
        ("Teta".into(), vec![(0, 0.0)]),
        ("q".into(), vec![(1, 0.0)]),
        ("C0".into(), vec![(0, 1.0)]),
        ("J0".into(), vec![(1, 0.0)]),
        ("C1".into(), vec![(0, 0.0)]),
        ("J1".into(), vec![(1, 0.0)]),
    ])
}

/// Evaluate a node-major trajectory with linear interpolation.
///
/// `BVP_sci` exposes node-major `Vec<f64>` values, while `BVP_Damp` exposes a
/// row-major `DMatrix`. This adapter deliberately uses the same piecewise
/// linear interpolation for both representations, so the comparison does not
/// depend on either solver's optional dense-output implementation.
fn interpolate_node_major(mesh: &[f64], values: &[f64], dimension: usize, x: f64) -> Vec<f64> {
    let right = mesh.partition_point(|node| *node <= x).saturating_sub(1);
    let left = right.min(mesh.len().saturating_sub(2));
    let h = mesh[left + 1] - mesh[left];
    let weight = if h > 0.0 { (x - mesh[left]) / h } else { 0.0 };
    (0..dimension)
        .map(|component| {
            let y0 = values[left * dimension + component];
            let y1 = values[(left + 1) * dimension + component];
            y0 + weight * (y1 - y0)
        })
        .collect()
}

fn interpolate_damped(mesh: &[f64], values: &DMatrix<f64>, x: f64) -> Vec<f64> {
    let right = mesh.partition_point(|node| *node <= x).saturating_sub(1);
    let left = right.min(mesh.len().saturating_sub(2));
    let h = mesh[left + 1] - mesh[left];
    let weight = if h > 0.0 { (x - mesh[left]) / h } else { 0.0 };
    (0..values.ncols())
        .map(|component| {
            values[(left, component)]
                + weight * (values[(left + 1, component)] - values[(left, component)])
        })
        .collect()
}

fn normalized_cross_solver_diff(
    sci_mesh: &[f64],
    sci_values: &[f64],
    damp_mesh: &[f64],
    damp_values: &DMatrix<f64>,
    dimension: usize,
    control_points: usize,
) -> f64 {
    let mut maximum = 0.0_f64;
    for index in 0..control_points {
        let sample = index as f64 / (control_points - 1) as f64;
        let sci = interpolate_node_major(sci_mesh, sci_values, dimension, sample);
        let damp = interpolate_damped(damp_mesh, damp_values, sample);
        for (sci_value, damp_value) in sci.iter().zip(damp.iter()) {
            let scale = 1.0 + sci_value.abs().max(damp_value.abs());
            maximum = maximum.max((sci_value - damp_value).abs() / scale);
        }
    }
    maximum
}

fn build_bvp_damp_combustion(nodes: usize) -> NRBVP {
    let names = ["Teta", "q", "C0", "J0", "C1", "J1"];
    let initial_guess = DMatrix::from_fn(
        names.len(),
        nodes,
        |row, _| {
            if row == 2 { 1.0 } else { 0.0 }
        },
    );
    let rel_tolerance = names
        .iter()
        .map(|name| ((*name).to_string(), 1e-5))
        .collect();
    let bounds = HashMap::from([
        ("Teta".into(), (-10.0, 10.0)),
        ("q".into(), (-10.0, 10.0)),
        ("C0".into(), (-2.0, 2.0)),
        ("J0".into(), (-10.0, 10.0)),
        ("C1".into(), (-2.0, 2.0)),
        ("J1".into(), (-10.0, 10.0)),
    ]);
    let generated = GeneratedBackendConfig::new()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
        .with_matrix_backend_override(MatrixBackend::Dense);
    let options = DampedSolverOptions::dense_damped()
        .with_generated_backend_config(generated)
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(4),
            max_damp_iter: Some(8),
            damp_factor: Some(0.5),
            adaptive: None,
        }))
        .with_abs_tolerance(1e-6)
        .with_rel_tolerance(rel_tolerance)
        .with_max_iterations(100)
        .with_bounds(bounds)
        .with_loglevel(Some("none".into()));
    let mut solver = NRBVP::new_with_options(
        combustion_like_equations(),
        initial_guess,
        names.iter().map(|name| (*name).to_string()).collect(),
        "x".into(),
        combustion_like_boundary_values(),
        0.0,
        1.0,
        nodes,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn bratu_equations() -> Vec<Expr> {
    vec![
        Expr::parse_expression("z"),
        Expr::parse_expression("-2*exp(y)"),
    ]
}

fn bratu_boundary_values() -> HashMap<String, Vec<(usize, f64)>> {
    HashMap::from([("y".into(), vec![(0, 0.0), (1, 0.0)])])
}

fn build_bvp_damp_bratu(nodes: usize) -> NRBVP {
    let names = ["y", "z"];
    let initial_guess = DMatrix::from_element(names.len(), nodes, 0.0);
    let rel_tolerance = HashMap::from([("y".into(), 1e-5), ("z".into(), 1e-5)]);
    let bounds = HashMap::from([("y".into(), (-20.0, 20.0)), ("z".into(), (-200.0, 200.0))]);
    let generated = GeneratedBackendConfig::new()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy)
        .with_matrix_backend_override(MatrixBackend::Dense);
    let options = DampedSolverOptions::dense_damped()
        .with_generated_backend_config(generated)
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(4),
            max_damp_iter: Some(8),
            damp_factor: Some(0.5),
            adaptive: None,
        }))
        .with_abs_tolerance(1e-6)
        .with_rel_tolerance(rel_tolerance)
        .with_max_iterations(100)
        .with_bounds(bounds)
        .with_loglevel(Some("none".into()));
    let mut solver = NRBVP::new_with_options(
        bratu_equations(),
        initial_guess,
        names.iter().map(|name| (*name).to_string()).collect(),
        "x".into(),
        bratu_boundary_values(),
        0.0,
        1.0,
        nodes,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn build_solver(
    assembly: BvpSciAssembly,
    layout: BvpSciMatrixLayout,
    parameter: f64,
    execution_policy: BvpSciExecutionPolicy,
) -> (BvpSciSolver, Arc<AtomicU64>) {
    build_solver_with_sparse_factorization(
        assembly,
        layout,
        parameter,
        execution_policy,
        BvpSciSparseFactorization::StandardFaer,
    )
}

fn build_solver_with_sparse_factorization(
    assembly: BvpSciAssembly,
    layout: BvpSciMatrixLayout,
    parameter: f64,
    execution_policy: BvpSciExecutionPolicy,
    sparse_factorization: BvpSciSparseFactorization,
) -> (BvpSciSolver, Arc<AtomicU64>) {
    // Story rows report both lifecycle counts and stage durations; the old
    // combined Timings behavior is now represented explicitly by Detailed.
    let started = BvpSciTelemetry::detailed();
    let plan = BvpSciLambdifyPlan::prepare(
        assembly,
        // Exact continuation family: y' = p * (y + 1), y(0) = 0.
        &[Expr::parse_expression("p*(y + 1)")],
        &["y".into()],
        &["p".into()],
        "x",
        started,
    )
    .expect("story frontend should prepare");
    let target = Arc::new(AtomicU64::new(parameter.to_bits()));
    let target_for_boundary = Arc::clone(&target);
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        move |ya, yb, _, output| {
            let parameter = f64::from_bits(target_for_boundary.load(Ordering::Relaxed));
            output[0] = ya[0];
            // Keep the continuation family mathematically consistent:
            // y' = p*(y + 1), y(0)=0 has y(1)=exp(p)-1. This gate measures
            // prepared-model reuse on an exactly solvable BVP.
            output[1] = yb[0] - (parameter.exp() - 1.0);
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mut options = BvpSciOptions::default();
    options.matrix_layout = layout;
    options.execution_policy = execution_policy;
    options.tolerance = 1e-6;
    // The repeated continuation story reaches p=8. Keep enough adaptive
    // headroom to measure reuse rather than turning the fixture into a node
    // budget test; budget exhaustion has its own dedicated gate.
    options.max_nodes = 1_024;
    options.max_newton_iterations = 20;
    options.max_jacobian_refreshes = 10;
    (
        BvpSciSolver::new_with_sparse_factorization(
            plan,
            boundary,
            vec![0.0, 0.5, 1.0],
            vec![0.0, (parameter * 0.5).exp() - 1.0, parameter.exp() - 1.0],
            vec![parameter],
            options,
            sparse_factorization,
        )
        .expect("story solver should construct"),
        target,
    )
}

fn logistic_exact_state(parameter: f64, x: f64) -> f64 {
    1.0 / (1.0 + 4.0 * (-parameter * x).exp())
}

fn build_large_logistic_sparse_solver(
    sparse_factorization: BvpSciSparseFactorization,
    nodes: usize,
    initial_parameter: f64,
) -> (BvpSciSolver, Arc<AtomicU64>) {
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::AtomViewNative,
        &[Expr::parse_expression("p*y*(1-y)")],
        &["y".into()],
        &["p".into()],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("large logistic frontend should prepare");
    let target = Arc::new(AtomicU64::new(initial_parameter.to_bits()));
    let target_for_boundary = Arc::clone(&target);
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        move |ya, yb, _, output| {
            let parameter = f64::from_bits(target_for_boundary.load(Ordering::Relaxed));
            output[0] = ya[0] - 0.2;
            output[1] = yb[0] - logistic_exact_state(parameter, 1.0);
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect();
    let mut initial_state = Vec::with_capacity(nodes);
    for &x in &mesh {
        let y = logistic_exact_state(initial_parameter, x);
        let perturbation = 1e-3 * (std::f64::consts::PI * x).sin();
        initial_state.push(y + perturbation);
    }
    let mut options = BvpSciOptions::default();
    options.matrix_layout = BvpSciMatrixLayout::Sparse;
    options.tolerance = 1e-6;
    options.max_nodes = nodes.saturating_mul(2).max(nodes);
    options.max_mesh_refinements = 4;
    options.max_newton_iterations = 24;
    options.max_jacobian_refreshes = 12;
    let solver = BvpSciSolver::new_with_sparse_factorization(
        plan,
        boundary,
        mesh,
        initial_state,
        vec![initial_parameter],
        options,
        sparse_factorization,
    )
    .expect("large logistic sparse solver should construct");
    (solver, target)
}

fn build_official_large_parameterized_sparse_solver(
    sparse_factorization: BvpSciSparseFactorization,
) -> BvpSciSolver {
    build_official_large_parameterized_sparse_solver_with_output(
        sparse_factorization,
        BvpSciOutputPolicy::MeshAndValues,
    )
}

fn build_official_large_parameterized_sparse_solver_with_output(
    sparse_factorization: BvpSciSparseFactorization,
    output_policy: BvpSciOutputPolicy,
) -> BvpSciSolver {
    let systems = 30usize;
    let dimension = systems * 2;
    // This parity fixture compares both SciPy operation counts and stage
    // durations, so it must opt into both telemetry dimensions.
    let telemetry = BvpSciTelemetry::detailed();
    let mut equations = Vec::with_capacity(dimension);
    let mut state_names = Vec::with_capacity(dimension);
    for system in 0..systems {
        let even_name = format!("y{}", 2 * system);
        let odd_name = format!("y{}", 2 * system + 1);
        let parameter_name = format!("p{}", system % 2);
        state_names.extend([even_name.clone(), odd_name.clone()]);
        equations.push(Expr::parse_expression(&odd_name));
        equations.push(Expr::parse_expression(&format!(
            "-{parameter_name}*{parameter_name}*{even_name}"
        )));
    }
    let parameter_names = vec!["p0".into(), "p1".into()];
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &equations,
        &state_names,
        &parameter_names,
        "x",
        telemetry.clone(),
    )
    .expect("official large ExprLegacy frontend should prepare");
    let boundary = BvpSciBoundaryCallbacks::new_with_jacobian(
        dimension + 2,
        move |ya, yb, parameters, output| {
            for system in 0..systems {
                output[system] = ya[2 * system];
                output[systems + system] = yb[2 * system];
            }
            output[dimension] = ya[1] - parameters[0];
            output[dimension + 1] = ya[3] - parameters[1];
            Ok(())
        },
        move |_ya, _yb, _parameters, dya, dyb, dp| {
            dya.fill(0.0);
            dyb.fill(0.0);
            dp.fill(0.0);
            for system in 0..systems {
                dya[system * dimension + 2 * system] = 1.0;
                dyb[(systems + system) * dimension + 2 * system] = 1.0;
            }
            dya[dimension * dimension + 1] = 1.0;
            dya[(dimension + 1) * dimension + 3] = 1.0;
            dp[dimension * 2] = -1.0;
            dp[(dimension + 1) * 2 + 1] = -1.0;
            Ok(())
        },
        telemetry,
    );
    let mesh: Vec<f64> = (0..5)
        .map(|index| std::f64::consts::PI * index as f64 / 4.0)
        .collect();
    let mut options = BvpSciOptions::default();
    options.matrix_layout = BvpSciMatrixLayout::Sparse;
    options.tolerance = 1e-3;
    options.max_nodes = 1000;
    options.output_policy = output_policy;
    BvpSciSolver::new_with_sparse_factorization(
        plan,
        boundary,
        mesh.clone(),
        vec![1.0; mesh.len() * dimension],
        vec![0.5, 0.5],
        options,
        sparse_factorization,
    )
    .expect("official large parameterized sparse solver should construct")
}

fn median_and_mad(samples: &[f64]) -> (f64, f64) {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let median = ordered[ordered.len() / 2];
    let mut deviations: Vec<f64> = samples.iter().map(|value| (value - median).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    (median, deviations[deviations.len() / 2])
}

fn build_bratu_solver(assembly: BvpSciAssembly, layout: BvpSciMatrixLayout) -> BvpSciSolver {
    let plan = BvpSciLambdifyPlan::prepare(
        assembly,
        &[
            Expr::parse_expression("z"),
            Expr::parse_expression("-2*exp(y)"),
        ],
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("Bratu-like frontend should prepare");
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mut options = BvpSciOptions::default();
    options.matrix_layout = layout;
    options.tolerance = 1e-4;
    options.max_nodes = 16;
    options.max_newton_iterations = 30;
    options.max_jacobian_refreshes = 12;
    options.max_mesh_refinements = 0;
    let nodes = 16;
    BvpSciSolver::new(
        plan,
        boundary,
        (0..nodes)
            .map(|index| index as f64 / (nodes - 1) as f64)
            .collect(),
        vec![0.0; nodes * 2],
        vec![],
        options,
    )
    .expect("Bratu-like solver should construct")
}

#[test]
fn scipy_controller_trace_is_compact_and_typed() {
    #[derive(Tabled)]
    struct ControllerRow {
        iteration: u64,
        residual_before: String,
        residual_after: String,
        affine_cost_before: String,
        affine_cost_after: String,
        armijo_alpha: String,
        backtracking_trials: u64,
        jacobian_refreshed: bool,
        accepted: bool,
    }

    let mut solver = build_bratu_solver(BvpSciAssembly::ExprLegacy, BvpSciMatrixLayout::Dense);
    let solution = solver.solve().expect("controller fixture should converge");
    let snapshot = solver.plan().telemetry_snapshot();
    assert_eq!(solution.status, BvpSciStatus::Success);
    assert!(!snapshot.newton_residual_history.is_empty());
    assert!(snapshot.newton_jacobian_refreshes <= 12);

    let rows: Vec<ControllerRow> = snapshot
        .newton_residual_history
        .iter()
        .map(|entry| {
            assert!(entry.affine_cost_before.is_finite());
            assert!(entry.armijo_alpha.is_finite());
            assert!(entry.armijo_alpha > 0.0 && entry.armijo_alpha <= 1.0);
            assert!(entry.backtracking_trials > 0);
            // SciPy's n_trial=4 means one full evaluation plus four
            // backtracking evaluations.
            assert!(entry.backtracking_trials <= 5);
            if let Some(cost_after) = entry.affine_cost_after {
                assert!(cost_after.is_finite());
            }
            ControllerRow {
                iteration: entry.iteration,
                residual_before: format!("{:.3e}", entry.residual_before),
                residual_after: entry
                    .residual_after
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "n/a".into()),
                affine_cost_before: format!("{:.3e}", entry.affine_cost_before),
                affine_cost_after: entry
                    .affine_cost_after
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "n/a".into()),
                armijo_alpha: format!("{:.3}", entry.armijo_alpha),
                backtracking_trials: entry.backtracking_trials,
                jacobian_refreshed: entry.jacobian_refreshed,
                accepted: entry.accepted,
            }
        })
        .collect();
    println!("# BVP_sci SciPy controller trace\n\n{}", Table::new(rows));
}

fn lane_emden_exact(x: f64) -> (f64, f64) {
    let factor = 1.0 + x * x / 3.0;
    (factor.powf(-0.5), -x / 3.0 * factor.powf(-1.5))
}

fn build_lane_emden_solver(assembly: BvpSciAssembly, layout: BvpSciMatrixLayout) -> BvpSciSolver {
    let left = 1.0e-3;
    let right = 1.0;
    let plan = BvpSciLambdifyPlan::prepare(
        assembly,
        &[
            Expr::parse_expression("z"),
            Expr::parse_expression("-2*z/x - y^5"),
        ],
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::timings(),
    )
    .expect("Lane-Emden frontend should prepare");
    let (left_y, _) = lane_emden_exact(left);
    let (right_y, _) = lane_emden_exact(right);
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        move |ya, yb, _, output| {
            output[0] = ya[0] - left_y;
            output[1] = yb[0] - right_y;
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let nodes = 24;
    let x: Vec<f64> = (0..nodes)
        .map(|index| left + (right - left) * index as f64 / (nodes - 1) as f64)
        .collect();
    let y = x
        .iter()
        .flat_map(|&x| {
            let (value, derivative) = lane_emden_exact(x);
            [value, derivative]
        })
        .collect();
    let mut options = BvpSciOptions::default();
    options.matrix_layout = layout;
    options.tolerance = 1e-3;
    options.max_nodes = 128;
    options.max_mesh_refinements = 8;
    options.max_newton_iterations = 40;
    options.max_jacobian_refreshes = 10;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    BvpSciSolver::new(plan, boundary, x, y, vec![], options)
        .expect("Lane-Emden solver should construct")
}

#[derive(Clone, Copy, Debug)]
enum DampExactWorkload {
    Linear,
    Oscillator,
    StiffDecay,
    TwoPoint,
    Clairaut,
    Parachute,
    StiffCoupled,
}

impl DampExactWorkload {
    fn label(self) -> &'static str {
        match self {
            Self::Linear => "linear-bvp",
            Self::Oscillator => "oscillator",
            Self::StiffDecay => "stiff-decay",
            Self::TwoPoint => "two-point-bvp",
            Self::Clairaut => "clairaut",
            Self::Parachute => "parachute",
            Self::StiffCoupled => "stiff-coupled",
        }
    }

    fn interval(self) -> (f64, f64) {
        match self {
            Self::Linear | Self::StiffDecay => (0.0, 1.0),
            Self::Oscillator => (0.0, std::f64::consts::FRAC_PI_2),
            Self::TwoPoint => (-1.0, 1.0),
            Self::Clairaut | Self::Parachute | Self::StiffCoupled => (0.0, 1.0),
        }
    }

    fn state_names(self) -> Vec<String> {
        match self {
            Self::Linear | Self::StiffDecay => vec!["y".into()],
            Self::Oscillator => vec!["y".into(), "z".into()],
            Self::TwoPoint | Self::Parachute => vec!["y".into(), "z".into()],
            Self::Clairaut => vec!["y".into(), "z".into(), "zz".into()],
            Self::StiffCoupled => vec!["y0".into(), "y1".into(), "y2".into()],
        }
    }

    fn equations(self) -> Vec<Expr> {
        match self {
            Self::Linear => vec![Expr::parse_expression("1")],
            Self::Oscillator => vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            Self::StiffDecay => vec![Expr::parse_expression("-20*y")],
            Self::TwoPoint => vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("-0.5*(1+2*ln(y))*y"),
            ],
            Self::Clairaut => vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("zz"),
                // The legacy BVP_Damp equation is not satisfied by its own
                // published polynomial profile under the standard first-order
                // interpretation. Use a nonlinear manufactured residual around
                // that profile instead of promoting the inconsistent fixture
                // to a false correctness oracle.
                Expr::parse_expression(
                    "-1+2*(x-1)+0.1*(y-(1+(x-1)^2-(x-1)^3/6+(x-1)^4/12))+0.01*(y-(1+(x-1)^2-(x-1)^3/6+(x-1)^4/12))^2",
                ),
            ],
            Self::Parachute => vec![Expr::parse_expression("z"), Expr::parse_expression("1-z^2")],
            Self::StiffCoupled => vec![
                Expr::parse_expression("-20*y0+10*y1+9"),
                Expr::parse_expression("-40*y1+20*y2+18-20*x"),
                Expr::parse_expression("-80*y2+77-240*x"),
            ],
        }
    }

    fn exact_state(self, x: f64) -> Vec<f64> {
        match self {
            Self::Linear => vec![x],
            Self::Oscillator => vec![x.sin(), x.cos()],
            Self::StiffDecay => vec![(-20.0 * x).exp()],
            Self::TwoPoint => {
                let y = (-x * x / 4.0).exp();
                vec![y, -0.5 * x * y]
            }
            Self::Clairaut => {
                let u = x - 1.0;
                vec![
                    1.0 + u * u - u.powi(3) / 6.0 + u.powi(4) / 12.0,
                    2.0 * u - u * u / 2.0 + u.powi(3) / 3.0,
                    2.0 - u + u * u,
                ]
            }
            Self::Parachute => vec![(((2.0 * x).exp() + 1.0) / 2.0).ln() - x, x.tanh()],
            Self::StiffCoupled => vec![1.0 - x, 1.0 - 2.0 * x, 1.0 - 3.0 * x],
        }
    }

    fn boundary_dimension(self) -> usize {
        self.state_names().len()
    }
}

fn build_damp_exact_solver(
    workload: DampExactWorkload,
    assembly: BvpSciAssembly,
    layout: BvpSciMatrixLayout,
) -> BvpSciSolver {
    let default_nodes = if matches!(workload, DampExactWorkload::StiffDecay) {
        64
    } else {
        20
    };
    build_damp_exact_solver_with_nodes(workload, assembly, layout, default_nodes)
}

fn build_damp_exact_solver_with_nodes(
    workload: DampExactWorkload,
    assembly: BvpSciAssembly,
    layout: BvpSciMatrixLayout,
    nodes: usize,
) -> BvpSciSolver {
    build_damp_exact_solver_with_nodes_and_telemetry(
        workload,
        assembly,
        layout,
        nodes,
        BvpSciTelemetry::timings(),
    )
}

fn build_damp_exact_solver_with_nodes_and_telemetry(
    workload: DampExactWorkload,
    assembly: BvpSciAssembly,
    layout: BvpSciMatrixLayout,
    nodes: usize,
    telemetry: BvpSciTelemetry,
) -> BvpSciSolver {
    build_damp_exact_solver_with_nodes_telemetry_and_sparse_factorization(
        workload,
        assembly,
        layout,
        nodes,
        telemetry,
        BvpSciSparseFactorization::StandardFaer,
    )
}

fn build_damp_exact_solver_with_nodes_telemetry_and_sparse_factorization(
    workload: DampExactWorkload,
    assembly: BvpSciAssembly,
    layout: BvpSciMatrixLayout,
    nodes: usize,
    telemetry: BvpSciTelemetry,
    sparse_factorization: BvpSciSparseFactorization,
) -> BvpSciSolver {
    let (left, right) = workload.interval();
    let plan = BvpSciLambdifyPlan::prepare(
        assembly,
        &workload.equations(),
        &workload.state_names(),
        &[],
        "x",
        telemetry,
    )
    .expect("BVP_Damp exact frontend should prepare");
    let left_state = workload.exact_state(left);
    let right_state = workload.exact_state(right);
    let boundary = match workload {
        DampExactWorkload::Linear => BvpSciBoundaryCallbacks::new(
            workload.boundary_dimension(),
            move |ya, _, _, output| {
                output[0] = ya[0] - left_state[0];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        DampExactWorkload::Oscillator => BvpSciBoundaryCallbacks::new(
            workload.boundary_dimension(),
            move |ya, yb, _, output| {
                output[0] = ya[0] - left_state[0];
                output[1] = yb[0] - right_state[0];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        DampExactWorkload::StiffDecay => BvpSciBoundaryCallbacks::new(
            workload.boundary_dimension(),
            move |ya, _, _, output| {
                output[0] = ya[0] - left_state[0];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        DampExactWorkload::TwoPoint => BvpSciBoundaryCallbacks::new(
            workload.boundary_dimension(),
            move |ya, yb, _, output| {
                output[0] = ya[0] - left_state[0];
                output[1] = yb[0] - right_state[0];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        DampExactWorkload::Clairaut => BvpSciBoundaryCallbacks::new(
            workload.boundary_dimension(),
            move |_ya, yb, _, output| {
                output[0] = yb[0] - right_state[0];
                output[1] = yb[1] - right_state[1];
                output[2] = yb[2] - right_state[2];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        DampExactWorkload::Parachute => BvpSciBoundaryCallbacks::new(
            workload.boundary_dimension(),
            move |ya, _yb, _, output| {
                output[0] = ya[0] - left_state[0];
                output[1] = ya[1] - left_state[1];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
        DampExactWorkload::StiffCoupled => BvpSciBoundaryCallbacks::new(
            workload.boundary_dimension(),
            move |ya, _, _, output| {
                output[0] = ya[0] - left_state[0];
                output[1] = ya[1] - left_state[1];
                output[2] = ya[2] - left_state[2];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        ),
    };
    let x: Vec<f64> = (0..nodes)
        .map(|index| left + (right - left) * index as f64 / (nodes - 1) as f64)
        .collect();
    let y = x
        .iter()
        .flat_map(|&x| workload.exact_state(x).into_iter().map(|value| 0.9 * value))
        .collect();
    let mut options = BvpSciOptions::default();
    options.matrix_layout = layout;
    options.tolerance = 2e-4;
    options.max_nodes = nodes;
    options.max_mesh_refinements = 0;
    options.max_newton_iterations = 30;
    options.max_jacobian_refreshes = 12;
    if matches!(workload, DampExactWorkload::StiffDecay) {
        options.max_jacobian_refreshes = 30;
    }
    BvpSciSolver::new_with_sparse_factorization(
        plan,
        boundary,
        x,
        y,
        vec![],
        options,
        sparse_factorization,
    )
    .expect("BVP_Damp exact solver should construct")
}

#[test]
fn lambdify_frontend_backend_fidelity_matrix_is_compact_and_complete() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_frontend_backend_fidelity_matrix",
    );
    let (mut reference, reference_target) = build_solver(
        BvpSciAssembly::ExprLegacy,
        BvpSciMatrixLayout::Dense,
        1.0,
        BvpSciExecutionPolicy::Sequential,
    );
    let reference_solution = reference.solve().expect("reference should converge");
    reference_target.store(2.0f64.to_bits(), Ordering::Relaxed);
    reference
        .set_parameters(vec![2.0])
        .expect("reference continuation rebind should work");
    let reference_continued = reference
        .solve()
        .expect("reference continuation should converge");
    let mut rows = Vec::new();
    for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
        for layout in [
            BvpSciMatrixLayout::Dense,
            BvpSciMatrixLayout::Sparse,
            BvpSciMatrixLayout::Banded {
                lower: 256,
                upper: 256,
            },
        ] {
            let started = Instant::now();
            let (mut solver, target) =
                build_solver(assembly, layout, 1.0, BvpSciExecutionPolicy::Sequential);
            let prepare_ms = started.elapsed().as_secs_f64() * 1e3;
            let solve_started = Instant::now();
            let solution = solver.solve().expect("matrix row should converge");
            let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
            let continuation_started = Instant::now();
            target.store(2.0f64.to_bits(), Ordering::Relaxed);
            solver
                .set_parameters(vec![2.0])
                .expect("rebind should work");
            let continued = solver.solve().expect("continuation row should converge");
            let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
            let continuation_diff = continued
                .y
                .iter()
                .zip(&reference_continued.y)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            let max_diff = solution
                .y
                .iter()
                .zip(&reference_solution.y)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            assert!(max_diff < 1e-5, "frontend/layout drift={max_diff}");
            assert!(
                continuation_diff < 1e-5,
                "continuation drift={continuation_diff}"
            );
            let telemetry = solver.plan().telemetry_snapshot();
            rows.push(MatrixRow {
                frontend: format!("{:?}", assembly),
                layout: layout_name(layout),
                prepare_ms: format!("{prepare_ms:.3}"),
                solve_ms: format!("{solve_ms:.3}"),
                continuation_ms: format!("{continuation_ms:.3}"),
                continuation_diff: format!("{continuation_diff:.3e}"),
                symbolic_jacobian_ms: timing(telemetry.symbolic_jacobian_ms),
                callback_ms: timing(telemetry.callback_ms),
                residual_ms: timing(telemetry.residual_evaluation_ms),
                jacobian_ms: timing(telemetry.jacobian_evaluation_ms),
                factorization_ms: timing(telemetry.factorization_ms),
                residual_norm: format!("{:.3e}", solution.residual_norm),
                max_fidelity_diff: format!("{max_diff:.3e}"),
                status: format!("{}", solution.status.code()),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci Lambdify frontend/backend fidelity\n\n{}",
        Table::new(&rows)
    ));
}

#[test]
fn lambdify_continuation_keeps_preparation_and_reports_stage_counters() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_continuation_lifecycle",
    );
    let (mut solver, target) = build_solver(
        BvpSciAssembly::AtomViewNative,
        BvpSciMatrixLayout::Sparse,
        1.0,
        BvpSciExecutionPolicy::Sequential,
    );
    solver
        .solve()
        .expect("initial continuation solve should work");
    for parameter in 2..=8 {
        let parameter = parameter as f64;
        target.store(parameter.to_bits(), Ordering::Relaxed);
        solver.set_parameters(vec![parameter]).unwrap();
        solver.solve().expect("continued solve should work");
    }
    let telemetry = solver.plan().telemetry_snapshot();
    assert_eq!(telemetry.atom_conversions, 1);
    assert!(telemetry.pattern_entries > 0);
    assert_eq!(telemetry.parameter_rebinds, 7);
    assert_eq!(telemetry.continuation_solves, 7);
    assert!(telemetry.allocations > 0);
    assert!(telemetry.residual_evaluations > 0);
    assert!(telemetry.jacobian_evaluations > 0);
    assert!(telemetry.parameter_jacobian_calls > 0);
    crate::Utils::test_reporting::capture_test_block(format!(
        "# Continuation lifecycle\n\n- atom_conversions: {}\n- pattern_entries: {}\n- parameter_rebinds: {}\n- continuation_solves: {}\n- allocations: {}\n- residual_evaluations: {}\n- jacobian_evaluations: {}\n- parameter_jacobian_calls: {}\n- factorization_ms: {:?}",
        telemetry.atom_conversions,
        telemetry.pattern_entries,
        telemetry.parameter_rebinds,
        telemetry.continuation_solves,
        telemetry.allocations,
        telemetry.residual_evaluations,
        telemetry.jacobian_evaluations,
        telemetry.parameter_jacobian_calls,
        telemetry.factorization_ms,
    ));
}

// Experimental A/B implementation. Registration lives in
// `story_tests_experimental` so exploratory evidence stays separate from
// supported solver story tests.
pub(super) fn run_sparse_reusable_kernel_full_solve_continuation_ab_is_compact() {
    #[derive(Tabled)]
    struct Row {
        sparse_kernel: String,
        effective_backend: String,
        superlu_fallbacks: u64,
        superlu_stage_ms: String,
        superlu_factor_fill: String,
        nodes: usize,
        system_dimension: usize,
        repeats: usize,
        continuation_solves: usize,
        prepare_median_mad_ms: String,
        solve_series_median_mad_ms: String,
        paired_series_delta_vs_standard_pct: String,
        backend_factorizations: u64,
        faer_numeric_factorizations: u64,
        symbolic_analyses: u64,
        symbolic_reuses: u64,
        numeric_factor_median_mad_ms: String,
        paired_factor_delta_vs_standard_pct: String,
        max_final_mesh_diff: String,
        max_final_state_diff: String,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "sparse_reusable_kernel_full_solve_continuation_ab",
    );
    let repeats = std::env::var("BVP_SCI_SPARSE_REUSE_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);
    let continuation_solves = std::env::var("BVP_SCI_SPARSE_REUSE_CONTINUATION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6)
        .max(1);
    let nodes = std::env::var("BVP_SCI_SPARSE_REUSE_NODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512)
        .max(16);
    let strategies = vec![
        ("standard-faer", BvpSciSparseFactorization::StandardFaer),
        (
            "reusable-faer-experimental",
            BvpSciSparseFactorization::ReusableFaerExperimental,
        ),
        (
            "reusable-faer-simplicial-experimental",
            BvpSciSparseFactorization::ReusableFaerSimplicialExperimental,
        ),
    ];
    #[cfg(feature = "superlu")]
    let strategies = {
        let mut strategies = strategies;
        strategies.push(("superlu", BvpSciSparseFactorization::SuperLu));
        strategies
    };
    struct Sample {
        prepare_ms: f64,
        series_ms: f64,
        factor_ms: f64,
        snapshot: BvpSciTelemetrySnapshot,
        x: Vec<f64>,
        y: Vec<f64>,
    }
    let mut samples: Vec<Vec<Sample>> = (0..strategies.len())
        .map(|_| Vec::with_capacity(repeats))
        .collect();
    let initial_parameter = 1.0;
    let mut max_mesh_diffs = vec![0.0_f64; strategies.len()];
    let mut max_state_diffs = vec![0.0_f64; strategies.len()];
    let mut paired_series_deltas = vec![Vec::with_capacity(repeats); strategies.len()];
    let mut paired_factor_deltas = vec![Vec::with_capacity(repeats); strategies.len()];

    // Rotate policy order on each repetition to reduce systematic CPU/frequency
    // drift between the first and last backend measured.
    for repetition in 0..repeats {
        let order: Vec<usize> = (0..strategies.len())
            .map(|offset| (offset + repetition) % strategies.len())
            .collect();
        for policy_index in order {
            let (name, strategy) = strategies[policy_index];
            let prepare_started = Instant::now();
            let (mut solver, target) =
                build_large_logistic_sparse_solver(strategy, nodes, initial_parameter);
            let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;

            let solve_started = Instant::now();
            let mut solution = solver.solve().unwrap_or_else(|error| {
                panic!("large logistic initial solve failed for {name}: {error:?}")
            });
            for index in 0..continuation_solves {
                let parameter = initial_parameter + (index + 1) as f64 * 0.1;
                target.store(parameter.to_bits(), Ordering::Relaxed);
                solver
                    .set_parameters(vec![parameter])
                    .unwrap_or_else(|error| {
                        panic!("large logistic parameter rebind failed for {name}: {error:?}")
                    });
                solution = solver.solve().unwrap_or_else(|error| {
                    panic!("large logistic continuation failed for {name}: {error:?}")
                });
            }
            let series_ms = solve_started.elapsed().as_secs_f64() * 1e3;
            let snapshot = solver.plan().telemetry_snapshot();
            let factor_ms = sparse_factor_timing_ms(&snapshot, strategy)
                .expect("selected sparse factor timing should be recorded");
            samples[policy_index].push(Sample {
                prepare_ms,
                series_ms,
                factor_ms,
                snapshot,
                x: solution.x,
                y: solution.y,
            });
        }

        let reference = samples[0]
            .last()
            .expect("standard policy must be measured in every repetition");
        for policy_index in 1..strategies.len() {
            let candidate = samples[policy_index]
                .last()
                .expect("each policy must be measured in every repetition");
            assert_eq!(candidate.x.len(), reference.x.len());
            assert_eq!(candidate.y.len(), reference.y.len());
            let mesh_diff = candidate
                .x
                .iter()
                .zip(&reference.x)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            let state_diff = candidate
                .y
                .iter()
                .zip(&reference.y)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            assert!(
                mesh_diff < 1e-12,
                "{name} mesh drift={mesh_diff:e}",
                name = strategies[policy_index].0
            );
            assert!(
                state_diff < 1e-7,
                "{name} solution drift={state_diff:e}",
                name = strategies[policy_index].0
            );
            max_mesh_diffs[policy_index] = max_mesh_diffs[policy_index].max(mesh_diff);
            max_state_diffs[policy_index] = max_state_diffs[policy_index].max(state_diff);
            paired_series_deltas[policy_index]
                .push(100.0 * (candidate.series_ms - reference.series_ms) / reference.series_ms);
            paired_factor_deltas[policy_index]
                .push(100.0 * (candidate.factor_ms - reference.factor_ms) / reference.factor_ms);
        }
    }

    let reference_counters = &samples[0]
        .last()
        .expect("standard policy must have a final sample")
        .snapshot;
    assert!(
        reference_counters.sparse_numeric_factorizations > 0,
        "large Bratu series must execute numeric sparse factorizations"
    );
    assert!(
        reference_counters.sparse_symbolic_analyses > 0,
        "large Bratu series must execute symbolic sparse analyses"
    );
    for (policy_index, (name, _)) in strategies.iter().enumerate().skip(1) {
        let snapshot = &samples[policy_index]
            .last()
            .expect("all policies must have a final sample")
            .snapshot;
        assert_eq!(
            snapshot.factorizations, reference_counters.factorizations,
            "{name} backend factorization count differs from StandardFaer"
        );
        assert_eq!(
            snapshot.linear_solves, reference_counters.linear_solves,
            "{name} linear solve count differs from StandardFaer"
        );
        if !matches!(
            strategies[policy_index].1,
            BvpSciSparseFactorization::SuperLu
        ) {
            assert_eq!(
                snapshot.sparse_numeric_factorizations,
                reference_counters.sparse_numeric_factorizations,
                "{name} numeric factorization count differs from StandardFaer"
            );
            assert_eq!(
                snapshot.sparse_symbolic_analyses, reference_counters.sparse_symbolic_analyses,
                "{name} symbolic analysis count differs from StandardFaer"
            );
            assert_eq!(
                snapshot.sparse_symbolic_reuses, reference_counters.sparse_symbolic_reuses,
                "{name} symbolic reuse count differs from StandardFaer"
            );
        }
    }

    let mut rows = Vec::with_capacity(strategies.len());
    for (policy_index, (name, _)) in strategies.iter().enumerate() {
        let preparations: Vec<f64> = samples[policy_index]
            .iter()
            .map(|sample| sample.prepare_ms)
            .collect();
        let series: Vec<f64> = samples[policy_index]
            .iter()
            .map(|sample| sample.series_ms)
            .collect();
        let factors: Vec<f64> = samples[policy_index]
            .iter()
            .map(|sample| sample.factor_ms)
            .collect();
        let (prepare_median, prepare_mad) = median_and_mad(&preparations);
        let (series_median, series_mad) = median_and_mad(&series);
        let (factor_median, factor_mad) = median_and_mad(&factors);
        let snapshot = &samples[policy_index]
            .last()
            .expect("all policies must have measurements")
            .snapshot;
        let (series_delta, _) = if policy_index == 0 {
            (0.0, 0.0)
        } else {
            median_and_mad(&paired_series_deltas[policy_index])
        };
        let (factor_delta, _) = if policy_index == 0 {
            (0.0, 0.0)
        } else {
            median_and_mad(&paired_factor_deltas[policy_index])
        };
        rows.push(Row {
            sparse_kernel: (*name).into(),
            effective_backend: sparse_policy_effective_backend(
                snapshot,
                strategies[policy_index].1,
            )
            .into(),
            superlu_fallbacks: snapshot.superlu_fallbacks,
            superlu_stage_ms: superlu_stage_summary(snapshot),
            superlu_factor_fill: superlu_factor_fill_summary(snapshot),
            nodes,
            system_dimension: nodes * 2,
            repeats,
            continuation_solves,
            prepare_median_mad_ms: format!("{prepare_median:.3} / {prepare_mad:.3}"),
            solve_series_median_mad_ms: format!("{series_median:.3} / {series_mad:.3}"),
            paired_series_delta_vs_standard_pct: format!("{series_delta:+.2}%"),
            backend_factorizations: snapshot.factorizations,
            faer_numeric_factorizations: snapshot.sparse_numeric_factorizations,
            symbolic_analyses: snapshot.sparse_symbolic_analyses,
            symbolic_reuses: snapshot.sparse_symbolic_reuses,
            numeric_factor_median_mad_ms: format!("{factor_median:.3} / {factor_mad:.3}"),
            paired_factor_delta_vs_standard_pct: format!("{factor_delta:+.2}%"),
            max_final_mesh_diff: format!("{:.3e}", max_mesh_diffs[policy_index]),
            max_final_state_diff: format!("{:.3e}", max_state_diffs[policy_index]),
            status: if strategies[policy_index].1 == BvpSciSparseFactorization::SuperLu
                && snapshot.superlu_fallbacks > 0
            {
                "fallback-not-superlu".into()
            } else {
                "ok".into()
            },
        });
    }

    crate::Utils::test_reporting::capture_test_block(format!(
        "# Large nonlinear sparse LU full-solve and continuation A/B\n\nWorkload: parameterized logistic BVP with AtomViewNative, `{nodes}` initial mesh nodes/unknowns, one initial solve plus `{continuation_solves}` prepared parameter rebinds. Run `{repeats}` paired repetitions; strategy order rotates between repetitions. Time columns are median / median absolute deviation (MAD). Paired deltas are median per-repeat percent changes against StandardFaer. Factor time and counters are cumulative over the final series; they are diagnostic scopes, not additive with wall time.\n\n{}",
        Table::new(&rows)
    ));
    assert!(
        samples.iter().enumerate().all(|(index, policy_samples)| {
            strategies[index].1 != BvpSciSparseFactorization::SuperLu
                || policy_samples
                    .iter()
                    .all(|sample| sample.snapshot.superlu_fallbacks == 0)
        }),
        "requested SuperLU policy fell back; inspect the captured effective_backend column"
    );
}

// Experimental policy matrix; the ignored test wrapper is in
// `story_tests_experimental`.
pub(super) fn run_sparse_kernel_real_workload_matrix_ab_is_compact() {
    #[derive(Tabled)]
    struct Row {
        workload: String,
        nodes: usize,
        dimension: usize,
        policy: String,
        effective_backend: String,
        superlu_fallbacks: u64,
        superlu_stage_ms: String,
        superlu_factor_fill: String,
        repeats: usize,
        prepare_median_mad_ms: String,
        full_solve_median_mad_ms: String,
        linear_assembly_ms: String,
        factor_median_mad_ms: String,
        linear_solve_ms: String,
        paired_solve_delta_vs_standard_pct: String,
        residual_calls: u64,
        jacobian_calls: u64,
        linear_solves: u64,
        backend_factorizations: u64,
        faer_numeric_factorizations: u64,
        symbolic_analyses: u64,
        symbolic_reuses: u64,
        max_exact_error: String,
        status: String,
    }

    struct Sample {
        prepare_ms: f64,
        solve_ms: f64,
        factor_ms: f64,
        exact_error: f64,
        snapshot: BvpSciTelemetrySnapshot,
        x: Vec<f64>,
        y: Vec<f64>,
    }

    fn selected_workloads() -> Vec<DampExactWorkload> {
        std::env::var("BVP_SCI_SPARSE_MATRIX_WORKLOADS")
            .unwrap_or_else(|_| "stiff-decay,stiff-coupled".into())
            .split(',')
            .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
                "stiff-decay" => Some(DampExactWorkload::StiffDecay),
                "stiff-coupled" => Some(DampExactWorkload::StiffCoupled),
                _ => None,
            })
            .collect()
    }

    fn workload_name(workload: DampExactWorkload) -> &'static str {
        match workload {
            DampExactWorkload::StiffDecay => "stiff-decay",
            DampExactWorkload::StiffCoupled => "stiff-coupled",
            _ => "unsupported",
        }
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "sparse_kernel_real_workload_matrix_ab",
    );
    let repeats = std::env::var("BVP_SCI_SPARSE_MATRIX_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .max(3);
    let nodes_list: Vec<usize> = std::env::var("BVP_SCI_SPARSE_MATRIX_NODES")
        .unwrap_or_else(|_| "64,256,1024".into())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .filter(|nodes: &usize| *nodes >= 16)
        .collect();
    let strategies = vec![
        ("standard-faer", BvpSciSparseFactorization::StandardFaer),
        (
            "reusable-faer-auto",
            BvpSciSparseFactorization::ReusableFaerExperimental,
        ),
        (
            "reusable-faer-simplicial",
            BvpSciSparseFactorization::ReusableFaerSimplicialExperimental,
        ),
    ];
    #[cfg(feature = "superlu")]
    let strategies = {
        let mut strategies = strategies;
        strategies.push(("superlu", BvpSciSparseFactorization::SuperLu));
        strategies
    };
    let mut rows = Vec::new();

    for workload in selected_workloads() {
        for &nodes in &nodes_list {
            if matches!(workload, DampExactWorkload::StiffDecay) && nodes < 64 {
                continue;
            }
            let mut samples: Vec<Vec<Sample>> = (0..strategies.len())
                .map(|_| Vec::with_capacity(repeats))
                .collect();
            let mut paired_deltas: Vec<Vec<f64>> = (0..strategies.len())
                .map(|_| Vec::with_capacity(repeats))
                .collect();

            // Interleave strategies so CPU frequency and thermal drift do not
            // systematically favor one factorization policy.
            for repetition in 0..repeats {
                for offset in 0..strategies.len() {
                    let policy_index = (offset + repetition) % strategies.len();
                    let (_, strategy) = strategies[policy_index];
                    let prepare_started = Instant::now();
                    let mut solver =
                        build_damp_exact_solver_with_nodes_telemetry_and_sparse_factorization(
                            workload,
                            BvpSciAssembly::AtomViewNative,
                            BvpSciMatrixLayout::Sparse,
                            nodes,
                            BvpSciTelemetry::timings(),
                            strategy,
                        );
                    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
                    let solve_started = Instant::now();
                    let solution = solver.solve().unwrap_or_else(|error| {
                        panic!(
                            "{} nodes={nodes} policy={} failed: {error:?}",
                            workload_name(workload),
                            strategies[policy_index].0
                        )
                    });
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
                    let snapshot = solver.plan().telemetry_snapshot();
                    let factor_ms = sparse_factor_timing_ms(&snapshot, strategy)
                        .expect("selected Sparse backend should record factor time");
                    let mut exact_error = 0.0_f64;
                    for (node, &x) in solution.x.iter().enumerate() {
                        let expected = workload.exact_state(x);
                        for (component, expected_value) in expected.iter().enumerate() {
                            let actual = solution.y[node * expected.len() + component];
                            exact_error = exact_error.max((actual - expected_value).abs());
                        }
                    }
                    assert!(
                        exact_error < 1e-3,
                        "{} nodes={nodes} exact-solution error={exact_error:e}",
                        workload_name(workload)
                    );
                    samples[policy_index].push(Sample {
                        prepare_ms,
                        solve_ms,
                        factor_ms,
                        exact_error,
                        snapshot,
                        x: solution.x,
                        y: solution.y,
                    });
                }

                let reference = samples[0]
                    .last()
                    .expect("StandardFaer is measured once per repetition");
                for policy_index in 1..strategies.len() {
                    let candidate = samples[policy_index]
                        .last()
                        .expect("every policy is measured once per repetition");
                    assert_eq!(candidate.x.len(), reference.x.len());
                    assert_eq!(candidate.y.len(), reference.y.len());
                    let mesh_diff = candidate
                        .x
                        .iter()
                        .zip(&reference.x)
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0, f64::max);
                    let state_diff = candidate
                        .y
                        .iter()
                        .zip(&reference.y)
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0, f64::max);
                    assert!(mesh_diff <= 1e-12, "mesh drift={mesh_diff:e}");
                    assert!(state_diff <= 1e-8, "state drift={state_diff:e}");
                    assert_eq!(
                        candidate.snapshot.factorizations, reference.snapshot.factorizations,
                        "backend factorization count differs for {} nodes={nodes}",
                        strategies[policy_index].0
                    );
                    assert_eq!(
                        candidate.snapshot.linear_solves, reference.snapshot.linear_solves,
                        "linear solve count differs for {} nodes={nodes}",
                        strategies[policy_index].0
                    );
                    if !matches!(
                        strategies[policy_index].1,
                        BvpSciSparseFactorization::SuperLu
                    ) {
                        assert_eq!(
                            candidate.snapshot.sparse_numeric_factorizations,
                            reference.snapshot.sparse_numeric_factorizations,
                            "faer numeric factorization count differs for {} nodes={nodes}",
                            strategies[policy_index].0
                        );
                        assert_eq!(
                            candidate.snapshot.sparse_symbolic_analyses,
                            reference.snapshot.sparse_symbolic_analyses,
                            "faer symbolic analysis count differs for {} nodes={nodes}",
                            strategies[policy_index].0
                        );
                        assert_eq!(
                            candidate.snapshot.sparse_symbolic_reuses,
                            reference.snapshot.sparse_symbolic_reuses,
                            "faer symbolic reuse count differs for {} nodes={nodes}",
                            strategies[policy_index].0
                        );
                    }
                    assert!(
                        (candidate.exact_error - reference.exact_error).abs() < 1e-8,
                        "solution accuracy differs for {} nodes={nodes}",
                        strategies[policy_index].0
                    );
                    paired_deltas[policy_index].push(
                        100.0 * (candidate.solve_ms - reference.solve_ms) / reference.solve_ms,
                    );
                }
            }

            for (policy_index, (policy, _)) in strategies.iter().enumerate() {
                let policy_samples = &samples[policy_index];
                let (prepare_median, prepare_mad) = median_and_mad(
                    &policy_samples
                        .iter()
                        .map(|sample| sample.prepare_ms)
                        .collect::<Vec<_>>(),
                );
                let (solve_median, solve_mad) = median_and_mad(
                    &policy_samples
                        .iter()
                        .map(|sample| sample.solve_ms)
                        .collect::<Vec<_>>(),
                );
                let (factor_median, factor_mad) = median_and_mad(
                    &policy_samples
                        .iter()
                        .map(|sample| sample.factor_ms)
                        .collect::<Vec<_>>(),
                );
                let linear_assembly_median = median_and_mad(
                    &policy_samples
                        .iter()
                        .map(|sample| {
                            sample
                                .snapshot
                                .linear_assembly_ms
                                .expect("linear assembly timing should be recorded")
                        })
                        .collect::<Vec<_>>(),
                )
                .0;
                let linear_solve_median = median_and_mad(
                    &policy_samples
                        .iter()
                        .map(|sample| {
                            sample
                                .snapshot
                                .linear_solve_ms
                                .expect("linear solve timing should be recorded")
                        })
                        .collect::<Vec<_>>(),
                )
                .0;
                let reference = &samples[0][0].snapshot;
                let snapshot = &policy_samples[0].snapshot;
                let paired_delta = if policy_index == 0 {
                    0.0
                } else {
                    median_and_mad(&paired_deltas[policy_index]).0
                };
                let max_exact_error = policy_samples
                    .iter()
                    .map(|sample| sample.exact_error)
                    .fold(0.0, f64::max);
                rows.push(Row {
                    workload: workload_name(workload).into(),
                    nodes,
                    dimension: nodes * workload.state_names().len(),
                    policy: (*policy).into(),
                    effective_backend: sparse_policy_effective_backend(
                        snapshot,
                        strategies[policy_index].1,
                    )
                    .into(),
                    superlu_fallbacks: snapshot.superlu_fallbacks,
                    superlu_stage_ms: superlu_stage_summary(snapshot),
                    superlu_factor_fill: superlu_factor_fill_summary(snapshot),
                    repeats,
                    prepare_median_mad_ms: format!("{prepare_median:.3}/{prepare_mad:.3}"),
                    full_solve_median_mad_ms: format!("{solve_median:.3}/{solve_mad:.3}"),
                    linear_assembly_ms: format!("{linear_assembly_median:.3}"),
                    factor_median_mad_ms: format!("{factor_median:.3}/{factor_mad:.3}"),
                    linear_solve_ms: format!("{linear_solve_median:.3}"),
                    paired_solve_delta_vs_standard_pct: format!("{paired_delta:+.2}%"),
                    residual_calls: snapshot.residual_evaluations,
                    jacobian_calls: snapshot.jacobian_evaluations,
                    linear_solves: snapshot.linear_solves,
                    backend_factorizations: snapshot.factorizations,
                    faer_numeric_factorizations: snapshot.sparse_numeric_factorizations,
                    symbolic_analyses: snapshot.sparse_symbolic_analyses,
                    symbolic_reuses: snapshot.sparse_symbolic_reuses,
                    max_exact_error: format!("{max_exact_error:.3e}"),
                    status: if strategies[policy_index].1 == BvpSciSparseFactorization::SuperLu
                        && snapshot.superlu_fallbacks > 0
                    {
                        "fallback-not-superlu".into()
                    } else if snapshot.factorizations == reference.factorizations
                        && snapshot.linear_solves == reference.linear_solves
                    {
                        "ok".into()
                    } else {
                        "counter-mismatch".into()
                    },
                });
            }
        }
    }

    assert!(!rows.is_empty(), "workload selection must produce A/B rows");
    crate::Utils::test_reporting::capture_test_block(format!(
        "# Sparse factorization policies on matched BVP workloads\n\nAtomViewNative full solves on fixed meshes; the only changed setting is the Sparse LU strategy. Standard, reusable AUTO, and forced simplicial are interleaved for `{repeats}` repetitions. Preparation, solve wall time, and cumulative factorization scope are reported separately; factor scopes are diagnostic, not additive. Banded-vs-Sparse layout evidence remains in `lambdify_layout_scale_matrix`.\n\n{}",
        Table::new(&rows)
    ));
    assert!(
        rows.iter().all(|row| row.status == "ok"),
        "backend comparison contains fallback or counter mismatch; inspect the captured table"
    );
}

// Experimental SuperLU promotion evidence on the official matched workload.
// Keep it out of stable correctness gates until portability and native-memory
// suitability are established.
pub(super) fn run_sparse_kernel_official_60_state_continuation_ab_is_compact() {
    #[derive(Tabled)]
    struct Row {
        policy: String,
        effective_backend: String,
        superlu_fallbacks: u64,
        superlu_stage_ms: String,
        superlu_factor_fill: String,
        repeats: usize,
        continuation_rebinds: usize,
        initial_solve_median_mad_ms: String,
        continuation_median_mad_ms: String,
        full_series_median_mad_ms: String,
        factor_median_mad_ms: String,
        backend_factorizations: u64,
        faer_symbolic_analyses: u64,
        faer_symbolic_reuses: u64,
        parameter_rebinds: u64,
        continuation_solves: u64,
        final_nodes: usize,
        final_system_dimension: usize,
        max_mesh_diff: String,
        max_state_diff: String,
        status: String,
    }

    struct Sample {
        initial_solve_ms: f64,
        continuation_ms: f64,
        full_series_ms: f64,
        factor_ms: f64,
        snapshot: BvpSciTelemetrySnapshot,
        trajectory: Vec<(Vec<f64>, Vec<f64>)>,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "sparse_kernel_official_60_state_continuation_ab",
    );
    let repeats = std::env::var("BVP_SCI_60_STATE_AB_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let continuation_rebinds = std::env::var("BVP_SCI_60_STATE_AB_CONTINUATION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
        .clamp(1, 4);
    let parameter_series = [[0.55, 0.45], [0.60, 0.40], [0.45, 0.55], [0.50, 0.50]];
    let strategies = vec![
        ("standard-faer", BvpSciSparseFactorization::StandardFaer),
        (
            "reusable-faer-auto",
            BvpSciSparseFactorization::ReusableFaerExperimental,
        ),
        (
            "reusable-faer-simplicial",
            BvpSciSparseFactorization::ReusableFaerSimplicialExperimental,
        ),
    ];
    #[cfg(feature = "superlu")]
    let strategies = {
        let mut strategies = strategies;
        strategies.push(("superlu", BvpSciSparseFactorization::SuperLu));
        strategies
    };
    let mut samples: Vec<Vec<Sample>> = (0..strategies.len())
        .map(|_| Vec::with_capacity(repeats))
        .collect();
    let mut paired_total_deltas = vec![Vec::with_capacity(repeats); strategies.len()];
    let mut max_mesh_diffs = vec![0.0_f64; strategies.len()];
    let mut max_state_diffs = vec![0.0_f64; strategies.len()];

    for repetition in 0..repeats {
        for offset in 0..strategies.len() {
            let policy_index = (offset + repetition) % strategies.len();
            let (_, strategy) = strategies[policy_index];
            let mut solver = build_official_large_parameterized_sparse_solver(strategy);

            let initial_started = Instant::now();
            let solution = solver.solve().unwrap_or_else(|error| {
                panic!("official 60-state initial solve failed: {error:?}")
            });
            assert_eq!(solution.status.code(), 0, "initial 60-state solve status");
            assert!(solution.residual_norm.is_finite());
            let mut trajectory = vec![(solution.x.clone(), solution.y.clone())];
            let initial_solve_ms = initial_started.elapsed().as_secs_f64() * 1e3;

            let continuation_started = Instant::now();
            for parameters in parameter_series.iter().take(continuation_rebinds) {
                solver
                    .set_parameters(parameters.to_vec())
                    .expect("valid 60-state parameter rebind");
                let continued = solver.solve().unwrap_or_else(|error| {
                    panic!("official 60-state continuation solve failed: {error:?}")
                });
                assert_eq!(continued.status.code(), 0, "continuation solve status");
                assert!(continued.residual_norm.is_finite());
                trajectory.push((continued.x.clone(), continued.y.clone()));
            }
            let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
            let snapshot = solver.plan().telemetry_snapshot();
            assert!(snapshot.factorizations > 0);
            if !matches!(strategy, BvpSciSparseFactorization::SuperLu) {
                assert!(snapshot.sparse_numeric_factorizations > 0);
                assert!(snapshot.sparse_symbolic_analyses > 0);
            }
            assert_eq!(snapshot.parameter_rebinds, continuation_rebinds as u64);
            assert_eq!(snapshot.continuation_solves, continuation_rebinds as u64);
            if !matches!(strategy, BvpSciSparseFactorization::SuperLu) {
                assert!(
                    snapshot.sparse_symbolic_reuses > 0,
                    "faer continuation must exercise same-pattern symbolic reuse"
                );
            }
            let factor_ms = sparse_factor_timing_ms(&snapshot, strategy)
                .expect("selected backend factorization timing should be present");
            samples[policy_index].push(Sample {
                initial_solve_ms,
                continuation_ms,
                full_series_ms: initial_solve_ms + continuation_ms,
                factor_ms,
                snapshot,
                trajectory,
            });
        }

        let reference = &samples[0]
            .last()
            .expect("StandardFaer should run once per repetition")
            .trajectory;
        for policy_index in 1..strategies.len() {
            let candidate = &samples[policy_index]
                .last()
                .expect("each strategy should run once per repetition")
                .trajectory;
            assert_eq!(candidate.len(), reference.len(), "trajectory solve count");
            for (segment, ((candidate_x, candidate_y), (reference_x, reference_y))) in
                candidate.iter().zip(reference).enumerate()
            {
                assert_eq!(
                    candidate_x.len(),
                    reference_x.len(),
                    "segment {segment} mesh size"
                );
                assert_eq!(
                    candidate_y.len(),
                    reference_y.len(),
                    "segment {segment} state size"
                );
                let mesh_diff = candidate_x
                    .iter()
                    .zip(reference_x)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                let state_diff = candidate_y
                    .iter()
                    .zip(reference_y)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    mesh_diff < 1e-12,
                    "segment {segment} mesh drift={mesh_diff:e}"
                );
                assert!(
                    state_diff < 1e-7,
                    "segment {segment} state drift={state_diff:e}"
                );
                max_mesh_diffs[policy_index] = max_mesh_diffs[policy_index].max(mesh_diff);
                max_state_diffs[policy_index] = max_state_diffs[policy_index].max(state_diff);
            }

            let reference_sample = samples[0].last().unwrap();
            let candidate_sample = samples[policy_index].last().unwrap();
            for (counter_name, candidate_count, reference_count) in [
                (
                    "Newton scopes",
                    candidate_sample.snapshot.newton_scope_calls,
                    reference_sample.snapshot.newton_scope_calls,
                ),
                (
                    "Jacobian evaluations",
                    candidate_sample.snapshot.jacobian_evaluations,
                    reference_sample.snapshot.jacobian_evaluations,
                ),
                (
                    "backend factorizations",
                    candidate_sample.snapshot.factorizations,
                    reference_sample.snapshot.factorizations,
                ),
                (
                    "linear solves",
                    candidate_sample.snapshot.linear_solves,
                    reference_sample.snapshot.linear_solves,
                ),
            ] {
                assert_eq!(
                    candidate_count, reference_count,
                    "{} {counter_name} differ",
                    strategies[policy_index].0
                );
            }
            if !matches!(
                strategies[policy_index].1,
                BvpSciSparseFactorization::SuperLu
            ) {
                for (counter_name, candidate_count, reference_count) in [
                    (
                        "faer symbolic analyses",
                        candidate_sample.snapshot.sparse_symbolic_analyses,
                        reference_sample.snapshot.sparse_symbolic_analyses,
                    ),
                    (
                        "faer symbolic reuses",
                        candidate_sample.snapshot.sparse_symbolic_reuses,
                        reference_sample.snapshot.sparse_symbolic_reuses,
                    ),
                ] {
                    assert_eq!(
                        candidate_count, reference_count,
                        "{} {counter_name} differ",
                        strategies[policy_index].0
                    );
                }
            }
            let reference_total = reference_sample.full_series_ms;
            paired_total_deltas[policy_index].push(
                100.0 * (candidate_sample.full_series_ms - reference_total) / reference_total,
            );
        }
    }

    let mut rows = Vec::with_capacity(strategies.len());
    for (policy_index, (name, _)) in strategies.iter().enumerate() {
        let policy_samples = &samples[policy_index];
        let (initial_median, initial_mad) = median_and_mad(
            &policy_samples
                .iter()
                .map(|sample| sample.initial_solve_ms)
                .collect::<Vec<_>>(),
        );
        let (continuation_median, continuation_mad) = median_and_mad(
            &policy_samples
                .iter()
                .map(|sample| sample.continuation_ms)
                .collect::<Vec<_>>(),
        );
        let (series_median, series_mad) = median_and_mad(
            &policy_samples
                .iter()
                .map(|sample| sample.full_series_ms)
                .collect::<Vec<_>>(),
        );
        let (factor_median, factor_mad) = median_and_mad(
            &policy_samples
                .iter()
                .map(|sample| sample.factor_ms)
                .collect::<Vec<_>>(),
        );
        let snapshot = &policy_samples.last().unwrap().snapshot;
        let paired_delta = if policy_index == 0 {
            0.0
        } else {
            median_and_mad(&paired_total_deltas[policy_index]).0
        };
        rows.push(Row {
            policy: (*name).into(),
            effective_backend: sparse_policy_effective_backend(
                snapshot,
                strategies[policy_index].1,
            )
            .into(),
            superlu_fallbacks: snapshot.superlu_fallbacks,
            superlu_stage_ms: superlu_stage_summary(snapshot),
            superlu_factor_fill: superlu_factor_fill_summary(snapshot),
            repeats,
            continuation_rebinds,
            initial_solve_median_mad_ms: format!("{initial_median:.3}/{initial_mad:.3}"),
            continuation_median_mad_ms: format!("{continuation_median:.3}/{continuation_mad:.3}"),
            full_series_median_mad_ms: format!(
                "{series_median:.3}/{series_mad:.3} ({paired_delta:+.2}%)"
            ),
            factor_median_mad_ms: format!("{factor_median:.3}/{factor_mad:.3}"),
            backend_factorizations: snapshot.factorizations,
            faer_symbolic_analyses: snapshot.sparse_symbolic_analyses,
            faer_symbolic_reuses: snapshot.sparse_symbolic_reuses,
            parameter_rebinds: snapshot.parameter_rebinds,
            continuation_solves: snapshot.continuation_solves,
            final_nodes: policy_samples
                .last()
                .unwrap()
                .trajectory
                .last()
                .unwrap()
                .0
                .len(),
            final_system_dimension: 60
                * policy_samples
                    .last()
                    .unwrap()
                    .trajectory
                    .last()
                    .unwrap()
                    .0
                    .len()
                + 2,
            max_mesh_diff: format!("{:.3e}", max_mesh_diffs[policy_index]),
            max_state_diff: format!("{:.3e}", max_state_diffs[policy_index]),
            status: if strategies[policy_index].1 == BvpSciSparseFactorization::SuperLu
                && snapshot.superlu_fallbacks > 0
            {
                "fallback-not-superlu".into()
            } else {
                "ok".into()
            },
        });
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# Official 60-state Sparse LU policy A/B\n\nSame 60-state/two-parameter equations and boundary conditions as the SciPy parity fixture, using the production ExprLegacy frontend and Sparse layout. Each sample includes an initial full solve followed by `{continuation_rebinds}` prepared parameter rebinds. Policies are interleaved; all trajectory meshes, states, and common Newton/Jacobian/factor/solve counters are checked against StandardFaer. Faer symbolic counters are reported separately because SuperLU has a different lifecycle. Times are median/MAD; factorization scopes overlap the solve wall clock and are not additive.\n\n{}",
        Table::new(&rows)
    ));
    assert!(
        rows.iter().all(|row| row.status == "ok"),
        "official backend comparison contains a SuperLU fallback; inspect the captured table"
    );
}

#[test]
fn lambdify_parameter_rebind_rebuilds_numeric_factorization_but_reuses_workspace() {
    #[derive(Tabled)]
    struct ContinuationContractRow {
        frontend: String,
        layout: String,
        initial_factorizations: u64,
        continuation_factorizations: u64,
        atom_conversions: u64,
        parameter_rebinds: u64,
        continuation_solves: u64,
        workspace_resizes: u64,
        logical_allocations: u64,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_continuation_numeric_factorization_contract",
    );
    let (mut solver, target) = build_solver(
        BvpSciAssembly::AtomViewNative,
        BvpSciMatrixLayout::Sparse,
        1.0,
        BvpSciExecutionPolicy::Sequential,
    );
    solver
        .solve()
        .expect("initial continuation contract solve should work");
    let before = solver.plan().telemetry_snapshot();
    target.store(2.0f64.to_bits(), Ordering::Relaxed);
    solver
        .set_parameters(vec![2.0])
        .expect("parameter rebind should work");
    solver
        .solve()
        .expect("continued continuation contract solve should work");
    let after = solver.plan().telemetry_snapshot();
    let continuation_factorizations = after.factorizations.saturating_sub(before.factorizations);

    // A parameter rebind changes the numeric Jacobian, so a new factorization
    // is required. The prepared symbolic plan and reusable workspace remain
    // valid and must not be rebuilt for this lifecycle transition.
    assert!(continuation_factorizations > 0);
    assert_eq!(after.atom_conversions, before.atom_conversions);
    assert_eq!(after.pattern_entries, before.pattern_entries);
    assert_eq!(after.parameter_rebinds, before.parameter_rebinds + 1);
    assert_eq!(after.continuation_solves, before.continuation_solves + 1);
    // Refinement may grow numeric storage; that is legitimate workspace
    // reuse, not symbolic re-preparation. The lifecycle must never shrink or
    // reset these monotonic diagnostic counters.
    assert!(after.workspace_resizes >= before.workspace_resizes);
    assert!(after.allocations >= before.allocations);
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci numeric continuation contract\n\n{}",
        Table::new(vec![ContinuationContractRow {
            frontend: "AtomViewNative".into(),
            layout: "Sparse".into(),
            initial_factorizations: before.factorizations,
            continuation_factorizations,
            atom_conversions: after.atom_conversions,
            parameter_rebinds: after.parameter_rebinds,
            continuation_solves: after.continuation_solves,
            workspace_resizes: after.workspace_resizes,
            logical_allocations: after.allocations,
            status: "rebind-reuses-prepared-workspace".into(),
        }])
    ));
}

#[test]
fn lambdify_restart_with_new_initial_state_preserves_prepared_frontend() {
    #[derive(Tabled)]
    struct RestartRow {
        frontend: String,
        layout: String,
        restarts: u64,
        atom_conversions: u64,
        pattern_entries: u64,
        workspace_resizes: u64,
        logical_allocations: u64,
        residual_norm: String,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_restart_frontend_matrix",
    );
    let mut rows = Vec::new();
    for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
        for layout in [
            BvpSciMatrixLayout::Dense,
            BvpSciMatrixLayout::Sparse,
            BvpSciMatrixLayout::Banded {
                lower: 256,
                upper: 256,
            },
        ] {
            let (mut solver, _) =
                build_solver(assembly, layout, 1.0, BvpSciExecutionPolicy::Sequential);
            solver.solve().expect("initial restart solve should work");
            let before = solver.plan().telemetry_snapshot();
            solver
                .restart(vec![0.0, 0.5, 1.0], vec![0.2, 0.55, 0.9])
                .expect("restart with a new initial state should work");
            let solution = solver.solve().expect("restarted solve should work");
            let after = solver.plan().telemetry_snapshot();
            assert_eq!(after.restarts, before.restarts + 1);
            assert_eq!(after.atom_conversions, before.atom_conversions);
            assert_eq!(after.pattern_entries, before.pattern_entries);
            assert!(after.workspace_resizes >= before.workspace_resizes);
            assert!(after.allocations >= before.allocations);
            assert!(solution.y.iter().all(|value| value.is_finite()));
            rows.push(RestartRow {
                frontend: format!("{assembly:?}"),
                layout: layout_name(layout),
                restarts: after.restarts,
                atom_conversions: after.atom_conversions,
                pattern_entries: after.pattern_entries,
                workspace_resizes: after.workspace_resizes,
                logical_allocations: after.allocations,
                residual_norm: format!("{:.3e}", solution.residual_norm),
                status: format!("{}", solution.status.code()),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci restart with new initial state\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn lambdify_symbolic_parameter_jacobian_matches_exact_derivative() {
    #[derive(Tabled)]
    struct ParameterJacobianRow {
        frontend: String,
        parameter_jacobian_nnz: usize,
        symbolic_derivatives: u64,
        callback_calls: u64,
        fd_probes: u64,
        max_value_error: String,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_symbolic_parameter_jacobian",
    );
    let mut rows = Vec::new();
    for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
        let telemetry = BvpSciTelemetry::detailed();
        let plan = BvpSciLambdifyPlan::prepare(
            assembly,
            &[
                Expr::parse_expression("p0*y0 + p1"),
                Expr::parse_expression("p1*y0^2"),
            ],
            &["y0".into(), "y1".into()],
            &["p0".into(), "p1".into()],
            "x",
            telemetry,
        )
        .expect("symbolic df/dp plan should prepare");
        let mut arguments = vec![0.0; 5];
        let mut actual = vec![0.0; 4];
        assert!(
            plan.evaluate_parameter_jacobian(
                0.25,
                &[3.0, 7.0],
                &[2.0, 5.0],
                &mut arguments,
                &mut actual,
            )
            .expect("symbolic df/dp should evaluate")
        );
        let expected = [3.0, 1.0, 0.0, 9.0];
        let max_value_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_value_error <= 1e-14, "{assembly:?}: {actual:?}");

        let snapshot = plan.telemetry_snapshot();
        assert_eq!(snapshot.finite_difference_probes, 0);
        assert_eq!(snapshot.parameter_jacobian_calls, 1);
        assert_eq!(plan.parameter_jacobian_nnz(), 3);
        rows.push(ParameterJacobianRow {
            frontend: format!("{assembly:?}"),
            parameter_jacobian_nnz: plan.parameter_jacobian_nnz(),
            symbolic_derivatives: snapshot.symbolic_jacobian_derivations,
            callback_calls: snapshot.parameter_jacobian_calls,
            fd_probes: snapshot.finite_difference_probes,
            max_value_error: format!("{max_value_error:.3e}"),
            status: "ok".into(),
        });
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci Lambdify symbolic parameter Jacobian\n\nExpected row-major `df/dp = [[y0, 1], [0, y0^2]]`; derivative compilation happens during preparation, with no runtime finite-difference probes.\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn lambdify_symbolic_parameter_and_fd_boundary_jacobians_have_telemetry() {
    #[derive(Tabled)]
    struct JacobianContractRow {
        frontend: String,
        layout: String,
        symbolic_jacobian_derivations: u64,
        jacobian_evaluations: u64,
        parameter_jacobian_calls: u64,
        finite_difference_probes: u64,
        jacobian_output_assemblies: u64,
        residual_evaluations: u64,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_symbolic_parameter_fd_boundary_jacobians",
    );
    let mut rows = Vec::new();
    for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
        for layout in [
            BvpSciMatrixLayout::Dense,
            BvpSciMatrixLayout::Sparse,
            BvpSciMatrixLayout::Banded {
                lower: 256,
                upper: 256,
            },
        ] {
            let (mut solver, _) =
                build_solver(assembly, layout, 1.0, BvpSciExecutionPolicy::Sequential);
            let solution = solver.solve().expect("Jacobian contract solve should work");
            let telemetry = solver.plan().telemetry_snapshot();
            assert!(telemetry.symbolic_jacobian_derivations > 0);
            assert!(telemetry.jacobian_evaluations > 0);
            assert!(telemetry.parameter_jacobian_calls > 0);
            assert!(telemetry.finite_difference_probes > 0);
            assert!(telemetry.jacobian_output_assemblies > 0);
            assert!(telemetry.residual_evaluations > 0);
            rows.push(JacobianContractRow {
                frontend: format!("{assembly:?}"),
                layout: layout_name(layout),
                symbolic_jacobian_derivations: telemetry.symbolic_jacobian_derivations,
                jacobian_evaluations: telemetry.jacobian_evaluations,
                parameter_jacobian_calls: telemetry.parameter_jacobian_calls,
                finite_difference_probes: telemetry.finite_difference_probes,
                jacobian_output_assemblies: telemetry.jacobian_output_assemblies,
                residual_evaluations: telemetry.residual_evaluations,
                status: format!(
                    "{};residual={:.3e}",
                    solution.status.code(),
                    solution.residual_norm
                ),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci Jacobian contract\n\nRHS `df/dp` uses prepared symbolic Lambdify evaluators; the nonzero finite-difference probe count in these rows comes from the numerical boundary-Jacobian fallback.\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn lambdify_failure_and_exhaustion_paths_are_typed_and_compact() {
    #[derive(Tabled)]
    struct FailureRow {
        case_name: String,
        error: String,
        status_code: String,
        typed: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_failure_and_exhaustion_paths",
    );
    let mut rows = Vec::new();
    let (mut solver, _) = build_solver(
        BvpSciAssembly::ExprLegacy,
        BvpSciMatrixLayout::Dense,
        1.0,
        BvpSciExecutionPolicy::Sequential,
    );
    let parameter_error = solver
        .set_parameters(vec![f64::NAN])
        .expect_err("non-finite continuation parameter must be rejected");
    assert!(matches!(
        parameter_error,
        BvpSciNewError::NonFinite {
            stage: BvpSciStage::Continuation
        }
    ));
    rows.push(FailureRow {
        case_name: "non-finite-parameter".into(),
        error: parameter_error.to_string(),
        status_code: "-".into(),
        typed: "NonFinite/Continuation".into(),
    });

    let (mut solver, _) = build_solver(
        BvpSciAssembly::ExprLegacy,
        BvpSciMatrixLayout::Dense,
        1.0,
        BvpSciExecutionPolicy::Sequential,
    );
    let shape_error = solver
        .restart(vec![0.0, 0.5, 1.0], vec![0.0, 1.0])
        .expect_err("restart shape mismatch must be rejected");
    assert!(matches!(
        shape_error,
        BvpSciNewError::ShapeMismatch {
            stage: BvpSciStage::Continuation,
            ..
        }
    ));
    rows.push(FailureRow {
        case_name: "restart-shape".into(),
        error: shape_error.to_string(),
        status_code: "-".into(),
        typed: "ShapeMismatch/Continuation".into(),
    });

    let (mut solver, _) = build_solver(
        BvpSciAssembly::ExprLegacy,
        BvpSciMatrixLayout::Dense,
        1.0,
        BvpSciExecutionPolicy::Sequential,
    );
    let mesh_error = solver
        .restart(vec![0.0, 1.0, 0.5], vec![0.0, 0.5, 1.0])
        .expect_err("non-increasing restart mesh must be rejected");
    assert!(matches!(
        mesh_error,
        BvpSciNewError::InvalidConfiguration(_)
    ));
    rows.push(FailureRow {
        case_name: "restart-mesh-order".into(),
        error: mesh_error.to_string(),
        status_code: "-".into(),
        typed: "InvalidConfiguration".into(),
    });

    let callback_plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("1")],
        &["y".into()],
        &[],
        "x",
        BvpSciTelemetry::disabled(),
    )
    .expect("callback failure fixture should prepare");
    let failing_boundary = BvpSciBoundaryCallbacks::new(
        1,
        |_ya, _yb, _parameters, _output| Err("synthetic boundary failure".into()),
        BvpSciTelemetry::disabled(),
    );
    let mut callback_options = BvpSciOptions::default();
    callback_options.matrix_layout = BvpSciMatrixLayout::Dense;
    callback_options.max_nodes = 2;
    let mut callback_solver = BvpSciSolver::new(
        callback_plan,
        failing_boundary,
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![],
        callback_options,
    )
    .expect("callback failure fixture should construct");
    let callback_error = callback_solver
        .solve()
        .expect_err("boundary callback failure must be returned as an error");
    assert!(matches!(
        callback_error,
        BvpSciNewError::Callback {
            stage: BvpSciStage::BoundaryCallback,
            ..
        }
    ));
    rows.push(FailureRow {
        case_name: "boundary-callback".into(),
        error: callback_error.to_string(),
        status_code: "-".into(),
        typed: "Callback/BoundaryCallback".into(),
    });

    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &bratu_equations(),
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::disabled(),
    )
    .expect("exhaustion fixture should prepare");
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mut options = BvpSciOptions::default();
    options.matrix_layout = BvpSciMatrixLayout::Dense;
    options.tolerance = 1e-12;
    options.max_nodes = 16;
    options.max_mesh_refinements = 0;
    options.max_newton_iterations = 30;
    options.max_jacobian_refreshes = 12;
    let mut exhausted = BvpSciSolver::new(
        plan,
        boundary,
        (0..16).map(|index| index as f64 / 15.0).collect(),
        vec![0.0; 32],
        vec![],
        options,
    )
    .expect("exhaustion fixture should construct");
    let exhaustion_error = exhausted
        .solve()
        .expect_err("zero-refinement Bratu gate must expose exhaustion");
    assert!(matches!(
        exhaustion_error,
        BvpSciNewError::MeshRefinementLimit { .. } | BvpSciNewError::MaxNodesExceeded { .. }
    ));
    rows.push(FailureRow {
        case_name: "mesh-exhaustion".into(),
        error: exhaustion_error.to_string(),
        status_code: exhaustion_error
            .status()
            .map(|status| status.code().to_string())
            .unwrap_or_else(|| "-".into()),
        typed: "MaxNodes/mesh-budget".into(),
    });
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci typed failure and exhaustion paths\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn nonlinear_bratu_workload_preserves_frontend_backend_fidelity() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_nonlinear_workload_matrix",
    );
    let mut reference = build_bratu_solver(BvpSciAssembly::ExprLegacy, BvpSciMatrixLayout::Dense);
    let reference_solution = reference.solve().expect("Bratu reference should converge");
    let mut rows = Vec::new();
    for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
        for layout in [
            BvpSciMatrixLayout::Dense,
            BvpSciMatrixLayout::Sparse,
            BvpSciMatrixLayout::Banded {
                lower: 256,
                upper: 256,
            },
        ] {
            let started = Instant::now();
            let mut solver = build_bratu_solver(assembly, layout);
            let solution = solver.solve().expect("Bratu matrix row should converge");
            let max_diff = solution
                .y
                .iter()
                .zip(&reference_solution.y)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            assert!(max_diff < 1e-5, "Bratu frontend/layout drift={max_diff}");
            rows.push(WorkloadRow {
                workload: "bratu-like".into(),
                frontend: format!("{assembly:?}"),
                layout: layout_name(layout),
                solve_ms: format!("{:.3}", started.elapsed().as_secs_f64() * 1e3),
                residual_norm: format!("{:.3e}", solution.residual_norm),
                max_fidelity_diff: format!("{max_diff:.3e}"),
                jacobian_calls: solver.plan().telemetry_snapshot().jacobian_evaluations,
                factorizations: solver.plan().telemetry_snapshot().factorizations,
                status: format!("{}", solution.status.code()),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci nonlinear workload matrix\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn lane_emden_exact_solution_preserves_frontend_backend_fidelity() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_lane_emden_exact_matrix",
    );
    let left = 1.0e-3;
    let right = 1.0;
    let mut rows = Vec::new();
    for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
        for layout in [
            BvpSciMatrixLayout::Dense,
            BvpSciMatrixLayout::Sparse,
            BvpSciMatrixLayout::Banded {
                lower: 256,
                upper: 256,
            },
        ] {
            let started = Instant::now();
            let mut solver = build_lane_emden_solver(assembly, layout);
            let solution = solver.solve().expect("Lane-Emden row should converge");
            let max_error = solution
                .x
                .iter()
                .enumerate()
                .map(|(node, &x)| {
                    let (expected_y, expected_z) = lane_emden_exact(x);
                    let state = &solution.y[node * 2..node * 2 + 2];
                    (state[0] - expected_y)
                        .abs()
                        .max((state[1] - expected_z).abs())
                })
                .fold(0.0, f64::max);
            assert!(max_error < 5e-4, "Lane-Emden fidelity drift={max_error}");
            assert_eq!(solution.status, BvpSciStatus::Success);
            let dense = solution
                .dense_output()
                .expect("Lane-Emden dense output should be retained");
            let midpoint = (left + right) * 0.5;
            let sampled = dense
                .evaluate(midpoint)
                .expect("Lane-Emden dense output query should work");
            let expected = lane_emden_exact(midpoint).0;
            assert!((sampled[0] - expected).abs() < 2e-3);
            let telemetry = solver.plan().telemetry_snapshot();
            rows.push(WorkloadRow {
                workload: "lane-emden-5".into(),
                frontend: format!("{assembly:?}"),
                layout: layout_name(layout),
                solve_ms: format!("{:.3}", started.elapsed().as_secs_f64() * 1e3),
                residual_norm: format!("{:.3e}", solution.residual_norm),
                max_fidelity_diff: format!("{max_error:.3e}"),
                jacobian_calls: telemetry.jacobian_evaluations,
                factorizations: telemetry.factorizations,
                status: format!("{}", solution.status.code()),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci Lane-Emden exact workload\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn bvp_damp_exact_models_preserve_frontend_fidelity() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "bvp_damp_exact_models_frontend_fidelity",
    );
    let mut rows = Vec::new();
    for workload in [
        DampExactWorkload::Linear,
        DampExactWorkload::Oscillator,
        DampExactWorkload::StiffDecay,
        DampExactWorkload::TwoPoint,
        DampExactWorkload::Clairaut,
        DampExactWorkload::Parachute,
        DampExactWorkload::StiffCoupled,
    ] {
        let mut reference = build_damp_exact_solver(
            workload,
            BvpSciAssembly::ExprLegacy,
            BvpSciMatrixLayout::Dense,
        );
        let reference_solution = reference.solve().unwrap_or_else(|error| {
            panic!(
                "Damped exact Dense reference should converge for {}: {error:?}",
                workload.label()
            )
        });
        for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
            for layout in [
                BvpSciMatrixLayout::Dense,
                BvpSciMatrixLayout::Sparse,
                BvpSciMatrixLayout::Banded {
                    lower: 256,
                    upper: 256,
                },
            ] {
                let started = Instant::now();
                let mut solver = build_damp_exact_solver(workload, assembly, layout);
                let solution = solver.solve().unwrap_or_else(|error| {
                    let frontend = match assembly {
                        BvpSciAssembly::Numerical => "Numerical",
                        BvpSciAssembly::ExprLegacy => "ExprLegacy",
                        BvpSciAssembly::AtomViewNative => "AtomViewNative",
                    };
                    panic!(
                        "Damped exact frontend/layout should converge: workload={} frontend={} layout={} error={error:?}",
                        workload.label(),
                        frontend,
                        layout_name(layout),
                    )
                });
                let state_dimension = workload.state_names().len();
                let solution_y = &solution.y;
                let exact_error = solution
                    .x
                    .iter()
                    .enumerate()
                    .flat_map(|(node, &x)| {
                        let expected = workload.exact_state(x);
                        expected
                            .into_iter()
                            .enumerate()
                            .map(move |(component, expected)| {
                                (solution_y[node * state_dimension + component] - expected).abs()
                            })
                    })
                    .fold(0.0, f64::max);
                let parity_diff = solution
                    .y
                    .iter()
                    .zip(&reference_solution.y)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max);
                assert!(
                    exact_error < 5e-3,
                    "{} exact error={exact_error:e}",
                    workload.label()
                );
                assert!(
                    parity_diff < 5e-3,
                    "{} frontend/layout drift={parity_diff:e}",
                    workload.label()
                );
                let telemetry = solver.plan().telemetry_snapshot();
                rows.push(WorkloadRow {
                    workload: workload.label().into(),
                    frontend: format!("{assembly:?}"),
                    layout: layout_name(layout),
                    solve_ms: format!("{:.3}", started.elapsed().as_secs_f64() * 1e3),
                    residual_norm: format!("{:.3e}", solution.residual_norm),
                    max_fidelity_diff: format!("{exact_error:.3e}"),
                    jacobian_calls: telemetry.jacobian_evaluations,
                    factorizations: telemetry.factorizations,
                    status: format!("{}", solution.status.code()),
                });
            }
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_Damp exact-model gates\n\n{}",
        Table::new(rows)
    ));
}

fn run_combustion_cross_solver_row(nodes: usize) -> CrossSolverRow {
    let state_names = ["Teta", "q", "C0", "J0", "C1", "J1"];
    let x: Vec<f64> = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect();
    let y = x
        .iter()
        .flat_map(|_| [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        .collect::<Vec<_>>();
    let boundary = BvpSciBoundaryCallbacks::new(
        state_names.len(),
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[1];
            output[2] = ya[2] - 1.0;
            output[3] = yb[3];
            output[4] = ya[4];
            output[5] = yb[5];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &combustion_like_equations(),
        &state_names
            .iter()
            .map(|name| (*name).into())
            .collect::<Vec<_>>(),
        &[],
        "x",
        BvpSciTelemetry::disabled(),
    )
    .expect("cross-solver combustion frontend should prepare");
    let mut options = BvpSciOptions::default();
    options.matrix_layout = BvpSciMatrixLayout::Dense;
    options.tolerance = 1e-5;
    options.max_nodes = nodes;
    options.max_mesh_refinements = 0;
    options.max_newton_iterations = 40;
    options.max_jacobian_refreshes = 12;
    let mut bvp_sci = BvpSciSolver::new(plan, boundary, x, y, vec![], options)
        .expect("cross-solver BVP_sci solver should construct");
    let sci_solution = bvp_sci
        .solve()
        .unwrap_or_else(|error| panic!("BVP_sci combustion nodes={nodes} failed: {error:?}"));

    let mut bvp_damp = build_bvp_damp_combustion(nodes - 1);
    bvp_damp
        .try_solve()
        .unwrap_or_else(|error| panic!("BVP_Damp combustion nodes={nodes} failed: {error:?}"));
    let damp_solution = bvp_damp
        .get_result()
        .expect("BVP_Damp should publish a solution");
    let damp_mesh: Vec<f64> = bvp_damp.x_mesh.iter().copied().collect();
    assert_eq!(damp_solution.nrows(), damp_mesh.len());
    assert_eq!(damp_solution.ncols(), state_names.len());
    assert!(
        sci_solution.y.iter().all(|value| value.is_finite())
            && damp_solution.iter().all(|value| value.is_finite()),
        "both nonlinear reference solutions must remain finite"
    );
    let control_points = 17;
    let max_interpolated_diff = normalized_cross_solver_diff(
        &sci_solution.x,
        &sci_solution.y,
        &damp_mesh,
        &damp_solution,
        6,
        control_points,
    );
    assert!(
        max_interpolated_diff < 5e-2,
        "BVP_sci/BVP_Damp nonlinear trajectory drift at nodes={nodes}: {max_interpolated_diff:.3e}"
    );
    CrossSolverRow {
        workload: format!("combustion-like-{nodes}"),
        control_points,
        bvp_sci_nodes: sci_solution.x.len(),
        bvp_damp_nodes: damp_mesh.len(),
        bvp_sci_residual: format!("{:.3e}", sci_solution.residual_norm),
        max_interpolated_diff: format!("{max_interpolated_diff:.3e}"),
        status: "ok".into(),
    }
}

#[test]
fn nonlinear_workloads_match_bvp_damp_on_a_common_control_mesh() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "cross_solver_nonlinear_correctness_gates",
    );
    let rows = vec![run_combustion_cross_solver_row(25)];
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci versus BVP_Damp nonlinear correctness\n\n{}",
        Table::new(rows)
    ));
}

#[test]
#[ignore = "medium/large BVP_sci versus BVP_Damp combustion release gate"]
fn medium_large_combustion_matches_bvp_damp_on_interpolated_trajectory() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "cross_solver_medium_large_combustion_correctness",
    );
    let nodes = std::env::var("BVP_SCI_STORY_CROSS_SOLVER_NODES")
        .unwrap_or_else(|_| "64,128".into())
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .collect::<Vec<_>>();
    let rows = nodes
        .into_iter()
        .map(run_combustion_cross_solver_row)
        .collect::<Vec<_>>();
    assert!(
        !rows.is_empty(),
        "cross-solver matrix selected no node counts"
    );
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci versus BVP_Damp medium/large combustion\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn bratu_workload_matches_bvp_damp_on_a_common_control_mesh() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "cross_solver_bratu_correctness_gate",
    );
    let state_names = ["y", "z"];
    let nodes = 64;
    let x: Vec<f64> = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect();
    let y = vec![0.0; nodes * state_names.len()];
    let boundary = BvpSciBoundaryCallbacks::new(
        state_names.len(),
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &bratu_equations(),
        &state_names
            .iter()
            .map(|name| (*name).into())
            .collect::<Vec<_>>(),
        &[],
        "x",
        BvpSciTelemetry::disabled(),
    )
    .expect("cross-solver Bratu frontend should prepare");
    let mut options = BvpSciOptions::default();
    options.matrix_layout = BvpSciMatrixLayout::Dense;
    options.tolerance = 1e-4;
    options.max_nodes = nodes;
    options.max_mesh_refinements = 0;
    options.max_newton_iterations = 40;
    options.max_jacobian_refreshes = 12;
    let mut bvp_sci = BvpSciSolver::new(plan, boundary, x, y, vec![], options)
        .expect("cross-solver Bratu BVP_sci solver should construct");
    let sci_solution = bvp_sci.solve().expect("BVP_sci Bratu gate should converge");

    let mut bvp_damp = build_bvp_damp_bratu(nodes);
    bvp_damp
        .try_solve()
        .expect("BVP_Damp Bratu gate should converge");
    let damp_solution = bvp_damp
        .get_result()
        .expect("BVP_Damp should publish a Bratu solution");
    let damp_mesh: Vec<f64> = bvp_damp.x_mesh.iter().copied().collect();
    assert!(
        sci_solution.y.iter().all(|value| value.is_finite())
            && damp_solution.iter().all(|value| value.is_finite()),
        "both Bratu reference solutions must remain finite"
    );
    let control_points = 17;
    let max_interpolated_diff = normalized_cross_solver_diff(
        &sci_solution.x,
        &sci_solution.y,
        &damp_mesh,
        &damp_solution,
        2,
        control_points,
    );
    assert!(
        max_interpolated_diff < 5e-2,
        "BVP_sci/BVP_Damp Bratu trajectory drift={max_interpolated_diff:.3e}"
    );
    let rows = vec![CrossSolverRow {
        workload: "bratu-like".into(),
        control_points,
        bvp_sci_nodes: sci_solution.x.len(),
        bvp_damp_nodes: damp_mesh.len(),
        bvp_sci_residual: format!("{:.3e}", sci_solution.residual_norm),
        max_interpolated_diff: format!("{max_interpolated_diff:.3e}"),
        status: "ok".into(),
    }];
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci versus BVP_Damp Bratu correctness\n\n{}",
        Table::new(rows)
    ));
}

#[test]
fn lambdify_parallel_policy_preserves_solution_and_reports_dispatches() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_parallel_policy_matrix",
    );
    let policies = [
        ("sequential", BvpSciExecutionPolicy::Sequential),
        ("parallel", BvpSciExecutionPolicy::Parallel { min_work: 0 }),
        ("auto", BvpSciExecutionPolicy::Auto { min_work: 0 }),
    ];
    let mut reference = build_solver(
        BvpSciAssembly::ExprLegacy,
        BvpSciMatrixLayout::Dense,
        1.0,
        BvpSciExecutionPolicy::Sequential,
    )
    .0;
    let expected = reference
        .solve()
        .expect("sequential reference should converge");
    let mut rows = Vec::new();
    let mut sequential_ms = None;
    for (policy_name, policy) in policies {
        let started = Instant::now();
        let (mut solver, _) = build_solver(
            BvpSciAssembly::AtomViewNative,
            BvpSciMatrixLayout::Sparse,
            1.0,
            policy,
        );
        let solution = solver.solve().expect("policy row should converge");
        let drift = solution
            .y
            .iter()
            .zip(&expected.y)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        assert!(drift < 1e-5, "{policy_name} drift={drift:e}");
        let telemetry = solver.plan().telemetry_snapshot();
        if policy_name == "sequential" {
            assert_eq!(telemetry.parallel_dispatches, 0);
            assert!(telemetry.sequential_dispatches > 0);
        } else {
            assert!(telemetry.parallel_dispatches + telemetry.sequential_dispatches > 0);
        }
        let solve_ms = started.elapsed().as_secs_f64() * 1e3;
        if policy_name == "sequential" {
            sequential_ms = Some(solve_ms);
        }
        rows.push((
            policy_name,
            solve_ms,
            format!("{drift:.3e}"),
            telemetry.parallel_dispatches,
            telemetry.sequential_dispatches,
            telemetry.max_worker_threads,
        ));
    }
    #[derive(Tabled)]
    struct PolicyRow {
        policy: String,
        solve_ms: String,
        break_even_vs_sequential: String,
        max_diff: String,
        parallel_dispatches: u64,
        sequential_dispatches: u64,
        max_worker_threads: u64,
    }
    let table = rows
        .into_iter()
        .map(
            |(
                policy,
                solve_ms,
                max_diff,
                parallel_dispatches,
                sequential_dispatches,
                max_worker_threads,
            )| PolicyRow {
                policy: policy.into(),
                break_even_vs_sequential: sequential_ms
                    .filter(|baseline| *baseline > 0.0)
                    .map(|baseline| {
                        let delta = (solve_ms / baseline - 1.0) * 100.0;
                        format!("{delta:+.1}%")
                    })
                    .unwrap_or_else(|| "not-applicable".into()),
                solve_ms: format!("{solve_ms:.3}"),
                max_diff,
                parallel_dispatches,
                sequential_dispatches,
                max_worker_threads,
            },
        )
        .collect::<Vec<_>>();
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci Lambdify execution policy\n\n{}",
        Table::new(table)
    ));
}

#[test]
#[ignore = "medium/large continuation release gate"]
fn lambdify_medium_large_parameter_continuation_matrix_is_compact() {
    #[derive(Tabled)]
    struct ContinuationRow {
        nodes: usize,
        count: usize,
        frontend: String,
        layout: String,
        prepare_ms: String,
        continuation_ms: String,
        final_parameter_abs: String,
        parameter_rebinds: u64,
        continuation_solves: u64,
        full_solve_ms: String,
        linear_solve_ms: String,
        factorization_ms: String,
        banded_route: String,
        structured_factorizations: u64,
        scalar_fallback_factorizations: u64,
        structured_solves: u64,
        scalar_fallback_solves: u64,
        fallback_switches: u64,
        status: String,
    }

    fn parse_list(name: &str, default: &str, minimum: usize) -> Vec<usize> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .filter_map(|value| value.trim().parse().ok())
            .filter(|value: &usize| *value >= minimum)
            .collect()
    }

    fn build_medium_solver(
        assembly: BvpSciAssembly,
        layout: BvpSciMatrixLayout,
        nodes: usize,
        parameter: f64,
    ) -> (BvpSciSolver, Arc<AtomicU64>) {
        let plan = BvpSciLambdifyPlan::prepare(
            assembly,
            &[Expr::parse_expression("p*y + 1")],
            &["y".into()],
            &["p".into()],
            "x",
            BvpSciTelemetry::detailed(),
        )
        .expect("medium continuation frontend should prepare");
        let target = Arc::new(AtomicU64::new(parameter.to_bits()));
        let target_for_boundary = Arc::clone(&target);
        let boundary = BvpSciBoundaryCallbacks::new(
            2,
            move |ya, yb, _, output| {
                let target = f64::from_bits(target_for_boundary.load(Ordering::Relaxed));
                output[0] = ya[0];
                output[1] = yb[0] - target;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let x: Vec<f64> = (0..nodes)
            .map(|index| index as f64 / (nodes - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&x| parameter * x).collect();
        let mut options = BvpSciOptions::default();
        options.matrix_layout = layout;
        options.tolerance = 1e-5;
        options.max_nodes = nodes;
        options.max_mesh_refinements = 0;
        options.max_newton_iterations = 20;
        options.max_jacobian_refreshes = 10;
        (
            BvpSciSolver::new(plan, boundary, x, y, vec![parameter], options)
                .expect("medium continuation solver should construct"),
            target,
        )
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_medium_large_continuation_matrix",
    );
    let nodes_list = parse_list("BVP_SCI_STORY_CONTINUATION_NODES", "64,256", 4);
    let counts = parse_list("BVP_SCI_STORY_CONTINUATION_COUNTS", "1,4", 1);
    let mut rows = Vec::new();
    for nodes in nodes_list {
        for count in &counts {
            for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
                for layout in [
                    BvpSciMatrixLayout::Dense,
                    BvpSciMatrixLayout::Sparse,
                    BvpSciMatrixLayout::Banded {
                        lower: 256,
                        upper: 256,
                    },
                ] {
                    let started = Instant::now();
                    let (mut solver, target) = build_medium_solver(assembly, layout, nodes, 1.0);
                    let prepare_ms = started.elapsed().as_secs_f64() * 1e3;
                    solver
                        .solve()
                        .expect("initial medium solve should converge");
                    let continuation_started = Instant::now();
                    for step in 0..*count {
                        let parameter = 1.0 + (step + 1) as f64 * 0.25;
                        target.store(parameter.to_bits(), Ordering::Relaxed);
                        solver
                            .set_parameters(vec![parameter])
                            .expect("continuation rebind should work");
                        solver
                            .solve()
                            .expect("continued medium solve should converge");
                    }
                    let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
                    let telemetry = solver.plan().telemetry_snapshot();
                    let (_, _, parameters) = solver.state();
                    let final_parameter_abs = parameters
                        .iter()
                        .map(|value| value.abs())
                        .fold(0.0, f64::max);
                    assert!(
                        parameters.iter().all(|value| value.is_finite()),
                        "continuation produced a non-finite parameter"
                    );
                    assert_eq!(telemetry.parameter_rebinds, *count as u64);
                    assert_eq!(telemetry.continuation_solves, *count as u64);
                    rows.push(ContinuationRow {
                        nodes,
                        count: *count,
                        frontend: format!("{assembly:?}"),
                        layout: layout_name(layout),
                        prepare_ms: format!("{prepare_ms:.3}"),
                        continuation_ms: format!("{continuation_ms:.3}"),
                        final_parameter_abs: format!("{final_parameter_abs:.3e}"),
                        parameter_rebinds: telemetry.parameter_rebinds,
                        continuation_solves: telemetry.continuation_solves,
                        full_solve_ms: timing(telemetry.full_solve_ms),
                        linear_solve_ms: timing(telemetry.linear_solve_ms),
                        factorization_ms: timing(telemetry.factorization_ms),
                        banded_route: if telemetry.banded_scalar_fallback_solves > 0
                            || telemetry.banded_scalar_fallback_factorizations > 0
                        {
                            "scalar-fallback".into()
                        } else if telemetry.banded_structured_solves > 0
                            || telemetry.banded_structured_factorizations > 0
                        {
                            "structured".into()
                        } else {
                            "-".into()
                        },
                        structured_factorizations: telemetry.banded_structured_factorizations,
                        scalar_fallback_factorizations: telemetry
                            .banded_scalar_fallback_factorizations,
                        structured_solves: telemetry.banded_structured_solves,
                        scalar_fallback_solves: telemetry.banded_scalar_fallback_solves,
                        fallback_switches: telemetry.banded_fallback_switches,
                        status: "ok".into(),
                    });
                }
            }
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci Lambdify medium/large continuation\n\n{}",
        Table::new(rows)
    ));
}

#[test]
#[ignore = "fresh/prepared/warm continuation and logical retention release gate"]
fn lambdify_continuation_fresh_prepared_warm_retention_matrix_is_compact() {
    #[derive(Tabled)]
    struct ContinuationLifecycleRow {
        mode: String,
        nodes: usize,
        count: usize,
        frontend: String,
        layout: String,
        prepare_ms: String,
        total_ms: String,
        per_solve_ms: String,
        continuation_ms: String,
        factorizations: u64,
        continuation_factorizations: u64,
        sparse_symbolic_analyses: u64,
        sparse_numeric_factorizations: u64,
        jacobian_refreshes: u64,
        parameter_rebinds: u64,
        continuation_solves: u64,
        workspace_resizes: u64,
        logical_allocations: u64,
        retention: String,
        status: String,
    }

    #[derive(Clone, Copy)]
    enum Mode {
        Fresh,
        Prepared,
        Warm,
    }

    impl Mode {
        fn label(self) -> &'static str {
            match self {
                Self::Fresh => "fresh",
                Self::Prepared => "prepared",
                Self::Warm => "warm",
            }
        }
    }

    #[derive(Default)]
    struct CounterTotals {
        factorizations: u64,
        sparse_symbolic_analyses: u64,
        sparse_numeric_factorizations: u64,
        jacobian_refreshes: u64,
        parameter_rebinds: u64,
        continuation_solves: u64,
        workspace_resizes: u64,
        logical_allocations: u64,
    }

    impl CounterTotals {
        fn add(&mut self, snapshot: &BvpSciTelemetrySnapshot) {
            self.factorizations += snapshot.factorizations;
            self.sparse_symbolic_analyses += snapshot.sparse_symbolic_analyses;
            self.sparse_numeric_factorizations += snapshot.sparse_numeric_factorizations;
            self.jacobian_refreshes += snapshot.newton_jacobian_refreshes;
            self.parameter_rebinds += snapshot.parameter_rebinds;
            self.continuation_solves += snapshot.continuation_solves;
            self.workspace_resizes += snapshot.workspace_resizes;
            self.logical_allocations += snapshot.allocations;
        }
    }

    fn parse_list(name: &str, default: &str, minimum: usize) -> Vec<usize> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .filter_map(|value| value.trim().parse().ok())
            .filter(|value: &usize| *value >= minimum)
            .collect()
    }

    fn parameter_at(index: usize) -> f64 {
        1.0 + index as f64 * 0.25
    }

    fn prepare_plan(assembly: BvpSciAssembly) -> Result<BvpSciLambdifyPlan, String> {
        BvpSciLambdifyPlan::prepare(
            assembly,
            &[Expr::parse_expression("p*y + 1")],
            &["y".into()],
            &["p".into()],
            "x",
            BvpSciTelemetry::timings(),
        )
        .map_err(|error| error.to_string())
    }

    fn build_solver(
        plan: BvpSciLambdifyPlan,
        layout: BvpSciMatrixLayout,
        nodes: usize,
        parameter: f64,
    ) -> Result<(BvpSciSolver, Arc<AtomicU64>), String> {
        let target = Arc::new(AtomicU64::new(parameter.to_bits()));
        let target_for_boundary = Arc::clone(&target);
        let boundary = BvpSciBoundaryCallbacks::new(
            2,
            move |ya, yb, _, output| {
                let target = f64::from_bits(target_for_boundary.load(Ordering::Relaxed));
                output[0] = ya[0];
                output[1] = yb[0] - target;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let x: Vec<f64> = (0..nodes)
            .map(|index| index as f64 / (nodes - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&value| parameter * value).collect();
        let mut options = BvpSciOptions::default();
        options.matrix_layout = layout;
        options.tolerance = 1e-5;
        options.max_nodes = nodes;
        options.max_mesh_refinements = 0;
        options.max_newton_iterations = 20;
        options.max_jacobian_refreshes = 10;
        BvpSciSolver::new(plan, boundary, x, y, vec![parameter], options)
            .map(|solver| (solver, target))
            .map_err(|error| error.to_string())
    }

    fn run_mode(
        mode: Mode,
        assembly: BvpSciAssembly,
        layout: BvpSciMatrixLayout,
        nodes: usize,
        count: usize,
    ) -> Result<ContinuationLifecycleRow, String> {
        let total_started = Instant::now();
        let mut prepare_ms = 0.0;
        let mut continuation_ms = None;
        let mut totals = CounterTotals::default();
        let mut continuation_factorizations = 0;
        let mut retention = "not-applicable";

        match mode {
            Mode::Fresh => {
                for index in 0..count {
                    let prepare_started = Instant::now();
                    let plan = prepare_plan(assembly)?;
                    prepare_ms += prepare_started.elapsed().as_secs_f64() * 1e3;
                    let (mut solver, _) = build_solver(plan, layout, nodes, parameter_at(index))?;
                    solver.solve().map_err(|error| error.to_string())?;
                    let snapshot = solver.plan().telemetry_snapshot();
                    totals.add(&snapshot);
                }
            }
            Mode::Prepared => {
                let prepare_started = Instant::now();
                let template = prepare_plan(assembly)?;
                prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
                for index in 0..count {
                    let (mut solver, _) =
                        build_solver(template.clone(), layout, nodes, parameter_at(index))?;
                    solver.solve().map_err(|error| error.to_string())?;
                }
                let snapshot = template.telemetry_snapshot();
                totals.add(&snapshot);
            }
            Mode::Warm => {
                let prepare_started = Instant::now();
                let plan = prepare_plan(assembly)?;
                prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
                let (mut solver, target) = build_solver(plan, layout, nodes, parameter_at(0))?;
                solver.solve().map_err(|error| error.to_string())?;
                let after_initial = solver.plan().telemetry_snapshot();
                let initial_workspace_resizes = after_initial.workspace_resizes;
                let initial_allocations = after_initial.allocations;
                let continuation_started = Instant::now();
                for index in 1..count {
                    let parameter = parameter_at(index);
                    target.store(parameter.to_bits(), Ordering::Relaxed);
                    solver
                        .set_parameters(vec![parameter])
                        .map_err(|error| error.to_string())?;
                    solver.solve().map_err(|error| error.to_string())?;
                }
                continuation_ms = Some(continuation_started.elapsed().as_secs_f64() * 1e3);
                let final_snapshot = solver.plan().telemetry_snapshot();
                continuation_factorizations = final_snapshot
                    .factorizations
                    .saturating_sub(after_initial.factorizations);
                let retained = final_snapshot.workspace_resizes == initial_workspace_resizes
                    && final_snapshot.allocations == initial_allocations;
                retention = if retained { "bounded" } else { "grew" };
                totals.add(&final_snapshot);
                if !retained {
                    return Err(format!(
                        "warm logical workspace grew: resizes {}->{} allocations {}->{}",
                        initial_workspace_resizes,
                        final_snapshot.workspace_resizes,
                        initial_allocations,
                        final_snapshot.allocations
                    ));
                }
            }
        }

        let expected_parameter = parameter_at(count.saturating_sub(1));
        let total_ms = total_started.elapsed().as_secs_f64() * 1e3;
        let per_solve_ms = total_ms / count.max(1) as f64;
        let telemetry = match mode {
            Mode::Fresh | Mode::Prepared => totals,
            Mode::Warm => totals,
        };
        Ok(ContinuationLifecycleRow {
            mode: mode.label().into(),
            nodes,
            count,
            frontend: format!("{assembly:?}"),
            layout: layout_name(layout),
            prepare_ms: format!("{prepare_ms:.3}"),
            total_ms: format!("{total_ms:.3}"),
            per_solve_ms: format!("{per_solve_ms:.3}"),
            continuation_ms: continuation_ms
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".into()),
            factorizations: telemetry.factorizations,
            continuation_factorizations,
            sparse_symbolic_analyses: telemetry.sparse_symbolic_analyses,
            sparse_numeric_factorizations: telemetry.sparse_numeric_factorizations,
            jacobian_refreshes: telemetry.jacobian_refreshes,
            parameter_rebinds: telemetry.parameter_rebinds,
            continuation_solves: telemetry.continuation_solves,
            workspace_resizes: telemetry.workspace_resizes,
            logical_allocations: telemetry.logical_allocations,
            retention: retention.into(),
            status: format!("ok:p={expected_parameter:.2}"),
        })
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_continuation_fresh_prepared_warm_retention",
    );
    let nodes_list = parse_list("BVP_SCI_STORY_CONTINUATION_LIFECYCLE_NODES", "32,64", 4);
    let counts = parse_list("BVP_SCI_STORY_CONTINUATION_LIFECYCLE_COUNTS", "1,4,16", 1);
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for nodes in nodes_list {
        for count in &counts {
            for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
                for layout in [
                    BvpSciMatrixLayout::Dense,
                    BvpSciMatrixLayout::Sparse,
                    BvpSciMatrixLayout::Banded {
                        lower: 256,
                        upper: 256,
                    },
                ] {
                    for mode in [Mode::Fresh, Mode::Prepared, Mode::Warm] {
                        match run_mode(mode, assembly, layout, nodes, *count) {
                            Ok(row) => rows.push(row),
                            Err(error) => failures.push(format!(
                                "mode={} nodes={} count={} frontend={assembly:?} layout={} error={error}",
                                mode.label(),
                                nodes,
                                count,
                                layout_name(layout)
                            )),
                        }
                    }
                }
            }
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci continuation lifecycle\n\n{}",
        Table::new(rows)
    ));
    assert!(
        failures.is_empty(),
        "continuation lifecycle failures: {failures:?}"
    );
}

#[test]
#[ignore = "medium/large Dense/Sparse/Banded scale and fallback release gate"]
fn lambdify_layout_scale_matrix_reports_banded_route_costs() {
    #[derive(Tabled)]
    struct ScaleRow {
        workload: String,
        nodes: usize,
        frontend: String,
        layout: String,
        prepare_ms: String,
        solve_ms: String,
        full_solve_ms: String,
        linear_assembly_ms: String,
        factorization_ms: String,
        linear_solve_ms: String,
        max_exact_diff: String,
        banded_route: String,
        structured_factorizations: u64,
        scalar_fallback_factorizations: u64,
        scalar_fallback_assemblies: u64,
        structured_solves: u64,
        scalar_fallback_solves: u64,
        sparse_fallback_factorizations: u64,
        sparse_fallback_solves: u64,
        sparse_fallback_assemblies: u64,
        structured_factorization_ms: String,
        structured_solve_ms: String,
        residual_guard_ms: String,
        rhs_permutation_ms: String,
        sparse_fallback_factorization_ms: String,
        sparse_fallback_solve_ms: String,
        residual_checks: u64,
        fallback_switches: u64,
        newton_trace: String,
        banded_factor_residual: String,
        banded_min_abs_pivot: String,
        banded_max_multiplier: String,
        banded_border_min_abs_pivot: String,
        banded_border_max_multiplier: String,
        structured_candidate_residual_inf: String,
        structured_candidate_solution_inf: String,
        sparse_fallback_assembly_ms: String,
        status: String,
    }

    fn parse_list(name: &str, default: &str, minimum: usize) -> Vec<usize> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .filter_map(|value| value.trim().parse().ok())
            .filter(|value: &usize| *value >= minimum)
            .collect()
    }

    fn selected_workloads() -> Vec<DampExactWorkload> {
        std::env::var("BVP_SCI_STORY_SCALE_WORKLOADS")
            .unwrap_or_else(|_| "stiff-decay,stiff-coupled".into())
            .split(',')
            .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
                "stiff-decay" => Some(DampExactWorkload::StiffDecay),
                "stiff-coupled" => Some(DampExactWorkload::StiffCoupled),
                _ => None,
            })
            .collect()
    }

    fn workload_name(workload: DampExactWorkload) -> &'static str {
        match workload {
            DampExactWorkload::StiffDecay => "stiff-decay",
            DampExactWorkload::StiffCoupled => "stiff-coupled",
            _ => "unsupported",
        }
    }

    fn minimum_nodes(workload: DampExactWorkload) -> usize {
        // The exact stiff-decay profile is exp(-20*x).  With mesh refinement
        // intentionally disabled, fewer than 64 nodes is a discretization
        // failure of the workload, not evidence about a linear backend.
        if matches!(workload, DampExactWorkload::StiffDecay) {
            64
        } else {
            2
        }
    }

    fn selected_layouts() -> Vec<BvpSciMatrixLayout> {
        std::env::var("BVP_SCI_STORY_SCALE_LAYOUTS")
            .unwrap_or_else(|_| "dense,sparse,banded".into())
            .split(',')
            .filter_map(|value| match value.trim().to_ascii_lowercase().as_str() {
                "dense" => Some(BvpSciMatrixLayout::Dense),
                "sparse" => Some(BvpSciMatrixLayout::Sparse),
                "banded" => Some(BvpSciMatrixLayout::Banded {
                    lower: 256,
                    upper: 256,
                }),
                _ => None,
            })
            .collect()
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_layout_scale_matrix",
    );
    let nodes_list = parse_list("BVP_SCI_STORY_SCALE_NODES", "20,64,256,1024", 2);
    let mut rows = Vec::new();
    for workload in selected_workloads() {
        for &nodes in &nodes_list {
            if nodes < minimum_nodes(workload) {
                continue;
            }
            for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
                for layout in selected_layouts() {
                    let started = Instant::now();
                    // This matrix reports route/fallback counters as well as
                    // timings, so capture both channels rather than timings
                    // alone.
                    let mut solver = build_damp_exact_solver_with_nodes_and_telemetry(
                        workload,
                        assembly,
                        layout,
                        nodes,
                        BvpSciTelemetry::detailed(),
                    );
                    let prepare_ms = started.elapsed().as_secs_f64() * 1e3;
                    let fmt = |value: Option<f64>| {
                        value
                            .map(|value| format!("{value:.3}"))
                            .unwrap_or_else(|| "-".into())
                    };
                    let fmt_trace = |telemetry: &crate::numerical::BVP_sci::new::telemetry::BvpSciTelemetrySnapshot| {
                        if telemetry.newton_residual_history.is_empty() {
                            return "-".to_string();
                        }
                        telemetry
                            .newton_residual_history
                            .iter()
                            .map(|entry| {
                                format!(
                                    "{}:{:.2e}->{:.2e};step={:.2e};bt={};jr={};{}",
                                    entry.iteration,
                                    entry.residual_before,
                                    entry.residual_after.unwrap_or(f64::NAN),
                                    entry.step_inf_norm,
                                    entry.backtracking_trials,
                                    u8::from(entry.jacobian_refreshed),
                                    if entry.accepted { "ok" } else { "reject" }
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("|")
                    };
                    let solve_started = Instant::now();
                    let solution = match solver.solve() {
                        Ok(solution) => solution,
                        Err(error) => {
                            let telemetry = solver.plan().telemetry_snapshot();
                            rows.push(ScaleRow {
                                workload: workload_name(workload).into(),
                                nodes,
                                frontend: format!("{assembly:?}"),
                                layout: layout_name(layout),
                                prepare_ms: format!("{prepare_ms:.3}"),
                                solve_ms: format!(
                                    "{:.3}",
                                    solve_started.elapsed().as_secs_f64() * 1e3
                                ),
                                full_solve_ms: fmt(telemetry.full_solve_ms),
                                linear_assembly_ms: fmt(telemetry.linear_assembly_ms),
                                factorization_ms: fmt(telemetry.factorization_ms),
                                linear_solve_ms: fmt(telemetry.linear_solve_ms),
                                max_exact_diff: "-".into(),
                                banded_route: if telemetry.banded_sparse_fallback_solves > 0
                                    || telemetry.banded_sparse_fallback_factorizations > 0
                                {
                                    "sparse-fallback".into()
                                } else if telemetry.banded_scalar_fallback_solves > 0
                                    || telemetry.banded_scalar_fallback_factorizations > 0
                                {
                                    "scalar-fallback".into()
                                } else if telemetry.banded_structured_solves > 0
                                    || telemetry.banded_structured_factorizations > 0
                                {
                                    "structured".into()
                                } else {
                                    "-".into()
                                },
                                structured_factorizations: telemetry
                                    .banded_structured_factorizations,
                                scalar_fallback_factorizations: telemetry
                                    .banded_scalar_fallback_factorizations,
                                scalar_fallback_assemblies: telemetry
                                    .banded_scalar_fallback_assemblies,
                                structured_solves: telemetry.banded_structured_solves,
                                scalar_fallback_solves: telemetry.banded_scalar_fallback_solves,
                                sparse_fallback_factorizations: telemetry
                                    .banded_sparse_fallback_factorizations,
                                sparse_fallback_solves: telemetry.banded_sparse_fallback_solves,
                                sparse_fallback_assemblies: telemetry
                                    .banded_sparse_fallback_assemblies,
                                structured_factorization_ms: fmt(
                                    telemetry.banded_structured_factorization_ms
                                ),
                                structured_solve_ms: fmt(telemetry.banded_structured_solve_ms),
                                residual_guard_ms: fmt(telemetry.banded_residual_guard_ms),
                                rhs_permutation_ms: fmt(telemetry.banded_rhs_permutation_ms),
                                sparse_fallback_factorization_ms: fmt(
                                    telemetry.banded_sparse_fallback_factorization_ms
                                ),
                                sparse_fallback_solve_ms: fmt(
                                    telemetry.banded_sparse_fallback_solve_ms
                                ),
                                residual_checks: telemetry.banded_residual_checks,
                                fallback_switches: telemetry.banded_fallback_switches,
                                newton_trace: fmt_trace(&telemetry),
                                banded_factor_residual: fmt(
                                    telemetry.banded_core_factor_residual_relative
                                ),
                                banded_min_abs_pivot: fmt(telemetry.banded_min_abs_pivot),
                                banded_max_multiplier: fmt(telemetry.banded_max_multiplier_norm),
                                banded_border_min_abs_pivot: fmt(
                                    telemetry.banded_border_min_abs_pivot
                                ),
                                banded_border_max_multiplier: fmt(
                                    telemetry.banded_border_max_multiplier_norm
                                ),
                                structured_candidate_residual_inf: fmt(
                                    telemetry.banded_last_residual_inf
                                ),
                                structured_candidate_solution_inf: fmt(
                                    telemetry.banded_last_solution_inf
                                ),
                                sparse_fallback_assembly_ms: fmt(
                                    telemetry.banded_sparse_fallback_assembly_ms
                                ),
                                status: format!("error: {error}"),
                            });
                            continue;
                        }
                    };
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
                    let telemetry = solver.plan().telemetry_snapshot();
                    let state_dimension = workload.state_names().len();
                    let solution_y = &solution.y;
                    let max_exact_diff = solution
                        .x
                        .iter()
                        .enumerate()
                        .flat_map(|(node, &x)| {
                            workload.exact_state(x).into_iter().enumerate().map(
                                move |(component, expected)| {
                                    (solution_y[node * state_dimension + component] - expected)
                                        .abs()
                                },
                            )
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        max_exact_diff < 5e-2,
                        "scale row has excessive exact-solution drift: workload={} nodes={} frontend={assembly:?} layout={layout:?} diff={max_exact_diff:e}",
                        workload_name(workload),
                        nodes
                    );
                    let banded_route = if telemetry.banded_sparse_fallback_solves > 0
                        || telemetry.banded_sparse_fallback_factorizations > 0
                    {
                        "sparse-fallback"
                    } else if telemetry.banded_scalar_fallback_solves > 0
                        || telemetry.banded_scalar_fallback_factorizations > 0
                    {
                        "scalar-fallback"
                    } else if telemetry.banded_structured_solves > 0
                        || telemetry.banded_structured_factorizations > 0
                    {
                        "structured"
                    } else {
                        "-"
                    };
                    if matches!(workload, DampExactWorkload::StiffCoupled)
                        && layout_name(layout) == "Banded"
                    {
                        let candidate_residual = telemetry
                            .banded_last_residual_inf
                            .expect("structured candidate must reach the residual guard");
                        assert!(
                            !candidate_residual.is_finite() || candidate_residual > 1.0e-8,
                            "stiff-coupled Banded candidate should be rejected by the residual guard; residual={candidate_residual:e}, route={banded_route}"
                        );
                        assert_ne!(
                            banded_route, "structured",
                            "a rejected structured candidate must return through a fallback"
                        );
                        assert!(
                            max_exact_diff < 1.0e-8,
                            "fallback solve must retain the accurate BVP solution: nodes={nodes}, error={max_exact_diff:e}"
                        );
                    }
                    assert!(
                        solution.residual_norm.is_finite(),
                        "non-finite residual in scale row"
                    );
                    rows.push(ScaleRow {
                        workload: workload_name(workload).into(),
                        nodes,
                        frontend: format!("{assembly:?}"),
                        layout: layout_name(layout),
                        prepare_ms: format!("{prepare_ms:.3}"),
                        solve_ms: format!("{solve_ms:.3}"),
                        full_solve_ms: fmt(telemetry.full_solve_ms),
                        linear_assembly_ms: fmt(telemetry.linear_assembly_ms),
                        factorization_ms: fmt(telemetry.factorization_ms),
                        linear_solve_ms: fmt(telemetry.linear_solve_ms),
                        max_exact_diff: format!("{max_exact_diff:.3e}"),
                        banded_route: banded_route.into(),
                        structured_factorizations: telemetry.banded_structured_factorizations,
                        scalar_fallback_factorizations: telemetry
                            .banded_scalar_fallback_factorizations,
                        scalar_fallback_assemblies: telemetry.banded_scalar_fallback_assemblies,
                        structured_solves: telemetry.banded_structured_solves,
                        scalar_fallback_solves: telemetry.banded_scalar_fallback_solves,
                        sparse_fallback_factorizations: telemetry
                            .banded_sparse_fallback_factorizations,
                        sparse_fallback_solves: telemetry.banded_sparse_fallback_solves,
                        sparse_fallback_assemblies: telemetry.banded_sparse_fallback_assemblies,
                        structured_factorization_ms: fmt(
                            telemetry.banded_structured_factorization_ms
                        ),
                        structured_solve_ms: fmt(telemetry.banded_structured_solve_ms),
                        residual_guard_ms: fmt(telemetry.banded_residual_guard_ms),
                        rhs_permutation_ms: fmt(telemetry.banded_rhs_permutation_ms),
                        sparse_fallback_factorization_ms: fmt(
                            telemetry.banded_sparse_fallback_factorization_ms
                        ),
                        sparse_fallback_solve_ms: fmt(telemetry.banded_sparse_fallback_solve_ms),
                        residual_checks: telemetry.banded_residual_checks,
                        fallback_switches: telemetry.banded_fallback_switches,
                        newton_trace: fmt_trace(&telemetry),
                        banded_factor_residual: fmt(telemetry.banded_core_factor_residual_relative),
                        banded_min_abs_pivot: fmt(telemetry.banded_min_abs_pivot),
                        banded_max_multiplier: fmt(telemetry.banded_max_multiplier_norm),
                        banded_border_min_abs_pivot: fmt(telemetry.banded_border_min_abs_pivot),
                        banded_border_max_multiplier: fmt(
                            telemetry.banded_border_max_multiplier_norm
                        ),
                        structured_candidate_residual_inf: fmt(telemetry.banded_last_residual_inf),
                        structured_candidate_solution_inf: fmt(telemetry.banded_last_solution_inf),
                        sparse_fallback_assembly_ms: fmt(
                            telemetry.banded_sparse_fallback_assembly_ms
                        ),
                        status: solution.status.code().to_string(),
                    });
                }
            }
        }
    }
    assert!(
        !rows.is_empty(),
        "scale matrix selected no workloads or nodes"
    );
    let failed_rows = rows.iter().filter(|row| row.status != "0").count();
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci Lambdify layout scale and Banded route matrix\n\n{}",
        Table::new(rows)
    ));
    assert_eq!(failed_rows, 0, "scale matrix contains failed backend rows");
}

#[test]
#[ignore = "captured stiff-coupled Banded direct-vs-refined diagnostic gate"]
fn banded_stiff_coupled_captured_matrix_compares_core_refinement() {
    use crate::numerical::BVP_sci::new::backends::BandedBackend;

    #[derive(Tabled)]
    struct RefinementRow {
        workload: String,
        nodes: usize,
        refinement_steps: usize,
        route: String,
        factor_ms: String,
        safety_factor_ms: String,
        solve_ms: String,
        residual_inf: String,
        backward_error: String,
        solution_inf: String,
        fallback_factorizations: u64,
        fallback_solves: u64,
        status: String,
    }

    fn matrix_quality(
        entries: &[(usize, usize, f64)],
        rhs: &[f64],
        solution: &[f64],
    ) -> (f64, f64, f64) {
        let mut residual_inf = 0.0_f64;
        let mut matrix_inf = 0.0_f64;
        let mut solution_inf = 0.0_f64;
        for value in solution {
            solution_inf = solution_inf.max(value.abs());
            if !value.is_finite() {
                return (f64::INFINITY, f64::INFINITY, solution_inf);
            }
        }
        let mut row_sums = vec![0.0_f64; rhs.len()];
        let mut residual = rhs.to_vec();
        for &(row, column, value) in entries {
            row_sums[row] += value.abs();
            residual[row] -= value * solution[column];
        }
        for (row, value) in residual.into_iter().enumerate() {
            residual_inf = residual_inf.max(value.abs());
            matrix_inf = matrix_inf.max(row_sums[row]);
        }
        let rhs_inf = rhs.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let scale = (matrix_inf * solution_inf + rhs_inf).max(f64::MIN_POSITIVE);
        (residual_inf, residual_inf / scale, solution_inf)
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "banded_stiff_coupled_refinement_ab",
    );
    let mut rows = Vec::new();
    for nodes in [64, 256, 1024] {
        let capture_telemetry = BvpSciTelemetry::timings();
        capture_telemetry.enable_linear_system_capture(1);
        let mut solver = build_damp_exact_solver_with_nodes_and_telemetry(
            DampExactWorkload::StiffCoupled,
            BvpSciAssembly::ExprLegacy,
            BvpSciMatrixLayout::Banded {
                lower: 256,
                upper: 256,
            },
            nodes,
            capture_telemetry.clone(),
        );
        solver
            .solve()
            .expect("the safety-routed stiff-coupled reference solve should converge");
        let capture = capture_telemetry
            .linear_system_capture()
            .expect("the first Newton matrix and RHS should be captured");
        let state_dimension = DampExactWorkload::StiffCoupled.state_names().len();

        for refinement_steps in [0, 2] {
            let telemetry = BvpSciTelemetry::detailed();
            let mut backend = BandedBackend::new_for_collocation_with_refinement_steps(
                capture.dimension,
                256,
                256,
                state_dimension,
                capture.mesh_nodes,
                0,
                telemetry.clone(),
                refinement_steps,
            )
            .expect("captured Banded matrix dimensions should be accepted");
            backend
                .assemble(&capture.entries)
                .expect("captured Newton matrix should assemble in native Banded storage");
            let factor_started = Instant::now();
            backend
                .factor()
                .expect("Banded matrix factorization should succeed");
            let factor_ms = factor_started.elapsed().as_secs_f64() * 1e3;
            let mut solution = capture.rhs.clone();
            let solve_started = Instant::now();
            backend
                .solve_in_place(&mut solution)
                .expect("structured or guarded Banded solve should return a correction");
            let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
            let (residual_inf, backward_error, solution_inf) =
                matrix_quality(&capture.entries, &capture.rhs, &solution);
            let snapshot = telemetry.snapshot();
            let route = if snapshot.banded_sparse_fallback_solves > 0 {
                "sparse-fallback"
            } else if snapshot.banded_scalar_fallback_solves > 0 {
                "scalar-fallback"
            } else {
                "structured"
            };
            let fallback_factorizations = snapshot.banded_scalar_fallback_factorizations
                + snapshot.banded_sparse_fallback_factorizations;
            let fallback_solves =
                snapshot.banded_scalar_fallback_solves + snapshot.banded_sparse_fallback_solves;
            let safety_factor_ms = snapshot
                .banded_scalar_fallback_factorization_ms
                .or(snapshot.banded_sparse_fallback_factorization_ms)
                .map(|elapsed| format!("{elapsed:.3}"))
                .unwrap_or_else(|| "-".into());
            let status = if backward_error.is_finite() && backward_error < 1e-10 {
                "ok"
            } else {
                "bad-residual"
            };
            rows.push(RefinementRow {
                workload: "stiff-coupled".into(),
                nodes,
                refinement_steps,
                route: route.into(),
                factor_ms: format!("{factor_ms:.3}"),
                safety_factor_ms,
                solve_ms: format!("{solve_ms:.3}"),
                residual_inf: format!("{residual_inf:.3e}"),
                backward_error: format!("{backward_error:.3e}"),
                solution_inf: format!("{solution_inf:.3e}"),
                fallback_factorizations,
                fallback_solves,
                status: status.into(),
            });
            assert_eq!(
                status, "ok",
                "captured matrix replay produced an inaccurate solve"
            );
        }
    }

    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci stiff-coupled Banded refinement A/B\n\nRows replay the same captured first-Newton matrix/RHS for each mesh; factor/solve scopes are local to the replay and fallback counters identify the route actually returned.\n\n{}",
        Table::new(&rows)
    ));
}

#[test]
#[ignore = "AOT compiler and lifecycle release gate"]
fn aot_frontend_backend_contract_matrix_is_compact_and_parity_checked() {
    #[derive(Debug, Tabled)]
    struct AotRow {
        frontend: String,
        layout: String,
        prepare_scope: String,
        prepare_ms: String,
        aot_prepare_ms: String,
        callback_ms: String,
        residual_ms: String,
        jacobian_ms: String,
        jacobian_nnz: usize,
        parameter_jacobian_nnz: usize,
        parameter_jacobian_calls: u64,
        fd_probes: u64,
        runtime_ready: u64,
        build_attempts: u64,
        link_attempts: u64,
        parallel_dispatches: u64,
        max_rhs_diff: String,
        max_jacobian_diff: String,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "aot_frontend_backend_contract_matrix",
    );
    let equations = vec![
        Expr::parse_expression("p*y0 + x"),
        Expr::parse_expression("y0 - y1 + p*x"),
    ];
    let states = vec!["y0".into(), "y1".into()];
    let parameters = vec!["p".into()];
    let output_root = std::env::var_os("BVP_SCI_AOT_STORY_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("target/bvp-sci-aot-story"));
    let compiler = std::env::var("BVP_SCI_AOT_COMPILER").unwrap_or_else(|_| "tcc".into());
    let layouts = [
        ("Dense", BvpSciMatrixLayout::Dense),
        ("Sparse", BvpSciMatrixLayout::Sparse),
        ("Banded", BvpSciMatrixLayout::Banded { lower: 1, upper: 1 }),
    ];
    let frontends = [
        ("ExprLegacy", BvpSciAssembly::ExprLegacy),
        ("AtomViewNative", BvpSciAssembly::AtomViewNative),
    ];
    let mut rows = Vec::new();
    for (frontend_name, frontend) in frontends {
        for (layout_name, layout) in layouts {
            let telemetry = BvpSciTelemetry::detailed();
            let mut config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                output_root.join(frontend_name).join(layout_name),
            );
            config = match std::env::var("BVP_SCI_AOT_COMPILER")
                .unwrap_or_else(|_| "tcc".into())
                .to_ascii_lowercase()
                .as_str()
            {
                "gcc" => config.with_c_gcc(),
                _ => config.with_c_tcc(),
            };
            let started = Instant::now();
            let plan = BvpSciLambdifyPlan::prepare_aot(
                frontend,
                layout,
                equations.clone(),
                states.clone(),
                parameters.clone(),
                "x",
                config,
                telemetry,
            )
            .expect("AOT route should publish a typed runtime");
            let prepare_ms = started.elapsed().as_secs_f64() * 1e3;
            assert!(plan.is_aot());
            assert_eq!(plan.assembly(), frontend);

            let mut arguments = vec![0.0; 4];
            let mut rhs = vec![0.0; 2];
            let mut jacobian = vec![0.0; 4];
            // Compact-Banded output includes zero boundary slots, so its
            // callback buffer can exceed the structural nonzero count.
            let mut scratch = vec![0.0; 16];
            let callback_started = Instant::now();
            plan.evaluate_rhs(0.25, &[1.0, 2.0], &[3.0], &mut arguments, &mut rhs)
                .expect("AOT residual callback should evaluate");
            plan.evaluate_jacobian_dense_with_scratch(
                0.25,
                &[1.0, 2.0],
                &[3.0],
                &mut arguments,
                &mut jacobian,
                &mut scratch,
            )
            .expect("AOT Jacobian callback should evaluate");
            let mut parameter_jacobian = vec![0.0; 2];
            plan.evaluate_parameter_jacobian(
                0.25,
                &[1.0, 2.0],
                &[3.0],
                &mut arguments,
                &mut parameter_jacobian,
            )
            .expect("AOT Lambdify parameter Jacobian should evaluate");
            let callback_ms = callback_started.elapsed().as_secs_f64() * 1e3;
            let expected_rhs = [3.25, -0.25];
            let expected_jacobian = [3.0, 0.0, 1.0, -1.0];
            let max_rhs_diff = rhs
                .iter()
                .zip(expected_rhs)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            let max_jacobian_diff = jacobian
                .iter()
                .zip(expected_jacobian)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(max_rhs_diff < 1e-12, "AOT residual drift={max_rhs_diff:e}");
            assert!(
                max_jacobian_diff < 1e-12,
                "AOT Jacobian drift={max_jacobian_diff:e}"
            );
            assert_eq!(parameter_jacobian, [1.0, 0.25]);

            let telemetry_snapshot = plan.telemetry_snapshot();
            telemetry_snapshot
                .validate_contract()
                .expect("AOT callback telemetry must satisfy its contract");
            let aot = telemetry_snapshot
                .aot
                .expect("AOT plan must expose shared lifecycle telemetry");
            assert!(aot.aot_runtime_ready > 0);
            assert!(!aot.aot_artifact_keys.is_empty());
            assert!(telemetry_snapshot.residual_evaluation_ms.is_some());
            assert!(telemetry_snapshot.jacobian_evaluation_ms.is_some());
            assert!(plan.has_parameter_jacobian());
            assert_eq!(plan.parameter_jacobian_nnz(), 2);
            assert_eq!(telemetry_snapshot.parameter_jacobian_calls, 1);
            assert_eq!(telemetry_snapshot.finite_difference_probes, 0);
            rows.push(AotRow {
                frontend: frontend_name.into(),
                layout: layout_name.into(),
                prepare_scope: "caller_wall_clock".into(),
                prepare_ms: measured_ms(prepare_ms),
                aot_prepare_ms: aot_stage_ms(
                    &aot,
                    crate::symbolic::ivp_telemetry::IvpColdStage::SolverPreparation,
                ),
                callback_ms: measured_ms(callback_ms),
                residual_ms: timing(telemetry_snapshot.residual_evaluation_ms),
                jacobian_ms: timing(telemetry_snapshot.jacobian_evaluation_ms),
                jacobian_nnz: plan.jacobian_nnz(),
                parameter_jacobian_nnz: plan.parameter_jacobian_nnz(),
                parameter_jacobian_calls: telemetry_snapshot.parameter_jacobian_calls,
                fd_probes: telemetry_snapshot.finite_difference_probes,
                runtime_ready: aot.aot_runtime_ready,
                build_attempts: aot.aot_build_attempts,
                link_attempts: aot.aot_link_attempts,
                parallel_dispatches: aot.aot_parallel_dispatches,
                max_rhs_diff: format!("{max_rhs_diff:.3e}"),
                max_jacobian_diff: format!("{max_jacobian_diff:.3e}"),
                status: "ok".into(),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci AOT frontend/backend contract matrix\n\n- lifecycle: BuildIfMissing + C/{compiler}\n- callback ABI: [x, parameters..., state...]\n- hot residual and `df/dy` use AOT; the sparse `df/dp` block is symbolically differentiated and Lambdified once with the same ExprLegacy/AtomView assembly\n- `residual_ms` and `jacobian_ms` are separate callback telemetry scopes\n- AOT telemetry is a separate non-additive IVP snapshot\n\n{}",
        Table::new(rows)
    ));
}

#[test]
#[ignore = "AOT solver, continuation and policy release gate"]
fn aot_solver_route_continuation_and_policy_matrix_is_compact() {
    #[derive(Debug, Tabled)]
    struct SolverRow {
        frontend: String,
        layout: String,
        sparse_factorization: String,
        policy: String,
        prepare_scope: String,
        prepare_ms: String,
        aot_prepare_ms: String,
        solve_ms: String,
        continuation_ms: String,
        parameter_rebinds: u64,
        continuation_solves: u64,
        runtime_ready: u64,
        parallel_dispatches: u64,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "aot_solver_route_continuation_and_policy_matrix",
    );
    let output_root = std::env::var_os("BVP_SCI_AOT_STORY_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("target/bvp-sci-aot-story"));
    let equations = vec![Expr::parse_expression("p*y")];
    let states = vec!["y".into()];
    let parameters = vec!["p".into()];
    let layouts = [
        ("Dense", BvpSciMatrixLayout::Dense),
        ("Sparse", BvpSciMatrixLayout::Sparse),
        (
            "Banded",
            // This production-route fixture has one state equation, so its
            // valid band is the diagonal-only 1x1 band.
            BvpSciMatrixLayout::Banded { lower: 0, upper: 0 },
        ),
    ];
    let frontends = [
        ("ExprLegacy", BvpSciAssembly::ExprLegacy),
        ("AtomViewNative", BvpSciAssembly::AtomViewNative),
    ];
    let policies = [
        ("sequential", BvpSciExecutionPolicy::Sequential),
        ("parallel", BvpSciExecutionPolicy::Parallel { min_work: 0 }),
        ("auto", BvpSciExecutionPolicy::Auto { min_work: 0 }),
    ];
    let mut rows = Vec::new();
    for (frontend_name, frontend) in frontends {
        for (layout_name, layout) in layouts {
            for (policy_name, execution_policy) in policies {
                let mut sparse_factorizations =
                    vec![("StandardFaer", BvpSciSparseFactorization::StandardFaer)];
                #[cfg(feature = "superlu")]
                if layout == BvpSciMatrixLayout::Sparse
                    && std::env::var_os("SUPERLU_LIBRARY").is_some()
                {
                    sparse_factorizations.push(("SuperLU", BvpSciSparseFactorization::SuperLu));
                }
                for (sparse_factorization_name, sparse_factorization) in &sparse_factorizations {
                    let telemetry = BvpSciTelemetry::detailed();
                    let mut config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                        output_root
                            .join("solver")
                            .join(frontend_name)
                            .join(layout_name)
                            .join(*sparse_factorization_name)
                            .join(policy_name),
                    );
                    config = match std::env::var("BVP_SCI_AOT_COMPILER")
                        .unwrap_or_else(|_| "tcc".into())
                        .to_ascii_lowercase()
                        .as_str()
                    {
                        "gcc" => config.with_c_gcc(),
                        _ => config.with_c_tcc(),
                    };
                    let prepare_started = Instant::now();
                    let plan = BvpSciLambdifyPlan::prepare_aot_with_policy(
                        frontend,
                        layout,
                        equations.clone(),
                        states.clone(),
                        parameters.clone(),
                        "x",
                        config,
                        telemetry,
                        execution_policy,
                    )
                    .expect("AOT solver plan should prepare");
                    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
                    let boundary = BvpSciBoundaryCallbacks::new(
                        2,
                        |ya, yb, _, output| {
                            output[0] = ya[0];
                            output[1] = yb[0];
                            Ok(())
                        },
                        BvpSciTelemetry::disabled(),
                    );
                    let mut options = BvpSciOptions::default();
                    options.execution = BvpSciExecution::Aot;
                    options.assembly = Some(frontend);
                    options.matrix_layout = layout;
                    options.execution_policy = execution_policy;
                    options.max_nodes = 32;
                    options.max_mesh_refinements = 0;
                    options.tolerance = 1e-8;
                    let mut solver = BvpSciSolver::new_with_sparse_factorization(
                        plan,
                        boundary,
                        vec![0.0, 0.5, 1.0],
                        vec![0.1, 0.1, 0.1],
                        vec![1.0],
                        options,
                        *sparse_factorization,
                    )
                    .expect("AOT solver route should construct");
                    let solve_started = Instant::now();
                    let solution = solver.solve().expect("AOT solver route should converge");
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
                    let continuation_started = Instant::now();
                    solver
                        .set_parameters(vec![2.0])
                        .expect("AOT continuation rebind should preserve schema");
                    let continued = solver
                        .solve()
                        .expect("AOT continuation solve should converge");
                    let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
                    assert!(solution.y.iter().all(|value| value.abs() < 1e-12));
                    assert!(continued.y.iter().all(|value| value.abs() < 1e-12));
                    let snapshot = solver.plan().telemetry_snapshot();
                    snapshot
                        .validate_contract()
                        .expect("AOT solver telemetry must satisfy its contract");
                    let aot = snapshot.aot.expect("solver AOT telemetry must be present");
                    assert!(aot.aot_runtime_ready > 0);
                    if *sparse_factorization == BvpSciSparseFactorization::SuperLu {
                        assert_eq!(snapshot.superlu_fallbacks, 0);
                        assert!(snapshot.superlu.native_factorization_calls > 0);
                    }
                    assert_eq!(snapshot.parameter_rebinds, 1);
                    assert_eq!(snapshot.continuation_solves, 1);
                    rows.push(SolverRow {
                        frontend: frontend_name.into(),
                        layout: layout_name.into(),
                        sparse_factorization: (*sparse_factorization_name).into(),
                        policy: policy_name.into(),
                        prepare_scope: "caller_wall_clock".into(),
                        prepare_ms: measured_ms(prepare_ms),
                        aot_prepare_ms: aot_stage_ms(
                            &aot,
                            crate::symbolic::ivp_telemetry::IvpColdStage::SolverPreparation,
                        ),
                        solve_ms: measured_ms(solve_ms),
                        continuation_ms: measured_ms(continuation_ms),
                        parameter_rebinds: snapshot.parameter_rebinds,
                        continuation_solves: snapshot.continuation_solves,
                        runtime_ready: aot.aot_runtime_ready,
                        parallel_dispatches: aot.aot_parallel_dispatches,
                        status: solution.status.code().to_string(),
                    });
                }
            }
        }
    }
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci AOT solver, continuation and policy matrix\n\n- exact fixture: y' = p*y with zero solution and zero boundary values; a nonzero initial guess exercises Newton and factorization\n- Sparse factorization rows include SuperLU when its feature and bridge are configured\n- policy is fixed during AOT preparation\n- full-solve timings include numerical work and are not callback-only timings\n\n{}",
        Table::new(rows)
    ));
}

#[test]
#[ignore = "AOT full-solve and matched frontend release gate"]
fn aot_full_solve_matches_lambdify_on_medium_parameterized_systems() {
    #[derive(Debug, Tabled)]
    struct FullSolveRow {
        frontend: String,
        layout: String,
        dimension: usize,
        nodes: usize,
        prepare_scope: String,
        prepare_ms: String,
        aot_prepare_ms: String,
        solve_ms: String,
        continuation_ms: String,
        full_solve_ms: String,
        callback_ms: String,
        residual_ms: String,
        jacobian_ms: String,
        factorization_ms: String,
        residual_calls: u64,
        jacobian_calls: u64,
        parameter_jacobian_calls: u64,
        finite_difference_probes: u64,
        aot_cache_hits: String,
        aot_cache_misses: String,
        build_attempts: String,
        link_attempts: String,
        runtime_ready: String,
        artifact_key: String,
        parity_diff: String,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "aot_full_solve_lambdify_parity_matrix",
    );
    let dimensions = std::env::var("BVP_SCI_AOT_FULL_SOLVE_DIMENSIONS")
        .unwrap_or_else(|_| "8,32".into())
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value >= 2)
        .collect::<Vec<_>>();
    let nodes = std::env::var("BVP_SCI_AOT_FULL_SOLVE_NODES")
        .unwrap_or_else(|_| "16,64".into())
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value >= 3)
        .collect::<Vec<_>>();
    let output_root = std::env::var_os("BVP_SCI_AOT_STORY_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("target/bvp-sci-aot-story"));
    let compiler = std::env::var("BVP_SCI_AOT_COMPILER").unwrap_or_else(|_| "tcc".into());
    let layouts = [
        ("Dense", BvpSciMatrixLayout::Dense),
        ("Sparse", BvpSciMatrixLayout::Sparse),
        ("Banded", BvpSciMatrixLayout::Banded { lower: 1, upper: 1 }),
    ];
    let frontends = [
        ("Lambdify-ExprLegacy", BvpSciAssembly::ExprLegacy, false),
        (
            "Lambdify-AtomViewNative",
            BvpSciAssembly::AtomViewNative,
            false,
        ),
        ("AOT-ExprLegacy", BvpSciAssembly::ExprLegacy, true),
        ("AOT-AtomViewNative", BvpSciAssembly::AtomViewNative, true),
    ];
    let mut rows = Vec::new();
    let mut reference_solutions = HashMap::<(usize, usize, String), Vec<f64>>::new();
    for &dimension in &dimensions {
        let equations = (0..dimension)
            .map(|index| Expr::parse_expression(&format!("p*(y{index} + 1)")))
            .collect::<Vec<_>>();
        let state_names = (0..dimension)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        for &node_count in &nodes {
            let mesh = (0..node_count)
                .map(|index| index as f64 / (node_count - 1) as f64)
                .collect::<Vec<_>>();
            for &(layout_name, layout) in &layouts {
                for &(frontend_name, assembly, is_aot) in &frontends {
                    let telemetry = BvpSciTelemetry::detailed();
                    let prepare_started = Instant::now();
                    let plan = if is_aot {
                        let mut config =
                            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                                output_root
                                    .join("full-solve")
                                    .join(frontend_name)
                                    .join(layout_name)
                                    .join(format!("{dimension}-{node_count}")),
                            );
                        config = if compiler.eq_ignore_ascii_case("gcc") {
                            config.with_c_gcc()
                        } else {
                            config.with_c_tcc()
                        };
                        BvpSciLambdifyPlan::prepare_aot(
                            assembly,
                            layout,
                            equations.clone(),
                            state_names.clone(),
                            vec!["p".into()],
                            "x",
                            config,
                            telemetry,
                        )
                    } else {
                        BvpSciLambdifyPlan::prepare(
                            assembly,
                            &equations,
                            &state_names,
                            &["p".into()],
                            "x",
                            telemetry,
                        )
                    };
                    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
                    let plan = plan.expect("full-solve frontend should prepare");
                    let target_dimension = dimension;
                    let boundary = BvpSciBoundaryCallbacks::new(
                        dimension + 1,
                        move |ya, yb, parameters, output| {
                            output[..target_dimension].copy_from_slice(ya);
                            let parameter = parameters.first().copied().unwrap_or(0.0);
                            output[target_dimension] = yb[0] - (parameter.exp() - 1.0);
                            Ok(())
                        },
                        BvpSciTelemetry::disabled(),
                    );
                    let mut options = BvpSciOptions::default();
                    options.execution = if is_aot {
                        BvpSciExecution::Aot
                    } else {
                        BvpSciExecution::Lambdify
                    };
                    options.assembly = Some(assembly);
                    options.matrix_layout = layout;
                    options.max_nodes = node_count.saturating_mul(16).max(128);
                    options.max_mesh_refinements = 8;
                    options.max_newton_iterations = 20;
                    options.max_jacobian_refreshes = 30;
                    options.tolerance = 1e-5;
                    // Keep the manufactured family nonzero but moderate for
                    // the smallest debug mesh. The full release matrix can
                    // raise this through a dedicated workload; this gate
                    // should not turn mesh resolution into its variable.
                    let initial_parameter = 0.5;
                    let mesh_initial_profile = mesh
                        .iter()
                        .flat_map(|&x| {
                            let exact = (initial_parameter * x).exp() - 1.0;
                            (0..dimension).map(move |_| {
                                0.90 * exact + 0.01 * (std::f64::consts::PI * x).sin()
                            })
                        })
                        .collect::<Vec<_>>();
                    let mut solver = BvpSciSolver::new(
                        plan,
                        boundary,
                        mesh.clone(),
                        // Use the same bounded non-exact profile for every
                        // frontend and layout. This forces the full-solve
                        // comparison through Newton, Jacobian and linear
                        // stages instead of measuring an already converged
                        // zero residual.
                        mesh_initial_profile,
                        vec![initial_parameter],
                        options,
                    )
                    .expect("full-solve solver route should construct");
                    let solve_started = Instant::now();
                    let solution = solver.solve().unwrap_or_else(|error| {
                        panic!(
                            "full-solve fixture should converge: frontend={frontend_name} layout={layout_name} dimension={dimension} nodes={node_count} error={error:?}"
                        )
                    });
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
                    solver
                        .set_parameters(vec![0.75])
                        .expect("full-solve continuation rebind should work");
                    let continuation_started = Instant::now();
                    let continued = solver.solve().unwrap_or_else(|error| {
                        panic!(
                            "full-solve continuation should converge: frontend={frontend_name} layout={layout_name} dimension={dimension} nodes={node_count} error={error:?}"
                        )
                    });
                    let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
                    assert!(
                        solution.y.iter().any(|value| value.abs() > 1e-3),
                        "nonzero full-solve fixture unexpectedly returned a zero solution"
                    );
                    assert!(
                        continued.y.iter().any(|value| value.abs() > 1e-3),
                        "nonzero continuation fixture unexpectedly returned a zero solution"
                    );
                    let snapshot = solver.plan().telemetry_snapshot();
                    snapshot
                        .validate_contract()
                        .expect("full-solve telemetry must satisfy its contract");
                    if is_aot {
                        assert!(solver.plan().has_parameter_jacobian());
                        assert!(
                            snapshot.parameter_jacobian_calls > 0,
                            "AOT solve must use its prepared symbolic df/dp evaluator"
                        );
                    }
                    let aot_provenance = snapshot.aot.as_ref().map(|aot| {
                        (
                            aot.aot_resolution_hits.to_string(),
                            aot.aot_resolution_misses.to_string(),
                            aot.aot_build_attempts.to_string(),
                            aot.aot_link_attempts.to_string(),
                            aot.aot_runtime_ready.to_string(),
                            aot.aot_artifact_keys
                                .first()
                                .cloned()
                                .unwrap_or_else(|| "-".into()),
                        )
                    });
                    let key = (dimension, node_count, layout_name.to_owned());
                    let parity_diff = reference_solutions
                        .get(&key)
                        .map(|reference| {
                            solution
                                .y
                                .iter()
                                .zip(reference)
                                .map(|(left, right)| (left - right).abs())
                                .fold(0.0, f64::max)
                        })
                        .unwrap_or(0.0);
                    reference_solutions.insert(key, solution.y.clone());
                    rows.push(FullSolveRow {
                        frontend: frontend_name.into(),
                        layout: layout_name.into(),
                        dimension,
                        nodes: node_count,
                        prepare_scope: snapshot.report_preparation_scope().into(),
                        prepare_ms: measured_ms(prepare_ms),
                        aot_prepare_ms: snapshot
                            .aot
                            .as_ref()
                            .map(|aot| {
                                aot_stage_ms(
                                    aot,
                                    crate::symbolic::ivp_telemetry::IvpColdStage::SolverPreparation,
                                )
                            })
                            .unwrap_or_else(|| "n/a".into()),
                        solve_ms: measured_ms(solve_ms),
                        continuation_ms: measured_ms(continuation_ms),
                        full_solve_ms: timing(snapshot.full_solve_ms),
                        callback_ms: timing(snapshot.callback_ms),
                        residual_ms: timing(snapshot.residual_evaluation_ms),
                        jacobian_ms: timing(snapshot.jacobian_evaluation_ms),
                        factorization_ms: format!("{}", timing(snapshot.factorization_ms)),
                        residual_calls: snapshot.residual_evaluations,
                        jacobian_calls: snapshot.jacobian_evaluations,
                        parameter_jacobian_calls: snapshot.parameter_jacobian_calls,
                        finite_difference_probes: snapshot.finite_difference_probes,
                        aot_cache_hits: aot_provenance
                            .as_ref()
                            .map(|value| value.0.clone())
                            .unwrap_or_else(|| "-".into()),
                        aot_cache_misses: aot_provenance
                            .as_ref()
                            .map(|value| value.1.clone())
                            .unwrap_or_else(|| "-".into()),
                        build_attempts: aot_provenance
                            .as_ref()
                            .map(|value| value.2.clone())
                            .unwrap_or_else(|| "-".into()),
                        link_attempts: aot_provenance
                            .as_ref()
                            .map(|value| value.3.clone())
                            .unwrap_or_else(|| "-".into()),
                        runtime_ready: aot_provenance
                            .as_ref()
                            .map(|value| value.4.clone())
                            .unwrap_or_else(|| "-".into()),
                        artifact_key: aot_provenance
                            .as_ref()
                            .map(|value| value.5.clone())
                            .unwrap_or_else(|| "-".into()),
                        parity_diff: format!("{parity_diff:.3e}"),
                        status: solution.status.code().to_string(),
                    });
                }
            }
        }
    }
    assert!(!rows.is_empty(), "full-solve matrix selected no rows");
    let report = format!(
        "# BVP_sci AOT versus Lambdify full solve and continuation\n\n- lifecycle: BuildIfMissing + C/{compiler}\n- matched manufactured family: `y' = p*(y + 1)`, `y(0) = 0`, `y(1) = exp(p) - 1`\n- every route uses the same bounded non-exact initial profile and `p=2 -> 3` continuation\n- `full_solve_ms` is inclusive per public solve; callback, residual, Jacobian and factorization scopes are diagnostic and non-additive\n- residual/Jacobian timings are accumulated across the initial and continued solves and are comparable only within the matched fixture\n\n{}",
        Table::new(rows)
    );
    crate::Utils::test_reporting::capture_test_block(report.replace("p=2 -> 3", "p=0.5 -> 0.75"));
}

#[test]
#[ignore = "AOT BuildIfMissing/RequirePrebuilt/RebuildAlways lifecycle gate"]
fn aot_lifecycle_policies_and_typed_failure_matrix_is_compact() {
    #[derive(Debug, Tabled)]
    struct LifecycleRow {
        policy: String,
        cache_hits: u64,
        cache_misses: u64,
        build_attempts: u64,
        build_successes: u64,
        link_attempts: u64,
        link_successes: u64,
        runtime_ready: u64,
        artifact_key: String,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_AOT",
        "aot_lifecycle_policies_and_typed_failure_matrix",
    );
    let root = tempfile::tempdir().expect("AOT lifecycle root");
    let handoff = root.path().join("producer-handoff.txt");
    // Keep this lifecycle fixture distinct from the other AOT stories that
    // run in the same test process. BuildIfMissing must observe a cold
    // artifact here; sharing the common `p*y` key would make the result depend
    // on story-test order rather than on the policy being exercised.
    let equations = vec![Expr::parse_expression("p*y + 0.2718281828459045")];
    let states = vec!["y".to_owned()];
    let parameters = vec!["p".to_owned()];
    let layout = BvpSciMatrixLayout::Dense;
    let mut rows = Vec::new();

    let producer_config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(root.path().join("build-if-missing")))
        .with_handoff_path(Some(handoff.clone()))
        .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Debug,
        })
        .with_c_tcc();
    let producer = BvpSciLambdifyPlan::prepare_aot(
        BvpSciAssembly::ExprLegacy,
        layout,
        equations.clone(),
        states.clone(),
        parameters.clone(),
        "x",
        producer_config.clone(),
        BvpSciTelemetry::detailed(),
    )
    .expect("BuildIfMissing producer should prepare");
    let producer_snapshot = producer
        .telemetry_snapshot()
        .aot
        .expect("producer AOT telemetry");
    producer_snapshot
        .validate_contract()
        .expect("producer lifecycle counters should be valid");
    assert!(producer_snapshot.aot_build_attempts >= 1);
    assert!(producer_snapshot.aot_build_successes >= 1);
    assert!(producer_snapshot.aot_runtime_ready >= 1);
    assert!(handoff.exists(), "producer must publish handoff metadata");
    rows.push(LifecycleRow {
        policy: "BuildIfMissing".into(),
        cache_hits: producer_snapshot.aot_resolution_hits,
        cache_misses: producer_snapshot.aot_resolution_misses,
        build_attempts: producer_snapshot.aot_build_attempts,
        build_successes: producer_snapshot.aot_build_successes,
        link_attempts: producer_snapshot.aot_link_attempts,
        link_successes: producer_snapshot.aot_link_successes,
        runtime_ready: producer_snapshot.aot_runtime_ready,
        artifact_key: producer_snapshot
            .aot_artifact_keys
            .first()
            .cloned()
            .unwrap_or_else(|| "-".into()),
        status: "published".into(),
    });

    // The consumer intentionally supplies no in-process resolver. The BVP
    // adapter reconstructs it from the producer handoff before RequirePrebuilt.
    let consumer_config = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_output_parent_dir(Some(root.path().join("build-if-missing")))
        .with_handoff_path(Some(handoff.clone()))
        .with_c_tcc();
    let consumer = BvpSciLambdifyPlan::prepare_aot(
        BvpSciAssembly::ExprLegacy,
        layout,
        equations.clone(),
        states.clone(),
        parameters.clone(),
        "x",
        consumer_config,
        BvpSciTelemetry::detailed(),
    );
    // Keep the argument order explicit in the report-producing gate. This
    // match also turns a future API drift into a useful test failure rather
    // than silently checking the wrong lifecycle route.
    let consumer = match consumer {
        Ok(plan) => plan,
        Err(error) => panic!("RequirePrebuilt handoff consumer failed: {error}"),
    };
    let consumer_snapshot = consumer
        .telemetry_snapshot()
        .aot
        .expect("consumer AOT telemetry");
    consumer_snapshot
        .validate_contract()
        .expect("consumer lifecycle counters should be valid");
    assert_eq!(consumer_snapshot.aot_build_attempts, 0);
    assert!(consumer_snapshot.aot_runtime_ready >= 1);
    rows.push(LifecycleRow {
        policy: "RequirePrebuilt".into(),
        cache_hits: consumer_snapshot.aot_resolution_hits,
        cache_misses: consumer_snapshot.aot_resolution_misses,
        build_attempts: consumer_snapshot.aot_build_attempts,
        build_successes: consumer_snapshot.aot_build_successes,
        link_attempts: consumer_snapshot.aot_link_attempts,
        link_successes: consumer_snapshot.aot_link_successes,
        runtime_ready: consumer_snapshot.aot_runtime_ready,
        artifact_key: consumer_snapshot
            .aot_artifact_keys
            .first()
            .cloned()
            .unwrap_or_else(|| "-".into()),
        status: "reconnected".into(),
    });

    let rebuild = BvpSciLambdifyPlan::prepare_aot(
        BvpSciAssembly::ExprLegacy,
        layout,
        equations.clone(),
        states.clone(),
        parameters.clone(),
        "x",
        SymbolicIvpGeneratedBackendConfig::new()
            .with_output_parent_dir(Some(root.path().join("rebuild-always")))
            .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Debug,
            })
            .with_c_tcc(),
        BvpSciTelemetry::detailed(),
    )
    .expect("RebuildAlways should prepare");
    let rebuild_snapshot = rebuild
        .telemetry_snapshot()
        .aot
        .expect("rebuild AOT telemetry");
    assert!(rebuild_snapshot.aot_build_attempts >= 1);
    rows.push(LifecycleRow {
        policy: "RebuildAlways".into(),
        cache_hits: rebuild_snapshot.aot_resolution_hits,
        cache_misses: rebuild_snapshot.aot_resolution_misses,
        build_attempts: rebuild_snapshot.aot_build_attempts,
        build_successes: rebuild_snapshot.aot_build_successes,
        link_attempts: rebuild_snapshot.aot_link_attempts,
        link_successes: rebuild_snapshot.aot_link_successes,
        runtime_ready: rebuild_snapshot.aot_runtime_ready,
        artifact_key: rebuild_snapshot
            .aot_artifact_keys
            .first()
            .cloned()
            .unwrap_or_else(|| "-".into()),
        status: "rebuilt".into(),
    });

    let missing = BvpSciLambdifyPlan::prepare_aot(
        BvpSciAssembly::ExprLegacy,
        layout,
        vec![Expr::parse_expression("p*y + 0.123456789")],
        vec!["y".into()],
        vec!["p".into()],
        "x",
        SymbolicIvpGeneratedBackendConfig::require_prebuilt()
            .with_output_parent_dir(Some(root.path().join("missing"))),
        BvpSciTelemetry::counters(),
    );
    let missing = match missing {
        Ok(_) => panic!("RequirePrebuilt must reject an unknown artifact"),
        Err(error) => error,
    };
    assert!(matches!(missing, BvpSciNewError::AotPreparation { .. }));
    rows.push(LifecycleRow {
        policy: "RequirePrebuilt/missing".into(),
        cache_hits: 0,
        cache_misses: 0,
        build_attempts: 0,
        build_successes: 0,
        link_attempts: 0,
        link_successes: 0,
        runtime_ready: 0,
        artifact_key: "-".into(),
        status: "typed-missing".into(),
    });

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci AOT lifecycle] cache/build/link/publication contract",
        &rows,
    );
}

#[test]
#[ignore = "long AOT continuation/restart and retention release gate"]
fn aot_continuation_restart_retention_matrix_is_compact() {
    #[derive(Debug, Tabled)]
    struct RetentionRow {
        frontend: String,
        layout: String,
        continuation_count: usize,
        parameter_rebinds: u64,
        continuation_solves: u64,
        restarts: u64,
        symbolic_preparation_before: String,
        symbolic_preparation_after: String,
        workspace_resizes_same_mesh: u64,
        workspace_resizes_after_mesh_change: u64,
        allocations_same_mesh: u64,
        allocations_after_mesh_change: u64,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_AOT",
        "aot_continuation_restart_retention_matrix",
    );
    let count = std::env::var("BVP_SCI_AOT_CONTINUATION_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16)
        .max(1);
    let root = tempfile::tempdir().expect("AOT continuation root");
    let layouts = [
        ("Dense", BvpSciMatrixLayout::Dense),
        ("Sparse", BvpSciMatrixLayout::Sparse),
        ("Banded", BvpSciMatrixLayout::Banded { lower: 0, upper: 0 }),
    ];
    let frontends = [
        ("ExprLegacy", BvpSciAssembly::ExprLegacy),
        ("AtomViewNative", BvpSciAssembly::AtomViewNative),
    ];
    let mut rows = Vec::new();
    for (frontend_name, frontend) in frontends {
        for (layout_name, layout) in layouts {
            let plan = BvpSciLambdifyPlan::prepare_aot(
                frontend,
                layout,
                vec![Expr::parse_expression("p*y")],
                vec!["y".into()],
                vec!["p".into()],
                "x",
                SymbolicIvpGeneratedBackendConfig::new()
                    .with_output_parent_dir(Some(root.path().join(frontend_name).join(layout_name)))
                    .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Debug,
                    })
                    .with_c_tcc(),
                BvpSciTelemetry::detailed(),
            )
            .expect("continuation AOT plan should prepare");
            let boundary = BvpSciBoundaryCallbacks::new(
                2,
                |ya, yb, _, output| {
                    output[0] = ya[0];
                    output[1] = yb[0];
                    Ok(())
                },
                BvpSciTelemetry::disabled(),
            );
            let mut options = BvpSciOptions::default();
            options.execution = BvpSciExecution::Aot;
            options.assembly = Some(frontend);
            options.matrix_layout = layout;
            options.max_nodes = 64;
            options.max_mesh_refinements = 0;
            options.tolerance = 1e-8;
            let mut solver = BvpSciSolver::new(
                plan,
                boundary,
                vec![0.0, 0.5, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![1.0],
                options,
            )
            .expect("continuation AOT solver should construct");
            solver.solve().expect("initial continuation solve");
            let before = solver.plan().telemetry_snapshot();
            let before_aot = before.aot.clone().expect("AOT retention telemetry");
            for index in 0..count {
                solver
                    .set_parameters(vec![1.0 + index as f64 * 0.01])
                    .expect("parameter rebind should preserve prepared model");
                let solution = solver.solve().expect("continuation solve should converge");
                assert!(solution.y.iter().all(|value| value.abs() < 1e-10));
            }
            let after_series = solver.plan().telemetry_snapshot();
            let after_series_aot = after_series.aot.clone().expect("AOT series telemetry");
            assert_eq!(before_aot.cold, after_series_aot.cold);
            assert_eq!(
                after_series.workspace_resizes, before.workspace_resizes,
                "same-mesh continuation must not resize workspace"
            );
            assert_eq!(
                after_series.allocations, before.allocations,
                "same-mesh continuation must not grow logical workspace"
            );
            let same_mesh_resizes = after_series
                .workspace_resizes
                .saturating_sub(before.workspace_resizes);
            let same_mesh_allocations = after_series.allocations.saturating_sub(before.allocations);

            solver
                .restart(vec![0.0, 0.25, 0.5, 0.75, 1.0], vec![0.0; 5])
                .expect("restart with a new mesh should preserve prepared model");
            solver.solve().expect("restarted continuation solve");
            let after_restart = solver.plan().telemetry_snapshot();
            let after_restart_aot = after_restart.aot.clone().expect("AOT restart telemetry");
            assert_eq!(before_aot.cold, after_restart_aot.cold);
            assert_eq!(after_restart.restarts, before.restarts + 1);
            assert!(
                after_restart
                    .workspace_resizes
                    .saturating_sub(before.workspace_resizes)
                    <= 1,
                "mesh restart should perform at most one numerical resize"
            );
            assert!(
                after_restart.allocations.saturating_sub(before.allocations) <= 1,
                "mesh restart should perform at most one logical allocation"
            );
            rows.push(RetentionRow {
                frontend: frontend_name.into(),
                layout: layout_name.into(),
                continuation_count: count,
                parameter_rebinds: after_restart.parameter_rebinds,
                continuation_solves: after_restart.continuation_solves,
                restarts: after_restart.restarts,
                symbolic_preparation_before: format!(
                    "cold_calls={}",
                    before_aot.cold.iter().map(|stage| stage.calls).sum::<u64>()
                ),
                symbolic_preparation_after: format!(
                    "cold_calls={}",
                    after_restart_aot
                        .cold
                        .iter()
                        .map(|stage| stage.calls)
                        .sum::<u64>()
                ),
                workspace_resizes_same_mesh: same_mesh_resizes,
                workspace_resizes_after_mesh_change: after_restart
                    .workspace_resizes
                    .saturating_sub(before.workspace_resizes),
                allocations_same_mesh: same_mesh_allocations,
                allocations_after_mesh_change: after_restart
                    .allocations
                    .saturating_sub(before.allocations),
                status: "ok".into(),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci AOT continuation] symbolic reuse, restart and logical workspace retention",
        &rows,
    );
}

const BVP_SCI_PROCESS_CHILD: &str =
    "numerical::BVP_sci::new::story_tests::aot_process_handoff_child";

fn run_bvp_sci_process_child(
    phase: &str,
    frontend: &str,
    layout: &str,
    output_dir: &Path,
) -> Output {
    let executable = std::env::current_exe().expect("BVP_sci test executable");
    let mut child = Command::new(executable)
        .arg("--exact")
        .arg(BVP_SCI_PROCESS_CHILD)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("BVP_SCI_PROCESS_PHASE", phase)
        .env("BVP_SCI_PROCESS_FRONTEND", frontend)
        .env("BVP_SCI_PROCESS_LAYOUT", layout)
        .env("BVP_SCI_PROCESS_OUTPUT", output_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("spawn BVP_sci {phase} child: {error}"));
    let started = Instant::now();
    loop {
        if child
            .try_wait()
            .expect("poll BVP_sci process child")
            .is_some()
        {
            return child
                .wait_with_output()
                .expect("collect BVP_sci child output");
        }
        if started.elapsed() > Duration::from_secs(120) {
            let _ = child.kill();
            panic!("BVP_sci process child timed out: {phase}/{frontend}/{layout}");
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn process_metric(output: &Output, key: &str) -> Option<String> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| line.contains("BVP_SCI_PROCESS "))
        .and_then(|line| {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix(&format!("{key}=")))
        })
        .map(str::to_owned)
}

#[test]
#[ignore = "process-isolated AOT producer/consumer handoff release gate"]
fn aot_process_isolated_producer_consumer_handoff_matrix_is_compact() {
    #[derive(Debug, Tabled)]
    struct HandoffRow {
        frontend: String,
        layout: String,
        producer_builds: String,
        producer_links: String,
        consumer_builds: String,
        consumer_links: String,
        consumer_reconnects: String,
        artifact_key: String,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_AOT_Process",
        "aot_process_isolated_producer_consumer_handoff_matrix",
    );
    let root = tempfile::tempdir().expect("BVP_sci process handoff root");
    let mut rows = Vec::new();
    for frontend in ["expr-legacy", "atom-native"] {
        for layout in ["dense", "sparse", "banded"] {
            let route = root.path().join(format!("{frontend}-{layout}"));
            std::fs::create_dir_all(&route).expect("process route directory");
            let producer = run_bvp_sci_process_child("producer", frontend, layout, &route);
            assert!(
                producer.status.success(),
                "producer failed: stdout={} stderr={}",
                String::from_utf8_lossy(&producer.stdout),
                String::from_utf8_lossy(&producer.stderr)
            );
            let consumer = run_bvp_sci_process_child("consumer", frontend, layout, &route);
            assert!(
                consumer.status.success(),
                "consumer failed: stdout={} stderr={}",
                String::from_utf8_lossy(&consumer.stdout),
                String::from_utf8_lossy(&consumer.stderr)
            );
            assert_eq!(process_metric(&consumer, "builds").as_deref(), Some("0"));
            rows.push(HandoffRow {
                frontend: frontend.into(),
                layout: layout.into(),
                producer_builds: process_metric(&producer, "builds").unwrap_or_else(|| "-".into()),
                producer_links: process_metric(&producer, "links").unwrap_or_else(|| "-".into()),
                consumer_builds: process_metric(&consumer, "builds").unwrap_or_else(|| "-".into()),
                consumer_links: process_metric(&consumer, "links").unwrap_or_else(|| "-".into()),
                consumer_reconnects: process_metric(&consumer, "reconnects")
                    .unwrap_or_else(|| "-".into()),
                artifact_key: process_metric(&consumer, "key").unwrap_or_else(|| "-".into()),
                status: "ok".into(),
            });
        }
    }
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci process-isolated AOT handoff] producer/consumer matrix",
        &rows,
    );
}

#[test]
fn aot_process_handoff_child() {
    let Some(phase) = std::env::var_os("BVP_SCI_PROCESS_PHASE") else {
        return;
    };
    let phase = phase.to_string_lossy();
    let frontend = std::env::var("BVP_SCI_PROCESS_FRONTEND").expect("child frontend");
    let layout_name = std::env::var("BVP_SCI_PROCESS_LAYOUT").expect("child layout");
    let output = PathBuf::from(std::env::var_os("BVP_SCI_PROCESS_OUTPUT").expect("child output"));
    let assembly = match frontend.as_str() {
        "expr-legacy" => BvpSciAssembly::ExprLegacy,
        "atom-native" => BvpSciAssembly::AtomViewNative,
        other => panic!("unknown BVP_sci child frontend {other}"),
    };
    let layout = match layout_name.as_str() {
        "dense" => BvpSciMatrixLayout::Dense,
        "sparse" => BvpSciMatrixLayout::Sparse,
        "banded" => BvpSciMatrixLayout::Banded { lower: 0, upper: 0 },
        other => panic!("unknown BVP_sci child layout {other}"),
    };
    let handoff = output.join("bvp-sci-process-handoff.txt");
    let config = if phase == "producer" {
        SymbolicIvpGeneratedBackendConfig::new()
            .with_output_parent_dir(Some(output.clone()))
            .with_handoff_path(Some(handoff.clone()))
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug,
            })
            .with_c_tcc()
    } else {
        SymbolicIvpGeneratedBackendConfig::require_prebuilt()
            .with_output_parent_dir(Some(output))
            .with_handoff_path(Some(handoff.clone()))
            .with_c_tcc()
    };
    let plan = BvpSciLambdifyPlan::prepare_aot(
        assembly,
        layout,
        vec![Expr::parse_expression("p*y")],
        vec!["y".into()],
        vec!["p".into()],
        "x",
        config,
        BvpSciTelemetry::counters(),
    )
    .unwrap_or_else(|error| panic!("BVP_sci {phase} preparation failed: {error}"));
    let aot = plan.telemetry_snapshot().aot.expect("child AOT telemetry");
    aot.validate_contract().expect("child AOT contract");
    if phase == "producer" {
        assert!(aot.aot_build_attempts >= 1);
        assert!(handoff.exists());
    } else {
        assert_eq!(aot.aot_build_attempts, 0);
        assert!(aot.aot_runtime_ready >= 1);
    }
    println!(
        "BVP_SCI_PROCESS {phase} frontend={frontend} layout={layout_name} builds={} links={} reconnects={} key={} status=ok",
        aot.aot_build_attempts,
        aot.aot_link_attempts,
        aot.aot_reconnects,
        aot.aot_artifact_keys
            .first()
            .map(String::as_str)
            .unwrap_or("-"),
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_linear_reference_matches_rust_on_common_evaluation_grid() {
    #[derive(Tabled)]
    struct ParityRow {
        case: &'static str,
        scipy_status: String,
        rust_status: String,
        sample_points: usize,
        scipy_jac_lu_solve: String,
        rust_jac_lu_solve: String,
        max_abs_diff: String,
        status: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fields = scipy_bvp_reference_row(&root, "linear");
    assert_eq!(fields.get("case").map(String::as_str), Some("linear"));
    assert_eq!(fields.get("success").map(String::as_str), Some("1"));
    assert_eq!(fields.get("status").map(String::as_str), Some("0"));
    let scipy_values = fields["sample_y"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy sample value"))
        .collect::<Vec<_>>();
    let sample_count = scipy_values.len();
    let samples = (0..sample_count)
        .map(|index| index as f64 / (sample_count - 1) as f64)
        .collect::<Vec<_>>();

    let mut rust = build_damp_exact_solver(
        DampExactWorkload::Linear,
        BvpSciAssembly::ExprLegacy,
        BvpSciMatrixLayout::Dense,
    );
    let rust_solution = rust.solve().expect("Rust linear BVP should converge");
    let mut max_abs_diff = 0.0_f64;
    for (x, expected) in samples.iter().zip(scipy_values.iter()) {
        let actual = interpolate_node_major(&rust_solution.x, &rust_solution.y, 1, *x)[0];
        max_abs_diff = max_abs_diff.max((actual - expected).abs());
    }
    assert!(
        max_abs_diff <= 1e-8,
        "SciPy/Rust sampled drift={max_abs_diff:e}"
    );
    let rust_telemetry = rust.plan().telemetry_snapshot();
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] common-grid numerical gate",
        &[ParityRow {
            case: "linear y'=1",
            scipy_status: fields["status"].to_owned(),
            rust_status: format!("{}", rust_solution.status.code()),
            sample_points: sample_count,
            scipy_jac_lu_solve: format!(
                "{}/{}/{}",
                fields["jacobian_builds"], fields["factorizations"], fields["linear_solves"]
            ),
            rust_jac_lu_solve: format!(
                "{}/{}/{}",
                rust_telemetry.newton_jacobian_refreshes,
                rust_telemetry.factorizations,
                rust_telemetry.linear_solves
            ),
            max_abs_diff: format!("{max_abs_diff:.3e}"),
            status: "ok",
        }],
    );
}

fn scipy_bvp_reference_row(root: &Path, case: &str) -> HashMap<String, String> {
    let python = std::env::var_os("BVP_SCIPY_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let local = root.join(".venv-gavin-win/Scripts/python.exe");
            if local.exists() {
                local
            } else {
                PathBuf::from("python")
            }
        });
    let runner = root.join("src/numerical/BVP_sci/scipy_reference/bvp_parity.py");
    let output = Command::new(&python)
        .arg(&runner)
        .arg(case)
        .arg("--format")
        .arg("tsv")
        .output()
        .unwrap_or_else(|error| panic!("failed to launch SciPy {case} reference: {error}"));
    assert!(
        output.status.success(),
        "SciPy {case} reference failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .split('\t')
        .filter_map(|field| field.split_once('='))
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect()
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_invalid_inputs_map_to_typed_rust_errors() {
    #[derive(Tabled)]
    struct ValidationRow {
        input: &'static str,
        scipy_exception: String,
        rust_error: &'static str,
        result: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "validation");
    let expected: HashMap<_, _> = reference["validation_failures"]
        .split(',')
        .filter_map(|field| field.split_once(':'))
        .collect();

    let plan = || {
        let numerical = BvpSciNumericalPlan::new(
            1,
            0,
            |_x, _y, _parameters, output| {
                output[0] = 1.0;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        )
        .expect("validation numerical plan");
        BvpSciLambdifyPlan::prepare_numerical(numerical)
    };
    let boundary = || {
        BvpSciBoundaryCallbacks::new(
            1,
            |ya, _yb, _parameters, output| {
                output[0] = ya[0];
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        )
    };
    let options = BvpSciOptions::default();
    let invalid_mesh = BvpSciSolver::new(
        plan(),
        boundary(),
        vec![0.0, 0.5, 0.4],
        vec![0.0; 3],
        vec![],
        options.clone(),
    )
    .err()
    .expect("non-increasing mesh must be rejected");
    assert!(matches!(
        invalid_mesh,
        BvpSciNewError::InvalidConfiguration(_)
    ));
    let invalid_state_shape = BvpSciSolver::new(
        plan(),
        boundary(),
        vec![0.0, 0.5, 1.0],
        vec![0.0; 2],
        vec![],
        options,
    )
    .err()
    .expect("state/mesh shape mismatch must be rejected");
    assert!(matches!(
        invalid_state_shape,
        BvpSciNewError::ShapeMismatch {
            stage: BvpSciStage::NumericalCore,
            ..
        }
    ));
    let invalid_boundary_shape = BvpSciSolver::new(
        plan(),
        BvpSciBoundaryCallbacks::new(
            0,
            |_ya, _yb, _parameters, _output| Ok(()),
            BvpSciTelemetry::disabled(),
        ),
        vec![0.0, 0.5, 1.0],
        vec![0.0; 3],
        vec![],
        BvpSciOptions::default(),
    )
    .err()
    .expect("boundary dimension mismatch must be rejected");
    assert!(matches!(
        invalid_boundary_shape,
        BvpSciNewError::ShapeMismatch {
            stage: BvpSciStage::BoundaryCallback,
            ..
        }
    ));

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] official input validation categories",
        &[
            ValidationRow {
                input: "non-increasing mesh",
                scipy_exception: expected["non_increasing_mesh"].into(),
                rust_error: "InvalidConfiguration",
                result: "rejected",
            },
            ValidationRow {
                input: "state/mesh shape mismatch",
                scipy_exception: expected["state_shape"].into(),
                rust_error: "ShapeMismatch(NumericalCore)",
                result: "rejected",
            },
            ValidationRow {
                input: "boundary residual shape mismatch",
                scipy_exception: expected["boundary_shape"].into(),
                rust_error: "ShapeMismatch(BoundaryCallback)",
                result: "rejected",
            },
        ],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_global_collocation_jacobian_matches_entry_by_entry() {
    #[derive(Tabled)]
    struct JacobianParityRow {
        shape: String,
        entries: usize,
        max_matrix_diff: String,
        max_residual_diff: String,
        result: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "global-jacobian");
    let scipy_jacobian = reference["global_jacobian"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy Jacobian entry"))
        .collect::<Vec<_>>();
    let scipy_residual = reference["collocation_residual"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy collocation residual"))
        .collect::<Vec<_>>();
    let dimension = 2usize;
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("z"), Expr::parse_expression("p*y")],
        &["y".into(), "z".into()],
        &["p".into()],
        "x",
        BvpSciTelemetry::disabled(),
    )
    .expect("direct Jacobian parity plan");
    let boundary = BvpSciBoundaryCallbacks::new_with_jacobian(
        dimension + 1,
        |ya, yb, parameters, output| {
            output[0] = ya[0];
            output[1] = yb[1];
            output[2] = ya[1] - parameters[0];
            Ok(())
        },
        |_, _, _, dya, dyb, dp| {
            dya.copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            dyb.copy_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            dp.copy_from_slice(&[0.0, 0.0, -1.0]);
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh = [0.0, 0.4, 1.0];
    let state_node_major = [1.0, 0.25, 0.9, 0.3, 0.7, -0.1];
    let parameters = [1.3];
    let mut workspace = super::workspace::BvpSciCollocationWorkspace::new(
        dimension,
        mesh.len(),
        parameters.len(),
        &BvpSciTelemetry::disabled(),
    )
    .expect("direct Jacobian parity workspace");
    super::collocation::evaluate_collocation(
        &plan,
        &mesh,
        &state_node_major,
        &parameters,
        None,
        &mut workspace,
    )
    .expect("Rust collocation residual");
    super::jacobian::assemble_global_jacobian(
        &plan,
        &boundary,
        &mesh,
        &state_node_major,
        &parameters,
        None,
        &mut workspace,
    )
    .expect("Rust global Jacobian");

    let system_size = dimension * mesh.len() + parameters.len();
    assert_eq!(scipy_jacobian.len(), system_size * system_size);
    let mut rust_jacobian = vec![0.0; system_size * system_size];
    for &(row, column, value) in &workspace.entries {
        rust_jacobian[row * system_size + column] += value;
    }
    let max_matrix_diff = rust_jacobian
        .iter()
        .zip(&scipy_jacobian)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let max_residual_diff = workspace
        .collocation_residual
        .iter()
        .zip(&scipy_residual)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_matrix_diff < 1e-12,
        "global matrix drift={max_matrix_diff:e}"
    );
    assert!(
        max_residual_diff < 1e-12,
        "collocation residual drift={max_residual_diff:e}"
    );

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] direct global collocation Jacobian",
        &[JacobianParityRow {
            shape: format!("{system_size}x{system_size}"),
            entries: scipy_jacobian.len(),
            max_matrix_diff: format!("{max_matrix_diff:.3e}"),
            max_residual_diff: format!("{max_residual_diff:.3e}"),
            result: "entrywise-match",
        }],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_refined_bratu_matches_dense_output_and_interval_defects() {
    #[derive(Tabled)]
    struct RefinedOutputRow {
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        max_value_diff: String,
        max_derivative_diff: String,
        max_interval_residual_diff: String,
        counters: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "bratu-refine");
    let expected_values = parse_scipy_samples(&reference);
    let expected_derivatives = parse_scipy_derivatives(&reference);
    let expected_residuals = reference["rms_residuals"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy interval residual"))
        .collect::<Vec<_>>();
    let sample_x = reference["sample_x"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy sample coordinate"))
        .collect::<Vec<_>>();
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &bratu_equations(),
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("refined Bratu parity plan");
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..6).map(|index| index as f64 / 5.0).collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-8;
    options.max_nodes = 200;
    options.max_mesh_refinements = 20;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![0.0; mesh.len() * 2],
        vec![],
        options,
    )
    .expect("refined Bratu solver");
    let solution = rust.solve().expect("refined Bratu should converge");
    let dense = solution.dense_output().expect("refined Bratu dense output");
    let telemetry = rust.plan().telemetry_snapshot();
    assert_eq!(reference["status"], "0");
    assert_eq!(solution.status.code(), 0);
    assert_eq!(
        solution.x.len().to_string(),
        reference["nodes"],
        "adaptive mesh nodes must match for interval-residual parity"
    );
    assert_scipy_counter_parity(
        "refined Bratu",
        &reference,
        &telemetry,
        telemetry.newton_scope_calls,
    );

    let mut max_value_diff = 0.0_f64;
    let mut max_derivative_diff = 0.0_f64;
    for (point, x) in sample_x.iter().enumerate() {
        let values = dense.evaluate(*x).expect("refined Bratu value");
        let derivatives = dense
            .evaluate_derivative(*x)
            .expect("refined Bratu derivative");
        for component in 0..values.len() {
            max_value_diff =
                max_value_diff.max((values[component] - expected_values[component][point]).abs());
            max_derivative_diff = max_derivative_diff
                .max((derivatives[component] - expected_derivatives[component][point]).abs());
        }
    }
    let max_interval_residual_diff = solution
        .interval_residuals
        .iter()
        .zip(&expected_residuals)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_value_diff < 1e-7,
        "Bratu refined value drift={max_value_diff:e}"
    );
    assert!(
        max_derivative_diff < 1e-6,
        "Bratu refined derivative drift={max_derivative_diff:e}"
    );
    assert_eq!(solution.interval_residuals.len(), expected_residuals.len());
    assert!(
        max_interval_residual_diff < 1e-10,
        "Bratu refined RMS defect drift={max_interval_residual_diff:e}"
    );
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] refined Bratu dense output and defects",
        &[RefinedOutputRow {
            scipy_status: reference["status"].clone(),
            rust_status: solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: solution.x.len().to_string(),
            max_value_diff: format!("{max_value_diff:.3e}"),
            max_derivative_diff: format!("{max_derivative_diff:.3e}"),
            max_interval_residual_diff: format!("{max_interval_residual_diff:.3e}"),
            counters: "exact",
        }],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_boundary_tolerance_exhaustion_maps_to_rust_status_three() {
    #[derive(Tabled)]
    struct StatusThreeRow {
        scipy_status: String,
        rust_status: String,
        scipy_passes: String,
        rust_passes: String,
        scipy_factorizations: String,
        rust_factorizations: String,
        scipy_mesh_changes: String,
        rust_mesh_changes: String,
        mapping: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "boundary-tolerance-exhaustion");
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("0")],
        &["y".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("status-three parity plan");
    let boundary_tolerance = f64::EPSILON / 16.0;
    let boundary = BvpSciBoundaryCallbacks::new(
        1,
        |ya, _, _, output| {
            output[0] = (ya[0] - 1.0) + f64::EPSILON / 4.0;
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..5).map(|index| index as f64 / 4.0).collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-3;
    options.boundary_tolerance = Some(boundary_tolerance);
    options.max_nodes = 100;
    options.max_mesh_refinements = 10;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![1.0; mesh.len()],
        vec![],
        options,
    )
    .expect("status-three parity solver");
    let error = rust
        .solve()
        .expect_err("sub-ulp boundary residual cannot be corrected");
    let telemetry = rust.plan().telemetry_snapshot();
    let rust_status = error.status().map(|status| status.code());
    assert_eq!(reference["status"], "3");
    assert_eq!(rust_status, Some(3), "Rust typed failure: {error:?}");
    assert_scipy_counter_parity(
        "boundary tolerance exhaustion",
        &reference,
        &telemetry,
        telemetry.newton_scope_calls,
    );

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] boundary-tolerance exhaustion status 3",
        &[StatusThreeRow {
            scipy_status: reference["status"].clone(),
            rust_status: rust_status.unwrap().to_string(),
            scipy_passes: reference["newton_passes"].clone(),
            rust_passes: telemetry.newton_scope_calls.to_string(),
            scipy_factorizations: reference["factorizations"].clone(),
            rust_factorizations: telemetry.factorizations.to_string(),
            scipy_mesh_changes: reference["mesh_modify_calls"].clone(),
            rust_mesh_changes: telemetry.mesh_refinements.to_string(),
            mapping: "status and counters match",
        }],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_official_shock_layer_matches_refined_rust_solution() {
    #[derive(Tabled)]
    struct ShockParityRow {
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_passes_jac_lu_solves: String,
        rust_passes_jac_lu_solves: String,
        scipy_linear_solves: String,
        rust_linear_solves: String,
        scipy_trial_trace: String,
        scipy_convergence_trace: String,
        rust_trial_trace: String,
        counter_difference_reason: &'static str,
        max_value_diff: String,
        max_derivative_diff: String,
        max_interval_residual_diff: String,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "shock-layer");
    let expected_values = parse_scipy_samples(&reference);
    let expected_derivatives = parse_scipy_derivatives(&reference);
    let expected_residuals = reference["rms_residuals"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy shock residual"))
        .collect::<Vec<_>>();
    let sample_x = reference["sample_x"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy shock sample"))
        .collect::<Vec<_>>();
    let epsilon = 1e-3;
    let plan = BvpSciNumericalPlan::new(
        2,
        0,
        move |x, y, _, output| {
            output[0] = y[1];
            output[1] = -(x * y[1]
                + epsilon * std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x).cos()
                + std::f64::consts::PI * x * (std::f64::consts::PI * x).sin())
                / epsilon;
            Ok(())
        },
        BvpSciTelemetry::detailed(),
    )
    .expect("shock-layer numerical plan");
    let plan = BvpSciLambdifyPlan::prepare_numerical(plan);
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0] + 2.0;
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..5).map(|index| -1.0 + 0.5 * index as f64).collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-3;
    options.max_nodes = 1000;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![0.0; mesh.len() * 2],
        vec![],
        options,
    )
    .expect("shock-layer solver");
    let solution = rust.solve().expect("shock-layer should converge");
    let dense = solution.dense_output().expect("shock-layer dense output");
    let telemetry = rust.plan().telemetry_snapshot();
    assert_eq!(reference["status"], "0");
    assert_eq!(solution.status.code(), 0);
    assert_eq!(solution.x.len().to_string(), reference["nodes"]);
    for (key, actual) in [
        ("newton_passes", telemetry.newton_scope_calls),
        ("jacobian_builds", telemetry.newton_jacobian_refreshes),
        ("factorizations", telemetry.factorizations),
        ("mesh_modify_calls", telemetry.mesh_refinements),
    ] {
        assert_eq!(
            reference[key].parse::<u64>().expect("SciPy counter"),
            actual,
            "official shock-layer {key} counter differs"
        );
    }
    let scipy_linear_solves = reference["linear_solves"]
        .parse::<u64>()
        .expect("SciPy linear-solve count");
    assert_eq!(
        telemetry.linear_solves,
        scipy_linear_solves + 1,
        "shock-layer's documented near-threshold branch adds one fixed-Jacobian solve"
    );
    let scipy_convergence = reference["newton_convergence_trace"]
        .split(';')
        .collect::<Vec<_>>();
    assert_eq!(scipy_convergence.len(), 8);
    let scipy_boundary_case_ratio = scipy_convergence[2]
        .split(':')
        .find_map(|field| field.strip_prefix('r'))
        .expect("SciPy normalized residual ratio")
        .parse::<f64>()
        .expect("SciPy normalized residual ratio");
    let rust_trace = &telemetry.newton_residual_history;
    assert_eq!(rust_trace.len(), scipy_convergence.len() + 1);
    assert!(
        scipy_boundary_case_ratio < 1.0
            && rust_trace[2].max_normalized_collocation_residual > 1.0
            && rust_trace[3].max_normalized_collocation_residual < 1.0,
        "the additional solve must be explained by crossing the same strict residual threshold"
    );
    let mut max_value_diff = 0.0_f64;
    let mut max_derivative_diff = 0.0_f64;
    for (point, x) in sample_x.iter().enumerate() {
        let values = dense.evaluate(*x).expect("shock-layer value");
        let derivatives = dense
            .evaluate_derivative(*x)
            .expect("shock-layer derivative");
        for component in 0..values.len() {
            max_value_diff =
                max_value_diff.max((values[component] - expected_values[component][point]).abs());
            max_derivative_diff = max_derivative_diff
                .max((derivatives[component] - expected_derivatives[component][point]).abs());
        }
    }
    let max_interval_residual_diff = solution
        .interval_residuals
        .iter()
        .zip(&expected_residuals)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_value_diff < 1e-5,
        "shock-layer value drift={max_value_diff:e}"
    );
    assert!(
        max_derivative_diff < 1e-3,
        "shock-layer derivative drift={max_derivative_diff:e}"
    );
    assert_eq!(solution.interval_residuals.len(), expected_residuals.len());
    assert!(
        max_interval_residual_diff < 1e-10,
        "shock-layer RMS defect drift={max_interval_residual_diff:e}"
    );
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] official shock-layer workload",
        &[ShockParityRow {
            scipy_status: reference["status"].clone(),
            rust_status: solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: solution.x.len().to_string(),
            scipy_passes_jac_lu_solves: format!(
                "{}/{}/{}/{}",
                reference["newton_passes"],
                reference["jacobian_builds"],
                reference["factorizations"],
                reference["linear_solves"]
            ),
            rust_passes_jac_lu_solves: format!(
                "{}/{}/{}/{}",
                telemetry.newton_scope_calls,
                telemetry.newton_jacobian_refreshes,
                telemetry.factorizations,
                telemetry.linear_solves
            ),
            scipy_linear_solves: reference["linear_solves"].clone(),
            rust_linear_solves: telemetry.linear_solves.to_string(),
            scipy_trial_trace: reference["newton_trace"]
                .split(';')
                .filter(|entry| !entry.is_empty())
                .collect::<Vec<_>>()
                .join(";"),
            scipy_convergence_trace: reference["newton_convergence_trace"]
                .split(';')
                .filter(|entry| !entry.is_empty())
                .collect::<Vec<_>>()
                .join(";"),
            rust_trial_trace: telemetry
                .newton_residual_history
                .iter()
                .map(|entry| {
                    format!(
                        "i{}:j{}:c{:.3e}>{:.3e}:r{:.3e}:b{:.3e}:a{:.3}:t{}",
                        entry.iteration,
                        u8::from(entry.jacobian_refreshed),
                        entry.affine_cost_before,
                        entry.affine_cost_after.unwrap_or(f64::NAN),
                        entry.max_normalized_collocation_residual,
                        entry.normalized_boundary_residual,
                        entry.armijo_alpha,
                        entry.backtracking_trials
                    )
                })
                .collect::<Vec<_>>()
                .join(";"),
            counter_difference_reason: "nalgebra/SuperLU roundoff crosses threshold; one extra solve",
            max_value_diff: format!("{max_value_diff:.3e}"),
            max_derivative_diff: format!("{max_derivative_diff:.3e}"),
            max_interval_residual_diff: format!("{max_interval_residual_diff:.3e}"),
        }],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_official_large_parameterized_problem_matches_rust() {
    #[derive(Tabled)]
    struct LargeParameterizedRow {
        dimension: usize,
        parameter_count: usize,
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_parameters: String,
        rust_parameters: String,
        scipy_passes_jac_lu_solves: String,
        rust_passes_jac_lu_solves: String,
        scipy_per_mesh_work: String,
        rust_per_mesh_work: String,
        max_value_diff: String,
        max_derivative_diff: String,
        max_interval_residual_diff: String,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "big-parameterized-analytic");
    let expected_values = parse_scipy_samples(&reference);
    let expected_derivatives = parse_scipy_derivatives(&reference);
    let expected_residuals = reference["rms_residuals"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy large residual"))
        .collect::<Vec<_>>();
    let sample_x = reference["sample_x"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy large sample"))
        .collect::<Vec<_>>();
    let systems = 30usize;
    let dimension = systems * 2;
    let telemetry = BvpSciTelemetry::detailed();
    let rhs = BvpSciNumericalPlan::new(
        dimension,
        2,
        move |_x, state, parameters, output| {
            for system in 0..systems {
                let parameter = if system % 2 == 0 { 0 } else { 1 };
                output[2 * system] = state[2 * system + 1];
                output[2 * system + 1] =
                    -(parameters[parameter] * parameters[parameter]) * state[2 * system];
            }
            Ok(())
        },
        telemetry.clone(),
    )
    .expect("large parameterized RHS plan")
    .with_rhs_jacobian(move |_x, _state, parameters, output| {
        output.fill(0.0);
        for system in 0..systems {
            let parameter = if system % 2 == 0 { 0 } else { 1 };
            output[(2 * system) * dimension + (2 * system + 1)] = 1.0;
            output[(2 * system + 1) * dimension + 2 * system] =
                -(parameters[parameter] * parameters[parameter]);
        }
        Ok(())
    })
    .with_rhs_parameter_jacobian(move |_x, state, parameters, output| {
        output.fill(0.0);
        for system in 0..systems {
            let parameter = if system % 2 == 0 { 0 } else { 1 };
            output[(2 * system + 1) * 2 + parameter] =
                -2.0 * parameters[parameter] * state[2 * system];
        }
        Ok(())
    });
    let plan = BvpSciLambdifyPlan::prepare_numerical(rhs);
    let boundary = BvpSciBoundaryCallbacks::new_with_jacobian(
        dimension + 2,
        move |ya, yb, parameters, output| {
            for system in 0..systems {
                output[system] = ya[2 * system];
                output[systems + system] = yb[2 * system];
            }
            output[dimension] = ya[1] - parameters[0];
            output[dimension + 1] = ya[3] - parameters[1];
            Ok(())
        },
        move |_ya, _yb, _parameters, dya, dyb, dp| {
            dya.fill(0.0);
            dyb.fill(0.0);
            dp.fill(0.0);
            for system in 0..systems {
                dya[system * dimension + 2 * system] = 1.0;
                dyb[(systems + system) * dimension + 2 * system] = 1.0;
            }
            dya[dimension * dimension + 1] = 1.0;
            dya[(dimension + 1) * dimension + 3] = 1.0;
            dp[dimension * 2] = -1.0;
            dp[(dimension + 1) * 2 + 1] = -1.0;
            Ok(())
        },
        telemetry,
    );
    let mesh: Vec<f64> = (0..5)
        .map(|index| std::f64::consts::PI * index as f64 / 4.0)
        .collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-3;
    options.max_nodes = 1000;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![1.0; mesh.len() * dimension],
        vec![0.5, 0.5],
        options,
    )
    .expect("large parameterized solver");
    let solution = rust
        .solve()
        .expect("official large parameterized system should converge");
    let dense = solution
        .dense_output()
        .expect("large parameterized dense output");
    let telemetry = rust.plan().telemetry_snapshot();
    assert_eq!(reference["status"], "0");
    assert_eq!(solution.status.code(), 0);
    assert_eq!(solution.x.len().to_string(), reference["nodes"]);
    assert!(telemetry.rhs_calls > 0);
    assert_eq!(telemetry.jacobian_calls, telemetry.parameter_jacobian_calls);
    assert!(telemetry.boundary_calls > 0);
    assert!(telemetry.boundary_jacobian_calls > 0);
    assert!(telemetry.rhs_callback_ms.is_some());
    assert!(telemetry.jacobian_callback_ms.is_some());
    assert!(telemetry.parameter_jacobian_callback_ms.is_some());
    assert!(telemetry.boundary_callback_ms.is_some());
    assert!(telemetry.boundary_jacobian_callback_ms.is_some());
    assert_scipy_counter_parity(
        "official large parameterized problem",
        &reference,
        &telemetry,
        telemetry.newton_scope_calls,
    );
    let mesh_pass_count = reference["mesh_pass_count"]
        .parse::<usize>()
        .expect("SciPy mesh-pass count");
    assert_eq!(telemetry.mesh_passes.len(), mesh_pass_count);
    for (pass, rust_pass) in telemetry.mesh_passes.iter().enumerate() {
        let prefix = format!("mesh.{pass}");
        let scipy_nodes = reference[&format!("{prefix}.nodes")]
            .parse::<usize>()
            .expect("SciPy pass node count");
        let scipy_dimension = reference[&format!("{prefix}.dimension")]
            .parse::<usize>()
            .expect("SciPy pass dimension");
        assert_eq!(rust_pass.mesh_nodes, scipy_nodes, "mesh pass {pass} nodes");
        assert_eq!(
            rust_pass.linear_system_dimension, scipy_dimension,
            "mesh pass {pass} system dimension"
        );
        assert!(
            rust_pass.newton_ms.is_some(),
            "mesh pass {pass} Newton timing"
        );
        assert!(
            rust_pass.defect_estimation_ms.is_some(),
            "mesh pass {pass} defect-estimation timing"
        );
        if rust_pass.factorizations > 0 {
            assert!(
                rust_pass.factorization_ms.is_some(),
                "mesh pass {pass} factorization timing"
            );
        }
        if rust_pass.linear_solves > 0 {
            assert!(
                rust_pass.linear_solve_ms.is_some(),
                "mesh pass {pass} linear-solve timing"
            );
        }
        for (counter, rust_count) in [
            ("solve_newton_calls", rust_pass.newton_passes),
            ("construct_global_jac_calls", rust_pass.jacobian_builds),
            ("factorization_calls", rust_pass.factorizations),
            ("linear_solve_calls", rust_pass.linear_solves),
        ] {
            let scipy_count = reference[&format!("{prefix}.counter.{counter}")]
                .parse::<u64>()
                .expect("SciPy per-mesh work counter");
            assert_eq!(
                rust_count, scipy_count,
                "mesh pass {pass} {counter} work differs"
            );
        }
    }

    let expected_parameters = reference["parameters"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy parameter"))
        .collect::<Vec<_>>();
    let max_parameter_diff = solution
        .parameters
        .iter()
        .zip(&expected_parameters)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let mut max_value_diff = max_parameter_diff;
    let mut max_derivative_diff = 0.0_f64;
    for (point, x) in sample_x.iter().enumerate() {
        let values = dense.evaluate(*x).expect("large parameterized state");
        let derivatives = dense
            .evaluate_derivative(*x)
            .expect("large parameterized derivative");
        for component in 0..dimension {
            max_value_diff =
                max_value_diff.max((values[component] - expected_values[component][point]).abs());
            max_derivative_diff = max_derivative_diff
                .max((derivatives[component] - expected_derivatives[component][point]).abs());
        }
    }
    let max_interval_residual_diff = solution
        .interval_residuals
        .iter()
        .zip(&expected_residuals)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_value_diff < 2e-4,
        "large parameterized value drift={max_value_diff:e}"
    );
    assert!(
        max_derivative_diff < 2e-3,
        "large parameterized derivative drift={max_derivative_diff:e}"
    );
    assert_eq!(solution.interval_residuals.len(), expected_residuals.len());
    assert!(
        max_interval_residual_diff < 1e-9,
        "large parameterized RMS defect drift={max_interval_residual_diff:e}"
    );
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] official large parameterized problem",
        &[LargeParameterizedRow {
            dimension,
            parameter_count: solution.parameters.len(),
            scipy_status: reference["status"].clone(),
            rust_status: solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: solution.x.len().to_string(),
            scipy_parameters: format!("{expected_parameters:?}"),
            rust_parameters: format!("{:?}", solution.parameters),
            scipy_passes_jac_lu_solves: format!(
                "{}/{}/{}/{}",
                reference["newton_passes"],
                reference["jacobian_builds"],
                reference["factorizations"],
                reference["linear_solves"]
            ),
            rust_passes_jac_lu_solves: format!(
                "{}/{}/{}/{}",
                telemetry.newton_scope_calls,
                telemetry.newton_jacobian_refreshes,
                telemetry.factorizations,
                telemetry.linear_solves
            ),
            scipy_per_mesh_work: (0..mesh_pass_count)
                .map(|pass| {
                    format!(
                        "{}:{}:{}/{}/{}",
                        pass,
                        reference[&format!("mesh.{pass}.nodes")],
                        reference[&format!("mesh.{pass}.counter.construct_global_jac_calls")],
                        reference[&format!("mesh.{pass}.counter.factorization_calls")],
                        reference[&format!("mesh.{pass}.counter.linear_solve_calls")],
                    )
                })
                .collect::<Vec<_>>()
                .join(";"),
            rust_per_mesh_work: telemetry
                .mesh_passes
                .iter()
                .map(|pass| {
                    format!(
                        "{}:{}:{}/{}/{}",
                        pass.mesh_pass,
                        pass.mesh_nodes,
                        pass.jacobian_builds,
                        pass.factorizations,
                        pass.linear_solves,
                    )
                })
                .collect::<Vec<_>>()
                .join(";"),
            max_value_diff: format!("{max_value_diff:.3e}"),
            max_derivative_diff: format!("{max_derivative_diff:.3e}"),
            max_interval_residual_diff: format!("{max_interval_residual_diff:.3e}"),
        }],
    );
}

#[cfg(feature = "superlu")]
#[test]
#[ignore = "SuperLU/SciPy parity requires the built bridge and pinned SciPy runtime"]
fn scipy_118_official_parameterized_superlu_backend_matches_reference() {
    #[derive(Tabled)]
    struct SuperLuSciPyRow {
        workload: &'static str,
        backend: &'static str,
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_factorizations: String,
        rust_factorizations: u64,
        scipy_linear_solves: String,
        rust_linear_solves: u64,
        superlu_factor_calls: u64,
        fallback_count: u64,
        max_sample_value_diff: String,
        status: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "big-parameterized-analytic");
    let expected_values = parse_scipy_samples(&reference);
    let sample_x = reference["sample_x"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy sample coordinate"))
        .collect::<Vec<_>>();
    let mut solver = build_official_large_parameterized_sparse_solver_with_output(
        BvpSciSparseFactorization::SuperLu,
        BvpSciOutputPolicy::DenseOutput,
    );
    let solution = solver
        .solve()
        .expect("official BVP should solve through the SuperLU backend");
    let dense = solution
        .dense_output()
        .expect("SuperLU parity route requested dense output");
    let telemetry = solver.plan().telemetry_snapshot();

    assert_eq!(reference["status"], "0");
    assert_eq!(solution.status.code().to_string(), reference["status"]);
    assert_eq!(solution.x.len().to_string(), reference["nodes"]);
    assert_eq!(
        telemetry.superlu_fallbacks, 0,
        "SuperLU must not silently fall back"
    );
    assert!(telemetry.superlu.native_factorization_calls > 0);
    assert_scipy_counter_parity(
        "official parameterized SuperLU backend",
        &reference,
        &telemetry,
        telemetry.newton_scope_calls,
    );

    let mut max_sample_value_diff = 0.0_f64;
    let mut actual = vec![0.0; dense.dimension()];
    for (sample_index, &x) in sample_x.iter().enumerate() {
        dense
            .evaluate_into(x, &mut actual)
            .expect("evaluate SuperLU solution on the SciPy sample grid");
        for (component, value) in actual.iter().enumerate() {
            max_sample_value_diff =
                max_sample_value_diff.max((value - expected_values[component][sample_index]).abs());
        }
    }
    assert!(
        max_sample_value_diff < 2e-4,
        "SuperLU/SciPy sample drift={max_sample_value_diff:e}"
    );

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] official parameterized SuperLU backend",
        &[SuperLuSciPyRow {
            workload: "official 60-state parameterized BVP",
            backend: "SuperLU opt-in",
            scipy_status: reference["status"].clone(),
            rust_status: solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: solution.x.len().to_string(),
            scipy_factorizations: reference["factorizations"].clone(),
            rust_factorizations: telemetry.factorizations,
            scipy_linear_solves: reference["linear_solves"].clone(),
            rust_linear_solves: telemetry.linear_solves,
            superlu_factor_calls: telemetry.superlu.native_factorization_calls,
            fallback_count: telemetry.superlu_fallbacks,
            max_sample_value_diff: format!("{max_sample_value_diff:.3e}"),
            status: "ok",
        }],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_exponential_two_point_problem_matches_exact_jacobian_trace() {
    #[derive(Tabled)]
    struct ParityRow {
        reference_route: &'static str,
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_newton_jac_lu_solve: String,
        rust_newton_jac_lu_solve: String,
        max_sample_diff: String,
        max_derivative_diff: String,
        max_interval_residual_diff: String,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sample_x: Vec<f64> = (0..9).map(|index| index as f64 / 8.0).collect();
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("z"), Expr::parse_expression("y")],
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("exponential SciPy parity plan");
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0] - 1.0;
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..5).map(|index| index as f64 / 4.0).collect();
    let mut options = BvpSciOptions::default();
    // Match the official SciPy test_no_params tolerance rather than imposing
    // a stricter unrelated threshold on this algorithm-parity case.
    options.tolerance = 1e-3;
    options.max_nodes = 100;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![0.0; mesh.len() * 2],
        vec![],
        options,
    )
    .expect("exponential SciPy parity solver");
    let rust_solution = rust.solve().expect("Rust exponential BVP should converge");
    let dense = rust_solution.dense_output().expect("Rust dense output");
    let telemetry = rust.plan().telemetry_snapshot();
    let rust_passes = telemetry.newton_scope_calls;
    let mut rows = Vec::new();

    for (case, route) in [
        ("exponential", "SciPy finite-difference Jacobians"),
        ("exponential-analytic", "SciPy exact Jacobians"),
    ] {
        let reference = scipy_bvp_reference_row(&root, case);
        let expected = parse_scipy_samples(&reference);
        let expected_derivatives = parse_scipy_derivatives(&reference);
        let expected_residuals = reference["rms_residuals"]
            .split(',')
            .map(|value| value.parse::<f64>().expect("SciPy interval residual"))
            .collect::<Vec<_>>();
        let mut max_diff = 0.0_f64;
        let mut max_derivative_diff = 0.0_f64;
        for (column, x) in sample_x.iter().enumerate() {
            let actual = dense
                .evaluate(*x)
                .expect("sample Rust exponential solution");
            let actual_derivative = dense
                .evaluate_derivative(*x)
                .expect("sample Rust exponential derivative");
            for (component, value) in actual.iter().enumerate() {
                max_diff = max_diff.max((value - expected[component][column]).abs());
                max_derivative_diff = max_derivative_diff.max(
                    (actual_derivative[component] - expected_derivatives[component][column]).abs(),
                );
            }
        }
        let max_interval_residual_diff = rust_solution
            .interval_residuals
            .iter()
            .zip(&expected_residuals)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 2e-5,
            "{case} SciPy/Rust drift={max_diff:e} exceeded the shared tol=1e-3 accuracy envelope"
        );
        assert!(
            max_derivative_diff < 2e-5,
            "{case} dense-output derivative drift={max_derivative_diff:e}"
        );
        assert_eq!(
            rust_solution.interval_residuals.len(),
            expected_residuals.len()
        );
        assert!(
            max_interval_residual_diff < 1e-10,
            "{case} interval residual drift={max_interval_residual_diff:e}"
        );
        assert_scipy_counter_parity(case, &reference, &telemetry, rust_passes);
        rows.push(ParityRow {
            reference_route: route,
            scipy_status: reference["status"].clone(),
            rust_status: rust_solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: rust_solution.x.len().to_string(),
            scipy_newton_jac_lu_solve: format!(
                "{}/{}/{}/{}",
                reference["newton_passes"],
                reference["jacobian_builds"],
                reference["factorizations"],
                reference["linear_solves"]
            ),
            rust_newton_jac_lu_solve: format!(
                "{}/{}/{}/{}",
                rust_passes,
                telemetry.newton_jacobian_refreshes,
                telemetry.factorizations,
                telemetry.linear_solves
            ),
            max_sample_diff: format!("{max_diff:.3e}"),
            max_derivative_diff: format!("{max_derivative_diff:.3e}"),
            max_interval_residual_diff: format!("{max_interval_residual_diff:.3e}"),
        });
    }

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] exponential two-point BVP and Jacobian routes",
        &rows,
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_sturm_liouville_unknown_parameter_matches_reference_routes() {
    #[derive(Tabled)]
    struct ParityRow {
        scipy_jacobian: &'static str,
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_parameter: String,
        rust_parameter: String,
        scipy_jac_lu_solves: String,
        rust_jac_lu_solves: String,
        rust_newton_trace: String,
        max_sample_diff: String,
        counter_parity: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let equations = [
        Expr::parse_expression("z"),
        // Parenthesize the positive power before applying unary minus so the
        // fixture does not depend on the expression parser's unary precedence.
        Expr::parse_expression("-(p^2)*y"),
    ];
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &equations,
        &["y".into(), "z".into()],
        &["p".into()],
        "x",
        BvpSciTelemetry::timings(),
    )
    .expect("Sturm-Liouville parity plan");
    let boundary = BvpSciBoundaryCallbacks::new(
        3,
        |ya, yb, parameters, output| {
            output[0] = ya[0];
            output[1] = yb[0];
            output[2] = ya[1] - parameters[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..5)
        .map(|index| std::f64::consts::PI * index as f64 / 4.0)
        .collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-3;
    options.max_nodes = 100;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let initial = vec![1.0; mesh.len() * 2];
    let mut rust = BvpSciSolver::new(plan, boundary, mesh, initial, vec![0.5], options)
        .expect("Sturm-Liouville parity solver");
    let solution = rust
        .solve()
        .expect("Rust Sturm-Liouville BVP should converge");
    let dense = solution
        .dense_output()
        .expect("Sturm-Liouville dense output");
    let telemetry = rust.plan().telemetry_snapshot();
    let rust_passes = telemetry.newton_scope_calls;
    let mut rows = Vec::new();

    for (case, jacobian_kind) in [
        ("sturm-liouville", "finite differences"),
        ("sturm-liouville-analytic", "analytic callbacks"),
    ] {
        let reference = scipy_bvp_reference_row(&root, case);
        let expected = parse_scipy_samples(&reference);
        let sample_x = reference["sample_x"]
            .split(',')
            .map(|value| value.parse::<f64>().expect("SciPy sample coordinate"))
            .collect::<Vec<_>>();
        let mut max_diff = (solution.parameters[0]
            - reference["parameters"]
                .parse::<f64>()
                .expect("SciPy parameter"))
        .abs();
        for (column, x) in sample_x.iter().enumerate() {
            let actual = dense.evaluate(*x).expect("sample Rust parameter solution");
            for (component, value) in actual.iter().enumerate() {
                max_diff = max_diff.max((value - expected[component][column]).abs());
            }
        }
        assert!(
            max_diff < 1e-4,
            "{case} SciPy/Rust parameter/trajectory drift={max_diff:e}; rust_p={:?}, scipy_p={}, nodes={}",
            solution.parameters,
            reference["parameters"],
            solution.x.len()
        );
        let scipy_counts_match = [
            ("newton_passes", rust_passes),
            ("jacobian_builds", telemetry.newton_jacobian_refreshes),
            ("factorizations", telemetry.factorizations),
            ("linear_solves", telemetry.linear_solves),
            ("mesh_modify_calls", telemetry.mesh_refinements),
        ]
        .iter()
        .all(|(key, actual)| reference[*key].parse::<u64>().ok() == Some(*actual));
        let rust_newton_trace = telemetry
            .newton_residual_history
            .iter()
            .map(|entry| {
                format!(
                    "{}:{:.1e}:a{:.3}:j{}",
                    entry.iteration,
                    entry.residual_before,
                    entry.armijo_alpha,
                    u8::from(entry.jacobian_refreshed)
                )
            })
            .collect::<Vec<_>>()
            .join(";");
        rows.push(ParityRow {
            scipy_jacobian: jacobian_kind,
            scipy_status: reference["status"].clone(),
            rust_status: solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: solution.x.len().to_string(),
            scipy_parameter: reference["parameters"].clone(),
            rust_parameter: format!("{:.9}", solution.parameters[0]),
            scipy_jac_lu_solves: format!(
                "{}/{}/{}",
                reference["jacobian_builds"],
                reference["factorizations"],
                reference["linear_solves"]
            ),
            rust_jac_lu_solves: format!(
                "{}/{}/{}",
                telemetry.newton_jacobian_refreshes,
                telemetry.factorizations,
                telemetry.linear_solves
            ),
            rust_newton_trace,
            max_sample_diff: format!("{max_diff:.3e}"),
            counter_parity: if scipy_counts_match {
                "exact"
            } else {
                "counter-drift"
            },
        });
    }

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] Sturm-Liouville unknown parameter",
        &rows,
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_nonlinear_boundary_conditions_match_rust_callbacks() {
    #[derive(Tabled)]
    struct ParityRow {
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_passes_jac_lu_solves: String,
        rust_passes_jac_lu_solves: String,
        max_sample_diff: String,
        counter_parity: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "nonlinear-boundary");
    let expected = parse_scipy_samples(&reference);
    let sample_x = reference["sample_x"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy sample coordinate"))
        .collect::<Vec<_>>();
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("z"), Expr::parse_expression("0")],
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::timings(),
    )
    .expect("nonlinear-boundary parity plan");
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            let (kappa, io_a, io_c, voltage, factor) = (1.64, 0.01, 1.0e-4, 0.5, 38.9);
            let h_a = -ya[0];
            let current_a = io_a * ((factor * h_a).exp() - (-factor * h_a).exp());
            let h_c = voltage - yb[0] - 1.0;
            let current_c = io_c * ((factor * h_c).exp() - (-factor * h_c).exp());
            output[0] = current_a + kappa * ya[1];
            output[1] = current_c - kappa * yb[1];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..5).map(|index| 0.1 * index as f64 / 4.0).collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-3;
    options.max_nodes = 100;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![0.0; mesh.len() * 2],
        vec![],
        options,
    )
    .expect("nonlinear-boundary parity solver");
    let solution = rust
        .solve()
        .expect("Rust nonlinear-boundary BVP should converge");
    let dense = solution
        .dense_output()
        .expect("nonlinear-boundary dense output");
    let telemetry = rust.plan().telemetry_snapshot();
    let rust_passes = telemetry.newton_scope_calls;
    let counters_match = [
        ("newton_passes", rust_passes),
        ("jacobian_builds", telemetry.newton_jacobian_refreshes),
        ("factorizations", telemetry.factorizations),
        ("linear_solves", telemetry.linear_solves),
        ("mesh_modify_calls", telemetry.mesh_refinements),
    ]
    .iter()
    .all(|(key, actual)| reference[*key].parse::<u64>().ok() == Some(*actual));
    let mut max_diff = 0.0_f64;
    for (column, x) in sample_x.iter().enumerate() {
        let actual = dense.evaluate(*x).expect("nonlinear-boundary sample");
        for (component, value) in actual.iter().enumerate() {
            max_diff = max_diff.max((value - expected[component][column]).abs());
        }
    }
    assert!(
        max_diff < 1e-8,
        "nonlinear boundary SciPy/Rust drift={max_diff:e}"
    );
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] nonlinear boundary-value callbacks",
        &[ParityRow {
            scipy_status: reference["status"].clone(),
            rust_status: solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: solution.x.len().to_string(),
            scipy_passes_jac_lu_solves: format!(
                "{}/{}/{}/{}",
                reference["newton_passes"],
                reference["jacobian_builds"],
                reference["factorizations"],
                reference["linear_solves"]
            ),
            rust_passes_jac_lu_solves: format!(
                "{}/{}/{}/{}",
                rust_passes,
                telemetry.newton_jacobian_refreshes,
                telemetry.factorizations,
                telemetry.linear_solves
            ),
            max_sample_diff: format!("{max_diff:.3e}"),
            counter_parity: if counters_match {
                "exact"
            } else {
                "counter-drift"
            },
        }],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_large_independent_system_matches_rust_dense_core() {
    #[derive(Tabled)]
    struct ParityRow {
        dimension: usize,
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_jac_lu_solves: String,
        rust_jac_lu_solves: String,
        max_sample_diff: String,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "big-problem");
    let expected = parse_scipy_samples(&reference);
    let sample_x = reference["sample_x"]
        .split(',')
        .map(|value| value.parse::<f64>().expect("SciPy sample coordinate"))
        .collect::<Vec<_>>();
    let systems = 30usize;
    let dimension = systems * 2;
    let states = (0..systems)
        .flat_map(|index| [format!("y{index}"), format!("z{index}")])
        .collect::<Vec<_>>();
    let equations = (0..systems)
        .flat_map(|index| {
            [
                Expr::parse_expression(&format!("z{index}")),
                Expr::parse_expression("0"),
            ]
        })
        .collect::<Vec<_>>();
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &equations,
        &states,
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("large independent parity plan");
    let boundary = BvpSciBoundaryCallbacks::new(
        dimension,
        move |ya, yb, _, output| {
            for index in 0..systems {
                output[index] = ya[index * 2];
                output[systems + index] = yb[index * 2] - 1.0;
            }
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..5).map(|index| index as f64 / 4.0).collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-3;
    options.max_nodes = 100;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![0.0; mesh.len() * dimension],
        vec![],
        options,
    )
    .expect("large independent parity solver");
    let solution = rust
        .solve()
        .expect("Rust large independent BVP should converge");
    let dense = solution.dense_output().expect("large-system dense output");
    let telemetry = rust.plan().telemetry_snapshot();
    assert_scipy_counter_parity(
        "big-problem",
        &reference,
        &telemetry,
        telemetry.newton_scope_calls,
    );
    let mut max_diff = 0.0_f64;
    for (column, x) in sample_x.iter().enumerate() {
        let actual = dense.evaluate(*x).expect("large-system sample");
        for (component, value) in actual.iter().enumerate() {
            max_diff = max_diff.max((value - expected[component][column]).abs());
        }
    }
    assert!(max_diff < 1e-10, "large SciPy/Rust drift={max_diff:e}");
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] official large independent BVP",
        &[ParityRow {
            dimension,
            scipy_status: reference["status"].clone(),
            rust_status: solution.status.code().to_string(),
            scipy_nodes: reference["nodes"].clone(),
            rust_nodes: solution.x.len().to_string(),
            scipy_jac_lu_solves: format!(
                "{}/{}/{}",
                reference["jacobian_builds"],
                reference["factorizations"],
                reference["linear_solves"]
            ),
            rust_jac_lu_solves: format!(
                "{}/{}/{}",
                telemetry.newton_jacobian_refreshes,
                telemetry.factorizations,
                telemetry.linear_solves
            ),
            max_sample_diff: format!("{max_diff:.3e}"),
        }],
    );
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_singular_collocation_jacobian_maps_to_rust_typed_failure() {
    #[derive(Tabled)]
    struct FailureRow {
        scipy_status: String,
        rust_status: String,
        scipy_factorization_calls: String,
        rust_factorizations: String,
        rust_error: String,
        parity: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = scipy_bvp_reference_row(&root, "singular-jacobian");
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("0"), Expr::parse_expression("0")],
        &["y0".into(), "y1".into()],
        &[],
        "x",
        BvpSciTelemetry::timings(),
    )
    .expect("singular Jacobian plan");
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0] - 1.0;
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-5;
    options.max_nodes = 5;
    let mut rust = BvpSciSolver::new(
        plan,
        boundary,
        vec![0.0, 1.0],
        vec![0.0; 4],
        vec![],
        options,
    )
    .expect("singular Jacobian solver construction");
    let error = rust
        .solve()
        .expect_err("incompatible constant ODE and endpoint constraints are singular");
    let rust_status = error.status().map(|status| status.code());
    let rust_telemetry = rust.plan().telemetry_snapshot();
    let scipy_status = reference["status"].parse::<u8>().expect("SciPy status");
    let status_parity = rust_status == Some(scipy_status);
    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] singular global collocation Jacobian",
        &[FailureRow {
            scipy_status: scipy_status.to_string(),
            rust_status: rust_status
                .map(|status| status.to_string())
                .unwrap_or_else(|| "unmapped".into()),
            scipy_factorization_calls: reference["factorizations"].clone(),
            rust_factorizations: rust_telemetry.factorizations.to_string(),
            rust_error: format!("{error:?}"),
            parity: if status_parity {
                "status-match"
            } else {
                "status-gap"
            },
        }],
    );
    assert!(
        status_parity,
        "SciPy status {scipy_status} must map to Rust typed error/status; got {error:?}"
    );
}

fn parse_scipy_samples(fields: &HashMap<String, String>) -> Vec<Vec<f64>> {
    fields["sample_y"]
        .split(';')
        .map(|component| {
            component
                .split(',')
                .map(|value| value.parse::<f64>().expect("SciPy sample value"))
                .collect()
        })
        .collect()
}

fn parse_scipy_derivatives(fields: &HashMap<String, String>) -> Vec<Vec<f64>> {
    fields["sample_dy"]
        .split(';')
        .map(|component| {
            component
                .split(',')
                .map(|value| value.parse::<f64>().expect("SciPy sample derivative"))
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "cross-language parity gate requires pinned SciPy Python runtime"]
fn scipy_118_nonlinear_singular_and_exhaustion_paths_match_or_map_explicitly() {
    #[derive(Tabled)]
    struct ParityRow {
        case: &'static str,
        scipy_status: String,
        rust_status: String,
        scipy_nodes: String,
        rust_nodes: String,
        scipy_passes: String,
        rust_passes: String,
        scipy_jac: String,
        rust_jac: String,
        scipy_lu: String,
        rust_lu: String,
        scipy_solves: String,
        rust_solves: String,
        scipy_mesh_calls: String,
        rust_mesh_changes: String,
        max_sample_diff: String,
        status: &'static str,
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sample_x: Vec<f64> = (0..9).map(|i| i as f64 / 8.0).collect();
    let mut rows = Vec::new();

    let bratu_reference = scipy_bvp_reference_row(&root, "bratu");
    let bratu_expected = parse_scipy_samples(&bratu_reference);
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &bratu_equations(),
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("Bratu parity plan");
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let mesh: Vec<f64> = (0..6).map(|i| i as f64 / 5.0).collect();
    let mut options = BvpSciOptions::default();
    options.tolerance = 1e-5;
    options.max_nodes = 200;
    options.max_mesh_refinements = 10;
    options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut bratu = BvpSciSolver::new(
        plan,
        boundary,
        mesh.clone(),
        vec![0.0; mesh.len() * 2],
        vec![],
        options,
    )
    .expect("Bratu parity solver");
    let bratu_solution = bratu.solve().expect("Rust Bratu should converge");
    let bratu_dense = bratu_solution.dense_output().expect("Bratu dense output");
    let mut bratu_diff = 0.0_f64;
    for (column, x) in sample_x.iter().enumerate() {
        let actual = bratu_dense.evaluate(*x).expect("Bratu sample");
        for (component, state) in actual.iter().enumerate() {
            bratu_diff = bratu_diff.max((state - bratu_expected[component][column]).abs());
        }
    }
    assert!(bratu_diff < 5e-4, "Bratu SciPy/Rust drift={bratu_diff:e}");
    let bratu_telemetry = bratu.plan().telemetry_snapshot();
    assert_scipy_counter_parity(
        "Bratu",
        &bratu_reference,
        &bratu_telemetry,
        bratu_telemetry.newton_scope_calls,
    );
    rows.push(ParityRow {
        case: "Bratu nonlinear",
        scipy_status: bratu_reference["status"].clone(),
        rust_status: bratu_solution.status.code().to_string(),
        scipy_nodes: bratu_reference["nodes"].clone(),
        rust_nodes: bratu_solution.x.len().to_string(),
        scipy_passes: bratu_reference["newton_passes"].clone(),
        rust_passes: bratu_telemetry.newton_scope_calls.to_string(),
        scipy_jac: bratu_reference["jacobian_builds"].clone(),
        rust_jac: bratu_telemetry.newton_jacobian_refreshes.to_string(),
        scipy_lu: bratu_reference["factorizations"].clone(),
        rust_lu: bratu_telemetry.factorizations.to_string(),
        scipy_solves: bratu_reference["linear_solves"].clone(),
        rust_solves: bratu_telemetry.linear_solves.to_string(),
        scipy_mesh_calls: bratu_reference["mesh_modify_calls"].clone(),
        rust_mesh_changes: bratu_telemetry.mesh_refinements.to_string(),
        max_sample_diff: format!("{bratu_diff:.3e}"),
        status: "ok",
    });

    let parameter_reference = scipy_bvp_reference_row(&root, "unknown-parameter");
    let parameter_expected = parse_scipy_samples(&parameter_reference);
    let parameter_plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("p")],
        &["y".into()],
        &["p".into()],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("unknown-parameter parity plan");
    let parameter_boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0] - 1.0;
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let parameter_mesh: Vec<f64> = (0..6).map(|i| i as f64 / 5.0).collect();
    let parameter_initial: Vec<f64> = parameter_mesh.iter().map(|x| 0.8 * x).collect();
    let mut parameter_options = BvpSciOptions::default();
    parameter_options.tolerance = 1e-8;
    parameter_options.max_nodes = 100;
    parameter_options.output_policy = BvpSciOutputPolicy::DenseOutput;
    let mut parameter_solver = BvpSciSolver::new(
        parameter_plan,
        parameter_boundary,
        parameter_mesh,
        parameter_initial,
        vec![0.8],
        parameter_options,
    )
    .expect("unknown-parameter solver");
    let parameter_solution = parameter_solver
        .solve()
        .expect("unknown-parameter solve should converge");
    let parameter_dense = parameter_solution
        .dense_output()
        .expect("parameter dense output");
    let mut parameter_diff = (parameter_solution.parameters[0] - 1.0).abs();
    for (column, x) in sample_x.iter().enumerate() {
        let actual = parameter_dense.evaluate(*x).expect("parameter sample")[0];
        parameter_diff = parameter_diff.max((actual - parameter_expected[0][column]).abs());
    }
    assert!(
        parameter_diff < 1e-8,
        "unknown parameter parity drift={parameter_diff:e}"
    );
    let parameter_telemetry = parameter_solver.plan().telemetry_snapshot();
    assert_scipy_counter_parity(
        "unknown parameter",
        &parameter_reference,
        &parameter_telemetry,
        parameter_telemetry.newton_scope_calls,
    );
    rows.push(ParityRow {
        case: "unknown parameter",
        scipy_status: parameter_reference["status"].clone(),
        rust_status: parameter_solution.status.code().to_string(),
        scipy_nodes: parameter_reference["nodes"].clone(),
        rust_nodes: parameter_solution.x.len().to_string(),
        scipy_passes: parameter_reference["newton_passes"].clone(),
        rust_passes: parameter_telemetry.newton_scope_calls.to_string(),
        scipy_jac: parameter_reference["jacobian_builds"].clone(),
        rust_jac: parameter_telemetry.newton_jacobian_refreshes.to_string(),
        scipy_lu: parameter_reference["factorizations"].clone(),
        rust_lu: parameter_telemetry.factorizations.to_string(),
        scipy_solves: parameter_reference["linear_solves"].clone(),
        rust_solves: parameter_telemetry.linear_solves.to_string(),
        scipy_mesh_calls: parameter_reference["mesh_modify_calls"].clone(),
        rust_mesh_changes: parameter_telemetry.mesh_refinements.to_string(),
        max_sample_diff: format!("{parameter_diff:.3e}"),
        status: "ok",
    });

    let singular_reference = scipy_bvp_reference_row(&root, "singular");
    let singular_expected = parse_scipy_samples(&singular_reference);
    let singular_plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("z"), Expr::parse_expression("-y^5")],
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("singular Lane-Emden plan");
    let singular_boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[1];
            output[1] = yb[0] - (3.0_f64 / 4.0).sqrt();
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let singular_mesh: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
    let initial_state: Vec<f64> = singular_mesh
        .iter()
        .flat_map(|_| [(3.0_f64 / 4.0).sqrt(), 1e-4])
        .collect();
    let mut singular_options = BvpSciOptions::default();
    singular_options.tolerance = 1e-3;
    singular_options.max_nodes = 100;
    singular_options.output_policy = BvpSciOutputPolicy::DenseOutput;
    singular_options.singular_term = Some(
        BvpSciSingularTerm::new(2, vec![0.0, 0.0, 0.0, -2.0])
            .expect("valid SciPy singular-term matrix"),
    );
    let mut singular = BvpSciSolver::new(
        singular_plan,
        singular_boundary,
        singular_mesh,
        initial_state,
        vec![],
        singular_options,
    )
    .expect("singular Lane-Emden solver");
    let singular_solution = singular.solve().expect("Rust singular BVP should converge");
    let singular_dense = singular_solution
        .dense_output()
        .expect("singular dense output");
    let mut singular_diff = 0.0_f64;
    for (column, x) in sample_x.iter().enumerate() {
        let actual = singular_dense.evaluate(*x).expect("singular sample");
        for (component, state) in actual.iter().enumerate() {
            singular_diff = singular_diff.max((state - singular_expected[component][column]).abs());
        }
    }
    assert!(
        singular_diff < 2e-3,
        "singular SciPy/Rust drift={singular_diff:e}"
    );
    let singular_telemetry = singular.plan().telemetry_snapshot();
    assert_scipy_counter_parity(
        "Lane-Emden singular",
        &singular_reference,
        &singular_telemetry,
        singular_telemetry.newton_scope_calls,
    );
    rows.push(ParityRow {
        case: "Lane-Emden singular S",
        scipy_status: singular_reference["status"].clone(),
        rust_status: singular_solution.status.code().to_string(),
        scipy_nodes: singular_reference["nodes"].clone(),
        rust_nodes: singular_solution.x.len().to_string(),
        scipy_passes: singular_reference["newton_passes"].clone(),
        rust_passes: singular_telemetry.newton_scope_calls.to_string(),
        scipy_jac: singular_reference["jacobian_builds"].clone(),
        rust_jac: singular_telemetry.newton_jacobian_refreshes.to_string(),
        scipy_lu: singular_reference["factorizations"].clone(),
        rust_lu: singular_telemetry.factorizations.to_string(),
        scipy_solves: singular_reference["linear_solves"].clone(),
        rust_solves: singular_telemetry.linear_solves.to_string(),
        scipy_mesh_calls: singular_reference["mesh_modify_calls"].clone(),
        rust_mesh_changes: singular_telemetry.mesh_refinements.to_string(),
        max_sample_diff: format!("{singular_diff:.3e}"),
        status: "ok",
    });

    let exhausted_reference = scipy_bvp_reference_row(&root, "bratu-exhaustion");
    let exhausted_plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &bratu_equations(),
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("exhaustion plan");
    let exhausted_boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::detailed(),
    );
    let exhausted_mesh: Vec<f64> = (0..6).map(|i| i as f64 / 5.0).collect();
    let mut exhausted_options = BvpSciOptions::default();
    exhausted_options.tolerance = 1e-8;
    exhausted_options.max_nodes = 8;
    exhausted_options.max_mesh_refinements = 10;
    let mut exhausted = BvpSciSolver::new(
        exhausted_plan,
        exhausted_boundary,
        exhausted_mesh.clone(),
        vec![0.0; exhausted_mesh.len() * 2],
        vec![],
        exhausted_options,
    )
    .expect("exhaustion solver");
    let error = exhausted
        .solve()
        .expect_err("eight-node budget must exhaust");
    assert_eq!(error.status().map(|status| status.code()), Some(1));
    assert_eq!(exhausted_reference["status"], "1");
    let exhausted_telemetry = exhausted.plan().telemetry_snapshot();
    assert_scipy_counter_parity(
        "Bratu max-nodes",
        &exhausted_reference,
        &exhausted_telemetry,
        exhausted_telemetry.newton_scope_calls,
    );
    rows.push(ParityRow {
        case: "Bratu max-nodes",
        scipy_status: exhausted_reference["status"].clone(),
        rust_status: error.status().unwrap().code().to_string(),
        scipy_nodes: exhausted_reference["nodes"].clone(),
        rust_nodes: "budget error".into(),
        scipy_passes: exhausted_reference["newton_passes"].clone(),
        rust_passes: exhausted_telemetry.newton_scope_calls.to_string(),
        scipy_jac: exhausted_reference["jacobian_builds"].clone(),
        rust_jac: exhausted_telemetry.newton_jacobian_refreshes.to_string(),
        scipy_lu: exhausted_reference["factorizations"].clone(),
        rust_lu: exhausted_telemetry.factorizations.to_string(),
        scipy_solves: exhausted_reference["linear_solves"].clone(),
        rust_solves: exhausted_telemetry.linear_solves.to_string(),
        scipy_mesh_calls: exhausted_reference["mesh_modify_calls"].clone(),
        rust_mesh_changes: exhausted_telemetry.mesh_refinements.to_string(),
        max_sample_diff: "n/a".into(),
        status: "status-mapped",
    });

    let tight_reference = scipy_bvp_reference_row(&root, "exponential-tight");
    let tight_plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("z"), Expr::parse_expression("y")],
        &["y".into(), "z".into()],
        &[],
        "x",
        BvpSciTelemetry::detailed(),
    )
    .expect("tight exponential plan");
    let tight_boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0] - 1.0;
            output[1] = yb[0];
            Ok(())
        },
        BvpSciTelemetry::disabled(),
    );
    let tight_mesh: Vec<f64> = (0..5).map(|index| index as f64 / 4.0).collect();
    let mut tight_options = BvpSciOptions::default();
    tight_options.tolerance = 1e-8;
    tight_options.max_nodes = 100;
    let mut tight = BvpSciSolver::new(
        tight_plan,
        tight_boundary,
        tight_mesh.clone(),
        vec![0.0; tight_mesh.len() * 2],
        vec![],
        tight_options,
    )
    .expect("tight exponential solver");
    let tight_error = tight
        .solve()
        .expect_err("strict exponential case should hit the shared node budget");
    let tight_telemetry = tight.plan().telemetry_snapshot();
    assert_eq!(tight_reference["status"], "1");
    assert_eq!(tight_error.status().map(|status| status.code()), Some(1));
    assert_scipy_counter_parity(
        "exponential tight max-nodes",
        &tight_reference,
        &tight_telemetry,
        tight_telemetry.newton_scope_calls,
    );
    rows.push(ParityRow {
        case: "exponential tol=1e-8 max-nodes",
        scipy_status: tight_reference["status"].clone(),
        rust_status: tight_error.status().unwrap().code().to_string(),
        scipy_nodes: tight_reference["nodes"].clone(),
        rust_nodes: "typed budget error".into(),
        scipy_passes: tight_reference["newton_passes"].clone(),
        rust_passes: tight_telemetry.newton_scope_calls.to_string(),
        scipy_jac: tight_reference["jacobian_builds"].clone(),
        rust_jac: tight_telemetry.newton_jacobian_refreshes.to_string(),
        scipy_lu: tight_reference["factorizations"].clone(),
        rust_lu: tight_telemetry.factorizations.to_string(),
        scipy_solves: tight_reference["linear_solves"].clone(),
        rust_solves: tight_telemetry.linear_solves.to_string(),
        scipy_mesh_calls: tight_reference["mesh_modify_calls"].clone(),
        rust_mesh_changes: tight_telemetry.mesh_refinements.to_string(),
        max_sample_diff: "n/a (SciPy status 1)".into(),
        status: "status-mapped",
    });

    crate::Utils::test_reporting::capture_test_table(
        "[BVP_sci SciPy 1.18.1 parity] nonlinear, singular and exhaustion cases",
        &rows,
    );
}

fn assert_scipy_counter_parity(
    case: &str,
    scipy: &HashMap<String, String>,
    rust: &super::BvpSciTelemetrySnapshot,
    rust_passes: u64,
) {
    let check = |key: &str, actual: u64, meaning: &str| {
        let expected = scipy[key]
            .parse::<u64>()
            .unwrap_or_else(|error| panic!("{case} SciPy {key} is not an integer: {error}"));
        assert_eq!(
            actual, expected,
            "{case} {meaning} counter differs: Rust={actual}, SciPy={expected}"
        );
    };
    check("newton_passes", rust_passes, "outer Newton pass");
    check(
        "jacobian_builds",
        rust.newton_jacobian_refreshes,
        "global Jacobian build",
    );
    check("factorizations", rust.factorizations, "LU factorization");
    check("linear_solves", rust.linear_solves, "linear solve");
    check(
        "mesh_modify_calls",
        rust.mesh_refinements,
        "accepted mesh modification",
    );
}
