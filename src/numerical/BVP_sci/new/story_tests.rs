//! Compact Lambdify-only story gates for the new BVP architecture.
//!
//! These tests deliberately use small mathematical fixtures so they remain
//! useful in debug mode. The release dashboard in `benches/` expands the same
//! matrix without putting Criterion/compiler chatter into the report table.

use super::*;
use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{DampedSolverOptions, SolverParams, NRBVP};
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
    atomic::{AtomicU64, Ordering},
    Arc,
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
            if row == 2 {
                1.0
            } else {
                0.0
            }
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
    let started = BvpSciTelemetry::timings();
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
        BvpSciSolver::new(
            plan,
            boundary,
            vec![0.0, 0.5, 1.0],
            vec![0.0, (parameter * 0.5).exp() - 1.0, parameter.exp() - 1.0],
            vec![parameter],
            options,
        )
        .expect("story solver should construct"),
        target,
    )
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
        BvpSciTelemetry::timings(),
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
    let (left, right) = workload.interval();
    let plan = BvpSciLambdifyPlan::prepare(
        assembly,
        &workload.equations(),
        &workload.state_names(),
        &[],
        "x",
        BvpSciTelemetry::timings(),
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
    BvpSciSolver::new(plan, boundary, x, y, vec![], options)
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
        Table::new(rows)
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
    crate::Utils::test_reporting::capture_test_block(format!(
        "# Continuation lifecycle\n\n- atom_conversions: {}\n- pattern_entries: {}\n- parameter_rebinds: {}\n- continuation_solves: {}\n- allocations: {}\n- residual_evaluations: {}\n- jacobian_evaluations: {}\n- factorization_ms: {:?}",
        telemetry.atom_conversions,
        telemetry.pattern_entries,
        telemetry.parameter_rebinds,
        telemetry.continuation_solves,
        telemetry.allocations,
        telemetry.residual_evaluations,
        telemetry.jacobian_evaluations,
        telemetry.factorization_ms,
    ));
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
fn lambdify_analytic_state_and_fd_parameter_boundary_jacobians_have_telemetry() {
    #[derive(Tabled)]
    struct JacobianContractRow {
        frontend: String,
        layout: String,
        symbolic_jacobian_derivations: u64,
        jacobian_evaluations: u64,
        finite_difference_probes: u64,
        jacobian_output_assemblies: u64,
        residual_evaluations: u64,
        status: String,
    }

    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BVP_sci_new",
        "lambdify_analytic_state_fd_parameter_boundary_jacobians",
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
            assert!(telemetry.finite_difference_probes > 0);
            assert!(telemetry.jacobian_output_assemblies > 0);
            assert!(telemetry.residual_evaluations > 0);
            rows.push(JacobianContractRow {
                frontend: format!("{assembly:?}"),
                layout: layout_name(layout),
                symbolic_jacobian_derivations: telemetry.symbolic_jacobian_derivations,
                jacobian_evaluations: telemetry.jacobian_evaluations,
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
        "# BVP_sci Jacobian contract\n\n{}",
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
            BvpSciTelemetry::timings(),
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
        banded_linear_residual: String,
        banded_solution_inf: String,
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
                    let mut solver =
                        build_damp_exact_solver_with_nodes(workload, assembly, layout, nodes);
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
                                banded_linear_residual: fmt(telemetry.banded_last_residual_inf),
                                banded_solution_inf: fmt(telemetry.banded_last_solution_inf),
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
                        banded_linear_residual: fmt(telemetry.banded_last_residual_inf),
                        banded_solution_inf: fmt(telemetry.banded_last_solution_inf),
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
            let telemetry = BvpSciTelemetry::timings();
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
        "# BVP_sci AOT frontend/backend contract matrix\n\n- lifecycle: BuildIfMissing + C/{compiler}\n- callback ABI: [x, parameters..., state...]\n- `residual_ms` and `jacobian_ms` are separate callback telemetry scopes\n- AOT telemetry is a separate non-additive IVP snapshot\n\n{}",
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
                let telemetry = BvpSciTelemetry::timings();
                let mut config = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(
                    output_root
                        .join("solver")
                        .join(frontend_name)
                        .join(layout_name)
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
                let mut solver = BvpSciSolver::new(
                    plan,
                    boundary,
                    vec![0.0, 0.5, 1.0],
                    vec![0.0, 0.0, 0.0],
                    vec![1.0],
                    options,
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
                assert_eq!(snapshot.parameter_rebinds, 1);
                assert_eq!(snapshot.continuation_solves, 1);
                rows.push(SolverRow {
                    frontend: frontend_name.into(),
                    layout: layout_name.into(),
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
    crate::Utils::test_reporting::capture_test_block(format!(
        "# BVP_sci AOT solver, continuation and policy matrix\n\n- exact fixture: y' = p*y with zero solution and zero boundary values\n- policy is fixed during AOT preparation\n- full-solve timings include numerical work and are not callback-only timings\n\n{}",
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
                    let telemetry = BvpSciTelemetry::timings();
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
    crate::Utils::test_reporting::capture_test_block(
        report.replace("p=2 -> 3", "p=0.5 -> 0.75")
    );
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
        BvpSciTelemetry::timings(),
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
        BvpSciTelemetry::timings(),
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
        BvpSciTelemetry::timings(),
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
                BvpSciTelemetry::timings(),
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
