//! # MF-1963 six-state stiff-combustion BVP: LOW and HIGH regimes
//!
//! This module reproduces the numerical case reported by KiThe from the
//! literature-MF1963 reactor fixture in both its low- and high-activation
//! regimes. It is intentionally built through the public `BVP_api` using only
//! symbolic equations, boundary conditions, a mesh, and a full node-aligned
//! initial profile. No KiThe model types or solver facade participate in
//! these tests. All MF-1963 solver-choice experiments live in this module.
//!
//! The independent coordinate is `z` on `[0, 1]`. The six state variables are
//! `theta`, `q`, `C0`, `J0`, `C1`, and `J1`. For the MF-LOW parameter values
//! `delta=1e-4`, `beta=0.05`, `gamma=0.01`, `lambda=0.1`, `D=1e-4`, and
//! `Pe=12017.324603855763`, define
//!
//! ```text
//! S = delta * exp(theta / (1 + beta*theta)) * C0 / (1 + beta*theta)
//! theta' = q / lambda
//! q'     = Pe*q - lambda*Pe^2*S
//! C0'    = J0 / D
//! J0'    = Pe*J0 + D*gamma*Pe^2*S
//! C1'    = J1 / D
//! J1'    = Pe*J1 - D*gamma*Pe^2*S
//! ```
//!
//! The mixed endpoint constraints are `theta(0)=0`, `q(1)=0`, `C0(0)=1`,
//! `J0(1)=0`, `C1(0)=0`, and `J1(1)=0`. MF-LOW (`delta=1e-4`,
//! `Pe=12017.324603855763`) uses the clustered 2049-node seed; MF-HIGH
//! (`delta=0.1`, `Pe=0.5427079731331537`) uses the uniform 301-node seed. The
//! source seeds and SciPy references are copied byte-for-byte from KiThe's
//! `literature_mf1963/fixtures`. The complete seed is supplied in state-major
//! `6 x node_count` form, so the facade must convert it to the route's native
//! packed ordering without resampling the mesh.
//!
//! BVP_sci with Sparse storage is the positive reference route. This suite then
//! attempts the same regime/seed across CorrectionNewton, Frozen/Naive, and
//! Damped using Sparse or automatically inferred Banded storage only. The
//! large Damped/CN/Frozen cases use Banded; BVP_sci is Sparse. Dense storage is
//! deliberately excluded because the stiff discretization has many thousands
//! of unknowns. Damped refinement strategies and Frozen Jacobian refresh
//! policies are exercised as explicit opt-in stories with recorded settings.
//!
//! These are numerical solver acceptance/diagnostic fixtures, not a claim that
//! every route implements the same discretization or must produce identical
//! meshes. A successful route is assessed against its own solve contract and
//! the independent SciPy profile at common coordinates; mesh counts and native
//! telemetry are recorded separately. A non-convergence is useful diagnostic
//! evidence and must not be confused with a successful physical solution.

use super::{
    BvpBoundaryCondition, BvpBoundarySide, BvpNativeSolver, BvpProblem, BvpSciSparseFactorization,
    BvpSolver, BvpSolverConfig, BvpSolverSettings,
};
use crate::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonMatrixLayout, CorrectionNewtonOptions, CorrectionNewtonTelemetryMode,
};
use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedSolverOptions, SolverParams,
};
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use crate::numerical::BVP_Damp::telemetry::{BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode};
use crate::numerical::BVP_api::BvpSolverTelemetrySnapshot;
use crate::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciOptions, BvpSciTelemetryMode,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::time::Instant;

const SEED_CSV: &str = include_str!("fixtures/mf1963_mf_low_six_state_seed.csv");
const SCIPY_CSV: &str = include_str!("fixtures/mf1963_mf_low_six_state_scipy_solution.csv");
const UNIFORM_CSV: &str = include_str!("fixtures/mf1963_mf_low_uniform_six_state_seed.csv");
const HIGH_SEED_CSV: &str = include_str!("fixtures/mf1963_mf_high_six_state_seed.csv");
const HIGH_SCIPY_CSV: &str = include_str!("fixtures/mf1963_mf_high_six_state_scipy_solution.csv");
const STATE_NAMES: [&str; 6] = ["theta", "q", "C0", "J0", "C1", "J1"];
const MF_LOW_DELTA: f64 = 1e-4;
const MF_LOW_PE: f64 = 12017.324603855763;
const MF_HIGH_DELTA: f64 = 0.1;
const MF_HIGH_PE: f64 = 0.5427079731331537;

#[derive(Clone)]
struct Fixture {
    mesh: Vec<f64>,
    values: DMatrix<f64>,
}

fn parse_fixture(csv: &str) -> Fixture {
    let mut mesh = Vec::new();
    let mut rows = Vec::new();
    for (line_index, line) in csv.lines().enumerate() {
        if line_index == 0 || line.trim().is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(|value| value.trim().parse::<f64>().expect("numeric fixture cell"))
            .collect::<Vec<_>>();
        assert_eq!(row.len(), 7, "fixture must contain z and six states");
        mesh.push(row[0]);
        rows.push(row[1..].to_vec());
    }
    assert!(mesh.len() >= 3, "fixture must contain a useful mesh");
    assert!(mesh.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(rows.iter().flatten().all(|value| value.is_finite()));
    let values = DMatrix::from_fn(6, mesh.len(), |state, node| rows[node][state]);
    Fixture { mesh, values }
}

fn clustered_seed() -> Fixture {
    parse_fixture(SEED_CSV)
}

fn scipy_reference_seed() -> Fixture {
    parse_fixture(SCIPY_CSV)
}

fn mf_high_seed() -> Fixture {
    parse_fixture(HIGH_SEED_CSV)
}

fn mf_high_scipy_reference() -> Fixture {
    parse_fixture(HIGH_SCIPY_CSV)
}

fn mf_high_smooth_seed(nodes: usize) -> Fixture {
    let peak_xi = 0.04358419907586729;
    let center = peak_xi / MF_HIGH_PE;
    let steepness = 35.0;
    let logistic = |z: f64| 1.0 / (1.0 + (-steepness * (z - center)).exp());
    let left = logistic(0.0);
    let right = logistic(1.0);
    let normalization = right - left;
    let mut mesh = Vec::with_capacity(nodes);
    let mut values = DMatrix::zeros(6, nodes);
    for node in 0..nodes {
        let z = node as f64 / (nodes - 1) as f64;
        let g = logistic(z);
        let theta = 100.0 * (g - left) / normalization;
        let dtheta_dz = 100.0 * steepness * g * (1.0 - g) / normalization;
        let c1 = 0.01 * theta;
        let j0 = -1e-4 * 0.01 * dtheta_dz;
        mesh.push(z);
        values[(0, node)] = theta;
        values[(1, node)] = 0.1 * dtheta_dz;
        values[(2, node)] = 1.0 - c1;
        values[(3, node)] = j0;
        values[(4, node)] = c1;
        values[(5, node)] = -j0;
    }
    Fixture { mesh, values }
}

fn bisect_fixture(fixture: &Fixture) -> Fixture {
    let mut mesh = Vec::with_capacity(2 * fixture.mesh.len() - 1);
    let mut values = DMatrix::zeros(6, 2 * fixture.mesh.len() - 1);
    for interval in 0..fixture.mesh.len() - 1 {
        let left = 2 * interval;
        let right = left + 1;
        let x0 = fixture.mesh[interval];
        let x1 = fixture.mesh[interval + 1];
        mesh.push(x0);
        mesh.push(0.5 * (x0 + x1));
        for state in 0..6 {
            let y0 = fixture.values[(state, interval)];
            let y1 = fixture.values[(state, interval + 1)];
            values[(state, left)] = y0;
            values[(state, right)] = 0.5 * (y0 + y1);
        }
    }
    mesh.push(*fixture.mesh.last().unwrap());
    let last_node = values.ncols() - 1;
    for state in 0..6 {
        values[(state, last_node)] = fixture.values[(state, fixture.mesh.len() - 1)];
    }
    Fixture { mesh, values }
}

fn damped_required_bounds() -> HashMap<String, (f64, f64)> {
    STATE_NAMES
        .iter()
        .map(|name| ((*name).to_owned(), (-1.0e6, 1.0e6)))
        .collect()
}

fn reference_error_by_state(
    view: crate::numerical::BVP_api::BvpSolutionView<'_>,
    reference: &Fixture,
) -> [f64; 6] {
    let mut max_errors = [0.0_f64; 6];
    // Compare only the original 2049 common coordinates. Refined solver meshes
    // retain these nodes, while the independent CSV is intentionally sampled
    // only on the seed grid.
    for (reference_node, &x) in reference.mesh.iter().enumerate() {
        let right = view.mesh().partition_point(|&solver_x| solver_x < x);
        let (left, right) = if right == view.mesh().len() {
            (view.mesh().len() - 2, view.mesh().len() - 1)
        } else if view.mesh()[right] == x {
            (right, right)
        } else {
            (right.saturating_sub(1), right)
        };
        for state in 0..6 {
            let expected = if left == right {
                view.value(left, state).unwrap()
            } else {
                let fraction = (x - view.mesh()[left]) / (view.mesh()[right] - view.mesh()[left]);
                view.value(left, state).unwrap() * (1.0 - fraction)
                    + view.value(right, state).unwrap() * fraction
            };
            max_errors[state] =
                max_errors[state].max((expected - reference.values[(state, reference_node)]).abs());
        }
    }
    max_errors
}

#[derive(Clone, Copy, Debug, Default)]
struct TrapezoidDefectSummary {
    max_abs_by_state: [f64; 6],
    max_scaled_by_state: [f64; 6],
    interval_by_state: [usize; 6],
    interval_width_by_state: [f64; 6],
}

fn trapezoid_defect_by_state(
    view: crate::numerical::BVP_api::BvpSolutionView<'_>,
    delta: f64,
    pe: f64,
) -> TrapezoidDefectSummary {
    let rhs_at = |node: usize| {
        let theta = view.value(node, 0).unwrap();
        let q = view.value(node, 1).unwrap();
        let c0 = view.value(node, 2).unwrap();
        let j0 = view.value(node, 3).unwrap();
        let j1 = view.value(node, 5).unwrap();
        let source = delta * (theta / (1.0 + 0.05 * theta)).exp() * c0 / (1.0 + 0.05 * theta);
        [
            q / 0.1,
            pe * q - 0.1 * pe.powi(2) * source,
            j0 / 1e-4,
            pe * j0 + 1e-4 * 0.01 * pe.powi(2) * source,
            j1 / 1e-4,
            pe * j1 - 1e-4 * 0.01 * pe.powi(2) * source,
        ]
    };
    let mut summary = TrapezoidDefectSummary {
        max_abs_by_state: [0.0; 6],
        max_scaled_by_state: [0.0; 6],
        interval_by_state: [0; 6],
        interval_width_by_state: [0.0; 6],
    };
    for interval in 0..view.mesh().len() - 1 {
        let h = view.mesh()[interval + 1] - view.mesh()[interval];
        let f0 = rhs_at(interval);
        let f1 = rhs_at(interval + 1);
        for state in 0..6 {
            let y0 = view.value(interval, state).unwrap();
            let y1 = view.value(interval + 1, state).unwrap();
            let defect = (y1 - y0 - 0.5 * h * (f0[state] + f1[state])).abs();
            let scale = 1.0 + y0.abs() + y1.abs() + 0.5 * h * (f0[state].abs() + f1[state].abs());
            let scaled = defect / scale;
            if defect > summary.max_abs_by_state[state] {
                summary.max_abs_by_state[state] = defect;
                summary.interval_by_state[state] = interval;
                summary.interval_width_by_state[state] = h;
            }
            summary.max_scaled_by_state[state] = summary.max_scaled_by_state[state].max(scaled);
        }
    }
    summary
}

fn physical_invariants(view: crate::numerical::BVP_api::BvpSolutionView<'_>) -> String {
    let mut theta_min = f64::INFINITY;
    let mut theta_max = f64::NEG_INFINITY;
    let mut c0_min = f64::INFINITY;
    let mut c1_min = f64::INFINITY;
    let mut closure_defect: f64 = 0.0;
    let mut conversion_defect: f64 = 0.0;
    let mut denominator_min = f64::INFINITY;
    for node in 0..view.mesh().len() {
        let theta = view.value(node, 0).unwrap();
        let c0 = view.value(node, 2).unwrap();
        let c1 = view.value(node, 4).unwrap();
        theta_min = theta_min.min(theta);
        theta_max = theta_max.max(theta);
        c0_min = c0_min.min(c0);
        c1_min = c1_min.min(c1);
        denominator_min = denominator_min.min(1.0 + 0.05 * theta);
        closure_defect = closure_defect.max((c0 + c1 - 1.0).abs());
        conversion_defect = conversion_defect.max((c1 - 0.01 * theta).abs());
    }
    format!(
        "theta=[{theta_min:.6e},{theta_max:.6e}] C0_min={c0_min:.6e} C1_min={c1_min:.6e} min(1+beta*theta)={denominator_min:.6e} closure_inf={closure_defect:.6e} C1-gamma*theta_inf={conversion_defect:.6e}"
    )
}

fn assert_mf_high_physical_profile(profile: &Fixture, label: &str) {
    let theta = profile.values.row(0);
    let c0 = profile.values.row(2);
    let c1 = profile.values.row(4);
    assert!(
        theta
            .iter()
            .all(|value| value.is_finite() && *value >= -1e-8)
    );
    assert!(c0.iter().all(|value| value.is_finite() && *value >= -1e-8));
    assert!(c1.iter().all(|value| value.is_finite() && *value >= -1e-8));
    assert!(
        profile
            .values
            .column_iter()
            .all(|column| (column[2] + column[4] - 1.0).abs() < 1e-8),
        "{label} violated MF-HIGH concentration closure"
    );
    assert!(
        profile
            .values
            .column_iter()
            .all(|column| (column[4] - 0.01 * column[0]).abs() < 1e-7),
        "{label} violated the MF-HIGH conversion invariant"
    );
    assert!(
        theta.iter().all(|value| 1.0 + 0.05 * value > 0.0),
        "{label} crossed the Arrhenius denominator singularity"
    );
}

fn boundary_defect_inf(view: crate::numerical::BVP_api::BvpSolutionView<'_>) -> f64 {
    [
        view.value(0, 0).unwrap().abs(),
        view.value(view.mesh().len() - 1, 1).unwrap().abs(),
        (view.value(0, 2).unwrap() - 1.0).abs(),
        view.value(view.mesh().len() - 1, 3).unwrap().abs(),
        view.value(0, 4).unwrap().abs(),
        view.value(view.mesh().len() - 1, 5).unwrap().abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn telemetry_summary(snapshot: BvpSolverTelemetrySnapshot) -> String {
    match snapshot {
        BvpSolverTelemetrySnapshot::CorrectionNewton(Some(snapshot)) => format!(
            "residual={}, jacobian={}, factorizations={}, linear_solves={}, damping_trials={}, rejected={}, callback={:?}, factor={:?}, linear={:?}",
            snapshot.residual_evaluations,
            snapshot.jacobian_evaluations,
            snapshot.factorizations,
            snapshot.linear_solves,
            snapshot.damping_trials,
            snapshot.damping_rejections,
            snapshot.callback_time,
            snapshot.factorization_time,
            snapshot.linear_solve_time,
        ),
        BvpSolverTelemetrySnapshot::CorrectionNewton(None) => "not collected".to_owned(),
        BvpSolverTelemetrySnapshot::Damped(snapshot)
        | BvpSolverTelemetrySnapshot::Frozen(snapshot) => format!(
            "status={:?}, iterations={}, residual_calls={}, jacobian_refreshes={}, linear_solves={}, refinements={}, logging={:?}, total={:?}, symbolic={:?}, grid={:?}",
            snapshot.last_solve_status,
            snapshot.counters.iterations,
            snapshot.counters.residual_calls,
            snapshot.counters.jacobian_recalculations,
            snapshot.counters.linear_solves,
            snapshot.counters.grid_refinements,
            snapshot.logging_config,
            snapshot.timings.total,
            snapshot.timings.symbolic_operations,
            snapshot.timings.grid_refinement,
        ),
        BvpSolverTelemetrySnapshot::Sci(snapshot) => format!(
            "rhs_calls={}, jacobian_calls={}, newton_iterations={}, jacobian_refreshes={}, backtracks={}",
            snapshot.rhs_calls,
            snapshot.jacobian_calls,
            snapshot.newton_iterations,
            snapshot.newton_jacobian_refreshes,
            snapshot.newton_backtracking_trials,
        ),
    }
}

fn print_frozen_iteration_trace(
    regime: &str,
    label: &str,
    trace: crate::numerical::BVP_Damp::telemetry::BvpFrozenIterationTrace,
) {
    eprintln!(
        "{regime} | {label} | Frozen Newton trace: retained={}, dropped={}",
        trace.events.len(),
        trace.dropped_events
    );
    for event in trace.events {
        eprintln!(
            "{regime} | Frozen step={} residual_l2={:.6e} correction_l2={:.6e} correction_inf={:.6e}@{} iterate_inf={:.6e}->{:.6e} jacobian_age={} refreshed={}",
            event.iteration,
            event.residual_l2,
            event.correction_l2,
            event.correction_inf,
            event.correction_inf_index,
            event.current_iterate_inf,
            event.next_iterate_inf,
            event.jacobian_age,
            event.jacobian_refreshed
        );
    }
    if let Some(failure) = trace.failure {
        eprintln!(
            "{regime} | Frozen attempted iteration={} failed at {:?}",
            failure.iteration, failure.stage
        );
    }
}

fn mesh_contains_with_roundoff_tolerance(mesh: &[f64], x: f64) -> bool {
    let index = mesh.partition_point(|mesh_x| *mesh_x < x);
    [index.checked_sub(1), (index < mesh.len()).then_some(index)]
        .into_iter()
        .flatten()
        .any(|candidate| {
            let mesh_x = mesh[candidate];
            (mesh_x - x).abs() <= 16.0 * f64::EPSILON * x.abs().max(mesh_x.abs()).max(1.0)
        })
}

fn compact_case_label(label: &str) -> &str {
    match label {
        label if label.contains("BVP_sci/Sparse/KiThe-profile") => "sci/KiThe",
        label if label.contains("BVP_sci/Sparse/independent-smooth") => "sci/smooth",
        label if label.contains("CorrectionNewton/BandedAuto") => "CorrectionNewton",
        label if label.contains("Damped/") && label.contains("TwoPoint") => "Damped/TwoPoint",
        label if label.contains("Damped/") && label.contains("Pearson") => "Damped/Pearson",
        label if label.contains("Damped/") && label.contains("GrcarSmooke") => "Damped/Grcar",
        label if label.contains("Damped/") && label.contains("DoublePoints") => {
            "Damped/DoublePoints"
        }
        label if label.contains("Frozen/") && label.contains("Frozen_naive") => "Frozen/default",
        label if label.contains("Frozen/") && label.contains("at_high_morm") => {
            "Frozen/refresh-large"
        }
        label if label.contains("Frozen/") && label.contains("every_m") => "Frozen/every-m",
        _ => label,
    }
}

fn problem_for(fixture: &Fixture, delta: f64, pe: f64) -> BvpProblem {
    let source = format!("{delta:.17e}*exp(theta/(1+0.05*theta))*C0/(1+0.05*theta)");
    let equations = vec![
        Expr::parse_expression("q/0.1"),
        Expr::parse_expression(&format!("{pe}*q-0.1*{pe}^2*({source})")),
        Expr::parse_expression("J0/1e-4"),
        Expr::parse_expression(&format!("{pe}*J0+1e-4*0.01*{pe}^2*({source})")),
        Expr::parse_expression("J1/1e-4"),
        Expr::parse_expression(&format!("{pe}*J1-1e-4*0.01*{pe}^2*({source})")),
    ];
    BvpProblem::new(
        equations,
        STATE_NAMES.iter().map(|name| (*name).to_owned()).collect(),
        "z",
        fixture.mesh.clone(),
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 0.0,
            },
            BvpBoundaryCondition {
                state: 1,
                side: BvpBoundarySide::Right,
                value: 0.0,
            },
            BvpBoundaryCondition {
                state: 2,
                side: BvpBoundarySide::Left,
                value: 1.0,
            },
            BvpBoundaryCondition {
                state: 3,
                side: BvpBoundarySide::Right,
                value: 0.0,
            },
            BvpBoundaryCondition {
                state: 4,
                side: BvpBoundarySide::Left,
                value: 0.0,
            },
            BvpBoundaryCondition {
                state: 5,
                side: BvpBoundarySide::Right,
                value: 0.0,
            },
        ],
    )
    .expect("MF-1963 symbolic BVP schema should be valid")
}

fn problem(fixture: &Fixture) -> BvpProblem {
    problem_for(fixture, MF_LOW_DELTA, MF_LOW_PE)
}

fn run_and_report_case(
    label: &str,
    problem: BvpProblem,
    config: BvpSolverConfig,
    regime: &str,
    delta: f64,
    pe: f64,
    reference_csv: &str,
    require_refinement: bool,
) -> Option<([f64; 6], Fixture)> {
    let started = Instant::now();
    let requested_mesh = problem.mesh.clone();
    let mut solver = match BvpSolver::new(problem, config) {
        Ok(solver) => solver,
        Err(error) => {
            eprintln!(
                "{regime} | {label} | prepare=error | elapsed={:?} | {error:?}",
                started.elapsed()
            );
            return None;
        }
    };
    let prepared = started.elapsed();
    let solve_started = Instant::now();
    let reference = parse_fixture(reference_csv);
    match solver.solve().map(|_| ()) {
        Ok(()) => {
            if let Some(trace) = solver.frozen_iteration_trace_snapshot() {
                print_frozen_iteration_trace(regime, label, trace);
            }
            let (mesh_nodes, reference_errors, trapezoid_defects, boundary_defect, invariants) =
                solver
                    .solution_view()
                    .map(|view| {
                        (
                            view.mesh().len(),
                            reference_error_by_state(view, &reference),
                            trapezoid_defect_by_state(view, delta, pe),
                            boundary_defect_inf(view),
                            physical_invariants(view),
                        )
                    })
                    .unwrap_or((
                        0,
                        [f64::NAN; 6],
                        TrapezoidDefectSummary::default(),
                        f64::NAN,
                        "unavailable".into(),
                    ));
            let view = solver
                .solution_view()
                .expect("successful solve must retain a solution view");
            let solution = Fixture {
                mesh: view.mesh().to_vec(),
                values: DMatrix::from_fn(STATE_NAMES.len(), view.mesh().len(), |state, node| {
                    view.value(node, state)
                        .expect("all fixture states are present")
                }),
            };
            assert!(
                requested_mesh
                    .iter()
                    .all(|x| mesh_contains_with_roundoff_tolerance(view.mesh(), *x)),
                "{label} must preserve every original seed node within floating-point roundoff"
            );
            assert!(boundary_defect <= 1e-7, "{label} violated an endpoint BC");
            if require_refinement {
                let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot()
                else {
                    panic!("Damped route must retain its native telemetry variant");
                };
                assert!(snapshot.counters.grid_refinements > 0);
            }
            eprintln!(
                "{regime} | {label} | converged | nodes={mesh_nodes} | bc={boundary_defect:.3e} | prepare={prepared:?} | solve={:?}",
                solve_started.elapsed()
            );
            let report_label = compact_case_label(label);
            for state in 0..6 {
                eprintln!(
                    "{regime} | {report_label} | s{state} scipy={:.3e} trap={:.3e} scaled={:.3e}",
                    reference_errors[state],
                    trapezoid_defects.max_abs_by_state[state],
                    trapezoid_defects.max_scaled_by_state[state]
                );
                eprintln!(
                    "{regime} | {report_label} | s{state} defect interval={} h={:.3e}",
                    trapezoid_defects.interval_by_state[state],
                    trapezoid_defects.interval_width_by_state[state]
                );
            }
            for invariant in invariants.split_whitespace() {
                eprintln!("{regime} | {report_label} | {invariant}");
            }
            for metric in telemetry_summary(solver.telemetry_snapshot()).split(", ") {
                eprintln!("{regime} | {report_label} | telemetry {metric}");
            }
            Some((reference_errors, solution))
        }
        Err(error) => {
            let mesh_nodes = solver
                .solution_view()
                .map(|view| view.mesh().len())
                .unwrap_or(0);
            eprintln!(
                "{regime} | {label} | status=error | nodes={mesh_nodes} | prepare={prepared:?} | solve={:?} | error={error:?} | telemetry=[{}]",
                solve_started.elapsed(),
                telemetry_summary(solver.telemetry_snapshot())
            );
            if let Some(trace) = solver.frozen_iteration_trace_snapshot() {
                print_frozen_iteration_trace(regime, label, trace);
            }
            None
        }
    }
}

fn run_and_report(
    label: &str,
    problem: BvpProblem,
    config: BvpSolverConfig,
) -> Option<([f64; 6], Fixture)> {
    run_and_report_case(
        label,
        problem,
        config,
        "MF-LOW",
        MF_LOW_DELTA,
        MF_LOW_PE,
        SCIPY_CSV,
        label.starts_with("Damped/"),
    )
}

fn sci_config(seed: DMatrix<f64>) -> BvpSolverConfig {
    BvpSolverConfig::Sci {
        options: BvpSciOptions::default()
            .with_execution(BvpSciExecution::Lambdify)
            .with_assembly(BvpSciAssembly::ExprLegacy)
            .with_matrix_layout(BvpSciMatrixLayout::Sparse)
            .with_tolerance(2e-7)
            .with_boundary_tolerance(2e-7)
            .with_limits(6000, 100, 20)
            .with_telemetry(BvpSciTelemetryMode::Detailed),
        sparse_factorization: BvpSciSparseFactorization::StandardFaer,
        aot_config: None,
        initial_guess: seed,
    }
}

fn correction_newton_config(seed: DMatrix<f64>) -> BvpSolverConfig {
    let mut options = CorrectionNewtonOptions::default();
    options.matrix_layout = CorrectionNewtonMatrixLayout::BandedAuto;
    options.absolute_tolerance = 2e-7;
    options.default_relative_tolerance = 2e-7;
    options.relative_tolerances = STATE_NAMES
        .iter()
        .map(|name| ((*name).to_owned(), 2e-7))
        .collect();
    options.max_iterations = 100;
    options.telemetry_mode = CorrectionNewtonTelemetryMode::Timings;
    BvpSolverSettings::CorrectionNewton(options).with_node_aligned_initial_guess(seed)
}

fn correction_newton_frozen_equivalent_config(seed: DMatrix<f64>) -> BvpSolverConfig {
    let full_state_size = STATE_NAMES.len() * seed.ncols();
    let mapped_tolerance = 2.0e-7 / (full_state_size as f64).sqrt();
    let mut options = CorrectionNewtonOptions::default();
    options.matrix_layout = CorrectionNewtonMatrixLayout::BandedAuto;
    options.absolute_tolerance = mapped_tolerance;
    options.default_relative_tolerance = 1.0e-30;
    options.relative_tolerances = STATE_NAMES
        .iter()
        .map(|name| ((*name).to_owned(), 1.0e-30))
        .collect();
    options.max_iterations = 100;
    options.telemetry_mode = CorrectionNewtonTelemetryMode::Timings;
    BvpSolverSettings::CorrectionNewton(options).with_node_aligned_initial_guess(seed)
}

fn frozen_config(seed: DMatrix<f64>) -> BvpSolverConfig {
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .trapezoid_derivative()
        .with_tolerance(2e-7)
        .with_max_iterations(100)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(128),
        );
    assert_eq!(
        options.generated_backend_config.bvp_logging_mode,
        BvpLoggingMode::Detailed,
        "MF-LOW Frozen configuration must request detailed iteration events"
    );
    assert_eq!(options.generated_backend_config.bvp_logging_max_events, 128);
    let config = BvpSolverSettings::Frozen(options).with_node_aligned_initial_guess(seed);
    if let BvpSolverConfig::FrozenNodeAligned { options, .. } = &config {
        assert_eq!(
            options.generated_backend_config.bvp_logging_mode,
            BvpLoggingMode::Detailed
        );
        assert_eq!(options.generated_backend_config.bvp_logging_max_events, 128);
    }
    config
}

fn frozen_callback_residual_inf(fixture: &Fixture) -> (f64, f64) {
    let mut solver = BvpSolver::new(problem(fixture), frozen_config(fixture.values.clone()))
        .expect("Frozen residual-probe route should prepare");
    let BvpNativeSolver::Frozen(native) = solver.native_mut() else {
        panic!("expected Frozen native solver");
    };
    native
        .try_eq_generate()
        .expect("Frozen callbacks should generate for residual probe");
    let column_major = nalgebra::DVector::from_column_slice(native.initial_guess.as_slice());
    let mut node_major = Vec::with_capacity(native.initial_guess.len());
    for node in 0..=native.n_steps {
        for (state, state_name) in native.values.iter().enumerate() {
            let conditions = &native.BorderConditions[state_name];
            let fixed_node = match conditions[0].0 {
                0 => 0,
                1 => native.n_steps,
                side => panic!("unexpected Frozen boundary side {side}"),
            };
            if node == fixed_node {
                continue;
            }
            let reduced_column = if fixed_node == 0 { node - 1 } else { node };
            node_major.push(native.initial_guess[(state, reduced_column)]);
        }
    }
    native.y = Vectors_type_casting(&column_major, native.y.vec_type().to_string());
    let column_major_inf = native
        .fun
        .try_call(native.p, &*native.y)
        .expect("Frozen residual callback should evaluate at the flattened profile")
        .to_DVectorType()
        .amax();
    let packed_node_major = nalgebra::DVector::from_column_slice(&node_major);
    native.y = Vectors_type_casting(&packed_node_major, native.y.vec_type().to_string());
    let node_major_inf = native
        .fun
        .try_call(native.p, &*native.y)
        .expect("Frozen residual callback should evaluate at the node-major profile")
        .to_DVectorType()
        .amax();
    (column_major_inf, node_major_inf)
}

fn damped_config(
    seed: DMatrix<f64>,
    method: GridRefinementMethod,
    max_refinements: usize,
) -> BvpSolverConfig {
    let options = DampedSolverOptions::banded_damped()
        .with_banded_lambdify()
        .trapezoid_derivative()
        .with_abs_tolerance(2e-7)
        .with_rel_tolerance(
            STATE_NAMES
                .iter()
                .map(|name| ((*name).to_owned(), 2e-7))
                .collect(),
        )
        .with_bounds(damped_required_bounds())
        .with_max_iterations(100)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(3),
            max_damp_iter: Some(10),
            damp_factor: Some(0.5),
            adaptive: Some(AdaptiveGridConfig {
                version: 1,
                max_refinements,
                grid_method: method,
            }),
        }));
    BvpSolverSettings::Damped(options).with_node_aligned_initial_guess(seed)
}

#[test]
fn mf1963_mflow_fixture_contract_is_valid_and_source_sized() {
    let clustered = clustered_seed();
    let reference = parse_fixture(SCIPY_CSV);
    let uniform = parse_fixture(UNIFORM_CSV);
    assert_eq!(clustered.mesh.len(), 2049);
    assert_eq!(reference.mesh.len(), clustered.mesh.len());
    assert_eq!(uniform.mesh.len(), clustered.mesh.len());
    assert_eq!(clustered.mesh.first(), Some(&0.0));
    assert_eq!(clustered.mesh.last(), Some(&1.0));
    assert!(
        clustered
            .mesh
            .iter()
            .zip(&uniform.mesh)
            .any(|(a, b)| a != b)
    );
    assert_eq!(problem(&clustered).mesh, clustered.mesh);
    let high_seed = mf_high_seed();
    let high_reference = mf_high_scipy_reference();
    assert_eq!(high_seed.mesh.len(), 301);
    assert_eq!(high_reference.mesh, high_seed.mesh);
    assert_eq!(
        problem_for(&high_seed, MF_HIGH_DELTA, MF_HIGH_PE).mesh,
        high_seed.mesh
    );
}

#[test]
#[ignore = "full 2049-node stiff reference solve; run as part of the BVP_api story matrix"]
fn mf1963_mflow_scipy_sparse_reference_solves_clustered_seed() {
    let seed = clustered_seed();
    let result = run_and_report(
        "BVP_sci/Sparse",
        problem(&seed),
        sci_config(seed.values.clone()),
    );
    let errors = result
        .expect("BVP_sci Sparse is the expected positive control")
        .0;
    assert!(
        errors.iter().copied().fold(0.0, f64::max) < 2e-5,
        "BVP_sci should match the independent profile at original seed nodes: {errors:?}"
    );
}

#[test]
#[ignore = "full 2049-node stiff CorrectionNewton experiment"]
fn mf1963_mflow_correction_newton_banded_attempts_clustered_seed() {
    let seed = clustered_seed();
    let mut options = CorrectionNewtonOptions::default();
    options.matrix_layout = CorrectionNewtonMatrixLayout::BandedAuto;
    options.absolute_tolerance = 2e-7;
    options.default_relative_tolerance = 2e-7;
    options.relative_tolerances = STATE_NAMES
        .iter()
        .map(|name| ((*name).to_owned(), 2e-7))
        .collect();
    options.max_iterations = 100;
    options.telemetry_mode = CorrectionNewtonTelemetryMode::Timings;
    let config = BvpSolverSettings::CorrectionNewton(options)
        .with_node_aligned_initial_guess(seed.values.clone());
    assert!(
        run_and_report("CorrectionNewton/BandedAuto", problem(&seed), config).is_some(),
        "CorrectionNewton should return a typed successful result on this fixture"
    );
}

#[test]
#[ignore = "full 2049-node Frozen diagnostic; confirms failure is not published as a solution"]
fn mf1963_mflow_frozen_banded_attempts_clustered_seed() {
    let seed = clustered_seed();
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .trapezoid_derivative()
        .with_tolerance(2e-7)
        .with_max_iterations(100)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(128),
        );
    let config =
        BvpSolverSettings::Frozen(options).with_node_aligned_initial_guess(seed.values.clone());
    assert!(run_and_report("Frozen/Banded", problem(&seed), config).is_none());
}

#[test]
#[ignore = "MF-LOW diagnostic: isolate Frozen full-step divergence from stale Jacobian reuse"]
fn mf1963_mflow_frozen_banded_refreshes_jacobian_after_large_correction() {
    let seed = clustered_seed();
    let (_, correction_profile) = run_and_report(
        "CorrectionNewton/BandedAuto/control-for-Frozen-refresh",
        problem(&seed),
        correction_newton_frozen_equivalent_config(seed.values.clone()),
    )
    .expect("CorrectionNewton control should solve the clustered seed");
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .trapezoid_derivative()
        .with_tolerance(2e-7)
        .with_max_iterations(100)
        .with_strategy_params(Some(HashMap::from([(
            "at_high_morm".to_owned(),
            Some(vec![1e-6]),
        )])))
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(128),
        );
    let config =
        BvpSolverSettings::Frozen(options).with_node_aligned_initial_guess(seed.values.clone());
    let (_, frozen_profile) = run_and_report(
        "Frozen/Banded/refresh-on-correction",
        problem(&seed),
        config,
    )
    .expect("fresh-Jacobian Frozen diagnostic should converge");
    let profile_distance = (&correction_profile.values - &frozen_profile.values).amax();
    eprintln!(
        "MF-LOW | CorrectionNewton vs Frozen/refresh-on-correction | max_profile_distance={profile_distance:.6e}"
    );
    assert!(profile_distance < 1e-4);

    let periodic_options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .trapezoid_derivative()
        .with_tolerance(2e-7)
        .with_max_iterations(100)
        .with_strategy_params(Some(HashMap::from([(
            "every_m".to_owned(),
            Some(vec![1.0]),
        )])))
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(128),
        );
    let periodic_config = BvpSolverSettings::Frozen(periodic_options)
        .with_node_aligned_initial_guess(seed.values.clone());
    let periodic_result =
        run_and_report("Frozen/Banded/every-m-1", problem(&seed), periodic_config);
    eprintln!(
        "MF-LOW | Frozen/every-m-1 | converged={}",
        periodic_result.is_some()
    );
}

#[test]
#[ignore = "full 4097-node stiff CorrectionNewton higher-resolution experiment"]
fn mf1963_mflow_correction_newton_banded_attempts_bisected_clustered_seed() {
    let seed = bisect_fixture(&clustered_seed());
    let mut options = CorrectionNewtonOptions::default();
    options.matrix_layout = CorrectionNewtonMatrixLayout::BandedAuto;
    options.absolute_tolerance = 2e-7;
    options.default_relative_tolerance = 2e-7;
    options.relative_tolerances = STATE_NAMES
        .iter()
        .map(|name| ((*name).to_owned(), 2e-7))
        .collect();
    options.max_iterations = 100;
    options.telemetry_mode = CorrectionNewtonTelemetryMode::Timings;
    let config = BvpSolverSettings::CorrectionNewton(options)
        .with_node_aligned_initial_guess(seed.values.clone());
    assert!(
        run_and_report(
            "CorrectionNewton/BandedAuto/4097-nodes",
            problem(&seed),
            config
        )
        .is_some(),
        "CorrectionNewton should return a typed result on the bisected nonuniform mesh"
    );
}

#[test]
#[ignore = "full 4097-node Frozen diagnostic; records current non-convergence"]
fn mf1963_mflow_frozen_banded_attempts_bisected_clustered_seed() {
    let seed = bisect_fixture(&clustered_seed());
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .trapezoid_derivative()
        .with_tolerance(2e-7)
        .with_max_iterations(100)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed);
    let config =
        BvpSolverSettings::Frozen(options).with_node_aligned_initial_guess(seed.values.clone());
    assert!(run_and_report("Frozen/Banded/4097-nodes", problem(&seed), config).is_none());
}

#[test]
#[ignore = "full 2049-node stiff CorrectionNewton reference-seeded basin experiment"]
fn mf1963_mflow_correction_newton_banded_attempts_scipy_reference_seed() {
    let seed = scipy_reference_seed();
    let mut options = CorrectionNewtonOptions::default();
    options.matrix_layout = CorrectionNewtonMatrixLayout::BandedAuto;
    options.absolute_tolerance = 2e-7;
    options.default_relative_tolerance = 2e-7;
    options.relative_tolerances = STATE_NAMES
        .iter()
        .map(|name| ((*name).to_owned(), 2e-7))
        .collect();
    options.max_iterations = 100;
    options.telemetry_mode = CorrectionNewtonTelemetryMode::Timings;
    let config = BvpSolverSettings::CorrectionNewton(options)
        .with_node_aligned_initial_guess(seed.values.clone());
    assert!(
        run_and_report(
            "CorrectionNewton/BandedAuto/SciPy-seed",
            problem(&seed),
            config
        )
        .is_some(),
        "CorrectionNewton should accept the reference profile as a full-node seed"
    );
}

#[test]
#[ignore = "full 2049-node CorrectionNewton run with Frozen-equivalent raw L2 stopping threshold"]
fn mf1963_mflow_correction_newton_banded_attempts_frozen_equivalent_stopping() {
    let seed = clustered_seed();
    // CorrectionNewton stops on weighted RMS <= 1. With a uniform absolute
    // scale tolerance/sqrt(full_state_size), this corresponds to a raw L2
    // correction threshold of approximately the Frozen tolerance.
    let config = correction_newton_frozen_equivalent_config(seed.values.clone());
    assert!(
        run_and_report(
            "CorrectionNewton/BandedAuto/Frozen-equivalent-L2-stop",
            problem(&seed),
            config
        )
        .is_some(),
        "CorrectionNewton should complete with the Frozen-equivalent raw L2 threshold"
    );
}

#[test]
#[ignore = "full cross-route test: run Frozen from a high-accuracy CorrectionNewton root"]
fn mf1963_mflow_frozen_cross_seeded_from_tight_correction_newton_root() {
    let initial = clustered_seed();
    let (_, tight_correction_profile) = run_and_report(
        "CrossSeed/source-CorrectionNewton-tight",
        problem(&initial),
        correction_newton_frozen_equivalent_config(initial.values.clone()),
    )
    .expect("tight CorrectionNewton source solve should converge");
    let (frozen_column_major_residual_inf, frozen_node_major_residual_inf) =
        frozen_callback_residual_inf(&tight_correction_profile);
    eprintln!(
        "MF-LOW | Frozen callback residual at tight CorrectionNewton root | column_major_inf={frozen_column_major_residual_inf:.6e} | node_major_inf={frozen_node_major_residual_inf:.6e}"
    );
    assert!(
        frozen_column_major_residual_inf > 1.0,
        "the legacy matrix flattening should be observably wrong for mixed endpoints"
    );
    assert!(
        frozen_node_major_residual_inf < 1e-6,
        "node-major callback residual should be small at the tight CorrectionNewton root"
    );
    let (_, frozen_profile) = run_and_report(
        "CrossSeed/Frozen-from-tight-CorrectionNewton",
        problem(&tight_correction_profile),
        frozen_config(tight_correction_profile.values.clone()),
    )
    .expect("Frozen should converge from the tight CorrectionNewton profile");
    let profile_distance = (&frozen_profile.values - &tight_correction_profile.values).amax();
    eprintln!(
        "MF-LOW | cross-seed tight CorrectionNewton -> Frozen | max_profile_distance={profile_distance:.6e}"
    );
    assert!(
        profile_distance < 1e-4,
        "Frozen must preserve the supplied high-accuracy root when its mixed-endpoint seed is packed correctly"
    );
}

#[test]
#[ignore = "full cross-route roundtrip through a converged Frozen profile"]
fn mf1963_mflow_correction_newton_cross_seeded_from_frozen_profile() {
    let initial = clustered_seed();
    let (_, correction_profile) = run_and_report(
        "CrossSeed/source-CorrectionNewton-for-Frozen",
        problem(&initial),
        correction_newton_frozen_equivalent_config(initial.values.clone()),
    )
    .expect("high-accuracy CorrectionNewton source solve should converge");
    let (_, frozen_profile) = run_and_report(
        "CrossSeed/Frozen-from-CorrectionNewton",
        problem(&correction_profile),
        frozen_config(correction_profile.values.clone()),
    )
    .expect("Frozen should preserve the supplied high-accuracy root");
    let (_, correction_profile) = run_and_report(
        "CrossSeed/CorrectionNewton-from-Frozen",
        problem(&frozen_profile),
        correction_newton_config(frozen_profile.values.clone()),
    )
    .expect("CorrectionNewton should converge from the Frozen solution profile");
    let profile_distance = (&correction_profile.values - &frozen_profile.values).amax();
    eprintln!(
        "MF-LOW | cross-seed CorrectionNewton -> Frozen -> CorrectionNewton | max_profile_distance={profile_distance:.6e}"
    );
    assert!(profile_distance < 1e-4);
}

#[test]
#[ignore = "full cross-seed experiment: initialize Frozen from the default CorrectionNewton profile"]
fn mf1963_mflow_frozen_cross_seeded_from_correction_newton_profile() {
    let initial = clustered_seed();
    let (_, correction_profile) = run_and_report(
        "CrossSeed/source-CorrectionNewton",
        problem(&initial),
        correction_newton_config(initial.values.clone()),
    )
    .expect("CorrectionNewton source solve should converge");
    let (_, frozen_profile) = run_and_report(
        "CrossSeed/Frozen-from-CorrectionNewton",
        problem(&correction_profile),
        frozen_config(correction_profile.values.clone()),
    )
    .expect("Frozen should converge from the CorrectionNewton solution profile");
    let profile_distance = (&frozen_profile.values - &correction_profile.values).amax();
    eprintln!(
        "MF-LOW | cross-seed CorrectionNewton -> Frozen | max_profile_distance={profile_distance:.6e}"
    );
}

#[test]
#[ignore = "full 2049-node Frozen diagnostic with SciPy seed; records current non-finite failure"]
fn mf1963_mflow_frozen_banded_attempts_scipy_reference_seed() {
    let seed = scipy_reference_seed();
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .trapezoid_derivative()
        .with_tolerance(2e-7)
        .with_max_iterations(100)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed);
    let config =
        BvpSolverSettings::Frozen(options).with_node_aligned_initial_guess(seed.values.clone());
    assert!(run_and_report("Frozen/Banded/SciPy-seed", problem(&seed), config).is_none());
}

fn run_damped_story(label: &str, method: GridRefinementMethod) {
    let seed = clustered_seed();
    let config = damped_config(seed.values.clone(), method, 2);
    assert!(
        run_and_report(label, problem(&seed), config).is_some(),
        "Damped should return a typed successful result for {label}"
    );
}

#[test]
#[ignore = "full 2049-node stiff Damped TwoPoint refinement experiment"]
fn mf1963_mflow_damped_two_point_refinement_attempts_clustered_seed() {
    run_damped_story(
        "Damped/Banded/TwoPoint",
        GridRefinementMethod::TwoPoint(0.01, 0.01, 2.0),
    );
}

#[test]
#[ignore = "full 2049-node stiff Damped Pearson refinement experiment"]
fn mf1963_mflow_damped_pearson_refinement_attempts_clustered_seed() {
    run_damped_story(
        "Damped/Banded/Pearson",
        GridRefinementMethod::Pearson(1e-3, 1.4),
    );
}

#[test]
#[ignore = "full 2049-node stiff Damped Grcar-Smooke refinement experiment"]
fn mf1963_mflow_damped_grcar_smooke_refinement_attempts_clustered_seed() {
    run_damped_story(
        "Damped/Banded/GrcarSmooke",
        GridRefinementMethod::GrcarSmooke(1e-3, 1e-2, 1.4),
    );
}

#[test]
#[ignore = "full 2049-node stiff Damped DoublePoints refinement experiment"]
fn mf1963_mflow_damped_double_points_refinement_attempts_clustered_seed() {
    let seed = clustered_seed();
    let config = damped_config(seed.values.clone(), GridRefinementMethod::DoublePoints, 4);
    assert!(
        run_and_report(
            "Damped/Banded/DoublePoints/4-refinements",
            problem(&seed),
            config
        )
        .is_some(),
        "Damped DoublePoints should return a typed successful result"
    );
}

#[test]
#[ignore = "MF-1963 MF-HIGH solver/seed suitability matrix; retains all routes in this story module"]
fn mf1963_mf_high_solver_seed_strategy_matrix() {
    let seed = mf_high_seed();
    let smooth = mf_high_smooth_seed(seed.mesh.len());
    let reference = mf_high_scipy_reference();
    let seed_shape_difference = (&seed.values - &smooth.values).amax();
    eprintln!(
        "MF-HIGH | source-vs-reconstructed-smooth seed max distance={seed_shape_difference:.6e}"
    );

    for (seed_name, fixture) in [
        ("KiThe-profile-seed", &seed),
        ("independent-smooth-seed", &smooth),
    ] {
        let result = run_and_report_case(
            &format!("BVP_sci/Sparse/{seed_name}"),
            problem_for(fixture, MF_HIGH_DELTA, MF_HIGH_PE),
            sci_config(fixture.values.clone()),
            "MF-HIGH",
            MF_HIGH_DELTA,
            MF_HIGH_PE,
            HIGH_SCIPY_CSV,
            false,
        )
        .unwrap_or_else(|| panic!("BVP_sci control failed from {seed_name}"));
        assert!(
            result.0.iter().copied().fold(0.0_f64, f64::max) < 2e-5,
            "BVP_sci should reproduce the MF-HIGH reference from {seed_name}: {:?}",
            result.0
        );
    }

    let correction = correction_newton_config(seed.values.clone());
    let correction_result = run_and_report_case(
        "CorrectionNewton/BandedAuto/KiThe-profile-seed",
        problem_for(&seed, MF_HIGH_DELTA, MF_HIGH_PE),
        correction,
        "MF-HIGH",
        MF_HIGH_DELTA,
        MF_HIGH_PE,
        HIGH_SCIPY_CSV,
        false,
    )
    .expect("CorrectionNewton should converge on the MF-HIGH KiThe seed");
    assert_mf_high_physical_profile(&correction_result.1, "CorrectionNewton");

    let refinement_cases = [
        (
            "TwoPoint",
            GridRefinementMethod::TwoPoint(0.01, 0.01, 2.0),
            2,
        ),
        ("Pearson", GridRefinementMethod::Pearson(1e-3, 1.4), 2),
        (
            "GrcarSmooke",
            GridRefinementMethod::GrcarSmooke(1e-3, 1e-2, 1.4),
            2,
        ),
        ("DoublePoints", GridRefinementMethod::DoublePoints, 4),
    ];
    for (strategy_name, method, max_refinements) in refinement_cases {
        let config = damped_config(seed.values.clone(), method, max_refinements);
        let result = run_and_report_case(
            &format!("Damped/Banded/Trapezoid/{strategy_name}/KiThe-profile-seed"),
            problem_for(&seed, MF_HIGH_DELTA, MF_HIGH_PE),
            config,
            "MF-HIGH",
            MF_HIGH_DELTA,
            MF_HIGH_PE,
            HIGH_SCIPY_CSV,
            true,
        )
        .unwrap_or_else(|| panic!("Damped/{strategy_name} should converge on MF-HIGH"));
        assert_mf_high_physical_profile(&result.1, &format!("Damped/{strategy_name}"));
    }

    let frozen_strategies = [
        ("Frozen_naive", None),
        (
            "at_high_morm=1e-6",
            Some(HashMap::from([(
                "at_high_morm".to_owned(),
                Some(vec![1e-6]),
            )])),
        ),
        (
            "every_m=1",
            Some(HashMap::from([("every_m".to_owned(), Some(vec![1.0]))])),
        ),
    ];
    let mut frozen_profiles = Vec::new();
    for (strategy_name, strategy_params) in frozen_strategies {
        let mut options = FrozenSolverOptions::banded_frozen()
            .with_banded_lambdify()
            .trapezoid_derivative()
            .with_tolerance(2e-7)
            .with_max_iterations(100)
            .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
            .with_bvp_logging_config(
                BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(128),
            );
        if let Some(params) = strategy_params {
            options = options.with_strategy_params(Some(params));
        }
        let config =
            BvpSolverSettings::Frozen(options).with_node_aligned_initial_guess(seed.values.clone());
        let result = run_and_report_case(
            &format!("Frozen/Banded/Trapezoid/{strategy_name}/KiThe-profile-seed"),
            problem_for(&seed, MF_HIGH_DELTA, MF_HIGH_PE),
            config,
            "MF-HIGH",
            MF_HIGH_DELTA,
            MF_HIGH_PE,
            HIGH_SCIPY_CSV,
            false,
        )
        .unwrap_or_else(|| panic!("Frozen/{strategy_name} should converge on MF-HIGH"));
        assert_mf_high_physical_profile(&result.1, &format!("Frozen/{strategy_name}"));
        frozen_profiles.push((strategy_name, result.1));
    }

    for (strategy_name, profile) in frozen_profiles {
        assert_eq!(profile.mesh.len(), correction_result.1.mesh.len());
        assert!(
            profile
                .mesh
                .iter()
                .zip(&correction_result.1.mesh)
                .all(|(&left, &right)| mesh_contains_with_roundoff_tolerance(
                    std::slice::from_ref(&right),
                    left
                ))
        );
        let distance = (&profile.values - &correction_result.1.values).amax();
        eprintln!(
            "MF-HIGH | CorrectionNewton vs Frozen/{strategy_name} | max profile distance={distance:.6e}"
        );
        assert!(
            distance < 2e-5,
            "MF-HIGH fixed-mesh CorrectionNewton/Frozen profiles unexpectedly diverged"
        );
    }

    // Keep the independent reference available here so this matrix documents
    // the source profile separately from the two deliberately named seeds.
    assert_eq!(reference.mesh, seed.mesh);
}
