//! Numerical-only regression for Damped initial-guess packing.
//!
//! The fixture is a six-state hot-branch BVP seed. Its public input is a
//! state-major profile with each state's constrained endpoint omitted.

use crate::numerical::BVP_Damp::telemetry::{BvpLoggingConfig, BvpLoggingMode};
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{DampedSolverOptions, SolverParams};
use crate::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSolver, BvpSolverConfig,
};
use crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::collections::HashMap;

const INTERVALS: usize = 301;
const XI_MAX: f64 = 0.542_707_973_133_153_7;
const DELTA: f64 = 0.1;
const BETA: f64 = 0.05;
const GAMMA: f64 = 0.01;
const LAMBDA: f64 = 0.1;
const RHO_D: f64 = 1.0e-4;

fn hot_seed_fixture() -> Vec<[f64; 7]> {
    include_str!("fixtures/rst_damped_initial_guess_seed_301.csv")
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let values = line
                .split(',')
                .map(|field| field.parse::<f64>().expect("numeric seed fixture"))
                .collect::<Vec<_>>();
            values
                .try_into()
                .expect("fixture row must contain x and six states")
        })
        .collect()
}

fn build_problem_and_seed() -> (BvpProblem, DMatrix<f64>, DMatrix<f64>) {
    let fixture = hot_seed_fixture();
    assert_eq!(fixture.len(), INTERVALS + 1);

    let xi = format!("{XI_MAX:.17e}");
    let delta = format!("{DELTA:.17e}");
    let beta = format!("{BETA:.17e}");
    let gamma = format!("{GAMMA:.17e}");
    let lambda = format!("{LAMBDA:.17e}");
    let rho_d = format!("{RHO_D:.17e}");
    let rate = format!("{delta}*exp(u0/(1+{beta}*u0))*u2/(1+{beta}*u0)");
    let q_source_scale = format!("({lambda}*{xi}*{xi})");
    let flux_source_scale = format!("({rho_d}*{gamma}*{xi}*{xi})");
    let equations = [
        format!("u1/({lambda})"),
        format!("{xi}*u1-{q_source_scale}*({rate})"),
        format!("u3/({rho_d})"),
        format!("{xi}*u3+{flux_source_scale}*({rate})"),
        format!("u5/({rho_d})"),
        format!("{xi}*u5-{flux_source_scale}*({rate})"),
    ]
    .into_iter()
    .map(|equation| Expr::parse_expression(&equation))
    .collect();
    let state_names = (0..6).map(|index| format!("u{index}")).collect();
    let mesh = (0..=INTERVALS)
        .map(|node| node as f64 / INTERVALS as f64)
        .collect();
    let boundary_conditions = [
        (0, BvpBoundarySide::Left, 0.0),
        (1, BvpBoundarySide::Right, 0.0),
        (2, BvpBoundarySide::Left, 1.0),
        (3, BvpBoundarySide::Right, 0.0),
        (4, BvpBoundarySide::Left, 0.0),
        (5, BvpBoundarySide::Right, 0.0),
    ]
    .into_iter()
    .map(|(state, side, value)| BvpBoundaryCondition { state, side, value })
    .collect();
    let problem = BvpProblem::new(equations, state_names, "x", mesh, boundary_conditions)
        .expect("valid standalone symbolic BVP");

    let full_seed = DMatrix::from_fn(6, fixture.len(), |state, node| {
        let [_, u0, u1, u2, u3, u4, u5] = fixture[node];
        [u0, u1, u2, u3, u4, u5][state]
    });
    let state_profile_seed = DMatrix::from_fn(6, INTERVALS, |state, interval| {
        let node = if matches!(state, 0 | 2 | 4) {
            interval + 1
        } else {
            interval
        };
        full_seed[(state, node)]
    });
    (problem, full_seed, state_profile_seed)
}

fn independent_trapezoid_residual(full_seed: &DMatrix<f64>) -> f64 {
    let h = 1.0 / INTERVALS as f64;
    let mut sum_squares = 0.0;
    for interval in 0..INTERVALS {
        let rhs = |node: usize| {
            let u0 = full_seed[(0, node)];
            let u1 = full_seed[(1, node)];
            let u2 = full_seed[(2, node)];
            let u3 = full_seed[(3, node)];
            let u5 = full_seed[(5, node)];
            let rate = DELTA * (u0 / (1.0 + BETA * u0)).exp() * u2 / (1.0 + BETA * u0);
            [
                u1 / LAMBDA,
                XI_MAX * u1 - LAMBDA * XI_MAX.powi(2) * rate,
                u3 / RHO_D,
                XI_MAX * u3 + RHO_D * GAMMA * XI_MAX.powi(2) * rate,
                u5 / RHO_D,
                XI_MAX * u5 - RHO_D * GAMMA * XI_MAX.powi(2) * rate,
            ]
        };
        let rhs0 = rhs(interval);
        let rhs1 = rhs(interval + 1);
        for state in 0..6 {
            let defect = full_seed[(state, interval + 1)]
                - full_seed[(state, interval)]
                - 0.5 * h * (rhs0[state] + rhs1[state]);
            sum_squares += defect * defect;
        }
    }
    sum_squares.sqrt()
}

fn initial_residual_norm(
    trace: &crate::numerical::BVP_Damp::telemetry::BvpDampedDecisionTrace,
) -> Option<f64> {
    use crate::numerical::BVP_Damp::telemetry::BvpDampedDecisionEvent;

    trace.events.iter().find_map(|event| match event {
        BvpDampedDecisionEvent::IterationDecision {
            base_residual_norm, ..
        } => *base_residual_norm,
        _ => None,
    })
}

#[test]
fn public_state_major_guess_preserves_the_damped_hot_branch() {
    let (problem, full_seed, state_profile_seed) = build_problem_and_seed();
    let seed_u0_max = (0..full_seed.ncols())
        .map(|node| full_seed[(0, node)])
        .fold(f64::NEG_INFINITY, f64::max);
    let independent_residual = independent_trapezoid_residual(&full_seed);
    assert!(seed_u0_max > 90.0, "fixture must select the hot branch");
    assert!(
        independent_residual < 1.0e-8,
        "seed residual={independent_residual:e}"
    );

    let bounds = HashMap::from([
        ("u0".to_owned(), (-19.9, 120.0)),
        ("u1".to_owned(), (-1.0e6, 1.0e6)),
        ("u2".to_owned(), (-1.0e12, 1.0e12)),
        ("u3".to_owned(), (-1.0e12, 1.0e12)),
        ("u4".to_owned(), (-1.0e12, 1.0e12)),
        ("u5".to_owned(), (-1.0e12, 1.0e12)),
    ]);
    let relative_tolerances = (0..6).map(|index| (format!("u{index}"), 1.0e-8)).collect();
    let base_options = DampedSolverOptions::sparse_damped();
    let generated = base_options
        .generated_backend_config
        .clone()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    let options = base_options
        .with_generated_backend_config(generated)
        .trapezoid_derivative()
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(1),
            max_damp_iter: Some(20),
            adaptive: None,
            ..SolverParams::default()
        }))
        .with_abs_tolerance(1.0e-8)
        .with_rel_tolerance(relative_tolerances)
        .with_max_iterations(150)
        .with_bounds(bounds)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(4096),
        );
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options,
            initial_guess: state_profile_seed,
        },
    )
    .expect("Damped should prepare the public state-major seed");

    solver
        .solve()
        .expect("Damped should solve from the hot-branch seed");
    let trace = solver
        .damped_decision_trace_snapshot()
        .expect("BVP_api exposes Damped's decision trace");
    let initial_residual =
        initial_residual_norm(&trace).expect("decision trace records the initial residual");
    let view = solver
        .solution_view()
        .expect("successful solution is published");
    let solution_u0_max = (0..view.mesh().len())
        .map(|node| view.value(node, 0).expect("u0 value"))
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(trace.dropped_events, 0);
    assert!(
        initial_residual < 1.0e-8,
        "initial residual={initial_residual:e}"
    );
    assert!(
        solution_u0_max > 90.0,
        "state-major guess should preserve the hot branch, got max u0={solution_u0_max:e}"
    );
}
