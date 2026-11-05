//! Large real-problem BVP reproducer ported from the KiThe integration test.
//! It intentionally uses only RST's symbolic `Expr` and public `BVP_api`;
//! state names are renamed to abstract `u0..u25`.

use std::collections::HashMap;

use crate::numerical::BVP_Cantera::CanteraExprBvpProblem;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedSolverOptions, SolverParams,
};
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use crate::numerical::BVP_Damp::telemetry::{BvpLoggingMode, BvpSolveStatus, BvpTelemetryMode};
use crate::numerical::BVP_api::{
    BvpApiError, BvpBoundaryCondition, BvpBoundarySide, BvpNativeSolver, BvpProblem,
    BvpSciSparseFactorization, BvpSolver, BvpSolverConfig, BvpSolverTelemetrySnapshot,
};
use crate::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciOptions, BvpSciTelemetryMode,
};
use crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

const STATE_COUNT: usize = 26;
const SCI_SEED_NODES: usize = 33;
const DAMPED_INTERVALS: usize = 33;

// RHS vector for u' = F(s, u), in state order u0..u25. The source equations
// below are ordinary scalar expressions; the renaming routine only maps their
// identifiers to abstract state coordinates before Expr parsing.
const RHS: [&str; STATE_COUNT] = [
    "q / 0.05",
    "q * 49.10419167354888 - (-103030398415.33754 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + -4128633572188.5405 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)) + 2280232215.9957895 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + 33299161969.77031 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + 712690917.1313457 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16) * 0.00000033333333333333335",
    "J0 / 0.000004844744729497463",
    "412.8184479613349 * J0 - 0.0001 * (28011.995829600797 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + 288944.2885029702 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)) + 423.17015412771735 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + 56060.951656390236 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + 624.6888893573005 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16)",
    "J1 / 0.000004844744729497463",
    "412.8184479613349 * J1 - 0.0001 * (-1239.9012171458817 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + -164260.26626664176 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + -1830.3571428571438 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16 + -82075.98618433015 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + -846615.4134633419 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)))",
    "J2 / 0.000004844744729497463",
    "412.8184479613349 * J2 - 0.0001 * (104440.78855307552 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + 1163.7868833320908 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16 + 52185.966291128585 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + 538299.2697684199 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)) + 788.3602272775099 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73)",
    "J3 / 0.000004844744729497463",
    "412.8184479613349 * J3 + 0.004188137016775251 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16",
    "J4 / 0.000004844744729497463",
    "412.8184479613349 * J4 + -0.008376274033550502 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16",
    "J5 / 0.000004844744729497463",
    "412.8184479613349 * J5 + -0.05070442518530647 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73",
    "J6 / 0.000004844744729497463",
    "412.8184479613349 * J6 + -6.337213343186177 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200))",
    "J7 / 0.000004844744729497463",
    "412.8184479613349 * J7 - 0.0001 * (-61494.109368260986 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + 634312.9488745617 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)))",
    "J8 / 0.000004844744729497463",
    "412.8184479613349 * J8",
    "J9 / 0.000004844744729497463",
    "412.8184479613349 * J9 + 5.965541506821332 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200))",
    "J10 / 0.000004844744729497463",
    "412.8184479613349 * J10 + 61.494109368260986 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200))",
    "J11 / 0.000004844744729497463",
    "412.8184479613349 * J11 - 0.0001 * (-478.67341611241034 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + 63413.94112538935 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)))",
];

const LEFT_VALUES: [f64; 12] = [
    0.0,
    0.018405139435376694,
    0.0,
    3.5094811081461935e-7,
    0.0,
    0.01254430598477255,
    0.0,
    0.002298292330465739,
    0.7656249768090929,
    5.570256770757038e-6,
    0.20052924374725334,
    0.0005921204881571657,
];

fn state_index(token: &str) -> Option<usize> {
    match token {
        "Teta" => Some(0),
        "q" => Some(1),
        _ => {
            let (prefix, suffix) = token.split_at(1);
            let index = suffix.parse::<usize>().ok()?;
            if index >= 12 {
                return None;
            }
            match prefix {
                "C" => Some(2 + 2 * index),
                "J" => Some(3 + 2 * index),
                _ => None,
            }
        }
    }
}

fn rename_state_tokens(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = String::with_capacity(source.len() + 8);
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
            {
                cursor += 1;
            }
            let token = &source[start..cursor];
            if let Some(state) = state_index(token) {
                output.push_str(&format!("u{state}"));
            } else {
                output.push_str(token);
            }
        } else {
            output.push(bytes[cursor] as char);
            cursor += 1;
        }
    }
    output
}

fn equations() -> Vec<Expr> {
    RHS.iter()
        .map(|rhs| Expr::parse_expression(&rename_state_tokens(rhs)))
        .collect()
}

fn boundary_conditions() -> Vec<BvpBoundaryCondition> {
    let mut conditions = Vec::with_capacity(STATE_COUNT);
    conditions.push(BvpBoundaryCondition {
        state: 0,
        side: BvpBoundarySide::Left,
        value: 0.0,
    });
    conditions.push(BvpBoundaryCondition {
        state: 1,
        side: BvpBoundarySide::Right,
        value: 0.0,
    });
    for (index, value) in LEFT_VALUES.into_iter().enumerate() {
        conditions.push(BvpBoundaryCondition {
            state: 2 + 2 * index,
            side: BvpBoundarySide::Left,
            value,
        });
        conditions.push(BvpBoundaryCondition {
            state: 3 + 2 * index,
            side: BvpBoundarySide::Right,
            value: 0.0,
        });
    }
    conditions
}

fn mesh(nodes: usize) -> Vec<f64> {
    (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect()
}

fn problem(nodes: usize) -> BvpProblem {
    let states = (0..STATE_COUNT).map(|index| format!("u{index}")).collect();
    BvpProblem::new(
        equations(),
        states,
        "s".to_string(),
        mesh(nodes),
        boundary_conditions(),
    )
    .expect("frozen numerical BVP fixture should validate")
}

fn inlet_guess(nodes: usize) -> DMatrix<f64> {
    DMatrix::from_fn(STATE_COUNT, nodes, |state, _| match state {
        0 | 1 => 0.0,
        odd if odd % 2 == 1 => 0.0,
        even => LEFT_VALUES[(even - 2) / 2],
    })
}

fn concentration_bounds() -> HashMap<String, (f64, f64)> {
    (0..STATE_COUNT)
        .map(|state| {
            let bounds = if state == 0 {
                (-1.0, 3.0)
            } else if state % 2 == 0 {
                (0.0, 1.1)
            } else {
                (-1e4, 1e4)
            };
            (format!("u{state}"), bounds)
        })
        .collect()
}

fn calculate_profile_metrics(solver: &BvpSolver) -> (usize, f64, f64, f64) {
    let view = solver.solution_view().expect("solver must retain a result");
    let node_count = view.mesh().len();
    let mut min_even_coordinate = f64::INFINITY;
    let mut max_abs_coordinate = 0.0_f64;
    let mut max_even_coordinate_sum_error = 0.0_f64;
    for node in 0..node_count {
        let mut even_coordinate_sum = 0.0;
        for state in 0..STATE_COUNT {
            max_abs_coordinate = max_abs_coordinate.max(view.value(node, state).unwrap().abs());
        }
        for species in 0..12 {
            let coordinate = view.value(node, 2 + 2 * species).unwrap();
            min_even_coordinate = min_even_coordinate.min(coordinate);
            even_coordinate_sum += coordinate;
        }
        max_even_coordinate_sum_error =
            max_even_coordinate_sum_error.max((even_coordinate_sum - 1.0).abs());
    }
    (
        node_count,
        min_even_coordinate,
        max_abs_coordinate,
        max_even_coordinate_sum_error,
    )
}

fn state_range(solver: &BvpSolver, state: usize) -> (f64, f64) {
    let view = solver.solution_view().expect("solver must retain a result");
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for node in 0..view.mesh().len() {
        let value = view
            .value(node, state)
            .expect("requested state is present in the solution");
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    (minimum, maximum)
}

fn interior_species_positivity(solver: &BvpSolver) -> (f64, usize) {
    let view = solver.solution_view().expect("solver must retain a result");
    let mut minimum = f64::INFINITY;
    let mut negative_count = 0;
    for node in 1..view.mesh().len().saturating_sub(1) {
        for state in (2..STATE_COUNT).step_by(2) {
            let value = view
                .value(node, state)
                .expect("species state is present in the solution");
            minimum = minimum.min(value);
            negative_count += usize::from(value < 0.0);
        }
    }
    (minimum, negative_count)
}

fn print_decision_events(
    label: &str,
    events: &[crate::numerical::BVP_Damp::telemetry::BvpLogEvent],
) {
    for event in events {
        println!(
            "{label} event seq={} kind={:?} iter={} trial={} mesh={} a={:.6e} b={:.6e}",
            event.sequence,
            event.kind,
            event.iteration,
            event.damping_trial,
            event.mesh_revision,
            event.value_a,
            event.value_b
        );
    }
}

#[test]
#[ignore = "standalone Expr/BVP_api route comparison; Sci run is intentionally expensive in Debug"]
fn rst_native_sci_route_accepts_the_frozen_equation_system() {
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_matrix_layout(BvpSciMatrixLayout::Dense)
        .with_tolerance(1e-5)
        .with_limits(256, 100, 20)
        .with_telemetry(BvpSciTelemetryMode::Off);
    let mut solver = BvpSolver::new(
        problem(SCI_SEED_NODES),
        BvpSolverConfig::Sci {
            options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: inlet_guess(SCI_SEED_NODES),
        },
    )
    .expect("RST Sci route should prepare");
    solver.solve().expect("RST Sci route should solve");
    let (nodes, min_even, max_abs, group_sum_error) = calculate_profile_metrics(&solver);
    assert_eq!(nodes, 78);
    assert!(min_even >= -1e-5);
    assert!(group_sum_error < 1e-3);
    assert!(max_abs.is_finite() && max_abs < 1e4);
    let (teta_min, teta_max) = state_range(&solver, 0);
    let (interior_species_min, interior_negative_count) = interior_species_positivity(&solver);
    let temperature_min_kelvin = 1200.0 + 300.0 * teta_min;
    let temperature_max_kelvin = 1200.0 + 300.0 * teta_max;
    assert!(temperature_min_kelvin.is_finite() && temperature_max_kelvin.is_finite());
    assert_eq!(
        interior_negative_count, 0,
        "Sci interior species must be nonnegative"
    );
    println!(
        "Sci: nodes={nodes} min_species={min_even:.6e} interior_min_species={interior_species_min:.6e} interior_negative_count={interior_negative_count} max_abs_state={max_abs:.6e} species_sum_error={group_sum_error:.6e} Teta=[{teta_min:.6e},{teta_max:.6e}] temperature_K=[{temperature_min_kelvin:.6e},{temperature_max_kelvin:.6e}]"
    );
}

#[test]
#[ignore = "standalone Expr/BVP_api route comparison"]
fn rst_native_frozen_dense_route_rejects_infeasible_bounded_iterate() {
    for mesh_nodes in [65, 129] {
        let mut unbounded_solver = BvpSolver::new(
            problem(mesh_nodes),
            BvpSolverConfig::Frozen {
                options: FrozenSolverOptions::dense_naive()
                    .with_tolerance(1e-5)
                    .with_max_iterations(500)
                    .with_bvp_telemetry_mode(BvpTelemetryMode::Counters),
                initial_guess: inlet_guess(mesh_nodes - 1),
            },
        )
        .expect("unbounded RST Frozen control should prepare");
        let unbounded_converged = unbounded_solver.solve().is_ok();
        assert!(
            unbounded_converged,
            "unbounded control must converge at {mesh_nodes} nodes"
        );
        let (nodes, min_even, max_abs, sum_error) = calculate_profile_metrics(&unbounded_solver);
        assert!(max_abs.is_finite());
        match mesh_nodes {
            65 => assert!(
                min_even < 0.0,
                "65-node control should reproduce negative interior states"
            ),
            129 => assert!(
                min_even >= 0.0,
                "129-node control should resolve nonnegative interior states"
            ),
            _ => unreachable!("test mesh sizes are enumerated above"),
        }
        println!(
            "Frozen unbounded nodes={nodes}: min_species={min_even:.6e}, max_abs={max_abs:.6e}, sum_error={sum_error:.6e}"
        );

        let mut bounded_solver = BvpSolver::new(
            problem(mesh_nodes),
            BvpSolverConfig::Frozen {
                options: FrozenSolverOptions::dense_naive()
                    .with_tolerance(1e-5)
                    .with_max_iterations(500)
                    .with_bounds(concentration_bounds())
                    .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
                    .with_bvp_logging_mode(BvpLoggingMode::Detailed),
                initial_guess: inlet_guess(mesh_nodes - 1),
            },
        )
        .expect("bounded RST Frozen route should prepare");
        let bounded_outcome = match bounded_solver.solve() {
            Ok(_) => "Converged",
            Err(BvpApiError::Frozen(
                crate::symbolic::bvp::legacy::BvpBackendIntegrationError::BoundStepBlocked {
                    ..
                },
            )) => "BoundStepBlocked",
            Err(BvpApiError::MissingResult) => "MissingResult",
            Err(error) => panic!("bounded Frozen solve failed unexpectedly: {error:?}"),
        };
        match (mesh_nodes, bounded_outcome) {
            (65, "BoundStepBlocked") | (129, "Converged") => {}
            _ => panic!(
                "unexpected bounded Frozen outcome at {mesh_nodes} mesh nodes: {bounded_outcome}"
            ),
        }
        if bounded_outcome == "Converged" {
            let (nodes, min_even, max_abs, sum_error) = calculate_profile_metrics(&bounded_solver);
            assert!(
                min_even >= 0.0,
                "successful bounded profile escaped its lower bounds"
            );
            assert!(max_abs.is_finite() && max_abs < 1e4);
            assert!(sum_error.is_finite());
            println!(
                "Frozen bounded nodes={nodes}: min_species={min_even:.6e}, max_abs={max_abs:.6e}, sum_error={sum_error:.6e}"
            );
        } else {
            assert!(bounded_solver.solution_view().is_none());
        }
        let BvpSolverTelemetrySnapshot::Frozen(snapshot) = bounded_solver.telemetry_snapshot()
        else {
            panic!("expected Frozen telemetry");
        };
        println!(
            "Frozen bounded mesh_nodes={mesh_nodes}: outcome={bounded_outcome}, status={:?}, counters={:#?}",
            snapshot.last_solve_status, snapshot.counters
        );
        if bounded_outcome == "BoundStepBlocked" {
            assert_eq!(snapshot.last_solve_status, BvpSolveStatus::Failed);
        }
    }
}

fn damped_options(
    grid_method: Option<GridRefinementMethod>,
    use_physical_bounds: bool,
) -> DampedSolverOptions {
    let backend = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    let strategy = SolverParams {
        max_jac: Some(1),
        max_damp_iter: Some(12),
        adaptive: grid_method.map(|grid_method| AdaptiveGridConfig {
            version: 1,
            max_refinements: 2,
            grid_method,
        }),
        ..SolverParams::default()
    };
    let mut bounds = HashMap::with_capacity(STATE_COUNT);
    let mut tolerances = HashMap::with_capacity(STATE_COUNT);
    for state in 0..STATE_COUNT {
        let name = format!("u{state}");
        let bound = if state == 0 {
            (-1.0, 3.0)
        } else if state == 1 || state % 2 == 1 {
            (-1e4, 1e4)
        } else {
            (-0.1, 1.1)
        };
        bounds.insert(name.clone(), bound);
        tolerances.insert(name, 1e-5);
    }
    if !use_physical_bounds {
        bounds = (0..STATE_COUNT)
            .map(|state| (format!("u{state}"), (-1e4, 1e4)))
            .collect();
    }
    DampedSolverOptions::sparse_damped()
        .with_generated_backend_config(backend)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_mode(BvpLoggingMode::Detailed)
        .with_scheme_name("trapezoid")
        .with_strategy_params(Some(strategy))
        .with_abs_tolerance(1e-5)
        .with_rel_tolerance(tolerances)
        .with_max_iterations(150)
        .with_bounds(bounds)
}

pub(crate) fn cantera_fixed_grid_fixture() -> BvpSolver {
    BvpSolver::new(
        problem(DAMPED_INTERVALS + 1),
        BvpSolverConfig::Damped {
            options: damped_options(None, true),
            initial_guess: inlet_guess(DAMPED_INTERVALS),
        },
    )
    .expect("large fixed-grid A/B fixture should construct")
}

pub(crate) fn cantera_fixed_grid_expr_problem() -> CanteraExprBvpProblem {
    let mut conditions = HashMap::<String, Vec<(usize, f64)>>::with_capacity(STATE_COUNT);
    for condition in boundary_conditions() {
        let endpoint = match condition.side {
            BvpBoundarySide::Left => 0,
            BvpBoundarySide::Right => 1,
        };
        conditions
            .entry(format!("u{}", condition.state))
            .or_default()
            .push((endpoint, condition.value));
    }
    CanteraExprBvpProblem {
        equations: equations(),
        initial_guess: inlet_guess(DAMPED_INTERVALS),
        state_names: (0..STATE_COUNT).map(|index| format!("u{index}")).collect(),
        independent_variable: "s".into(),
        boundary_conditions: conditions,
        interval: (0.0, 1.0),
        steps: DAMPED_INTERVALS,
    }
}

#[test]
#[ignore = "standalone Expr/BVP_api route comparison"]
fn rst_native_damped_fixed_grid_converges_with_correct_trapezoid_jacobian() {
    for assembly in [
        BvpSymbolicAssemblyBackend::ExprLegacy,
        BvpSymbolicAssemblyBackend::AtomView,
    ] {
        let generated_backend = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(assembly);
        let mut options =
            damped_options(None, true).with_generated_backend_config(generated_backend);
        options = options.with_scheme_name("trapezoid");
        let mut solver = BvpSolver::new(
            problem(DAMPED_INTERVALS + 1),
            BvpSolverConfig::Damped {
                options,
                initial_guess: inlet_guess(DAMPED_INTERVALS),
            },
        )
        .unwrap_or_else(|error| panic!("{assembly:?} Damped route should prepare: {error}"));
        solver
            .solve()
            .unwrap_or_else(|error| panic!("{assembly:?} fixed-grid Damped solve: {error}"));
        let (nodes, minimum_species, maximum_state, sum_error) = calculate_profile_metrics(&solver);
        assert_eq!(nodes, DAMPED_INTERVALS + 1);
        assert!(
            minimum_species >= -0.1,
            "{assembly:?} profile violates configured lower bounds"
        );
        assert!(maximum_state.is_finite());
        assert!(sum_error.is_finite());
        let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
            panic!("expected Damped telemetry");
        };
        println!(
            "Damped fixed {assembly:?}: status={:?} iterations={} damping_trials={} linear_solves={}",
            snapshot.last_solve_status,
            snapshot.counters.iterations,
            snapshot.counters.damping_trials,
            snapshot.counters.linear_solves
        );
        println!(
            "Damped fixed {assembly:?} detailed timings: {:#?}",
            snapshot.timings
        );
        print_decision_events(&format!("Damped fixed {assembly:?}"), &snapshot.log_events);
        assert_eq!(snapshot.last_solve_status, BvpSolveStatus::Converged);
        assert!(snapshot.counters.iterations > 0);
        assert!(snapshot.counters.linear_solves > 0);
        println!(
            "Damped fixed {assembly:?} profile: nodes={nodes}, minimum_species={minimum_species:.6e}, max_abs_state={maximum_state:.6e}, sum_error={sum_error:.6e}"
        );
    }
}

#[test]
#[ignore = "diagnostic: compare KiThe Trapezoid Jacobians with residual finite differences"]
fn rst_native_damped_fixed_grid_trapezoid_jacobian_matches_both_frontends() {
    use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;

    for assembly in [
        BvpSymbolicAssemblyBackend::ExprLegacy,
        BvpSymbolicAssemblyBackend::AtomView,
    ] {
        let generated_backend = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
            .with_symbolic_assembly_backend(assembly);
        let mut options =
            damped_options(None, true).with_generated_backend_config(generated_backend);
        options = options.with_scheme_name("trapezoid");
        let mut facade = BvpSolver::new(
            problem(DAMPED_INTERVALS + 1),
            BvpSolverConfig::Damped {
                options,
                initial_guess: inlet_guess(DAMPED_INTERVALS),
            },
        )
        .unwrap_or_else(|error| panic!("{assembly:?} fixture should construct: {error}"));
        let BvpNativeSolver::Damped(solver) = facade.native_mut() else {
            panic!("expected Damped solver");
        };
        solver
            .try_set_initial_guess(solver.initial_guess.clone())
            .unwrap();
        solver.try_eq_generate(None, None).unwrap();

        // Node zero has 13 fixed even components; u1 at node one is column 14.
        let column = 14;
        let state = solver.y.to_DVectorType();
        let jacobian = solver
            .jac
            .as_mut()
            .unwrap()
            .try_call(solver.p, &*solver.y)
            .unwrap()
            .to_DMatrixType();
        let eps = 1e-7;
        let mut plus = state.clone();
        let mut minus = state;
        plus[column] += eps;
        minus[column] -= eps;
        let plus = Vectors_type_casting(&plus, "Sparse".to_string());
        let minus = Vectors_type_casting(&minus, "Sparse".to_string());
        let plus_residual = solver
            .fun
            .try_call(solver.p, &*plus)
            .unwrap()
            .to_DVectorType();
        let minus_residual = solver
            .fun
            .try_call(solver.p, &*minus)
            .unwrap()
            .to_DVectorType();
        let fd = (plus_residual[0] - minus_residual[0]) / (2.0 * eps);
        let expected = -0.5 / (DAMPED_INTERVALS as f64 * 0.05);
        println!(
            "large fixture {assembly:?} row=0 col={column}: analytic={expected:.12e} fd={fd:.12e} generated={:.12e}",
            jacobian[(0, column)]
        );
        assert!((fd - expected).abs() < 1e-8);
        assert!(
            (jacobian[(0, column)] - fd).abs() < 1e-8,
            "{assembly:?} generated Sparse Jacobian disagrees with residual FD: generated={:.12e}, fd={fd:.12e}",
            jacobian[(0, column)]
        );
    }
}

#[test]
#[ignore = "standalone Expr/BVP_api DoublePoints panic reproducer"]
fn rst_native_damped_double_points_returns_typed_outcome_without_panic() {
    let mut solver = BvpSolver::new(
        problem(DAMPED_INTERVALS + 1),
        BvpSolverConfig::Damped {
            options: damped_options(Some(GridRefinementMethod::DoublePoints), true),
            initial_guess: inlet_guess(DAMPED_INTERVALS),
        },
    )
    .expect("RST Damped route should prepare");
    let outcome = match solver.solve() {
        Ok(_) => "Converged",
        Err(BvpApiError::MissingResult) => "MissingResult",
        Err(error) => panic!("DoublePoints returned unexpected typed error: {error:?}"),
    };
    let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
        panic!("expected Damped telemetry");
    };
    println!(
        "Damped DoublePoints outcome={outcome} status={:?} counters={:#?}",
        snapshot.last_solve_status, snapshot.counters
    );
    print_decision_events("Damped DoublePoints", &snapshot.log_events);
    assert!(snapshot.counters.grid_refinements > 0);
    match outcome {
        "Converged" => {
            assert_eq!(snapshot.last_solve_status, BvpSolveStatus::Converged);
            let (nodes, min_species, max_abs, sum_error) = calculate_profile_metrics(&solver);
            assert!(max_abs.is_finite() && max_abs < 1e4);
            assert!(sum_error.is_finite());
            println!(
                "Damped DoublePoints converged: nodes={nodes}, min_species={min_species:.6e}, max_abs={max_abs:.6e}, sum_error={sum_error:.6e}"
            );
        }
        "MissingResult" => {
            assert_eq!(snapshot.last_solve_status, BvpSolveStatus::NoResult);
            assert!(solver.solution_view().is_none());
            println!("Damped DoublePoints: typed no-result returned without panic");
        }
        _ => unreachable!("diagnostic outcome is exhaustively mapped above"),
    }
}

#[test]
#[ignore = "large Debug matrix for Damped adaptive-grid solver strategies"]
fn rst_native_damped_large_problem_grid_strategy_matrix() {
    let strategies = [
        (
            "TwoPoint(0.02,0.02,2.5)",
            GridRefinementMethod::TwoPoint(0.02, 0.02, 2.5),
        ),
        (
            "Pearson(1e-3,1.4)",
            GridRefinementMethod::Pearson(1e-3, 1.4),
        ),
        (
            "GrcarSmooke(1e-3,1e-2,1.4)",
            GridRefinementMethod::GrcarSmooke(1e-3, 1e-2, 1.4),
        ),
    ];

    for (name, strategy) in strategies {
        let mut solver = BvpSolver::new(
            problem(DAMPED_INTERVALS + 1),
            BvpSolverConfig::Damped {
                options: damped_options(Some(strategy), true),
                initial_guess: inlet_guess(DAMPED_INTERVALS),
            },
        )
        .unwrap_or_else(|error| panic!("Damped {name} preparation failed: {error:?}"));
        let outcome = match solver.solve() {
            Ok(_) => "Converged",
            Err(BvpApiError::MissingResult) => "MissingResult",
            Err(error) => panic!("Damped {name} failed with unexpected error: {error:?}"),
        };
        let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
            panic!("expected Damped telemetry for {name}");
        };
        let final_mesh_nodes = match solver.native() {
            BvpNativeSolver::Damped(native) => native.x_mesh.len(),
            _ => unreachable!("solver route was configured as Damped"),
        };
        assert!(snapshot.counters.grid_refinements <= 2);
        match outcome {
            "Converged" => {
                let (nodes, min_species, max_abs, sum_error) = calculate_profile_metrics(&solver);
                assert!(max_abs.is_finite() && max_abs < 1e4);
                assert!(sum_error.is_finite());
                println!(
                    "Damped {name}: converged nodes={nodes}, native_mesh_nodes={final_mesh_nodes}, refinements={}, iterations={}, min_species={min_species:.6e}, max_abs={max_abs:.6e}, sum_error={sum_error:.6e}",
                    snapshot.counters.grid_refinements, snapshot.counters.iterations
                );
            }
            "MissingResult" => {
                assert!(solver.solution_view().is_none());
                println!(
                    "Damped {name}: MissingResult, native_mesh_nodes={final_mesh_nodes}, refinements={}, iterations={}, damping_trials={}, linear_solves={}",
                    snapshot.counters.grid_refinements,
                    snapshot.counters.iterations,
                    snapshot.counters.damping_trials,
                    snapshot.counters.linear_solves
                );
            }
            _ => unreachable!("diagnostic outcome is exhaustively mapped above"),
        }
        println!(
            "Damped {name} terminal telemetry: status={:?}, timings={:#?}",
            snapshot.last_solve_status, snapshot.timings
        );
        for event in snapshot.log_events.iter().filter(|event| {
            matches!(
                event.kind,
                crate::numerical::BVP_Damp::BvpLogEventKind::BoundLimitedStep
                    | crate::numerical::BVP_Damp::BvpLogEventKind::MeshRevision
                    | crate::numerical::BVP_Damp::BvpLogEventKind::Termination
            )
        }) {
            println!(
                "Damped {name} event seq={} kind={:?} iter={} mesh={} a={:.6e} b={:.6e}",
                event.sequence,
                event.kind,
                event.iteration,
                event.mesh_revision,
                event.value_a,
                event.value_b
            );
        }
        assert!(matches!(
            snapshot.last_solve_status,
            BvpSolveStatus::Converged | BvpSolveStatus::NoResult
        ));
    }
}

#[test]
#[ignore = "large Debug control to separate active Damped bounds from grid refinement"]
fn rst_native_damped_wide_bounds_grcar_refinement_control() {
    let mut solver = BvpSolver::new(
        problem(DAMPED_INTERVALS + 1),
        BvpSolverConfig::Damped {
            options: damped_options(
                Some(GridRefinementMethod::GrcarSmooke(1e-3, 1e-2, 1.4)),
                false,
            ),
            initial_guess: inlet_guess(DAMPED_INTERVALS),
        },
    )
    .expect("wide-bounds Damped Grcar-Smooke route should prepare");
    let outcome = match solver.solve() {
        Ok(_) => "Converged",
        Err(BvpApiError::MissingResult) => "MissingResult",
        Err(error) => {
            println!("wide-bounds Damped typed failure: {error:?}");
            "TypedFailure"
        }
    };
    let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
        panic!("expected Damped telemetry");
    };
    let final_mesh_nodes = match solver.native() {
        BvpNativeSolver::Damped(native) => native.x_mesh.len(),
        _ => unreachable!("solver route was configured as Damped"),
    };
    println!(
        "Damped wide-bounds Grcar-Smooke: outcome={outcome}, mesh_nodes={final_mesh_nodes}, refinements={}, iterations={}, damping_trials={}, linear_solves={}, status={:?}, timings={:#?}",
        snapshot.counters.grid_refinements,
        snapshot.counters.iterations,
        snapshot.counters.damping_trials,
        snapshot.counters.linear_solves,
        snapshot.last_solve_status,
        snapshot.timings
    );
    if outcome == "Converged" {
        let (nodes, min_species, max_abs, sum_error) = calculate_profile_metrics(&solver);
        assert!(max_abs.is_finite() && max_abs < 1e4);
        assert!(sum_error.is_finite());
        println!(
            "Damped wide-bounds Grcar-Smooke result: nodes={nodes}, min_species={min_species:.6e}, max_abs={max_abs:.6e}, sum_error={sum_error:.6e}"
        );
    } else {
        assert!(solver.solution_view().is_none());
        assert!(matches!(
            snapshot.last_solve_status,
            BvpSolveStatus::NoResult | BvpSolveStatus::Failed
        ));
    }
}
