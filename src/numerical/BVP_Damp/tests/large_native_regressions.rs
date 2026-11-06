//! Large real-problem BVP reproducer ported from the KiThe integration test.
//! It intentionally uses only RST's symbolic `Expr` and public `BVP_api`;
//! state names are renamed to abstract `u0..u25`.

use std::collections::HashMap;

use crate::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonMatrixLayout, CorrectionNewtonOptions, CorrectionNewtonProblem,
    CorrectionNewtonSolver, CorrectionNewtonSymbolicFrontend, CorrectionNewtonTelemetryMode,
};
use crate::numerical::BVP_Damp::BVP_traits::VectorType;
use crate::numerical::BVP_Damp::BVP_traits::{Fun, Jac, MatrixType, Vectors_type_casting};
use crate::numerical::BVP_Damp::BVP_utils::construct_full_solution;
use crate::numerical::BVP_Damp::BVP_utils_damped::{
    convergence_condition, weighted_norm_reduced_unknowns,
};
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
use nalgebra::{DMatrix, DVector};

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

pub(crate) fn correction_newton_fixed_grid_expr_problem() -> CorrectionNewtonProblem {
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
    CorrectionNewtonProblem {
        equations: equations(),
        initial_guess: inlet_guess(DAMPED_INTERVALS),
        state_names: (0..STATE_COUNT).map(|index| format!("u{index}")).collect(),
        independent_variable: "s".into(),
        boundary_conditions: conditions,
        interval: (0.0, 1.0),
        steps: DAMPED_INTERVALS,
    }
}

fn full_grid_trapezoid_defect(problem: &CorrectionNewtonProblem, profile: &DMatrix<f64>) -> f64 {
    let argument_names = std::iter::once(problem.independent_variable.as_str())
        .chain(problem.state_names.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let rhs = problem
        .equations
        .iter()
        .map(|equation| equation.lambdify_borrowed_thread_safe(&argument_names))
        .collect::<Vec<_>>();
    let mut maximum = 0.0_f64;
    for interval in 0..problem.steps {
        let x_left = problem.interval.0
            + (problem.interval.1 - problem.interval.0) * interval as f64 / problem.steps as f64;
        let x_right = problem.interval.0
            + (problem.interval.1 - problem.interval.0) * (interval + 1) as f64
                / problem.steps as f64;
        let h = x_right - x_left;
        let left_args = std::iter::once(x_left)
            .chain(profile.row(interval).iter().copied())
            .collect::<Vec<_>>();
        let right_args = std::iter::once(x_right)
            .chain(profile.row(interval + 1).iter().copied())
            .collect::<Vec<_>>();
        for (state, evaluate_rhs) in rhs.iter().enumerate() {
            let defect = profile[(interval + 1, state)]
                - profile[(interval, state)]
                - 0.5 * h * (evaluate_rhs(&left_args) + evaluate_rhs(&right_args));
            maximum = maximum.max(defect.abs());
        }
    }
    maximum
}

fn adaptive_mesh_trapezoid_defect(
    problem: &CorrectionNewtonProblem,
    mesh: &[f64],
    profile: &DMatrix<f64>,
) -> f64 {
    assert_eq!(mesh.len(), profile.nrows());
    let argument_names = std::iter::once(problem.independent_variable.as_str())
        .chain(problem.state_names.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let rhs = problem
        .equations
        .iter()
        .map(|equation| equation.lambdify_borrowed_thread_safe(&argument_names))
        .collect::<Vec<_>>();
    let mut maximum = 0.0_f64;
    for interval in 0..mesh.len() - 1 {
        let h = mesh[interval + 1] - mesh[interval];
        let left_args = std::iter::once(mesh[interval])
            .chain(profile.row(interval).iter().copied())
            .collect::<Vec<_>>();
        let right_args = std::iter::once(mesh[interval + 1])
            .chain(profile.row(interval + 1).iter().copied())
            .collect::<Vec<_>>();
        for (state, evaluate_rhs) in rhs.iter().enumerate() {
            let defect = profile[(interval + 1, state)]
                - profile[(interval, state)]
                - 0.5 * h * (evaluate_rhs(&left_args) + evaluate_rhs(&right_args));
            maximum = maximum.max(defect.abs());
        }
    }
    maximum
}

#[derive(Debug)]
struct ReferenceProfileComparison {
    max_abs: f64,
    x_at_max: f64,
    state_at_max: usize,
    actual_at_max: f64,
    reference_at_max: f64,
    per_state_max_abs: Vec<f64>,
}

fn interpolated_reference_error(
    reference: &[(f64, Vec<f64>)],
    mesh: &[f64],
    profile: &DMatrix<f64>,
) -> ReferenceProfileComparison {
    let mut worst = ReferenceProfileComparison {
        max_abs: 0.0,
        x_at_max: 0.0,
        state_at_max: 0,
        actual_at_max: 0.0,
        reference_at_max: 0.0,
        per_state_max_abs: vec![0.0; profile.ncols()],
    };
    for (node, &x) in mesh.iter().enumerate() {
        let right = reference.partition_point(|(reference_x, _)| *reference_x < x);
        let right = right.min(reference.len() - 1);
        let left = right.saturating_sub(1);
        let (left_x, left_values) = &reference[left];
        let (right_x, right_values) = &reference[right];
        let fraction = if right == left || right_x == left_x {
            0.0
        } else {
            (x - left_x) / (right_x - left_x)
        };
        for state in 0..profile.ncols() {
            let expected =
                left_values[state] + fraction * (right_values[state] - left_values[state]);
            let actual = profile[(node, state)];
            let error = (actual - expected).abs();
            worst.per_state_max_abs[state] = worst.per_state_max_abs[state].max(error);
            if error > worst.max_abs {
                worst.max_abs = error;
                worst.x_at_max = x;
                worst.state_at_max = state;
                worst.actual_at_max = actual;
                worst.reference_at_max = expected;
            }
        }
    }
    worst
}

fn damped_terminal_weighted_rms(
    solver: &BvpSolver,
    profile: &DMatrix<f64>,
    relative_tolerance: f64,
) -> f64 {
    let BvpNativeSolver::Damped(native) = solver.native() else {
        panic!("expected native Damped solver");
    };
    let correction = native.step(native.p, &*native.y).0;
    let full_len = profile.len();
    let fixed_positions = native
        .BC_position_and_value
        .iter()
        .map(|&(position, _, _)| position)
        .collect::<std::collections::HashSet<_>>();
    let full_positions = (0..full_len)
        .filter(|position| !fixed_positions.contains(position))
        .collect::<Vec<_>>();
    let relative_tolerances = vec![relative_tolerance; native.y.len()];
    let mut component_sums = vec![0.0; profile.ncols()];
    weighted_norm_reduced_unknowns(
        &*native.y,
        &*correction,
        &relative_tolerances,
        native.abs_tolerance,
        profile.ncols(),
        profile.nrows(),
        &full_positions,
        &native.BC_position_and_value,
        &mut component_sums,
    )
    .expect("valid terminal state should produce a finite weighted RMS")
}

fn correction_newton_options_for_kithe_fixture(
    problem: &CorrectionNewtonProblem,
) -> CorrectionNewtonOptions {
    CorrectionNewtonOptions {
        symbolic_frontend: CorrectionNewtonSymbolicFrontend::ExprLegacy,
        matrix_layout: CorrectionNewtonMatrixLayout::Sparse,
        absolute_tolerance: 1e-5,
        relative_tolerances: problem
            .state_names
            .iter()
            .map(|name| (name.clone(), 1e-5))
            .collect(),
        bounds: problem
            .state_names
            .iter()
            .enumerate()
            .map(|(state, name)| {
                let bounds = if state == 0 {
                    (-1.0, 3.0)
                } else if state == 1 || state % 2 == 1 {
                    (-1e4, 1e4)
                } else {
                    (-0.1, 1.1)
                };
                (name.clone(), bounds)
            })
            .collect(),
        max_iterations: 150,
        max_jacobian_age: 1,
        max_damping_trials: 12,
        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
        ..CorrectionNewtonOptions::default()
    }
}

fn damped_options_for_correction_newton_fixture(
    problem: &CorrectionNewtonProblem,
    logging: BvpLoggingMode,
    bounds: &HashMap<String, (f64, f64)>,
    assembly: BvpSymbolicAssemblyBackend,
) -> DampedSolverOptions {
    let generated_backend = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::sparse_defaults()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(assembly);
    let relative_tolerances = problem
        .state_names
        .iter()
        .map(|name| (name.clone(), 1.0e-8))
        .collect();
    let strategy = SolverParams {
        max_jac: Some(1),
        max_damp_iter: Some(12),
        ..SolverParams::default()
    };
    DampedSolverOptions::sparse_damped()
        .with_generated_backend_config(generated_backend)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Detailed)
        .with_bvp_logging_mode(logging)
        .with_scheme_name("trapezoid")
        .with_strategy_params(Some(strategy))
        .with_abs_tolerance(1.0e-8)
        .with_rel_tolerance(relative_tolerances)
        .with_max_iterations(100)
        .with_bounds(bounds.clone())
}

fn damped_packed_seed(problem: &CorrectionNewtonProblem) -> DMatrix<f64> {
    let state_count = problem.state_names.len();
    let node_count = problem.steps + 1;
    let mut full = vec![f64::NAN; node_count * state_count];
    let mut fixed = std::collections::HashSet::with_capacity(state_count);
    for (state_name, conditions) in &problem.boundary_conditions {
        let state = problem
            .state_names
            .iter()
            .position(|name| name == state_name)
            .expect("boundary state belongs to the fixture");
        assert_eq!(conditions.len(), 1, "fixture has one endpoint per state");
        let (endpoint, value) = conditions[0];
        let node = if endpoint == 0 { 0 } else { problem.steps };
        let position = node * state_count + state;
        full[position] = value;
        fixed.insert(position);
        for column in 0..problem.steps {
            let node = if endpoint == 0 { column + 1 } else { column };
            full[node * state_count + state] = problem.initial_guess[(state, column)];
        }
    }
    assert!(full.iter().all(|value| value.is_finite()));
    let packed = full
        .into_iter()
        .enumerate()
        .filter_map(|(position, value)| (!fixed.contains(&position)).then_some(value))
        .collect::<Vec<_>>();
    assert_eq!(packed.len(), state_count * problem.steps);
    DMatrix::from_column_slice(state_count, problem.steps, &packed)
}

fn full_profile_from_problem(problem: &CorrectionNewtonProblem) -> DMatrix<f64> {
    let state_count = problem.state_names.len();
    let node_count = problem.steps + 1;
    let mut full = vec![f64::NAN; node_count * state_count];
    for (state, name) in problem.state_names.iter().enumerate() {
        let conditions = problem
            .boundary_conditions
            .get(name)
            .expect("fixture state has a boundary condition");
        assert_eq!(conditions.len(), 1);
        let (endpoint, value) = conditions[0];
        let boundary_node = if endpoint == 0 { 0 } else { problem.steps };
        full[boundary_node * state_count + state] = value;
        for column in 0..problem.steps {
            let node = if endpoint == 0 { column + 1 } else { column };
            full[node * state_count + state] = problem.initial_guess[(state, column)];
        }
    }
    assert!(full.iter().all(|value| value.is_finite()));
    DMatrix::from_fn(node_count, state_count, |node, state| {
        full[node * state_count + state]
    })
}

fn eval_damped_residual(solver: &mut BvpSolver, values: &[f64]) -> DVector<f64> {
    let BvpNativeSolver::Damped(native) = solver.native_mut() else {
        panic!("expected native Damped solver")
    };
    let dense = DVector::from_column_slice(values);
    let vector_type = match native.y.vec_type().as_str() {
        "Sparse_3" => "Sparse".to_string(),
        other => other.to_string(),
    };
    let typed = Vectors_type_casting(&dense, vector_type);
    native
        .fun
        .try_call(native.p, &*typed)
        .expect("Damped residual callback evaluates")
        .to_DVectorType()
}

fn eval_damped_residual_and_jacobian(
    solver: &mut BvpSolver,
    values: &[f64],
) -> (DVector<f64>, DMatrix<f64>) {
    let BvpNativeSolver::Damped(native) = solver.native_mut() else {
        panic!("expected native Damped solver")
    };
    let dense = DVector::from_column_slice(values);
    let vector_type = match native.y.vec_type().as_str() {
        "Sparse_3" => "Sparse".to_string(),
        other => other.to_string(),
    };
    let typed = Vectors_type_casting(&dense, vector_type);
    let residual = native
        .fun
        .try_call(native.p, &*typed)
        .expect("Damped residual callback evaluates")
        .to_DVectorType();
    let jacobian = native
        .jac
        .as_mut()
        .expect("Damped symbolic Jacobian callback exists")
        .try_call(native.p, &*typed)
        .expect("Damped Jacobian callback evaluates")
        .to_DMatrixType();
    (residual, jacobian)
}

fn bounded_common_seed(
    problem: &CorrectionNewtonProblem,
    bounds: &HashMap<String, (f64, f64)>,
) -> (CorrectionNewtonProblem, usize, f64) {
    // Both solver APIs consume the same state-by-interval matrix. Keep this
    // representation intact; flattening/repacking it changes state ownership.
    let mut projected_count = 0;
    let mut maximum_adjustment = 0.0_f64;
    let mut matched = problem.clone();
    for (state, name) in problem.state_names.iter().enumerate() {
        let (lower, upper) = bounds.get(name).expect("matched bounds cover every state");
        for column in 0..problem.steps {
            let value = &mut matched.initial_guess[(state, column)];
            let projected = value.clamp(*lower, *upper);
            if projected != *value {
                projected_count += 1;
                maximum_adjustment = maximum_adjustment.max((projected - *value).abs());
                *value = projected;
            }
        }
    }
    (matched, projected_count, maximum_adjustment)
}

fn profile_from_solver(solver: &BvpSolver, state_count: usize) -> DMatrix<f64> {
    let view = solver
        .solution_view()
        .expect("converged solver publishes its full-grid profile");
    DMatrix::from_fn(view.mesh().len(), state_count, |node, state| {
        view.value(node, state)
            .expect("all fixture states are present")
    })
}

fn bvp_problem_from_correction_newton(problem: &CorrectionNewtonProblem) -> BvpProblem {
    let boundary_conditions = problem
        .boundary_conditions
        .iter()
        .flat_map(|(state_name, conditions)| {
            let state = problem
                .state_names
                .iter()
                .position(|candidate| candidate == state_name)
                .expect("fixture boundary state is declared");
            conditions
                .iter()
                .map(move |&(endpoint, value)| BvpBoundaryCondition {
                    state,
                    side: match endpoint {
                        0 => BvpBoundarySide::Left,
                        1 => BvpBoundarySide::Right,
                        _ => panic!("fixture endpoint is left (0) or right (1)"),
                    },
                    value,
                })
        })
        .collect();
    let mesh = (0..=problem.steps)
        .map(|node| {
            problem.interval.0
                + (problem.interval.1 - problem.interval.0) * node as f64 / problem.steps as f64
        })
        .collect();
    BvpProblem::new(
        problem.equations.clone(),
        problem.state_names.clone(),
        problem.independent_variable.clone(),
        mesh,
        boundary_conditions,
    )
    .expect("shared fixed-mesh combustion fixture is valid for BVP_api")
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
#[ignore = "fixed-mesh production Damped versus CorrectionNewton A/B on the KiThe combustion fixture"]
fn rst_native_damped_vs_correction_newton_fixed_mesh_kithe_combustion_ab() {
    let problem = correction_newton_fixed_grid_expr_problem();
    let correction_options = correction_newton_options_for_kithe_fixture(&problem);
    let matched_bounds = correction_options.bounds.clone();

    let damped_options = damped_options(None, true)
        .with_bounds(matched_bounds.clone())
        .with_bvp_telemetry_mode(BvpTelemetryMode::Counters)
        .with_bvp_logging_mode(BvpLoggingMode::Off);
    // The matched comparison is Sparse/ExprLegacy on both production routes.
    let mut damped = BvpSolver::new(
        bvp_problem_from_correction_newton(&problem),
        BvpSolverConfig::Damped {
            options: damped_options.clone(),
            initial_guess: damped_packed_seed(&problem),
        },
    )
    .expect("matched Damped production route should prepare");
    damped
        .solve()
        .expect("matched Damped production route should converge");

    // Candidate A/B arm: same prepared production Damped route and fixture,
    // but using CorrectionNewton's norm and geometric backtracking in the
    // fixed-mesh Newton loop. This does not change the production default.
    let mut candidate_damped = BvpSolver::new(
        bvp_problem_from_correction_newton(&problem),
        BvpSolverConfig::Damped {
            options: damped_options
                .clone()
                .with_bvp_telemetry_mode(BvpTelemetryMode::Counters)
                .with_bvp_logging_mode(BvpLoggingMode::Off),
            initial_guess: damped_packed_seed(&problem),
        },
    )
    .expect("candidate Damped route should prepare");
    match candidate_damped.native_mut() {
        BvpNativeSolver::Damped(native) => {
            native.enable_correction_newton_criteria_for_test();
        }
        _ => panic!("expected candidate Damped route"),
    }
    candidate_damped
        .solve()
        .expect("candidate Damped criteria should converge on the matched fixture");

    let mut correction_newton =
        CorrectionNewtonSolver::from_symbolic(problem.clone(), correction_options)
            .expect("matched CorrectionNewton production route should prepare");
    let correction_report = correction_newton
        .solve()
        .expect("matched CorrectionNewton production route should converge");

    let damped_view = damped
        .solution_view()
        .expect("converged Damped route publishes a profile");
    let damped_profile = DMatrix::from_fn(damped_view.mesh().len(), STATE_COUNT, |node, state| {
        damped_view
            .value(node, state)
            .expect("all fixture states are present")
    });
    let candidate_view = candidate_damped
        .solution_view()
        .expect("converged candidate Damped route publishes a profile");
    let candidate_profile =
        DMatrix::from_fn(candidate_view.mesh().len(), STATE_COUNT, |node, state| {
            candidate_view
                .value(node, state)
                .expect("all candidate fixture states are present")
        });
    let correction_profile = correction_newton
        .full_grid_values(&correction_report.state)
        .expect("CorrectionNewton profile expands onto the fixed mesh");
    assert_eq!(damped_profile.shape(), correction_profile.shape());

    let correction_callback_values =
        crate::numerical::BVP_CorrectionNewton::evaluate_symbolic_callbacks_for_test(
            problem.clone(),
            correction_newton_options_for_kithe_fixture(&problem),
            &[damped_profile.clone(), correction_profile.clone()],
        );
    let mut damped_callback_values = Vec::with_capacity(2);
    match damped.native_mut() {
        BvpNativeSolver::Damped(native) => {
            let fixed_positions = native
                .BC_position_and_value
                .iter()
                .map(|&(position, _, _)| position)
                .collect::<std::collections::HashSet<_>>();
            for profile in [&damped_profile, &correction_profile] {
                let packed = DVector::from_iterator(
                    native.y.len(),
                    (0..profile.len())
                        .filter(|position| !fixed_positions.contains(position))
                        .map(|position| profile[(position / STATE_COUNT, position % STATE_COUNT)]),
                );
                let runtime_vector_type = match native.y.vec_type().as_str() {
                    "Sparse_3" => "Sparse".to_string(),
                    other => other.to_string(),
                };
                let typed = Vectors_type_casting(&packed, runtime_vector_type);
                let residual = native
                    .fun
                    .try_call(native.p, &*typed)
                    .expect("Damped residual callback should evaluate on shared profile")
                    .to_DVectorType();
                let jacobian = native
                    .jac
                    .as_mut()
                    .expect("Damped symbolic Jacobian callback exists")
                    .try_call(native.p, &*typed)
                    .expect("Damped Jacobian callback should evaluate on shared profile")
                    .to_DMatrixType();
                damped_callback_values.push((residual, jacobian));
            }
        }
        _ => panic!("expected native Damped solver"),
    }
    assert_eq!(
        damped_callback_values.len(),
        correction_callback_values.len()
    );
    for (
        profile_index,
        ((damped_residual, damped_jacobian), (correction_residual, correction_jacobian)),
    ) in damped_callback_values
        .iter()
        .zip(&correction_callback_values)
        .enumerate()
    {
        assert_eq!(damped_residual.len(), correction_residual.len());
        assert_eq!(damped_jacobian.shape(), correction_jacobian.shape());
        let residual_max_diff = (damped_residual - correction_residual).amax();
        let jacobian_max_diff = (damped_jacobian - correction_jacobian).amax();
        println!(
            "shared-profile callbacks profile={profile_index}: residual_max_diff={residual_max_diff:.6e}, jacobian_max_diff={jacobian_max_diff:.6e}"
        );
        assert!(
            residual_max_diff < 1e-12,
            "Damped and CorrectionNewton residuals differ at shared profile {profile_index}: {residual_max_diff:e}"
        );
        assert!(
            jacobian_max_diff < 1e-10,
            "Damped and CorrectionNewton Jacobians differ at shared profile {profile_index}: {jacobian_max_diff:e}"
        );
    }

    let profile_max_diff = (&damped_profile - &correction_profile).amax();
    let candidate_profile_max_diff = (&candidate_profile - &correction_profile).amax();
    let damped_defect = full_grid_trapezoid_defect(&problem, &damped_profile);
    let candidate_defect = full_grid_trapezoid_defect(&problem, &candidate_profile);
    let correction_defect = full_grid_trapezoid_defect(&problem, &correction_profile);
    let invariant_error = |profile: &DMatrix<f64>| {
        (0..profile.nrows())
            .map(|node| {
                ((2..STATE_COUNT)
                    .step_by(2)
                    .map(|state| profile[(node, state)])
                    .sum::<f64>()
                    - 1.0)
                    .abs()
            })
            .fold(0.0_f64, f64::max)
    };
    let damped_invariant_error = invariant_error(&damped_profile);
    let candidate_invariant_error = invariant_error(&candidate_profile);
    let correction_invariant_error = invariant_error(&correction_profile);
    let damped_stats = match damped.telemetry_snapshot() {
        BvpSolverTelemetrySnapshot::Damped(snapshot) => snapshot,
        _ => panic!("expected Damped telemetry snapshot"),
    };
    let candidate_stats = match candidate_damped.telemetry_snapshot() {
        BvpSolverTelemetrySnapshot::Damped(snapshot) => snapshot,
        _ => panic!("expected candidate Damped telemetry"),
    };
    println!(
        "CorrectionNewton-criteria Damped A/B: status={:?}, iterations={}, jacobians={}, damping_trials={}, linear_solves={}, max_profile_diff={candidate_profile_max_diff:.6e}, max_trapezoid_defect={candidate_defect:.6e}; reference CorrectionNewton defect={correction_defect:.6e}",
        candidate_stats.last_solve_status,
        candidate_stats.counters.iterations,
        candidate_stats.counters.jacobian_recalculations,
        candidate_stats.counters.damping_trials,
        candidate_stats.counters.linear_solves,
    );
    let candidate_weighted_rms =
        damped_terminal_weighted_rms(&candidate_damped, &candidate_profile, 1e-5);
    let (damped_final_correction_norm, damped_convergence_threshold, damped_weighted_rms) =
        match damped.native() {
            BvpNativeSolver::Damped(native) => {
                let correction = native.step(native.p, &*native.y).0;
                let relative_tolerances = vec![1e-5; native.y.len()];
                let correction_dense = correction.to_DVectorType();
                let correction_norm = correction_dense.norm();
                let scalar_threshold =
                    convergence_condition(&*native.y, &native.abs_tolerance, &relative_tolerances);
                let fixed_zero = native
                    .BC_position_and_value
                    .iter()
                    .map(|&(position, state, _)| (position, state, 0.0))
                    .collect();
                let full_correction = construct_full_solution(correction_dense, fixed_zero);
                let point_count = damped_profile.nrows();
                let component_count = damped_profile.ncols();
                let component_mean_abs = (0..component_count)
                    .map(|state| {
                        (0..point_count)
                            .map(|node| damped_profile[(node, state)].abs())
                            .sum::<f64>()
                            / point_count as f64
                    })
                    .collect::<Vec<_>>();
                let mut scaled_square_sum = 0.0;
                for node in 0..point_count {
                    for state in 0..component_count {
                        let full_index = node * component_count + state;
                        let scale = 1e-5 + 1e-5 * component_mean_abs[state];
                        let scaled = full_correction[full_index] / scale;
                        scaled_square_sum += scaled * scaled;
                    }
                }
                let weighted_rms =
                    (scaled_square_sum / (point_count * component_count) as f64).sqrt();
                (correction_norm, scalar_threshold, weighted_rms)
            }
            _ => panic!("expected native Damped solver"),
        };

    for profile in [&damped_profile, &candidate_profile, &correction_profile] {
        assert!(profile.iter().all(|value| value.is_finite()));
        for (name, conditions) in &problem.boundary_conditions {
            let state = problem
                .state_names
                .iter()
                .position(|candidate| candidate == name)
                .expect("fixture boundary state is declared");
            for &(endpoint, value) in conditions {
                let node = if endpoint == 0 {
                    0
                } else {
                    profile.nrows() - 1
                };
                assert!(
                    (profile[(node, state)] - value).abs() < 1e-12,
                    "{name} endpoint {endpoint}: got {}, expected {value}",
                    profile[(node, state)]
                );
            }
        }
    }
    assert!(damped_defect.is_finite() && damped_defect < 1e-3);
    assert!(candidate_defect.is_finite() && candidate_defect < 1e-3);
    assert!(correction_defect.is_finite() && correction_defect < 1e-3);
    assert!(damped_invariant_error < 1e-3);
    assert!(candidate_invariant_error < 1e-3);
    assert!(correction_invariant_error < 1e-3);
    assert_eq!(damped_stats.last_solve_status, BvpSolveStatus::Converged);
    assert!(damped_stats.counters.iterations > 0);
    assert!(correction_report.telemetry.jacobian_evaluations > 0);
    assert!(
        damped_final_correction_norm <= damped_convergence_threshold,
        "Damped terminal correction {damped_final_correction_norm:e} exceeds its scalar threshold {damped_convergence_threshold:e}"
    );
    assert!(damped_weighted_rms.is_finite() && damped_weighted_rms <= 1.0);
    assert!(candidate_weighted_rms.is_finite() && candidate_weighted_rms <= 1.0);
    assert!(
        correction_report
            .correction_norm
            .is_some_and(|norm| norm <= 1.0),
        "CorrectionNewton terminal weighted RMS must be <= 1: {:?}",
        correction_report.correction_norm
    );

    println!(
        "KiThe fixed-mesh production A/B: mesh_nodes={}, legacy_vs_CorrectionNewton_profile_diff={profile_max_diff:.6e}, DampedLegacy(iterations={}, jacobians={}, damping_trials={}, linear_solves={}, correction_l2={damped_final_correction_norm:.6e}, scalar_threshold={damped_convergence_threshold:.6e}, comparable_weighted_rms={damped_weighted_rms:.6e}, trapezoid_defect={damped_defect:.6e}, invariant_error={damped_invariant_error:.6e}), DampedCorrectionNewton(iterations={}, jacobians={}, damping_trials={}, linear_solves={}, weighted_rms={candidate_weighted_rms:.6e}, profile_diff={candidate_profile_max_diff:.6e}, trapezoid_defect={candidate_defect:.6e}, invariant_error={candidate_invariant_error:.6e}), CorrectionNewton(iterations={}, jacobians={}, damping_trials={}, linear_solves={}, weighted_rms={:?}, trapezoid_defect={correction_defect:.6e}, invariant_error={correction_invariant_error:.6e})",
        damped_profile.nrows(),
        damped_stats.counters.iterations,
        damped_stats.counters.jacobian_recalculations,
        damped_stats.counters.damping_trials,
        damped_stats.counters.linear_solves,
        candidate_stats.counters.iterations,
        candidate_stats.counters.jacobian_recalculations,
        candidate_stats.counters.damping_trials,
        candidate_stats.counters.linear_solves,
        correction_report.iterations,
        correction_report.telemetry.jacobian_evaluations,
        correction_report.telemetry.damping_trials,
        correction_report.telemetry.linear_solves,
        correction_report.correction_norm,
    );
}

#[test]
#[ignore = "fixed-mesh COMB-01/02 Damped versus CorrectionNewton work decomposition"]
fn rst_native_damped_vs_correction_newton_comb01_comb02_ab() {
    use crate::numerical::BVP_CorrectionNewton::tests::combustion::{
        COMB02_REFERENCE, comb01_problem, comb02_perturbed_reference_problem, options,
        parse_reference,
    };

    let combustion02_reference = parse_reference(COMB02_REFERENCE, 8);
    for steps in [32, 64, 128, 256] {
        let fixtures = [
            ("COMB-01", comb01_problem(steps)),
            (
                "COMB-02",
                comb02_perturbed_reference_problem(steps, &combustion02_reference),
            ),
        ];

        for (fixture_name, original_problem) in fixtures {
            let name = format!("{fixture_name}/N={steps}");
            let mut correction_options = options(
                &original_problem.state_names,
                CorrectionNewtonSymbolicFrontend::ExprLegacy,
                CorrectionNewtonMatrixLayout::Sparse,
            );
            correction_options.max_jacobian_age = 1;
            correction_options.logging_level = crate::numerical::logging::SolverLogLevel::Off;
            correction_options.telemetry_mode = CorrectionNewtonTelemetryMode::Timings;
            let matched_bounds = correction_options.bounds.clone();
            let (problem, projected_count, max_adjustment) =
                bounded_common_seed(&original_problem, &matched_bounds);
            println!(
                "{name} shared feasible seed: projected_free_values={projected_count}, max_adjustment={max_adjustment:.9e}"
            );
            let state_count = problem.state_names.len();
            let shared_seed_profile = full_profile_from_problem(&problem);
            for (state, state_name) in problem.state_names.iter().enumerate() {
                let (lower, upper) = matched_bounds
                    .get(state_name)
                    .expect("bounds cover each state");
                assert!(
                    shared_seed_profile
                        .column(state)
                        .iter()
                        .all(|value| { value.is_finite() && *value >= *lower && *value <= *upper })
                );
            }
            let bvp_problem = bvp_problem_from_correction_newton(&problem);
            let make_damped = |logging| {
                let mut solver = BvpSolver::new(
                    bvp_problem.clone(),
                    BvpSolverConfig::Damped {
                        options: damped_options_for_correction_newton_fixture(
                            &problem,
                            logging,
                            &matched_bounds,
                            BvpSymbolicAssemblyBackend::ExprLegacy,
                        )
                        .with_lambdify_telemetry_mode(
                            crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Detailed,
                        ),
                        // Damped's public matrix is row-wise per state, while its
                        // Newton iterate omits mixed-side BCs in node-major order.
                        // Prepare safely, then install the exact common free-state.
                        initial_guess: DMatrix::zeros(state_count, steps),
                    },
                )
                .unwrap_or_else(|error| panic!("{name} matched Damped prepare failed: {error}"));
                let BvpNativeSolver::Damped(native) = solver.native_mut() else {
                    panic!("expected Damped route")
                };
                native
                    .try_set_prepared_iterate(damped_packed_seed(&problem))
                    .unwrap_or_else(|error| panic!("{name} install matched Damped seed: {error}"));
                solver
            };

            let mut legacy = make_damped(BvpLoggingMode::Detailed);
            if let Err(error) = legacy.solve() {
                if let BvpSolverTelemetrySnapshot::Damped(snapshot) = legacy.telemetry_snapshot() {
                    print_decision_events(&format!("{name} legacy"), &snapshot.log_events);
                }
                if let BvpNativeSolver::Damped(native) = legacy.native() {
                    let limiting = legacy.telemetry_snapshot();
                    let limiting_index = match limiting {
                    BvpSolverTelemetrySnapshot::Damped(snapshot) => snapshot
                        .log_events
                        .iter()
                        .find(|event| {
                            event.kind
                                == crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::BoundLimitedStep
                        })
                        .map(|event| event.value_b as usize),
                    _ => None,
                };
                    if let Some(index) = limiting_index {
                        println!("{name} native state order: {:?}", native.values);
                        let fixed_positions = native
                            .BC_position_and_value
                            .iter()
                            .map(|&(position, _, _)| position)
                            .collect::<std::collections::HashSet<_>>();
                        let full_position = (0..state_count * steps + state_count)
                            .filter(|position| !fixed_positions.contains(position))
                            .nth(index)
                            .expect("bound-limited packed index maps to a free grid coordinate");
                        let state = full_position % state_count;
                        let node = full_position / state_count;
                        let correction = native.step(native.p, &*native.y).0;
                        println!(
                            "{name} bound-limited unknown: packed={index} full={full_position} node={node} state={} value={:.9e} delta={:.9e} bounds={:?}",
                            native.values[state],
                            native.y.get_val(index),
                            correction.get_val(index),
                            native
                                .Bounds
                                .as_ref()
                                .and_then(|bounds| bounds.get(&problem.state_names[state]))
                        );
                    }
                }
                println!(
                    "{name} legacy Damped failure snapshot: {:#?}",
                    legacy.telemetry_snapshot()
                );
                panic!("{name} legacy Damped solve failed: {error:?}");
            }

            let mut candidate = make_damped(BvpLoggingMode::Detailed);
            match candidate.native_mut() {
                BvpNativeSolver::Damped(native) => {
                    native.enable_correction_newton_criteria_for_test()
                }
                _ => panic!("expected candidate Damped route"),
            }
            candidate
                .solve()
                .unwrap_or_else(|error| panic!("{name} candidate Damped solve failed: {error}"));

            let mut correction =
                CorrectionNewtonSolver::from_symbolic(problem.clone(), correction_options)
                    .unwrap_or_else(|error| {
                        panic!("{name} CorrectionNewton prepare failed: {error}")
                    });
            let correction_report = correction
                .solve()
                .unwrap_or_else(|error| panic!("{name} CorrectionNewton solve failed: {error}"));

            let legacy_profile = profile_from_solver(&legacy, state_count);
            let candidate_profile = profile_from_solver(&candidate, state_count);
            let correction_profile = correction
                .full_grid_values(&correction_report.state)
                .expect("CorrectionNewton solution expands onto the fixed mesh");
            // Capture solve-only work before diagnostic callback probes below.
            let legacy_solve_snapshot = legacy.telemetry_snapshot();
            let candidate_solve_snapshot = candidate.telemetry_snapshot();
            let correction_callback_values =
                crate::numerical::BVP_CorrectionNewton::evaluate_symbolic_callbacks_for_test(
                    problem.clone(),
                    options(
                        &problem.state_names,
                        CorrectionNewtonSymbolicFrontend::ExprLegacy,
                        CorrectionNewtonMatrixLayout::Sparse,
                    ),
                    &[
                        shared_seed_profile.clone(),
                        legacy_profile.clone(),
                        correction_profile.clone(),
                    ],
                );
            let mut damped_callback_values = Vec::with_capacity(3);
            match legacy.native_mut() {
                BvpNativeSolver::Damped(native) => {
                    let fixed_positions = native
                        .BC_position_and_value
                        .iter()
                        .map(|&(position, _, _)| position)
                        .collect::<std::collections::HashSet<_>>();
                    for profile in [&shared_seed_profile, &legacy_profile, &correction_profile] {
                        let packed = DVector::from_iterator(
                            native.y.len(),
                            (0..profile.len())
                                .filter(|position| !fixed_positions.contains(position))
                                .map(|position| {
                                    profile[(position / state_count, position % state_count)]
                                }),
                        );
                        let runtime_vector_type = match native.y.vec_type().as_str() {
                            "Sparse_3" => "Sparse".to_string(),
                            other => other.to_string(),
                        };
                        let typed = Vectors_type_casting(&packed, runtime_vector_type);
                        let residual = native
                            .fun
                            .try_call(native.p, &*typed)
                            .unwrap_or_else(|error| {
                                panic!("{name} Damped residual callback failed: {error}")
                            })
                            .to_DVectorType();
                        let jacobian = native
                            .jac
                            .as_mut()
                            .expect("Damped symbolic Jacobian callback exists")
                            .try_call(native.p, &*typed)
                            .unwrap_or_else(|error| {
                                panic!("{name} Damped Jacobian callback failed: {error}")
                            })
                            .to_DMatrixType();
                        damped_callback_values.push((residual, jacobian));
                    }
                }
                _ => panic!("expected native Damped solver"),
            }
            assert_eq!(
                damped_callback_values.len(),
                correction_callback_values.len()
            );
            for (profile_index, ((damped_residual, damped_jacobian), (cn_residual, cn_jacobian))) in
                damped_callback_values
                    .iter()
                    .zip(&correction_callback_values)
                    .enumerate()
            {
                let residual_max_diff = (damped_residual - cn_residual).amax();
                let jacobian_max_diff = (damped_jacobian - cn_jacobian).amax();
                println!(
                    "{name} shared-profile callbacks profile={profile_index}: residual_max_diff={residual_max_diff:.6e}, jacobian_max_diff={jacobian_max_diff:.6e}"
                );
                assert!(residual_max_diff < 1e-12);
                assert!(jacobian_max_diff < 1e-10);
            }
            let BvpSolverTelemetrySnapshot::Damped(legacy_stats) = legacy_solve_snapshot else {
                panic!("expected legacy Damped telemetry");
            };
            let BvpSolverTelemetrySnapshot::Damped(candidate_stats) = candidate_solve_snapshot
            else {
                panic!("expected candidate Damped telemetry");
            };
            let legacy_callback = legacy_stats
                .legacy_lambdify
                .expect("ExprLegacy Lambdify callback telemetry is enabled");
            let candidate_callback = candidate_stats
                .legacy_lambdify
                .expect("candidate ExprLegacy Lambdify callback telemetry is enabled");
            let legacy_terminal_rms = damped_terminal_weighted_rms(&legacy, &legacy_profile, 1e-8);
            let candidate_terminal_rms =
                damped_terminal_weighted_rms(&candidate, &candidate_profile, 1e-8);
            let defect = |profile: &DMatrix<f64>| full_grid_trapezoid_defect(&problem, profile);
            let candidate_profile_diff = (&candidate_profile - &correction_profile).amax();
            let legacy_profile_diff = (&legacy_profile - &correction_profile).amax();
            let trials_by_iteration = candidate_stats
                .log_events
                .iter()
                .filter(|event| {
                    event.kind
                        == crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingTrial
                })
                .fold(HashMap::<u64, usize>::new(), |mut counts, event| {
                    *counts.entry(event.iteration).or_default() += 1;
                    counts
                });

            println!(
                "{name} fixed-mesh matched A/B ({steps} intervals): legacy Damped iters={} J={} factors={} cache_hits={} residual_calls={} residual_ms={:.3} callback_ms={:.3} linear_solves={} trials={} rejections={} terminal_RMS={legacy_terminal_rms:.6e} defect={:.6e}; candidate Damped iters={} J={} factors={} cache_hits={} residual_calls={} residual_ms={:.3} callback_ms={:.3} linear_solves={} trials={} rejections={} trials_by_iter={trials_by_iteration:?} terminal_RMS={candidate_terminal_rms:.6e} defect={:.6e} profile_diff={candidate_profile_diff:.6e}; CorrectionNewton iters={} J={} factors={} residual_calls={} residual_ms={:.3} linear_solves={} trials={} rejections={} terminal_RMS={:?} defect={:.6e} profile_diff={legacy_profile_diff:.6e}",
                legacy_stats.counters.iterations,
                legacy_stats.counters.jacobian_recalculations,
                legacy_stats.counters.factorizations,
                legacy_stats.counters.factorization_cache_hits,
                legacy_stats.counters.residual_calls,
                legacy_stats.timings.residual.as_secs_f64() * 1_000.0,
                legacy_callback.residual_elapsed.as_secs_f64() * 1_000.0,
                legacy_stats.counters.linear_solves,
                legacy_stats.counters.damping_trials,
                legacy_stats.counters.damping_rejections,
                defect(&legacy_profile),
                candidate_stats.counters.iterations,
                candidate_stats.counters.jacobian_recalculations,
                candidate_stats.counters.factorizations,
                candidate_stats.counters.factorization_cache_hits,
                candidate_stats.counters.residual_calls,
                candidate_stats.timings.residual.as_secs_f64() * 1_000.0,
                candidate_callback.residual_elapsed.as_secs_f64() * 1_000.0,
                candidate_stats.counters.linear_solves,
                candidate_stats.counters.damping_trials,
                candidate_stats.counters.damping_rejections,
                defect(&candidate_profile),
                correction_report.iterations,
                correction_report.telemetry.jacobian_evaluations,
                correction_report.telemetry.factorizations,
                correction_report.telemetry.residual_evaluations,
                correction_report
                    .telemetry
                    .residual_callback_time
                    .as_secs_f64()
                    * 1_000.0,
                correction_report.telemetry.linear_solves,
                correction_report.telemetry.damping_trials,
                correction_report.telemetry.damping_rejections,
                correction_report.correction_norm,
                defect(&correction_profile),
            );
            println!(
                "{name} mesh summary: work legacy/candidate/CN iters={}/{}/{}, J={}/{}/{}, residuals={}/{}/{}, solves={}/{}/{}, trials={}/{}/{}; max_callback_delta residual={:.3e}, jacobian={:.3e}; defects={:.3e}/{:.3e}/{:.3e}",
                legacy_stats.counters.iterations,
                candidate_stats.counters.iterations,
                correction_report.iterations,
                legacy_stats.counters.jacobian_recalculations,
                candidate_stats.counters.jacobian_recalculations,
                correction_report.telemetry.jacobian_evaluations,
                legacy_stats.counters.residual_calls,
                candidate_stats.counters.residual_calls,
                correction_report.telemetry.residual_evaluations,
                legacy_stats.counters.linear_solves,
                candidate_stats.counters.linear_solves,
                correction_report.telemetry.linear_solves,
                legacy_stats.counters.damping_trials,
                candidate_stats.counters.damping_trials,
                correction_report.telemetry.damping_trials,
                damped_callback_values
                    .iter()
                    .zip(&correction_callback_values)
                    .map(|((residual, _), (cn_residual, _))| (residual - cn_residual).amax())
                    .fold(0.0_f64, f64::max),
                damped_callback_values
                    .iter()
                    .zip(&correction_callback_values)
                    .map(|((_, jacobian), (_, cn_jacobian))| (jacobian - cn_jacobian).amax())
                    .fold(0.0_f64, f64::max),
                defect(&legacy_profile),
                defect(&candidate_profile),
                defect(&correction_profile),
            );
            if steps == 128 {
                print_decision_events(&name, &candidate_stats.log_events);
            }
            assert_eq!(legacy_stats.last_solve_status, BvpSolveStatus::Converged);
            assert_eq!(candidate_stats.last_solve_status, BvpSolveStatus::Converged);
            assert_eq!(
                legacy_stats.counters.iterations, correction_report.iterations as u64,
                "{name} legacy Damped and CorrectionNewton iteration counts"
            );
            assert_eq!(
                legacy_stats.counters.jacobian_recalculations,
                correction_report.telemetry.jacobian_evaluations as u64,
                "{name} legacy Damped and CorrectionNewton Jacobian refresh counts"
            );
            assert_eq!(
                legacy_stats.counters.factorizations,
                correction_report.telemetry.factorizations as u64,
                "{name} legacy Damped and CorrectionNewton factorization counts"
            );
            assert_eq!(
                legacy_stats.counters.linear_solves,
                correction_report.telemetry.linear_solves as u64,
                "{name} legacy Damped and CorrectionNewton linear solve counts"
            );
            assert_eq!(
                legacy_stats.counters.damping_trials,
                correction_report.telemetry.damping_trials as u64,
                "{name} legacy Damped and CorrectionNewton damping trial counts"
            );
            assert_eq!(
                legacy_stats.counters.damping_rejections,
                correction_report.telemetry.damping_rejections as u64,
                "{name} legacy Damped and CorrectionNewton rejected-trial counts"
            );
            assert_eq!(
                candidate_stats.counters.iterations, correction_report.iterations as u64,
                "{name} candidate and CorrectionNewton iteration counts"
            );
            assert_eq!(
                candidate_stats.counters.jacobian_recalculations,
                correction_report.telemetry.jacobian_evaluations as u64,
                "{name} candidate and CorrectionNewton Jacobian refresh counts"
            );
            assert_eq!(
                candidate_stats.counters.factorizations,
                correction_report.telemetry.factorizations as u64,
                "{name} candidate and CorrectionNewton factorization counts"
            );
            assert_eq!(
                candidate_stats.counters.linear_solves,
                correction_report.telemetry.linear_solves as u64,
                "{name} candidate and CorrectionNewton linear solve counts"
            );
            assert_eq!(
                candidate_stats.counters.damping_trials,
                correction_report.telemetry.damping_trials as u64,
                "{name} candidate and CorrectionNewton damping trial counts"
            );
            assert_eq!(
                candidate_stats.counters.damping_rejections,
                correction_report.telemetry.damping_rejections as u64,
                "{name} candidate and CorrectionNewton rejected-trial counts"
            );
            assert_eq!(
                legacy_stats.counters.residual_calls,
                correction_report.telemetry.residual_evaluations as u64,
                "{name} legacy Damped must reuse its initial and accepted-trial residuals"
            );
            assert_eq!(
                candidate_stats.counters.residual_calls,
                correction_report.telemetry.residual_evaluations as u64,
                "{name} candidate Damped must reuse its initial and accepted-trial residuals"
            );
            assert!(legacy_stats.timings.residual > std::time::Duration::ZERO);
            assert!(candidate_stats.timings.residual > std::time::Duration::ZERO);
            assert_eq!(
                legacy_callback.residual_calls,
                legacy_stats.counters.residual_calls
            );
            assert_eq!(
                candidate_callback.residual_calls,
                candidate_stats.counters.residual_calls
            );
            assert!(legacy_callback.residual_elapsed > std::time::Duration::ZERO);
            assert!(candidate_callback.residual_elapsed > std::time::Duration::ZERO);
            assert!(correction_report.telemetry.residual_callback_time > std::time::Duration::ZERO);
            assert!(legacy_terminal_rms <= 1.0);
            assert!(candidate_terminal_rms <= 1.0);
            assert!(
                correction_report
                    .correction_norm
                    .is_some_and(|norm| norm <= 1.0)
            );
            assert!(candidate_profile_diff < 1e-12);
            assert!(legacy_profile_diff < 1e-12);
            assert!(candidate_profile_diff.is_finite());
            assert!(defect(&candidate_profile).is_finite());
            assert!(defect(&correction_profile).is_finite());
            assert!(defect(&candidate_profile) < 1e-6);
            assert!(defect(&correction_profile) < 1e-6);
        }
    }
}

#[test]
#[ignore = "COMB-01/02 residual and Jacobian parity across ExprLegacy and AtomView"]
fn rst_native_damped_combustion_callback_parity_both_frontends() {
    use crate::numerical::BVP_CorrectionNewton::tests::combustion::{
        COMB02_REFERENCE, comb01_problem, comb02_perturbed_reference_problem, options,
        parse_reference,
    };

    let steps = 128;
    let reference = parse_reference(COMB02_REFERENCE, 8);
    let fixtures = [
        ("COMB-01", comb01_problem(steps)),
        (
            "COMB-02",
            comb02_perturbed_reference_problem(steps, &reference),
        ),
    ];

    for (name, problem) in fixtures {
        let full_seed = full_profile_from_problem(&problem);
        for (assembly, frontend) in [
            (
                BvpSymbolicAssemblyBackend::ExprLegacy,
                CorrectionNewtonSymbolicFrontend::ExprLegacy,
            ),
            (
                BvpSymbolicAssemblyBackend::AtomView,
                CorrectionNewtonSymbolicFrontend::AtomView,
            ),
        ] {
            let cn_options = options(
                &problem.state_names,
                frontend,
                CorrectionNewtonMatrixLayout::Sparse,
            );
            let damped_options = damped_options_for_correction_newton_fixture(
                &problem,
                BvpLoggingMode::Off,
                &cn_options.bounds,
                assembly,
            )
            .with_lambdify_telemetry_mode(
                crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Detailed,
            );
            let bvp_problem = bvp_problem_from_correction_newton(&problem);
            let mut damped = BvpSolver::new(
                bvp_problem,
                BvpSolverConfig::Damped {
                    options: damped_options,
                    initial_guess: DMatrix::zeros(problem.state_names.len(), steps),
                },
            )
            .unwrap_or_else(|error| panic!("{name}/{assembly:?} prepare: {error}"));

            let seed = damped_packed_seed(&problem);
            let packed_seed = match damped.native_mut() {
                BvpNativeSolver::Damped(native) => {
                    native
                        .try_eq_generate(None, None)
                        .unwrap_or_else(|error| panic!("{name}/{assembly:?} callbacks: {error}"));
                    native
                        .try_set_prepared_iterate(seed)
                        .unwrap_or_else(|error| panic!("{name}/{assembly:?} seed: {error}"));
                    native.y.iterate().collect::<Vec<_>>()
                }
                _ => panic!("expected native Damped solver"),
            };
            let (damped_residual, damped_jacobian) =
                eval_damped_residual_and_jacobian(&mut damped, &packed_seed);
            let cn_callbacks =
                crate::numerical::BVP_CorrectionNewton::evaluate_symbolic_callbacks_for_test(
                    problem.clone(),
                    cn_options,
                    &[full_seed.clone()],
                );
            let (cn_residual, cn_jacobian) = &cn_callbacks[0];
            let residual_max_diff = (&damped_residual - cn_residual).amax();
            let jacobian_max_diff = (&damped_jacobian - cn_jacobian).amax();
            assert!(residual_max_diff < 1e-12, "{name}/{assembly:?}");
            assert!(jacobian_max_diff < 1e-10, "{name}/{assembly:?}");

            let step = (packed_seed
                .iter()
                .fold(0.0_f64, |scale, value| scale.max(value.abs()))
                + 1.0)
                * 1.0e-6;
            let selected_columns = [
                0,
                packed_seed.len() / 7,
                packed_seed.len() / 3,
                packed_seed.len() / 2,
                packed_seed.len() * 2 / 3,
                packed_seed.len() - 1,
            ];
            let mut max_scaled_fd_error = 0.0_f64;
            for column in selected_columns {
                let mut plus = packed_seed.clone();
                let mut minus = packed_seed.clone();
                plus[column] += step;
                minus[column] -= step;
                let plus_residual = eval_damped_residual(&mut damped, &plus);
                let minus_residual = eval_damped_residual(&mut damped, &minus);
                for row in 0..damped_jacobian.nrows() {
                    let finite_difference =
                        (plus_residual[row] - minus_residual[row]) / (2.0 * step);
                    let analytic = damped_jacobian[(row, column)];
                    max_scaled_fd_error = max_scaled_fd_error
                        .max((finite_difference - analytic).abs() / (1.0 + analytic.abs()));
                }
            }
            println!(
                "{name}/{assembly:?} callback parity: residual={residual_max_diff:.3e}, jacobian={jacobian_max_diff:.3e}, selected_column_FD_scaled_error={max_scaled_fd_error:.3e}"
            );
            assert!(max_scaled_fd_error < 2.0e-5, "{name}/{assembly:?}");
        }
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

fn run_adaptive_refinement_acceptance_case(
    name: &str,
    fixture: &CorrectionNewtonProblem,
    initial_guess: DMatrix<f64>,
    options: DampedSolverOptions,
    bounds: &HashMap<String, (f64, f64)>,
    reference: Option<&[(f64, Vec<f64>)]>,
    invariant: impl Fn(&[f64]) -> f64,
    invariant_tolerance: f64,
    defect_tolerance: f64,
    profile_tolerance: Option<f64>,
    install_packed_seed: bool,
) {
    let problem = bvp_problem_from_correction_newton(fixture);
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options: options
                .with_bvp_telemetry_mode(BvpTelemetryMode::Counters)
                .with_bvp_logging_mode(BvpLoggingMode::Off),
            initial_guess,
        },
    )
    .unwrap_or_else(|error| panic!("{name} Damped preparation failed: {error:?}"));
    if install_packed_seed {
        let BvpNativeSolver::Damped(native) = solver.native_mut() else {
            unreachable!("the fixture was configured as Damped")
        };
        native
            .try_set_prepared_iterate(damped_packed_seed(fixture))
            .unwrap_or_else(|error| panic!("{name} install original prepared seed: {error}"));
    }

    match solver.solve() {
        Err(BvpApiError::MissingResult) => {
            assert!(
                solver.solution_view().is_none(),
                "{name} must not publish a profile after non-convergence"
            );
            let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
                unreachable!("the fixture was configured as Damped")
            };
            assert_eq!(snapshot.last_solve_status, BvpSolveStatus::NoResult);
            println!(
                "{name}: typed non-convergence; refinements={} iterations={} damping_trials={} linear_solves={}",
                snapshot.counters.grid_refinements,
                snapshot.counters.iterations,
                snapshot.counters.damping_trials,
                snapshot.counters.linear_solves
            );
        }
        Err(error) => {
            assert!(
                solver.solution_view().is_none(),
                "{name} published a profile despite typed failure {error:?}"
            );
            let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
                unreachable!("the fixture was configured as Damped")
            };
            assert_eq!(snapshot.last_solve_status, BvpSolveStatus::Failed);
            println!("{name}: typed failure without published profile: {error:?}");
        }
        Ok(_) => {
            let view = solver
                .solution_view()
                .unwrap_or_else(|| panic!("{name} reported success without publishing a profile"));
            let mesh = view.mesh().to_vec();
            let profile = DMatrix::from_fn(mesh.len(), fixture.state_names.len(), |node, state| {
                view.value(node, state)
                    .expect("every declared state is present in an accepted profile")
            });
            assert!(
                profile.iter().all(|value| value.is_finite()),
                "{name} non-finite profile"
            );
            if let Some(pair) = mesh.windows(2).find(|pair| pair[0] >= pair[1]) {
                panic!("{name} mesh is not strictly increasing: {pair:?}");
            }
            assert_eq!(mesh.first().copied(), Some(fixture.interval.0));
            assert_eq!(mesh.last().copied(), Some(fixture.interval.1));

            for (state, state_name) in fixture.state_names.iter().enumerate() {
                let (lower, upper) = bounds
                    .get(state_name)
                    .unwrap_or_else(|| panic!("{name} is missing bounds for {state_name}"));
                for node in 0..profile.nrows() {
                    let value = profile[(node, state)];
                    assert!(
                        value >= *lower - 1e-12 && value <= *upper + 1e-12,
                        "{name} state={state_name} node={node} value={value:e} escaped [{lower:e}, {upper:e}]"
                    );
                }
                for &(endpoint, boundary_value) in fixture
                    .boundary_conditions
                    .get(state_name)
                    .expect("every fixture state declares a boundary condition")
                {
                    let node = if endpoint == 0 {
                        0
                    } else {
                        profile.nrows() - 1
                    };
                    assert!(
                        (profile[(node, state)] - boundary_value).abs() <= 1e-12,
                        "{name} {state_name} endpoint={endpoint} got={} expected={boundary_value:e}",
                        profile[(node, state)]
                    );
                }
            }

            let invariant_error = (0..profile.nrows())
                .map(|node| {
                    let row = profile.row(node).iter().copied().collect::<Vec<_>>();
                    invariant(&row).abs()
                })
                .fold(0.0_f64, f64::max);
            let defect = adaptive_mesh_trapezoid_defect(fixture, &mesh, &profile);
            assert!(
                invariant_error <= invariant_tolerance,
                "{name} invariant error {invariant_error:e} exceeds {invariant_tolerance:e}"
            );
            assert!(
                defect <= defect_tolerance,
                "{name} independent Trapezoid defect {defect:e} exceeds {defect_tolerance:e}"
            );
            let profile_error =
                reference.map(|reference| interpolated_reference_error(reference, &mesh, &profile));
            // COMB fixtures alternate physical temperature/species states and flux derivatives.
            let physical_profile_error = profile_error.as_ref().map(|comparison| {
                comparison
                    .per_state_max_abs
                    .iter()
                    .step_by(2)
                    .copied()
                    .fold(0.0_f64, f64::max)
            });
            if let Some(error) = physical_profile_error {
                assert!(
                    error <= 1e-2,
                    "{name} physical-state reference error {error:e} exceeds 1% absolute gate"
                );
            }
            if let (Some(error), Some(tolerance)) = (profile_error.as_ref(), profile_tolerance) {
                if error.max_abs > tolerance {
                    println!(
                        "{name}: reference profile error {:.3e} at x={:.6e} state={} (actual={:.9e}, reference={:.9e}) exceeds review target {tolerance:.3e}; per-state max={:?}",
                        error.max_abs,
                        error.x_at_max,
                        error.state_at_max,
                        error.actual_at_max,
                        error.reference_at_max,
                        error.per_state_max_abs
                    );
                }
            }
            let BvpSolverTelemetrySnapshot::Damped(snapshot) = solver.telemetry_snapshot() else {
                unreachable!("the fixture was configured as Damped")
            };
            assert_eq!(snapshot.last_solve_status, BvpSolveStatus::Converged);
            assert!(snapshot.counters.iterations > 0);
            assert!(snapshot.counters.jacobian_recalculations > 0);
            assert!(snapshot.counters.linear_solves > 0);
            println!(
                "{name}: Converged nodes={} refinements={} iterations={} J={} residuals={} solves={} trials={} rejected={} invariant={invariant_error:.3e} defect={defect:.3e} physical_reference_error={physical_profile_error:?} reference_error={profile_error:?}",
                mesh.len(),
                snapshot.counters.grid_refinements,
                snapshot.counters.iterations,
                snapshot.counters.jacobian_recalculations,
                snapshot.counters.residual_calls,
                snapshot.counters.linear_solves,
                snapshot.counters.damping_trials,
                snapshot.counters.damping_rejections,
            );
        }
    }
}

#[test]
#[ignore = "10-interval adaptive-refinement quality matrix for KiThe, COMB-01 and COMB-02"]
fn rst_native_damped_refinement_strategies_on_three_combustion_fixtures_from_10_intervals() {
    use crate::numerical::BVP_CorrectionNewton::tests::combustion::{
        COMB01_REFERENCE, COMB02_REFERENCE, comb01_problem, comb02_perturbed_reference_problem,
        options as correction_options, parse_reference,
    };

    const START_INTERVALS: usize = 10;
    let strategies = [
        ("TwoPoint", GridRefinementMethod::TwoPoint(0.02, 0.02, 2.5)),
        ("Pearson", GridRefinementMethod::Pearson(1e-3, 1.4)),
        (
            "GrcarSmooke",
            GridRefinementMethod::GrcarSmooke(1e-3, 1e-2, 1.4),
        ),
        ("DoublePoints", GridRefinementMethod::DoublePoints),
    ];

    let mut kithe = correction_newton_fixed_grid_expr_problem();
    kithe.steps = START_INTERVALS;
    kithe.initial_guess = inlet_guess(START_INTERVALS);
    let kithe_bounds = (0..STATE_COUNT)
        .map(|state| {
            let bounds = if state == 0 {
                (-1.0, 3.0)
            } else if state == 1 || state % 2 == 1 {
                (-1e4, 1e4)
            } else {
                (-0.1, 1.1)
            };
            (format!("u{state}"), bounds)
        })
        .collect::<HashMap<_, _>>();

    let reference01 = parse_reference(COMB01_REFERENCE, 4);
    let reference02 = parse_reference(COMB02_REFERENCE, 8);
    let comb01_seed = comb01_problem(START_INTERVALS);
    let comb02_seed = comb02_perturbed_reference_problem(START_INTERVALS, &reference02);
    for (
        fixture_name,
        fixture,
        reference,
        invariant_tolerance,
        defect_tolerance,
        profile_tolerance,
        invariant,
    ) in [
        (
            "KiThe-26-state",
            kithe.clone(),
            None,
            1e-3,
            1e-3,
            None,
            Box::new(|state: &[f64]| {
                ((2..STATE_COUNT)
                    .step_by(2)
                    .map(|index| state[index])
                    .sum::<f64>()
                    - 1.0)
                    .abs()
            }) as Box<dyn Fn(&[f64]) -> f64>,
        ),
        (
            "COMB-01",
            comb01_seed.clone(),
            Some(reference01.as_slice()),
            2e-6,
            2e-5,
            Some(1e-3),
            Box::new(|state: &[f64]| {
                (state[0] + 2.0 * state[2] - 2.0)
                    .abs()
                    .max((state[1] + 2.0 * state[3]).abs())
            }) as Box<dyn Fn(&[f64]) -> f64>,
        ),
        (
            "COMB-02",
            comb02_seed.clone(),
            Some(reference02.as_slice()),
            2e-6,
            2e-5,
            Some(2e-3),
            Box::new(|state: &[f64]| {
                (state[0] + 3.0 * state[2] + 2.0 * state[4] - 3.0)
                    .abs()
                    .max((state[2] + state[4] + state[6] - 1.0).abs())
                    .max((state[1] + 3.0 * state[3] + 2.0 * state[5]).abs())
            }) as Box<dyn Fn(&[f64]) -> f64>,
        ),
    ] {
        let fixture_bounds = if fixture_name == "KiThe-26-state" {
            kithe_bounds.clone()
        } else {
            correction_options(
                &fixture.state_names,
                CorrectionNewtonSymbolicFrontend::ExprLegacy,
                CorrectionNewtonMatrixLayout::Sparse,
            )
            .bounds
        };
        let (fixture, projected_values, maximum_projection) =
            bounded_common_seed(&fixture, &fixture_bounds);
        assert_eq!(
            projected_values, 0,
            "{fixture_name} initial fixture must already satisfy its physical bounds; max adjustment={maximum_projection:e}"
        );

        for (strategy_name, strategy) in &strategies {
            let name = format!("{fixture_name}/{strategy_name}/N={START_INTERVALS}");
            let strategy_config = SolverParams {
                max_jac: Some(1),
                max_damp_iter: Some(12),
                adaptive: Some(AdaptiveGridConfig {
                    version: 1,
                    max_refinements: 2,
                    grid_method: strategy.clone(),
                }),
                ..SolverParams::default()
            };
            let mut options = if fixture_name == "KiThe-26-state" {
                damped_options(None, true).with_strategy_params(Some(strategy_config))
            } else {
                damped_options_for_correction_newton_fixture(
                    &fixture,
                    BvpLoggingMode::Off,
                    &fixture_bounds,
                    BvpSymbolicAssemblyBackend::ExprLegacy,
                )
                .with_strategy_params(Some(strategy_config))
            };
            options = options
                .with_bvp_telemetry_mode(BvpTelemetryMode::Counters)
                .with_bvp_logging_mode(BvpLoggingMode::Off);
            let initial_guess = if fixture_name == "KiThe-26-state" {
                inlet_guess(START_INTERVALS)
            } else {
                DMatrix::zeros(fixture.state_names.len(), START_INTERVALS)
            };
            run_adaptive_refinement_acceptance_case(
                &name,
                &fixture,
                initial_guess,
                options,
                &fixture_bounds,
                reference,
                &invariant,
                invariant_tolerance,
                defect_tolerance,
                profile_tolerance,
                fixture_name != "KiThe-26-state",
            );
        }
    }
}

#[test]
#[ignore = "focused COMB-01 Pearson reference-delta localization from a coarse mesh"]
fn rst_native_damped_comb01_pearson_reference_delta_diagnostic() {
    use crate::numerical::BVP_CorrectionNewton::tests::combustion::{
        COMB01_REFERENCE, comb01_problem, options as correction_options, parse_reference,
    };

    let fixture = comb01_problem(10);
    let bounds = correction_options(
        &fixture.state_names,
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonMatrixLayout::Sparse,
    )
    .bounds;
    let (fixture, projected, maximum_adjustment) = bounded_common_seed(&fixture, &bounds);
    assert_eq!(projected, 0, "COMB-01 coarse seed must remain feasible");
    assert_eq!(maximum_adjustment, 0.0);
    let reference = parse_reference(COMB01_REFERENCE, 4);
    let strategy = GridRefinementMethod::Pearson(1e-3, 1.4);
    let options = damped_options_for_correction_newton_fixture(
        &fixture,
        BvpLoggingMode::Off,
        &bounds,
        BvpSymbolicAssemblyBackend::ExprLegacy,
    )
    .with_strategy_params(Some(SolverParams {
        max_jac: Some(1),
        max_damp_iter: Some(12),
        adaptive: Some(AdaptiveGridConfig {
            version: 1,
            max_refinements: 3,
            grid_method: strategy,
        }),
        ..SolverParams::default()
    }));
    run_adaptive_refinement_acceptance_case(
        "COMB-01/Pearson/reference-diagnostic/N=10",
        &fixture,
        DMatrix::zeros(fixture.state_names.len(), 10),
        options,
        &bounds,
        Some(reference.as_slice()),
        &|state: &[f64]| {
            (state[0] + 2.0 * state[2] - 2.0)
                .abs()
                .max((state[1] + 2.0 * state[3]).abs())
        },
        2e-6,
        2e-5,
        None,
        true,
    );
}

#[test]
#[ignore = "focused COMB-02 TwoPoint reference-delta localization from a coarse mesh"]
fn rst_native_damped_comb02_twopoint_reference_delta_diagnostic() {
    use crate::numerical::BVP_CorrectionNewton::tests::combustion::{
        COMB02_REFERENCE, comb02_perturbed_reference_problem, options as correction_options,
        parse_reference,
    };

    let reference = parse_reference(COMB02_REFERENCE, 8);
    let fixture = comb02_perturbed_reference_problem(10, &reference);
    let bounds = correction_options(
        &fixture.state_names,
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonMatrixLayout::Sparse,
    )
    .bounds;
    let (fixture, projected, maximum_adjustment) = bounded_common_seed(&fixture, &bounds);
    assert_eq!(projected, 0, "COMB-02 coarse seed must remain feasible");
    assert_eq!(maximum_adjustment, 0.0);
    let options = damped_options_for_correction_newton_fixture(
        &fixture,
        BvpLoggingMode::Off,
        &bounds,
        BvpSymbolicAssemblyBackend::ExprLegacy,
    )
    .with_strategy_params(Some(SolverParams {
        max_jac: Some(1),
        max_damp_iter: Some(12),
        adaptive: Some(AdaptiveGridConfig {
            version: 1,
            max_refinements: 3,
            grid_method: GridRefinementMethod::TwoPoint(0.02, 0.02, 2.5),
        }),
        ..SolverParams::default()
    }));
    run_adaptive_refinement_acceptance_case(
        "COMB-02/TwoPoint/reference-diagnostic/N=10",
        &fixture,
        DMatrix::zeros(fixture.state_names.len(), 10),
        options,
        &bounds,
        Some(reference.as_slice()),
        &|state: &[f64]| {
            (state[0] + 3.0 * state[2] + 2.0 * state[4] - 3.0)
                .abs()
                .max((state[2] + state[4] + state[6] - 1.0).abs())
                .max((state[1] + 3.0 * state[3] + 2.0 * state[5]).abs())
        },
        2e-6,
        2e-5,
        None,
        true,
    );
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
