//! Isolated RST-only reproducer for Damped adaptive-grid strategies.
//!
//! The problem is assembled directly with `Expr` and the public `BVP_api`.
//! It deliberately does not construct or depend on a KiThe reactor model.
//! Run with:
//! `cargo test --lib --no-default-features rst_damped_adaptive_grid_strategy_diagnostic_matrix -- --nocapture --test-threads=1`
//!
//! It compares ExprLegacy and AtomView symbolic assembly, sequential and
//! parallel Lambdify callbacks, and Pearson/Grcar-Smooke refinement. The six
//! first-order equations use states `u0..u5` and RHS expressions
//! `300*u1, 30*u1, 300000*u3, 30*u3, 300000*u5, 30*u5` on `[0, 1]`.

use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedSolverOptions, SolverParams,
};
use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use crate::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSolver, BvpSolverConfig,
    BvpSolverTelemetrySnapshot,
};
use crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::collections::HashMap;

const INTERVALS: usize = 32;
const MAX_REFINEMENTS: usize = 2;

fn run_case(
    frontend: BvpSymbolicAssemblyBackend,
    execution_policy: BvpLambdifyExecutionPolicy,
    method_name: &str,
    grid_method: GridRefinementMethod,
) -> Result<(usize, u64), String> {
    let equations = vec![
        Expr::parse_expression("300.0*u1"),
        Expr::parse_expression("30.0*u1"),
        Expr::parse_expression("300000.0*u3"),
        Expr::parse_expression("30.0*u3"),
        Expr::parse_expression("300000.0*u5"),
        Expr::parse_expression("30.0*u5"),
    ];
    let states = vec!["u0", "u1", "u2", "u3", "u4", "u5"];
    let boundary_conditions = vec![
        BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        },
        BvpBoundaryCondition {
            state: 1,
            side: BvpBoundarySide::Right,
            value: 0.025,
        },
        BvpBoundaryCondition {
            state: 2,
            side: BvpBoundarySide::Left,
            value: 0.6,
        },
        BvpBoundaryCondition {
            state: 3,
            side: BvpBoundarySide::Right,
            value: 1.5e-5,
        },
        BvpBoundaryCondition {
            state: 4,
            side: BvpBoundarySide::Left,
            value: 0.4,
        },
        BvpBoundaryCondition {
            state: 5,
            side: BvpBoundarySide::Right,
            value: -1.5e-5,
        },
    ];
    let mesh: Vec<f64> = (0..=INTERVALS)
        .map(|index| index as f64 / INTERVALS as f64)
        .collect();
    let problem = BvpProblem::new(
        equations,
        states.into_iter().map(str::to_owned).collect(),
        "x",
        mesh,
        boundary_conditions,
    )
    .map_err(|error| format!("problem construction: {error:?}"))?;

    let strategy_params = SolverParams {
        adaptive: Some(AdaptiveGridConfig {
            version: 1,
            max_refinements: MAX_REFINEMENTS,
            grid_method,
        }),
        ..SolverParams::default()
    };
    let base_options = DampedSolverOptions::banded_damped();
    let generated_backend_config = base_options
        .generated_backend_config
        .clone()
        .with_lambdify_execution_policy(execution_policy);
    let options = base_options
        .with_generated_backend_config(generated_backend_config)
        .with_symbolic_assembly_backend(frontend)
        .with_strategy_params(Some(strategy_params))
        .with_abs_tolerance(1e-8)
        .with_max_iterations(100)
        .with_rel_tolerance(
            ["u0", "u1", "u2", "u3", "u4", "u5"]
                .into_iter()
                .map(|state| (state.to_string(), 1e-8))
                .collect(),
        )
        .with_bounds(HashMap::from([
            ("u0".to_string(), (-10.0, 10.0)),
            ("u1".to_string(), (-10.0, 10.0)),
            ("u2".to_string(), (-1.0, 2.0)),
            ("u3".to_string(), (-10.0, 10.0)),
            ("u4".to_string(), (-1.0, 2.0)),
            ("u5".to_string(), (-10.0, 10.0)),
        ]))
        .with_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    let initial_guess = DMatrix::from_fn(6, INTERVALS, |state, _| match state {
        0 => 0.0,
        1 => 0.025,
        2 => 0.6,
        3 => 0.0,
        4 => 0.4,
        5 => 0.0,
        _ => unreachable!(),
    });
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options,
            initial_guess,
        },
    )
    .map_err(|error| format!("solver construction: {error:?}"))?;

    match solver.solve() {
        Ok(_) => {
            let solution = solver
                .solution_view()
                .expect("successful solve publishes a solution");
            let mesh_nodes = solution.mesh().len();
            if solution.state_count() != 6
                || solution.mesh().iter().any(|x| !x.is_finite())
                || (0..mesh_nodes).any(|node| {
                    (0..solution.state_count()).any(|state| {
                        solution
                            .value(node, state)
                            .is_none_or(|value| !value.is_finite())
                    })
                })
            {
                return Err("successful solve published an invalid profile".to_owned());
            }
            let refinements = match solver.telemetry_snapshot() {
                BvpSolverTelemetrySnapshot::Damped(snapshot) => snapshot.counters.grid_refinements,
                other => return Err(format!("unexpected telemetry route: {other:?}")),
            };
            println!(
                "frontend={frontend:?} policy={execution_policy:?} strategy={method_name} status=solved nodes={mesh_nodes} grid_refinements={refinements}"
            );
            Ok((mesh_nodes, refinements))
        }
        Err(error) => {
            let message = format!("{error:?}");
            println!(
                "frontend={frontend:?} policy={execution_policy:?} strategy={method_name} status=error error={message}"
            );
            Err(message)
        }
    }
}

#[test]
fn rst_damped_adaptive_grid_strategy_diagnostic_matrix() {
    let strategies = [
        (
            "Pearson(1e-3,1.4)",
            GridRefinementMethod::Pearson(1e-3, 1.4),
        ),
        (
            "GrcarSmooke(1e-3,1e-2,1.4)",
            GridRefinementMethod::GrcarSmooke(1e-3, 1e-2, 1.4),
        ),
    ];
    let frontends = [
        BvpSymbolicAssemblyBackend::ExprLegacy,
        BvpSymbolicAssemblyBackend::AtomView,
    ];
    let policies = [
        BvpLambdifyExecutionPolicy::Sequential,
        BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
    ];

    println!(
        "RST Damped adaptive-grid diagnostic: intervals={INTERVALS}, max_refinements={MAX_REFINEMENTS}, equations=6, states=6"
    );
    let mut outcomes = Vec::new();
    for frontend in frontends {
        for execution_policy in policies {
            for (name, method) in &strategies {
                outcomes.push((
                    format!("{frontend:?}/{execution_policy:?}/{name}"),
                    run_case(frontend, execution_policy, name, method.clone()),
                ));
            }
        }
    }

    for (case, result) in outcomes {
        let (mesh_nodes, refinements) =
            result.unwrap_or_else(|error| panic!("grid-refinement case failed: {case}: {error}"));
        assert_eq!(
            refinements, MAX_REFINEMENTS as u64,
            "expected both adaptive refinements in {case}"
        );
        assert_eq!(mesh_nodes, 786, "unexpected final mesh in {case}");
    }
}
