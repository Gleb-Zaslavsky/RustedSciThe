use std::collections::HashMap;

use RustedSciThe::command_interpreter::task_parser::DocumentParser;
use RustedSciThe::command_interpreter::task_parser_bvp::{
    BoundaryConditionSpec, BvpEquationSpec, BvpInitialGuessSpec, BvpMeshSpec, BvpProblemSpec,
    build_bvp_solver_from_problem_and_settings, build_bvp_solver_from_spec,
    parse_bvp_solver_settings_from_document, parse_bvp_task_from_str,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

// Run with:
// cargo run --example bvp_task_shell_guide

fn parse_settings_document(
    text: &str,
) -> RustedSciThe::command_interpreter::task_parser::DocumentMap {
    let mut parser = DocumentParser::new(text.to_string());
    parser
        .parse_document()
        .expect("settings document should parse");
    parser
        .get_result()
        .expect("settings document map should exist")
        .clone()
}

fn main() {
    let full_task = r#"
task
solver: BVP
strategy: Damped
scheme: forward
method: Sparse

equations
arg: x
unknowns: z, y
rhs: y-z, -z^3

boundary_conditions
z_left: 1.0
y_right: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 20

initial_guess
guess: 0.0, 0.0

solver_options
tolerance: 1e-5
max_iterations: 20
loglevel: off

postprocessing
save_csv: false
plot: false
"#;

    let spec = parse_bvp_task_from_str(full_task).expect("full BVP task should parse");
    println!(
        "full-task path: strategy={:?}, backend={:?}, unknowns={:?}",
        spec.solver.strategy, spec.solver.backend, spec.equations.unknowns
    );

    let mut solver = build_bvp_solver_from_spec(&spec).expect("full BVP task should build");
    solver.solve();
    let full_matrix = solver
        .get_result()
        .expect("full-task solve should produce a result matrix");
    println!(
        "full-task solution shape = {} x {}",
        full_matrix.nrows(),
        full_matrix.ncols()
    );

    let settings_only = r#"
task
solver: BVP
strategy: Frozen
scheme: forward
method: Banded

solver_options
tolerance: 1e-6
max_iterations: 24
generated_backend: banded_aot_tcc
aot_build_policy: build_if_missing
aot_build_profile: debug
aot_execution_policy: sequential
banded_linear_solver: faithful
refinement_steps: 1
"#;

    let settings_doc = parse_settings_document(settings_only);
    let settings = parse_bvp_solver_settings_from_document(&settings_doc)
        .expect("solver-only settings should parse");

    let problem = BvpProblemSpec {
        equations: BvpEquationSpec {
            arg: "x".to_string(),
            unknowns: vec!["y".to_string()],
            rhs: vec![Expr::parse_expression("-y")],
            parameter_names: vec![],
            parameter_values: HashMap::new(),
        },
        boundary_conditions: BoundaryConditionSpec {
            conditions: HashMap::from([("y".to_string(), vec![(0usize, 1.0f64)])]),
        },
        mesh: BvpMeshSpec {
            t0: 0.0,
            t_end: 1.0,
            n_steps: 16,
        },
        initial_guess: BvpInitialGuessSpec { values: vec![0.0] },
    };

    println!(
        "split path: strategy={:?}, backend={:?}, generated_backend={:?}",
        settings.solver.strategy,
        settings.solver.backend,
        settings.solver_options.generated_backend.preset
    );

    let mut split_solver = build_bvp_solver_from_problem_and_settings(&problem, &settings)
        .expect("Rust problem + task-doc settings should build");
    split_solver.solve();
    let split_matrix = split_solver
        .get_result()
        .expect("split-path solve should produce a result matrix");
    println!(
        "split-path solution shape = {} x {}",
        split_matrix.nrows(),
        split_matrix.ncols()
    );
}
