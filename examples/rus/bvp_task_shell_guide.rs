use std::collections::HashMap;

use RustedSciThe::command_interpreter::task_parser::DocumentParser;
use RustedSciThe::command_interpreter::task_parser_bvp::{
    build_bvp_solver_from_problem_and_settings, build_bvp_solver_from_spec,
    parse_bvp_solver_settings_from_document, parse_bvp_task_from_str, BoundaryConditionSpec,
    BvpEquationSpec, BvpInitialGuessSpec, BvpMeshSpec, BvpProblemSpec,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

// Запуск:
// cargo run --example rus_bvp_task_shell_guide

fn parse_settings_document(
    text: &str,
) -> RustedSciThe::command_interpreter::task_parser::DocumentMap {
    let mut parser = DocumentParser::new(text.to_string());
    parser.parse_document().expect("документ со settings должен распарситься");
    parser
        .get_result()
        .expect("карта документа settings должна существовать")
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

    let spec = parse_bvp_task_from_str(full_task).expect("полная BVP-задача должна парситься");
    println!(
        "полный путь: strategy={:?}, backend={:?}, unknowns={:?}",
        spec.solver.strategy, spec.solver.backend, spec.equations.unknowns
    );

    let mut solver = build_bvp_solver_from_spec(&spec).expect("полная BVP-задача должна собираться");
    solver.solve();
    let full_matrix = solver
        .get_result()
        .expect("полный путь должен вернуть матрицу результата");
    println!(
        "shape полного пути = {} x {}",
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
        .expect("solver-only settings должны парситься");

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
        "разделённый путь: strategy={:?}, backend={:?}, generated_backend={:?}",
        settings.solver.strategy,
        settings.solver.backend,
        settings.solver_options.generated_backend.preset
    );

    let mut split_solver = build_bvp_solver_from_problem_and_settings(&problem, &settings)
        .expect("Rust problem + settings из task-doc должны собираться");
    split_solver.solve();
    let split_matrix = split_solver
        .get_result()
        .expect("разделённый путь должен вернуть матрицу результата");
    println!(
        "shape разделённого пути = {} x {}",
        split_matrix.nrows(),
        split_matrix.ncols()
    );
}
