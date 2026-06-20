use RustedSciThe::command_interpreter::task_parser_bvp::{
    parse_bvp_task_from_str, run_bvp_task_from_str,
};
// запуск: cargo run --example bvp_task_shell_guide

fn main() {
    // Это минимальный BVP-документ, который всё ещё прогоняет полный shell:
    // символические уравнения, граничные условия, сетка, начальное приближение
    // и настройки солвера всё приходят из текстового слоя.
    let task = r#"
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

    let spec = parse_bvp_task_from_str(task).expect("task should parse");
    println!(
        "parsed BVP task: strategy={:?}, backend={:?}, unknowns={:?}",
        spec.solver.strategy, spec.solver.backend, spec.equations.unknowns
    );

    let result = run_bvp_task_from_str(task).expect("task should solve");
    let matrix = result
        .result
        .expect("solver should produce a result matrix");
    println!("solution shape = {} x {}", matrix.nrows(), matrix.ncols());
}
