use RustedSciThe::command_interpreter::task_parser_ivp::{
    parse_ivp_task_from_str, run_ivp_task_from_str,
};
// запуск: cargo run --example ivp_task_shell_guide
fn main() {
    // Гайд намеренно использует текстовый shell вместо ручной сборки
    // `UniversalODESolver`. Именно такой workflow мы хотим показать внешнему пользователю первым.
    let task = r#"
task
solver: IVP
method: RK45

equations
arg: t
parameters: a
parameter_values: 1.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 0.5
y0: 1.0

solver_options
step_size: 1e-3

postprocessing
save_csv: false
plot: false
"#;

    let spec = parse_ivp_task_from_str(task).expect("task should parse");
    println!(
        "parsed IVP task: method={:?}, unknowns={:?}, arg={}",
        spec.solver.method, spec.equations.unknowns, spec.equations.arg
    );

    let result = run_ivp_task_from_str(task).expect("task should solve");
    println!("status = {:?}", result.status);

    if let Some(y) = result.y_result {
        let final_y = y[(y.nrows() - 1, 0)];
        println!("final y(t_end) = {final_y:.6}");
    }
}
