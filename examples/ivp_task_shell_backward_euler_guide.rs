use RustedSciThe::command_interpreter::task_parser_ivp::{
    parse_ivp_task_from_str, run_ivp_task_from_str,
};
// run cargo run --example ivp_task_shell_backward_euler_guide

fn main() {
    // This guide focuses on a stiff-capable method while keeping the textual
    // document as close as possible to what a user would write by hand.
    let task = r#"
task
solver: IVP
method: BackwardEuler

equations
arg: t
y: -12.0*y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
step_size: 1e-3
tolerance: 1e-8
max_iterations: 100

postprocessing
save_csv: false
plot: false
"#;

    let spec = parse_ivp_task_from_str(task).expect("task should parse");
    println!(
        "parsed stiff IVP task: method={:?}, unknowns={:?}",
        spec.solver.method, spec.equations.unknowns
    );

    let result = run_ivp_task_from_str(task).expect("task should solve");
    println!("status = {:?}", result.status);

    if let Some(y) = result.y_result {
        let final_y = y[(y.nrows() - 1, 0)];
        println!("final y(t_end) = {final_y:.6}");
    }
}
