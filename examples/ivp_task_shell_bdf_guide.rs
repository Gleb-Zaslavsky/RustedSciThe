use RustedSciThe::command_interpreter::task_parser_ivp::{
    parse_ivp_task_from_str, run_ivp_task_from_str,
};

fn main() {
    // BDF is often the first stiff method users reach for, so it deserves its
    // own minimal guide in the text-shell examples.
    let task = r#"
task
solver: IVP
method: BDF

equations
arg: t
parameters: a
parameter_values: 20.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05

postprocessing
save_csv: false
plot: false
"#;

    let spec = parse_ivp_task_from_str(task).expect("task should parse");
    println!(
        "parsed BDF IVP task: method={:?}, parameters={:?}",
        spec.solver.method, spec.equations.parameter_values
    );

    let result = run_ivp_task_from_str(task).expect("task should solve");
    println!("status = {:?}", result.status);

    if let Some(y) = result.y_result {
        let final_y = y[(y.nrows() - 1, 0)];
        println!("final y(t_end) = {final_y:.6}");
    }
}
