use RustedSciThe::command_interpreter::task_runner::{
    TaskRunResult, detect_task_kind_from_str, run_task_from_file,
};

fn main() {
    let worker = std::thread::Builder::new()
        .name("task-dispatcher-example".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(run_examples)
        .expect("example worker thread should start");
    worker
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn run_examples() {
    let ivp_path = "examples/task_docs/ivp_decay_task.txt";
    let bvp_path = "examples/task_docs/bvp_reference_task.txt";

    for path in [ivp_path, bvp_path] {
        let text = std::fs::read_to_string(path).expect("example task file should exist");
        let kind = detect_task_kind_from_str(&text).expect("task kind should be detectable");
        println!("{} -> detected {:?}", path, kind);

        let result = run_task_from_file(path).expect("task run should succeed");
        match result {
            TaskRunResult::Ivp(run) => {
                println!(
                    "IVP done, status={}, points={}",
                    run.status.as_deref().unwrap_or("unknown"),
                    run.t_result.as_ref().map(|t| t.len()).unwrap_or(0)
                );
            }
            TaskRunResult::Bvp(run) => {
                let rows = run.result.as_ref().map(|m| m.nrows()).unwrap_or(0);
                let cols = run.result.as_ref().map(|m| m.ncols()).unwrap_or(0);
                println!("BVP done, grid={}x{}", rows, cols);
            }
            TaskRunResult::Nonlinear(_) => println!("Nonlinear task done"),
            TaskRunResult::Optimization(_) => println!("Optimization task done"),
        }
    }
}
