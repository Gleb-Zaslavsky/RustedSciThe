fn main() {
    let result = std::thread::Builder::new()
        .name("rst-task-cli".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(RustedSciThe::task_shell_cli::run)
        .expect("task CLI worker thread should start")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));

    if let Err(code) = result {
        std::process::exit(code);
    }
}
