use RustedSciThe::command_interpreter::task_parser_bvp::create_bvp_template_file;
use RustedSciThe::command_interpreter::task_parser_ivp::create_ivp_template_file;
use RustedSciThe::command_interpreter::task_runner::{
    TaskDocumentKind, TaskRunResult, parse_task_spec_from_file, render_task_preview,
    run_task_from_spec,
};
use std::path::PathBuf;

fn print_usage() {
    println!("RustedSciThe task-shell CLI");
    println!();
    println!("Usage:");
    println!("  rustedscithe <task-file>");
    println!("  rustedscithe --template ivp [output-path]");
    println!("  rustedscithe --template bvp [output-path]");
    println!("  rustedscithe --showcase");
    println!();
    println!("Examples:");
    println!("  rustedscithe examples/task_docs/ivp_decay_task.txt");
    println!("  rustedscithe --template ivp");
    println!("  rustedscithe --showcase");
}

fn print_showcase() {
    println!("RustedSciThe showcase (minimal crate storefront)");
    println!();
    println!("1) Task-shell quick start (text task -> solver run):");
    println!("  rustedscithe examples/task_docs/ivp_decay_task.txt");
    println!("  rustedscithe examples/task_docs/bvp_reference_task.txt");
    println!("  rustedscithe examples/task_docs/ivp_solid_combustion_with_sublimation_task.txt");
    println!("  rustedscithe examples/task_docs/bvp_combustion_1000_task.txt");
    println!("  rustedscithe --template ivp");
    println!("  rustedscithe --template bvp");
    println!();
    println!("2) Dispatcher / shell infrastructure:");
    println!("  cargo run --example task_dispatcher_guide");
    println!("  cargo run --example task_shell_guides_index");
    println!();
    println!("3) Legacy compatibility mini-showcase:");
    println!("  cargo run --example legacy_main_compat -- 0");
    println!("  cargo run --example legacy_main_compat -- 2");
    println!("  cargo run --example legacy_main_compat -- 4");
    println!("  cargo run --example legacy_main_compat -- 5");
    println!("  cargo run --example legacy_main_compat -- 9");
    println!("  cargo run --example legacy_main_compat -- 10");
    println!("  cargo run --example legacy_main_compat -- 11");
    println!("  cargo run --example legacy_main_compat -- 15");
    println!("  cargo run --example legacy_main_compat -- 19");
    println!();
    println!("4) Modern IVP / LSODE2 guides:");
    println!("  cargo run --example ivp_backends_guide");
    println!("  cargo run --example lsode2_task_shell_guide");
    println!("  cargo run --example lsode2_numerical_guide");
    println!("  cargo run --example lsode2_lambdify_guide");
    println!("  cargo run --example lsode2_aot_guide");
    println!("  cargo run --example lsode2_manual_bdf_guide");
    println!("  cargo run --example lsode2_manual_adams_guide");
    println!();
    println!("5) Modern BVP / banded guides:");
    println!("  cargo run --example bvp_backends_guide");
    println!("  cargo run --example banded_solvers_guide");
    println!();
    println!("6) More examples:");
    println!("  cargo run --example task_shell_guides_index");
    println!("  cargo run --example radau_backends_guide");
    println!("  cargo run --example curve_fitting_guide");
    println!("  cargo run --example nonlinear_systems_guide");
}

fn print_run_summary(result: &TaskRunResult) {
    match result {
        TaskRunResult::Ivp(ivp) => {
            let status = ivp.status.as_deref().unwrap_or("unknown");
            let rows = ivp.t_result.as_ref().map(|t| t.len()).unwrap_or(0);
            let cols = ivp.y_result.as_ref().map(|y| y.ncols()).unwrap_or(0);
            println!("Task kind : IVP");
            println!("Status    : {status}");
            println!("Grid rows : {rows}");
            println!("State dim : {cols}");
        }
        TaskRunResult::Bvp(bvp) => {
            let (rows, cols) = bvp
                .result
                .as_ref()
                .map(|m| (m.nrows(), m.ncols()))
                .unwrap_or((0, 0));
            println!("Task kind : BVP");
            println!("Status    : finished");
            println!("Grid rows : {rows}");
            println!("State dim : {cols}");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        print_usage();
        std::process::exit(1);
    }

    if args[1] == "--help" || args[1] == "-h" {
        print_usage();
        return;
    }
    if args[1] == "--showcase" {
        print_showcase();
        return;
    }

    if args[1] == "--template" {
        if args.len() < 3 {
            eprintln!("`--template` expects `ivp` or `bvp`.");
            print_usage();
            std::process::exit(2);
        }
        let output = if args.len() >= 4 {
            Some(PathBuf::from(&args[3]))
        } else {
            None
        };
        match args[2].to_ascii_lowercase().as_str() {
            "ivp" => {
                create_ivp_template_file(output);
                println!("IVP template created.");
            }
            "bvp" => {
                create_bvp_template_file(output);
                println!("BVP template created.");
            }
            other => {
                eprintln!("Unknown template kind `{other}` (expected `ivp` or `bvp`).");
                std::process::exit(2);
            }
        }
        return;
    }

    let input_path = PathBuf::from(&args[1]);
    match parse_task_spec_from_file(&input_path) {
        Ok(spec) => {
            let kind = match spec.kind() {
                TaskDocumentKind::Ivp => "IVP",
                TaskDocumentKind::Bvp => "BVP",
            };
            println!("Task file : {}", input_path.display());
            println!("Detected  : {kind}");
            println!();
            println!("{}", render_task_preview(&spec));
            match run_task_from_spec(spec) {
                Ok(result) => {
                    print_run_summary(&result);
                }
                Err(err) => {
                    eprintln!("Task execution failed: {err}");
                    std::process::exit(3);
                }
            }
        }
        Err(err) => {
            eprintln!("Task execution failed: {err}");
            std::process::exit(3);
        }
    }
}
