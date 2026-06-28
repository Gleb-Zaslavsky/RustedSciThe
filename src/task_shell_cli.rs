use crate::command_interpreter::bvp_dialogue::run_bvp_template_dialogue;
use crate::command_interpreter::docx_parser::{process_docx, process_docx_check};
use crate::command_interpreter::ivp_dialogue::run_ivp_template_dialogue;
use crate::command_interpreter::task_runner::{
    parse_task_spec_from_file, render_task_check, render_task_preview, run_task_from_spec,
    TaskDocumentKind, TaskRunResult,
};
use clap::{Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Parser)]
#[command(
    name = "rustedscithe",
    version,
    about = "RustedSciThe task-shell CLI",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run a task document directly.
    Run {
        task_file: PathBuf,
    },
    /// Print a substituted preview without running the solver.
    Check {
        task_file: PathBuf,
    },
    /// Convert DOCX to a task document.
    Convert {
        docx_file: PathBuf,
    },
    /// Convert DOCX and print a task check preview.
    #[command(name = "convert-check", alias = "convert_check")]
    ConvertCheck {
        docx_file: PathBuf,
    },
    /// Generate a parser-ready task template.
    Template {
        kind: TemplateKind,
        output_path: Option<PathBuf>,
    },
    /// Probe external toolchain/runtime dependencies used by the project.
    #[command(name = "check-ffi-dependencies")]
    CheckFfiDependencies,
    /// Print a curated command showcase.
    Showcase,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TemplateKind {
    Ivp,
    Bvp,
}

fn normalize_legacy_cli_args(args: Vec<OsString>) -> Vec<OsString> {
    if args.len() <= 1 {
        return args;
    }

    let first = args[1].to_string_lossy();
    match first.as_ref() {
        "--help" | "-h" => args,
        "run" | "check" | "convert" | "convert-check" | "template" | "showcase"
        | "check-ffi-dependencies" => args,
        "convert_check" => {
            let mut normalized = args;
            normalized[1] = OsString::from("convert-check");
            normalized
        }
        "check_ffi_dependances" | "check_ffi_dependencies" => {
            let mut normalized = args;
            normalized[1] = OsString::from("check-ffi-dependencies");
            normalized
        }
        "--showcase" => {
            let mut normalized = args;
            normalized[1] = OsString::from("showcase");
            normalized
        }
        "--check" => {
            let mut normalized = vec![args[0].clone(), OsString::from("check")];
            normalized.extend(args.into_iter().skip(2));
            normalized
        }
        "--template" => {
            let mut normalized = vec![args[0].clone(), OsString::from("template")];
            normalized.extend(args.into_iter().skip(2));
            normalized
        }
        other if !other.starts_with('-') => {
            let mut normalized = vec![args[0].clone(), OsString::from("run"), args[1].clone()];
            normalized.extend(args.into_iter().skip(2));
            normalized
        }
        _ => args,
    }
}

fn print_showcase() {
    println!("RustedSciThe showcase (minimal crate storefront)");
    println!();
    println!("1) Task-shell quick start (text task -> solver run):");
    println!("  rustedscithe run examples/task_docs/ivp_decay_task.txt");
    println!("  rustedscithe run examples/task_docs/bvp_reference_task.txt");
    println!("  rustedscithe run examples/task_docs/ivp_solid_combustion_with_sublimation_task.txt");
    println!("  rustedscithe run examples/task_docs/bvp_combustion_1000_task.txt");
    println!("  rustedscithe template ivp");
    println!("  rustedscithe template bvp");
    println!("  rustedscithe template ivp examples/task_docs/ivp_task_template.txt");
    println!("  rustedscithe template bvp examples/task_docs/bvp_task_template.txt");
    println!();
    println!("2) Dispatcher / shell infrastructure:");
    println!("  cargo run --example task_dispatcher_guide");
    println!("  cargo run --example task_shell_guides_index");
    println!("  rustedscithe check-ffi-dependencies");
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

fn command_available(command: &str, probe_arg: &str) -> Option<String> {
    let output = Command::new(command).arg(probe_arg).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            Some("(no version text)".to_string())
        } else {
            Some(stderr)
        }
    } else {
        Some(stdout)
    }
}

fn print_dependency_row(name: &str, status: &str, detail: &str) {
    println!("{name:<14} | {status:<8} | {detail}");
}

fn print_ffi_dependency_report() {
    println!("RustedSciThe external dependency report");
    println!("platform: {}", std::env::consts::OS);
    println!("probe strategy: PATH-based process checks (portable across Windows, Linux, and macOS)");
    println!("name           | status   | detail");
    println!("----------------------------------------------");

    let rustc = command_available("rustc", "--version");
    let cargo = command_available("cargo", "--version");
    let tcc = command_available("tcc", "-v");
    let gcc = command_available("gcc", "--version");
    let zig = command_available("zig", "version");
    let gnuplot = command_available("gnuplot", "--version");
    let pandoc = command_available("pandoc", "--version");

    print_dependency_row(
        "rustc",
        if rustc.is_some() { "ok" } else { "missing" },
        rustc.as_deref().unwrap_or("not found on PATH"),
    );
    print_dependency_row(
        "cargo",
        if cargo.is_some() { "ok" } else { "missing" },
        cargo.as_deref().unwrap_or("not found on PATH"),
    );
    print_dependency_row(
        "tcc",
        if tcc.is_some() { "ok" } else { "missing" },
        tcc.as_deref().unwrap_or("not found on PATH"),
    );
    print_dependency_row(
        "gcc",
        if gcc.is_some() { "ok" } else { "missing" },
        gcc.as_deref().unwrap_or("not found on PATH"),
    );
    print_dependency_row(
        "zig",
        if zig.is_some() { "ok" } else { "missing" },
        zig.as_deref().unwrap_or("not found on PATH"),
    );
    print_dependency_row(
        "gnuplot",
        if gnuplot.is_some() { "ok" } else { "missing" },
        gnuplot.as_deref().unwrap_or("not found on PATH"),
    );
    print_dependency_row(
        "pandoc",
        if pandoc.is_some() { "ok" } else { "missing" },
        pandoc.as_deref().unwrap_or("not found on PATH"),
    );

    let arrayfire_enabled = cfg!(feature = "arrayfire");
    if arrayfire_enabled {
        print_dependency_row("arrayfire", "feature", "compiled with `arrayfire` feature");
        let runtime_probe = std::panic::catch_unwind(|| {
            #[cfg(feature = "arrayfire")]
            {
                arrayfire::info();
            }
        });
        match runtime_probe {
            Ok(_) => print_dependency_row("arrayfire", "runtime", "info() probe executed"),
            Err(_) => print_dependency_row(
                "arrayfire",
                "runtime",
                "info() probe panicked or runtime library unavailable",
            ),
        }
    } else {
        print_dependency_row("arrayfire", "disabled", "crate compiled without `arrayfire` feature");
    }
}

pub fn run() -> Result<(), i32> {
    let raw_args: Vec<OsString> = std::env::args_os().collect();
    let normalized_args = normalize_legacy_cli_args(raw_args);
    let cli = Cli::parse_from(normalized_args);

    match cli.command {
        Commands::Run { task_file } => match parse_task_spec_from_file(&task_file) {
            Ok(spec) => {
                let kind = match spec.kind() {
                    TaskDocumentKind::Ivp => "IVP",
                    TaskDocumentKind::Bvp => "BVP",
                };
                println!("Task file : {}", task_file.display());
                println!("Detected  : {kind}");
                println!();
                println!("{}", render_task_preview(&spec));
                match run_task_from_spec(spec) {
                    Ok(result) => {
                        print_run_summary(&result);
                        Ok(())
                    }
                    Err(err) => {
                        eprintln!("Task execution failed: {err}");
                        Err(3)
                    }
                }
            }
            Err(err) => {
                eprintln!("Task execution failed: {err}");
                Err(3)
            }
        },
        Commands::Check { task_file } => match parse_task_spec_from_file(&task_file) {
            Ok(spec) => {
                let kind = match spec.kind() {
                    TaskDocumentKind::Ivp => "IVP",
                    TaskDocumentKind::Bvp => "BVP",
                };
                println!("Task file : {}", task_file.display());
                println!("Detected  : {kind}");
                println!();
                println!("{}", render_task_check(&spec));
                Ok(())
            }
            Err(err) => {
                eprintln!("Task execution failed: {err}");
                Err(3)
            }
        },
        Commands::Convert { docx_file } => match process_docx(docx_file.to_string_lossy().as_ref()) {
            Ok(text) => {
                println!("{text}");
                Ok(())
            }
            Err(err) => {
                eprintln!("{err}");
                Err(3)
            }
        },
        Commands::ConvertCheck { docx_file } => {
            match process_docx_check(docx_file.to_string_lossy().as_ref()) {
                Ok(text) => {
                    println!("{text}");
                    Ok(())
                }
                Err(err) => {
                    eprintln!("{err}");
                    Err(3)
                }
            }
        }
        Commands::Template { kind, output_path } => {
            match kind {
                TemplateKind::Ivp => run_ivp_template_dialogue(output_path),
                TemplateKind::Bvp => run_bvp_template_dialogue(output_path),
            }
            Ok(())
        }
        Commands::CheckFfiDependencies => {
            print_ffi_dependency_report();
            Ok(())
        }
        Commands::Showcase => {
            print_showcase();
            Ok(())
        }
    }
}
