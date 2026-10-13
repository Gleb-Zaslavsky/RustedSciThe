use crate::command_interpreter::bvp_dialogue::{
    run_bvp_damp_template_dialogue, run_bvp_sci_template_dialogue,
    write_bvp_sci_task_template, write_bvp_task_template,
};
use crate::command_interpreter::docx_parser::{process_docx, process_docx_check};
use crate::command_interpreter::ivp_dialogue::{run_ivp_template_dialogue, write_ivp_task_template};
use crate::command_interpreter::task_runner::{
    anchor_task_paths, parse_task_spec_from_file, render_task_check, render_task_preview,
    requested_output_paths, run_task_batch, run_task_from_spec, ParsedTaskSpec, TaskDocumentKind,
    TaskRunResult, TaskRunnerError,
};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use dialoguer::{theme::ColorfulTheme, Select};
use std::ffi::OsString;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;

const EXIT_TASK_FAILURE: i32 = 3;
const EXIT_BATCH_FAILURE: i32 = 4;
const EXIT_OUTPUT_FAILURE: i32 = 5;

#[derive(Debug, Parser)]
#[command(
    name = "rustedscithe",
    version,
    about = "RustedSciThe task-shell CLI",
    arg_required_else_help = true,
    after_help = "Exit codes:\n  0  success or cancelled interactive wizard\n  2  command-line usage error\n  3  parse, validation, or unsupported-option failure\n  4  batch completed with at least one failed task\n  5  task input/output or postprocessing failure\n  6  solver execution failure"
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
        #[arg(long, value_enum, default_value_t = RunFormat::Table)]
        format: RunFormat,
        #[arg(long, help = "Allow replacing requested output files/directories")]
        force: bool,
    },
    /// Print a substituted preview without running the solver.
    Check { task_file: PathBuf },
    /// Run several task documents and continue after individual failures.
    Batch {
        #[arg(value_name = "TASK_FILE", num_args = 1..)]
        task_files: Vec<PathBuf>,
        #[arg(long, value_name = "REPORT_PATH")]
        report: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = BatchFormat::Table)]
        format: BatchFormat,
        #[arg(long, help = "Allow replacing an existing report file")]
        force: bool,
    },
    /// Convert DOCX to a task document.
    Convert { docx_file: PathBuf },
    /// Convert DOCX and print a task check preview.
    #[command(name = "convert-check", alias = "convert_check")]
    ConvertCheck { docx_file: PathBuf },
    /// Generate a parser-ready task template.
    Template {
        #[arg(
            value_name = "KIND",
            help = "Template family: ivp, bvp, or bvp-sci (omit to choose interactively)"
        )]
        kind: Option<TemplateKind>,
        output_path: Option<PathBuf>,
    },
    /// Open the interactive task wizard.
    Wizard,
    /// Probe external toolchain/runtime dependencies used by the project.
    #[command(name = "check-ffi-dependencies")]
    CheckFfiDependencies {
        #[arg(long, value_name = "TASK_FILE")]
        task: Option<PathBuf>,
    },
    /// Print a curated command showcase.
    Showcase,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TemplateKind {
    Ivp,
    Bvp,
    #[value(name = "bvp-sci")]
    BvpSci,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BatchFormat {
    Table,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RunFormat {
    Table,
    Json,
}

fn normalize_legacy_cli_args(args: Vec<OsString>) -> Vec<OsString> {
    if args.len() <= 1 {
        return args;
    }

    let first = args[1].to_string_lossy();
    match first.as_ref() {
        "--help" | "-h" => args,
        "help" => {
            let mut normalized = vec![args[0].clone()];
            if args.len() > 2 {
                normalized.extend(args.into_iter().skip(2));
                normalized.push(OsString::from("--help"));
            } else {
                normalized.push(OsString::from("--help"));
            }
            normalized
        }
        "run"
        | "check"
        | "batch"
        | "convert"
        | "convert-check"
        | "template"
        | "wizard"
        | "showcase"
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

fn choose_template_kind() -> Option<TemplateKind> {
    let choices = ["IVP", "BVP_Damp (damped/frozen)", "BVP_sci (SciPy-like)"];
    match Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Choose a task-template family (Esc cancels without writing)")
        .items(choices)
        .default(0)
        .interact_opt()
    {
        Ok(Some(0)) => Some(TemplateKind::Ivp),
        Ok(Some(1)) => Some(TemplateKind::Bvp),
        Ok(Some(_)) => Some(TemplateKind::BvpSci),
        Ok(None) => None,
        Err(error) => {
            eprintln!("Template selection cancelled: {error}");
            None
        }
    }
}

fn run_interactive_wizard(output_path: Option<PathBuf>) -> std::io::Result<()> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        println!("Interactive wizard requires a terminal; no file was written.");
        println!(
            "Use `template <ivp|bvp|bvp-sci> [output-path]` for scripted generation."
        );
        return Ok(());
    }
    let Some(kind) = choose_template_kind() else {
        println!("Wizard cancelled; no file was written.");
        return Ok(());
    };
    match kind {
        TemplateKind::Ivp => run_ivp_template_dialogue(output_path),
        TemplateKind::Bvp => run_bvp_damp_template_dialogue(output_path),
        TemplateKind::BvpSci => run_bvp_sci_template_dialogue(output_path),
    }
}

fn report_wizard_error(error: std::io::Error) -> i32 {
    eprintln!("Failed to write task template: {error}");
    EXIT_OUTPUT_FAILURE
}

fn report_task_error(context: &str, error: &TaskRunnerError) -> i32 {
    eprintln!("{context}: [{}] {error}", error.category());
    let mut cause = std::error::Error::source(error);
    while let Some(source) = cause {
        eprintln!("Cause: {source}");
        cause = source.source();
    }
    eprintln!("Remedy: {}", remedy_for_task_category(error.category()));
    error.exit_code()
}

fn remedy_for_task_category(category: &str) -> &'static str {
    match category {
        "io" => "check the task path and output permissions",
        "configuration" | "parse" => {
            "run `check <task-file>` and correct the reported section or field"
        }
        "unsupported_option" | "unsupported_route" => {
            "choose one of the routes listed by `check <task-file>`"
        }
        "continuation" => "verify continuation values, lifecycle mode, and restart policy",
        "postprocess" => "check output paths and requested postprocessing tools",
        "solver"
        | "solver_result"
        | "solver_exhaustion"
        | "solver_singular"
        | "solver_tolerance" => {
            "inspect tolerances, solver limits, and the native solver diagnostics"
        }
        "solver_linear" => "inspect matrix layout, backend availability, and Jacobian conditioning",
        "callback" => "check callback shapes, finite values, and symbolic bindings",
        "aot_compiler" | "aot_link" => {
            "check the selected compiler/linker and rerun check-ffi-dependencies"
        }
        "aot_timeout" => "increase the AOT timeout or inspect the compiler process log",
        "aot_artifact" | "aot_publication" | "aot_handoff" => {
            "check AOT output/provenance paths and lifecycle policy"
        }
        "aot_runtime" => "check the generated ABI and runtime library availability",
        _ => "inspect the task diagnostics and executable help",
    }
}

fn print_showcase() {
    println!("RustedSciThe showcase (minimal crate storefront)");
    println!();
    println!("Task-shell commands:");
    println!("  rustedscithe run <task-file> [--format table|json] [--force]");
    println!("  rustedscithe check <task-file>");
    println!("  rustedscithe batch <task-file>... [--format table|json] [--report summary.md] [--force]");
    println!("  rustedscithe convert <docx-file>");
    println!("  rustedscithe convert-check <docx-file>");
    println!("  rustedscithe template (interactive family picker)");
    println!("  rustedscithe template <ivp|bvp|bvp-sci> [output-file]");
    println!("  rustedscithe wizard (interactive, editable task configuration)");
    println!("  rustedscithe help");
    println!("  rustedscithe check-ffi-dependencies [--task task-file]");
    println!("  rustedscithe showcase");
    println!();
    println!("7) Installed executable (copy task files beside the binary):");
    println!("  RustedSciThe.exe check tasks/ivp_rk45_single.txt");
    println!("  RustedSciThe.exe run tasks/ivp_rk45_single.txt --format table");
    println!("  RustedSciThe.exe run tasks/bvp_sci_task.txt --format json");
    println!("  RustedSciThe.exe batch tasks/*.txt --format json --report reports/night.json");
    println!();
    println!("1) Task-shell quick start (text task -> solver run):");
    println!("  rustedscithe run examples/task_docs/ivp_decay_task.txt");
    println!("  rustedscithe run examples/task_docs/bvp_reference_task.txt");
    println!(
        "  rustedscithe run examples/task_docs/ivp_solid_combustion_with_sublimation_task.txt"
    );
    println!("  rustedscithe run examples/task_docs/bvp_combustion_1000_task.txt");
    println!("  rustedscithe template ivp");
    println!("  rustedscithe template bvp");
    println!("  rustedscithe template bvp-sci examples/task_docs/bvp_sci_task_template.txt");
    println!("  rustedscithe template ivp examples/task_docs/ivp_task_template.txt");
    println!("  rustedscithe template bvp examples/task_docs/bvp_task_template.txt");
    println!();
    println!("2) Dispatcher / shell infrastructure:");
    println!("  cargo run --example task_dispatcher_guide");
    println!("  cargo run --example task_shell_guides_index");
    println!("  rustedscithe check-ffi-dependencies [--task task-file]");
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
    println!("  cargo run --example nonlinear_systems_modern_guide");
}

fn print_run_summary(result: &TaskRunResult) {
    match result {
        TaskRunResult::Ivp(ivp) => {
            let status = ivp.status.as_deref().unwrap_or("unknown");
            let rows = ivp.t_result.as_ref().map(|t| t.len()).unwrap_or(0);
            let cols = ivp.y_result.as_ref().map(|y| y.ncols()).unwrap_or(0);
            println!("Task kind : IVP");
            println!("Status    : {status}");
            if let Some(status_code) = &ivp.status_code {
                println!("Status code: {status_code:?}");
            }
            println!("Grid rows : {rows}");
            println!("State dim : {cols}");
            if let Some(continuation) = &ivp.continuation {
                let prepare_ms = continuation
                    .segments
                    .iter()
                    .map(|segment| segment.prepare_ms)
                    .sum::<f64>();
                let solve_ms = continuation
                    .segments
                    .iter()
                    .map(|segment| segment.solve_ms)
                    .sum::<f64>();
                println!(
                    "Continuation: mode={:?}, segments={}, preparations={}, prepared_reuses={}, prepare_ms={:.3}, solve_ms={:.3}",
                    continuation.mode,
                    continuation.segments.len(),
                    continuation.fresh_preparations,
                    continuation.prepared_reuses,
                    prepare_ms,
                    solve_ms,
                );
            } else {
                println!("Continuation: none");
            }
            print_ivp_artifact_summary(&ivp.specification.postprocessing);
        }
        TaskRunResult::Bvp(bvp) => {
            let (rows, cols) = bvp
                .result
                .as_ref()
                .map(|m| (m.nrows(), m.ncols()))
                .unwrap_or((0, 0));
            let family = match bvp.specification.solver.family {
                crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Damp => {
                    "BVP_Damp"
                }
                crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Sci => "BVP_sci",
            };
            let frontend = bvp
                .specification
                .solver
                .frontend
                .map(|value| format!("{value:?}"))
                .unwrap_or_else(|| "solver default".to_string());
            let execution = bvp
                .specification
                .solver
                .execution
                .map(|value| format!("{value:?}"))
                .unwrap_or_else(|| "solver default".to_string());
            println!("Task kind : BVP");
            println!("Solver    : {family}");
            println!("Frontend  : {frontend}");
            println!("Execution : {execution}");
            println!("Layout    : {:?}", bvp.specification.solver.backend);
            if let Some(factorization) = bvp.specification.solver_options.sparse_factorization {
                println!("Sparse factorization requested: {factorization:?}");
            }
            if let Some(telemetry) = &bvp.runtime_telemetry {
                println!("Solver telemetry: {:?}", telemetry.mode);
                if telemetry.sparse_backend_outcome
                    != crate::command_interpreter::task_parser_bvp::BvpTaskSparseBackendOutcome::NotApplicable
                {
                    println!("Sparse backend runtime: {:?}", telemetry.sparse_backend_outcome);
                }
                if let Some(fallbacks) = telemetry.superlu_fallbacks {
                    println!("SuperLU fallback count: {fallbacks}");
                }
                if let Some(factorizations) = telemetry.superlu_factorizations {
                    println!("SuperLU native factorizations: {factorizations}");
                }
                if let Some(factorizations) = telemetry.linear_factorizations {
                    println!("Linear factorizations: {factorizations}");
                }
                if let Some(solves) = telemetry.linear_solves {
                    println!("Linear solves: {solves}");
                }
            }
            println!("Status    : solution returned");
            println!("Grid rows : {rows}");
            println!("State dim : {cols}");
            println!("Segments  : {}", bvp.continuation_telemetry.segments);
            println!(
                "Prepared reuses: {}",
                bvp.continuation_telemetry.prepared_reuses
            );
            println!(
                "Mesh/guess restarts: {}/{}",
                bvp.continuation_telemetry.mesh_restarts,
                bvp.continuation_telemetry.initial_guess_restarts
            );
            print_bvp_artifact_summary(&bvp.specification.postprocessing);
        }
    }
}

fn json_quote(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32))
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

/// Stable, compact success summary for scripts which need one task per
/// process. This deliberately exposes lifecycle facts, not solver traces or
/// the potentially large trajectory itself.
fn render_run_summary_json(result: &TaskRunResult) -> String {
    match result {
        TaskRunResult::Ivp(ivp) => {
            let status = ivp.status.as_deref().unwrap_or("unknown");
            let rows = ivp.t_result.as_ref().map(|t| t.len()).unwrap_or(0);
            let state_dim = ivp.y_result.as_ref().map(|y| y.ncols()).unwrap_or(0);
            let continuation = ivp.continuation.as_ref().map(|value| {
                let prepare_ms = value
                    .segments
                    .iter()
                    .map(|segment| segment.prepare_ms)
                    .sum::<f64>();
                let solve_ms = value
                    .segments
                    .iter()
                    .map(|segment| segment.solve_ms)
                    .sum::<f64>();
                format!(
                    "{{\"mode\":{},\"segments\":{},\"fresh_preparations\":{},\"prepared_reuses\":{},\"prepare_ms\":{:.6},\"solve_ms\":{:.6}}}",
                    json_quote(&format!("{:?}", value.mode)),
                    value.segments.len(),
                    value.fresh_preparations,
                    value.prepared_reuses,
                    prepare_ms,
                    solve_ms
                )
            });
            format!(
                "{{\"status\":\"passed\",\"kind\":\"IVP\",\"solver\":{},\"solver_status\":{},\"status_code\":{},\"grid_rows\":{},\"state_dim\":{},\"continuation\":{}}}\n",
                json_quote(&format!("{:?}", ivp.specification.solver.method)),
                json_quote(status),
                ivp.status_code
                    .as_ref()
                    .map(|value| json_quote(&format!("{:?}", value)))
                    .unwrap_or_else(|| "null".to_string()),
                rows,
                state_dim,
                continuation.unwrap_or_else(|| "null".to_string())
            )
        }
        TaskRunResult::Bvp(bvp) => {
            let family = match bvp.specification.solver.family {
                crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Damp => {
                    "BVP_Damp"
                }
                crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Sci => "BVP_sci",
            };
            let frontend = bvp
                .specification
                .solver
                .frontend
                .map(|value| format!("{:?}", value));
            let execution = bvp
                .specification
                .solver
                .execution
                .map(|value| format!("{:?}", value));
            let (rows, state_dim) = bvp
                .result
                .as_ref()
                .map(|matrix| (matrix.nrows(), matrix.ncols()))
                .unwrap_or((0, 0));
            let telemetry = bvp.runtime_telemetry.as_ref().map(|value| {
                format!(
                    "{{\"mode\":{},\"sparse_backend\":{},\"linear_factorizations\":{},\"linear_solves\":{}}}",
                    json_quote(&format!("{:?}", value.mode)),
                    json_quote(&format!("{:?}", value.sparse_backend_outcome)),
                    value.linear_factorizations.map(|n| n.to_string()).unwrap_or_else(|| "null".to_string()),
                    value.linear_solves.map(|n| n.to_string()).unwrap_or_else(|| "null".to_string())
                )
            });
            format!(
                "{{\"status\":\"passed\",\"kind\":\"BVP\",\"solver\":{},\"frontend\":{},\"execution\":{},\"layout\":{},\"result_rows\":{},\"state_dim\":{},\"telemetry\":{},\"continuation_segments\":{}}}\n",
                json_quote(family),
                frontend.as_deref().map(json_quote).unwrap_or_else(|| "null".to_string()),
                execution.as_deref().map(json_quote).unwrap_or_else(|| "null".to_string()),
                json_quote(&format!("{:?}", bvp.specification.solver.backend)),
                rows,
                state_dim,
                telemetry.unwrap_or_else(|| "null".to_string()),
                bvp.continuation_segments.len()
            )
        }
    }
}

fn print_machine_error(error: &TaskRunnerError) {
    println!(
        "{{\"status\":\"failed\",\"category\":{},\"exit_code\":{},\"error\":{}}}",
        json_quote(error.category()),
        error.exit_code(),
        json_quote(&error.to_string())
    );
}

fn print_ivp_artifact_summary(
    output: &crate::command_interpreter::task_parser_ivp::PostprocessingSpec,
) {
    let mut reported = false;
    for (enabled, label, path) in [
        (
            output.save_csv,
            "CSV",
            output.csv_path.as_deref().unwrap_or("ivp_result.csv"),
        ),
        (
            output.save_txt,
            "Text",
            output.txt_path.as_deref().unwrap_or("ivp_result.txt"),
        ),
        (
            output.write_report,
            "Report",
            output.report_path.as_deref().unwrap_or("ivp_report.md"),
        ),
        (
            output.plotters_png,
            "Plotters",
            output.plotters_dir.as_deref().unwrap_or("ivp_plotters"),
        ),
        (
            output.gnuplot_png,
            "Gnuplot",
            output.gnuplot_dir.as_deref().unwrap_or("ivp_gnuplot"),
        ),
    ] {
        if enabled {
            println!("Output    : {label} -> {path}");
            reported = true;
        }
    }
    if output.terminal_plot {
        println!("Output    : terminal plot");
        reported = true;
    }
    if !reported {
        println!("Outputs   : no files or plots requested");
    }
}

fn print_bvp_artifact_summary(
    output: &crate::command_interpreter::task_parser_bvp::BvpPostprocessingSpec,
) {
    let mut reported = false;
    for (enabled, label, path) in [
        (
            output.save_csv,
            "CSV",
            output.csv_path.as_deref().unwrap_or("bvp_result.csv"),
        ),
        (
            output.save_txt,
            "Text",
            output.txt_path.as_deref().unwrap_or("bvp_result.txt"),
        ),
        (
            output.write_report,
            "Report",
            output.report_path.as_deref().unwrap_or("bvp_report.md"),
        ),
    ] {
        if enabled {
            println!("Output    : {label} -> {path}");
            reported = true;
        }
    }
    match output.output_policy {
        crate::command_interpreter::task_parser_bvp::BvpOutputPolicy::None => {}
        crate::command_interpreter::task_parser_bvp::BvpOutputPolicy::Plotters
            if !output.plotters_png =>
        {
            println!("Output    : Plotters -> bvp_plotters");
            reported = true;
        }
        crate::command_interpreter::task_parser_bvp::BvpOutputPolicy::Gnuplot
            if !output.gnuplot_png =>
        {
            println!("Output    : Gnuplot -> bvp_gnuplot");
            reported = true;
        }
        crate::command_interpreter::task_parser_bvp::BvpOutputPolicy::Terminal
            if !output.terminal_plot =>
        {
            println!("Output    : terminal plot");
            reported = true;
        }
        _ => {}
    }
    for (enabled, label, path) in [
        (
            output.plotters_png,
            "Plotters",
            output.plotters_dir.as_deref().unwrap_or("bvp_plotters"),
        ),
        (
            output.gnuplot_png,
            "Gnuplot",
            output.gnuplot_dir.as_deref().unwrap_or("bvp_gnuplot"),
        ),
    ] {
        if enabled {
            println!("Output    : {label} -> {path}");
            reported = true;
        }
    }
    if !reported
        && matches!(
            output.output_policy,
            crate::command_interpreter::task_parser_bvp::BvpOutputPolicy::None
        )
    {
        println!("Outputs   : no files or plots requested");
    }
}

fn command_available(command: &str, probe_arg: &str) -> Option<String> {
    let output = Command::new(command).arg(probe_arg).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    stdout
        .lines()
        .chain(stderr.lines())
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_owned)
        .or_else(|| Some("(no version text)".to_string()))
}

fn print_dependency_row(name: &str, status: &str, detail: &str) {
    println!("{name:<14} | {status:<8} | {detail}");
}

fn superlu_library_configuration(library_path: Option<&Path>) -> (&'static str, String) {
    let Some(path) = library_path else {
        return (
            "missing",
            "set SUPERLU_LIBRARY to the platform bridge shared-library path".into(),
        );
    };
    if !path.is_file() {
        return (
            "invalid",
            format!("SUPERLU_LIBRARY is not a file: {}", path.display()),
        );
    }
    (
        "configured",
        format!("bridge candidate: {}", path.display()),
    )
}

fn superlu_probe() -> (&'static str, String) {
    let configured = superlu_library_configuration(
        std::env::var_os("SUPERLU_LIBRARY")
            .as_deref()
            .map(Path::new),
    );

    #[cfg(feature = "superlu")]
    if configured.0 == "configured" {
        let Some(path) = std::env::var_os("SUPERLU_LIBRARY") else {
            return configured;
        };
        return match crate::somelinalg::superlu::SolverLibrary::load(&path) {
            Ok(library) => {
                let version = library.version();
                (
                    "ok",
                    format!(
                        "bridge loaded; SuperLU {}.{}.{}; path: {}",
                        version.0,
                        version.1,
                        version.2,
                        Path::new(&path).display()
                    ),
                )
            }
            Err(error) => ("error", format!("bridge load/ABI check failed: {error}")),
        };
    }

    #[cfg(not(feature = "superlu"))]
    if configured.0 == "configured" {
        return (
            "configured",
            format!(
                "file exists but crate lacks `superlu`; rebuild with `--features superlu` to load-check it: {}",
                std::env::var_os("SUPERLU_LIBRARY")
                    .as_deref()
                    .map(Path::new)
                    .expect("configured path was present")
                    .display()
            ),
        );
    }

    configured
}

fn print_task_dependency_requirements(task: &Path) {
    println!("task: {}", task.display());
    match parse_task_spec_from_file(task) {
        Err(error) => {
            println!("task preflight: [{}] {error}", error.category());
            println!("task requirements: unavailable until the document parses");
        }
        Ok(ParsedTaskSpec::Ivp(spec)) => {
            println!("task preflight: valid; family=IVP");
            if let Some(lsode2) = &spec.solver_options.lsode2 {
                let execution = format!("{:?}", lsode2.symbolic_execution);
                if execution.contains("Aot") {
                    println!("task requirement: AOT toolchain ({execution})");
                }
            }
        }
        Ok(ParsedTaskSpec::Bvp(spec)) => {
            println!("task preflight: valid; family=BVP");
            let execution = format!("{:?}", spec.solver.execution);
            if execution.contains("Aot") {
                println!(
                    "task requirement: AOT toolchain; codegen={}",
                    spec.solver_options
                        .generated_backend
                        .aot_codegen_backend
                        .as_deref()
                        .unwrap_or("default")
                );
            }
            if format!("{:?}", spec.solver.backend) == "Sparse"
                && format!("{:?}", spec.solver_options.sparse_factorization).contains("SuperLu")
            {
                println!(
                    "task requirement: SuperLU Cargo feature plus SUPERLU_LIBRARY; runtime fallback is StandardFaer"
                );
            }
        }
    }
}

fn print_ffi_dependency_report(task: Option<&Path>) {
    println!("RustedSciThe external dependency report");
    println!("platform: {}", std::env::consts::OS);
    println!("probe strategy: PATH process checks plus configured native-library load/ABI checks");
    if let Some(task) = task {
        print_task_dependency_requirements(task);
    }
    println!("name           | status   | detail");
    println!("----------------------------------------------");

    let rustc = command_available("rustc", "--version");
    let cargo = command_available("cargo", "--version");
    let tcc = command_available("tcc", "-v");
    let gcc = command_available("gcc", "--version");
    let zig = command_available("zig", "version");
    let gnuplot = command_available("gnuplot", "--version");
    let pandoc = command_available("pandoc", "--version");
    let (superlu_status, superlu_detail) = superlu_probe();

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
    print_dependency_row(
        "SuperLU feature",
        if cfg!(feature = "superlu") {
            "enabled"
        } else {
            "disabled"
        },
        if cfg!(feature = "superlu") {
            "compiled with `--features superlu`"
        } else {
            "rebuild with `--features superlu` to enable bridge loading"
        },
    );
    print_dependency_row("SuperLU bridge", superlu_status, &superlu_detail);

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
        print_dependency_row(
            "arrayfire",
            "disabled",
            "crate compiled without `arrayfire` feature",
        );
    }
}

pub fn run() -> Result<(), i32> {
    let raw_args: Vec<OsString> = std::env::args_os().collect();
    if raw_args.len() == 1 {
        let mut command = Cli::command();
        let _ = command.print_help();
        println!();
        return Ok(());
    }
    let normalized_args = normalize_legacy_cli_args(raw_args);
    let run_json = normalized_args.windows(2).any(|window| {
        window[0].to_string_lossy() == "--format"
            && window[1].to_string_lossy().eq_ignore_ascii_case("json")
    });
    let cli = Cli::parse_from(normalized_args);

    match cli.command {
        Commands::Run {
            task_file,
            format,
            force,
        } => match parse_task_spec_from_file(&task_file) {
            Ok(mut spec) => {
                anchor_task_paths(&mut spec, &task_file);
                let existing_outputs = requested_output_paths(&spec)
                    .into_iter()
                    .filter(|path| path.exists())
                    .collect::<Vec<_>>();
                if !existing_outputs.is_empty() && !force {
                    let paths = existing_outputs
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    if matches!(format, RunFormat::Json) {
                        println!(
                            "{{\"status\":\"failed\",\"category\":\"postprocess\",\"exit_code\":{},\"error\":{}}}",
                            EXIT_OUTPUT_FAILURE,
                            json_quote(&format!("refusing to overwrite task outputs: {paths}"))
                        );
                    }
                    eprintln!(
                        "Refusing to overwrite task outputs: {paths}; use `run --force <task-file>` to replace them"
                    );
                    return Err(EXIT_OUTPUT_FAILURE);
                }
                let kind = match spec.kind() {
                    TaskDocumentKind::Ivp => "IVP",
                    TaskDocumentKind::Bvp => "BVP",
                };
                if matches!(format, RunFormat::Table) {
                    println!("Task file : {}", task_file.display());
                    println!("Detected  : {kind}");
                    println!();
                    println!("{}", render_task_preview(&spec));
                }
                match run_task_from_spec(spec) {
                    Ok(result) => {
                        match format {
                            RunFormat::Table => print_run_summary(&result),
                            RunFormat::Json => print!("{}", render_run_summary_json(&result)),
                        }
                        Ok(())
                    }
                    Err(err) => {
                        if matches!(format, RunFormat::Json) {
                            print_machine_error(&err);
                        }
                        Err(report_task_error(
                            &format!("Task execution failed: `{}`", task_file.display()),
                            &err,
                        ))
                    }
                }
            }
            Err(err) => {
                if run_json {
                    print_machine_error(&err);
                }
                Err(report_task_error(
                    &format!("Task preflight failed: `{}`", task_file.display()),
                    &err,
                ))
            }
        },
        Commands::Batch {
            task_files,
            report,
            format,
            force,
        } => {
            let batch = run_task_batch(task_files);
            println!(
                "[Batch] passed={} failed={}",
                batch.passed_count(),
                batch.failed_count()
            );
            match format {
                BatchFormat::Table => println!("{}", batch.render_table()),
                BatchFormat::Json => print!("{}", batch.render_json()),
            }
            if let Some(report_path) = report {
                if report_path.exists() && !force {
                    eprintln!(
                        "Refusing to overwrite existing batch report `{}`; use --force to replace it",
                        report_path.display()
                    );
                    return Err(EXIT_OUTPUT_FAILURE);
                }
                let write_result = match format {
                    BatchFormat::Table => batch.write_table(&report_path),
                    BatchFormat::Json => batch.write_json(&report_path),
                };
                if let Err(error) = write_result {
                    eprintln!(
                        "Failed to write batch report `{}`: {error}",
                        report_path.display()
                    );
                    return Err(EXIT_OUTPUT_FAILURE);
                }
                println!("Batch report: {}", report_path.display());
            }
            if batch.failed_count() > 0 {
                Err(EXIT_BATCH_FAILURE)
            } else {
                Ok(())
            }
        }
        Commands::Check { task_file } => match parse_task_spec_from_file(&task_file) {
            Ok(mut spec) => {
                anchor_task_paths(&mut spec, &task_file);
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
            Err(err) => Err(report_task_error("Task preflight failed", &err)),
        },
        Commands::Convert { docx_file } => match process_docx(docx_file.to_string_lossy().as_ref())
        {
            Ok(text) => {
                println!("{text}");
                Ok(())
            }
            Err(err) => {
                eprintln!("{err}");
                Err(EXIT_TASK_FAILURE)
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
                    Err(EXIT_TASK_FAILURE)
                }
            }
        }
        Commands::Template { kind, output_path } => {
            if let Some(kind) = kind {
                match kind {
                    TemplateKind::Ivp => {
                        let path = write_ivp_task_template(output_path).map_err(report_wizard_error)?;
                        println!("IVP task template saved to {}", path.display());
                    }
                    TemplateKind::Bvp => {
                        let path = write_bvp_task_template(output_path).map_err(report_wizard_error)?;
                        println!("BVP task template saved to {}", path.display());
                    }
                    TemplateKind::BvpSci => {
                        let path =
                            write_bvp_sci_task_template(output_path).map_err(report_wizard_error)?;
                        println!("BVP_sci task template saved to {}", path.display());
                    }
                }
            } else {
                run_interactive_wizard(output_path).map_err(report_wizard_error)?;
            }
            Ok(())
        }
        Commands::Wizard => {
            run_interactive_wizard(None).map_err(report_wizard_error)?;
            Ok(())
        }
        Commands::CheckFfiDependencies { task } => {
            print_ffi_dependency_report(task.as_deref());
            Ok(())
        }
        Commands::Showcase => {
            print_showcase();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{superlu_library_configuration, Cli, Commands, TemplateKind};
    use clap::Parser;
    use std::path::Path;

    #[test]
    fn cli_exposes_noninteractive_bvp_sci_template_command() {
        let cli = Cli::try_parse_from([
            "rustedscithe",
            "template",
            "bvp-sci",
            "examples/task_docs/bvp_sci_task_template.txt",
        ])
        .expect("bvp-sci template command should parse");
        assert!(matches!(
            cli.command,
            Commands::Template {
                kind: Some(TemplateKind::BvpSci),
                output_path: Some(_)
            }
        ));
    }

    #[test]
    fn template_command_without_kind_enters_interactive_picker() {
        let cli = Cli::try_parse_from(["rustedscithe", "template"])
            .expect("template kind is optional when the wizard picker is used");
        assert!(matches!(
            cli.command,
            Commands::Template {
                kind: None,
                output_path: None
            }
        ));
    }

    #[test]
    fn cli_exposes_explicit_interactive_wizard_command() {
        let cli =
            Cli::try_parse_from(["rustedscithe", "wizard"]).expect("wizard command should parse");
        assert!(matches!(cli.command, Commands::Wizard));
    }

    #[test]
    fn cli_exposes_batch_command_with_multiple_tasks_and_report_path() {
        let cli = Cli::try_parse_from([
            "rustedscithe",
            "batch",
            "first.txt",
            "second.txt",
            "--report",
            "summary.md",
        ])
        .expect("batch command should parse");
        assert!(matches!(cli.command, Commands::Batch { .. }));
    }

    #[test]
    fn superlu_preflight_distinguishes_feature_and_bridge_path() {
        assert_eq!(superlu_library_configuration(None).0, "missing");
        assert_eq!(
            superlu_library_configuration(Some(Path::new("missing-superlu-bridge"))).0,
            "invalid"
        );
    }
}
