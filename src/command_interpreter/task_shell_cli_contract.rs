//! Process-level contracts for the console task shell.
//!
//! These tests live under `src/` so they follow the repository's policy of
//! keeping maintained code outside `src/` limited to examples, scripts and
//! benchmarks. Cargo declares this file as a repository-only integration test.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::tempdir;

fn cli_binary() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_RustedSciThe")
        .or_else(|| std::env::var_os("CARGO_BIN_EXE_rustedscithe"))
        .map(PathBuf::from)
        .expect("Cargo should provide the task-shell executable path")
}

fn run_cli(args: &[&str]) -> Output {
    Command::new(cli_binary())
        .args(args)
        .output()
        .expect("task-shell process should start")
}

fn run_cli_without_superlu_library(args: &[&str]) -> Output {
    Command::new(cli_binary())
        .args(args)
        .env_remove("SUPERLU_LIBRARY")
        .output()
        .expect("task-shell process should start")
}

fn reference_task(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/task_docs/reference")
        .join(name)
}

fn write_temp_task(contents: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("temporary task directory should be created");
    let path = dir.path().join("task.txt");
    fs::write(&path, contents).expect("temporary task should be written");
    (dir, path)
}

#[test]
fn cli_run_executes_ivp_and_reports_a_solution() {
    let task = reference_task("ivp_rk45_single.txt");
    let output = run_cli(&["run", task.to_string_lossy().as_ref()]);

    assert!(
        output.status.success(),
        "IVP run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Task kind : IVP"));
    assert!(stdout.contains("Grid rows :"));
    assert!(stdout.contains("State dim : 1"));
    assert!(stdout.contains("Continuation: none"));
    assert!(stdout.contains("Outputs   : no files or plots requested"));
}

#[test]
fn cli_run_json_emits_one_compact_machine_summary() {
    let task = reference_task("ivp_rk45_single.txt");
    let output = run_cli(&["run", "--format", "json", task.to_string_lossy().as_ref()]);

    assert!(
        output.status.success(),
        "JSON IVP run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("{\"status\":\"passed\""));
    assert!(stdout.contains("\"kind\":\"IVP\""));
    assert!(stdout.contains("\"grid_rows\""));
    assert!(!stdout.contains("Task kind :"));
}

#[test]
fn cli_check_reports_bvp_sci_choices_without_claiming_convergence() {
    let task = reference_task("bvp_sci_contract.txt");
    let output = run_cli(&["check", task.to_string_lossy().as_ref()]);

    assert!(
        output.status.success(),
        "BVP_sci preflight failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("solver_family=BVP_sci"));
    assert!(stdout.contains("frontends=ExprLegacy/AtomViewNative"));
    assert!(stdout.contains("executions=Lambdify/AOT"));
    assert!(stdout.contains("continuation: parameters=rate, segments=2"));
    assert!(stdout.contains("outputs: none"));
    assert!(stdout.contains("preflight=valid; numerical convergence has not been evaluated"));
}

#[test]
fn cli_run_executes_bvp_sci_and_reports_csv_artifact() {
    let (dir, task) = write_temp_task("");
    let csv_path = dir.path().join("solution.csv");
    let task_contents = format!(
        "task\nsolver: BVP_sci\nfrontend: ExprLegacy\nexecution: Lambdify\nmethod: Dense\n\nequations\narg: x\nunknowns: y\nrhs: -y\n\nboundary_conditions\ny_left: 1.0\n\nmesh\nt0: 0.0\nt_end: 1.0\nn_steps: 8\n\ninitial_guess\ny: 0.5\n\nsolver_options\ntolerance: 1e-3\nmax_nodes: 128\n\npostprocessing\nsave_csv: true\ncsv_path: {}\n",
        csv_path.display()
    );
    fs::write(&task, task_contents).expect("BVP_sci task should be updated with artifact path");

    let output = run_cli(&["run", task.to_string_lossy().as_ref()]);
    assert!(
        output.status.success(),
        "BVP_sci run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Solver    : BVP_sci"));
    assert!(stdout.contains("Frontend  : ExprLegacy"));
    assert!(stdout.contains("Layout    : Dense"));
    assert!(stdout.contains("Status    : solution returned"));
    assert!(stdout.contains("Solver telemetry: Off"));
    assert!(!stdout.contains("Linear factorizations:"));
    assert!(stdout.contains("Output    : CSV ->"));
    assert!(stdout.contains("Mesh/guess restarts:"));
    assert!(csv_path.is_file(), "requested CSV artifact should exist");
}

#[test]
fn cli_run_anchors_relative_artifacts_to_task_directory() {
    let (dir, task) = write_temp_task(
        "task\nsolver: BVP_sci\nfrontend: ExprLegacy\nexecution: Lambdify\nmethod: Dense\n\nequations\narg: x\nunknowns: y\nrhs: -y\n\nboundary_conditions\ny_left: 1.0\n\nmesh\nt0: 0.0\nt_end: 1.0\nn_steps: 8\n\ninitial_guess\ny: 0.5\n\nsolver_options\ntolerance: 1e-3\nmax_nodes: 128\n\npostprocessing\nsave_csv: true\ncsv_path: relative_solution.csv\n",
    );

    let output = run_cli(&["run", task.to_string_lossy().as_ref()]);
    assert!(
        output.status.success(),
        "relative-artifact BVP run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(dir.path().join("relative_solution.csv").is_file());
}

#[test]
fn cli_run_refuses_existing_outputs_without_force() {
    let (dir, task) = write_temp_task(
        "task\nsolver: IVP\nmethod: RK45\n\nequations\narg: t\nunknowns: y\nrhs: -y\n\ninitial_conditions\nt0: 0.0\nt_end: 0.1\ny0: 1.0\n\npostprocessing\nsave_txt: true\ntxt_path: result.txt\n",
    );
    fs::write(dir.path().join("result.txt"), "old result\n")
        .expect("existing task output should be created");

    let output = run_cli(&["run", task.to_string_lossy().as_ref()]);
    assert_eq!(output.status.code(), Some(5));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--force"));
    assert_eq!(
        fs::read_to_string(dir.path().join("result.txt")).expect("existing output should remain"),
        "old result\n"
    );

    let forced = run_cli(&["run", "--force", task.to_string_lossy().as_ref()]);
    assert!(
        forced.status.success(),
        "--force should allow replacement: {}",
        String::from_utf8_lossy(&forced.stderr)
    );
    assert_ne!(
        fs::read_to_string(dir.path().join("result.txt")).expect("forced output should exist"),
        "old result\n"
    );
}

#[test]
fn cli_run_reports_observed_superlu_fallback_separately_from_request() {
    let task_doc = "task\nsolver: BVP_sci\nfrontend: ExprLegacy\nexecution: Lambdify\nmethod: Sparse\n\nequations\narg: x\nunknowns: y\nrhs: -y\n\nboundary_conditions\ny_left: 1.0\n\nmesh\nt0: 0.0\nt_end: 1.0\nn_steps: 8\n\ninitial_guess\ny: 0.5\n\nsolver_options\ntolerance: 1e-3\nmax_nodes: 128\nsparse_factorization: superlu\ntelemetry: counters\n";
    let (_dir, task) = write_temp_task(task_doc);
    let output = run_cli_without_superlu_library(&["run", task.to_string_lossy().as_ref()]);

    assert!(
        output.status.success(),
        "BVP_sci fallback solve failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Sparse factorization requested: SuperLu"));
    assert!(stdout.contains("Sparse backend runtime: StandardFaerFallback"));
    assert!(stdout.contains("SuperLU fallback count: 1"));
    assert!(stdout.contains("SuperLU native factorizations: 0"));
    assert!(stdout.contains("Linear factorizations: "));
}

#[test]
fn cli_invalid_task_returns_stable_preflight_failure_code() {
    let (_dir, task) = write_temp_task("task\nsolver: IVP\nmethod: not-a-solver\n");
    let output = run_cli(&["run", task.to_string_lossy().as_ref()]);

    assert_eq!(output.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Task preflight failed:"));
}

#[test]
fn cli_invalid_symbol_keeps_source_location_and_actionable_category() {
    let (_dir, task) = write_temp_task(
        "task\nsolver: IVP\nmethod: RK45\n\nequations\narg: t\nunknowns: y\nrhs: -y + missing_gain\n\ninitial_conditions\nt0: 0.0\nt_end: 1.0\ny0: 1.0\n",
    );
    let output = run_cli(&["check", task.to_string_lossy().as_ref()]);

    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("[configuration]"));
    assert!(stderr.contains("missing_gain"));
    assert!(stderr.contains("line"));
    assert!(stderr.contains("column"));
    assert!(stderr.contains("Remedy:"));
}

#[test]
fn cli_help_and_showcase_list_the_supported_commands() {
    let help = run_cli(&["--help"]);
    assert!(
        help.status.success(),
        "CLI help failed: {}",
        String::from_utf8_lossy(&help.stderr)
    );
    let help_text = String::from_utf8_lossy(&help.stdout);
    for command in [
        "run",
        "check",
        "batch",
        "convert",
        "convert-check",
        "template",
        "wizard",
        "check-ffi-dependencies",
        "showcase",
    ] {
        assert!(
            help_text.contains(command),
            "help omitted {command}: {help_text}"
        );
    }
    assert!(help_text.contains("Exit codes:"));
    assert!(help_text.contains("batch completed with at least one failed task"));

    let showcase = run_cli(&["showcase"]);
    assert!(showcase.status.success());
    let showcase_text = String::from_utf8_lossy(&showcase.stdout);
    for command in [
        "run <task-file>",
        "check <task-file>",
        "batch <task-file>... [--format table|json] [--report summary.md]",
        "convert <docx-file>",
        "convert-check <docx-file>",
        "template (interactive family picker)",
        "template <ivp|bvp|bvp-sci>",
        "wizard (interactive, editable task configuration)",
        "help",
        "check-ffi-dependencies [--task task-file]",
        "showcase",
    ] {
        assert!(
            showcase_text.contains(command),
            "showcase omitted {command}: {showcase_text}"
        );
    }
}

#[test]
fn cli_batch_continues_after_failure_and_writes_compact_report() {
    let good = reference_task("ivp_rk45_single.txt");
    let dir = tempdir().expect("batch report directory should be created");
    let missing = dir.path().join("missing-task.txt");
    let report = dir.path().join("batch_summary.md");
    let good_arg = good.to_string_lossy().into_owned();
    let missing_arg = missing.to_string_lossy().into_owned();
    let report_arg = report.to_string_lossy().into_owned();

    let output = run_cli(&["batch", &good_arg, &missing_arg, "--report", &report_arg]);

    assert_eq!(output.status.code(), Some(4));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[Batch] passed=1 failed=1"));
    assert!(stdout.contains("error_category"));
    assert!(
        report.is_file(),
        "batch report should be written after failure"
    );
    let report_text = fs::read_to_string(report).expect("batch report should be readable");
    assert!(report_text.contains("passed"));
    assert!(report_text.contains("failed"));
}

#[test]
fn cli_batch_json_is_compact_and_writes_machine_report() {
    let good = reference_task("ivp_rk45_single.txt");
    let dir = tempdir().expect("batch report directory should be created");
    let report = dir.path().join("batch_summary.json");
    let good_arg = good.to_string_lossy().into_owned();
    let report_arg = report.to_string_lossy().into_owned();

    let output = run_cli(&[
        "batch",
        &good_arg,
        "--format",
        "json",
        "--report",
        &report_arg,
    ]);

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"passed\": 1"));
    assert!(stdout.contains("\"entries\""));
    let report_text = fs::read_to_string(report).expect("JSON report should be readable");
    assert!(report_text.contains("\"status\":\"passed\""));
    assert!(report_text.contains("\\\\"));
}

#[test]
fn cli_uses_the_same_task_through_check_run_and_batch() {
    let task = reference_task("ivp_rk45_single.txt");
    let task_arg = task.to_string_lossy().into_owned();
    let check = run_cli(&["check", &task_arg]);
    assert!(check.status.success());
    assert!(String::from_utf8_lossy(&check.stdout).contains("preflight=valid"));

    let run = run_cli(&["run", &task_arg]);
    assert!(run.status.success());
    assert!(String::from_utf8_lossy(&run.stdout).contains("Task kind : IVP"));

    let dir = tempdir().expect("batch report directory should be created");
    let report = dir.path().join("same-task.md");
    let report_arg = report.to_string_lossy().into_owned();
    let batch = run_cli(&["batch", &task_arg, "--report", &report_arg]);
    assert_eq!(batch.status.code(), Some(0));
    assert!(report.is_file());
    assert!(fs::read_to_string(report)
        .expect("same-task batch report should be readable")
        .contains("passed"));
}

#[test]
fn cli_ffi_preflight_reports_superlu_feature_and_bridge_independently() {
    let output = run_cli_without_superlu_library(&["check-ffi-dependencies"]);
    assert!(
        output.status.success(),
        "dependency preflight failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("SuperLU feature"));
    assert!(stdout.contains("SuperLU bridge"));
    assert!(stdout.contains("SUPERLU_LIBRARY"));
    assert!(stdout.contains("--features superlu"));
}

#[test]
fn cli_ffi_preflight_reports_requirements_for_selected_task() {
    let task_doc = "task\nsolver: BVP_sci\nfrontend: ExprLegacy\nexecution: Lambdify\nmethod: Sparse\n\nequations\narg: x\nunknowns: y\nrhs: -y\n\nboundary_conditions\ny_left: 1.0\n\nmesh\nt0: 0.0\nt_end: 1.0\nn_steps: 8\n\ninitial_guess\ny: 0.5\n\nsolver_options\nsparse_factorization: superlu\n";
    let (_dir, task) = write_temp_task(task_doc);
    let output = run_cli(&["check-ffi-dependencies", "--task", task.to_string_lossy().as_ref()]);

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("task preflight: valid; family=BVP"));
    assert!(stdout.contains("task requirement: SuperLU Cargo feature"));
}

#[test]
fn cli_help_alias_and_empty_invocation_show_help_successfully() {
    for args in [&["help"][..], &[][..]] {
        let output = run_cli(args);
        assert!(
            output.status.success(),
            "help invocation {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Usage:"),
            "help output was missing: {stdout}"
        );
        assert!(stdout.contains("template"));
    }

    let template_help = run_cli(&["template", "--help"]);
    assert!(template_help.status.success());
    let stdout = String::from_utf8_lossy(&template_help.stdout);
    assert!(stdout.contains("ivp"));
    assert!(stdout.contains("bvp"));
    assert!(stdout.contains("bvp-sci"));
    assert!(stdout.contains("omit to choose interactively"));
}

#[test]
fn template_without_kind_cancels_cleanly_when_no_terminal_is_available() {
    let output = run_cli(&["template"]);
    assert!(
        output.status.success(),
        "non-interactive template invocation should not panic or fail: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("no file was written"));
}

#[test]
fn cli_legacy_convert_check_alias_reaches_the_converter() {
    let dir = tempdir().expect("temporary directory should be created");
    let missing_docx = dir.path().join("missing.docx");
    let output = run_cli(&["convert_check", missing_docx.to_string_lossy().as_ref()]);

    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Usage:"),
        "alias was not dispatched: {stderr}"
    );
}

#[test]
fn cli_creates_and_preflights_noninteractive_bvp_sci_template() {
    let dir = tempdir().expect("temporary directory should be created");
    let task = dir.path().join("bvp_sci_template.txt");
    let created = run_cli(&["template", "bvp-sci", task.to_string_lossy().as_ref()]);
    assert!(
        created.status.success(),
        "template creation failed: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    assert!(task.is_file(), "CLI should publish the requested template");

    let checked = run_cli(&["check", task.to_string_lossy().as_ref()]);
    assert!(
        checked.status.success(),
        "generated template failed preflight: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let report = String::from_utf8_lossy(&checked.stdout);
    assert!(report.contains("solver_family=BVP_sci"));
    assert!(report.contains("preflight=valid; numerical convergence has not been evaluated"));
}

#[test]
fn cli_explicit_template_kinds_write_without_opening_a_wizard() {
    let dir = tempdir().expect("template directory should be created");
    for (kind, file_name, marker) in [
        ("ivp", "ivp.txt", "solver: IVP"),
        ("bvp", "bvp.txt", "solver: BVP"),
    ] {
        let path = dir.path().join(file_name);
        let path_arg = path.to_string_lossy().into_owned();
        let output = run_cli(&["template", kind, &path_arg]);
        assert!(
            output.status.success(),
            "template {kind} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = fs::read_to_string(&path).expect("deterministic template should be written");
        assert!(text.contains(marker));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Choose"));
    }
}
