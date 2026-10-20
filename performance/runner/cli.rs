use crate::baseline;
use crate::comparison::{self, Verdict};
use crate::environment;
use crate::error::RunnerError;
use crate::measurement::{self, CounterStats, MeasurementSeries};
use crate::registry::{self, Scenario, Selector};
use crate::report;
use clap::{Args, Parser, Subcommand};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_RUNS: usize = 15;
const DEFAULT_MAX_RELATIVE_MAD: f64 = 0.10;
const DEFAULT_TOLERANCE: f64 = 0.10;
const CHILD_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Parser)]
#[command(
    name = "perf-gates",
    about = "Addressable performance regression gates",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand)]
enum CommandKind {
    List {
        #[command(flatten)]
        selector: OptionalSelector,
    },
    Machine {
        #[command(subcommand)]
        command: MachineCommand,
    },
    Record(RecordArgs),
    Compare(CompareArgs),
    #[command(hide = true, name = "__measure")]
    Measure(MeasureArgs),
}

#[derive(Subcommand)]
enum MachineCommand {
    Register {
        #[arg(long)]
        name: String,
    },
}

#[derive(Args, Clone, Default)]
struct OptionalSelector {
    #[arg(long, conflicts_with_all = ["scenario", "group", "all"])]
    solver: Option<String>,
    #[arg(long, conflicts_with_all = ["solver", "group", "all"])]
    scenario: Option<String>,
    #[arg(long, conflicts_with_all = ["solver", "scenario", "all"])]
    group: Option<String>,
    #[arg(long, conflicts_with_all = ["solver", "scenario", "group"])]
    all: bool,
}

#[derive(Args)]
struct RecordArgs {
    #[command(flatten)]
    selector: OptionalSelector,
    #[arg(long)]
    machine: String,
    #[arg(long, default_value_t = DEFAULT_WARMUPS)]
    warmups: usize,
    #[arg(long, default_value_t = DEFAULT_RUNS)]
    runs: usize,
    #[arg(long, default_value_t = DEFAULT_TOLERANCE)]
    tolerance: f64,
    #[arg(long, default_value_t = DEFAULT_MAX_RELATIVE_MAD)]
    max_relative_mad: f64,
    #[arg(long)]
    force: bool,
    /// Explicitly allow recording a reference from a dirty or unversioned source tree.
    #[arg(long)]
    allow_dirty_source: bool,
}

#[derive(Args)]
struct CompareArgs {
    #[command(flatten)]
    selector: OptionalSelector,
    #[arg(long)]
    machine: String,
    #[arg(long, default_value_t = DEFAULT_WARMUPS)]
    warmups: usize,
    #[arg(long, default_value_t = DEFAULT_RUNS)]
    runs: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_RELATIVE_MAD)]
    max_relative_mad: f64,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Args)]
struct MeasureArgs {
    #[arg(long)]
    scenario: String,
    #[arg(long)]
    warmups: usize,
    #[arg(long)]
    runs: usize,
}

pub fn run() -> Result<(), RunnerError> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::List { selector } => list(selector),
        CommandKind::Machine {
            command: MachineCommand::Register { name },
        } => {
            let profile = environment::register(&name)?;
            println!(
                "registered machine {:?} with local installation ID {}",
                profile.alias, profile.installation_id
            );
            Ok(())
        }
        CommandKind::Record(args) => record(args),
        CommandKind::Compare(args) => compare(args),
        CommandKind::Measure(args) => child_measure(args),
    }
}

fn list(args: OptionalSelector) -> Result<(), RunnerError> {
    let selector = into_selector(args);
    let selected = if selector.solver.is_none()
        && selector.scenario.is_none()
        && selector.group.is_none()
        && !selector.all
    {
        registry::SCENARIOS.iter().collect()
    } else {
        registry::select(&selector)?
    };
    let machine = environment::fingerprint();
    let local = environment::find_local_profile();
    report::print_list(&selected, local.as_ref(), &machine, |scenario| {
        local
            .as_ref()
            .is_some_and(|profile| baseline::path(profile, &machine, scenario).is_file())
    });
    Ok(())
}

fn record(args: RecordArgs) -> Result<(), RunnerError> {
    require_release_measurement()?;
    validate_runs(args.warmups, args.runs, args.tolerance)?;
    validate_runs(args.warmups, args.runs, args.max_relative_mad)?;
    let selected = registry::select(&into_selector(args.selector))?;
    let machine = environment::resolve(&args.machine)?;
    let env = environment::fingerprint();
    if env.source_dirty && !args.allow_dirty_source {
        return Err(RunnerError::Configuration(format!(
            "refusing to record a baseline from dirty source revision {}; commit the source or pass --allow-dirty-source explicitly",
            env.source_revision
        )));
    }
    let run_dir = create_run_dir()?;
    let scenario_ids = selected
        .iter()
        .map(|scenario| scenario.id.to_owned())
        .collect::<Vec<_>>();
    let mut errors: Vec<(i32, String)> = Vec::new();
    let mut created = 0usize;
    for scenario in selected {
        let target = baseline::path(&machine, &env, scenario);
        match isolated_measure(scenario, args.warmups, args.runs) {
            Ok(series) => {
                if let Some(metric) = series.metrics.iter().find(|metric| {
                    series.gate_metrics.contains(&metric.id)
                        && (metric.median <= 0.0
                            || metric.mad / metric.median > args.max_relative_mad)
                }) {
                    errors.push((
                        5,
                        format!(
                            "{}: cannot record unstable metric {} (MAD/median={:.3})",
                            scenario.id,
                            metric.id,
                            metric.mad / metric.median.max(f64::MIN_POSITIVE)
                        ),
                    ));
                    continue;
                }
                let result = baseline::write(
                    &target,
                    &machine,
                    &env,
                    scenario,
                    &series,
                    args.tolerance,
                    args.force,
                );
                if let Err(error) = result {
                    errors.push((error.exit_code(), format!("{}: {error}", scenario.id)));
                    continue;
                }
                persist_run(&run_dir, scenario.id, &series, None)?;
                report::print_record(&machine, &env, scenario, &series, &target);
                created += 1;
            }
            Err(error) => errors.push((error.exit_code(), format!("{}: {error}", scenario.id))),
        }
    }
    persist_summary(&run_dir, "record", &machine, &env, &scenario_ids, &errors)?;
    if errors.is_empty() {
        println!(
            "BASELINE RECORD COMPLETE: created {created} baseline(s) for {} under {}",
            scenario_ids.join(", "),
            baseline::root().display()
        );
        println!("run report: {}", run_dir.display());
    } else {
        println!(
            "BASELINE RECORD INCOMPLETE: created {created} of {} baseline(s); {} failure(s)",
            scenario_ids.len(),
            errors.len()
        );
        println!("run report: {}", run_dir.display());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(batch_error(errors))
    }
}

fn compare(args: CompareArgs) -> Result<(), RunnerError> {
    validate_runs(args.warmups, args.runs, args.max_relative_mad)?;
    if !args.dry_run {
        require_release_measurement()?;
    }
    let selected = registry::select(&into_selector(args.selector))?;
    let machine = environment::resolve(&args.machine)?;
    let env = environment::fingerprint();
    let scenario_ids = selected
        .iter()
        .map(|scenario| scenario.id.to_owned())
        .collect::<Vec<_>>();
    let run_dir = if args.dry_run {
        None
    } else {
        Some(create_run_dir()?)
    };
    let mut errors: Vec<(i32, String)> = Vec::new();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut unstable = 0usize;
    let mut dry_run_checked = 0usize;
    for scenario in selected {
        let baseline_path = baseline::path(&machine, &env, scenario);
        let saved = match baseline::read(&baseline_path) {
            Ok(value) => value,
            Err(error) => {
                errors.push((error.exit_code(), format!("{}: {error}", scenario.id)));
                continue;
            }
        };
        if let Err(error) = validate_compatibility(&saved, &machine, &env, scenario) {
            errors.push((error.exit_code(), format!("{}: {error}", scenario.id)));
            continue;
        }
        if args.dry_run {
            println!(
                "DRY-RUN {}: baseline={} compatible; scenario selected, no solve performed",
                scenario.id,
                baseline_path.display()
            );
            dry_run_checked += 1;
            continue;
        }
        let primary = match isolated_measure(scenario, args.warmups, args.runs) {
            Ok(series) => series,
            Err(error) => {
                errors.push((error.exit_code(), format!("{}: {error}", scenario.id)));
                continue;
            }
        };
        let primary_rows = match comparison::compare(&saved, &primary, args.max_relative_mad) {
            Ok(rows) => rows,
            Err(error) => {
                errors.push((error.exit_code(), format!("{}: {error}", scenario.id)));
                continue;
            }
        };
        let requires_confirmation = primary_rows.iter().any(|row| row.verdict != Verdict::Pass);
        let confirmation = if requires_confirmation {
            match isolated_measure(scenario, args.warmups, args.runs).and_then(|series| {
                comparison::compare(&saved, &series, args.max_relative_mad)
                    .map(|rows| (series, rows))
            }) {
                Ok((series, rows)) => Some((series, rows)),
                Err(error) => {
                    errors.push((
                        error.exit_code(),
                        format!("{} confirmation: {error}", scenario.id),
                    ));
                    continue;
                }
            }
        } else {
            None
        };
        let verdict = report::print_compare(
            &machine,
            &env,
            scenario,
            &primary,
            &primary_rows,
            confirmation.as_ref().map(|(_, rows)| rows.as_slice()),
            &saved,
        );
        if let Some(dir) = &run_dir {
            persist_run(
                dir,
                scenario.id,
                &primary,
                confirmation.as_ref().map(|(series, _)| series),
            )?;
        }
        match verdict {
            Verdict::Pass => passed += 1,
            Verdict::Fail => {
                failed += 1;
                errors.push((
                    1,
                    format!("{}: confirmed performance regression", scenario.id),
                ));
            }
            Verdict::Unstable => {
                unstable += 1;
                errors.push((5, format!("{}: measurements are unstable", scenario.id)))
            }
        }
    }
    if let Some(run_dir) = &run_dir {
        persist_summary(run_dir, "compare", &machine, &env, &scenario_ids, &errors)?;
    }
    if args.dry_run && errors.is_empty() {
        println!(
            "COMPARE DRY-RUN COMPLETE: validated {dry_run_checked} selected baseline(s); no numerical solves performed"
        );
    } else if args.dry_run {
        println!(
            "COMPARE DRY-RUN INCOMPLETE: validated {dry_run_checked} baseline(s); {} issue(s), no numerical solves performed",
            errors.len()
        );
    } else if errors.is_empty() {
        println!(
            "BASELINE COMPARISON COMPLETE: PASS for {passed} scenario(s) on {} / {}",
            machine.alias, env.id
        );
    } else {
        println!(
            "BASELINE COMPARISON INCOMPLETE: {passed} pass, {failed} regression(s), {unstable} unstable, {} other failure(s)",
            errors.len().saturating_sub(failed + unstable)
        );
    }
    if let Some(run_dir) = &run_dir {
        println!("run report: {}", run_dir.display());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(batch_error(errors))
    }
}

fn require_release_measurement() -> Result<(), RunnerError> {
    if cfg!(debug_assertions) {
        return Err(RunnerError::Configuration(
            "record and live compare require a Release build; use `cargo run --release --features perf-gates --bin perf-gates -- ...`"
                .into(),
        ));
    }
    Ok(())
}

fn child_measure(args: MeasureArgs) -> Result<(), RunnerError> {
    let selector = Selector {
        scenario: Some(args.scenario.clone()),
        ..Selector::default()
    };
    let scenario = registry::select(&selector)?
        .into_iter()
        .next()
        .expect("selection contains one scenario");
    let series = measurement::measure(scenario, args.warmups, args.runs)?;
    let encoded = series_to_toml(&series);
    println!("{}", toml::to_string(&encoded)?);
    Ok(())
}

fn isolated_measure(
    scenario: &Scenario,
    warmups: usize,
    runs: usize,
) -> Result<MeasurementSeries, RunnerError> {
    let mut child = Command::new(std::env::current_exe()?)
        .arg("__measure")
        .arg("--scenario")
        .arg(scenario.id)
        .arg("--warmups")
        .arg(warmups.to_string())
        .arg("--runs")
        .arg(runs.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| RunnerError::Child("child stdout was not captured".into()))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| RunnerError::Child("child stderr was not captured".into()))?;
    let stdout_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let started = std::time::Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Err(RunnerError::Child(format!(
                "{} exceeded {} second timeout",
                scenario.id,
                CHILD_TIMEOUT.as_secs()
            )));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let status = child.wait()?;
    let stdout = stdout_reader
        .join()
        .map_err(|_| RunnerError::Child("child stdout reader panicked".into()))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| RunnerError::Child("child stderr reader panicked".into()))??;
    if !status.success() {
        return Err(RunnerError::Child(format!(
            "{} exited with {}; stderr: {}",
            scenario.id,
            status,
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    let stdout = String::from_utf8(stdout)
        .map_err(|error| RunnerError::Child(format!("child output is not UTF-8: {error}")))?;
    let value = toml::Value::Table(toml::from_str::<toml::Table>(&stdout)?);
    series_from_toml(&value)
}

fn validate_compatibility(
    saved: &toml::Value,
    machine: &environment::MachineProfile,
    env: &environment::EnvironmentFingerprint,
    scenario: &Scenario,
) -> Result<(), RunnerError> {
    let fields = [
        ("machine_id", machine.installation_id.as_str()),
        ("environment_id", env.id.as_str()),
        ("scenario_id", scenario.id),
        ("solver", scenario.solver),
        ("target", env.target.as_str()),
        ("profile", env.profile.as_str()),
        ("cpu", env.cpu.as_str()),
        ("cpu_flags", env.cpu_flags.as_str()),
        ("worker_config", env.worker_config.as_str()),
        ("compiled_features", env.compiled_features.as_str()),
    ];
    for (field, expected) in fields {
        if saved.get(field).and_then(toml::Value::as_str) != Some(expected) {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "{field} differs from current run"
            )));
        }
    }
    if saved
        .get("scenario_version")
        .and_then(toml::Value::as_integer)
        != Some(scenario.version as i64)
    {
        return Err(RunnerError::IncompatibleBaseline(
            "scenario version differs".into(),
        ));
    }
    Ok(())
}

fn validate_runs(warmups: usize, runs: usize, threshold: f64) -> Result<(), RunnerError> {
    if runs < 3 {
        return Err(RunnerError::Configuration(
            "--runs must be at least 3; single-run comparisons are not accepted".into(),
        ));
    }
    if warmups > 1000 || runs > 1000 {
        return Err(RunnerError::Configuration(
            "warmup/sample count exceeds safety limit 1000".into(),
        ));
    }
    if !threshold.is_finite() || !(0.0..=10.0).contains(&threshold) {
        return Err(RunnerError::Configuration(
            "tolerance/noise threshold must be finite and in [0, 10]".into(),
        ));
    }
    Ok(())
}

fn into_selector(args: OptionalSelector) -> Selector {
    Selector {
        solver: args.solver,
        scenario: args.scenario,
        group: args.group,
        all: args.all,
    }
}

fn batch_error(errors: Vec<(i32, String)>) -> RunnerError {
    let exit_code = errors
        .iter()
        .map(|(code, _)| *code)
        .max_by_key(|code| match code {
            2 => 6,
            4 => 5,
            3 => 4,
            1 => 3,
            5 => 2,
            _ => 1,
        })
        .unwrap_or(2);
    RunnerError::Batch {
        exit_code,
        details: errors
            .into_iter()
            .map(|(_, message)| message)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn create_run_dir() -> Result<PathBuf, RunnerError> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("performance")
        .join("runs")
        .join(format!("{stamp}-{}", std::process::id()));
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn persist_run(
    path: &Path,
    scenario: &str,
    primary: &MeasurementSeries,
    confirmation: Option<&MeasurementSeries>,
) -> Result<(), RunnerError> {
    let mut text = format!(
        "# Performance run: {scenario}\n\n- scenario: `{scenario}`\n- measurement_runs: {}\n\n## Primary\n\n",
        primary.measurement_runs
    );
    append_metrics(&mut text, primary);
    if let Some(series) = confirmation {
        text.push_str("\n## Confirmation\n\n");
        append_metrics(&mut text, series);
    }
    fs::write(path.join(format!("{scenario}.md")), text)?;
    let mut raw = series_to_toml(primary);
    if let Some(series) = confirmation {
        raw.as_table_mut()
            .expect("root table")
            .insert("confirmation".into(), series_to_toml(series));
    }
    fs::write(
        path.join(format!("{scenario}.toml")),
        toml::to_string_pretty(&raw)?,
    )?;
    Ok(())
}

fn persist_summary(
    path: &Path,
    command: &str,
    machine: &environment::MachineProfile,
    env: &environment::EnvironmentFingerprint,
    scenarios: &[String],
    errors: &[(i32, String)],
) -> Result<(), RunnerError> {
    let mut markdown = format!(
        "# Performance {command} run\n\n- machine: `{}`\n- environment: `{}`\n- target/profile: `{} / {}`\n- rustc: `{}`\n- source: `{}{}`\n- Cargo.lock: `{}`\n- cpu: `{}`\n- selected scenarios: {}\n- outcome: `{}`\n",
        machine.alias,
        env.id,
        env.target,
        env.profile,
        env.rustc,
        env.source_revision,
        if env.source_dirty { " (dirty)" } else { "" },
        env.lockfile_fingerprint,
        env.cpu,
        scenarios.join(", "),
        if errors.is_empty() { "ok" } else { "failed" },
    );
    if !errors.is_empty() {
        markdown.push_str("\n## Errors\n");
        for (code, message) in errors {
            markdown.push_str(&format!("- exit `{code}`: {message}\n"));
        }
    }
    fs::write(path.join("summary.md"), markdown)?;

    let mut root = toml::map::Map::new();
    root.insert("schema_version".into(), toml::Value::Integer(1));
    root.insert("command".into(), toml::Value::String(command.into()));
    root.insert(
        "machine_alias".into(),
        toml::Value::String(machine.alias.clone()),
    );
    root.insert("environment_id".into(), toml::Value::String(env.id.clone()));
    root.insert("target".into(), toml::Value::String(env.target.clone()));
    root.insert("profile".into(), toml::Value::String(env.profile.clone()));
    root.insert("rustc".into(), toml::Value::String(env.rustc.clone()));
    root.insert("cpu".into(), toml::Value::String(env.cpu.clone()));
    root.insert(
        "source_revision".into(),
        toml::Value::String(env.source_revision.clone()),
    );
    root.insert(
        "source_dirty".into(),
        toml::Value::Boolean(env.source_dirty),
    );
    root.insert(
        "lockfile_fingerprint".into(),
        toml::Value::String(env.lockfile_fingerprint.clone()),
    );
    root.insert(
        "selected_scenarios".into(),
        toml::Value::Array(scenarios.iter().cloned().map(toml::Value::String).collect()),
    );
    root.insert(
        "errors".into(),
        toml::Value::Array(
            errors
                .iter()
                .map(|(code, message)| {
                    let mut item = toml::map::Map::new();
                    item.insert("exit_code".into(), toml::Value::Integer(*code as i64));
                    item.insert("message".into(), toml::Value::String(message.clone()));
                    toml::Value::Table(item)
                })
                .collect(),
        ),
    );
    fs::write(
        path.join("summary.toml"),
        toml::to_string_pretty(&toml::Value::Table(root))?,
    )?;
    Ok(())
}

fn append_metrics(text: &mut String, series: &MeasurementSeries) {
    text.push_str("| Metric | Phase | Scope | Parent scope | Samples | Median | MAD | Min | Max |\n| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
    for metric in &series.metrics {
        text.push_str(&format!(
            "| {} ({}) | {} | {} | {} | {} | {:.6} | {:.6} | {:.6} | {:.6} |\n",
            metric.id,
            metric.unit,
            metric.phase,
            metric.scope,
            metric.parent_scope.as_deref().unwrap_or("-"),
            metric.samples.len(),
            metric.median,
            metric.mad,
            metric.min,
            metric.max
        ));
    }
    for counter in &series.counters {
        text.push_str(&format!(
            "- counter {}: median {}, range {}..{}\n",
            counter.id, counter.median, counter.min, counter.max
        ));
    }
    if !series.diagnostics.is_empty() {
        text.push_str("\nCorrectness diagnostics from the final sample:\n");
        for (name, unit, value) in &series.diagnostics {
            text.push_str(&format!("- {name}: `{value:.6e} {unit}`\n"));
        }
    }
}

fn series_to_toml(series: &MeasurementSeries) -> toml::Value {
    let mut root = toml::map::Map::new();
    root.insert(
        "scenario_id".into(),
        toml::Value::String(series.scenario_id.clone()),
    );
    root.insert(
        "scenario_version".into(),
        toml::Value::Integer(series.scenario_version as i64),
    );
    root.insert(
        "warmup_runs".into(),
        toml::Value::Integer(series.warmup_runs as i64),
    );
    root.insert(
        "measurement_runs".into(),
        toml::Value::Integer(series.measurement_runs as i64),
    );
    root.insert(
        "gate_metrics".into(),
        toml::Value::Array(
            series
                .gate_metrics
                .iter()
                .cloned()
                .map(toml::Value::String)
                .collect(),
        ),
    );
    root.insert(
        "metrics".into(),
        toml::Value::Array(
            series
                .metrics
                .iter()
                .map(|metric| {
                    let mut item = toml::map::Map::new();
                    item.insert("id".into(), toml::Value::String(metric.id.clone()));
                    item.insert("unit".into(), toml::Value::String(metric.unit.clone()));
                    item.insert("phase".into(), toml::Value::String(metric.phase.clone()));
                    item.insert("scope".into(), toml::Value::String(metric.scope.clone()));
                    item.insert(
                        "parent_scope".into(),
                        toml::Value::String(metric.parent_scope.clone().unwrap_or_default()),
                    );
                    item.insert("median".into(), toml::Value::Float(metric.median));
                    item.insert("mad".into(), toml::Value::Float(metric.mad));
                    item.insert("min".into(), toml::Value::Float(metric.min));
                    item.insert("max".into(), toml::Value::Float(metric.max));
                    item.insert(
                        "samples".into(),
                        toml::Value::Array(
                            metric
                                .samples
                                .iter()
                                .copied()
                                .map(toml::Value::Float)
                                .collect(),
                        ),
                    );
                    toml::Value::Table(item)
                })
                .collect(),
        ),
    );
    root.insert(
        "counters".into(),
        toml::Value::Array(
            series
                .counters
                .iter()
                .map(|counter| {
                    let mut item = toml::map::Map::new();
                    item.insert("id".into(), toml::Value::String(counter.id.clone()));
                    item.insert("median".into(), toml::Value::Float(counter.median));
                    item.insert("min".into(), toml::Value::Integer(counter.min as i64));
                    item.insert("max".into(), toml::Value::Integer(counter.max as i64));
                    item.insert(
                        "samples".into(),
                        toml::Value::Array(
                            counter
                                .samples
                                .iter()
                                .map(|value| toml::Value::Integer(*value as i64))
                                .collect(),
                        ),
                    );
                    toml::Value::Table(item)
                })
                .collect(),
        ),
    );
    root.insert(
        "diagnostics".into(),
        toml::Value::Array(
            series
                .diagnostics
                .iter()
                .map(|(name, unit, value)| {
                    let mut item = toml::map::Map::new();
                    item.insert("id".into(), toml::Value::String(name.clone()));
                    item.insert("unit".into(), toml::Value::String(unit.clone()));
                    item.insert("value".into(), toml::Value::Float(*value));
                    toml::Value::Table(item)
                })
                .collect(),
        ),
    );
    toml::Value::Table(root)
}

fn series_from_toml(value: &toml::Value) -> Result<MeasurementSeries, RunnerError> {
    let field = |key: &str| {
        value
            .get(key)
            .ok_or_else(|| RunnerError::Child(format!("child result lacks {key}")))
    };
    let integer = |key: &str| {
        field(key)?
            .as_integer()
            .ok_or_else(|| RunnerError::Child(format!("child result has invalid {key}")))
    };
    let metrics_value = field("metrics")?
        .as_array()
        .ok_or_else(|| RunnerError::Child("metrics is not an array".into()))?;
    let mut metrics = Vec::with_capacity(metrics_value.len());
    let gate_metrics = field("gate_metrics")?
        .as_array()
        .ok_or_else(|| RunnerError::Child("gate_metrics is not an array".into()))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| RunnerError::Child("invalid gate metric ID".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for item in metrics_value {
        let required = |key: &str| {
            item.get(key)
                .ok_or_else(|| RunnerError::Child(format!("metric lacks {key}")))
        };
        let string = |key: &str| {
            required(key)?
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| RunnerError::Child(format!("metric has invalid {key}")))
        };
        let number = |key: &str| {
            required(key)?
                .as_float()
                .or_else(|| {
                    required(key)
                        .ok()
                        .and_then(toml::Value::as_integer)
                        .map(|n| n as f64)
                })
                .ok_or_else(|| RunnerError::Child(format!("metric has invalid {key}")))
        };
        let samples = required("samples")?
            .as_array()
            .ok_or_else(|| RunnerError::Child("samples is not an array".into()))?
            .iter()
            .map(|v| {
                v.as_float()
                    .or_else(|| v.as_integer().map(|n| n as f64))
                    .ok_or_else(|| RunnerError::Child("invalid sample".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        metrics.push(measurement::MetricStats {
            id: string("id")?,
            unit: string("unit")?,
            phase: string("phase")?,
            scope: string("scope")?,
            parent_scope: required("parent_scope")?
                .as_str()
                .filter(|scope| !scope.is_empty())
                .map(str::to_owned),
            median: number("median")?,
            mad: number("mad")?,
            min: number("min")?,
            max: number("max")?,
            samples,
        });
    }
    let counters = field("counters")?
        .as_array()
        .ok_or_else(|| RunnerError::Child("counters is not an array".into()))?
        .iter()
        .map(|item| {
            let id = item
                .get("id")
                .and_then(toml::Value::as_str)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| RunnerError::Child("counter lacks a valid id".into()))?
                .to_owned();
            let number = |key: &str| {
                item.get(key)
                    .and_then(toml::Value::as_integer)
                    .filter(|value| *value >= 0)
                    .map(|value| value as u64)
                    .ok_or_else(|| RunnerError::Child(format!("counter has invalid {key}")))
            };
            let median = item
                .get("median")
                .and_then(toml::Value::as_float)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .ok_or_else(|| RunnerError::Child("counter has invalid median".into()))?;
            let samples = item
                .get("samples")
                .and_then(toml::Value::as_array)
                .ok_or_else(|| RunnerError::Child("counter has no raw samples".into()))?
                .iter()
                .map(|value| {
                    value
                        .as_integer()
                        .filter(|value| *value >= 0)
                        .map(|value| value as u64)
                        .ok_or_else(|| RunnerError::Child("invalid counter sample".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CounterStats {
                id,
                median,
                min: number("min")?,
                max: number("max")?,
                samples,
            })
        })
        .collect::<Result<Vec<_>, RunnerError>>()?;
    let diagnostics = field("diagnostics")?
        .as_array()
        .ok_or_else(|| RunnerError::Child("diagnostics is not an array".into()))?
        .iter()
        .map(|item| {
            let id = item
                .get("id")
                .and_then(toml::Value::as_str)
                .ok_or_else(|| RunnerError::Child("diagnostic lacks id".into()))?
                .to_owned();
            let unit = item
                .get("unit")
                .and_then(toml::Value::as_str)
                .ok_or_else(|| RunnerError::Child("diagnostic lacks unit".into()))?
                .to_owned();
            let value = item
                .get("value")
                .and_then(toml::Value::as_float)
                .or_else(|| {
                    item.get("value")
                        .and_then(toml::Value::as_integer)
                        .map(|n| n as f64)
                })
                .filter(|value| value.is_finite())
                .ok_or_else(|| RunnerError::Child("diagnostic has invalid value".into()))?;
            Ok((id, unit, value))
        })
        .collect::<Result<Vec<_>, RunnerError>>()?;
    Ok(MeasurementSeries {
        scenario_id: field("scenario_id")?
            .as_str()
            .ok_or_else(|| RunnerError::Child("invalid scenario_id".into()))?
            .into(),
        scenario_version: integer("scenario_version")? as u32,
        gate_metrics,
        warmup_runs: integer("warmup_runs")? as usize,
        measurement_runs: integer("measurement_runs")? as usize,
        metrics,
        counters,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::{CounterStats, MetricStats};

    #[test]
    fn child_protocol_roundtrips_metric_and_counter_samples() {
        let series = MeasurementSeries {
            scenario_id: "fixture".into(),
            scenario_version: 2,
            gate_metrics: vec!["solve_ms".into()],
            warmup_runs: 3,
            measurement_runs: 3,
            metrics: vec![MetricStats {
                id: "solve_ms".into(),
                unit: "ms".into(),
                phase: "gate_off".into(),
                scope: "scenario_component".into(),
                parent_scope: Some("scenario_outer_wall".into()),
                samples: vec![8.0, 10.0, 12.0],
                median: 10.0,
                mad: 2.0,
                min: 8.0,
                max: 12.0,
            }],
            counters: vec![CounterStats {
                id: "steps".into(),
                samples: vec![4, 5, 6],
                median: 5.0,
                min: 4,
                max: 6,
            }],
            diagnostics: vec![("residual".into(), "absolute".into(), 1e-12)],
        };
        let decoded = series_from_toml(&series_to_toml(&series)).unwrap();
        assert_eq!(decoded.scenario_id, series.scenario_id);
        assert_eq!(decoded.metrics[0].samples, series.metrics[0].samples);
        assert_eq!(decoded.metrics[0].phase, series.metrics[0].phase);
        assert_eq!(decoded.metrics[0].scope, series.metrics[0].scope);
        assert_eq!(
            decoded.metrics[0].parent_scope,
            series.metrics[0].parent_scope
        );
        assert_eq!(decoded.counters[0].samples, series.counters[0].samples);
        assert_eq!(decoded.diagnostics, series.diagnostics);
    }

    #[test]
    fn run_summary_persists_partial_failure_context() {
        let directory = tempfile::tempdir().unwrap();
        let machine = environment::MachineProfile {
            alias: "test-machine".into(),
            installation_id: "test-id".into(),
        };
        let env = environment::EnvironmentFingerprint {
            id: "test-env".into(),
            target: "test-target".into(),
            profile: "debug".into(),
            cpu: "test-cpu".into(),
            rustc: "rustc test".into(),
            cpu_flags: "test-flags".into(),
            worker_config: "1".into(),
            compiled_features: String::new(),
            source_revision: "test-revision".into(),
            source_dirty: false,
            lockfile_fingerprint: "test-lock".into(),
        };
        persist_summary(
            directory.path(),
            "compare",
            &machine,
            &env,
            &["fixture".into()],
            &[(2, "fixture child failed".into())],
        )
        .unwrap();
        let summary = fs::read_to_string(directory.path().join("summary.md")).unwrap();
        assert!(summary.contains("fixture child failed"));
        assert!(summary.contains("failed"));
        assert!(summary.contains("- source: `test-revision`"));
        assert!(summary.contains("- Cargo.lock: `test-lock`"));
        let raw = toml::Value::Table(
            toml::from_str::<toml::Table>(
                &fs::read_to_string(directory.path().join("summary.toml")).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(raw["errors"][0]["exit_code"].as_integer(), Some(2));
    }

    #[test]
    fn release_gate_is_explicit_for_timing_commands() {
        if cfg!(debug_assertions) {
            assert!(require_release_measurement().is_err());
        } else {
            assert!(require_release_measurement().is_ok());
        }
    }
}
