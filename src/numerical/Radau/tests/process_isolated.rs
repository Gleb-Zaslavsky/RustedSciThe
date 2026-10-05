//! Process-isolated AOT producer/consumer evidence.
//!
//! The parent test only orchestrates two child test processes.  Preparation,
//! dynamic loading and continuation happen in the child so an in-process
//! resolver or linker cache cannot accidentally make the handoff look valid.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tabled::Tabled;

use super::super::api::{
    RadauAotConfig, RadauExecution, RadauFrontend, RadauMatrixLayout, RadauSolver,
    RadauTelemetryMode,
};
use super::story_support::{max_diff, workload_config, workload_problem};
use crate::numerical::ivp_workloads::{WorkloadKind, parameter_continuation_target};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;

const CHILD_TEST: &str = "numerical::Radau::tests::process_isolated::radau_aot_process_child";
const CHILD_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Tabled)]
struct ProcessHandoffRow {
    frontend: String,
    layout: String,
    producer: String,
    consumer: String,
    consumer_builds: String,
    consumer_links: String,
    reconnects: String,
    max_diff: String,
    status: String,
}

fn layout_label(layout: RadauMatrixLayout) -> &'static str {
    match layout {
        RadauMatrixLayout::Dense => "dense",
        RadauMatrixLayout::Sparse => "sparse",
        RadauMatrixLayout::Banded { .. } => "banded",
    }
}

fn parse_layout(value: &str) -> RadauMatrixLayout {
    match value {
        "dense" => RadauMatrixLayout::Dense,
        "sparse" => RadauMatrixLayout::Sparse,
        "banded" => RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        other => panic!("unknown Radau child layout: {other}"),
    }
}

fn run_child(phase: &str, frontend: &str, layout: &str, output_dir: &Path) -> Output {
    let executable = std::env::current_exe().expect("Radau test executable");
    let mut child = Command::new(executable)
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("RADAU_PROCESS_PHASE", phase)
        .env("RADAU_PROCESS_FRONTEND", frontend)
        .env("RADAU_PROCESS_LAYOUT", layout)
        .env("RADAU_PROCESS_OUTPUT_DIR", output_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("spawn Radau process child {phase}/{frontend}: {error}"));

    let started = Instant::now();
    loop {
        if child
            .try_wait()
            .unwrap_or_else(|error| panic!("poll Radau process child: {error}"))
            .is_some()
        {
            return child
                .wait_with_output()
                .unwrap_or_else(|error| panic!("collect Radau process child: {error}"));
        }
        if started.elapsed() > CHILD_TIMEOUT {
            let _ = child.kill();
            panic!("Radau process child timed out: phase={phase} frontend={frontend}");
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn child_metric(output: &Output, key: &str) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|line| {
            line.find("RADAU_PROCESS consumer")
                .map(|start| &line[start..])
        })
        .and_then(|line| {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix(&format!("{key}=")))
        })
        .unwrap_or("missing")
        .to_owned()
}

fn output_text(output: &Output) -> String {
    format!(
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
#[ignore = "release process-isolated AOT handoff; requires an available tcc toolchain"]
fn radau_aot_process_isolated_producer_consumer_handoff() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_AOT_Process",
        "numerical::Radau::tests::process_isolated::radau_aot_process_isolated_producer_consumer_handoff",
    );
    let output_dir = tempfile::tempdir().expect("shared Radau process output directory");
    let mut rows = Vec::new();

    for frontend in ["expr-legacy", "atom-native"] {
        for layout in ["dense", "sparse", "banded"] {
            let route_dir = output_dir.path().join(format!("{frontend}-{layout}"));
            std::fs::create_dir_all(&route_dir).expect("route output directory");
            let producer = run_child("producer", frontend, layout, &route_dir);
            assert!(
                producer.status.success(),
                "producer failed for {frontend}/{layout}: {}",
                output_text(&producer)
            );
            let consumer = run_child("consumer", frontend, layout, &route_dir);
            assert!(
                consumer.status.success(),
                "consumer failed for {frontend}/{layout}: {}",
                output_text(&consumer)
            );
            let consumer_summary = String::from_utf8_lossy(&consumer.stdout);
            assert!(
                consumer_summary
                    .lines()
                    .any(|line| line.contains("RADAU_PROCESS consumer")),
                "consumer did not emit a compact handoff summary: {}",
                output_text(&consumer)
            );
            rows.push(ProcessHandoffRow {
                frontend: frontend.to_owned(),
                layout: layout.to_owned(),
                producer: producer.status.to_string(),
                consumer: consumer.status.to_string(),
                consumer_builds: child_metric(&consumer, "builds"),
                consumer_links: child_metric(&consumer, "links"),
                reconnects: child_metric(&consumer, "reconnects"),
                max_diff: child_metric(&consumer, "max_diff"),
                status: "ok".to_owned(),
            });
        }
    }

    crate::Utils::test_reporting::capture_test_table(
        "[Radau process-isolated AOT handoff] matrix=Dense/Sparse/Banded",
        &rows,
    );
}

#[test]
fn radau_aot_process_child() {
    if std::env::var_os("RADAU_PROCESS_PHASE").is_none() {
        return;
    }
    let phase = std::env::var("RADAU_PROCESS_PHASE").expect("child phase");
    let frontend = match std::env::var("RADAU_PROCESS_FRONTEND").as_deref() {
        Ok("expr-legacy") => RadauFrontend::ExprLegacy,
        Ok("atom-native") => RadauFrontend::AtomViewNative,
        other => panic!("unknown Radau child frontend: {other:?}"),
    };
    let layout = parse_layout(&std::env::var("RADAU_PROCESS_LAYOUT").expect("child matrix layout"));
    let output_dir = PathBuf::from(
        std::env::var_os("RADAU_PROCESS_OUTPUT_DIR").expect("child output directory"),
    );
    let handoff = output_dir.join("radau-process-handoff.txt");

    match phase.as_str() {
        "producer" => {
            let aot =
                RadauAotConfig::build_if_missing_release(output_dir.clone()).with_c_compiler("tcc");
            let (problem, _, _) = workload_problem(WorkloadKind::DiffusionChain, 16, frontend);
            let mut config = workload_config(
                WorkloadKind::DiffusionChain,
                frontend,
                RadauExecution::Aot,
                layout,
                super::super::api::RadauExecutionPolicy::Sequential,
                RadauTelemetryMode::Counters,
            );
            config.aot = Some(aot);
            let solver = RadauSolver::prepare(problem, config)
                .unwrap_or_else(|error| panic!("Radau producer preparation failed: {error}"));
            let resolver = solver
                .config()
                .aot
                .as_ref()
                .and_then(|aot| aot.generated.resolver.clone())
                .expect("producer must publish AOT resolver");
            assert!(resolver.registry().problem_keys().iter().all(|key| {
                resolver
                    .registry()
                    .get_by_problem_key(key)
                    .and_then(|artifact| artifact.codegen_backend)
                    == Some(AotCodegenBackend::C)
            }));
            resolver
                .write_handoff(&handoff)
                .expect("write Radau process handoff");
            println!("RADAU_PROCESS producer frontend={frontend:?} status=ok");
        }
        "consumer" => {
            let resolver = AotResolver::read_handoff(&handoff)
                .unwrap_or_else(|error| panic!("read Radau process handoff: {error}"));
            assert!(resolver.registry().problem_keys().iter().all(|key| {
                resolver
                    .registry()
                    .get_by_problem_key(key)
                    .and_then(|artifact| artifact.codegen_backend)
                    == Some(AotCodegenBackend::C)
            }));
            let aot = RadauAotConfig::require_prebuilt().with_resolver(Some(resolver));
            let (problem, initial_state, parameters) =
                workload_problem(WorkloadKind::DiffusionChain, 16, frontend);
            let mut config = workload_config(
                WorkloadKind::DiffusionChain,
                frontend,
                RadauExecution::Aot,
                layout,
                super::super::api::RadauExecutionPolicy::Sequential,
                RadauTelemetryMode::Counters,
            );
            config.aot = Some(aot);
            let mut solver = RadauSolver::prepare(problem, config)
                .unwrap_or_else(|error| panic!("Radau consumer preparation failed: {error}"));
            let target =
                parameter_continuation_target(&nalgebra::DVector::from_vec(parameters.clone()), 1);
            let first = solver
                .solve_with_parameters(&initial_state, &parameters)
                .expect("Radau consumer solve");
            let continued = solver
                .continue_with_parameters(&initial_state, target.as_slice())
                .expect("Radau consumer continuation");
            let (reference_problem, _, _) =
                workload_problem(WorkloadKind::DiffusionChain, 16, frontend);
            let reference_config = workload_config(
                WorkloadKind::DiffusionChain,
                frontend,
                RadauExecution::Lambdify,
                layout,
                super::super::api::RadauExecutionPolicy::Sequential,
                RadauTelemetryMode::Off,
            );
            let mut reference = RadauSolver::prepare(reference_problem, reference_config)
                .expect("Radau process reference preparation");
            let reference_first = reference
                .solve_with_parameters(&initial_state, &parameters)
                .expect("Radau process reference solve");
            let reference_continued = reference
                .continue_with_parameters(&initial_state, target.as_slice())
                .expect("Radau process reference continuation");
            let counters = &continued.telemetry().counters;
            assert_eq!(counters["aot_build_attempts"], 0);
            // A fresh process has no linked registry entry.  RequirePrebuilt
            // must therefore perform one dynamic-load/reconnect attempt, but
            // it must never invoke the compiler.
            assert_eq!(counters["aot_link_attempts"], 1);
            assert_eq!(counters["aot_link_successes"], 1);
            assert_eq!(counters["aot_link_failures"], 0);
            assert!(counters["aot_reconnects"] >= 1);
            let max_diff = max_diff(&first.y, &reference_first.y)
                .max(max_diff(&continued.y, &reference_continued.y));
            assert!(
                max_diff < 5.0e-6,
                "process handoff parity drift={max_diff:.3e}"
            );
            assert!(first.y.iter().all(|value| value.is_finite()));
            assert!(continued.y.iter().all(|value| value.is_finite()));
            println!(
                "RADAU_PROCESS consumer frontend={frontend:?} layout={} reconnects={} builds={} links={} max_diff={max_diff:.3e} status=ok",
                layout_label(layout),
                counters["aot_reconnects"],
                counters["aot_build_attempts"],
                counters["aot_link_attempts"],
            );
        }
        other => panic!("unknown Radau child phase: {other}"),
    }
}
