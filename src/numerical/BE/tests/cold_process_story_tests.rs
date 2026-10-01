use super::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const CHILD_TEST: &str = "numerical::BE::cold_process_story_tests::be_cold_e2e_process_child";
const CHILD_ROUTE_ENV: &str = "BE_COLD_E2E_CHILD_ROUTE";
const CHILD_OUTPUT_ENV: &str = "BE_COLD_E2E_CHILD_OUTPUT";
const RESULT_MARKER: &str = "BE_COLD_E2E_RESULT|";
const DIMENSION: usize = 8;
const STEP: f64 = 0.002;
const FINAL_TIME: f64 = 0.02;
const SAMPLES_PER_ROUTE: usize = 4;

#[derive(Debug)]
struct ChildObservation {
    route: String,
    e2e_ms: f64,
    final_state: Vec<f64>,
    build_attempts: u64,
    link_attempts: u64,
    build_ms: f64,
    link_ms: f64,
    artifact_key: String,
}

#[derive(Debug)]
struct Sample {
    round: usize,
    order: usize,
    process_wall_ms: f64,
    observation: ChildObservation,
}

#[test]
#[ignore = "release process-isolated cold E2E matrix; requires tcc and launches eight child test processes"]
fn be_process_isolated_cold_e2e_matrix_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "BE",
        "be_process_isolated_cold_e2e_matrix_story",
    );
    assert!(
        command_available("tcc"),
        "this release story requires tcc on PATH"
    );

    // Keep compiler artifacts on the same volume as the in-process lifecycle
    // story so storage location does not confound the compiler timing comparison.
    let artifact_root = tempfile::Builder::new()
        .prefix("be-cold-e2e-")
        .tempdir_in("target")
        .expect("unique AOT artifact directory under target should exist");
    let timeout_ms = std::env::var("BE_COLD_E2E_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(120_000);

    println!(
        "[BE process-isolated cold E2E] fixture=shared-diffusion-chain-{DIMENSION}; samples_per_route={SAMPLES_PER_ROUTE}; route_order=alternating; artifact_root=target (same volume as lifecycle story); AOT=AtomViewNative/tcc/RebuildAlways; timeout_ms={timeout_ms}; telemetry=Timings"
    );
    println!(
        "sample | route | child_e2e_ms | process_wall_ms | build_ms | link_ms | build/link_attempts | artifact_key"
    );

    let mut lambdify_samples = Vec::with_capacity(SAMPLES_PER_ROUTE);
    let mut aot_samples = Vec::with_capacity(SAMPLES_PER_ROUTE);
    for round in 0..SAMPLES_PER_ROUTE {
        let routes = if round % 2 == 0 {
            ["lambdify", "aot"]
        } else {
            ["aot", "lambdify"]
        };
        for (order, route) in routes.into_iter().enumerate() {
            let sample_output_dir = artifact_root.path().join(format!("round-{round}-{route}"));
            let (observation, process_wall_ms) = run_child(route, &sample_output_dir, timeout_ms)
                .unwrap_or_else(|error| {
                    panic!(
                        "child failed: round={round} route={route} timeout_ms={timeout_ms}: {error}"
                    )
                });
            assert_eq!(observation.route, route);
            if route == "aot" {
                assert_eq!(
                    observation.build_attempts, 1,
                    "each AOT child must build once"
                );
                assert_eq!(
                    observation.link_attempts, 1,
                    "each AOT child must link once"
                );
                assert!(!observation.artifact_key.is_empty());
                aot_samples.push(Sample {
                    round,
                    order,
                    process_wall_ms,
                    observation,
                });
            } else {
                assert_eq!(observation.build_attempts, 0);
                assert_eq!(observation.link_attempts, 0);
                lambdify_samples.push(Sample {
                    round,
                    order,
                    process_wall_ms,
                    observation,
                });
            }
        }
    }

    let mut max_state_diff = 0.0_f64;
    for (lambdify, aot) in lambdify_samples.iter().zip(&aot_samples) {
        assert_eq!(lambdify.round, aot.round);
        assert_eq!(
            lambdify.observation.final_state.len(),
            aot.observation.final_state.len()
        );
        for (left, right) in lambdify
            .observation
            .final_state
            .iter()
            .zip(&aot.observation.final_state)
        {
            max_state_diff = max_state_diff.max((left - right).abs());
        }
    }
    assert!(
        max_state_diff <= 1e-10,
        "route endpoint mismatch: {max_state_diff:e}"
    );

    let mut ordered_samples: Vec<&Sample> = lambdify_samples.iter().chain(&aot_samples).collect();
    ordered_samples.sort_by_key(|sample| (sample.round, sample.order));
    for sample in ordered_samples {
        let item = &sample.observation;
        println!(
            "{} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {}/{} | {}",
            sample.round + 1,
            item.route,
            item.e2e_ms,
            sample.process_wall_ms,
            item.build_ms,
            item.link_ms,
            item.build_attempts,
            item.link_attempts,
            if item.artifact_key.is_empty() {
                "-"
            } else {
                &item.artifact_key
            },
        );
    }
    print_summary("Lambdify", &lambdify_samples);
    print_summary("AOT", &aot_samples);
    println!("[BE process-isolated parity] max_endpoint_state_diff={max_state_diff:.3e}");
}

#[test]
#[ignore = "internal child entry point for be_process_isolated_cold_e2e_matrix_story"]
fn be_cold_e2e_process_child() {
    let Some(route) = std::env::var_os(CHILD_ROUTE_ENV) else {
        return;
    };
    let route = route.to_string_lossy().into_owned();
    assert!(
        route == "lambdify" || route == "aot",
        "invalid child route: {route}"
    );
    let output_dir = std::env::var_os(CHILD_OUTPUT_ENV)
        .map(PathBuf::from)
        .expect("parent must provide a unique output directory");

    let started = Instant::now();
    let workload = crate::numerical::ivp_workloads::diffusion_chain(DIMENSION);
    let parameter_names: Vec<&str> = workload
        .parameter_names
        .iter()
        .map(String::as_str)
        .collect();
    let mut options = BeSolverOptions::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        1e-10,
        20,
        Some(STEP),
        0.0,
        FINAL_TIME,
        workload.initial_state,
    )
    .with_symbolic_assembly_backend(BeSymbolicAssemblyBackend::AtomViewNative)
    .with_telemetry_mode(BeTelemetryMode::Timings);
    if route == "aot" {
        use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
        use crate::symbolic::symbolic_ivp_generated::{
            SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
        };

        let config = SymbolicIvpGeneratedBackendConfig::new()
            .with_output_parent_dir(Some(output_dir))
            .with_c_tcc()
            .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release,
            });
        options = options.with_generated_backend_config(config);
    }

    let mut solver = BE::try_new_with_options(options).expect("BE setup must succeed");
    solver
        .try_set_equation_parameters(Some(&parameter_names))
        .expect("shared workload parameter schema must be accepted");
    solver
        .set_parameter_values(workload.parameter_values)
        .expect("shared workload parameter values must bind");
    solver.try_solve().expect("cold full solve must succeed");
    let e2e_ms = started.elapsed().as_secs_f64() * 1_000.0;
    assert_eq!(solver.status(), BeStatus::Finished);
    let final_state: Vec<f64> = solver
        .trajectory()
        .1
        .row(solver.trajectory().1.nrows() - 1)
        .iter()
        .copied()
        .collect();

    let (build_attempts, link_attempts, build_ms, link_ms, artifact_key) = if route == "aot" {
        use crate::symbolic::ivp_telemetry::IvpColdStage;

        let snapshot = solver
            .symbolic_ivp_telemetry_snapshot()
            .expect("AOT route must expose lifecycle telemetry");
        assert!(snapshot.aot_runtime_ready > 0);
        assert_eq!(snapshot.aot_build_attempts, 1);
        assert_eq!(snapshot.aot_link_attempts, 1);
        assert_eq!(snapshot.aot_build_successes, 1);
        assert_eq!(snapshot.aot_link_successes, 1);
        (
            snapshot.aot_build_attempts,
            snapshot.aot_link_attempts,
            snapshot
                .cold_stage(IvpColdStage::AotBuild)
                .elapsed
                .as_secs_f64()
                * 1_000.0,
            snapshot
                .cold_stage(IvpColdStage::AotLink)
                .elapsed
                .as_secs_f64()
                * 1_000.0,
            snapshot
                .aot_artifact_keys
                .first()
                .cloned()
                .unwrap_or_default(),
        )
    } else {
        (0, 0, 0.0, 0.0, String::new())
    };

    println!(
        "{RESULT_MARKER}route={route}|e2e_ms={e2e_ms:.6}|final_state={}|build_attempts={build_attempts}|link_attempts={link_attempts}|build_ms={build_ms:.6}|link_ms={link_ms:.6}|artifact_key={artifact_key}",
        final_state
            .iter()
            .map(|value| format!("{value:.17e}"))
            .collect::<Vec<_>>()
            .join(",")
    );
}

fn run_child(
    route: &str,
    output_dir: &std::path::Path,
    timeout_ms: u64,
) -> Result<(ChildObservation, f64), String> {
    let executable = std::env::current_exe().map_err(|error| format!("current_exe: {error}"))?;
    let started = Instant::now();
    let mut child = Command::new(executable)
        .arg("--ignored")
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_ROUTE_ENV, route)
        .env(CHILD_OUTPUT_ENV, output_dir)
        .env("RAYON_NUM_THREADS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("spawn: {error}"))?;
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("timeout after {timeout_ms}ms"));
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("wait: {error}"));
            }
        }
    };
    let output = child
        .wait_with_output()
        .map_err(|error| format!("collect output after {status}: {error}"))?;
    let process_wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if !status.success() {
        return Err(format!(
            "child status={status}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result_line = stdout
        .lines()
        .find(|line| line.contains(RESULT_MARKER))
        .ok_or_else(|| format!("child result marker missing; stdout={stdout}"))?;
    let fields: HashMap<&str, &str> = result_line
        .split(RESULT_MARKER)
        .nth(1)
        .unwrap_or_default()
        .split('|')
        .filter_map(|part| part.split_once('='))
        .collect();
    let required = |key: &str| {
        fields
            .get(key)
            .copied()
            .ok_or_else(|| format!("child result field {key:?} missing: {result_line}"))
    };
    let parse_f64 = |key: &str| -> Result<f64, String> {
        required(key)?
            .parse()
            .map_err(|_| format!("invalid child result field {key:?}: {result_line}"))
    };
    let parse_u64 = |key: &str| -> Result<u64, String> {
        required(key)?
            .parse()
            .map_err(|_| format!("invalid child result field {key:?}: {result_line}"))
    };
    let final_state = required("final_state")?
        .split(',')
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| format!("invalid final-state value in {result_line}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let observation = ChildObservation {
        route: required("route")?.to_string(),
        e2e_ms: parse_f64("e2e_ms")?,
        final_state,
        build_attempts: parse_u64("build_attempts")?,
        link_attempts: parse_u64("link_attempts")?,
        build_ms: parse_f64("build_ms")?,
        link_ms: parse_f64("link_ms")?,
        artifact_key: required("artifact_key")?.to_string(),
    };
    Ok((observation, process_wall_ms))
}

fn print_summary(route: &str, samples: &[Sample]) {
    let values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.observation.e2e_ms)
        .collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_process_wall = samples
        .iter()
        .map(|sample| sample.process_wall_ms)
        .sum::<f64>()
        / samples.len() as f64;
    println!(
        "[BE cold E2E summary] route={route} n={} child_e2e_mean_population_std_ms={mean:.3}+/-{std:.3} range_ms=[{min:.3},{max:.3}] process_wall_mean_ms={mean_process_wall:.3}",
        values.len()
    );
}

fn command_available(command: &str) -> bool {
    let locator = if cfg!(windows) { "where" } else { "which" };
    Command::new(locator)
        .arg(command)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}
