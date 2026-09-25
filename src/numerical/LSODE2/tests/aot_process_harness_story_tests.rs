//! Process-isolated AOT/Lambdify protocol.
//!
//! The child process owns all preparation and solver work.  The parent only
//! starts children, parses their machine-readable records, and writes the
//! dated story report.  This keeps process startup, stdout handling, and
//! artifact orchestration out of the measured solver phases.

use super::aot_trajectory_parity_story_tests::trajectory_config;
use super::{
    IvpTelemetry, IvpWarmStage, Lsode2AotProfile, Lsode2AotToolchain, Lsode2LinearSystemStructure,
    Lsode2Solver, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use tempfile::tempdir;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

const CHILD_TEST: &str =
    "numerical::LSODE2::aot_process_harness_story_tests::aot_process_harness_child";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HarnessPhase {
    Cold,
    Warm,
    CallbackOnly,
}

impl HarnessPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold_e2e",
            Self::Warm => "warm_solve",
            Self::CallbackOnly => "callback_only",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "cold_e2e" => Some(Self::Cold),
            "warm_solve" => Some(Self::Warm),
            "callback_only" => Some(Self::CallbackOnly),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HarnessRoute {
    LambdifyExprLegacy,
    LambdifyAtomViewNative,
    AotAtomView(Lsode2AotToolchain),
}

impl HarnessRoute {
    fn as_str(self) -> &'static str {
        match self {
            Self::LambdifyExprLegacy => "lambdify_exprlegacy",
            Self::LambdifyAtomViewNative => "lambdify_atomview_native",
            Self::AotAtomView(Lsode2AotToolchain::Rust) => "aot_atomview_rust",
            Self::AotAtomView(Lsode2AotToolchain::CTcc) => "aot_atomview_c_tcc",
            Self::AotAtomView(Lsode2AotToolchain::CGcc) => "aot_atomview_c_gcc",
            Self::AotAtomView(Lsode2AotToolchain::Zig) => "aot_atomview_zig",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "lambdify_exprlegacy" => Self::LambdifyExprLegacy,
            "lambdify_atomview_native" => Self::LambdifyAtomViewNative,
            "aot_atomview_rust" => Self::AotAtomView(Lsode2AotToolchain::Rust),
            "aot_atomview_c_tcc" => Self::AotAtomView(Lsode2AotToolchain::CTcc),
            "aot_atomview_c_gcc" => Self::AotAtomView(Lsode2AotToolchain::CGcc),
            "aot_atomview_zig" => Self::AotAtomView(Lsode2AotToolchain::Zig),
            _ => return None,
        })
    }

    fn command(self) -> Option<&'static str> {
        match self {
            Self::AotAtomView(Lsode2AotToolchain::CTcc) => Some("tcc"),
            Self::AotAtomView(Lsode2AotToolchain::CGcc) => Some("gcc"),
            Self::AotAtomView(Lsode2AotToolchain::Zig) => Some("zig"),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PhaseRecord {
    phase: HarnessPhase,
    route: &'static str,
    status: &'static str,
    repetitions: usize,
    prepare_ms: f64,
    solve_ms: f64,
    callback_ms: f64,
    residual_calls: usize,
    jacobian_calls: usize,
    linear_solves: usize,
}

fn generated_config(
    output_parent: &Path,
    route: HarnessRoute,
    policy: SymbolicIvpAotBuildPolicy,
) -> SymbolicIvpGeneratedBackendConfig {
    let config = SymbolicIvpGeneratedBackendConfig::defaults()
        .with_build_policy(policy)
        .with_output_parent_dir(Some(output_parent.to_path_buf()));
    match route {
        HarnessRoute::AotAtomView(Lsode2AotToolchain::Rust) => config.with_rust(),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc) => config.with_c_tcc(),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::CGcc) => config.with_c_gcc(),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::Zig) => config.with_zig(),
        HarnessRoute::LambdifyExprLegacy | HarnessRoute::LambdifyAtomViewNative => config,
    }
}

fn build_solver(
    route: HarnessRoute,
    phase: HarnessPhase,
    output_parent: &Path,
) -> Result<Lsode2Solver, String> {
    if let HarnessRoute::AotAtomView(toolchain) = route {
        if !matches!(phase, HarnessPhase::Cold) {
            // A resolver is process-local today.  Warm/callback children
            // bootstrap the artifact outside the measured phase, then strict
            // RequirePrebuilt reconnects through the linked runtime registry.
            let bootstrap_config = generated_config(
                output_parent,
                route,
                SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                },
            );
            let mut bootstrap = Lsode2Solver::new(trajectory_config(
                Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
                Lsode2SymbolicAssemblyBackend::AtomView,
                Lsode2SymbolicExecutionMode::Aot {
                    toolchain,
                    profile: Lsode2AotProfile::Debug,
                },
                Some(bootstrap_config),
                IvpTelemetry::disabled(),
            ))
            .map_err(|error| format!("warm bootstrap construction failed: {error}"))?;
            bootstrap
                .prepare()
                .map_err(|error| format!("warm bootstrap preparation failed: {error}"))?;
        }
    }
    let (assembly, execution, generated) = match route {
        HarnessRoute::LambdifyExprLegacy => (
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        ),
        HarnessRoute::LambdifyAtomViewNative => (
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
        ),
        HarnessRoute::AotAtomView(toolchain) => (
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::Aot {
                toolchain,
                profile: Lsode2AotProfile::Debug,
            },
            Some(generated_config(
                output_parent,
                route,
                match phase {
                    HarnessPhase::Cold => SymbolicIvpAotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Debug,
                    },
                    HarnessPhase::Warm | HarnessPhase::CallbackOnly => {
                        SymbolicIvpAotBuildPolicy::RequirePrebuilt
                    }
                },
            )),
        ),
    };
    Lsode2Solver::new(trajectory_config(
        Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
        assembly,
        execution,
        generated,
        IvpTelemetry::detailed(),
    ))
    .map_err(|error| error.to_string())
}

fn run_child_phase(
    route: HarnessRoute,
    phase: HarnessPhase,
    output_parent: &Path,
    repetitions: usize,
) -> Result<PhaseRecord, String> {
    let mut solver = build_solver(route, phase, output_parent)?;
    let prepare_started = Instant::now();
    solver.prepare().map_err(|error| error.to_string())?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1.0e3;

    let mut solve_ms = 0.0;
    let mut callback_ms = 0.0;
    let mut residual_calls = 0usize;
    let mut jacobian_calls = 0usize;
    let mut linear_solves = 0usize;
    for _ in 0..repetitions.max(1) {
        let started = (phase != HarnessPhase::CallbackOnly).then(Instant::now);
        solver
            .solve_with_summary()
            .map_err(|error| error.to_string())?;
        if let Some(started) = started {
            solve_ms += started.elapsed().as_secs_f64() * 1.0e3;
        }
        let snapshot = solver.telemetry_snapshot();
        residual_calls = snapshot.residual_evaluations as usize;
        jacobian_calls = snapshot.jacobian_evaluations as usize;
        linear_solves = snapshot.linear_solve_requests as usize;
        callback_ms += (snapshot.warm_stage(IvpWarmStage::ResidualCallback).elapsed
            + snapshot.warm_stage(IvpWarmStage::JacobianCallback).elapsed)
            .as_secs_f64()
            * 1.0e3;
    }
    let divisor = repetitions.max(1) as f64;
    let record = PhaseRecord {
        phase,
        route: route.as_str(),
        status: "ok",
        repetitions: repetitions.max(1),
        prepare_ms,
        solve_ms: solve_ms / divisor,
        callback_ms: callback_ms / divisor,
        residual_calls,
        jacobian_calls,
        linear_solves,
    };
    println!(
        "RST_AOT_HARNESS|phase={}|route={}|status={}|repetitions={}|prepare_ms={:.6}|solve_ms={:.6}|callback_ms={:.6}|residual_calls={}|jacobian_calls={}|linear_solves={}",
        record.phase.as_str(),
        record.route,
        record.status,
        record.repetitions,
        record.prepare_ms,
        record.solve_ms,
        record.callback_ms,
        record.residual_calls,
        record.jacobian_calls,
        record.linear_solves,
    );
    Ok(record)
}

fn parse_record(stdout: &str) -> Result<PhaseRecord, String> {
    let line = stdout
        .lines()
        .find_map(|line| line.find("RST_AOT_HARNESS|").map(|index| &line[index..]))
        .ok_or_else(|| format!("child did not emit a harness record; stdout={stdout:?}"))?;
    let mut values = std::collections::BTreeMap::new();
    for field in line.split('|').skip(1) {
        let (key, value) = field
            .split_once('=')
            .ok_or_else(|| format!("invalid child field {field:?}"))?;
        values.insert(key, value);
    }
    let phase = HarnessPhase::parse(values.get("phase").copied().unwrap_or_default())
        .ok_or_else(|| "invalid child phase".to_string())?;
    let route = values
        .get("route")
        .copied()
        .ok_or_else(|| "child route is missing".to_string())?;
    let route = HarnessRoute::parse(route).ok_or_else(|| "child route is invalid".to_string())?;
    let status = match values.get("status").copied().unwrap_or("error") {
        "ok" => "ok",
        "skipped" => "skipped",
        _ => "error",
    };
    Ok(PhaseRecord {
        phase,
        route: route.as_str(),
        status,
        repetitions: parse_field(&values, "repetitions")?,
        prepare_ms: parse_field(&values, "prepare_ms")?,
        solve_ms: parse_field(&values, "solve_ms")?,
        callback_ms: parse_field(&values, "callback_ms")?,
        residual_calls: parse_field(&values, "residual_calls")?,
        jacobian_calls: parse_field(&values, "jacobian_calls")?,
        linear_solves: parse_field(&values, "linear_solves")?,
    })
}

fn parse_field<T: std::str::FromStr>(
    values: &std::collections::BTreeMap<&str, &str>,
    key: &str,
) -> Result<T, String> {
    values
        .get(key)
        .ok_or_else(|| format!("child field {key:?} is missing"))?
        .parse()
        .map_err(|_| format!("child field {key:?} is invalid"))
}

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run_isolated(
    route: HarnessRoute,
    phase: HarnessPhase,
    output_parent: &Path,
    repetitions: usize,
) -> Result<PhaseRecord, String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let output = Command::new(executable)
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("LSODE2_AOT_HARNESS_CHILD", "1")
        .env("LSODE2_AOT_HARNESS_PHASE", phase.as_str())
        .env("LSODE2_AOT_HARNESS_ROUTE", route.as_str())
        .env("LSODE2_AOT_HARNESS_OUTPUT", output_parent)
        .env("LSODE2_AOT_HARNESS_REPETITIONS", repetitions.to_string())
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .map_err(|error| format!("failed to start child: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "child failed for {}/{}: status={} stdout={} stderr={}",
            route.as_str(),
            phase.as_str(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_record(&String::from_utf8_lossy(&output.stdout))
}

fn run_parent_matrix(routes: &[HarnessRoute], repetitions: usize) {
    let output_parent = tempdir().expect("isolated AOT output directory should exist");
    reportln!(
        "[LSODE2 process-isolated harness] fixture=scalar-parameterized; matrix=Banded; thread_policy=1; repetitions={repetitions}; child_process_timing_excluded; callback_only=telemetry callback stages, solve_ms=0"
    );
    reportln!(
        "route | phase | prepare_ms | solve_ms | callback_ms | residual_calls | jacobian_calls | linear_solves | status"
    );
    for &route in routes {
        if let Some(command) = route.command() {
            if !command_available(command) {
                reportln!(
                    "{} | skipped | - | - | - | - | - | - | missing:{command}",
                    route.as_str()
                );
                continue;
            }
        }
        for phase in [
            HarnessPhase::Cold,
            HarnessPhase::Warm,
            HarnessPhase::CallbackOnly,
        ] {
            let record = run_isolated(route, phase, output_parent.path(), repetitions)
                .unwrap_or_else(|error| {
                    panic!("isolated {} {}: {error}", route.as_str(), phase.as_str())
                });
            assert_eq!(record.route, route.as_str());
            assert_eq!(record.phase, phase);
            assert_eq!(record.status, "ok");
            assert!(record.repetitions >= 1);
            assert!(record.prepare_ms.is_finite());
            assert!(record.solve_ms.is_finite());
            assert!(record.callback_ms.is_finite());
            reportln!(
                "{} | {} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {}",
                record.route,
                record.phase.as_str(),
                record.prepare_ms,
                record.solve_ms,
                record.callback_ms,
                record.residual_calls,
                record.jacobian_calls,
                record.linear_solves,
                record.status,
            );
        }
    }
}

#[test]
fn aot_process_isolated_harness_protocol_smoke() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_process_harness_story_tests::aot_process_isolated_harness_protocol_smoke",
    );
    run_parent_matrix(
        &[
            HarnessRoute::LambdifyExprLegacy,
            HarnessRoute::LambdifyAtomViewNative,
            HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc),
        ],
        1,
    );
}

#[test]
#[ignore = "release process-isolated matrix; compare all toolchains after the AOT route is ready"]
fn aot_process_isolated_release_apple_to_apple_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_process_harness_story_tests::aot_process_isolated_release_apple_to_apple_matrix",
    );
    run_parent_matrix(
        &[
            HarnessRoute::LambdifyExprLegacy,
            HarnessRoute::LambdifyAtomViewNative,
            HarnessRoute::AotAtomView(Lsode2AotToolchain::Rust),
            HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc),
            HarnessRoute::AotAtomView(Lsode2AotToolchain::CGcc),
            HarnessRoute::AotAtomView(Lsode2AotToolchain::Zig),
        ],
        5,
    );
}

#[test]
fn aot_process_harness_child() {
    if std::env::var_os("LSODE2_AOT_HARNESS_CHILD").is_none() {
        return;
    }
    let phase = std::env::var("LSODE2_AOT_HARNESS_PHASE")
        .ok()
        .and_then(|value| HarnessPhase::parse(&value))
        .expect("child phase should be valid");
    let route = std::env::var("LSODE2_AOT_HARNESS_ROUTE")
        .ok()
        .and_then(|value| HarnessRoute::parse(&value))
        .expect("child route should be valid");
    let output_parent = PathBuf::from(
        std::env::var("LSODE2_AOT_HARNESS_OUTPUT").expect("child output directory should exist"),
    );
    let repetitions = std::env::var("LSODE2_AOT_HARNESS_REPETITIONS")
        .expect("child repetitions should exist")
        .parse()
        .expect("child repetitions should be numeric");
    run_child_phase(route, phase, &output_parent, repetitions)
        .unwrap_or_else(|error| panic!("child harness phase failed: {error}"));
}
