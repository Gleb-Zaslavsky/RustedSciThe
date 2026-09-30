//! Process-isolated AOT/Lambdify protocol.
//!
//! The child process owns all preparation and solver work.  The parent only
//! starts children, parses their machine-readable records, and writes the
//! dated story report.  This keeps process startup, stdout handling, and
//! artifact orchestration out of the measured solver phases.

use super::aot_trajectory_parity_story_tests::trajectory_config;
use super::{
    IvpColdStage, IvpTelemetry, IvpWarmStage, Lsode2AotProfile, Lsode2AotToolchain,
    Lsode2LinearSystemStructure, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::DVector;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
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
    Producer,
    Consumer,
    ConsumerContinuation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandoffVariant {
    Baseline,
    ParameterSchema,
    Layout,
    JacobianPattern,
}

impl HandoffVariant {
    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ParameterSchema => "parameter_schema",
            Self::Layout => "layout",
            Self::JacobianPattern => "jacobian_pattern",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "baseline" => Some(Self::Baseline),
            "parameter_schema" => Some(Self::ParameterSchema),
            "layout" => Some(Self::Layout),
            "jacobian_pattern" => Some(Self::JacobianPattern),
            _ => None,
        }
    }

    fn consumer_label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ParameterSchema => "schema",
            Self::Layout => "layout",
            Self::JacobianPattern => "jacobian-pattern",
        }
    }
}

impl HarnessPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold_e2e",
            Self::Warm => "warm_solve",
            Self::CallbackOnly => "callback_only",
            Self::Producer => "producer",
            Self::Consumer => "consumer",
            Self::ConsumerContinuation => "consumer_continuation",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "cold_e2e" => Some(Self::Cold),
            "warm_solve" => Some(Self::Warm),
            "callback_only" => Some(Self::CallbackOnly),
            "producer" => Some(Self::Producer),
            "consumer" => Some(Self::Consumer),
            "consumer_continuation" => Some(Self::ConsumerContinuation),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HarnessRoute {
    LambdifyExprLegacy,
    LambdifyAtomViewNative,
    AotExprLegacy(Lsode2AotToolchain),
    AotAtomView(Lsode2AotToolchain),
}

impl HarnessRoute {
    fn as_str(self) -> &'static str {
        match self {
            Self::LambdifyExprLegacy => "lambdify_exprlegacy",
            Self::LambdifyAtomViewNative => "lambdify_atomview_native",
            Self::AotExprLegacy(Lsode2AotToolchain::Rust) => "aot_exprlegacy_rust",
            Self::AotExprLegacy(Lsode2AotToolchain::CTcc) => "aot_exprlegacy_c_tcc",
            Self::AotExprLegacy(Lsode2AotToolchain::CGcc) => "aot_exprlegacy_c_gcc",
            Self::AotExprLegacy(Lsode2AotToolchain::Zig) => "aot_exprlegacy_zig",
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
            "aot_exprlegacy_rust" => Self::AotExprLegacy(Lsode2AotToolchain::Rust),
            "aot_exprlegacy_c_tcc" => Self::AotExprLegacy(Lsode2AotToolchain::CTcc),
            "aot_exprlegacy_c_gcc" => Self::AotExprLegacy(Lsode2AotToolchain::CGcc),
            "aot_exprlegacy_zig" => Self::AotExprLegacy(Lsode2AotToolchain::Zig),
            "aot_atomview_rust" => Self::AotAtomView(Lsode2AotToolchain::Rust),
            "aot_atomview_c_tcc" => Self::AotAtomView(Lsode2AotToolchain::CTcc),
            "aot_atomview_c_gcc" => Self::AotAtomView(Lsode2AotToolchain::CGcc),
            "aot_atomview_zig" => Self::AotAtomView(Lsode2AotToolchain::Zig),
            _ => return None,
        })
    }

    fn command(self) -> Option<&'static str> {
        match self {
            Self::AotExprLegacy(Lsode2AotToolchain::CTcc)
            | Self::AotAtomView(Lsode2AotToolchain::CTcc) => Some("tcc"),
            Self::AotExprLegacy(Lsode2AotToolchain::CGcc)
            | Self::AotAtomView(Lsode2AotToolchain::CGcc) => Some("gcc"),
            Self::AotExprLegacy(Lsode2AotToolchain::Zig)
            | Self::AotAtomView(Lsode2AotToolchain::Zig) => Some("zig"),
            Self::AotExprLegacy(Lsode2AotToolchain::Rust)
            | Self::AotAtomView(Lsode2AotToolchain::Rust) => None,
            Self::LambdifyExprLegacy | Self::LambdifyAtomViewNative => None,
        }
    }

    fn aot_toolchain(self) -> Option<Lsode2AotToolchain> {
        match self {
            Self::AotExprLegacy(toolchain) | Self::AotAtomView(toolchain) => Some(toolchain),
            Self::LambdifyExprLegacy | Self::LambdifyAtomViewNative => None,
        }
    }

    fn aot_assembly(self) -> Option<Lsode2SymbolicAssemblyBackend> {
        match self {
            Self::AotExprLegacy(_) => Some(Lsode2SymbolicAssemblyBackend::ExprLegacy),
            Self::AotAtomView(_) => Some(Lsode2SymbolicAssemblyBackend::AtomView),
            Self::LambdifyExprLegacy | Self::LambdifyAtomViewNative => None,
        }
    }
}

#[derive(Debug, Clone)]
struct PhaseRecord {
    phase: HarnessPhase,
    route: &'static str,
    status: &'static str,
    repetitions: usize,
    parameter: f64,
    prepare_ms: f64,
    solve_ms: f64,
    callback_ms: f64,
    controller_ms: f64,
    controller_iteration_ms: f64,
    solve_telemetry_ms: f64,
    solve_minus_controller_ms: f64,
    argument_binding_ms: f64,
    residual_callback_ms: f64,
    jacobian_callback_ms: f64,
    residual_eval_ms: f64,
    jacobian_eval_ms: f64,
    residual_output_ms: f64,
    jacobian_output_ms: f64,
    factorization_ms: f64,
    rhs_ms: f64,
    native_engine_setup_ms: f64,
    native_result_assembly_ms: f64,
    final_t: f64,
    final_state_0: f64,
    residual_calls: usize,
    jacobian_calls: usize,
    linear_solves: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    jacobian_rebuilds: usize,
    parameter_binds: usize,
    method_switches: usize,
    errors: usize,
    copies: usize,
    copied_bytes: u64,
    allocated_bytes: u64,
    parallel_dispatches: usize,
    sequential_dispatches: usize,
    worker_count: usize,
    auto_min_work_per_job: usize,
    aot_chunk_dispatches: usize,
    aot_parallel_dispatches: usize,
    aot_chunks: usize,
    aot_worker_callbacks: usize,
    aot_resolution_hits: usize,
    aot_resolution_misses: usize,
    aot_reconnects: usize,
    aot_build_attempts: usize,
    aot_link_attempts: usize,
    aot_link_ms: f64,
    aot_publication_ms: f64,
    aot_runtime_ready: usize,
    aot_artifact_keys: Vec<String>,
}

fn stage_ms(snapshot: &super::IvpTelemetrySnapshot, stage: IvpWarmStage, divisor: f64) -> f64 {
    snapshot.warm_stage(stage).elapsed.as_secs_f64() * 1.0e3 / divisor
}

fn aot_provenance(record: &PhaseRecord) -> &'static str {
    // Lambdify shares resolver counters with the solver telemetry, but it is
    // not an AOT producer or consumer. Keep those rows out of AOT cache
    // accounting so the process report remains semantically comparable.
    if !record.route.starts_with("aot_") {
        return "non_aot";
    }
    if record.aot_build_attempts > 0 {
        "producer_build"
    } else if record.aot_reconnects > 0 || record.aot_resolution_hits > 0 {
        "consumer_reconnect"
    } else if record.aot_resolution_misses > 0 {
        "cache_miss"
    } else {
        "aot_unknown"
    }
}

#[derive(Debug)]
enum HarnessFailure {
    Spawn(String),
    Timeout {
        route: &'static str,
        phase: HarnessPhase,
        timeout_ms: u64,
    },
    Child {
        route: &'static str,
        phase: HarnessPhase,
        details: String,
    },
    Protocol(String),
}

impl std::fmt::Display for HarnessFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(message) => write!(formatter, "spawn failure: {message}"),
            Self::Timeout {
                route,
                phase,
                timeout_ms,
            } => write!(
                formatter,
                "timeout: route={} phase={} timeout_ms={timeout_ms}",
                route,
                phase.as_str(),
            ),
            Self::Child {
                route,
                phase,
                details,
            } => write!(
                formatter,
                "child failure: route={} phase={} {details}",
                route,
                phase.as_str(),
            ),
            Self::Protocol(message) => write!(formatter, "protocol failure: {message}"),
        }
    }
}

impl std::error::Error for HarnessFailure {}

fn generated_config(
    output_parent: &Path,
    route: HarnessRoute,
    policy: SymbolicIvpAotBuildPolicy,
) -> SymbolicIvpGeneratedBackendConfig {
    generated_config_with_resolver(output_parent, route, policy, None)
}

fn generated_config_with_resolver(
    output_parent: &Path,
    route: HarnessRoute,
    policy: SymbolicIvpAotBuildPolicy,
    resolver: Option<AotResolver>,
) -> SymbolicIvpGeneratedBackendConfig {
    let config = SymbolicIvpGeneratedBackendConfig::defaults()
        .with_build_policy(policy)
        .with_output_parent_dir(Some(output_parent.to_path_buf()))
        .with_resolver(resolver);
    match route {
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::Rust)
        | HarnessRoute::AotAtomView(Lsode2AotToolchain::Rust) => config.with_rust(),
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CTcc)
        | HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc) => config.with_c_tcc(),
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CGcc)
        | HarnessRoute::AotAtomView(Lsode2AotToolchain::CGcc) => config.with_c_gcc(),
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::Zig)
        | HarnessRoute::AotAtomView(Lsode2AotToolchain::Zig) => config.with_zig(),
        HarnessRoute::LambdifyExprLegacy | HarnessRoute::LambdifyAtomViewNative => config,
    }
}

fn handoff_structure(variant: HandoffVariant) -> Lsode2LinearSystemStructure {
    match variant {
        HandoffVariant::Layout => Lsode2LinearSystemStructure::Sparse,
        HandoffVariant::JacobianPattern => Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 },
        HandoffVariant::Baseline | HandoffVariant::ParameterSchema => {
            Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 }
        }
    }
}

fn apply_handoff_variant(
    mut config: super::Lsode2ProblemConfig,
    variant: HandoffVariant,
    parameter: f64,
) -> super::Lsode2ProblemConfig {
    match variant {
        HandoffVariant::Baseline | HandoffVariant::Layout => {
            config.eq_system = vec![
                Expr::parse_expression("-a*y1"),
                Expr::parse_expression("-a*y2"),
            ];
            config.values = vec!["y1".to_string(), "y2".to_string()];
            config.y0 = DVector::from_vec(vec![1.0, 0.5]);
            config.equation_parameters = Some(vec!["a".to_string()]);
            config.equation_parameter_values = Some(DVector::from_vec(vec![parameter]));
            if variant == HandoffVariant::Layout {
                // Reassert the consumer's public layout after backend
                // synchronization. This is the contract being invalidated;
                // it must not be inferred from the producer's handoff.
                config.linear_system_structure = Lsode2LinearSystemStructure::Sparse;
                config.backend.linear_solver_backend = super::Lsode2LinearSolverBackend::SparseFaer;
                config.linear_solver_policy = super::Lsode2LinearSolverPolicy::Force(
                    super::Lsode2LinearSolverChoice::FaerSparseLu,
                );
            } else {
                // A zero-width scalar band intentionally falls back to the
                // residual-only identity. Use a real compact band here so
                // the producer/consumer gate exercises layout provenance.
                config.linear_system_structure =
                    Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 };
            }
        }
        HandoffVariant::ParameterSchema => {
            config.eq_system = vec![
                Expr::parse_expression("-a*y1+b"),
                Expr::parse_expression("-a*y2"),
            ];
            config.values = vec!["y1".to_string(), "y2".to_string()];
            config.y0 = DVector::from_vec(vec![1.0, 0.5]);
            config.equation_parameters = Some(vec!["a".to_string(), "b".to_string()]);
            config.equation_parameter_values = Some(DVector::from_vec(vec![parameter, 0.25]));
            config.linear_system_structure = Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 };
        }
        HandoffVariant::JacobianPattern => {
            config.eq_system = vec![
                Expr::parse_expression("-a*y1"),
                Expr::parse_expression("-a*y1+b*y2"),
            ];
            config.values = vec!["y1".to_string(), "y2".to_string()];
            config.y0 = DVector::from_vec(vec![1.0, 0.25]);
            config.equation_parameters = Some(vec!["a".to_string(), "b".to_string()]);
            config.equation_parameter_values = Some(DVector::from_vec(vec![parameter, 0.5]));
            config.linear_system_structure = Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 };
        }
    }
    config
}

fn build_solver(
    route: HarnessRoute,
    phase: HarnessPhase,
    output_parent: &Path,
    parameter: f64,
    variant: HandoffVariant,
) -> Result<Lsode2Solver, String> {
    if let Some(toolchain) = route.aot_toolchain() {
        if matches!(phase, HarnessPhase::Warm | HarnessPhase::CallbackOnly) {
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
            let mut bootstrap = Lsode2Solver::new(apply_handoff_variant(
                trajectory_config(
                    handoff_structure(variant),
                    route
                        .aot_assembly()
                        .expect("AOT route should select a symbolic assembly backend"),
                    Lsode2SymbolicExecutionMode::Aot {
                        toolchain,
                        profile: Lsode2AotProfile::Debug,
                    },
                    Some(bootstrap_config),
                    IvpTelemetry::disabled(),
                ),
                variant,
                parameter,
            ))
            .map_err(|error| format!("warm bootstrap construction failed: {error}"))?;
            bootstrap
                .prepare()
                .map_err(|error| format!("warm bootstrap preparation failed: {error}"))?;
        }
    }
    let handoff_resolver = if matches!(
        phase,
        HarnessPhase::Consumer | HarnessPhase::ConsumerContinuation
    ) && route.aot_toolchain().is_some()
    {
        let handoff = output_parent.join(format!("{}.aot-handoff", route.as_str()));
        Some(AotResolver::read_handoff(&handoff).map_err(|error| {
            format!(
                "consumer handoff load failed at {}: {error}",
                handoff.display()
            )
        })?)
    } else {
        None
    };
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
        HarnessRoute::AotExprLegacy(toolchain) | HarnessRoute::AotAtomView(toolchain) => (
            route
                .aot_assembly()
                .expect("AOT route should select a symbolic assembly backend"),
            Lsode2SymbolicExecutionMode::Aot {
                toolchain,
                profile: Lsode2AotProfile::Debug,
            },
            Some(
                generated_config_with_resolver(
                    output_parent,
                    route,
                    match phase {
                        HarnessPhase::Cold | HarnessPhase::Producer => {
                            SymbolicIvpAotBuildPolicy::BuildIfMissing {
                                profile: AotBuildProfile::Debug,
                            }
                        }
                        HarnessPhase::Warm
                        | HarnessPhase::CallbackOnly
                        | HarnessPhase::Consumer
                        | HarnessPhase::ConsumerContinuation => {
                            SymbolicIvpAotBuildPolicy::RequirePrebuilt
                        }
                    },
                    handoff_resolver,
                )
                .with_handoff_path(
                    matches!(phase, HarnessPhase::Producer)
                        .then(|| output_parent.join("aot-registry-handoff")),
                ),
            ),
        ),
    };
    Lsode2Solver::new(apply_handoff_variant(
        trajectory_config(
            handoff_structure(variant),
            assembly,
            execution,
            generated,
            IvpTelemetry::detailed(),
        ),
        variant,
        parameter,
    ))
    .map_err(|error| error.to_string())
}

fn run_child_phase(
    route: HarnessRoute,
    phase: HarnessPhase,
    output_parent: &Path,
    repetitions: usize,
    parameter: f64,
    variant: HandoffVariant,
) -> Result<PhaseRecord, String> {
    let mut solver = build_solver(route, phase, output_parent, parameter, variant)?;
    let prepare_started = Instant::now();
    solver.prepare().map_err(|error| error.to_string())?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1.0e3;

    if matches!(phase, HarnessPhase::Producer) {
        let generic_handoff = output_parent.join("aot-registry-handoff");
        let handoff = output_parent.join(format!("{}.aot-handoff", route.as_str()));
        if !generic_handoff.exists() {
            return Err(format!(
                "producer did not publish an AOT handoff at {}",
                generic_handoff.display()
            ));
        }
        std::fs::copy(&generic_handoff, &handoff).map_err(|error| {
            format!(
                "producer handoff copy failed at {}: {error}",
                handoff.display()
            )
        })?;
    }

    let mut solve_ms = 0.0;
    let mut last_snapshot = None;
    let mut final_t = f64::NAN;
    let mut final_state_0 = f64::NAN;
    let mut reported_parameter = parameter;
    if phase == HarnessPhase::ConsumerContinuation {
        let rebound_parameter = parameter + 1.0;
        solver
            .set_parameter_values(DVector::from_vec(vec![rebound_parameter]))
            .map_err(|error| format!("consumer continuation rebind failed: {error}"))?;
        reported_parameter = rebound_parameter;
    }
    for _ in 0..repetitions.max(1) {
        let started = (phase != HarnessPhase::CallbackOnly).then(Instant::now);
        let summary = solver
            .solve_with_summary()
            .map_err(|error| error.to_string())?;
        final_t = summary.final_t.unwrap_or(f64::NAN);
        final_state_0 = summary
            .final_y
            .as_ref()
            .and_then(|state| state.get(0).copied())
            .unwrap_or(f64::NAN);
        if let Some(started) = started {
            solve_ms += started.elapsed().as_secs_f64() * 1.0e3;
        }
        last_snapshot = Some(solver.telemetry_snapshot());
    }
    let divisor = repetitions.max(1) as f64;
    let snapshot = last_snapshot.expect("at least one process-harness solve is required");
    let solver_stages_in_scope = phase != HarnessPhase::CallbackOnly;
    let scoped_stage_ms = |stage| {
        solver_stages_in_scope
            .then(|| stage_ms(&snapshot, stage, divisor))
            .unwrap_or(0.0)
    };
    let controller_ms = scoped_stage_ms(IvpWarmStage::Controller);
    let solve_telemetry_ms = scoped_stage_ms(IvpWarmStage::Solve);
    let callback_ms = stage_ms(&snapshot, IvpWarmStage::ResidualCallback, divisor)
        + stage_ms(&snapshot, IvpWarmStage::JacobianCallback, divisor);
    let aot_artifact_keys = if route.aot_toolchain().is_some() {
        snapshot.aot_artifact_keys.clone()
    } else {
        Vec::new()
    };
    let aot_link_ms = if route.aot_toolchain().is_some() {
        snapshot
            .cold_stage(IvpColdStage::AotLink)
            .elapsed
            .as_secs_f64()
            * 1.0e3
    } else {
        0.0
    };
    let aot_publication_ms = if route.aot_toolchain().is_some() {
        snapshot
            .cold_stage(IvpColdStage::AotPublication)
            .elapsed
            .as_secs_f64()
            * 1.0e3
    } else {
        0.0
    };
    let record = PhaseRecord {
        phase,
        route: route.as_str(),
        status: "ok",
        repetitions: repetitions.max(1),
        parameter: reported_parameter,
        prepare_ms,
        solve_ms: solve_ms / divisor,
        callback_ms,
        controller_ms,
        controller_iteration_ms: scoped_stage_ms(IvpWarmStage::ControllerIteration),
        solve_telemetry_ms,
        solve_minus_controller_ms: solve_ms / divisor - controller_ms,
        argument_binding_ms: stage_ms(&snapshot, IvpWarmStage::ArgumentBinding, divisor),
        residual_callback_ms: stage_ms(&snapshot, IvpWarmStage::ResidualCallback, divisor),
        jacobian_callback_ms: stage_ms(&snapshot, IvpWarmStage::JacobianCallback, divisor),
        residual_eval_ms: stage_ms(&snapshot, IvpWarmStage::ResidualEvaluation, divisor),
        jacobian_eval_ms: stage_ms(&snapshot, IvpWarmStage::JacobianEvaluation, divisor),
        residual_output_ms: stage_ms(&snapshot, IvpWarmStage::ResidualOutputAssembly, divisor),
        jacobian_output_ms: stage_ms(&snapshot, IvpWarmStage::JacobianOutputAssembly, divisor),
        factorization_ms: scoped_stage_ms(IvpWarmStage::Factorization),
        rhs_ms: scoped_stage_ms(IvpWarmStage::RhsSolve),
        native_engine_setup_ms: scoped_stage_ms(IvpWarmStage::NativeEngineSetup),
        native_result_assembly_ms: scoped_stage_ms(IvpWarmStage::NativeResultAssembly),
        final_t,
        final_state_0,
        residual_calls: snapshot.residual_evaluations as usize,
        jacobian_calls: snapshot.jacobian_evaluations as usize,
        linear_solves: snapshot.linear_solve_requests as usize,
        accepted_steps: snapshot.accepted_steps as usize,
        rejected_steps: snapshot.rejected_steps as usize,
        jacobian_rebuilds: snapshot.jacobian_rebuilds as usize,
        parameter_binds: snapshot.parameter_binds as usize,
        method_switches: snapshot.method_switches as usize,
        errors: snapshot.errors as usize,
        copies: snapshot.copies as usize,
        copied_bytes: snapshot.copied_bytes,
        allocated_bytes: snapshot.allocated_bytes,
        parallel_dispatches: snapshot.parallel_dispatches as usize,
        sequential_dispatches: snapshot.sequential_dispatches as usize,
        worker_count: snapshot.lambdify_worker_count,
        auto_min_work_per_job: snapshot.lambdify_auto_min_work_per_job,
        aot_chunk_dispatches: snapshot.aot_chunk_dispatches as usize,
        aot_parallel_dispatches: snapshot.aot_parallel_dispatches as usize,
        aot_chunks: snapshot.aot_chunks as usize,
        aot_worker_callbacks: snapshot.aot_worker_callbacks as usize,
        aot_resolution_hits: snapshot.aot_resolution_hits as usize,
        aot_resolution_misses: snapshot.aot_resolution_misses as usize,
        aot_reconnects: snapshot.aot_reconnects as usize,
        aot_build_attempts: snapshot.aot_build_attempts as usize,
        aot_link_attempts: snapshot.aot_link_attempts as usize,
        aot_link_ms,
        aot_publication_ms,
        aot_runtime_ready: snapshot.aot_runtime_ready as usize,
        aot_artifact_keys,
    };
    println!(
        "RST_AOT_HARNESS|phase={}|route={}|status={}|repetitions={}|parameter={:.17e}|prepare_ms={:.6}|solve_ms={:.6}|callback_ms={:.6}|aot_link_ms={:.6}|aot_publication_ms={:.6}|aot_runtime_ready={}|controller_ms={:.6}|controller_iteration_ms={:.6}|solve_telemetry_ms={:.6}|solve_minus_controller_ms={:.6}|argument_binding_ms={:.6}|residual_callback_ms={:.6}|jacobian_callback_ms={:.6}|residual_eval_ms={:.6}|jacobian_eval_ms={:.6}|residual_output_ms={:.6}|jacobian_output_ms={:.6}|factorization_ms={:.6}|rhs_ms={:.6}|native_engine_setup_ms={:.6}|native_result_assembly_ms={:.6}|final_t={:.17e}|final_state_0={:.17e}|residual_calls={}|jacobian_calls={}|linear_solves={}|accepted_steps={}|rejected_steps={}|jacobian_rebuilds={}|parameter_binds={}|method_switches={}|errors={}|copies={}|copied_bytes={}|allocated_bytes={}|parallel_dispatches={}|sequential_dispatches={}|worker_count={}|auto_min_work_per_job={}|aot_chunk_dispatches={}|aot_parallel_dispatches={}|aot_chunks={}|aot_worker_callbacks={}|aot_provenance={}|aot_resolution_hits={}|aot_resolution_misses={}|aot_reconnects={}|aot_build_attempts={}|aot_link_attempts={}|aot_artifact_keys={}",
        record.phase.as_str(),
        record.route,
        record.status,
        record.repetitions,
        record.parameter,
        record.prepare_ms,
        record.solve_ms,
        record.callback_ms,
        record.aot_link_ms,
        record.aot_publication_ms,
        record.aot_runtime_ready,
        record.controller_ms,
        record.controller_iteration_ms,
        record.solve_telemetry_ms,
        record.solve_minus_controller_ms,
        record.argument_binding_ms,
        record.residual_callback_ms,
        record.jacobian_callback_ms,
        record.residual_eval_ms,
        record.jacobian_eval_ms,
        record.residual_output_ms,
        record.jacobian_output_ms,
        record.factorization_ms,
        record.rhs_ms,
        record.native_engine_setup_ms,
        record.native_result_assembly_ms,
        record.final_t,
        record.final_state_0,
        record.residual_calls,
        record.jacobian_calls,
        record.linear_solves,
        record.accepted_steps,
        record.rejected_steps,
        record.jacobian_rebuilds,
        record.parameter_binds,
        record.method_switches,
        record.errors,
        record.copies,
        record.copied_bytes,
        record.allocated_bytes,
        record.parallel_dispatches,
        record.sequential_dispatches,
        record.worker_count,
        record.auto_min_work_per_job,
        record.aot_chunk_dispatches,
        record.aot_parallel_dispatches,
        record.aot_chunks,
        record.aot_worker_callbacks,
        aot_provenance(&record),
        record.aot_resolution_hits,
        record.aot_resolution_misses,
        record.aot_reconnects,
        record.aot_build_attempts,
        record.aot_link_attempts,
        record.aot_artifact_keys.join(","),
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
        parameter: parse_field(&values, "parameter")?,
        prepare_ms: parse_field(&values, "prepare_ms")?,
        solve_ms: parse_field(&values, "solve_ms")?,
        callback_ms: parse_field(&values, "callback_ms")?,
        controller_ms: parse_field(&values, "controller_ms")?,
        controller_iteration_ms: parse_field(&values, "controller_iteration_ms")?,
        solve_telemetry_ms: parse_field(&values, "solve_telemetry_ms")?,
        solve_minus_controller_ms: parse_field(&values, "solve_minus_controller_ms")?,
        argument_binding_ms: parse_field(&values, "argument_binding_ms")?,
        residual_callback_ms: parse_field(&values, "residual_callback_ms")?,
        jacobian_callback_ms: parse_field(&values, "jacobian_callback_ms")?,
        residual_eval_ms: parse_field(&values, "residual_eval_ms")?,
        jacobian_eval_ms: parse_field(&values, "jacobian_eval_ms")?,
        residual_output_ms: parse_field(&values, "residual_output_ms")?,
        jacobian_output_ms: parse_field(&values, "jacobian_output_ms")?,
        factorization_ms: parse_field(&values, "factorization_ms")?,
        rhs_ms: parse_field(&values, "rhs_ms")?,
        native_engine_setup_ms: parse_field(&values, "native_engine_setup_ms")?,
        native_result_assembly_ms: parse_field(&values, "native_result_assembly_ms")?,
        final_t: parse_field(&values, "final_t")?,
        final_state_0: parse_field(&values, "final_state_0")?,
        residual_calls: parse_field(&values, "residual_calls")?,
        jacobian_calls: parse_field(&values, "jacobian_calls")?,
        linear_solves: parse_field(&values, "linear_solves")?,
        accepted_steps: parse_field(&values, "accepted_steps")?,
        rejected_steps: parse_field(&values, "rejected_steps")?,
        jacobian_rebuilds: parse_field(&values, "jacobian_rebuilds")?,
        parameter_binds: parse_field(&values, "parameter_binds")?,
        method_switches: parse_field(&values, "method_switches")?,
        errors: parse_field(&values, "errors")?,
        copies: parse_field(&values, "copies")?,
        copied_bytes: parse_field(&values, "copied_bytes")?,
        allocated_bytes: parse_field(&values, "allocated_bytes")?,
        parallel_dispatches: parse_field(&values, "parallel_dispatches")?,
        sequential_dispatches: parse_field(&values, "sequential_dispatches")?,
        worker_count: parse_field(&values, "worker_count")?,
        auto_min_work_per_job: parse_field(&values, "auto_min_work_per_job")?,
        aot_chunk_dispatches: parse_field(&values, "aot_chunk_dispatches")?,
        aot_parallel_dispatches: parse_field(&values, "aot_parallel_dispatches")?,
        aot_chunks: parse_field(&values, "aot_chunks")?,
        aot_worker_callbacks: parse_field(&values, "aot_worker_callbacks")?,
        aot_resolution_hits: parse_field(&values, "aot_resolution_hits")?,
        aot_resolution_misses: parse_field(&values, "aot_resolution_misses")?,
        aot_reconnects: parse_field(&values, "aot_reconnects")?,
        aot_build_attempts: parse_field(&values, "aot_build_attempts")?,
        aot_link_attempts: parse_field(&values, "aot_link_attempts")?,
        aot_link_ms: parse_field(&values, "aot_link_ms")?,
        aot_publication_ms: parse_field(&values, "aot_publication_ms")?,
        aot_runtime_ready: parse_field(&values, "aot_runtime_ready")?,
        aot_artifact_keys: values
            .get("aot_artifact_keys")
            .copied()
            .unwrap_or_default()
            .split(',')
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
            .collect(),
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

fn harness_timeout_ms() -> u64 {
    std::env::var("LSODE2_AOT_HARNESS_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(120_000)
}

fn run_isolated(
    route: HarnessRoute,
    phase: HarnessPhase,
    output_parent: &Path,
    repetitions: usize,
    parameter: f64,
    variant: HandoffVariant,
) -> Result<PhaseRecord, HarnessFailure> {
    let executable =
        std::env::current_exe().map_err(|error| HarnessFailure::Spawn(error.to_string()))?;
    let mut child = Command::new(executable)
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("LSODE2_AOT_HARNESS_CHILD", "1")
        .env("LSODE2_AOT_HARNESS_PHASE", phase.as_str())
        .env("LSODE2_AOT_HARNESS_ROUTE", route.as_str())
        .env("LSODE2_AOT_HARNESS_OUTPUT", output_parent)
        .env("LSODE2_AOT_HARNESS_REPETITIONS", repetitions.to_string())
        .env("LSODE2_AOT_HARNESS_PARAMETER", format!("{parameter:.17e}"))
        .env("LSODE2_AOT_HARNESS_VARIANT", variant.as_str())
        .env("RAYON_NUM_THREADS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| HarnessFailure::Spawn(error.to_string()))?;
    let timeout_ms = harness_timeout_ms();
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(HarnessFailure::Timeout {
                    route: route.as_str(),
                    phase,
                    timeout_ms,
                });
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(HarnessFailure::Child {
                    route: route.as_str(),
                    phase,
                    details: format!("wait failure: {error}"),
                });
            }
        }
    };
    let output = child
        .wait_with_output()
        .map_err(|error| HarnessFailure::Child {
            route: route.as_str(),
            phase,
            details: format!("output collection failure after status {status}: {error}"),
        })?;
    if !status.success() {
        return Err(HarnessFailure::Child {
            route: route.as_str(),
            phase,
            details: format!(
                "status={status} stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
        });
    }
    parse_record(&String::from_utf8_lossy(&output.stdout)).map_err(HarnessFailure::Protocol)
}

fn run_parent_matrix(routes: &[HarnessRoute], repetitions: usize) {
    let output_parent = tempdir().expect("isolated AOT output directory should exist");
    reportln!(
        "[LSODE2 process-isolated harness] fixture=two-state-parameterized; matrix=Banded(kl=1,ku=1); thread_policy=1; repetitions={repetitions}; child_process_timing_excluded; callback_only=telemetry callback stages, solve_ms=0"
    );
    reportln!(
        "route | phase | provenance | artifact_keys | prepare_ms | solve_ms | callback_ms | link_ms | publication_ms | runtime_ready | residual_calls | jacobian_calls | linear_solves | resolution_hits | resolution_misses | reconnects | build_attempts | link_attempts | status"
    );
    reportln!(
        "[LSODE2 process-isolated telemetry] nested stages are diagnostic; callback_only zeros solver-only stages; solve_minus_controller_ms is a signed remainder, not an additive stage"
    );
    reportln!(
        "route | phase | controller_ms | iteration_ms | solve_telemetry_ms | solve_minus_controller_ms | binding_ms | residual_cb_ms | jacobian_cb_ms | residual_eval_ms | jacobian_eval_ms | residual_output_ms | jacobian_output_ms | factorization_ms | rhs_ms | engine_setup_ms | result_assembly_ms | copies | copied_bytes | allocated_bytes | chunks | chunk_dispatches | parallel_dispatches | worker_callbacks | workers | auto_min_work"
    );
    reportln!("[LSODE2 process-isolated counters] solver and callback counters are kept separate");
    reportln!(
        "route | phase | residual_calls | jacobian_calls | linear_solves | accepted | rejected | jacobian_rebuilds | parameter_binds | method_switches | errors | sequential_dispatches | parallel_dispatches | aot_parallel_dispatches | aot_resolution_hits | aot_resolution_misses | reconnects | build_attempts | link_attempts"
    );
    for &route in routes {
        if let Some(command) = route.command() {
            if !command_available(command) {
                reportln!(
                    "{} | skipped | unavailable | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | missing:{command}",
                    route.as_str(),
                );
                continue;
            }
        }
        let mut cold_artifact_keys: Option<Vec<String>> = None;
        let mut phase_records = Vec::new();
        for phase in [
            HarnessPhase::Cold,
            HarnessPhase::Warm,
            HarnessPhase::CallbackOnly,
        ] {
            reportln!(
                "[LSODE2 process-isolated progress] route={} phase={} timeout_ms={} started",
                route.as_str(),
                phase.as_str(),
                harness_timeout_ms(),
            );
            let record = run_isolated(
                route,
                phase,
                output_parent.path(),
                repetitions,
                2.0,
                HandoffVariant::Baseline,
            )
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
            assert!(record.aot_link_ms.is_finite());
            assert!(record.aot_publication_ms.is_finite());
            assert!(record.final_t.is_finite());
            assert!(record.final_state_0.is_finite());
            assert!(record.residual_calls > 0);
            assert!(record.jacobian_calls > 0);
            assert!(record.linear_solves > 0);
            assert_eq!(record.errors, 0);
            assert_eq!(record.repetitions, repetitions.max(1));
            if phase == HarnessPhase::CallbackOnly {
                assert_eq!(record.solve_ms, 0.0);
                assert_eq!(record.controller_ms, 0.0);
                assert_eq!(record.controller_iteration_ms, 0.0);
                assert_eq!(record.solve_telemetry_ms, 0.0);
                assert_eq!(record.solve_minus_controller_ms, 0.0);
                assert_eq!(record.factorization_ms, 0.0);
                assert_eq!(record.rhs_ms, 0.0);
                assert_eq!(record.native_engine_setup_ms, 0.0);
                assert_eq!(record.native_result_assembly_ms, 0.0);
            }
            if route.aot_toolchain().is_some() {
                assert!(
                    !record.aot_artifact_keys.is_empty(),
                    "AOT phase {} for {} did not report artifact provenance",
                    phase.as_str(),
                    route.as_str()
                );
                assert!(
                    record.aot_runtime_ready > 0,
                    "AOT phase {} for {} did not report runtime_ready",
                    phase.as_str(),
                    route.as_str()
                );
                match phase {
                    HarnessPhase::Cold => {
                        cold_artifact_keys = Some(record.aot_artifact_keys.clone())
                    }
                    HarnessPhase::Warm | HarnessPhase::CallbackOnly => {
                        let cold_keys = cold_artifact_keys.clone().unwrap_or_default();
                        assert!(
                            record
                                .aot_artifact_keys
                                .iter()
                                .all(|key| cold_keys.iter().any(|cold| cold == key)),
                            "AOT artifact provenance changed between cold and {} for {}",
                            phase.as_str(),
                            route.as_str()
                        );
                    }
                    HarnessPhase::Producer
                    | HarnessPhase::Consumer
                    | HarnessPhase::ConsumerContinuation => {
                        panic!("producer/consumer phases do not belong to the cold/warm matrix")
                    }
                }
            } else {
                assert!(record.aot_artifact_keys.is_empty());
                assert_eq!(record.aot_build_attempts, 0);
                assert_eq!(record.aot_link_attempts, 0);
                assert_eq!(record.aot_reconnects, 0);
                assert_eq!(record.aot_runtime_ready, 0);
            }
            phase_records.push(record.clone());
            reportln!(
                "{} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                record.route,
                record.phase.as_str(),
                aot_provenance(&record),
                if record.aot_artifact_keys.is_empty() {
                    "-".to_string()
                } else {
                    record.aot_artifact_keys.join(",")
                },
                record.prepare_ms,
                record.solve_ms,
                record.callback_ms,
                record.aot_link_ms,
                record.aot_publication_ms,
                record.aot_runtime_ready,
                record.residual_calls,
                record.jacobian_calls,
                record.linear_solves,
                record.aot_resolution_hits,
                record.aot_resolution_misses,
                record.aot_reconnects,
                record.aot_build_attempts,
                record.aot_link_attempts,
                record.status,
            );
            reportln!(
                "{} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                record.route,
                record.phase.as_str(),
                record.controller_ms,
                record.controller_iteration_ms,
                record.solve_telemetry_ms,
                record.solve_minus_controller_ms,
                record.argument_binding_ms,
                record.residual_callback_ms,
                record.jacobian_callback_ms,
                record.residual_eval_ms,
                record.jacobian_eval_ms,
                record.residual_output_ms,
                record.jacobian_output_ms,
                record.factorization_ms,
                record.rhs_ms,
                record.native_engine_setup_ms,
                record.native_result_assembly_ms,
                record.copies,
                record.copied_bytes,
                record.allocated_bytes,
                record.aot_chunks,
                record.aot_chunk_dispatches,
                record.parallel_dispatches,
                record.aot_worker_callbacks,
                record.worker_count,
                record.auto_min_work_per_job,
            );
            reportln!(
                "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                record.route,
                record.phase.as_str(),
                record.residual_calls,
                record.jacobian_calls,
                record.linear_solves,
                record.accepted_steps,
                record.rejected_steps,
                record.jacobian_rebuilds,
                record.parameter_binds,
                record.method_switches,
                record.errors,
                record.sequential_dispatches,
                record.parallel_dispatches,
                record.aot_parallel_dispatches,
                record.aot_resolution_hits,
                record.aot_resolution_misses,
                record.aot_reconnects,
                record.aot_build_attempts,
                record.aot_link_attempts,
            );
        }
        let cold = &phase_records[0];
        for phase in &phase_records[1..] {
            assert_eq!(phase.residual_calls, cold.residual_calls);
            assert_eq!(phase.jacobian_calls, cold.jacobian_calls);
            assert_eq!(phase.linear_solves, cold.linear_solves);
            assert_eq!(phase.accepted_steps, cold.accepted_steps);
            assert_eq!(phase.rejected_steps, cold.rejected_steps);
            assert_eq!(phase.jacobian_rebuilds, cold.jacobian_rebuilds);
            assert_eq!(phase.parameter_binds, cold.parameter_binds);
            assert_eq!(phase.method_switches, cold.method_switches);
            assert!((phase.final_t - cold.final_t).abs() <= 1.0e-12);
            assert!((phase.final_state_0 - cold.final_state_0).abs() <= 1.0e-10);
        }
    }
}

fn run_parent_parameter_handoff(routes: &[HarnessRoute]) {
    let output_parent = tempdir().expect("parameter handoff output directory should exist");
    reportln!(
        "[LSODE2 producer/consumer handoff] fixture=two-state-parameterized; producer_parameter=2.0; consumer_parameter=3.0; matrix=Banded(kl=1,ku=1); process_isolated=true"
    );
    reportln!(
        "route | producer_key | consumer_key | producer_prepare_ms | producer_link_ms | producer_publication_ms | consumer_prepare_ms | consumer_link_ms | consumer_publication_ms | consumer_hits | consumer_reconnects | consumer_build_attempts | consumer_link_attempts | reference_diff | status"
    );
    reportln!(
        "[LSODE2 producer/consumer telemetry] phase rows are per-process; preparation is outside solve_ms"
    );
    reportln!(
        "route | phase | prepare_ms | solve_ms | callback_ms | link_ms | publication_ms | binding_ms | residual_cb_ms | jacobian_cb_ms | residual_eval_ms | jacobian_eval_ms | factorization_ms | rhs_ms | copies | copied_bytes | allocated_bytes | errors"
    );
    reportln!(
        "route | phase | residual_calls | jacobian_calls | linear_solves | accepted | rejected | jacobian_rebuilds | parameter_binds | build_attempts | link_attempts | resolution_hits | reconnects"
    );
    for &route in routes {
        let producer = run_isolated(
            route,
            HarnessPhase::Producer,
            output_parent.path(),
            1,
            2.0,
            HandoffVariant::Baseline,
        )
        .unwrap_or_else(|error| panic!("producer {}: {error}", route.as_str()));
        let consumer = run_isolated(
            route,
            HarnessPhase::Consumer,
            output_parent.path(),
            1,
            3.0,
            HandoffVariant::Baseline,
        )
        .unwrap_or_else(|error| panic!("consumer {}: {error}", route.as_str()));
        let reference = run_isolated(
            HarnessRoute::LambdifyAtomViewNative,
            HarnessPhase::Consumer,
            output_parent.path(),
            1,
            3.0,
            HandoffVariant::Baseline,
        )
        .unwrap_or_else(|error| panic!("parameter reference: {error}"));

        assert_eq!(producer.status, "ok");
        assert_eq!(consumer.status, "ok");
        assert_eq!(producer.parameter, 2.0);
        assert_eq!(consumer.parameter, 3.0);
        for (label, record) in [("producer", &producer), ("consumer", &consumer)] {
            assert!(
                record.final_t.is_finite(),
                "{label} final time is not finite"
            );
            assert!(
                record.final_state_0.is_finite(),
                "{label} final state is not finite"
            );
            assert!(record.solve_ms.is_finite());
            assert!(record.callback_ms.is_finite());
            assert!(record.residual_calls > 0);
            assert!(record.jacobian_calls > 0);
            assert!(record.linear_solves > 0);
            assert_eq!(record.errors, 0);
            if label == "producer" {
                assert!(record.aot_build_attempts > 0);
            } else {
                assert_eq!(record.aot_build_attempts, 0);
            }
        }
        assert!(!consumer.aot_artifact_keys.is_empty());
        assert!(
            consumer
                .aot_artifact_keys
                .iter()
                .all(|key| producer.aot_artifact_keys.iter().any(|known| known == key)),
            "consumer resolved an artifact key that producer did not publish"
        );
        assert_eq!(consumer.aot_build_attempts, 0);
        // Link-attempt accounting is implementation-specific: a consumer may
        // reconnect an already registered process-local runtime without
        // incrementing that counter. Reconnect and zero builds are the stable
        // continuation gates; link attempts remain diagnostic output.
        assert!(consumer.aot_resolution_hits > 0);
        assert!(consumer.aot_reconnects > 0);
        assert!((consumer.final_t - reference.final_t).abs() <= 1.0e-10);
        assert!((consumer.final_state_0 - reference.final_state_0).abs() <= 1.0e-9);
        assert_eq!(consumer.residual_calls, reference.residual_calls);
        assert_eq!(consumer.jacobian_calls, reference.jacobian_calls);
        assert_eq!(consumer.linear_solves, reference.linear_solves);
        assert_eq!(consumer.accepted_steps, reference.accepted_steps);
        assert_eq!(consumer.rejected_steps, reference.rejected_steps);
        assert_eq!(consumer.jacobian_rebuilds, reference.jacobian_rebuilds);
        assert_eq!(consumer.parameter_binds, reference.parameter_binds);
        assert_eq!(consumer.method_switches, reference.method_switches);
        reportln!(
            "{} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {:.3e} | ok",
            route.as_str(),
            producer.aot_artifact_keys.join(","),
            consumer.aot_artifact_keys.join(","),
            producer.prepare_ms,
            producer.aot_link_ms,
            producer.aot_publication_ms,
            consumer.prepare_ms,
            consumer.aot_link_ms,
            consumer.aot_publication_ms,
            consumer.aot_resolution_hits,
            consumer.aot_reconnects,
            consumer.aot_build_attempts,
            consumer.aot_link_attempts,
            (consumer.final_state_0 - reference.final_state_0).abs(),
        );
        for (phase, record) in [("producer", &producer), ("consumer", &consumer)] {
            reportln!(
                "{} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {}",
                route.as_str(),
                phase,
                record.prepare_ms,
                record.solve_ms,
                record.callback_ms,
                record.aot_link_ms,
                record.aot_publication_ms,
                record.argument_binding_ms,
                record.residual_callback_ms,
                record.jacobian_callback_ms,
                record.residual_eval_ms,
                record.jacobian_eval_ms,
                record.factorization_ms,
                record.rhs_ms,
                record.copies,
                record.copied_bytes,
                record.allocated_bytes,
                record.errors,
            );
            reportln!(
                "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                route.as_str(),
                phase,
                record.residual_calls,
                record.jacobian_calls,
                record.linear_solves,
                record.accepted_steps,
                record.rejected_steps,
                record.jacobian_rebuilds,
                record.parameter_binds,
                record.aot_build_attempts,
                record.aot_link_attempts,
                record.aot_resolution_hits,
                record.aot_reconnects,
            );
        }
    }
}

fn run_parent_parameter_continuation(routes: &[HarnessRoute]) {
    let output_parent = tempdir().expect("parameter continuation output directory should exist");
    reportln!(
        "[LSODE2 producer/consumer continuation] producer_parameter=2.0; consumer_initial=3.0; consumer_rebound=4.0; matrix=Banded(kl=1,ku=1); process_isolated=true"
    );
    reportln!(
        "route | producer_key | consumer_key | consumer_parameter | consumer_prepare_ms | consumer_rebinds | consumer_build_attempts | consumer_reconnects | reference_diff | status"
    );
    reportln!(
        "[LSODE2 producer/consumer continuation telemetry] producer and rebound consumer are separate processes; preparation is outside solve_ms"
    );
    reportln!(
        "route | phase | prepare_ms | solve_ms | callback_ms | link_ms | publication_ms | runtime_ready | binding_ms | residual_cb_ms | jacobian_cb_ms | residual_eval_ms | jacobian_eval_ms | factorization_ms | rhs_ms | copies | copied_bytes | allocated_bytes | errors"
    );
    reportln!(
        "route | phase | residual_calls | jacobian_calls | linear_solves | accepted | rejected | jacobian_rebuilds | parameter_binds | build_attempts | link_attempts | resolution_hits | reconnects"
    );

    for &route in routes {
        let producer = run_isolated(
            route,
            HarnessPhase::Producer,
            output_parent.path(),
            1,
            2.0,
            HandoffVariant::Baseline,
        )
        .unwrap_or_else(|error| panic!("continuation producer {}: {error}", route.as_str()));
        let consumer = run_isolated(
            route,
            HarnessPhase::ConsumerContinuation,
            output_parent.path(),
            1,
            3.0,
            HandoffVariant::Baseline,
        )
        .unwrap_or_else(|error| panic!("continuation consumer {}: {error}", route.as_str()));
        let reference = run_isolated(
            HarnessRoute::LambdifyAtomViewNative,
            HarnessPhase::Consumer,
            output_parent.path(),
            1,
            4.0,
            HandoffVariant::Baseline,
        )
        .unwrap_or_else(|error| panic!("continuation reference: {error}"));

        assert_eq!(producer.status, "ok");
        assert_eq!(consumer.status, "ok");
        assert_eq!(consumer.parameter, 4.0);
        assert!(!consumer.aot_artifact_keys.is_empty());
        assert!(
            consumer
                .aot_artifact_keys
                .iter()
                .all(|key| producer.aot_artifact_keys.iter().any(|known| known == key)),
            "consumer resolved an artifact key that producer did not publish"
        );
        assert_eq!(consumer.aot_build_attempts, 0);
        assert_eq!(consumer.parameter_binds, 1);
        assert!(consumer.aot_resolution_hits > 0);
        assert!(consumer.aot_reconnects > 0);
        assert!(producer.aot_runtime_ready > 0);
        assert!(consumer.aot_runtime_ready > 0);
        assert_eq!(consumer.errors, 0);
        assert!((consumer.final_t - reference.final_t).abs() <= 1.0e-10);
        let reference_diff = (consumer.final_state_0 - reference.final_state_0).abs();
        assert!(
            reference_diff <= 1.0e-9,
            "continuation drift: {reference_diff:e}"
        );
        assert_eq!(consumer.residual_calls, reference.residual_calls);
        assert_eq!(consumer.jacobian_calls, reference.jacobian_calls);
        assert_eq!(consumer.linear_solves, reference.linear_solves);
        assert_eq!(consumer.accepted_steps, reference.accepted_steps);
        assert_eq!(consumer.rejected_steps, reference.rejected_steps);
        assert_eq!(consumer.jacobian_rebuilds, reference.jacobian_rebuilds);

        reportln!(
            "{} | {} | {} | {:.1} | {:.3} | {} | {} | {} | {:.3e} | ok",
            route.as_str(),
            producer.aot_artifact_keys.join(","),
            consumer.aot_artifact_keys.join(","),
            consumer.parameter,
            consumer.prepare_ms,
            consumer.parameter_binds,
            consumer.aot_build_attempts,
            consumer.aot_reconnects,
            reference_diff,
        );
        for (phase, record) in [("producer", &producer), ("consumer_rebound", &consumer)] {
            reportln!(
                "{} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {}",
                route.as_str(),
                phase,
                record.prepare_ms,
                record.solve_ms,
                record.callback_ms,
                record.aot_link_ms,
                record.aot_publication_ms,
                record.aot_runtime_ready,
                record.argument_binding_ms,
                record.residual_callback_ms,
                record.jacobian_callback_ms,
                record.residual_eval_ms,
                record.jacobian_eval_ms,
                record.factorization_ms,
                record.rhs_ms,
                record.copies,
                record.copied_bytes,
                record.allocated_bytes,
                record.errors,
            );
            reportln!(
                "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                route.as_str(),
                phase,
                record.residual_calls,
                record.jacobian_calls,
                record.linear_solves,
                record.accepted_steps,
                record.rejected_steps,
                record.jacobian_rebuilds,
                record.parameter_binds,
                record.aot_build_attempts,
                record.aot_link_attempts,
                record.aot_resolution_hits,
                record.aot_reconnects,
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
            HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CTcc),
            HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc),
        ],
        2,
    );
}

#[test]
fn aot_process_isolated_producer_consumer_parameter_handoff() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_process_harness_story_tests::aot_process_isolated_producer_consumer_parameter_handoff",
    );
    run_parent_parameter_handoff(&[
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CTcc),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc),
    ]);
}

#[test]
fn aot_process_isolated_parameter_continuation_reuses_producer_artifact() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_process_harness_story_tests::aot_process_isolated_parameter_continuation_reuses_producer_artifact",
    );
    run_parent_parameter_continuation(&[
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CTcc),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc),
    ]);
}

#[test]
#[ignore = "release process-isolated parameter continuation matrix across all AOT toolchains"]
fn aot_process_isolated_release_parameter_continuation_matrix() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_process_harness_story_tests::aot_process_isolated_release_parameter_continuation_matrix",
    );
    reportln!(
        "[LSODE2 producer/consumer continuation] release matrix; all AOT toolchains; producer artifact must be reused after numeric parameter rebind"
    );
    run_parent_parameter_continuation(&[
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::Rust),
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CTcc),
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CGcc),
        HarnessRoute::AotExprLegacy(Lsode2AotToolchain::Zig),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::Rust),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::CGcc),
        HarnessRoute::AotAtomView(Lsode2AotToolchain::Zig),
    ]);
}

#[test]
fn aot_process_isolated_handoff_rejects_schema_layout_and_jacobian_pattern_changes() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_process_harness_story_tests::aot_process_isolated_handoff_rejects_schema_layout_and_jacobian_pattern_changes",
    );
    reportln!(
        "[LSODE2 producer/consumer invalidation] two-state baseline artifact must be rejected after schema, layout or Jacobian-pattern mutation"
    );
    reportln!("mutation | producer_key | consumer_result | status");

    for variant in [
        HandoffVariant::ParameterSchema,
        HandoffVariant::Layout,
        HandoffVariant::JacobianPattern,
    ] {
        let output_parent = tempdir().expect("invalidation output directory should exist");
        let route = HarnessRoute::AotAtomView(Lsode2AotToolchain::CTcc);
        let producer = run_isolated(
            route,
            HarnessPhase::Producer,
            output_parent.path(),
            1,
            2.0,
            HandoffVariant::Baseline,
        )
        .unwrap_or_else(|error| panic!("producer for {}: {error}", variant.as_str()));
        assert_eq!(producer.status, "ok");
        assert!(!producer.aot_artifact_keys.is_empty());

        let rejection = run_isolated(
            route,
            HarnessPhase::Consumer,
            output_parent.path(),
            1,
            3.0,
            variant,
        );
        let details = match rejection {
            Err(HarnessFailure::Child { details, .. }) => details,
            Err(other) => panic!(
                "{} mutation did not reach typed child rejection: {other}",
                variant.as_str()
            ),
            Ok(record) => panic!(
                "{} mutation unexpectedly executed with status {}; artifact_keys={}",
                variant.as_str(),
                record.status,
                record.aot_artifact_keys.join(",")
            ),
        };
        let normalized = details.to_ascii_lowercase();
        assert!(
            normalized.contains("aot")
                || normalized.contains("artifact")
                || normalized.contains("prebuilt")
                || normalized.contains("resolver"),
            "{} rejection did not expose an AOT lifecycle diagnostic: {details}",
            variant.as_str()
        );
        reportln!(
            "{} | {} | {} | rejected",
            variant.consumer_label(),
            producer.aot_artifact_keys.join(","),
            details.replace('\n', " ").replace('|', "/"),
        );
    }
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
            HarnessRoute::AotExprLegacy(Lsode2AotToolchain::Rust),
            HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CTcc),
            HarnessRoute::AotExprLegacy(Lsode2AotToolchain::CGcc),
            HarnessRoute::AotExprLegacy(Lsode2AotToolchain::Zig),
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
    let parameter = std::env::var("LSODE2_AOT_HARNESS_PARAMETER")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2.0);
    let variant = std::env::var("LSODE2_AOT_HARNESS_VARIANT")
        .ok()
        .and_then(|value| HandoffVariant::parse(&value))
        .unwrap_or(HandoffVariant::Baseline);
    run_child_phase(
        route,
        phase,
        &output_parent,
        repetitions,
        parameter,
        variant,
    )
    .unwrap_or_else(|error| panic!("child harness phase failed: {error}"));
}
