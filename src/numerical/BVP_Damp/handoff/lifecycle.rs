fn bvp_aot_lifecycle_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn insert_elapsed_ms(diagnostics: &mut HashMap<String, String>, key: &str, began: Instant) {
    diagnostics.insert(
        key.to_string(),
        format!("{:.6}", began.elapsed().as_secs_f64() * 1_000.0),
    );
}

fn insert_aot_stage_elapsed_ms(
    diagnostics: &mut HashMap<String, String>,
    key: &str,
    began: Instant,
    prepared: &crate::symbolic::bvp::legacy::BvpPreparedSparseAotProblem,
) {
    let elapsed = began.elapsed();
    diagnostics.insert(
        key.to_string(),
        format!("{:.6}", elapsed.as_secs_f64() * 1_000.0),
    );
    let stage = match key {
        "generated.aot.materialize_ms" => Some(BvpAotColdStage::Materialization),
        "generated.aot.compile_link_ms" => Some(BvpAotColdStage::Build),
        "generated.aot.register_link_ms" => Some(BvpAotColdStage::Link),
        _ => None,
    };
    if let Some(stage) = stage {
        prepared.record_aot_cold_stage_duration(stage, elapsed);
    }
}

fn copy_prepare_diagnostics(
    target: &mut HashMap<String, String>,
    bundle: &BvpSparseSolverBundle,
    phase: &str,
) {
    for (key, value) in bundle.runtime_diagnostics() {
        if let Some(stage) = key.strip_prefix("generated.prepare.") {
            target.insert(format!("generated.handoff.{phase}.{stage}"), value.clone());
        }
    }
}

fn append_aot_artifact_breakdown(
    diagnostics: &mut HashMap<String, String>,
    breakdown: &BvpGeneratedAotCrateBreakdown,
) {
    for (key, value) in [
        ("module_ms", breakdown.module_build_ms),
        ("module_init_ms", breakdown.prepared_module_init_ms),
        ("residual_lower_ms", breakdown.prepared_residual_blocks_ms),
        ("jacobian_lower_ms", breakdown.prepared_jacobian_blocks_ms),
        ("source_emit_ms", breakdown.language_source_emit_ms),
        ("c_header_ms", breakdown.c_header_emit_ms),
        ("packaging_ms", breakdown.artifact_packaging_ms),
    ] {
        diagnostics.insert(
            format!("generated.aot.artifact.{key}"),
            format!("{value:.6}"),
        );
    }
}

fn append_typed_aot_telemetry(
    diagnostics: &mut HashMap<String, String>,
    prepared: &crate::symbolic::bvp::legacy::BvpPreparedSparseAotProblem,
) {
    let Some(snapshot) = prepared.aot_telemetry_snapshot() else {
        return;
    };
    snapshot.append_compatibility_diagnostics(diagnostics);
}

/// Sends lifecycle transitions to the typed telemetry owned by the prepared
/// AtomView plan.  The presentation diagnostics map remains a boundary
/// concern; no lifecycle event is stored as a string in the prepared runtime.
fn record_bundle_aot_lifecycle(bundle: &BvpSparseSolverBundle, event: BvpAotLifecycleEvent) {
    if let Some(plan) = bundle
        .execution
        .selected()
        .prepared_problem
        .atom_aot_plan
        .as_ref()
    {
        plan.record_lifecycle(event);
    }
}

/// Records the failure side of the same common lifecycle stream used for
/// successful builds.  The common codegen layer retains the detailed typed
/// diagnostics; this adapter only publishes the small event classification
/// into the prepared plan without parsing language-specific compiler text.
fn record_bundle_aot_failure(bundle: &BvpSparseSolverBundle, error: &BvpBackendIntegrationError) {
    let is_link_failure = matches!(
        error,
        BvpBackendIntegrationError::AotLifecycleFailure { diagnostics }
            if matches!(
                diagnostics.stage,
                crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleStage::Link
            )
    );
    record_bundle_aot_lifecycle(
        bundle,
        if is_link_failure {
            BvpAotLifecycleEvent::LinkFailed
        } else {
            BvpAotLifecycleEvent::BuildFailed
        },
    );
    if matches!(
        error,
        BvpBackendIntegrationError::AotLifecycleFailure { diagnostics }
            if diagnostics.quarantine_attempted
    ) {
        record_bundle_aot_lifecycle(bundle, BvpAotLifecycleEvent::Quarantined);
    }
}

fn append_registered_artifact_contract(
    diagnostics: &mut HashMap<String, String>,
    artifact: &RegisteredAotArtifact,
) {
    diagnostics.insert(
        "generated.aot.artifact.problem_key".to_string(),
        artifact.problem_key.clone(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_key".to_string(),
        artifact.manifest_problem_key(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_key_matches".to_string(),
        artifact.manifest_key_matches().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_file".to_string(),
        artifact.manifest_file.display().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_file_exists".to_string(),
        artifact.manifest_file_exists().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_cdylib".to_string(),
        artifact.expected_cdylib.display().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_cdylib_exists".to_string(),
        artifact.expected_cdylib.exists().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_rlib".to_string(),
        artifact.expected_rlib.display().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_rlib_exists".to_string(),
        artifact.expected_rlib.exists().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.contract_issues".to_string(),
        artifact.lifecycle_contract_issues().join(" | "),
    );
}

fn is_missing_aot_toolchain_failure(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    low.contains("program not found")
        || low.contains("os error 2")
        || low.contains("no such file or directory")
        || low.contains("the system cannot find the file specified")
        || low.contains("is not recognized as an internal or external command")
}

fn is_transient_aot_infra_failure(text: &str) -> bool {
    if is_missing_aot_toolchain_failure(text) {
        return false;
    }
    let low = text.to_ascii_lowercase();
    low.contains("permission denied")
        || low.contains("access is denied")
        || low.contains("being used by another process")
        || low.contains("resource busy")
        || low.contains("temporarily unavailable")
        || low.contains("failed to spawn")
        || low.contains("failed to spawn build runner")
        || low.contains("could not write")
        || low.contains("file is locked")
        || low.contains("sharing violation")
}

fn retry_exhausted_aot_message(
    build_context: &str,
    attempts: usize,
    detail: &str,
    transient: bool,
) -> String {
    let missing_toolchain = is_missing_aot_toolchain_failure(detail);
    let class = if missing_toolchain {
        "missing or unreachable toolchain"
    } else if transient {
        "transient infrastructure failure"
    } else {
        "deterministic build failure"
    };
    let toolchain_hint = if missing_toolchain {
        "The selected compiler/toolchain could not be spawned; check PATH, explicit compiler overrides, and the requested AOT backend.\n"
    } else {
        ""
    };
    format!(
        "BVP AOT build execution failed ({build_context}) after {attempts} attempt(s); classified as {class}. \
If this is a transient infrastructure failure, check stale compiler processes, file locks and antivirus/indexer interference; \
otherwise inspect compiler stdout/stderr below.\n\
{toolchain_hint}\
detail:\n{detail}"
    )
}

fn execute_aot_build_with_retry<F>(mut execute: F, build_context: &str) -> Result<(), String>
where
    F: FnMut() -> Result<(bool, Option<i32>, String, String), String>,
{
    const MAX_ATTEMPTS: usize = 3;
    let mut last_failure: Option<String> = None;
    let mut last_transient = false;
    let mut attempts_used = 0usize;

    for attempt in 1..=MAX_ATTEMPTS {
        attempts_used = attempt;
        match execute() {
            Ok((true, _, _, _)) => return Ok(()),
            Ok((false, status, stdout, stderr)) => {
                let detail = format!("status={status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}");
                let transient = is_transient_aot_infra_failure(&detail);
                last_transient = transient;
                last_failure = Some(detail);
                if transient && attempt < MAX_ATTEMPTS {
                    sleep(Duration::from_millis((attempt as u64) * 120));
                    continue;
                }
            }
            Err(err) => {
                let transient = is_transient_aot_infra_failure(&err);
                last_transient = transient;
                last_failure = Some(err);
                if transient && attempt < MAX_ATTEMPTS {
                    sleep(Duration::from_millis((attempt as u64) * 120));
                    continue;
                }
            }
        }
        break;
    }

    let detail = last_failure.unwrap_or_else(|| "unknown build failure".to_string());
    Err(retry_exhausted_aot_message(
        build_context,
        attempts_used,
        &detail,
        last_transient,
    ))
}
