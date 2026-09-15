//! High-level generated-backend orchestration for nonlinear symbolic problems.
//!
//! This module is the user-facing layer above:
//! - symbolic problem setup,
//! - backend selection,
//! - dense AOT lifecycle materialization/build,
//! - and resolver reuse.
//!
//! It gives `Nonlinear_systems` the same kind of ergonomic surface that the
//! newer BVP stack already has: callers can choose a simple mode such as
//! "defaults", "require prebuilt", or "build if missing" without manually
//! stitching together lifecycle layers.

use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::numerical::Nonlinear_systems::symbolic::{
    SymbolicArtifactAction, SymbolicArtifactPolicy, SymbolicBackendConfig, SymbolicBackendKind,
    SymbolicDenseAotOptions, SymbolicNonlinearProblem, SymbolicPreparationReport,
    SymbolicProblemOptions,
};
use crate::numerical::Nonlinear_systems::symbolic_aot::materialize_symbolic_nonlinear_aot_build_with_compile_config;
use crate::numerical::Nonlinear_systems::symbolic_backend::{
    SelectedSymbolicNonlinearBackendKind, SymbolicBackendSelectionPolicy,
    select_symbolic_nonlinear_backend,
};
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::register_generated_dense_cdylib_backend;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile, AotBuildResult, AotCompileConfig,
};
use crate::symbolic::symbolic_engine::Expr;
use fs2::FileExt;
use log::{info, warn};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const NONLINEAR_AOT_FILE_LOCK_TIMEOUT: Duration = Duration::from_secs(300);
const NONLINEAR_AOT_FILE_LOCK_POLL: Duration = Duration::from_millis(50);
const NONLINEAR_AOT_READY_MARKER: &str = ".nonlinear-aot-ready";

fn nonlinear_aot_lifecycle_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Process-shared advisory lock for one generated nonlinear artifact path.
///
/// The lock file is intentionally retained after unlock. Removing it would
/// create a pathname race: a waiter could still hold the old inode while a
/// third process creates a new file with the same name. OS file locks are
/// released automatically when the owning process exits.
#[derive(Debug)]
struct NonlinearAotFileLock {
    file: File,
}

impl Drop for NonlinearAotFileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn nonlinear_aot_file_lock_path(output_parent: &Path, problem_key: &str) -> PathBuf {
    output_parent.join(format!(".nonlinear-aot-{problem_key}.lifecycle.lock"))
}

fn acquire_nonlinear_aot_file_lock(
    output_parent: &Path,
    problem_key: &str,
    timeout: Duration,
) -> Result<NonlinearAotFileLock, SolveError> {
    fs::create_dir_all(output_parent).map_err(|error| {
        SolveError::AotBuildFailed(format!(
            "cannot create nonlinear AOT lifecycle lock directory '{}': {error}",
            output_parent.display()
        ))
    })?;
    let lock_path = nonlinear_aot_file_lock_path(output_parent, problem_key);
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|error| {
            SolveError::AotBuildFailed(format!(
                "cannot open nonlinear AOT lifecycle lock '{}': {error}",
                lock_path.display()
            ))
        })?;
    let started = Instant::now();
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => {
                let mut file = file;
                file.set_len(0).map_err(|error| {
                    SolveError::AotBuildFailed(format!(
                        "cannot initialize nonlinear AOT lifecycle lock '{}': {error}",
                        lock_path.display()
                    ))
                })?;
                file.seek(SeekFrom::Start(0)).map_err(|error| {
                    SolveError::AotBuildFailed(format!(
                        "cannot seek nonlinear AOT lifecycle lock '{}': {error}",
                        lock_path.display()
                    ))
                })?;
                writeln!(file, "pid={} problem_key={problem_key}", std::process::id())
                    .and_then(|_| file.flush())
                    .map_err(|error| {
                        SolveError::AotBuildFailed(format!(
                            "cannot write nonlinear AOT lifecycle lock '{}': {error}",
                            lock_path.display()
                        ))
                    })?;
                return Ok(NonlinearAotFileLock { file });
            }
            Err(error) if is_nonlinear_aot_lock_contention(&error) => {
                if started.elapsed() >= timeout {
                    return Err(SolveError::AotBuildFailed(format!(
                        "timed out after {:?} waiting for nonlinear AOT lifecycle lock '{}' (problem key {})",
                        timeout,
                        lock_path.display(),
                        problem_key
                    )));
                }
                thread::sleep(NONLINEAR_AOT_FILE_LOCK_POLL);
            }
            Err(error) => {
                return Err(SolveError::AotBuildFailed(format!(
                    "cannot acquire nonlinear AOT lifecycle lock '{}' (problem key {}): {error}",
                    lock_path.display(),
                    problem_key
                )));
            }
        }
    }
}

fn is_nonlinear_aot_lock_contention(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::WouldBlock {
        return true;
    }
    #[cfg(windows)]
    {
        return matches!(error.raw_os_error(), Some(32) | Some(33));
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn nonlinear_aot_ready_marker_path(build: &AotBuildResult) -> PathBuf {
    build.written.crate_dir.join(NONLINEAR_AOT_READY_MARKER)
}

/// Computes a small deterministic fingerprint for an already materialized DLL.
///
/// This is an identity check for the local lifecycle marker, not a security
/// hash. It catches truncated or overwritten output without adding a crypto
/// dependency to the solver's AOT hot path.
fn nonlinear_aot_file_fingerprint(path: &Path) -> io::Result<(u64, u64)> {
    let mut file = File::open(path)?;
    let mut length = 0_u64;
    let mut hash = 14_695_981_039_346_656_037_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        length = length.saturating_add(read as u64);
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }
    Ok((length, hash))
}

fn nonlinear_aot_ready_marker_matches(build: &AotBuildResult, problem_key: &str) -> bool {
    let Ok((artifact_len, artifact_hash)) = nonlinear_aot_file_fingerprint(&build.expected_cdylib)
    else {
        return false;
    };
    let marker = nonlinear_aot_ready_marker_path(build);
    let Ok(contents) = fs::read_to_string(marker) else {
        return false;
    };
    let expected_cdylib = build.expected_cdylib.to_string_lossy();
    let mut marker_key = None;
    let mut marker_cdylib = None;
    let mut marker_len = None;
    let mut marker_hash = None;
    for line in contents.lines() {
        if let Some(value) = line.strip_prefix("problem_key=") {
            marker_key = Some(value);
        } else if let Some(value) = line.strip_prefix("expected_cdylib=") {
            marker_cdylib = Some(value);
        } else if let Some(value) = line.strip_prefix("artifact_len=") {
            marker_len = value.parse::<u64>().ok();
        } else if let Some(value) = line.strip_prefix("artifact_hash=") {
            marker_hash = u64::from_str_radix(value, 16).ok();
        }
    }
    marker_key == Some(problem_key)
        && marker_cdylib == Some(expected_cdylib.as_ref())
        && marker_len == Some(artifact_len)
        && marker_hash == Some(artifact_hash)
}

fn write_nonlinear_aot_ready_marker(build: &AotBuildResult, problem_key: &str) -> io::Result<()> {
    let (artifact_len, artifact_hash) = nonlinear_aot_file_fingerprint(&build.expected_cdylib)?;
    let marker = nonlinear_aot_ready_marker_path(build);
    let temporary = marker.with_extension(format!("tmp-{}", std::process::id()));
    let contents = format!(
        "problem_key={problem_key}\nexpected_cdylib={}\nartifact_len={artifact_len}\nartifact_hash={artifact_hash:016x}\n",
        build.expected_cdylib.display(),
    );
    fs::write(&temporary, contents)?;
    if marker.exists() {
        fs::remove_file(&marker)?;
    }
    fs::rename(temporary, marker)
}

/// Invalidates publication metadata before replacing an artifact.
///
/// The generated DLL is deliberately left in place: a process may already
/// have it loaded, and deleting a live library would turn a recoverable build
/// failure into a runtime failure. Removing the ready marker is sufficient to
/// quarantine the output from `RequirePrebuilt` until a complete replacement
/// has been built and fingerprinted.
fn invalidate_nonlinear_aot_ready_marker(build: &AotBuildResult) -> io::Result<bool> {
    let marker = nonlinear_aot_ready_marker_path(build);
    if marker.exists() {
        fs::remove_file(marker)?;
        return Ok(true);
    }
    Ok(false)
}

/// High-level dense nonlinear generated-backend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseGeneratedBackendMode {
    /// Prefer compiled AOT when possible and otherwise keep the lambdify path.
    #[default]
    Defaults,
    /// Require a prebuilt compiled AOT backend.
    RequirePrebuilt,
    /// Build a release AOT artifact when it is missing.
    BuildIfMissingRelease,
}

/// Build policy for dense nonlinear generated backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolicAotBuildPolicy {
    /// Use compiled AOT only when an artifact is already available.
    #[default]
    UseIfAvailable,
    /// Require an existing compiled artifact.
    RequirePrebuilt,
    /// Build the generated crate if the compiled artifact is missing.
    BuildIfMissing { profile: AotBuildProfile },
    /// Always rebuild the generated crate.
    RebuildAlways { profile: AotBuildProfile },
}

/// Bounded retry policy for transient nonlinear AOT infrastructure failures.
///
/// Only failures that look like process/file-system contention are retried.
/// Compiler diagnostics for invalid generated code are returned immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicAotBuildRetryPolicy {
    /// Maximum number of materialize/build attempts, including the first one.
    pub max_attempts: usize,
    /// Base delay before a retry. The delay grows linearly with the attempt number.
    pub retry_delay: Duration,
}

impl SymbolicAotBuildRetryPolicy {
    /// Creates a bounded policy. Zero attempts is normalized to one attempt.
    pub fn new(max_attempts: usize, retry_delay: Duration) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            retry_delay,
        }
    }

    /// Disables retries while retaining one initial build attempt.
    pub fn no_retry() -> Self {
        Self::new(1, Duration::ZERO)
    }
}

impl Default for SymbolicAotBuildRetryPolicy {
    fn default() -> Self {
        Self::new(3, Duration::from_millis(120))
    }
}

/// User-facing configuration for dense nonlinear generated backend orchestration.
#[derive(Debug, Clone, Default)]
pub struct SymbolicGeneratedBackendConfig {
    /// Optional explicit backend policy override.
    pub backend_policy_override: Option<SymbolicBackendSelectionPolicy>,
    /// Optional resolver snapshot reused across calls.
    pub resolver: Option<AotResolver>,
    /// Dense AOT runtime-plan chunking options.
    pub aot_options: SymbolicDenseAotOptions,
    /// Lifecycle build policy.
    pub build_policy: SymbolicAotBuildPolicy,
    /// Retry policy for transient materialize/build failures.
    pub retry_policy: SymbolicAotBuildRetryPolicy,
    /// Explicit Rust compile settings for generated AOT cold builds.
    ///
    /// The default keeps the existing Cargo profile unchanged. Use
    /// `AotCompileConfig::fast_build()` when cold-build latency is more
    /// important than maximum generated-code optimization.
    pub aot_compile_config: AotCompileConfig,
    /// Parent directory where generated crates should be materialized when a build is requested.
    pub output_parent_dir: Option<PathBuf>,
    /// Optional explicit generated crate name.
    pub crate_name_override: Option<String>,
    /// Optional explicit generated module name.
    pub module_name_override: Option<String>,
}

impl SymbolicGeneratedBackendConfig {
    /// Creates an empty generated-backend configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Production-oriented defaults for dense nonlinear generated backends.
    pub fn defaults() -> Self {
        Self::new().with_backend_policy_override(Some(
            SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
        ))
    }

    /// Configuration that requires a prebuilt compiled backend.
    pub fn require_prebuilt() -> Self {
        Self::defaults()
            .with_backend_policy_override(Some(SymbolicBackendSelectionPolicy::AotOnly))
            .with_build_policy(SymbolicAotBuildPolicy::RequirePrebuilt)
    }

    /// Configuration that builds a release artifact when it is missing.
    pub fn build_if_missing_release(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::defaults()
            .with_output_parent_dir(Some(output_parent_dir.into()))
            .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
    }

    /// Creates a configuration from one high-level mode.
    pub fn from_mode(mode: DenseGeneratedBackendMode) -> Self {
        match mode {
            DenseGeneratedBackendMode::Defaults => Self::defaults(),
            DenseGeneratedBackendMode::RequirePrebuilt => Self::require_prebuilt(),
            DenseGeneratedBackendMode::BuildIfMissingRelease => {
                Self::defaults().with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
            }
        }
    }

    /// Sets an explicit backend policy override.
    pub fn with_backend_policy_override(
        mut self,
        backend_policy_override: Option<SymbolicBackendSelectionPolicy>,
    ) -> Self {
        self.backend_policy_override = backend_policy_override;
        self
    }

    /// Installs a resolver snapshot that may be reused across calls.
    pub fn with_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// Sets dense nonlinear AOT runtime-plan options.
    pub fn with_aot_options(mut self, aot_options: SymbolicDenseAotOptions) -> Self {
        self.aot_options = aot_options;
        self
    }

    /// Sets the lifecycle build policy.
    pub fn with_build_policy(mut self, build_policy: SymbolicAotBuildPolicy) -> Self {
        self.build_policy = build_policy;
        self
    }

    /// Sets the bounded retry policy used by automatic AOT builds.
    pub fn with_retry_policy(mut self, retry_policy: SymbolicAotBuildRetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Sets Rust compile settings for generated AOT builds.
    pub fn with_aot_compile_config(mut self, aot_compile_config: AotCompileConfig) -> Self {
        self.aot_compile_config = aot_compile_config;
        self
    }

    /// Sets the output directory used by automatic builds.
    pub fn with_output_parent_dir(mut self, output_parent_dir: Option<PathBuf>) -> Self {
        self.output_parent_dir = output_parent_dir;
        self
    }

    /// Overrides the generated crate name used by automatic builds.
    pub fn with_crate_name_override(mut self, crate_name_override: Option<String>) -> Self {
        self.crate_name_override = crate_name_override;
        self
    }

    /// Overrides the generated module name used by automatic builds.
    pub fn with_module_name_override(mut self, module_name_override: Option<String>) -> Self {
        self.module_name_override = module_name_override;
        self
    }

    fn effective_backend_policy(&self) -> SymbolicBackendSelectionPolicy {
        self.backend_policy_override
            .unwrap_or(match self.build_policy {
                SymbolicAotBuildPolicy::RequirePrebuilt => SymbolicBackendSelectionPolicy::AotOnly,
                _ => SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
            })
    }

    fn output_parent_dir(&self) -> Result<&Path, SolveError> {
        self.output_parent_dir
            .as_deref()
            .ok_or(SolveError::AotBuildOutputDirMissing)
    }
}

/// Result of preparing one nonlinear symbolic problem through the high-level
/// generated-backend orchestration layer.
pub struct PreparedGeneratedSymbolicProblem {
    /// Final solver-facing symbolic problem.
    pub problem: SymbolicNonlinearProblem,
    /// Effective backend branch that ended up being used.
    pub selected_backend: SelectedSymbolicNonlinearBackendKind,
    /// Updated resolver snapshot after any materialized build/reuse.
    pub updated_resolver: Option<AotResolver>,
    /// Materialized build metadata when this call performed a build step.
    pub build_result: Option<AotBuildResult>,
    /// Common preparation/build telemetry for the selected backend branch.
    pub preparation_report: SymbolicPreparationReport,
}

impl PreparedGeneratedSymbolicProblem {
    /// Consumes the orchestration result and returns just the symbolic problem.
    pub fn into_problem(self) -> SymbolicNonlinearProblem {
        self.problem
    }

    /// Consumes the orchestration result and preserves its prepared backend.
    ///
    /// No symbolic or AOT work is repeated here; this only changes the public
    /// lifecycle wrapper used for subsequent parameter bindings.
    pub fn into_prepared(
        self,
    ) -> crate::numerical::Nonlinear_systems::symbolic::PreparedSymbolicNonlinearProblem {
        let mut problem = self.problem;
        problem.replace_preparation_report(self.preparation_report);
        crate::numerical::Nonlinear_systems::symbolic::PreparedSymbolicNonlinearProblem::from_problem(
            problem,
        )
    }
}

fn generated_names(problem_key: &str, config: &SymbolicGeneratedBackendConfig) -> (String, String) {
    let suffix = problem_key
        .chars()
        .take(16)
        .collect::<String>()
        .replace('-', "_");
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_nonlinear_dense_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_nonlinear_dense_module_{suffix}"));
    (crate_name, module_name)
}

fn select_with_config<'a>(
    problem: &'a SymbolicNonlinearProblem,
    config: &SymbolicGeneratedBackendConfig,
    resolver: Option<&AotResolver>,
) -> crate::numerical::Nonlinear_systems::symbolic_backend::SelectedSymbolicNonlinearBackend<'a> {
    select_symbolic_nonlinear_backend(
        problem,
        config.effective_backend_policy(),
        resolver,
        config.aot_options,
    )
}

fn should_build_for_selection(
    config: &SymbolicGeneratedBackendConfig,
    selected_backend: SelectedSymbolicNonlinearBackendKind,
) -> bool {
    match config.build_policy {
        SymbolicAotBuildPolicy::UseIfAvailable | SymbolicAotBuildPolicy::RequirePrebuilt => false,
        SymbolicAotBuildPolicy::BuildIfMissing { .. } => {
            selected_backend != SelectedSymbolicNonlinearBackendKind::AotCompiled
        }
        SymbolicAotBuildPolicy::RebuildAlways { .. } => true,
    }
}

fn build_profile(policy: SymbolicAotBuildPolicy) -> Option<AotBuildProfile> {
    match policy {
        SymbolicAotBuildPolicy::BuildIfMissing { profile }
        | SymbolicAotBuildPolicy::RebuildAlways { profile } => Some(profile),
        SymbolicAotBuildPolicy::UseIfAvailable | SymbolicAotBuildPolicy::RequirePrebuilt => None,
    }
}

fn is_transient_nonlinear_aot_failure(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    low.contains("permission denied")
        || low.contains("access is denied")
        || low.contains("being used by another process")
        || low.contains("resource busy")
        || low.contains("temporarily unavailable")
        || low.contains("failed to spawn")
        || low.contains("could not write")
        || low.contains("file is locked")
        || low.contains("sharing violation")
        || low.contains("os error 5")
        || low.contains("lnk1104")
        || low.contains("blocking waiting for file lock")
}

fn nonlinear_aot_retry_message(
    build_context: &str,
    attempts: usize,
    detail: &str,
    transient: bool,
) -> String {
    let class = if transient {
        "transient infrastructure failure"
    } else {
        "deterministic build failure"
    };
    format!(
        "AOT build execution failed ({build_context}) after {attempts} attempt(s); classified as {class}. \
If this is a transient infrastructure failure, check stale compiler processes, file locks and antivirus/indexer interference; \
otherwise inspect compiler stdout/stderr below.\n\
detail:\n{detail}"
    )
}

fn resolver_from_nonlinear_aot_build(
    prepared: &crate::numerical::Nonlinear_systems::symbolic::PreparedSymbolicNonlinearAotProblem<
        '_,
    >,
    build: &AotBuildResult,
    resolver_snapshot: Option<AotResolver>,
) -> Result<AotResolver, SolveError> {
    let problem_key = prepared.problem_key();
    let mut registry = resolver_snapshot
        .as_ref()
        .map(|resolver| resolver.registry().clone())
        .unwrap_or_default();
    registry.register_materialized_build(prepared.manifest(), build);
    let resolver = AotResolver::new(registry);
    let registered = resolver
        .registry()
        .get_by_problem_key(&problem_key)
        .ok_or_else(|| {
            SolveError::AotBuildFailed(
                "successful dense AOT build was not present in the resolver".to_string(),
            )
        })?;
    register_generated_dense_cdylib_backend(registered).map_err(|err| {
        SolveError::AotBuildFailed(format!(
            "dense AOT cdylib load failed for problem key {}: {err}",
            problem_key
        ))
    })?;
    Ok(resolver)
}

fn perform_requested_build(
    baseline_problem: &SymbolicNonlinearProblem,
    config: &SymbolicGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<AotBuildResult>, Option<AotResolver>), SolveError> {
    let profile = match build_profile(config.build_policy) {
        Some(profile) => profile,
        None => return Ok((None, resolver_snapshot)),
    };

    let prepared = baseline_problem.prepare_dense_aot_problem(config.aot_options);
    let problem_key = prepared.problem_key();
    let (crate_name, module_name) = generated_names(&problem_key, config);
    info!(
        "Materializing dense nonlinear AOT build for crate '{}' with profile {:?}",
        crate_name, profile
    );
    let output_parent_dir = config.output_parent_dir()?;
    let max_attempts = config.retry_policy.max_attempts.max(1);
    let mut last_failure = None;
    let mut last_transient = false;
    let mut successful_build = None;
    let mut built_now = false;
    for attempt in 1..=max_attempts {
        let materialized = materialize_symbolic_nonlinear_aot_build_with_compile_config(
            &crate_name,
            &module_name,
            baseline_problem,
            config.aot_options,
            output_parent_dir,
            profile,
            config.aot_compile_config.clone(),
        );
        let result = match materialized {
            Ok(build)
                if matches!(
                    config.build_policy,
                    SymbolicAotBuildPolicy::BuildIfMissing { .. }
                ) && build.expected_cdylib.exists()
                    && nonlinear_aot_ready_marker_matches(&build, &problem_key) =>
            {
                info!(
                    "Reusing persisted nonlinear AOT artifact for problem key {}",
                    problem_key
                );
                successful_build = Some(build);
                break;
            }
            Ok(build) => {
                if invalidate_nonlinear_aot_ready_marker(&build).map_err(|error| {
                    SolveError::AotBuildFailed(format!(
                        "cannot quarantine previous nonlinear AOT publication marker '{}': {error}",
                        nonlinear_aot_ready_marker_path(&build).display()
                    ))
                })? {
                    info!(
                        "Invalidated previous nonlinear AOT ready marker before rebuild for problem key {}",
                        problem_key
                    );
                }
                match build.execute() {
                    Ok(executed) if executed.succeeded() => {
                        built_now = true;
                        if let Err(error) = write_nonlinear_aot_ready_marker(&build, &problem_key) {
                            warn!(
                                "Nonlinear AOT build succeeded but ready marker '{}' could not be written: {error}",
                                nonlinear_aot_ready_marker_path(&build).display()
                            );
                        }
                        Ok(build)
                    }
                    Ok(executed) => Err(format!(
                        "status={:?}\nstdout:\n{}\nstderr:\n{}",
                        executed.status_code, executed.stdout, executed.stderr
                    )),
                    Err(error) => Err(error.to_string()),
                }
            }
            Err(error) => Err(error.to_string()),
        };
        match result {
            Ok(build) => {
                successful_build = Some(build);
                break;
            }
            Err(detail) => {
                let transient = is_transient_nonlinear_aot_failure(&detail);
                last_transient = transient;
                last_failure = Some((attempt, detail));
                if !transient || attempt == max_attempts {
                    break;
                }
                warn!(
                    "Transient nonlinear AOT build failure on attempt {attempt}/{max_attempts}; retrying"
                );
                thread::sleep(
                    config
                        .retry_policy
                        .retry_delay
                        .saturating_mul(attempt as u32),
                );
            }
        }
    }
    let build = match successful_build {
        Some(build) => build,
        None => {
            let (attempts, detail) =
                last_failure.unwrap_or_else(|| (0, "unknown build failure".to_string()));
            return Err(SolveError::AotBuildFailed(nonlinear_aot_retry_message(
                &format!("problem key {problem_key}"),
                attempts,
                &detail,
                last_transient,
            )));
        }
    };
    let resolver = resolver_from_nonlinear_aot_build(&prepared, &build, resolver_snapshot)?;
    Ok((built_now.then_some(build), Some(resolver)))
}

fn discover_persisted_nonlinear_aot_resolver(
    baseline_problem: &SymbolicNonlinearProblem,
    config: &SymbolicGeneratedBackendConfig,
) -> Result<Option<AotResolver>, SolveError> {
    if config.resolver.is_some()
        || !matches!(config.build_policy, SymbolicAotBuildPolicy::RequirePrebuilt)
    {
        return Ok(None);
    }
    let Some(output_parent_dir) = config.output_parent_dir.as_deref() else {
        return Ok(None);
    };

    let _lifecycle_guard = nonlinear_aot_lifecycle_lock().lock().map_err(|_| {
        SolveError::AotBuildFailed("nonlinear AOT lifecycle lock poisoned".to_string())
    })?;
    let prepared = baseline_problem.prepare_dense_aot_problem(config.aot_options);
    let problem_key = prepared.problem_key();
    let _file_lock = acquire_nonlinear_aot_file_lock(
        output_parent_dir,
        &problem_key,
        NONLINEAR_AOT_FILE_LOCK_TIMEOUT,
    )?;
    let (crate_name, module_name) = generated_names(&problem_key, config);

    // RequirePrebuilt has no profile field for historical API compatibility.
    // Inspect both standard Cargo locations without compiling anything.
    for profile in [AotBuildProfile::Release, AotBuildProfile::Debug] {
        let build = materialize_symbolic_nonlinear_aot_build_with_compile_config(
            &crate_name,
            &module_name,
            baseline_problem,
            config.aot_options,
            output_parent_dir,
            profile,
            config.aot_compile_config.clone(),
        )
        .map_err(|error| SolveError::AotBuildFailed(error.to_string()))?;
        if build.expected_cdylib.exists()
            && nonlinear_aot_ready_marker_matches(&build, &problem_key)
        {
            info!(
                "Discovered persisted nonlinear AOT artifact for RequirePrebuilt, problem key {}",
                problem_key
            );
            return resolver_from_nonlinear_aot_build(&prepared, &build, None).map(Some);
        }
    }
    Ok(None)
}

fn fallback_lambdify_problem(
    equations: Vec<Expr>,
    mut options: SymbolicProblemOptions,
) -> Result<SymbolicNonlinearProblem, SolveError> {
    options.backend_config = SymbolicBackendConfig::lambdify();
    SymbolicNonlinearProblem::from_expressions_with_options(equations, options)
}

fn report_artifact_policy(policy: SymbolicAotBuildPolicy) -> SymbolicArtifactPolicy {
    match policy {
        SymbolicAotBuildPolicy::UseIfAvailable => SymbolicArtifactPolicy::UseIfAvailable,
        SymbolicAotBuildPolicy::RequirePrebuilt => SymbolicArtifactPolicy::RequirePrebuilt,
        SymbolicAotBuildPolicy::BuildIfMissing { .. } => SymbolicArtifactPolicy::BuildIfMissing,
        SymbolicAotBuildPolicy::RebuildAlways { .. } => SymbolicArtifactPolicy::RebuildAlways,
    }
}

impl SymbolicNonlinearProblem {
    /// Builds a symbolic nonlinear problem through the high-level generated-backend layer.
    ///
    /// This is the preferred user-facing path when the caller wants mode/build
    /// policy behavior such as:
    /// - use compiled AOT if already available,
    /// - require a prebuilt artifact,
    /// - or build a dense generated crate automatically when needed.
    pub fn from_expressions_with_generated_backend(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
        config: SymbolicGeneratedBackendConfig,
    ) -> Result<PreparedGeneratedSymbolicProblem, SolveError> {
        let preparation_started = Instant::now();
        let baseline_problem = fallback_lambdify_problem(equations.clone(), options.clone())?;
        let initial_resolver =
            config
                .resolver
                .clone()
                .or(discover_persisted_nonlinear_aot_resolver(
                    &baseline_problem,
                    &config,
                )?);
        let initial_selection =
            select_with_config(&baseline_problem, &config, initial_resolver.as_ref());

        let build_requested =
            should_build_for_selection(&config, initial_selection.effective_backend);
        let build_started = Instant::now();
        let (build_result, resolver_snapshot) =
            if should_build_for_selection(&config, initial_selection.effective_backend) {
                let _lifecycle_guard = nonlinear_aot_lifecycle_lock().lock().map_err(|_| {
                    SolveError::AotBuildFailed("nonlinear AOT lifecycle lock poisoned".to_string())
                })?;
                let selection_after_wait =
                    select_with_config(&baseline_problem, &config, initial_resolver.as_ref());
                if should_build_for_selection(&config, selection_after_wait.effective_backend) {
                    let prepared = baseline_problem.prepare_dense_aot_problem(config.aot_options);
                    let _file_lock = acquire_nonlinear_aot_file_lock(
                        config.output_parent_dir()?,
                        &prepared.problem_key(),
                        NONLINEAR_AOT_FILE_LOCK_TIMEOUT,
                    )?;
                    perform_requested_build(&baseline_problem, &config, initial_resolver.clone())?
                } else {
                    (None, initial_resolver.clone())
                }
            } else {
                (None, initial_resolver.clone())
            };
        // A build-policy request is not proof that this invocation performed
        // a build: another process may have published the artifact while we
        // waited for the lifecycle lock. Report the interval only for an
        // operation that actually returned build metadata; lock wait time is
        // already included in the end-to-end preparation duration.
        let build_duration = build_result.as_ref().map(|_| build_started.elapsed());

        let final_selection =
            select_with_config(&baseline_problem, &config, resolver_snapshot.as_ref());
        let final_backend = final_selection.effective_backend;
        let (artifact_key, generated_residual_jobs, generated_jacobian_jobs) = final_selection
            .prepared_aot_problem
            .as_ref()
            .map(|prepared| {
                let manifest = prepared.manifest();
                (
                    Some(prepared.problem_key()),
                    Some(manifest.functions.residual_chunks.len()),
                    Some(manifest.functions.jacobian_chunks.len()),
                )
            })
            .unwrap_or((None, None, None));
        let artifact_action = match final_backend {
            SelectedSymbolicNonlinearBackendKind::AotCompiled if build_result.is_some() => {
                SymbolicArtifactAction::Built
            }
            SelectedSymbolicNonlinearBackendKind::AotCompiled => SymbolicArtifactAction::Reused,
            SelectedSymbolicNonlinearBackendKind::Lambdify
                if config.effective_backend_policy()
                    == SymbolicBackendSelectionPolicy::PreferAotThenLambdify =>
            {
                SymbolicArtifactAction::FallbackToLambdify
            }
            _ => SymbolicArtifactAction::NotApplicable,
        };
        let preparation_report = SymbolicPreparationReport {
            effective_backend: if final_backend == SelectedSymbolicNonlinearBackendKind::AotCompiled
            {
                SymbolicBackendKind::Aot
            } else {
                SymbolicBackendKind::Lambdify
            },
            artifact_policy: report_artifact_policy(config.build_policy),
            artifact_action,
            artifact_key,
            preparation_duration: preparation_started.elapsed(),
            build_duration,
            generated_residual_jobs,
            generated_jacobian_jobs,
        };
        match final_backend {
            SelectedSymbolicNonlinearBackendKind::Lambdify => {
                let mut problem = baseline_problem;
                problem.replace_preparation_report(preparation_report.clone());
                Ok(PreparedGeneratedSymbolicProblem {
                    problem,
                    selected_backend: SelectedSymbolicNonlinearBackendKind::Lambdify,
                    updated_resolver: resolver_snapshot,
                    build_result,
                    preparation_report,
                })
            }
            SelectedSymbolicNonlinearBackendKind::AotCompiled => {
                match SymbolicNonlinearProblem::from_expressions_with_backend_selection(
                    equations.clone(),
                    options.clone(),
                    config.effective_backend_policy(),
                    resolver_snapshot.as_ref(),
                    config.aot_options,
                ) {
                    Ok(mut problem) => {
                        problem.replace_preparation_report(preparation_report.clone());
                        Ok(PreparedGeneratedSymbolicProblem {
                            problem,
                            selected_backend: SelectedSymbolicNonlinearBackendKind::AotCompiled,
                            updated_resolver: resolver_snapshot,
                            build_result,
                            preparation_report,
                        })
                    }
                    Err(SolveError::CompiledAotRuntimeUnavailable(message))
                        if config.effective_backend_policy()
                            == SymbolicBackendSelectionPolicy::PreferAotThenLambdify =>
                    {
                        warn!(
                            "Dense nonlinear compiled AOT artifact exists but no linked runtime is registered; falling back to lambdify: {}",
                            message
                        );
                        let mut problem = baseline_problem;
                        problem.replace_preparation_report(preparation_report.clone());
                        Ok(PreparedGeneratedSymbolicProblem {
                            problem,
                            selected_backend: SelectedSymbolicNonlinearBackendKind::Lambdify,
                            updated_resolver: resolver_snapshot,
                            build_result,
                            preparation_report,
                        })
                    }
                    Err(err) => Err(err),
                }
            }
            SelectedSymbolicNonlinearBackendKind::AotRegisteredButNotBuilt => {
                Err(SolveError::CompiledAotArtifactNotBuilt(
                    "dense nonlinear AOT artifact is registered but not built".to_string(),
                ))
            }
            SelectedSymbolicNonlinearBackendKind::AotMissing => {
                Err(SolveError::CompiledAotArtifactMissing(
                    "dense nonlinear AOT artifact is missing".to_string(),
                ))
            }
        }
    }

    /// String-based convenience wrapper around
    /// [`Self::from_expressions_with_generated_backend`].
    pub fn from_strings_with_generated_backend(
        equations: Vec<String>,
        options: SymbolicProblemOptions,
        config: SymbolicGeneratedBackendConfig,
    ) -> Result<PreparedGeneratedSymbolicProblem, SolveError> {
        let expressions = equations
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        Self::from_expressions_with_generated_backend(expressions, options, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::problem::JacobianProvider;
    use crate::numerical::Nonlinear_systems::problem::NonlinearProblem;
    use crate::numerical::Nonlinear_systems::symbolic_aot::materialize_symbolic_nonlinear_aot_build;
    use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::{
        aot_solver_test_guard, linked_dense_resolver_for_problem,
    };
    use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use std::process::Command;
    use tempfile::tempdir;

    fn elementary_options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_lambdify_backend()
    }

    fn elementary_equations() -> Vec<String> {
        vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()]
    }

    fn elementary_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_expressions_with_options(
            elementary_equations()
                .iter()
                .map(|equation| Expr::parse_expression(equation))
                .collect(),
            elementary_options(),
        )
        .expect("elementary problem should prepare")
    }

    fn parameterized_options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()])
    }

    fn parameterized_equations() -> Vec<String> {
        vec!["a*x+y-3".to_string(), "x-y".to_string()]
    }

    #[test]
    fn generated_backend_defaults_fall_back_to_lambdify_when_aot_is_missing() {
        let prepared = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::defaults(),
        )
        .expect("defaults should fall back to lambdify");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicNonlinearBackendKind::Lambdify
        );
        assert_eq!(prepared.problem.backend_kind().as_str(), "lambdify");
        assert!(prepared.updated_resolver.is_none());
        assert!(prepared.build_result.is_none());
    }

    #[test]
    fn generated_backend_require_prebuilt_surfaces_missing_artifact() {
        let result = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::require_prebuilt(),
        );

        match result {
            Err(SolveError::CompiledAotArtifactMissing(_)) => {}
            Err(other) => panic!("expected missing compiled artifact error, got {other}"),
            Ok(_) => panic!("missing prebuilt artifact should be surfaced"),
        }
    }

    #[test]
    fn generated_backend_build_if_missing_materializes_build_and_updates_resolver() {
        let _guard = aot_solver_test_guard();
        let dir = tempdir().expect("tempdir should exist");
        let prepared = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("build-if-missing should succeed");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        assert!(prepared.build_result.is_some());
        assert_eq!(
            prepared.preparation_report.effective_backend,
            SymbolicBackendKind::Aot
        );
        assert_eq!(
            prepared.preparation_report.artifact_policy,
            SymbolicArtifactPolicy::BuildIfMissing
        );
        assert_eq!(
            prepared.preparation_report.artifact_action,
            SymbolicArtifactAction::Built
        );
        assert!(prepared.preparation_report.artifact_key.is_some());
        assert!(prepared.preparation_report.build_duration.is_some());
        let resolver = prepared
            .updated_resolver
            .as_ref()
            .expect("build should update resolver");
        assert_eq!(resolver.registry().len(), 1);
        let problem_key = resolver
            .registry()
            .problem_keys()
            .into_iter()
            .next()
            .expect("resolver should contain one problem key");
        let resolved = resolver.resolve_by_problem_key(&problem_key);
        assert!(resolved.is_compiled());
        assert!(unregister_linked_dense_backend(&problem_key).is_some());
    }

    #[test]
    fn generated_backend_reuses_updated_resolver_with_linked_runtime() {
        let _guard = aot_solver_test_guard();
        let dir = tempdir().expect("tempdir should exist");
        let first = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("first build should succeed");
        let resolver = first
            .updated_resolver
            .clone()
            .expect("build should produce resolver");
        let problem_key = resolver
            .registry()
            .problem_keys()
            .into_iter()
            .next()
            .expect("resolver should contain one problem key");
        let second = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
        )
        .expect("second call should reuse compiled resolver");

        assert_eq!(
            second.selected_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        assert!(second.build_result.is_none());
        assert_eq!(
            second.preparation_report.effective_backend,
            SymbolicBackendKind::Aot
        );
        assert_eq!(
            second.preparation_report.artifact_policy,
            SymbolicArtifactPolicy::RequirePrebuilt
        );
        assert_eq!(
            second.preparation_report.artifact_action,
            SymbolicArtifactAction::Reused
        );
        assert!(second.preparation_report.artifact_key.is_some());
        assert!(second.preparation_report.build_duration.is_none());
        assert_eq!(second.problem.backend_kind().as_str(), "aot");
        let x0 = DVector::from_vec(vec![3.0, -1.0]);
        let residual = second.problem.residual(&x0).expect("residual");
        assert_relative_eq!(residual[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(residual[1], 0.0, epsilon = 1e-12);

        let reusable = second.into_prepared();
        let report_before_bind = reusable.preparation_report().clone();
        let key_before_bind = reusable
            .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
            .problem_key();
        let first_bound = reusable
            .bind_without_parameters()
            .expect("non-parameterized AOT binding should succeed");
        let second_bound = reusable
            .bind_without_parameters()
            .expect("second non-parameterized AOT binding should succeed");
        assert!(first_bound.residual(&x0).is_ok());
        assert!(second_bound.jacobian(&x0).is_ok());
        assert_eq!(
            reusable.preparation_report(),
            &report_before_bind,
            "binding must not repeat or mutate symbolic/AOT preparation"
        );
        let key_after_bind = reusable
            .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
            .problem_key();
        assert_eq!(key_before_bind, key_after_bind);

        unregister_linked_dense_backend(&problem_key);
    }

    #[test]
    fn generated_backend_require_prebuilt_rejects_missing_compiled_output() {
        let problem = elementary_problem();
        let (_dir, resolver, problem_key) = linked_dense_resolver_for_problem(&problem);
        let artifact = resolver
            .registry()
            .get_by_problem_key(&problem_key)
            .expect("resolver should contain the fixture artifact");
        std::fs::remove_file(&artifact.expected_rlib)
            .expect("fixture compiled output should be removable");

        let result = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
            elementary_equations()
                .iter()
                .map(|equation| Expr::parse_expression(equation))
                .collect(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
        );

        match result {
            Err(SolveError::CompiledAotArtifactNotBuilt(message)) => {
                assert!(message.contains("not built"));
            }
            Err(other) => panic!("expected missing compiled output error, got {other}"),
            Ok(_) => panic!("RequirePrebuilt must reject a registered artifact without output"),
        }
    }

    #[test]
    fn generated_backend_require_prebuilt_does_not_reuse_different_parameter_schema() {
        let problem = elementary_problem();
        let (_dir, resolver, elementary_key) = linked_dense_resolver_for_problem(&problem);
        let parameterized = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
            parameterized_equations()
                .iter()
                .map(|equation| Expr::parse_expression(equation))
                .collect(),
            parameterized_options(),
            SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
        );

        assert_ne!(
            elementary_key,
            SymbolicNonlinearProblem::from_expressions_with_options(
                parameterized_equations()
                    .iter()
                    .map(|equation| Expr::parse_expression(equation))
                    .collect(),
                parameterized_options(),
            )
            .expect("parameterized problem should prepare")
            .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
            .problem_key()
        );
        match parameterized {
            Err(SolveError::CompiledAotArtifactMissing(message)) => {
                assert!(message.contains("missing"));
            }
            Err(other) => panic!("expected missing artifact for different schema, got {other}"),
            Ok(_) => panic!("RequirePrebuilt must not reuse another parameter schema"),
        }
    }

    #[test]
    fn parameterized_generated_backend_reuses_artifact_for_multiple_bindings() {
        let _guard = aot_solver_test_guard();
        let dir = tempdir().expect("tempdir should exist");
        let first = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            parameterized_equations(),
            parameterized_options(),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("parameterized build should succeed");
        assert!(first.build_result.is_some());

        let resolver = first
            .updated_resolver
            .clone()
            .expect("parameterized build should produce resolver");
        let problem_key = resolver
            .registry()
            .problem_keys()
            .into_iter()
            .next()
            .expect("parameterized resolver should contain one problem key");
        let second = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            parameterized_equations(),
            parameterized_options(),
            SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
        )
        .expect("parameterized prebuilt call should succeed");
        assert_eq!(
            second.selected_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        assert!(second.build_result.is_none());

        let reusable = second.into_prepared();
        let key_before = reusable
            .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
            .problem_key();
        let first_bound = reusable
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("first parameter binding should succeed");
        let second_bound = reusable
            .bind_values(DVector::from_vec(vec![4.0]))
            .expect("second parameter binding should succeed");
        let first_residual = first_bound
            .residual(&DVector::from_vec(vec![1.0, 1.0]))
            .expect("first bound residual");
        let second_residual = second_bound
            .residual(&DVector::from_vec(vec![0.6, 0.6]))
            .expect("second bound residual");
        assert!(first_residual.iter().all(|value| value.abs() < 1e-12));
        assert!(second_residual.iter().all(|value| value.abs() < 1e-12));
        let key_after = reusable
            .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
            .problem_key();
        assert_eq!(key_before, key_after);

        unregister_linked_dense_backend(&problem_key);
    }

    #[test]
    fn nonlinear_aot_file_lock_reports_contention_and_releases_cleanly() {
        let output_dir = tempdir().expect("AOT output directory should exist");
        let problem_key = "file-lock-test";
        let first =
            acquire_nonlinear_aot_file_lock(output_dir.path(), problem_key, Duration::from_secs(1))
                .expect("first lifecycle lock should succeed");
        let parent = output_dir.path().to_path_buf();
        let waiter = std::thread::spawn(move || {
            acquire_nonlinear_aot_file_lock(&parent, problem_key, Duration::from_millis(100))
                .expect_err("second lifecycle owner should time out")
                .to_string()
        });

        let message = waiter.join().expect("lock waiter should not panic");
        assert!(
            message.contains("timed out"),
            "unexpected lock contention error: {message}"
        );
        drop(first);
        acquire_nonlinear_aot_file_lock(output_dir.path(), problem_key, Duration::from_secs(1))
            .expect("lock should be available after the owner drops");
    }

    #[test]
    fn nonlinear_aot_retry_policy_and_failure_classification_are_explicit() {
        assert_eq!(
            SymbolicAotBuildRetryPolicy::new(0, Duration::from_millis(5)).max_attempts,
            1
        );
        assert_eq!(
            SymbolicAotBuildRetryPolicy::no_retry(),
            SymbolicAotBuildRetryPolicy::new(1, Duration::ZERO)
        );

        for detail in [
            "LINK : fatal error LNK1104: cannot open file",
            "error: Access is denied (os error 5)",
            "Blocking waiting for file lock on artifact directory",
            "failed to spawn build runner",
        ] {
            assert!(
                is_transient_nonlinear_aot_failure(detail),
                "expected transient classification for {detail}"
            );
        }
        assert!(!is_transient_nonlinear_aot_failure(
            "error[E0425]: cannot find value `missing` in this scope"
        ));

        let message = nonlinear_aot_retry_message(
            "problem key retry-test",
            2,
            "status=Some(101)\nstderr:\nerror[E0425]: missing",
            false,
        );
        assert!(message.contains("after 2 attempt(s)"));
        assert!(message.contains("deterministic build failure"));
        assert!(message.contains("error[E0425]"));
    }

    #[test]
    fn nonlinear_aot_ready_marker_round_trips_and_rejects_wrong_key() {
        let problem = elementary_problem();
        let output_dir = tempdir().expect("AOT output directory should exist");
        let build = materialize_symbolic_nonlinear_aot_build(
            "generated_nonlinear_marker_fixture",
            "generated_nonlinear_marker_module",
            &problem,
            SymbolicDenseAotOptions::default(),
            output_dir.path(),
            AotBuildProfile::Release,
        )
        .expect("AOT request should materialize");
        let problem_key = problem
            .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
            .problem_key();

        std::fs::create_dir_all(
            build
                .expected_cdylib
                .parent()
                .expect("fixture cdylib should have a parent"),
        )
        .expect("fixture artifact directory should be writable");
        std::fs::write(&build.expected_cdylib, b"complete fixture artifact")
            .expect("fixture cdylib should be writable");

        write_nonlinear_aot_ready_marker(&build, &problem_key)
            .expect("ready marker should be written atomically");
        assert!(nonlinear_aot_ready_marker_matches(&build, &problem_key));
        assert!(!nonlinear_aot_ready_marker_matches(
            &build,
            "different-problem-key"
        ));
        assert!(nonlinear_aot_ready_marker_path(&build).exists());

        std::fs::write(&build.expected_cdylib, b"truncated fixture artifact")
            .expect("fixture cdylib should remain writable");
        assert!(
            !nonlinear_aot_ready_marker_matches(&build, &problem_key),
            "mutated output must not remain trusted by the old marker"
        );
        assert!(
            invalidate_nonlinear_aot_ready_marker(&build)
                .expect("marker invalidation should succeed")
        );
        assert!(!nonlinear_aot_ready_marker_path(&build).exists());
        assert!(
            !invalidate_nonlinear_aot_ready_marker(&build)
                .expect("missing marker invalidation should be harmless")
        );
    }

    #[test]
    fn nonlinear_aot_compiler_failure_quarantines_previous_publication() {
        let _guard = aot_solver_test_guard();
        let problem = elementary_problem();
        let output_dir = tempdir().expect("AOT output directory should exist");
        let config = SymbolicGeneratedBackendConfig::new()
            .with_backend_policy_override(Some(SymbolicBackendSelectionPolicy::AotOnly))
            .with_build_policy(SymbolicAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Debug,
            })
            .with_retry_policy(SymbolicAotBuildRetryPolicy::no_retry())
            .with_output_parent_dir(Some(output_dir.path().to_path_buf()));
        let prepared = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
        let problem_key = prepared.problem_key();
        let (crate_name, module_name) = generated_names(&problem_key, &config);
        let previous = materialize_symbolic_nonlinear_aot_build(
            &crate_name,
            &module_name,
            &problem,
            config.aot_options,
            output_dir.path(),
            AotBuildProfile::Debug,
        )
        .expect("previous AOT layout should materialize");
        std::fs::create_dir_all(
            previous
                .expected_cdylib
                .parent()
                .expect("fixture cdylib should have a parent"),
        )
        .expect("fixture artifact directory should be writable");
        std::fs::write(&previous.expected_cdylib, b"previous complete artifact")
            .expect("previous artifact should be writable");
        write_nonlinear_aot_ready_marker(&previous, &problem_key)
            .expect("previous publication marker should be writable");

        let old_cargo = std::env::var_os("CARGO");
        let fake_runner = "rustedscithe-intentionally-missing-aot-runner";
        // The test deliberately injects a process that cannot perform a Cargo
        // build; environment access is synchronized by the AOT test guard.
        unsafe { std::env::set_var("CARGO", fake_runner) };
        let failure = perform_requested_build(&problem, &config, None)
            .expect_err("injected compiler failure must be returned");
        unsafe {
            if let Some(value) = old_cargo {
                std::env::set_var("CARGO", value);
            } else {
                std::env::remove_var("CARGO");
            }
        }

        let message = failure.to_string();
        assert!(
            message.contains("AOT") || message.contains("aot"),
            "unexpected injected-build failure: {message}"
        );
        assert!(
            !nonlinear_aot_ready_marker_path(&previous).exists(),
            "failed replacement must not leave an old ready marker published: {}",
            nonlinear_aot_ready_marker_path(&previous).display()
        );
        assert!(
            previous.expected_cdylib.exists(),
            "quarantine must not delete a DLL that may be loaded by another owner"
        );
    }

    #[test]
    fn nonlinear_aot_stale_lock_path_is_reusable() {
        let output_dir = tempdir().expect("AOT output directory should exist");
        let problem_key = "stale-lock-path-test";
        let lock_path = nonlinear_aot_file_lock_path(output_dir.path(), problem_key);
        std::fs::write(&lock_path, "pid=stale problem_key=stale-lock-path-test\n")
            .expect("stale lock path should be writable");

        let lock =
            acquire_nonlinear_aot_file_lock(output_dir.path(), problem_key, Duration::from_secs(1))
                .expect("an unowned stale lock path must be reusable");
        drop(lock);
        let contents =
            std::fs::read_to_string(&lock_path).expect("lock metadata should be readable");
        assert!(contents.contains(&format!("pid={}", std::process::id())));
    }

    #[test]
    #[ignore = "builds in one process and verifies RequirePrebuilt discovery in a fresh process"]
    fn cross_process_build_if_missing_then_require_prebuilt_reuses_artifact() {
        const CHILD_ENV: &str = "NONLINEAR_AOT_CROSS_PROCESS_CHILD";
        const OUTPUT_ENV: &str = "NONLINEAR_AOT_CROSS_PROCESS_OUTPUT";
        const TEST_NAME: &str = "numerical::Nonlinear_systems::symbolic_generated::tests::cross_process_build_if_missing_then_require_prebuilt_reuses_artifact";

        if std::env::var_os(CHILD_ENV).is_some() {
            let output_parent = PathBuf::from(
                std::env::var_os(OUTPUT_ENV).expect("cross-process output directory should be set"),
            );
            let warm = SymbolicNonlinearProblem::from_strings_with_generated_backend(
                elementary_equations(),
                elementary_options(),
                SymbolicGeneratedBackendConfig::require_prebuilt()
                    .with_output_parent_dir(Some(output_parent)),
            )
            .expect("fresh process should discover the persisted AOT artifact");
            assert_eq!(
                warm.selected_backend,
                SelectedSymbolicNonlinearBackendKind::AotCompiled
            );
            assert!(warm.build_result.is_none());
            let x0 = DVector::from_vec(vec![3.0, -1.0]);
            let residual = warm
                .problem
                .residual(&x0)
                .expect("AOT residual should load");
            assert_relative_eq!(residual[0], 0.0, epsilon = 1e-12);
            assert_relative_eq!(residual[1], 0.0, epsilon = 1e-12);
            println!(
                "[Nonlinear AOT cross-process] child selected={:?} build_result=false residual=[{:.3e}, {:.3e}]",
                warm.selected_backend, residual[0], residual[1]
            );
            return;
        }

        let _guard = aot_solver_test_guard();
        let output_dir = tempdir().expect("AOT output directory should exist");
        let cold = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            elementary_equations(),
            elementary_options(),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(output_dir.path().to_path_buf())),
        )
        .expect("parent process should build the AOT artifact");
        assert_eq!(
            cold.selected_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        let build = cold
            .build_result
            .as_ref()
            .expect("cold process should report a build");
        assert!(build.expected_cdylib.exists());
        assert!(nonlinear_aot_ready_marker_path(build).exists());
        let problem_key = cold
            .problem
            .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
            .problem_key();
        assert!(nonlinear_aot_ready_marker_matches(build, &problem_key));

        drop(cold);
        unregister_linked_dense_backend(&problem_key);
        let child = Command::new(
            std::env::current_exe().expect("current test executable should be available"),
        )
        .arg("--exact")
        .arg(TEST_NAME)
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_ENV, "1")
        .env(OUTPUT_ENV, output_dir.path())
        .output()
        .expect("cross-process child should start");
        assert!(
            child.status.success(),
            "fresh-process RequirePrebuilt acceptance failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        let child_stdout = String::from_utf8_lossy(&child.stdout);
        assert!(
            child_stdout.contains("build_result=false"),
            "child did not report a prebuilt reuse\nstdout:\n{}\nstderr:\n{}",
            child_stdout,
            String::from_utf8_lossy(&child.stderr)
        );
        assert!(
            child_stdout.contains("residual=["),
            "child did not report its AOT residual\nstdout:\n{}\nstderr:\n{}",
            child_stdout,
            String::from_utf8_lossy(&child.stderr)
        );
    }

    #[test]
    #[ignore = "verifies that an OS advisory lock is released after an owner process exits"]
    fn cross_process_lock_release_after_owner_exit() {
        const CHILD_ENV: &str = "NONLINEAR_AOT_LOCK_CRASH_CHILD";
        const TEST_NAME: &str = "numerical::Nonlinear_systems::symbolic_generated::tests::cross_process_lock_release_after_owner_exit";
        let output_dir = tempdir().expect("AOT output directory should exist");
        let problem_key = "crash-recovery-lock-test";

        if std::env::var_os(CHILD_ENV).is_some() {
            let _lock = acquire_nonlinear_aot_file_lock(
                &output_dir_path_from_env(),
                problem_key,
                Duration::from_secs(1),
            )
            .expect("child should acquire the lifecycle lock");
            println!("[Nonlinear AOT crash recovery] child acquired lock and exits");
            std::process::exit(0);
        }

        let child = Command::new(
            std::env::current_exe().expect("current test executable should be available"),
        )
        .arg("--exact")
        .arg(TEST_NAME)
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_ENV, "1")
        .env("NONLINEAR_AOT_LOCK_OUTPUT", output_dir.path())
        .output()
        .expect("crash-recovery child should start");
        assert!(
            child.status.success(),
            "lock-owner child failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );

        let lock =
            acquire_nonlinear_aot_file_lock(output_dir.path(), problem_key, Duration::from_secs(1))
                .expect("parent should acquire the lock after child process exit");
        drop(lock);
        let contents =
            std::fs::read_to_string(nonlinear_aot_file_lock_path(output_dir.path(), problem_key))
                .expect("lock metadata should remain readable");
        assert!(contents.contains(&format!("pid={}", std::process::id())));

        fn output_dir_path_from_env() -> PathBuf {
            PathBuf::from(
                std::env::var_os("NONLINEAR_AOT_LOCK_OUTPUT")
                    .expect("crash-recovery output directory should be set"),
            )
        }
    }

    #[test]
    #[ignore = "builds the same nonlinear AOT artifact concurrently from two threads"]
    fn concurrent_build_if_missing_serializes_shared_artifact_lifecycle() {
        let _guard = aot_solver_test_guard();
        let output_dir = tempdir().expect("AOT output directory should exist");
        let output_parent = output_dir.path().to_path_buf();

        let workers = (0..2)
            .map(|_| {
                let output_parent = output_parent.clone();
                std::thread::spawn(move || {
                    let prepared = SymbolicNonlinearProblem::from_strings_with_generated_backend(
                        parameterized_equations(),
                        parameterized_options(),
                        SymbolicGeneratedBackendConfig::defaults()
                            .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                                profile: AotBuildProfile::Debug,
                            })
                            .with_output_parent_dir(Some(output_parent)),
                    )
                    .map_err(|error| error.to_string())?;
                    let key = prepared
                        .problem
                        .prepare_dense_aot_problem(SymbolicDenseAotOptions::default())
                        .problem_key();
                    Ok::<_, String>((prepared.selected_backend, key))
                })
            })
            .collect::<Vec<_>>();

        let results = workers
            .into_iter()
            .map(|worker| worker.join().expect("AOT worker should not panic"))
            .collect::<Result<Vec<_>, _>>()
            .expect("both concurrent AOT builds should succeed");

        assert_eq!(results.len(), 2);
        assert!(
            results.iter().all(|(backend, _)| {
                *backend == SelectedSymbolicNonlinearBackendKind::AotCompiled
            })
        );
        assert_eq!(results[0].1, results[1].1);
        assert!(unregister_linked_dense_backend(&results[0].1).is_some());
    }
}
