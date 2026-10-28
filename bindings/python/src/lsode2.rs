//! Python adapter for the LSODE2 IVP facade.
//!
//! The adapter exposes the stable symbolic Lambdify slice of LSODE2: both
//! ExprLegacy and AtomView preparation, Dense/Sparse/Banded layouts, and the
//! Sequential/Parallel/Auto callback policy. Python-owned analytical
//! callbacks are available on the native Dense, Sparse and Banded routes.
//! AOT uses the native lifecycle configuration; Python callbacks remain a
//! Lambdify-only contract because a Python callable cannot be embedded into a
//! generated artifact.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nalgebra::DVector;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::path::PathBuf;

use rustedscithe_core::numerical::LSODE2::{
    IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetryMode, Lsode2AotProfile,
    Lsode2AotToolchain, Lsode2Error, Lsode2LinearSystemStructure, Lsode2Method,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use rustedscithe_core::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use rustedscithe_core::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind;
use rustedscithe_core::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use rustedscithe_core::symbolic::symbolic_ivp::IvpBackendError;
use rustedscithe_core::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};

use crate::contract::SolverOptions;
use crate::errors::{
    AotArtifactError, AotCompilerError, AotLifecycleError, AotLinkError, AotTimeoutError,
    CallbackFailureState, InvalidConfigurationError, PreparationError, SolverError,
    SolverPanicState, UnsupportedOptionError, annotate, catch_panic,
};
use crate::expr::PyExpr;
use crate::telemetry::write_report;
use crate::parameter_loop::{PyIvpParameterLoopPoint, run_parameter_loop};

/// Adapt Python callbacks to LSODE2's `Send + Sync` Rust callback contract.
///
/// The numerical core cannot return a `PyErr` from its callback type, so a
/// callback failure is recorded out-of-band and the callback returns NaNs to
/// force a safe numerical failure. `solve()` checks the recorded error first
/// and exposes it as `CallbackError` instead of leaking a misleading linear
/// algebra or convergence message.
pub(crate) fn build_python_callbacks(
    residual: Py<PyAny>,
    jacobian: Py<PyAny>,
    dimension: usize,
) -> PyResult<(
    rustedscithe_core::numerical::LSODE2::Lsode2AnalyticalCallbacks,
    CallbackFailureState,
)> {
    let failure = CallbackFailureState::default();
    Python::attach(|py| {
        if !residual.bind(py).is_callable() || !jacobian.bind(py).is_callable() {
            return Err(InvalidConfigurationError::new_err(
                "residual_callback and jacobian_callback must both be callable",
            ));
        }
        let residual = residual.clone_ref(py);
        let jacobian = jacobian.clone_ref(py);
        let residual_failure = failure.clone();
        let jacobian_failure = failure.clone();

        let residual_callback = move |time: f64, state: &DVector<f64>| -> DVector<f64> {
            let result = Python::attach(|py| -> PyResult<Vec<f64>> {
                let value = residual.bind(py).call1((time, state.as_slice().to_vec()))?;
                value.extract::<Vec<f64>>()
            });
            match result {
                Ok(values) if values.len() == dimension && values.iter().all(|v| v.is_finite()) => {
                    DVector::from_vec(values)
                }
                Ok(values) => {
                    residual_failure.record(
                        format!(
                            "residual_callback must return {dimension} finite values, got {}",
                            values.len()
                        ),
                        None,
                    );
                    DVector::from_element(dimension, f64::NAN)
                }
                Err(error) => {
                    residual_failure
                        .record(format!("residual_callback raised: {error}"), Some(error));
                    DVector::from_element(dimension, f64::NAN)
                }
            }
        };

        let jacobian_callback = move |time: f64, state: &DVector<f64>| {
            let result = Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
                let value = jacobian.bind(py).call1((time, state.as_slice().to_vec()))?;
                value.extract::<Vec<Vec<f64>>>()
            });
            match result {
                Ok(rows)
                    if rows.len() == dimension
                        && rows.iter().all(|row| {
                            row.len() == dimension && row.iter().all(|v| v.is_finite())
                        }) =>
                {
                    let flat = rows.into_iter().flatten().collect::<Vec<_>>();
                    nalgebra::DMatrix::from_row_slice(dimension, dimension, &flat)
                }
                Ok(rows) => {
                    jacobian_failure.record(
                        format!(
                            "jacobian_callback must return a {dimension}x{dimension} finite matrix, got {} rows",
                            rows.len()
                        ),
                        None,
                    );
                    nalgebra::DMatrix::from_element(dimension, dimension, f64::NAN)
                }
                Err(error) => {
                    jacobian_failure
                        .record(format!("jacobian_callback raised: {error}"), Some(error));
                    nalgebra::DMatrix::from_element(dimension, dimension, f64::NAN)
                }
            }
        };

        Ok((
            rustedscithe_core::numerical::LSODE2::Lsode2AnalyticalCallbacks {
                residual: Arc::new(residual_callback),
                jacobian: Arc::new(jacobian_callback),
            },
            failure,
        ))
    })
}

pub(crate) fn finite_vector(values: &[f64], label: &str) -> PyResult<()> {
    if values.is_empty() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must not be empty"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain only finite values"
        )));
    }
    Ok(())
}

pub(crate) fn validate_scalar(value: f64, label: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must be finite"
        )));
    }
    Ok(())
}

pub(crate) fn validate_names(names: &[String], label: &str) -> PyResult<()> {
    if names.is_empty() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must not be empty"
        )));
    }
    if names.iter().any(|name| name.trim().is_empty()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must not contain empty names"
        )));
    }
    let mut seen = HashSet::with_capacity(names.len());
    if names.iter().any(|name| !seen.insert(name.as_str())) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain unique names"
        )));
    }
    Ok(())
}

pub(crate) fn merge_band_width(
    explicit: Option<usize>,
    configured: Option<usize>,
    label: &str,
) -> PyResult<Option<usize>> {
    match (explicit, configured) {
        (Some(explicit), Some(configured)) if explicit != configured => {
            Err(InvalidConfigurationError::new_err(format!(
                "conflicting {label}: constructor value {explicit} differs from SolverOptions value {configured}"
            )))
        }
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

pub(crate) fn validate_route(
    options: &SolverOptions,
    band_lower: Option<usize>,
    band_upper: Option<usize>,
) -> PyResult<(
    Lsode2SymbolicAssemblyBackend,
    Lsode2LinearSystemStructure,
    IvpLambdifyExecutionPolicy,
    IvpTelemetry,
    Lsode2SymbolicExecutionMode,
)> {
    options.validate_aot_options()?;
    crate::contract::validate_backend_options_for(options, "lsode2")?;
    let frontend = match options.frontend() {
        "expr_legacy" => Lsode2SymbolicAssemblyBackend::ExprLegacy,
        "atom_native" => Lsode2SymbolicAssemblyBackend::AtomView,
        _ => {
            return Err(UnsupportedOptionError::new_err(
                "LSODE2 supports frontend='expr_legacy' or 'atom_native'",
            ));
        }
    };
    let execution = match options.execution() {
        "lambdify" => Lsode2SymbolicExecutionMode::LambdifyExpr,
        "aot" => Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CGcc,
            profile: Lsode2AotProfile::Release,
        },
        _ => unreachable!("SolverOptions validates execution before this route"),
    };
    let structure = match options.matrix_layout() {
        "dense" => {
            if band_lower.is_some() || band_upper.is_some() {
                return Err(InvalidConfigurationError::new_err(
                    "band_lower/band_upper are only valid for matrix_layout='banded'",
                ));
            }
            Lsode2LinearSystemStructure::Dense
        }
        "sparse" => {
            if band_lower.is_some() || band_upper.is_some() {
                return Err(InvalidConfigurationError::new_err(
                    "band_lower/band_upper are only valid for matrix_layout='banded'",
                ));
            }
            Lsode2LinearSystemStructure::Sparse
        }
        "banded" => match (band_lower, band_upper) {
            (Some(kl), Some(ku)) => Lsode2LinearSystemStructure::Banded { kl, ku },
            (None, None) => Lsode2LinearSystemStructure::BandedAuto,
            _ => {
                return Err(InvalidConfigurationError::new_err(
                    "band_lower and band_upper must be provided together, or both omitted for symbolic bandwidth inference",
                ));
            }
        },
        _ => {
            return Err(UnsupportedOptionError::new_err(
                "LSODE2 supports matrix_layout='dense', 'sparse' or 'banded'",
            ));
        }
    };
    let execution_policy = match options.execution_policy() {
        "sequential" => IvpLambdifyExecutionPolicy::Sequential,
        "parallel" => IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        "auto" => IvpLambdifyExecutionPolicy::Auto { min_work: 1 },
        _ => {
            return Err(InvalidConfigurationError::new_err(
                "unknown LSODE2 execution policy",
            ));
        }
    };
    let telemetry = match options.telemetry() {
        "off" => IvpTelemetry::disabled(),
        "counters" => IvpTelemetry::counters(),
        "timings" => IvpTelemetry::timings(),
        "detailed" => IvpTelemetry::detailed(),
        _ => {
            return Err(InvalidConfigurationError::new_err(
                "unknown telemetry mode in SolverOptions",
            ));
        }
    };
    Ok((frontend, structure, execution_policy, telemetry, execution))
}

/// Translate the Python AOT vocabulary into the native generated-backend
/// policy. Code generation stays entirely in the core crate; this adapter only
/// validates and forwards the lifecycle choice and toolchain metadata.
pub(crate) fn aot_backend_config(
    options: &SolverOptions,
) -> PyResult<SymbolicIvpGeneratedBackendConfig> {
    let profile = match options.aot_profile() {
        "debug" => AotBuildProfile::Debug,
        "release" => AotBuildProfile::Release,
        _ => unreachable!("SolverOptions validates the AOT profile"),
    };
    let policy = match options.aot_policy() {
        "require_prebuilt" => SymbolicIvpAotBuildPolicy::RequirePrebuilt,
        "build_if_missing" => SymbolicIvpAotBuildPolicy::BuildIfMissing { profile },
        "rebuild_always" => SymbolicIvpAotBuildPolicy::RebuildAlways { profile },
        _ => unreachable!("SolverOptions validates the AOT policy"),
    };
    let mut config = SymbolicIvpGeneratedBackendConfig::new().with_build_policy(policy);
    match options.aot_toolchain() {
        "gcc" => {
            config = config
                .with_aot_codegen_backend(AotCodegenBackend::C)
                .with_aot_c_compiler("gcc");
        }
        "tcc" => {
            config = config
                .with_aot_codegen_backend(AotCodegenBackend::C)
                .with_aot_c_compiler("tcc");
        }
        "zig" => config = config.with_aot_codegen_backend(AotCodegenBackend::Zig),
        "rust" => config = config.with_aot_codegen_backend(AotCodegenBackend::Rust),
        _ => unreachable!("SolverOptions validates the AOT toolchain"),
    }
    if let Some(output_dir) = options.aot_output_dir() {
        config = config.with_output_parent_dir(Some(PathBuf::from(output_dir)));
    }
    if let Some(handoff_path) = options.aot_handoff_path() {
        config = config.with_handoff_path(Some(PathBuf::from(handoff_path)));
    }
    Ok(config)
}

pub(crate) fn validate_expression_symbols(
    expressions: &[PyExpr],
    state_variables: &[String],
    time_variable: &str,
    parameters: &[String],
) -> PyResult<()> {
    let allowed: HashSet<&str> = state_variables
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(time_variable))
        .chain(parameters.iter().map(String::as_str))
        .collect();
    for (index, expression) in expressions.iter().enumerate() {
        if let Some(unknown) = expression
            .inner
            .extract_variables()
            .1
            .iter()
            .find(|name| !allowed.contains(name.as_str()))
        {
            return Err(InvalidConfigurationError::new_err(format!(
                "RHS expression {index} references undeclared symbol '{unknown}'"
            )));
        }
    }
    Ok(())
}

pub(crate) fn map_lsode2_error(error: Lsode2Error) -> PyErr {
    let native_kind = crate::errors::native_variant(&error);
    crate::errors::annotate_native(
        match error {
            Lsode2Error::UnsupportedBackend(message) => UnsupportedOptionError::new_err(message),
            Lsode2Error::InvalidConfig(message) => InvalidConfigurationError::new_err(message),
            Lsode2Error::GeneratedBackend(error) => map_ivp_backend_error(error),
            Lsode2Error::NativeStep(message) => SolverError::new_err(message),
        },
        "solve",
        "lsode2",
        "native",
        &native_kind,
    )
}

fn map_ivp_backend_error(error: IvpBackendError) -> PyErr {
    annotate(
        match error {
            IvpBackendError::AotArtifactMissing { message }
            | IvpBackendError::AotArtifactNotBuilt { message } => {
                AotArtifactError::new_err(message)
            }
            IvpBackendError::AotRuntimeUnavailable { message } => AotLinkError::new_err(message),
            IvpBackendError::AotBuildOutputDirMissing => InvalidConfigurationError::new_err(
                "AOT execution requires an output directory for generated artifacts",
            ),
            IvpBackendError::AotLifecycleFailure {
                stage,
                kind,
                artifact_key,
                message,
            } => {
                let detail =
                    format!("{message} (stage={stage:?}, kind={kind:?}, artifact={artifact_key})");
                match kind {
                    AotFailureKind::Compiler => AotCompilerError::new_err(detail),
                    AotFailureKind::Link => AotLinkError::new_err(detail),
                    AotFailureKind::Timeout => AotTimeoutError::new_err(detail),
                    AotFailureKind::PartialArtifact | AotFailureKind::StaleArtifact => {
                        AotArtifactError::new_err(detail)
                    }
                    _ => AotLifecycleError::new_err(detail),
                }
            }
            other => PreparationError::new_err(other.to_string()),
        },
        "preparation",
        "lsode2",
        "aot",
    )
}

/// Detached LSODE2 trajectory and stage-aware telemetry snapshot.
///
/// The result records the selected and executed controller family, switching
/// diagnostics, detached trajectory, and native telemetry. Arrays returned by
/// NumPy accessors own their storage and do not borrow the solver.
#[pyclass(name = "Lsode2Solution", module = "rustedscithe")]
#[derive(Clone)]
pub struct Lsode2Solution {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    status: String,
    method: String,
    controller_mode: String,
    active_family: String,
    executed_family: Option<String>,
    switch_reason: String,
    switch_uses_fallback: bool,
    telemetry_mode: String,
    evaluation_scope: String,
    evaluation_counter_note: String,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
    aot_artifact_keys: Vec<String>,
    aot_metadata: HashMap<String, String>,
    aot_counters: HashMap<String, u64>,
    aot_timings_ms: HashMap<String, f64>,
}

impl Lsode2Solution {
    pub(crate) fn from_solver(solver: &Lsode2Solver) -> Self {
        let summary = solver.summary();
        let (times, values) = solver.get_result();
        let statistics = summary.statistics;
        let native = summary.native_statistics;
        let evaluation = summary.evaluation_telemetry;
        let ivp_telemetry = solver.config().telemetry.snapshot();
        let (aot_metadata, aot_artifact_keys, aot_counters, aot_timings_ms) =
            crate::bvp::aot_telemetry_maps(Some(&ivp_telemetry));
        let telemetry_mode = match solver.config().telemetry.mode() {
            IvpTelemetryMode::Off => "off",
            IvpTelemetryMode::Counters => "counters",
            IvpTelemetryMode::Timings => "timings",
            IvpTelemetryMode::Detailed => "detailed",
        };
        let mut counters = HashMap::from([
            (
                "backend_prepare_calls".to_owned(),
                statistics.backend_prepare_calls as u64,
            ),
            ("solve_calls".to_owned(), statistics.solve_calls as u64),
            ("step_calls".to_owned(), statistics.step_calls as u64),
            (
                "accepted_steps".to_owned(),
                statistics.accepted_steps_total as u64,
            ),
            (
                "rejected_steps".to_owned(),
                statistics.rejected_step_attempts_total as u64,
            ),
            (
                "residual_calls".to_owned(),
                statistics.residual_calls as u64,
            ),
            (
                "jacobian_calls".to_owned(),
                statistics.jacobian_calls as u64,
            ),
            (
                "linear_solve_calls".to_owned(),
                statistics.linear_solve_attempts_total as u64,
            ),
            (
                "evaluation_residual_calls".to_owned(),
                evaluation.residual_evaluations as u64,
            ),
            (
                "evaluation_jacobian_calls".to_owned(),
                evaluation.jacobian_evaluations as u64,
            ),
            (
                "evaluation_linear_solves".to_owned(),
                evaluation.linear_solves as u64,
            ),
            (
                "evaluation_accepted_steps".to_owned(),
                evaluation.accepted_steps as u64,
            ),
            (
                "evaluation_rejected_steps".to_owned(),
                evaluation.rejected_steps as u64,
            ),
            (
                "nonlinear_solves".to_owned(),
                statistics.nonlinear_solve_calls as u64,
            ),
            (
                "algorithm_decisions".to_owned(),
                native.algorithm_decision_calls as u64,
            ),
            (
                "native_step_attempts".to_owned(),
                native.native_step_attempts as u64,
            ),
            (
                "native_jacobian_refreshes".to_owned(),
                native.native_jacobian_refresh_requests as u64,
            ),
            (
                "aot_resolution_hits".to_owned(),
                ivp_telemetry.aot_resolution_hits,
            ),
            (
                "aot_resolution_misses".to_owned(),
                ivp_telemetry.aot_resolution_misses,
            ),
            ("aot_reconnects".to_owned(), ivp_telemetry.aot_reconnects),
            (
                "aot_build_attempts".to_owned(),
                ivp_telemetry.aot_build_attempts,
            ),
            (
                "aot_build_retries".to_owned(),
                ivp_telemetry.aot_build_retries,
            ),
            (
                "aot_build_successes".to_owned(),
                ivp_telemetry.aot_build_successes,
            ),
            (
                "aot_build_failures".to_owned(),
                ivp_telemetry.aot_build_failures,
            ),
            (
                "aot_link_attempts".to_owned(),
                ivp_telemetry.aot_link_attempts,
            ),
            (
                "aot_link_successes".to_owned(),
                ivp_telemetry.aot_link_successes,
            ),
            (
                "aot_link_failures".to_owned(),
                ivp_telemetry.aot_link_failures,
            ),
            (
                "aot_runtime_ready".to_owned(),
                ivp_telemetry.aot_runtime_ready,
            ),
            (
                "aot_chunk_dispatches".to_owned(),
                ivp_telemetry.aot_chunk_dispatches,
            ),
            (
                "aot_dispatch_requests".to_owned(),
                ivp_telemetry.aot_dispatch_requests,
            ),
            (
                "aot_parallel_dispatches".to_owned(),
                ivp_telemetry.aot_parallel_dispatches,
            ),
            (
                "aot_completed_dispatches".to_owned(),
                ivp_telemetry.aot_completed_dispatches,
            ),
            (
                "aot_failed_dispatches".to_owned(),
                ivp_telemetry.aot_failed_dispatches,
            ),
        ]);
        let mut timings_ms = HashMap::from([
            (
                "preparation_ms".to_owned(),
                statistics.backend_prepare_ms_total,
            ),
            ("solve_ms".to_owned(), statistics.solve_ms_total),
            (
                "integration_loop_ms".to_owned(),
                statistics.integration_loop_ms_total,
            ),
            (
                "linear_factorization_ms".to_owned(),
                statistics.linear_factorization_ms_total,
            ),
            (
                "linear_matrix_assembly_ms".to_owned(),
                statistics.linear_matrix_assembly_ms_total,
            ),
            (
                "linear_solve_ms".to_owned(),
                statistics.linear_solve_ms_total,
            ),
            ("residual_ms".to_owned(), statistics.residual_ms_total),
            ("jacobian_ms".to_owned(), statistics.jacobian_ms_total),
            (
                "evaluation_residual_ms".to_owned(),
                evaluation.residual_ms_total,
            ),
            (
                "evaluation_jacobian_ms".to_owned(),
                evaluation.jacobian_ms_total,
            ),
            (
                "evaluation_linear_solve_ms".to_owned(),
                evaluation.linear_ms_total,
            ),
            (
                "native_linear_solve_ms".to_owned(),
                native.native_linear_solve_ms_total,
            ),
            (
                "native_residual_ms".to_owned(),
                native.native_residual_ms_total,
            ),
            (
                "native_jacobian_ms".to_owned(),
                native.native_jacobian_ms_total,
            ),
        ]);
        crate::telemetry::apply_collection_mode(telemetry_mode, &mut counters, &mut timings_ms);
        Self {
            times: times.iter().copied().collect(),
            states: values
                .row_iter()
                .map(|row| row.iter().copied().collect())
                .collect(),
            status: summary.status,
            method: summary.method.to_owned(),
            controller_mode: summary.algorithm.controller_mode.to_owned(),
            active_family: summary.algorithm.active_family.to_owned(),
            executed_family: summary.algorithm.executed_family.map(str::to_owned),
            switch_reason: summary.algorithm.switch_reason.to_owned(),
            switch_uses_fallback: summary.algorithm.switch_uses_fallback,
            telemetry_mode: telemetry_mode.to_owned(),
            evaluation_scope: evaluation.scope.label().to_owned(),
            evaluation_counter_note: evaluation.scope.counter_note().to_owned(),
            counters,
            timings_ms,
            aot_artifact_keys,
            aot_metadata,
            aot_counters,
            aot_timings_ms,
        }
    }
}

#[pymethods]
impl Lsode2Solution {
    /// Accepted output times, including the initial point.
    pub fn times(&self) -> Vec<f64> {
        self.times.clone()
    }

    /// State rows corresponding to [`Self::times`].
    pub fn states(&self) -> Vec<Vec<f64>> {
        self.states.clone()
    }

    /// Owned NumPy array with shape `(accepted_times, state_dimension)`.
    pub fn states_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        crate::numpy_support::matrix(py, &self.states)
    }

    /// Owned NumPy array containing the accepted output times.
    pub fn times_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.times)
    }

    #[getter]
    /// Native integration status captured for this trajectory.
    pub fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    /// LSODE2 method/controller alias used for this solve.
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Controller mode selected for this solve, for example `bdf_only` or
    /// `automatic_adams_bdf`. This is diagnostics, not a second solver knob.
    #[getter]
    pub fn controller_mode(&self) -> &str {
        &self.controller_mode
    }

    /// Method family active when the solve summary was captured.
    #[getter]
    pub fn active_family(&self) -> &str {
        &self.active_family
    }

    /// Method family actually executed by the native route, if known.
    #[getter]
    pub fn executed_family(&self) -> Option<&str> {
        self.executed_family.as_deref()
    }

    /// Typed controller explanation retained as a stable diagnostic label.
    #[getter]
    pub fn switch_reason(&self) -> &str {
        &self.switch_reason
    }

    #[getter]
    /// Whether automatic controller switching used its fallback path.
    pub fn switch_uses_fallback(&self) -> bool {
        self.switch_uses_fallback
    }

    #[getter]
    /// Final accepted time, or `None` if no output point was produced.
    pub fn final_time(&self) -> Option<f64> {
        self.times.last().copied()
    }

    #[getter]
    /// Final state vector, or `None` if no output point was produced.
    pub fn final_state(&self) -> Option<Vec<f64>> {
        self.states.last().cloned()
    }

    #[getter]
    /// Telemetry collection level captured for this solution.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    /// Counter/timer scope used for the effective LSODE2 evaluation statistics.
    ///
    /// Native faithful integration and the BDF bridge count different events;
    /// use the `evaluation_*` telemetry keys with this label when comparing runs.
    #[getter]
    pub fn evaluation_scope(&self) -> &str {
        &self.evaluation_scope
    }

    /// Explain which layer the effective residual/Jacobian counters describe.
    #[getter]
    pub fn evaluation_counter_note(&self) -> &str {
        &self.evaluation_counter_note
    }

    /// Return discrete solver counters; the map is empty when collection is off.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    /// Return measured stage durations in milliseconds; empty when telemetry
    /// is off or counters-only.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Artifact keys observed during AOT preparation/publication.
    ///
    /// The list is detached from native state so a producer can persist it
    /// alongside `provenance()` and a consumer can verify which generated
    /// artifact was reconnected after a process boundary.
    pub fn aot_artifact_keys(&self) -> Vec<String> {
        self.aot_artifact_keys.clone()
    }

    /// Return the selected execution route and native snapshot metadata.
    pub fn telemetry_aot_metadata(&self) -> HashMap<String, String> {
        self.aot_metadata.clone()
    }

    /// Return generated-backend lifecycle counters collected in the selected mode.
    pub fn telemetry_aot_counters(&self) -> HashMap<String, u64> {
        self.aot_counters.clone()
    }

    /// Return cold/warm preparation and generated-route stage timings in milliseconds.
    pub fn telemetry_aot_timings_ms(&self) -> HashMap<String, f64> {
        self.aot_timings_ms.clone()
    }

    /// Return a common result snapshot while retaining LSODE2-specific
    /// controller diagnostics in the same mapping.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        let mut diagnostics = HashMap::from([
            ("solver".to_owned(), "lsode2".to_owned()),
            ("status".to_owned(), self.status.clone()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
            ("evaluation_scope".to_owned(), self.evaluation_scope.clone()),
            (
                "evaluation_counter_note".to_owned(),
                self.evaluation_counter_note.clone(),
            ),
            ("sample_count".to_owned(), self.times.len().to_string()),
            (
                "state_dimension".to_owned(),
                self.states.first().map_or(0, Vec::len).to_string(),
            ),
            (
                "final_time".to_owned(),
                self.times
                    .last()
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
            ),
            ("controller_mode".to_owned(), self.controller_mode.clone()),
            ("active_family".to_owned(), self.active_family.clone()),
            ("switch_reason".to_owned(), self.switch_reason.clone()),
            (
                "switch_uses_fallback".to_owned(),
                self.switch_uses_fallback.to_string(),
            ),
        ]);
        if let Some(family) = self.executed_family.as_deref() {
            diagnostics.insert("executed_family".to_owned(), family.to_owned());
        }
        diagnostics
    }

    /// Compact deterministic text for batch logs; nested stages remain named
    /// explicitly and must not be added as independent wall-clock scopes.
    pub fn telemetry_report(&self) -> String {
        let mut counters = self.counters.iter().collect::<Vec<_>>();
        counters.sort_by(|left, right| left.0.cmp(right.0));
        let mut timings = self.timings_ms.iter().collect::<Vec<_>>();
        timings.sort_by(|left, right| left.0.cmp(right.0));
        let counters = counters
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(" ");
        let timings = timings
            .into_iter()
            .map(|(key, value)| format!("{key}={value:.6}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "telemetry={}; method={}; controller={}; active_family={}; executed_family={}; switch_reason={}; fallback={}; counters: {}; timings_ms: {}",
            self.telemetry_mode,
            self.method,
            self.controller_mode,
            self.active_family,
            self.executed_family.as_deref().unwrap_or("unknown"),
            self.switch_reason,
            self.switch_uses_fallback,
            counters,
            timings
        )
    }

    /// Write only the compact report, without capturing process output.
    pub fn write_telemetry_report(&self, path: &str) -> PyResult<()> {
        write_report(path, &self.telemetry_report())
    }

    fn __repr__(&self) -> String {
        format!(
            "Lsode2Solution(points={}, state_dimension={}, method='{}', status='{}')",
            self.times.len(),
            self.states.first().map_or(0, Vec::len),
            self.method,
            self.status
        )
    }
}

/// Prepared LSODE2 symbolic IVP adapter.
///
/// Select `method="bdf"`, `"adams"`, or `"auto"` independently from the
/// symbolic frontend, execution route, matrix layout and callback execution
/// policy in [`SolverOptions`]. The adapter supports symbolic equations and
/// optional native residual/Jacobian callbacks. `prepare()` builds the
/// selected route without stepping; time-segment solves reuse it, while
/// `restart()` changes the initial state and interval.
#[pyclass(name = "Lsode2Solver", module = "rustedscithe", unsendable)]
pub struct Lsode2PythonSolver {
    solver: Lsode2Solver,
    parameter_count: usize,
    method: String,
    frontend: String,
    execution: String,
    matrix_layout: String,
    execution_policy: String,
    telemetry_mode: String,
    aot_policy: String,
    aot_toolchain: String,
    aot_profile: String,
    aot_output_dir: Option<String>,
    aot_handoff_path: Option<String>,
    // Continuation must begin at the last successful accepted state rather
    // than silently falling back to the original initial condition.
    last_time: Option<f64>,
    last_state: Option<Vec<f64>>,
    callback_failure: Option<CallbackFailureState>,
    panic_state: SolverPanicState,
}

#[pymethods]
impl Lsode2PythonSolver {
    /// Configure an LSODE2 initial-value problem and controller family.
    ///
    /// Provide one RHS expression per state variable, initial state `y0`,
    /// interval `[t0, t_end]`, and step/tolerance controls. `method` selects
    /// the native BDF, Adams or automatic controller. `options` independently
    /// selects frontend, execution route, layout, policy and telemetry.
    /// Optional callback arguments provide a residual and, when available, an
    /// analytical Jacobian; omit the Jacobian to use the native fallback.
    #[new]
    #[pyo3(signature=(expressions, state_variables, time_variable, y0, t0, t_end, max_step, rtol=1e-6, atol=1e-8, options=None, method="bdf", band_lower=None, band_upper=None, parameters=None, parameter_values=None, residual_callback=None, jacobian_callback=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        time_variable: String,
        y0: Vec<f64>,
        t0: f64,
        t_end: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        options: Option<PyRef<'_, SolverOptions>>,
        method: &str,
        band_lower: Option<usize>,
        band_upper: Option<usize>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        residual_callback: Option<Py<PyAny>>,
        jacobian_callback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables")?;
        finite_vector(&y0, "y0")?;
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if time_variable.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "time_variable must not be empty",
            ));
        }
        if state_variables.iter().any(|name| name == &time_variable) {
            return Err(InvalidConfigurationError::new_err(
                "time_variable must differ from state_variables",
            ));
        }
        let parameters = parameters.unwrap_or_default();
        if !parameters.is_empty() {
            validate_names(&parameters, "parameters")?;
        }
        if parameters
            .iter()
            .any(|name| name == &time_variable || state_variables.iter().any(|state| state == name))
        {
            return Err(InvalidConfigurationError::new_err(
                "parameters must not collide with time_variable or state_variables",
            ));
        }
        if let Some(values) = &parameter_values {
            finite_vector(values, "parameter_values")?;
            if values.len() != parameters.len() {
                return Err(InvalidConfigurationError::new_err(format!(
                    "expected {} parameter values, got {}",
                    parameters.len(),
                    values.len()
                )));
            }
        } else if !parameters.is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "parameter_values are required when parameters are declared",
            ));
        }
        validate_expression_symbols(&expressions, &state_variables, &time_variable, &parameters)?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        validate_scalar(max_step, "max_step")?;
        if max_step <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "max_step must be greater than zero",
            ));
        }
        validate_scalar(rtol, "rtol")?;
        validate_scalar(atol, "atol")?;
        if rtol <= 0.0 || atol <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "rtol and atol must be greater than zero",
            ));
        }
        let method = method.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        let (requested_method, controller) = match method.as_str() {
            "bdf" => (Lsode2Method::Bdf, "bdf"),
            "adams" => (Lsode2Method::Adams, "adams"),
            "auto" | "automatic" | "lsoda" => (Lsode2Method::Auto, "auto"),
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "LSODE2 method must be 'bdf', 'adams' or 'auto' (alias: 'lsoda')",
                ));
            }
        };

        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let band_lower = merge_band_width(band_lower, options.band_lower(), "band_lower")?;
        let band_upper = merge_band_width(band_upper, options.band_upper(), "band_upper")?;
        let (frontend, structure, execution_policy, telemetry, execution) =
            validate_route(&options, band_lower, band_upper)?;
        let generated_backend = if options.execution() == "aot" {
            Some(aot_backend_config(&options)?)
        } else {
            None
        };
        let callback_pair = match (residual_callback, jacobian_callback) {
            (None, None) => None,
            (Some(_), None) | (None, Some(_)) => {
                return Err(InvalidConfigurationError::new_err(
                    "residual_callback and jacobian_callback must be provided together",
                ));
            }
            (Some(residual), Some(jacobian)) => {
                if structure == Lsode2LinearSystemStructure::BandedAuto {
                    return Err(UnsupportedOptionError::new_err(
                        "automatic banded layout requires symbolic expressions; numerical Python callbacks must provide band_lower and band_upper",
                    ));
                }
                if !matches!(execution, Lsode2SymbolicExecutionMode::LambdifyExpr) {
                    return Err(UnsupportedOptionError::new_err(
                        "Python analytical callbacks require execution='lambdify'; use symbolic expressions for execution='aot'",
                    ));
                }
                Some(build_python_callbacks(residual, jacobian, y0.len())?)
            }
        };
        let parameter_count = parameters.len();
        let mut config = Lsode2ProblemConfig::new(
            expressions
                .into_iter()
                .map(|expression| expression.inner)
                .collect(),
            state_variables,
            time_variable,
            t0,
            DVector::from_vec(y0),
            t_end,
            max_step,
            rtol,
            atol,
        )
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: frontend,
            execution,
        })
        .with_linear_system_structure(structure)
        .with_lambdify_execution_policy(execution_policy)
        .with_telemetry(telemetry);
        config.method = requested_method;
        config = match controller {
            "bdf" => config.with_bdf_only_controller(),
            "adams" => config.with_adams_only_controller(),
            "auto" => config.with_automatic_adams_bdf_controller(),
            _ => {
                return Err(InvalidConfigurationError::new_err(
                    "LSODE2 controller was not normalized before construction",
                ));
            }
        };
        if let Some(generated_backend) = generated_backend {
            config.backend.generated_backend = generated_backend;
        }
        if !parameters.is_empty() {
            config = config.with_equation_parameters(parameters);
            config = config.with_equation_parameter_values(DVector::from_vec(
                parameter_values.ok_or_else(|| {
                    InvalidConfigurationError::new_err("parameter values were not validated")
                })?,
            ));
        }
        if let Some((callbacks, _)) = &callback_pair {
            config = config.with_analytical_callback_bundle(callbacks.clone());
        }
        let route = format!(
            "{}:{}:{}",
            options.frontend(),
            options.execution(),
            options.matrix_layout()
        );
        let logging_level = crate::contract::native_logging_level(&options, "lsode2")?;
        let mut solver = catch_panic("construct", "lsode2", &route, || {
            Lsode2Solver::new(config).map_err(map_lsode2_error)
        })?;
        solver.set_logging_level(logging_level);
        Ok(Self {
            solver,
            parameter_count,
            method,
            frontend: options.frontend().to_owned(),
            execution: options.execution().to_owned(),
            matrix_layout: options.matrix_layout().to_owned(),
            execution_policy: options.execution_policy().to_owned(),
            telemetry_mode: options.telemetry().to_owned(),
            aot_policy: options.aot_policy().to_owned(),
            aot_toolchain: options.aot_toolchain().to_owned(),
            aot_profile: options.aot_profile().to_owned(),
            aot_output_dir: options.aot_output_dir().map(str::to_owned),
            aot_handoff_path: options.aot_handoff_path().map(str::to_owned),
            last_time: None,
            last_state: None,
            callback_failure: callback_pair.map(|(_, failure)| failure),
            panic_state: SolverPanicState::default(),
        })
    }

    /// Prepare symbolic callbacks and linear backend wiring without solving.
    /// Prepare the selected symbolic or callback route without integrating.
    ///
    /// Preparation can be called explicitly to measure setup separately;
    /// later solve and continuation calls reuse the prepared backend.
    pub fn prepare(&mut self) -> PyResult<()> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("prepare", "lsode2", &route, || {
            self.solver.prepare().map_err(map_lsode2_error)
        })
    }

    #[getter]
    /// Whether symbolic preparation has completed successfully.
    pub fn is_prepared(&self) -> bool {
        self.solver.is_prepared()
    }

    /// Integrate the configured initial-value problem and return its trajectory.
    pub fn solve(&mut self) -> PyResult<Lsode2Solution> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "lsode2", &route, || {
            if let Some(failure) = &self.callback_failure {
                failure.clear();
            }
            if let Err(error) = self.solver.solve_with_summary() {
                if let Some(callback_error) = self
                    .callback_failure
                    .as_ref()
                    .and_then(|failure| failure.take_pyerr("solve", "lsode2", &route))
                {
                    return Err(callback_error);
                }
                return Err(map_lsode2_error(error));
            }
            if let Some(callback_error) = self
                .callback_failure
                .as_ref()
                .and_then(|failure| failure.take_pyerr("solve", "lsode2", &route))
            {
                return Err(callback_error);
            }
            let solution = Lsode2Solution::from_solver(&self.solver);
            self.last_time = solution.final_time();
            self.last_state = solution.final_state();
            Ok(solution)
        })
    }

    /// Start a fresh solve with new parameter values and initial state.
    /// The configured interval and prepared callbacks are retained.
    pub fn solve_with_parameter_values(
        &mut self,
        y0: Vec<f64>,
        parameter_values: Vec<f64>,
    ) -> PyResult<Lsode2Solution> {
        solve_with_parameter_values_inner(self, None, y0, parameter_values)
    }

    /// Run independent solves in caller-specified parameter order. Every
    /// point uses this same initial state and configured time interval.
    pub fn solve_parameter_loop(
        &mut self,
        py: Python<'_>,
        y0: Vec<f64>,
        parameter_vectors: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Py<PyIvpParameterLoopPoint>>> {
        finite_vector(&y0, "y0")?;
        let route = format!("{}:{}:{}", self.frontend, self.execution, self.matrix_layout);
        run_parameter_loop(py, "lsode2", &route, parameter_vectors, |parameters| {
            let solution = self.solve_with_parameter_values(y0.clone(), parameters.to_vec())?;
            Py::new(py, solution).map(Py::into_any)
        })
    }

    /// Start a fresh solve with new parameters, initial state and interval.
    /// The prepared symbolic/AOT route is retained.
    pub fn solve_with_parameter_values_in_interval(
        &mut self,
        t0: f64,
        y0: Vec<f64>,
        t_end: f64,
        parameter_values: Vec<f64>,
    ) -> PyResult<Lsode2Solution> {
        solve_with_parameter_values_inner(self, Some((t0, t_end)), y0, parameter_values)
    }

    /// Restart integration without repeating symbolic preparation.
    /// Start a new segment from `y0` over `[t0, t_end]` using prepared callbacks.
    pub fn restart(&mut self, t0: f64, y0: Vec<f64>, t_end: f64) -> PyResult<Lsode2Solution> {
        finite_vector(&y0, "y0")?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "lsode2", &route, || {
            self.solver
                .try_restart_with_initial_state(t0, DVector::from_vec(y0), t_end)
                .map_err(map_lsode2_error)?;
            self.solve()
        })
    }

    /// Rebind declared parameters and continue from the last successful state.
    ///
    /// This is deliberately different from [`Self::restart`]: symbolic
    /// preparation and its compiled callbacks remain alive, while the
    /// numerical trajectory starts at the previous solution endpoint.
    /// Rebind declared parameters and continue from the last successful state.
    ///
    /// Continuation begins at the last accepted endpoint, not the original
    /// `y0`; use `restart()` to provide a new state or interval.
    pub fn continue_with_parameters(
        &mut self,
        parameter_values: Vec<f64>,
        t_end: f64,
    ) -> PyResult<Lsode2Solution> {
        let start_time = self.last_time.ok_or_else(|| {
            InvalidConfigurationError::new_err(
                "LSODE2 continuation requires one successful solve before rebinding parameters",
            )
        })?;
        let start_state = self.last_state.clone().ok_or_else(|| {
            InvalidConfigurationError::new_err(
                "LSODE2 continuation has no saved state from a successful solve",
            )
        })?;
        if !self.solver.is_prepared() {
            return Err(InvalidConfigurationError::new_err(
                "LSODE2 continuation requires a prepared model",
            ));
        }
        if parameter_values.len() != self.parameter_count {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected {} parameter values, got {}",
                self.parameter_count,
                parameter_values.len()
            )));
        }
        finite_vector(&parameter_values, "parameter_values")?;
        validate_scalar(t_end, "t_end")?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "lsode2", &route, || {
            self.solver
                .set_parameter_values(DVector::from_vec(parameter_values))
                .map_err(map_lsode2_error)?;
            self.solver
                .try_restart_with_initial_state(start_time, DVector::from_vec(start_state), t_end)
                .map_err(map_lsode2_error)?;
            self.solve()
        })
    }

    /// Return the native integration status for the current solver state.
    pub fn status(&self) -> String {
        self.solver.status().to_owned()
    }

    /// Return a compact deterministic report of controller state and collected telemetry.
    pub fn telemetry_report(&self) -> String {
        let summary = self.solver.summary();
        let algorithm = summary.algorithm;
        if self.telemetry_mode == "off" {
            return format!(
                "telemetry=off; method={}; status={}; controller={}; active_family={}; executed_family={}; switch_reason={}; fallback={}; evaluation_scope={}; evaluation_counter_note={}; counters: disabled timings_ms: disabled",
                summary.method,
                summary.status,
                algorithm.controller_mode,
                algorithm.active_family,
                algorithm.executed_family.unwrap_or("unknown"),
                algorithm.switch_reason,
                algorithm.switch_uses_fallback,
                summary.evaluation_telemetry.scope.label(),
                summary.evaluation_telemetry.scope.counter_note(),
            );
        }
        format!(
            "telemetry={}; method={}; status={}; controller={}; active_family={}; executed_family={}; switch_reason={}; fallback={}; evaluation_scope={}; evaluation_counter_note={}; {}",
            self.telemetry_mode,
            summary.method,
            summary.status,
            algorithm.controller_mode,
            algorithm.active_family,
            algorithm.executed_family.unwrap_or("unknown"),
            algorithm.switch_reason,
            algorithm.switch_uses_fallback,
            summary.evaluation_telemetry.scope.label(),
            summary.evaluation_telemetry.scope.counter_note(),
            summary.statistics.table_report()
        )
    }

    /// Return route and controller metadata needed to reproduce this solver configuration.
    pub fn provenance(&self) -> HashMap<String, String> {
        let algorithm = self.solver.algorithm_snapshot();
        HashMap::from([
            ("solver".to_owned(), "lsode2".to_owned()),
            ("method".to_owned(), self.method.clone()),
            (
                "controller_mode".to_owned(),
                algorithm.controller_mode.to_owned(),
            ),
            (
                "active_family".to_owned(),
                algorithm.active_family.to_owned(),
            ),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), self.matrix_layout.clone()),
            ("execution_policy".to_owned(), self.execution_policy.clone()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
            (
                "aot_policy".to_owned(),
                if self.execution == "aot" {
                    self.aot_policy.clone()
                } else {
                    "not_applicable".to_owned()
                },
            ),
            (
                "aot_toolchain".to_owned(),
                if self.execution == "aot" {
                    self.aot_toolchain.clone()
                } else {
                    "not_applicable".to_owned()
                },
            ),
            (
                "aot_profile".to_owned(),
                if self.execution == "aot" {
                    self.aot_profile.clone()
                } else {
                    "not_applicable".to_owned()
                },
            ),
            (
                "aot_output_dir".to_owned(),
                self.aot_output_dir
                    .as_deref()
                    .unwrap_or("default")
                    .to_owned(),
            ),
            (
                "aot_handoff_path".to_owned(),
                self.aot_handoff_path
                    .as_deref()
                    .unwrap_or("default")
                    .to_owned(),
            ),
        ])
    }

    #[getter]
    /// Telemetry collection level configured on this solver.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    fn __repr__(&self) -> String {
        format!(
            "Lsode2Solver(prepared={}, method='{}', execution='{}', layout='{}', telemetry='{}', aot_policy='{}')",
            self.is_prepared(),
            self.method,
            self.execution,
            self.matrix_layout,
            self.telemetry_mode,
            self.aot_policy
        )
    }
}

fn solve_with_parameter_values_inner(
    solver: &mut Lsode2PythonSolver,
    interval: Option<(f64, f64)>,
    y0: Vec<f64>,
    parameter_values: Vec<f64>,
) -> PyResult<Lsode2Solution> {
    finite_vector(&y0, "y0")?;
    finite_vector(&parameter_values, "parameter_values")?;
    if parameter_values.len() != solver.parameter_count {
        return Err(InvalidConfigurationError::new_err(format!(
            "expected {} parameter values, got {}",
            solver.parameter_count,
            parameter_values.len()
        )));
    }
    if let Some((t0, t_end)) = interval {
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
    }
    let route = format!(
        "{}:{}:{}",
        solver.frontend, solver.execution, solver.matrix_layout
    );
    let panic_state = solver.panic_state.clone();
    panic_state.run("solve", "lsode2", &route, || {
        if let Some(failure) = &solver.callback_failure {
            failure.clear();
        }
        let result = match interval {
            Some((t0, t_end)) => solver.solver.try_solve_with_parameter_values_in_interval(
                t0,
                DVector::from_vec(y0),
                t_end,
                DVector::from_vec(parameter_values),
            ),
            None => solver.solver.try_solve_with_parameter_values(
                DVector::from_vec(y0),
                DVector::from_vec(parameter_values),
            ),
        };
        if let Err(error) = result {
            if let Some(callback_error) = solver
                .callback_failure
                .as_ref()
                .and_then(|failure| failure.take_pyerr("solve", "lsode2", &route))
            {
                return Err(callback_error);
            }
            return Err(map_lsode2_error(error));
        }
        if let Some(callback_error) = solver
            .callback_failure
            .as_ref()
            .and_then(|failure| failure.take_pyerr("solve", "lsode2", &route))
        {
            return Err(callback_error);
        }
        let solution = Lsode2Solution::from_solver(&solver.solver);
        solver.last_time = solution.final_time();
        solver.last_state = solution.final_state();
        Ok(solution)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_requires_bandwidth_for_banded_layout() {
        let options = SolverOptions::new(
            Some("expr_legacy"),
            Some("lambdify"),
            Some("banded"),
            Some("sequential"),
            Some("off"),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(validate_route(&options, None, None).is_err());
        assert!(validate_route(&options, Some(1), Some(1)).is_ok());
    }
}
