//! Thin Python adapter for the stable second-generation Radau facade.
//!
//! This module intentionally exposes a complete lifecycle for the supported
//! symbolic routes instead of pretending that every native Radau combination
//! is already a Python contract. Symbolic Lambdify/AOT routes support both
//! frontends, Dense/Sparse/Banded storage and Sequential/Parallel/Auto
//! callback policies. Structured storage requires an analytic Jacobian so the
//! callback can write the selected layout directly; the dense finite-difference
//! fallback is rejected with a typed capability error.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use rustedscithe_core::numerical::Radau::api::{
    RadauConfig, RadauError, RadauErrorKind, RadauExecution, RadauExecutionPolicy, RadauFrontend,
    RadauJacobianSource, RadauMatrixLayout, RadauNativeSolver, RadauOutputPolicy, RadauProblem,
    RadauSolution as CoreRadauSolution, RadauSolver as CoreRadauSolver, RadauTelemetryMode,
};

use crate::aot::{generated_config, provenance_fields};
use crate::contract::SolverOptions;
use crate::errors::{
    CallbackFailureState, ExhaustionError, InvalidConfigurationError, PreparationError,
    SolverError, SolverPanicState, UnsupportedOptionError, annotate_native, catch_panic,
};
use crate::expr::PyExpr;
use crate::telemetry::write_report;

/// Adapt Python callbacks to the native Radau callback ABI.
///
/// The numerical core cannot return a `PyErr` from its `Send + Sync` callback
/// contract. A callback failure is therefore retained out-of-band and the
/// closure returns non-finite data to stop the numerical path safely. The
/// public solve method translates the retained cause into `CallbackError`.
fn build_native_callbacks(
    residual: Py<PyAny>,
    jacobian: Option<Py<PyAny>>,
    dimension: usize,
) -> PyResult<(
    Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>,
    Option<Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>>,
    CallbackFailureState,
)> {
    let failure = CallbackFailureState::default();
    Python::attach(|py| {
        if !residual.bind(py).is_callable() {
            return Err(InvalidConfigurationError::new_err(
                "residual_callback must be callable",
            ));
        }
        if let Some(jacobian) = &jacobian {
            if !jacobian.bind(py).is_callable() {
                return Err(InvalidConfigurationError::new_err(
                    "jacobian_callback must be callable when provided",
                ));
            }
        }
        let residual = residual.clone_ref(py);
        let residual_failure = failure.clone();
        let residual_callback: Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync> =
            Arc::new(move |time, state| {
                let result = Python::attach(|py| -> PyResult<Vec<f64>> {
                    residual
                        .bind(py)
                        .call1((time, state.as_slice().to_vec()))?
                        .extract()
                });
                match result {
                    Ok(values)
                        if values.len() == dimension && values.iter().all(|v| v.is_finite()) =>
                    {
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
            });

        let jacobian_callback = jacobian.map(|jacobian| {
            let jacobian = jacobian.clone_ref(py);
            let jacobian_failure = failure.clone();
            let callback: Arc<
                dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync,
            > = Arc::new(move |time, state| {
                let result = Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
                    jacobian
                        .bind(py)
                        .call1((time, state.as_slice().to_vec()))?
                        .extract()
                });
                match result {
                    Ok(rows)
                        if rows.len() == dimension
                            && rows.iter().all(|row| {
                                row.len() == dimension && row.iter().all(|v| v.is_finite())
                            }) =>
                    {
                        let flat = rows.into_iter().flatten().collect::<Vec<_>>();
                        DMatrix::from_row_slice(dimension, dimension, &flat)
                    }
                    Ok(rows) => {
                        jacobian_failure.record(
                            format!(
                                "jacobian_callback must return a {dimension}x{dimension} finite matrix, got {} rows",
                                rows.len()
                            ),
                            None,
                        );
                        DMatrix::from_element(dimension, dimension, f64::NAN)
                    }
                    Err(error) => {
                        jacobian_failure.record(
                            format!("jacobian_callback raised: {error}"),
                            Some(error),
                        );
                        DMatrix::from_element(dimension, dimension, f64::NAN)
                    }
                }
            });
            callback
        });
        Ok((residual_callback, jacobian_callback, failure))
    })
}

fn finite_vector(values: &[f64], label: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_names(names: &[String], label: &str, allow_empty: bool) -> PyResult<()> {
    if !allow_empty && names.is_empty() {
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

fn validate_scalar(value: f64, label: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must be finite"
        )));
    }
    Ok(())
}

fn validate_route(
    options: &SolverOptions,
) -> PyResult<(
    RadauFrontend,
    RadauTelemetryMode,
    RadauExecutionPolicy,
    RadauMatrixLayout,
)> {
    options.validate_aot_options()?;
    let frontend = match options.frontend() {
        "expr_legacy" => RadauFrontend::ExprLegacy,
        "atom_native" => RadauFrontend::AtomViewNative,
        _ => {
            return Err(UnsupportedOptionError::new_err(
                "Radau supports frontend='expr_legacy' or 'atom_native'",
            ));
        }
    };
    if !matches!(options.execution(), "lambdify" | "aot") {
        return Err(UnsupportedOptionError::new_err(
            "Radau execution must be 'lambdify' or 'aot'",
        ));
    }
    let matrix_layout = match options.matrix_layout() {
        "dense" => RadauMatrixLayout::Dense,
        "sparse" => RadauMatrixLayout::Sparse,
        "banded" => RadauMatrixLayout::Banded {
            lower: options.band_lower().ok_or_else(|| {
                InvalidConfigurationError::new_err(
                    "Banded Radau requires options.set_banded_width(lower, upper)",
                )
            })?,
            upper: options.band_upper().ok_or_else(|| {
                InvalidConfigurationError::new_err(
                    "Banded Radau requires options.set_banded_width(lower, upper)",
                )
            })?,
        },
        _ => {
            return Err(UnsupportedOptionError::new_err(
                "Radau matrix_layout must be 'dense', 'sparse' or 'banded'",
            ));
        }
    };
    let policy = match options.execution_policy() {
        "sequential" => RadauExecutionPolicy::Sequential,
        "parallel" => RadauExecutionPolicy::Parallel { min_work: 1 },
        "auto" => RadauExecutionPolicy::Auto { min_work: 1 },
        _ => {
            return Err(InvalidConfigurationError::new_err(
                "unknown execution policy in SolverOptions",
            ));
        }
    };
    let telemetry = match options.telemetry() {
        "off" => RadauTelemetryMode::Off,
        "counters" => RadauTelemetryMode::Counters,
        "timings" => RadauTelemetryMode::Timings,
        "detailed" => RadauTelemetryMode::Detailed,
        _ => {
            return Err(InvalidConfigurationError::new_err(
                "unknown telemetry mode in SolverOptions",
            ));
        }
    };
    Ok((frontend, telemetry, policy, matrix_layout))
}

fn map_radau_error(error: RadauError) -> PyErr {
    let native_kind = crate::errors::native_variant(&error.kind());
    let message = error.to_string();
    annotate_native(
        match error.kind() {
            RadauErrorKind::Configuration => InvalidConfigurationError::new_err(message),
            RadauErrorKind::Unsupported => UnsupportedOptionError::new_err(message),
            RadauErrorKind::Aot | RadauErrorKind::Workspace => PreparationError::new_err(message),
            RadauErrorKind::StepBudget => ExhaustionError::new_err(message),
            RadauErrorKind::Callback
            | RadauErrorKind::NonFiniteCallback
            | RadauErrorKind::Shape
            | RadauErrorKind::Newton
            | RadauErrorKind::StepUnderflow
            | RadauErrorKind::StepControl
            | RadauErrorKind::LinearSolve
            | RadauErrorKind::Output => SolverError::new_err(message),
        },
        "solve",
        "radau5",
        "native",
        &native_kind,
    )
}

fn validate_expression_symbols(
    expressions: &[PyExpr],
    state_variables: &[String],
    time_variable: &str,
    parameters: &[String],
    label: &str,
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
                "{label} expression {index} references undeclared symbol '{unknown}'"
            )));
        }
    }
    Ok(())
}

/// Detached Radau result exposed at the Python boundary.
///
/// The default output policy stores only the terminal state. Configure
/// `output="dense"` on [`RadauSolver`] to enable interpolation through
/// `sample(time)`. The result also owns the native counter/timing snapshot and
/// AOT artifact identities for the completed segment.
#[pyclass(name = "RadauSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct RadauSolution {
    final_time: f64,
    final_state: Vec<f64>,
    attempts: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    telemetry_mode: String,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
    aot_artifact_keys: Vec<String>,
    native: CoreRadauSolution,
}

impl RadauSolution {
    fn from_native(native: CoreRadauSolution) -> Self {
        let report = native.telemetry();
        let telemetry_mode = format!("{:?}", report.mode).to_ascii_lowercase();
        let mut counters: HashMap<String, u64> = report
            .counters
            .iter()
            .map(|(key, value)| ((*key).to_owned(), *value))
            .collect();
        let mut timings_ms: HashMap<String, f64> = report
            .timings_ms
            .iter()
            .map(|(key, value)| ((*key).to_owned(), *value))
            .collect();
        if telemetry_mode == "timings" {
            counters.clear();
        }
        crate::telemetry::apply_collection_mode(&telemetry_mode, &mut counters, &mut timings_ms);
        Self {
            final_time: native.t,
            final_state: native.y.clone(),
            attempts: native.attempts,
            accepted_steps: native.accepted_steps,
            rejected_steps: native.rejected_steps,
            telemetry_mode,
            counters,
            timings_ms,
            aot_artifact_keys: report.aot_artifact_keys.clone(),
            native,
        }
    }
}

#[pymethods]
impl RadauSolution {
    #[getter]
    /// Terminal integration time of this detached solution.
    pub fn final_time(&self) -> f64 {
        self.final_time
    }

    #[getter]
    /// State vector at `final_time`.
    pub fn final_state(&self) -> Vec<f64> {
        self.final_state.clone()
    }

    /// Owned NumPy array containing the terminal state.
    pub fn final_state_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.final_state)
    }

    #[getter]
    /// Total attempted integration steps, including rejected attempts.
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    #[getter]
    /// Number of accepted integration steps.
    pub fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }

    #[getter]
    /// Number of rejected integration steps.
    pub fn rejected_steps(&self) -> usize {
        self.rejected_steps
    }

    #[getter]
    /// Telemetry collection level captured for this solution.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    /// Stable counter names; empty unless the selected mode collects counters.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    /// Stable measured timing names; empty unless the selected mode collects timings.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Artifact identities observed by the native AOT lifecycle.
    pub fn aot_artifact_keys(&self) -> Vec<String> {
        self.aot_artifact_keys.clone()
    }

    /// Return the common solver-independent result snapshot. Radau exposes a
    /// terminal state rather than a sampled grid, so `sample_count` is one;
    /// dense-output samples remain available through `sample(...)`.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "radau5".to_owned()),
            ("status".to_owned(), "finished".to_owned()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
            ("sample_count".to_owned(), "1".to_owned()),
            (
                "state_dimension".to_owned(),
                self.final_state.len().to_string(),
            ),
            ("final_time".to_owned(), self.final_time.to_string()),
        ])
    }

    /// Render a compact, deterministic report suitable for batch logs.
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
            "telemetry={}; counters: {}; timings_ms: {}",
            self.telemetry_mode, counters, timings
        )
    }

    /// Write only the compact report, without capturing process output.
    pub fn write_telemetry_report(&self, path: &str) -> PyResult<()> {
        write_report(path, &self.telemetry_report())
    }

    /// Evaluate dense output when the solver was configured with `output='dense'`.
    pub fn sample(&self, time: f64) -> PyResult<Vec<f64>> {
        catch_panic("postprocess", "radau5", "dense_output", || {
            self.native.sample(time).map_err(map_radau_error)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RadauSolution(final_time={}, state_dimension={}, accepted_steps={})",
            self.final_time,
            self.final_state.len(),
            self.accepted_steps
        )
    }
}

/// Prepared symbolic Radau model.
///
/// The solver implements the fixed fifth-order Radau method. Use
/// [`SolverOptions`] to select ExprLegacy/AtomNative, Lambdify/AOT,
/// Dense/Sparse/Banded layout, callback execution policy and telemetry.
/// Symbolic preparation is explicit through `prepare()` or lazy through
/// `solve()`. An optional explicit Jacobian is supplied as a flattened
/// row-major sequence of `n * n` expressions; when omitted, the native
/// finite-difference route is used.
#[pyclass(name = "RadauSolver", module = "rustedscithe", unsendable)]
pub struct RadauSolver {
    pending_problem: Option<RadauProblem>,
    config: RadauConfig,
    solver: Option<CoreRadauSolver>,
    parameter_count: usize,
    telemetry_mode: RadauTelemetryMode,
    frontend: String,
    execution: String,
    execution_policy: String,
    matrix_layout: String,
    aot_metadata: HashMap<String, String>,
    panic_state: SolverPanicState,
}

#[pymethods]
impl RadauSolver {
    /// Configure a Radau initial-value problem.
    ///
    /// `expressions` contains one RHS expression per state variable.
    /// `t0`/`t_end`, `max_step`, `rtol`, and `atol` define the integration
    /// problem; `first_step` can override automatic initial-step selection.
    /// `max_steps`, `max_newton_iterations`, and `max_retries` bound work.
    /// Set `output="dense"` to retain interpolation data; otherwise only the
    /// final state is returned. `jacobian`, if supplied, is row-major.
    #[new]
    #[pyo3(signature=(expressions, state_variables, time_variable, t0, t_end, max_step, rtol=1e-3, atol=1e-6, options=None, parameters=None, jacobian=None, output="final", first_step=None, max_steps=1_000_000, max_newton_iterations=6, max_retries=12))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        time_variable: String,
        t0: f64,
        t_end: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        options: Option<PyRef<'_, SolverOptions>>,
        parameters: Option<Vec<String>>,
        jacobian: Option<Vec<PyExpr>>,
        output: &str,
        first_step: Option<f64>,
        max_steps: usize,
        max_newton_iterations: usize,
        max_retries: usize,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables", false)?;
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one residual expression per state variable, got {} expressions for {} variables",
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
        validate_names(&parameters, "parameters", true)?;
        if parameters
            .iter()
            .any(|name| name == &time_variable || state_variables.iter().any(|state| state == name))
        {
            return Err(InvalidConfigurationError::new_err(
                "parameters must not collide with time_variable or state_variables",
            ));
        }
        validate_expression_symbols(
            &expressions,
            &state_variables,
            &time_variable,
            &parameters,
            "residual",
        )?;
        if let Some(jacobian) = &jacobian {
            if jacobian.len() != state_variables.len() * state_variables.len() {
                return Err(InvalidConfigurationError::new_err(
                    "jacobian must contain state_dimension squared expressions",
                ));
            }
            validate_expression_symbols(
                jacobian,
                &state_variables,
                &time_variable,
                &parameters,
                "jacobian",
            )?;
        }
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
        if let Some(first_step) = first_step {
            validate_scalar(first_step, "first_step")?;
            if first_step <= 0.0 || first_step > (t_end - t0).abs() {
                return Err(InvalidConfigurationError::new_err(
                    "first_step must be positive and no larger than the integration interval",
                ));
            }
        }
        if max_steps == 0 || max_newton_iterations == 0 || max_retries == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_steps, max_newton_iterations and max_retries must be greater than zero",
            ));
        }
        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let _ = crate::contract::native_logging_level(&options, "radau5")?;
        let (frontend, telemetry_mode, execution_policy, matrix_layout) = validate_route(&options)?;
        let output = match output.trim().to_ascii_lowercase().as_str() {
            "final" | "final_only" => RadauOutputPolicy::FinalOnly,
            "dense" => RadauOutputPolicy::Dense,
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "Radau output must be 'final' or 'dense'",
                ));
            }
        };
        let execution = if options.execution() == "aot" {
            RadauExecution::Aot
        } else {
            RadauExecution::Lambdify
        };
        let aot = if options.execution() == "aot" {
            Some(rustedscithe_core::numerical::Radau::api::RadauAotConfig {
                generated: generated_config(&options)?,
            })
        } else {
            None
        };
        let config = RadauConfig {
            t0,
            t_bound: t_end,
            rtol,
            atol,
            first_step,
            max_step,
            max_steps,
            max_newton_iterations,
            max_retries,
            execution,
            frontend,
            matrix_layout,
            jacobian_source: if jacobian.is_some() {
                RadauJacobianSource::Analytic
            } else {
                RadauJacobianSource::FiniteDifference
            },
            telemetry: telemetry_mode,
            execution_policy,
            output,
            aot,
            ..RadauConfig::default()
        };
        let parameter_count = parameters.len();
        let mut problem = RadauProblem::new(
            expressions
                .into_iter()
                .map(|expression| expression.inner)
                .collect(),
            state_variables,
            time_variable,
        )
        .with_parameters(parameters);
        if let Some(jacobian) = jacobian {
            problem = problem.with_jacobian(
                jacobian
                    .into_iter()
                    .map(|expression| expression.inner)
                    .collect(),
            );
        }
        Ok(Self {
            pending_problem: Some(problem),
            config,
            solver: None,
            parameter_count,
            telemetry_mode,
            frontend: options.frontend().to_owned(),
            execution: options.execution().to_owned(),
            execution_policy: options.execution_policy().to_owned(),
            matrix_layout: options.matrix_layout().to_owned(),
            aot_metadata: provenance_fields(&options),
            panic_state: SolverPanicState::default(),
        })
    }

    /// Prepare symbolic callbacks and the selected linear backend.
    ///
    /// Preparation is idempotent and does not advance the solution. Calling
    /// this separately makes preparation cost observable independently.
    pub fn prepare(&mut self) -> PyResult<()> {
        if self.solver.is_some() {
            return Ok(());
        }
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("prepare", "radau5", &route, || {
            let problem = self.pending_problem.as_ref().cloned().ok_or_else(|| {
                InvalidConfigurationError::new_err("Radau problem has already been consumed")
            })?;
            let prepared =
                CoreRadauSolver::prepare(problem, self.config.clone()).map_err(map_radau_error)?;
            self.solver = Some(prepared);
            self.pending_problem = None;
            Ok(())
        })
    }

    /// Return whether symbolic callbacks and the selected linear backend are prepared.
    pub fn is_prepared(&self) -> bool {
        self.solver.is_some()
    }

    /// Solve one initial state, optionally binding declared value-only parameters.
    ///
    /// The method prepares lazily if needed and returns the terminal result
    /// plus telemetry. For repeated parameter values, use
    /// `continue_with_parameters()` to retain prepared symbolic callbacks.
    #[pyo3(signature=(y0, parameters=None))]
    pub fn solve(&mut self, y0: Vec<f64>, parameters: Option<Vec<f64>>) -> PyResult<RadauSolution> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "radau5", &route, || {
            self.solve_inner(y0, parameters)
        })
    }

    /// Reuse prepared callbacks for another parameter segment.
    ///
    /// `y0` is the initial state for this segment and `parameters` must match
    /// the declared parameter count. The returned solution is detached.
    pub fn continue_with_parameters(
        &mut self,
        y0: Vec<f64>,
        parameters: Vec<f64>,
    ) -> PyResult<RadauSolution> {
        finite_vector(&y0, "y0")?;
        finite_vector(&parameters, "parameters")?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "radau5", &route, || {
            self.prepare()?;
            if parameters.len() != self.parameter_count {
                return Err(InvalidConfigurationError::new_err(format!(
                    "expected {} parameter values, got {}",
                    self.parameter_count,
                    parameters.len()
                )));
            }
            self.solver
                .as_mut()
                .ok_or_else(|| InvalidConfigurationError::new_err("Radau solver is not prepared"))?
                .continue_with_parameters(&y0, &parameters)
                .map(RadauSolution::from_native)
                .map_err(map_radau_error)
        })
    }

    /// Change the interval while retaining prepared symbolic callbacks.
    ///
    /// This changes the configured integration bounds, not the initial state;
    /// pass the desired state to `solve()` or continuation.
    pub fn restart(&mut self, t0: f64, t_end: f64) -> PyResult<()> {
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "radau5", &route, || {
            self.prepare()?;
            self.solver
                .as_mut()
                .ok_or_else(|| InvalidConfigurationError::new_err("Radau solver is not prepared"))?
                .restart(t0, t_end)
                .map_err(map_radau_error)
        })
    }

    /// Stable route metadata needed to reproduce this Python-side solve.
    pub fn provenance(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::from([
            ("solver".to_owned(), "radau5".to_owned()),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), self.matrix_layout.clone()),
            ("execution_policy".to_owned(), self.execution_policy.clone()),
            ("telemetry".to_owned(), self.telemetry_mode().to_owned()),
        ]);
        metadata.extend(self.aot_metadata.clone());
        metadata.insert(
            "first_step".to_owned(),
            self.config
                .first_step
                .map_or_else(|| "auto".to_owned(), |value| value.to_string()),
        );
        metadata.insert("max_steps".to_owned(), self.config.max_steps.to_string());
        metadata.insert(
            "max_newton_iterations".to_owned(),
            self.config.max_newton_iterations.to_string(),
        );
        metadata.insert(
            "max_retries".to_owned(),
            self.config.max_retries.to_string(),
        );
        metadata
    }

    #[getter]
    /// Telemetry collection level configured on this solver.
    pub fn telemetry_mode(&self) -> &'static str {
        match self.telemetry_mode {
            RadauTelemetryMode::Off => "off",
            RadauTelemetryMode::Counters => "counters",
            RadauTelemetryMode::Timings => "timings",
            RadauTelemetryMode::Detailed => "detailed",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RadauSolver(prepared={}, telemetry='{}')",
            self.is_prepared(),
            self.telemetry_mode()
        )
    }
}

impl RadauSolver {
    fn solve_inner(
        &mut self,
        y0: Vec<f64>,
        parameters: Option<Vec<f64>>,
    ) -> PyResult<RadauSolution> {
        finite_vector(&y0, "y0")?;
        if let Some(parameters) = parameters.as_deref() {
            finite_vector(parameters, "parameters")?;
        }
        self.prepare()?;
        let parameters = parameters.unwrap_or_default();
        if parameters.len() != self.parameter_count {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected {} parameter values, got {}",
                self.parameter_count,
                parameters.len()
            )));
        }
        self.solver
            .as_mut()
            .ok_or_else(|| InvalidConfigurationError::new_err("Radau solver is not prepared"))?
            .solve_with_parameters(&y0, &parameters)
            .map(RadauSolution::from_native)
            .map_err(map_radau_error)
    }
}

/// Dense Radau route backed by Python-owned residual and Jacobian callbacks.
///
/// This is intentionally separate from [`RadauSolver`]: symbolic expressions
/// have a preparation/lifecycle contract, while callbacks are already
/// prepared by the caller. The Jacobian is optional; when omitted, the native
/// Radau core performs its component-wise finite-difference fallback. This
/// route is Dense, sequential and callback-only; it does not perform symbolic
/// preparation or AOT compilation.
#[pyclass(name = "RadauCallbackSolver", module = "rustedscithe", unsendable)]
pub struct RadauCallbackSolver {
    solver: RadauNativeSolver,
    failure: CallbackFailureState,
    dimension: usize,
    telemetry_mode: String,
    jacobian_source: String,
    config: RadauConfig,
    panic_state: SolverPanicState,
}

#[pymethods]
impl RadauCallbackSolver {
    /// Configure a Dense Radau solve using Python residual and Jacobian
    /// callbacks.
    ///
    /// `residual_callback(t, y)` returns a finite vector of length
    /// `state_dimension`; optional `jacobian_callback(t, y)` returns a finite
    /// square row-major matrix. If omitted, Radau uses its native finite-
    /// difference Jacobian. Callback execution is synchronous and sequential.
    /// Choose `output="dense"` to allow `RadauSolution.sample()`.
    #[new]
    #[pyo3(signature=(state_dimension, t0, t_end, max_step, residual_callback, jacobian_callback=None, rtol=1e-3, atol=1e-6, options=None, output="final", first_step=None, max_steps=1_000_000, max_newton_iterations=6, max_retries=12))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state_dimension: usize,
        t0: f64,
        t_end: f64,
        max_step: f64,
        residual_callback: Py<PyAny>,
        jacobian_callback: Option<Py<PyAny>>,
        rtol: f64,
        atol: f64,
        options: Option<PyRef<'_, SolverOptions>>,
        output: &str,
        first_step: Option<f64>,
        max_steps: usize,
        max_newton_iterations: usize,
        max_retries: usize,
    ) -> PyResult<Self> {
        if state_dimension == 0 {
            return Err(InvalidConfigurationError::new_err(
                "state_dimension must be greater than zero",
            ));
        }
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
        if let Some(first_step) = first_step {
            validate_scalar(first_step, "first_step")?;
            if first_step <= 0.0 || first_step > (t_end - t0).abs() {
                return Err(InvalidConfigurationError::new_err(
                    "first_step must be positive and no larger than the integration interval",
                ));
            }
        }
        if max_steps == 0 || max_newton_iterations == 0 || max_retries == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_steps, max_newton_iterations and max_retries must be greater than zero",
            ));
        }
        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let _ = crate::contract::native_logging_level(&options, "radau5")?;
        let (_, telemetry, _, matrix_layout) = validate_route(&options)?;
        if options.execution() != "lambdify" {
            return Err(UnsupportedOptionError::new_err(
                "RadauCallbackSolver requires execution='lambdify'; Python callbacks cannot be embedded into AOT artifacts",
            ));
        }
        if options.frontend() != "expr_legacy" {
            return Err(UnsupportedOptionError::new_err(
                "RadauCallbackSolver does not use a symbolic frontend; use frontend='expr_legacy' or omit frontend",
            ));
        }
        if matrix_layout != RadauMatrixLayout::Dense {
            return Err(UnsupportedOptionError::new_err(
                "RadauCallbackSolver currently supports only matrix_layout='dense'",
            ));
        }
        if options.execution_policy() != "sequential" {
            return Err(UnsupportedOptionError::new_err(
                "RadauCallbackSolver currently supports only execution_policy='sequential'",
            ));
        }
        let output = match output.trim().to_ascii_lowercase().as_str() {
            "final" | "final_only" => RadauOutputPolicy::FinalOnly,
            "dense" => RadauOutputPolicy::Dense,
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "Radau callback output must be 'final' or 'dense'",
                ));
            }
        };
        let jacobian_source = if jacobian_callback.is_some() {
            RadauJacobianSource::Analytic
        } else {
            RadauJacobianSource::FiniteDifference
        };
        let config = RadauConfig {
            t0,
            t_bound: t_end,
            rtol,
            atol,
            first_step,
            max_step,
            max_steps,
            max_newton_iterations,
            max_retries,
            execution: RadauExecution::NativeCallbacks,
            frontend: RadauFrontend::ExprLegacy,
            matrix_layout: RadauMatrixLayout::Dense,
            jacobian_source,
            telemetry,
            execution_policy: RadauExecutionPolicy::Sequential,
            output,
            aot: None,
        };
        let (residual, jacobian, failure) =
            build_native_callbacks(residual_callback, jacobian_callback, state_dimension)?;
        // `RadauNativeSolver::prepare` accepts concrete closures. Keep the
        // erased callback storage private to this adapter and add one tiny
        // forwarding closure at the Rust/Python boundary.
        let solver = catch_panic("construct", "radau5", "python_callback:dense", || {
            let solver = if let Some(jacobian) = jacobian {
                let residual_callback =
                    move |time: f64, state: &DVector<f64>| residual(time, state);
                let jacobian_callback =
                    move |time: f64, state: &DVector<f64>| jacobian(time, state);
                RadauNativeSolver::prepare(
                    config.clone(),
                    residual_callback,
                    Some(jacobian_callback),
                )
            } else {
                let residual_callback =
                    move |time: f64, state: &DVector<f64>| residual(time, state);
                let no_jacobian: Option<fn(f64, &DVector<f64>) -> DMatrix<f64>> = None;
                RadauNativeSolver::prepare(config.clone(), residual_callback, no_jacobian)
            }
            .map_err(map_radau_error)?;
            Ok(solver)
        })?;
        Ok(Self {
            solver,
            failure,
            dimension: state_dimension,
            telemetry_mode: options.telemetry().to_owned(),
            jacobian_source: if matches!(jacobian_source, RadauJacobianSource::Analytic) {
                "analytic"
            } else {
                "finite_difference"
            }
            .to_owned(),
            config,
            panic_state: SolverPanicState::default(),
        })
    }

    #[getter]
    /// This callback adapter is immediately usable and has no separate prepare phase.
    pub fn is_prepared(&self) -> bool {
        true
    }

    /// Integrate from `y0` and return the terminal result and telemetry.
    pub fn solve(&mut self, y0: Vec<f64>) -> PyResult<RadauSolution> {
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "radau5", "python_callback", || {
            self.solve_checked(y0)
        })
    }

    /// Continue the existing interval from a new initial state.
    ///
    /// The callback bindings and configured tolerances are retained. Use
    /// `restart()` first if the time interval also needs to change.
    pub fn continue_with_initial_state(&mut self, y0: Vec<f64>) -> PyResult<RadauSolution> {
        if y0.len() != self.dimension {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected y0 with {} values, got {}",
                self.dimension,
                y0.len()
            )));
        }
        finite_vector(&y0, "y0")?;
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "radau5", "python_callback", || {
            let result = self.solver.continue_with_initial_state(&y0);
            if let Some(error) =
                self.failure
                    .take_pyerr("continuation", "radau5", "python_callback")
            {
                return Err(error);
            }
            result
                .map(RadauSolution::from_native)
                .map_err(map_radau_error)
        })
    }

    /// Replace the integration interval while retaining callbacks and policy.
    pub fn restart(&mut self, t0: f64, t_end: f64) -> PyResult<()> {
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "radau5", "python_callback", || {
            self.solver.restart(t0, t_end).map_err(map_radau_error)?;
            self.config.t0 = t0;
            self.config.t_bound = t_end;
            Ok(())
        })
    }

    /// Describe the fixed Dense callback route and Jacobian source.
    pub fn provenance(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "radau5".to_owned()),
            ("frontend".to_owned(), "python_callback".to_owned()),
            ("execution".to_owned(), "native_callback".to_owned()),
            ("matrix_layout".to_owned(), "dense".to_owned()),
            ("execution_policy".to_owned(), "sequential".to_owned()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
            ("jacobian_source".to_owned(), self.jacobian_source.clone()),
            (
                "first_step".to_owned(),
                self.config
                    .first_step
                    .map_or_else(|| "auto".to_owned(), |value| value.to_string()),
            ),
            ("max_steps".to_owned(), self.config.max_steps.to_string()),
            (
                "max_newton_iterations".to_owned(),
                self.config.max_newton_iterations.to_string(),
            ),
            (
                "max_retries".to_owned(),
                self.config.max_retries.to_string(),
            ),
        ])
    }

    fn __repr__(&self) -> String {
        format!(
            "RadauCallbackSolver(prepared=true, state_dimension={}, jacobian='{}')",
            self.dimension, self.jacobian_source
        )
    }
}

impl RadauCallbackSolver {
    fn solve_checked(&mut self, y0: Vec<f64>) -> PyResult<RadauSolution> {
        if y0.len() != self.dimension {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected y0 with {} values, got {}",
                self.dimension,
                y0.len()
            )));
        }
        finite_vector(&y0, "y0")?;
        let result = self.solver.solve(&y0);
        if let Some(error) = self
            .failure
            .take_pyerr("solve", "radau5", "python_callback")
        {
            return Err(error);
        }
        result
            .map(RadauSolution::from_native)
            .map_err(map_radau_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radau_route_accepts_sparse_and_validates_banded_widths() {
        let options = SolverOptions::new(
            Some("atom_native"),
            Some("lambdify"),
            Some("sparse"),
            Some("sequential"),
            Some("off"),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(validate_route(&options).is_ok());

        let mut banded = SolverOptions::new(
            Some("atom_native"),
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
        assert!(validate_route(&banded).is_err());
        banded.set_banded_width(1, 1);
        assert!(validate_route(&banded).is_ok());
    }
}
