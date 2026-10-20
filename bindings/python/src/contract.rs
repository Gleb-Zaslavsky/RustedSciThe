use std::collections::HashMap;
use std::fmt;

use pyo3::prelude::*;

use crate::errors::{InvalidConfigurationError, UnsupportedOptionError};
use rustedscithe_core::numerical::logging::SolverLogLevel;

/// Symbolic preparation route understood by the native solver APIs.
#[pyclass(name = "SymbolicFrontend", module = "rustedscithe")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolicFrontend {
    ExprLegacy,
    AtomNative,
}

impl fmt::Display for SymbolicFrontend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomNative => "AtomNative",
        })
    }
}

#[pymethods]
impl SymbolicFrontend {
    fn __str__(&self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomNative => "AtomNative",
        }
    }

    fn __repr__(&self) -> String {
        format!("SymbolicFrontend.{}", self)
    }
}

/// Runtime route used after symbolic preparation.
#[pyclass(name = "ExecutionRoute", module = "rustedscithe")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionRoute {
    Lambdify,
    AOT,
    Numerical,
}

impl fmt::Display for ExecutionRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Lambdify => "Lambdify",
            Self::AOT => "AOT",
            Self::Numerical => "Numerical",
        })
    }
}

#[pymethods]
impl ExecutionRoute {
    fn __str__(&self) -> &'static str {
        match self {
            Self::Lambdify => "Lambdify",
            Self::AOT => "AOT",
            Self::Numerical => "Numerical",
        }
    }

    fn __repr__(&self) -> String {
        format!("ExecutionRoute.{}", self)
    }
}

/// Linear algebra storage selected by a solver.
#[pyclass(name = "MatrixLayout", module = "rustedscithe")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixLayout {
    Dense,
    Sparse,
    Banded,
}

impl fmt::Display for MatrixLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        })
    }
}

#[pymethods]
impl MatrixLayout {
    fn __str__(&self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
    }

    fn __repr__(&self) -> String {
        format!("MatrixLayout.{}", self)
    }
}

/// Callback evaluation policy for a solver.
#[pyclass(name = "ExecutionPolicy", module = "rustedscithe")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionPolicy {
    Sequential,
    Parallel,
    Auto,
}

impl fmt::Display for ExecutionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sequential => "Sequential",
            Self::Parallel => "Parallel",
            Self::Auto => "Auto",
        })
    }
}

#[pymethods]
impl ExecutionPolicy {
    fn __str__(&self) -> &'static str {
        match self {
            Self::Sequential => "Sequential",
            Self::Parallel => "Parallel",
            Self::Auto => "Auto",
        }
    }

    fn __repr__(&self) -> String {
        format!("ExecutionPolicy.{}", self)
    }
}

/// Amount of runtime telemetry collected by a solver.
#[pyclass(name = "TelemetryMode", module = "rustedscithe")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryMode {
    Off,
    Counters,
    Timings,
    Detailed,
}

impl fmt::Display for TelemetryMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "Off",
            Self::Counters => "Counters",
            Self::Timings => "Timings",
            Self::Detailed => "Detailed",
        })
    }
}

#[pymethods]
impl TelemetryMode {
    fn __str__(&self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Counters => "Counters",
            Self::Timings => "Timings",
            Self::Detailed => "Detailed",
        }
    }

    fn __repr__(&self) -> String {
        format!("TelemetryMode.{}", self)
    }
}

fn normalize(value: Option<&str>, default: &str, field: &str) -> PyResult<String> {
    let value = value.unwrap_or(default).trim();
    if value.is_empty() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{field} must not be empty"
        )));
    }
    Ok(value.to_ascii_lowercase().replace(['-', ' '], "_"))
}

fn choose(value: Option<&str>, default: &str, field: &str, allowed: &[&str]) -> PyResult<String> {
    let normalized = normalize(value, default, field)?;
    if allowed.iter().any(|item| *item == normalized) {
        Ok(normalized)
    } else {
        Err(UnsupportedOptionError::new_err(format!(
            "unsupported {field} '{normalized}'; expected one of {}",
            allowed.join(", ")
        )))
    }
}

fn choose_frontend(value: Option<&str>) -> PyResult<String> {
    match choose(
        value,
        "expr_legacy",
        "frontend",
        &[
            "expr_legacy",
            "expr",
            "atom_native",
            "atom_view",
            "numerical",
        ],
    )?
    .as_str()
    {
        "expr_legacy" | "expr" => Ok("expr_legacy".to_owned()),
        "atom_native" | "atom_view" => Ok("atom_native".to_owned()),
        "numerical" => Ok("numerical".to_owned()),
        _ => unreachable!("choose_frontend only accepts the four listed aliases"),
    }
}

/// Common solver configuration values shared by the native APIs.
///
/// This object only validates and stores the common vocabulary. A concrete
/// solver wrapper remains responsible for applying solver-specific options.
#[pyclass(name = "SolverOptions", module = "rustedscithe")]
#[derive(Clone, Debug)]
pub struct SolverOptions {
    frontend: String,
    execution: String,
    matrix_layout: String,
    execution_policy: String,
    telemetry: String,
    logging_level: String,
    aot_policy: String,
    aot_toolchain: String,
    aot_profile: String,
    aot_output_dir: Option<String>,
    aot_handoff_path: Option<String>,
    band_lower: Option<usize>,
    band_upper: Option<usize>,
    sparse_factorization: String,
}

#[pymethods]
impl SolverOptions {
    /// Create shared options for a concrete solver wrapper.
    ///
    /// Values are normalized case-insensitively. Defaults are
    /// `frontend="expr_legacy"`, `execution="lambdify"`,
    /// `matrix_layout="dense"`, `execution_policy="sequential"`, and
    /// `telemetry="timings"`. These are independent choices; a solver rejects
    /// combinations it does not support instead of silently ignoring them.
    /// AOT-specific arguments are meaningful only with `execution="aot"`.
    ///
    /// Common spellings include `expr_legacy`/`expr` and
    /// `atom_native`/`atom_view`; `lambdify`/`aot` (plus solver-specific
    /// `numerical` callbacks); `dense`/`sparse`/`banded`;
    /// `sequential`/`parallel`/`auto`; and `off`/`counters`/`timings`/`detailed`.
    #[new]
    #[pyo3(signature = (*, frontend=None, execution=None, matrix_layout=None, execution_policy=None, telemetry=None, aot_policy=None, aot_toolchain=None, aot_profile=None, aot_output_dir=None))]
    pub fn new(
        frontend: Option<&str>,
        execution: Option<&str>,
        matrix_layout: Option<&str>,
        execution_policy: Option<&str>,
        telemetry: Option<&str>,
        aot_policy: Option<&str>,
        aot_toolchain: Option<&str>,
        aot_profile: Option<&str>,
        aot_output_dir: Option<&str>,
    ) -> PyResult<Self> {
        let aot_output_dir = match aot_output_dir {
            Some(value) if value.trim().is_empty() => {
                return Err(InvalidConfigurationError::new_err(
                    "aot_output_dir must not be empty",
                ));
            }
            Some(value) => Some(value.trim().to_owned()),
            None => None,
        };
        Ok(Self {
            frontend: choose_frontend(frontend)?,
            execution: choose(
                execution,
                "lambdify",
                "execution",
                &["lambdify", "aot", "numerical"],
            )?,
            matrix_layout: choose(
                matrix_layout,
                "dense",
                "matrix_layout",
                &["dense", "sparse", "banded"],
            )?,
            execution_policy: choose(
                execution_policy,
                "sequential",
                "execution_policy",
                &["sequential", "parallel", "auto"],
            )?,
            telemetry: choose(
                telemetry,
                "timings",
                "telemetry",
                &["off", "counters", "timings", "detailed"],
            )?,
            logging_level: "off".to_owned(),
            aot_policy: choose(
                aot_policy,
                "build_if_missing",
                "aot_policy",
                &["require_prebuilt", "build_if_missing", "rebuild_always"],
            )?,
            aot_toolchain: choose(
                aot_toolchain,
                "gcc",
                "aot_toolchain",
                &["gcc", "tcc", "zig", "rust"],
            )?,
            aot_profile: choose(aot_profile, "release", "aot_profile", &["debug", "release"])?,
            aot_output_dir,
            aot_handoff_path: None,
            band_lower: None,
            band_upper: None,
            sparse_factorization: "standard_faer".to_owned(),
        })
    }

    /// Preparation frontend: symbolic Expr/Atom or direct numerical callbacks.
    #[getter]
    pub(crate) fn frontend(&self) -> &str {
        &self.frontend
    }

    /// Execution route: Lambdify, AOT, or a solver's direct numerical callbacks.
    #[getter]
    pub(crate) fn execution(&self) -> &str {
        &self.execution
    }

    /// Linear-system layout requested from the selected solver.
    #[getter]
    pub(crate) fn matrix_layout(&self) -> &str {
        &self.matrix_layout
    }

    /// Callback execution policy: sequential, parallel, or automatic.
    #[getter]
    pub(crate) fn execution_policy(&self) -> &str {
        &self.execution_policy
    }

    /// Telemetry level: off, counters, timings, or detailed.
    #[getter]
    pub(crate) fn telemetry(&self) -> &str {
        &self.telemetry
    }

    /// Rust `log` verbosity requested for solvers that expose native logging.
    #[getter]
    pub(crate) fn logging_level(&self) -> &str {
        &self.logging_level
    }

    /// Select native Rust logging verbosity; logging remains disabled by default.
    pub fn set_logging_level(&mut self, level: &str) -> PyResult<()> {
        self.logging_level = choose(
            Some(level),
            "off",
            "logging_level",
            &["off", "error", "warn", "warning", "info", "debug"],
        )?;
        if self.logging_level == "warning" {
            self.logging_level = "warn".to_owned();
        }
        Ok(())
    }

    #[getter]
    /// AOT artifact lifecycle policy selected for generated execution.
    pub(crate) fn aot_policy(&self) -> &str {
        &self.aot_policy
    }

    #[getter]
    /// Compiler/toolchain name requested for AOT generation.
    pub(crate) fn aot_toolchain(&self) -> &str {
        &self.aot_toolchain
    }

    #[getter]
    /// AOT build optimization profile (`debug` or `release`).
    pub(crate) fn aot_profile(&self) -> &str {
        &self.aot_profile
    }

    #[getter]
    /// Optional directory used for generated AOT artifacts.
    pub(crate) fn aot_output_dir(&self) -> Option<&str> {
        self.aot_output_dir.as_deref()
    }

    /// Configure the durable registry file shared by an AOT producer and
    /// consumer. It is opt-in so existing constructor call sites remain
    /// source-compatible. This process-isolated handoff setting is rejected
    /// for a Lambdify route.
    pub fn set_aot_handoff_path(&mut self, path: Option<&str>) -> PyResult<()> {
        self.aot_handoff_path = match path {
            Some(value) if value.trim().is_empty() => {
                return Err(InvalidConfigurationError::new_err(
                    "aot_handoff_path must not be empty",
                ));
            }
            Some(value) => Some(value.trim().to_owned()),
            None => None,
        };
        Ok(())
    }

    #[getter]
    /// Optional registry path for process-isolated AOT artifact handoff.
    pub(crate) fn aot_handoff_path(&self) -> Option<&str> {
        self.aot_handoff_path.as_deref()
    }

    /// Set the lower and upper half-bandwidth used by a Banded route.
    ///
    /// The values are intentionally separate from `matrix_layout`: callers
    /// may build one option object first and select the concrete layout later.
    pub fn set_banded_width(&mut self, lower: usize, upper: usize) {
        self.band_lower = Some(lower);
        self.band_upper = Some(upper);
    }

    /// Select a solver-specific sparse factorization policy.
    ///
    /// The common option object stores the spelling without forcing every
    /// solver to support every sparse implementation. BVP_sci currently
    /// accepts `standard_faer`, the reusable-faer experiments and `superlu`.
    pub fn set_sparse_factorization(&mut self, value: &str) -> PyResult<()> {
        self.sparse_factorization = choose(
            Some(value),
            "standard_faer",
            "sparse_factorization",
            &[
                "standard_faer",
                "reusable_faer_experimental",
                "reusable_faer_simplicial_experimental",
                "superlu",
            ],
        )?;
        Ok(())
    }

    /// Return the selected sparse factorization policy label.
    pub(crate) fn sparse_factorization(&self) -> &str {
        &self.sparse_factorization
    }

    /// Return the lower half-bandwidth, or `None` if no band width was configured.
    pub(crate) fn band_lower(&self) -> Option<usize> {
        self.band_lower
    }

    /// Return the upper half-bandwidth, or `None` if no band width was configured.
    pub(crate) fn band_upper(&self) -> Option<usize> {
        self.band_upper
    }

    /// Validate common options against a concrete solver capability set.
    ///
    /// Use a registered name such as `bdf`, `radau`, or `bvp_sci`. This checks
    /// common route dimensions, including known frontend/execution/layout
    /// constraints. Solver-specific numeric settings are checked by that
    /// solver's constructor.
    pub fn validate_for(&self, solver: &str) -> PyResult<()> {
        self.validate_aot_options()?;
        let capabilities = capabilities_for(solver)?;
        let _ = native_logging_level(self, &capabilities.solver)?;
        if !capabilities.frontends.iter().any(|v| v == &self.frontend)
            || !capabilities.executions.iter().any(|v| v == &self.execution)
            || !capabilities
                .layouts
                .iter()
                .any(|v| v == &self.matrix_layout)
            || !capabilities
                .policies
                .iter()
                .any(|v| v == &self.execution_policy)
        {
            return Err(UnsupportedOptionError::new_err(format!(
                "options are not supported by solver '{solver}'"
            )));
        }
        if !capabilities
            .telemetry_modes
            .iter()
            .any(|v| v == &self.telemetry)
        {
            return Err(UnsupportedOptionError::new_err(format!(
                "telemetry mode '{}' is not supported by solver '{solver}'",
                self.telemetry
            )));
        }
        validate_backend_options_for(self, &capabilities.solver)?;
        if capabilities.solver == "bvp_sci"
            && ((self.frontend == "numerical") != (self.execution == "numerical")
                || (self.frontend == "numerical" && self.matrix_layout != "dense"))
        {
            return Err(UnsupportedOptionError::new_err(
                "BVP_sci numerical callbacks require frontend='numerical', execution='numerical' and matrix_layout='dense'",
            ));
        }
        Ok(())
    }

    /// Reject AOT-only knobs when the selected execution route is Lambdify.
    ///
    /// Keeping this check in the common contract prevents a setting from
    /// being silently accepted by one wrapper and ignored by another.
    pub(crate) fn validate_aot_options(&self) -> PyResult<()> {
        if self.execution != "aot"
            && (self.aot_policy != "build_if_missing"
                || self.aot_toolchain != "gcc"
                || self.aot_profile != "release"
                || self.aot_output_dir.is_some()
                || self.aot_handoff_path.is_some())
        {
            return Err(UnsupportedOptionError::new_err(
                "aot_policy, aot_toolchain, aot_profile, aot_output_dir and aot_handoff_path require execution='aot'",
            ));
        }
        Ok(())
    }

    /// Return explicitly configured options as strings, omitting unset values.
    ///
    /// This is a normalized inspection snapshot, not a serialization format or
    /// an alternate way to configure a solver.
    fn as_dict(&self) -> HashMap<String, String> {
        let mut values = HashMap::from([
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), self.matrix_layout.clone()),
            ("execution_policy".to_owned(), self.execution_policy.clone()),
            ("telemetry".to_owned(), self.telemetry.clone()),
            ("logging_level".to_owned(), self.logging_level.clone()),
            ("aot_policy".to_owned(), self.aot_policy.clone()),
            ("aot_toolchain".to_owned(), self.aot_toolchain.clone()),
            ("aot_profile".to_owned(), self.aot_profile.clone()),
            (
                "sparse_factorization".to_owned(),
                self.sparse_factorization.clone(),
            ),
        ]);
        if let Some(output_dir) = &self.aot_output_dir {
            values.insert("aot_output_dir".to_owned(), output_dir.clone());
        }
        if let Some(handoff_path) = &self.aot_handoff_path {
            values.insert("aot_handoff_path".to_owned(), handoff_path.clone());
        }
        if let Some(lower) = self.band_lower {
            values.insert("band_lower".to_owned(), lower.to_string());
        }
        if let Some(upper) = self.band_upper {
            values.insert("band_upper".to_owned(), upper.to_string());
        }
        values
    }

    fn __repr__(&self) -> String {
        format!(
            "SolverOptions(frontend='{}', execution='{}', matrix_layout='{}', execution_policy='{}', telemetry='{}', logging_level='{}', aot_policy='{}', aot_toolchain='{}', aot_profile='{}', aot_output_dir={:?}, aot_handoff_path={:?}, band_lower={:?}, band_upper={:?})",
            self.frontend,
            self.execution,
            self.matrix_layout,
            self.execution_policy,
            self.telemetry,
            self.logging_level,
            self.aot_policy,
            self.aot_toolchain,
            self.aot_profile,
            self.aot_output_dir,
            self.aot_handoff_path,
            self.band_lower,
            self.band_upper,
        )
    }
}

/// Reject layout/backend knobs that a selected native route would ignore.
///
/// This common check runs both through `SolverOptions.validate_for()` and in
/// wrappers with additional route-specific validation.
pub(crate) fn validate_backend_options_for(options: &SolverOptions, solver: &str) -> PyResult<()> {
    let has_lower = options.band_lower.is_some();
    let has_upper = options.band_upper.is_some();
    if has_lower != has_upper {
        return Err(InvalidConfigurationError::new_err(
            "band_lower and band_upper must be configured together",
        ));
    }
    if (has_lower || has_upper) && options.matrix_layout != "banded" {
        return Err(UnsupportedOptionError::new_err(
            "band_lower/band_upper are only valid for matrix_layout='banded'",
        ));
    }
    if options.sparse_factorization != "standard_faer"
        && (solver != "bvp_sci" || options.matrix_layout != "sparse")
    {
        return Err(UnsupportedOptionError::new_err(
            "non-default sparse_factorization is supported only by the BVP_sci Sparse route (matrix_layout='sparse')",
        ));
    }
    Ok(())
}

/// Translate common Python logging configuration to a supported native level.
pub(crate) fn native_logging_level(
    options: &SolverOptions,
    solver: &str,
) -> PyResult<SolverLogLevel> {
    if options.logging_level != "off" && !supports_common_logging_level(solver) {
        return Err(UnsupportedOptionError::new_err(format!(
            "solver '{solver}' does not expose native Rust logging"
        )));
    }
    Ok(match options.logging_level.as_str() {
        "off" => SolverLogLevel::Off,
        "error" => SolverLogLevel::Error,
        "warn" | "warning" => SolverLogLevel::Warn,
        "info" => SolverLogLevel::Info,
        "debug" => SolverLogLevel::Debug,
        _ => unreachable!("logging level validated by SolverOptions"),
    })
}

fn supports_common_logging_level(solver: &str) -> bool {
    matches!(
        solver,
        "bdf" | "backward_euler" | "lsode2" | "bvp_sci" | "shooting_bvp"
    )
}

/// Capabilities advertised by a concrete solver family.
///
/// Axis lists describe values available somewhere in that route family; they
/// are not implicitly a Cartesian product. For `universal_fitting`, the axes
/// are the union across methods. Query `universal_fitting_lm`,
/// `universal_fitting_gavin`, or `universal_fitting_varpro` for method-specific
/// capabilities. `SolverOptions.validate_for()` checks shared cross-axis
/// constraints, while concrete wrappers validate additional combinations.
#[pyclass(name = "SolverCapabilities", module = "rustedscithe")]
#[derive(Clone, Debug)]
pub struct SolverCapabilities {
    solver: String,
    frontends: Vec<String>,
    executions: Vec<String>,
    methods: Vec<String>,
    layouts: Vec<String>,
    policies: Vec<String>,
    telemetry_modes: Vec<String>,
    supports_continuation: bool,
    supports_analytical_callbacks: bool,
    analytical_callback_layouts: Vec<String>,
}

#[pymethods]
impl SolverCapabilities {
    #[getter]
    /// Canonical name of the solver family described by this capability record.
    pub(crate) fn solver(&self) -> &str {
        &self.solver
    }

    #[getter]
    /// Symbolic frontends supported by the solver.
    pub(crate) fn frontends(&self) -> Vec<String> {
        self.frontends.clone()
    }

    #[getter]
    /// Runtime execution routes supported by the solver.
    pub(crate) fn executions(&self) -> Vec<String> {
        self.executions.clone()
    }

    /// Solver-specific method/controller values accepted by the Python API.
    /// An empty list means that the solver has no selectable method in this
    /// contract and uses its single native algorithm.
    #[getter]
    pub(crate) fn methods(&self) -> Vec<String> {
        self.methods.clone()
    }

    #[getter]
    /// Matrix layouts supported by the solver.
    pub(crate) fn layouts(&self) -> Vec<String> {
        self.layouts.clone()
    }

    #[getter]
    /// Callback execution policies supported by the solver.
    pub(crate) fn policies(&self) -> Vec<String> {
        self.policies.clone()
    }

    #[getter]
    /// Telemetry modes actually collected by the solver route.
    pub(crate) fn telemetry_modes(&self) -> Vec<String> {
        self.telemetry_modes.clone()
    }

    #[getter]
    /// Whether `SolverOptions.logging_level` maps to the native Rust `log` API.
    /// Solver-specific logging controls are reported separately by their facade.
    pub(crate) fn supports_logging(&self) -> bool {
        supports_common_logging_level(&self.solver)
    }

    #[getter]
    /// Whether the solver supports parameter continuation.
    pub(crate) fn supports_continuation(&self) -> bool {
        self.supports_continuation
    }

    #[getter]
    /// Whether analytical residual/Jacobian callback input is supported.
    pub(crate) fn supports_analytical_callbacks(&self) -> bool {
        self.supports_analytical_callbacks
    }

    /// Matrix layouts for which Python-owned analytical callbacks are valid.
    ///
    /// This is intentionally separate from the boolean capability because
    /// different matrix layouts may expose different callback contracts.
    #[getter]
    pub(crate) fn analytical_callback_layouts(&self) -> Vec<String> {
        self.analytical_callback_layouts.clone()
    }

    /// Selectable sparse factorization policies when the wrapper exposes them.
    ///
    /// These are not matrix layouts. BVP_sci is currently the only Python
    /// route with a selectable sparse implementation; experimental policies
    /// are named as such to avoid presenting them as production defaults.
    #[getter]
    pub(crate) fn sparse_factorization_policies(&self) -> Vec<String> {
        if self.solver == "bvp_sci" {
            vec![
                "standard_faer".to_owned(),
                "reusable_faer_experimental".to_owned(),
                "reusable_faer_simplicial_experimental".to_owned(),
                "superlu".to_owned(),
            ]
        } else {
            Vec::new()
        }
    }

    fn __repr__(&self) -> String {
        format!("SolverCapabilities(solver='{}')", self.solver)
    }
}

fn capabilities_for(solver: &str) -> PyResult<SolverCapabilities> {
    let solver = solver.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let common_frontends = vec!["expr_legacy".to_owned(), "atom_native".to_owned()];
    match solver.as_str() {
        "lapack_banded" | "somelinalg_banded" => Ok(SolverCapabilities {
            solver: "lapack_banded".to_owned(),
            // This is a numerical linear-algebra kernel, not a symbolic
            // solver. Its public API accepts compact NumPy bands directly.
            frontends: Vec::new(),
            executions: vec![
                "factor".to_owned(),
                "solve".to_owned(),
                "solve_multiple".to_owned(),
                "refinement".to_owned(),
            ],
            methods: Vec::new(),
            layouts: vec!["banded".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: vec!["off".to_owned()],
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "pivoted_tridiagonal" | "tridiagonal_lu_pp" => Ok(SolverCapabilities {
            solver: "pivoted_tridiagonal".to_owned(),
            // This is a numerical kernel. It consumes compact one-dimensional
            // bands directly and intentionally has no symbolic frontend axis.
            frontends: Vec::new(),
            executions: vec![
                "factor".to_owned(),
                "solve".to_owned(),
                "solve_multiple".to_owned(),
            ],
            methods: Vec::new(),
            layouts: vec!["tridiagonal".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: vec!["off".to_owned()],
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "block_tridiagonal" | "block_tridiagonal_lu_consistent" => Ok(SolverCapabilities {
            solver: "block_tridiagonal".to_owned(),
            // This is a structured numerical kernel.  The Python route owns
            // compact block arrays and deliberately has no symbolic or
            // telemetry axis.
            frontends: Vec::new(),
            executions: vec![
                "factor".to_owned(),
                "solve".to_owned(),
                "solve_multiple".to_owned(),
                "refinement".to_owned(),
            ],
            methods: Vec::new(),
            layouts: vec!["block_tridiagonal".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: vec!["off".to_owned()],
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "radau" | "radau5" => Ok(SolverCapabilities {
            solver: "radau5".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned(), "aot".to_owned()],
            methods: vec!["radau5".to_owned()],
            layouts: vec!["dense".to_owned(), "sparse".to_owned(), "banded".to_owned()],
            policies: vec![
                "sequential".to_owned(),
                "parallel".to_owned(),
                "auto".to_owned(),
            ],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            // The separate RadauCallbackSolver exposes the native Dense
            // callback contract; symbolic RadauSolver remains expression-only.
            supports_analytical_callbacks: true,
            analytical_callback_layouts: vec!["dense".to_owned()],
        }),
        "bdf" => Ok(SolverCapabilities {
            solver,
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned(), "aot".to_owned()],
            methods: vec!["bdf".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec![
                "sequential".to_owned(),
                "parallel".to_owned(),
                "auto".to_owned(),
            ],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            // BdfCallbackSolver exposes the native Dense callback contract;
            // symbolic BdfSolver remains the expression-oriented entry point.
            supports_analytical_callbacks: true,
            analytical_callback_layouts: vec!["dense".to_owned()],
        }),
        "backward_euler" | "be" => Ok(SolverCapabilities {
            solver: "backward_euler".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned(), "aot".to_owned()],
            methods: vec!["backward_euler".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "lsode2" => Ok(SolverCapabilities {
            solver: "lsode2".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned(), "aot".to_owned()],
            methods: vec![
                "bdf".to_owned(),
                "adams".to_owned(),
                "auto".to_owned(),
                "lsoda".to_owned(),
            ],
            layouts: vec!["dense".to_owned(), "sparse".to_owned(), "banded".to_owned()],
            policies: vec![
                "sequential".to_owned(),
                "parallel".to_owned(),
                "auto".to_owned(),
            ],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            // Python callbacks are available on all native LSODE2 layouts.
            supports_analytical_callbacks: true,
            analytical_callback_layouts: vec![
                "dense".to_owned(),
                "sparse".to_owned(),
                "banded".to_owned(),
            ],
        }),
        "ode_api2_nonstiff" | "nonstiff" => Ok(SolverCapabilities {
            solver: "ode_api2_nonstiff".to_owned(),
            frontends: vec!["expr_legacy".to_owned()],
            executions: vec!["lambdify".to_owned()],
            methods: vec!["rk45".to_owned(), "dopri".to_owned(), "ab4".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: vec!["off".to_owned()],
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "shooting_bvp" | "shooting" => Ok(SolverCapabilities {
            solver: "shooting_bvp".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned()],
            methods: vec!["scalar_second_order_shooting".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned(), "parallel".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "bvp" | "bvp_damp" | "bvp_damped" => Ok(SolverCapabilities {
            solver: "bvp_damp".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned(), "aot".to_owned()],
            methods: vec!["damped".to_owned(), "naive".to_owned(), "frozen".to_owned()],
            layouts: vec!["dense".to_owned(), "sparse".to_owned(), "banded".to_owned()],
            policies: vec![
                "sequential".to_owned(),
                "parallel".to_owned(),
                "auto".to_owned(),
            ],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "bvp_sci" | "bvpsci" | "bvp_scipy" => Ok(SolverCapabilities {
            solver: "bvp_sci".to_owned(),
            frontends: vec![
                "expr_legacy".to_owned(),
                "atom_native".to_owned(),
                "numerical".to_owned(),
            ],
            executions: vec![
                "lambdify".to_owned(),
                "aot".to_owned(),
                "numerical".to_owned(),
            ],
            methods: vec!["scipy_like".to_owned()],
            layouts: vec!["dense".to_owned(), "sparse".to_owned(), "banded".to_owned()],
            policies: vec![
                "sequential".to_owned(),
                "parallel".to_owned(),
                "auto".to_owned(),
            ],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            supports_analytical_callbacks: true,
            analytical_callback_layouts: vec!["dense".to_owned()],
        }),
        "nonlinear" | "nonlinear_system" | "nonlinear_systems" => Ok(SolverCapabilities {
            solver: "nonlinear_system".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned(), "aot".to_owned()],
            methods: vec![
                "newton".to_owned(),
                "damped_newton".to_owned(),
                "damped_newton_advanced".to_owned(),
                "levenberg_marquardt".to_owned(),
                "backtracking_levenberg_marquardt".to_owned(),
                "levenberg_marquardt_minpack".to_owned(),
                "nielsen_levenberg_marquardt".to_owned(),
                "nielsen_levenberg_marquardt_advanced".to_owned(),
                "trust_region".to_owned(),
                "powell_dogleg".to_owned(),
                "trust_region_lm".to_owned(),
            ],
            layouts: vec!["dense".to_owned()],
            policies: vec![
                "sequential".to_owned(),
                "parallel".to_owned(),
                "auto".to_owned(),
            ],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "gavin" | "gavin_fitting" => Ok(SolverCapabilities {
            solver: "gavin_fitting".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned()],
            methods: vec!["gavin_lm".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned(), "parallel".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "varpro" | "varpro_fitting" => Ok(SolverCapabilities {
            solver: "varpro".to_owned(),
            frontends: vec!["expr_legacy".to_owned()],
            executions: vec!["lambdify".to_owned()],
            methods: vec!["variable_projection".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "universal_fitting" | "universal_fitting_all" => Ok(SolverCapabilities {
            solver: "universal_fitting".to_owned(),
            // Union across methods; use the method-specific names below when
            // validating one concrete route.
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned()],
            methods: vec!["lm".to_owned(), "gavin".to_owned(), "varpro".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned(), "parallel".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "universal_fitting_lm" | "universal_lm" => Ok(SolverCapabilities {
            solver: "universal_fitting_lm".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned()],
            methods: vec!["lm".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned(), "parallel".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "universal_fitting_gavin" | "universal_gavin" => Ok(SolverCapabilities {
            solver: "universal_fitting_gavin".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned()],
            methods: vec!["gavin".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned(), "parallel".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "universal_fitting_varpro" | "universal_varpro" => Ok(SolverCapabilities {
            solver: "universal_fitting_varpro".to_owned(),
            frontends: vec!["expr_legacy".to_owned()],
            executions: vec!["lambdify".to_owned()],
            methods: vec!["varpro".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: false,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        "fitting" | "least_squares" => Ok(SolverCapabilities {
            solver: "fitting".to_owned(),
            frontends: common_frontends,
            executions: vec!["lambdify".to_owned()],
            methods: vec!["levenberg_marquardt".to_owned()],
            layouts: vec!["dense".to_owned()],
            policies: vec!["sequential".to_owned()],
            telemetry_modes: common_telemetry_modes(),
            supports_continuation: true,
            supports_analytical_callbacks: false,
            analytical_callback_layouts: Vec::new(),
        }),
        _ => Err(InvalidConfigurationError::new_err(format!(
            "unknown solver '{solver}'; capabilities are defined for lapack_banded, pivoted_tridiagonal, block_tridiagonal, radau5, bdf, backward_euler, lsode2, ode_api2_nonstiff, bvp_damp, bvp_sci, shooting_bvp, nonlinear_system, fitting, gavin_fitting, varpro and method-specific UniversalFitting routes"
        ))),
    }
}

fn common_telemetry_modes() -> Vec<String> {
    ["off", "counters", "timings", "detailed"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[pyfunction]
pub fn capabilities(solver: &str) -> PyResult<SolverCapabilities> {
    capabilities_for(solver)
}
