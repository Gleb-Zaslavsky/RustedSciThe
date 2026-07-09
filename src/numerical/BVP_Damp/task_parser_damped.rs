//! # BVP Damped Task Parser Module
//!
//! Configuration parser for the damped BVP solver.
//!
//! The module is split into typed parsing plus thin compatibility wrappers:
//! parse task documents into specs, build solver-ready options from those specs,
//! and keep the legacy `apply_*` / `set_*` entry points as bridges.
//!
//! ## Core Functionality
//! - **Configuration Parsing**: Parse solver settings from files or strings into typed specs
//! - **Parameter Mapping**: Convert parsed data into solver-specific data structures
//! - **Pseudonym Support**: Handle common typos and alternative names for configuration keys
//! - **Grid Refinement**: Parse adaptive grid refinement strategies and parameters
//! - **Postprocessing**: Configure output options (plotting, saving, logging)
//! - **Typed Split Parsing**: Parse solver settings and postprocessing into
//!   fallible spec structs before mutating the solver state
//! - **Generated Backend Selection**: Own sparse/banded/AOT selection, compiler
//!   choice, build policy, execution policy, and chunking from the typed layer
//! - **Template Generation**: Create configuration file templates for users
//!
//! ## Main Methods (NRBVP Implementation)
//!
//! ### Core Parsing Methods
//! - `before_solve_preprocessing()` - Initialize mesh and grid settings before solving
//! - `parse_bvp_damped_task_from_str(input)` - Parse a full typed task from DSL text
//! - `parse_bvp_damped_task_from_document(document)` - Parse a full typed task from a document map
//! - `try_parse_bvp_damped_solver_settings_from_document()` - Parse settings into a typed spec
//! - `try_parse_bvp_damped_postprocessing_from_document()` - Parse postprocessing into a typed spec
//! - `try_parse_bvp_damped_task_from_document()` - Parse the full typed BVP Damp task contract
//! - `build_bvp_damped_solver_options_from_spec()` - Build solver-ready damped options from spec
//! - `build_bvp_damped_generated_backend_config_from_spec()` - Build generated-backend config from spec
//! - `apply_bvp_damped_solver_settings()` - Apply typed settings to the solver
//! - `apply_bvp_damped_postprocessing()` - Apply typed postprocessing actions
//! - `apply_bvp_damped_task_spec()` - Apply both solver settings and postprocessing
//! - `set_params_from_hashmap(result)` - Map parsed DocumentMap to solver parameters
//! - `set_postpocessing_from_hashmap(parser)` - Configure and execute postprocessing options
//!
//! ### String-based Parsing
//! - `parse_settings_from_str_with_exact_names(input)` - Legacy solver-settings-only parse with exact key names
//! - `parse_settings(parser)` - Parse with pseudonym support for common typos
//! - `parse_bvp_damped_task_from_file(path)` - Parse a full typed task from a file
//!
//! ### File-based Parsing
//! - `parse_file(path)` - Load configuration from file and return DocumentParser
//! - `parse_settings_with_exact_names(parser)` - Parse from DocumentParser with exact names
//!
//! ### Utility Functions
//! - `create_template_file(path)` - Generate configuration file template
//!
//! ## Configuration Structure
//!
//! ### solver_settings Section
//! - `scheme`: Discretization scheme ("forward" or "trapezoid")
//! - `method`: Matrix backend ("Dense", "Sparse", or "Banded")
//! - `strategy`: Solver strategy ("Damped", "Naive", "Frozen")
//! - `linear_sys_method`: Linear system solver method (Optional)
//! - `generated_backend`: Optional generated-backend preset such as `banded_aot_tcc`
//! - `matrix_backend`, `backend_policy`, `symbolic_backend`
//! - `aot_codegen_backend`, `aot_c_compiler`, `aot_build_policy`, `aot_build_profile`
//! - `aot_compile_preset`, `aot_execution_policy`
//! - `banded_linear_solver`, `refinement_steps`
//! - `abs_tolerance`: Absolute convergence tolerance
//! - `max_iterations`: Maximum solver iterations
//! - `loglevel`: Logging level (Optional)
//!
//! ### bounds Section
//! Variable-specific solution bounds: `variable_name: min_value, max_value`
//!
//! ### rel_tolerance Section
//! Variable-specific relative tolerances: `variable_name: tolerance_value`
//!
//! ### strategy_params Section
//! - `max_jac`: Maximum iterations with old Jacobian (Optional)
//! - `max_damp_iter`: Maximum damped iterations (Optional)
//! - `damp_factor`: Damping factor reduction (Optional)
//!
//! ### adaptive_strategy Section
//! - `version`: Grid refinement version
//! - `max_refinements`: Maximum refinement iterations
//!
//! ### grid_refinement Section
//! Grid refinement methods with parameters:
//! - `doubleoints`: Double grid points
//! - `easy`: Simple refinement with single parameter
//! - `grcarsmooke`: Grcar-Smooke method with 3 parameters
//! - `pearson`: Pearson method with 2 parameters
//! - `twopnt`: Two-point method with 3 parameters
//!
//! ### postprocessing Section
//! - `plot`: Enable plotting (boolean)
//! - `gnuplot`: Enable gnuplot output (boolean)
//! - `save`: Save results to file (boolean)
//! - `save_to_csv`: Save as CSV (boolean)
//! - `dont_save_log`: Disable log saving (boolean)
//!
//! ## Non-obvious Features and Tips
//!
//! ### 1. Pseudonym System for User-Friendly Input
//! The parser supports common typos and alternative names:
//! - "solver_settings" ↔ "solve_settings", "solving_settings"
//! - "bounds" ↔ "bound", "boundary", "boundaries"
//! - "abs_tolerance" ↔ "abs_tol", "absolute_error", "absolute_tolerance"
//! - "rel_tolerance" ↔ "rel_tol", "relative_error", "relevant_error"
//!
//! ### 2. Variable Name Preservation
//! Uses `keys_to_lower_case()` with exceptions for "bounds" and "rel_tolerance" sections
//! to preserve user-defined variable names while normalizing configuration keys.
//!
//! ### 3. Grid Refinement Method Parsing
//! Complex parsing logic that:
//! - Extracts method name from section key
//! - Parses vector parameters for each method
//! - Maps string names to GridRefinementMethod enum variants
//! - Handles different parameter counts per method
//!
//! ### 4. Optional Value Handling
//! Extensive use of `as_option_*()` methods to handle nullable configuration values,
//! allowing users to specify `None` or `Some(value)` in configuration files.
//!
//! ### 5. Postprocessing Automation
//! `set_postpocessing_from_hashmap()` not only parses settings but immediately executes
//! the requested postprocessing actions (plotting, saving, etc.).
//!
//! ### 6. Error Handling Strategy
//! Typed parsing uses fallible helpers and returns structured errors for missing
//! sections, missing fields, invalid values, and unsupported backend names. The
//! legacy compatibility wrappers still exist, but the typed path does not rely on
//! user-input panics.
//!
//! ### 7. Mesh Initialization
//! `before_solve_preprocessing()` creates uniform mesh from t0, t_end, and n_steps,
//! and determines if adaptive grid refinement is enabled based on strategy_params.
//!
//! ### 8. Template System
//! `create_template_file()` provides a structured template with comments explaining
//! each configuration option, making it easier for users to create valid config files.
//!
//! ## Usage Examples
//!
//! ```rust
//! // Parse from string with exact names
//! let mut solver = NRBVP::default();
//! solver.parse_settings_from_str_with_exact_names(config_string);
//!
//! // Parse from file with pseudonym support
//! let mut solver = NRBVP::default();
//! let mut parser = solver.parse_file(Some(config_path)).unwrap();
//! solver.parse_settings(&mut parser).unwrap();
//!
//! // Setup and solve
//! solver.before_solve_preprocessing();
//! solver.solve();
//! solver.set_postpocessing_from_hashmap(&mut parser);
//! ```

use crate::command_interpreter::task_parser::{DocumentMap, DocumentParser, Value};
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedSolverOptions, NRBVP, SolverParams,
};
use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
};
use crate::somelinalg::banded::{LinearSolverConfig, LinearSolverPolicy};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use nalgebra::DVector;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

/// Typed error returned by the split BVP Damp parser helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvpDampedTaskError {
    Parser(String),
    MissingSection(&'static str),
    MissingField { section: &'static str, field: String },
    InvalidField {
        section: &'static str,
        field: String,
        message: String,
    },
    UnknownGridRefinementMethod(String),
}

impl Display for BvpDampedTaskError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parser(message) => write!(f, "parser error: {}", message),
            Self::MissingSection(section) => write!(f, "missing section `{}`", section),
            Self::MissingField {
                section,
                field,
            } => write!(f, "missing field `{}.{}`", section, field),
            Self::InvalidField {
                section,
                field,
                message,
            } => write!(f, "invalid field `{}.{}`: {}", section, field, message),
            Self::UnknownGridRefinementMethod(method) => {
                write!(f, "unknown grid refinement method `{}`", method)
            }
        }
    }
}

impl Error for BvpDampedTaskError {}

/// Typed solver settings parsed from a BVP Damp task document.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpDampedSolverSettingsSpec {
    pub scheme: String,
    pub strategy: String,
    pub linear_sys_method: Option<String>,
    pub method: String,
    pub generated_backend: BvpDampedGeneratedBackendSpec,
    pub abs_tolerance: f64,
    pub max_iterations: usize,
    pub loglevel: Option<String>,
    pub dont_save_log: bool,
    pub bounds: Option<HashMap<String, (f64, f64)>>,
    pub rel_tolerance: Option<HashMap<String, f64>>,
    pub strategy_params: Option<SolverParams>,
}

/// Typed generated-backend settings parsed from a BVP Damp task document.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BvpDampedGeneratedBackendSpec {
    pub preset: Option<String>,
    pub matrix_backend: Option<String>,
    pub backend_policy: Option<String>,
    pub symbolic_backend: Option<String>,
    pub aot_codegen_backend: Option<String>,
    pub aot_c_compiler: Option<String>,
    pub aot_build_policy: Option<String>,
    pub aot_build_profile: Option<String>,
    pub aot_compile_preset: Option<String>,
    pub aot_execution_policy: Option<String>,
    pub banded_linear_solver: Option<String>,
    pub refinement_steps: Option<usize>,
}

/// Typed postprocessing settings parsed from a BVP Damp task document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BvpDampedPostprocessingSpec {
    pub plot: bool,
    pub gnuplot: bool,
    pub save: bool,
    pub save_to_csv: bool,
    pub filename: Option<String>,
}

/// Full typed BVP Damp task contract: solver settings plus postprocessing.
///
/// This is the lightweight analog of the BVP/IVP split-task specs: the damped
/// solver does not own the physical equations here, but it still benefits from a
/// single typed object that groups the solver-facing configuration with the
/// output plan.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpDampedTaskSpec {
    pub solver_settings: BvpDampedSolverSettingsSpec,
    pub postprocessing: BvpDampedPostprocessingSpec,
}

impl BvpDampedTaskSpec {
    /// Extract the solver-settings subset.
    pub fn solver_settings_spec(&self) -> BvpDampedSolverSettingsSpec {
        self.solver_settings.clone()
    }

    /// Extract the postprocessing subset.
    pub fn postprocessing_spec(&self) -> BvpDampedPostprocessingSpec {
        self.postprocessing.clone()
    }
}

impl BvpDampedGeneratedBackendSpec {
    /// Build the generated-backend config without mutating a live solver.
    pub fn build_generated_backend_config(
        &self,
        method: &str,
    ) -> Result<GeneratedBackendConfig, BvpDampedTaskError> {
        build_bvp_damped_generated_backend_config_from_spec(self, method)
    }
}

impl BvpDampedSolverSettingsSpec {
    /// Build solver-ready damped options from the typed settings spec.
    pub fn build_solver_options(&self) -> Result<DampedSolverOptions, BvpDampedTaskError> {
        build_bvp_damped_solver_options_from_spec(self)
    }
}

fn required_section<'a>(
    document: &'a DocumentMap,
    section: &'static str,
) -> Result<&'a GenericSectionMap, BvpDampedTaskError> {
    document
        .get(section)
        .ok_or(BvpDampedTaskError::MissingSection(section))
}

fn required_values<'a>(
    section: &'a GenericSectionMap,
    section_name: &'static str,
    field: &str,
) -> Result<&'a Vec<Value>, BvpDampedTaskError> {
    section
        .get(field)
        .ok_or_else(|| BvpDampedTaskError::MissingField {
            section: section_name,
            field: field.to_string(),
        })?
        .as_ref()
        .ok_or_else(|| BvpDampedTaskError::MissingField {
            section: section_name,
            field: field.to_string(),
        })
}

fn first_string(
    section: &GenericSectionMap,
    section_name: &'static str,
    field: &str,
) -> Result<String, BvpDampedTaskError> {
    required_values(section, section_name, field)?
        .first()
        .and_then(|value| value.as_string())
        .cloned()
        .ok_or_else(|| BvpDampedTaskError::InvalidField {
            section: section_name,
            field: field.to_string(),
            message: "expected string".to_string(),
        })
}

fn first_usize(
    section: &GenericSectionMap,
    section_name: &'static str,
    field: &str,
) -> Result<usize, BvpDampedTaskError> {
    required_values(section, section_name, field)?
        .first()
        .and_then(|value| value.as_usize())
        .ok_or_else(|| BvpDampedTaskError::InvalidField {
            section: section_name,
            field: field.to_string(),
            message: "expected usize".to_string(),
        })
}

fn first_float(
    section: &GenericSectionMap,
    section_name: &'static str,
    field: &str,
) -> Result<f64, BvpDampedTaskError> {
    required_values(section, section_name, field)?
        .first()
        .and_then(|value| value.as_float())
        .ok_or_else(|| BvpDampedTaskError::InvalidField {
            section: section_name,
            field: field.to_string(),
            message: "expected float".to_string(),
        })
}

fn first_bool(
    section: &GenericSectionMap,
    section_name: &'static str,
    field: &str,
) -> Result<bool, BvpDampedTaskError> {
    required_values(section, section_name, field)?
        .first()
        .and_then(|value| value.as_boolean())
        .ok_or_else(|| BvpDampedTaskError::InvalidField {
            section: section_name,
            field: field.to_string(),
            message: "expected bool".to_string(),
        })
}

fn get_optional_string(
    section: &GenericSectionMap,
    field: &str,
    section_name: &'static str,
) -> Result<Option<String>, BvpDampedTaskError> {
    match section.get(field) {
        Some(Some(values)) => {
            let value = values
                .first()
                .and_then(|value| value.as_string().or_else(|| value.as_option_string()))
                .cloned()
                .ok_or_else(|| BvpDampedTaskError::InvalidField {
                    section: section_name,
                    field: field.to_string(),
                    message: "expected string".to_string(),
                })?;
            Ok(Some(value))
        }
        Some(None) => Ok(None),
        None => Ok(None),
    }
}

fn get_optional_usize(
    section: &GenericSectionMap,
    field: &str,
    section_name: &'static str,
) -> Result<Option<usize>, BvpDampedTaskError> {
    match section.get(field) {
        Some(Some(values)) => {
            let value = values
                .first()
                .and_then(|value| value.as_usize().or_else(|| value.as_option_usize()))
                .ok_or_else(|| BvpDampedTaskError::InvalidField {
                    section: section_name,
                    field: field.to_string(),
                    message: "expected usize".to_string(),
                })?;
            Ok(Some(value))
        }
        Some(None) => Ok(None),
        None => Ok(None),
    }
}

fn parse_grid_refinement_method(
    method_name: &str,
    method_params: &[f64],
) -> Result<GridRefinementMethod, BvpDampedTaskError> {
    let method = match method_name {
        "doubleoints" => GridRefinementMethod::DoublePoints,
        "easy" => GridRefinementMethod::Easy(*method_params.first().ok_or_else(|| {
            BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected at least 1 parameter".to_string(),
            }
        })?),
        "grcarsmooke" => GridRefinementMethod::GrcarSmooke(
            *method_params.first().ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 3 parameters".to_string(),
            })?,
            *method_params.get(1).ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 3 parameters".to_string(),
            })?,
            *method_params.get(2).ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 3 parameters".to_string(),
            })?,
        ),
        "pearson" => GridRefinementMethod::Pearson(
            *method_params.first().ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 2 parameters".to_string(),
            })?,
            *method_params.get(1).ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 2 parameters".to_string(),
            })?,
        ),
        "twopnt" => GridRefinementMethod::TwoPoint(
            *method_params.first().ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 3 parameters".to_string(),
            })?,
            *method_params.get(1).ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 3 parameters".to_string(),
            })?,
            *method_params.get(2).ok_or_else(|| BvpDampedTaskError::InvalidField {
                section: "grid_refinement",
                field: method_name.to_string(),
                message: "expected 3 parameters".to_string(),
            })?,
        ),
        other => {
            return Err(BvpDampedTaskError::UnknownGridRefinementMethod(
                other.to_string(),
            ))
        }
    };
    Ok(method)
}

fn normalized_option(raw: Option<&str>) -> Option<String> {
    raw.map(normalize_token).filter(|value| !value.is_empty())
}

fn normalize_token(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn invalid_solver_option(field: &str, message: String) -> BvpDampedTaskError {
    BvpDampedTaskError::InvalidField {
        section: "solver_settings",
        field: field.to_string(),
        message,
    }
}

fn parse_generated_backend_options(
    section: &GenericSectionMap,
) -> Result<BvpDampedGeneratedBackendSpec, BvpDampedTaskError> {
    Ok(BvpDampedGeneratedBackendSpec {
        preset: get_optional_string(section, "generated_backend", "solver_settings")?,
        matrix_backend: get_optional_string(section, "matrix_backend", "solver_settings")?,
        backend_policy: get_optional_string(section, "backend_policy", "solver_settings")?,
        symbolic_backend: get_optional_string(section, "symbolic_backend", "solver_settings")?,
        aot_codegen_backend: get_optional_string(
            section,
            "aot_codegen_backend",
            "solver_settings",
        )?,
        aot_c_compiler: get_optional_string(section, "aot_c_compiler", "solver_settings")?,
        aot_build_policy: get_optional_string(section, "aot_build_policy", "solver_settings")?,
        aot_build_profile: get_optional_string(section, "aot_build_profile", "solver_settings")?,
        aot_compile_preset: get_optional_string(section, "aot_compile_preset", "solver_settings")?,
        aot_execution_policy: get_optional_string(
            section,
            "aot_execution_policy",
            "solver_settings",
        )?,
        banded_linear_solver: get_optional_string(
            section,
            "banded_linear_solver",
            "solver_settings",
        )?,
        refinement_steps: get_optional_usize(section, "refinement_steps", "solver_settings")?,
    })
}

fn parse_aot_build_profile(raw: &str) -> Result<AotBuildProfile, BvpDampedTaskError> {
    match normalize_token(raw).as_str() {
        "release" => Ok(AotBuildProfile::Release),
        "debug" => Ok(AotBuildProfile::Debug),
        other => Err(invalid_solver_option(
            "aot_build_profile",
            format!("unknown AOT build profile `{other}`"),
        )),
    }
}

fn parse_aot_build_policy(
    raw_policy: Option<&str>,
    raw_profile: Option<&str>,
) -> Result<AotBuildPolicy, BvpDampedTaskError> {
    let profile = parse_aot_build_profile(raw_profile.unwrap_or("release"))?;
    match normalized_option(raw_policy).as_deref() {
        None | Some("use_if_available") | Some("use") | Some("auto") => {
            Ok(AotBuildPolicy::UseIfAvailable)
        }
        Some("build_if_missing") | Some("build") => Ok(AotBuildPolicy::BuildIfMissing { profile }),
        Some("require_prebuilt") | Some("require") | Some("prebuilt") => {
            Ok(AotBuildPolicy::RequirePrebuilt)
        }
        Some("rebuild_always") | Some("rebuild") => Ok(AotBuildPolicy::RebuildAlways { profile }),
        Some(other) => Err(invalid_solver_option(
            "aot_build_policy",
            format!("unknown AOT build policy `{other}`"),
        )),
    }
}

fn parse_aot_codegen_backend(raw: &str) -> Result<AotCodegenBackend, BvpDampedTaskError> {
    match normalize_token(raw).as_str() {
        "rust" | "rs" => Ok(AotCodegenBackend::Rust),
        "c" => Ok(AotCodegenBackend::C),
        "zig" => Ok(AotCodegenBackend::Zig),
        other => Err(invalid_solver_option(
            "aot_codegen_backend",
            format!("unknown AOT codegen backend `{other}`"),
        )),
    }
}

fn parse_aot_execution_policy(raw: &str) -> Result<AotExecutionPolicy, BvpDampedTaskError> {
    match normalize_token(raw).as_str() {
        "auto" => Ok(AotExecutionPolicy::Auto),
        "sequential" | "sequential_only" | "seq" => Ok(AotExecutionPolicy::SequentialOnly),
        "parallel" => Err(invalid_solver_option(
            "aot_execution_policy",
            "`parallel` requires a ParallelExecutorConfig and is not yet exposed in task files"
                .to_string(),
        )),
        other => Err(invalid_solver_option(
            "aot_execution_policy",
            format!("unknown AOT execution policy `{other}`"),
        )),
    }
}

fn parse_backend_policy(raw: &str) -> Result<BackendSelectionPolicy, BvpDampedTaskError> {
    match normalize_token(raw).as_str() {
        "numeric" | "numeric_only" => Ok(BackendSelectionPolicy::NumericOnly),
        "lambdify" | "lambdify_only" => Ok(BackendSelectionPolicy::LambdifyOnly),
        "aot" | "aot_only" => Ok(BackendSelectionPolicy::AotOnly),
        "prefer_aot" | "prefer_aot_then_lambdify" | "aot_then_lambdify" => {
            Ok(BackendSelectionPolicy::PreferAotThenLambdify)
        }
        "prefer_aot_then_numeric" | "aot_then_numeric" => {
            Ok(BackendSelectionPolicy::PreferAotThenNumeric)
        }
        "prefer_lambdify_then_numeric" | "lambdify_then_numeric" => {
            Ok(BackendSelectionPolicy::PreferLambdifyThenNumeric)
        }
        other => Err(invalid_solver_option(
            "backend_policy",
            format!("unknown backend policy `{other}`"),
        )),
    }
}

fn parse_symbolic_backend(raw: &str) -> Result<BvpSymbolicAssemblyBackend, BvpDampedTaskError> {
    match normalize_token(raw).as_str() {
        "exprlegacy" | "expr_legacy" | "legacy" => Ok(BvpSymbolicAssemblyBackend::ExprLegacy),
        "atomview" | "atom_view" | "atom" => Ok(BvpSymbolicAssemblyBackend::AtomView),
        other => Err(invalid_solver_option(
            "symbolic_backend",
            format!("unknown symbolic backend `{other}`"),
        )),
    }
}

fn apply_aot_compile_preset(
    config: GeneratedBackendConfig,
    raw: &str,
) -> Result<GeneratedBackendConfig, BvpDampedTaskError> {
    match normalize_token(raw).as_str() {
        "production" | "prod" | "release" => Ok(config.with_aot_compile_production()),
        "fast_build" | "fastbuild" => Ok(config.with_aot_compile_fast_build()),
        "dev_fastest" | "devfastest" | "debug_fast" => Ok(config.with_aot_compile_dev_fastest()),
        other => Err(invalid_solver_option(
            "aot_compile_preset",
            format!("unknown AOT compile preset `{other}`"),
        )),
    }
}

fn apply_matrix_backend_override(
    config: GeneratedBackendConfig,
    raw: &str,
) -> Result<GeneratedBackendConfig, BvpDampedTaskError> {
    match normalize_token(raw).as_str() {
        "default" | "none" => Ok(config),
        "sparse" | "sparse_col" | "sparsecol" => {
            Ok(config.with_matrix_backend_override(MatrixBackend::SparseCol))
        }
        "banded" => Ok(config.with_matrix_backend_override(MatrixBackend::Banded)),
        "dense" => Ok(config.with_matrix_backend_override(MatrixBackend::Dense)),
        other => Err(invalid_solver_option(
            "matrix_backend",
            format!("unknown matrix backend `{other}`"),
        )),
    }
}

fn parse_banded_linear_solver_config(
    raw_solver: Option<&str>,
    refinement_steps: Option<usize>,
) -> Result<LinearSolverConfig, BvpDampedTaskError> {
    let refinement_steps = refinement_steps.unwrap_or(0);
    let config = match normalized_option(raw_solver).as_deref() {
        None | Some("default") | Some("auto") => {
            LinearSolverConfig::auto().with_iterative_refinement_steps(refinement_steps)
        }
        Some("faithful") | Some("lapack") | Some("lapack_style_banded_lu") => {
            LinearSolverConfig::faithful_banded_with_refinement(refinement_steps)
        }
        Some("block_tridiagonal") | Some("block_tridiagonal_lu") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceBlockTridiagonal,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some("block_tridiagonal_consistent")
        | Some("block_tridiagonal_lu_consistent")
        | Some("consistent") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceBlockTridiagonalConsistent,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some("faer_sparse") | Some("faer_sparse_lu") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceFaerSparse,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some("general_partial_pivot") | Some("dense_general_pivot") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceGeneralBandedPartialPivot,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some(other) => {
            return Err(invalid_solver_option(
                "banded_linear_solver",
                format!("unknown banded linear solver `{other}`"),
            ))
        }
    };
    Ok(config)
}

/// Build the generated-backend config from typed damped solver settings.
pub fn build_bvp_damped_generated_backend_config_from_spec(
    spec: &BvpDampedGeneratedBackendSpec,
    method: &str,
) -> Result<GeneratedBackendConfig, BvpDampedTaskError> {
    let mut config = match normalized_option(spec.preset.as_deref()).as_deref() {
        None | Some("default") | Some("defaults") => match normalize_token(method).as_str() {
            "banded" => GeneratedBackendConfig::banded_defaults(),
            "dense" | "sparse" => GeneratedBackendConfig::sparse_defaults(),
            other => {
                return Err(invalid_solver_option(
                    "method",
                    format!("unknown solver matrix route `{other}`"),
                ))
            }
        },
        Some("sparse") | Some("sparse_default") | Some("sparse_defaults") => {
            GeneratedBackendConfig::sparse_defaults()
        }
        Some("sparse_lambdify") | Some("lambdify_sparse") => {
            GeneratedBackendConfig::sparse_defaults()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        }
        Some("sparse_aot") | Some("sparse_build_if_missing") => {
            GeneratedBackendConfig::sparse_build_if_missing_release()
        }
        Some("sparse_aot_gcc") | Some("sparse_atomview_gcc") => {
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc()
        }
        Some("sparse_aot_tcc") | Some("sparse_atomview_tcc") | Some("sparse_repeated") => {
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
        }
        Some("sparse_aot_zig") | Some("sparse_atomview_zig") => {
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig()
        }
        Some("banded") | Some("banded_default") | Some("banded_defaults") => {
            GeneratedBackendConfig::banded_defaults()
        }
        Some("banded_lambdify") | Some("lambdify_banded") => {
            GeneratedBackendConfig::banded_lambdify_defaults()
        }
        Some("banded_aot") | Some("banded_build_if_missing") => {
            GeneratedBackendConfig::banded_build_if_missing_release()
        }
        Some("banded_aot_gcc") | Some("banded_atomview_gcc") => {
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc()
        }
        Some("banded_aot_tcc") | Some("banded_atomview_tcc") | Some("banded_repeated") => {
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
        }
        Some("banded_aot_zig") | Some("banded_atomview_zig") => {
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig()
        }
        Some(other) => {
            return Err(invalid_solver_option(
                "generated_backend",
                format!("unknown generated backend preset `{other}`"),
            ))
        }
    };

    if let Some(matrix_backend) = spec.matrix_backend.as_deref() {
        config = apply_matrix_backend_override(config, matrix_backend)?;
    }
    if let Some(policy) = spec.backend_policy.as_deref() {
        config = config.with_backend_policy_override(Some(parse_backend_policy(policy)?));
    }
    if let Some(symbolic_backend) = spec.symbolic_backend.as_deref() {
        config = config.with_symbolic_assembly_backend(parse_symbolic_backend(symbolic_backend)?);
    }
    if let Some(codegen_backend) = spec.aot_codegen_backend.as_deref() {
        config = config.with_aot_codegen_backend(parse_aot_codegen_backend(codegen_backend)?);
    }
    if let Some(compiler) = spec.aot_c_compiler.as_deref() {
        config = config.with_aot_c_compiler(compiler);
    }
    if spec.aot_build_policy.is_some() || spec.aot_build_profile.is_some() {
        config = config.with_aot_build_policy(parse_aot_build_policy(
            spec.aot_build_policy.as_deref(),
            spec.aot_build_profile.as_deref(),
        )?);
    }
    if let Some(compile_preset) = spec.aot_compile_preset.as_deref() {
        config = apply_aot_compile_preset(config, compile_preset)?;
    }
    if let Some(execution_policy) = spec.aot_execution_policy.as_deref() {
        config = config.with_aot_execution_policy(parse_aot_execution_policy(execution_policy)?);
    }
    if spec.banded_linear_solver.is_some() || spec.refinement_steps.is_some() {
        config = config.with_banded_linear_solver_config(parse_banded_linear_solver_config(
            spec.banded_linear_solver.as_deref(),
            spec.refinement_steps,
        )?);
    }

    Ok(config)
}

/// Build solver-ready damped options from typed solver settings.
pub fn build_bvp_damped_solver_options_from_spec(
    spec: &BvpDampedSolverSettingsSpec,
) -> Result<DampedSolverOptions, BvpDampedTaskError> {
    let generated_backend_config =
        build_bvp_damped_generated_backend_config_from_spec(&spec.generated_backend, &spec.method)?;
    Ok(DampedSolverOptions::new(
        spec.scheme.clone(),
        spec.strategy.clone(),
        spec.strategy_params.clone(),
        spec.linear_sys_method.clone(),
        spec.method.clone(),
        spec.abs_tolerance,
        spec.rel_tolerance.clone(),
        spec.max_iterations,
        spec.bounds.clone(),
        spec.loglevel.clone(),
    )
    .with_generated_backend_config(generated_backend_config))
}

/// Parse the solver-settings block into a typed spec without mutating the solver.
pub fn parse_bvp_damped_solver_settings_from_document(
    document: &DocumentMap,
) -> Result<BvpDampedSolverSettingsSpec, BvpDampedTaskError> {
    let solver_settings = required_section(document, "solver_settings")?;
    let scheme = first_string(solver_settings, "solver_settings", "scheme")?;
    let strategy = first_string(solver_settings, "solver_settings", "strategy")?;
    let linear_sys_method = solver_settings
        .get("linear_sys_method")
        .and_then(|value| value.as_ref())
        .and_then(|values| values.first())
        .and_then(|value| value.as_option_string())
        .cloned();
    let method = first_string(solver_settings, "solver_settings", "method")?;
    let abs_tolerance = first_float(solver_settings, "solver_settings", "abs_tolerance")?;
    let max_iterations = first_usize(solver_settings, "solver_settings", "max_iterations")?;
    let loglevel = solver_settings
        .get("loglevel")
        .and_then(|value| value.as_ref())
        .and_then(|values| values.first())
        .and_then(|value| value.as_option_string())
        .cloned();
    let dont_save_log = if solver_settings.get("dont_save_log").is_some() {
        first_bool(solver_settings, "solver_settings", "dont_save_log")?
    } else {
        true
    };
    let generated_backend = parse_generated_backend_options(solver_settings)?;

    let bounds = if let Some(bounds_section) = document.get("bounds") {
        let bounds = bounds_section
            .iter()
            .map(|(key, value)| {
                let values = value.as_ref().ok_or_else(|| BvpDampedTaskError::InvalidField {
                    section: "bounds",
                    field: key.clone(),
                    message: "expected pair of floats".to_string(),
                })?;
                let lower = values
                    .first()
                    .and_then(|value| value.as_float())
                    .ok_or_else(|| BvpDampedTaskError::InvalidField {
                        section: "bounds",
                        field: key.clone(),
                        message: "expected lower bound float".to_string(),
                    })?;
                let upper = values
                    .get(1)
                    .and_then(|value| value.as_float())
                    .ok_or_else(|| BvpDampedTaskError::InvalidField {
                        section: "bounds",
                        field: key.clone(),
                        message: "expected upper bound float".to_string(),
                    })?;
                Ok((key.clone(), (lower, upper)))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        Some(bounds)
    } else {
        None
    };

    let rel_tolerance = if let Some(rel_tolerance_section) = document.get("rel_tolerance") {
        let rel_tolerance = rel_tolerance_section
            .iter()
            .map(|(key, value)| {
                let values = value.as_ref().ok_or_else(|| BvpDampedTaskError::InvalidField {
                    section: "rel_tolerance",
                    field: key.clone(),
                    message: "expected single float".to_string(),
                })?;
                let tolerance = values
                    .first()
                    .and_then(|value| value.as_float())
                    .ok_or_else(|| BvpDampedTaskError::InvalidField {
                        section: "rel_tolerance",
                        field: key.clone(),
                        message: "expected float".to_string(),
                    })?;
                Ok((key.clone(), tolerance))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        Some(rel_tolerance)
    } else {
        None
    };

    let strategy_params = if let Some(strategy_params_section) = document.get("strategy_params") {
        let max_jac = strategy_params_section
            .get("max_jac")
            .and_then(|value| value.as_ref())
            .and_then(|values| values.first())
            .and_then(|value| value.as_option_usize());
        let max_damp_iter = strategy_params_section
            .get("max_damp_iter")
            .and_then(|value| value.as_ref())
            .and_then(|values| values.first())
            .and_then(|value| value.as_option_usize());
        let damp_factor = strategy_params_section
            .get("damp_factor")
            .and_then(|value| value.as_ref())
            .and_then(|values| values.first())
            .and_then(|value| value.as_option_float());

        let adaptive = if let Some(adaptive_strategy) = document.get("adaptive_strategy") {
            let version = first_usize(adaptive_strategy, "adaptive_strategy", "version")?;
            let max_refinements =
                first_usize(adaptive_strategy, "adaptive_strategy", "max_refinements")?;
            let grid_method_section = required_section(document, "grid_refinement")?;
            let (method_name, method_params) = grid_method_section
                .iter()
                .next()
                .ok_or(BvpDampedTaskError::MissingSection("grid_refinement"))?;
            let method_values = method_params
                .as_ref()
                .ok_or_else(|| BvpDampedTaskError::InvalidField {
                    section: "grid_refinement",
                    field: method_name.clone(),
                    message: "expected vector".to_string(),
                })?;
            let method_params = method_values
                .first()
                .and_then(|value| value.as_vector())
                .ok_or_else(|| BvpDampedTaskError::InvalidField {
                    section: "grid_refinement",
                    field: method_name.clone(),
                    message: "expected vector payload".to_string(),
                })?;
            let grid_method = parse_grid_refinement_method(method_name, method_params)?;
            Some(AdaptiveGridConfig {
                version,
                max_refinements,
                grid_method,
            })
        } else {
            None
        };

        Some(SolverParams {
            max_jac,
            max_damp_iter,
            damp_factor,
            adaptive,
        })
    } else {
        None
    };

    Ok(BvpDampedSolverSettingsSpec {
        scheme,
        strategy,
        linear_sys_method,
        method,
        generated_backend,
        abs_tolerance,
        max_iterations,
        loglevel,
        dont_save_log,
        bounds,
        rel_tolerance,
        strategy_params,
    })
}

/// Parse the postprocessing block into a typed spec without mutating the solver.
pub fn parse_bvp_damped_postprocessing_from_document(
    document: &DocumentMap,
) -> Result<BvpDampedPostprocessingSpec, BvpDampedTaskError> {
    let postprocessing = if let Some(section) = document.get("postprocessing") {
        section
    } else {
        return Ok(BvpDampedPostprocessingSpec {
            plot: false,
            gnuplot: false,
            save: false,
            save_to_csv: false,
            filename: None,
        });
    };

    let plot = postprocessing
        .get("plot")
        .and_then(|value| value.as_ref())
        .and_then(|values| values.first())
        .and_then(|value| value.as_boolean())
        .unwrap_or(false);
    let gnuplot = postprocessing
        .get("gnuplot")
        .and_then(|value| value.as_ref())
        .and_then(|values| values.first())
        .and_then(|value| value.as_boolean())
        .unwrap_or(false);
    let save = postprocessing
        .get("save")
        .and_then(|value| value.as_ref())
        .and_then(|values| values.first())
        .and_then(|value| value.as_boolean())
        .unwrap_or(false);
    let save_to_csv = postprocessing
        .get("save_to_csv")
        .and_then(|value| value.as_ref())
        .and_then(|values| values.first())
        .and_then(|value| value.as_boolean())
        .unwrap_or(false);
    let filename = postprocessing
        .get("filename")
        .and_then(|value| value.as_ref())
        .and_then(|values| values.first())
        .and_then(|value| value.as_string())
        .cloned();

    Ok(BvpDampedPostprocessingSpec {
        plot,
        gnuplot,
        save,
        save_to_csv,
        filename,
    })
}

/// Parse the full typed BVP Damp task contract from a document.
pub fn parse_bvp_damped_task_from_document(
    document: &DocumentMap,
) -> Result<BvpDampedTaskSpec, BvpDampedTaskError> {
    let solver_settings = parse_bvp_damped_solver_settings_from_document(document)?;
    let postprocessing = parse_bvp_damped_postprocessing_from_document(document)?;
    Ok(BvpDampedTaskSpec {
        solver_settings,
        postprocessing,
    })
}

/// Parse the full typed BVP Damp task contract from DSL text.
pub fn parse_bvp_damped_task_from_str(input: &str) -> Result<BvpDampedTaskSpec, BvpDampedTaskError> {
    let mut parser = DocumentParser::new(input.to_owned());
    parser
        .parse_document()
        .map_err(BvpDampedTaskError::Parser)?;
    parser.keys_to_lower_case(Some(vec![
        "bounds".to_string(),
        "rel_tolerance".to_string(),
    ]));
    let result = parser
        .get_result()
        .ok_or(BvpDampedTaskError::MissingSection("task document"))?;
    parse_bvp_damped_task_from_document(result)
}
impl NRBVP {
    /// in standard NRBVP approach this operation is made in the new(..) method but when
    /// but if params are passed from the hashmap (and may be by parsing task files), it is done here

    pub fn before_solve_preprocessing(&mut self) {
        let t0 = self.t0;
        let t_end = self.t_end;
        let n_steps = self.n_steps;
        let h = (t_end - t0) / n_steps as f64;
        let T_list: Vec<f64> = (0..n_steps + 1)
            .map(|i| t0 + (i as f64) * h)
            .collect::<Vec<_>>();

        self.x_mesh = DVector::from_vec(T_list);

        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        let new_grid_enabled_: bool = if let Some(ref params) = self.strategy_params {
            params.adaptive.is_some()
        } else {
            false
        };
        self.new_grid_enabled = new_grid_enabled_;
    }

    /// Apply typed solver settings to the solver state.
    ///
    /// This remains a compatibility wrapper around the fallible typed path.
    pub fn apply_bvp_damped_solver_settings(&mut self, spec: &BvpDampedSolverSettingsSpec) {
        self.try_apply_bvp_damped_solver_settings(spec)
            .expect("typed damped solver settings should produce a generated backend config");
    }

    /// Fallible bridge that applies typed solver settings to the solver state.
    pub fn try_apply_bvp_damped_solver_settings(
        &mut self,
        spec: &BvpDampedSolverSettingsSpec,
    ) -> Result<(), BvpDampedTaskError> {
        self.scheme = spec.scheme.clone();
        self.strategy = spec.strategy.clone();
        self.linear_sys_method = spec.linear_sys_method.clone();
        self.method = spec.method.clone();
        self.abs_tolerance = spec.abs_tolerance;
        self.max_iterations = spec.max_iterations;
        self.loglevel = spec.loglevel.clone();
        self.dont_save_log(spec.dont_save_log);
        self.Bounds = spec.bounds.clone();
        self.rel_tolerance = spec.rel_tolerance.clone();
        self.strategy_params = spec.strategy_params.clone();
        let config =
            build_bvp_damped_generated_backend_config_from_spec(&spec.generated_backend, &spec.method)?;
        self.set_generated_backend_config(config);
        Ok(())
    }

    /// Apply typed postprocessing settings to the solver state.
    pub fn apply_bvp_damped_postprocessing(&mut self, spec: &BvpDampedPostprocessingSpec) {
        if spec.plot {
            self.plot_result();
        }
        if spec.gnuplot {
            self.gnuplot_result();
        }
        if spec.save {
            self.save_to_file(spec.filename.clone());
        }
        if spec.save_to_csv {
            self.save_to_csv(spec.filename.clone());
        }
    }

    /// Parse solver settings into a typed spec without mutating the solver.
    pub fn try_parse_bvp_damped_solver_settings_from_document(
        &self,
        result: &DocumentMap,
    ) -> Result<BvpDampedSolverSettingsSpec, BvpDampedTaskError> {
        parse_bvp_damped_solver_settings_from_document(result)
    }

    /// Parse postprocessing settings into a typed spec without mutating the solver.
    pub fn try_parse_bvp_damped_postprocessing_from_document(
        &self,
        result: &DocumentMap,
    ) -> Result<BvpDampedPostprocessingSpec, BvpDampedTaskError> {
        parse_bvp_damped_postprocessing_from_document(result)
    }

    /// Parse the full typed BVP Damp task contract without mutating the solver.
    pub fn try_parse_bvp_damped_task_from_document(
        &self,
        result: &DocumentMap,
    ) -> Result<BvpDampedTaskSpec, BvpDampedTaskError> {
        parse_bvp_damped_task_from_document(result)
    }

    /// Parse a full typed BVP Damp task from DSL text.
    pub fn try_parse_bvp_damped_task_from_str(
        &self,
        input: &str,
    ) -> Result<BvpDampedTaskSpec, BvpDampedTaskError> {
        parse_bvp_damped_task_from_str(input)
    }

    pub fn set_params_from_hashmap(&mut self, result: DocumentMap) {
        let spec = parse_bvp_damped_solver_settings_from_document(&result)
            .expect("Failed to parse BVP Damp solver settings");
        self.apply_bvp_damped_solver_settings(&spec);
    }

    pub fn set_postpocessing_from_hashmap(&mut self, parser: &mut DocumentParser) {
        let result: DocumentMap = parser.get_result().unwrap().clone();
        let spec = parse_bvp_damped_postprocessing_from_document(&result)
            .expect("Failed to parse BVP Damp postprocessing");
        self.apply_bvp_damped_postprocessing(&spec);
    }

    /// Apply the typed full task contract to the solver state.
    pub fn apply_bvp_damped_task_spec(&mut self, spec: &BvpDampedTaskSpec) {
        self.apply_bvp_damped_solver_settings(&spec.solver_settings);
        self.apply_bvp_damped_postprocessing(&spec.postprocessing);
    }

    pub fn parse_settings_from_str_with_exact_names(&mut self, input: &str) {
        let mut parser = DocumentParser::new(input.to_owned());

        let _ = parser.parse_document();
        // keys inside bounds header of rel_tolerance don't need to be converted to lowecase
        // because they contain problem variables names which are arbituary
        parser.keys_to_lower_case(Some(vec![
            "bounds".to_string(),
            "rel_tolerance".to_string(),
        ]));
        let result: DocumentMap = parser.get_result().expect("Failed to get result").clone();
        let spec = parse_bvp_damped_solver_settings_from_document(&result)
            .expect("Failed to parse BVP Damp solver settings");
        self.apply_bvp_damped_solver_settings(&spec);
    }

    pub fn parse_file(
        &mut self,
        path: Option<PathBuf>,
    ) -> Result<DocumentParser, String> {
        let mut parser = DocumentParser::new(String::new());
        parser.setting_from_file(path)?;
        Ok(parser)
    }

    /// Parse a full typed BVP Damp task from an on-disk task document.
    pub fn parse_bvp_damped_task_from_file(
        &self,
        path: Option<PathBuf>,
    ) -> Result<BvpDampedTaskSpec, BvpDampedTaskError> {
        let mut parser = DocumentParser::new(String::new());
        parser
            .setting_from_file(path)
            .map_err(|e| BvpDampedTaskError::InvalidField {
                section: "task document",
                field: "file".to_string(),
                message: e,
            })?;
        parser
            .parse_document()
            .map_err(BvpDampedTaskError::Parser)?;
        parser.keys_to_lower_case(Some(vec![
            "bounds".to_string(),
            "rel_tolerance".to_string(),
        ]));
        let result = parser
            .get_result()
            .ok_or(BvpDampedTaskError::MissingSection("task document"))?;
        parse_bvp_damped_task_from_document(result)
    }

    pub fn parse_settings_with_exact_names(
        &mut self,
        parser: &mut DocumentParser,
    ) -> Result<(), String> {
        let _ = parser.parse_document();
        parser.keys_to_lower_case(Some(vec![
            "bounds".to_string(),
            "rel_tolerance".to_string(),
        ]));
        let result: DocumentMap = parser
            .get_result()
            .ok_or("No result after parsing")?
            .clone();
        let spec = parse_bvp_damped_solver_settings_from_document(&result)
            .map_err(|e| e.to_string())?;
        self.apply_bvp_damped_solver_settings(&spec);
        Ok(())
    }
    /// Parses the settings from a string and covers some common typos
    pub fn parse_settings(&mut self, parser: &mut DocumentParser) -> Result<(), String> {
        let headers_pseudonims: HashMap<String, Vec<String>> = HashMap::from([
            (
                "solver_settings".to_string(),
                vec!["solve_settings".to_string(), "solving_settings".to_string()],
            ),
            (
                "bounds".to_string(),
                vec![
                    "bound".to_string(),
                    "boundary".to_string(),
                    "boundaries".to_string(),
                ],
            ),
            (
                "rel_tolerance".to_string(),
                vec![
                    "rel_tol".to_string(),
                    "rel_error".to_string(),
                    "relative_error".to_string(),
                    "relevant_error".to_string(),
                    "relative_tolerance".to_string(),
                ],
            ),
            (
                "strategy_params".to_string(),
                vec![
                    "solver_params".to_string(),
                    "strategy_parameters".to_string(),
                    "solver_parameters".to_string(),
                ],
            ),
            (
                "adaptive_strategy".to_string(),
                vec!["adaptive".to_string(), "adaptive_settings".to_string()],
            ),
            (
                "grid_refinement".to_string(),
                vec![
                    "grid".to_string(),
                    "grid_ref".to_string(),
                    "grid_refinement_method".to_string(),
                ],
            ),
        ]);
        let field_name_pseudonims: HashMap<String, Vec<String>> = HashMap::from([
            (
                "abs_tolerance".to_string(),
                vec![
                    "abs_tol".to_string(),
                    "abs_error".to_string(),
                    "absolute_error".to_string(),
                    "absolute_tolerance".to_string(),
                ],
            ),
            (
                "max_iterations".to_string(),
                vec![
                    "max_iter".to_string(),
                    "max_iterations".to_string(),
                    "max_iterations_number".to_string(),
                ],
            ),
        ]);

        parser.with_pseudonims(Some(headers_pseudonims), Some(field_name_pseudonims));
        let _ = parser.parse_document();
        let result: DocumentMap = parser
            .get_result()
            .ok_or("No result after parsing")?
            .clone();
        let spec = parse_bvp_damped_solver_settings_from_document(&result)
            .map_err(|e| e.to_string())?;
        self.apply_bvp_damped_solver_settings(&spec);
        Ok(())
    }
}

/// Creates a configuration file template with all available options and comments
///
/// Generates a structured template showing all configuration sections and their expected format.
/// Users can copy this template and fill in their specific values.
pub fn create_template_file(path: Option<std::path::PathBuf>) {
    let form = r#"
    #####################ADVANCED SETTINGS#######################
    // BVP Solver Configuration Template
    solver_settings
    // Discretization scheme - "forward" or "trapezoid"
    scheme: forward
    // Matrix backend - "Dense" or "Sparse"
    method: Dense
    // Solver strategy - "Damped", "Naive", or "Frozen"
    strategy: Damped
    // Linear system method (optional) - None or specific method
    linear_sys_method: None
    // Absolute convergence tolerance
    abs_tolerance: 1e-6
    // Maximum solver iterations
    max_iterations: 100
    // Logging level (optional) - None or Some(info/warn/error)
    loglevel: Some(info)
    // Disable log file saving
     dont_save_log: true
    // Solution bounds for each variable (variable_name: min, max)
    bounds
    // var1: -10.0, 10.0
    // var2: -5.0, 5.0
    
    // Relative tolerance for each variable
    rel_tolerance
    // var1: 1e-4
    // var2: 1e-4
   
    // Strategy-specific parameters
    strategy_params
        // Maximum iterations with old Jacobian (optional)
        max_jac: Some(3)
        // Maximum damped iterations (optional)
        max_damp_iter: Some(5)
        // Damping factor reduction (optional)
        damp_factor: Some(0.5)
        
    // Adaptive grid refinement settings (optional)
    adaptive_strategy
        // Refinement version
        version: 1
        // Maximum refinement iterations
        max_refinements: 3
        
    // Grid refinement method and parameters
    grid_refinement
        // Available methods:
        // doubleoints: []
        // easy: [parameter]
        // grcarsmooke: [param1, param2, param3]
        // pearson: [param1, param2]
        // twopnt: [param1, param2, param3]
        pearson: [0.1, 1.5]
    
    // Postprocessing options
    postprocessing
        // Enable plotting with plotters crate
        plot: false
        // Enable gnuplot output
        gnuplot: false
        // Save results to text file
        save: false
        // Save results to CSV file
        save_to_csv: false
        filename: somename

    "#;

    use std::env;
    use std::fs::File;
    use std::io::Write;

    let file_path = match path {
        Some(p) => p,
        None => {
            let mut default_path =
                env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            default_path.push("bvp_config_template.txt");
            default_path
        }
    };

    match File::create(&file_path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(form.as_bytes()) {
                eprintln!("Failed to write template file: {}", e);
            } else {
                println!("Template file created at: {:?}", file_path);
            }
        }
        Err(e) => {
            eprintln!("Failed to create template file: {}", e);
        }
    }
}
#[cfg(test)]
mod tests {
    //! Comprehensive tests for BVP configuration parsing
    //!
    //! Tests cover:
    //! - Basic configuration parsing without bounds
    //! - Full configuration with bounds and tolerances
    //! - File-based configuration loading
    //! - Complex settings with adaptive grid and pseudonyms
    //! - Postprocessing configuration and execution

    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        AdaptiveGridConfig, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotBuildProfile, AotExecutionPolicy,
    };
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::somelinalg::banded::LinearSolverPolicy;
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
    use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    use super::*;
    use tempfile::tempdir;

    use nalgebra::{DMatrix, DVector};

    fn parse_document_for_damped(text: &str) -> DocumentMap {
        let mut parser = DocumentParser::new(text.to_owned());
        let _ = parser.parse_document();
        parser.keys_to_lower_case(Some(vec![
            "bounds".to_string(),
            "rel_tolerance".to_string(),
        ]));
        parser.get_result().expect("parser produced no result").clone()
    }

    #[test]
    fn test_BVP_with_setting_parsing_no_bounds() {
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        // also checks if keys_to_lower_case() works
        let input = "
        solver_settings
        scheme: forward
        // THIS IS COMMENT
        METHOD: Dense
        strategy: Damped
        linear_sys_method: None
        abs_tolerance: 1e-5
        # THIS IS COMMENT
        max_iterations: 100
        loglevel: Some(info)
        ";
        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
        let Bounds = HashMap::from([
            ("z".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-7.0, 7.0)),
        ]);
        let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();

        nr.parse_settings_from_str_with_exact_names(input);
        nr.rel_tolerance = Some(rel_tolerance);
        nr.Bounds = Some(Bounds);
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.dont_save_log(true);
        nr.solve();

        let solution = nr.get_result().unwrap();
        let x_mesh = &nr.x_mesh;
        println!("x_mesh = {:?}", x_mesh);
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps + 1);
        nr.gnuplot_result();
        // println!("result = {:?}", solution);
        // nr.plot_result();
    }
    #[test]
    fn test_BVP_with_setting_parsing_with_bounds() {
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let input = "
        solver_settings
        scheme: forward
        method: Dense
        strategy: Damped
        linear_sys_method: None
        abs_tolerance: 1e-5
        max_iterations: 100
        loglevel: Some(info)
        bounds
        z: -10.0, 10.0
        y: -7.0, 7.0
        rel_tolerance
        z: 1e-4
        y: 1e-4
        strategy_params
        max_jac: Some(3)
        max_damp_iter: Some(10)
        damp_factor: Some(0.5)
        adaptive: None
        ";
        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();

        nr.parse_settings_from_str_with_exact_names(input);
        let rel_toleranse = nr.rel_tolerance.clone().unwrap();
        assert_eq!(rel_toleranse["z"], 1e-4);
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.dont_save_log(true);
        nr.solve();

        let solution = nr.get_result().unwrap();
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps + 1);
        // println!("result = {:?}", solution);
        nr.plot_result();
    }

    #[test]
    fn test_BVP_with_setting_from_file() {
        use std::fs::File;
        use std::io::Write;

        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10;

        // Create temporary file with settings
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("problem_config.txt");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "solver_settings").unwrap();
        writeln!(file, "scheme: forward").unwrap();
        writeln!(file, "method: Dense").unwrap();
        writeln!(file, "strategy: Damped").unwrap();
        writeln!(file, "linear_sys_method: None").unwrap();
        writeln!(file, "abs_tolerance: 1e-5").unwrap();
        writeln!(file, "max_iterations: 100").unwrap();
        writeln!(file, "loglevel: Some(info)").unwrap();
        writeln!(file, "bounds").unwrap();
        writeln!(file, "z: -10.0, 10.0").unwrap();
        writeln!(file, "y: -7.0, 7.0").unwrap();
        writeln!(file, "rel_tolerance").unwrap();
        writeln!(file, "z: 1e-4").unwrap();
        writeln!(file, "y: 1e-4").unwrap();
        writeln!(file, "strategy_params").unwrap();
        writeln!(file, "max_jac: Some(3)").unwrap();
        writeln!(file, "max_damp_iter: Some(10)").unwrap();
        writeln!(file, "damp_factor: Some(0.5)").unwrap();
        writeln!(file, "adaptive: None").unwrap();

        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();

        let mut parser = nr.parse_file(Some(file_path)).unwrap();
        nr.parse_settings_with_exact_names(&mut parser).unwrap();

        let rel_tolerance = nr.rel_tolerance.clone().unwrap();
        assert_eq!(rel_tolerance["z"], 1e-4);
        let strategy_params = nr.strategy_params.clone().unwrap();
        assert_eq!(
            strategy_params,
            SolverParams {
                max_jac: Some(3),
                max_damp_iter: Some(10),
                damp_factor: Some(0.5),
                adaptive: None
            }
        );
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.dont_save_log(true);
        nr.solve();

        let solution = nr.get_result().unwrap();
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps + 1);
    }

    #[test]
    fn test_BVP_with_setting_from_file_with_complicated_settings_and_pseudonims_and_postpoc() {
        use std::fs::File;
        use std::io::Write;

        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10;

        // Create temporary file with settings
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("problem_config.txt");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "solver_settings").unwrap();
        writeln!(file, "scheme: forward").unwrap();
        writeln!(file, "method: Dense").unwrap();
        writeln!(file, "strategy: Damped").unwrap();
        writeln!(file, "linear_sys_method: None").unwrap();
        writeln!(file, "absolute_tolerance: 1e-5").unwrap();
        writeln!(file, "max_iterations: 100").unwrap();
        writeln!(file, "loglevel: Some(info)").unwrap();
        writeln!(file, "dont_save_log:true").unwrap();
        writeln!(file, "bounds").unwrap();
        writeln!(file, "z: -10.0, 10.0").unwrap();
        writeln!(file, "y: -7.0, 7.0").unwrap();
        writeln!(file, "rel_tolerance").unwrap();
        writeln!(file, "z: 1e-4").unwrap();
        writeln!(file, "y: 1e-4").unwrap();
        writeln!(file, "strategy_params").unwrap();
        writeln!(file, "max_jac: Some(3)").unwrap();
        writeln!(file, "max_damp_iter: Some(10)").unwrap();
        writeln!(file, "damp_factor: Some(0.5)").unwrap();
        writeln!(file, "adaptive_strategy").unwrap();
        writeln!(file, "version: 1").unwrap();
        writeln!(file, "max_refinements: 1").unwrap();
        writeln!(file, "grid_refinement").unwrap();
        writeln!(file, "pearson: [0.01, 1.5]").unwrap();
        writeln!(file, "postprocessing").unwrap();
        writeln!(file, "gnuplot: true").unwrap();
        writeln!(file, "save_to_csv:true").unwrap();
        writeln!(file, "filename:meow").unwrap();
        //  writeln!(file, "save: true").unwrap();

        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();
        let mut parser = nr.parse_file(Some(file_path)).unwrap();
        nr.parse_settings(&mut parser).unwrap();

        let rel_tolerance = nr.rel_tolerance.clone().unwrap();
        assert_eq!(rel_tolerance["z"], 1e-4);
        let strategy_params = nr.strategy_params.clone().unwrap();
        assert_eq!(
            strategy_params,
            SolverParams {
                max_jac: Some(3),
                max_damp_iter: Some(10),
                damp_factor: Some(0.5),
                adaptive: Some(AdaptiveGridConfig {
                    version: 1,
                    max_refinements: 1,
                    grid_method: GridRefinementMethod::Pearson(0.01, 1.5)
                })
            }
        );
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.solve();
        println!("{:?}", parser.get_result());
        nr.set_postpocessing_from_hashmap(&mut parser);

        // let solution = nr.get_result().unwrap();
        // println!("solution {:?}", solution);
    }

    #[test]
    fn bvp_damped_solver_settings_spec_parses_typed_settings() {
        let input = r#"
        solver_settings
        scheme: forward
        strategy: Damped
        method: Dense
        linear_sys_method: None
        abs_tolerance: 1e-6
        max_iterations: 120
        loglevel: Some(info)
        dont_save_log: false
        bounds
        y: -1.0, 1.0
        rel_tolerance
        y: 1e-4
        strategy_params
        max_jac: Some(3)
        max_damp_iter: Some(10)
        damp_factor: Some(0.5)
        adaptive_strategy
        version: 1
        max_refinements: 2
        grid_refinement
        pearson: [0.1, 1.5]
        "#;
        let result = parse_document_for_damped(input);
        let spec = parse_bvp_damped_solver_settings_from_document(&result).unwrap();
        assert_eq!(spec.scheme, "forward");
        assert_eq!(spec.strategy, "Damped");
        assert_eq!(spec.method, "Dense");
        assert_eq!(spec.linear_sys_method, None);
        assert_eq!(spec.abs_tolerance, 1e-6);
        assert_eq!(spec.max_iterations, 120);
        assert_eq!(spec.loglevel, Some("info".to_string()));
        assert!(!spec.dont_save_log);
        assert_eq!(spec.bounds.as_ref().unwrap()["y"], (-1.0, 1.0));
        assert_eq!(spec.rel_tolerance.as_ref().unwrap()["y"], 1e-4);
        assert_eq!(
            spec.strategy_params,
            Some(SolverParams {
                max_jac: Some(3),
                max_damp_iter: Some(10),
                damp_factor: Some(0.5),
                adaptive: Some(AdaptiveGridConfig {
                    version: 1,
                    max_refinements: 2,
                    grid_method: GridRefinementMethod::Pearson(0.1, 1.5),
                }),
            })
        );
    }

    #[test]
    fn bvp_damped_solver_settings_spec_builds_generated_backend_config() {
        let input = r#"
        solver_settings
        scheme: trapezoid
        strategy: Damped
        method: Banded
        linear_sys_method: faithful
        abs_tolerance: 1e-6
        max_iterations: 120
        generated_backend: banded_aot_tcc
        matrix_backend: banded
        backend_policy: prefer_aot_then_lambdify
        symbolic_backend: AtomView
        aot_codegen_backend: C
        aot_c_compiler: tcc
        aot_build_policy: build_if_missing
        aot_build_profile: release
        aot_compile_preset: dev_fastest
        aot_execution_policy: sequential
        banded_linear_solver: faithful
        refinement_steps: 0
        "#;
        let result = parse_document_for_damped(input);
        let spec = parse_bvp_damped_solver_settings_from_document(&result).unwrap();
        let options = build_bvp_damped_solver_options_from_spec(&spec).unwrap();

        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            options.generated_backend_config.matrix_backend_override,
            Some(MatrixBackend::Banded)
        );
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            options.generated_backend_config.aot_codegen_backend,
            AotCodegenBackend::C
        );
        assert_eq!(
            options.generated_backend_config.aot_c_compiler.as_deref(),
            Some("tcc")
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
        assert_eq!(
            options.generated_backend_config.aot_execution_policy,
            AotExecutionPolicy::SequentialOnly
        );
        assert_eq!(
            options.generated_backend_config.banded_linear_solver_config.policy,
            LinearSolverPolicy::ForceBanded
        );
        assert_eq!(
            options.generated_backend_config
                .banded_linear_solver_config
                .iterative_refinement_steps,
            0
        );
    }

    #[test]
    fn bvp_damped_try_apply_solver_settings_populates_generated_backend_config() {
        let input = r#"
        solver_settings
        scheme: forward
        strategy: Damped
        method: Sparse
        linear_sys_method: None
        abs_tolerance: 1e-6
        max_iterations: 100
        generated_backend: sparse_aot_tcc
        backend_policy: prefer_aot_then_lambdify
        symbolic_backend: AtomView
        aot_codegen_backend: C
        aot_c_compiler: tcc
        aot_build_policy: build_if_missing
        aot_build_profile: release
        aot_execution_policy: sequential
        "#;
        let result = parse_document_for_damped(input);
        let spec = parse_bvp_damped_solver_settings_from_document(&result).unwrap();

        let mut nr = NRBVP::default();
        nr.try_apply_bvp_damped_solver_settings(&spec).unwrap();

        assert_eq!(
            nr.generated_backend_config().backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            nr.generated_backend_config().symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            nr.generated_backend_config().aot_codegen_backend,
            AotCodegenBackend::C
        );
        assert_eq!(
            nr.generated_backend_config().aot_c_compiler.as_deref(),
            Some("tcc")
        );
        assert_eq!(
            nr.generated_backend_config().aot_build_policy,
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
        assert_eq!(
            nr.generated_backend_config().aot_execution_policy,
            AotExecutionPolicy::SequentialOnly
        );
    }

    #[test]
    fn bvp_damped_postprocessing_spec_defaults_when_section_missing() {
        let result = parse_document_for_damped(
            r#"
            solver_settings
            scheme: forward
            strategy: Damped
            method: Dense
            linear_sys_method: None
            abs_tolerance: 1e-6
            max_iterations: 100
            "#,
        );
        let spec = parse_bvp_damped_postprocessing_from_document(&result).unwrap();
        assert!(!spec.plot);
        assert!(!spec.gnuplot);
        assert!(!spec.save);
        assert!(!spec.save_to_csv);
        assert_eq!(spec.filename, None);
    }

    #[test]
    fn bvp_damped_full_task_spec_parses_settings_and_postprocessing() {
        let result = parse_document_for_damped(
            r#"
            solver_settings
            scheme: trapezoid
            strategy: Frozen
            method: Sparse
            linear_sys_method: Some(faithful)
            abs_tolerance: 1e-7
            max_iterations: 42
            loglevel: Some(info)
            dont_save_log: false

            postprocessing
            plot: true
            gnuplot: true
            save: true
            save_to_csv: true
            filename: damped_task_output
            "#,
        );

        let spec = parse_bvp_damped_task_from_document(&result).unwrap();
        assert_eq!(spec.solver_settings.scheme, "trapezoid");
        assert_eq!(spec.solver_settings.strategy, "Frozen");
        assert_eq!(spec.solver_settings.method, "Sparse");
        assert_eq!(spec.solver_settings.linear_sys_method.as_deref(), Some("faithful"));
        assert_eq!(spec.solver_settings.abs_tolerance, 1e-7);
        assert_eq!(spec.solver_settings.max_iterations, 42);
        assert_eq!(spec.solver_settings.loglevel.as_deref(), Some("info"));
        assert!(!spec.solver_settings.dont_save_log);
        assert!(spec.postprocessing.plot);
        assert!(spec.postprocessing.gnuplot);
        assert!(spec.postprocessing.save);
        assert!(spec.postprocessing.save_to_csv);
        assert_eq!(spec.postprocessing.filename.as_deref(), Some("damped_task_output"));

        let mut nr = NRBVP::default();
        nr.apply_bvp_damped_solver_settings(&spec.solver_settings);
        assert_eq!(nr.scheme, "trapezoid");
        assert_eq!(nr.strategy, "Frozen");
        assert_eq!(nr.method, "Sparse");
        assert_eq!(nr.linear_sys_method.as_deref(), Some("faithful"));
    }

    #[test]
    fn bvp_damped_full_task_spec_parses_from_str() {
        let input = r#"
        solver_settings
        scheme: forward
        strategy: Damped
        method: Dense
        linear_sys_method: None
        abs_tolerance: 1e-6
        max_iterations: 100
        loglevel: Some(info)

        postprocessing
        plot: false
        gnuplot: false
        save: true
        save_to_csv: false
        filename: damped_task.txt
        "#;

        let spec = parse_bvp_damped_task_from_str(input).unwrap();
        assert_eq!(spec.solver_settings.scheme, "forward");
        assert_eq!(spec.solver_settings.strategy, "Damped");
        assert_eq!(spec.solver_settings.method, "Dense");
        assert_eq!(spec.solver_settings.linear_sys_method, None);
        assert!(spec.postprocessing.save);
        assert_eq!(spec.postprocessing.filename.as_deref(), Some("damped_task.txt"));
    }

    #[test]
    fn bvp_damped_solver_settings_rejects_unknown_grid_refinement_method() {
        let result = parse_document_for_damped(
            r#"
            solver_settings
            scheme: forward
            strategy: Damped
            method: Dense
            linear_sys_method: None
            abs_tolerance: 1e-6
            max_iterations: 100
            strategy_params
            max_jac: Some(3)
            max_damp_iter: Some(10)
            damp_factor: Some(0.5)
            adaptive_strategy
            version: 1
            max_refinements: 2
            grid_refinement
            alien: [0.1, 1.5]
            "#,
        );
        let err = parse_bvp_damped_solver_settings_from_document(&result).unwrap_err();
        assert_eq!(
            err,
            BvpDampedTaskError::UnknownGridRefinementMethod("alien".to_string())
        );
    }
}
