//! Solver-settings-only adapter for applications that build their own BVP model.
//!
//! This parser accepts the existing `DocumentMap` representation, but only
//! solver-related sections. It never reads equations, mesh, boundary data or an
//! initial guess. The returned `BvpSolverSettings` keeps native family options;
//! the application attaches its own guess and constructs `BvpProblem`.

use crate::command_interpreter::task_parser::{DocumentMap, Value};
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotChunkingPolicy, AotExecutionPolicy, GeneratedBackendConfig,
};
use crate::numerical::BVP_Damp::task_parser_damped::{
    BvpDampedGeneratedBackendSpec, BvpDampedTaskError,
    parse_bvp_damped_solver_settings_from_document,
};
use crate::numerical::BVP_Damp::telemetry::BvpLoggingMode;
use crate::numerical::BVP_api::{
    BvpApiTelemetryMode, BvpSciSparseFactorization, BvpSettingsError, BvpSolverSettings,
};
use crate::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciExecutionPolicy, BvpSciMatrixLayout, BvpSciOptions,
    BvpSciOutputPolicy,
};
use crate::numerical::logging::SolverLogLevel;
use crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy;
use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_orchestrator::{
    ParallelExecutorConfig, ParallelFallbackPolicy,
};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::BandedChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotOptimizationLevel;
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Solver family selected independently from its nonlinear strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridBvpSolverFamily {
    Damped,
    Frozen,
    Sci,
}

/// Errors from parsing hybrid BVP solver settings.
#[derive(Debug, Error)]
pub enum HybridBvpSettingsError {
    #[error("missing `solver_settings` section")]
    MissingSettings,
    #[error("invalid field `{path}`: {message}")]
    InvalidField { path: String, message: String },
    #[error("unsupported field `{path}` for {solver}")]
    UnsupportedField { path: String, solver: &'static str },
    #[error("invalid native Damped settings: {0}")]
    Damped(#[from] BvpDampedTaskError),
    #[error(transparent)]
    Telemetry(#[from] BvpSettingsError),
}

type Section = HashMap<String, Option<Vec<Value>>>;

/// Parse a document containing only solver-settings sections into native
/// BVP_api options. `solver_settings.solver_family` selects the route; the
/// separate `strategy` field is validated only for Damped/Frozen families.
pub fn parse_hybrid_bvp_solver_settings_from_document(
    document: &DocumentMap,
) -> Result<BvpSolverSettings, HybridBvpSettingsError> {
    validate_sections(document)?;
    let settings = document
        .get("solver_settings")
        .ok_or(HybridBvpSettingsError::MissingSettings)?;
    validate_scalar_cardinality(settings)?;
    let family = parse_family(settings)?;
    let strategy = optional_string(settings, "strategy")?;
    let telemetry = optional_telemetry(settings)?;

    let parsed = match family {
        HybridBvpSolverFamily::Damped => {
            if strategy.as_deref().unwrap_or("damped").to_ascii_lowercase() != "damped" {
                return Err(invalid(
                    "solver_settings.strategy",
                    "Damped family requires strategy `damped`",
                ));
            }
            ensure_known_fields(
                settings,
                "BVP_Damp Damped",
                &[
                    "solver_family",
                    "strategy",
                    "scheme",
                    "linear_sys_method",
                    "method",
                    "abs_tolerance",
                    "max_iterations",
                    "loglevel",
                    "generated_backend",
                    "matrix_backend",
                    "backend_policy",
                    "symbolic_backend",
                    "aot_codegen_backend",
                    "aot_c_compiler",
                    "aot_build_policy",
                    "aot_build_profile",
                    "aot_compile_preset",
                    "aot_execution_policy",
                    "banded_linear_solver",
                    "refinement_steps",
                    "aot_compile_optimization",
                    "aot_codegen_units",
                    "aot_residual_chunking",
                    "aot_residual_chunk_size",
                    "aot_sparse_jacobian_chunking",
                    "aot_sparse_jacobian_chunk_size",
                    "aot_jobs_per_worker",
                    "aot_max_residual_jobs",
                    "aot_max_sparse_jobs",
                    "aot_fallback_policy",
                    "aot_min_residual_outputs",
                    "aot_min_sparse_values",
                    "atom_optimization_profile",
                    "lambdify_execution_policy",
                    "lambdify_min_work",
                    "expr_indexed_lambdify",
                    "expr_dependency_columns",
                    "bvp_logging_mode",
                    "bvp_logging_max_events",
                    "telemetry",
                ],
            )?;
            let mut native_document = document.clone();
            let native_settings = native_document
                .get_mut("solver_settings")
                .expect("validated solver settings section");
            native_settings.remove("solver_family");
            native_settings.remove("telemetry");
            native_settings.remove("aot_execution_policy");
            let spec = parse_bvp_damped_solver_settings_from_document(&native_document)?;
            let mut options = spec.build_solver_options()?;
            apply_generated_backend_extensions(settings, &mut options.generated_backend_config)?;
            BvpSolverSettings::Damped(options)
        }
        HybridBvpSolverFamily::Frozen => {
            let strategy = strategy.unwrap_or_else(|| "frozen".to_string());
            let strategy = match strategy.trim().to_ascii_lowercase().as_str() {
                "frozen" => "Frozen",
                "naive" => "Naive",
                _ => {
                    return Err(invalid(
                        "solver_settings.strategy",
                        "Frozen family supports only `frozen` or `naive`",
                    ));
                }
            };
            ensure_known_fields(
                settings,
                "BVP_Damp Frozen/Naive",
                &[
                    "solver_family",
                    "strategy",
                    "scheme",
                    "linear_sys_method",
                    "method",
                    "tolerance",
                    "max_iterations",
                    "loglevel",
                    "generated_backend",
                    "matrix_backend",
                    "backend_policy",
                    "symbolic_backend",
                    "aot_codegen_backend",
                    "aot_c_compiler",
                    "aot_build_policy",
                    "aot_build_profile",
                    "aot_compile_preset",
                    "aot_execution_policy",
                    "banded_linear_solver",
                    "refinement_steps",
                    "aot_compile_optimization",
                    "aot_codegen_units",
                    "aot_residual_chunking",
                    "aot_residual_chunk_size",
                    "aot_sparse_jacobian_chunking",
                    "aot_sparse_jacobian_chunk_size",
                    "aot_jobs_per_worker",
                    "aot_max_residual_jobs",
                    "aot_max_sparse_jobs",
                    "aot_fallback_policy",
                    "aot_min_residual_outputs",
                    "aot_min_sparse_values",
                    "atom_optimization_profile",
                    "lambdify_execution_policy",
                    "lambdify_min_work",
                    "expr_indexed_lambdify",
                    "expr_dependency_columns",
                    "bvp_logging_mode",
                    "bvp_logging_max_events",
                    "telemetry",
                ],
            )?;
            reject_sections_for_frozen(document)?;
            let method = optional_string(settings, "method")?.unwrap_or_else(|| "Sparse".into());
            let generated =
                generated_backend_spec(settings)?.build_generated_backend_config(&method)?;
            let mut generated = generated;
            apply_generated_backend_extensions(settings, &mut generated)?;
            let mut options = FrozenSolverOptions::new(
                strategy.to_string(),
                parse_strategy_params(document.get("strategy_params"))?,
                optional_string(settings, "linear_sys_method")?,
                method,
                optional_float(settings, "tolerance")?.unwrap_or(1e-6),
                optional_usize(settings, "max_iterations")?.unwrap_or(50),
            )
            .with_scheme_name(
                optional_string(settings, "scheme")?.unwrap_or_else(|| "forward".into()),
            )
            .with_generated_backend_config(generated);
            if let Some(level) = optional_string(settings, "loglevel")? {
                options = options.with_logging_level(parse_log_level(&level)?);
            }
            BvpSolverSettings::Frozen(options)
        }
        HybridBvpSolverFamily::Sci => {
            if strategy.is_some() {
                return Err(HybridBvpSettingsError::UnsupportedField {
                    path: "solver_settings.strategy".into(),
                    solver: "BVP_sci (its collocation strategy is fixed)",
                });
            }
            ensure_known_fields(
                settings,
                "BVP_sci",
                &[
                    "solver_family",
                    "execution",
                    "assembly",
                    "matrix_layout",
                    "banded_lower",
                    "banded_upper",
                    "tolerance",
                    "boundary_tolerance",
                    "max_nodes",
                    "max_newton_iterations",
                    "max_jacobian_refreshes",
                    "max_mesh_refinements",
                    "max_backtracking_steps",
                    "execution_policy",
                    "execution_min_work",
                    "output_policy",
                    "sparse_factorization",
                    "loglevel",
                    "telemetry",
                    "aot_build_policy",
                    "aot_build_profile",
                    "aot_codegen_backend",
                    "aot_c_compiler",
                    "aot_output_dir",
                    "aot_handoff_path",
                    "aot_crate_name",
                    "aot_module_name",
                    "aot_residual_chunking",
                    "aot_residual_chunk_size",
                    "aot_sparse_jacobian_chunking",
                    "aot_sparse_jacobian_chunk_size",
                    "aot_banded_jacobian_chunking",
                    "aot_banded_jacobian_chunk_size",
                ],
            )?;
            reject_non_sci_sections(document)?;
            let execution = match optional_string(settings, "execution")?
                .as_deref()
                .unwrap_or("lambdify")
                .to_ascii_lowercase()
                .as_str()
            {
                "lambdify" => BvpSciExecution::Lambdify,
                "aot" => BvpSciExecution::Aot,
                "numerical" | "callbacks" => {
                    return Err(HybridBvpSettingsError::UnsupportedField {
                        path: "solver_settings.execution".into(),
                        solver: "BVP_api symbolic routes (numerical callbacks use native BVP_sci)",
                    });
                }
                _ => {
                    return Err(invalid(
                        "solver_settings.execution",
                        "expected lambdify or aot",
                    ));
                }
            };
            let assembly = match optional_string(settings, "assembly")?
                .as_deref()
                .unwrap_or("expr_legacy")
                .to_ascii_lowercase()
                .as_str()
            {
                "expr" | "expr_legacy" | "legacy" => BvpSciAssembly::ExprLegacy,
                "atom" | "atom_view" | "atomview" | "atom_view_native" => {
                    BvpSciAssembly::AtomViewNative
                }
                _ => {
                    return Err(invalid(
                        "solver_settings.assembly",
                        "expected expr_legacy or atom_view_native",
                    ));
                }
            };
            let layout = parse_matrix_layout(settings)?;
            if !matches!(layout, BvpSciMatrixLayout::Banded { .. })
                && (settings.contains_key("banded_lower") || settings.contains_key("banded_upper"))
            {
                return Err(invalid(
                    "solver_settings.banded_lower/banded_upper",
                    "band widths apply only when matrix_layout is `banded`",
                ));
            }
            if settings.contains_key("execution_min_work")
                && !settings.contains_key("execution_policy")
            {
                return Err(invalid(
                    "solver_settings.execution_min_work",
                    "requires an explicit execution_policy",
                ));
            }
            let mut options = BvpSciOptions::default()
                .with_execution(execution)
                .with_assembly(assembly)
                .with_matrix_layout(layout)
                .with_tolerance(optional_float(settings, "tolerance")?.unwrap_or(1e-3))
                .with_limits(
                    optional_usize(settings, "max_nodes")?.unwrap_or(1000),
                    optional_usize(settings, "max_newton_iterations")?.unwrap_or(8),
                    optional_usize(settings, "max_mesh_refinements")?.unwrap_or(10),
                )
                .with_backtracking_limit(
                    optional_usize(settings, "max_backtracking_steps")?.unwrap_or(4),
                )
                .with_execution_policy(parse_execution_policy(settings)?)
                .with_output_policy(parse_output_policy(settings)?);
            options.max_jacobian_refreshes =
                optional_usize(settings, "max_jacobian_refreshes")?.unwrap_or(4);
            if let Some(tolerance) = optional_float(settings, "boundary_tolerance")? {
                options = options.with_boundary_tolerance(tolerance);
            }
            if let Some(level) = optional_string(settings, "loglevel")? {
                options = options.with_logging_level(parse_log_level(&level)?);
            }
            let telemetry = telemetry.unwrap_or(BvpApiTelemetryMode::Off);
            options.telemetry = match telemetry {
                BvpApiTelemetryMode::Off => {
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Off
                }
                BvpApiTelemetryMode::Counters => {
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Counters
                }
                BvpApiTelemetryMode::Timings => {
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Timings
                }
                BvpApiTelemetryMode::Detailed => {
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Detailed
                }
            };
            let sparse_factorization = match optional_string(settings, "sparse_factorization")?
                .as_deref()
                .unwrap_or("standard_faer")
                .to_ascii_lowercase()
                .as_str()
            {
                "standard_faer" | "faer" => BvpSciSparseFactorization::StandardFaer,
                "superlu" => BvpSciSparseFactorization::SuperLu,
                _ => {
                    return Err(invalid(
                        "solver_settings.sparse_factorization",
                        "supported values are standard_faer and superlu",
                    ));
                }
            };
            let aot_config = if execution == BvpSciExecution::Aot {
                let mut config = parse_aot_config(settings)?;
                config.residual_chunking_strategy =
                    parse_residual_chunking(settings)?.unwrap_or(config.residual_chunking_strategy);
                config.sparse_jacobian_chunking_strategy = parse_sparse_chunking(settings)?
                    .unwrap_or(config.sparse_jacobian_chunking_strategy);
                config.banded_jacobian_chunking_strategy = parse_banded_chunking(settings)?
                    .unwrap_or(config.banded_jacobian_chunking_strategy);
                Some(config)
            } else {
                reject_aot_options_without_aot(settings)?;
                None
            };
            BvpSolverSettings::Sci {
                options,
                sparse_factorization,
                aot_config,
            }
        }
    };

    if family != HybridBvpSolverFamily::Sci {
        parsed
            .with_telemetry_mode(telemetry.unwrap_or(BvpApiTelemetryMode::Off))
            .map_err(Into::into)
    } else {
        Ok(parsed)
    }
}

fn validate_sections(document: &DocumentMap) -> Result<(), HybridBvpSettingsError> {
    const ALLOWED: &[&str] = &[
        "solver_settings",
        "strategy_params",
        "adaptive_strategy",
        "grid_refinement",
        "bounds",
        "rel_tolerance",
    ];
    if let Some(section) = document
        .keys()
        .find(|name| !ALLOWED.contains(&name.as_str()))
    {
        return Err(HybridBvpSettingsError::UnsupportedField {
            path: section.clone(),
            solver: "hybrid settings parser (problem/task sections are caller-owned)",
        });
    }
    Ok(())
}

fn parse_family(settings: &Section) -> Result<HybridBvpSolverFamily, HybridBvpSettingsError> {
    match required_string(settings, "solver_family")?
        .to_ascii_lowercase()
        .as_str()
    {
        "damped" | "bvp_damp" => Ok(HybridBvpSolverFamily::Damped),
        "frozen" => Ok(HybridBvpSolverFamily::Frozen),
        "sci" | "bvp_sci" => Ok(HybridBvpSolverFamily::Sci),
        _ => Err(invalid(
            "solver_settings.solver_family",
            "expected damped, frozen, or sci",
        )),
    }
}

fn ensure_known_fields(
    section: &Section,
    solver: &'static str,
    allowed: &[&str],
) -> Result<(), HybridBvpSettingsError> {
    if let Some(field) = section
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(HybridBvpSettingsError::UnsupportedField {
            path: format!("solver_settings.{field}"),
            solver,
        });
    }
    Ok(())
}

fn validate_scalar_cardinality(section: &Section) -> Result<(), HybridBvpSettingsError> {
    if let Some((field, _)) = section
        .iter()
        .find(|(_, values)| values.as_ref().is_some_and(|values| values.len() != 1))
    {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "expected exactly one scalar value",
        ));
    }
    Ok(())
}

fn reject_sections_for_frozen(document: &DocumentMap) -> Result<(), HybridBvpSettingsError> {
    for section in [
        "bounds",
        "rel_tolerance",
        "adaptive_strategy",
        "grid_refinement",
    ] {
        if document.contains_key(section) {
            return Err(HybridBvpSettingsError::UnsupportedField {
                path: section.into(),
                solver: "BVP_Damp Frozen/Naive",
            });
        }
    }
    Ok(())
}

fn reject_non_sci_sections(document: &DocumentMap) -> Result<(), HybridBvpSettingsError> {
    for section in [
        "strategy_params",
        "adaptive_strategy",
        "grid_refinement",
        "bounds",
        "rel_tolerance",
    ] {
        if document.contains_key(section) {
            return Err(HybridBvpSettingsError::UnsupportedField {
                path: section.into(),
                solver: "BVP_sci",
            });
        }
    }
    Ok(())
}

fn required_string(section: &Section, field: &str) -> Result<String, HybridBvpSettingsError> {
    optional_string(section, field)?.ok_or_else(|| {
        invalid(
            &format!("solver_settings.{field}"),
            "required string is missing",
        )
    })
}

fn optional_string(
    section: &Section,
    field: &str,
) -> Result<Option<String>, HybridBvpSettingsError> {
    let Some(values) = section.get(field) else {
        return Ok(None);
    };
    let Some(values) = values else {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "value is missing",
        ));
    };
    if values.len() != 1 {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "expected exactly one value",
        ));
    }
    match &values[0] {
        Value::String(value) => Ok(Some(value.clone())),
        Value::Optional(Some(value)) => value
            .as_string()
            .cloned()
            .map(Some)
            .ok_or_else(|| invalid(&format!("solver_settings.{field}"), "expected string")),
        Value::Optional(None) => Ok(None),
        _ => Err(invalid(
            &format!("solver_settings.{field}"),
            "expected string",
        )),
    }
}

fn optional_float(section: &Section, field: &str) -> Result<Option<f64>, HybridBvpSettingsError> {
    let Some(values) = section.get(field) else {
        return Ok(None);
    };
    let Some(values) = values else {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "value is missing",
        ));
    };
    if values.len() != 1 {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "expected exactly one value",
        ));
    }
    fn decode(value: &Value) -> Option<Option<f64>> {
        match value {
            Value::Float(number) => Some(Some(*number)),
            Value::Integer(number) => Some(Some(*number as f64)),
            Value::Optional(Some(inner)) => decode(inner),
            Value::Optional(None) => Some(None),
            _ => None,
        }
    }
    match decode(&values[0]) {
        Some(Some(value)) if value.is_finite() => Ok(Some(value)),
        Some(None) => Ok(None),
        _ => Err(invalid(
            &format!("solver_settings.{field}"),
            "expected a finite number",
        )),
    }
}

fn optional_usize(section: &Section, field: &str) -> Result<Option<usize>, HybridBvpSettingsError> {
    let Some(values) = section.get(field) else {
        return Ok(None);
    };
    let Some(values) = values else {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "value is missing",
        ));
    };
    if values.len() != 1 {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "expected exactly one value",
        ));
    }
    fn decode(value: &Value) -> Option<Option<usize>> {
        match value {
            Value::Usize(number) => Some(Some(*number)),
            Value::Integer(number) => usize::try_from(*number).ok().map(Some),
            Value::Optional(Some(inner)) => decode(inner),
            Value::Optional(None) => Some(None),
            _ => None,
        }
    }
    match decode(&values[0]) {
        Some(Some(value)) => Ok(Some(value)),
        Some(None) => Ok(None),
        None => Err({
            invalid(
                &format!("solver_settings.{field}"),
                "expected a non-negative integer",
            )
        }),
    }
}

fn optional_bool(section: &Section, field: &str) -> Result<Option<bool>, HybridBvpSettingsError> {
    let Some(values) = section.get(field) else {
        return Ok(None);
    };
    let Some(values) = values else {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "value is missing",
        ));
    };
    if values.len() != 1 {
        return Err(invalid(
            &format!("solver_settings.{field}"),
            "expected exactly one value",
        ));
    }
    match &values[0] {
        Value::Boolean(value) => Ok(Some(*value)),
        Value::Optional(Some(value)) => match value.as_ref() {
            Value::Boolean(value) => Ok(Some(*value)),
            _ => Err(invalid(
                &format!("solver_settings.{field}"),
                "expected a boolean",
            )),
        },
        Value::Optional(None) => Ok(None),
        _ => Err(invalid(
            &format!("solver_settings.{field}"),
            "expected a boolean",
        )),
    }
}

fn optional_telemetry(
    section: &Section,
) -> Result<Option<BvpApiTelemetryMode>, HybridBvpSettingsError> {
    optional_string(section, "telemetry")?
        .map(|value| match value.to_ascii_lowercase().as_str() {
            "off" => Ok(BvpApiTelemetryMode::Off),
            "counters" => Ok(BvpApiTelemetryMode::Counters),
            "timings" => Ok(BvpApiTelemetryMode::Timings),
            "detailed" => Ok(BvpApiTelemetryMode::Detailed),
            _ => Err(invalid(
                "solver_settings.telemetry",
                "expected off, counters, timings, or detailed",
            )),
        })
        .transpose()
}

fn generated_backend_spec(
    section: &Section,
) -> Result<BvpDampedGeneratedBackendSpec, HybridBvpSettingsError> {
    Ok(BvpDampedGeneratedBackendSpec {
        preset: optional_string(section, "generated_backend")?,
        matrix_backend: optional_string(section, "matrix_backend")?,
        backend_policy: optional_string(section, "backend_policy")?,
        symbolic_backend: optional_string(section, "symbolic_backend")?,
        aot_codegen_backend: optional_string(section, "aot_codegen_backend")?,
        aot_c_compiler: optional_string(section, "aot_c_compiler")?,
        aot_build_policy: optional_string(section, "aot_build_policy")?,
        aot_build_profile: optional_string(section, "aot_build_profile")?,
        aot_compile_preset: optional_string(section, "aot_compile_preset")?,
        // The hybrid parser builds this typed policy directly so it can also
        // represent ParallelExecutorConfig without losing its thresholds.
        aot_execution_policy: None,
        banded_linear_solver: optional_string(section, "banded_linear_solver")?,
        refinement_steps: optional_usize(section, "refinement_steps")?,
    })
}

fn apply_generated_backend_extensions(
    section: &Section,
    config: &mut GeneratedBackendConfig,
) -> Result<(), HybridBvpSettingsError> {
    if matches!(
        config.backend_policy_override,
        Some(BackendSelectionPolicy::LambdifyOnly | BackendSelectionPolicy::NumericOnly)
    ) {
        const AOT_FIELDS: &[&str] = &[
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_compile_preset",
            "aot_execution_policy",
            "aot_compile_optimization",
            "aot_codegen_units",
            "aot_residual_chunking",
            "aot_residual_chunk_size",
            "aot_sparse_jacobian_chunking",
            "aot_sparse_jacobian_chunk_size",
            "aot_jobs_per_worker",
            "aot_max_residual_jobs",
            "aot_max_sparse_jobs",
            "aot_fallback_policy",
            "aot_min_residual_outputs",
            "aot_min_sparse_values",
            "atom_optimization_profile",
        ];
        if let Some(field) = AOT_FIELDS
            .iter()
            .find(|field| section.contains_key(**field))
        {
            return Err(HybridBvpSettingsError::UnsupportedField {
                path: format!("solver_settings.{field}"),
                solver: "explicit non-AOT Damped/Frozen backend policy",
            });
        }
    }

    if let Some(level) = optional_string(section, "aot_compile_optimization")? {
        let optimization = match level.trim().to_ascii_lowercase().as_str() {
            "default" => AotOptimizationLevel::Default,
            "o0" => AotOptimizationLevel::O0,
            "o1" => AotOptimizationLevel::O1,
            "o2" => AotOptimizationLevel::O2,
            "o3" => AotOptimizationLevel::O3,
            "os" => AotOptimizationLevel::Os,
            "oz" => AotOptimizationLevel::Oz,
            _ => {
                return Err(invalid(
                    "solver_settings.aot_compile_optimization",
                    "expected default, o0, o1, o2, o3, os, or oz",
                ));
            }
        };
        let mut compile = config.aot_compile_config.clone();
        compile.optimization = optimization;
        config.aot_compile_config = compile;
    }
    if let Some(codegen_units) = optional_usize(section, "aot_codegen_units")? {
        if codegen_units == 0 {
            return Err(invalid(
                "solver_settings.aot_codegen_units",
                "must be greater than zero",
            ));
        }
        let mut compile = config.aot_compile_config.clone();
        compile.codegen_units = Some(codegen_units);
        config.aot_compile_config = compile;
    }

    let residual_chunking = parse_residual_chunking(section)?;
    let sparse_chunking = parse_sparse_chunking(section)?;
    if residual_chunking.is_some() || sparse_chunking.is_some() {
        let current = config.aot_chunking_policy;
        config.aot_chunking_policy = AotChunkingPolicy::with_parts(
            residual_chunking.or(current.residual),
            sparse_chunking.or(current.sparse_jacobian),
        );
    }

    if let Some(policy) = optional_string(section, "aot_execution_policy")? {
        let normalized = policy.trim().to_ascii_lowercase();
        let has_parallel_controls = [
            "aot_jobs_per_worker",
            "aot_max_residual_jobs",
            "aot_max_sparse_jobs",
            "aot_fallback_policy",
            "aot_min_residual_outputs",
            "aot_min_sparse_values",
        ]
        .iter()
        .any(|field| section.contains_key(*field));
        if normalized != "parallel" && has_parallel_controls {
            return Err(invalid(
                "solver_settings.aot_execution_policy",
                "parallel executor controls require aot_execution_policy = parallel",
            ));
        }
        let policy = match normalized.as_str() {
            "auto" => AotExecutionPolicy::Auto,
            "sequential" | "sequential_only" | "seq" => AotExecutionPolicy::SequentialOnly,
            "parallel" => {
                let parallel = ParallelExecutorConfig {
                    jobs_per_worker: optional_usize(section, "aot_jobs_per_worker")?.unwrap_or(1),
                    max_residual_jobs: optional_usize(section, "aot_max_residual_jobs")?,
                    max_sparse_jobs: optional_usize(section, "aot_max_sparse_jobs")?,
                    fallback_policy: parse_parallel_fallback(section)?,
                };
                if parallel.jobs_per_worker == 0
                    || parallel.max_residual_jobs == Some(0)
                    || parallel.max_sparse_jobs == Some(0)
                {
                    return Err(invalid(
                        "solver_settings.aot_jobs_per_worker/aot_max_*_jobs",
                        "parallel job counts must be greater than zero",
                    ));
                }
                AotExecutionPolicy::Parallel(parallel)
            }
            _ => {
                return Err(invalid(
                    "solver_settings.aot_execution_policy",
                    "expected auto, sequential, or parallel",
                ));
            }
        };
        config.aot_execution_policy = policy;
    } else if [
        "aot_jobs_per_worker",
        "aot_max_residual_jobs",
        "aot_max_sparse_jobs",
        "aot_fallback_policy",
        "aot_min_residual_outputs",
        "aot_min_sparse_values",
    ]
    .iter()
    .any(|field| section.contains_key(*field))
    {
        return Err(invalid(
            "solver_settings.aot_execution_policy",
            "parallel executor controls require aot_execution_policy = parallel",
        ));
    }

    let lambdify_min_work = optional_usize(section, "lambdify_min_work")?.unwrap_or(0);
    config.lambdify_execution_policy = match optional_string(section, "lambdify_execution_policy")?
    {
        Some(policy) => match policy.trim().to_ascii_lowercase().as_str() {
            "sequential" | "seq" => BvpLambdifyExecutionPolicy::Sequential,
            "parallel" | "par" => BvpLambdifyExecutionPolicy::Parallel {
                min_work: lambdify_min_work,
            },
            "auto" => BvpLambdifyExecutionPolicy::Auto {
                min_work: lambdify_min_work,
            },
            _ => {
                return Err(invalid(
                    "solver_settings.lambdify_execution_policy",
                    "expected sequential, parallel, or auto",
                ));
            }
        },
        None if section.contains_key("lambdify_min_work") => {
            return Err(invalid(
                "solver_settings.lambdify_min_work",
                "requires an explicit lambdify_execution_policy",
            ));
        }
        None => config.lambdify_execution_policy,
    };

    if let Some(profile) = optional_string(section, "atom_optimization_profile")? {
        config.atom_optimization_profile = match profile.trim().to_ascii_lowercase().as_str() {
            "full" => AtomOptimizationProfile::Full,
            "fast_bootstrap" => AtomOptimizationProfile::FastBootstrap,
            "no_cse" => AtomOptimizationProfile::NoCse,
            "no_peephole" => AtomOptimizationProfile::NoPeephole,
            "no_temp_reuse" => AtomOptimizationProfile::NoTempReuse,
            _ => {
                return Err(invalid(
                    "solver_settings.atom_optimization_profile",
                    "expected full, fast_bootstrap, no_cse, no_peephole, or no_temp_reuse",
                ));
            }
        };
    }
    if let Some(enabled) = optional_bool(section, "expr_indexed_lambdify")? {
        config.expr_indexed_lambdify_enabled = enabled;
    }
    if let Some(enabled) = optional_bool(section, "expr_dependency_columns")? {
        config.expr_dependency_columns_enabled = enabled;
    }

    if let Some(mode) = optional_string(section, "bvp_logging_mode")? {
        config.bvp_logging_mode = match mode.trim().to_ascii_lowercase().as_str() {
            "off" => BvpLoggingMode::Off,
            "warnings" | "warn" => BvpLoggingMode::Warnings,
            "detailed" => BvpLoggingMode::Detailed,
            _ => {
                return Err(invalid(
                    "solver_settings.bvp_logging_mode",
                    "expected off, warnings, or detailed",
                ));
            }
        };
    }
    if let Some(max_events) = optional_usize(section, "bvp_logging_max_events")? {
        config.bvp_logging_max_events = max_events;
    }
    Ok(())
}

fn parse_residual_chunking(
    section: &Section,
) -> Result<Option<ResidualChunkingStrategy>, HybridBvpSettingsError> {
    let mode = optional_string(section, "aot_residual_chunking")?;
    let size = optional_usize(section, "aot_residual_chunk_size")?;
    match (mode.as_deref().map(str::to_ascii_lowercase), size) {
        (Some(mode), size) => parse_residual_chunking_mode(&mode, size),
        (None, None) => Ok(None),
        (None, Some(_)) => Err(invalid(
            "solver_settings.aot_residual_chunk_size",
            "requires aot_residual_chunking",
        )),
    }
}

fn parse_residual_chunking_mode(
    mode: &str,
    size: Option<usize>,
) -> Result<Option<ResidualChunkingStrategy>, HybridBvpSettingsError> {
    match (mode, size) {
        ("whole", None) => Ok(Some(ResidualChunkingStrategy::Whole)),
        ("target_chunks", Some(count)) if count > 0 => {
            Ok(Some(ResidualChunkingStrategy::ByTargetChunkCount {
                target_chunks: count,
            }))
        }
        ("output_count", Some(count)) if count > 0 => {
            Ok(Some(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: count,
            }))
        }
        ("target_chunks" | "output_count", Some(_)) => Err(invalid(
            "solver_settings.aot_residual_chunk_size",
            "chunk size/count must be greater than zero",
        )),
        ("whole", Some(_)) => Err(invalid(
            "solver_settings.aot_residual_chunk_size",
            "must be omitted when chunking is whole",
        )),
        (_, None) => Err(invalid(
            "solver_settings.aot_residual_chunk_size",
            "is required for target_chunks or output_count chunking",
        )),
        (_, Some(_)) => Err(invalid(
            "solver_settings.aot_residual_chunking",
            "expected whole, target_chunks, or output_count",
        )),
    }
}

fn parse_sparse_chunking(
    section: &Section,
) -> Result<Option<SparseChunkingStrategy>, HybridBvpSettingsError> {
    let mode = optional_string(section, "aot_sparse_jacobian_chunking")?;
    let size = optional_usize(section, "aot_sparse_jacobian_chunk_size")?;
    match (mode.as_deref().map(str::to_ascii_lowercase), size) {
        (Some(mode), size) => parse_sparse_chunking_mode(&mode, size),
        (None, None) => Ok(None),
        (None, Some(_)) => Err(invalid(
            "solver_settings.aot_sparse_jacobian_chunk_size",
            "requires aot_sparse_jacobian_chunking",
        )),
    }
}

fn parse_sparse_chunking_mode(
    mode: &str,
    size: Option<usize>,
) -> Result<Option<SparseChunkingStrategy>, HybridBvpSettingsError> {
    match (mode, size) {
        ("whole", None) => Ok(Some(SparseChunkingStrategy::Whole)),
        ("target_chunks", Some(count)) if count > 0 => {
            Ok(Some(SparseChunkingStrategy::ByTargetChunkCount {
                target_chunks: count,
            }))
        }
        ("nnz", Some(count)) if count > 0 => Ok(Some(SparseChunkingStrategy::ByNonZeroCount {
            max_entries_per_chunk: count,
        })),
        ("rows", Some(count)) if count > 0 => Ok(Some(SparseChunkingStrategy::ByRowCount {
            rows_per_chunk: count,
        })),
        ("target_chunks" | "nnz" | "rows", Some(_)) => Err(invalid(
            "solver_settings.aot_sparse_jacobian_chunk_size",
            "chunk size/count must be greater than zero",
        )),
        ("whole", Some(_)) => Err(invalid(
            "solver_settings.aot_sparse_jacobian_chunk_size",
            "must be omitted when chunking is whole",
        )),
        (_, None) => Err(invalid(
            "solver_settings.aot_sparse_jacobian_chunk_size",
            "is required for target_chunks, nnz, or rows chunking",
        )),
        (_, Some(_)) => Err(invalid(
            "solver_settings.aot_sparse_jacobian_chunking",
            "expected whole, target_chunks, nnz, or rows",
        )),
    }
}

fn parse_parallel_fallback(
    section: &Section,
) -> Result<ParallelFallbackPolicy, HybridBvpSettingsError> {
    let residual = optional_usize(section, "aot_min_residual_outputs")?;
    let sparse = optional_usize(section, "aot_min_sparse_values")?;
    match optional_string(section, "aot_fallback_policy")?
        .as_deref()
        .unwrap_or("auto")
        .to_ascii_lowercase()
        .as_str()
    {
        "auto" if residual.is_none() && sparse.is_none() => Ok(ParallelFallbackPolicy::Auto),
        "never" if residual.is_none() && sparse.is_none() => Ok(ParallelFallbackPolicy::Never),
        "thresholds" => match (residual, sparse) {
            (Some(min_residual_outputs), Some(min_sparse_values)) => {
                Ok(ParallelFallbackPolicy::Thresholds {
                    min_residual_outputs,
                    min_sparse_values,
                })
            }
            _ => Err(invalid(
                "solver_settings.aot_min_residual_outputs/aot_min_sparse_values",
                "threshold fallback requires both values",
            )),
        },
        "auto" | "never" => Err(invalid(
            "solver_settings.aot_min_residual_outputs/aot_min_sparse_values",
            "threshold values require aot_fallback_policy = thresholds",
        )),
        _ => Err(invalid(
            "solver_settings.aot_fallback_policy",
            "expected auto, never, or thresholds",
        )),
    }
}

fn parse_banded_chunking(
    section: &Section,
) -> Result<Option<BandedChunkingStrategy>, HybridBvpSettingsError> {
    let mode = optional_string(section, "aot_banded_jacobian_chunking")?;
    let size = optional_usize(section, "aot_banded_jacobian_chunk_size")?;
    let Some(mode) = mode else {
        return if size.is_none() {
            Ok(None)
        } else {
            Err(invalid(
                "solver_settings.aot_banded_jacobian_chunk_size",
                "requires aot_banded_jacobian_chunking",
            ))
        };
    };
    match (mode.trim().to_ascii_lowercase().as_str(), size) {
        ("whole", None) => Ok(Some(BandedChunkingStrategy::Whole)),
        ("target_chunks", Some(count)) if count > 0 => {
            Ok(Some(BandedChunkingStrategy::ByTargetChunkCount {
                target_chunks: count,
            }))
        }
        ("diagonals", Some(count)) if count > 0 => {
            Ok(Some(BandedChunkingStrategy::ByDiagonalCount {
                diagonals_per_chunk: count,
            }))
        }
        ("target_chunks" | "diagonals", Some(_)) => Err(invalid(
            "solver_settings.aot_banded_jacobian_chunk_size",
            "chunk size/count must be greater than zero",
        )),
        ("whole", Some(_)) => Err(invalid(
            "solver_settings.aot_banded_jacobian_chunk_size",
            "must be omitted when chunking is whole",
        )),
        (_, None) => Err(invalid(
            "solver_settings.aot_banded_jacobian_chunk_size",
            "is required for target_chunks or diagonals chunking",
        )),
        (_, Some(_)) => Err(invalid(
            "solver_settings.aot_banded_jacobian_chunking",
            "expected whole, target_chunks, or diagonals",
        )),
    }
}

fn parse_strategy_params(
    document: Option<&Section>,
) -> Result<Option<HashMap<String, Option<Vec<f64>>>>, HybridBvpSettingsError> {
    document
        .map(|section| {
            section
                .iter()
                .map(|(name, values)| {
                    let values = values.as_ref().ok_or_else(|| {
                        invalid(&format!("strategy_params.{name}"), "value is missing")
                    })?;
                    let parsed = match values.as_slice() {
                        [Value::Vector(values)] if values.iter().all(|v| v.is_finite()) => {
                            Some(values.clone())
                        }
                        [Value::Optional(Some(value))] => match value.as_ref() {
                            Value::Vector(values) if values.iter().all(|v| v.is_finite()) => {
                                Some(values.clone())
                            }
                            _ => {
                                return Err(invalid(
                                    &format!("strategy_params.{name}"),
                                    "expected a finite vector or None",
                                ));
                            }
                        },
                        [Value::Optional(None)] => None,
                        _ => {
                            return Err(invalid(
                                &format!("strategy_params.{name}"),
                                "expected a finite vector or None",
                            ));
                        }
                    };
                    Ok((name.clone(), parsed))
                })
                .collect::<Result<HashMap<_, _>, HybridBvpSettingsError>>()
        })
        .transpose()
}

fn parse_matrix_layout(section: &Section) -> Result<BvpSciMatrixLayout, HybridBvpSettingsError> {
    match optional_string(section, "matrix_layout")?
        .as_deref()
        .unwrap_or("dense")
        .to_ascii_lowercase()
        .as_str()
    {
        "dense" => Ok(BvpSciMatrixLayout::Dense),
        "sparse" => Ok(BvpSciMatrixLayout::Sparse),
        "banded_auto" | "bandedauto" => Ok(BvpSciMatrixLayout::BandedAuto),
        "banded" => Ok(BvpSciMatrixLayout::Banded {
            lower: optional_usize(section, "banded_lower")?.unwrap_or(1),
            upper: optional_usize(section, "banded_upper")?.unwrap_or(1),
        }),
        _ => Err(invalid(
            "solver_settings.matrix_layout",
            "expected dense, sparse, banded, or banded_auto",
        )),
    }
}

fn parse_execution_policy(
    section: &Section,
) -> Result<BvpSciExecutionPolicy, HybridBvpSettingsError> {
    let min_work = optional_usize(section, "execution_min_work")?.unwrap_or(0);
    match optional_string(section, "execution_policy")?
        .as_deref()
        .unwrap_or("sequential")
        .to_ascii_lowercase()
        .as_str()
    {
        "sequential" | "seq" => Ok(BvpLambdifyExecutionPolicy::Sequential),
        "parallel" | "par" => Ok(BvpLambdifyExecutionPolicy::Parallel { min_work }),
        "auto" => Ok(BvpLambdifyExecutionPolicy::Auto { min_work }),
        _ => Err(invalid(
            "solver_settings.execution_policy",
            "expected sequential, parallel, or auto",
        )),
    }
}

fn parse_output_policy(section: &Section) -> Result<BvpSciOutputPolicy, HybridBvpSettingsError> {
    match optional_string(section, "output_policy")?
        .as_deref()
        .unwrap_or("mesh_and_values")
        .to_ascii_lowercase()
        .as_str()
    {
        "final_only" | "finalonly" => Ok(BvpSciOutputPolicy::FinalOnly),
        "mesh_and_values" | "meshandvalues" => Ok(BvpSciOutputPolicy::MeshAndValues),
        "dense_output" | "denseoutput" => Ok(BvpSciOutputPolicy::DenseOutput),
        _ => Err(invalid(
            "solver_settings.output_policy",
            "expected final_only, mesh_and_values, or dense_output",
        )),
    }
}

fn parse_log_level(raw: &str) -> Result<SolverLogLevel, HybridBvpSettingsError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Ok(SolverLogLevel::Off),
        "error" => Ok(SolverLogLevel::Error),
        "warn" | "warning" => Ok(SolverLogLevel::Warn),
        "info" => Ok(SolverLogLevel::Info),
        "debug" => Ok(SolverLogLevel::Debug),
        _ => Err(invalid(
            "solver_settings.loglevel",
            "expected off, error, warn, info, or debug",
        )),
    }
}

fn parse_aot_config(
    section: &Section,
) -> Result<SymbolicIvpGeneratedBackendConfig, HybridBvpSettingsError> {
    let profile = match optional_string(section, "aot_build_profile")?
        .as_deref()
        .unwrap_or("release")
        .to_ascii_lowercase()
        .as_str()
    {
        "release" => AotBuildProfile::Release,
        "debug" => AotBuildProfile::Debug,
        _ => {
            return Err(invalid(
                "solver_settings.aot_build_profile",
                "expected release or debug",
            ));
        }
    };
    let raw_policy = optional_string(section, "aot_build_policy")?;
    let policy = match raw_policy
        .as_deref()
        .unwrap_or("use_if_available")
        .to_ascii_lowercase()
        .as_str()
    {
        "use_if_available" | "use" | "auto" => SymbolicIvpAotBuildPolicy::UseIfAvailable,
        "require_prebuilt" | "require" => SymbolicIvpAotBuildPolicy::RequirePrebuilt,
        "build_if_missing" | "build" => SymbolicIvpAotBuildPolicy::BuildIfMissing { profile },
        "rebuild_always" | "rebuild" => SymbolicIvpAotBuildPolicy::RebuildAlways { profile },
        _ => {
            return Err(invalid(
                "solver_settings.aot_build_policy",
                "unknown AOT build policy",
            ));
        }
    };
    if section.contains_key("aot_build_profile")
        && !matches!(
            policy,
            SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
                | SymbolicIvpAotBuildPolicy::RebuildAlways { .. }
        )
    {
        return Err(invalid(
            "solver_settings.aot_build_profile",
            "is meaningful only with build_if_missing or rebuild_always",
        ));
    }
    let backend = match optional_string(section, "aot_codegen_backend")?
        .as_deref()
        .unwrap_or("c")
        .to_ascii_lowercase()
        .as_str()
    {
        "c" => AotCodegenBackend::C,
        "rust" => AotCodegenBackend::Rust,
        "zig" => AotCodegenBackend::Zig,
        _ => {
            return Err(invalid(
                "solver_settings.aot_codegen_backend",
                "expected c, rust, or zig",
            ));
        }
    };
    if !matches!(backend, AotCodegenBackend::C) && section.contains_key("aot_c_compiler") {
        return Err(invalid(
            "solver_settings.aot_c_compiler",
            "is only meaningful for the C codegen backend",
        ));
    }
    let mut config = SymbolicIvpGeneratedBackendConfig::default()
        .with_build_policy(policy)
        .with_aot_codegen_backend(backend);
    if let Some(compiler) = optional_string(section, "aot_c_compiler")? {
        config = config.with_aot_c_compiler(compiler);
    }
    if let Some(path) = optional_string(section, "aot_output_dir")? {
        config = config.with_output_parent_dir(Some(PathBuf::from(path)));
    }
    if let Some(path) = optional_string(section, "aot_handoff_path")? {
        config.handoff_path = Some(PathBuf::from(path));
    }
    config.crate_name_override = optional_string(section, "aot_crate_name")?;
    config.module_name_override = optional_string(section, "aot_module_name")?;
    Ok(config)
}

fn reject_aot_options_without_aot(section: &Section) -> Result<(), HybridBvpSettingsError> {
    for field in [
        "aot_build_policy",
        "aot_build_profile",
        "aot_codegen_backend",
        "aot_c_compiler",
        "aot_output_dir",
        "aot_handoff_path",
        "aot_crate_name",
        "aot_module_name",
        "aot_residual_chunking",
        "aot_residual_chunk_size",
        "aot_sparse_jacobian_chunking",
        "aot_sparse_jacobian_chunk_size",
        "aot_banded_jacobian_chunking",
        "aot_banded_jacobian_chunk_size",
    ] {
        if section.contains_key(field) {
            return Err(HybridBvpSettingsError::UnsupportedField {
                path: format!("solver_settings.{field}"),
                solver: "BVP_sci Lambdify execution",
            });
        }
    }
    Ok(())
}

fn invalid(path: &str, message: &str) -> HybridBvpSettingsError {
    HybridBvpSettingsError::InvalidField {
        path: path.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(settings: Section) -> DocumentMap {
        HashMap::from([("solver_settings".to_string(), settings)])
    }

    fn values(entries: &[(&str, Value)]) -> Section {
        entries
            .iter()
            .map(|(key, value)| ((*key).into(), Some(vec![value.clone()])))
            .collect()
    }

    fn sci_settings(entries: &[(&str, Value)]) -> Section {
        let mut settings = values(&[
            ("solver_family", Value::String("sci".into())),
            ("execution", Value::String("lambdify".into())),
        ]);
        settings.extend(values(entries));
        settings
    }

    fn parsed_sci(settings: Section) -> (BvpSciOptions, BvpSciSparseFactorization) {
        let BvpSolverSettings::Sci {
            options,
            sparse_factorization,
            ..
        } = parse_hybrid_bvp_solver_settings_from_document(&document(settings))
            .expect("valid Sci settings")
        else {
            panic!("expected Sci settings");
        };
        (options, sparse_factorization)
    }

    #[test]
    fn route_defaults_match_the_documented_native_contracts() {
        let damped = document(values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("damped".into())),
            ("scheme", Value::String("forward".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(1e-6)),
            ("max_iterations", Value::Integer(25)),
        ]));
        let BvpSolverSettings::Damped(damped) =
            parse_hybrid_bvp_solver_settings_from_document(&damped).unwrap()
        else {
            panic!("expected Damped settings");
        };
        assert_eq!(damped.scheme, "forward");
        assert_eq!(damped.max_iterations, 25);
        assert_eq!(
            damped.generated_backend_config.bvp_telemetry_mode,
            crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Off
        );

        let frozen = document(values(&[("solver_family", Value::String("frozen".into()))]));
        let BvpSolverSettings::Frozen(frozen) =
            parse_hybrid_bvp_solver_settings_from_document(&frozen).unwrap()
        else {
            panic!("expected Frozen settings");
        };
        assert_eq!(frozen.strategy, "Frozen");
        assert_eq!(frozen.scheme, "forward");
        assert_eq!(frozen.method, "Sparse");
        assert_eq!(frozen.tolerance, 1e-6);
        assert_eq!(frozen.max_iterations, 50);
        assert_eq!(
            frozen.generated_backend_config.bvp_telemetry_mode,
            crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Off
        );

        let (sci, factorization) =
            parsed_sci(values(&[("solver_family", Value::String("sci".into()))]));
        assert_eq!(sci.execution, BvpSciExecution::Lambdify);
        assert_eq!(sci.assembly, Some(BvpSciAssembly::ExprLegacy));
        assert_eq!(sci.matrix_layout, BvpSciMatrixLayout::Dense);
        assert_eq!(sci.tolerance, 1e-3);
        assert_eq!(sci.boundary_tolerance, None);
        assert_eq!(sci.max_nodes, 1000);
        assert_eq!(sci.max_newton_iterations, 8);
        assert_eq!(sci.max_jacobian_refreshes, 4);
        assert_eq!(sci.max_mesh_refinements, 10);
        assert_eq!(sci.max_backtracking_steps, 4);
        assert_eq!(sci.execution_policy, BvpSciExecutionPolicy::Sequential);
        assert_eq!(sci.output_policy, BvpSciOutputPolicy::MeshAndValues);
        assert_eq!(
            sci.telemetry,
            crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Off
        );
        assert_eq!(factorization, BvpSciSparseFactorization::StandardFaer);
    }

    #[test]
    fn accepted_family_assembly_layout_and_factorization_aliases_are_exhaustive() {
        for family in ["sci", "bvp_sci"] {
            let (options, _) =
                parsed_sci(values(&[("solver_family", Value::String(family.into()))]));
            assert_eq!(options.execution, BvpSciExecution::Lambdify);
        }

        for alias in ["expr", "expr_legacy", "legacy"] {
            let (options, _) =
                parsed_sci(sci_settings(&[("assembly", Value::String(alias.into()))]));
            assert_eq!(
                options.assembly,
                Some(BvpSciAssembly::ExprLegacy),
                "{alias}"
            );
        }
        for alias in ["atom", "atom_view", "atomview", "atom_view_native"] {
            let (options, _) =
                parsed_sci(sci_settings(&[("assembly", Value::String(alias.into()))]));
            assert_eq!(
                options.assembly,
                Some(BvpSciAssembly::AtomViewNative),
                "{alias}"
            );
        }
        for (alias, expected) in [
            ("dense", BvpSciMatrixLayout::Dense),
            ("sparse", BvpSciMatrixLayout::Sparse),
            ("banded_auto", BvpSciMatrixLayout::BandedAuto),
            ("bandedauto", BvpSciMatrixLayout::BandedAuto),
            ("banded", BvpSciMatrixLayout::Banded { lower: 1, upper: 1 }),
        ] {
            let (options, _) = parsed_sci(sci_settings(&[(
                "matrix_layout",
                Value::String(alias.into()),
            )]));
            assert_eq!(options.matrix_layout, expected, "{alias}");
        }
        for alias in ["standard_faer", "faer"] {
            let (_, factorization) = parsed_sci(sci_settings(&[(
                "sparse_factorization",
                Value::String(alias.into()),
            )]));
            assert_eq!(
                factorization,
                BvpSciSparseFactorization::StandardFaer,
                "{alias}"
            );
        }
        let (_, factorization) = parsed_sci(sci_settings(&[(
            "sparse_factorization",
            Value::String("superlu".into()),
        )]));
        assert_eq!(factorization, BvpSciSparseFactorization::SuperLu);

        let damped_alias = document(values(&[
            ("solver_family", Value::String("bvp_damp".into())),
            ("strategy", Value::String("DAMPED".into())),
            ("scheme", Value::String("forward".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(1e-6)),
            ("max_iterations", Value::Integer(25)),
        ]));
        assert!(matches!(
            parse_hybrid_bvp_solver_settings_from_document(&damped_alias),
            Ok(BvpSolverSettings::Damped(_))
        ));
    }

    #[test]
    fn accepted_execution_logging_output_telemetry_and_aot_aliases_are_exhaustive() {
        for (alias, expected) in [
            ("sequential", BvpSciExecutionPolicy::Sequential),
            ("seq", BvpSciExecutionPolicy::Sequential),
            ("parallel", BvpSciExecutionPolicy::Parallel { min_work: 0 }),
            ("par", BvpSciExecutionPolicy::Parallel { min_work: 0 }),
            ("auto", BvpSciExecutionPolicy::Auto { min_work: 0 }),
        ] {
            let (options, _) = parsed_sci(sci_settings(&[(
                "execution_policy",
                Value::String(alias.into()),
            )]));
            assert_eq!(options.execution_policy, expected, "{alias}");
        }
        for (alias, expected) in [
            ("final_only", BvpSciOutputPolicy::FinalOnly),
            ("finalonly", BvpSciOutputPolicy::FinalOnly),
            ("mesh_and_values", BvpSciOutputPolicy::MeshAndValues),
            ("meshandvalues", BvpSciOutputPolicy::MeshAndValues),
            ("dense_output", BvpSciOutputPolicy::DenseOutput),
            ("denseoutput", BvpSciOutputPolicy::DenseOutput),
        ] {
            let (options, _) = parsed_sci(sci_settings(&[(
                "output_policy",
                Value::String(alias.into()),
            )]));
            assert_eq!(options.output_policy, expected, "{alias}");
        }
        for (alias, expected) in [
            (
                "off",
                crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Off,
            ),
            (
                "counters",
                crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Counters,
            ),
            (
                "timings",
                crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Timings,
            ),
            (
                "detailed",
                crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Detailed,
            ),
        ] {
            let (options, _) =
                parsed_sci(sci_settings(&[("telemetry", Value::String(alias.into()))]));
            assert_eq!(options.telemetry, expected, "{alias}");
        }
        for (alias, expected) in [
            ("off", SolverLogLevel::Off),
            ("none", SolverLogLevel::Off),
            ("error", SolverLogLevel::Error),
            ("warn", SolverLogLevel::Warn),
            ("warning", SolverLogLevel::Warn),
            ("info", SolverLogLevel::Info),
            ("debug", SolverLogLevel::Debug),
        ] {
            let (options, _) =
                parsed_sci(sci_settings(&[("loglevel", Value::String(alias.into()))]));
            assert_eq!(options.logging_level, expected, "{alias}");
        }
        for (alias, expected) in [
            (
                "use_if_available",
                SymbolicIvpAotBuildPolicy::UseIfAvailable,
            ),
            ("use", SymbolicIvpAotBuildPolicy::UseIfAvailable),
            ("auto", SymbolicIvpAotBuildPolicy::UseIfAvailable),
            (
                "require_prebuilt",
                SymbolicIvpAotBuildPolicy::RequirePrebuilt,
            ),
            ("require", SymbolicIvpAotBuildPolicy::RequirePrebuilt),
            (
                "build_if_missing",
                SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
            ),
            (
                "build",
                SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
            ),
            (
                "rebuild_always",
                SymbolicIvpAotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                },
            ),
            (
                "rebuild",
                SymbolicIvpAotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                },
            ),
        ] {
            let settings = values(&[
                ("solver_family", Value::String("sci".into())),
                ("execution", Value::String("aot".into())),
                ("aot_build_policy", Value::String(alias.into())),
            ]);
            let BvpSolverSettings::Sci {
                aot_config: Some(config),
                ..
            } = parse_hybrid_bvp_solver_settings_from_document(&document(settings)).unwrap()
            else {
                panic!("expected AOT route for policy alias {alias}");
            };
            assert_eq!(config.build_policy, expected, "{alias}");
        }
        for (alias, expected) in [
            ("c", AotCodegenBackend::C),
            ("rust", AotCodegenBackend::Rust),
            ("zig", AotCodegenBackend::Zig),
        ] {
            let settings = values(&[
                ("solver_family", Value::String("sci".into())),
                ("execution", Value::String("aot".into())),
                ("aot_codegen_backend", Value::String(alias.into())),
            ]);
            let BvpSolverSettings::Sci {
                aot_config: Some(config),
                ..
            } = parse_hybrid_bvp_solver_settings_from_document(&document(settings)).unwrap()
            else {
                panic!("expected AOT route for codegen backend {alias}");
            };
            assert_eq!(config.aot_codegen_backend, expected, "{alias}");
        }
    }

    #[test]
    fn sci_settings_parser_accepts_a_settings_only_document_and_preserves_route_options() {
        let document = document(values(&[
            ("solver_family", Value::String("sci".into())),
            ("execution", Value::String("lambdify".into())),
            ("assembly", Value::String("atom_view_native".into())),
            ("matrix_layout", Value::String("banded_auto".into())),
            ("boundary_tolerance", Value::Float(2e-5)),
            ("max_nodes", Value::Integer(4096)),
            ("max_newton_iterations", Value::Integer(13)),
            ("max_jacobian_refreshes", Value::Integer(5)),
            ("max_mesh_refinements", Value::Integer(7)),
            ("max_backtracking_steps", Value::Integer(6)),
            ("execution_policy", Value::String("auto".into())),
            ("execution_min_work", Value::Integer(32)),
            ("output_policy", Value::String("final_only".into())),
            ("sparse_factorization", Value::String("superlu".into())),
            ("loglevel", Value::String("debug".into())),
            ("telemetry", Value::String("timings".into())),
        ]));
        let BvpSolverSettings::Sci {
            options,
            sparse_factorization,
            ..
        } = parse_hybrid_bvp_solver_settings_from_document(&document).unwrap()
        else {
            panic!("expected Sci settings");
        };
        assert_eq!(options.assembly, Some(BvpSciAssembly::AtomViewNative));
        assert_eq!(options.matrix_layout, BvpSciMatrixLayout::BandedAuto);
        assert_eq!(options.max_nodes, 4096);
        assert_eq!(options.boundary_tolerance, Some(2e-5));
        assert_eq!(options.max_newton_iterations, 13);
        assert_eq!(options.max_jacobian_refreshes, 5);
        assert_eq!(options.max_mesh_refinements, 7);
        assert_eq!(options.max_backtracking_steps, 6);
        assert_eq!(
            options.execution_policy,
            BvpSciExecutionPolicy::Auto { min_work: 32 }
        );
        assert_eq!(options.output_policy, BvpSciOutputPolicy::FinalOnly);
        assert_eq!(options.logging_level, SolverLogLevel::Debug);
        assert_eq!(
            options.telemetry,
            crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Timings
        );
        assert_eq!(sparse_factorization, BvpSciSparseFactorization::SuperLu);
    }

    #[test]
    fn sci_aot_settings_forward_compiler_policy_and_output_path() {
        let document = document(values(&[
            ("solver_family", Value::String("sci".into())),
            ("execution", Value::String("aot".into())),
            ("aot_build_policy", Value::String("build_if_missing".into())),
            ("aot_build_profile", Value::String("debug".into())),
            ("aot_codegen_backend", Value::String("zig".into())),
            ("aot_output_dir", Value::String("artifacts/bvp".into())),
            (
                "aot_handoff_path",
                Value::String("artifacts/handoff.json".into()),
            ),
            ("aot_crate_name", Value::String("hybrid_generated".into())),
            ("aot_module_name", Value::String("hybrid_module".into())),
            (
                "aot_residual_chunking",
                Value::String("target_chunks".into()),
            ),
            ("aot_residual_chunk_size", Value::Integer(3)),
            ("aot_sparse_jacobian_chunking", Value::String("nnz".into())),
            ("aot_sparse_jacobian_chunk_size", Value::Integer(17)),
            (
                "aot_banded_jacobian_chunking",
                Value::String("diagonals".into()),
            ),
            ("aot_banded_jacobian_chunk_size", Value::Integer(5)),
        ]));
        let BvpSolverSettings::Sci {
            aot_config: Some(config),
            options,
            ..
        } = parse_hybrid_bvp_solver_settings_from_document(&document).unwrap()
        else {
            panic!("expected an AOT Sci route");
        };
        assert_eq!(options.execution, BvpSciExecution::Aot);
        assert_eq!(
            config.build_policy,
            SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug
            }
        );
        assert_eq!(config.aot_codegen_backend, AotCodegenBackend::Zig);
        assert_eq!(
            config.output_parent_dir.as_deref(),
            Some(std::path::Path::new("artifacts/bvp"))
        );
        assert_eq!(
            config.handoff_path.as_deref(),
            Some(std::path::Path::new("artifacts/handoff.json"))
        );
        assert_eq!(
            config.crate_name_override.as_deref(),
            Some("hybrid_generated")
        );
        assert_eq!(
            config.module_name_override.as_deref(),
            Some("hybrid_module")
        );
        assert_eq!(
            config.residual_chunking_strategy,
            ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }
        );
        assert_eq!(
            config.sparse_jacobian_chunking_strategy,
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: 17
            }
        );
        assert_eq!(
            config.banded_jacobian_chunking_strategy,
            BandedChunkingStrategy::ByDiagonalCount {
                diagonals_per_chunk: 5
            }
        );
    }

    #[test]
    fn damped_settings_reuse_native_parser_and_apply_hybrid_telemetry_default() {
        let document = document(values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("Damped".into())),
            ("scheme", Value::String("trapezoid".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(1e-7)),
            ("max_iterations", Value::Integer(24)),
        ]));
        let BvpSolverSettings::Damped(options) =
            parse_hybrid_bvp_solver_settings_from_document(&document).unwrap()
        else {
            panic!("expected Damped options");
        };
        assert_eq!(options.scheme, "trapezoid");
        assert_eq!(options.abs_tolerance, 1e-7);
        assert_eq!(options.max_iterations, 24);
        assert_eq!(
            options.generated_backend_config.bvp_telemetry_mode,
            crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Off
        );
    }

    #[test]
    fn damped_native_strategy_sections_and_model_bounds_forward_without_loss() {
        let mut document = document(values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("damped".into())),
            ("scheme", Value::String("forward".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(1e-8)),
            ("max_iterations", Value::Integer(19)),
        ]));
        document.insert(
            "strategy_params".into(),
            values(&[
                (
                    "max_jac",
                    Value::Optional(Some(Box::new(Value::Integer(4)))),
                ),
                (
                    "max_damp_iter",
                    Value::Optional(Some(Box::new(Value::Integer(7)))),
                ),
                (
                    "damp_factor",
                    Value::Optional(Some(Box::new(Value::Float(0.35)))),
                ),
            ]),
        );
        document.insert(
            "bounds".into(),
            HashMap::from([(
                "temperature".into(),
                Some(vec![Value::Float(250.0), Value::Float(4000.0)]),
            )]),
        );
        document.insert(
            "rel_tolerance".into(),
            values(&[("temperature", Value::Float(2e-5))]),
        );
        document.insert(
            "adaptive_strategy".into(),
            values(&[
                ("version", Value::Integer(1)),
                ("max_refinements", Value::Integer(6)),
            ]),
        );
        document.insert(
            "grid_refinement".into(),
            values(&[("easy", Value::Vector(vec![0.2]))]),
        );

        let BvpSolverSettings::Damped(options) =
            parse_hybrid_bvp_solver_settings_from_document(&document).unwrap()
        else {
            panic!("expected Damped options");
        };
        assert_eq!(
            options.bounds.as_ref().unwrap()["temperature"],
            (250.0, 4000.0)
        );
        assert_eq!(options.rel_tolerance.as_ref().unwrap()["temperature"], 2e-5);
        let strategy = options.strategy_params.unwrap();
        assert_eq!(strategy.max_jac, Some(4));
        assert_eq!(strategy.max_damp_iter, Some(7));
        assert_eq!(strategy.damp_factor, Some(0.35));
        let adaptive = strategy.adaptive.unwrap();
        assert_eq!(adaptive.version, 1);
        assert_eq!(adaptive.max_refinements, 6);
        assert_eq!(
            adaptive.grid_method,
            crate::numerical::BVP_Damp::grid_api::GridRefinementMethod::Easy(0.2)
        );
    }

    #[test]
    fn hybrid_parser_rejects_native_report_toggle_it_cannot_forward() {
        let document = document(values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("damped".into())),
            ("scheme", Value::String("forward".into())),
            ("method", Value::String("Sparse".into())),
            ("dont_save_log", Value::Boolean(true)),
        ]));
        let error = parse_hybrid_bvp_solver_settings_from_document(&document)
            .err()
            .expect("report toggle must be rejected");
        assert!(matches!(
            error,
            HybridBvpSettingsError::UnsupportedField { ref path, .. }
                if path == "solver_settings.dont_save_log"
        ));
    }

    #[test]
    fn sci_aot_build_profile_requires_a_building_policy() {
        let document = document(values(&[
            ("solver_family", Value::String("sci".into())),
            ("execution", Value::String("aot".into())),
            ("aot_build_profile", Value::String("debug".into())),
        ]));
        let error = parse_hybrid_bvp_solver_settings_from_document(&document)
            .err()
            .expect("build profile without a build policy must be rejected");
        assert!(matches!(
            error,
            HybridBvpSettingsError::InvalidField { ref path, .. }
                if path == "solver_settings.aot_build_profile"
        ));
    }

    #[test]
    fn sci_lambdify_rejects_generated_backend_controls() {
        let document = document(values(&[
            ("solver_family", Value::String("sci".into())),
            ("execution", Value::String("lambdify".into())),
            (
                "aot_residual_chunking",
                Value::String("target_chunks".into()),
            ),
            ("aot_residual_chunk_size", Value::Integer(4)),
        ]));
        let error = parse_hybrid_bvp_solver_settings_from_document(&document)
            .err()
            .expect("generated controls must be rejected for Lambdify");
        assert!(matches!(
            error,
            HybridBvpSettingsError::UnsupportedField { ref path, .. }
                if path == "solver_settings.aot_residual_chunking"
        ));
    }

    #[test]
    fn frozen_settings_preserve_naive_strategy_and_native_scheme() {
        let document = document(values(&[
            ("solver_family", Value::String("frozen".into())),
            ("strategy", Value::String("naive".into())),
            ("scheme", Value::String("trapezoid".into())),
            ("method", Value::String("Dense".into())),
            ("tolerance", Value::Float(2e-8)),
            ("max_iterations", Value::Integer(31)),
            (
                "backend_policy",
                Value::String("prefer_aot_then_lambdify".into()),
            ),
            ("aot_execution_policy", Value::String("sequential".into())),
            ("bvp_logging_mode", Value::String("warnings".into())),
            ("telemetry", Value::String("counters".into())),
        ]));
        let BvpSolverSettings::Frozen(options) =
            parse_hybrid_bvp_solver_settings_from_document(&document).unwrap()
        else {
            panic!("expected Frozen options");
        };
        assert_eq!(options.strategy, "Naive");
        assert_eq!(options.scheme, "trapezoid");
        assert_eq!(options.method, "Dense");
        assert_eq!(options.tolerance, 2e-8);
        assert_eq!(options.max_iterations, 31);
        assert_eq!(
            options.generated_backend_config.aot_execution_policy,
            AotExecutionPolicy::SequentialOnly
        );
        assert_eq!(
            options.generated_backend_config.bvp_logging_mode,
            BvpLoggingMode::Warnings
        );
        assert_eq!(
            options.generated_backend_config.bvp_telemetry_mode,
            crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Counters
        );
    }

    #[test]
    fn family_and_strategy_are_distinct_and_bad_routes_report_the_field_path() {
        let document = document(values(&[
            ("solver_family", Value::String("frozen".into())),
            ("strategy", Value::String("damped".into())),
        ]));
        let error = match parse_hybrid_bvp_solver_settings_from_document(&document) {
            Ok(_) => panic!("invalid family/strategy combination was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("solver_settings.strategy"));
    }

    #[test]
    fn hybrid_settings_parser_rejects_problem_sections_instead_of_ignoring_them() {
        let mut document = document(values(&[("solver_family", Value::String("sci".into()))]));
        document.insert("equations".into(), HashMap::new());
        let error = match parse_hybrid_bvp_solver_settings_from_document(&document) {
            Ok(_) => panic!("problem section should be rejected by the settings-only parser"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("equations"));
    }

    #[test]
    fn damped_timings_requires_an_explicit_lambdify_only_route() {
        let settings = values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("damped".into())),
            ("scheme", Value::String("forward".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(1e-6)),
            ("max_iterations", Value::Integer(25)),
            ("telemetry", Value::String("timings".into())),
        ]);
        let error =
            match parse_hybrid_bvp_solver_settings_from_document(&document(settings.clone())) {
                Ok(_) => panic!("ambiguous backend policy may select AOT"),
                Err(error) => error,
            };
        assert!(matches!(
            error,
            HybridBvpSettingsError::Telemetry(BvpSettingsError::UnsupportedTelemetry {
                mode: BvpApiTelemetryMode::Timings,
                ..
            })
        ));

        let mut aot_route = settings.clone();
        aot_route.insert(
            "backend_policy".into(),
            Some(vec![Value::String("prefer_aot_then_lambdify".into())]),
        );
        assert!(matches!(
            parse_hybrid_bvp_solver_settings_from_document(&document(aot_route)),
            Err(HybridBvpSettingsError::Telemetry(
                BvpSettingsError::UnsupportedTelemetry {
                    mode: BvpApiTelemetryMode::Timings,
                    ..
                }
            ))
        ));

        let mut settings = settings;
        settings.insert(
            "backend_policy".into(),
            Some(vec![Value::String("lambdify_only".into())]),
        );
        let BvpSolverSettings::Damped(options) =
            parse_hybrid_bvp_solver_settings_from_document(&document(settings)).unwrap()
        else {
            panic!("expected Damped options");
        };
        assert_eq!(
            options.generated_backend_config.bvp_telemetry_mode,
            crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Timings
        );
        assert_eq!(
            options.generated_backend_config.lambdify_telemetry_mode,
            crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Timings
        );
        assert_eq!(
            options.generated_backend_config.aot_telemetry_mode,
            crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Off
        );
    }

    #[test]
    fn damped_generated_backend_controls_forward_without_loss() {
        let document = document(values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("damped".into())),
            ("scheme", Value::String("trapezoid".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(3e-8)),
            ("max_iterations", Value::Integer(42)),
            (
                "backend_policy",
                Value::String("prefer_aot_then_lambdify".into()),
            ),
            ("aot_execution_policy", Value::String("parallel".into())),
            ("aot_jobs_per_worker", Value::Integer(2)),
            ("aot_max_residual_jobs", Value::Integer(6)),
            ("aot_max_sparse_jobs", Value::Integer(8)),
            ("aot_fallback_policy", Value::String("thresholds".into())),
            ("aot_min_residual_outputs", Value::Integer(64)),
            ("aot_min_sparse_values", Value::Integer(96)),
            ("aot_compile_optimization", Value::String("o3".into())),
            ("aot_codegen_units", Value::Integer(12)),
            (
                "aot_residual_chunking",
                Value::String("output_count".into()),
            ),
            ("aot_residual_chunk_size", Value::Integer(32)),
            ("aot_sparse_jacobian_chunking", Value::String("rows".into())),
            ("aot_sparse_jacobian_chunk_size", Value::Integer(4)),
            ("atom_optimization_profile", Value::String("no_cse".into())),
            ("lambdify_execution_policy", Value::String("auto".into())),
            ("lambdify_min_work", Value::Integer(48)),
            ("expr_indexed_lambdify", Value::Boolean(false)),
            ("expr_dependency_columns", Value::Boolean(false)),
            ("bvp_logging_mode", Value::String("detailed".into())),
            ("bvp_logging_max_events", Value::Integer(128)),
        ]));
        let BvpSolverSettings::Damped(options) =
            parse_hybrid_bvp_solver_settings_from_document(&document).unwrap()
        else {
            panic!("expected Damped options");
        };
        let generated = options.generated_backend_config;
        let AotExecutionPolicy::Parallel(parallel) = generated.aot_execution_policy else {
            panic!("parallel AOT settings were not forwarded");
        };
        assert_eq!(parallel.jobs_per_worker, 2);
        assert_eq!(parallel.max_residual_jobs, Some(6));
        assert_eq!(parallel.max_sparse_jobs, Some(8));
        assert_eq!(
            parallel.fallback_policy,
            ParallelFallbackPolicy::Thresholds {
                min_residual_outputs: 64,
                min_sparse_values: 96,
            }
        );
        assert_eq!(
            generated.aot_compile_config.optimization,
            AotOptimizationLevel::O3
        );
        assert_eq!(generated.aot_compile_config.codegen_units, Some(12));
        assert_eq!(
            generated.aot_chunking_policy.residual,
            Some(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 32,
            })
        );
        assert_eq!(
            generated.aot_chunking_policy.sparse_jacobian,
            Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 4 })
        );
        assert_eq!(
            generated.atom_optimization_profile,
            AtomOptimizationProfile::NoCse
        );
        assert_eq!(
            generated.lambdify_execution_policy,
            BvpLambdifyExecutionPolicy::Auto { min_work: 48 }
        );
        assert!(!generated.expr_indexed_lambdify_enabled);
        assert!(!generated.expr_dependency_columns_enabled);
        assert_eq!(generated.bvp_logging_mode, BvpLoggingMode::Detailed);
        assert_eq!(generated.bvp_logging_max_events, 128);
        assert_eq!(options.scheme, "trapezoid");
        assert_eq!(options.abs_tolerance, 3e-8);
        assert_eq!(options.max_iterations, 42);
    }

    #[test]
    fn aot_and_parallel_settings_reject_incompatible_orphan_controls() {
        let mut lambda_only = values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("damped".into())),
            ("scheme", Value::String("forward".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(1e-6)),
            ("max_iterations", Value::Integer(25)),
            ("backend_policy", Value::String("lambdify_only".into())),
            ("aot_residual_chunking", Value::String("whole".into())),
        ]);
        let error =
            match parse_hybrid_bvp_solver_settings_from_document(&document(lambda_only.clone())) {
                Ok(_) => panic!("AOT chunking must not be silently ignored on LambdifyOnly"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("aot_residual_chunking"));

        lambda_only.remove("backend_policy");
        lambda_only.remove("aot_residual_chunking");
        lambda_only.insert("aot_max_sparse_jobs".into(), Some(vec![Value::Integer(4)]));
        lambda_only.insert(
            "aot_execution_policy".into(),
            Some(vec![Value::String("auto".into())]),
        );
        let error = match parse_hybrid_bvp_solver_settings_from_document(&document(lambda_only)) {
            Ok(_) => panic!("parallel controls require the parallel execution policy"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("aot_execution_policy"));
    }

    #[test]
    fn aot_codegen_units_must_be_positive() {
        let document = document(values(&[
            ("solver_family", Value::String("damped".into())),
            ("strategy", Value::String("damped".into())),
            ("scheme", Value::String("forward".into())),
            ("method", Value::String("Sparse".into())),
            ("abs_tolerance", Value::Float(1e-6)),
            ("max_iterations", Value::Integer(25)),
            ("aot_codegen_units", Value::Integer(0)),
        ]));
        let error = parse_hybrid_bvp_solver_settings_from_document(&document)
            .err()
            .expect("zero AOT codegen units must be rejected");
        assert!(
            matches!(
                error,
                HybridBvpSettingsError::InvalidField { ref path, .. }
                    if path == "solver_settings.aot_codegen_units"
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn unknown_and_multivalued_settings_fail_with_precise_paths() {
        let mut unknown = values(&[("solver_family", Value::String("sci".into()))]);
        unknown.insert("tolerance_typo".into(), Some(vec![Value::Float(1e-5)]));
        let error = match parse_hybrid_bvp_solver_settings_from_document(&document(unknown)) {
            Ok(_) => panic!("unknown settings must not be silently discarded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("solver_settings.tolerance_typo"));

        let mut multiple = values(&[("solver_family", Value::String("sci".into()))]);
        multiple.insert(
            "tolerance".into(),
            Some(vec![Value::Float(1e-5), Value::Float(2e-5)]),
        );
        let error = match parse_hybrid_bvp_solver_settings_from_document(&document(multiple)) {
            Ok(_) => panic!("scalar settings must reject duplicate values"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("solver_settings.tolerance"));
    }
}
