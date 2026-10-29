//! Settings-only adapter for applications that construct their IVP model.
//!
//! The caller owns physical data, symbolic RHS construction and initial
//! conditions. This module reads only the task's solver selection and shared
//! stiff policy; native family-specific controls remain in the native
//! configuration passed through `StiffSolverRequest`.

use crate::command_interpreter::task_parser::{DocumentMap, Value};
use crate::command_interpreter::task_parser_ivp::{
    IvpMethodSpec, IvpSolverSettingsSpec, IvpTaskError, Lsode2TaskExecutionSpec,
    parse_ivp_solver_settings_from_document,
};
use crate::numerical::LSODE2::{
    Lsode2ControllerConfig, Lsode2LinearSolverPolicy, Lsode2NativeExecutionConfig,
    Lsode2StopCondition, Lsode2SymbolicAssemblyBackend,
};
use crate::numerical::SolverTelemetryMode;
use crate::numerical::StiffAPI::{
    PreparedStiffSolver, StiffApiError, StiffCallbackPolicy, StiffExecution, StiffFrontend,
    StiffMatrixLayout, StiffPreset, StiffSolverKind, StiffSolverRequest, StiffSolverSettings,
};
use std::fmt::Write as _;

/// Parse the solver selector and common policy from an external application's
/// document. The returned settings can only prepare a request for the same
/// solver family; the numerical model itself is never read here.
pub fn parse_hybrid_stiff_settings_from_document(
    document: &DocumentMap,
) -> Result<HybridStiffSettings, IvpTaskError> {
    let parsed = parse_ivp_solver_settings_from_document(document)?;
    let kind = match parsed.solver.method {
        IvpMethodSpec::Bdf => StiffSolverKind::Bdf,
        IvpMethodSpec::Radau5 => StiffSolverKind::Radau,
        IvpMethodSpec::BackwardEuler => StiffSolverKind::BackwardEuler,
        IvpMethodSpec::Lsode2 => StiffSolverKind::Lsode2,
        _ => {
            return Err(IvpTaskError::UnsupportedOption {
                method: format!("{:?}", parsed.solver.method),
                option:
                    "hybrid_stiff_settings (supported families: BDF, Radau5, BackwardEuler, LSODE2)"
                        .to_string(),
            });
        }
    };
    reject_duplicate_lsode2_common_fields(document)?;

    let mut preset = StiffPreset::default();
    if let Some(frontend) = parsed.solver_options.symbolic_assembly {
        preset = preset.with_frontend(match frontend {
            Lsode2SymbolicAssemblyBackend::ExprLegacy => StiffFrontend::ExprLegacy,
            Lsode2SymbolicAssemblyBackend::AtomView => StiffFrontend::AtomViewNative,
        });
    }
    if let Some(layout) = parsed.solver_options.matrix_layout {
        preset = preset.with_matrix_layout(match layout {
            crate::numerical::LSODE2::Lsode2LinearSystemStructure::Dense => {
                StiffMatrixLayout::Dense
            }
            crate::numerical::LSODE2::Lsode2LinearSystemStructure::Sparse => {
                StiffMatrixLayout::Sparse
            }
            crate::numerical::LSODE2::Lsode2LinearSystemStructure::BandedAuto => {
                StiffMatrixLayout::BandedAuto
            }
            crate::numerical::LSODE2::Lsode2LinearSystemStructure::Banded { kl, ku } => {
                StiffMatrixLayout::Banded {
                    lower: kl,
                    upper: ku,
                }
            }
        });
    }
    if let Some(execution) = parsed.solver_options.symbolic_execution.as_ref() {
        preset = match execution {
            Lsode2TaskExecutionSpec::LambdifyExpr => {
                preset.with_execution(StiffExecution::Lambdify)
            }
            Lsode2TaskExecutionSpec::Aot { .. } => preset.with_execution(StiffExecution::Aot(
                super::task_parser_ivp::generated_backend_config_for_task(execution)
                    .expect("AOT task execution always creates a generated backend config"),
            )),
        };
    }
    if let Some(policy) = parsed.solver_options.execution_policy {
        preset = preset.with_callback_policy(match policy {
            crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Sequential => {
                StiffCallbackPolicy::Sequential
            }
            crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Parallel { min_work } => {
                StiffCallbackPolicy::Parallel { min_work }
            }
            crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Auto { min_work } => {
                StiffCallbackPolicy::Auto { min_work }
            }
        });
    }
    if let Some(telemetry) = parsed.solver_options.telemetry {
        preset = preset.with_telemetry(match telemetry {
            crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Off => {
                SolverTelemetryMode::Off
            }
            crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Counters => {
                SolverTelemetryMode::Counters
            }
            crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Timings => {
                SolverTelemetryMode::Timings
            }
            crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Detailed => {
                SolverTelemetryMode::Detailed
            }
        });
    }
    if let Some(logging) = parsed.solver_options.logging {
        preset = preset.with_logging(logging);
    }

    let family = parse_family_settings(kind, &parsed);
    Ok(HybridStiffSettings {
        selection: StiffSolverSettings::new(kind).with_preset(preset),
        family,
    })
}

/// Parsed solver selection, common preset and family-specific overrides.
/// This is a document adapter, not a replacement for native solver configs.
/// The caller still supplies the numerical problem through `StiffSolverRequest`.
#[derive(Debug, Clone)]
pub struct HybridStiffSettings {
    selection: StiffSolverSettings,
    pub family: HybridStiffFamilySettings,
}

impl HybridStiffSettings {
    pub const fn solver_kind(&self) -> StiffSolverKind {
        self.selection.solver_kind()
    }

    /// Apply parsed numeric controls to the caller-built native request, then
    /// use StiffAPI's normal capability validation and preparation path.
    pub fn prepare(
        self,
        request: StiffSolverRequest,
    ) -> Result<PreparedStiffSolver, StiffApiError> {
        let request = apply_family_settings(self.family, request)?;
        self.selection.prepare(request)
    }
}

#[derive(Debug, Clone)]
pub enum HybridStiffFamilySettings {
    Bdf {
        max_step: Option<f64>,
        rtol: Option<f64>,
        atol: Option<f64>,
        first_step: Option<f64>,
        max_steps: Option<usize>,
        vectorized: Option<bool>,
    },
    Radau {
        max_step: Option<f64>,
        rtol: Option<f64>,
        atol: Option<f64>,
        first_step: Option<f64>,
        max_newton_iterations: Option<usize>,
    },
    BackwardEuler {
        step_size: Option<f64>,
        tolerance: Option<f64>,
        max_iterations: Option<usize>,
    },
    Lsode2 {
        max_step: Option<f64>,
        rtol: Option<f64>,
        atol: Option<f64>,
        first_step: Option<f64>,
        vectorized: Option<bool>,
        controller: Option<Lsode2ControllerConfig>,
        linear_solver_policy: Option<Lsode2LinearSolverPolicy>,
        native_execution: Option<Lsode2NativeExecutionConfig>,
        stop_conditions: Vec<Lsode2StopCondition>,
    },
}

fn parse_family_settings(
    kind: StiffSolverKind,
    parsed: &IvpSolverSettingsSpec,
) -> HybridStiffFamilySettings {
    let options = &parsed.solver_options;
    match kind {
        StiffSolverKind::Bdf => HybridStiffFamilySettings::Bdf {
            max_step: options.max_step,
            rtol: options.rtol,
            atol: options.atol,
            first_step: options.first_step,
            max_steps: options.max_iterations,
            vectorized: options.vectorized,
        },
        StiffSolverKind::Radau => HybridStiffFamilySettings::Radau {
            max_step: options.max_step,
            rtol: options.rtol,
            atol: options.atol,
            first_step: options.first_step,
            max_newton_iterations: options.max_iterations,
        },
        StiffSolverKind::BackwardEuler => HybridStiffFamilySettings::BackwardEuler {
            step_size: options.step_size,
            tolerance: options.tolerance,
            max_iterations: options.max_iterations,
        },
        StiffSolverKind::Lsode2 => {
            let native = options.lsode2.as_ref();
            HybridStiffFamilySettings::Lsode2 {
                max_step: options.max_step,
                rtol: options.rtol,
                atol: options.atol,
                first_step: options.first_step,
                vectorized: options.vectorized,
                controller: native.and_then(|value| value.controller),
                linear_solver_policy: native.and_then(|value| value.linear_solver_policy),
                native_execution: native.and_then(|value| value.native_execution),
                stop_conditions: native
                    .map_or_else(Vec::new, |value| value.stop_conditions.clone()),
            }
        }
    }
}

fn apply_family_settings(
    settings: HybridStiffFamilySettings,
    request: StiffSolverRequest,
) -> Result<StiffSolverRequest, StiffApiError> {
    match (settings, request) {
        (
            HybridStiffFamilySettings::Bdf {
                max_step,
                rtol,
                atol,
                first_step,
                max_steps,
                vectorized,
            },
            StiffSolverRequest::Bdf { mut options },
        ) => {
            if let Some(value) = max_step {
                options.max_step = value;
            }
            if let Some(value) = rtol {
                options.rtol = value;
            }
            if let Some(value) = atol {
                options.atol = value;
            }
            if let Some(value) = first_step {
                options.first_step = Some(value);
            }
            if let Some(value) = max_steps {
                options.max_steps = value;
            }
            if let Some(value) = vectorized {
                options.vectorized = value;
            }
            Ok(StiffSolverRequest::Bdf { options })
        }
        (
            HybridStiffFamilySettings::Radau {
                max_step,
                rtol,
                atol,
                first_step,
                max_newton_iterations,
            },
            StiffSolverRequest::Radau {
                problem,
                mut config,
                initial_state,
                parameters,
            },
        ) => {
            if let Some(value) = max_step {
                config.max_step = value;
            }
            if let Some(value) = rtol {
                config.rtol = value;
            }
            if let Some(value) = atol {
                config.atol = value;
            }
            if let Some(value) = first_step {
                config.first_step = Some(value);
            }
            if let Some(value) = max_newton_iterations {
                config.max_newton_iterations = value;
            }
            Ok(StiffSolverRequest::Radau {
                problem,
                config,
                initial_state,
                parameters,
            })
        }
        (
            HybridStiffFamilySettings::BackwardEuler {
                step_size,
                tolerance,
                max_iterations,
            },
            StiffSolverRequest::BackwardEuler {
                mut options,
                equation_parameters,
                parameter_values,
            },
        ) => {
            if let Some(value) = step_size {
                options.h = Some(value);
            }
            if let Some(value) = tolerance {
                options.tolerance = value;
            }
            if let Some(value) = max_iterations {
                options.max_iterations = value;
            }
            Ok(StiffSolverRequest::BackwardEuler {
                options,
                equation_parameters,
                parameter_values,
            })
        }
        (
            HybridStiffFamilySettings::Lsode2 {
                max_step,
                rtol,
                atol,
                first_step,
                vectorized,
                controller,
                linear_solver_policy,
                native_execution,
                stop_conditions,
            },
            StiffSolverRequest::Lsode2 {
                mut config,
                logging_level,
            },
        ) => {
            if let Some(value) = max_step {
                config.max_step = value;
            }
            if let Some(value) = rtol {
                config.rtol = value;
            }
            if let Some(value) = atol {
                config.atol = value;
            }
            if let Some(value) = first_step {
                config.first_step = Some(value);
            }
            if let Some(value) = vectorized {
                config.vectorized = value;
            }
            if let Some(value) = controller {
                config.controller = value;
            }
            if let Some(value) = linear_solver_policy {
                config = config.with_linear_solver_policy(value);
            }
            if let Some(value) = native_execution {
                config.native_execution = value;
            }
            config.stop_conditions.extend(stop_conditions);
            Ok(StiffSolverRequest::Lsode2 {
                config,
                logging_level,
            })
        }
        (settings, request) => {
            let selected = match settings {
                HybridStiffFamilySettings::Bdf { .. } => StiffSolverKind::Bdf,
                HybridStiffFamilySettings::Radau { .. } => StiffSolverKind::Radau,
                HybridStiffFamilySettings::BackwardEuler { .. } => StiffSolverKind::BackwardEuler,
                HybridStiffFamilySettings::Lsode2 { .. } => StiffSolverKind::Lsode2,
            };
            Err(StiffApiError::SolverKindMismatch {
                selected,
                request: request.solver_kind(),
            })
        }
    }
}

/// Serialize only the task selector and solver-options sections. This returns
/// a fragment for the caller to merge; it never rewrites or drops the
/// application's physical/domain sections.
pub fn serialize_hybrid_solver_fragment(document: &DocumentMap) -> Result<String, IvpTaskError> {
    parse_hybrid_stiff_settings_from_document(document)?;
    let mut output = String::new();
    for section_name in ["task", "solver_options"] {
        let Some(section) = document.get(section_name) else {
            continue;
        };
        let mut entries = section.iter().collect::<Vec<_>>();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        let mut lines = Vec::new();
        for (key, values) in entries {
            // The task section is intentionally restricted to the selector;
            // other task metadata belongs to the hosting application.
            if section_name == "task" && !matches!(key.as_str(), "solver" | "method") {
                continue;
            }
            let Some(values) = values else { continue };
            if values.is_empty() {
                continue;
            }
            let rendered = values
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("{key}: {rendered}"));
        }
        if !lines.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            writeln!(&mut output, "{section_name}").expect("writing to String cannot fail");
            for line in lines {
                writeln!(&mut output, "{line}").expect("writing to String cannot fail");
            }
        }
    }
    Ok(output)
}

fn reject_duplicate_lsode2_common_fields(document: &DocumentMap) -> Result<(), IvpTaskError> {
    let Some(options) = document.get("solver_options") else {
        return Ok(());
    };
    for (common, family) in [
        ("symbolic_frontend", "lsode2_symbolic_assembly"),
        ("symbolic_assembly", "lsode2_symbolic_assembly"),
        ("symbolic_execution", "lsode2_symbolic_execution"),
        ("matrix_layout", "lsode2_linear_structure"),
    ] {
        if options.contains_key(common) && options.contains_key(family) {
            return Err(IvpTaskError::InvalidConfiguration {
                field: format!("solver_options.{common}"),
                message: format!(
                    "conflicts with solver_options.{family}; specify this policy axis once"
                ),
            });
        }
    }
    Ok(())
}
