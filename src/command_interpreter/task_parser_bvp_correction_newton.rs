//! Full-document adapter sharing the native settings contract with hybrid BVPs.

use super::*;
use crate::command_interpreter::task_parser_bvp_hybrid::parse_hybrid_bvp_solver_settings_from_document;
use crate::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonOptions, CorrectionNewtonProblem, CorrectionNewtonSolver,
};
use crate::numerical::BVP_api::BvpSolverSettings;

/// Validated native settings retained without duplicating the hybrid schema.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpCorrectionNewtonOptionsSpec {
    document: DocumentMap,
}

impl BvpCorrectionNewtonOptionsSpec {
    /// Build native options using the current full-document solver selection.
    /// No equations, initial guess, or solver preparation are required.
    pub fn build_options(
        &self,
        solver: &BvpSolverSelectionSpec,
    ) -> Result<CorrectionNewtonOptions, BvpTaskError> {
        if solver.family != BvpSolverFamilySpec::CorrectionNewton
            || solver.strategy != BvpStrategySpec::Damped
            || solver.scheme != "trapezoid"
        {
            return Err(BvpTaskError::UnsupportedRoute(
                "CorrectionNewton task settings require the fixed trapezoid Newton route".into(),
            ));
        }
        let mut document = self.document.clone();
        let settings = document.entry("solver_settings".into()).or_default();
        for (field, value) in [
            ("solver_family", "correction_newton"),
            (
                "assembly",
                match solver.frontend.unwrap_or(BvpFrontendSpec::ExprLegacy) {
                    BvpFrontendSpec::ExprLegacy => "expr_legacy",
                    BvpFrontendSpec::AtomViewNative => "atom_view_native",
                },
            ),
            (
                "execution",
                match solver.execution.unwrap_or(BvpExecutionSpec::Lambdify) {
                    BvpExecutionSpec::Lambdify => "lambdify",
                    BvpExecutionSpec::Aot => "aot",
                },
            ),
            (
                "matrix_layout",
                match solver.backend {
                    BvpLinearBackendSpec::Dense => "dense",
                    BvpLinearBackendSpec::Sparse => "sparse",
                    BvpLinearBackendSpec::Banded => "banded_auto",
                },
            ),
        ] {
            settings.insert(field.into(), Some(vec![Value::String(value.into())]));
        }
        match parse_hybrid_bvp_solver_settings_from_document(&document)
            .map_err(map_settings_error)?
        {
            BvpSolverSettings::CorrectionNewton(options) => Ok(options),
            _ => unreachable!("CorrectionNewton family selected above"),
        }
    }
}

fn map_settings_error(error: HybridBvpSettingsError) -> BvpTaskError {
    match error {
        HybridBvpSettingsError::UnsupportedField { path, solver } => {
            BvpTaskError::UnsupportedOption {
                option: path.replacen("solver_settings.", "solver_options.", 1),
                solver: solver.into(),
            }
        }
        HybridBvpSettingsError::InvalidField { path, message } => {
            let path = path.replacen("solver_settings.", "solver_options.", 1);
            let (section, field) = path.split_once('.').unwrap_or(("solver_options", &path));
            BvpTaskError::InvalidField {
                section: section.into(),
                field: field.into(),
                message,
            }
        }
        other => BvpTaskError::HybridSettings(other),
    }
}

pub(super) fn parse_options(
    document: &DocumentMap,
    solver: &BvpSolverSelectionSpec,
) -> Result<BvpCorrectionNewtonOptionsSpec, BvpTaskError> {
    const ALLOWED: &[&str] = &[
        "task",
        "equations",
        "where",
        "parameters",
        "boundary_conditions",
        "mesh",
        "initial_guess",
        "solver_options",
        "postprocessing",
        "bounds",
        "rel_tolerance",
    ];
    if document.contains_key("continuation") {
        return Err(BvpTaskError::UnsupportedContinuation(
            "BVP_CorrectionNewton task documents currently support a single fixed-mesh solve"
                .into(),
        ));
    }
    if let Some(section) = document
        .keys()
        .find(|name| !ALLOWED.contains(&name.as_str()))
    {
        return Err(BvpTaskError::UnsupportedOption {
            option: section.clone(),
            solver: "BVP_CorrectionNewton".into(),
        });
    }
    let settings = document.get("solver_options").cloned().unwrap_or_default();
    for field in ["solver_family", "assembly", "execution", "matrix_layout"] {
        if settings.contains_key(field) {
            return Err(BvpTaskError::UnsupportedOption {
                option: format!("solver_options.{field}"),
                solver: "BVP_CorrectionNewton full documents (select routes in task)".into(),
            });
        }
    }
    let mut native = DocumentMap::from([("solver_settings".into(), settings)]);
    for section in ["bounds", "rel_tolerance"] {
        if let Some(values) = document.get(section) {
            native.insert(section.into(), values.clone());
        }
    }
    let spec = BvpCorrectionNewtonOptionsSpec { document: native };
    spec.build_options(solver)?;
    Ok(spec)
}

pub(super) fn validate_options(
    solver: &BvpSolverSelectionSpec,
    options: &BvpSolverOptionsSpec,
) -> Result<(), BvpTaskError> {
    // Reject programmatic cross-family overrides rather than silently ignoring them.
    let mut common = options.clone();
    common.correction_newton = None;
    if common != BvpSolverOptionsSpec::default() {
        return Err(BvpTaskError::UnsupportedOption {
            option: "solver_options (non-CorrectionNewton fields)".into(),
            solver: "BVP_CorrectionNewton".into(),
        });
    }
    native_options(solver, options).map(|_| ())
}

fn native_options(
    solver: &BvpSolverSelectionSpec,
    options: &BvpSolverOptionsSpec,
) -> Result<CorrectionNewtonOptions, BvpTaskError> {
    options
        .correction_newton
        .as_ref()
        .ok_or_else(|| BvpTaskError::InvalidConfiguration {
            field: "solver_options.correction_newton".into(),
            message: "missing native CorrectionNewton settings".into(),
        })?
        .build_options(solver)
}

pub(super) fn validate_task(spec: &BvpTaskSpec) -> Result<(), BvpTaskError> {
    if spec.continuation.is_some() {
        return Err(BvpTaskError::UnsupportedContinuation(
            "BVP_CorrectionNewton task documents currently support a single fixed-mesh solve"
                .into(),
        ));
    }
    if spec.postprocessing.plot {
        return Err(BvpTaskError::UnsupportedOption {
            option: "postprocessing.plot".into(),
            solver: "BVP_CorrectionNewton (use typed postprocessing actions)".into(),
        });
    }
    let options = native_options(&spec.solver, &spec.solver_options)?;
    for (section, names) in [
        ("bounds", options.bounds.keys().collect::<Vec<_>>()),
        (
            "rel_tolerance",
            options.relative_tolerances.keys().collect::<Vec<_>>(),
        ),
    ] {
        for name in names {
            if !spec.equations.unknowns.contains(name) {
                return Err(BvpTaskError::InvalidField {
                    section: section.into(),
                    field: name.clone(),
                    message: "unknown state".into(),
                });
            }
        }
    }
    for name in &spec.equations.unknowns {
        if spec.boundary_conditions.conditions.get(name).map(Vec::len) != Some(1) {
            return Err(BvpTaskError::UnsupportedOption {
                option: format!("boundary_conditions.{name}"),
                solver: "BVP_CorrectionNewton (requires exactly one endpoint condition per state)"
                    .into(),
            });
        }
    }
    Ok(())
}

pub(super) fn run_task(spec: BvpTaskSpec) -> Result<BvpTaskRunResult, BvpTaskError> {
    let options = native_options(&spec.solver, &spec.solver_options)?;
    let problem = CorrectionNewtonProblem {
        equations: spec.equations.rhs.clone(),
        initial_guess: DMatrix::from_fn(
            spec.equations.unknowns.len(),
            spec.mesh.n_steps,
            |row, _| spec.initial_guess.values[row],
        ),
        state_names: spec.equations.unknowns.clone(),
        independent_variable: spec.equations.arg.clone(),
        boundary_conditions: spec.boundary_conditions.conditions.clone(),
        interval: (spec.mesh.t0, spec.mesh.t_end),
        steps: spec.mesh.n_steps,
    };
    let mut solver = CorrectionNewtonSolver::from_symbolic(problem, options)
        .map_err(BvpTaskError::CorrectionNewton)?;
    let report = solver.solve().map_err(BvpTaskError::CorrectionNewton)?;
    let values = solver
        .full_grid_values(&report.state)
        .map_err(BvpTaskError::CorrectionNewton)?;
    let mesh = uniform_bvp_mesh(spec.mesh.t0, spec.mesh.t_end, values.nrows());
    let plan = spec.postprocessing.to_plan("bvp_result.csv");
    if !plan.actions.is_empty() {
        let dataset = PostprocessDataset::new(
            spec.equations.arg.clone(),
            spec.equations.unknowns.clone(),
            DVector::from_vec(mesh.clone()),
            values.clone(),
        )
        .map_err(BvpTaskError::Postprocess)?;
        plan.execute(&dataset).map_err(BvpTaskError::Postprocess)?;
    }
    let result = values.transpose();
    let continuation_telemetry = BvpTaskContinuationTelemetry {
        segments: 1,
        fresh_preparations: 1,
        peak_result_elements: result.len(),
        ..Default::default()
    };
    let segment = BvpTaskSegmentResult {
        index: 0,
        parameters: Vec::new(),
        mesh: mesh.clone(),
        result: result.clone(),
    };
    Ok(BvpTaskRunResult {
        specification: spec,
        result: Some(result),
        mesh: Some(mesh),
        continuation_segments: vec![segment],
        continuation_telemetry,
        runtime_telemetry: None,
        correction_newton_report: Some(report),
    })
}
