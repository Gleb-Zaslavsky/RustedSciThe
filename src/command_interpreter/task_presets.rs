//! Fixed, machine-readable starter configurations for agent-authored tasks.
//!
//! Presets select a verified solver route and fill only omitted settings.
//! They are not model-aware recommendations; callers remain responsible for
//! selecting a route suitable for their problem and can override tunable
//! values explicitly. Every expanded document is validated by the normal task
//! inspection path before it is returned.

use crate::command_interpreter::task_contract::inspect_task_json;
use crate::command_interpreter::task_json::{TaskJsonError, json_value_from_json};
use crate::command_interpreter::task_runner::TaskRunnerError;
use serde_json::{Map, Value, json};

#[derive(Debug, thiserror::Error)]
pub enum TaskPresetError {
    #[error("unknown task preset `{0}`")]
    UnknownPreset(String),
    #[error("task JSON root must be an object")]
    InvalidRoot,
    #[error("task JSON field `{0}` must be an object for this preset")]
    InvalidShape(&'static str),
    #[error(
        "preset `{preset}` conflicts with explicit route field `{path}`: expected `{expected}`, found `{found}`"
    )]
    RouteConflict {
        preset: &'static str,
        path: &'static str,
        expected: &'static str,
        found: String,
    },
    #[error("preset `{preset}` expects task family `{expected}`, found `{found}`")]
    FamilyMismatch {
        preset: &'static str,
        expected: &'static str,
        found: String,
    },
    #[error("invalid task JSON: {0}")]
    Json(#[from] TaskJsonError),
    #[error("expanded task is invalid: {0}")]
    Task(#[from] TaskRunnerError),
}

#[derive(Clone)]
struct PresetDefinition {
    id: &'static str,
    family: &'static str,
    purpose: &'static str,
    selectors: &'static [(&'static str, &'static str)],
    defaults: Value,
}

fn definitions() -> Vec<PresetDefinition> {
    vec![
        PresetDefinition {
            id: "ivp_rk45_default",
            family: "ivp",
            purpose: "IVP task using the RK45 nonstiff route and its native defaults.",
            selectors: &[("task.solver", "IVP"), ("task.method", "RK45")],
            defaults: json!({}),
        },
        PresetDefinition {
            id: "ivp_dop853_default",
            family: "ivp",
            purpose: "IVP task using the DOP853 nonstiff route and its native defaults.",
            selectors: &[("task.solver", "IVP"), ("task.method", "DOP853")],
            defaults: json!({}),
        },
        PresetDefinition {
            id: "ivp_bdf_sparse",
            family: "ivp",
            purpose: "IVP task using standalone BDF with a sparse linear layout.",
            selectors: &[("task.solver", "IVP"), ("task.method", "BDF")],
            defaults: json!({
                "solver_options": {
                    "rtol": 1.0e-3,
                    "atol": 1.0e-6,
                    "max_iterations": 1_000_000,
                    "matrix_layout": "Sparse"
                }
            }),
        },
        PresetDefinition {
            id: "bvp_sci_sparse",
            family: "bvp",
            purpose: "BVP_sci with sparse linear algebra and native defaults for other controls.",
            selectors: &[("task.solver", "BVP_sci"), ("task.method", "Sparse")],
            defaults: json!({}),
        },
        PresetDefinition {
            id: "bvp_damped_sparse",
            family: "bvp",
            purpose: "BVP_Damp Damped strategy using trapezoid discretization and sparse linear algebra.",
            selectors: &[
                ("task.solver", "BVP_Damp"),
                ("task.strategy", "Damped"),
                ("task.scheme", "trapezoid"),
                ("task.method", "Sparse"),
            ],
            defaults: json!({}),
        },
        PresetDefinition {
            id: "bvp_correction_newton_dense",
            family: "bvp",
            purpose: "Fixed-mesh BVP_CorrectionNewton with its supported dense route.",
            selectors: &[
                ("task.solver", "BVP_CorrectionNewton"),
                ("task.method", "Dense"),
            ],
            defaults: json!({}),
        },
        PresetDefinition {
            id: "nonlinear_root_newton",
            family: "nonlinear",
            purpose: "Square nonlinear root task using Newton with explicit standard limits.",
            selectors: &[("problem.formulation", "root"), ("solver.method", "newton")],
            defaults: json!({
                "solver": {
                    "tolerance": 1.0e-6,
                    "max_iterations": 100,
                    "telemetry": "off"
                }
            }),
        },
        PresetDefinition {
            id: "fitting_lm",
            family: "optimization",
            purpose: "Equation-model fitting using Levenberg-Marquardt and telemetry Off.",
            selectors: &[("model.method", "lm")],
            defaults: json!({ "solver": { "telemetry": "off" } }),
        },
    ]
}

/// Return the versioned preset catalog embedded in the Rust task API.
pub fn task_presets_json() -> Value {
    json!({
        "catalog_version": 1,
        "merge_policy": "Preset route selectors must match; omitted settings are filled, explicit tuning values are preserved.",
        "presets": definitions().into_iter().map(|preset| json!({
            "id": preset.id,
            "task_family": preset.family,
            "purpose": preset.purpose,
            "route": preset.selectors.iter().map(|(path, value)| json!({
                "path": path,
                "value": value,
            })).collect::<Vec<_>>(),
            "default_settings": preset.defaults,
            "model_aware_recommendation": false,
        })).collect::<Vec<_>>(),
    })
}

/// Expand a named preset into a task document and validate the result.
///
/// Preset route selectors are fixed: a conflicting explicit solver/method is
/// rejected. Other preset values act as defaults, so explicit numeric settings
/// in the supplied document always win. The expanded JSON contains no preset
/// metadata and remains a regular version-1 task document.
pub fn expand_task_preset_json(input: &str, preset_id: &str) -> Result<Value, TaskPresetError> {
    let preset = definitions()
        .into_iter()
        .find(|preset| preset.id == preset_id)
        .ok_or_else(|| TaskPresetError::UnknownPreset(preset_id.to_owned()))?;
    let mut task = json_value_from_json(input)?;
    if !task.is_object() {
        return Err(TaskPresetError::InvalidRoot);
    }

    for &(path, expected) in preset.selectors {
        set_selector(&mut task, path, expected, preset.id)?;
    }
    merge_missing(&mut task, &preset.defaults);

    let expanded = serde_json::to_string(&task).expect("JSON value serialization is infallible");
    let inspection = inspect_task_json(&expanded)?;
    if inspection.task_family != preset.family {
        return Err(TaskPresetError::FamilyMismatch {
            preset: preset.id,
            expected: preset.family,
            found: inspection.task_family,
        });
    }
    Ok(task)
}

fn set_selector(
    root: &mut Value,
    path: &'static str,
    expected: &'static str,
    preset: &'static str,
) -> Result<(), TaskPresetError> {
    let mut current = root;
    let mut parts = path.split('.').peekable();
    while let Some(part) = parts.next() {
        let Some(object) = current.as_object_mut() else {
            return Err(TaskPresetError::InvalidShape(path));
        };
        if parts.peek().is_none() {
            if let Some(existing) = object.get(part)
                && !existing
                    .as_str()
                    .is_some_and(|value| value.eq_ignore_ascii_case(expected))
            {
                return Err(TaskPresetError::RouteConflict {
                    preset,
                    path,
                    expected,
                    found: existing.to_string(),
                });
            }
            object.insert(part.to_owned(), Value::String(expected.to_owned()));
            return Ok(());
        }
        current = object
            .entry(part)
            .or_insert_with(|| Value::Object(Map::new()));
    }
    unreachable!("selector paths are non-empty")
}

fn merge_missing(target: &mut Value, defaults: &Value) {
    let (Some(target), Some(defaults)) = (target.as_object_mut(), defaults.as_object()) else {
        return;
    };
    for (key, default) in defaults {
        match target.get_mut(key) {
            Some(existing) => merge_missing(existing, default),
            None => {
                target.insert(key.clone(), default.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TaskPresetError, expand_task_preset_json, task_presets_json};
    use crate::command_interpreter::task_contract::inspect_task_json;
    use serde_json::{Value, json};

    const IVP: &str = include_str!("../agent_interop/examples/ivp_decay.json");
    const BVP_SCI: &str = include_str!("../agent_interop/examples/bvp_constant.json");
    const NONLINEAR: &str = include_str!("../agent_interop/examples/nonlinear_root.json");
    const FITTING: &str = include_str!("../agent_interop/examples/fitting_linear.json");

    #[test]
    fn catalog_is_versioned_and_has_unique_preset_ids() {
        let catalog = task_presets_json();
        assert_eq!(catalog["catalog_version"], 1);
        assert!(
            catalog["merge_policy"]
                .as_str()
                .unwrap()
                .contains("explicit tuning values are preserved")
        );
        let presets = catalog["presets"].as_array().unwrap();
        let ids = presets
            .iter()
            .map(|preset| preset["id"].as_str().unwrap())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(ids.len(), presets.len());
        assert!(
            presets
                .iter()
                .all(|preset| preset["model_aware_recommendation"] == false)
        );
    }

    #[test]
    fn every_catalog_default_is_present_after_expansion_and_route_preflight() {
        let catalog = task_presets_json();
        for preset in catalog["presets"].as_array().unwrap() {
            let id = preset["id"].as_str().unwrap();
            let family = preset["task_family"].as_str().unwrap();
            let mut task = match family {
                "ivp" => serde_json::from_str::<Value>(IVP).unwrap(),
                "bvp" => serde_json::from_str::<Value>(BVP_SCI).unwrap(),
                "nonlinear" => serde_json::from_str::<Value>(NONLINEAR).unwrap(),
                "optimization" => serde_json::from_str::<Value>(FITTING).unwrap(),
                other => panic!("unexpected preset family `{other}`"),
            };

            for selector in preset["route"].as_array().unwrap() {
                set_json_path(
                    &mut task,
                    selector["path"].as_str().unwrap(),
                    selector["value"].clone(),
                );
            }
            remove_default_fields(&mut task, &preset["default_settings"]);

            let expanded = expand_task_preset_json(&task.to_string(), id)
                .unwrap_or_else(|error| panic!("preset `{id}` failed expansion: {error}"));
            assert_eq!(
                inspect_task_json(&expanded.to_string())
                    .unwrap_or_else(|error| panic!("preset `{id}` failed preflight: {error}"))
                    .task_family,
                family,
                "preset `{id}` family drift"
            );
            assert_defaults_applied(
                &expanded,
                &preset["default_settings"],
                &format!("preset `{id}`"),
            );
        }
    }

    #[test]
    fn preset_expansion_validates_representative_families() {
        for (task, method, preset, family) in [
            (IVP, Some("RK45"), "ivp_rk45_default", "ivp"),
            (IVP, Some("DOP853"), "ivp_dop853_default", "ivp"),
            (IVP, Some("BDF"), "ivp_bdf_sparse", "ivp"),
            (BVP_SCI, None, "bvp_sci_sparse", "bvp"),
            (NONLINEAR, None, "nonlinear_root_newton", "nonlinear"),
            (FITTING, None, "fitting_lm", "optimization"),
        ] {
            let task = if let Some(method) = method {
                let mut value: Value = serde_json::from_str(task).unwrap();
                value["task"]["method"] = json!(method);
                value.to_string()
            } else {
                task.to_owned()
            };
            let expanded = expand_task_preset_json(&task, preset).unwrap();
            let expanded = serde_json::to_string(&expanded).unwrap();
            let inspection = inspect_task_json(&expanded).unwrap();
            assert_eq!(inspection.task_family, family, "preset {preset}");
        }
    }

    #[test]
    fn all_bvp_route_presets_validate_on_their_declared_route() {
        let mut damp: Value = serde_json::from_str(BVP_SCI).unwrap();
        damp["task"]["solver"] = json!("BVP_Damp");
        damp["task"]["strategy"] = json!("Damped");
        damp["task"]["scheme"] = json!("trapezoid");
        let expanded = expand_task_preset_json(&damp.to_string(), "bvp_damped_sparse").unwrap();
        assert_eq!(
            inspect_task_json(&expanded.to_string())
                .unwrap()
                .task_family,
            "bvp"
        );

        let mut correction: Value = serde_json::from_str(BVP_SCI).unwrap();
        correction["task"]["solver"] = json!("BVP_CorrectionNewton");
        correction["task"]["method"] = json!("Dense");
        let expanded =
            expand_task_preset_json(&correction.to_string(), "bvp_correction_newton_dense")
                .unwrap();
        assert_eq!(
            inspect_task_json(&expanded.to_string())
                .unwrap()
                .task_family,
            "bvp"
        );
    }

    #[test]
    fn dop853_preset_runs_through_the_json_ivp_adapter() {
        let mut task: Value = serde_json::from_str(IVP).unwrap();
        task["task"]["method"] = json!("DOP853");
        let expanded = expand_task_preset_json(&task.to_string(), "ivp_dop853_default").unwrap();
        let response = crate::command_interpreter::task_contract::run_task_json_envelope(
            &expanded.to_string(),
        );
        assert_eq!(response["status"], "succeeded", "{response}");
        assert_eq!(response["result"]["solver"], "dop853");
    }

    #[test]
    fn preset_fills_missing_defaults_but_preserves_explicit_tuning() {
        let mut task: Value = serde_json::from_str(IVP).unwrap();
        task["task"]["method"] = json!("BDF");
        task["solver_options"] = json!({ "rtol": 0.02 });
        let expanded = expand_task_preset_json(&task.to_string(), "ivp_bdf_sparse").unwrap();
        assert_eq!(expanded["solver_options"]["rtol"], 0.02);
        assert_eq!(expanded["solver_options"]["atol"], 1.0e-6);
        assert_eq!(expanded["solver_options"]["max_iterations"], 1_000_000);
        assert_eq!(expanded["solver_options"]["matrix_layout"], "Sparse");
    }

    #[test]
    fn conflicting_routes_and_unknown_presets_are_typed_errors() {
        let conflict = expand_task_preset_json(IVP, "ivp_dop853_default").unwrap_err();
        assert!(matches!(conflict, TaskPresetError::RouteConflict { .. }));

        let unknown = expand_task_preset_json(IVP, "not-a-preset").unwrap_err();
        assert!(matches!(unknown, TaskPresetError::UnknownPreset(_)));
    }

    #[test]
    fn task_capability_response_embeds_the_same_preset_catalog() {
        let capabilities = crate::command_interpreter::task_contract::task_capabilities_json();
        let presets = task_presets_json();
        assert_eq!(capabilities["preset_catalog"], presets);
    }

    #[test]
    fn malformed_or_wrong_shaped_input_never_returns_unvalidated_json() {
        let invalid = r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"}}"#;
        assert!(matches!(
            expand_task_preset_json(invalid, "ivp_rk45_default"),
            Err(TaskPresetError::Task(_))
        ));
        assert!(matches!(
            expand_task_preset_json("[]", "ivp_rk45_default"),
            Err(TaskPresetError::InvalidRoot)
        ));
    }

    fn set_json_path(root: &mut Value, path: &str, value: Value) {
        let mut parts = path.split('.').peekable();
        let mut current = root;
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                current[part] = value;
                return;
            }
            if !current[part].is_object() {
                current[part] = json!({});
            }
            current = &mut current[part];
        }
    }

    fn assert_defaults_applied(actual: &Value, defaults: &Value, path: &str) {
        let Some(defaults) = defaults.as_object() else {
            assert_eq!(
                actual, defaults,
                "missing or changed preset default at {path}"
            );
            return;
        };
        for (key, default) in defaults {
            let child_path = format!("{path}.{key}");
            let actual_value = actual
                .get(key)
                .unwrap_or_else(|| panic!("preset default missing at {child_path}"));
            assert_defaults_applied(actual_value, default, &child_path);
        }
    }

    fn remove_default_fields(target: &mut Value, defaults: &Value) {
        let (Some(target), Some(defaults)) = (target.as_object_mut(), defaults.as_object()) else {
            return;
        };
        for (key, default) in defaults {
            if default.is_object() {
                if let Some(value) = target.get_mut(key) {
                    remove_default_fields(value, default);
                }
            } else {
                target.remove(key);
            }
        }
    }
}
