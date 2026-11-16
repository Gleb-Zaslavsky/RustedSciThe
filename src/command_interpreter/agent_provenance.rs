//! Stable task identity and runtime provenance for worker-owned artifacts.
//!
//! Model identity excludes values intended to vary between solves while
//! preserving their names and shapes. Run identity hashes the complete task
//! and is emitted only when the caller explicitly requests it.

use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

const IDENTITY_VERSION: u32 = 1;
const DIGEST_ALGORITHM: &str = "blake3-256";
const CARGO_LOCK_BYTES: &[u8] = include_bytes!("../../Cargo.lock");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotCompilerIdentity {
    pub toolchain: String,
    pub version: String,
}

/// Construct explicit provenance and a cache identity from validated task
/// inputs and runtime/build identity.
pub fn task_provenance(
    task: &Value,
    inspection: &Value,
    include_run_digest: bool,
    compiler: Option<&AotCompilerIdentity>,
) -> (Value, String) {
    let model = normalized_model_task(task);
    let defaults = &inspection["effective_defaults"];
    let model_digest = digest(
        b"rst-agent-model-v1\0",
        &json!({
            "identity_version": IDENTITY_VERSION,
            "task": model,
            "effective_defaults": defaults,
        }),
    );
    let run_digest = include_run_digest.then(|| {
        digest(
            b"rst-agent-run-v1\0",
            &json!({"model_digest": model_digest, "task": canonicalize(task)}),
        )
    });
    let runtime = runtime_identity(compiler);
    let cache_key = digest(
        b"rst-agent-aot-cache-v1\0",
        &json!({
            "model_digest": model_digest,
            "runtime": runtime,
        }),
    );
    let provenance = json!({
        "schema_version": IDENTITY_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "model_digest": model_digest,
        "run_identity": {
            "availability": if include_run_digest { "included" } else { "not_requested" },
            "digest": run_digest,
        },
        "runtime": runtime,
    });
    (provenance, cache_key)
}

fn runtime_identity(compiler: Option<&AotCompilerIdentity>) -> Value {
    let cargo_lock_digest = blake3::hash(CARGO_LOCK_BYTES).to_hex().to_string();
    json!({
        "rst_version": env!("CARGO_PKG_VERSION"),
        "target_os": std::env::consts::OS,
        "target_arch": std::env::consts::ARCH,
        "target_pointer_width": usize::BITS,
        "target_endian": if cfg!(target_endian = "little") { "little" } else { "big" },
        "target_env": if cfg!(target_env = "msvc") { "msvc" } else if cfg!(target_env = "gnu") { "gnu" } else { "unknown" },
        "debug_assertions": cfg!(debug_assertions),
        "cargo_lock_digest": cargo_lock_digest,
        "aot_compiler": compiler.map(|compiler| json!({
            "toolchain": compiler.toolchain,
            "version": compiler.version,
        })),
    })
}

fn normalized_model_task(task: &Value) -> Value {
    let mut model = canonicalize(task);
    let family = task["task"]["solver"]
        .as_str()
        .unwrap_or_default()
        .to_ascii_lowercase();
    match family.as_str() {
        "ivp" => {
            normalize_field(&mut model, "parameters");
            normalize_field(&mut model, "initial_conditions");
            normalize_field(&mut model, "continuation");
            normalize_equation_parameter_values(&mut model);
        }
        "bvp_sci" | "bvp_damp" | "bvp_correctionnewton" | "bvp_correction_newton" => {
            normalize_field(&mut model, "parameters");
            normalize_field(&mut model, "boundary_conditions");
            normalize_field(&mut model, "initial_guess");
            normalize_field(&mut model, "continuation");
            normalize_mesh_values(&mut model);
            normalize_equation_parameter_values(&mut model);
        }
        _ => match task["problem"]["formulation"].as_str() {
            Some("root") | Some("least_squares") => {
                normalize_nested_field(&mut model, "problem", "initial_guess");
            }
            _ => {
                normalize_field(&mut model, "data");
                normalize_nested_field(&mut model, "model", "initial_guess");
            }
        },
    }
    canonicalize(&model)
}

fn normalize_field(root: &mut Value, field: &str) {
    if let Some(value) = root.get_mut(field) {
        *value = normalize_numeric_shape(value);
    }
}

fn normalize_nested_field(root: &mut Value, section: &str, field: &str) {
    if let Some(value) = root.get_mut(section).and_then(|value| value.get_mut(field)) {
        *value = normalize_numeric_shape(value);
    }
}

fn normalize_mesh_values(root: &mut Value) {
    let Some(mesh) = root.get_mut("mesh").and_then(Value::as_object_mut) else {
        return;
    };
    for (name, value) in mesh.iter_mut() {
        if !matches!(name.as_str(), "n_steps" | "points" | "nodes") {
            *value = normalize_numeric_shape(value);
        }
    }
}

fn normalize_equation_parameter_values(root: &mut Value) {
    if let Some(values) = root
        .get_mut("equations")
        .and_then(Value::as_object_mut)
        .and_then(|equations| equations.get_mut("parameter_values"))
    {
        *values = normalize_numeric_shape(values);
    }
}

fn normalize_numeric_shape(value: &Value) -> Value {
    match value {
        Value::Number(_) => Value::String("<numeric>".into()),
        Value::Array(values) => Value::Array(values.iter().map(normalize_numeric_shape).collect()),
        Value::Object(values) => {
            let sorted = values
                .iter()
                .map(|(key, value)| (key.clone(), normalize_numeric_shape(value)))
                .collect::<BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect::<Map<_, _>>())
        }
        _ => value.clone(),
    }
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        Value::Object(values) => {
            let sorted = values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect::<BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect::<Map<_, _>>())
        }
        _ => value.clone(),
    }
}

fn digest(domain: &[u8], value: &Value) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&serde_json::to_vec(&canonicalize(value)).expect("JSON value serializes"));
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::{AotCompilerIdentity, task_provenance};
    use serde_json::json;

    #[test]
    fn model_identity_is_stable_and_separate_from_run_values() {
        let first = json!({
            "format_version": 1,
            "task": {"solver":"IVP", "method":"BDF"},
            "equations": {"arg":"t", "y":"-a*y", "parameters":["a"], "parameter_values":[1.0]},
            "parameters": {"a": 1.0},
            "initial_conditions": {"t0":0.0, "t_end":1.0, "y0":2.0}
        });
        let second = json!({
            "initial_conditions": {"t_end":1.0, "y0":3.0, "t0":0.0},
            "parameters": {"a": 4.0},
            "equations": {"parameter_values":[4.0], "parameters":["a"], "y":"-a*y", "arg":"t"},
            "task": {"method":"BDF", "solver":"IVP"},
            "format_version": 1
        });
        let inspection = json!({"effective_defaults":{"matrix_layout":"Dense"}});
        let (first_provenance, first_key) = task_provenance(&first, &inspection, true, None);
        let (second_provenance, second_key) = task_provenance(&second, &inspection, true, None);
        assert_eq!(
            first_provenance["model_digest"],
            second_provenance["model_digest"]
        );
        assert_eq!(first_key, second_key);
        assert_ne!(
            first_provenance["run_identity"]["digest"],
            second_provenance["run_identity"]["digest"]
        );

        let (without_run, _) = task_provenance(&first, &inspection, false, None);
        assert_eq!(without_run["run_identity"]["availability"], "not_requested");
        assert!(without_run["run_identity"]["digest"].is_null());
    }

    #[test]
    fn cache_identity_includes_compiler_and_build_runtime_fingerprint() {
        let task = json!({"format_version":1,"task":{"solver":"IVP"}});
        let inspection = json!({});
        let compiler = AotCompilerIdentity {
            toolchain: "tcc".into(),
            version: "TinyCC 0.9.27".into(),
        };
        let (_, key_a) = task_provenance(&task, &inspection, false, Some(&compiler));
        let changed = AotCompilerIdentity {
            version: "TinyCC 0.9.28".into(),
            ..compiler
        };
        let (_, key_b) = task_provenance(&task, &inspection, false, Some(&changed));
        assert_ne!(key_a, key_b);
    }
}
