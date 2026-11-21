//! Strict JSON wire-format adapter for native task documents.
//!
//! The JSON shape requires a root `format_version: 1`; remaining top-level
//! objects mirror the existing task DSL sections. Each section maps field
//! names to scalar or array values. Nested objects are rejected because
//! [`DocumentMap`] has no nested-object representation.

use crate::command_interpreter::task_parser::{DocumentMap, Value};
use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value as JsonValue};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Draft 2020-12 schema for the versioned task JSON accepted by the runner.
/// This is embedded to make schema discovery available without file I/O.
pub const TASK_DOCUMENT_JSON_SCHEMA: &str = include_str!("task_document.schema.json");

#[derive(Debug)]
pub enum TaskJsonError {
    InvalidJson(serde_json::Error),
    MissingFormatVersion,
    InvalidFormatVersion,
    UnsupportedFormatVersion {
        found: u64,
        supported: u64,
    },
    InvalidShape {
        path: String,
        expected: &'static str,
    },
    UnsupportedValue {
        path: String,
        kind: &'static str,
    },
    UnknownField {
        path: String,
    },
    NumberOutOfRange {
        path: String,
    },
}

impl fmt::Display for TaskJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(error) => write!(f, "invalid task JSON: {error}"),
            Self::MissingFormatVersion => {
                write!(
                    f,
                    "task JSON is missing the required root `format_version: 1`"
                )
            }
            Self::InvalidFormatVersion => {
                write!(
                    f,
                    "task JSON root `format_version` must be an unsigned integer"
                )
            }
            Self::UnsupportedFormatVersion { found, supported } => write!(
                f,
                "unsupported task JSON format_version {found}; this build supports {supported}"
            ),
            Self::InvalidShape { path, expected } => {
                write!(f, "invalid task JSON at `{path}`: expected {expected}")
            }
            Self::UnsupportedValue { path, kind } => {
                write!(f, "unsupported task JSON value at `{path}`: {kind}")
            }
            Self::UnknownField { path } => write!(f, "unknown task JSON field `{path}`"),
            Self::NumberOutOfRange { path } => {
                write!(
                    f,
                    "task JSON number at `{path}` is outside the supported range"
                )
            }
        }
    }
}

impl Error for TaskJsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidJson(error) => Some(error),
            _ => None,
        }
    }
}

/// Convert task JSON into the normalized map consumed by all family parsers.
pub fn document_map_from_json(input: &str) -> Result<DocumentMap, TaskJsonError> {
    let value = json_value_from_json(input)?;
    document_map_from_value(&value)
}

/// Parse versioned JSON while converting only the named settings sections.
/// Other top-level fields belong to the hosting client's domain model and are
/// deliberately left opaque, including nested objects that `DocumentMap`
/// cannot represent.
pub(crate) fn settings_document_map_from_json(
    input: &str,
    settings_sections: &[&str],
) -> Result<DocumentMap, TaskJsonError> {
    let value = json_value_from_json(input)?;
    let root = value
        .as_object()
        .ok_or_else(|| TaskJsonError::InvalidShape {
            path: "$".to_string(),
            expected: "an object containing format_version and solver settings",
        })?;
    let mut selected = Map::with_capacity(settings_sections.len() + 1);
    if let Some(version) = root.get("format_version") {
        selected.insert("format_version".into(), version.clone());
    }
    for section in settings_sections {
        if let Some(value) = root.get(*section) {
            selected.insert((*section).into(), value.clone());
        }
    }
    document_map_from_value(&JsonValue::Object(selected))
}

pub(crate) fn json_value_from_json(input: &str) -> Result<JsonValue, TaskJsonError> {
    let mut deserializer = serde_json::Deserializer::from_str(input);
    let UniqueJsonValue(value) =
        UniqueJsonValue::deserialize(&mut deserializer).map_err(TaskJsonError::InvalidJson)?;
    deserializer.end().map_err(TaskJsonError::InvalidJson)?;
    Ok(value)
}

/// `serde_json::Value` intentionally keeps the last duplicate object key. Task
/// documents instead reject duplicates so a typo cannot silently override a
/// solver setting or equation.
struct UniqueJsonValue(JsonValue);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonValueVisitor)
    }
}

struct UniqueJsonValueVisitor;

impl<'de> Visitor<'de> for UniqueJsonValueVisitor {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(JsonValue::Null))
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(JsonValue::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(JsonValue::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(JsonValue::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(JsonValue::Number)
            .map(UniqueJsonValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(JsonValue::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(JsonValue::String(value)))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(UniqueJsonValue(value)) = seq.next_element()? {
            values.push(value);
        }
        Ok(UniqueJsonValue(JsonValue::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::with_capacity(map.size_hint().unwrap_or(0));
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key `{key}`"
                )));
            }
            let UniqueJsonValue(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueJsonValue(JsonValue::Object(values)))
    }
}

/// Convert a parsed JSON value into the existing task-parser representation.
pub fn document_map_from_value(value: &JsonValue) -> Result<DocumentMap, TaskJsonError> {
    let root = value
        .as_object()
        .ok_or_else(|| TaskJsonError::InvalidShape {
            path: "$".to_string(),
            expected: "an object of task sections",
        })?;

    let version_value = root
        .get("format_version")
        .ok_or(TaskJsonError::MissingFormatVersion)?;
    let version = version_value
        .as_u64()
        .or_else(|| {
            version_value
                .as_f64()
                .filter(|value| value.is_finite() && value.fract() == 0.0 && *value >= 0.0)
                .and_then(|value| (value < 18_446_744_073_709_551_616.0).then_some(value as u64))
        })
        .ok_or(TaskJsonError::InvalidFormatVersion)?;
    if version != 1 {
        return Err(TaskJsonError::UnsupportedFormatVersion {
            found: version,
            supported: 1,
        });
    }

    let mut document = DocumentMap::with_capacity(root.len().saturating_sub(1));
    for (section_name, section_value) in root {
        if section_name == "format_version" {
            continue;
        }
        let section_path = format!("$.{section_name}");
        let section = section_value
            .as_object()
            .ok_or_else(|| TaskJsonError::InvalidShape {
                path: section_path.clone(),
                expected: "an object mapping field names to values",
            })?;
        let fields = section_map_from_json(section, &section_path)?;
        document.insert(section_name.clone(), fields);
    }
    Ok(document)
}

fn section_map_from_json(
    section: &Map<String, JsonValue>,
    section_path: &str,
) -> Result<HashMap<String, Option<Vec<Value>>>, TaskJsonError> {
    let mut fields = HashMap::with_capacity(section.len());
    for (field_name, json_value) in section {
        let path = format!("{section_path}.{field_name}");
        if matches!(section_path, "$.model" | "$.problem") && field_name == "substitutions" {
            let definitions =
                json_value
                    .as_object()
                    .ok_or_else(|| TaskJsonError::InvalidShape {
                        path: path.clone(),
                        expected: "an object mapping substitution names to expression strings",
                    })?;
            let values = definitions
                .iter()
                .map(|(name, expression)| {
                    let expression =
                        expression
                            .as_str()
                            .ok_or_else(|| TaskJsonError::InvalidShape {
                                path: format!("{path}.{name}"),
                                expected: "a symbolic expression string",
                            })?;
                    Ok(Value::String(format!("{name}={expression}")))
                })
                .collect::<Result<Vec<_>, TaskJsonError>>()?;
            fields.insert(field_name.clone(), Some(values));
            continue;
        }
        let values = match json_value {
            JsonValue::Null => vec![Value::Optional(None)],
            JsonValue::Array(items) => array_values(items, &path)?,
            scalar => vec![scalar_value(scalar, &path)?],
        };
        fields.insert(field_name.clone(), Some(values));
    }
    Ok(fields)
}

fn array_values(items: &[JsonValue], path: &str) -> Result<Vec<Value>, TaskJsonError> {
    if items.iter().all(JsonValue::is_number) {
        let mut vector = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            let item_path = format!("{path}[{index}]");
            let number = item.as_number().expect("all items were checked as numbers");
            vector.push(number_as_f64(number, &item_path)?);
        }
        return Ok(vec![Value::Vector(vector)]);
    }

    if let Some(first) = items.first() {
        let first_kind = json_scalar_kind(first);
        if items
            .iter()
            .any(|item| json_scalar_kind(item) != first_kind)
        {
            return Err(TaskJsonError::InvalidShape {
                path: path.to_string(),
                expected: "a homogeneous array of scalar values",
            });
        }
    }

    items
        .iter()
        .enumerate()
        .map(|(index, item)| scalar_value(item, &format!("{path}[{index}]")))
        .collect()
}

fn json_scalar_kind(value: &JsonValue) -> u8 {
    match value {
        JsonValue::Null => 0,
        JsonValue::Bool(_) => 1,
        JsonValue::Number(_) => 2,
        JsonValue::String(_) => 3,
        JsonValue::Array(_) => 4,
        JsonValue::Object(_) => 5,
    }
}

fn scalar_value(value: &JsonValue, path: &str) -> Result<Value, TaskJsonError> {
    match value {
        JsonValue::Null => Ok(Value::Optional(None)),
        JsonValue::Bool(value) => Ok(Value::Boolean(*value)),
        JsonValue::Number(number) => number_value(number, path),
        JsonValue::String(value) => Ok(Value::String(value.clone())),
        JsonValue::Array(_) => Err(TaskJsonError::UnsupportedValue {
            path: path.to_string(),
            kind: "nested arrays are not supported",
        }),
        JsonValue::Object(_) => Err(TaskJsonError::UnsupportedValue {
            path: path.to_string(),
            kind: "nested objects are not supported",
        }),
    }
}

fn number_value(number: &Number, path: &str) -> Result<Value, TaskJsonError> {
    if let Some(value) = number.as_i64() {
        return Ok(Value::Integer(value));
    }
    if let Some(value) = number.as_u64() {
        return usize::try_from(value).map(Value::Usize).map_err(|_| {
            TaskJsonError::NumberOutOfRange {
                path: path.to_string(),
            }
        });
    }
    number
        .as_f64()
        .filter(|value| value.is_finite())
        .map(Value::Float)
        .ok_or_else(|| TaskJsonError::NumberOutOfRange {
            path: path.to_string(),
        })
}

fn number_as_f64(number: &Number, path: &str) -> Result<f64, TaskJsonError> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| TaskJsonError::NumberOutOfRange {
            path: path.to_string(),
        })?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_sections_scalars_vectors_lists_and_nulls() {
        let map = document_map_from_json(
            r#"{
                "format_version": 1,
                "task": {"solver": "IVP", "schema_version": 1},
                "equations": {"arg": "t", "y": ["-y", "-2*y"]},
                "initial_conditions": {"y0": [1.0, 2.0], "optional": null}
            }"#,
        )
        .unwrap();

        assert_eq!(
            map["task"]["solver"],
            Some(vec![Value::String("IVP".to_string())])
        );
        assert_eq!(map["task"]["schema_version"], Some(vec![Value::Integer(1)]));
        assert_eq!(
            map["equations"]["y"],
            Some(vec![
                Value::String("-y".to_string()),
                Value::String("-2*y".to_string())
            ])
        );
        assert_eq!(
            map["initial_conditions"]["y0"],
            Some(vec![Value::Vector(vec![1.0, 2.0])])
        );
        assert_eq!(
            map["initial_conditions"]["optional"],
            Some(vec![Value::Optional(None)])
        );
    }

    #[test]
    fn rejects_non_object_sections_and_nested_field_values_with_paths() {
        let error =
            document_map_from_json(r#"{"format_version":1,"problem": {"equations": {"x": 1}}}"#)
                .unwrap_err();
        assert!(error.to_string().contains("$.problem.equations"));

        let error = document_map_from_json(r#"{"format_version":1,"problem": []}"#).unwrap_err();
        assert!(error.to_string().contains("$.problem"));

        let error =
            document_map_from_json(r#"{"format_version":1,"problem":{"variables":["x",1]}}"#)
                .unwrap_err();
        assert!(error.to_string().contains("homogeneous array"));
    }

    #[test]
    fn preserves_unsigned_integer_values_without_signed_wrap() {
        let map = document_map_from_json(
            r#"{"format_version":1,"solver": {"max_iterations": 18446744073709551615}}"#,
        )
        .unwrap();
        assert_eq!(
            map["solver"]["max_iterations"],
            Some(vec![Value::Usize(usize::MAX)])
        );
    }

    #[test]
    fn converts_optimization_substitution_object_to_native_string_entries() {
        let map = document_map_from_json(
            r#"{"format_version":1,"data":{"x":[0,1],"y":[0,1]},"model":{"method":"lm","parameters":["a"],"initial_guess":[0],"equations":["model"],"substitutions":{"model":"a*x"}}}"#,
        )
        .unwrap();
        assert_eq!(
            map["model"]["substitutions"],
            Some(vec![Value::String("model=a*x".into())])
        );
    }

    #[test]
    fn converts_nonlinear_substitution_object_to_native_string_entries() {
        let map = document_map_from_json(
            r#"{"format_version":1,"problem":{"formulation":"root","variables":["x","y"],"equations":["sum-3","x-y-1"],"substitutions":{"sum":"x+y"}}}"#,
        )
        .unwrap();
        assert_eq!(
            map["problem"]["substitutions"],
            Some(vec![Value::String("sum=x+y".into())])
        );
    }

    #[test]
    fn rejects_non_string_nonlinear_substitution_values_with_json_path() {
        let error = document_map_from_json(
            r#"{"format_version":1,"problem":{"formulation":"root","variables":["x"],"equations":["x-1"],"substitutions":{"alias":1}}}"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("$.problem.substitutions.alias"));
    }

    #[test]
    fn rejects_non_string_optimization_substitution_values_with_json_path() {
        let error = document_map_from_json(
            r#"{"format_version":1,"data":{"x":[0,1],"y":[0,1]},"model":{"method":"lm","parameters":["a"],"initial_guess":[0],"equations":["model"],"substitutions":{"model":1}}}"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("$.model.substitutions.model"));
    }

    #[test]
    fn requires_supported_json_format_version() {
        assert!(matches!(
            document_map_from_json(r#"{"problem":{}}"#),
            Err(TaskJsonError::MissingFormatVersion)
        ));
        assert!(matches!(
            document_map_from_json(r#"{"format_version":1.5,"problem":{}}"#),
            Err(TaskJsonError::InvalidFormatVersion)
        ));
        assert!(matches!(
            document_map_from_json(r#"{"format_version":2,"problem":{}}"#),
            Err(TaskJsonError::UnsupportedFormatVersion {
                found: 2,
                supported: 1
            })
        ));
    }

    #[test]
    fn rejects_duplicate_keys_in_root_and_nested_objects() {
        for input in [
            r#"{"format_version":1,"format_version":1,"task":{}}"#,
            r#"{"format_version":1,"task":{"solver":"IVP","solver":"BVP"}}"#,
        ] {
            let error = document_map_from_json(input).unwrap_err();
            assert!(error.to_string().contains("duplicate JSON object key"));
        }
    }
}
