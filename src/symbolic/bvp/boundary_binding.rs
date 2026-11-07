//! Stable runtime slots for numeric values of an unchanged BVP boundary schema.

use std::collections::HashMap;

/// One endpoint value in the fixed boundary-condition ABI.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BvpBoundaryValueSlot {
    /// Index into the solver's state-variable list.
    pub(crate) variable_index: usize,
    /// `0` denotes the left endpoint and `1` the right endpoint.
    pub(crate) endpoint: usize,
    /// Full node-major position in the discretized state.
    pub(crate) full_position: usize,
    /// Reserved evaluator input name for this numeric value.
    pub(crate) parameter_name: String,
    /// Initial/current numeric value.
    pub(crate) value: f64,
}

/// Ordered BC input ABI. Its slot order is independent of `HashMap` iteration.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct BvpBoundaryValueSchema {
    slots: Vec<BvpBoundaryValueSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BvpBoundaryValueSchemaError {
    InvalidMeshSize(usize),
    InvalidEndpoint { variable: String, endpoint: usize },
    DuplicateEndpoint { variable: String, endpoint: usize },
    ReservedNameCollision(String),
}

impl std::fmt::Display for BvpBoundaryValueSchemaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMeshSize(size) => write!(
                formatter,
                "runtime BVP boundary binding requires at least two mesh points, got {size}"
            ),
            Self::InvalidEndpoint { variable, endpoint } => write!(
                formatter,
                "boundary endpoint {endpoint} for `{variable}` must be 0 (left) or 1 (right)"
            ),
            Self::DuplicateEndpoint { variable, endpoint } => write!(
                formatter,
                "boundary schema contains duplicate endpoint {endpoint} for `{variable}`"
            ),
            Self::ReservedNameCollision(name) => write!(
                formatter,
                "symbolic input `{name}` collides with a reserved BVP boundary slot"
            ),
        }
    }
}

impl std::error::Error for BvpBoundaryValueSchemaError {}

impl BvpBoundaryValueSchema {
    /// Creates deterministic slots in state-variable order, then left/right order.
    pub(crate) fn new(
        variables: &[String],
        mesh_points: usize,
        conditions: &HashMap<String, Vec<(usize, f64)>>,
        occupied_names: impl IntoIterator<Item = String>,
    ) -> Result<Self, BvpBoundaryValueSchemaError> {
        if mesh_points < 2 {
            return Err(BvpBoundaryValueSchemaError::InvalidMeshSize(mesh_points));
        }

        let occupied_names = occupied_names
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let mut slots = Vec::new();
        for (variable_index, variable) in variables.iter().enumerate() {
            let Some(entries) = conditions.get(variable) else {
                continue;
            };
            let mut by_endpoint = [None, None];
            for &(endpoint, value) in entries {
                let Some(slot) = by_endpoint.get_mut(endpoint) else {
                    return Err(BvpBoundaryValueSchemaError::InvalidEndpoint {
                        variable: variable.clone(),
                        endpoint,
                    });
                };
                if slot.replace(value).is_some() {
                    return Err(BvpBoundaryValueSchemaError::DuplicateEndpoint {
                        variable: variable.clone(),
                        endpoint,
                    });
                }
            }
            for (endpoint, value) in by_endpoint.into_iter().enumerate() {
                let Some(value) = value else { continue };
                let slot_index = slots.len();
                let parameter_name = format!("__rst_bvp_boundary_value_{slot_index}");
                if occupied_names.contains(&parameter_name) {
                    return Err(BvpBoundaryValueSchemaError::ReservedNameCollision(
                        parameter_name,
                    ));
                }
                let node = if endpoint == 0 { 0 } else { mesh_points - 1 };
                slots.push(BvpBoundaryValueSlot {
                    variable_index,
                    endpoint,
                    full_position: node * variables.len() + variable_index,
                    parameter_name,
                    value,
                });
            }
        }
        Ok(Self { slots })
    }

    pub(crate) fn slots(&self) -> &[BvpBoundaryValueSlot] {
        &self.slots
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.slots.iter().map(|slot| slot.parameter_name.as_str())
    }

    pub(crate) fn values(&self) -> Vec<f64> {
        self.slots.iter().map(|slot| slot.value).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{BvpBoundaryValueSchema, BvpBoundaryValueSchemaError};
    use std::collections::HashMap;

    fn conditions(entries: &[(&str, &[(usize, f64)])]) -> HashMap<String, Vec<(usize, f64)>> {
        entries
            .iter()
            .map(|(name, values)| ((*name).to_string(), values.to_vec()))
            .collect()
    }

    #[test]
    fn boundary_slots_have_stable_state_then_endpoint_order() {
        let variables = vec!["y".to_string(), "z".to_string()];
        let conditions = conditions(&[("z", &[(1, 4.0)]), ("y", &[(1, 2.0), (0, 1.0)])]);
        let schema =
            BvpBoundaryValueSchema::new(&variables, 5, &conditions, Vec::<String>::new()).unwrap();

        assert_eq!(schema.values(), vec![1.0, 2.0, 4.0]);
        assert_eq!(
            schema.names().collect::<Vec<_>>(),
            vec![
                "__rst_bvp_boundary_value_0",
                "__rst_bvp_boundary_value_1",
                "__rst_bvp_boundary_value_2"
            ]
        );
        assert_eq!(
            schema
                .slots()
                .iter()
                .map(|slot| (slot.variable_index, slot.endpoint, slot.full_position))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0), (0, 1, 8), (1, 1, 9)]
        );
    }

    #[test]
    fn boundary_schema_rejects_invalid_or_ambiguous_topology() {
        let variables = vec!["y".to_string()];
        let invalid = conditions(&[("y", &[(2, 1.0)])]);
        assert!(matches!(
            BvpBoundaryValueSchema::new(&variables, 3, &invalid, Vec::<String>::new()),
            Err(BvpBoundaryValueSchemaError::InvalidEndpoint { .. })
        ));

        let duplicate = conditions(&[("y", &[(0, 1.0), (0, 2.0)])]);
        assert!(matches!(
            BvpBoundaryValueSchema::new(&variables, 3, &duplicate, Vec::<String>::new()),
            Err(BvpBoundaryValueSchemaError::DuplicateEndpoint { .. })
        ));
    }
}
