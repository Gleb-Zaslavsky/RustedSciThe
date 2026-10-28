//! Structural utilities shared by symbolic solver frontends.

use super::symbolic_engine::Expr;
use std::collections::HashMap;

/// Infer the conservative lower/upper bandwidth of a row-wise symbolic system.
///
/// Only named state-variable dependencies participate. Independent variables
/// and parameters are intentionally ignored; no numerical probing or dense
/// Jacobian construction is performed.
pub(crate) fn infer_rowwise_bandwidth(
    equations: &[Expr],
    state_names: &[String],
) -> Result<(usize, usize), String> {
    if equations.is_empty() || equations.len() != state_names.len() {
        return Err(format!(
            "bandwidth inference needs one equation per state ({} equations, {} states)",
            equations.len(),
            state_names.len()
        ));
    }

    let mut lower = 0usize;
    let mut upper = 0usize;
    let columns = state_names
        .iter()
        .enumerate()
        .map(|(column, name)| (name.as_str(), column))
        .collect::<HashMap<_, _>>();
    for (row, equation) in equations.iter().enumerate() {
        for variable in equation.all_arguments_are_variables() {
            if let Some(&column) = columns.get(variable.as_str()) {
                if row >= column {
                    lower = lower.max(row - column);
                } else {
                    upper = upper.max(column - row);
                }
            }
        }
    }
    Ok((lower, upper))
}

/// Infer a conservative bandwidth from a row-major symbolic Jacobian.
pub(crate) fn infer_jacobian_bandwidth(
    jacobian: &[Expr],
    dimension: usize,
    state_names: &[String],
) -> Result<(usize, usize), String> {
    let expected = dimension.checked_mul(dimension).ok_or_else(|| {
        "symbolic Jacobian dimension overflows while inferring bandwidth".to_owned()
    })?;
    if dimension != state_names.len() || jacobian.len() != expected {
        return Err(format!(
            "bandwidth inference needs a {dimension}x{dimension} Jacobian and {dimension} state names"
        ));
    }

    let mut lower = 0usize;
    let mut upper = 0usize;
    for (index, derivative) in jacobian.iter().enumerate() {
        if derivative.is_zero() {
            continue;
        }
        let row = index / dimension;
        let column = index % dimension;
        if row >= column {
            lower = lower.max(row - column);
        } else {
            upper = upper.max(column - row);
        }
    }
    Ok((lower, upper))
}

/// Reject explicit widths that would discard structurally nonzero entries.
pub(crate) fn validate_bandwidth_hint(
    inferred: (usize, usize),
    hint: (usize, usize),
) -> Result<(), String> {
    if hint.0 < inferred.0 || hint.1 < inferred.1 {
        return Err(format!(
            "explicit band ({}, {}) is narrower than the inferred symbolic pattern ({}, {})",
            hint.0, hint.1, inferred.0, inferred.1
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_asymmetric_rowwise_band_without_dense_jacobian() {
        let equations = [
            Expr::parse_expression("y0 + y1 + t"),
            Expr::parse_expression("y0 + y1 + y2"),
            Expr::parse_expression("y1 + y2"),
        ];
        let names = ["y0", "y1", "y2"].map(str::to_owned);
        assert_eq!(infer_rowwise_bandwidth(&equations, &names).unwrap(), (1, 1));
    }

    #[test]
    fn diagonal_jacobian_has_zero_half_bandwidths() {
        let equations = [
            Expr::parse_expression("-a*y0"),
            Expr::parse_expression("-b*y1"),
        ];
        let names = ["y0", "y1"].map(str::to_owned);
        assert_eq!(infer_rowwise_bandwidth(&equations, &names).unwrap(), (0, 0));
    }

    #[test]
    fn rejects_shape_mismatch_instead_of_guessing_a_band() {
        let equations = [Expr::parse_expression("y0")];
        let names = ["y0", "y1"].map(str::to_owned);
        assert!(infer_rowwise_bandwidth(&equations, &names).is_err());
    }
}
