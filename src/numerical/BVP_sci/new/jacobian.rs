//! Global collocation Jacobian assembly.
//!
//! The assembled entries are immediately consumable by Dense, Sparse or
//! Banded storage.  This module never creates a dense global matrix, which is
//! important for large mesh sizes.  Small pointwise `n x n` Jacobian blocks
//! remain in the workspace because they are the natural output of a residual
//! evaluator and are reused for all three layouts.

use super::{
    callbacks::{BvpSciBoundary, evaluate_boundary},
    error::{BvpSciNewError, BvpSciStage},
    prepared::BvpSciLambdifyPlan,
    singular::SingularTermRuntime,
    workspace::BvpSciCollocationWorkspace,
};

const FD_EPS: f64 = f64::EPSILON;

/// Evaluate pointwise Jacobians, finite-difference parameter/boundary blocks,
/// and append the global block-tridiagonal-plus-border entries to `workspace`.
pub(crate) fn assemble_global_jacobian(
    plan: &BvpSciLambdifyPlan,
    boundary: &dyn BvpSciBoundary,
    x: &[f64],
    y: &[f64],
    parameters: &[f64],
    singular: Option<&SingularTermRuntime>,
    workspace: &mut BvpSciCollocationWorkspace,
) -> Result<(), BvpSciNewError> {
    let n = plan.dimension();
    let m = x.len();
    let k = parameters.len();
    if workspace.n != n || workspace.m != m || workspace.k != k {
        return Err(BvpSciNewError::InvalidConfiguration(
            "Jacobian workspace does not match the collocation state".into(),
        ));
    }
    if boundary.residual_dimension() != n + k {
        return Err(BvpSciNewError::ShapeMismatch {
            stage: BvpSciStage::BoundaryCallback,
            expected: n + k,
            actual: boundary.residual_dimension(),
        });
    }

    // The pointwise analytic Jacobian is already in the prepared Lambdify
    // plan.  No symbolic differentiation or Expr/Atom conversion occurs here.
    for node in 0..m {
        let state = &y[node * n..(node + 1) * n];
        let output = &mut workspace.jacobian_nodes[node * n * n..(node + 1) * n * n];
        plan.evaluate_jacobian_dense_with_scratch(
            x[node],
            state,
            parameters,
            &mut workspace.callback_arguments,
            output,
            &mut workspace.callback_jacobian,
        )?;
        if let Some(singular) = singular {
            workspace.callback_jacobian[..output.len()].copy_from_slice(output);
            singular.apply_jacobian(
                x[node],
                &workspace.callback_jacobian[..output.len()],
                output,
            );
            plan.telemetry().record_singular_term_application();
        }
    }
    for interval in 0..m - 1 {
        let state = &workspace.y_middle[interval * n..(interval + 1) * n];
        let output = &mut workspace.jacobian_middle[interval * n * n..(interval + 1) * n * n];
        plan.evaluate_jacobian_dense_with_scratch(
            workspace.x_mid[interval],
            state,
            parameters,
            &mut workspace.callback_arguments,
            output,
            &mut workspace.callback_jacobian,
        )?;
        if let Some(singular) = singular {
            workspace.callback_jacobian[..output.len()].copy_from_slice(output);
            singular.apply_jacobian(
                workspace.x_mid[interval],
                &workspace.callback_jacobian[..output.len()],
                output,
            );
            plan.telemetry().record_singular_term_application();
        }
    }

    // Parameter derivatives are numerical only because the initial frontend
    // contract deliberately exposes state Jacobians first. They are computed
    // into reusable buffers and do not rebuild the prepared symbolic plan.
    // The perturbation is relative to the current parameter magnitude, which
    // avoids a fixed absolute step silently losing all signal for scaled
    // continuation parameters.
    if k > 0 && plan.has_parameter_jacobian() {
        for node in 0..m {
            let state = &y[node * n..(node + 1) * n];
            let output = &mut workspace.parameter_nodes[node * n * k..(node + 1) * n * k];
            plan.evaluate_parameter_jacobian(
                x[node],
                state,
                parameters,
                &mut workspace.callback_arguments,
                output,
            )?;
        }
        for interval in 0..m - 1 {
            let state = &workspace.y_middle[interval * n..(interval + 1) * n];
            let output = &mut workspace.parameter_middle[interval * n * k..(interval + 1) * n * k];
            plan.evaluate_parameter_jacobian(
                workspace.x_mid[interval],
                state,
                parameters,
                &mut workspace.callback_arguments,
                output,
            )?;
        }
    } else if k > 0 {
        let step = |value: f64| FD_EPS.sqrt() * (1.0 + value.abs());
        for parameter in 0..k {
            workspace.trial_parameters.copy_from_slice(parameters);
            let nominal_delta = step(parameters[parameter]);
            workspace.trial_parameters[parameter] += nominal_delta;
            // Use the representable increment, as SciPy does.  At large
            // magnitudes `x + h - x` can differ from the requested h.
            let delta = workspace.trial_parameters[parameter] - parameters[parameter];
            if delta == 0.0 || !delta.is_finite() {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "parameter finite-difference step is not representable".into(),
                ));
            }
            for node in 0..m {
                let state = &y[node * n..(node + 1) * n];
                plan.evaluate_rhs(
                    x[node],
                    state,
                    &workspace.trial_parameters,
                    &mut workspace.callback_arguments,
                    &mut workspace.callback_output,
                )?;
                if let Some(singular) = singular {
                    workspace
                        .callback_rhs
                        .copy_from_slice(&workspace.callback_output);
                    singular.apply_rhs(
                        x[node],
                        state,
                        &workspace.callback_rhs,
                        &mut workspace.callback_output,
                    );
                    plan.telemetry().record_singular_term_application();
                }
                for row in 0..n {
                    workspace.parameter_nodes[(node * n + row) * k + parameter] =
                        (workspace.callback_output[row] - workspace.f_nodes[node * n + row])
                            / delta;
                }
                plan.telemetry().record_finite_difference_probe();
            }
            for interval in 0..m - 1 {
                let state = &workspace.y_middle[interval * n..(interval + 1) * n];
                plan.evaluate_rhs(
                    workspace.x_mid[interval],
                    state,
                    &workspace.trial_parameters,
                    &mut workspace.callback_arguments,
                    &mut workspace.callback_output,
                )?;
                if let Some(singular) = singular {
                    workspace
                        .callback_rhs
                        .copy_from_slice(&workspace.callback_output);
                    singular.apply_rhs(
                        workspace.x_mid[interval],
                        state,
                        &workspace.callback_rhs,
                        &mut workspace.callback_output,
                    );
                    plan.telemetry().record_singular_term_application();
                }
                for row in 0..n {
                    workspace.parameter_middle[(interval * n + row) * k + parameter] =
                        (workspace.callback_output[row] - workspace.f_middle[interval * n + row])
                            / delta;
                }
                plan.telemetry().record_finite_difference_probe();
            }
        }
    }

    let ya = &y[..n];
    let yb = &y[(m - 1) * n..m * n];
    workspace.boundary_y_a.copy_from_slice(ya);
    workspace.boundary_y_b.copy_from_slice(yb);
    workspace.boundary_parameters.copy_from_slice(parameters);
    evaluate_boundary(
        boundary,
        plan.telemetry(),
        ya,
        yb,
        parameters,
        &mut workspace.boundary_residual,
    )?;
    finite_or_error(&workspace.boundary_residual, BvpSciStage::BoundaryCallback)?;

    let boundary_analytic = boundary.evaluate_jacobian(
        ya,
        yb,
        parameters,
        &mut workspace.boundary_ya_jacobian,
        &mut workspace.boundary_yb_jacobian,
        &mut workspace.boundary_parameter_jacobian,
    )?;
    let boundary_state_jacobian_len = (n + k).checked_mul(n).ok_or_else(|| {
        BvpSciNewError::InvalidConfiguration("boundary Jacobian size overflow".into())
    })?;
    let boundary_parameter_jacobian_len = (n + k).checked_mul(k).ok_or_else(|| {
        BvpSciNewError::InvalidConfiguration("boundary parameter Jacobian size overflow".into())
    })?;
    if workspace.boundary_ya_jacobian.len() != boundary_state_jacobian_len
        || workspace.boundary_yb_jacobian.len() != boundary_state_jacobian_len
        || workspace.boundary_parameter_jacobian.len() != boundary_parameter_jacobian_len
    {
        return Err(BvpSciNewError::ShapeMismatch {
            stage: BvpSciStage::JacobianCallback,
            expected: boundary_state_jacobian_len,
            actual: workspace
                .boundary_ya_jacobian
                .len()
                .max(workspace.boundary_yb_jacobian.len())
                .max(workspace.boundary_parameter_jacobian.len()),
        });
    }

    // Boundary callbacks may depend on both endpoint states and parameters.
    // Their derivatives use reusable trial output for one column at a time
    // only when the user did not provide the analytical boundary blocks.
    if !boundary_analytic {
        let boundary_step = |value: f64| FD_EPS.sqrt() * (1.0 + value.abs());
        for column in 0..n {
            let nominal_delta = boundary_step(ya[column]);
            workspace.boundary_y_a[column] = ya[column] + nominal_delta;
            let delta = workspace.boundary_y_a[column] - ya[column];
            if delta == 0.0 || !delta.is_finite() {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "left boundary finite-difference step is not representable".into(),
                ));
            }
            evaluate_boundary(
                boundary,
                plan.telemetry(),
                &workspace.boundary_y_a,
                yb,
                parameters,
                &mut workspace.boundary_trial_output,
            )?;
            for row in 0..n + k {
                workspace.boundary_ya_jacobian[row * n + column] =
                    (workspace.boundary_trial_output[row] - workspace.boundary_residual[row])
                        / delta;
            }
            workspace.boundary_y_a[column] = ya[column];
            plan.telemetry().record_finite_difference_probe();
        }
        for column in 0..n {
            let nominal_delta = boundary_step(yb[column]);
            workspace.boundary_y_b[column] = yb[column] + nominal_delta;
            let delta = workspace.boundary_y_b[column] - yb[column];
            if delta == 0.0 || !delta.is_finite() {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "right boundary finite-difference step is not representable".into(),
                ));
            }
            evaluate_boundary(
                boundary,
                plan.telemetry(),
                ya,
                &workspace.boundary_y_b,
                parameters,
                &mut workspace.boundary_trial_output,
            )?;
            for row in 0..n + k {
                workspace.boundary_yb_jacobian[row * n + column] =
                    (workspace.boundary_trial_output[row] - workspace.boundary_residual[row])
                        / delta;
            }
            workspace.boundary_y_b[column] = yb[column];
            plan.telemetry().record_finite_difference_probe();
        }
        for parameter in 0..k {
            let nominal_delta = boundary_step(parameters[parameter]);
            workspace.boundary_parameters[parameter] = parameters[parameter] + nominal_delta;
            let delta = workspace.boundary_parameters[parameter] - parameters[parameter];
            if delta == 0.0 || !delta.is_finite() {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "boundary parameter finite-difference step is not representable".into(),
                ));
            }
            evaluate_boundary(
                boundary,
                plan.telemetry(),
                ya,
                yb,
                &workspace.boundary_parameters,
                &mut workspace.boundary_trial_output,
            )?;
            for row in 0..n + k {
                workspace.boundary_parameter_jacobian[row * k + parameter] =
                    (workspace.boundary_trial_output[row] - workspace.boundary_residual[row])
                        / delta;
            }
            workspace.boundary_parameters[parameter] = parameters[parameter];
            plan.telemetry().record_finite_difference_probe();
        }
    }

    // Only now convert pointwise blocks into global collocation entries. The
    // list is the backend-neutral sparse representation; each selected
    // backend consumes it directly without constructing another global matrix
    // format in this layer.
    let assembly_started = plan.telemetry().start_timing();
    workspace.entries.clear();
    for interval in 0..m - 1 {
        let h = workspace.h[interval];
        let row_start = interval * n;
        let left_point = interval;
        let right_point = interval + 1;
        for row in 0..n {
            for column in 0..n {
                let left = block_value(&workspace.jacobian_nodes, n, left_point, row, column);
                let mid = block_value(&workspace.jacobian_middle, n, interval, row, column);
                let right = block_value(&workspace.jacobian_nodes, n, right_point, row, column);
                let left_product = matrix_product(
                    &workspace.jacobian_middle,
                    &workspace.jacobian_nodes,
                    n,
                    interval,
                    left_point,
                    row,
                    column,
                );
                let right_product = matrix_product(
                    &workspace.jacobian_middle,
                    &workspace.jacobian_nodes,
                    n,
                    interval,
                    right_point,
                    row,
                    column,
                );
                workspace.entries.push((
                    row_start + row,
                    interval * n + column,
                    if row == column { -1.0 } else { 0.0 }
                        - h / 6.0 * (left + 2.0 * mid)
                        - h * h / 12.0 * left_product,
                ));
                workspace.entries.push((
                    row_start + row,
                    (interval + 1) * n + column,
                    if row == column { 1.0 } else { 0.0 } - h / 6.0 * (right + 2.0 * mid)
                        + h * h / 12.0 * right_product,
                ));
            }
        }
        // The parameter block is independent of the state-column loop.
        // Keeping it outside that loop is mathematically necessary and
        // avoids adding each parameter entry n times.
        for row in 0..n {
            for parameter in 0..k {
                let dp_left = workspace.parameter_nodes[(left_point * n + row) * k + parameter];
                let dp_right = workspace.parameter_nodes[(right_point * n + row) * k + parameter];
                let dp_mid = workspace.parameter_middle[(interval * n + row) * k + parameter];
                let product =
                    parameter_product(workspace, interval, left_point, right_point, row, parameter);
                workspace.entries.push((
                    row_start + row,
                    m * n + parameter,
                    -h / 6.0 * (dp_left + dp_right + 4.0 * dp_mid) + h * h / 12.0 * product,
                ));
            }
        }
    }

    let boundary_row_start = (m - 1) * n;
    for row in 0..n + k {
        for column in 0..n {
            workspace.entries.push((
                boundary_row_start + row,
                column,
                workspace.boundary_ya_jacobian[row * n + column],
            ));
            workspace.entries.push((
                boundary_row_start + row,
                (m - 1) * n + column,
                workspace.boundary_yb_jacobian[row * n + column],
            ));
        }
        for parameter in 0..k {
            workspace.entries.push((
                boundary_row_start + row,
                m * n + parameter,
                workspace.boundary_parameter_jacobian[row * k + parameter],
            ));
        }
    }
    plan.telemetry()
        .record_jacobian_output_assembly(assembly_started);
    Ok(())
}

fn finite_or_error(values: &[f64], stage: BvpSciStage) -> Result<(), BvpSciNewError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(BvpSciNewError::NonFinite { stage })
    }
}

#[inline]
fn block_value(blocks: &[f64], n: usize, point: usize, row: usize, column: usize) -> f64 {
    blocks[(point * n + row) * n + column]
}

fn matrix_product(
    middle: &[f64],
    endpoint: &[f64],
    n: usize,
    interval: usize,
    endpoint_point: usize,
    row: usize,
    column: usize,
) -> f64 {
    (0..n)
        .map(|inner| {
            block_value(middle, n, interval, row, inner)
                * block_value(endpoint, n, endpoint_point, inner, column)
        })
        .sum()
}

fn parameter_product(
    workspace: &BvpSciCollocationWorkspace,
    interval: usize,
    left_point: usize,
    right_point: usize,
    row: usize,
    parameter: usize,
) -> f64 {
    (0..workspace.n)
        .map(|inner| {
            block_value(
                &workspace.jacobian_middle,
                workspace.n,
                interval,
                row,
                inner,
            ) * (workspace.parameter_nodes
                [(right_point * workspace.n + inner) * workspace.k + parameter]
                - workspace.parameter_nodes
                    [(left_point * workspace.n + inner) * workspace.k + parameter])
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_sci::new::collocation::evaluate_collocation;
    use crate::numerical::BVP_sci::new::{
        BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciLambdifyPlan, BvpSciTelemetry,
    };
    use crate::symbolic::symbolic_engine::Expr;

    fn zero_boundary(dimension: usize) -> BvpSciBoundaryCallbacks {
        BvpSciBoundaryCallbacks::new(
            dimension,
            move |_, _, _, output| {
                output.fill(0.0);
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        )
    }

    #[test]
    fn global_state_blocks_match_full_collocation_derivative() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[
                Expr::parse_expression("2*y0"),
                Expr::parse_expression("3*y1"),
            ],
            &["y0".into(), "y1".into()],
            &[],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        let boundary = zero_boundary(2);
        let x = [0.0, 1.0];
        let y = [1.0, 2.0, 1.5, 2.5];
        let mut workspace =
            BvpSciCollocationWorkspace::new(2, 2, 0, &BvpSciTelemetry::disabled()).unwrap();
        evaluate_collocation(&plan, &x, &y, &[], None, &mut workspace).unwrap();
        assemble_global_jacobian(&plan, &boundary, &x, &y, &[], None, &mut workspace).unwrap();

        let left_y0 = workspace
            .entries
            .iter()
            .find(|&&(row, column, _)| row == 0 && column == 0)
            .map(|&(_, _, value)| value)
            .unwrap();
        let right_y0 = workspace
            .entries
            .iter()
            .find(|&&(row, column, _)| row == 0 && column == 2)
            .map(|&(_, _, value)| value)
            .unwrap();
        // SciPy: -I - h/6*(J_left + 2J_mid) - h^2/12*(J_mid*J_left)
        // and I - h/6*(J_right + 2J_mid) + h^2/12*(J_mid*J_right).
        assert!((left_y0 - (-7.0 / 3.0)).abs() < 1e-12);
        assert!(
            (right_y0 - (1.0 / 3.0)).abs() < 1e-12,
            "right block={right_y0}"
        );
    }

    #[test]
    fn parameter_block_is_emitted_once_per_row() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[
                Expr::parse_expression("y0+p"),
                Expr::parse_expression("y1+p"),
            ],
            &["y0".into(), "y1".into()],
            &["p".into()],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        let boundary = zero_boundary(3);
        let x = [0.0, 1.0];
        let y = [0.0, 0.0, 0.0, 0.0];
        let mut workspace =
            BvpSciCollocationWorkspace::new(2, 2, 1, &BvpSciTelemetry::disabled()).unwrap();
        evaluate_collocation(&plan, &x, &y, &[1.0], None, &mut workspace).unwrap();
        assemble_global_jacobian(&plan, &boundary, &x, &y, &[1.0], None, &mut workspace).unwrap();

        let parameter_entries: Vec<_> = workspace
            .entries
            .iter()
            .filter(|&&(row, column, _)| row < 2 && column == 4)
            .collect();
        assert_eq!(
            parameter_entries.len(),
            2,
            "parameter entries={parameter_entries:?}"
        );
        assert!(
            parameter_entries
                .iter()
                .all(|&&(_, _, value)| (value + 1.0).abs() < 1e-12),
            "parameter entries={parameter_entries:?}"
        );
    }

    #[test]
    fn parameter_block_matches_scipy_collocation_finite_difference() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[Expr::parse_expression("p*y0")],
            &["y0".into()],
            &["p".into()],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        let boundary = zero_boundary(2);
        let x = [0.0, 1.0];
        let y = [1.0, 1.5];
        let parameters = [2.0];
        let mut workspace =
            BvpSciCollocationWorkspace::new(1, 2, 1, &BvpSciTelemetry::disabled()).unwrap();
        evaluate_collocation(&plan, &x, &y, &parameters, None, &mut workspace).unwrap();
        assemble_global_jacobian(&plan, &boundary, &x, &y, &parameters, None, &mut workspace)
            .unwrap();

        let assembled = workspace
            .entries
            .iter()
            .find(|&&(row, column, _)| row == 0 && column == 2)
            .map(|&(_, _, value)| value)
            .unwrap();

        // SciPy's parameter block is the derivative of the complete
        // collocation residual, not just the quadrature of df/dp. Compare
        // against the same forward difference used by the reference code.
        let nominal_delta = f64::EPSILON.sqrt() * (1.0 + parameters[0].abs());
        let delta = (parameters[0] + nominal_delta) - parameters[0];
        let mut trial =
            BvpSciCollocationWorkspace::new(1, 2, 1, &BvpSciTelemetry::disabled()).unwrap();
        evaluate_collocation(
            &plan,
            &x,
            &y,
            &[parameters[0] + nominal_delta],
            None,
            &mut trial,
        )
        .unwrap();
        let reference = (trial.residual[0] - workspace.residual[0]) / delta;
        assert!(
            (assembled - reference).abs() < 1e-7,
            "assembled={assembled} reference={reference}"
        );
    }
}
