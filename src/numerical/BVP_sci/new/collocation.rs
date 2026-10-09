//! Faithful SciPy BVP collocation residuals.
//!
//! The state is node-major in the new API.  The formula is the same cubic
//! collocation formula used by SciPy: midpoint values are reconstructed from
//! endpoint values and endpoint RHS values, then the midpoint RHS is sampled.

use super::{
    error::{BvpSciNewError, BvpSciStage},
    prepared::BvpSciLambdifyPlan,
    singular::SingularTermRuntime,
    workspace::BvpSciCollocationWorkspace,
};

/// Evaluate node and midpoint RHS values and the collocation residual.
pub(crate) fn evaluate_collocation(
    plan: &BvpSciLambdifyPlan,
    x: &[f64],
    y: &[f64],
    parameters: &[f64],
    singular: Option<&SingularTermRuntime>,
    workspace: &mut BvpSciCollocationWorkspace,
) -> Result<(), BvpSciNewError> {
    let n = plan.dimension();
    let m = x.len();
    let state_len = n.checked_mul(m).ok_or_else(|| {
        BvpSciNewError::InvalidConfiguration("collocation state size overflows usize".into())
    })?;
    if m < 2 || y.len() != state_len || parameters.len() != plan.parameter_dimension() {
        return Err(BvpSciNewError::ShapeMismatch {
            stage: BvpSciStage::NumericalCore,
            expected: state_len + plan.parameter_dimension(),
            actual: y.len() + parameters.len(),
        });
    }
    if workspace.n != n || workspace.m != m || workspace.k != parameters.len() {
        return Err(BvpSciNewError::InvalidConfiguration(
            "collocation workspace does not match the prepared problem".into(),
        ));
    }
    for (interval, pair) in x.windows(2).enumerate() {
        if !pair[0].is_finite() || !pair[1].is_finite() || pair[1] <= pair[0] {
            return Err(BvpSciNewError::InvalidConfiguration(
                "mesh must be finite and strictly increasing".into(),
            ));
        }
        workspace.h[interval] = pair[1] - pair[0];
        workspace.x_mid[interval] = pair[0] + 0.5 * workspace.h[interval];
    }

    let started = plan.telemetry().start_timing();
    for node in 0..m {
        let state = &y[node * n..(node + 1) * n];
        let output = &mut workspace.f_nodes[node * n..(node + 1) * n];
        plan.evaluate_rhs(
            x[node],
            state,
            parameters,
            &mut workspace.callback_arguments,
            output,
        )?;
        if let Some(singular) = singular {
            workspace.callback_output.copy_from_slice(output);
            singular.apply_rhs(x[node], state, &workspace.callback_output, output);
            plan.telemetry().record_singular_term_application();
        }
    }
    for interval in 0..m - 1 {
        let h = workspace.h[interval];
        let left = &y[interval * n..(interval + 1) * n];
        let right = &y[(interval + 1) * n..(interval + 2) * n];
        for component in 0..n {
            let f_left = workspace.f_nodes[interval * n + component];
            let f_right = workspace.f_nodes[(interval + 1) * n + component];
            workspace.y_middle[interval * n + component] =
                0.5 * (left[component] + right[component]) - 0.125 * h * (f_right - f_left);
        }
        let midpoint = &workspace.y_middle[interval * n..(interval + 1) * n];
        let output = &mut workspace.f_middle[interval * n..(interval + 1) * n];
        plan.evaluate_rhs(
            workspace.x_mid[interval],
            midpoint,
            parameters,
            &mut workspace.callback_arguments,
            output,
        )?;
        if let Some(singular) = singular {
            workspace.callback_output.copy_from_slice(output);
            singular.apply_rhs(
                workspace.x_mid[interval],
                midpoint,
                &workspace.callback_output,
                output,
            );
            plan.telemetry().record_singular_term_application();
        }
        for component in 0..n {
            workspace.collocation_residual[interval * n + component] = right[component]
                - left[component]
                - h / 6.0
                    * (workspace.f_nodes[interval * n + component]
                        + workspace.f_nodes[(interval + 1) * n + component]
                        + 4.0 * workspace.f_middle[interval * n + component]);
        }
    }
    for interval in 0..m - 1 {
        workspace.residual[interval * n..(interval + 1) * n]
            .copy_from_slice(&workspace.collocation_residual[interval * n..(interval + 1) * n]);
    }
    plan.telemetry().record_collocation(started);
    Ok(())
}

/// Infinity norm of the collocation residual normalized by interval length.
pub fn collocation_rms(workspace: &BvpSciCollocationWorkspace) -> f64 {
    workspace
        .collocation_residual
        .chunks_exact(workspace.n)
        .zip(&workspace.h)
        .map(|(residual, h)| {
            let mean = residual.iter().map(|value| value * value).sum::<f64>() / workspace.n as f64;
            mean.sqrt() / h.max(f64::MIN_POSITIVE).powf(1.5)
        })
        .fold(0.0, f64::max)
}

/// Maximum absolute value in a residual block.
pub fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_sci::new::{BvpSciAssembly, BvpSciTelemetry};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn collocation_residual_matches_linear_exact_solution() {
        let plan = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &[Expr::parse_expression("1")],
            &["y".into()],
            &[],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .unwrap();
        let mut workspace = BvpSciCollocationWorkspace::new(1, 3, 0, plan.telemetry()).unwrap();
        evaluate_collocation(
            &plan,
            &[0.0, 0.5, 1.0],
            &[0.0, 0.5, 1.0],
            &[],
            None,
            &mut workspace,
        )
        .unwrap();
        assert!(max_abs(&workspace.collocation_residual) < 1e-14);
    }
}
