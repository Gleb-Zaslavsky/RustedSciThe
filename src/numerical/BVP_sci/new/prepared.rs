//! Prepared model lifecycle.
//!
//! This layer owns the frontend choice and exposes one callback contract to
//! the numerical core.  The enum is deliberately not a trait object: the
//! selected frontend is known after preparation, so callback dispatch stays
//! monomorphic and does not add a virtual call to Newton/collocation loops.

use super::config::{BvpSciAssembly, BvpSciExecutionPolicy};
use super::error::BvpSciNewError;
use super::frontends::{AtomViewNativeLambdifyPlan, BvpSciAotPlan, ExprLegacyLambdifyPlan};
use super::numerical::BvpSciNumericalPlan;
use super::telemetry::{BvpSciTelemetry, BvpSciTelemetrySnapshot};
use crate::symbolic::symbolic_engine::Expr;

/// Prepared Lambdify frontend used by the new BVP numerical core.
#[derive(Clone)]
pub enum BvpSciLambdifyPlan {
    /// Caller-owned numerical callbacks with analytic or finite-difference
    /// Jacobians. This variant is intended for the Dense route.
    Numerical(BvpSciNumericalPlan),
    /// Legacy Expr representation, retained as an independent reference path.
    ExprLegacy(ExprLegacyLambdifyPlan),
    /// Packed AtomView representation with native symbolic differentiation.
    AtomViewNative(AtomViewNativeLambdifyPlan),
    /// Generated native callbacks selected through the shared AOT lifecycle.
    Aot(BvpSciAotPlan),
}

impl BvpSciLambdifyPlan {
    /// Prepare exactly one symbolic representation and its reusable evaluators.
    pub fn prepare(
        assembly: BvpSciAssembly,
        equations: &[Expr],
        state_names: &[String],
        parameter_names: &[String],
        independent_name: impl Into<String>,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        let independent_name = independent_name.into();
        match assembly {
            BvpSciAssembly::Numerical => Err(BvpSciNewError::UnsupportedRoute(
                "symbolic preparation cannot use the Numerical assembly".into(),
            )),
            BvpSciAssembly::ExprLegacy => Ok(Self::ExprLegacy(ExprLegacyLambdifyPlan::prepare(
                equations,
                state_names,
                parameter_names,
                independent_name,
                telemetry,
            )?)),
            BvpSciAssembly::AtomViewNative => {
                Ok(Self::AtomViewNative(AtomViewNativeLambdifyPlan::prepare(
                    equations,
                    state_names,
                    parameter_names,
                    independent_name,
                    telemetry,
                )?))
            }
        }
    }

    /// Prepare a generated AOT plan for Dense, Sparse or Banded storage.
    ///
    /// The owned equation/name vectors make ownership of the generated
    /// lifecycle explicit and avoid retaining borrowed symbolic input in a
    /// long-lived continuation solver.
    pub fn prepare_aot(
        assembly: BvpSciAssembly,
        layout: super::config::BvpSciMatrixLayout,
        equations: Vec<Expr>,
        state_names: Vec<String>,
        parameter_names: Vec<String>,
        independent_name: impl Into<String>,
        config: crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        Ok(Self::Aot(BvpSciAotPlan::prepare(
            assembly,
            layout,
            equations,
            state_names,
            parameter_names,
            independent_name,
            config,
            telemetry,
        )?))
    }

    /// AOT preparation variant with an explicit callback execution policy.
    pub fn prepare_aot_with_policy(
        assembly: BvpSciAssembly,
        layout: super::config::BvpSciMatrixLayout,
        equations: Vec<Expr>,
        state_names: Vec<String>,
        parameter_names: Vec<String>,
        independent_name: impl Into<String>,
        config: crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig,
        telemetry: BvpSciTelemetry,
        policy: BvpSciExecutionPolicy,
    ) -> Result<Self, BvpSciNewError> {
        Ok(Self::Aot(BvpSciAotPlan::prepare_with_policy(
            assembly,
            layout,
            equations,
            state_names,
            parameter_names,
            independent_name,
            config,
            telemetry,
            policy,
        )?))
    }

    /// Wrap a numerical callback plan in the shared collocation lifecycle.
    pub fn prepare_numerical(plan: BvpSciNumericalPlan) -> Self {
        Self::Numerical(plan)
    }

    /// Return the symbolic representation fixed during preparation.
    pub fn assembly(&self) -> BvpSciAssembly {
        match self {
            Self::Numerical(_) => BvpSciAssembly::Numerical,
            Self::ExprLegacy(_) => BvpSciAssembly::ExprLegacy,
            Self::AtomViewNative(_) => BvpSciAssembly::AtomViewNative,
            Self::Aot(plan) => plan.assembly(),
        }
    }

    /// Whether this plan owns a generated native callback runtime.
    pub fn is_aot(&self) -> bool {
        matches!(self, Self::Aot(_))
    }

    /// Return the fixed callback policy selected during AOT preparation.
    pub fn aot_execution_policy(&self) -> Option<BvpSciExecutionPolicy> {
        match self {
            Self::Numerical(_) => None,
            Self::Aot(plan) => Some(plan.execution_policy()),
            _ => None,
        }
    }

    /// Select the warm callback policy without rebuilding symbolic state.
    pub fn with_execution_policy(self, policy: BvpSciExecutionPolicy) -> Self {
        match self {
            Self::Numerical(plan) => Self::Numerical(plan.with_execution_policy(policy)),
            Self::ExprLegacy(plan) => Self::ExprLegacy(plan.with_execution_policy(policy)),
            Self::AtomViewNative(plan) => Self::AtomViewNative(plan.with_execution_policy(policy)),
            // AOT policy is fixed during generated preparation because it
            // determines emitted chunk ownership. Rebinding it afterwards
            // would make lifecycle telemetry and callback provenance false.
            Self::Aot(plan) => Self::Aot(plan),
        }
    }

    /// Return the state dimension of the prepared first-order system.
    pub fn dimension(&self) -> usize {
        match self {
            Self::Numerical(plan) => plan.dimension(),
            Self::ExprLegacy(plan) => plan.dimension(),
            Self::AtomViewNative(plan) => plan.dimension(),
            Self::Aot(plan) => plan.dimension(),
        }
    }

    /// Return the number of runtime parameters accepted by the callbacks.
    pub fn parameter_dimension(&self) -> usize {
        match self {
            Self::Numerical(plan) => plan.parameter_dimension(),
            Self::ExprLegacy(plan) => plan.parameter_dimension(),
            Self::AtomViewNative(plan) => plan.parameter_dimension(),
            Self::Aot(plan) => plan.parameter_dimension(),
        }
    }

    /// Return the fixed pointwise Jacobian structural nonzero count.
    pub fn jacobian_nnz(&self) -> usize {
        match self {
            Self::Numerical(plan) => plan.jacobian_nnz(),
            Self::ExprLegacy(plan) => plan.jacobian_nnz(),
            Self::AtomViewNative(plan) => plan.jacobian_nnz(),
            Self::Aot(plan) => plan.jacobian_nnz(),
        }
    }

    /// Return the caller-owned scratch length required by the hot Jacobian
    /// callback. Compact AOT layouts may need more slots than the dense
    /// pointwise output that the collocation assembler ultimately consumes.
    pub fn jacobian_callback_scratch_len(&self) -> Result<usize, BvpSciNewError> {
        match self {
            Self::Numerical(plan) => plan
                .dimension()
                .checked_mul(plan.dimension())
                .ok_or_else(|| {
                    BvpSciNewError::InvalidConfiguration("Jacobian scratch size overflow".into())
                }),
            Self::ExprLegacy(plan) => {
                plan.dimension()
                    .checked_mul(plan.dimension())
                    .ok_or_else(|| {
                        BvpSciNewError::InvalidConfiguration(
                            "Jacobian scratch size overflow".into(),
                        )
                    })
            }
            Self::AtomViewNative(plan) => plan
                .dimension()
                .checked_mul(plan.dimension())
                .ok_or_else(|| {
                    BvpSciNewError::InvalidConfiguration("Jacobian scratch size overflow".into())
                }),
            Self::Aot(plan) => plan.jacobian_callback_scratch_len(),
        }
    }

    /// Return the fixed structural Jacobian order for native storage.
    pub fn jacobian_pattern(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Numerical(plan) => plan.jacobian_pattern(),
            Self::ExprLegacy(plan) => plan.jacobian_pattern().collect(),
            Self::AtomViewNative(plan) => plan.jacobian_pattern().collect(),
            Self::Aot(plan) => plan.jacobian_pattern(),
        }
    }

    /// Snapshot preparation and callback telemetry without exposing internals.
    pub fn telemetry_snapshot(&self) -> BvpSciTelemetrySnapshot {
        let mut snapshot = match self {
            Self::Numerical(plan) => plan.telemetry_snapshot(),
            Self::ExprLegacy(plan) => plan.telemetry_snapshot(),
            Self::AtomViewNative(plan) => plan.telemetry_snapshot(),
            Self::Aot(plan) => plan.telemetry().snapshot(),
        };
        if let Self::Aot(plan) = self {
            snapshot.aot = Some(plan.aot_telemetry_snapshot());
        }
        snapshot
    }

    /// Borrow the shared telemetry handle used by this prepared model.
    pub fn telemetry(&self) -> &BvpSciTelemetry {
        match self {
            Self::Numerical(plan) => plan.telemetry(),
            Self::ExprLegacy(plan) => plan.telemetry(),
            Self::AtomViewNative(plan) => plan.telemetry(),
            Self::Aot(plan) => plan.telemetry(),
        }
    }

    /// Evaluate the prepared pointwise residual into caller-owned buffers.
    ///
    /// `arguments` is scratch storage for `[x, state..., parameters...]` and
    /// is deliberately supplied by the numerical workspace so repeated calls
    /// do not allocate. `output` must have exactly `dimension()` slots.
    pub fn evaluate_rhs(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        match self {
            Self::Numerical(plan) => plan.evaluate_rhs(x, state, parameters, arguments, output),
            Self::ExprLegacy(plan) => plan.evaluate_rhs(x, state, parameters, arguments, output),
            Self::AtomViewNative(plan) => {
                plan.evaluate_rhs(x, state, parameters, arguments, output)
            }
            Self::Aot(plan) => plan.evaluate_rhs(x, state, parameters, arguments, output),
        }
    }

    /// Evaluate the pointwise state Jacobian in row-major dense order.
    ///
    /// Frontends keep a sparse structural plan internally, but the numerical
    /// collocation layer can request a dense pointwise block before it packs
    /// that block into the selected global layout.
    pub fn evaluate_jacobian_dense(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        match self {
            Self::Numerical(plan) => {
                let mut scratch = vec![0.0; output.len()];
                plan.evaluate_jacobian_dense_with_scratch(
                    x, state, parameters, arguments, output, &mut scratch,
                )
            }
            Self::ExprLegacy(plan) => {
                plan.evaluate_jacobian_dense(x, state, parameters, arguments, output)
            }
            Self::AtomViewNative(plan) => {
                plan.evaluate_jacobian_dense(x, state, parameters, arguments, output)
            }
            Self::Aot(plan) => {
                plan.evaluate_jacobian_dense(x, state, parameters, arguments, output)
            }
        }
    }

    /// Hot-path Jacobian entry point with caller-owned scratch for structured
    /// AOT output. Lambdify frontends ignore the extra scratch buffer.
    pub fn evaluate_jacobian_dense_with_scratch(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        match self {
            Self::Numerical(plan) => plan.evaluate_jacobian_dense_with_scratch(
                x, state, parameters, arguments, output, scratch,
            ),
            Self::ExprLegacy(plan) => {
                plan.evaluate_jacobian_dense(x, state, parameters, arguments, output)
            }
            Self::AtomViewNative(plan) => {
                plan.evaluate_jacobian_dense(x, state, parameters, arguments, output)
            }
            Self::Aot(plan) => plan.evaluate_jacobian_dense_with_scratch(
                x, state, parameters, arguments, output, scratch,
            ),
        }
    }

    /// Evaluate an analytical `df/dp` block when the numerical frontend has
    /// one. Other frontends return `false`, leaving the existing FD path in
    /// the common Jacobian assembler responsible for parameter columns.
    pub fn evaluate_parameter_jacobian(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<bool, BvpSciNewError> {
        match self {
            Self::Numerical(plan) => {
                plan.evaluate_parameter_jacobian(x, state, parameters, arguments, output)
            }
            _ => Ok(false),
        }
    }

    pub fn has_parameter_jacobian(&self) -> bool {
        matches!(self, Self::Numerical(plan) if plan.has_rhs_parameter_jacobian())
    }

    /// Evaluate Jacobian values directly in the prepared structural order.
    ///
    /// The returned order is stable for the lifetime of the plan and is used
    /// by Sparse/Banded assembly without re-differentiating or discovering a
    /// pattern during Newton iterations.
    pub fn evaluate_jacobian_values(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        match self {
            Self::Numerical(plan) => {
                let mut scratch = vec![0.0; values.len()];
                plan.evaluate_jacobian_dense_with_scratch(
                    x, state, parameters, arguments, values, &mut scratch,
                )
            }
            Self::ExprLegacy(plan) => {
                plan.evaluate_jacobian_values(x, state, parameters, arguments, values)
            }
            Self::AtomViewNative(plan) => {
                plan.evaluate_jacobian_values(x, state, parameters, arguments, values)
            }
            Self::Aot(plan) => {
                plan.evaluate_jacobian_values(x, state, parameters, arguments, values)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BvpSciLambdifyPlan;
    use crate::numerical::BVP_sci::new::{BvpSciAssembly, BvpSciTelemetry, BvpSciTelemetryMode};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn unified_plan_preserves_frontend_parity_and_stage_telemetry() {
        let equation = [Expr::parse_expression("p*y + x")];
        let state_names = ["y".to_owned()];
        let parameter_names = ["p".to_owned()];
        let expr = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::ExprLegacy,
            &equation,
            &state_names,
            &parameter_names,
            "x",
            BvpSciTelemetry::timings(),
        )
        .expect("ExprLegacy plan should prepare");
        let atom = BvpSciLambdifyPlan::prepare(
            BvpSciAssembly::AtomViewNative,
            &equation,
            &state_names,
            &parameter_names,
            "x",
            BvpSciTelemetry::timings(),
        )
        .expect("AtomView plan should prepare");

        let mut expr_args = [0.0; 3];
        let mut atom_args = [0.0; 3];
        let mut expr_rhs = [0.0; 1];
        let mut atom_rhs = [0.0; 1];
        expr.evaluate_rhs(2.0, &[3.0], &[4.0], &mut expr_args, &mut expr_rhs)
            .expect("ExprLegacy RHS should evaluate");
        atom.evaluate_rhs(2.0, &[3.0], &[4.0], &mut atom_args, &mut atom_rhs)
            .expect("AtomView RHS should evaluate");

        assert_eq!(expr.assembly(), BvpSciAssembly::ExprLegacy);
        assert_eq!(atom.assembly(), BvpSciAssembly::AtomViewNative);
        assert_eq!(expr_rhs, atom_rhs);
        assert_eq!(expr.jacobian_nnz(), atom.jacobian_nnz());
        let mut expr_values = [0.0; 1];
        let mut atom_values = [0.0; 1];
        expr.evaluate_jacobian_values(2.0, &[3.0], &[4.0], &mut expr_args, &mut expr_values)
            .expect("ExprLegacy structural Jacobian should evaluate");
        atom.evaluate_jacobian_values(2.0, &[3.0], &[4.0], &mut atom_args, &mut atom_values)
            .expect("AtomView structural Jacobian should evaluate");
        assert_eq!(expr.jacobian_pattern(), atom.jacobian_pattern());
        assert_eq!(expr_values, atom_values);
        assert_eq!(expr.telemetry_snapshot().mode, BvpSciTelemetryMode::Timings);
        assert_eq!(atom.telemetry_snapshot().atom_conversions, 1);
        assert_eq!(atom.telemetry_snapshot().pattern_entries, 1);
        assert!(atom.telemetry_snapshot().expr_to_atom_ms.is_some());
        assert_eq!(atom.telemetry_snapshot().residual_evaluator_compilations, 1);
        assert_eq!(atom.telemetry_snapshot().jacobian_evaluator_compilations, 1);
        assert!(atom
            .telemetry_snapshot()
            .residual_evaluator_compilation_ms
            .is_some());
        assert!(atom
            .telemetry_snapshot()
            .jacobian_evaluator_compilation_ms
            .is_some());

        // Parameter continuation changes callback values only. The prepared
        // symbolic structures and evaluator counts must remain unchanged.
        atom.evaluate_rhs(2.0, &[3.0], &[5.0], &mut atom_args, &mut atom_rhs)
            .expect("AtomView continuation RHS should evaluate");
        assert_eq!(atom_rhs, [17.0]);
        let continued = atom.telemetry_snapshot();
        assert_eq!(continued.atom_conversions, 1);
        assert_eq!(continued.pattern_entries, 1);
        assert_eq!(continued.evaluator_compilations, 2);
        assert_eq!(continued.residual_evaluator_compilations, 1);
        assert_eq!(continued.jacobian_evaluator_compilations, 1);
    }
}
