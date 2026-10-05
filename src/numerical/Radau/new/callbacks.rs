//! Residual and Jacobian callback contracts for the new Radau solver.
//!
//! The eventual implementation will keep symbolic preparation outside the
//! step loop and make callback failures part of the typed solver error path.

use super::error::{RadauConfigError, RadauError, RadauStage, RadauUnsupportedRoute};
use super::{
    aot::{AotPlan, AotWorkspace},
    atom_native::{AtomNativePlan, AtomNativeWorkspace},
    config::{RadauAssembly, RadauMatrixLayout},
    lambdify::{LambdifyPlan, LambdifyWorkspace},
    telemetry::{RadauFrontendStage, RadauTelemetry, RadauTelemetryMode},
};
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::symbolic_engine::Expr;

/// The symbolic frontend is selected once during preparation.
///
/// The numerical core only sees this common callback boundary and never
/// branches on the symbolic representation inside a Newton step.  Keeping the
/// prepared plan separate from its mutable workspace is what makes parameter
/// rebind and continuation possible without repeating symbolic construction.
pub(crate) enum PreparedSymbolicCallbacks {
    ExprLegacy(LambdifyPlan),
    AtomViewNative(AtomNativePlan),
    Aot(AotPlan),
}

/// Scratch storage paired with the prepared symbolic frontend.
///
/// These buffers belong to the callback/session layer, not to the Radau
/// Newton workspace.  That ownership boundary lets telemetry attribute
/// symbolic preparation and callback binding separately from linear algebra.
pub(crate) enum SymbolicCallbackWorkspace {
    ExprLegacy(LambdifyWorkspace),
    AtomViewNative(AtomNativeWorkspace),
    Aot(AotWorkspace),
}

/// Mutable callback binding for one solve/continuation session.
///
/// The prepared frontend remains immutable and can be shared by multiple
/// sessions. Parameter values and evaluator scratch are isolated per session;
/// value-only rebinds therefore do not rebuild symbolic closures.
pub(crate) struct SymbolicCallbackSession<'a> {
    callbacks: &'a PreparedSymbolicCallbacks,
    workspace: SymbolicCallbackWorkspace,
    /// Cached once so Sparse/Banded preparation is not repeated per step.
    jacobian_pattern: Vec<(usize, usize)>,
    parameters: Vec<f64>,
    generation: u64,
    telemetry: RadauTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
}

impl PreparedSymbolicCallbacks {
    /// Prepare one selected symbolic frontend and its immutable callback plan.
    pub(crate) fn prepare(
        assembly: RadauAssembly,
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
    ) -> Result<Self, RadauError> {
        let mut telemetry = RadauTelemetry::new(Default::default());
        Self::prepare_with_telemetry(
            assembly,
            residual,
            jacobian,
            independent_variable,
            variables,
            parameters,
            &mut telemetry,
        )
    }

    /// Prepare a frontend while collecting detailed cold-stage telemetry.
    pub(crate) fn prepare_with_telemetry(
        assembly: RadauAssembly,
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
        telemetry: &mut RadauTelemetry,
    ) -> Result<Self, RadauError> {
        Self::prepare_with_telemetry_and_policy(
            assembly,
            residual,
            jacobian,
            independent_variable,
            variables,
            parameters,
            telemetry,
            IvpLambdifyExecutionPolicy::Sequential,
        )
    }

    /// Prepare one frontend while retaining the selected callback scheduling policy.
    pub(crate) fn prepare_with_telemetry_and_policy(
        assembly: RadauAssembly,
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
        telemetry: &mut RadauTelemetry,
        execution_policy: IvpLambdifyExecutionPolicy,
    ) -> Result<Self, RadauError> {
        match assembly {
            RadauAssembly::ExprLegacy => {
                telemetry.measure_frontend(RadauFrontendStage::ExprLegacyPrepare, || {
                    LambdifyPlan::from_expressions_with_policy(
                        residual,
                        jacobian,
                        independent_variable,
                        variables,
                        parameters,
                        execution_policy,
                    )
                    .map(Self::ExprLegacy)
                })
            }
            RadauAssembly::AtomViewNative => {
                let telemetry_mode = telemetry.mode();
                let result = telemetry.measure_frontend(RadauFrontendStage::AtomConversion, || {
                    AtomNativePlan::from_expressions_with_telemetry_and_policy(
                        residual,
                        jacobian,
                        independent_variable,
                        variables,
                        parameters,
                        telemetry_mode,
                        execution_policy,
                    )
                });
                result.map(|plan| {
                    plan.absorb_preparation_telemetry(telemetry);
                    Self::AtomViewNative(plan)
                })
            }
        }
    }

    /// Allocate session-local callback buffers without rebuilding the plan.
    pub(crate) fn workspace(&self) -> SymbolicCallbackWorkspace {
        self.workspace_with_telemetry(RadauTelemetryMode::Off)
    }

    /// Allocate workspace and, for AOT, the shared runtime telemetry stream.
    pub(crate) fn workspace_with_telemetry(
        &self,
        mode: RadauTelemetryMode,
    ) -> SymbolicCallbackWorkspace {
        match self {
            Self::ExprLegacy(plan) => SymbolicCallbackWorkspace::ExprLegacy(plan.workspace()),
            Self::AtomViewNative(plan) => {
                SymbolicCallbackWorkspace::AtomViewNative(plan.workspace())
            }
            Self::Aot(plan) => SymbolicCallbackWorkspace::Aot(plan.workspace_with_telemetry(mode)),
        }
    }

    pub(crate) fn execution_policy(&self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::ExprLegacy(plan) => plan.execution_policy(),
            Self::AtomViewNative(plan) => plan.execution_policy(),
            Self::Aot(plan) => plan.execution_policy(),
        }
    }

    /// Create a callback session with telemetry disabled.
    pub(crate) fn session(&self) -> SymbolicCallbackSession<'_> {
        self.session_with_telemetry(RadauTelemetryMode::Off)
    }

    /// Create a session with caller-selected telemetry and isolated buffers.
    pub(crate) fn session_with_telemetry(
        &self,
        mode: RadauTelemetryMode,
    ) -> SymbolicCallbackSession<'_> {
        let parameter_count = match self {
            Self::ExprLegacy(plan) => plan.parameter_count(),
            Self::AtomViewNative(plan) => plan.parameter_count(),
            Self::Aot(plan) => plan.parameter_count(),
        };
        SymbolicCallbackSession {
            callbacks: self,
            workspace: self.workspace_with_telemetry(mode),
            jacobian_pattern: self.jacobian_pattern(),
            parameters: vec![0.0; parameter_count],
            generation: 1,
            telemetry: RadauTelemetry::new(mode),
            execution_policy: self.execution_policy(),
        }
    }

    pub(crate) fn evaluate_residual(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut SymbolicCallbackWorkspace,
    ) -> Result<(), RadauError> {
        let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Off);
        self.evaluate_residual_with_telemetry(
            t,
            state,
            parameters,
            output,
            workspace,
            &mut telemetry,
        )
    }

    pub(crate) fn evaluate_residual_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut SymbolicCallbackWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        match (self, workspace) {
            (Self::ExprLegacy(plan), SymbolicCallbackWorkspace::ExprLegacy(workspace)) => plan
                .evaluate_residual_with_telemetry(
                    t, state, parameters, output, workspace, telemetry,
                ),
            (Self::AtomViewNative(plan), SymbolicCallbackWorkspace::AtomViewNative(workspace)) => {
                plan.evaluate_residual_with_telemetry(
                    t, state, parameters, output, workspace, telemetry,
                )
            }
            (Self::Aot(plan), SymbolicCallbackWorkspace::Aot(workspace)) => {
                plan.evaluate_residual(t, state, parameters, output, workspace, telemetry)
            }
            _ => Err(RadauConfigError::CallbackFrontendMismatch.into()),
        }
    }

    pub(crate) fn evaluate_jacobian(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut SymbolicCallbackWorkspace,
    ) -> Result<(), RadauError> {
        let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Off);
        self.evaluate_jacobian_with_telemetry(
            t,
            state,
            parameters,
            output,
            workspace,
            &mut telemetry,
        )
    }

    pub(crate) fn evaluate_jacobian_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut SymbolicCallbackWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        match (self, workspace) {
            (Self::ExprLegacy(plan), SymbolicCallbackWorkspace::ExprLegacy(workspace)) => plan
                .evaluate_jacobian_with_telemetry(
                    t, state, parameters, output, workspace, telemetry,
                ),
            (Self::AtomViewNative(plan), SymbolicCallbackWorkspace::AtomViewNative(workspace)) => {
                plan.evaluate_jacobian_with_telemetry(
                    t, state, parameters, output, workspace, telemetry,
                )
            }
            (Self::Aot(plan), SymbolicCallbackWorkspace::Aot(workspace)) => plan.evaluate_jacobian(
                t,
                state,
                parameters,
                RadauMatrixLayout::Dense,
                output,
                workspace,
                telemetry,
            ),
            _ => Err(RadauConfigError::CallbackFrontendMismatch.into()),
        }
    }

    /// Return the structural pattern needed to construct a native backend.
    ///
    /// The returned vector is a preparation-time snapshot.  The numerical
    /// step later borrows the session's cached pattern and writes only values.
    pub(crate) fn jacobian_pattern(&self) -> Vec<(usize, usize)> {
        match self {
            Self::ExprLegacy(plan) => plan.jacobian_pattern().to_vec(),
            Self::AtomViewNative(plan) => plan.jacobian_pattern(),
            Self::Aot(plan) => plan.jacobian_pattern().to_vec(),
        }
    }

    /// Evaluate directly in the matrix layout requested by the future Radau
    /// linear backend. Dense uses row-major values; Sparse uses the prepared
    /// immutable entry order; Banded uses compact `(kl + ku + 1) x n` storage.
    pub(crate) fn evaluate_jacobian_layout(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
        workspace: &mut SymbolicCallbackWorkspace,
    ) -> Result<(), RadauError> {
        let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Off);
        self.evaluate_jacobian_layout_with_telemetry(
            t,
            state,
            parameters,
            layout,
            output,
            workspace,
            &mut telemetry,
        )
    }

    pub(crate) fn evaluate_jacobian_layout_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
        workspace: &mut SymbolicCallbackWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        match (self, workspace) {
            (Self::ExprLegacy(plan), SymbolicCallbackWorkspace::ExprLegacy(workspace)) => plan
                .evaluate_jacobian_layout_with_telemetry(
                    t, state, parameters, layout, output, workspace, telemetry,
                ),
            (Self::AtomViewNative(plan), SymbolicCallbackWorkspace::AtomViewNative(workspace)) => {
                plan.evaluate_jacobian_layout_with_telemetry(
                    t, state, parameters, layout, output, workspace, telemetry,
                )
            }
            (Self::Aot(plan), SymbolicCallbackWorkspace::Aot(workspace)) => {
                plan.evaluate_jacobian(t, state, parameters, layout, output, workspace, telemetry)
            }
            _ => Err(RadauConfigError::CallbackFrontendMismatch.into()),
        }
    }
}

impl<'a> SymbolicCallbackSession<'a> {
    /// Borrow the session's diagnostic counters and timings.
    pub(crate) fn telemetry(&self) -> &RadauTelemetry {
        &self.telemetry
    }

    /// Mutably borrow the session telemetry for integration with a solve.
    pub(crate) fn telemetry_mut(&mut self) -> &mut RadauTelemetry {
        &mut self.telemetry
    }

    /// Borrow the current parameter values.
    pub(crate) fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Return the monotonically increasing parameter generation.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Return the immutable Jacobian pattern prepared by the frontend.
    pub(crate) fn jacobian_pattern(&self) -> &[(usize, usize)] {
        &self.jacobian_pattern
    }

    /// Return the capacity of the frontend-specific callback workspace.
    pub(crate) fn workspace_capacity(&self) -> usize {
        match &self.workspace {
            SymbolicCallbackWorkspace::ExprLegacy(workspace) => workspace.argument_capacity(),
            SymbolicCallbackWorkspace::AtomViewNative(workspace) => workspace.value_capacity(),
            SymbolicCallbackWorkspace::Aot(workspace) => workspace.argument_capacity(),
        }
    }

    /// Rebind values without recompiling or replacing callback storage.
    pub(crate) fn rebind_parameters(&mut self, values: &[f64]) -> Result<(), RadauError> {
        if values.len() != self.parameters.len() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Preparation,
                expected: self.parameters.len(),
                actual: values.len(),
            });
        }
        self.telemetry.measure_callback(
            super::telemetry::RadauCallbackStage::ParameterRebind,
            || self.parameters.copy_from_slice(values),
        );
        self.telemetry.count_copy();
        self.generation = self.generation.saturating_add(1);
        Ok(())
    }

    pub(crate) fn evaluate_residual(
        &mut self,
        t: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), RadauError> {
        // Keep the solver-level callback count distinct from the evaluator
        // count. The former counts logical residual requests; the latter
        // records frontend scalar evaluation work inside that request.
        self.telemetry.count_stage(RadauStage::Residual);
        self.telemetry
            .record_policy_dispatch(self.execution_policy, output.len(), output.len());
        self.callbacks.evaluate_residual_with_telemetry(
            t,
            state,
            &self.parameters,
            output,
            &mut self.workspace,
            &mut self.telemetry,
        )
    }

    pub(crate) fn evaluate_jacobian(
        &mut self,
        t: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), RadauError> {
        // Structured Jacobian callbacks are still one logical solver-level
        // Jacobian request even when their evaluator writes a projected
        // Sparse/Banded layout directly.
        self.telemetry.count_stage(RadauStage::Jacobian);
        self.telemetry
            .record_policy_dispatch(self.execution_policy, output.len(), output.len());
        self.callbacks.evaluate_jacobian_with_telemetry(
            t,
            state,
            &self.parameters,
            output,
            &mut self.workspace,
            &mut self.telemetry,
        )
    }

    pub(crate) fn evaluate_jacobian_layout(
        &mut self,
        t: f64,
        state: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
    ) -> Result<(), RadauError> {
        self.telemetry.count_stage(RadauStage::Jacobian);
        self.telemetry
            .record_policy_dispatch(self.execution_policy, output.len(), output.len());
        self.callbacks.evaluate_jacobian_layout_with_telemetry(
            t,
            state,
            &self.parameters,
            layout,
            output,
            &mut self.workspace,
            &mut self.telemetry,
        )
    }

    /// Merge shared AOT worker/chunk telemetry after the session is no longer borrowed.
    pub(crate) fn absorb_runtime_telemetry(&mut self) {
        if let (PreparedSymbolicCallbacks::Aot(plan), SymbolicCallbackWorkspace::Aot(workspace)) =
            (self.callbacks, &self.workspace)
        {
            plan.absorb_runtime_telemetry(workspace, &mut self.telemetry);
        }
    }
}

/// Caller-owned output contract for the residual callback.
/// Minimal residual callback used by the numerical stepper.
pub(crate) trait ResidualCallback {
    fn eval(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError>;
}

/// Caller-owned output contract for a dense Jacobian callback.
/// Dense analytic Jacobian callback used by the numerical stepper.
pub(crate) trait DenseJacobianCallback {
    fn eval_into(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError>;
}

/// Combined callback boundary for a solver step using one mutable evaluator
/// session for both residual and Jacobian calls.
/// Unified callback adapter used by symbolic and direct numerical steps.
pub(crate) trait RadauStepCallbacks {
    /// Evaluate `f(t, y)` into caller-owned storage.
    fn eval_residual(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError>;
    /// Evaluate a Jacobian in the callback's default representation.
    fn eval_jacobian(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError>;
    /// Evaluate a Jacobian directly in the selected matrix layout.
    ///
    /// Symbolic callbacks override this for Sparse/Banded routes.  A generic
    /// closure has no safe way to promise a structural layout, so the default
    /// implementation deliberately returns a typed capability error instead
    /// of falling back to an implicit dense conversion.
    fn eval_jacobian_layout(
        &mut self,
        t: f64,
        y: &[f64],
        layout: RadauMatrixLayout,
        out: &mut [f64],
    ) -> Result<(), RadauError> {
        if layout == RadauMatrixLayout::Dense {
            self.eval_jacobian(t, y, out)
        } else {
            Err(RadauUnsupportedRoute::StructuredJacobianCallbackRequired.into())
        }
    }
}

impl<'a> RadauStepCallbacks for SymbolicCallbackSession<'a> {
    fn eval_residual(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        self.evaluate_residual(t, y, out)
    }

    fn eval_jacobian(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        self.evaluate_jacobian(t, y, out)
    }

    fn eval_jacobian_layout(
        &mut self,
        t: f64,
        y: &[f64],
        layout: RadauMatrixLayout,
        out: &mut [f64],
    ) -> Result<(), RadauError> {
        self.evaluate_jacobian_layout(t, y, layout, out)
    }
}

impl<F> ResidualCallback for F
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), RadauError>,
{
    fn eval(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        self(t, y, out)
    }
}

impl<F> DenseJacobianCallback for F
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), RadauError>,
{
    fn eval_into(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        self(t, y, out)
    }
}

/// Validate callback shape and reject non-finite values before linear algebra.
pub(crate) fn validate_callback_output(
    stage: RadauStage,
    expected: usize,
    values: &[f64],
) -> Result<(), RadauError> {
    if values.len() != expected {
        return Err(RadauError::ShapeMismatch {
            stage,
            expected,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(RadauError::NonFiniteCallback { stage });
    }
    Ok(())
}
