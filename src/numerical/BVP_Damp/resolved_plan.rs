//! Read-only normalized runtime plan for BVP_Damp.
//!
//! The existing solver options remain source-compatible and still contain a
//! few historical strings.  This module is the typed boundary for new code:
//! it resolves those strings and generated-backend settings once, without
//! mutating a solver or constructing symbolic/numeric state.

use super::NR_Damp_solver_damped::{BvpDerivativeScheme, DampedSolverOptions};
use super::NR_Damp_solver_frozen::FrozenSolverOptions;
use super::generated_solver_handoff::{AotBuildPolicy, AotExecutionPolicy, GeneratedBackendConfig};
use crate::somelinalg::banded::LinearSolverPolicy;
use crate::symbolic::bvp::legacy::{BvpMatrixBackend, BvpSymbolicAssemblyBackend};
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;

/// Evaluator family used by a prepared BVP callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvpEvaluatorKind {
    /// Configuration allows more than one runtime branch; preparation has not
    /// selected the concrete evaluator yet.
    Auto,
    Numeric,
    ExprLegacy,
    AtomView,
    Aot,
}

/// Linear algorithm selected by the normalized plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvpLinearAlgorithmKind {
    DenseNalgebraLu,
    FaerSparseLu,
    FaithfulBandedLu,
    Configured,
}

/// Typed nonlinear strategy visible in the normalized plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BvpStrategyKind {
    Damped,
    Frozen,
    Naive,
    Custom(String),
}

/// Numeric binding for symbolic parameters that are not Newton unknowns.
///
/// The names define the evaluator ABI and the values are prepended to the
/// runtime unknown vector. A binding is separate from the nonlinear state:
/// changing it does not change the symbolic Jacobian structure, but it does
/// invalidate numeric callback/factor state in the current compatibility ABI.
#[derive(Clone, Debug, PartialEq)]
pub struct BvpParameterBinding {
    pub names: Vec<String>,
    pub values: Vec<f64>,
}

impl BvpParameterBinding {
    /// Creates a checked binding with one value per symbolic parameter name.
    pub fn try_new<N, I>(names: I, values: Vec<f64>) -> Result<Self, String>
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        let names = names.into_iter().map(Into::into).collect::<Vec<_>>();
        if names.len() != values.len() {
            return Err(format!(
                "parameter binding length mismatch: {} names, {} values",
                names.len(),
                values.len()
            ));
        }
        if names.iter().any(|name| name.trim().is_empty()) {
            return Err("parameter binding contains an empty name".to_string());
        }
        Ok(Self { names, values })
    }
}

/// A stable, read-only view of solver configuration and backend selection.
#[derive(Clone, Debug, PartialEq)]
pub struct BvpResolvedPlan {
    pub scheme: BvpDerivativeScheme,
    pub strategy: BvpStrategyKind,
    pub matrix_backend: MatrixBackend,
    pub evaluator: BvpEvaluatorKind,
    pub linear_algorithm: BvpLinearAlgorithmKind,
    pub backend_policy: BackendSelectionPolicy,
    pub selected_backend: Option<SelectedBackendKind>,
    pub aot_execution_policy: AotExecutionPolicy,
    pub aot_build_policy: AotBuildPolicy,
    pub symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
}

impl BvpResolvedPlan {
    /// Resolves a damped options object without touching solver state.
    pub fn from_damped_options(options: &DampedSolverOptions) -> Self {
        Self::from_common(
            &options.generated_backend_config,
            &options.method,
            &options.scheme,
            strategy_from_name(&options.strategy),
        )
    }

    /// Resolves a frozen options object without touching solver state.
    pub fn from_frozen_options(options: &FrozenSolverOptions) -> Self {
        Self::from_common(
            &options.generated_backend_config,
            &options.method,
            &options.scheme,
            strategy_from_name(&options.strategy),
        )
    }

    /// Adds the backend actually selected after preparation.
    pub fn with_selected_backend(mut self, selected: SelectedBackendKind) -> Self {
        self.evaluator =
            evaluator_for_selected(selected, self.evaluator, self.symbolic_assembly_backend);
        self.selected_backend = Some(selected);
        self
    }

    fn from_common(
        config: &GeneratedBackendConfig,
        method: &str,
        scheme: &str,
        strategy: BvpStrategyKind,
    ) -> Self {
        let effective_method = config.effective_method(method);
        let matrix_backend = config
            .matrix_backend_override
            .unwrap_or_else(|| matrix_backend_for_method(&effective_method));
        let backend_policy = config.effective_backend_policy(&effective_method);
        let evaluator = evaluator_for_policy(config.symbolic_assembly_backend, backend_policy);
        let linear_algorithm = linear_algorithm_for(matrix_backend, config);
        Self {
            scheme: scheme_from_name(scheme),
            strategy,
            matrix_backend,
            evaluator,
            linear_algorithm,
            backend_policy,
            selected_backend: None,
            aot_execution_policy: config.aot_execution_policy.clone(),
            aot_build_policy: config.aot_build_policy,
            symbolic_assembly_backend: config.symbolic_assembly_backend,
        }
    }

    pub(crate) fn from_common_for_solver(
        config: &GeneratedBackendConfig,
        method: &str,
        scheme: &str,
        strategy: BvpStrategyKind,
    ) -> Self {
        Self::from_common(config, method, scheme, strategy)
    }
}

impl DampedSolverOptions {
    /// Returns the normalized plan without preparing equations or mutating
    /// the solver.  This is the preferred inspection API for new callers.
    pub fn resolved_plan(&self) -> BvpResolvedPlan {
        BvpResolvedPlan::from_damped_options(self)
    }
}

impl FrozenSolverOptions {
    /// Returns the normalized plan without preparing equations or mutating
    /// the solver.  This keeps Frozen and Damped configuration inspection
    /// on one typed contract.
    pub fn resolved_plan(&self) -> BvpResolvedPlan {
        BvpResolvedPlan::from_frozen_options(self)
    }
}

fn scheme_from_name(name: &str) -> BvpDerivativeScheme {
    match name.to_ascii_lowercase().as_str() {
        "trapezoid" | "trapezoidal" => BvpDerivativeScheme::Trapezoid,
        _ => BvpDerivativeScheme::Forward,
    }
}

pub(crate) fn strategy_from_name(name: &str) -> BvpStrategyKind {
    match name.to_ascii_lowercase().as_str() {
        "damped" => BvpStrategyKind::Damped,
        "frozen" => BvpStrategyKind::Frozen,
        "naive" => BvpStrategyKind::Naive,
        _ => BvpStrategyKind::Custom(name.to_string()),
    }
}

fn matrix_backend_for_method(method: &str) -> MatrixBackend {
    match method.to_ascii_lowercase().as_str() {
        "dense" => MatrixBackend::Dense,
        "banded" => MatrixBackend::Banded,
        "sparse_1" => MatrixBackend::CsMat,
        "sparse_2" => MatrixBackend::CsMatrix,
        _ => MatrixBackend::SparseCol,
    }
}

fn evaluator_for_policy(
    assembly: BvpSymbolicAssemblyBackend,
    policy: BackendSelectionPolicy,
) -> BvpEvaluatorKind {
    match policy {
        BackendSelectionPolicy::NumericOnly => BvpEvaluatorKind::Numeric,
        BackendSelectionPolicy::AotOnly => BvpEvaluatorKind::Aot,
        BackendSelectionPolicy::PreferAotThenLambdify
        | BackendSelectionPolicy::PreferAotThenNumeric => BvpEvaluatorKind::Auto,
        BackendSelectionPolicy::LambdifyOnly
        | BackendSelectionPolicy::PreferLambdifyThenNumeric => match assembly {
            BvpSymbolicAssemblyBackend::AtomView => BvpEvaluatorKind::AtomView,
            BvpSymbolicAssemblyBackend::ExprLegacy => BvpEvaluatorKind::ExprLegacy,
        },
    }
}

fn evaluator_for_selected(
    selected: SelectedBackendKind,
    requested: BvpEvaluatorKind,
    assembly: BvpSymbolicAssemblyBackend,
) -> BvpEvaluatorKind {
    match selected {
        SelectedBackendKind::Numeric => BvpEvaluatorKind::Numeric,
        SelectedBackendKind::Lambdify => match requested {
            BvpEvaluatorKind::Auto => match assembly {
                BvpSymbolicAssemblyBackend::AtomView => BvpEvaluatorKind::AtomView,
                BvpSymbolicAssemblyBackend::ExprLegacy => BvpEvaluatorKind::ExprLegacy,
            },
            other => other,
        },
        SelectedBackendKind::AotCompiled
        | SelectedBackendKind::AotRegisteredButNotBuilt
        | SelectedBackendKind::AotMissing => BvpEvaluatorKind::Aot,
    }
}

fn linear_algorithm_for(
    matrix_backend: MatrixBackend,
    config: &GeneratedBackendConfig,
) -> BvpLinearAlgorithmKind {
    match matrix_backend {
        MatrixBackend::Dense => BvpLinearAlgorithmKind::DenseNalgebraLu,
        MatrixBackend::Banded => match config.banded_linear_solver_config.policy {
            LinearSolverPolicy::ForceBanded | LinearSolverPolicy::Auto => {
                BvpLinearAlgorithmKind::FaithfulBandedLu
            }
            _ => BvpLinearAlgorithmKind::Configured,
        },
        MatrixBackend::SparseCol
        | MatrixBackend::CsMat
        | MatrixBackend::CsMatrix
        | MatrixBackend::ValuesOnly => BvpLinearAlgorithmKind::FaerSparseLu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damped_and_frozen_defaults_share_the_same_normalized_backend_contract() {
        let damped = BvpResolvedPlan::from_damped_options(&DampedSolverOptions::default());
        let frozen = BvpResolvedPlan::from_frozen_options(&FrozenSolverOptions::default());
        assert_eq!(damped.matrix_backend, MatrixBackend::SparseCol);
        assert_eq!(frozen.matrix_backend, MatrixBackend::SparseCol);
        assert_eq!(damped.evaluator, BvpEvaluatorKind::Auto);
        assert_eq!(frozen.evaluator, BvpEvaluatorKind::Auto);
        assert_eq!(
            damped.linear_algorithm,
            BvpLinearAlgorithmKind::FaerSparseLu
        );
    }

    #[test]
    fn banded_plan_reports_native_linear_algorithm_and_typed_assembly() {
        let options = DampedSolverOptions::banded_damped().with_banded_lambdify();
        let plan = BvpResolvedPlan::from_damped_options(&options);
        assert_eq!(plan.matrix_backend, MatrixBackend::Banded);
        assert_eq!(plan.evaluator, BvpEvaluatorKind::AtomView);
        assert_eq!(
            plan.linear_algorithm,
            BvpLinearAlgorithmKind::FaithfulBandedLu
        );
        assert_eq!(plan.selected_backend, None);
    }

    #[test]
    fn typed_matrix_backend_api_overrides_legacy_method_string() {
        let damped = DampedSolverOptions::default().with_matrix_backend(MatrixBackend::Banded);
        let frozen = FrozenSolverOptions::default().with_matrix_backend(MatrixBackend::Dense);

        assert_eq!(
            BvpResolvedPlan::from_damped_options(&damped).matrix_backend,
            MatrixBackend::Banded
        );
        assert_eq!(
            BvpResolvedPlan::from_frozen_options(&frozen).matrix_backend,
            MatrixBackend::Dense
        );
        assert_eq!(
            BvpMatrixBackend::from_legacy_method("Sparse"),
            Some(BvpMatrixBackend::FaerSparseCol)
        );
        assert_eq!(BvpMatrixBackend::FaerSparseCol.legacy_method(), "Sparse");
    }

    #[test]
    fn actual_backend_selection_can_be_attached_without_rebuilding_the_plan() {
        let plan = BvpResolvedPlan::from_damped_options(
            &DampedSolverOptions::banded_damped().with_banded_lambdify(),
        )
        .with_selected_backend(SelectedBackendKind::Lambdify);
        assert_eq!(plan.selected_backend, Some(SelectedBackendKind::Lambdify));
        assert_eq!(plan.evaluator, BvpEvaluatorKind::AtomView);
    }

    #[test]
    fn parameter_binding_is_typed_and_ordered() {
        let binding = BvpParameterBinding::try_new(["alpha", "beta"], vec![1.0, -2.0])
            .expect("matching parameter names and values should bind");
        assert_eq!(binding.names, vec!["alpha", "beta"]);
        assert_eq!(binding.values, vec![1.0, -2.0]);
        assert!(BvpParameterBinding::try_new(["alpha"], vec![1.0, 2.0]).is_err());
    }
}
