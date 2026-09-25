//! Shared fixtures for the thematic LSODE2 story-test modules.
//!
//! Keep this module limited to deterministic construction helpers. Reporting,
//! timing and file I/O belong to the individual stories and must stay outside
//! measured callback work.

use super::{
    Lsode2ControllerConfig, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::ivp_telemetry::{IvpLambdifyExecutionPolicy, IvpTelemetry};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, PreparedSymbolicIvpProblem, PreparedSymbolicIvpResidualProblem,
    SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem, prepare_symbolic_ivp_residual_problem,
};
use nalgebra::DVector;

pub(crate) fn chain_equations(dimension: usize) -> Vec<Expr> {
    (0..dimension)
        .map(|index| {
            let left = (index > 0)
                .then(|| format!("y{}", index - 1))
                .unwrap_or_else(|| "0".to_string());
            let right = (index + 1 < dimension)
                .then(|| format!("y{}", index + 1))
                .unwrap_or_else(|| "0".to_string());
            Expr::parse_expression(&format!(
                "-k*y{index} + d*({left} - 2*y{index} + {right}) + q*exp(-t) - nl*y{index}*y{index}"
            ))
        })
        .collect()
}

pub(crate) fn chain_state(dimension: usize) -> DVector<f64> {
    DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| 0.2 + 0.01 * (index % 11) as f64),
    )
}

pub(crate) fn prepare_chain(
    dimension: usize,
    frontend: IvpSymbolicAssemblyBackend,
    policy: IvpLambdifyExecutionPolicy,
    telemetry: IvpTelemetry,
) -> PreparedSymbolicIvpProblem {
    let variables = (0..dimension)
        .map(|index| format!("y{index}"))
        .collect::<Vec<_>>();
    prepare_symbolic_ivp_problem(
        chain_equations(dimension),
        variables,
        "t".to_string(),
        SymbolicIvpProblemOptions::new()
            .with_equation_parameters(vec![
                "k".to_string(),
                "d".to_string(),
                "q".to_string(),
                "nl".to_string(),
            ])
            .with_equation_parameter_values(DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]))
            .with_symbolic_assembly_backend(frontend)
            .with_lambdify_execution_policy(policy)
            .with_telemetry(telemetry),
    )
    .expect("deterministic LSODE2 chain fixture should prepare")
}

pub(crate) fn prepare_chain_residual(
    dimension: usize,
    frontend: IvpSymbolicAssemblyBackend,
    policy: IvpLambdifyExecutionPolicy,
    telemetry: IvpTelemetry,
) -> PreparedSymbolicIvpResidualProblem {
    prepare_symbolic_ivp_residual_problem(
        chain_equations(dimension),
        (0..dimension).map(|index| format!("y{index}")).collect(),
        "t".to_string(),
        SymbolicIvpProblemOptions::new()
            .with_equation_parameters(vec![
                "k".to_string(),
                "d".to_string(),
                "q".to_string(),
                "nl".to_string(),
            ])
            .with_equation_parameter_values(DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]))
            .with_symbolic_assembly_backend(frontend)
            .with_lambdify_execution_policy(policy)
            .with_telemetry(telemetry),
    )
    .expect("deterministic LSODE2 residual-only chain should prepare")
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ChainMatrixRoute {
    Sparse,
    Banded,
}

impl ChainMatrixRoute {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
    }
}

/// Builds the production-shaped solver fixture shared by large performance
/// stories. Dense is intentionally not represented by this helper.
pub(crate) fn chain_solver_config(
    dimension: usize,
    frontend: IvpSymbolicAssemblyBackend,
    matrix: ChainMatrixRoute,
    policy: IvpLambdifyExecutionPolicy,
    telemetry: IvpTelemetry,
) -> Lsode2ProblemConfig {
    let assembly = match frontend {
        IvpSymbolicAssemblyBackend::ExprLegacy => Lsode2SymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView => Lsode2SymbolicAssemblyBackend::AtomView,
        // The compatibility route is intentionally not part of the
        // production large-system matrix; keep the helper exhaustive without
        // silently presenting it as AtomViewNative evidence.
        IvpSymbolicAssemblyBackend::AtomViewExprCompat => Lsode2SymbolicAssemblyBackend::ExprLegacy,
    };
    let mut config = Lsode2ProblemConfig::new(
        chain_equations(dimension),
        (0..dimension).map(|index| format!("y{index}")).collect(),
        "t".to_string(),
        0.0,
        chain_state(dimension),
        2.0,
        0.02,
        1.0e-7,
        1.0e-9,
    )
    .with_equation_parameters(vec![
        "k".to_string(),
        "d".to_string(),
        "q".to_string(),
        "nl".to_string(),
    ])
    .with_equation_parameter_values(DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010]))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_lambdify_execution_policy(policy)
    .with_telemetry(telemetry);

    config = match matrix {
        ChainMatrixRoute::Sparse => config.with_native_sparse_faer_backend(),
        ChainMatrixRoute::Banded => config.with_native_banded_faithful_backend(),
    };
    config
}

pub(crate) fn max_vector_diff(left: &DVector<f64>, right: &DVector<f64>) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

pub(crate) fn max_matrix_diff(
    left: &nalgebra::DMatrix<f64>,
    right: &nalgebra::DMatrix<f64>,
) -> f64 {
    assert_eq!(left.shape(), right.shape());
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

pub(crate) fn native_frontend() -> IvpSymbolicAssemblyBackend {
    IvpSymbolicAssemblyBackend::AtomView
}

pub(crate) fn legacy_frontend() -> IvpSymbolicAssemblyBackend {
    IvpSymbolicAssemblyBackend::ExprLegacy
}
