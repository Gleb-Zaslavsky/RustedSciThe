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
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, PreparedSymbolicIvpProblem, PreparedSymbolicIvpResidualProblem,
    SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem, prepare_symbolic_ivp_residual_problem,
};
use nalgebra::DVector;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn short_error(message: &str) -> String {
    const LIMIT: usize = 240;
    let flat = message.replace(['\r', '\n'], " ");
    if flat.len() <= LIMIT {
        flat
    } else {
        format!("{}...", &flat[..LIMIT])
    }
}

#[derive(Default)]
pub(crate) struct RaceStats {
    values: Vec<f64>,
}

impl RaceStats {
    pub(crate) fn push(&mut self, value: f64) {
        self.values.push(value);
    }

    pub(crate) fn summary(&self) -> Option<(f64, f64, f64, f64)> {
        if self.values.is_empty() {
            return None;
        }
        let n = self.values.len() as f64;
        let mean = self.values.iter().copied().sum::<f64>() / n;
        let var = self
            .values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f64>()
            / n;
        Some((
            mean,
            var.sqrt(),
            self.values.iter().copied().fold(f64::INFINITY, f64::min),
            self.values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        ))
    }
}

pub(crate) struct BackendRaceRow {
    pub(crate) matrix: &'static str,
    pub(crate) route: &'static str,
    pub(crate) counter_scope: Option<&'static str>,
    pub(crate) runs_ok: usize,
    pub(crate) runs_total: usize,
    pub(crate) first_failure: Option<String>,
    pub(crate) total_ms: RaceStats,
    pub(crate) prepare_ms: RaceStats,
    pub(crate) solve_ms: RaceStats,
    pub(crate) final_diff: RaceStats,
    pub(crate) residual_calls: RaceStats,
    pub(crate) jacobian_calls: RaceStats,
    pub(crate) nlu_or_native_linear: RaceStats,
    pub(crate) residual_ms: RaceStats,
    pub(crate) jacobian_ms: RaceStats,
    pub(crate) linear_ms: RaceStats,
    pub(crate) accepted_steps: RaceStats,
    pub(crate) rejected_steps: RaceStats,
}

impl BackendRaceRow {
    pub(crate) fn new(matrix: &'static str, route: &'static str) -> Self {
        Self {
            matrix,
            route,
            counter_scope: None,
            runs_ok: 0,
            runs_total: 0,
            first_failure: None,
            total_ms: RaceStats::default(),
            prepare_ms: RaceStats::default(),
            solve_ms: RaceStats::default(),
            final_diff: RaceStats::default(),
            residual_calls: RaceStats::default(),
            jacobian_calls: RaceStats::default(),
            nlu_or_native_linear: RaceStats::default(),
            residual_ms: RaceStats::default(),
            jacobian_ms: RaceStats::default(),
            linear_ms: RaceStats::default(),
            accepted_steps: RaceStats::default(),
            rejected_steps: RaceStats::default(),
        }
    }

    pub(crate) fn record_failure(&mut self, message: impl AsRef<str>) {
        if self.first_failure.is_none() {
            self.first_failure = Some(short_error(message.as_ref()));
        }
    }

    pub(crate) fn status_label(&self) -> String {
        let base = if self.runs_total == 0 {
            "not_run".to_string()
        } else if self.runs_ok == self.runs_total {
            format!("ok {}/{}", self.runs_ok, self.runs_total)
        } else if self.runs_ok == 0 {
            format!("failed {}/{}", self.runs_ok, self.runs_total)
        } else {
            format!("partial {}/{}", self.runs_ok, self.runs_total)
        };
        match &self.first_failure {
            Some(first_failure) if self.runs_ok < self.runs_total => {
                format!("{base}, first_failure={first_failure}")
            }
            // A post-run diagnostic (for example a long-horizon drift) can be
            // recorded after every individual solve succeeded. Do not report
            // such a row as clean, because that hides the diagnostic entirely.
            Some(first_failure) => format!("diagnostic {base}, note={first_failure}"),
            _ => base,
        }
    }
}

pub(crate) fn unique_story_short_tag() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}", std::process::id(), nanos & 0xFFFFF)
}

pub(crate) fn chain_equations(dimension: usize) -> Vec<crate::symbolic::symbolic_engine::Expr> {
    super::workload_fixtures::diffusion_chain(dimension).equations
}

pub(crate) fn chain_state(dimension: usize) -> DVector<f64> {
    super::workload_fixtures::diffusion_chain(dimension).initial_state
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
