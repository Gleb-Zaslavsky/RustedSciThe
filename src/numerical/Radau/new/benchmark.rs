//! Bounded diagnostic harness for the public Radau route.
//!
//! This is intentionally not the stable solver API. It gives Criterion and
//! story tests one shared way to exercise prepared callback and linear stages
//! without duplicating workload construction.

use super::aot::AotPlan;
use super::callbacks::PreparedSymbolicCallbacks;
use super::coefficients::RadauIia5;
use super::config::{RadauAssembly, RadauConfig, RadauExecution, RadauMatrixLayout};
use super::linear::{JacobianValues, PreparedLinearBackend};
use super::solver::try_solve_symbolic_dense;
use super::telemetry::{RadauTelemetry, RadauTelemetryMode};
use crate::numerical::ivp_workloads::{
    WorkloadKind, build_workload, parameter_continuation_target,
};
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use std::time::Instant;

/// Symbolic assembly route used by a diagnostic benchmark.
#[derive(Debug, Clone, Copy)]
pub enum BenchmarkAssembly {
    /// Legacy expression lambdification.
    ExprLegacy,
    /// AtomView-native residual/Jacobian preparation.
    AtomViewNative,
}

impl BenchmarkAssembly {
    fn internal(self) -> RadauAssembly {
        match self {
            Self::ExprLegacy => RadauAssembly::ExprLegacy,
            Self::AtomViewNative => RadauAssembly::AtomViewNative,
        }
    }
}

/// Native matrix layout used by a diagnostic benchmark.
#[derive(Debug, Clone, Copy)]
pub enum BenchmarkLayout {
    /// Fully dense Jacobian storage.
    Dense,
    /// Prepared CSC pattern.
    Sparse,
    /// Compact band storage.
    Banded { lower: usize, upper: usize },
}

/// Scheduling policy used by the bounded Radau benchmark harness.
#[derive(Debug, Clone, Copy)]
pub enum BenchmarkExecutionPolicy {
    Sequential,
    Parallel { min_work: usize },
    Auto { min_work: usize },
}

impl BenchmarkExecutionPolicy {
    fn internal(self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::Sequential => IvpLambdifyExecutionPolicy::Sequential,
            Self::Parallel { min_work } => IvpLambdifyExecutionPolicy::Parallel { min_work },
            Self::Auto { min_work } => IvpLambdifyExecutionPolicy::Auto { min_work },
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel { .. } => "parallel",
            Self::Auto { .. } => "auto",
        }
    }
}

impl BenchmarkLayout {
    fn internal(self) -> RadauMatrixLayout {
        match self {
            Self::Dense => RadauMatrixLayout::Dense,
            Self::Sparse => RadauMatrixLayout::Sparse,
            Self::Banded { lower, upper } => RadauMatrixLayout::Banded { lower, upper },
        }
    }
}

/// Prepared workload owner used by debug stories and Criterion benches.
pub struct PreparedBenchmark {
    callbacks: PreparedSymbolicCallbacks,
    config: RadauConfig,
    initial_state: Vec<f64>,
    base_parameters: Vec<f64>,
}

/// Prepared AOT callback owner used by release-only callback benchmarks.
///
/// This is deliberately separate from [`PreparedBenchmark`]. The latter
/// exercises the full numerical solve, while this harness measures only the
/// generated residual/Jacobian boundary. The caller owns the output directory
/// embedded in `generated_config` and must keep it alive for the benchmark.
pub struct PreparedAotCallbacks {
    plan: AotPlan,
    layout: RadauMatrixLayout,
    initial_state: Vec<f64>,
    parameters: Vec<f64>,
}

/// Separate warm residual/Jacobian measurements for the compact report.
///
/// The checksum is intentionally returned with the timings so callers can
/// keep the callback work observable without putting checksum traversal into
/// either measured stage.
#[derive(Debug, Clone, Copy)]
pub struct CallbackBreakdown {
    pub residual_ms: f64,
    pub jacobian_ms: f64,
    pub callback_ms: f64,
    pub checksum: f64,
}

impl PreparedAotCallbacks {
    /// Prepare one generated callback route without constructing a solver.
    pub fn prepare(
        workload: WorkloadKind,
        dimension: usize,
        assembly: BenchmarkAssembly,
        layout: BenchmarkLayout,
        policy: BenchmarkExecutionPolicy,
        generated_config: SymbolicIvpGeneratedBackendConfig,
    ) -> Result<Self, String> {
        let workload_data = build_workload(workload, dimension);
        let mut preparation = RadauTelemetry::new(RadauTelemetryMode::Off);
        let initial_state = workload_data.initial_state.as_slice().to_vec();
        let parameters = workload_data.parameter_values.as_slice().to_vec();
        let layout = layout.internal();
        let plan = AotPlan::prepare_with_policy(
            assembly.internal(),
            layout,
            workload_data.equations,
            None,
            workload_data.time_variable,
            workload_data.variables,
            workload_data.parameter_names,
            generated_config,
            &mut preparation,
            policy.internal(),
        )
        .map_err(|error| error.to_string())?
        .plan;
        Ok(Self {
            plan,
            layout,
            initial_state,
            parameters,
        })
    }

    /// Repeatedly evaluate the generated residual and native Jacobian only.
    pub fn warm_callbacks(&self, repetitions: usize) -> Result<f64, String> {
        Ok(self.warm_callback_breakdown(repetitions)?.checksum)
    }

    /// Measure generated residual and Jacobian evaluation independently.
    pub fn warm_callback_breakdown(&self, repetitions: usize) -> Result<CallbackBreakdown, String> {
        let dimension = self.initial_state.len();
        let jacobian_len = match self.layout {
            RadauMatrixLayout::Dense => dimension.saturating_mul(dimension),
            RadauMatrixLayout::Sparse => self.plan.jacobian_pattern().len(),
            RadauMatrixLayout::Banded { lower, upper } => {
                (lower + upper + 1).saturating_mul(dimension)
            }
        };
        let mut workspace = self.plan.workspace_with_telemetry(RadauTelemetryMode::Off);
        let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Off);
        let mut residual = vec![0.0; dimension];
        let mut jacobian = vec![0.0; jacobian_len];
        let callback_started = Instant::now();
        let residual_started = Instant::now();
        for _ in 0..repetitions {
            self.plan
                .evaluate_residual(
                    0.0,
                    &self.initial_state,
                    &self.parameters,
                    &mut residual,
                    &mut workspace,
                    &mut telemetry,
                )
                .map_err(|error| error.to_string())?;
        }
        let residual_ms = residual_started.elapsed().as_secs_f64() * 1_000.0;
        let jacobian_started = Instant::now();
        for _ in 0..repetitions {
            self.plan
                .evaluate_jacobian(
                    0.0,
                    &self.initial_state,
                    &self.parameters,
                    self.layout,
                    &mut jacobian,
                    &mut workspace,
                    &mut telemetry,
                )
                .map_err(|error| error.to_string())?;
        }
        let jacobian_ms = jacobian_started.elapsed().as_secs_f64() * 1_000.0;
        let callback_ms = callback_started.elapsed().as_secs_f64() * 1_000.0;
        let checksum = residual
            .iter()
            .chain(jacobian.iter())
            .map(|value| value.abs())
            .sum::<f64>()
            * repetitions as f64;
        Ok(CallbackBreakdown {
            residual_ms,
            jacobian_ms,
            callback_ms,
            checksum,
        })
    }
}

impl PreparedBenchmark {
    /// Prepare one shared workload without running the numerical solver.
    pub fn prepare(
        workload: WorkloadKind,
        dimension: usize,
        assembly: BenchmarkAssembly,
        layout: BenchmarkLayout,
    ) -> Result<Self, String> {
        Self::prepare_with_policy(
            workload,
            dimension,
            assembly,
            layout,
            BenchmarkExecutionPolicy::Sequential,
        )
    }

    /// Prepare a workload with a fixed callback scheduling policy.
    pub fn prepare_with_policy(
        workload: WorkloadKind,
        dimension: usize,
        assembly: BenchmarkAssembly,
        layout: BenchmarkLayout,
        policy: BenchmarkExecutionPolicy,
    ) -> Result<Self, String> {
        let workload_data = build_workload(workload, dimension);
        let variables: Vec<&str> = workload_data.variables.iter().map(String::as_str).collect();
        let jacobian = match assembly {
            BenchmarkAssembly::ExprLegacy => Some(
                workload_data
                    .equations
                    .iter()
                    .flat_map(|equation| {
                        variables
                            .iter()
                            .map(move |variable| equation.diff(variable))
                    })
                    .collect(),
            ),
            BenchmarkAssembly::AtomViewNative => None,
        };
        let mut preparation = RadauTelemetry::new(RadauTelemetryMode::Off);
        let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry_and_policy(
            assembly.internal(),
            workload_data.equations,
            jacobian,
            &workload_data.time_variable,
            &variables,
            &workload_data
                .parameter_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            &mut preparation,
            policy.internal(),
        )
        .map_err(|error| error.to_string())?;
        let config = RadauConfig {
            execution: RadauExecution::Lambdify,
            assembly: Some(assembly.internal()),
            matrix_layout: layout.internal(),
            t_bound: benchmark_t_bound(workload),
            first_step: Some(benchmark_first_step(workload)),
            max_step: benchmark_max_step(workload),
            rtol: 1.0e-7,
            atol: 1.0e-10,
            max_steps: 2_000,
            max_newton_iterations: 8,
            max_retries: 24,
            execution_policy: policy.internal(),
            ..RadauConfig::default()
        };
        Ok(Self {
            callbacks,
            config,
            initial_state: workload_data
                .initial_state
                .as_slice()
                .iter()
                .copied()
                .collect(),
            base_parameters: workload_data
                .parameter_values
                .as_slice()
                .iter()
                .copied()
                .collect(),
        })
    }

    /// Run one prepared full solve and return a stable scalar checksum.
    pub fn solve(&self) -> Result<f64, String> {
        self.solve_with_parameters(&self.base_parameters)
    }

    /// Run one solve after binding a value-only parameter vector.
    pub fn solve_with_parameters(&self, parameters: &[f64]) -> Result<f64, String> {
        let mut session = self.callbacks.session();
        session
            .rebind_parameters(parameters)
            .map_err(|error| error.to_string())?;
        let result = try_solve_symbolic_dense(&self.config, &mut session, &self.initial_state)
            .map_err(|error| error.to_string())?;
        Ok(result.y.iter().map(|value| value.abs()).sum())
    }

    /// Exercise residual/Jacobian callbacks repeatedly with one session.
    pub fn warm_callbacks(&self, repetitions: usize) -> Result<f64, String> {
        Ok(self.warm_callback_breakdown(repetitions)?.checksum)
    }

    /// Measure residual and Jacobian evaluation independently while reusing
    /// one prepared callback session and its output workspace.
    pub fn warm_callback_breakdown(&self, repetitions: usize) -> Result<CallbackBreakdown, String> {
        let mut session = self.callbacks.session();
        session
            .rebind_parameters(&self.base_parameters)
            .map_err(|error| error.to_string())?;
        let dimension = self.initial_state.len();
        let mut residual = vec![0.0; dimension];
        let mut jacobian = match self.config.matrix_layout {
            RadauMatrixLayout::Dense => vec![0.0; dimension * dimension],
            RadauMatrixLayout::Sparse => vec![0.0; self.callbacks.jacobian_pattern().len()],
            RadauMatrixLayout::Banded { lower, upper } => {
                vec![0.0; (lower + upper + 1) * dimension]
            }
        };
        let callback_started = Instant::now();
        let residual_started = Instant::now();
        for _ in 0..repetitions {
            session
                .evaluate_residual(0.0, &self.initial_state, &mut residual)
                .map_err(|error| error.to_string())?;
        }
        let residual_ms = residual_started.elapsed().as_secs_f64() * 1_000.0;
        let jacobian_started = Instant::now();
        for _ in 0..repetitions {
            session
                .evaluate_jacobian_layout(
                    0.0,
                    &self.initial_state,
                    self.config.matrix_layout,
                    &mut jacobian,
                )
                .map_err(|error| error.to_string())?;
        }
        let jacobian_ms = jacobian_started.elapsed().as_secs_f64() * 1_000.0;
        let callback_ms = callback_started.elapsed().as_secs_f64() * 1_000.0;
        let checksum = residual
            .iter()
            .chain(jacobian.iter())
            .map(|value| value.abs())
            .sum::<f64>()
            * repetitions as f64;
        Ok(CallbackBreakdown {
            residual_ms,
            jacobian_ms,
            callback_ms,
            checksum,
        })
    }

    /// Rebind and solve a bounded continuation series with one prepared plan.
    pub fn continuation(&self, count: usize) -> Result<f64, String> {
        let mut session = self.callbacks.session();
        session
            .rebind_parameters(&self.base_parameters)
            .map_err(|error| error.to_string())?;
        let mut checksum = 0.0;
        for index in 0..count {
            let parameters = parameter_continuation_target(
                &nalgebra::DVector::from_vec(self.base_parameters.clone()),
                index,
            );
            session
                .rebind_parameters(parameters.as_slice())
                .map_err(|error| error.to_string())?;
            let result = try_solve_symbolic_dense(&self.config, &mut session, &self.initial_state)
                .map_err(|error| error.to_string())?;
            checksum += result.y.iter().map(|value| value.abs()).sum::<f64>();
        }
        Ok(checksum)
    }

    /// Isolate native matrix assembly, factorization, and linear solves.
    ///
    /// The callback frontend is not called here.  This diagnostic is useful
    /// for separating Radau's backend cost from the symbolic and Newton
    /// stages before making a layout-specific optimization decision.
    pub fn linear_kernel(&self, repetitions: usize) -> Result<f64, String> {
        let dimension = self.initial_state.len();
        let pattern = self.callbacks.jacobian_pattern();
        let backend =
            PreparedLinearBackend::from_layout_and_pattern(self.config.matrix_layout, &pattern);
        let mut workspace = backend
            .create_workspace(dimension)
            .map_err(|error| error.to_string())?;
        let coefficients = RadauIia5::new();
        let mut telemetry = RadauTelemetry::new(RadauTelemetryMode::Off);
        let mut checksum = 0.0;
        let mut dense_values = vec![0.0; dimension * dimension];
        let mut banded_values = Vec::new();
        let mut sparse_values = vec![0.0; pattern.len()];
        let mut real_rhs = vec![0.0; dimension];
        let mut complex_rhs_real = vec![0.0; dimension];
        let mut complex_rhs_imag = vec![0.0; dimension];

        match self.config.matrix_layout {
            RadauMatrixLayout::Dense => {
                for index in 0..dimension {
                    dense_values[index * dimension + index] = -1.0;
                }
            }
            RadauMatrixLayout::Banded { lower, upper } => {
                banded_values.resize((lower + upper + 1) * dimension, 0.0);
                for index in 0..dimension {
                    banded_values[upper * dimension + index] = -1.0;
                }
            }
            RadauMatrixLayout::Sparse => {
                for (index, &(row, column)) in pattern.iter().enumerate() {
                    if row == column {
                        sparse_values[index] = -1.0;
                    }
                }
            }
        }

        for _ in 0..repetitions {
            real_rhs.fill(1.0);
            complex_rhs_real.fill(1.0);
            complex_rhs_imag.fill(0.5);
            let values = match self.config.matrix_layout {
                RadauMatrixLayout::Dense => JacobianValues::Dense {
                    values: &dense_values,
                },
                RadauMatrixLayout::Banded { lower, upper } => JacobianValues::Banded {
                    values: &banded_values,
                    lower,
                    upper,
                },
                RadauMatrixLayout::Sparse => JacobianValues::Sparse {
                    values: &sparse_values,
                    entries: &pattern,
                },
            };
            backend
                .assemble_shifted_into(values, 0.1, &coefficients, &mut workspace, &mut telemetry)
                .map_err(|error| error.to_string())?;
            backend
                .factor_into(&mut workspace, &mut telemetry)
                .map_err(|error| error.to_string())?;
            backend
                .solve_real_into(&mut workspace, &mut real_rhs, &mut telemetry)
                .map_err(|error| error.to_string())?;
            backend
                .solve_complex_into(
                    &mut workspace,
                    &mut complex_rhs_real,
                    &mut complex_rhs_imag,
                    &mut telemetry,
                )
                .map_err(|error| error.to_string())?;
            checksum += real_rhs
                .iter()
                .chain(complex_rhs_real.iter())
                .chain(complex_rhs_imag.iter())
                .map(|value| value.abs())
                .sum::<f64>();
        }
        Ok(checksum)
    }
}

fn benchmark_t_bound(workload: WorkloadKind) -> f64 {
    match workload {
        WorkloadKind::DiffusionChain => 0.01,
        WorkloadKind::CombustionLike => 0.002,
        WorkloadKind::StiffScalar => 0.01,
        WorkloadKind::Robertson => 0.002,
        WorkloadKind::ThreeBody => 0.002,
    }
}

fn benchmark_first_step(workload: WorkloadKind) -> f64 {
    benchmark_t_bound(workload) * 0.1
}

fn benchmark_max_step(workload: WorkloadKind) -> f64 {
    benchmark_t_bound(workload) * 0.25
}
