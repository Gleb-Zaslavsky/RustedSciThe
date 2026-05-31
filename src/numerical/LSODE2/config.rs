use super::algorithm::Lsode2ControllerConfig;
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_runtime_api::{
    recommended_dense_jacobian_chunking_for_parallelism,
    recommended_residual_chunking_for_parallelism, recommended_row_chunking_for_parallelism,
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use std::path::PathBuf;
use std::sync::Arc;

pub type Lsode2AnalyticalResidualCallback =
    Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>;
pub type Lsode2AnalyticalJacobianCallback =
    Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>;

#[derive(Clone)]
pub struct Lsode2AnalyticalCallbacks {
    pub residual: Lsode2AnalyticalResidualCallback,
    pub jacobian: Lsode2AnalyticalJacobianCallback,
}

/// Time-integration family selected by the LSODE2 facade.
///
/// The first implementation milestone routes this to the crate's tested BDF
/// engine.  Keeping the method enum explicit makes room for future LSODE-like
/// Adams/BDF switching without overloading stringly-typed options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2Method {
    /// Variable-order BDF method, currently backed by `numerical::BDF`.
    Bdf,
}

impl Default for Lsode2Method {
    fn default() -> Self {
        Self::Bdf
    }
}

impl Lsode2Method {
    pub(crate) fn as_bdf_method_name(self) -> String {
        match self {
            Self::Bdf => "BDF".to_string(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Bdf => "bdf",
        }
    }
}

/// Jacobian assembly/evaluation route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2JacobianBackend {
    /// Use generated symbolic IVP machinery, including Lambdify/AOT selection.
    SymbolicGenerated,
    /// Use user-provided analytical residual/Jacobian callbacks.
    ///
    /// Current execution support is native sparse/banded solve paths.
    AnalyticClosure,
    /// Reserved for finite-difference Jacobians.
    FiniteDifference,
}

impl Default for Lsode2JacobianBackend {
    fn default() -> Self {
        Self::SymbolicGenerated
    }
}

impl Lsode2JacobianBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::SymbolicGenerated => "symbolic_generated",
            Self::AnalyticClosure => "analytic_closure",
            Self::FiniteDifference => "finite_difference",
        }
    }
}

/// Linear solver backend used for Newton systems.
///
/// Dense mode keeps the legacy dense Jacobian route. Sparse and banded modes
/// install native symbolic Jacobian evaluators in the reused BDF engine, so
/// Newton systems can be assembled and factored without a dense matrix
/// round-trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2LinearSolverBackend {
    /// Dense generated/lambdified Jacobian plus nalgebra dense LU.
    Dense,
    /// Native sparse triplet Jacobian plus `faer` sparse LU.
    SparseFaer,
    /// Native banded Jacobian plus faithful LAPACK-style banded LU.
    BandedFaithful,
}

impl Default for Lsode2LinearSolverBackend {
    fn default() -> Self {
        Self::Dense
    }
}

impl Lsode2LinearSolverBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense_lu",
            Self::SparseFaer => "faer_sparse_lu",
            Self::BandedFaithful => "lapack_faithful_banded_lu",
        }
    }
}

/// Symbolic expression assembly backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lsode2SymbolicAssemblyBackend {
    /// Legacy expression traversal backend.
    #[default]
    ExprLegacy,
    /// Packed `AtomView` symbolic backend.
    AtomView,
}

impl Lsode2SymbolicAssemblyBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "expr_legacy",
            Self::AtomView => "atom_view",
        }
    }
}

/// Symbolic execution mode for residual/Jacobian callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2SymbolicExecutionMode {
    LambdifyExpr,
    Aot {
        toolchain: Lsode2AotToolchain,
        profile: Lsode2AotProfile,
    },
}

impl Default for Lsode2SymbolicExecutionMode {
    fn default() -> Self {
        Self::LambdifyExpr
    }
}

impl Lsode2SymbolicExecutionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::LambdifyExpr => "lambdify_expr",
            Self::Aot { .. } => "aot",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lsode2AotToolchain {
    #[default]
    CGcc,
    CTcc,
    Zig,
    Rust,
}

impl Lsode2AotToolchain {
    pub fn label(self) -> &'static str {
        match self {
            Self::CGcc => "c_gcc",
            Self::CTcc => "c_tcc",
            Self::Zig => "zig",
            Self::Rust => "rust",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lsode2AotProfile {
    Debug,
    #[default]
    Release,
}

impl Lsode2AotProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

/// Residual/Jacobian source declaration for LSODE2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2ResidualJacobianSource {
    Analytical,
    Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend,
        execution: Lsode2SymbolicExecutionMode,
    },
}

impl Default for Lsode2ResidualJacobianSource {
    fn default() -> Self {
        Self::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        }
    }
}

impl Lsode2ResidualJacobianSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Analytical => "analytical",
            Self::Symbolic { .. } => "symbolic",
        }
    }
}

/// Jacobian/Newton-system structure used by LSODE2 linear backend policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2LinearSystemStructure {
    Dense,
    Sparse,
    Banded { kl: usize, ku: usize },
}

impl Default for Lsode2LinearSystemStructure {
    fn default() -> Self {
        Self::Dense
    }
}

impl Lsode2LinearSystemStructure {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::Banded { .. } => "banded",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2LinearSolverChoice {
    DenseLu,
    FaerSparseLu,
    LapackFaithfulBandedLu,
}

impl Lsode2LinearSolverChoice {
    pub fn label(self) -> &'static str {
        match self {
            Self::DenseLu => "dense_lu",
            Self::FaerSparseLu => "faer_sparse_lu",
            Self::LapackFaithfulBandedLu => "lapack_faithful_banded_lu",
        }
    }

    pub fn to_backend(self) -> Lsode2LinearSolverBackend {
        match self {
            Self::DenseLu => Lsode2LinearSolverBackend::Dense,
            Self::FaerSparseLu => Lsode2LinearSolverBackend::SparseFaer,
            Self::LapackFaithfulBandedLu => Lsode2LinearSolverBackend::BandedFaithful,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lsode2LinearSolverPolicy {
    #[default]
    Auto,
    Force(Lsode2LinearSolverChoice),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2StopComparator {
    GreaterEqual,
    LessEqual,
    AbsDistance,
}

impl Lsode2StopComparator {
    pub fn label(self) -> &'static str {
        match self {
            Self::GreaterEqual => "ge",
            Self::LessEqual => "le",
            Self::AbsDistance => "abs_distance",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2StopCondition {
    pub variable: String,
    pub target: f64,
    pub comparator: Lsode2StopComparator,
    pub tolerance: f64,
}

impl Lsode2StopCondition {
    pub fn greater_equal(variable: impl Into<String>, target: f64) -> Self {
        Self {
            variable: variable.into(),
            target,
            comparator: Lsode2StopComparator::GreaterEqual,
            tolerance: 0.0,
        }
    }

    pub fn less_equal(variable: impl Into<String>, target: f64) -> Self {
        Self {
            variable: variable.into(),
            target,
            comparator: Lsode2StopComparator::LessEqual,
            tolerance: 0.0,
        }
    }

    pub fn abs_distance(variable: impl Into<String>, target: f64, tolerance: f64) -> Self {
        Self {
            variable: variable.into(),
            target,
            comparator: Lsode2StopComparator::AbsDistance,
            tolerance: tolerance.abs(),
        }
    }
}

/// Resolved backend plan after policy/validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lsode2ResolvedPlan {
    pub source: Lsode2ResidualJacobianSource,
    pub structure: Lsode2LinearSystemStructure,
    pub linear_solver: Lsode2LinearSolverChoice,
    pub linear_solver_reason: &'static str,
}

/// Optional solver-owned native execution hook for LSODE2.
///
/// This does not replace the current bridge-backed `solve()` path yet.
/// Instead it allows the solver to run a bounded native sparse/banded stepping
/// fragment under explicit configuration, so native execution can mature behind
/// a stable API before it becomes the default path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2NativeExecutionConfig {
    /// Execute through the legacy bridge-backed BDF solver path.
    BridgeSolve,
    /// Compatibility alias for bridge-backed solve path.
    Disabled,
    ProbeBeforeBridge {
        max_step_attempts: usize,
        max_accepted_steps: usize,
    },
    NativeSolve {
        max_step_attempts: usize,
        max_accepted_steps: usize,
    },
}

impl Default for Lsode2NativeExecutionConfig {
    fn default() -> Self {
        Self::native_solve(200_000, 200_000)
    }
}

impl Lsode2NativeExecutionConfig {
    pub fn bridge_solve() -> Self {
        Self::BridgeSolve
    }

    /// Compatibility alias for [`Self::bridge_solve`].
    pub fn disabled() -> Self {
        Self::bridge_solve()
    }

    pub fn probe_before_bridge(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self::ProbeBeforeBridge {
            max_step_attempts: max_step_attempts.max(1),
            max_accepted_steps: max_accepted_steps.max(1),
        }
    }

    /// Compatibility alias for [`Self::probe_before_bridge`].
    pub fn preview_before_bridge(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self::probe_before_bridge(max_step_attempts, max_accepted_steps)
    }

    /// Runs the LSODE2-native sparse/banded step engine as the active solve
    /// path.
    ///
    /// This is the preferred API name for native execution mode. The legacy
    /// `experimental_native_solve` constructor remains as a compatibility alias.
    pub fn native_solve(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self::NativeSolve {
            max_step_attempts: max_step_attempts.max(1),
            max_accepted_steps: max_accepted_steps.max(1),
        }
    }

    /// Preferred naming for the LSODE2 faithful native BDF solve path.
    pub fn faithful_bdf_solve(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self::native_solve(max_step_attempts, max_accepted_steps)
    }

    /// Compatibility alias for [`Self::native_solve`].
    pub fn experimental_native_solve(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self::native_solve(max_step_attempts, max_accepted_steps)
    }
}

/// Backend configuration for LSODE2.
///
/// This groups the knobs that were missing from the old LSODE prototype.
/// `generated_backend` controls the symbolic residual lifecycle for every mode.
/// In dense mode it prepares residual + dense Jacobian together. In native
/// sparse/banded modes residuals are prepared through Lambdify/AOT and Jacobians
/// stay in native sparse-triplet/compact-banded storage; for symbolic AOT
/// execution, Jacobian value callbacks are also prepared from compiled sparse AOT
/// runtime backends.
#[derive(Clone)]
pub struct Lsode2BackendConfig {
    pub jacobian_backend: Lsode2JacobianBackend,
    pub linear_solver_backend: Lsode2LinearSolverBackend,
    pub generated_backend: SymbolicIvpGeneratedBackendConfig,
}

impl Default for Lsode2BackendConfig {
    fn default() -> Self {
        Self {
            jacobian_backend: Lsode2JacobianBackend::SymbolicGenerated,
            linear_solver_backend: Lsode2LinearSolverBackend::Dense,
            generated_backend: SymbolicIvpGeneratedBackendConfig::defaults(),
        }
    }
}

impl Lsode2BackendConfig {
    /// Dense symbolic route: generated/lambdified residual and dense Jacobian,
    /// followed by the legacy dense LU backend.
    pub fn dense_symbolic_defaults() -> Self {
        Self::default()
    }

    /// Dense symbolic route with a high-level generated-backend lifecycle mode.
    pub fn dense_symbolic_mode(mode: DenseIvpGeneratedBackendMode) -> Self {
        Self::dense_symbolic_defaults().with_dense_generated_backend_mode(mode)
    }

    /// Dense symbolic route using compiled dense IVP callbacks via `C + tcc`.
    ///
    /// This preset optimizes for low AOT build/startup latency. It is a dense
    /// Jacobian route, not the native sparse/banded LSODE2 path.
    pub fn dense_aot_c_tcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::dense_symbolic_defaults().with_dense_generated_backend_c_tcc(output_parent_dir)
    }

    /// Dense symbolic route using compiled dense IVP callbacks via `C + gcc`.
    ///
    /// This is the runtime-oriented dense AOT preset for repeated solves.
    pub fn dense_aot_c_gcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::dense_symbolic_defaults().with_dense_generated_backend_c_gcc(output_parent_dir)
    }

    /// Dense symbolic route using compiled dense IVP callbacks via Zig.
    pub fn dense_aot_zig(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::dense_symbolic_defaults().with_dense_generated_backend_zig(output_parent_dir)
    }

    /// Dense symbolic route with the crate's repeated-solve AOT preset.
    pub fn dense_aot_for_repeated_solves(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::dense_symbolic_defaults()
            .with_dense_generated_backend_for_repeated_solves(output_parent_dir)
    }

    /// Native sparse route: symbolic residual plus symbolic sparse Jacobian
    /// triplets factored by `faer` sparse LU.
    ///
    /// This path intentionally avoids constructing a dense Jacobian. Attach a
    /// generated backend config when you want residual AOT while keeping the
    /// Jacobian native sparse.
    pub fn native_sparse_faer() -> Self {
        Self::default().with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer)
    }

    /// Native sparse route with explicit Lambdify/AOT residual lifecycle config.
    pub fn native_sparse_faer_with_generated_backend(
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        Self::native_sparse_faer().with_generated_backend(config)
    }

    /// Native sparse route with AOT lifecycle via `C + tcc`.
    ///
    /// With `residual_jacobian_source = Symbolic { execution = Aot { .. } }`,
    /// both residual and sparse Jacobian values are compiled and linked.
    pub fn native_sparse_faer_aot_c_tcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_sparse_faer_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Native sparse route with AOT lifecycle via `C + gcc`.
    pub fn native_sparse_faer_aot_c_gcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_sparse_faer_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        )
    }

    /// Native sparse route with AOT lifecycle via Zig.
    pub fn native_sparse_faer_aot_zig(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_sparse_faer_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        )
    }

    /// Native banded route: symbolic residual plus symbolic compact-banded
    /// Jacobian factored by the faithful LAPACK-style banded LU.
    ///
    /// This is the preferred banded LSODE2 backend once the problem structure is
    /// actually banded. It avoids dense Jacobian materialization. Attach a
    /// generated backend config when you want residual AOT while keeping the
    /// Jacobian native banded.
    pub fn native_banded_faithful() -> Self {
        Self::default().with_linear_solver_backend(Lsode2LinearSolverBackend::BandedFaithful)
    }

    /// Native banded route with explicit Lambdify/AOT residual lifecycle config.
    pub fn native_banded_faithful_with_generated_backend(
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        Self::native_banded_faithful().with_generated_backend(config)
    }

    /// Native banded route with AOT lifecycle via `C + tcc`.
    ///
    /// With `residual_jacobian_source = Symbolic { execution = Aot { .. } }`,
    /// Jacobian values are sourced from compiled sparse AOT callbacks and mapped
    /// into compact banded storage.
    pub fn native_banded_faithful_aot_c_tcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_banded_faithful_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Native banded route with AOT lifecycle via `C + gcc`.
    pub fn native_banded_faithful_aot_c_gcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_banded_faithful_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        )
    }

    /// Native banded route with AOT lifecycle via Zig.
    pub fn native_banded_faithful_aot_zig(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_banded_faithful_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        )
    }

    pub fn with_generated_backend(mut self, config: SymbolicIvpGeneratedBackendConfig) -> Self {
        self.generated_backend = config;
        self
    }

    /// Overrides generated-backend AOT chunking options.
    ///
    /// This controls how generated residual/Jacobian kernels are split into
    /// runtime chunks. It is useful when tuning backend-level parallel work
    /// granularity.
    pub fn with_generated_backend_aot_options(mut self, options: SymbolicIvpAotOptions) -> Self {
        self.generated_backend = self.generated_backend.clone().with_aot_options(options);
        self
    }

    /// Overrides sparse-Jacobian chunking for native sparse/banded generated
    /// Jacobian callbacks.
    pub fn with_generated_backend_sparse_chunking_strategy(
        mut self,
        strategy: SparseChunkingStrategy,
    ) -> Self {
        self.generated_backend = self
            .generated_backend
            .clone()
            .with_sparse_jacobian_chunking_strategy(strategy);
        self
    }

    /// Sets explicit target chunk counts for generated residual and dense
    /// Jacobian runtime plans.
    ///
    /// This is a direct, deterministic way to control generated backend
    /// parallel granularity without relying on machine-dependent auto sizing.
    pub fn with_generated_backend_target_chunks(
        self,
        residual_target_chunks: usize,
        jacobian_target_chunks: usize,
    ) -> Self {
        assert!(
            residual_target_chunks > 0,
            "residual_target_chunks must be positive"
        );
        assert!(
            jacobian_target_chunks > 0,
            "jacobian_target_chunks must be positive"
        );
        self.with_generated_backend_aot_options(SymbolicIvpAotOptions {
            residual_strategy: ResidualChunkingStrategy::ByTargetChunkCount {
                target_chunks: residual_target_chunks,
            },
            jacobian_strategy: DenseJacobianChunkingStrategy::ByTargetChunkCount {
                target_chunks: jacobian_target_chunks,
            },
        })
        .with_generated_backend_sparse_chunking_strategy(
            SparseChunkingStrategy::ByTargetChunkCount {
                target_chunks: jacobian_target_chunks,
            },
        )
    }

    /// Applies one high-level dense generated-backend lifecycle mode.
    pub fn with_dense_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        let mut config = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
        config.resolver = self.generated_backend.resolver.clone();
        config.aot_options = self.generated_backend.aot_options;
        config.sparse_jacobian_chunking_strategy =
            self.generated_backend.sparse_jacobian_chunking_strategy;
        config.aot_codegen_backend = self.generated_backend.aot_codegen_backend;
        config.aot_c_compiler = self.generated_backend.aot_c_compiler.clone();
        config.output_parent_dir = self.generated_backend.output_parent_dir.clone();
        config.crate_name_override = self.generated_backend.crate_name_override.clone();
        config.module_name_override = self.generated_backend.module_name_override.clone();
        self.generated_backend = config;
        self
    }

    /// Uses compiled dense IVP callbacks via `C + tcc`.
    pub fn with_dense_generated_backend_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Uses compiled dense IVP callbacks via `C + gcc`.
    pub fn with_dense_generated_backend_c_gcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        )
    }

    /// Uses compiled dense IVP callbacks via Zig.
    pub fn with_dense_generated_backend_zig(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        )
    }

    /// Uses the dense IVP repeated-solve preset.
    pub fn with_dense_generated_backend_for_repeated_solves(
        self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .for_repeated_solves(),
        )
    }

    pub fn with_linear_solver_backend(mut self, backend: Lsode2LinearSolverBackend) -> Self {
        self.linear_solver_backend = backend;
        self
    }

    pub fn with_jacobian_backend(mut self, backend: Lsode2JacobianBackend) -> Self {
        self.jacobian_backend = backend;
        self
    }
}

/// Complete symbolic IVP setup consumed by [`crate::numerical::LSODE2::Lsode2Solver`].
#[derive(Clone)]
pub struct Lsode2ProblemConfig {
    pub eq_system: Vec<Expr>,
    pub values: Vec<String>,
    pub arg: String,
    pub equation_parameters: Option<Vec<String>>,
    pub equation_parameter_values: Option<DVector<f64>>,
    pub analytical_callbacks: Option<Lsode2AnalyticalCallbacks>,
    pub method: Lsode2Method,
    pub t0: f64,
    pub y0: DVector<f64>,
    pub t_bound: f64,
    pub max_step: f64,
    pub rtol: f64,
    pub atol: f64,
    pub jac_sparsity: Option<DMatrix<f64>>,
    pub vectorized: bool,
    pub first_step: Option<f64>,
    pub stop_conditions: Vec<Lsode2StopCondition>,
    pub residual_jacobian_source: Lsode2ResidualJacobianSource,
    pub linear_system_structure: Lsode2LinearSystemStructure,
    pub linear_solver_policy: Lsode2LinearSolverPolicy,
    pub controller: Lsode2ControllerConfig,
    pub backend: Lsode2BackendConfig,
    pub native_execution: Lsode2NativeExecutionConfig,
}

impl Lsode2ProblemConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
    ) -> Self {
        Self {
            eq_system,
            values,
            arg,
            equation_parameters: None,
            equation_parameter_values: None,
            analytical_callbacks: None,
            method: Lsode2Method::Bdf,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,
            jac_sparsity: None,
            vectorized: false,
            first_step: None,
            stop_conditions: Vec::new(),
            residual_jacobian_source: Lsode2ResidualJacobianSource::default(),
            linear_system_structure: Lsode2LinearSystemStructure::default(),
            linear_solver_policy: Lsode2LinearSolverPolicy::default(),
            controller: Lsode2ControllerConfig::default(),
            backend: Lsode2BackendConfig::default(),
            native_execution: Lsode2NativeExecutionConfig::default(),
        }
    }

    pub fn with_controller(mut self, controller: Lsode2ControllerConfig) -> Self {
        self.controller = controller;
        self
    }

    /// Declares intent to use the non-stiff Adams family only.
    ///
    /// Bridge execution currently remains BDF-only and rejects this mode.
    /// Native execution modes (`probe_before_bridge` and `native_solve`) are
    /// allowed to run Adams-like stepping so
    /// we can mirror LSODE control-plane behavior while the native path is
    /// being hardened.
    pub fn with_adams_only_controller(self) -> Self {
        self.with_controller(Lsode2ControllerConfig::adams_only())
    }

    pub fn with_bdf_only_controller(self) -> Self {
        self.with_controller(Lsode2ControllerConfig::bdf_only())
    }

    /// Declares intent to use automatic Adams/BDF switching.
    ///
    /// Note on parity: classic `LSODE` uses a fixed method selected by `MF`
    /// (Adams or BDF). Automatic stiffness/method switching is an `LSODA`-style
    /// behavior and is therefore exposed here as an explicit opt-in extension.
    ///
    /// The current milestone still executes through the tested BDF engine and
    /// reports that fallback explicitly in [`crate::numerical::LSODE2::Lsode2SolveSummary`].
    pub fn with_automatic_adams_bdf_controller(self) -> Self {
        self.with_controller(Lsode2ControllerConfig::automatic_adams_bdf())
    }

    pub fn with_backend(mut self, backend: Lsode2BackendConfig) -> Self {
        self.backend = backend;
        self.sync_policy_fields_from_legacy_backend();
        self
    }

    pub fn with_residual_jacobian_source(mut self, source: Lsode2ResidualJacobianSource) -> Self {
        self.residual_jacobian_source = source;
        self.sync_legacy_backend_from_policy();
        self
    }

    pub fn with_linear_system_structure(mut self, structure: Lsode2LinearSystemStructure) -> Self {
        self.linear_system_structure = structure;
        self.sync_legacy_backend_from_policy();
        self
    }

    pub fn with_linear_solver_policy(mut self, policy: Lsode2LinearSolverPolicy) -> Self {
        self.linear_solver_policy = policy;
        self.sync_legacy_backend_from_policy();
        self
    }

    pub fn resolve_plan(&self) -> Lsode2ResolvedPlan {
        let linear_solver = match self.linear_solver_policy {
            Lsode2LinearSolverPolicy::Force(choice) => choice,
            Lsode2LinearSolverPolicy::Auto => match self.linear_system_structure {
                Lsode2LinearSystemStructure::Dense => Lsode2LinearSolverChoice::DenseLu,
                Lsode2LinearSystemStructure::Sparse => Lsode2LinearSolverChoice::FaerSparseLu,
                Lsode2LinearSystemStructure::Banded { .. } => {
                    Lsode2LinearSolverChoice::LapackFaithfulBandedLu
                }
            },
        };

        let linear_solver_reason = match self.linear_solver_policy {
            Lsode2LinearSolverPolicy::Force(_) => "forced_by_linear_solver_policy",
            Lsode2LinearSolverPolicy::Auto => match self.linear_system_structure {
                Lsode2LinearSystemStructure::Dense => "auto_from_linear_structure_dense",
                Lsode2LinearSystemStructure::Sparse => "auto_from_linear_structure_sparse",
                Lsode2LinearSystemStructure::Banded { .. } => "auto_from_linear_structure_banded",
            },
        };

        Lsode2ResolvedPlan {
            source: self.residual_jacobian_source,
            structure: self.linear_system_structure,
            linear_solver,
            linear_solver_reason,
        }
    }

    pub(crate) fn sync_legacy_backend_from_policy(&mut self) {
        let resolved = self.resolve_plan();
        self.backend.linear_solver_backend = resolved.linear_solver.to_backend();
        self.backend.jacobian_backend = match self.residual_jacobian_source {
            Lsode2ResidualJacobianSource::Analytical => {
                if self.backend.jacobian_backend == Lsode2JacobianBackend::FiniteDifference {
                    Lsode2JacobianBackend::FiniteDifference
                } else {
                    Lsode2JacobianBackend::AnalyticClosure
                }
            }
            Lsode2ResidualJacobianSource::Symbolic { .. } => {
                Lsode2JacobianBackend::SymbolicGenerated
            }
        };

        if let Lsode2ResidualJacobianSource::Symbolic { execution, .. } =
            self.residual_jacobian_source
        {
            match execution {
                Lsode2SymbolicExecutionMode::LambdifyExpr => {
                    self.backend.generated_backend.build_policy =
                        SymbolicIvpAotBuildPolicy::UseIfAvailable;
                }
                Lsode2SymbolicExecutionMode::Aot { toolchain, profile } => {
                    if matches!(
                        self.backend.generated_backend.build_policy,
                        SymbolicIvpAotBuildPolicy::UseIfAvailable
                    ) {
                        self.backend.generated_backend.build_policy =
                            SymbolicIvpAotBuildPolicy::BuildIfMissing {
                                profile: match profile {
                                    Lsode2AotProfile::Debug => AotBuildProfile::Debug,
                                    Lsode2AotProfile::Release => AotBuildProfile::Release,
                                },
                            };
                    }
                    match toolchain {
                        Lsode2AotToolchain::CGcc => {
                            self.backend.generated_backend.aot_codegen_backend =
                                AotCodegenBackend::C;
                            self.backend.generated_backend.aot_c_compiler = Some("gcc".to_string());
                        }
                        Lsode2AotToolchain::CTcc => {
                            self.backend.generated_backend.aot_codegen_backend =
                                AotCodegenBackend::C;
                            self.backend.generated_backend.aot_c_compiler = Some("tcc".to_string());
                        }
                        Lsode2AotToolchain::Zig => {
                            self.backend.generated_backend.aot_codegen_backend =
                                AotCodegenBackend::Zig;
                            self.backend.generated_backend.aot_c_compiler = None;
                        }
                        Lsode2AotToolchain::Rust => {
                            self.backend.generated_backend.aot_codegen_backend =
                                AotCodegenBackend::Rust;
                            self.backend.generated_backend.aot_c_compiler = None;
                        }
                    }
                    if self.backend.generated_backend.output_parent_dir.is_none()
                        && matches!(
                            self.backend.generated_backend.build_policy,
                            SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
                                | SymbolicIvpAotBuildPolicy::RebuildAlways { .. }
                        )
                    {
                        self.backend.generated_backend.output_parent_dir =
                            Some(PathBuf::from("target/lsode2-generated"));
                    }
                }
            }
        }
    }

    fn sync_policy_fields_from_legacy_backend(&mut self) {
        let preserve_analytical_source = self.analytical_callbacks.is_some()
            && self.residual_jacobian_source == Lsode2ResidualJacobianSource::Analytical;
        let (preserved_assembly, preserved_execution) = match self.residual_jacobian_source {
            Lsode2ResidualJacobianSource::Symbolic {
                assembly,
                execution,
            } => (assembly, Some(execution)),
            Lsode2ResidualJacobianSource::Analytical => {
                (Lsode2SymbolicAssemblyBackend::ExprLegacy, None)
            }
        };
        if preserve_analytical_source {
            self.residual_jacobian_source = Lsode2ResidualJacobianSource::Analytical;
            if self.backend.jacobian_backend != Lsode2JacobianBackend::FiniteDifference {
                self.backend.jacobian_backend = Lsode2JacobianBackend::AnalyticClosure;
            }
        } else {
            self.residual_jacobian_source = match self.backend.jacobian_backend {
                Lsode2JacobianBackend::SymbolicGenerated => {
                    Lsode2ResidualJacobianSource::Symbolic {
                        assembly: preserved_assembly,
                        execution: preserved_execution.unwrap_or_else(|| {
                            execution_mode_from_generated_backend(&self.backend.generated_backend)
                        }),
                    }
                }
                Lsode2JacobianBackend::AnalyticClosure => Lsode2ResidualJacobianSource::Analytical,
                Lsode2JacobianBackend::FiniteDifference => Lsode2ResidualJacobianSource::Analytical,
            };
        };

        let (structure, choice) = match self.backend.linear_solver_backend {
            Lsode2LinearSolverBackend::Dense => (
                Lsode2LinearSystemStructure::Dense,
                Lsode2LinearSolverChoice::DenseLu,
            ),
            Lsode2LinearSolverBackend::SparseFaer => (
                Lsode2LinearSystemStructure::Sparse,
                Lsode2LinearSolverChoice::FaerSparseLu,
            ),
            Lsode2LinearSolverBackend::BandedFaithful => (
                Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
                Lsode2LinearSolverChoice::LapackFaithfulBandedLu,
            ),
        };
        self.linear_system_structure = structure;
        self.linear_solver_policy = Lsode2LinearSolverPolicy::Force(choice);
    }

    pub fn with_native_execution(mut self, native_execution: Lsode2NativeExecutionConfig) -> Self {
        self.native_execution = native_execution;
        self
    }

    pub fn with_bridge_solve(self) -> Self {
        self.with_native_execution(Lsode2NativeExecutionConfig::bridge_solve())
    }

    pub fn with_native_preview_before_bridge(
        self,
        max_step_attempts: usize,
        max_accepted_steps: usize,
    ) -> Self {
        self.with_native_execution(Lsode2NativeExecutionConfig::probe_before_bridge(
            max_step_attempts,
            max_accepted_steps,
        ))
    }

    pub fn with_native_probe_before_bridge(
        self,
        max_step_attempts: usize,
        max_accepted_steps: usize,
    ) -> Self {
        self.with_native_execution(Lsode2NativeExecutionConfig::probe_before_bridge(
            max_step_attempts,
            max_accepted_steps,
        ))
    }

    pub fn with_experimental_native_solve(
        self,
        max_step_attempts: usize,
        max_accepted_steps: usize,
    ) -> Self {
        self.with_native_execution(Lsode2NativeExecutionConfig::experimental_native_solve(
            max_step_attempts,
            max_accepted_steps,
        ))
    }

    pub fn with_native_solve(self, max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        self.with_native_execution(Lsode2NativeExecutionConfig::native_solve(
            max_step_attempts,
            max_accepted_steps,
        ))
    }

    pub fn with_faithful_bdf_solve(
        self,
        max_step_attempts: usize,
        max_accepted_steps: usize,
    ) -> Self {
        self.with_native_execution(Lsode2NativeExecutionConfig::faithful_bdf_solve(
            max_step_attempts,
            max_accepted_steps,
        ))
    }

    /// Uses dense generated/lambdified residual and dense Jacobian callbacks.
    pub fn with_dense_symbolic_backend(self) -> Self {
        self.with_backend(Lsode2BackendConfig::dense_symbolic_defaults())
    }

    /// Uses dense symbolic callbacks with a high-level generated-backend mode.
    pub fn with_dense_symbolic_backend_mode(self, mode: DenseIvpGeneratedBackendMode) -> Self {
        self.with_backend(Lsode2BackendConfig::dense_symbolic_mode(mode))
    }

    /// Uses dense symbolic callbacks compiled via `C + tcc`.
    pub fn with_dense_aot_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_backend(Lsode2BackendConfig::dense_aot_c_tcc(output_parent_dir))
    }

    /// Uses dense symbolic callbacks compiled via `C + gcc`.
    pub fn with_dense_aot_c_gcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_backend(Lsode2BackendConfig::dense_aot_c_gcc(output_parent_dir))
    }

    /// Uses dense symbolic callbacks compiled via Zig.
    pub fn with_dense_aot_zig(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_backend(Lsode2BackendConfig::dense_aot_zig(output_parent_dir))
    }

    /// Uses the dense IVP repeated-solve AOT preset.
    pub fn with_dense_aot_for_repeated_solves(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_backend(Lsode2BackendConfig::dense_aot_for_repeated_solves(
            output_parent_dir,
        ))
    }

    /// Uses native sparse symbolic Jacobian triplets plus `faer` sparse LU.
    pub fn with_native_sparse_faer_backend(self) -> Self {
        self.with_backend(Lsode2BackendConfig::native_sparse_faer())
    }

    /// Uses native sparse symbolic Jacobian triplets plus a configured residual
    /// Lambdify/AOT lifecycle.
    pub fn with_native_sparse_faer_generated_backend(
        self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.with_backend(Lsode2BackendConfig::native_sparse_faer_with_generated_backend(config))
    }

    /// Uses native sparse Jacobian storage and residual-only AOT via `C + tcc`.
    pub fn with_native_sparse_faer_aot_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_backend(Lsode2BackendConfig::native_sparse_faer_aot_c_tcc(
            output_parent_dir,
        ))
    }

    /// Uses native sparse Jacobian storage and residual-only AOT via `C + gcc`.
    pub fn with_native_sparse_faer_aot_c_gcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_backend(Lsode2BackendConfig::native_sparse_faer_aot_c_gcc(
            output_parent_dir,
        ))
    }

    /// Uses native sparse Jacobian storage and residual-only AOT via Zig.
    pub fn with_native_sparse_faer_aot_zig(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_backend(Lsode2BackendConfig::native_sparse_faer_aot_zig(
            output_parent_dir,
        ))
    }

    /// Uses native compact-banded symbolic Jacobian plus faithful banded LU.
    pub fn with_native_banded_faithful_backend(self) -> Self {
        self.with_backend(Lsode2BackendConfig::native_banded_faithful())
    }

    /// Uses native compact-banded symbolic Jacobian plus a configured residual
    /// Lambdify/AOT lifecycle.
    pub fn with_native_banded_faithful_generated_backend(
        self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.with_backend(
            Lsode2BackendConfig::native_banded_faithful_with_generated_backend(config),
        )
    }

    /// Uses native compact-banded Jacobian storage and residual-only AOT via
    /// `C + tcc`.
    pub fn with_native_banded_faithful_aot_c_tcc(
        self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.with_backend(Lsode2BackendConfig::native_banded_faithful_aot_c_tcc(
            output_parent_dir,
        ))
    }

    /// Uses native compact-banded Jacobian storage and residual-only AOT via
    /// `C + gcc`.
    pub fn with_native_banded_faithful_aot_c_gcc(
        self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.with_backend(Lsode2BackendConfig::native_banded_faithful_aot_c_gcc(
            output_parent_dir,
        ))
    }

    /// Uses native compact-banded Jacobian storage and residual-only AOT via Zig.
    pub fn with_native_banded_faithful_aot_zig(
        self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.with_backend(Lsode2BackendConfig::native_banded_faithful_aot_zig(
            output_parent_dir,
        ))
    }

    pub fn with_first_step(mut self, first_step: Option<f64>) -> Self {
        self.first_step = first_step;
        self
    }

    /// Adds a stop condition with `current(variable) >= target`.
    ///
    /// Typical combustion-style usage:
    /// `with_stop_condition("eta", 0.999)`.
    pub fn with_stop_condition(self, variable: impl Into<String>, target: f64) -> Self {
        self.with_stop_condition_ge(variable, target)
    }

    pub fn with_stop_condition_ge(mut self, variable: impl Into<String>, target: f64) -> Self {
        self.stop_conditions
            .push(Lsode2StopCondition::greater_equal(variable, target));
        self
    }

    pub fn with_stop_condition_le(mut self, variable: impl Into<String>, target: f64) -> Self {
        self.stop_conditions
            .push(Lsode2StopCondition::less_equal(variable, target));
        self
    }

    pub fn with_stop_condition_abs(
        mut self,
        variable: impl Into<String>,
        target: f64,
        tolerance: f64,
    ) -> Self {
        self.stop_conditions.push(Lsode2StopCondition::abs_distance(
            variable, target, tolerance,
        ));
        self
    }

    pub fn with_jac_sparsity(mut self, jac_sparsity: Option<DMatrix<f64>>) -> Self {
        self.jac_sparsity = jac_sparsity;
        self
    }

    pub fn with_vectorized(mut self, vectorized: bool) -> Self {
        self.vectorized = vectorized;
        self
    }

    /// Declares symbolic parameter names used by the IVP right-hand side.
    ///
    /// Parameterized symbolic evaluators use the same input order as the shared
    /// IVP backend: `t, params..., y...`.
    pub fn with_equation_parameters(mut self, parameters: Vec<String>) -> Self {
        self.equation_parameters = Some(parameters);
        self
    }

    /// Installs initial numeric values for declared symbolic parameters.
    pub fn with_equation_parameter_values(mut self, values: DVector<f64>) -> Self {
        self.equation_parameter_values = Some(values);
        self
    }

    /// Installs fully analytical residual/Jacobian callbacks and switches
    /// source policy to `Analytical`.
    ///
    /// This route is currently executed through LSODE2 native sparse/banded
    /// paths (`NativeSolve`), not the dense bridge path.
    pub fn with_analytical_callbacks<R, J>(mut self, residual: R, jacobian: J) -> Self
    where
        R: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        self.analytical_callbacks = Some(Lsode2AnalyticalCallbacks {
            residual: Arc::new(residual),
            jacobian: Arc::new(jacobian),
        });
        self.residual_jacobian_source = Lsode2ResidualJacobianSource::Analytical;
        self.sync_legacy_backend_from_policy();
        self
    }

    /// Overrides generated-backend AOT chunking options.
    ///
    /// This keeps chunking control at the LSODE2 config surface so callers do
    /// not need to build `SymbolicIvpGeneratedBackendConfig` manually.
    pub fn with_aot_chunking_options(mut self, options: SymbolicIvpAotOptions) -> Self {
        self.backend = self
            .backend
            .clone()
            .with_generated_backend_aot_options(options);
        self
    }

    /// Overrides sparse-Jacobian chunking for native sparse/banded generated
    /// Jacobian callbacks.
    pub fn with_aot_sparse_chunking_strategy(mut self, strategy: SparseChunkingStrategy) -> Self {
        self.backend = self
            .backend
            .clone()
            .with_generated_backend_sparse_chunking_strategy(strategy);
        self
    }

    /// Sets explicit target chunk counts for generated residual and dense
    /// Jacobian runtime plans.
    pub fn with_aot_target_chunks(
        mut self,
        residual_target_chunks: usize,
        jacobian_target_chunks: usize,
    ) -> Self {
        self.backend = self
            .backend
            .clone()
            .with_generated_backend_target_chunks(residual_target_chunks, jacobian_target_chunks);
        self
    }

    /// Chooses backend chunking from available parallelism and requested
    /// chunk-overprovisioning.
    ///
    /// `chunks_per_worker > 1` typically improves load balancing for uneven
    /// symbolic workloads by exposing more independent runtime chunks.
    pub fn with_aot_parallel_chunking(self, chunks_per_worker: usize) -> Self {
        assert!(chunks_per_worker > 0, "chunks_per_worker must be positive");
        let total_rows = self.eq_system.len().max(1);
        let options = SymbolicIvpAotOptions {
            residual_strategy: recommended_residual_chunking_for_parallelism(
                total_rows,
                chunks_per_worker,
            ),
            jacobian_strategy: recommended_dense_jacobian_chunking_for_parallelism(
                total_rows,
                chunks_per_worker,
            ),
        };
        self.with_aot_chunking_options(options)
            .with_aot_sparse_chunking_strategy(recommended_row_chunking_for_parallelism(
                total_rows,
                chunks_per_worker,
            ))
    }
}

fn execution_mode_from_generated_backend(
    backend: &SymbolicIvpGeneratedBackendConfig,
) -> Lsode2SymbolicExecutionMode {
    match backend.build_policy {
        SymbolicIvpAotBuildPolicy::UseIfAvailable => Lsode2SymbolicExecutionMode::LambdifyExpr,
        SymbolicIvpAotBuildPolicy::RequirePrebuilt
        | SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
        | SymbolicIvpAotBuildPolicy::RebuildAlways { .. } => Lsode2SymbolicExecutionMode::Aot {
            toolchain: match backend.aot_codegen_backend {
                AotCodegenBackend::C => {
                    if backend
                        .aot_c_compiler
                        .as_deref()
                        .unwrap_or("gcc")
                        .eq_ignore_ascii_case("tcc")
                    {
                        Lsode2AotToolchain::CTcc
                    } else {
                        Lsode2AotToolchain::CGcc
                    }
                }
                AotCodegenBackend::Zig => Lsode2AotToolchain::Zig,
                AotCodegenBackend::Rust => Lsode2AotToolchain::Rust,
            },
            profile: match backend.build_policy {
                SymbolicIvpAotBuildPolicy::BuildIfMissing { profile }
                | SymbolicIvpAotBuildPolicy::RebuildAlways { profile } => match profile {
                    AotBuildProfile::Debug => Lsode2AotProfile::Debug,
                    AotBuildProfile::Release => Lsode2AotProfile::Release,
                },
                SymbolicIvpAotBuildPolicy::RequirePrebuilt
                | SymbolicIvpAotBuildPolicy::UseIfAvailable => Lsode2AotProfile::Release,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_analytical_config() -> Lsode2ProblemConfig {
        Lsode2ProblemConfig::new(
            vec![Expr::parse_expression("0")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1.0e-6,
            1.0e-8,
        )
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
        )
    }

    #[test]
    fn native_linear_backend_preserves_analytical_callbacks_source() {
        let sparse = scalar_analytical_config().with_native_sparse_faer_backend();
        assert_eq!(
            sparse.residual_jacobian_source,
            Lsode2ResidualJacobianSource::Analytical
        );
        assert_eq!(
            sparse.backend.jacobian_backend,
            Lsode2JacobianBackend::AnalyticClosure
        );

        let banded = scalar_analytical_config().with_native_banded_faithful_backend();
        assert_eq!(
            banded.residual_jacobian_source,
            Lsode2ResidualJacobianSource::Analytical
        );
        assert_eq!(
            banded.backend.jacobian_backend,
            Lsode2JacobianBackend::AnalyticClosure
        );
    }
}
