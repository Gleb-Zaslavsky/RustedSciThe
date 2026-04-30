use super::algorithm::Lsode2ControllerConfig;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use std::path::PathBuf;

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
    /// Reserved for direct user callbacks in the next LSODE2 milestones.
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
    Disabled,
    PreviewBeforeBridge {
        max_step_attempts: usize,
        max_accepted_steps: usize,
    },
    ExperimentalNativeSolve {
        max_step_attempts: usize,
        max_accepted_steps: usize,
    },
}

impl Default for Lsode2NativeExecutionConfig {
    fn default() -> Self {
        Self::Disabled
    }
}

impl Lsode2NativeExecutionConfig {
    pub fn preview_before_bridge(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self::PreviewBeforeBridge {
            max_step_attempts: max_step_attempts.max(1),
            max_accepted_steps: max_accepted_steps.max(1),
        }
    }

    pub fn experimental_native_solve(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self::ExperimentalNativeSolve {
            max_step_attempts: max_step_attempts.max(1),
            max_accepted_steps: max_accepted_steps.max(1),
        }
    }
}

/// Backend configuration for LSODE2.
///
/// This groups the knobs that were missing from the old LSODE prototype.
/// `generated_backend` controls the symbolic residual lifecycle for every mode.
/// In dense mode it prepares residual + dense Jacobian together. In native
/// sparse/banded modes it prepares only the residual through Lambdify/AOT while
/// Jacobians stay native sparse triplets or compact banded storage.
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

    /// Native sparse route with residual-only AOT callbacks via `C + tcc`.
    ///
    /// The Jacobian remains native sparse triplets and Newton systems are still
    /// factored by `faer` sparse LU; only the residual evaluator is compiled.
    pub fn native_sparse_faer_aot_c_tcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_sparse_faer_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Native sparse route with residual-only AOT callbacks via `C + gcc`.
    pub fn native_sparse_faer_aot_c_gcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_sparse_faer_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        )
    }

    /// Native sparse route with residual-only AOT callbacks via Zig.
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

    /// Native banded route with residual-only AOT callbacks via `C + tcc`.
    ///
    /// This keeps compact-banded symbolic Jacobians and faithful LAPACK-style
    /// banded LU for Newton systems. Only the residual evaluator is compiled.
    pub fn native_banded_faithful_aot_c_tcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_banded_faithful_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Native banded route with residual-only AOT callbacks via `C + gcc`.
    pub fn native_banded_faithful_aot_c_gcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::native_banded_faithful_with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        )
    }

    /// Native banded route with residual-only AOT callbacks via Zig.
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

    /// Applies one high-level dense generated-backend lifecycle mode.
    pub fn with_dense_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        let mut config = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
        config.resolver = self.generated_backend.resolver.clone();
        config.aot_options = self.generated_backend.aot_options;
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
    /// Native execution modes (`preview_before_bridge` and
    /// `experimental_native_solve`) are allowed to run Adams-like stepping so
    /// we can mirror LSODE control-plane behavior while the native path is
    /// being hardened.
    pub fn with_adams_only_controller(self) -> Self {
        self.with_controller(Lsode2ControllerConfig::adams_only())
    }

    /// Declares intent to use LSODE-like automatic Adams/BDF switching.
    ///
    /// The current milestone still executes through the tested BDF engine and
    /// reports that fallback explicitly in [`crate::numerical::LSODE2::Lsode2SolveSummary`].
    pub fn with_automatic_adams_bdf_controller(self) -> Self {
        self.with_controller(Lsode2ControllerConfig::automatic_adams_bdf())
    }

    pub fn with_backend(mut self, backend: Lsode2BackendConfig) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_native_execution(mut self, native_execution: Lsode2NativeExecutionConfig) -> Self {
        self.native_execution = native_execution;
        self
    }

    pub fn with_native_preview_before_bridge(
        self,
        max_step_attempts: usize,
        max_accepted_steps: usize,
    ) -> Self {
        self.with_native_execution(Lsode2NativeExecutionConfig::preview_before_bridge(
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
}
