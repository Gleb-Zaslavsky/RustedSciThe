//! Current BVP symbolic/AOT dataflow.
//!
//! This module is the boundary between symbolic BVP construction and
//! solver-facing execution. It now supports both the long-standing lambdify
//! path and the newer sparse AOT lifecycle.
//!
//! Current sparse mainline:
//! 1. `Jacobian` receives symbolic residual equations, unknown names,
//!    optional parameters, discretization settings, and boundary conditions.
//! 2. `generate_BVP_with_params(...)` discretizes the system and fills the
//!    symbolic residual cache plus dense/sparse Jacobian caches.
//! 3. `prepare_sparse_aot_problem(...)` turns the sparse symbolic cache into an
//!    owned `BvpPreparedSparseAotProblem`.
//! 4. Backend selection combines requested policy (`Numeric`, `Lambdify`,
//!    `Aot`) with resolver state and yields a `BvpSparseExecutionPlan`.
//! 5. The solver then consumes either:
//!    - `BvpSparseSolverProvider<'_>` / `BvpSparseSolverBundle` for sparse
//!      lambdify or linked compiled AOT,
//!    - `BvpLegacySolverBundle` for the remaining compatibility paths.
//!
//! Compiled AOT scenario:
//! - `BvpPreparedSparseAotProblem` converts into a shared
//!   `PreparedSparseProblem`.
//! - lifecycle modules can emit/build/register/resolve a generated crate.
//! - once a linked runtime backend is registered in-process, the exact same
//!   solver bundle shape becomes callable through compiled callbacks.
//!
//! The production-oriented branch in this file is the sparse faer backend
//! (`SparseColMat` / `Col`). Dense and older sparse matrix backends are kept
//! for compatibility, validation, and gradual migration of the wider BVP
//! stack.
//!
//! Legacy notes for the original BVP symbolic module layout.
//!
//! ## Main Purpose
//! This module provides high-performance symbolic computation and automatic differentiation
//! capabilities specifically designed for solving Boundary Value Problems (BVPs) in differential
//! equations. It bridges symbolic mathematics with numerical computation by:
//! - Converting continuous ODEs into discretized algebraic systems
//! - Computing analytical Jacobians for Newton-Raphson and other iterative methods
//! - Providing multiple sparse and dense matrix backends for optimal performance
//! - Supporting parallel computation throughout the symbolic-to-numerical pipeline
//!
//! ## Main Structures and Methods
//!
//! ### `Jacobian` Struct
//! The core structure that manages the entire symbolic-to-numerical transformation pipeline:
//! - **Fields**:
//!   - `vector_of_functions`: Symbolic expressions representing the discretized BVP system
//!   - `symbolic_jacobian`: 2D matrix of symbolic partial derivatives
//!   - `jac_function`: Compiled numerical Jacobian function (trait object)
//!   - `residiual_function`: Compiled numerical residual function (trait object)
//!   - `bandwidth`: Optional sparse matrix bandwidth for banded systems
//!   - `bounds`/`rel_tolerance_vec`: Constraint and tolerance vectors for variables
//!
//! ### Key Methods
//!
//! #### Discretization
//! - `discretization_system_BVP_par()`: Converts continuous BVP to discrete algebraic system
//!   with parallel boundary condition processing and variable tracking
//!
//! #### Jacobian Computation
//! - `calc_jacobian_parallel_smart()`: Parallel symbolic differentiation with sparsity optimization
//! - `calc_jacobian_parallel()`: Standard parallel symbolic differentiation
//!
//! #### Function Compilation (Multiple Backends)
//! - **Dense (nalgebra)**: `lambdify_jacobian_DMatrix_par()`, `lambdify_residual_DVector()`
//! - **Sparse (faer)**: `lambdify_jacobian_SparseColMat_parallel2()`, `lambdify_residual_Col_parallel2()`
//! - **Sparse (sprs)**: `lambdify_jacobian_CsMat()`, `lambdify_residual_CsVec()`
//! - **Sparse (nalgebra)**: `lambdify_jacobian_CsMatrix()`, residual functions
//!
//! #### Utility Functions
//! - `find_bandwidths()`: Automatic sparse matrix bandwidth detection
//! - `process_bounds_and_tolerances()`: Efficient constraint processing with string optimization
//!
//! ## Interesting Tips and Non-Obvious Code Features
//!
//! ### Performance Optimizations
//! 1. **Parallel Outer Loop Pattern**: Functions like `lambdify_jacobian_SparseColMat_parallel2()`
//!    use `(0..n).into_par_iter().flat_map()` instead of nested loops for better load balancing
//!
//! 2. **Pre-compilation Strategy**: Functions are compiled once during setup and stored as
//!    `Box<dyn Fn + Send + Sync>` to avoid repeated lambdification during evaluation
//!
//! 3. **Smart Sparsity Detection**: `calc_jacobian_parallel_smart()` only computes derivatives
//!    for variables actually present in each equation, dramatically reducing computation
//!
//! 4. **String Processing Optimization**: `process_bounds_and_tolerances()` uses `rfind('_')`
//!    instead of regex for 10-100x faster variable name processing
//!
//! 5. **Boundary Condition Caching**: `discretization_system_BVP_par()` pre-computes HashSets
//!    and HashMaps for O(1) boundary condition lookups instead of O(n²) nested loops
//!
//! ### Thread Safety Patterns
//! 1. **Mutex-Protected Collections**: Parallel triplet collection uses `Mutex<Vec<Triplet>>`
//!    for thread-safe sparse matrix assembly
//!
//! 2. **Lifetime Management**: Functions convert `Vec<&str>` to `Vec<String>` early to avoid
//!    lifetime issues in `'static` closures: `variable_str.iter().map(|s| s.to_string()).collect()`
//!
//! 3. **Send + Sync Bounds**: Thread-safe lambdification uses `Box<dyn Fn + Send + Sync>`
//!    instead of regular `Box<dyn Fn>` for parallel evaluation
//!
//! ### Memory Management Tricks
//! 1. **Pre-allocation**: Vectors are pre-allocated with known capacity to avoid reallocations:
//!    `Vec::with_capacity(len)`
//!
//! 2. **Zero-Copy Operations**: Uses `ColRef::from_slice().to_owned()` for efficient faer
//!    matrix construction without intermediate copies
//!
//! 3. **Selective Cloning**: Only clones `Expr` objects when necessary due to their complex
//!    recursive structure - uses references and moves where possible
//!
//! ### Discretization Schemes
//! - **Forward Euler**: `"forward"` - First-order explicit scheme
//! - **Trapezoidal**: `"trapezoid"` - Second-order implicit scheme with better stability
//!
//! ### Matrix Backend Strategy
//! The module supports 4 different matrix backends, each optimized for different scenarios:
//! - **Dense (nalgebra)**: Best for small, dense systems
//! - **Sparse (faer)**: Modern, high-performance sparse matrices with excellent parallel support
//! - **Sparse (sprs)**: Mature Rust sparse matrix library
//! - **Sparse (nalgebra)**: Integrated with nalgebra ecosystem
//!
//! This multi-backend approach allows users to choose the optimal performance characteristics
//! for their specific problem size and sparsity pattern.

#![allow(non_camel_case_types)]
use crate::global::THRESHOLD as T;
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, JacEnum, MatrixType, VectorType, Vectors_type_casting, convert_to_fun,
    convert_to_jac,
};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use crate::symbolic::codegen_aot_resolution::{AotResolver, ResolvedAotArtifact};
use crate::symbolic::codegen_aot_runtime_link::resolve_linked_sparse_backend;
use crate::symbolic::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind, select_backend,
};
use crate::symbolic::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen_orchestrator::{ParallelExecutorConfig, ParallelFallbackPolicy};
use crate::symbolic::codegen_provider_api::{BackendKind, MatrixBackend, PreparedSparseProblem};
use crate::symbolic::codegen_runtime_api::{
    ResidualChunkPlan, ResidualChunkingStrategy, ResidualRuntimePlan, SparseJacobianRuntimePlan,
    SparseJacobianStructure, SparseJacobianValuesChunkPlan,
    recommended_residual_chunking_for_parallelism, recommended_row_chunking_for_parallelism,
};
use crate::symbolic::codegen_tasks::{
    CodegenOutputLayout, CodegenTaskKind, CodegenTaskPlan, PlannedOutput, SparseChunkingStrategy,
    SparseExprEntry,
};
use crate::symbolic::symbolic_engine::Expr;
//use crate::symbolic::symbolic_traits::SymbolicType;
use faer::col::Col;
use faer::col::ColRef;

use faer::sparse::{SparseColMat, Triplet};
use log::{error, info};
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector};

use rayon::prelude::*;

use sprs::{CsMat, CsVec};
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};

fn normalize_timer_percent(value: &mut f64, total: f64) {
    if total <= f64::EPSILON {
        *value = 0.0;
    } else {
        *value /= total / 100.0;
    }
}

/// High-level backend family used by the BVP symbolic pipeline.
///
/// This enum intentionally separates "how the numerical evaluator appears"
/// from the lower-level matrix storage choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpBackendKind {
    Numeric,
    Lambdify,
    Aot,
}

/// Matrix/vector backend used by the BVP Jacobian/residual pipeline.
///
/// `FaerSparseCol` is the main production-oriented branch in this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpMatrixBackend {
    Dense,
    SprsCsMat,
    NalgebraCsMatrix,
    FaerSparseCol,
}

impl BvpMatrixBackend {
    pub const fn legacy_method(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::SprsCsMat => "Sparse_1",
            Self::NalgebraCsMatrix => "Sparse_2",
            Self::FaerSparseCol => "Sparse",
        }
    }

    pub fn from_legacy_method(method: &str) -> Option<Self> {
        match method {
            "Dense" => Some(Self::Dense),
            "Sparse_1" => Some(Self::SprsCsMat),
            "Sparse_2" => Some(Self::NalgebraCsMatrix),
            "Sparse" => Some(Self::FaerSparseCol),
            _ => None,
        }
    }

    pub const fn from_provider_matrix_backend(matrix_backend: MatrixBackend) -> Self {
        match matrix_backend {
            MatrixBackend::Dense => Self::Dense,
            MatrixBackend::SparseCol => Self::FaerSparseCol,
            MatrixBackend::CsMat => Self::SprsCsMat,
            MatrixBackend::CsMatrix => Self::NalgebraCsMatrix,
            MatrixBackend::ValuesOnly => Self::FaerSparseCol,
        }
    }
}

/// User-facing backend selection for BVP symbolic compilation.
///
/// For now only the `Lambdify` branch is executed inside this module, while
/// `Numeric` and `Aot` are reserved for the staged integration layer above it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BvpBackendConfig {
    pub backend_kind: BvpBackendKind,
    pub matrix_backend: BvpMatrixBackend,
}

impl BvpBackendConfig {
    pub const fn new(backend_kind: BvpBackendKind, matrix_backend: BvpMatrixBackend) -> Self {
        Self {
            backend_kind,
            matrix_backend,
        }
    }

    pub const fn lambdify(matrix_backend: BvpMatrixBackend) -> Self {
        Self::new(BvpBackendKind::Lambdify, matrix_backend)
    }
}

impl Default for BvpBackendConfig {
    fn default() -> Self {
        Self::lambdify(BvpMatrixBackend::FaerSparseCol)
    }
}
/// Core structure for BVP symbolic-to-numerical transformation pipeline.
///
/// This struct represents a Jacobian for BVPs, which is a matrix of partial derivatives of a vector function.
/// It contains the symbolic and numerical representations of the jacobian, as well as the functions used to evaluate it.
/// Manages the complete workflow from symbolic BVP discretization to high-performance numerical evaluation.
///
/// # Performance Features
/// - Supports multiple matrix backends (dense/sparse) for optimal performance
/// - Parallel symbolic differentiation and function compilation
/// - Smart sparsity detection and bandwidth optimization
/// - Thread-safe evaluation with Send+Sync closures
//#[derive(Clone)]
pub struct Jacobian {
    /// Vector of symbolic functions/expressions representing the discretized BVP system.
    /// After discretization, contains the residual equations F(y) = 0 that need to be solved.
    pub vector_of_functions: Vec<Expr>,

    /// Vector of lambdified functions (symbolic functions converted to rust functions).
    /// Legacy field - modern implementations use trait objects in `jac_function` and `residiual_function`.
    pub lambdified_functions: Vec<Box<dyn Fn(Vec<f64>) -> f64>>,

    /// String identifier for the matrix backend method being used.
    /// Values: "Dense", "Sparse", "Sparse_1" (sprs), "Sparse_2" (nalgebra), "Sparse_3" (faer)
    pub method: String,

    /// Explicit backend configuration used by the modernized BVP entrypoints.
    ///
    /// The legacy `method` string is still kept for compatibility with the
    /// wider BVP stack, but new code should prefer this field.
    pub backend_config: BvpBackendConfig,

    /// Vector of symbolic variables representing the unknowns in the BVP system.
    /// Contains Expr::Var objects for each discretized variable (e.g., y_0, y_1, ..., y_n)
    pub vector_of_variables: Vec<Expr>,

    /// String representations of the variable names for efficient lookups.
    /// Used during lambdification and variable processing (e.g., ["y_0", "y_1", "z_0", "z_1"])
    pub variable_string: Vec<String>,

    /// Optional symbolic parameter names that affect evaluation but are not
    /// Newton unknowns of the BVP system.
    pub parameters_string: Vec<String>,

    /// Current numeric values for `parameters_string`.
    ///
    /// At runtime the residual/Jacobian evaluators flatten arguments in the
    /// order `[params..., unknowns...]`, while the outer Newton solver still
    /// passes only the unknown vector.
    pub parameter_values: Option<Vec<f64>>,

    /// 2D matrix of symbolic partial derivatives ∂F_i/∂x_j.
    /// Core of the analytical Jacobian - computed once symbolically, then compiled to numerical functions
    pub symbolic_jacobian: Vec<Vec<Expr>>,

    /// Sparse symbolic Jacobian cache containing only explicit non-zero entries.
    ///
    /// This cache is maintained alongside the legacy dense matrix so the BVP
    /// path can migrate gradually toward sparse-first symbolic storage.
    pub symbolic_jacobian_sparse: Vec<(usize, usize, Expr)>,

    /// Optional single Jacobian element evaluator function.
    /// Legacy field for element-wise Jacobian evaluation - modern code uses `jac_function`
    pub lambdified_jac_element: Option<Box<dyn Fn(f64, usize, usize) -> f64>>,

    /// Compiled numerical Jacobian function as trait object.
    /// Supports multiple backends through JacEnum variants (Dense, Sparse_1, Sparse_2, Sparse_3)
    pub jac_function: Option<Box<dyn Jac>>,

    /// Compiled numerical residual function as trait object.
    /// Evaluates F(x,y) for the discretized BVP system through FunEnum variants
    pub residiual_function: Box<dyn Fun>,

    /// Optional variable bounds as (min, max) pairs for each discretized variable.
    /// Used by constrained solvers to enforce physical constraints during Newton iteration
    pub bounds: Option<Vec<(f64, f64)>>,

    /// Optional relative tolerance vector for adaptive error control.
    /// Per-variable tolerances for adaptive mesh refinement and convergence criteria
    pub rel_tolerance_vec: Option<Vec<f64>>,

    /// Optional sparse matrix bandwidth (kl, ku) for banded Jacobians.
    /// kl = number of subdiagonals, ku = number of superdiagonals. Enables banded matrix optimizations
    pub bandwidth: Option<(usize, usize)>,

    /// Variable tracking for smart sparsity detection.
    /// variables_for_all_disrete[i] contains variables actually used in equation i,
    /// enabling zero-derivative skipping in calc_jacobian_parallel_smart()
    pub variables_for_all_disrete: Vec<Vec<String>>,

    /// Boundary condition positions and values for full solution reconstruction.
    /// Contains (position_in_full_vector, boundary_value) pairs to insert BC values
    /// back into the solution vector of unknowns to create the complete solution.
    pub BC_pos_n_values: Vec<(usize, usize, f64)>,
}

/// Owned sparse AOT-ready BVP problem produced by the legacy BVP symbolic
/// builder.
///
/// This bridge object keeps symbolic residuals, variable names, parameter
/// names, and sparse Jacobian entries in owned form so it can later expose a
/// borrow-based [`PreparedSparseProblem`] without leaking temporary vectors.
#[derive(Debug, Clone)]
pub struct BvpPreparedSparseAotProblem {
    pub residual_fn_name: String,
    pub jacobian_fn_name: String,
    pub variable_names: Vec<String>,
    pub param_names: Vec<String>,
    pub residuals: Vec<Expr>,
    pub sparse_entries: Vec<(usize, usize, Expr)>,
    pub shape: (usize, usize),
    pub residual_strategy: ResidualChunkingStrategy,
    pub jacobian_strategy: SparseChunkingStrategy,
}

/// Owned BVP-side backend selection result for the sparse main path.
///
/// This keeps the selected branch together with the owned sparse prepared
/// problem bridge, allowing the BVP module to consult lifecycle metadata
/// without leaking borrow-based `PreparedProblem` values across layers.
#[derive(Debug, Clone)]
pub struct BvpSelectedSparseBackend {
    pub prepared_problem: BvpPreparedSparseAotProblem,
    pub requested_backend: BackendKind,
    pub effective_backend: SelectedBackendKind,
    pub matrix_backend: MatrixBackend,
    pub aot_resolution: Option<ResolvedAotArtifact>,
}

impl BvpSelectedSparseBackend {
    /// Returns `true` when the selected path points to a compiled AOT artifact.
    pub fn is_compiled_aot(&self) -> bool {
        self.effective_backend == SelectedBackendKind::AotCompiled
    }
}

/// BVP-side sparse backend execution plan.
///
/// This is the first solver-adjacent use layer on top of backend selection:
/// - lambdify variants mean `jac_function` / `residiual_function` are ready,
/// - AOT variants mean lifecycle metadata is resolved and the caller can
///   continue into the compiled-backend branch,
/// - numeric remains a reserved integration point.
#[derive(Debug, Clone)]
pub enum BvpSparseExecutionPlan {
    NumericRequested(BvpSelectedSparseBackend),
    LambdifyReady(BvpSelectedSparseBackend),
    AotCompiled(BvpSelectedSparseBackend),
    AotRegisteredButNotBuilt(BvpSelectedSparseBackend),
    AotMissing(BvpSelectedSparseBackend),
}

impl BvpSparseExecutionPlan {
    pub fn selected(&self) -> &BvpSelectedSparseBackend {
        match self {
            Self::NumericRequested(selected)
            | Self::LambdifyReady(selected)
            | Self::AotCompiled(selected)
            | Self::AotRegisteredButNotBuilt(selected)
            | Self::AotMissing(selected) => selected,
        }
    }
}

/// Transitional BVP-side sparse provider bridge for the main faer-oriented path.
///
/// This object is intentionally thin:
/// - for `LambdifyReady`, it can evaluate residuals and sparse Jacobian values
///   through the already prepared legacy callbacks;
/// - for `Aot*`, it exposes the resolved metadata and sparse structure needed
///   by the next lifecycle stage that will attach compiled callbacks.
///
/// It lets the solver-facing integration use one object shape before the final
/// in-process compiled AOT runtime hook is introduced.
pub struct BvpSparseSolverProvider<'a> {
    jacobian: &'a mut Jacobian,
    execution: BvpSparseExecutionPlan,
    structure: SparseJacobianStructure,
}

impl<'a> BvpSparseSolverProvider<'a> {
    /// Returns the already-selected sparse execution plan.
    ///
    /// Outer integration layers use this to inspect whether the current path is
    /// lambdify-ready, compiled AOT, or still missing a built artifact.
    pub fn execution_plan(&self) -> &BvpSparseExecutionPlan {
        &self.execution
    }

    /// Effective backend after policy + resolver selection.
    pub fn effective_backend(&self) -> SelectedBackendKind {
        self.execution.selected().effective_backend
    }

    /// Backend family originally requested by the upper configuration layer.
    pub fn requested_backend(&self) -> BackendKind {
        self.execution.selected().requested_backend
    }

    /// Returns `true` when residual/Jacobian callbacks can be executed right now.
    ///
    /// That is true for:
    /// - `LambdifyReady`
    /// - `AotCompiled` with a registered linked runtime backend
    pub fn is_runtime_callable(&self) -> bool {
        matches!(self.execution, BvpSparseExecutionPlan::LambdifyReady(_))
            || matches!(self.execution, BvpSparseExecutionPlan::AotCompiled(_))
                && resolve_linked_sparse_backend(
                    &self.execution.selected().prepared_problem.problem_key(),
                )
                .is_some()
    }

    /// Returns resolved lifecycle metadata for compiled AOT artifacts, when present.
    pub fn resolved_aot_artifact(&self) -> Option<&ResolvedAotArtifact> {
        self.execution.selected().aot_resolution.as_ref()
    }

    /// Number of residual outputs in the current sparse BVP system.
    pub fn residual_len(&self) -> usize {
        self.execution.selected().prepared_problem.shape.0
    }

    /// Sparse Jacobian shape `(rows, cols)` for the current prepared problem.
    pub fn jacobian_shape(&self) -> (usize, usize) {
        self.execution.selected().prepared_problem.shape
    }

    /// Sparse structural pattern shared by lambdify and linked AOT branches.
    pub fn jacobian_structure(&self) -> &SparseJacobianStructure {
        &self.structure
    }

    /// Casts the raw dense solver vector into the matrix/vector backend
    /// expected by the legacy callback layer.
    fn typed_variables_from_args(&self, args: &[f64]) -> Box<dyn VectorType> {
        let dense = DVector::from_column_slice(args);
        let backend = BvpMatrixBackend::from_provider_matrix_backend(
            self.execution.selected().matrix_backend,
        );
        Vectors_type_casting(&dense, backend.legacy_method().to_string())
    }

    /// Evaluates residual outputs into the caller-provided buffer.
    ///
    /// The provider keeps one solver-facing contract while internally routing
    /// either to legacy lambdified callbacks or to linked compiled AOT
    /// callbacks.
    pub fn residual_into(&self, args: &[f64], out: &mut [f64]) {
        assert!(
            self.is_runtime_callable(),
            "BvpSparseSolverProvider residual_into is only callable for runtime-ready sparse plans"
        );
        assert_eq!(
            out.len(),
            self.residual_len(),
            "residual output length must match residual_len"
        );
        match &self.execution {
            BvpSparseExecutionPlan::LambdifyReady(_) => {
                let typed = self.typed_variables_from_args(args);
                let residual = self.jacobian.residiual_function.call(1.0, &*typed);
                let dense = residual.to_DVectorType();
                out.copy_from_slice(dense.as_slice());
            }
            BvpSparseExecutionPlan::AotCompiled(selected) => {
                let linked =
                    resolve_linked_sparse_backend(&selected.prepared_problem.problem_key())
                        .expect("AotCompiled provider path must resolve a linked backend");
                let mut flat_args = Vec::with_capacity(
                    self.jacobian
                        .parameter_values
                        .as_ref()
                        .map_or(0, |params| params.len())
                        + args.len(),
                );
                if let Some(params) = self.jacobian.parameter_values.as_ref() {
                    flat_args.extend_from_slice(params.as_slice());
                }
                flat_args.extend_from_slice(args);
                (linked.residual_eval)(&flat_args, out);
            }
            _ => {
                unreachable!("runtime-callable provider should only expose lambdify or linked AOT")
            }
        }
    }

    /// Evaluates sparse Jacobian explicit values into the caller-provided buffer.
    ///
    /// The output order always follows `jacobian_structure()`, regardless of
    /// whether the active branch is lambdify or linked compiled AOT.
    pub fn jacobian_values_into(&mut self, args: &[f64], values_out: &mut [f64]) {
        assert!(
            self.is_runtime_callable(),
            "BvpSparseSolverProvider jacobian_values_into is only callable for runtime-ready sparse plans"
        );
        assert_eq!(
            values_out.len(),
            self.structure.nnz(),
            "Jacobian values output length must match sparse structure nnz"
        );
        match &mut self.execution {
            BvpSparseExecutionPlan::LambdifyReady(_) => {
                let typed = self.typed_variables_from_args(args);
                let dense = self
                    .jacobian
                    .jac_function
                    .as_mut()
                    .expect("jacobian callback must exist for LambdifyReady plan")
                    .call(1.0, &*typed)
                    .to_DMatrixType();

                for (value, (&row, &col)) in values_out.iter_mut().zip(
                    self.structure
                        .row_indices
                        .iter()
                        .zip(self.structure.col_indices.iter()),
                ) {
                    *value = dense[(row, col)];
                }
            }
            BvpSparseExecutionPlan::AotCompiled(selected) => {
                let linked =
                    resolve_linked_sparse_backend(&selected.prepared_problem.problem_key())
                        .expect("AotCompiled provider path must resolve a linked backend");
                let mut flat_args = Vec::with_capacity(
                    self.jacobian
                        .parameter_values
                        .as_ref()
                        .map_or(0, |params| params.len())
                        + args.len(),
                );
                if let Some(params) = self.jacobian.parameter_values.as_ref() {
                    flat_args.extend_from_slice(params.as_slice());
                }
                flat_args.extend_from_slice(args);
                (linked.jacobian_values_eval)(&flat_args, values_out);
            }
            _ => {
                unreachable!("runtime-callable provider should only expose lambdify or linked AOT")
            }
        }
    }
}

/// Owned solver-facing sparse BVP bundle.
///
/// This is the assembled object the outer solver infrastructure wants to see:
/// equations/settings go in, and on success it receives one bundle containing:
/// - the selected backend branch,
/// - optional residual/Jacobian callbacks ready to use,
/// - and all solver metadata that used to be read piecemeal from `Jacobian`.
///
/// The solver does not need to know whether the callbacks came from lambdify,
/// AOT, or a future numeric path. It only inspects whether runtime callbacks
/// are available and uses the exposed metadata consistently.
pub struct BvpSparseSolverBundle {
    pub execution: BvpSparseExecutionPlan,
    pub residual_function: Option<Box<dyn Fun>>,
    pub jacobian_function: Option<Box<dyn Jac>>,
    pub variable_string: Vec<String>,
    pub bounds_vec: Option<Vec<(f64, f64)>>,
    pub rel_tolerance_vec: Option<Vec<f64>>,
    pub bandwidth: Option<(usize, usize)>,
    pub bc_position_and_value: Vec<(usize, usize, f64)>,
    pub sparse_structure: SparseJacobianStructure,
}

/// Transitional owned bundle for legacy lambdify/numeric BVP callbacks.
///
/// This keeps the remaining compatibility path assembled in one object instead
/// of forcing solver handoff layers to read multiple `Jacobian` fields
/// directly. It is intentionally smaller than [`BvpSparseSolverBundle`] and
/// exists only while non-sparse solver paths are still converging on the same
/// provider model.
pub struct BvpLegacySolverBundle {
    pub residual_function: Box<dyn Fun>,
    pub jacobian_function: Option<Box<dyn Jac>>,
    pub variable_string: Vec<String>,
    pub bounds_vec: Option<Vec<(f64, f64)>>,
    pub rel_tolerance_vec: Option<Vec<f64>>,
    pub bandwidth: Option<(usize, usize)>,
    pub bc_position_and_value: Vec<(usize, usize, f64)>,
}

/// Fallible wrapper error for the staged BVP backend-selection / AOT path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvpBackendIntegrationError {
    /// The BVP pipeline panicked while building or selecting a backend.
    PipelinePanicked(String),
    /// The solver was configured with an unsupported logging level string.
    InvalidLogLevel { level: String },
    /// The solver could not create the requested log file.
    LogFileCreationFailed { path: String, message: String },
    /// A compiled AOT backend was required by the outer policy, but the
    /// selection layer could not provide a compiled artifact.
    CompiledAotRequiredButUnavailable {
        problem_key: String,
        effective_backend: SelectedBackendKind,
    },
    /// The lifecycle policy asked the solver layer to build a missing AOT
    /// artifact automatically, but that orchestration is not wired here yet.
    AutomaticAotBuildRequested { problem_key: String },
    /// The lifecycle policy asked the solver layer to rebuild an AOT artifact,
    /// but rebuild orchestration is not wired here yet.
    AutomaticAotRebuildRequested { problem_key: String },
    /// The lifecycle layer attempted to build a generated AOT crate but the
    /// nested Cargo build failed.
    AutomaticAotBuildFailed {
        problem_key: String,
        message: String,
    },
    /// A compiled AOT artifact was selected, but callable runtime callbacks
    /// were not available in the current process.
    CompiledAotRuntimeUnavailable { problem_key: String },
}

impl BvpSparseSolverBundle {
    pub fn effective_backend(&self) -> SelectedBackendKind {
        self.execution.selected().effective_backend
    }

    pub fn requested_backend(&self) -> BackendKind {
        self.execution.selected().requested_backend
    }

    pub fn is_runtime_callable(&self) -> bool {
        self.residual_function.is_some() && self.jacobian_function.is_some()
    }

    pub fn resolved_aot_artifact(&self) -> Option<&ResolvedAotArtifact> {
        self.execution.selected().aot_resolution.as_ref()
    }

    pub fn residual_len(&self) -> usize {
        self.execution.selected().prepared_problem.shape.0
    }

    pub fn jacobian_shape(&self) -> (usize, usize) {
        self.execution.selected().prepared_problem.shape
    }

    pub fn residual_call(&self, p: f64, y: &dyn VectorType) -> Option<Box<dyn VectorType>> {
        self.residual_function.as_ref().map(|fun| fun.call(p, y))
    }

    pub fn jacobian_call(&mut self, p: f64, y: &dyn VectorType) -> Option<Box<dyn MatrixType>> {
        self.jacobian_function.as_mut().map(|jac| jac.call(p, y))
    }

    pub fn into_runtime_callbacks(self) -> Option<(Box<dyn Fun>, Box<dyn Jac>)> {
        match (self.residual_function, self.jacobian_function) {
            (Some(fun), Some(jac)) => Some((fun, jac)),
            _ => None,
        }
    }

    /// Rebuilds linked compiled-AOT runtime callbacks with an optional
    /// parallel execution policy.
    ///
    /// This keeps the outer solver-facing bundle shape stable while allowing
    /// solver-level execution settings to choose between whole-callback and
    /// chunk-aware linked AOT execution.
    pub fn rebind_linked_runtime_callbacks(
        &mut self,
        parameter_values: Option<&[f64]>,
        parallel_config: Option<ParallelExecutorConfig>,
    ) -> bool {
        if !matches!(self.execution, BvpSparseExecutionPlan::AotCompiled(_)) {
            return false;
        }

        let Some((residual, jacobian)) = linked_sparse_runtime_callbacks(
            &self.execution.selected().prepared_problem,
            parameter_values,
            parallel_config,
        ) else {
            return false;
        };

        self.residual_function = Some(residual);
        self.jacobian_function = Some(jacobian);
        true
    }
}

impl Jacobian {
    /// Consumes the legacy lambdify/numeric callback fields into one assembled
    /// compatibility bundle.
    ///
    /// This is a transitional helper for non-sparse solver paths while they
    /// are still being migrated away from direct `Jacobian` field access.
    pub fn into_legacy_solver_bundle(self) -> BvpLegacySolverBundle {
        BvpLegacySolverBundle {
            residual_function: self.residiual_function,
            jacobian_function: self.jac_function,
            variable_string: self.variable_string,
            bounds_vec: self.bounds,
            rel_tolerance_vec: self.rel_tolerance_vec,
            bandwidth: self.bandwidth,
            bc_position_and_value: self.BC_pos_n_values,
        }
    }

    /// Builds a legacy lambdify/numeric BVP pipeline and returns the assembled
    /// compatibility bundle in one step.
    ///
    /// This keeps the remaining non-sparse solver handoff on the symbolic
    /// layer, so outer solver glue does not need to know which internal
    /// `Jacobian` fields carry callbacks and metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_legacy_solver_bundle_with_params(
        mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        border_conditions: HashMap<String, Vec<(usize, f64)>>,
        bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
    ) -> BvpLegacySolverBundle {
        self.generate_BVP_with_params(
            eq_system,
            values,
            arg,
            params,
            t0,
            param,
            n_steps,
            h,
            mesh,
            border_conditions,
            bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
        );
        self.into_legacy_solver_bundle()
    }
}

impl BvpPreparedSparseAotProblem {
    /// Returns the flattened symbolic input ordering used across the sparse
    /// AOT path: `[params..., variables...]`.
    fn flattened_input_names(&self) -> Vec<&str> {
        let mut names = Vec::with_capacity(
            self.param_names
                .len()
                .saturating_add(self.variable_names.len()),
        );
        names.extend(self.param_names.iter().map(|name| name.as_str()));
        names.extend(self.variable_names.iter().map(|name| name.as_str()));
        names
    }

    /// Rebuilds borrowed sparse Jacobian entries from the owned cache.
    ///
    /// This is the bridge between the owned BVP representation and the shared
    /// borrow-based codegen/runtime plan layer.
    fn borrowed_sparse_entries(&self) -> Vec<SparseExprEntry<'_>> {
        self.sparse_entries
            .iter()
            .map(|(row, col, expr)| SparseExprEntry {
                row: *row,
                col: *col,
                expr,
            })
            .collect()
    }

    /// Builds the residual runtime plan used by code generation, manifests,
    /// and linked runtime rebinding.
    fn residual_runtime_plan(&self) -> ResidualRuntimePlan<'_> {
        let chunk_size = match self.residual_strategy {
            ResidualChunkingStrategy::Whole => self.residuals.len().max(1),
            ResidualChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                self.residuals.len().max(1).div_ceil(target_chunks).max(1)
            }
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk,
            } => {
                assert!(
                    max_outputs_per_chunk > 0,
                    "max_outputs_per_chunk must be positive"
                );
                max_outputs_per_chunk
            }
        };

        let input_names = self.flattened_input_names();
        let chunks = self
            .residuals
            .chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, residuals)| {
                let output_offset = chunk_index * chunk_size;
                let fn_name = if chunk_index == 0 && residuals.len() == self.residuals.len() {
                    self.residual_fn_name.clone()
                } else {
                    format!("{}_chunk_{}", self.residual_fn_name, chunk_index)
                };

                let outputs = residuals
                    .iter()
                    .map(|expr| PlannedOutput {
                        expr,
                        coordinate: None,
                    })
                    .collect();

                ResidualChunkPlan {
                    fn_name: fn_name.clone(),
                    output_offset,
                    residuals,
                    plan: CodegenTaskPlan {
                        fn_name: std::borrow::Cow::Owned(fn_name),
                        kind: CodegenTaskKind::Residual,
                        input_names: input_names.clone(),
                        outputs,
                        layout: CodegenOutputLayout::Vector {
                            len: residuals.len(),
                        },
                    },
                }
            })
            .collect();

        ResidualRuntimePlan {
            fn_name: self.residual_fn_name.as_str(),
            output_len: self.residuals.len(),
            input_names,
            chunks,
        }
    }

    /// Builds the sparse Jacobian runtime plan used by code generation,
    /// manifests, and linked runtime rebinding.
    fn sparse_runtime_plan(&self) -> SparseJacobianRuntimePlan<'_> {
        let all_entries = self.borrowed_sparse_entries();
        let input_names = self.flattened_input_names();

        let chunked_entries: Vec<(usize, Vec<SparseExprEntry<'_>>)> = match self.jacobian_strategy {
            SparseChunkingStrategy::Whole => vec![(0usize, all_entries.clone())],
            SparseChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                let rows_per_chunk = self.shape.0.max(1).div_ceil(target_chunks).max(1);
                chunk_sparse_entries_by_rows(&all_entries, rows_per_chunk)
            }
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk,
            } => {
                assert!(
                    max_entries_per_chunk > 0,
                    "max_entries_per_chunk must be positive"
                );
                all_entries
                    .chunks(max_entries_per_chunk)
                    .enumerate()
                    .map(|(chunk_index, entries)| {
                        (chunk_index * max_entries_per_chunk, entries.to_vec())
                    })
                    .collect()
            }
            SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
                assert!(rows_per_chunk > 0, "rows_per_chunk must be positive");
                chunk_sparse_entries_by_rows(&all_entries, rows_per_chunk)
            }
        };

        let chunks = chunked_entries
            .into_iter()
            .enumerate()
            .map(|(chunk_index, (entry_offset, entries))| {
                let fn_name = if chunk_index == 0 && entries.len() == all_entries.len() {
                    self.jacobian_fn_name.clone()
                } else {
                    format!("{}_chunk_{}", self.jacobian_fn_name, chunk_index)
                };
                let outputs = entries
                    .iter()
                    .map(|entry| PlannedOutput {
                        expr: entry.expr,
                        coordinate: Some((entry.row, entry.col)),
                    })
                    .collect();

                SparseJacobianValuesChunkPlan {
                    fn_name: fn_name.clone(),
                    value_offset: entry_offset,
                    entries: entries.clone(),
                    plan: CodegenTaskPlan {
                        fn_name: std::borrow::Cow::Owned(fn_name),
                        kind: CodegenTaskKind::SparseJacobianValues,
                        input_names: input_names.clone(),
                        outputs,
                        layout: CodegenOutputLayout::SparseValues {
                            rows: self.shape.0,
                            cols: self.shape.1,
                            nnz: entries.len(),
                        },
                    },
                }
            })
            .collect();

        SparseJacobianRuntimePlan {
            fn_name: self.jacobian_fn_name.as_str(),
            input_names,
            structure: SparseJacobianStructure {
                rows: self.shape.0,
                cols: self.shape.1,
                row_indices: all_entries.iter().map(|entry| entry.row).collect(),
                col_indices: all_entries.iter().map(|entry| entry.col).collect(),
            },
            chunks,
        }
    }

    /// Converts the owned BVP bridge object into the shared sparse prepared
    /// problem shape used by provider/build/registry layers.
    pub fn as_prepared_problem(&self) -> PreparedSparseProblem<'_> {
        PreparedSparseProblem::new(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            self.residual_runtime_plan(),
            self.sparse_runtime_plan(),
        )
    }

    /// Returns the manifest-derived problem key used by lifecycle and linked-backend registries.
    pub fn problem_key(&self) -> String {
        let prepared = crate::symbolic::codegen_provider_api::PreparedProblem::sparse(
            self.as_prepared_problem(),
        );
        PreparedProblemManifest::from(&prepared).problem_key()
    }
}

/// Resolves an in-process linked sparse AOT backend and wraps it into the
/// legacy solver callback traits used by the BVP runtime.
///
/// Dataflow:
/// 1. resolve a previously registered compiled backend by `problem_key`;
/// 2. prepend runtime parameter values to the Newton unknown vector;
/// 3. evaluate either whole-module callbacks or chunked callbacks;
/// 4. reassemble sparse values back into `SparseColMat` for the solver.
///
/// When no linked backend is registered yet, `None` is returned and the caller
/// remains on the lambdify/fallback branch.
fn linked_sparse_runtime_callbacks(
    prepared_problem: &BvpPreparedSparseAotProblem,
    parameter_values: Option<&[f64]>,
    parallel_config: Option<ParallelExecutorConfig>,
) -> Option<(Box<dyn Fun>, Box<dyn Jac>)> {
    let linked = resolve_linked_sparse_backend(&prepared_problem.problem_key())?;
    let parameter_values = parameter_values.unwrap_or(&[]).to_vec();
    let structure = prepared_problem
        .as_prepared_problem()
        .jacobian_structure()
        .clone();
    let shape = prepared_problem.shape;
    let residual_len = prepared_problem.shape.0;

    let residual_link = linked.clone();
    let residual_parameter_values = parameter_values.clone();
    let residual_parallel_config = parallel_config;
    let residual_fun = convert_to_fun(Box::new(move |_x: f64, vec: &dyn VectorType| {
        let dense = vec.to_DVectorType();
        let mut flat_args = Vec::with_capacity(residual_parameter_values.len() + dense.len());
        flat_args.extend_from_slice(residual_parameter_values.as_slice());
        flat_args.extend_from_slice(dense.as_slice());
        let mut out = vec![0.0; residual_len];
        eval_linked_residual_outputs(
            residual_link.clone(),
            residual_parallel_config,
            &flat_args,
            &mut out,
        );
        Box::new(ColRef::from_slice(out.as_slice()).to_owned())
    }));

    let jacobian_link = linked;
    let jacobian_parameter_values = parameter_values.clone();
    let jacobian_parallel_config = parallel_config;
    let jacobian = convert_to_jac(Box::new(move |_x: f64, vec: &dyn VectorType| {
        let dense = vec.to_DVectorType();
        let mut flat_args = Vec::with_capacity(jacobian_parameter_values.len() + dense.len());
        flat_args.extend_from_slice(jacobian_parameter_values.as_slice());
        flat_args.extend_from_slice(dense.as_slice());
        let mut values = vec![0.0; structure.nnz()];
        eval_linked_sparse_values(
            jacobian_link.clone(),
            jacobian_parallel_config,
            &flat_args,
            &mut values,
        );
        let triplets = structure
            .row_indices
            .iter()
            .zip(structure.col_indices.iter())
            .zip(values.iter())
            .map(|((&row, &col), &value)| Triplet::new(row, col, value))
            .collect::<Vec<_>>();
        let matrix = SparseColMat::try_new_from_triplets(shape.0, shape.1, triplets.as_slice())
            .expect("linked sparse backend must return valid sparse values");
        Box::new(matrix)
    }));

    Some((residual_fun, jacobian))
}

const LINKED_MIN_RESIDUAL_OUTPUTS_FOR_PARALLEL: usize = 128;
const LINKED_MIN_SPARSE_VALUES_FOR_PARALLEL: usize = 256;

/// Splits a contiguous list of chunk descriptors into contiguous job ranges.
///
/// This keeps offsets monotonic inside each job, which later allows each worker
/// to fill one compact temporary buffer and copy it back into the final output.
fn linked_contiguous_job_ranges(len: usize, max_jobs: usize) -> Vec<std::ops::Range<usize>> {
    if len == 0 {
        return Vec::new();
    }
    let target_jobs = len.min(max_jobs.max(1));
    let items_per_job = len.div_ceil(target_jobs);
    let mut ranges = Vec::with_capacity(target_jobs);
    let mut start = 0usize;
    while start < len {
        let end = (start + items_per_job).min(len);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// Decides whether the linked compiled runtime should fan out work in parallel.
///
/// The decision depends on both the configured fallback policy and the actual
/// workload size so that tiny problems still stay on the cheaper sequential
/// callback path.
fn linked_should_parallelize(
    config: ParallelExecutorConfig,
    workload: usize,
    chunk_count: usize,
    default_threshold: usize,
    sparse: bool,
) -> bool {
    if chunk_count <= 1 {
        return false;
    }
    match config.fallback_policy {
        ParallelFallbackPolicy::Auto => workload >= default_threshold,
        ParallelFallbackPolicy::Never => true,
        ParallelFallbackPolicy::Thresholds {
            min_residual_outputs,
            min_sparse_values,
        } => {
            if sparse {
                workload >= min_sparse_values
            } else {
                workload >= min_residual_outputs
            }
        }
    }
}

/// Returns the effective number of parallel jobs allowed for linked compiled
/// residual or sparse-value evaluation.
fn linked_max_jobs(config: ParallelExecutorConfig, sparse: bool) -> usize {
    let workers = rayon::current_num_threads().max(1);
    let worker_jobs = workers.saturating_mul(config.jobs_per_worker.max(1));
    if sparse {
        config.max_sparse_jobs.unwrap_or(worker_jobs).max(1)
    } else {
        config.max_residual_jobs.unwrap_or(worker_jobs).max(1)
    }
}

/// Evaluates compiled residual outputs, optionally by dispatching contiguous
/// ranges of residual chunks in parallel.
///
/// Sequential and parallel branches must produce identical values; parallelism
/// only changes scheduling and temporary buffering strategy.
fn eval_linked_residual_outputs(
    linked: crate::symbolic::codegen_aot_runtime_link::LinkedSparseAotBackend,
    parallel_config: Option<ParallelExecutorConfig>,
    flat_args: &[f64],
    out: &mut [f64],
) {
    let Some(config) = parallel_config else {
        (linked.residual_eval)(flat_args, out);
        return;
    };
    if linked.residual_chunks.is_empty()
        || !linked_should_parallelize(
            config,
            linked.residual_len,
            linked.residual_chunks.len(),
            LINKED_MIN_RESIDUAL_OUTPUTS_FOR_PARALLEL,
            false,
        )
    {
        (linked.residual_eval)(flat_args, out);
        return;
    }

    let job_ranges =
        linked_contiguous_job_ranges(linked.residual_chunks.len(), linked_max_jobs(config, false));
    if job_ranges.len() <= 1 {
        (linked.residual_eval)(flat_args, out);
        return;
    }

    let results = std::sync::Mutex::new(Vec::<(usize, Vec<f64>)>::with_capacity(job_ranges.len()));
    rayon::scope(|scope| {
        for chunk_range in job_ranges {
            let chunks = linked.residual_chunks[chunk_range].to_vec();
            let results = &results;
            scope.spawn(move |_| {
                let start = chunks.first().map(|chunk| chunk.output_offset).unwrap_or(0);
                let end = chunks
                    .last()
                    .map(|chunk| chunk.output_offset + chunk.output_len)
                    .unwrap_or(start);
                let mut job_out = vec![0.0; end.saturating_sub(start)];
                for chunk in chunks {
                    let local_start = chunk.output_offset - start;
                    let local_end = local_start + chunk.output_len;
                    (chunk.eval)(flat_args, &mut job_out[local_start..local_end]);
                }
                results
                    .lock()
                    .expect("linked residual parallel results lock poisoned")
                    .push((start, job_out));
            });
        }
    });
    for (start, job_out) in results
        .into_inner()
        .expect("linked residual parallel results lock poisoned")
    {
        let end = start + job_out.len();
        out[start..end].copy_from_slice(job_out.as_slice());
    }
}

/// Evaluates compiled sparse Jacobian values, optionally by dispatching
/// contiguous ranges of Jacobian chunks in parallel.
///
/// Offsets come from the runtime chunk plan, so the values are written back in
/// exactly the same order as the symbolic sparse structure advertised to the
/// solver and lifecycle layers.
fn eval_linked_sparse_values(
    linked: crate::symbolic::codegen_aot_runtime_link::LinkedSparseAotBackend,
    parallel_config: Option<ParallelExecutorConfig>,
    flat_args: &[f64],
    values_out: &mut [f64],
) {
    let Some(config) = parallel_config else {
        (linked.jacobian_values_eval)(flat_args, values_out);
        return;
    };
    if linked.jacobian_value_chunks.is_empty()
        || !linked_should_parallelize(
            config,
            linked.nnz,
            linked.jacobian_value_chunks.len(),
            LINKED_MIN_SPARSE_VALUES_FOR_PARALLEL,
            true,
        )
    {
        (linked.jacobian_values_eval)(flat_args, values_out);
        return;
    }

    let job_ranges = linked_contiguous_job_ranges(
        linked.jacobian_value_chunks.len(),
        linked_max_jobs(config, true),
    );
    if job_ranges.len() <= 1 {
        (linked.jacobian_values_eval)(flat_args, values_out);
        return;
    }

    let results = std::sync::Mutex::new(Vec::<(usize, Vec<f64>)>::with_capacity(job_ranges.len()));
    rayon::scope(|scope| {
        for chunk_range in job_ranges {
            let chunks = linked.jacobian_value_chunks[chunk_range].to_vec();
            let results = &results;
            scope.spawn(move |_| {
                let start = chunks.first().map(|chunk| chunk.value_offset).unwrap_or(0);
                let end = chunks
                    .last()
                    .map(|chunk| chunk.value_offset + chunk.value_len)
                    .unwrap_or(start);
                let mut job_out = vec![0.0; end.saturating_sub(start)];
                for chunk in chunks {
                    let local_start = chunk.value_offset - start;
                    let local_end = local_start + chunk.value_len;
                    (chunk.eval)(flat_args, &mut job_out[local_start..local_end]);
                }
                results
                    .lock()
                    .expect("linked sparse parallel results lock poisoned")
                    .push((start, job_out));
            });
        }
    });
    for (start, job_out) in results
        .into_inner()
        .expect("linked sparse parallel results lock poisoned")
    {
        let end = start + job_out.len();
        values_out[start..end].copy_from_slice(job_out.as_slice());
    }
}

/// Groups sparse symbolic entries by row buckets while preserving the original
/// value ordering offset expected by sparse runtime/codegen plans.
///
/// This helper is used when the sparse Jacobian is chunked by rows or by a
/// target chunk count that is later translated into row buckets.
fn chunk_sparse_entries_by_rows<'a>(
    entries: &[SparseExprEntry<'a>],
    rows_per_chunk: usize,
) -> Vec<(usize, Vec<SparseExprEntry<'a>>)> {
    let mut chunk_entries: Vec<Vec<SparseExprEntry<'a>>> = Vec::new();
    let mut chunk_offsets: Vec<usize> = Vec::new();

    for (entry_index, entry) in entries.iter().copied().enumerate() {
        let bucket = entry.row / rows_per_chunk;
        if bucket >= chunk_entries.len() {
            chunk_entries.resize_with(bucket + 1, Vec::new);
            chunk_offsets.resize(bucket + 1, 0);
        }
        if chunk_entries[bucket].is_empty() {
            chunk_offsets[bucket] = entry_index;
        }
        chunk_entries[bucket].push(entry);
    }

    chunk_entries
        .into_iter()
        .enumerate()
        .filter(|(_, entries)| !entries.is_empty())
        .map(|(chunk_index, entries)| (chunk_offsets[chunk_index], entries))
        .collect()
}

impl Jacobian {
    /// Runs the shared symbolic BVP preparation stage before backend-specific
    /// lambdify/AOT branching.
    ///
    /// This stage performs discretization, symbolic Jacobian construction,
    /// bandwidth discovery or enforcement, and fills the timer map used by the
    /// compatibility reporting path.
    fn prepare_symbolic_bvp_stage_with_params(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
    ) -> (
        String,
        Vec<String>,
        BvpMatrixBackend,
        HashMap<String, f64>,
        Instant,
    ) {
        let total_start = Instant::now();
        let mut timer_hash: HashMap<String, f64> = HashMap::new();
        self.set_params(params);
        let param = arg.clone();
        let matrix_backend = BvpMatrixBackend::from_legacy_method(method.as_str())
            .unwrap_or_else(|| panic!("unknown method: {}", method));
        self.set_backend_config(BvpBackendConfig::lambdify(matrix_backend));

        let begin = Instant::now();
        self.discretization_system_BVP_par(
            eq_system,
            values,
            arg,
            t0,
            n_steps,
            h,
            mesh,
            BorderConditions,
            Bounds,
            rel_tolerance,
            scheme,
        );
        timer_hash.insert(
            "discretization time".to_string(),
            begin.elapsed().as_secs_f64(),
        );
        info!("system discretized");

        let indexed_values = self.variable_string.clone();

        if let Some(bandwidth_) = bandwidth {
            let now = Instant::now();
            self.bandwidth = Some(bandwidth_);
            info!("Bandwidth provided: {:?}", self.bandwidth);
            timer_hash.insert(
                "find bandwidth time".to_string(),
                now.elapsed().as_secs_f64(),
            );

            info!("Calculating jacobian");
            let now = Instant::now();
            self.calc_jacobian_parallel_smart_optimized_with_given_bandwidth();
            let elapsed = now.elapsed();
            info!("Jacobian calculation time:");
            info!("{:?}", elapsed_time(elapsed));
            timer_hash.insert("symbolic jacobian time".to_string(), elapsed.as_secs_f64());
        } else {
            info!("Calculating jacobian");
            let now = Instant::now();
            self.calc_jacobian_parallel_smart_optimized();
            let elapsed = now.elapsed();
            info!("Jacobian calculation time:");
            info!("{:?}", elapsed_time(elapsed));
            timer_hash.insert("symbolic jacobian time".to_string(), elapsed.as_secs_f64());

            let now = Instant::now();
            self.find_bandwidths();
            info!("Bandwidth calculated:");
            info!("(kl, ku) = {:?}", self.bandwidth);
            timer_hash.insert(
                "find bandwidth time".to_string(),
                now.elapsed().as_secs_f64(),
            );
        }

        (
            param,
            indexed_values,
            matrix_backend,
            timer_hash,
            total_start,
        )
    }

    /// Finalizes the legacy timing table for `generate_BVP*` style entrypoints.
    ///
    /// The modern lifecycle path logs through structured stages, while this
    /// helper keeps the historical human-readable timing summary stable for the
    /// older BVP APIs and tests that still print it.
    fn finalize_generate_bvp_timer_table(
        &self,
        mut timer_hash: HashMap<String, f64>,
        total_end: f64,
    ) {
        for key in [
            "discretization time",
            "symbolic jacobian time",
            "jacobian lambdify time",
            "residual functions lambdify time",
            "find bandwidth time",
            "sparse AOT preparation time",
            "backend selection time",
            "runtime binding time",
        ] {
            if let Some(value) = timer_hash.get_mut(key) {
                normalize_timer_percent(value, total_end);
            }
        }
        timer_hash.insert("total time, sec".to_string(), total_end);

        let mut table = Builder::from(timer_hash.clone()).build();
        table.with(Style::modern_rounded());
        println!("{}", table.to_string());
        info!(
            "\n \n ____________END OF GENERATE BVP ________________________________________________________________"
        );
    }

    /// Creates a new Jacobian instance with default values.
    ///
    /// Initializes all fields to empty/default states and sets up a dummy residual function
    /// that returns the input vector unchanged. This serves as a safe default until the
    /// actual BVP system is discretized and compiled.
    ///
    /// # Returns
    /// A new Jacobian instance ready for BVP discretization and symbolic computation.
    pub fn new() -> Self {
        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        Self {
            vector_of_functions: Vec::new(),
            lambdified_functions: Vec::new(),
            method: String::new(),
            backend_config: BvpBackendConfig::default(),
            vector_of_variables: Vec::new(),
            variable_string: Vec::new(),
            parameters_string: Vec::new(),
            parameter_values: None,
            symbolic_jacobian: Vec::new(),
            symbolic_jacobian_sparse: Vec::new(),
            lambdified_jac_element: None,
            jac_function: None,
            residiual_function: boxed_fun,
            bounds: None,
            rel_tolerance_vec: None,
            bandwidth: None,
            variables_for_all_disrete: Vec::new(),
            BC_pos_n_values: Vec::new(),
        }
    }

    /// Initializes the Jacobian with a vector of symbolic functions and variable names.
    ///
    /// This method sets up the basic symbolic representation of the system before discretization.
    /// It converts string variable names to symbolic Expr::Var objects for internal processing.
    ///
    /// # Arguments
    /// * `vector_of_functions` - Vector of symbolic expressions representing the system equations
    /// * `variable_string` - Vector of variable names as strings (e.g., ["y", "z"])
    ///
    /// # Example
    /// ```ignore
    /// let mut jac = Jacobian::new();
    /// let funcs = vec![Expr::parse_expression("y + z"), Expr::parse_expression("y - z")];
    /// let vars = vec!["y".to_string(), "z".to_string()];
    /// jac.from_vectors(funcs, vars);
    /// ```
    pub fn from_vectors(&mut self, vector_of_functions: Vec<Expr>, variable_string: Vec<String>) {
        self.vector_of_functions = vector_of_functions;
        self.variable_string = variable_string.clone();
        self.vector_of_variables =
            Expr::parse_vector_expression(variable_string.iter().map(|s| s.as_str()).collect());
        println!(" {:?}", self.vector_of_functions);
        println!(" {:?}", self.vector_of_variables);
    }

    /// Sets symbolic parameter names used during residual/Jacobian evaluation.
    pub fn set_params(&mut self, params: Option<&[&str]>) {
        self.parameters_string = params
            .map(|params| params.iter().map(|name| (*name).to_string()).collect())
            .unwrap_or_default();
    }

    /// Sets current numeric parameter values.
    pub fn set_param_values(&mut self, values: Option<Vec<f64>>) {
        if let Some(ref values) = values {
            assert_eq!(
                values.len(),
                self.parameters_string.len(),
                "parameter_values length must match parameters_string length"
            );
        }
        self.parameter_values = values;
    }

    /// Returns borrowed sparse symbolic Jacobian entries from the internal
    /// owned sparse cache.
    pub fn symbolic_jacobian_sparse_entries(
        &self,
    ) -> Vec<crate::symbolic::codegen_tasks::SparseExprEntry<'_>> {
        self.symbolic_jacobian_sparse
            .iter()
            .map(
                |(row, col, expr)| crate::symbolic::codegen_tasks::SparseExprEntry {
                    row: *row,
                    col: *col,
                    expr,
                },
            )
            .collect()
    }

    /// Returns owned sparse symbolic Jacobian entries.
    ///
    /// This is primarily useful for AOT/performance scaffolding that needs to
    /// hand sparse expressions to downstream planning code without keeping an
    /// immutable borrow of `self` alive across later mutable operations.
    pub fn symbolic_jacobian_sparse_entries_owned(&self) -> Vec<(usize, usize, Expr)> {
        self.symbolic_jacobian_sparse.clone()
    }

    /// Builds a sparse AOT-ready prepared problem from the current BVP state.
    ///
    /// This is a thin bridge from the legacy BVP symbolic builder to the newer
    /// AOT lifecycle layers. It does not change how the current numeric or
    /// lambdify branches execute; it only packages the already-built symbolic
    /// residual and sparse Jacobian into the common prepared-problem shape.
    pub fn prepare_sparse_aot_problem(
        &self,
        residual_fn_name: &str,
        jacobian_fn_name: &str,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> BvpPreparedSparseAotProblem {
        BvpPreparedSparseAotProblem {
            residual_fn_name: residual_fn_name.to_string(),
            jacobian_fn_name: jacobian_fn_name.to_string(),
            variable_names: self.variable_string.clone(),
            param_names: self.parameters_string.clone(),
            residuals: self.vector_of_functions.clone(),
            sparse_entries: self.symbolic_jacobian_sparse_entries_owned(),
            shape: (
                self.vector_of_functions.len(),
                self.vector_of_variables.len(),
            ),
            residual_strategy,
            jacobian_strategy,
        }
    }

    /// Selects the sparse BVP backend branch through the shared lifecycle
    /// selection layer while keeping an owned BVP bridge object.
    ///
    /// This is the main thin integration entrypoint for the BVP module:
    /// - symbolic/discretized BVP data stays owned here,
    /// - the generic lifecycle stack still decides whether AOT is compiled,
    ///   merely registered, or missing,
    /// - the caller can then route into lambdify/numeric/AOT execution paths
    ///   without rebuilding prepared-problem metadata by hand.
    pub fn select_sparse_backend(
        &self,
        residual_fn_name: &str,
        jacobian_fn_name: &str,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
    ) -> BvpSelectedSparseBackend {
        let prepared_problem = self.prepare_sparse_aot_problem(
            residual_fn_name,
            jacobian_fn_name,
            residual_strategy,
            jacobian_strategy,
        );
        let prepared_binding = prepared_problem.clone();
        let prepared = crate::symbolic::codegen_provider_api::PreparedProblem::sparse(
            prepared_binding.as_prepared_problem(),
        );
        let selected = select_backend(&prepared, policy, resolver);

        BvpSelectedSparseBackend {
            prepared_problem,
            requested_backend: selected.requested_backend,
            effective_backend: selected.effective_backend,
            matrix_backend: selected.matrix_backend,
            aot_resolution: selected.aot_resolution,
        }
    }

    fn prepare_sparse_backend_execution_timed(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        residual_fn_name: &str,
        jacobian_fn_name: &str,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        timer_hash: &mut HashMap<String, f64>,
    ) -> BvpSparseExecutionPlan {
        let prepare_started = Instant::now();
        let prepared_problem = self.prepare_sparse_aot_problem(
            residual_fn_name,
            jacobian_fn_name,
            residual_strategy,
            jacobian_strategy,
        );
        timer_hash.insert(
            "sparse AOT preparation time".to_string(),
            prepare_started.elapsed().as_secs_f64(),
        );

        let selection_started = Instant::now();
        let prepared_binding = prepared_problem.clone();
        let prepared = crate::symbolic::codegen_provider_api::PreparedProblem::sparse(
            prepared_binding.as_prepared_problem(),
        );
        let selected = select_backend(&prepared, policy, resolver);
        timer_hash.insert(
            "backend selection time".to_string(),
            selection_started.elapsed().as_secs_f64(),
        );

        let selected = BvpSelectedSparseBackend {
            prepared_problem,
            requested_backend: selected.requested_backend,
            effective_backend: selected.effective_backend,
            matrix_backend: selected.matrix_backend,
            aot_resolution: selected.aot_resolution,
        };

        let binding_started = Instant::now();
        let execution = match selected.effective_backend {
            SelectedBackendKind::Numeric => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Numeric,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::NumericRequested(selected)
            }
            SelectedBackendKind::Lambdify => {
                let config = BvpBackendConfig::new(
                    BvpBackendKind::Lambdify,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                );
                self.compile_lambdified_problem_with_config(arg, variable_str, config);
                BvpSparseExecutionPlan::LambdifyReady(selected)
            }
            SelectedBackendKind::AotCompiled => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Aot,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::AotCompiled(selected)
            }
            SelectedBackendKind::AotRegisteredButNotBuilt => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Aot,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::AotRegisteredButNotBuilt(selected)
            }
            SelectedBackendKind::AotMissing => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Aot,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::AotMissing(selected)
            }
        };
        timer_hash.insert(
            "runtime binding time".to_string(),
            binding_started.elapsed().as_secs_f64(),
        );
        execution
    }

    /// Builds a sparse BVP execution plan and, when the selected branch is
    /// lambdify, eagerly prepares the current faer/dense legacy evaluators.
    pub fn prepare_sparse_backend_execution(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        residual_fn_name: &str,
        jacobian_fn_name: &str,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
    ) -> BvpSparseExecutionPlan {
        let selected = self.select_sparse_backend(
            residual_fn_name,
            jacobian_fn_name,
            residual_strategy,
            jacobian_strategy,
            policy,
            resolver,
        );

        match selected.effective_backend {
            SelectedBackendKind::Numeric => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Numeric,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::NumericRequested(selected)
            }
            SelectedBackendKind::Lambdify => {
                let config = BvpBackendConfig::new(
                    BvpBackendKind::Lambdify,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                );
                self.compile_lambdified_problem_with_config(arg, variable_str, config);
                BvpSparseExecutionPlan::LambdifyReady(selected)
            }
            SelectedBackendKind::AotCompiled => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Aot,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::AotCompiled(selected)
            }
            SelectedBackendKind::AotRegisteredButNotBuilt => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Aot,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::AotRegisteredButNotBuilt(selected)
            }
            SelectedBackendKind::AotMissing => {
                self.set_backend_config(BvpBackendConfig::new(
                    BvpBackendKind::Aot,
                    BvpMatrixBackend::from_provider_matrix_backend(selected.matrix_backend),
                ));
                BvpSparseExecutionPlan::AotMissing(selected)
            }
        }
    }

    /// Wraps a prepared sparse execution plan into a BVP-side transitional
    /// solver provider.
    ///
    /// For now the returned provider is directly callable only for the main
    /// `LambdifyReady` sparse path. AOT branches still expose resolved artifact
    /// metadata and sparse structure through the same object shape so the
    /// higher-level solver integration can use one route while compiled
    /// callback hookup is introduced incrementally.
    pub fn sparse_solver_provider<'a>(
        &'a mut self,
        execution: BvpSparseExecutionPlan,
    ) -> BvpSparseSolverProvider<'a> {
        let structure = execution
            .selected()
            .prepared_problem
            .as_prepared_problem()
            .jacobian_structure()
            .clone();

        BvpSparseSolverProvider {
            jacobian: self,
            execution,
            structure,
        }
    }

    /// Consumes the current symbolic BVP object into an assembled
    /// solver-facing sparse bundle.
    ///
    /// This is the main "collected" handoff shape for outer BVP/NR solvers:
    /// backend selection, runtime callbacks, sparse structure, and solver
    /// metadata are packaged together instead of being pulled out from several
    /// `Jacobian` fields manually.
    pub fn into_sparse_solver_bundle(
        self,
        execution: BvpSparseExecutionPlan,
    ) -> BvpSparseSolverBundle {
        let selected = execution.selected().clone();
        let sparse_structure = selected
            .prepared_problem
            .as_prepared_problem()
            .jacobian_structure()
            .clone();
        let linked_callbacks = if matches!(execution, BvpSparseExecutionPlan::AotCompiled(_)) {
            linked_sparse_runtime_callbacks(
                &selected.prepared_problem,
                self.parameter_values.as_deref(),
                None,
            )
        } else {
            None
        };
        let callbacks_available = matches!(execution, BvpSparseExecutionPlan::LambdifyReady(_))
            || linked_callbacks.is_some();
        let (linked_residual, linked_jacobian) =
            linked_callbacks.map_or((None, None), |(fun, jac)| (Some(fun), Some(jac)));

        BvpSparseSolverBundle {
            execution,
            residual_function: if callbacks_available {
                if matches!(selected.effective_backend, SelectedBackendKind::Lambdify) {
                    Some(self.residiual_function)
                } else {
                    linked_residual
                }
            } else {
                None
            },
            jacobian_function: if callbacks_available {
                if matches!(selected.effective_backend, SelectedBackendKind::Lambdify) {
                    self.jac_function
                } else {
                    linked_jacobian
                }
            } else {
                None
            },
            variable_string: self.variable_string,
            bounds_vec: self.bounds,
            rel_tolerance_vec: self.rel_tolerance_vec,
            bandwidth: self.bandwidth,
            bc_position_and_value: self.BC_pos_n_values,
            sparse_structure,
        }
    }

    /// Fallible wrapper around `generate_BVP_with_backend_selection`.
    ///
    /// This is the first end-to-end error boundary for the new BVP/AOT path:
    /// unexpected panics in symbolic preparation, backend selection, or
    /// callback preparation are converted into a typed error so outer solver
    /// orchestration can log or fall back deliberately.
    #[allow(clippy::too_many_arguments)]
    pub fn try_generate_BVP_with_backend_selection(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
    ) -> Result<BvpSparseExecutionPlan, BvpBackendIntegrationError> {
        info!(
            "BVP backend selection started: backend_policy={policy:?}, method={method}, params={}, steps={:?}",
            params.map_or(0, |p| p.len()),
            n_steps
        );
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.generate_BVP_with_backend_selection(
                eq_system,
                values,
                arg,
                params,
                t0,
                param,
                n_steps,
                h,
                mesh,
                BorderConditions,
                Bounds,
                rel_tolerance,
                scheme,
                method,
                bandwidth,
                policy,
                resolver,
            )
        }));

        match result {
            Ok(execution) => {
                info!(
                    "BVP backend selection finished: effective_backend={:?}",
                    execution.selected().effective_backend
                );
                Ok(execution)
            }
            Err(payload) => {
                let message = if let Some(message) = payload.downcast_ref::<&str>() {
                    (*message).to_string()
                } else if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else {
                    "unknown panic payload".to_string()
                };
                error!("BVP backend selection failed: {message}");
                Err(BvpBackendIntegrationError::PipelinePanicked(message))
            }
        }
    }

    /// Fallible wrapper returning the assembled solver-facing sparse bundle.
    #[allow(clippy::too_many_arguments)]
    pub fn try_generate_sparse_solver_bundle_with_backend_selection(
        mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
    ) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
        let execution = self.try_generate_BVP_with_backend_selection(
            eq_system,
            values,
            arg,
            params,
            t0,
            param,
            n_steps,
            h,
            mesh,
            BorderConditions,
            Bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
            policy,
            resolver,
        )?;
        Ok(self.into_sparse_solver_bundle(execution))
    }

    /// Fallible wrapper around `generate_BVP_with_backend_selection_and_chunking`.
    ///
    /// This variant allows the outer solver layer to override residual and
    /// sparse-Jacobian chunk planning instead of relying on the internal BVP
    /// heuristics.
    #[allow(clippy::too_many_arguments)]
    pub fn try_generate_BVP_with_backend_selection_and_chunking(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> Result<BvpSparseExecutionPlan, BvpBackendIntegrationError> {
        info!(
            "BVP backend selection with explicit chunking started: backend_policy={policy:?}, method={method}, params={}, steps={:?}, residual_strategy={residual_strategy:?}, jacobian_strategy={jacobian_strategy:?}",
            params.map_or(0, |p| p.len()),
            n_steps
        );
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.generate_BVP_with_backend_selection_and_chunking(
                eq_system,
                values,
                arg,
                params,
                t0,
                param,
                n_steps,
                h,
                mesh,
                BorderConditions,
                Bounds,
                rel_tolerance,
                scheme,
                method,
                bandwidth,
                policy,
                resolver,
                residual_strategy,
                jacobian_strategy,
            )
        }));

        match result {
            Ok(execution) => {
                info!(
                    "BVP backend selection with explicit chunking finished: effective_backend={:?}",
                    execution.selected().effective_backend
                );
                Ok(execution)
            }
            Err(payload) => {
                let message = if let Some(message) = payload.downcast_ref::<&str>() {
                    (*message).to_string()
                } else if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else {
                    "unknown panic payload".to_string()
                };
                error!("BVP backend selection with explicit chunking failed: {message}");
                Err(BvpBackendIntegrationError::PipelinePanicked(message))
            }
        }
    }

    /// Fallible wrapper returning the assembled solver-facing sparse bundle
    /// with explicit chunking strategies.
    #[allow(clippy::too_many_arguments)]
    pub fn try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
        mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
        let execution = self.try_generate_BVP_with_backend_selection_and_chunking(
            eq_system,
            values,
            arg,
            params,
            t0,
            param,
            n_steps,
            h,
            mesh,
            BorderConditions,
            Bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
            policy,
            resolver,
            residual_strategy,
            jacobian_strategy,
        )?;
        Ok(self.into_sparse_solver_bundle(execution))
    }

    fn flattened_argument_names_owned(&self, variable_str: &[&str]) -> Vec<String> {
        let mut names = Vec::with_capacity(
            self.parameters_string
                .len()
                .saturating_add(variable_str.len()),
        );
        names.extend(self.parameters_string.iter().cloned());
        names.extend(variable_str.iter().map(|name| (*name).to_string()));
        names
    }

    /// Sets the modern backend configuration and mirrors the matrix choice into
    /// the legacy `method` string used by the existing BVP solver stack.
    pub fn set_backend_config(&mut self, config: BvpBackendConfig) {
        self.backend_config = config;
        self.method = config.matrix_backend.legacy_method().to_string();
    }

    /// Returns the best-known backend configuration for this instance.
    ///
    /// If the legacy `method` field was changed directly, this method keeps the
    /// matrix backend in sync while preserving the currently selected backend
    /// family (`Numeric`, `Lambdify`, `Aot`).
    pub fn backend_config(&self) -> BvpBackendConfig {
        if let Some(matrix_backend) = BvpMatrixBackend::from_legacy_method(self.method.as_str()) {
            BvpBackendConfig::new(self.backend_config.backend_kind, matrix_backend)
        } else {
            self.backend_config
        }
    }

    fn compile_jacobian_for_matrix_backend(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        matrix_backend: BvpMatrixBackend,
    ) {
        match matrix_backend {
            BvpMatrixBackend::Dense => self.lambdify_jacobian_DMatrix_par(arg, variable_str),
            BvpMatrixBackend::SprsCsMat => self.lambdify_jacobian_CsMat(arg, variable_str),
            BvpMatrixBackend::NalgebraCsMatrix => {
                self.lambdify_jacobian_CsMatrix(arg, variable_str)
            }
            BvpMatrixBackend::FaerSparseCol => {
                self.lambdify_jacobian_SparseColMat_parallel2(arg, variable_str)
            }
        }
    }

    fn compile_residual_for_matrix_backend(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        matrix_backend: BvpMatrixBackend,
    ) {
        match matrix_backend {
            BvpMatrixBackend::Dense => self.lambdify_residual_DVector(arg, variable_str),
            BvpMatrixBackend::SprsCsMat => self.lambdify_residual_CsVec(arg, variable_str),
            BvpMatrixBackend::NalgebraCsMatrix => self.lambdify_residual_DVector(arg, variable_str),
            BvpMatrixBackend::FaerSparseCol => {
                self.lambdify_residual_Col_parallel2(arg, variable_str)
            }
        }
    }

    /// Compiles both Jacobian and residual evaluators using an explicit backend
    /// config while preserving the legacy BVP solver interfaces.
    pub fn compile_lambdified_problem_with_config(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        config: BvpBackendConfig,
    ) {
        match config.backend_kind {
            BvpBackendKind::Lambdify => {
                self.set_backend_config(config);
                self.compile_jacobian_for_matrix_backend(
                    arg,
                    variable_str.clone(),
                    config.matrix_backend,
                );
                self.compile_residual_for_matrix_backend(arg, variable_str, config.matrix_backend);
            }
            BvpBackendKind::Numeric => {
                panic!("Numeric backend integration is not wired inside symbolic_functions_BVP yet")
            }
            BvpBackendKind::Aot => {
                panic!("AOT backend integration is not wired inside symbolic_functions_BVP yet")
            }
        }
    }

    /// Computes the symbolic Jacobian matrix with bandwidth optimization and smart sparsity detection.
    ///
    /// This is the most optimized Jacobian computation method, combining bandwidth-aware computation
    /// with smart variable tracking for maximum performance on banded sparse systems.
    ///
    /// # Performance Optimizations
    /// 1. **Bandwidth optimization**: Only computes derivatives within the specified band (kl, ku)
    /// 2. **HashSet variable lookup**: O(1) variable presence checking using pre-computed HashSets
    /// 3. **Smart sparsity**: Skips derivatives for variables not present in each equation
    /// 4. **Parallel row computation**: Uses rayon for concurrent processing of Jacobian rows
    /// 5. **Pre-allocated rows**: Full-width rows initialized with zeros, selective assignment
    ///
    /// # Bandwidth Logic
    /// For equation i, only computes derivatives for variables j where:
    /// - `left_border ≤ j < right_border`
    /// - `left_border = max(0, i - kl - 1)`
    /// - `right_border = min(n_vars, i + ku + 1)`
    ///
    /// # When to Use
    /// - **RECOMMENDED** for large banded sparse systems with known bandwidth
    /// - BVP discretizations with local coupling (most common case)
    /// - Systems where bandwidth << matrix_size for maximum benefit
    pub fn calc_jacobian_parallel_smart_optimized_with_given_bandwidth(&mut self) {
        assert!(self.variables_for_all_disrete.len() > 0);
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = &self.variable_string;
        let bandwidth = self.bandwidth;
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        // Convert to HashSet for O(1) lookup
        let variable_sets: Vec<HashSet<&String>> = self
            .variables_for_all_disrete
            .iter()
            .map(|vars| vars.iter().collect())
            .collect();

        let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = (0
            ..vector_of_functions_len)
            .into_par_iter()
            .map(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    (vector_of_variables_len, 0)
                };

                let mut row = vec![Expr::Const(0.0); vector_of_variables_len];
                let mut sparse_entries = Vec::new();
                for j in left_border..right_border {
                    let variable = &variable_string_vec[j];
                    if variable_sets[i].contains(variable) {
                        let mut partial = Expr::diff(&self.vector_of_functions[i], variable);
                        partial = partial.simplify();
                        if !partial.is_zero() {
                            sparse_entries.push((i, j, partial.clone()));
                        }
                        row[j] = partial;
                    }
                }
                (row, sparse_entries)
            })
            .collect();
        self.symbolic_jacobian = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
        self.symbolic_jacobian_sparse = rows_and_sparse
            .into_iter()
            .flat_map(|(_, entries)| entries)
            .collect();
    }

    /// Computes the symbolic Jacobian matrix with smart sparsity detection and HashSet optimization.
    ///
    /// This method provides significant performance improvements over naive differentiation by using
    /// variable tracking to skip zero derivatives. It's the optimal choice when bandwidth is unknown.
    ///
    /// # Performance Optimizations
    /// 1. **HashSet variable lookup**: Converts variable lists to HashSets for O(1) contains() checks
    /// 2. **Smart sparsity**: Only computes ∂F_i/∂x_j if variable x_j appears in equation F_i
    /// 3. **Parallel computation**: Uses rayon for concurrent processing across equations
    /// 4. **Pre-allocation**: Vectors allocated with exact capacity to avoid reallocations
    /// 5. **Zero insertion**: Directly inserts Expr::Const(0.0) for missing variables
    ///
    /// # Algorithm Complexity
    /// - **Without optimization**: O(n_eqs × n_vars × diff_cost)
    /// - **With this optimization**: O(n_eqs × avg_vars_per_eq × diff_cost)
    /// - **Typical speedup**: 5-50x for sparse BVP systems
    ///
    /// # When to Use
    /// - **RECOMMENDED** for sparse systems without known bandwidth
    /// - General BVP discretizations where bandwidth detection is expensive
    /// - Systems with highly variable sparsity patterns per equation
    pub fn calc_jacobian_parallel_smart_optimized(&mut self) {
        assert!(self.variables_for_all_disrete.len() > 0);
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = &self.variable_string;

        // Convert to HashSet for O(1) lookup
        let variable_sets: Vec<HashSet<&String>> = self
            .variables_for_all_disrete
            .iter()
            .map(|vars| vars.iter().collect())
            .collect();

        let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = self
            .vector_of_functions
            .par_iter()
            .enumerate()
            .map(|(i, function)| {
                // Pre-allocate with exact capacity
                let mut vector_of_partial_derivatives =
                    Vec::with_capacity(self.vector_of_variables.len());
                let mut sparse_entries = Vec::new();

                for j in 0..self.vector_of_variables.len() {
                    let variable = &variable_string_vec[j];
                    if variable_sets[i].contains(variable) {
                        let mut partial = Expr::diff(function, variable);
                        partial = partial.simplify();
                        if !partial.is_zero() {
                            sparse_entries.push((i, j, partial.clone()));
                        }
                        vector_of_partial_derivatives.push(partial);
                    } else {
                        vector_of_partial_derivatives.push(Expr::Const(0.0));
                    }
                }
                (vector_of_partial_derivatives, sparse_entries)
            })
            .collect();

        self.symbolic_jacobian = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
        self.symbolic_jacobian_sparse = rows_and_sparse
            .into_iter()
            .flat_map(|(_, entries)| entries)
            .collect();
    }

    /// Computes the symbolic Jacobian matrix using parallel differentiation with smart sparsity optimization.
    ///
    /// This is the most efficient Jacobian computation method. It uses the `variables_for_all_disrete`
    /// field to only compute partial derivatives for variables that actually appear in each equation,
    /// dramatically reducing computation time for sparse systems.
    ///
    /// # Performance Features
    /// - Parallel computation across equations using rayon
    /// - Smart zero-derivative detection: skips ∂F_i/∂x_j if x_j not in equation i
    /// - Automatic symbolic simplification of computed derivatives
    ///
    /// # Panics
    /// - If `vector_of_functions` is empty
    /// - If `vector_of_variables` is empty  
    /// - If `variables_for_all_disrete` is empty (must call discretization first)
    ///
    /// # Note
    /// Must be called after `discretization_system_BVP_par()` which populates `variables_for_all_disrete`.
    pub fn calc_jacobian_parallel_smart(&mut self) {
        assert!(self.variables_for_all_disrete.len() > 0);
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = self.variable_string.clone();
        let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = self
            .vector_of_functions
            .par_iter()
            .enumerate()
            .map(|(i, function)| {
                let mut vector_of_partial_derivatives = Vec::new();
                let mut sparse_entries = Vec::new();
                // let function = function.clone();
                for j in 0..self.vector_of_variables.len() {
                    let variable = &variable_string_vec[j]; // obviously if function does not contain variable its derivative should be 0
                    let list_of_vaiables_for_this_eq = &self.variables_for_all_disrete[i]; // so we can only calculate derivative for variables that are used in this equation
                    if list_of_vaiables_for_this_eq.contains(variable) {
                        let mut partial = Expr::diff(&function, variable);
                        partial = partial.simplify();
                        if !partial.is_zero() {
                            sparse_entries.push((i, j, partial.clone()));
                        }
                        vector_of_partial_derivatives.push(partial);
                    } else {
                        vector_of_partial_derivatives.push(Expr::Const(0.0));
                    }
                }
                (vector_of_partial_derivatives, sparse_entries)
            })
            .collect();

        self.symbolic_jacobian = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
        self.symbolic_jacobian_sparse = rows_and_sparse
            .into_iter()
            .flat_map(|(_, entries)| entries)
            .collect();
    }
    /// Computes the symbolic Jacobian matrix using standard parallel differentiation.
    ///
    /// This method computes all partial derivatives ∂F_i/∂x_j without sparsity optimization.
    /// Use `calc_jacobian_parallel_smart()` for better performance on sparse systems.
    ///
    /// # Performance Features
    /// - Parallel computation across equations using rayon
    /// - Automatic symbolic simplification of computed derivatives
    ///
    /// # Panics
    /// - If `vector_of_functions` is empty
    /// - If `vector_of_variables` is empty
    ///
    /// # When to Use
    /// - For dense systems where most variables appear in most equations
    /// - When `variables_for_all_disrete` is not available (before discretization)
    pub fn calc_jacobian_parallel(&mut self) {
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = self.variable_string.clone();
        let rows_and_sparse: Vec<(Vec<Expr>, Vec<(usize, usize, Expr)>)> = self
            .vector_of_functions
            .par_iter()
            .enumerate()
            .map(|(i, function)| {
                let mut vector_of_partial_derivatives = Vec::new();
                let mut sparse_entries = Vec::new();
                // let function = function.clone();
                for j in 0..self.vector_of_variables.len() {
                    let mut partial = Expr::diff(&function, &variable_string_vec[j]);
                    partial = partial.simplify();
                    if !partial.is_zero() {
                        sparse_entries.push((i, j, partial.clone()));
                    }
                    vector_of_partial_derivatives.push(partial);
                }
                (vector_of_partial_derivatives, sparse_entries)
            })
            .collect();

        self.symbolic_jacobian = rows_and_sparse.iter().map(|(row, _)| row.clone()).collect();
        self.symbolic_jacobian_sparse = rows_and_sparse
            .into_iter()
            .flat_map(|(_, entries)| entries)
            .collect();
    }
    //

    ////////////////////////////////////////////////////////////////////////////////////
    //  GENERIC FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////////

    /// Compiles residual functions using generic VectorType trait for backend flexibility.
    ///
    /// This method creates a generic residual function that works with any vector type
    /// implementing the VectorType trait. It's part of the experimental generic backend system.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (typically time "t" or "x")
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Note
    /// This is an experimental generic interface. For production use, prefer the specific
    /// backend methods like `lambdify_residual_DVector()` or `lambdify_residual_Col_parallel2()`.
    pub fn lambdify_residual_VectorType(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            _arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>> {
            Box::new(move |_x: f64, v: &dyn VectorType| -> Box<dyn VectorType> {
                let mut result = v.zeros(vector_of_functions.len());

                // Iterate through functions and assign values
                for (i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_borrowed_thread_safe(
                        &func,
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                    let v: Vec<f64> = v.iterate().collect();
                    let value = func(v.as_slice());
                    result = result.assign_value(i, value);
                }
                result
            })
        }

        let fun: Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>> = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.iter().map(|s| s.to_string()).collect(),
        );
        let residual_function = convert_to_fun(fun);
        self.residiual_function = residual_function;
    }

    /// Generates a generic parallel Jacobian evaluator using trait objects.
    ///
    /// Creates a closure that evaluates the symbolic Jacobian matrix using generic
    /// VectorType and MatrixType traits. Supports bandwidth optimization for sparse matrices.
    ///
    /// # Arguments
    /// * `jac` - 2D vector of symbolic partial derivatives
    /// * `vector_of_functions_len` - Number of equations in the system
    /// * `vector_of_variables_len` - Number of variables in the system
    /// * `variable_str` - Variable names as owned strings
    /// * `_arg` - Independent variable name (unused in current implementation)
    /// * `bandwidth` - Optional (kl, ku) bandwidth for banded matrix optimization
    ///
    /// # Returns
    /// A boxed closure that takes (time, variables) and returns a matrix of partial derivatives
    ///
    /// # Performance Features
    /// - Parallel evaluation using rayon
    /// - Bandwidth-aware computation for sparse matrices
    /// - Thread-safe triplet collection using Mutex
    /// - Zero-threshold filtering (T = 1e-12) to maintain sparsity
    pub fn jacobian_generate_generic_par(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
        bandwidth: Option<(usize, usize)>,
    ) -> Box<dyn Fn(f64, &(dyn VectorType + Send + Sync)) -> Box<dyn MatrixType>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(
            move |_x: f64, v: &(dyn VectorType + Send + Sync)| -> Box<dyn MatrixType> {
                let mut vector_of_derivatives =
                    vec![0.0; vector_of_functions_len * vector_of_variables_len];
                let vector_of_derivatives_mutex = std::sync::Mutex::new(&mut vector_of_derivatives);

                let mut vector_of_triplets = Vec::new();
                let vector_of_triplets_mutex = std::sync::Mutex::new(&mut vector_of_triplets);
                (0..vector_of_functions_len).into_par_iter().for_each(|i| {
                    let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                        let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                        let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                            0
                        } else {
                            i - kl - 1
                        };
                        (right_border, left_border)
                    } else {
                        let right_border = vector_of_variables_len;
                        let left_border = 0;
                        (right_border, left_border)
                    };
                    for j in left_border..right_border {
                        // if jac[i][j] != Expr::Const(0.0) { println!("i = {}, j = {}, {}", i, j,  &jac[i][j]);}
                        let partial_func = Expr::lambdify_borrowed_thread_safe(
                            &jac[i][j],
                            variable_str
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .as_slice(),
                        );

                        let v_vec: Vec<f64> = v.iterate().collect();
                        let _vec_type = &v.vec_type();
                        let P = partial_func(v_vec.as_slice());
                        if P.abs() > T {
                            vector_of_derivatives_mutex.lock().unwrap()
                                [i * vector_of_functions_len + j] = P;
                            let triplet = (i, j, P);
                            vector_of_triplets_mutex.lock().unwrap().push(triplet);
                        }
                    }
                }); //par_iter()

                let new_function_jacobian: Box<dyn MatrixType> = v.from_vector(
                    vector_of_functions_len,
                    vector_of_variables_len,
                    &vector_of_derivatives,
                    vector_of_triplets,
                );
                //  panic!("stop here");
                new_function_jacobian
            },
        ) // end of box
    } // end of function

    /// Compiles the symbolic Jacobian using the generic trait-based backend system.
    ///
    /// This method creates a generic Jacobian function that works with any matrix type
    /// implementing the MatrixType trait. Currently incomplete due to Send+Sync trait limitations.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Status
    /// This method is currently incomplete and commented out due to trait object limitations.
    /// The generic backend system needs further development for full functionality.
    ///
    /// # Note
    /// For production use, prefer the specific backend methods like `lambdify_jacobian_DMatrix_par()`.
    pub fn lambdify_jacobian_generic(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let _new_jac = Jacobian::jacobian_generate_generic_par(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            bandwidth,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;
        // TODO! Send + Sync trait is not implemented
        //   let boxed_jac: Box<dyn Jac> = convert_to_jac(new_jac);
        //  self.jac_function = Some(boxed_jac);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    ///                             NALGEBRA DENSE
    ////////////////////////////////////////////////////////////////////////////////////

    /// Compiles the symbolic Jacobian to a dense nalgebra DMatrix evaluator with parallel optimization.
    ///
    /// This method creates a high-performance dense matrix Jacobian evaluator using nalgebra's DMatrix.
    /// It pre-compiles all non-zero symbolic derivatives and uses parallel evaluation for optimal performance.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (typically "t" or "x")
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Performance Features
    /// - **Pre-compilation**: All symbolic derivatives compiled once during setup
    /// - **Parallel evaluation**: Uses rayon for concurrent derivative evaluation
    /// - **Bandwidth optimization**: Respects sparse matrix bandwidth if set
    /// - **Zero filtering**: Only evaluates non-zero symbolic derivatives
    /// - **Thread-safe**: Uses `Send + Sync` closures and Mutex for thread safety
    /// - **Outer loop parallelization**: Uses `flat_map()` pattern for better load balancing
    ///
    /// # Matrix Structure
    /// Creates a dense DMatrix where element (i,j) = ∂F_i/∂x_j evaluated at the given point.
    ///
    /// # When to Use
    /// - Small to medium systems (< 1000 variables)
    /// - Dense or moderately sparse Jacobians
    /// - When nalgebra ecosystem integration is important
    pub fn lambdify_jacobian_DMatrix_par(&mut self, arg: &str, variable_str: Vec<&str>) {
        let jac = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let flattened_argument_names_owned =
            self.flattened_argument_names_owned(variable_str.as_slice());
        let parameter_values = self.parameter_values.clone();
        let parameter_count = self.parameters_string.len();

        // Convert to owned strings to avoid lifetime issues
        //  let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();

        // Create jacobian positions in parallel using outer loop parallelization
        let jacobian_positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = (0
            ..vector_of_functions_len)
            .into_par_iter()
            .flat_map(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    (vector_of_variables_len, 0)
                };

                (left_border..right_border)
                    .filter_map(|j| {
                        let symbolic_partial_derivative = &jac[i][j];
                        if !symbolic_partial_derivative.is_zero() {
                            let compiled_func: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> =
                                Expr::lambdify_borrowed_thread_safe(
                                    &symbolic_partial_derivative,
                                    flattened_argument_names_owned
                                        .iter()
                                        .map(|s| s.as_str())
                                        .collect::<Vec<_>>()
                                        .as_slice(),
                                );
                            Some((i, j, compiled_func))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let new_jac = Box::new(move |_x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let mut v_vec = Vec::with_capacity(parameter_count + v.len());
            if parameter_count > 0 {
                let params = parameter_values.as_ref().unwrap_or_else(|| {
                    panic!("parameter values must be provided when parameters are configured")
                });
                v_vec.extend_from_slice(params.as_slice());
            }
            v_vec.extend(v.iter().cloned());
            let mut matrix = DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            let matrix_mutex = Mutex::new(&mut matrix);

            jacobian_positions
                .par_iter()
                .for_each(|(i, j, compiled_func)| {
                    let P = compiled_func(v_vec.as_slice());
                    if P.abs() > T {
                        let mut mat = matrix_mutex.lock().unwrap();
                        mat[(*i, *j)] = P;
                    }
                });

            matrix
        });

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Dense(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    /// Compiles the residual functions to a dense nalgebra DVector evaluator with parallel optimization.
    ///
    /// Creates a high-performance residual function evaluator that returns nalgebra DVector results.
    /// All symbolic functions are pre-compiled for maximum efficiency during repeated evaluations.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (typically "t" or "x")
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Performance Features
    /// - **Pre-compilation**: All residual functions compiled once during setup
    /// - **Parallel evaluation**: Uses rayon for concurrent function evaluation
    /// - **Thread-safe closures**: Uses `Send + Sync` bounds for parallel safety
    /// - **Memory efficient**: Direct vector construction without intermediate collections
    ///
    /// # Output Format
    /// Returns DVector where element i = F_i(t, y) for the i-th residual equation.
    ///
    /// # When to Use
    /// - Dense vector operations with nalgebra
    /// - Small to medium systems where dense storage is acceptable
    /// - Integration with nalgebra-based linear solvers
    pub fn lambdify_residual_DVector(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        let parameter_values = self.parameter_values.clone();
        let parameter_count = self.parameters_string.len();

        // Convert to owned strings to avoid lifetime issues
        let variable_str_owned = self.flattened_argument_names_owned(variable_str.as_slice());

        let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = vector_of_functions
            .par_iter()
            .map(|func| {
                Expr::lambdify_borrowed_thread_safe(
                    func,
                    variable_str_owned
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
            })
            .collect();

        let fun = Box::new(move |_x: f64, v: &DVector<f64>| -> DVector<f64> {
            let mut v_vec = Vec::with_capacity(parameter_count + v.len());
            if parameter_count > 0 {
                let params = parameter_values.as_ref().unwrap_or_else(|| {
                    panic!("parameter values must be provided when parameters are configured")
                });
                v_vec.extend_from_slice(params.as_slice());
            }
            v_vec.extend(v.iter().cloned());
            let result: Vec<_> = compiled_functions
                .par_iter()
                .map(|func| func(v_vec.as_slice()))
                .collect();
            DVector::from_vec(result)
        });
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun));
        self.residiual_function = boxed_fun;
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    ////                             SPRS CRATE SPARSE FUNCTIONS
    ///////////////////////////////////////////////////////////////////////////////////////////

    /// Generates a sparse Jacobian evaluator using the sprs crate's CsMat format.
    ///
    /// Creates a closure that evaluates the symbolic Jacobian matrix and returns it as a
    /// sprs::CsMat (Compressed Sparse Matrix). This is a legacy method with sequential evaluation.
    ///
    /// # Arguments
    /// * `jac` - 2D vector of symbolic partial derivatives
    /// * `vector_of_functions_len` - Number of equations in the system
    /// * `vector_of_variables_len` - Number of variables in the system
    /// * `variable_str` - Variable names as owned strings
    /// * `_arg` - Independent variable name (unused)
    ///
    /// # Returns
    /// A mutable closure that takes (time, sparse_vector) and returns a sparse matrix
    ///
    /// # Performance Notes
    /// - Sequential evaluation (not parallelized)
    /// - Uses sprs::CsMat compressed sparse row format
    /// - Evaluates all matrix elements (no sparsity optimization)
    ///
    /// # When to Use
    /// - Legacy code compatibility with sprs crate
    /// - When mutable closure semantics are required
    /// - Small systems where parallelization overhead isn't justified
    pub fn jacobian_generate_CsMat(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
    ) -> Box<dyn FnMut(f64, &CsVec<f64>) -> CsMat<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &CsVec<f64>| -> CsMat<f64> {
            let mut new_function_jacobian: CsMat<f64> =
                CsMat::zero((vector_of_functions_len, vector_of_variables_len));
            for i in 0..vector_of_functions_len {
                for j in 0..vector_of_variables_len {
                    // println!("i = {}, j = {}", i, j);
                    let partial_func = Expr::lambdify_borrowed_thread_safe(
                        &jac[i][j],
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.to_dense().to_vec();
                    new_function_jacobian.insert(j, i, partial_func(v_vec.as_slice()));
                }
            }
            new_function_jacobian
        })
    } // end of function
    /// Compiles the symbolic Jacobian to a sprs CsMat evaluator.
    ///
    /// Sets up the Jacobian function using the sprs crate's compressed sparse matrix format.
    /// This method wraps the generated CsMat evaluator in the trait object system.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Backend Details
    /// - Uses sprs::CsMat (Compressed Sparse Row format)
    /// - Sequential evaluation (not parallelized)
    /// - Wrapped in JacEnum::Sparse_1 variant
    ///
    /// # When to Use
    /// - Legacy compatibility with sprs-based code
    /// - When CSR format is specifically required
    /// - Small to medium sparse systems
    pub fn lambdify_jacobian_CsMat(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_CsMat(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_1(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    /// Compiles residual functions to a sprs CsVec evaluator for sparse vector operations.
    ///
    /// Creates a residual function evaluator that works with sprs sparse vectors (CsVec).
    /// This method is designed for systems where the residual vector itself is sparse.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (used in IVP-style evaluation)
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Backend Details
    /// - Uses sprs::CsVec (Compressed Sparse Vector format)
    /// - Supports IVP-style evaluation with time parameter
    /// - Sequential evaluation of residual functions
    /// - Wrapped in FunEnum::Sparse_1 variant
    ///
    /// # When to Use
    /// - Sparse residual vectors (many zero elements)
    /// - Integration with sprs-based linear algebra
    /// - IVP problems where time-dependent evaluation is needed
    pub fn lambdify_residual_CsVec(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64> + 'static> {
            Box::new(move |x: f64, v: &CsVec<f64>| -> CsVec<f64> {
                //  let mut result: CsVec<f64> = CsVec::new(n, indices, data);
                let mut result: CsVec<f64> = CsVec::empty(vector_of_functions.len());
                for (i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_IVP_owned(
                        func.to_owned(),
                        arg.as_str(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    result.append(i, func(x, v.to_dense().to_vec()));
                }
                result
            }) //enf of box
        } // end of function
        let fun = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_1(fun));
        self.residiual_function = boxed_fun;
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    //                            NALGEBRA SPARSE CRATE
    ///////////////////////////////////////////////////////////////////////////////////////////

    /// Generates a sparse Jacobian evaluator using nalgebra's CsMatrix format.
    ///
    /// Creates a closure that evaluates the symbolic Jacobian and returns it as nalgebra's
    /// compressed sparse matrix format. Uses dense intermediate storage for simplicity.
    ///
    /// # Arguments
    /// * `jac` - 2D vector of symbolic partial derivatives
    /// * `vector_of_functions_len` - Number of equations in the system
    /// * `vector_of_variables_len` - Number of variables in the system
    /// * `variable_str` - Variable names as owned strings
    /// * `_arg` - Independent variable name (unused)
    ///
    /// # Returns
    /// A mutable closure that takes (time, dense_vector) and returns a sparse matrix
    ///
    /// # Implementation Details
    /// - Uses DMatrix as intermediate storage, then converts to CsMatrix
    /// - Sequential evaluation (not parallelized)
    /// - Evaluates all matrix elements regardless of sparsity
    ///
    /// # When to Use
    /// - Integration with nalgebra's sparse matrix ecosystem
    /// - When CsMatrix format is specifically required
    /// - Small to medium systems where conversion overhead is acceptable
    pub fn jacobian_generate_CsMatrix(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> CsMatrix<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &DVector<f64>| -> CsMatrix<f64> {
            let _number_of_possible_non_zero: usize = jac.len();
            //let mut new_function_jacobian: CsMatrix<f64> = CsMatrix::new_uninitialized_generic(Dyn(vector_of_functions_len),
            //Dyn(vector_of_variables_len), number_of_possible_non_zero);
            let mut new_function_jacobian: DMatrix<f64> =
                DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            for i in 0..vector_of_functions_len {
                for j in 0..vector_of_variables_len {
                    // println!("i = {}, j = {}", i, j);
                    let partial_func = Expr::lambdify_borrowed_thread_safe(
                        &jac[i][j],
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    //new_function_jacobian = CsMatrix::from_triplet(vector_of_functions_len,
                    // vector_of_variables_len, &[i], &[j], &[partial_func(x, v_vec.clone())]   );
                    new_function_jacobian[(i, j)] = partial_func(v_vec.as_slice());
                }
            }
            new_function_jacobian.into()
        })
    } // end of function
    /// Compiles the symbolic Jacobian to a nalgebra CsMatrix evaluator.
    ///
    /// Sets up the Jacobian function using nalgebra's compressed sparse matrix format.
    /// This method integrates with nalgebra's sparse matrix ecosystem.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Backend Details
    /// - Uses nalgebra::sparse::CsMatrix format
    /// - Sequential evaluation with dense intermediate storage
    /// - Wrapped in JacEnum::Sparse_2 variant
    ///
    /// # When to Use
    /// - Integration with nalgebra's sparse linear algebra
    /// - When nalgebra CsMatrix format is specifically required
    /// - Medium-sized sparse systems
    pub fn lambdify_jacobian_CsMatrix(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_CsMatrix(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_2(new_jac));
        self.jac_function = Some(boxed_jac);
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    ///         FAER SPARSE CRATE
    ////////////////////////////////////////////////////////////////////////////////////////

    /// Compiles the symbolic Jacobian to a faer SparseColMat evaluator (sequential version).
    ///
    /// Creates a high-performance sparse Jacobian evaluator using the faer crate's SparseColMat.
    /// This version pre-compiles all functions sequentially to avoid thread safety issues.
    ///
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Performance Features
    /// - **Pre-compilation**: All non-zero derivatives compiled once during setup
    /// - **Bandwidth optimization**: Respects sparse matrix bandwidth if set
    /// - **Zero filtering**: Only compiles and stores non-zero symbolic derivatives
    /// - **Triplet assembly**: Uses faer's efficient triplet-based sparse matrix construction
    /// - **Sequential safety**: Avoids Send+Sync issues by compiling sequentially
    ///
    /// # Matrix Format
    /// Uses faer::sparse::SparseColMat (Compressed Sparse Column format) for optimal performance.
    ///
    /// # When to Use
    /// - Large sparse systems where faer's performance is critical
    /// - When thread safety during compilation is a concern
    /// - Systems with well-defined sparsity patterns
    pub fn lambdify_jacobian_SparseColMat_modified(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;

        // Pre-compile all non-zero jacobian elements sequentially to avoid Send issues
        let mut compiled_jacobian_elements = Vec::new();
        for i in 0..vector_of_functions_len {
            let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                    0
                } else {
                    i - kl - 1
                };
                (right_border, left_border)
            } else {
                (vector_of_variables_len, 0)
            };

            for j in left_border..right_border {
                let symbolic_partial_derivative = &symbolic_jacobian[i][j];
                if !symbolic_partial_derivative.is_zero() {
                    let compiled_func = Expr::lambdify_borrowed_thread_safe(
                        &symbolic_partial_derivative,
                        variable_str.as_slice(),
                    );
                    compiled_jacobian_elements.push((i, j, compiled_func));
                }
            }
        }

        let new_jac = Box::new(move |_x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();

            let mut vector_of_triplets = Vec::new();
            for (i, j, func) in &compiled_jacobian_elements {
                let P = func(v_vec.as_slice());
                if P.abs() > T {
                    vector_of_triplets.push(Triplet::new(*i, *j, P));
                }
            }

            SparseColMat::try_new_from_triplets(
                vector_of_functions_len,
                vector_of_variables_len,
                vector_of_triplets.as_slice(),
            )
            .unwrap()
        });

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_3(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    /// Compiles the symbolic Jacobian to a faer SparseColMat evaluator (parallel version 2 - RECOMMENDED).
    ///
    /// Creates the most efficient parallel sparse Jacobian evaluator using outer loop parallelization.
    /// This is the recommended method for large sparse systems due to superior load balancing.
    ///
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Performance Features
    /// - **Outer loop parallelization**: Uses `flat_map()` for optimal load balancing
    /// - **Pre-compilation**: All non-zero derivatives compiled during setup
    /// - **Thread-safe closures**: Uses `lambdify_borrowed_thread_safe()` for efficiency
    /// - **Bandwidth optimization**: Respects sparse matrix bandwidth if set
    /// - **Parallel evaluation**: Concurrent triplet assembly with Mutex protection
    ///
    /// # Parallelization Strategy
    /// ```ignore
    /// (0..n_rows).into_par_iter().flat_map(|i| {
    ///     (cols_for_row_i).filter_map(|j| compile_and_store(i, j))
    /// })
    /// ```
    ///
    /// # Why This is Best
    /// - Better load balancing than nested loops
    /// - Avoids thread contention in inner loops
    /// - Scales well with number of CPU cores
    /// - Optimal for sparse matrices with varying row densities
    ///
    /// # When to Use
    /// - **RECOMMENDED** for all large sparse systems
    /// - Production code requiring maximum performance
    /// - Systems with > 1000 variables
    pub fn lambdify_jacobian_SparseColMat_parallel2(
        &mut self,
        _arg: &str,
        variable_str: Vec<&str>,
    ) {
        let jac = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let flattened_argument_names_owned =
            self.flattened_argument_names_owned(variable_str.as_slice());
        let parameter_values = self.parameter_values.clone();
        let parameter_count = self.parameters_string.len();

        // Convert to owned strings to avoid lifetime issues
        //  let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();

        // Create jacobian positions in parallel using outer loop parallelization
        let jacobian_positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = (0
            ..vector_of_functions_len)
            .into_par_iter()
            .flat_map(|i| {
                let flattened_argument_names_owned = flattened_argument_names_owned.clone();
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    (vector_of_variables_len, 0)
                };

                (left_border..right_border)
                    .filter_map(|j| {
                        let symbolic_partial_derivative = &jac[i][j];
                        if !symbolic_partial_derivative.is_zero() {
                            let compiled_func: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> =
                                Expr::lambdify_borrowed_thread_safe(
                                    &symbolic_partial_derivative,
                                    flattened_argument_names_owned
                                        .iter()
                                        .map(|s| s.as_str())
                                        .collect::<Vec<_>>()
                                        .as_slice(),
                                );
                            Some((i, j, compiled_func))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let new_jac = Box::new(move |_x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let mut v_vec = Vec::with_capacity(parameter_count + v.len());
            if parameter_count > 0 {
                let params = parameter_values.as_ref().unwrap_or_else(|| {
                    panic!("parameter values must be provided when parameters are configured")
                });
                v_vec.extend_from_slice(params.as_slice());
            }
            v_vec.extend(v.iter().cloned());
            let triplets_mutex = std::sync::Mutex::new(Vec::new());

            // Compile and evaluate in parallel without storing closures
            jacobian_positions
                .par_iter()
                .for_each(|(i, j, compiled_func)| {
                    let P = compiled_func(v_vec.as_slice());
                    if P.abs() > T {
                        let mut triplets = triplets_mutex.lock().unwrap();
                        triplets.push(Triplet::new(*i, *j, P));
                    }
                });

            let triplets = triplets_mutex.lock().unwrap();
            SparseColMat::try_new_from_triplets(
                vector_of_functions_len,
                vector_of_variables_len,
                triplets.as_slice(),
            )
            .unwrap()
        });

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_3(new_jac));
        self.jac_function = Some(boxed_jac);
    }
    ////////////////////////////////RESIDUAL LAMBDIFICATION/////////////////////////////////////////////////////////

    /// Compiles residual functions to a faer Col evaluator (sequential pre-compilation version).
    ///
    /// Creates an optimized residual function evaluator using faer's Col vector format.
    /// This version pre-compiles all functions sequentially for maximum efficiency.
    ///
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Performance Features
    /// - **Pre-compilation**: All residual functions compiled once during setup (CRITICAL optimization)
    /// - **Sequential compilation**: Avoids thread safety issues during function compilation
    /// - **Efficient evaluation**: Direct vector construction without intermediate collections
    /// - **Memory efficient**: Uses `ColRef::from_slice().to_owned()` for zero-copy construction
    ///
    /// # Why Pre-compilation Matters
    /// This is the most significant optimization - compiling symbolic functions once during setup
    /// instead of on every evaluation provides 10-100x speedup for repeated evaluations.
    ///
    /// # When to Use
    /// - Production code where maximum performance is critical
    /// - Systems with frequent residual evaluations
    /// - When thread safety during compilation is a concern
    pub fn lambdify_residual_Col_modified(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;

        // Pre-compile all functions once (most significant optimization)
        let compiled_functions: Vec<_> = vector_of_functions
            .iter()
            .map(|func| Expr::lambdify_borrowed_thread_safe(&func, variable_str.as_slice()))
            .collect();

        let fun = Box::new(move |_x: f64, v: &Col<f64>| -> Col<f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let result: Vec<_> = compiled_functions
                .iter()
                .map(|func| func(v_vec.as_slice()))
                .collect();
            ColRef::from_slice(result.as_slice()).to_owned()
        });
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_3(fun));
        self.residiual_function = boxed_fun;
    }

    /// Compiles residual functions to a faer Col evaluator (parallel version 2 - RECOMMENDED).
    ///
    /// Creates the most efficient parallel residual function evaluator using borrowed references
    /// for optimal memory usage and performance.
    ///
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    ///
    /// # Performance Features
    /// - **Parallel compilation**: Functions compiled concurrently using rayon
    /// - **Borrowed references**: Uses `lambdify_borrowed_thread_safe()` for efficiency
    /// - **Parallel evaluation**: Concurrent function evaluation during residual computation
    /// - **Memory efficient**: Avoids unnecessary cloning during compilation
    ///
    /// # Why This is Recommended
    /// - More memory efficient than parallel version 1
    /// - Uses borrowed references instead of cloning expressions
    /// - Better performance for large symbolic expressions
    /// - Optimal balance of compilation and evaluation speed
    ///
    /// # When to Use
    /// - **RECOMMENDED** for all parallel residual evaluation
    /// - Large systems with complex symbolic expressions
    /// - Production code requiring optimal memory usage
    pub fn lambdify_residual_Col_parallel2(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        let parameter_values = self.parameter_values.clone();
        let parameter_count = self.parameters_string.len();

        // Convert to owned strings to avoid lifetime issues
        let variable_str_owned = self.flattened_argument_names_owned(variable_str.as_slice());

        let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = vector_of_functions
            .into_par_iter()
            .map(|func| {
                let compiled = Expr::lambdify_borrowed_thread_safe(
                    func,
                    variable_str_owned
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                compiled
            })
            .collect();

        let fun = Box::new(move |_x: f64, v: &Col<f64>| -> Col<f64> {
            let mut v_vec = Vec::with_capacity(parameter_count + v.len());
            if parameter_count > 0 {
                let params = parameter_values.as_ref().unwrap_or_else(|| {
                    panic!("parameter values must be provided when parameters are configured")
                });
                v_vec.extend_from_slice(params.as_slice());
            }
            v_vec.extend(v.iter().cloned());
            let result: Vec<_> = compiled_functions
                .par_iter()
                .map(|func| func(v_vec.as_slice()))
                .collect();
            ColRef::from_slice(result.as_slice()).to_owned()
        });
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_3(fun));
        self.residiual_function = boxed_fun;
    }
    ///////////////////////////////////////////////////////////////////////////
    ///              DISCRETIZED FUNCTIONS
    ///////////////////////////////////////////////////////////////////////////

    /// Removes numeric suffix from discretized variable names for efficient processing.
    ///
    /// Extracts the base variable name from discretized variables like "y_0", "y_1", "z_0", etc.
    /// This is used for mapping discretized variables back to their original names for bounds
    /// and tolerance processing.
    ///
    /// # Arguments
    /// * `input` - Variable name with potential numeric suffix (e.g., "y_0", "z_15")
    ///
    /// # Returns
    /// Base variable name without suffix (e.g., "y", "z")
    ///
    /// # Performance
    /// Uses `rfind('_')` for O(n) string processing, much faster than regex alternatives.
    ///
    /// # Examples
    /// ```ignore
    /// assert_eq!(remove_numeric_suffix("y_0"), "y");
    /// assert_eq!(remove_numeric_suffix("temperature_15"), "temperature");
    /// assert_eq!(remove_numeric_suffix("x"), "x"); // No suffix
    /// ```
    pub fn remove_numeric_suffix(input: &str) -> String {
        if let Some(pos) = input.rfind('_') {
            input[..pos].to_string()
        } else {
            input.to_string()
        }
    }
    /// Applies discretization scheme to a single equation at a specific time step.
    ///
    /// Transforms a continuous ODE equation into its discretized form using the specified
    /// numerical scheme. Handles variable renaming and time substitution for the discretization.
    ///
    /// # Arguments
    /// * `matrix_of_names` - 2D matrix of discretized variable names [time_step][variable_index]
    /// * `eq_i` - The symbolic equation to discretize
    /// * `values` - Original variable names (e.g., ["y", "z"])
    /// * `arg` - Independent variable name (typically "t" or "x")
    /// * `j` - Current time step index
    /// * `t` - Time value at step j
    /// * `scheme` - Discretization scheme ("forward" or "trapezoid")
    ///
    /// # Returns
    /// Discretized equation with renamed variables and substituted time value
    ///
    /// # Supported Schemes
    /// - **"forward"**: Forward Euler - f(t_j, y_j)
    /// - **"trapezoid"**: Trapezoidal rule - 0.5 * (f(t_j, y_j) + f(t_{j+1}, y_{j+1}))
    ///
    /// # Variable Renaming
    /// Original variables like "y", "z" become "y_j", "z_j" for the j-th time step.
    fn eq_step(
        matrix_of_names: &[Vec<String>],
        eq_i: &Expr,
        values: &[String],
        arg: &str,
        j: usize,
        t: f64,
        scheme: &str,
    ) -> Expr {
        let vec_of_names_on_step = &matrix_of_names[j];
        let hashmap_for_rename: HashMap<String, String> = values
            .iter()
            .zip(vec_of_names_on_step.iter())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        //set time value of j-th time step
        let eq_step_j = eq_i.rename_variables(&hashmap_for_rename);
        let eq_step_j = eq_step_j.set_variable(arg, t);

        match scheme {
            "forward" => eq_step_j,
            "trapezoid" => {
                let vec_of_names_on_step = &matrix_of_names[j + 1];
                let hashmap_for_rename: HashMap<String, String> = values
                    .iter()
                    .zip(vec_of_names_on_step.iter())
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();

                let eq_step_j_plus_1 = eq_i
                    .rename_variables(&hashmap_for_rename)
                    .set_variable(arg, t);
                Expr::Const(0.5) * (eq_step_j + eq_step_j_plus_1)
            }
            _ => panic!("Invalid scheme"),
        }
    }
    //
    /// Creates the time/spatial mesh for BVP discretization.
    ///
    /// Generates the discretization mesh either from explicit mesh points or by creating
    /// a uniform grid. Returns step sizes as symbolic expressions and mesh points.
    ///
    /// # Arguments
    /// * `n_steps` - Number of discretization steps (if uniform mesh)
    /// * `h` - Step size (if uniform mesh)
    /// * `mesh` - Explicit mesh points (overrides n_steps/h if provided)
    /// * `t0` - Starting point of the domain
    ///
    /// # Returns
    /// Tuple of (step_sizes, mesh_points, n_steps) where:
    /// - `step_sizes`: Vector of Expr::Const representing h_i = t_{i+1} - t_i
    /// - `mesh_points`: Vector of f64 mesh coordinates
    /// - `n_steps`: Total number of mesh points
    ///
    /// # Mesh Types
    /// - **Explicit mesh**: Uses provided mesh points with variable step sizes
    /// - **Uniform mesh**: Creates evenly spaced points with constant step size
    ///
    /// # Default Values
    /// - n_steps: 101 (100 intervals + 1 endpoint)
    /// - h: 1.0
    fn create_mesh(
        &self,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        t0: f64,
    ) -> (Vec<Expr>, Vec<f64>, usize) {
        // mesh of t's can be defined directly or by size of step -h, and number of steps
        if let Some(mesh) = mesh {
            let n_steps = mesh.len();
            info!("mesh with n_steps = {} is defined directly", n_steps);
            let H: Vec<Expr> = mesh
                .windows(2)
                .map(|window| Expr::Const(window[1] - window[0]))
                .collect();
            (H, mesh, n_steps)
        } else {
            let n_steps = n_steps.unwrap_or(100) + 1;
            info!(
                "mesh is not defined, creating evenly distributed mesh of length {}",
                n_steps
            );
            let h = h.unwrap_or(1.0);
            let H: Vec<Expr> = vec![Expr::Const(h); n_steps - 1]; // number of intervals = n_steps -1
            let T_list: Vec<f64> = (0..n_steps).map(|i| t0 + (i as f64) * h).collect();
            (H, T_list, n_steps)
        }
    }

    /// Processes variable bounds and tolerances for discretized variables with optimized string operations.
    ///
    /// Efficiently maps bounds and tolerances from original variable names to all discretized
    /// variable instances. Uses fast string processing to extract base names from discretized variables.
    ///
    /// # Arguments
    /// * `Bounds` - Optional bounds map: {"y": (min, max), "z": (min, max)}
    /// * `rel_tolerance` - Optional tolerance map: {"y": tol, "z": tol}
    /// * `flat_list_of_names` - All discretized variable names ["y_0", "y_1", "z_0", "z_1", ...]
    ///
    /// # Performance Optimizations
    /// - **Fast string processing**: Uses `rfind('_')` instead of regex (10-100x faster)
    /// - **Pre-allocation**: Vectors allocated with known capacity
    /// - **Single pass**: Processes all variables in one iteration
    ///
    /// # Output
    /// Sets `self.bounds` and `self.rel_tolerance_vec` with per-variable values:
    /// - bounds[i] = (min, max) for discretized variable i
    /// - rel_tolerance_vec[i] = tolerance for discretized variable i
    ///
    /// # Example
    /// ```ignore
    /// // Input: Bounds = {"y": (0.0, 10.0)}, flat_list = ["y_0", "y_1", "z_0"]
    /// // Output: bounds = [(0.0, 10.0), (0.0, 10.0), default_for_z]
    /// ```
    fn process_bounds_and_tolerances(
        &mut self,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        flat_list_of_names: Vec<String>,
    ) {
        let len = flat_list_of_names.len();

        self.bounds = Bounds.as_ref().map(|bounds_map| {
            let mut vec_of_bounds = Vec::with_capacity(len);
            for name in &flat_list_of_names {
                // Fast string processing: find last underscore and extract base name
                let base_name = if let Some(pos) = name.rfind('_') {
                    &name[..pos]
                } else {
                    name
                };

                if let Some(&bound_pair) = bounds_map.get(base_name) {
                    vec_of_bounds.push(bound_pair);
                }
            }
            vec_of_bounds
        });

        self.rel_tolerance_vec = rel_tolerance.as_ref().map(|tolerance_map| {
            let mut vec_of_tolerance = Vec::with_capacity(len);
            for name in &flat_list_of_names {
                // Fast string processing: find last underscore and extract base name
                let base_name = if let Some(pos) = name.rfind('_') {
                    &name[..pos]
                } else {
                    name
                };

                if let Some(&tolerance) = tolerance_map.get(base_name) {
                    vec_of_tolerance.push(tolerance);
                }
            }
            vec_of_tolerance
        });
    }

    /// High-performance parallel BVP discretization with comprehensive optimization.
    ///
    /// Converts a continuous BVP system into a discretized algebraic system ready for Newton-Raphson
    /// solving. This is the core discretization method with extensive performance optimizations.
    ///
    /// # Arguments
    /// * `eq_system` - Vector of symbolic ODE equations [dy/dt = f1(t,y,z), dz/dt = f2(t,y,z)]
    /// * `values` - Original variable names ["y", "z"]
    /// * `arg` - Independent variable name ("t" for time, "x" for space)
    /// * `t0` - Starting point of the domain
    /// * `n_steps` - Number of discretization steps (creates n_steps+1 points)
    /// * `h` - Step size for uniform mesh
    /// * `mesh` - Explicit mesh points (overrides n_steps/h)
    /// * `BorderConditions` - Boundary conditions: {"y": [(0, value), (1, value)]}
    ///   - 0 = initial condition, 1 = final condition
    /// * `Bounds` - Variable bounds for constrained solving
    /// * `rel_tolerance` - Per-variable relative tolerances
    /// * `scheme` - Discretization scheme ("forward" or "trapezoid")
    ///
    /// # Performance Optimizations
    /// 1. **Parallel discretization**: Uses rayon for concurrent equation processing
    /// 2. **Boundary condition caching**: Pre-computes HashMaps for O(1) BC lookups
    /// 3. **Variable tracking**: Tracks which variables appear in each equation for smart Jacobian
    /// 4. **Memory pre-allocation**: Pre-allocates vectors with known capacity
    /// 5. **Efficient BC application**: Parallel boundary condition substitution
    /// 6. **Fast string processing**: Optimized variable name processing
    ///
    /// # Output
    /// Creates discretized system: y_{i+1} - y_i - h*f(t_i, y_i, z_i) = 0
    /// Sets up all internal data structures for subsequent Jacobian computation.
    ///
    /// # Timing
    /// Provides detailed timing breakdown of each optimization phase.
    pub fn discretization_system_BVP_par(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
    ) {
        let total_start = Instant::now();
        let mut timer_hash: HashMap<String, f64> = HashMap::new();
        let (H, T_list, n_steps) = self.create_mesh(n_steps, h, mesh, t0);
        let (matrix_of_expr, matrix_of_names) = Expr::IndexedVarsMatrix(n_steps, values.clone());
        let bc_handling = Instant::now();
        // Pre-compute boundary conditions lookup for O(1) access
        let bc_lookup: HashMap<String, HashMap<usize, f64>> = BorderConditions
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect();

        // Pre-compute variables to exclude from boundary conditions
        let mut vars_for_boundary_conditions = HashMap::new();
        let mut vars_to_exclude = HashSet::new();
        let mut bc_pos_n_values = Vec::new();

        for (var_name, conditions) in &bc_lookup {
            if let Some(var_idx) = values.iter().position(|v| v == var_name) {
                for (&pos, &value) in conditions {
                    match pos {
                        0 => {
                            // Initial condition
                            let var_name = &matrix_of_names[0][var_idx];
                            vars_for_boundary_conditions.insert(var_name.clone(), value);
                            vars_to_exclude.insert(var_name.clone());
                            let full_pos = 0 * values.len() + var_idx;
                            bc_pos_n_values.push((full_pos, 0 as usize, value));
                        }
                        1 => {
                            // Final condition
                            let var_name = &matrix_of_names[n_steps - 1][var_idx];
                            vars_for_boundary_conditions.insert(var_name.clone(), value);
                            vars_to_exclude.insert(var_name.clone());
                            let full_pos = (n_steps - 1) * values.len() + var_idx;
                            bc_pos_n_values.push((full_pos, 1 as usize, value));
                        }
                        _ => {}
                    }
                }
            }
        }
        timer_hash.insert(
            "bc handling".to_string(),
            bc_handling.elapsed().as_millis() as f64,
        );
        // DISCRETAZING EQUATIONS
        println!("creating discretization equations");
        let discretization_start = Instant::now();

        // Optimized discretization with variable tracking
        let (discreditized_system, variables_for_all_discrete): (Vec<Expr>, Vec<Vec<String>>) = (0
            ..n_steps - 1)
            .into_par_iter()
            .flat_map(|j| {
                let t = T_list[j];
                eq_system
                    .par_iter()
                    .enumerate()
                    .map(|(i, eq_i)| {
                        let eq_step_j =
                            Self::eq_step(&matrix_of_names, eq_i, &values, &arg, j, t, &scheme);
                        let Y_j_plus_1 = &matrix_of_expr[j + 1][i];
                        let Y_j = &matrix_of_expr[j][i];
                        let res_ij = Y_j_plus_1.clone() - Y_j.clone() - H[j].clone() * eq_step_j;

                        // Track variables used in this equation (excluding boundary condition vars)
                        let mut vars_in_equation = Vec::new();
                        let y_j_plus_1_name = &matrix_of_names[j + 1][i];
                        let y_j_name = &matrix_of_names[j][i];

                        if !vars_to_exclude.contains(y_j_plus_1_name) {
                            vars_in_equation.push(y_j_plus_1_name.clone());
                        }
                        if !vars_to_exclude.contains(y_j_name) {
                            vars_in_equation.push(y_j_name.clone());
                        }

                        // Add variables from eq_step_j (from original equation at this time step)
                        for var_idx in 0..values.len() {
                            let var_name = &matrix_of_names[j][var_idx];
                            if !vars_to_exclude.contains(var_name) {
                                vars_in_equation.push(var_name.clone());
                            }
                        }

                        (res_ij.simplify(), vars_in_equation)
                    })
                    .collect::<Vec<_>>()
            })
            .unzip();

        self.variables_for_all_disrete = variables_for_all_discrete;
        timer_hash.insert(
            "discretization of equations".to_string(),
            discretization_start.elapsed().as_millis() as f64,
        );
        let start_flat_list = Instant::now();

        // Efficient flat list creation with pre-allocation
        let total_vars = values.len() * n_steps;
        let mut flat_list_of_names = Vec::with_capacity(total_vars);
        let mut flat_list_of_expr = Vec::with_capacity(total_vars);
        for time_idx in 0..n_steps {
            for var_idx in 0..values.len() {
                let name = &matrix_of_names[time_idx][var_idx];
                if !vars_to_exclude.contains(name) {
                    flat_list_of_names.push(name.clone());
                    flat_list_of_expr.push(matrix_of_expr[time_idx][var_idx].clone());
                }
            }
        }

        timer_hash.insert(
            "flat list creation".to_string(),
            start_flat_list.elapsed().as_millis() as f64,
        );
        let BC_application_start = Instant::now();
        // Apply boundary conditions to discretized system in parallel
        let discreditized_system_with_BC: Vec<Expr> = discreditized_system
            .into_par_iter()
            .map(|mut eq_i| {
                for (var_name, &value) in &vars_for_boundary_conditions {
                    eq_i = eq_i.set_variable(var_name, value);
                }
                eq_i.simplify()
            })
            .collect();

        let discreditized_system_flat = discreditized_system_with_BC;
        timer_hash.insert(
            "BC application".to_string(),
            BC_application_start.elapsed().as_millis() as f64,
        );
        let consistency_start = Instant::now();
        // Simplified consistency test using pre-computed variables
        let hashset_of_vars: HashSet<&String> = flat_list_of_names.iter().collect();
        let mut missing_vars = Vec::new();

        for var_list in &self.variables_for_all_disrete {
            for var in var_list {
                if !hashset_of_vars.contains(var) {
                    missing_vars.push(var.clone());
                }
            }
        }

        if !missing_vars.is_empty() {
            missing_vars.sort_unstable();
            missing_vars.dedup();
            panic!("Variables not found in system: {:?}", missing_vars);
        }
        timer_hash.insert(
            "consistency test".to_string(),
            consistency_start.elapsed().as_millis() as f64,
        );

        self.vector_of_functions = discreditized_system_flat;
        self.vector_of_variables = flat_list_of_expr;
        self.variable_string = flat_list_of_names.clone();
        self.BC_pos_n_values = bc_pos_n_values;
        let bounds_and_tolerances_start = Instant::now();
        self.process_bounds_and_tolerances(Bounds, rel_tolerance, flat_list_of_names);
        timer_hash.insert(
            "bounds and tolerances".to_string(),
            bounds_and_tolerances_start.elapsed().as_millis() as f64,
        );

        // timing
        let total_end = total_start.elapsed().as_millis() as f64;
        normalize_timer_percent(timer_hash.get_mut("bc handling").unwrap(), total_end);
        normalize_timer_percent(
            timer_hash.get_mut("discretization of equations").unwrap(),
            total_end,
        );
        normalize_timer_percent(timer_hash.get_mut("BC application").unwrap(), total_end);
        normalize_timer_percent(timer_hash.get_mut("flat list creation").unwrap(), total_end);
        normalize_timer_percent(timer_hash.get_mut("consistency test").unwrap(), total_end);
        normalize_timer_percent(
            timer_hash.get_mut("bounds and tolerances").unwrap(),
            total_end,
        );
        timer_hash.insert("total time, sec".to_string(), total_end);

        let mut table = Builder::from(timer_hash.clone()).build();
        table.with(Style::modern_rounded());
        println!("{}", table.to_string());
    }

    /// Automatically detects the bandwidth of the sparse Jacobian matrix using parallel computation.
    ///
    /// Analyzes the symbolic Jacobian to determine the banded structure, computing the number
    /// of sub- and super-diagonals. This information is crucial for banded matrix optimizations.
    ///
    /// # Algorithm
    /// For each non-zero element at position (i,j):
    /// - If j > i: contributes to super-diagonal width (ku)
    /// - If i > j: contributes to sub-diagonal width (kl)
    ///
    /// # Performance Features
    /// - **Parallel computation**: Uses rayon to process rows concurrently
    /// - **Reduction pattern**: Efficiently combines results from parallel workers
    /// - **Early termination**: Skips zero elements for efficiency
    ///
    /// # Output
    /// Sets `self.bandwidth = Some((kl, ku))` where:
    /// - kl = maximum number of sub-diagonals
    /// - ku = maximum number of super-diagonals
    ///
    /// # When to Use
    /// - Automatically called by `generate_BVP()` if bandwidth not provided
    /// - Essential for banded matrix solvers and storage optimization
    /// - Particularly important for large sparse systems
    fn find_bandwidths(&mut self) {
        let A = &self.symbolic_jacobian;
        let n = A.len();
        // kl  Number of subdiagonals
        // ku = 0; Number of superdiagonals

        /*
            Matrix Iteration: The function find_bandwidths iterates through each element of the matrix A.
        Subdiagonal Width (kl): For each non-zero element below the main diagonal (i.e., i > j), it calculates the distance from the diagonal and updates
        kl if this distance is greater than the current value of kl.
        Superdiagonal Width (ku): Similarly, for each non-zero element above the main diagonal (i.e., j > i), it calculates the distance from the diagonal
         and updates ku if this distance is greater than the current value of ku.
             */
        let (kl, ku) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row_kl = 0;
                let mut row_ku = 0;
                for j in 0..n {
                    if A[i][j] != Expr::Const(0.0) {
                        if j > i {
                            row_ku = std::cmp::max(row_ku, j - i);
                        } else if i > j {
                            row_kl = std::cmp::max(row_kl, i - j);
                        }
                    }
                }
                (row_kl, row_ku)
            })
            .reduce(
                || (0, 0),
                |acc, row| (std::cmp::max(acc.0, row.0), std::cmp::max(acc.1, row.1)),
            );

        self.bandwidth = Some((kl, ku));
    }

    /// **MAIN FUNCTION**: Complete BVP symbolic-to-numerical transformation pipeline.
    ///
    /// This is the primary entry point for BVP solving. It orchestrates the entire process from
    /// symbolic ODE system to high-performance numerical functions ready for Newton-Raphson solving.
    ///
    /// # Complete Workflow
    /// 1. **Discretization**: Converts continuous BVP to algebraic system
    /// 2. **Symbolic Jacobian**: Computes analytical partial derivatives
    /// 3. **Bandwidth Detection**: Analyzes sparsity structure
    /// 4. **Function Compilation**: Creates optimized numerical evaluators
    /// 5. **Performance Timing**: Provides detailed timing breakdown
    ///
    /// # Arguments
    /// * `eq_system` - Vector of symbolic ODE equations
    /// * `values` - Original variable names ["y", "z"]
    /// * `arg` - Independent variable name ("t" or "x")
    /// * `t0` - Domain starting point
    /// * `param` - Parameter name (defaults to `arg` if None)
    /// * `n_steps` - Number of discretization steps
    /// * `h` - Step size for uniform mesh
    /// * `mesh` - Explicit mesh points
    /// * `BorderConditions` - Boundary conditions specification
    /// * `Bounds` - Variable bounds for constrained solving
    /// * `rel_tolerance` - Per-variable relative tolerances
    /// * `scheme` - Discretization scheme ("forward" or "trapezoid")
    /// * `method` - Matrix backend ("Dense", "Sparse", "Sparse_1", "Sparse_2")
    /// * `bandwidth` - Optional pre-computed bandwidth (auto-detected if None)
    ///
    /// # Supported Matrix Backends
    /// - **"Dense"**: nalgebra DMatrix - best for small, dense systems
    /// - **"Sparse"**: faer SparseColMat - recommended for large sparse systems
    /// - **"Sparse_1"**: sprs CsMat - legacy sparse support
    /// - **"Sparse_2"**: nalgebra CsMatrix - nalgebra ecosystem integration
    ///
    /// # Performance Features
    /// - Comprehensive timing analysis with percentage breakdown
    /// - Automatic bandwidth detection for sparse optimization
    /// - Smart sparsity-aware Jacobian computation
    /// - Multi-backend support for optimal performance
    ///
    /// # Output
    /// Sets up `self.jac_function` and `self.residiual_function` ready for BVP solving.
    /// Prints detailed timing table showing performance breakdown.
    ///
    /// # Example Usage
    /// ```ignore
    /// let mut jacobian = Jacobian::new();
    /// jacobian.generate_BVP(
    ///     equations, variables, "t", 0.0, None, Some(100), None, None,
    ///     boundary_conditions, bounds, tolerances, "trapezoid".to_string(),
    ///     "Sparse".to_string(), None
    /// );
    /// ```
    pub fn generate_BVP(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
    ) {
        self.generate_BVP_with_params(
            eq_system,
            values,
            arg,
            None,
            t0,
            param,
            n_steps,
            h,
            mesh,
            BorderConditions,
            Bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
        );
    }

    /// Same as `generate_BVP`, but with explicit symbolic parameter names that
    /// affect evaluation without becoming Newton unknowns.
    pub fn generate_BVP_with_params(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
    ) {
        let (arg_name, indexed_values_owned, matrix_backend, mut timer_hash, total_start) = self
            .prepare_symbolic_bvp_stage_with_params(
                eq_system,
                values,
                arg,
                params,
                t0,
                n_steps,
                h,
                mesh,
                BorderConditions,
                Bounds,
                rel_tolerance,
                scheme,
                method,
                bandwidth,
            );
        let param = param.unwrap_or(arg_name);
        let indexed_values: Vec<&str> = indexed_values_owned.iter().map(|x| x.as_str()).collect();

        let now = Instant::now();
        self.compile_jacobian_for_matrix_backend(
            param.as_str(),
            indexed_values.clone(),
            matrix_backend,
        );
        timer_hash.insert(
            "jacobian lambdify time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        info!("Jacobian lambdified");
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            assert_eq!(vec_s.len(), *n, "jacobian not square ");
        }

        let now = Instant::now();
        self.compile_residual_for_matrix_backend(param.as_str(), indexed_values, matrix_backend);
        timer_hash.insert(
            "residual functions lambdify time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        info!("Residuals vector lambdified");

        let total_end = total_start.elapsed().as_secs_f64();
        self.finalize_generate_bvp_timer_table(timer_hash, total_end);
    }

    /// Same symbolic/discretization path as `generate_BVP_with_params`, but the
    /// numerical backend branch is selected through the shared lifecycle layer.
    ///
    /// This is a thin staged integration entrypoint:
    /// - the legacy faer/lambdify path remains the main execution path,
    /// - AOT selection metadata is returned when an artifact is registered or compiled,
    /// - the caller can decide whether to continue with lambdify fallback or a
    ///   compiled AOT branch above this module.
    pub fn generate_BVP_with_backend_selection(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
    ) -> BvpSparseExecutionPlan {
        let (arg_name, indexed_values_owned, _matrix_backend, mut timer_hash, total_start) = self
            .prepare_symbolic_bvp_stage_with_params(
                eq_system,
                values,
                arg,
                params,
                t0,
                n_steps,
                h,
                mesh,
                BorderConditions,
                Bounds,
                rel_tolerance,
                scheme,
                method,
                bandwidth,
            );
        let effective_arg_name = param.unwrap_or(arg_name);
        let residual_strategy =
            recommended_residual_chunking_for_parallelism(self.vector_of_functions.len(), 4);
        let jacobian_strategy =
            recommended_row_chunking_for_parallelism(self.vector_of_functions.len(), 4);

        let execution = self.prepare_sparse_backend_execution_timed(
            effective_arg_name.as_str(),
            indexed_values_owned
                .iter()
                .map(|name| name.as_str())
                .collect(),
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            residual_strategy,
            jacobian_strategy,
            policy,
            resolver,
            &mut timer_hash,
        );
        let total_end = total_start.elapsed().as_secs_f64();
        self.finalize_generate_bvp_timer_table(timer_hash, total_end);
        execution
    }

    /// Same as `generate_BVP_with_backend_selection`, but with explicit
    /// residual and sparse-Jacobian chunking strategies supplied by the outer
    /// solver layer.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_BVP_with_backend_selection_and_chunking(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        params: Option<&[&str]>,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
        policy: BackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        residual_strategy: ResidualChunkingStrategy,
        jacobian_strategy: SparseChunkingStrategy,
    ) -> BvpSparseExecutionPlan {
        let (arg_name, indexed_values_owned, _matrix_backend, mut timer_hash, total_start) = self
            .prepare_symbolic_bvp_stage_with_params(
                eq_system,
                values,
                arg,
                params,
                t0,
                n_steps,
                h,
                mesh,
                BorderConditions,
                Bounds,
                rel_tolerance,
                scheme,
                method,
                bandwidth,
            );
        let effective_arg_name = param.unwrap_or(arg_name);

        let execution = self.prepare_sparse_backend_execution_timed(
            effective_arg_name.as_str(),
            indexed_values_owned
                .iter()
                .map(|name| name.as_str())
                .collect(),
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            residual_strategy,
            jacobian_strategy,
            policy,
            resolver,
            &mut timer_hash,
        );
        let total_end = total_start.elapsed().as_secs_f64();
        self.finalize_generate_bvp_timer_table(timer_hash, total_end);
        execution
    }
} // end of impl
