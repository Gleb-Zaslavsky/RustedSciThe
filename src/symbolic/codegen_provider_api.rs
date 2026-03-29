//! Solver-facing provider traits and backend selection metadata for AOT.
//!
//! The current codegen stack already knows how to:
//! - describe symbolic tasks,
//! - turn them into runtime plans,
//! - lower those plans into generated code,
//! - and execute generated chunks through orchestrators.
//!
//! This module introduces the next lifecycle layer above those pieces:
//! a stable provider-oriented contract that Newton/BVP-style solvers can
//! consume without caring whether the underlying implementation came from:
//! - plain numeric closures,
//! - lambdified symbolic expressions,
//! - or generated AOT code.
//!
//! The goal is to keep solver integration simple:
//! - solvers ask for residuals and Jacobians,
//! - backend selection stays outside solver code,
//! - matrix storage selection is explicit instead of being encoded in ad-hoc
//!   booleans or path-specific conventions.

use crate::symbolic::codegen_runtime_api::{
    DenseJacobianRuntimePlan, ResidualRuntimePlan, SparseJacobianRuntimePlan,
    SparseJacobianStructure,
};

/// High-level evaluation backend selected for a prepared problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Direct numeric closures without symbolic lambdify/codegen.
    Numeric,
    /// Symbolic expressions evaluated through the existing lambdify path.
    Lambdify,
    /// Ahead-of-time generated residual/Jacobian code.
    Aot,
}

impl BackendKind {
    /// Stable string form suitable for manifests and generated metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Numeric => "numeric",
            Self::Lambdify => "lambdify",
            Self::Aot => "aot",
        }
    }
}

/// Matrix/value backend exposed by the prepared problem.
///
/// This stays separate from [`BackendKind`]:
/// - `BackendKind` answers how residual/Jacobian code was produced;
/// - `MatrixBackend` answers what matrix form the caller expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixBackend {
    /// Dense row-major Jacobian values or dense matrix assembly.
    Dense,
    /// Sparse Jacobian values intended for `faer::sparse::SparseColMat`.
    SparseCol,
    /// Sparse Jacobian values intended for `sprs::CsMat`.
    CsMat,
    /// Sparse Jacobian values intended for another CSC/CSR-like backend.
    CsMatrix,
    /// Values-only sparse path with structure supplied separately.
    ValuesOnly,
}

impl MatrixBackend {
    /// Stable string form suitable for manifests and generated metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::SparseCol => "sparse_col",
            Self::CsMat => "cs_mat",
            Self::CsMatrix => "cs_matrix",
            Self::ValuesOnly => "values_only",
        }
    }
}

/// Solver-facing provider trait for dense residual/Jacobian problems.
pub trait DenseProblemProvider {
    /// Writes the residual vector into `out`.
    fn residual_into(&self, args: &[f64], out: &mut [f64]);

    /// Writes the dense Jacobian in row-major order into `out`.
    fn jacobian_into(&self, args: &[f64], out: &mut [f64]);

    /// Returns the residual length.
    fn residual_len(&self) -> usize;

    /// Returns the dense Jacobian shape `(rows, cols)`.
    fn jacobian_shape(&self) -> (usize, usize);
}

/// Solver-facing provider trait for sparse residual/Jacobian problems.
pub trait SparseProblemProvider {
    /// Writes the residual vector into `out`.
    fn residual_into(&self, args: &[f64], out: &mut [f64]);

    /// Writes sparse Jacobian values in the explicit global entry order into
    /// `values_out`.
    fn jacobian_values_into(&self, args: &[f64], values_out: &mut [f64]);

    /// Returns the residual length.
    fn residual_len(&self) -> usize;

    /// Returns the sparse Jacobian shape `(rows, cols)`.
    fn jacobian_shape(&self) -> (usize, usize);

    /// Returns the sparse Jacobian structure expected by
    /// [`Self::jacobian_values_into`].
    fn jacobian_structure(&self) -> &SparseJacobianStructure;
}

/// Prepared dense problem specification built by a higher layer such as
/// symbolic `Jacobian` orchestration or future AOT manifests.
#[derive(Debug, Clone)]
pub struct PreparedDenseProblem<'a> {
    pub backend_kind: BackendKind,
    pub matrix_backend: MatrixBackend,
    pub residual_plan: ResidualRuntimePlan<'a>,
    pub jacobian_plan: DenseJacobianRuntimePlan<'a>,
}

impl<'a> PreparedDenseProblem<'a> {
    /// Creates a prepared dense problem from already-built runtime plans.
    pub fn new(
        backend_kind: BackendKind,
        matrix_backend: MatrixBackend,
        residual_plan: ResidualRuntimePlan<'a>,
        jacobian_plan: DenseJacobianRuntimePlan<'a>,
    ) -> Self {
        assert_eq!(
            residual_plan.input_names, jacobian_plan.input_names,
            "dense residual and Jacobian plans must share the same flattened input order"
        );

        Self {
            backend_kind,
            matrix_backend,
            residual_plan,
            jacobian_plan,
        }
    }

    /// Returns the residual length.
    pub fn residual_len(&self) -> usize {
        self.residual_plan.output_len
    }

    /// Returns the dense Jacobian shape `(rows, cols)`.
    pub fn jacobian_shape(&self) -> (usize, usize) {
        (self.jacobian_plan.rows, self.jacobian_plan.cols)
    }

    /// Returns the flattened input names shared by residual and Jacobian.
    pub fn input_names(&self) -> &[&'a str] {
        &self.residual_plan.input_names
    }
}

/// Prepared sparse problem specification built by a higher layer such as
/// symbolic `Jacobian` orchestration or future AOT manifests.
#[derive(Debug, Clone)]
pub struct PreparedSparseProblem<'a> {
    pub backend_kind: BackendKind,
    pub matrix_backend: MatrixBackend,
    pub residual_plan: ResidualRuntimePlan<'a>,
    pub jacobian_plan: SparseJacobianRuntimePlan<'a>,
}

impl<'a> PreparedSparseProblem<'a> {
    /// Creates a prepared sparse problem from already-built runtime plans.
    pub fn new(
        backend_kind: BackendKind,
        matrix_backend: MatrixBackend,
        residual_plan: ResidualRuntimePlan<'a>,
        jacobian_plan: SparseJacobianRuntimePlan<'a>,
    ) -> Self {
        assert_eq!(
            residual_plan.input_names, jacobian_plan.input_names,
            "sparse residual and Jacobian plans must share the same flattened input order"
        );

        Self {
            backend_kind,
            matrix_backend,
            residual_plan,
            jacobian_plan,
        }
    }

    /// Returns the residual length.
    pub fn residual_len(&self) -> usize {
        self.residual_plan.output_len
    }

    /// Returns the sparse Jacobian shape `(rows, cols)`.
    pub fn jacobian_shape(&self) -> (usize, usize) {
        (
            self.jacobian_plan.structure.rows,
            self.jacobian_plan.structure.cols,
        )
    }

    /// Returns the sparse Jacobian structure.
    pub fn jacobian_structure(&self) -> &SparseJacobianStructure {
        &self.jacobian_plan.structure
    }

    /// Returns the flattened input names shared by residual and Jacobian.
    pub fn input_names(&self) -> &[&'a str] {
        &self.residual_plan.input_names
    }
}

/// Unified prepared problem description used by future lifecycle layers.
#[derive(Debug, Clone)]
pub enum PreparedProblem<'a> {
    Dense(PreparedDenseProblem<'a>),
    Sparse(PreparedSparseProblem<'a>),
}

impl<'a> PreparedProblem<'a> {
    /// Wraps a dense prepared problem.
    pub fn dense(problem: PreparedDenseProblem<'a>) -> Self {
        Self::Dense(problem)
    }

    /// Wraps a sparse prepared problem.
    pub fn sparse(problem: PreparedSparseProblem<'a>) -> Self {
        Self::Sparse(problem)
    }

    /// Returns the selected code-generation/evaluation backend.
    pub fn backend_kind(&self) -> BackendKind {
        match self {
            Self::Dense(problem) => problem.backend_kind,
            Self::Sparse(problem) => problem.backend_kind,
        }
    }

    /// Returns the selected matrix/value backend.
    pub fn matrix_backend(&self) -> MatrixBackend {
        match self {
            Self::Dense(problem) => problem.matrix_backend,
            Self::Sparse(problem) => problem.matrix_backend,
        }
    }

    /// Returns the residual length.
    pub fn residual_len(&self) -> usize {
        match self {
            Self::Dense(problem) => problem.residual_len(),
            Self::Sparse(problem) => problem.residual_len(),
        }
    }

    /// Returns the Jacobian shape `(rows, cols)`.
    pub fn jacobian_shape(&self) -> (usize, usize) {
        match self {
            Self::Dense(problem) => problem.jacobian_shape(),
            Self::Sparse(problem) => problem.jacobian_shape(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen_tasks::{
        JacobianTask, ResidualTask, SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn prepared_dense_problem_reports_shape_and_backend() {
        let residuals = vec![
            Expr::parse_expression("x + y"),
            Expr::parse_expression("x - y"),
        ];
        let jacobian = vec![
            vec![Expr::parse_expression("1"), Expr::parse_expression("1")],
            vec![Expr::parse_expression("1"), Expr::parse_expression("-1")],
        ];
        let vars = vec!["x", "y"];

        let residual_task = ResidualTask {
            fn_name: "eval_dense_residual",
            residuals: &residuals,
            variables: &vars,
            params: None,
        };
        let jacobian_task = JacobianTask {
            fn_name: "eval_dense_jacobian",
            jacobian: &jacobian,
            variables: &vars,
            params: None,
        };

        let prepared = PreparedDenseProblem {
            backend_kind: BackendKind::Aot,
            matrix_backend: MatrixBackend::Dense,
            residual_plan: residual_task.runtime_plan(ResidualChunkingStrategy::Whole),
            jacobian_plan: jacobian_task.runtime_plan(DenseJacobianChunkingStrategy::Whole),
        };

        assert_eq!(prepared.residual_len(), 2);
        assert_eq!(prepared.jacobian_shape(), (2, 2));
        assert_eq!(prepared.input_names(), &["x", "y"]);
        assert_eq!(prepared.backend_kind, BackendKind::Aot);
        assert_eq!(prepared.matrix_backend, MatrixBackend::Dense);
    }

    #[test]
    fn prepared_sparse_problem_exposes_structure_and_shape() {
        let residuals = vec![
            Expr::parse_expression("x + p"),
            Expr::parse_expression("y - p"),
        ];
        let entry0 = Expr::parse_expression("1");
        let entry1 = Expr::parse_expression("2");
        let vars = vec!["x", "y"];
        let params = vec!["p"];
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &entry0,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &entry1,
            },
        ];

        let residual_task = ResidualTask {
            fn_name: "eval_sparse_residual",
            residuals: &residuals,
            variables: &vars,
            params: Some(&params),
        };
        let sparse_task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (2, 2),
            entries: &entries,
            variables: &vars,
            params: Some(&params),
        };

        let prepared = PreparedSparseProblem::new(
            BackendKind::Lambdify,
            MatrixBackend::SparseCol,
            residual_task.runtime_plan(ResidualChunkingStrategy::Whole),
            sparse_task.runtime_plan(SparseChunkingStrategy::Whole),
        );

        assert_eq!(prepared.residual_len(), 2);
        assert_eq!(prepared.jacobian_shape(), (2, 2));
        assert_eq!(prepared.input_names(), &["p", "x", "y"]);
        assert_eq!(prepared.jacobian_structure().nnz(), 2);
        assert_eq!(prepared.jacobian_structure().row_indices, vec![0, 1]);
        assert_eq!(prepared.jacobian_structure().col_indices, vec![0, 1]);
    }

    #[test]
    fn prepared_problem_preserves_backend_metadata() {
        let residuals = vec![Expr::parse_expression("x")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];

        let dense = PreparedProblem::dense(PreparedDenseProblem::new(
            BackendKind::Numeric,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: None,
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            JacobianTask {
                fn_name: "eval_jacobian",
                jacobian: &jacobian,
                variables: &vars,
                params: None,
            }
            .runtime_plan(DenseJacobianChunkingStrategy::Whole),
        ));

        assert_eq!(dense.backend_kind(), BackendKind::Numeric);
        assert_eq!(dense.matrix_backend(), MatrixBackend::Dense);
        assert_eq!(dense.residual_len(), 1);
        assert_eq!(dense.jacobian_shape(), (1, 1));
    }

    #[test]
    #[should_panic(expected = "must share the same flattened input order")]
    fn prepared_dense_problem_rejects_mismatched_input_order() {
        let residuals = vec![Expr::parse_expression("p + x")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];
        let params = vec!["p"];

        let residual_plan = ResidualTask {
            fn_name: "eval_residual",
            residuals: &residuals,
            variables: &vars,
            params: Some(&params),
        }
        .runtime_plan(ResidualChunkingStrategy::Whole);

        let mut jacobian_plan = JacobianTask {
            fn_name: "eval_jacobian",
            jacobian: &jacobian,
            variables: &vars,
            params: Some(&params),
        }
        .runtime_plan(DenseJacobianChunkingStrategy::Whole);
        jacobian_plan.input_names = vec!["x", "p"];

        let _ = PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            residual_plan,
            jacobian_plan,
        );
    }
}
