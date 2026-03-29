//! Owned metadata manifests for prepared code-generated problems.
//!
//! Unlike runtime plans, manifests are meant to be easy to store, compare,
//! hash, or pass into a future generated-crate writer. They intentionally own
//! their strings and basic scalar metadata so they are independent from the
//! borrowed symbolic task inputs that produced them.

use crate::symbolic::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedDenseProblem, PreparedProblem, PreparedSparseProblem,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Owned input/output metadata shared by future AOT manifests.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProblemIoManifest {
    pub input_names: Vec<String>,
    pub residual_len: usize,
    pub jacobian_rows: usize,
    pub jacobian_cols: usize,
    pub jacobian_nnz: Option<usize>,
}

/// Generated function naming metadata extracted from runtime plans.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeneratedFunctionsManifest {
    pub residual_fn_name: String,
    pub residual_chunk_names: Vec<String>,
    pub jacobian_fn_name: String,
    pub jacobian_chunk_names: Vec<String>,
}

/// Compact owned manifest for a prepared dense or sparse problem.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedProblemManifest {
    pub backend_kind: BackendKind,
    pub matrix_backend: MatrixBackend,
    pub io: ProblemIoManifest,
    pub functions: GeneratedFunctionsManifest,
}

impl PreparedProblemManifest {
    /// Returns a stable hash-like key for registry/index lookup of one
    /// prepared problem shape plus generated function layout.
    pub fn problem_key(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

impl<'a> From<&PreparedDenseProblem<'a>> for PreparedProblemManifest {
    fn from(problem: &PreparedDenseProblem<'a>) -> Self {
        Self {
            backend_kind: problem.backend_kind,
            matrix_backend: problem.matrix_backend,
            io: ProblemIoManifest {
                input_names: problem
                    .input_names()
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
                residual_len: problem.residual_len(),
                jacobian_rows: problem.jacobian_plan.rows,
                jacobian_cols: problem.jacobian_plan.cols,
                jacobian_nnz: None,
            },
            functions: GeneratedFunctionsManifest {
                residual_fn_name: problem.residual_plan.fn_name.to_string(),
                residual_chunk_names: problem
                    .residual_plan
                    .chunks
                    .iter()
                    .map(|chunk| chunk.fn_name.clone())
                    .collect(),
                jacobian_fn_name: problem.jacobian_plan.fn_name.to_string(),
                jacobian_chunk_names: problem
                    .jacobian_plan
                    .chunks
                    .iter()
                    .map(|chunk| chunk.fn_name.clone())
                    .collect(),
            },
        }
    }
}

impl<'a> From<&PreparedSparseProblem<'a>> for PreparedProblemManifest {
    fn from(problem: &PreparedSparseProblem<'a>) -> Self {
        Self {
            backend_kind: problem.backend_kind,
            matrix_backend: problem.matrix_backend,
            io: ProblemIoManifest {
                input_names: problem
                    .input_names()
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
                residual_len: problem.residual_len(),
                jacobian_rows: problem.jacobian_plan.structure.rows,
                jacobian_cols: problem.jacobian_plan.structure.cols,
                jacobian_nnz: Some(problem.jacobian_plan.structure.nnz()),
            },
            functions: GeneratedFunctionsManifest {
                residual_fn_name: problem.residual_plan.fn_name.to_string(),
                residual_chunk_names: problem
                    .residual_plan
                    .chunks
                    .iter()
                    .map(|chunk| chunk.fn_name.clone())
                    .collect(),
                jacobian_fn_name: problem.jacobian_plan.fn_name.to_string(),
                jacobian_chunk_names: problem
                    .jacobian_plan
                    .chunks
                    .iter()
                    .map(|chunk| chunk.fn_name.clone())
                    .collect(),
            },
        }
    }
}

impl<'a> From<&PreparedProblem<'a>> for PreparedProblemManifest {
    fn from(problem: &PreparedProblem<'a>) -> Self {
        match problem {
            PreparedProblem::Dense(problem) => Self::from(problem),
            PreparedProblem::Sparse(problem) => Self::from(problem),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_provider_api::{PreparedDenseProblem, PreparedSparseProblem};
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen_tasks::{
        JacobianTask, ResidualTask, SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn dense_manifest_captures_owned_function_and_io_metadata() {
        let residuals = vec![Expr::parse_expression("x + y")];
        let jacobian = vec![vec![
            Expr::parse_expression("1"),
            Expr::parse_expression("1"),
        ]];
        let vars = vec!["x", "y"];

        let prepared = PreparedDenseProblem::new(
            BackendKind::Aot,
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
        );

        let manifest = PreparedProblemManifest::from(&prepared);

        assert_eq!(manifest.backend_kind, BackendKind::Aot);
        assert_eq!(manifest.matrix_backend, MatrixBackend::Dense);
        assert_eq!(
            manifest.io.input_names,
            vec!["x".to_string(), "y".to_string()]
        );
        assert_eq!(manifest.io.residual_len, 1);
        assert_eq!(manifest.io.jacobian_rows, 1);
        assert_eq!(manifest.io.jacobian_cols, 2);
        assert_eq!(manifest.io.jacobian_nnz, None);
        assert_eq!(manifest.functions.residual_fn_name, "eval_residual");
        assert_eq!(manifest.functions.jacobian_fn_name, "eval_jacobian");
    }

    #[test]
    fn sparse_manifest_keeps_nnz_and_chunk_names() {
        let residuals = vec![
            Expr::parse_expression("p + x"),
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

        let prepared = PreparedSparseProblem::new(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 1,
            }),
            SparseJacobianTask {
                fn_name: "eval_sparse_values",
                shape: (2, 2),
                entries: &entries,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 1 }),
        );

        let manifest = PreparedProblemManifest::from(&PreparedProblem::sparse(prepared));

        assert_eq!(manifest.backend_kind, BackendKind::Aot);
        assert_eq!(manifest.matrix_backend, MatrixBackend::SparseCol);
        assert_eq!(manifest.io.input_names, vec!["p", "x", "y"]);
        assert_eq!(manifest.io.jacobian_nnz, Some(2));
        assert_eq!(
            manifest.functions.residual_chunk_names,
            vec![
                "eval_residual_chunk_0".to_string(),
                "eval_residual_chunk_1".to_string()
            ]
        );
        assert_eq!(
            manifest.functions.jacobian_chunk_names,
            vec![
                "eval_sparse_values_chunk_0".to_string(),
                "eval_sparse_values_chunk_1".to_string()
            ]
        );
    }

    #[test]
    fn manifest_problem_key_changes_with_function_layout() {
        let residuals = vec![Expr::parse_expression("x")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];

        let prepared0 = PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual_a",
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
        );
        let prepared1 = PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            ResidualTask {
                fn_name: "eval_residual_b",
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
        );

        let manifest0 = PreparedProblemManifest::from(&prepared0);
        let manifest1 = PreparedProblemManifest::from(&prepared1);

        assert_ne!(manifest0.problem_key(), manifest1.problem_key());
    }
}
