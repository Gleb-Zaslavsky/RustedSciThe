//! Driver helpers that turn prepared problems into emitted AOT artifacts.
//!
//! This layer sits between:
//! - prepared solver-facing problems,
//! - generic `CodegenModule` emission,
//! - and the separate generated-crate writer.
//!
//! Its job is to remove manual glue:
//! callers should not need to rebuild a `CodegenModule` by hand once they
//! already have a [`PreparedProblem`](crate::symbolic::codegen_provider_api::PreparedProblem).

use crate::symbolic::CodegenIR::CodegenModule;
use crate::symbolic::codegen_aot_crate::GeneratedAotCrate;
use crate::symbolic::codegen_provider_api::{
    PreparedDenseProblem, PreparedProblem, PreparedSparseProblem,
};
use log::info;

/// Builds a `CodegenModule` for a prepared dense problem.
pub fn codegen_module_from_prepared_dense_problem(
    module_name: &str,
    problem: &PreparedDenseProblem<'_>,
) -> CodegenModule {
    info!(
        "Building dense AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let mut module = CodegenModule::new(module_name);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_dense_jacobian_plan(&chunk.plan);
    }
    module
}

/// Builds a `CodegenModule` for a prepared sparse problem.
pub fn codegen_module_from_prepared_sparse_problem(
    module_name: &str,
    problem: &PreparedSparseProblem<'_>,
) -> CodegenModule {
    info!(
        "Building sparse AOT codegen module '{}' (residual_chunks={}, jacobian_chunks={})",
        module_name,
        problem.residual_plan.chunks.len(),
        problem.jacobian_plan.chunks.len()
    );
    let mut module = CodegenModule::new(module_name);
    for chunk in &problem.residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &problem.jacobian_plan.chunks {
        module.push_sparse_values_plan(&chunk.plan);
    }
    module
}

/// Builds a `CodegenModule` for any prepared problem.
pub fn codegen_module_from_prepared_problem(
    module_name: &str,
    problem: &PreparedProblem<'_>,
) -> CodegenModule {
    match problem {
        PreparedProblem::Dense(problem) => {
            codegen_module_from_prepared_dense_problem(module_name, problem)
        }
        PreparedProblem::Sparse(problem) => {
            codegen_module_from_prepared_sparse_problem(module_name, problem)
        }
    }
}

/// Builds a generated AOT crate directly from a prepared dense problem.
pub fn generated_aot_crate_from_prepared_dense_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedDenseProblem<'_>,
) -> GeneratedAotCrate {
    info!(
        "Assembling generated dense AOT crate '{}' from prepared problem",
        crate_name
    );
    let module = codegen_module_from_prepared_dense_problem(module_name, problem);
    GeneratedAotCrate::from_prepared_dense_problem(crate_name, problem, &module)
}

/// Builds a generated AOT crate directly from a prepared sparse problem.
pub fn generated_aot_crate_from_prepared_sparse_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedSparseProblem<'_>,
) -> GeneratedAotCrate {
    info!(
        "Assembling generated sparse AOT crate '{}' from prepared problem",
        crate_name
    );
    let module = codegen_module_from_prepared_sparse_problem(module_name, problem);
    GeneratedAotCrate::from_prepared_sparse_problem(crate_name, problem, &module)
}

/// Builds a generated AOT crate directly from any prepared problem.
pub fn generated_aot_crate_from_prepared_problem(
    crate_name: &str,
    module_name: &str,
    problem: &PreparedProblem<'_>,
) -> GeneratedAotCrate {
    info!(
        "Assembling generated AOT crate '{}' from prepared problem with {:?} matrix backend",
        crate_name,
        problem.matrix_backend()
    );
    let module = codegen_module_from_prepared_problem(module_name, problem);
    GeneratedAotCrate::from_prepared_problem(crate_name, problem, &module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_provider_api::{
        BackendKind, MatrixBackend, PreparedDenseProblem, PreparedSparseProblem,
    };
    use crate::symbolic::codegen_runtime_api::{
        DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
    };
    use crate::symbolic::codegen_tasks::{
        JacobianTask, ResidualTask, SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
    };
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn dense_driver_builds_module_with_residual_and_dense_jacobian_blocks() {
        let residuals = vec![Expr::parse_expression("x + 1")];
        let jacobian = vec![vec![Expr::parse_expression("1")]];
        let vars = vec!["x"];

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

        let module = codegen_module_from_prepared_dense_problem("dense_module_fixture", &prepared);
        let source = module.emit_source();

        assert!(source.contains("pub mod dense_module_fixture"));
        assert!(source.contains("pub fn eval_residual"));
        assert!(source.contains("pub fn eval_jacobian"));
    }

    #[test]
    fn sparse_driver_builds_generated_crate_from_prepared_problem() {
        let residuals = vec![
            Expr::parse_expression("x + p"),
            Expr::parse_expression("y - p"),
        ];
        let vars = vec!["x", "y"];
        let params = vec!["p"];
        let entry0 = Expr::parse_expression("1");
        let entry1 = Expr::parse_expression("2");
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

        let prepared = PreparedProblem::sparse(PreparedSparseProblem::new(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(ResidualChunkingStrategy::Whole),
            SparseJacobianTask {
                fn_name: "eval_sparse_values",
                shape: (2, 2),
                entries: &entries,
                variables: &vars,
                params: Some(&params),
            }
            .runtime_plan(SparseChunkingStrategy::Whole),
        ));

        let crate_spec = generated_aot_crate_from_prepared_problem(
            "generated_sparse_driver_fixture",
            "sparse_module_fixture",
            &prepared,
        );

        assert_eq!(crate_spec.crate_name, "generated_sparse_driver_fixture");
        assert_eq!(crate_spec.manifest.backend_kind.as_str(), "aot");
        assert_eq!(crate_spec.manifest.matrix_backend.as_str(), "sparse_col");
        assert!(
            crate_spec
                .module_source
                .contains("pub mod sparse_module_fixture")
        );
        assert!(crate_spec.module_source.contains("pub fn eval_residual"));
        assert!(
            crate_spec
                .module_source
                .contains("pub fn eval_sparse_values")
        );
    }
}
