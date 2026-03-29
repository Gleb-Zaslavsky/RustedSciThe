#![cfg(test)]

//! Adapters between task descriptions, runtime plans, and generic codegen IR.
//!
//! The project now has three distinct layers:
//! - `codegen_tasks`: mathematical task descriptions,
//! - `codegen_runtime_api`: solver-facing runtime plans,
//! - `CodegenIR`: generic lowering / optimization / Rust emission.
//!
//! This module hosts the thin bridges between those layers so that tests and
//! higher-level orchestration code do not need to manually repeat:
//! - `task -> runtime_plan`
//! - `runtime_plan -> GeneratedBlock`
//! - `runtime_plan -> CodegenModule`
//! - `runtime_plan -> IR blocks`
//!
//! Current scope:
//! - residual vectors
//! - dense Jacobian row blocks
//! - sparse Jacobian value blocks

use crate::symbolic::CodegenIR::{CodegenModule, GeneratedBlock, LinearBlock};
use crate::symbolic::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedDenseProblem, PreparedSparseProblem,
};
use crate::symbolic::codegen_runtime_api::{
    DenseJacobianChunkPlan, DenseJacobianChunkingStrategy, DenseJacobianRuntimePlan,
    ResidualChunkPlan, ResidualChunkingStrategy, ResidualRuntimePlan, SparseJacobianRuntimePlan,
    SparseJacobianValuesChunkPlan,
};
use crate::symbolic::codegen_tasks::{
    IvpJacobianTask, IvpResidualTask, JacobianTask, ResidualTask, SparseChunkingStrategy,
    SparseJacobianTask,
};

pub fn generated_block_from_residual_chunk(chunk: &ResidualChunkPlan<'_>) -> GeneratedBlock {
    GeneratedBlock::from_residual_plan(&chunk.plan)
}

pub fn generated_block_from_sparse_chunk(
    chunk: &SparseJacobianValuesChunkPlan<'_>,
) -> GeneratedBlock {
    GeneratedBlock::from_sparse_values_plan(&chunk.plan)
}

pub fn generated_block_from_dense_chunk(chunk: &DenseJacobianChunkPlan<'_>) -> GeneratedBlock {
    GeneratedBlock::from_dense_jacobian_plan(&chunk.plan)
}

pub fn residual_runtime_plan<'a>(
    task: &ResidualTask<'a>,
    strategy: ResidualChunkingStrategy,
) -> ResidualRuntimePlan<'a> {
    task.runtime_plan(strategy)
}

pub fn dense_runtime_plan<'a>(
    task: &JacobianTask<'a>,
    strategy: DenseJacobianChunkingStrategy,
) -> DenseJacobianRuntimePlan<'a> {
    task.runtime_plan(strategy)
}

pub fn ivp_residual_runtime_plan<'a>(
    task: &IvpResidualTask<'a>,
    strategy: ResidualChunkingStrategy,
) -> ResidualRuntimePlan<'a> {
    task.runtime_plan(strategy)
}

pub fn ivp_dense_runtime_plan<'a>(
    task: &IvpJacobianTask<'a>,
    strategy: DenseJacobianChunkingStrategy,
) -> DenseJacobianRuntimePlan<'a> {
    task.runtime_plan(strategy)
}

pub fn sparse_runtime_plan<'a>(
    task: &SparseJacobianTask<'a>,
    strategy: SparseChunkingStrategy,
) -> SparseJacobianRuntimePlan<'a> {
    task.runtime_plan(strategy)
}

pub fn prepared_dense_problem<'a>(
    backend_kind: BackendKind,
    matrix_backend: MatrixBackend,
    residual_task: &ResidualTask<'a>,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_task: &JacobianTask<'a>,
    jacobian_strategy: DenseJacobianChunkingStrategy,
) -> PreparedDenseProblem<'a> {
    PreparedDenseProblem::new(
        backend_kind,
        matrix_backend,
        residual_runtime_plan(residual_task, residual_strategy),
        dense_runtime_plan(jacobian_task, jacobian_strategy),
    )
}

pub fn prepared_sparse_problem<'a>(
    backend_kind: BackendKind,
    matrix_backend: MatrixBackend,
    residual_task: &ResidualTask<'a>,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_task: &SparseJacobianTask<'a>,
    jacobian_strategy: SparseChunkingStrategy,
) -> PreparedSparseProblem<'a> {
    PreparedSparseProblem::new(
        backend_kind,
        matrix_backend,
        residual_runtime_plan(residual_task, residual_strategy),
        sparse_runtime_plan(jacobian_task, jacobian_strategy),
    )
}

pub fn residual_ir_blocks(plan: &ResidualRuntimePlan<'_>) -> Vec<(usize, LinearBlock)> {
    plan.chunks
        .iter()
        .map(|chunk| {
            let block = generated_block_from_residual_chunk(chunk);
            (chunk.output_offset, block.ir)
        })
        .collect()
}

pub fn sparse_ir_blocks(plan: &SparseJacobianRuntimePlan<'_>) -> Vec<(usize, LinearBlock)> {
    plan.chunks
        .iter()
        .map(|chunk| {
            let block = generated_block_from_sparse_chunk(chunk);
            (chunk.value_offset, block.ir)
        })
        .collect()
}

pub fn dense_ir_blocks(plan: &DenseJacobianRuntimePlan<'_>) -> Vec<(usize, LinearBlock)> {
    plan.chunks
        .iter()
        .map(|chunk| {
            let block = generated_block_from_dense_chunk(chunk);
            (chunk.value_offset, block.ir)
        })
        .collect()
}

pub fn ivp_residual_ir_blocks(plan: &ResidualRuntimePlan<'_>) -> Vec<(usize, LinearBlock)> {
    residual_ir_blocks(plan)
}

pub fn ivp_dense_ir_blocks(plan: &DenseJacobianRuntimePlan<'_>) -> Vec<(usize, LinearBlock)> {
    dense_ir_blocks(plan)
}

pub fn residual_module(module_name: &str, plan: &ResidualRuntimePlan<'_>) -> CodegenModule {
    let mut module = CodegenModule::new(module_name);
    for chunk in &plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    module
}

pub fn dense_module(module_name: &str, plan: &DenseJacobianRuntimePlan<'_>) -> CodegenModule {
    let mut module = CodegenModule::new(module_name);
    for chunk in &plan.chunks {
        module.push_dense_jacobian_plan(&chunk.plan);
    }
    module
}

pub fn ivp_residual_module(module_name: &str, plan: &ResidualRuntimePlan<'_>) -> CodegenModule {
    residual_module(module_name, plan)
}

pub fn ivp_dense_module(module_name: &str, plan: &DenseJacobianRuntimePlan<'_>) -> CodegenModule {
    dense_module(module_name, plan)
}

pub fn sparse_module(module_name: &str, plan: &SparseJacobianRuntimePlan<'_>) -> CodegenModule {
    let mut module = CodegenModule::new(module_name);
    for chunk in &plan.chunks {
        module.push_sparse_values_plan(&chunk.plan);
    }
    module
}

pub fn residual_and_sparse_module(
    module_name: &str,
    residual_plan: &ResidualRuntimePlan<'_>,
    sparse_plan: &SparseJacobianRuntimePlan<'_>,
) -> CodegenModule {
    let mut module = CodegenModule::new(module_name);
    for chunk in &residual_plan.chunks {
        module.push_residual_block_plan(&chunk.plan);
    }
    for chunk in &sparse_plan.chunks {
        module.push_sparse_values_plan(&chunk.plan);
    }
    module
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen_provider_api::{BackendKind, MatrixBackend};
    use crate::symbolic::codegen_tasks::SparseExprEntry;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn residual_adapter_builds_ir_blocks_with_offsets() {
        let residuals = vec![
            Expr::parse_expression("x + 1"),
            Expr::parse_expression("y - 2"),
            Expr::parse_expression("x*y"),
        ];
        let vars = vec!["x", "y"];
        let task = ResidualTask {
            fn_name: "residual_eval",
            residuals: &residuals,
            variables: &vars,
            params: None,
        };

        let plan = residual_runtime_plan(
            &task,
            ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 2,
            },
        );
        let blocks = residual_ir_blocks(&plan);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, 0);
        assert_eq!(blocks[1].0, 2);
        assert_eq!(blocks[0].1.outputs.len(), 2);
        assert_eq!(blocks[1].1.outputs.len(), 1);
    }

    #[test]
    fn sparse_adapter_builds_module_with_sparse_blocks() {
        let expr0 = Expr::parse_expression("x + 1");
        let expr1 = Expr::parse_expression("y - 2");
        let expr2 = Expr::parse_expression("x*y");
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &expr0,
            },
            SparseExprEntry {
                row: 1,
                col: 0,
                expr: &expr1,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &expr2,
            },
        ];
        let vars = vec!["x", "y"];
        let task = SparseJacobianTask {
            fn_name: "sparse_values_eval",
            shape: (2, 2),
            entries: &entries,
            variables: &vars,
            params: None,
        };

        let plan = sparse_runtime_plan(
            &task,
            SparseChunkingStrategy::ByRowCount { rows_per_chunk: 1 },
        );
        let module = sparse_module("generated_sparse_test", &plan);
        let source = module.emit_source();

        assert!(source.contains("pub mod generated_sparse_test"));
        assert!(source.contains("sparse_values_eval_chunk_0"));
        assert!(source.contains("sparse_values_eval_chunk_1"));
    }

    #[test]
    fn dense_adapter_builds_ir_blocks_with_row_major_offsets() {
        let jacobian = vec![
            vec![Expr::parse_expression("x + 1"), Expr::parse_expression("y")],
            vec![Expr::parse_expression("2"), Expr::parse_expression("x*y")],
            vec![Expr::parse_expression("x - y"), Expr::parse_expression("3")],
        ];
        let vars = vec!["x", "y"];
        let task = JacobianTask {
            fn_name: "dense_jac_eval",
            jacobian: &jacobian,
            variables: &vars,
            params: None,
        };

        let plan = dense_runtime_plan(
            &task,
            DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 },
        );
        let blocks = dense_ir_blocks(&plan);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, 0);
        assert_eq!(blocks[1].0, 4);
        assert_eq!(blocks[0].1.outputs.len(), 4);
        assert_eq!(blocks[1].1.outputs.len(), 2);
    }

    #[test]
    fn prepared_sparse_problem_adapter_builds_provider_ready_shape() {
        let residuals = vec![
            Expr::parse_expression("x + 1"),
            Expr::parse_expression("y - 2"),
        ];
        let expr0 = Expr::parse_expression("1");
        let expr1 = Expr::parse_expression("x");
        let vars = vec!["x", "y"];
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &expr0,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &expr1,
            },
        ];

        let prepared = prepared_sparse_problem(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            &ResidualTask {
                fn_name: "eval_residual",
                residuals: &residuals,
                variables: &vars,
                params: None,
            },
            ResidualChunkingStrategy::Whole,
            &SparseJacobianTask {
                fn_name: "eval_sparse_values",
                shape: (2, 2),
                entries: &entries,
                variables: &vars,
                params: None,
            },
            SparseChunkingStrategy::Whole,
        );

        assert_eq!(prepared.backend_kind, BackendKind::Aot);
        assert_eq!(prepared.matrix_backend, MatrixBackend::SparseCol);
        assert_eq!(prepared.residual_len(), 2);
        assert_eq!(prepared.jacobian_shape(), (2, 2));
    }
}
