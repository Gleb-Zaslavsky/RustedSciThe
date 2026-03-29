//! Scenario and task descriptions for symbolic code generation.
//!
//! This module sits one level above [`crate::symbolic::CodegenIR`].
//! `CodegenIR` is responsible for lowering expressions into a linear IR and
//! emitting Rust source, while this module describes *what kind of
//! mathematical object* should be generated.
//!
//! The main design rule is that every residual or Jacobian task explicitly
//! carries two groups of symbolic names:
//! - `variables`: names that belong to the state / unknown vector and may be
//!   used for symbolic differentiation,
//! - `params`: optional names of extra arguments that participate in numeric
//!   evaluation but are **not** Jacobian differentiation variables.
//!
//! Keeping parameters separate is important for RustedSciThe use-cases:
//! - algebraic residuals and Jacobians may depend on tunable model constants,
//! - IVP systems may depend on both state variables and external parameters,
//! - large BVP / discretized systems often need codegen that knows which
//!   arguments are differentiable state entries and which are fixed parameters.
//!
//! This separation lets future codegen layers choose their own concrete
//! calling convention while preserving the mathematical meaning of the task.

use crate::symbolic::symbolic_engine::Expr;
use std::borrow::Cow;

/// Shared argument description for code-generated symbolic tasks.
///
/// Parameters are optional because many problems have no external model
/// constants, while others need them in every generated evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskArguments<'a> {
    /// Variables that belong to the state / unknown vector.
    pub variables: &'a [&'a str],
    /// Extra symbolic arguments that affect evaluation but are not Jacobian
    /// differentiation variables.
    pub params: Option<&'a [&'a str]>,
}

impl<'a> TaskArguments<'a> {
    /// Creates a new grouped argument description.
    pub fn new(variables: &'a [&'a str], params: Option<&'a [&'a str]>) -> Self {
        Self { variables, params }
    }

    /// Returns the number of differentiable state variables.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Returns the number of optional parameter arguments.
    pub fn parameter_count(&self) -> usize {
        self.params.map_or(0, |params| params.len())
    }

    /// Returns the full numeric argument count if parameters and variables are
    /// flattened into one slice.
    pub fn total_argument_count(&self) -> usize {
        self.variable_count() + self.parameter_count()
    }

    /// Returns argument names in the default flattened order:
    /// parameters first, then variables.
    pub fn flattened_names(&self) -> Vec<&'a str> {
        let mut names = Vec::with_capacity(self.total_argument_count());
        if let Some(params) = self.params {
            names.extend(params.iter().copied());
        }
        names.extend(self.variables.iter().copied());
        names
    }
}

/// Shared behavior for residual/Jacobian codegen tasks.
pub trait CodegenTask<'a> {
    /// Name of the generated Rust function.
    fn fn_name(&self) -> &'a str;

    /// Symbolic names that are differentiated with respect to.
    fn variables(&self) -> &'a [&'a str];

    /// Optional symbolic parameter names used only at evaluation time.
    fn params(&self) -> Option<&'a [&'a str]>;

    /// Returns grouped arguments for the task.
    fn arguments(&self) -> TaskArguments<'a> {
        TaskArguments::new(self.variables(), self.params())
    }

    /// Returns flattened argument names in the default order for future
    /// codegen backends.
    fn flattened_argument_names(&self) -> Vec<&'a str> {
        self.arguments().flattened_names()
    }
}

/// High-level kind of mathematical object to be emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenTaskKind {
    Residual,
    DenseJacobian,
    IvpResidual,
    IvpJacobian,
    SparseJacobianValues,
}

/// Layout of outputs produced by a generated task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenOutputLayout {
    Vector {
        len: usize,
    },
    Matrix {
        rows: usize,
        cols: usize,
    },
    SparseValues {
        rows: usize,
        cols: usize,
        nnz: usize,
    },
}

/// One flattened symbolic output produced by a task plan.
#[derive(Debug, Clone, Copy)]
pub struct PlannedOutput<'a> {
    /// Symbolic expression to lower and emit.
    pub expr: &'a Expr,
    /// Optional matrix coordinate for Jacobian-style outputs.
    pub coordinate: Option<(usize, usize)>,
}

/// Flattened task description ready for a lowerer/emitter pipeline.
///
/// This is the bridge between mathematical task types and the lower IR stage.
/// It intentionally contains only:
/// - a function name,
/// - flattened input names,
/// - flattened output expressions,
/// - output layout metadata.
#[derive(Debug, Clone)]
pub struct CodegenTaskPlan<'a> {
    pub fn_name: Cow<'a, str>,
    pub kind: CodegenTaskKind,
    pub input_names: Vec<&'a str>,
    pub outputs: Vec<PlannedOutput<'a>>,
    pub layout: CodegenOutputLayout,
}

impl<'a> CodegenTaskPlan<'a> {
    /// Returns plain expression references in output order.
    pub fn output_exprs(&self) -> Vec<&'a Expr> {
        self.outputs.iter().map(|output| output.expr).collect()
    }
}

/// Algebraic residual task `F(y, p)`.
#[derive(Debug, Clone)]
pub struct ResidualTask<'a> {
    /// Name of the generated Rust function.
    pub fn_name: &'a str,
    /// Residual expressions in output order.
    pub residuals: &'a [Expr],
    /// Differentiable state variables / unknowns.
    pub variables: &'a [&'a str],
    /// Optional parameter arguments used during evaluation.
    pub params: Option<&'a [&'a str]>,
}

impl<'a> CodegenTask<'a> for ResidualTask<'a> {
    fn fn_name(&self) -> &'a str {
        self.fn_name
    }

    fn variables(&self) -> &'a [&'a str] {
        self.variables
    }

    fn params(&self) -> Option<&'a [&'a str]> {
        self.params
    }
}

impl<'a> ResidualTask<'a> {
    /// Builds a flattened residual-generation plan.
    pub fn plan(&self) -> CodegenTaskPlan<'a> {
        CodegenTaskPlan {
            fn_name: Cow::Borrowed(self.fn_name()),
            kind: CodegenTaskKind::Residual,
            input_names: self.flattened_argument_names(),
            outputs: self
                .residuals
                .iter()
                .map(|expr| PlannedOutput {
                    expr,
                    coordinate: None,
                })
                .collect(),
            layout: CodegenOutputLayout::Vector {
                len: self.residuals.len(),
            },
        }
    }
}

/// Dense algebraic Jacobian task `J(y, p)`.
#[derive(Debug, Clone)]
pub struct JacobianTask<'a> {
    /// Name of the generated Rust function.
    pub fn_name: &'a str,
    /// Symbolic Jacobian rows in matrix order.
    pub jacobian: &'a [Vec<Expr>],
    /// Variables used for symbolic differentiation.
    pub variables: &'a [&'a str],
    /// Optional parameter arguments used during evaluation.
    pub params: Option<&'a [&'a str]>,
}

impl<'a> CodegenTask<'a> for JacobianTask<'a> {
    fn fn_name(&self) -> &'a str {
        self.fn_name
    }

    fn variables(&self) -> &'a [&'a str] {
        self.variables
    }

    fn params(&self) -> Option<&'a [&'a str]> {
        self.params
    }
}

impl<'a> JacobianTask<'a> {
    /// Builds a flattened dense Jacobian-generation plan in row-major order.
    pub fn plan(&self) -> CodegenTaskPlan<'a> {
        let rows = self.jacobian.len();
        let cols = self.jacobian.first().map_or(0, |row| row.len());
        let mut outputs = Vec::with_capacity(rows * cols);

        for (row_idx, row) in self.jacobian.iter().enumerate() {
            for (col_idx, expr) in row.iter().enumerate() {
                outputs.push(PlannedOutput {
                    expr,
                    coordinate: Some((row_idx, col_idx)),
                });
            }
        }

        CodegenTaskPlan {
            fn_name: Cow::Borrowed(self.fn_name()),
            kind: CodegenTaskKind::DenseJacobian,
            input_names: self.flattened_argument_names(),
            outputs,
            layout: CodegenOutputLayout::Matrix { rows, cols },
        }
    }
}

/// IVP residual task `f(t, y, p)`.
#[derive(Debug, Clone)]
pub struct IvpResidualTask<'a> {
    /// Name of the generated Rust function.
    pub fn_name: &'a str,
    /// Name of the distinguished scalar argument (usually time).
    pub time_arg: &'a str,
    /// Residual / right-hand-side expressions in output order.
    pub residuals: &'a [Expr],
    /// State variables that form the `y` vector.
    pub variables: &'a [&'a str],
    /// Optional parameter arguments used during evaluation.
    pub params: Option<&'a [&'a str]>,
}

impl<'a> CodegenTask<'a> for IvpResidualTask<'a> {
    fn fn_name(&self) -> &'a str {
        self.fn_name
    }

    fn variables(&self) -> &'a [&'a str] {
        self.variables
    }

    fn params(&self) -> Option<&'a [&'a str]> {
        self.params
    }
}

impl<'a> IvpResidualTask<'a> {
    /// Returns flattened names in the default IVP order:
    /// time first, then parameters, then state variables.
    pub fn flattened_argument_names_with_time(&self) -> Vec<&'a str> {
        let mut names = Vec::with_capacity(1 + self.arguments().total_argument_count());
        names.push(self.time_arg);
        names.extend(self.flattened_argument_names());
        names
    }

    /// Builds a flattened IVP residual-generation plan.
    pub fn plan(&self) -> CodegenTaskPlan<'a> {
        CodegenTaskPlan {
            fn_name: Cow::Borrowed(self.fn_name()),
            kind: CodegenTaskKind::IvpResidual,
            input_names: self.flattened_argument_names_with_time(),
            outputs: self
                .residuals
                .iter()
                .map(|expr| PlannedOutput {
                    expr,
                    coordinate: None,
                })
                .collect(),
            layout: CodegenOutputLayout::Vector {
                len: self.residuals.len(),
            },
        }
    }
}

/// IVP Jacobian task `J(t, y, p)`.
#[derive(Debug, Clone)]
pub struct IvpJacobianTask<'a> {
    /// Name of the generated Rust function.
    pub fn_name: &'a str,
    /// Name of the distinguished scalar argument (usually time).
    pub time_arg: &'a str,
    /// Symbolic Jacobian rows in matrix order.
    pub jacobian: &'a [Vec<Expr>],
    /// State variables that form the `y` vector and the Jacobian columns.
    pub variables: &'a [&'a str],
    /// Optional parameter arguments used during evaluation.
    pub params: Option<&'a [&'a str]>,
}

impl<'a> CodegenTask<'a> for IvpJacobianTask<'a> {
    fn fn_name(&self) -> &'a str {
        self.fn_name
    }

    fn variables(&self) -> &'a [&'a str] {
        self.variables
    }

    fn params(&self) -> Option<&'a [&'a str]> {
        self.params
    }
}

impl<'a> IvpJacobianTask<'a> {
    /// Returns flattened names in the default IVP order:
    /// time first, then parameters, then state variables.
    pub fn flattened_argument_names_with_time(&self) -> Vec<&'a str> {
        let mut names = Vec::with_capacity(1 + self.arguments().total_argument_count());
        names.push(self.time_arg);
        names.extend(self.flattened_argument_names());
        names
    }

    /// Builds a flattened IVP Jacobian-generation plan in row-major order.
    pub fn plan(&self) -> CodegenTaskPlan<'a> {
        let rows = self.jacobian.len();
        let cols = self.jacobian.first().map_or(0, |row| row.len());
        let mut outputs = Vec::with_capacity(rows * cols);

        for (row_idx, row) in self.jacobian.iter().enumerate() {
            for (col_idx, expr) in row.iter().enumerate() {
                outputs.push(PlannedOutput {
                    expr,
                    coordinate: Some((row_idx, col_idx)),
                });
            }
        }

        CodegenTaskPlan {
            fn_name: Cow::Borrowed(self.fn_name()),
            kind: CodegenTaskKind::IvpJacobian,
            input_names: self.flattened_argument_names_with_time(),
            outputs,
            layout: CodegenOutputLayout::Matrix { rows, cols },
        }
    }
}

/// One explicitly stored non-zero symbolic Jacobian entry.
#[derive(Debug, Clone, Copy)]
pub struct SparseExprEntry<'a> {
    pub row: usize,
    pub col: usize,
    pub expr: &'a Expr,
}

/// Sparse Jacobian task for large structured systems.
#[derive(Debug, Clone)]
pub struct SparseJacobianTask<'a> {
    /// Name of the generated Rust function.
    pub fn_name: &'a str,
    /// Matrix shape `(rows, cols)`.
    pub shape: (usize, usize),
    /// Explicit list of known non-zero entries.
    pub entries: &'a [SparseExprEntry<'a>],
    /// Variables used for symbolic differentiation.
    pub variables: &'a [&'a str],
    /// Optional parameter arguments used during evaluation.
    pub params: Option<&'a [&'a str]>,
}

/// Practical chunking strategies for sparse Jacobian values codegen.
///
/// The best strategy is workload-dependent, so the planning layer exposes
/// several options instead of forcing one policy too early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseChunkingStrategy {
    /// Keep the full sparse values task as one block.
    Whole,
    /// Split rows into a target number of coarse chunks. This is useful when
    /// runtime scheduling should operate on a few heavier super-blocks instead
    /// of many tiny row groups.
    ByTargetChunkCount { target_chunks: usize },
    /// Split by a maximum number of explicit non-zero entries per chunk.
    ByNonZeroCount { max_entries_per_chunk: usize },
    /// Split by row groups of fixed height. Entries keep their original
    /// explicit order inside each produced chunk.
    ByRowCount { rows_per_chunk: usize },
}

/// One chunk of a sparse Jacobian values task.
///
/// Chunking is the planning-stage bridge to future parallel execution:
/// each chunk keeps the same input signature and global matrix shape, but owns
/// only a contiguous sub-slice of explicit non-zero entries.
#[derive(Debug, Clone)]
pub struct SparseJacobianChunkTask<'a> {
    /// Base function name shared by all chunks.
    pub base_fn_name: &'a str,
    /// Zero-based chunk index in the original explicit-entry order.
    pub chunk_index: usize,
    /// Global offset of the first entry in this chunk.
    pub entry_offset: usize,
    /// Matrix shape `(rows, cols)`.
    pub shape: (usize, usize),
    /// Explicit list of known non-zero entries for this chunk.
    pub entries: Vec<SparseExprEntry<'a>>,
    /// Variables used for symbolic differentiation.
    pub variables: &'a [&'a str],
    /// Optional parameter arguments used during evaluation.
    pub params: Option<&'a [&'a str]>,
}

impl<'a> CodegenTask<'a> for SparseJacobianTask<'a> {
    fn fn_name(&self) -> &'a str {
        self.fn_name
    }

    fn variables(&self) -> &'a [&'a str] {
        self.variables
    }

    fn params(&self) -> Option<&'a [&'a str]> {
        self.params
    }
}

impl<'a> SparseJacobianTask<'a> {
    /// Builds a flattened sparse-Jacobian-values plan in explicit entry order.
    pub fn plan(&self) -> CodegenTaskPlan<'a> {
        CodegenTaskPlan {
            fn_name: Cow::Borrowed(self.fn_name()),
            kind: CodegenTaskKind::SparseJacobianValues,
            input_names: self.flattened_argument_names(),
            outputs: self
                .entries
                .iter()
                .map(|entry| PlannedOutput {
                    expr: entry.expr,
                    coordinate: Some((entry.row, entry.col)),
                })
                .collect(),
            layout: CodegenOutputLayout::SparseValues {
                rows: self.shape.0,
                cols: self.shape.1,
                nnz: self.entries.len(),
            },
        }
    }

    /// Splits explicit sparse entries into contiguous chunks.
    ///
    /// The resulting chunks preserve:
    /// - global matrix shape,
    /// - parameter list,
    /// - variable list,
    /// - explicit entry order.
    pub fn chunk_by_nnz(&self, max_entries_per_chunk: usize) -> Vec<SparseJacobianChunkTask<'a>> {
        assert!(
            max_entries_per_chunk > 0,
            "max_entries_per_chunk must be positive"
        );

        self.entries
            .chunks(max_entries_per_chunk)
            .enumerate()
            .map(|(chunk_index, entries)| SparseJacobianChunkTask {
                base_fn_name: self.fn_name,
                chunk_index,
                entry_offset: chunk_index * max_entries_per_chunk,
                shape: self.shape,
                entries: entries.to_vec(),
                variables: self.variables,
                params: self.params,
            })
            .collect()
    }

    /// Splits sparse entries using one of several practical chunking
    /// strategies.
    pub fn chunk_with_strategy(
        &self,
        strategy: SparseChunkingStrategy,
    ) -> Vec<SparseJacobianChunkTask<'a>> {
        match strategy {
            SparseChunkingStrategy::Whole => vec![SparseJacobianChunkTask {
                base_fn_name: self.fn_name,
                chunk_index: 0,
                entry_offset: 0,
                shape: self.shape,
                entries: self.entries.to_vec(),
                variables: self.variables,
                params: self.params,
            }],
            SparseChunkingStrategy::ByTargetChunkCount { target_chunks } => {
                assert!(target_chunks > 0, "target_chunks must be positive");
                let rows_per_chunk = self.shape.0.max(1).div_ceil(target_chunks).max(1);
                self.chunk_with_strategy(SparseChunkingStrategy::ByRowCount { rows_per_chunk })
            }
            SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk,
            } => self.chunk_by_nnz(max_entries_per_chunk),
            SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
                assert!(rows_per_chunk > 0, "rows_per_chunk must be positive");

                let mut chunk_entries: Vec<Vec<SparseExprEntry<'a>>> = Vec::new();
                let mut chunk_offsets: Vec<usize> = Vec::new();

                for (entry_index, entry) in self.entries.iter().copied().enumerate() {
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
                    .map(|(chunk_index, entries)| SparseJacobianChunkTask {
                        base_fn_name: self.fn_name,
                        chunk_index,
                        entry_offset: chunk_offsets[chunk_index],
                        shape: self.shape,
                        entries,
                        variables: self.variables,
                        params: self.params,
                    })
                    .collect()
            }
        }
    }
}

impl<'a> CodegenTask<'a> for SparseJacobianChunkTask<'a> {
    fn fn_name(&self) -> &'a str {
        self.base_fn_name
    }

    fn variables(&self) -> &'a [&'a str] {
        self.variables
    }

    fn params(&self) -> Option<&'a [&'a str]> {
        self.params
    }
}

impl<'a> SparseJacobianChunkTask<'a> {
    /// Returns a stable generated function name for this chunk.
    pub fn chunk_fn_name(&self) -> String {
        format!("{}_chunk_{}", self.base_fn_name, self.chunk_index)
    }

    /// Builds a flattened sparse-values plan for this chunk.
    pub fn plan(&self) -> CodegenTaskPlan<'a> {
        CodegenTaskPlan {
            fn_name: Cow::Owned(self.chunk_fn_name()),
            kind: CodegenTaskKind::SparseJacobianValues,
            input_names: self.flattened_argument_names(),
            outputs: self
                .entries
                .iter()
                .map(|entry| PlannedOutput {
                    expr: entry.expr,
                    coordinate: Some((entry.row, entry.col)),
                })
                .collect(),
            layout: CodegenOutputLayout::SparseValues {
                rows: self.shape.0,
                cols: self.shape.1,
                nnz: self.entries.len(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_arguments_flatten_params_before_variables() {
        let args = TaskArguments::new(&["y0", "y1"], Some(&["alpha", "beta"]));

        assert_eq!(args.parameter_count(), 2);
        assert_eq!(args.variable_count(), 2);
        assert_eq!(args.total_argument_count(), 4);
        assert_eq!(args.flattened_names(), vec!["alpha", "beta", "y0", "y1"]);
    }

    #[test]
    fn task_arguments_handle_absent_params() {
        let args = TaskArguments::new(&["x", "y"], None);

        assert_eq!(args.parameter_count(), 0);
        assert_eq!(args.total_argument_count(), 2);
        assert_eq!(args.flattened_names(), vec!["x", "y"]);
    }

    #[test]
    fn ivp_task_keeps_time_argument_outside_param_group() {
        let jacobian = vec![vec![Expr::Const(1.0)]];
        let task = IvpJacobianTask {
            fn_name: "eval_ivp_jacobian",
            time_arg: "t",
            jacobian: &jacobian,
            variables: &["y"],
            params: Some(&["alpha"]),
        };

        assert_eq!(
            task.flattened_argument_names_with_time(),
            vec!["t", "alpha", "y"]
        );
    }

    #[test]
    fn residual_plan_keeps_param_then_variable_input_order() {
        let residuals = vec![Expr::Const(1.0), Expr::Var("y".to_string())];
        let task = ResidualTask {
            fn_name: "eval_residual",
            residuals: &residuals,
            variables: &["y"],
            params: Some(&["alpha"]),
        };

        let plan = task.plan();

        assert_eq!(plan.fn_name, "eval_residual");
        assert_eq!(plan.kind, CodegenTaskKind::Residual);
        assert_eq!(plan.input_names, vec!["alpha", "y"]);
        assert_eq!(plan.outputs.len(), 2);
        assert_eq!(plan.layout, CodegenOutputLayout::Vector { len: 2 });
        assert!(
            plan.outputs
                .iter()
                .all(|output| output.coordinate.is_none())
        );
    }

    #[test]
    fn dense_jacobian_plan_flattens_row_major_with_coordinates() {
        let jacobian = vec![
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())],
            vec![Expr::Var("c".to_string()), Expr::Var("d".to_string())],
        ];
        let task = JacobianTask {
            fn_name: "eval_jacobian",
            jacobian: &jacobian,
            variables: &["x", "y"],
            params: Some(&["alpha"]),
        };

        let plan = task.plan();

        assert_eq!(plan.input_names, vec!["alpha", "x", "y"]);
        assert_eq!(
            plan.layout,
            CodegenOutputLayout::Matrix { rows: 2, cols: 2 }
        );
        assert_eq!(plan.outputs.len(), 4);
        assert_eq!(plan.outputs[0].coordinate, Some((0, 0)));
        assert_eq!(plan.outputs[1].coordinate, Some((0, 1)));
        assert_eq!(plan.outputs[2].coordinate, Some((1, 0)));
        assert_eq!(plan.outputs[3].coordinate, Some((1, 1)));
    }

    #[test]
    fn sparse_jacobian_plan_preserves_explicit_entry_order() {
        let e0 = Expr::Var("j00".to_string());
        let e1 = Expr::Var("j12".to_string());
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 1,
                col: 2,
                expr: &e1,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (3, 4),
            entries: &entries,
            variables: &["y0", "y1"],
            params: None,
        };

        let plan = task.plan();

        assert_eq!(
            plan.layout,
            CodegenOutputLayout::SparseValues {
                rows: 3,
                cols: 4,
                nnz: 2
            }
        );
        assert_eq!(plan.outputs.len(), 2);
        assert_eq!(plan.outputs[0].coordinate, Some((0, 0)));
        assert_eq!(plan.outputs[1].coordinate, Some((1, 2)));
        assert!(std::ptr::eq(plan.outputs[0].expr, &e0));
        assert!(std::ptr::eq(plan.outputs[1].expr, &e1));
    }

    #[test]
    fn sparse_jacobian_chunking_preserves_order_and_params() {
        let e0 = Expr::Var("y0".to_string());
        let e1 = Expr::Var("y1".to_string());
        let e2 = Expr::Var("y2".to_string());
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &e1,
            },
            SparseExprEntry {
                row: 2,
                col: 2,
                expr: &e2,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (4, 4),
            entries: &entries,
            variables: &["y0", "y1", "y2"],
            params: Some(&["alpha"]),
        };

        let chunks = task.chunk_by_nnz(2);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].entry_offset, 0);
        assert_eq!(chunks[0].entries.len(), 2);
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[1].entry_offset, 2);
        assert_eq!(chunks[1].entries.len(), 1);
        assert_eq!(
            chunks[0].flattened_argument_names(),
            vec!["alpha", "y0", "y1", "y2"]
        );
        assert_eq!(chunks[1].chunk_fn_name(), "eval_sparse_values_chunk_1");

        let plan0 = chunks[0].plan();
        let plan1 = chunks[1].plan();
        assert_eq!(
            plan0.layout,
            CodegenOutputLayout::SparseValues {
                rows: 4,
                cols: 4,
                nnz: 2
            }
        );
        assert_eq!(
            plan1.layout,
            CodegenOutputLayout::SparseValues {
                rows: 4,
                cols: 4,
                nnz: 1
            }
        );
        assert_eq!(plan0.outputs[0].coordinate, Some((0, 0)));
        assert_eq!(plan0.outputs[1].coordinate, Some((1, 1)));
        assert_eq!(plan1.outputs[0].coordinate, Some((2, 2)));
    }

    #[test]
    fn sparse_chunking_strategy_whole_preserves_single_chunk() {
        let e0 = Expr::Var("y0".to_string());
        let e1 = Expr::Var("y1".to_string());
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 2,
                col: 3,
                expr: &e1,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (4, 4),
            entries: &entries,
            variables: &["y0", "y1"],
            params: None,
        };

        let chunks = task.chunk_with_strategy(SparseChunkingStrategy::Whole);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].entry_offset, 0);
        assert_eq!(chunks[0].entries.len(), 2);
    }

    #[test]
    fn sparse_chunking_strategy_by_row_count_groups_rows() {
        let e0 = Expr::Var("y0".to_string());
        let e1 = Expr::Var("y1".to_string());
        let e2 = Expr::Var("y2".to_string());
        let e3 = Expr::Var("y3".to_string());
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &e1,
            },
            SparseExprEntry {
                row: 2,
                col: 2,
                expr: &e2,
            },
            SparseExprEntry {
                row: 3,
                col: 3,
                expr: &e3,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (4, 4),
            entries: &entries,
            variables: &["y0", "y1", "y2", "y3"],
            params: Some(&["alpha"]),
        };

        let chunks =
            task.chunk_with_strategy(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 2 });

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].entry_offset, 0);
        assert_eq!(chunks[0].entries.len(), 2);
        assert_eq!(chunks[0].entries[0].row, 0);
        assert_eq!(chunks[0].entries[1].row, 1);
        assert_eq!(chunks[1].entry_offset, 2);
        assert_eq!(chunks[1].entries.len(), 2);
        assert_eq!(chunks[1].entries[0].row, 2);
        assert_eq!(chunks[1].entries[1].row, 3);
        assert_eq!(
            chunks[1].flattened_argument_names(),
            vec!["alpha", "y0", "y1", "y2", "y3"]
        );
    }

    #[test]
    fn sparse_chunking_strategy_by_target_chunk_count_creates_coarse_row_groups() {
        let e0 = Expr::Var("y0".to_string());
        let e1 = Expr::Var("y1".to_string());
        let e2 = Expr::Var("y2".to_string());
        let e3 = Expr::Var("y3".to_string());
        let entries = vec![
            SparseExprEntry {
                row: 0,
                col: 0,
                expr: &e0,
            },
            SparseExprEntry {
                row: 1,
                col: 1,
                expr: &e1,
            },
            SparseExprEntry {
                row: 2,
                col: 2,
                expr: &e2,
            },
            SparseExprEntry {
                row: 3,
                col: 3,
                expr: &e3,
            },
        ];
        let task = SparseJacobianTask {
            fn_name: "eval_sparse_values",
            shape: (4, 4),
            entries: &entries,
            variables: &["y0", "y1", "y2", "y3"],
            params: None,
        };

        let chunks = task
            .chunk_with_strategy(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 2 });

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].entries.len(), 2);
        assert_eq!(chunks[1].entries.len(), 2);
        assert_eq!(chunks[0].entries[0].row, 0);
        assert_eq!(chunks[1].entries[0].row, 2);
    }
}
