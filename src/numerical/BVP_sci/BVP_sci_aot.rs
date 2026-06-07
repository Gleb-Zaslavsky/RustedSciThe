//! AtomView + AOT backend wiring for `BVP_sci`.
//!
//! This module isolates the compiled-backend lifecycle from the ExprLegacy path:
//! - symbolic preparation for AtomView
//! - sparse runtime planning
//! - AOT build/materialize/link
//! - faer-native residual/Jacobian adapters
//!
//! The goal is to mirror the mature `BVP_Damp` AOT pipeline while keeping the
//! solver core in `BVP_sci_faer` unchanged.

use crate::numerical::BVP_sci::BVP_sci_faer::{
    faer_col, faer_dense_mat, faer_mat, BCFunction, ODEFunction, ODEJacobian,
};
use crate::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciBackendError};
use crate::numerical::BVP_sci::BVP_sci_symbolic_functions::Jacobian_sci_faer;
use crate::symbolic::codegen::c_backend::codegen_c_aot_build::CAotCompileConfig;
use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::register_generated_c_sparse_backend;
use crate::symbolic::codegen::codegen_aot_driver::{
    generated_aot_artifact_from_prepared_problem, generated_aot_build_request_from_artifact,
    AotBuildPreset, AotCodegenBackend, ExecutedGeneratedAotBuild, GeneratedAotBuildRequest,
    GeneratedAotBuildResult,
};
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    register_generated_sparse_cdylib_backend, resolve_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedProblem, PreparedSparseProblem,
};
use crate::symbolic::codegen::codegen_runtime_api::{
    ResidualChunkPlan, ResidualRuntimePlan, SparseJacobianRuntimePlan, SparseJacobianStructure,
    SparseJacobianValuesChunkPlan,
};
use crate::symbolic::codegen::codegen_tasks::{
    CodegenOutputLayout, CodegenTaskKind, CodegenTaskPlan, PlannedOutput, SparseExprEntry,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_registry::register_zig_build_in_registry;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::register_generated_zig_sparse_backend;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
/// Backend mode for BVP_sci generated (AOT-compiled) sparse Jacobian/residual evaluation.
///
/// # Constraint
///
/// **All 8 variants operate exclusively on the sparse matrix backend.**
/// There is no dense AOT path in BVP_sci. If you need a dense generated backend,
/// use the BVP_Damp solver infrastructure instead.
///
/// # Variant overview
///
/// | Variant | Description |
/// |---------|-------------|
/// | [`LambdifyOnly`](BvpSciGeneratedBackendMode::LambdifyOnly) | Pure interpreted lambdify — no AOT compilation. |
/// | [`RequirePrebuiltAot`](BvpSciGeneratedBackendMode::RequirePrebuiltAot) | Fail if no pre-built AOT artifact is found. |
/// | [`BuildIfMissingRelease`](BvpSciGeneratedBackendMode::BuildIfMissingRelease) | Build C AOT (default C backend) if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseRust`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust) | Build Rust AOT if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseGcc`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc) | Build C AOT (GCC) if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseTcc`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc) | Build C AOT (TCC) if missing, release profile. |
/// | [`AtomViewBuildIfMissingReleaseZig`](BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig) | Build Zig AOT if missing, release profile. |
/// | [`AtomViewForRepeatedSolves`](BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves) | Build once, reuse across repeated solves. |
///
/// # Future variants
///
/// This enum is `#[non_exhaustive]` — adding new backends (e.g. dense AOT,
/// GPU offload) is a non-breaking change.
#[non_exhaustive]
pub enum BvpSciGeneratedBackendMode {
    /// Pure interpreted lambdify — no AOT compilation.
    /// Uses the ExprLegacySmartSparseLambdify workflow.
    #[default]
    LambdifyOnly,
    /// Fail with an error if no pre-built AOT artifact is found.
    /// Useful in CI/deployment where the artifact must already exist.
    RequirePrebuiltAot,
    /// Build C AOT (default C backend) if missing, release profile.
    BuildIfMissingRelease,
    /// Build Rust AOT if missing, release profile.
    AtomViewBuildIfMissingReleaseRust,
    /// Build C AOT (GCC) if missing, release profile.
    AtomViewBuildIfMissingReleaseGcc,
    /// Build C AOT (TCC) if missing, release profile.
    AtomViewBuildIfMissingReleaseTcc,
    /// Build Zig AOT if missing, release profile.
    AtomViewBuildIfMissingReleaseZig,
    /// Build once, reuse across repeated solves.
    /// The artifact is cached and not rebuilt on subsequent solver calls.
    AtomViewForRepeatedSolves,
}

#[derive(Clone, Debug)]
pub struct BvpSciGeneratedBackendConfig {
    pub mode: BvpSciGeneratedBackendMode,
    pub output_parent_dir: Option<PathBuf>,
}

impl BvpSciGeneratedBackendConfig {
    pub fn default_lambdify() -> Self {
        Self::default()
    }

    pub fn workflow(&self) -> super::BVP_sci_symb::BvpSciWorkflow {
        match self.mode {
            BvpSciGeneratedBackendMode::LambdifyOnly => {
                super::BVP_sci_symb::BvpSciWorkflow::ExprLegacySmartSparseLambdify
            }
            BvpSciGeneratedBackendMode::RequirePrebuiltAot
            | BvpSciGeneratedBackendMode::BuildIfMissingRelease
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc
            | BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig
            | BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves => {
                super::BVP_sci_symb::BvpSciWorkflow::AtomViewAotSparse
            }
        }
    }

    pub fn from_mode(mode: BvpSciGeneratedBackendMode) -> Self {
        Self {
            mode,
            output_parent_dir: None,
        }
    }

    pub fn sparse_atomview_require_prebuilt() -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::RequirePrebuiltAot)
    }

    pub fn sparse_atomview_build_if_missing_release(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::BuildIfMissingRelease)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_build_if_missing_release_gcc(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_build_if_missing_release_rust(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_build_if_missing_release_tcc(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_build_if_missing_release_zig(
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn sparse_atomview_for_repeated_solves(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves)
            .with_output_parent_dir(output_parent_dir)
    }

    pub fn with_output_parent_dir(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.output_parent_dir = Some(output_parent_dir.into());
        self
    }
}

impl Default for BvpSciGeneratedBackendConfig {
    fn default() -> Self {
        Self::from_mode(BvpSciGeneratedBackendMode::LambdifyOnly)
    }
}

#[derive(Clone)]
pub(crate) struct PreparedBvpSciSparseProblem {
    residual_eval: Arc<ODEFunction>,
    jacobian_eval: Option<Arc<ODEJacobian>>,
    boundary_conditions: std::collections::HashMap<String, Vec<(usize, f64)>>,
    values: Vec<String>,
}

/// AtomView-oriented symbolic payload for sparse AOT generation.
#[derive(Clone, Debug)]
pub(crate) struct AtomViewPreparedBvpSciProblem {
    pub(crate) equations: Vec<Expr>,
    pub(crate) symbolic_jacobian_sparse: Vec<(usize, usize, Expr)>,
    pub(crate) symbolic_param_jacobian_sparse: Option<Vec<(usize, usize, Expr)>>,
    pub(crate) time_arg: String,
    pub(crate) variables: Vec<String>,
    pub(crate) equation_parameters: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
struct BvpSciPointwiseSparsePrepared {
    residuals: Vec<Expr>,
    variable_names: Vec<String>,
    parameter_names: Vec<String>,
    time_arg: String,
    residual_fn_name: String,
    jacobian_fn_name: String,
    sparse_entries: Vec<(usize, usize, Expr)>,
    shape: (usize, usize),
}

impl BvpSciPointwiseSparsePrepared {
    fn flattened_input_names(&self) -> Vec<&str> {
        let mut names =
            Vec::with_capacity(1 + self.parameter_names.len() + self.variable_names.len());
        names.push(self.time_arg.as_str());
        names.extend(self.variable_names.iter().map(|name| name.as_str()));
        names.extend(self.parameter_names.iter().map(|name| name.as_str()));
        names
    }

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

    fn residual_runtime_plan(&self) -> ResidualRuntimePlan<'_> {
        let input_names = self.flattened_input_names();
        let outputs = self
            .residuals
            .iter()
            .map(|expr| PlannedOutput {
                expr,
                coordinate: None,
            })
            .collect::<Vec<_>>();

        ResidualRuntimePlan {
            fn_name: self.residual_fn_name.as_str(),
            output_len: self.residuals.len(),
            input_names: input_names.clone(),
            chunks: vec![ResidualChunkPlan {
                fn_name: self.residual_fn_name.clone(),
                output_offset: 0,
                residuals: &self.residuals,
                plan: CodegenTaskPlan {
                    fn_name: Cow::Owned(self.residual_fn_name.clone()),
                    kind: CodegenTaskKind::IvpResidual,
                    input_names,
                    outputs,
                    layout: CodegenOutputLayout::Vector {
                        len: self.residuals.len(),
                    },
                },
            }],
        }
    }

    fn sparse_runtime_plan(&self) -> SparseJacobianRuntimePlan<'_> {
        let all_entries = self.borrowed_sparse_entries();
        let input_names = self.flattened_input_names();
        let outputs = all_entries
            .iter()
            .map(|entry| PlannedOutput {
                expr: entry.expr,
                coordinate: Some((entry.row, entry.col)),
            })
            .collect::<Vec<_>>();

        SparseJacobianRuntimePlan {
            fn_name: self.jacobian_fn_name.as_str(),
            input_names: input_names.clone(),
            structure: SparseJacobianStructure {
                rows: self.shape.0,
                cols: self.shape.1,
                row_indices: all_entries.iter().map(|entry| entry.row).collect(),
                col_indices: all_entries.iter().map(|entry| entry.col).collect(),
            },
            chunks: vec![SparseJacobianValuesChunkPlan {
                fn_name: self.jacobian_fn_name.clone(),
                value_offset: 0,
                entries: all_entries.clone(),
                plan: CodegenTaskPlan {
                    fn_name: Cow::Owned(self.jacobian_fn_name.clone()),
                    kind: CodegenTaskKind::SparseJacobianValues,
                    input_names,
                    outputs,
                    layout: CodegenOutputLayout::SparseValues {
                        rows: self.shape.0,
                        cols: self.shape.1,
                        nnz: all_entries.len(),
                    },
                },
            }],
        }
    }

    fn as_prepared_problem(&self) -> PreparedSparseProblem<'_> {
        PreparedSparseProblem::new(
            BackendKind::Aot,
            MatrixBackend::SparseCol,
            self.residual_runtime_plan(),
            self.sparse_runtime_plan(),
        )
    }
}

/// Builds a sparse matrix family evaluator from sparse symbolic entries.
fn compile_bvp_sci_sparse_matrix_family_eval(
    sparse_entries: &[(usize, usize, Expr)],
    shape: (usize, usize),
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
) -> Arc<dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> Vec<faer_mat> + Send + Sync> {
    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |params| params.len()),
    );
    names.push(time_arg.to_string());
    names.extend(variables.iter().cloned());
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().cloned());
    }
    let name_refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();
    let compiled = sparse_entries
        .iter()
        .map(|(_, _, expr)| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
        .collect::<Vec<_>>();
    let structure = SparseJacobianStructure {
        rows: shape.0,
        cols: shape.1,
        row_indices: sparse_entries.iter().map(|(row, _, _)| *row).collect(),
        col_indices: sparse_entries.iter().map(|(_, col, _)| *col).collect(),
    };
    let variable_names = variables.to_vec();

    Arc::new(
        move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> Vec<faer_mat> {
            (0..y.ncols())
                .map(|point_index| {
                    let args = BVPwrap::flattened_args_for_mesh_point(
                        x,
                        y,
                        p,
                        point_index,
                        &variable_names,
                        &bounds,
                    );
                    let sparse_values = compiled.iter().map(|func| func(&args)).collect::<Vec<_>>();
                    structure.assemble_sparse_col_mat(sparse_values.as_slice())
                })
                .collect()
        },
    )
}

impl BVPwrap {
    /// Generates the symbolic payload needed by the AtomView/AOT pipeline.
    pub(crate) fn prepare_atomview_pointwise_problem(
        &self,
    ) -> Result<Arc<AtomViewPreparedBvpSciProblem>, BvpSciBackendError> {
        if self.param.is_empty() {
            if self
                .param_values
                .as_ref()
                .is_some_and(|values| !values.is_empty())
            {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci AtomView prepare received {} parameter values but no parameter names",
                        self.param_values.as_ref().map_or(0, |values| values.len())
                    ),
                });
            }
        } else {
            let values = self.param_values.as_ref().ok_or_else(|| {
                BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci AtomView prepare expected {} parameter values but none were provided",
                        self.param.len()
                    ),
                }
            })?;
            if values.len() != self.param.len() {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci AtomView prepare expected {} parameter values, got {}",
                        self.param.len(),
                        values.len()
                    ),
                });
            }
        }

        let mut legacy = Jacobian_sci_faer::new();
        legacy.from_vectors(self.eq_system.clone(), self.values.clone());
        legacy.calc_jacobian_parallel_smart();

        Ok(Arc::new(AtomViewPreparedBvpSciProblem {
            equations: self.eq_system.clone(),
            symbolic_jacobian_sparse: legacy.symbolic_jacobian_sparse,
            symbolic_param_jacobian_sparse: if self.param.is_empty() {
                None
            } else {
                Some(
                    self.eq_system
                        .iter()
                        .enumerate()
                        .flat_map(|(row, expr)| {
                            self.param
                                .iter()
                                .enumerate()
                                .filter_map(move |(col, parameter)| {
                                    let partial = expr.diff(parameter).simplify();
                                    if partial.is_zero() {
                                        None
                                    } else {
                                        Some((row, col, partial))
                                    }
                                })
                        })
                        .collect(),
                )
            },
            time_arg: self.arg.clone(),
            variables: self.values.clone(),
            equation_parameters: if self.param.is_empty() {
                None
            } else {
                Some(self.param.clone())
            },
        }))
    }

    fn flattened_args_for_mesh_point(
        x: &faer_col,
        y: &faer_dense_mat,
        p: &faer_col,
        point_index: usize,
        values: &[String],
        bounds: &Option<HashMap<String, Vec<(usize, f64)>>>,
    ) -> Vec<f64> {
        let n = y.nrows();
        let mut args = Vec::with_capacity(1 + n + p.nrows());
        args.push(x[point_index]);
        let bounds_ref = bounds.as_ref();
        for row in 0..n {
            args.push(Jacobian_sci_faer::handle_bounds(
                row,
                *y.get(row, point_index),
                bounds_ref,
                values,
            ));
        }
        for param_index in 0..p.nrows() {
            args.push(p[param_index]);
        }
        args
    }

    pub(crate) fn wrap_prepared_sparse_problem(
        &self,
        prepared: PreparedBvpSciSparseProblem,
    ) -> (
        Option<Box<ODEJacobian>>,
        Box<ODEFunction>,
        Option<Box<BCFunction>>,
    ) {
        let residual_eval = Arc::clone(&prepared.residual_eval);
        let residual = Box::new(move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
            (residual_eval)(x, y, p)
        }) as Box<ODEFunction>;

        let jacobian = prepared.jacobian_eval.map(|jacobian_eval| {
            Box::new(move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| (jacobian_eval)(x, y, p))
                as Box<ODEJacobian>
        });

        let bc_func = Self::BC_closure_creater(prepared.boundary_conditions, prepared.values);
        (jacobian, residual, bc_func)
    }

    fn pointwise_sparse_entries(
        pointwise: &AtomViewPreparedBvpSciProblem,
    ) -> Vec<SparseExprEntry<'_>> {
        pointwise
            .symbolic_jacobian_sparse
            .iter()
            .map(|(row, col, expr)| SparseExprEntry {
                row: *row,
                col: *col,
                expr,
            })
            .collect()
    }

    pub(crate) fn symbolic_sparse_structure(
        problem: &AtomViewPreparedBvpSciProblem,
    ) -> SparseJacobianStructure {
        let rows = problem.variables.len();
        let cols = problem.variables.len();
        let row_indices = problem
            .symbolic_jacobian_sparse
            .iter()
            .map(|(row, _, _)| *row)
            .collect();
        let col_indices = problem
            .symbolic_jacobian_sparse
            .iter()
            .map(|(_, col, _)| *col)
            .collect();

        SparseJacobianStructure {
            rows,
            cols,
            row_indices,
            col_indices,
        }
    }

    fn pointwise_sparse_prepared_problem(
        pointwise: &AtomViewPreparedBvpSciProblem,
    ) -> BvpSciPointwiseSparsePrepared {
        let sparse_entries = Self::pointwise_sparse_entries(pointwise)
            .into_iter()
            .map(|entry| (entry.row, entry.col, entry.expr.clone()))
            .collect::<Vec<_>>();

        BvpSciPointwiseSparsePrepared {
            residuals: pointwise.equations.clone(),
            variable_names: pointwise.variables.clone(),
            parameter_names: pointwise.equation_parameters.clone().unwrap_or_default(),
            time_arg: pointwise.time_arg.clone(),
            residual_fn_name: "eval_bvp_sci_pointwise_residual".to_string(),
            jacobian_fn_name: "eval_bvp_sci_pointwise_sparse_values".to_string(),
            sparse_entries,
            shape: (pointwise.variables.len(), pointwise.variables.len()),
        }
    }

    fn generated_sparse_names(
        problem_key: &str,
        config: &SymbolicIvpGeneratedBackendConfig,
    ) -> (String, String) {
        let suffix = problem_key
            .chars()
            .take(16)
            .collect::<String>()
            .replace('-', "_");
        let crate_name = config
            .crate_name_override
            .clone()
            .unwrap_or_else(|| format!("generated_bvp_sci_sparse_{suffix}"));
        let module_name = config
            .module_name_override
            .clone()
            .unwrap_or_else(|| format!("generated_bvp_sci_sparse_module_{suffix}"));
        (crate_name, module_name)
    }

    fn build_preset(policy: SymbolicIvpAotBuildPolicy) -> Option<AotBuildPreset> {
        match policy {
            SymbolicIvpAotBuildPolicy::BuildIfMissing { profile }
            | SymbolicIvpAotBuildPolicy::RebuildAlways { profile } => match profile {
                AotBuildProfile::Debug => Some(AotBuildPreset::DevFastest),
                AotBuildProfile::Release => Some(AotBuildPreset::Production),
            },
            SymbolicIvpAotBuildPolicy::UseIfAvailable
            | SymbolicIvpAotBuildPolicy::RequirePrebuilt => None,
        }
    }

    fn linked_sparse_residual_eval(
        &self,
        linked_problem_key: &str,
    ) -> Result<Arc<ODEFunction>, BvpSciBackendError> {
        let linked = resolve_linked_sparse_backend(linked_problem_key).ok_or_else(|| {
            BvpSciBackendError::GeneratedBackendFailure {
                message: "compiled sparse BVP_sci artifact exists but no linked sparse runtime is registered"
                    .to_string(),
            }
        })?;
        let values = self.values.clone();
        let bounds = self.Bounds.clone();
        Ok(Arc::new(
            move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> faer_dense_mat {
                let mut out = faer_dense_mat::zeros(linked.residual_len, y.ncols());
                for point_index in 0..y.ncols() {
                    let args = BVPwrap::flattened_args_for_mesh_point(
                        x,
                        y,
                        p,
                        point_index,
                        &values,
                        &bounds,
                    );
                    let mut point_out = vec![0.0; linked.residual_len];
                    (linked.residual_eval)(&args, &mut point_out);
                    for (row, value) in point_out.iter().enumerate() {
                        *out.get_mut(row, point_index) = *value;
                    }
                }
                out
            },
        ))
    }

    fn linked_sparse_jacobian_eval(
        &self,
        linked_problem_key: &str,
        sparse_structure: SparseJacobianStructure,
        param_jacobian_eval: Option<
            Arc<dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> Vec<faer_mat> + Send + Sync>,
        >,
    ) -> Result<Arc<ODEJacobian>, BvpSciBackendError> {
        let linked = resolve_linked_sparse_backend(linked_problem_key).ok_or_else(|| {
            BvpSciBackendError::GeneratedBackendFailure {
                message: "compiled sparse BVP_sci artifact exists but no linked sparse runtime is registered"
                    .to_string(),
            }
        })?;
        let values = self.values.clone();
        let bounds = self.Bounds.clone();
        Ok(Arc::new(
            move |x: &faer_col,
                  y: &faer_dense_mat,
                  p: &faer_col|
                  -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
                let jac_y = (0..y.ncols())
                    .map(|point_index| {
                        let args = BVPwrap::flattened_args_for_mesh_point(
                            x,
                            y,
                            p,
                            point_index,
                            &values,
                            &bounds,
                        );
                        let mut sparse_values = vec![0.0; linked.nnz];
                        (linked.jacobian_values_eval)(&args, &mut sparse_values);
                        sparse_structure.assemble_sparse_col_mat(sparse_values.as_slice())
                    })
                    .collect::<Vec<_>>();
                let jac_p = param_jacobian_eval
                    .as_ref()
                    .map(|param_eval| (param_eval)(x, y, p));
                (jac_y, jac_p)
            },
        ))
    }

    /// Ensures a compiled sparse backend is linked or builds one if requested.
    pub(crate) fn ensure_sparse_generated_runtime(
        &self,
        pointwise: &AtomViewPreparedBvpSciProblem,
        config: &SymbolicIvpGeneratedBackendConfig,
    ) -> Result<String, BvpSciBackendError> {
        let prepared_sparse = Self::pointwise_sparse_prepared_problem(pointwise);
        let prepared_problem = PreparedProblem::sparse(prepared_sparse.as_prepared_problem());
        let manifest = PreparedProblemManifest::from(&prepared_problem);
        let problem_key = manifest.problem_key();
        if resolve_linked_sparse_backend(problem_key.as_str()).is_some() {
            return Ok(problem_key);
        }

        match config.build_policy {
            SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: "symbolic BVP_sci sparse AOT artifact is missing".to_string(),
                });
            }
            SymbolicIvpAotBuildPolicy::UseIfAvailable => {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message:
                        "symbolic BVP_sci sparse AOT artifact is not linked in the current process"
                            .to_string(),
                });
            }
            SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
            | SymbolicIvpAotBuildPolicy::RebuildAlways { .. } => {}
        }

        let preset = Self::build_preset(config.build_policy).ok_or_else(|| {
            BvpSciBackendError::GeneratedBackendFailure {
                message: "symbolic BVP_sci sparse build policy did not provide a build preset"
                    .to_string(),
            }
        })?;
        let output_parent_dir = config.output_parent_dir.as_deref().ok_or(
            BvpSciBackendError::GeneratedBackendFailure {
                message: "symbolic BVP_sci sparse build requested without output directory"
                    .to_string(),
            },
        )?;
        let (artifact_name, module_name) =
            Self::generated_sparse_names(problem_key.as_str(), config);
        let artifact = generated_aot_artifact_from_prepared_problem(
            &artifact_name,
            &module_name,
            &prepared_problem,
            config.aot_codegen_backend,
        );
        let mut request =
            generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset);
        if let (GeneratedAotBuildRequest::C(c_request), Some(compiler)) =
            (&mut request, config.aot_c_compiler.as_ref())
        {
            let compile_config = match preset {
                AotBuildPreset::Production => CAotCompileConfig::production(),
                AotBuildPreset::FastBuild => CAotCompileConfig::fast_build(),
                AotBuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
            }
            .with_compiler(compiler.clone());
            *c_request = c_request.clone().with_compile_config(compile_config);
        }

        let build =
            request
                .materialize()
                .map_err(|err| BvpSciBackendError::GeneratedBackendFailure {
                    message: err.to_string(),
                })?;
        let executed =
            build
                .execute()
                .map_err(|err| BvpSciBackendError::GeneratedBackendFailure {
                    message: err.to_string(),
                })?;
        if !executed.succeeded() {
            let (status, stdout, stderr) = match &executed {
                ExecutedGeneratedAotBuild::Rust(result) => (
                    result.status_code,
                    result.stdout.clone(),
                    result.stderr.clone(),
                ),
                ExecutedGeneratedAotBuild::C(result) => (
                    result.status_code,
                    result.stdout.clone(),
                    result.stderr.clone(),
                ),
                ExecutedGeneratedAotBuild::Zig(result) => (
                    result.status_code,
                    result.stdout.clone(),
                    result.stderr.clone(),
                ),
            };
            return Err(BvpSciBackendError::GeneratedBackendFailure {
                message: format!("status={status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}"),
            });
        }

        let mut registry = AotRegistry::new();
        let registered = match &build {
            GeneratedAotBuildResult::Rust(result) => registry
                .register_materialized_build(manifest.clone(), result)
                .clone(),
            GeneratedAotBuildResult::C(result) => {
                register_c_build_in_registry(&mut registry, manifest.clone(), result).clone()
            }
            GeneratedAotBuildResult::Zig(result) => {
                register_zig_build_in_registry(&mut registry, manifest.clone(), result).clone()
            }
        };
        match config.aot_codegen_backend {
            AotCodegenBackend::Rust => register_generated_sparse_cdylib_backend(&registered),
            AotCodegenBackend::C => register_generated_c_sparse_backend(&registered),
            AotCodegenBackend::Zig => register_generated_zig_sparse_backend(&registered),
        }
        .map_err(|message| BvpSciBackendError::GeneratedBackendFailure { message })?;
        Ok(problem_key)
    }

    pub(crate) fn prepare_generated_sparse_problem(
        &self,
    ) -> Result<PreparedBvpSciSparseProblem, BvpSciBackendError> {
        let backend_config = self
            .bvp_generated_backend_config()?
            .expect("generated backend config should exist");
        let pointwise = self.prepare_atomview_pointwise_problem()?;
        let sparse_structure = Self::symbolic_sparse_structure(&pointwise);
        let linked_problem_key =
            self.ensure_sparse_generated_runtime(&pointwise, &backend_config)?;
        let residual_eval = self.linked_sparse_residual_eval(linked_problem_key.as_str())?;
        let param_jacobian_eval = pointwise.symbolic_param_jacobian_sparse.as_ref().map(
            |entries: &Vec<(usize, usize, Expr)>| {
                compile_bvp_sci_sparse_matrix_family_eval(
                    entries.as_slice(),
                    (
                        pointwise.equations.len(),
                        pointwise
                            .equation_parameters
                            .as_ref()
                            .map_or(0, |params| params.len()),
                    ),
                    pointwise.time_arg.as_str(),
                    pointwise.variables.as_slice(),
                    pointwise.equation_parameters.as_deref(),
                    self.Bounds.clone(),
                )
            },
        );
        let jacobian_eval = if self.use_analytical_jacobian {
            Some(self.linked_sparse_jacobian_eval(
                linked_problem_key.as_str(),
                sparse_structure,
                param_jacobian_eval,
            )?)
        } else {
            None
        };

        Ok(PreparedBvpSciSparseProblem {
            residual_eval,
            jacobian_eval,
            boundary_conditions: self.BoundaryConditions.clone(),
            values: self.values.clone(),
        })
    }

    pub(crate) fn bvp_generated_backend_config(
        &self,
    ) -> Result<Option<SymbolicIvpGeneratedBackendConfig>, BvpSciBackendError> {
        let output_parent_dir = self.generated_backend_config.output_parent_dir.clone();
        let config = match self.generated_backend_config.mode {
            BvpSciGeneratedBackendMode::LambdifyOnly => return Ok(None),
            BvpSciGeneratedBackendMode::RequirePrebuiltAot => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::RequirePrebuilt,
                )
            }
            BvpSciGeneratedBackendMode::BuildIfMissingRelease => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_tcc()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_rust()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_gcc()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_tcc()
            }
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_zig()
            }
            BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves => {
                SymbolicIvpGeneratedBackendConfig::from_mode(
                    DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
                )
                .with_c_tcc()
            }
        };
        Ok(Some(match output_parent_dir {
            Some(output_parent_dir) => config.with_output_parent_dir(Some(output_parent_dir)),
            None => config,
        }))
    }

    /// Entry point for AtomView+AOT backend generation.
    pub(crate) fn eq_generate_generated(
        &mut self,
    ) -> Result<
        (
            Option<Box<crate::numerical::BVP_sci::BVP_sci_faer::ODEJacobian>>,
            Box<crate::numerical::BVP_sci::BVP_sci_faer::ODEFunction>,
            Option<Box<crate::numerical::BVP_sci::BVP_sci_faer::BCFunction>>,
        ),
        BvpSciBackendError,
    > {
        let prepared = self.prepare_generated_sparse_problem()?;
        Ok(self.wrap_prepared_sparse_problem(prepared))
    }
}
