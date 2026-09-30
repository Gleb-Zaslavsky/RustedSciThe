use crate::numerical::BDF::BDF_solver::BdfJacobian;
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::View::evaluate::PreparedEvaluator;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedJacobianLayout, LinkedSparseAotBackend,
};
use crate::symbolic::codegen::codegen_runtime_api::SparseJacobianStructure;
use crate::symbolic::ivp_telemetry::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpWarmStage,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, PreparedNativeAtomJacobian,
    SharedIvpParameterValues, SymbolicIvpProblemOptions, prepare_native_atom_jacobian,
    with_shared_parameter_values,
};
use crate::symbolic::symbolic_ivp_generated::{
    SelectedSymbolicIvpBackendKind, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_banded_backend, prepare_generated_symbolic_ivp_sparse_backend,
};
use faer::sparse::Triplet;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

type CompiledEntry = (usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>);

/// Fallible native Jacobian callback used before the solver-facing
/// compatibility adapter. Keeping this boundary typed prevents callback
/// evaluation and output-layout failures from being converted to `NaN`
/// while the native runtime is being tested or composed by another caller.
pub type NativeJacobianCallback =
    dyn FnMut(f64, &DVector<f64>) -> Result<BdfJacobian, IvpBackendError>;

#[derive(Default)]
struct NativeJacobianWorkspace {
    values: Vec<f64>,
}

impl NativeJacobianWorkspace {
    fn with_capacity(values_len: usize) -> Self {
        Self {
            values: Vec::with_capacity(values_len),
        }
    }
}

/// Precomputed addresses for the compact Banded output buffer.
///
/// The symbolic sparsity pattern is immutable after preparation. Keeping the
/// physical slots beside the entries avoids repeating band arithmetic and
/// bounds checks on every native Jacobian callback.
#[derive(Clone)]
struct NativeBandedLayout {
    kl: usize,
    ku: usize,
    data_len: usize,
    slots: Arc<[usize]>,
}

/// Native symbolic Jacobian storage requested by the LSODE2 linear backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeJacobianStorage {
    Dense,
    SparseTriplets,
    Banded { bandwidth: Option<(usize, usize)> },
}

enum NativePreparedJacobianStorage {
    Dense,
    SparseTriplets,
    Banded(NativeBandedLayout),
}

/// Prepared native AtomView Jacobian runtime.
///
/// The symbolic plan, fixed sparse ordering, band slots and evaluation
/// workspace are owned by one object. Caller-owned output methods reuse the
/// caller's buffers, while the compatibility callback below still materializes
/// the historical `BdfJacobian` values when that ABI is required.
pub struct NativeAtomJacobianRuntime {
    plan: Arc<PreparedNativeAtomJacobian>,
    storage: NativePreparedJacobianStorage,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
    workspace: NativeJacobianWorkspace,
}

impl NativeAtomJacobianRuntime {
    /// Number of rows in the prepared Jacobian.
    pub fn rows(&self) -> usize {
        self.plan.rows
    }

    /// Number of columns in the prepared Jacobian.
    pub fn cols(&self) -> usize {
        self.plan.cols
    }

    /// Returns cold-path evaluator shape diagnostics for the prepared Atom
    /// Jacobian. This is used by performance stories, never by callbacks.
    pub(crate) fn plan_metrics(
        &self,
    ) -> crate::symbolic::symbolic_ivp::PreparedNativeJacobianMetrics {
        self.plan.metrics()
    }

    /// Returns the immutable sparse pattern in callback value order.
    ///
    /// This is a cold-path metadata query. The returned vector is intentional:
    /// callers can retain it without borrowing the runtime during callbacks.
    pub fn sparse_pattern(&self) -> Vec<(usize, usize)> {
        self.plan
            .entries
            .iter()
            .map(|entry| (entry.row, entry.col))
            .collect()
    }

    /// Returns `(lower, upper, compact_data_len)` for a prepared banded plan.
    pub fn banded_layout(&self) -> Option<(usize, usize, usize)> {
        match &self.storage {
            NativePreparedJacobianStorage::Banded(layout) => {
                Some((layout.kl, layout.ku, layout.data_len))
            }
            _ => None,
        }
    }

    fn storage_name(&self) -> &'static str {
        match &self.storage {
            NativePreparedJacobianStorage::Dense => "Dense",
            NativePreparedJacobianStorage::SparseTriplets => "SparseTriplets",
            NativePreparedJacobianStorage::Banded(_) => "Banded",
        }
    }

    /// Evaluates a dense Jacobian directly into caller-owned matrix storage.
    pub fn try_evaluate_dense_into(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        out: &mut DMatrix<f64>,
    ) -> Result<(), IvpBackendError> {
        if !matches!(&self.storage, NativePreparedJacobianStorage::Dense) {
            self.telemetry.record_error();
            return Err(IvpBackendError::InvalidJacobianStorage {
                expected: "Dense".to_string(),
                actual: self.storage_name().to_string(),
            });
        }
        if out.nrows() != self.plan.rows || out.ncols() != self.plan.cols {
            self.telemetry.record_error();
            return Err(IvpBackendError::InvalidMatrixShape {
                stage: "dense Jacobian".to_string(),
                expected_rows: self.plan.rows,
                expected_cols: self.plan.cols,
                actual_rows: out.nrows(),
                actual_cols: out.ncols(),
            });
        }

        prepare_native_atom_values_buffer(
            &mut self.workspace.values,
            self.plan.entries.len(),
            &self.telemetry,
        );
        evaluate_native_atom_values_into(
            &self.plan,
            t,
            y,
            self.parameter_values_handle.as_ref(),
            &self.telemetry,
            self.execution_policy,
            &mut self.workspace.values,
        )?;
        let output_started = self
            .telemetry
            .start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
        out.fill(0.0);
        for (entry, value) in self.plan.entries.iter().zip(self.workspace.values.iter()) {
            out[(entry.row, entry.col)] = *value;
        }
        self.telemetry
            .record_copy_bytes(self.plan.entries.len() * std::mem::size_of::<f64>());
        self.telemetry
            .record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
        Ok(())
    }

    /// Evaluates sparse Jacobian values in the prepared fixed pattern order.
    pub fn try_evaluate_sparse_values_into(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        out: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        if !matches!(&self.storage, NativePreparedJacobianStorage::SparseTriplets) {
            self.telemetry.record_error();
            return Err(IvpBackendError::InvalidJacobianStorage {
                expected: "SparseTriplets".to_string(),
                actual: self.storage_name().to_string(),
            });
        }
        if out.len() != self.plan.entries.len() {
            self.telemetry.record_error();
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "sparse Jacobian values".to_string(),
                expected: self.plan.entries.len(),
                actual: out.len(),
            });
        }

        // Sparse values already use the prepared entry order. Evaluate
        // directly into the caller buffer instead of copying through scratch.
        evaluate_native_atom_values_into(
            &self.plan,
            t,
            y,
            self.parameter_values_handle.as_ref(),
            &self.telemetry,
            self.execution_policy,
            out,
        )?;
        let output_started = self
            .telemetry
            .start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
        self.telemetry
            .record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
        Ok(())
    }

    /// Evaluates a compact banded Jacobian into caller-owned storage.
    pub fn try_evaluate_banded_values_into(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        out: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        let layout = match &self.storage {
            NativePreparedJacobianStorage::Banded(layout) => layout,
            _ => {
                self.telemetry.record_error();
                return Err(IvpBackendError::InvalidJacobianStorage {
                    expected: "Banded".to_string(),
                    actual: self.storage_name().to_string(),
                });
            }
        };
        if out.len() != layout.data_len {
            self.telemetry.record_error();
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "banded Jacobian values".to_string(),
                expected: layout.data_len,
                actual: out.len(),
            });
        }

        prepare_native_atom_values_buffer(
            &mut self.workspace.values,
            self.plan.entries.len(),
            &self.telemetry,
        );
        evaluate_native_atom_values_into(
            &self.plan,
            t,
            y,
            self.parameter_values_handle.as_ref(),
            &self.telemetry,
            self.execution_policy,
            &mut self.workspace.values,
        )?;
        let output_started = self
            .telemetry
            .start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
        out.fill(0.0);
        for (slot, value) in layout.slots.iter().zip(self.workspace.values.iter()) {
            out[*slot] = *value;
        }
        self.telemetry
            .record_copy_bytes(self.plan.entries.len() * std::mem::size_of::<f64>());
        self.telemetry
            .record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
        Ok(())
    }

    /// Materializes the historical solver-facing Jacobian enum.
    fn try_evaluate(&mut self, t: f64, y: &DVector<f64>) -> Result<BdfJacobian, IvpBackendError> {
        match &self.storage {
            NativePreparedJacobianStorage::Dense => {
                let mut matrix = DMatrix::zeros(self.plan.rows, self.plan.cols);
                self.try_evaluate_dense_into(t, y, &mut matrix)?;
                Ok(BdfJacobian::Dense(matrix))
            }
            NativePreparedJacobianStorage::SparseTriplets => {
                let mut values = vec![0.0; self.plan.entries.len()];
                self.try_evaluate_sparse_values_into(t, y, &mut values)?;
                let triplets = self
                    .plan
                    .entries
                    .iter()
                    .zip(values)
                    .map(|(entry, value)| Triplet::new(entry.row, entry.col, value))
                    .collect::<Vec<_>>();
                self.telemetry.record_allocation(
                    triplets.len() * std::mem::size_of::<Triplet<usize, usize, f64>>(),
                );
                Ok(BdfJacobian::SparseTriplets {
                    n: self.plan.rows,
                    triplets,
                })
            }
            NativePreparedJacobianStorage::Banded(layout) => {
                let (kl, ku, data_len) = (layout.kl, layout.ku, layout.data_len);
                let mut data = vec![0.0; data_len];
                self.try_evaluate_banded_values_into(t, y, &mut data)?;
                let banded =
                    Banded::<f64>::from_vec(self.plan.rows, kl, ku, data).map_err(|error| {
                        self.telemetry.record_error();
                        IvpBackendError::AtomEvaluationFailure {
                            stage: "banded Jacobian output assembly".to_string(),
                            index: 0,
                            message: format!("{error:?}"),
                        }
                    })?;
                Ok(BdfJacobian::Banded(banded))
            }
        }
    }
}

/// Builds a native Jacobian evaluator from symbolic IVP equations.
///
/// The evaluator uses the same argument order as the IVP lambdify layer:
/// `t, y0, y1, ...`.
pub fn compile_native_symbolic_jacobian(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    storage: NativeJacobianStorage,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    compile_native_symbolic_jacobian_with_parameters(
        equations, variables, time_arg, None, None, storage,
    )
}

/// Builds a native symbolic Jacobian evaluator with IVP parameter support.
///
/// Parameterized evaluators mirror the shared IVP lambdify/AOT input order:
/// `t, params..., y...`.
pub fn compile_native_symbolic_jacobian_with_parameters(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
    storage: NativeJacobianStorage,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    let parameter_values_handle =
        equation_parameter_values.map(|values| Arc::new(RwLock::new(values)));
    compile_native_symbolic_jacobian_with_parameter_handle(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
    )
}

/// Builds a native symbolic Jacobian evaluator backed by shared parameter values.
///
/// Residual and native Jacobian can use the same parameter storage, preventing
/// stale Newton matrices when a prepared solver updates parameters.
pub fn compile_native_symbolic_jacobian_with_parameter_handle(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpTelemetry::disabled(),
    )
}

/// Telemetry-aware variant used by the native Lambdify route. The original
/// function remains a compatibility wrapper for callers that do not collect
/// diagnostics.
pub fn compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
    telemetry: IvpTelemetry,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry_and_policy(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
        symbolic_assembly_backend,
        telemetry,
        IvpLambdifyExecutionPolicy::default(),
    )
}

/// Telemetry-aware native Jacobian builder with explicit callback dispatch.
pub fn compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry_and_policy(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
    if symbolic_assembly_backend == IvpSymbolicAssemblyBackend::AtomView {
        return compile_native_atom_jacobian(
            equations,
            variables,
            time_arg,
            equation_parameters,
            parameter_values_handle,
            storage,
            telemetry,
            execution_policy,
        )
        .expect("native AtomView Jacobian preparation should succeed");
    }

    let symbolic_started = telemetry.start_cold_stage(IvpColdStage::SymbolicJacobian);
    let symbolic_jacobian = crate::symbolic::symbolic_ivp::build_symbolic_jacobian(
        equations,
        variables,
        symbolic_assembly_backend,
        &telemetry,
    );
    telemetry.record_cold_stage(IvpColdStage::SymbolicJacobian, symbolic_started);
    telemetry.record_symbolic_jacobian_build();
    validate_parameter_handle(equation_parameters, parameter_values_handle.as_ref());

    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |parameters| parameters.len()),
    );
    names.push(time_arg.to_string());
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().cloned());
    }
    names.extend(variables.iter().cloned());
    let name_refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();

    let rows = symbolic_jacobian.len();
    let cols = symbolic_jacobian.first().map_or(0, |row| row.len());
    let compilation_started = telemetry.start_cold_stage(IvpColdStage::JacobianCompilation);
    let entries = compile_nonzero_entries(&symbolic_jacobian, &name_refs, &telemetry);
    telemetry.record_cold_stage(IvpColdStage::JacobianCompilation, compilation_started);

    match storage {
        NativeJacobianStorage::Dense => Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
            let callback_started = telemetry.start_warm_stage(IvpWarmStage::JacobianCallback);
            let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
            let output_started = telemetry.start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
            let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
            let args = build_args(t, parameter_values.as_ref(), y);
            telemetry.record_copy_bytes(args.len() * std::mem::size_of::<f64>());
            telemetry.record_allocation(args.capacity() * std::mem::size_of::<f64>());
            telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
            let evaluation_started = telemetry.start_warm_stage(IvpWarmStage::JacobianEvaluation);
            let mut matrix = nalgebra::DMatrix::<f64>::zeros(rows, cols);
            for (row, col, eval) in &entries {
                matrix[(*row, *col)] = eval(&args);
            }
            telemetry.record_warm_stage(IvpWarmStage::JacobianEvaluation, evaluation_started);
            telemetry.record_scalar_evaluations(entries.len());
            telemetry.record_allocation(rows * cols * std::mem::size_of::<f64>());
            telemetry.record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
            telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
            telemetry.record_jacobian_evaluation_count();
            BdfJacobian::Dense(matrix)
        }),
        NativeJacobianStorage::SparseTriplets => {
            Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                let callback_started = telemetry.start_warm_stage(IvpWarmStage::JacobianCallback);
                let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
                let output_started =
                    telemetry.start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
                let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
                let args = build_args(t, parameter_values.as_ref(), y);
                telemetry.record_copy_bytes(args.len() * std::mem::size_of::<f64>());
                telemetry.record_allocation(args.capacity() * std::mem::size_of::<f64>());
                telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                let evaluation_started =
                    telemetry.start_warm_stage(IvpWarmStage::JacobianEvaluation);
                telemetry.record_scalar_evaluations(entries.len());
                let triplets = entries
                    .iter()
                    .map(|(row, col, eval)| Triplet::new(*row, *col, eval(&args)))
                    .collect::<Vec<_>>();
                telemetry.record_warm_stage(IvpWarmStage::JacobianEvaluation, evaluation_started);
                telemetry.record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
                telemetry.record_allocation(
                    triplets.len() * std::mem::size_of::<Triplet<usize, usize, f64>>(),
                );
                telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
                telemetry.record_jacobian_evaluation_count();
                BdfJacobian::SparseTriplets { n: rows, triplets }
            })
        }
        NativeJacobianStorage::Banded { bandwidth } => {
            let (kl, ku) = bandwidth.unwrap_or_else(|| infer_bandwidth(rows, cols, &entries));
            Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
                let callback_started = telemetry.start_warm_stage(IvpWarmStage::JacobianCallback);
                let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
                let output_started =
                    telemetry.start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
                let parameter_values = read_parameter_values(parameter_values_handle.as_ref());
                let args = build_args(t, parameter_values.as_ref(), y);
                telemetry.record_copy_bytes(args.len() * std::mem::size_of::<f64>());
                telemetry.record_allocation(args.capacity() * std::mem::size_of::<f64>());
                telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                let evaluation_started =
                    telemetry.start_warm_stage(IvpWarmStage::JacobianEvaluation);
                let mut banded = Banded::<f64>::zeros(rows, kl, ku)
                    .expect("symbolic Jacobian bandwidth should define valid banded storage");
                for (row, col, eval) in &entries {
                    banded
                        .set(*row, *col, eval(&args))
                        .expect("compiled symbolic entry must fit inferred bandwidth");
                }
                telemetry.record_warm_stage(IvpWarmStage::JacobianEvaluation, evaluation_started);
                telemetry.record_scalar_evaluations(entries.len());
                telemetry.record_allocation(rows * (kl + ku + 1) * std::mem::size_of::<f64>());
                telemetry.record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
                telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
                telemetry.record_jacobian_evaluation_count();
                BdfJacobian::Banded(banded)
            })
        }
    }
}

/// Fallible native AtomView constructor used by the LSODE2 preparation path.
///
/// The older generic constructor below keeps its infallible callback contract
/// for compatibility. New native callers should use this boundary so invalid
/// parameter schemas and symbolic preparation failures remain typed.
pub fn try_compile_native_atomview_jacobian_with_parameter_handle_and_telemetry_and_policy(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Result<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>, IvpBackendError> {
    compile_native_atom_jacobian(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
        telemetry,
        execution_policy,
    )
}

/// Compiles the native AtomView Jacobian and preserves typed runtime errors.
///
/// The existing `try_compile_native_atomview_jacobian_*` function keeps the
/// historical infallible solver callback signature. New integrations should
/// prefer this function and decide explicitly how a callback error is mapped
/// at their own numerical boundary.
pub fn try_compile_native_atomview_jacobian_callback(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Result<Box<NativeJacobianCallback>, IvpBackendError> {
    compile_native_atom_jacobian_typed(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
        telemetry,
        execution_policy,
    )
}

fn prepare_native_atom_values_buffer(values: &mut Vec<f64>, len: usize, telemetry: &IvpTelemetry) {
    let capacity_before = values.capacity();
    values.resize(len, 0.0);
    if values.capacity() > capacity_before {
        telemetry
            .record_allocation((values.capacity() - capacity_before) * std::mem::size_of::<f64>());
    }
}

fn evaluate_native_atom_values_into(
    plan: &Arc<PreparedNativeAtomJacobian>,
    t: f64,
    y: &DVector<f64>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
    telemetry: &IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
    values: &mut [f64],
) -> Result<(), IvpBackendError> {
    let callback_started = telemetry.start_warm_stage(IvpWarmStage::JacobianCallback);
    if y.len() != plan.cols {
        telemetry.record_error();
        telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
        return Err(IvpBackendError::InvalidStateShape {
            expected: plan.cols,
            actual: y.len(),
        });
    }
    let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
    if values.len() != plan.entries.len() {
        telemetry.record_error();
        telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
        telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
        return Err(IvpBackendError::InvalidOutputShape {
            stage: "native Jacobian values".to_string(),
            expected: plan.entries.len(),
            actual: values.len(),
        });
    }
    let mut binding_recorded = false;
    let evaluation = with_shared_parameter_values(
        parameter_values_handle,
        || {
            telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
            binding_recorded = true;
        },
        |parameters| {
            let parallel =
                execution_policy.should_parallel_with_tasks(plan.entries.len(), plan.entries.len());
            telemetry.record_lambdify_dispatch(parallel);
            telemetry.record_scalar_evaluations(plan.entries.len());
            let evaluation_started = telemetry.start_warm_stage(IvpWarmStage::JacobianEvaluation);
            let evaluation = if parallel {
                let worker_count = rayon::current_num_threads().max(1);
                let chunk_size = (plan.entries.len() + worker_count - 1)
                    .checked_div(worker_count)
                    .unwrap_or(1)
                    .max(1);
                plan.entries
                    .par_chunks(chunk_size)
                    .zip(values.par_chunks_mut(chunk_size))
                    .enumerate()
                    .try_for_each(|(chunk_index, (entries, values))| {
                        PreparedEvaluator::evaluate_many_thread_local_ivp(
                            entries.iter().map(|entry| &entry.evaluator),
                            t,
                            parameters,
                            y.as_slice(),
                            plan.expected_input_len,
                            values,
                        )
                        .map_err(|(index, message)| {
                            IvpBackendError::AtomEvaluationFailure {
                                stage: "Jacobian".to_string(),
                                index: chunk_index * chunk_size + index,
                                message,
                            }
                        })
                    })
            } else {
                PreparedEvaluator::evaluate_many_thread_local_ivp(
                    plan.entries.iter().map(|entry| &entry.evaluator),
                    t,
                    parameters,
                    y.as_slice(),
                    plan.expected_input_len,
                    values,
                )
                .map_err(|(index, message)| {
                    IvpBackendError::AtomEvaluationFailure {
                        stage: "Jacobian".to_string(),
                        index,
                        message,
                    }
                })
            };
            telemetry.record_warm_stage(IvpWarmStage::JacobianEvaluation, evaluation_started);
            evaluation
        },
    );
    if !binding_recorded {
        telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
    }
    if let Err(error) = evaluation {
        // Close the inclusive callback scope on every typed failure. Without
        // this branch, evaluator and poisoned-parameter failures disappear
        // from both the error counter and the stage accounting.
        telemetry.record_error();
        telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
        return Err(error);
    }
    telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
    telemetry.record_jacobian_evaluation_count();
    Ok(())
}

fn compile_native_atom_jacobian(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Result<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>, IvpBackendError> {
    let mut callback = compile_native_atom_jacobian_typed(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
        telemetry,
        execution_policy,
    )?;
    Ok(Box::new(move |t, y| match callback(t, y) {
        Ok(jacobian) => jacobian,
        Err(error) => {
            log::warn!(
                target: "rusted_scithe::lsode2::native_jacobian",
                "native AtomView Jacobian compatibility callback failed: {error}"
            );
            BdfJacobian::Dense(DMatrix::from_element(0, 0, f64::NAN))
        }
    }))
}

fn compile_native_atom_jacobian_typed(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Result<Box<NativeJacobianCallback>, IvpBackendError> {
    let mut runtime = try_prepare_native_atomview_jacobian_runtime(
        equations,
        variables,
        time_arg,
        equation_parameters,
        parameter_values_handle,
        storage,
        telemetry,
        execution_policy,
    )?;
    Ok(Box::new(move |t, y| runtime.try_evaluate(t, y)))
}

/// Prepares the native Jacobian plan and its reusable callback workspace.
///
/// This is intentionally separate from solver-facing callback materialization:
/// caller-owned sparse and banded APIs can now reuse the same prepared plan
/// without allocating a temporary `BdfJacobian` on every evaluation.
pub fn try_prepare_native_atomview_jacobian_runtime(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Result<NativeAtomJacobianRuntime, IvpBackendError> {
    try_validate_parameter_handle(equation_parameters, parameter_values_handle.as_ref())?;
    let symbolic_started = telemetry.start_cold_stage(IvpColdStage::SymbolicJacobian);
    let plan = prepare_native_atom_jacobian(
        equations,
        variables,
        time_arg,
        equation_parameters,
        &telemetry,
    )?;
    telemetry.record_cold_stage(IvpColdStage::SymbolicJacobian, symbolic_started);
    telemetry.record_symbolic_jacobian_build();
    let plan = Arc::new(plan);

    let storage = match storage {
        NativeJacobianStorage::Dense => NativePreparedJacobianStorage::Dense,
        NativeJacobianStorage::SparseTriplets => NativePreparedJacobianStorage::SparseTriplets,
        NativeJacobianStorage::Banded { bandwidth } => {
            let (kl, ku) = bandwidth.unwrap_or_else(|| {
                let mut kl = 0usize;
                let mut ku = 0usize;
                for entry in plan.entries.iter() {
                    kl = kl.max(entry.row.saturating_sub(entry.col));
                    ku = ku.max(entry.col.saturating_sub(entry.row));
                }
                (kl, ku)
            });
            if plan.rows != plan.cols {
                return Err(IvpBackendError::InvalidBandedStorage {
                    rows: plan.rows,
                    cols: plan.cols,
                    lower: kl,
                    upper: ku,
                });
            }
            let data_len = checked_banded_data_len(plan.rows, kl, ku)?;
            let slots = prepare_banded_slots(&plan, kl, ku)?;
            NativePreparedJacobianStorage::Banded(NativeBandedLayout {
                kl,
                ku,
                data_len,
                slots: slots.into(),
            })
        }
    };

    let values_capacity = plan.entries.len();
    Ok(NativeAtomJacobianRuntime {
        plan,
        storage,
        parameter_values_handle,
        telemetry,
        execution_policy,
        workspace: NativeJacobianWorkspace::with_capacity(values_capacity),
    })
}

/// Builds a native Jacobian evaluator from compiled sparse AOT callbacks.
///
/// This path keeps LSODE2 sparse/banded Jacobian evaluation in the AOT branch
/// instead of falling back to lambdified symbolic Jacobian entries.
pub fn compile_native_sparse_aot_jacobian_with_parameter_handle(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    generated_backend: SymbolicIvpGeneratedBackendConfig,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
) -> Result<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>, IvpBackendError> {
    compile_native_sparse_aot_jacobian_with_parameter_handle_and_telemetry(
        equations,
        variables,
        time_arg,
        equation_parameters,
        equation_parameter_values,
        parameter_values_handle,
        storage,
        generated_backend,
        symbolic_assembly_backend,
        IvpTelemetry::disabled(),
    )
}

/// Telemetry-aware variant used by the native LSODE2 runtime. The infallible
/// return type is retained only at the solver compatibility boundary; the
/// underlying callback remains fallible and reports typed failures before the
/// adapter converts them to the historical NaN matrix.
pub fn compile_native_sparse_aot_jacobian_with_parameter_handle_and_telemetry(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    generated_backend: SymbolicIvpGeneratedBackendConfig,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
    telemetry: IvpTelemetry,
) -> Result<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>, IvpBackendError> {
    let mut callback = compile_native_sparse_aot_jacobian_callback_with_telemetry(
        equations,
        variables,
        time_arg,
        equation_parameters,
        equation_parameter_values,
        parameter_values_handle,
        storage,
        generated_backend,
        symbolic_assembly_backend,
        telemetry,
    )?;
    Ok(Box::new(move |t, y| match callback(t, y) {
        Ok(jacobian) => jacobian,
        Err(error) => {
            log::warn!(
                target: "rusted_scithe::lsode2::native_jacobian",
                "compiled AOT Jacobian compatibility callback failed: {error}"
            );
            BdfJacobian::Dense(DMatrix::from_element(0, 0, f64::NAN))
        }
    }))
}

/// Fallible compiled AOT Jacobian callback used by new solver integrations.
///
/// The callback owns its argument and value buffers and reuses them across
/// invocations. The historical solver-facing function above intentionally
/// remains an infallible compatibility adapter; new code should keep this
/// typed boundary so malformed linked output, poisoned parameters and invalid
/// state shapes cannot become a silent NaN matrix.
pub fn compile_native_sparse_aot_jacobian_callback(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    generated_backend: SymbolicIvpGeneratedBackendConfig,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
) -> Result<Box<NativeJacobianCallback>, IvpBackendError> {
    compile_native_sparse_aot_jacobian_callback_with_telemetry(
        equations,
        variables,
        time_arg,
        equation_parameters,
        equation_parameter_values,
        parameter_values_handle,
        storage,
        generated_backend,
        symbolic_assembly_backend,
        IvpTelemetry::disabled(),
    )
}

/// Fallible compiled AOT Jacobian callback with opt-in stage telemetry.
pub fn compile_native_sparse_aot_jacobian_callback_with_telemetry(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    generated_backend: SymbolicIvpGeneratedBackendConfig,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
    telemetry: IvpTelemetry,
) -> Result<Box<NativeJacobianCallback>, IvpBackendError> {
    // AtomView-native AOT publishes residual and Jacobian symbols in one
    // combined artifact. Reusing the exact backend identity here is what lets
    // LSODE2 reconnect both callbacks from one resolver entry. The `_sj`
    // suffix remains only for the historical ExprLegacy compatibility route,
    // where the old separate Jacobian artifact contract is still supported.
    let generated_backend =
        lsode2_sparse_jacobian_artifact_config(generated_backend, symbolic_assembly_backend);
    let handoff_path = generated_backend.handoff_path.clone();
    let options = SymbolicIvpProblemOptions::new()
        .with_equation_parameters(equation_parameters.unwrap_or(&[]).to_vec())
        .with_equation_parameter_values(
            equation_parameter_values.unwrap_or_else(|| DVector::zeros(0)),
        )
        .with_symbolic_assembly_backend(symbolic_assembly_backend)
        .with_telemetry(telemetry.clone());

    let prepared = match storage {
        NativeJacobianStorage::Banded {
            bandwidth: Some((kl, ku)),
        } if symbolic_assembly_backend == IvpSymbolicAssemblyBackend::AtomView => {
            prepare_generated_symbolic_ivp_banded_backend(
                equations.to_vec(),
                variables.to_vec(),
                time_arg.to_string(),
                (kl, ku),
                options,
                generated_backend,
            )
        }
        _ => prepare_generated_symbolic_ivp_sparse_backend(
            equations.to_vec(),
            variables.to_vec(),
            time_arg.to_string(),
            options,
            generated_backend,
        ),
    }
    .map_err(|err| IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    })?;

    if let Some(handoff_path) = handoff_path.as_ref() {
        let resolver = prepared.updated_resolver.as_ref().ok_or_else(|| {
            IvpBackendError::GeneratedBackendFailure {
                message: "AOT producer did not publish an updated resolver".to_string(),
            }
        })?;
        resolver.merge_handoff(handoff_path).map_err(|err| {
            IvpBackendError::GeneratedBackendFailure {
                message: format!(
                    "AOT producer handoff publication failed at {}: {err}",
                    handoff_path.display()
                ),
            }
        })?;
    }

    let linked = prepared
        .linked_backend
        .ok_or_else(|| IvpBackendError::GeneratedBackendFailure {
            message:
                "LSODE2 sparse/banded AOT Jacobian path selected compiled backend but no runtime link is available"
                    .to_string(),
        })?;

    if prepared.selected_backend != SelectedSymbolicIvpBackendKind::AotCompiled {
        return Err(IvpBackendError::GeneratedBackendFailure {
            message: "LSODE2 sparse/banded AOT Jacobian path expected a compiled sparse backend"
                .to_string(),
        });
    }

    compile_native_sparse_aot_jacobian_from_linked_backend(
        linked,
        prepared.jacobian_structure,
        equation_parameters,
        parameter_values_handle,
        storage,
        telemetry,
    )
}

/// Build the solver-facing Jacobian callback from an already linked combined
/// AtomView artifact.  This is intentionally separate from the orchestration
/// function above so LSODE2 can reuse the backend prepared by the residual
/// lifecycle without repeating symbolic/layout/build work.
pub(crate) fn compile_native_sparse_aot_jacobian_from_linked_backend(
    linked: LinkedSparseAotBackend,
    jacobian_structure: SparseJacobianStructure,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    telemetry: IvpTelemetry,
) -> Result<Box<NativeJacobianCallback>, IvpBackendError> {
    try_validate_parameter_handle(equation_parameters, parameter_values_handle.as_ref())?;

    let rows = jacobian_structure.rows;
    let cols = jacobian_structure.cols;
    let pattern = jacobian_structure
        .row_indices
        .iter()
        .copied()
        .zip(jacobian_structure.col_indices.iter().copied())
        .collect::<Vec<_>>();

    match storage {
        NativeJacobianStorage::Dense => {
            let mut args =
                Vec::with_capacity(1 + equation_parameters.map_or(0, <[String]>::len) + cols);
            let mut values = vec![0.0_f64; pattern.len()];
            let callback_telemetry = telemetry.clone();
            Ok(Box::new(
                move |t: f64, y: &DVector<f64>| -> Result<BdfJacobian, IvpBackendError> {
                    let _callback_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::JacobianCallback);
                    callback_telemetry.record_scalar_evaluations(pattern.len());
                    if y.len() != cols {
                        return Err(IvpBackendError::InvalidStateShape {
                            expected: cols,
                            actual: y.len(),
                        });
                    }
                    let _binding_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::ArgumentBinding);
                    let parameter_values =
                        try_read_parameter_values(parameter_values_handle.as_ref())?;
                    build_args_into(t, parameter_values.as_ref(), y, &mut args);
                    drop(_binding_scope);
                    let _evaluation_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::JacobianEvaluation);
                    linked
                        .try_jacobian_values_eval(args.as_slice(), values.as_mut_slice())
                        .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                            message: error.to_string(),
                        })?;
                    drop(_evaluation_scope);
                    let _output_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::JacobianOutputAssembly);
                    callback_telemetry.record_allocation(rows * cols * std::mem::size_of::<f64>());
                    let mut matrix = nalgebra::DMatrix::<f64>::zeros(rows, cols);
                    for ((row, col), value) in pattern.iter().zip(values.iter().copied()) {
                        matrix[(*row, *col)] = value;
                    }
                    callback_telemetry.record_jacobian_evaluation_count();
                    Ok(BdfJacobian::Dense(matrix))
                },
            ))
        }
        NativeJacobianStorage::SparseTriplets => {
            let mut args =
                Vec::with_capacity(1 + equation_parameters.map_or(0, <[String]>::len) + cols);
            let mut values = vec![0.0_f64; pattern.len()];
            let callback_telemetry = telemetry.clone();
            Ok(Box::new(
                move |t: f64, y: &DVector<f64>| -> Result<BdfJacobian, IvpBackendError> {
                    let _callback_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::JacobianCallback);
                    callback_telemetry.record_scalar_evaluations(pattern.len());
                    if y.len() != cols {
                        return Err(IvpBackendError::InvalidStateShape {
                            expected: cols,
                            actual: y.len(),
                        });
                    }
                    let _binding_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::ArgumentBinding);
                    let parameter_values =
                        try_read_parameter_values(parameter_values_handle.as_ref())?;
                    build_args_into(t, parameter_values.as_ref(), y, &mut args);
                    drop(_binding_scope);
                    let _evaluation_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::JacobianEvaluation);
                    linked
                        .try_jacobian_values_eval(args.as_slice(), values.as_mut_slice())
                        .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                            message: error.to_string(),
                        })?;
                    drop(_evaluation_scope);
                    let _output_scope =
                        callback_telemetry.scoped_warm_stage(IvpWarmStage::JacobianOutputAssembly);
                    callback_telemetry.record_allocation(
                        pattern.len() * std::mem::size_of::<Triplet<usize, usize, f64>>(),
                    );
                    let triplets = pattern
                        .iter()
                        .zip(values.iter().copied())
                        .map(|((row, col), value)| Triplet::new(*row, *col, value))
                        .collect::<Vec<_>>();
                    callback_telemetry.record_jacobian_evaluation_count();
                    Ok(BdfJacobian::SparseTriplets { n: rows, triplets })
                },
            ))
        }
        NativeJacobianStorage::Banded { bandwidth } => {
            let (kl, ku) =
                bandwidth.unwrap_or_else(|| infer_bandwidth_from_pattern(rows, cols, &pattern));
            match linked.jacobian_layout {
                LinkedJacobianLayout::BandedCompact {
                    rows: linked_rows,
                    cols: linked_cols,
                    kl: linked_kl,
                    ku: linked_ku,
                } => {
                    if (linked_rows, linked_cols, linked_kl, linked_ku) != (rows, cols, kl, ku) {
                        return Err(IvpBackendError::GeneratedBackendFailure {
                            message: format!(
                                "linked compact Banded layout mismatch: callback=({linked_rows}x{linked_cols}, kl={linked_kl}, ku={linked_ku}), requested=({rows}x{cols}, kl={kl}, ku={ku})"
                            ),
                        });
                    }
                    let mut args = Vec::with_capacity(
                        1 + equation_parameters.map_or(0, <[String]>::len) + cols,
                    );
                    let mut values = vec![0.0_f64; (kl + ku + 1) * rows];
                    let callback_telemetry = telemetry.clone();
                    Ok(Box::new(
                        move |t: f64, y: &DVector<f64>| -> Result<BdfJacobian, IvpBackendError> {
                            let _callback_scope = callback_telemetry
                                .scoped_warm_stage(IvpWarmStage::JacobianCallback);
                            callback_telemetry.record_jacobian_request();
                            callback_telemetry.record_scalar_evaluations(values.len());
                            if y.len() != cols {
                                return Err(IvpBackendError::InvalidStateShape {
                                    expected: cols,
                                    actual: y.len(),
                                });
                            }
                            let _binding_scope =
                                callback_telemetry.scoped_warm_stage(IvpWarmStage::ArgumentBinding);
                            let parameter_values =
                                try_read_parameter_values(parameter_values_handle.as_ref())?;
                            build_args_into(t, parameter_values.as_ref(), y, &mut args);
                            drop(_binding_scope);
                            let output_len = (kl + ku + 1) * rows;
                            let _evaluation_scope = callback_telemetry
                                .scoped_warm_stage(IvpWarmStage::JacobianEvaluation);
                            linked
                                .try_jacobian_values_eval(&args, &mut values)
                                .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                                    message: error.to_string(),
                                })?;
                            drop(_evaluation_scope);
                            let _output_scope = callback_telemetry
                                .scoped_warm_stage(IvpWarmStage::JacobianOutputAssembly);
                            callback_telemetry
                                .record_allocation(output_len * std::mem::size_of::<f64>());
                            callback_telemetry
                                .record_copy_bytes(values.len() * std::mem::size_of::<f64>());
                            if values.len() != output_len {
                                return Err(IvpBackendError::InvalidOutputShape {
                                    stage: "compiled AOT compact Banded Jacobian".to_string(),
                                    expected: output_len,
                                    actual: values.len(),
                                });
                            }
                            let mut banded =
                                Banded::<f64>::zeros(rows, kl, ku).map_err(|error| {
                                    IvpBackendError::GeneratedBackendFailure {
                                        message: format!(
                                            "compiled AOT compact Banded output is invalid: {error}"
                                        ),
                                    }
                                })?;
                            banded.as_mut_slice().copy_from_slice(&values);
                            callback_telemetry.record_jacobian_evaluation_count();
                            Ok(BdfJacobian::Banded(banded))
                        },
                    ))
                }
                LinkedJacobianLayout::ExplicitValues => {
                    let mut args = Vec::with_capacity(
                        1 + equation_parameters.map_or(0, <[String]>::len) + cols,
                    );
                    let mut values = vec![0.0_f64; pattern.len()];
                    let callback_telemetry = telemetry.clone();
                    Ok(Box::new(
                        move |t: f64, y: &DVector<f64>| -> Result<BdfJacobian, IvpBackendError> {
                            let _callback_scope = callback_telemetry
                                .scoped_warm_stage(IvpWarmStage::JacobianCallback);
                            callback_telemetry.record_jacobian_request();
                            callback_telemetry.record_scalar_evaluations(values.len());
                            if y.len() != cols {
                                return Err(IvpBackendError::InvalidStateShape {
                                    expected: cols,
                                    actual: y.len(),
                                });
                            }
                            let _binding_scope =
                                callback_telemetry.scoped_warm_stage(IvpWarmStage::ArgumentBinding);
                            let parameter_values =
                                try_read_parameter_values(parameter_values_handle.as_ref())?;
                            build_args_into(t, parameter_values.as_ref(), y, &mut args);
                            drop(_binding_scope);
                            let _evaluation_scope = callback_telemetry
                                .scoped_warm_stage(IvpWarmStage::JacobianEvaluation);
                            linked
                                .try_jacobian_values_eval(&args, values.as_mut_slice())
                                .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                                    message: error.to_string(),
                                })?;
                            drop(_evaluation_scope);
                            let _output_scope = callback_telemetry
                                .scoped_warm_stage(IvpWarmStage::JacobianOutputAssembly);
                            callback_telemetry.record_allocation(
                                rows * (kl + ku + 1) * std::mem::size_of::<f64>(),
                            );
                            let mut banded =
                                Banded::<f64>::zeros(rows, kl, ku).map_err(|error| {
                                    IvpBackendError::GeneratedBackendFailure {
                                        message: format!(
                                            "compiled AOT Banded output storage is invalid: {error}"
                                        ),
                                    }
                                })?;
                            for ((row, col), value) in pattern.iter().zip(values.iter().copied()) {
                                banded.set(*row, *col, value).map_err(|error| {
                                IvpBackendError::GeneratedBackendFailure {
                                    message: format!(
                                        "compiled AOT sparse value does not fit Banded storage: {error}"
                                    ),
                                }
                            })?;
                            }
                            callback_telemetry.record_jacobian_evaluation_count();
                            Ok(BdfJacobian::Banded(banded))
                        },
                    ))
                }
            }
        }
    }
}

/// Compatibility adapter for the historical solver-facing infallible Jacobian
/// callback.  Preparation is still shared; only the final callback ABI keeps
/// its legacy NaN-on-error behavior.
pub(crate) fn compile_native_sparse_aot_jacobian_from_linked_backend_compat(
    linked: LinkedSparseAotBackend,
    jacobian_structure: SparseJacobianStructure,
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    storage: NativeJacobianStorage,
    telemetry: IvpTelemetry,
) -> Result<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>, IvpBackendError> {
    let mut callback = compile_native_sparse_aot_jacobian_from_linked_backend(
        linked,
        jacobian_structure,
        equation_parameters,
        parameter_values_handle,
        storage,
        telemetry,
    )?;
    Ok(Box::new(move |t, y| match callback(t, y) {
        Ok(jacobian) => jacobian,
        Err(error) => {
            log::warn!(
                target: "rusted_scithe::lsode2::native_jacobian",
                "compiled AOT Jacobian compatibility callback failed: {error}"
            );
            BdfJacobian::Dense(DMatrix::from_element(0, 0, f64::NAN))
        }
    }))
}

fn with_lsode2_sparse_jacobian_artifact_suffix(
    mut config: SymbolicIvpGeneratedBackendConfig,
) -> SymbolicIvpGeneratedBackendConfig {
    const SUFFIX: &str = "_sj";
    if let Some(crate_name) = config.crate_name_override.clone() {
        config.crate_name_override = Some(format!("{crate_name}{SUFFIX}"));
    }
    if let Some(module_name) = config.module_name_override.clone() {
        config.module_name_override = Some(format!("{module_name}{SUFFIX}"));
    }
    config
}

fn lsode2_sparse_jacobian_artifact_config(
    config: SymbolicIvpGeneratedBackendConfig,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
) -> SymbolicIvpGeneratedBackendConfig {
    if symbolic_assembly_backend == IvpSymbolicAssemblyBackend::AtomView {
        config
    } else {
        with_lsode2_sparse_jacobian_artifact_suffix(config)
    }
}

fn validate_parameter_handle(
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
) {
    try_validate_parameter_handle(equation_parameters, parameter_values_handle)
        .expect("native symbolic Jacobian parameter schema should be valid");
}

fn try_validate_parameter_handle(
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
) -> Result<(), IvpBackendError> {
    match (equation_parameters, parameter_values_handle) {
        (Some(parameters), Some(handle)) => {
            let values = handle
                .read()
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)?;
            if parameters.len() != values.len() {
                return Err(IvpBackendError::ParameterCountMismatch {
                    expected: parameters.len(),
                    actual: values.len(),
                });
            }
        }
        (Some(parameters), None) => {
            if !parameters.is_empty() {
                return Err(IvpBackendError::MissingParameterValues {
                    expected: parameters.len(),
                });
            }
        }
        (None, Some(handle)) => {
            let values = handle
                .read()
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)?;
            if !values.is_empty() {
                return Err(IvpBackendError::ParameterCountMismatch {
                    expected: 0,
                    actual: values.len(),
                });
            }
        }
        (None, None) => {}
    }
    Ok(())
}

fn read_parameter_values(handle: Option<&SharedIvpParameterValues>) -> Option<DVector<f64>> {
    handle.map(|handle| {
        handle
            .read()
            .expect("shared IVP parameter state lock poisoned")
            .clone()
    })
}

fn try_read_parameter_values(
    handle: Option<&SharedIvpParameterValues>,
) -> Result<Option<DVector<f64>>, IvpBackendError> {
    handle
        .map(|handle| {
            handle
                .read()
                .map(|values| values.clone())
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)
        })
        .transpose()
}

fn compile_nonzero_entries(
    symbolic_jacobian: &[Vec<Expr>],
    name_refs: &[&str],
    telemetry: &IvpTelemetry,
) -> Vec<CompiledEntry> {
    let lambdification_started = telemetry.start_cold_stage(IvpColdStage::JacobianLambdification);
    let mut entries = Vec::new();
    for (row, symbolic_row) in symbolic_jacobian.iter().enumerate() {
        for (col, expr) in symbolic_row.iter().enumerate() {
            if !expr.is_zero() {
                entries.push((
                    row,
                    col,
                    // Lambdification is the cold part of the native callback;
                    // the enclosing JacobianCompilation scope remains the
                    // compatibility aggregate for existing reports.
                    Expr::lambdify_borrowed_thread_safe(expr, name_refs),
                ));
            }
        }
    }
    telemetry.record_cold_stage(IvpColdStage::JacobianLambdification, lambdification_started);
    entries
}

fn infer_bandwidth(rows: usize, cols: usize, entries: &[CompiledEntry]) -> (usize, usize) {
    let mut kl = 0usize;
    let mut ku = 0usize;
    for (row, col, _) in entries {
        kl = kl.max(row.saturating_sub(*col));
        ku = ku.max(col.saturating_sub(*row));
    }
    if rows == cols && rows > 0 {
        (kl, ku)
    } else {
        (rows.saturating_sub(1), cols.saturating_sub(1))
    }
}

fn infer_bandwidth_from_pattern(
    rows: usize,
    cols: usize,
    pattern: &[(usize, usize)],
) -> (usize, usize) {
    let mut kl = 0usize;
    let mut ku = 0usize;
    for (row, col) in pattern {
        kl = kl.max(row.saturating_sub(*col));
        ku = ku.max(col.saturating_sub(*row));
    }
    if rows == cols && rows > 0 {
        (kl, ku)
    } else {
        (rows.saturating_sub(1), cols.saturating_sub(1))
    }
}

fn checked_banded_data_len(n: usize, kl: usize, ku: usize) -> Result<usize, IvpBackendError> {
    kl.checked_add(ku)
        .and_then(|bands| bands.checked_add(1))
        .and_then(|bands| bands.checked_mul(n))
        .ok_or_else(|| IvpBackendError::AtomEvaluationFailure {
            stage: "banded Jacobian layout preparation".to_string(),
            index: 0,
            message: format!("banded storage size overflowed for n={n}, kl={kl}, ku={ku}"),
        })
}

fn prepare_banded_slots(
    plan: &PreparedNativeAtomJacobian,
    kl: usize,
    ku: usize,
) -> Result<Vec<usize>, IvpBackendError> {
    plan.entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let in_band = entry.row < plan.rows
                && entry.col < plan.cols
                && entry.col <= entry.row.saturating_add(ku)
                && entry.row <= entry.col.saturating_add(kl);
            if !in_band {
                return Err(IvpBackendError::AtomEvaluationFailure {
                    stage: "banded Jacobian layout preparation".to_string(),
                    index,
                    message: format!(
                        "symbolic entry ({}, {}) is outside kl={}, ku={} band",
                        entry.row, entry.col, kl, ku
                    ),
                });
            }
            let band_row = ku
                .checked_add(entry.row)
                .and_then(|value| value.checked_sub(entry.col))
                .ok_or_else(|| IvpBackendError::AtomEvaluationFailure {
                    stage: "banded Jacobian layout preparation".to_string(),
                    index,
                    message: "band slot arithmetic overflowed".to_string(),
                })?;
            band_row
                .checked_mul(plan.rows)
                .and_then(|value| value.checked_add(entry.col))
                .ok_or_else(|| IvpBackendError::AtomEvaluationFailure {
                    stage: "banded Jacobian layout preparation".to_string(),
                    index,
                    message: "band slot arithmetic overflowed".to_string(),
                })
        })
        .collect()
}

fn build_args(t: f64, parameter_values: Option<&DVector<f64>>, y: &DVector<f64>) -> Vec<f64> {
    let mut args =
        Vec::with_capacity(1 + y.len() + parameter_values.map_or(0, |values| values.len()));
    args.push(t);
    if let Some(values) = parameter_values {
        args.extend(values.iter().copied());
    }
    args.extend(y.iter().copied());
    args
}

fn build_args_into(
    t: f64,
    parameter_values: Option<&DVector<f64>>,
    y: &DVector<f64>,
    args: &mut Vec<f64>,
) {
    args.clear();
    args.reserve(1 + y.len() + parameter_values.map_or(0, |values| values.len()));
    args.push(t);
    if let Some(values) = parameter_values {
        args.extend(values.iter().copied());
    }
    args.extend(y.iter().copied());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;

    #[test]
    fn sparse_native_jacobian_evaluates_only_symbolic_nonzeros() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::SparseTriplets,
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![1.0, 2.0]));
        let BdfJacobian::SparseTriplets { n, triplets } = out else {
            panic!("expected sparse triplet Jacobian");
        };
        assert_eq!(n, 2);
        assert_eq!(triplets.len(), 3);
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 0 && entry.val == -2.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 1 && entry.val == 1.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 1 && entry.col == 0 && entry.val == 3.0)
        );
    }

    #[test]
    fn dense_native_jacobian_evaluates_symbolic_entries_into_dense_matrix() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1 - 4*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::Dense,
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![1.0, 2.0]));
        let BdfJacobian::Dense(matrix) = out else {
            panic!("expected dense Jacobian");
        };
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], -2.0);
        assert_eq!(matrix[(0, 1)], 1.0);
        assert_eq!(matrix[(1, 0)], 3.0);
        assert_eq!(matrix[(1, 1)], -4.0);
    }

    #[test]
    fn banded_native_jacobian_preserves_symbolic_bandwidth() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1 - 4*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::Banded { bandwidth: None },
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![1.0, 2.0]));
        let BdfJacobian::Banded(banded) = out else {
            panic!("expected banded Jacobian");
        };
        assert_eq!(banded.n(), 2);
        assert_eq!(banded.kl(), 1);
        assert_eq!(banded.ku(), 1);
        assert_eq!(banded[(0, 0)], -2.0);
        assert_eq!(banded[(0, 1)], 1.0);
        assert_eq!(banded[(1, 0)], 3.0);
        assert_eq!(banded[(1, 1)], -4.0);
    }

    #[test]
    fn native_jacobian_uses_parameter_input_order() {
        let equations = vec![
            Expr::parse_expression("a*y1 + b*y2"),
            Expr::parse_expression("y1"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string()];
        let mut jacobian = compile_native_symbolic_jacobian_with_parameters(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(DVector::from_vec(vec![2.0, -3.0])),
            NativeJacobianStorage::SparseTriplets,
        );

        let out = jacobian(0.0, &DVector::from_vec(vec![10.0, 20.0]));
        let BdfJacobian::SparseTriplets { n, triplets } = out else {
            panic!("expected sparse triplet Jacobian");
        };
        assert_eq!(n, 2);
        assert_eq!(triplets.len(), 3);
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 0 && entry.val == 2.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 1 && entry.val == -3.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 1 && entry.col == 0 && entry.val == 1.0)
        );
    }

    #[test]
    fn native_jacobian_reads_updated_shared_parameter_values() {
        let equations = vec![
            Expr::parse_expression("a*y1 + b*y2"),
            Expr::parse_expression("y1"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string()];
        let handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0, -3.0])));
        let mut jacobian = compile_native_symbolic_jacobian_with_parameter_handle(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(handle.clone()),
            NativeJacobianStorage::SparseTriplets,
        );

        {
            let mut values = handle
                .write()
                .expect("shared IVP parameter state lock should be writable");
            *values = DVector::from_vec(vec![5.0, 7.0]);
        }

        let out = jacobian(0.0, &DVector::from_vec(vec![10.0, 20.0]));
        let BdfJacobian::SparseTriplets { triplets, .. } = out else {
            panic!("expected sparse triplet Jacobian");
        };
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 0 && entry.val == 5.0)
        );
        assert!(
            triplets
                .iter()
                .any(|entry| entry.row == 0 && entry.col == 1 && entry.val == 7.0)
        );
    }

    #[test]
    fn native_atomview_jacobian_reports_parameter_schema_errors_without_panicking() {
        let equations = vec![Expr::parse_expression("a*y1")];
        let variables = vec!["y1".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string()];
        let handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0])));

        let result =
            try_compile_native_atomview_jacobian_with_parameter_handle_and_telemetry_and_policy(
                &equations,
                &variables,
                "t",
                Some(&parameters),
                Some(handle),
                NativeJacobianStorage::SparseTriplets,
                IvpTelemetry::counters(),
                IvpLambdifyExecutionPolicy::Sequential,
            );

        assert!(matches!(
            result,
            Err(IvpBackendError::ParameterCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn native_atomview_typed_callback_reports_invalid_state_without_nan_fallback() {
        let equations = vec![Expr::parse_expression("a*y1")];
        let variables = vec!["y1".to_string()];
        let parameters = vec!["a".to_string()];
        let handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0])));
        let telemetry = IvpTelemetry::counters();
        let mut callback = try_compile_native_atomview_jacobian_callback(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(handle),
            NativeJacobianStorage::SparseTriplets,
            telemetry.clone(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("typed AtomView callback should prepare");

        let error = callback(0.0, &DVector::zeros(0)).unwrap_err();
        assert!(matches!(
            error,
            IvpBackendError::InvalidStateShape {
                expected: 1,
                actual: 0
            }
        ));
        assert_eq!(telemetry.snapshot().errors, 1);
    }

    #[test]
    fn native_atomview_rejects_unrepresentable_banded_layout_before_publish() {
        let equations = vec![Expr::parse_expression("y1")];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let result = try_compile_native_atomview_jacobian_callback(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::Banded {
                bandwidth: Some((1, 0)),
            },
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        );
        let error = match result {
            Ok(_) => panic!("invalid banded layout must remain a typed preparation error"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            IvpBackendError::InvalidBandedStorage {
                rows: 1,
                cols: 2,
                lower: 1,
                upper: 0
            }
        ));
    }

    #[test]
    fn native_atomview_rejects_entry_outside_explicit_banded_layout_before_publish() {
        let equations = vec![Expr::parse_expression("y2"), Expr::parse_expression("y1")];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let result = try_compile_native_atomview_jacobian_callback(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::Banded {
                bandwidth: Some((0, 0)),
            },
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        );
        let error = match result {
            Ok(_) => panic!("narrow band must be rejected during preparation"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            IvpBackendError::AtomEvaluationFailure {
                stage,
                index: 0,
                ..
            } if stage == "banded Jacobian layout preparation"
        ));
    }

    #[test]
    fn native_jacobian_atomview_matches_exprlegacy_and_reports_backend_stages() {
        let equations = vec![
            Expr::parse_expression("a*y1 + y2"),
            Expr::parse_expression("y1*y1 - 3*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let parameters = vec!["a".to_string()];
        let values = Some(DVector::from_vec(vec![2.5]));
        let state = DVector::from_vec(vec![1.25, -0.5]);

        let mut expr_legacy = compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            values.clone().map(|values| Arc::new(RwLock::new(values))),
            NativeJacobianStorage::SparseTriplets,
            IvpSymbolicAssemblyBackend::ExprLegacy,
            IvpTelemetry::counters(),
        );
        let atom_telemetry = IvpTelemetry::counters();
        let mut atom_view = compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            values.map(|values| Arc::new(RwLock::new(values))),
            NativeJacobianStorage::SparseTriplets,
            IvpSymbolicAssemblyBackend::AtomView,
            atom_telemetry.clone(),
        );

        let BdfJacobian::SparseTriplets {
            triplets: expr_triplets,
            ..
        } = expr_legacy(0.0, &state)
        else {
            panic!("expected ExprLegacy sparse triplets");
        };
        let BdfJacobian::SparseTriplets {
            triplets: atom_triplets,
            ..
        } = atom_view(0.0, &state)
        else {
            panic!("expected AtomView sparse triplets");
        };

        assert_eq!(expr_triplets.len(), atom_triplets.len());
        for (expr_entry, atom_entry) in expr_triplets.iter().zip(atom_triplets.iter()) {
            assert_eq!(
                (expr_entry.row, expr_entry.col),
                (atom_entry.row, atom_entry.col)
            );
            assert!((expr_entry.val - atom_entry.val).abs() < 1e-12);
        }
        let snapshot = atom_telemetry.snapshot();
        assert!(snapshot.cold_stage(IvpColdStage::ExprToAtom).calls > 0);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
        for stage in [
            IvpColdStage::AtomDependencyAnalysis,
            IvpColdStage::SymbolicDifferentiation,
            IvpColdStage::NativeJacobianEvaluatorPreparation,
        ] {
            assert_eq!(snapshot.cold_stage(stage).calls, 1, "{stage:?}");
        }
    }

    #[test]
    fn native_runtime_fills_dense_sparse_and_banded_caller_buffers() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1 - 4*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let state = DVector::from_vec(vec![1.0, 2.0]);

        let mut dense = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::Dense,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("dense native runtime should prepare");
        let mut dense_out = DMatrix::from_element(2, 2, f64::NAN);
        dense
            .try_evaluate_dense_into(0.0, &state, &mut dense_out)
            .expect("dense caller buffer should be filled");
        assert_eq!(
            dense_out,
            DMatrix::from_row_slice(2, 2, &[-2.0, 1.0, 3.0, -4.0])
        );

        let mut sparse = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::SparseTriplets,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("sparse native runtime should prepare");
        assert_eq!(
            sparse.sparse_pattern(),
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        let mut sparse_values = vec![f64::NAN; 4];
        sparse
            .try_evaluate_sparse_values_into(0.0, &state, &mut sparse_values)
            .expect("sparse caller buffer should be filled");
        assert_eq!(sparse_values, vec![-2.0, 1.0, 3.0, -4.0]);

        let mut banded = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::Banded { bandwidth: None },
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("banded native runtime should prepare");
        assert_eq!(banded.banded_layout(), Some((1, 1, 6)));
        let mut banded_values = vec![f64::NAN; 6];
        banded
            .try_evaluate_banded_values_into(0.0, &state, &mut banded_values)
            .expect("banded caller buffer should be filled");
        let banded_matrix = Banded::from_vec(2, 1, 1, banded_values)
            .expect("prepared compact banded values should be valid");
        assert_eq!(banded_matrix[(0, 0)], -2.0);
        assert_eq!(banded_matrix[(0, 1)], 1.0);
        assert_eq!(banded_matrix[(1, 0)], 3.0);
        assert_eq!(banded_matrix[(1, 1)], -4.0);
    }

    #[test]
    fn native_runtime_dense_sparse_and_banded_layouts_are_componentwise_equal() {
        let equations = vec![
            Expr::parse_expression("a*y1 + y2 + t"),
            Expr::parse_expression("y1 - b*y2 + y3"),
            Expr::parse_expression("y2 - y3"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string(), "y3".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string()];
        let handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0, 4.0])));
        let state = DVector::from_vec(vec![1.5, -2.0, 0.25]);

        let mut dense = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(handle.clone()),
            NativeJacobianStorage::Dense,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("dense parity runtime should prepare");
        let mut dense_values = DMatrix::from_element(3, 3, f64::NAN);
        dense
            .try_evaluate_dense_into(0.75, &state, &mut dense_values)
            .expect("dense parity runtime should evaluate");

        let mut sparse = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(handle.clone()),
            NativeJacobianStorage::SparseTriplets,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("sparse parity runtime should prepare");
        assert_eq!(
            sparse.sparse_pattern(),
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
        );
        let sparse_pattern = sparse.sparse_pattern();
        let mut sparse_values = vec![f64::NAN; sparse_pattern.len()];
        sparse
            .try_evaluate_sparse_values_into(0.75, &state, &mut sparse_values)
            .expect("sparse parity runtime should evaluate");
        for ((row, col), value) in sparse_pattern.iter().zip(sparse_values.iter()) {
            assert!((dense_values[(*row, *col)] - value).abs() <= 1.0e-12);
        }

        let mut banded = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(handle),
            NativeJacobianStorage::Banded {
                bandwidth: Some((1, 1)),
            },
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("banded parity runtime should prepare");
        assert_eq!(banded.banded_layout(), Some((1, 1, 9)));
        let mut banded_values = vec![f64::NAN; 9];
        banded
            .try_evaluate_banded_values_into(0.75, &state, &mut banded_values)
            .expect("banded parity runtime should evaluate");
        let banded_matrix = Banded::from_vec(3, 1, 1, banded_values)
            .expect("banded parity storage should be valid");
        for row in 0..3 {
            for col in 0..3 {
                if (row as isize - col as isize).abs() <= 1 {
                    assert!(
                        (dense_values[(row, col)] - banded_matrix[(row, col)]).abs() <= 1.0e-12,
                        "banded slot differs at ({row}, {col})"
                    );
                } else {
                    assert_eq!(
                        dense_values[(row, col)],
                        0.0,
                        "test fixture must not contain out-of-band entries at ({row}, {col})"
                    );
                }
            }
        }
    }

    #[test]
    fn native_runtime_reuses_caller_buffers_and_rebinds_parameters() {
        let equations = vec![
            Expr::parse_expression("a*y1 + y2"),
            Expr::parse_expression("y1 - b*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string()];
        let handle = Arc::new(RwLock::new(DVector::from_vec(vec![2.0, 3.0])));
        let telemetry = IvpTelemetry::counters();
        let mut runtime = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            Some(&parameters),
            Some(handle.clone()),
            NativeJacobianStorage::SparseTriplets,
            telemetry.clone(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("parameterized native runtime should prepare");
        let state = DVector::from_vec(vec![10.0, 20.0]);
        let mut values = vec![0.0; 4];
        let values_ptr = values.as_ptr();

        runtime
            .try_evaluate_sparse_values_into(0.0, &state, &mut values)
            .expect("initial parameterized evaluation should succeed");
        assert_eq!(values, vec![2.0, 1.0, 1.0, -3.0]);

        *handle
            .write()
            .expect("shared parameter values should remain writable") =
            DVector::from_vec(vec![5.0, 7.0]);
        runtime
            .try_evaluate_sparse_values_into(0.0, &state, &mut values)
            .expect("rebound parameterized evaluation should succeed");
        assert_eq!(values, vec![5.0, 1.0, 1.0, -7.0]);
        assert_eq!(values.as_ptr(), values_ptr);
        assert_eq!(
            telemetry.snapshot().copied_bytes,
            0,
            "native sparse values should be evaluated directly into caller storage"
        );
    }

    #[test]
    fn native_runtime_parallel_policies_preserve_caller_owned_sparse_values() {
        let equations = vec![
            Expr::parse_expression("-2*y1 + y2"),
            Expr::parse_expression("3*y1 - 4*y2"),
        ];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let state = DVector::from_vec(vec![1.0, 2.0]);
        let sequential_telemetry = IvpTelemetry::counters();
        let parallel_telemetry = IvpTelemetry::counters();
        let auto_telemetry = IvpTelemetry::counters();
        let mut sequential = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::SparseTriplets,
            sequential_telemetry.clone(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("sequential native runtime should prepare");
        let mut parallel = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::SparseTriplets,
            parallel_telemetry.clone(),
            IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        )
        .expect("parallel native runtime should prepare");
        let mut auto = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::SparseTriplets,
            auto_telemetry.clone(),
            IvpLambdifyExecutionPolicy::Auto { min_work: 1 },
        )
        .expect("auto native runtime should prepare");
        let mut sequential_values = vec![0.0; 4];
        let mut parallel_values = vec![0.0; 4];
        let mut auto_values = vec![0.0; 4];
        sequential
            .try_evaluate_sparse_values_into(0.0, &state, &mut sequential_values)
            .expect("sequential caller-owned evaluation should succeed");
        parallel
            .try_evaluate_sparse_values_into(0.0, &state, &mut parallel_values)
            .expect("parallel caller-owned evaluation should succeed");
        auto.try_evaluate_sparse_values_into(0.0, &state, &mut auto_values)
            .expect("auto caller-owned evaluation should succeed");

        assert_eq!(parallel_values, sequential_values);
        assert_eq!(auto_values, sequential_values);
        assert_eq!(sequential_telemetry.snapshot().sequential_dispatches, 1);
        assert_eq!(parallel_telemetry.snapshot().parallel_dispatches, 1);
        let auto_snapshot = auto_telemetry.snapshot();
        assert_eq!(
            auto_snapshot.parallel_dispatches + auto_snapshot.sequential_dispatches,
            1
        );
    }

    #[test]
    fn native_runtime_rejects_wrong_storage_and_output_shapes_typed() {
        let equations = vec![Expr::parse_expression("y1"), Expr::parse_expression("y2")];
        let variables = vec!["y1".to_string(), "y2".to_string()];
        let state = DVector::from_vec(vec![1.0, 2.0]);
        let mut sparse = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::SparseTriplets,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("sparse native runtime should prepare");

        let mut dense_out = DMatrix::zeros(2, 2);
        assert!(matches!(
            sparse.try_evaluate_dense_into(0.0, &state, &mut dense_out),
            Err(IvpBackendError::InvalidJacobianStorage { expected, actual })
                if expected == "Dense" && actual == "SparseTriplets"
        ));
        let mut sparse_values = vec![0.0; 1];
        assert!(matches!(
            sparse.try_evaluate_sparse_values_into(0.0, &state, &mut sparse_values),
            Err(IvpBackendError::InvalidOutputShape { stage, expected: 2, actual: 1 })
                if stage == "sparse Jacobian values"
        ));

        let mut dense = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            None,
            None,
            NativeJacobianStorage::Dense,
            IvpTelemetry::counters(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("dense native runtime should prepare");
        let mut wrong_matrix = DMatrix::zeros(1, 4);
        assert!(matches!(
            dense.try_evaluate_dense_into(0.0, &state, &mut wrong_matrix),
            Err(IvpBackendError::InvalidMatrixShape {
                expected_rows: 2,
                expected_cols: 2,
                actual_rows: 1,
                actual_cols: 4,
                ..
            })
        ));
    }

    #[test]
    fn sparse_aot_jacobian_artifact_suffix_avoids_residual_name_collision() {
        let cfg =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/lsode2-tests")
                .with_crate_name_override(Some("generated_lsode2_same_name".to_string()))
                .with_module_name_override(Some("generated_lsode2_same_name".to_string()));
        let patched = with_lsode2_sparse_jacobian_artifact_suffix(cfg);

        assert_eq!(
            patched.crate_name_override.as_deref(),
            Some("generated_lsode2_same_name_sj")
        );
        assert_eq!(
            patched.module_name_override.as_deref(),
            Some("generated_lsode2_same_name_sj")
        );
    }

    #[test]
    fn atomview_aot_jacobian_reuses_combined_residual_artifact_identity() {
        let cfg =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/lsode2-tests")
                .with_crate_name_override(Some("generated_lsode2_same_name".to_string()))
                .with_module_name_override(Some("generated_lsode2_same_name".to_string()));
        let prepared =
            lsode2_sparse_jacobian_artifact_config(cfg, IvpSymbolicAssemblyBackend::AtomView);

        assert_eq!(
            prepared.crate_name_override.as_deref(),
            Some("generated_lsode2_same_name")
        );
        assert_eq!(
            prepared.module_name_override.as_deref(),
            Some("generated_lsode2_same_name")
        );
    }
}
