//! Experimental reusable Atom callback adapters for the opt-in PreparedAtom route.
//!
//! This route is not recommended for production: release A/B measurements found
//! cold full solves slower than the established solver-native callback path.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::DVector;

use crate::numerical::BVP_Damp::BVP_traits::{
    BandedMatrixType, BvpCallbackError, BvpLinearSolveError, Fun, Jac, MatrixType, VectorType,
};
use crate::somelinalg::banded::banded_assembly::BandedAssembly;
use crate::symbolic::View::{
    FunctionMap, PreparedAtomSystem, PreparedAtomSystemScratch, bvp::DiscretizedBvpAtomSystem,
};
use crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle;

pub(crate) struct PreparedCallbacks {
    pub fun: Box<dyn Fun>,
    pub jac: Box<dyn Jac>,
    pub preparation_time: Duration,
    pub jacobian_nnz: usize,
}

pub(crate) fn try_build_callbacks(
    discretized: &DiscretizedBvpAtomSystem,
    mut delegate: Box<dyn Jac>,
    bandwidth: (usize, usize),
    runtime_parameter_names: &[String],
    parameter_binding: BvpParameterBindingHandle,
) -> Result<PreparedCallbacks, String> {
    let begin = Instant::now();
    let unknown_count = discretized.variable_string.len();
    let mut input_names = Vec::with_capacity(runtime_parameter_names.len() + unknown_count);
    input_names.extend(runtime_parameter_names.iter().cloned());
    input_names.extend(discretized.variable_string.iter().cloned());
    let system = Arc::new(
        PreparedAtomSystem::try_from_atoms(
            &discretized.vector_of_functions,
            &input_names,
            None,
            FunctionMap::default(),
        )
        .map_err(|error| error.to_string())?,
    );
    if system.input_len() != input_names.len() {
        return Err(format!(
            "prepared input length {} does not match runtime parameters plus Newton unknowns ({})",
            system.input_len(),
            input_names.len()
        ));
    }

    // Reuse the native callback's matrix metadata and solver configuration;
    // only the matrix values are supplied by PreparedAtomSystem from here on.
    let template_input = DVector::zeros(unknown_count);
    let template_matrix = delegate.call(0.0, &template_input);
    let matrix_template = template_matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .ok_or_else(|| {
            format!(
                "PreparedAtom requires a BandedMatrixType template, got shape {:?}",
                template_matrix.shape()
            )
        })?
        .clone();
    if matrix_template.shape() != (unknown_count, unknown_count) {
        return Err(format!(
            "Banded matrix template shape {:?} does not match prepared system dimension {}",
            matrix_template.shape(),
            unknown_count
        ));
    }

    let jacobian_nnz = system
        .jacobian_entries()
        .filter(|entry| entry.col >= runtime_parameter_names.len())
        .count();
    let evaluated_jacobian_entries = system.jacobian_entries().len();
    Ok(PreparedCallbacks {
        fun: Box::new(PreparedResidual {
            system: Arc::clone(&system),
            parameter_binding: parameter_binding.clone(),
            parameter_count: runtime_parameter_names.len(),
            scratch: RefCell::new(PreparedAtomSystemScratch::default()),
        }),
        jac: Box::new(PreparedJacobian {
            system,
            delegate,
            parameter_binding,
            parameter_count: runtime_parameter_names.len(),
            unknown_count,
            scratch: PreparedAtomSystemScratch::default(),
            // PreparedAtom evaluates the full symbolic Jacobian, including
            // columns for runtime boundary-value inputs. Those columns are
            // discarded only when scattering into the Newton matrix.
            jacobian_values: vec![0.0; evaluated_jacobian_entries],
            bandwidth,
            matrix_template,
        }),
        preparation_time: begin.elapsed(),
        jacobian_nnz,
    })
}

struct PreparedResidual {
    system: Arc<PreparedAtomSystem>,
    parameter_binding: BvpParameterBindingHandle,
    parameter_count: usize,
    scratch: RefCell<PreparedAtomSystemScratch>,
}

impl Fun for PreparedResidual {
    fn call(&self, _x: f64, values: &dyn VectorType) -> Box<dyn VectorType> {
        self.try_call(_x, values)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn try_call(
        &self,
        _x: f64,
        values: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpCallbackError> {
        let input = values
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .ok_or_else(|| BvpCallbackError::TypeMismatch {
                expected: "DVector<f64>",
                actual: values.vec_type(),
            })?;
        let mut output = vec![0.0; self.system.residual_len()];
        let parameters =
            self.parameter_binding
                .snapshot()
                .ok_or_else(|| BvpCallbackError::Panic {
                    message: "PreparedAtom runtime boundary-value binding is unavailable".into(),
                })?;
        if parameters.len() != self.parameter_count {
            return Err(BvpCallbackError::Panic {
                message: format!(
                    "PreparedAtom expected {} runtime boundary values, got {}",
                    self.parameter_count,
                    parameters.len()
                ),
            });
        }
        let mut inputs = Vec::with_capacity(parameters.len() + input.len());
        inputs.extend(parameters.iter().copied());
        inputs.extend_from_slice(input.as_slice());
        self.system
            .residual_into(&inputs, &mut output, &mut self.scratch.borrow_mut())
            .map_err(|error| BvpCallbackError::Panic {
                message: format!("PreparedAtom residual evaluation failed: {error}"),
            })?;
        Ok(Box::new(DVector::from_vec(output)))
    }
}

struct PreparedJacobian {
    system: Arc<PreparedAtomSystem>,
    delegate: Box<dyn Jac>,
    parameter_binding: BvpParameterBindingHandle,
    parameter_count: usize,
    unknown_count: usize,
    scratch: PreparedAtomSystemScratch,
    jacobian_values: Vec<f64>,
    bandwidth: (usize, usize),
    matrix_template: BandedMatrixType,
}

impl PreparedJacobian {
    fn evaluate(&mut self, input: &[f64]) -> Result<BandedMatrixType, BvpCallbackError> {
        let parameters =
            self.parameter_binding
                .snapshot()
                .ok_or_else(|| BvpCallbackError::Panic {
                    message: "PreparedAtom runtime boundary-value binding is unavailable".into(),
                })?;
        if parameters.len() != self.parameter_count {
            return Err(BvpCallbackError::Panic {
                message: format!(
                    "PreparedAtom expected {} runtime boundary values, got {}",
                    self.parameter_count,
                    parameters.len()
                ),
            });
        }
        let mut inputs = Vec::with_capacity(parameters.len() + input.len());
        inputs.extend(parameters.iter().copied());
        inputs.extend_from_slice(input);
        self.system
            .jacobian_values_into(&inputs, &mut self.jacobian_values, &mut self.scratch)
            .map_err(|error| BvpCallbackError::Panic {
                message: format!("PreparedAtom Jacobian evaluation failed: {error}"),
            })?;
        let mut assembly =
            BandedAssembly::zeros(self.unknown_count, self.bandwidth.0, self.bandwidth.1).map_err(
                |error| BvpCallbackError::Panic {
                    message: format!("PreparedAtom band allocation failed: {error}"),
                },
            )?;
        for (entry, value) in self.system.jacobian_entries().zip(&self.jacobian_values) {
            if entry.col < self.parameter_count {
                continue;
            }
            assembly
                .set(entry.row, entry.col - self.parameter_count, *value)
                .map_err(|error| BvpCallbackError::Panic {
                    message: format!("PreparedAtom band scatter failed: {error}"),
                })?;
        }
        let mut matrix = self.matrix_template.clone();
        matrix.set_assembly(assembly);
        Ok(matrix)
    }
}

impl Jac for PreparedJacobian {
    fn call(&mut self, _x: f64, vector: &dyn VectorType) -> Box<dyn MatrixType> {
        let input = vector
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .expect("Banded callbacks receive a dense Newton vector");
        Box::new(
            self.evaluate(input.as_slice())
                .unwrap_or_else(|error| panic!("{error}")),
        )
    }

    fn try_call(
        &mut self,
        _x: f64,
        vector: &dyn VectorType,
    ) -> Result<Box<dyn MatrixType>, BvpCallbackError> {
        let input = vector
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .ok_or_else(|| BvpCallbackError::TypeMismatch {
                expected: "DVector<f64>",
                actual: vector.vec_type(),
            })?;
        Ok(Box::new(self.evaluate(input.as_slice())?))
    }

    fn inv(&mut self, matrix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType> {
        self.delegate.inv(matrix, tol, max_iter)
    }

    fn try_inv(
        &mut self,
        matrix: &dyn MatrixType,
        tol: f64,
        max_iter: usize,
    ) -> Result<Box<dyn MatrixType>, BvpLinearSolveError> {
        self.delegate.try_inv(matrix, tol, max_iter)
    }

    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vector: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Box<dyn VectorType> {
        self.delegate.solve_sys(matrix, vector, tol, max_iter)
    }

    fn try_solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vector: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        self.delegate.try_solve_sys(matrix, vector, tol, max_iter)
    }
}
