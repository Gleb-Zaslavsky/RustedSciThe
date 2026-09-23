pub trait Jac {
    fn call(&mut self, x: f64, vec: &dyn VectorType) -> Box<dyn MatrixType>;
    /// Fallible compatibility entry point for the typed solver path.
    fn try_call(
        &mut self,
        x: f64,
        vec: &dyn VectorType,
    ) -> Result<Box<dyn MatrixType>, BvpCallbackError> {
        catch_unwind(AssertUnwindSafe(|| self.call(x, vec))).map_err(|payload| {
            BvpCallbackError::Panic {
                message: panic_message(payload),
            }
        })
    }
    fn inv(&mut self, matix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType>;
    /// Fallible compatibility boundary for inverse-Jacobian callbacks.
    ///
    /// Existing implementations retain the historical `inv` ABI, while the
    /// typed path converts backend/type/panic failures into the same linear
    /// error family used by `MatrixType::try_solve_sys`.
    fn try_inv(
        &mut self,
        matrix: &dyn MatrixType,
        tol: f64,
        max_iter: usize,
    ) -> Result<Box<dyn MatrixType>, BvpLinearSolveError> {
        catch_unwind(AssertUnwindSafe(|| self.inv(matrix, tol, max_iter))).map_err(|payload| {
            BvpLinearSolveError::LegacyPanic {
                backend: "jacobian-callback",
                message: panic_message(payload),
            }
        })
    }
    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Box<dyn VectorType>;
    /// Fallible compatibility boundary for Jacobian-owned linear solves.
    fn try_solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        if matrix.shape().0 != matrix.shape().1 || vec.len() != matrix.shape().0 {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows: matrix.shape().0,
                matrix_columns: matrix.shape().1,
                rhs_len: vec.len(),
            });
        }
        catch_unwind(AssertUnwindSafe(|| {
            self.solve_sys(matrix, vec, tol, max_iter)
        }))
        .map_err(|payload| BvpLinearSolveError::LegacyPanic {
            backend: "jacobian-callback",
            message: panic_message(payload),
        })
    }
}

pub enum JacEnum {
    Dense(Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>),
    Sparse_1(Box<dyn FnMut(f64, &CsVec<f64>) -> CsMat<f64>>),
    Sparse_3(Box<dyn FnMut(f64, &faer_col) -> faer_mat>),
}

impl Jac for JacEnum {
    fn call(&mut self, x: f64, vec: &dyn VectorType) -> Box<dyn MatrixType> {
        match self {
            JacEnum::Dense(jac) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                    Box::new(jac(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
            JacEnum::Sparse_1(jac) => {
                if let Some(cs_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
                    Box::new(jac(x, cs_vec))
                } else {
                    panic!("Type mismatch: expected CsVec")
                }
            }
            JacEnum::Sparse_3(jac) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                    Box::new(jac(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
        }
    } // call

    fn try_call(
        &mut self,
        x: f64,
        vec: &dyn VectorType,
    ) -> Result<Box<dyn MatrixType>, BvpCallbackError> {
        match self {
            JacEnum::Dense(jac) => {
                let d_vec = vec.as_any().downcast_ref::<DVector<f64>>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "DVector",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(jac(x, d_vec)) as Box<dyn MatrixType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
            JacEnum::Sparse_1(jac) => {
                let sparse_vec = vec.as_any().downcast_ref::<CsVec<f64>>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "CsVec",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(jac(x, sparse_vec)) as Box<dyn MatrixType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
            JacEnum::Sparse_3(jac) => {
                let faer_vec = vec.as_any().downcast_ref::<faer_col>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "faer_col",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(jac(x, faer_vec)) as Box<dyn MatrixType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
        }
    }

    fn inv(&mut self, matrix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType> {
        match self {
            JacEnum::Dense(_jac) => {
                if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
                    Box::new(mat.to_owned().try_inverse().unwrap())
                } else {
                    panic!("Type mismatch: expected DMatrix")
                }
            }
            JacEnum::Sparse_1(_jac) => {
                if let Some(mat) = matrix.as_any().downcast_ref::<CsMat<f64>>() {
                    let inv_mat = invers_csmat(mat.to_owned(), tol, max_iter).unwrap();
                    Box::new(inv_mat)
                } else {
                    panic!("Type mismatch: expected CsMat")
                }
            }
            JacEnum::Sparse_3(_jac) => {
                if let Some(mat_) = matrix.as_any().downcast_ref::<SparseColMat<usize, f64>>() {
                    //  let inv_mat = invers_Mat_mult(mat_.to_owned().expect("REASON"), tol, max_iter).unwrap();//why Result()?;
                    let inv_mat = invers_Mat_LU(mat_.to_owned(), tol, max_iter).unwrap();
                    //  mat_.to_owned().unwrap().sort_indices();
                    // let inv_mat =  solve_with_upper_triangular(mat_.to_owned().expect("REASON"), tol, max_iter).unwrap();
                    Box::new(inv_mat)
                } else {
                    panic!("Type mismatch: expected faer_mat")
                }
            }
        }
    }

    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        _tol: f64,
        _max_iter: usize,
    ) -> Box<dyn VectorType> {
        match self {
            JacEnum::Dense(_jac) => {
                if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
                    if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                        let lu = mat.to_owned().lu();
                        let res = lu.solve(d_vec).unwrap();
                        Box::new(res)
                    } else {
                        panic!("Type mismatch: expected DVector")
                    }
                } else {
                    panic!("Type mismatch: expected DMatrix")
                }
            } //dense
            JacEnum::Sparse_3(_jac) => {
                if let Some(mat_) = matrix.as_any().downcast_ref::<SparseColMat<usize, f64>>() {
                    if let Some(d_vec) = vec.as_any().downcast_ref::<Col<f64>>() {
                        let lhs: MatRef<f64> = d_vec.as_mat();
                        let LU = mat_.sp_lu().unwrap();
                        let res: Mat<f64> = LU.solve(lhs);

                        let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();

                        let res = ColRef::from_slice(res_vec.as_slice()).to_owned();
                        Box::new(res)
                    } else {
                        panic!("Type mismatch: expected DVector")
                    }
                } else {
                    panic!("Type mismatch: expected faer_mat")
                }
            }
            _ => {
                panic!("Type mismatch: expected DMatrix")
            }
        }
    }
}

pub struct JacWrapper(Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn MatrixType>>);

// Implement Fun for the wrapper
impl Jac for JacWrapper {
    fn call(&mut self, x: f64, vec: &dyn VectorType) -> Box<dyn MatrixType> {
        (self.0)(x, vec)
    }
    fn inv(&mut self, matrix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType> {
        if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
            Box::new(mat.to_owned().try_inverse().unwrap())
        } else if let Some(mat) = matrix.as_any().downcast_ref::<CsMat<f64>>() {
            let inv_mat = invers_csmat(mat.to_owned(), tol, max_iter).unwrap();
            Box::new(inv_mat)
        } else if let Some(mat) = matrix.as_any().downcast_ref::<faer_mat>() {
            let inv_mat = invers_Mat_LU(mat.to_owned(), tol, max_iter).unwrap();
            Box::new(inv_mat)
        } else {
            panic!("Unsupported matrix type for inversion")
        }
    }
    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        _tol: f64,
        _max_iter: usize,
    ) -> Box<dyn VectorType> {
        if let Some(mat) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                let lu = mat.to_owned().lu();
                let res = lu.solve(d_vec).unwrap();
                Box::new(res)
            } else {
                panic!("Vector type mismatch for solving system")
            }
        } else if let Some(mat) = matrix.as_any().downcast_ref::<faer_mat>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                let lhs: MatRef<f64> = d_vec.as_mat();
                let LU = mat.sp_lu().unwrap();
                let res: Mat<f64> = LU.solve(lhs);

                let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();

                let res = ColRef::from_slice(res_vec.as_slice()).to_owned();

                Box::new(res)
            } else {
                panic!("Vector type mismatch for solving system")
            }
        } else {
            panic!("Unsupported matrix type for solving system")
        }
    }
}

// Then create a conversion function
pub fn convert_to_jac(f: Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn MatrixType>>) -> Box<dyn Jac> {
    Box::new(JacWrapper(f))
}

/// Builds a finite-difference Jacobian matrix for the provided residual callback.
///
/// The routine preserves the currently selected vector/matrix backend by
/// constructing the result through [`VectorType::from_vector`].
pub fn finite_difference_jacobian(
    fun: &dyn Fun,
    x: f64,
    vec: &dyn VectorType,
    step_scale: f64,
) -> Box<dyn MatrixType> {
    try_finite_difference_jacobian(fun, x, vec, step_scale)
        .unwrap_or_else(|error| panic!("finite-difference Jacobian failed: {error}"))
}

/// Fallible finite-difference Jacobian construction for typed solver paths.
///
/// Residual callback panics and shape mismatches are returned instead of
/// aborting the solve.  The non-fallible [`finite_difference_jacobian`]
/// wrapper remains for existing callers which rely on the old API.
pub fn try_finite_difference_jacobian(
    fun: &dyn Fun,
    x: f64,
    vec: &dyn VectorType,
    step_scale: f64,
) -> Result<Box<dyn MatrixType>, BvpCallbackError> {
    let n = vec.len();
    let f0 = fun.try_call(x, vec)?.to_DVectorType();
    if f0.len() != n {
        return Err(BvpCallbackError::OutputShapeMismatch {
            expected: n,
            actual: f0.len(),
        });
    }

    let mut dense = vec![0.0_f64; n * n];
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(n * n);
    let min_step = f64::EPSILON.sqrt();

    for col in 0..n {
        let y_col = vec.get_val(col);
        let h = (step_scale * y_col.abs()).max(min_step);
        let perturbed = vec.assign_value(col, y_col + h);
        let f1 = fun.try_call(x, &*perturbed)?.to_DVectorType();
        if f1.len() != n {
            return Err(BvpCallbackError::OutputShapeMismatch {
                expected: n,
                actual: f1.len(),
            });
        }

        for row in 0..n {
            let value = (f1[row] - f0[row]) / h;
            dense[row * n + col] = value;
            if value != 0.0 {
                triplets.push((row, col, value));
            }
        }
    }

    Ok(vec.from_vector(n, n, &dense, triplets))
}

//_________________________________________________Y___________________________________________________________
