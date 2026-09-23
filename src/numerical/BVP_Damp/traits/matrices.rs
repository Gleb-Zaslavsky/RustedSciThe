/// NUMERICAL REPRESENTATION OF JACOBIAN FOR DIFFERENT CRATES
#[allow(dead_code)]
pub enum JacTypes {
    Dense(DMatrix<f64>),
    Sparse_1(CsMat<f64>),
    Sparse_3(faer_mat),
}

/// Runtime matrix wrapper for the native banded BVP backend.
///
/// We keep the fast assembly storage (`BandedAssembly`) together with the
/// node-major layout metadata required to convert into the block-tridiagonal
/// factorization format right before the linear solve.
#[derive(Clone, Debug)]
pub struct BandedMatrixType {
    /// Compact numeric matrix values.
    ///
    /// This field remains public for legacy callers. New code should prefer
    /// [`Self::set_assembly`], which also invalidates the cached factorization.
    pub assembly: BandedAssembly,
    pub layout: NodeMajorLayout,
    /// Linear solver policy used to build the native factorization.
    ///
    /// Prefer [`Self::set_solver_config`] when changing it after construction.
    pub solver_config: LinearSolverConfig,
    /// Shared immutable factorization for clones of the same Jacobian.
    ///
    /// The lock is only used once, during lazy factor construction. RHS solves
    /// borrow the already-built native solver and do not lock the hot path.
    pub cached_solver: Arc<OnceLock<crate::somelinalg::banded::linear_solver::LinearSolver>>,
    /// Duration spent constructing the currently cached factorization.
    ///
    /// This is separate from RHS solve time so a repeated solve can expose
    /// whether factor reuse actually happened. The value is shared by cloned
    /// Jacobian handles together with the cached solver.
    pub factorization_duration: Arc<OnceLock<Duration>>,
    /// Accumulated time spent solving RHS vectors with the cached factor.
    ///
    /// RHS solves are independent and do not need a lock; an atomic nanosecond
    /// counter keeps this diagnostic out of the hot path's synchronization
    /// structures.
    pub rhs_solve_duration_nanos: Arc<AtomicU64>,
    /// `true` for scalar/general banded assembly, `false` for generated
    /// node-major assembly that can use block-tridiagonal factorization.
    general_banded: bool,
}

impl BandedMatrixType {
    pub fn new(
        assembly: BandedAssembly,
        layout: NodeMajorLayout,
        solver_config: LinearSolverConfig,
    ) -> Self {
        Self {
            assembly,
            layout,
            solver_config,
            cached_solver: Arc::new(OnceLock::new()),
            factorization_duration: Arc::new(OnceLock::new()),
            rhs_solve_duration_nanos: Arc::new(AtomicU64::new(0)),
            general_banded: false,
        }
    }

    /// Construct a compact scalar-banded matrix produced by the numeric
    /// finite-difference route. Unlike [`Self::new`], this must not reinterpret
    /// the scalar bandwidth as a node-major block layout.
    pub fn new_general_banded(
        assembly: BandedAssembly,
        layout: NodeMajorLayout,
        solver_config: LinearSolverConfig,
    ) -> Self {
        Self {
            assembly,
            layout,
            solver_config,
            cached_solver: Arc::new(OnceLock::new()),
            factorization_duration: Arc::new(OnceLock::new()),
            rhs_solve_duration_nanos: Arc::new(AtomicU64::new(0)),
            general_banded: true,
        }
    }

    /// Discard the factorization after a legacy in-place field mutation.
    ///
    /// The native factor is intentionally immutable after construction so
    /// repeated RHS solves stay lock-free. Call this method after mutating the
    /// public `assembly` or `solver_config` fields directly.
    pub fn invalidate_factorization(&mut self) {
        self.cached_solver = Arc::new(OnceLock::new());
        self.factorization_duration = Arc::new(OnceLock::new());
        self.rhs_solve_duration_nanos = Arc::new(AtomicU64::new(0));
    }

    /// Returns the time spent constructing the currently cached factor.
    pub fn factorization_duration(&self) -> Duration {
        self.factorization_duration
            .get()
            .copied()
            .unwrap_or(Duration::ZERO)
    }

    /// Returns accumulated time spent solving RHS vectors with this factor.
    pub fn rhs_solve_duration(&self) -> Duration {
        Duration::from_nanos(self.rhs_solve_duration_nanos.load(Ordering::Relaxed))
    }

    /// Replace matrix values and invalidate the native factorization.
    pub fn set_assembly(&mut self, assembly: BandedAssembly) {
        self.assembly = assembly;
        self.invalidate_factorization();
    }

    /// Replace the linear solver policy and invalidate the native factorization.
    pub fn set_solver_config(&mut self, solver_config: LinearSolverConfig) {
        self.solver_config = solver_config;
        self.invalidate_factorization();
    }

    /// Fallible native solve used by the typed BVP boundary.
    ///
    /// Unlike the compatibility `solve_sys` implementation below, this
    /// method keeps factory and RHS errors as values and never uses
    /// `expect`/`unwrap` for user-visible linear algebra failures.
    fn try_native_solve(
        &self,
        vec: &dyn VectorType,
    ) -> Result<(Box<dyn VectorType>, LinearSolveTiming), BvpLinearSolveError> {
        let factorization_before = self.factorization_duration();
        let rhs_before = self.rhs_solve_duration();
        let solver = if let Some(solver) = self.cached_solver.get() {
            solver
        } else {
            let factor_begin = Instant::now();
            let system = if self.general_banded {
                LinearSystemRef::BandedAssembly(&self.assembly)
            } else {
                LinearSystemRef::NodeMajorAssembly {
                    assembly: &self.assembly,
                    layout: self.layout,
                }
            };
            let candidate =
                build_solver_for_system(system, self.solver_config).map_err(|error| {
                    BvpLinearSolveError::FactorizationFailed {
                        backend: "native-banded",
                        message: format!("{error:?}"),
                    }
                })?;
            let _ = self.factorization_duration.set(factor_begin.elapsed());
            let _ = self.cached_solver.set(candidate);
            self.cached_solver
                .get()
                .ok_or_else(|| BvpLinearSolveError::FactorizationFailed {
                    backend: "native-banded",
                    message: "factorization was built but could not be published".to_string(),
                })?
        };

        // Built-in Banded callbacks normally pass a nalgebra DVector. Avoid
        // materializing a temporary DVector before copying into the mutable
        // RHS buffer required by the native solver. Other VectorType
        // implementations retain the compatibility conversion fallback.
        let mut rhs = dense_rhs_vec(vec);
        let rhs_begin = Instant::now();
        solver.solve_in_place(rhs.as_mut_slice()).map_err(|error| {
            BvpLinearSolveError::SolveFailed {
                backend: "native-banded",
                message: format!("{error:?}"),
            }
        })?;
        self.rhs_solve_duration_nanos.fetch_add(
            rhs_begin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );

        Ok((
            Box::new(DVector::from_vec(rhs)),
            LinearSolveTiming {
                factorization: self
                    .factorization_duration()
                    .saturating_sub(factorization_before),
                rhs_solve: self.rhs_solve_duration().saturating_sub(rhs_before),
            },
        ))
    }
}

#[inline]
fn dense_rhs_vec(vec: &dyn VectorType) -> Vec<f64> {
    if let Some(dense) = vec.as_any().downcast_ref::<DVector<f64>>() {
        dense.as_slice().to_vec()
    } else {
        vec.to_DVectorType().as_slice().to_vec()
    }
}

pub trait MatrixType: Any {
    fn as_any(&self) -> &dyn Any;
    fn inverse(self) -> Box<dyn MatrixType>; // inverse
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType>; // multiplication of matrix and vector
    fn clone_box(&self) -> Box<dyn MatrixType>; // clone
    fn solve_sys(
        // solve linear system
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType>;
    /// Fallible counterpart of [`Self::solve_sys`].
    ///
    /// The default is deliberately a compatibility adapter for external
    /// `MatrixType` implementations. Built-in backends override it where
    /// they can expose native factorization/solve errors. Keeping the
    /// adapter here lets the solver migrate incrementally without silently
    /// changing the historical object-safe trait contract.
    fn try_solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        let (matrix_rows, matrix_columns) = self.shape();
        if matrix_rows != matrix_columns || vec.len() != matrix_rows {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len: vec.len(),
            });
        }

        catch_unwind(AssertUnwindSafe(|| {
            self.solve_sys(vec, linear_sys_method, tol, max_iter, bandwidth, old_vec)
        }))
        .map_err(|payload| BvpLinearSolveError::LegacyPanic {
            backend: "external-matrix",
            message: panic_message(payload),
        })
    }

    /// Timed fallible solve used by the typed solver path.
    fn try_solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Result<(Box<dyn VectorType>, LinearSolveTiming), BvpLinearSolveError> {
        let begin = Instant::now();
        let result =
            self.try_solve_sys(vec, linear_sys_method, tol, max_iter, bandwidth, old_vec)?;
        Ok((
            result,
            LinearSolveTiming {
                factorization: Duration::ZERO,
                rhs_solve: begin.elapsed(),
            },
        ))
    }
    /// Solve a linear system and expose typed factor/RHS timing.
    ///
    /// This is an additive API. Existing implementors only need to keep
    /// implementing `solve_sys`; the default classifies the complete legacy
    /// call as `rhs_solve`. Built-in Dense, faer Sparse and native Banded
    /// implementations override it with a real stage split.
    fn solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> (Box<dyn VectorType>, LinearSolveTiming) {
        let begin = Instant::now();
        let result = self.solve_sys(vec, linear_sys_method, tol, max_iter, bandwidth, old_vec);
        (
            result,
            LinearSolveTiming {
                factorization: Duration::ZERO,
                rhs_solve: begin.elapsed(),
            },
        )
    }
    /// Returns whether this matrix already owns a reusable factorization.
    /// Backends without a reusable workspace conservatively return `false`.
    fn factorization_ready(&self) -> bool {
        false
    }
    /// Returns typed time spent constructing the current factorization.
    fn factorization_duration(&self) -> Duration {
        Duration::ZERO
    }
    /// Returns typed accumulated time spent solving RHS vectors.
    fn rhs_solve_duration(&self) -> Duration {
        Duration::ZERO
    }
    fn shape(&self) -> (usize, usize); // shape of the matrix
    fn to_DMatrixType(&self) -> DMatrix<f64>; // convert to DMatrix
}
/*
impl Clone for dyn MatrixType {

    fn clone(&self) -> Self {
          self.as_any().downcast_ref::<Self>().clone()
    }
}
*/
////////////////////////////////////////////////////////////////
//           NALGEBRA CRATE
////////////////////////////////////////////////////////////////
impl MatrixType for DMatrix<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn inverse(self) -> Box<dyn MatrixType> {
        let inverse = self.try_inverse();
        let inverse = inverse.unwrap();
        Box::new(inverse)
    }
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
            Box::new(self * d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }

    fn try_solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        let (matrix_rows, matrix_columns) = self.shape();
        if matrix_rows != matrix_columns || vec.len() != matrix_rows {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len: vec.len(),
            });
        }
        if let Some(policy) = linear_sys_method {
            return Err(BvpLinearSolveError::UnsupportedLinearPolicy {
                backend: "dense",
                policy,
            });
        }
        let rhs = vec.as_any().downcast_ref::<DVector<f64>>().ok_or(
            BvpLinearSolveError::VectorTypeMismatch {
                backend: "dense",
                expected: "DVector<f64>",
            },
        )?;

        if matrix_rows > 10 * (bandwidth.0 + bandwidth.1) {
            catch_unwind(AssertUnwindSafe(|| {
                solve_dense_nalgebra_linear_system(self, rhs, bandwidth)
            }))
            .map_err(|payload| BvpLinearSolveError::FactorizationFailed {
                backend: "dense",
                message: panic_message(payload),
            })
            .and_then(|solution| {
                if solution.iter().all(|value| value.is_finite()) {
                    Ok(Box::new(solution) as Box<dyn VectorType>)
                } else {
                    Err(BvpLinearSolveError::FactorizationFailed {
                        backend: "dense",
                        message: "dense banded solve returned a non-finite solution".to_string(),
                    })
                }
            })
        } else {
            self.clone()
                .lu()
                .solve(rhs)
                .filter(|solution| solution.iter().all(|value| value.is_finite()))
                .map(|solution| Box::new(solution) as Box<dyn VectorType>)
                .ok_or_else(|| BvpLinearSolveError::FactorizationFailed {
                    backend: "dense",
                    message: "LU factorization could not produce a finite solution".to_string(),
                })
        }
    }

    fn try_solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Result<(Box<dyn VectorType>, LinearSolveTiming), BvpLinearSolveError> {
        let (matrix_rows, matrix_columns) = self.shape();
        if matrix_rows != matrix_columns || vec.len() != matrix_rows {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len: vec.len(),
            });
        }
        if let Some(policy) = linear_sys_method {
            return Err(BvpLinearSolveError::UnsupportedLinearPolicy {
                backend: "dense",
                policy,
            });
        }
        let rhs = vec.as_any().downcast_ref::<DVector<f64>>().ok_or(
            BvpLinearSolveError::VectorTypeMismatch {
                backend: "dense",
                expected: "DVector<f64>",
            },
        )?;
        let result = catch_unwind(AssertUnwindSafe(|| {
            solve_dense_nalgebra_linear_system_timed(self, rhs, bandwidth)
        }))
        .map_err(|payload| BvpLinearSolveError::FactorizationFailed {
            backend: "dense",
            message: panic_message(payload),
        })?;
        if !result.0.iter().all(|value| value.is_finite()) {
            return Err(BvpLinearSolveError::FactorizationFailed {
                backend: "dense",
                message: "dense solve returned a non-finite solution".to_string(),
            });
        }
        Ok((Box::new(result.0), result.1))
    }

    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        if let Some(mat_) = self.as_any().downcast_ref::<DMatrix<f64>>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                assert!(
                    linear_sys_method.is_none(),
                    "Dense nalgebra BVP solve_sys only supports automatic full/banded selection"
                );
                let res = solve_dense_nalgebra_linear_system(mat_, d_vec, bandwidth);
                Box::new(res)
            } else {
                panic!("Type mismatch: expected DMatrix")
            }
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }

    fn solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> (Box<dyn VectorType>, LinearSolveTiming) {
        assert!(
            linear_sys_method.is_none(),
            "Dense nalgebra BVP solve_sys only supports automatic full/banded selection"
        );
        let d_vec = vec
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .expect("Dense nalgebra solve expects DVector");
        let (solution, timing) = solve_dense_nalgebra_linear_system_timed(self, d_vec, bandwidth);
        (Box::new(solution), timing)
    }

    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    fn to_DMatrixType(&self) -> DMatrix<f64> {
        self.to_owned()
    }
}
////////////////////////////////////////////////////////////////////////////////////
//      CRATE SPRS
////////////////////////////////////////////////////////////////////////////////////
impl MatrixType for CsMat<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn inverse(self) -> Box<dyn MatrixType> {
        let inv_mat = invers_csmat(self.clone(), 1e-8, 100).unwrap();
        Box::new(inv_mat)
    }
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
            Box::new(self * d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }
    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        _linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        _bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        if let Some(mat_) = self.as_any().downcast_ref::<CsMat<f64>>() {
            if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
                if let Some(old_vec) = old_vec.as_any().downcast_ref::<CsVec<f64>>() {
                    let res = solve_csmat(mat_, d_vec, tol, max_iter, old_vec).unwrap();
                    Box::new(res)
                } else {
                    panic!("Type mismatch: expected DMatrix")
                }
            } else {
                panic!("Type mismatch: expected DMatrix")
            }
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    fn to_DMatrixType(&self) -> DMatrix<f64> {
        let (nrows, ncols) = self.shape();
        let t = self.to_dense();
        let csmat = t.as_slice().unwrap();
        let dmatrix = DMatrix::from_row_slice(nrows, ncols, csmat);
        dmatrix
    }
}
////////////////////////////////////////////////////////////////
//            FAER CRATE
////////////////////////////////////////////////////////////////
impl MatrixType for faer_mat {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn inverse(self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    } //
    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
            Box::new(self * d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }

    fn try_solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        _bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        let (matrix_rows, matrix_columns) = self.shape();
        if matrix_rows != matrix_columns || vec.len() != matrix_rows {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len: vec.len(),
            });
        }
        let rhs = vec.as_any().downcast_ref::<faer_col>().ok_or(
            BvpLinearSolveError::VectorTypeMismatch {
                backend: "faer-sparse",
                expected: "faer::Col<f64>",
            },
        )?;

        match linear_sys_method {
            None => {
                let factor =
                    self.sp_lu()
                        .map_err(|error| BvpLinearSolveError::FactorizationFailed {
                            backend: "faer-sparse",
                            message: format!("{error:?}"),
                        })?;
                let lhs: MatRef<f64> = rhs.as_mat();
                let result: Mat<f64> = factor.solve(lhs);
                let values: Vec<f64> = result.row_iter().map(|row| row[0]).collect();
                Ok(Box::new(ColRef::from_slice(values.as_slice()).to_owned()))
            }
            Some(_) => {
                let previous = old_vec.as_any().downcast_ref::<faer_col>().ok_or(
                    BvpLinearSolveError::VectorTypeMismatch {
                        backend: "faer-sparse",
                        expected: "faer::Col<f64> old_vec",
                    },
                )?;
                solve_sys_SparseColMat(self, rhs.as_mat(), tol, max_iter, previous)
                    .map(|solution| Box::new(solution) as Box<dyn VectorType>)
                    .ok_or_else(|| BvpLinearSolveError::SolveFailed {
                        backend: "faer-sparse",
                        message: "iterative sparse solve returned no solution".to_string(),
                    })
            }
        }
    }

    fn try_solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        _bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Result<(Box<dyn VectorType>, LinearSolveTiming), BvpLinearSolveError> {
        let (matrix_rows, matrix_columns) = self.shape();
        if matrix_rows != matrix_columns || vec.len() != matrix_rows {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len: vec.len(),
            });
        }
        let rhs = vec.as_any().downcast_ref::<faer_col>().ok_or(
            BvpLinearSolveError::VectorTypeMismatch {
                backend: "faer-sparse",
                expected: "faer::Col<f64>",
            },
        )?;

        match linear_sys_method {
            None => {
                let factor_begin = Instant::now();
                let factor =
                    self.sp_lu()
                        .map_err(|error| BvpLinearSolveError::FactorizationFailed {
                            backend: "faer-sparse",
                            message: format!("{error:?}"),
                        })?;
                let factorization = factor_begin.elapsed();
                let rhs_begin = Instant::now();
                let lhs: MatRef<f64> = rhs.as_mat();
                let result: Mat<f64> = factor.solve(lhs);
                let values: Vec<f64> = result.row_iter().map(|row| row[0]).collect();
                if !values.iter().all(|value| value.is_finite()) {
                    return Err(BvpLinearSolveError::SolveFailed {
                        backend: "faer-sparse",
                        message: "direct sparse solve returned a non-finite solution".to_string(),
                    });
                }
                Ok((
                    Box::new(ColRef::from_slice(values.as_slice()).to_owned()),
                    LinearSolveTiming {
                        factorization,
                        rhs_solve: rhs_begin.elapsed(),
                    },
                ))
            }
            Some(_) => {
                let previous = old_vec.as_any().downcast_ref::<faer_col>().ok_or(
                    BvpLinearSolveError::VectorTypeMismatch {
                        backend: "faer-sparse",
                        expected: "faer::Col<f64> old_vec",
                    },
                )?;
                let rhs_begin = Instant::now();
                let result = solve_sys_SparseColMat(self, rhs.as_mat(), tol, max_iter, previous)
                    .ok_or_else(|| BvpLinearSolveError::SolveFailed {
                        backend: "faer-sparse",
                        message: "iterative sparse solve returned no solution".to_string(),
                    })?;
                Ok((
                    Box::new(result),
                    LinearSolveTiming {
                        factorization: Duration::ZERO,
                        rhs_solve: rhs_begin.elapsed(),
                    },
                ))
            }
        }
    }

    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        _bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        self.solve_sys_with_timing(vec, linear_sys_method, tol, max_iter, (0, 0), old_vec)
            .0
    } //solve_sys

    fn solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        _bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> (Box<dyn VectorType>, LinearSolveTiming) {
        let mat_ = self;
        let d_vec = vec
            .as_any()
            .downcast_ref::<faer_col>()
            .expect("faer sparse solve expects Col<f64>");
        assert_eq!(
            mat_.ncols(),
            d_vec.nrows(),
            " matrix {} and vector {} have different sizes",
            mat_.ncols(),
            d_vec.nrows()
        );
        assert_eq!(d_vec.len(), mat_.ncols());
        assert_eq!(d_vec.len(), mat_.nrows());

        match linear_sys_method {
            None => {
                let factor_begin = Instant::now();
                let lu = mat_.sp_lu().unwrap();
                let factorization = factor_begin.elapsed();
                let rhs_begin = Instant::now();
                let lhs: MatRef<f64> = d_vec.as_mat();
                let res: Mat<f64> = lu.solve(lhs);
                let res_vec: Vec<f64> = res.row_iter().map(|x| x[0]).collect();
                let res = ColRef::from_slice(res_vec.as_slice()).to_owned();
                (
                    Box::new(res),
                    LinearSolveTiming {
                        factorization,
                        rhs_solve: rhs_begin.elapsed(),
                    },
                )
            }
            Some(_gmres) => {
                let old_vec = old_vec
                    .as_any()
                    .downcast_ref::<faer_col>()
                    .expect("old vec must be of type Col<f64>");
                let begin = Instant::now();
                let lhs: MatRef<f64> = d_vec.as_mat();
                let res = solve_sys_SparseColMat(mat_, lhs, tol, max_iter, old_vec).unwrap();
                (
                    Box::new(res),
                    LinearSolveTiming {
                        factorization: Duration::ZERO,
                        rhs_solve: begin.elapsed(),
                    },
                )
            }
        }
    }
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn to_DMatrixType(&self) -> DMatrix<f64> {
        let (nrows, ncols) = self.shape();
        let dense = self.to_dense();
        let mut dmatrix = DMatrix::zeros(nrows, ncols);
        for (i, col) in dense.col_iter().enumerate() {
            let col = col.to_owned().iter().map(|x| *x).collect::<Vec<f64>>();
            dmatrix.column_mut(i).copy_from(&DVector::from_vec(col));
        }

        dmatrix
    }
} //impl

impl MatrixType for BandedMatrixType {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn inverse(self) -> Box<dyn MatrixType> {
        // The damped solver mostly uses direct solves instead of explicit
        // inverses. When an inverse is requested through the legacy API, fall
        // back to a dense inverse to preserve correctness.
        let dense = self.to_DMatrixType();
        let inverse = dense
            .try_inverse()
            .expect("banded matrix inverse requested for a singular matrix");
        Box::new(inverse)
    }

    fn mul(&self, vec: &dyn VectorType) -> Box<dyn VectorType> {
        let dense = self.to_DMatrixType();
        if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
            Box::new(&dense * d_vec)
        } else {
            let dense_vec = vec.to_DVectorType();
            Box::new(&dense * dense_vec)
        }
    }

    fn clone_box(&self) -> Box<dyn MatrixType> {
        Box::new(self.clone())
    }

    fn try_solve_sys(
        &self,
        vec: &dyn VectorType,
        _linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        _bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        let (matrix_rows, matrix_columns) = self.shape();
        if matrix_rows != matrix_columns || vec.len() != matrix_rows {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len: vec.len(),
            });
        }
        self.try_native_solve(vec).map(|(solution, _)| solution)
    }

    fn try_solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        _linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        _bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Result<(Box<dyn VectorType>, LinearSolveTiming), BvpLinearSolveError> {
        let (matrix_rows, matrix_columns) = self.shape();
        if matrix_rows != matrix_columns || vec.len() != matrix_rows {
            return Err(BvpLinearSolveError::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len: vec.len(),
            });
        }
        self.try_native_solve(vec)
    }

    fn solve_sys(
        &self,
        vec: &dyn VectorType,
        _linear_sys_method: Option<String>,
        _tol: f64,
        _max_iter: usize,
        _bandwidth: (usize, usize),
        _old_vec: &dyn VectorType,
    ) -> Box<dyn VectorType> {
        let solver = self.cached_solver.get_or_init(|| {
            let factor_begin = Instant::now();
            let system = if self.general_banded {
                LinearSystemRef::BandedAssembly(&self.assembly)
            } else {
                LinearSystemRef::NodeMajorAssembly {
                    assembly: &self.assembly,
                    layout: self.layout,
                }
            };
            let solver = build_solver_for_system(system, self.solver_config)
                .expect("banded linear solver factorization failed");
            let _ = self.factorization_duration.set(factor_begin.elapsed());
            solver
        });
        let mut rhs = dense_rhs_vec(vec);

        let rhs_begin = Instant::now();
        solver
            .solve_in_place(rhs.as_mut_slice())
            .expect("banded linear solver failed");
        self.rhs_solve_duration_nanos.fetch_add(
            rhs_begin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );

        Box::new(DVector::from_vec(rhs))
    }

    fn solve_sys_with_timing(
        &self,
        vec: &dyn VectorType,
        linear_sys_method: Option<String>,
        tol: f64,
        max_iter: usize,
        bandwidth: (usize, usize),
        old_vec: &dyn VectorType,
    ) -> (Box<dyn VectorType>, LinearSolveTiming) {
        let factorization_before = self.factorization_duration();
        let rhs_before = self.rhs_solve_duration();
        let result = self.solve_sys(vec, linear_sys_method, tol, max_iter, bandwidth, old_vec);
        (
            result,
            LinearSolveTiming {
                factorization: self
                    .factorization_duration()
                    .saturating_sub(factorization_before),
                rhs_solve: self.rhs_solve_duration().saturating_sub(rhs_before),
            },
        )
    }

    fn factorization_ready(&self) -> bool {
        self.cached_solver.get().is_some()
    }

    fn factorization_duration(&self) -> Duration {
        BandedMatrixType::factorization_duration(self)
    }

    fn rhs_solve_duration(&self) -> Duration {
        BandedMatrixType::rhs_solve_duration(self)
    }

    fn shape(&self) -> (usize, usize) {
        (self.assembly.n(), self.assembly.n())
    }

    fn to_DMatrixType(&self) -> DMatrix<f64> {
        let n = self.assembly.n();
        let mut dense = DMatrix::zeros(n, n);

        for offset in self.assembly.min_offset()..=self.assembly.max_offset() {
            if let Some(diag) = self.assembly.diag(offset) {
                for (pos, value) in diag.iter().enumerate() {
                    let (i, j) = self
                        .assembly
                        .diag_pos_to_ij(offset, pos)
                        .expect("banded diagonal position must stay in bounds");
                    dense[(i, j)] = *value;
                }
            }
        }

        dense
    }
}

impl Debug for dyn MatrixType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dense) = self.as_any().downcast_ref::<DMatrix<f64>>() {
            write!(f, "{:?}", dense)
        } else if let Some(sparse) = self.as_any().downcast_ref::<CsMat<f64>>() {
            write!(f, "{:?}", sparse)
        } else if let Some(faer_mat) = self.as_any().downcast_ref::<faer_mat>() {
            write!(f, "{:?}", faer_mat)
        } else if let Some(banded) = self.as_any().downcast_ref::<BandedMatrixType>() {
            write!(f, "{:?}", banded)
        } else {
            write!(f, "Unknown MatrixType")
        }
    }
}
////////////////////////////////////////////////////////////////////////
//  JACOBIAN MATRIX-FUNCTION
////////////////////////////////////////////////////////////////
