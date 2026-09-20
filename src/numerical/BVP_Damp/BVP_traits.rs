#![allow(non_camel_case_types)] //
use crate::somelinalg::RustedLINPACK::lu_band_nalg::LU_nalgebra;
use crate::somelinalg::banded::{
    LinearSystemRef, NodeMajorLayout, banded_assembly::BandedAssembly,
    linear_solver::build_solver_for_system, solver_policy::LinearSolverConfig,
    solver_traits::DirectLinearSolver,
};
use crate::somelinalg::iterative_solvers_cpu::LUsolver::invers_Mat_LU;
use crate::somelinalg::iterative_solvers_cpu::Lx_eq_b::{solve_csmat, solve_sys_SparseColMat};
use crate::somelinalg::iterative_solvers_cpu::some_matrix_inv::invers_csmat;

use faer::col::{Col, ColRef};
use faer::linalg::solvers::Solve;
use faer::mat::Mat;
use faer::mat::MatRef;

use faer::sparse::{SparseColMat, Triplet};
use nalgebra::{DMatrix, DVector, Matrix};
use sprs::{CsMat, CsVec, TriMat};
use std::any::Any;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::ops::Sub;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
type faer_mat = SparseColMat<usize, f64>;
type faer_col = Col<f64>; // Mat<f64>;

/// Typed timing for one matrix/RHS solve.
///
/// `factorization` measures construction of a reusable factor for this
/// operation, while `rhs_solve` measures applying that factor to the RHS.
/// Backends which cannot expose these stages keep the compatibility fallback
/// in [`MatrixType::solve_sys_with_timing`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearSolveTiming {
    pub factorization: Duration,
    pub rhs_solve: Duration,
}

/// Fallible error boundary for linear solves owned by the typed BVP path.
///
/// `MatrixType::solve_sys` predates the typed solver pipeline and therefore
/// returns a bare vector, with several implementations reporting bad input
/// through `panic!`/`unwrap()`.  New code should use `try_solve_sys`; the
/// legacy method remains available for downstream compatibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BvpLinearSolveError {
    DimensionMismatch {
        matrix_rows: usize,
        matrix_columns: usize,
        rhs_len: usize,
    },
    UnsupportedLinearPolicy {
        backend: &'static str,
        policy: String,
    },
    VectorTypeMismatch {
        backend: &'static str,
        expected: &'static str,
    },
    FactorizationFailed {
        backend: &'static str,
        message: String,
    },
    SolveFailed {
        backend: &'static str,
        message: String,
    },
    LegacyPanic {
        backend: &'static str,
        message: String,
    },
}

impl std::fmt::Display for BvpLinearSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                matrix_rows,
                matrix_columns,
                rhs_len,
            } => write!(
                f,
                "linear system dimension mismatch: matrix={matrix_rows}x{matrix_columns}, rhs={rhs_len}"
            ),
            Self::UnsupportedLinearPolicy { backend, policy } => {
                write!(
                    f,
                    "backend {backend} does not support linear policy {policy:?}"
                )
            }
            Self::VectorTypeMismatch { backend, expected } => {
                write!(f, "backend {backend} requires vector type {expected}")
            }
            Self::FactorizationFailed { backend, message } => {
                write!(f, "backend {backend} factorization failed: {message}")
            }
            Self::SolveFailed { backend, message } => {
                write!(f, "backend {backend} solve failed: {message}")
            }
            Self::LegacyPanic { backend, message } => {
                write!(
                    f,
                    "backend {backend} panicked in compatibility solve: {message}"
                )
            }
        }
    }
}

impl std::error::Error for BvpLinearSolveError {}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "linear backend panicked with a non-string payload".to_string()
    }
}

/// Fallible boundary for residual and Jacobian callback execution.
///
/// The original callback traits predate the typed solver pipeline and expose
/// only `call`, so backend type mismatches and user callback panics used to
/// escape as process-level panics.  The `try_call` methods below catch that
/// compatibility behaviour at the solver boundary without changing the
/// existing trait object ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BvpCallbackError {
    /// The callback was invoked with a vector/backend other than the one it
    /// was constructed for.
    TypeMismatch {
        expected: &'static str,
        actual: String,
    },
    /// The callback panicked while evaluating a residual or Jacobian.
    Panic { message: String },
    /// The callback returned a vector with an unexpected number of entries.
    OutputShapeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for BvpCallbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch { expected, actual } => {
                write!(f, "callback expected {expected} input, got {actual}")
            }
            Self::Panic { message } => write!(f, "callback panicked: {message}"),
            Self::OutputShapeMismatch { expected, actual } => write!(
                f,
                "callback output shape mismatch: expected length {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for BvpCallbackError {}

fn solve_dense_nalgebra_linear_system_timed(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    bandwidth: (usize, usize),
) -> (DVector<f64>, LinearSolveTiming) {
    let condition_for_banded_matrix = matrix.nrows() > 10 * (bandwidth.0 + bandwidth.1);
    if condition_for_banded_matrix {
        let factor_begin = Instant::now();
        let mut lu = LU_nalgebra::new(matrix.to_owned(), Some(bandwidth));
        lu.LU();
        let factorization = factor_begin.elapsed();
        let rhs_begin = Instant::now();
        let solution = lu.solve_linear_system_easy(rhs);
        (
            solution,
            LinearSolveTiming {
                factorization,
                rhs_solve: rhs_begin.elapsed(),
            },
        )
    } else {
        assert_eq!(
            matrix.nrows(),
            rhs.len(),
            "dimensions of matrix and vector must match"
        );
        let factor_begin = Instant::now();
        let lu = matrix.to_owned().lu();
        let factorization = factor_begin.elapsed();
        let rhs_begin = Instant::now();
        let solution = lu.solve(rhs).expect("dense nalgebra linear solve failed");
        (
            solution,
            LinearSolveTiming {
                factorization,
                rhs_solve: rhs_begin.elapsed(),
            },
        )
    }
}

fn solve_dense_nalgebra_linear_system(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    bandwidth: (usize, usize),
) -> DVector<f64> {
    solve_dense_nalgebra_linear_system_timed(matrix, rhs, bandwidth).0
}

/*
In RST BVP solvers there is an option to use different linear algebra crates
- Nalgebra
- SPRS
- FAER
- Nalgebra SPARSE
Generics in rust allow us to replace specific types with a placeholder that represents multiple types to remove code duplication.
So there are generic types:
 - VectorType (generic type that represents vectors of residuals and Newton steps)
 - MatrixType (generic type that represents matrices of jacobian)
 - Jac (generic type that represents functions for jacobian)
 - Fun (generic type that represents functions for vector)
 "method" variable (from the BVP task) is a keyword that is used to specify the crate that will be used for linear algebra operations
 "Dense" - nalgebra crate
 "Sparse" - faer crate
"Sparse_1" - sprs crate

*/
////////////////////////////////////////////////////////////////
//  VECTORTYPE - geneic type to store vectors of residuals and Newton steps
////////////////////////////////////////////////////////////////
pub enum YEnum {
    /// Dense Newton vector paired with a compact scalar-banded Jacobian.
    /// Keeping a distinct variant prevents the numeric Banded route from
    /// silently falling back to a dense Jacobian representation.
    Banded(DVector<f64>),
    Dense(DVector<f64>),  // dense vector NALGEBRA CRATE
    Sparse_1(CsVec<f64>), // sparse vector SPRS CRATE
    Sparse_3(faer_col),   // sparse vector  FAER CRATE
}
// basic funcionality for vectors
pub trait VectorType: Any {
    fn as_any(&self) -> &dyn Any;
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType>; //
    fn norm(&self) -> f64; // norm of vector
    fn to_DVectorType(&self) -> DVector<f64>; // convert to dense vector
    fn clone_box(&self) -> Box<dyn VectorType>; // cloning the vector into box
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_>; // iterating over vector
    fn get_val(&self, index: usize) -> f64; // get the value by index
    fn mul_float(&self, float: f64) -> Box<dyn VectorType>; // float multiplication
    fn len(&self) -> usize; // vector length
    fn zeros(&self, len: usize) -> Box<dyn VectorType>; // creating zero vector of length len
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType>; // assign value to index
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType>;
    /// Reports the dense representation without forcing external implementors
    /// to add a new required trait method. Built-in types override this with
    /// an allocation-free match; the compatibility default keeps the legacy
    /// `vec_type` semantics for downstream implementations.
    fn is_dense(&self) -> bool {
        self.vec_type() == "Dense"
    }
    fn vec_type(&self) -> String;
}

/// Borrows the dense Newton state for built-in vector representations.
///
/// The Banded Lambdify callback historically normalized every input through
/// `to_DVectorType`, which cloned the already-dense `DVector` hidden inside
/// both `DVector` and `YEnum::Banded`. External vector implementations still
/// use the compatibility allocation fallback.
#[inline]
pub(crate) fn dense_slice_or_copy<'a>(vector: &'a dyn VectorType) -> Cow<'a, [f64]> {
    if let Some(dense) = vector.as_any().downcast_ref::<DVector<f64>>() {
        Cow::Borrowed(dense.as_slice())
    } else {
        Cow::Owned(vector.to_DVectorType().as_slice().to_vec())
    }
}

impl Sub for &dyn VectorType {
    type Output = Box<dyn VectorType>;
    fn sub(self, other: Self) -> Self::Output {
        self.subtract(other)
    }
}

impl Iterator for YEnum {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            YEnum::Dense(vec) => vec.iter().next().copied(),
            YEnum::Banded(vec) => vec.iter().next().copied(),
            YEnum::Sparse_1(vec) => vec.iter().map(|x| *x.1).next(),
            YEnum::Sparse_3(vec) => vec.iter().next().copied(),
        }
    }
}

impl VectorType for YEnum {
    fn as_any(&self) -> &dyn Any {
        match self {
            YEnum::Dense(vec) => vec,
            YEnum::Banded(vec) => vec,
            YEnum::Sparse_1(vec) => vec,
            YEnum::Sparse_3(vec) => vec,
        }
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => {
                let other_dense = other.to_DVectorType();
                Box::new(vec - other_dense)
            }
            YEnum::Banded(vec) => {
                let other_dense = other.to_DVectorType();
                Box::new(YEnum::Banded(vec - other_dense))
            }
            YEnum::Sparse_1(vec) => {
                if let Some(c_vec) = other.as_any().downcast_ref::<CsVec<f64>>() {
                    Box::new(vec - c_vec)
                } else {
                    panic!("Type mismatch: expected CsVec")
                }
            }
            YEnum::Sparse_3(vec) => {
                let other_dense = other.to_DVectorType();
                assert_eq!(vec.len(), other_dense.len());
                let other_faer = faer_col::from_fn(other_dense.len(), |i| other_dense[i]);
                let subs = vec.sub(&other_faer);
                Box::new(subs)
            }
        }
    } // subtract
    fn norm(&self) -> f64 {
        match self {
            YEnum::Dense(vec) => Matrix::norm(vec),
            YEnum::Banded(vec) => Matrix::norm(vec),
            YEnum::Sparse_1(vec) => CsVec::l2_norm(vec),
            YEnum::Sparse_3(vec) => vec.norm_l2(),
        }
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        match self {
            YEnum::Dense(vec) => vec.clone(),
            YEnum::Banded(vec) => vec.clone(),
            YEnum::Sparse_1(vec) => {
                let length = vec.dim();

                DVector::from_iterator(
                    length,
                    (0..length).map(|index| vec.get(index).copied().unwrap_or(0.0)),
                )
            }
            YEnum::Sparse_3(vec) => {
                let length = vec.nrows();

                DVector::from_iterator(length, (0..length).map(|index| vec[index]))
            }
        }
    } //to_iterator
    fn clone_box(&self) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => Box::new(vec.clone()),
            YEnum::Banded(vec) => Box::new(YEnum::Banded(vec.clone())),
            YEnum::Sparse_1(vec) => Box::new(vec.clone()),
            YEnum::Sparse_3(vec) => Box::new(vec.clone()),
        }
    }

    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            YEnum::Dense(vec) => Box::new(vec.iter().map(|x| *x)),
            YEnum::Banded(vec) => Box::new(vec.iter().map(|x| *x)),
            YEnum::Sparse_1(vec) => Box::new(vec.iter().map(|x| *x.1)),
            YEnum::Sparse_3(vec) => Box::new(vec.iter().map(|x| *x)),
        }
    }
    fn get_val(&self, index: usize) -> f64 {
        match self {
            YEnum::Dense(vec) => vec[index],
            YEnum::Banded(vec) => vec[index],
            YEnum::Sparse_1(vec) => vec[index],
            YEnum::Sparse_3(vec) => vec[index],
        }
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => Box::new(vec * float),
            YEnum::Banded(vec) => Box::new(YEnum::Banded(vec * float)),
            YEnum::Sparse_1(vec) => Box::new(vec.map(|x| x * (float))),
            YEnum::Sparse_3(vec) => Box::new(vec * float),
        }
    }
    fn len(&self) -> usize {
        match self {
            YEnum::Dense(vec) => vec.len(),
            YEnum::Banded(vec) => vec.len(),
            YEnum::Sparse_1(vec) => vec.len(),
            YEnum::Sparse_3(vec) => vec.len(),
        }
    }
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(_) => Box::new(DVector::zeros(len)),
            YEnum::Banded(_) => Box::new(YEnum::Banded(DVector::zeros(len))),
            YEnum::Sparse_1(_) => Box::new(CsVec::empty(len)),
            YEnum::Sparse_3(_) => Box::new(faer_col::zeros(len)), // faer_col::zeros(len)
        }
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        match self {
            YEnum::Dense(vec) => {
                let mut new_vec = vec.clone();
                new_vec[index] = value;
                Box::new(new_vec)
            }
            YEnum::Banded(vec) => {
                let mut new_vec = vec.clone();
                new_vec[index] = value;
                Box::new(YEnum::Banded(new_vec))
            }
            YEnum::Sparse_1(vec) => {
                let mut new_vec = vec.clone();
                new_vec.append(index, value);
                Box::new(new_vec)
            }
            YEnum::Sparse_3(vec) => {
                let mut new_vec = vec.clone();
                new_vec[index] = value;
                Box::new(new_vec)
            }
        }
    } // end assign
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        match self {
            YEnum::Dense(_) => {
                let new_matrix: DMatrix<f64> =
                    DMatrix::from_row_slice(nrows, ncols, vec_with_zeros);
                Box::new(new_matrix)
            }
            YEnum::Banded(_) => {
                assert_eq!(nrows, ncols, "banded Jacobian must be square");
                let mut lower = 0usize;
                let mut upper = 0usize;
                for &(row, col, value) in &non_zero_triplet {
                    if value == 0.0 {
                        continue;
                    }
                    assert!(
                        row < nrows && col < ncols,
                        "banded triplet is out of bounds"
                    );
                    lower = lower.max(row.saturating_sub(col));
                    upper = upper.max(col.saturating_sub(row));
                }

                let mut assembly = BandedAssembly::zeros(nrows, lower, upper)
                    .expect("numeric Banded Jacobian dimensions must be valid");
                for (row, col, value) in non_zero_triplet {
                    if value != 0.0 {
                        assembly
                            .set(row, col, value)
                            .expect("numeric Banded Jacobian triplet must fit its inferred band");
                    }
                }
                let layout = NodeMajorLayout::new(nrows, 1)
                    .expect("numeric Banded Jacobian layout must be non-empty");
                Box::new(BandedMatrixType::new_general_banded(
                    assembly,
                    layout,
                    LinearSolverConfig::faithful_banded(),
                ))
            }
            YEnum::Sparse_1(_) => {
                let new_matrix = sprs_triplet_to_csc(nrows, ncols, &non_zero_triplet);
                Box::new(new_matrix)
            }
            YEnum::Sparse_3(_) => {
                let triplet: Vec<Triplet<usize, usize, f64>> = non_zero_triplet
                    .iter()
                    .map(|triplet| Triplet::new(triplet.0, triplet.1, triplet.2))
                    .collect::<Vec<_>>();
                let new_matrix: SparseColMat<usize, f64> =
                    SparseColMat::try_new_from_triplets(nrows, ncols, triplet.as_slice()).unwrap();
                Box::new(new_matrix)
            }
        }
    } //from_vector
    fn is_dense(&self) -> bool {
        matches!(self, YEnum::Dense(_))
    }
    fn vec_type(&self) -> String {
        match self {
            YEnum::Dense(_) => "Dense".to_string(),
            YEnum::Banded(_) => "Banded".to_string(),
            YEnum::Sparse_1(_) => "Sparse_1".to_string(),
            YEnum::Sparse_3(_) => "Sparse_3".to_string(),
        }
    }
} //end impl

////////////////////////////////////////////////////////////////
//           NALGEBRA CRATE
////////////////////////////////////////////////////////////////
impl VectorType for DVector<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = other.as_any().downcast_ref::<DVector<f64>>() {
            Box::new(self - d_vec)
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
    fn norm(&self) -> f64 {
        Matrix::norm(self)
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        self.clone()
    }
    fn clone_box(&self) -> Box<dyn VectorType> {
        Box::new(self.clone())
    }
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(self.iter().map(|x| *x))
    }
    fn get_val(&self, index: usize) -> f64 {
        self[index]
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        Box::new(self * float)
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        Box::new(DVector::zeros(len))
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        Box::new({
            let mut vec = self.clone();
            vec[index] = value;
            vec
        })
    }
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        vec_with_zeros: &Vec<f64>,
        _non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        let new_matrix: DMatrix<f64> = DMatrix::from_row_slice(nrows, ncols, vec_with_zeros);
        Box::new(new_matrix)
    }
    fn is_dense(&self) -> bool {
        true
    }
    fn vec_type(&self) -> String {
        "Dense".to_string()
    }
}
////////////////////////////////
//           SPRS CRATE
////////////////////////////////
impl VectorType for CsVec<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(c_vec) = other.as_any().downcast_ref::<CsVec<f64>>() {
            Box::new(self - c_vec)
        } else {
            panic!("Type mismatch: expected CsVec")
        }
    }
    fn norm(&self) -> f64 {
        CsVec::l2_norm(self)
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        let length = self.dim();

        DVector::from_iterator(
            length,
            (0..length).map(|index| self.get(index).copied().unwrap_or(0.0)),
        )
    }
    fn clone_box(&self) -> Box<dyn VectorType> {
        Box::new(self.clone())
    }
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(self.iter().map(|x| *x.1))
    }
    fn get_val(&self, index: usize) -> f64 {
        self[index]
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        Box::new(self.map(|x| x * (float)))
    }
    fn len(&self) -> usize {
        self.dim()
    }
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        Box::new(CsVec::empty(len))
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        Box::new({
            let mut new_vec = self.clone();
            new_vec.append(index, value);
            new_vec
        })
    }
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        _vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        let new_matrix = sprs_triplet_to_csc(nrows, ncols, &non_zero_triplet);
        Box::new(new_matrix)
    }
    fn is_dense(&self) -> bool {
        false
    }
    fn vec_type(&self) -> String {
        "Sparse_1".to_string()
    }
}
////////////////////////////////////////////////////////////////////////////
//  FAER CRATE
////////////////////////////////////////////////////////////////////////////
impl VectorType for faer_col {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn subtract(&self, other: &dyn VectorType) -> Box<dyn VectorType> {
        let other_dense = other.to_DVectorType();
        assert_eq!(self.nrows(), other_dense.len());
        let other_faer = faer_col::from_fn(other_dense.len(), |i| other_dense[i]);
        let subs = self.sub(&other_faer);
        Box::new(subs)
    }
    fn norm(&self) -> f64 {
        self.norm_l2()
    }
    fn to_DVectorType(&self) -> DVector<f64> {
        let length = self.nrows();

        DVector::from_iterator(length, (0..length).map(|index| self[index]))
    }
    fn clone_box(&self) -> Box<dyn VectorType> {
        Box::new(self.clone())
    }
    fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(self.iter().map(|x| *x))
    }
    fn get_val(&self, index: usize) -> f64 {
        self[index]
    }
    fn mul_float(&self, float: f64) -> Box<dyn VectorType> {
        Box::new(self * float)
    }
    fn len(&self) -> usize {
        self.nrows()
    }
    fn zeros(&self, len: usize) -> Box<dyn VectorType> {
        Box::new(faer_col::zeros(len)) //faer_col::zeros(len)
    }
    fn assign_value(&self, index: usize, value: f64) -> Box<dyn VectorType> {
        Box::new({
            let mut new_vec = self.clone();
            new_vec[index] = value;
            new_vec
        })
    }
    fn from_vector(
        &self,
        nrows: usize,
        ncols: usize,
        _vec_with_zeros: &Vec<f64>,
        non_zero_triplet: Vec<(usize, usize, f64)>,
    ) -> Box<dyn MatrixType> {
        let non_zero_triplet: Vec<Triplet<usize, usize, f64>> = non_zero_triplet
            .iter()
            .map(|triplet| Triplet::new(triplet.0, triplet.1, triplet.2))
            .collect::<Vec<_>>();
        let new_matrix: SparseColMat<usize, f64> =
            SparseColMat::try_new_from_triplets(nrows, ncols, non_zero_triplet.as_slice()).unwrap();
        Box::new(new_matrix)
    }
    fn is_dense(&self) -> bool {
        false
    }
    fn vec_type(&self) -> String {
        "Sparse_3".to_string()
    }
}

/////////////////////////////////////////////////////////////
//      FUN - VECTOR-FUNCTION OF RESIDUALS
///////////////////////////////////////////////////////////////
pub trait Fun {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType>;

    /// Fallible compatibility entry point for the typed solver path.
    fn try_call(
        &self,
        x: f64,
        vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpCallbackError> {
        catch_unwind(AssertUnwindSafe(|| self.call(x, vec))).map_err(|payload| {
            BvpCallbackError::Panic {
                message: panic_message(payload),
            }
        })
    }
}

pub enum FunEnum {
    Dense(Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>),
    Sparse_1(Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64>>),
    Sparse_3(Box<dyn Fn(f64, &faer_col) -> faer_col>),
}

impl Fun for FunEnum {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        match self {
            FunEnum::Dense(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected DVector")
                }
            }
            FunEnum::Sparse_1(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected CsVec")
                }
            }
            FunEnum::Sparse_3(fun) => {
                if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
                    Box::new(fun(x, d_vec))
                } else {
                    panic!("Type mismatch: expected faer_col")
                }
            }
        }
    }

    fn try_call(
        &self,
        x: f64,
        vec: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpCallbackError> {
        match self {
            FunEnum::Dense(fun) => {
                let d_vec = vec.as_any().downcast_ref::<DVector<f64>>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "DVector",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(fun(x, d_vec)) as Box<dyn VectorType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
            FunEnum::Sparse_1(fun) => {
                let sparse_vec = vec.as_any().downcast_ref::<CsVec<f64>>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "CsVec",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(fun(x, sparse_vec)) as Box<dyn VectorType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
            FunEnum::Sparse_3(fun) => {
                let faer_vec = vec.as_any().downcast_ref::<faer_col>().ok_or_else(|| {
                    BvpCallbackError::TypeMismatch {
                        expected: "faer_col",
                        actual: vec.vec_type(),
                    }
                })?;
                catch_unwind(AssertUnwindSafe(|| {
                    Box::new(fun(x, faer_vec)) as Box<dyn VectorType>
                }))
                .map_err(|payload| BvpCallbackError::Panic {
                    message: panic_message(payload),
                })
            }
        }
    }
}

impl Fun for dyn Fn(f64, &DVector<f64>) -> DVector<f64> {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<DVector<f64>>() {
            Box::new(self(x, d_vec))
        } else {
            panic!("Type mismatch: expected DVector")
        }
    }
}
impl Fun for dyn Fn(f64, &faer_col) -> faer_col {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<faer_col>() {
            Box::new(self(x, d_vec))
        } else {
            panic!("Type mismatch: expected faer_col")
        }
    }
}
impl Fun for dyn Fn(f64, &CsVec<f64>) -> CsVec<f64> {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        if let Some(d_vec) = vec.as_any().downcast_ref::<CsVec<f64>>() {
            Box::new(self(x, d_vec))
        } else {
            panic!("Type mismatch: expected CsVec")
        }
    }
}
// First, create a wrapper struct for the function
pub struct FunctionWrapper(Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>>);

// Implement Fun for the wrapper
impl Fun for FunctionWrapper {
    fn call(&self, x: f64, vec: &dyn VectorType) -> Box<dyn VectorType> {
        (self.0)(x, vec)
    }
}

// Then create a conversion function
pub fn convert_to_fun(f: Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>>) -> Box<dyn Fun> {
    Box::new(FunctionWrapper(f))
}

impl fmt::Display for YEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YEnum::Dense(vec) => write!(f, "Dense Vector: {:?}", vec),
            YEnum::Banded(vec) => write!(f, "Banded Vector: {:?}", vec),
            YEnum::Sparse_1(vec) => write!(f, "Sparse Vector 1: {:?}", vec),
            YEnum::Sparse_3(vec) => write!(f, "Sparse Vector 3: {:?}", vec),
        }
    }
}

impl fmt::Display for dyn VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dense) = self.as_any().downcast_ref::<DVector<f64>>() {
            write!(f, "{}", dense)
        } else if let Some(sparse) = self.as_any().downcast_ref::<CsVec<f64>>() {
            write!(f, "{:?}", sparse)
        } else if let Some(y_enum) = self.as_any().downcast_ref::<YEnum>() {
            write!(f, "{}", y_enum)
        } else {
            write!(f, "Unknown VectorType")
        }
    }
}
impl Debug for dyn VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dense) = self.as_any().downcast_ref::<DVector<f64>>() {
            write!(f, "{:?}", dense)
        } else if let Some(sparse) = self.as_any().downcast_ref::<CsVec<f64>>() {
            write!(f, "{:?}", sparse)
        } else if let Some(sparse) = self.as_any().downcast_ref::<faer_col>() {
            write!(f, "{:?}", sparse)
        } else {
            write!(f, "Unknown VectorType")
        }
    }
}

////////////////////////////////////////////////////////////////////////////
//_________________________________Jacobian______________________________
////////////////////////////////////////////////////////////////////////////
///  NUMERICAL REPRESENTATION OF JACOBIAN FOR DIFFERENT CRATES
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
    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vec: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Box<dyn VectorType>;
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
pub trait Y: Any {
    fn as_any(&self) -> Box<&dyn Any>;
    fn clone_box(&self) -> Box<dyn Y>;
}

impl Y for DVector<f64> {
    fn as_any(&self) -> Box<&dyn Any> {
        Box::new(self)
    }

    fn clone_box(&self) -> Box<dyn Y> {
        Box::new(self.clone())
    }
}

impl Y for CsVec<f64> {
    fn as_any(&self) -> Box<&dyn Any> {
        Box::new(self)
    }

    fn clone_box(&self) -> Box<dyn Y> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Y> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
///////////////////////////////
//     Miscellaneous
///////////////////////////////
//___________________________________________________________________
pub fn Vectors_type_casting(vec: &DVector<f64>, desired_type: String) -> Box<dyn VectorType> {
    let res: Box<dyn VectorType> = if desired_type == "Dense".to_string() {
        Box::new(YEnum::Dense(vec.clone()))
    } else if desired_type == "Banded".to_string() {
        // Newton vectors remain dense, but the distinct variant ensures that
        // Jacobian assembly selects compact BandedAssembly instead of DMatrix.
        Box::new(YEnum::Banded(vec.clone()))
    } else if desired_type == "Sparse_1".to_string() {
        //sprs crate
        let mut ind = Vec::new();
        let mut val = Vec::new();
        vec.iter().enumerate().for_each(|(i, x)| {
            if x.abs() > 0.0 {
                ind.push(i);
                val.push(*x);
            }
        });
        Box::new(YEnum::Sparse_1(
            CsVec::new_from_unsorted(vec.len(), ind, val).expect("trouble with initial vector!"),
        ))
    } else if desired_type == "Sparse".to_string() {
        // faer crate
        let Mat_vec = ColRef::from_slice(vec.as_slice()).to_owned();

        Box::new(YEnum::Sparse_3(Mat_vec))
    } else {
        panic!("Unsupported vector type: {}", desired_type);
    };
    res
}

#[cfg(test)]
mod y_trait_object_clone_tests {
    use super::*;

    #[test]
    fn dense_callback_view_borrows_builtin_newton_state() {
        let vector: Box<dyn VectorType> =
            Box::new(YEnum::Banded(DVector::from_vec(vec![1.0, 2.0])));
        let view = dense_slice_or_copy(&*vector);

        assert!(matches!(&view, Cow::Borrowed(_)));
        assert_eq!(view.as_ref(), &[1.0, 2.0]);
    }

    #[test]
    fn boxed_y_clone_preserves_dense_payload_without_recursion() {
        let original: Box<dyn Y> = Box::new(DVector::from_vec(vec![1.0, -2.0, 3.5]));
        let cloned = original.clone();
        let any = cloned.as_any();
        let dense = (*any)
            .downcast_ref::<DVector<f64>>()
            .expect("cloned Box<dyn Y> should keep dense payload type");

        assert_eq!(dense.as_slice(), &[1.0, -2.0, 3.5]);
    }

    #[test]
    fn boxed_y_clone_preserves_sparse_payload_without_recursion() {
        let sparse = CsVec::new(5, vec![0, 3], vec![2.0, -4.0]);
        let original: Box<dyn Y> = Box::new(sparse);
        let cloned = original.clone();
        let any = cloned.as_any();
        let sparse = (*any)
            .downcast_ref::<CsVec<f64>>()
            .expect("cloned Box<dyn Y> should keep sparse payload type");

        assert_eq!(sparse.dim(), 5);
        assert_eq!(sparse.indices(), &[0, 3]);
        assert_eq!(sparse.data(), &[2.0, -4.0]);
    }

    #[test]
    fn sparse_vector_to_dense_preserves_implicit_zero_coordinates() {
        let expected = [2.0, 0.0, 0.0, -4.0, 0.0];

        let sprs = CsVec::new(5, vec![0, 3], vec![2.0, -4.0]);
        let sprs_direct = sprs.to_DVectorType();
        let sprs_enum = YEnum::Sparse_1(sprs).to_DVectorType();
        assert_eq!(sprs_direct.as_slice(), &expected);
        assert_eq!(sprs_enum.as_slice(), &expected);

        let faer = faer_col::from_fn(5, |index| match index {
            0 => 2.0,
            3 => -4.0,
            _ => 0.0,
        });
        let faer_direct = faer.to_DVectorType();
        let faer_enum = YEnum::Sparse_3(faer).to_DVectorType();
        assert_eq!(faer_direct.as_slice(), &expected);
        assert_eq!(faer_enum.as_slice(), &expected);
    }

    #[test]
    fn sparse_matrix_adapters_preserve_shape_and_coordinates() {
        let triplets = vec![(0, 1, 2.0), (0, 1, 3.0), (2, 0, -4.0), (1, 1, 0.0)];
        let dense_values = vec![0.0; 9];

        let sprs_matrix = from_vector_to_matrix(
            "Sparse_1".to_string(),
            3,
            3,
            &dense_values,
            triplets.clone(),
        );
        let sprs_matrix = sprs_matrix
            .as_any()
            .downcast_ref::<CsMat<f64>>()
            .expect("Sparse_1 adapter should produce CsMat");
        assert_eq!(sprs_matrix.shape(), (3, 3));
        assert_eq!(sprs_matrix.get(0, 1), Some(&5.0));
        assert_eq!(sprs_matrix.get(2, 0), Some(&-4.0));
        assert_eq!(sprs_matrix.get(1, 1).copied().unwrap_or(0.0), 0.0);

        let faer_matrix =
            from_vector_to_matrix("Sparse_3".to_string(), 3, 3, &dense_values, triplets);
        let faer_matrix = faer_matrix
            .as_any()
            .downcast_ref::<SparseColMat<usize, f64>>()
            .expect("Sparse_3 adapter should produce faer SparseColMat");
        assert_eq!((faer_matrix.nrows(), faer_matrix.ncols()), (3, 3));
        assert_eq!(faer_matrix.get(0, 1), Some(&5.0));
        assert_eq!(faer_matrix.get(2, 0), Some(&-4.0));
        assert_eq!(faer_matrix.get(1, 1).copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn finite_difference_jacobian_reports_callback_shape_mismatch() {
        let fun = FunEnum::Dense(Box::new(|_, _| DVector::from_element(1, 0.0)));
        let vec = DVector::from_vec(vec![1.0, 2.0]);

        let error = try_finite_difference_jacobian(&fun, 0.0, &vec, 1e-8)
            .expect_err("malformed finite-difference residual must be typed");
        assert_eq!(
            error,
            BvpCallbackError::OutputShapeMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn callback_try_call_captures_legacy_panic() {
        let fun = FunEnum::Dense(Box::new(|_, _| {
            panic!("callback panic for typed-boundary test")
        }));
        let vec = DVector::from_element(1, 0.0);

        let error = fun
            .try_call(0.0, &vec)
            .expect_err("callback panic must be captured by try_call");
        assert!(matches!(
            error,
            BvpCallbackError::Panic { message }
                if message.contains("callback panic for typed-boundary test")
        ));
    }

    #[test]
    fn callback_try_call_reports_builtin_backend_mismatch() {
        let fun = FunEnum::Dense(Box::new(|_, value| value.clone()));
        let sparse = CsVec::new(1, vec![0], vec![1.0]);

        let error = fun
            .try_call(0.0, &sparse)
            .expect_err("a Dense callback must reject a sparse vector explicitly");
        assert!(matches!(
            error,
            BvpCallbackError::TypeMismatch { expected, actual }
                if expected == "DVector" && actual == "Sparse_1"
        ));
    }

    #[test]
    fn jacobian_try_call_reports_builtin_backend_mismatch() {
        let mut jacobian = JacEnum::Dense(Box::new(|_, value| {
            DMatrix::identity(value.len(), value.len())
        }));
        let sparse = CsVec::new(1, vec![0], vec![1.0]);

        let error = jacobian
            .try_call(0.0, &sparse)
            .expect_err("a Dense Jacobian callback must reject a sparse vector explicitly");
        assert!(matches!(
            error,
            BvpCallbackError::TypeMismatch { expected, actual }
                if expected == "DVector" && actual == "Sparse_1"
        ));
    }

    #[test]
    fn jacobian_try_call_captures_legacy_panic() {
        let mut jacobian = JacEnum::Dense(Box::new(|_, _| {
            panic!("Jacobian panic for typed-boundary test")
        }));
        let vector = DVector::from_element(1, 0.0);

        let error = jacobian
            .try_call(0.0, &vector)
            .expect_err("Jacobian callback panic must be captured by try_call");
        assert!(matches!(
            error,
            BvpCallbackError::Panic { message }
                if message.contains("Jacobian panic for typed-boundary test")
        ));
    }
}
pub fn from_vector_to_matrix(
    vec_type: String,
    nrows: usize,
    ncols: usize,
    vec_with_zeros: &Vec<f64>,
    non_zero_triplet: Vec<(usize, usize, f64)>,
) -> Box<dyn MatrixType> {
    match vec_type.as_str() {
        "Dense" => {
            let new_matrix: DMatrix<f64> = DMatrix::from_row_slice(nrows, ncols, vec_with_zeros);
            Box::new(new_matrix)
        }
        "Sparse_1" => {
            let new_matrix = sprs_triplet_to_csc(nrows, ncols, &non_zero_triplet);
            Box::new(new_matrix)
        }
        "Sparse_3" => {
            let non_zero_triplet: Vec<Triplet<usize, usize, f64>> = non_zero_triplet
                .iter()
                .map(|triplet| Triplet::new(triplet.0, triplet.1, triplet.2))
                .collect::<Vec<_>>();

            let new_matrix: SparseColMat<usize, f64> =
                SparseColMat::try_new_from_triplets(nrows, ncols, non_zero_triplet.as_slice())
                    .unwrap();
            Box::new(new_matrix)
        }
        _ => panic!("Unsupported matrix type: {}", vec_type),
    }
} //from_vecto

fn sprs_triplet_to_csc(
    nrows: usize,
    ncols: usize,
    triplets: &Vec<(usize, usize, f64)>,
) -> CsMat<f64> {
    let mut triplet_matrix = TriMat::new((nrows, ncols));
    for (i, j, v) in triplets {
        triplet_matrix.add_triplet(*i, *j, *v);
    }

    // Convert the triplet matrix to a CSC matrix
    let csc_matrix: CsMat<f64> = triplet_matrix.to_csc();
    csc_matrix
}
