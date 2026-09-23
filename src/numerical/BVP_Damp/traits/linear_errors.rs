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
