use thiserror::Error;

#[derive(Debug, Error)]
pub enum BandedError {
    #[error("dimension mismatch")]
    DimensionMismatch,

    #[error("invalid band parameters: n={n}, kl={kl}, ku={ku}")]
    InvalidBand { n: usize, kl: usize, ku: usize },

    #[error("index out of band or out of bounds: i={i}, j={j}")]
    OutOfBounds { i: usize, j: usize },

    #[error("singular matrix at pivot index {index}")]
    Singular { index: usize },

    #[error("zero or numerically tiny pivot at index {index}, value={value}")]
    ZeroPivot { index: usize, value: f64 },

    #[error("solver has not been factorized")]
    NotFactorized,

    #[error("invalid RHS layout: rhs_len={rhs_len}, n={n}, nrhs={nrhs}, ldb={ldb}")]
    InvalidRhsLayout {
        rhs_len: usize,
        n: usize,
        nrhs: usize,
        ldb: usize,
    },

    #[error("LAPACK backend not enabled")]
    BackendUnavailable,

    #[error("LAPACK routine {routine} reported invalid argument at position {arg_index}")]
    LapackArgument {
        routine: &'static str,
        arg_index: i32,
    },
}
