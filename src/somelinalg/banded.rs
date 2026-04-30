pub mod banded_assembly;
pub mod block_tridiag_bench_helpers;
pub mod block_tridiagonal;
pub mod block_tridiagonal_lu;
pub mod block_tridiagonal_lu_consistent;
pub mod dense_block_kernels;
pub mod error;
pub mod general_lu;
pub mod general_lu_partial_pivot;
pub mod lapack_backend;
pub mod lapack_style_banded;
//#[path = "banded/lapack_style_banded_legathy"]
//pub mod lapack_style_banded_legathy;
pub mod linear_solver;
pub mod lu_storage;
pub mod node_major_layout;
pub mod ops;
pub mod pure_tridiagonal;
pub mod solver_factory;
pub mod solver_policy;
pub mod solver_traits;
pub mod storage;
pub mod superblock_layout;
pub mod tests;

pub use error::BandedError;
pub use general_lu::GeneralBandedLuNoPivot;
pub use general_lu_partial_pivot::GeneralBandedLuPartialPivot;
pub use lapack_backend::{BandedLu, GeneralBandedSolver};
pub use lapack_style_banded::LapackStyleBandedLuFaithful;
pub use ops::{
    banded_matvec, banded_to_dense, dense_diff_linf, dense_matmul, dense_matvec, residual_l2,
    residual_linf, vec_diff_linf,
};
pub use storage::Banded;

pub use block_tridiagonal::BlockTridiagonal;
pub use dense_block_kernels::{
    dense_lu_pivot_in_place, dense_lu_pivot_right_solve_block_in_place,
    dense_lu_pivot_solve_in_place, dense_lu_pivot_solve_transpose_in_place,
};

pub use block_tridiagonal_lu::BlockTridiagonalLu;
pub use block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent;
pub use solver_traits::{DirectLinearSolver, FaerSparseLuSolver};

pub use solver_policy::{FallbackPolicy, LinearSolveError, LinearSolverConfig, LinearSolverPolicy};

pub use linear_solver::{LinearSolver, LinearSystemRef, build_solver_for_system};
pub use node_major_layout::NodeMajorLayout;
pub use solver_factory::factor_block_tridiagonal;
