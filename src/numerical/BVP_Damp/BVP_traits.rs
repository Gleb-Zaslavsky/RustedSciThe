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

include!("traits/linear_errors.rs");
include!("traits/vectors.rs");
include!("traits/callbacks.rs");
include!("traits/matrices.rs");
include!("traits/jacobian.rs");
include!("traits/compat.rs");
include!("traits/tests.rs");
include!("traits/tail.rs");
