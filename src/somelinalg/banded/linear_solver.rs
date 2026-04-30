//! Unified native linear-solver entry point for banded and structured BVP systems.
//!
//! The default general banded solver is `LapackStyleBandedLuFaithful`: a compact
//! band-storage implementation that mirrors LAPACK `DGBTRF`/`DGBTRS` semantics,
//! including the band-column choreography and pivot application order. In
//! practice this is the solver to start with whenever the matrix is "just
//! banded".
//!
//! Structured block-tridiagonal solvers remain available, but they are opt-in
//! tools for matrices that really have that block layout. The dense
//! general-pivot backend is intentionally exposed only as an explicit
//! diagnostic/legacy policy, not as the production banded default.
//!
//! Refinement is also opt-in. `LinearSolverConfig::default()` and
//! `LinearSolverConfig::faithful_banded()` use zero refinement steps because the
//! faithful backend is already accurate on the current BVP workloads and an
//! unconditional correction pass can cost time without improving the result.

use super::solver_traits::FaerSparseLuSolver;
use super::{
    banded_assembly::BandedAssembly, block_tridiagonal::BlockTridiagonal,
    block_tridiagonal_lu::BlockTridiagonalLu,
    block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent, error::BandedError,
    general_lu_partial_pivot::GeneralBandedLuPartialPivot,
    lapack_style_banded::LapackStyleBandedLuFaithful, node_major_layout::NodeMajorLayout,
    solver_traits::DirectLinearSolver, storage::Banded,
};
#[derive(Debug)]
pub enum LinearSolver {
    BlockTridiagonal(BlockTridiagonalLu),
    BlockTridiagonalConsistent {
        solver: BlockTridiagonalLuConsistent,
        matrix: BlockTridiagonal,
        iterative_refinement_steps: usize,
    },
    BandedLapack {
        solver: LapackStyleBandedLuFaithful,
        matrix: Banded<f64>,
        iterative_refinement_steps: usize,
    },
    DenseGeneralPivot(GeneralBandedLuPartialPivot),
    FaerSparse(FaerSparseLuSolver),
}

impl LinearSolver {
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::BlockTridiagonal(_) => "block_tridiagonal_lu",
            Self::BlockTridiagonalConsistent { .. } => "block_tridiagonal_lu_consistent",
            Self::BandedLapack {
                iterative_refinement_steps,
                ..
            } if *iterative_refinement_steps > 0 => "lapack_style_banded_lu+refine",
            Self::BandedLapack { .. } => "lapack_style_banded_lu",
            Self::DenseGeneralPivot(_) => "banded_lu_partial_pivot_dense",
            Self::FaerSparse(_) => "faer_sparse_lu",
        }
    }
}

impl DirectLinearSolver for LinearSolver {
    fn n(&self) -> usize {
        match self {
            Self::BlockTridiagonal(s) => s.n(),
            Self::BlockTridiagonalConsistent { solver, .. } => solver.n(),
            Self::BandedLapack { solver, .. } => solver.n(),
            Self::DenseGeneralPivot(s) => s.n(),
            Self::FaerSparse(s) => s.n(),
        }
    }

    fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), BandedError> {
        match self {
            Self::BlockTridiagonal(s) => s.solve_in_place(rhs),
            Self::BlockTridiagonalConsistent {
                solver,
                matrix,
                iterative_refinement_steps,
            } => solver.solve_in_place_with_iterative_refinement(
                matrix,
                rhs,
                *iterative_refinement_steps,
            ),
            Self::BandedLapack {
                solver,
                matrix,
                iterative_refinement_steps,
            } => {
                if *iterative_refinement_steps == 0 {
                    solver.solve_in_place(rhs)
                } else {
                    solver
                        .solve_banded_in_place_with_refinement(
                            matrix,
                            rhs,
                            *iterative_refinement_steps,
                            1e-12,
                        )
                        .map(|_| ())
                }
            }
            Self::DenseGeneralPivot(s) => s.solve_in_place(rhs),
            Self::FaerSparse(s) => s.solve_in_place(rhs),
        }
    }

    fn solve_multiple_in_place(
        &self,
        rhs: &mut [f64],
        nrhs: usize,
        ldb: usize,
    ) -> Result<(), BandedError> {
        match self {
            Self::BlockTridiagonal(s) => s.solve_multiple_in_place(rhs, nrhs, ldb),
            Self::BlockTridiagonalConsistent {
                solver,
                matrix,
                iterative_refinement_steps,
            } => {
                if ldb < solver.n() || rhs.len() < nrhs.saturating_mul(ldb) {
                    return Err(BandedError::DimensionMismatch);
                }
                for col in 0..nrhs {
                    let start = col * ldb;
                    let end = start + solver.n();
                    solver.solve_in_place_with_iterative_refinement(
                        matrix,
                        &mut rhs[start..end],
                        *iterative_refinement_steps,
                    )?;
                }
                Ok(())
            }
            Self::BandedLapack {
                solver,
                matrix,
                iterative_refinement_steps,
            } => {
                if ldb < solver.n() || rhs.len() < nrhs.saturating_mul(ldb) {
                    return Err(BandedError::DimensionMismatch);
                }
                for col in 0..nrhs {
                    let start = col * ldb;
                    let end = start + solver.n();
                    if *iterative_refinement_steps == 0 {
                        solver.solve_in_place(&mut rhs[start..end])?;
                    } else {
                        solver.solve_banded_in_place_with_refinement(
                            matrix,
                            &mut rhs[start..end],
                            *iterative_refinement_steps,
                            1e-12,
                        )?;
                    }
                }
                Ok(())
            }
            Self::DenseGeneralPivot(s) => s.solve_multiple_in_place(rhs, nrhs, ldb),
            Self::FaerSparse(s) => s.solve_multiple_in_place(rhs, nrhs, ldb),
        }
    }
}

//=======================================================================================================
use crate::somelinalg::banded::{
    solver_factory::factor_block_tridiagonal,
    solver_policy::{LinearSolveError, LinearSolverConfig, LinearSolverPolicy},
};
use faer::sparse::Triplet;

pub enum LinearSystemRef<'a> {
    BlockTridiagonal(&'a BlockTridiagonal),
    NodeMajorAssembly {
        assembly: &'a BandedAssembly,
        layout: NodeMajorLayout,
    },
    BandedAssembly(&'a BandedAssembly),
}

fn banded_to_triplets(
    assembly: &BandedAssembly,
) -> Result<Vec<Triplet<usize, usize, f64>>, BandedError> {
    let mut triplets = Vec::new();
    for offset in assembly.min_offset()..=assembly.max_offset() {
        if let Some(diag) = assembly.diag(offset) {
            for (pos, value) in diag.iter().enumerate() {
                if *value == 0.0 {
                    continue;
                }
                let (i, j) = assembly.diag_pos_to_ij(offset, pos)?;
                triplets.push(Triplet::new(i, j, *value));
            }
        }
    }
    Ok(triplets)
}

fn build_lapack_banded_solver(
    assembly: &BandedAssembly,
    iterative_refinement_steps: usize,
) -> Result<LinearSolver, LinearSolveError> {
    let compact = assembly.to_banded()?;
    let mut lu = LapackStyleBandedLuFaithful::new(compact.n(), compact.kl(), compact.ku())?;
    lu.factor_from(&compact)?;
    Ok(LinearSolver::BandedLapack {
        solver: lu,
        matrix: compact,
        iterative_refinement_steps,
    })
}

fn build_dense_general_banded_solver(
    assembly: &BandedAssembly,
) -> Result<LinearSolver, LinearSolveError> {
    let compact = assembly.to_banded()?;
    let mut lu = GeneralBandedLuPartialPivot::new(compact.n(), compact.kl(), compact.ku())?;
    lu.factor_from(&compact)?;
    Ok(LinearSolver::DenseGeneralPivot(lu))
}

fn build_faer_sparse_from_assembly(
    assembly: &BandedAssembly,
) -> Result<LinearSolver, LinearSolveError> {
    let triplets = banded_to_triplets(assembly)?;
    let solver = FaerSparseLuSolver::from_triplets(assembly.n(), &triplets)?;
    Ok(LinearSolver::FaerSparse(solver))
}

pub fn build_solver_for_system(
    system: LinearSystemRef<'_>,
    config: LinearSolverConfig,
) -> Result<LinearSolver, LinearSolveError> {
    match system {
        LinearSystemRef::BlockTridiagonal(block) => build_linear_solver(block, config),
        LinearSystemRef::NodeMajorAssembly { assembly, layout } => match config.policy {
            LinearSolverPolicy::ForceGeneralBandedPartialPivot => {
                build_dense_general_banded_solver(assembly)
            }
            LinearSolverPolicy::ForceBanded => {
                build_lapack_banded_solver(assembly, config.iterative_refinement_steps)
            }
            LinearSolverPolicy::ForceFaerSparse => build_faer_sparse_from_assembly(assembly),
            _ => {
                let block = assembly
                    .to_block_tridiagonal(layout.n_blocks(), layout.block_size())
                    .map_err(LinearSolveError::from)?;
                build_linear_solver(&block, config)
            }
        },
        LinearSystemRef::BandedAssembly(assembly) => match config.policy {
            LinearSolverPolicy::ForceGeneralBandedPartialPivot => {
                build_dense_general_banded_solver(assembly)
            }
            LinearSolverPolicy::ForceBanded | LinearSolverPolicy::Auto => {
                build_lapack_banded_solver(assembly, config.iterative_refinement_steps)
            }
            LinearSolverPolicy::ForceFaerSparse => build_faer_sparse_from_assembly(assembly),
            LinearSolverPolicy::ForceBlockTridiagonal
            | LinearSolverPolicy::ForceBlockTridiagonalConsistent => {
                Err(LinearSolveError::InvalidPolicy(
                    "block-tridiagonal solvers require layout metadata; use NodeMajorAssembly input",
                ))
            }
        },
    }
}
/// High-level entry point for building a linear solver from a Jacobian.
///
/// This is intended for Newton/BVP solvers.
pub fn build_linear_solver(
    jac: &BlockTridiagonal,
    config: LinearSolverConfig,
) -> Result<LinearSolver, LinearSolveError> {
    factor_block_tridiagonal(jac, config)
}
/// Performs one Newton linear solve step:
/// J * dx = -F
/// Solves J * dx = -residual in place.
/// On return, `residual` is overwritten with `dx`.
pub fn newton_solve_step(
    jac: &BlockTridiagonal,
    residual: &mut [f64],
    config: LinearSolverConfig,
) -> Result<(), LinearSolveError> {
    let solver = build_linear_solver(jac, config)?;

    // решаем J * dx = -F
    for v in residual.iter_mut() {
        *v = -*v;
    }

    solver
        .solve_in_place(residual)
        .map_err(LinearSolveError::from)?;

    Ok(())
}

pub struct CachedLinearSolver {
    solver: Option<LinearSolver>,
}

impl CachedLinearSolver {
    pub fn new() -> Self {
        Self { solver: None }
    }

    pub fn factor(
        &mut self,
        jac: &BlockTridiagonal,
        config: LinearSolverConfig,
    ) -> Result<(), LinearSolveError> {
        let solver = build_linear_solver(jac, config)?;
        self.solver = Some(solver);
        Ok(())
    }

    pub fn solve_in_place(&self, rhs: &mut [f64]) -> Result<(), LinearSolveError> {
        let solver = self
            .solver
            .as_ref()
            .ok_or(LinearSolveError::InvalidPolicy("solver not factorized"))?;

        solver.solve_in_place(rhs).map_err(LinearSolveError::from)
    }
}
