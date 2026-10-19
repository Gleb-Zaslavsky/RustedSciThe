use super::{
    block_tridiagonal::BlockTridiagonal,
    block_tridiagonal_lu::BlockTridiagonalLu,
    block_tridiagonal_lu_consistent::BlockTridiagonalLuConsistent,
    linear_solver::LinearSolver,
    solver_policy::{FallbackPolicy, LinearSolveError, LinearSolverConfig, LinearSolverPolicy},
};

use super::solver_traits::FaerSparseLuSolver;
use faer::sparse::Triplet;

fn block_to_triplets(a: &BlockTridiagonal) -> Vec<Triplet<usize, usize, f64>> {
    let nb = a.n_blocks();
    let bs = a.block_size();

    let mut triplets = Vec::new();

    for blk in 0..nb {
        let r0 = blk * bs;
        let c0 = blk * bs;

        let d = a.diag_block(blk).unwrap();
        for i in 0..bs {
            for j in 0..bs {
                let v = d[i * bs + j];
                if v != 0.0 {
                    triplets.push(Triplet::new(r0 + i, c0 + j, v));
                }
            }
        }

        if blk > 0 {
            let l = a.lower_block(blk - 1).unwrap();
            let lr0 = blk * bs;
            let lc0 = (blk - 1) * bs;
            for i in 0..bs {
                for j in 0..bs {
                    let v = l[i * bs + j];
                    if v != 0.0 {
                        triplets.push(Triplet::new(lr0 + i, lc0 + j, v));
                    }
                }
            }
        }

        if blk + 1 < nb {
            let u = a.upper_block(blk).unwrap();
            let ur0 = blk * bs;
            let uc0 = (blk + 1) * bs;
            for i in 0..bs {
                for j in 0..bs {
                    let v = u[i * bs + j];
                    if v != 0.0 {
                        triplets.push(Triplet::new(ur0 + i, uc0 + j, v));
                    }
                }
            }
        }
    }

    triplets
}

pub fn factor_block_tridiagonal(
    a: &BlockTridiagonal,
    config: LinearSolverConfig,
) -> Result<LinearSolver, LinearSolveError> {
    match config.policy {
        LinearSolverPolicy::ForceBlockTridiagonal => {
            let mut lu = BlockTridiagonalLu::new(a.n_blocks(), a.block_size())?;
            lu.factor_from(a)?;
            Ok(LinearSolver::BlockTridiagonal(lu))
        }
        LinearSolverPolicy::ForceBlockTridiagonalConsistent => {
            let mut lu = BlockTridiagonalLuConsistent::new(a.n_blocks(), a.block_size())?;
            lu.factor_from(a)?;
            Ok(LinearSolver::BlockTridiagonalConsistent {
                solver: lu,
                matrix: a.clone(),
                iterative_refinement_steps: config.iterative_refinement_steps,
            })
        }

        LinearSolverPolicy::ForceFaerSparse => {
            {
                let triplets = block_to_triplets(a);
                let solver = FaerSparseLuSolver::from_triplets(a.n(), &triplets)?;
                return Ok(LinearSolver::FaerSparse(solver));
            }

            {
                Err(LinearSolveError::InvalidPolicy(
                    "ForceFaerSparse requested but faer-sparse feature is disabled",
                ))
            }
        }

        LinearSolverPolicy::ForceBanded | LinearSolverPolicy::ForceGeneralBandedPartialPivot => {
            Err(LinearSolveError::InvalidPolicy(
                "general banded solvers require BandedAssembly input; use build_solver_for_system",
            ))
        }

        LinearSolverPolicy::Auto => {
            let try_native = || -> Result<LinearSolver, LinearSolveError> {
                let mut lu = BlockTridiagonalLuConsistent::new(a.n_blocks(), a.block_size())?;
                lu.factor_from(a)?;
                Ok(LinearSolver::BlockTridiagonalConsistent {
                    solver: lu,
                    matrix: a.clone(),
                    iterative_refinement_steps: config.iterative_refinement_steps,
                })
            };

            match try_native() {
                Ok(solver) => Ok(solver),
                Err(native_err) => match config.fallback {
                    FallbackPolicy::Never => Err(native_err),
                    FallbackPolicy::ToFaerSparse => {
                        {
                            let triplets = block_to_triplets(a);
                            let solver = FaerSparseLuSolver::from_triplets(a.n(), &triplets)?;
                            return Ok(LinearSolver::FaerSparse(solver));
                        }

                        { Err(native_err) }
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::factor_block_tridiagonal;
    use crate::somelinalg::banded::linear_solver::newton_solve_step;
    use crate::somelinalg::banded::{
        banded_assembly::BandedAssembly,
        block_tridiagonal::BlockTridiagonal,
        linear_solver::{LinearSolver, LinearSystemRef, build_solver_for_system},
        ops::banded_to_dense,
        solver_policy::{FallbackPolicy, LinearSolveError, LinearSolverConfig, LinearSolverPolicy},
        solver_traits::DirectLinearSolver,
    };
    fn small_block_system() -> BlockTridiagonal {
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        // B0
        a.set_diag(0, 0, 0, 4.0).unwrap();
        a.set_diag(0, 0, 1, 1.0).unwrap();
        a.set_diag(0, 1, 0, 2.0).unwrap();
        a.set_diag(0, 1, 1, 3.0).unwrap();

        // C0
        a.set_upper(0, 0, 0, 0.2).unwrap();
        a.set_upper(0, 1, 1, 0.2).unwrap();

        // A1
        a.set_lower(0, 0, 0, 0.1).unwrap();
        a.set_lower(0, 1, 1, 0.1).unwrap();

        // B1
        a.set_diag(1, 0, 0, 5.0).unwrap();
        a.set_diag(1, 0, 1, 0.5).unwrap();
        a.set_diag(1, 1, 0, 0.5).unwrap();
        a.set_diag(1, 1, 1, 4.0).unwrap();

        a
    }

    fn dense_from_block(a: &BlockTridiagonal) -> Vec<Vec<f64>> {
        a.to_dense()
    }

    fn dense_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i][j] * x[j];
            }
        }
        y
    }

    fn vec_diff_linf(x: &[f64], y: &[f64]) -> f64 {
        let mut m = 0.0;
        for i in 0..x.len() {
            let d = (x[i] - y[i]).abs();
            if d > m {
                m = d;
            }
        }
        m
    }

    #[test]
    fn force_block_tridiagonal_returns_block_backend() {
        let a = small_block_system();

        let solver = factor_block_tridiagonal(
            &a,
            LinearSolverConfig {
                policy: LinearSolverPolicy::ForceBlockTridiagonal,
                fallback: FallbackPolicy::Never,
                iterative_refinement_steps: 0,
            },
        )
        .unwrap();

        match solver {
            LinearSolver::BlockTridiagonal(_) => {}
            _ => panic!("expected BlockTridiagonal backend"),
        }
    }

    #[test]
    fn auto_returns_native_block_backend_on_good_case() {
        let a = small_block_system();

        let solver = factor_block_tridiagonal(&a, LinearSolverConfig::default()).unwrap();

        match solver {
            LinearSolver::BlockTridiagonalConsistent { .. } => {}
            _ => panic!("expected BlockTridiagonal backend in Auto mode"),
        }
    }

    #[test]
    fn force_block_tridiagonal_solves_system() {
        let a = small_block_system();
        let dense = dense_from_block(&a);

        let x_true = vec![1.0, -1.0, 2.0, 0.5];
        let mut rhs = dense_matvec(&dense, &x_true);

        let solver = factor_block_tridiagonal(
            &a,
            LinearSolverConfig {
                policy: LinearSolverPolicy::ForceBlockTridiagonal,
                fallback: FallbackPolicy::Never,
                iterative_refinement_steps: 0,
            },
        )
        .unwrap();

        solver.solve_in_place(&mut rhs).unwrap();

        assert!(vec_diff_linf(&rhs, &x_true) < 1e-10);
    }

    #[test]
    fn force_banded_for_block_input_is_rejected_for_now() {
        let a = small_block_system();

        let err = factor_block_tridiagonal(
            &a,
            LinearSolverConfig {
                policy: LinearSolverPolicy::ForceBanded,
                fallback: FallbackPolicy::Never,
                iterative_refinement_steps: 0,
            },
        )
        .unwrap_err();

        assert!(matches!(err, LinearSolveError::InvalidPolicy(_)));
    }
    #[test]
    fn newton_helper_solves_small_system() {
        let a = small_block_system();
        let dense = a.to_dense();

        let x_true = vec![1.0, -1.0, 2.0, 0.5];
        let minus_x_true: Vec<f64> = x_true.iter().map(|v| -v).collect();

        let mut residual = dense_matvec(&dense, &x_true);

        let config = LinearSolverConfig::default();
        newton_solve_step(&a, &mut residual, config).unwrap();

        assert!(vec_diff_linf(&residual, &minus_x_true) < 1e-10);
    }

    #[test]
    fn force_banded_assembly_returns_lapack_style_backend_with_refinement() {
        let n = 6;
        let mut a = BandedAssembly::zeros(n, 2, 2).unwrap();
        for j in 0..n {
            let i0 = j.saturating_sub(2);
            let i1 = (j + 3).min(n);
            let mut col_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = 0.05 * ((2 * i + 5 * j + 3) as f64).cos();
                a.set(i, j, v).unwrap();
                col_sum += v.abs();
            }
            a.set(j, j, col_sum + 1.0).unwrap();
        }
        a.set(0, 0, 1e-10).unwrap();
        a.set(1, 0, 1.0).unwrap();

        let compact = a.to_banded().unwrap();
        let dense = banded_to_dense(&compact);
        let x_true: Vec<f64> = (0..n).map(|i| 0.25 * (i as f64 + 1.0).sin()).collect();
        let mut rhs = dense_matvec(&dense, &x_true);

        let solver = build_solver_for_system(
            LinearSystemRef::BandedAssembly(&a),
            LinearSolverConfig {
                policy: LinearSolverPolicy::ForceBanded,
                fallback: FallbackPolicy::Never,
                iterative_refinement_steps: 1,
            },
        )
        .unwrap();

        assert_eq!(solver.backend_name(), "lapack_style_banded_lu+refine");
        solver.solve_in_place(&mut rhs).unwrap();

        assert!(vec_diff_linf(&rhs, &x_true) < 1e-9);
    }
}
