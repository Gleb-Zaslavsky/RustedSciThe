#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_bordered_solver::{
        BvpSciBorderedBandedBlocks, extract_bordered_banded_blocks,
        factor_bordered_banded_structured, solve_bordered_banded_reference,
        solve_bordered_banded_structured,
    };
    use crate::numerical::BVP_sci::BVP_sci_faer::{
        construct_global_jac, faer_col, faer_dense_mat, faer_mat,
    };
    use faer::linalg::solvers::Solve;
    use faer::sparse::{SparseColMat, Triplet};
    use std::time::Instant;

    fn dense_sparse(rows: usize, cols: usize, values: &[(usize, usize, f64)]) -> faer_mat {
        let triplets = values
            .iter()
            .map(|&(row, col, value)| Triplet::new(row, col, value))
            .collect::<Vec<_>>();
        SparseColMat::try_new_from_triplets(rows, cols, &triplets).unwrap()
    }

    fn scalar_sparse(value: f64) -> faer_mat {
        dense_sparse(1, 1, &[(0, 0, value)])
    }

    #[test]
    fn bordered_banded_extraction_reconstructs_parameter_free_global_jacobian() {
        let n = 2;
        let m = 3;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.5);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 2.0), (1, 1, -1.0)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.25), (1, 0, -0.5)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 2, &[(1, 0, 1.0)]);

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            None,
            None,
            &dbc_dya,
            &dbc_dyb,
            None,
        );
        let blocks = extract_bordered_banded_blocks(&jac, n, m, k).unwrap();

        assert_eq!(blocks.diagonal_blocks.len(), m - 1);
        assert_eq!(blocks.offdiag_blocks.len(), m - 1);
        assert_eq!(blocks.boundary_left.nrows(), n);
        assert_eq!(blocks.boundary_left.ncols(), n);
        assert_eq!(blocks.boundary_right.nrows(), n);
        assert_eq!(blocks.boundary_right.ncols(), n);
        assert!(blocks.collocation_parameter_blocks.is_none());
        assert!(blocks.boundary_parameters.is_none());
        assert!(blocks.max_abs_diff_against_sparse(&jac).unwrap() < 1e-14);
    }

    #[test]
    fn bordered_banded_extraction_preserves_parameter_blocks() {
        let n = 1;
        let m = 4;
        let k = 1;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![scalar_sparse(0.5); m];
        let df_dy_middle = vec![scalar_sparse(0.75); m - 1];
        let df_dp = Some(vec![scalar_sparse(2.0); m]);
        let df_dp_middle = Some(vec![scalar_sparse(3.0); m - 1]);
        let dbc_dya = dense_sparse(2, 1, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 1, &[(1, 0, -1.0)]);
        let dbc_dp = Some(dense_sparse(2, 1, &[(1, 0, 4.0)]));

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            df_dp.as_ref().map(|v| v.as_slice()),
            df_dp_middle.as_ref().map(|v| v.as_slice()),
            &dbc_dya,
            &dbc_dyb,
            dbc_dp.as_ref(),
        );
        let blocks = extract_bordered_banded_blocks(&jac, n, m, k).unwrap();

        let collocation_param_blocks = blocks.collocation_parameter_blocks.as_ref().unwrap();
        assert_eq!(collocation_param_blocks.len(), m - 1);
        assert_eq!(collocation_param_blocks[0].nrows(), n);
        assert_eq!(collocation_param_blocks[0].ncols(), k);
        let boundary_parameters = blocks.boundary_parameters.as_ref().unwrap();
        assert_eq!(boundary_parameters.nrows(), n + k);
        assert_eq!(boundary_parameters.ncols(), k);
        assert!(blocks.max_abs_diff_against_sparse(&jac).unwrap() < 1e-14);
    }

    #[test]
    fn bordered_banded_reference_solve_matches_sparse_lu_parameter_free() {
        let n = 2;
        let m = 3;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.5);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, -0.25)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.1), (1, 0, -0.2)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, 0.5)]);
        let dbc_dyb = dense_sparse(2, 2, &[(0, 0, -0.3), (1, 1, 1.0)]);

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            None,
            None,
            &dbc_dya,
            &dbc_dyb,
            None,
        );

        assert_reference_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_reference_solve_matches_sparse_lu_with_parameter() {
        let n = 1;
        let m = 4;
        let k = 1;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![scalar_sparse(0.5); m];
        let df_dy_middle = vec![scalar_sparse(0.75); m - 1];
        let df_dp = Some(vec![scalar_sparse(2.0); m]);
        let df_dp_middle = Some(vec![scalar_sparse(3.0); m - 1]);
        let dbc_dya = dense_sparse(2, 1, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 1, &[(0, 0, 0.5), (1, 0, -1.0)]);
        let dbc_dp = Some(dense_sparse(2, 1, &[(1, 0, 4.0)]));

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            df_dp.as_ref().map(|v| v.as_slice()),
            df_dp_middle.as_ref().map(|v| v.as_slice()),
            &dbc_dya,
            &dbc_dyb,
            dbc_dp.as_ref(),
        );

        assert_reference_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_structured_solve_matches_sparse_lu_parameter_free() {
        let n = 2;
        let m = 4;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 0.8), (1, 1, -0.4)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.15), (1, 0, -0.05)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, 0.25)]);
        let dbc_dyb = dense_sparse(2, 2, &[(0, 0, 0.2), (1, 1, -1.0)]);

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            None,
            None,
            &dbc_dya,
            &dbc_dyb,
            None,
        );

        assert_structured_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_structured_solve_matches_sparse_lu_with_parameter() {
        let n = 1;
        let m = 5;
        let k = 1;
        let h = faer_col::from_fn(m - 1, |_| 0.2);
        let df_dy = vec![scalar_sparse(0.4); m];
        let df_dy_middle = vec![scalar_sparse(0.6); m - 1];
        let df_dp = Some(vec![scalar_sparse(1.5); m]);
        let df_dp_middle = Some(vec![scalar_sparse(2.5); m - 1]);
        let dbc_dya = dense_sparse(2, 1, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 1, &[(0, 0, 0.2), (1, 0, -1.0)]);
        let dbc_dp = Some(dense_sparse(2, 1, &[(1, 0, 3.0)]));

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            df_dp.as_ref().map(|v| v.as_slice()),
            df_dp_middle.as_ref().map(|v| v.as_slice()),
            &dbc_dya,
            &dbc_dyb,
            dbc_dp.as_ref(),
        );

        assert_structured_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_structured_factorization_reuses_multiple_rhs() {
        let n = 2;
        let m = 4;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 0.8), (1, 1, -0.4)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.15), (1, 0, -0.05)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, 0.25)]);
        let dbc_dyb = dense_sparse(2, 2, &[(0, 0, 0.2), (1, 1, -1.0)]);

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            None,
            None,
            &dbc_dya,
            &dbc_dyb,
            None,
        );
        let blocks = extract_bordered_banded_blocks(&jac, n, m, k).unwrap();
        let factorization = factor_bordered_banded_structured(&blocks).unwrap();
        let sparse_lu = jac.sp_lu().unwrap();

        for rhs_shift in [0.0, 0.25, -0.5] {
            let rhs = faer_col::from_fn(blocks.total_size(), |row| {
                ((row + 5) as f64 + rhs_shift).cos() - 0.05 * (row as f64)
            });
            let cached = factorization.solve(&rhs).unwrap();
            let one_shot = solve_bordered_banded_structured(&blocks, &rhs).unwrap();
            let sparse_solution = sparse_lu.solve(rhs.as_mat());
            let sparse =
                faer_col::from_fn(sparse_solution.nrows(), |row| *sparse_solution.get(row, 0));

            let max_cached_vs_one_shot = (0..cached.nrows())
                .map(|row| (cached[row] - one_shot[row]).abs())
                .fold(0.0_f64, f64::max);
            let max_cached_vs_sparse = (0..cached.nrows())
                .map(|row| (cached[row] - sparse[row]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_cached_vs_one_shot < 1e-12,
                "cached bordered solve must match one-shot structured solve, max_diff={max_cached_vs_one_shot:e}"
            );
            assert!(
                max_cached_vs_sparse < 1e-10,
                "cached bordered solve must match sparse LU, max_diff={max_cached_vs_sparse:e}"
            );
        }
    }

    #[test]
    #[ignore]
    fn bordered_banded_factor_solve_microbench_vs_sparse_lu() {
        let mesh_points = std::env::var("BVP_SCI_BORDERED_MICRO_M")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1500);
        let runs = std::env::var("BVP_SCI_BORDERED_MICRO_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5);
        let rhs_count = std::env::var("BVP_SCI_BORDERED_MICRO_RHS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(12);

        let scenarios = [
            (
                "left-fixed",
                synthetic_parameter_free_blocks(4, mesh_points, 0),
            ),
            (
                "split-endpoint",
                synthetic_parameter_free_blocks(6, mesh_points, 1),
            ),
            (
                "mixed-endpoint",
                synthetic_parameter_free_blocks(8, mesh_points, 2),
            ),
        ];

        println!(
            "\n[BVP_sci bordered microbench] m={mesh_points}, runs={runs}, rhs_count={rhs_count}"
        );
        println!(
            "scenario       | n | matrix | factor_ms mean+/-std | solve_all_ms mean+/-std | solve_rhs_ms | total_1rhs_ms | total_batch_ms | max_diff_vs_sparse | verdict_1rhs | verdict_batch"
        );
        println!("{}", "-".repeat(188));

        for (scenario, blocks) in scenarios {
            let sparse = sparse_from_bordered_blocks(&blocks);
            let mut bordered_factor_ms = Vec::with_capacity(runs);
            let mut bordered_solve_ms = Vec::with_capacity(runs);
            let mut sparse_factor_ms = Vec::with_capacity(runs);
            let mut sparse_solve_ms = Vec::with_capacity(runs);
            let mut max_diff = 0.0_f64;

            let rhs_values = (0..rhs_count)
                .map(|rhs_idx| {
                    faer_col::from_fn(blocks.total_size(), |row| {
                        ((row + 11 * rhs_idx + 3) as f64 * 0.013).sin() + 0.0001 * (row % 97) as f64
                    })
                })
                .collect::<Vec<_>>();

            for _ in 0..runs {
                let started = Instant::now();
                let bordered_factor = factor_bordered_banded_structured(&blocks).unwrap();
                bordered_factor_ms.push(started.elapsed().as_secs_f64() * 1000.0);

                let started = Instant::now();
                let bordered_solutions = rhs_values
                    .iter()
                    .map(|rhs| bordered_factor.solve(rhs).unwrap())
                    .collect::<Vec<_>>();
                bordered_solve_ms.push(started.elapsed().as_secs_f64() * 1000.0);

                let started = Instant::now();
                let sparse_lu = sparse.sp_lu().unwrap();
                sparse_factor_ms.push(started.elapsed().as_secs_f64() * 1000.0);

                let started = Instant::now();
                for (rhs, bordered_solution) in rhs_values.iter().zip(bordered_solutions.iter()) {
                    let sparse_solution = sparse_lu.solve(rhs.as_mat());
                    for row in 0..bordered_solution.nrows() {
                        max_diff = max_diff
                            .max((bordered_solution[row] - *sparse_solution.get(row, 0)).abs());
                    }
                }
                sparse_solve_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            }

            let bordered_factor = summarize(&bordered_factor_ms);
            let bordered_solve = summarize(&bordered_solve_ms);
            let sparse_factor = summarize(&sparse_factor_ms);
            let sparse_solve = summarize(&sparse_solve_ms);
            let bordered_solve_per_rhs = bordered_solve.mean / rhs_count as f64;
            let sparse_solve_per_rhs = sparse_solve.mean / rhs_count as f64;
            let bordered_total_one_rhs = bordered_factor.mean + bordered_solve_per_rhs;
            let sparse_total_one_rhs = sparse_factor.mean + sparse_solve_per_rhs;
            let bordered_total_batch = bordered_factor.mean + bordered_solve.mean;
            let sparse_total_batch = sparse_factor.mean + sparse_solve.mean;
            let verdict_one_rhs = if bordered_total_one_rhs < sparse_total_one_rhs {
                "bordered"
            } else {
                "sparse"
            };
            let verdict_batch = if bordered_total_batch < sparse_total_batch {
                "bordered"
            } else {
                "sparse"
            };

            println!(
                "{scenario:<14} | {:>1} | Bordered | {:>8.3} +/- {:<8.3} | {:>10.3} +/- {:<8.3} | {:>12.3} | {:>13.3} | {:>14.3} | {:>18.3e} | {:>12} | {:>13}",
                blocks.variable_count,
                bordered_factor.mean,
                bordered_factor.std,
                bordered_solve.mean,
                bordered_solve.std,
                bordered_solve_per_rhs,
                bordered_total_one_rhs,
                bordered_total_batch,
                max_diff,
                verdict_one_rhs,
                verdict_batch,
            );
            println!(
                "{scenario:<14} | {:>1} | Sparse   | {:>8.3} +/- {:<8.3} | {:>10.3} +/- {:<8.3} | {:>12.3} | {:>13.3} | {:>14.3} | {:>18} | {:>12} | {:>13}",
                blocks.variable_count,
                sparse_factor.mean,
                sparse_factor.std,
                sparse_solve.mean,
                sparse_solve.std,
                sparse_solve_per_rhs,
                sparse_total_one_rhs,
                sparse_total_batch,
                "-",
                "baseline",
                "baseline",
            );
            println!(
                "{scenario:<14} | {:>1} | delta    | {:>8.3}            | {:>10.3}            | {:>12.3} | {:>13.3} | {:>14.3} | {:>18} | {:>12} | {:>13}",
                blocks.variable_count,
                bordered_factor.mean - sparse_factor.mean,
                bordered_solve.mean - sparse_solve.mean,
                bordered_solve_per_rhs - sparse_solve_per_rhs,
                bordered_total_one_rhs - sparse_total_one_rhs,
                bordered_total_batch - sparse_total_batch,
                "-",
                if bordered_total_one_rhs < sparse_total_one_rhs {
                    "win"
                } else {
                    "loss"
                },
                if bordered_total_batch < sparse_total_batch {
                    "win"
                } else {
                    "loss"
                },
            );

            if bordered_total_batch < sparse_total_batch {
                println!(
                    "[BVP_sci bordered microbench] {scenario}: bordered wins for the full RHS batch"
                );
            } else {
                println!(
                    "[BVP_sci bordered microbench] {scenario}: Sparse wins for the full RHS batch; check one_rhs verdict for Newton-like usage"
                );
            }

            assert!(
                max_diff < 1e-8,
                "{scenario}: bordered solve must match Sparse LU, max_diff={max_diff:e}"
            );
        }
    }

    #[test]
    fn bordered_banded_structured_solve_rejects_wrong_rhs_length() {
        let blocks = simple_valid_blocks_without_parameters();
        let rhs = faer_col::zeros(blocks.total_size() - 1);
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("expected rhs length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bordered_banded_structured_solve_rejects_malformed_block_layout() {
        let mut blocks = simple_valid_blocks_without_parameters();
        blocks.offdiag_blocks.pop();
        let rhs = faer_col::zeros(blocks.total_size());
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("expected 2 offdiag blocks"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bordered_banded_structured_solve_reports_singular_offdiag_block() {
        let mut blocks = simple_valid_blocks_without_parameters();
        blocks.offdiag_blocks[0] = faer_dense_mat::zeros(1, 1);
        let rhs = faer_col::from_fn(blocks.total_size(), |row| row as f64 + 1.0);
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("singular offdiag block at interval 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bordered_banded_structured_solve_reports_singular_border_system() {
        let mut blocks = simple_valid_blocks_without_parameters();
        blocks.boundary_left = faer_dense_mat::zeros(1, 1);
        blocks.boundary_right = faer_dense_mat::zeros(1, 1);
        let rhs = faer_col::from_fn(blocks.total_size(), |row| row as f64 + 1.0);
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("singular border system"),
            "unexpected error: {err}"
        );
    }

    fn assert_reference_solve_matches_sparse(jac: &faer_mat, n: usize, m: usize, k: usize) {
        let blocks = extract_bordered_banded_blocks(jac, n, m, k).unwrap();
        let rhs = faer_col::from_fn(blocks.total_size(), |row| {
            ((row + 3) as f64).sin() + 0.1 * (row as f64)
        });

        let reference = solve_bordered_banded_reference(&blocks, &rhs).unwrap();
        let sparse_solution = jac.sp_lu().unwrap().solve(rhs.as_mat());
        let sparse = faer_col::from_fn(sparse_solution.nrows(), |row| *sparse_solution.get(row, 0));

        let max_diff = (0..reference.nrows())
            .map(|row| (reference[row] - sparse[row]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "bordered-banded reference solve must match sparse LU, max_diff={max_diff:e}"
        );
    }

    fn assert_structured_solve_matches_sparse(jac: &faer_mat, n: usize, m: usize, k: usize) {
        let blocks = extract_bordered_banded_blocks(jac, n, m, k).unwrap();
        let rhs = faer_col::from_fn(blocks.total_size(), |row| {
            ((row + 5) as f64).cos() - 0.05 * (row as f64)
        });

        let structured = solve_bordered_banded_structured(&blocks, &rhs).unwrap();
        let sparse_solution = jac.sp_lu().unwrap().solve(rhs.as_mat());
        let sparse = faer_col::from_fn(sparse_solution.nrows(), |row| *sparse_solution.get(row, 0));

        let max_diff = (0..structured.nrows())
            .map(|row| (structured[row] - sparse[row]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "bordered-banded structured solve must match sparse LU, max_diff={max_diff:e}"
        );
    }

    fn simple_valid_blocks_without_parameters() -> BvpSciBorderedBandedBlocks {
        BvpSciBorderedBandedBlocks {
            variable_count: 1,
            mesh_points: 3,
            parameter_count: 0,
            diagonal_blocks: vec![faer_dense_mat::from_fn(1, 1, |_, _| -1.0); 2],
            offdiag_blocks: vec![faer_dense_mat::from_fn(1, 1, |_, _| 1.0); 2],
            collocation_parameter_blocks: None,
            boundary_left: faer_dense_mat::from_fn(1, 1, |_, _| 1.0),
            boundary_right: faer_dense_mat::from_fn(1, 1, |_, _| 0.5),
            boundary_parameters: None,
        }
    }

    #[derive(Clone, Copy)]
    struct Summary {
        mean: f64,
        std: f64,
    }

    fn summarize(values: &[f64]) -> Summary {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;
        Summary {
            mean,
            std: variance.sqrt(),
        }
    }

    fn synthetic_parameter_free_blocks(
        variable_count: usize,
        mesh_points: usize,
        boundary_mode: usize,
    ) -> BvpSciBorderedBandedBlocks {
        let mut diagonal_blocks = Vec::with_capacity(mesh_points - 1);
        let mut offdiag_blocks = Vec::with_capacity(mesh_points - 1);
        for interval in 0..mesh_points - 1 {
            diagonal_blocks.push(faer_dense_mat::from_fn(
                variable_count,
                variable_count,
                |row, col| {
                    if row == col {
                        -1.0 - 0.000001 * interval as f64 - 0.001 * row as f64
                    } else if row.abs_diff(col) == 1 {
                        0.004
                    } else {
                        0.0005 / (1 + row.abs_diff(col)) as f64
                    }
                },
            ));
            offdiag_blocks.push(faer_dense_mat::from_fn(
                variable_count,
                variable_count,
                |row, col| {
                    if row == col {
                        1.0 + 0.000001 * interval as f64 + 0.001 * row as f64
                    } else if row.abs_diff(col) == 1 {
                        -0.004
                    } else {
                        0.0
                    }
                },
            ));
        }

        let boundary_left =
            faer_dense_mat::from_fn(
                variable_count,
                variable_count,
                |row, col| match boundary_mode {
                    0 => {
                        if row == col {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    1 => {
                        if row < variable_count / 2 && row == col {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        if row == col {
                            0.6 + 0.02 * row as f64
                        } else if row.abs_diff(col) == 1 {
                            0.03
                        } else {
                            0.0
                        }
                    }
                },
            );
        let boundary_right =
            faer_dense_mat::from_fn(
                variable_count,
                variable_count,
                |row, col| match boundary_mode {
                    0 => 0.0,
                    1 => {
                        if row >= variable_count / 2 && row == col {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        if row == col {
                            0.35 + 0.01 * row as f64
                        } else if row.abs_diff(col) == 1 {
                            -0.02
                        } else {
                            0.0
                        }
                    }
                },
            );

        BvpSciBorderedBandedBlocks {
            variable_count,
            mesh_points,
            parameter_count: 0,
            diagonal_blocks,
            offdiag_blocks,
            collocation_parameter_blocks: None,
            boundary_left,
            boundary_right,
            boundary_parameters: None,
        }
    }

    fn sparse_from_bordered_blocks(blocks: &BvpSciBorderedBandedBlocks) -> faer_mat {
        let n = blocks.variable_count;
        let m = blocks.mesh_points;
        let mut triplets = Vec::new();
        for interval in 0..m.saturating_sub(1) {
            let row_start = interval * n;
            let diag_col_start = interval * n;
            let offdiag_col_start = (interval + 1) * n;
            push_dense_block_triplets(
                &mut triplets,
                row_start,
                diag_col_start,
                &blocks.diagonal_blocks[interval],
            );
            push_dense_block_triplets(
                &mut triplets,
                row_start,
                offdiag_col_start,
                &blocks.offdiag_blocks[interval],
            );
        }
        let bc_row_start = blocks.collocation_rows();
        push_dense_block_triplets(&mut triplets, bc_row_start, 0, &blocks.boundary_left);
        push_dense_block_triplets(
            &mut triplets,
            bc_row_start,
            n * m.saturating_sub(1),
            &blocks.boundary_right,
        );
        SparseColMat::try_new_from_triplets(blocks.total_size(), blocks.total_size(), &triplets)
            .unwrap()
    }

    fn push_dense_block_triplets(
        triplets: &mut Vec<Triplet<usize, usize, f64>>,
        row_start: usize,
        col_start: usize,
        block: &faer_dense_mat,
    ) {
        for row in 0..block.nrows() {
            for col in 0..block.ncols() {
                let value = *block.get(row, col);
                if value != 0.0 {
                    triplets.push(Triplet::new(row_start + row, col_start + col, value));
                }
            }
        }
    }
}
