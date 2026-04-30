use crate::somelinalg::banded::{Banded, BlockTridiagonal};
use faer::sparse::{SparseColMat, Triplet};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub fn generate_block_tridiagonal_dense(
    n_blocks: usize,
    block_size: usize,
    seed: u64,
) -> BlockTridiagonal {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = BlockTridiagonal::zeros(n_blocks, block_size).unwrap();

    for blk in 0..n_blocks {
        // diagonal block
        for j in 0..block_size {
            let mut abs_sum = 0.0;

            for i in 0..block_size {
                if i == j {
                    continue;
                }
                let v = rng.gen_range(-0.2..0.2);
                a.set_diag(blk, i, j, v).unwrap();
                abs_sum += v.abs();
            }

            if blk > 0 {
                for i in 0..block_size {
                    let v = rng.gen_range(-0.05..0.05);
                    a.set_lower(blk - 1, i, j, v).unwrap();
                    abs_sum += v.abs();
                }
            }

            if blk + 1 < n_blocks {
                for i in 0..block_size {
                    let v = rng.gen_range(-0.05..0.05);
                    a.set_upper(blk, i, j, v).unwrap();
                    abs_sum += v.abs();
                }
            }

            a.set_diag(blk, j, j, abs_sum + rng.gen_range(1.0..2.0))
                .unwrap();
        }
    }

    a
}

pub fn generate_block_tridiagonal_narrow(
    n_blocks: usize,
    block_size: usize,
    diag_half_bw: usize,
    coupling_half_bw: usize,
    seed: u64,
) -> BlockTridiagonal {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = BlockTridiagonal::zeros(n_blocks, block_size).unwrap();

    for blk in 0..n_blocks {
        for j in 0..block_size {
            let mut abs_sum = 0.0;

            let i0 = j.saturating_sub(diag_half_bw);
            let i1 = (j + diag_half_bw + 1).min(block_size);

            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = rng.gen_range(-0.2..0.2);
                a.set_diag(blk, i, j, v).unwrap();
                abs_sum += v.abs();
            }

            if blk > 0 {
                let i0 = j.saturating_sub(coupling_half_bw);
                let i1 = (j + coupling_half_bw + 1).min(block_size);

                for i in i0..i1 {
                    let v = rng.gen_range(-0.05..0.05);
                    a.set_lower(blk - 1, i, j, v).unwrap();
                    abs_sum += v.abs();
                }
            }

            if blk + 1 < n_blocks {
                let i0 = j.saturating_sub(coupling_half_bw);
                let i1 = (j + coupling_half_bw + 1).min(block_size);

                for i in i0..i1 {
                    let v = rng.gen_range(-0.05..0.05);
                    a.set_upper(blk, i, j, v).unwrap();
                    abs_sum += v.abs();
                }
            }

            a.set_diag(blk, j, j, abs_sum + rng.gen_range(1.0..2.0))
                .unwrap();
        }
    }

    a
}

pub fn block_tridiagonal_matvec(a: &BlockTridiagonal, x: &[f64]) -> Vec<f64> {
    let nb = a.n_blocks();
    let bs = a.block_size();
    let n = a.n();
    assert_eq!(x.len(), n);

    let mut y = vec![0.0; n];

    for blk in 0..nb {
        let r0 = blk * bs;

        // diagonal block
        let d = a.diag_block(blk).unwrap();
        for i in 0..bs {
            for j in 0..bs {
                y[r0 + i] += d[i * bs + j] * x[r0 + j];
            }
        }

        // lower block
        if blk > 0 {
            let l = a.lower_block(blk - 1).unwrap();
            let c0 = (blk - 1) * bs;
            for i in 0..bs {
                for j in 0..bs {
                    y[r0 + i] += l[i * bs + j] * x[c0 + j];
                }
            }
        }

        // upper block
        if blk + 1 < nb {
            let u = a.upper_block(blk).unwrap();
            let c0 = (blk + 1) * bs;
            for i in 0..bs {
                for j in 0..bs {
                    y[r0 + i] += u[i * bs + j] * x[c0 + j];
                }
            }
        }
    }

    y
}

pub fn generate_rhs_from_known_solution_block(
    a: &BlockTridiagonal,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x_true: Vec<f64> = (0..a.n()).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b = block_tridiagonal_matvec(a, &x_true);
    (x_true, b)
}

pub fn block_to_banded(a: &BlockTridiagonal) -> Banded<f64> {
    let nb = a.n_blocks();
    let bs = a.block_size();
    let n = a.n();

    let kl = 2 * bs - 1;
    let ku = 2 * bs - 1;

    let mut out = Banded::<f64>::zeros(n, kl, ku).unwrap();

    for blk in 0..nb {
        let r0 = blk * bs;
        let c0 = blk * bs;

        let d = a.diag_block(blk).unwrap();
        for i in 0..bs {
            for j in 0..bs {
                out[(r0 + i, c0 + j)] = d[i * bs + j];
            }
        }

        if blk > 0 {
            let l = a.lower_block(blk - 1).unwrap();
            let lr0 = blk * bs;
            let lc0 = (blk - 1) * bs;
            for i in 0..bs {
                for j in 0..bs {
                    out[(lr0 + i, lc0 + j)] = l[i * bs + j];
                }
            }
        }

        if blk + 1 < nb {
            let u = a.upper_block(blk).unwrap();
            let ur0 = blk * bs;
            let uc0 = (blk + 1) * bs;
            for i in 0..bs {
                for j in 0..bs {
                    out[(ur0 + i, uc0 + j)] = u[i * bs + j];
                }
            }
        }
    }

    out
}

pub fn block_to_triplets(a: &BlockTridiagonal) -> Vec<Triplet<usize, usize, f64>> {
    let nb = a.n_blocks();
    let bs = a.block_size();

    let mut triplets = Vec::new();

    for blk in 0..nb {
        let r0 = blk * bs;
        let c0 = blk * bs;

        // diagonal block
        let d = a.diag_block(blk).unwrap();
        for i in 0..bs {
            for j in 0..bs {
                let v = d[i * bs + j];
                if v != 0.0 {
                    triplets.push(Triplet::new(r0 + i, c0 + j, v));
                }
            }
        }

        // lower block
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

        // upper block
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
