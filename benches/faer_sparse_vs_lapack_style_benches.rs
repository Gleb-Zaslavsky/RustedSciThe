use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;

use RustedSciThe::somelinalg::banded::{
    Banded, banded_matvec, lapack_style_banded::LapackStyleBandedLuFaithful,
};

fn generate_rhs_from_known_solution(a: &Banded<f64>, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x_true: Vec<f64> = (0..a.n()).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b = banded_matvec(a, &x_true).unwrap();
    (x_true, b)
}

/// Dense block-tridiagonal benchmark matrix.
/// Global scalar half-bandwidth = 2*block_size - 1.
fn generate_block_tridiagonal_banded(n_blocks: usize, block_size: usize, seed: u64) -> Banded<f64> {
    assert!(n_blocks > 0);
    assert!(block_size > 0);

    let n = n_blocks * block_size;
    let kl = 2 * block_size - 1;
    let ku = 2 * block_size - 1;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

    for blk in 0..n_blocks {
        let row0 = blk * block_size;
        let col0 = blk * block_size;

        for local_j in 0..block_size {
            let j = col0 + local_j;
            let mut abs_sum = 0.0;

            // diagonal block
            for local_i in 0..block_size {
                let i = row0 + local_i;
                if i == j {
                    continue;
                }

                let v = rng.random_range(-0.2..0.2);
                a[(i, j)] = v;
                abs_sum += v.abs();
            }

            // lower neighboring block
            if blk > 0 {
                let prev_row0 = (blk - 1) * block_size;
                for local_i in 0..block_size {
                    let i = prev_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            // upper neighboring block
            if blk + 1 < n_blocks {
                let next_row0 = (blk + 1) * block_size;
                for local_i in 0..block_size {
                    let i = next_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            a[(j, j)] = abs_sum + rng.random_range(1.0..2.0);
        }
    }

    a
}

/// Narrower block-tridiagonal matrix:
/// - inside-node coupling only within diag_half_bw
/// - neighbor coupling only within coupling_half_bw
fn generate_block_tridiagonal_narrow_coupling_banded(
    n_blocks: usize,
    block_size: usize,
    diag_half_bw: usize,
    coupling_half_bw: usize,
    seed: u64,
) -> Banded<f64> {
    assert!(n_blocks > 0);
    assert!(block_size > 0);

    let n = n_blocks * block_size;
    let kl = block_size + coupling_half_bw;
    let ku = block_size + coupling_half_bw;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

    for blk in 0..n_blocks {
        let row0 = blk * block_size;
        let col0 = blk * block_size;

        for local_j in 0..block_size {
            let j = col0 + local_j;
            let mut abs_sum = 0.0;

            // diagonal block, but narrow inside the node
            let li0 = local_j.saturating_sub(diag_half_bw);
            let li1 = (local_j + diag_half_bw + 1).min(block_size);

            for local_i in li0..li1 {
                let i = row0 + local_i;
                if i == j {
                    continue;
                }

                let v = rng.random_range(-0.2..0.2);
                a[(i, j)] = v;
                abs_sum += v.abs();
            }

            // lower neighboring block
            if blk > 0 {
                let prev_row0 = (blk - 1) * block_size;
                let li0 = local_j.saturating_sub(coupling_half_bw);
                let li1 = (local_j + coupling_half_bw + 1).min(block_size);

                for local_i in li0..li1 {
                    let i = prev_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            // upper neighboring block
            if blk + 1 < n_blocks {
                let next_row0 = (blk + 1) * block_size;
                let li0 = local_j.saturating_sub(coupling_half_bw);
                let li1 = (local_j + coupling_half_bw + 1).min(block_size);

                for local_i in li0..li1 {
                    let i = next_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            a[(j, j)] = abs_sum + rng.random_range(1.0..2.0);
        }
    }

    a
}

/// More BVP 1D-like matrix:
/// - node-major ordering
/// - `vars_per_node` unknowns per mesh point
/// - local bandwidth inside one node is narrow
/// - nearest-neighbor coupling is also narrow
fn generate_bvp1d_banded(
    n_nodes: usize,
    vars_per_node: usize,
    local_half_bw: usize,
    coupling_half_bw: usize,
    seed: u64,
) -> Banded<f64> {
    assert!(n_nodes > 0);
    assert!(vars_per_node > 0);

    let n = n_nodes * vars_per_node;
    let kl = vars_per_node + coupling_half_bw;
    let ku = vars_per_node + coupling_half_bw;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

    for node in 0..n_nodes {
        let row0 = node * vars_per_node;
        let col0 = node * vars_per_node;

        for local_j in 0..vars_per_node {
            let j = col0 + local_j;
            let mut abs_sum = 0.0;

            // local/node block
            let li0 = local_j.saturating_sub(local_half_bw);
            let li1 = (local_j + local_half_bw + 1).min(vars_per_node);

            for local_i in li0..li1 {
                let i = row0 + local_i;
                if i == j {
                    continue;
                }
                let v = rng.random_range(-0.2..0.2);
                a[(i, j)] = v;
                abs_sum += v.abs();
            }

            // left neighbor node
            if node > 0 {
                let prev_row0 = (node - 1) * vars_per_node;
                let li0 = local_j.saturating_sub(coupling_half_bw);
                let li1 = (local_j + coupling_half_bw + 1).min(vars_per_node);

                for local_i in li0..li1 {
                    let i = prev_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            // right neighbor node
            if node + 1 < n_nodes {
                let next_row0 = (node + 1) * vars_per_node;
                let li0 = local_j.saturating_sub(coupling_half_bw);
                let li1 = (local_j + coupling_half_bw + 1).min(vars_per_node);

                for local_i in li0..li1 {
                    let i = next_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            a[(j, j)] = abs_sum + rng.random_range(1.0..2.0);
        }
    }

    a
}

fn banded_to_triplets(a: &Banded<f64>) -> Vec<Triplet<usize, usize, f64>> {
    let n = a.n();
    let mut triplets = Vec::new();

    for j in 0..n {
        let i0 = j.saturating_sub(a.ku());
        let i1 = (j + a.kl() + 1).min(n);

        for i in i0..i1 {
            let v = a[(i, j)];
            if v != 0.0 {
                triplets.push(Triplet::new(i, j, v));
            }
        }
    }

    triplets
}

fn rhs_vec_to_faer_col(rhs: &[f64]) -> faer::Col<f64> {
    faer::Col::from_fn(rhs.len(), |i| rhs[i])
}

fn vec_linf_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn relative_banded_residual(a: &Banded<f64>, x: &[f64], b: &[f64]) -> f64 {
    let ax = banded_matvec(a, x).unwrap();
    let mut rmax = 0.0_f64;
    let mut bmax = 0.0_f64;

    for i in 0..b.len() {
        rmax = rmax.max((ax[i] - b[i]).abs());
        bmax = bmax.max(b[i].abs());
    }

    rmax / bmax.max(1.0)
}

#[allow(dead_code)]
fn compare_against_faer_once(a: &Banded<f64>, b: &[f64]) {
    let n = a.n();
    let triplets = banded_to_triplets(a);
    let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

    let mut lu = LapackStyleBandedLuFaithful::new(n, a.kl(), a.ku()).unwrap();
    lu.factor_from(a).unwrap();

    let mut x_band = b.to_vec();
    lu.solve_in_place(&mut x_band).unwrap();

    let x_faer = sparse.sp_lu().unwrap().solve(&rhs_vec_to_faer_col(b));
    let x_faer_vec: Vec<f64> = (0..n).map(|i| x_faer[i]).collect();

    let rr_band = relative_banded_residual(a, &x_band, b);
    let rr_faer = relative_banded_residual(a, &x_faer_vec, b);

    let diff = x_band
        .iter()
        .zip(x_faer_vec.iter())
        .map(|(u, v)| (u - v).abs())
        .fold(0.0_f64, f64::max);

    eprintln!("compare_once: rr_band={rr_band:e}, rr_faer={rr_faer:e}, x_diff_linf={diff:e}");
}

fn bench_dense_block_tridiag_factor_vs_faer(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_block_tridiag_factor_vs_faer");
    group.sample_size(10);

    let cases = [
        (50, 50),   // n=2500, half-bandwidth=99
        (100, 50),  // n=5000, half-bandwidth=99
        (100, 100), // n=10000, half-bandwidth=199
    ];

    for &(n_blocks, block_size) in &cases {
        let a = generate_block_tridiagonal_banded(n_blocks, block_size, 42);
        let n = n_blocks * block_size;
        let kl = a.kl();
        let ku = a.ku();

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        group.bench_with_input(
            BenchmarkId::new(
                "lapack_style_banded_lu",
                format!("n={n},block={block_size}"),
            ),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("faer_sparse_lu", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let lu = black_box(&sparse).sp_lu().unwrap();
                    black_box(lu);
                });
            },
        );
    }

    group.finish();
}

fn bench_dense_block_tridiag_factor_plus_solve_vs_faer(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_block_tridiag_factor_plus_solve_vs_faer");
    group.sample_size(10);

    let cases = [(50, 50), (100, 50), (100, 100)];

    for &(n_blocks, block_size) in &cases {
        let a = generate_block_tridiagonal_banded(n_blocks, block_size, 123);
        let (_, b0) = generate_rhs_from_known_solution(&a, 777);

        let n = n_blocks * block_size;
        let kl = a.kl();
        let ku = a.ku();

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let rhs_faer = rhs_vec_to_faer_col(&b0);

        group.bench_with_input(
            BenchmarkId::new(
                "lapack_style_banded_lu",
                format!("n={n},block={block_size}"),
            ),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();

                    let mut rhs = b0.clone();
                    lu.solve_in_place(black_box(&mut rhs)).unwrap();
                    black_box(rhs);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("faer_sparse_lu", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let lu = black_box(&sparse).sp_lu().unwrap();
                    let x = lu.solve(black_box(&rhs_faer));
                    black_box(x);
                });
            },
        );
    }

    group.finish();
}

fn bench_narrow_block_tridiag_factor_vs_faer(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrow_block_tridiag_factor_vs_faer");
    group.sample_size(10);

    let cases = [(100, 100, 8, 4), (100, 100, 8, 8), (100, 100, 12, 12)];

    for &(n_blocks, block_size, diag_half_bw, coupling_half_bw) in &cases {
        let a = generate_block_tridiagonal_narrow_coupling_banded(
            n_blocks,
            block_size,
            diag_half_bw,
            coupling_half_bw,
            42,
        );

        let n = n_blocks * block_size;
        let kl = a.kl();
        let ku = a.ku();

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let tag =
            format!("n={n},block={block_size},diagbw={diag_half_bw},couplingbw={coupling_half_bw}");

        group.bench_with_input(
            BenchmarkId::new("lapack_style_banded_lu", &tag),
            &(n_blocks, block_size, diag_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("faer_sparse_lu", &tag),
            &(n_blocks, block_size, diag_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let lu = black_box(&sparse).sp_lu().unwrap();
                    black_box(lu);
                });
            },
        );
    }

    group.finish();
}

fn bench_bvp1d_factor_vs_faer(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvp1d_factor_vs_faer");
    group.sample_size(10);

    let cases = [
        (100, 6, 2, 1),   // n=600
        (500, 6, 2, 1),   // n=3000
        (1000, 6, 2, 1),  // n=6000
        (1000, 12, 3, 2), // n=12000
    ];

    for &(n_nodes, vars_per_node, local_half_bw, coupling_half_bw) in &cases {
        let a = generate_bvp1d_banded(n_nodes, vars_per_node, local_half_bw, coupling_half_bw, 42);

        let n = a.n();
        let kl = a.kl();
        let ku = a.ku();

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let tag = format!(
            "n={n},vars={vars_per_node},localbw={local_half_bw},couplingbw={coupling_half_bw}"
        );

        group.bench_with_input(
            BenchmarkId::new("lapack_style_banded_lu", &tag),
            &(n_nodes, vars_per_node, local_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("faer_sparse_lu", &tag),
            &(n_nodes, vars_per_node, local_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let lu = black_box(&sparse).sp_lu().unwrap();
                    black_box(lu);
                });
            },
        );
    }

    group.finish();
}

fn bench_bvp1d_factor_plus_solve_vs_faer(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvp1d_factor_plus_solve_vs_faer");
    group.sample_size(10);

    let cases = [
        (100, 6, 2, 1),
        (500, 6, 2, 1),
        (1000, 6, 2, 1),
        (1000, 12, 3, 2),
    ];

    for &(n_nodes, vars_per_node, local_half_bw, coupling_half_bw) in &cases {
        let a = generate_bvp1d_banded(n_nodes, vars_per_node, local_half_bw, coupling_half_bw, 123);

        let (_, b0) = generate_rhs_from_known_solution(&a, 777);

        let n = a.n();
        let kl = a.kl();
        let ku = a.ku();

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let rhs_faer = rhs_vec_to_faer_col(&b0);

        let tag = format!(
            "n={n},vars={vars_per_node},localbw={local_half_bw},couplingbw={coupling_half_bw}"
        );

        group.bench_with_input(
            BenchmarkId::new("lapack_style_banded_lu", &tag),
            &(n_nodes, vars_per_node, local_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let mut lu = LapackStyleBandedLuFaithful::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();

                    let mut rhs = b0.clone();
                    lu.solve_in_place(black_box(&mut rhs)).unwrap();
                    black_box(rhs);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("faer_sparse_lu", &tag),
            &(n_nodes, vars_per_node, local_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let lu = black_box(&sparse).sp_lu().unwrap();
                    let x = lu.solve(black_box(&rhs_faer));
                    black_box(x);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    faer_sparse_benches,
    bench_dense_block_tridiag_factor_vs_faer,
    bench_dense_block_tridiag_factor_plus_solve_vs_faer,
    bench_narrow_block_tridiag_factor_vs_faer,
    bench_bvp1d_factor_vs_faer,
    bench_bvp1d_factor_plus_solve_vs_faer,
);
criterion_main!(faer_sparse_benches);
