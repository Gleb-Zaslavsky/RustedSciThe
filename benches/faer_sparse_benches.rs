use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;

use RustedSciThe::somelinalg::banded::{Banded, GeneralBandedLuPartialPivot, banded_matvec};

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

/// Narrow-coupling block-tridiagonal benchmark matrix.
/// Global scalar half-bandwidth = block_size + coupling_half_bw.
fn generate_block_tridiagonal_narrow_coupling_banded(
    n_blocks: usize,
    block_size: usize,
    diag_half_bw: usize,
    coupling_half_bw: usize,
    seed: u64,
) -> Banded<f64> {
    assert!(n_blocks > 0);
    assert!(block_size > 0);
    assert!(diag_half_bw < block_size);
    assert!(coupling_half_bw < block_size);

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

            // diagonal block
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
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        group.bench_with_input(
            BenchmarkId::new("banded_partial_pivot", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
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
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let rhs_faer = rhs_vec_to_faer_col(&b0);

        group.bench_with_input(
            BenchmarkId::new("banded_partial_pivot", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
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
        let kl = block_size + coupling_half_bw;
        let ku = block_size + coupling_half_bw;

        let triplets = banded_to_triplets(&a);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let tag =
            format!("n={n},block={block_size},diagbw={diag_half_bw},couplingbw={coupling_half_bw}");

        group.bench_with_input(
            BenchmarkId::new("banded_partial_pivot", &tag),
            &(n_blocks, block_size, diag_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
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

criterion_group!(
    faer_sparse_benches,
    bench_dense_block_tridiag_factor_vs_faer,
    bench_dense_block_tridiag_factor_plus_solve_vs_faer,
    bench_narrow_block_tridiag_factor_vs_faer,
);
criterion_main!(faer_sparse_benches);
