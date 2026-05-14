use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;

use RustedSciThe::somelinalg::banded::{
    Banded, GeneralBandedLuNoPivot, GeneralBandedLuPartialPivot, banded_matvec,
};

fn generate_diag_dominant_banded(n: usize, kl: usize, ku: usize, seed: u64) -> Banded<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

    for j in 0..n {
        let i0 = j.saturating_sub(ku);
        let i1 = (j + kl + 1).min(n);

        let mut col_abs_sum = 0.0;
        for i in i0..i1 {
            if i == j {
                continue;
            }
            let v = rng.random_range(-1.0..1.0);
            a[(i, j)] = v;
            col_abs_sum += v.abs();
        }

        a[(j, j)] = col_abs_sum + rng.random_range(1.0..2.0);
    }

    a
}

fn generate_rhs(a: &Banded<f64>, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x_true: Vec<f64> = (0..a.n()).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b = banded_matvec(a, &x_true).unwrap();
    (x_true, b)
}

fn bench_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("factor");

    let cases = [
        (100, 1, 1),
        (1000, 1, 1),
        (1000, 2, 2),
        (1000, 4, 4),
        (5000, 2, 2),
    ];

    for &(n, kl, ku) in &cases {
        let a = generate_diag_dominant_banded(n, kl, ku, 42);

        group.bench_with_input(
            BenchmarkId::new("no_pivot", format!("n={n},kl={kl},ku={ku}")),
            &(n, kl, ku),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuNoPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("partial_pivot", format!("n={n},kl={kl},ku={ku}")),
            &(n, kl, ku),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );
    }

    group.finish();
}

fn bench_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");

    let cases = [(1000, 1, 1), (1000, 2, 2), (5000, 2, 2)];

    for &(n, kl, ku) in &cases {
        let a = generate_diag_dominant_banded(n, kl, ku, 123);
        let (_, b0) = generate_rhs(&a, 777);

        let mut lu_np = GeneralBandedLuNoPivot::new(n, kl, ku).unwrap();
        lu_np.factor_from(&a).unwrap();

        let mut lu_pp = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
        lu_pp.factor_from(&a).unwrap();

        group.bench_with_input(
            BenchmarkId::new("no_pivot", format!("n={n},kl={kl},ku={ku}")),
            &(n, kl, ku),
            |b, _| {
                b.iter(|| {
                    let mut rhs = b0.clone();
                    lu_np.solve_in_place(black_box(&mut rhs)).unwrap();
                    black_box(rhs);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("partial_pivot", format!("n={n},kl={kl},ku={ku}")),
            &(n, kl, ku),
            |b, _| {
                b.iter(|| {
                    let mut rhs = b0.clone();
                    lu_pp.solve_in_place(black_box(&mut rhs)).unwrap();
                    black_box(rhs);
                });
            },
        );
    }

    group.finish();
}

fn bench_factor_plus_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("factor_plus_solve");

    let cases = [(100, 1, 1), (1000, 2, 2), (5000, 2, 2)];

    for &(n, kl, ku) in &cases {
        let a = generate_diag_dominant_banded(n, kl, ku, 999);
        let (_, b0) = generate_rhs(&a, 555);

        group.bench_with_input(
            BenchmarkId::new("partial_pivot", format!("n={n},kl={kl},ku={ku}")),
            &(n, kl, ku),
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
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_factor,
    bench_solve,
    bench_factor_plus_solve,
    bench_block_tridiag_solve,
    bench_block_tridiag_factor_plus_solve,
    bench_block_tridiag_dense_factor,
    bench_block_tridiag_narrow_factor,
);
criterion_main!(benches);

//============================================================================================================

/// Generate a block-tridiagonal banded matrix.
///
/// Structure:
///   [ B0  C0   0   0  ... ]
///   [ A1  B1  C1   0  ... ]
///   [  0  A2  B2  C2 ... ]
///   [ ...                ]
///
/// where each block is block_size x block_size, and the total matrix size is:
///   n = n_blocks * block_size
///
/// The resulting scalar matrix is banded with:
///   kl = ku = block_size
///
/// Diagonal blocks are made strongly diagonally dominant to improve numerical stability.
fn generate_block_tridiagonal_banded(n_blocks: usize, block_size: usize, seed: u64) -> Banded<f64> {
    assert!(n_blocks > 0);
    assert!(block_size > 0);

    let n = n_blocks * block_size;

    // IMPORTANT:
    // dense block-tridiagonal with block size b has scalar half-bandwidth 2*b - 1
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

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

    for blk in 0..n_blocks {
        let row0 = blk * block_size;
        let col0 = blk * block_size;

        for local_j in 0..block_size {
            let j = col0 + local_j;
            let mut abs_sum = 0.0;

            let li0 = local_j.saturating_sub(diag_half_bw);
            let li1 = (local_j + diag_half_bw + 1).min(block_size);

            for local_i in li0..li1 {
                let i = row0 + local_i;
                if i == j {
                    continue;
                }

                let v = rng.random_range(-0.2..0.2);
                debug_assert!(a.in_band(i, j), "diag block: ({i}, {j})");
                a[(i, j)] = v;
                abs_sum += v.abs();
            }

            if blk > 0 {
                let prev_row0 = (blk - 1) * block_size;

                let li0 = local_j.saturating_sub(coupling_half_bw);
                let li1 = (local_j + coupling_half_bw + 1).min(block_size);

                for local_i in li0..li1 {
                    let i = prev_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    debug_assert!(a.in_band(i, j), "lower neighbor: ({i}, {j})");
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            if blk + 1 < n_blocks {
                let next_row0 = (blk + 1) * block_size;

                let li0 = local_j.saturating_sub(coupling_half_bw);
                let li1 = (local_j + coupling_half_bw + 1).min(block_size);

                for local_i in li0..li1 {
                    let i = next_row0 + local_i;
                    let v = rng.random_range(-0.05..0.05);
                    debug_assert!(a.in_band(i, j), "upper neighbor: ({i}, {j})");
                    a[(i, j)] = v;
                    abs_sum += v.abs();
                }
            }

            a[(j, j)] = abs_sum + rng.random_range(1.0..2.0);
        }
    }

    a
}
fn generate_rhs_from_known_solution(a: &Banded<f64>, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x_true: Vec<f64> = (0..a.n()).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b = RustedSciThe::somelinalg::banded::banded_matvec(a, &x_true).unwrap();
    (x_true, b)
}

fn bench_block_tridiag_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_tridiag_factor");

    let cases = [
        (50, 50),   // n=2500, dense block bandwidth = 99
        (100, 50),  // n=5000, dense block bandwidth = 99
        (100, 100), // n=10000, dense block bandwidth = 199
    ];

    for &(n_blocks, block_size) in &cases {
        let a = generate_block_tridiagonal_banded(n_blocks, block_size, 42);
        let n = n_blocks * block_size;
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        group.bench_with_input(
            BenchmarkId::new("partial_pivot", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );
    }

    group.finish();
}

fn bench_block_tridiag_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_tridiag_solve");

    let cases = [(100, 50), (100, 100)];

    for &(n_blocks, block_size) in &cases {
        let a = generate_block_tridiagonal_banded(n_blocks, block_size, 123);
        let (_, b0) = generate_rhs_from_known_solution(&a, 777);

        let n = n_blocks * block_size;
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;
        let mut lu =
            RustedSciThe::somelinalg::banded::GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();

        lu.factor_from(&a).unwrap();

        group.bench_with_input(
            BenchmarkId::new("partial_pivot", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut rhs = b0.clone();
                    lu.solve_in_place(black_box(&mut rhs)).unwrap();
                    black_box(rhs);
                });
            },
        );
    }

    group.finish();
}

fn bench_block_tridiag_factor_plus_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_tridiag_factor_plus_solve");

    let cases = [(50, 50), (100, 50), (100, 100)];

    for &(n_blocks, block_size) in &cases {
        let a = generate_block_tridiagonal_banded(n_blocks, block_size, 999);
        let (_, b0) = generate_rhs_from_known_solution(&a, 555);
        let n = n_blocks * block_size;

        group.bench_with_input(
            BenchmarkId::new("partial_pivot", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let kl = 2 * block_size - 1;
                    let ku = 2 * block_size - 1;

                    let mut lu =
                        RustedSciThe::somelinalg::banded::GeneralBandedLuPartialPivot::new(
                            n, kl, ku,
                        )
                        .unwrap();

                    lu.factor_from(black_box(&a)).unwrap();

                    let mut rhs = b0.clone();
                    lu.solve_in_place(black_box(&mut rhs)).unwrap();

                    black_box(rhs);
                });
            },
        );
    }

    group.finish();
}
fn bench_block_tridiag_dense_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_tridiag_dense_factor");

    let cases = [(50, 50), (100, 50), (100, 100)];

    for &(n_blocks, block_size) in &cases {
        let a = generate_block_tridiagonal_banded(n_blocks, block_size, 42);
        let n = n_blocks * block_size;
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        group.bench_with_input(
            BenchmarkId::new("partial_pivot", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );
    }

    group.finish();
}
fn bench_block_tridiag_narrow_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_tridiag_narrow_factor");

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

        group.bench_with_input(
            BenchmarkId::new(
                "partial_pivot",
                format!(
                    "n={n},block={block_size},diagbw={diag_half_bw},couplingbw={coupling_half_bw}"
                ),
            ),
            &(n_blocks, block_size, diag_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a)).unwrap();
                    black_box(lu);
                });
            },
        );
    }

    group.finish();
}
#[test]
fn block_tridiag_large_system_solves_with_small_residual() {
    let a = generate_block_tridiagonal_banded(20, 20, 123);
    let (x_true, b) = generate_rhs_from_known_solution(&a, 456);

    let mut x = b.clone();
    let mut lu = GeneralBandedLuPartialPivot::new(a.n(), a.kl(), a.ku()).unwrap();
    lu.factor_from(&a).unwrap();
    lu.solve_in_place(&mut x).unwrap();

    let res = RustedSciThe::somelinalg::banded::residual_linf(&a, &x, &b).unwrap();
    assert!(res < 1e-8);

    for i in 0..x.len() {
        assert!((x[i] - x_true[i]).abs() < 1e-8);
    }
}
