use RustedSciThe::somelinalg::banded::{BlockTridiagonalLu, GeneralBandedLuPartialPivot};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::prelude::Solve;
use faer::sparse::SparseColMat;
use std::hint::black_box;

use RustedSciThe::somelinalg::banded::block_tridiag_bench_helpers::{
    block_to_banded, block_to_triplets, generate_block_tridiagonal_dense,
    generate_block_tridiagonal_narrow, generate_rhs_from_known_solution_block,
};

fn bench_block_dense_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_dense_factor");
    group.sample_size(10);

    let cases = [(50, 50), (100, 50), (100, 100)];

    for &(n_blocks, block_size) in &cases {
        let a_block = generate_block_tridiagonal_dense(n_blocks, block_size, 42);
        let a_banded = block_to_banded(&a_block);

        let n = a_block.n();
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        let triplets = block_to_triplets(&a_block);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        group.bench_with_input(
            BenchmarkId::new("block_lu", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = BlockTridiagonalLu::new(n_blocks, block_size).unwrap();
                    lu.factor_from(black_box(&a_block)).unwrap();
                    black_box(lu);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_banded_lu", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a_banded)).unwrap();
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

fn bench_block_dense_factor_plus_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_dense_factor_plus_solve");
    group.sample_size(10);

    let cases = [(50, 50), (100, 50), (100, 100)];

    for &(n_blocks, block_size) in &cases {
        let a_block = generate_block_tridiagonal_dense(n_blocks, block_size, 123);
        let a_banded = block_to_banded(&a_block);

        let (_, b0) = generate_rhs_from_known_solution_block(&a_block, 777);

        let n = a_block.n();
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        let triplets = block_to_triplets(&a_block);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        group.bench_with_input(
            BenchmarkId::new("block_lu", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = BlockTridiagonalLu::new(n_blocks, block_size).unwrap();
                    lu.factor_from(black_box(&a_block)).unwrap();

                    let mut rhs = b0.clone();
                    lu.solve_in_place(black_box(&mut rhs)).unwrap();
                    black_box(rhs);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_banded_lu", format!("n={n},block={block_size}")),
            &(n_blocks, block_size),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a_banded)).unwrap();

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
                    let x = lu.solve(black_box(&faer::Col::from_fn(n, |i| b0[i])));
                    black_box(x);
                });
            },
        );
    }

    group.finish();
}

fn bench_block_narrow_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_narrow_factor");
    group.sample_size(10);

    let cases = [(100, 100, 8, 4), (100, 100, 8, 8), (100, 100, 12, 12)];

    for &(n_blocks, block_size, diag_half_bw, coupling_half_bw) in &cases {
        let a_block = generate_block_tridiagonal_narrow(
            n_blocks,
            block_size,
            diag_half_bw,
            coupling_half_bw,
            42,
        );

        let n = a_block.n();
        let kl = 2 * block_size - 1;
        let ku = 2 * block_size - 1;

        let a_banded = block_to_banded(&a_block);
        let triplets = block_to_triplets(&a_block);
        let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let tag =
            format!("n={n},block={block_size},diagbw={diag_half_bw},couplingbw={coupling_half_bw}");

        group.bench_with_input(
            BenchmarkId::new("block_lu", &tag),
            &(n_blocks, block_size, diag_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let mut lu = BlockTridiagonalLu::new(n_blocks, block_size).unwrap();
                    lu.factor_from(black_box(&a_block)).unwrap();
                    black_box(lu);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_banded_lu", &tag),
            &(n_blocks, block_size, diag_half_bw, coupling_half_bw),
            |b, _| {
                b.iter(|| {
                    let mut lu = GeneralBandedLuPartialPivot::new(n, kl, ku).unwrap();
                    lu.factor_from(black_box(&a_banded)).unwrap();
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
    block_tridiag_benches,
    bench_block_dense_factor,
    bench_block_dense_factor_plus_solve,
    bench_block_narrow_factor,
);
criterion_main!(block_tridiag_benches);
