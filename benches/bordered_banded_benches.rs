//! Structured bordered-banded solve versus a standard `faer` sparse LU.
//!
//! The matrix is assembled once before each benchmark group. The cold case
//! measures factorization plus solve; the warm case factors once and measures
//! repeated solves only. This isolates the structured linear algebra question
//! from BVP mesh assembly and symbolic preparation.

use RustedSciThe::somelinalg::banded::BorderedBlockTridiagonal;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::prelude::Solve;
use faer::sparse::{SparseColMat, Triplet};
use std::hint::black_box;

struct Case {
    structured: BorderedBlockTridiagonal,
    sparse: SparseColMat<usize, f64>,
    rhs: Vec<f64>,
}

fn push_block(
    triplets: &mut Vec<Triplet<usize, usize, f64>>,
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
    values: &[f64],
) {
    for row in 0..rows {
        for column in 0..cols {
            let value = values[row * cols + column];
            if value != 0.0 {
                triplets.push(Triplet::new(row0 + row, col0 + column, value));
            }
        }
    }
}

fn make_case(n_blocks: usize) -> Case {
    let block_size = 2;
    let border_size = 2;
    let core_dimension = n_blocks * block_size;
    let n = core_dimension + border_size;
    let mut structured = BorderedBlockTridiagonal::zeros(n_blocks, block_size, border_size)
        .expect("valid benchmark dimensions");
    let mut triplets = Vec::new();

    for block in 0..n_blocks {
        let diagonal = [
            4.5 + 0.01 * block as f64,
            0.2,
            0.1,
            4.0 + 0.01 * block as f64,
        ];
        for row in 0..block_size {
            for column in 0..block_size {
                structured
                    .core_mut()
                    .set_diag(block, row, column, diagonal[row * block_size + column])
                    .unwrap();
            }
        }
        push_block(
            &mut triplets,
            block * block_size,
            block * block_size,
            block_size,
            block_size,
            &diagonal,
        );

        if block + 1 < n_blocks {
            let lower = [-0.4, 0.05, 0.0, -0.3];
            let upper = [-0.2, 0.0, 0.08, -0.25];
            for row in 0..block_size {
                for column in 0..block_size {
                    structured
                        .core_mut()
                        .set_lower(block, row, column, lower[row * block_size + column])
                        .unwrap();
                    structured
                        .core_mut()
                        .set_upper(block, row, column, upper[row * block_size + column])
                        .unwrap();
                }
            }
            push_block(
                &mut triplets,
                (block + 1) * block_size,
                block * block_size,
                block_size,
                block_size,
                &lower,
            );
            push_block(
                &mut triplets,
                block * block_size,
                (block + 1) * block_size,
                block_size,
                block_size,
                &upper,
            );
        }
    }

    let mut core_to_border = vec![0.0; core_dimension * border_size];
    let mut border_to_core = vec![0.0; border_size * core_dimension];
    for row in 0..core_dimension {
        core_to_border[row * border_size] = 0.3 / (1.0 + row as f64);
        core_to_border[row * border_size + 1] = -0.15 / (1.0 + row as f64);
        border_to_core[row] = 0.2 / (1.0 + row as f64);
        border_to_core[core_dimension + row] = -0.1 / (1.0 + row as f64);
    }
    structured
        .core_to_border_mut()
        .copy_from_slice(&core_to_border);
    structured
        .border_to_core_mut()
        .copy_from_slice(&border_to_core);
    push_block(
        &mut triplets,
        0,
        core_dimension,
        core_dimension,
        border_size,
        &core_to_border,
    );
    push_block(
        &mut triplets,
        core_dimension,
        0,
        border_size,
        core_dimension,
        &border_to_core,
    );

    let border = [3.0, 0.2, 0.1, 2.5];
    structured.border_mut().copy_from_slice(&border);
    push_block(
        &mut triplets,
        core_dimension,
        core_dimension,
        border_size,
        border_size,
        &border,
    );

    let expected = (0..n)
        .map(|index| 0.25 + 0.17 * index as f64)
        .collect::<Vec<_>>();
    let mut rhs = vec![0.0; n];
    for triplet in &triplets {
        rhs[triplet.row] += triplet.val * expected[triplet.col];
    }
    let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .expect("benchmark matrix must be valid");

    Case {
        structured,
        sparse,
        rhs,
    }
}

fn bench_bordered_vs_faer(c: &mut Criterion) {
    let mut group = c.benchmark_group("bordered_banded_vs_faer_sparse");
    group.sample_size(10);

    for n_blocks in [16usize, 64, 128] {
        let case = make_case(n_blocks);
        let n = case.rhs.len();
        let tag = format!("n={n},blocks={n_blocks},block=2,border=2");

        group.bench_with_input(
            BenchmarkId::new("structured/factor_plus_solve", &tag),
            &case,
            |b, case| {
                b.iter_batched(
                    || (case.structured.clone(), case.rhs.clone()),
                    |(mut structured, mut rhs)| {
                        structured.factor().unwrap();
                        structured.solve_in_place(&mut rhs).unwrap();
                        black_box(rhs);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("faer_sparse/factor_plus_solve", &tag),
            &case,
            |b, case| {
                b.iter_batched(
                    || case.rhs.clone(),
                    |rhs| {
                        let lu = case.sparse.sp_lu().unwrap();
                        let solution = lu.solve(&faer::Col::from_fn(n, |i| rhs[i]));
                        black_box(solution);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        let mut structured = case.structured.clone();
        structured.factor().unwrap();
        let mut structured_rhs = case.rhs.clone();
        group.bench_function(BenchmarkId::new("structured/warm_solve", &tag), |b| {
            b.iter(|| {
                structured_rhs.copy_from_slice(&case.rhs);
                structured.solve_in_place(&mut structured_rhs).unwrap();
                black_box(&structured_rhs);
            });
        });

        let sparse_lu = case.sparse.sp_lu().unwrap();
        group.bench_function(BenchmarkId::new("faer_sparse/warm_solve", &tag), |b| {
            b.iter(|| {
                let solution = sparse_lu.solve(&faer::Col::from_fn(n, |i| case.rhs[i]));
                black_box(solution);
            });
        });
    }

    group.finish();
}

criterion_group!(bordered_banded_benches, bench_bordered_vs_faer);
criterion_main!(bordered_banded_benches);
