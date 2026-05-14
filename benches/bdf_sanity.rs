use RustedSciThe::numerical::LSODE::core::{BdfOptions, FdOptions, OdeRhs, SolverStats, TolMode};
use RustedSciThe::numerical::LSODE::solver::{new_bdf_dense_analytic, new_bdf_dense_fd};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// Подставь правильные пути импорта под свою crate-структуру.

struct Robertson;

impl RustedSciThe::numerical::LSODE::core::OdeRhs for Robertson {
    fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
        let y1 = y[0];
        let y2 = y[1];
        let y3 = y[2];

        fy[0] = -0.04 * y1 + 1.0e4 * y2 * y3;
        fy[1] = 0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2;
        fy[2] = 3.0e7 * y2 * y2;
    }
}

fn robertson_jac(_t: f64, y: &[f64], jac: &mut [f64], n: usize) {
    assert_eq!(n, 3);
    jac.fill(0.0);

    let y2 = y[1];
    let y3 = y[2];
    let idx = |i: usize, j: usize| -> usize { j * n + i };

    jac[idx(0, 0)] = -0.04;
    jac[idx(0, 1)] = 1.0e4 * y3;
    jac[idx(0, 2)] = 1.0e4 * y2;

    jac[idx(1, 0)] = 0.04;
    jac[idx(1, 1)] = -1.0e4 * y3 - 6.0e7 * y2;
    jac[idx(1, 2)] = -1.0e4 * y2;

    jac[idx(2, 1)] = 6.0e7 * y2;
}

struct DiagonalStiff3;

impl OdeRhs for DiagonalStiff3 {
    fn eval(&mut self, _t: f64, y: &[f64], fy: &mut [f64]) {
        fy[0] = -1.0 * y[0];
        fy[1] = -100.0 * y[1];
        fy[2] = -10_000.0 * y[2];
    }
}

fn diagonal_stiff3_jac(_t: f64, _y: &[f64], jac: &mut [f64], n: usize) {
    assert_eq!(n, 3);
    jac.fill(0.0);
    let idx = |i: usize, j: usize| -> usize { j * n + i };
    jac[idx(0, 0)] = -1.0;
    jac[idx(1, 1)] = -100.0;
    jac[idx(2, 2)] = -10_000.0;
}

fn bench_robertson_analytic(c: &mut Criterion) {
    c.bench_function("robertson analytic to 1e-2", |b| {
        b.iter(|| {
            let tol = TolMode::Scalar {
                rtol: 1e-6,
                atol: 1e-12,
            };
            let opts = BdfOptions {
                h0: Some(1e-8),
                max_steps: 500_000,
                ..Default::default()
            };

            let mut solver = new_bdf_dense_analytic(
                Robertson,
                robertson_jac,
                0.0,
                vec![1.0, 0.0, 0.0],
                tol,
                opts,
            )
            .unwrap();

            solver.integrate_to(1.0e-2).unwrap();

            black_box(solver.y().to_vec());
            black_box(solver.stats().clone());
        })
    });
}

fn bench_robertson_fd(c: &mut Criterion) {
    c.bench_function("robertson fd to 1e-2", |b| {
        b.iter(|| {
            let tol = TolMode::Scalar {
                rtol: 1e-6,
                atol: 1e-12,
            };
            let opts = BdfOptions {
                h0: Some(1e-8),
                max_steps: 500_000,
                ..Default::default()
            };

            let mut solver = new_bdf_dense_fd(
                Robertson,
                0.0,
                vec![1.0, 0.0, 0.0],
                tol,
                opts,
                FdOptions {
                    rel_step: 1e-7,
                    abs_step: 1e-8,
                },
            )
            .unwrap();

            solver.integrate_to(1.0e-2).unwrap();

            black_box(solver.y().to_vec());
            black_box(solver.stats().clone());
        })
    });
}

fn bench_diagonal_stiff3_analytic(c: &mut Criterion) {
    c.bench_function("diagonal stiff3 analytic to 0.1", |b| {
        b.iter(|| {
            let tol = TolMode::Scalar {
                rtol: 1e-6,
                atol: 1e-10,
            };
            let opts = BdfOptions {
                h0: Some(1e-6),
                max_steps: 300_000,
                ..Default::default()
            };

            let mut solver = new_bdf_dense_analytic(
                DiagonalStiff3,
                diagonal_stiff3_jac,
                0.0,
                vec![1.0, 1.0, 1.0],
                tol,
                opts,
            )
            .unwrap();

            solver.integrate_to(0.1).unwrap();

            black_box(solver.y().to_vec());
            black_box(solver.stats().clone());
        })
    });
}

criterion_group!(
    benches,
    bench_robertson_analytic,
    bench_robertson_fd,
    bench_diagonal_stiff3_analytic
);
criterion_main!(benches);

fn print_stats(label: &str, stats: &SolverStats) {
    eprintln!(
        "{}: steps={}, accepted={}, rejected={}, f={}, j={}, lu={}, lin={}, newton={}",
        label,
        stats.n_steps,
        stats.n_accepted,
        stats.n_rejected,
        stats.n_f_evals,
        stats.n_j_evals,
        stats.n_lu_factorizations,
        stats.n_linear_solves,
        stats.n_newton_iters,
    );
}
