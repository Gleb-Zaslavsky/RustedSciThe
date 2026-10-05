//! Чисто числовой гайд по `BVP_sci`.
//!
//! Этот пример - пользовательская точка входа для BVP без символики.
//! Он следует тому же closure-first шаблону, что и чисто числовой BVP_Damp,
//! но остаётся специфичным для BVP:
//! - `NumericalBvpClosureProblem::new_fd(...)` - самый простой режим: передайте
//!   closures для `rhs()` и `boundary_residual()`.
//! - `NumericalBvpClosureProblem::new_with_jacobian(...)` - режим с упором на
//!   производительность: передайте те же closures плюс pointwise якобианы.
//! - `NumericalBvpSolveOptions::with_singular_term(Some(S))` включает ветку
//!   в стиле SciPy для singular-term `dy/dx = f(x, y, p) + S * y / (x - a)`.
//!
//! Практическое правило из текущего набора сравнений:
//! - начинайте с `FiniteDifference` для маленьких систем или быстрого прототипирования;
//! - переходите к `AnalyticalPointwise`, когда задача становится средней/большой или ту же модель нужно решать многократно с хорошим throughput.
//! 
//! запуск: cargo run --example bvp_sci_numerical_guide

use std::time::Instant;

use RustedSciThe::numerical::BVP_Damp::BVP_utils::CustomTimer;
use RustedSciThe::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat, faer_mat};
use RustedSciThe::numerical::BVP_sci::BVP_sci_numerical::{
    NumericalBvpClosureProblem, NumericalBvpSolveOptions, NumericalJacobianMode,
    solve_numerical_bvp, solve_numerical_bvp_fd, solve_numerical_bvp_with_jacobian,
};
use faer::sparse::Triplet;

fn harmonic_mesh(n_steps: usize) -> faer_col {
    faer_col::from_fn(n_steps, |i| {
        std::f64::consts::PI * i as f64 / (n_steps.saturating_sub(1)) as f64
    })
}

fn harmonic_initial_guess(mesh: &faer_col) -> faer_dense_mat {
    faer_dense_mat::from_fn(2, mesh.nrows(), |i, j| match i {
        0 => mesh[j].sin(),
        1 => mesh[j].cos(),
        _ => 0.0,
    })
}

fn harmonic_fd_problem() -> NumericalBvpClosureProblem {
    NumericalBvpClosureProblem::new_fd(
        2,
        0,
        |_x, y, _p, out| {
            out[0] = y[1];
            out[1] = -y[0];
        },
        |ya, _yb, _p, out| {
            out[0] = ya[0];
            out[1] = ya[1] - 1.0;
        },
    )
}

fn harmonic_analytical_problem() -> NumericalBvpClosureProblem {
    NumericalBvpClosureProblem::new_with_jacobian(
        2,
        0,
        |_x, y, _p, out| {
            out[0] = y[1];
            out[1] = -y[0];
        },
        |ya, _yb, _p, out| {
            out[0] = ya[0];
            out[1] = ya[1] - 1.0;
        },
        |_x, _y, _p| {
            Some(
                faer_mat::try_new_from_triplets(
                    2,
                    2,
                    &[
                        Triplet::new(0usize, 1usize, 1.0),
                        Triplet::new(1usize, 0usize, -1.0),
                    ],
                )
                .expect("harmonic rhs Jacobian should be constructible"),
            )
        },
        |_ya, _yb, _p| {
            let dya = faer_mat::try_new_from_triplets(
                2,
                2,
                &[
                    Triplet::new(0usize, 0usize, 1.0),
                    Triplet::new(1usize, 1usize, 1.0),
                ],
            )
            .expect("harmonic left BC Jacobian should be constructible");
            let dyb = faer_mat::try_new_from_triplets(2, 2, &[])
                .expect("harmonic right BC Jacobian should be constructible");
            Some((dya, dyb, None))
        },
    )
}

fn run_case(label: &str, problem: NumericalBvpClosureProblem, n_steps: usize) {
    let mesh = harmonic_mesh(n_steps);
    let guess = harmonic_initial_guess(&mesh);
    let custom_timer = CustomTimer::new();
    let started = Instant::now();
    let result = if label == "FiniteDifference" {
        solve_numerical_bvp_fd(
            problem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512)
                .with_verbose(0)
                .with_custom_timer(Some(custom_timer)),
        )
    } else {
        solve_numerical_bvp_with_jacobian(
            problem,
            NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512)
                .with_verbose(0)
                .with_custom_timer(Some(custom_timer)),
        )
    }
    .unwrap_or_else(|err| panic!("{label} failed: {err}"));
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let (nrows, ncols) = result.y.shape();
    let mut max_abs_solution = 0.0_f64;
    for i in 0..nrows {
        for j in 0..ncols {
            max_abs_solution = max_abs_solution.max(result.y[(i, j)].abs());
        }
    }
    let timer = &result.custom_timer;

    println!(
        "{label:28} total_ms = {total_ms:9.3}, residual_ms = {:8.3}, jacobian_ms = {:8.3}, linear_ms = {:8.3}, max_abs_solution = {:.6e}",
        timer.fun.as_secs_f64() * 1_000.0,
        timer.jac.as_secs_f64() * 1_000.0,
        timer.linear_system.as_secs_f64() * 1_000.0,
        max_abs_solution
    );
}

fn print_guide() {
    println!("BVP_sci чисто числовой гайд");
    println!("============================");
    println!("Два публичных режима и два явных helper-а:");
    println!("  - NumericalBvpClosureProblem::new_fd(...): easiest closure-first API");
    println!(
        "  - NumericalBvpClosureProblem::new_with_jacobian(...): closure-first analytical API"
    );
    println!(
        "  - NumericalBvpSolveOptions::with_singular_term(Some(S)): SciPy-style singular-term plumbing"
    );
    println!("  - solve_numerical_bvp_fd(...): explicit FD route");
    println!("  - solve_numerical_bvp_with_jacobian(...): explicit analytical route");
    println!();
    println!("Когда что выбирать:");
    println!("  - используйте FiniteDifference для быстрых прототипов и маленьких систем;");
    println!("  - используйте AnalyticalPointwise, когда стоимость якобиана начинает доминировать");
    println!("    или когда одну и ту же числовую модель нужно решать многократно.");
    println!();
    println!("Демо-задача: y'' + y = 0 с y(0) = 0 и y'(0) = 1.");
    println!("Точное решение: y(x) = sin(x), y'(x) = cos(x).");
    println!();
}

fn main() {
    let n_steps = 64;
    print_guide();
    run_case("FiniteDifference", harmonic_fd_problem(), n_steps);
    run_case(
        "AnalyticalPointwise",
        harmonic_analytical_problem(),
        n_steps,
    );
}
