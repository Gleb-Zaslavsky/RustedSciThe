//! Guide to the current native banded solvers.
//!
//! The default choice for a general compact banded matrix is
//! `LapackStyleBandedLuFaithful`.
//!
//! That solver is the production path because it mirrors LAPACK banded LU
//! (`DGBTRF` + `DGBTRS`) without hiding a dense `n x n` factorization behind the
//! API. It is the solver selected by the high-level `ForceBanded`/`Auto` banded
//! assembly path.
//!
//! The block-tridiagonal solvers are still valuable, but they are specialized:
//! use them only when the matrix really is a block-tridiagonal chain and you
//! deliberately want to exploit that structure.
//!
//! Practical rule of thumb:
//! 1. If you only know that the matrix is banded, start with
//!    `LapackStyleBandedLuFaithful`.
//! 2. If the matrix is genuinely block-tridiagonal and every local block solve
//!    is well behaved, try `BlockTridiagonalLu` for speed.
//! 3. If the block-tridiagonal chain is structurally difficult, try
//!    `BlockTridiagonalLuConsistent` as an opt-in structured experiment.
//! 4. Keep iterative refinement explicit. The faithful LAPACK-style solver uses
//!    `refine = 0` by default because the extra correction pass has not helped
//!    the current BVP workloads.

use RustedSciThe::somelinalg::banded::block_tridiag_bench_helpers::{
    block_to_banded, generate_block_tridiagonal_dense, generate_rhs_from_known_solution_block,
};
use RustedSciThe::somelinalg::banded::{
    Banded, BlockTridiagonal, BlockTridiagonalLu, BlockTridiagonalLuConsistent,
    LapackStyleBandedLuFaithful, residual_linf, vec_diff_linf,
};

fn solve_with_lapack_faithful(a: &Banded<f64>, b: &[f64]) -> Vec<f64> {
    let mut solver = LapackStyleBandedLuFaithful::new(a.n(), a.kl(), a.ku())
        .expect("faithful LAPACK-style workspace should be constructible");
    solver
        .factor_from(a)
        .expect("faithful LAPACK-style banded LU factorization should succeed");
    let mut x = b.to_vec();
    solver
        .solve_in_place(&mut x)
        .expect("faithful LAPACK-style banded solve should succeed");
    x
}

fn solve_with_block_legacy(a: &BlockTridiagonal, b: &[f64]) -> Vec<f64> {
    let mut solver = BlockTridiagonalLu::new(a.n_blocks(), a.block_size())
        .expect("block-tridiagonal legacy solver workspace should be constructible");
    solver
        .factor_from(a)
        .expect("legacy block-tridiagonal factorization should succeed");
    let mut x = b.to_vec();
    solver
        .solve_in_place(&mut x)
        .expect("legacy block-tridiagonal solve should succeed");
    x
}

fn solve_with_block_consistent(a: &BlockTridiagonal, b: &[f64]) -> Vec<f64> {
    let mut solver = BlockTridiagonalLuConsistent::new(a.n_blocks(), a.block_size())
        .expect("consistent block-tridiagonal solver workspace should be constructible");
    solver
        .factor_from(a)
        .expect("consistent block-tridiagonal factorization should succeed");
    let mut x = b.to_vec();
    solver
        .solve_in_place(&mut x)
        .expect("consistent block-tridiagonal solve should succeed");
    x
}

fn print_row(label: &str, x: &[f64], x_true: &[f64], residual: f64) {
    let x_err = vec_diff_linf(x, x_true);
    println!("{label:<36} | x_err_linf = {x_err:>12.3e} | residual_linf = {residual:>12.3e}");
}

fn main() {
    println!("Banded solvers guide");
    println!("====================");
    println!("Recommended default:");
    println!("  - LapackStyleBandedLuFaithful: faithful LAPACK-style compact banded LU");
    println!("Specialized opt-in solvers:");
    println!("  - BlockTridiagonalLu: fast legacy solver for clean block-tridiagonal systems");
    println!("  - BlockTridiagonalLuConsistent: newer structured block-tridiagonal solver");
    println!();

    // Use a diagonally dominant random block-tridiagonal system so the default
    // banded solver and both structured solvers can be demonstrated together.
    let n_blocks = 6usize;
    let block_size = 3usize;
    let a_block = generate_block_tridiagonal_dense(n_blocks, block_size, 20260418);
    let a_banded = block_to_banded(&a_block);
    let (x_true, b) = generate_rhs_from_known_solution_block(&a_block, 20260419);

    let x_faithful = solve_with_lapack_faithful(&a_banded, &b);
    let x_legacy = solve_with_block_legacy(&a_block, &b);
    let x_consistent = solve_with_block_consistent(&a_block, &b);

    let r_faithful = residual_linf(&a_banded, &x_faithful, &b)
        .expect("faithful banded residual should be computable");
    let r_legacy = vec_diff_linf(
        &RustedSciThe::somelinalg::banded::block_tridiag_bench_helpers::block_tridiagonal_matvec(
            &a_block, &x_legacy,
        ),
        &b,
    );
    let r_consistent = vec_diff_linf(
        &RustedSciThe::somelinalg::banded::block_tridiag_bench_helpers::block_tridiagonal_matvec(
            &a_block,
            &x_consistent,
        ),
        &b,
    );

    println!(
        "Demo system: n_blocks = {}, block_size = {}, global_n = {}",
        n_blocks,
        block_size,
        a_block.n()
    );
    println!();
    println!("Solver results");
    println!("--------------");
    print_row(
        "LapackStyleBandedLuFaithful",
        &x_faithful,
        &x_true,
        r_faithful,
    );
    print_row("BlockTridiagonalLu", &x_legacy, &x_true, r_legacy);
    print_row(
        "BlockTridiagonalLuConsistent",
        &x_consistent,
        &x_true,
        r_consistent,
    );
    println!();
    println!("How to choose");
    println!("-------------");
    println!("  - General banded matrix: use LapackStyleBandedLuFaithful.");
    println!("  - Known clean block-tridiagonal matrix: try BlockTridiagonalLu for speed.");
    println!("  - Difficult block-tridiagonal chain: try BlockTridiagonalLuConsistent explicitly.");
    println!(
        "  - Need refinement experiments: opt in explicitly; production banded defaults use refine = 0."
    );
}
