#![cfg(test)]

//! Checked-in generated dense AOT fixtures.
//!
//! This module mirrors the current dense algebraic reference case used by
//! `codegen_dense_performance_tests`:
//! - residual vector of length 3
//! - dense Jacobian of shape 3 x 2
//! - flattened argument order `params, variables = [a, b, x, y]`
//!
//! The functions below are intentionally plain Rust and represent the shape of
//! source that the AOT codegen pipeline is expected to emit and orchestrate.

pub mod generated_dense_fixture {
    pub fn fixture_dense_residual_chunk_0(args: &[f64], out: &mut [f64]) {
        let a = args[0];
        let b = args[1];
        let x = args[2];
        let y = args[3];

        out[0] = x * x + y + a;
        out[1] = x - y * y + b;
    }

    pub fn fixture_dense_residual_chunk_1(args: &[f64], out: &mut [f64]) {
        let a = args[0];
        let b = args[1];
        let x = args[2];
        let y = args[3];

        out[0] = x * y + a * b;
    }

    pub fn fixture_dense_jacobian_chunk_0(args: &[f64], out: &mut [f64]) {
        let x = args[2];
        let y = args[3];

        out[0] = 2.0 * x;
        out[1] = 1.0;
        out[2] = 1.0;
        out[3] = -2.0 * y;
    }

    pub fn fixture_dense_jacobian_chunk_1(args: &[f64], out: &mut [f64]) {
        let x = args[2];
        let y = args[3];

        out[0] = y;
        out[1] = x;
    }
}
