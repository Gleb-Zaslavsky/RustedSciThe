#![cfg(test)]

//! Checked-in generated fixtures for parameterized dense algebraic systems.
//!
//! Argument order follows the current AOT contract:
//! `params` first, then `variables`.

pub mod generated_parameterized_dense_fixture {
    pub fn fixture_parameterized_residual_chunk_split_0(args: &[f64], out: &mut [f64]) {
        let a = args[0];
        let b = args[1];
        let x = args[3];
        let y = args[4];

        out[0] = a * x + b * y;
    }

    pub fn fixture_parameterized_residual_chunk_split_1(args: &[f64], out: &mut [f64]) {
        let c = args[2];
        let x = args[3];
        let y = args[4];

        out[0] = c * x * y;
    }

    pub fn fixture_parameterized_residual_chunk_0(args: &[f64], out: &mut [f64]) {
        let a = args[0];
        let b = args[1];
        let c = args[2];
        let x = args[3];
        let y = args[4];

        out[0] = a * x + b * y;
        out[1] = c * x * y;
    }

    pub fn fixture_parameterized_jacobian_chunk_0(args: &[f64], out: &mut [f64]) {
        let a = args[0];
        let c = args[2];
        let x = args[3];
        let y = args[4];

        out[0] = a;
        out[1] = args[1];
        out[2] = c * y;
        out[3] = c * x;
    }

    pub fn fixture_parameterized_jacobian_chunk_split_0(args: &[f64], out: &mut [f64]) {
        let a = args[0];
        let b = args[1];

        out[0] = a;
        out[1] = b;
    }

    pub fn fixture_parameterized_jacobian_chunk_split_1(args: &[f64], out: &mut [f64]) {
        let c = args[2];
        let x = args[3];
        let y = args[4];

        out[0] = c * y;
        out[1] = c * x;
    }
}
