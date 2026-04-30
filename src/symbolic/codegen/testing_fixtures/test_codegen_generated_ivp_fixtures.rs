#![cfg(test)]

//! Checked-in generated fixtures for the first IVP AOT vertical slice.
//!
//! These functions represent generated AOT chunks for a small dense IVP
//! example:
//! - `f0(t, y, z) = t + y`
//! - `f1(t, y, z) = t*y - z`
//!
//! The flattened argument order is:
//! - `t`
//! - `y`
//! - `z`

pub mod generated_ivp_fixture {
    pub fn fixture_ivp_residual_chunk_split_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 3);
        debug_assert!(out.len() >= 1);

        let t = args[0];
        let y = args[1];

        out[0] = t + y;
    }

    pub fn fixture_ivp_residual_chunk_split_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 3);
        debug_assert!(out.len() >= 1);

        let t = args[0];
        let y = args[1];
        let z = args[2];

        out[0] = t * y - z;
    }

    pub fn fixture_ivp_residual_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 3);
        debug_assert!(out.len() >= 2);

        let t = args[0];
        let y = args[1];
        let z = args[2];

        out[0] = t + y;
        out[1] = t * y - z;
    }

    pub fn fixture_ivp_jacobian_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 3);
        debug_assert!(out.len() >= 4);

        let t = args[0];

        // Row-major Jacobian wrt [y, z]:
        // [1,  0]
        // [t, -1]
        out[0] = 1.0;
        out[1] = 0.0;
        out[2] = t;
        out[3] = -1.0;
    }
}
