#![cfg(test)]

//! Checked-in generated fixtures for the parameterized IVP AOT slice.
//!
//! Example system:
//! - `f0(t, y, z, a, b, c) = a*t + y + b*z`
//! - `f1(t, y, z, a, b, c) = c*y - z + b*t`
//!
//! Jacobian is taken only with respect to `[y, z]`.
//!
//! Flattened argument order:
//! - `t`
//! - `a`
//! - `b`
//! - `c`
//! - `y`
//! - `z`

pub mod generated_parameterized_ivp_fixture {
    pub fn fixture_parameterized_ivp_residual_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 6);
        debug_assert!(out.len() >= 2);

        let t = args[0];
        let a = args[1];
        let b = args[2];
        let c = args[3];
        let y = args[4];
        let z = args[5];

        let _ = c;
        out[0] = a * t + y + b * z;
        out[1] = c * y - z + b * t;
    }

    pub fn fixture_parameterized_ivp_residual_chunk_split_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 6);
        debug_assert!(out.len() >= 1);

        let t = args[0];
        let a = args[1];
        let b = args[2];
        let y = args[4];
        let z = args[5];

        out[0] = a * t + y + b * z;
    }

    pub fn fixture_parameterized_ivp_residual_chunk_split_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 6);
        debug_assert!(out.len() >= 1);

        let t = args[0];
        let b = args[2];
        let c = args[3];
        let y = args[4];
        let z = args[5];

        out[0] = c * y - z + b * t;
    }

    pub fn fixture_parameterized_ivp_jacobian_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 6);
        debug_assert!(out.len() >= 4);

        let b = args[2];
        let c = args[3];

        // Row-major Jacobian wrt [y, z]:
        // [1,  b]
        // [c, -1]
        out[0] = 1.0;
        out[1] = b;
        out[2] = c;
        out[3] = -1.0;
    }
}
