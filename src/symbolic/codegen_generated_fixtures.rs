//! Checked-in generated Rust fixtures for `CodegenIR` tests.
//!
//! These functions intentionally look like emitted `RustEmitter` output.
//! They let us verify an important intermediate milestone:
//! `Expr -> emitted Rust source -> compiled Rust function -> numerical result`.
//!
//! This module is test-only for now. Later, the same pattern can be upgraded
//! into real file-based code generation for large residual and Jacobian blocks.

/// Fixture scalar function resembling `RustEmitter::emit_function(...)`.
pub fn fixture_scalar_eval(args: &[f64]) -> f64 {
    debug_assert!(args.len() >= 2, "expected at least 2 arguments");
    let t0 = args[0];
    let t1 = args[1];
    let t2 = 2.00000000000000000e0_f64;
    let t3 = t0 + t1;
    let t4 = t3.powf(t2);
    let t5 = t0.sin();
    let t6 = t4 + t5;
    let t7 = 3.00000000000000000e0_f64;
    let t8 = t7 * t1;
    let t9 = t6 - t8;
    t9
}

/// Fixture block function resembling `RustEmitter::emit_block_function(...)`.
pub fn fixture_block_eval(args: &[f64], out: &mut [f64]) {
    debug_assert!(args.len() >= 2, "expected at least 2 arguments");
    debug_assert!(out.len() >= 3, "expected at least 3 output slots");
    let t0 = args[0];
    let t1 = args[1];
    let t2 = 1.00000000000000000e0_f64;
    let t3 = t0 + t2;
    let t4 = t0 * t1;
    let t5 = t0 - t1;
    let t6 = t5.cos();
    out[0] = t3;
    out[1] = t4;
    out[2] = t6;
}
