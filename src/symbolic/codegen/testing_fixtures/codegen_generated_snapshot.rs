// =========================================
// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// =========================================

#![allow(clippy::all)]
#![allow(non_snake_case)]
#![allow(unused_parens)]

pub mod generated_fixture_snapshot {
    pub fn fixture_scalar_eval(args: &[f64]) -> f64 {
        debug_assert!(args.len() >= 2, "expected at least 2 arguments");
        let t0 = args[0];
        let t1 = args[1];
        let t2 = t0 + t1;
        let t3 = 2.00000000000000000e0_f64;
        let t4 = t2.powf(t3);
        let t5 = t0.sin();
        let t6 = t4 + t5;
        let t7 = 3.00000000000000000e0_f64;
        let t8 = t7 * t1;
        let t9 = t6 - t8;
        t9
    }

    pub fn fixture_block_eval(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 2, "expected at least 2 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = args[0];
        let t1 = 1.00000000000000000e0_f64;
        let t2 = t0 + t1;
        let t3 = args[1];
        let t4 = t0 * t3;
        let t5 = t0 - t3;
        let t6 = t5.cos();
        out[0] = t2;
        out[1] = t4;
        out[2] = t6;
    }

}
