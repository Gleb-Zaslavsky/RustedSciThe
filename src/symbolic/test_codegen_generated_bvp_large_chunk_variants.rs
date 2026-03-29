//! Checked-in generated Rust fixtures for several large BVP chunk-size variants.
//!
//! These functions are emitted by the real AOT pipeline for the same
//! `BVP_Damp1`-style case with `n_steps = 32`, but with multiple residual
//! and sparse Jacobian chunk-size choices so hot-path runtime can be
//! compared on compiled code instead of estimated from planning alone.

// =========================================
// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// =========================================

#![allow(clippy::all)]
#![allow(non_snake_case)]
#![allow(unused_parens)]

pub mod generated_bvp_large_chunk_variants {
    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = args[1];
        let t1 = args[0];
        let t2 = 1.00000000000000000e0_f64;
        let t3 = t1 - t2;
        let t4 = t0 - t3;
        let t5 = t4 - t2;
        let t6 = -1.00000000000000000e0_f64;
        let t7 = t6 * t1;
        let t8 = args[2];
        let t9 = t7 + t8;
        let t10 = t9 - t2;
        let t11 = t6 * t0;
        let t12 = args[3];
        let t13 = t11 + t12;
        let t14 = t8 - t0;
        let t15 = t13 - t14;
        let t16 = 3.00000000000000000e0_f64;
        let t17 = t0.powf(t16);
        let t18 = t6 * t17;
        let t19 = t18 - t8;
        let t20 = args[4];
        let t21 = t19 + t20;
        out[0] = t5;
        out[1] = t10;
        out[2] = t15;
        out[3] = t21;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[3];
        let t2 = t0 * t1;
        let t3 = args[5];
        let t4 = t2 + t3;
        let t5 = args[4];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[6];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[7];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[8];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[7];
        let t2 = t0 * t1;
        let t3 = args[9];
        let t4 = t2 + t3;
        let t5 = args[8];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[10];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[11];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[12];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[11];
        let t2 = t0 * t1;
        let t3 = args[13];
        let t4 = t2 + t3;
        let t5 = args[12];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[14];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[15];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[16];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_4(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[15];
        let t2 = t0 * t1;
        let t3 = args[17];
        let t4 = t2 + t3;
        let t5 = args[16];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[18];
        let t13 = t11 + t12;
        let t14 = args[19];
        let t15 = t14 - t3;
        let t16 = t12 - t3;
        let t17 = t15 - t16;
        let t18 = t3.powf(t8);
        let t19 = t0 * t18;
        let t20 = args[20];
        let t21 = t19 + t20;
        let t22 = t21 - t12;
        out[0] = t7;
        out[1] = t13;
        out[2] = t17;
        out[3] = t22;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_5(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[19];
        let t2 = t0 * t1;
        let t3 = args[21];
        let t4 = t2 + t3;
        let t5 = args[20];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[22];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[23];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[24];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_6(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[23];
        let t2 = t0 * t1;
        let t3 = args[25];
        let t4 = t2 + t3;
        let t5 = args[24];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[26];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[27];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[28];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_7(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[27];
        let t2 = t0 * t1;
        let t3 = args[29];
        let t4 = t2 + t3;
        let t5 = args[28];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[30];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[31];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[32];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_8(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[31];
        let t2 = t0 * t1;
        let t3 = args[33];
        let t4 = t2 + t3;
        let t5 = args[32];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[34];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[35];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[36];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_9(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[35];
        let t2 = t0 * t1;
        let t3 = args[37];
        let t4 = t2 + t3;
        let t5 = args[36];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[38];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[39];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[40];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_10(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[39];
        let t2 = t0 * t1;
        let t3 = args[41];
        let t4 = t2 + t3;
        let t5 = args[40];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[42];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[43];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[44];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_11(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[43];
        let t2 = t0 * t1;
        let t3 = args[45];
        let t4 = t2 + t3;
        let t5 = args[44];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[46];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[47];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[48];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_12(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[47];
        let t2 = t0 * t1;
        let t3 = args[49];
        let t4 = t2 + t3;
        let t5 = args[48];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[50];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[51];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[52];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_13(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[51];
        let t2 = t0 * t1;
        let t3 = args[53];
        let t4 = t2 + t3;
        let t5 = args[52];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[54];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[55];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[56];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_14(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[55];
        let t2 = t0 * t1;
        let t3 = args[57];
        let t4 = t2 + t3;
        let t5 = args[56];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[58];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[59];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[60];
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 4 outputs
    pub fn fixture_bvp32_residual_o4_chunk_15(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 4, "expected at least 4 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[59];
        let t2 = t0 * t1;
        let t3 = args[61];
        let t4 = t2 + t3;
        let t5 = args[60];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[62];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[63];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = 1.00000000000000000e0_f64;
        let t23 = t21 + t22;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = args[1];
        let t1 = args[0];
        let t2 = 1.00000000000000000e0_f64;
        let t3 = t1 - t2;
        let t4 = t0 - t3;
        let t5 = t4 - t2;
        let t6 = -1.00000000000000000e0_f64;
        let t7 = t6 * t1;
        let t8 = args[2];
        let t9 = t7 + t8;
        let t10 = t9 - t2;
        let t11 = t6 * t0;
        let t12 = args[3];
        let t13 = t11 + t12;
        let t14 = t8 - t0;
        let t15 = t13 - t14;
        let t16 = 3.00000000000000000e0_f64;
        let t17 = t0.powf(t16);
        let t18 = t6 * t17;
        let t19 = t18 - t8;
        let t20 = args[4];
        let t21 = t19 + t20;
        let t22 = t6 * t12;
        let t23 = args[5];
        let t24 = t22 + t23;
        let t25 = t20 - t12;
        let t26 = t24 - t25;
        let t27 = t12.powf(t16);
        let t28 = t6 * t27;
        let t29 = t28 - t20;
        let t30 = args[6];
        let t31 = t29 + t30;
        let t32 = t6 * t23;
        let t33 = args[7];
        let t34 = t32 + t33;
        let t35 = t30 - t23;
        let t36 = t34 - t35;
        let t37 = t23.powf(t16);
        let t38 = t6 * t37;
        let t39 = t38 - t30;
        let t40 = args[8];
        let t41 = t39 + t40;
        out[0] = t5;
        out[1] = t10;
        out[2] = t15;
        out[3] = t21;
        out[4] = t26;
        out[5] = t31;
        out[6] = t36;
        out[7] = t41;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[7];
        let t2 = t0 * t1;
        let t3 = args[9];
        let t4 = t2 + t3;
        let t5 = args[8];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[10];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[11];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[12];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[13];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[14];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[15];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = args[16];
        let t43 = t41 + t42;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[15];
        let t2 = t0 * t1;
        let t3 = args[17];
        let t4 = t2 + t3;
        let t5 = args[16];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[18];
        let t13 = t11 + t12;
        let t14 = args[19];
        let t15 = t14 - t3;
        let t16 = t12 - t3;
        let t17 = t15 - t16;
        let t18 = t3.powf(t8);
        let t19 = t0 * t18;
        let t20 = args[20];
        let t21 = t19 + t20;
        let t22 = t21 - t12;
        let t23 = t0 * t14;
        let t24 = args[21];
        let t25 = t23 + t24;
        let t26 = t20 - t14;
        let t27 = t25 - t26;
        let t28 = t14.powf(t8);
        let t29 = t0 * t28;
        let t30 = t29 - t20;
        let t31 = args[22];
        let t32 = t30 + t31;
        let t33 = t0 * t24;
        let t34 = args[23];
        let t35 = t33 + t34;
        let t36 = t31 - t24;
        let t37 = t35 - t36;
        let t38 = t24.powf(t8);
        let t39 = t0 * t38;
        let t40 = t39 - t31;
        let t41 = args[24];
        let t42 = t40 + t41;
        out[0] = t7;
        out[1] = t13;
        out[2] = t17;
        out[3] = t22;
        out[4] = t27;
        out[5] = t32;
        out[6] = t37;
        out[7] = t42;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[23];
        let t2 = t0 * t1;
        let t3 = args[25];
        let t4 = t2 + t3;
        let t5 = args[24];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[26];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[27];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[28];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[29];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[30];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[31];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = args[32];
        let t43 = t41 + t42;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_4(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[31];
        let t2 = t0 * t1;
        let t3 = args[33];
        let t4 = t2 + t3;
        let t5 = args[32];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[34];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[35];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[36];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[37];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[38];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[39];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = args[40];
        let t43 = t41 + t42;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_5(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[39];
        let t2 = t0 * t1;
        let t3 = args[41];
        let t4 = t2 + t3;
        let t5 = args[40];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[42];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[43];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[44];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[45];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[46];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[47];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = args[48];
        let t43 = t41 + t42;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_6(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[47];
        let t2 = t0 * t1;
        let t3 = args[49];
        let t4 = t2 + t3;
        let t5 = args[48];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[50];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[51];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[52];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[53];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[54];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[55];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = args[56];
        let t43 = t41 + t42;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
    }

    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_o8_chunk_7(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 8, "expected at least 8 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[55];
        let t2 = t0 * t1;
        let t3 = args[57];
        let t4 = t2 + t3;
        let t5 = args[56];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[58];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[59];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[60];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[61];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[62];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[63];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = 1.00000000000000000e0_f64;
        let t43 = t41 + t42;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
    }

    // Residual block: 16 outputs
    pub fn fixture_bvp32_residual_o16_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 16, "expected at least 16 output slots");
        let t0 = args[1];
        let t1 = args[0];
        let t2 = 1.00000000000000000e0_f64;
        let t3 = t1 - t2;
        let t4 = t0 - t3;
        let t5 = t4 - t2;
        let t6 = -1.00000000000000000e0_f64;
        let t7 = t6 * t1;
        let t8 = args[2];
        let t9 = t7 + t8;
        let t10 = t9 - t2;
        let t11 = t6 * t0;
        let t12 = args[3];
        let t13 = t11 + t12;
        let t14 = t8 - t0;
        let t15 = t13 - t14;
        let t16 = 3.00000000000000000e0_f64;
        let t17 = t0.powf(t16);
        let t18 = t6 * t17;
        let t19 = t18 - t8;
        let t20 = args[4];
        let t21 = t19 + t20;
        let t22 = t6 * t12;
        let t23 = args[5];
        let t24 = t22 + t23;
        let t25 = t20 - t12;
        let t26 = t24 - t25;
        let t27 = t12.powf(t16);
        let t28 = t6 * t27;
        let t29 = t28 - t20;
        let t30 = args[6];
        let t31 = t29 + t30;
        let t32 = t6 * t23;
        let t33 = args[7];
        let t34 = t32 + t33;
        let t35 = t30 - t23;
        let t36 = t34 - t35;
        let t37 = t23.powf(t16);
        let t38 = t6 * t37;
        let t39 = t38 - t30;
        let t40 = args[8];
        let t41 = t39 + t40;
        let t42 = t6 * t33;
        let t43 = args[9];
        let t44 = t42 + t43;
        let t45 = t40 - t33;
        let t46 = t44 - t45;
        let t47 = t33.powf(t16);
        let t48 = t6 * t47;
        let t49 = t48 - t40;
        let t50 = args[10];
        let t51 = t49 + t50;
        let t52 = t6 * t43;
        let t53 = args[11];
        let t54 = t52 + t53;
        let t55 = t50 - t43;
        let t56 = t54 - t55;
        let t57 = t43.powf(t16);
        let t58 = t6 * t57;
        let t59 = t58 - t50;
        let t60 = args[12];
        let t61 = t59 + t60;
        let t62 = t6 * t53;
        let t63 = args[13];
        let t64 = t62 + t63;
        let t65 = t60 - t53;
        let t66 = t64 - t65;
        let t67 = t53.powf(t16);
        let t68 = t6 * t67;
        let t69 = t68 - t60;
        let t70 = args[14];
        let t71 = t69 + t70;
        let t72 = t6 * t63;
        let t73 = args[15];
        let t74 = t72 + t73;
        let t75 = t70 - t63;
        let t76 = t74 - t75;
        let t77 = t63.powf(t16);
        let t78 = t6 * t77;
        let t79 = t78 - t70;
        let t80 = args[16];
        let t81 = t79 + t80;
        out[0] = t5;
        out[1] = t10;
        out[2] = t15;
        out[3] = t21;
        out[4] = t26;
        out[5] = t31;
        out[6] = t36;
        out[7] = t41;
        out[8] = t46;
        out[9] = t51;
        out[10] = t56;
        out[11] = t61;
        out[12] = t66;
        out[13] = t71;
        out[14] = t76;
        out[15] = t81;
    }

    // Residual block: 16 outputs
    pub fn fixture_bvp32_residual_o16_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 16, "expected at least 16 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[15];
        let t2 = t0 * t1;
        let t3 = args[17];
        let t4 = t2 + t3;
        let t5 = args[16];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[18];
        let t13 = t11 + t12;
        let t14 = args[19];
        let t15 = t14 - t3;
        let t16 = t12 - t3;
        let t17 = t15 - t16;
        let t18 = t3.powf(t8);
        let t19 = t0 * t18;
        let t20 = args[20];
        let t21 = t19 + t20;
        let t22 = t21 - t12;
        let t23 = t0 * t14;
        let t24 = args[21];
        let t25 = t23 + t24;
        let t26 = t20 - t14;
        let t27 = t25 - t26;
        let t28 = t14.powf(t8);
        let t29 = t0 * t28;
        let t30 = t29 - t20;
        let t31 = args[22];
        let t32 = t30 + t31;
        let t33 = t0 * t24;
        let t34 = args[23];
        let t35 = t33 + t34;
        let t36 = t31 - t24;
        let t37 = t35 - t36;
        let t38 = t24.powf(t8);
        let t39 = t0 * t38;
        let t40 = t39 - t31;
        let t41 = args[24];
        let t42 = t40 + t41;
        let t43 = t0 * t34;
        let t44 = args[25];
        let t45 = t43 + t44;
        let t46 = t41 - t34;
        let t47 = t45 - t46;
        let t48 = t34.powf(t8);
        let t49 = t0 * t48;
        let t50 = t49 - t41;
        let t51 = args[26];
        let t52 = t50 + t51;
        let t53 = t0 * t44;
        let t54 = args[27];
        let t55 = t53 + t54;
        let t56 = t51 - t44;
        let t57 = t55 - t56;
        let t58 = t44.powf(t8);
        let t59 = t0 * t58;
        let t60 = t59 - t51;
        let t61 = args[28];
        let t62 = t60 + t61;
        let t63 = t0 * t54;
        let t64 = args[29];
        let t65 = t63 + t64;
        let t66 = t61 - t54;
        let t67 = t65 - t66;
        let t68 = t54.powf(t8);
        let t69 = t0 * t68;
        let t70 = t69 - t61;
        let t71 = args[30];
        let t72 = t70 + t71;
        let t73 = t0 * t64;
        let t74 = args[31];
        let t75 = t73 + t74;
        let t76 = t71 - t64;
        let t77 = t75 - t76;
        let t78 = t64.powf(t8);
        let t79 = t0 * t78;
        let t80 = t79 - t71;
        let t81 = args[32];
        let t82 = t80 + t81;
        out[0] = t7;
        out[1] = t13;
        out[2] = t17;
        out[3] = t22;
        out[4] = t27;
        out[5] = t32;
        out[6] = t37;
        out[7] = t42;
        out[8] = t47;
        out[9] = t52;
        out[10] = t57;
        out[11] = t62;
        out[12] = t67;
        out[13] = t72;
        out[14] = t77;
        out[15] = t82;
    }

    // Residual block: 16 outputs
    pub fn fixture_bvp32_residual_o16_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 16, "expected at least 16 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[31];
        let t2 = t0 * t1;
        let t3 = args[33];
        let t4 = t2 + t3;
        let t5 = args[32];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[34];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[35];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[36];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[37];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[38];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[39];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = args[40];
        let t43 = t41 + t42;
        let t44 = t0 * t35;
        let t45 = args[41];
        let t46 = t44 + t45;
        let t47 = t42 - t35;
        let t48 = t46 - t47;
        let t49 = t35.powf(t8);
        let t50 = t0 * t49;
        let t51 = t50 - t42;
        let t52 = args[42];
        let t53 = t51 + t52;
        let t54 = t0 * t45;
        let t55 = args[43];
        let t56 = t54 + t55;
        let t57 = t52 - t45;
        let t58 = t56 - t57;
        let t59 = t45.powf(t8);
        let t60 = t0 * t59;
        let t61 = t60 - t52;
        let t62 = args[44];
        let t63 = t61 + t62;
        let t64 = t0 * t55;
        let t65 = args[45];
        let t66 = t64 + t65;
        let t67 = t62 - t55;
        let t68 = t66 - t67;
        let t69 = t55.powf(t8);
        let t70 = t0 * t69;
        let t71 = t70 - t62;
        let t72 = args[46];
        let t73 = t71 + t72;
        let t74 = t0 * t65;
        let t75 = args[47];
        let t76 = t74 + t75;
        let t77 = t72 - t65;
        let t78 = t76 - t77;
        let t79 = t65.powf(t8);
        let t80 = t0 * t79;
        let t81 = t80 - t72;
        let t82 = args[48];
        let t83 = t81 + t82;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
        out[8] = t48;
        out[9] = t53;
        out[10] = t58;
        out[11] = t63;
        out[12] = t68;
        out[13] = t73;
        out[14] = t78;
        out[15] = t83;
    }

    // Residual block: 16 outputs
    pub fn fixture_bvp32_residual_o16_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 16, "expected at least 16 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = args[47];
        let t2 = t0 * t1;
        let t3 = args[49];
        let t4 = t2 + t3;
        let t5 = args[48];
        let t6 = t5 - t1;
        let t7 = t4 - t6;
        let t8 = 3.00000000000000000e0_f64;
        let t9 = t1.powf(t8);
        let t10 = t0 * t9;
        let t11 = t10 - t5;
        let t12 = args[50];
        let t13 = t11 + t12;
        let t14 = t0 * t3;
        let t15 = args[51];
        let t16 = t14 + t15;
        let t17 = t12 - t3;
        let t18 = t16 - t17;
        let t19 = t3.powf(t8);
        let t20 = t0 * t19;
        let t21 = t20 - t12;
        let t22 = args[52];
        let t23 = t21 + t22;
        let t24 = t0 * t15;
        let t25 = args[53];
        let t26 = t24 + t25;
        let t27 = t22 - t15;
        let t28 = t26 - t27;
        let t29 = t15.powf(t8);
        let t30 = t0 * t29;
        let t31 = t30 - t22;
        let t32 = args[54];
        let t33 = t31 + t32;
        let t34 = t0 * t25;
        let t35 = args[55];
        let t36 = t34 + t35;
        let t37 = t32 - t25;
        let t38 = t36 - t37;
        let t39 = t25.powf(t8);
        let t40 = t0 * t39;
        let t41 = t40 - t32;
        let t42 = args[56];
        let t43 = t41 + t42;
        let t44 = t0 * t35;
        let t45 = args[57];
        let t46 = t44 + t45;
        let t47 = t42 - t35;
        let t48 = t46 - t47;
        let t49 = t35.powf(t8);
        let t50 = t0 * t49;
        let t51 = t50 - t42;
        let t52 = args[58];
        let t53 = t51 + t52;
        let t54 = t0 * t45;
        let t55 = args[59];
        let t56 = t54 + t55;
        let t57 = t52 - t45;
        let t58 = t56 - t57;
        let t59 = t45.powf(t8);
        let t60 = t0 * t59;
        let t61 = t60 - t52;
        let t62 = args[60];
        let t63 = t61 + t62;
        let t64 = t0 * t55;
        let t65 = args[61];
        let t66 = t64 + t65;
        let t67 = t62 - t55;
        let t68 = t66 - t67;
        let t69 = t55.powf(t8);
        let t70 = t0 * t69;
        let t71 = t70 - t62;
        let t72 = args[62];
        let t73 = t71 + t72;
        let t74 = t0 * t65;
        let t75 = args[63];
        let t76 = t74 + t75;
        let t77 = t72 - t65;
        let t78 = t76 - t77;
        let t79 = t65.powf(t8);
        let t80 = t0 * t79;
        let t81 = t80 - t72;
        let t82 = 1.00000000000000000e0_f64;
        let t83 = t81 + t82;
        out[0] = t7;
        out[1] = t13;
        out[2] = t18;
        out[3] = t23;
        out[4] = t28;
        out[5] = t33;
        out[6] = t38;
        out[7] = t43;
        out[8] = t48;
        out[9] = t53;
        out[10] = t58;
        out[11] = t63;
        out[12] = t68;
        out[13] = t73;
        out[14] = t78;
        out[15] = t83;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 9 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 9, "expected at least 9 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[1];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        out[0] = t0;
        out[1] = t1;
        out[2] = t0;
        out[3] = t1;
        out[4] = t0;
        out[5] = t1;
        out[6] = t6;
        out[7] = t0;
        out[8] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[3];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[5];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[7];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[9];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[11];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[13];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_4(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[15];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[17];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_5(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[19];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[21];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_6(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[23];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[25];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_7(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[27];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[29];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_8(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[31];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[33];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_9(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[35];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[37];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_10(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[39];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[41];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_11(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[43];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[45];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_12(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[47];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[49];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_13(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[51];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[53];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 10 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_14(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 10, "expected at least 10 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[55];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[57];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 9 non-zero values
    pub fn fixture_bvp32_sparse_r4_chunk_15(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 9, "expected at least 9 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[59];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[61];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 19 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 19, "expected at least 19 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[1];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[3];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[5];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        out[0] = t0;
        out[1] = t1;
        out[2] = t0;
        out[3] = t1;
        out[4] = t0;
        out[5] = t1;
        out[6] = t6;
        out[7] = t0;
        out[8] = t1;
        out[9] = t0;
        out[10] = t1;
        out[11] = t9;
        out[12] = t0;
        out[13] = t1;
        out[14] = t0;
        out[15] = t1;
        out[16] = t12;
        out[17] = t0;
        out[18] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 20 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 20, "expected at least 20 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[7];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[9];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[11];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[13];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 20 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 20, "expected at least 20 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[15];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[17];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[19];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[21];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 20 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 20, "expected at least 20 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[23];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[25];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[27];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[29];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 20 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_4(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 20, "expected at least 20 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[31];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[33];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[35];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[37];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 20 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_5(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 20, "expected at least 20 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[39];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[41];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[43];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[45];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 20 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_6(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 20, "expected at least 20 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[47];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[49];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[51];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[53];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 19 non-zero values
    pub fn fixture_bvp32_sparse_r8_chunk_7(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 19, "expected at least 19 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[55];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[57];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[59];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[61];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 39 non-zero values
    pub fn fixture_bvp32_sparse_r16_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 39, "expected at least 39 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[1];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[3];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[5];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[7];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        let t16 = args[9];
        let t17 = t16.powf(t4);
        let t18 = t2 * t17;
        let t19 = args[11];
        let t20 = t19.powf(t4);
        let t21 = t2 * t20;
        let t22 = args[13];
        let t23 = t22.powf(t4);
        let t24 = t2 * t23;
        out[0] = t0;
        out[1] = t1;
        out[2] = t0;
        out[3] = t1;
        out[4] = t0;
        out[5] = t1;
        out[6] = t6;
        out[7] = t0;
        out[8] = t1;
        out[9] = t0;
        out[10] = t1;
        out[11] = t9;
        out[12] = t0;
        out[13] = t1;
        out[14] = t0;
        out[15] = t1;
        out[16] = t12;
        out[17] = t0;
        out[18] = t1;
        out[19] = t0;
        out[20] = t1;
        out[21] = t15;
        out[22] = t0;
        out[23] = t1;
        out[24] = t0;
        out[25] = t1;
        out[26] = t18;
        out[27] = t0;
        out[28] = t1;
        out[29] = t0;
        out[30] = t1;
        out[31] = t21;
        out[32] = t0;
        out[33] = t1;
        out[34] = t0;
        out[35] = t1;
        out[36] = t24;
        out[37] = t0;
        out[38] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 40 non-zero values
    pub fn fixture_bvp32_sparse_r16_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 40, "expected at least 40 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[15];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[17];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[19];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[21];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        let t16 = args[23];
        let t17 = t16.powf(t4);
        let t18 = t2 * t17;
        let t19 = args[25];
        let t20 = t19.powf(t4);
        let t21 = t2 * t20;
        let t22 = args[27];
        let t23 = t22.powf(t4);
        let t24 = t2 * t23;
        let t25 = args[29];
        let t26 = t25.powf(t4);
        let t27 = t2 * t26;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
        out[20] = t0;
        out[21] = t1;
        out[22] = t18;
        out[23] = t0;
        out[24] = t1;
        out[25] = t0;
        out[26] = t1;
        out[27] = t21;
        out[28] = t0;
        out[29] = t1;
        out[30] = t0;
        out[31] = t1;
        out[32] = t24;
        out[33] = t0;
        out[34] = t1;
        out[35] = t0;
        out[36] = t1;
        out[37] = t27;
        out[38] = t0;
        out[39] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 40 non-zero values
    pub fn fixture_bvp32_sparse_r16_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 40, "expected at least 40 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[31];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[33];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[35];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[37];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        let t16 = args[39];
        let t17 = t16.powf(t4);
        let t18 = t2 * t17;
        let t19 = args[41];
        let t20 = t19.powf(t4);
        let t21 = t2 * t20;
        let t22 = args[43];
        let t23 = t22.powf(t4);
        let t24 = t2 * t23;
        let t25 = args[45];
        let t26 = t25.powf(t4);
        let t27 = t2 * t26;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
        out[20] = t0;
        out[21] = t1;
        out[22] = t18;
        out[23] = t0;
        out[24] = t1;
        out[25] = t0;
        out[26] = t1;
        out[27] = t21;
        out[28] = t0;
        out[29] = t1;
        out[30] = t0;
        out[31] = t1;
        out[32] = t24;
        out[33] = t0;
        out[34] = t1;
        out[35] = t0;
        out[36] = t1;
        out[37] = t27;
        out[38] = t0;
        out[39] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 39 non-zero values
    pub fn fixture_bvp32_sparse_r16_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 39, "expected at least 39 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        let t2 = -3.00000000000000000e0_f64;
        let t3 = args[47];
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t2 * t5;
        let t7 = args[49];
        let t8 = t7.powf(t4);
        let t9 = t2 * t8;
        let t10 = args[51];
        let t11 = t10.powf(t4);
        let t12 = t2 * t11;
        let t13 = args[53];
        let t14 = t13.powf(t4);
        let t15 = t2 * t14;
        let t16 = args[55];
        let t17 = t16.powf(t4);
        let t18 = t2 * t17;
        let t19 = args[57];
        let t20 = t19.powf(t4);
        let t21 = t2 * t20;
        let t22 = args[59];
        let t23 = t22.powf(t4);
        let t24 = t2 * t23;
        let t25 = args[61];
        let t26 = t25.powf(t4);
        let t27 = t2 * t26;
        out[0] = t0;
        out[1] = t1;
        out[2] = t6;
        out[3] = t0;
        out[4] = t1;
        out[5] = t0;
        out[6] = t1;
        out[7] = t9;
        out[8] = t0;
        out[9] = t1;
        out[10] = t0;
        out[11] = t1;
        out[12] = t12;
        out[13] = t0;
        out[14] = t1;
        out[15] = t0;
        out[16] = t1;
        out[17] = t15;
        out[18] = t0;
        out[19] = t1;
        out[20] = t0;
        out[21] = t1;
        out[22] = t18;
        out[23] = t0;
        out[24] = t1;
        out[25] = t0;
        out[26] = t1;
        out[27] = t21;
        out[28] = t0;
        out[29] = t1;
        out[30] = t0;
        out[31] = t1;
        out[32] = t24;
        out[33] = t0;
        out[34] = t1;
        out[35] = t0;
        out[36] = t1;
        out[37] = t27;
        out[38] = t0;
    }
}

pub mod generated_bvp_large_chunk_variant_bindings {
    use super::generated_bvp_large_chunk_variants;

    pub const N_STEPS: usize = 32;

    pub const RESIDUAL_O4: [fn(&[f64], &mut [f64]); 16] = [
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_0,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_1,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_2,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_3,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_4,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_5,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_6,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_7,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_8,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_9,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_10,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_11,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_12,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_13,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_14,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o4_chunk_15,
    ];

    pub const RESIDUAL_O8: [fn(&[f64], &mut [f64]); 8] = [
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_0,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_1,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_2,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_3,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_4,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_5,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_6,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o8_chunk_7,
    ];

    pub const RESIDUAL_O16: [fn(&[f64], &mut [f64]); 4] = [
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o16_chunk_0,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o16_chunk_1,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o16_chunk_2,
        generated_bvp_large_chunk_variants::fixture_bvp32_residual_o16_chunk_3,
    ];

    pub const SPARSE_R4: [fn(&[f64], &mut [f64]); 16] = [
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_0,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_1,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_2,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_3,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_4,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_5,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_6,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_7,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_8,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_9,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_10,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_11,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_12,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_13,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_14,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r4_chunk_15,
    ];

    pub const SPARSE_R8: [fn(&[f64], &mut [f64]); 8] = [
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_0,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_1,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_2,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_3,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_4,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_5,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_6,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r8_chunk_7,
    ];

    pub const SPARSE_R16: [fn(&[f64], &mut [f64]); 4] = [
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r16_chunk_0,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r16_chunk_1,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r16_chunk_2,
        generated_bvp_large_chunk_variants::fixture_bvp32_sparse_r16_chunk_3,
    ];
}
