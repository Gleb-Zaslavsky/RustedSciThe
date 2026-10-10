// =========================================
// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// =========================================

#![allow(clippy::all)]
#![allow(non_snake_case)]
#![allow(unused_parens)]

pub mod generated_bvp_large_fixture {
    // Residual block: 8 outputs
    pub fn fixture_bvp32_residual_chunk_0(args: &[f64], out: &mut [f64]) {
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
        let t10 = t9 + t2;
        let t11 = t6 * t0;
        let t12 = args[3];
        let t13 = t11 + t12;
        let t14 = t8 - t0;
        let t15 = t13 - t14;
        let t16 = t6 * t8;
        let t17 = args[4];
        let t18 = t16 + t17;
        let t19 = 3.00000000000000000e0_f64;
        let t20 = t11.powf(t19);
        let t21 = t18 - t20;
        let t22 = t6 * t12;
        let t23 = args[5];
        let t24 = t22 + t23;
        let t25 = t17 - t12;
        let t26 = t24 - t25;
        let t27 = t6 * t17;
        let t28 = args[6];
        let t29 = t27 + t28;
        let t30 = t22.powf(t19);
        let t31 = t29 - t30;
        let t32 = t6 * t23;
        let t33 = args[7];
        let t34 = t32 + t33;
        let t35 = t28 - t23;
        let t36 = t34 - t35;
        let t37 = t6 * t28;
        let t38 = args[8];
        let t39 = t37 + t38;
        let t40 = t32.powf(t19);
        let t41 = t39 - t40;
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
    pub fn fixture_bvp32_residual_chunk_1(args: &[f64], out: &mut [f64]) {
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
        let t8 = t0 * t5;
        let t9 = args[10];
        let t10 = t8 + t9;
        let t11 = 3.00000000000000000e0_f64;
        let t12 = t2.powf(t11);
        let t13 = t10 - t12;
        let t14 = t0 * t3;
        let t15 = args[11];
        let t16 = t14 + t15;
        let t17 = t9 - t3;
        let t18 = t16 - t17;
        let t19 = t0 * t9;
        let t20 = args[12];
        let t21 = t19 + t20;
        let t22 = t14.powf(t11);
        let t23 = t21 - t22;
        let t24 = t0 * t15;
        let t25 = args[13];
        let t26 = t24 + t25;
        let t27 = t20 - t15;
        let t28 = t26 - t27;
        let t29 = t0 * t20;
        let t30 = args[14];
        let t31 = t29 + t30;
        let t32 = t24.powf(t11);
        let t33 = t31 - t32;
        let t34 = t0 * t25;
        let t35 = args[15];
        let t36 = t34 + t35;
        let t37 = t30 - t25;
        let t38 = t36 - t37;
        let t39 = t0 * t30;
        let t40 = args[16];
        let t41 = t39 + t40;
        let t42 = t34.powf(t11);
        let t43 = t41 - t42;
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
    pub fn fixture_bvp32_residual_chunk_2(args: &[f64], out: &mut [f64]) {
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
        let t8 = t0 * t5;
        let t9 = args[18];
        let t10 = t8 + t9;
        let t11 = 3.00000000000000000e0_f64;
        let t12 = t2.powf(t11);
        let t13 = t10 - t12;
        let t14 = args[19];
        let t15 = t14 - t3;
        let t16 = t9 - t3;
        let t17 = t15 - t16;
        let t18 = args[20];
        let t19 = t18 - t9;
        let t20 = t0 * t3;
        let t21 = t20.powf(t11);
        let t22 = t19 - t21;
        let t23 = t0 * t14;
        let t24 = args[21];
        let t25 = t23 + t24;
        let t26 = t18 - t14;
        let t27 = t25 - t26;
        let t28 = t0 * t18;
        let t29 = args[22];
        let t30 = t28 + t29;
        let t31 = t23.powf(t11);
        let t32 = t30 - t31;
        let t33 = t0 * t24;
        let t34 = args[23];
        let t35 = t33 + t34;
        let t36 = t29 - t24;
        let t37 = t35 - t36;
        let t38 = t0 * t29;
        let t39 = args[24];
        let t40 = t38 + t39;
        let t41 = t33.powf(t11);
        let t42 = t40 - t41;
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
    pub fn fixture_bvp32_residual_chunk_3(args: &[f64], out: &mut [f64]) {
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
        let t8 = t0 * t5;
        let t9 = args[26];
        let t10 = t8 + t9;
        let t11 = 3.00000000000000000e0_f64;
        let t12 = t2.powf(t11);
        let t13 = t10 - t12;
        let t14 = t0 * t3;
        let t15 = args[27];
        let t16 = t14 + t15;
        let t17 = t9 - t3;
        let t18 = t16 - t17;
        let t19 = t0 * t9;
        let t20 = args[28];
        let t21 = t19 + t20;
        let t22 = t14.powf(t11);
        let t23 = t21 - t22;
        let t24 = t0 * t15;
        let t25 = args[29];
        let t26 = t24 + t25;
        let t27 = t20 - t15;
        let t28 = t26 - t27;
        let t29 = t0 * t20;
        let t30 = args[30];
        let t31 = t29 + t30;
        let t32 = t24.powf(t11);
        let t33 = t31 - t32;
        let t34 = t0 * t25;
        let t35 = args[31];
        let t36 = t34 + t35;
        let t37 = t30 - t25;
        let t38 = t36 - t37;
        let t39 = t0 * t30;
        let t40 = args[32];
        let t41 = t39 + t40;
        let t42 = t34.powf(t11);
        let t43 = t41 - t42;
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
    pub fn fixture_bvp32_residual_chunk_4(args: &[f64], out: &mut [f64]) {
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
        let t8 = t0 * t5;
        let t9 = args[34];
        let t10 = t8 + t9;
        let t11 = 3.00000000000000000e0_f64;
        let t12 = t2.powf(t11);
        let t13 = t10 - t12;
        let t14 = t0 * t3;
        let t15 = args[35];
        let t16 = t14 + t15;
        let t17 = t9 - t3;
        let t18 = t16 - t17;
        let t19 = t0 * t9;
        let t20 = args[36];
        let t21 = t19 + t20;
        let t22 = t14.powf(t11);
        let t23 = t21 - t22;
        let t24 = t0 * t15;
        let t25 = args[37];
        let t26 = t24 + t25;
        let t27 = t20 - t15;
        let t28 = t26 - t27;
        let t29 = t0 * t20;
        let t30 = args[38];
        let t31 = t29 + t30;
        let t32 = t24.powf(t11);
        let t33 = t31 - t32;
        let t34 = t0 * t25;
        let t35 = args[39];
        let t36 = t34 + t35;
        let t37 = t30 - t25;
        let t38 = t36 - t37;
        let t39 = t0 * t30;
        let t40 = args[40];
        let t41 = t39 + t40;
        let t42 = t34.powf(t11);
        let t43 = t41 - t42;
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
    pub fn fixture_bvp32_residual_chunk_5(args: &[f64], out: &mut [f64]) {
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
        let t8 = t0 * t5;
        let t9 = args[42];
        let t10 = t8 + t9;
        let t11 = 3.00000000000000000e0_f64;
        let t12 = t2.powf(t11);
        let t13 = t10 - t12;
        let t14 = t0 * t3;
        let t15 = args[43];
        let t16 = t14 + t15;
        let t17 = t9 - t3;
        let t18 = t16 - t17;
        let t19 = t0 * t9;
        let t20 = args[44];
        let t21 = t19 + t20;
        let t22 = t14.powf(t11);
        let t23 = t21 - t22;
        let t24 = t0 * t15;
        let t25 = args[45];
        let t26 = t24 + t25;
        let t27 = t20 - t15;
        let t28 = t26 - t27;
        let t29 = t0 * t20;
        let t30 = args[46];
        let t31 = t29 + t30;
        let t32 = t24.powf(t11);
        let t33 = t31 - t32;
        let t34 = t0 * t25;
        let t35 = args[47];
        let t36 = t34 + t35;
        let t37 = t30 - t25;
        let t38 = t36 - t37;
        let t39 = t0 * t30;
        let t40 = args[48];
        let t41 = t39 + t40;
        let t42 = t34.powf(t11);
        let t43 = t41 - t42;
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
    pub fn fixture_bvp32_residual_chunk_6(args: &[f64], out: &mut [f64]) {
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
        let t8 = t0 * t5;
        let t9 = args[50];
        let t10 = t8 + t9;
        let t11 = 3.00000000000000000e0_f64;
        let t12 = t2.powf(t11);
        let t13 = t10 - t12;
        let t14 = t0 * t3;
        let t15 = args[51];
        let t16 = t14 + t15;
        let t17 = t9 - t3;
        let t18 = t16 - t17;
        let t19 = t0 * t9;
        let t20 = args[52];
        let t21 = t19 + t20;
        let t22 = t14.powf(t11);
        let t23 = t21 - t22;
        let t24 = t0 * t15;
        let t25 = args[53];
        let t26 = t24 + t25;
        let t27 = t20 - t15;
        let t28 = t26 - t27;
        let t29 = t0 * t20;
        let t30 = args[54];
        let t31 = t29 + t30;
        let t32 = t24.powf(t11);
        let t33 = t31 - t32;
        let t34 = t0 * t25;
        let t35 = args[55];
        let t36 = t34 + t35;
        let t37 = t30 - t25;
        let t38 = t36 - t37;
        let t39 = t0 * t30;
        let t40 = args[56];
        let t41 = t39 + t40;
        let t42 = t34.powf(t11);
        let t43 = t41 - t42;
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
    pub fn fixture_bvp32_residual_chunk_7(args: &[f64], out: &mut [f64]) {
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
        let t8 = t0 * t5;
        let t9 = args[58];
        let t10 = t8 + t9;
        let t11 = 3.00000000000000000e0_f64;
        let t12 = t2.powf(t11);
        let t13 = t10 - t12;
        let t14 = t0 * t3;
        let t15 = args[59];
        let t16 = t14 + t15;
        let t17 = t9 - t3;
        let t18 = t16 - t17;
        let t19 = t0 * t9;
        let t20 = args[60];
        let t21 = t19 + t20;
        let t22 = t14.powf(t11);
        let t23 = t21 - t22;
        let t24 = t0 * t15;
        let t25 = args[61];
        let t26 = t24 + t25;
        let t27 = t20 - t15;
        let t28 = t26 - t27;
        let t29 = t0 * t20;
        let t30 = args[62];
        let t31 = t29 + t30;
        let t32 = t24.powf(t11);
        let t33 = t31 - t32;
        let t34 = t0 * t25;
        let t35 = args[63];
        let t36 = t34 + t35;
        let t37 = t30 - t25;
        let t38 = t36 - t37;
        let t39 = t0 * t30;
        let t40 = t34.powf(t11);
        let t41 = t39 - t40;
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

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[1];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_4(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_5(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[3];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_6(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_7(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[5];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_8(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_9(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[7];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_10(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_11(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[9];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_12(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_13(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[11];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_14(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_15(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[13];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_16(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_17(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[15];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_18(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_19(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[17];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_20(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_21(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[19];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_22(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_23(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[21];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_24(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_25(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[23];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_26(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_27(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[25];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_28(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_29(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[27];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_30(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_31(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[29];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_32(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_33(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[31];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_34(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_35(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[33];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_36(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_37(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[35];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_38(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_39(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[37];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_40(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_41(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[39];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_42(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_43(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[41];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_44(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_45(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[43];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_46(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_47(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[45];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_48(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_49(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[47];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_50(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_51(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[49];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_52(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_53(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[51];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_54(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_55(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[53];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_56(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_57(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[55];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_58(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_59(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[57];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_60(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 3 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_61(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 3, "expected at least 3 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[59];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        let t7 = 1.00000000000000000e0_f64;
        out[0] = t6;
        out[1] = t1;
        out[2] = t7;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_62(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 64 rows x 64 cols, 2 non-zero values
    pub fn fixture_bvp32_sparse_values_chunk_63(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 64, "expected at least 64 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[61];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        out[0] = t6;
        out[1] = t1;
    }

}

pub mod generated_bvp_large_fixture_bindings {
    use super::generated_bvp_large_fixture;
    pub const N_STEPS: usize = 32;
    pub const RESIDUAL_CHUNKS: [fn(&[f64], &mut [f64]); 8] = [
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_0,
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_1,
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_2,
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_3,
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_4,
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_5,
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_6,
        generated_bvp_large_fixture::fixture_bvp32_residual_chunk_7,
    ];

    pub const SPARSE_VALUE_CHUNKS: [fn(&[f64], &mut [f64]); 64] = [
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_0,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_1,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_2,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_3,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_4,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_5,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_6,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_7,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_8,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_9,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_10,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_11,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_12,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_13,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_14,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_15,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_16,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_17,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_18,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_19,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_20,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_21,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_22,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_23,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_24,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_25,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_26,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_27,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_28,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_29,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_30,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_31,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_32,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_33,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_34,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_35,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_36,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_37,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_38,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_39,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_40,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_41,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_42,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_43,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_44,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_45,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_46,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_47,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_48,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_49,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_50,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_51,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_52,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_53,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_54,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_55,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_56,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_57,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_58,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_59,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_60,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_61,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_62,
        generated_bvp_large_fixture::fixture_bvp32_sparse_values_chunk_63,
    ];

}
