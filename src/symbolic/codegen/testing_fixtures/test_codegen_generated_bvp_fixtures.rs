// =========================================
// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// =========================================

#![allow(clippy::all)]
#![allow(non_snake_case)]
#![allow(unused_parens)]

pub mod generated_bvp_fixture {
    // Residual block: 8 outputs
    pub fn fixture_bvp_residual_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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
    pub fn fixture_bvp_residual_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_0(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_1(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_2(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 3 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_3(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_4(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 3 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_5(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_6(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 3 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_7(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_8(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 3 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_9(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_10(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 3 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_11(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_12(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 3 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_13(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
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

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_14(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = -1.00000000000000000e0_f64;
        let t1 = 1.00000000000000000e0_f64;
        out[0] = t0;
        out[1] = t1;
    }

    // Sparse Jacobian values block: 16 rows x 16 cols, 2 non-zero values
    pub fn fixture_bvp_sparse_values_chunk_15(args: &[f64], out: &mut [f64]) {
        debug_assert!(args.len() >= 16, "expected at least 16 arguments");
        debug_assert!(out.len() >= 2, "expected at least 2 output slots");
        let t0 = 3.00000000000000000e0_f64;
        let t1 = -1.00000000000000000e0_f64;
        let t2 = args[13];
        let t3 = t1 * t2;
        let t4 = 2.00000000000000000e0_f64;
        let t5 = t3.powf(t4);
        let t6 = t0 * t5;
        out[0] = t6;
        out[1] = t1;
    }
}
