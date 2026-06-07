//! Solver-facing generated-backend comparison diagnostics for `BVP_sci`.
//!
//! These tests intentionally sit one level above low-level callback correctness:
//! they compare complete solver runs through the public symbolic wrapper and are
//! meant to answer the practical question:
//! "if a user gives us a BVP and asks for the fastest correct end-to-end solve,
//! when does the compiled AtomView/AOT path beat the Lambdify baseline?"
//!
//! Current engineering takeaways from repeated release-mode runs:
//! - single-run timings were too noisy for reliable conclusions, so these tests
//!   now report repeated medians/means/minima instead of one-shot measurements;
//! - once timings are aggregated, the compiled backends form a fairly tight
//!   performance tier on small and medium problems such as `linear-2`,
//!   `exponential-2`, and `lane-emden-2-512`;
//! - the pure numerical branch is intentionally represented in two modes:
//!   `Direct-num-FD` for the low-friction rhs-only workflow that relies on
//!   finite-difference Jacobians, and `Direct-num` for the analytical
//!   pointwise-Jacobian upper bound;
//! - repeated release runs now make the practical tradeoff explicit:
//!   `Direct-num-FD` is a realistic convenience path for small systems, but its
//!   finite-difference Jacobian can become both expensive and less reliable on
//!   medium and large problems, while `Direct-num` remains the realistic
//!   high-performance pure numerical path when analytical pointwise derivatives
//!   are available;
//! - this makes the compare layer answer three different user-facing questions
//!   at once: how the symbolic Lambdify baseline behaves, how far the AOT
//!   backends can push the same problem, and what a plain numerical Rust
//!   workflow looks like when a user never touches symbolic APIs at all;
//! - the Rust generated backend should be viewed in two modes:
//!   cold `Rust`, which can still absorb occasional build/bootstrap overhead
//!   into the mean, and `Rust-warm`, which measures the prebuilt/reused path;
//! - on repeated runs, `Rust-warm` is no longer an outlier and is frequently
//!   near the front of the compiled pack, sometimes taking the best median on
//!   small and medium scenarios;
//! - heavier workloads such as `exponential-2-512` and `combustion-1000` remain
//!   the most representative production-like signals: here all compiled
//!   backends, including Rust, are close enough that medians matter more than
//!   isolated one-off wins, while Lambdify, `Direct-num-FD`, and `Direct-num`
//!   provide useful baselines from the symbolic, convenience-numerical, and
//!   high-performance pure numerical sides respectively.
//!
//! The production-like test keeps one compact user-facing table, while the
//! compare-table test also exposes stage breakdowns so we can see whether wins
//! come from residuals, Jacobians, linear solves, or mesh work.
#[cfg(test)]
mod tests_generated_backend_compare {
    use crate::numerical::BVP_sci::BVP_sci_aot::BvpSciGeneratedBackendConfig;
    use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat, faer_mat};
    use crate::numerical::BVP_sci::BVP_sci_numerical::{
        solve_numerical_bvp, NumericalBvpProblem, NumericalBvpSolveOptions, NumericalJacobianMode,
    };
    use crate::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciSolverOptions};
    use crate::numerical::Examples_and_utils::NonlinEquation;
    use crate::symbolic::symbolic_engine::Expr;
    use faer::sparse::Triplet;
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::process::Command;
    use std::sync::OnceLock;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    use tabled::{settings::Style, Table, Tabled};

    const GENERATED_TEST_ARTIFACT_REV: &str = "r2";
    const DEFAULT_COMPARE_REPEATS: usize = 5;

    #[derive(Clone)]
    struct CompareScenario {
        label: &'static str,
        options: BvpSciSolverOptions,
    }

    #[derive(Clone, Tabled)]
    struct CompareTimingRow {
        variant: String,
        total_ms: String,
        setup_ms: String,
        solve_ms: String,
        max_abs_solution: String,
        status: String,
    }

    #[derive(Clone, Tabled)]
    struct CompareBreakdownRow {
        variant: String,
        speedup_vs_lambdify: String,
        solution_diff_vs_lambdify: String,
        residual_ms_total: String,
        jacobian_ms_total: String,
        linear_ms_total: String,
        grid_refine_ms_total: String,
    }

    #[derive(Clone, Tabled)]
    struct CompareWorkRow {
        variant: String,
        niter: usize,
        linear_solves: usize,
        jacobian_rebuilds: usize,
        grid_refinements: usize,
        nodes: usize,
        max_rms_residual: String,
    }

    #[derive(Clone, Tabled)]
    struct CompareProductionRow {
        variant: String,
        total_ms: String,
        setup_ms: String,
        solve_ms: String,
        speedup_vs_lambdify: String,
        max_abs_solution: String,
        solution_diff_vs_lambdify: String,
        residual_ms_total: String,
        jacobian_ms_total: String,
        linear_ms_total: String,
        niter: usize,
        linear_solves: usize,
        jacobian_rebuilds: usize,
        nodes: usize,
        status: String,
    }

    #[derive(Clone)]
    struct CompareOutcome {
        timing: CompareTimingRow,
        breakdown: CompareBreakdownRow,
        work: CompareWorkRow,
        solution: Option<DMatrix<f64>>,
        total_ms_value: Option<f64>,
    }

    #[derive(Clone)]
    struct CompareRunMetrics {
        total_ms: f64,
        setup_ms: f64,
        solve_ms: f64,
        max_abs_solution: f64,
        solution_diff: f64,
        residual_ms_total: f64,
        jacobian_ms_total: f64,
        linear_ms_total: f64,
        grid_refine_ms_total: f64,
        niter: usize,
        linear_solves: usize,
        jacobian_rebuilds: usize,
        grid_refinements: usize,
        nodes: usize,
        max_rms_residual: f64,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum CompareVariant {
        Lambdify,
        DirectNumericFd,
        DirectNumeric,
        Rust,
        RustWarm,
        CGcc,
        CTcc,
        Zig,
    }

    impl CompareVariant {
        fn label(self) -> &'static str {
            match self {
                Self::Lambdify => "Lambdify",
                Self::DirectNumericFd => "Direct-num-FD",
                Self::DirectNumeric => "Direct-num",
                Self::Rust => "Rust",
                Self::RustWarm => "Rust-warm",
                Self::CGcc => "C-gcc",
                Self::CTcc => "C-tcc",
                Self::Zig => "Zig",
            }
        }
    }

    fn tcc_is_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_TCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        Command::new("tcc").arg("-v").output().is_ok()
    }

    fn gcc_is_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_GCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_C_COMPILER") {
            if std::path::Path::new(&explicit).is_file() {
                return explicit.to_ascii_lowercase().contains("gcc");
            }
        }
        Command::new("gcc").arg("-v").output().is_ok()
    }

    fn rust_codegen_is_available() -> bool {
        Command::new("cargo").arg("--version").output().is_ok()
    }

    fn zig_is_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_ZIG") {
            return std::path::Path::new(&explicit).is_file();
        }
        Command::new("zig").arg("version").output().is_ok()
    }

    fn linear_problem_options() -> BvpSciSolverOptions {
        let eq_system = vec![Expr::parse_expression("z"), Expr::parse_expression("0")];
        let values = vec!["y".to_string(), "z".to_string()];
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0), (1, 1.0)]);
        BvpSciSolverOptions::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(8),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-4,
            100,
            DMatrix::zeros(2, 8),
        )
        .with_loglevel(Some("off".to_string()))
    }

    fn exponential_problem_options() -> BvpSciSolverOptions {
        let eq_system = Expr::parse_vector_expression(vec!["z", "-(2.0/4.0)*(1+2.0*ln((y)))*y"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let mut boundary_conditions = HashMap::new();
        let a = 4.0_f64;
        let bc_val = (-1.0 / a).exp();
        boundary_conditions.insert("y".to_string(), vec![(0, bc_val), (1, bc_val)]);
        let mut bounds = HashMap::new();
        bounds.insert("y".to_string(), vec![(0, 1e-10)]);
        BvpSciSolverOptions::new(
            None,
            Some(-1.0),
            Some(1.0),
            Some(32),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            2000,
            DMatrix::zeros(2, 32),
        )
        .with_bounds(Some(bounds))
        .with_loglevel(Some("off".to_string()))
    }

    fn exponential_problem_options_large(n_steps: usize) -> BvpSciSolverOptions {
        let eq_system = Expr::parse_vector_expression(vec!["z", "-(2.0/4.0)*(1+2.0*ln((y)))*y"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let mut boundary_conditions = HashMap::new();
        let a = 4.0_f64;
        let bc_val = (-1.0 / a).exp();
        boundary_conditions.insert("y".to_string(), vec![(0, bc_val), (1, bc_val)]);
        let mut bounds = HashMap::new();
        bounds.insert("y".to_string(), vec![(0, 1e-10)]);
        BvpSciSolverOptions::new(
            None,
            Some(-1.0),
            Some(1.0),
            Some(n_steps),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            (n_steps * 2).max(2048),
            DMatrix::zeros(2, n_steps),
        )
        .with_bounds(Some(bounds))
        .with_loglevel(Some("off".to_string()))
    }

    fn lane_emden_problem_options(n_steps: usize) -> BvpSciSolverOptions {
        let lane_emden = NonlinEquation::LaneEmden5;
        let eq_system = lane_emden.setup();
        let values = lane_emden.values();
        let eps = 1e-3_f64;
        let start_and_end = (eps, lane_emden.span(None, None).1);
        let y_left = (1.0 + eps * eps / 3.0).powf(-0.5);
        let z_left = -(eps / 3.0) * (1.0 + eps * eps / 3.0).powf(-1.5);
        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0usize, y_left)]),
            ("z".to_string(), vec![(0usize, z_left)]),
        ]);
        let initial_guess = DMatrix::from_fn(2, n_steps, |i, j| {
            let x = start_and_end.0
                + j as f64 * (start_and_end.1 - start_and_end.0)
                    / (n_steps.saturating_sub(1)) as f64;
            match i {
                0 => (1.0 + x * x / 3.0).powf(-0.5),
                1 => -(x / 3.0) * (1.0 + x * x / 3.0).powf(-1.5),
                _ => 0.0,
            }
        });
        BvpSciSolverOptions::new(
            None,
            Some(start_and_end.0),
            Some(start_and_end.1),
            Some(n_steps),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-8,
            (n_steps * 2).max(2048),
            initial_guess,
        )
        .with_loglevel(Some("off".to_string()))
    }

    fn combustion_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
        DMatrix::from_element(variable_count, n_steps, value)
    }

    fn combustion_problem_options(n_steps: usize) -> BvpSciSolverOptions {
        let values = vec![
            "Teta".to_string(),
            "q".to_string(),
            "C0".to_string(),
            "J0".to_string(),
            "C1".to_string(),
            "J1".to_string(),
        ];
        let teta = Expr::parse_expression("Teta");
        let q = Expr::parse_expression("q");
        let c0 = Expr::parse_expression("C0");
        let j0 = Expr::parse_expression("J0");
        let j1 = Expr::parse_expression("J1");

        let q_heat = 3000.0 * 1e3 * 0.034;
        let dt = 600.0;
        let t_scale = 600.0;
        let l: f64 = 3e-4;
        let m0 = 34.2 / 1000.0;
        let lambda = 0.07;
        let p = 2e6;
        let tm = 1500.0;
        let c1_0 = 1.0;
        let t_initial = 1000.0;
        let pe_q = 0.0090168;
        let d_ro = 2.88e-4;
        let pe_d = 1.50e-3;
        let ro_m_ = m0 * p / (8.314 * tm);

        let dt_sym = Expr::Const(dt);
        let t_scale_sym = Expr::Const(t_scale);
        let lambda_sym = Expr::Const(lambda);
        let q_heat = Expr::Const(q_heat);
        let a = Expr::Const(1.3e5);
        let e = Expr::Const(5000.0 * 4.184);
        let m = Expr::Const(m0);
        let r_g = Expr::Const(8.314);
        let ro_m = Expr::Const(ro_m_);
        let qm = Expr::Const(l.powf(2.0) / t_scale);
        let qs = Expr::Const(l.powf(2.0));
        let pe_q_sym = Expr::Const(pe_q);
        let ro_d = [Expr::Const(d_ro), Expr::Const(d_ro)];
        let pe_d = [Expr::Const(pe_d), Expr::Const(pe_d)];
        let minus = Expr::Const(-1.0);
        let m_reag = Expr::Const(0.342);

        let rate = a
            * Expr::exp(-e / (r_g * (teta.clone() * t_scale_sym + dt_sym)))
            * c0.clone()
            * (ro_m.clone() / m_reag.clone());
        let eq_system = vec![
            q.clone() / lambda_sym,
            q * pe_q_sym - q_heat * rate.clone() * qm,
            j0.clone() / ro_d[0].clone(),
            j0 * pe_d[0].clone()
                - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone(),
            j1.clone() / ro_d[1].clone(),
            j1 * pe_d[1].clone() - (m.clone() * rate * ro_m / m) * qs,
        ];

        let boundary_conditions = HashMap::from([
            ("Teta".to_string(), vec![(0, (t_initial - dt) / t_scale)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(), vec![(0, c1_0)]),
            ("J0".to_string(), vec![(1, 1e-7)]),
            ("C1".to_string(), vec![(0, 1e-3)]),
            ("J1".to_string(), vec![(1, 1e-7)]),
        ]);
        let bounds = HashMap::from([
            ("Teta".to_string(), vec![(0, 0.0), (1, 10.0)]),
            ("q".to_string(), vec![(0, -1e20), (1, 1e20)]),
            ("C0".to_string(), vec![(0, 0.0), (1, 1.5)]),
            ("J0".to_string(), vec![(0, -1e2), (1, 1e2)]),
            ("C1".to_string(), vec![(0, 0.0), (1, 1.5)]),
            ("J1".to_string(), vec![(0, -1e2), (1, 1e2)]),
        ]);

        BvpSciSolverOptions::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(n_steps),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            n_steps * 2,
            combustion_initial_guess(6, n_steps, 0.99),
        )
        .with_bounds(Some(bounds))
        .with_loglevel(Some("off".to_string()))
    }

    fn compare_scenarios() -> Vec<CompareScenario> {
        vec![
            CompareScenario {
                label: "linear-2",
                options: linear_problem_options(),
            },
            CompareScenario {
                label: "exponential-2",
                options: exponential_problem_options(),
            },
            CompareScenario {
                label: "exponential-2-512",
                options: exponential_problem_options_large(512),
            },
            CompareScenario {
                label: "lane-emden-2-512",
                options: lane_emden_problem_options(512),
            },
            CompareScenario {
                label: "combustion-1000",
                options: combustion_problem_options(7000),
            },
        ]
    }

    fn compare_output_dir(
        scenario: &str,
        variant: CompareVariant,
        namespace: &str,
        repeat_index: Option<usize>,
    ) -> String {
        let suffix = match variant {
            CompareVariant::Lambdify => "l",
            CompareVariant::DirectNumericFd => "dnfd",
            CompareVariant::DirectNumeric => "dn",
            CompareVariant::Rust => "r",
            CompareVariant::RustWarm => "rw",
            CompareVariant::CGcc => "cg",
            CompareVariant::CTcc => "ct",
            CompareVariant::Zig => "z",
        };
        let scenario = match scenario {
            "linear-2" => "l2",
            "exponential-2" => "e2",
            "exponential-2-512" => "e512",
            "lane-emden-2-512" => "le512",
            "combustion-1000" => "c1000",
            other => other,
        };
        let base =
            format!("target/bsc/{GENERATED_TEST_ARTIFACT_REV}/{namespace}/{scenario}/{suffix}");
        match repeat_index {
            Some(index) => format!("{base}/run-{index:02}"),
            None => base,
        }
    }

    fn compare_invocation_id() -> &'static str {
        static INVOCATION_ID: OnceLock<String> = OnceLock::new();
        INVOCATION_ID.get_or_init(|| {
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0);
            format!("p{:x}-{:x}", std::process::id(), millis % 0x100000000)
        })
    }

    fn compare_run_namespace(base: &str) -> String {
        format!("{base}-{}", compare_invocation_id())
    }

    #[test]
    fn compare_output_dirs_isolate_rust_cold_builds_from_loaded_dlls() {
        let cold_0 = compare_output_dir("lane-emden-2-512", CompareVariant::Rust, "story", Some(0));
        let cold_1 = compare_output_dir("lane-emden-2-512", CompareVariant::Rust, "story", Some(1));
        let warm = compare_output_dir("lane-emden-2-512", CompareVariant::RustWarm, "story", None);

        assert_ne!(cold_0, cold_1);
        assert_ne!(cold_0, warm);
        assert!(cold_0.ends_with("r/run-00"));
        assert!(cold_1.ends_with("r/run-01"));
        assert!(warm.ends_with("rw"));
    }

    #[test]
    fn compare_run_namespace_is_process_unique_for_stale_dll_hardening() {
        let namespace = compare_run_namespace("production-like");
        assert!(namespace.starts_with("production-like-p"));
        assert!(namespace.contains(&format!("{:x}", std::process::id())));

        let cold = compare_output_dir("linear-2", CompareVariant::Rust, &namespace, Some(0));
        assert!(cold.contains(&namespace));
        assert!(cold.ends_with("r/run-00"));
        assert!(cold.starts_with("target/bsc/r2/"));
    }

    fn make_solver_quiet(mut options: BvpSciSolverOptions) -> BvpSciSolverOptions {
        options.loglevel = Some("off".to_string());
        options
    }

    struct ScenarioNumericalProblem {
        label: &'static str,
        jacobian_mode: NumericalJacobianMode,
    }

    fn sparse_from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: Vec<Triplet<usize, usize, f64>>,
    ) -> faer_mat {
        faer_mat::try_new_from_triplets(nrows, ncols, &triplets)
            .expect("scenario sparse matrix should be constructible")
    }

    impl NumericalBvpProblem for ScenarioNumericalProblem {
        fn dimension(&self) -> usize {
            match self.label {
                "linear-2" => 2,
                "exponential-2" | "exponential-2-512" => 2,
                "lane-emden-2-512" => 2,
                "combustion-1000" => 6,
                other => {
                    panic!("no pure numerical BVP_sci problem registered for scenario `{other}`")
                }
            }
        }

        fn jacobian_mode(&self) -> NumericalJacobianMode {
            self.jacobian_mode
        }

        fn rhs(&self, x: f64, y: &[f64], _p: &[f64], out: &mut [f64]) {
            match self.label {
                "linear-2" => {
                    out[0] = y[1];
                    out[1] = 0.0;
                }
                "exponential-2" | "exponential-2-512" => {
                    let y0 = y[0].max(1e-10);
                    out[0] = y[1];
                    out[1] = -(2.0 / 4.0) * (1.0 + 2.0 * y0.ln()) * y0;
                }
                "lane-emden-2-512" => {
                    let x = x.max(1e-8);
                    out[0] = y[1];
                    out[1] = -(2.0 / x) * y[1] - y[0].powi(5);
                }
                "combustion-1000" => {
                    let q_heat = 3000.0 * 1e3 * 0.034;
                    let dt = 600.0;
                    let t_scale = 600.0;
                    let l: f64 = 3e-4;
                    let m0 = 34.2 / 1000.0;
                    let lambda = 0.07;
                    let p = 2e6;
                    let tm = 1500.0;
                    let pe_q = 0.0090168;
                    let d_ro = 2.88e-4;
                    let pe_d = 1.50e-3;
                    let ro_m = m0 * p / (8.314 * tm);
                    let qm = l.powi(2) / t_scale;
                    let qs = l.powi(2);
                    let rate = 1.3e5
                        * (-(5000.0 * 4.184) / (8.314 * (y[0] * t_scale + dt))).exp()
                        * y[2]
                        * (ro_m / 0.342);

                    out[0] = y[1] / lambda;
                    out[1] = y[1] * pe_q - q_heat * rate * qm;
                    out[2] = y[3] / d_ro;
                    out[3] = y[3] * pe_d - (-rate * ro_m) * qs;
                    out[4] = y[5] / d_ro;
                    out[5] = y[5] * pe_d - rate * ro_m * qs;
                }
                other => panic!("no RHS registered for scenario `{other}`"),
            }
        }

        fn boundary_residual(&self, ya: &[f64], yb: &[f64], _p: &[f64], out: &mut [f64]) {
            match self.label {
                "linear-2" => {
                    out[0] = ya[0];
                    out[1] = yb[0] - 1.0;
                }
                "exponential-2" | "exponential-2-512" => {
                    let bc_val = (-1.0_f64 / 4.0).exp();
                    out[0] = ya[0] - bc_val;
                    out[1] = yb[0] - bc_val;
                }
                "lane-emden-2-512" => {
                    let eps = 1e-3_f64;
                    let y_left = (1.0 + eps * eps / 3.0).powf(-0.5);
                    let z_left = -(eps / 3.0) * (1.0 + eps * eps / 3.0).powf(-1.5);
                    out[0] = ya[0] - y_left;
                    out[1] = ya[1] - z_left;
                }
                "combustion-1000" => {
                    let dt = 600.0;
                    let t_scale = 600.0;
                    let t_initial = 1000.0;
                    out[0] = ya[0] - (t_initial - dt) / t_scale;
                    out[1] = yb[1] - 1e-10;
                    out[2] = ya[2] - 1.0;
                    out[3] = yb[3] - 1e-7;
                    out[4] = ya[4] - 1e-3;
                    out[5] = yb[5] - 1e-7;
                }
                other => panic!("no BC registered for scenario `{other}`"),
            }
        }

        fn rhs_jacobian(&self, x: f64, y: &[f64], _p: &[f64]) -> Option<faer_mat> {
            let jacobian = match self.label {
                "linear-2" => sparse_from_triplets(2, 2, vec![Triplet::new(0usize, 1usize, 1.0)]),
                "exponential-2" | "exponential-2-512" => {
                    let y0 = y[0].max(1e-10);
                    sparse_from_triplets(
                        2,
                        2,
                        vec![
                            Triplet::new(0usize, 1usize, 1.0),
                            Triplet::new(1usize, 0usize, -1.5 - y0.ln()),
                        ],
                    )
                }
                "lane-emden-2-512" => {
                    let x = x.max(1e-8);
                    sparse_from_triplets(
                        2,
                        2,
                        vec![
                            Triplet::new(0usize, 1usize, 1.0),
                            Triplet::new(1usize, 0usize, -5.0 * y[0].powi(4)),
                            Triplet::new(1usize, 1usize, -(2.0 / x)),
                        ],
                    )
                }
                "combustion-1000" => {
                    let q_heat = 3000.0 * 1e3 * 0.034;
                    let dt = 600.0;
                    let t_scale = 600.0;
                    let l: f64 = 3e-4;
                    let m0 = 34.2 / 1000.0;
                    let lambda = 0.07;
                    let p = 2e6;
                    let tm = 1500.0;
                    let pe_q = 0.0090168;
                    let d_ro = 2.88e-4;
                    let pe_d = 1.50e-3;
                    let ro_m = m0 * p / (8.314 * tm);
                    let qm = l.powi(2) / t_scale;
                    let qs = l.powi(2);
                    let temp = y[0] * t_scale + dt;
                    let exp_term = (-(5000.0 * 4.184) / (8.314 * temp)).exp();
                    let rate_prefactor = 1.3e5 * exp_term * (ro_m / 0.342);
                    let rate = rate_prefactor * y[2];
                    let drate_dtheta = rate * (5000.0 * 4.184) * t_scale / (8.314 * temp * temp);
                    let drate_dc0 = rate_prefactor;

                    sparse_from_triplets(
                        6,
                        6,
                        vec![
                            Triplet::new(0usize, 1usize, 1.0 / lambda),
                            Triplet::new(1usize, 0usize, -q_heat * qm * drate_dtheta),
                            Triplet::new(1usize, 1usize, pe_q),
                            Triplet::new(1usize, 2usize, -q_heat * qm * drate_dc0),
                            Triplet::new(2usize, 3usize, 1.0 / d_ro),
                            Triplet::new(3usize, 0usize, ro_m * qs * drate_dtheta),
                            Triplet::new(3usize, 2usize, ro_m * qs * drate_dc0),
                            Triplet::new(3usize, 3usize, pe_d),
                            Triplet::new(4usize, 5usize, 1.0 / d_ro),
                            Triplet::new(5usize, 0usize, -ro_m * qs * drate_dtheta),
                            Triplet::new(5usize, 2usize, -ro_m * qs * drate_dc0),
                            Triplet::new(5usize, 5usize, pe_d),
                        ],
                    )
                }
                other => panic!("no RHS Jacobian registered for scenario `{other}`"),
            };

            Some(jacobian)
        }

        fn boundary_jacobian(
            &self,
            _ya: &[f64],
            _yb: &[f64],
            _p: &[f64],
        ) -> Option<(faer_mat, faer_mat, Option<faer_mat>)> {
            let (dya, dyb) = match self.label {
                "linear-2" => (
                    sparse_from_triplets(2, 2, vec![Triplet::new(0usize, 0usize, 1.0)]),
                    sparse_from_triplets(2, 2, vec![Triplet::new(1usize, 0usize, 1.0)]),
                ),
                "exponential-2" | "exponential-2-512" => (
                    sparse_from_triplets(2, 2, vec![Triplet::new(0usize, 0usize, 1.0)]),
                    sparse_from_triplets(2, 2, vec![Triplet::new(1usize, 0usize, 1.0)]),
                ),
                "lane-emden-2-512" => (
                    sparse_from_triplets(
                        2,
                        2,
                        vec![
                            Triplet::new(0usize, 0usize, 1.0),
                            Triplet::new(1usize, 1usize, 1.0),
                        ],
                    ),
                    sparse_from_triplets(2, 2, Vec::new()),
                ),
                "combustion-1000" => (
                    sparse_from_triplets(
                        6,
                        6,
                        vec![
                            Triplet::new(0usize, 0usize, 1.0),
                            Triplet::new(2usize, 2usize, 1.0),
                            Triplet::new(4usize, 4usize, 1.0),
                        ],
                    ),
                    sparse_from_triplets(
                        6,
                        6,
                        vec![
                            Triplet::new(1usize, 1usize, 1.0),
                            Triplet::new(3usize, 3usize, 1.0),
                            Triplet::new(5usize, 5usize, 1.0),
                        ],
                    ),
                ),
                other => panic!("no BC Jacobian registered for scenario `{other}`"),
            };

            Some((dya, dyb, None))
        }
    }

    fn faer_dense_from_dmatrix(values: &DMatrix<f64>) -> faer_dense_mat {
        faer_dense_mat::from_fn(values.nrows(), values.ncols(), |i, j| values[(i, j)])
    }

    fn dmatrix_from_faer_dense(values: &faer_dense_mat) -> DMatrix<f64> {
        let (nrows, ncols) = values.shape();
        let mut dmatrix = DMatrix::zeros(ncols, nrows);
        for (i, row) in values.row_iter().enumerate() {
            let row_values = row.to_owned().iter().copied().collect::<Vec<f64>>();
            dmatrix
                .column_mut(i)
                .copy_from(&nalgebra::DVector::from_vec(row_values));
        }
        dmatrix
    }

    fn mesh_from_options(options: &BvpSciSolverOptions) -> faer_col {
        if let Some(mesh) = options.x_mesh_set.as_ref() {
            return faer_col::from_iter(mesh.iter().copied());
        }

        let start = options
            .t0
            .expect("BVP_sci compare scenario should provide t0 when mesh is implicit");
        let end = options
            .t_end
            .expect("BVP_sci compare scenario should provide t_end when mesh is implicit");
        let steps = options
            .n_steps
            .expect("BVP_sci compare scenario should provide n_steps when mesh is implicit");

        faer_col::from_fn(steps, |i| {
            start + (end - start) * i as f64 / (steps.saturating_sub(1)) as f64
        })
    }

    fn configure_variant(
        scenario: &CompareScenario,
        variant: CompareVariant,
        namespace: &str,
        repeat_index: usize,
    ) -> BvpSciSolverOptions {
        let options = make_solver_quiet(scenario.options.clone());
        match variant {
            CompareVariant::Lambdify => options,
            CompareVariant::DirectNumericFd => {
                panic!("DirectNumericFd variant bypasses symbolic configuration")
            }
            CompareVariant::DirectNumeric => {
                panic!("DirectNumeric variant bypasses symbolic configuration")
            }
            CompareVariant::Rust => options.with_sparse_atomview_rust(compare_output_dir(
                scenario.label,
                variant,
                namespace,
                Some(repeat_index),
            )),
            CompareVariant::RustWarm => options.with_generated_backend_config(
                BvpSciGeneratedBackendConfig::sparse_atomview_require_prebuilt()
                    .with_output_parent_dir(compare_output_dir(
                        scenario.label,
                        CompareVariant::RustWarm,
                        namespace,
                        None,
                    )),
            ),
            CompareVariant::CGcc => options.with_sparse_atomview_c_gcc(compare_output_dir(
                scenario.label,
                variant,
                namespace,
                Some(repeat_index),
            )),
            CompareVariant::CTcc => options.with_sparse_atomview_c_tcc(compare_output_dir(
                scenario.label,
                variant,
                namespace,
                Some(repeat_index),
            )),
            CompareVariant::Zig => options.with_sparse_atomview_zig(compare_output_dir(
                scenario.label,
                variant,
                namespace,
                Some(repeat_index),
            )),
        }
    }

    fn max_abs_matrix(matrix: &DMatrix<f64>) -> f64 {
        matrix
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    fn max_abs_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
    }

    fn max_abs_faer_col(column: &faer_col) -> f64 {
        column
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    fn dominant_hot_path_label(
        residual_ms_total: f64,
        jacobian_ms_total: f64,
        linear_ms_total: f64,
        grid_refinement_ms_total: f64,
    ) -> &'static str {
        let buckets = [
            ("residual-dominated", residual_ms_total),
            ("jacobian-dominated", jacobian_ms_total),
            ("linear-solve-dominated", linear_ms_total),
            ("grid-refinement-dominated", grid_refinement_ms_total),
        ];
        let mut sorted = buckets;
        sorted.sort_by(|(_, lhs), (_, rhs)| rhs.partial_cmp(lhs).unwrap());
        if sorted[0].1 <= 0.0 {
            "mixed"
        } else if sorted[1].1 <= 0.0 || sorted[0].1 >= sorted[1].1 * 1.2 {
            sorted[0].0
        } else {
            "mixed"
        }
    }

    fn outcome_from_direct_result(
        variant: CompareVariant,
        total_ms: f64,
        result: crate::numerical::BVP_sci::BVP_sci_faer::BVPResult,
        solution: DMatrix<f64>,
        baseline_solution: Option<&DMatrix<f64>>,
        baseline_total_ms: Option<f64>,
    ) -> CompareOutcome {
        let timer = &result.custom_timer;
        let get_count = |key: &str| result.calc_statistics.get(key).copied().unwrap_or(0);
        let residual_ms_total = timer.fun.as_secs_f64() * 1_000.0;
        let jacobian_ms_total = timer.jac.as_secs_f64() * 1_000.0;
        let linear_ms_total = timer.linear_system.as_secs_f64() * 1_000.0;
        let grid_refine_ms_total = timer.grid_refinement.as_secs_f64() * 1_000.0;
        let solve_ms = total_ms;
        let max_abs_solution = max_abs_matrix(&solution);
        let solution_diff = baseline_solution
            .map(|baseline| max_abs_diff(baseline, &solution))
            .unwrap_or(0.0);
        let speedup_vs_lambdify = baseline_total_ms
            .filter(|baseline| *baseline > 0.0)
            .map(|baseline| format!("{:.3}x", baseline / total_ms))
            .unwrap_or_else(|| "1.000x".to_string());
        let status = if result.success {
            "finished".to_string()
        } else {
            format!("status={} {}", result.status, result.message)
        };

        CompareOutcome {
            timing: CompareTimingRow {
                variant: variant.label().to_string(),
                total_ms: format!("{:.3}", total_ms),
                setup_ms: format!("{:.3}", 0.0),
                solve_ms: format!("{:.3}", solve_ms),
                max_abs_solution: format!("{:.6e}", max_abs_solution),
                status: status.clone(),
            },
            breakdown: CompareBreakdownRow {
                variant: variant.label().to_string(),
                speedup_vs_lambdify,
                solution_diff_vs_lambdify: format!("{:.6e}", solution_diff),
                residual_ms_total: format!("{:.3}", residual_ms_total),
                jacobian_ms_total: format!("{:.3}", jacobian_ms_total),
                linear_ms_total: format!("{:.3}", linear_ms_total),
                grid_refine_ms_total: format!("{:.3}", grid_refine_ms_total),
            },
            work: CompareWorkRow {
                variant: variant.label().to_string(),
                niter: result.niter,
                linear_solves: get_count("number of solving linear systems"),
                jacobian_rebuilds: get_count("number of jacobians recalculations"),
                grid_refinements: get_count("number of grid refinements"),
                nodes: result.x.nrows(),
                max_rms_residual: format!(
                    "{:.6e}",
                    result
                        .rms_residuals
                        .iter()
                        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
                ),
            },
            solution: Some(solution),
            total_ms_value: Some(total_ms),
        }
    }

    fn run_direct_numeric_variant(
        scenario: &CompareScenario,
        variant: CompareVariant,
        baseline_solution: Option<&DMatrix<f64>>,
        baseline_total_ms: Option<f64>,
    ) -> CompareOutcome {
        let options = make_solver_quiet(scenario.options.clone());
        let mesh = mesh_from_options(&options);
        let initial_guess = faer_dense_from_dmatrix(&options.initial_guess);
        let parameters = options
            .param_values
            .as_ref()
            .map(|values| faer_col::from_iter(values.iter().copied()));
        let numerical_problem = ScenarioNumericalProblem {
            label: scenario.label,
            jacobian_mode: match variant {
                CompareVariant::DirectNumericFd => NumericalJacobianMode::FiniteDifference,
                CompareVariant::DirectNumeric => NumericalJacobianMode::AnalyticalPointwise,
                other => panic!(
                    "run_direct_numeric_variant should only be used for direct numerical variants, got `{}`",
                    other.label()
                ),
            },
        };
        let solve_options = NumericalBvpSolveOptions::new(
            mesh,
            initial_guess,
            options.tolerance,
            options.max_nodes,
        )
        .with_parameters(parameters)
        .with_verbose(0);

        let begin = Instant::now();
        match solve_numerical_bvp(numerical_problem, solve_options) {
            Ok(result) => {
                let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
                let solution = dmatrix_from_faer_dense(&result.y);
                outcome_from_direct_result(
                    variant,
                    total_ms,
                    result,
                    solution,
                    baseline_solution,
                    baseline_total_ms,
                )
            }
            Err(err) => CompareOutcome {
                timing: CompareTimingRow {
                    variant: variant.label().to_string(),
                    total_ms: "NaN".to_string(),
                    setup_ms: "NaN".to_string(),
                    solve_ms: "NaN".to_string(),
                    max_abs_solution: "NaN".to_string(),
                    status: err,
                },
                breakdown: CompareBreakdownRow {
                    variant: variant.label().to_string(),
                    speedup_vs_lambdify: "NaNx".to_string(),
                    solution_diff_vs_lambdify: "NaN".to_string(),
                    residual_ms_total: "NaN".to_string(),
                    jacobian_ms_total: "NaN".to_string(),
                    linear_ms_total: "NaN".to_string(),
                    grid_refine_ms_total: "NaN".to_string(),
                },
                work: CompareWorkRow {
                    variant: variant.label().to_string(),
                    niter: 0,
                    linear_solves: 0,
                    jacobian_rebuilds: 0,
                    grid_refinements: 0,
                    nodes: 0,
                    max_rms_residual: "NaN".to_string(),
                },
                solution: None,
                total_ms_value: None,
            },
        }
    }

    fn failed_compare_outcome(
        variant: CompareVariant,
        status: impl Into<String>,
    ) -> CompareOutcome {
        CompareOutcome {
            timing: CompareTimingRow {
                variant: variant.label().to_string(),
                total_ms: "NaN".to_string(),
                setup_ms: "NaN".to_string(),
                solve_ms: "NaN".to_string(),
                max_abs_solution: "NaN".to_string(),
                status: status.into(),
            },
            breakdown: CompareBreakdownRow {
                variant: variant.label().to_string(),
                speedup_vs_lambdify: "NaNx".to_string(),
                solution_diff_vs_lambdify: "NaN".to_string(),
                residual_ms_total: "NaN".to_string(),
                jacobian_ms_total: "NaN".to_string(),
                linear_ms_total: "NaN".to_string(),
                grid_refine_ms_total: "NaN".to_string(),
            },
            work: CompareWorkRow {
                variant: variant.label().to_string(),
                niter: 0,
                linear_solves: 0,
                jacobian_rebuilds: 0,
                grid_refinements: 0,
                nodes: 0,
                max_rms_residual: "NaN".to_string(),
            },
            solution: None,
            total_ms_value: None,
        }
    }

    fn median_of(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        let mid = values.len() / 2;
        if values.len() % 2 == 1 {
            values[mid]
        } else {
            (values[mid - 1] + values[mid]) * 0.5
        }
    }

    fn mean_of(values: &[f64]) -> f64 {
        if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn compare_repeats() -> usize {
        std::env::var("RUSTEDSCITHE_COMPARE_REPEATS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_COMPARE_REPEATS)
    }

    fn aggregate_compare_runs(
        variant: CompareVariant,
        runs: &[CompareRunMetrics],
        status: String,
        baseline_median_total_ms: Option<f64>,
    ) -> CompareOutcome {
        let mut total_values = runs.iter().map(|run| run.total_ms).collect::<Vec<_>>();
        let median_total_ms = median_of(&mut total_values);
        let mean_total_ms = mean_of(&total_values);
        let min_total_ms = total_values.iter().copied().fold(f64::INFINITY, f64::min);

        let mut setup_values = runs.iter().map(|run| run.setup_ms).collect::<Vec<_>>();
        let median_setup_ms = median_of(&mut setup_values);

        let mut solve_values = runs.iter().map(|run| run.solve_ms).collect::<Vec<_>>();
        let median_solve_ms = median_of(&mut solve_values);

        let mut max_abs_values = runs
            .iter()
            .map(|run| run.max_abs_solution)
            .collect::<Vec<_>>();
        let median_max_abs = median_of(&mut max_abs_values);

        let mut solution_diff_values = runs.iter().map(|run| run.solution_diff).collect::<Vec<_>>();
        let median_solution_diff = median_of(&mut solution_diff_values);

        let mut residual_values = runs
            .iter()
            .map(|run| run.residual_ms_total)
            .collect::<Vec<_>>();
        let median_residual_ms = median_of(&mut residual_values);

        let mut jacobian_values = runs
            .iter()
            .map(|run| run.jacobian_ms_total)
            .collect::<Vec<_>>();
        let median_jacobian_ms = median_of(&mut jacobian_values);

        let mut linear_values = runs
            .iter()
            .map(|run| run.linear_ms_total)
            .collect::<Vec<_>>();
        let median_linear_ms = median_of(&mut linear_values);

        let mut grid_values = runs
            .iter()
            .map(|run| run.grid_refine_ms_total)
            .collect::<Vec<_>>();
        let median_grid_ms = median_of(&mut grid_values);

        let mut rms_values = runs
            .iter()
            .map(|run| run.max_rms_residual)
            .collect::<Vec<_>>();
        let median_rms = median_of(&mut rms_values);

        let mean_niter = runs.iter().map(|run| run.niter as f64).sum::<f64>() / runs.len() as f64;
        let mean_linear_solves =
            runs.iter().map(|run| run.linear_solves as f64).sum::<f64>() / runs.len() as f64;
        let mean_jacobian_rebuilds = runs
            .iter()
            .map(|run| run.jacobian_rebuilds as f64)
            .sum::<f64>()
            / runs.len() as f64;
        let mean_grid_refinements = runs
            .iter()
            .map(|run| run.grid_refinements as f64)
            .sum::<f64>()
            / runs.len() as f64;
        let mean_nodes = runs.iter().map(|run| run.nodes as f64).sum::<f64>() / runs.len() as f64;

        let speedup_vs_lambdify = baseline_median_total_ms
            .filter(|baseline| *baseline > 0.0)
            .map(|baseline| format!("{:.3}x", baseline / median_total_ms))
            .unwrap_or_else(|| "1.000x".to_string());

        CompareOutcome {
            timing: CompareTimingRow {
                variant: variant.label().to_string(),
                total_ms: format!(
                    "{:.3} med / {:.3} mean / {:.3} min",
                    median_total_ms, mean_total_ms, min_total_ms
                ),
                setup_ms: format!("{:.3} med", median_setup_ms),
                solve_ms: format!("{:.3} med", median_solve_ms),
                max_abs_solution: format!("{:.6e}", median_max_abs),
                status,
            },
            breakdown: CompareBreakdownRow {
                variant: variant.label().to_string(),
                speedup_vs_lambdify,
                solution_diff_vs_lambdify: format!("{:.6e}", median_solution_diff),
                residual_ms_total: format!("{:.3} med", median_residual_ms),
                jacobian_ms_total: format!("{:.3} med", median_jacobian_ms),
                linear_ms_total: format!("{:.3} med", median_linear_ms),
                grid_refine_ms_total: format!("{:.3} med", median_grid_ms),
            },
            work: CompareWorkRow {
                variant: variant.label().to_string(),
                niter: mean_niter.round() as usize,
                linear_solves: mean_linear_solves.round() as usize,
                jacobian_rebuilds: mean_jacobian_rebuilds.round() as usize,
                grid_refinements: mean_grid_refinements.round() as usize,
                nodes: mean_nodes.round() as usize,
                max_rms_residual: format!("{:.6e}", median_rms),
            },
            solution: None,
            total_ms_value: Some(median_total_ms),
        }
    }

    fn run_compare_variant(
        scenario: &CompareScenario,
        variant: CompareVariant,
        baseline_solution: Option<&DMatrix<f64>>,
        baseline_total_ms: Option<f64>,
        namespace: &str,
        repeat_index: usize,
    ) -> CompareOutcome {
        if matches!(
            variant,
            CompareVariant::DirectNumeric | CompareVariant::DirectNumericFd
        ) {
            return run_direct_numeric_variant(
                scenario,
                variant,
                baseline_solution,
                baseline_total_ms,
            );
        }
        let mut solver = BVPwrap::new_with_options(configure_variant(
            scenario,
            variant,
            namespace,
            repeat_index,
        ));
        let begin = Instant::now();
        let result = solver.try_solve();
        let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;

        match result {
            Ok(()) => {
                let stats = solver.get_statistics();
                let solution = solver
                    .get_result()
                    .expect("BVP_sci compare solve should produce a result matrix");
                let solve_ms = (total_ms - stats.symbolic_prepare_ms_total).max(0.0);
                let max_abs_solution = max_abs_matrix(&solution);
                let solution_diff = baseline_solution
                    .map(|baseline| max_abs_diff(baseline, &solution))
                    .unwrap_or(0.0);
                let speedup_vs_lambdify = baseline_total_ms
                    .filter(|baseline| *baseline > 0.0)
                    .map(|baseline| format!("{:.3}x", baseline / total_ms))
                    .unwrap_or_else(|| "1.000x".to_string());
                let status = if solver.result.success {
                    "finished".to_string()
                } else {
                    format!("status={} {}", solver.result.status, solver.result.message)
                };
                CompareOutcome {
                    timing: CompareTimingRow {
                        variant: variant.label().to_string(),
                        total_ms: format!("{:.3}", total_ms),
                        setup_ms: format!("{:.3}", stats.symbolic_prepare_ms_total),
                        solve_ms: format!("{:.3}", solve_ms),
                        max_abs_solution: format!("{:.6e}", max_abs_solution),
                        status: status.clone(),
                    },
                    breakdown: CompareBreakdownRow {
                        variant: variant.label().to_string(),
                        speedup_vs_lambdify,
                        solution_diff_vs_lambdify: format!("{:.6e}", solution_diff),
                        residual_ms_total: format!("{:.3}", stats.residual_ms_total),
                        jacobian_ms_total: format!("{:.3}", stats.jacobian_ms_total),
                        linear_ms_total: format!("{:.3}", stats.linear_system_ms_total),
                        grid_refine_ms_total: format!("{:.3}", stats.grid_refinement_ms_total),
                    },
                    work: CompareWorkRow {
                        variant: variant.label().to_string(),
                        niter: solver.result.niter,
                        linear_solves: stats.number_of_linear_solves,
                        jacobian_rebuilds: stats.number_of_jacobian_recalculations,
                        grid_refinements: stats.number_of_grid_refinements,
                        nodes: stats.number_of_grid_points,
                        max_rms_residual: format!(
                            "{:.6e}",
                            max_abs_faer_col(&solver.result.rms_residuals)
                        ),
                    },
                    solution: Some(solution),
                    total_ms_value: Some(total_ms),
                }
            }
            Err(err) => CompareOutcome {
                timing: CompareTimingRow {
                    variant: variant.label().to_string(),
                    total_ms: "NaN".to_string(),
                    setup_ms: "NaN".to_string(),
                    solve_ms: "NaN".to_string(),
                    max_abs_solution: "NaN".to_string(),
                    status: err.to_string(),
                },
                breakdown: CompareBreakdownRow {
                    variant: variant.label().to_string(),
                    speedup_vs_lambdify: "NaNx".to_string(),
                    solution_diff_vs_lambdify: "NaN".to_string(),
                    residual_ms_total: "NaN".to_string(),
                    jacobian_ms_total: "NaN".to_string(),
                    linear_ms_total: "NaN".to_string(),
                    grid_refine_ms_total: "NaN".to_string(),
                },
                work: CompareWorkRow {
                    variant: variant.label().to_string(),
                    niter: 0,
                    linear_solves: 0,
                    jacobian_rebuilds: 0,
                    grid_refinements: 0,
                    nodes: 0,
                    max_rms_residual: "NaN".to_string(),
                },
                solution: None,
                total_ms_value: None,
            },
        }
    }

    fn run_compare_variant_repeated(
        scenario: &CompareScenario,
        variant: CompareVariant,
        baseline_solution: Option<&DMatrix<f64>>,
        baseline_total_ms: Option<f64>,
        repeats: usize,
        namespace: &str,
    ) -> CompareOutcome {
        let mut runs = Vec::with_capacity(repeats);
        let mut final_solution = None;

        for repeat_index in 0..repeats {
            let outcome = run_compare_variant(
                scenario,
                variant,
                baseline_solution,
                baseline_total_ms,
                namespace,
                repeat_index,
            );
            if outcome.solution.is_none() {
                return outcome;
            }

            let total_ms = outcome.total_ms_value.unwrap_or(f64::NAN);
            let setup_ms = outcome
                .timing
                .setup_ms
                .split_whitespace()
                .next()
                .and_then(|raw| raw.parse::<f64>().ok())
                .unwrap_or(f64::NAN);
            let solve_ms = outcome
                .timing
                .solve_ms
                .split_whitespace()
                .next()
                .and_then(|raw| raw.parse::<f64>().ok())
                .unwrap_or(f64::NAN);
            let max_abs_solution = outcome
                .timing
                .max_abs_solution
                .parse::<f64>()
                .unwrap_or(f64::NAN);
            let solution_diff = outcome
                .breakdown
                .solution_diff_vs_lambdify
                .parse::<f64>()
                .unwrap_or(f64::NAN);
            let residual_ms_total = outcome
                .breakdown
                .residual_ms_total
                .parse::<f64>()
                .unwrap_or(f64::NAN);
            let jacobian_ms_total = outcome
                .breakdown
                .jacobian_ms_total
                .parse::<f64>()
                .unwrap_or(f64::NAN);
            let linear_ms_total = outcome
                .breakdown
                .linear_ms_total
                .parse::<f64>()
                .unwrap_or(f64::NAN);
            let grid_refine_ms_total = outcome
                .breakdown
                .grid_refine_ms_total
                .parse::<f64>()
                .unwrap_or(f64::NAN);
            let max_rms_residual = outcome
                .work
                .max_rms_residual
                .parse::<f64>()
                .unwrap_or(f64::NAN);

            final_solution = outcome.solution.clone();
            runs.push(CompareRunMetrics {
                total_ms,
                setup_ms,
                solve_ms,
                max_abs_solution,
                solution_diff,
                residual_ms_total,
                jacobian_ms_total,
                linear_ms_total,
                grid_refine_ms_total,
                niter: outcome.work.niter,
                linear_solves: outcome.work.linear_solves,
                jacobian_rebuilds: outcome.work.jacobian_rebuilds,
                grid_refinements: outcome.work.grid_refinements,
                nodes: outcome.work.nodes,
                max_rms_residual,
            });
        }

        let mut aggregated = aggregate_compare_runs(
            variant,
            &runs,
            format!("finished x{repeats}"),
            baseline_total_ms,
        );
        aggregated.solution = final_solution;
        aggregated
    }

    fn run_compare_variant_with_warmup(
        scenario: &CompareScenario,
        variant: CompareVariant,
        baseline_solution: Option<&DMatrix<f64>>,
        baseline_total_ms: Option<f64>,
        repeats: usize,
        namespace: &str,
    ) -> CompareOutcome {
        if matches!(variant, CompareVariant::RustWarm) {
            let mut warmup_solver = BVPwrap::new_with_options(
                make_solver_quiet(scenario.options.clone()).with_sparse_atomview_rust(
                    compare_output_dir(scenario.label, CompareVariant::RustWarm, namespace, None),
                ),
            );
            if let Err(err) = warmup_solver.try_solve() {
                return failed_compare_outcome(variant, format!("warmup_failed: {err}"));
            }
        }
        run_compare_variant_repeated(
            scenario,
            variant,
            baseline_solution,
            baseline_total_ms,
            repeats,
            namespace,
        )
    }

    #[test]
    #[ignore = "diagnostic compare for BVP_sci Lambdify vs generated backends"]
    fn bvp_sci_generated_backend_compare_table() {
        let scenarios = compare_scenarios();
        let repeats = compare_repeats();
        let namespace = compare_run_namespace("compare-table");
        println!("[BVP_sci backend compare] artifact namespace={namespace}");

        for scenario in scenarios {
            let mut outcomes = Vec::new();
            let baseline_outcome = run_compare_variant_repeated(
                &scenario,
                CompareVariant::Lambdify,
                None,
                None,
                repeats,
                &namespace,
            );
            let baseline_solution = baseline_outcome
                .solution
                .clone()
                .expect("lambdify baseline should produce a solution");
            let baseline_total_ms = baseline_outcome
                .total_ms_value
                .expect("lambdify baseline should produce total timing");
            outcomes.push(baseline_outcome);
            outcomes.push(run_compare_variant_repeated(
                &scenario,
                CompareVariant::DirectNumericFd,
                Some(&baseline_solution),
                Some(baseline_total_ms),
                repeats,
                &namespace,
            ));
            outcomes.push(run_compare_variant_repeated(
                &scenario,
                CompareVariant::DirectNumeric,
                Some(&baseline_solution),
                Some(baseline_total_ms),
                repeats,
                &namespace,
            ));

            if rust_codegen_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::Rust,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
                outcomes.push(run_compare_variant_with_warmup(
                    &scenario,
                    CompareVariant::RustWarm,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }
            if gcc_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::CGcc,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }
            if tcc_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::CTcc,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }
            if zig_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::Zig,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }

            println!(
                "[BVP_sci backend compare] scenario={}, variants={}, repeats={}",
                scenario.label,
                outcomes.len(),
                repeats
            );
            let best_total = outcomes
                .iter()
                .filter_map(|outcome| {
                    outcome
                        .total_ms_value
                        .map(|ms| (&outcome.timing.variant, ms))
                })
                .min_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| "n/a".to_string());

            let mut timing_table = Table::new(
                outcomes
                    .iter()
                    .map(|outcome| outcome.timing.clone())
                    .collect::<Vec<_>>(),
            );
            timing_table.with(Style::modern_rounded());
            println!("{timing_table}");

            let mut breakdown_table = Table::new(
                outcomes
                    .iter()
                    .map(|outcome| outcome.breakdown.clone())
                    .collect::<Vec<_>>(),
            );
            breakdown_table.with(Style::modern_rounded());
            println!("{breakdown_table}");

            let mut work_table = Table::new(
                outcomes
                    .iter()
                    .map(|outcome| outcome.work.clone())
                    .collect::<Vec<_>>(),
            );
            work_table.with(Style::modern_rounded());
            println!("{work_table}");

            let baseline_breakdown = &outcomes[0].breakdown;
            let dominant_hot_path = dominant_hot_path_label(
                baseline_breakdown
                    .residual_ms_total
                    .parse::<f64>()
                    .unwrap_or(0.0),
                baseline_breakdown
                    .jacobian_ms_total
                    .parse::<f64>()
                    .unwrap_or(0.0),
                baseline_breakdown
                    .linear_ms_total
                    .parse::<f64>()
                    .unwrap_or(0.0),
                baseline_breakdown
                    .grid_refine_ms_total
                    .parse::<f64>()
                    .unwrap_or(0.0),
            );

            println!(
                "[BVP_sci backend compare] summary: dominant_hot_path={}, best_total={}, baseline_residual_ms_total={}, baseline_jacobian_ms_total={}, baseline_linear_ms_total={}",
                dominant_hot_path,
                best_total,
                baseline_breakdown.residual_ms_total,
                baseline_breakdown.jacobian_ms_total,
                baseline_breakdown.linear_ms_total,
            );
            println!(
                "[BVP_sci backend compare] finished scenario `{}` baseline_total_ms_med={:.3}",
                scenario.label, baseline_total_ms
            );
        }
    }

    #[test]
    #[ignore = "production-like end-to-end compare for BVP_sci Lambdify vs generated backends"]
    fn bvp_sci_production_like_end_to_end_compare_table() {
        let scenarios = compare_scenarios();
        let repeats = compare_repeats();
        let namespace = compare_run_namespace("production-like");
        println!("[BVP_sci production-like] artifact namespace={namespace}");

        for scenario in scenarios {
            let mut outcomes = Vec::new();
            let baseline_outcome = run_compare_variant_repeated(
                &scenario,
                CompareVariant::Lambdify,
                None,
                None,
                repeats,
                &namespace,
            );
            let baseline_solution = baseline_outcome
                .solution
                .clone()
                .expect("lambdify baseline should produce a solution");
            let baseline_total_ms = baseline_outcome
                .total_ms_value
                .expect("lambdify baseline should produce total timing");
            outcomes.push(baseline_outcome);
            outcomes.push(run_compare_variant_repeated(
                &scenario,
                CompareVariant::DirectNumericFd,
                Some(&baseline_solution),
                Some(baseline_total_ms),
                repeats,
                &namespace,
            ));
            outcomes.push(run_compare_variant_repeated(
                &scenario,
                CompareVariant::DirectNumeric,
                Some(&baseline_solution),
                Some(baseline_total_ms),
                repeats,
                &namespace,
            ));

            if rust_codegen_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::Rust,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
                outcomes.push(run_compare_variant_with_warmup(
                    &scenario,
                    CompareVariant::RustWarm,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }
            if gcc_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::CGcc,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }
            if tcc_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::CTcc,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }
            if zig_is_available() {
                outcomes.push(run_compare_variant_repeated(
                    &scenario,
                    CompareVariant::Zig,
                    Some(&baseline_solution),
                    Some(baseline_total_ms),
                    repeats,
                    &namespace,
                ));
            }

            let rows = outcomes
                .iter()
                .map(|outcome| CompareProductionRow {
                    variant: outcome.timing.variant.clone(),
                    total_ms: outcome.timing.total_ms.clone(),
                    setup_ms: outcome.timing.setup_ms.clone(),
                    solve_ms: outcome.timing.solve_ms.clone(),
                    speedup_vs_lambdify: outcome.breakdown.speedup_vs_lambdify.clone(),
                    max_abs_solution: outcome.timing.max_abs_solution.clone(),
                    solution_diff_vs_lambdify: outcome.breakdown.solution_diff_vs_lambdify.clone(),
                    residual_ms_total: outcome.breakdown.residual_ms_total.clone(),
                    jacobian_ms_total: outcome.breakdown.jacobian_ms_total.clone(),
                    linear_ms_total: outcome.breakdown.linear_ms_total.clone(),
                    niter: outcome.work.niter,
                    linear_solves: outcome.work.linear_solves,
                    jacobian_rebuilds: outcome.work.jacobian_rebuilds,
                    nodes: outcome.work.nodes,
                    status: outcome.timing.status.clone(),
                })
                .collect::<Vec<_>>();

            let best_total = rows
                .iter()
                .zip(outcomes.iter())
                .filter_map(|(row, outcome)| outcome.total_ms_value.map(|ms| (&row.variant, ms)))
                .min_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
                .map(|(name, _)| name.clone())
                .unwrap_or_else(|| "n/a".to_string());

            println!(
                "[BVP_sci production-like] scenario={}, variants={}, repeats={}",
                scenario.label,
                rows.len(),
                repeats
            );
            let mut table = Table::new(rows);
            table.with(Style::modern_rounded());
            println!("{table}");
            println!(
                "[BVP_sci production-like] best_total={} scenario={}",
                best_total, scenario.label
            );
            println!(
                "[BVP_sci production-like] finished scenario `{}`",
                scenario.label
            );
        }
    }
}
