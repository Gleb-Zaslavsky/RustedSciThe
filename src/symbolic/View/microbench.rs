//! Microbenchmarks comparing the boxed [`Expr`] differentiation path with the
//! packed [`Atom`] / [`AtomView`] differentiation path.
//!
//! These tests are intentionally `#[ignore]`: they are performance diagnostics,
//! not regression tests. The goal is to quantify the expected speedup for the
//! future BVP pipeline:
//! `Vec<Expr> -> Atom/View -> derivative -> CodegenIR`.

#[cfg(test)]
mod tests {
    use std::{
        fs,
        hint::black_box,
        path::PathBuf,
        time::{Duration, Instant},
    };

    use crate::symbolic::{
        View::{
            Atom,
            CodegenIR_atom::BlockDagStats,
            CodegenIR_atom::Lowerer as AtomLowerer,
            bvp::discretization_system_bvp_par_atom,
            bvp_codegen::prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown,
            conversions::{atom_to_expr, expr_to_atom},
            jacobian::PreparedSparseAtomSystem,
            state::Symbol,
        },
        codegen::CodegenIR::AtomTempReusePolicy,
        codegen::CodegenIR::GeneratedBlock,
        codegen::CodegenIR::Lowerer as ExprLowerer,
        codegen::codegen_backend_selection::BackendSelectionPolicy,
        codegen::codegen_tasks::CodegenOutputLayout,
        codegen::rust_backend::codegen_aot_build::{
            AotBuildProfile, AotBuildRequest, AotCompileConfig, AotOptimizationLevel,
        },
        symbolic_engine::Expr,
        symbolic_functions_BVP::{BvpSymbolicAssemblyBackend, Jacobian},
    };

    const PERF_REPEATS: usize = 9;
    const SINGLE_EXPR_ITERS: usize = 2_000;
    const JACOBIAN_ITERS: usize = 160;
    // This benchmark targets a real discretized BVP and is intended primarily
    // for release runs. Keep the default debug workload moderate so the test
    // stays runnable while still exercising a realistic sparse Jacobian path.
    const REAL_BVP_STEPS: usize = 100;
    const REAL_BVP_ITERS: usize = 2;
    const XL_BVP_STEPS: usize = 1000;

    fn median_duration<F>(mut f: F) -> Duration
    where
        F: FnMut(),
    {
        let mut samples = Vec::with_capacity(PERF_REPEATS);
        for _ in 0..PERF_REPEATS {
            let start = Instant::now();
            f();
            samples.push(start.elapsed());
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn speedup_vs(base: Duration, candidate: Duration) -> f64 {
        base.as_secs_f64() / candidate.as_secs_f64()
    }

    fn duration_stats<F>(mut f: F) -> (Duration, Duration, Duration)
    where
        F: FnMut(),
    {
        let mut samples = Vec::with_capacity(PERF_REPEATS);
        for _ in 0..PERF_REPEATS {
            let start = Instant::now();
            f();
            samples.push(start.elapsed());
        }
        samples.sort_unstable();
        (
            samples[0],
            samples[samples.len() / 2],
            *samples.last().unwrap(),
        )
    }

    fn optional_aot_toolchain() -> Option<String> {
        std::env::var("RUSTEDSCITHE_AOT_TOOLCHAIN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn optional_aot_rustflags() -> Option<String> {
        std::env::var("RUSTEDSCITHE_AOT_RUSTFLAGS")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn optional_aot_opt_level() -> Option<AotOptimizationLevel> {
        match std::env::var("RUSTEDSCITHE_AOT_OPT_LEVEL")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("0") => Some(AotOptimizationLevel::O0),
            Some("1") => Some(AotOptimizationLevel::O1),
            Some("2") => Some(AotOptimizationLevel::O2),
            Some("3") => Some(AotOptimizationLevel::O3),
            Some("s") => Some(AotOptimizationLevel::Os),
            Some("z") => Some(AotOptimizationLevel::Oz),
            Some("default") | None => None,
            Some(other) => panic!("unsupported RUSTEDSCITHE_AOT_OPT_LEVEL value: {other}"),
        }
    }

    fn optional_aot_codegen_units() -> Option<usize> {
        std::env::var("RUSTEDSCITHE_AOT_CODEGEN_UNITS")
            .ok()
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .expect("RUSTEDSCITHE_AOT_CODEGEN_UNITS must be a positive integer")
            })
    }

    fn optional_aot_compile_config() -> Option<AotCompileConfig> {
        let opt_level = optional_aot_opt_level();
        let codegen_units = optional_aot_codegen_units();
        if opt_level.is_none() && codegen_units.is_none() {
            return None;
        }
        let mut config = AotCompileConfig::new();
        if let Some(level) = opt_level {
            config = config.with_optimization(level);
        }
        if let Some(units) = codegen_units {
            config = config.with_codegen_units(units);
        }
        Some(config)
    }

    fn compile_config_label(config: &AotCompileConfig) -> String {
        config.label()
    }

    fn configured_aot_build_request(
        crate_spec: crate::symbolic::codegen::rust_backend::codegen_aot_crate::GeneratedAotCrate,
        output_parent_dir: impl Into<PathBuf>,
        profile: AotBuildProfile,
    ) -> AotBuildRequest {
        let mut request = AotBuildRequest::new(crate_spec, output_parent_dir, profile);
        if let Some(toolchain) = optional_aot_toolchain() {
            request = request.with_toolchain(toolchain);
        }
        if let Some(rustflags) = optional_aot_rustflags() {
            request = request.with_rustflags(rustflags);
        }
        if let Some(config) = optional_aot_compile_config() {
            request = request.with_compile_config(config);
        }
        let cargo_timings_enabled = std::env::var("RUSTEDSCITHE_AOT_CARGO_TIMINGS")
            .ok()
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if cargo_timings_enabled {
            request = request.with_extra_cargo_arg("--timings");
        }
        request
    }

    fn eval_expr(expr: &Expr, vars: &[&str], values: &[f64]) -> f64 {
        let f = expr.lambdify_borrowed_thread_safe(vars);
        f(values)
    }

    fn assert_single_derivative_equivalent(expr: &Expr, var: &str, view_var: Symbol) {
        let classic = expr.diff(var).simplify();
        let view = atom_to_expr(&expr_to_atom(expr).derivative(view_var)).simplify();
        let probe_vars = ["x", "y", "z", "w"];
        let probe_values = [1.25, 0.4, -0.2, 0.75];
        let classic_eval = eval_expr(&classic, &probe_vars, &probe_values);
        let view_eval = eval_expr(&view, &probe_vars, &probe_values);
        assert!(
            (classic_eval - view_eval).abs() < 1e-10,
            "Expr/View derivatives disagree: {} vs {}",
            classic_eval,
            view_eval
        );
    }

    fn assert_jacobian_equivalent(
        exprs: &[Expr],
        vars: &[&str],
        view_vars: &[Symbol],
        atoms: &[Atom],
    ) {
        let probe_vars = ["y0", "y1", "y2", "y3", "y4", "y5"];
        let probe_values = [0.8, 0.4, 0.6, 0.2, 0.7, 0.3];

        for (expr, atom) in exprs.iter().zip(atoms.iter()) {
            for (var, view_var) in vars.iter().zip(view_vars.iter()) {
                let classic = expr.diff(var).simplify();
                let view = atom_to_expr(&atom.derivative(*view_var)).simplify();
                let classic_eval = eval_expr(&classic, &probe_vars, &probe_values);
                let view_eval = eval_expr(&view, &probe_vars, &probe_values);
                assert!(
                    (classic_eval - view_eval).abs() < 1e-9,
                    "Expr/View Jacobian entry disagrees for d()/d{}: {} vs {}",
                    var,
                    classic_eval,
                    view_eval
                );
            }
        }
    }

    fn parse_expr(input: &str) -> Expr {
        Expr::parse_expression(input)
    }

    fn single_expression_fixture() -> Expr {
        parse_expr("((x+y)^3+sin(x*y)+exp(x-z))*(x^2+y^2+z^2)+log(x+2)*(y+z)^4+w*x^3")
    }

    fn bvp_rhs_fixture() -> (Vec<Expr>, Vec<&'static str>) {
        let rhs = vec![
            parse_expr("-0.5*y0 + sin(y1) + exp(y2) - (y3+y4)^2 + log(y5+2)"),
            parse_expr("y0*y1 + y2^3 - cos(y3) + (y4+1)^2/(y5+2)"),
            parse_expr("exp(y0-y2) + y1^2*y3 - log(y4+2) + y5^3"),
            parse_expr("sin(y0*y4) + cos(y1+y5) + y2*y3^2 - y4"),
            parse_expr("(y0+y1+y2)^2 + exp(y3) - y4*y5 + log(y2+3)"),
            parse_expr("y0^2*y5 + sin(y1*y2) - cos(y3*y4) + exp(y5-y0)"),
        ];
        let vars = vec!["y0", "y1", "y2", "y3", "y4", "y5"];
        (rhs, vars)
    }

    fn combustion_bvp_fixture(
        n_steps: usize,
    ) -> (
        Vec<Expr>,
        Vec<String>,
        std::collections::HashMap<String, Vec<(usize, f64)>>,
    ) {
        use std::collections::HashMap;

        let unknowns_str: Vec<&str> = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
        let unknowns: Vec<Expr> = Expr::parse_vector_expression(unknowns_str.clone());
        let teta = unknowns[0].clone();
        let q = unknowns[1].clone();
        let c0 = unknowns[2].clone();
        let j0 = unknowns[3].clone();
        let j1 = unknowns[5].clone();

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
        let ro_d = vec![Expr::Const(d_ro), Expr::Const(d_ro)];
        let pe_d = vec![Expr::Const(pe_d), Expr::Const(pe_d)];
        let minus = Expr::Const(-1.0);
        let m_reag = Expr::Const(0.342);

        let rate = a
            * Expr::exp(-e / (r_g * (teta.clone() * t_scale_sym + dt_sym)))
            * c0.clone()
            * (ro_m.clone() / m_reag.clone());
        let eq_t = q.clone() / lambda_sym;
        let eq_q = q * pe_q_sym - q_heat * rate.clone() * qm;
        let eq_c0 = j0.clone() / ro_d[0].clone();
        let eq_j0 = j0 * pe_d[0].clone()
            - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone();
        let eq_c1 = j1.clone() / ro_d[1].clone();
        let eq_j1 = j1 * pe_d[1].clone() - (m.clone() * rate * ro_m / m) * qs;
        let eqs = vec![eq_t, eq_q, eq_c0, eq_j0, eq_c1, eq_j1];

        let boundary_conditions = HashMap::from([
            ("Teta".to_string(), vec![(0, (t_initial - dt) / t_scale)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(), vec![(0, c1_0)]),
            ("J0".to_string(), vec![(1, 1e-7)]),
            ("C1".to_string(), vec![(0, 1e-3)]),
            ("J1".to_string(), vec![(1, 1e-7)]),
        ]);

        let values = unknowns_str
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let _ = n_steps;
        (eqs, values, boundary_conditions)
    }

    fn unique_microbench_artifact_dir(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-artifacts")
            .join("view-microbench")
            .join(format!("{label}-{}-{nonce}", std::process::id()))
    }

    fn build_real_combustion_discretized_bvp(
        n_steps: usize,
    ) -> (
        Vec<Expr>,
        Vec<String>,
        Vec<Vec<String>>,
        Option<(usize, usize)>,
    ) {
        let (eqs, values, boundary_conditions) = combustion_bvp_fixture(n_steps);
        let mut jac = Jacobian::new();
        jac.discretization_system_BVP_par(
            eqs,
            values,
            "x".to_string(),
            0.0,
            Some(n_steps),
            None,
            None,
            boundary_conditions,
            None,
            None,
            "trapezoid".to_string(),
        );
        (
            jac.vector_of_functions,
            jac.variable_string,
            jac.variables_for_all_disrete,
            jac.bandwidth,
        )
    }

    fn estimate_bandwidth_from_variable_usage(
        variable_names: &[String],
        variables_for_all_discrete: &[Vec<String>],
    ) -> Option<(usize, usize)> {
        use ahash::HashMap;

        let index_by_name: HashMap<&str, usize> = variable_names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.as_str(), idx))
            .collect();

        let mut lower = 0usize;
        let mut upper = 0usize;
        let mut saw_any = false;

        for (row_idx, vars) in variables_for_all_discrete.iter().enumerate() {
            for var in vars {
                if let Some(&col_idx) = index_by_name.get(var.as_str()) {
                    saw_any = true;
                    if row_idx >= col_idx {
                        lower = lower.max(row_idx - col_idx);
                    } else {
                        upper = upper.max(col_idx - row_idx);
                    }
                }
            }
        }

        saw_any.then_some((lower, upper))
    }

    fn classic_sparse_jacobian_entries_expr_with_bandwidth(
        functions: &[Expr],
        variable_names: &[String],
        variables_for_all_discrete: &[Vec<String>],
        bandwidth: Option<(usize, usize)>,
    ) -> Vec<(usize, usize, Expr)> {
        use ahash::HashSet;

        let variable_sets: Vec<HashSet<&String>> = variables_for_all_discrete
            .iter()
            .map(|vars| vars.iter().collect())
            .collect();

        let mut sparse_entries = Vec::new();
        for (i, function) in functions.iter().enumerate() {
            let (left, right) = if let Some((kl, ku)) = bandwidth {
                let right = std::cmp::min(i + ku + 1, variable_names.len());
                let left = if i as i32 - (kl as i32) - 1 < 0 {
                    0
                } else {
                    i - kl - 1
                };
                (left, right)
            } else {
                (0, variable_names.len())
            };
            for (j, variable) in variable_names
                .iter()
                .enumerate()
                .skip(left)
                .take(right - left)
            {
                if variable_sets[i].contains(variable) {
                    let partial = function.diff(variable);
                    if !partial.is_zero() {
                        sparse_entries.push((i, j, partial));
                    }
                }
            }
        }
        sparse_entries
    }

    #[test]
    #[ignore = "microbenchmark for Expr vs View single-expression differentiation"]
    fn view_single_derivative_microbenchmark_vs_expr() {
        let expr = single_expression_fixture();
        let atom = expr_to_atom(&expr);
        let view_var = Symbol::new(crate::wrap_symbol!("x"));
        assert_single_derivative_equivalent(&expr, "x", view_var);

        let classic_diff = median_duration(|| {
            for _ in 0..SINGLE_EXPR_ITERS {
                black_box(expr.diff("x"));
            }
        });

        let classic_diff_simplify = median_duration(|| {
            for _ in 0..SINGLE_EXPR_ITERS {
                black_box(expr.diff("x").simplify());
            }
        });

        let view_diff = median_duration(|| {
            for _ in 0..SINGLE_EXPR_ITERS {
                black_box(atom.derivative(view_var));
            }
        });

        println!(
            "[View microbench] single-expression derivative, iters={}",
            SINGLE_EXPR_ITERS
        );
        println!("expr_len={}", expr.to_string().len());
        println!(
            "classic diff only:           {:>10.3} ms",
            classic_diff.as_secs_f64() * 1_000.0
        );
        println!(
            "classic diff + simplify:     {:>10.3} ms",
            classic_diff_simplify.as_secs_f64() * 1_000.0
        );
        println!(
            "view atom derivative:        {:>10.3} ms",
            view_diff.as_secs_f64() * 1_000.0
        );
        println!(
            "speedup vs classic diff:     {:>10.3}x",
            speedup_vs(classic_diff, view_diff)
        );
        println!(
            "speedup vs diff+simplify:    {:>10.3}x",
            speedup_vs(classic_diff_simplify, view_diff)
        );
    }

    #[test]
    #[ignore = "microbenchmark for Expr vs View Jacobian-workload differentiation"]
    fn view_bvp_jacobian_microbenchmark_vs_expr() {
        let (rhs, vars) = bvp_rhs_fixture();
        let atoms: Vec<_> = rhs.iter().map(expr_to_atom).collect();
        let view_vars: Vec<_> = vars
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(*name)))
            .collect();
        assert_jacobian_equivalent(&rhs, &vars, &view_vars, &atoms);

        let total_derivatives = rhs.len() * vars.len() * JACOBIAN_ITERS;

        let classic_diff = median_duration(|| {
            for _ in 0..JACOBIAN_ITERS {
                for expr in &rhs {
                    for var in &vars {
                        black_box(expr.diff(var));
                    }
                }
            }
        });

        let classic_diff_simplify = median_duration(|| {
            for _ in 0..JACOBIAN_ITERS {
                for expr in &rhs {
                    for var in &vars {
                        black_box(expr.diff(var).simplify());
                    }
                }
            }
        });

        let view_diff = median_duration(|| {
            for _ in 0..JACOBIAN_ITERS {
                for atom in &atoms {
                    for view_var in &view_vars {
                        black_box(atom.derivative(*view_var));
                    }
                }
            }
        });

        let classic_ns_per_deriv =
            classic_diff.as_secs_f64() * 1_000_000_000.0 / total_derivatives as f64;
        let classic_simplified_ns_per_deriv =
            classic_diff_simplify.as_secs_f64() * 1_000_000_000.0 / total_derivatives as f64;
        let view_ns_per_deriv =
            view_diff.as_secs_f64() * 1_000_000_000.0 / total_derivatives as f64;

        println!(
            "[View microbench] BVP-style Jacobian workload, rhs={}, vars={}, iters={}, derivatives={}",
            rhs.len(),
            vars.len(),
            JACOBIAN_ITERS,
            total_derivatives
        );
        println!(
            "classic diff only total:         {:>10.3} ms   ({:>10.1} ns/deriv)",
            classic_diff.as_secs_f64() * 1_000.0,
            classic_ns_per_deriv
        );
        println!(
            "classic diff + simplify total:   {:>10.3} ms   ({:>10.1} ns/deriv)",
            classic_diff_simplify.as_secs_f64() * 1_000.0,
            classic_simplified_ns_per_deriv
        );
        println!(
            "view atom derivative total:      {:>10.3} ms   ({:>10.1} ns/deriv)",
            view_diff.as_secs_f64() * 1_000.0,
            view_ns_per_deriv
        );
        println!(
            "speedup vs classic diff:         {:>10.3}x",
            speedup_vs(classic_diff, view_diff)
        );
        println!(
            "speedup vs diff+simplify:        {:>10.3}x",
            speedup_vs(classic_diff_simplify, view_diff)
        );
    }

    #[test]
    #[ignore = "microbenchmark for real combustion BVP discretized Jacobian and lowering"]
    fn view_real_combustion_bvp_jacobian_and_lowering_microbenchmark() {
        let (functions, variable_names, variables_for_all_discrete, bandwidth) =
            build_real_combustion_discretized_bvp(REAL_BVP_STEPS);
        let effective_bandwidth = bandwidth.or_else(|| {
            estimate_bandwidth_from_variable_usage(&variable_names, &variables_for_all_discrete)
        });
        let total_entries: usize = variables_for_all_discrete
            .iter()
            .map(|vars| vars.len())
            .sum();
        let var_refs: Vec<&str> = variable_names.iter().map(|name| name.as_str()).collect();
        let prepared = PreparedSparseAtomSystem::from_exprs(
            &functions,
            &variable_names,
            &variables_for_all_discrete,
        );
        let view_vars = prepared.variable_symbols().to_vec();

        let classic_sparse = classic_sparse_jacobian_entries_expr_with_bandwidth(
            &functions,
            &variable_names,
            &variables_for_all_discrete,
            effective_bandwidth,
        );
        let atom_sparse = prepared.calc_sparse_jacobian_with_bandwidth(effective_bandwidth);

        assert_eq!(
            classic_sparse.len(),
            atom_sparse.len(),
            "classic and Atom sparse Jacobian entry counts differ"
        );

        let classic_diff = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                black_box(classic_sparse_jacobian_entries_expr_with_bandwidth(
                    &functions,
                    &variable_names,
                    &variables_for_all_discrete,
                    effective_bandwidth,
                ));
            }
        });

        let atom_diff = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                black_box(prepared.calc_sparse_jacobian_with_bandwidth(effective_bandwidth));
            }
        });

        let classic_lower = median_duration(|| {
            let sparse_exprs: Vec<_> = classic_sparse
                .iter()
                .map(|(_, _, expr)| expr.clone())
                .collect();
            for _ in 0..REAL_BVP_ITERS {
                black_box(ExprLowerer::new(&var_refs).lower_many(&sparse_exprs));
            }
        });

        let atom_lower = median_duration(|| {
            let sparse_views: Vec<_> = atom_sparse
                .iter()
                .map(|entry| entry.value.as_view())
                .collect();
            for _ in 0..REAL_BVP_ITERS {
                black_box(AtomLowerer::new(&view_vars).lower_many(&sparse_views));
            }
        });

        let classic_full = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let sparse = classic_sparse_jacobian_entries_expr_with_bandwidth(
                    &functions,
                    &variable_names,
                    &variables_for_all_discrete,
                    effective_bandwidth,
                );
                let exprs: Vec<_> = sparse.into_iter().map(|(_, _, expr)| expr).collect();
                black_box(ExprLowerer::new(&var_refs).lower_many(&exprs));
            }
        });

        let atom_full = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let sparse = prepared.calc_sparse_jacobian_with_bandwidth(effective_bandwidth);
                let views: Vec<_> = sparse.iter().map(|entry| entry.value.as_view()).collect();
                black_box(AtomLowerer::new(&view_vars).lower_many(&views));
            }
        });

        println!(
            "[View microbench] real combustion BVP Jacobian, steps={}, residuals={}, vars={}, approx_derivatives={}, iters={}",
            REAL_BVP_STEPS,
            functions.len(),
            variable_names.len(),
            total_entries,
            REAL_BVP_ITERS
        );
        println!("bandwidth={bandwidth:?}, effective_bandwidth={effective_bandwidth:?}");
        println!(
            "classic sparse Jacobian diff:    {:>10.3} ms",
            classic_diff.as_secs_f64() * 1_000.0
        );
        println!(
            "atom sparse Jacobian diff:       {:>10.3} ms   speedup {:>8.3}x",
            atom_diff.as_secs_f64() * 1_000.0,
            speedup_vs(classic_diff, atom_diff)
        );
        println!(
            "classic lowering only:           {:>10.3} ms",
            classic_lower.as_secs_f64() * 1_000.0
        );
        println!(
            "atom lowering only:              {:>10.3} ms   speedup {:>8.3}x",
            atom_lower.as_secs_f64() * 1_000.0,
            speedup_vs(classic_lower, atom_lower)
        );
        println!(
            "classic diff + lowering:         {:>10.3} ms",
            classic_full.as_secs_f64() * 1_000.0
        );
        println!(
            "atom diff + lowering:            {:>10.3} ms   speedup {:>8.3}x",
            atom_full.as_secs_f64() * 1_000.0,
            speedup_vs(classic_full, atom_full)
        );
        println!("classic nnz estimate / iter:     {}", classic_sparse.len());
    }

    #[test]
    #[ignore = "microbenchmark comparing direct atom Jacobian with production BVP codegen helper"]
    fn view_real_combustion_bvp_codegen_helper_microbenchmark() {
        let (
            legacy_functions,
            legacy_variable_names,
            legacy_variables_for_all_discrete,
            legacy_bandwidth,
        ) = build_real_combustion_discretized_bvp(REAL_BVP_STEPS);
        let legacy_effective_bandwidth = legacy_bandwidth.or_else(|| {
            estimate_bandwidth_from_variable_usage(
                &legacy_variable_names,
                &legacy_variables_for_all_discrete,
            )
        });
        let legacy_var_refs: Vec<&str> = legacy_variable_names
            .iter()
            .map(|name| name.as_str())
            .collect();

        let (eqs, values, boundary_conditions) = combustion_bvp_fixture(REAL_BVP_STEPS);
        let atom_discretized = discretization_system_bvp_par_atom(
            eqs,
            values.clone(),
            "x".to_string(),
            0.0,
            Some(REAL_BVP_STEPS),
            None,
            None,
            boundary_conditions,
            None,
            None,
            "trapezoid".to_string(),
        );
        let effective_bandwidth = estimate_bandwidth_from_variable_usage(
            &atom_discretized.variable_string,
            &atom_discretized.variables_for_all_discrete,
        );

        let direct_prepared = PreparedSparseAtomSystem::from_atoms(
            &atom_discretized.vector_of_functions,
            &atom_discretized.variable_string,
            &atom_discretized.variables_for_all_discrete,
        );
        let view_vars = direct_prepared.variable_symbols().to_vec();
        let direct_sparse =
            direct_prepared.calc_sparse_jacobian_with_bandwidth(effective_bandwidth);

        let legacy_full = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let sparse = classic_sparse_jacobian_entries_expr_with_bandwidth(
                    &legacy_functions,
                    &legacy_variable_names,
                    &legacy_variables_for_all_discrete,
                    legacy_effective_bandwidth,
                );
                let exprs: Vec<_> = sparse.into_iter().map(|(_, _, expr)| expr).collect();
                black_box(ExprLowerer::new(&legacy_var_refs).lower_many(&exprs));
            }
        });

        let direct_jac = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                black_box(direct_prepared.calc_sparse_jacobian_with_bandwidth(effective_bandwidth));
            }
        });

        let direct_full = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let sparse =
                    direct_prepared.calc_sparse_jacobian_with_bandwidth(effective_bandwidth);
                let views: Vec<_> = sparse.iter().map(|entry| entry.value.as_view()).collect();
                black_box(AtomLowerer::new(&view_vars).lower_many(&views));
            }
        });

        let helper_prepare = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let (_prepared, breakdown) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                    &atom_discretized,
                    "eval_bvp_residual",
                    "eval_bvp_sparse_values",
                    Vec::new(),
                    effective_bandwidth,
                    crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                        max_outputs_per_chunk: 192,
                    },
                    crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                        max_entries_per_chunk: 256,
                    },
                );
                black_box(breakdown);
            }
        });

        let helper_full = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let (prepared, breakdown) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                    &atom_discretized,
                    "eval_bvp_residual",
                    "eval_bvp_sparse_values",
                    Vec::new(),
                    effective_bandwidth,
                    crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                        max_outputs_per_chunk: 192,
                    },
                    crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                        max_entries_per_chunk: 256,
                    },
                );
                let (module, module_breakdown) =
                    prepared.codegen_module_with_breakdown("microbench_generated_atom_bvp");
                black_box((
                    breakdown,
                    module_breakdown,
                    module.block_count(),
                    module.total_block_instruction_count(),
                ));
            }
        });

        let helper_module_never = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let (prepared, _) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                    &atom_discretized,
                    "eval_bvp_residual",
                    "eval_bvp_sparse_values",
                    Vec::new(),
                    effective_bandwidth,
                    crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                        max_outputs_per_chunk: 192,
                    },
                    crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                        max_entries_per_chunk: 256,
                    },
                );
                let (module, module_breakdown) = prepared
                    .codegen_module_with_breakdown_and_reuse_policy(
                        "microbench_generated_atom_bvp_never",
                        AtomTempReusePolicy::Never,
                    );
                black_box((
                    module_breakdown,
                    module.block_count(),
                    module.total_block_instruction_count(),
                ));
            }
        });

        let helper_module_auto = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let (prepared, _) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                    &atom_discretized,
                    "eval_bvp_residual",
                    "eval_bvp_sparse_values",
                    Vec::new(),
                    effective_bandwidth,
                    crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                        max_outputs_per_chunk: 192,
                    },
                    crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                        max_entries_per_chunk: 256,
                    },
                );
                let (module, module_breakdown) = prepared
                    .codegen_module_with_breakdown_and_reuse_policy(
                        "microbench_generated_atom_bvp_auto",
                        AtomTempReusePolicy::Auto,
                    );
                black_box((
                    module_breakdown,
                    module.block_count(),
                    module.total_block_instruction_count(),
                ));
            }
        });

        let helper_module_always = median_duration(|| {
            for _ in 0..REAL_BVP_ITERS {
                let (prepared, _) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                    &atom_discretized,
                    "eval_bvp_residual",
                    "eval_bvp_sparse_values",
                    Vec::new(),
                    effective_bandwidth,
                    crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                        max_outputs_per_chunk: 192,
                    },
                    crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                        max_entries_per_chunk: 256,
                    },
                );
                let (module, module_breakdown) = prepared
                    .codegen_module_with_breakdown_and_reuse_policy(
                        "microbench_generated_atom_bvp_always",
                        AtomTempReusePolicy::Always,
                    );
                black_box((
                    module_breakdown,
                    module.block_count(),
                    module.total_block_instruction_count(),
                ));
            }
        });

        let never_stats = duration_stats(|| {
            let (prepared, _) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                &atom_discretized,
                "eval_bvp_residual",
                "eval_bvp_sparse_values",
                Vec::new(),
                effective_bandwidth,
                crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                    max_outputs_per_chunk: 192,
                },
                crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                    max_entries_per_chunk: 256,
                },
            );
            let (module, module_breakdown) = prepared
                .codegen_module_with_breakdown_and_reuse_policy(
                    "microbench_generated_atom_bvp_never_stats",
                    AtomTempReusePolicy::Never,
                );
            black_box((
                module_breakdown,
                module.block_count(),
                module.total_block_instruction_count(),
            ));
        });

        let auto_stats = duration_stats(|| {
            let (prepared, _) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                &atom_discretized,
                "eval_bvp_residual",
                "eval_bvp_sparse_values",
                Vec::new(),
                effective_bandwidth,
                crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                    max_outputs_per_chunk: 192,
                },
                crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                    max_entries_per_chunk: 256,
                },
            );
            let (module, module_breakdown) = prepared
                .codegen_module_with_breakdown_and_reuse_policy(
                    "microbench_generated_atom_bvp_auto_stats",
                    AtomTempReusePolicy::Auto,
                );
            black_box((
                module_breakdown,
                module.block_count(),
                module.total_block_instruction_count(),
            ));
        });

        let always_stats = duration_stats(|| {
            let (prepared, _) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
                &atom_discretized,
                "eval_bvp_residual",
                "eval_bvp_sparse_values",
                Vec::new(),
                effective_bandwidth,
                crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                    max_outputs_per_chunk: 192,
                },
                crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                    max_entries_per_chunk: 256,
                },
            );
            let (module, module_breakdown) = prepared
                .codegen_module_with_breakdown_and_reuse_policy(
                    "microbench_generated_atom_bvp_always_stats",
                    AtomTempReusePolicy::Always,
                );
            black_box((
                module_breakdown,
                module.block_count(),
                module.total_block_instruction_count(),
            ));
        });

        let residual_views = atom_discretized
            .vector_of_functions
            .iter()
            .map(|atom| atom.as_view())
            .collect::<Vec<_>>();
        let residual_input_names = atom_discretized.variable_string.clone();
        let residual_input_symbols = residual_input_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();

        let residual_policy_never = duration_stats(|| {
            let (block, breakdown) =
                GeneratedBlock::from_atom_views_with_symbols_with_breakdown_and_reuse_policy(
                    "microbench_generated_atom_residual_never",
                    &residual_views,
                    &residual_input_names,
                    &residual_input_symbols,
                    Some(CodegenOutputLayout::Vector {
                        len: residual_views.len(),
                    }),
                    AtomTempReusePolicy::Never,
                );
            black_box((breakdown, block.ir.num_temps, block.ir.instructions.len()));
        });

        let residual_policy_auto = duration_stats(|| {
            let (block, breakdown) =
                GeneratedBlock::from_atom_views_with_symbols_with_breakdown_and_reuse_policy(
                    "microbench_generated_atom_residual_auto",
                    &residual_views,
                    &residual_input_names,
                    &residual_input_symbols,
                    Some(CodegenOutputLayout::Vector {
                        len: residual_views.len(),
                    }),
                    AtomTempReusePolicy::Auto,
                );
            black_box((breakdown, block.ir.num_temps, block.ir.instructions.len()));
        });

        let residual_policy_always = duration_stats(|| {
            let (block, breakdown) =
                GeneratedBlock::from_atom_views_with_symbols_with_breakdown_and_reuse_policy(
                    "microbench_generated_atom_residual_always",
                    &residual_views,
                    &residual_input_names,
                    &residual_input_symbols,
                    Some(CodegenOutputLayout::Vector {
                        len: residual_views.len(),
                    }),
                    AtomTempReusePolicy::Always,
                );
            black_box((breakdown, block.ir.num_temps, block.ir.instructions.len()));
        });

        let (one_prepared, one_breakdown) = prepare_sparse_bvp_codegen_from_discretized_system_with_breakdown(
            &atom_discretized,
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            Vec::new(),
            effective_bandwidth,
            crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 192,
            },
            crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: 256,
            },
        );

        println!(
            "[View microbench] combustion production helper compare, steps={}, residuals={}, vars={}, iters={}",
            REAL_BVP_STEPS,
            atom_discretized.vector_of_functions.len(),
            atom_discretized.variable_string.len(),
            REAL_BVP_ITERS
        );
        println!(
            "effective_bandwidth={effective_bandwidth:?}, direct_nnz={}, helper_nnz={}",
            direct_sparse.len(),
            one_breakdown.sparse_nnz
        );
        println!(
            "direct atom sparse Jacobian:     {:>10.3} ms",
            direct_jac.as_secs_f64() * 1_000.0
        );
        println!(
            "legacy jacobian + lowering:      {:>10.3} ms",
            legacy_full.as_secs_f64() * 1_000.0
        );
        println!(
            "atom direct jacobian + lowering: {:>10.3} ms   speedup {:>8.3}x",
            direct_full.as_secs_f64() * 1_000.0,
            speedup_vs(legacy_full, direct_full)
        );
        println!(
            "helper prepare total:            {:>10.3} ms   slowdown {:>8.3}x",
            helper_prepare.as_secs_f64() * 1_000.0,
            speedup_vs(direct_jac, helper_prepare)
        );
        println!(
            "helper full + module:            {:>10.3} ms   slowdown {:>8.3}x",
            helper_full.as_secs_f64() * 1_000.0,
            speedup_vs(direct_jac, helper_full)
        );
        println!(
            "module reuse policy:             never={:>8.3} ms  auto={:>8.3} ms  always={:>8.3} ms",
            helper_module_never.as_secs_f64() * 1_000.0,
            helper_module_auto.as_secs_f64() * 1_000.0,
            helper_module_always.as_secs_f64() * 1_000.0
        );
        println!(
            "module reuse stats (ms):         never[min/med/max]={:>7.3}/{:>7.3}/{:>7.3}  auto={:>7.3}/{:>7.3}/{:>7.3}  always={:>7.3}/{:>7.3}/{:>7.3}",
            never_stats.0.as_secs_f64() * 1_000.0,
            never_stats.1.as_secs_f64() * 1_000.0,
            never_stats.2.as_secs_f64() * 1_000.0,
            auto_stats.0.as_secs_f64() * 1_000.0,
            auto_stats.1.as_secs_f64() * 1_000.0,
            auto_stats.2.as_secs_f64() * 1_000.0,
            always_stats.0.as_secs_f64() * 1_000.0,
            always_stats.1.as_secs_f64() * 1_000.0,
            always_stats.2.as_secs_f64() * 1_000.0
        );
        println!(
            "residual-only policy (ms):       never[min/med/max]={:>7.3}/{:>7.3}/{:>7.3}  auto={:>7.3}/{:>7.3}/{:>7.3}  always={:>7.3}/{:>7.3}/{:>7.3}",
            residual_policy_never.0.as_secs_f64() * 1_000.0,
            residual_policy_never.1.as_secs_f64() * 1_000.0,
            residual_policy_never.2.as_secs_f64() * 1_000.0,
            residual_policy_auto.0.as_secs_f64() * 1_000.0,
            residual_policy_auto.1.as_secs_f64() * 1_000.0,
            residual_policy_auto.2.as_secs_f64() * 1_000.0,
            residual_policy_always.0.as_secs_f64() * 1_000.0,
            residual_policy_always.1.as_secs_f64() * 1_000.0,
            residual_policy_always.2.as_secs_f64() * 1_000.0
        );
        println!(
            "atom helper overhead vs direct:  {:>10.3} ms",
            (helper_prepare.as_secs_f64() - direct_jac.as_secs_f64()).max(0.0) * 1_000.0
        );
        println!(
            "atom module overhead vs direct:  {:>10.3} ms",
            (helper_full.as_secs_f64() - direct_full.as_secs_f64()).max(0.0) * 1_000.0
        );
        println!(
            "single-run helper breakdown: lookup={:.3} ms, jac={:.3} ms, finalize={:.3} ms",
            one_breakdown.sparse_lookup_prepare_ms,
            one_breakdown.sparse_jacobian_build_ms,
            one_breakdown.finalize_codegen_plan_ms
        );
        let (one_module, one_module_breakdown) =
            one_prepared.codegen_module_with_breakdown("microbench_generated_atom_bvp_once");
        let residual_dag_stats = BlockDagStats::for_views(&residual_views);
        let sparse_views_once = direct_sparse
            .iter()
            .map(|entry| entry.value.as_view())
            .collect::<Vec<_>>();
        let sparse_dag_stats = BlockDagStats::for_views(&sparse_views_once);
        println!(
            "single-run helper module: blocks={}, instr={}, temps={}",
            one_module.block_count(),
            one_module.total_block_instruction_count(),
            one_module.total_block_temp_count()
        );
        println!(
            "single-run DAG stats: residual(outputs={}, intern_calls={}, unique_nodes={}, exact_hits={}, content_hits={}, compression={:.3})",
            residual_dag_stats.outputs,
            residual_dag_stats.intern_calls,
            residual_dag_stats.unique_nodes,
            residual_dag_stats.exact_hits,
            residual_dag_stats.content_hits,
            if residual_dag_stats.intern_calls == 0 {
                1.0
            } else {
                residual_dag_stats.unique_nodes as f64 / residual_dag_stats.intern_calls as f64
            }
        );
        println!(
            "single-run DAG stats: sparse  (outputs={}, intern_calls={}, unique_nodes={}, exact_hits={}, content_hits={}, compression={:.3})",
            sparse_dag_stats.outputs,
            sparse_dag_stats.intern_calls,
            sparse_dag_stats.unique_nodes,
            sparse_dag_stats.exact_hits,
            sparse_dag_stats.content_hits,
            if sparse_dag_stats.intern_calls == 0 {
                1.0
            } else {
                sparse_dag_stats.unique_nodes as f64 / sparse_dag_stats.intern_calls as f64
            }
        );
        println!(
            "single-run module breakdown: res(view={:.3}, lower={:.3}, peephole={:.3}, reuse={:.3}, push={:.3}, reuse_blocks={}) ms",
            one_module_breakdown.residual_view_collect_ms,
            one_module_breakdown.residual_lower_many_ms,
            one_module_breakdown.residual_peephole_ms,
            one_module_breakdown.residual_reuse_temps_ms,
            one_module_breakdown.residual_push_ms,
            one_module_breakdown.residual_reuse_temps_blocks
        );
        println!(
            "single-run module breakdown: sp (view={:.3}, lower={:.3}, peephole={:.3}, reuse={:.3}, push={:.3}, reuse_blocks={}) ms",
            one_module_breakdown.sparse_view_collect_ms,
            one_module_breakdown.sparse_lower_many_ms,
            one_module_breakdown.sparse_peephole_ms,
            one_module_breakdown.sparse_reuse_temps_ms,
            one_module_breakdown.sparse_push_ms,
            one_module_breakdown.sparse_reuse_temps_blocks
        );
    }

    #[test]
    #[ignore = "one-to-one end-to-end ExprLegacy vs AtomView compare for large combustion BVP including AOT build"]
    fn view_real_combustion_bvp_end_to_end_compare_1000() {
        let residual_chunking =
            crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 192,
            };
        let sparse_chunking =
            crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: 256,
            };

        let legacy_discretization_begin = Instant::now();
        let (
            legacy_functions,
            legacy_variable_names,
            legacy_variables_for_all_discrete,
            legacy_bandwidth,
        ) = build_real_combustion_discretized_bvp(XL_BVP_STEPS);
        let legacy_discretization_ms =
            legacy_discretization_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_effective_bandwidth = legacy_bandwidth.or_else(|| {
            estimate_bandwidth_from_variable_usage(
                &legacy_variable_names,
                &legacy_variables_for_all_discrete,
            )
        });
        let legacy_var_refs: Vec<&str> = legacy_variable_names
            .iter()
            .map(|name| name.as_str())
            .collect();

        let (eqs, values, boundary_conditions) = combustion_bvp_fixture(XL_BVP_STEPS);
        let atom_discretization_begin = Instant::now();
        let atom_discretized = discretization_system_bvp_par_atom(
            eqs.clone(),
            values.clone(),
            "x".to_string(),
            0.0,
            Some(XL_BVP_STEPS),
            None,
            None,
            boundary_conditions.clone(),
            None,
            None,
            "trapezoid".to_string(),
        );
        let atom_discretization_ms = atom_discretization_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_effective_bandwidth = estimate_bandwidth_from_variable_usage(
            &atom_discretized.variable_string,
            &atom_discretized.variables_for_all_discrete,
        );

        let legacy_jacobian_begin = Instant::now();
        let legacy_sparse = classic_sparse_jacobian_entries_expr_with_bandwidth(
            &legacy_functions,
            &legacy_variable_names,
            &legacy_variables_for_all_discrete,
            legacy_effective_bandwidth,
        );
        let legacy_jacobian_ms = legacy_jacobian_begin.elapsed().as_secs_f64() * 1_000.0;

        let atom_prepared = PreparedSparseAtomSystem::from_atoms(
            &atom_discretized.vector_of_functions,
            &atom_discretized.variable_string,
            &atom_discretized.variables_for_all_discrete,
        );
        let atom_view_vars = atom_prepared.variable_symbols().to_vec();
        let atom_jacobian_begin = Instant::now();
        let atom_sparse =
            atom_prepared.calc_sparse_jacobian_with_bandwidth(atom_effective_bandwidth);
        let atom_jacobian_ms = atom_jacobian_begin.elapsed().as_secs_f64() * 1_000.0;

        let legacy_lower_begin = Instant::now();
        let legacy_sparse_exprs: Vec<_> = legacy_sparse
            .iter()
            .map(|(_, _, expr)| expr.clone())
            .collect();
        let legacy_ir = ExprLowerer::new(&legacy_var_refs).lower_many(&legacy_sparse_exprs);
        let legacy_lower_ms = legacy_lower_begin.elapsed().as_secs_f64() * 1_000.0;

        let atom_lower_begin = Instant::now();
        let atom_sparse_views: Vec<_> = atom_sparse
            .iter()
            .map(|entry| entry.value.as_view())
            .collect();
        let atom_ir = AtomLowerer::new(&atom_view_vars).lower_many(&atom_sparse_views);
        let atom_lower_ms = atom_lower_begin.elapsed().as_secs_f64() * 1_000.0;

        let mut legacy_jacobian = Jacobian::new();
        legacy_jacobian.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
        let legacy_bundle_begin = Instant::now();
        let legacy_bundle = legacy_jacobian
            .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                eqs.clone(),
                values.clone(),
                "x".to_string(),
                None,
                0.0,
                None,
                Some(XL_BVP_STEPS),
                None,
                None,
                boundary_conditions.clone(),
                None,
                None,
                "trapezoid".to_string(),
                "Sparse".to_string(),
                None,
                BackendSelectionPolicy::AotOnly,
                None,
                residual_chunking,
                sparse_chunking,
            )
            .expect("legacy sparse bundle should build");
        let legacy_bundle_ms = legacy_bundle_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_prepared = legacy_bundle.execution.selected().prepared_problem.clone();

        let mut atom_jacobian_builder = Jacobian::new();
        atom_jacobian_builder.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
        let atom_bundle_begin = Instant::now();
        let atom_bundle = atom_jacobian_builder
            .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                eqs,
                values,
                "x".to_string(),
                None,
                0.0,
                None,
                Some(XL_BVP_STEPS),
                None,
                None,
                boundary_conditions,
                None,
                None,
                "trapezoid".to_string(),
                "Sparse".to_string(),
                None,
                BackendSelectionPolicy::AotOnly,
                None,
                residual_chunking,
                sparse_chunking,
            )
            .expect("atom sparse bundle should build");
        let atom_bundle_ms = atom_bundle_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_prepared_bridge = atom_bundle.execution.selected().prepared_problem.clone();

        let (legacy_crate, legacy_breakdown) = legacy_prepared.generated_aot_crate_with_breakdown(
            format!("microbench_bvp_expr_{}", legacy_prepared.problem_key()),
            "microbench_bvp_expr_module",
        );
        let (atom_crate, atom_breakdown) = atom_prepared_bridge.generated_aot_crate_with_breakdown(
            format!("microbench_bvp_atom_{}", atom_prepared_bridge.problem_key()),
            "microbench_bvp_atom_module",
        );

        let legacy_materialize_dir = unique_microbench_artifact_dir("expr-1000");
        let atom_materialize_dir = unique_microbench_artifact_dir("atom-1000");

        let legacy_materialize_begin = Instant::now();
        let legacy_materialized = configured_aot_build_request(
            legacy_crate,
            &legacy_materialize_dir,
            AotBuildProfile::Release,
        )
        .materialize()
        .expect("legacy AOT crate should materialize");
        let legacy_materialize_ms = legacy_materialize_begin.elapsed().as_secs_f64() * 1_000.0;

        let atom_materialize_begin = Instant::now();
        let atom_materialized = configured_aot_build_request(
            atom_crate,
            &atom_materialize_dir,
            AotBuildProfile::Release,
        )
        .materialize()
        .expect("atom AOT crate should materialize");
        let atom_materialize_ms = atom_materialize_begin.elapsed().as_secs_f64() * 1_000.0;

        let legacy_build_begin = Instant::now();
        let legacy_build = legacy_materialized
            .execute()
            .expect("legacy AOT build should execute");
        let legacy_build_ms = legacy_build_begin.elapsed().as_secs_f64() * 1_000.0;
        assert!(
            legacy_build.succeeded(),
            "legacy AOT build failed: {}",
            legacy_build.stderr
        );
        let legacy_rebuild_begin = Instant::now();
        let legacy_rebuild = legacy_materialized
            .execute()
            .expect("legacy warm AOT build should execute");
        let legacy_rebuild_ms = legacy_rebuild_begin.elapsed().as_secs_f64() * 1_000.0;
        assert!(
            legacy_rebuild.succeeded(),
            "legacy warm AOT build failed: {}",
            legacy_rebuild.stderr
        );

        let atom_build_begin = Instant::now();
        let atom_build = atom_materialized
            .execute()
            .expect("atom AOT build should execute");
        let atom_build_ms = atom_build_begin.elapsed().as_secs_f64() * 1_000.0;
        assert!(
            atom_build.succeeded(),
            "atom AOT build failed: {}",
            atom_build.stderr
        );
        let atom_rebuild_begin = Instant::now();
        let atom_rebuild = atom_materialized
            .execute()
            .expect("atom warm AOT build should execute");
        let atom_rebuild_ms = atom_rebuild_begin.elapsed().as_secs_f64() * 1_000.0;
        assert!(
            atom_rebuild.succeeded(),
            "atom warm AOT build failed: {}",
            atom_rebuild.stderr
        );

        println!("[View microbench] combustion 1000-step one-to-one ExprLegacy vs AtomView");
        if optional_aot_toolchain().is_some() || optional_aot_rustflags().is_some() {
            println!(
                "[View microbench] build override: toolchain={:?}, rustflags={:?}",
                optional_aot_toolchain(),
                optional_aot_rustflags()
            );
        }
        if let Some(config) = optional_aot_compile_config() {
            println!(
                "[View microbench] compile override: {}",
                compile_config_label(&config)
            );
        }
        println!(
            "backend      | discretization_ms | jacobian_ms | lower_ir_ms | bundle_ms | module_ms | source_ms | materialize_ms | build_ms | rebuild_ms | instr | temps | source_kb"
        );
        println!(
            "ExprLegacy   | {:>17.3} | {:>11.3} | {:>11.3} | {:>9.3} | {:>9.3} | {:>9.3} | {:>14.3} | {:>8.3} | {:>10.3} | {:>5} | {:>5} | {:>9.1}",
            legacy_discretization_ms,
            legacy_jacobian_ms,
            legacy_lower_ms,
            legacy_bundle_ms,
            legacy_breakdown.module_build_ms,
            legacy_breakdown.source_emit_ms,
            legacy_materialize_ms,
            legacy_build_ms,
            legacy_rebuild_ms,
            legacy_ir.instructions.len(),
            legacy_ir.num_temps,
            legacy_breakdown.source_kb
        );
        println!(
            "AtomView     | {:>17.3} | {:>11.3} | {:>11.3} | {:>9.3} | {:>9.3} | {:>9.3} | {:>14.3} | {:>8.3} | {:>10.3} | {:>5} | {:>5} | {:>9.1}",
            atom_discretization_ms,
            atom_jacobian_ms,
            atom_lower_ms,
            atom_bundle_ms,
            atom_breakdown.module_build_ms,
            atom_breakdown.source_emit_ms,
            atom_materialize_ms,
            atom_build_ms,
            atom_rebuild_ms,
            atom_ir.instructions.len(),
            atom_ir.num_temps,
            atom_breakdown.source_kb
        );
        println!(
            "[AtomView-only overhead] lookup_ms={:.3}, jac_build_ms={:.3}, finalize_ms={:.3}, res_view_ms={:.3}, res_lower_ms={:.3}, sp_view_ms={:.3}, sp_lower_ms={:.3}, nnz={}",
            atom_breakdown.atom_sparse_lookup_prepare_ms,
            atom_breakdown.atom_sparse_jacobian_build_ms,
            atom_breakdown.atom_finalize_codegen_plan_ms,
            atom_breakdown.atom_residual_view_collect_ms,
            atom_breakdown.atom_residual_lower_many_ms,
            atom_breakdown.atom_sparse_view_collect_ms,
            atom_breakdown.atom_sparse_lower_many_ms,
            atom_breakdown.atom_sparse_nnz
        );
        let cargo_timings_enabled = std::env::var("RUSTEDSCITHE_AOT_CARGO_TIMINGS")
            .ok()
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if cargo_timings_enabled {
            println!(
                "[View microbench] cargo timings directories: ExprLegacy={}, AtomView={}",
                legacy_materialized.artifact_dir.display(),
                atom_materialized.artifact_dir.display()
            );
        }

        if !cargo_timings_enabled {
            let _ = fs::remove_dir_all(legacy_materialize_dir);
            let _ = fs::remove_dir_all(atom_materialize_dir);
        }
    }

    #[test]
    #[ignore = "1000-step combustion generated module compile-config sweep for ExprLegacy vs AtomView"]
    fn view_real_combustion_bvp_build_config_sweep_1000() {
        let residual_chunking =
            crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 192,
            };
        let sparse_chunking =
            crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy::ByNonZeroCount {
                max_entries_per_chunk: 256,
            };

        let (eqs, values, boundary_conditions) = combustion_bvp_fixture(XL_BVP_STEPS);

        let mut legacy_jacobian = Jacobian::new();
        legacy_jacobian.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
        let legacy_bundle = legacy_jacobian
            .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                eqs.clone(),
                values.clone(),
                "x".to_string(),
                None,
                0.0,
                None,
                Some(XL_BVP_STEPS),
                None,
                None,
                boundary_conditions.clone(),
                None,
                None,
                "trapezoid".to_string(),
                "Sparse".to_string(),
                None,
                BackendSelectionPolicy::AotOnly,
                None,
                residual_chunking,
                sparse_chunking,
            )
            .expect("legacy sparse bundle should build");
        let legacy_prepared = legacy_bundle.execution.selected().prepared_problem.clone();

        let mut atom_jacobian_builder = Jacobian::new();
        atom_jacobian_builder.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
        let atom_bundle = atom_jacobian_builder
            .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                eqs,
                values,
                "x".to_string(),
                None,
                0.0,
                None,
                Some(XL_BVP_STEPS),
                None,
                None,
                boundary_conditions,
                None,
                None,
                "trapezoid".to_string(),
                "Sparse".to_string(),
                None,
                BackendSelectionPolicy::AotOnly,
                None,
                residual_chunking,
                sparse_chunking,
            )
            .expect("atom sparse bundle should build");
        let atom_prepared = atom_bundle.execution.selected().prepared_problem.clone();

        let configs = vec![
            ("default", AotCompileConfig::production()),
            (
                "O2-cgu16",
                AotCompileConfig::new()
                    .with_optimization(AotOptimizationLevel::O2)
                    .with_codegen_units(16),
            ),
            ("O1-cgu16", AotCompileConfig::fast_build()),
            ("O0-cgu16", AotCompileConfig::dev_fastest()),
        ];

        println!("[View microbench] combustion 1000-step build config sweep");
        println!(
            "backend      | config      | module_ms | source_ms | materialize_ms | build_ms | rebuild_ms | source_kb"
        );

        for (backend_label, prepared, crate_prefix) in [
            ("ExprLegacy", &legacy_prepared, "expr"),
            ("AtomView", &atom_prepared, "atom"),
        ] {
            for (config_label, compile_config) in &configs {
                let (crate_spec, breakdown) = prepared.generated_aot_crate_with_breakdown(
                    format!(
                        "microbench_bvp_{}_{}_{}",
                        crate_prefix,
                        config_label.to_ascii_lowercase().replace('-', "_"),
                        prepared.problem_key()
                    ),
                    &format!(
                        "microbench_bvp_{}_{}_module",
                        crate_prefix,
                        config_label.to_ascii_lowercase().replace('-', "_")
                    ),
                );
                let materialize_dir = unique_microbench_artifact_dir(&format!(
                    "{}-1000-{}",
                    crate_prefix,
                    config_label.to_ascii_lowercase().replace('-', "_")
                ));

                let materialize_begin = Instant::now();
                let build =
                    AotBuildRequest::new(crate_spec, &materialize_dir, AotBuildProfile::Release)
                        .with_compile_config(compile_config.clone())
                        .materialize()
                        .expect("compile-config sweep AOT crate should materialize");
                let materialize_ms = materialize_begin.elapsed().as_secs_f64() * 1_000.0;

                let build_begin = Instant::now();
                let executed = build
                    .execute()
                    .expect("compile-config sweep build should execute");
                let build_ms = build_begin.elapsed().as_secs_f64() * 1_000.0;
                assert!(
                    executed.succeeded(),
                    "{backend_label} {config_label} build failed: {}",
                    executed.stderr
                );

                let rebuild_begin = Instant::now();
                let rebuilt = build
                    .execute()
                    .expect("compile-config sweep rebuild should execute");
                let rebuild_ms = rebuild_begin.elapsed().as_secs_f64() * 1_000.0;
                assert!(
                    rebuilt.succeeded(),
                    "{backend_label} {config_label} rebuild failed: {}",
                    rebuilt.stderr
                );

                println!(
                    "{:<12} | {:<11} | {:>9.3} | {:>9.3} | {:>14.3} | {:>8.3} | {:>10.3} | {:>9.1}",
                    backend_label,
                    config_label,
                    breakdown.module_build_ms,
                    breakdown.source_emit_ms,
                    materialize_ms,
                    build_ms,
                    rebuild_ms,
                    breakdown.source_kb
                );

                let _ = fs::remove_dir_all(materialize_dir);
            }
        }
    }
}
