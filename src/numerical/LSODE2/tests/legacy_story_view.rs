#[test]
#[ignore = "release Lambdify regression gate: historical AtomView versus AtomViewExprCompat"]
fn lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story",
    );
    let config = combustion_like_story_base_config();
    let parameters = config.equation_parameters.as_deref();
    let parameter_values = config.equation_parameter_values.clone();
    let state = config.y0.clone();
    let repetitions = std::env::var("LSODE2_ATOMVIEW_REGRESSION_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(20);
    let warm_repetitions = repetitions.saturating_mul(1_000).max(1_000);

    println!(
        "[LSODE2 AtomView regression] historical HEAD adapter versus current ExprLegacy/AtomViewExprCompat; fixture=combustion-like; repetitions={repetitions}; callback_measurement_repetitions={warm_repetitions}; telemetry=off; AOT excluded"
    );
    println!(
        "matrix | route                 | prepare_ms | residual_ns/call | jacobian_ns/call | residual_diff | jacobian_diff"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------"
    );

    let historical_symbolic =
        legacy_atomview_lambdify::symbolic_jacobian(&config.eq_system, &config.values);
    let expr_symbolic = build_symbolic_jacobian(
        &config.eq_system,
        &config.values,
        IvpSymbolicAssemblyBackend::ExprLegacy,
        &IvpTelemetry::disabled(),
    );
    let compat_symbolic = build_symbolic_jacobian(
        &config.eq_system,
        &config.values,
        IvpSymbolicAssemblyBackend::AtomViewExprCompat,
        &IvpTelemetry::disabled(),
    );
    let scalar_names = story_jacobian_name_refs(&config);
    let scalar_args = story_jacobian_args(&config, &state);
    let historical_scalar = compile_story_scalar_jacobian(&historical_symbolic, &scalar_names);
    let expr_scalar = compile_story_scalar_jacobian(&expr_symbolic, &scalar_names);
    let compat_scalar = compile_story_scalar_jacobian(&compat_symbolic, &scalar_names);
    let historical_scalar_values = evaluate_story_scalar_jacobian(&historical_scalar, &scalar_args);
    let expr_scalar_values = evaluate_story_scalar_jacobian(&expr_scalar, &scalar_args);
    let compat_scalar_values = evaluate_story_scalar_jacobian(&compat_scalar, &scalar_args);
    let expr_scalar_diff =
        story_scalar_values_max_diff(&historical_scalar_values, &expr_scalar_values);
    let compat_scalar_diff =
        story_scalar_values_max_diff(&historical_scalar_values, &compat_scalar_values);
    assert!(expr_scalar_diff <= 1.0e-9);
    assert!(compat_scalar_diff <= 1.0e-9);

    println!(
        "[LSODE2 Jacobian scalar callback isolation] matrix assembly excluded; same prepared state and flattened args for every route"
    );
    println!("route                 | nonzero | scalar_eval_ns/call | max_diff_vs_historical");
    println!("--------------------------------------------------------------------------------");
    for (route, entries, values, max_diff) in [
        (
            "historical-AtomView",
            &historical_scalar,
            &historical_scalar_values,
            0.0_f64,
        ),
        (
            "ExprLegacy",
            &expr_scalar,
            &expr_scalar_values,
            expr_scalar_diff,
        ),
        (
            "AtomViewExprCompat",
            &compat_scalar,
            &compat_scalar_values,
            compat_scalar_diff,
        ),
    ] {
        let scalar_ns = measure_story_scalar_jacobian(entries, &scalar_args, warm_repetitions);
        println!(
            "{:<21} | {:>7} | {:>19.3} | {:>22.3e}",
            route,
            values.len(),
            scalar_ns,
            max_diff,
        );
    }

    for matrix in [BackendRaceMatrix::Sparse, BackendRaceMatrix::Banded] {
        let storage = match matrix {
            BackendRaceMatrix::Sparse => NativeJacobianStorage::SparseTriplets,
            BackendRaceMatrix::Banded => NativeJacobianStorage::Banded { bandwidth: None },
            BackendRaceMatrix::Dense => {
                unreachable!("the regression gate covers Sparse and Banded only")
            }
        };
        let legacy_started = Instant::now();
        let mut legacy = legacy_atomview_lambdify::prepare(
            &config.eq_system,
            &config.values,
            &config.arg,
            parameters,
            parameter_values.clone(),
            storage,
        );
        let legacy_prepare_ms = legacy_started.elapsed().as_secs_f64() * 1_000.0;

        let expr_started = Instant::now();
        let mut expr_options = SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::ExprLegacy)
            .with_equation_parameters(parameters.map(|values| values.to_vec()).unwrap_or_default())
            .with_telemetry(IvpTelemetry::disabled());
        if let Some(values) = parameter_values.clone() {
            expr_options = expr_options.with_equation_parameter_values(values);
        }
        let expr_residual = prepare_symbolic_ivp_residual_problem(
            config.eq_system.clone(),
            config.values.clone(),
            config.arg.clone(),
            expr_options,
        )
        .expect("current ExprLegacy residual preparation should succeed");
        let expr_parameter_handle = expr_residual.parameter_values_handle();
        let mut expr_jacobian =
            compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry(
                &config.eq_system,
                &config.values,
                &config.arg,
                parameters,
                expr_parameter_handle,
                storage,
                IvpSymbolicAssemblyBackend::ExprLegacy,
                IvpTelemetry::disabled(),
            );
        let expr_prepare_ms = expr_started.elapsed().as_secs_f64() * 1_000.0;

        let compat_started = Instant::now();
        let mut compat_options = SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomViewExprCompat)
            .with_equation_parameters(parameters.map(|values| values.to_vec()).unwrap_or_default())
            .with_telemetry(IvpTelemetry::disabled());
        if let Some(values) = parameter_values.clone() {
            compat_options = compat_options.with_equation_parameter_values(values);
        }
        let compat_residual = prepare_symbolic_ivp_residual_problem(
            config.eq_system.clone(),
            config.values.clone(),
            config.arg.clone(),
            compat_options,
        )
        .expect("current AtomViewExprCompat residual preparation should succeed");
        let parameter_handle: Option<SharedIvpParameterValues> =
            compat_residual.parameter_values_handle();
        let mut compat_jacobian =
            compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry(
                &config.eq_system,
                &config.values,
                &config.arg,
                parameters,
                parameter_handle,
                storage,
                IvpSymbolicAssemblyBackend::AtomViewExprCompat,
                IvpTelemetry::disabled(),
            );
        let compat_prepare_ms = compat_started.elapsed().as_secs_f64() * 1_000.0;

        let legacy_residual_value = (legacy.residual)(config.t0, &state);
        let compat_residual_value = (compat_residual.residual)(config.t0, &state);
        let residual_diff = legacy_residual_value
            .iter()
            .zip(compat_residual_value.iter())
            .map(|(legacy, compat)| (legacy - compat).abs())
            .fold(0.0_f64, f64::max);

        let legacy_jacobian_value = (legacy.jacobian)(config.t0, &state);
        let compat_jacobian_value = (compat_jacobian)(config.t0, &state);
        let jacobian_diff = bdf_jacobian_max_diff(&legacy_jacobian_value, &compat_jacobian_value);
        let expr_residual_value = (expr_residual.residual)(config.t0, &state);
        let expr_jacobian_value = (expr_jacobian)(config.t0, &state);
        let expr_residual_diff = legacy_residual_value
            .iter()
            .zip(expr_residual_value.iter())
            .map(|(legacy, expr)| (legacy - expr).abs())
            .fold(0.0_f64, f64::max);
        let expr_jacobian_diff =
            bdf_jacobian_max_diff(&legacy_jacobian_value, &expr_jacobian_value);

        let legacy_residual_ns =
            measure_residual_callback(&*legacy.residual, config.t0, &state, warm_repetitions);
        let expr_residual_ns = measure_residual_callback(
            &*expr_residual.residual,
            config.t0,
            &state,
            warm_repetitions,
        );
        let compat_residual_ns = measure_residual_callback(
            &*compat_residual.residual,
            config.t0,
            &state,
            warm_repetitions,
        );
        let legacy_jacobian_ns =
            measure_jacobian_callback(&mut *legacy.jacobian, config.t0, &state, warm_repetitions);
        let expr_jacobian_ns =
            measure_jacobian_callback(&mut *expr_jacobian, config.t0, &state, warm_repetitions);
        let compat_jacobian_ns =
            measure_jacobian_callback(&mut *compat_jacobian, config.t0, &state, warm_repetitions);

        println!(
            "{:<6} | {:<21} | {:>10.3} | {:>16.3} | {:>16.3} | {:>13.3e} | {:>13.3e}",
            matrix.label(),
            "historical-AtomView",
            legacy_prepare_ms,
            legacy_residual_ns,
            legacy_jacobian_ns,
            0.0_f64,
            0.0_f64,
        );
        println!(
            "{:<6} | {:<21} | {:>10.3} | {:>16.3} | {:>16.3} | {:>13.3e} | {:>13.3e}",
            matrix.label(),
            "ExprLegacy",
            expr_prepare_ms,
            expr_residual_ns,
            expr_jacobian_ns,
            expr_residual_diff,
            expr_jacobian_diff,
        );
        println!(
            "{:<6} | {:<21} | {:>10.3} | {:>16.3} | {:>16.3} | {:>13.3e} | {:>13.3e}",
            matrix.label(),
            "AtomViewExprCompat",
            compat_prepare_ms,
            compat_residual_ns,
            compat_jacobian_ns,
            residual_diff,
            jacobian_diff,
        );
        assert!(expr_residual_diff <= 1.0e-9);
        assert!(expr_jacobian_diff <= 1.0e-9);
        assert!(residual_diff <= 1.0e-9);
        assert!(jacobian_diff <= 1.0e-9);

        let historical_shape = expr_shape_metrics(&legacy_atomview_lambdify::symbolic_jacobian(
            &config.eq_system,
            &config.values,
        ));
        let expr_shape = expr_shape_metrics(&build_symbolic_jacobian(
            &config.eq_system,
            &config.values,
            IvpSymbolicAssemblyBackend::ExprLegacy,
            &IvpTelemetry::disabled(),
        ));
        let compat_shape = expr_shape_metrics(&build_symbolic_jacobian(
            &config.eq_system,
            &config.values,
            IvpSymbolicAssemblyBackend::AtomViewExprCompat,
            &IvpTelemetry::disabled(),
        ));
        println!(
            "[LSODE2 Jacobian Expr shape] matrix={}; dense_entries={}; nonzero | nodes | unique | repeated | depth | chars | add | mul | sub | div | pow | funcs | pow-1 | pow-frac",
            matrix.label(),
            config.eq_system.len() * config.values.len(),
        );
        for (route, shape) in [
            ("historical-AtomView", historical_shape),
            ("ExprLegacy", expr_shape),
            ("AtomViewExprCompat", compat_shape),
        ] {
            println!(
                "{:<21} | {:>6} | {:>5} | {:>6} | {:>8} | {:>5} | {:>5} | {:>4} | {:>4} | {:>3} | {:>3} | {:>4} | {:>5} | {:>5} | {:>8}",
                route,
                shape.nonzero_roots,
                shape.tree_nodes,
                shape.unique_subexpressions,
                shape.repeated_subexpressions,
                shape.max_depth,
                shape.serialized_chars,
                shape.operations.additions,
                shape.operations.multiplications,
                shape.operations.subtractions,
                shape.operations.divisions,
                shape.operations.powers,
                shape.operations.functions,
                shape.power_integer_negative,
                shape.power_fractional,
            );
            println!(
                "  [LSODE2 Jacobian Expr fingerprint] matrix={} route={} {}",
                matrix.label(),
                route,
                shape.operation_fingerprint(),
            );
        }
    }
}

#[test]
#[ignore = "debug structural diagnosis for larger real LSODE2 Jacobians"]
fn lsode2_atomview_exprcompat_large_real_jacobian_shape_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_atomview_exprcompat_large_real_jacobian_shape_story",
    );
    let chain_dimension = std::env::var("LSODE2_SHAPE_LARGE_DIM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(128);
    let workloads = [
        (
            "three-body",
            super::aot_three_body_story_tests::three_body_story_base_config(),
        ),
        (
            "diffusion-chain",
            large_diffusion_chain_config(chain_dimension),
        ),
    ];

    println!(
        "[LSODE2 large real Jacobian shape] workloads=three-body,diffusion-chain; chain_dimension={chain_dimension}; matrix assembly and timing excluded"
    );
    println!(
        "workload | dimension | route                 | nonzero | nodes | unique | repeated | depth | div | pow | pow-1 | pow-frac | functions | max_diff"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------"
    );

    for (workload, config) in workloads {
        let names = story_jacobian_name_refs(&config);
        let args = story_jacobian_args(&config, &config.y0);
        let symbolic_routes = [
            (
                "historical-AtomView",
                legacy_atomview_lambdify::symbolic_jacobian(&config.eq_system, &config.values),
            ),
            (
                "ExprLegacy",
                build_symbolic_jacobian(
                    &config.eq_system,
                    &config.values,
                    IvpSymbolicAssemblyBackend::ExprLegacy,
                    &IvpTelemetry::disabled(),
                ),
            ),
            (
                "AtomViewExprCompat",
                build_symbolic_jacobian(
                    &config.eq_system,
                    &config.values,
                    IvpSymbolicAssemblyBackend::AtomViewExprCompat,
                    &IvpTelemetry::disabled(),
                ),
            ),
        ];
        let mut baseline_values = None;

        for (route, symbolic) in symbolic_routes {
            let shape = expr_shape_metrics(&symbolic);
            let compiled = compile_story_scalar_jacobian(&symbolic, &names);
            let values = evaluate_story_scalar_jacobian(&compiled, &args);
            let max_diff = baseline_values
                .as_ref()
                .map(|baseline: &Vec<(usize, usize, f64)>| {
                    story_scalar_values_max_diff(baseline, &values)
                })
                .unwrap_or(0.0);
            if baseline_values.is_none() {
                baseline_values = Some(values);
            }
            assert!(
                max_diff <= 1.0e-9,
                "{workload}/{route} scalar Jacobian drifted by {max_diff:e}"
            );
            println!(
                "{workload:<20} | {:>9} | {route:<21} | {:>7} | {:>5} | {:>6} | {:>8} | {:>5} | {:>3} | {:>3} | {:>5} | {:>8} | {:<20} | {:>9.3e}",
                config.values.len(),
                shape.nonzero_roots,
                shape.tree_nodes,
                shape.unique_subexpressions,
                shape.repeated_subexpressions,
                shape.max_depth,
                shape.operations.divisions,
                shape.operations.powers,
                shape.power_integer_negative,
                shape.power_fractional,
                shape
                    .function_kinds
                    .iter()
                    .map(|(name, count)| format!("{name}:{count}"))
                    .collect::<Vec<_>>()
                    .join(","),
                max_diff,
            );
            println!(
                "  {workload}/{route} operation_fingerprint: {}",
                shape.operation_fingerprint()
            );
        }
    }
}

#[test]
#[ignore = "debug operation/lowering diagnosis for real LSODE2 Jacobians"]
fn lsode2_atomview_exprcompat_real_closure_lowering_cost_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_atomview_exprcompat_real_closure_lowering_cost_story",
    );
    let repetitions = std::env::var("LSODE2_CLOSURE_PROFILE_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2_000);
    let chain_dimension = std::env::var("LSODE2_SHAPE_LARGE_DIM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(128);
    let workloads = [
        (
            "three-body",
            super::aot_three_body_story_tests::three_body_story_base_config(),
        ),
        (
            "diffusion-chain",
            large_diffusion_chain_config(chain_dimension),
        ),
    ];

    println!(
        "[LSODE2 real closure lowering cost] workloads=three-body,diffusion-chain; chain_dimension={chain_dimension}; repetitions={repetitions}; matrix assembly excluded"
    );
    println!(
        "workload | dimension | route                 | nonzero | nodes | div | pow | pow-1 | pow-frac | funcs | symbolic_prepare_ms | closure_compile_ms | scalar_eval_ns/call | max_diff"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );

    for (workload, config) in workloads {
        let names = story_jacobian_name_refs(&config);
        let args = story_jacobian_args(&config, &config.y0);
        let mut baseline_values: Option<Vec<(usize, usize, f64)>> = None;

        for route in ["historical-AtomView", "ExprLegacy", "AtomViewExprCompat"] {
            let symbolic_prepare_started = Instant::now();
            let symbolic = match route {
                "historical-AtomView" => {
                    legacy_atomview_lambdify::symbolic_jacobian(&config.eq_system, &config.values)
                }
                "ExprLegacy" => build_symbolic_jacobian(
                    &config.eq_system,
                    &config.values,
                    IvpSymbolicAssemblyBackend::ExprLegacy,
                    &IvpTelemetry::disabled(),
                ),
                "AtomViewExprCompat" => build_symbolic_jacobian(
                    &config.eq_system,
                    &config.values,
                    IvpSymbolicAssemblyBackend::AtomViewExprCompat,
                    &IvpTelemetry::disabled(),
                ),
                _ => unreachable!("real closure route is fixed"),
            };
            let symbolic_prepare_ms = symbolic_prepare_started.elapsed().as_secs_f64() * 1_000.0;
            let shape = expr_shape_metrics(&symbolic);

            let closure_started = Instant::now();
            let compiled = compile_story_scalar_jacobian(&symbolic, &names);
            let closure_compile_ms = closure_started.elapsed().as_secs_f64() * 1_000.0;
            let values = evaluate_story_scalar_jacobian(&compiled, &args);
            let max_diff = baseline_values
                .as_ref()
                .map(|baseline| story_scalar_values_max_diff(baseline, &values))
                .unwrap_or(0.0);
            if baseline_values.is_none() {
                baseline_values = Some(values);
            }
            assert!(
                max_diff <= 1.0e-9,
                "{workload}/{route} scalar Jacobian drifted by {max_diff:e}"
            );

            let scalar_eval_ns = measure_story_scalar_jacobian(&compiled, &args, repetitions);
            println!(
                "{workload:<20} | {:>9} | {route:<21} | {:>7} | {:>5} | {:>3} | {:>3} | {:>5} | {:>8} | {:>5} | {:>19.3} | {:>18.3} | {:>19.3} | {:>9.3e}",
                config.values.len(),
                shape.nonzero_roots,
                shape.tree_nodes,
                shape.operations.divisions,
                shape.operations.powers,
                shape.power_integer_negative,
                shape.power_fractional,
                shape.operations.functions,
                symbolic_prepare_ms,
                closure_compile_ms,
                scalar_eval_ns,
                max_diff,
            );
            println!(
                "  {workload}/{route} operation_fingerprint: {}",
                shape.operation_fingerprint()
            );
        }
    }
}

#[test]
#[ignore = "release real-Jacobian three-boundary comparison; preparation and callback stages are isolated"]
fn lsode2_view_three_boundary_real_jacobian_release_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::lsode2_view_three_boundary_real_jacobian_release_story",
    );
    let repetitions = std::env::var("LSODE2_THREE_BOUNDARY_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2_000);
    let chain_dimension = std::env::var("LSODE2_THREE_BOUNDARY_CHAIN_DIM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(128);
    let workloads = [
        (
            "three-body",
            super::aot_three_body_story_tests::three_body_story_base_config(),
        ),
        (
            "diffusion-chain",
            large_diffusion_chain_config(chain_dimension),
        ),
    ];

    println!(
        "[LSODE2 View three-boundary real Jacobian] workloads=three-body,diffusion-chain; chain_dimension={chain_dimension}; repetitions={repetitions}; matrix assembly and linear solve excluded"
    );
    println!(
        "workload | route                 | nonzero | symbolic_ms | atom_convert_ms | closure_ms | eval_ns/call | max_diff"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------"
    );

    for (workload, config) in workloads {
        let names = story_jacobian_name_refs(&config);
        let args = story_jacobian_args(&config, &config.y0);
        let symbolic_started = Instant::now();
        let expr_symbolic = build_symbolic_jacobian(
            &config.eq_system,
            &config.values,
            IvpSymbolicAssemblyBackend::ExprLegacy,
            &IvpTelemetry::disabled(),
        );
        let symbolic_ms = symbolic_started.elapsed().as_secs_f64() * 1_000.0;

        let atom_started = Instant::now();
        let atom_symbolic = expr_symbolic
            .iter()
            .map(|row| row.iter().map(expr_to_atom).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let atom_convert_ms = atom_started.elapsed().as_secs_f64() * 1_000.0;

        let compat_started = Instant::now();
        let compat_symbolic = atom_symbolic
            .iter()
            .map(|row| row.iter().map(atom_to_expr).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let compat_convert_ms = compat_started.elapsed().as_secs_f64() * 1_000.0;

        let expr_closure_started = Instant::now();
        let expr_callbacks = compile_story_scalar_jacobian(&expr_symbolic, &names);
        let expr_closure_ms = expr_closure_started.elapsed().as_secs_f64() * 1_000.0;
        let expr_values = evaluate_story_scalar_jacobian(&expr_callbacks, &args);
        let expr_eval_ns = measure_story_scalar_jacobian(&expr_callbacks, &args, repetitions);

        let compat_closure_started = Instant::now();
        let compat_callbacks = compile_story_scalar_jacobian(&compat_symbolic, &names);
        let compat_closure_ms = compat_closure_started.elapsed().as_secs_f64() * 1_000.0;
        let compat_values = evaluate_story_scalar_jacobian(&compat_callbacks, &args);
        let compat_eval_ns = measure_story_scalar_jacobian(&compat_callbacks, &args, repetitions);

        let native_closure_started = Instant::now();
        let mut native_callbacks = Vec::new();
        for (row, symbolic_row) in atom_symbolic.iter().enumerate() {
            for (col, atom) in symbolic_row.iter().enumerate() {
                if !atom.is_zero() {
                    native_callbacks.push((row, col, atom.lambdify_borrowed_thread_safe(&names)));
                }
            }
        }
        let native_closure_ms = native_closure_started.elapsed().as_secs_f64() * 1_000.0;
        let native_values = evaluate_story_scalar_jacobian(&native_callbacks, &args);
        let native_eval_ns = measure_story_scalar_jacobian(&native_callbacks, &args, repetitions);

        let compat_diff = story_scalar_values_max_diff(&expr_values, &compat_values);
        let native_diff = story_scalar_values_max_diff(&expr_values, &native_values);
        assert!(
            compat_diff <= 1.0e-9,
            "{workload}/AtomViewExprCompat Jacobian drifted by {compat_diff:e}"
        );
        assert!(
            native_diff <= 1.0e-9,
            "{workload}/AtomNative Jacobian drifted by {native_diff:e}"
        );

        println!(
            "{workload:<20} | {:<21} | {:>7} | {:>11.3} | {:>15.3} | {:>10.3} | {:>12.3} | {:>9.3e}",
            "ExprLegacy",
            expr_values.len(),
            symbolic_ms,
            0.0,
            expr_closure_ms,
            expr_eval_ns,
            0.0,
        );
        println!(
            "{workload:<20} | {:<21} | {:>7} | {:>11.3} | {:>15.3} | {:>10.3} | {:>12.3} | {:>9.3e}",
            "AtomViewExprCompat",
            compat_values.len(),
            0.0,
            compat_convert_ms,
            compat_closure_ms,
            compat_eval_ns,
            compat_diff,
        );
        println!(
            "{workload:<20} | {:<21} | {:>7} | {:>11.3} | {:>15.3} | {:>10.3} | {:>12.3} | {:>9.3e}",
            "AtomNative",
            native_values.len(),
            0.0,
            atom_convert_ms,
            native_closure_ms,
            native_eval_ns,
            native_diff,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ExprShapeMetrics {
    nonzero_entries: usize,
    nodes: usize,
    max_depth: usize,
    serialized_chars: usize,
}

type StoryScalarJacobianEntry = (usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>);

fn story_jacobian_name_refs(config: &Lsode2ProblemConfig) -> Vec<&str> {
    let mut names = Vec::with_capacity(
        1 + config.values.len() + config.equation_parameters.as_ref().map_or(0, Vec::len),
    );
    names.push(config.arg.as_str());
    if let Some(parameters) = config.equation_parameters.as_ref() {
        names.extend(parameters.iter().map(String::as_str));
    }
    names.extend(config.values.iter().map(String::as_str));
    names
}

fn story_jacobian_args(config: &Lsode2ProblemConfig, state: &DVector<f64>) -> Vec<f64> {
    let mut args = Vec::with_capacity(
        1 + state.len()
            + config
                .equation_parameter_values
                .as_ref()
                .map_or(0, DVector::len),
    );
    args.push(config.t0);
    if let Some(parameters) = config.equation_parameter_values.as_ref() {
        args.extend(parameters.iter().copied());
    }
    args.extend(state.iter().copied());
    args
}

fn compile_story_scalar_jacobian(
    symbolic_jacobian: &[Vec<Expr>],
    names: &[&str],
) -> Vec<StoryScalarJacobianEntry> {
    let names = names.to_vec();
    let mut compiled = Vec::new();
    for (row, symbolic_row) in symbolic_jacobian.iter().enumerate() {
        for (col, expr) in symbolic_row.iter().enumerate() {
            if !expr.is_zero() {
                compiled.push((
                    row,
                    col,
                    Expr::lambdify_borrowed_thread_safe(expr, names.as_slice()),
                ));
            }
        }
    }
    compiled
}

fn evaluate_story_scalar_jacobian(
    entries: &[StoryScalarJacobianEntry],
    args: &[f64],
) -> Vec<(usize, usize, f64)> {
    entries
        .iter()
        .map(|(row, col, eval)| (*row, *col, eval(args)))
        .collect()
}

fn story_scalar_values_max_diff(
    left: &[(usize, usize, f64)],
    right: &[(usize, usize, f64)],
) -> f64 {
    assert_eq!(left.len(), right.len(), "scalar Jacobian nnz differs");
    left.iter()
        .zip(right.iter())
        .map(
            |((left_row, left_col, left_value), (right_row, right_col, right_value))| {
                assert_eq!(left_row, right_row, "scalar Jacobian row differs");
                assert_eq!(left_col, right_col, "scalar Jacobian column differs");
                (left_value - right_value).abs()
            },
        )
        .fold(0.0_f64, f64::max)
}

fn measure_story_scalar_jacobian(
    entries: &[StoryScalarJacobianEntry],
    args: &[f64],
    repetitions: usize,
) -> f64 {
    for _ in 0..3 {
        let checksum = entries.iter().map(|(_, _, eval)| eval(args)).sum::<f64>();
        std::hint::black_box(checksum);
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        let checksum = entries.iter().map(|(_, _, eval)| eval(args)).sum::<f64>();
        std::hint::black_box(checksum);
    }
    started.elapsed().as_secs_f64() * 1_000_000_000.0 / repetitions as f64
}

fn expr_shape_metrics(jacobian: &[Vec<Expr>]) -> ExpressionMetrics {
    // Keep zero structural entries out of the comparison, matching the old
    // story metric. The diagnostic walk itself is intentionally outside the
    // timed callback measurement.
    let nonzero: Vec<Expr> = jacobian
        .iter()
        .flat_map(|row| row.iter())
        .filter(|expr| !expr.is_zero())
        .cloned()
        .collect();
    inspect_exprs(&nonzero)
}

fn measure_residual_callback(
    callback: &dyn Fn(f64, &DVector<f64>) -> DVector<f64>,
    t: f64,
    state: &DVector<f64>,
    repetitions: usize,
) -> f64 {
    for _ in 0..3 {
        std::hint::black_box(callback(t, state));
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(callback(t, state));
    }
    started.elapsed().as_secs_f64() * 1_000_000_000.0 / repetitions as f64
}

fn measure_jacobian_callback(
    callback: &mut dyn FnMut(f64, &DVector<f64>) -> BdfJacobian,
    t: f64,
    state: &DVector<f64>,
    repetitions: usize,
) -> f64 {
    for _ in 0..3 {
        std::hint::black_box(callback(t, state));
    }
    let started = Instant::now();
    for _ in 0..repetitions {
        std::hint::black_box(callback(t, state));
    }
    started.elapsed().as_secs_f64() * 1_000_000_000.0 / repetitions as f64
}

fn bdf_jacobian_max_diff(left: &BdfJacobian, right: &BdfJacobian) -> f64 {
    match (left, right) {
        (
            BdfJacobian::SparseTriplets { triplets: left, .. },
            BdfJacobian::SparseTriplets {
                triplets: right, ..
            },
        ) => {
            assert_eq!(
                left.len(),
                right.len(),
                "historical/current sparse nnz differs"
            );
            left.iter()
                .zip(right.iter())
                .map(|(left, right)| {
                    assert_eq!(left.row, right.row, "historical/current sparse row differs");
                    assert_eq!(
                        left.col, right.col,
                        "historical/current sparse column differs"
                    );
                    (left.val - right.val).abs()
                })
                .fold(0.0_f64, f64::max)
        }
        (BdfJacobian::Banded(left), BdfJacobian::Banded(right)) => {
            assert_eq!(
                left.n(),
                right.n(),
                "historical/current banded size differs"
            );
            let mut max_diff = 0.0_f64;
            for row in 0..left.n() {
                for col in 0..left.n() {
                    max_diff = max_diff.max((left[(row, col)] - right[(row, col)]).abs());
                }
            }
            max_diff
        }
        _ => f64::INFINITY,
    }
}
