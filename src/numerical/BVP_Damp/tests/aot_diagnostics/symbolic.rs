#[test]
fn symbolic_assembly_backends_match_two_point_jacobian_on_small_sparse_bundle() {
    aot_test_report!(symbolic_assembly_backends_match_two_point_jacobian_on_small_sparse_bundle);
    let label = "two-point-24";
    let equation = NonlinEquation::TwoPointBVP;
    let n_steps = 24usize;

    let mut legacy_solver = make_example_solver(
        &equation,
        n_steps,
        Some(SolverParams::default()),
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
    );
    let legacy_request = legacy_solver.build_solver_request(None, None);
    let atom_request = make_example_solver(
        &equation,
        n_steps,
        Some(SolverParams::default()),
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
    )
    .build_solver_request(None, None);

    let (mut legacy_bundle, legacy_ms) = measure_sparse_bundle_build_with_symbolic_backend(
        legacy_request,
        BvpSymbolicAssemblyBackend::ExprLegacy,
    )
    .expect("legacy sparse bundle should build for exact-case backend comparison");
    let (mut atom_bundle, atom_ms) = measure_sparse_bundle_build_with_symbolic_backend(
        atom_request,
        BvpSymbolicAssemblyBackend::AtomView,
    )
    .expect("atom sparse bundle should build for exact-case backend comparison");

    let args = DVector::from_element(atom_bundle.variable_string.len(), 0.7);
    let (residual_max_diff, jacobian_max_diff) =
        compare_sparse_bundles_numerically(&mut legacy_bundle, &mut atom_bundle, &args, label);
    println!(
        "[BVP symbolic assembly exact-case] label={label}, n_steps={n_steps}, legacy_ms={legacy_ms:.3}, atom_ms={atom_ms:.3}, speedup={:.3}x, residual_max_diff={residual_max_diff:.6e}, jacobian_max_diff={jacobian_max_diff:.6e}",
        legacy_ms / atom_ms,
    );
    assert!(
        jacobian_max_diff < 1.0e-6,
        "{label}: jacobian disagreement between ExprLegacy and AtomView is too large: {jacobian_max_diff}"
    );
    assert!(
        residual_max_diff < 1.0e-6,
        "{label}: residual disagreement between ExprLegacy and AtomView is too large: {residual_max_diff}"
    );
}

#[test]
#[ignore = "diagnostic timing table for symbolic assembly backends across representative BVP fixtures"]
fn symbolic_assembly_backends_report_representative_fixture_table() {
    aot_test_report!(symbolic_assembly_backends_report_representative_fixture_table);
    let cases = [
        ("two-point-72", NonlinEquation::TwoPointBVP, 72usize),
        ("clairaut-72", NonlinEquation::Clairaut, 72usize),
        ("parachute-48", NonlinEquation::ParachuteEquation, 48usize),
        ("lane-emden-48", NonlinEquation::LaneEmden5, 48usize),
    ];

    println!();
    println!(
        "{:<18} | {:>7} | {:>14} | {:>12} | {:>12} | {:>14} | {:>14}",
        "fixture", "n_steps", "status", "legacy_ms", "atom_ms", "residual_diff", "jacobian_diff"
    );
    println!("{}", "-".repeat(108));

    for (label, equation, n_steps) in cases {
        let mut legacy_solver = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        );
        let legacy_request = legacy_solver.build_solver_request(None, None);
        let atom_request = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        )
        .build_solver_request(None, None);
        let legacy_result = measure_sparse_bundle_build_with_symbolic_backend(
            legacy_request,
            BvpSymbolicAssemblyBackend::ExprLegacy,
        );
        let atom_result = measure_sparse_bundle_build_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        );

        match (legacy_result, atom_result) {
            (Ok((mut legacy_bundle, legacy_ms)), Ok((mut atom_bundle, atom_ms))) => {
                let args = DVector::from_element(atom_bundle.variable_string.len(), 0.7);
                let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                    &mut legacy_bundle,
                    &mut atom_bundle,
                    &args,
                    label,
                );
                println!(
                    "{:<18} | {:>7} | {:>14} | {:>12.3} | {:>12.3} | {:>14.6e} | {:>14.6e}",
                    label,
                    n_steps,
                    if atom_ms < legacy_ms {
                        "atom faster"
                    } else {
                        "legacy faster"
                    },
                    legacy_ms,
                    atom_ms,
                    residual_max_diff,
                    jacobian_max_diff
                );
            }
            (Ok((_legacy_bundle, legacy_ms)), Err(atom_error)) => {
                println!(
                    "{:<18} | {:>7} | {:>14} | {:>12.3} | {:>12} | {:>14} | {:>14}",
                    label,
                    n_steps,
                    "atom failed",
                    legacy_ms,
                    "-",
                    "-",
                    format!("{atom_error:?}")
                );
            }
            (Err(legacy_error), Ok((_atom_bundle, atom_ms))) => {
                println!(
                    "{:<18} | {:>7} | {:>14} | {:>12} | {:>12.3} | {:>14} | {:>14}",
                    label,
                    n_steps,
                    "legacy failed",
                    "-",
                    atom_ms,
                    format!("{legacy_error:?}"),
                    "-"
                );
            }
            (Err(legacy_error), Err(atom_error)) => {
                println!(
                    "{:<18} | {:>7} | {:>14} | {:>12} | {:>12} | {:>14} | {:>14}",
                    label,
                    n_steps,
                    "both failed",
                    "-",
                    "-",
                    format!("{legacy_error:?}"),
                    format!("{atom_error:?}")
                );
            }
        }
    }
}

#[test]
#[ignore = "diagnostic solver-level ExprLegacy vs AtomView compare on representative exact-like BVP fixtures"]
fn symbolic_assembly_backends_report_representative_solver_table() {
    aot_test_report!(symbolic_assembly_backends_report_representative_solver_table);
    #[derive(Debug)]
    struct RepresentativeSolverRow {
        label: &'static str,
        backend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
        generate_ms: f64,
        solve_ms: f64,
        max_diff_vs_legacy: f64,
    }

    let cases = [
        ("parachute-96", NonlinEquation::ParachuteEquation, 96usize),
        ("lane-emden-96", NonlinEquation::LaneEmden5, 96usize),
    ];
    let mut rows = Vec::new();
    let base_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

    for (label, equation, n_steps) in cases {
        let mut legacy_solver = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_generate_begin = Instant::now();
        legacy_solver
            .try_eq_generate(None, None)
            .expect("ExprLegacy representative solve-path generate should succeed");
        let legacy_generate_ms = legacy_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_solve_begin = Instant::now();
        legacy_solver
            .try_solve()
            .expect("ExprLegacy representative solve-path solve should succeed");
        let legacy_solve_ms = legacy_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_solution = legacy_solver
            .get_result()
            .expect("ExprLegacy representative solve-path should produce a solution");

        let mut atom_solver = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_generate_begin = Instant::now();
        atom_solver
            .try_eq_generate(None, None)
            .expect("AtomView representative solve-path generate should succeed");
        let atom_generate_ms = atom_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_solve_begin = Instant::now();
        atom_solver
            .try_solve()
            .expect("AtomView representative solve-path solve should succeed");
        let atom_solve_ms = atom_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_solution = atom_solver
            .get_result()
            .expect("AtomView representative solve-path should produce a solution");

        let max_diff_vs_legacy = legacy_solution
            .iter()
            .zip(atom_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);

        rows.push(RepresentativeSolverRow {
            label,
            backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            n_steps,
            generate_ms: legacy_generate_ms,
            solve_ms: legacy_solve_ms,
            max_diff_vs_legacy: 0.0,
        });
        rows.push(RepresentativeSolverRow {
            label,
            backend: BvpSymbolicAssemblyBackend::AtomView,
            n_steps,
            generate_ms: atom_generate_ms,
            solve_ms: atom_solve_ms,
            max_diff_vs_legacy,
        });
    }

    println!("[BVP symbolic assembly representative solver compare] ExprLegacy vs AtomView");
    println!(
        "{:<16} | {:<12} | {:>7} | {:>12} | {:>10} | {:>18}",
        "fixture", "backend", "n_steps", "generate_ms", "solve_ms", "max_diff_vs_legacy"
    );
    println!("{}", "-".repeat(92));
    for row in &rows {
        let backend = match row.backend {
            BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
            BvpSymbolicAssemblyBackend::AtomView => "AtomView",
        };
        println!(
            "{:<16} | {:<12} | {:>7} | {:>12.3} | {:>10.3} | {:>18.6e}",
            row.label, backend, row.n_steps, row.generate_ms, row.solve_ms, row.max_diff_vs_legacy
        );
    }
}

#[test]
#[ignore = "diagnostic compare for symbolic assembly backends across full BVP fixtures"]
fn symbolic_assembly_backends_build_sparse_bundles_for_representative_examples() {
    aot_test_report!(symbolic_assembly_backends_build_sparse_bundles_for_representative_examples);
    let exact_examples = [
        ("two-point-72", NonlinEquation::TwoPointBVP, 72usize),
        ("clairaut-72", NonlinEquation::Clairaut, 72usize),
        ("parachute-48", NonlinEquation::ParachuteEquation, 48usize),
        ("lane-emden-48", NonlinEquation::LaneEmden5, 48usize),
    ];

    for (label, equation, n_steps) in exact_examples {
        let mut solver = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        );
        let legacy_request = solver.build_solver_request(None, None);
        let atom_request = make_example_solver(
            &equation,
            n_steps,
            Some(SolverParams::default()),
            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
        )
        .build_solver_request(None, None);
        let legacy_result = measure_sparse_bundle_build_with_symbolic_backend(
            legacy_request,
            BvpSymbolicAssemblyBackend::ExprLegacy,
        );
        let atom_result = measure_sparse_bundle_build_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        );
        match (legacy_result, atom_result) {
            (Ok((mut legacy_bundle, legacy_ms)), Ok((mut atom_bundle, atom_ms))) => {
                assert_eq!(legacy_bundle.variable_string, atom_bundle.variable_string);
                assert_eq!(legacy_bundle.jacobian_shape(), atom_bundle.jacobian_shape());
                assert_eq!(legacy_bundle.residual_len(), atom_bundle.residual_len());
                let args = DVector::from_vec(
                    (0..atom_bundle.variable_string.len())
                        .map(|index| 0.2 + index as f64 * 0.01)
                        .collect(),
                );
                let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                    &mut legacy_bundle,
                    &mut atom_bundle,
                    &args,
                    label,
                );
                println!(
                    "[BVP symbolic assembly compare] label={label}, n_steps={n_steps}, legacy_ms={legacy_ms:.3}, atom_ms={atom_ms:.3}, speedup={:.3}x, residual_max_diff={residual_max_diff:.6e}, jacobian_max_diff={jacobian_max_diff:.6e}",
                    legacy_ms / atom_ms,
                );
            }
            (Ok((_legacy_bundle, legacy_ms)), Err(error)) => {
                println!(
                    "[BVP symbolic assembly compare] label={label}, n_steps={n_steps}, legacy_ms={legacy_ms:.3}, atom_status=ERR({error:?})"
                );
            }
            (Err(error), Ok((_atom_bundle, atom_ms))) => {
                panic!(
                    "{label}: legacy symbolic assembly unexpectedly failed while atom succeeded in {atom_ms:.3} ms: {error:?}"
                );
            }
            (Err(legacy_error), Err(atom_error)) => {
                panic!(
                    "{label}: both symbolic assembly backends failed, legacy={legacy_error:?}, atom={atom_error:?}"
                );
            }
        }
    }
}

#[test]
#[ignore = "debug localization of the Lane-Emden ExprLegacy/AtomView symbolic drift"]
fn lane_emden_symbolic_parity_localization() {
    aot_test_report!(lane_emden_symbolic_parity_localization);
    let equation = NonlinEquation::LaneEmden5;
    let n_steps = 48usize;
    let config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

    let mut legacy_solver = make_example_solver(
        &equation,
        n_steps,
        Some(SolverParams::default()),
        config.clone(),
    );
    let legacy_request = legacy_solver.build_solver_request(None, None);
    let mut atom_solver =
        make_example_solver(&equation, n_steps, Some(SolverParams::default()), config);
    let atom_request = atom_solver.build_solver_request(None, None);

    let (mut legacy_bundle, legacy_ms) = measure_sparse_bundle_build_with_symbolic_backend(
        legacy_request,
        BvpSymbolicAssemblyBackend::ExprLegacy,
    )
    .expect("Lane-Emden ExprLegacy bundle should build");
    let (mut atom_bundle, atom_ms) = measure_sparse_bundle_build_with_symbolic_backend(
        atom_request,
        BvpSymbolicAssemblyBackend::AtomView,
    )
    .expect("Lane-Emden AtomView bundle should build");

    assert_eq!(legacy_bundle.variable_string, atom_bundle.variable_string);
    assert_eq!(legacy_bundle.jacobian_shape(), atom_bundle.jacobian_shape());
    assert_eq!(legacy_bundle.residual_len(), atom_bundle.residual_len());
    println!(
        "[Lane-Emden symbolic localization] n_steps={n_steps}, legacy_ms={legacy_ms:.3}, atom_ms={atom_ms:.3}, residual_len={}, jacobian_shape={:?}",
        legacy_bundle.residual_len(),
        legacy_bundle.jacobian_shape()
    );

    let states = [
        (
            "uniform-0.7",
            DVector::from_element(legacy_bundle.variable_string.len(), 0.7),
        ),
        (
            "ramp-0.2",
            DVector::from_iterator(
                legacy_bundle.variable_string.len(),
                (0..legacy_bundle.variable_string.len()).map(|index| 0.2 + index as f64 * 0.01),
            ),
        ),
    ];
    for (label, args) in states {
        let (residual_max_diff, jacobian_max_diff) = report_top_sparse_bundle_differences(
            &mut legacy_bundle,
            &mut atom_bundle,
            &args,
            label,
        );
        assert!(
            residual_max_diff <= 1.0e-9 && jacobian_max_diff <= 1.0e-9,
            "Lane-Emden symbolic parity exceeded the 1e-9 gate for {label}: residual={residual_max_diff:.6e}, jacobian={jacobian_max_diff:.6e}"
        );
    }
}

#[test]
#[ignore = "diagnostic symbolic assembly comparison on real combustion sparse bundle"]
fn symbolic_assembly_backends_report_combustion_sparse_bundle_timings() {
    aot_test_report!(symbolic_assembly_backends_report_combustion_sparse_bundle_timings);
    let n_steps = 100usize;
    let mut solver = make_combustion_solver(
        n_steps,
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
    );
    let legacy_request = solver.build_solver_request(None, None);
    let atom_request = make_combustion_solver(
        n_steps,
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
    )
    .build_solver_request(None, None);
    let legacy_result = measure_sparse_bundle_build_with_symbolic_backend(
        legacy_request,
        BvpSymbolicAssemblyBackend::ExprLegacy,
    );
    let atom_result = measure_sparse_bundle_build_with_symbolic_backend(
        atom_request,
        BvpSymbolicAssemblyBackend::AtomView,
    );

    println!();
    println!("[BVP symbolic assembly compare] combustion sparse bundle, n_steps={n_steps}");
    match (legacy_result, atom_result) {
        (Ok((mut legacy_bundle, legacy_ms)), Ok((mut atom_bundle, atom_ms))) => {
            let args = DVector::from_vec(
                (0..atom_bundle.variable_string.len())
                    .map(|index| 0.3 + index as f64 * 0.005)
                    .collect(),
            );
            let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
                &mut legacy_bundle,
                &mut atom_bundle,
                &args,
                "combustion-assembly-compare",
            );
            println!("{:<20} | {:>12}", "backend", "build_ms");
            println!("{}", "-".repeat(37));
            println!("{:<20} | {:>12.3}", "expr-legacy", legacy_ms);
            println!("{:<20} | {:>12.3}", "atom-view", atom_ms);
            println!("{:<20} | {:>12.6e}", "residual_max_diff", residual_max_diff);
            println!("{:<20} | {:>12.6e}", "jacobian_max_diff", jacobian_max_diff);
            println!(
                "[BVP symbolic assembly winner] backend={}, speedup={:.3}x",
                if atom_ms < legacy_ms {
                    "atom-view"
                } else {
                    "expr-legacy"
                },
                if atom_ms < legacy_ms {
                    legacy_ms / atom_ms
                } else {
                    atom_ms / legacy_ms
                }
            );
        }
        (Ok((_legacy_bundle, legacy_ms)), Err(atom_error)) => {
            println!("{:<20} | {:>12}", "backend", "status");
            println!("{}", "-".repeat(37));
            println!("{:<20} | {:>12.3} ms", "expr-legacy", legacy_ms);
            println!(
                "{:<20} | {:>12}",
                "atom-view",
                format!("ERR({atom_error:?})")
            );
        }
        (Err(legacy_error), Ok((_atom_bundle, atom_ms))) => {
            println!("{:<20} | {:>12}", "backend", "status");
            println!("{}", "-".repeat(37));
            println!(
                "{:<20} | {:>12}",
                "expr-legacy",
                format!("ERR({legacy_error:?})")
            );
            println!("{:<20} | {:>12.3} ms", "atom-view", atom_ms);
        }
        (Err(legacy_error), Err(atom_error)) => {
            println!("{:<20} | {:>12}", "backend", "status");
            println!("{}", "-".repeat(37));
            println!(
                "{:<20} | {:>12}",
                "expr-legacy",
                format!("ERR({legacy_error:?})")
            );
            println!(
                "{:<20} | {:>12}",
                "atom-view",
                format!("ERR({atom_error:?})")
            );
        }
    }
}

#[test]
#[ignore = "diagnostic row-level compare for combustion discretized residual assembly"]
fn symbolic_assembly_backends_report_combustion_discretized_row_diagnostics() {
    aot_test_report!(symbolic_assembly_backends_report_combustion_discretized_row_diagnostics);
    let n_steps = 100usize;
    let mut solver = make_combustion_solver(
        n_steps,
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
    );
    let request = solver.build_solver_request(None, None);

    let mut legacy = Jacobian::new();
    let request_eq_system = request.eq_system.clone();
    let request_values = request.values.clone();
    let request_arg = request.arg.clone();
    let request_mesh = request.mesh.clone();
    let request_border_conditions = request.border_conditions.clone();
    let request_scheme = request.scheme.clone();
    let request_t0 = request.t0;
    let request_h = request.h;
    legacy.discretization_system_BVP_par(
        request.eq_system.clone(),
        request.values.clone(),
        request.arg.clone(),
        request.t0,
        request.n_steps,
        request.h,
        request.mesh.clone(),
        request.border_conditions.clone(),
        request.bounds.clone(),
        request.rel_tolerance.clone(),
        request.scheme.clone(),
    );

    let atom_discretized = discretization_system_bvp_par_atom(
        request.eq_system,
        request.values,
        request.arg,
        request.t0,
        request.n_steps,
        request.h,
        request.mesh,
        request.border_conditions,
        request.bounds,
        request.rel_tolerance,
        request.scheme,
    );

    let atom_exprs = atom_discretized
        .vector_of_functions
        .iter()
        .map(atom_to_expr)
        .collect::<Vec<_>>();
    let variable_names = legacy.variable_string.clone();
    let args = (0..variable_names.len())
        .map(|index| 0.25 + index as f64 * 0.0025)
        .collect::<Vec<_>>();
    let variable_refs = variable_names
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();

    let mut max_diff = 0.0f64;
    let mut max_index = 0usize;
    for index in 0..legacy.vector_of_functions.len() {
        let legacy_value = legacy.vector_of_functions[index].eval_expression(&variable_refs, &args);
        let atom_value = atom_exprs[index].eval_expression(&variable_refs, &args);
        let diff = (legacy_value - atom_value).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }

    println!(
        "[combustion discretized row diagnostics] n_steps={n_steps}, max_index={}, max_diff={:.6e}",
        max_index, max_diff
    );
    println!(
        "[combustion discretized row diagnostics] legacy_row={}",
        legacy.vector_of_functions[max_index]
    );
    println!(
        "[combustion discretized row diagnostics] atom_row={}",
        atom_exprs[max_index]
    );

    let row_step = max_index / request_values.len();
    let row_eq = max_index % request_values.len();
    let matrix_of_names = (0..=n_steps)
        .map(|step| {
            request_values
                .iter()
                .map(|name| format!("{name}_{step}"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bc_value_map = request_border_conditions
        .iter()
        .flat_map(|(name, entries)| {
            entries.iter().filter_map(move |(side, value)| match side {
                0 => Some((format!("{name}_0"), *value)),
                1 => Some((format!("{name}_{n_steps}"), *value)),
                _ => None,
            })
        })
        .collect::<HashMap<_, _>>();

    let legacy_eq_step = {
        let rename_map = request_values
            .iter()
            .zip(matrix_of_names[row_step].iter())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<_, _>>();
        request_eq_system[row_eq]
            .rename_variables(&rename_map)
            .set_variable(
                &request_arg,
                request_mesh
                    .as_ref()
                    .map(|m| m[row_step])
                    .unwrap_or(request_t0 + request_h.unwrap_or(1.0) * row_step as f64),
            )
    };
    let atom_eq_step = eq_step_atom(
        &crate::symbolic::View::conversions::expr_to_atom(&request_eq_system[row_eq]),
        &matrix_of_names,
        &request_values,
        &request_arg,
        row_step,
        request_mesh
            .as_ref()
            .map(|m| m[row_step])
            .unwrap_or(request_t0 + request_h.unwrap_or(1.0) * row_step as f64),
        &request_scheme,
    );
    let legacy_eq_step_bc = legacy_eq_step
        .set_variable_from_map(&bc_value_map)
        .simplify();
    let atom_eq_step_bc =
        atom_to_expr(&crate::symbolic::View::transform::substitute_symbol_values(
            &atom_eq_step,
            &bc_value_map
                .iter()
                .map(|(name, value)| {
                    (
                        crate::symbolic::View::state::Symbol::new(crate::wrap_symbol!(
                            name.as_str()
                        )),
                        *value,
                    )
                })
                .collect(),
        ));
    println!(
        "[combustion discretized row diagnostics] row_step={}, row_eq={}",
        row_step, row_eq
    );
    println!(
        "[combustion discretized row diagnostics] legacy_eq_step={}",
        legacy_eq_step
    );
    println!(
        "[combustion discretized row diagnostics] atom_eq_step={}",
        atom_to_expr(&atom_eq_step)
    );
    println!(
        "[combustion discretized row diagnostics] legacy_eq_step_bc={}",
        legacy_eq_step_bc
    );
    println!(
        "[combustion discretized row diagnostics] atom_eq_step_bc={}",
        atom_eq_step_bc
    );

    for index in
        max_index.saturating_sub(2)..=(max_index + 2).min(legacy.vector_of_functions.len() - 1)
    {
        let legacy_value = legacy.vector_of_functions[index].eval_expression(&variable_refs, &args);
        let atom_value = atom_exprs[index].eval_expression(&variable_refs, &args);
        println!(
            "[combustion discretized row diagnostics] row={}, legacy={:.6e}, atom={:.6e}, diff={:.6e}",
            index,
            legacy_value,
            atom_value,
            (legacy_value - atom_value).abs()
        );
    }
}

#[test]
#[ignore = "diagnostic solver-level ExprLegacy vs AtomView stress comparison on combustion"]
fn symbolic_assembly_backends_report_combustion_solver_stress_table() {
    aot_test_report!(symbolic_assembly_backends_report_combustion_solver_stress_table);
    #[derive(Debug)]
    struct SymbolicAssemblyStressRow {
        backend: BvpSymbolicAssemblyBackend,
        n_steps: usize,
        bundle_ms: f64,
        generate_ms: f64,
        solve_ms: f64,
        residual_max_diff: f64,
        jacobian_max_diff: f64,
        max_diff_vs_legacy: f64,
    }

    let mut rows = Vec::new();
    let n_steps_list = [200usize, 300usize];
    let base_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

    for &n_steps in &n_steps_list {
        let mut legacy_request_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_request = legacy_request_solver.build_solver_request(None, None);
        let (mut legacy_bundle, legacy_bundle_ms) =
            measure_sparse_bundle_build_with_symbolic_backend(
                legacy_request,
                BvpSymbolicAssemblyBackend::ExprLegacy,
            )
            .expect("ExprLegacy symbolic backend should build combustion sparse bundle");

        let mut atom_request_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_request = atom_request_solver.build_solver_request(None, None);
        let (mut atom_bundle, atom_bundle_ms) = measure_sparse_bundle_build_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        )
        .expect("AtomView symbolic backend should build combustion sparse bundle");

        let args = DVector::from_vec(
            (0..legacy_bundle.variable_string.len())
                .map(|index| 0.25 + index as f64 * 0.0025)
                .collect(),
        );
        let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
            &mut legacy_bundle,
            &mut atom_bundle,
            &args,
            &format!("combustion-solver-stress-{n_steps}"),
        );

        let mut legacy_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_generate_begin = Instant::now();
        legacy_solver
            .try_eq_generate(None, None)
            .expect("ExprLegacy combustion generate should succeed");
        let legacy_generate_ms = legacy_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_solve_begin = Instant::now();
        legacy_solver
            .try_solve()
            .expect("ExprLegacy combustion solve should succeed");
        let legacy_solve_ms = legacy_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_solution = legacy_solver
            .get_result()
            .expect("ExprLegacy combustion solve should produce a solution");

        let mut atom_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_generate_begin = Instant::now();
        atom_solver
            .try_eq_generate(None, None)
            .expect("AtomView combustion generate should succeed");
        let atom_generate_ms = atom_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_solve_begin = Instant::now();
        atom_solver
            .try_solve()
            .expect("AtomView combustion solve should succeed");
        let atom_solve_ms = atom_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_solution = atom_solver
            .get_result()
            .expect("AtomView combustion solve should produce a solution");

        let max_diff_vs_legacy = legacy_solution
            .iter()
            .zip(atom_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);

        rows.push(SymbolicAssemblyStressRow {
            backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            n_steps,
            bundle_ms: legacy_bundle_ms,
            generate_ms: legacy_generate_ms,
            solve_ms: legacy_solve_ms,
            residual_max_diff: 0.0,
            jacobian_max_diff: 0.0,
            max_diff_vs_legacy: 0.0,
        });
        rows.push(SymbolicAssemblyStressRow {
            backend: BvpSymbolicAssemblyBackend::AtomView,
            n_steps,
            bundle_ms: atom_bundle_ms,
            generate_ms: atom_generate_ms,
            solve_ms: atom_solve_ms,
            residual_max_diff,
            jacobian_max_diff,
            max_diff_vs_legacy,
        });
    }

    println!("[BVP symbolic assembly solver stress] combustion ExprLegacy vs AtomView");
    println!(
        "{:<12} | {:>7} | {:>10} | {:>12} | {:>10} | {:>16} | {:>16} | {:>16}",
        "backend",
        "n_steps",
        "bundle_ms",
        "generate_ms",
        "solve_ms",
        "residual_diff",
        "jacobian_diff",
        "max_diff_vs_legacy"
    );
    println!("{}", "-".repeat(124));
    for row in &rows {
        let backend = match row.backend {
            BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
            BvpSymbolicAssemblyBackend::AtomView => "AtomView",
        };
        println!(
            "{:<12} | {:>7} | {:>10.3} | {:>12.3} | {:>10.3} | {:>16.6e} | {:>16.6e} | {:>16.6e}",
            backend,
            row.n_steps,
            row.bundle_ms,
            row.generate_ms,
            row.solve_ms,
            row.residual_max_diff,
            row.jacobian_max_diff,
            row.max_diff_vs_legacy
        );
    }
}

#[test]
#[ignore = "diagnostic repeated 300-step combustion compare with min/median/max for ExprLegacy vs AtomView"]
fn symbolic_assembly_backends_report_combustion_solver_stress_stats_1000() {
    aot_test_report!(symbolic_assembly_backends_report_combustion_solver_stress_stats_1000);
    #[derive(Debug)]
    struct StressSample {
        bundle_ms: f64,
        generate_ms: f64,
        solve_ms: f64,
        residual_max_diff: f64,
        jacobian_max_diff: f64,
        max_diff_vs_legacy: f64,
    }

    #[derive(Debug)]
    struct StageStats {
        min: f64,
        median: f64,
        max: f64,
    }

    fn compute_stage_stats(mut values: Vec<f64>) -> StageStats {
        values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
        let len = values.len();
        let median = if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) * 0.5
        } else {
            values[len / 2]
        };
        StageStats {
            min: values[0],
            median,
            max: values[len - 1],
        }
    }

    let iterations = 5usize;
    let n_steps = 200usize;
    let base_config =
        crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default();

    let mut legacy_samples = Vec::with_capacity(iterations);
    let mut atom_samples = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut legacy_request_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_request = legacy_request_solver.build_solver_request(None, None);
        let (mut legacy_bundle, legacy_bundle_ms) =
            measure_sparse_bundle_build_with_symbolic_backend(
                legacy_request,
                BvpSymbolicAssemblyBackend::ExprLegacy,
            )
            .expect("ExprLegacy symbolic backend should build combustion sparse bundle");

        let mut atom_request_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_request = atom_request_solver.build_solver_request(None, None);
        let (mut atom_bundle, atom_bundle_ms) = measure_sparse_bundle_build_with_symbolic_backend(
            atom_request,
            BvpSymbolicAssemblyBackend::AtomView,
        )
        .expect("AtomView symbolic backend should build combustion sparse bundle");

        let args = DVector::from_vec(
            (0..legacy_bundle.variable_string.len())
                .map(|index| 0.25 + index as f64 * 0.0025)
                .collect(),
        );
        let (residual_max_diff, jacobian_max_diff) = compare_sparse_bundles_numerically(
            &mut legacy_bundle,
            &mut atom_bundle,
            &args,
            "combustion-solver-stress-stats-1000",
        );

        let mut legacy_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy),
        );
        let legacy_generate_begin = Instant::now();
        legacy_solver
            .try_eq_generate(None, None)
            .expect("ExprLegacy combustion generate should succeed");
        let legacy_generate_ms = legacy_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_solve_begin = Instant::now();
        legacy_solver
            .try_solve()
            .expect("ExprLegacy combustion solve should succeed");
        let legacy_solve_ms = legacy_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let legacy_solution = legacy_solver
            .get_result()
            .expect("ExprLegacy combustion solve should produce a solution");

        let mut atom_solver = make_combustion_solver(
            n_steps,
            base_config
                .clone()
                .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView),
        );
        let atom_generate_begin = Instant::now();
        atom_solver
            .try_eq_generate(None, None)
            .expect("AtomView combustion generate should succeed");
        let atom_generate_ms = atom_generate_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_solve_begin = Instant::now();
        atom_solver
            .try_solve()
            .expect("AtomView combustion solve should succeed");
        let atom_solve_ms = atom_solve_begin.elapsed().as_secs_f64() * 1_000.0;
        let atom_solution = atom_solver
            .get_result()
            .expect("AtomView combustion solve should produce a solution");

        let max_diff_vs_legacy = legacy_solution
            .iter()
            .zip(atom_solution.iter())
            .map(|(&lhs, &rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);

        legacy_samples.push(StressSample {
            bundle_ms: legacy_bundle_ms,
            generate_ms: legacy_generate_ms,
            solve_ms: legacy_solve_ms,
            residual_max_diff: 0.0,
            jacobian_max_diff: 0.0,
            max_diff_vs_legacy: 0.0,
        });
        atom_samples.push(StressSample {
            bundle_ms: atom_bundle_ms,
            generate_ms: atom_generate_ms,
            solve_ms: atom_solve_ms,
            residual_max_diff,
            jacobian_max_diff,
            max_diff_vs_legacy,
        });
    }

    let legacy_bundle =
        compute_stage_stats(legacy_samples.iter().map(|row| row.bundle_ms).collect());
    let legacy_generate =
        compute_stage_stats(legacy_samples.iter().map(|row| row.generate_ms).collect());
    let legacy_solve = compute_stage_stats(legacy_samples.iter().map(|row| row.solve_ms).collect());

    let atom_bundle = compute_stage_stats(atom_samples.iter().map(|row| row.bundle_ms).collect());
    let atom_generate =
        compute_stage_stats(atom_samples.iter().map(|row| row.generate_ms).collect());
    let atom_solve = compute_stage_stats(atom_samples.iter().map(|row| row.solve_ms).collect());

    let atom_residual = compute_stage_stats(
        atom_samples
            .iter()
            .map(|row| row.residual_max_diff)
            .collect(),
    );
    let atom_jacobian = compute_stage_stats(
        atom_samples
            .iter()
            .map(|row| row.jacobian_max_diff)
            .collect(),
    );
    let atom_solution_diff = compute_stage_stats(
        atom_samples
            .iter()
            .map(|row| row.max_diff_vs_legacy)
            .collect(),
    );

    println!(
        "[BVP symbolic assembly solver stress stats] combustion ExprLegacy vs AtomView, n_steps=1000, runs={iterations}"
    );
    println!(
        "{:<12} | {:<8} | {:>10} | {:>10} | {:>10}",
        "backend", "stage", "min_ms", "median_ms", "max_ms"
    );
    println!("{}", "-".repeat(63));
    for (backend, stage, stats) in [
        ("ExprLegacy", "bundle", &legacy_bundle),
        ("ExprLegacy", "generate", &legacy_generate),
        ("ExprLegacy", "solve", &legacy_solve),
        ("AtomView", "bundle", &atom_bundle),
        ("AtomView", "generate", &atom_generate),
        ("AtomView", "solve", &atom_solve),
    ] {
        println!(
            "{:<12} | {:<8} | {:>10.3} | {:>10.3} | {:>10.3}",
            backend, stage, stats.min, stats.median, stats.max
        );
    }

    println!("[BVP symbolic assembly solver stress stats] AtomView numeric diffs, n_steps=300");
    println!(
        "residual_diff   min/median/max = {:.6e} / {:.6e} / {:.6e}",
        atom_residual.min, atom_residual.median, atom_residual.max
    );
    println!(
        "jacobian_diff   min/median/max = {:.6e} / {:.6e} / {:.6e}",
        atom_jacobian.min, atom_jacobian.median, atom_jacobian.max
    );
    println!(
        "solution_diff   min/median/max = {:.6e} / {:.6e} / {:.6e}",
        atom_solution_diff.min, atom_solution_diff.median, atom_solution_diff.max
    );
}
