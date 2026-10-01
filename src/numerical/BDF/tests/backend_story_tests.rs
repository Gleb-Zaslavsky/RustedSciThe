use crate::numerical::BDF::BDF_api::{
    BdfSolveError, BdfSolverOptions, BdfTelemetryMode, ODEsolver,
};
use crate::numerical::ivp_workloads::{WorkloadKind, build_workload};
use crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
use nalgebra::DVector;

fn run_workload(
    kind: WorkloadKind,
    assembly: IvpSymbolicAssemblyBackend,
) -> (Vec<f64>, DVector<f64>, &'static str) {
    let workload = build_workload(kind, 3);
    let (t_bound, max_step) = match kind {
        WorkloadKind::StiffScalar => (0.05, 0.001),
        WorkloadKind::Robertson => (0.1, 0.002),
        WorkloadKind::CombustionLike => (0.01, 0.0005),
        _ => panic!("unsupported backend story workload: {}", kind.label()),
    };

    let mut options = BdfSolverOptions::for_bdf(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        t_bound,
        max_step,
        1e-7,
        1e-10,
        None,
        false,
        Some(max_step),
    )
    .with_symbolic_assembly_backend(assembly);
    if !workload.parameter_names.is_empty() {
        options = options
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }

    let mut solver = ODEsolver::new_with_options(options);
    solver.try_generate().unwrap_or_else(|error| {
        panic!(
            "{} {:?} preparation failed: {error}",
            kind.label(),
            assembly
        )
    });
    solver.solve();
    let (times, states) = solver.get_result_ref();
    assert!(!times.is_empty(), "{} returned no samples", kind.label());
    let final_state = states.row(states.nrows() - 1).transpose();
    (times.as_slice().to_vec(), final_state, solver.get_status())
}

#[test]
fn bdf_symbolic_assembly_lambdify_routes_match_shared_workloads() {
    let workloads = [
        WorkloadKind::StiffScalar,
        WorkloadKind::Robertson,
        WorkloadKind::CombustionLike,
    ];
    println!("[BDF symbolic backend parity] execution=Lambdify; comparison=ExprLegacy vs AtomView");
    println!(
        "workload | expr_status | atom_status | final_state_linf_diff | expr_samples | atom_samples"
    );

    for workload in workloads {
        let (expr_times, expr_state, expr_status) =
            run_workload(workload, IvpSymbolicAssemblyBackend::ExprLegacy);
        let (atom_times, atom_state, atom_status) =
            run_workload(workload, IvpSymbolicAssemblyBackend::AtomView);
        let final_diff = expr_state
            .iter()
            .zip(atom_state.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);

        assert_eq!(
            expr_status,
            "finished",
            "{} ExprLegacy route",
            workload.label()
        );
        assert_eq!(
            atom_status,
            "finished",
            "{} AtomView route",
            workload.label()
        );
        assert!(
            final_diff <= 2e-5,
            "{} frontend final-state drift {final_diff:e}",
            workload.label()
        );
        println!(
            "{} | {} | {} | {:.3e} | {} | {}",
            workload.label(),
            expr_status,
            atom_status,
            final_diff,
            expr_times.len(),
            atom_times.len()
        );
    }
}

#[test]
fn bdf_parameterized_scalar_has_independent_analytic_correctness_oracle() {
    let equation = crate::symbolic::symbolic_engine::Expr::parse_expression("-rate*y");
    let expected = (-2.5_f64 * 0.2).exp();

    for assembly in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let options = BdfSolverOptions::for_bdf(
            vec![equation.clone()],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_element(1, 1.0),
            0.2,
            0.01,
            1e-9,
            1e-12,
            None,
            false,
            Some(0.01),
        )
        .with_equation_parameters(vec!["rate".to_string()])
        .with_equation_parameter_values(DVector::from_element(1, 2.5))
        .with_symbolic_assembly_backend(assembly);
        let mut solver = ODEsolver::new_with_options(options);
        solver.solve();

        let (_, states) = solver.get_result_ref();
        let actual = states[(states.nrows() - 1, 0)];
        assert_eq!(solver.get_status(), "finished", "{assembly:?}");
        assert!(
            (actual - expected).abs() <= 2e-7,
            "{assembly:?} analytic error={:e}",
            (actual - expected).abs()
        );
    }
}

#[test]
fn bdf_parameter_rebind_then_regenerate_matches_fresh_solve() {
    let equation = crate::symbolic::symbolic_engine::Expr::parse_expression("-rate*y");
    let make_solver = |assembly, rate| {
        let options = BdfSolverOptions::for_bdf(
            vec![equation.clone()],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_element(1, 1.0),
            0.2,
            0.01,
            1e-9,
            1e-12,
            None,
            false,
            Some(0.01),
        )
        .with_equation_parameters(vec!["rate".to_string()])
        .with_equation_parameter_values(DVector::from_element(1, rate))
        .with_symbolic_assembly_backend(assembly);
        ODEsolver::new_with_options(options)
    };

    for assembly in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let mut rebound = make_solver(assembly, 2.5);
        rebound.solve();
        rebound
            .set_parameter_values(DVector::from_element(1, 3.0))
            .expect("valid parameter rebind");
        rebound
            .try_generate()
            .expect("explicit re-generation resets BDF integration state");
        assert_eq!(
            rebound.get_status(),
            "running",
            "{assembly:?} regenerated solver status"
        );
        assert!(
            rebound.get_result_ref().0.is_empty() && rebound.get_result_ref().1.nrows() == 0,
            "{assembly:?} regeneration must discard the previous trajectory"
        );
        rebound.solve();

        let mut fresh = make_solver(assembly, 3.0);
        fresh.solve();
        let (_, rebound_states) = rebound.get_result_ref();
        let (_, fresh_states) = fresh.get_result_ref();
        let rebound_final = rebound_states[(rebound_states.nrows() - 1, 0)];
        let fresh_final = fresh_states[(fresh_states.nrows() - 1, 0)];
        assert_eq!(
            rebound_states.nrows(),
            fresh_states.nrows(),
            "{assembly:?} regenerated trajectory must not stop at the first step"
        );
        assert!(rebound_states.nrows() > 2, "{assembly:?} full trajectory");
        assert!((rebound.get_result_ref().0[rebound_states.nrows() - 1] - 0.2).abs() < 1e-14);
        assert_eq!(rebound.get_status(), "finished", "{assembly:?} rebound");
        assert_eq!(fresh.get_status(), "finished", "{assembly:?} fresh");
        assert!(
            (rebound_final - fresh_final).abs() <= 1e-12,
            "{assembly:?} rebound/fresh drift={:e}",
            (rebound_final - fresh_final).abs()
        );
        println!(
            "[BDF parameter rebind] assembly={assembly:?} rebind=3.0 explicit_regenerate=true fresh_diff={:.3e} status=ok",
            (rebound_final - fresh_final).abs()
        );
    }
}

#[test]
fn bdf_parameter_continuation_reuses_prepared_callbacks_and_restarts_history() {
    let equation = crate::symbolic::symbolic_engine::Expr::parse_expression("-rate*y");
    let make_solver = |assembly, t0, y0, t_bound, rate| {
        let options = BdfSolverOptions::for_bdf(
            vec![equation.clone()],
            vec!["y".to_string()],
            "t".to_string(),
            t0,
            y0,
            t_bound,
            0.01,
            1e-9,
            1e-12,
            None,
            false,
            Some(0.005),
        )
        .with_equation_parameters(vec!["rate".to_string()])
        .with_equation_parameter_values(DVector::from_element(1, rate))
        .with_symbolic_assembly_backend(assembly)
        .with_telemetry_mode(BdfTelemetryMode::Counters);
        ODEsolver::new_with_options(options)
    };

    for assembly in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let mut continued = make_solver(
            assembly,
            0.0,
            DVector::from_element(1, 1.0),
            0.1,
            2.5,
        );
        continued.try_solve().expect("first parameter segment");
        let (times, states) = continued.get_result_ref();
        let t_join = times[times.len() - 1];
        let y_join = states[(states.nrows() - 1, 0)];
        let prepare_calls = continued.get_statistics().backend_prepare_calls;
        assert_eq!(prepare_calls, 1, "one symbolic preparation before continuation");
        assert!(matches!(
            continued.try_continue_with_parameter_values(DVector::zeros(0), 0.2),
            Err(BdfSolveError::Backend(_))
        ));
        let (times_after_invalid_rebind, states_after_invalid_rebind) = continued.get_result_ref();
        assert_eq!(times_after_invalid_rebind[times_after_invalid_rebind.len() - 1], t_join);
        assert_eq!(states_after_invalid_rebind[(states_after_invalid_rebind.nrows() - 1, 0)], y_join);
        assert_eq!(continued.get_statistics().backend_prepare_calls, prepare_calls);

        continued
            .try_continue_with_parameter_values(DVector::from_element(1, 3.0), 0.2)
            .expect("parameter continuation restart");
        assert_eq!(continued.get_status(), "running", "{assembly:?}");
        assert_eq!(continued.get_statistics().backend_prepare_calls, prepare_calls);
        let (segment_times, segment_states) = continued.get_result_ref();
        assert_eq!(segment_times.len(), 0, "new segment clears old segment output");
        assert_eq!(segment_states.nrows(), 0);
        continued.try_solve().expect("second parameter segment");

        let mut fresh = make_solver(
            assembly,
            t_join,
            DVector::from_element(1, y_join),
            0.2,
            3.0,
        );
        fresh.try_solve().expect("fresh second-segment reference");
        let (continued_times, continued_states) = continued.get_result_ref();
        let (fresh_times, fresh_states) = fresh.get_result_ref();
        let continued_final = continued_states[(continued_states.nrows() - 1, 0)];
        let fresh_final = fresh_states[(fresh_states.nrows() - 1, 0)];
        assert_eq!(continued_times[0], t_join);
        assert_eq!(fresh_times[0], t_join);
        assert!((continued_times[continued_times.len() - 1] - 0.2).abs() < 1e-14);
        assert!((continued_final - fresh_final).abs() < 2e-11);
        assert_eq!(continued.get_status(), "finished");
        assert_eq!(continued.get_statistics().backend_prepare_calls, prepare_calls);
        println!(
            "[BDF parameter continuation] assembly={assembly:?} parameter=2.5->3.0 prepare_calls={} final_diff={:.3e} status=ok",
            continued.get_statistics().backend_prepare_calls,
            (continued_final - fresh_final).abs()
        );
    }
}

#[test]
fn bdf_robertson_routes_preserve_mass_and_nonnegative_state() {
    for assembly in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let (times, _, status) = run_workload(WorkloadKind::Robertson, assembly);
        assert_eq!(status, "finished", "{assembly:?}");
        assert!(times.windows(2).all(|pair| pair[1] > pair[0]));

        let workload = build_workload(WorkloadKind::Robertson, 3);
        let options = BdfSolverOptions::for_bdf(
            workload.equations,
            workload.variables,
            workload.time_variable,
            0.0,
            workload.initial_state,
            0.1,
            0.002,
            1e-7,
            1e-10,
            None,
            false,
            Some(0.002),
        )
        .with_symbolic_assembly_backend(assembly);
        let mut solver = ODEsolver::new_with_options(options);
        solver.solve();
        let (_, states) = solver.get_result_ref();
        for row in 0..states.nrows() {
            let mass = states[(row, 0)] + states[(row, 1)] + states[(row, 2)];
            assert!((mass - 1.0).abs() < 2e-5, "{assembly:?} mass={mass}");
            assert!(
                (0..3).all(|column| states[(row, column)] >= -1e-8),
                "{assembly:?} negative Robertson concentration at row {row}"
            );
        }
    }
}

#[test]
fn bdf_hundred_state_stiff_diagonal_matches_independent_analytic_reference() {
    let n = 100usize;
    let t_bound = 0.02;
    let rates: Vec<f64> = (0..n).map(|index| 1.0 + 10.0 * index as f64).collect();
    let initial: Vec<f64> = (0..n).map(|index| 0.2 + 0.001 * index as f64).collect();
    let equations = rates
        .iter()
        .enumerate()
        .map(|(index, rate)| {
            crate::symbolic::symbolic_engine::Expr::parse_expression(&format!("-{rate}*y{index}"))
        })
        .collect::<Vec<_>>();
    let variables = (0..n).map(|index| format!("y{index}")).collect::<Vec<_>>();

    for assembly in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let options = BdfSolverOptions::for_bdf(
            equations.clone(),
            variables.clone(),
            "t".to_string(),
            0.0,
            DVector::from_vec(initial.clone()),
            t_bound,
            0.001,
            2e-8,
            1e-11,
            None,
            false,
            Some(0.0005),
        )
        .with_symbolic_assembly_backend(assembly);
        let mut solver = ODEsolver::new_with_options(options);
        solver
            .try_solve()
            .unwrap_or_else(|error| panic!("{assembly:?} n={n} stiff solve failed: {error}"));
        assert_eq!(solver.get_status(), "finished", "{assembly:?}");
        let (times, states) = solver.get_result_ref();
        assert_eq!(states.ncols(), n);
        assert!((times[times.len() - 1] - t_bound).abs() < 1e-14);

        let max_scaled_error = (0..n)
            .map(|index| {
                let expected = initial[index] * (-rates[index] * t_bound).exp();
                let actual = states[(states.nrows() - 1, index)];
                (actual - expected).abs() / (1.0 + expected.abs())
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_scaled_error < 3e-7,
            "{assembly:?} n={n} analytic scaled error={max_scaled_error:e}"
        );
        println!(
            "[BDF dense stiff reference] assembly={assembly:?} n={n} samples={} max_scaled_error={max_scaled_error:.3e} status=ok",
            times.len()
        );
    }
}

#[test]
#[ignore = "release AOT lifecycle gate; requires a working tcc toolchain"]
fn bdf_parameterized_aot_cache_handoff_is_independent_per_assembly_backend() {
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::symbolic_ivp_generated::{
        SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    };
    use std::time::Instant;

    let root = tempfile::tempdir().expect("temporary AOT directory");
    let equation = crate::symbolic::symbolic_engine::Expr::parse_expression("-rate*y");

    for assembly in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let route_dir = root.path().join(format!("{assembly:?}"));
        let build_config = SymbolicIvpGeneratedBackendConfig::new()
            .with_output_parent_dir(Some(route_dir))
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
            .with_c_tcc();
        let make_options = |rate, t0, y0, t_bound, config| {
            BdfSolverOptions::for_bdf(
                vec![equation.clone()],
                vec!["y".to_string()],
                "t".to_string(),
                t0,
                y0,
                t_bound,
                0.01,
                1e-9,
                1e-12,
                None,
                false,
                Some(0.01),
            )
            .with_equation_parameters(vec!["rate".to_string()])
            .with_equation_parameter_values(DVector::from_element(1, rate))
            .with_symbolic_assembly_backend(assembly)
            .with_telemetry_mode(BdfTelemetryMode::Counters)
            .with_generated_backend_config(config)
        };

        let build_start = Instant::now();
        let mut producer = ODEsolver::new_with_options(make_options(
            2.5,
            0.0,
            DVector::from_element(1, 1.0),
            0.1,
            build_config,
        ));
        producer
            .try_generate()
            .unwrap_or_else(|error| panic!("{assembly:?} AOT producer failed: {error}"));
        let producer_prepare_ms = build_start.elapsed().as_secs_f64() * 1_000.0;
        let resolver = producer
            .generated_backend_config()
            .resolver
            .clone()
            .expect("AOT producer should publish a resolver");
        producer.solve();
        let prepare_calls_before_continuation = producer.get_statistics().backend_prepare_calls;
        producer
            .try_continue_with_parameter_values(DVector::from_element(1, 3.0), 0.2)
            .unwrap_or_else(|error| panic!("{assembly:?} AOT continuation restart failed: {error}"));
        assert_eq!(
            producer.get_statistics().backend_prepare_calls,
            prepare_calls_before_continuation,
            "AOT continuation must retain the prepared generated callbacks"
        );
        producer
            .try_solve()
            .unwrap_or_else(|error| panic!("{assembly:?} AOT continuation solve failed: {error}"));

        let consumer_config = producer
            .generated_backend_config()
            .clone()
            .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt);
        let consumer_start = Instant::now();
        let mut consumer = ODEsolver::new_with_options(make_options(
            3.0,
            0.0,
            DVector::from_element(1, 1.0),
            0.2,
            consumer_config,
        ));
        consumer
            .try_generate()
            .unwrap_or_else(|error| panic!("{assembly:?} AOT cache consumer failed: {error}"));
        let consumer_prepare_ms = consumer_start.elapsed().as_secs_f64() * 1_000.0;
        assert!(
            consumer.generated_backend_config().resolver.is_some(),
            "{assembly:?} RequirePrebuilt consumer lost resolver"
        );
        consumer.solve();

        let (_, state) = consumer.get_result_ref();
        let actual = state[(state.nrows() - 1, 0)];
        let expected = (-3.0_f64 * 0.2).exp();
        let producer_final = producer.get_result_ref().1[(producer.get_result_ref().1.nrows() - 1, 0)];
        let segmented_expected = (-2.5_f64 * 0.1 - 3.0_f64 * (0.2 - 0.1)).exp();
        assert_eq!(consumer.get_status(), "finished", "{assembly:?}");
        assert!(
            (actual - expected).abs() <= 2e-7,
            "{assembly:?} stale AOT parameters"
        );
        assert!(
            (producer_final - segmented_expected).abs() <= 2e-7,
            "{assembly:?} piecewise AOT continuation drift={:e}",
            (producer_final - segmented_expected).abs()
        );
        assert_eq!(resolver.registry().problem_keys().len(), 1);
        println!(
            "[BDF AOT lifecycle] assembly={assembly:?} producer_prepare_ms={producer_prepare_ms:.3} consumer_prepare_ms={consumer_prepare_ms:.3} parameter_rebind=2.5->3.0 continuation_reused_callbacks=true segmented_final={producer_final:.12e} segmented_expected={segmented_expected:.12e} consumer_final={actual:.12e} consumer_expected={expected:.12e} status=ok"
        );
    }
}
