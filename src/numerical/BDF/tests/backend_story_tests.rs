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
        let mut continued = make_solver(assembly, 0.0, DVector::from_element(1, 1.0), 0.1, 2.5);
        continued.try_solve().expect("first parameter segment");
        let (times, states) = continued.get_result_ref();
        let t_join = times[times.len() - 1];
        let y_join = states[(states.nrows() - 1, 0)];
        let prepare_calls = continued.get_statistics().backend_prepare_calls;
        assert_eq!(
            prepare_calls, 1,
            "one symbolic preparation before continuation"
        );
        assert!(matches!(
            continued.try_continue_with_parameter_values(DVector::zeros(0), 0.2),
            Err(BdfSolveError::Backend(_))
        ));
        let (times_after_invalid_rebind, states_after_invalid_rebind) = continued.get_result_ref();
        assert_eq!(
            times_after_invalid_rebind[times_after_invalid_rebind.len() - 1],
            t_join
        );
        assert_eq!(
            states_after_invalid_rebind[(states_after_invalid_rebind.nrows() - 1, 0)],
            y_join
        );
        assert_eq!(
            continued.get_statistics().backend_prepare_calls,
            prepare_calls
        );

        continued
            .try_continue_with_parameter_values(DVector::from_element(1, 3.0), 0.2)
            .expect("parameter continuation restart");
        assert_eq!(continued.get_status(), "running", "{assembly:?}");
        assert_eq!(
            continued.get_statistics().backend_prepare_calls,
            prepare_calls
        );
        let (segment_times, segment_states) = continued.get_result_ref();
        assert_eq!(
            segment_times.len(),
            0,
            "new segment clears old segment output"
        );
        assert_eq!(segment_states.nrows(), 0);
        continued.try_solve().expect("second parameter segment");

        let mut fresh = make_solver(assembly, t_join, DVector::from_element(1, y_join), 0.2, 3.0);
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
        assert_eq!(
            continued.get_statistics().backend_prepare_calls,
            prepare_calls
        );
        println!(
            "[BDF parameter continuation] assembly={assembly:?} parameter=2.5->3.0 prepare_calls={} final_diff={:.3e} status=ok",
            continued.get_statistics().backend_prepare_calls,
            (continued_final - fresh_final).abs()
        );
    }
}

#[test]
fn bdf_prepared_model_restarts_with_new_initial_state_without_repreparation() {
    let equation = crate::symbolic::symbolic_engine::Expr::parse_expression("-rate*y");
    let make_solver = |assembly, t0, y0, t_bound| {
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
        .with_equation_parameter_values(DVector::from_element(1, 2.5))
        .with_symbolic_assembly_backend(assembly)
        .with_telemetry_mode(BdfTelemetryMode::Counters);
        ODEsolver::new_with_options(options)
    };

    for assembly in [
        IvpSymbolicAssemblyBackend::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView,
    ] {
        let mut restarted = make_solver(assembly, 0.0, DVector::from_element(1, 1.0), 0.1);
        restarted.try_solve().expect("initial prepared segment");
        let preparation_calls = restarted.get_statistics().backend_prepare_calls;
        let (times, states) = restarted.get_result_ref();
        let t0 = times[times.len() - 1];
        let y0 = states[(states.nrows() - 1, 0)];

        assert!(matches!(
            restarted.try_restart_with_initial_state(t0, DVector::from_vec(vec![y0, 0.0]), 0.2,),
            Err(BdfSolveError::Configuration(
                crate::numerical::BDF::BDF_solver::BdfConfigurationError::InitialStateDimension
            ))
        ));

        restarted
            .try_restart_with_initial_state(t0, DVector::from_element(1, y0), 0.2)
            .expect("prepared model restart");
        assert_eq!(
            restarted.get_statistics().backend_prepare_calls,
            preparation_calls
        );
        assert!(restarted.get_result_ref().0.is_empty());
        restarted.try_solve().expect("restarted prepared segment");

        let mut fresh = make_solver(assembly, t0, DVector::from_element(1, y0), 0.2);
        fresh.try_solve().expect("fresh restart reference");
        let (_, restarted_states) = restarted.get_result_ref();
        let (_, fresh_states) = fresh.get_result_ref();
        let restarted_final = restarted_states[(restarted_states.nrows() - 1, 0)];
        let fresh_final = fresh_states[(fresh_states.nrows() - 1, 0)];
        assert_eq!(restarted.get_status(), "finished");
        assert_eq!(fresh.get_status(), "finished");
        assert!((restarted_final - fresh_final).abs() <= 2e-11);
        println!(
            "[BDF prepared restart] assembly={assembly:?} preparation_calls={} final_diff={:.3e} status=ok",
            restarted.get_statistics().backend_prepare_calls,
            (restarted_final - fresh_final).abs()
        );
    }
}

#[test]
fn bdf_long_parameter_series_reuses_prepared_model_and_bounds_segment_output() {
    let equation = crate::symbolic::symbolic_engine::Expr::parse_expression("-rate*y");
    let options = BdfSolverOptions::for_bdf(
        vec![equation],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_element(1, 1.0),
        0.01,
        0.002,
        1e-8,
        1e-11,
        None,
        false,
        Some(0.001),
    )
    .with_equation_parameters(vec!["rate".to_string()])
    .with_equation_parameter_values(DVector::from_element(1, 2.0))
    .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut solver = ODEsolver::new_with_options(options);
    solver.try_solve().expect("initial parameter segment");
    let preparation_calls = solver.get_statistics().backend_prepare_calls;

    for index in 0..24 {
        let (times, states) = solver.get_result_ref();
        let t0 = *times.as_slice().last().expect("segment has a boundary");
        let y0 = states.row(states.nrows() - 1).transpose();
        let next_rate = 2.0 + index as f64 * 0.01;
        solver
            .try_continue_with_parameter_values(DVector::from_element(1, next_rate), t0 + 0.01)
            .expect("parameter segment restart");
        assert!(solver.get_result_ref().0.is_empty());
        solver.try_solve().expect("continued parameter segment");
        let (_, segment_states) = solver.get_result_ref();
        assert!(
            segment_states.nrows() < 256,
            "segment output must stay bounded"
        );
        assert_eq!(segment_states.ncols(), y0.len());
        assert_eq!(
            solver.get_statistics().backend_prepare_calls,
            preparation_calls
        );
    }

    println!(
        "[BDF long parameter series] segments=24 preparation_calls={} final_rows={} status={}",
        solver.get_statistics().backend_prepare_calls,
        solver.get_result_ref().1.nrows(),
        solver.get_status()
    );
    assert_eq!(solver.get_status(), "finished");
}

#[test]
fn bdf_parameter_continuation_matches_fresh_segments_on_shared_workloads() {
    use crate::numerical::ivp_workloads::parameter_continuation_target;

    fn make_solver(
        kind: WorkloadKind,
        dimension: usize,
        assembly: IvpSymbolicAssemblyBackend,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        parameters: DVector<f64>,
    ) -> ODEsolver {
        let workload = build_workload(kind, dimension);
        let max_step = match kind {
            WorkloadKind::CombustionLike => 1.0e-4,
            WorkloadKind::DiffusionChain => 2.5e-4,
            _ => unreachable!("continuation matrix only uses parameterized workloads"),
        };
        let options = BdfSolverOptions::for_bdf(
            workload.equations,
            workload.variables,
            workload.time_variable,
            t0,
            y0,
            t_bound,
            max_step,
            1.0e-7,
            1.0e-10,
            None,
            false,
            Some(max_step),
        )
        .with_equation_parameters(workload.parameter_names)
        .with_equation_parameter_values(parameters)
        .with_symbolic_assembly_backend(assembly)
        .with_telemetry_mode(BdfTelemetryMode::Counters);
        ODEsolver::new_with_options(options)
    }

    let workloads = [
        (WorkloadKind::CombustionLike, 1, 2.0e-4),
        (WorkloadKind::DiffusionChain, 6, 5.0e-4),
    ];
    println!(
        "[BDF parameter continuation matrix] oracle=fresh BDF segment; workloads=combustion-like,diffusion-chain; assemblies=ExprLegacy,AtomView"
    );
    println!(
        "workload | dimension | assembly | segments | max_state_diff | preparation_calls | status"
    );

    for (kind, dimension, segment_width) in workloads {
        let base = build_workload(kind, dimension);
        for assembly in [
            IvpSymbolicAssemblyBackend::ExprLegacy,
            IvpSymbolicAssemblyBackend::AtomView,
        ] {
            let mut continued = make_solver(
                kind,
                dimension,
                assembly,
                0.0,
                base.initial_state.clone(),
                segment_width,
                base.parameter_values.clone(),
            );
            continued.try_solve().expect("initial workload segment");
            let preparation_calls = continued.get_statistics().backend_prepare_calls;
            assert_eq!(preparation_calls, 1, "{kind:?}/{assembly:?}");

            let mut max_diff = 0.0_f64;
            for index in 1..=3 {
                let (times, states) = continued.get_result_ref();
                let t0 = *times
                    .as_slice()
                    .last()
                    .expect("continued segment has final time");
                let y0 = states.row(states.nrows() - 1).transpose();
                let target = parameter_continuation_target(&base.parameter_values, index);
                let t_bound = t0 + segment_width;

                continued
                    .try_continue_with_parameter_values(target.clone(), t_bound)
                    .expect("parameterized workload continuation");
                continued.try_solve().expect("continued workload segment");

                let mut fresh = make_solver(kind, dimension, assembly, t0, y0, t_bound, target);
                fresh.try_solve().expect("fresh workload segment reference");
                let (_, continued_states) = continued.get_result_ref();
                let (_, fresh_states) = fresh.get_result_ref();
                let continued_final = continued_states.row(continued_states.nrows() - 1);
                let fresh_final = fresh_states.row(fresh_states.nrows() - 1);
                let diff = continued_final
                    .iter()
                    .zip(fresh_final.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                max_diff = max_diff.max(diff);
                assert!(
                    diff <= 2.0e-9,
                    "{kind:?}/{assembly:?} segment {index} continuation drift={diff:e}"
                );
                assert_eq!(
                    continued.get_statistics().backend_prepare_calls,
                    preparation_calls,
                    "numeric parameter continuation must reuse prepared symbolic callbacks"
                );
                assert_eq!(continued.get_status(), "finished");
            }

            println!(
                "{} | {} | {assembly:?} | 3 | {max_diff:.3e} | {} | ok",
                kind.label(),
                if kind == WorkloadKind::DiffusionChain {
                    dimension
                } else {
                    3
                },
                continued.get_statistics().backend_prepare_calls
            );
        }
    }
}

#[test]
fn bdf_piecewise_parameter_continuation_matches_lsode2_and_backward_euler() {
    use crate::numerical::BE::{BE, BeTelemetryMode};
    use crate::numerical::LSODE2::config::{
        Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend,
        Lsode2SymbolicExecutionMode,
    };
    use crate::numerical::LSODE2::solver::Lsode2Solver;
    use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
    use crate::symbolic::symbolic_engine::Expr;

    const SPLIT: f64 = 0.1;
    const END: f64 = 0.2;
    const RATE_0: f64 = 2.5;
    const RATE_1: f64 = 3.0;
    let equation = Expr::parse_expression("-rate*y");
    let expected = (-RATE_0 * SPLIT - RATE_1 * (END - SPLIT)).exp();

    let mut bdf = ODEsolver::new_with_options(
        BdfSolverOptions::for_bdf(
            vec![equation.clone()],
            vec!["y".into()],
            "t".into(),
            0.0,
            DVector::from_element(1, 1.0),
            SPLIT,
            0.0025,
            1.0e-10,
            1.0e-12,
            None,
            false,
            Some(0.001),
        )
        .with_equation_parameters(vec!["rate".into()])
        .with_equation_parameter_values(DVector::from_element(1, RATE_0)),
    );
    bdf.try_solve().expect("BDF first piecewise segment");
    bdf.try_continue_with_parameter_values(DVector::from_element(1, RATE_1), END)
        .expect("BDF parameter change at accepted boundary");
    bdf.try_solve().expect("BDF second piecewise segment");
    let bdf_final = bdf.get_result_ref().1[(bdf.get_result_ref().1.nrows() - 1, 0)];

    let make_lsode_segment = |t0: f64, y0: f64, t_bound: f64, rate: f64| {
        let config = Lsode2ProblemConfig::new(
            vec![equation.clone()],
            vec!["y".into()],
            "t".into(),
            t0,
            DVector::from_element(1, y0),
            t_bound,
            0.001,
            1.0e-10,
            1.0e-12,
        )
        .with_equation_parameters(vec!["rate".into()])
        .with_equation_parameter_values(DVector::from_element(1, rate))
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential);
        let mut solver = Lsode2Solver::new(config).expect("LSODE2 reference construction");
        solver.solve().expect("LSODE2 reference segment");
        solver.get_result().1[(solver.get_result().1.nrows() - 1, 0)]
    };
    let lsode_mid = make_lsode_segment(0.0, 1.0, SPLIT, RATE_0);
    let lsode_final = make_lsode_segment(SPLIT, lsode_mid, END, RATE_1);

    let mut be = BE::new();
    be.try_set_initial(
        vec![equation],
        vec!["y".into()],
        "t".into(),
        1.0e-12,
        20,
        Some(5.0e-5),
        0.0,
        SPLIT,
        DVector::from_element(1, 1.0),
    )
    .expect("BE initial piecewise segment");
    be.try_set_equation_parameters(Some(&["rate"]))
        .expect("BE parameter schema");
    be.set_parameter_values(DVector::from_element(1, RATE_0))
        .expect("BE initial rate");
    be.try_set_telemetry_mode(BeTelemetryMode::Off)
        .expect("disable nonessential test telemetry");
    be.try_solve().expect("BE first piecewise segment");
    be.set_parameter_values(DVector::from_element(1, RATE_1))
        .expect("BE parameter change");
    be.try_continue_to(END)
        .expect("BE second piecewise segment");
    let be_result = be.get_result().1.expect("BE trajectory");
    let be_final = be_result[(be_result.nrows() - 1, 0)];

    let bdf_error = (bdf_final - expected).abs();
    let lsode_error = (lsode_final - expected).abs();
    let be_error = (be_final - expected).abs();
    assert!(bdf_error <= 2.0e-8, "BDF analytic error={bdf_error:e}");
    assert!(
        lsode_error <= 2.0e-8,
        "LSODE2 analytic error={lsode_error:e}"
    );
    assert!(
        (bdf_final - lsode_final).abs() <= 2.0e-8,
        "BDF/LSODE2 cross-solver drift={:e}",
        (bdf_final - lsode_final).abs()
    );
    assert!(
        be_error <= 2.0e-4,
        "BE fixed-step analytic error={be_error:e}"
    );
    println!(
        "[BDF continuation cross-solver reference] piecewise_decay=2.5->3.0; exact={expected:.12e}; BDF={bdf_final:.12e}; LSODE2={lsode_final:.12e}; BE(h=5e-5)={be_final:.12e}; errors={bdf_error:.3e}/{lsode_error:.3e}/{be_error:.3e}; status=ok"
    );
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
            .unwrap_or_else(|error| {
                panic!("{assembly:?} AOT continuation restart failed: {error}")
            });
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
        let producer_final =
            producer.get_result_ref().1[(producer.get_result_ref().1.nrows() - 1, 0)];
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

#[test]
#[ignore = "release paired cold-AOT E2E diagnostic; requires working tcc toolchain"]
fn bdf_robertson_aot_cold_e2e_routes_alternate_for_noise_check() {
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
    use std::time::Instant;

    fn summary(samples: &[f64]) -> (f64, f64, f64) {
        let mut ordered = samples.to_vec();
        ordered.sort_by(f64::total_cmp);
        let median = ordered[ordered.len() / 2];
        (median, ordered[0], ordered[ordered.len() - 1])
    }

    let repetitions = 9;
    let routes = [
        (0, IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (1, IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];
    let (_, reference, reference_status) = run_workload(
        WorkloadKind::Robertson,
        IvpSymbolicAssemblyBackend::ExprLegacy,
    );
    assert_eq!(reference_status, "finished");
    let mut samples: [Vec<(f64, f64, f64)>; 2] = [Vec::new(), Vec::new()];

    eprintln!(
        "[BDF Robertson paired cold-AOT] repetitions={repetitions}; compiler=tcc; build=RebuildAlways; route order alternates; timings are diagnostics, no performance threshold"
    );
    eprintln!(
        "route | prepare_median_ms | prepare_min_ms | prepare_max_ms | solve_median_ms | solve_min_ms | solve_max_ms | e2e_median_ms | e2e_min_ms | e2e_max_ms"
    );

    for repetition in 0..repetitions {
        let ordered_routes = if repetition % 2 == 0 {
            routes
        } else {
            [routes[1], routes[0]]
        };
        for (index, assembly, route) in ordered_routes {
            let artifact_dir = tempfile::tempdir().expect("isolated Robertson AOT directory");
            let config = SymbolicIvpGeneratedBackendConfig::new()
                .with_output_parent_dir(Some(artifact_dir.path().to_path_buf()))
                .with_build_policy(crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                })
                .with_c_tcc();
            let workload = build_workload(WorkloadKind::Robertson, 3);
            let mut options = BdfSolverOptions::for_bdf(
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
            .with_symbolic_assembly_backend(assembly)
            .with_generated_backend_config(config);
            if !workload.parameter_names.is_empty() {
                options = options
                    .with_equation_parameters(workload.parameter_names)
                    .with_equation_parameter_values(workload.parameter_values);
            }
            let mut solver = ODEsolver::new_with_options(options);

            let e2e_started = Instant::now();
            let prepare_started = Instant::now();
            solver
                .try_generate()
                .unwrap_or_else(|error| panic!("Robertson {route} AOT prepare failed: {error}"));
            let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
            let solve_started = Instant::now();
            solver
                .try_solve()
                .unwrap_or_else(|error| panic!("Robertson {route} AOT solve failed: {error}"));
            let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
            let e2e_ms = e2e_started.elapsed().as_secs_f64() * 1_000.0;
            assert_eq!(solver.get_status(), "finished", "Robertson {route}");

            let (_, trajectory) = solver.get_result_ref();
            let state = trajectory.row(trajectory.nrows() - 1).transpose();
            let max_diff = reference
                .iter()
                .zip(state.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff <= 2e-5,
                "Robertson {route} AOT parity drift={max_diff:e}"
            );
            samples[index].push((prepare_ms, solve_ms, e2e_ms));
        }
    }

    for (index, _, route) in routes {
        let prepare = summary(
            &samples[index]
                .iter()
                .map(|sample| sample.0)
                .collect::<Vec<_>>(),
        );
        let solve = summary(
            &samples[index]
                .iter()
                .map(|sample| sample.1)
                .collect::<Vec<_>>(),
        );
        let e2e = summary(
            &samples[index]
                .iter()
                .map(|sample| sample.2)
                .collect::<Vec<_>>(),
        );
        eprintln!(
            "{route} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}",
            prepare.0, prepare.1, prepare.2, solve.0, solve.1, solve.2, e2e.0, e2e.1, e2e.2
        );
    }
}
