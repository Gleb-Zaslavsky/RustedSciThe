use crate::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode, ODEsolver};
use crate::numerical::ivp_workloads::{WorkloadKind, build_workload, dense_coupled};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::ivp_telemetry::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetrySnapshot, IvpWarmStage,
};
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_problem,
};
use nalgebra::DVector;
use std::hint::black_box;
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn measured_solver(
    kind: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> ODEsolver {
    let workload = build_workload(kind, dimension.max(1));
    let (t_bound, max_step) = match kind {
        WorkloadKind::StiffScalar => (0.05, 0.001),
        WorkloadKind::Robertson => (0.1, 0.002),
        WorkloadKind::CombustionLike => (0.01, 0.0005),
        WorkloadKind::ThreeBody => (0.01, 0.001),
        WorkloadKind::DiffusionChain => (0.01, 0.001),
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
    .with_symbolic_assembly_backend(assembly)
    .with_telemetry_mode(BdfTelemetryMode::Timings);
    if !workload.parameter_names.is_empty() {
        options = options
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }
    ODEsolver::new_with_options(options)
}

fn dense_coupled_solver(
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    telemetry_mode: BdfTelemetryMode,
) -> ODEsolver {
    let workload = dense_coupled(dimension);
    let options = BdfSolverOptions::for_bdf(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.01,
        0.001,
        1e-7,
        1e-10,
        None,
        false,
        Some(0.001),
    )
    .with_symbolic_assembly_backend(assembly)
    .with_equation_parameters(workload.parameter_names)
    .with_equation_parameter_values(workload.parameter_values)
    .with_telemetry_mode(telemetry_mode);
    ODEsolver::new_with_options(options)
}

fn aot_prebuilt_solver(
    kind: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> (ODEsolver, TempDir) {
    let artifact_dir = tempfile::tempdir().expect("isolated BDF AOT artifact directory");
    let config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(artifact_dir.path().to_path_buf()))
        .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
            profile: if cfg!(debug_assertions) {
                AotBuildProfile::Debug
            } else {
                AotBuildProfile::Release
            },
        })
        .with_c_tcc();
    let mut producer =
        measured_solver(kind, dimension, assembly).with_generated_backend_config(config);
    producer
        .try_generate()
        .unwrap_or_else(|error| panic!("{kind:?} {assembly:?} AOT producer: {error}"));
    let consumer_config = producer
        .generated_backend_config()
        .clone()
        .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt);
    let consumer =
        measured_solver(kind, dimension, assembly).with_generated_backend_config(consumer_config);
    (consumer, artifact_dir)
}

fn final_state(solver: &ODEsolver) -> DVector<f64> {
    let (_, trajectory) = solver.get_result_ref();
    trajectory
        .row(trajectory.nrows() - 1)
        .transpose()
        .into_owned()
}

fn prepare_symbolic_frontend(
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> (f64, IvpTelemetrySnapshot) {
    let workload = dense_coupled(dimension);
    let telemetry = IvpTelemetry::detailed();
    let mut options = SymbolicIvpProblemOptions::new()
        .with_symbolic_assembly_backend(assembly)
        .with_telemetry(telemetry);
    options = options
        .with_equation_parameters(workload.parameter_names)
        .with_equation_parameter_values(workload.parameter_values);

    let started = Instant::now();
    let prepared = prepare_symbolic_ivp_problem(
        workload.equations,
        workload.variables,
        workload.time_variable,
        options,
    )
    .expect("symbolic IVP frontend preparation");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let snapshot = prepared.telemetry.snapshot();
    drop(prepared);
    (elapsed_ms, snapshot)
}

fn stage_ms(snapshot: &IvpTelemetrySnapshot, stage: IvpColdStage) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1_000.0
}

fn duration_summary(samples: &mut [Duration]) -> (f64, f64, f64) {
    samples.sort_unstable();
    let to_ms = |duration: Duration| duration.as_secs_f64() * 1_000.0;
    (
        to_ms(samples[samples.len() / 2]),
        to_ms(samples[0]),
        to_ms(samples[samples.len() - 1]),
    )
}

fn warm_stage_delta_ms(
    before: &IvpTelemetrySnapshot,
    after: &IvpTelemetrySnapshot,
    stage: IvpWarmStage,
    calls: usize,
) -> f64 {
    (after.warm_stage(stage).elapsed - before.warm_stage(stage).elapsed).as_secs_f64() * 1_000.0
        / calls as f64
}

struct LambdifyCallbackSample {
    residual: DVector<f64>,
    jacobian: nalgebra::DMatrix<f64>,
    telemetry: IvpTelemetrySnapshot,
}

fn lambdify_callback_telemetry(
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    repetitions: usize,
) -> LambdifyCallbackSample {
    let workload = build_workload(WorkloadKind::DiffusionChain, dimension);
    let telemetry = IvpTelemetry::detailed();
    let mut options = SymbolicIvpProblemOptions::new()
        .with_symbolic_assembly_backend(assembly)
        .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
        .with_telemetry(telemetry.clone());
    if !workload.parameter_names.is_empty() {
        options = options
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }
    let prepared = prepare_symbolic_ivp_problem(
        workload.equations,
        workload.variables,
        workload.time_variable,
        options,
    )
    .expect("Lambdify callback telemetry preparation");
    let state = workload.initial_state;
    let mut residual = DVector::zeros(dimension);
    let mut args = Vec::with_capacity(dimension + 5);
    let before_residual = telemetry.snapshot();
    let residual_started = Instant::now();
    for _ in 0..repetitions {
        prepared
            .try_evaluate_residual_into_with_workspace(0.01, &state, &mut residual, &mut args)
            .expect("Lambdify residual telemetry callback");
    }
    let residual_wall_ms = residual_started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64;
    let after_residual = telemetry.snapshot();
    eprintln!(
        "diffusion-chain | {dimension} | {} | residual | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {} | {} | {} | {} | {}",
        if assembly == IvpSymbolicAssemblyBackend::AtomView {
            "atom-native"
        } else {
            "expr-legacy"
        },
        residual_wall_ms,
        warm_stage_delta_ms(
            &before_residual,
            &after_residual,
            IvpWarmStage::ResidualCallback,
            repetitions
        ),
        warm_stage_delta_ms(
            &before_residual,
            &after_residual,
            IvpWarmStage::ArgumentBinding,
            repetitions
        ),
        warm_stage_delta_ms(
            &before_residual,
            &after_residual,
            IvpWarmStage::ResidualEvaluation,
            repetitions
        ),
        warm_stage_delta_ms(
            &before_residual,
            &after_residual,
            IvpWarmStage::ResidualOutputAssembly,
            repetitions
        ),
        (after_residual.copies - before_residual.copies) / repetitions as u64,
        (after_residual.copied_bytes - before_residual.copied_bytes) / repetitions as u64,
        (after_residual.allocated_bytes - before_residual.allocated_bytes) / repetitions as u64,
        (after_residual.sequential_dispatches - before_residual.sequential_dispatches)
            / repetitions as u64,
        (after_residual.parallel_dispatches - before_residual.parallel_dispatches)
            / repetitions as u64,
    );

    let before_jacobian = telemetry.snapshot();
    let jacobian_started = Instant::now();
    let mut jacobian = nalgebra::DMatrix::zeros(dimension, dimension);
    for _ in 0..repetitions {
        jacobian = prepared
            .try_evaluate_jacobian(0.01, &state)
            .expect("Lambdify Jacobian telemetry callback");
    }
    let jacobian_wall_ms = jacobian_started.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64;
    let after_jacobian = telemetry.snapshot();
    eprintln!(
        "diffusion-chain | {dimension} | {} | jacobian | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {} | {} | {} | {} | {}",
        if assembly == IvpSymbolicAssemblyBackend::AtomView {
            "atom-native"
        } else {
            "expr-legacy"
        },
        jacobian_wall_ms,
        warm_stage_delta_ms(
            &before_jacobian,
            &after_jacobian,
            IvpWarmStage::JacobianCallback,
            repetitions
        ),
        warm_stage_delta_ms(
            &before_jacobian,
            &after_jacobian,
            IvpWarmStage::ArgumentBinding,
            repetitions
        ),
        warm_stage_delta_ms(
            &before_jacobian,
            &after_jacobian,
            IvpWarmStage::JacobianEvaluation,
            repetitions
        ),
        warm_stage_delta_ms(
            &before_jacobian,
            &after_jacobian,
            IvpWarmStage::JacobianOutputAssembly,
            repetitions
        ),
        (after_jacobian.copies - before_jacobian.copies) / repetitions as u64,
        (after_jacobian.copied_bytes - before_jacobian.copied_bytes) / repetitions as u64,
        (after_jacobian.allocated_bytes - before_jacobian.allocated_bytes) / repetitions as u64,
        (after_jacobian.sequential_dispatches - before_jacobian.sequential_dispatches)
            / repetitions as u64,
        (after_jacobian.parallel_dispatches - before_jacobian.parallel_dispatches)
            / repetitions as u64,
    );
    LambdifyCallbackSample {
        residual,
        jacobian,
        telemetry: telemetry.snapshot(),
    }
}

#[test]
#[ignore = "release dense-size symbolic preparation and solver parity matrix"]
fn bdf_dense_size_preparation_and_solver_matrix_story() {
    let dimensions = [32, 64, 100];
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];

    eprintln!(
        "[BDF dense-size matrix] dimensions={dimensions:?}; fully coupled dense Jacobian, BDF dense storage/linear algebra"
    );
    eprintln!(
        "[BDF frontend preparation only] no solver construction or integration; stage scopes can overlap and are not additive"
    );
    eprintln!(
        "dimension | route | prepare_ms | expr_to_atom_ms | residual_evaluator_ms | atom_jacobian_ms | atom_dependency_ms | differentiation_ms | native_jac_evaluator_ms | symbolic_jacobian_ms | simplify_ms"
    );

    for dimension in dimensions {
        for (assembly, route) in routes {
            let (prepare_ms, snapshot) = prepare_symbolic_frontend(dimension, assembly);
            assert_eq!(snapshot.state_dimension, dimension);
            assert_eq!(snapshot.residual_dimension, dimension);
            match assembly {
                IvpSymbolicAssemblyBackend::ExprLegacy => assert!(
                    snapshot
                        .cold_stage(IvpColdStage::SymbolicDifferentiation)
                        .calls
                        > 0
                ),
                IvpSymbolicAssemblyBackend::AtomView => {
                    assert_eq!(
                        snapshot.cold_stage(IvpColdStage::ExprToAtom).calls,
                        1,
                        "AtomView should convert the equation set exactly once and share it with residual/Jacobian preparation"
                    );
                    assert_eq!(
                        snapshot
                            .cold_stage(IvpColdStage::AtomJacobianPreparation)
                            .calls,
                        1
                    );
                    assert!(
                        snapshot
                            .cold_stage(IvpColdStage::NativeJacobianEvaluatorPreparation)
                            .calls
                            > 0
                    );
                    assert!(
                        snapshot
                            .cold_stage(IvpColdStage::ResidualLambdification)
                            .calls
                            > 0
                    );
                }
                IvpSymbolicAssemblyBackend::AtomViewExprCompat => unreachable!(),
            }
            eprintln!(
                "{dimension} | {route} | {prepare_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}",
                stage_ms(&snapshot, IvpColdStage::ExprToAtom),
                stage_ms(&snapshot, IvpColdStage::ResidualLambdification),
                stage_ms(&snapshot, IvpColdStage::AtomJacobianPreparation),
                stage_ms(&snapshot, IvpColdStage::AtomDependencyAnalysis),
                stage_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
                stage_ms(&snapshot, IvpColdStage::NativeJacobianEvaluatorPreparation),
                stage_ms(&snapshot, IvpColdStage::SymbolicJacobian),
                stage_ms(&snapshot, IvpColdStage::Simplification),
            );
        }
    }

    eprintln!(
        "[BDF dense-size solver matrix] route | workload | n | prepare_ms | solve_ms | residual_calls | jacobian_calls | factorizations | accepted | rejected | max_state_abs; detailed BDF scopes follow per row and are nested/non-additive"
    );

    for (kind, dimension) in [(WorkloadKind::CombustionLike, 0)] {
        let mut results = Vec::new();
        for (assembly, route) in routes {
            let mut solver = measured_solver(kind, dimension, assembly);
            solver
                .try_solve()
                .unwrap_or_else(|error| panic!("{} {route}: {error}", kind.label()));
            assert_eq!(solver.get_status(), "finished");

            let stats = solver.get_statistics();
            assert!(stats.backend_prepare_ms_total > 0.0);
            assert!(stats.solve_ms_total > 0.0);
            assert!(stats.bdf_nfev_total > 0);
            assert!(stats.bdf_njev_total > 0);
            assert!(stats.bdf_nlu_total > 0);
            let state = final_state(&solver);
            let max_state_abs = state.iter().map(|value| value.abs()).fold(0.0, f64::max);
            let actual_dimension = state.len();
            eprintln!(
                "{route} | {} | {} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {:.6e}",
                kind.label(),
                actual_dimension,
                stats.backend_prepare_ms_total,
                stats.solve_ms_total,
                stats.residual_calls,
                stats.jacobian_calls,
                stats.bdf_nlu_total,
                stats.accepted_steps_total,
                stats.rejected_step_attempts_total,
                max_state_abs,
            );
            eprintln!(
                "[BDF dense step scopes] workload={} n={} route={} {}",
                kind.label(),
                actual_dimension,
                route,
                stats.table_report(),
            );
            results.push(state);
        }

        let max_diff = results[0]
            .iter()
            .zip(results[1].iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff <= 2e-5,
            "{} dimension {dimension} backend parity drift: {max_diff:e}",
            kind.label()
        );
        eprintln!(
            "[BDF dense-size parity] workload={} dimension={} max_final_diff={max_diff:.3e} status=ok",
            kind.label(),
            results[0].len(),
        );
    }

    for dimension in dimensions {
        let mut results = Vec::new();
        for (assembly, route) in routes {
            let mut solver = dense_coupled_solver(dimension, assembly, BdfTelemetryMode::Timings);
            solver
                .try_solve()
                .unwrap_or_else(|error| panic!("dense-coupled n={dimension} {route}: {error}"));
            assert_eq!(solver.get_status(), "finished");

            let stats = solver.get_statistics();
            assert!(stats.backend_prepare_ms_total > 0.0);
            assert!(stats.solve_ms_total > 0.0);
            assert!(stats.bdf_nfev_total > 0);
            assert!(stats.bdf_njev_total > 0);
            assert!(stats.bdf_nlu_total > 0);
            let state = final_state(&solver);
            let max_state_abs = state.iter().map(|value| value.abs()).fold(0.0, f64::max);
            eprintln!(
                "{route} | dense-coupled | {dimension} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {:.6e}",
                stats.backend_prepare_ms_total,
                stats.solve_ms_total,
                stats.residual_calls,
                stats.jacobian_calls,
                stats.bdf_nlu_total,
                stats.accepted_steps_total,
                stats.rejected_step_attempts_total,
                max_state_abs,
            );
            let preparation = solver.preparation_timings();
            assert!(preparation.total_ms > 0.0);
            assert!(preparation.symbolic_backend_ms > 0.0);
            assert!(preparation.runtime_initialization_ms > 0.0);
            eprintln!(
                "[BDF preparation stages] workload=dense-coupled n={dimension} route={route} {}",
                preparation.table_report(),
            );
            let symbolic = solver
                .preparation_backend_telemetry()
                .expect("timings mode should retain symbolic preparation stages");
            assert_eq!(
                symbolic.mode,
                crate::symbolic::ivp_telemetry::IvpTelemetryMode::Detailed
            );
            eprintln!(
                "[BDF symbolic cold stages] workload=dense-coupled n={dimension} route={route} execution={} validation_ms={:.3} parameter_binding_ms={:.3} expr_to_atom_ms={:.3} symbolic_jacobian_ms={:.3} differentiation_ms={:.3} simplification_ms={:.3} atom_residual_ms={:.3} atom_jacobian_ms={:.3} atom_dependency_ms={:.3} native_jacobian_evaluator_ms={:.3} residual_lambdification_ms={:.3} jacobian_lambdification_ms={:.3} aot_problem_key_ms={:.3} aot_cache_lookup_ms={:.3} (nested scopes are diagnostic, not additive)",
                symbolic.execution.label(),
                stage_ms(symbolic, IvpColdStage::Validation),
                stage_ms(symbolic, IvpColdStage::ParameterBinding),
                stage_ms(symbolic, IvpColdStage::ExprToAtom),
                stage_ms(symbolic, IvpColdStage::SymbolicJacobian),
                stage_ms(symbolic, IvpColdStage::SymbolicDifferentiation),
                stage_ms(symbolic, IvpColdStage::Simplification),
                stage_ms(symbolic, IvpColdStage::AtomResidualPreparation),
                stage_ms(symbolic, IvpColdStage::AtomJacobianPreparation),
                stage_ms(symbolic, IvpColdStage::AtomDependencyAnalysis),
                stage_ms(symbolic, IvpColdStage::NativeJacobianEvaluatorPreparation),
                stage_ms(symbolic, IvpColdStage::ResidualLambdification),
                stage_ms(symbolic, IvpColdStage::JacobianLambdification),
                stage_ms(symbolic, IvpColdStage::AotProblemKeyConstruction),
                stage_ms(symbolic, IvpColdStage::AotCacheLookup),
            );
            eprintln!(
                "[BDF dense step scopes] workload=dense-coupled n={dimension} route={route} {}",
                stats.table_report(),
            );
            results.push(state);
        }

        let max_diff = results[0]
            .iter()
            .zip(results[1].iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff <= 2e-5,
            "dense-coupled dimension {dimension} backend parity drift: {max_diff:e}"
        );
        eprintln!(
            "[BDF dense-size parity] workload=dense-coupled dimension={dimension} max_final_diff={max_diff:.3e} status=ok"
        );
    }
}

#[test]
#[ignore = "release stage breakdown and parity matrix"]
fn bdf_large_workload_stage_breakdown_story() {
    let workloads = [
        (WorkloadKind::CombustionLike, 0),
        (WorkloadKind::DiffusionChain, 8),
        (WorkloadKind::DiffusionChain, 16),
    ];
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];

    eprintln!(
        "[BDF release stage breakdown] timings enabled for diagnostic pass only; nested scopes are not additive"
    );
    eprintln!(
        "workload | dimension | route | prepare_ms | solve_ms | integration_loop_ms | bdf_step_ms | output_collection_ms | result_assembly_ms | factor_ms | linear_solve_ms | snapshot_ms | predictor_setup_ms | newton_rhs_assembly_ms | newton_correction_norm_ms | newton_state_update_ms | error_estimate_ms | nordsieck_update_ms | residual_ms/call | jacobian_ms/call | steps | rejected | newton_solves | newton_iters | rhs/jac/factor/linear_calls"
    );

    for (kind, dimension) in workloads {
        let mut route_results = Vec::new();
        let dimension_label = if matches!(kind, WorkloadKind::DiffusionChain) {
            dimension.to_string()
        } else {
            "fixed".to_string()
        };
        for (assembly, route) in routes {
            let mut solver = measured_solver(kind, dimension, assembly);
            solver
                .try_solve()
                .unwrap_or_else(|error| panic!("{} {route}: {error}", kind.label()));
            assert_eq!(solver.get_status(), "finished");

            let stats = solver.get_statistics();
            assert!(stats.solve_ms_total > 0.0);
            assert!(stats.integration_loop_ms_total > 0.0);
            assert!(stats.bdf_step_ms_total > 0.0);
            assert!(stats.residual_calls > 0);
            assert!(stats.jacobian_calls > 0);
            assert!(stats.bdf_nfev_total > 0);
            assert!(stats.bdf_njev_total > 0);
            assert!(stats.bdf_nlu_total > 0);
            assert!(
                stats.integration_loop_ms_total + stats.result_assembly_ms_total
                    <= stats.solve_ms_total + 1e-6,
                "integration and result assembly are sequential solve children"
            );
            assert!(
                stats.bdf_step_ms_total + stats.output_collection_ms_total
                    <= stats.integration_loop_ms_total + 1e-6,
                "step and output collection are nested integration-loop scopes"
            );

            let state = final_state(&solver);
            eprintln!(
                "{} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.6} | {:.6} | {} | {} | {} | {} | {}/{}/{}/{}",
                kind.label(),
                dimension_label,
                route,
                stats.backend_prepare_ms_total,
                stats.solve_ms_total,
                stats.integration_loop_ms_total,
                stats.bdf_step_ms_total,
                stats.output_collection_ms_total,
                stats.result_assembly_ms_total,
                stats.linear_factorization_ms_total,
                stats.linear_solve_ms_total,
                stats.bdf_step_snapshot_ms_total,
                stats.bdf_step_predictor_setup_ms_total,
                stats.bdf_newton_rhs_assembly_ms_total,
                stats.bdf_newton_correction_norm_ms_total,
                stats.bdf_newton_state_update_ms_total,
                stats.bdf_step_error_estimate_ms_total,
                stats.bdf_step_nordsieck_update_ms_total,
                stats.avg_residual_ms().unwrap_or(0.0),
                stats.avg_jacobian_ms().unwrap_or(0.0),
                stats.accepted_steps_total,
                stats.rejected_step_attempts_total,
                stats.nonlinear_solve_calls,
                stats.nonlinear_iterations_total,
                stats.bdf_nfev_total,
                stats.bdf_njev_total,
                stats.bdf_nlu_total,
                stats.linear_solve_attempts_total,
            );
            eprintln!(
                "[BDF telemetry detail] {} {route}: {}",
                kind.label(),
                stats.table_report()
            );
            route_results.push(state);
        }

        let max_diff = route_results[0]
            .iter()
            .zip(route_results[1].iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff <= 2e-5,
            "{} AtomView-vs-ExprLegacy parity drift: {max_diff:e}",
            kind.label()
        );
        eprintln!(
            "[BDF stage parity] workload={} dimension={} max_final_diff={max_diff:.3e} status=ok",
            kind.label(),
            dimension_label,
        );
    }
}

#[test]
#[ignore = "release paired alternating ExprLegacy-vs-AtomView dense fresh-E2E noise check"]
fn bdf_dense_fresh_e2e_routes_alternate_to_control_host_order_bias() {
    let dimensions = [32, 64, 100];
    let repetitions = 9;
    let routes = [
        (0, IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (1, IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];
    let mut samples = [Vec::new(), Vec::new()];

    eprintln!(
        "[BDF paired fresh-E2E] dimensions={dimensions:?}; repetitions={repetitions}; telemetry=Off; fixture/solver construction excluded; route order alternates each pair"
    );
    eprintln!("dimension | route | median_ms | min_ms | max_ms | same-pair-max-diff");

    for dimension in dimensions {
        samples[0].clear();
        samples[1].clear();
        for repetition in 0..repetitions {
            let ordered_routes = if repetition % 2 == 0 {
                routes
            } else {
                [routes[1], routes[0]]
            };
            let mut states = [None, None];
            for (route_index, assembly, route) in ordered_routes {
                let mut solver = dense_coupled_solver(dimension, assembly, BdfTelemetryMode::Off);
                let started = Instant::now();
                solver
                    .try_solve()
                    .unwrap_or_else(|error| panic!("dense n={dimension} {route}: {error}"));
                let state = final_state(&solver);
                let elapsed = started.elapsed();
                assert_eq!(solver.get_status(), "finished");
                black_box(&state);
                samples[route_index].push(elapsed);
                states[route_index] = Some(state);
            }
            let expr_state = states[0].as_ref().expect("ExprLegacy result");
            let atom_state = states[1].as_ref().expect("AtomView result");
            let max_diff = expr_state
                .iter()
                .zip(atom_state.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            assert!(max_diff <= 2e-5, "n={dimension} parity drift={max_diff:e}");
        }

        let expr_summary = duration_summary(&mut samples[0]);
        let atom_summary = duration_summary(&mut samples[1]);
        let ratio = atom_summary.0 / expr_summary.0;
        eprintln!(
            "{} | expr-legacy | {:.3} | {:.3} | {:.3} | -",
            dimension, expr_summary.0, expr_summary.1, expr_summary.2
        );
        eprintln!(
            "{} | atom-native | {:.3} | {:.3} | {:.3} | ratio={ratio:.3}x",
            dimension, atom_summary.0, atom_summary.1, atom_summary.2
        );
    }
}

#[test]
#[ignore = "release Lambdify callback stage and copy/allocation telemetry matrix"]
fn bdf_lambdify_callback_stage_telemetry_story() {
    let dimensions = [128, 512, 1024];
    let repetitions = 20;
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];
    eprintln!(
        "[BDF Lambdify callback telemetry] workload=diffusion-chain dimensions={dimensions:?}; repetitions={repetitions}; policy=Sequential; telemetry=Detailed; timing includes telemetry overhead; allocation bytes are telemetry estimates, not allocator measurements"
    );
    eprintln!(
        "workload | n | route | phase | wall_ms/call | callback_ms/call | binding_ms/call | eval_ms/call | output_ms/call | copies/call | copied_bytes/call | estimated_alloc_bytes/call | sequential_dispatches/call | parallel_dispatches/call"
    );

    for dimension in dimensions {
        let mut outputs = Vec::with_capacity(routes.len());
        for (assembly, _) in routes {
            outputs.push(lambdify_callback_telemetry(
                dimension,
                assembly,
                repetitions,
            ));
        }
        for sample in &outputs {
            assert_eq!(
                sample
                    .telemetry
                    .warm_stage(IvpWarmStage::ResidualCallback)
                    .calls,
                repetitions as u64,
                "residual callback telemetry count must match the repeated workload"
            );
            assert_eq!(
                sample
                    .telemetry
                    .warm_stage(IvpWarmStage::JacobianCallback)
                    .calls,
                repetitions as u64,
                "Jacobian callback telemetry count must match the repeated workload"
            );
            assert_eq!(
                sample.telemetry.residual_evaluations, repetitions as u64,
                "residual evaluator count must match the repeated workload"
            );
            assert_eq!(
                sample.telemetry.jacobian_evaluations, repetitions as u64,
                "Jacobian evaluator count must match the repeated workload"
            );
            assert_eq!(
                sample.telemetry.parallel_dispatches, 0,
                "Sequential Lambdify telemetry must not report parallel dispatches"
            );
            assert_eq!(sample.telemetry.errors, 0);
        }

        let expr_residual = &outputs[0].residual;
        let expr_jacobian = &outputs[0].jacobian;
        let atom_residual = &outputs[1].residual;
        let atom_jacobian = &outputs[1].jacobian;
        let residual_diff = expr_residual
            .iter()
            .zip(atom_residual.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        let jacobian_diff = expr_jacobian
            .iter()
            .zip(atom_jacobian.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            residual_diff <= 1e-12,
            "n={dimension} residual parity={residual_diff:e}"
        );
        assert!(
            jacobian_diff <= 1e-12,
            "n={dimension} Jacobian parity={jacobian_diff:e}"
        );
        eprintln!(
            "[BDF Lambdify callback parity] n={dimension} residual_diff={residual_diff:.3e} jacobian_diff={jacobian_diff:.3e} status=ok"
        );
    }
}

#[test]
#[ignore = "release generated-AOT key and native-plan preparation attribution matrix"]
fn bdf_generated_aot_key_and_native_plan_stage_story() {
    let dimensions = [32, 64, 100];
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];
    eprintln!(
        "[BDF generated AOT preparation attribution] dimensions={dimensions:?}; compiler=tcc; RebuildAlways; one fresh artifact directory per row; key-identity and native-plan stages may be nested/non-additive"
    );
    eprintln!(
        "n | route | total_prepare_ms | aot_key_ms | aot_key_calls | native_jacobian_ms | native_jacobian_calls | aot_atom_plan_ms | aot_atom_plan_calls | dependency_ms | lowering_ms | materialize_ms | build_ms | link_ms | build_attempts | link_attempts | runtime_ready"
    );

    for dimension in dimensions {
        for (assembly, route) in routes {
            let workload = dense_coupled(dimension);
            let telemetry = IvpTelemetry::detailed();
            let options = SymbolicIvpProblemOptions::new()
                .with_symbolic_assembly_backend(assembly)
                .with_equation_parameters(workload.parameter_names)
                .with_equation_parameter_values(workload.parameter_values)
                .with_telemetry(telemetry.clone());
            let artifact_dir = tempfile::tempdir().expect("isolated dense AOT output");
            let config = SymbolicIvpGeneratedBackendConfig::new()
                .with_output_parent_dir(Some(artifact_dir.path().to_path_buf()))
                .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Release,
                })
                .with_c_tcc();
            let started = Instant::now();
            let prepared = prepare_generated_symbolic_ivp_problem(
                workload.equations,
                workload.variables,
                workload.time_variable,
                options,
                config,
            )
            .unwrap_or_else(|error| panic!("dense n={dimension} {route} AOT prep: {error}"));
            let total_prepare_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let snapshot = prepared.problem.telemetry.snapshot();
            let key = snapshot.cold_stage(IvpColdStage::AotProblemKeyConstruction);
            let native_jacobian = snapshot.cold_stage(IvpColdStage::AtomJacobianPreparation);
            let aot_atom_plan = snapshot.cold_stage(IvpColdStage::AotAtomPlanPreparation);
            assert_eq!(key.calls, 1, "canonical AOT identity should be built once");
            let expected_plan_calls = if assembly == IvpSymbolicAssemblyBackend::AtomView {
                1
            } else {
                0
            };
            assert_eq!(
                aot_atom_plan.calls, expected_plan_calls,
                "AtomView dense AOT plan should be prepared once and ExprLegacy should not prepare one"
            );
            assert_eq!(snapshot.aot_build_attempts, 1);
            assert_eq!(snapshot.aot_link_attempts, 1);
            assert_eq!(snapshot.aot_runtime_ready, 1);
            eprintln!(
                "{dimension} | {route} | {total_prepare_ms:.3} | {:.3} | {} | {:.3} | {} | {:.3} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {}",
                key.elapsed.as_secs_f64() * 1_000.0,
                key.calls,
                native_jacobian.elapsed.as_secs_f64() * 1_000.0,
                native_jacobian.calls,
                aot_atom_plan.elapsed.as_secs_f64() * 1_000.0,
                aot_atom_plan.calls,
                stage_ms(&snapshot, IvpColdStage::AtomDependencyAnalysis),
                stage_ms(&snapshot, IvpColdStage::AotLowering),
                stage_ms(&snapshot, IvpColdStage::AotMaterialization),
                stage_ms(&snapshot, IvpColdStage::AotBuild),
                stage_ms(&snapshot, IvpColdStage::AotLink),
                snapshot.aot_build_attempts,
                snapshot.aot_link_attempts,
                snapshot.aot_runtime_ready,
            );
            assert_eq!(prepared.selected_backend, crate::symbolic::symbolic_ivp_generated::SelectedSymbolicIvpBackendKind::AotCompiled);
        }
    }
}

#[test]
#[ignore = "attribution for the large AOT warm-solve anomaly"]
fn bdf_aot_prebuilt_warm_solver_attribution_story() {
    let kind = WorkloadKind::DiffusionChain;
    let dimension = 1024;
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];
    let mut reference = measured_solver(kind, dimension, IvpSymbolicAssemblyBackend::ExprLegacy);
    reference
        .try_solve()
        .expect("Lambdify reference diffusion solve");
    let reference_state = final_state(&reference);

    let profile = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };
    eprintln!(
        "[BDF AOT prebuilt warm attribution] workload={} n={dimension}; producer=RebuildAlways({profile}), consumer=RequirePrebuilt; telemetry=Timings",
        kind.label()
    );
    eprintln!(
        "route | prepare_ms | solve_ms | nfev | njev | nlu | accepted | rejected | nonlinear_solves | nonlinear_iters | factor_ms | matrix_ms | linear_solve_ms | step_ms | output_ms | max_diff"
    );

    for (assembly, route) in routes {
        let (mut solver, _artifact_dir) = aot_prebuilt_solver(kind, dimension, assembly);
        solver
            .try_generate()
            .unwrap_or_else(|error| panic!("{route} consumer preparation: {error}"));
        let preparation = solver.preparation_timings();
        solver
            .try_solve()
            .unwrap_or_else(|error| panic!("{route} consumer solve: {error}"));
        assert_eq!(solver.get_status(), "finished");
        let stats = solver.get_statistics();
        let state = final_state(&solver);
        let max_diff = state
            .iter()
            .zip(reference_state.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff <= 2e-5, "{route} AOT parity drift={max_diff:e}");
        assert!(stats.bdf_nfev_total > 0);
        assert!(stats.bdf_njev_total > 0);
        assert!(stats.bdf_nlu_total > 0);
        eprintln!(
            "{route} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3e}",
            preparation.total_ms,
            stats.solve_ms_total,
            stats.bdf_nfev_total,
            stats.bdf_njev_total,
            stats.bdf_nlu_total,
            stats.accepted_steps_total,
            stats.rejected_step_attempts_total,
            stats.nonlinear_solve_calls,
            stats.nonlinear_iterations_total,
            stats.linear_factorization_ms_total,
            stats.linear_matrix_assembly_ms_total,
            stats.linear_solve_ms_total,
            stats.bdf_step_ms_total,
            stats.output_collection_ms_total,
            max_diff,
        );
        eprintln!(
            "[BDF AOT prebuilt warm scopes] route={route} {}",
            stats.table_report()
        );
    }
}
