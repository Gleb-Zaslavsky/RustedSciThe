//! Experimental LSODE2 callback adapter backed by `PreparedAtomSystem`.
//!
//! This module is isolated from the default symbolic callback route and is not
//! recommended for production: the measured LSODE2 release workload had a
//! slower full solve than AtomViewNative. The solver currently opts into it
//! only for sequential AtomView/Banded runs.

use super::config::Lsode2ProblemConfig;
use super::native_jacobian::NativeJacobianStorage;
use crate::numerical::BDF::BDF_solver::BdfJacobian;
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::View::{PreparedAtomSystem, PreparedAtomSystemScratch};
use crate::symbolic::ivp_telemetry::{IvpColdStage, IvpWarmStage};
use crate::symbolic::symbolic_ivp::{IvpBackendError, SharedIvpParameterValues};
use nalgebra::DVector;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

type ResidualCallback = dyn Fn(f64, &DVector<f64>) -> DVector<f64>;
type JacobianCallback = dyn FnMut(f64, &DVector<f64>) -> BdfJacobian;

struct EvaluationWorkspace {
    scratch: PreparedAtomSystemScratch,
    args: Vec<f64>,
    residual: Vec<f64>,
    jacobian: Vec<f64>,
}

pub(super) struct PreparedCallbacks {
    pub(super) residual: Rc<ResidualCallback>,
    pub(super) jacobian: Rc<RefCell<Box<JacobianCallback>>>,
    pub(super) parameter_values_handle: Option<SharedIvpParameterValues>,
}

pub(super) fn prepare(
    config: &Lsode2ProblemConfig,
    storage: NativeJacobianStorage,
) -> Result<PreparedCallbacks, IvpBackendError> {
    let (kl, ku) = match storage {
        NativeJacobianStorage::Banded {
            bandwidth: Some((kl, ku)),
        } => (kl, ku),
        NativeJacobianStorage::Banded { bandwidth: None } => {
            return Err(backend_error(
                "PreparedAtom requires explicit Banded lower/upper widths",
            ));
        }
        _ => return Err(backend_error("PreparedAtom requires Banded storage")),
    };

    let mut input_names = Vec::with_capacity(
        1 + config.equation_parameters.as_ref().map_or(0, Vec::len) + config.values.len(),
    );
    input_names.push(config.arg.clone());
    if let Some(parameters) = config.equation_parameters.as_ref() {
        input_names.extend(parameters.iter().cloned());
    }
    input_names.extend(config.values.iter().cloned());

    let preparation_started = config
        .telemetry
        .start_cold_stage(IvpColdStage::AtomPreparation);
    let prepared_result = PreparedAtomSystem::try_from_exprs(&config.eq_system, &input_names, None);
    config
        .telemetry
        .record_cold_stage(IvpColdStage::AtomPreparation, preparation_started);
    let prepared = prepared_result
        .map_err(|error| backend_error(&format!("PreparedAtom preparation failed: {error}")))?;
    config.telemetry.record_symbolic_jacobian_build();
    let state_column_start = 1 + config.equation_parameters.as_ref().map_or(0, Vec::len);
    let mut state_jacobian = Vec::new();
    for (value_index, entry) in prepared.jacobian_entries().enumerate() {
        if entry.col < state_column_start {
            continue;
        }
        let col = entry.col - state_column_start;
        if col >= config.values.len() {
            return Err(backend_error(
                "PreparedAtom returned an invalid Jacobian column",
            ));
        }
        if entry.row.saturating_sub(col) > kl || col.saturating_sub(entry.row) > ku {
            return Err(backend_error(&format!(
                "PreparedAtom found a state Jacobian entry ({}, {col}) outside configured Banded widths kl={kl}, ku={ku}",
                entry.row
            )));
        }
        state_jacobian.push((value_index, entry.row, col));
    }

    let parameter_values_handle: Option<SharedIvpParameterValues> = match (
        &config.equation_parameters,
        &config.equation_parameter_values,
    ) {
        (Some(names), Some(values)) if names.len() != values.len() => {
            return Err(IvpBackendError::ParameterCountMismatch {
                expected: names.len(),
                actual: values.len(),
            });
        }
        (Some(names), None) if !names.is_empty() => {
            return Err(IvpBackendError::MissingParameterValues {
                expected: names.len(),
            });
        }
        (_, Some(values)) => Some(Arc::new(RwLock::new(values.clone()))),
        (None, None) | (Some(_), None) => None,
    };
    let shared_prepared = Arc::new(prepared);
    let workspace = Rc::new(RefCell::new(EvaluationWorkspace {
        scratch: PreparedAtomSystemScratch::default(),
        args: Vec::with_capacity(input_names.len()),
        residual: vec![0.0; config.eq_system.len()],
        jacobian: vec![0.0; shared_prepared.jacobian_len()],
    }));

    let residual = {
        let system = Arc::clone(&shared_prepared);
        let workspace = Rc::clone(&workspace);
        let parameter_handle = parameter_values_handle.clone();
        let telemetry = config.telemetry.clone();
        Rc::new(move |time: f64, state: &DVector<f64>| {
            let started = telemetry.start_warm_stage(IvpWarmStage::ResidualCallback);
            telemetry.record_residual_request();
            let mut workspace = workspace.borrow_mut();
            let EvaluationWorkspace {
                scratch,
                args,
                residual,
                ..
            } = &mut *workspace;
            let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
            let binding = bind_args(time, state, parameter_handle.as_ref(), args);
            telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
            let evaluation_started = telemetry.start_warm_stage(IvpWarmStage::ResidualEvaluation);
            let result = binding.and_then(|()| {
                system
                    .residual_into(args, residual, scratch)
                    .map_err(|error| error.to_string())
            });
            telemetry.record_scalar_evaluations(system.residual_len());
            if result.is_err() {
                telemetry.record_error();
                telemetry.record_warm_stage(IvpWarmStage::ResidualEvaluation, evaluation_started);
                telemetry.record_warm_stage(IvpWarmStage::ResidualCallback, started);
                return DVector::from_element(state.len(), f64::NAN);
            }
            telemetry.record_residual_evaluation(evaluation_started);
            let output_started = telemetry.start_warm_stage(IvpWarmStage::ResidualOutputAssembly);
            let output = DVector::from_vec(residual.clone());
            telemetry.record_warm_stage(IvpWarmStage::ResidualOutputAssembly, output_started);
            telemetry.record_warm_stage(IvpWarmStage::ResidualCallback, started);
            output
        }) as Rc<ResidualCallback>
    };

    let jacobian = {
        let system = Arc::clone(&shared_prepared);
        let workspace = Rc::clone(&workspace);
        let parameter_handle = parameter_values_handle.clone();
        let telemetry = config.telemetry.clone();
        Rc::new(RefCell::new(
            Box::new(move |time: f64, state: &DVector<f64>| {
                let started = telemetry.start_warm_stage(IvpWarmStage::JacobianCallback);
                telemetry.record_jacobian_request();
                let mut workspace = workspace.borrow_mut();
                let EvaluationWorkspace {
                    scratch,
                    args,
                    jacobian,
                    ..
                } = &mut *workspace;
                let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
                let binding = bind_args(time, state, parameter_handle.as_ref(), args);
                telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                let evaluation_started =
                    telemetry.start_warm_stage(IvpWarmStage::JacobianEvaluation);
                let result = binding.and_then(|()| {
                    system
                        .jacobian_values_into(args, jacobian, scratch)
                        .map_err(|error| error.to_string())
                });
                telemetry.record_scalar_evaluations(system.jacobian_len());
                let output_started =
                    telemetry.start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
                let mut banded = Banded::<f64>::zeros(state.len(), kl, ku)
                    .expect("validated PreparedAtom Banded shape");
                if result.is_ok() {
                    for (value_index, row, col) in &state_jacobian {
                        let _ = banded.set(*row, *col, jacobian[*value_index]);
                    }
                } else {
                    for row in 0..state.len() {
                        let _ = banded.set(row, row, f64::NAN);
                    }
                    telemetry.record_error();
                }
                if result.is_ok() {
                    telemetry.record_jacobian_evaluation(evaluation_started);
                } else {
                    telemetry
                        .record_warm_stage(IvpWarmStage::JacobianEvaluation, evaluation_started);
                }
                telemetry.record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
                telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, started);
                BdfJacobian::Banded(banded)
            }) as Box<JacobianCallback>,
        ))
    };

    Ok(PreparedCallbacks {
        residual,
        jacobian,
        parameter_values_handle,
    })
}

fn bind_args(
    time: f64,
    state: &DVector<f64>,
    parameter_handle: Option<&SharedIvpParameterValues>,
    args: &mut Vec<f64>,
) -> Result<(), String> {
    args.clear();
    args.push(time);
    if let Some(handle) = parameter_handle {
        let values = handle
            .read()
            .map_err(|_| "PreparedAtom parameter lock is poisoned".to_string())?;
        args.extend(values.iter().copied());
    }
    args.extend(state.iter().copied());
    Ok(())
}

fn backend_error(message: &str) -> IvpBackendError {
    IvpBackendError::GeneratedBackendFailure {
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::LSODE2::Lsode2Solver;
    use crate::numerical::LSODE2::algorithm::Lsode2ControllerConfig;
    use crate::numerical::LSODE2::config::{
        Lsode2LinearSystemStructure, Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend,
        Lsode2SymbolicExecutionMode,
    };
    use crate::numerical::LSODE2::workload_fixtures::diffusion_chain;
    use crate::symbolic::ivp_telemetry::{
        IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetryExecution,
    };
    use crate::symbolic::symbolic_ivp::{
        IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions,
        prepare_symbolic_ivp_residual_problem,
    };
    use std::hint::black_box;
    use std::time::Instant;

    macro_rules! reportln {
        ($($arg:tt)*) => {
            crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
        };
    }

    #[derive(Clone, Copy)]
    enum Route {
        Expr,
        Atom,
        PreparedAtom,
    }

    impl Route {
        fn label(self) -> &'static str {
            match self {
                Self::Expr => "ExprLegacy",
                Self::Atom => "AtomNative",
                Self::PreparedAtom => "PreparedAtom",
            }
        }
    }

    fn diffusion_config(dimension: usize, route: Route) -> Lsode2ProblemConfig {
        let workload = diffusion_chain(dimension);
        let mut config = Lsode2ProblemConfig::new(
            workload.equations,
            workload.variables,
            workload.time_variable,
            0.0,
            workload.initial_state,
            2.0,
            0.02,
            1.0e-7,
            1.0e-9,
        )
        .with_equation_parameters(workload.parameter_names)
        .with_equation_parameter_values(workload.parameter_values)
        .with_controller(Lsode2ControllerConfig::bdf_only())
        .with_faithful_bdf_solve(10_000, 10_000);

        let (assembly, execution) = match route {
            Route::Expr => (
                Lsode2SymbolicAssemblyBackend::ExprLegacy,
                Lsode2SymbolicExecutionMode::LambdifyExpr,
            ),
            Route::Atom => (
                Lsode2SymbolicAssemblyBackend::AtomView,
                Lsode2SymbolicExecutionMode::LambdifyExpr,
            ),
            Route::PreparedAtom => (
                Lsode2SymbolicAssemblyBackend::AtomView,
                Lsode2SymbolicExecutionMode::PreparedAtom,
            ),
        };
        config = config
            .with_native_banded_faithful_backend()
            .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
                assembly,
                execution,
            })
            .with_linear_system_structure(Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 })
            .with_telemetry(IvpTelemetry::detailed())
            .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential);
        config
    }

    #[test]
    fn prepared_atom_banded_route_matches_atom_lambdify_full_solve() {
        let mut atom = Lsode2Solver::new(diffusion_config(32, Route::Atom)).unwrap();
        let mut prepared = Lsode2Solver::new(diffusion_config(32, Route::PreparedAtom)).unwrap();
        atom.solve().unwrap();
        prepared.solve().unwrap();

        let (atom_times, atom_states) = atom.get_result();
        let (prepared_times, prepared_states) = prepared.get_result();
        assert_eq!(atom_times.len(), prepared_times.len());
        assert_eq!(atom_states.shape(), prepared_states.shape());
        let max_time_diff = atom_times
            .iter()
            .zip(prepared_times.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        let max_state_diff = atom_states
            .iter()
            .zip(prepared_states.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        assert_eq!(max_time_diff, 0.0, "accepted grids must match exactly");
        assert!(max_state_diff <= 1.0e-11, "state drift={max_state_diff:e}");

        let atom_stats = atom.native_statistics();
        let prepared_stats = prepared.native_statistics();
        let atom_summary = atom.summary();
        let prepared_summary = prepared.summary();
        assert_eq!(
            atom_summary.evaluation_telemetry.accepted_steps,
            prepared_summary.evaluation_telemetry.accepted_steps
        );
        assert_eq!(
            atom_summary.evaluation_telemetry.rejected_steps,
            prepared_summary.evaluation_telemetry.rejected_steps
        );
        assert_eq!(
            atom_stats.native_residual_calls,
            prepared_stats.native_residual_calls
        );
        assert_eq!(
            atom_stats.native_jacobian_calls,
            prepared_stats.native_jacobian_calls
        );
        assert_eq!(
            atom_summary.evaluation_telemetry.linear_solves,
            prepared_summary.evaluation_telemetry.linear_solves
        );
        assert_eq!(
            prepared.telemetry_snapshot().execution,
            IvpTelemetryExecution::PreparedAtom
        );
    }

    #[test]
    fn prepared_atom_banded_route_rejects_unsupported_execution_policy() {
        let config = diffusion_config(32, Route::PreparedAtom)
            .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Auto { min_work: 1 });
        let mut solver = Lsode2Solver::new(config).unwrap();
        let error = solver
            .prepare()
            .expect_err("PreparedAtom must reject Auto rather than ignore the requested policy");
        assert!(error.to_string().contains("sequential callbacks only"));

        let config = diffusion_config(32, Route::PreparedAtom)
            .with_linear_system_structure(Lsode2LinearSystemStructure::Dense)
            .with_linear_solver_policy(super::super::config::Lsode2LinearSolverPolicy::Auto);
        let mut solver = Lsode2Solver::new(config).unwrap();
        let error = solver
            .prepare()
            .expect_err("PreparedAtom currently requires faithful Banded storage");
        assert!(
            error.to_string().contains("faithful Banded backend"),
            "unexpected dense-layout rejection: {error}"
        );
    }

    #[test]
    fn prepared_atom_rejects_legacy_assembly_and_too_narrow_bandwidth() {
        let config = diffusion_config(32, Route::PreparedAtom).with_residual_jacobian_source(
            Lsode2ResidualJacobianSource::Symbolic {
                assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
                execution: Lsode2SymbolicExecutionMode::PreparedAtom,
            },
        );
        let mut solver = Lsode2Solver::new(config).unwrap();
        let error = solver
            .prepare()
            .expect_err("PreparedAtom cannot run with ExprLegacy assembly");
        assert!(error.to_string().contains("requires AtomView"));

        let config = diffusion_config(32, Route::PreparedAtom)
            .with_linear_system_structure(Lsode2LinearSystemStructure::Banded { kl: 0, ku: 1 });
        let mut solver = Lsode2Solver::new(config).unwrap();
        let error = solver
            .prepare()
            .expect_err("PreparedAtom must reject a band that drops known Jacobian entries");
        assert!(
            error
                .to_string()
                .contains("outside configured Banded widths"),
            "unexpected narrow-band rejection: {error}"
        );
    }

    #[test]
    fn prepared_atom_residual_and_banded_jacobian_match_expr_and_atom_callbacks() {
        let dimension = 64;
        let state = DVector::from_iterator(
            dimension,
            (0..dimension).map(|index| 0.25 + 0.001 * (index % 17) as f64),
        );
        let routes = [Route::Expr, Route::Atom, Route::PreparedAtom];
        let callbacks = routes.map(|route| prepare_callbacks(dimension, route));
        let (expected_residual, expected_jacobian) = callback_values(&callbacks[0], &state);

        for (route, callback) in routes.into_iter().zip(callbacks.iter()) {
            let (residual, jacobian) = callback_values(callback, &state);
            assert!(
                max_slice_diff(&residual, &expected_residual) <= 1.0e-12,
                "residual mismatch for {}",
                route.label()
            );
            assert!(
                max_slice_diff(&jacobian, &expected_jacobian) <= 1.0e-12,
                "compact Banded Jacobian mismatch for {}",
                route.label()
            );
        }
    }

    #[derive(Default)]
    struct Samples(Vec<f64>);

    impl Samples {
        fn median_mad(&mut self) -> (f64, f64) {
            self.0.sort_by(f64::total_cmp);
            let median = self.0[self.0.len() / 2];
            let mut deviations = self
                .0
                .iter()
                .map(|sample| (sample - median).abs())
                .collect::<Vec<_>>();
            deviations.sort_by(f64::total_cmp);
            (median, deviations[deviations.len() / 2])
        }
    }

    fn run_full_solve(dimension: usize, route: Route) -> (f64, f64, Vec<f64>, Vec<f64>, Vec<u64>) {
        let prepare_started = Instant::now();
        let mut solver = Lsode2Solver::new(diffusion_config(dimension, route))
            .expect("LSODE2 A/B config should construct");
        solver
            .prepare()
            .expect("LSODE2 A/B preparation should succeed");
        let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
        let solve_started = Instant::now();
        solver.solve().expect("LSODE2 A/B solve should succeed");
        let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
        let (times, states) = solver.get_result();
        let trajectory = states.iter().copied().collect::<Vec<_>>();
        let counters = solver.telemetry_snapshot();
        let work = vec![
            counters.accepted_steps,
            counters.rejected_steps,
            counters.residual_evaluations,
            counters.jacobian_evaluations,
            counters.linear_solve_requests,
        ];
        (
            prepare_ms,
            solve_ms,
            times.as_slice().to_vec(),
            trajectory,
            work,
        )
    }

    fn max_slice_diff(left: &[f64], right: &[f64]) -> f64 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max)
    }

    struct CallbackPair {
        residual: Rc<ResidualCallback>,
        jacobian: Rc<RefCell<Box<JacobianCallback>>>,
        setup_ms: f64,
    }

    fn prepare_callbacks(dimension: usize, route: Route) -> CallbackPair {
        let workload = diffusion_chain(dimension);
        let config = diffusion_config(dimension, route).with_telemetry(IvpTelemetry::disabled());
        let started = Instant::now();
        let residual: Rc<ResidualCallback>;
        let jacobian: Rc<RefCell<Box<JacobianCallback>>>;
        if matches!(route, Route::PreparedAtom) {
            let callbacks = prepare(
                &config,
                NativeJacobianStorage::Banded {
                    bandwidth: Some((1, 1)),
                },
            )
            .expect("PreparedAtom callbacks should prepare");
            residual = callbacks.residual;
            jacobian = callbacks.jacobian;
        } else {
            let assembly = if matches!(route, Route::Expr) {
                IvpSymbolicAssemblyBackend::ExprLegacy
            } else {
                IvpSymbolicAssemblyBackend::AtomView
            };
            let variables = workload.variables;
            let parameters = workload.parameter_names;
            let values = workload.parameter_values;
            let residual_problem = prepare_symbolic_ivp_residual_problem(
                workload.equations.clone(),
                variables.clone(),
                workload.time_variable.clone(),
                SymbolicIvpProblemOptions::new()
                    .with_symbolic_assembly_backend(assembly)
                    .with_equation_parameters(parameters.clone())
                    .with_equation_parameter_values(values)
                    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
                    .with_telemetry(IvpTelemetry::disabled()),
            )
            .expect("native LSODE2 residual callback should prepare");
            let parameter_handle = residual_problem.parameter_values_handle();
            let callback_residual: Rc<ResidualCallback> =
                Rc::new(move |time, state| (residual_problem.residual)(time, state));
            let compiled_jacobian = crate::numerical::LSODE2::native_jacobian::
                compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry_and_policy(
                    &workload.equations,
                    &variables,
                    &workload.time_variable,
                    Some(&parameters),
                    parameter_handle,
                    NativeJacobianStorage::Banded {
                        bandwidth: Some((1, 1)),
                    },
                    assembly,
                    IvpTelemetry::disabled(),
                    IvpLambdifyExecutionPolicy::Sequential,
                );
            residual = callback_residual;
            jacobian = Rc::new(RefCell::new(compiled_jacobian));
        }
        CallbackPair {
            residual,
            jacobian,
            setup_ms: started.elapsed().as_secs_f64() * 1_000.0,
        }
    }

    fn callback_values(pair: &CallbackPair, state: &DVector<f64>) -> (Vec<f64>, Vec<f64>) {
        let residual = (pair.residual)(0.5, state).as_slice().to_vec();
        let jacobian = match (pair.jacobian.borrow_mut())(0.5, state) {
            BdfJacobian::Banded(matrix) => matrix.as_slice().to_vec(),
            other => panic!("expected Banded callback output, got {other:?}"),
        };
        (residual, jacobian)
    }

    fn time_callbacks(pair: &CallbackPair, state: &DVector<f64>, repeats: usize) -> (f64, f64) {
        let started_residual = Instant::now();
        let mut checksum = 0.0;
        for index in 0..repeats {
            let values = (pair.residual)(0.5 + index as f64 * 1.0e-5, state);
            checksum += values[index % values.len()];
        }
        let residual_ms = started_residual.elapsed().as_secs_f64() * 1_000.0 / repeats as f64;
        let started_jacobian = Instant::now();
        for index in 0..repeats {
            match (pair.jacobian.borrow_mut())(0.5 + index as f64 * 1.0e-5, state) {
                BdfJacobian::Banded(matrix) => {
                    checksum += matrix.as_slice()[index % matrix.as_slice().len()]
                }
                other => panic!("expected Banded callback output, got {other:?}"),
            }
        }
        black_box(checksum);
        let jacobian_ms = started_jacobian.elapsed().as_secs_f64() * 1_000.0 / repeats as f64;
        (residual_ms, jacobian_ms)
    }

    #[test]
    #[ignore = "release A/B evidence for LSODE2 PreparedAtom; use --ignored --nocapture --test-threads=1"]
    fn prepared_atom_diffusion_chain_release_ab_story() {
        let _report = crate::Utils::test_reporting::TestReportCapture::new(
            "LSODE2_PreparedAtom",
            "numerical::LSODE2::prepared_atom_runtime::tests::prepared_atom_diffusion_chain_release_ab_story",
        );
        let dimension = std::env::var("LSODE2_PREPARED_ATOM_DIMENSION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value >= 8)
            .unwrap_or(256);
        let repeats = std::env::var("LSODE2_PREPARED_ATOM_REPEATS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value >= 3)
            .unwrap_or(7);
        let routes = [Route::Expr, Route::Atom, Route::PreparedAtom];
        let mut preparation = [Samples::default(), Samples::default(), Samples::default()];
        let mut solve = [Samples::default(), Samples::default(), Samples::default()];
        let callback_repeats = std::env::var("LSODE2_PREPARED_ATOM_CALLBACK_REPEATS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(128);
        let state = DVector::from_iterator(
            dimension,
            (0..dimension).map(|index| 0.25 + 0.001 * (index % 17) as f64),
        );
        let callbacks = routes.map(|route| prepare_callbacks(dimension, route));
        let expected_callbacks = callback_values(&callbacks[0], &state);
        let mut callback_residual = [Samples::default(), Samples::default(), Samples::default()];
        let mut callback_jacobian = [Samples::default(), Samples::default(), Samples::default()];
        for (index, route) in routes.iter().copied().enumerate() {
            let actual = callback_values(&callbacks[index], &state);
            assert!(max_slice_diff(&actual.0, &expected_callbacks.0) <= 1.0e-12);
            assert!(max_slice_diff(&actual.1, &expected_callbacks.1) <= 1.0e-12);
            reportln!(
                "callback preparation | {} | {:.3} ms",
                route.label(),
                callbacks[index].setup_ms
            );
        }
        let mut route_work: [Option<Vec<u64>>; 3] = [None, None, None];
        let mut expr_atom_max_time_diff = 0.0_f64;
        let mut expr_atom_max_state_diff = 0.0_f64;

        reportln!(
            "[LSODE2 PreparedAtom A/B] dimension={dimension}; repeats={repeats}; same diffusion-chain BandedFaithful workload; sequential callbacks"
        );
        reportln!(
            "Preparation and solve wall times are separate; A/B routes are interleaved per repetition."
        );

        for _ in 0..repeats {
            let mut round: [Option<(Vec<f64>, Vec<f64>, Vec<u64>)>; 3] = [None, None, None];
            for (route_index, route) in routes.iter().copied().enumerate() {
                let (residual_ms, jacobian_ms) =
                    time_callbacks(&callbacks[route_index], &state, callback_repeats);
                callback_residual[route_index].0.push(residual_ms);
                callback_jacobian[route_index].0.push(jacobian_ms);
                let (prepare_ms, solve_ms, times, states, work) = run_full_solve(dimension, route);
                preparation[route_index].0.push(prepare_ms);
                solve[route_index].0.push(solve_ms);
                if let Some(expected_work) = route_work[route_index].as_ref() {
                    assert_eq!(
                        work.as_slice(),
                        expected_work.as_slice(),
                        "discrete work varies between repeats for {}",
                        route.label()
                    );
                } else {
                    route_work[route_index] = Some(work.clone());
                }
                round[route_index] = Some((times, states, work));
            }
            let expr = round[0].as_ref().unwrap();
            let atom = round[1].as_ref().unwrap();
            let prepared = round[2].as_ref().unwrap();
            expr_atom_max_time_diff = expr_atom_max_time_diff.max(max_slice_diff(&expr.0, &atom.0));
            expr_atom_max_state_diff =
                expr_atom_max_state_diff.max(max_slice_diff(&expr.1, &atom.1));
            assert_eq!(
                atom.2, prepared.2,
                "AtomNative and PreparedAtom discrete work must match"
            );
            assert!(
                max_slice_diff(&atom.0, &prepared.0) <= 1.0e-14,
                "AtomNative and PreparedAtom time grids must match"
            );
            assert!(
                max_slice_diff(&atom.1, &prepared.1) <= 1.0e-10,
                "AtomNative and PreparedAtom trajectories must match"
            );
        }

        reportln!(
            "ExprLegacy versus AtomNative diagnostic: max_time_diff={expr_atom_max_time_diff:.3e}; max_state_diff={expr_atom_max_state_diff:.3e}; counts are reported per route, not used as PreparedAtom acceptance gate."
        );

        reportln!(
            "route | residual_callback_us | residual_MAD_us | jacobian_callback_us | jacobian_MAD_us | prepare_median_ms | prepare_MAD_ms | solve_median_ms | solve_MAD_ms | accepted | rejected | residuals | jacobians | linear_solves | correctness"
        );
        reportln!(
            "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---"
        );
        for (index, route) in routes.iter().copied().enumerate() {
            let (prepare_median, prepare_mad) = preparation[index].median_mad();
            let (solve_median, solve_mad) = solve[index].median_mad();
            let (residual_median, residual_mad) = callback_residual[index].median_mad();
            let (jacobian_median, jacobian_mad) = callback_jacobian[index].median_mad();
            let work = route_work[index].as_ref().unwrap();
            let correctness_scope = match route {
                Route::Expr => "ExprLegacy diagnostic-only",
                Route::Atom | Route::PreparedAtom => "strict AtomNative/PreparedAtom gate",
            };
            reportln!(
                "{} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {}",
                route.label(),
                residual_median * 1000.0,
                residual_mad * 1000.0,
                jacobian_median * 1000.0,
                jacobian_mad * 1000.0,
                prepare_median,
                prepare_mad,
                solve_median,
                solve_mad,
                work[0],
                work[1],
                work[2],
                work[3],
                work[4],
                correctness_scope
            );
        }
        black_box((expr_atom_max_time_diff, expr_atom_max_state_diff));
    }
}
