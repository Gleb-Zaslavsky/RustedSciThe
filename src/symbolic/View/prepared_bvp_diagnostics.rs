//! Debug-only correctness checks for the experimental PreparedAtomSystem BVP route.
//!
//! These tests intentionally live with the symbolic runtime rather than in a
//! benchmark. Iteration-count changes are treated as numerical evidence, not
//! as performance noise.

use super::{FunctionMap, PreparedAtomSystem, PreparedAtomSystemScratch};
use crate::numerical::BVP_Damp::BVP_traits::{
    BandedMatrixType, BvpCallbackError, BvpLinearSolveError, Fun, Jac, MatrixType, VectorType,
    Vectors_type_casting,
};
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{DampedSolverOptions, NRBVP, SolverParams};
use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use crate::somelinalg::banded::banded_assembly::BandedAssembly;
use crate::symbolic::bvp::telemetry::{BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode};
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{BvpSymbolicAssemblyBackend, Jacobian};
use nalgebra::{DMatrix, DVector};
use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex},
};

type CallbackTrace = Arc<Mutex<Vec<(u8, Vec<f64>, Vec<f64>)>>>;

fn trace_callback(trace: &CallbackTrace, kind: u8, input: &[f64], output: &[f64]) {
    trace.lock().expect("callback trace lock poisoned").push((
        kind,
        input.to_vec(),
        output.to_vec(),
    ));
}

fn banded_values(matrix: &dyn MatrixType) -> Vec<f64> {
    matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("BVP Jacobian callback must remain compact Banded")
        .assembly
        .diagonals()
        .iter()
        .flat_map(|diagonal| diagonal.iter().copied())
        .collect()
}

fn combustion_problem() -> (Vec<Expr>, Vec<String>, HashMap<String, Vec<(usize, f64)>>) {
    let names = ["Teta", "q", "C0", "J0", "C1", "J1"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let vars = Expr::parse_vector_expression(names.iter().map(String::as_str).collect::<Vec<_>>());
    let [teta, q, c0, j0, _c1, j1]: [Expr; 6] = vars.try_into().expect("six combustion variables");

    let q_heat = Expr::Const(3000.0 * 1e3 * 0.034);
    let dt = Expr::Const(600.0);
    let t_scale = Expr::Const(600.0);
    let length: f64 = 3e-4;
    let m0 = 34.2 / 1000.0;
    let lambda = Expr::Const(0.07);
    let pressure = 2e6;
    let temperature = 1500.0;
    let pe_q = Expr::Const(0.0090168);
    let d_ro = [Expr::Const(2.88e-4), Expr::Const(2.88e-4)];
    let pe_d = [Expr::Const(1.50e-3), Expr::Const(1.50e-3)];
    let ro_m_value = m0 * pressure / (8.314 * temperature);
    let mass = Expr::Const(m0);
    let r_g = Expr::Const(8.314);
    let ro_m = Expr::Const(ro_m_value);
    let m_reag = Expr::Const(0.342);
    let q_scale = Expr::Const(length.powf(2.0) / 600.0);
    let flux_scale = Expr::Const(length.powf(2.0));
    let rate = Expr::Const(1.3e5)
        * Expr::exp(-Expr::Const(5000.0 * 4.184) / (r_g * (teta * t_scale + dt)))
        * c0
        * (ro_m.clone() / m_reag.clone());
    let equations = vec![
        q.clone() / lambda,
        q * pe_q - q_heat * rate.clone() * q_scale,
        j0.clone() / d_ro[0].clone(),
        j0 * pe_d[0].clone()
            - (mass.clone() * Expr::Const(-1.0) * rate.clone() * ro_m.clone() / mass.clone())
                * flux_scale.clone(),
        j1.clone() / d_ro[1].clone(),
        j1 * pe_d[1].clone() - (mass.clone() * rate * ro_m / mass) * flux_scale,
    ];
    let boundary = HashMap::from([
        ("Teta".to_string(), vec![(0, (1000.0 - 600.0) / 600.0)]),
        ("q".to_string(), vec![(1, 1e-10)]),
        ("C0".to_string(), vec![(0, 1.0)]),
        ("J0".to_string(), vec![(1, 1e-7)]),
        ("C1".to_string(), vec![(0, 1e-3)]),
        ("J1".to_string(), vec![(1, 1e-7)]),
    ]);
    (equations, names, boundary)
}

fn options(frontend: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
    let mut options = DampedSolverOptions::sparse_damped()
        .with_banded_lambdify()
        .with_generated_backend_config(
            GeneratedBackendConfig::banded_lambdify_defaults()
                .with_symbolic_assembly_backend(frontend)
                .with_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Off)
                .with_lambdify_execution_policy(BvpLambdifyExecutionPolicy::Sequential),
        )
        .with_strategy_params(Some(SolverParams {
            max_jac: Some(6),
            max_damp_iter: Some(6),
            damp_factor: Some(0.5),
            adaptive: None,
        }))
        .with_abs_tolerance(1e-6)
        .with_rel_tolerance(
            ["Teta", "q", "C0", "J0", "C1", "J1"]
                .into_iter()
                .map(|name| (name.to_string(), 1e-5))
                .collect(),
        )
        .with_max_iterations(100)
        .with_loglevel(Some("off".to_string()))
        .with_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    options.bounds = Some(HashMap::from([
        ("Teta".to_string(), (0.0, 10.0)),
        ("q".to_string(), (-1e20, 1e20)),
        ("C0".to_string(), (0.0, 1.5)),
        ("J0".to_string(), (-1e2, 1e2)),
        ("C1".to_string(), (0.0, 1.5)),
        ("J1".to_string(), (-1e2, 1e2)),
    ]));
    options
}

fn solver(steps: usize, frontend: BvpSymbolicAssemblyBackend) -> NRBVP {
    let (equations, names, boundary) = combustion_problem();
    let guess = DMatrix::from_element(names.len(), steps, 0.99);
    let mut solver = NRBVP::new_with_options(
        equations,
        guess,
        names,
        "x".to_string(),
        boundary,
        0.0,
        1.0,
        steps,
        options(frontend),
    );
    solver.dont_save_log(true);
    solver
}

struct PreparedFun {
    system: Arc<PreparedAtomSystem>,
    scratch: RefCell<PreparedAtomSystemScratch>,
    trace: CallbackTrace,
}

impl Fun for PreparedFun {
    fn call(&self, x: f64, values: &dyn VectorType) -> Box<dyn VectorType> {
        self.try_call(x, values)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn try_call(
        &self,
        _x: f64,
        values: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpCallbackError> {
        let input = values
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .ok_or_else(|| BvpCallbackError::TypeMismatch {
                expected: "DVector",
                actual: values.vec_type(),
            })?;
        let mut output = vec![0.0; self.system.residual_len()];
        self.system
            .residual_into(
                input.as_slice(),
                &mut output,
                &mut self.scratch.borrow_mut(),
            )
            .map_err(|error| BvpCallbackError::Panic {
                message: error.to_string(),
            })?;
        trace_callback(&self.trace, 0, input.as_slice(), &output);
        Ok(Box::new(DVector::from_vec(output)))
    }
}

struct PreparedJac {
    system: Arc<PreparedAtomSystem>,
    delegate: Box<dyn Jac>,
    scratch: PreparedAtomSystemScratch,
    residual: Vec<f64>,
    values: Vec<f64>,
    bandwidth: (usize, usize),
    template: BandedMatrixType,
    trace: CallbackTrace,
}

impl PreparedJac {
    fn evaluate(&mut self, input: &[f64]) -> Result<BandedMatrixType, BvpCallbackError> {
        self.system
            .residual_and_jacobian_into(
                input,
                &mut self.residual,
                &mut self.values,
                &mut self.scratch,
            )
            .map_err(|error| BvpCallbackError::Panic {
                message: error.to_string(),
            })?;
        let mut assembly =
            BandedAssembly::zeros(self.system.input_len(), self.bandwidth.0, self.bandwidth.1)
                .map_err(|error| BvpCallbackError::Panic {
                    message: error.to_string(),
                })?;
        for (entry, value) in self.system.jacobian_entries().zip(&self.values) {
            assembly
                .set(entry.row, entry.col, *value)
                .map_err(|error| BvpCallbackError::Panic {
                    message: error.to_string(),
                })?;
        }
        let mut matrix = self.template.clone();
        matrix.set_assembly(assembly);
        let output = matrix
            .assembly
            .diagonals()
            .iter()
            .flat_map(|diagonal| diagonal.iter().copied())
            .collect::<Vec<_>>();
        trace_callback(&self.trace, 1, input, &output);
        Ok(matrix)
    }
}

impl Jac for PreparedJac {
    fn call(&mut self, _x: f64, vector: &dyn VectorType) -> Box<dyn MatrixType> {
        let input = vector
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .expect("Banded Newton state is a DVector");
        Box::new(
            self.evaluate(input.as_slice())
                .unwrap_or_else(|error| panic!("{error}")),
        )
    }
    fn inv(&mut self, matrix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType> {
        self.delegate.inv(matrix, tol, max_iter)
    }
    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vector: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Box<dyn VectorType> {
        self.delegate.solve_sys(matrix, vector, tol, max_iter)
    }
    fn try_solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vector: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        self.delegate.try_solve_sys(matrix, vector, tol, max_iter)
    }
}

struct TracedFun {
    delegate: Box<dyn Fun>,
    trace: CallbackTrace,
}

impl Fun for TracedFun {
    fn call(&self, x: f64, values: &dyn VectorType) -> Box<dyn VectorType> {
        let input = values
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .expect("BVP state is DVector");
        let output = self.delegate.call(x, values).to_DVectorType();
        trace_callback(&self.trace, 0, input.as_slice(), output.as_slice());
        Box::new(output)
    }

    fn try_call(
        &self,
        x: f64,
        values: &dyn VectorType,
    ) -> Result<Box<dyn VectorType>, BvpCallbackError> {
        let input = values
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .ok_or_else(|| BvpCallbackError::TypeMismatch {
                expected: "DVector",
                actual: values.vec_type(),
            })?;
        let output = self.delegate.try_call(x, values)?.to_DVectorType();
        trace_callback(&self.trace, 0, input.as_slice(), output.as_slice());
        Ok(Box::new(output))
    }
}

struct TracedJac {
    delegate: Box<dyn Jac>,
    trace: CallbackTrace,
}

impl Jac for TracedJac {
    fn call(&mut self, x: f64, vector: &dyn VectorType) -> Box<dyn MatrixType> {
        let input = vector
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .expect("BVP state is DVector");
        let output = self.delegate.call(x, vector);
        trace_callback(&self.trace, 1, input.as_slice(), &banded_values(&*output));
        output
    }
    fn inv(&mut self, matrix: &dyn MatrixType, tol: f64, max_iter: usize) -> Box<dyn MatrixType> {
        self.delegate.inv(matrix, tol, max_iter)
    }
    fn solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vector: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Box<dyn VectorType> {
        self.delegate.solve_sys(matrix, vector, tol, max_iter)
    }
    fn try_solve_sys(
        &mut self,
        matrix: &dyn MatrixType,
        vector: &dyn VectorType,
        tol: f64,
        max_iter: usize,
    ) -> Result<Box<dyn VectorType>, BvpLinearSolveError> {
        self.delegate.try_solve_sys(matrix, vector, tol, max_iter)
    }
}

struct NullFun;

impl Fun for NullFun {
    fn call(&self, _: f64, _: &dyn VectorType) -> Box<dyn VectorType> {
        unreachable!("temporary callback replaced before solve")
    }
}

fn prepared_system(steps: usize) -> (Arc<PreparedAtomSystem>, (usize, usize)) {
    let (equations, names, boundary) = combustion_problem();
    let bounds = HashMap::from([
        ("Teta".to_string(), (0.0, 10.0)),
        ("q".to_string(), (-1e20, 1e20)),
        ("C0".to_string(), (0.0, 1.5)),
        ("J0".to_string(), (-1e2, 1e2)),
        ("C1".to_string(), (0.0, 1.5)),
        ("J1".to_string(), (-1e2, 1e2)),
    ]);
    let tolerances = names
        .iter()
        .map(|name| (name.clone(), 1e-5))
        .collect::<HashMap<_, _>>();
    let mut jacobian = Jacobian::new();
    jacobian.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
    jacobian.set_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Off);
    jacobian.set_lambdify_execution_policy(BvpLambdifyExecutionPolicy::Sequential);
    let bundle = jacobian
        .try_generate_sparse_solver_bundle_with_backend_selection(
            equations,
            names,
            "x".to_string(),
            None,
            0.0,
            None,
            Some(steps),
            Some(1.0 / steps as f64),
            None,
            boundary,
            Some(bounds),
            Some(tolerances),
            "forward".to_string(),
            "Banded".to_string(),
            None,
            BackendSelectionPolicy::LambdifyOnly,
            None,
        )
        .expect("AtomView/Banded discretization should build");
    let bandwidth = bundle.bandwidth.expect("Banded route exposes bandwidth");
    let atom_system = bundle
        .execution
        .selected()
        .prepared_problem
        .atom_discretized_system
        .as_ref()
        .expect("AtomView route retains discretized atoms");
    let prepared = PreparedAtomSystem::try_from_atoms(
        atom_system.vector_of_functions.as_slice(),
        atom_system.variable_string.as_slice(),
        Some(bandwidth),
        FunctionMap::new(),
    )
    .expect("PreparedAtomSystem should compile combustion atoms");
    (Arc::new(prepared), bandwidth)
}

fn install_prepared(
    solver: &mut NRBVP,
    system: Arc<PreparedAtomSystem>,
    bandwidth: (usize, usize),
    trace: CallbackTrace,
) {
    solver
        .try_eq_generate(None, None)
        .expect("native solver generation");
    let guess = DVector::from_element(solver.values.len() * solver.n_steps, 0.99);
    let vector = Vectors_type_casting(&guess, "Banded".to_string());
    let mut delegate = solver.jac.take().expect("native Banded Jacobian");
    let matrix = delegate.call(1.0, &*vector);
    let template = matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("generated Banded matrix")
        .clone();
    let fun = PreparedFun {
        system: Arc::clone(&system),
        scratch: RefCell::default(),
        trace: Arc::clone(&trace),
    };
    let jac = PreparedJac {
        residual: vec![0.0; system.residual_len()],
        values: vec![0.0; system.jacobian_len()],
        scratch: PreparedAtomSystemScratch::default(),
        system,
        delegate,
        bandwidth,
        template,
        trace,
    };
    solver
        .try_replace_prepared_callbacks(Box::new(fun), Some(Box::new(jac)))
        .expect("explicit callback replacement should preserve prepared lifecycle");
}

fn callback_max_diffs(
    native: &mut NRBVP,
    prepared: &mut NRBVP,
    state: &DVector<f64>,
) -> (f64, f64) {
    let vector = Vectors_type_casting(state, "Banded".to_string());
    let native_residual = native.fun.call(native.p, &*vector).to_DVectorType();
    let prepared_residual = prepared.fun.call(prepared.p, &*vector).to_DVectorType();
    let residual_diff = native_residual
        .iter()
        .zip(prepared_residual.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);

    let native_matrix = native
        .jac
        .as_deref_mut()
        .expect("native Jacobian")
        .call(native.p, &*vector);
    let prepared_matrix = prepared
        .jac
        .as_deref_mut()
        .expect("Prepared Jacobian")
        .call(prepared.p, &*vector);
    let native_band = native_matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("native compact matrix");
    let prepared_band = prepared_matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("Prepared compact matrix");
    assert_eq!(
        native_band.assembly.diagonals().len(),
        prepared_band.assembly.diagonals().len()
    );
    let jacobian_diff = native_band
        .assembly
        .diagonals()
        .iter()
        .zip(prepared_band.assembly.diagonals())
        .flat_map(|(lhs, rhs)| lhs.iter().zip(rhs))
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    (residual_diff, jacobian_diff)
}

#[test]
fn combustion_prepared_route_preserves_debug_iteration_trace() {
    let steps = std::env::var("RST_PREPARED_BVP_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize);
    let (system, bandwidth) = prepared_system(steps);
    for sample in 0..3 {
        // The release combustion stress creates a fresh solver in each
        // process-isolated sample. Keep this correctness protocol equivalent:
        // do not turn sample repetition into a continuation on one mutable solver.
        let mut native = solver(steps, BvpSymbolicAssemblyBackend::AtomView);
        native
            .try_eq_generate(None, None)
            .expect("native generation");
        let native_trace = Arc::new(Mutex::new(Vec::new()));
        let native_fun = TracedFun {
            delegate: std::mem::replace(&mut native.fun, Box::new(NullFun)),
            trace: Arc::clone(&native_trace),
        };
        let native_jac = TracedJac {
            delegate: native.jac.take().expect("native Jacobian"),
            trace: Arc::clone(&native_trace),
        };
        native
            .try_replace_prepared_callbacks(Box::new(native_fun), Some(Box::new(native_jac)))
            .expect("native trace wrappers keep the generated runtime valid");

        let mut prepared = solver(steps, BvpSymbolicAssemblyBackend::AtomView);
        let prepared_trace = Arc::new(Mutex::new(Vec::new()));
        install_prepared(
            &mut prepared,
            Arc::clone(&system),
            bandwidth,
            Arc::clone(&prepared_trace),
        );
        let initial = DVector::from_column_slice(native.initial_guess.as_slice());
        let (initial_residual_diff, initial_jacobian_diff) =
            callback_max_diffs(&mut native, &mut prepared, &initial);
        assert!(
            initial_residual_diff <= 1e-13,
            "sample {sample}: initial residual mismatch: {initial_residual_diff:e}"
        );
        assert!(
            initial_jacobian_diff <= 1e-13,
            "sample {sample}: initial Jacobian mismatch: {initial_jacobian_diff:e}"
        );

        native_trace.lock().expect("native trace lock").clear();
        prepared_trace.lock().expect("Prepared trace lock").clear();
        let native_result = native.try_solver_prepared().expect("native solve");
        let native_stats = native.get_statistics().telemetry.counters;
        let prepared_result = prepared.try_solver_prepared().expect("Prepared solve");
        let prepared_stats = prepared.get_statistics().telemetry.counters;
        let state_diff = match (&native_result, &prepared_result) {
            (Some(lhs), Some(rhs)) => lhs
                .iter()
                .zip(rhs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
            _ => f64::INFINITY,
        };
        eprintln!(
            "[Prepared BVP debug parity] steps={steps} independent_sample={sample} native_converged={} prepared_converged={} native(iter={}, residual={}, jacobian={}, trials={}, solves={}) prepared(iter={}, residual={}, jacobian={}, trials={}, solves={}) state_max_diff={state_diff:.3e}",
            native_result.is_some(),
            prepared_result.is_some(),
            native_stats.iterations,
            native_stats.residual_requests,
            native_stats.jacobian_requests,
            native_stats.damping_trials,
            native_stats.linear_solves,
            prepared_stats.iterations,
            prepared_stats.residual_requests,
            prepared_stats.jacobian_requests,
            prepared_stats.damping_trials,
            prepared_stats.linear_solves,
        );
        let native_events = native_trace.lock().expect("native trace lock").clone();
        let prepared_events = prepared_trace.lock().expect("Prepared trace lock").clone();
        if native_events != prepared_events {
            let first_diff = native_events
                .iter()
                .zip(&prepared_events)
                .position(|(lhs, rhs)| lhs != rhs)
                .unwrap_or(native_events.len().min(prepared_events.len()));
            eprintln!(
                "[Prepared BVP callback trace divergence] sample={sample} native_events={} prepared_events={} first_diff={first_diff} native_event={:?} prepared_event={:?}",
                native_events.len(),
                prepared_events.len(),
                native_events.get(first_diff).map(|(kind, input, output)| (
                    *kind,
                    input.len(),
                    output.len()
                )),
                prepared_events
                    .get(first_diff)
                    .map(|(kind, input, output)| (*kind, input.len(), output.len())),
            );
        }
        assert_eq!(
            native_events, prepared_events,
            "callback input sequence differs on independent sample {sample}"
        );

        assert_eq!(
            native_result.is_some(),
            prepared_result.is_some(),
            "convergence status differs on independent sample {sample}"
        );
        assert!(
            native_result.is_some(),
            "solver-native combustion solve must converge on independent sample {sample}"
        );
        assert!(
            prepared_result.is_some(),
            "Prepared combustion solve must converge on independent sample {sample}"
        );
        assert_eq!(
            native_stats.iterations, prepared_stats.iterations,
            "Newton iteration count differs on independent sample {sample}"
        );
        assert_eq!(
            native_stats.residual_requests, prepared_stats.residual_requests,
            "residual call count differs on independent sample {sample}"
        );
        assert_eq!(
            native_stats.jacobian_requests, prepared_stats.jacobian_requests,
            "Jacobian call count differs on independent sample {sample}"
        );
        assert!(
            state_diff <= 1e-8,
            "independent sample {sample}: converged states differ by {state_diff:e}"
        );
        let native_state = native_result.as_ref().expect("convergence asserted above");
        let (residual_diff, jacobian_diff) =
            callback_max_diffs(&mut native, &mut prepared, native_state);
        eprintln!(
            "[Prepared BVP callback parity] steps={steps} independent_sample={sample} converged residual_max_diff={residual_diff:.3e} Jacobian_max_diff={jacobian_diff:.3e} per_solve_iterations={} residual_requests={} jacobian_requests={}",
            native_stats.iterations, native_stats.residual_requests, native_stats.jacobian_requests,
        );
        assert!(
            residual_diff <= 1e-13,
            "independent sample {sample}: residual mismatch {residual_diff:e}"
        );
        assert!(
            jacobian_diff <= 1e-13,
            "independent sample {sample}: Jacobian mismatch {jacobian_diff:e}"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolveCounterDelta {
    iterations: u64,
    residual_requests: u64,
    jacobian_requests: u64,
    jacobian_recalculations: u64,
    damping_trials: u64,
    damping_rejections: u64,
    linear_solves: u64,
    factorizations: u64,
    rhs_solves: u64,
}

fn solve_counter_delta(
    after: &crate::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
    before: &crate::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
) -> SolveCounterDelta {
    SolveCounterDelta {
        iterations: after.iterations - before.iterations,
        residual_requests: after.residual_requests - before.residual_requests,
        jacobian_requests: after.jacobian_requests - before.jacobian_requests,
        jacobian_recalculations: after.jacobian_recalculations - before.jacobian_recalculations,
        damping_trials: after.damping_trials - before.damping_trials,
        damping_rejections: after.damping_rejections - before.damping_rejections,
        linear_solves: after.linear_solves - before.linear_solves,
        factorizations: after.factorizations - before.factorizations,
        rhs_solves: after.rhs_solves - before.rhs_solves,
    }
}

fn solution_max_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn combustion_same_instance_replay_requires_explicit_initial_guess_reset() {
    let steps = std::env::var("RST_PREPARED_BVP_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize);
    let (system, bandwidth) = prepared_system(steps);
    let initial_guess = DMatrix::from_element(6, steps, 0.99);

    let mut atom = solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    atom.try_eq_generate(None, None)
        .expect("AtomNative preparation");
    let mut prepared = solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    install_prepared(
        &mut prepared,
        system,
        bandwidth,
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut expr = solver(steps, BvpSymbolicAssemblyBackend::ExprLegacy);
    expr.try_eq_generate(None, None)
        .expect("ExprLegacy preparation");

    let mut routes = [
        ("AtomNative", &mut atom),
        ("PreparedAtom", &mut prepared),
        ("ExprLegacy", &mut expr),
    ];
    let mut previous = Vec::with_capacity(routes.len());
    let mut first_solutions = Vec::with_capacity(routes.len());
    for (name, route) in &mut routes {
        let before = route.get_statistics().telemetry.counters;
        let solution = route
            .try_solver_prepared()
            .unwrap_or_else(|error| panic!("{name} first solve failed: {error}"));
        let after = route.get_statistics().telemetry.counters;
        let delta = solve_counter_delta(&after, &before);
        eprintln!(
            "[BVP same-instance replay] route={name} phase=initial status={} per_solve={delta:?}",
            solution.is_some()
        );
        first_solutions.push(solution.expect("initial combustion solve should converge"));
        previous.push(after);
    }
    for (route_index, name) in ["PreparedAtom", "ExprLegacy"].into_iter().enumerate() {
        let diff = solution_max_diff(
            first_solutions[0].as_slice(),
            first_solutions[route_index + 1].as_slice(),
        );
        assert!(
            diff <= 1e-8,
            "initial AtomNative/{name} solution diff={diff:e}"
        );
    }

    let mut replay_solutions = Vec::with_capacity(routes.len());
    let mut replay_deltas = Vec::with_capacity(routes.len());
    for (index, (name, route)) in routes.iter_mut().enumerate() {
        let solution = route
            .try_solver_prepared()
            .unwrap_or_else(|error| panic!("{name} raw same-instance replay failed: {error}"));
        let after = route.get_statistics().telemetry.counters;
        let delta = solve_counter_delta(&after, &previous[index]);
        eprintln!(
            "[BVP same-instance replay] route={name} phase=raw_replay status={} per_solve_delta={delta:?} cumulative_iterations={}",
            solution.is_some(),
            after.iterations
        );
        replay_solutions.push(solution);
        replay_deltas.push(delta);
        previous[index] = after;
    }
    assert_eq!(
        replay_solutions[0].is_some(),
        replay_solutions[1].is_some(),
        "AtomNative and PreparedAtom raw replay status diverged"
    );
    assert_eq!(
        replay_solutions[0].is_some(),
        replay_solutions[2].is_some(),
        "AtomNative and ExprLegacy raw replay status diverged"
    );
    assert_eq!(
        replay_deltas[0], replay_deltas[1],
        "AtomNative/PreparedAtom raw replay counter deltas diverged"
    );
    assert_eq!(
        replay_deltas[0], replay_deltas[2],
        "AtomNative/ExprLegacy raw replay counter deltas diverged"
    );

    for (name, route) in &mut routes {
        route
            .try_set_initial_guess(initial_guess.clone())
            .unwrap_or_else(|error| panic!("{name} explicit replay reset failed: {error}"));
    }
    let mut reset_solutions = Vec::with_capacity(routes.len());
    let mut reset_deltas = Vec::with_capacity(routes.len());
    for (index, (name, route)) in routes.iter_mut().enumerate() {
        let solution = route
            .try_solver_prepared()
            .unwrap_or_else(|error| panic!("{name} reset replay failed: {error}"))
            .expect("explicit initial-guess reset should restore convergence");
        let after = route.get_statistics().telemetry.counters;
        let delta = solve_counter_delta(&after, &previous[index]);
        eprintln!(
            "[BVP same-instance replay] route={name} phase=reset_replay per_solve_delta={delta:?} cumulative_iterations={}",
            after.iterations
        );
        reset_solutions.push(solution);
        reset_deltas.push(delta);
    }
    assert_eq!(
        reset_deltas[0], reset_deltas[1],
        "AtomNative/PreparedAtom reset replay counter deltas diverged"
    );
    assert_eq!(
        reset_deltas[0], reset_deltas[2],
        "AtomNative/ExprLegacy reset replay counter deltas diverged"
    );
    for index in 1..reset_solutions.len() {
        let diff = solution_max_diff(
            reset_solutions[0].as_slice(),
            reset_solutions[index].as_slice(),
        );
        assert!(
            diff <= 1e-8,
            "reset replay route[{index}] solution diff={diff:e}"
        );
    }
}
