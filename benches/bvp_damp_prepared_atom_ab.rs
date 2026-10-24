//! Addressed A/B diagnostic: production BVP_Damp AtomView/Banded callbacks
//! versus the opt-in solver-independent `PreparedAtomSystem` on combustion.
//!
//! `PreparedAtomSystem` remains opt-in: this benchmark installs it through the
//! solver's explicit prepared-callback replacement seam, without changing the
//! production default. Full-solve routes use the same combustion equations,
//! unknown ordering, band layout and sequential policy. Every
//! warm-solve samples use a fresh solver object with callback preparation
//! measured separately; a distinct end-to-end series times normal `try_solve`
//! including callback regeneration. Same-instance replay is checked separately.
//! Newton/factorization state is not carried over between independent samples.
//! Run normally for uninstrumented timing; enable
//! `prepared-atom-allocation-audit` separately for allocator counts and
//! retained-memory deltas.
//!
//! Run a Debug correctness/diagnostic pass with:
//! `cargo bench --profile dev --bench bvp_damp_prepared_atom_ab`
//! Run the allocator audit separately with:
//! `cargo bench --profile dev --features prepared-atom-allocation-audit --bench bvp_damp_prepared_atom_ab`
//! Set `BVP_PREPARED_ATOM_STEPS=3000` for the large combustion-3000 workload.

#[cfg(feature = "prepared-atom-allocation-audit")]
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::hint::black_box;
#[cfg(feature = "prepared-atom-allocation-audit")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
    time::Instant,
};

use RustedSciThe::numerical::BVP_Damp::BVP_traits::{
    BandedMatrixType, BvpCallbackError, BvpLinearSolveError, Fun, Jac, MatrixType, VectorType,
    Vectors_type_casting,
};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::somelinalg::banded::banded_assembly::BandedAssembly;
use RustedSciThe::symbolic::View::{FunctionMap, PreparedAtomSystem, PreparedAtomSystemScratch};
use RustedSciThe::symbolic::bvp::telemetry::{
    BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode,
};
use RustedSciThe::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_functions_BVP::{BvpSymbolicAssemblyBackend, Jacobian};
use nalgebra::{DMatrix, DVector};
use tabled::{builder::Builder, settings::Style};

#[cfg(feature = "prepared-atom-allocation-audit")]
struct CountingAllocator;

#[cfg(feature = "prepared-atom-allocation-audit")]
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "prepared-atom-allocation-audit")]
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "prepared-atom-allocation-audit")]
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "prepared-atom-allocation-audit")]
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "prepared-atom-allocation-audit")]
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "prepared-atom-allocation-audit")]
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "prepared-atom-allocation-audit")]
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[cfg(feature = "prepared-atom-allocation-audit")]
fn update_peak(candidate: usize) {
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while candidate > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            peak,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

// SAFETY: allocation operations forward their original layouts to System.
#[cfg(feature = "prepared-atom-allocation-audit")]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is passed through unchanged to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            update_peak(live);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout are forwarded unchanged to System.
        unsafe { System.dealloc(pointer, layout) };
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: pointer, old layout and requested size are forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let live = if new_size >= layout.size() {
                LIVE_BYTES.fetch_add(new_size - layout.size(), Ordering::Relaxed)
                    + (new_size - layout.size())
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed)
                    - (layout.size() - new_size)
            };
            update_peak(live);
        }
        new_pointer
    }
}

#[derive(Clone, Copy, Default)]
struct AllocationSnapshot {
    allocations: usize,
    allocated_bytes: usize,
    deallocations: usize,
    deallocated_bytes: usize,
}

#[cfg(feature = "prepared-atom-allocation-audit")]
fn allocation_snapshot() -> AllocationSnapshot {
    AllocationSnapshot {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

#[cfg(not(feature = "prepared-atom-allocation-audit"))]
fn allocation_snapshot() -> AllocationSnapshot {
    AllocationSnapshot::default()
}

fn allocation_delta(before: AllocationSnapshot, after: AllocationSnapshot) -> AllocationSnapshot {
    #[cfg(not(feature = "prepared-atom-allocation-audit"))]
    let _ = (before, after);
    AllocationSnapshot {
        #[cfg(feature = "prepared-atom-allocation-audit")]
        allocations: after.allocations - before.allocations,
        #[cfg(not(feature = "prepared-atom-allocation-audit"))]
        allocations: 0,
        #[cfg(feature = "prepared-atom-allocation-audit")]
        allocated_bytes: after.allocated_bytes - before.allocated_bytes,
        #[cfg(not(feature = "prepared-atom-allocation-audit"))]
        allocated_bytes: 0,
        #[cfg(feature = "prepared-atom-allocation-audit")]
        deallocations: after.deallocations - before.deallocations,
        #[cfg(not(feature = "prepared-atom-allocation-audit"))]
        deallocations: 0,
        #[cfg(feature = "prepared-atom-allocation-audit")]
        deallocated_bytes: after.deallocated_bytes - before.deallocated_bytes,
        #[cfg(not(feature = "prepared-atom-allocation-audit"))]
        deallocated_bytes: 0,
    }
}

fn audit_value(value: impl std::fmt::Display) -> String {
    if cfg!(feature = "prepared-atom-allocation-audit") {
        value.to_string()
    } else {
        "n/a".to_string()
    }
}

fn median_and_mad(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let mut deviations = sorted
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    (median, deviations[deviations.len() / 2])
}

fn live_bytes() -> usize {
    #[cfg(feature = "prepared-atom-allocation-audit")]
    {
        LIVE_BYTES.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "prepared-atom-allocation-audit"))]
    {
        0
    }
}

fn reset_peak_live_bytes() {
    #[cfg(feature = "prepared-atom-allocation-audit")]
    PEAK_LIVE_BYTES.store(live_bytes(), Ordering::Relaxed);
}

fn peak_live_bytes() -> usize {
    #[cfg(feature = "prepared-atom-allocation-audit")]
    {
        PEAK_LIVE_BYTES.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "prepared-atom-allocation-audit"))]
    {
        0
    }
}

fn combustion_equations() -> (Vec<Expr>, Vec<String>, HashMap<String, Vec<(usize, f64)>>) {
    let variable_names = ["Teta", "q", "C0", "J0", "C1", "J1"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let variables = Expr::parse_vector_expression(
        variable_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    let [teta, q, c0, j0, _c1, j1]: [Expr; 6] = variables
        .try_into()
        .expect("combustion state dimension is fixed at six");

    let q_heat = Expr::Const(3000.0 * 1e3 * 0.034);
    let dt = Expr::Const(600.0);
    let t_scale = Expr::Const(600.0);
    let length: f64 = 3e-4;
    let m0 = 34.2 / 1000.0;
    let lambda = Expr::Const(0.07);
    let pressure = 2e6;
    let temperature = 1500.0;
    let initial_c0 = 1.0;
    let initial_temperature = 1000.0;
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
    let minus = Expr::Const(-1.0);
    let equations = vec![
        q.clone() / lambda,
        q * pe_q - q_heat * rate.clone() * q_scale,
        j0.clone() / d_ro[0].clone(),
        j0 * pe_d[0].clone()
            - (mass.clone() * minus * rate.clone() * ro_m.clone() / mass.clone())
                * flux_scale.clone(),
        j1.clone() / d_ro[1].clone(),
        j1 * pe_d[1].clone() - (mass.clone() * rate * ro_m / mass) * flux_scale,
    ];
    let boundary_conditions = HashMap::from([
        (
            "Teta".to_string(),
            vec![(0, (initial_temperature - 600.0) / 600.0)],
        ),
        ("q".to_string(), vec![(1, 1e-10)]),
        ("C0".to_string(), vec![(0, initial_c0)]),
        ("J0".to_string(), vec![(1, 1e-7)]),
        ("C1".to_string(), vec![(0, 1e-3)]),
        ("J1".to_string(), vec![(1, 1e-7)]),
    ]);
    (equations, variable_names, boundary_conditions)
}

struct PreparedResidualCallback {
    system: Arc<PreparedAtomSystem>,
    scratch: RefCell<PreparedAtomSystemScratch>,
    trace: Option<Arc<Mutex<Vec<Vec<f64>>>>>,
}

impl Fun for PreparedResidualCallback {
    fn call(&self, _x: f64, values: &dyn VectorType) -> Box<dyn VectorType> {
        self.try_call(_x, values)
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
        if let Some(trace) = &self.trace {
            let mut trace = trace.lock().expect("state trace mutex poisoned");
            if trace.len() < 64 {
                trace.push(input.as_slice().to_vec());
            }
        }
        let mut residual = vec![0.0; self.system.residual_len()];
        let mut scratch = self.scratch.borrow_mut();
        self.system
            .residual_into(input.as_slice(), &mut residual, &mut scratch)
            .map_err(|error| BvpCallbackError::Panic {
                message: format!("PreparedAtomSystem residual evaluation failed: {error}"),
            })?;
        Ok(Box::new(DVector::from_vec(residual)))
    }
}

struct PreparedJacobianCallback {
    system: Arc<PreparedAtomSystem>,
    delegate: Box<dyn Jac>,
    scratch: PreparedAtomSystemScratch,
    residual: Vec<f64>,
    values: Vec<f64>,
    bandwidth: (usize, usize),
    matrix_template: BandedMatrixType,
    trace: Option<Arc<Mutex<Vec<Vec<f64>>>>>,
}

impl PreparedJacobianCallback {
    fn evaluate(&mut self, input: &[f64]) -> Result<BandedMatrixType, BvpCallbackError> {
        if let Some(trace) = &self.trace {
            let mut trace = trace.lock().expect("state trace mutex poisoned");
            if trace.len() < 64 {
                trace.push(input.to_vec());
            }
        }
        self.system
            .residual_and_jacobian_into(
                input,
                &mut self.residual,
                &mut self.values,
                &mut self.scratch,
            )
            .map_err(|error| BvpCallbackError::Panic {
                message: format!("PreparedAtomSystem Jacobian evaluation failed: {error}"),
            })?;
        let mut assembly =
            BandedAssembly::zeros(self.system.input_len(), self.bandwidth.0, self.bandwidth.1)
                .map_err(|error| BvpCallbackError::Panic {
                    message: format!("PreparedAtomSystem band allocation failed: {error}"),
                })?;
        for (entry, value) in self.system.jacobian_entries().zip(&self.values) {
            assembly
                .set(entry.row, entry.col, *value)
                .map_err(|error| BvpCallbackError::Panic {
                    message: format!("PreparedAtomSystem band scatter failed: {error}"),
                })?;
        }
        let mut matrix = self.matrix_template.clone();
        matrix.set_assembly(assembly);
        Ok(matrix)
    }
}

impl Jac for PreparedJacobianCallback {
    fn call(&mut self, _x: f64, vector: &dyn VectorType) -> Box<dyn MatrixType> {
        let input = vector
            .as_any()
            .downcast_ref::<DVector<f64>>()
            .expect("Banded BVP callbacks receive a dense Newton vector");
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

fn combustion_solver_options(assembly: BvpSymbolicAssemblyBackend) -> DampedSolverOptions {
    let mut options = DampedSolverOptions::sparse_damped()
        .with_banded_lambdify()
        .with_generated_backend_config(
            GeneratedBackendConfig::banded_lambdify_defaults()
                .with_symbolic_assembly_backend(assembly)
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
        .with_loglevel(Some("off".to_string()));
    options = options.with_bvp_telemetry_mode(BvpTelemetryMode::Counters);
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

fn new_combustion_solver(steps: usize, assembly: BvpSymbolicAssemblyBackend) -> NRBVP {
    let (equations, variable_names, boundary_conditions) = combustion_equations();
    let initial_guess = DMatrix::from_element(variable_names.len(), steps, 0.99);
    let mut solver = NRBVP::new_with_options(
        equations,
        initial_guess,
        variable_names,
        "x".to_string(),
        boundary_conditions,
        0.0,
        1.0,
        steps,
        combustion_solver_options(assembly),
    );
    solver.dont_save_log(true);
    solver
}

fn combustion_route(
    index: usize,
) -> (
    &'static str,
    BvpSymbolicAssemblyBackend,
    RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime,
) {
    use RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::{PreparedAtom, SolverNative};
    match index {
        0 => (
            "AtomNative",
            BvpSymbolicAssemblyBackend::AtomView,
            SolverNative,
        ),
        1 => (
            "PreparedAtom",
            BvpSymbolicAssemblyBackend::AtomView,
            PreparedAtom,
        ),
        _ => (
            "ExprLegacy",
            BvpSymbolicAssemblyBackend::ExprLegacy,
            SolverNative,
        ),
    }
}

fn assert_bvp_counters_doubled(
    once: &RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
    twice: &RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
    route: &str,
) {
    macro_rules! doubled {
        ($field:ident) => {
            assert_eq!(
                twice.$field,
                once.$field * 2,
                "{route} {} counter",
                stringify!($field)
            );
        };
    }
    doubled!(iterations);
    doubled!(scalar_evaluations);
    doubled!(residual_calls);
    doubled!(residual_requests);
    doubled!(jacobian_requests);
    doubled!(jacobian_recalculations);
    doubled!(residual_chunks);
    doubled!(jacobian_chunks);
    doubled!(conversions);
    doubled!(copies);
    doubled!(damping_trials);
    doubled!(damping_rejections);
    doubled!(linear_solves);
    doubled!(factorizations);
    doubled!(factorization_cache_hits);
    doubled!(factorization_invalidations);
    doubled!(rhs_solves);
    doubled!(grid_refinements);
}

fn same_instance_combustion_replay(steps: usize) {
    let mut replay_states = Vec::with_capacity(3);
    let mut replay_counters = Vec::with_capacity(3);
    for route_index in 0..3 {
        let (name, assembly, runtime) = combustion_route(route_index);
        let mut solver = new_combustion_solver(steps, assembly);
        solver.set_callback_runtime(runtime);
        solver
            .try_solve()
            .unwrap_or_else(|error| panic!("first {name} combustion replay failed: {error:?}"))
            .unwrap_or_else(|| panic!("first {name} combustion replay did not converge"));
        let first = solver
            .get_result()
            .expect("first combustion result")
            .clone();
        let first_counters = solver.get_statistics().telemetry.counters.clone();
        solver
            .try_solve()
            .unwrap_or_else(|error| panic!("second {name} combustion replay failed: {error:?}"))
            .unwrap_or_else(|| panic!("second {name} combustion replay did not converge"));
        let second = solver
            .get_result()
            .expect("replayed combustion result")
            .clone();
        let max_diff = first
            .iter()
            .zip(second.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff <= 1e-8, "{name} replay state diff={max_diff:e}");
        let total_counters = solver.get_statistics().telemetry.counters.clone();
        assert_bvp_counters_doubled(&first_counters, &total_counters, name);
        replay_states.push(second);
        replay_counters.push(total_counters);
    }
    for route_index in 1..3 {
        let max_diff = replay_states[0]
            .iter()
            .zip(replay_states[route_index].iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff <= 1e-8,
            "same-instance route {route_index} state diff={max_diff:e}"
        );
        assert_eq!(
            replay_counters[0], replay_counters[route_index],
            "same-instance route {route_index} changed cumulative discrete work"
        );
    }
    println!(
        "Same-instance combustion replay passed: n_steps={steps}; two normal try_solve calls per solver; AtomNative/PreparedAtom/ExprLegacy trajectories and cumulative counters match."
    );
}

fn install_prepared_callbacks(
    solver: &mut NRBVP,
    system: Arc<PreparedAtomSystem>,
    bandwidth: (usize, usize),
    trace: Option<Arc<Mutex<Vec<Vec<f64>>>>>,
) -> std::time::Duration {
    let setup_start = Instant::now();
    solver
        .try_eq_generate(None, None)
        .expect("BVP solver-native AtomView setup should succeed");

    let guess = DVector::from_element(solver.values.len() * solver.n_steps, 0.99);
    let vector = Vectors_type_casting(&guess, "Banded".to_string());
    let mut delegate = solver
        .jac
        .take()
        .expect("generated Banded route should publish a Jacobian callback");
    let matrix = delegate.call(1.0, &*vector);
    let banded = matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("generated Banded route should return BandedMatrixType");
    let matrix_template = banded.clone();

    let fun = PreparedResidualCallback {
        system: Arc::clone(&system),
        scratch: RefCell::new(PreparedAtomSystemScratch::default()),
        trace: trace.clone(),
    };
    let jac = PreparedJacobianCallback {
        system: Arc::clone(&system),
        delegate,
        scratch: PreparedAtomSystemScratch::default(),
        residual: vec![0.0; system.residual_len()],
        values: vec![0.0; system.jacobian_len()],
        bandwidth,
        matrix_template,
        trace,
    };
    solver
        .try_replace_prepared_callbacks(Box::new(fun), Some(Box::new(jac)))
        .expect("experimental callbacks should replace the prepared runtime safely");
    setup_start.elapsed()
}

fn prepare_integrated_prepared_atom_runtime(solver: &mut NRBVP) -> std::time::Duration {
    let setup_start = Instant::now();
    solver.set_callback_runtime(
        RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::PreparedAtom,
    );
    solver
        .try_eq_generate(None, None)
        .expect("integrated PreparedAtom Banded setup should succeed");
    setup_start.elapsed()
}

fn callback_parity_at(
    reference_fun: &dyn Fun,
    reference_jac: &mut dyn Jac,
    candidate_fun: &dyn Fun,
    candidate_jac: &mut dyn Jac,
    state: &dyn VectorType,
) -> (f64, f64) {
    let reference_residual = reference_fun.call(1.0, state).to_DVectorType();
    let candidate_residual = candidate_fun.call(1.0, state).to_DVectorType();
    let residual_diff = reference_residual
        .iter()
        .zip(candidate_residual.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    let reference_matrix = reference_jac.call(1.0, state);
    let candidate_matrix = candidate_jac.call(1.0, state);
    let reference_band = reference_matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("reference callback should preserve compact Banded storage");
    let candidate_band = candidate_matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("Prepared callback should preserve compact Banded storage");
    let jacobian_diff = reference_band
        .assembly
        .diagonals()
        .iter()
        .zip(candidate_band.assembly.diagonals())
        .flat_map(|(lhs, rhs)| lhs.iter().zip(rhs))
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    (residual_diff, jacobian_diff)
}

fn initial_newton_step_parity(
    system: Arc<PreparedAtomSystem>,
    steps: usize,
    bandwidth: (usize, usize),
) -> f64 {
    let mut atom = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    atom.try_eq_generate(None, None)
        .expect("AtomNative step parity setup should succeed");
    let mut prepared = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    install_prepared_callbacks(&mut prepared, system, bandwidth, None);
    let guess = DVector::from_column_slice(atom.initial_guess.as_slice());
    let input = Vectors_type_casting(&guess, "Banded".to_string());
    let atom_residual = atom.fun.call(atom.p, &*input);
    let mut atom_jacobian = atom.jac.take().expect("AtomNative Jacobian is prepared");
    let atom_matrix = atom_jacobian.call(atom.p, &*input);
    let atom_step = atom_matrix
        .try_solve_sys(
            &*atom_residual,
            atom.linear_sys_method.clone(),
            atom.abs_tolerance,
            atom.max_iterations,
            bandwidth,
            &*input,
        )
        .expect("AtomNative initial linear system should solve")
        .to_DVectorType();
    atom.jac = Some(atom_jacobian);

    let prepared_residual = prepared.fun.call(prepared.p, &*input);
    let mut prepared_jacobian = prepared
        .jac
        .take()
        .expect("Prepared Jacobian callback is installed");
    let prepared_matrix = prepared_jacobian.call(prepared.p, &*input);
    let prepared_step = prepared_matrix
        .try_solve_sys(
            &*prepared_residual,
            prepared.linear_sys_method.clone(),
            prepared.abs_tolerance,
            prepared.max_iterations,
            bandwidth,
            &*input,
        )
        .expect("Prepared initial linear system should solve")
        .to_DVectorType();
    prepared.jac = Some(prepared_jacobian);
    let difference = atom_step
        .iter()
        .zip(prepared_step.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(atom_step.len(), prepared_step.len());
    difference
}

fn full_solve_ab(system: Arc<PreparedAtomSystem>, steps: usize, bandwidth: (usize, usize)) {
    if cfg!(feature = "prepared-atom-allocation-audit") {
        println!("Full-solve A/B skipped: allocator audit perturbs solve timing.");
        return;
    }
    let samples = std::env::var("BVP_PREPARED_ATOM_SOLVE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|samples| *samples >= 3)
        .unwrap_or(3);

    let mut atom_solver = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    let atom_setup_start = Instant::now();
    atom_solver
        .try_eq_generate(None, None)
        .expect("solver-native AtomView full-solve setup should succeed");
    let atom_setup = atom_setup_start.elapsed();

    let mut prepared_solver = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    let prepared_setup = prepare_integrated_prepared_atom_runtime(&mut prepared_solver);
    let mut expr_solver = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::ExprLegacy);
    let expr_solver_setup_start = Instant::now();
    expr_solver
        .try_eq_generate(None, None)
        .expect("ExprLegacy full-solve setup should succeed");
    let expr_solver_setup = expr_solver_setup_start.elapsed();
    let mut trace_reference_solver =
        new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    trace_reference_solver
        .try_eq_generate(None, None)
        .expect("AtomNative trace reference setup should succeed");
    let initial_step_diff = initial_newton_step_parity(Arc::clone(&system), steps, bandwidth);
    assert!(
        initial_step_diff <= 1e-8,
        "first Newton step differs between AtomNative and Prepared: {initial_step_diff:e}"
    );
    println!("Newton-step preflight: initial max_abs_diff={initial_step_diff:.3e}");

    let mut parity_solver = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
    parity_solver
        .try_eq_generate(None, None)
        .expect("AtomNative converged-state parity setup should succeed");
    let parity_solution = parity_solver
        .try_solver_prepared()
        .expect("AtomNative converged-state solve should succeed")
        .expect("AtomNative converged-state fixture should converge");
    let parity_state = Vectors_type_casting(&parity_solution, "Banded".to_string());
    let (native_solution_residual_diff, native_solution_jacobian_diff) = callback_parity_at(
        trace_reference_solver.fun.as_ref(),
        trace_reference_solver
            .jac
            .as_deref_mut()
            .expect("AtomNative trace reference Jacobian should be prepared"),
        prepared_solver.fun.as_ref(),
        prepared_solver
            .jac
            .as_deref_mut()
            .expect("Prepared Jacobian remains installed"),
        &*parity_state,
    );
    println!(
        "Callbacks at AtomNative converged state: residual_max_abs_diff={native_solution_residual_diff:.3e}; Jacobian_max_abs_diff={native_solution_jacobian_diff:.3e}"
    );
    assert!(
        native_solution_residual_diff <= 1e-10 && native_solution_jacobian_diff <= 1e-10,
        "Prepared callback parity failed at the AtomNative converged solution"
    );

    let mut atom_times = Vec::with_capacity(samples);
    let mut prepared_times = Vec::with_capacity(samples);
    let mut expr_times = Vec::with_capacity(samples);
    let mut atom_end_to_end_times = Vec::with_capacity(samples);
    let mut prepared_end_to_end_times = Vec::with_capacity(samples);
    let mut expr_end_to_end_times = Vec::with_capacity(samples);
    let mut atom_solution = None;
    let mut prepared_solution = None;
    let mut expr_solution = None;
    let mut atom_counters = None;
    let mut prepared_counters = None;
    let mut expr_counters = None;
    for sample in 0..samples {
        // Each row is an independent solve from the production initial guess.
        // Reusing NRBVP here would turn later rows into continuation calls and
        // make cumulative telemetry look like per-solve counters.
        let mut atom_sample = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
        atom_sample
            .try_eq_generate(None, None)
            .expect("independent AtomNative sample preparation should succeed");
        let mut prepared_sample =
            new_combustion_solver(steps, BvpSymbolicAssemblyBackend::AtomView);
        prepare_integrated_prepared_atom_runtime(&mut prepared_sample);
        let mut expr_sample = new_combustion_solver(steps, BvpSymbolicAssemblyBackend::ExprLegacy);
        expr_sample
            .try_eq_generate(None, None)
            .expect("independent ExprLegacy sample preparation should succeed");
        let run_atom = |solver: &mut NRBVP| {
            let start = Instant::now();
            let solution = solver
                .try_solver_prepared()
                .expect("solver-native BVP solve should succeed")
                .expect("solver-native combustion BVP should converge");
            let counters = solver.get_statistics().telemetry.counters;
            println!(
                "AtomNative full solve sample: iterations={}, residual_requests={}, jacobian_requests={}, jacobian_recalculations={}, damping_trials={}, damping_rejections={}, factorizations={}, rhs_solves={}, linear_solves={}",
                counters.iterations,
                counters.residual_requests,
                counters.jacobian_requests,
                counters.jacobian_recalculations,
                counters.damping_trials,
                counters.damping_rejections,
                counters.factorizations,
                counters.rhs_solves,
                counters.linear_solves
            );
            (start.elapsed().as_secs_f64() * 1_000.0, solution, counters)
        };
        let run_expr = |solver: &mut NRBVP| {
            let start = Instant::now();
            let solution = solver
                .try_solver_prepared()
                .expect("ExprLegacy BVP solve should succeed")
                .expect("ExprLegacy combustion BVP should converge");
            let counters = solver.get_statistics().telemetry.counters;
            println!(
                "ExprLegacy full solve sample: iterations={}, residual_requests={}, jacobian_requests={}, jacobian_recalculations={}, damping_trials={}, damping_rejections={}, factorizations={}, rhs_solves={}, linear_solves={}",
                counters.iterations,
                counters.residual_requests,
                counters.jacobian_requests,
                counters.jacobian_recalculations,
                counters.damping_trials,
                counters.damping_rejections,
                counters.factorizations,
                counters.rhs_solves,
                counters.linear_solves
            );
            (start.elapsed().as_secs_f64() * 1_000.0, solution, counters)
        };
        let run_prepared = |solver: &mut NRBVP| {
            let start = Instant::now();
            let outcome = solver
                .try_solver_prepared()
                .expect("PreparedAtomSystem BVP solve should succeed");
            let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
            let counters = solver.get_statistics().telemetry.counters;
            let Some(solution) = outcome else {
                println!(
                    "Prepared full solve did not converge: wall_ms={elapsed_ms:.3}; iterations={}, residual_requests={}, jacobian_requests={}, damping_trials={}, damping_rejections={}, linear_solves={}",
                    counters.iterations,
                    counters.residual_requests,
                    counters.jacobian_requests,
                    counters.damping_trials,
                    counters.damping_rejections,
                    counters.linear_solves
                );
                panic!(
                    "PreparedAtomSystem combustion did not converge: iterations={}, residual_requests={}, jacobian_requests={}, damping_trials={}, damping_rejections={}, linear_solves={}",
                    counters.iterations,
                    counters.residual_requests,
                    counters.jacobian_requests,
                    counters.damping_trials,
                    counters.damping_rejections,
                    counters.linear_solves
                );
            };
            (elapsed_ms, solution, counters)
        };
        let mut measure_atom = |solver: &mut NRBVP| {
            let (time, solution, counters) = run_atom(solver);
            atom_times.push(time);
            atom_solution = Some(solution);
            atom_counters = Some(counters);
        };
        let mut measure_prepared = |solver: &mut NRBVP| {
            let (time, solution, counters) = run_prepared(solver);
            prepared_times.push(time);
            prepared_solution = Some(solution);
            prepared_counters = Some(counters);
        };
        let mut measure_expr = |solver: &mut NRBVP| {
            let (time, solution, counters) = run_expr(solver);
            expr_times.push(time);
            expr_solution = Some(solution);
            expr_counters = Some(counters);
        };
        match sample % 3 {
            0 => {
                measure_atom(&mut atom_sample);
                measure_prepared(&mut prepared_sample);
                measure_expr(&mut expr_sample);
            }
            1 => {
                measure_prepared(&mut prepared_sample);
                measure_expr(&mut expr_sample);
                measure_atom(&mut atom_sample);
            }
            _ => {
                measure_expr(&mut expr_sample);
                measure_atom(&mut atom_sample);
                measure_prepared(&mut prepared_sample);
            }
        }
        let atom_state = atom_solution
            .as_ref()
            .expect("AtomNative sample should publish a solution");
        let prepared_state = prepared_solution
            .as_ref()
            .expect("Prepared sample should publish a solution");
        let expr_state = expr_solution
            .as_ref()
            .expect("ExprLegacy sample should publish a solution");
        let sample_solution_diff = atom_state
            .iter()
            .zip(prepared_state.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        let atom_sample_counters = atom_counters.as_ref().expect("AtomNative sample counters");
        let prepared_sample_counters = prepared_counters
            .as_ref()
            .expect("Prepared sample counters");
        assert!(
            sample_solution_diff <= 1e-8,
            "independent sample {sample} solution mismatch {sample_solution_diff:e}"
        );
        let expr_solution_diff = atom_state
            .iter()
            .zip(expr_state.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            expr_solution_diff <= 1e-8,
            "independent sample {sample} ExprLegacy solution mismatch {expr_solution_diff:e}"
        );
        assert_eq!(
            atom_sample_counters.iterations, prepared_sample_counters.iterations,
            "independent sample {sample} iteration count mismatch"
        );
        assert_eq!(
            atom_sample_counters.residual_requests, prepared_sample_counters.residual_requests,
            "independent sample {sample} residual count mismatch"
        );
        assert_eq!(
            atom_sample_counters.jacobian_requests, prepared_sample_counters.jacobian_requests,
            "independent sample {sample} Jacobian count mismatch"
        );
        assert_eq!(
            atom_sample_counters.linear_solves, prepared_sample_counters.linear_solves,
            "independent sample {sample} linear solve count mismatch"
        );
        let expr_sample_counters = expr_counters.as_ref().expect("ExprLegacy sample counters");
        assert_eq!(
            atom_sample_counters.iterations,
            expr_sample_counters.iterations
        );
        assert_eq!(
            atom_sample_counters.residual_requests,
            expr_sample_counters.residual_requests
        );
        assert_eq!(
            atom_sample_counters.jacobian_requests,
            expr_sample_counters.jacobian_requests
        );
        assert_eq!(
            atom_sample_counters.linear_solves,
            expr_sample_counters.linear_solves
        );
        let atom_work = (
            atom_sample_counters.jacobian_recalculations,
            atom_sample_counters.damping_trials,
            atom_sample_counters.damping_rejections,
            atom_sample_counters.factorizations,
            atom_sample_counters.rhs_solves,
        );
        let prepared_work = (
            prepared_sample_counters.jacobian_recalculations,
            prepared_sample_counters.damping_trials,
            prepared_sample_counters.damping_rejections,
            prepared_sample_counters.factorizations,
            prepared_sample_counters.rhs_solves,
        );
        let expr_work = (
            expr_sample_counters.jacobian_recalculations,
            expr_sample_counters.damping_trials,
            expr_sample_counters.damping_rejections,
            expr_sample_counters.factorizations,
            expr_sample_counters.rhs_solves,
        );
        assert_eq!(
            atom_work, prepared_work,
            "Atom/Prepared detailed work counters differ"
        );
        assert_eq!(
            atom_work, expr_work,
            "Atom/Expr detailed work counters differ"
        );
        println!(
            "independent sample {sample}: per-solve counters={atom_sample_counters:?}; Atom/Prepared_diff={sample_solution_diff:.3e}; Atom/Expr_diff={expr_solution_diff:.3e}",
        );
    }

    // Measure the public cold lifecycle separately: try_solve regenerates the
    // selected callback runtime before entering Newton, unlike the warm
    // try_solver_prepared series above.
    for sample in 0..samples {
        let mut solutions: [Option<DVector<f64>>; 3] = [None, None, None];
        let mut counters: [Option<
            RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
        >; 3] = [None, None, None];
        for offset in 0..3 {
            let route_index = (sample + offset) % 3;
            let (name, assembly, runtime) = combustion_route(route_index);
            let mut solver = new_combustion_solver(steps, assembly);
            solver.set_callback_runtime(runtime);
            let start = Instant::now();
            let solution = solver
                .try_solve()
                .unwrap_or_else(|error| panic!("{name} end-to-end solve failed: {error:?}"))
                .unwrap_or_else(|| panic!("{name} end-to-end combustion did not converge"));
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            match route_index {
                0 => atom_end_to_end_times.push(elapsed),
                1 => prepared_end_to_end_times.push(elapsed),
                _ => expr_end_to_end_times.push(elapsed),
            }
            solutions[route_index] = Some(solution);
            counters[route_index] = Some(solver.get_statistics().telemetry.counters.clone());
        }
        let reference = solutions[0].as_ref().expect("AtomNative result exists");
        for route_index in 1..3 {
            let candidate = solutions[route_index]
                .as_ref()
                .expect("comparison route result exists");
            let max_diff = reference
                .iter()
                .zip(candidate.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff <= 1e-8,
                "end-to-end route {route_index} solution mismatch: {max_diff:e}"
            );
            assert_eq!(
                counters[0].as_ref().expect("AtomNative counters"),
                counters[route_index]
                    .as_ref()
                    .expect("comparison route counters"),
                "end-to-end route {route_index} changed discrete solver work"
            );
        }
    }

    same_instance_combustion_replay(steps);

    let atom_solution = atom_solution.expect("at least one AtomNative solve is required");
    let prepared_solution = prepared_solution.expect("at least one Prepared solve is required");
    let solution_diff = atom_solution
        .iter()
        .zip(prepared_solution.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(atom_solution.len(), prepared_solution.len());
    assert!(
        solution_diff <= 1e-8,
        "full BVP solution mismatch {solution_diff:e}"
    );
    let expr_solution = expr_solution.expect("at least one ExprLegacy solve is required");
    let expr_solution_diff = atom_solution
        .iter()
        .zip(expr_solution.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(atom_solution.len(), expr_solution.len());
    assert!(
        expr_solution_diff <= 1e-8,
        "Atom/Expr solution diff {expr_solution_diff:e}"
    );
    let atom_counters = atom_counters.expect("AtomNative counter snapshot is required");
    let prepared_counters = prepared_counters.expect("Prepared counter snapshot is required");
    let expr_counters = expr_counters.expect("ExprLegacy counter snapshot is required");
    assert_eq!(atom_counters.iterations, prepared_counters.iterations);
    assert_eq!(
        atom_counters.residual_requests,
        prepared_counters.residual_requests
    );
    assert_eq!(
        atom_counters.jacobian_requests,
        prepared_counters.jacobian_requests
    );
    assert_eq!(atom_counters.linear_solves, prepared_counters.linear_solves);
    assert_eq!(atom_counters.iterations, expr_counters.iterations);
    assert_eq!(
        atom_counters.residual_requests,
        expr_counters.residual_requests
    );
    assert_eq!(
        atom_counters.jacobian_requests,
        expr_counters.jacobian_requests
    );
    assert_eq!(atom_counters.linear_solves, expr_counters.linear_solves);
    assert_eq!(
        (
            atom_counters.jacobian_recalculations,
            atom_counters.damping_trials,
            atom_counters.damping_rejections,
            atom_counters.factorizations,
            atom_counters.rhs_solves,
        ),
        (
            prepared_counters.jacobian_recalculations,
            prepared_counters.damping_trials,
            prepared_counters.damping_rejections,
            prepared_counters.factorizations,
            prepared_counters.rhs_solves,
        )
    );
    assert_eq!(
        (
            atom_counters.jacobian_recalculations,
            atom_counters.damping_trials,
            atom_counters.damping_rejections,
            atom_counters.factorizations,
            atom_counters.rhs_solves,
        ),
        (
            expr_counters.jacobian_recalculations,
            expr_counters.damping_trials,
            expr_counters.damping_rejections,
            expr_counters.factorizations,
            expr_counters.rhs_solves,
        )
    );

    let (atom_median, atom_mad) = median_and_mad(&atom_times);
    let (prepared_median, prepared_mad) = median_and_mad(&prepared_times);
    let (expr_median, expr_mad) = median_and_mad(&expr_times);
    let (atom_end_to_end_median, atom_end_to_end_mad) = median_and_mad(&atom_end_to_end_times);
    let (prepared_end_to_end_median, prepared_end_to_end_mad) =
        median_and_mad(&prepared_end_to_end_times);
    let (expr_end_to_end_median, expr_end_to_end_mad) = median_and_mad(&expr_end_to_end_times);
    println!(
        "[BVP_Damp independent full-solve A/B] combustion n_steps={steps}; dimension={}; bandwidth={bandwidth:?}; samples={samples}; solver telemetry=Counters; fresh solver per sample; callbacks prepared before timing; frontends=AtomView,ExprLegacy; backend=Banded; nonlinear settings identical",
        system.input_len()
    );
    println!(
        "setup_ms: AtomNative={:.3}, Prepared_adapter_including_native_generation={:.3}, ExprLegacy={:.3}; full_solve_ms median/MAD: AtomNative={atom_median:.3}/{atom_mad:.3}, PreparedAtom={prepared_median:.3}/{prepared_mad:.3}, ExprLegacy={expr_median:.3}/{expr_mad:.3}",
        atom_setup.as_secs_f64() * 1e3,
        prepared_setup.as_secs_f64() * 1e3,
        expr_solver_setup.as_secs_f64() * 1e3
    );
    println!(
        "normal_try_solve_end_to_end_ms median/MAD: AtomNative={atom_end_to_end_median:.3}/{atom_end_to_end_mad:.3}, PreparedAtom={prepared_end_to_end_median:.3}/{prepared_end_to_end_mad:.3}, ExprLegacy={expr_end_to_end_median:.3}/{expr_end_to_end_mad:.3}; includes callback preparation and solve"
    );
    println!(
        "parity: Atom/Prepared max_solution_diff={solution_diff:.3e}; Atom/Expr max_solution_diff={expr_solution_diff:.3e}; complete per-solve counters={atom_counters:?}"
    );
}

fn main() {
    let steps = std::env::var("BVP_PREPARED_ATOM_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|steps| *steps >= 2)
        .unwrap_or(1_000);
    let samples = std::env::var("BVP_PREPARED_ATOM_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|samples| *samples >= 3)
        .unwrap_or(21);
    let callbacks_per_sample = std::env::var("BVP_PREPARED_ATOM_CALLBACKS_PER_SAMPLE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|callbacks| *callbacks >= 1)
        .unwrap_or(3);
    let (equations, variable_names, boundary_conditions) = combustion_equations();
    let bounds = HashMap::from([
        ("Teta".to_string(), (0.0, 10.0)),
        ("q".to_string(), (-1e20, 1e20)),
        ("C0".to_string(), (0.0, 1.5)),
        ("J0".to_string(), (-1e2, 1e2)),
        ("C1".to_string(), (0.0, 1.5)),
        ("J1".to_string(), (-1e2, 1e2)),
    ]);
    let tolerances = HashMap::from([
        ("Teta".to_string(), 1e-5),
        ("q".to_string(), 1e-5),
        ("C0".to_string(), 1e-5),
        ("J0".to_string(), 1e-5),
        ("C1".to_string(), 1e-5),
        ("J1".to_string(), 1e-5),
    ]);

    let expr_setup_start = Instant::now();
    let mut expr_jacobian = Jacobian::new();
    expr_jacobian.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    expr_jacobian.set_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Off);
    expr_jacobian.set_lambdify_execution_policy(BvpLambdifyExecutionPolicy::Sequential);
    let mut expr_bundle = expr_jacobian
        .try_generate_sparse_solver_bundle_with_backend_selection(
            equations.clone(),
            variable_names.clone(),
            "x".to_string(),
            None,
            0.0,
            None,
            Some(steps),
            Some(1.0 / steps as f64),
            None,
            boundary_conditions.clone(),
            Some(bounds.clone()),
            Some(tolerances.clone()),
            "forward".to_string(),
            "Banded".to_string(),
            None,
            BackendSelectionPolicy::LambdifyOnly,
            None,
        )
        .expect("ExprLegacy BVP_Damp/Banded generation should succeed");
    let expr_setup = expr_setup_start.elapsed();

    // The policy mirrors the existing BVP_Damp AtomView/Banded Lambdify
    // route; only the explicit PreparedAtomSystem row is experimental.
    let mut jacobian = Jacobian::new();
    jacobian.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);
    jacobian.set_lambdify_telemetry_mode(BvpLambdifyTelemetryMode::Off);
    jacobian.set_lambdify_execution_policy(BvpLambdifyExecutionPolicy::Sequential);

    let production_setup_start = Instant::now();
    let production_alloc_start = allocation_snapshot();
    let production_live_start = live_bytes();
    reset_peak_live_bytes();
    let mut bundle = jacobian
        .try_generate_sparse_solver_bundle_with_backend_selection(
            equations,
            variable_names.clone(),
            "x".to_string(),
            None,
            0.0,
            None,
            Some(steps),
            Some(1.0 / steps as f64),
            None,
            boundary_conditions,
            Some(bounds),
            Some(tolerances),
            "forward".to_string(),
            "Banded".to_string(),
            None,
            BackendSelectionPolicy::LambdifyOnly,
            None,
        )
        .expect("production BVP_Damp AtomView/Banded generation should succeed");
    let production_setup = production_setup_start.elapsed();
    let production_alloc = allocation_delta(production_alloc_start, allocation_snapshot());
    let production_retained_bytes = live_bytes().saturating_sub(production_live_start);
    let production_peak_bytes = peak_live_bytes().saturating_sub(production_live_start);
    assert_eq!(bundle.effective_backend(), SelectedBackendKind::Lambdify);
    assert_eq!(bundle.jacobian_shape().0, bundle.jacobian_shape().1);

    let selected = bundle.execution.selected();
    let atom_problem = selected
        .prepared_problem
        .atom_discretized_system
        .as_ref()
        .expect("AtomView route must retain its native discretized atoms");
    let residual_atoms = atom_problem.vector_of_functions.as_slice();
    let input_names = atom_problem.variable_string.as_slice();
    let bandwidth = bundle
        .bandwidth
        .expect("Banded combustion route must publish scalar bandwidth");

    let prepared_alloc_start = allocation_snapshot();
    let live_before_prepared = live_bytes();
    reset_peak_live_bytes();
    let prepared_start = Instant::now();
    let prepared = PreparedAtomSystem::try_from_atoms(
        residual_atoms,
        input_names,
        Some(bandwidth),
        FunctionMap::new(),
    )
    .expect("PreparedAtomSystem should prepare from the production Atom residuals");
    let prepared_setup = prepared_start.elapsed();
    let prepared_alloc = allocation_delta(prepared_alloc_start, allocation_snapshot());
    let prepared_retained_bytes = live_bytes().saturating_sub(live_before_prepared);
    let prepared_peak_bytes = peak_live_bytes().saturating_sub(live_before_prepared);

    assert_eq!(prepared.input_len(), input_names.len());
    assert_eq!(prepared.residual_len(), residual_atoms.len());
    assert_eq!(prepared.jacobian_len(), bundle.sparse_structure.nnz());

    let dimension = prepared.input_len();
    let args = vec![0.99; dimension];
    let mut residual_values = vec![0.0; prepared.residual_len()];
    let mut jacobian_values = vec![0.0; prepared.jacobian_len()];
    let mut scratch = PreparedAtomSystemScratch::default();
    let mut prepared_assembly = BandedAssembly::zeros(dimension, bandwidth.0, bandwidth.1)
        .expect("valid combustion banded shape");

    let input_vector = Vectors_type_casting(&DVector::from_vec(args.clone()), "Banded".to_string());
    let production_residual = bundle
        .residual_function
        .as_ref()
        .expect("Lambdify bundle must include residual callback")
        .call(1.0, &*input_vector)
        .to_DVectorType();
    let production_jacobian_callback = bundle
        .jacobian_function
        .as_mut()
        .expect("Lambdify bundle must include Jacobian callback");
    let production_matrix = production_jacobian_callback.call(1.0, &*input_vector);
    let production_banded = production_matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("Banded route must return native compact BandedMatrixType");
    let expr_residual = expr_bundle
        .residual_function
        .as_ref()
        .expect("ExprLegacy bundle must include residual callback")
        .call(1.0, &*input_vector)
        .to_DVectorType();
    let expr_jacobian_callback = expr_bundle
        .jacobian_function
        .as_mut()
        .expect("ExprLegacy bundle must include Jacobian callback");
    let expr_matrix = expr_jacobian_callback.call(1.0, &*input_vector);
    let expr_banded = expr_matrix
        .as_any()
        .downcast_ref::<BandedMatrixType>()
        .expect("ExprLegacy Banded route must return compact BandedMatrixType");

    prepared
        .residual_and_jacobian_into(
            &args,
            &mut residual_values,
            &mut jacobian_values,
            &mut scratch,
        )
        .expect("prepared residual/Jacobian callback should evaluate");
    let residual_max_diff = production_residual
        .iter()
        .zip(&residual_values)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(production_residual.len(), residual_values.len());
    assert!(
        residual_max_diff <= 1e-10,
        "residual mismatch {residual_max_diff:e}"
    );

    for (entry, value) in prepared.jacobian_entries().zip(&jacobian_values) {
        prepared_assembly
            .set(entry.row, entry.col, *value)
            .expect("prepared Jacobian entry must fit the published band");
    }
    assert_eq!(
        production_banded.assembly.diagonals().len(),
        prepared_assembly.diagonals().len(),
        "the two routes must use the same compact band storage"
    );
    let jacobian_max_diff = production_banded
        .assembly
        .diagonals()
        .iter()
        .zip(prepared_assembly.diagonals())
        .flat_map(|(lhs, rhs)| lhs.iter().zip(rhs))
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        jacobian_max_diff <= 1e-10,
        "Jacobian mismatch {jacobian_max_diff:e}"
    );
    let expr_residual_max_diff = expr_residual
        .iter()
        .zip(&production_residual)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    let expr_jacobian_max_diff = expr_banded
        .assembly
        .diagonals()
        .iter()
        .zip(production_banded.assembly.diagonals())
        .flat_map(|(lhs, rhs)| lhs.iter().zip(rhs))
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        expr_residual_max_diff <= 1e-10 && expr_jacobian_max_diff <= 1e-10,
        "ExprLegacy/AtomView callback parity failed: residual={expr_residual_max_diff:e}, Jacobian={expr_jacobian_max_diff:e}"
    );

    let total_warm_callbacks = samples * callbacks_per_sample;
    let mut production_sample_ms = Vec::with_capacity(samples);
    let mut expr_sample_ms = Vec::with_capacity(samples);
    let mut prepared_sample_ms = Vec::with_capacity(samples);
    let mut production_batch = || {
        for _ in 0..callbacks_per_sample {
            let residual = bundle
                .residual_function
                .as_ref()
                .unwrap()
                .call(1.0, &*input_vector);
            let matrix = bundle
                .jacobian_function
                .as_mut()
                .unwrap()
                .call(1.0, &*input_vector);
            black_box((residual, matrix));
        }
    };
    let mut prepared_batch = || {
        for _ in 0..callbacks_per_sample {
            prepared
                .residual_and_jacobian_into(
                    black_box(&args),
                    &mut residual_values,
                    &mut jacobian_values,
                    &mut scratch,
                )
                .expect("prepared warm callback should not fail");
            for (entry, value) in prepared.jacobian_entries().zip(&jacobian_values) {
                prepared_assembly
                    .set(entry.row, entry.col, *value)
                    .expect("prepared banded scatter should remain valid");
            }
            black_box((&residual_values, &prepared_assembly));
        }
    };
    let mut expr_batch = || {
        for _ in 0..callbacks_per_sample {
            let residual = expr_bundle
                .residual_function
                .as_ref()
                .unwrap()
                .call(1.0, &*input_vector);
            let matrix = expr_bundle
                .jacobian_function
                .as_mut()
                .unwrap()
                .call(1.0, &*input_vector);
            black_box((residual, matrix));
        }
    };
    for sample in 0..samples {
        // Rotate order to reduce systematic cache/thermal bias across routes.
        let mut measure = |route: usize| {
            let started = Instant::now();
            match route {
                0 => production_batch(),
                1 => expr_batch(),
                _ => prepared_batch(),
            }
            let elapsed_ms =
                started.elapsed().as_secs_f64() * 1_000.0 / callbacks_per_sample as f64;
            match route {
                0 => production_sample_ms.push(elapsed_ms),
                1 => expr_sample_ms.push(elapsed_ms),
                _ => prepared_sample_ms.push(elapsed_ms),
            }
        };
        match sample % 3 {
            0 => {
                measure(0);
                measure(1);
                measure(2);
            }
            1 => {
                measure(1);
                measure(2);
                measure(0);
            }
            _ => {
                measure(2);
                measure(0);
                measure(1);
            }
        }
    }
    let (production_warm_alloc, prepared_warm_alloc) =
        if cfg!(feature = "prepared-atom-allocation-audit") {
            let before = allocation_snapshot();
            for _ in 0..samples {
                production_batch();
            }
            let production_alloc = allocation_delta(before, allocation_snapshot());
            let before = allocation_snapshot();
            for _ in 0..samples {
                prepared_batch();
            }
            (
                production_alloc,
                allocation_delta(before, allocation_snapshot()),
            )
        } else {
            (AllocationSnapshot::default(), AllocationSnapshot::default())
        };
    drop((production_batch, prepared_batch, expr_batch));
    let (production_warm_median, production_warm_mad) = median_and_mad(&production_sample_ms);
    let (expr_warm_median, expr_warm_mad) = median_and_mad(&expr_sample_ms);
    let (prepared_warm_median, prepared_warm_mad) = median_and_mad(&prepared_sample_ms);

    let scratch_capacity = scratch.capacity();
    let rows = [
        vec![
            "production AtomView/Banded".to_string(),
            format!("{:.3}", production_setup.as_secs_f64() * 1e3),
            format!("{production_warm_median:.3}/{production_warm_mad:.3}"),
            audit_value(format!(
                "{:.2}",
                production_warm_alloc.allocations as f64 / total_warm_callbacks as f64
            )),
            audit_value(format!(
                "{:.1}",
                production_warm_alloc.allocated_bytes as f64 / total_warm_callbacks as f64
            )),
            audit_value(production_retained_bytes),
            audit_value(production_peak_bytes),
        ],
        vec![
            "PreparedAtomSystem + band scatter".to_string(),
            format!("{:.3}", prepared_setup.as_secs_f64() * 1e3),
            format!("{prepared_warm_median:.3}/{prepared_warm_mad:.3}"),
            audit_value(format!(
                "{:.2}",
                prepared_warm_alloc.allocations as f64 / total_warm_callbacks as f64
            )),
            audit_value(format!(
                "{:.1}",
                prepared_warm_alloc.allocated_bytes as f64 / total_warm_callbacks as f64
            )),
            audit_value(prepared_retained_bytes),
            audit_value(prepared_peak_bytes),
        ],
        vec![
            "ExprLegacy/Banded Lambdify".to_string(),
            format!("{:.3}", expr_setup.as_secs_f64() * 1e3),
            format!("{expr_warm_median:.3}/{expr_warm_mad:.3}"),
            "n/a".to_string(),
            "n/a".to_string(),
            "n/a".to_string(),
            "n/a".to_string(),
        ],
    ];
    let mut table = Builder::from_iter(rows).build();
    table.with(Style::modern_rounded());

    println!(
        "[BVP_Damp callback A/B] fixture=combustion; n_steps={steps}; dimension={dimension}; bandwidth={bandwidth:?}; samples={samples}; callbacks_per_sample={callbacks_per_sample}; frontends=AtomView,ExprLegacy; backend=Banded; execution=Sequential; callback_telemetry=Off; allocator_audit={}",
        cfg!(feature = "prepared-atom-allocation-audit")
    );
    println!(
        "Correctness: Prepared/Atom residual_max_abs_diff={residual_max_diff:.3e}; Prepared/Atom Jacobian_max_abs_diff={jacobian_max_diff:.3e}; Expr/Atom residual_max_abs_diff={expr_residual_max_diff:.3e}; Expr/Atom Jacobian_max_abs_diff={expr_jacobian_max_diff:.3e}; residuals={}; jacobian_nnz={}; no dense matrix materialized",
        prepared.residual_len(),
        prepared.jacobian_len()
    );
    if cfg!(feature = "prepared-atom-allocation-audit") {
        println!(
            "Preparation allocation diagnostics: production_setup_allocs={} allocated_bytes={} deallocs={} deallocated_bytes={}; prepared_setup_allocs={} allocated_bytes={} deallocs={} deallocated_bytes={}",
            production_alloc.allocations,
            production_alloc.allocated_bytes,
            production_alloc.deallocations,
            production_alloc.deallocated_bytes,
            prepared_alloc.allocations,
            prepared_alloc.allocated_bytes,
            prepared_alloc.deallocations,
            prepared_alloc.deallocated_bytes
        );
    } else {
        println!(
            "Allocation/memory counters disabled; use --features prepared-atom-allocation-audit in a separate run. Timing above is uninstrumented."
        );
    }
    if cfg!(feature = "prepared-atom-allocation-audit") {
        println!(
            "Warm callback allocation diagnostics: production_allocs/eval={:.2} allocated_bytes/eval={:.1}; prepared_allocs/eval={:.2} allocated_bytes/eval={:.1}; prepared_scratch_capacity={scratch_capacity}",
            production_warm_alloc.allocations as f64 / total_warm_callbacks as f64,
            production_warm_alloc.allocated_bytes as f64 / total_warm_callbacks as f64,
            prepared_warm_alloc.allocations as f64 / total_warm_callbacks as f64,
            prepared_warm_alloc.allocated_bytes as f64 / total_warm_callbacks as f64
        );
    } else {
        println!(
            "Prepared scratch capacity proxy={scratch_capacity}; allocator counters unavailable in this timing run."
        );
    }
    println!("{table}");
    println!(
        "Table latency is median/MAD in ms per warm residual+Jacobian callback batch. It is suitable for comparison only when allocator_audit=false. With allocator_audit=true, timing is perturbed and must be ignored. Retained/peak are incremental allocator-live deltas around each preparation while its owner remains alive; production includes callbacks+symbolic bridge, Prepared is additional runtime on the same Atom discretization. These are allocator-observed bytes, not RSS."
    );
    full_solve_ab(Arc::new(prepared), steps, bandwidth);
    if cfg!(feature = "prepared-atom-allocation-audit") {
        bratu_integrated_callback_allocation_audit(
            std::env::var("BVP_PREPARED_ATOM_BRATU_STEPS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value >= 8)
                .unwrap_or(1_024),
        );
    } else {
        bratu_full_solve_ab(samples.max(9));
    }
}

fn bratu_solver(
    n_steps: usize,
    assembly: BvpSymbolicAssemblyBackend,
    callback_runtime: RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime,
) -> NRBVP {
    let options = DampedSolverOptions::sparse_damped()
        .with_banded_lambdify()
        .with_symbolic_assembly_backend(assembly)
        .with_abs_tolerance(1e-8)
        .with_max_iterations(60)
        .with_rel_tolerance(HashMap::from([
            ("y".to_string(), 1e-5),
            ("z".to_string(), 1e-5),
        ]))
        .with_bounds(HashMap::from([
            ("y".to_string(), (-1.0, 5.0)),
            ("z".to_string(), (-20.0, 20.0)),
        ]));
    let config = options
        .generated_backend_config
        .clone()
        .with_lambdify_execution_policy(BvpLambdifyExecutionPolicy::Sequential);
    let options = options.with_generated_backend_config(config);
    let equations = vec![
        Expr::parse_expression("z"),
        Expr::parse_expression("-2*exp(y)"),
    ];
    let initial_guess = DMatrix::from_fn(2, n_steps, |row, _| if row == 0 { 4.0 } else { 0.0 });
    let mut solver = NRBVP::new_with_options(
        equations,
        initial_guess,
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([("y".to_string(), vec![(0, 0.0), (1, 0.0)])]),
        0.0,
        1.0,
        n_steps,
        options,
    );
    solver.set_callback_runtime(callback_runtime);
    solver.dont_save_log(true);
    solver
}

fn bratu_integrated_callback_allocation_audit(n_steps: usize) {
    let routes = [
        ("AtomNative", BvpSymbolicAssemblyBackend::AtomView, false),
        ("PreparedAtom", BvpSymbolicAssemblyBackend::AtomView, true),
        ("ExprLegacy", BvpSymbolicAssemblyBackend::ExprLegacy, false),
    ];
    println!(
        "[BVP_Damp integrated callback allocation audit] fixture=Bratu; n_steps={n_steps}; counts wrap the installed production Fun/Jac callbacks; timing is omitted"
    );
    for (name, assembly, use_prepared) in routes {
        let runtime = if use_prepared {
            RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::PreparedAtom
        } else {
            RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::SolverNative
        };
        let mut solver = bratu_solver(n_steps, assembly, runtime);
        solver
            .try_eq_generate(None, None)
            .unwrap_or_else(|error| panic!("{name} allocation setup failed: {error:?}"));
        let input = DVector::from_fn(
            2 * n_steps,
            |index, _| {
                if index % 2 == 0 { 4.0 } else { 0.0 }
            },
        );
        let input = Vectors_type_casting(&input, "Banded".to_string());
        let before = allocation_snapshot();
        for _ in 0..5 {
            let residual = solver
                .fun
                .as_ref()
                .try_call(0.0, &*input)
                .unwrap_or_else(|error| panic!("{name} residual audit failed: {error}"));
            let jacobian = solver
                .jac
                .as_deref_mut()
                .expect("solver Jacobian callback exists")
                .try_call(0.0, &*input)
                .unwrap_or_else(|error| panic!("{name} Jacobian audit failed: {error}"));
            black_box((residual, jacobian));
        }
        let delta = allocation_delta(before, allocation_snapshot());
        println!(
            "{name}: allocations/pair={:.2}; allocated_bytes/pair={:.1}; pairs=5",
            delta.allocations as f64 / 5.0,
            delta.allocated_bytes as f64 / 5.0
        );
    }
}

fn bratu_full_solve_ab(samples: usize) {
    let n_steps = std::env::var("BVP_PREPARED_ATOM_BRATU_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(1_024);
    let routes = [
        ("AtomNative", BvpSymbolicAssemblyBackend::AtomView, false),
        ("PreparedAtom", BvpSymbolicAssemblyBackend::AtomView, true),
        ("ExprLegacy", BvpSymbolicAssemblyBackend::ExprLegacy, false),
    ];
    let mut setup = [
        Vec::with_capacity(samples),
        Vec::with_capacity(samples),
        Vec::with_capacity(samples),
    ];
    let mut solve = [
        Vec::with_capacity(samples),
        Vec::with_capacity(samples),
        Vec::with_capacity(samples),
    ];
    let mut end_to_end = [
        Vec::with_capacity(samples),
        Vec::with_capacity(samples),
        Vec::with_capacity(samples),
    ];
    let mut reference_solution: Option<DMatrix<f64>> = None;
    let mut reference_counters = None;
    let mut max_solution_diff = [0.0_f64; 3];

    for sample in 0..samples {
        let mut sample_solutions = Vec::with_capacity(3);
        let mut sample_counters = Vec::with_capacity(3);
        for offset in 0..3 {
            let route_index = (sample + offset) % routes.len();
            let (name, assembly, is_prepared) = routes[route_index];
            let callback_runtime = if is_prepared {
                RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::PreparedAtom
            } else {
                RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::SolverNative
            };
            let mut solver = bratu_solver(n_steps, assembly, callback_runtime);
            let prepare_start = Instant::now();
            solver
                .try_eq_generate(None, None)
                .unwrap_or_else(|error| panic!("Bratu {name} preparation failed: {error:?}"));
            setup[route_index].push(prepare_start.elapsed().as_secs_f64() * 1e3);
            let solve_start = Instant::now();
            solver
                .try_solver_prepared()
                .unwrap_or_else(|error| panic!("Bratu {name} solve failed: {error:?}"))
                .unwrap_or_else(|| panic!("Bratu {name} did not converge"));
            solve[route_index].push(solve_start.elapsed().as_secs_f64() * 1e3);
            let counters = solver.get_statistics().telemetry.counters;
            let solution_matrix = solver
                .get_result()
                .expect("Bratu route publishes reconstructed solution")
                .clone();
            sample_solutions.push((route_index, solution_matrix));
            sample_counters.push((route_index, counters));
        }
        let atom = &sample_solutions
            .iter()
            .find(|(index, _)| *index == 0)
            .expect("AtomNative Bratu result exists")
            .1;
        for (index, candidate) in &sample_solutions {
            let difference = atom
                .iter()
                .zip(candidate.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max);
            max_solution_diff[*index] = max_solution_diff[*index].max(difference);
            assert!(
                difference <= 1e-9,
                "Bratu route {index} solution diff={difference:e}"
            );
        }
        let atom_counters = &sample_counters
            .iter()
            .find(|(index, _)| *index == 0)
            .expect("AtomNative Bratu counters exist")
            .1;
        for (index, counters) in &sample_counters {
            assert_eq!(
                atom_counters, counters,
                "Bratu route {index} changed discrete solver work"
            );
        }

        let mut end_to_end_solutions: [Option<DVector<f64>>; 3] = [None, None, None];
        let mut end_to_end_counters: [Option<
            RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryCounters,
        >; 3] = [None, None, None];
        for offset in 0..3 {
            let route_index = (sample + offset) % routes.len();
            let (name, assembly, is_prepared) = routes[route_index];
            let callback_runtime = if is_prepared {
                RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::PreparedAtom
            } else {
                RustedSciThe::numerical::BVP_Damp::DampedCallbackRuntime::SolverNative
            };
            let mut solver = bratu_solver(n_steps, assembly, callback_runtime);
            let start = Instant::now();
            let solution = solver
                .try_solve()
                .unwrap_or_else(|error| panic!("Bratu {name} end-to-end solve failed: {error:?}"))
                .unwrap_or_else(|| panic!("Bratu {name} end-to-end solve did not converge"));
            end_to_end[route_index].push(start.elapsed().as_secs_f64() * 1e3);
            end_to_end_solutions[route_index] = Some(solution);
            end_to_end_counters[route_index] =
                Some(solver.get_statistics().telemetry.counters.clone());
        }
        let end_to_end_reference = end_to_end_solutions[0]
            .as_ref()
            .expect("AtomNative end-to-end result exists");
        for route_index in 1..3 {
            let candidate = end_to_end_solutions[route_index]
                .as_ref()
                .expect("Bratu end-to-end comparison result exists");
            let max_diff = end_to_end_reference
                .iter()
                .zip(candidate.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff <= 1e-9,
                "Bratu end-to-end route {route_index} solution diff={max_diff:e}"
            );
            assert_eq!(
                end_to_end_counters[0]
                    .as_ref()
                    .expect("AtomNative end-to-end counters"),
                end_to_end_counters[route_index]
                    .as_ref()
                    .expect("Bratu end-to-end counters"),
                "Bratu end-to-end route {route_index} changed discrete work"
            );
        }
        reference_solution = Some(atom.clone());
        reference_counters = Some(atom_counters.clone());
    }

    let rows = routes.iter().enumerate().map(|(index, (name, _, _))| {
        let (setup_median, setup_mad) = median_and_mad(&setup[index]);
        let (solve_median, solve_mad) = median_and_mad(&solve[index]);
        vec![
            name.to_string(),
            format!("{setup_median:.3}/{setup_mad:.3}"),
            format!("{solve_median:.3}/{solve_mad:.3}"),
        ]
    });
    let mut table = Builder::from_iter(rows).build();
    table.with(Style::modern_rounded());
    println!(
        "[BVP_Damp integrated callback-runtime A/B] fixture=Bratu; n_steps={n_steps}; samples={samples}; setup and solve separate; rows are median/MAD ms"
    );
    println!("{table}");
    println!(
        "Bratu correctness: max |solution-AtomNative| = {max_solution_diff:?}; complete discrete counter parity = {:?}; reference shape={:?}",
        reference_counters.expect("samples are non-zero"),
        reference_solution.expect("samples are non-zero").shape()
    );
    let e2e_rows = routes.iter().enumerate().map(|(index, (name, _, _))| {
        let (median, mad) = median_and_mad(&end_to_end[index]);
        vec![name.to_string(), format!("{median:.3}/{mad:.3}")]
    });
    let mut e2e_table = Builder::from_iter(e2e_rows).build();
    e2e_table.with(Style::modern_rounded());
    println!(
        "[BVP_Damp public try_solve end-to-end] fixture=Bratu; n_steps={n_steps}; includes callback preparation and solve; median/MAD ms"
    );
    println!("{e2e_table}");
}
