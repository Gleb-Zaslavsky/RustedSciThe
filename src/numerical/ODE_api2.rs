//! Modern universal IVP facade.
//!
//! This module is the user-facing entry point for selecting IVP solvers without
//! manually navigating the solver-specific modules. Unlike the historical
//! `HashMap<String, SolverParam>` facade, this implementation routes through the
//! current typed solver options of `BE`, `BDF`, and `Radau`, while still keeping
//! thin legacy wrappers for older call-sites that pass string-keyed parameter
//! bags.

use crate::Utils::plots::plots;
use crate::numerical::BDF::BDF_api::{BdfSolverOptions, ODEsolver as BdfOdeSolver};
use crate::numerical::BE::{BE, BeSolverOptions};
use crate::numerical::NonStiff_api::nonstiffODE;
use crate::numerical::Radau::Radau_main::{Radau, RadauOrder, RadauSolverOptions, RadauStatistics};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::IvpBackendError;
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, IvpBackendStatistics, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum SolverType {
    NonStiff(String),
    Radau(RadauOrder),
    BDF,
    BackwardEuler,
}

#[derive(Clone, Debug)]
pub enum SolverParam {
    Float(f64),
    Int(usize),
    Bool(bool),
    OptionalFloat(Option<f64>),
    OptionalInt(Option<usize>),
    OptionalMatrix(Option<DMatrix<f64>>),
}

#[derive(Clone, Debug, Default)]
pub struct UniversalIvpStatistics {
    pub method_label: String,
    pub backend_label: String,
    pub setup_ms_total: f64,
    pub solve_ms_total: f64,
    pub residual_calls: usize,
    pub residual_ms_total: f64,
    pub jacobian_calls: usize,
    pub jacobian_ms_total: f64,
    pub step_calls: usize,
    pub nonlinear_solve_calls: usize,
    pub nonlinear_iterations_total: usize,
    pub bdf_nfev_total: usize,
    pub bdf_njev_total: usize,
    pub bdf_nlu_total: usize,
}

impl UniversalIvpStatistics {
    pub fn avg_residual_ms(&self) -> Option<f64> {
        (self.residual_calls > 0).then(|| self.residual_ms_total / self.residual_calls as f64)
    }

    pub fn avg_jacobian_ms(&self) -> Option<f64> {
        (self.jacobian_calls > 0).then(|| self.jacobian_ms_total / self.jacobian_calls as f64)
    }

    pub fn avg_nonlinear_iterations(&self) -> Option<f64> {
        (self.nonlinear_solve_calls > 0)
            .then(|| self.nonlinear_iterations_total as f64 / self.nonlinear_solve_calls as f64)
    }

    pub fn table_report(&self) -> String {
        format!(
            "method={} backend={} setup_ms_total={:.3} solve_ms_total={:.3} steps={} res_calls={} res_ms_total={:.3} res_ms_avg={:.6} jac_calls={} jac_ms_total={:.3} jac_ms_avg={:.6} nonlinear_solves={} nonlinear_iters_total={} nonlinear_iters_avg={:.3} bdf_nfev={} bdf_njev={} bdf_nlu={}",
            self.method_label,
            self.backend_label,
            self.setup_ms_total,
            self.solve_ms_total,
            self.step_calls,
            self.residual_calls,
            self.residual_ms_total,
            self.avg_residual_ms().unwrap_or(0.0),
            self.jacobian_calls,
            self.jacobian_ms_total,
            self.avg_jacobian_ms().unwrap_or(0.0),
            self.nonlinear_solve_calls,
            self.nonlinear_iterations_total,
            self.avg_nonlinear_iterations().unwrap_or(0.0),
            self.bdf_nfev_total,
            self.bdf_njev_total,
            self.bdf_nlu_total,
        )
    }

    fn from_backend_stats(
        method_label: impl Into<String>,
        backend_label: impl Into<String>,
        stats: &IvpBackendStatistics,
    ) -> Self {
        Self {
            method_label: method_label.into(),
            backend_label: backend_label.into(),
            setup_ms_total: stats.backend_prepare_ms_total,
            solve_ms_total: stats.solve_ms_total,
            residual_calls: stats.residual_calls,
            residual_ms_total: stats.residual_ms_total,
            jacobian_calls: stats.jacobian_calls,
            jacobian_ms_total: stats.jacobian_ms_total,
            step_calls: stats.step_calls,
            nonlinear_solve_calls: stats.nonlinear_solve_calls,
            nonlinear_iterations_total: stats.nonlinear_iterations_total,
            bdf_nfev_total: stats.bdf_nfev_total,
            bdf_njev_total: stats.bdf_njev_total,
            bdf_nlu_total: stats.bdf_nlu_total,
        }
    }

    fn from_radau_stats(
        method_label: impl Into<String>,
        backend_label: impl Into<String>,
        stats: &RadauStatistics,
    ) -> Self {
        Self {
            method_label: method_label.into(),
            backend_label: backend_label.into(),
            setup_ms_total: stats.backend_prepare_ms_total,
            solve_ms_total: stats.solve_ms_total,
            residual_calls: stats.residual_calls,
            residual_ms_total: stats.residual_ms_total,
            jacobian_calls: stats.jacobian_calls,
            jacobian_ms_total: stats.jacobian_ms_total,
            step_calls: stats.step_calls,
            nonlinear_solve_calls: stats.newton_solve_calls,
            nonlinear_iterations_total: stats.newton_iterations_total,
            bdf_nfev_total: 0,
            bdf_njev_total: 0,
            bdf_nlu_total: stats.lu_factorizations,
        }
    }
}

#[derive(Debug)]
pub enum UniversalOdeError {
    Backend(IvpBackendError),
    UnsupportedGeneratedBackendForMethod { method: String },
}

impl std::fmt::Display for UniversalOdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(err) => write!(f, "{err}"),
            Self::UnsupportedGeneratedBackendForMethod { method } => {
                write!(
                    f,
                    "generated/AOT backend selection is not supported for method `{method}`"
                )
            }
        }
    }
}

impl std::error::Error for UniversalOdeError {}

impl From<IvpBackendError> for UniversalOdeError {
    fn from(value: IvpBackendError) -> Self {
        Self::Backend(value)
    }
}

pub enum SolverInstance {
    NonStiff(nonstiffODE),
    Radau(Radau),
    BDF(BdfOdeSolver),
    BE(BE),
}

pub struct UniversalODESolver {
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    solver_type: SolverType,
    solver_instance: Option<SolverInstance>,
    t_result: Option<DVector<f64>>,
    y_result: Option<DMatrix<f64>>,
    stop_condition: Option<HashMap<String, f64>>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
    first_step: Option<f64>,
    vectorized: bool,
    jac_sparsity: Option<DMatrix<f64>>,
    neighborhood_check: Option<f64>,
    parallel: bool,
    generated_backend_config: Option<SymbolicIvpGeneratedBackendConfig>,
    solver_params_legacy: HashMap<String, SolverParam>,
}

impl UniversalODESolver {
    /// Create a modern universal solver facade.
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Self {
        Self {
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            solver_type,
            solver_instance: None,
            t_result: None,
            y_result: None,
            stop_condition: None,
            step_size: None,
            tolerance: None,
            max_iterations: None,
            rtol: None,
            atol: None,
            max_step: None,
            first_step: None,
            vectorized: false,
            jac_sparsity: None,
            neighborhood_check: None,
            parallel: false,
            generated_backend_config: None,
            solver_params_legacy: HashMap::new(),
        }
    }

    fn method_label(&self) -> String {
        match &self.solver_type {
            SolverType::NonStiff(method) => method.clone(),
            SolverType::Radau(order) => format!("Radau::{order:?}"),
            SolverType::BDF => "BDF".to_string(),
            SolverType::BackwardEuler => "BackwardEuler".to_string(),
        }
    }

    fn backend_label(&self) -> String {
        if let Some(config) = self.generated_backend_config.as_ref() {
            format!(
                "{:?}:{:?}:{:?}",
                config.build_policy, config.aot_codegen_backend, config.aot_c_compiler
            )
        } else {
            "Lambdify".to_string()
        }
    }

    /// Legacy constructor kept for old call-sites that still think in terms of
    /// the historical facade. Prefer [`Self::new`] for new code.
    pub fn new_legacy(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Self {
        Self::new(eq_system, values, arg, solver_type, t0, y0, t_bound)
    }

    pub fn set_step_size(&mut self, step: f64) {
        self.step_size = Some(step);
        self.solver_params_legacy
            .insert("step_size".to_string(), SolverParam::Float(step));
    }

    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = Some(tolerance);
        self.solver_params_legacy
            .insert("tolerance".to_string(), SolverParam::Float(tolerance));
    }

    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = Some(max_iter);
        self.solver_params_legacy
            .insert("max_iterations".to_string(), SolverParam::Int(max_iter));
    }

    pub fn set_rtol(&mut self, rtol: f64) {
        self.rtol = Some(rtol);
        self.solver_params_legacy
            .insert("rtol".to_string(), SolverParam::Float(rtol));
    }

    pub fn set_atol(&mut self, atol: f64) {
        self.atol = Some(atol);
        self.solver_params_legacy
            .insert("atol".to_string(), SolverParam::Float(atol));
    }

    pub fn set_max_step(&mut self, max_step: f64) {
        self.max_step = Some(max_step);
        self.solver_params_legacy
            .insert("max_step".to_string(), SolverParam::Float(max_step));
    }

    pub fn set_first_step(&mut self, first_step: Option<f64>) {
        self.first_step = first_step;
        self.solver_params_legacy.insert(
            "first_step".to_string(),
            SolverParam::OptionalFloat(first_step),
        );
    }

    pub fn set_vectorized(&mut self, vectorized: bool) {
        self.vectorized = vectorized;
        self.solver_params_legacy
            .insert("vectorized".to_string(), SolverParam::Bool(vectorized));
    }

    pub fn set_jac_sparsity(&mut self, jac_sparsity: Option<DMatrix<f64>>) {
        self.jac_sparsity = jac_sparsity.clone();
        self.solver_params_legacy.insert(
            "jac_sparsity".to_string(),
            SolverParam::OptionalMatrix(jac_sparsity),
        );
    }

    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
        self.solver_params_legacy
            .insert("parallel".to_string(), SolverParam::Bool(parallel));
    }

    pub fn set_stop_condition(&mut self, stop_condition: HashMap<String, f64>) {
        self.stop_condition = Some(stop_condition);
    }

    pub fn set_neighborhood_check(&mut self, tolerance: f64) {
        self.neighborhood_check = Some(tolerance);
        self.solver_params_legacy.insert(
            "neighborhood_check".to_string(),
            SolverParam::Float(tolerance),
        );
    }

    pub fn set_generated_backend_config(&mut self, config: SymbolicIvpGeneratedBackendConfig) {
        self.generated_backend_config = Some(config);
    }

    pub fn set_generated_backend_mode(&mut self, mode: DenseIvpGeneratedBackendMode) {
        let config = self
            .generated_backend_config
            .clone()
            .map(|current| {
                let mut next = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
                next.resolver = current.resolver.clone();
                next.aot_options = current.aot_options;
                next.aot_codegen_backend = current.aot_codegen_backend;
                next.aot_c_compiler = current.aot_c_compiler.clone();
                next.output_parent_dir = current.output_parent_dir.clone();
                next.crate_name_override = current.crate_name_override.clone();
                next.module_name_override = current.module_name_override.clone();
                next
            })
            .unwrap_or_else(|| SymbolicIvpGeneratedBackendConfig::from_mode(mode));
        self.set_generated_backend_config(config);
    }

    pub fn set_generated_backend_c_tcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        );
    }

    pub fn set_generated_backend_c_gcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        );
    }

    pub fn set_generated_backend_zig(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        );
    }

    pub fn set_generated_backend_for_repeated_solves(
        &mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .for_repeated_solves(),
        );
    }

    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.set_generated_backend_config(config);
        self
    }

    pub fn with_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        self.set_generated_backend_mode(mode);
        self
    }

    pub fn with_generated_backend_c_tcc(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.set_generated_backend_c_tcc(output_parent_dir);
        self
    }

    pub fn with_generated_backend_c_gcc(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.set_generated_backend_c_gcc(output_parent_dir);
        self
    }

    pub fn with_generated_backend_zig(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.set_generated_backend_zig(output_parent_dir);
        self
    }

    pub fn with_generated_backend_for_repeated_solves(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_generated_backend_for_repeated_solves(output_parent_dir);
        self
    }

    pub fn set_parameter(&mut self, key: &str, value: SolverParam) {
        self.solver_params_legacy
            .insert(key.to_string(), value.clone());
        match (key, value) {
            ("step_size", SolverParam::Float(v)) => self.set_step_size(v),
            ("tolerance", SolverParam::Float(v)) => self.set_tolerance(v),
            ("max_iterations", SolverParam::Int(v)) => self.set_max_iterations(v),
            ("rtol", SolverParam::Float(v)) => self.set_rtol(v),
            ("atol", SolverParam::Float(v)) => self.set_atol(v),
            ("max_step", SolverParam::Float(v)) => self.set_max_step(v),
            ("first_step", SolverParam::OptionalFloat(v)) => self.set_first_step(v),
            ("first_step", SolverParam::Float(v)) => self.set_first_step(Some(v)),
            ("vectorized", SolverParam::Bool(v)) => self.set_vectorized(v),
            ("jac_sparsity", SolverParam::OptionalMatrix(v)) => self.set_jac_sparsity(v),
            ("parallel", SolverParam::Bool(v)) => self.set_parallel(v),
            ("neighborhood_check", SolverParam::Float(v)) => self.set_neighborhood_check(v),
            _ => {}
        }
    }

    pub fn set_parameters(&mut self, params: HashMap<String, SolverParam>) {
        for (key, value) in params {
            self.set_parameter(&key, value);
        }
    }

    /// Legacy string-key setter preserved for compatibility with older wrappers.
    pub fn set_parameter_legacy(&mut self, key: &str, value: SolverParam) {
        self.set_parameter(key, value);
    }

    /// Legacy bulk string-key setter preserved for compatibility with older wrappers.
    pub fn set_parameters_legacy(&mut self, params: HashMap<String, SolverParam>) {
        self.set_parameters(params);
    }

    pub fn initialize(&mut self) {
        self.try_initialize()
            .expect("universal ODE solver initialization should succeed");
    }

    /// Legacy initialize wrapper. Prefer [`Self::try_initialize`] or [`Self::initialize`].
    pub fn initialize_legacy(&mut self) {
        self.initialize();
    }

    pub fn try_initialize(&mut self) -> Result<(), UniversalOdeError> {
        self.solver_instance = Some(match &self.solver_type {
            SolverType::NonStiff(method) => {
                if self.generated_backend_config.is_some() {
                    return Err(UniversalOdeError::UnsupportedGeneratedBackendForMethod {
                        method: method.clone(),
                    });
                }
                let mut solver = nonstiffODE::new(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    method.clone(),
                    self.t0,
                    self.y0.clone(),
                    self.t_bound,
                    self.step_size.unwrap_or(1e-3),
                    self.stop_condition.clone(),
                );
                if let Some(tol) = self.neighborhood_check {
                    solver.set_neighborhood_check(tol);
                }
                SolverInstance::NonStiff(solver)
            }
            SolverType::Radau(order) => {
                let mut options = RadauSolverOptions::new(
                    order.clone(),
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.tolerance.unwrap_or(1e-6),
                    self.max_iterations.unwrap_or(50),
                    self.step_size,
                    self.t0,
                    self.t_bound,
                    self.y0.clone(),
                )
                .with_parallel(self.parallel);
                if let Some(config) = self.generated_backend_config.clone() {
                    options = options.with_generated_backend_config(config);
                }
                let mut solver = Radau::new_with_options(options);
                if let Some(stop_condition) = self.stop_condition.clone() {
                    solver.set_stop_condition(stop_condition);
                }
                SolverInstance::Radau(solver)
            }
            SolverType::BDF => {
                let mut options = BdfSolverOptions::new(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    "BDF".to_string(),
                    self.t0,
                    self.y0.clone(),
                    self.t_bound,
                    self.max_step.unwrap_or(1e-3),
                    self.rtol.unwrap_or(1e-5),
                    self.atol.unwrap_or(1e-5),
                    self.jac_sparsity.clone(),
                    self.vectorized,
                    self.first_step,
                );
                if let Some(config) = self.generated_backend_config.clone() {
                    options = options.with_generated_backend_config(config);
                }
                let mut solver = BdfOdeSolver::new_with_options(options);
                if let Some(stop_condition) = self.stop_condition.clone() {
                    solver.set_stop_condition(stop_condition);
                }
                SolverInstance::BDF(solver)
            }
            SolverType::BackwardEuler => {
                let mut options = BeSolverOptions::new(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.tolerance.unwrap_or(1e-6),
                    self.max_iterations.unwrap_or(50),
                    self.step_size,
                    self.t0,
                    self.t_bound,
                    self.y0.clone(),
                );
                if let Some(config) = self.generated_backend_config.clone() {
                    options = options.with_generated_backend_config(config);
                }
                let mut solver = BE::new_with_options(options);
                if let Some(stop_condition) = self.stop_condition.clone() {
                    solver.set_stop_condition(stop_condition);
                }
                if let Some(tol) = self.neighborhood_check {
                    solver.set_neighborhood_check(tol);
                }
                SolverInstance::BE(solver)
            }
        });
        Ok(())
    }

    pub fn try_solve(&mut self) -> Result<(), UniversalOdeError> {
        if self.solver_instance.is_none() {
            self.try_initialize()?;
        }

        match self
            .solver_instance
            .as_mut()
            .expect("solver instance should be initialized")
        {
            SolverInstance::NonStiff(solver) => {
                solver.solve();
                let (t_result, y_result) = solver.get_result();
                self.t_result = Some(t_result);
                self.y_result = Some(y_result);
            }
            SolverInstance::Radau(solver) => {
                solver.try_solve()?;
                let (t_result, y_result) = solver.get_result();
                self.t_result = t_result;
                self.y_result = y_result;
            }
            SolverInstance::BDF(solver) => {
                solver.solve();
                let (t_result, y_result) = solver.get_result();
                self.t_result = Some(t_result);
                self.y_result = Some(y_result);
            }
            SolverInstance::BE(solver) => {
                solver.try_solve()?;
                let (t_result, y_result) = solver.get_result();
                self.t_result = t_result;
                self.y_result = y_result;
            }
        }
        Ok(())
    }

    pub fn solve(&mut self) {
        self.try_solve()
            .expect("universal ODE solver execution should succeed");
    }

    /// Legacy solve wrapper mirroring the historical facade.
    pub fn solve_legacy(&mut self) {
        self.solve();
    }

    pub fn get_result(&self) -> (Option<DVector<f64>>, Option<DMatrix<f64>>) {
        (self.t_result.clone(), self.y_result.clone())
    }

    pub fn get_status(&self) -> Option<String> {
        match self.solver_instance.as_ref()? {
            SolverInstance::NonStiff(solver) => Some(solver.get_status().clone()),
            SolverInstance::Radau(solver) => Some(solver.get_status().clone()),
            SolverInstance::BDF(solver) => Some(solver.get_status().clone()),
            SolverInstance::BE(solver) => Some(solver.get_status().clone()),
        }
    }

    pub fn get_statistics(&self) -> Option<UniversalIvpStatistics> {
        let method = self.method_label();
        let backend = self.backend_label();
        match self.solver_instance.as_ref()? {
            SolverInstance::NonStiff(_) => None,
            SolverInstance::Radau(solver) => Some(UniversalIvpStatistics::from_radau_stats(
                method,
                backend,
                &solver.get_statistics(),
            )),
            SolverInstance::BDF(solver) => Some(UniversalIvpStatistics::from_backend_stats(
                method,
                backend,
                &solver.get_statistics(),
            )),
            SolverInstance::BE(solver) => Some(UniversalIvpStatistics::from_backend_stats(
                method,
                backend,
                &solver.get_statistics(),
            )),
        }
    }

    pub fn statistics_report(&self) -> Option<String> {
        self.get_statistics().map(|stats| stats.table_report())
    }

    pub fn plot_result(&self) {
        if let (Some(t_result), Some(y_result)) = (&self.t_result, &self.y_result) {
            plots(
                self.arg.clone(),
                self.values.clone(),
                t_result.clone(),
                y_result.clone(),
            );
        }
    }

    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self.solver_instance.as_ref() {
            Some(SolverInstance::NonStiff(solver)) => solver.save_result(),
            Some(SolverInstance::BDF(solver)) => solver.save_result(),
            _ => Ok(()),
        }
    }
}

impl UniversalODESolver {
    pub fn rk45(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff("RK45".to_string()),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    pub fn dopri(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff("DOPRI".to_string()),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    pub fn ab4(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff("AB4".to_string()),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    pub fn radau(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        order: RadauOrder,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: Option<f64>,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::Radau(order),
            t0,
            y0,
            t_bound,
        );
        solver.set_tolerance(tolerance);
        solver.set_max_iterations(max_iterations);
        if let Some(step) = step_size {
            solver.set_step_size(step);
        }
        solver
    }

    pub fn bdf(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
    ) -> Self {
        let mut solver = Self::new(eq_system, values, arg, SolverType::BDF, t0, y0, t_bound);
        solver.set_max_step(max_step);
        solver.set_rtol(rtol);
        solver.set_atol(atol);
        solver
    }

    pub fn backward_euler(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: Option<f64>,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::BackwardEuler,
            t0,
            y0,
            t_bound,
        );
        solver.set_tolerance(tolerance);
        solver.set_max_iterations(max_iterations);
        if let Some(step) = step_size {
            solver.set_step_size(step);
        }
        solver
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn simple_decay_problem() -> (Vec<Expr>, Vec<String>, String, DVector<f64>) {
        let x = Expr::Var("x".to_string());
        let eq_system = vec![-x.clone()];
        let values = vec!["x".to_string()];
        let arg = "t".to_string();
        let y0 = DVector::from_vec(vec![1.0]);
        (eq_system, values, arg, y0)
    }

    #[test]
    fn universal_ode_api_legacy_nonstiff_smoke() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver = UniversalODESolver::new_legacy(
            eq_system,
            values,
            arg,
            SolverType::NonStiff("RK45".to_string()),
            0.0,
            y0,
            0.1,
        );
        solver.set_parameter_legacy("step_size", SolverParam::Float(1e-3));
        solver.initialize_legacy();
        solver.solve_legacy();
        let (t, y) = solver.get_result();
        assert!(t.is_some());
        assert!(y.is_some());
        assert_eq!(solver.get_status().as_deref(), Some("finished"));
    }

    #[test]
    fn universal_ode_api_be_exposes_statistics() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            0.0,
            y0,
            0.1,
            1e-8,
            20,
            Some(1e-2),
        );
        solver.solve();
        let stats = solver
            .get_statistics()
            .expect("BE should expose normalized statistics");
        assert_eq!(stats.method_label, "BackwardEuler");
        assert!(stats.step_calls > 0);
    }

    #[test]
    fn universal_ode_api_rejects_generated_backend_for_nonstiff() {
        let (eq_system, values, arg, y0) = simple_decay_problem();
        let mut solver = UniversalODESolver::rk45(eq_system, values, arg, 0.0, y0, 0.1, 1e-3)
            .with_generated_backend_c_tcc("target/test-artifacts/ode-api2");
        let err = solver
            .try_initialize()
            .expect_err("nonstiff facade should reject generated backend selection");
        assert!(matches!(
            err,
            UniversalOdeError::UnsupportedGeneratedBackendForMethod { .. }
        ));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Tests
/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests2 {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    #[test]
    fn test_universal_rk45() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let step_size = 1e-4;

        let mut solver =
            UniversalODESolver::rk45(eq_system, values, arg, t0, y0, t_bound, step_size);

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());

        let y_final = y_result.clone().unwrap()[(y_result.as_ref().unwrap().nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        assert_relative_eq!(y_final, expected, epsilon = 1e-2);
    }
    #[test]
    fn test_rk45_exponential() {
        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0 y'(0) = 1
        // solution y(x) = 1/2 (e^x - e^(-x))

        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;

        let y0 = DVector::from_vec(vec![0.0, 1.0]);
        let t_bound = 0.5;
        let step = 1e-3;
        let solver_type = SolverType::NonStiff("RK45".to_owned());
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = 0.5 * (x.exp() - (-x).exp());
            assert_relative_eq!(y, expected, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_ab4_cos() {
        // y'' + y = 0,
        // y0' =y1,
        // y1' =- y0
        // y(0) = 1, y'(0) = 0 (solution: y = cos(x))
        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        // cos(0)=1, -sin(0)=0
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI;
        let step = 1e-5;
        let solver_type = SolverType::NonStiff("AB4".to_owned());
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        let y1: DVector<f64> = y_final.column(1).into();
        // println!("{:?} \n {:?}", y0, y1);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x.cos();
            let expected1 = -x.sin();
            assert_relative_eq!(y, expected, epsilon = 1e-4);
            assert_relative_eq!(y1[i], expected1, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_rk45_cos2() {
        // y'' + y = 0,
        // y0' =y1,
        // y1' =- y0
        // y(0) = 1, y'(0) = 0 (solution: y = cos(x))
        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
        // cos(0)=1, -sin(0)=0
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI;
        let step = 1e-5;
        let solver_type = SolverType::NonStiff("RK45".to_owned());
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        let y1: DVector<f64> = y_final.column(1).into();
        // println!("{:?} \n {:?}", y0, y1);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x.cos();
            let expected1 = -x.sin();
            assert_relative_eq!(y, expected, epsilon = 1e-4);
            assert_relative_eq!(y1[i], expected1, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_universal_radau() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::radau(
            eq_system,
            values,
            arg,
            RadauOrder::Order3,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(1e-3),
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }

    #[test]
    fn test_universal_bdf() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }
    #[test]
    fn test_bdf_exponential() {
        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0 y' = 1
        // solution y(x) = 1/2 (e^x - e^(-x))

        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;

        let y0 = DVector::from_vec(vec![0.0, 1.0]);
        let t_bound = 0.5;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = 0.5 * (x.exp() - (-x).exp());
            assert_relative_eq!(y, expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_bdf_linear() {
        // y'' = 0, y(0) = 0, y(1) = 1 (solution: y = x)
        // y0' = y1
        // y1' = 0
        //
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let y0 = DVector::from_vec(vec![0.0, 0.9997384083916304]);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut solver =
            UniversalODESolver::bdf(eq_vec, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x;
            assert_relative_eq!(y, expected, epsilon = 1e-3);
        }
    }
    #[test]
    fn test_universal_backward_euler() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(1e-3),
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }
    #[test]
    fn test_direct_setting() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::new(
            eq_system,
            values,
            arg,
            SolverType::Radau(RadauOrder::Order3),
            t0,
            y0,
            t_bound,
        );
        solver.set_max_iterations(100);
        solver.set_tolerance(1e-6);
        solver.set_step_size(1e-3);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }

    #[test]
    fn test_universal_stop_condition_rk45() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0; // Large bound to ensure stop condition triggers first
        let step_size = 0.01;

        let mut solver =
            UniversalODESolver::rk45(eq_system, values, arg, t0, y0, t_bound, step_size);

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);
        solver.set_neighborhood_check(1e-2);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= 1e-2);
    }

    #[test]
    fn test_universal_stop_condition_radau() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;

        let mut solver = UniversalODESolver::radau(
            eq_system,
            values,
            arg,
            RadauOrder::Order3,
            t0,
            y0,
            t_bound,
            1e-2,
            50,
            Some(0.01),
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= 1e-2); // Uses Radau's tolerance
    }

    #[test]
    fn test_universal_stop_condition_bdf() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-3);

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= 1e-2); // Uses BDF's atol
    }

    #[test]
    fn test_universal_stop_condition_be() {
        // Test: y' = y, y(0) = 1, stop when y reaches 1.5
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;

        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(0.01),
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 1.5);
        solver.set_stop_condition(stop_condition);
        solver.set_neighborhood_check(1e-2);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 1.5).abs() <= 1e-2);
    }

    #[test]
    fn test_universal_no_stop_condition() {
        // Test without stop condition - should run to t_bound
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let step_size = 0.1;

        let mut solver =
            UniversalODESolver::rk45(eq_system, values, arg, t0, y0, t_bound, step_size);

        solver.solve();

        assert_eq!(solver.get_status().unwrap(), "finished");
        let (t_result, _) = solver.get_result();
        let t_res = t_result.unwrap();
        let final_t = t_res[t_res.len() - 1];
        assert!((final_t - t_bound).abs() <= step_size);
    }
}
