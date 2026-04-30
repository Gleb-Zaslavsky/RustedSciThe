use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, PreparedSymbolicIvpProblem, SharedIvpParameterValues,
    SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, IvpBackendStatistics, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_problem,
};
use log::info;
use nalgebra::{DMatrix, DVector, Matrix};
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
// solve algebraic nonlinear system with free parameter t
//#[derive(Debug)]
#[derive(Clone)]
pub struct NreSolverOptions {
    pub eq_system: Vec<Expr>,
    pub initial_guess: DVector<f64>,
    pub values: Vec<String>,
    pub arg: String,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub dt: f64,
    pub global_timestepping: bool,
    pub t_bound: Option<f64>,
    pub generated_backend_config: SymbolicIvpGeneratedBackendConfig,
}

impl NreSolverOptions {
    pub fn new(
        eq_system: Vec<Expr>,
        initial_guess: DVector<f64>,
        values: Vec<String>,
        arg: String,
        tolerance: f64,
        max_iterations: usize,
        dt: f64,
        global_timestepping: bool,
        t_bound: Option<f64>,
    ) -> Self {
        Self {
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            dt,
            global_timestepping,
            t_bound,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
        }
    }

    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.generated_backend_config = config;
        self
    }

    pub fn with_dense_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        let mut config = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
        config.resolver = self.generated_backend_config.resolver.clone();
        config.aot_options = self.generated_backend_config.aot_options;
        config.aot_codegen_backend = self.generated_backend_config.aot_codegen_backend;
        config.aot_c_compiler = self.generated_backend_config.aot_c_compiler.clone();
        config.output_parent_dir = self.generated_backend_config.output_parent_dir.clone();
        config.crate_name_override = self.generated_backend_config.crate_name_override.clone();
        config.module_name_override = self.generated_backend_config.module_name_override.clone();
        self.generated_backend_config = config;
        self
    }

    /// Uses compiled dense IVP path via `C + tcc` when startup latency matters most.
    ///
    /// This is usually the first compiled backend worth trying for larger stiff
    /// Backward Euler problems.
    pub fn with_dense_generated_backend_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Uses compiled dense IVP path via `C + gcc` when runtime throughput matters more.
    ///
    /// Prefer this when repeated dense Newton solves matter more than startup
    /// latency.
    pub fn with_dense_generated_backend_c_gcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        )
    }

    /// Uses compiled dense IVP path via Zig.
    pub fn with_dense_generated_backend_zig(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        )
    }

    /// Recommended generated-backend preset for dense IVP repeated solves.
    ///
    /// For Backward Euler this is a reasonable compiled preset once the system
    /// is large enough that Jacobian throughput starts to matter.
    pub fn with_dense_generated_backend_for_repeated_solves(
        self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .for_repeated_solves(),
        )
    }
}

pub struct NRE {
    pub eq_system: Vec<Expr>,        //
    pub initial_guess: DVector<f64>, // initial guess
    pub values: Vec<String>,
    pub arg: String,
    pub tolerance: f64,               // tolerance
    pub max_iterations: usize,        // max number of iterations
    pub max_error: f64,               // max error
    pub result: Option<DVector<f64>>, // result of the iteration
    pub jacobian: Option<Vec<Vec<Expr>>>,
    pub fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    pub jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>>,
    pub equation_parameters: Option<Vec<String>>,
    pub equation_parameter_values: Option<DVector<f64>>,
    pub t: f64,
    pub y: DVector<f64>,
    pub dt: f64,
    n: usize,
    pub global_timestepping: bool,
    pub t_bound: Option<f64>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    statistics: Arc<Mutex<IvpBackendStatistics>>,
}

impl Display for NRE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //   write!(f, "{}", self.eq_system);
        write!(
            f,
            "Initial guess: {:?}, tolerance: {}, max_iterations: {}, max_error: {}, result: {:?}",
            self.initial_guess, self.tolerance, self.max_iterations, self.max_error, self.result
        )
    }
}

impl NRE {
    pub fn new(
        eq_system: Vec<Expr>,        //
        initial_guess: DVector<f64>, // initial guess
        values: Vec<String>,
        arg: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations

        dt: f64,
        global_timestepping: bool,
        t_bound: Option<f64>,
    ) -> NRE {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        NRE {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            tolerance,
            max_iterations,

            dt,
            global_timestepping,
            t_bound,
            result: None,
            jacobian: None,
            fun: Box::new(|_t, y| y.clone()),
            jac: None,
            equation_parameters: None,
            equation_parameter_values: None,
            t: 0.0,
            y: initial_guess.clone(),
            max_error: 1e-3,
            n: 0,
            parameter_values_handle: None,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
            statistics: Arc::new(Mutex::new(IvpBackendStatistics::default())),
        }
    }

    /// Preferred grouped setup path for Newton-Raphson-for-Euler backend.
    pub fn new_with_options(options: NreSolverOptions) -> Self {
        Self::new(
            options.eq_system,
            options.initial_guess,
            options.values,
            options.arg,
            options.tolerance,
            options.max_iterations,
            options.dt,
            options.global_timestepping,
            options.t_bound,
        )
        .with_generated_backend_config(options.generated_backend_config)
    }

    pub fn set_generated_backend_config(&mut self, config: SymbolicIvpGeneratedBackendConfig) {
        self.generated_backend_config = config;
        self.jac = None;
    }

    pub fn generated_backend_config(&self) -> &SymbolicIvpGeneratedBackendConfig {
        &self.generated_backend_config
    }

    pub fn statistics(&self) -> IvpBackendStatistics {
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .clone()
    }

    pub(crate) fn statistics_handle(&self) -> Arc<Mutex<IvpBackendStatistics>> {
        Arc::clone(&self.statistics)
    }

    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.set_generated_backend_config(config);
        self
    }

    pub fn set_dense_generated_backend_mode(&mut self, mode: DenseIvpGeneratedBackendMode) {
        let mut config = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
        config.resolver = self.generated_backend_config.resolver.clone();
        config.aot_options = self.generated_backend_config.aot_options;
        config.aot_codegen_backend = self.generated_backend_config.aot_codegen_backend;
        config.aot_c_compiler = self.generated_backend_config.aot_c_compiler.clone();
        config.output_parent_dir = self.generated_backend_config.output_parent_dir.clone();
        config.crate_name_override = self.generated_backend_config.crate_name_override.clone();
        config.module_name_override = self.generated_backend_config.module_name_override.clone();
        self.set_generated_backend_config(config);
    }

    pub fn with_dense_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        self.set_dense_generated_backend_mode(mode);
        self
    }

    /// Uses compiled dense IVP path via `C + tcc` when startup latency matters most.
    pub fn set_dense_generated_backend_c_tcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        );
    }

    /// Uses compiled dense IVP path via `C + gcc` for runtime-oriented repeated solves.
    pub fn set_dense_generated_backend_c_gcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        );
    }

    /// Uses compiled dense IVP path via Zig.
    pub fn set_dense_generated_backend_zig(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        );
    }

    /// Recommended generated-backend preset for dense IVP repeated solves.
    pub fn set_dense_generated_backend_for_repeated_solves(
        &mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .for_repeated_solves(),
        );
    }

    /// Builder-style alias for `C + tcc` dense generated backend setup.
    pub fn with_dense_generated_backend_c_tcc(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_c_tcc(output_parent_dir);
        self
    }

    /// Builder-style alias for `C + gcc` dense generated backend setup.
    pub fn with_dense_generated_backend_c_gcc(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_c_gcc(output_parent_dir);
        self
    }

    /// Builder-style alias for Zig dense generated backend setup.
    pub fn with_dense_generated_backend_zig(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_zig(output_parent_dir);
        self
    }

    /// Builder-style alias for the recommended repeated-solve IVP preset.
    pub fn with_dense_generated_backend_for_repeated_solves(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_for_repeated_solves(output_parent_dir);
        self
    }
    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn set_equation_parameters(&mut self, params: Option<&[&str]>) {
        self.equation_parameters =
            params.map(|params| params.iter().map(|p| (*p).to_string()).collect());
        self.jac = None;
    }

    pub fn set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), IvpBackendError> {
        if let Some(parameters) = self.equation_parameters.as_ref() {
            if parameters.len() != values.len() {
                return Err(IvpBackendError::ParameterCountMismatch {
                    expected: parameters.len(),
                    actual: values.len(),
                });
            }
        } else if !values.is_empty() {
            return Err(IvpBackendError::ParameterCountMismatch {
                expected: 0,
                actual: values.len(),
            });
        }

        if let Some(handle) = self.parameter_values_handle.as_ref() {
            let mut slot = handle
                .write()
                .expect("shared IVP parameter state lock poisoned");
            *slot = values.clone();
        }
        self.equation_parameter_values = Some(values);
        Ok(())
    }

    pub(crate) fn install_prepared_backend(&mut self, prepared: PreparedSymbolicIvpProblem) {
        self.jacobian = Some(prepared.symbolic_jacobian.clone());
        self.parameter_values_handle = prepared.parameter_values_handle();
        self.equation_parameters = prepared.equation_parameters.clone();

        let residual = prepared.residual;
        let jacobian = prepared.jacobian;
        let stats_for_residual = self.statistics_handle();
        self.fun = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let start = Instant::now();
            let out = residual(t, y);
            stats_for_residual
                .lock()
                .expect("IVP statistics lock poisoned")
                .record_residual_duration(start.elapsed());
            out
        });
        let stats_for_jacobian = self.statistics_handle();
        self.jac = Some(Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
            let start = Instant::now();
            let out = jacobian(t, y);
            stats_for_jacobian
                .lock()
                .expect("IVP statistics lock poisoned")
                .record_jacobian_duration(start.elapsed());
            out
        }));
        self.n = self.eq_system.len();
    }

    pub fn try_eq_generate(&mut self) -> Result<(), IvpBackendError> {
        info!("generating equations and jacobian");
        let start = Instant::now();
        let mut options = SymbolicIvpProblemOptions::new();
        if let Some(parameters) = self.equation_parameters.clone() {
            options = options.with_equation_parameters(parameters);
        }
        if let Some(values) = self.equation_parameter_values.clone() {
            options = options.with_equation_parameter_values(values);
        }

        let prepared = prepare_generated_symbolic_ivp_problem(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
            options.with_aot_options(self.generated_backend_config.aot_options),
            self.generated_backend_config.clone(),
        )
        .map_err(|err| IvpBackendError::GeneratedBackendFailure {
            message: err.to_string(),
        })?;
        self.generated_backend_config.resolver = prepared.updated_resolver.clone();
        self.install_prepared_backend(prepared.into_problem());
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .record_backend_prepare_duration(start.elapsed());
        assert_eq!(&self.eq_system.len(), &self.n);
        Ok(())
    }

    pub fn eq_generate(&mut self) {
        self.try_eq_generate()
            .expect("NR_for_Euler symbolic IVP backend generation should succeed");
    }

    pub fn set_new_step(&mut self, t: f64, y: DVector<f64>, initial_guess: DVector<f64>) {
        self.t = t;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_t(&mut self, t: f64) {
        self.t = t;
    }
    pub fn set_initial_guess(&mut self, initial_guess: DVector<f64>) {
        self.initial_guess = initial_guess;
    }

    ///Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self) -> DVector<f64> {
        assert_eq!(&self.y.is_empty(), &false, "y is empty");
        let t = self.t;
        let y = &self.y;
        let f = (self.fun)(t, &y);
        let new_j = &self.jac.as_mut().unwrap()(t, &y);
        let dt = if self.global_timestepping == true {
            self.dt
        } else {
            let t_bound = self
                .t_bound
                .expect("if global_timestepping = false, t_bound must be set");
            let JF = new_j * &f;
            let eps = self.tolerance; // ?????????????????????
            let norm_JF = JF.amax();
            let dt_ = if norm_JF > 0.0 {
                (2.0 * eps / norm_JF).sqrt()
            } else {
                t_bound - self.t
            };
            let dt = dt_.min(t_bound - self.t);
            self.dt = dt; // update global dt for next iteration
            dt
        };

        let y_k_minus_1 = &self.initial_guess;

        //   println!("Newton-Raphson iteration {}", &y);
        let new_G = y - y_k_minus_1 - dt * f;
        //   println!("new_f = {:?}", &new_G);

        let I = DMatrix::identity(self.n, self.n);
        // if new_j is jacobian of jacobian of f(t_k+1, y_k+1),  then jacobian of function G = y_k+1 - y_k - h*f(t_k+1, y_k+1) is
        let J = I - dt * new_j;
        //    println!("J = {:?} /n", &J);
        //equation J*deltay  = -G
        let lu = J.lu();
        let neg_f = -1.0 * new_G;
        let delta_y = lu.solve(&neg_f).expect("The matrix should be invertible");
        //    println!("delta_y = {:?},\n", &delta_y );
        let new_y: DVector<f64> = y + delta_y;

        new_y
    }
    // main function to solve the system of equations

    pub fn solve(&mut self) -> Option<DVector<f64>> {
        //  println!("solving system of equations with Newton-Raphson method");
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .nonlinear_solve_calls += 1;
        let mut y: DVector<f64> = self.initial_guess.clone();
        self.y = y.clone();
        let mut i = 0;
        while i < self.max_iterations {
            let new_y = self.iteration();

            let dy = new_y.clone() - y.clone();

            let error = Matrix::norm(&dy);
            //  println!("new_y = {:?}, dy = {:?}, error = {}", &new_y, &dy, error);
            if error < self.tolerance {
                //  println!("converged in {} iterations", i);
                self.result = Some(new_y.clone());
                self.max_error = error;
                self.statistics
                    .lock()
                    .expect("IVP statistics lock poisoned")
                    .nonlinear_iterations_total += i + 1;
                return Some(new_y);
            } else {
                y = new_y.clone();
                self.y = new_y;
                i += 1;
                //  if i==5 {panic!("Too many iterations")}
                //  println!("\n \n iteration = {}, error = {}", i, error)
            }
        }
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .nonlinear_iterations_total += i;
        None
    }

    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.result.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn nre_new_with_options_installs_generated_backend_mode() {
        let nr = NRE::new_with_options(
            NreSolverOptions::new(
                vec![Expr::parse_expression("y")],
                DVector::from_vec(vec![1.0]),
                vec!["y".to_string()],
                "t".to_string(),
                1e-6,
                20,
                1e-3,
                true,
                None,
            )
            .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::RequirePrebuilt),
        );

        assert_eq!(
            nr.generated_backend_config().build_policy,
            crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy::RequirePrebuilt
        );
    }

    #[test]
    fn nre_generated_backend_surface_keeps_selected_c_backend() {
        let nr = NRE::new_with_options(
            NreSolverOptions::new(
                vec![Expr::parse_expression("y")],
                DVector::from_vec(vec![1.0]),
                vec!["y".to_string()],
                "t".to_string(),
                1e-6,
                20,
                1e-3,
                true,
                None,
            )
            .with_dense_generated_backend_c_gcc("target/generated-ivp-tests")
            .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease),
        );

        assert_eq!(
            nr.generated_backend_config().aot_codegen_backend,
            crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
        );
        assert_eq!(
            nr.generated_backend_config().aot_c_compiler.as_deref(),
            Some("gcc")
        );
    }

    #[test]
    fn nre_generated_backend_repeated_solves_alias_prefers_c_gcc() {
        let nr = NRE::new_with_options(
            NreSolverOptions::new(
                vec![Expr::parse_expression("y")],
                DVector::from_vec(vec![1.0]),
                vec!["y".to_string()],
                "t".to_string(),
                1e-6,
                20,
                1e-3,
                true,
                None,
            )
            .with_dense_generated_backend_for_repeated_solves("target/generated-ivp-tests"),
        );

        assert_eq!(
            nr.generated_backend_config().aot_codegen_backend,
            crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
        );
        assert_eq!(
            nr.generated_backend_config().aot_c_compiler.as_deref(),
            Some("gcc")
        );
    }

    #[test]
    fn test_newton_raphson_solver_for_Euler() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-3;
        let max_iterations = 50;

        let h = 1e-5;

        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            true,
            None,
        );
        nr.eq_generate();

        assert_eq!(nr.eq_system.len(), 2);
        nr.set_t(1.0);
        let solution = nr.solve().unwrap();
        assert_eq!(solution.len(), 2);
        // assert_eq!()
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }

    #[test]
    fn test_newton_raphson_solver_for_Euler_2() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-3;
        let max_iterations = 50;

        let h = 1e-5;

        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            false,
            Some(1.0),
        );
        nr.eq_generate();

        assert_eq!(nr.eq_system.len(), 2);
        nr.set_t(1.0);
        let solution = nr.solve().unwrap();
        assert_eq!(solution.len(), 2);
        // assert_eq!()
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }
}
