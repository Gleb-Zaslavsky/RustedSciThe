use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{SolverParams, NRBVP as NRBDVPd};
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use nalgebra::DMatrix;
use std::any::Any;
use std::collections::HashMap;
//BVP is general api for all variants of BVP solvers

pub struct BVP {
    pub eq_system: Vec<Expr>, // the system of ODEs defined in the symbolic format
    pub initial_guess: DMatrix<f64>, // initial guess s - matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
    pub values: Vec<String>,         //unknown variables
    pub arg: String,                 // time or coordinate
    pub BorderConditions: HashMap<String, Vec<(usize, f64)>>, // hashmap where keys are variable names and values are vectors of tuples with the index of the boundary condition (0 for inititial condition 1 for ending condition) and the value.
    pub t0: f64,                                              // initial value of argument
    pub t_end: f64,                                           // end of argument
    pub n_steps: usize,                                       //  number of  steps
    pub scheme: String,   // define crate using for matrices and vectors
    pub strategy: String, // name of the strategy
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>, // solver parameters
    pub linear_sys_method: Option<String>, // method for solving linear system
    pub method: String,   // define crate using for matrices and vectors
    pub tolerance: f64,   // abs_tolerance for NR_Damp_solver_damped
    pub max_iterations: usize, // maximum number of iterations
    // fields only for damped version of method
    pub rel_tolerance: Option<HashMap<String, f64>>, // absolute tolerance - hashmap of the var names and values of tolerance for them

    pub Bounds: Option<HashMap<String, (f64, f64)>>,
    pub loglevel: Option<String>, //
    pub structure: Option<NRBVP>,
    pub structure_damp: Option<NRBDVPd>,
}

pub fn convert_hashmap_to_solver_params(
    hashmap: &Option<HashMap<String, Option<Vec<f64>>>>,
) -> Option<SolverParams> {
    hashmap.as_ref().map(|params| {
        let max_jac = params
            .get("max_jac")
            .and_then(|v| v.as_ref().map(|vec| vec[0] as usize));
        let max_damp_iter = params
            .get("maxDampIter")
            .and_then(|v| v.as_ref().map(|vec| vec[0] as usize));
        let damp_factor = params
            .get("DampFacor")
            .and_then(|v| v.as_ref().map(|vec| vec[0]));

        SolverParams {
            max_jac,
            max_damp_iter,
            damp_factor,
            adaptive: None, // TODO: Add adaptive conversion if needed
        }
    })
}

impl BVP {
    /// Bind symbolic parameters on the prepared Damped solver.
    ///
    /// The discovery facade historically had no parameter-aware method. This
    /// narrow adapter lets task documents use the native prepared continuation
    /// contract without exposing the legacy `BVP` field layout to the parser.
    pub fn try_set_damped_parameter_binding(
        &mut self,
        names: &[String],
        values: Vec<f64>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let Some(solver) = self.structure_damp.as_mut() else {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy".to_string(),
                value: self.strategy.clone(),
                message: "prepared parameter binding is available only for Damped BVP".to_string(),
            });
        };
        let names = names.iter().map(String::as_str).collect::<Vec<_>>();
        solver.try_set_params(Some(&names))?;
        solver.try_set_param_values(Some(values))
    }

    /// Update the Damped solver's mesh while preserving its prepared callback
    /// contract. A changed mesh invalidates only the runtime layout; the caller
    /// must regenerate before the next prepared solve.
    pub fn try_set_damped_mesh(
        &mut self,
        t0: f64,
        t_end: f64,
        n_steps: usize,
    ) -> Result<(), BvpBackendIntegrationError> {
        let Some(solver) = self.structure_damp.as_mut() else {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy".to_string(),
                value: self.strategy.clone(),
                message: "mesh restart is available only for Damped BVP".to_string(),
            });
        };
        solver.try_set_mesh(t0, t_end, n_steps)
    }

    /// Replace the Damped Newton initial profile without symbolic rebuilding.
    pub fn try_set_damped_initial_guess(
        &mut self,
        initial_guess: DMatrix<f64>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let Some(solver) = self.structure_damp.as_mut() else {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy".to_string(),
                value: self.strategy.clone(),
                message: "initial-guess restart is available only for Damped BVP".to_string(),
            });
        };
        solver.try_set_initial_guess(initial_guess)
    }

    /// Install a warm-start iterate while retaining the Damped prepared plan.
    pub fn try_set_damped_prepared_iterate(
        &mut self,
        iterate: DMatrix<f64>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let Some(solver) = self.structure_damp.as_mut() else {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy".to_string(),
                value: self.strategy.clone(),
                message: "prepared iterate is available only for Damped BVP".to_string(),
            });
        };
        solver.try_set_prepared_iterate(iterate)
    }

    /// Run the Damped solver through its typed cold or prepared entrypoint.
    pub fn try_solve_damped(&mut self, prepared: bool) -> Result<(), BvpBackendIntegrationError> {
        let Some(solver) = self.structure_damp.as_mut() else {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy".to_string(),
                value: self.strategy.clone(),
                message: "typed Damped solve is unavailable for this strategy".to_string(),
            });
        };
        if prepared {
            solver.try_solver_prepared()?;
        } else {
            solver.try_solver()?;
        }
        Ok(())
    }

    /// Prepare the Damped callback/backend lifecycle without entering Newton.
    pub fn try_prepare_damped(&mut self) -> Result<(), BvpBackendIntegrationError> {
        let Some(solver) = self.structure_damp.as_mut() else {
            return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: "strategy".to_string(),
                value: self.strategy.clone(),
                message: "typed Damped preparation is unavailable for this strategy".to_string(),
            });
        };
        solver.try_eq_generate(None, None)
    }

    pub fn from_hashmap(hashmap: &HashMap<String, &dyn Any>) -> BVP {
        let eq_system: Vec<Expr> = hashmap
            .get("eq_system")
            .and_then(|v| v.downcast_ref::<Vec<Expr>>())
            .cloned()
            .unwrap_or_default();
        let initial_guess: DMatrix<f64> = hashmap
            .get("initial_guess")
            .and_then(|v| v.downcast_ref::<DMatrix<f64>>())
            .cloned()
            .unwrap_or_default();
        let values: Vec<String> = hashmap
            .get("values")
            .and_then(|v| v.downcast_ref::<Vec<String>>())
            .cloned()
            .unwrap_or_default();
        let arg: String = hashmap
            .get("arg")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let BorderConditions: HashMap<String, Vec<(usize, f64)>> = hashmap
            .get("BorderConditions")
            .and_then(|v| v.downcast_ref::<HashMap<String, Vec<(usize, f64)>>>())
            .cloned()
            .unwrap_or_default();
        let t0: f64 = hashmap
            .get("t0")
            .and_then(|v| v.downcast_ref::<f64>())
            .cloned()
            .unwrap_or_default();
        let t_end: f64 = hashmap
            .get("t_end")
            .and_then(|v| v.downcast_ref::<f64>())
            .cloned()
            .unwrap_or_default();
        let n_steps: usize = hashmap
            .get("n_steps")
            .and_then(|v| v.downcast_ref::<usize>())
            .cloned()
            .unwrap_or_default();
        let scheme: String = hashmap
            .get("scheme")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let strategy: String = hashmap
            .get("strategy")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let strategy_params: Option<HashMap<String, Option<Vec<f64>>>> = hashmap
            .get("strategy_params")
            .and_then(|v| v.downcast_ref::<Option<HashMap<String, Option<Vec<f64>>>>>())
            .cloned()
            .unwrap_or_default();
        let linear_sys_method: Option<String> = hashmap
            .get("linear_sys_method")
            .and_then(|v| v.downcast_ref::<Option<String>>())
            .cloned()
            .unwrap_or_default();
        let method: String = hashmap
            .get("method")
            .and_then(|v| v.downcast_ref::<String>())
            .cloned()
            .unwrap_or_default();
        let tolerance: f64 = hashmap
            .get("tolerance")
            .and_then(|v| v.downcast_ref::<f64>())
            .cloned()
            .unwrap_or_default();
        let max_iterations: usize = hashmap
            .get("max_iterations")
            .and_then(|v| v.downcast_ref::<usize>())
            .cloned()
            .unwrap_or_default();
        let rel_tolerance: Option<HashMap<String, f64>> = hashmap
            .get("rel_tolerance")
            .and_then(|v| v.downcast_ref::<Option<HashMap<String, f64>>>())
            .cloned()
            .unwrap_or_default();
        let Bounds: Option<HashMap<String, (f64, f64)>> = hashmap
            .get("Bounds")
            .and_then(|v| v.downcast_ref::<Option<HashMap<String, (f64, f64)>>>())
            .cloned()
            .unwrap_or_default();
        let loglevel: Option<String> = hashmap
            .get("loglevel")
            .and_then(|v| v.downcast_ref::<Option<String>>())
            .cloned()
            .unwrap_or_default();

        BVP {
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
            rel_tolerance,
            Bounds,
            loglevel,
            structure: None,
            structure_damp: None,
        }
    }
    pub fn new(
        eq_system: Vec<Expr>,        // the system of ODEs defined in the symbolic format
        initial_guess: DMatrix<f64>, // initial guess s - matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
        values: Vec<String>,         //unknown variables
        arg: String,                 // time or coordinate
        BorderConditions: HashMap<String, Vec<(usize, f64)>>, // hashmap where keys are variable names and values are vectors of tuples with the index of the boundary condition (0 for inititial condition 1 for ending condition) and the value.
        t0: f64,                                              // initial value of argument
        t_end: f64,                                           // end of argument
        n_steps: usize,                                       //  number of  steps
        scheme: String,
        strategy: String, // name of the strategy
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>, // solver parameters
        linear_sys_method: Option<String>, // method for solving linear system
        method: String,   // define crate using for matrices and vectors
        tolerance: f64,   // abs_tolerance for NR_Damp_solver_damped
        max_iterations: usize, // maximum number of iterations
        // fields only for damped version of method
        rel_tolerance: Option<HashMap<String, f64>>, // absolute tolerance - hashmap of the var names and values of tolerance for them
        Bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>, //
    ) -> BVP {
        let (structure, structure_damp, rel_tolerance, Bounds) =
            if strategy == "Frozen" || strategy == "Naive" {
                let rel_tolerance: Option<HashMap<String, f64>> = None;
                let Bounds: Option<HashMap<String, (f64, f64)>> = None;
                let nrbvp = NRBVP::new(
                    eq_system.clone(),
                    initial_guess.clone(),
                    values.clone(),
                    arg.clone(),
                    BorderConditions.clone(),
                    t0,
                    t_end,
                    n_steps,
                    strategy.clone(),
                    strategy_params.clone(),
                    linear_sys_method.clone(),
                    method.clone(),
                    tolerance,
                    max_iterations,
                );
                let structure = Some(nrbvp);
                let structure_damp = None;
                (structure, structure_damp, rel_tolerance, Bounds)
            } else if strategy == "Damped" {
                let converted_params = convert_hashmap_to_solver_params(&strategy_params);
                let nrbvpd = NRBDVPd::new(
                    eq_system.clone(),
                    initial_guess.clone(),
                    values.clone(),
                    arg.clone(),
                    BorderConditions.clone(),
                    t0,
                    t_end,
                    n_steps,
                    scheme.clone(),
                    strategy.clone(),
                    converted_params,
                    linear_sys_method.clone(),
                    method.clone(),
                    tolerance,
                    rel_tolerance.clone(),
                    max_iterations,
                    Bounds.clone(),
                    loglevel.clone(),
                );
                let structure = None;
                let structure_damp = Some(nrbvpd);

                (structure, structure_damp, rel_tolerance, Bounds)
            } else {
                panic!("Unknown strategy");
            };
        BVP {
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
            rel_tolerance,
            Bounds,
            loglevel,
            structure,
            structure_damp,
        }
    }
    pub fn plot_result(&self) {
        if let Some(ref structure) = self.structure {
            structure.plot_result();
        }
        if let Some(ref structure_damp) = self.structure_damp {
            structure_damp.plot_result();
        }
    }
    pub fn gnuplot_result(&self) {
        if let Some(ref structure) = self.structure {
            structure.plot_result();
        }
        if let Some(ref structure_damp) = self.structure_damp {
            structure_damp.gnuplot_result();
        }
    }
    pub fn solve(&mut self) {
        if let Some(ref mut structure) = self.structure {
            structure.solve();
        }
        if let Some(ref mut structure_damp) = self.structure_damp {
            structure_damp.solve();
        }
    }
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        if let Some(structure) = &self.structure {
            let res = structure.get_result();
            return res;
        }
        match &self.structure_damp {
            Some(structure_damp) => {
                let res = structure_damp.get_result();
                return res;
            }
            _ => {
                panic!("Invalid structure!");
            }
        }
    }
    pub fn postprocess_dataset(&self) -> Result<PostprocessDataset, PostprocessError> {
        if let Some(structure) = &self.structure {
            return structure.postprocess_dataset();
        }
        if let Some(structure_damp) = &self.structure_damp {
            return structure_damp.postprocess_dataset();
        }
        Err(PostprocessError::InvalidDataset(
            "BVP postprocess_dataset requires an initialized solver structure".to_string(),
        ))
    }
    pub fn execute_postprocessing(
        &self,
        plan: &PostprocessPlan,
    ) -> Result<PostprocessReport, PostprocessError> {
        let dataset = self.postprocess_dataset()?;
        plan.execute(&dataset)
    }
    pub fn save_to_file(&mut self, filename: Option<String>) {
        if let Some(structure) = &mut self.structure {
            structure.save_to_file();
        }
        if let Some(structure_damp) = &mut self.structure_damp {
            structure_damp.save_to_file(filename);
        }
    }
    pub fn save_to_csv(&mut self, filename: Option<String>) {
        if let Some(structure) = &mut self.structure {
            structure.save_to_file();
        }
        if let Some(structure_damp) = &mut self.structure_damp {
            structure_damp.save_to_csv(filename);
        }
    }
}

// ---------------------------------------------------------------------------
// Typed solver-selection facade
// ---------------------------------------------------------------------------

use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use crate::numerical::BVP_sci::new::{
    BvpSciMatrixLayout, BvpSciNewError, BvpSciSolution, BvpSciSolver,
};
use thiserror::Error;

/// Which boundary-value algorithm the discovery facade should instantiate.
///
/// This enum is intentionally small. It is a routing tool for experimentation,
/// not a replacement for the detailed public API of any individual solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSolverKind {
    /// Adaptive damped Newton from `BVP_Damp`.
    Damped,
    /// Frozen/modified-Newton route from `BVP_Damp`.
    Frozen,
    /// SciPy-like adaptive collocation route from `BVP_sci`.
    SciPyLike,
}

/// Matrix storage requested by the discovery facade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpMatrixBackend {
    Dense,
    Sparse,
    Banded { lower: usize, upper: usize },
}

impl BvpMatrixBackend {
    fn legacy_name(self) -> String {
        match self {
            Self::Dense => "Dense".into(),
            Self::Sparse => "Sparse".into(),
            Self::Banded { .. } => "Banded".into(),
        }
    }

    fn sci_layout(self) -> BvpSciMatrixLayout {
        match self {
            Self::Dense => BvpSciMatrixLayout::Dense,
            Self::Sparse => BvpSciMatrixLayout::Sparse,
            Self::Banded { lower, upper } => BvpSciMatrixLayout::Banded { lower, upper },
        }
    }
}

/// Which endpoint of a first-order BVP carries a scalar condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpBoundarySide {
    Left,
    Right,
}

/// One typed scalar boundary condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvpBoundaryCondition {
    pub state: usize,
    pub side: BvpBoundarySide,
    pub value: f64,
}

/// Solver-independent problem description used by [`BvpSolverBuilder`].
///
/// Unlike the historical facade this type contains no `Any`, string strategy
/// flags or parameter hash maps. The equation vector and initial guess remain
/// symbolic because all three discovery routes currently start from the same
/// first-order symbolic BVP representation.
#[derive(Clone, Debug)]
pub struct BvpProblem {
    pub equations: Vec<Expr>,
    pub state_names: Vec<String>,
    pub independent_variable: String,
    pub mesh: Vec<f64>,
    pub initial_guess: DMatrix<f64>,
    pub boundary_conditions: Vec<BvpBoundaryCondition>,
}

impl BvpProblem {
    /// Create a typed problem from a strictly increasing mesh and a
    /// state-major initial guess (`state x node`).
    pub fn new(
        equations: Vec<Expr>,
        state_names: Vec<String>,
        independent_variable: impl Into<String>,
        mesh: Vec<f64>,
        initial_guess: DMatrix<f64>,
        boundary_conditions: Vec<BvpBoundaryCondition>,
    ) -> Result<Self, BvpApiError> {
        let problem = Self {
            equations,
            state_names,
            independent_variable: independent_variable.into(),
            mesh,
            initial_guess,
            boundary_conditions,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Build a uniform mesh and a constant initial profile.
    pub fn uniform(
        equations: Vec<Expr>,
        state_names: Vec<String>,
        independent_variable: impl Into<String>,
        t0: f64,
        t_end: f64,
        node_count: usize,
        initial_value: f64,
        boundary_conditions: Vec<BvpBoundaryCondition>,
    ) -> Result<Self, BvpApiError> {
        if node_count < 2 || !t0.is_finite() || !t_end.is_finite() || t_end <= t0 {
            return Err(BvpApiError::InvalidConfiguration(
                "uniform BVP mesh requires at least two nodes and t_end > t0".into(),
            ));
        }
        let mesh = (0..node_count)
            .map(|index| t0 + (t_end - t0) * index as f64 / (node_count - 1) as f64)
            .collect::<Vec<_>>();
        let initial_guess = DMatrix::from_element(state_names.len(), node_count, initial_value);
        Self::new(
            equations,
            state_names,
            independent_variable,
            mesh,
            initial_guess,
            boundary_conditions,
        )
    }

    fn validate(&self) -> Result<(), BvpApiError> {
        let dimension = self.state_names.len();
        if dimension == 0 || self.equations.len() != dimension {
            return Err(BvpApiError::InvalidConfiguration(
                "equation and state-name dimensions must be non-zero and equal".into(),
            ));
        }
        if self.mesh.len() < 2
            || self.mesh.iter().any(|value| !value.is_finite())
            || !self.mesh.windows(2).all(|pair| pair[1] > pair[0])
        {
            return Err(BvpApiError::InvalidConfiguration(
                "BVP mesh must be finite and strictly increasing".into(),
            ));
        }
        if self.initial_guess.nrows() != dimension || self.initial_guess.ncols() != self.mesh.len()
        {
            return Err(BvpApiError::InvalidConfiguration(
                "initial guess must have shape state_count x mesh_node_count".into(),
            ));
        }
        if self.boundary_conditions.len() != dimension
            || self
                .boundary_conditions
                .iter()
                .any(|condition| condition.state >= dimension || !condition.value.is_finite())
        {
            return Err(BvpApiError::InvalidConfiguration(
                "exactly one finite boundary condition is required per state".into(),
            ));
        }
        let mut seen = vec![false; dimension];
        for condition in &self.boundary_conditions {
            if seen[condition.state] {
                return Err(BvpApiError::InvalidConfiguration(
                    "a state may have only one discovery-facade boundary condition".into(),
                ));
            }
            seen[condition.state] = true;
        }
        Ok(())
    }
}

/// Configuration shared by the discovery facade.
#[derive(Clone, Debug)]
pub struct BvpSolverOptions {
    pub solver: BvpSolverKind,
    pub backend: BvpMatrixBackend,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for BvpSolverOptions {
    fn default() -> Self {
        Self {
            solver: BvpSolverKind::Damped,
            backend: BvpMatrixBackend::Sparse,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// Fluent construction of the lightweight solver-discovery API.
#[derive(Clone, Debug)]
pub struct BvpSolverBuilder {
    problem: BvpProblem,
    options: BvpSolverOptions,
}

impl BvpSolverBuilder {
    pub fn new(problem: BvpProblem) -> Self {
        Self {
            problem,
            options: BvpSolverOptions::default(),
        }
    }

    pub fn with_solver(mut self, solver: BvpSolverKind) -> Self {
        self.options.solver = solver;
        self
    }

    pub fn with_backend(mut self, backend: BvpMatrixBackend) -> Self {
        self.options.backend = backend;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.options.tolerance = tolerance;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.options.max_iterations = max_iterations;
        self
    }

    /// Construct one selected solver. Detailed solver-specific options remain
    /// available on `BVP_Damp` and `BVP_sci` directly.
    pub fn build(self) -> Result<BvpSolver, BvpApiError> {
        self.problem.validate()?;
        if !self.options.tolerance.is_finite() || self.options.tolerance <= 0.0 {
            return Err(BvpApiError::InvalidConfiguration(
                "BVP tolerance must be finite and positive".into(),
            ));
        }
        if self.options.max_iterations == 0 {
            return Err(BvpApiError::InvalidConfiguration(
                "BVP max_iterations must be positive".into(),
            ));
        }
        let problem = self.problem;
        let options = self.options;
        let backend = match options.solver {
            BvpSolverKind::Damped => {
                let solver = crate::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP::new_with_options(
                    problem.equations.clone(),
                    problem.initial_guess.clone(),
                    problem.state_names.clone(),
                    problem.independent_variable.clone(),
                    problem.to_legacy_boundary_conditions(),
                    problem.mesh[0],
                    *problem.mesh.last().unwrap(),
                    problem.mesh.len(),
                    crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions::default()
                        .with_abs_tolerance(options.tolerance)
                        .with_max_iterations(options.max_iterations)
                        .with_generated_backend_config(
                            crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::default(),
                        )
                        .with_matrix_backend(match options.backend {
                            BvpMatrixBackend::Dense => crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Dense,
                            BvpMatrixBackend::Sparse => crate::symbolic::codegen::codegen_provider_api::MatrixBackend::SparseCol,
                            BvpMatrixBackend::Banded { .. } => crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Banded,
                        }),
                );
                BvpSolverBackend::Damped(solver)
            }
            BvpSolverKind::Frozen => {
                let solver =
                    crate::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP::new_with_options(
                        problem.equations.clone(),
                        problem.initial_guess.clone(),
                        problem.state_names.clone(),
                        problem.independent_variable.clone(),
                        problem.to_legacy_boundary_conditions(),
                        problem.mesh[0],
                        *problem.mesh.last().unwrap(),
                        problem.mesh.len(),
                        FrozenSolverOptions::new(
                            "Frozen".into(),
                            None,
                            None,
                            options.backend.legacy_name(),
                            options.tolerance,
                            options.max_iterations,
                        ),
                    );
                BvpSolverBackend::Frozen(solver)
            }
            BvpSolverKind::SciPyLike => {
                BvpSolverBackend::SciPyLike(build_sci_solver(&problem, &options)?)
            }
        };
        Ok(BvpSolver {
            problem,
            selected: options.solver,
            backend,
            last_solution: None,
        })
    }
}

impl BvpProblem {
    fn to_legacy_boundary_conditions(&self) -> HashMap<String, Vec<(usize, f64)>> {
        self.boundary_conditions
            .iter()
            .map(|condition| {
                let side = match condition.side {
                    BvpBoundarySide::Left => 0,
                    BvpBoundarySide::Right => 1,
                };
                (
                    self.state_names[condition.state].clone(),
                    vec![(side, condition.value)],
                )
            })
            .collect()
    }
}

fn build_sci_solver(
    problem: &BvpProblem,
    options: &BvpSolverOptions,
) -> Result<BvpSciSolver, BvpApiError> {
    let dimension = problem.state_names.len();
    let mut initial_state = vec![0.0; dimension * problem.mesh.len()];
    for node in 0..problem.mesh.len() {
        for state in 0..dimension {
            initial_state[node * dimension + state] = problem.initial_guess[(state, node)];
        }
    }
    let conditions = problem.boundary_conditions.clone();
    let builder = BvpSciSolver::builder(problem.equations.clone(), problem.state_names.clone())
        .with_independent_variable(problem.independent_variable.clone())
        .with_mesh_and_initial_state(problem.mesh.clone(), initial_state)
        .with_matrix_layout(options.backend.sci_layout())
        .with_tolerance(options.tolerance)
        .with_limits(
            problem.mesh.len().saturating_mul(16).max(128),
            options.max_iterations,
            10,
        )
        .with_boundary_callback(move |ya, yb, _parameters, output| {
            output.fill(0.0);
            for condition in &conditions {
                let state = condition.state;
                output[state] = match condition.side {
                    BvpBoundarySide::Left => ya[state] - condition.value,
                    BvpBoundarySide::Right => yb[state] - condition.value,
                };
            }
            Ok(())
        });
    builder.build().map_err(BvpApiError::SciPy)
}

/// Unified solution returned by the discovery facade.
#[derive(Clone, Debug)]
pub struct BvpSolution {
    pub mesh: Vec<f64>,
    pub values: DMatrix<f64>,
    pub solver: BvpSolverKind,
}

/// Typed errors at the facade boundary. Detailed backend errors remain
/// available through the specialized solver APIs.
#[derive(Debug, Error)]
pub enum BvpApiError {
    #[error("invalid BVP configuration: {0}")]
    InvalidConfiguration(String),
    #[error("{solver:?} BVP backend failed: {message}")]
    Backend {
        solver: BvpSolverKind,
        message: String,
    },
    #[error("SciPy-like BVP backend failed: {0}")]
    SciPy(#[from] BvpSciNewError),
    #[error("BVP solver did not produce a converged solution")]
    NotConverged,
    #[error("no BVP solution is available before solve")]
    SolutionUnavailable,
    #[error("postprocessing failed: {0}")]
    Postprocess(String),
}

enum BvpSolverBackend {
    Damped(crate::numerical::BVP_Damp::NR_Damp_solver_damped::NRBVP),
    Frozen(crate::numerical::BVP_Damp::NR_Damp_solver_frozen::NRBVP),
    SciPyLike(BvpSciSolver),
}

/// Solver-discovery facade for quick algorithm/layout experiments.
pub struct BvpSolver {
    problem: BvpProblem,
    selected: BvpSolverKind,
    backend: BvpSolverBackend,
    last_solution: Option<BvpSolution>,
}

impl BvpSolver {
    pub fn builder(problem: BvpProblem) -> BvpSolverBuilder {
        BvpSolverBuilder::new(problem)
    }

    pub fn selected_solver(&self) -> BvpSolverKind {
        self.selected
    }

    pub fn solve(&mut self) -> Result<&BvpSolution, BvpApiError> {
        let solution = match &mut self.backend {
            BvpSolverBackend::Damped(solver) => {
                solver.try_solver().map_err(|err| BvpApiError::Backend {
                    solver: BvpSolverKind::Damped,
                    message: format!("{err:?}"),
                })?;
                let values = solver.get_result().ok_or(BvpApiError::NotConverged)?;
                BvpSolution {
                    mesh: solver.x_mesh.as_slice().to_vec(),
                    values,
                    solver: BvpSolverKind::Damped,
                }
            }
            BvpSolverBackend::Frozen(solver) => {
                solver.try_solver().map_err(|err| BvpApiError::Backend {
                    solver: BvpSolverKind::Frozen,
                    message: format!("{err:?}"),
                })?;
                let values = solver.get_result().ok_or(BvpApiError::NotConverged)?;
                BvpSolution {
                    mesh: solver.x_mesh.as_slice().to_vec(),
                    values,
                    solver: BvpSolverKind::Frozen,
                }
            }
            BvpSolverBackend::SciPyLike(solver) => {
                let result = solver.solve()?;
                solution_from_sci(result, BvpSolverKind::SciPyLike)
            }
        };
        self.last_solution = Some(solution);
        Ok(self.last_solution.as_ref().unwrap())
    }

    pub fn solution(&self) -> Result<&BvpSolution, BvpApiError> {
        self.last_solution
            .as_ref()
            .ok_or(BvpApiError::SolutionUnavailable)
    }

    pub fn postprocess_dataset(&self) -> Result<PostprocessDataset, PostprocessError> {
        let solution = self
            .last_solution
            .as_ref()
            .ok_or_else(|| PostprocessError::InvalidDataset("BVP has not been solved".into()))?;
        PostprocessDataset::new(
            self.problem.independent_variable.clone(),
            self.problem.state_names.clone(),
            nalgebra::DVector::from_vec(solution.mesh.clone()),
            solution.values.clone(),
        )
    }
}

fn solution_from_sci(solution: BvpSciSolution, solver: BvpSolverKind) -> BvpSolution {
    let dimension = solution.dimension();
    let node_count = solution.x.len();
    let values = DMatrix::from_fn(dimension, node_count, |state, node| {
        solution.y[node * dimension + state]
    });
    BvpSolution {
        mesh: solution.x,
        values,
        solver,
    }
}

#[cfg(test)]
mod typed_facade_tests {
    use super::{
        BvpApiError, BvpBoundaryCondition, BvpBoundarySide, BvpMatrixBackend, BvpProblem,
        BvpSolver, BvpSolverKind,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::DMatrix;

    #[test]
    fn typed_problem_rejects_duplicate_state_conditions() {
        let result = BvpProblem::new(
            vec![Expr::parse_expression("y")],
            vec!["y".into()],
            "x",
            vec![0.0, 1.0],
            DMatrix::zeros(1, 2),
            vec![
                BvpBoundaryCondition {
                    state: 0,
                    side: BvpBoundarySide::Left,
                    value: 0.0,
                },
                BvpBoundaryCondition {
                    state: 0,
                    side: BvpBoundarySide::Right,
                    value: 0.0,
                },
            ],
        );
        assert!(matches!(result, Err(BvpApiError::InvalidConfiguration(_))));
    }

    fn simple_problem() -> BvpProblem {
        BvpProblem::new(
            vec![Expr::parse_expression("y")],
            vec!["y".into()],
            "x",
            vec![0.0, 0.5, 1.0],
            DMatrix::zeros(1, 3),
            vec![BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 0.0,
            }],
        )
        .expect("simple typed BVP should validate")
    }

    #[test]
    fn typed_facade_builds_all_solver_routes() {
        for (kind, backend) in [
            (BvpSolverKind::Damped, BvpMatrixBackend::Sparse),
            (BvpSolverKind::Frozen, BvpMatrixBackend::Dense),
            (BvpSolverKind::SciPyLike, BvpMatrixBackend::Dense),
        ] {
            let solver = BvpSolver::builder(simple_problem())
                .with_solver(kind)
                .with_backend(backend)
                .build()
                .expect("typed facade route should build");
            assert_eq!(solver.selected_solver(), kind);
        }
    }
}
