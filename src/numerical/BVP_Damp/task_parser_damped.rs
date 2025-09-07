//! # BVP Damped Task Parser Module
//!
//! Configuration parser for Boundary Value Problem (BVP) solver with damped Newton-Raphson method.
//! Extends the NRBVP solver with file-based and string-based configuration parsing capabilities,
//! allowing users to define solver parameters, boundary conditions, and postprocessing options
//! through structured text files or strings.
//!
//! ## Core Functionality
//! - **Configuration Parsing**: Parse solver settings from files or strings into NRBVP solver
//! - **Parameter Mapping**: Convert parsed data into solver-specific data structures
//! - **Pseudonym Support**: Handle common typos and alternative names for configuration keys
//! - **Grid Refinement**: Parse adaptive grid refinement strategies and parameters
//! - **Postprocessing**: Configure output options (plotting, saving, logging)
//! - **Template Generation**: Create configuration file templates for users
//!
//! ## Main Methods (NRBVP Implementation)
//!
//! ### Core Parsing Methods
//! - `before_solve_preprocessing()` - Initialize mesh and grid settings before solving
//! - `set_params_from_hashmap(result)` - Map parsed DocumentMap to solver parameters
//! - `set_postpocessing_from_hashmap(parser)` - Configure and execute postprocessing options
//!
//! ### String-based Parsing
//! - `parse_settings_from_str_with_exact_names(input)` - Parse from string with exact key names
//! - `parse_settings(parser)` - Parse with pseudonym support for common typos
//!
//! ### File-based Parsing
//! - `parse_file(path)` - Load configuration from file and return DocumentParser
//! - `parse_settings_with_exact_names(parser)` - Parse from DocumentParser with exact names
//!
//! ### Utility Functions
//! - `create_template_file(path)` - Generate configuration file template
//!
//! ## Configuration Structure
//!
//! ### solver_settings Section
//! - `scheme`: Discretization scheme ("forward" or "trapezoid")
//! - `method`: Matrix backend ("Dense" or "Sparse")
//! - `strategy`: Solver strategy ("Damped", "Naive", "Frozen")
//! - `linear_sys_method`: Linear system solver method (Optional)
//! - `abs_tolerance`: Absolute convergence tolerance
//! - `max_iterations`: Maximum solver iterations
//! - `loglevel`: Logging level (Optional)
//!
//! ### bounds Section
//! Variable-specific solution bounds: `variable_name: min_value, max_value`
//!
//! ### rel_tolerance Section
//! Variable-specific relative tolerances: `variable_name: tolerance_value`
//!
//! ### strategy_params Section
//! - `max_jac`: Maximum iterations with old Jacobian (Optional)
//! - `max_damp_iter`: Maximum damped iterations (Optional)
//! - `damp_factor`: Damping factor reduction (Optional)
//!
//! ### adaptive_strategy Section
//! - `version`: Grid refinement version
//! - `max_refinements`: Maximum refinement iterations
//!
//! ### grid_refinement Section
//! Grid refinement methods with parameters:
//! - `doubleoints`: Double grid points
//! - `easy`: Simple refinement with single parameter
//! - `grcarsmooke`: Grcar-Smooke method with 3 parameters
//! - `pearson`: Pearson method with 2 parameters
//! - `twopnt`: Two-point method with 3 parameters
//!
//! ### postprocessing Section
//! - `plot`: Enable plotting (boolean)
//! - `gnuplot`: Enable gnuplot output (boolean)
//! - `save`: Save results to file (boolean)
//! - `save_to_csv`: Save as CSV (boolean)
//! - `dont_save_log`: Disable log saving (boolean)
//!
//! ## Non-obvious Features and Tips
//!
//! ### 1. Pseudonym System for User-Friendly Input
//! The parser supports common typos and alternative names:
//! - "solver_settings" ↔ "solve_settings", "solving_settings"
//! - "bounds" ↔ "bound", "boundary", "boundaries"
//! - "abs_tolerance" ↔ "abs_tol", "absolute_error", "absolute_tolerance"
//! - "rel_tolerance" ↔ "rel_tol", "relative_error", "relevant_error"
//!
//! ### 2. Variable Name Preservation
//! Uses `keys_to_lower_case()` with exceptions for "bounds" and "rel_tolerance" sections
//! to preserve user-defined variable names while normalizing configuration keys.
//!
//! ### 3. Grid Refinement Method Parsing
//! Complex parsing logic that:
//! - Extracts method name from section key
//! - Parses vector parameters for each method
//! - Maps string names to GridRefinementMethod enum variants
//! - Handles different parameter counts per method
//!
//! ### 4. Optional Value Handling
//! Extensive use of `as_option_*()` methods to handle nullable configuration values,
//! allowing users to specify `None` or `Some(value)` in configuration files.
//!
//! ### 5. Postprocessing Automation
//! `set_postpocessing_from_hashmap()` not only parses settings but immediately executes
//! the requested postprocessing actions (plotting, saving, etc.).
//!
//! ### 6. Error Handling Strategy
//! Uses `expect()` with descriptive messages for required fields and graceful
//! `unwrap_or()` defaults for optional fields to provide clear error feedback.
//!
//! ### 7. Mesh Initialization
//! `before_solve_preprocessing()` creates uniform mesh from t0, t_end, and n_steps,
//! and determines if adaptive grid refinement is enabled based on strategy_params.
//!
//! ### 8. Template System
//! `create_template_file()` provides a structured template with comments explaining
//! each configuration option, making it easier for users to create valid config files.
//!
//! ## Usage Examples
//!
//! ```rust
//! // Parse from string with exact names
//! let mut solver = NRBVP::default();
//! solver.parse_settings_from_str_with_exact_names(config_string);
//!
//! // Parse from file with pseudonym support
//! let mut solver = NRBVP::default();
//! let mut parser = solver.parse_file(Some(config_path)).unwrap();
//! solver.parse_settings(&mut parser).unwrap();
//!
//! // Setup and solve
//! solver.before_solve_preprocessing();
//! solver.solve();
//! solver.set_postpocessing_from_hashmap(&mut parser);
//! ```

use crate::Utils::task_parser::{DocumentMap, DocumentParser};
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{AdaptiveGridConfig, NRBVP, SolverParams};
use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use nalgebra::DVector;
use std::collections::HashMap;
impl NRBVP {
    /// in standard NRBVP approach this operation is made in the new(..) method but when
    /// but if params are passed from the hashmap (and may be by parsing task files), it is done here

    pub fn before_solve_preprocessing(&mut self) {
        let t0 = self.t0;
        let t_end = self.t_end;
        let n_steps = self.n_steps;
        let h = (t_end - t0) / n_steps as f64;
        let T_list: Vec<f64> = (0..n_steps + 1)
            .map(|i| t0 + (i as f64) * h)
            .collect::<Vec<_>>();

        self.x_mesh = DVector::from_vec(T_list);

        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        let new_grid_enabled_: bool = if let Some(ref params) = self.strategy_params {
            params.adaptive.is_some()
        } else {
            false
        };
        self.new_grid_enabled = new_grid_enabled_;
    }
    pub fn set_params_from_hashmap(&mut self, result: DocumentMap) {
        let solver_settings = result
            .get("solver_settings")
            .expect("Failed to get solver settings");
        let scheme = solver_settings
            .get("scheme")
            .expect("Failed to get scheme")
            .clone()
            .unwrap()[0]
            .as_string()
            .expect("Failed to get scheme as string")
            .clone();
        self.scheme = scheme;
        let strategy = solver_settings
            .get("strategy")
            .expect("Failed to get strategy")
            .clone()
            .unwrap()[0]
            .as_string()
            .expect("Failed to get strategy as string")
            .clone();
        self.strategy = strategy;
        let bind = solver_settings
            .clone()
            .get("linear_sys_method")
            .expect("Failed to get linear_sys_method")
            .clone()
            .unwrap();
        let linear_sys_method = bind[0].as_option_string();
        self.linear_sys_method = linear_sys_method.cloned();
        let method = solver_settings
            .get("method")
            .expect("Failed to get method")
            .clone()
            .unwrap()[0]
            .as_string()
            .expect("Failed to get method as string")
            .clone();
        self.method = method;
        let abs_tolerance = solver_settings
            .get("abs_tolerance")
            .expect("Failed to get abs_tolerance")
            .clone()
            .unwrap()[0]
            .as_float()
            .expect("Failed to get abs_tolerance as float");
        self.abs_tolerance = abs_tolerance;
        let max_iterations = solver_settings
            .get("max_iterations")
            .expect("Failed to get max_iterations")
            .clone()
            .unwrap()[0]
            .as_usize()
            .expect("Failed to get max_iterations as usize");
        self.max_iterations = max_iterations;
        let loglevel = solver_settings
            .get("loglevel")
            .expect("Failed to get loglevel")
            .clone()
            .unwrap()[0]
            .as_string()
            .cloned();
        self.loglevel = loglevel;
        let dont_save_log = if let Some(dont_save_log) = solver_settings.get("dont_save_log") {
            dont_save_log.clone().unwrap()[0]
                .as_boolean()
                .expect("Failed to get dont_save_log as bool")
        } else {
            true
        };
        self.dont_save_log(dont_save_log);
        if let Some(bounds) = result.get("bounds") {
            let bounds: HashMap<String, (f64, f64)> = bounds
                .iter()
                .map(|(key, value)| {
                    let binding = value.clone().unwrap();
                    let value0 = binding[0].as_float().unwrap();
                    let value1 = binding[1].as_float().unwrap();
                    (key.to_owned(), (value0, value1))
                })
                .collect();
            self.Bounds = Some(bounds);
        } else {
            self.Bounds = None
        };
        if let Some(rel_tolerance) = result.get("rel_tolerance") {
            let rel_tolerance: HashMap<String, f64> = rel_tolerance
                .iter()
                .map(|(key, value)| {
                    let value = value.clone().unwrap()[0]
                        .as_float()
                        .expect("Failed to get rel_tolerance as float");
                    (key.to_owned(), value)
                })
                .collect();
            self.rel_tolerance = Some(rel_tolerance);
        } else {
            self.rel_tolerance = None
        };
        if let Some(strategy_params) = result.get("strategy_params") {
            let max_jac = strategy_params
                .get("max_jac")
                .expect("Failed to get max_jac")
                .clone()
                .unwrap()[0]
                .as_option_usize();
            let max_damp_iter = strategy_params
                .get("max_damp_iter")
                .expect("Failed to get maxDampIter")
                .clone()
                .unwrap()[0]
                .as_option_usize();
            let damp_factor = strategy_params
                .get("damp_factor")
                .expect("Failed to get DampFacor")
                .clone()
                .unwrap()[0]
                .as_option_float();
            let adaptive = if let Some(adaptive_strategy) = result.get("adaptive_strategy") {
                let version = adaptive_strategy
                    .get("version")
                    .expect("Failed to get version")
                    .clone()
                    .unwrap()[0]
                    .as_usize()
                    .expect("Failed to get version as string")
                    .clone();
                let max_refinements = adaptive_strategy
                    .get("max_refinements")
                    .expect("Failed to get max_refinements")
                    .clone()
                    .unwrap()[0]
                    .as_usize()
                    .expect("Failed to get max_refinements as int");
                let grid_method = result.get("grid_refinement").expect("0").clone();
                let method_name = grid_method.keys().next().unwrap().to_owned();
                let method_params = grid_method.get(&method_name).unwrap().clone().unwrap()[0]
                    .as_vector()
                    .expect("Failed to get grid_refinement as vector")
                    .clone();
                let grid_method: GridRefinementMethod = match method_name.as_str() {
                    "doubleoints" => GridRefinementMethod::DoublePoints,
                    "easy" => GridRefinementMethod::Easy(method_params[0]),
                    "grcarsmooke" => GridRefinementMethod::GrcarSmooke(
                        method_params[0],
                        method_params[1],
                        method_params[2],
                    ),
                    "pearson" => GridRefinementMethod::Pearson(method_params[0], method_params[1]),
                    "twopnt" => GridRefinementMethod::TwoPoint(
                        method_params[0],
                        method_params[1],
                        method_params[2],
                    ),

                    _ => {
                        panic!("Unknown grid refinement method: {}", method_name)
                    }
                };
                let adaptive = AdaptiveGridConfig {
                    version,

                    max_refinements,

                    grid_method,
                };

                Some(adaptive)
            } else {
                None
            };

            let solver_params = SolverParams {
                max_jac,
                max_damp_iter,
                damp_factor,
                adaptive,
            };
            self.strategy_params = Some(solver_params);
        } else {
            self.strategy_params = None
        }
    }

    pub fn set_postpocessing_from_hashmap(&mut self, parser: &mut DocumentParser) {
        let result: DocumentMap = parser.get_result().unwrap().clone();
        let solver_settings = result
            .get("postprocessing")
            .expect("Failed to get postpocessing");
        let plot_flag = if let Some(plot) = solver_settings.get("plot") {
            plot.clone().unwrap()[0]
                .as_boolean()
                .expect("Failed to get plot as bool")
        } else {
            false
        };
        let gnuplot_flag = if let Some(gnuplotflag) = solver_settings.get("gnuplot") {
            gnuplotflag.clone().unwrap()[0]
                .as_boolean()
                .expect("Failed to get gnuplot as bool")
        } else {
            false
        };
        let save_flag = if let Some(save) = solver_settings.get("save") {
            save.clone().unwrap()[0]
                .as_boolean()
                .expect("Failed to get save as bool")
        } else {
            false
        };
        let save_to_csv = if let Some(save) = solver_settings.get("save_to_csv") {
            save.clone().unwrap()[0]
                .as_boolean()
                .expect("Failed to get save_to_csv as bool")
        } else {
            false
        };

        let name = if let Some(name) = solver_settings.get("filename") {
            name.clone().unwrap()[0].as_string().cloned()
        } else {
            None
        };

        if plot_flag {
            self.plot_result();
        }
        if gnuplot_flag {
            self.gnuplot_result()
        };
        if save_flag {
            self.save_to_file(name.clone());
        };
        if save_to_csv {
            self.save_to_csv(name);
        };
    }

    pub fn parse_settings_from_str_with_exact_names(&mut self, input: &str) {
        let mut parser = DocumentParser::new(input.to_owned());

        let _ = parser.parse_document();
        // keys inside bounds header of rel_tolerance don't need to be converted to lowecase
        // because they contain problem variables names which are arbituary
        parser.keys_to_lower_case(Some(vec![
            "bounds".to_string(),
            "rel_tolerance".to_string(),
        ]));
        let result: DocumentMap = parser.get_result().expect("Failed to get result").clone();
        self.set_params_from_hashmap(result);
    }

    pub fn parse_file(
        &mut self,
        path: Option<std::path::PathBuf>,
    ) -> Result<DocumentParser, String> {
        let mut parser = DocumentParser::new(String::new());
        parser.setting_from_file(path)?;
        Ok(parser)
    }

    pub fn parse_settings_with_exact_names(
        &mut self,
        parser: &mut DocumentParser,
    ) -> Result<(), String> {
        let _ = parser.parse_document();
        parser.keys_to_lower_case(Some(vec![
            "bounds".to_string(),
            "rel_tolerance".to_string(),
        ]));
        let result: DocumentMap = parser
            .get_result()
            .ok_or("No result after parsing")?
            .clone();
        self.set_params_from_hashmap(result);
        Ok(())
    }
    /// Parses the settings from a string and covers some common typos
    pub fn parse_settings(&mut self, parser: &mut DocumentParser) -> Result<(), String> {
        let headers_pseudonims: HashMap<String, Vec<String>> = HashMap::from([
            (
                "solver_settings".to_string(),
                vec!["solve_settings".to_string(), "solving_settings".to_string()],
            ),
            (
                "bounds".to_string(),
                vec![
                    "bound".to_string(),
                    "boundary".to_string(),
                    "boundaries".to_string(),
                ],
            ),
            (
                "rel_tolerance".to_string(),
                vec![
                    "rel_tol".to_string(),
                    "rel_error".to_string(),
                    "relative_error".to_string(),
                    "relevant_error".to_string(),
                    "relative_tolerance".to_string(),
                ],
            ),
            (
                "strategy_params".to_string(),
                vec![
                    "solver_params".to_string(),
                    "strategy_parameters".to_string(),
                    "solver_parameters".to_string(),
                ],
            ),
            (
                "adaptive_strategy".to_string(),
                vec!["adaptive".to_string(), "adaptive_settings".to_string()],
            ),
            (
                "grid_refinement".to_string(),
                vec![
                    "grid".to_string(),
                    "grid_ref".to_string(),
                    "grid_refinement_method".to_string(),
                ],
            ),
        ]);
        let field_name_pseudonims: HashMap<String, Vec<String>> = HashMap::from([
            (
                "abs_tolerance".to_string(),
                vec![
                    "abs_tol".to_string(),
                    "abs_error".to_string(),
                    "absolute_error".to_string(),
                    "absolute_tolerance".to_string(),
                ],
            ),
            (
                "max_iterations".to_string(),
                vec![
                    "max_iter".to_string(),
                    "max_iterations".to_string(),
                    "max_iterations_number".to_string(),
                ],
            ),
        ]);

        parser.with_pseudonims(Some(headers_pseudonims), Some(field_name_pseudonims));
        let _ = parser.parse_document();
        let result: DocumentMap = parser
            .get_result()
            .ok_or("No result after parsing")?
            .clone();
        self.set_params_from_hashmap(result);
        Ok(())
    }
}

/// Creates a configuration file template with all available options and comments
///
/// Generates a structured template showing all configuration sections and their expected format.
/// Users can copy this template and fill in their specific values.
pub fn create_template_file(path: Option<std::path::PathBuf>) {
    let form = r#"
    #####################ADVANCED SETTINGS#######################
    // BVP Solver Configuration Template
    solver_settings
    // Discretization scheme - "forward" or "trapezoid"
    scheme: forward
    // Matrix backend - "Dense" or "Sparse"
    method: Dense
    // Solver strategy - "Damped", "Naive", or "Frozen"
    strategy: Damped
    // Linear system method (optional) - None or specific method
    linear_sys_method: None
    // Absolute convergence tolerance
    abs_tolerance: 1e-6
    // Maximum solver iterations
    max_iterations: 100
    // Logging level (optional) - None or Some(info/warn/error)
    loglevel: Some(info)
    // Disable log file saving
     dont_save_log: true
    // Solution bounds for each variable (variable_name: min, max)
    bounds
    // var1: -10.0, 10.0
    // var2: -5.0, 5.0
    
    // Relative tolerance for each variable
    rel_tolerance
    // var1: 1e-4
    // var2: 1e-4
   
    // Strategy-specific parameters
    strategy_params
        // Maximum iterations with old Jacobian (optional)
        max_jac: Some(3)
        // Maximum damped iterations (optional)
        max_damp_iter: Some(5)
        // Damping factor reduction (optional)
        damp_factor: Some(0.5)
        
    // Adaptive grid refinement settings (optional)
    adaptive_strategy
        // Refinement version
        version: 1
        // Maximum refinement iterations
        max_refinements: 3
        
    // Grid refinement method and parameters
    grid_refinement
        // Available methods:
        // doubleoints: []
        // easy: [parameter]
        // grcarsmooke: [param1, param2, param3]
        // pearson: [param1, param2]
        // twopnt: [param1, param2, param3]
        pearson: [0.1, 1.5]
    
    // Postprocessing options
    postprocessing
        // Enable plotting with plotters crate
        plot: false
        // Enable gnuplot output
        gnuplot: false
        // Save results to text file
        save: false
        // Save results to CSV file
        save_to_csv: false
        filename: somename

    "#;

    use std::env;
    use std::fs::File;
    use std::io::Write;

    let file_path = match path {
        Some(p) => p,
        None => {
            let mut default_path =
                env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            default_path.push("bvp_config_template.txt");
            default_path
        }
    };

    match File::create(&file_path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(form.as_bytes()) {
                eprintln!("Failed to write template file: {}", e);
            } else {
                println!("Template file created at: {:?}", file_path);
            }
        }
        Err(e) => {
            eprintln!("Failed to create template file: {}", e);
        }
    }
}
#[cfg(test)]
mod tests {
    //! Comprehensive tests for BVP configuration parsing
    //!
    //! Tests cover:
    //! - Basic configuration parsing without bounds
    //! - Full configuration with bounds and tolerances
    //! - File-based configuration loading
    //! - Complex settings with adaptive grid and pseudonyms
    //! - Postprocessing configuration and execution

    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        AdaptiveGridConfig, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    use tempfile::tempdir;

    use nalgebra::{DMatrix, DVector};
    #[test]
    fn test_BVP_with_setting_parsing_no_bounds() {
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        // also checks if keys_to_lower_case() works
        let input = "
        solver_settings
        scheme: forward
        // THIS IS COMMENT
        METHOD: Dense
        strategy: Damped
        linear_sys_method: None
        abs_tolerance: 1e-5
        # THIS IS COMMENT
        max_iterations: 100
        loglevel: Some(info)
        ";
        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
        let Bounds = HashMap::from([
            ("z".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-7.0, 7.0)),
        ]);
        let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();

        nr.parse_settings_from_str_with_exact_names(input);
        nr.rel_tolerance = Some(rel_tolerance);
        nr.Bounds = Some(Bounds);
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.dont_save_log(true);
        nr.solve();

        let solution = nr.get_result().unwrap();
        let x_mesh = &nr.x_mesh;
        println!("x_mesh = {:?}", x_mesh);
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps + 1);
        nr.gnuplot_result();
        // println!("result = {:?}", solution);
        // nr.plot_result();
    }
    #[test]
    fn test_BVP_with_setting_parsing_with_bounds() {
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10; // Dense: 200 -300ms, 400 - 2s, 800 - 22s, 1600 - 2 min,
        let input = "
        solver_settings
        scheme: forward
        method: Dense
        strategy: Damped
        linear_sys_method: None
        abs_tolerance: 1e-5
        max_iterations: 100
        loglevel: Some(info)
        bounds
        z: -10.0, 10.0
        y: -7.0, 7.0
        rel_tolerance
        z: 1e-4
        y: 1e-4
        strategy_params
        max_jac: Some(3)
        max_damp_iter: Some(10)
        damp_factor: Some(0.5)
        adaptive: None
        ";
        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();

        nr.parse_settings_from_str_with_exact_names(input);
        let rel_toleranse = nr.rel_tolerance.clone().unwrap();
        assert_eq!(rel_toleranse["z"], 1e-4);
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.dont_save_log(true);
        nr.solve();

        let solution = nr.get_result().unwrap();
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps + 1);
        // println!("result = {:?}", solution);
        nr.plot_result();
    }

    #[test]
    fn test_BVP_with_setting_from_file() {
        use std::fs::File;
        use std::io::Write;

        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10;

        // Create temporary file with settings
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("problem_config.txt");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "solver_settings").unwrap();
        writeln!(file, "scheme: forward").unwrap();
        writeln!(file, "method: Dense").unwrap();
        writeln!(file, "strategy: Damped").unwrap();
        writeln!(file, "linear_sys_method: None").unwrap();
        writeln!(file, "abs_tolerance: 1e-5").unwrap();
        writeln!(file, "max_iterations: 100").unwrap();
        writeln!(file, "loglevel: Some(info)").unwrap();
        writeln!(file, "bounds").unwrap();
        writeln!(file, "z: -10.0, 10.0").unwrap();
        writeln!(file, "y: -7.0, 7.0").unwrap();
        writeln!(file, "rel_tolerance").unwrap();
        writeln!(file, "z: 1e-4").unwrap();
        writeln!(file, "y: 1e-4").unwrap();
        writeln!(file, "strategy_params").unwrap();
        writeln!(file, "max_jac: Some(3)").unwrap();
        writeln!(file, "max_damp_iter: Some(10)").unwrap();
        writeln!(file, "damp_factor: Some(0.5)").unwrap();
        writeln!(file, "adaptive: None").unwrap();

        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();

        let mut parser = nr.parse_file(Some(file_path)).unwrap();
        nr.parse_settings_with_exact_names(&mut parser).unwrap();

        let rel_tolerance = nr.rel_tolerance.clone().unwrap();
        assert_eq!(rel_tolerance["z"], 1e-4);
        let strategy_params = nr.strategy_params.clone().unwrap();
        assert_eq!(
            strategy_params,
            SolverParams {
                max_jac: Some(3),
                max_damp_iter: Some(10),
                damp_factor: Some(0.5),
                adaptive: None
            }
        );
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.dont_save_log(true);
        nr.solve();

        let solution = nr.get_result().unwrap();
        let (n, _m) = solution.shape();
        assert_eq!(n, n_steps + 1);
    }

    #[test]
    fn test_BVP_with_setting_from_file_with_complicated_settings_and_pseudonims_and_postpoc() {
        use std::fs::File;
        use std::io::Write;

        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z^3");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 10;

        // Create temporary file with settings
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("problem_config.txt");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "solver_settings").unwrap();
        writeln!(file, "scheme: forward").unwrap();
        writeln!(file, "method: Dense").unwrap();
        writeln!(file, "strategy: Damped").unwrap();
        writeln!(file, "linear_sys_method: None").unwrap();
        writeln!(file, "absolute_tolerance: 1e-5").unwrap();
        writeln!(file, "max_iterations: 100").unwrap();
        writeln!(file, "loglevel: Some(info)").unwrap();
        writeln!(file, "dont_save_log:true").unwrap();
        writeln!(file, "bounds").unwrap();
        writeln!(file, "z: -10.0, 10.0").unwrap();
        writeln!(file, "y: -7.0, 7.0").unwrap();
        writeln!(file, "rel_tolerance").unwrap();
        writeln!(file, "z: 1e-4").unwrap();
        writeln!(file, "y: 1e-4").unwrap();
        writeln!(file, "strategy_params").unwrap();
        writeln!(file, "max_jac: Some(3)").unwrap();
        writeln!(file, "max_damp_iter: Some(10)").unwrap();
        writeln!(file, "damp_factor: Some(0.5)").unwrap();
        writeln!(file, "adaptive_strategy").unwrap();
        writeln!(file, "version: 1").unwrap();
        writeln!(file, "max_refinements: 1").unwrap();
        writeln!(file, "grid_refinement").unwrap();
        writeln!(file, "pearson: [0.01, 1.5]").unwrap();
        writeln!(file, "postprocessing").unwrap();
        writeln!(file, "gnuplot: true").unwrap();
        writeln!(file, "save_to_csv:true").unwrap();
        writeln!(file, "filename:meow").unwrap();
        //  writeln!(file, "save: true").unwrap();

        let ones = vec![0.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

        assert_eq!(&eq_system.len(), &2);

        let mut nr = NRBVP::default();
        let mut parser = nr.parse_file(Some(file_path)).unwrap();
        nr.parse_settings(&mut parser).unwrap();

        let rel_tolerance = nr.rel_tolerance.clone().unwrap();
        assert_eq!(rel_tolerance["z"], 1e-4);
        let strategy_params = nr.strategy_params.clone().unwrap();
        assert_eq!(
            strategy_params,
            SolverParams {
                max_jac: Some(3),
                max_damp_iter: Some(10),
                damp_factor: Some(0.5),
                adaptive: Some(AdaptiveGridConfig {
                    version: 1,
                    max_refinements: 1,
                    grid_method: GridRefinementMethod::Pearson(0.01, 1.5)
                })
            }
        );
        nr.eq_system = eq_system;
        nr.values = values;
        nr.arg = arg;
        nr.t0 = t0;
        nr.t_end = t_end;
        nr.n_steps = n_steps;
        nr.BorderConditions = BorderConditions;
        nr.initial_guess = initial_guess;
        nr.before_solve_preprocessing();
        nr.solve();
        println!("{:?}", parser.get_result());
        nr.set_postpocessing_from_hashmap(&mut parser);

        // let solution = nr.get_result().unwrap();
        // println!("solution {:?}", solution);
    }
}
