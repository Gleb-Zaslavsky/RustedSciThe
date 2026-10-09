//! Comprehensive tests for BVP configuration parsing
//!
//! Tests cover:
//! - Basic configuration parsing without bounds
//! - Full configuration with bounds and tolerances
//! - File-based configuration loading
//! - Complex settings with adaptive grid and pseudonyms
//! - Postprocessing configuration and execution

use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{AdaptiveGridConfig, NRBVP, SolverParams};
use crate::command_interpreter::task_parser::ParseErrorKind;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy,
};
use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use crate::somelinalg::banded::LinearSolverPolicy;
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use std::collections::HashMap;

use super::*;
use tempfile::tempdir;

#[test]
fn damped_adapter_keeps_typed_document_errors() {
    let error = parse_bvp_damped_task_from_str("task\nsolver Damped")
        .expect_err("malformed task header must fail");
    match error {
        BvpDampedTaskError::Document(error) => {
            assert_eq!(error.kind, ParseErrorKind::InvalidSection)
        }
        other => panic!("expected typed document error, got {other:?}"),
    }
}

use nalgebra::{DMatrix, DVector};

fn parse_document_for_damped(text: &str) -> DocumentMap {
    let mut parser = DocumentParser::new(text.to_owned());
    let _ = parser.parse_document();
    parser.keys_to_lower_case(Some(vec![
        "bounds".to_string(),
        "rel_tolerance".to_string(),
    ]));
    parser
        .get_result()
        .expect("parser produced no result")
        .clone()
}

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

#[test]
fn bvp_damped_solver_settings_spec_parses_typed_settings() {
    let input = r#"
        solver_settings
        scheme: forward
        strategy: Damped
        method: Dense
        linear_sys_method: None
        abs_tolerance: 1e-6
        max_iterations: 120
        loglevel: Some(info)
        dont_save_log: false
        bounds
        y: -1.0, 1.0
        rel_tolerance
        y: 1e-4
        strategy_params
        max_jac: Some(3)
        max_damp_iter: Some(10)
        damp_factor: Some(0.5)
        adaptive_strategy
        version: 1
        max_refinements: 2
        grid_refinement
        pearson: [0.1, 1.5]
        "#;
    let result = parse_document_for_damped(input);
    let spec = parse_bvp_damped_solver_settings_from_document(&result).unwrap();
    assert_eq!(spec.scheme, "forward");
    assert_eq!(spec.strategy, "Damped");
    assert_eq!(spec.method, "Dense");
    assert_eq!(spec.linear_sys_method, None);
    assert_eq!(spec.abs_tolerance, 1e-6);
    assert_eq!(spec.max_iterations, 120);
    assert_eq!(spec.loglevel, Some("info".to_string()));
    assert!(!spec.dont_save_log);
    assert_eq!(spec.bounds.as_ref().unwrap()["y"], (-1.0, 1.0));
    assert_eq!(spec.rel_tolerance.as_ref().unwrap()["y"], 1e-4);
    assert_eq!(
        spec.strategy_params,
        Some(SolverParams {
            max_jac: Some(3),
            max_damp_iter: Some(10),
            damp_factor: Some(0.5),
            adaptive: Some(AdaptiveGridConfig {
                version: 1,
                max_refinements: 2,
                grid_method: GridRefinementMethod::Pearson(0.1, 1.5),
            }),
        })
    );
}

#[test]
fn bvp_damped_solver_settings_spec_builds_generated_backend_config() {
    let input = r#"
        solver_settings
        scheme: trapezoid
        strategy: Damped
        method: Banded
        linear_sys_method: faithful
        abs_tolerance: 1e-6
        max_iterations: 120
        generated_backend: banded_aot_tcc
        matrix_backend: banded
        backend_policy: prefer_aot_then_lambdify
        symbolic_backend: AtomView
        aot_codegen_backend: C
        aot_c_compiler: tcc
        aot_build_policy: build_if_missing
        aot_build_profile: release
        aot_compile_preset: dev_fastest
        aot_execution_policy: sequential
        banded_linear_solver: faithful
        refinement_steps: 0
        "#;
    let result = parse_document_for_damped(input);
    let spec = parse_bvp_damped_solver_settings_from_document(&result).unwrap();
    let options = build_bvp_damped_solver_options_from_spec(&spec).unwrap();

    assert_eq!(
        options.generated_backend_config.backend_policy_override,
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        options.generated_backend_config.matrix_backend_override,
        Some(MatrixBackend::Banded)
    );
    assert_eq!(
        options.generated_backend_config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        options.generated_backend_config.aot_codegen_backend,
        AotCodegenBackend::C
    );
    assert_eq!(
        options.generated_backend_config.aot_c_compiler.as_deref(),
        Some("tcc")
    );
    assert_eq!(
        options.generated_backend_config.aot_build_policy,
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
    assert_eq!(
        options.generated_backend_config.aot_execution_policy,
        AotExecutionPolicy::SequentialOnly
    );
    assert_eq!(
        options
            .generated_backend_config
            .banded_linear_solver_config
            .policy,
        LinearSolverPolicy::ForceBanded
    );
    assert_eq!(
        options
            .generated_backend_config
            .banded_linear_solver_config
            .iterative_refinement_steps,
        0
    );
}

#[test]
fn bvp_damped_try_apply_solver_settings_populates_generated_backend_config() {
    let input = r#"
        solver_settings
        scheme: forward
        strategy: Damped
        method: Sparse
        linear_sys_method: None
        abs_tolerance: 1e-6
        max_iterations: 100
        generated_backend: sparse_aot_tcc
        backend_policy: prefer_aot_then_lambdify
        symbolic_backend: AtomView
        aot_codegen_backend: C
        aot_c_compiler: tcc
        aot_build_policy: build_if_missing
        aot_build_profile: release
        aot_execution_policy: sequential
        "#;
    let result = parse_document_for_damped(input);
    let spec = parse_bvp_damped_solver_settings_from_document(&result).unwrap();

    let mut nr = NRBVP::default();
    nr.try_apply_bvp_damped_solver_settings(&spec).unwrap();

    assert_eq!(
        nr.generated_backend_config().backend_policy_override,
        Some(BackendSelectionPolicy::PreferAotThenLambdify)
    );
    assert_eq!(
        nr.generated_backend_config().symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        nr.generated_backend_config().aot_codegen_backend,
        AotCodegenBackend::C
    );
    assert_eq!(
        nr.generated_backend_config().aot_c_compiler.as_deref(),
        Some("tcc")
    );
    assert_eq!(
        nr.generated_backend_config().aot_build_policy,
        AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release
        }
    );
    assert_eq!(
        nr.generated_backend_config().aot_execution_policy,
        AotExecutionPolicy::SequentialOnly
    );
}

#[test]
fn bvp_damped_postprocessing_spec_defaults_when_section_missing() {
    let result = parse_document_for_damped(
        r#"
            solver_settings
            scheme: forward
            strategy: Damped
            method: Dense
            linear_sys_method: None
            abs_tolerance: 1e-6
            max_iterations: 100
            "#,
    );
    let spec = parse_bvp_damped_postprocessing_from_document(&result).unwrap();
    assert!(!spec.plot);
    assert!(!spec.gnuplot);
    assert!(!spec.save);
    assert!(!spec.save_to_csv);
    assert_eq!(spec.filename, None);
}

#[test]
fn bvp_damped_full_task_spec_parses_settings_and_postprocessing() {
    let result = parse_document_for_damped(
        r#"
            solver_settings
            scheme: trapezoid
            strategy: Frozen
            method: Sparse
            linear_sys_method: Some(faithful)
            abs_tolerance: 1e-7
            max_iterations: 42
            loglevel: Some(info)
            dont_save_log: false

            postprocessing
            plot: true
            gnuplot: true
            save: true
            save_to_csv: true
            filename: damped_task_output
            "#,
    );

    let spec = parse_bvp_damped_task_from_document(&result).unwrap();
    assert_eq!(spec.solver_settings.scheme, "trapezoid");
    assert_eq!(spec.solver_settings.strategy, "Frozen");
    assert_eq!(spec.solver_settings.method, "Sparse");
    assert_eq!(
        spec.solver_settings.linear_sys_method.as_deref(),
        Some("faithful")
    );
    assert_eq!(spec.solver_settings.abs_tolerance, 1e-7);
    assert_eq!(spec.solver_settings.max_iterations, 42);
    assert_eq!(spec.solver_settings.loglevel.as_deref(), Some("info"));
    assert!(!spec.solver_settings.dont_save_log);
    assert!(spec.postprocessing.plot);
    assert!(spec.postprocessing.gnuplot);
    assert!(spec.postprocessing.save);
    assert!(spec.postprocessing.save_to_csv);
    assert_eq!(
        spec.postprocessing.filename.as_deref(),
        Some("damped_task_output")
    );

    let mut nr = NRBVP::default();
    nr.apply_bvp_damped_solver_settings(&spec.solver_settings);
    assert_eq!(nr.scheme, "trapezoid");
    assert_eq!(nr.strategy, "Frozen");
    assert_eq!(nr.method, "Sparse");
    assert_eq!(nr.linear_sys_method.as_deref(), Some("faithful"));
}

#[test]
fn bvp_damped_full_task_spec_parses_from_str() {
    let input = r#"
        solver_settings
        scheme: forward
        strategy: Damped
        method: Dense
        linear_sys_method: None
        abs_tolerance: 1e-6
        max_iterations: 100
        loglevel: Some(info)

        postprocessing
        plot: false
        gnuplot: false
        save: true
        save_to_csv: false
        filename: damped_task.txt
        "#;

    let spec = parse_bvp_damped_task_from_str(input).unwrap();
    assert_eq!(spec.solver_settings.scheme, "forward");
    assert_eq!(spec.solver_settings.strategy, "Damped");
    assert_eq!(spec.solver_settings.method, "Dense");
    assert_eq!(spec.solver_settings.linear_sys_method, None);
    assert!(spec.postprocessing.save);
    assert_eq!(
        spec.postprocessing.filename.as_deref(),
        Some("damped_task.txt")
    );
}

#[test]
fn bvp_damped_solver_settings_rejects_unknown_grid_refinement_method() {
    let result = parse_document_for_damped(
        r#"
            solver_settings
            scheme: forward
            strategy: Damped
            method: Dense
            linear_sys_method: None
            abs_tolerance: 1e-6
            max_iterations: 100
            strategy_params
            max_jac: Some(3)
            max_damp_iter: Some(10)
            damp_factor: Some(0.5)
            adaptive_strategy
            version: 1
            max_refinements: 2
            grid_refinement
            alien: [0.1, 1.5]
            "#,
    );
    let err = parse_bvp_damped_solver_settings_from_document(&result).unwrap_err();
    assert_eq!(
        err,
        BvpDampedTaskError::UnknownGridRefinementMethod("alien".to_string())
    );
}
