use super::*;
use std::fs;
use tempfile::tempdir;

fn parse_document_for_ivp(input: &str) -> DocumentMap {
    let mut parser = DocumentParser::new(input.to_string());
    let pseudonyms = default_ivp_pseudonyms();
    parser.with_pseudonims(Some(pseudonyms.0), Some(pseudonyms.1));
    parser.parse_document().expect("IVP document should parse");
    parser.keys_to_lower_case(Some(vec![
        "equations".to_string(),
        "parameters".to_string(),
        "where".to_string(),
        "substitute".to_string(),
    ]));
    parser
        .get_result()
        .expect("IVP document map should exist")
        .clone()
}

#[test]
fn ivp_task_parser_supports_file_roundtrip() {
    let dir = tempdir().expect("temp dir should be created");
    let path = dir.path().join("ivp_task.txt");
    fs::write(
        &path,
        r#"
task
solver: IVP
method: RK45

equations
arg: t
unknowns: y
rhs: -y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
step_size: 1e-3
        "#,
    )
    .expect("test task file should be written");

    let spec = parse_ivp_task_from_file(Some(path)).expect("IVP file task should parse");
    assert_eq!(
        spec.solver.method,
        IvpMethodSpec::NonStiff("RK45".to_string())
    );
    assert_eq!(spec.equations.unknowns, vec!["y".to_string()]);
    assert_eq!(spec.initial_conditions.y0, vec![1.0]);
}

#[test]
fn ivp_task_parser_supports_pair_style_equations() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
parameters: a
parameter_values: 2.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
step_size: 1e-3
"#;

    let spec = parse_ivp_task_from_str(input).expect("pair-style IVP task should parse");
    assert_eq!(spec.equations.unknowns, vec!["y".to_string()]);
    assert_eq!(spec.equations.parameter_values["a"], 2.0);
    assert_eq!(spec.initial_conditions.y0, vec![1.0]);
    assert_eq!(
        spec.solver.method,
        IvpMethodSpec::NonStiff("RK45".to_string())
    );
}

#[test]
fn ivp_task_parser_preserves_readable_parameter_section_symbol_case() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
unknowns: T
rhs: -R*T

parameters
R: 2.0

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0
"#;

    let spec = parse_ivp_task_from_str(input)
        .expect("IVP task with a readable parameter section should parse");

    assert_eq!(spec.equations.parameter_names, vec!["R"]);
    assert_eq!(spec.equations.parameter_values["R"], 2.0);
    let rhs = spec.equations.rhs[0].lambdify_borrowed_thread_safe(&["t", "T"]);
    assert!((rhs(&[0.0, 3.0]) + 6.0).abs() < 1e-12);
}

#[test]
fn ivp_task_parser_supports_where_symbolic_substitutions() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
y: heat - y

where
arr: 2.0*t
heat: arr + 1.0

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
step_size: 1e-3
"#;

    let spec = parse_ivp_task_from_str(input).expect("IVP with where substitutions should parse");
    let expr = spec.equations.rhs[0].clone();
    let f = expr.lambdify_borrowed_thread_safe(&["t", "y"]);
    let value = f(&[0.5, 1.0]);
    assert!((value - 1.0).abs() < 1e-12);
}

#[test]
fn ivp_task_parser_resolves_multilevel_where_with_numeric_parameters() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
parameters: gain
parameter_values: 3.0
y: source - y

where
base: gain * t
shifted: base + 1.0
source: 2.0 * shifted

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
step_size: 1e-3
"#;

    let spec = parse_ivp_task_from_str(input)
        .expect("IVP parser should resolve a multilevel symbolic definition chain");
    let rhs = spec.equations.rhs[0].lambdify_borrowed_thread_safe(&["t", "y"]);

    // source = 2 * (gain * t + 1), so source - y = 4 at t=0.5, y=1.
    assert!((rhs(&[0.5, 1.0]) - 4.0).abs() < 1e-12);
}

#[test]
fn ivp_task_parser_reports_bad_where_expression() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
y: heat - y

where
heat: sin(

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
step_size: 1e-3
"#;

    let err = parse_ivp_task_from_str(input)
        .expect_err("bad where expression should be reported as an IVP parser error");
    let message = err.to_string();
    assert!(message.contains("where/substitute"));
    assert!(message.contains("failed to parse symbolic expression"));
}

#[test]
fn ivp_task_runner_solves_simple_decay() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
unknowns: y
rhs: -y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
step_size: 1e-3
"#;

    let result = run_ivp_task_from_str(input).expect("simple IVP task should solve");
    assert_eq!(result.status.as_deref(), Some("finished"));
    let y = result.y_result.expect("solver should produce y_result");
    let final_y = y[(y.nrows() - 1, 0)];
    let expected = (-0.2_f64).exp();
    assert!((final_y - expected).abs() < 1e-2);
}

#[test]
fn ivp_task_runner_can_save_csv() {
    let dir = tempdir().expect("tempdir should be created");
    let csv_path = dir.path().join("ivp_task_output.csv");
    let input = format!(
        r#"
task
solver: IVP
method: RK45

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.05
y0: 1.0

solver_options
step_size: 1e-3

postprocessing
save_csv: true
csv_path: {}
"#,
        csv_path.display()
    );

    let result = run_ivp_task_from_str(&input).expect("IVP task should solve and save CSV");
    assert_eq!(result.status.as_deref(), Some("finished"));
    let contents =
        std::fs::read_to_string(&csv_path).expect("CSV file should be readable after solve");
    assert!(contents.contains("t,y"));
}

#[test]
fn ivp_task_runner_can_execute_modern_postprocessing_plan() {
    let dir = tempdir().expect("tempdir should be created");
    let txt_path = dir.path().join("ivp_task_output.txt");
    let report_path = dir.path().join("ivp_task_report.md");
    let input = format!(
        r#"
task
solver: IVP
method: RK45

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.05
y0: 1.0

solver_options
step_size: 1e-3

postprocessing
save_txt: true
txt_path: {}
write_report: true
report_path: {}
"#,
        txt_path.display(),
        report_path.display()
    );

    let result = run_ivp_task_from_str(&input).expect("IVP task should solve and postprocess");
    assert_eq!(result.status.as_deref(), Some("finished"));
    assert!(txt_path.exists());
    let report = std::fs::read_to_string(&report_path)
        .expect("report should be readable after postprocessing");
    assert!(report.contains("Solver Result Report"));
    assert!(report.contains("axis: t"));
}

#[test]
fn ivp_task_runner_supports_backward_euler() {
    let input = r#"
task
solver: IVP
method: BackwardEuler

equations
arg: t
y: -10.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
step_size: 1e-3
tolerance: 1e-8
max_iterations: 100
"#;

    let result = run_ivp_task_from_str(input).expect("Backward Euler IVP task should solve");
    assert_eq!(result.status.as_deref(), Some("finished"));
    assert!(result.y_result.is_some());
}

#[test]
fn ivp_task_runner_supports_bdf() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
y: -20.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05
"#;

    let result = run_ivp_task_from_str(input).expect("BDF IVP task should solve");
    assert_eq!(result.status.as_deref(), Some("finished"));
    assert!(result.y_result.is_some());
}

#[test]
fn ivp_task_runner_supports_radau5() {
    let input = r#"
task
solver: IVP
method: Radau5

equations
arg: t
y: -15.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05
"#;

    let result = run_ivp_task_from_str(input).expect("Radau5 IVP task should solve");
    assert_eq!(result.status.as_deref(), Some("finished"));
    assert!(result.y_result.is_some());
}

#[test]
fn ivp_task_parser_supports_lsode2_method_and_options() {
    let input = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -2.0*y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_assembly: AtomView
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: faer_sparse_lu
lsode2_native_execution: faithful_bdf_solve
lsode2_stop_variable: y
lsode2_stop_comparator: le
lsode2_stop_target: 0.5
"#;

    let spec = parse_ivp_task_from_str(input).expect("LSODE2 document should parse");
    assert_eq!(spec.solver.method, IvpMethodSpec::Lsode2);
    let lsode2 = spec
        .solver_options
        .lsode2
        .as_ref()
        .expect("LSODE2 options should be present");
    assert_eq!(
        lsode2.symbolic_assembly,
        Some(Lsode2SymbolicAssemblyBackend::AtomView)
    );
    assert_eq!(
        lsode2.linear_system_structure,
        Some(Lsode2LinearSystemStructure::Sparse)
    );
    assert_eq!(lsode2.stop_conditions.len(), 1);
    assert_eq!(lsode2.stop_conditions[0].variable, "y");
    assert_eq!(
        lsode2.stop_conditions[0].comparator,
        crate::numerical::LSODE2::Lsode2StopComparator::LessEqual
    );

    let config = build_lsode2_problem_config_from_spec(&spec)
        .expect("parsed LSODE2 stop condition should reach the native config");
    assert_eq!(config.stop_conditions.len(), 1);
    assert_eq!(config.stop_conditions[0].variable, "y");
    assert_eq!(config.stop_conditions[0].target, 0.5);
}

#[test]
fn ivp_task_parser_rejects_partial_lsode2_stop_condition() {
    let input = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
lsode2_stop_target: 0.1
"#;

    let error = parse_ivp_task_from_str(input)
        .expect_err("a stop target without its state variable must be rejected");
    assert!(matches!(error, IvpTaskError::MissingField { .. }));
    assert!(error.to_string().contains("lsode2_stop_variable"));
}

#[test]
fn ivp_task_parser_maps_lsode2_method_family_to_controller_config() {
    use crate::numerical::LSODE2::Lsode2ControllerMode;

    let cases = [
        ("LSODA", "auto", Lsode2ControllerMode::AutomaticAdamsBdf),
        ("LSODE", "adams", Lsode2ControllerMode::AdamsOnly),
        ("LSODE", "bdf", Lsode2ControllerMode::BdfOnly),
    ];

    for (method, family, expected_mode) in cases {
        let input = format!(
            r#"
task
solver: IVP
method: {method}

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_method_family: {family}
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: auto
"#
        );
        let spec =
            parse_ivp_task_from_str(&input).expect("LSODE2 controller document should parse");
        let config = build_lsode2_problem_config_from_spec(&spec)
            .expect("LSODE2 controller document should build problem config");
        assert_eq!(
            config.controller.mode, expected_mode,
            "{method}/{family} should map to the expected controller mode"
        );
    }
}

#[test]
fn ivp_task_parser_builds_lsode2_banded_auto_resolved_plan() {
    use crate::numerical::LSODE2::{
        Lsode2ControllerMode, Lsode2LinearSolverBackend, Lsode2LinearSolverChoice,
    };

    let input = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -2.0*y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_method_family: bdf
lsode2_symbolic_assembly: AtomView
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: banded
lsode2_linear_solver_policy: auto
lsode2_native_execution: faithful_bdf_solve
"#;

    let spec = parse_ivp_task_from_str(input).expect("LSODE2 banded document should parse");
    let config = build_lsode2_problem_config_from_spec(&spec)
        .expect("LSODE2 banded document should build problem config");
    let resolved = config.resolve_plan();

    assert_eq!(config.controller.mode, Lsode2ControllerMode::BdfOnly);
    assert_eq!(
        config.residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        }
    );
    assert_eq!(
        resolved.structure,
        Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 }
    );
    assert_eq!(
        resolved.linear_solver,
        Lsode2LinearSolverChoice::LapackFaithfulBandedLu
    );
    assert_eq!(
        resolved.linear_solver_reason,
        "auto_from_linear_structure_banded"
    );
    assert_eq!(
        config.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
    assert_eq!(
        config.backend.jacobian_backend,
        Lsode2JacobianBackend::SymbolicGenerated
    );
}

#[test]
fn ivp_task_parser_keeps_lsode2_forced_policy_visible_in_resolved_plan() {
    use crate::numerical::LSODE2::{Lsode2LinearSolverBackend, Lsode2LinearSolverChoice};

    let input = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -2.0*y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: lapack_faithful_banded_lu
"#;

    let spec = parse_ivp_task_from_str(input).expect("LSODE2 forced-policy document should parse");
    let config = build_lsode2_problem_config_from_spec(&spec)
        .expect("LSODE2 forced-policy document should build problem config");
    let resolved = config.resolve_plan();

    assert_eq!(resolved.structure, Lsode2LinearSystemStructure::Sparse);
    assert_eq!(
        resolved.linear_solver,
        Lsode2LinearSolverChoice::LapackFaithfulBandedLu
    );
    assert_eq!(
        resolved.linear_solver_reason,
        "forced_by_linear_solver_policy"
    );
    assert_eq!(
        config.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
}

#[test]
fn ivp_task_parser_builds_lsode2_aot_toolchain_backend_contract() {
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy;

    let output_dir = PathBuf::from("target/lsode2-task-parser-aot-contract");
    let input = format!(
        r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -2.0*y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_assembly: AtomView
lsode2_symbolic_execution: AOT
lsode2_aot_toolchain: zig
lsode2_aot_profile: debug
lsode2_aot_output_dir: {}
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: auto
"#,
        output_dir.display()
    );

    let spec = parse_ivp_task_from_str(&input).expect("LSODE2 AOT document should parse");
    let config = build_lsode2_problem_config_from_spec(&spec)
        .expect("LSODE2 AOT document should build problem config");
    let resolved = config.resolve_plan();

    assert_eq!(
        resolved.source,
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::Zig,
                profile: Lsode2AotProfile::Debug,
            },
        }
    );
    assert_eq!(
        resolved.linear_solver,
        Lsode2LinearSolverChoice::FaerSparseLu
    );
    assert_eq!(
        config.backend.generated_backend.aot_codegen_backend,
        AotCodegenBackend::Zig
    );
    assert_eq!(config.backend.generated_backend.aot_c_compiler, None);
    assert_eq!(
        config.backend.generated_backend.output_parent_dir,
        Some(output_dir)
    );
    assert_eq!(
        config.backend.generated_backend.build_policy,
        SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Debug,
        }
    );
}

#[test]
fn ivp_task_runner_supports_lsode2_lambdify_path() {
    let input = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: dense
lsode2_linear_solver_policy: auto
"#;

    let result = run_ivp_task_from_str(input).expect("LSODE2 task should solve");
    assert!(result.status.is_some());
    assert!(result.y_result.is_some());
}

#[test]
fn ivp_task_runner_executes_lsode2_modern_postprocessing_plan() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let csv_path = dir.path().join("lsode2_task_solution.csv");
    let report_path = dir.path().join("lsode2_task_report.md");
    let input = format!(
        r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: dense
lsode2_linear_solver_policy: auto

postprocessing
save_csv: true
csv_path: {}
write_report: true
report_path: {}
"#,
        csv_path.display(),
        report_path.display()
    );

    let result = run_ivp_task_from_str(&input).expect("LSODE2 task should solve");
    let t = result
        .t_result
        .as_ref()
        .expect("LSODE2 task should return a time mesh");
    let y = result
        .y_result
        .as_ref()
        .expect("LSODE2 task should return a solution matrix");
    assert_eq!(t.len(), y.nrows());
    assert_eq!(y.ncols(), 1);
    assert!(csv_path.exists());
    assert!(report_path.exists());

    let csv = std::fs::read_to_string(&csv_path).expect("CSV should be readable");
    assert!(csv.contains("t,y"));
    let report = std::fs::read_to_string(&report_path).expect("report should be readable");
    assert!(report.contains("Solver Result Report"));
    assert!(report.contains("axis: t"));
}

#[test]
fn ivp_task_parser_can_split_problem_and_solver_settings() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
parameters: a
parameter_values: 2.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
step_size: 1e-3
parallel: false
"#;

    let document = parse_document_for_ivp(input);
    let problem = parse_ivp_problem_from_document(&document)
        .expect("problem subset should parse independently");
    let settings = parse_ivp_solver_settings_from_document(&document)
        .expect("solver settings subset should parse independently");

    assert_eq!(problem.equations.unknowns, vec!["y".to_string()]);
    assert_eq!(problem.equations.parameter_values["a"], 2.0);
    assert_eq!(
        settings.solver.method,
        IvpMethodSpec::NonStiff("RK45".to_string())
    );

    let solver = build_ivp_solver_from_problem_and_settings(&problem, &settings)
        .expect("split IVP mode should build a solver");
    let _ = solver;
}

#[test]
fn ivp_task_parser_reads_solver_settings_without_problem_sections() {
    let input = r#"
task
solver: IVP
method: LSODE2

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: dense
lsode2_linear_solver_policy: auto
"#;

    let document = parse_document_for_ivp(input);
    let settings = parse_ivp_solver_settings_from_document(&document)
        .expect("settings-only document should parse");

    assert_eq!(settings.solver.method, IvpMethodSpec::Lsode2);
    assert!(settings.solver_options.lsode2.is_some());
}

#[test]
fn ivp_task_parser_supports_full_end_to_end_document_with_lsode2_settings() {
    let input = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_method_family: bdf
lsode2_symbolic_assembly: AtomView
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: banded
lsode2_linear_solver_policy: auto
lsode2_native_execution: faithful_bdf_solve
"#;

    let spec = parse_ivp_task_from_str(input).expect("full IVP task doc should parse");
    assert_eq!(spec.solver.method, IvpMethodSpec::Lsode2);
    assert_eq!(
        spec.solver_options
            .lsode2
            .as_ref()
            .and_then(|opts| opts.symbolic_assembly.clone()),
        Some(Lsode2SymbolicAssemblyBackend::AtomView)
    );
    assert_eq!(spec.problem_spec().initial_conditions.t0, 0.0);
}

#[test]
fn ivp_task_parser_can_build_solver_from_rust_problem_and_task_doc_settings() {
    let problem = IvpProblemSpec {
        equations: EquationSpec {
            arg: "t".to_string(),
            unknowns: vec!["y".to_string()],
            rhs: vec![Expr::parse_expression("-y")],
            parameter_names: vec![],
            parameter_values: HashMap::new(),
        },
        initial_conditions: InitialConditionSpec {
            t0: 0.0,
            t_end: 1.0,
            y0: vec![1.0],
        },
    };

    let settings_doc = r#"
task
solver: IVP
method: RK45

solver_options
step_size: 1e-3
parallel: false
"#;
    let document = parse_document_for_ivp(settings_doc);
    let settings =
        parse_ivp_solver_settings_from_document(&document).expect("solver settings should parse");
    let solver = build_ivp_solver_from_problem_and_settings(&problem, &settings)
        .expect("Rust problem + DSL settings should build");
    let _ = solver;
}

#[test]
fn ivp_task_parser_reports_missing_parameter_values_during_lsode2_build() {
    let problem = IvpProblemSpec {
        equations: EquationSpec {
            arg: "t".to_string(),
            unknowns: vec!["y".to_string()],
            rhs: vec![Expr::parse_expression("-a*y")],
            parameter_names: vec!["a".to_string()],
            parameter_values: HashMap::new(),
        },
        initial_conditions: InitialConditionSpec {
            t0: 0.0,
            t_end: 1.0,
            y0: vec![1.0],
        },
    };

    let settings_doc = r#"
task
solver: IVP
method: LSODE2

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.05
lsode2_symbolic_execution: LambdifyExpr
lsode2_linear_structure: sparse
lsode2_linear_solver_policy: auto
"#;
    let document = parse_document_for_ivp(settings_doc);
    let settings =
        parse_ivp_solver_settings_from_document(&document).expect("solver settings should parse");

    match build_ivp_solver_from_problem_and_settings(&problem, &settings) {
        Ok(_) => panic!("missing parameter values should be reported"),
        Err(err) => {
            let message = err.to_string();
            assert!(message.contains("parameter_values[a]"));
        }
    }
}
