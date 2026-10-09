use super::*;
use crate::command_interpreter::task_parser::ParseErrorKind;
use crate::command_interpreter::task_runner::{
    parse_task_spec_from_file, ParsedTaskSpec, TaskRunnerError,
};
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
fn ivp_task_parser_preserves_symbolic_model_for_continuation() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
unknowns: y
rhs: -rate*y
parameters: rate
parameter_values: 1.0

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

continuation
parameter: rate
values: 0.5, 1.0, 2.0
mode: prepared
restart_each: false
"#;

    let spec = parse_ivp_task_from_str(input).expect("continuation task should parse");
    let continuation = spec
        .continuation
        .expect("continuation plan should be present");
    assert_eq!(spec.schema_version, 1);
    assert_eq!(continuation.parameter, "rate");
    assert_eq!(continuation.values, vec![0.5, 1.0, 2.0]);
    assert_eq!(continuation.mode, ContinuationMode::Prepared);
    assert!(!continuation.restart_each);
    assert!(continuation.symbolic_rhs[0].to_string().contains("rate"));
    assert!(!spec.equations.rhs[0].to_string().contains("rate"));
}

#[test]
fn ivp_continuation_supports_parameter_grids_and_segment_overrides() {
    let input = r#"
task
schema_version: 1
solver: IVP
method: BDF

equations
arg: t
unknowns: y
rhs: -(rate + source)*y
parameters: rate, source
parameter_values: 1.0, 0.0

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

continuation
parameters: rate, source
rate_values: 1.0, 2.0
source_values: 0.0, 3.0
mode: fresh
restart_policy: restart_with_state
monotonic: allow
y0_values: [1.0], [2.0], [3.0], [4.0]
t0_values: 0.0, 0.1, 0.2, 0.3
t_end_values: 1.0, 1.1, 1.2, 1.3
"#;

    let spec = parse_ivp_task_from_str(input).expect("parameter grid should parse");
    let continuation = spec
        .continuation
        .clone()
        .expect("continuation should be present");
    assert_eq!(continuation.parameters, vec!["rate", "source"]);
    assert_eq!(continuation.value_grid.len(), 4);
    assert_eq!(continuation.value_grid[0], vec![1.0, 0.0]);
    assert_eq!(continuation.value_grid[3], vec![2.0, 3.0]);
    assert_eq!(continuation.y0_values.as_ref().unwrap().len(), 4);
    assert_eq!(continuation.t_end_values.as_ref().unwrap()[2], 1.2);
    assert_eq!(
        continuation.restart_policy,
        ContinuationRestartPolicy::RestartWithState
    );
    let run = run_ivp_continuation(spec).expect("fresh grid continuation should execute");
    assert_eq!(run.segments.len(), 4);
    assert!(run
        .segments
        .iter()
        .all(|segment| segment.status_code.is_some()));
}

#[test]
fn ivp_task_schema_rejects_unknown_version() {
    let input = r#"
task
schema_version: 2
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
"#;
    let error = parse_ivp_task_from_str(input).expect_err("unknown schema must be rejected");
    assert!(matches!(error, IvpTaskError::InvalidConfiguration { .. }));
    assert!(error.to_string().contains("schema version"));
}

#[test]
fn ivp_continuation_rejects_non_monotone_values_when_requested() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
unknowns: y
rhs: -rate*y
parameters: rate
parameter_values: 1.0

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

continuation
parameter: rate
values: 1.0, 0.5, 2.0
monotonic: increasing
"#;
    let error = parse_ivp_task_from_str(input).expect_err("non-monotone values must be rejected");
    assert!(error.to_string().contains("not monotonic"));
}

#[test]
fn ivp_symbol_diagnostics_retain_source_position() {
    let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
unknowns: y
rhs: -y + misspelled_gain

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0
"#;
    let error = parse_ivp_task_from_str(input).expect_err("undeclared symbol must fail");
    match error {
        IvpTaskError::SymbolDiagnostic {
            token,
            line: Some(line),
            column: Some(column),
            ..
        } => {
            assert_eq!(token, "misspelled_gain");
            assert!(line > 1);
            assert!(column > 1);
        }
        other => panic!("expected positioned symbol diagnostic, got {other:?}"),
    }
}

#[test]
fn document_parser_rejects_alias_collisions() {
    let mut parser = DocumentParser::new("task\nsolver: IVP".to_string());
    let headers = HashMap::from([
        ("task".to_string(), vec!["task".to_string()]),
        ("settings".to_string(), vec!["task".to_string()]),
    ]);
    let error = parser
        .try_with_pseudonims(Some(headers), None)
        .expect_err("one alias cannot denote two sections");
    assert!(error.contains("pseudonym collision"));
}

#[test]
fn typed_parser_errors_preserve_collision_categories() {
    let mut parser = DocumentParser::new("task\nsolver: IVP".to_string());
    let headers = HashMap::from([
        ("task".to_string(), vec!["task".to_string()]),
        ("settings".to_string(), vec!["task".to_string()]),
    ]);
    let error = parser
        .try_with_pseudonims_typed(Some(headers), None)
        .expect_err("alias collision must be typed");
    assert_eq!(error.kind, ParseErrorKind::AliasCollision);
    assert!(error.to_string().contains("pseudonym collision"));

    let mut parser = DocumentParser::new("Task\nsolver: IVP\ntask\nmethod: BDF".to_string());
    parser
        .parse_document_typed()
        .expect("mixed-case sections parse first");
    let error = parser
        .try_keys_to_lower_case_typed(None)
        .expect_err("normalization collision must be typed");
    assert_eq!(error.kind, ParseErrorKind::CaseNormalizationCollision);
    assert!(error.line.is_some());
    assert!(error.column.is_some());
}

#[test]
fn ivp_adapter_keeps_syntax_error_category() {
    let error = parse_ivp_task_from_str("task\nsolver IVP")
        .expect_err("missing key/value separator must fail");
    match error {
        IvpTaskError::Document(error) => assert_eq!(error.kind, ParseErrorKind::InvalidSection),
        other => panic!("expected typed document error, got {other:?}"),
    }
}

#[test]
fn ivp_adapter_preserves_native_error_category_and_source() {
    let error = IvpTaskError::Native(IvpNativeSolverError::Universal(
        UniversalOdeError::UnsupportedGeneratedBackendForMethod {
            method: "BDF".to_string(),
        },
    ));

    assert!(error
        .to_string()
        .contains("generated/AOT backend selection"));
    assert!(std::error::Error::source(&error).is_some());
    assert!(matches!(
        error,
        IvpTaskError::Native(IvpNativeSolverError::Universal(
            UniversalOdeError::UnsupportedGeneratedBackendForMethod { .. }
        ))
    ));
}

#[test]
fn document_parser_rejects_field_alias_collisions() {
    let mut parser = DocumentParser::new("task\nsolver: IVP".to_string());
    let fields = HashMap::from([(
        "method".to_string(),
        vec!["solver".to_string(), "method".to_string()],
    )]);
    parser
        .try_with_pseudonims(None, Some(fields))
        .expect("aliases themselves are valid");
    let document = HashMap::from([(
        "task".to_string(),
        HashMap::from([
            (
                "solver".to_string(),
                Some(vec![Value::String("IVP".into())]),
            ),
            (
                "method".to_string(),
                Some(vec![Value::String("BDF".into())]),
            ),
        ]),
    )]);
    let error = parser
        .try_to_real_names(Some(document))
        .expect_err("two fields must not collapse into one");
    assert!(error.contains("field alias collision"));
}

#[test]
fn document_parser_rejects_case_normalization_collisions() {
    let mut parser = DocumentParser::new("Task\nsolver: IVP\ntask\nmethod: BDF".to_string());
    parser
        .parse_document()
        .expect("mixed-case sections parse first");
    let error = parser
        .try_keys_to_lower_case(None)
        .expect_err("normalization must not discard a section");
    assert!(error.contains("case-normalization collision"));
}

#[test]
fn ivp_duplicate_semantic_declarations_are_rejected_with_source_location() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
parameters: rate, rate
parameter_values: 1.0, 2.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"#;

    let error = parse_ivp_task_from_str(input)
        .expect_err("duplicate parameter declarations must be rejected");
    match error {
        IvpTaskError::SymbolDiagnostic {
            message,
            token,
            line,
            column,
        } => {
            assert!(message.contains("duplicate parameter name"));
            assert_eq!(token, "rate");
            assert!(line.is_some());
            assert!(column.is_some());
        }
        other => panic!("expected a located semantic diagnostic, got {other:?}"),
    }
}

#[test]
fn ivp_control_keys_are_case_insensitive_but_symbol_names_keep_case() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: T
parameters: K
parameter_values: 2.0
Y: -K*Y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
RTOL: 1e-5
ATOL: 1e-7
MAX_STEP: 0.01
"#;
    let spec = parse_ivp_task_from_str(input).expect("mixed control and symbol case is valid");
    assert_eq!(spec.equations.arg, "T");
    assert_eq!(spec.equations.unknowns, vec!["Y"]);
    assert_eq!(spec.equations.parameter_names, vec!["K"]);
    assert_eq!(spec.solver_options.rtol, Some(1e-5));
    assert_eq!(spec.solver_options.atol, Some(1e-7));
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
fn ivp_task_parser_rejects_unsupported_radau3_route() {
    let input = r#"
task
solver: IVP
method: Radau3

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0
"#;

    match parse_ivp_task_from_str(input) {
        Err(IvpTaskError::UnknownMethod(method)) => assert_eq!(method, "radau3"),
        other => panic!("Radau3 must be rejected as an unknown method: {other:?}"),
    }
}

#[test]
fn ivp_task_parser_keeps_lsode_and_lsoda_aliases_distinct() {
    for (method, expected) in [
        ("LSODE", IvpMethodSpec::Lsode),
        ("LSODA", IvpMethodSpec::Lsoda),
        ("LSODE2", IvpMethodSpec::Lsode2),
    ] {
        let input = format!("task\nsolver: IVP\nmethod: {method}\n");
        let document = parse_document_for_ivp(&input);
        let settings = parse_ivp_solver_settings_from_document(&document)
            .expect("LSODE family method should parse");
        assert_eq!(settings.solver.method, expected);
    }
}

#[test]
fn ivp_task_parser_rejects_an_option_that_native_bdf_cannot_honor() {
    let input = "task\nsolver: IVP\nmethod: BDF\n\nsolver_options\nstep_size: 1e-3\n";
    let document = parse_document_for_ivp(input);
    let error = parse_ivp_solver_settings_from_document(&document)
        .expect_err("BDF must not silently ignore step_size");
    assert!(matches!(
        error,
        IvpTaskError::UnsupportedOption { method, option }
            if method == "BDF" && option == "step_size"
    ));
}

#[test]
fn ivp_task_runner_executes_prepared_bdf_continuation() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
parameters: rate
parameter_values: 1.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
rtol: 1e-5
atol: 1e-7
max_step: 0.02

continuation
parameter: rate
values: 0.5, 1.0, 2.0
mode: prepared
restart_each: false
"#;

    let result = run_ivp_task_from_str(input).expect("prepared continuation should solve");
    let continuation = result
        .continuation
        .expect("run result should retain continuation segments");
    assert_eq!(continuation.segments.len(), 3);
    assert_eq!(continuation.fresh_preparations, 1);
    assert_eq!(continuation.prepared_reuses, 2);
    assert_eq!(continuation.parameter, "rate");
    assert!(continuation
        .segments
        .iter()
        .all(|segment| segment.status.as_deref() == Some("finished")));
    assert!(continuation
        .segments
        .iter()
        .all(|segment| segment.status_code == Some(IvpRunStatus::Finished)));
    assert!(continuation
        .segments
        .iter()
        .all(|segment| matches!(segment.trajectory, Some(IvpTrajectory::Grid { .. }))));
}

#[test]
fn ivp_prepared_continuation_long_series_has_explicit_bounded_result_storage() {
    let values = (0..64)
        .map(|index| format!("{:.3}", 0.5 + index as f64 * 0.025))
        .collect::<Vec<_>>()
        .join(", ");
    let input = format!(
        r#"
task
solver: IVP
method: BDF

equations
arg: t
parameters: rate
parameter_values: 1.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.02
y0: 1.0

solver_options
rtol: 1e-5
atol: 1e-7
max_step: 0.01

continuation
parameter: rate
values: {values}
mode: prepared
restart_each: false
"#
    );

    let first = run_ivp_task_from_str(&input).expect("long prepared series should solve");
    let first_series = first
        .continuation
        .expect("continuation result should be present");
    assert_eq!(first_series.segments.len(), 64);
    assert_eq!(first_series.segments.capacity(), 64);
    assert!(first_series
        .segments
        .iter()
        .all(|segment| segment.status_code == Some(IvpRunStatus::Finished)));

    let second = run_ivp_task_from_str(&input).expect("repeat long series should solve");
    let second_series = second
        .continuation
        .expect("repeat result should be present");
    assert_eq!(second_series.segments.len(), first_series.segments.len());
    assert_eq!(
        second_series.segments.capacity(),
        first_series.segments.capacity()
    );
}

#[test]
fn ivp_prepared_continuation_reports_typed_mid_series_failure() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
parameters: dummy
parameter_values: 1.0
y: -y

initial_conditions
t0: 0.0
t_end: 0.01
y0: 1.0

solver_options
max_step: 0.005

continuation
parameter: dummy
values: 1.0, 2.0, 3.0
mode: prepared
restart_policy: restart_with_state
y0_values: [1.0], [1.0, 2.0], [1.0]
t0_values: 0.0, 0.0, 0.0
t_end_values: 0.01, 0.01, 0.01
"#;

    let error = run_ivp_task_from_str(input)
        .expect_err("a malformed middle segment must stop with a typed error");
    assert!(matches!(error.category(), "configuration" | "solver"));
    assert!(!error.to_string().is_empty());
}

#[test]
fn ivp_be_and_lsode_restart_contract_is_explicit_for_new_state() {
    let input = r#"
task
solver: IVP
method: BackwardEuler

equations
arg: t
parameters: rate
parameter_values: 1.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.02
y0: 1.0

solver_options
step_size: 0.01
tolerance: 1e-6
max_iterations: 20

continuation
parameter: rate
values: 1.0, 2.0
mode: prepared
restart_policy: restart_with_state
y0_values: [1.0], [2.0]
t0_values: 0.0, 0.0
t_end_values: 0.02, 0.02
"#;
    let result = run_ivp_task_from_str(input).expect("BE prepared restart should execute");
    let continuation = result
        .continuation
        .expect("BE restart should return continuation segments");
    assert_eq!(continuation.segments.len(), 2);
    assert!(continuation
        .segments
        .iter()
        .all(|segment| segment.status_code == Some(IvpRunStatus::Finished)));

    let lsode = input
        .replace("BackwardEuler", "LSODE2")
        .replace("step_size: 0.01\n", "max_step: 0.01\n")
        .replace("tolerance: 1e-6\n", "rtol: 1e-6\n")
        .replace("max_iterations: 20\n", "");
    let result = run_ivp_task_from_str(&lsode).expect("LSODE2 prepared restart should execute");
    let continuation = result
        .continuation
        .expect("LSODE2 restart should return continuation segments");
    assert_eq!(continuation.segments.len(), 2);
    assert!(continuation
        .segments
        .iter()
        .all(|segment| segment.status_code == Some(IvpRunStatus::Finished)));
}

#[test]
fn ivp_task_runner_executes_backward_euler_continuation() {
    let input = r#"
task
solver: IVP
method: BackwardEuler

equations
arg: t
parameters: rate
parameter_values: 2.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.05
y0: 1.0

solver_options
step_size: 1e-3
tolerance: 1e-8
max_iterations: 100

continuation
parameter: rate
values: 1.0, 2.0, 3.0
mode: warm
restart_each: false
"#;

    let result = run_ivp_task_from_str(input).expect("BE continuation should solve");
    let continuation = result
        .continuation
        .expect("continuation result should be typed");
    assert_eq!(continuation.segments.len(), 3);
    assert!(continuation
        .segments
        .iter()
        .all(|segment| segment.status_code == Some(IvpRunStatus::Finished)));
}

#[test]
fn ivp_task_runner_executes_radau5_continuation() {
    let input = r#"
task
solver: IVP
method: Radau5

equations
arg: t
parameters: rate
parameter_values: 2.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.05
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.02

continuation
parameter: rate
values: 1.0, 2.0, 3.0
mode: prepared
restart_each: false
"#;

    let result = run_ivp_task_from_str(input).expect("Radau continuation should solve");
    let continuation = result
        .continuation
        .expect("continuation result should be typed");
    assert_eq!(continuation.segments.len(), 3);
    assert!(continuation
        .segments
        .iter()
        .all(|segment| segment.status_code == Some(IvpRunStatus::Finished)));
    assert!(continuation
        .segments
        .iter()
        .all(|segment| matches!(segment.trajectory, Some(IvpTrajectory::Radau(_)))));
}

#[test]
fn ivp_task_runner_executes_lsode2_restart_continuation() {
    let input = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
parameters: rate
parameter_values: 2.0
y: -rate*y

initial_conditions
t0: 0.0
t_end: 0.05
y0: 1.0

solver_options
max_step: 0.02

continuation
parameter: rate
values: 1.0, 2.0, 3.0
mode: prepared
restart_each: true
"#;

    let result = run_ivp_task_from_str(input).expect("LSODE2 continuation should solve");
    let continuation = result
        .continuation
        .expect("continuation result should be typed");
    assert_eq!(continuation.segments.len(), 3);
    assert!(continuation
        .segments
        .iter()
        .all(|segment| segment.status_code == Some(IvpRunStatus::Finished)));
}

#[test]
fn ivp_task_parser_rejects_non_finite_configuration_before_build() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
y: -y

initial_conditions
t0: NaN
t_end: 1.0
y0: 1.0
"#;

    let error = parse_ivp_task_from_str(input).expect_err("NaN t0 must be rejected early");
    assert!(matches!(
        error,
        IvpTaskError::InvalidConfiguration { field, .. }
            if field == "initial_conditions.t0"
    ));
}

#[test]
fn ivp_task_parser_rejects_zero_iteration_limit_before_build() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
max_iterations: 0
"#;

    let error = parse_ivp_task_from_str(input).expect_err("zero max_iterations must fail");
    assert!(matches!(
        error,
        IvpTaskError::InvalidConfiguration { field, .. }
            if field == "solver_options.max_iterations"
    ));
}

#[test]
fn ivp_task_runner_preserves_bdf_exhaustion_error() {
    let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
max_iterations: 1
max_step: 1e-3
"#;

    let error = run_ivp_task_from_str(input).expect_err("one BDF step budget must be exhausted");
    assert!(matches!(
        error,
        IvpTaskError::Native(IvpNativeSolverError::Bdf(
            BdfSolveError::MaxStepsExceeded { .. }
        ))
    ));
}

#[test]
fn ivp_task_file_errors_preserve_io_source() {
    let path = tempdir()
        .expect("tempdir should be created")
        .path()
        .join("missing_ivp_task.txt");
    let error = parse_ivp_task_from_file(Some(path)).expect_err("missing task must fail");
    assert!(matches!(error, IvpTaskError::Io { .. }));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn reference_task_documents_parse_as_a_complete_matrix() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("task_docs")
        .join("reference");
    fn collect_reference_documents(
        directory: &std::path::Path,
        paths: &mut Vec<std::path::PathBuf>,
    ) {
        for entry in fs::read_dir(directory).expect("reference task directory should exist") {
            let path = entry
                .expect("reference directory entry should be readable")
                .path();
            if path.is_dir() {
                collect_reference_documents(&path, paths);
            } else if path.extension().is_some_and(|extension| extension == "txt") {
                paths.push(path);
            }
        }
    }

    let mut paths = Vec::new();
    collect_reference_documents(&root, &mut paths);
    paths.sort();
    assert!(!paths.is_empty(), "reference task matrix must not be empty");
    for path in paths {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("reference document {} unreadable: {error}", path.display())
        });
        let declared_solver = source
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(key, _)| key.trim().eq_ignore_ascii_case("solver"))
            .map(|(_, value)| value.trim().to_ascii_lowercase())
            .unwrap_or_default();
        match parse_task_spec_from_file(path.clone()) {
            Ok(spec) => assert!(matches!(
                spec,
                ParsedTaskSpec::Ivp(_) | ParsedTaskSpec::Bvp(_)
            )),
            Err(TaskRunnerError::UnsupportedSolver { value })
                if declared_solver == "bvp_sci" && value == declared_solver => {}
            Err(error) => panic!("reference document {} failed: {error}", path.display()),
        }
    }
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

    let invalid_assembly = input.replace(
        "lsode2_symbolic_assembly: AtomView",
        "lsode2_symbolic_assembly: Lambdify",
    );
    let error = parse_ivp_task_from_str(&invalid_assembly)
        .expect_err("execution route must not be accepted as symbolic assembly");
    assert!(matches!(
        error,
        IvpTaskError::InvalidField { .. } | IvpTaskError::SymbolDiagnostic { .. }
    ));
    assert!(error.to_string().contains("symbolic assembly"));

    let invalid_execution = input.replace(
        "lsode2_symbolic_execution: LambdifyExpr",
        "lsode2_symbolic_execution: ExprLegacy",
    );
    let error = parse_ivp_task_from_str(&invalid_execution)
        .expect_err("symbolic assembly must not be accepted as execution route");
    assert!(matches!(
        error,
        IvpTaskError::InvalidField { .. } | IvpTaskError::SymbolDiagnostic { .. }
    ));
    assert!(error.to_string().contains("symbolic execution"));
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
            symbolic_rhs: vec![Expr::parse_expression("-y")],
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
            symbolic_rhs: vec![Expr::parse_expression("-a*y")],
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
