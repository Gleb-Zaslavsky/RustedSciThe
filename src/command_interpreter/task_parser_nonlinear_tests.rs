use super::*;
use std::collections::HashMap;

fn section(fields: &[(&str, Vec<Value>)]) -> HashMap<String, Option<Vec<Value>>> {
    fields
        .iter()
        .map(|(name, values)| ((*name).to_string(), Some(values.clone())))
        .collect()
}

fn root_task() -> DocumentMap {
    HashMap::from([
        (
            "problem".into(),
            section(&[
                ("formulation", vec![Value::String("root".into())]),
                (
                    "variables",
                    vec![Value::String("x".into()), Value::String("y".into())],
                ),
                (
                    "equations",
                    vec![
                        Value::String("x + y - 3".into()),
                        Value::String("x - y - 1".into()),
                    ],
                ),
                ("initial_guess", vec![Value::Vector(vec![0.0, 0.0])]),
            ]),
        ),
        (
            "solver".into(),
            section(&[
                ("method", vec![Value::String("newton".into())]),
                ("max_iterations", vec![Value::Usize(80)]),
            ]),
        ),
    ])
}

#[test]
fn parses_square_root_task_and_typed_solver_options() {
    let parsed = parse_nonlinear_task_from_document(&root_task()).expect("valid root task");
    assert_eq!(parsed.problem.formulation, NonlinearFormulation::Root);
    assert_eq!(parsed.problem.variables, ["x", "y"]);
    assert_eq!(parsed.problem.equations.len(), 2);
    assert_eq!(parsed.solver_options.max_iterations, Some(80));
}

#[test]
fn nonlinear_task_telemetry_is_explicit_and_reaches_native_results() {
    match run_nonlinear_task(
        parse_nonlinear_task_from_document(&root_task()).expect("default telemetry parses"),
    )
    .expect("default root solve")
    {
        NonlinearTaskRunResult::Root(result) => {
            assert!(!result.statistics.availability.is_collected());
            assert!(!result.statistics.timings_collected);
        }
        _ => panic!("expected root result"),
    }

    let mut task = root_task();
    task.get_mut("solver").unwrap().insert(
        "telemetry".into(),
        Some(vec![Value::String("detailed".into())]),
    );
    let spec = parse_nonlinear_task_from_document(&task).expect("detailed telemetry parses");
    assert_eq!(
        spec.solver_options.telemetry,
        Some(TaskTelemetryMode::Detailed)
    );
    match run_nonlinear_task(spec).expect("root solve") {
        NonlinearTaskRunResult::Root(result) => {
            assert!(result.statistics.availability.is_collected());
            assert!(result.statistics.timings_collected);
        }
        _ => panic!("expected root result"),
    }

    let mut task = root_task();
    task.get_mut("solver").unwrap().insert(
        "telemetry".into(),
        Some(vec![Value::String("counters".into())]),
    );
    let spec = parse_nonlinear_task_from_document(&task).expect("counter telemetry parses");
    assert_eq!(
        spec.solver_options.telemetry,
        Some(TaskTelemetryMode::Counters)
    );
    match run_nonlinear_task(spec).expect("counter-enabled root solve") {
        NonlinearTaskRunResult::Root(result) => {
            assert!(result.statistics.availability.is_collected());
            assert!(!result.statistics.timings_collected);
        }
        _ => panic!("expected root result"),
    }
}

#[test]
fn nonlinear_task_rejects_unknown_telemetry_mode() {
    let mut task = root_task();
    task.get_mut("solver").unwrap().insert(
        "telemetry".into(),
        Some(vec![Value::String("verbose".into())]),
    );
    assert!(matches!(
        parse_nonlinear_task_from_document(&task),
        Err(NonlinearTaskError::InvalidField {
            section: "solver",
            field: "telemetry",
            ..
        })
    ));
}

#[test]
fn nonlinear_root_task_forwards_pre_cancelled_token_to_native_snapshot() {
    let spec = parse_nonlinear_task_from_document(&root_task()).expect("valid root task");
    let token = crate::numerical::CancellationToken::new();
    token.cancel();
    let error = run_nonlinear_task_with_cancellation_token(spec, &token)
        .expect_err("pre-cancelled root task must not publish a result");
    let NonlinearTaskRunError::Root(
        crate::numerical::Nonlinear_systems::error::SolveError::Cancelled { snapshot },
    ) = error
    else {
        panic!("expected typed root cancellation snapshot");
    };
    assert_eq!(snapshot.completed_iterations, 0);
    assert_eq!(snapshot.current_state, [0.0, 0.0]);
    assert!(snapshot.current_residual.is_none());
}

#[test]
fn nonlinear_least_squares_task_forwards_pre_cancelled_token_to_native_snapshot() {
    let mut task = root_task();
    task.get_mut("problem").unwrap().insert(
        "formulation".into(),
        Some(vec![Value::String("least_squares".into())]),
    );
    task.get_mut("solver").unwrap().insert(
        "method".into(),
        Some(vec![Value::String("levenberg_marquardt".into())]),
    );
    let spec = parse_nonlinear_task_from_document(&task).expect("valid least-squares task");
    let token = crate::numerical::CancellationToken::new();
    token.cancel();
    let error = run_nonlinear_task_with_cancellation_token(spec, &token)
        .expect_err("pre-cancelled least-squares task must not publish a result");
    let NonlinearTaskRunError::LeastSquares(
        crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::Cancelled {
            snapshot,
        },
    ) = error
    else {
        panic!("expected typed least-squares cancellation snapshot");
    };
    assert_eq!(snapshot.completed_iterations, 0);
    assert_eq!(snapshot.parameters, [0.0, 0.0]);
    assert!(snapshot.residual_norm.is_none());
}

#[test]
fn native_runner_solves_root_and_rectangular_least_squares_tasks() {
    let task = root_task();
    let result = run_nonlinear_task(parse_nonlinear_task_from_document(&task).expect("parse root"))
        .expect("root solve");
    match result {
        NonlinearTaskRunResult::Root(result) => {
            assert!((result.x[0] - 2.0).abs() < 1e-8);
            assert!((result.x[1] - 1.0).abs() < 1e-8);
        }
        _ => panic!("expected root result"),
    }

    let mut task = root_task();
    task.get_mut("problem").unwrap().insert(
        "formulation".into(),
        Some(vec![Value::String("least_squares".into())]),
    );
    task.get_mut("problem")
        .unwrap()
        .insert("variables".into(), Some(vec![Value::String("x".into())]));
    task.get_mut("problem").unwrap().insert(
        "equations".into(),
        Some(vec![
            Value::String("x - 1".into()),
            Value::String("2*x - 2".into()),
            Value::String("3*x - 3".into()),
        ]),
    );
    task.get_mut("problem")
        .unwrap()
        .insert("initial_guess".into(), Some(vec![Value::Vector(vec![0.0])]));
    task.get_mut("solver")
        .unwrap()
        .insert("method".into(), Some(vec![Value::String("lm".into())]));
    task.get_mut("solver").unwrap().insert(
        "telemetry".into(),
        Some(vec![Value::String("detailed".into())]),
    );
    let result = run_nonlinear_task(parse_nonlinear_task_from_document(&task).expect("parse LS"))
        .expect("least-squares solve");
    match result {
        NonlinearTaskRunResult::LeastSquares {
            report,
            solution: Some(solution),
        } => {
            assert!(report.termination.was_successful());
            assert!(report.statistics.availability.is_collected());
            assert!(report.statistics.timings_collected);
            assert!((solution[0] - 1.0).abs() < 1e-6);
        }
        _ => panic!("expected successful least-squares result"),
    }
}

#[test]
fn text_parser_normalizes_section_and_field_names() {
    let text = "problem\nformulation: root\nvariables: x\nequations: x - 2\ninitial_guess: 0.0\nsolver\nmethod: newton\n";
    let parsed = parse_nonlinear_task_from_text(text).expect("text task should parse");
    assert_eq!(parsed.problem.variables, ["x"]);
}

#[test]
fn rectangular_least_squares_is_accepted_but_rectangular_root_is_rejected() {
    let mut task = root_task();
    task.get_mut("problem").unwrap().insert(
        "formulation".into(),
        Some(vec![Value::String("least_squares".into())]),
    );
    task.get_mut("problem").unwrap().insert(
        "equations".into(),
        Some(vec![
            Value::String("x - 1".into()),
            Value::String("2*x - 2".into()),
            Value::String("3*x - 3".into()),
        ]),
    );
    task.get_mut("problem")
        .unwrap()
        .insert("variables".into(), Some(vec![Value::String("x".into())]));
    task.get_mut("problem")
        .unwrap()
        .insert("initial_guess".into(), Some(vec![Value::Vector(vec![0.0])]));
    task.get_mut("solver")
        .unwrap()
        .insert("method".into(), Some(vec![Value::String("lm".into())]));
    assert!(parse_nonlinear_task_from_document(&task).is_ok());

    task.get_mut("problem").unwrap().insert(
        "formulation".into(),
        Some(vec![Value::String("root".into())]),
    );
    assert!(matches!(
        parse_nonlinear_task_from_document(&task),
        Err(NonlinearTaskError::NonSquareRoot { .. })
    ));
}

#[test]
fn rejects_unknown_symbols_bad_guess_and_nonpositive_iteration_limit() {
    let mut task = root_task();
    task.get_mut("problem").unwrap().insert(
        "equations".into(),
        Some(vec![
            Value::String("x + undeclared".into()),
            Value::String("y".into()),
        ]),
    );
    assert!(matches!(
        parse_nonlinear_task_from_document(&task),
        Err(NonlinearTaskError::InvalidField {
            field: "equations",
            ..
        })
    ));

    let mut task = root_task();
    task.get_mut("problem")
        .unwrap()
        .insert("initial_guess".into(), Some(vec![Value::Vector(vec![0.0])]));
    assert!(matches!(
        parse_nonlinear_task_from_document(&task),
        Err(NonlinearTaskError::InvalidField {
            field: "initial_guess",
            ..
        })
    ));

    let mut task = root_task();
    task.get_mut("solver")
        .unwrap()
        .insert("max_iterations".into(), Some(vec![Value::Integer(0)]));
    assert!(matches!(
        parse_nonlinear_task_from_document(&task),
        Err(NonlinearTaskError::InvalidField {
            field: "max_iterations",
            ..
        })
    ));
}
