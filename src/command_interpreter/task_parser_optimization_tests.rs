use super::*;
use std::collections::HashMap;

fn section(fields: &[(&str, Vec<Value>)]) -> HashMap<String, Option<Vec<Value>>> {
    fields
        .iter()
        .map(|(name, values)| ((*name).to_string(), Some(values.clone())))
        .collect()
}

fn lm_task() -> DocumentMap {
    HashMap::from([
        (
            "data".into(),
            section(&[
                ("arg", vec![Value::String("x".into())]),
                ("x", vec![Value::Vector(vec![0.0, 1.0, 2.0])]),
                ("y", vec![Value::Vector(vec![1.0, 3.0, 5.0])]),
            ]),
        ),
        (
            "model".into(),
            section(&[
                ("method", vec![Value::String("lm".into())]),
                (
                    "parameters",
                    vec![Value::String("a".into()), Value::String("b".into())],
                ),
                ("initial_guess", vec![Value::Vector(vec![0.0, 0.0])]),
                ("equations", vec![Value::String("a*x + b".into())]),
            ]),
        ),
    ])
}

#[test]
fn parses_equation_fitting_task_and_checks_data_shape() {
    let parsed = parse_optimization_task_from_document(&lm_task()).expect("valid LM task");
    assert_eq!(parsed.method, OptimizationMethodSpec::LevenbergMarquardt);
    assert_eq!(parsed.data.x.len(), parsed.data.y.len());
    assert!(matches!(
        parsed.model,
        OptimizationModelSpec::EquationModel { .. }
    ));

    let mut invalid_task = lm_task();
    invalid_task
        .get_mut("data")
        .unwrap()
        .insert("y".into(), Some(vec![Value::Vector(vec![1.0, 3.0])]));
    assert!(matches!(
        parse_optimization_task_from_document(&invalid_task),
        Err(OptimizationTaskError::InvalidField {
            section: "data",
            ..
        })
    ));
}

#[test]
fn native_runner_fits_equation_model_and_returns_reusable_result() {
    let result = run_optimization_task(
        parse_optimization_task_from_document(&lm_task()).expect("parse fitting task"),
    )
    .expect("fit should succeed");
    let parameters = result.solution_map().expect("fitted parameters");
    assert!((parameters["a"] - 2.0).abs() < 1e-5);
    assert!((parameters["b"] - 1.0).abs() < 1e-5);
    assert!(result.r_squared().unwrap_or(0.0) > 0.99999);
    if let UniversalFittingResult::LM(fit) = &result {
        let statistics = fit
            .backend()
            .last_statistics()
            .expect("native result retains its availability marker");
        assert!(!statistics.availability.is_collected());
        assert!(!statistics.timings_collected);
    } else {
        panic!("default fitting task should use LM");
    }
}

#[test]
fn fitting_task_telemetry_is_opt_in_and_reaches_lm_result() {
    let default_spec = parse_optimization_task_from_document(&lm_task()).expect("default task");
    assert_eq!(default_spec.solver_options.telemetry, None);

    let mut task = lm_task();
    task.insert(
        "solver".into(),
        section(&[("telemetry", vec![Value::String("detailed".into())])]),
    );
    let spec = parse_optimization_task_from_document(&task).expect("detailed mode parses");
    assert_eq!(
        spec.solver_options.telemetry,
        Some(TaskTelemetryMode::Detailed)
    );
    match run_optimization_task(spec).expect("fit with telemetry") {
        UniversalFittingResult::LM(fit) => {
            let statistics = fit
                .backend()
                .last_statistics()
                .expect("LM telemetry should be retained");
            assert!(statistics.availability.is_collected());
            assert!(statistics.timings_collected);
        }
        _ => panic!("expected LM result"),
    }
}

#[test]
fn gavin_task_telemetry_uses_its_native_mode_and_unknown_mode_is_rejected() {
    let mut task = lm_task();
    task.get_mut("model")
        .unwrap()
        .insert("method".into(), Some(vec![Value::String("gavin".into())]));
    task.insert(
        "solver".into(),
        section(&[("telemetry", vec![Value::String("counters".into())])]),
    );
    let spec = parse_optimization_task_from_document(&task).expect("Gavin mode parses");
    assert_eq!(
        spec.solver_options.telemetry,
        Some(TaskTelemetryMode::Counters)
    );
    assert!(matches!(
        run_optimization_task(spec).expect("Gavin fit"),
        UniversalFittingResult::GAVIN(_)
    ));

    task.get_mut("solver").unwrap().insert(
        "telemetry".into(),
        Some(vec![Value::String("verbose".into())]),
    );
    assert!(matches!(
        parse_optimization_task_from_document(&task),
        Err(OptimizationTaskError::InvalidField {
            section: "solver",
            field: "telemetry",
            ..
        })
    ));
}

#[test]
fn text_parser_accepts_compact_fitting_document() {
    let text = "data\nx: 0.0, 1.0, 2.0\ny: 1.0, 3.0, 5.0\nmodel\nmethod: lm\nparameters: a, b\ninitial_guess: 0.0, 0.0\nequations: a*x + b\n";
    let parsed = parse_optimization_task_from_text(text).expect("text task should parse");
    assert_eq!(parsed.data.x.len(), 3);
    assert!(matches!(
        parsed.model,
        OptimizationModelSpec::EquationModel { .. }
    ));
}

#[test]
fn parses_varpro_basis_and_invariant_model() {
    let task = HashMap::from([
        (
            "data".into(),
            section(&[
                ("x", vec![Value::Vector(vec![0.0, 1.0])]),
                ("y", vec![Value::Vector(vec![1.0, 0.5])]),
            ]),
        ),
        (
            "model".into(),
            section(&[
                ("method", vec![Value::String("varpro".into())]),
                ("parameters", vec![Value::String("tau".into())]),
                ("initial_guess", vec![Value::Vector(vec![1.0])]),
                ("basis_parameters", vec![Value::String("tau".into())]),
                ("basis_equations", vec![Value::String("exp(-x/tau)".into())]),
                ("invariants", vec![Value::String("1".into())]),
            ]),
        ),
    ]);
    let parsed = parse_optimization_task_from_document(&task).expect("valid VarPro task");
    match parsed.model {
        OptimizationModelSpec::VarProModel {
            basis, invariants, ..
        } => {
            assert_eq!(basis.len(), 1);
            assert_eq!(invariants.len(), 1);
        }
        _ => panic!("expected VarPro model"),
    }
}

#[test]
fn rejects_incompatible_fields_unknown_symbols_and_bad_gavin_weights() {
    let mut task = lm_task();
    task.get_mut("model").unwrap().insert(
        "basis_parameters".into(),
        Some(vec![Value::String("tau".into())]),
    );
    assert!(matches!(
        parse_optimization_task_from_document(&task),
        Err(OptimizationTaskError::UnsupportedField { .. })
    ));

    let mut task = lm_task();
    task.get_mut("model").unwrap().insert(
        "equations".into(),
        Some(vec![Value::String("a*x + missing".into())]),
    );
    assert!(matches!(
        parse_optimization_task_from_document(&task),
        Err(OptimizationTaskError::InvalidField {
            field: "equations",
            ..
        })
    ));

    let mut task = lm_task();
    task.get_mut("model")
        .unwrap()
        .insert("method".into(), Some(vec![Value::String("gavin".into())]));
    task.insert(
        "solver".into(),
        section(&[("gavin_weights", vec![Value::Vector(vec![1.0])])]),
    );
    assert!(matches!(
        parse_optimization_task_from_document(&task),
        Err(OptimizationTaskError::InvalidField {
            field: "gavin_weights",
            ..
        })
    ));
}
