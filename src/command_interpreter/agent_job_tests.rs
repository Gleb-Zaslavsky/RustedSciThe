use super::{
    ACCEPTANCE_REPORT_JSON_SCHEMA, AcceptanceCheck, AcceptanceCheckStatus, AcceptancePolicy,
    AgentJobError, LargeResultPolicy, evaluate_acceptance, parse_agent_job_json,
};
use crate::command_interpreter::task_contract::run_task_json_envelope;
use serde_json::{Value, json};

const TASK: &str = r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":0.1,"y0":1.0}}"#;

fn job(task: &str, fields: &str) -> String {
    format!(r#"{{"job_contract_version":1,"job_id":"test-1","task":{task}{fields}}}"#)
}

#[test]
fn parses_worker_metadata_and_preflights_embedded_task() {
    let input = job(
        TASK,
        r#", "acceptance":{"checks":["solver_converged","finite_result"]}, "resource_budget":{"wall_time_ms":5000,"memory_bytes":1048576,"cpu_threads":2}, "output_policy":{"large_result":"omit"}"#,
    );
    let parsed = parse_agent_job_json(&input).expect("valid job wrapper");
    assert_eq!(parsed.job_id.as_deref(), Some("test-1"));
    assert_eq!(parsed.task_inspection.task_family, "ivp");
    assert_eq!(parsed.resource_budget.wall_time_ms, Some(5000));
    assert_eq!(parsed.resource_budget.cpu_threads, Some(2));
    assert_eq!(parsed.output_policy.large_result, LargeResultPolicy::Omit);
    assert_eq!(
        parsed.acceptance.checks,
        vec![
            AcceptanceCheck::SolverConverged,
            AcceptanceCheck::FiniteResult
        ]
    );
}

#[test]
fn rejects_unknown_fields_versions_duplicates_and_invalid_budgets() {
    let unknown = job(TASK, r#", "surprise":true"#);
    assert!(matches!(
        parse_agent_job_json(&unknown),
        Err(AgentJobError::Envelope(_))
    ));

    let bad_version = format!(r#"{{"job_contract_version":2,"task":{TASK}}}"#);
    assert!(matches!(
        parse_agent_job_json(&bad_version),
        Err(AgentJobError::UnsupportedVersion { found: 2 })
    ));

    let long_multibyte_id = format!(
        r#"{{"job_contract_version":1,"job_id":"{}","task":{TASK}}}"#,
        "я".repeat(129)
    );
    assert!(matches!(
        parse_agent_job_json(&long_multibyte_id),
        Err(AgentJobError::InvalidJobId)
    ));
    let whitespace_job_id = format!(r#"{{"job_contract_version":1,"job_id":"  ","task":{TASK}}}"#);
    assert!(matches!(
        parse_agent_job_json(&whitespace_job_id),
        Err(AgentJobError::InvalidJobId)
    ));

    let zero_budget = job(TASK, r#", "resource_budget":{"wall_time_ms":0}"#);
    assert!(matches!(
        parse_agent_job_json(&zero_budget),
        Err(AgentJobError::InvalidResourceBudget {
            field: "wall_time_ms"
        })
    ));

    let duplicate_check = job(
        TASK,
        r#", "acceptance":{"checks":["finite_result","finite_result"]}"#,
    );
    assert!(matches!(
        parse_agent_job_json(&duplicate_check),
        Err(AgentJobError::DuplicateAcceptanceCheck { .. })
    ));

    let unknown_check = job(TASK, r#", "acceptance":{"checks":["residual_below"]}"#);
    assert!(matches!(
        parse_agent_job_json(&unknown_check),
        Err(AgentJobError::Envelope(_))
    ));
}

#[test]
fn validates_task_payload_through_existing_semantic_parser() {
    let invalid_task = r#"{"format_version":1,"task":{"solver":"IVP","method":"BDF"}}"#;
    let input = job(invalid_task, "");
    assert!(matches!(
        parse_agent_job_json(&input),
        Err(AgentJobError::Task(_))
    ));
}

#[test]
fn acceptance_checks_are_separate_from_solver_status() {
    let policy = AcceptancePolicy {
        checks: vec![
            AcceptanceCheck::SolverConverged,
            AcceptanceCheck::FiniteResult,
        ],
    };
    let successful = run_task_json_envelope(TASK);
    let report = evaluate_acceptance(&successful, &policy);
    assert_eq!(report.status, "accepted");
    assert!(
        report
            .checks
            .iter()
            .all(|check| check.status == AcceptanceCheckStatus::Passed)
    );

    let non_converged_task = r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2-2","initial_guess":100.0},"solver":{"method":"newton","max_iterations":1}}"#;
    let non_converged = run_task_json_envelope(non_converged_task);
    let report = evaluate_acceptance(&non_converged, &policy);
    assert_eq!(non_converged["status"], "not_converged");
    assert_eq!(report.status, "rejected");
    assert_eq!(report.checks[0].status, AcceptanceCheckStatus::Failed);
    assert_eq!(report.checks[1].status, AcceptanceCheckStatus::Passed);

    let execution_failure = run_task_json_envelope("not-json");
    let report = evaluate_acceptance(&execution_failure, &policy);
    assert_eq!(report.status, "not_evaluated");
    assert!(
        report
            .checks
            .iter()
            .all(|check| check.status == AcceptanceCheckStatus::NotEvaluated)
    );

    let mixed = evaluate_acceptance(&json!({"status":"not_converged", "result":{}}), &policy);
    assert_eq!(mixed.status, "rejected");
    assert_eq!(mixed.checks[0].status, AcceptanceCheckStatus::Failed);
    assert_eq!(mixed.checks[1].status, AcceptanceCheckStatus::NotEvaluated);

    let failed_with_known_non_finite = evaluate_acceptance(
        &json!({
            "status":"not_converged",
            "result":{"non_finite_values_encoded_as_null":1}
        }),
        &policy,
    );
    assert_eq!(failed_with_known_non_finite.status, "rejected");
    assert_eq!(
        failed_with_known_non_finite.checks[0].status,
        AcceptanceCheckStatus::Failed
    );
    assert_eq!(
        failed_with_known_non_finite.checks[1].status,
        AcceptanceCheckStatus::Failed
    );

    let non_finite_result = json!({
        "status": "succeeded",
        "result": {"non_finite_values_encoded_as_null": 2}
    });
    let report = evaluate_acceptance(
        &non_finite_result,
        &AcceptancePolicy {
            checks: vec![AcceptanceCheck::FiniteResult],
        },
    );
    assert_eq!(report.status, "rejected");
    assert_eq!(report.checks[0].status, AcceptanceCheckStatus::Failed);
    assert_eq!(report.to_json()["checks"][0]["observed"], 2);
}

#[test]
fn no_checks_means_not_requested_not_implicitly_accepted() {
    let report = evaluate_acceptance(&Value::Null, &AcceptancePolicy::default());
    assert_eq!(report.status, "not_requested");
    assert!(report.checks.is_empty());
}

#[test]
fn published_agent_job_schema_matches_runtime_acceptance_vocabulary() {
    let schema: Value = serde_json::from_str(super::AGENT_JOB_JSON_SCHEMA).unwrap();
    let task_schema: Value =
        serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
            .unwrap();
    let validator = jsonschema::options()
        .with_resource(
            "urn:rustedscithe:task-document:v1",
            jsonschema::Resource::from_contents(task_schema).unwrap(),
        )
        .build(&schema)
        .expect("agent job schema and its TaskDocument reference compile");

    let valid: Value = serde_json::from_str(&job(
        TASK,
        r#", "acceptance":{"checks":["solver_converged"]}, "resource_budget":{"cpu_threads":1}"#,
    ))
    .unwrap();
    assert!(validator.is_valid(&valid));
    assert!(parse_agent_job_json(&valid.to_string()).is_ok());

    let invalid: Value = serde_json::from_str(&job(
        TASK,
        r#", "acceptance":{"checks":["physical_temperature_valid"]}"#,
    ))
    .unwrap();
    assert!(!validator.is_valid(&invalid));
    assert!(parse_agent_job_json(&invalid.to_string()).is_err());

    let blank_id: Value = json!({"job_contract_version":1, "job_id":"   ", "task":serde_json::from_str::<Value>(TASK).unwrap()});
    assert!(!validator.is_valid(&blank_id));
    assert!(parse_agent_job_json(&blank_id.to_string()).is_err());
}

#[test]
fn published_family_examples_preflight_and_execute_through_public_json_paths() {
    let examples = [
        include_str!("../agent_interop/examples/ivp_decay.json"),
        include_str!("../agent_interop/examples/bvp_constant.json"),
        include_str!("../agent_interop/examples/nonlinear_root.json"),
        include_str!("../agent_interop/examples/fitting_linear.json"),
    ];
    for example in examples {
        parse_agent_job_json(&format!(
            "{{\"job_contract_version\":1,\"task\":{example}}}"
        ))
        .unwrap_or_else(|error| panic!("family example must pass typed preflight: {error}"));
        let response = run_task_json_envelope(example);
        assert_eq!(
            response["status"], "succeeded",
            "family example failed: {response}"
        );
    }

    let job = parse_agent_job_json(include_str!("../agent_interop/examples/job_ivp_decay.json"))
        .expect("outer job example should validate");
    assert_eq!(job.job_id.as_deref(), Some("example-decay-001"));
    let response = run_task_json_envelope(&job.task.to_string());
    assert_eq!(response["status"], "succeeded");
    let acceptance = super::evaluate_acceptance(&response, &job.acceptance);
    assert_eq!(acceptance.status, "accepted");
}

#[test]
fn canonical_examples_have_expected_numerical_results() {
    let ivp = run_task_json_envelope(include_str!("../agent_interop/examples/ivp_decay.json"));
    let ivp_rows = ivp["result"]["data"]["states"]
        .as_array()
        .expect("IVP state rows");
    let ivp_final = ivp_rows.last().expect("final IVP state row")[0]
        .as_f64()
        .expect("one-state final result");
    assert!(
        (ivp_final - (-0.1_f64).exp()).abs() < 5e-5,
        "IVP endpoint {ivp_final} differs from exp(-0.1)"
    );

    let bvp = run_task_json_envelope(include_str!("../agent_interop/examples/bvp_constant.json"));
    let values = bvp["result"]["data"]["values"]
        .as_array()
        .expect("BVP profile rows");
    assert!(values.iter().all(|row| {
        row[0]
            .as_f64()
            .is_some_and(|value| (value - 1.0).abs() < 1e-7)
    }));

    let root = run_task_json_envelope(include_str!(
        "../agent_interop/examples/nonlinear_root.json"
    ));
    let root_value = root["result"]["solution"][0]
        .as_f64()
        .expect("root solution");
    assert!((root_value - 2.0_f64.sqrt()).abs() < 1e-7);

    let fit = run_task_json_envelope(include_str!(
        "../agent_interop/examples/fitting_linear.json"
    ));
    let parameters = &fit["result"]["parameters"];
    assert!((parameters["a"].as_f64().unwrap() - 2.0).abs() < 1e-5);
    assert!((parameters["b"].as_f64().unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn acceptance_report_schema_matches_all_report_statuses() {
    let schema: Value = serde_json::from_str(ACCEPTANCE_REPORT_JSON_SCHEMA).unwrap();
    let validator =
        jsonschema::validator_for(&schema).expect("acceptance report schema should compile");
    let successful = run_task_json_envelope(TASK);
    let failed = run_task_json_envelope("not-json");
    let non_converged = run_task_json_envelope(
        r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2-2","initial_guess":100.0},"solver":{"method":"newton","max_iterations":1}}"#,
    );
    let policies = [
        AcceptancePolicy::default(),
        AcceptancePolicy {
            checks: vec![
                AcceptanceCheck::SolverConverged,
                AcceptanceCheck::FiniteResult,
            ],
        },
    ];
    for (envelope, policy) in [
        (&successful, &policies[0]),
        (&successful, &policies[1]),
        (&failed, &policies[1]),
        (&non_converged, &policies[1]),
    ] {
        let report = evaluate_acceptance(envelope, policy).to_json();
        assert!(validator.is_valid(&report), "schema rejected {report}");
    }
}
