//! Workload-level end-to-end exercise of the RST JSONL worker contract.
//!
//! Run with:
//! `cargo test --release --test ai_agent_interop_capstone -- --ignored --nocapture --test-threads=1`

use RustedSciThe::command_interpreter::agent_worker::{
    WorkerAotPolicy, serve_jsonl, serve_jsonl_with_aot_policy, serve_jsonl_with_control,
};
use RustedSciThe::command_interpreter::agent_worker_control::WorkerControlServer;
use RustedSciThe::command_interpreter::task_runner::TaskAotToolchain;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Cursor, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn job(id: &str, task: Value) -> Value {
    json!({
        "job_contract_version": 1,
        "job_id": id,
        "task": task,
        "acceptance": {"checks": ["solver_converged", "finite_result"]},
        "output_policy": {"large_result": "omit"}
    })
}

fn protocol_request(operation: &str, payload: Value) -> Value {
    json!({
        "worker_protocol_version": 1,
        "operation": operation,
        "payload": payload
    })
}

fn execute(request: Value) -> Value {
    let input = format!("{}\n", request);
    let mut reader = Cursor::new(input.into_bytes());
    let mut output = Vec::new();
    serve_jsonl(&mut reader, &mut output).expect("worker should process one JSONL request");
    let line = String::from_utf8(output).expect("worker response should be UTF-8");
    serde_json::from_str(line.trim()).expect("worker should return one JSON response")
}

fn execute_with_aot_policy(request: Value, policy: &WorkerAotPolicy) -> Value {
    let input = format!("{}\n", request);
    let mut reader = Cursor::new(input.into_bytes());
    let mut output = Vec::new();
    serve_jsonl_with_aot_policy(&mut reader, &mut output, policy)
        .expect("AOT-enabled worker should process one JSONL request");
    let line = String::from_utf8(output).expect("worker response should be UTF-8");
    serde_json::from_str(line.trim()).expect("worker should return one JSON response")
}

fn send_control(server: &WorkerControlServer, request: Value) -> Value {
    let mut stream = TcpStream::connect(server.address()).expect("connect worker control channel");
    serde_json::to_writer(&mut stream, &request).expect("write worker control request");
    stream
        .write_all(b"\n")
        .expect("terminate worker control request");
    let mut response = String::new();
    BufReader::new(stream)
        .read_line(&mut response)
        .expect("read worker control response");
    serde_json::from_str(&response).expect("parse worker control response")
}

fn assert_accepted<'a>(response: &'a Value, family: &str) -> &'a Value {
    assert_eq!(response["operation"], "execute", "{response}");
    assert_eq!(response["status"], "succeeded", "{response}");
    let task_result = &response["result"]["task_result"];
    assert_eq!(task_result["task_family"], family, "{response}");
    assert_eq!(task_result["status"], "succeeded", "{response}");
    assert_eq!(
        response["result"]["acceptance"]["status"], "accepted",
        "{response}"
    );
    task_result
}

fn work_size(family: &str, task: &Value, result: &Value) -> usize {
    match family {
        "ivp" | "bvp" => result["sample_count"].as_u64().unwrap_or_default() as usize,
        "nonlinear" => result["solution_length"].as_u64().unwrap_or_default() as usize,
        "optimization" => task["data"]["x"].as_array().map_or(0, Vec::len),
        _ => 0,
    }
}

fn capstone_report_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("RST_AI_AGENT_CAPSTONE_REPORT_DIR") {
        return PathBuf::from(path);
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX epoch")
        .as_millis();
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test_reports")
        .join("AI_agent_interop_capstone")
        .join(stamp.to_string())
}

fn family_checkpoint_path(report_dir: &Path, family: &str) -> PathBuf {
    report_dir.join(format!("family_{family}.json"))
}

fn load_family_checkpoint(report_dir: &Path, family: &str, job: &Value) -> Option<(Value, f64)> {
    let bytes = std::fs::read(family_checkpoint_path(report_dir, family)).ok()?;
    let checkpoint: Value = serde_json::from_slice(&bytes).ok()?;
    let input_digest = blake3::hash(&serde_json::to_vec(job).ok()?)
        .to_hex()
        .to_string();
    let profile = std::env::var("RST_TEST_REPORT_PROFILE").unwrap_or_default();
    if checkpoint["schema_version"] != 1
        || checkpoint["family"] != family
        || checkpoint["profile"] != profile
        || checkpoint["status"] != "passed"
        || checkpoint["input_blake3_256"] != input_digest
    {
        return None;
    }
    Some((
        checkpoint["response"].clone(),
        checkpoint["wall_ms"].as_f64()?,
    ))
}

fn save_family_checkpoint(
    report_dir: &Path,
    family: &str,
    job: &Value,
    response: &Value,
    wall_ms: f64,
) {
    let input_digest = blake3::hash(&serde_json::to_vec(job).expect("job should serialize"))
        .to_hex()
        .to_string();
    let profile = std::env::var("RST_TEST_REPORT_PROFILE").unwrap_or_default();
    let checkpoint = json!({
        "schema_version": 1,
        "family": family,
        "profile": profile,
        "status": "passed",
        "input_blake3_256": input_digest,
        "wall_ms": wall_ms,
        "response": response
    });
    let bytes = serde_json::to_vec_pretty(&checkpoint).expect("checkpoint should serialize");
    let path = family_checkpoint_path(report_dir, family);
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, bytes).expect("family checkpoint should be written");
    if path.exists() {
        std::fs::remove_file(&path).expect("stale family checkpoint should be replaced");
    }
    std::fs::rename(temporary, path).expect("family checkpoint should be published atomically");
}

fn execute_checkpointed_job(
    report_dir: &Path,
    checkpoint: &str,
    family: &str,
    job: &Value,
) -> (Value, f64, bool) {
    let validation = execute(protocol_request(
        "validate",
        json!({"task":job["task"].clone()}),
    ));
    assert_eq!(
        validation["status"], "succeeded",
        "{checkpoint}: {validation}"
    );
    assert_eq!(
        validation["result"]["preflight"]["numerical_execution"], "not_started",
        "{checkpoint}: validation must not start numerical work"
    );

    if let Some((response, elapsed_ms)) = load_family_checkpoint(report_dir, checkpoint, job) {
        assert_accepted(&response, family);
        return (response, elapsed_ms, true);
    }

    let started = Instant::now();
    let response = execute(protocol_request("execute", json!({"job":job})));
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_accepted(&response, family);
    save_family_checkpoint(report_dir, checkpoint, job, &response, elapsed_ms);
    (response, elapsed_ms, false)
}

fn ivp_job() -> Value {
    job(
        "capstone-ivp-robertson-parameter-sweep",
        json!({
            "format_version": 1,
            "task": {"solver":"IVP", "method":"BDF"},
            "equations": {
                "arg":"t",
                "unknowns":["A", "B", "C"],
                "rhs":[
                    "-k1*A + k2*B*C",
                    "k1*A - k2*B*C - k3*B^2",
                    "k3*B^2"
                ],
                "parameters":["k1", "k2", "k3"],
                "parameter_values":[0.04, 10000.0, 30000000.0]
            },
            "initial_conditions":{"t0":0.0, "t_end":10.0, "y0":[1.0, 0.0, 0.0]},
            "continuation":{
                "parameter":"k1",
                "values":[0.04, 0.05],
                "mode":"fresh",
                "restart_each":true
            },
            "solver_options":{
                "rtol":1e-6,
                "atol":1e-10,
                "max_step":0.1,
                "symbolic_frontend":"ExprLegacy",
                "symbolic_execution":"Lambdify",
                "matrix_layout":"sparse",
                "telemetry":"detailed"
            }
        }),
    )
}

fn bvp_job() -> Value {
    job(
        "capstone-bvp-sparse-parameter-sweep",
        json!({
            "format_version": 1,
            "task":{"solver":"BVP_sci", "method":"Sparse"},
            "equations":{
                "arg":"x",
                "unknowns":["y"],
                "rhs":["rate"],
                "parameters":["rate"],
                "parameter_values":[0.5]
            },
            "boundary_conditions":{"y_left":0.0},
            "mesh":{"t0":0.0, "t_end":1.0, "n_steps":2000},
            "initial_guess":{"y":0.0},
            "continuation":{"parameter":"rate", "values":[0.5, 1.25], "mode":"fresh"},
            "solver_options":{"tolerance":1e-7, "max_iterations":100, "telemetry":"counters"}
        }),
    )
}

fn combustion_bvp_job(intervals: usize, frontend: &str) -> Value {
    job(
        &format!("capstone-combustion-bvp-six-state-{intervals}-intervals-{frontend}"),
        json!({
            "format_version": 1,
            "task":{"solver":"BVP_Damp", "strategy":"Damped", "method":"Banded", "frontend":frontend, "execution":"Lambdify"},
            "equations":{
                "arg":"x",
                "unknowns":["Teta", "q", "C0", "J0", "C1", "J1"],
                "rhs":[
                    "q/0.07",
                    "q*0.0090168 - (3000.0*1000.0*0.034)*(130000.0*exp(-(5000.0*4.184)/(8.314*(Teta*600.0 + 600.0)))*C0*((0.0342*(2000000.0)/(8.314*1500.0))/0.342))*((0.0003)^2/600.0)",
                    "J0/0.000288",
                    "J0*0.00150 - (0.0342*(-1.0)*(130000.0*exp(-(5000.0*4.184)/(8.314*(Teta*600.0 + 600.0)))*C0*((0.0342*(2000000.0)/(8.314*1500.0))/0.342))*((0.0342*(2000000.0)/(8.314*1500.0))/0.0342))*((0.0003)^2)",
                    "J1/0.000288",
                    "J1*0.00150 - (0.0342*(130000.0*exp(-(5000.0*4.184)/(8.314*(Teta*600.0 + 600.0)))*C0*((0.0342*(2000000.0)/(8.314*1500.0))/0.342))*((0.0342*(2000000.0)/(8.314*1500.0))/0.0342))*((0.0003)^2)"
                ]
            },
            "boundary_conditions":{
                "Teta_left":0.6666666667, "q_right":1e-10,
                "C0_left":1.0, "J0_right":1e-7,
                "C1_left":1e-3, "J1_right":1e-7
            },
            "mesh":{"t0":0.0, "t_end":1.0, "n_steps":intervals},
            "initial_guess":{"Teta":0.99, "q":0.99, "C0":0.99, "J0":0.99, "C1":0.99, "J1":0.99},
            "solver_options":{
                "tolerance":1e-6,
                "max_iterations":100,
                "generated_backend":"banded_lambdify",
                "banded_linear_solver":"lapack",
                "refinement_steps":0,
                "telemetry":"detailed"
            }
        }),
    )
}

fn kithe_26_state_combustion_bvp_job() -> Value {
    const RHS: [&str; 26] = [
        "q / 0.05",
        "q * 49.10419167354888 - (-103030398415.33754 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + -4128633572188.5405 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)) + 2280232215.9957895 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + 33299161969.77031 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + 712690917.1313457 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16) * 0.00000033333333333333335",
        "J0 / 0.000004844744729497463",
        "412.8184479613349 * J0 - 0.0001 * (28011.995829600797 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + 288944.2885029702 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)) + 423.17015412771735 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + 56060.951656390236 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + 624.6888893573005 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16)",
        "J1 / 0.000004844744729497463",
        "412.8184479613349 * J1 - 0.0001 * (-1239.9012171458817 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + -164260.26626664176 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + -1830.3571428571438 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16 + -82075.98618433015 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + -846615.4134633419 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)))",
        "J2 / 0.000004844744729497463",
        "412.8184479613349 * J2 - 0.0001 * (104440.78855307552 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)) + 1163.7868833320908 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16 + 52185.966291128585 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + 538299.2697684199 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)) + 788.3602272775099 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73)",
        "J3 / 0.000004844744729497463",
        "412.8184479613349 * J3 + 0.004188137016775251 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16",
        "J4 / 0.000004844744729497463",
        "412.8184479613349 * J4 + -0.008376274033550502 * C1 * C3 * exp(-1210.310319942266 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 1.16",
        "J5 / 0.000004844744729497463",
        "412.8184479613349 * J5 + -0.05070442518530647 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73",
        "J6 / 0.000004844744729497463",
        "412.8184479613349 * J6 + -6.337213343186177 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200))",
        "J7 / 0.000004844744729497463",
        "412.8184479613349 * J7 - 0.0001 * (-61494.109368260986 * C1 * C7 * exp(-6000.220351214819 / (300 * Teta + 1200)) + 634312.9488745617 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200)))",
        "J8 / 0.000004844744729497463",
        "412.8184479613349 * J8",
        "J9 / 0.000004844744729497463",
        "412.8184479613349 * J9 + 5.965541506821332 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200))",
        "J10 / 0.000004844744729497463",
        "412.8184479613349 * J10 + 61.494109368260986 * C1 * C10 * exp(-19702.140967043542 / (300 * Teta + 1200))",
        "J11 / 0.000004844744729497463",
        "412.8184479613349 * J11 - 0.0001 * (-478.67341611241034 * C1 * C11 * exp(560.1145056531153 / (300 * Teta + 1200)) * (300 * Teta + 1200) ^ 0.73 + 63413.94112538935 * C1 * C9 * exp(-909.8715419773874 / (300 * Teta + 1200)))",
    ];
    const LEFT_VALUES: [f64; 12] = [
        0.0,
        0.018405139435376694,
        0.0,
        3.5094811081461935e-7,
        0.0,
        0.01254430598477255,
        0.0,
        0.002298292330465739,
        0.7656249768090929,
        5.570256770757038e-6,
        0.20052924374725334,
        0.0005921204881571657,
    ];

    let rhs = RHS
        .iter()
        .map(|source| {
            let mut expression = source.replace("Teta", "u0").replace("q", "u1");
            for index in (0..12).rev() {
                expression = expression
                    .replace(&format!("C{index}"), &format!("u{}", 2 + 2 * index))
                    .replace(&format!("J{index}"), &format!("u{}", 3 + 2 * index));
            }
            expression
        })
        .collect::<Vec<_>>();
    let unknowns = (0..26).map(|state| format!("u{state}")).collect::<Vec<_>>();
    let mut boundary_conditions = serde_json::Map::new();
    boundary_conditions.insert("u0_left".into(), json!(0.0));
    boundary_conditions.insert("u1_right".into(), json!(0.0));
    let mut initial_guess = serde_json::Map::new();
    initial_guess.insert("u0".into(), json!(0.0));
    initial_guess.insert("u1".into(), json!(0.0));
    let mut bounds = serde_json::Map::new();
    for state in 0..26 {
        let name = format!("u{state}");
        if state >= 2 && state % 2 == 0 {
            let value = LEFT_VALUES[(state - 2) / 2];
            boundary_conditions.insert(format!("{name}_left"), json!(value));
            initial_guess.insert(name.clone(), json!(value));
        } else if state >= 3 {
            boundary_conditions.insert(format!("{name}_right"), json!(0.0));
            initial_guess.insert(name.clone(), json!(0.0));
        }
        let interval = if state == 0 {
            [-1.0, 3.0]
        } else if state % 2 == 1 {
            [-1.0e4, 1.0e4]
        } else {
            [-0.1, 1.1]
        };
        bounds.insert(name, json!(interval));
    }

    job(
        "capstone-kithe-combustion-26-state-banded",
        json!({
            "format_version":1,
            "task":{"solver":"BVP_Damp", "strategy":"Damped", "method":"Banded", "scheme":"trapezoid", "frontend":"ExprLegacy", "execution":"Lambdify"},
            "equations":{"arg":"s", "unknowns":unknowns, "rhs":rhs},
            "boundary_conditions":boundary_conditions,
            "mesh":{"t0":0.0, "t_end":1.0, "n_steps":33},
            "initial_guess":initial_guess,
            "bounds":bounds,
            "solver_options":{
                "tolerance":1e-5, "max_iterations":150,
                "generated_backend":"banded_lambdify", "banded_linear_solver":"lapack",
                "telemetry":"detailed"
            }
        }),
    )
}

fn combustion_ivp_job(method: &str, frontend: &str) -> Value {
    job(
        &format!("capstone-combustion-ivp-{method}-{frontend}"),
        json!({
            "format_version":1,
            "task":{"solver":"IVP", "method":method},
            "equations":{
                "arg":"t",
                "unknowns":["A", "B", "T"],
                "rhs":[
                    "-k*exp(-E/(R*T))*A*A",
                    "0.5*k*exp(-E/(R*T))*A*A - kloss*B",
                    "Qcrho*k*exp(-E/(R*T))*A*A - cooling*(T-T0)"
                ],
                "parameters":["k", "E", "R", "T0", "Qcrho", "kloss", "cooling"],
                "parameter_values":[10000000.0, 50000.0, 8.314, 300.0, 500.0, 0.0, 0.5]
            },
            "initial_conditions":{"t0":0.0, "t_end":0.001, "y0":[1.0, 0.0, 300.0]},
            "solver_options":{
                "rtol":1e-7, "atol":1e-10, "max_step":0.0001,
                "symbolic_frontend":frontend, "symbolic_execution":"Lambdify",
                "matrix_layout":"sparse", "telemetry":"detailed"
            }
        }),
    )
}

fn nonlinear_job(dimension: usize) -> Value {
    let exact = (0..dimension)
        .map(|index| 1.0 + index as f64 * 0.005)
        .collect::<Vec<_>>();
    let equations = (0..dimension)
        .map(|index| {
            let previous = if index == 0 { dimension - 1 } else { index - 1 };
            format!(
                "x{index}^2 + 0.02*x{previous}*x{index} - {:.17}",
                exact[index].powi(2) + 0.02 * exact[previous] * exact[index]
            )
        })
        .collect::<Vec<_>>();
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let initial_guess = exact.iter().map(|value| value * 0.95).collect::<Vec<_>>();
    job(
        &format!("capstone-nonlinear-coupled-root-{dimension}"),
        json!({
            "format_version":1,
            "problem":{
                "formulation":"root",
                "variables":variables,
                "equations":equations,
                "initial_guess":initial_guess
            },
            "solver":{"method":"newton", "tolerance":1e-10, "max_iterations":100, "telemetry":"counters"}
        }),
    )
}

fn fitting_job(samples: usize) -> Value {
    let x = (0..samples)
        .map(|index| 5.0 * index as f64 / (samples - 1) as f64)
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|value| 2.0 * (-0.7 * value).exp() + 0.25)
        .collect::<Vec<_>>();
    job(
        &format!("capstone-fitting-exponential-{samples}-observations"),
        json!({
            "format_version":1,
            "data":{"arg":"x", "x":x, "y":y},
            "model":{
                "method":"lm",
                "parameters":["a", "b", "c"],
                "initial_guess":[1.2, 0.4, 0.1],
                "equations":"a*exp(-b*x)+c"
            },
            "solver":{"tolerance":1e-10, "max_iterations":200, "telemetry":"counters"}
        }),
    )
}

#[test]
#[ignore = "workload-level JSON worker acceptance gate; run explicitly via scripts/ai_agent_interop_capstone.ps1"]
fn four_family_json_worker_capstone() {
    let report_dir = capstone_report_dir();
    std::fs::create_dir_all(&report_dir).expect("capstone report directory should be created");
    let capabilities = execute(protocol_request("capabilities", json!({})));
    assert_eq!(capabilities["status"], "succeeded", "{capabilities}");
    assert!(capabilities["result"]["task_capabilities"].is_object());
    let presets = execute(protocol_request("presets", json!({})));
    assert_eq!(presets["status"], "succeeded", "{presets}");
    assert!(presets["result"]["presets"].is_array());
    let invalid_task = execute(protocol_request(
        "validate",
        json!({"task":{"format_version":1, "task":{"solver":"IVP"}}}),
    ));
    assert_eq!(invalid_task["status"], "rejected", "{invalid_task}");
    assert_eq!(
        invalid_task["error"]["code"],
        "worker.task.validation_failed"
    );
    assert!(invalid_task["error"]["details"]["code"].is_string());

    let workloads = [
        ("ivp", ivp_job()),
        ("bvp", bvp_job()),
        ("nonlinear", nonlinear_job(64)),
        ("optimization", fitting_job(100_001)),
    ];
    let mut rows = Vec::new();
    let mut reports = Vec::new();

    for (family, job) in workloads {
        let task = job["task"].clone();
        let validation = execute(protocol_request("validate", json!({"task":task})));
        assert_eq!(validation["status"], "succeeded", "{family}: {validation}");
        assert_eq!(
            validation["result"]["preflight"]["numerical_execution"],
            "not_started"
        );

        let (response, elapsed_ms, resumed) =
            match load_family_checkpoint(&report_dir, family, &job) {
                Some((response, elapsed_ms)) => (response, elapsed_ms, true),
                None => {
                    let started = Instant::now();
                    let response = execute(protocol_request("execute", json!({"job":job})));
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    (response, elapsed_ms, false)
                }
            };
        let task_result = assert_accepted(&response, family);
        let result = &task_result["result"];
        match family {
            "ivp" => {
                assert_eq!(result["continuation"]["segment_count"], 2);
                let states = result["data"]["states"]
                    .as_array()
                    .expect("inline IVP states");
                let final_state = states.last().unwrap().as_array().unwrap();
                let conserved = final_state
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .sum::<f64>();
                assert!(
                    (conserved - 1.0).abs() <= 1e-6,
                    "Robertson mass balance={conserved}"
                );
                assert!(
                    response["diagnostics"].is_object(),
                    "detailed continuation diagnostics missing: {response}"
                );
            }
            "bvp" => {
                assert_eq!(result["sample_count"], 2000);
                assert!(result["continuation"].is_null());
                assert_eq!(task_result["telemetry"]["mode"], "counters");
                assert!(
                    task_result["telemetry"]["metrics"]["linear_solves"]["value"]
                        .as_u64()
                        .unwrap_or_default()
                        > 0
                );
                let values = result["data"]["values"]
                    .as_array()
                    .expect("inline BVP values");
                let final_value = values[0]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .as_f64()
                    .unwrap();
                assert!(
                    (final_value - 1.25).abs() <= 1e-5,
                    "BVP endpoint={final_value}"
                );
            }
            "nonlinear" => {
                assert_eq!(result["solution_length"], 64);
                assert!(result["residual_norm"].as_f64().unwrap() <= 1e-8);
                assert_eq!(task_result["telemetry"]["mode"], "counters");
                assert!(
                    task_result["telemetry"]["metrics"]["residual_evaluations"]["value"]
                        .as_u64()
                        .unwrap_or_default()
                        > 0
                );
                let solution = result["solution"].as_array().expect("inline root solution");
                assert!((solution[0].as_f64().unwrap() - 1.0).abs() < 1e-7);
            }
            "optimization" => {
                assert_eq!(result["parameter_count"], 3);
                assert_eq!(task_result["telemetry"]["mode"], "counters");
                assert!(
                    task_result["telemetry"]["metrics"]["residual_evaluations"]["value"]
                        .as_u64()
                        .unwrap_or_default()
                        > 0
                );
                let parameters = result["parameters"].as_object().expect("fitted parameters");
                assert!((parameters["a"].as_f64().unwrap() - 2.0).abs() < 1e-5);
                assert!((parameters["b"].as_f64().unwrap() - 0.7).abs() < 1e-5);
                assert!((parameters["c"].as_f64().unwrap() - 0.25).abs() < 1e-5);
            }
            _ => unreachable!(),
        }
        rows.push(format!(
            "| {family} | {} | {} | {:.3} | {}{} |",
            result["solver"]
                .as_str()
                .or_else(|| result["method"].as_str())
                .or_else(|| result["formulation"].as_str())
                .unwrap_or("native"),
            work_size(family, &task, result),
            elapsed_ms,
            task_result["telemetry"]["mode"].as_str().unwrap_or("n/a"),
            if resumed { " (resumed)" } else { "" },
        ));
        if !resumed {
            save_family_checkpoint(&report_dir, family, &job, &response, elapsed_ms);
        }
        reports.push(response);
    }

    let combustion_cases = [
        (
            "bvp-combustion-301-expr",
            combustion_bvp_job(300, "ExprLegacy"),
            301,
            false,
        ),
        (
            "bvp-combustion-551-atom",
            combustion_bvp_job(550, "AtomView"),
            551,
            false,
        ),
        (
            "bvp-combustion-1001-bounded-atom",
            combustion_bvp_job(1000, "AtomView"),
            1001,
            true,
        ),
    ];
    for (checkpoint, combustion_job, expected_nodes, expect_omission) in combustion_cases {
        let (combustion_response, elapsed_ms, resumed) =
            execute_checkpointed_job(&report_dir, checkpoint, "bvp", &combustion_job);
        let combustion_result = assert_accepted(&combustion_response, "bvp");
        let telemetry = &combustion_result["telemetry"];
        assert_eq!(telemetry["mode"], "detailed", "{checkpoint}: {telemetry}");
        for metric in [
            "nonlinear_iterations",
            "residual_evaluations",
            "jacobian_evaluations",
            "linear_solves",
            "factorizations",
            "mesh_refinements",
        ] {
            assert_eq!(
                telemetry["metrics"][metric]["availability"], "measured",
                "{checkpoint}: {telemetry}"
            );
            let value = telemetry["metrics"][metric]["value"]
                .as_u64()
                .unwrap_or_default();
            if metric != "mesh_refinements" {
                assert!(
                    value > 0,
                    "{checkpoint}: missing native {metric} count: {telemetry}"
                );
            }
        }
        assert_eq!(
            combustion_response["diagnostics"]["kind"], "bvp_damped_newton_diagnostics",
            "{checkpoint}: detailed decision trace missing"
        );
        assert!(
            combustion_response["diagnostics"]["segments"][0]["trace"]["events"]
                .as_array()
                .is_some_and(|events| !events.is_empty()),
            "{checkpoint}: detailed Newton event trace is empty"
        );
        let result = &combustion_result["result"];
        let combustion_data = &result["data"];
        assert_eq!(result["sample_count"], expected_nodes);
        assert_eq!(result["state_count"], 6);
        if expect_omission {
            assert_eq!(result["data_omitted"], true, "{checkpoint}: {result}");
            assert_eq!(result["data"], Value::Null, "{checkpoint}: {result}");
            assert_eq!(
                result["omission_reason"], "inline_value_limit_exceeded",
                "{checkpoint}: {result}"
            );
            assert_eq!(result["numeric_value_count"], expected_nodes * 7);
            assert_eq!(result["non_finite_values_encoded_as_null"], 0);
        } else {
            assert_eq!(combustion_data["mesh_orientation"], "rows");
            let combustion_mesh = combustion_data["mesh"]
                .as_array()
                .expect("combustion mesh should be inline");
            let combustion_profile = combustion_data["values"]
                .as_array()
                .expect("combustion profile should be inline for independent boundary checks");
            assert_eq!(combustion_mesh.len(), combustion_profile.len());
            assert_eq!(combustion_profile.len(), expected_nodes);
            let left = combustion_profile.first().unwrap().as_array().unwrap();
            let right = combustion_profile.last().unwrap().as_array().unwrap();
            for (actual, expected) in [
                (left[0].as_f64().unwrap(), 0.6666666667),
                (left[2].as_f64().unwrap(), 1.0),
                (left[4].as_f64().unwrap(), 1e-3),
                (right[1].as_f64().unwrap(), 1e-10),
                (right[3].as_f64().unwrap(), 1e-7),
                (right[5].as_f64().unwrap(), 1e-7),
            ] {
                assert!(
                    (actual - expected).abs() <= 1e-6,
                    "{checkpoint}: combustion boundary {actual} != {expected}"
                );
            }
        }
        rows.push(format!(
            "| {checkpoint} | BVP_Damp/Banded/Lambdify | 6 x {expected_nodes} | {:.3} | detailed{} |",
            elapsed_ms,
            if resumed { " (resumed)" } else { "" },
        ));
        reports.push(combustion_response);
    }

    let (kithe_response, kithe_elapsed_ms, kithe_resumed) = execute_checkpointed_job(
        &report_dir,
        "bvp-combustion-kithe-26-state",
        "bvp",
        &kithe_26_state_combustion_bvp_job(),
    );
    let kithe_task_result = assert_accepted(&kithe_response, "bvp");
    assert_eq!(kithe_task_result["telemetry"]["mode"], "detailed");
    for metric in [
        "nonlinear_iterations",
        "residual_evaluations",
        "jacobian_evaluations",
        "linear_solves",
        "factorizations",
    ] {
        assert_eq!(
            kithe_task_result["telemetry"]["metrics"][metric]["availability"], "measured",
            "KiThe 26-state BVP is missing {metric}: {}",
            kithe_task_result["telemetry"]
        );
        assert!(
            kithe_task_result["telemetry"]["metrics"][metric]["value"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "KiThe 26-state BVP reports no {metric}: {}",
            kithe_task_result["telemetry"]
        );
    }
    assert_eq!(
        kithe_response["diagnostics"]["kind"],
        "bvp_damped_newton_diagnostics"
    );
    let kithe_result = &kithe_task_result["result"];
    assert_eq!(kithe_result["sample_count"], 34);
    assert_eq!(kithe_result["state_count"], 26);
    let kithe_values = kithe_result["data"]["values"]
        .as_array()
        .expect("KiThe 26-state profile should fit inside the bounded inline response");
    assert_eq!(kithe_values.len(), 34);
    for (node, values) in kithe_values.iter().enumerate() {
        let values = values
            .as_array()
            .expect("KiThe node should contain 26 states");
        assert_eq!(values.len(), 26);
        for (state, value) in values.iter().enumerate() {
            let value = value
                .as_f64()
                .unwrap_or_else(|| panic!("KiThe u{state} at node {node} is not finite: {value}"));
            let (lower, upper) = if state == 0 {
                (-1.0, 3.0)
            } else if state % 2 == 1 {
                (-1.0e4, 1.0e4)
            } else {
                (-0.1, 1.1)
            };
            assert!(
                value.is_finite() && (lower..=upper).contains(&value),
                "KiThe state u{state} at node {node}={value} lies outside [{lower}, {upper}]"
            );
        }
    }
    let left = kithe_values.first().unwrap().as_array().unwrap();
    let right = kithe_values.last().unwrap().as_array().unwrap();
    assert_eq!(left.len(), 26);
    assert_eq!(right.len(), 26);
    for (species, expected) in [
        0.0,
        0.018405139435376694,
        0.0,
        3.5094811081461935e-7,
        0.0,
        0.01254430598477255,
        0.0,
        0.002298292330465739,
        0.7656249768090929,
        5.570256770757038e-6,
        0.20052924374725334,
        0.0005921204881571657,
    ]
    .into_iter()
    .enumerate()
    {
        let state = 2 + 2 * species;
        let actual = left[state].as_f64().expect("finite left boundary value");
        assert!(
            (actual - expected).abs() <= 1e-8,
            "KiThe left boundary u{state}: expected {expected}, got {actual}"
        );
    }
    for state in (1..26).step_by(2) {
        let actual = right[state].as_f64().expect("finite right boundary value");
        assert!(
            actual.abs() <= 1e-8,
            "KiThe right boundary u{state}: expected zero, got {actual}"
        );
    }
    let right_temperature = right[0].as_f64().expect("finite KiThe right temperature");
    assert!(
        right_temperature.is_finite() && right_temperature > 0.0,
        "KiThe combustion temperature should remain positive, got {right_temperature}"
    );
    rows.push(format!(
        "| bvp-combustion-kithe-26-state | BVP_Damp/Banded/Lambdify | 26 x 34 | {:.3} | detailed{} |",
        kithe_elapsed_ms,
        if kithe_resumed { " (resumed)" } else { "" },
    ));
    reports.push(kithe_response);

    for method in ["BDF", "Radau5", "LSODE2"] {
        for frontend in ["ExprLegacy", "AtomView"] {
            let checkpoint = format!("ivp-combustion-{method}-{frontend}");
            let combustion_job = combustion_ivp_job(method, frontend);
            let (response, elapsed_ms, resumed) =
                execute_checkpointed_job(&report_dir, &checkpoint, "ivp", &combustion_job);
            let task_result = assert_accepted(&response, "ivp");
            assert_eq!(task_result["telemetry"]["mode"], "detailed", "{checkpoint}");
            for metric in ["accepted_time_steps", "residual_evaluations"] {
                assert_eq!(
                    task_result["telemetry"]["metrics"][metric]["availability"], "measured",
                    "{checkpoint}: {}",
                    task_result["telemetry"]
                );
                assert!(
                    task_result["telemetry"]["metrics"][metric]["value"]
                        .as_u64()
                        .unwrap_or_default()
                        > 0,
                    "{checkpoint}: missing {metric}"
                );
            }
            let result = &task_result["result"];
            let states = result["data"]["states"]
                .as_array()
                .expect("combustion IVP states should be inline");
            let initial = states.first().unwrap().as_array().unwrap();
            let final_state = states.last().unwrap().as_array().unwrap();
            assert_eq!(initial.len(), 3);
            assert!(
                final_state
                    .iter()
                    .all(|value| value.as_f64().unwrap().is_finite())
            );
            assert!(
                final_state[0].as_f64().unwrap() > 0.0,
                "{checkpoint}: A depleted below zero"
            );
            assert!(
                final_state[1].as_f64().unwrap() >= -1e-8,
                "{checkpoint}: B became negative"
            );
            assert!(
                final_state[2].as_f64().unwrap() >= 300.0,
                "{checkpoint}: combustion cooled below inlet"
            );
            rows.push(format!(
                "| {checkpoint} | {method}/{frontend}/Lambdify | 3 x {} | {:.3} | detailed{} |",
                result["sample_count"].as_u64().unwrap_or_default(),
                elapsed_ms,
                if resumed { " (resumed)" } else { "" },
            ));
            reports.push(response);
        }
    }

    let (large_nonlinear, elapsed_ms, resumed) = execute_checkpointed_job(
        &report_dir,
        "nonlinear-coupled-root-256",
        "nonlinear",
        &nonlinear_job(256),
    );
    let large_root = assert_accepted(&large_nonlinear, "nonlinear");
    assert_eq!(large_root["result"]["solution_length"], 256);
    assert!(large_root["result"]["residual_norm"].as_f64().unwrap() <= 1e-8);
    rows.push(format!(
        "| nonlinear-coupled-root-256 | Newton | 256 | {:.3} | counters{} |",
        elapsed_ms,
        if resumed { " (resumed)" } else { "" },
    ));
    reports.push(large_nonlinear);

    let (large_fit, elapsed_ms, resumed) = execute_checkpointed_job(
        &report_dir,
        "optimization-exponential-150001",
        "optimization",
        &fitting_job(150_001),
    );
    let large_fit_result = assert_accepted(&large_fit, "optimization");
    let fit_parameters = large_fit_result["result"]["parameters"]
        .as_object()
        .expect("large fit should return parameters");
    assert!((fit_parameters["a"].as_f64().unwrap() - 2.0).abs() < 1e-5);
    assert!((fit_parameters["b"].as_f64().unwrap() - 0.7).abs() < 1e-5);
    rows.push(format!(
        "| optimization-exponential-150001 | LM | 150001 observations | {:.3} | counters{} |",
        elapsed_ms,
        if resumed { " (resumed)" } else { "" },
    ));
    reports.push(large_fit);

    let aot_request = json!({
        "format_version":1,
        "task":{"solver":"IVP", "method":"BDF"},
        "equations":{"arg":"t", "y":"-y"},
        "initial_conditions":{"t0":0.0, "t_end":1.0, "y0":1.0},
        "solver_options":{"symbolic_execution":"aot", "aot_toolchain":"rust"}
    });
    let rejected_aot = execute(protocol_request(
        "execute",
        json!({"job":job("capstone-aot-policy-rejection", aot_request)}),
    ));
    assert_eq!(rejected_aot["status"], "rejected", "{rejected_aot}");
    assert_eq!(rejected_aot["error"]["code"], "worker.execute.rejected");
    assert_eq!(
        rejected_aot["error"]["details"]["code"],
        "worker.aot_policy.toolchain_not_allowed"
    );

    let failing_job = job(
        "capstone-typed-native-failure",
        json!({
            "format_version":1,
            "task":{"solver":"IVP", "method":"BDF"},
            "equations":{"arg":"t", "y":"1/y"},
            "initial_conditions":{"t0":0.0, "t_end":1.0, "y0":0.0},
            "solver_options":{"rtol":1e-6, "atol":1e-9, "telemetry":"detailed"}
        }),
    );
    let failed = execute(protocol_request("execute", json!({"job":failing_job})));
    assert_eq!(failed["status"], "failed", "{failed}");
    assert_eq!(failed["result"]["task_result"]["status"], "failed");
    assert_ne!(failed["result"]["acceptance"]["status"], "accepted");
    assert_eq!(
        failed["result"]["task_result"]["telemetry"]["outcome"], "partial",
        "{failed}"
    );
    assert_eq!(
        failed["result"]["task_result"]["telemetry"]["metrics"]["residual_evaluations"]["availability"],
        "partial",
        "{failed}"
    );
    assert!(
        failed["result"]["task_result"]["error"].is_object(),
        "{failed}"
    );
    reports.push(failed);

    let cancellation_server = WorkerControlServer::bind_loopback()
        .expect("worker cancellation control should bind loopback");
    let cancellation_job = job(
        "capstone-cooperative-cancellation",
        json!({
            "format_version":1,
            "task":{"solver":"IVP", "method":"BDF"},
            "equations":{"arg":"t", "y":"-y"},
            "initial_conditions":{"t0":0.0, "t_end":1000.0, "y0":1.0},
            "solver_options":{"rtol":1e-7, "atol":1e-10, "max_step":1e-5, "telemetry":"detailed"}
        }),
    );
    let cancellation_input = format!(
        "{}\n",
        protocol_request("execute", json!({"job":cancellation_job}))
    );
    let cancellation_output = std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let mut reader = Cursor::new(cancellation_input.into_bytes());
            let mut output = Vec::new();
            serve_jsonl_with_control(&mut reader, &mut output, &cancellation_server)
                .expect("cancellable worker should serve the job");
            output
        });
        let mut active = false;
        for _ in 0..300 {
            let status = send_control(
                &cancellation_server,
                json!({"op":"status", "token":cancellation_server.capability_token()}),
            );
            if status["status"] == "busy"
                && status["active_job_id"] == "capstone-cooperative-cancellation"
                && status["cancellable"] == true
            {
                active = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(active, "cancellable job did not become active");
        let cancel = send_control(
            &cancellation_server,
            json!({
                "op":"cancel",
                "job_id":"capstone-cooperative-cancellation",
                "token":cancellation_server.capability_token()
            }),
        );
        assert_eq!(cancel["status"], "cancellation_requested", "{cancel}");
        worker.join().expect("worker thread should complete")
    });
    let cancellation_response: Value = serde_json::from_slice(
        cancellation_output
            .split(|byte| *byte == b'\n')
            .next()
            .unwrap(),
    )
    .expect("cancellation response should be JSON");
    assert_eq!(
        cancellation_response["status"], "failed",
        "{cancellation_response}"
    );
    assert_eq!(
        cancellation_response["result"]["task_result"]["error"]["category"], "cancelled",
        "{cancellation_response}"
    );
    assert!(cancellation_response["result"]["task_result"]["result"].is_null());
    assert_eq!(cancellation_response["diagnostics"]["outcome"], "partial");
    reports.push(cancellation_response);

    let cache_dir = tempfile::tempdir().expect("worker AOT cache root should be available");
    let aot_policy = WorkerAotPolicy::allow_toolchains([
        TaskAotToolchain::Rust,
        TaskAotToolchain::Gcc,
        TaskAotToolchain::Tcc,
        TaskAotToolchain::Zig,
    ])
    .with_artifact_root(cache_dir.path().to_path_buf());
    let aot_task = |y0, rate| {
        json!({
            "format_version":1,
            "task":{"solver":"IVP", "method":"BDF"},
            "equations":{
                "arg":"t", "unknowns":["y"], "rhs":["-rate*y"],
                "parameters":["rate"], "parameter_values":[rate]
            },
            "initial_conditions":{"t0":0.0, "t_end":0.25, "y0":[y0]},
            "solver_options":{
                "symbolic_execution":"aot", "aot_toolchain":"rust", "aot_profile":"debug",
                "matrix_layout":"sparse", "rtol":1e-6, "atol":1e-9, "telemetry":"counters"
            }
        })
    };
    let aot_first = execute_with_aot_policy(
        protocol_request(
            "execute",
            json!({"job":job("capstone-aot-cache-first", aot_task(1.0, 0.75))}),
        ),
        &aot_policy,
    );
    assert_eq!(aot_first["status"], "succeeded", "{aot_first}");
    assert_eq!(
        aot_first["provenance"]["artifact_cache"]["entry_reused"],
        false
    );
    let first_states = aot_first["result"]["task_result"]["result"]["data"]["states"]
        .as_array()
        .expect("AOT result data should be inline");
    let first_final = first_states.last().unwrap()[0].as_f64().unwrap();
    assert!((first_final - (-0.75f64 * 0.25).exp()).abs() < 1e-5);

    let aot_second = execute_with_aot_policy(
        protocol_request(
            "execute",
            json!({"job":job("capstone-aot-cache-reuse", aot_task(2.0, 1.25))}),
        ),
        &aot_policy,
    );
    assert_eq!(aot_second["status"], "succeeded", "{aot_second}");
    assert_eq!(
        aot_second["provenance"]["artifact_cache"]["entry_reused"],
        true
    );
    let second_states = aot_second["result"]["task_result"]["result"]["data"]["states"]
        .as_array()
        .expect("reused AOT result data should be inline");
    let second_final = second_states.last().unwrap()[0].as_f64().unwrap();
    assert!((second_final - 2.0 * (-1.25f64 * 0.25).exp()).abs() < 1e-5);
    reports.push(aot_first);
    reports.push(aot_second);

    for (toolchain, toolchain_name) in [
        (TaskAotToolchain::Rust, "rust"),
        (TaskAotToolchain::Gcc, "gcc"),
        (TaskAotToolchain::Tcc, "c_tcc"),
        (TaskAotToolchain::Zig, "zig"),
    ] {
        let mut aot_combustion_job = combustion_ivp_job("BDF", "AtomView");
        aot_combustion_job["job_id"] = json!(format!(
            "capstone-aot-combustion-bdf-atomview-{toolchain_name}"
        ));
        aot_combustion_job["task"]["solver_options"]["symbolic_execution"] = json!("AOT");
        aot_combustion_job["task"]["solver_options"]["aot_toolchain"] = json!(toolchain_name);
        aot_combustion_job["task"]["solver_options"]["aot_profile"] = json!("debug");
        let aot_combustion_started = Instant::now();
        let aot_combustion = execute_with_aot_policy(
            protocol_request("execute", json!({"job":aot_combustion_job})),
            &aot_policy,
        );
        let aot_combustion_elapsed_ms = aot_combustion_started.elapsed().as_secs_f64() * 1000.0;
        let aot_combustion_result = assert_accepted(&aot_combustion, "ivp");
        let aot_combustion_states = aot_combustion_result["result"]["data"]["states"]
            .as_array()
            .expect("AOT combustion trajectory should be inline");
        assert_eq!(aot_combustion_result["telemetry"]["mode"], "detailed");
        assert!(
            aot_combustion_result["telemetry"]["metrics"]["residual_evaluations"]["value"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
        let aot_combustion_final = aot_combustion_states.last().unwrap().as_array().unwrap();
        assert!(aot_combustion_final[0].as_f64().unwrap() > 0.0);
        assert!(aot_combustion_final[1].as_f64().unwrap() >= -1e-8);
        assert!(aot_combustion_final[2].as_f64().unwrap() >= 300.0);
        rows.push(format!(
            "| ivp-combustion-BDF-AtomView-AOT-{} | BDF/AtomView/{}-AOT | 3 x {} | {:.3} | detailed |",
            match toolchain {
                TaskAotToolchain::Rust => "rust",
                TaskAotToolchain::Gcc => "gcc",
                TaskAotToolchain::Tcc => "tcc",
                TaskAotToolchain::Zig => "zig",
            },
            match toolchain {
                TaskAotToolchain::Rust => "rust",
                TaskAotToolchain::Gcc => "gcc",
                TaskAotToolchain::Tcc => "tcc",
                TaskAotToolchain::Zig => "zig",
            },
            aot_combustion_result["result"]["sample_count"]
                .as_u64()
                .unwrap_or_default(),
            aot_combustion_elapsed_ms,
        ));
        reports.push(aot_combustion);
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let mut markdown = vec![
        "# AI-Agent Interop Four-Family JSON Worker Capstone".to_owned(),
        String::new(),
        format!(
            "- status: passed\n- worker_protocol_version: 1\n- job_contract_version: 1\n- deterministic_inputs: true\n- resumable_named_scenario_checkpoints: responses are keyed by exact job BLAKE3 digest and Debug/Release profile, then revalidated on resume\n- combustion_bvp: six-state BVP_Damp/Banded/Lambdify at 301 and 551 inline nodes plus 1001-node bounded AtomView output, and the 26-state KiThe Banded fixture; endpoint conditions checked independently\n- combustion_ivp: BDF, Radau5, LSODE2 each run with ExprLegacy and AtomView Lambdify; three-state BDF/AtomView AOT compiled via Rust, GCC, TCC, and Zig\n- large_nonlinear: coupled 256-variable root with residual norm assertion\n- large_fitting: 150001 observations with fitted coefficient checks\n- continuation_rebind: Robertson two-segment IVP sweep, 2000-node BVP_sci two-value sweep, and AOT same-artifact rebind probe\n- typed_failure: native callback failure retained partial compact telemetry and did not satisfy acceptance\n- cancellation: authenticated worker request returned a partial snapshot with no result publication\n- AOT_policy_probe: default deny rejected an unapproved request; allowlisted toolchains used the same worker-owned cache identity after changing parameter and initial values\n- AOT_identity_caveat: directory reuse is not proof of the exact dynamic library loaded by the solve\n- recorded_at_unix_ms: {stamp}"
        ),
        String::new(),
        "| family | route | samples/unknowns | wall_ms | telemetry |".to_owned(),
        "|---|---|---:|---:|---|".to_owned(),
    ];
    markdown.extend(rows);
    markdown.push(String::new());
    markdown.push("All successful jobs passed worker preflight/execution, declared acceptance, and independent numeric assertions. The suite covers stiff Robertson continuation; combustion IVP across BDF, Radau5, and LSODE2 with both symbolic frontends and Rust/GCC/TCC/Zig AOT; sparse 2,000-node BVP rebind; six-state combustion BVP at 301, 551, and 1001 nodes across ExprLegacy/AtomView, including bounded-output metadata; the 26-state KiThe combustion BVP on BVP_Damp/Banded; a coupled 256-variable nonlinear root; and 150,001 fitting observations. Detailed IVP continuation diagnostics use the separate worker channel. Checkpoints are not shared across Debug and Release profiles. The AOT rebind probe verifies cache-key normalization and directory reuse only; it does not prove exact loaded-library identity because native loaders do not publish verified loaded-artifact identity.".to_owned());
    std::fs::write(report_dir.join("summary.md"), markdown.join("\n"))
        .expect("capstone summary should be written");
    std::fs::write(
        report_dir.join("responses.json"),
        serde_json::to_vec_pretty(&reports).unwrap(),
    )
    .expect("raw worker responses should be written");
    eprintln!("AI interop capstone report={}", report_dir.display());
}
