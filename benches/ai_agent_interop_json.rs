//! Isolates task/job JSON orchestration costs from numerical solve costs.

use RustedSciThe::command_interpreter::agent_job::parse_agent_job_json;
use RustedSciThe::command_interpreter::task_contract::{inspect_task_json, run_task_json_envelope};
use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::Value;
use std::hint::black_box;

const IVP: &str = include_str!("../src/agent_interop/examples/ivp_decay.json");
const BVP: &str = include_str!("../src/agent_interop/examples/bvp_constant.json");
const NONLINEAR: &str = include_str!("../src/agent_interop/examples/nonlinear_root.json");
const FITTING: &str = include_str!("../src/agent_interop/examples/fitting_linear.json");
const JOB: &str = include_str!("../src/agent_interop/examples/job_ivp_decay.json");

fn benchmark_agent_interop_json(c: &mut Criterion) {
    let tasks = [
        ("ivp", IVP),
        ("bvp", BVP),
        ("nonlinear_root", NONLINEAR),
        ("fitting", FITTING),
    ];

    for (family, input) in tasks {
        c.bench_function(&format!("agent_json/task_preflight/{family}"), |b| {
            b.iter(|| inspect_task_json(black_box(input)).expect("valid task"))
        });

        let response: Value = run_task_json_envelope(input);
        c.bench_function(&format!("agent_json/result_json_encode/{family}"), |b| {
            b.iter(|| serde_json::to_vec(black_box(&response)).expect("serialize envelope"))
        });

        c.bench_function(&format!("agent_json/full_run/{family}"), |b| {
            b.iter(|| run_task_json_envelope(black_box(input)))
        });
    }

    c.bench_function("agent_json/outer_job_preflight/ivp", |b| {
        b.iter(|| parse_agent_job_json(black_box(JOB)).expect("valid outer job"))
    });
}

criterion_group!(benches, benchmark_agent_interop_json);
criterion_main!(benches);
