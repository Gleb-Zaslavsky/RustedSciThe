# AI-Agent Interop Performance Notes

This document records measurements for the Rust JSON task/job surface. It is
not a solver comparison or regression baseline: the canonical tasks are small
and most complete runs are far below the crate's usual workload-sized
performance-gate threshold. These numbers track adapter-scale costs while
separating JSON parsing, result serialization, and numerical execution.

## Correctness Gate, 2026-11-15

The dedicated release matrix ran
`cargo test --release --lib command_interpreter:: -- --nocapture --test-threads=1`:
324 passed, 0 failed, 0 ignored, 1231.5 seconds. The corresponding Debug suite
also passed 324 tests. Full technical output and the generated summary are in
`test_reports/AI_agent_interop_release_manual/20261115T004259Z/`.

## Criterion JSON Matrix, 2026-11-15

Command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
scripts/ai_agent_interop_perf_matrix.ps1`. The script ran all 13 benchmarks
using the Criterion bench profile and generated the Tabled Markdown report at
`test_reports/AI_agent_interop_perf_manual/20261115T011858Z/reports/performance_summary.md`.
The full Criterion output is in the sibling `technical/criterion.log`.

Values below are the recorded mean estimates and 95% confidence intervals
from this run. Units are microseconds per operation.

| Family | Operation | Mean | Mean 95% CI |
|---|---|---:|---:|
| IVP | Task preflight | 7.480 | [7.333, 7.650] |
| BVP | Task preflight | 12.762 | [12.525, 12.980] |
| Nonlinear root | Task preflight | 8.690 | [8.501, 8.864] |
| Fitting | Task preflight | 11.182 | [11.008, 11.335] |
| IVP | Result JSON serialization | 4.968 | [4.879, 5.051] |
| BVP | Result JSON serialization | 0.761 | [0.735, 0.788] |
| Nonlinear root | Result JSON serialization | 0.331 | [0.317, 0.346] |
| Fitting | Result JSON serialization | 0.392 | [0.384, 0.398] |
| IVP | JSON parse + solve + envelope | 196.624 | [192.527, 200.587] |
| BVP | JSON parse + solve + envelope | 20.994 | [20.703, 21.249] |
| Nonlinear root | JSON parse + solve + envelope | 14.196 | [14.018, 14.345] |
| Fitting | JSON parse + solve + envelope | 32.658 | [32.377, 32.952] |
| Outer job | Job preflight (IVP) | 13.369 | [13.190, 13.521] |

The full-run rows include solver execution and cannot be interpreted as JSON
overhead. The fixtures are small smoke workloads; there is no A/B variant and
no claim of performance improvement. Add representative large-task scenarios
only after their acceptance, telemetry, and artifact contracts are agreed.

## Repeatable Commands

- Correctness: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
  scripts/ai_agent_interop_release_matrix.ps1`
- JSON measurements: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
  scripts/ai_agent_interop_perf_matrix.ps1`
