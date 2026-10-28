# StiffAPI Story Tests

This document records focused correctness and lifecycle evidence for the
solver facade. It is not a performance baseline; numerical timing claims
belong in the solver-specific benchmark reports and `performance/` baselines.

## Independent IVP Parameter-Loop Release Contract (2026-10-28)

The dedicated Release matrix ran one isolated parameter-loop test for BDF,
Backward Euler, LSODE2, Radau, and the StiffAPI facade. All five commands
passed; each selected one test and filtered out 3928 unrelated library tests.
Cargo command durations were 0.8-0.9 seconds, while individual tests took
0.00-0.01 seconds. These durations are test-runner overhead, not solver
performance measurements.

| route | result | release assertion focus |
| --- | --- | --- |
| BDF | 1 passed | Point order, typed malformed-input failure, recovery, fixed initial state, telemetry activity |
| Backward Euler | 1 passed | Point-local failure, recovery, same initial state, parameter-dependent solution |
| LSODE2 | 1 passed | ExprLegacy/Lambdify Bridge, retained native error, recovery, trajectory and telemetry |
| Radau | 1 passed | Typed point failure, recovery, fixed initial state, Dense output and telemetry |
| StiffAPI | 1 passed | Facade retains each BDF point outcome and telemetry while continuing after failure |

The StiffAPI test in this Release matrix exercises the BDF facade route; it
does not claim a single Release test exercised every facade variant. Direct
solver tests cover the other three native loops, and Python contract tests
cover all four StiffAPI routes separately.

The captured summary and per-command technical logs are retained under
[`IVP parameter-loop Release report`](../../../test_reports/IVP_parameter_loop_release/20261028T093912725Z-12916/summary.md).
