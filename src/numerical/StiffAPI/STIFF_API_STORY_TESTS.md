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

## Fresh Solve With A Replacement Initial State (2026-11-07)

`PreparedStiffSolver::try_solve_with_initial_state(y0)` starts a fresh solve on
the configured interval and preserves the current parameter binding and
prepared callbacks/artifacts. BDF, Backward Euler and LSODE2 reset their native
integration state; Radau supplies the new initial state directly to its
stateless solve route. The explicit interval-changing operation remains
`try_solve_with_initial_state_in_interval(t0, y0, t_bound)`. Neither method is
parameter continuation or continuation in time.

Debug gate `same_interval_initial_state_rebind_is_consistent_across_all_routes`
passed across all four families on the parameterized scalar decay equation.
The test checks that the new initial value is the first trajectory row, the
configured decay parameter is still in effect, and the final states match the
same analytic solution tolerance, including Backward Euler's discretization
error. The complete StiffAPI Debug module passed: 46 passed, 0 failed.
Python contract tests passed for the direct BDF/BE/LSODE2/Radau wrappers, the
StiffAPI facade, and the BDF numerical-callback wrapper.

## BDF callback-policy telemetry parity (2026-11-02)

An integration report showed successful standalone BDF solves through
StiffAPI while the nested symbolic snapshot reported zero residual/Jacobian
evaluations and dispatches. Inspection confirmed that BDF applied the
requested callback policy to its Lambdify evaluators and those evaluators
recorded work in the shared `IvpTelemetry`; the facade was returning a
value-snapshot captured at backend preparation, before integration began.

BDF now retains the cheap shared telemetry handle alongside the immutable
preparation snapshot. `StiffTelemetrySnapshot::Bdf` reads a fresh symbolic
snapshot, so Detailed reports include callback activity after the solve. The
preparation-only accessor remains unchanged for callers that need the exact
post-preparation view. Native BDF counters and symbolic callback counters
remain separate layers: native counters describe solver requests/work, while
the symbolic snapshot describes evaluator calls, scalar evaluations and
Sequential/Rayon dispatch. Do not assume equality field-by-field.

Debug gate `bdf_lambdify_policy_and_live_callback_telemetry_cover_completed_solve`
passed for Sequential, Auto `{ min_work: 0 }`, and forced Parallel
`{ min_work: 0 }` on the same four-state symbolic IVP. Every route reported
nonzero native residual/Jacobian calls and symbolic residual/Jacobian
evaluations. Sequential reported sequential dispatch only; forced Parallel
reported parallel dispatch; Auto reported its effective dispatch. Final states
matched within `1e-10` across policies. This confirms callback scheduling and
telemetry behavior in debug; it is not a performance claim.
