# Agent-Facing Telemetry Contract

Status: task-envelope contract and native inventory, 2026-11-16. Compact JSON
projections cover BVP runtime/CorrectionNewton and IVP snapshots,
nonlinear root/least-squares, and LM/GAVIN/VarPro fitting. Detailed diagnostics are
transported separately for IVP (including continuation segments), successful
Damped/Frozen task runs, BVP_sci/CorrectionNewton retained reports, nonlinear
attempts, and method-specific GAVIN telemetry. The operation worker also
forwards typed failure/cancellation details as bounded partial diagnostics.
The task-envelope route/mode contract is complete for the supported JSON
solver routes. Native telemetry remains solver-specific; the inventory records
differences instead of forcing unrelated native fields into artificial parity.

## Nonstiff IVP Task Projection

The JSON IVP task adapter accepts all four shared modes for RK45, DOPRI, AB4,
and DOP853. Counters and Detailed retain the native symbolic RHS callback
evaluation count. DOP853 additionally reports accepted/rejected adaptive steps
from its native statistics; fixed-step legacy routes mark those fields
unsupported rather than returning fabricated zeros. Timings reports the
task-level solve wall duration. Detailed also sends the selected callback
route, dimensions, dispatch counts, and recorded preparation/evaluation stage
timings through the separate diagnostics object. Telemetry Off remains the
default and leaves the native callback recorder disabled. Native IVP failures
and continuation cancellation preserve the same opt-in Detailed callback
snapshot in a separate `diagnostics` object while the typed failure and source
chain remain in `error`; completed continuation segments are capped by the
existing inline segment limit.

## Agreed Rules

- Telemetry is opt-in. `Off` is the default on every supported telemetry-capable
  solver route and must not perform telemetry-only evaluation, clock reads,
  allocation, or copying.
- Text logging and solver telemetry are independent channels. Enabling logging
  must not implicitly collect public counters; enabling telemetry must not
  implicitly emit log messages.
- The result envelope carries a compact summary when requested. A potentially
  large decision trace, per-step/per-iteration trace, or detailed stage report
  is delivered separately and only when its explicit mode/configuration asks
  for it.
- Native telemetry remains available to native Rust callers. This contract
  defines an adapter projection, not a replacement for solver-specific
  snapshots.

## Value and Absence Semantics

Every projected metric is represented as a state plus an optional value:

| State | Meaning | Value |
| --- | --- | --- |
| `measured` | The selected route collected this metric. | Required, including numeric zero. |
| `not_requested` | Telemetry mode was `Off` or did not request this detail. | `null`. |
| `unsupported` | The selected solver/route has no corresponding measurement. | `null`, with a reason where available. |
| `partial` | A failure/cancellation interrupted collection after some work. | Last valid value, with partial status. |

In the JSON projection the `value` key is retained and set to `null` when no
value is available. Never encode an uncollected metric as zero, `NaN`, or an invented estimate.
When a native legacy snapshot uses zero-filled fields for `Off`, the task
adapter must consult its mode before projecting values.

## Common Compact Metric Set

The envelope projection uses a fixed, versioned set. Applicability is expressed
per metric, not by changing names between solver families.

| Metric | Unit / scope | Notes |
| --- | --- | --- |
| `solve_elapsed` | milliseconds; solver solve call only | Excludes caller work and is not `scenario_wall_ms`. |
| `nonlinear_iterations` | count; nonlinear solver iterations attempted | Does not count IVP time steps. |
| `residual_evaluations` | count; native residual evaluations | Does not mean scalar callback invocations or wrapper dispatches. |
| `jacobian_evaluations` | count; native Jacobian evaluations | Assembly/factorization remain separate stages. |
| `linear_solves` | count; native linear-system solve requests | A factorization may be reused and is not a solve count. |
| `factorizations` | count; native factorization attempts | Failed attempts are included when the native route reports them. |
| `accepted_time_steps` | count; accepted IVP time steps | Not applicable to BVP/root/fitting routes. |
| `rejected_time_steps` | count; rejected IVP time steps | Not applicable to BVP/root/fitting routes. |
| `accepted_nonlinear_updates` | count; accepted Newton/LM state updates | Not a synonym for attempted nonlinear iterations. |
| `rejected_nonlinear_trials` | count; rejected Newton/LM candidate trials | Unsupported when the algorithm has no trial/rejection concept. |
| `mesh_refinements` | count; BVP mesh revisions | Unsupported for fixed-mesh/no-refinement routes. |

Every field carries one of the value states above. Counts are non-negative
integers. Durations are finite, non-negative milliseconds. The compact report
also identifies telemetry mode and solve outcome; neither is inferred from
metric values.

## Separate Diagnostic Channel

Detailed diagnostics may include native stage timings and their inclusive/
child/standalone scope metadata, solver-specific counters, callback dispatch
telemetry, bounded decision traces, AOT lifecycle data, and partial failure
snapshots. These fields remain namespaced by family/solver and retain their
native typed meaning. Large arrays/traces must not be copied into the compact
envelope. RST exposes typed diagnostics in-process; workers/clients own any
serialization, persistence, and external report references per decision 4.

## Initial Native Inventory

`Off` defaults below were source-audited on the listed setup/configuration
paths. This table records mode availability, not field-by-field metric parity.

| Family / route | Default | Native collection modes | Current caveat |
| --- | --- | --- | --- |
| IVP BDF | `Off` | `Off`, `Counters`, `Timings`, `Detailed` | Compact task snapshots expose native solver counts; symbolic callback dispatch remains a distinct layer. |
| IVP Radau | `Off` | `Off`, `Counters`, `Timings`, `Detailed` | Stage scope metadata is available in `RadauTelemetryReport`. |
| IVP Backward Euler | `Off` after this audit | `Off`, `Counters`, `Timings`, `Detailed` | Direct, StiffAPI, and ODE_api2 setup now agree; ODE_api2 exposes explicit `.with_be_telemetry_mode(...)`. |
| IVP LSODE2 | `Off` | Shared IVP telemetry modes on supported generated/native paths | Commonly available solver counters are projected; integration counters and symbolic callback dispatches remain distinct layers, and unavailable fields are marked `unsupported`. |
| IVP nonstiff RK/DOPRI/AB4 | `Off` | Symbolic callback telemetry follows the configured shared IVP mode | Explicit fixed-step routes expose symbolic callback counters when requested; solve-level timing/counters are not a common native report. |
| IVP DOP853 | `Off` | Native adaptive counters/timings follow the selected shared telemetry mode | `Dop853Statistics` marks counter/timing availability explicitly; the normalized ODE facade returns no statistics when telemetry is Off. Internal step/evaluation indices remain only for solver limits and typed callback errors. |
| BVP Damped / Frozen | `Off` after this audit | `Off`, `Counters`, `Timings`, `Detailed` | Previously defaulted to `Counters`; decision logging is separately opt-in. |
| BVP_sci | `Off` | `Off`, `Counters`, `Timings`, `Detailed` | Detailed per-mesh/native backend data is route-specific. |
| BVP CorrectionNewton | `Off` | Native `Off`, `Counters`, `Timings`; task JSON `off`, `counters`, `detailed` | Task `detailed` maps to native `Timings`, which includes counters. No native total solve duration is currently retained, so compact `solve_elapsed` is explicitly unsupported in task `detailed`. |
| Nonlinear root systems | `Off` | Shared `SolverTelemetryMode` | Result and callback metrics differ by formulation/backend. |
| Nonlinear least squares | `Off` | Native `LeastSquaresTelemetryMode` | Mode capabilities must be audited independently from root solvers. |
| Optimization/fitting | `Off` | Native least-squares and/or `SolverTelemetryMode` | UniversalFitting delegates to method-specific telemetry contracts. |

The BDF row distinguishes solver-native evaluations from symbolic callback
telemetry: both can legitimately be present and need not have equal counts.
Likewise, `solve_elapsed` is not interchangeable with performance-gate outer
wall time or preparation time.

## Delivery Status

- The JSON result envelope now has an optional top-level `telemetry` object
  with `schema_version`, requested mode, outcome, and the fixed common metric
  set. On successful results, `null` means telemetry was Off or unavailable on
  that route. On task errors, a requested mode with no native snapshot is
  represented by explicit `unsupported` metrics; telemetry Off remains null.
  Successful IVP single-solve results
  retain a compact native snapshot; timings are clocked only when Timings or
  Detailed was requested. Ordinary native IVP solve failures retain the typed
  source error plus a post-failure snapshot of available counters, projected
  with `outcome: partial`; cancellation error details continue to include the
  solver-owned bounded snapshots. BVP_sci failures now retain their native
  partial snapshot; compact counters/timing use the envelope and Detailed adds
  a separately bounded Newton trace summary. CorrectionNewton failures and
  cancellation now project their native partial report into compact metrics
  and bounded Detailed state/counter diagnostics. The JSONL operation worker now lifts typed
  error details into a separate partial diagnostics object for both
  cancellation and ordinary task failure, capped at 256 KiB. Oversized details
  are explicitly omitted; the size check uses a capped writer, not a second
  full-size serialization buffer. Direct legacy Agent Job responses retain
  their prior shape.
  IVP parameter-loop results sum additive counters and solver elapsed time from
  already-collected segment snapshots; incomplete segment data is marked
  partial. Per-segment snapshots remain in detailed diagnostics, and no solver
  evaluation is repeated during projection. The aggregate preserves
  `not_requested`/`unsupported` when no segment measured a field and reports
  partial when any contributing segment is incomplete.
  Reported metrics distinguish `measured`, `not_requested`, and
  `unsupported` explicitly. Nonlinear root/least-squares maps
  `SolveStatistics`; BVP maps retained runtime counters/timings and the
  CorrectionNewton solve report. Measurements from unsuccessful numerical
  results are marked `partial`. Detailed iteration traces are transported only
  through the separate bounded diagnostics channel when requested.
- Task-layer readiness differs by family: BVP retains `runtime_telemetry` for
  supported routes and a separate CorrectionNewton report, and those available
  fields are now projected; IVP single solves retain and project a common
  snapshot from native solver statistics. Continuation segments retain
  independent compact snapshots. Nonlinear and optimization task specs
  now accept opt-in `solver.telemetry` (`off`, `counters`, `timings`,
  `detailed`) and forward it into their native root/least-squares or fitting
  route. Defaults remain native `Off`; Gavin uses its distinct native
  `SolverTelemetryMode`, while LM/VarPro use `LeastSquaresTelemetryMode`.
  Nonlinear root/least-squares and LM/VarPro fitting statistics are projected.
  GAVIN now has a compact projection for semantically compatible iteration,
  Jacobian, linear-solve and accepted/rejected update metrics; its objective
  calls are deliberately not relabeled as residual evaluations.
  Damped/Frozen task routes project bounded decision traces only in `Detailed`.
  Native Damped/Frozen failures snapshot counters, current packed state, and mesh
  only after solve returns; BVP continuation failures retain completed segments
  and the active post-stop snapshot where available. The telemetry-Off path does
  not copy these snapshots.
- Task-envelope route/mode regression matrices now exercise Damped, Frozen,
  Naive, BVP_sci and CorrectionNewton; BDF, Radau, Backward Euler and LSODE2;
  nonlinear root/least-squares; and LM, GAVIN and VarPro fitting. They validate
  schema compatibility, Off/null, requested counter/timing availability and
  route applicability. Fixed-mesh BVP routes (Frozen, Naive and
  CorrectionNewton) report `mesh_refinements` as `unsupported`, not a misleading
  measured zero. Damped and BVP_sci expose collected refinement counts; their
  zero values mean the selected solve made no mesh changes. GAVIN maps its
  accepted/rejected nonlinear updates and Jacobian work to the common fields,
  but deliberately does not equate objective evaluations with residual calls
  or invent factorization counts. Its native objective metrics remain in the
  separate detailed diagnostics. CorrectionNewton native failure regression
  covers its partial compact metrics and state/counter diagnostics.
- CorrectionNewton task telemetry accepts `detailed`, maps it to native
  `Timings` (which also collects counters), and reports the common mode as
  `detailed`. Since that native report has no total solve duration,
  `solve_elapsed` is `unsupported` with an explicit reason rather than a zero.
- Nonstiff IVP routes RK45/DOPRI/AB4/DOP853 accept opt-in telemetry and expose
  symbolic callback evaluations. DOP853 additionally exposes accepted/rejected
  adaptive-step counts; fixed-step routes mark those metrics unsupported.
  Detailed callback stages remain separate from the compact envelope.
- The task adapter does not fabricate ordinary failure snapshots for nonlinear
  or fitting native errors whose public error type retains none. If telemetry
  was requested, the error envelope retains the requested mode and reports
  common metrics as `unsupported` with reason
  `native_task_error_did_not_retain_partial_snapshot`; the original typed error
  and source chain remain unchanged. Cancellation snapshots are retained for
  nonlinear root/least-squares and LM/VarPro/GAVIN fitting routes.
- Native telemetry snapshots are intentionally not field-by-field equivalent
  solver families. Detailed data are returned inline in a separately bounded
  diagnostics object rather than as a durable artifact; persistence remains
  worker/client-owned. The common JSON adapter contract is tested separately
  from these solver-specific native diagnostic structures.
- Current regression coverage verifies Off/null, IVP BDF counters, IVP solve
  timing opt-in, requested-versus-unrequested fields, continuation snapshots,
  nonlinear attempt diagnostics, GAVIN-specific diagnostics, Damped decision
  traces, and separate worker diagnostics transport. Worker tests also verify
  partial failure/cancellation detail forwarding and bounded omission. BVP_sci
  and Damped/Frozen failure regressions verify compact partial counters,
  bounded diagnostics, and preserved typed source errors. Additional cases
  cover completed BVP continuation segments, GAVIN cancellation, and requested
  telemetry unavailable on a native failure. Damped/Frozen task routes also
  retain post-stop snapshots when the native call returns successfully but
  publishes no result; continuation envelopes preserve prior completed
  segments alongside the active no-result snapshot. These result-contract
  failures keep their typed `BvpSolverResultFailure` source and use the same
  `bvp.solver.result_invalid` classification as the non-telemetry path.

