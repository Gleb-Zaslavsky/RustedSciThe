# Performance Metric Contracts

Audit date: 2026-10-20.

This document maps metrics currently emitted by `performance/runner/scenarios`
to their Rust sources and lifecycle. It is runner documentation, not a public
solver API. All `*_ms` values are milliseconds. Counters are integer counts
unless stated otherwise. A missing optional metric means “not observed/not
applicable”; it is not a measured zero.

## Interpretation Rules

- `scenario_wall_ms` is an Off-mode timer around the scenario's declared outer
  operation. Its exact interval is family-specific and is listed below. It is
  comparable only within the same versioned scenario contract.
- `solver_setup_ms`, `prepare_ms`, `solve_ms`, and similar Off-mode metrics are
  runner `Instant` intervals. They are not native telemetry fields. They can be
  nested in `scenario_wall_ms` and must not be added to it.
- `instrumented_diagnostic` metrics are gathered by a separate execution with
  telemetry enabled. They do not decompose the Off-mode wall sample.
- When native scope metadata is available, the persisted `scope` value is
  `native_{kind}_{stage}` and `parent_scope` is the native parent key. Metrics
  without such metadata keep the generic runner scope labels.
- Native cumulative fields cover the lifetime represented by the snapshot.
  Unless the source API documents a reset or “last call” contract, this ledger
  does not infer one from a field name.
- BVP_Damp exposes aggregate typed scopes rather than a per-field parent map;
  see `Known Metadata Gaps` before interpreting its stage metrics additively.

## BDF

| Runner metric | Native source / unit | Scope and lifecycle |
| --- | --- | --- |
| `stage_backend_prepare_ms` | Solver statistics field `backend_prepare_ms_total` | Cumulative backend preparation for the diagnostic solver instance. |
| `stage_integration_loop_ms` | Solver statistics field `integration_loop_ms_total` | Integration-loop total; contains step work and therefore overlaps the linear, residual and Jacobian timings. |
| `stage_linear_factorization_ms` | Solver statistics field `linear_factorization_ms_total` | Cumulative factorization work inside integration. |
| `stage_linear_solve_ms` | Solver statistics field `linear_solve_ms_total` | Cumulative linear solves inside integration. |
| `stage_residual_ms`, `stage_jacobian_ms` | Solver statistics fields `{residual,jacobian}_ms_total` | Cumulative callback work; these are not disjoint from the integration-loop total. |
| `stage_prepare_total_ms` | `BdfPreparationTimings::total_ms` | One backend-preparation report. Its component fields (`configuration`, symbolic backend, callback wiring, runtime initialization, linear backend setup, runtime publication) are documented by `table_report()` as disjoint diagnostic scopes. This total is separate from solve-loop time. |

Counts `accepted_steps`, `candidate_steps`, `rejected_steps`, `nfev`, `njev`,
and `nlu` map respectively to solver-statistics fields `{accepted_steps_total,
candidate_step_attempts_total,rejected_step_attempts_total,bdf_nfev_total,
bdf_njev_total,bdf_nlu_total}`. They aggregate the diagnostic integration;
`state_dimension` is the final-state vector length. The runner emits the BDF
preparation *total*, not each disjoint preparation component.

## Radau

The adapter emits `stage_{key}` from positive finite entries of
`RadauTelemetryReport::timings_ms`. Native keys include preparation, callback,
Newton, linear algebra, output and step-control scopes, callback sub-stages,
linear factorization/solve sub-stages, and AOT preparation stages. The exact
available key set depends on route. Native zero entries are omitted by the
adapter, so a missing key is not evidence that the stage is unsupported.

Native `timing_scopes` declares parent/child relationships: preparation,
callback, Newton and linear totals are inclusive; for example,
`factorization_ms`, `real_solve_ms` and `complex_solve_ms` are children of
`linear_ms`; residual/Jacobian callback stages are children of `callback_ms`;
symbolic/AOT preparation stages are children of `preparation_ms`. Output,
step-control, workspace and parameter-rebind stages are standalone as declared
by `RadauTelemetryReport::timing_scope_metadata()`.

The runner retains `timing_scopes` per emitted metric in the baseline
`scope`/`parent_scope` fields. Do not sum all Radau `stage_*` entries. Counts
emitted by the scenario (`accepted_steps`,
`rejected_steps`, `attempts`) come from the solution object; they are from the
Off-mode trajectory, after the detailed trajectory is checked for parity.

## LSODE2

| Runner metric | Native source / unit | Scope and lifecycle |
| --- | --- | --- |
| `stage_residual_ms` | `evaluation_telemetry.residual_ms_total` | Accumulated callback evaluation time on the diagnostic solve. |
| `stage_jacobian_ms` | `evaluation_telemetry.jacobian_ms_total` | Accumulated Jacobian callback time on the diagnostic solve. |
| `stage_linear_ms` | `evaluation_telemetry.linear_ms_total` | Accumulated linear work on the diagnostic solve; it is not asserted to be disjoint from other native stage totals. |
| `stage_cold_{Stage:?}_ms` | `IvpTelemetrySnapshot::cold_stage(stage).elapsed` | Emitted only when the stage's `recorded` flag is true. Covers the current prepared lifecycle snapshot, not a per-step value. IDs use Rust `Debug` names and are therefore coupled to enum spelling. |

The cold-stage selection is SolverPreparation, SymbolicJacobian,
AtomJacobianPreparation, AotLowering, AotBuild and AotLink. Counts come from
`Lsode2EvaluationTelemetry` and the IVP snapshot: state/trajectory dimensions,
accepted/rejected steps, residual/Jacobian evaluations, linear solves and
Lambdify/AOT parallel dispatches. They describe this diagnostic solver
execution; do not interpret the AOT preparation counters as per-step counts.

## BVP_Damp

| Runner metric | Native source / unit | Scope and lifecycle |
| --- | --- | --- |
| `stage_residual_ms`, `stage_jacobian_ms`, `stage_linear_ms`, `stage_factorization_ms` | `BvpTelemetrySnapshot::timings.{residual,jacobian,linear_system,factorization}` | Detailed solver telemetry for the entire scenario lifecycle, including initial solve and continuation. These scopes overlap with outer Off wall timers, not with the separate Off execution. |
| `stage_generation_ms` | `BvpGenerationTelemetrySnapshot::total` | Optional frontend/backend generation preparation; present only when generation telemetry exists. |
| `stage_symbolic_jacobian_ms`, `stage_backend_selection_ms`, `stage_runtime_binding_ms` | Corresponding `BvpGenerationTelemetrySnapshot` fields | Optional generation sub-stages. Do not add them to `stage_generation_ms` without verifying the native generation scope contract. |
| `stage_atom_discretization_ms` | `BvpAtomDiscretizationTelemetrySnapshot::total` | Optional Atom preparation lifecycle. |
| `stage_lambdify_callbacks_ms` | Sum of `residual_elapsed + jacobian_elapsed` from the active legacy/Atom Lambdify snapshot | Runner-combined callback time; it intentionally has no separate residual/Jacobian breakdown and is not necessarily exclusive of broader solver timing. |

The runner also exports the typed aggregate durations as
`native_scope_solve_ms`, `native_scope_mesh_revision_ms`,
`native_scope_iteration_ms` and `native_scope_damping_trial_ms`. These values
are tagged as typed aggregates without inferred parent relations. The Off
timers separately report construction (`solver_setup_ms`), initial
preparation (`prepare_ms`), first solve, continuation, continuation
preparation, and outer scenario wall. Continuation stage/counter telemetry is
aggregate for the full Detailed scenario run. Counters map to
`BvpTelemetrySnapshot::{counters, ...}`: mesh/state dimensions, iterations,
residual/Jacobian requests, factorizations/cache hits, and grid refinements.
`mesh_nodes` is the solution row count; `state_dimension` is its column count.

## BVP_sci

The primary numeric stages are optional `BvpSciTelemetrySnapshot` fields:
`preparation_ms`, `rhs_callback_ms`, `boundary_callback_ms`,
`residual_evaluation_ms`, `jacobian_evaluation_ms`, `linear_assembly_ms`,
`factorization_ms`, `linear_solve_ms`, and `mesh_refinement_ms`. The adapter
preserves `None` by omitting the metric. Timing scope kind, parent, invocation
count and elapsed value are natively available in `timing_scopes`; inclusive
stages must not be added to their children, and callback-exclusive stages are
identified by native metadata.

The runner also maps recorded AOT cold/warm IVP stages to
`aot_cold_{label}_ms` and `aot_warm_{label}_ms`. These remain distinct lifecycle
scopes: cold materialization/lowering/build/link/publication and warm chunk
dispatch/worker/copy/output work are not interchangeable. AOT cache lifecycle
is route-specific.

Counters map to the snapshot's final-node, Newton, residual/Jacobian,
factorization/solve, mesh-refinement, dispatch, sparse symbolic analysis/reuse
and Banded structured-factorization fields. SuperLU-specific counters are not
currently emitted as counters by this adapter; fallback status is available
through correctness/route diagnostics elsewhere and must be checked before
using a requested-SuperLU result as SuperLU evidence.

The adapter retains the native `timing_scopes` kind and parent per mapped
metric in the baseline `scope`/`parent_scope` fields. RHS and boundary callback
metrics both refer to the same native callback stage; their IDs retain the
distinction between fields.

## Backward Euler

`preparation_ms`, `symbolic_jacobian_ms`, `residual_lambdification_ms`,
`jacobian_lambdification_ms`, `aot_build_ms` and `aot_link_ms` map to recorded
`IvpColdStage` entries. `residual_ms`, `jacobian_ms`, `factorization_ms`,
`linear_solve_ms` and `native_solve_stage_ms` map to recorded
`IvpWarmStage::{ResidualEvaluation,JacobianEvaluation,Factorization,RhsSolve,
Solve}` entries. Unrecorded stages are absent, not zero.

`be_factorization_ms_total` and `be_linear_solve_ms_total` map to
`BeDetailedStatistics::{factorization_ms_total,linear_solve_ms_total}` and
aggregate across the fixed-step workload. They overlap the corresponding
shared IVP warm stage totals; do not sum the two representations. Counts map to
the BE details and IVP snapshot: trajectory/state sizes, residual/Jacobian
evaluations, factorization/solve calls, accepted and failed steps. The route is
asserted against the requested Lambdify/AOT execution before output.

## Nonlinear Systems

Gate metrics `preparation_ms`, `parameter_binding_ms`, `solve_ms` and
`scenario_wall_ms` are adapter `Instant` intervals around the respective
operations. Detailed preparation maps `preparation_total_ms` to
`PreparationReport::preparation_duration`; `prepare_stage_{key}_ms` maps to
the matching optional `PreparationStageTiming::wall_time`; `aot_build_ms`
maps to `PreparationReport::build_duration` when present. The per-stage
preparation timings are diagnostic sub-stages of preparation and should not be
added to the preparation total.

Detailed solve metrics map to `SolveStatistics::{residual_duration,
jacobian_duration,linear_solve_duration,linear_factorization_duration,
linear_system_solve_duration,total_duration}`. These can overlap: notably
`linear_solve_ms` and `linear_system_solve_ms` are distinct native fields, and
the runner preserves both rather than assuming equality. `solver_native_total_ms`
is a native total, not the adapter's `solve_ms`. Counters map to the result's
iterations, residual/Jacobian evaluation, factorization and linear-solve
counts. The plain numerical nonlinear scenario has only adapter wall timings
and result counters; it does not enable native preparation/solve telemetry.

## Optimisation / Fitting

For Gavin standalone and UniversalGavin, counters map to the native telemetry
snapshot (`iterations`, objective and analytic-Jacobian evaluations, FD
Jacobians, accepted/rejected steps, linear solves). Detailed durations map to
`total_time`, `objective_time`, `analytic_jacobian_time`,
`linearization_time`, and `linear_solve_time`; standalone fitting additionally
reports symbolic `preparation_duration`. The total and component durations are
not declared disjoint by the runner and must not be summed.

For VarPro and UniversalVarPro, `snapshot_from_statistics` maps
`residual_duration`, `jacobian_duration`, `linear_factorization_duration`,
`linear_system_solve_duration`, and `total_duration` from `SolveStatistics`;
counts map to iterations, residual/Jacobian evaluations, factorizations and
solves. The same no-additivity rule applies. `scenario_wall_ms` is separately
measured around the Off route. Detailed diagnostics are from a second solve.

## Standalone Banded LU Microcase

`factorization_ms` and `solve_ms` are adapter `Instant` intervals directly
around `LapackStyleBandedLuFaithful::factor_from` and `solve_in_place`.
`scenario_wall_ms` begins before factorization and ends after solve, so it
contains both intervals plus timer/dispatch overhead. Matrix assembly and
known-solution residual checks are outside `scenario_wall_ms` as currently
implemented. This is a fixed tridiagonal microcase, not a nonlinear-solver
full-lifecycle timing.

## Known Metadata Gaps

1. BVP_Damp exports its typed aggregate scopes, but its native snapshot does
   not map residual/Jacobian/linear/factorization values to those aggregates
   with explicit parent relations. Do not infer per-stage hierarchy for them.
2. Some diagnostic totals have no proved reset or cross-call accumulation
   contract in the adapter. This ledger calls them “snapshot lifecycle totals”
   rather than inferring that they are per-solve or process-cumulative.
3. Native metrics whose values default to `0.0` without an availability flag
   cannot distinguish “measured zero” from “stage not reached”; Radau currently
   filters non-positive stage values. This behavior is intentionally documented
   and must not be silently changed to zero-filled samples.
4. This audit verifies what each adapter reads, not the correctness of every
   native timer implementation. Native timing implementation and reset tests
   remain owned by the corresponding solver.

These are runner-contract limitations, not requests to add instrumentation to
production hot paths. Continue with lifecycle tests only where the native
contract can be established; do not add production APIs solely for gates.
