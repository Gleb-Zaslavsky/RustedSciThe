# StiffAPI Contract Audit

Audit scope: the production BDF, Radau, Backward Euler and LSODE2 adapters.
StiffAPI is a dispatch/configuration facade, not a second solver API.
`PreparedStiffSolver::try_solve()` dispatches the first solve without requiring
an application-side solver match. Typed `as_*`/`as_*_mut` accessors preserve
native lifecycle methods, configuration fields, results and error details.
`StiffSolveOutcome` and `StiffSolveError` retain solver-specific variants
rather than flattening them into a lossy common schema.

## Defaults, Lifecycle And Failures

| Solver | Defaults and configuration ownership | Lifecycle/result contract | Failure and recovery contract |
| --- | --- | --- | --- |
| BDF | `BdfSolverOptions` is passed through; an empty preset changes no field. Dense is the native supported linear layout. | Native `try_generate`, `try_solve`, parameter update/continuation and restart methods remain available through `as_bdf[_mut]`; borrowed result and status remain native. | `BdfSolveError` and native status are not converted. Running/unsolved result is rejected by `trajectory_view`; partial/result semantics follow BDF. |
| Radau | `RadauProblem`, `RadauConfig`, initial state and ordered run parameters are retained; no facade logging control exists in `RadauConfig`. | Unified solve returns owned `RadauSolution`; fresh parameterized solves take `y0` explicitly, and restart changes the interval. The solver does not retain a common trajectory or implicit continuation state. | `RadauError` and its stable kind remain intact inside `StiffSolveError`; sampling errors remain `RadauError` inside `StiffPostprocessError`. |
| Backward Euler | `BeSolverOptions` is passed through; fixed/heuristic step, nonlinear limits and native defaults are not flattened into adaptive-IVP fields. | Native fallible solve, continuation and restart remain available through `as_backward_euler[_mut]`; the native initial/partial trajectory is retained. | `BeError` is preserved. Partial trajectory is not represented as a successful empty result; consumers can inspect native status and trajectory. |
| LSODE2 | Complete `Lsode2ProblemConfig` is retained, including controller, linear structure, forced backend, stop conditions and `Native/Bridge` execution. | Native prepare/solve, parameter update, reconfigure and restart remain available through `as_lsode2[_mut]`. | `Lsode2Error` variants and source chain are preserved. Native and Bridge are distinct execution routes; the facade does not reinterpret their counters or statuses. |

The facade preparation functions apply only the requested `StiffPreset` axes.
With no preset, they preserve the supplied native configuration. Unsupported
axes fail with `StiffApiError::UnsupportedCapability`; structured layouts are
never silently densified. Presets describe conditional applicability rather
than promising a Cartesian product of axes.

| Solver | Layout/route constraints enforced by the adapter | Solver-specific controls kept native |
| --- | --- | --- |
| BDF | Dense linear layout only; requests for Sparse/Banded are rejected. | BDF order/step controls, callback factories, parameter binding, stop conditions and restart policy remain in `BdfSolverOptions`/`BdfSolver`. |
| Radau | Dense/Sparse/BandedAuto and explicit symbolic bandwidth are distinct; logging is rejected because the native config has no logging field. | Radau tolerances, step/retry limits, output policy, fresh parameterized solves and dense-output sampling remain in `RadauConfig`/`RadauSolution`. |
| Backward Euler | Dense only; Sparse/Banded and forced-Lambdify requests are rejected rather than relabeled. | Fixed/heuristic step policy, Newton tolerance/limits, parameters, stop conditions and continuation remain in `BeSolverOptions`/`BE`. |
| LSODE2 | Dense/Sparse/BandedAuto/explicit widths are preserved; automatic bandwidth requires symbolic structure. `PreparedAtom` and unsupported AOT/callback combinations are rejected. Native and Bridge execution are separate, not flattened. | Adams/BDF/Auto controller, forced linear backend, stop conditions, callback source, native execution limits and reconfiguration remain in `Lsode2ProblemConfig`/`Lsode2Solver`. |

These constraints are guarded by `bdf_adapter_forwards_native_axes_and_rejects_unsupported_layouts`,
`backend_rejects_structured_layout_instead_of_densifying`, the preset catalog
applicability tests, `lsode2_banded_auto_is_explicit_and_symbolic_only_at_stiff_api_boundary`,
and the solver-specific forwarding tests in `tests.rs`. A new native option must
be added to that route's contract tests when its adapter behavior changes.

## Telemetry And Logging

| Solver | Snapshot payload | Scope/availability rule |
| --- | --- | --- |
| BDF | Full `IvpBackendStatistics`; optional `IvpTelemetrySnapshot` for symbolic preparation | Backend fields retain cumulative totals and their native nested timing semantics. Missing symbolic telemetry is `None`; numeric zero is not converted to missing. `try_solve` returns a typed success marker; results remain on the solver. |
| Backward Euler | Full `IvpBackendStatistics`, `BeDetailedStatistics`, optional symbolic snapshot | The detailed payload includes BE/NRE-specific parameter binding, failure, FD, factorization, solve, accepted/failed-step and output-assembly data. Availability follows the configured telemetry mode and native lifecycle. `try_solve` returns a typed success marker; trajectory remains on the solver. |
| LSODE2 | Full `IvpBackendStatistics`, `Lsode2NativeStatistics`, full symbolic `IvpTelemetrySnapshot` | Native and Bridge scopes intentionally differ. Native fields are cumulative native statistics; bridge callback fields are separate. Zero-valued fields remain zero and are not evidence that a stage was measured. Unified dispatch returns `Lsode2SolveSummary`. |
| Radau | Full `RadauTelemetryReport` from `RadauSolution` | Report includes native counters, timing map, scope metadata, AOT artifact keys and adaptive trace. It is unavailable before a successful result is returned; `PreparedStiffSolver::telemetry_snapshot` reports that boundary explicitly. |

Snapshots retain native struct/map types rather than copying fields into a
second StiffAPI schema. Consequently newly added native fields remain visible
through the typed variant. Counter parity is exact for matched direct/facade
runs; timings are observational and must not be compared for exact equality
across separate runs. Nested/inclusive scopes must not be summed as disjoint.

Logging is forwarded only for solvers exposing a native logging option: BDF,
BE and LSODE2. Radau rejects the request because `RadauConfig` has no logging
control. StiffAPI does not install a logger; the existing native logger owns
the timestamped process file and its setup/I/O behavior.

## Results And Postprocessing

| Solver | Common facade support | Native-only support retained |
| --- | --- | --- |
| BDF | Borrowed row-oriented trajectory, explicit owned dataset conversion, `PostprocessPlan` execution | Native CSV export and plotting through `as_bdf`; the native CSV error remains boxed. |
| Backward Euler | Borrowed initial/partial trajectory and common dataset/plan conversion | Native result/status and legacy plotting through `as_backward_euler`. |
| Radau | Explicit-time sampling from a returned `RadauSolution` into a dataset; Dense output must have been requested | Native solution and sampling methods remain available on the returned value; there is no retained solver trajectory to borrow. |
| LSODE2 | Borrowed retained trajectory, dataset conversion and common `PostprocessPlan` execution | Native export/result/status methods remain available through `as_lsode2`. |

File output, plots and reports are explicit user actions, not side effects of
solve. A postprocessing error is distinct from a numerical solve error and does
not consume or erase the source result. StiffAPI deliberately does not invent
one cross-solver plotting/export API where native formats, optional tools and
error types differ; use the native accessor for such operations.

## Contract Tests

`tests.rs` verifies preset field forwarding/rejection, direct-vs-unified-solve
solution and discrete-statistics parity, native error/source preservation,
pre-solve telemetry availability, trajectory ownership, Radau sampling and
postprocessing failure recovery. The LSODE2 Bridge test additionally compares
Bridge results and discrete counters against direct native execution through
the common dispatch, then restart both native instances through the typed
accessor. Timing fields are intentionally excluded from
cross-run equality assertions.

The exhaustive Cartesian product of every solver-specific option is not a
valid contract: each solver has distinct controls and some combinations are
intentionally unsupported. Add route-specific tests when a new native setting
or route is introduced; never infer support from independent preset axes.
