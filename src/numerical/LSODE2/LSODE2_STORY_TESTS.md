# LSODE2 Story Tests

This file is the navigation index for the LSODE2 story corpus. Historical
tables remain unchanged in [LSODE2_STORY_ARCHIVE.md](LSODE2_STORY_ARCHIVE.md).
New evidence belongs to one thematic document below and to a dated report in
`test_reports/`.

## Evidence Map

- [Correctness and trajectory](LSODE2_STORY_CORRECTNESS.md): residual/Jacobian
  parity, layouts, non-finite values, invalidation and solver trajectories.
- [Lambdify](LSODE2_STORY_LAMBDIFY.md): symbolic stages, prepared callbacks,
  large callback-only workloads, parameter continuation and evaluator policy.
- [Performance](LSODE2_STORY_PERFORMANCE.md): full solve versus callback-only
  measurements, Sparse/Banded scaling and Sequential/Parallel/Auto baselines.
- [AOT](LSODE2_STORY_AOT.md): generated backend lifecycle, process isolation,
  toolchains, chunking and AOT versus Lambdify comparisons.
- [Lifecycle and telemetry](LSODE2_STORY_LIFECYCLE.md): report provenance,
  cache semantics, counter ownership, continuation and stage attribution.

## Test Source Layout

The executable stories live under `src/numerical/LSODE2/tests/`. Lambdify
stories are primarily in `lambdify_stage_story_tests.rs`,
`lambdify_large_scale_story_tests.rs`, `large_system_story_tests.rs`,
`large_performance_story_tests.rs` and
`parameter_continuation_story_tests.rs`. AOT stories are isolated in the
`aot_*_story_tests.rs` modules. `legacy_story_support.rs` remains only as a
compatibility wrapper for historical runner names; new Lambdify stories must
not be added there.

## Report Policy

`TestReportCapture` now records the Cargo profile and writes reports below
`test_reports/<suite>/<debug|release>/`. A debug run cannot overwrite a
release baseline. Release performance conclusions require release reports;
debug reports are correctness or diagnostic evidence only. The environment
variable `RST_TEST_REPORT_PROFILE` may override the profile label for an
isolated child-process harness.

## Current Release Gates

The next release slice should repeat the Lambdify stage breakdown, large
Sparse/Banded callback corpus, combustion policy, parameter continuation and
multi-worker `Sequential/Parallel/Auto` matrix. The new callback corpus uses
dimensions `1024/2048` and a larger combustion-like fixture, with controller
and linear-solver time excluded from callback rows. Full-solve and callback
break-even must remain separate claims.

Portable Auto break-even remains open. The release worker sweep is a safety
and observability gate: it must preserve numerical parity, counters, chunk
order and typed errors before any policy change is considered.

The 2026-09-29 combined native-plan regression sweep passed the full LSODE2
debug corpus (`441 passed`, `0 failed`, `36 ignored`). The corresponding AOT
diagnostic showed one AtomView build/link lifecycle for direct and solver
preparation at `512/1024/2048`; the post-fix release archive is still pending.
