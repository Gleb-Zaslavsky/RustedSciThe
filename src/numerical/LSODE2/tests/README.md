# LSODE2 Test Layout

This directory contains the thematic LSODE2 correctness and story-test
modules. It is intentionally separate from the solver implementation files.

Current thematic modules:

- `../workload_fixtures.rs`: public, solver-independent mathematical corpus
  shared by stories and `benches/`; includes diffusion, combustion, stiff
  scalar, Robertson, and three-body workloads.
- `../../../../benches/lsode2_workload_aot.rs`: Criterion cold preparation,
  cache-aware warm full-solve, and cold end-to-end AOT measurements over the
  same corpus. Keep its benchmark output separate from dated story reports.
- `story_support.rs`: deterministic fixtures and comparison helpers.
- `large_system_story_tests.rs`: production-shaped callback parity and
  detailed stage-scaling gates; Dense is excluded from large cases.
- `evaluator_policy_story_tests.rs`: Sequential/Parallel/Auto correctness.
- `lambdify_stage_story_tests.rs`: Lambdify preparation and callback stories.
- `telemetry_stage_story_tests.rs`: typed telemetry report coverage.
- `lifecycle_story_tests.rs`: parameter rebind and caller-owned layout checks.
- `correctness_story_tests.rs`: debug trajectory, invalidation, non-finite,
  Sparse-order, compact-Banded-slot, structural-layout and high-cardinality
  parameter gates.
- `three_body_story_tests.rs`: the large three-body physics/AOT story. It stays
  nested under the compatibility story module because it consumes historical
  race-table helpers, but its source is now in this folder.
- `legacy_story_impl.rs`: transitional implementation namespace for the
  remaining historical dashboards, split across thematic include files.
- `legacy_story_support.rs`: thin compatibility facade that re-exports the
  historical runner functions without owning their implementation.

## Safe Infrastructure Contracts

- Story reports are profile-qualified by `test_reporting`: debug output goes
  to `test_reports/<suite>/debug/` and release output goes to
  `test_reports/<suite>/release/`. Set `RST_TEST_REPORT_DIR` only when an
  isolated archive root is required; set `RST_TEST_REPORT_PROFILE` only for a
  deliberately classified process-harness run.
- `LSODE2_STORY_BENCHMARKS.md` is the canonical command and archive protocol
  for Criterion runs. Criterion output stays under `target/criterion`; dated
  story reports remain under `test_reports` and must not be mixed with bench
  output.
- The AOT workload bench uses the default diffusion dimensions `64,128`.
  Larger release sweeps are opt-in through
  `LSODE2_BENCH_AOT_DIFFUSION_DIMENSIONS=128,256,512,1024,2048`, so a normal
  bench invocation remains a safe smoke run.
- The callback workload bench uses default diffusion dimensions `64,256,512`.
  Use `LSODE2_BENCH_CALLBACK_DIFFUSION_DIMENSIONS` for `1024/2048` callback
  runs. Both benches accept `LSODE2_BENCH_SAMPLE_SIZE` and
  `LSODE2_BENCH_MEASUREMENT_TIME_SECS` and print metadata outside measured
  regions.
- `legacy_story_impl.rs` is a transitional include namespace. The included
  `legacy_story_{core,race,solver_quality,combustion,view,lifecycle}.rs`
  files are the remaining physical owners; new stories must not be added to
  that namespace. Further extraction is a test-only cleanup task.

The older historical runner names remain compatibility paths while their
blocks are migrated incrementally. A migration must preserve dated report
keys and be validated in debug before a release baseline is rerun.
