# LSODE2 Story Index

This index maps each evidence class to one thematic owner. Test paths remain
stable so release commands and archived report names stay reproducible.

## Story Documents

| Document | Evidence class | Primary scope |
|---|---|---|
| `LSODE2_STORY_CORRECTNESS.md` | correctness | layout, invalidation and numerical parity |
| `LSODE2_STORY_LAMBDIFY.md` | Lambdify | callbacks, continuation and evaluator policy |
| `LSODE2_STORY_PERFORMANCE.md` | performance | large systems and callback/full-solve timing |
| `LSODE2_STORY_AOT.md` | AOT | toolchains and generated backends |
| `LSODE2_STORY_LIFECYCLE.md` | lifecycle | cache, rebind and telemetry |
| `LSODE2_STORY_BENCHMARKS.md` | benchmark protocol | Criterion commands and archive policy |
| `LSODE2_STORY_ARCHIVE.md` | archive | dated release evidence |
| `LSODE2_STORY_TESTS.md` | legacy overview | historical cross-links |

## Rust Test Owners

| Owner module | Evidence class | Report suite |
|---|---|---|
| `tests/correctness_story_tests.rs` | debug correctness | `LSODE2_Lambdify` |
| `tests/lambdify_stage_story_tests.rs` | Lambdify stages | `LSODE2_Lambdify` |
| `tests/lambdify_large_scale_story_tests.rs` | large callbacks | `LSODE2_Lambdify` |
| `tests/parameter_continuation_story_tests.rs` | continuation | `LSODE2_Lambdify` / `LSODE2_AOT` |
| `tests/evaluator_policy_story_tests.rs` | Sequential/Parallel/Auto | `LSODE2_Lambdify` |
| `tests/large_system_story_tests.rs` | large correctness/stages | `LSODE2_Lambdify` |
| `tests/large_performance_story_tests.rs` | large wall-clock | `LSODE2_Lambdify` |
| `tests/lifecycle_story_tests.rs` | lifecycle | `LSODE2_Lambdify` |
| `tests/telemetry_stage_story_tests.rs` | typed telemetry schema | `LSODE2_Lambdify` |
| `tests/aot_correctness_story_tests.rs` | AOT correctness | `LSODE2_AOT` |
| `tests/aot_performance_story_tests.rs` | AOT callback/full solve | `LSODE2_AOT` |
| `tests/aot_lifecycle_story_tests.rs` | AOT lifecycle | `LSODE2_AOT` |
| `tests/aot_process_harness_story_tests.rs` | producer/consumer | `LSODE2_AOT` |
| `tests/aot_chunking_story_tests.rs` | chunks and workers | `LSODE2_AOT` |
| `tests/aot_trajectory_parity_story_tests.rs` | trajectory parity | `LSODE2_AOT` |

`legacy_story_support.rs` is a compatibility facade only. Its implementation
is physically split into `legacy_story_core.rs`, `legacy_story_race.rs`,
`legacy_story_solver_quality.rs`, `legacy_story_combustion.rs`,
`legacy_story_view.rs` and `legacy_story_lifecycle.rs`.

## Bench And Release Runner

| Target/script | Role |
|---|---|
| `benches/lsode2_workloads.rs` | bounded Tabled matrix across frontend, execution policy, all three linear backends, continuation and telemetry |
| `benches/lsode2_workload_callbacks.rs` | detailed callback-only Criterion statistics |
| `benches/lsode2_workload_aot.rs` | detailed AOT cold/warm/full-solve Criterion statistics |
| `benches/lsode2_parameter_continuation.rs` | detailed warm/fresh continuation amortization |
| `scripts/lsode2_release_matrix.ps1` | non-fail-fast release sequence with separate report/technical directories |

The compact target is the release index; detailed Criterion targets remain
opt-in statistical instruments and are not merged into its table.

## Canonical Commands

```powershell
cargo test --lib --no-default-features numerical::LSODE2::correctness_story_tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::LSODE2::telemetry_stage_story_tests -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::aot_correctness_story_tests -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::aot_performance_story_tests -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::aot_lifecycle_story_tests -- --nocapture --test-threads=1
cargo bench --no-default-features --bench lsode2_workloads -- --noplot
```

Criterion ownership remains with `benches/lsode2_workload_callbacks.rs`,
`benches/lsode2_workload_aot.rs` and
`benches/lsode2_parameter_continuation.rs`; compact matrix and archive rules
are in `LSODE2_STORY_BENCHMARKS.md`.
